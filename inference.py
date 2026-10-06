# -*- coding: utf-8 -*-
"""
Inference core for the screening app: model loading, geographic routing, factor attribution and predict().

ROUTING. Two models, chosen by where the point falls, never by what the user types:

    inside the local region  ->  Comilla-trained model
    anywhere else in Bangladesh -> national model
    outside Bangladesh       ->  refused

The local region is the eight upazilas that contain study wells (assets/local_region.geojson, built by
App/build_boundaries.py). Both models take the same three user inputs, latitude, longitude and depth; every
other feature is looked up from the shipped assets. So the user never chooses a model and never sees a form
that changes shape.

WHAT REPLACED THE TIERS. The previous version routed four models by which optional inputs the user filled
in: borehole log, water chemistry, both, neither. Chemistry was dropped in July because it needs a water sample from a well that already exists, and
the borehole model is dropped here: it needs sediment logged at screen depth, which does not exist before
the hole is made, so it cannot serve a pre-drilling tool. Those models stay on disk under obsolete/.

EVERY POLICY CHOICE IS A CONSTANT BELOW. Which local model runs, whether the Bangladesh standard is shown,
whether the label quotes the test or the cross-validation figure, and where the safe/unsafe cut sits are all
settings. None of them requires retraining, and all ten models stay on disk whatever they are set to.
"""
import os, json, warnings, threading

import numpy as np
import pandas as pd
import joblib

from pipeline_utils import PearsonCorrelationFilter   # noqa: needed so joblib can reload the pipeline
import features

# The preprocessed matrix is plain numpy by construction, exactly as in training, so this warning is noise.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- policy

# Which Comilla model serves. All four are built and parity-checked; see reports/APP_MODEL_DECISIONS.md.
# Accuracies below are best-on-test, the rule the paper reports, at 10 and 50 ug/L. They were restated
# on 2026-08-28 when export_app_models.py switched from the CV-selected model to the best of the nine;
# local, local_sat and local_drill were already the same model under both rules and did not move.
#   local        terrain, 93.8 / 93.8, needs nothing the user cannot supply
#   local_drill  screen sediment, 93.8 / 90.6, NOT pre-drilling
#   local_sat    satellite and soil, 90.6 / 93.8
#   local_geo    surface geology, 90.6 / 93.8, identical inputs to the national model
LOCAL_MODEL = "local"

# 10 ug/L is the WHO guideline, 50 ug/L the Bangladesh national standard. The flag gates the models,
# /api/meta and the interface together rather than the interface alone, because hiding a threshold in the UI
# would leave its numbers readable from the API. The 50 ug/L models, cards and LIME backgrounds stay on disk,
# so flipping this back restores them with no rebuild.
#
# WHO ONLY (user, 2026-08-14). Briefly served both from 2026-08-13. The two thresholds are answered by two
# separately selected models with uncalibrated scores and nothing tying them together, so the app could print
# a higher chance of exceeding 50 than of exceeding 10: 27 of 160 Comilla wells and 276 of 3,534 national
# wells. Anything above 50 is above 10, so that pair cannot be true of a real well. The paper still reports
# both thresholds, where they appear in separate tables rather than side by side on one well.
SERVE_THRESHOLD_50 = False
THRESHOLDS = [10, 50] if SERVE_THRESHOLD_50 else [10]

# Which accuracy the interface quotes. "test" is the held-out figure and the one the paper reports.
LABEL_ACCURACY = "test"

# Probability of safe at or above which a point is called safe. 0.5 maximises total accuracy and treats both
# mistakes as equally costly; on the 32 Comilla test wells that puts every error on the dangerous side, an
# unsafe well called safe, with no false alarms at all. Raising it trades headline accuracy for caution.
DECISION_CUTOFF = 0.5

STANDARD_NAME = {10: "WHO guideline", 50: "Bangladesh standard"}

# Column names as a member of the public should read them. The models keep the study's own names, so this
# is presentation only and never reaches a fitted pipeline.
PRETTY = {
    "Latitude": "Latitude", "Longitude": "Longitude", "Actual_Depth(m)": "Drilling depth",
    "elevation": "Ground elevation", "slope": "Slope", "twi": "Wetness index",
    "hand": "Height above nearest stream", "surf_geo": "Surface geology",
    "Well_Screen_Lithology": "Sediment at screen depth",
    "soil_organic_carbon": "Soil organic carbon",
    "soil_clay": "Soil clay",
    "soil_sand": "Soil sand",
    "soil_silt": "Soil silt",
}


def pretty(col):
    return PRETTY.get(col, col)

# ---------------------------------------------------------------- models

SCOPES = {"local": LOCAL_MODEL, "national": "national"}

MODELS = {}          # (scope, threshold) -> (pipeline, card)
for scope, key in SCOPES.items():
    for th in THRESHOLDS:
        mp = os.path.join(HERE, f"model_{key}_as{th}.joblib")
        cp = os.path.join(HERE, f"model_card_{key}_as{th}.json")
        if os.path.exists(mp) and os.path.exists(cp):
            MODELS[(scope, th)] = (joblib.load(mp), json.load(open(cp, encoding="utf-8")))

BANDS = [(0.66, "HIGH RISK"), (0.34, "ELEVATED RISK"), (-1, "LOWER RISK")]


def band(p_unsafe):
    return next(name for cut, name in BANDS if p_unsafe > cut or cut < 0)


# ---------------------------------------------------------------- factors

_explainers = {}
_lime_lock = threading.Lock()
LIME_SAMPLES = int(os.environ.get("LIME_SAMPLES", "5000"))


def _explainer(scope, th):
    """LIME over the study's own preprocessed training background, so the app's factors and the paper's
    LIME figures are explaining the same thing.

    Built under the lock. The endpoints are plain `def`, so uvicorn runs them in a threadpool and two
    requests can reach the empty-cache branch together. Nothing they returned would have differed, because
    every explanation reseeds in place first and a cached explainer reproduces a freshly built one exactly,
    but the loser of that race threw away a finished explainer on a 512 MB instance.
    """
    with _lime_lock:
        if (scope, th) not in _explainers:
            from lime.lime_tabular import LimeTabularExplainer
            bg = np.load(os.path.join(HERE, f"lime_bg_{SCOPES[scope]}_as{th}.npz"), allow_pickle=True)
            _explainers[(scope, th)] = LimeTabularExplainer(
                bg["X"], feature_names=list(bg["features"]), class_names=["unsafe", "safe"],
                mode="classification", discretize_continuous=True, random_state=3)
        return _explainers[(scope, th)]


def local_factors(model, card, row_df, scope, th, k=4):
    """Top contributing features, seeded and reproducible. Falls back to occlusion against the training
    medians if LIME or its background is unavailable."""
    try:
        ex = _explainer(scope, th)
        prep, clf = model.named_steps["prep"], model.named_steps["clf"]
        xt = prep.transform(row_df)[0]
        # ASK FOR THE NOTEBOOK'S NUMBER OF FEATURES, then show the top k. This is not padding.
        # LIME chooses its feature-selection method from num_features: at 6 or below it runs
        # forward_selection, refitting a ridge as each feature is added, and above 6 it runs
        # highest_weights, one ridge over all features. The published figures ask for min(8, n) and so get
        # highest_weights. Asking for 4 here got forward_selection and returned weights that differed from
        # the paper's in the fourth decimal, on the same well, from the same model. Measured on BC11:
        # 7.8e-04 apart with num_features=4, and 0.0e+00 with min(8, n). The interface still lists four.
        n_ask = min(8, len(ex.feature_names))
        with _lime_lock:
            # LIME's sampler, discretizer and base share ONE RandomState object, so replacing the attribute
            # does not reset the discretizer. Reseeding it in place is what makes repeated calls identical,
            # and it reproduces a freshly built explainer exactly (verified, 0.0e+00).
            ex.random_state.seed(3)
            exp = ex.explain_instance(xt, clf.predict_proba, labels=(0,), num_features=n_ask,
                                      num_samples=LIME_SAMPLES)
        # Label 0 is unsafe, so a positive weight pushes toward unsafe. LIME returns as_map() already
        # ordered by descending |weight|, so the first k are the strongest. float() is not decoration:
        # LIME hands back numpy scalars and FastAPI's JSON encoder refuses them.
        pairs = [(ex.feature_names[i], round(float(w), 4)) for i, w in exp.as_map()[0] if abs(w) > 1e-6]
        return pairs[:k]
    except Exception as e:
        print(f"LIME unavailable ({type(e).__name__}: {e}); occlusion fallback")

    med = card.get("feature_medians", {})
    if not med:
        return []
    p0 = float(model.predict_proba(row_df)[0][0])
    out = []
    for f in card["features"]:
        if f not in med or med[f] is None:
            continue
        r = row_df.copy()
        r.iloc[0, r.columns.get_loc(f)] = med[f]
        out.append((f, p0 - float(model.predict_proba(r)[0][0])))
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return [(f, round(float(c), 4)) for f, c in out[:k] if abs(c) > 1e-6]


def confidence(scope, p_unsafe, warns):
    """Heuristic, not a calibrated interval. The local model has the tighter validation region, coverage
    warnings pull it down, and a probability sitting near the cut pulls it down."""
    level = 2 if scope == "local" else 1
    if not warns and not (0.34 <= p_unsafe <= 0.66):
        level += 1
    if warns:
        level -= 1
    level = int(np.clip(level, 0, 2))
    return ["Low", "Moderate", "High"][level], "●" * (level + 1) + "○" * (2 - level)


# ---------------------------------------------------------------- prediction

def _build_row(card, lat, lon, depth, scope):
    """Assemble the feature vector in the pipeline's own column order, looking up what the user cannot
    supply. Column order is not cosmetic: a fitted pipeline will predict happily from a mis-ordered frame."""
    warns = []
    row = {"Latitude": float(lat), "Longitude": float(lon), "Actual_Depth(m)": float(depth)}
    cols = card["features"]

    if any(c in cols for c in features.TERRAIN_BANDS):
        t, w = features.terrain(lat, lon)
        warns += w
        if t is None:
            return None, warns
        row.update(t)
    if any(c in cols for c in features.SOIL_BANDS):
        s, w = features.soil(lat, lon)
        warns += w
        if s is None:
            return None, warns
        row.update(s)
    if "surf_geo" in cols:
        g, w = features.surf_geo(lat, lon)
        warns += w
        row["surf_geo"] = g

    missing = [c for c in cols if c not in row]
    if missing:
        return None, warns + [f"This model needs {', '.join(missing)}, which the app cannot look up."]

    # A NAMED frame in the fitted order is the only thing between a reordered feature vector and a
    # confident wrong answer. scikit-learn raises on a reordered DataFrame, because the first pipeline step
    # carries feature_names_in_ from fit time, and silently accepts a reordered array or list, returning
    # different numbers. So this must never be handed to predict as a list or an ndarray.
    return pd.DataFrame([{c: row[c] for c in cols}])[cols], warns


def fitted_columns(model):
    """The column order recorded at fit time, read off the pipeline rather than off the card."""
    step = model.named_steps["prep"]
    first = step.steps[0][1] if hasattr(step, "steps") else step
    return list(getattr(first, "feature_names_in_", []))


def predict(lat, lon, depth, threshold=10):
    """Structured prediction for one point, one depth, one threshold."""
    try:
        lat, lon, depth = float(lat), float(lon), float(depth)
    except (TypeError, ValueError):
        return {"ok": False, "error": "Enter numeric latitude, longitude and depth."}
    if threshold not in THRESHOLDS:
        return {"ok": False, "error": "That threshold is not served."}
    if depth <= 0 or depth > 400:
        return {"ok": False, "error": "Enter a drilling depth between 1 and 400 metres."}
    if not features.in_bangladesh(lat, lon):
        return {"ok": False, "error": "Point is outside Bangladesh. The models and their validation cover "
                                      "Bangladesh only; pick a location inside the country."}

    inside, upazila = features.in_local_region(lat, lon)
    scope = "local" if inside else "national"
    if (scope, threshold) not in MODELS:
        return {"ok": False, "error": f"The {scope} model is not built. Run App/export_app_models.py."}
    model, card = MODELS[(scope, threshold)]

    row, warns = _build_row(card, lat, lon, depth, scope)
    if row is None:
        # A point inside the local region whose raster lookup fails, terrain or soil, would otherwise be
        # served a prediction built entirely from filled-in values, so it falls through to the national
        # model instead. The national model reads surf_geo from a polygon and needs no raster at all.
        if scope == "local" and ("national", threshold) in MODELS:
            model, card = MODELS[("national", threshold)]
            scope, upazila = "national", None
            row, w2 = _build_row(card, lat, lon, depth, scope)
            warns += w2
        if row is None:
            return {"ok": False, "error": "Features unavailable here: " + "; ".join(warns)}

    p_unsafe = float(model.predict_proba(row)[0][0])         # class 0 is unsafe
    safe = (1.0 - p_unsafe) >= DECISION_CUTOFF
    conf, dots = confidence(scope, p_unsafe, warns)
    acc = card["test_accuracy"] if LABEL_ACCURACY == "test" else card["cv_accuracy"]

    return {
        "ok": True,
        "threshold_ugl": threshold, "standard": STANDARD_NAME[threshold],
        "p_unsafe": round(p_unsafe, 4), "verdict": "likely safe" if safe else "likely unsafe",
        "band": band(p_unsafe), "cutoff": DECISION_CUTOFF,
        "scope": scope, "upazila": upazila,
        "model_label": model_label(scope, threshold),
        # The inputs are the same at both thresholds but the model and its accuracy are not, so the
        # interface prints this once and the accuracy inside each threshold's own card.
        "model_inputs": model_inputs(scope, threshold),
        "confidence": conf, "confidence_dots": dots,
        "warnings": warns,
        "factors": [{"feature": pretty(f), "column": f, "weight": w,
                     "direction": "pushes unsafe" if w > 0 else "pushes safe"}
                    for f, w in local_factors(model, card, row, scope, threshold)],
        "features": {pretty(c): (None if pd.isna(v) else round(float(v), 4))
                     for c, v in row.iloc[0].items()},
        "accuracy": acc,
        "disclaimer": card["disclaimer"],
    }


def model_label(scope, threshold):
    """The sentence printed above the result. Approved wording: accuracy and inputs only, no well counts
    and no miss rate. Both of those stay in the model cards and in the paper."""
    card = MODELS[(scope, threshold)][1]
    acc = card["test_accuracy"] if LABEL_ACCURACY == "test" else card["cv_accuracy"]
    return f"{model_inputs(scope, threshold)} {acc * 100:.1f}% accurate."


def model_inputs(scope, threshold):
    """The same sentence without the accuracy. Two thresholds are served by two separately selected models
    whose accuracies differ (nationally 83.2% and 87.3%), so an accuracy printed once above both verdicts
    is wrong for one of them."""
    card = MODELS[(scope, threshold)][1]
    who = "Local model (Comilla)" if scope == "local" else "National model"
    return f"{who}: {card['label']}."


def meta():
    """Static metadata for the frontend. Nothing about an unserved threshold appears here, so gating is not
    something the interface can be talked out of."""
    b = features.terrain_bounds()
    labels = features.place_labels()
    models = []
    for (scope, th), (_, card) in sorted(MODELS.items()):
        models.append({
            "scope": scope, "threshold_ugl": th, "standard": STANDARD_NAME[th],
            "label": model_label(scope, th),
            "inputs": card["label"],
            "accuracy": card["test_accuracy"] if LABEL_ACCURACY == "test" else card["cv_accuracy"],
            "coverage": ("the eight upazilas around Comilla where the study sampled wells"
                         if scope == "local" else "Bangladesh"),
        })
    # The basemap is a fixed image with a fixed extent, so the frontend needs its geographic bounds to turn
    # a click into a coordinate and to place labels. It is a separate extent from the terrain crop.
    mp = os.path.join(HERE, "basemap_meta.json")
    basemap = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None

    return {
        "thresholds": THRESHOLDS,
        "standards": {str(t): STANDARD_NAME[t] for t in THRESHOLDS},
        "models": models,
        "map": basemap,
        "local_region": [{"name": n, "ring": r} for n, r in features.local_region_rings()],
        "local_bounds": None if b is None else dict(west=b[0], south=b[1], east=b[2], north=b[3]),
        # Place names are NOT sent any more. The map draws OpenStreetMap tiles, which already carry district,
        # upazila, union and village names rendered at the zoom each belongs to. Shipping our own 285 KB of
        # label coordinates to every visitor, then laying them out and collision-testing them in the browser,
        # was solving a problem the tiles do not have. place_labels.json stays on disk for other uses.
        "attribution": labels.get("attribution", ""),
        "cutoff": DECISION_CUTOFF,
        "disclaimer": next(iter(MODELS.values()))[1]["disclaimer"] if MODELS else "",
    }
