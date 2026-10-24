# -*- coding: utf-8 -*-
"""Load models, select one by location, and return predictions and feature contributions.

The local model covers the eight study upazilas listed in assets/local_region.geojson.
Other locations in Bangladesh use the national model. Locations outside Bangladesh
are rejected. Both models take latitude, longitude and planned depth; the remaining
features come from the bundled soil and surface-geology data.

The constants below select the local model, served thresholds, accuracy label and
decision cutoff. The deployed app serves the 10 ug/L threshold only."""
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

# The local model uses R5_soil: location, depth, four soil bands and surface geology.
# Its held-out test accuracy is 0.9375. See the model card for inputs and selection details.
LOCAL_MODEL = "local"

# Serve the WHO 10 ug/L guideline only. This flag controls model loading and API metadata.
# The study reports 50 ug/L separately. Its independently fitted, uncalibrated models
# can give inconsistent probabilities across the two thresholds.
SERVE_THRESHOLD_50 = False
THRESHOLDS = [10, 50] if SERVE_THRESHOLD_50 else [10]

# Which accuracy the interface quotes. "test" is the held-out figure and the one the paper reports.
LABEL_ACCURACY = "test"

# Minimum predicted probability of the safe class for a safe classification.
# Increasing this cutoff requires a higher predicted probability before classifying a point as safe.
DECISION_CUTOFF = 0.5

STANDARD_NAME = {10: "WHO guideline", 50: "Bangladesh standard"}

# Display labels only. The fitted pipelines use the original feature names.
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
    """Cache a LIME explainer using the study's preprocessed training rows.

    The lock prevents concurrent requests from building duplicate explainers."""
    with _lime_lock:
        if (scope, th) not in _explainers:
            from lime.lime_tabular import LimeTabularExplainer
            bg = np.load(os.path.join(HERE, f"lime_bg_{SCOPES[scope]}_as{th}.npz"), allow_pickle=True)
            _explainers[(scope, th)] = LimeTabularExplainer(
                bg["X"], feature_names=list(bg["features"]), class_names=["unsafe", "safe"],
                mode="classification", discretize_continuous=True, random_state=3)
        return _explainers[(scope, th)]


def local_factors(model, card, row_df, scope, th, k=4):
    """Return the largest feature contributions using a fixed random seed.

    If LIME or its background is unavailable, compare each feature with its training median."""
    try:
        ex = _explainer(scope, th)
        prep, clf = model.named_steps["prep"], model.named_steps["clf"]
        xt = prep.transform(row_df)[0]
        # Use the study's min(8, n) feature count before selecting the top k for display.
        # LIME uses forward_selection for num_features <= 6 and highest_weights above 6.
        # Requesting only four features can therefore change the explanation weights.
        n_ask = min(8, len(ex.feature_names))
        with _lime_lock:
            # The sampler, discretizer and base share a RandomState. Reseed it in place
            # to reset all three and keep repeated explanations reproducible.
            ex.random_state.seed(3)
            exp = ex.explain_instance(xt, clf.predict_proba, labels=(0,), num_features=n_ask,
                                      num_samples=LIME_SAMPLES)
        # Positive weights increase the contribution toward class 0 (unsafe).
        # as_map() sorts by absolute weight. Convert numpy scalars to float for JSON encoding.
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
    """Return a heuristic confidence label, not a calibrated interval.

    The label depends on model scope, data warnings and proximity to the risk-band cutoffs."""
    level = 2 if scope == "local" else 1
    if not warns and not (0.34 <= p_unsafe <= 0.66):
        level += 1
    if warns:
        level -= 1
    level = int(np.clip(level, 0, 2))
    return ["Low", "Moderate", "High"][level], "●" * (level + 1) + "○" * (2 - level)


# ---------------------------------------------------------------- prediction

def _build_row(card, lat, lon, depth, scope):
    """Look up the required features and return a DataFrame in the fitted column order."""
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

    # Preserve the fitted column order in a named DataFrame. scikit-learn checks
    # DataFrame column names, but a reordered array can produce incorrect predictions.
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
        # Use the national model if local raster data is unavailable.
        # The national model needs surface geology but no raster features.
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
        # Keep accuracy with each threshold's result, even when the input labels are shared.
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
    """Format the model inputs and accuracy for display.

    Well counts and miss rates are recorded in the model cards and paper."""
    card = MODELS[(scope, threshold)][1]
    acc = card["test_accuracy"] if LABEL_ACCURACY == "test" else card["cv_accuracy"]
    return f"{model_inputs(scope, threshold)} {acc * 100:.1f}% accurate."


def model_inputs(scope, threshold):
    """Format the model's region and inputs, leaving accuracy to each threshold's result."""
    card = MODELS[(scope, threshold)][1]
    who = "Local model (Comilla)" if scope == "local" else "National model"
    return f"{who}: {card['label']}."


def meta():
    """Return frontend metadata for the served models and thresholds."""
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
    # Keep metadata for the legacy image basemap when available.
    # The current frontend uses map tiles.
    mp = os.path.join(HERE, "basemap_meta.json")
    basemap = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None

    return {
        "thresholds": THRESHOLDS,
        "standards": {str(t): STANDARD_NAME[t] for t in THRESHOLDS},
        "models": models,
        "map": basemap,
        "local_region": [{"name": n, "ring": r} for n, r in features.local_region_rings()],
        "local_bounds": None if b is None else dict(west=b[0], south=b[1], east=b[2], north=b[3]),
        # Map tiles supply place names. Only the stored attribution is returned here.
        "attribution": labels.get("attribution", ""),
        "cutoff": DECISION_CUTOFF,
        "disclaimer": next(iter(MODELS.values()))[1]["disclaimer"] if MODELS else "",
    }
