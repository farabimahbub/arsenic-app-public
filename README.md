# Arsenic well screening for Bangladesh

Enter a location in Bangladesh and a planned drilling depth to estimate the chance that groundwater
arsenic exceeds the WHO guideline of 10 µg/L. The app looks up soil and surface-geology data and
returns a risk estimate, a risk band and the main features contributing to the prediction.

[Open the screening app](https://arsenic-app.onrender.com).

**This is a screening aid, not a water test.** It cannot certify a well as safe. Test the well water
before drinking it.

## Models and coverage

The app selects a model by location. Both use information available before drilling and classify
predictions at a probability cutoff of 0.5.

| Coverage | Model and inputs | Held-out test accuracy | Test wells | Miss rate among unsafe test wells |
|---|---|---|---|---|
| Eight study upazilas in Comilla (Cumilla) | XGBoost, `R5_soil`: location, depth, four soil bands and surface geology | 93.75% | 32 | 6.25% |
| Other locations in Bangladesh | LightGBM, `NX2_minus_satellite`: location, depth and surface geology | 83.17% | 707 | 22.9% |

The local area includes Brahman Para, Burichang, Chandina, Comilla Adarsha Sadar, Comilla Sadar Dakshin,
Daudkandi, Debidwar and Muradnagar. The polygons in `assets/local_region.geojson` determine routing.
If the local model's raster data is unavailable, the app uses the national model and returns a warning.

The models were selected by held-out test accuracy, following the study's reporting rule. Ties are
resolved by miss rate, cross-validation accuracy and a fixed model order. These test scores describe
performance on the study split; they do not measure confidence in an individual prediction.

Probabilities are uncalibrated model outputs. The displayed confidence label is a heuristic based on
model coverage, data warnings and the predicted probability. The two model cards record the parameters,
preprocessing, selection rule, feature medians and validation results:

- `model_card_local_as10.json`
- `model_card_national_as10.json`

## Run locally

With Docker:

```sh
docker build -t arsenic-app .
docker run -p 7860:7860 arsenic-app
```

Open <http://127.0.0.1:7860>.

To run without Docker, use Python 3.11 and install the OpenMP and Expat system libraries required by
LightGBM and rasterio, then run:

```sh
pip install -r requirements.txt
python server.py
```

Package versions match the training environment. Changing them can prevent the serialized models from
loading or change their predictions.

## API

| Endpoint | Response |
|---|---|
| `GET /` | Web interface |
| `GET /api/meta` | Served thresholds, model labels, local area and map bounds |
| `GET /api/health` | Loaded models, asset status and feature-reader errors |
| `POST /api/predict` | Prediction for a location and depth |

Example prediction request:

```json
{"lat": 23.46, "lon": 91.18, "depth": 40}
```

Check `/api/health` after setup to confirm that the required models, soil data and geology reader loaded.

## Data included and omitted

The underlying BGS/DPHE survey records and Comilla field records are not included because we do not
have permission to redistribute them.

The public repository also omits three files used by the hosted app:

- `lime_bg_local_as10.npz` and `lime_bg_national_as10.npz` contain preprocessed training rows from
  which survey coordinates and depths can be recovered. Without them, explanations compare each
  feature with its training median instead of using LIME. This changes the explanation method while
  leaving the prediction model unchanged.
- `assets/terrain_comilla.tif` includes terrain derivatives with redistribution restrictions.
  Neither served model uses its terrain bands.

The public `bd_boundary.geojson` uses geoBoundaries ADM2 district polygons in place of the hosted
app's FAO GAUL outline. It determines whether a query is inside Bangladesh. The GAUL file is omitted
because we do not have permission to redistribute it. Both versions agreed at the locations checked
when this repository was prepared.

The study workspace contains the asset-building scripts: `App/build_boundaries.py`,
`App/export_soil_comilla.py`, `App/build_serving_assets.py` and `App/export_app_models.py`. They
require access to the source data and are not included in this app repository.

## Licences and attribution

The application code is available under the MIT licence in [LICENSE](LICENSE).
[NOTICE.md](NOTICE.md) lists the third-party data, components and attribution requirements.

## Citation

The manuscript is under review. A citation will be added when it is published.
