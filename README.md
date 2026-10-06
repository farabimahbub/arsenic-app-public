# Arsenic well screening, Bangladesh

This service estimates the chance that groundwater at a point holds more arsenic than the WHO
guideline of 10 ug/L. You give it a latitude, a longitude and a depth. It looks up soil and surface
geology at that point, then returns a probability, a risk band, and the inputs that moved the answer.

Every input is known before anyone drills. The service never needs a water sample.

**This is a screening aid. It does not test water.** No estimate here can certify a well as safe.
Arsenic has to be confirmed by testing water drawn from the finished well.

A live instance runs at <https://arsenic-app.onrender.com>.

## What it serves

Two models answer, both at 10 ug/L, both deciding at a probability of 0.5.

Inside the eight upazilas of Cumilla district that contain study wells, the local model answers. It
reads location, depth, four SoilGrids bands and the surface geology class. It is an XGBoost model and
it scored 0.9375 on its 32 held-out wells, against a majority baseline of 0.5125. It missed one unsafe
well of the sixteen in that split.

Everywhere else in Bangladesh, the national model answers. It reads location, depth and the surface
geology class. It is a LightGBM model and it scored 0.8317 on its 707 held-out wells, against a
baseline of 0.5792. Its miss rate on unsafe wells is 0.229, which is the number to weigh before
trusting a safe verdict outside Cumilla.

The eight local upazilas are Brahman Para, Burichang, Chandina, Comilla Adarsha Sadar, Comilla Sadar
Dakshin, Daudkandi, Debidwar and Muradnagar. `assets/local_region.geojson` holds their outlines and
decides which model runs.

Both models carry a card next to them. `model_card_local_as10.json` and
`model_card_national_as10.json` record the classifier, its parameters, the split, the accuracy, the
preprocessing, the feature medians and the configuration each came from.

## Run it

With Docker:

    docker build -t arsenic-app .
    docker run -p 7860:7860 arsenic-app

Then open <http://127.0.0.1:7860>.

Without Docker you need Python 3.11, plus the OpenMP and Expat system libraries that LightGBM and
rasterio load:

    pip install -r requirements.txt
    python server.py

The versions in `requirements.txt` are pinned to the ones that fitted the models. A serialized
scikit-learn model is not portable across versions, so changing them can change an answer or stop the
model loading.

## The API

    GET  /             the interface
    GET  /api/meta     served thresholds, model labels, the local outline, map bounds
    GET  /api/health   what loaded and what did not
    POST /api/predict  {"lat": 23.46, "lon": 91.18, "depth": 40} returns the result

Check `/api/health` first on a new deployment. A missing system library once broke the raster reader,
the service answered from a fallback without complaining, and predictions drifted by up to 0.33. That
endpoint is how it was found.

## What this repository leaves out

The study ran on well records from the British Geological Survey and DPHE national survey and on a
field dataset from Cumilla. Neither is ours to pass on, so no well record appears here. Three files
that ship inside the running container are held back too.

`lime_bg_local_as10.npz` and `lime_bg_national_as10.npz` hold the preprocessed training rows that LIME
samples from. The scaler inside the shipped models turns those rows back into exact well coordinates
and depths, so they are survey records in another coat. Without them the service explains an answer by
occlusion against the training medians, which are in the model cards. You still get the inputs that
moved the answer, computed a different way.

`assets/terrain_comilla.tif` carries elevation and slope from SRTM, which is public domain, together
with TWI and HAND derived from MERIT Hydro, which is not. Neither served model reads a terrain band,
so the file is dead weight in the container and nothing is lost by dropping it.

One file is here but was rebuilt. `bd_boundary.geojson` decides whether a point lies in Bangladesh at
all. The deployed copy came from FAO GAUL, whose licence forbids passing it on without written consent,
so this one is built instead from geoBoundaries ADM2, which is CC BY 3.0 IGO. It holds the 64 district
polygons rather than one dissolved outline, because the only code that reads it asks whether any
polygon contains the point. Nothing draws it. It refuses Kolkata, Shillong and the Bay of Bengal and
accepts Dhaka and Cumilla, which is what the deployed copy does.

The code handles the three absences on its own. No file in this repository was edited to make that
work, and predictions match the live service exactly.

If you hold the rights to the underlying data, the study code rebuilds each one:
`App/build_boundaries.py` for the boundaries, `App/export_soil_comilla.py` for the soil raster,
`App/build_serving_assets.py` for the terrain crop and the geology coverage, and
`App/export_app_models.py` for the models, their cards and the LIME backgrounds.

## Data and licences

The code in this repository is under the MIT licence in `LICENSE`.

The data files are not. `NOTICE.md` names every third-party source, its licence and the attribution
that has to travel with it. Read it before you redistribute any asset in `assets/`.

## Citation

The manuscript describing this work is under review. A citation will be added here when it is
published.
