# -*- coding: utf-8 -*-
"""
The screening app's web server: the static frontend plus a small JSON API over the inference core.

Run locally:  python server.py        (serves http://127.0.0.1:7860)

Endpoints:
  GET  /              the app (static/index.html)
  GET  /basemap.png   the click-to-place map, rendered offline by make_basemap.py
  GET  /api/meta      served thresholds, model labels, local region outline, place names, map bounds
  GET  /api/health    what loaded and what did not, for diagnosing a deployment from the outside
  POST /api/predict   {lat, lon, depth, threshold?} -> structured result

/api/meta carries nothing about an unserved threshold, so gating the Bangladesh standard is enforced in the
inference core rather than in the interface.
"""
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import inference

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Arsenic Well Screening, Bangladesh")

# Serves the vendored Leaflet build. Leaflet is shipped inside the image rather than pulled from a CDN, so
# the interface does not depend on a third-party script host being reachable. Map tiles still come over the
# network; that is the one runtime dependency the map introduces.
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


class PredictIn(BaseModel):
    lat: float
    lon: float
    depth: float
    threshold: int = 10


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/meta")
def meta():
    return JSONResponse(inference.meta())


@app.get("/api/health")
def health():
    """Deliberately verbose. In July a missing libexpat1 broke rasterio, the app fell back silently, and
    predictions drifted by up to 0.33 with nothing in the logs. This endpoint is how that was found."""
    import features
    return JSONResponse({
        "models_loaded": [f"{s}_as{t}" for s, t in sorted(inference.MODELS)],
        "local_model": inference.LOCAL_MODEL,
        "serve_threshold_50": inference.SERVE_THRESHOLD_50,
        "label_accuracy": inference.LABEL_ACCURACY,
        "cutoff": inference.DECISION_CUTOFF,
        **features.status(),
    })


@app.post("/api/predict")
def predict(inp: PredictIn):
    return JSONResponse(inference.predict(inp.lat, inp.lon, inp.depth, threshold=inp.threshold))


if __name__ == "__main__":
    import uvicorn
    # Local runs stay loopback-only; the container sets HOST=0.0.0.0.
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")))
