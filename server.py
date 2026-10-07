# -*- coding: utf-8 -*-
"""
Web server: the static page plus a small JSON API over the inference core.

Run locally:  python server.py        (serves http://127.0.0.1:7860)

  GET  /             the page
  GET  /api/meta     served thresholds, model labels, the local outline, map bounds
  GET  /api/health   what loaded and what did not
  POST /api/predict  {lat, lon, depth, threshold?} -> result

/api/meta says nothing about an unserved threshold. The inference core gates that, not the page.
"""
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import inference

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Arsenic Well Screening, Bangladesh")

# Serves the vendored Leaflet build. Shipping it in the image drops the dependency on a script CDN.
# Map tiles still come over the network, which is the map's one runtime dependency.
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
    """Deliberately verbose. Check it first on a new deployment.

    A missing system library once broke the raster reader. The app fell back without erroring and
    predictions drifted by up to 0.33, with nothing in the logs. This endpoint is how that was found.
    """
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
    # Local runs stay on loopback. The container sets HOST=0.0.0.0.
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")))
