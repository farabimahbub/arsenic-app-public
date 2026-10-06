FROM python:3.11-slim

# System libs the wheels rely on: OpenMP runtime (lightgbm, xgboost) and Expat (GDAL/rasterio XML parsing).
# The slim base image omits both. Leaving libexpat1 out is what broke the July deployment: rasterio failed
# to import, the app fell back to demo mode without erroring, map clicks died, and predictions drifted up to
# 0.33 with nothing in the logs. /api/health exists to make that failure visible instead of silent.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# bind on all interfaces inside the container; the platform injects PORT (server.py defaults to 7860),
# and the grid URL/token arrive via ARSENIC_GRID / ARSENIC_GRID_TOKEN env vars
ENV HOST=0.0.0.0
EXPOSE 7860
CMD ["python", "server.py"]
