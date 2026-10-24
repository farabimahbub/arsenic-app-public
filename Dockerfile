FROM python:3.11-slim

# Install OpenMP for the tree libraries and Expat for GDAL/rasterio XML parsing.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Listen on all container interfaces. The platform can override PORT (default 7860).
ENV HOST=0.0.0.0
EXPOSE 7860
CMD ["python", "server.py"]
