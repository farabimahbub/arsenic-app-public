# -*- coding: utf-8 -*-
"""
Serving-side feature lookup.

Everything the two served models need follows from a clicked point and a typed depth:

  national   lat, lon, depth, surf_geo
  local      lat, lon, depth, surf_geo, and four soil bands

surf_geo comes from a point-in-polygon hit on the surface geology coverage in assets/, read by
surface_geology.py. That module carries its own shapefile and dBASE readers, so the container needs no
geopandas, shapely or fiona. The soil bands come from assets/soil_comilla.tif, sampled NEAREST so a
query reads the cell the training extraction read.

This module also reads four terrain bands, which no served model asks for. That raster is absent here,
so terrain() reports itself unavailable.

Every lookup returns (value, warnings). Warnings are user-facing sentences, so the page can show a
degraded reading rather than pass a filled-in guess off as a measurement.
"""
import os
import json
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
TERRAIN_PATH = os.path.join(ASSETS, "terrain_comilla.tif")
SOIL_PATH = os.path.join(ASSETS, "soil_comilla.tif")
GEOLOGY_PATH = os.path.join(ASSETS, "geo8bg.shp")
BOUNDARY_PATH = os.path.join(HERE, "bd_boundary.geojson")
BD_BBOX = dict(lat=(20.5, 26.7), lon=(88.0, 92.7))       # fallback when the geojson is absent

TERRAIN_BANDS = ["elevation", "slope", "twi", "hand"]
# The four SoilGrids inputs of configuration R5_soil, cropped to Comilla on a fixed pixel grid so a
# click reads the same cell the study trained on.
SOIL_BANDS = ["soil_organic_carbon", "soil_clay", "soil_sand", "soil_silt"]

_terrain = None          # rasterio dataset, or False after a failed open
_terrain_error = None
_soil = None             # rasterio dataset, or False after a failed open
_soil_error = None
_geology = None          # the surface_geology module handle, or False

# A GDAL dataset handle is not thread-safe and the server runs sync endpoints in a threadpool. Two
# simultaneous reads raise RasterioIOError inside the driver, which one user can trigger alone.
# Serializing costs microseconds, because each read takes one pixel from a memory-mapped local file.
_terrain_lock = threading.Lock()
_soil_lock = threading.Lock()


# ---------------------------------------------------------------- terrain

def _open_terrain():
    global _terrain, _terrain_error
    if _terrain is None:
        if not os.path.exists(TERRAIN_PATH):
            _terrain_error = f"missing {TERRAIN_PATH}"
            _terrain = False
        else:
            try:
                import rasterio
                ds = rasterio.open(TERRAIN_PATH)
                if ds.count != len(TERRAIN_BANDS):
                    raise ValueError(f"{ds.count} bands, expected {len(TERRAIN_BANDS)}")
                _terrain = ds
            except Exception as e:
                _terrain_error = f"{type(e).__name__}: {e}"
                print(f"features: terrain unavailable ({_terrain_error})")
                _terrain = False
    return _terrain or None


def terrain_bounds():
    """(west, south, east, north) of the terrain crop, or None if it failed to open."""
    ds = _open_terrain()
    if ds is None:
        return None
    b = ds.bounds
    return (b.left, b.bottom, b.right, b.top)


def has_terrain(lat, lon):
    """Is this point inside the terrain crop at all? Upazila routing is a separate, stricter check."""
    b = terrain_bounds()
    return bool(b) and b[0] <= float(lon) <= b[2] and b[1] <= float(lat) <= b[3]


def terrain(lat, lon):
    """The four terrain features at a point -> (dict | None, warnings)."""
    ds = _open_terrain()
    if ds is None:
        return None, ["Terrain data is unavailable, so the local model cannot run here."]
    if not has_terrain(lat, lon):
        return None, ["This point sits outside the terrain map, so the local model cannot run here."]

    with _terrain_lock:
        vals = np.array(next(ds.sample([(float(lon), float(lat))])), dtype=float)
    if ds.nodata is not None:
        vals[vals == ds.nodata] = np.nan
    feats = dict(zip(TERRAIN_BANDS, [float(v) for v in vals]))

    missing = [k for k, v in feats.items() if np.isnan(v)]
    warns = []
    if len(missing) == len(TERRAIN_BANDS):
        warns.append("No terrain data covers this exact point, so every value is being filled in. "
                     "Treat this estimate with strong caution.")
    elif missing:
        warns.append(f"No terrain data for {', '.join(missing)} at this point, so those values are being "
                     f"filled in. The estimate is less reliable here.")
    return feats, warns


# ---------------------------------------------------------------- soil

def _open_soil():
    global _soil, _soil_error
    if _soil is None:
        if not os.path.exists(SOIL_PATH):
            _soil_error = f"missing {SOIL_PATH}"
            _soil = False
        else:
            try:
                import rasterio
                ds = rasterio.open(SOIL_PATH)
                if ds.count != len(SOIL_BANDS):
                    raise ValueError(f"{ds.count} bands, expected {len(SOIL_BANDS)}")
                if list(ds.descriptions) != SOIL_BANDS:
                    raise ValueError(f"band order {ds.descriptions}, expected {SOIL_BANDS}")
                _soil = ds
            except Exception as e:
                _soil_error = f"{type(e).__name__}: {e}"
                print(f"features: soil unavailable ({_soil_error})")
                _soil = False
    return _soil or None


def soil_bounds():
    """(west, south, east, north) of the soil crop, or None if it failed to open."""
    ds = _open_soil()
    if ds is None:
        return None
    b = ds.bounds
    return (b.left, b.bottom, b.right, b.top)


def has_soil(lat, lon):
    """Is this point inside the soil crop at all?"""
    b = soil_bounds()
    return bool(b) and b[0] <= float(lon) <= b[2] and b[1] <= float(lat) <= b[3]


def soil(lat, lon):
    """The four soil features at a point -> (dict | None, warnings)."""
    ds = _open_soil()
    if ds is None:
        return None, ["Soil data is unavailable, so the local model cannot run here."]
    if not has_soil(lat, lon):
        return None, ["This point sits outside the soil map, so the local model cannot run here."]

    with _soil_lock:
        vals = np.array(next(ds.sample([(float(lon), float(lat))])), dtype=float)
    if ds.nodata is not None:
        vals[vals == ds.nodata] = np.nan
    feats = dict(zip(SOIL_BANDS, [float(v) for v in vals]))

    missing = [k for k, v in feats.items() if np.isnan(v)]
    warns = []
    if len(missing) == len(SOIL_BANDS):
        warns.append("No soil data covers this exact point, so every value is being filled in. "
                     "Treat this estimate with strong caution.")
    elif missing:
        warns.append(f"No soil data for {', '.join(missing)} at this point, so those values are being "
                     f"filled in. The estimate is less reliable here.")
    return feats, warns


# ---------------------------------------------------------------- surface geology

def _open_geology():
    global _geology
    if _geology is None:
        if not os.path.exists(GEOLOGY_PATH):
            _geology = False
        else:
            try:
                import surface_geology                      # noqa: kept as a module handle
                _geology = surface_geology
            except Exception as e:
                print(f"features: geology reader unavailable ({type(e).__name__}: {e})")
                _geology = False
    return _geology or None


def surf_geo(lat, lon):
    """The GSB surface-geology rank at a point -> (float | nan, warnings).

    A point over open water, over a lake, or outside the mapped area falls back to the nearest genuine
    geological polygon, exactly as the training assignment did, and the distance drives the warning."""
    sg = _open_geology()
    if sg is None:
        return float("nan"), ["Surface geology is unavailable, so the national model is filling that value "
                              "in and the estimate is weaker."]
    rec = sg.assign([float(lat)], [float(lon)], GEOLOGY_PATH)[0]
    val = rec.get("geo_rank")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan"), ["No surface geology unit could be assigned here, so the model is filling that "
                              "value in."]

    warns = []
    # The map sheet and its FGDC metadata both state a 250 m RMS transformation error, so a point
    # within that distance of a contact cannot be assigned confidently to either side. Do not take
    # this figure from the scan resolution in the same paragraph, which is dots per inch.
    if rec.get("method") == "nearest":
        km = float(rec.get("fallback_km") or 0.0)
        warns.append(f"This point is not on mapped land geology, so the nearest mapped unit was used "
                     f"({km:.1f} km away). The estimate is less certain here.")
    elif float(rec.get("dist_to_edge_km") or 9.9) <= 0.25:
        warns.append("This point sits within the geological map's own 250 m accuracy limit of a boundary "
                     "between units, so the geology here could go either way.")
    return float(val), warns


# ---------------------------------------------------------------- Bangladesh scope

def _load_rings():
    if not os.path.exists(BOUNDARY_PATH):
        return None
    g = json.load(open(BOUNDARY_PATH))["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    return [(p[0], p[1:]) for p in polys]


_rings = _load_rings()


def _ring_contains(ring, x, y):
    inside = False
    for i in range(len(ring)):
        x1, y1 = ring[i - 1][0], ring[i - 1][1]
        x2, y2 = ring[i][0], ring[i][1]
        if (y1 > y) != (y2 > y):
            if x1 + (y - y1) * (x2 - x1) / (y2 - y1) > x:
                inside = not inside
    return inside


def in_bangladesh(lat, lon):
    """Point-in-polygon against the GAUL boundary; bounding box fallback if the geojson is missing."""
    x, y = float(lon), float(lat)
    if _rings is None:
        return BD_BBOX["lat"][0] <= y <= BD_BBOX["lat"][1] and BD_BBOX["lon"][0] <= x <= BD_BBOX["lon"][1]
    for outer, holes in _rings:
        if _ring_contains(outer, x, y) and not any(_ring_contains(h, x, y) for h in holes):
            return True
    return False


def boundary_lonlat():
    """Outer boundary rings as [[(lon, lat), ...], ...] for drawing; [] if the geojson is missing."""
    return [outer for outer, _ in _rings] if _rings else []


# ---------------------------------------------------------------- local region and place names

LOCAL_REGION_PATH = os.path.join(ASSETS, "local_region.geojson")
LABELS_PATH = os.path.join(ASSETS, "place_labels.json")

_region = None      # [(name, [(outer, holes), ...]), ...] or False
_labels = None


def _load_region():
    global _region
    if _region is None:
        if not os.path.exists(LOCAL_REGION_PATH):
            _region = False
        else:
            g = json.load(open(LOCAL_REGION_PATH, encoding="utf-8"))
            out = []
            for f in g["features"]:
                geom = f["geometry"]
                polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
                out.append((f["properties"]["name"], [(p[0], p[1:]) for p in polys]))
            _region = out
    return _region or None


def in_local_region(lat, lon):
    """(inside, upazila name or None). Decides which model runs.

    The region is the set of upazilas that contain at least one study well, built by
    App/build_boundaries.py. It is deliberately derived from the wells rather than from a list of thana
    names: the six sampling thanas span eight current upazilas, because Comilla Sadar was split and two
    clusters cross boundaries."""
    reg = _load_region()
    if reg is None:
        return False, None
    x, y = float(lon), float(lat)
    for name, polys in reg:
        for outer, holes in polys:
            if _ring_contains(outer, x, y) and not any(_ring_contains(h, x, y) for h in holes):
                return True, name
    return False, None


def local_region_rings():
    """Outer rings of the local region as [(name, [(lon, lat), ...]), ...], for drawing the outline."""
    reg = _load_region()
    return [(name, outer) for name, polys in (reg or []) for outer, _ in polys]


def place_labels():
    """District and upazila label anchors for the map, or empty lists if the file is absent."""
    global _labels
    if _labels is None:
        _labels = (json.load(open(LABELS_PATH, encoding="utf-8")) if os.path.exists(LABELS_PATH)
                   else {"districts": [], "upazilas": [], "attribution": ""})
    return _labels


# ---------------------------------------------------------------- diagnostics

def status():
    """For /api/health. Names what is loaded and why anything absent is absent."""
    ds = _open_terrain()
    return {
        "terrain": "ok" if ds is not None else "unavailable",
        "terrain_error": _terrain_error,
        "terrain_bands": None if ds is None else list(ds.descriptions or TERRAIN_BANDS),
        "terrain_bounds": terrain_bounds(),
        "soil": "ok" if _open_soil() is not None else "unavailable",
        "soil_error": _soil_error,
        "soil_bounds": soil_bounds(),
        "geology": "ok" if _open_geology() is not None else "unavailable",
        "boundary_polygons": 0 if _rings is None else len(_rings),
        "local_region_upazilas": len(_load_region() or []),
        "assets_mb": round(sum(os.path.getsize(os.path.join(ASSETS, f))
                               for f in os.listdir(ASSETS)) / 1e6, 1) if os.path.isdir(ASSETS) else 0,
    }
