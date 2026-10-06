# -*- coding: utf-8 -*-
"""
Assign GSB surface-geology units to well coordinates by point-in-polygon.

Source map: "Geological Map of Bangladesh", Alam, Hasan and Khan (Geological Survey of Bangladesh)
with Whitney (USGS), 1:1,000,000, published 1990; digitised by Persits, Wandrey, Milici (USGS) and
Manwar (GSB), released as USGS Open File Report 97-470H. The digital coverage stores the unit in a
field named GLG, which is the column the domain expert pointed at.

The map sheet states an overall RMS error of 250 m for the paper-to-digital transformation, so a well
sitting within roughly 250 m of a contact cannot be assigned confidently. `dist_to_edge_km` is
returned per well so those cases can be found rather than assumed away.

No third-party dependency: the shapefile and dBASE readers below are minimal but complete for the
polygon and character/numeric field types this coverage actually uses. Geometry is plain WGS84
lon/lat, the same frame the well coordinates use, so no reprojection is involved.
"""
import struct

import numpy as np

# ---------------------------------------------------------------- legend, read off the map sheet
# The 1990 sheet prints "St. Marin limestone"; the unit is St. Martin's Island limestone.
GLG_NAME = {
    "csd": "Beach and dune sand",
    "dsw": "Mangrove swamp deposit",
    "dm": "Tidal mud",
    "dt": "Tidal deltaic deposits",
    "de": "Estuarine deposits",
    "dsl": "Deltaic silt",
    "dsd": "Deltaic sand",
    "ppc": "Marsh clay and peat",
    "asd": "Alluvial sand",
    "asl": "Alluvial silt",
    "asc": "Alluvial silt and clay",
    "ac": "Chandina alluvium",
    "ava": "Valley alluvium and colluvium",
    "afy": "Young gravelly sand",
    "afo": "Old gravelly sand",
    "rb": "Barind clay residuum",
    "rm": "Madhupur clay residuum",
    "Qsm": "St. Martin's limestone (Pleistocene)",
    "QTdd": "Dihing and Dupi Tila Formation undivided",
    "QTdi": "Dihing Formation (Pleistocene and Pliocene)",
    "QTdt": "Dupi Tila Formation (Pleistocene and Pliocene)",
    "QTg": "Girujan Clay (Pleistocene and Neogene)",
    "Tt": "Tipam Sandstone (Neogene)",
    "Tbb": "Boka Bil Formation (Neogene)",
    "Tb": "Bhuban Formation (Miocene)",
    "Tba": "Barail Formation (Oligocene)",
    "Tj": "Jaintia Group (Kopili Fm, Sylhet Limestone, Tura Fm)",
    "lake": "Lake",
    "H2O": "Ocean and wide river",
    "U": "Areas outside of Bangladesh",
}

# Stratigraphic order exactly as the map sheet prints its legend: youngest coastal Holocene first,
# oldest bedrock last. Used as the ordinal encoding for modelling. The order comes from the published
# legend alone and never from the arsenic labels, so it introduces no target leakage; ordering these
# categories by their measured safety rate would be target encoding and is deliberately not done.
# Age is the mechanistically relevant axis here: young reducing delta sediment releases arsenic,
# old weathered and oxidised terrace material does not.
GLG_STRAT_ORDER = [
    "csd",                                          # coastal
    "dsw", "dm", "dt", "de", "dsl", "dsd", "ppc",   # deltaic
    "asd", "asl", "asc", "ac", "ava",               # alluvial
    "afy", "afo",                                   # alluvial fan
    "rb", "rm",                                     # residual, Pleistocene terrace
    "Qsm", "QTdd", "QTdi", "QTdt", "QTg", "Tt", "Tbb", "Tb", "Tba", "Tj",   # bedrock, young to old
]
GLG_AGE = {code: i for i, code in enumerate(GLG_STRAT_ORDER)}

# Groupings the map sheet itself uses, for the broad class.
GLG_GROUP = {
    **{k: "Holocene coastal/deltaic" for k in ("csd", "dsw", "dm", "dt", "de", "dsl", "dsd", "ppc")},
    **{k: "Holocene alluvial" for k in ("asd", "asl", "asc", "ac", "ava")},
    **{k: "Alluvial fan" for k in ("afy", "afo")},
    **{k: "Residual (Pleistocene terrace)" for k in ("rb", "rm")},
    **{k: "Bedrock" for k in ("Qsm", "QTdd", "QTdi", "QTdt", "QTg", "Tt", "Tbb", "Tb", "Tba", "Tj")},
    **{k: "Not geology" for k in ("lake", "H2O", "U")},
}

NON_GEOLOGY = ("lake", "H2O", "U")


# ---------------------------------------------------------------- minimal readers
def read_dbf(path):
    """Read a dBASE III/IV table into a list of dicts. Character and numeric fields only."""
    b = open(path, "rb").read()
    nrec, hlen, rlen = struct.unpack("<i2H", b[4:12])
    fields, off = [], 32
    while b[off] != 0x0D:
        name = b[off:off + 11].split(b"\x00")[0].decode("latin-1")
        ftype = chr(b[off + 11])
        flen = b[off + 16]
        fields.append((name, ftype, flen))
        off += 32
    rows = []
    for i in range(nrec):
        p = hlen + i * rlen
        if b[p:p + 1] == b"*":                       # deleted record
            continue
        rec, q = {}, p + 1
        for name, ftype, flen in fields:
            raw = b[q:q + flen].decode("latin-1").strip()
            if ftype in "NF":
                rec[name] = float(raw) if raw not in ("", "-", ".") else np.nan
            else:
                rec[name] = raw
            q += flen
        rows.append(rec)
    return rows


def read_polygons(path):
    """
    Read polygon geometry from a .shp file.

    Returns (bboxes, rings) where bboxes is an (n, 4) array of xmin, ymin, xmax, ymax and rings[i]
    is a list of (m, 2) arrays, one per ring of shape i (outer rings and holes alike; the even-odd
    crossing rule below handles holes without needing to tell them apart).
    """
    b = open(path, "rb").read()
    shp_type, = struct.unpack("<i", b[32:36])
    if shp_type != 5:
        raise ValueError(f"expected polygon shapefile (type 5), found type {shp_type}")
    bboxes, rings, off = [], [], 100
    while off < len(b):
        _, rlen = struct.unpack(">ii", b[off:off + 8])
        p = off + 8
        stype, = struct.unpack("<i", b[p:p + 4])
        if stype == 0:                                # null shape
            bboxes.append([np.nan] * 4); rings.append([])
            off += 8 + rlen * 2
            continue
        box = struct.unpack("<4d", b[p + 4:p + 36])
        nparts, npoints = struct.unpack("<ii", b[p + 36:p + 44])
        parts = struct.unpack(f"<{nparts}i", b[p + 44:p + 44 + 4 * nparts])
        q = p + 44 + 4 * nparts
        pts = np.frombuffer(b, dtype="<f8", count=npoints * 2, offset=q).reshape(-1, 2)
        idx = list(parts) + [npoints]
        bboxes.append(list(box))
        rings.append([pts[idx[k]:idx[k + 1]] for k in range(nparts)])
        off += 8 + rlen * 2
    return np.asarray(bboxes, float), rings


# ---------------------------------------------------------------- point in polygon
def _in_rings(x, y, ring_list):
    """Even-odd ray crossing across every ring of one shape, so interior holes exclude correctly."""
    inside = False
    for r in ring_list:
        x1, y1 = r[:-1, 0], r[:-1, 1]
        x2, y2 = r[1:, 0], r[1:, 1]
        straddles = (y1 > y) != (y2 > y)
        if not straddles.any():
            continue
        x1, y1, x2, y2 = x1[straddles], y1[straddles], x2[straddles], y2[straddles]
        xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        if int((x < xint).sum()) % 2:
            inside = not inside
    return inside


def _dist_to_edge_km(x, y, ring_list):
    """Shortest distance from a point to any ring segment, in km (local equirectangular)."""
    kx = 111.32 * np.cos(np.radians(y))
    best = np.inf
    for r in ring_list:
        px = (r[:, 0] - x) * kx
        py = (r[:, 1] - y) * 110.57
        ax, ay = px[:-1], py[:-1]
        bx, by = px[1:], py[1:]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(-(ax * dx + ay * dy) / np.where(L2 == 0, 1.0, L2), 0.0, 1.0)
        d = np.hypot(ax + t * dx, ay + t * dy)
        if d.size:
            best = min(best, float(d.min()))
    return best


def assign(lats, lons, shp_path, dbf_path=None, field="GLG", fallback=True):
    """
    Assign a GLG unit to each (lat, lon).

    fallback=True replaces a lake/H2O/U hit, or a miss, with the nearest genuine geological polygon
    and records how far away it was, so no well is silently dropped. Returns a list of dicts.
    """
    dbf_path = dbf_path or shp_path[:-4] + ".dbf"
    attrs = read_dbf(dbf_path)
    bboxes, rings = read_polygons(shp_path)
    if len(attrs) != len(rings):
        raise ValueError(f"{len(attrs)} attribute rows against {len(rings)} shapes")
    codes = [str(a.get(field, "")).strip() for a in attrs]
    is_geo = np.array([c not in NON_GEOLOGY and c != "" for c in codes])

    out = []
    for lat, lon in zip(np.asarray(lats, float), np.asarray(lons, float)):
        cand = np.where((bboxes[:, 0] <= lon) & (bboxes[:, 2] >= lon) &
                        (bboxes[:, 1] <= lat) & (bboxes[:, 3] >= lat))[0]
        hits = [i for i in cand if _in_rings(lon, lat, rings[i])]
        # smallest-area hit wins where polygons nest
        direct = None
        if hits:
            areas = [(bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1]) for i in hits]
            direct = hits[int(np.argmin(areas))]

        rec = dict(glg_raw=codes[direct] if direct is not None else "",
                   glg=None, name=None, group=None, method=None,
                   dist_to_edge_km=np.nan, fallback_km=0.0)

        if direct is not None and is_geo[direct]:
            rec.update(glg=codes[direct], method="inside",
                       dist_to_edge_km=round(_dist_to_edge_km(lon, lat, rings[direct]), 3))
        elif fallback:
            # nearest genuine geology polygon, searched over a widening window
            best_i, best_d = None, np.inf
            for pad in (0.05, 0.15, 0.4, 1.0, 3.0):
                near = np.where(is_geo &
                                (bboxes[:, 0] - pad <= lon) & (bboxes[:, 2] + pad >= lon) &
                                (bboxes[:, 1] - pad <= lat) & (bboxes[:, 3] + pad >= lat))[0]
                for i in near:
                    d = _dist_to_edge_km(lon, lat, rings[i])
                    if d < best_d:
                        best_i, best_d = i, d
                if best_i is not None:
                    break
            if best_i is not None:
                rec.update(glg=codes[best_i], method="nearest",
                           fallback_km=round(best_d, 3), dist_to_edge_km=round(best_d, 3))
        if rec["glg"]:
            rec["name"] = GLG_NAME.get(rec["glg"], rec["glg"])
            rec["group"] = GLG_GROUP.get(rec["glg"], "unknown")
            rec["geo_rank"] = GLG_AGE.get(rec["glg"], np.nan)
        else:
            rec["geo_rank"] = np.nan
        out.append(rec)
    return out


GROUP_ORDER = ["Holocene coastal/deltaic", "Holocene alluvial", "Alluvial fan",
               "Residual (Pleistocene terrace)", "Bedrock"]
GROUP_CODE = {g: i for i, g in enumerate(GROUP_ORDER)}
