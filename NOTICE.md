# Third-party data and components

The MIT licence in `LICENSE` covers the code in this repository. It does not cover the data files
below. Each one keeps the licence of its source, and the attribution has to travel with it.

## Surface geology

`assets/geo8bg.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`

Digital coverage of the 1990 Geological Survey of Bangladesh 1:1,000,000 map, digitised and published
by the United States Geological Survey.

Persits, F.M., Wandrey, C.J., Milici, R.C., and Manwar, A. Digital geologic and geophysical data of
Bangladesh. USGS Open-File Report 97-470H. <https://doi.org/10.3133/ofr97470H>

A work of the US Government, in the public domain under 17 U.S.C. 105.

## Soil

`assets/soil_comilla.tif`

Four SoilGrids 2.0 bands over Cumilla at the 15 to 30 cm depth interval: organic carbon, clay, sand
and silt. Cropped to the study area and resampled onto the pixel grid the models were trained on.

ISRIC World Soil Information, SoilGrids 2.0. <https://soilgrids.org>

Licensed CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

## Administrative boundaries

`assets/local_region.geojson` and `bd_boundary.geojson`

The first holds eight upazila polygons from geoBoundaries gbOpen, Bangladesh ADM3. The second holds the
64 district polygons from geoBoundaries gbOpen, Bangladesh ADM2, carrying the attribution in its own
properties block. Both are 2020 boundaries, built from Bangladesh Bureau of Statistics and OCHA ROAP
data distributed through <https://data.humdata.org>.

geoBoundaries. <https://www.geoboundaries.org>

Licensed CC BY 3.0 IGO. <https://creativecommons.org/licenses/by/3.0/igo/>

Required attribution, to travel with either file and to be shown in anything built on them:

> Administrative boundaries: geoBoundaries (gbOpen), Bangladesh ADM2 and ADM3, 2020, CC BY 3.0 IGO,
> from Bangladesh Bureau of Statistics and OCHA ROAP via data.humdata.org

Citation:

> Runfola, D. et al. (2020) geoBoundaries: A global database of political administrative boundaries.
> PLoS ONE 15(4): e0231866.

The deployed container uses an FAO GAUL national outline in place of `bd_boundary.geojson`. GAUL may not
be redistributed without written consent from FAO, so it is not in this repository and the geoBoundaries
rebuild stands in for it. The two agree on every point tested.

## Map library

`static/leaflet/`

Leaflet 1.9.4. Copyright 2010 to 2023 Vladimir Agafonkin, copyright 2010 to 2011 CloudMade.
<https://leafletjs.com>

Licensed BSD 2-Clause. The licence header is preserved in `leaflet.js` and `leaflet.css`.

## Map tiles

Fetched by the browser at run time, not stored here. Raster tiles come from CARTO and the underlying
map data from OpenStreetMap contributors, licensed ODbL. The interface carries both attributions, as
their terms require. Point the tile URL in `static/index.html` somewhere else if you need a different
provider.

## Python packages

Installed from `requirements.txt` and each under its own licence. None is redistributed here.

## Models

`model_local_as10.joblib` and `model_national_as10.joblib` were fitted on well records from the
British Geological Survey and DPHE national survey and on a field dataset from Cumilla. Neither
dataset is redistributed in this repository, and neither is ours to pass on.

The fitted objects do carry aggregates over the training wells: the scaler means and scale factors,
the imputer statistics, and the split thresholds of the tree ensembles. They hold no well record. The
two LIME background files that do hold records are not in this repository, and `README.md` explains
what that changes.
