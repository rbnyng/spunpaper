# Building the input features from scratch

This directory exists so the code itself is for reference to someone who wants to rebuild the features from primary sources, extend them to new samples, or change a model. Nothing here is needed to reproduce the results.

## What produces what

| Script | Produces | Needs |
|---|---|---|
| `geotessera_embeddings.py` | `{sample_id}.npy` Tessera patch cache (`REPRESENTATIONS_DIR`) | `geotessera` and a lot of disk space |
| `prepare_fetch.py` | Shards the samples by tile and prints parallel fetch commands | as above |
| `crop_cache.py` | A narrower patch cache from a wider one | an existing cache |
| `generate_soil_features.py` | `SOIL_CACHE_DIR` | SoilGrids WCS access |
| `generate_worldcover_features.py` | `WORLDCOVER_CSV` | ESA WorldCover tiles |
| `generate_spectral_features.py` | Spectral-index baseline patches | raw Sentinel-1/2 stacks |
| `acquisition/` | Raw Sentinel-1/2 stacks | Planetary Computer STAC |

## Rebuilding the embedding cache

```bash
python data_generation/prepare_fetch.py --workers 4 --year 2024 --patch-size 7
# run the commands it prints, then
python data_generation/crop_cache.py --src <p7 cache> --dst <p3 cache> --size 3
```
