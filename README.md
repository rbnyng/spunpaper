# spunpaper

The repository covers the full pipeline from raw Sentinel-1/Sentinel-2 patch processing through embedding generation, model training, ablation studies and downstream mapping / inference.

## Repository layout

- `patches/` — Pipeline that takes biodiversity sample locations (latitude / longitude from the globalfungi dataset), downloads and processes the corresponding Sentinel-1 and Sentinel-2 image stacks, and runs inference with the pretrained Tessera foundation model to produce per-sample embeddings. The embedding model itself is not contained in this repository as it is hosted at [ucam-eo/tessera](https://github.com/ucam-eo/tessera). The `patches/` folder is the data-preparation and inference wrapper around that model. The pretrained weights checkpoint is loaded via `--checkpoint_path` when invoking the inference scripts.
- `modeling/` — Downstream modelling code. Loads precomputed embeddings, fetches WorldClim, SoilGrids and WorldCover covariates, trains Random Forest / LightGBM / XGBoost regressors of mycorrhizal richness, runs ablations, and produces the figures and 10m maps used in the paper.

## 1. System requirements

### Operating system

- Tested on Linux (Ubuntu 24.02). Any modern Linux with a working CUDA driver should work. The shell pipeline scripts assume `bash` and standard GNU userland. the SLURM batch scripts under `patches/` assume a SLURM cluster but the underlying Python entry points should run fine standalone.

### Hardware

- Embedding inference (`patches/`) requires an NVIDIA GPU with CUDA support
  (tested on a Tesla T4). CPU-only inference is also supported via `--mode cpu` in `patches/src/multi_tile_infer.py` and is the default for the hybrid runner `patches/infer_all_tiles.sh` when `CPU_GPU_SPLIT` includes a CPU share, but it is much slower.
- Downstream modelling (`modeling/`) runs on CPU. the 10m mapping scripts are I/O- and memory-heavy and were run on a cluster with 1.5 TB RAM. the core training and evaluation pipeline (sections 3 and 4.3) should run on a standard desktop with 32 GB RAM. the 1.5 TB figure applies only to wall-to-wall 10m inference over entire national parks, which is not required to reproduce the paper's main results.
- No other non-standard hardware is required.

### Software dependencies

- Python >= 3.10 (tested on 3.10)
- PyTorch >= 2.1 with a CUDA build matching your driver (only needed for the `patches/` inference pipeline)
- Core scientific stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `joblib`
- Gradient-boosting models: `xgboost`, `lightgbm`
- Dimensionality reduction: `umap-learn`
- Geospatial: `rasterio`, `geopandas`, `shapely`, `pyproj`, `richdem`, `contextily`, `scikit-image`, `pystac-client`, `dask`, `owslib`
- Misc: `requests`, `psutil`, `setproctitle`

The two binaries shipped in `patches/` (`s1_stack` and `s2_process_tile_downstream_wo_json`) are Linux executables provided by the Tessera team used to stack Sentinel-1 and Sentinel-2 raw tiles into the per-patch arrays that the Tessera model consumes. Sample invocations are in `patches/command_cheat_sheet.txt`.

A minimal install of the Python dependencies:

```bash
conda create -n spunpaper python=3.10 -y
conda activate spunpaper
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm joblib \
            xgboost lightgbm umap-learn \
            rasterio geopandas shapely pyproj richdem contextily scikit-image \
            pystac-client dask owslib requests psutil setproctitle
```

## 2. Installation guide

```bash
git clone https://github.com/rbnyng/spunpaper.git
cd spunpaper
# (create and activate a Python environment as above)
```

There is no compilation step for the Python code itself. The shipped `s1_stack` and `s2_process_tile_downstream_wo_json` binaries are ready-to-run on Linux x86_64.

To use the embedding pipeline you also need the pretrained Tessera weights. Clone the model repository and follow its instructions to obtain a checkpoint:

```bash
git clone https://github.com/ucam-eo/tessera.git
```

**Typical install time** on a normal desktop with a reasonable internet connection might be ~5–15 minutes for the Python environment. Cloning this repository itself should only take a couple of seconds.

## 3. Demo

### Bundled sample data

`sample_data/` contains a small slice of the data sufficient to test the downstream pipeline end-to-end:

- `sample_data/ECM_richness_europe.csv` — European ECM richness table (`sample_id`, `latitude`, `longitude`, `continent`, `raw_obs_div`, `rarefied`).
- `sample_data/representations/` — 100 Tessera per-sample embedding patches (shape `3 x 3 x 128`, `float32`), one `.npy` per sample, with filenames of the form `2021_<sample_id>_rarefied.npy`.

### Quick-start demo

```bash
python demo/run_demo.py
```

`demo/run_demo.py` filters the CSV down to the 100 samples that have matching embeddings and runs `modeling/spun_train_patch.py` on them with the satellite embeddings as the only feature source (`--use-satellite --no-use-climate --no-use-soil --no-use-worldcover`, `--model rf --num_runs 3 --dim_reduction pca --dim_reduction_components 16`). It is intended as a pipeline test. No network access is required, no GPU, and no covariate caches.
With only 100 samples and embeddings-only features the demo is designed to verify that the loader, dimensionality reduction, Random Forest training, metric reporting, and plotting code all run on a user's installation.

### Expected output

The demo writes a timestamped run directory under `demo/demo_run/results/rf_sat(pca)_<timestamp>/` containing:

- `per_run_performance_metrics.{csv,json}` — per-seed regression metrics across the 3 random seeds.
- `evaluation_summary.json` — aggregated mean / std of R2, MAE, RMSE, median AE.
- `feature_importance_*.png/.csv` — RF feature-importance plots and underlying data.
- `error_analysis_*.png/.csv` — predicted-vs-actual scatter, residual plots, and per-sample error CSV.

For reference, on a recent run the summary printed values like:

```
num_samples: 100
num_features: 18
r2_mean: -0.42 (std 0.08)
mae_mean: 5.37
medae_mean: 3.97
```

Note that demo results (R2 ~ -0.4) are not representative of manuscript performance. With only 100 samples, the model cannot learn meaningful patterns. See paper for full results on ~12,000 samples.

The demo completes in **~5–10 seconds** on a normal desktop (single-digit-second training over 3 seeds plus a few seconds of plot rendering).

## 4. Instructions for use

### 4.1 Generating embeddings on your own data (`patches/`)

The `patches/` pipeline turns raw biodiversity sample locations (CSV with `sample_id`, `latitude`, `longitude`) into per-sample Tessera embeddings.

1. **Rasterise sample locations into small per-patch TIFFs** — one TIFF per sample, used as the spatial AOI for satellite download:

   ```bash
   python patches/spun_to_tiff_patches.py
   ```

   (Edit the `CSV_FILE`, `OUTPUT_DIR`, `PATCH_SIZE` constants at the top of the script for your inputs.)

2. **Download and stack Sentinel-1 / Sentinel-2 imagery** for each patch. This is wrapped in the shell scripts and example calls are in `patches/command_cheat_sheet.txt`:

   ```bash
   bash patches/process_s1_patches.sh --input_dir <patches_dir> --output_dir <out>
   bash patches/process_s2_patches.sh --input_dir <patches_dir> --output_dir <out>
   ```

   These produce a `data_processed/` directory per patch containing the stacked S1 and S2 arrays expected by the embedding model. The `s1_stack` / `s2_process_tile_downstream_wo_json` binaries handle the actual stacking step.

3. **Run Tessera inference** to produce embeddings:

   ```bash
   bash patches/infer_all_tiles.sh
   ```

   or call the Python entry point directly:

   ```bash
   python patches/src/multi_tile_infer.py \
     --config configs/multi_tile_infer_config.py \
     --mode gpu --gpu_id 0 \
     --tile_list path/to/tile_list.txt \
     --checkpoint_path /path/to/tessera_checkpoint.pt \
     --output_dir /path/to/ecm_representations
   ```

   The `--checkpoint_path` argument should point at a Tessera pretrained checkpoint obtained from [ucam-eo/tessera](https://github.com/ucam-eo/tessera).The model definition consumed by `multi_tile_infer.py` lives in `patches/src/models/` and mirrors the architecture in that upstream repository so that the released weights load cleanly.

### 4.2 Building covariate caches (`modeling/`)

The downstream models in `spun_train_patch.py` consume three precomputed covariate sources alongside the satellite embeddings. Each is generated by a script in `modeling/` and only needs to be run once per sample set; the outputs are cached and reused on subsequent runs. Before running any of these, open the script and edit the path constants at the top (`BIODIVERSITY_CSV_PATHS`, output / cache directories, etc.) to match your environment.

#### WorldClim (climate rasters)

`modeling/worldclim_download.sh` downloads the WorldClim v2.1 30-arc-second global rasters (`tmin`, `tmax`, `tavg`, `prec`, `srad`, `wind`, `vapr`, the 19 BIOCLIM variables and elevation) from `geodata.ucdavis.edu` and unzips them into a `data/` directory. Point `--climate_data_dir` at this directory when running `spun_train_patch.py`; the first time the trainer is invoked it will extract per-sample values and write them to `--climate_cache_dir`. Requires `curl` and `unzip`; expect ~10–20 GB of raster data and a download time dominated by your network.

```bash
cd modeling
bash worldclim_download.sh   # writes rasters under ./data
```

#### SoilGrids (soil chemistry / physics)

`modeling/generate_soil_features.py` queries the ISRIC SoilGrids 2.0 Web Coverage Service for each sample point and extracts pH, soil organic carbon, nitrogen, clay/silt/sand fractions, CEC, bulk density, field-capacity / wilting-point water content and the WRB soil class, at the 0–5 cm and 5–15 cm depth intervals. Results are written as a per-sample CSV inside `SOIL_CACHE_DIR`; point `--soil_cache_dir` at the same directory when training. Requires `owslib` and outbound HTTPS to `maps.isric.org`. Runtime scales linearly with the number of samples and is dominated by WCS round-trips (~hours for tens of thousands of samples; `MAX_WORKERS` in the script controls concurrency).

```bash
python modeling/generate_soil_features.py
```

#### ESA WorldCover (land-cover class)

`modeling/generate_worldcover_features.py` downloads the relevant ESA WorldCover 2021 v200 (or 2020 v100) 10 m GeoTIFF tiles from the public S3 bucket, looks up the land-cover class at each sample point, and writes a single tidy CSV mapping `sample_id` → WorldCover class. Pass that CSV to `spun_train_patch.py` via `--worldcover_path`. Requires outbound HTTPS to `esa-worldcover.s3.eu-central-1.amazonaws.com`; downloaded tiles are cached in `WORLDCOVER_TILES_DIR` so subsequent runs are fast.

```bash
python modeling/generate_worldcover_features.py
```

After the three commands above complete, the trainer in §4.3 can be run with `--use-satellite --use-climate --use-soil --use-worldcover` to use all four feature sources.

### 4.3 Training and evaluating richness models

The main entry point is `modeling/spun_train_patch.py`. See `--help` for the full flag set. the most consequential options are:

- `--model {rf,lightgbm,xgboost}` — choice of regressor
- `--num_runs N` — number of random-seed repetitions (50 was used in the paper)
- `--use-satellite/--no-use-satellite`, `--use-climate/--no-use-climate`, `--use-soil/--no-use-soil`, `--use-worldcover/--no-use-worldcover` — toggle each feature source
- `--dim_reduction {none,pca,umap}`, `--dim_reduction_components K` — dimensionality reduction applied to satellite embeddings
- `--use-biome-filter` plus `--biome-iqr-multiplier` — biome-conditioned outlier removal using RESOLVE Ecoregions 2017

### 4.4 Reproducing the paper

Once the trainer is wired up, switch to the real data and turn on the other feature sources. Build the covariate caches first (see section above), then:

```bash
cd modeling
python spun_train_patch.py \
  --model rf \
  --num_runs 50 \
  --biodiversity_csvs /path/to/ECM_richness_europe.csv /path/to/ECM_richness_Asia.csv \
  --representations_dir /path/to/ecm_representations \
  --climate_data_dir /path/to/worldclim/data \
  --climate_cache_dir /path/to/climate_features_cache \
  --soil_cache_dir /path/to/soil_features_cache \
  --worldcover_path /path/to/worldcover_features.csv \
  --results_dir paper_results
```

This produces one set of the results seen in Table 1 of the paper. On a cluster machine (256-core CPU, 1.5 TB RAM) one full 50-seed run takes a few hours for one model. the bulk of the time is per-seed training.

For individual results from the paper:

- **Ablation grid** (all models x all single-source / all-source feature combinations):

  ```bash
  cd modeling
  python run_ablations.py
  ```

  This calls `spun_train_patch.py` once per cell of the grid, logging each run to its own `ablation_log_*.txt` file.

- **Aggregate results**:

  ```bash
  python summarize_results.py
  python plot_errors.py
  python plot_feature_importance.py
  python plot_kde_hist.py
  python boxplot_error.py
  ```

- **Figure 1 and embedding-correlation analyses**:

  ```bash
  python fig1_plots.py
  python embedding_correlations.py
  ```

- **Wall-to-wall 10 m mapping** of predicted richness over a region of
  interest, and the masked variant that respects WorldCover land-cover
  classes:

  ```bash
  python 10m_map.py
  python 10m_map_masked.py
  ```

- **Time-series difference analyses** and **profiling of tile inference**:

  ```bash
  python time_series_diff.py
  python time_series_diff2.py
  python profile_tile_inference.py
  ```

- **Downstream application — UK national parks inference**:

  ```bash
  python uk_national_parks_inference.py --workers 8
  ```

## License

MIT License

## Code availability

The source repository is [https://github.com/rbnyng/spunpaper](https://github.com/rbnyng/spunpaper).

The pretrained foundation model used to generate embeddings is hosted separately at [https://github.com/ucam-eo/tessera](https://github.com/ucam-eo/tessera).
