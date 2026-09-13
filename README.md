# Below-ground fungal biodiversity with tessera embeddings codebase

## Setup

```bash
pip install -r requirements.txt
# fetch and extract the feature caches, then point config.env at them
./run_all.sh --check                 # what data is present, and what can run
./run_all.sh                         # stage 1 is skipped when the deposit is present
```

## Configuration

Paths are in `config.env` at the repository root. Precedence is environment variable, then that file, then a fallback. Point either at a different file with `SPUN_CONFIG=/path/to/other.env`.

## Running it

```bash
./run_all.sh --check       # what data is present
./run_all.sh --list        # list of stages
./run_all.sh --dry-run     # print the commands
./run_all.sh               # run everything skipping completed stages
./run_all.sh --group main  # main body only
./run_all.sh --group si    # supplementary only
./run_all.sh --from 3      # resume
./run_all.sh --only 12     # one stage
```

`--check` reports which data caches are present and the stages that will be skipped without them.

```bash
python tests_synthetic.py      # end-to-end check on synthetic data which needs no inputs
```

## bundled sample

`reprod_sample.tar.xz` is a fixed-seed 5% draw of the sample IDs so the codebase can be run end to end easily to test the code implementation.

```bash
tar -xJf reprod_sample.tar.xz
SPUN_CONFIG=config.sample.env ./run_all.sh --check
SPUN_CONFIG=config.sample.env DIM_COMPONENTS=32 NUM_RUNS=5 ./run_all.sh --group si
```

it contains 592 samples. Stages 1–5 and 9–16 run but 6–8 skip, because figures 1, 3 and 5 need wall-to-wall raster tiles rather than the per-sample patches we provide. Numbers from the sample will not match the manuscript, and are not meant to since 5% is too few samples.

## regenerating the tables and figures

`results_bundle.tar.xz` holds per-seed metrics and analysis outputs reported in the manuscript. It unpacks to `results/`. Stages 3–5 read just `RESULTS_DIR`, so we can regenerate the results table, the feature-importance figure, the error maps and the R2 distributions from it:

```bash
tar -xJf results_bundle.tar.xz          # -> results/
RESULTS_DIR=results/table1_runs \
  IMPORTANCE_DIR=results/importance_2024 \
  SI_DIR=$PWD/si_rerun \
  ./run_all.sh --group figures
```

## script to result correspondence

| Manuscript item | Stage | Script |
|---|---|---|
| Figure 1 (overview panels) | 7 | `src/fig1_plots.py` partially, other parts are hand drawn |
| Figure 2 (performance + feature importance) | 4 | `src/plot_fig2.py` |
| Figure 3 (10m prediction map) | 6 | `src/10m_map_masked.py` |
| Figure 4 (spatial error map) | 5 | `src/plot_errors.py` |
| Figure 5 (UK National Parks) | 8 | `src/uk_national_parks_inference.py` |
| Figure S1 (outlier sensitivity) | 5 | `src/plot_kde.py` |
| Table S14 (full ablation grid), SI-S16 | 2 | `src/run_ablations.py` |
| Table S15 (paired comparisons), SI-S16 | 3 | `src/analyze_table1_runs.py` |
| Table S3–S5, SI-S8 (PCA structure) | 16 | `src/run_embedding_pca.py`, `src/embedding_pca_correlations.py` |
| Table S6, SI-S9 (spatially blocked CV) | 9 | `src/run_blocked_cv.py` |
| Table S7, SI-S10 (embedding year) | 10 | `src/run_year_ablation.py` |
| Table S8, SI-S11 (patch size) | 11 | `src/run_patch_size_ablation.py` |
| Table S9–S10, SI-S12 (spectral-index baseline) | 12 | `src/evaluate_feature_set.py` |
| Table S11, SI-S13 (cross-continental transfer) | 13 | `src/run_continent_transfer.py` |
| Table S12, SI-S14 (prediction intervals) | 14 | `src/run_uncertainty.py` |
| Table S13, SI-S15 (high-error characterization) | 15 | `src/run_error_outliers.py` |

## repo structure

| Path | Desc |
|---|---|
| `config.env` | filesystem paths for both shell and Python |
| `run_all.sh` | runner helper shell script |
| `tests_synthetic.py` | end-to-end check on synthetic data |
| `config.sample.env` | points the pipeline at the bundled 5% reproduction sample |
| `reprod_sample.tar.xz` | that sample. see `data_generation/make_reprod_sample.py` |
| `src/` | the codebase |
| `data_generation/` | everything that builds the feature cache, not needed for reproduction |