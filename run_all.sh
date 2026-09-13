#!/usr/bin/env bash
#
# codebase helper shell script
#
# stages 1-8 produce the main body results. stages 9-16 produce the SI analyses. 
#
# Paths come from config.env at the repository root. Override by editing that
# file or by exporting the variables:
#
#     SPUN_DATA_ROOT=/my/data SPUN_SCRATCH=/my/scratch ./run_all.sh
#
# Usage:
#     ./run_all.sh                  # every stage skipping completed ones
#     ./run_all.sh --list           # list stages and exit
#     ./run_all.sh --check          # check what can be run with current data
#     ./run_all.sh --dry-run        # print the commands
#     ./run_all.sh --group main     # main body only
#     ./run_all.sh --group si       # si only
#     ./run_all.sh --from 3         # resume from stage 3
#     ./run_all.sh --only 12        # run one stage
#     ./run_all.sh --force          # force rerun
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

# configuration 
SPUN_ROOT="${SPUN_ROOT:-$(cd "$HERE/.." && pwd)}"
export SPUN_ROOT
_SPUN_CFG="${SPUN_CONFIG:-$HERE/config.env}"
if [ -f "$_SPUN_CFG" ]; then
  # shellcheck disable=SC1090
  . "$_SPUN_CFG"
else
  echo "WARNING: $_SPUN_CFG not found; relying on environment variables only." >&2
fi
PY="${PY:-python3}"
YEAR="${YEAR:-2024}"
PATCH_SIZE="${PATCH_SIZE:-3}"
NUM_RUNS="${NUM_RUNS:-50}"
DIM_COMPONENTS="${DIM_COMPONENTS:-256}"
SATELLITE_SOURCE="${SATELLITE_SOURCE:-}"
IMPORTANCE_DIR="${IMPORTANCE_DIR:-}"
LOG_DIR="${LOG_DIR:-$HERE/logs}"
FIG_DIR="${FIG_DIR:-$HERE/figures}"
if [ -n "${SI_DIR:-}" ]; then
  :
elif [ -n "${SPUN_RESULTS_ROOT:-}" ]; then
  SI_DIR="$SPUN_RESULTS_ROOT/si"
else
  SI_DIR="$HERE/results"
fi

DRY_RUN=0; FORCE=0; FROM=1; ONLY=""; LIST=""; GROUP="all"; CHECK_ONLY=0

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }  # line range: keep the header block above in step

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE=1 ;;
    --from)    FROM="$2"; shift ;;
    --only)    ONLY="$2"; shift ;;
    --group)   GROUP="$2"; shift ;;
    --list)    LIST=1 ;;
    --check)   CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

STAGES=(
  "1|main|Input features - use cache, or build data (data_generation/)"
  "2|main|Table 1 - model x feature-set ablations (15 configurations)"
  "3|main|Table 1 - aggregation and paired significance tests"
  "4|main|Figure 2 - performance and feature importance"
  "5|main|Figure 4 + Figure S1 - error maps and R2 distributions"
  "6|main|Figure 3 - 10 m prediction map (Alpine ROI)"
  "7|main|Figure 1 - overview panels"
  "8|main|Figure 5 - UK National Parks"
  "9|si|SI - spatially blocked cross-validation"
  "10|si|SI - sensitivity to the year of satellite imagery"
  "11|si|SI - patch size ablation"
  "12|si|SI - spectral-index baseline comparison"
  "13|si|SI - cross-continental transfer"
  "14|si|SI - prediction intervals"
  "15|si|SI - error structure and high-error samples"
  "16|si|SI - embedding PCA structure and environment correlations"
)

if [ -n "$LIST" ]; then
  printf '%s\n' "${STAGES[@]}" | awk -F'|' '{printf "  %-3s %-7s %s\n", $1, "["$2"]", $3}'
  exit 0
fi

mkdir -p "$LOG_DIR"
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
warn() { printf '   \033[33m%s\033[0m\n' "$*" >&2; }

stage_group() { printf '%s\n' "${STAGES[@]}" | awk -F'|' -v n="$1" '$1==n {print $2}'; }

selected() {                        # selected <n> -> in scope?
  local n="$1"
  if [ -n "$ONLY" ]; then [ "$ONLY" = "$n" ]; return; fi
  [ "$n" -ge "$FROM" ] || return 1
  [ "$GROUP" = "all" ] && return 0
  if [ "$GROUP" = "figures" ]; then
    case "$n" in 3|4|5) return 0 ;; *) return 1 ;; esac
  fi
  [ "$(stage_group "$n")" = "$GROUP" ]
}

SKIPPED_STAGES=()

run_stage() {                       # run_stage <n> <desc> <command...>
  local n="$1" desc="$2"; shift 2
  selected "$n" || return 0
  say "stage $n: $desc"
  note "\$ $*"
  [ "$DRY_RUN" -eq 1 ] && return 0
  local log="$LOG_DIR/stage_${n}.log"
  if "$@" >"$log" 2>&1; then
    note "ok  (log: $log)"
  else
    echo "   FAILED (exit $?). Last lines of $log:" >&2
    tail -n 15 "$log" >&2
    exit 1
  fi
}

skip_stage() {                      # skip_stage <n> <desc> <why>
  local n="$1" desc="$2" why="$3"
  selected "$n" || return 0
  say "stage $n: $desc"
  warn "SKIPPED: $why"
  SKIPPED_STAGES+=("$n")
}

nonempty_dir() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

say "preflight"
note "config      : $_SPUN_CFG"
note "python      : $($PY --version 2>&1)"
note "year        : $YEAR    patch size: ${PATCH_SIZE}x${PATCH_SIZE}    runs: $NUM_RUNS    components: $DIM_COMPONENTS"
note "results     : ${RESULTS_DIR:-<unset>}"
note "SI output   : $SI_DIR"
note "figures     : $FIG_DIR"

# Refuse to write supplementary output on top of an unpacked results_bundle.
# That archive ships MANIFEST.md at its root and holds the numbers reported in
# the manuscript; a sample or partial run landing in the same directory would
# overwrite them with no warning and no way to tell afterwards.
if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ] \
   && [ -f "$SI_DIR/MANIFEST.md" ] && [ "$FORCE" -eq 0 ]; then
  echo "" >&2
  echo "   REFUSING to write into $SI_DIR" >&2
  echo "   It contains MANIFEST.md, so it looks like an unpacked results_bundle." >&2
  echo "   Writing there would overwrite the deposited manuscript results." >&2
  echo "" >&2
  echo "   Send this run somewhere else:" >&2
  echo "     SI_DIR=\$PWD/si_rerun $0 $*" >&2
  echo "   or pass --force if overwriting the deposit is what you intend." >&2
  exit 3
fi

deposit_present() {
  nonempty_dir "${REPRESENTATIONS_DIR:-/nonexistent}" \
    && nonempty_dir "${SOIL_CACHE_DIR:-/nonexistent}" \
    && [ -f "${WORLDCOVER_CSV:-/nonexistent}" ]
}

missing=0
for f in "${ECM_EUROPE_CSV:-}" "${ECM_ASIA_CSV:-}"; do
  if [ -z "$f" ] || [ ! -f "$f" ]; then
    echo "   MISSING ground truth CSV: ${f:-<unset>}" >&2; missing=1
  fi
done

if [ ! -d "${WORLDCLIM_DIR:-/nonexistent}" ]; then
  if nonempty_dir "${CLIMATE_CACHE_DIR:-/nonexistent}"; then
    note "WorldClim absent, but the climate cache is present - fine"
  else
    echo "   MISSING WorldClim, and no climate cache to fall back on: ${WORLDCLIM_DIR:-<unset>}" >&2
    missing=1
  fi
fi

if [ "$missing" -eq 1 ]; then
  echo "
Required inputs are missing. They are all public:
  GlobalFungi richness CSVs   -> ECM_EUROPE_CSV / ECM_ASIA_CSV
  WorldClim 2.1               -> WORLDCLIM_DIR   (only to build the climate cache)
  SoilGrids 2.0, ESA WorldCover are fetched by stage 1.
Set the paths in $_SPUN_CFG and re-run." >&2
  results_only=0
  [ "$GROUP" = "figures" ] && results_only=1
  case "$ONLY" in 3|4|5) results_only=1 ;; esac
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ] && [ "$results_only" -eq 0 ]; then
    exit 1
  fi
  [ "$results_only" -eq 1 ] && note "ignored: the selected stages read RESULTS_DIR only"
fi

YEAR_CACHES_OK=1
for y in 2022 2023 2024; do
  d="${YEAR_CACHE_TEMPLATE:-/nonexistent}"; d="${d//\{year\}/$y}"
  nonempty_dir "$d" || YEAR_CACHES_OK=0
done
nonempty_dir "${REPRESENTATIONS_P7_DIR:-/nonexistent}" && P7_OK=1 || P7_OK=0
nonempty_dir "${SPECTRAL_REPRESENTATIONS_DIR:-/nonexistent}" && SPECTRAL_OK=1 || SPECTRAL_OK=0
[ -d "${SPUN_NATIONALPARKS:-/nonexistent}" ] && PARKS_OK=1 || PARKS_OK=0

command -v geotessera >/dev/null 2>&1 && GEOTESSERA_OK=1 || GEOTESSERA_OK=0

FIG5_OK=1
for s in retrain_with_pca.py time_series_diff2.py; do
  [ -f "$SRC/$s" ] || FIG5_OK=0
done

if [ -z "$SATELLITE_SOURCE" ]; then
  if deposit_present; then SATELLITE_SOURCE=precomputed; else SATELLITE_SOURCE=geotessera; fi
fi
export SATELLITE_SOURCE

note "deposit     : $(deposit_present && echo present || echo 'absent - stage 1 will build')"
note "sat source  : $SATELLITE_SOURCE"
note "p7 cache    : $([ "$P7_OK" = 1 ] && echo present || echo 'absent - stage 11 skipped')"
note "spectral    : $([ "$SPECTRAL_OK" = 1 ] && echo present || echo 'absent - stage 12 skipped')"
note "year caches : $([ "$YEAR_CACHES_OK" = 1 ] && echo present || echo 'absent - stage 10 skipped')"
note "parks data  : $([ "$PARKS_OK" = 1 ] && echo present || echo 'absent - stage 8 skipped')"
note "fig5 scripts: $([ "$FIG5_OK" = 1 ] && echo present || echo 'absent - stage 8 skipped')"
note "geotessera  : $([ "$GEOTESSERA_OK" = 1 ] && echo present || echo 'absent - stages 6,7 skipped')"

$PY "$SRC/paths.py" | sed 's/^/   /'

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "check complete (no stages run)"
  exit 0
fi

mkdir -p "$FIG_DIR" "$SI_DIR"
CSVS=("${ECM_EUROPE_CSV:-}" "${ECM_ASIA_CSV:-}")

# =============================================================================
#  MAIN BODY
# =============================================================================
if selected 1; then
  say "stage 1: input features"
  if deposit_present && [ "$FORCE" -eq 0 ]; then
    note "caches present - nothing to build"
    note "  representations : $REPRESENTATIONS_DIR"
    note "  soil / landcover: $SOIL_CACHE_DIR, $WORLDCOVER_CSV"
  elif [ "$DRY_RUN" -eq 1 ]; then
    note "would use the deposit, or build from data_generation/"
  else
    warn "no deposit found - building the caches from scratch"
    warn "this needs geotessera (Python >= 3.12) and streams ~190 GB of tiles"
    run_stage 1 "Fetch Tessera embeddings (geotessera)" \
      $PY "$HERE/data_generation/geotessera_embeddings.py" \
        --coords-csv "${CSVS[@]}" \
        --output-dir "$REPRESENTATIONS_DIR" \
        --year "$YEAR" --patch-size "$PATCH_SIZE" --prune-tiles
    run_stage 1 "Build environmental baseline features" \
      bash -c "$PY '$HERE/data_generation/generate_soil_features.py' && \
               $PY '$HERE/data_generation/generate_worldcover_features.py'"
  fi
fi

run_stage 2 "Table 1 - model x feature-set ablations" \
  $PY "$SRC/run_ablations.py" --num-runs "$NUM_RUNS" --results-dir "$RESULTS_DIR" \
    --dim-reduction-components "$DIM_COMPONENTS" --satellite-source "$SATELLITE_SOURCE"

run_stage 3 "Table 1 - aggregation and paired tests" \
  bash -c "$PY '$SRC/summarize_results.py' && \
           $PY '$SRC/analyze_table1_runs.py' \
              --runs-dir '$RESULTS_DIR' --out-dir '$SI_DIR/table1_paired'"

run_stage 4 "Figure 2 - performance and feature importance" \
  $PY "$SRC/plot_fig2.py" --runs-dir "$RESULTS_DIR" --out "$FIG_DIR/table1_final" \
    ${IMPORTANCE_DIR:+--importance-dir "$IMPORTANCE_DIR"}

run_stage 5 "Figure 4 + Figure S1 - error maps and R2 distributions" \
  bash -c "$PY '$SRC/plot_errors.py' --out-dir '$FIG_DIR' && \
           $PY '$SRC/plot_kde.py'    --out-dir '$FIG_DIR'"

if [ "$GEOTESSERA_OK" = 0 ]; then
  skip_stage 6 "Figure 3 - 10 m prediction map" \
    "needs the geotessera CLI to fetch ROI tiles (pip install geotessera, Python >= 3.12)"
else
  run_stage 6 "Figure 3 - 10 m prediction map" \
    $PY "$SRC/10m_map_masked.py"
fi

if [ "$GEOTESSERA_OK" = 0 ]; then
  skip_stage 7 "Figure 1 - overview panels" \
    "needs the geotessera CLI to fetch ROI tiles (pip install geotessera, Python >= 3.12)"
else
  run_stage 7 "Figure 1 - overview panels" \
    $PY "$SRC/fig1_plots.py"
fi

# Needs a PCA-reduced model (see the note in src/uk_national_parks_inference.py)
# and the temporal comparison plots.
if [ "$FIG5_OK" = 0 ]; then
  skip_stage 8 "Figure 5 - UK National Parks temporal analysis" \
    "src/retrain_with_pca.py and/or src/time_series_diff2.py are not in this checkout"
elif [ "$PARKS_OK" = 0 ]; then
  skip_stage 8 "Figure 5 - UK National Parks temporal analysis" \
    "SPUN_NATIONALPARKS not found: ${SPUN_NATIONALPARKS:-<unset>}"
else
  run_stage 8 "Figure 5 - UK National Parks temporal analysis" \
    bash -c "$PY '$SRC/retrain_with_pca.py' && \
             $PY '$SRC/uk_national_parks_inference.py' && \
             $PY '$SRC/time_series_diff2.py'"
fi

# =============================================================================
#  SI
# =============================================================================

run_stage 9 "SI - spatially blocked cross-validation" \
  $PY "$SRC/run_blocked_cv.py" \
    --biodiversity-csvs "${CSVS[@]}" \
    --representations-dir "$REPRESENTATIONS_DIR" \
    --satellite-source "$SATELLITE_SOURCE" \
    --dim-reduction-components "$DIM_COMPONENTS" \
    --out-dir "$SI_DIR/spatial_cv"

if [ "$YEAR_CACHES_OK" = 0 ]; then
  skip_stage 10 "SI - sensitivity to the year of satellite imagery" \
    "one 3x3 cache per year is required; set YEAR_CACHE_TEMPLATE (currently ${YEAR_CACHE_TEMPLATE:-<unset>})"
else
  run_stage 10 "SI - sensitivity to the year of satellite imagery" \
    $PY "$SRC/run_year_ablation.py" \
      --samples-csv "${CSVS[0]}" \
      --cache-template "$YEAR_CACHE_TEMPLATE" \
      --years 2022 2023 2024 \
      --n-components "$DIM_COMPONENTS" \
      --out-dir "$SI_DIR/year_ablation"
fi

if [ "$P7_OK" = 0 ]; then
  skip_stage 11 "SI - patch size ablation" \
    "needs the 7x7 cache; set REPRESENTATIONS_P7_DIR (currently ${REPRESENTATIONS_P7_DIR:-<unset>})"
else
  run_stage 11 "SI - patch size ablation" \
    $PY "$SRC/run_patch_size_ablation.py" \
      --cache-dir "$REPRESENTATIONS_P7_DIR" \
      --native-patch-size 7 \
      --n-components "$DIM_COMPONENTS" \
      --biodiversity-csvs "${CSVS[@]}" \
      --out-dir "$SI_DIR/patch_size"
fi

if [ "$SPECTRAL_OK" = 0 ]; then
  skip_stage 12 "SI - spectral-index baseline comparison" \
    "needs the spectral feature cache; set SPECTRAL_REPRESENTATIONS_DIR (currently ${SPECTRAL_REPRESENTATIONS_DIR:-<unset>})"
else
  run_stage 12 "SI - spectral-index baseline comparison" \
    bash -c "$PY '$SRC/evaluate_feature_set.py' \
                --biodiversity-csvs '${CSVS[0]}' '${CSVS[1]}' \
                --representations-dir '$REPRESENTATIONS_DIR' \
                --satellite-source '$SATELLITE_SOURCE' \
                --out-dir '$SI_DIR/feature_set_tessera' && \
             $PY '$SRC/evaluate_feature_set.py' \
                --biodiversity-csvs '${CSVS[0]}' '${CSVS[1]}' \
                --representations-dir '$SPECTRAL_REPRESENTATIONS_DIR' \
                --satellite-source precomputed \
                --out-dir '$SI_DIR/feature_set_spectral'"
fi

run_stage 13 "SI - cross-continental transfer" \
  $PY "$SRC/run_continent_transfer.py" \
    --biodiversity-csvs "${CSVS[@]}" \
    --representations-dir "$REPRESENTATIONS_DIR" \
    --satellite-source "$SATELLITE_SOURCE" \
    --n-components "$DIM_COMPONENTS" \
    --out-dir "$SI_DIR/continent_transfer"

run_stage 14 "SI - prediction intervals" \
  $PY "$SRC/run_uncertainty.py" \
    --biodiversity-csvs "${CSVS[@]}" \
    --representations-dir "$REPRESENTATIONS_DIR" \
    --satellite-source "$SATELLITE_SOURCE" \
    --dim-reduction-components "$DIM_COMPONENTS" \
    --out-dir "$SI_DIR/prediction_intervals"

run_stage 15 "SI - error structure and high-error samples" \
  $PY "$SRC/run_error_outliers.py" \
    --biodiversity-csvs "${CSVS[@]}" \
    --representations-dir "$REPRESENTATIONS_DIR" \
    --satellite-source "$SATELLITE_SOURCE" \
    --dim-reduction-components "$DIM_COMPONENTS" \
    --out-dir "$SI_DIR/error_outliers"

run_stage 16 "SI - embedding PCA structure and environment correlations" \
  bash -c "$PY '$SRC/run_embedding_pca.py' \
              --biodiversity-csvs '${CSVS[0]}' '${CSVS[1]}' \
              --representations-dir '$REPRESENTATIONS_DIR' \
              --satellite-source '$SATELLITE_SOURCE' \
              --out-dir '$SI_DIR/pca_structure' && \
           $PY '$SRC/embedding_pca_correlations.py' \
              --output_dir '$SI_DIR/pca_structure'"

# =============================================================================
say "done"
note "results  : ${RESULTS_DIR:-<unset>}"
note "SI output: $SI_DIR"
note "figures  : $FIG_DIR"
note "logs     : $LOG_DIR"
if [ "${#SKIPPED_STAGES[@]}" -gt 0 ]; then
  warn "skipped stages: ${SKIPPED_STAGES[*]}  (see the messages above)"
fi