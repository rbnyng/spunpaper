#!/bin/bash -l

# Reprocess failed S2 patches
# Reads the fail log and reruns the failed files

usage() {
  cat <<EOF
Usage: bash $0 --input_dir <path> --output_dir <path> --fail_log <path> [options]

Required arguments:
  --input_dir       Directory containing {year}_{tile_id}_agbm.tif files
  --output_dir      Output root directory
  --fail_log        Path to the fail log file

Optional arguments:
  --max_parallel    Max parallel patches (default 24)
  --cores_per_patch CPU cores per patch (default 2)
  --dask_workers    Dask workers per patch (default 1)
  --worker_memory   Memory per worker in GB (default 4)
  --max_cloud       Max cloud cover percentage (default 90)
  --resolution      Output resolution in metres (default 10)
  --overwrite       Overwrite existing files
  --debug           Emit debug logs

Example:
bash reprocess_failed_s2_patches.sh \
  --input_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_masks_10m \
  --output_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_d-pixel \
  --fail_log /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_d-pixel/logs_s2/s2_processing_fail.log \
  --overwrite

EOF
  exit 1
}

# Default parameters
MAX_PARALLEL=24
CORES_PER_PATCH=1
DASK_WORKERS=1
WORKER_MEMORY=4
MAX_CLOUD=90
RESOLUTION=10
OVERWRITE=""
DEBUG=""

# Temporary directory
export TEMP_DIR="${TEMP_DIR:-${PATCH_PROC_TMP:-/tmp/spun_patch_proc}}"

# Processor script path
S2_PROCESSOR="./s2_fast_processor_small_patches.py"
# --- shared configuration -----------------------------------------------------
# Paths come from config.env at the repository root so they can be swapped without
# editing this script. Precedence: environment variable > config.env > default.
_SPUN_CFG="${SPUN_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config.env}"
# shellcheck disable=SC1090
[ -f "$_SPUN_CFG" ] && . "$_SPUN_CFG"
export PYTHON_ENV="${PYTHON_ENV:-${ACQ_PYTHON_ENV:-python3}}"
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir)      INPUT_DIR=$2; shift 2;;
    --output_dir)     OUTPUT_DIR=$2; shift 2;;
    --fail_log)       FAIL_LOG=$2; shift 2;;
    --max_parallel)   MAX_PARALLEL=$2; shift 2;;
    --cores_per_patch) CORES_PER_PATCH=$2; shift 2;;
    --dask_workers)   DASK_WORKERS=$2; shift 2;;
    --worker_memory)  WORKER_MEMORY=$2; shift 2;;
    --max_cloud)      MAX_CLOUD=$2; shift 2;;
    --resolution)     RESOLUTION=$2; shift 2;;
    --overwrite)      OVERWRITE="--overwrite"; shift 1;;
    --debug)          DEBUG="--debug"; shift 1;;
    -h|--help)        usage;;
    *)                echo "Unknown option: $1"; usage;;
  esac
done

[[ -z "${INPUT_DIR:-}" || -z "${OUTPUT_DIR:-}" || -z "${FAIL_LOG:-}" ]] && usage

# Logging helpers
log_error() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ ERROR: $1" >&2
}

log_info() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ℹ️  INFO: $1"
}

# Validate inputs
if [[ ! -d "$INPUT_DIR" ]]; then
  log_error "Input directory does not exist: $INPUT_DIR"
  exit 1
fi

if [[ ! -f "$FAIL_LOG" ]]; then
  log_error "Fail log file does not exist: $FAIL_LOG"
  exit 1
fi

if [[ ! -f "$S2_PROCESSOR" ]]; then
  log_error "S2 processor not found: $S2_PROCESSOR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

LOG_DIR="$OUTPUT_DIR/logs_s2"
mkdir -p "$LOG_DIR"

log_info "🔄 Reprocessing failed S2 patches"
log_info "Fail log file: $FAIL_LOG"

# Parse the fail log and extract the actual filenames
declare -a FAILED_PATCHES
while IFS= read -r line; do
  # Skip blank and comment lines
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  
  # Extract the filename: drop the s2_ prefix and the _P**** suffix
  if [[ $line =~ ^s2_(.+)_P[0-9]+$ ]]; then
    patch_name="${BASH_REMATCH[1]}"
    FAILED_PATCHES+=("$patch_name")
  else
    log_error "Cannot parse fail log line: $line"
  fi
done < "$FAIL_LOG"

if [[ ${#FAILED_PATCHES[@]} -eq 0 ]]; then
  log_error "No valid failed files found in the fail log"
  exit 1
fi

log_info "Parsed ${#FAILED_PATCHES[@]} failed patches from the fail log"

# Verify that the patch files exist
declare -a VALID_PATCHES
for patch_name in "${FAILED_PATCHES[@]}"; do
  patch_file="$INPUT_DIR/${patch_name}.tif"
  if [[ -f "$patch_file" ]]; then
    VALID_PATCHES+=("$patch_file")
    log_info "Found failed patch file: $patch_file"
  else
    log_error "Failed patch file does not exist: $patch_file"
  fi
done

if [[ ${#VALID_PATCHES[@]} -eq 0 ]]; then
  log_error "No valid failed patch files found"
  exit 1
fi

log_info "Validation passed, reprocessing ${#VALID_PATCHES[@]} patch files"

# Parse patch info
parse_patch_info() {
  local patch_file=$1
  local filename=$(basename "$patch_file" .tif)
  
  if [[ $filename =~ ^([0-9]{4})_([a-fA-F0-9]+)_agbm$ ]]; then
    local year=${BASH_REMATCH[1]}
    local tile_id=${BASH_REMATCH[2]}
    echo "$year:$tile_id:$filename"
  else
    log_error "Filename format is incorrect: $filename"
    return 1
  fi
}

# Process a single patch
process_single_patch() {
  local patch_file=$1
  local process_id=$2
  
  local patch_info
  if ! patch_info=$(parse_patch_info "$patch_file"); then
    log_error "[$process_id] Parse failed: $patch_file"
    return 1
  fi
  
  IFS=':' read -r year tile_id filename <<< "$patch_info"
  
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local s2_output_dir="$patch_output_dir/data_raw"
  mkdir -p "$s2_output_dir"
  
  local start_date="${year}-01-01T00:00:00"
  local end_date="${year}-12-31T23:59:59"
  
  local log_file="$LOG_DIR/s2_${filename}_${process_id}_retry.log"
  
  log_info "[$process_id] Reprocessing $filename (year $year, tile: $tile_id)"
  
  local start_time=$(date +%s)
  
  (
    trap 'exit 130' INT TERM
    $PYTHON_ENV "$S2_PROCESSOR" \
      --input_tiff "$patch_file" \
      --start_date "$start_date" \
      --end_date "$end_date" \
      --output "$s2_output_dir" \
      --max_cloud "$MAX_CLOUD" \
      --dask_workers "$DASK_WORKERS" \
      --worker_memory "$WORKER_MEMORY" \
      --chunksize 256 \
      --resolution "$RESOLUTION" \
      --min_coverage 5.0 \
      --partition_id "${process_id}_${filename}_retry" \
      --temp_dir "$TEMP_DIR" \
      $OVERWRITE $DEBUG
  ) > "$log_file" 2>&1
  
  local exit_code=$?
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  local minutes=$((duration / 60))
  local seconds=$((duration % 60))
  
  if [[ $exit_code -eq 0 ]]; then
    local output_count=0
    if [[ -d "$s2_output_dir" ]]; then
      for band_dir in "$s2_output_dir"/*; do
        if [[ -d "$band_dir" ]]; then
          band_files=$(find "$band_dir" -name "*.tiff" -type f 2>/dev/null | wc -l)
          output_count=$((output_count + band_files))
        fi
      done
    fi
    log_info "[$process_id] ✅ $filename reprocessed in ${minutes}m${seconds}s, $output_count files written"
    return 0
  else
    log_error "[$process_id] ❌ $filename reprocessing failed (exit code: $exit_code) after ${minutes}m${seconds}s"
    return 1
  fi
}

# Main processing loop
log_info "Starting reprocessing, max parallel: $MAX_PARALLEL"

# Global counters
total_patches=${#VALID_PATCHES[@]}
completed_patches=0
failed_patches=0
current_task_index=0

# Track running processes
declare -A running_pids
overall_start_time=$(date +%s)

# Start a new task
start_new_task() {
  if [[ $current_task_index -ge $total_patches ]]; then
    return 1
  fi
  
  patch_file=${VALID_PATCHES[$current_task_index]}
  filename=$(basename "$patch_file" .tif)
  process_id="RETRY_$((current_task_index + 1))"
  
  process_single_patch "$patch_file" "$process_id" &
  pid=$!
  
  running_pids[$pid]="$patch_file:$filename:$process_id:$(date +%s)"
  
  current_task_index=$((current_task_index + 1))
  
  log_info "Started retry task $process_id: $filename (PID: $pid, $((total_patches - current_task_index)) tasks left)"
  return 0
}

# Check finished processes
check_completed_processes() {
  local completed_this_round=0
  local failed_this_round=0
  
  for pid in "${!running_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      IFS=':' read -r patch_file filename process_id start_time <<< "${running_pids[$pid]}"
      
      if wait "$pid" 2>/dev/null; then
        completed_this_round=$((completed_this_round + 1))
        log_info "✅ $process_id ($filename) reprocessed successfully"
      else
        failed_this_round=$((failed_this_round + 1))
        log_error "❌ $process_id ($filename) reprocessing failed"
      fi
      
      unset running_pids[$pid]
    fi
  done
  
  completed_patches=$((completed_patches + completed_this_round))
  failed_patches=$((failed_patches + failed_this_round))
  
  return $((completed_this_round + failed_this_round))
}

# Current number of running processes
get_running_count() {
  echo "${#running_pids[@]}"
}

log_info "Starting the initial parallel processes..."
for ((i=0; i<MAX_PARALLEL && i<total_patches; i++)); do
  if ! start_new_task; then
    break
  fi
  sleep 0.5
done

log_info "Entering the main monitor loop (checks every 15s)..."

# Main monitor loop
while true; do
  check_completed_processes
  running_count=$(get_running_count)
  
  progress_pct=$(( (completed_patches + failed_patches) * 100 / total_patches ))
  echo "$(date '+%H:%M:%S') 📊 Running: $running_count, done: $completed_patches, failed: $failed_patches, progress: ${progress_pct}% (task index: $current_task_index/$total_patches)"
  
  if [[ $((completed_patches + failed_patches)) -ge $total_patches ]]; then
    log_info "All retry tasks complete, leaving the monitor loop"
    break
  fi
  
  while [[ $running_count -lt $MAX_PARALLEL ]] && [[ $current_task_index -lt $total_patches ]]; do
    if start_new_task; then
      running_count=$((running_count + 1))
      sleep 0.5
    else
      break
    fi
  done
  
  sleep 15
done

# Wait for the remaining processes
log_info "Waiting for the remaining processes..."
while [[ $(get_running_count) -gt 0 ]]; do
  check_completed_processes
  running_count=$(get_running_count)
  echo "$(date '+%H:%M:%S') 🔄 Waiting for the last $running_count processes..."
  sleep 5
done

# Final summary
overall_end_time=$(date +%s)
overall_duration=$((overall_end_time - overall_start_time))
overall_hours=$((overall_duration / 3600))
overall_minutes=$(( (overall_duration % 3600) / 60 ))

echo ""
echo "🎉 Failed patch reprocessing complete!"
echo ""
echo "📊 Reprocessing statistics:"
echo "   Total retried patches: $total_patches"
echo "   Successfully processed: $completed_patches"
echo "   Failed: $failed_patches"
echo "   Total time: ${overall_hours}h${overall_minutes}m"
if [[ $total_patches -gt 0 ]]; then
  echo "   Success rate: $(( completed_patches * 100 / total_patches ))%"
  echo "   Average time: $(( overall_duration / total_patches ))s/patch"
fi
echo "   End time: $(date)"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "📄 Log directory: $LOG_DIR"

retry_summary_log="$LOG_DIR/s2_retry_processing_summary.log"
{
  echo "Sentinel-2 Failed Patches Retry Processing Summary"
  echo "Generated at: $(date)"
  echo "======================================"
  echo ""
  echo "Original fail log: $FAIL_LOG"
  echo "Input directory: $INPUT_DIR"
  echo "Output directory: $OUTPUT_DIR"
  echo "Total retry patches: $total_patches"
  echo "Successfully processed: $completed_patches"
  echo "Failed: $failed_patches"
  if [[ $total_patches -gt 0 ]]; then
    echo "Success rate: $(( completed_patches * 100 / total_patches ))%"
  fi
  echo "Total processing time: ${overall_hours}h ${overall_minutes}m"
  if [[ $total_patches -gt 0 ]]; then
    echo "Average time per patch: $(( overall_duration / total_patches ))s"
  fi
  echo ""
  echo "Processing parameters:"
  echo "- Max parallel: $MAX_PARALLEL"
  echo "- Cores per patch: $CORES_PER_PATCH"
  echo "- Dask workers: $DASK_WORKERS"
  echo "- Worker memory: ${WORKER_MEMORY}GB"
  echo "- Max cloud cover: $MAX_CLOUD%"
  echo "- Resolution: ${RESOLUTION}m"
  echo "- Temp directory: $TEMP_DIR"
} > "$retry_summary_log"

log_info "Retry summary log saved: $retry_summary_log"

if [[ $failed_patches -gt 0 ]]; then
  log_info "$failed_patches patches failed on retry"
  if [[ $completed_patches -eq 0 ]]; then
    exit 1  # all failed
  else
    exit 2  # partial failure
  fi
fi

log_info "All failed patches reprocessed successfully!"
exit 0