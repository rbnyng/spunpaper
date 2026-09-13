#!/bin/bash -l

#SBATCH --job-name=btfm-s2-process
#SBATCH --partition=pvc9
#SBATCH --account=AIRR-P3-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=36:00:00
#SBATCH --output=btfm_s2-process_%A_%a.out
#SBATCH --error=btfm_s2-process_%A_%a.err

# --- shared configuration -----------------------------------------------------
# Paths come from config.env at the repository root so they can be swapped without
# editing this script. Precedence: environment variable > config.env > default.
_SPUN_CFG="${SPUN_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config.env}"
# shellcheck disable=SC1090
[ -f "$_SPUN_CFG" ] && . "$_SPUN_CFG"
export PYTHON_ENV="${PYTHON_ENV:-${ACQ_PYTHON_ENV:-python3}}"
set -uo pipefail

#######################################
# Usage
#######################################
usage() {
  cat <<EOF
Usage: sbatch $0 --input_dir <path> --output_dir <path> [options]
   or: bash $0 --input_dir <path> --output_dir <path> [options]

Required arguments:
  --input_dir       Directory containing {year}_{tile_id}_agbm.tif files
  --output_dir      Output root directory

Optional arguments:
  --max_parallel    Max parallel patches (default 80)
  --cores_per_patch CPU cores per patch (default 2)
  --dask_workers    Dask workers per patch (default 1)
  --worker_memory   Memory per worker in GB (default 4, tuned for small patches)
  --max_cloud       Max cloud cover percentage (default 90)
  --resolution      Output resolution in metres (default 10)
  --overwrite       Overwrite existing files
  --debug           Emit debug logs

Example:
bash process_s2_patches.sh \
  --input_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_masks_10m \
  --output_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_d-pixel \
  --max_parallel 36 \
  --max_cloud 90 \
  --overwrite

EOF
  exit 1
}

#######################################
# Default parameters
#######################################
MAX_PARALLEL=48
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

#######################################
# Parse command-line arguments
#######################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir)      INPUT_DIR=$2; shift 2;;
    --output_dir)     OUTPUT_DIR=$2; shift 2;;
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

[[ -z "${INPUT_DIR:-}" || -z "${OUTPUT_DIR:-}" ]] && usage

#######################################
# SLURM environment info
#######################################
echo "🚀 SLURM job info:"
echo "   Job ID: ${SLURM_JOB_ID:-N/A}"
echo "   Node: ${SLURM_JOB_NODELIST:-N/A}"
echo "   Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "   CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "   Memory: ${SLURM_MEM_PER_NODE:-N/A}MB"
echo "   Start time: $(date)"
echo ""

#######################################
# Logging helpers
#######################################
log_error() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ ERROR: $1" >&2
}

log_info() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ℹ️  INFO: $1"
}

log_debug() {
  if [[ -n "$DEBUG" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 🐛 DEBUG: $1"
  fi
}

#######################################
# Validate inputs
#######################################
if [[ ! -d "$INPUT_DIR" ]]; then
  log_error "Input directory does not exist: $INPUT_DIR"
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

FAIL_LOG="$LOG_DIR/s2_processing_fail.log"
> "$FAIL_LOG"  # truncate or create the fail log
log_info "Fail log initialised: $FAIL_LOG"

#######################################
# Find all patch files
#######################################
log_info "Scanning patch files..."
mapfile -t PATCH_FILES < <(find "$INPUT_DIR" -name "*_*_rarefied.tif" -type f | sort)

if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
  log_error "No patch files matching the expected format (*_*_agbm.tif)"
  exit 1
fi

log_info "Found ${#PATCH_FILES[@]} patch files"

#######################################
# Parse patch info
#######################################
parse_patch_info() {
  local patch_file=$1
  local filename=$(basename "$patch_file" .tif)
  
  if [[ $filename =~ ^([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(.+)$ ]]; then
    local sample_id=${BASH_REMATCH[1]}
    local column_name=${BASH_REMATCH[2]}
    echo "$sample_id:$column_name:$filename"
  else
    log_error "Filename format is incorrect: '$filename'. Expected 'SAMPLEID_COLUMNNAME.tif'."
    return 1
  fi
}

#######################################
# Record a failed file in the fail log
#######################################
record_failure() {
  local filename=$1
  local process_id=$2
  local fail_entry="s2_${filename}_${process_id}"
  
  # File lock keeps concurrent writes safe
  (
    flock -x 200
    echo "$fail_entry" >> "$FAIL_LOG"
  ) 200>"$FAIL_LOG.lock"
  
  log_debug "Recorded failure: $fail_entry"
}

#######################################
# Process a single patch
#######################################
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
  
  local log_file="$LOG_DIR/s2_${filename}_${process_id}.log"
  
  log_info "[$process_id] Processing $filename (year $year, tile: $tile_id, cloud ≤$MAX_CLOUD%)"
  
  local start_time=$(date +%s)
  
  # trap keeps handling sane even when the process is interrupted by a signal
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
      --partition_id "${process_id}_${filename}" \
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
    log_info "[$process_id] $filename done in ${minutes}m${seconds}s, $output_count files written"
    return 0
  else
    log_error "[$process_id] $filename failed (exit code: $exit_code) after ${minutes}m${seconds}s, log: $log_file"
    record_failure "$filename" "$process_id"
    return 1
  fi
}

#######################################
# Global variable initialisation
#######################################
log_info "Initialising processing parameters..."
echo "📊 Processing parameters:"
echo "   Input directory: $INPUT_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo "   Patch count: ${#PATCH_FILES[@]}"
echo "   Max parallel: $MAX_PARALLEL (dynamic scheduling)"
echo "   Cores per patch: $CORES_PER_PATCH"
echo "   Dask workers: $DASK_WORKERS"
echo "   Worker memory: ${WORKER_MEMORY}GB"
echo "   Max cloud cover: $MAX_CLOUD%"
echo "   Output resolution: ${RESOLUTION}m"
echo "   Monitor interval: 15s"
echo "   Fail log: $FAIL_LOG"
echo ""

# Global counters
total_patches=${#PATCH_FILES[@]}
completed_patches=0
failed_patches=0
current_task_index=0

# Track running processes - state persisted in temporary files
STATUS_DIR="$LOG_DIR/status"
mkdir -p "$STATUS_DIR"

# State files
RUNNING_PIDS_FILE="$STATUS_DIR/running_pids.txt"
COMPLETED_COUNT_FILE="$STATUS_DIR/completed_count.txt"
FAILED_COUNT_FILE="$STATUS_DIR/failed_count.txt"
TASK_INDEX_FILE="$STATUS_DIR/task_index.txt"

# Initialise state files
echo "0" > "$COMPLETED_COUNT_FILE"
echo "0" > "$FAILED_COUNT_FILE"
echo "0" > "$TASK_INDEX_FILE"
> "$RUNNING_PIDS_FILE"

overall_start_time=$(date +%s)

#######################################
# State read/update helpers
#######################################
read_completed_count() {
  if [[ -f "$COMPLETED_COUNT_FILE" ]]; then
    cat "$COMPLETED_COUNT_FILE"
  else
    echo "0"
  fi
}

read_failed_count() {
  if [[ -f "$FAILED_COUNT_FILE" ]]; then
    cat "$FAILED_COUNT_FILE"
  else
    echo "0"
  fi
}

read_task_index() {
  if [[ -f "$TASK_INDEX_FILE" ]]; then
    cat "$TASK_INDEX_FILE"
  else
    echo "0"
  fi
}

update_completed_count() {
  local new_count=$1
  echo "$new_count" > "$COMPLETED_COUNT_FILE"
  completed_patches=$new_count
}

update_failed_count() {
  local new_count=$1
  echo "$new_count" > "$FAILED_COUNT_FILE"
  failed_patches=$new_count
}

update_task_index() {
  local new_index=$1
  echo "$new_index" > "$TASK_INDEX_FILE"
  current_task_index=$new_index
}

#######################################
# Start a new task
#######################################
start_new_task() {
  current_task_index=$(read_task_index)
  
  if [[ $current_task_index -ge $total_patches ]]; then
    log_debug "No more tasks to start"
    return 1
  fi
  
  patch_file=${PATCH_FILES[$current_task_index]}
  filename=$(basename "$patch_file" .tif)
  process_id="P$((current_task_index + 1))"
  
  log_debug "Preparing task $process_id: $filename"
  
  process_single_patch "$patch_file" "$process_id" &
  pid=$!
  
  echo "$pid:$patch_file:$filename:$process_id:$(date +%s)" >> "$RUNNING_PIDS_FILE"
  
  # Set CPU affinity
  if command -v taskset >/dev/null 2>&1; then
    cpu_start=$(( (current_task_index % (96 / CORES_PER_PATCH)) * CORES_PER_PATCH ))
    cpu_end=$(( cpu_start + CORES_PER_PATCH - 1 ))
    taskset -cp "${cpu_start}-${cpu_end}" $pid >/dev/null 2>&1 || true
  fi
  
  update_task_index $((current_task_index + 1))
  
  log_info "Started task $process_id: $filename (PID: $pid, $((total_patches - current_task_index - 1)) tasks left)"
  return 0
}

#######################################
# Check finished processes
#######################################
check_completed_processes() {
  if [[ ! -f "$RUNNING_PIDS_FILE" ]]; then
    return 0
  fi
  
  completed_patches=$(read_completed_count)
  failed_patches=$(read_failed_count)
  
  # Temp file holds the still-running processes
  temp_running_file=$(mktemp)
  completed_this_round=0
  failed_this_round=0
  
  while IFS=':' read -r pid patch_file filename process_id start_time; do
    [[ -z "$pid" ]] && continue
    
    if kill -0 "$pid" 2>/dev/null; then
      echo "$pid:$patch_file:$filename:$process_id:$start_time" >> "$temp_running_file"
    else
      # Process has finished: check its exit status
      if wait "$pid" 2>/dev/null; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(( (duration % 3600) / 60 ))
        seconds=$((duration % 60))
        
        completed_this_round=$((completed_this_round + 1))
        log_info "✅ $process_id ($filename) done in ${hours}h${minutes}m${seconds}s"
      else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(( (duration % 3600) / 60 ))
        seconds=$((duration % 60))
        
        failed_this_round=$((failed_this_round + 1))
        log_error "❌ $process_id ($filename) failed after ${hours}h${minutes}m${seconds}s"
        
        # Record the failure in case process_single_patch did not
        record_failure "$filename" "$process_id"
      fi
    fi
  done < "$RUNNING_PIDS_FILE"
  
  mv "$temp_running_file" "$RUNNING_PIDS_FILE"
  
  if [[ $completed_this_round -gt 0 || $failed_this_round -gt 0 ]]; then
    update_completed_count $((completed_patches + completed_this_round))
    update_failed_count $((failed_patches + failed_this_round))
    log_debug "This round - done: $completed_this_round, failed: $failed_this_round"
  fi
  
  return $((completed_this_round + failed_this_round))
}

#######################################
# Current number of running processes
#######################################
get_running_count() {
  if [[ -f "$RUNNING_PIDS_FILE" ]]; then
    wc -l < "$RUNNING_PIDS_FILE" | tr -d ' '
  else
    echo "0"
  fi
}

#######################################
# Main scheduling loop
#######################################
log_info "Starting dynamic scheduling, keeping up to $MAX_PARALLEL processes running in parallel..."

log_info "Starting the initial parallel processes..."
for ((i=0; i<MAX_PARALLEL && i<total_patches; i++)); do
  if ! start_new_task; then
    break
  fi
  sleep 0.5  # stagger the launches
done

log_info "Entering the main monitor loop (checks every 15s)..."

# Main monitor loop
while true; do
  check_completed_processes
  
  completed_patches=$(read_completed_count)
  failed_patches=$(read_failed_count)
  current_task_index=$(read_task_index)
  running_count=$(get_running_count)
  
  progress_pct=$(( (completed_patches + failed_patches) * 100 / total_patches ))
  echo "$(date '+%H:%M:%S') 📊 Running: $running_count, done: $completed_patches, failed: $failed_patches, progress: ${progress_pct}% (task index: $current_task_index/$total_patches)"
  
  if [[ $((completed_patches + failed_patches)) -ge $total_patches ]]; then
    log_info "All tasks complete, leaving the monitor loop"
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

#######################################
# Wait for the remaining processes
#######################################
log_info "Waiting for the remaining processes..."
while [[ $(get_running_count) -gt 0 ]]; do
  check_completed_processes
  running_count=$(get_running_count)
  echo "$(date '+%H:%M:%S') 🔄 Waiting for the last $running_count processes..."
  sleep 5
done

#######################################
# Final summary
#######################################
overall_end_time=$(date +%s)
overall_duration=$((overall_end_time - overall_start_time))
overall_hours=$((overall_duration / 3600))
overall_minutes=$(( (overall_duration % 3600) / 60 ))

completed_patches=$(read_completed_count)
failed_patches=$(read_failed_count)

fail_log_count=0
if [[ -f "$FAIL_LOG" ]]; then
  fail_log_count=$(wc -l < "$FAIL_LOG" 2>/dev/null | tr -d ' ')
fi

echo ""
echo "🎉 Sentinel-2 patches SLURM job complete!"
echo ""
echo "📊 Final statistics:"
echo "   SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo "   Total patches: $total_patches"
echo "   Successfully processed: $completed_patches"
echo "   Failed: $failed_patches"
echo "   Fail log entries: $fail_log_count"
echo "   Total time: ${overall_hours}h${overall_minutes}m"
echo "   Success rate: $(( completed_patches * 100 / total_patches ))%"
if [[ $total_patches -gt 0 ]]; then
  echo "   Average time: $(( overall_duration / total_patches ))s/patch"
fi
echo "   End time: $(date)"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "📄 Log directory: $LOG_DIR"
echo "📄 Fail log: $FAIL_LOG"

summary_log="$LOG_DIR/s2_processing_summary.log"
{
  echo "Sentinel-2 Small Patches SLURM Processing Summary"
  echo "Generated at: $(date)"
  echo "======================================"
  echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
  echo "Node: ${SLURM_JOB_NODELIST:-N/A}"
  echo "Partition: ${SLURM_JOB_PARTITION:-N/A}"
  echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
  echo "Memory: ${SLURM_MEM_PER_NODE:-N/A}MB"
  echo ""
  echo "Input directory: $INPUT_DIR"
  echo "Output directory: $OUTPUT_DIR"
  echo "Total patches: $total_patches"
  echo "Successfully processed: $completed_patches"
  echo "Failed: $failed_patches"
  echo "Failed log entries: $fail_log_count"
  echo "Success rate: $(( completed_patches * 100 / total_patches ))%"
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
  echo ""
  echo "Log files:"
  echo "- Summary log: $summary_log"
  echo "- Failed patches log: $FAIL_LOG"
} > "$summary_log"

log_info "Summary log saved: $summary_log"

if [[ $failed_patches -gt 0 ]]; then
  log_info "$failed_patches patches failed"
  log_info "Failed files recorded in: $FAIL_LOG"
  if [[ $fail_log_count -gt 0 ]]; then
    log_info "Use the reprocessing script to retry the failed files"
  fi
fi

rm -rf "$STATUS_DIR"

if [[ $failed_patches -gt 0 ]]; then
  log_info "$failed_patches patches failed"
  if [[ $completed_patches -eq 0 ]]; then
    exit 1  # all failed
  else
    exit 2  # partial failure
  fi
fi

log_info "All patches processed successfully!"
exit 0