#!/bin/bash -l

#SBATCH --job-name=btfm-s1-process
#SBATCH --partition=pvc
#SBATCH --account=climate-dawn-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=36:00:00
#SBATCH --output=btfm_s1-process_%A_%a.out
#SBATCH --error=btfm_s1-process_%A_%a.err


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
  --max_parallel    Max parallel patches (default 12)
  --cores_per_patch CPU cores per patch (default 1)
  --dask_workers    Dask workers per patch (default 1)
  --worker_memory   Memory per worker in GB (default 4)
  --orbit_state     Orbit state (ascending/descending/both, default both)
  --timeout_minutes Per-patch timeout in minutes (default 30)
  --overwrite       Overwrite existing files
  --debug           Emit debug logs

Example:
bash process_s1_patches.sh \
  --input_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_masks_10m \
  --output_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_d-pixel \
  --max_parallel 12 \
  --orbit_state both \
  --overwrite

EOF
  exit 1
}

#######################################
# Default parameters
#######################################
MAX_PARALLEL=12
CORES_PER_PATCH=1
DASK_WORKERS=1
WORKER_MEMORY=4
ORBIT_STATE="both"
TIMEOUT_MINUTES=30
OVERWRITE=""
DEBUG=""

# Processor script path
S1_PROCESSOR="./s1_fast_processor.py"

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
    --orbit_state)    ORBIT_STATE=$2; shift 2;;
    --timeout_minutes) TIMEOUT_MINUTES=$2; shift 2;;
    --overwrite)      OVERWRITE="--overwrite"; shift 1;;
    --debug)          DEBUG="--debug"; shift 1;;
    -h|--help)        usage;;
    *)                echo "Unknown option: $1"; usage;;
  esac
done

[[ -z "${INPUT_DIR:-}" || -z "${OUTPUT_DIR:-}" ]] && usage

TIMEOUT_SECONDS=$((TIMEOUT_MINUTES * 60))

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

log_timeout() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ⏰ TIMEOUT: $1"
}

#######################################
# Validate inputs
#######################################
if [[ ! -d "$INPUT_DIR" ]]; then
  log_error "Input directory does not exist: $INPUT_DIR"
  exit 1
fi

if [[ ! -f "$S1_PROCESSOR" ]]; then
  log_error "S1 processor not found: $S1_PROCESSOR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

LOG_DIR="$OUTPUT_DIR/logs_s1"
mkdir -p "$LOG_DIR"

FAIL_LOG="$LOG_DIR/s1_processing_fail.log"
> "$FAIL_LOG"  # truncate or create the fail log
log_info "Fail log initialised: $FAIL_LOG"

#######################################
# Find all patch files
#######################################
log_info "Scanning patch files..."
mapfile -t PATCH_FILES < <(find "$INPUT_DIR" -name "*_*_rarefied.tif" -type f | sort)

if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
  log_error "No patch files matching the expected format (*_*_rarefied.tif)"
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
  local fail_type=${3:-"failed"}  # failed, timeout
  local fail_entry="s1_${filename}_${process_id}_${fail_type}"
  
  # File lock keeps concurrent writes safe
  (
    flock -x 200
    echo "$fail_entry" >> "$FAIL_LOG"
  ) 200>"$FAIL_LOG.lock"
  
  log_debug "Recorded failure: $fail_entry"
}

#######################################
# Check the log file to decide whether a task succeeded
#######################################
check_log_success() {
  local filename=$1
  local process_id=$2
  
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local sar_output_dir="$patch_output_dir/data_sar_raw"
  local log_file="$sar_output_dir/s1_${process_id}_${filename}_detail.log"
  
  if [[ ! -f "$log_file" ]]; then
    return 2  # unknown
  fi
  
  if grep -q "Partition complete: succeeded" "$log_file" 2>/dev/null; then
    return 0  # success
  elif grep -q "Partition complete:" "$log_file" 2>/dev/null; then
    return 1  # failure
  else
    return 2  # unknown
  fi
}

#######################################
# Check whether a patch already completed successfully (fast path)
#######################################
is_patch_completed() {
  local filename=$1
  
  if [[ -f "$CONFIRMED_SUCCESS_FILE" ]] && grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
    return 0  # confirmed success
  fi
  
  # Check every possible process_id
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local sar_output_dir="$patch_output_dir/data_sar_raw"
  
  # No directory means definitely not finished
  if [[ ! -d "$sar_output_dir" ]]; then
    return 1
  fi
  
  if find "$sar_output_dir" -name "s1_*_${filename}_detail.log" -type f -exec grep -l "Partition complete: succeeded" {} \; 2>/dev/null | head -1 | grep -q .; then
    echo "$filename" >> "$CONFIRMED_SUCCESS_FILE"
    return 0
  fi
  
  return 1  # not finished
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
echo "   Orbit state: $ORBIT_STATE"
echo "   Timeout: ${TIMEOUT_MINUTES} minutes"
echo "   Monitor interval: 15s"
echo "   Fail log: $FAIL_LOG"
echo ""

# Global counters
total_patches=${#PATCH_FILES[@]}
completed_patches=0
failed_patches=0
timeout_patches=0
current_task_index=0
skipped_patches=0  # patches skipped

# Track running processes - state persisted in temporary files
STATUS_DIR="$LOG_DIR/status"
mkdir -p "$STATUS_DIR"

# State files
RUNNING_PIDS_FILE="$STATUS_DIR/running_pids.txt"
COMPLETED_COUNT_FILE="$STATUS_DIR/completed_count.txt"
FAILED_COUNT_FILE="$STATUS_DIR/failed_count.txt"
TIMEOUT_COUNT_FILE="$STATUS_DIR/timeout_count.txt"
TASK_INDEX_FILE="$STATUS_DIR/task_index.txt"
CONFIRMED_SUCCESS_FILE="$STATUS_DIR/confirmed_success.txt"  # files confirmed successful
PATCH_STATUS_FILE="$STATUS_DIR/patch_status.txt"  # status cache for all patches

# Initialise state files
echo "0" > "$COMPLETED_COUNT_FILE"
echo "0" > "$FAILED_COUNT_FILE"
echo "0" > "$TIMEOUT_COUNT_FILE"
echo "0" > "$TASK_INDEX_FILE"
> "$RUNNING_PIDS_FILE"
> "$CONFIRMED_SUCCESS_FILE"
> "$PATCH_STATUS_FILE"

overall_start_time=$(date +%s)

#######################################
# Initial scan: find already-completed tasks
#######################################
log_info "Initial scan: checking for already-completed tasks..."
initial_scan_start=$(date +%s)

# Patch index map (filename -> index)
declare -A PATCH_INDEX_MAP
declare -A PATCH_FILE_MAP

for i in "${!PATCH_FILES[@]}"; do
  patch_file="${PATCH_FILES[$i]}"
  filename=$(basename "$patch_file" .tif)
  PATCH_INDEX_MAP["$filename"]=$i
  PATCH_FILE_MAP["$filename"]="$patch_file"
done

# Scan all patches to establish the initial state
for patch_file in "${PATCH_FILES[@]}"; do
  filename=$(basename "$patch_file" .tif)
  
  if is_patch_completed "$filename"; then
    echo "${filename}:completed" >> "$PATCH_STATUS_FILE"
    completed_patches=$((completed_patches + 1))
    skipped_patches=$((skipped_patches + 1))
    log_debug "Already done: $filename"
  else
    echo "${filename}:pending" >> "$PATCH_STATUS_FILE"
  fi
done

initial_scan_duration=$(($(date +%s) - initial_scan_start))
log_info "Initial scan finished in ${initial_scan_duration}s, found $completed_patches completed tasks"

echo "$completed_patches" > "$COMPLETED_COUNT_FILE"

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

read_timeout_count() {
  if [[ -f "$TIMEOUT_COUNT_FILE" ]]; then
    cat "$TIMEOUT_COUNT_FILE"
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

update_timeout_count() {
  local new_count=$1
  echo "$new_count" > "$TIMEOUT_COUNT_FILE"
  timeout_patches=$new_count
}

update_task_index() {
  local new_index=$1
  echo "$new_index" > "$TASK_INDEX_FILE"
  current_task_index=$new_index
}

#######################################
# Incremental quick scan (running tasks only)
#######################################
quick_scan_update() {
  local changes_made=false
  
  # Only inspect running tasks
  if [[ -f "$RUNNING_PIDS_FILE" ]] && [[ -s "$RUNNING_PIDS_FILE" ]]; then
    while IFS=':' read -r pid patch_file filename process_id start_time; do
      [[ -z "$pid" ]] && continue
      
      # Process has exited: check its status
      if ! kill -0 "$pid" 2>/dev/null; then
        # Let the log finish being written
        sleep 0.5
        
        check_log_success "$filename" "$process_id"
        local log_status=$?
        
        if [[ $log_status -eq 0 ]]; then
          # Completed successfully
          if ! grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
            echo "$filename" >> "$CONFIRMED_SUCCESS_FILE"
            completed_patches=$((completed_patches + 1))
            update_completed_count $completed_patches
            changes_made=true
            log_info "✅ $process_id ($filename) succeeded"
          fi
        elif [[ $log_status -eq 1 ]]; then
          # Failed
          if grep -q "${filename}.*timeout" "$FAIL_LOG" 2>/dev/null; then
            timeout_patches=$((timeout_patches + 1))
            update_timeout_count $timeout_patches
          else
            failed_patches=$((failed_patches + 1))
            update_failed_count $failed_patches
          fi
          changes_made=true
          log_error "❌ $process_id ($filename) failed"
        fi
      fi
    done < "$RUNNING_PIDS_FILE"
  fi
  
  return $([ "$changes_made" = true ] && echo 0 || echo 1)
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
  
  # Re-check completion to avoid a race
  if is_patch_completed "$filename"; then
    log_info "[$process_id] $filename already completed successfully, skipping"
    return 0
  fi
  
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local sar_output_dir="$patch_output_dir/data_sar_raw"
  mkdir -p "$sar_output_dir"
  
  local start_date="${year}-01-01"
  local end_date="${year}-12-31"
  
  local log_file="$LOG_DIR/s1_${filename}_${process_id}.log"
  
  log_info "[$process_id] Processing $filename (year $year, tile: $tile_id, orbit: $ORBIT_STATE)"
  
  local start_time=$(date +%s)
  
  # trap keeps handling sane even when the process is interrupted by a signal
  (
    trap 'exit 130' INT TERM
    $PYTHON_ENV "$S1_PROCESSOR" \
      --input_tiff "$patch_file" \
      --start_date "$start_date" \
      --end_date "$end_date" \
      --output "$sar_output_dir" \
      --orbit_state "$ORBIT_STATE" \
      --dask_workers "$DASK_WORKERS" \
      --worker_memory "$WORKER_MEMORY" \
      --chunksize 512 \
      --min_coverage 5.0 \
      --partition_id "${process_id}_${filename}" \
      $OVERWRITE $DEBUG
  ) > "$log_file" 2>&1
  
  local exit_code=$?
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  local minutes=$((duration / 60))
  local seconds=$((duration % 60))
  
  if [[ $exit_code -eq 0 ]]; then
    local output_count=0
    if [[ -d "$sar_output_dir" ]]; then
      output_count=$(find "$sar_output_dir" -name "*.tiff" -type f 2>/dev/null | wc -l)
    fi
    log_info "[$process_id] $filename done in ${minutes}m${seconds}s, $output_count files written"
    return 0
  elif [[ $exit_code -eq 130 ]]; then
    # Interrupted by a signal (killed on timeout)
    log_timeout "[$process_id] $filename killed on timeout after ${minutes}m${seconds}s, log: $log_file"
    record_failure "$filename" "$process_id" "timeout"
    return 130
  else
    log_error "[$process_id] $filename failed (exit code: $exit_code) after ${minutes}m${seconds}s, log: $log_file"
    record_failure "$filename" "$process_id" "failed"
    return 1
  fi
}

#######################################
# Start a new task
#######################################
start_new_task() {
  current_task_index=$(read_task_index)
  
  # Find the next unfinished task
  while [[ $current_task_index -lt $total_patches ]]; do
    patch_file=${PATCH_FILES[$current_task_index]}
    filename=$(basename "$patch_file" .tif)
    
    if grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
      log_debug "Task $((current_task_index + 1))/$total_patches ($filename) already done, skipping"
      current_task_index=$((current_task_index + 1))
      update_task_index $current_task_index
      continue
    fi
    
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
    
    log_info "Started task $process_id: $filename (PID: $pid, $((total_patches - current_task_index - skipped_patches)) tasks left)"
    return 0
  done
  
  log_debug "No more tasks to start"
  return 1
}

#######################################
# Kill timed-out processes
#######################################
kill_timeout_processes() {
  if [[ ! -f "$RUNNING_PIDS_FILE" ]] || [[ ! -s "$RUNNING_PIDS_FILE" ]]; then
    return 0
  fi
  
  current_time=$(date +%s)
  temp_running_file=$(mktemp)
  killed_count=0
  
  # Check each running process for timeout
  while IFS=':' read -r pid patch_file filename process_id start_time; do
    [[ -z "$pid" ]] && continue
    
    if kill -0 "$pid" 2>/dev/null; then
      # Still running: check for timeout
      runtime=$((current_time - start_time))
      if [[ $runtime -gt $TIMEOUT_SECONDS ]]; then
        log_timeout "Killing timed-out process $process_id ($filename) after $((runtime / 60))m$((runtime % 60))s"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
          kill -KILL "$pid" 2>/dev/null || true
        fi
        killed_count=$((killed_count + 1))
        
        record_failure "$filename" "$process_id" "timeout"
      else
        echo "$pid:$patch_file:$filename:$process_id:$start_time" >> "$temp_running_file"
      fi
    else
      continue
    fi
  done < "$RUNNING_PIDS_FILE"
  
  mv "$temp_running_file" "$RUNNING_PIDS_FILE"
  
  return $killed_count
}

#######################################
# Check finished processes
#######################################
check_completed_processes() {
  if [[ ! -f "$RUNNING_PIDS_FILE" ]] || [[ ! -s "$RUNNING_PIDS_FILE" ]]; then
    return 0
  fi
  
  # Temp file holds the still-running processes
  temp_running_file=$(mktemp)
  completed_this_round=0
  
  # Batch-load all PIDs
  declare -A pid_status
  while IFS=':' read -r pid patch_file filename process_id start_time; do
    [[ -z "$pid" ]] && continue
    pid_status["$pid"]="$pid:$patch_file:$filename:$process_id:$start_time"
  done < "$RUNNING_PIDS_FILE"
  
  for pid in "${!pid_status[@]}"; do
    IFS=':' read -r _ patch_file filename process_id start_time <<< "${pid_status[$pid]}"
    
    if kill -0 "$pid" 2>/dev/null; then
      echo "${pid_status[$pid]}" >> "$temp_running_file"
    else
      completed_this_round=$((completed_this_round + 1))
      
      # Let the log file finish being written
      sleep 0.5
      
      check_log_success "$filename" "$process_id"
      local log_status=$?
      
      if [[ $log_status -eq 0 ]]; then
        # Success: add to the confirmed list
        if ! grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
          echo "$filename" >> "$CONFIRMED_SUCCESS_FILE"
          completed_patches=$((completed_patches + 1))
          update_completed_count $completed_patches
        fi
        log_info "✅ $process_id ($filename) succeeded"
      elif [[ $log_status -eq 1 ]]; then
        # Distinguish timeout from plain failure
        if grep -q "${filename}.*timeout" "$FAIL_LOG" 2>/dev/null; then
          timeout_patches=$((timeout_patches + 1))
          update_timeout_count $timeout_patches
          log_error "⏰ $process_id ($filename) timed out"
        else
          failed_patches=$((failed_patches + 1))
          update_failed_count $failed_patches
          log_error "❌ $process_id ($filename) failed"
        fi
      fi
    fi
  done
  
  mv "$temp_running_file" "$RUNNING_PIDS_FILE"
  
  return $completed_this_round
}

#######################################
# Current number of running processes
#######################################
get_running_count() {
  if [[ -f "$RUNNING_PIDS_FILE" ]] && [[ -s "$RUNNING_PIDS_FILE" ]]; then
    local line_count=$(wc -l < "$RUNNING_PIDS_FILE" | tr -d ' ')
    echo "$line_count"
  else
    echo "0"
  fi
}

#######################################
# Progress percentage (fractional)
#######################################
calculate_progress() {
  local finished=$1
  local total=$2
  
  if [[ $total -eq 0 ]]; then
    echo "0"
  else
    echo "$finished $total" | awk '{printf "%.1f", $1 * 100.0 / $2}'
  fi
}

#######################################
# Main scheduling loop
#######################################
log_info "Starting dynamic scheduling, keeping up to $MAX_PARALLEL processes running in parallel..."

# Number of tasks that actually need processing
tasks_to_process=$((total_patches - skipped_patches))
if [[ $tasks_to_process -eq 0 ]]; then
  log_info "All tasks already complete, nothing to do"
else
  log_info "Need to process $tasks_to_process tasks (skipped $skipped_patches already-completed tasks)"
  
  log_info "Starting the initial parallel processes..."
  for ((i=0; i<MAX_PARALLEL && i<tasks_to_process; i++)); do
    if ! start_new_task; then
      break
    fi
    sleep 0.2  # stagger the launches
  done
  
  log_info "Entering the main monitor loop (checks every 15s, including timeouts)..."
  
  # Main monitor loop
  loop_counter=0
  consecutive_idle_loops=0  # consecutive idle loop count
  
  while true; do
    kill_timeout_processes
    
    check_completed_processes
    
    completed_patches=$(read_completed_count)
    failed_patches=$(read_failed_count)
    timeout_patches=$(read_timeout_count)
    current_task_index=$(read_task_index)
    running_count=$(get_running_count)
    
    total_finished=$((completed_patches + failed_patches + timeout_patches))
    progress_pct=$(calculate_progress $total_finished $total_patches)
    echo "$(date '+%H:%M:%S') 📊 Running: $running_count, done: $completed_patches, failed: $failed_patches, timed out: $timeout_patches, progress: ${progress_pct}% (task index: $current_task_index/$total_patches)"
    
    if [[ $current_task_index -ge $total_patches ]] && [[ $running_count -eq 0 ]]; then
      log_info "All tasks complete"
      break
    fi
    
    # No running tasks and nothing left to start: bump the idle counter
    if [[ $running_count -eq 0 ]] && [[ $current_task_index -ge $total_patches ]]; then
      consecutive_idle_loops=$((consecutive_idle_loops + 1))
      if [[ $consecutive_idle_loops -ge 3 ]]; then
        log_info "No active tasks for 3 consecutive checks, leaving the loop"
        break
      fi
    else
      consecutive_idle_loops=0
    fi
    
    while [[ $running_count -lt $MAX_PARALLEL ]] && [[ $current_task_index -lt $total_patches ]]; do
      if start_new_task; then
        running_count=$((running_count + 1))
        sleep 0.2
      else
        break
      fi
    done
    
    sleep 15
    loop_counter=$((loop_counter + 1))
  done
fi

#######################################
# Final summary
#######################################
overall_end_time=$(date +%s)
overall_duration=$((overall_end_time - overall_start_time))
overall_hours=$((overall_duration / 3600))
overall_minutes=$(( (overall_duration % 3600) / 60 ))

completed_patches=$(read_completed_count)
failed_patches=$(read_failed_count)
timeout_patches=$(read_timeout_count)

fail_log_count=0
if [[ -f "$FAIL_LOG" ]]; then
  fail_log_count=$(wc -l < "$FAIL_LOG" 2>/dev/null | tr -d ' ')
fi

UNPROCESSED_LOG="$LOG_DIR/s1_unprocessed_patches.log"
> "$UNPROCESSED_LOG"

log_info "Building the unprocessed patch list..."
unprocessed_count=0
for patch_file in "${PATCH_FILES[@]}"; do
  filename=$(basename "$patch_file" .tif)
  
  if ! grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
    echo "$patch_file" >> "$UNPROCESSED_LOG"
    unprocessed_count=$((unprocessed_count + 1))
  fi
done

echo ""
echo "🎉 Sentinel-1 patches SLURM job complete!"
echo ""
echo "📊 Final statistics:"
echo "   SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo "   Total patches: $total_patches"
echo "   Successfully processed: $completed_patches (including $skipped_patches pre-completed)"
echo "   Failed: $failed_patches"
echo "   Killed on timeout: $timeout_patches"
echo "   Unprocessed: $unprocessed_count"
echo "   Fail log entries: $fail_log_count"
echo "   Total time: ${overall_hours}h${overall_minutes}m"
if [[ $total_patches -gt 0 ]]; then
  success_rate=$(calculate_progress $completed_patches $total_patches)
  echo "   Success rate: ${success_rate}%"
  if [[ $tasks_to_process -gt 0 ]]; then
    echo "   Time this run: $(( overall_duration / tasks_to_process ))s/patch"
  fi
fi
echo "   End time: $(date)"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "📄 Log directory: $LOG_DIR"
echo "📄 Fail log: $FAIL_LOG"
echo "📄 Unprocessed patches: $UNPROCESSED_LOG"
echo "📄 Confirmed success list: $CONFIRMED_SUCCESS_FILE"

summary_log="$LOG_DIR/s1_processing_summary.log"
{
  echo "Sentinel-1 Patches SLURM Processing Summary"
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
  echo "Successfully processed: $completed_patches (including $skipped_patches pre-completed)"
  echo "Failed: $failed_patches"
  echo "Timeout killed: $timeout_patches"
  echo "Unprocessed: $unprocessed_count"
  echo "Failed log entries: $fail_log_count"
  if [[ $total_patches -gt 0 ]]; then
    success_rate=$(calculate_progress $completed_patches $total_patches)
    echo "Success rate: ${success_rate}%"
    if [[ $tasks_to_process -gt 0 ]]; then
      echo "Average time per patch (this run): $(( overall_duration / tasks_to_process ))s"
    fi
  fi
  echo "Total processing time: ${overall_hours}h ${overall_minutes}m"
  echo ""
  echo "Processing parameters:"
  echo "- Max parallel: $MAX_PARALLEL"
  echo "- Cores per patch: $CORES_PER_PATCH"
  echo "- Dask workers: $DASK_WORKERS"
  echo "- Worker memory: ${WORKER_MEMORY}GB"
  echo "- Orbit state: $ORBIT_STATE"
  echo "- Timeout: ${TIMEOUT_MINUTES} minutes"
  echo ""
  echo "Log files:"
  echo "- Summary log: $summary_log"
  echo "- Failed patches log: $FAIL_LOG"
  echo "- Unprocessed patches log: $UNPROCESSED_LOG"
  echo "- Confirmed success list: $CONFIRMED_SUCCESS_FILE"
} > "$summary_log"

log_info "Summary log saved: $summary_log"

total_failed=$((failed_patches + timeout_patches))
if [[ $total_failed -gt 0 ]]; then
  log_info "$total_failed patches failed (failed: $failed_patches, timed out: $timeout_patches)"
  log_info "Failed files recorded in: $FAIL_LOG"
  if [[ $fail_log_count -gt 0 ]]; then
    log_info "Use the reprocessing script to retry the failed files"
  fi
fi

if [[ $unprocessed_count -gt 0 ]]; then
  log_info "$unprocessed_count patches were not processed"
  log_info "Unprocessed files recorded in: $UNPROCESSED_LOG"
fi

# Clean up state files (keep the useful logs)
rm -f "$RUNNING_PIDS_FILE" "$TASK_INDEX_FILE" "$PATCH_STATUS_FILE"

if [[ $total_failed -gt 0 ]] || [[ $unprocessed_count -gt 0 ]]; then
  if [[ $completed_patches -eq 0 ]]; then
    exit 1  # all failed
  else
    exit 2  # partial failure
  fi
fi

log_info "All patches processed successfully!"
exit 0