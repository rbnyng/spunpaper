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

## pvc
## climate-dawn-gpu
## pvc9
## AIRR-P3-DAWN-GPU

# Python environment that has the required packages installed
export PYTHON_ENV="/maps/zf281/miniconda3/envs/detectree-env/bin/python"

set -uo pipefail

#######################################
# 帮助信息
#######################################
usage() {
  cat <<EOF
Usage: sbatch $0 --input_dir <path> --output_dir <path> [options]
   or: bash $0 --input_dir <path> --output_dir <path> [options]

必选参数:
  --input_dir       包含 {year}_{tile_id}_agbm.tif 文件的目录
  --output_dir      输出根目录

可选参数:
  --max_parallel    最大并行patches数量 (默认 12)
  --cores_per_patch 每个patch的CPU核心数 (默认 1)
  --dask_workers    每个patch的Dask worker数 (默认 1)
  --worker_memory   每个worker内存GB (默认 4)
  --orbit_state     轨道状态 (ascending/descending/both, 默认 both)
  --timeout_minutes 单个patch超时时间（分钟，默认 30）
  --overwrite       覆盖已存在文件
  --debug           输出调试日志

示例:
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
# 默认参数
#######################################
MAX_PARALLEL=12
CORES_PER_PATCH=1
DASK_WORKERS=1
WORKER_MEMORY=4
ORBIT_STATE="both"
TIMEOUT_MINUTES=30
OVERWRITE=""
DEBUG=""

# 处理器脚本路径
S1_PROCESSOR="./s1_fast_processor.py"

#######################################
# 解析命令行参数
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

# 计算超时秒数
TIMEOUT_SECONDS=$((TIMEOUT_MINUTES * 60))

#######################################
# SLURM 环境信息
#######################################
echo "🚀 SLURM作业信息:"
echo "   作业ID: ${SLURM_JOB_ID:-N/A}"
echo "   节点: ${SLURM_JOB_NODELIST:-N/A}"
echo "   分区: ${SLURM_JOB_PARTITION:-N/A}"
echo "   CPU数: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "   内存: ${SLURM_MEM_PER_NODE:-N/A}MB"
echo "   开始时间: $(date)"
echo ""

#######################################
# 错误处理函数
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
# 验证输入
#######################################
if [[ ! -d "$INPUT_DIR" ]]; then
  log_error "输入目录不存在: $INPUT_DIR"
  exit 1
fi

if [[ ! -f "$S1_PROCESSOR" ]]; then
  log_error "S1处理器不存在: $S1_PROCESSOR"
  exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 创建日志目录
LOG_DIR="$OUTPUT_DIR/logs_s1"
mkdir -p "$LOG_DIR"

# 初始化失败日志文件
FAIL_LOG="$LOG_DIR/s1_processing_fail.log"
> "$FAIL_LOG"  # 清空或创建失败日志文件
log_info "失败日志文件初始化: $FAIL_LOG"

#######################################
# 查找所有patch文件
#######################################
log_info "扫描patch文件..."
mapfile -t PATCH_FILES < <(find "$INPUT_DIR" -name "*_*_rarefied.tif" -type f | sort)

if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
  log_error "未找到符合格式的patch文件 (*_*_rarefied.tif)"
  exit 1
fi

log_info "找到 ${#PATCH_FILES[@]} 个patch文件"

#######################################
# 解析patch信息
#######################################
parse_patch_info() {
  local patch_file=$1
  local filename=$(basename "$patch_file" .tif)
  
  # ^([a-zA-Z0-9]+)  -> Capture Group 1: The Sample ID (one or more letters/numbers)
  # _                 -> A literal underscore
  # (.+)              -> Capture Group 2: The Column Name (one or more of any character)
  # $                 -> End of the string
  if [[ $filename =~ ^([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(.+)$ ]]; then
    local sample_id=${BASH_REMATCH[1]}
    local column_name=${BASH_REMATCH[2]}
    # Echo the parsed components in a colon-separated format
    echo "$sample_id:$column_name:$filename"
  else
    log_error "Filename format is incorrect: '$filename'. Expected 'SAMPLEID_COLUMNNAME.tif'."
    return 1
  fi
}

#######################################
# 记录失败文件到日志的函数
#######################################
record_failure() {
  local filename=$1
  local process_id=$2
  local fail_type=${3:-"failed"}  # failed, timeout
  local fail_entry="s1_${filename}_${process_id}_${fail_type}"
  
  # 使用文件锁确保并发写入安全
  (
    flock -x 200
    echo "$fail_entry" >> "$FAIL_LOG"
  ) 200>"$FAIL_LOG.lock"
  
  log_debug "记录失败文件到日志: $fail_entry"
}

#######################################
# 检查日志文件判断任务是否成功（优化版）
#######################################
check_log_success() {
  local filename=$1
  local process_id=$2
  
  # 构建日志文件路径
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local sar_output_dir="$patch_output_dir/data_sar_raw"
  local log_file="$sar_output_dir/s1_${process_id}_${filename}_detail.log"
  
  # 检查日志文件是否存在
  if [[ ! -f "$log_file" ]]; then
    return 2  # 未知状态
  fi
  
  # 使用更高效的方式检查成功标记
  if grep -q "分区处理完成: 成功" "$log_file" 2>/dev/null; then
    return 0  # 成功
  elif grep -q "分区处理完成:" "$log_file" 2>/dev/null; then
    return 1  # 失败
  else
    return 2  # 未知状态
  fi
}

#######################################
# 检查patch是否已成功完成（快速版）
#######################################
is_patch_completed() {
  local filename=$1
  
  # 首先检查缓存文件
  if [[ -f "$CONFIRMED_SUCCESS_FILE" ]] && grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
    return 0  # 已确认成功
  fi
  
  # 检查所有可能的process_id
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local sar_output_dir="$patch_output_dir/data_sar_raw"
  
  # 如果目录不存在，肯定未完成
  if [[ ! -d "$sar_output_dir" ]]; then
    return 1
  fi
  
  # 查找成功的日志文件
  if find "$sar_output_dir" -name "s1_*_${filename}_detail.log" -type f -exec grep -l "分区处理完成: 成功" {} \; 2>/dev/null | head -1 | grep -q .; then
    # 找到成功日志，添加到缓存
    echo "$filename" >> "$CONFIRMED_SUCCESS_FILE"
    return 0
  fi
  
  return 1  # 未完成
}

#######################################
# 全局变量初始化
#######################################
log_info "初始化处理参数..."
echo "📊 处理参数:"
echo "   输入目录: $INPUT_DIR"
echo "   输出目录: $OUTPUT_DIR"
echo "   Patch数量: ${#PATCH_FILES[@]}"
echo "   最大并行数: $MAX_PARALLEL（动态调度）"
echo "   每patch核心数: $CORES_PER_PATCH"
echo "   Dask workers: $DASK_WORKERS"
echo "   Worker内存: ${WORKER_MEMORY}GB"
echo "   轨道状态: $ORBIT_STATE"
echo "   超时时间: ${TIMEOUT_MINUTES}分钟"
echo "   监视间隔: 15秒"
echo "   失败日志: $FAIL_LOG"
echo ""

# 全局计数器
total_patches=${#PATCH_FILES[@]}
completed_patches=0
failed_patches=0
timeout_patches=0
current_task_index=0
skipped_patches=0  # 新增：跳过的patches数

# 跟踪运行中的进程 - 使用临时文件来持久化状态
STATUS_DIR="$LOG_DIR/status"
mkdir -p "$STATUS_DIR"

# 状态文件
RUNNING_PIDS_FILE="$STATUS_DIR/running_pids.txt"
COMPLETED_COUNT_FILE="$STATUS_DIR/completed_count.txt"
FAILED_COUNT_FILE="$STATUS_DIR/failed_count.txt"
TIMEOUT_COUNT_FILE="$STATUS_DIR/timeout_count.txt"
TASK_INDEX_FILE="$STATUS_DIR/task_index.txt"
CONFIRMED_SUCCESS_FILE="$STATUS_DIR/confirmed_success.txt"  # 新增：已确认成功的文件
PATCH_STATUS_FILE="$STATUS_DIR/patch_status.txt"  # 新增：所有patch的状态缓存

# 初始化状态文件
echo "0" > "$COMPLETED_COUNT_FILE"
echo "0" > "$FAILED_COUNT_FILE"
echo "0" > "$TIMEOUT_COUNT_FILE"
echo "0" > "$TASK_INDEX_FILE"
> "$RUNNING_PIDS_FILE"
> "$CONFIRMED_SUCCESS_FILE"
> "$PATCH_STATUS_FILE"

overall_start_time=$(date +%s)

#######################################
# 初始扫描：检查已完成的任务
#######################################
log_info "初始扫描：检查已完成的任务..."
initial_scan_start=$(date +%s)

# 创建patch索引映射（filename -> index）
declare -A PATCH_INDEX_MAP
declare -A PATCH_FILE_MAP

for i in "${!PATCH_FILES[@]}"; do
  patch_file="${PATCH_FILES[$i]}"
  filename=$(basename "$patch_file" .tif)
  PATCH_INDEX_MAP["$filename"]=$i
  PATCH_FILE_MAP["$filename"]="$patch_file"
done

# 扫描所有patches，建立初始状态
for patch_file in "${PATCH_FILES[@]}"; do
  filename=$(basename "$patch_file" .tif)
  
  if is_patch_completed "$filename"; then
    echo "${filename}:completed" >> "$PATCH_STATUS_FILE"
    completed_patches=$((completed_patches + 1))
    skipped_patches=$((skipped_patches + 1))
    log_debug "已完成: $filename"
  else
    echo "${filename}:pending" >> "$PATCH_STATUS_FILE"
  fi
done

initial_scan_duration=$(($(date +%s) - initial_scan_start))
log_info "初始扫描完成，用时 ${initial_scan_duration}秒，发现 $completed_patches 个已完成的任务"

# 更新计数文件
echo "$completed_patches" > "$COMPLETED_COUNT_FILE"

#######################################
# 读取/更新状态的函数
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
# 快速扫描增量更新（只扫描运行中的任务）
#######################################
quick_scan_update() {
  local changes_made=false
  
  # 仅检查运行中的任务
  if [[ -f "$RUNNING_PIDS_FILE" ]] && [[ -s "$RUNNING_PIDS_FILE" ]]; then
    while IFS=':' read -r pid patch_file filename process_id start_time; do
      [[ -z "$pid" ]] && continue
      
      # 如果进程已结束，检查其状态
      if ! kill -0 "$pid" 2>/dev/null; then
        # 等待日志写入完成
        sleep 0.5
        
        # 检查日志状态
        check_log_success "$filename" "$process_id"
        local log_status=$?
        
        if [[ $log_status -eq 0 ]]; then
          # 成功完成
          if ! grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
            echo "$filename" >> "$CONFIRMED_SUCCESS_FILE"
            completed_patches=$((completed_patches + 1))
            update_completed_count $completed_patches
            changes_made=true
            log_info "✅ $process_id ($filename) 处理成功"
          fi
        elif [[ $log_status -eq 1 ]]; then
          # 失败
          if grep -q "${filename}.*timeout" "$FAIL_LOG" 2>/dev/null; then
            timeout_patches=$((timeout_patches + 1))
            update_timeout_count $timeout_patches
          else
            failed_patches=$((failed_patches + 1))
            update_failed_count $failed_patches
          fi
          changes_made=true
          log_error "❌ $process_id ($filename) 处理失败"
        fi
      fi
    done < "$RUNNING_PIDS_FILE"
  fi
  
  return $([ "$changes_made" = true ] && echo 0 || echo 1)
}

#######################################
# 处理单个patch的函数（优化版）
#######################################
process_single_patch() {
  local patch_file=$1
  local process_id=$2
  
  # 解析patch信息
  local patch_info
  if ! patch_info=$(parse_patch_info "$patch_file"); then
    log_error "[$process_id] 解析失败: $patch_file"
    return 1
  fi
  
  IFS=':' read -r year tile_id filename <<< "$patch_info"
  
  # 再次检查是否已完成（防止竞态条件）
  if is_patch_completed "$filename"; then
    log_info "[$process_id] $filename 已经成功完成，跳过"
    return 0
  fi
  
  # 创建输出目录结构
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local sar_output_dir="$patch_output_dir/data_sar_raw"
  mkdir -p "$sar_output_dir"
  
  # 设置时间范围（整年）
  local start_date="${year}-01-01"
  local end_date="${year}-12-31"
  
  # 日志文件
  local log_file="$LOG_DIR/s1_${filename}_${process_id}.log"
  
  log_info "[$process_id] 开始处理 $filename ($year年, tile: $tile_id, 轨道: $ORBIT_STATE)"
  
  # 运行S1处理器
  local start_time=$(date +%s)
  
  # 使用trap来确保即使进程被信号中断也能正常处理
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
    # 统计输出文件
    local output_count=0
    if [[ -d "$sar_output_dir" ]]; then
      output_count=$(find "$sar_output_dir" -name "*.tiff" -type f 2>/dev/null | wc -l)
    fi
    log_info "[$process_id] $filename 处理完成，用时 ${minutes}分${seconds}秒，生成 $output_count 个文件"
    return 0
  elif [[ $exit_code -eq 130 ]]; then
    # 被信号中断（超时杀死）
    log_timeout "[$process_id] $filename 处理被超时杀死，用时 ${minutes}分${seconds}秒，日志: $log_file"
    record_failure "$filename" "$process_id" "timeout"
    return 130
  else
    log_error "[$process_id] $filename 处理失败 (退出码: $exit_code)，用时 ${minutes}分${seconds}秒，日志: $log_file"
    record_failure "$filename" "$process_id" "failed"
    return 1
  fi
}

#######################################
# 启动新任务的函数（优化版）
#######################################
start_new_task() {
  current_task_index=$(read_task_index)
  
  # 寻找下一个未完成的任务
  while [[ $current_task_index -lt $total_patches ]]; do
    patch_file=${PATCH_FILES[$current_task_index]}
    filename=$(basename "$patch_file" .tif)
    
    # 检查是否已完成
    if grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
      log_debug "任务 $((current_task_index + 1))/$total_patches ($filename) 已完成，跳过"
      current_task_index=$((current_task_index + 1))
      update_task_index $current_task_index
      continue
    fi
    
    # 找到一个未完成的任务
    process_id="P$((current_task_index + 1))"
    
    log_debug "准备启动任务 $process_id: $filename"
    
    # 启动后台进程
    process_single_patch "$patch_file" "$process_id" &
    pid=$!
    
    # 记录PID到文件
    echo "$pid:$patch_file:$filename:$process_id:$(date +%s)" >> "$RUNNING_PIDS_FILE"
    
    # 设置CPU亲和性
    if command -v taskset >/dev/null 2>&1; then
      cpu_start=$(( (current_task_index % (96 / CORES_PER_PATCH)) * CORES_PER_PATCH ))
      cpu_end=$(( cpu_start + CORES_PER_PATCH - 1 ))
      taskset -cp "${cpu_start}-${cpu_end}" $pid >/dev/null 2>&1 || true
    fi
    
    # 更新任务索引
    update_task_index $((current_task_index + 1))
    
    log_info "启动任务 $process_id: $filename (PID: $pid, 剩余待处理 $((total_patches - current_task_index - skipped_patches)) 个任务)"
    return 0
  done
  
  # 没有更多任务可启动
  log_debug "没有更多任务可启动"
  return 1
}

#######################################
# 检查超时进程并杀死（优化版）
#######################################
kill_timeout_processes() {
  if [[ ! -f "$RUNNING_PIDS_FILE" ]] || [[ ! -s "$RUNNING_PIDS_FILE" ]]; then
    return 0
  fi
  
  current_time=$(date +%s)
  temp_running_file=$(mktemp)
  killed_count=0
  
  # 检查每个运行中的进程是否超时
  while IFS=':' read -r pid patch_file filename process_id start_time; do
    # 跳过空行
    [[ -z "$pid" ]] && continue
    
    if kill -0 "$pid" 2>/dev/null; then
      # 进程仍在运行，检查是否超时
      runtime=$((current_time - start_time))
      if [[ $runtime -gt $TIMEOUT_SECONDS ]]; then
        # 超时，杀死进程
        log_timeout "杀死超时进程 $process_id ($filename)，运行时间 $((runtime / 60))分$((runtime % 60))秒"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
          # 如果进程还在，强制杀死
          kill -KILL "$pid" 2>/dev/null || true
        fi
        killed_count=$((killed_count + 1))
        
        # 记录超时失败到日志
        record_failure "$filename" "$process_id" "timeout"
      else
        # 未超时，保留到临时文件
        echo "$pid:$patch_file:$filename:$process_id:$start_time" >> "$temp_running_file"
      fi
    else
      # 进程已结束，不需要保留
      continue
    fi
  done < "$RUNNING_PIDS_FILE"
  
  # 更新运行中的进程文件
  mv "$temp_running_file" "$RUNNING_PIDS_FILE"
  
  return $killed_count
}

#######################################
# 检查完成的进程（优化版）
#######################################
check_completed_processes() {
  if [[ ! -f "$RUNNING_PIDS_FILE" ]] || [[ ! -s "$RUNNING_PIDS_FILE" ]]; then
    return 0
  fi
  
  # 创建临时文件存储仍在运行的进程
  temp_running_file=$(mktemp)
  completed_this_round=0
  
  # 批量检查所有PID
  declare -A pid_status
  while IFS=':' read -r pid patch_file filename process_id start_time; do
    [[ -z "$pid" ]] && continue
    pid_status["$pid"]="$pid:$patch_file:$filename:$process_id:$start_time"
  done < "$RUNNING_PIDS_FILE"
  
  # 检查每个进程
  for pid in "${!pid_status[@]}"; do
    IFS=':' read -r _ patch_file filename process_id start_time <<< "${pid_status[$pid]}"
    
    if kill -0 "$pid" 2>/dev/null; then
      # 进程仍在运行，保留到临时文件
      echo "${pid_status[$pid]}" >> "$temp_running_file"
    else
      # 进程已完成
      completed_this_round=$((completed_this_round + 1))
      
      # 等待一下让日志文件写入完成
      sleep 0.5
      
      # 基于日志检查实际状态
      check_log_success "$filename" "$process_id"
      local log_status=$?
      
      if [[ $log_status -eq 0 ]]; then
        # 成功，添加到确认列表
        if ! grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
          echo "$filename" >> "$CONFIRMED_SUCCESS_FILE"
          completed_patches=$((completed_patches + 1))
          update_completed_count $completed_patches
        fi
        log_info "✅ $process_id ($filename) 处理成功"
      elif [[ $log_status -eq 1 ]]; then
        # 检查是否是超时
        if grep -q "${filename}.*timeout" "$FAIL_LOG" 2>/dev/null; then
          timeout_patches=$((timeout_patches + 1))
          update_timeout_count $timeout_patches
          log_error "⏰ $process_id ($filename) 处理超时"
        else
          failed_patches=$((failed_patches + 1))
          update_failed_count $failed_patches
          log_error "❌ $process_id ($filename) 处理失败"
        fi
      fi
    fi
  done
  
  # 更新运行中的进程文件
  mv "$temp_running_file" "$RUNNING_PIDS_FILE"
  
  return $completed_this_round
}

#######################################
# 获取当前运行中的进程数（优化版）
#######################################
get_running_count() {
  if [[ -f "$RUNNING_PIDS_FILE" ]] && [[ -s "$RUNNING_PIDS_FILE" ]]; then
    # 使用wc -l 快速计算行数
    local line_count=$(wc -l < "$RUNNING_PIDS_FILE" | tr -d ' ')
    echo "$line_count"
  else
    echo "0"
  fi
}

#######################################
# 计算进度百分比（支持小数）
#######################################
calculate_progress() {
  local finished=$1
  local total=$2
  
  if [[ $total -eq 0 ]]; then
    echo "0"
  else
    # 使用awk进行浮点数计算
    echo "$finished $total" | awk '{printf "%.1f", $1 * 100.0 / $2}'
  fi
}

#######################################
# 主调度循环
#######################################
log_info "开始动态调度处理，保持最多 $MAX_PARALLEL 个进程并行运行..."

# 计算实际需要处理的任务数
tasks_to_process=$((total_patches - skipped_patches))
if [[ $tasks_to_process -eq 0 ]]; then
  log_info "所有任务已经完成，无需处理"
else
  log_info "需要处理 $tasks_to_process 个任务（已跳过 $skipped_patches 个已完成的任务）"
  
  # 首先启动初始的并行进程
  log_info "启动初始并行进程..."
  for ((i=0; i<MAX_PARALLEL && i<tasks_to_process; i++)); do
    if ! start_new_task; then
      break
    fi
    sleep 0.2  # 错开启动时间
  done
  
  log_info "进入主监视循环（每15秒检查一次，含超时监控）..."
  
  # 主监视循环
  loop_counter=0
  consecutive_idle_loops=0  # 连续空闲循环计数
  
  while true; do
    # 首先检查并杀死超时进程
    kill_timeout_processes
    
    # 然后检查完成的进程
    check_completed_processes
    
    # 读取最新状态
    completed_patches=$(read_completed_count)
    failed_patches=$(read_failed_count)
    timeout_patches=$(read_timeout_count)
    current_task_index=$(read_task_index)
    running_count=$(get_running_count)
    
    # 显示状态
    total_finished=$((completed_patches + failed_patches + timeout_patches))
    progress_pct=$(calculate_progress $total_finished $total_patches)
    echo "$(date '+%H:%M:%S') 📊 运行中: $running_count, 已完成: $completed_patches, 失败: $failed_patches, 超时: $timeout_patches, 进度: ${progress_pct}% (任务索引: $current_task_index/$total_patches)"
    
    # 检查是否所有任务都完成
    if [[ $current_task_index -ge $total_patches ]] && [[ $running_count -eq 0 ]]; then
      log_info "所有任务已完成"
      break
    fi
    
    # 如果没有运行中的任务且没有更多任务可启动，增加空闲计数
    if [[ $running_count -eq 0 ]] && [[ $current_task_index -ge $total_patches ]]; then
      consecutive_idle_loops=$((consecutive_idle_loops + 1))
      if [[ $consecutive_idle_loops -ge 3 ]]; then
        log_info "连续3次检查无活动任务，退出循环"
        break
      fi
    else
      consecutive_idle_loops=0
    fi
    
    # 如果运行中的进程数少于最大值，启动新任务
    while [[ $running_count -lt $MAX_PARALLEL ]] && [[ $current_task_index -lt $total_patches ]]; do
      if start_new_task; then
        running_count=$((running_count + 1))
        sleep 0.2
      else
        break
      fi
    done
    
    # 等待15秒后再次检查
    sleep 15
    loop_counter=$((loop_counter + 1))
  done
fi

#######################################
# 最终汇总
#######################################
overall_end_time=$(date +%s)
overall_duration=$((overall_end_time - overall_start_time))
overall_hours=$((overall_duration / 3600))
overall_minutes=$(( (overall_duration % 3600) / 60 ))

# 读取最终计数
completed_patches=$(read_completed_count)
failed_patches=$(read_failed_count)
timeout_patches=$(read_timeout_count)

# 统计失败日志中的条目数
fail_log_count=0
if [[ -f "$FAIL_LOG" ]]; then
  fail_log_count=$(wc -l < "$FAIL_LOG" 2>/dev/null | tr -d ' ')
fi

# 生成未处理的patches列表
UNPROCESSED_LOG="$LOG_DIR/s1_unprocessed_patches.log"
> "$UNPROCESSED_LOG"

log_info "生成未处理patches列表..."
unprocessed_count=0
for patch_file in "${PATCH_FILES[@]}"; do
  filename=$(basename "$patch_file" .tif)
  
  # 检查是否已处理成功
  if ! grep -q "^${filename}$" "$CONFIRMED_SUCCESS_FILE" 2>/dev/null; then
    echo "$patch_file" >> "$UNPROCESSED_LOG"
    unprocessed_count=$((unprocessed_count + 1))
  fi
done

echo ""
echo "🎉 Sentinel-1 patches SLURM作业处理完成！"
echo ""
echo "📊 最终统计:"
echo "   SLURM作业ID: ${SLURM_JOB_ID:-N/A}"
echo "   总patch数: $total_patches"
echo "   成功处理: $completed_patches (包含 $skipped_patches 个预先完成的)"
echo "   处理失败: $failed_patches"
echo "   超时杀死: $timeout_patches"
echo "   未处理: $unprocessed_count"
echo "   失败日志条目: $fail_log_count"
echo "   总用时: ${overall_hours}小时${overall_minutes}分钟"
if [[ $total_patches -gt 0 ]]; then
  success_rate=$(calculate_progress $completed_patches $total_patches)
  echo "   成功率: ${success_rate}%"
  if [[ $tasks_to_process -gt 0 ]]; then
    echo "   本次处理时间: $(( overall_duration / tasks_to_process ))秒/patch"
  fi
fi
echo "   结束时间: $(date)"
echo ""
echo "📁 输出目录: $OUTPUT_DIR"
echo "📄 日志目录: $LOG_DIR"
echo "📄 失败日志: $FAIL_LOG"
echo "📄 未处理patches: $UNPROCESSED_LOG"
echo "📄 已确认成功列表: $CONFIRMED_SUCCESS_FILE"

# 生成汇总日志
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

log_info "汇总日志已保存: $summary_log"

# 如果有失败的文件，显示失败日志信息
total_failed=$((failed_patches + timeout_patches))
if [[ $total_failed -gt 0 ]]; then
  log_info "有 $total_failed 个patch处理失败（失败: $failed_patches, 超时: $timeout_patches）"
  log_info "失败文件列表已记录到: $FAIL_LOG"
  if [[ $fail_log_count -gt 0 ]]; then
    log_info "可使用重新处理脚本处理失败的文件"
  fi
fi

if [[ $unprocessed_count -gt 0 ]]; then
  log_info "有 $unprocessed_count 个patch未处理"
  log_info "未处理文件列表已记录到: $UNPROCESSED_LOG"
fi

# 清理状态文件（保留有用的日志）
rm -f "$RUNNING_PIDS_FILE" "$TASK_INDEX_FILE" "$PATCH_STATUS_FILE"

# 根据结果设置退出码
if [[ $total_failed -gt 0 ]] || [[ $unprocessed_count -gt 0 ]]; then
  if [[ $completed_patches -eq 0 ]]; then
    exit 1  # 全部失败
  else
    exit 2  # 部分失败
  fi
fi

log_info "所有patches处理成功！"
exit 0