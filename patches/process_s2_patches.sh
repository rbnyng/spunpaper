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
  --max_parallel    最大并行patches数量 (默认 80)
  --cores_per_patch 每个patch的CPU核心数 (默认 2)
  --dask_workers    每个patch的Dask worker数 (默认 1)
  --worker_memory   每个worker内存GB (默认 4，为小patch优化)
  --max_cloud       最大云量百分比 (默认 90)
  --resolution      输出分辨率米 (默认 10)
  --overwrite       覆盖已存在文件
  --debug           输出调试日志

示例:
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
# 默认参数
#######################################
MAX_PARALLEL=48
CORES_PER_PATCH=1
DASK_WORKERS=1
WORKER_MEMORY=4
MAX_CLOUD=90
RESOLUTION=10
OVERWRITE=""
DEBUG=""

# 临时目录设置
export TEMP_DIR="/scratch/ray25/spun_patch_proc/tmp"

# 处理器脚本路径
S2_PROCESSOR="./s2_fast_processor_small_patches.py"

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

#######################################
# 验证输入
#######################################
if [[ ! -d "$INPUT_DIR" ]]; then
  log_error "输入目录不存在: $INPUT_DIR"
  exit 1
fi

if [[ ! -f "$S2_PROCESSOR" ]]; then
  log_error "S2处理器不存在: $S2_PROCESSOR"
  exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# 创建日志目录
LOG_DIR="$OUTPUT_DIR/logs_s2"
mkdir -p "$LOG_DIR"

# 初始化失败日志文件
FAIL_LOG="$LOG_DIR/s2_processing_fail.log"
> "$FAIL_LOG"  # 清空或创建失败日志文件
log_info "失败日志文件初始化: $FAIL_LOG"

#######################################
# 查找所有patch文件
#######################################
log_info "扫描patch文件..."
mapfile -t PATCH_FILES < <(find "$INPUT_DIR" -name "*_*_rarefied.tif" -type f | sort)

if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
  log_error "未找到符合格式的patch文件 (*_*_agbm.tif)"
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
  local fail_entry="s2_${filename}_${process_id}"
  
  # 使用文件锁确保并发写入安全
  (
    flock -x 200
    echo "$fail_entry" >> "$FAIL_LOG"
  ) 200>"$FAIL_LOG.lock"
  
  log_debug "记录失败文件到日志: $fail_entry"
}

#######################################
# 处理单个patch的函数
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
  
  # 创建输出目录结构
  local patch_output_dir="$OUTPUT_DIR/${filename}"
  local s2_output_dir="$patch_output_dir/data_raw"
  mkdir -p "$s2_output_dir"
  
  # 设置时间范围
  local start_date="${year}-01-01T00:00:00"
  local end_date="${year}-12-31T23:59:59"
  
  # 日志文件
  local log_file="$LOG_DIR/s2_${filename}_${process_id}.log"
  
  log_info "[$process_id] 开始处理 $filename ($year年, tile: $tile_id, 云量≤$MAX_CLOUD%)"
  
  # 运行S2处理器
  local start_time=$(date +%s)
  
  # 使用trap来确保即使进程被信号中断也能正常处理
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
    # 统计输出文件
    local output_count=0
    if [[ -d "$s2_output_dir" ]]; then
      for band_dir in "$s2_output_dir"/*; do
        if [[ -d "$band_dir" ]]; then
          band_files=$(find "$band_dir" -name "*.tiff" -type f 2>/dev/null | wc -l)
          output_count=$((output_count + band_files))
        fi
      done
    fi
    log_info "[$process_id] $filename 处理完成，用时 ${minutes}分${seconds}秒，生成 $output_count 个文件"
    return 0
  else
    log_error "[$process_id] $filename 处理失败 (退出码: $exit_code)，用时 ${minutes}分${seconds}秒，日志: $log_file"
    # 记录失败文件到失败日志
    record_failure "$filename" "$process_id"
    return 1
  fi
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
echo "   最大云量: $MAX_CLOUD%"
echo "   输出分辨率: ${RESOLUTION}m"
echo "   监视间隔: 15秒"
echo "   失败日志: $FAIL_LOG"
echo ""

# 全局计数器
total_patches=${#PATCH_FILES[@]}
completed_patches=0
failed_patches=0
current_task_index=0

# 跟踪运行中的进程 - 使用临时文件来持久化状态
STATUS_DIR="$LOG_DIR/status"
mkdir -p "$STATUS_DIR"

# 状态文件
RUNNING_PIDS_FILE="$STATUS_DIR/running_pids.txt"
COMPLETED_COUNT_FILE="$STATUS_DIR/completed_count.txt"
FAILED_COUNT_FILE="$STATUS_DIR/failed_count.txt"
TASK_INDEX_FILE="$STATUS_DIR/task_index.txt"

# 初始化状态文件
echo "0" > "$COMPLETED_COUNT_FILE"
echo "0" > "$FAILED_COUNT_FILE"
echo "0" > "$TASK_INDEX_FILE"
> "$RUNNING_PIDS_FILE"

overall_start_time=$(date +%s)

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
# 启动新任务的函数
#######################################
start_new_task() {
  current_task_index=$(read_task_index)
  
  if [[ $current_task_index -ge $total_patches ]]; then
    log_debug "没有更多任务可启动"
    return 1
  fi
  
  patch_file=${PATCH_FILES[$current_task_index]}
  filename=$(basename "$patch_file" .tif)
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
  
  log_info "启动任务 $process_id: $filename (PID: $pid, 剩余 $((total_patches - current_task_index - 1)) 个任务)"
  return 0
}

#######################################
# 检查完成的进程
#######################################
check_completed_processes() {
  if [[ ! -f "$RUNNING_PIDS_FILE" ]]; then
    return 0
  fi
  
  # 读取当前计数
  completed_patches=$(read_completed_count)
  failed_patches=$(read_failed_count)
  
  # 创建临时文件存储仍在运行的进程
  temp_running_file=$(mktemp)
  completed_this_round=0
  failed_this_round=0
  
  # 检查每个运行中的进程
  while IFS=':' read -r pid patch_file filename process_id start_time; do
    # 跳过空行
    [[ -z "$pid" ]] && continue
    
    if kill -0 "$pid" 2>/dev/null; then
      # 进程仍在运行，保留到临时文件
      echo "$pid:$patch_file:$filename:$process_id:$start_time" >> "$temp_running_file"
    else
      # 进程已完成，检查退出状态
      if wait "$pid" 2>/dev/null; then
        # 成功完成
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(( (duration % 3600) / 60 ))
        seconds=$((duration % 60))
        
        completed_this_round=$((completed_this_round + 1))
        log_info "✅ $process_id ($filename) 处理完成，用时 ${hours}h${minutes}m${seconds}s"
      else
        # 失败
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(( (duration % 3600) / 60 ))
        seconds=$((duration % 60))
        
        failed_this_round=$((failed_this_round + 1))
        log_error "❌ $process_id ($filename) 处理失败，用时 ${hours}h${minutes}m${seconds}s"
        
        # 记录失败文件到失败日志（如果process_single_patch函数没有记录的话）
        record_failure "$filename" "$process_id"
      fi
    fi
  done < "$RUNNING_PIDS_FILE"
  
  # 更新运行中的进程文件
  mv "$temp_running_file" "$RUNNING_PIDS_FILE"
  
  # 更新计数器
  if [[ $completed_this_round -gt 0 || $failed_this_round -gt 0 ]]; then
    update_completed_count $((completed_patches + completed_this_round))
    update_failed_count $((failed_patches + failed_this_round))
    log_debug "本轮完成: $completed_this_round, 失败: $failed_this_round"
  fi
  
  return $((completed_this_round + failed_this_round))
}

#######################################
# 获取当前运行中的进程数
#######################################
get_running_count() {
  if [[ -f "$RUNNING_PIDS_FILE" ]]; then
    wc -l < "$RUNNING_PIDS_FILE" | tr -d ' '
  else
    echo "0"
  fi
}

#######################################
# 主调度循环
#######################################
log_info "开始动态调度处理，保持最多 $MAX_PARALLEL 个进程并行运行..."

# 首先启动初始的并行进程
log_info "启动初始并行进程..."
for ((i=0; i<MAX_PARALLEL && i<total_patches; i++)); do
  if ! start_new_task; then
    break
  fi
  sleep 0.5  # 错开启动时间
done

log_info "进入主监视循环（每15秒检查一次）..."

# 主监视循环
while true; do
  # 检查完成的进程
  check_completed_processes
  
  # 读取最新状态
  completed_patches=$(read_completed_count)
  failed_patches=$(read_failed_count)
  current_task_index=$(read_task_index)
  running_count=$(get_running_count)
  
  # 显示状态
  progress_pct=$(( (completed_patches + failed_patches) * 100 / total_patches ))
  echo "$(date '+%H:%M:%S') 📊 运行中: $running_count, 已完成: $completed_patches, 失败: $failed_patches, 进度: ${progress_pct}% (任务索引: $current_task_index/$total_patches)"
  
  # 检查是否所有任务都完成
  if [[ $((completed_patches + failed_patches)) -ge $total_patches ]]; then
    log_info "所有任务已完成，退出监视循环"
    break
  fi
  
  # 如果运行中的进程数少于最大值，启动新任务
  while [[ $running_count -lt $MAX_PARALLEL ]] && [[ $current_task_index -lt $total_patches ]]; do
    if start_new_task; then
      running_count=$((running_count + 1))
      sleep 0.5
    else
      break
    fi
  done
  
  # 等待15秒后再次检查
  sleep 15
done

#######################################
# 等待所有剩余进程完成
#######################################
log_info "等待所有剩余进程完成..."
while [[ $(get_running_count) -gt 0 ]]; do
  check_completed_processes
  running_count=$(get_running_count)
  echo "$(date '+%H:%M:%S') 🔄 等待最后 $running_count 个进程完成..."
  sleep 5
done

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

# 统计失败日志中的条目数
fail_log_count=0
if [[ -f "$FAIL_LOG" ]]; then
  fail_log_count=$(wc -l < "$FAIL_LOG" 2>/dev/null | tr -d ' ')
fi

echo ""
echo "🎉 Sentinel-2 patches SLURM作业处理完成！"
echo ""
echo "📊 最终统计:"
echo "   SLURM作业ID: ${SLURM_JOB_ID:-N/A}"
echo "   总patch数: $total_patches"
echo "   成功处理: $completed_patches"
echo "   处理失败: $failed_patches"
echo "   失败日志条目: $fail_log_count"
echo "   总用时: ${overall_hours}小时${overall_minutes}分钟"
echo "   成功率: $(( completed_patches * 100 / total_patches ))%"
if [[ $total_patches -gt 0 ]]; then
  echo "   平均处理时间: $(( overall_duration / total_patches ))秒/patch"
fi
echo "   结束时间: $(date)"
echo ""
echo "📁 输出目录: $OUTPUT_DIR"
echo "📄 日志目录: $LOG_DIR"
echo "📄 失败日志: $FAIL_LOG"

# 生成汇总日志
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

log_info "汇总日志已保存: $summary_log"

# 如果有失败的文件，显示失败日志信息
if [[ $failed_patches -gt 0 ]]; then
  log_info "有 $failed_patches 个patch处理失败"
  log_info "失败文件列表已记录到: $FAIL_LOG"
  if [[ $fail_log_count -gt 0 ]]; then
    log_info "可使用重新处理脚本处理失败的文件"
  fi
fi

# 清理状态文件
rm -rf "$STATUS_DIR"

# 根据结果设置退出码
if [[ $failed_patches -gt 0 ]]; then
  log_info "有 $failed_patches 个patch处理失败"
  if [[ $completed_patches -eq 0 ]]; then
    exit 1  # 全部失败
  else
    exit 2  # 部分失败
  fi
fi

log_info "所有patches处理成功！"
exit 0