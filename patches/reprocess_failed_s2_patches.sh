#!/bin/bash -l

# reprocess_failed_s2_patches.sh
# 重新处理失败的S2 patches脚本
# 读取失败日志并重新处理失败的文件

usage() {
  cat <<EOF
Usage: bash $0 --input_dir <path> --output_dir <path> --fail_log <path> [options]

必选参数:
  --input_dir       包含 {year}_{tile_id}_agbm.tif 文件的目录
  --output_dir      输出根目录
  --fail_log        失败日志文件路径

可选参数:
  --max_parallel    最大并行patches数量 (默认 24)
  --cores_per_patch 每个patch的CPU核心数 (默认 2)
  --dask_workers    每个patch的Dask worker数 (默认 1)
  --worker_memory   每个worker内存GB (默认 4)
  --max_cloud       最大云量百分比 (默认 90)
  --resolution      输出分辨率米 (默认 10)
  --overwrite       覆盖已存在文件
  --debug           输出调试日志

示例:
bash reprocess_failed_s2_patches.sh \
  --input_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_masks_10m \
  --output_dir /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_d-pixel \
  --fail_log /scratch/zf281/create_d-pixels_biomassters/data/train_agbm_d-pixel/logs_s2/s2_processing_fail.log \
  --overwrite

EOF
  exit 1
}

# 默认参数
MAX_PARALLEL=24
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
export PYTHON_ENV="/maps/zf281/miniconda3/envs/detectree-env/bin/python"

# 解析命令行参数
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

# 日志函数
log_error() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ ERROR: $1" >&2
}

log_info() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ℹ️  INFO: $1"
}

# 验证输入
if [[ ! -d "$INPUT_DIR" ]]; then
  log_error "输入目录不存在: $INPUT_DIR"
  exit 1
fi

if [[ ! -f "$FAIL_LOG" ]]; then
  log_error "失败日志文件不存在: $FAIL_LOG"
  exit 1
fi

if [[ ! -f "$S2_PROCESSOR" ]]; then
  log_error "S2处理器不存在: $S2_PROCESSOR"
  exit 1
fi

# 创建输出和临时目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# 创建日志目录
LOG_DIR="$OUTPUT_DIR/logs_s2"
mkdir -p "$LOG_DIR"

log_info "🔄 开始重新处理失败的S2 patches"
log_info "失败日志文件: $FAIL_LOG"

# 解析失败日志文件，提取真正的文件名
declare -a FAILED_PATCHES
while IFS= read -r line; do
  # 跳过空行和注释行
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  
  # 提取文件名：去掉 s2_ 前缀和 _P**** 后缀
  if [[ $line =~ ^s2_(.+)_P[0-9]+$ ]]; then
    patch_name="${BASH_REMATCH[1]}"
    FAILED_PATCHES+=("$patch_name")
  else
    log_error "无法解析失败日志行: $line"
  fi
done < "$FAIL_LOG"

if [[ ${#FAILED_PATCHES[@]} -eq 0 ]]; then
  log_error "失败日志中没有找到有效的失败文件"
  exit 1
fi

log_info "从失败日志中解析到 ${#FAILED_PATCHES[@]} 个失败的patch"

# 验证patch文件是否存在
declare -a VALID_PATCHES
for patch_name in "${FAILED_PATCHES[@]}"; do
  patch_file="$INPUT_DIR/${patch_name}.tif"
  if [[ -f "$patch_file" ]]; then
    VALID_PATCHES+=("$patch_file")
    log_info "找到失败patch文件: $patch_file"
  else
    log_error "失败patch文件不存在: $patch_file"
  fi
done

if [[ ${#VALID_PATCHES[@]} -eq 0 ]]; then
  log_error "没有找到任何有效的失败patch文件"
  exit 1
fi

log_info "验证通过，将重新处理 ${#VALID_PATCHES[@]} 个patch文件"

# 解析patch信息函数
parse_patch_info() {
  local patch_file=$1
  local filename=$(basename "$patch_file" .tif)
  
  if [[ $filename =~ ^([0-9]{4})_([a-fA-F0-9]+)_agbm$ ]]; then
    local year=${BASH_REMATCH[1]}
    local tile_id=${BASH_REMATCH[2]}
    echo "$year:$tile_id:$filename"
  else
    log_error "文件名格式不正确: $filename"
    return 1
  fi
}

# 处理单个patch的函数
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
  local log_file="$LOG_DIR/s2_${filename}_${process_id}_retry.log"
  
  log_info "[$process_id] 重新处理 $filename ($year年, tile: $tile_id)"
  
  # 运行S2处理器
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
    log_info "[$process_id] ✅ $filename 重新处理成功，用时 ${minutes}分${seconds}秒，生成 $output_count 个文件"
    return 0
  else
    log_error "[$process_id] ❌ $filename 重新处理失败 (退出码: $exit_code)，用时 ${minutes}分${seconds}秒"
    return 1
  fi
}

# 主处理循环
log_info "开始重新处理，最大并行数: $MAX_PARALLEL"

# 全局计数器
total_patches=${#VALID_PATCHES[@]}
completed_patches=0
failed_patches=0
current_task_index=0

# 跟踪运行中的进程
declare -A running_pids
overall_start_time=$(date +%s)

# 启动新任务函数
start_new_task() {
  if [[ $current_task_index -ge $total_patches ]]; then
    return 1
  fi
  
  patch_file=${VALID_PATCHES[$current_task_index]}
  filename=$(basename "$patch_file" .tif)
  process_id="RETRY_$((current_task_index + 1))"
  
  # 启动后台进程
  process_single_patch "$patch_file" "$process_id" &
  pid=$!
  
  # 记录PID
  running_pids[$pid]="$patch_file:$filename:$process_id:$(date +%s)"
  
  # 更新任务索引
  current_task_index=$((current_task_index + 1))
  
  log_info "启动重试任务 $process_id: $filename (PID: $pid, 剩余 $((total_patches - current_task_index)) 个任务)"
  return 0
}

# 检查完成的进程
check_completed_processes() {
  local completed_this_round=0
  local failed_this_round=0
  
  for pid in "${!running_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      # 进程已完成
      IFS=':' read -r patch_file filename process_id start_time <<< "${running_pids[$pid]}"
      
      if wait "$pid" 2>/dev/null; then
        # 成功完成
        completed_this_round=$((completed_this_round + 1))
        log_info "✅ $process_id ($filename) 重新处理成功"
      else
        # 失败
        failed_this_round=$((failed_this_round + 1))
        log_error "❌ $process_id ($filename) 重新处理失败"
      fi
      
      # 从跟踪列表中移除
      unset running_pids[$pid]
    fi
  done
  
  completed_patches=$((completed_patches + completed_this_round))
  failed_patches=$((failed_patches + failed_this_round))
  
  return $((completed_this_round + failed_this_round))
}

# 获取当前运行中的进程数
get_running_count() {
  echo "${#running_pids[@]}"
}

# 首先启动初始的并行进程
log_info "启动初始并行进程..."
for ((i=0; i<MAX_PARALLEL && i<total_patches; i++)); do
  if ! start_new_task; then
    break
  fi
  sleep 0.5
done

log_info "进入主监视循环（每15秒检查一次）..."

# 主监视循环
while true; do
  check_completed_processes
  running_count=$(get_running_count)
  
  # 显示状态
  progress_pct=$(( (completed_patches + failed_patches) * 100 / total_patches ))
  echo "$(date '+%H:%M:%S') 📊 运行中: $running_count, 已完成: $completed_patches, 失败: $failed_patches, 进度: ${progress_pct}% (任务索引: $current_task_index/$total_patches)"
  
  # 检查是否所有任务都完成
  if [[ $((completed_patches + failed_patches)) -ge $total_patches ]]; then
    log_info "所有重试任务已完成，退出监视循环"
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

# 等待所有剩余进程完成
log_info "等待所有剩余进程完成..."
while [[ $(get_running_count) -gt 0 ]]; do
  check_completed_processes
  running_count=$(get_running_count)
  echo "$(date '+%H:%M:%S') 🔄 等待最后 $running_count 个进程完成..."
  sleep 5
done

# 最终汇总
overall_end_time=$(date +%s)
overall_duration=$((overall_end_time - overall_start_time))
overall_hours=$((overall_duration / 3600))
overall_minutes=$(( (overall_duration % 3600) / 60 ))

echo ""
echo "🎉 失败patch重新处理完成！"
echo ""
echo "📊 重新处理统计:"
echo "   总重试patch数: $total_patches"
echo "   成功处理: $completed_patches"
echo "   处理失败: $failed_patches"
echo "   总用时: ${overall_hours}小时${overall_minutes}分钟"
if [[ $total_patches -gt 0 ]]; then
  echo "   成功率: $(( completed_patches * 100 / total_patches ))%"
  echo "   平均处理时间: $(( overall_duration / total_patches ))秒/patch"
fi
echo "   结束时间: $(date)"
echo ""
echo "📁 输出目录: $OUTPUT_DIR"
echo "📄 日志目录: $LOG_DIR"

# 生成重新处理汇总日志
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

log_info "重新处理汇总日志已保存: $retry_summary_log"

# 根据结果设置退出码
if [[ $failed_patches -gt 0 ]]; then
  log_info "有 $failed_patches 个patch重新处理失败"
  if [[ $completed_patches -eq 0 ]]; then
    exit 1  # 全部失败
  else
    exit 2  # 部分失败
  fi
fi

log_info "所有失败patches重新处理成功！"
exit 0