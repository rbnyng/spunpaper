import os
import glob

def find_failed_logs(logs_dir):
    """
    遍历logs目录，找出处理失败的log文件对应的id
    
    Args:
        logs_dir: log文件所在目录
    
    Returns:
        list: 失败的id列表
    """
    failed_ids = []
    
    # 获取所有.log文件
    log_pattern = os.path.join(logs_dir, "*.log")
    log_files = glob.glob(log_pattern)
    
    print(f"找到 {len(log_files)} 个log文件，开始检查...")
    
    for log_file in log_files:
        try:
            # 读取log文件内容
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含成功标志
            if "分区处理完成: 成功" not in content:
                # 提取id（去掉路径和.log后缀）
                filename = os.path.basename(log_file)
                id_name = filename.replace('.log', '')
                failed_ids.append(id_name)
                
        except Exception as e:
            print(f"读取文件 {log_file} 时出错: {e}")
            # 如果文件读取失败，也认为是处理失败
            filename = os.path.basename(log_file)
            id_name = filename.replace('.log', '')
            failed_ids.append(id_name)
    
    return failed_ids

def main():
    # 设置log目录路径
    logs_dir = "/scratch/zf281/create_d-pixels_biomassters/data/test_agbm_d-pixel/logs_s2"
    
    # 检查目录是否存在
    if not os.path.exists(logs_dir):
        print(f"错误: 目录不存在 - {logs_dir}")
        return
    
    # 查找失败的log
    failed_ids = find_failed_logs(logs_dir)
    
    # 打印结果
    print("\n" + "="*50)
    if failed_ids:
        print("处理失败的ID列表:")
        print("-"*30)
        for id_name in failed_ids:
            print(id_name)
        print(f"\n总共有 {len(failed_ids)} 个处理失败的任务")
    else:
        print("🎉 所有任务都处理成功！")
    print("="*50)

if __name__ == "__main__":
    main()