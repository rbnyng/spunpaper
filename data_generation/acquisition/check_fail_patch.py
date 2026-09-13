import os
import glob
import paths

def find_failed_logs(logs_dir):
    """Return the IDs of the runs whose log lacks the success marker."""
    failed_ids = []
    
    log_pattern = os.path.join(logs_dir, "*.log")
    log_files = glob.glob(log_pattern)
    
    print(f"Found {len(log_files)} log files, checking...")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "Partition complete: succeeded" not in content:
                filename = os.path.basename(log_file)
                id_name = filename.replace('.log', '')
                failed_ids.append(id_name)
                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            # An unreadable file counts as a failure
            filename = os.path.basename(log_file)
            id_name = filename.replace('.log', '')
            failed_ids.append(id_name)
    
    return failed_ids

def main():
    logs_dir = paths.ACQ_LOG_DIR
    
    if not os.path.exists(logs_dir):
        print(f"Error: directory not found - {logs_dir}")
        return
    
    failed_ids = find_failed_logs(logs_dir)
    
    print("\n" + "="*50)
    if failed_ids:
        print("Failed IDs:")
        print("-"*30)
        for id_name in failed_ids:
            print(id_name)
        print(f"\n{len(failed_ids)} tasks failed in total")
    else:
        print("🎉 All tasks completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()