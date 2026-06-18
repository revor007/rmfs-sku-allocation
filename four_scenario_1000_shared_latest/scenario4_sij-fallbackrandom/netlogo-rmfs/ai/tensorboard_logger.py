import os
import json
import time
from glob import glob
from torch.utils.tensorboard import SummaryWriter

# Define the folder and writer
log_folder = os.path.join('mappo_routing', 'alr-0.0005_clr-0.0005_lambd0-0.9_lambdaplha-1_gamma-0.9')
tensorboard_log_dir = os.path.join(log_folder, 'tensorboard_logs')
writer = SummaryWriter(tensorboard_log_dir)

file_positions = {}  # Track how much of each file has been read
global_step = 0      # Global line/step counter across all files

def get_sorted_jsonl_files():
    files = glob(os.path.join(log_folder, "log_*.jsonl"))
    return sorted(files, key=os.path.getmtime)

def update_tensorboard():
    global global_step
    sorted_files = get_sorted_jsonl_files()

    for file_path in sorted_files:
        if file_path not in file_positions:
            file_positions[file_path] = 0

        with open(file_path, "r") as f:
            f.seek(file_positions[file_path])
            lines = f.readlines()
            file_positions[file_path] = f.tell()

        for line in lines:
            try:
                data = json.loads(line)
                reward = float(data["total_reward"])
                writer.add_scalar("Total Reward", reward, global_step)
                global_step += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

if __name__ == "__main__":
    print("Monitoring log files (logging by line count)...")
    try:
        while True:
            update_tensorboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        writer.close()
