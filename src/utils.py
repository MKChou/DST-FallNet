import psutil
import torch
import gc

def get_system_usage():
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        return f"{mem_percent:.1f}%", f"{cpu_percent:.1f}%"
    except Exception as e:
        print(f"Failed to get system resource usage: {str(e)}")
        return "81.8%", "0.0%"

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

def progress_bar(score, length=15, color=None):
    normalized_value = min(1.0, max(0.0, score))
    height = int(normalized_value * length)
    bar = '█' * height + '░' * (length - height)
    if color is None:
        if score >= 0.8:
            color = RED
        elif score >= 0.5:
            color = YELLOW
        else:
            color = GREEN
    return f"{color}{bar}   {score:.3f}{RESET}"

def create_vertical_bar(value, max_height=10):
    normalized_value = min(1.0, max(0.0, value))
    height = int(normalized_value * max_height)
    bar = '█' * height + '░' * (max_height - height)
    return bar 