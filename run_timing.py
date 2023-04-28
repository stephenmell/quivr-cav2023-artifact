import sys
import json
import subprocess

def sysrun(exec, *args):
    print("Executing:", exec, *args)
    return subprocess.run([exec, *args], check = True)

_, task_list_name, torch_cpu_device, torch_gpu_device = sys.argv
assert torch_cpu_device in {"cpu", "skip"}
assert torch_gpu_device != "cpu"

with open(f"task_lists/{task_list_name}", "r") as f:
    tasks_list = json.load(f)

for timing_argument in ["0", "1", "2"]:
    for dataset_name, task_name, task_id in tasks_list:
        for dev in [torch_cpu_device, torch_gpu_device]:
            if dev == "skip":
                continue
            for parameter_search_method in ["quantitative", "binary"]:
                sysrun("python", "quivr/do_active_learning.py", dataset_name, task_name, parameter_search_method, timing_argument, dev, "smart")