import sys
import json
import subprocess

def sysrun(exec, *args):
    print("Executing:", exec, *args)
    return subprocess.run([exec, *args], check = True)

_, task_list_name, torch_device = sys.argv

with open(f"task_lists/{task_list_name}", "r") as f:
    tasks_list = json.load(f)

for dataset_name in set(dataset_name for dataset_name, _task_name, _task_id in tasks_list):
    # lstm pretraining
    sysrun("python", "lstm/train.py", dataset_name, torch_device)

for dataset_name, task_name, task_id in tasks_list:
    # smart
    sysrun("python", "quivr/do_active_learning.py", dataset_name, task_name, "quantitative", "no", torch_device, "smart")

    # random
    sysrun("python", "quivr/do_active_learning.py", dataset_name, task_name, "quantitative", "no", torch_device, "random")

    # lstm
    sysrun("python", "lstm/eval.py", dataset_name, task_name, torch_device)

    # transformer
    sysrun("python", "transformer/run_transformer.py", dataset_name, task_name, "train", torch_device)
    sysrun("python", "transformer/run_transformer.py", dataset_name, task_name, "eval", torch_device)