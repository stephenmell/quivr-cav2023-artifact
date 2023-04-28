import sys
import json

_, task_list_name, torch_device = sys.argv

with open(f"task_lists/{task_list_name}", "r") as f:
    tasks_list = json.load(f)

from config import *
set_cfg(
    TORCH_DEVICE = torch_device,
    PARAMETER_SEARCH_METHOD = None,
)

from misc_utils import *
from torch_utils import *
import experiment_utils
from tqdm import tqdm
import numpy as np
import math


table_latex_lines = []
all_totals = {}

for dataset_name, task_name, task_id in tasks_list:
    dataset = torch_load(dataset_path(dataset_name))
    task = torch_load(task_path(dataset_name, task_name))

    time_totals = {}
    all_totals[dataset_name, task_name] = time_totals

    for dev_type_str in ["gpu", "cpu"]:
        for parameter_search_method in ["quantitative", "binary"]:
            cur_time_totals = []
            time_totals[dev_type_str, parameter_search_method] = cur_time_totals

            for timing_argument in ["0", "1", "2"]:
                time_total = 0.
                for i in range(25):
                    try:
                        time_total += torch_load(
                            al_time_path(
                                dataset_name,
                                task_name,
                                dev_type_str,
                                parameter_search_method,
                                str(timing_argument),
                                str(i),
                            )
                        )
                    except Exception as e:
                        print(e)
                        print("Skipping...")
                        time_total = None
                        break
                if time_total is not None:
                    cur_time_totals.append(time_total)

    cells = []

    cells.append(task_id)

    for t in [
        time_totals["cpu", "binary"],
        time_totals["cpu", "quantitative"],
        time_totals["gpu", "binary"],
        time_totals["gpu", "quantitative"],
    ]:
        avg = np.mean(t)
        dev = np.std(t)
        err = dev/math.sqrt(len(t))
        if math.isnan(avg):
            cells.append("--")
        else:
            cells.append(f"${avg:,.0f}$" + "\\hspace{0.03in} $\pm$")
        if math.isnan(err):
            cells.append("--")
        else:
            cells.append(f"${err:,.0f}$" + " \\hspace{0.10in}")

    table_latex_lines.append(
        " & ".join(cells)
    )


out_path = f"tex_output/{task_list_name}_timing"
with open(out_path, "w") as f:
    for line in table_latex_lines:
        f.write(line + " \\\\\n")

if False:
    10408/737.
    2880/110.
    2272/110

    gpu_quant_tot = 174+110+113+119+50+60+30+32+23+36+30+57+56+264+127+167+183
    gpu_bin_tot = 737+428+376+370+225+285+219+185+163+227+163+252+245+1393+811+970+1141
    gpu_tot = gpu_quant_tot + gpu_bin_tot
    gpu_quant_tot/60/60
    gpu_bin_tot/60/60

    ks = [
        ("cpu", "binary"),
        ("cpu", "quantitative"),
        ("gpu", "binary"),
        ("gpu", "quantitative"),
    ]
    averages = {k: [] for k in ks}

    for (dataset_name, task_name), totals in all_totals.items():
        for k in ks:
            averages[k].append(np.mean(totals[k]))
    
    overall_avgs = {k: np.mean(v) for k, v in averages.items()}
    # overall_avgs = {k: np.median(v) for k, v in averages.items()}

    cpu_b2q = overall_avgs["cpu", "binary"] / overall_avgs["cpu", "quantitative"]
    gpu_b2q = overall_avgs["gpu", "binary"] / overall_avgs["gpu", "quantitative"]
    bin_c2g = overall_avgs["cpu", "binary"] / overall_avgs["gpu", "binary"]
    quant_c2g = overall_avgs["cpu", "quantitative"] / overall_avgs["gpu", "quantitative"]
    tot = overall_avgs["cpu", "binary"] / overall_avgs["gpu", "quantitative"]

    print(f"bin, CPU to GPU: {bin_c2g:.1f}")
    print(f"GPU, bin to quant: {gpu_b2q:.1f}")
    # print(f"CPU, bin to quant: {cpu_b2q:.1f}")
    # print(f"quant, CPU to GPU: {quant_c2g:.1f}")