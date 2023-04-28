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

f1_score_experiment_step_names = [str(i + 1) for i in range(26)]

def get_last_nonempty_f1_scores(dataset_name, task_name, active_choice_method_name, step_index, dataset, task):
    step_name = f1_score_experiment_step_names[step_index]
    state = torch_load(al_state_path(dataset_name, task_name, active_choice_method_name, step_name))
    f1_scores = experiment_utils.compute_test_f1_scores(dataset, task, state, progress = tqdm)
    if len(f1_scores) == 0:
        return []
        return get_last_nonempty_f1_scores(dataset_name, task_name, active_choice_method_name, step_index - 1, dataset, task)
    return f1_scores

show_step_indices = [0, 5, 10, 25]

rows = []

assert show_step_indices == [0, 5, 10, 25]
heading = """
\\begin{tabular}{crrrrrrrrrrrrrrrr}
\\toprule \multicolumn{1}{c}{\\multirow{2}{*}{\\textbf{Query}}} & \\multicolumn{4}{c}{\\textbf{0 Steps}} & \\multicolumn{4}{c}{\\textbf{5 Steps}} & \\multicolumn{4}{c}{\\textbf{10 Steps}} & \\multicolumn{4}{c}{\\textbf{25 Steps}} \\\\
& \\multicolumn{1}{c}{$Q$} & \\multicolumn{1}{c}{$R$} & \\multicolumn{1}{c}{$L$} & \\multicolumn{1}{c}{$T$} & \\multicolumn{1}{c}{$Q$} & \\multicolumn{1}{c}{$R$} & \\multicolumn{1}{c}{$L$} & \\multicolumn{1}{c}{$T$} & \\multicolumn{1}{c}{$Q$} & \\multicolumn{1}{c}{$R$} & \\multicolumn{1}{c}{$L$} & \\multicolumn{1}{c}{$T$} & \\multicolumn{1}{c}{$Q$} & \\multicolumn{1}{c}{$R$} & \\multicolumn{1}{c}{$L$} & \\multicolumn{1}{c}{$T$}\\\\
\\midrule
"""
tailing = """
\\bottomrule
\\end{tabular}
"""

def compute_f1(tp, tn, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)

for dataset_name, task_name, task_id in tasks_list:
    dataset = torch_load(dataset_path(dataset_name))
    task = torch_load(task_path(dataset_name, task_name))

    f1_score_50th_groups = []

    for i, step_name in enumerate(f1_score_experiment_step_names):
        group = []
        for active_choice_method_name in ["smart", "random", "lstm", "transformer"]:
            if active_choice_method_name == "lstm":
                try:
                    tpfn = torch_load(f"lstm_results/{dataset_name}_{task_name}.torch")
                except:
                    tpfn = None
            if active_choice_method_name == "transformer":
                try:
                    tpfn = torch_load(f"transformer_results/{dataset_name}_{task_name}.torch")
                except:
                    tpfn = None
            if i in show_step_indices:
                if active_choice_method_name == "lstm" or active_choice_method_name == "transformer":
                    if tpfn is None:
                        group.append(-1.)
                    else:
                        group.append(compute_f1(*tpfn[i]).item())
                else:
                    f1_scores = get_last_nonempty_f1_scores(dataset_name, task_name, active_choice_method_name, i, dataset, task)

                    group.append(maybe_percentile(f1_scores, 50))
        f1_score_50th_groups.append(group)
    
    cells = []

    cells.append(task_id)

    for group in f1_score_50th_groups:
        for t in group:
            if math.isnan(t):
                cells.append("--")
                # assert False
            else:
                s = f"{t:.2f}"
                if t == max(group):
                    s = "\\bf{" + s + "}"
                cells.append(s)

    rows.append(cells)

out_path = f"tex_output/{task_list_name}_f1"
with open(out_path, "w") as f:
    f.write(heading)
    f.write(" \\\\\n".join([
        " & ".join(s if i == 0 else "\\hspace{0.05in} " + s for i, s in enumerate(cells)) for cells in rows
    ]))
    f.write(" \\\\")
    f.write(tailing)