import sys

if "get_ipython" in globals():
    dataset_name = "mabe22"
    task_name = "approach"
    parameter_search_method = "quantitative"
    timing_argument = "0"
    torch_device = "cuda:0"

else:
    assert len(sys.argv) == 7
    dataset_name = sys.argv[1]
    task_name = sys.argv[2]
    parameter_search_method = sys.argv[3]
    timing_argument = sys.argv[4]
    torch_device = sys.argv[5]
    active_choice_method_name = sys.argv[6]

from config import *
set_cfg(
    TORCH_DEVICE = torch_device,
    PARAMETER_SEARCH_METHOD = parameter_search_method,
)

if timing_argument == "no":
    is_timing_run = False
else:
    is_timing_run = True
    if torch_device == "cpu":
        dev_type_str = "cpu"
    elif torch_device.startswith("cuda"):
        dev_type_str = "gpu"
    else:
        assert False, torch_device
    timing_iter = int(timing_argument)


import experiment_utils
if active_choice_method_name == "smart":
    active_choice_method = experiment_utils.do_smart_choosing
elif active_choice_method_name == "random":
    active_choice_method = experiment_utils.do_random_choosing
    assert not is_timing_run
else:
    assert False, active_choice_method_name


if dataset_name == "maritime_surveillance":
    cfg_wrap_with_anything = False
else:
    cfg_wrap_with_anything = True

cfg_max_preds = 3
cfg_max_pred1s = 2




from misc_utils import *
from torch_utils import *
from tqdm import tqdm
import enumeration
import time

dataset = torch_load(dataset_path(dataset_name))
task = torch_load(task_path(dataset_name, task_name))

def enumerator(pred0_exprs, pred1_exprs):
    enumerated = {
        (0, 0): frozenset(),
        (1, 0): pred0_exprs,
        (0, 1): pred1_exprs,
    }
    for i in range(cfg_max_pred1s + 1):
        enumeration.fill_up_to(enumerated, cfg_max_preds - i, i)

    if cfg_wrap_with_anything:
        return frozenset(
            ("SEQ", (
                ("PRED0", "anything"),
                e,
                ("PRED0", "anything"),
            ))
            for k, v in enumerated.items()
            for e in v
        )
    else:
        return frozenset(e for k, v in enumerated.items() for e in v)

time_prev = time.perf_counter()
for i in range(27):
    if i == 0:
        experiment_state = experiment_utils.initial_enumerative(dataset, task, enumerator)
    else:
        experiment_state = experiment_utils.do_filtering(dataset, task, experiment_state, progress = tqdm)
        experiment_state = active_choice_method(dataset, task, experiment_state, progress = tqdm)
    if not is_timing_run:
        torch_save_mkdirs(experiment_state, al_state_path(dataset_name, task_name, active_choice_method_name, str(i)))
    else:
        time_next = time.perf_counter()
        time_delta = time_next - time_prev
        time_prev = time_next
        torch_save_mkdirs(
            time_delta,
            al_time_path(
                dataset_name,
                task_name,
                dev_type_str,
                parameter_search_method,
                str(timing_iter),
                str(i),
            )
        )