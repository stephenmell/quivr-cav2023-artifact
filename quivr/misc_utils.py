import sys
import os.path
import subprocess
import os
import numpy as np

def fold_taking_first(op, iter):
    ret = None
    have_first = False
    for x in iter:
        if not have_first:
            have_first = True
            ret = x
        else:
            ret = op(ret, x)
    assert have_first
    return ret

def data_path(path = None):
    run_dir = os.path.dirname(sys.argv[0])
    data_dir = "./"

    if path is None:
        return data_dir
    else:
        return f"{data_dir}/{path}"

def list_in_dir_with_prefix_suffix(dir, prefix = "", suffix = ""):
    if len(suffix) > 0:
        return tuple(
            x[len(prefix):-len(suffix)]
            for x in os.listdir(dir)
            if x.startswith(prefix) and x.endswith(suffix)
        )
    else:
        return tuple(
            x[len(prefix):]
            for x in os.listdir(dir)
            if x.startswith(prefix)
        )

def dataset_path(dataset):
    return data_path(f"datasets/{dataset}/data.torch")

def task_path(dataset, task):
    return data_path(f"datasets/{dataset}/task_{task}/task.torch")

def al_state_path(dataset, task, active_choice_method_name, index):
    return data_path(f"quivr_results/{dataset}/task_{task}/al_state_{active_choice_method_name}_{index}.torch")

def al_time_path(
        dataset,
        task, 
        dev_type,
        parameter_search_method,
        iter,
        index,
    ):
    return data_path(f"quivr_results/{dataset}/task_{task}/al_time_{dev_type}_{parameter_search_method}_{iter}_{index}")

# ways of splitting n into k numbers that sum to n
def integer_partition(k, n):
    assert k == 2
    for i in range(n + 1):
        yield (i, n - i)

# ways of splitting ns into k tuples that sum (component-wise) to ns
import itertools
def integer_tuple_partition(k, ns):
    return itertools.product(*(integer_partition(k, n) for n in ns))

# takes expr, start_index; returns new expr, list of pred1 names
def fill_holes_with_indices(expr, start_index):
    index = start_index
    match expr:
        case ("PRED0", pred):
            ret_expr = ("PRED0", pred)
            ret_names = ()
        
        case ("PRED1", pred, name):
            assert name is None
            ret_expr = ("PRED1", pred, index)
            ret_names = (pred,)
        
        case ("SEQ", subexprs):
            subexprs_so_far = []
            names_so_far = []
            for subexpr in subexprs:
                subexpr_new, names_new = fill_holes_with_indices(subexpr, start_index + len(names_so_far))
                subexprs_so_far.append(subexpr_new)
                names_so_far.extend(names_new)
            ret_expr = ("SEQ", tuple(subexprs_so_far))
            ret_names = tuple(names_so_far)
        
        case ("CONJ", subexprs):
            subexprs_so_far = []
            names_so_far = []
            for subexpr in subexprs:
                subexpr_new, names_new = fill_holes_with_indices(subexpr, start_index + len(names_so_far))
                subexprs_so_far.append(subexpr_new)
                names_so_far.extend(names_new)
            ret_expr = ("CONJ", tuple(subexprs_so_far))
            ret_names = tuple(names_so_far)

        case ("DISJ", subexprs):
            subexprs_so_far = []
            names_so_far = []
            for subexpr in subexprs:
                subexpr_new, names_new = fill_holes_with_indices(subexpr, start_index + len(names_so_far))
                subexprs_so_far.append(subexpr_new)
                names_so_far.extend(names_new)
            ret_expr = ("DISJ", tuple(subexprs_so_far))
            ret_names = tuple(names_so_far)
    
    return ret_expr, ret_names


# takes expr, start_index; returns new expr, list of pred1 names
def expr_blank_out_indices(expr):
    match expr:
        case ("PRED0", pred):
            return ("PRED0", pred)
        
        case ("PRED1", pred, name):
            return ("PRED1", pred, None)
        
        case ("SEQ", subexprs):
            return ("SEQ", tuple(expr_blank_out_indices(subexpr) for subexpr in subexprs))
        
        case ("CONJ", subexprs):
            return ("CONJ", tuple(expr_blank_out_indices(subexpr) for subexpr in subexprs))

        case ("DISJ", subexprs):
            return ("DISJ", tuple(expr_blank_out_indices(subexpr) for subexpr in subexprs))

def trace_shape(trace):
    ret = None
    for pred in trace.values():
        if ret is None:
            ret = pred.shape
        else:
            assert ret == pred.shape
    return ret

def index_trace_mask(traces, mask):
    return {k: v[mask, :, :] for k, v in traces.items()}
    
def list_remove_by_id(l, x):
    for i, y in enumerate(l):
        if y is x:
            l.pop(i)
            return
    assert False

def sysrun(exec, *args):
    return subprocess.run([exec, *args], check = True)

def maybe_percentile(a, p):
    if len(a) == 0:
        return float("NaN")
    else:
        return np.percentile(a, p)