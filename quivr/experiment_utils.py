import enumeration
from torch_utils import *
from misc_utils import *
import parameter_search
from matrix_semantics import *
import sklearn.metrics
import random

# we need padding to ensure that the ground truth never literally lies on the boundary
# consider CONJ(t >= theta, P). in order to classify anything as positive,
# theta has to be 300, so we need its upper bound to be 300 + SMALL
PADDING = 1.

def initialize_enumerative_inner(pred0, pred1_bounds, enumerator):
    pred0_exprs = tuple(
        ("PRED0", k)
        for k in sorted(pred0)
    )

    pred1_exprs = tuple(
        ("PRED1", k, None)
        for k in sorted(pred1_bounds.keys())
    )

    all_queries = []
    for expr_with_holes in enumerator(pred0_exprs, pred1_exprs):
        expr, pred1_names = fill_holes_with_indices(expr_with_holes, 0)
        if len(pred1_names) > 0:
            initial_bound = torch.stack(tuple(
                torch_literal(pred1_bounds[pred1_name])
                for pred1_name in pred1_names
            ), axis = -2)
            initial_bound[..., 0] -= PADDING
            initial_bound[..., 1] += PADDING
        else:
            initial_bound = torch_zeros((0, 2))

        all_queries.append((expr, initial_bound))
    
    return tuple(
        (expr, (), (initial_bound,)) for expr, initial_bound in all_queries
    )

def initial_enumerative(dataset, task, enumerator):
    pred0_names = tuple(
        k
        for k, v in dataset["traces"].items()
        if v.dtype == torch.bool
    )
    return {
        "unpruned_queries": initialize_enumerative_inner(
            pred0_names,
            dataset["pred1_bounds"],
            enumerator,
        ),
        "error_queries": (),
        "labeled_indices": task["labeled_initial_indices"],
    }

def expand_until_sat_or_timeout(traces, labels, expr, initial_sat, initial_unk, max_iters, pick_fn = parameter_search.pick_biggest):
    match CFG["PARAMETER_SEARCH_METHOD"]:
        case "quantitative":
            def compute_theta_pn(d):
                return parameter_search.compute_theta_pn_quantitative_semantics
        case "binary":
            def compute_theta_pn(d):
                precision_tensor = torch_empty((d,))
                precision_tensor[...] = CFG["PARAMETER_SEARCH_BINARY_PRECISION"]
                return parameter_search.compute_theta_pn_binary_search(precision_tensor)
        case x:
            assert False, x

    sat_boxes = list(initial_sat)
    unk_boxes = list(initial_unk)

    n_iters = 0
    while len(sat_boxes) == 0 and len(unk_boxes) != 0:
        next_box = pick_fn(unk_boxes)
        if n_iters >= max_iters:
            break

        # print("picked", next_box_vol_frac)
        # print("picked")
        # print(expr)
        # print(next_box)
        list_remove_by_id(unk_boxes, next_box)

        if next_box.shape[-2] == 0:
            # this is the zero-dimensional case
            # TODO: unify it with the other dimensional cases
            traces_pos = index_trace_mask(traces, labels)
            traces_neg = index_trace_mask(traces, ~labels)
            sat_pos = whole_trace_eval_binary(traces_pos, next_box[..., 0], expr).all()
            sat_neg = (~whole_trace_eval_binary(traces_neg, next_box[..., 0], expr)).all()
            if sat_pos and sat_neg:
                new_sat_boxes = (next_box,)
            else:
                new_sat_boxes = ()
            new_unk_boxes = ()
        else:
            theta_p, theta_n = compute_theta_pn(next_box.shape[-2])(traces, labels, next_box, expr)
            new_sat_boxes, new_unk_boxes = parameter_search.do_expand_box(traces, labels, expr, next_box, theta_p, theta_n)
        # print("new unks")
        # for b in new_unk_boxes:
        #     print(b)
        # if new_sat_boxes:
        #     print("new sats")
        #     for b in new_sat_boxes:
        #         print(b)
        sat_boxes.extend(new_sat_boxes)
        unk_boxes.extend(new_unk_boxes)
        n_iters += 1
    
    return sat_boxes, unk_boxes

def do_filtering(dataset, task, experiment_state, progress = lambda x: x):
    labeled_traces = {
        k: v if v.shape[0] == 1 else v[experiment_state["labeled_indices"]]
        for k, v in dataset["traces"].items()
    }
    labeled_labels = task["labels"][experiment_state["labeled_indices"]]

    # continue pruning with new dataset
    search_res = {}
    error_res = {}
    for expr, sat_boxes, unk_boxes in progress(experiment_state["unpruned_queries"]):
        # TODO: sat boxes should become unk in the labeling step, not the filtering step
        new_unk = sat_boxes + unk_boxes
        # print(expr)
        # print(initial_bound)
        if len(new_unk) == 0:
            search_res[expr] = (), ()
        else:
    #        try:
            if True:
                search_res[expr] = expand_until_sat_or_timeout(
                    labeled_traces,
                    labeled_labels,
                    expr,
                    (),
                    new_unk,
                    5,
                )
    #        except Exception as e:
    #            error_res[expr] = (sat_boxes, unk_boxes, e)
    #            print("failure at", expr, e)
    return {
        "unpruned_queries": tuple(
            (expr, sat, unk) for expr, (sat, unk) in search_res.items()
        ),
        # "error_queries": experiment_state["error_queries"] + tuple(
        #     (expr, sat_boxes, unk_boxes, e)
        #     for expr, (sat_boxes, unk_boxes, e) in error_res.items()
        # ),
        "labeled_indices": experiment_state["labeled_indices"],
    }


def do_smart_choosing(dataset, task, experiment_state, progress = lambda x: x):
    dataset_size = trace_shape(dataset["traces"])
    unlabeled_mask = (
        ~tensor_indices_to_mask(dataset_size[0], experiment_state["labeled_indices"]) &
        tensor_indices_to_mask(dataset_size[0], dataset["train_indices"])
    )
    unlabeled_traces = {
        k: v if v.shape[0] == 1 else v[unlabeled_mask]
        for k, v in dataset["traces"].items()
    }

    
    # decide which trajectory to label
    concrete_queries = queryboxes_to_queries(experiment_state["unpruned_queries"])

    cumulative_preds = torch_zeros((trace_shape(unlabeled_traces)[0],))
    for i, (expr, param) in enumerate(progress(concrete_queries)):
        semant = whole_trace_eval_trinary(unlabeled_traces, param, expr)

        preds = semant >= 0.

        cumulative_preds += preds.int()

    fractional_preds = cumulative_preds/len(concrete_queries)
    trace_to_label_localidx = torch.min(torch.abs(fractional_preds - 0.5), axis = 0).indices.item()
    trace_to_label_globalidx = unlabeled_mask.nonzero()[trace_to_label_localidx].item()

    # this is needed because torch doesn't play nice with the gc...
    unlabeled_traces.clear()
    return {
        "unpruned_queries": experiment_state["unpruned_queries"],
        "labeled_indices": torch.concat((
            experiment_state["labeled_indices"],
            torch_literal([trace_to_label_globalidx])
        ), axis = 0),
    }

def do_random_choosing(dataset, task, experiment_state, progress = lambda x: x):
    dataset_size = trace_shape(dataset["traces"])
    unlabeled_mask = (
        ~tensor_indices_to_mask(dataset_size[0], experiment_state["labeled_indices"]) &
        tensor_indices_to_mask(dataset_size[0], dataset["train_indices"])
    )
    unlabeled_indices = unlabeled_mask.nonzero()
    (n, _) = unlabeled_indices.shape

    rand = random.Random()
    rand.seed(len(experiment_state["labeled_indices"]))
    trace_to_label_localidx = rand.randint(0, n - 1)
    trace_to_label_globalidx = unlabeled_mask.nonzero()[trace_to_label_localidx].item()

    return {
        "unpruned_queries": experiment_state["unpruned_queries"],
        "labeled_indices": torch.concat((
            experiment_state["labeled_indices"],
            torch_literal([trace_to_label_globalidx])
        ), axis = 0),
    }

def queryboxes_to_queries(queryboxes):
    concrete_queries = []
    for expr, sat_boxes, unk_boxes in queryboxes:
        if len(sat_boxes) > 0:
            sat_box = sat_boxes[0]
            concrete_queries.append((expr, (sat_box[..., 0] + sat_box[..., 1])/2))
    return tuple(concrete_queries)

def compute_test_f1_scores(dataset, task, al_state, progress = lambda x: x):
    test_traces = {
        k: v if v.shape[0] == 1 else v[dataset["test_indices"]]
        for k, v in dataset["traces"].items()
    }
    test_labels = task["labels"][dataset["test_indices"]]

    concrete_queries = queryboxes_to_queries(al_state["unpruned_queries"])

    return tuple(
        sklearn.metrics.f1_score(test_labels.cpu(), whole_trace_eval_binary(test_traces, param, expr).cpu())
        for expr, param in progress(concrete_queries)
    )