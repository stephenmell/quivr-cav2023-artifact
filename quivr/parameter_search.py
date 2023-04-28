import torch
from maxmin_utils import *
from misc_utils import *
from matrix_semantics import *
from box_utils import *
from torch_utils import *

from itertools import product
make_quadrants = lambda param_d: torch_literal(tuple(product(set((True, False)), repeat = param_d))[1:-1])

def result_on_dataset(traces, labels, expr, param):
    preds = whole_trace_eval_robustness(traces, param, expr)

    pos_all_satisfied = torch.min(preds[labels], -1).values
    neg_all_satisfied = torch.min(-preds[~labels], -1).values

    return torch.minimum(pos_all_satisfied, neg_all_satisfied)

def check_result(traces, labels, expr, bound):
    a = result_on_dataset(traces, labels, expr, -0.1 * bound[..., 0] + 1.1 * bound[..., 1])
    b = result_on_dataset(traces, labels, expr,  0.0 * bound[..., 0] + 1.0 * bound[..., 1])
    c = result_on_dataset(traces, labels, expr,  0.5 * bound[..., 0] + 0.5 * bound[..., 1])
    d = result_on_dataset(traces, labels, expr,  1.0 * bound[..., 0] + 0.0 * bound[..., 1])
    e = result_on_dataset(traces, labels, expr,  1.1 * bound[..., 0] - 0.1 * bound[..., 1])
    # FIXME: probably we shouldn't be using approximate comparisons. think about discrete lattices
    if not (
        tensor_approx_lteq(a, torch_literal(0.)).all() and
        tensor_approx_eq(b, torch_literal(0.)).all() and
        tensor_approx_gteq(c, torch_literal(0.)).all() and
        tensor_approx_eq(d, torch_literal(0.)).all() and
        tensor_approx_lteq(e, torch_literal(0.)).all()
    ):
        print("WARNING, check_result failed")
        print("lower_bound", bound[..., 0])
        print("upper_bound", bound[..., 1])
        print("a", a, -1)
        print("b", b, 0)
        print("c", c, 1)
        print("d", d, 0)
        print("e", e, -1)
        # assert False

# traces is m x n x n
# labels is m
# expr has d parameters
# bounds is d x 2
def do_expand_box(traces, labels, expr, bound, theta_p, theta_n):
    # if either satisfaction point is outside the bounds we have, then we're unsat
    # these are exact comparisons because otherwise `inner` could end up slightly
    # outside `bound`, which will is an error for `fracture_box_with_box`
    if (theta_p <= bound[..., 0]).any() or (theta_n >= bound[..., 1]).any():
        # it's not an error per se, but if t_p > t_n, that means we're sat, but it's
        # occurring outside of `bound`
        # assert (theta_p <= theta_n).all(), (theta_p, theta_n)
        # FIXME: some floating point issue
        if not (theta_p - theta_n <= 0.0001).all():
            print("WARNING jyer:",(theta_p, theta_n))
        return (), ()
    else:
        # for t_p = t_n, it doesn't matter whether we consider it sat, since the
        # inner and face pieces will have volume zero, and corner pieces are
        # unk regardless of t_p vs t_n
        inner_is_sat = (theta_p > theta_n).all()
        inner = torch.stack(
            (theta_n, theta_p) if inner_is_sat else (theta_p, theta_n),
            axis = -1,
        )
        ib, fb, cb = fracture_box_with_box(bound, inner)
        # we exclude the first (the low corner) and the last (the high corner)
        cb_except_two = cb[1:-1]
        if inner_is_sat:
            sat = tuple(filter_empty_boxes(ib))
            unk = tuple(filter_empty_boxes(cb_except_two)) + tuple(filter_empty_boxes(fb))
            if CFG["DEBUG"]:
                # TODO: we only need traces, labels, expr for this call, so we could move the check outside of this function
                check_result(traces, labels, expr, inner)
                assert len(sat) == 1 
            return sat, unk
        else:
            return (), tuple(filter_empty_boxes(cb_except_two))

pick_biggest = lambda unk_boxes: sorted(unk_boxes, key = compute_volume, reverse = True)[0]
pick_smallest = lambda unk_boxes: sorted(unk_boxes, key = compute_volume, reverse = False)[0]

def point_is_lower_bound_theta_p(traces, expr):
    def f(midpoint):
        res = whole_trace_eval_binary(traces, midpoint, expr)
        if res.all():
            # if they're all positive, we could be a bit more restrictive
            # bigger is more restrictive, so we increase the lower bound
            return True
        else:
            # if something is negative, we must be less restrictive
            # smaller is less restrictive, so we decrease the upper bound
            return False
    return f

def point_is_lower_bound_theta_n(traces, expr):
    def f(midpoint):
        res = whole_trace_eval_binary(traces, midpoint, expr)
        if res.any():
            # if any of them is negative, we must be more restrictive
            # bigger is more restrictive, so we increase the lower bound
            return True
        else:
            # if they're all negative, could be a bit less restrictive
            # smaller is less restrictive, so we decrease the upper bound
            return False
    return f

# computes the point at which, along the diagonal of `bound`, up to `precision`, all traces are positive
# and one is marginally so. going more negative makes the more things positive looser, so we want the max
# this is good for computing theta_p
_TRACE_SEARCH_STEPS = []
def binary_search_find_threshold(point_is_lower_bound, precision, bound):
    cover = torch_empty(bound.shape)
    cover[..., 0] = float("-inf")
    cover[..., 1] = float("inf")

    i = 0
    def prune_cover(x):
        if CFG["DEBUG"]:
            assert (cover[..., 0] <= x).all()
            assert (cover[..., 1] >= x).all()
        
        if point_is_lower_bound(x):
            cover[..., 0] = x
        else:
            cover[..., 1] = x
        nonlocal i
        i += 1

    while True:
        intersection_lower = torch.maximum(cover[..., 0], bound[..., 0])
        intersection_upper = torch.minimum(cover[..., 1], bound[..., 1])

        if (intersection_upper - intersection_lower <= precision).all():
            break

        prune_cover((intersection_lower + intersection_upper)/2)

    if (cover[..., 0] < bound[..., 0]).any():
        if CFG["DEBUG"]:
            print("testing lower bound")
        prune_cover(bound[..., 0])

    if (cover[..., 1] > bound[..., 1]).any():
        if CFG["DEBUG"]:
            print("testing upper bound")
        prune_cover(bound[..., 1])
    _TRACE_SEARCH_STEPS.append(i)
    if (cover[..., 0] < bound[..., 0]).any():
        return ("below",)
    elif (cover[..., 1] > bound[..., 1]).any():
        return ("above",)
    else:
        return ("contained", cover)

if False:
    gt = torch_literal([2., 1., 3.])
    prec = torch_literal([.1, .1, .1])
    bound = torch_literal([
        [0., 6.],
        [0., 3.],
        [0., 9.],
    ])
    def gt(x):
        r = (x <= 0.001).all()
        print("qurey", x, r)
        return r

    binary_search_find_threshold(gt, _prec, bound)

def compute_theta_pn_binary_search(precision):
    def f(traces, labels, bound, expr):
        traces_pos = index_trace_mask(traces, labels)
        traces_neg = index_trace_mask(traces, ~labels)

        match binary_search_find_threshold(point_is_lower_bound_theta_p(traces_pos, expr), precision, bound):
            case "contained", cover:
                theta_p = (cover[..., 0] + cover[..., 1])/2
            case "below",:
                theta_p = bound[..., 0]
            case "above",:
                theta_p = bound[..., 1]
            case x:
                assert False, x
        match binary_search_find_threshold(point_is_lower_bound_theta_n(traces_neg, expr), precision, bound):
            case "contained", cover:
                theta_n = (cover[..., 0] + cover[..., 1])/2
            case "below",:
                theta_n = bound[..., 0]
            case "above",:
                theta_n = bound[..., 1]
            case x:
                assert False, x

        if CFG["DEBUG"]:
            q_theta_p, q_theta_n = compute_theta_pn_quantitative_semantics(traces, labels, bound, expr)

            err_p = torch.abs(q_theta_p - theta_p)
            err_n = torch.abs(q_theta_n - theta_n)
            print("bound vol", compute_volume(bound))
            print("err_p")
            print(err_p)
            print("err_n")
            print(err_n)
            if compute_volume(bound) > 0.01:
                assert (err_p <= precision).all()
                assert (err_n <= precision).all()
        return theta_p, theta_n
    return f

def compute_theta_pn_quantitative_semantics(traces, labels, bound, expr):
    origin = bound[..., 0]
    scale = bound[..., 1] - bound[..., 0]
    t_thresh_per_trace = whole_trace_eval_origin_scale(traces, origin, scale, expr)
    # t_thresh_per_trace = whole_trace_eval_origin_scale_nonbatched(traces, origin, scale, expr)

    t_p = torch.min(t_thresh_per_trace[labels], -1).values
    t_n = torch.max(t_thresh_per_trace[~labels], -1).values

    if CFG["DEBUG"]:
        print("origin")
        print(origin)
        print("scale")
        print(scale)
        print("t_p")
        print(t_p)
        print("t_n")
        print(t_n)
    
    theta_p = origin + t_p * scale
    theta_n = origin + t_n * scale

    if (theta_p < bound[..., 0]).any():
        theta_p = bound[..., 0]

    if (theta_p > bound[..., 1]).any():
        theta_p = bound[..., 1]

    if (theta_n < bound[..., 0]).any():
        theta_n = bound[..., 0]

    if (theta_n > bound[..., 1]).any():
        theta_n = bound[..., 1]

    return theta_p, theta_n