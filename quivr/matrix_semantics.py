import torch
from maxmin_utils import *
from misc_utils import *
from torch_utils import *

def cubic_eval_origin_scale(traces, origin, heading, initial_expr):
    def denote(expr):
        match expr:
            case ("PRED0", pred):
                x = traces[pred]
                assert x.dtype == torch.bool, (pred, x.dtype, x.shape)
                return torch.where(x, float("inf"), float("-inf"))
            
            case ("PRED1", pred, param_name):
                x = traces[pred]
                assert x.dtype == torch.float32, (pred, x.dtype, x.shape)
                predicate_origin = origin[param_name]
                predicate_scale = heading[param_name]
                return neginfty_triu((x - predicate_origin) / predicate_scale)
            
            case ("SEQ", subexprs):
                return fold_taking_first(maxmin_matmul, (
                    denote(subexpr) for subexpr in subexprs
                ))
            
            case ("CONJ", subexprs):
                return fold_taking_first(torch.minimum, (
                    denote(subexpr) for subexpr in subexprs
                ))

            case ("DISJ", subexprs):
                return fold_taking_first(torch.maximum, (
                    denote(subexpr) for subexpr in subexprs
                ))
            
            case x:
                assert False, x
    return denote(initial_expr)

def quadratic_eval_origin_scale(traces, origin, heading, initial_expr, initial_rvec):
    def denote(expr, rvec):
        match expr:
            case ("PRED0", pred):
                x = traces[pred]
                assert x.dtype == torch.bool, (pred, x.dtype, x.shape)
                return maxmin_matvecmul(
                    torch.where(x, float("inf"), float("-inf")),
                    rvec,
                )
            
            case ("PRED1", pred, param_name):
                x = traces[pred]
                assert x.dtype == torch.float32, (pred, x.dtype, x.shape)
                predicate_origin = origin[param_name]
                predicate_scale = heading[param_name]
                return maxmin_matvecmul(
                    neginfty_triu((x - predicate_origin) / predicate_scale),
                    rvec,
                )
            
            case ("SEQ", subexprs):
                cur_vec = rvec
                for subexpr in reversed(subexprs):
                    cur_vec = denote(subexpr, cur_vec)
                return cur_vec
            
            case ("CONJ", subexprs):
                return fold_taking_first(torch.minimum, (
                    denote(subexpr, rvec) for subexpr in subexprs
                ))

            case ("DISJ", subexprs):
                return fold_taking_first(torch.maximum, (
                    denote(subexpr, rvec) for subexpr in subexprs
                ))
            
            case x:
                assert False, x
    ret = denote(initial_expr, initial_rvec)

    if CFG["DEBUG_QUADRATIC_MATRIX_SEMANTICS"]:
        correctish = maxmin_matvecmul(
            cubic_eval_origin_scale(traces, origin, heading, initial_expr),
            initial_rvec,
        )
        assert (ret == correctish).all(), (ret, correctish)

    return ret

def whole_trace_eval_origin_scale(traces, origin, heading, initial_expr):
    rvec = torch_empty((trace_shape(traces)[-1],))
    rvec[:-1] = float("-inf")
    rvec[-1] = float("inf")

    return quadratic_eval_origin_scale(traces, origin, heading, initial_expr, rvec)[..., 0]

def whole_trace_eval_origin_scale_nonbatched(traces, origin, heading, initial_expr):
    assert len(trace_shape(traces)) == 3

    rvec = torch_empty((trace_shape(traces)[-1],))
    rvec[:-1] = float("-inf")
    rvec[-1] = float("inf")

    batch_size = trace_shape(traces)[0]
    ret = torch_empty((batch_size,))

    for i in range(batch_size):
        trace = {k: v[i] for k, v in traces.items()}
        ret[i] = quadratic_eval_origin_scale(trace, origin, heading, initial_expr, rvec)[..., 0]
    
    return ret

def whole_trace_eval_trinary(traces, threshold, initial_expr):
    # since we're taking the sign at the end, the actual value of the heading doesn't matter anyway
    # we just get whether it's marginal (0), satisfied (1), or unsatisfied (-1)
    t_thresh_per_track = whole_trace_eval_robustness(traces, threshold, initial_expr)
    return torch.where(
        tensor_approx_eq(t_thresh_per_track, torch_literal(0.)),
        torch_literal(0.),
        torch.sign(t_thresh_per_track)
    )

def whole_trace_eval_robustness(traces, threshold, initial_expr):
    # if we make the heading be all ones, then the origin_scale semantics for predicates
    # become just subtracing the origin (threshold) off from the trace values
    heading = torch_ones(threshold.shape)
    return whole_trace_eval_origin_scale(traces, threshold, heading, initial_expr)

def whole_trace_eval_binary_float(traces, threshold, initial_expr):
    return whole_trace_eval_trinary(traces, threshold, initial_expr) >= 0.

def quadratic_eval_bool(traces, threshold, initial_expr, initial_rvec):
    def denote(expr, rvec):
        match expr:
            case ("PRED0", pred):
                x = traces[pred]
                assert x.dtype == torch.bool, (pred, x.dtype, x.shape)
                return disjconj_matvecmul(
                    x,
                    rvec,
                )
            
            case ("PRED1", pred, param_name):
                x = traces[pred]
                assert x.dtype == torch.float32, (pred, x.dtype, x.shape)
                return disjconj_matvecmul(
                    x >= threshold[param_name],
                    rvec,
                )
            
            case ("SEQ", subexprs):
                cur_vec = rvec
                for subexpr in reversed(subexprs):
                    cur_vec = denote(subexpr, cur_vec)
                return cur_vec
            
            case ("CONJ", subexprs):
#                return fold_taking_first(torch.minimum, (
                return fold_taking_first(lambda a, b: a & b, (
                    denote(subexpr, rvec) for subexpr in subexprs
                ))

            case ("DISJ", subexprs):
#                return fold_taking_first(torch.maximum, (
                return fold_taking_first(lambda a, b: a | b, (
                    denote(subexpr, rvec) for subexpr in subexprs
                ))
            
            case x:
                assert False, x
    ret = denote(initial_expr, initial_rvec)

    return ret

def whole_trace_eval_binary_bool(traces, threshold, initial_expr):
    rvec = torch_empty((trace_shape(traces)[-1],), dtype = torch.bool)
    rvec[:-1] = False
    rvec[-1] = True

    ret = quadratic_eval_bool(traces, threshold, initial_expr, rvec)[..., 0]

    if CFG["DEBUG"]:
        correctish = whole_trace_eval_binary_float(traces, threshold, initial_expr)
        assert (ret == correctish).all(), (ret, correctish)
    
    return ret

def quadratic_eval_byte(traces, threshold, initial_expr, initial_rvec):
    def denote(expr, rvec):
        match expr:
            case ("PRED0", pred):
                x = traces[pred]
                assert x.dtype == torch.bool, (pred, x.dtype, x.shape)
                return maxmin_matvecmul(
                    x,
                    rvec,
                ).byte()
            
            case ("PRED1", pred, param_name):
                x = traces[pred]
                assert x.dtype == torch.float32, (pred, x.dtype, x.shape)
                return maxmin_matvecmul(
                    x >= threshold[param_name],
                    rvec,
                ).byte()
            
            case ("SEQ", subexprs):
                cur_vec = rvec
                for subexpr in reversed(subexprs):
                    cur_vec = denote(subexpr, cur_vec)
                return cur_vec
            
            case ("CONJ", subexprs):
                return fold_taking_first(torch.minimum, (
                    denote(subexpr, rvec) for subexpr in subexprs
                ))

            case ("DISJ", subexprs):
                return fold_taking_first(torch.maximum, (
                    denote(subexpr, rvec) for subexpr in subexprs
                ))
            
            case x:
                assert False, x
    ret = denote(initial_expr, initial_rvec)

    return ret

def whole_trace_eval_binary_byte(traces, threshold, initial_expr):
    rvec = torch_empty((trace_shape(traces)[-1],), dtype = torch.bool)
    rvec[:-1] = False
    rvec[-1] = True

    ret = quadratic_eval_byte(traces, threshold, initial_expr, rvec)[..., 0]

    if CFG["DEBUG"]:
        correctish = whole_trace_eval_binary_float(traces, threshold, initial_expr)
        assert (ret == correctish).all(), (ret, correctish)
    
    return ret

match CFG["BINARY_SEMANTICS_DTYPE"]:
    case "float":
        whole_trace_eval_binary = whole_trace_eval_binary_float
    case "bool":
        whole_trace_eval_binary = whole_trace_eval_binary_bool
    case "byte":
        whole_trace_eval_binary = whole_trace_eval_binary_byte
    case x:
        assert False, x