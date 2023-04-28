import torch
from torch_utils import *

# out(i, j) = min_k max(A(i, k), B(k, j))
def maxmin_matmul(input, other):
    assert input.shape[-1] == other.shape[-2]
    input_big = input.unsqueeze(-1)
    other_big = other.unsqueeze(-3)
    big = torch.min(input_big, other_big)
    return torch.max(big, -2).values

# out(j) = min_k max(V(k), B(k, j))
def maxmin_vecmatmul(vec, mat):
    assert vec.shape[-1] == mat.shape[-2]
    vec_big = vec.unsqueeze(-1)
    big = torch.min(vec_big, mat)
    return torch.max(big, -2).values

# out(i) = min_k max(A(i, k), V(k))
def maxmin_matvecmul(mat, vec):
    assert vec.shape[-1] == mat.shape[-1]
    vec_big = vec.unsqueeze(-2)
    big = torch.min(vec_big, mat)
    return torch.max(big, -1).values

# out(i) = min_k max(A(i, k), V(k))
def disjconj_matvecmul(mat, vec):
    assert vec.shape[-1] == mat.shape[-1]
    vec_big = vec.unsqueeze(-2)
    big = vec_big & mat
    return torch.any(big, -1)

if False:
    maxmin_vecmatmul(
        torch_literal([1., 0.5, 0.]),
        torch_literal([ 
            [1., 1., 0., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 1.],
        ]),
    )

    maxmin_matvecmul(
        torch_literal([ 
            [1., 1., 0., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 1.],
        ]),
        torch_literal([1., 0.5, 0., 0.75]),
    )

def min_over_intervals(t):
    n = t.shape[-1]
    ret = torch.zeros(t.shape + (n,), dtype = t.dtype, device = t.device)

    for a in range(n):
        for b in range(n - a):
            i = b
            j = b + a
            if j == i:
                ret[..., i, i] = t[..., i]
            else:
                ret[..., i, j] = torch.min(ret[..., i + 1, j], ret[..., i, j - 1])
    return ret