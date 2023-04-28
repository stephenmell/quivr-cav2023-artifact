import torch
import itertools
from misc_utils import *
from torch_utils import *

make_quadrants_new = lambda param_d: torch_literal(tuple(itertools.product(
    set((True, False)),
    repeat = param_d,
)))

corners_by_dim = {
    1: make_quadrants_new(1),
    2: make_quadrants_new(2),
    3: make_quadrants_new(3),
}
corners_by_dim[2]

def filter_empty_boxes(boxes):
    empty = (boxes[..., 0] == boxes[..., 1]).any(-1)
    return boxes[~empty]

# inner, outer : ..., d, 2
def _corner_boxes(outer, inner):
    d = outer.shape[-2]
    assert d == 2 or d == 1
    if d == 1:
        return torch_zeros((0, d, 2))
    
    outer_lower = outer[..., 0]
    outer_upper = outer[..., 1]
    inner_lower = inner[..., 0]
    inner_upper = inner[..., 1]

    # 2^d, d -> Bool
    d_corners = corners_by_dim[d]
    # each quadrant has either a new upper (False) or lower (True) bound in each dimension
    # if it's a new lower bound, then we take inner_upper, else we take outer_lower
    cb_lower = torch.where(d_corners, inner_upper.unsqueeze(-2), outer_lower.unsqueeze(-2))
    # if it's a new upper bound, then we take inner_lower, else we take outer_upper
    cb_upper = torch.where(~d_corners, inner_lower.unsqueeze(-2), outer_upper.unsqueeze(-2))
    
    return torch.stack((cb_lower, cb_upper), axis = -1)

# inner, outer : ..., d, 2
def _face_boxes(outer, inner):
    d = outer.shape[-2]
    outer_lower = outer[..., 0]
    outer_upper = outer[..., 1]
    inner_lower = inner[..., 0]
    inner_upper = inner[..., 1]
    
    # for face boxes
    # we basically want the inner box, except that we replace the coordinate component
    fb_lower = []
    fb_upper = []
    for i in range(d):
        l_lower = inner_lower.clone()
        l_lower[i] = outer_lower[i]
        l_upper = inner_upper.clone()
        l_upper[i] = inner_lower[i]

        u_lower = inner_lower.clone()
        u_lower[i] = inner_upper[i]
        u_upper = inner_upper.clone()
        u_upper[i] = outer_upper[i]
        
        fb_lower.append(l_lower)
        fb_upper.append(l_upper)
        fb_lower.append(u_lower)
        fb_upper.append(u_upper)
    return torch.stack((
        torch.stack(fb_lower, axis = -2),
        torch.stack(fb_upper, axis = -2),
    ), axis = -1)

# inner, outer : ..., d, 2
def fracture_box_with_box(outer, inner):
    if CFG["DEBUG"]:
        d = outer.shape[-1]
        assert d == inner.shape[-1]
        assert (outer[..., 0] <= inner[..., 0]).all()
        assert (inner[..., 0] <= inner[..., 1]).all()
        assert (inner[..., 1] <= outer[..., 1]).all()

    cb = _corner_boxes(outer, inner)
    fb = _face_boxes(outer, inner)
    ib = inner.unsqueeze(-3)

    if CFG["DEBUG"]:
        try:
            combined = torch.concat([ib, fb, cb], axis = -3)
            assert (combined[..., 0] <= combined[..., 1]).all()
            fracture_volume = torch.sum(compute_volume(combined), -1)
            outer_volume = compute_volume(outer)
            # we use approx here because the volume computation is lossy
            # but this doesn't affect the result, it's just a sanity check
            assert tensor_approx_eq(fracture_volume, outer_volume), (fracture_volume - outer_volume, )
            assert_all_disjoint(combined)
        except Exception as e:
            print("outer")
            print(outer.shape)
            print(outer)
            print("inner")
            print(inner.shape)
            print(inner)
            print("fb")
            print(fb.shape)
            print(fb)
            print("cb")
            print(cb.shape)
            print(cb)
            print("combined")
            print(combined.shape)
            print(combined)
            print("combined volume")
            print(compute_volume(combined))
            raise e

    return (ib, fb, cb)

def compute_volume(box):
    return torch.prod(box[..., 1] - box[..., 0], axis = -1)

def pair_disjoint(a, b):
    return ((a[..., 1] <= b[..., 0]) | (b[..., 1] <= a[..., 0])).any(axis = -1)

def assert_all_disjoint(boxes):
    n = boxes.shape[-3]
    for i in range(n):
        for j in range(i + 1, n):
            res = pair_disjoint(boxes[..., i, :, :], boxes[..., j, :, :])
            assert res, (i, j)


if False:
    boxes = fracture_box_with_box(
        torch_literal([
            [0., 2.],
            [0., 1.],
        ]),
        torch_literal([
            [0.5, 1.],
            [0.5, 1.],
        ]),
    )
    boxes = fracture_box_with_box(
        torch_literal([
            [0., 2.],
        ]),
        torch_literal([
            [0.5, 1.],
        ]),
    )
    boxes