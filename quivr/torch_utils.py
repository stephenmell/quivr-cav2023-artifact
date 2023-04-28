import torch
import pathlib
from config import *

_torch_main_device = None
def torch_main_device():
    global _torch_main_device

    if _torch_main_device is None:
        print("Warning: torch main device was never set")
        _torch_main_device = torch.device("cpu")
    
    return _torch_main_device

def set_torch_main_device(dev):
    global _torch_main_device
    assert _torch_main_device is None
    _torch_main_device = dev

set_torch_main_device(torch.device(CFG["TORCH_DEVICE"]))

def torch_save(t, path):
    torch.save(t, path)

def torch_load(path):
    return torch.load(path, map_location = torch_main_device())

def torch_literal(*args, **kwargs):
    return torch.tensor(*args, device = torch_main_device(), **kwargs)

def torch_zeros(*args, **kwargs):
    return torch.zeros(*args, device = torch_main_device(), **kwargs)

def torch_ones(*args, **kwargs):
    return torch.ones(*args, device = torch_main_device(), **kwargs)

def torch_empty(*args, **kwargs):
    return torch.empty(*args, device = torch_main_device(), **kwargs)

def torch_from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).to(device = torch_main_device())

def neginfty_triu(t):
    ret = torch_empty(t.shape).fill_(float("-inf"))
    row_idx, col_idx = torch.triu_indices(t.shape[-2], t.shape[-1])
    ret[..., row_idx, col_idx] = t[..., row_idx, col_idx]
    return ret

def tensor_approx_eq(a, b):
    return torch.isclose(a, b, atol = CFG["FLOAT_ATOL"])

def tensor_approx_lt(a, b):
    return ~tensor_approx_eq(a, b) & (a < b)

def tensor_approx_lteq(a, b):
    return tensor_approx_eq(a, b) | (a < b)

def tensor_approx_gt(a, b):
    return tensor_approx_lt(b, a)

def tensor_approx_gteq(a, b):
    return tensor_approx_lteq(b, a)

def generator_from_seed(seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen

def permute_from_seed(t, generator, axis = -1):
    assert axis == -1, "unimplemented"
    return t[..., torch.randperm(t.shape[-1], generator = generator)]

def dataset_subsample_balanced(data, label_mask, n_pos, n_neg, generator):
    ret_pos = permute_from_seed(data[..., label_mask], generator)[0:n_pos]
    ret_neg = permute_from_seed(data[..., ~label_mask], generator)[0:n_neg]
    return torch.concat((ret_pos, ret_neg), axis = -1)

def tensor_indices_to_mask(n, indices):
    shape = indices.shape[:-1] + (n,)
    assert len(shape) == 1
    ret = torch_zeros(shape, dtype = torch.bool)
    # ret[..., indices] = True
    ret[indices] = True
    return ret

def build_replacement_mapping_from_values(x, additional):
    assert not torch.isnan(x).any()
    values = torch.concat([x.unique(), additional], axis = 0)
    return values.unique()

def replace_values_with_indices_f32(x, mapping):
    assert len(mapping) < 2**24 # bound on contiguous integers representable in f32
    
    ret = torch_empty(x.shape, dtype = torch.float32)

    if CFG["DEBUG"]:
        from tqdm import tqdm
        iterable = tqdm(mapping)
    else:
        iterable = mapping
    
    for i, val in enumerate(iterable):
        ret[x == val] = i

    return ret

def torch_save_mkdirs(t, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch_save(t, path)