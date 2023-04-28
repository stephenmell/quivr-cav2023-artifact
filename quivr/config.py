
_default_cfg = {
    "DEBUG": False,
    "DEBUG_QUADRATIC_MATRIX_SEMANTICS": False,
    "FLOAT_ATOL": 1.e-06,
    # "FLOAT_ATOL": 1.e-08,
    # "TORCH_DEVICE": "cuda:0",
    # "TORCH_DEVICE": "cpu",
    "BINARY_SEMANTICS_DTYPE": "bool",
    # "BINARY_SEMANTICS_DTYPE": "byte",
    # "BINARY_SEMANTICS_DTYPE": "float",
    # "PARAMETER_SEARCH_METHOD": "quantitative",
    # "PARAMETER_SEARCH_METHOD": "binary",
    "PARAMETER_SEARCH_BINARY_PRECISION": 0.001, # used by learningSTL
}

CFG = None

def set_cfg(**kwargs):
    assert "TORCH_DEVICE" in kwargs
    assert "PARAMETER_SEARCH_METHOD" in kwargs
    
    global CFG
    new_cfg = {
        **_default_cfg,
        **kwargs,
    }

    assert CFG is None or CFG == new_cfg
    CFG = new_cfg