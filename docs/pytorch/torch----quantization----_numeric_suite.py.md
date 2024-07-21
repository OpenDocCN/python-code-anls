# `.\pytorch\torch\quantization\_numeric_suite.py`

```py
# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/ns/_numeric_suite.py`, while adding an import statement
here.
"""

# 从 torch.ao.ns._numeric_suite 模块中导入以下函数和类
from torch.ao.ns._numeric_suite import (
    _convert_tuple_to_list,      # 导入 _convert_tuple_to_list 函数
    _dequantize_tensor_list,     # 导入 _dequantize_tensor_list 函数
    _find_match,                 # 导入 _find_match 函数
    _get_logger_dict_helper,     # 导入 _get_logger_dict_helper 函数
    _is_identical_module_type,   # 导入 _is_identical_module_type 函数
    compare_model_outputs,       # 导入 compare_model_outputs 函数
    compare_model_stub,          # 导入 compare_model_stub 函数
    compare_weights,             # 导入 compare_weights 函数
    get_logger_dict,             # 导入 get_logger_dict 函数
    get_matching_activations,    # 导入 get_matching_activations 函数
    Logger,                      # 导入 Logger 类
    NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST,  # 导入 NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST 变量
    OutputLogger,                # 导入 OutputLogger 类
    prepare_model_outputs,       # 导入 prepare_model_outputs 函数
    prepare_model_with_stubs,    # 导入 prepare_model_with_stubs 函数
    Shadow,                      # 导入 Shadow 类
    ShadowLogger,                # 导入 ShadowLogger 类
)
```