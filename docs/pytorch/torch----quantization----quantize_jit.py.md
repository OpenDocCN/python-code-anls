# `.\pytorch\torch\quantization\quantize_jit.py`

```py
# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantize_jit.py`, while adding an import statement
here.
"""

# 从 torch/ao/quantization/quantize_jit.py 导入以下函数和变量，用于量化 JIT 模型
from torch.ao.quantization.quantize_jit import (
    _check_forward_method,                  # 导入 _check_forward_method 函数
    _check_is_script_module,                # 导入 _check_is_script_module 函数
    _convert_jit,                           # 导入 _convert_jit 函数
    _prepare_jit,                           # 导入 _prepare_jit 函数
    _prepare_ondevice_dynamic_jit,          # 导入 _prepare_ondevice_dynamic_jit 函数
    _quantize_jit,                          # 导入 _quantize_jit 函数
    convert_dynamic_jit,                    # 导入 convert_dynamic_jit 函数
    convert_jit,                            # 导入 convert_jit 函数
    fuse_conv_bn_jit,                       # 导入 fuse_conv_bn_jit 函数
    prepare_dynamic_jit,                    # 导入 prepare_dynamic_jit 函数
    prepare_jit,                            # 导入 prepare_jit 函数
    quantize_dynamic_jit,                   # 导入 quantize_dynamic_jit 函数
    quantize_jit,                           # 导入 quantize_jit 函数
    script_qconfig,                         # 导入 script_qconfig 变量
    script_qconfig_dict,                    # 导入 script_qconfig_dict 变量
)
```