# `.\pytorch\torch\quantization\quantize_fx.py`

```
# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantize_fx.py`, while adding an import statement
here.
"""

# 从 torch.ao.quantization.fx.graph_module 模块导入 ObservedGraphModule 类
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
# 从 torch.ao.quantization.quantize_fx 模块导入多个函数和类
from torch.ao.quantization.quantize_fx import (
    _check_is_graph_module,                   # 导入 _check_is_graph_module 函数
    _convert_fx,                              # 导入 _convert_fx 函数
    _convert_standalone_module_fx,             # 导入 _convert_standalone_module_fx 函数
    _fuse_fx,                                 # 导入 _fuse_fx 函数
    _prepare_fx,                              # 导入 _prepare_fx 函数
    _prepare_standalone_module_fx,             # 导入 _prepare_standalone_module_fx 函数
    _swap_ff_with_fxff,                       # 导入 _swap_ff_with_fxff 函数
    convert_fx,                               # 导入 convert_fx 函数
    fuse_fx,                                  # 导入 fuse_fx 函数
    prepare_fx,                               # 导入 prepare_fx 函数
    prepare_qat_fx,                           # 导入 prepare_qat_fx 函数
    QuantizationTracer,                       # 导入 QuantizationTracer 类
    Scope,                                    # 导入 Scope 类
    ScopeContextManager,                      # 导入 ScopeContextManager 类
)
```