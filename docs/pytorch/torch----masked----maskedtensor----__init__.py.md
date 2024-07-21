# `.\pytorch\torch\masked\maskedtensor\__init__.py`

```py
# 版权声明，指明代码版权归 Meta Platforms, Inc. 及其关联公司所有
# flake8: noqa

# 导入 _apply_native_binary 和 _is_native_binary 函数，来自当前目录下的 binary 模块
from .binary import _apply_native_binary, _is_native_binary
# 导入 is_masked_tensor 和 MaskedTensor 类，来自当前目录下的 core 模块
from .core import is_masked_tensor, MaskedTensor
# 导入 _apply_pass_through_fn 和 _is_pass_through_fn 函数，来自当前目录下的 passthrough 模块
from .passthrough import _apply_pass_through_fn, _is_pass_through_fn
# 导入 _apply_reduction 和 _is_reduction 函数，来自当前目录下的 reductions 模块
from .reductions import _apply_reduction, _is_reduction
# 导入 _apply_native_unary 和 _is_native_unary 函数，来自当前目录下的 unary 模块
from .unary import _apply_native_unary, _is_native_unary
```