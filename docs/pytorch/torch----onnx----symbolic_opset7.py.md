# `.\pytorch\torch\onnx\symbolic_opset7.py`

```py
"""
Note [ONNX operators that are added/updated from opset 7 to opset 8]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
  Expand

Updated operators:
  Min, Max, Sum, Mean: supports multidirectional broadcasting.
  MaxPool: added optional indices output.
  Scan
"""

import functools  # 导入 functools 模块，用于创建偏函数
import warnings  # 导入 warnings 模块，用于发出警告信息

from torch.onnx import symbolic_helper, symbolic_opset9 as opset9  # 导入 torch.onnx 下的 symbolic_helper 和 opset9
from torch.onnx._internal import jit_utils, registration  # 导入 torch.onnx._internal 下的 jit_utils 和 registration

# 定义一个偏函数 _onnx_symbolic，指定 opset=7 的注册函数
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=7)

# 阻止转换的操作符列表
block_listed_operators = (
    "scan",
    "expand",
    "expand_as",
    "meshgrid",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
)

# NOTE: max, min, sum, mean: broadcasting is not supported in opset 7.
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
# 定义符号化函数 @_onnx_symbolic("aten::max")，用于转换 torch.max 操作
@_onnx_symbolic("aten::max")
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.max(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to max operators "
            "have different shapes"
        )
    return opset9.max(g, self, dim_or_y, keepdim)

# 定义符号化函数 @_onnx_symbolic("aten::min")，用于转换 torch.min 操作
@_onnx_symbolic("aten::min")
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.min(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to min operators "
            "have different shapes"
        )
    return opset9.min(g, self, dim_or_y, keepdim)

# 对于阻止转换的操作符列表中的每个操作符，注册符号化函数
for block_listed_op in block_listed_operators:
    _onnx_symbolic(f"aten::{block_listed_op}")(
        symbolic_helper._block_list_in_opset(block_listed_op)
    )
```