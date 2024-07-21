# `.\pytorch\torch\onnx\symbolic_opset19.py`

```
"""
这个文件为 opset 19 导出 ONNX 操作符。

Note [ONNX Operators that are added/updated in opset 19]
opset 19 中新增/更新的 ONNX 操作符列表。

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-19-of-the-default-onnx-operator-set
新操作符：
AveragePool
Cast
CastLike
Constant
DeformConv
DequantizeLinear
Equal
Identity
If
Loop
Pad
QuantizeLinear
Reshape
Resize
Scan
Shape
Size
"""

from typing import List

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py
# 如果你要编辑这个文件，请先阅读 symbolic_helper.py 中的 Note [Edit Symbolic Files]

__all__: List[str] = []
# 定义一个空列表，用于声明本文件中公开的符号名称
```