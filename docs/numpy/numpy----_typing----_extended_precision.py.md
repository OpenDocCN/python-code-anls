# `.\numpy\numpy\_typing\_extended_precision.py`

```py
# 导入 NumPy 库，用于科学计算和数组操作
import numpy as np

# 导入当前包中的特定扩展精度类型定义
from . import (
    _80Bit,
    _96Bit,
    _128Bit,
    _256Bit,
)

# 定义无符号整数类型为 128 位
uint128 = np.unsignedinteger[_128Bit]

# 定义无符号整数类型为 256 位
uint256 = np.unsignedinteger[_256Bit]

# 定义有符号整数类型为 128 位
int128 = np.signedinteger[_128Bit]

# 定义有符号整数类型为 256 位
int256 = np.signedinteger[_256Bit]

# 定义单精度浮点数类型为 80 位
float80 = np.floating[_80Bit]

# 定义双精度浮点数类型为 96 位
float96 = np.floating[_96Bit]

# 定义双精度浮点数类型为 128 位
float128 = np.floating[_128Bit]

# 定义双精度浮点数类型为 256 位
float256 = np.floating[_256Bit]

# 定义复数类型，实部和虚部均为 80 位
complex160 = np.complexfloating[_80Bit, _80Bit]

# 定义复数类型，实部和虚部均为 96 位
complex192 = np.complexfloating[_96Bit, _96Bit]

# 定义复数类型，实部和虚部均为 128 位
complex256 = np.complexfloating[_128Bit, _128Bit]

# 定义复数类型，实部和虚部均为 256 位
complex512 = np.complexfloating[_256Bit, _256Bit]
```