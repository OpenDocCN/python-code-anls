# `D:\src\scipysrc\scipy\scipy\stats\_sobol.pyi`

```
# 导入 NumPy 库并将其命名为 np
import numpy as np
# 从 scipy._lib._util 模块中导入 IntNumber 类型
from scipy._lib._util import IntNumber
# 从 typing 模块中导入 Literal 类型提示
from typing import Literal

# 定义一个私有函数 _initialize_v，用于初始化向量 v
def _initialize_v(
    v : np.ndarray,    # 参数 v 是一个 NumPy 数组
    dim : IntNumber,   # 参数 dim 是一个 IntNumber 类型的整数
    bits: IntNumber    # 参数 bits 是一个 IntNumber 类型的整数
) -> None: ...        # 函数没有具体实现

# 定义一个私有函数 _cscramble，用于乱序处理操作
def _cscramble (
    dim : IntNumber,   # 参数 dim 是一个 IntNumber 类型的整数
    bits: IntNumber,   # 参数 bits 是一个 IntNumber 类型的整数
    ltm : np.ndarray,  # 参数 ltm 是一个 NumPy 数组
    sv: np.ndarray     # 参数 sv 是一个 NumPy 数组
) -> None: ...        # 函数没有具体实现

# 定义一个私有函数 _fill_p_cumulative，用于填充累积概率分布数组
def _fill_p_cumulative(
    p: np.ndarray,         # 参数 p 是一个 NumPy 数组
    p_cumulative: np.ndarray  # 参数 p_cumulative 是一个 NumPy 数组
) -> None: ...             # 函数没有具体实现

# 定义一个私有函数 _draw，用于生成随机抽样数据
def _draw(
    n : IntNumber,         # 参数 n 是一个 IntNumber 类型的整数
    num_gen: IntNumber,    # 参数 num_gen 是一个 IntNumber 类型的整数
    dim: IntNumber,        # 参数 dim 是一个 IntNumber 类型的整数
    scale: float,          # 参数 scale 是一个浮点数
    sv: np.ndarray,        # 参数 sv 是一个 NumPy 数组
    quasi: np.ndarray,     # 参数 quasi 是一个 NumPy 数组
    sample: np.ndarray     # 参数 sample 是一个 NumPy 数组
) -> None: ...             # 函数没有具体实现

# 定义一个私有函数 _fast_forward，用于快速前进处理
def _fast_forward(
    n: IntNumber,          # 参数 n 是一个 IntNumber 类型的整数
    num_gen: IntNumber,    # 参数 num_gen 是一个 IntNumber 类型的整数
    dim: IntNumber,        # 参数 dim 是一个 IntNumber 类型的整数
    sv: np.ndarray,        # 参数 sv 是一个 NumPy 数组
    quasi: np.ndarray      # 参数 quasi 是一个 NumPy 数组
) -> None: ...             # 函数没有具体实现

# 定义一个私有函数 _categorize，用于分类处理
def _categorize(
    draws: np.ndarray,     # 参数 draws 是一个 NumPy 数组
    p_cumulative: np.ndarray,  # 参数 p_cumulative 是一个 NumPy 数组
    result: np.ndarray     # 参数 result 是一个 NumPy 数组
) -> None: ...             # 函数没有具体实现

# 定义常量 _MAXDIM，表示最大维度，值为 21201
_MAXDIM: Literal[21201]
# 定义常量 _MAXDEG，表示最大度数，值为 18
_MAXDEG: Literal[18]

# 定义一个私有函数 _test_find_index，用于查找索引测试
def _test_find_index(
    p_cumulative: np.ndarray,   # 参数 p_cumulative 是一个 NumPy 数组
    size: int,                  # 参数 size 是一个整数
    value: float                # 参数 value 是一个浮点数
) -> int: ...                   # 函数没有具体实现
```