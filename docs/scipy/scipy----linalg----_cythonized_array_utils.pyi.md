# `D:\src\scipysrc\scipy\scipy\linalg\_cythonized_array_utils.pyi`

```
# 导入需要的类型提示模块
from numpy.typing import NDArray
from typing import Any

# 定义 bandwidth 函数，返回两个整数元组
def bandwidth(a: NDArray[Any]) -> tuple[int, int]:
    ...

# 定义 issymmetric 函数，检查给定数组是否对称
def issymmetric(
    a: NDArray[Any],
    atol: None | float = ...,
    rtol: None | float = ...,
) -> bool:
    ...

# 定义 ishermitian 函数，检查给定数组是否为厄米特矩阵（复数域中的对称矩阵）
def ishermitian(
    a: NDArray[Any],
    atol: None | float = ...,
    rtol: None | float = ...,
) -> bool:
    ...
```