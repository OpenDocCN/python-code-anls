# `D:\src\scipysrc\scipy\scipy\linalg\_matfuncs_expm.pyi`

```
# 从 numpy.typing 模块中导入 NDArray 类型，用于标注 NumPy 数组
# 从 typing 模块中导入 Any 类型，用于标注任意类型
from numpy.typing import NDArray
from typing import Any

# 定义 pick_pade_structure 函数，接受一个 NumPy 数组 a，返回一个元组，元组包含两个整数值
def pick_pade_structure(a: NDArray[Any]) -> tuple[int, int]: ...

# 定义 pade_UV_calc 函数，接受三个参数：NumPy 数组 Am，以及两个整数 n 和 m，返回 None
def pade_UV_calc(Am: NDArray[Any], n: int, m: int) -> None: ...
```