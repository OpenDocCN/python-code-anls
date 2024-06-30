# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_lu_cython.pyi`

```
from numpy.typing import NDArray  # 从 numpy.typing 模块导入 NDArray 类型
from typing import Any  # 导入 Any 类型，用于泛型类型注解

# LU 分解函数定义，接受一个二维数组 a 作为输入，以及 lu、perm 和 permute_l 参数
def lu_decompose(a: NDArray[Any], lu: NDArray[Any], perm: NDArray[Any], permute_l: bool) -> None:
    ...  # 占位符，实际函数体未提供

# LU 分解函数调度器定义，接受一个二维数组 a 作为输入，以及 lu、perm 和 permute_l 参数
def lu_dispatcher(a: NDArray[Any], lu: NDArray[Any], perm: NDArray[Any], permute_l: bool) -> None:
    ...  # 占位符，实际函数体未提供
```