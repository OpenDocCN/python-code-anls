# `D:\src\scipysrc\numpy\numpy\matrixlib\defmatrix.pyi`

```py
# 导入必要的模块和类型定义
from collections.abc import Sequence, Mapping  # 导入 collections.abc 模块中的 Sequence 和 Mapping
from typing import Any  # 导入 typing 模块中的 Any 类型
from numpy import matrix as matrix  # 从 numpy 中导入 matrix 类，并将其命名为 matrix
from numpy._typing import ArrayLike, DTypeLike, NDArray  # 导入 numpy._typing 中的 ArrayLike, DTypeLike 和 NDArray 类型

__all__: list[str]  # 定义 __all__ 变量，类型为 list，其中元素类型为 str

# 定义函数 bmat，接受一个字符串、数组类别的序列或 NDArray 类型的对象，
# 并可选地接受一个局部字典和全局字典，返回一个 numpy matrix 类型的对象
def bmat(
    obj: str | Sequence[ArrayLike] | NDArray[Any],  # obj 参数可以是 str、Sequence[ArrayLike] 或 NDArray[Any] 类型
    ldict: None | Mapping[str, Any] = ...,  # ldict 参数可选，可以是 None 或 Mapping[str, Any] 类型，默认值为省略号
    gdict: None | Mapping[str, Any] = ...,  # gdict 参数可选，可以是 None 或 Mapping[str, Any] 类型，默认值为省略号
) -> matrix[Any, Any]:  # 函数返回一个 numpy matrix 类型，可以包含任意类型的数据

# 定义函数 asmatrix，接受一个 ArrayLike 类型的数据和一个可选的 dtype 参数，
# 返回一个 numpy matrix 类型的对象
def asmatrix(data: ArrayLike, dtype: DTypeLike = ...) -> matrix[Any, Any]:  # data 参数是 ArrayLike 类型，dtype 参数是 DTypeLike 类型，默认值为省略号

mat = asmatrix  # 将 asmatrix 函数赋值给 mat 变量
```