# `D:\src\scipysrc\pandas\pandas\_libs\ops.pyi`

```
# 导入必要的类型别名和函数签名
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    TypeAlias,
    overload,
)

# 导入 NumPy 库
import numpy as np

# 导入 pandas 库中的类型别名 npt
from pandas._typing import npt

# 定义类型别名 _BinOp，表示一个可调用对象，接受两个任意类型参数并返回任意类型结果
_BinOp: TypeAlias = Callable[[Any, Any], Any]

# 定义类型别名 _BoolOp，表示一个可调用对象，接受两个任意类型参数并返回布尔类型结果
_BoolOp: TypeAlias = Callable[[Any, Any], bool]

# 函数定义，对比单个标量和数组的值，返回布尔类型的 NumPy 数组
def scalar_compare(
    values: np.ndarray,  # 传入的 NumPy 数组，包含任意对象类型的值
    val: object,  # 待比较的标量值
    op: _BoolOp,  # 比较操作的函数对象，例如 operator.eq 表示等于操作
) -> npt.NDArray[np.bool_]:  # 返回布尔类型的 NumPy 数组
    ...

# 函数定义，对比两个数组的对应元素，返回布尔类型的 NumPy 数组
def vec_compare(
    left: npt.NDArray[np.object_],  # 第一个输入数组，包含任意对象类型的值
    right: npt.NDArray[np.object_],  # 第二个输入数组，包含任意对象类型的值
    op: _BoolOp,  # 比较操作的函数对象，例如 operator.eq 表示等于操作
) -> npt.NDArray[np.bool_]:  # 返回布尔类型的 NumPy 数组
    ...

# 函数定义，对单个标量值或数组中的每个元素执行二元操作，返回 NumPy 数组
def scalar_binop(
    values: np.ndarray,  # 输入的 NumPy 数组，包含任意对象类型的值
    val: object,  # 二元操作的第二个参数，可以是任意对象类型的值
    op: _BinOp,  # 二元操作的函数对象，接受两个任意类型参数并返回任意类型结果
) -> np.ndarray:  # 返回 NumPy 数组
    ...

# 函数定义，对两个输入数组中的对应元素执行二元操作，返回 NumPy 数组
def vec_binop(
    left: np.ndarray,  # 第一个输入的 NumPy 数组，包含任意对象类型的值
    right: np.ndarray,  # 第二个输入的 NumPy 数组，包含任意对象类型的值
    op: _BinOp,  # 二元操作的函数对象，接受两个任意类型参数并返回任意类型结果
) -> np.ndarray:  # 返回 NumPy 数组
    ...

# 函数重载，根据参数条件，可能将对象数组转换为布尔类型数组或两个布尔类型数组的元组
@overload
def maybe_convert_bool(
    arr: npt.NDArray[np.object_],  # 输入的 NumPy 对象数组，包含任意对象类型的值
    true_values: Iterable | None = None,  # 可选的真值序列，可以是可迭代对象或 None
    false_values: Iterable | None = None,  # 可选的假值序列，可以是可迭代对象或 None
    convert_to_masked_nullable: Literal[False] = ...,  # 是否将结果转换为可屏蔽可空类型的布尔值，缺省为 False
) -> tuple[np.ndarray, None]:  # 返回元组，包含布尔类型的 NumPy 数组和 None
    ...

@overload
def maybe_convert_bool(
    arr: npt.NDArray[np.object_],  # 输入的 NumPy 对象数组，包含任意对象类型的值
    true_values: Iterable = ...,  # 必须的真值序列，可以是任意可迭代对象
    false_values: Iterable = ...,  # 必须的假值序列，可以是任意可迭代对象
    *,
    convert_to_masked_nullable: Literal[True],  # 将结果转换为可屏蔽可空类型的布尔值，此时为 True
) -> tuple[np.ndarray, np.ndarray]:  # 返回元组，包含两个布尔类型的 NumPy 数组
    ...
```