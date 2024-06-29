# `D:\src\scipysrc\numpy\numpy\lib\_shape_base_impl.pyi`

```py
# 导入系统模块 sys
import sys
# 导入集合抽象基类中的 Callable 和 Sequence 类
from collections.abc import Callable, Sequence
# 导入类型变量 TypeVar、Any，以及 overload 和 SupportsIndex 类型注解
from typing import TypeVar, Any, overload, SupportsIndex, Protocol

# 根据 Python 版本导入不同的类型注解
if sys.version_info >= (3, 10):
    from typing import ParamSpec, Concatenate
else:
    from typing_extensions import ParamSpec, Concatenate

# 导入 NumPy 库，并将其命名为 np
import numpy as np
# 从 NumPy 中导入多个类型，如 generic、integer 等
from numpy import (
    generic,
    integer,
    ufunc,
    unsignedinteger,
    signedinteger,
    floating,
    complexfloating,
    object_,
)

# 从 NumPy 的内部模块 _typing 中导入多个类型别名
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ShapeLike,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
)

# 从 NumPy 的内部模块 _core.shape_base 中导入 vstack 函数
from numpy._core.shape_base import vstack

# 定义类型变量 _P，用于参数规范
_P = ParamSpec("_P")
# 定义类型变量 _SCT，继承自 generic 泛型类型
_SCT = TypeVar("_SCT", bound=generic)

# 定义 _ArrayWrap 协议，描述 __array_wrap__ 方法的签名
class _ArrayWrap(Protocol):
    def __call__(
        self,
        array: NDArray[Any],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
        return_scalar: bool = ...,
        /,
    ) -> Any: ...

# 定义 _SupportsArrayWrap 协议，描述 __array_wrap__ 属性的签名
class _SupportsArrayWrap(Protocol):
    @property
    def __array_wrap__(self) -> _ArrayWrap: ...

# 声明 __all__ 列表，用于模块级别的公开接口
__all__: list[str]

# 定义 take_along_axis 函数，用于在指定轴上获取元素
def take_along_axis(
    arr: _SCT | NDArray[_SCT],
    indices: NDArray[integer[Any]],
    axis: None | int,
) -> NDArray[_SCT]: ...

# 定义 put_along_axis 函数，用于在指定轴上放置元素
def put_along_axis(
    arr: NDArray[_SCT],
    indices: NDArray[integer[Any]],
    values: ArrayLike,
    axis: None | int,
) -> None: ...

# apply_along_axis 函数的函数重载，用于在指定轴上应用函数
@overload
def apply_along_axis(
    func1d: Callable[Concatenate[NDArray[Any], _P], _ArrayLike[_SCT]],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> NDArray[_SCT]: ...
@overload
def apply_along_axis(
    func1d: Callable[Concatenate[NDArray[Any], _P], ArrayLike],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> NDArray[Any]: ...

# 定义 apply_over_axes 函数，用于在指定轴上应用函数
def apply_over_axes(
    func: Callable[[NDArray[Any], int], NDArray[_SCT]],
    a: ArrayLike,
    axes: int | Sequence[int],
) -> NDArray[_SCT]: ...

# expand_dims 函数的函数重载，用于扩展数组的维度
@overload
def expand_dims(
    a: _ArrayLike[_SCT],
    axis: _ShapeLike,
) -> NDArray[_SCT]: ...
@overload
def expand_dims(
    a: ArrayLike,
    axis: _ShapeLike,
) -> NDArray[Any]: ...

# column_stack 函数的函数重载，用于按列堆叠输入数组
@overload
def column_stack(tup: Sequence[_ArrayLike[_SCT]]) -> NDArray[_SCT]: ...
@overload
def column_stack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

# dstack 函数的函数重载，用于按深度堆叠输入数组
@overload
def dstack(tup: Sequence[_ArrayLike[_SCT]]) -> NDArray[_SCT]: ...
@overload
def dstack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

# array_split 函数的函数重载，用于在指定轴上分割数组
@overload
def array_split(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[_SCT]]: ...
@overload
def array_split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[Any]]: ...

# split 函数的函数重载，用于在指定轴上分割数组
@overload
def split(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[_SCT]]: ...
@overload
def split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,


indices_or_sections: _ShapeLike,
axis: SupportsIndex = ...,
# 定义一个类型提示，表明函数返回一个包含 NDArray[Any] 元素的列表
) -> list[NDArray[Any]]: ...

# hsplit 函数的重载定义，接受一个 _ArrayLike[_SCT] 类型的数组 ary 和一个 _ShapeLike 类型的参数 indices_or_sections
@overload
def hsplit(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_SCT]]: ...
# hsplit 函数的重载定义，接受一个 ArrayLike 类型的数组 ary 和一个 _ShapeLike 类型的参数 indices_or_sections
@overload
def hsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

# vsplit 函数的重载定义，接受一个 _ArrayLike[_SCT] 类型的数组 ary 和一个 _ShapeLike 类型的参数 indices_or_sections
@overload
def vsplit(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_SCT]]: ...
# vsplit 函数的重载定义，接受一个 ArrayLike 类型的数组 ary 和一个 _ShapeLike 类型的参数 indices_or_sections
@overload
def vsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

# dsplit 函数的重载定义，接受一个 _ArrayLike[_SCT] 类型的数组 ary 和一个 _ShapeLike 类型的参数 indices_or_sections
@overload
def dsplit(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_SCT]]: ...
# dsplit 函数的重载定义，接受一个 ArrayLike 类型的数组 ary 和一个 _ShapeLike 类型的参数 indices_or_sections
@overload
def dsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

# get_array_wrap 函数的重载定义，接受多个参数，返回 _ArrayWrap 类型或者 None
@overload
def get_array_wrap(*args: _SupportsArrayWrap) -> _ArrayWrap: ...
# get_array_wrap 函数的重载定义，接受任意对象类型的参数，返回 None 或者 _ArrayWrap 类型
@overload
def get_array_wrap(*args: object) -> None | _ArrayWrap: ...

# kron 函数的重载定义，接受两个 _ArrayLikeBool_co 类型的参数 a 和 b，返回一个布尔类型的 NDArray
@overload
def kron(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
# kron 函数的重载定义，接受两个 _ArrayLikeUInt_co 类型的参数 a 和 b，返回一个无符号整数类型的 NDArray
@overload
def kron(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
# kron 函数的重载定义，接受两个 _ArrayLikeInt_co 类型的参数 a 和 b，返回一个有符号整数类型的 NDArray
@overload
def kron(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
# kron 函数的重载定义，接受两个 _ArrayLikeFloat_co 类型的参数 a 和 b，返回一个浮点数类型的 NDArray
@overload
def kron(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
# kron 函数的重载定义，接受两个 _ArrayLikeComplex_co 类型的参数 a 和 b，返回一个复数类型的 NDArray
@overload
def kron(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
# kron 函数的重载定义，接受一个 _ArrayLikeObject_co 类型的参数 a 和一个任意类型的参数 b，返回一个对象类型的 NDArray
@overload
def kron(a: _ArrayLikeObject_co, b: Any) -> NDArray[object_]: ...
# kron 函数的重载定义，接受一个任意类型的参数 a 和一个 _ArrayLikeObject_co 类型的参数 b，返回一个对象类型的 NDArray
@overload
def kron(a: Any, b: _ArrayLikeObject_co) -> NDArray[object_]: ...

# tile 函数的重载定义，接受一个 _ArrayLike[_SCT] 类型的数组 A 和一个整数或整数序列类型的参数 reps，返回一个与 A 类型相同的数组
@overload
def tile(
    A: _ArrayLike[_SCT],
    reps: int | Sequence[int],
) -> NDArray[_SCT]: ...
# tile 函数的重载定义，接受一个 ArrayLike 类型的数组 A 和一个整数或整数序列类型的参数 reps，返回一个任意类型的 NDArray
@overload
def tile(
    A: ArrayLike,
    reps: int | Sequence[int],
) -> NDArray[Any]: ...
```