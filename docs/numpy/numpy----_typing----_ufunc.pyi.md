# `D:\src\scipysrc\numpy\numpy\_typing\_ufunc.pyi`

```py
"""
A module with private type-check-only `numpy.ufunc` subclasses.

The signatures of the ufuncs are too varied to reasonably type
with a single class. So instead, `ufunc` has been expanded into
four private subclasses, one for each combination of
`~ufunc.nin` and `~ufunc.nout`.

"""

# 导入必要的模块和类型定义
from typing import (
    Any,
    Generic,
    overload,
    TypeVar,
    Literal,
    SupportsIndex,
    Protocol,
)

# 导入需要的 numpy 类型和对象
from numpy import ufunc, _CastingKind, _OrderKACF
from numpy.typing import NDArray

# 导入私有模块中定义的类型和对象
from ._shape import _ShapeLike
from ._scalars import _ScalarLike_co
from ._array_like import ArrayLike, _ArrayLikeBool_co, _ArrayLikeInt_co
from ._dtype_like import DTypeLike

# 定义类型变量
_T = TypeVar("_T")
_2Tuple = tuple[_T, _T]
_3Tuple = tuple[_T, _T, _T]
_4Tuple = tuple[_T, _T, _T, _T]

# 定义用于泛型约束的类型变量
_NTypes = TypeVar("_NTypes", bound=int)
_IDType = TypeVar("_IDType", bound=Any)
_NameType = TypeVar("_NameType", bound=str)
_Signature = TypeVar("_Signature", bound=str)


class _SupportsArrayUFunc(Protocol):
    # 定义支持 __array_ufunc__ 协议的方法签名
    def __array_ufunc__(
        self,
        ufunc: ufunc,
        method: Literal[
            "__call__", "reduce", "reduceat", "accumulate", "outer", "at"
        ],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...


# NOTE: `reduce`, `accumulate`, `reduceat` and `outer` raise a ValueError for
# ufuncs that don't accept two input arguments and return one output argument.
# In such cases the respective methods are simply typed as `None`.

# NOTE: Similarly, `at` won't be defined for ufuncs that return
# multiple outputs; in such cases `at` is typed as `None`

# NOTE: If 2 output types are returned then `out` must be a
# 2-tuple of arrays. Otherwise `None` or a plain array are also acceptable

class _UFunc_Nin1_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    # 定义具有 1 个输入和 1 个输出的 ufunc 私有子类
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[2]: ...
    @property
    def signature(self) -> None: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...

    # 对 `__call__` 方法进行重载，定义了多种可能的参数和返回类型
    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        out: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
    ) -> Any: ...
    @overload
    # 定义一个特殊方法 __call__，使对象能够像函数一样被调用
    def __call__(
        self,
        __x1: ArrayLike,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
    ) -> NDArray[Any]: ...
    
    # 重载 __call__ 方法，支持另一种参数类型的调用方式
    @overload
    def __call__(
        self,
        __x1: _SupportsArrayUFunc,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
    ) -> Any: ...

    # 定义方法 `at`，用于在数组上执行赋值操作，指定索引位置
    def at(
        self,
        a: _SupportsArrayUFunc,
        indices: _ArrayLikeInt_co,
        /,
    ) -> None: ...
class _UFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> None: ...

    # 第一个重载：处理标量类型参数的函数调用
    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        out: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> Any: ...
    
    # 第二个重载：处理数组类型参数的函数调用
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> NDArray[Any]: ...

    # 将指定索引处的值更新为给定值
    def at(
        self,
        a: NDArray[Any],
        indices: _ArrayLikeInt_co,
        b: ArrayLike,
        /,
    ) -> None: ...

    # 对数组沿指定轴进行归约操作
    def reduce(
        self,
        array: ArrayLike,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
        keepdims: bool = ...,
        initial: Any = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...

    # 对数组沿指定轴进行累加操作
    def accumulate(
        self,
        array: ArrayLike,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...

    # 在指定位置执行归约操作
    def reduceat(
        self,
        array: ArrayLike,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...

    # 扩展 `**kwargs` 参数为明确的仅关键字参数
    # 第一个重载：处理标量类型参数的外积计算
    @overload
    def outer(
        self,
        A: _ScalarLike_co,
        B: _ScalarLike_co,
        /, *,
        out: None = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> Any: ...
    
    # 第二个重载：处理数组类型参数的外积计算
    @overload
    def outer(  # type: ignore[misc]
        self,
        A: ArrayLike,
        B: ArrayLike,
        /, *,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> NDArray[Any]: ...
    # 声明一个函数，该函数接受一些参数并且返回一个NumPy数组
    -> NDArray[Any]: ...
class _UFunc_Nin1_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    # 定义一个类，表示一个具有1个输入和2个输出的通用函数（ufunc），泛型参数包括名称类型、输入类型和标识类型

    @property
    def __name__(self) -> _NameType: ...
    # 返回函数名称，类型为泛型参数 _NameType

    @property
    def ntypes(self) -> _NTypes: ...
    # 返回函数可处理的输入数据类型，类型为泛型参数 _NTypes

    @property
    def identity(self) -> _IDType: ...
    # 返回函数的标识元素，类型为泛型参数 _IDType

    @property
    def nin(self) -> Literal[1]: ...
    # 返回函数的输入数量，固定为1

    @property
    def nout(self) -> Literal[2]: ...
    # 返回函数的输出数量，固定为2

    @property
    def nargs(self) -> Literal[3]: ...
    # 返回函数的参数数量，固定为3

    @property
    def signature(self) -> None: ...
    # 返回函数的签名信息，但此处返回类型为 None，可能未实现具体的签名功能

    @property
    def at(self) -> None: ...
    # 返回函数的 at 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def reduce(self) -> None: ...
    # 返回函数的 reduce 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def accumulate(self) -> None: ...
    # 返回函数的 accumulate 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def reduceat(self) -> None: ...
    # 返回函数的 reduceat 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def outer(self) -> None: ...
    # 返回函数的 outer 方法，但此处返回类型为 None，可能未实现具体的功能

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> _2Tuple[Any]: ...
    # 定义 __call__ 方法的重载，用于处理标量输入 __x1，返回一个包含两个元素的元组

    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...
    # 定义 __call__ 方法的重载，用于处理数组类型输入 __x1，返回一个包含两个元素的元组，支持输出参数

    @overload
    def __call__(
        self,
        __x1: _SupportsArrayUFunc,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> _2Tuple[Any]: ...
    # 定义 __call__ 方法的重载，用于处理支持 ufunc 的数组类型输入 __x1，返回一个包含两个元素的元组

class _UFunc_Nin2_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    # 定义一个类，表示一个具有2个输入和2个输出的通用函数（ufunc），泛型参数包括名称类型、输入类型和标识类型

    @property
    def __name__(self) -> _NameType: ...
    # 返回函数名称，类型为泛型参数 _NameType

    @property
    def ntypes(self) -> _NTypes: ...
    # 返回函数可处理的输入数据类型，类型为泛型参数 _NTypes

    @property
    def identity(self) -> _IDType: ...
    # 返回函数的标识元素，类型为泛型参数 _IDType

    @property
    def nin(self) -> Literal[2]: ...
    # 返回函数的输入数量，固定为2

    @property
    def nout(self) -> Literal[2]: ...
    # 返回函数的输出数量，固定为2

    @property
    def nargs(self) -> Literal[4]: ...
    # 返回函数的参数数量，固定为4

    @property
    def signature(self) -> None: ...
    # 返回函数的签名信息，但此处返回类型为 None，可能未实现具体的签名功能

    @property
    def at(self) -> None: ...
    # 返回函数的 at 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def reduce(self) -> None: ...
    # 返回函数的 reduce 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def accumulate(self) -> None: ...
    # 返回函数的 accumulate 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def reduceat(self) -> None: ...
    # 返回函数的 reduceat 方法，但此处返回类型为 None，可能未实现具体的功能

    @property
    def outer(self) -> None: ...
    # 返回函数的 outer 方法，但此处返回类型为 None，可能未实现具体的功能

    @overload
    # 以下省略部分重载定义，但均符合上述类似的结构
    # 定义一个方法重载，用于接受标量参数，并返回两个任意类型的值的元组
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _4Tuple[None | str] = ...,
    ) -> _2Tuple[Any]: ...

    # 方法重载，用于接受数组参数，并返回两个 NDArray[Any] 类型的值的元组
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _4Tuple[None | str] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...
# 定义一个私有类 _GUFunc_Nin2_Nout1，继承自 ufunc 泛型类，支持泛型类型 _NameType、_NTypes、_IDType 和 _Signature
class _GUFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType, _Signature]):  # type: ignore[misc]

    # 返回属性 __name__，类型为 _NameType，表示函数名或操作名称
    @property
    def __name__(self) -> _NameType: ...

    # 返回属性 ntypes，类型为 _NTypes，表示函数适用的数据类型
    @property
    def ntypes(self) -> _NTypes: ...

    # 返回属性 identity，类型为 _IDType，表示函数的身份元素
    @property
    def identity(self) -> _IDType: ...

    # 返回属性 nin，类型为 Literal[2]，表示函数接受的参数个数为两个
    @property
    def nin(self) -> Literal[2]: ...

    # 返回属性 nout，类型为 Literal[1]，表示函数返回的参数个数为一个
    @property
    def nout(self) -> Literal[1]: ...

    # 返回属性 nargs，类型为 Literal[3]，表示函数期望的参数个数为三个
    @property
    def nargs(self) -> Literal[3]: ...

    # 返回属性 signature，类型为 _Signature，表示函数的签名信息
    @property
    def signature(self) -> _Signature: ...

    # 返回属性 reduce，类型为 None，表示此函数不支持 reduce 操作
    @property
    def reduce(self) -> None: ...

    # 返回属性 accumulate，类型为 None，表示此函数不支持 accumulate 操作
    @property
    def accumulate(self) -> None: ...

    # 返回属性 reduceat，类型为 None，表示此函数不支持 reduceat 操作
    @property
    def reduceat(self) -> None: ...

    # 返回属性 outer，类型为 None，表示此函数不支持 outer 操作
    @property
    def outer(self) -> None: ...

    # 返回属性 at，类型为 None，表示此函数不支持 at 操作
    @property
    def at(self) -> None: ...

    # 重载函数调用操作，处理一维数组的标量情况；如果是多维数组，则返回 ndarray 类型
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None = ...,
        *,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        axes: list[_2Tuple[SupportsIndex]] = ...,
    ) -> Any: ...

    # 重载函数调用操作，处理一维数组的标量情况，并指定输出为 NDArray[Any] 或其元组
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: NDArray[Any] | tuple[NDArray[Any]],
        *,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        axes: list[_2Tuple[SupportsIndex]] = ...,
    ) -> NDArray[Any]: ...
```