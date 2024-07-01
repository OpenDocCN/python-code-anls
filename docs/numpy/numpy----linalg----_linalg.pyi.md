# `.\numpy\numpy\linalg\_linalg.pyi`

```py
# 从 collections.abc 导入 Iterable 类，支持可迭代对象的抽象基类
from collections.abc import Iterable
# 从 typing 模块导入多个类型别名和函数重载相关的声明
from typing import (
    Literal as L,  # 引入类型文字类型别名 L
    overload,  # 用于函数重载的装饰器
    TypeVar,  # 泛型类型变量声明
    Any,  # 任意类型
    SupportsIndex,  # 支持索引操作的类型
    SupportsInt,  # 支持整数类型操作的类型
    NamedTuple,  # 命名元组类型
    Generic,  # 泛型基类
)

# 导入 numpy 库，并指定导入部分函数和类型
import numpy as np
# 从 numpy 中导入特定类型
from numpy import (
    generic,  # 泛型类型
    floating,  # 浮点数类型
    complexfloating,  # 复数浮点数类型
    signedinteger,  # 有符号整数类型
    unsignedinteger,  # 无符号整数类型
    timedelta64,  # 时间增量类型
    object_,  # 对象类型
    int32,  # 32位整数类型
    float64,  # 64位浮点数类型
    complex128,  # 128位复数类型
)

# 从 numpy.linalg 中导入 LinAlgError 错误类别
from numpy.linalg import LinAlgError as LinAlgError

# 导入 numpy._typing 中定义的多个类型别名
from numpy._typing import (
    NDArray,  # numpy 数组类型
    ArrayLike,  # 数组或类数组类型
    _ArrayLikeUnknown,  # 未知数组类型
    _ArrayLikeBool_co,  # 布尔类型的协变数组类型
    _ArrayLikeInt_co,  # 整数类型的协变数组类型
    _ArrayLikeUInt_co,  # 无符号整数类型的协变数组类型
    _ArrayLikeFloat_co,  # 浮点数类型的协变数组类型
    _ArrayLikeComplex_co,  # 复数类型的协变数组类型
    _ArrayLikeTD64_co,  # 时间增量类型的协变数组类型
    _ArrayLikeObject_co,  # 对象类型的协变数组类型
    DTypeLike,  # 数据类型或数据类型列表的类型别名
)

# 定义一个泛型类型变量 _T
_T = TypeVar("_T")
# 定义一个泛型类型变量 _ArrayType，限定为 NDArray 的子类
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])
# 定义一个协变泛型类型变量 _SCT
_SCT = TypeVar("_SCT", bound=generic, covariant=True)
# 定义另一个协变泛型类型变量 _SCT2
_SCT2 = TypeVar("_SCT2", bound=generic, covariant=True)

# 定义一个元组类型 _2Tuple，包含两个泛型类型变量 _T
_2Tuple = tuple[_T, _T]
# 定义一个字面值类型别名 L，包含字符串 "reduced", "complete", "r", "raw"
_ModeKind = L["reduced", "complete", "r", "raw"]

# 定义一个空字符串列表 __all__
__all__: list[str]

# 定义一个命名元组类 EigResult，包含 eigenvalues 和 eigenvectors 两个成员
class EigResult(NamedTuple):
    eigenvalues: NDArray[Any]  # 特征值数组
    eigenvectors: NDArray[Any]  # 特征向量数组

# 定义一个命名元组类 EighResult，包含 eigenvalues 和 eigenvectors 两个成员
class EighResult(NamedTuple):
    eigenvalues: NDArray[Any]  # 特征值数组
    eigenvectors: NDArray[Any]  # 特征向量数组

# 定义一个命名元组类 QRResult，包含 Q 和 R 两个成员
class QRResult(NamedTuple):
    Q: NDArray[Any]  # Q 矩阵
    R: NDArray[Any]  # R 矩阵

# 定义一个命名元组类 SlogdetResult，包含 sign 和 logabsdet 两个成员
class SlogdetResult(NamedTuple):
    # TODO: `sign` and `logabsdet` are scalars for input 2D arrays and
    # a `(x.ndim - 2)`` dimensionl arrays otherwise
    sign: Any  # 符号值
    logabsdet: Any  # 对数绝对值行列式

# 定义一个命名元组类 SVDResult，包含 U、S 和 Vh 三个成员
class SVDResult(NamedTuple):
    U: NDArray[Any]  # U 矩阵
    S: NDArray[Any]  # S 矩阵
    Vh: NDArray[Any]  # V 的共轭转置矩阵

# 函数重载定义：tensorsolve 函数，针对不同类型的数组求解张量方程
@overload
def tensorsolve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: None | Iterable[int] =...,
) -> NDArray[float64]: ...
@overload
def tensorsolve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: None | Iterable[int] =...,
) -> NDArray[floating[Any]]: ...
@overload
def tensorsolve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: None | Iterable[int] =...,
) -> NDArray[complexfloating[Any, Any]]: ...

# 函数重载定义：solve 函数，针对不同类型的数组求解线性方程组
@overload
def solve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
) -> NDArray[float64]: ...
@overload
def solve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
) -> NDArray[floating[Any]]: ...
@overload
def solve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
) -> NDArray[complexfloating[Any, Any]]: ...

# 函数重载定义：tensorinv 函数，针对不同类型的数组求解张量的逆
@overload
def tensorinv(
    a: _ArrayLikeInt_co,
    ind: int = ...,
) -> NDArray[float64]: ...
@overload
def tensorinv(
    a: _ArrayLikeFloat_co,
    ind: int = ...,
) -> NDArray[floating[Any]]: ...
@overload
def tensorinv(
    a: _ArrayLikeComplex_co,
    ind: int = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# 函数重载定义：inv 函数，针对不同类型的数组求解矩阵的逆
@overload
def inv(a: _ArrayLikeInt_co) -> NDArray[float64]: ...
@overload
def inv(a: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
@overload
def inv(a: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# TODO: The supported input and output dtypes are dependent on the value of `n`.
# For example: `n < 0` always casts integer types to float64
# 支持的输入和输出数据类型取决于 `n` 的值
# 例如：当 `n < 0` 时，总是将整数类型转换为 float64 类型
# 计算给定矩阵的整数、浮点数或复数的幂次方，返回结果作为多维数组
def matrix_power(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    n: SupportsIndex,
) -> NDArray[Any]: ...

# Cholesky 分解的函数重载，输入整数数组并返回浮点数数组
@overload
def cholesky(a: _ArrayLikeInt_co) -> NDArray[float64]: ...
# Cholesky 分解的函数重载，输入浮点数数组并返回浮点数数组
@overload
def cholesky(a: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
# Cholesky 分解的函数重载，输入复数数组并返回复数浮点数数组
@overload
def cholesky(a: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 计算两个数组的外积，返回结果作为多维数组
@overload
def outer(x1: _ArrayLikeUnknown, x2: _ArrayLikeUnknown) -> NDArray[Any]: ...
# 计算两个布尔数组的外积，返回结果作为布尔类型的多维数组
@overload
def outer(x1: _ArrayLikeBool_co, x2: _ArrayLikeBool_co) -> NDArray[np.bool]: ...
# 计算两个无符号整数数组的外积，返回结果作为无符号整数类型的多维数组
@overload
def outer(x1: _ArrayLikeUInt_co, x2: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...
# 计算两个有符号整数数组的外积，返回结果作为有符号整数类型的多维数组
@overload
def outer(x1: _ArrayLikeInt_co, x2: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
# 计算两个浮点数数组的外积，返回结果作为浮点数类型的多维数组
@overload
def outer(x1: _ArrayLikeFloat_co, x2: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
# 计算两个复数数组的外积，返回结果作为复数浮点数类型的多维数组
@overload
def outer(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
) -> NDArray[complexfloating[Any, Any]]: ...
# 计算两个时间间隔数组的外积，返回结果作为时间间隔类型的多维数组
@overload
def outer(
    x1: _ArrayLikeTD64_co,
    x2: _ArrayLikeTD64_co,
    out: None = ...,
) -> NDArray[timedelta64]: ...
# 计算两个对象数组的外积，返回结果作为对象类型的多维数组
@overload
def outer(x1: _ArrayLikeObject_co, x2: _ArrayLikeObject_co) -> NDArray[object_]: ...
# 计算两个复杂类型（复数、时间间隔或对象）数组的外积，返回结果作为相应类型的多维数组
@overload
def outer(
    x1: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    x2: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
) -> _ArrayType: ...

# QR 分解的函数重载，输入整数数组并返回 QRResult 结果
@overload
def qr(a: _ArrayLikeInt_co, mode: _ModeKind = ...) -> QRResult: ...
# QR 分解的函数重载，输入浮点数数组并返回 QRResult 结果
@overload
def qr(a: _ArrayLikeFloat_co, mode: _ModeKind = ...) -> QRResult: ...
# QR 分解的函数重载，输入复数数组并返回 QRResult 结果
@overload
def qr(a: _ArrayLikeComplex_co, mode: _ModeKind = ...) -> QRResult: ...

# 计算给定数组的特征值，输入整数数组并返回浮点数或复数类型的多维数组
@overload
def eigvals(a: _ArrayLikeInt_co) -> NDArray[float64] | NDArray[complex128]: ...
# 计算给定数组的特征值，输入浮点数数组并返回浮点数或复数类型的多维数组
@overload
def eigvals(a: _ArrayLikeFloat_co) -> NDArray[floating[Any]] | NDArray[complexfloating[Any, Any]]: ...
# 计算给定数组的特征值，输入复数数组并返回复数浮点数类型的多维数组
@overload
def eigvals(a: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 计算给定数组的特征值，返回结果作为整数或浮点数类型的多维数组
@overload
def eigvalsh(a: _ArrayLikeInt_co, UPLO: L["L", "U", "l", "u"] = ...) -> NDArray[float64]: ...
# 计算给定数组的特征值，返回结果作为浮点数类型的多维数组
@overload
def eigvalsh(a: _ArrayLikeComplex_co, UPLO: L["L", "U", "l", "u"] = ...) -> NDArray[floating[Any]]: ...

# 计算给定数组的特征值和右特征向量，返回结果作为 EigResult 对象
@overload
def eig(a: _ArrayLikeInt_co) -> EigResult: ...
# 计算给定数组的特征值和右特征向量，返回结果作为 EigResult 对象
@overload
def eig(a: _ArrayLikeFloat_co) -> EigResult: ...
# 计算给定数组的特征值和右特征向量，返回结果作为 EigResult 对象
@overload
def eig(a: _ArrayLikeComplex_co) -> EigResult: ...

# 计算给定数组的厄米特特征值分解，返回结果作为 EighResult 对象
@overload
def eigh(
    a: _ArrayLikeInt_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> EighResult: ...
# 计算给定数组的厄米特特征值分解，返回结果作为 EighResult 对象
@overload
def eigh(
    a: _ArrayLikeFloat_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> EighResult: ...
# 计算给定数组的厄米特特征值分解，返回结果作为 EighResult 对象
@overload
def eigh(
    a: _ArrayLikeComplex_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> EighResult: ...

# 对给定数组进行奇异值分解，返回结果作为 SVDResult 对象
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...
# 对给定数组进行奇异值分解，返回结果作为 SVDResult 对象
@overload
def svd(
    a: _ArrayLikeFloat_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...
# 对给定数组进行奇异值分解，返回结果作为 SVDResult 对象
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...
    full_matrices: bool = ...,  # 控制是否计算全尺寸的特征向量和特征值，布尔类型
    compute_uv: L[True] = ...,  # 控制是否计算左奇异向量 U 和右奇异向量 V^T，列表类型，值为 True
    hermitian: bool = ...,  # 控制是否假定输入矩阵是厄米特矩阵（共轭对称矩阵），布尔类型
# TODO: Defines a type signature for a function `svd` returning `SVDResult`
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...

# TODO: Defines a type signature for a function `svd` returning `NDArray[float64]`
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> NDArray[float64]: ...

# Defines a function `svdvals` that computes singular values of a matrix-like input `x`
def svdvals(
    x: _ArrayLikeInt_co | _ArrayLikeFloat_co | _ArrayLikeComplex_co
) -> NDArray[floating[Any]]: ...

# TODO: Defines a function `cond` that calculates the condition number of a complex array `x`
# The result type varies depending on the shape and optional parameter `p`
def cond(x: _ArrayLikeComplex_co, p: None | float | L["fro", "nuc"] = ...) -> Any: ...

# TODO: Defines a function `matrix_rank` that computes the rank of a complex array `A`
# The result type varies depending on the array shape and tolerance parameters
def matrix_rank(
    A: _ArrayLikeComplex_co,
    tol: None | _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
    *,
    rtol: None | _ArrayLikeFloat_co = ...,
) -> Any: ...

# TODO: Defines an overload of `pinv` for integer arrays, returning `NDArray[float64]`
@overload
def pinv(
    a: _ArrayLikeInt_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
) -> NDArray[float64]: ...

# TODO: Defines an overload of `pinv` for float arrays, returning `NDArray[floating[Any]]`
@overload
def pinv(
    a: _ArrayLikeFloat_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
) -> NDArray[floating[Any]]: ...

# TODO: Defines an overload of `pinv` for complex arrays, returning `NDArray[complexfloating[Any, Any]]`
@overload
def pinv(
    a: _ArrayLikeComplex_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# TODO: Defines a function `slogdet` that computes the sign and log determinant of a complex array `a`
# The result type varies depending on the array shape
def slogdet(a: _ArrayLikeComplex_co) -> SlogdetResult: ...

# TODO: Defines a function `det` that computes the determinant of a complex array `a`
# The result type varies depending on the array shape
def det(a: _ArrayLikeComplex_co) -> Any: ...

# TODO: Defines an overload of `lstsq` for integer arrays `a` and `b`, returning a tuple of arrays
@overload
def lstsq(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, rcond: None | float = ...) -> tuple[
    NDArray[float64],
    NDArray[float64],
    int32,
    NDArray[float64],
]: ...

# TODO: Defines an overload of `lstsq` for float arrays `a` and `b`, returning a tuple of arrays
@overload
def lstsq(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, rcond: None | float = ...) -> tuple[
    NDArray[floating[Any]],
    NDArray[floating[Any]],
    int32,
    NDArray[floating[Any]],
]: ...

# TODO: Defines an overload of `lstsq` for complex arrays `a` and `b`, returning a tuple of arrays
@overload
def lstsq(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, rcond: None | float = ...) -> tuple[
    NDArray[complexfloating[Any, Any]],
    NDArray[floating[Any]],
    int32,
    NDArray[floating[Any]],
]: ...

# TODO: Defines an overload of `norm` for computing the matrix norm of `x`
# The result type varies depending on the parameters `ord` and `axis`
@overload
def norm(
    x: ArrayLike,
    ord: None | float | L["fro", "nuc"] = ...,
    axis: None = ...,
    keepdims: bool = ...,
) -> floating[Any]: ...

# TODO: Defines an overload of `norm` for computing the matrix norm of `x` along specific axes
# The result type varies depending on the parameters `ord`, `axis`, and `keepdims`
@overload
def norm(
    x: ArrayLike,
    ord: None | float | L["fro", "nuc"] = ...,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] = ...,
    keepdims: bool = ...,
) -> Any: ...

# TODO: Defines a function `matrix_norm` for computing the matrix norm of `x`
# The result type varies depending on the parameters `ord` and `keepdims`
@overload
def matrix_norm(
    x: ArrayLike,
    ord: None | float | L["fro", "nuc"] = ...,
    keepdims: bool = ...,
) -> floating[Any]: ...

# TODO: Defines a function `matrix_norm` for computing the matrix norm of `x` with flexible output type
# The result type varies depending on the parameters `ord` and `keepdims`
@overload
def matrix_norm(
    x: ArrayLike,
    ord: None | float | L["fro", "nuc"] = ...,
    keepdims: bool = ...,
) -> Any: ...

# TODO: Defines a function `vector_norm` for computing the vector norm of `x`
# The result type varies depending on the parameters `axis` and `ord`
@overload
def vector_norm(
    x: ArrayLike,
    axis: None = ...,
    ord: None | float = ...,
): ...
    # keepdims参数，指定是否保持维度。此处应为布尔类型，表示函数是否保持输出的维度与输入一致。
    keepdims: bool = ...,
# 定义一个不返回值的函数签名，其参数类型为任意类型，返回类型为浮点数
) -> floating[Any]: ...

# 重载：计算向量范数
def vector_norm(
    x: ArrayLike,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] = ...,
    ord: None | float = ...,
    keepdims: bool = ...,
) -> Any: ...

# TODO: 返回一个标量或数组
def multi_dot(
    arrays: Iterable[_ArrayLikeComplex_co | _ArrayLikeObject_co | _ArrayLikeTD64_co],
    *,
    out: None | NDArray[Any] = ...,
) -> Any: ...

# 计算数组的对角线元素
def diagonal(
    x: ArrayLike,  # 至少是二维数组
    offset: SupportsIndex = ...,
) -> NDArray[Any]: ...

# 计算数组的迹（trace）
def trace(
    x: ArrayLike,  # 至少是二维数组
    offset: SupportsIndex = ...,
    dtype: DTypeLike = ...,
) -> Any: ...

# 重载：计算两个向量的叉乘
@overload
def cross(
    a: _ArrayLikeUInt_co,
    b: _ArrayLikeUInt_co,
    axis: int = ...,
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def cross(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axis: int = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def cross(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axis: int = ...,
) -> NDArray[floating[Any]]: ...
@overload
def cross(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axis: int = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# 重载：矩阵乘法
@overload
def matmul(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
) -> NDArray[signedinteger[Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
) -> NDArray[floating[Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
) -> NDArray[complexfloating[Any, Any]]: ...
```