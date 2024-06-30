# `D:\src\scipysrc\scipy\scipy\spatial\distance.pyi`

```
# 从未来导入类型注解相关模块，用于支持注解中的特定语法
from __future__ import annotations
# 导入多种类型注解，包括重载函数和字面量类型等
from typing import (overload, Any, SupportsFloat, Literal, Protocol, SupportsIndex)

# 导入 NumPy 库，使用简称 np
import numpy as np
# 导入 NumPy 的类型标注，包括 ArrayLike 和 NDArray
from numpy.typing import ArrayLike, NDArray

# _FloatValue 可以接受的类型定义，兼容于 np.float64 数组的赋值
_FloatValue = None | str | bytes | SupportsFloat | SupportsIndex

# 定义 Protocol _MetricCallback1，表示具有指定调用特征的类型
class _MetricCallback1(Protocol):
    def __call__(
        self, __XA: NDArray[Any], __XB: NDArray[Any]
    ) -> _FloatValue: ...

# 定义 Protocol _MetricCallback2，具有相似但包含关键字参数的调用特征
class _MetricCallback2(Protocol):
    def __call__(
        self, __XA: NDArray[Any], __XB: NDArray[Any], **kwargs: Any
    ) -> _FloatValue: ...

# _MetricCallback 表示可以是 _MetricCallback1 或 _MetricCallback2 类型的变量
_MetricCallback = _MetricCallback1 | _MetricCallback2

# _MetricKind 表示支持的度量标准的字面量类型，如布雷卡蒂斯、曼哈顿等
_MetricKind = Literal[
    'braycurtis',
    'canberra',
    'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch',
    'cityblock', 'cblock', 'cb', 'c',
    'correlation', 'co',
    'cosine', 'cos',
    'dice',
    'euclidean', 'euclid', 'eu', 'e',
    'hamming', 'hamm', 'ha', 'h',
    'minkowski', 'mi', 'm', 'pnorm',
    'jaccard', 'jacc', 'ja', 'j',
    'jensenshannon', 'js',
    'kulczynski1',
    'mahalanobis', 'mahal', 'mah',
    'rogerstanimoto',
    'russellrao',
    'seuclidean', 'se', 's',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean', 'sqe', 'sqeuclid',
    'yule',
]

# 函数注解部分

# 函数 braycurtis 接受 ArrayLike 类型的参数 u, v 和可选的 w，返回 np.float64 类型
def braycurtis(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64: ...

# 函数 canberra 接受 ArrayLike 类型的参数 u, v 和可选的 w，返回 np.float64 类型
def canberra(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64: ...

# cdist 函数的第一个重载版本，接受 XA 和 XB 两个 ArrayLike 参数，还有 metric 参数和其它关键字参数，
# 返回类型为 NDArray[np.floating[Any]]
@overload
def cdist(
    XA: ArrayLike,
    XB: ArrayLike,
    metric: _MetricKind = ...,
    *,
    out: None | NDArray[np.floating[Any]] = ...,
    p: float = ...,
    w: ArrayLike | None = ...,
    V: ArrayLike | None = ...,
    VI: ArrayLike | None = ...,
) -> NDArray[np.floating[Any]]: ...

# cdist 函数的第二个重载版本，接受 XA 和 XB 两个 ArrayLike 参数，还有 metric 参数作为 _MetricCallback 类型，
# 返回类型为 NDArray[np.floating[Any]]
@overload
def cdist(
    XA: ArrayLike,
    XB: ArrayLike,
    metric: _MetricCallback,
    *,
    out: None | NDArray[np.floating[Any]] = ...,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]: ...

# chebyshev 函数接受 ArrayLike 类型的参数 u, v 和可选的 w，返回任意类型的结果
def chebyshev(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> Any: ...

# cityblock 函数接受 ArrayLike 类型的参数 u, v 和可选的 w，返回任意类型的结果
def cityblock(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> Any: ...

# correlation 函数接受 ArrayLike 类型的参数 u, v，可选的 w 和 centered 参数，返回 np.float64 类型
def correlation(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ..., centered: bool = ...
) -> np.float64: ...

# cosine 函数接受 ArrayLike 类型的参数 u, v 和可选的 w，返回 np.float64 类型
def cosine(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64: ...

# dice 函数接受 ArrayLike 类型的参数 u, v 和可选的 w，返回 float 类型
def dice(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> float: ...

# directed_hausdorff 函数接受 ArrayLike 类型的参数 u, v 和可选的 seed，返回元组类型 (float, int, int)
def directed_hausdorff(
    u: ArrayLike, v: ArrayLike, seed: int | None = ...
) -> tuple[float, int, int]: ...

# euclidean 函数接受 ArrayLike 类型的参数 u, v 和可选的 w，返回 np.float64 类型
def euclidean(
    # 定义函数的参数：u、v 是类数组对象（可能是列表、元组、NumPy 数组等），w 是可选的类数组对象或者 None
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
# 定义一个函数签名，指定函数返回类型为 float
def hamming(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64:
    # TODO: Implement Hamming distance calculation between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 bool
def is_valid_dm(
    D: ArrayLike,
    tol: float = ...,
    throw: bool = ...,
    name: str | None = ...,
    warning: bool = ...
) -> bool:
    # TODO: Validate a distance matrix D with optional tolerance, throwing errors or warnings as specified
    ...

# 定义一个函数签名，指定函数返回类型为 bool
def is_valid_y(
    y: ArrayLike,
    warning: bool = ...,
    throw: bool = ...,
    name: str | None = ...
) -> bool:
    # TODO: Validate a condensed distance matrix vector y, optionally throwing errors or warnings
    ...

# 定义一个函数签名，指定函数返回类型为 np.float64
def jaccard(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64:
    # TODO: Calculate Jaccard distance/similarity between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 np.float64
def jensenshannon(
    p: ArrayLike, q: ArrayLike, base: float | None = ...
) -> np.float64:
    # TODO: Calculate Jensen-Shannon divergence between probability distributions p and q
    ...

# 定义一个函数签名，指定函数返回类型为 np.float64
def kulczynski1(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64:
    # TODO: Calculate Kulczynski similarity coefficient between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 np.float64
def mahalanobis(
    u: ArrayLike, v: ArrayLike, VI: ArrayLike
) -> np.float64:
    # TODO: Calculate Mahalanobis distance between vectors u and v using inverse covariance matrix VI
    ...

# 定义一个函数签名，指定函数返回类型为 float
def minkowski(
    u: ArrayLike, v: ArrayLike, p: float = ..., w: ArrayLike | None = ...
) -> float:
    # TODO: Calculate Minkowski distance between arrays u and v with parameter p and optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 int
def num_obs_dm(d: ArrayLike) -> int:
    # TODO: Determine the number of observations in a distance matrix d
    ...

# 定义一个函数签名，指定函数返回类型为 int
def num_obs_y(Y: ArrayLike) -> int:
    # TODO: Determine the number of observations in a condensed distance matrix vector Y
    ...

# 添加 `metric`-specific overloads for the pdist function using type annotations and optional arguments
@overload
def pdist(
    X: ArrayLike,
    metric: _MetricKind = ...,
    *,
    out: None | NDArray[np.floating[Any]] = ...,
    p: float = ...,
    w: ArrayLike | None = ...,
    V: ArrayLike | None = ...,
    VI: ArrayLike | None = ...,
) -> NDArray[np.floating[Any]]:
    ...

@overload
def pdist(
    X: ArrayLike,
    metric: _MetricCallback,
    *,
    out: None | NDArray[np.floating[Any]] = ...,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    ...

# 定义一个函数签名，指定函数返回类型为 float
def seuclidean(
    u: ArrayLike, v: ArrayLike, V: ArrayLike
) -> float:
    # TODO: Calculate standardized Euclidean distance between arrays u and v using variance vector V
    ...

# 定义一个函数签名，指定函数返回类型为 float
def sokalmichener(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> float:
    # TODO: Calculate Sokal-Michener similarity coefficient between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 np.float64
def sokalsneath(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64:
    # TODO: Calculate Sokal-Sneath similarity coefficient between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 np.float64
def sqeuclidean(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> np.float64:
    # TODO: Calculate squared Euclidean distance between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 NDArray[Any]
def squareform(
    X: ArrayLike,
    force: Literal["no", "tomatrix", "tovector"] = ...,
    checks: bool = ...
) -> NDArray[Any]:
    # TODO: Convert a condensed distance matrix X to a square-form distance matrix or vector based on options
    ...

# 定义一个函数签名，指定函数返回类型为 float
def rogerstanimoto(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> float:
    # TODO: Calculate Rogers-Tanimoto similarity coefficient between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 float
def russellrao(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> float:
    # TODO: Calculate Russell-Rao similarity coefficient between arrays u and v with optional weights w
    ...

# 定义一个函数签名，指定函数返回类型为 float
def yule(
    u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...
) -> float:
    # TODO: Calculate Yule similarity coefficient between arrays u and v with optional weights w
    ...
```