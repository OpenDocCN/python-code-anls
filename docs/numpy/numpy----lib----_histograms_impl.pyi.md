# `.\numpy\numpy\lib\_histograms_impl.pyi`

```py
# 从 collections.abc 导入 Sequence 类，用于声明序列类型的抽象基类
from collections.abc import Sequence
# 从 typing 模块导入类型别名 Literal 为 L，以及其他类型注解
from typing import (
    Literal as L,
    Any,
    SupportsIndex,
)

# 从 numpy._typing 导入类型别名 NDArray 和 ArrayLike，用于数组和类数组的类型标注
from numpy._typing import (
    NDArray,
    ArrayLike,
)

# 定义 _BinKind 类型别名，表示直方图的分箱方法，可以是预定义的字符串集合
_BinKind = L[
    "stone",
    "auto",
    "doane",
    "fd",
    "rice",
    "scott",
    "sqrt",
    "sturges",
]

# 声明 __all__ 变量，用于指定模块中公开的所有名称
__all__: list[str]

# 定义直方图分箱边界计算函数 histogram_bin_edges
def histogram_bin_edges(
    a: ArrayLike,
    bins: _BinKind | SupportsIndex | ArrayLike = ...,
    range: None | tuple[float, float] = ...,
    weights: None | ArrayLike = ...,
) -> NDArray[Any]: ...

# 定义直方图计算函数 histogram
def histogram(
    a: ArrayLike,
    bins: _BinKind | SupportsIndex | ArrayLike = ...,
    range: None | tuple[float, float] = ...,
    density: bool = ...,
    weights: None | ArrayLike = ...,
) -> tuple[NDArray[Any], NDArray[Any]]: ...

# 定义多维直方图计算函数 histogramdd
def histogramdd(
    sample: ArrayLike,
    bins: SupportsIndex | ArrayLike = ...,
    range: Sequence[tuple[float, float]] = ...,
    density: None | bool = ...,
    weights: None | ArrayLike = ...,
) -> tuple[NDArray[Any], tuple[NDArray[Any], ...]]: ...
```