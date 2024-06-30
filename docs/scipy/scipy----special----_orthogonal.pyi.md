# `D:\src\scipysrc\scipy\scipy\special\_orthogonal.pyi`

```
# 导入从 Python 3.7 开始的 __future__ 模块中的 annotations 特性，用于支持函数类型提示中的类型自引用
from __future__ import annotations
# 引入类型提示相关的模块和类型
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    overload,
)

# 引入 NumPy 库，并将其命名为 np
import numpy as np

# 定义整数类型 _IntegerType，可以是 int 或者 np.integer
_IntegerType = int | np.integer
# 定义浮点数类型 _FloatingType，可以是 float 或者 np.floating
_FloatingType = float | np.floating
# 定义一对点和权重的类型 _PointsAndWeights，包含两个 NumPy 数组
_PointsAndWeights = tuple[np.ndarray, np.ndarray]
# 定义一对点、权重和参数 mu 的类型 _PointsAndWeightsAndMu，包含一个 NumPy 数组和一个浮点数
_PointsAndWeightsAndMu = tuple[np.ndarray, np.ndarray, float]

# 定义数组样式的0维类型 _ArrayLike0D，可以是 bool、int、float、complex、str、bytes 或者 np.generic
_ArrayLike0D = bool | int | float | complex | str | bytes | np.generic

# 定义公开的函数列表 __all__，列出了本模块中公开的所有函数名
__all__ = [
    'legendre',
    'chebyt',
    'chebyu',
    'chebyc',
    'chebys',
    'jacobi',
    'laguerre',
    'genlaguerre',
    'hermite',
    'hermitenorm',
    'gegenbauer',
    'sh_legendre',
    'sh_chebyt',
    'sh_chebyu',
    'sh_jacobi',
    'roots_legendre',
    'roots_chebyt',
    'roots_chebyu',
    'roots_chebyc',
    'roots_chebys',
    'roots_jacobi',
    'roots_laguerre',
    'roots_genlaguerre',
    'roots_hermite',
    'roots_hermitenorm',
    'roots_gegenbauer',
    'roots_sh_legendre',
    'roots_sh_chebyt',
    'roots_sh_chebyu',
    'roots_sh_jacobi',
]

# 函数重载声明，计算 Jacobi 多项式的根和权重
@overload
def roots_jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
) -> _PointsAndWeights: ...

# 函数重载声明，计算 Jacobi 多项式的根和权重，支持不包含 mu 参数的情况
@overload
def roots_jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# 函数重载声明，计算 Jacobi 多项式的根、权重和 mu 参数
@overload
def roots_jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 函数重载声明，计算 Jacobi 伴随多项式的根和权重
@overload
def roots_sh_jacobi(
        n: _IntegerType,
        p1: _FloatingType,
        q1: _FloatingType,
) -> _PointsAndWeights: ...

# 函数重载声明，计算 Jacobi 伴随多项式的根和权重，支持不包含 mu 参数的情况
@overload
def roots_sh_jacobi(
        n: _IntegerType,
        p1: _FloatingType,
        q1: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# 函数重载声明，计算 Jacobi 伴随多项式的根、权重和 mu 参数
@overload
def roots_sh_jacobi(
        n: _IntegerType,
        p1: _FloatingType,
        q1: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 函数重载声明，计算广义拉盖尔多项式的根和权重
@overload
def roots_genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
) -> _PointsAndWeights: ...

# 函数重载声明，计算广义拉盖尔多项式的根和权重，支持不包含 mu 参数的情况
@overload
def roots_genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# 函数重载声明，计算广义拉盖尔多项式的根、权重和 mu 参数
@overload
def roots_genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 函数重载声明，计算拉盖尔多项式的根和权重
@overload
def roots_laguerre(n: _IntegerType) -> _PointsAndWeights: ...

# 函数重载声明，计算拉盖尔多项式的根和权重，支持不包含 mu 参数的情况
@overload
def roots_laguerre(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# 函数重载声明，计算拉盖尔多项式的根、权重和 mu 参数
@overload
def roots_laguerre(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 函数重载声明，计算埃尔米特多项式的根和权重
@overload
def roots_hermite(n: _IntegerType) -> _PointsAndWeights: ...

# 函数重载声明，计算埃尔米特多项式的根和权重，支持不包含 mu 参数的情况
@overload
def roots_hermite(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# 函数重载声明，计算埃尔米特多项式的根、权重和 mu 参数
@overload
def roots_hermite(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...
# 定义 roots_hermitenorm 函数，用于计算 Hermite 形式正交多项式的根和权重
def roots_hermitenorm(n: _IntegerType) -> _PointsAndWeights: ...

# roots_hermitenorm 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_hermitenorm(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_hermitenorm 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_hermitenorm(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_gegenbauer 函数，用于计算 Gegenbauer 形式正交多项式的根和权重
@overload
def roots_gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
) -> _PointsAndWeights: ...

# roots_gegenbauer 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_gegenbauer 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_chebyt 函数，用于计算 Chebyshev T 形式正交多项式的根和权重
@overload
def roots_chebyt(n: _IntegerType) -> _PointsAndWeights: ...

# roots_chebyt 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_chebyt(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_chebyt 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_chebyt(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_chebyu 函数，用于计算 Chebyshev U 形式正交多项式的根和权重
@overload
def roots_chebyu(n: _IntegerType) -> _PointsAndWeights: ...

# roots_chebyu 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_chebyu(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_chebyu 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_chebyu(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_chebyc 函数，用于计算 Chebyshev C 形式正交多项式的根和权重
@overload
def roots_chebyc(n: _IntegerType) -> _PointsAndWeights: ...

# roots_chebyc 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_chebyc(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_chebyc 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_chebyc(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_chebys 函数，用于计算 Chebyshev S 形式正交多项式的根和权重
@overload
def roots_chebys(n: _IntegerType) -> _PointsAndWeights: ...

# roots_chebys 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_chebys(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_chebys 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_chebys(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_sh_chebyt 函数，用于计算 spherical harmonics Chebyshev T 形式正交多项式的根和权重
@overload
def roots_sh_chebyt(n: _IntegerType) -> _PointsAndWeights: ...

# roots_sh_chebyt 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_sh_chebyt(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_sh_chebyt 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_sh_chebyt(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_sh_chebyu 函数，用于计算 spherical harmonics Chebyshev U 形式正交多项式的根和权重
@overload
def roots_sh_chebyu(n: _IntegerType) -> _PointsAndWeights: ...

# roots_sh_chebyu 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_sh_chebyu(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_sh_chebyu 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_sh_chebyu(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_legendre 函数，用于计算 Legendre 形式正交多项式的根和权重
@overload
def roots_legendre(n: _IntegerType) -> _PointsAndWeights: ...

# roots_legendre 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_legendre(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_legendre 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_legendre(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义 roots_sh_legendre 函数，用于计算 spherical harmonics Legendre 形式正交多项式的根和权重
@overload
def roots_sh_legendre(n: _IntegerType) -> _PointsAndWeights: ...

# roots_sh_legendre 函数的重载，当 mu 参数为 False 时，返回类型为 _PointsAndWeights
@overload
def roots_sh_legendre(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...

# roots_sh_legendre 函数的重载，当 mu 参数为 True 时，返回类型为 _PointsAndWeightsAndMu
@overload
def roots_sh_legendre(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...
# 定义一个函数 `roots_sh_legendre`，接受一个整数 `n` 和一个字面值 `mu`（值为 `True`），返回 `_PointsAndWeightsAndMu` 类型的结果
def roots_sh_legendre(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

# 定义一个类 `orthopoly1d`，继承自 `np.poly1d`
class orthopoly1d(np.poly1d):
    # 初始化方法，接受根数组 `roots`、权重数组 `weights`，可选参数 `hn`、`kn`，可选的权重函数 `wfunc`，可选的限制 `limits`，是否单项式 `monic`，以及评估函数 `eval_func`
    def __init__(
            self,
            roots: np.typing.ArrayLike,
            weights: np.typing.ArrayLike | None,
            hn: float = ...,
            kn: float = ...,
            wfunc = Optional[Callable[[float], float]],  # noqa: UP007
            limits = tuple[float, float] | None,
            monic: bool = ...,
            eval_func: np.ufunc = ...,
    ) -> None: ...
    
    # 限制属性的 getter 方法，返回限制元组 `(float, float)`
    @property
    def limits(self) -> tuple[float, float]: ...

    # 权重函数方法，接受参数 `x`，返回 `float` 类型的结果
    def weight_func(self, x: float) -> float: ...

    # 函数重载：第一个版本接受参数 `x`，返回 `Any` 类型的结果
    @overload
    def __call__(self, x: _ArrayLike0D) -> Any: ...

    # 函数重载：第二个版本接受参数 `x`，返回 `np.poly1d` 类型的结果，忽略重载重叠警告
    @overload
    def __call__(self, x: np.poly1d) -> np.poly1d: ...  # type: ignore[overload-overlap]

    # 函数重载：第三个版本接受参数 `x`，返回 `np.ndarray` 类型的结果
    @overload
    def __call__(self, x: np.typing.ArrayLike) -> np.ndarray: ...

# 定义函数 `legendre`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def legendre(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `chebyt`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def chebyt(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `chebyu`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def chebyu(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `chebyc`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def chebyc(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `chebys`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def chebys(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `jacobi`，接受整数 `n`、浮点数 `alpha`、`beta` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...

# 定义函数 `laguerre`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def laguerre(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `genlaguerre`，接受整数 `n`、浮点数 `alpha` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...

# 定义函数 `hermite`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def hermite(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `hermitenorm`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def hermitenorm(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `gegenbauer`，接受整数 `n`、浮点数 `alpha` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...

# 定义函数 `sh_legendre`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def sh_legendre(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `sh_chebyt`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def sh_chebyt(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `sh_chebyu`，接受整数 `n` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def sh_chebyu(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...

# 定义函数 `sh_jacobi`，接受整数 `n`、浮点数 `p`、`q` 和可选参数 `monic`（默认值为 `...`），返回 `orthopoly1d` 类型的结果
def sh_jacobi(
        n: _IntegerType,
        p: _FloatingType,
        q: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...

# 这些函数不是公共的，但仍然需要存根，因为它们在测试中被检查
# 定义函数 `_roots_hermite_asy`，接受整数 `n`，返回 `_PointsAndWeights` 类型的结果
def _roots_hermite_asy(n: _IntegerType) -> _PointsAndWeights: ...
```