# `.\numpy\numpy\fft\_pocketfft.pyi`

```py
# 从 collections.abc 模块中导入 Sequence 类型
from collections.abc import Sequence
# 从 typing 模块中导入 Literal 别名 L
from typing import Literal as L
# 从 numpy 中导入特定类型和函数
from numpy import complex128, float64
# 从 numpy._typing 中导入 ArrayLike, NDArray 和 _ArrayLikeNumber_co 类型
from numpy._typing import ArrayLike, NDArray, _ArrayLikeNumber_co

# 定义 _NormKind 类型为 Literal 类型，包含 None, "backward", "ortho", "forward" 四种取值
_NormKind = L[None, "backward", "ortho", "forward"]

# 声明 __all__ 变量为字符串列表类型
__all__: list[str]

# 定义 fft 函数，计算快速傅里叶变换
def fft(
    a: ArrayLike,
    n: None | int = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 ifft 函数，计算反快速傅里叶变换
def ifft(
    a: ArrayLike,
    n: None | int = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 rfft 函数，计算实部快速傅里叶变换
def rfft(
    a: ArrayLike,
    n: None | int = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 irfft 函数，计算反实部快速傅里叶变换
def irfft(
    a: ArrayLike,
    n: None | int = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: None | NDArray[float64] = ...,
) -> NDArray[float64]: ...

# 定义 hfft 函数，计算 Hermite 傅里叶变换
# 输入数组必须兼容 `np.conjugate`
def hfft(
    a: _ArrayLikeNumber_co,
    n: None | int = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: None | NDArray[float64] = ...,
) -> NDArray[float64]: ...

# 定义 ihfft 函数，计算反 Hermite 傅里叶变换
def ihfft(
    a: ArrayLike,
    n: None | int = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 fftn 函数，计算 n 维快速傅里叶变换
def fftn(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 ifftn 函数，计算 n 维反快速傅里叶变换
def ifftn(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 rfftn 函数，计算 n 维实部快速傅里叶变换
def rfftn(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 irfftn 函数，计算 n 维反实部快速傅里叶变换
def irfftn(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[float64] = ...,
) -> NDArray[float64]: ...

# 定义 fft2 函数，计算二维快速傅里叶变换
def fft2(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 ifft2 函数，计算二维反快速傅里叶变换
def ifft2(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 rfft2 函数，计算二维实部快速傅里叶变换
def rfft2(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[complex128] = ...,
) -> NDArray[complex128]: ...

# 定义 irfft2 函数，计算二维反实部快速傅里叶变换
def irfft2(
    a: ArrayLike,
    s: None | Sequence[int] = ...,
    axes: None | Sequence[int] = ...,
    norm: _NormKind = ...,
    out: None | NDArray[float64] = ...,
) -> NDArray[float64]: ...
```