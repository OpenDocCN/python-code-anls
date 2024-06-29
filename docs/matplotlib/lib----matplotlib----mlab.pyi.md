# `D:\src\scipysrc\matplotlib\lib\matplotlib\mlab.pyi`

```py
# 从 collections.abc 模块导入 Callable 类型
from collections.abc import Callable
# 导入 functools 模块
import functools
# 从 typing 模块导入 Literal 类型
from typing import Literal

# 导入 numpy 库并使用 np 别名
import numpy as np
# 从 numpy.typing 模块导入 ArrayLike 类型
from numpy.typing import ArrayLike

# 定义 window_hanning 函数，接受 ArrayLike 类型参数 x，返回 ArrayLike 类型结果
def window_hanning(x: ArrayLike) -> ArrayLike: ...

# 定义 window_none 函数，接受 ArrayLike 类型参数 x，返回 ArrayLike 类型结果
def window_none(x: ArrayLike) -> ArrayLike: ...

# 定义 detrend 函数，接受 ArrayLike 类型参数 x，key 参数可选类型包括默认字符串或 Callable 或 None
# axis 参数可选类型为 int 或 None，默认为 ...
def detrend(
    x: ArrayLike,
    key: Literal["default", "constant", "mean", "linear", "none"]
    | Callable[[ArrayLike, int | None], ArrayLike]
    | None = ...,
    axis: int | None = ...,
) -> ArrayLike: ...

# 定义 detrend_mean 函数，接受 ArrayLike 类型参数 x，axis 参数可选类型为 int 或 None，默认为 ...
def detrend_mean(x: ArrayLike, axis: int | None = ...) -> ArrayLike: ...

# 定义 detrend_none 函数，接受 ArrayLike 类型参数 x，axis 参数可选类型为 int 或 None，默认为 ...
def detrend_none(x: ArrayLike, axis: int | None = ...) -> ArrayLike: ...

# 定义 detrend_linear 函数，接受 ArrayLike 类型参数 y，返回 ArrayLike 类型结果
def detrend_linear(y: ArrayLike) -> ArrayLike: ...

# 定义 psd 函数，接受 ArrayLike 类型参数 x，返回包含两个 ArrayLike 类型结果的元组
def psd(
    x: ArrayLike,
    NFFT: int | None = ...,
    Fs: float | None = ...,
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike, int | None], ArrayLike]
    | None = ...,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
    noverlap: int | None = ...,
    pad_to: int | None = ...,
    sides: Literal["default", "onesided", "twosided"] | None = ...,
    scale_by_freq: bool | None = ...,
) -> tuple[ArrayLike, ArrayLike]: ...

# 定义 csd 函数，接受两个 ArrayLike 类型参数 x 和 y，返回包含两个 ArrayLike 类型结果的元组
def csd(
    x: ArrayLike,
    y: ArrayLike | None,
    NFFT: int | None = ...,
    Fs: float | None = ...,
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike, int | None], ArrayLike]
    | None = ...,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
    noverlap: int | None = ...,
    pad_to: int | None = ...,
    sides: Literal["default", "onesided", "twosided"] | None = ...,
    scale_by_freq: bool | None = ...,
) -> tuple[ArrayLike, ArrayLike]: ...

# 使用 functools.partial 创建一个偏函数，返回一个 tuple[ArrayLike, ArrayLike] 类型的结果
complex_spectrum = functools.partial(tuple[ArrayLike, ArrayLike])
# 使用 functools.partial 创建一个偏函数，返回一个 tuple[ArrayLike, ArrayLike] 类型的结果
magnitude_spectrum = functools.partial(tuple[ArrayLike, ArrayLike])
# 使用 functools.partial 创建一个偏函数，返回一个 tuple[ArrayLike, ArrayLike] 类型的结果
angle_spectrum = functools.partial(tuple[ArrayLike, ArrayLike])
# 使用 functools.partial 创建一个偏函数，返回一个 tuple[ArrayLike, ArrayLike] 类型的结果
phase_spectrum = functools.partial(tuple[ArrayLike, ArrayLike])

# 定义 specgram 函数，接受 ArrayLike 类型参数 x，返回包含三个 ArrayLike 类型结果的元组
def specgram(
    x: ArrayLike,
    NFFT: int | None = ...,
    Fs: float | None = ...,
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike, int | None], ArrayLike]
    | None = ...,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
    noverlap: int | None = ...,
    pad_to: int | None = ...,
    sides: Literal["default", "onesided", "twosided"] | None = ...,
    scale_by_freq: bool | None = ...,
    mode: Literal["psd", "complex", "magnitude", "angle", "phase"] | None = ...,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...

# 定义 cohere 函数，接受两个 ArrayLike 类型参数 x 和 y，返回包含两个 ArrayLike 类型结果的元组
def cohere(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: int = ...,
    Fs: float = ...,
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike, int | None], ArrayLike]
    = ...,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike = ...,
    noverlap: int = ...,
    pad_to: int | None = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    scale_by_freq: bool | None = ...,
) -> tuple[ArrayLike, ArrayLike]: ...

# 定义 GaussianKDE 类
class GaussianKDE:
    # 类属性 dataset 类型为 ArrayLike
    dataset: ArrayLike
    # 类属性 dim 类型为 int
    dim: int
    # 类属性 num_dp 类型为 int
    num_dp: int
    # 类属性 factor 类型为 float
    factor: float
    data_covariance: ArrayLike
    data_inv_cov: ArrayLike
    covariance: ArrayLike
    inv_cov: ArrayLike
    norm_factor: float


# 定义类的成员变量，用于存储协方差、逆协方差矩阵及其相关的归一化因子
data_covariance: ArrayLike          # 数据的协方差矩阵
data_inv_cov: ArrayLike             # 数据的逆协方差矩阵
covariance: ArrayLike               # 协方差矩阵
inv_cov: ArrayLike                  # 逆协方差矩阵
norm_factor: float                  # 归一化因子


    def __init__(
        self,
        dataset: ArrayLike,
        bw_method: Literal["scott", "silverman"]
        | float
        | Callable[[GaussianKDE], float]
        | None = ...,
    ) -> None: ...


# 初始化方法，用于创建 GaussianKDE 类的实例
def __init__(
    self,
    dataset: ArrayLike,  # 数据集，类型为 ArrayLike
    bw_method: Literal["scott", "silverman"] | float | Callable[[GaussianKDE], float] | None = ...,  # 带宽选择方法，可以是字符串"scott"或"silverman"，也可以是浮点数或回调函数
) -> None:
    ...


    def scotts_factor(self) -> float: ...
    def silverman_factor(self) -> float: ...
    def covariance_factor(self) -> float: ...


# 计算不同带宽选择方法的系数因子
def scotts_factor(self) -> float:    # 计算 Scott's 方法的系数因子
def silverman_factor(self) -> float: # 计算 Silverman 方法的系数因子
def covariance_factor(self) -> float: # 计算协方差的系数因子
    ...


    def evaluate(self, points: ArrayLike) -> np.ndarray: ...
    def __call__(self, points: ArrayLike) -> np.ndarray: ...


# 评估核密度估计值的方法
def evaluate(self, points: ArrayLike) -> np.ndarray:  # 对给定数据点进行核密度估计并返回结果数组
def __call__(self, points: ArrayLike) -> np.ndarray:  # 调用对象时，对给定数据点进行核密度估计并返回结果数组
    ...
```