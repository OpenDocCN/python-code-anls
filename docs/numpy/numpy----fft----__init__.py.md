# `.\numpy\numpy\fft\__init__.py`

```
"""
Discrete Fourier Transform (:mod:`numpy.fft`)
=============================================

.. currentmodule:: numpy.fft

The SciPy module `scipy.fft` is a more comprehensive superset
of ``numpy.fft``, which includes only a basic set of routines.

Standard FFTs
-------------

.. autosummary::
   :toctree: generated/

   fft       Discrete Fourier transform.
   ifft      Inverse discrete Fourier transform.
   fft2      Discrete Fourier transform in two dimensions.
   ifft2     Inverse discrete Fourier transform in two dimensions.
   fftn      Discrete Fourier transform in N-dimensions.
   ifftn     Inverse discrete Fourier transform in N dimensions.

Real FFTs
---------

.. autosummary::
   :toctree: generated/

   rfft      Real discrete Fourier transform.
   irfft     Inverse real discrete Fourier transform.
   rfft2     Real discrete Fourier transform in two dimensions.
   irfft2    Inverse real discrete Fourier transform in two dimensions.
   rfftn     Real discrete Fourier transform in N dimensions.
   irfftn    Inverse real discrete Fourier transform in N dimensions.

Hermitian FFTs
--------------

.. autosummary::
   :toctree: generated/

   hfft      Hermitian discrete Fourier transform.
   ihfft     Inverse Hermitian discrete Fourier transform.

Helper routines
---------------

.. autosummary::
   :toctree: generated/

   fftfreq   Discrete Fourier Transform sample frequencies.
   rfftfreq  DFT sample frequencies (for usage with rfft, irfft).
   fftshift  Shift zero-frequency component to center of spectrum.
   ifftshift Inverse of fftshift.


Background information
----------------------

Fourier analysis is fundamentally a method for expressing a function as a
sum of periodic components, and for recovering the function from those
components.  When both the function and its Fourier transform are
replaced with discretized counterparts, it is called the discrete Fourier
transform (DFT).  The DFT has become a mainstay of numerical computing in
part because of a very fast algorithm for computing it, called the Fast
Fourier Transform (FFT), which was known to Gauss (1805) and was brought
to light in its current form by Cooley and Tukey [CT]_.  Press et al. [NR]_
provide an accessible introduction to Fourier analysis and its
applications.

Because the discrete Fourier transform separates its input into
components that contribute at discrete frequencies, it has a great number
of applications in digital signal processing, e.g., for filtering, and in
this context the discretized input to the transform is customarily
referred to as a *signal*, which exists in the *time domain*.  The output
is called a *spectrum* or *transform* and exists in the *frequency
domain*.

Implementation details
----------------------

There are many ways to define the DFT, varying in the sign of the
exponent, normalization, etc.  In this implementation, the DFT is defined
as
"""
# 计算离散傅立叶变换（DFT）的公式定义，其中 a_m 是输入信号的复指数形式
A_k =  \\sum_{m=0}^{n-1} a_m \\exp\\left\\{-2\\pi i{mk \\over n}\\right\\}
\\qquad k = 0,\\ldots,n-1.

# DFT 通常用于复数输入和输出。单一频率分量线性频率为 f，表示为复指数形式
# 其中 \\Delta t 是采样间隔。
a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}

# 结果的值遵循“标准”顺序：如果 A = fft(a, n)，那么 A[0] 包含零频率项（信号总和），对于实数输入始终是纯实数。
# 然后 A[1:n/2] 包含正频率项，A[n/2+1:] 包含负频率项，按照逐渐减小的负频率顺序排列。
# 对于偶数个输入点，A[n/2] 表示正和负 Nyquist 频率，对于实数输入也是纯实数。
# 对于奇数个输入点，A[(n-1)/2] 包含最大的正频率，而 A[(n+1)/2] 包含最大的负频率。
# 函数 np.fft.fftfreq(n) 返回一个数组，给出输出中对应元素的频率。
# 函数 np.fft.fftshift(A) 将变换及其频率移位，将零频率分量放在中间位置，
# 而 np.fft.ifftshift(A) 撤消了该移位。

# 当输入 a 是时域信号，且 A = fft(a) 时，np.abs(A) 是其幅度谱，np.abs(A)**2 是其功率谱。
# 相位谱通过 np.angle(A) 获得。

# 反向离散傅立叶变换（IDFT）的定义
a_m = \\frac{1}{n}\\sum_{k=0}^{n-1}A_k\\exp\\left\\{2\\pi i{mk\\over n}\\right\\}
\\qquad m = 0,\\ldots,n-1.

# 它与正向变换的区别在于指数参数的符号和默认的归一化因子 1/n。

Type Promotion
--------------

# `numpy.fft` 将 `float32` 和 `complex64` 数组提升为分别为 `float64` 和 `complex128` 的数组。
# 若要不提升输入数组的 FFT 实现，请参见 `scipy.fftpack`。

Normalization
-------------

# 参数 `norm` 指示直接/反向变换对应的缩放方向及归一化因子。
# 默认归一化（`"backward"`）使得直接（正向）变换不缩放，而反向（逆向）变换缩放为 1/n。
# 通过将关键字参数 `norm` 设置为 `"ortho"` 可以获得单位化变换，使得直接和逆向变换均缩放为 1/\\sqrt{n}。
# 最后，将关键字参数 `norm` 设置为 `"forward"` 使得直接变换缩放为 1/n，而逆向变换不缩放（与默认的 `"backward"` 完全相反）。
# `None` 是默认选项 `"backward"` 的别名，用于向后兼容。

Real and Hermitian transforms
-----------------------------

# 当输入为纯实数时，其变换是 Hermite 对称的，即
# 从本地导入 _pocketfft 和 _helper 模块
from . import _pocketfft, _helper

# TODO: `numpy.fft.helper`` 在 NumPy 2.0 中被弃用。一旦下游库迁移到 `numpy.fft`，应删除此导入语句。
# 从本地导入 helper 模块（已弃用）
from . import helper

# 导入 _pocketfft 模块中所有公共名称
from ._pocketfft import *

# 导入 _helper 模块中所有公共名称
from ._helper import *

# 将 _pocketfft 模块的所有公共名称添加到 __all__ 列表中
__all__ = _pocketfft.__all__.copy()

# 将 _helper 模块的所有公共名称也添加到 __all__ 列表中
__all__ += _helper.__all__

# 从 numpy._pytesttester 导入 PytestTester 类
from numpy._pytesttester import PytestTester

# 创建一个 PytestTester 实例并将其赋给名为 test 的变量
test = PytestTester(__name__)

# 删除 PytestTester 类，避免污染命名空间
del PytestTester
```