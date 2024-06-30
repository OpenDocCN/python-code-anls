# `D:\src\scipysrc\scipy\scipy\fftpack\__init__.py`

```
# 定义模块的文档字符串，描述了 scipy.fftpack 模块的内容和用途
"""
=========================================================
Legacy discrete Fourier transforms (:mod:`scipy.fftpack`)
=========================================================

.. legacy::

   New code should use :mod:`scipy.fft`.

Fast Fourier Transforms (FFTs)
==============================

.. autosummary::
   :toctree: generated/

   fft - Fast (discrete) Fourier Transform (FFT)
   ifft - Inverse FFT
   fft2 - 2-D FFT
   ifft2 - 2-D inverse FFT
   fftn - N-D FFT
   ifftn - N-D inverse FFT
   rfft - FFT of strictly real-valued sequence
   irfft - Inverse of rfft
   dct - Discrete cosine transform
   idct - Inverse discrete cosine transform
   dctn - N-D Discrete cosine transform
   idctn - N-D Inverse discrete cosine transform
   dst - Discrete sine transform
   idst - Inverse discrete sine transform
   dstn - N-D Discrete sine transform
   idstn - N-D Inverse discrete sine transform

Differential and pseudo-differential operators
==============================================

.. autosummary::
   :toctree: generated/

   diff - Differentiation and integration of periodic sequences
   tilbert - Tilbert transform:         cs_diff(x,h,h)
   itilbert - Inverse Tilbert transform: sc_diff(x,h,h)
   hilbert - Hilbert transform:         cs_diff(x,inf,inf)
   ihilbert - Inverse Hilbert transform: sc_diff(x,inf,inf)
   cs_diff - cosh/sinh pseudo-derivative of periodic sequences
   sc_diff - sinh/cosh pseudo-derivative of periodic sequences
   ss_diff - sinh/sinh pseudo-derivative of periodic sequences
   cc_diff - cosh/cosh pseudo-derivative of periodic sequences
   shift - Shift periodic sequences

Helper functions
================

.. autosummary::
   :toctree: generated/

   fftshift - Shift the zero-frequency component to the center of the spectrum
   ifftshift - The inverse of `fftshift`
   fftfreq - Return the Discrete Fourier Transform sample frequencies
   rfftfreq - DFT sample frequencies (for usage with rfft, irfft)
   next_fast_len - Find the optimal length to zero-pad an FFT for speed

Note that ``fftshift``, ``ifftshift`` and ``fftfreq`` are numpy functions
exposed by ``fftpack``; importing them from ``numpy`` should be preferred.

Convolutions (:mod:`scipy.fftpack.convolve`)
============================================

.. module:: scipy.fftpack.convolve

.. autosummary::
   :toctree: generated/

   convolve
   convolve_z
   init_convolution_kernel
   destroy_convolve_cache
"""

# 定义模块的 __all__ 列表，指定了导出到外部的函数和类名
__all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
           'fft2','ifft2',
           'diff',
           'tilbert','itilbert','hilbert','ihilbert',
           'sc_diff','cs_diff','cc_diff','ss_diff',
           'shift',
           'fftfreq', 'rfftfreq',
           'fftshift', 'ifftshift',
           'next_fast_len',
           'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'
           ]

# 导入模块的各个子模块和函数
from ._basic import *
from ._pseudo_diffs import *
from ._helper import *
from ._realtransforms import *
# 引入已弃用的命名空间，将在 v2.0.0 版本中移除
from . import basic, helper, pseudo_diffs, realtransforms

# 从 scipy._lib._testutils 模块中导入 PytestTester 类
from scipy._lib._testutils import PytestTester

# 创建 PytestTester 类的实例 test，并将当前模块的名称传递给它
test = PytestTester(__name__)

# 删除 PytestTester 类，清理命名空间，避免后续代码中的不必要引用
del PytestTester
```