# `D:\src\scipysrc\scipy\scipy\stats\sampling.py`

```
"""
======================================================
Random Number Generators (:mod:`scipy.stats.sampling`)
======================================================

.. currentmodule:: scipy.stats.sampling

This module contains a collection of random number generators to sample
from univariate continuous and discrete distributions. It uses the
implementation of a C library called "UNU.RAN". The only exception is
RatioUniforms, which is a pure Python implementation of the
Ratio-of-Uniforms method.

Generators Wrapped
==================

For continuous distributions
----------------------------

.. autosummary::
   :toctree: generated/

   NumericalInverseHermite
       使用数值逆Hermite方法生成连续分布的随机数。
   NumericalInversePolynomial
       使用数值逆多项式方法生成连续分布的随机数。
   TransformedDensityRejection
       使用转换密度拒绝方法生成连续分布的随机数。
   SimpleRatioUniforms
       使用简单比例均匀法生成连续分布的随机数。
   RatioUniforms
       使用比例均匀法生成连续分布的随机数（纯Python实现）。

For discrete distributions
--------------------------

.. autosummary::
   :toctree: generated/

   DiscreteAliasUrn
       使用离散别名方法生成离散分布的随机数。
   DiscreteGuideTable
       使用离散引导表方法生成离散分布的随机数。

Warnings / Errors used in :mod:`scipy.stats.sampling`
-----------------------------------------------------

.. autosummary::
   :toctree: generated/

   UNURANError
       用于此模块的错误类。

Generators for pre-defined distributions
========================================

To easily apply the above methods for some of the continuous distributions
in :mod:`scipy.stats`, the following functionality can be used:

.. autosummary::
   :toctree: generated/

   FastGeneratorInversion
       提供了快速生成器反演的功能，用于一些预定义连续分布的生成。
"""

from ._sampling import FastGeneratorInversion, RatioUniforms  # noqa: F401
from ._unuran.unuran_wrapper import (  # noqa: F401
   TransformedDensityRejection,
   DiscreteAliasUrn,
   DiscreteGuideTable,
   NumericalInversePolynomial,
   NumericalInverseHermite,
   SimpleRatioUniforms,
   UNURANError
)

__all__ = ["NumericalInverseHermite", "NumericalInversePolynomial",
           "TransformedDensityRejection", "SimpleRatioUniforms",
           "RatioUniforms", "DiscreteAliasUrn", "DiscreteGuideTable",
           "UNURANError", "FastGeneratorInversion"]
```