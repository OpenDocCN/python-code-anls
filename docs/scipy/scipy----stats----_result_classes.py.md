# `D:\src\scipysrc\scipy\scipy\stats\_result_classes.py`

```
# This module exists only to allow Sphinx to generate docs
# for the result objects returned by some functions in stats
# _without_ adding them to the main stats documentation page.

"""
Result classes
--------------

.. currentmodule:: scipy.stats._result_classes

.. autosummary::
   :toctree: generated/

   RelativeRiskResult
   BinomTestResult
   TukeyHSDResult
   DunnettResult
   PearsonRResult
   FitResult
   OddsRatioResult
   TtestResult
   ECDFResult
   EmpiricalDistributionFunction

"""

# 导出模块中定义的所有结果类名，以便在模块外部能直接访问
__all__ = ['BinomTestResult', 'RelativeRiskResult', 'TukeyHSDResult',
           'PearsonRResult', 'FitResult', 'OddsRatioResult',
           'TtestResult', 'DunnettResult', 'ECDFResult',
           'EmpiricalDistributionFunction']

# 导入各个结果类，使它们在当前命名空间可用
from ._binomtest import BinomTestResult
from ._odds_ratio import OddsRatioResult
from ._relative_risk import RelativeRiskResult
from ._hypotests import TukeyHSDResult
from ._multicomp import DunnettResult
from ._stats_py import PearsonRResult, TtestResult
from ._fit import FitResult
from ._survival import ECDFResult, EmpiricalDistributionFunction
```