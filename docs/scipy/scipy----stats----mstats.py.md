# `D:\src\scipysrc\scipy\scipy\stats\mstats.py`

```
"""
===================================================================
Statistical functions for masked arrays (:mod:`scipy.stats.mstats`)
===================================================================

.. currentmodule:: scipy.stats.mstats

This module contains a large number of statistical functions that can
be used with masked arrays.

Most of these functions are similar to those in `scipy.stats` but might
have small differences in the API or in the algorithm used. Since this
is a relatively new package, some API changes are still possible.

Summary statistics
==================

.. autosummary::
   :toctree: generated/

   describe
   gmean
   hmean
   kurtosis
   mode
   mquantiles
   hdmedian
   hdquantiles
   hdquantiles_sd
   idealfourths
   plotting_positions
   meppf
   moment
   skew
   tmean
   tvar
   tmin
   tmax
   tsem
   variation
   find_repeats
   sem
   trimmed_mean
   trimmed_mean_ci
   trimmed_std
   trimmed_var

Frequency statistics
====================

.. autosummary::
   :toctree: generated/

   scoreatpercentile

Correlation functions
=====================

.. autosummary::
   :toctree: generated/

   f_oneway
   pearsonr
   spearmanr
   pointbiserialr
   kendalltau
   kendalltau_seasonal
   linregress
   siegelslopes
   theilslopes
   sen_seasonal_slopes

Statistical tests
=================

.. autosummary::
   :toctree: generated/

   ttest_1samp
   ttest_onesamp
   ttest_ind
   ttest_rel
   chisquare
   kstest
   ks_2samp
   ks_1samp
   ks_twosamp
   mannwhitneyu
   rankdata
   kruskal
   kruskalwallis
   friedmanchisquare
   brunnermunzel
   skewtest
   kurtosistest
   normaltest

Transformations
===============

.. autosummary::
   :toctree: generated/

   obrientransform
   trim
   trima
   trimmed_stde
   trimr
   trimtail
   trimboth
   winsorize
   zmap
   zscore

Other
=====

.. autosummary::
   :toctree: generated/

   argstoarray
   count_tied_groups
   msign
   compare_medians_ms
   median_cihs
   mjci
   mquantiles_cimj
   rsh

"""
# 导入_mstats_basic和_mstats_extras模块
from . import _mstats_basic
from . import _mstats_extras
# 从_mstats_basic模块导入所有变量到当前命名空间，禁止F403错误（flake8 linter）
from ._mstats_basic import *  # noqa: F403
# 从_mstats_extras模块导入所有变量到当前命名空间，禁止F403错误（flake8 linter）
from ._mstats_extras import *  # noqa: F403
# 导入scipy.stats中支持带有掩码数组输入的函数到当前命名空间
from scipy.stats import gmean, hmean, zmap, zscore, chisquare

# 定义模块中导出的全部变量列表为_mstats_basic模块中的全部变量列表加上_mstats_extras模块中的全部变量列表
__all__ = _mstats_basic.__all__ + _mstats_extras.__all__
# 向__all__列表中追加gmean、hmean、zmap、zscore、chisquare这几个变量名
__all__ += ['gmean', 'hmean', 'zmap', 'zscore', 'chisquare']
```