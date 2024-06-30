# `D:\src\scipysrc\scipy\scipy\stats\__init__.py`

```
"""
.. _statsrefmanual:

==========================================
Statistical functions (:mod:`scipy.stats`)
==========================================

.. currentmodule:: scipy.stats

This module contains a large number of probability distributions,
summary and frequency statistics, correlation functions and statistical
tests, masked statistics, kernel density estimation, quasi-Monte Carlo
functionality, and more.

Statistics is a very large area, and there are topics that are out of scope
for SciPy and are covered by other packages. Some of the most important ones
are:

- `statsmodels <https://www.statsmodels.org/stable/index.html>`__:
  regression, linear models, time series analysis, extensions to topics
  also covered by ``scipy.stats``.
- `Pandas <https://pandas.pydata.org/>`__: tabular data, time series
  functionality, interfaces to other statistical languages.
- `PyMC <https://docs.pymc.io/>`__: Bayesian statistical
  modeling, probabilistic machine learning.
- `scikit-learn <https://scikit-learn.org/>`__: classification, regression,
  model selection.
- `Seaborn <https://seaborn.pydata.org/>`__: statistical data visualization.
- `rpy2 <https://rpy2.github.io/>`__: Python to R bridge.


Probability distributions
=========================

Each univariate distribution is an instance of a subclass of `rv_continuous`
(`rv_discrete` for discrete distributions):

.. autosummary::
   :toctree: generated/

   rv_continuous
   rv_discrete
   rv_histogram

Continuous distributions
------------------------

The ``fit`` method of the univariate continuous distributions uses
maximum likelihood estimation to fit the distribution to a data set.
The ``fit`` method can accept regular data or *censored data*.
Censored data is represented with instances of the `CensoredData`
class.

.. autosummary::
   :toctree: generated/

   CensoredData


Multivariate distributions
--------------------------

.. autosummary::
   :toctree: generated/

   multivariate_normal    -- Multivariate normal distribution
   matrix_normal          -- Matrix normal distribution
   dirichlet              -- Dirichlet
   dirichlet_multinomial  -- Dirichlet multinomial distribution
   wishart                -- Wishart
   invwishart             -- Inverse Wishart
   multinomial            -- Multinomial distribution
   special_ortho_group    -- SO(N) group
   ortho_group            -- O(N) group
   unitary_group          -- U(N) group
   random_correlation     -- random correlation matrices
   multivariate_t         -- Multivariate t-distribution
   multivariate_hypergeom -- Multivariate hypergeometric distribution
   random_table           -- Distribution of random tables with given marginals
   uniform_direction      -- Uniform distribution on S(N-1)
   vonmises_fisher        -- Von Mises-Fisher distribution

`scipy.stats.multivariate_normal` methods accept instances
of the following class to represent the covariance.
"""
# Covariance - 协方差矩阵的表示
class Covariance:
    pass


# 离散分布

## 伯努利分布
def bernoulli():
    pass

## 贝塔二项分布
def betabinom():
    pass

## 贝塔负二项分布
def betanbinom():
    pass

## 二项分布
def binom():
    pass

## 波尔兹曼分布（截断离散指数分布）
def boltzmann():
    pass

## 离散拉普拉斯分布
def dlaplace():
    pass

## 几何分布
def geom():
    pass

## 超几何分布
def hypergeom():
    pass

## 对数分布（对数级数分布）
def logser():
    pass

## 负二项分布
def nbinom():
    pass

## 费舍尔非中心超几何分布
def nchypergeom_fisher():
    pass

## 瓦伦尼乌斯非中心超几何分布
def nchypergeom_wallenius():
    pass

## 负超几何分布
def nhypergeom():
    pass

## 普朗克分布（离散指数分布）
def planck():
    pass

## 泊松分布
def poisson():
    pass

## 离散均匀分布
def randint():
    pass

## 斯凯勒姆分布
def skellam():
    pass

## 尤尔-西蒙分布
def yulesimon():
    pass

## 齐普夫分布（齐普夫分布）
def zipf():
    pass

## 齐普夫分布（齐普夫分布）
def zipfian():
    pass


# 统计函数概览

## 描述性统计
def describe():
    pass

## 几何平均数
def gmean():
    pass

## 调和平均数
def hmean():
    pass

## 幂平均数
def pmean():
    pass

## 峰度（费舍尔或皮尔逊峰度）
def kurtosis():
    pass

## 众数
def mode():
    pass

## 中心矩
def moment():
    pass

## 期望分位数
def expectile():
    pass

## 偏度
def skew():
    pass

## kstat
def kstat():
    pass

## kstatvar
def kstatvar():
    pass

## 截尾算术平均数
def tmean():
    pass

## 截尾方差
def tvar():
    pass

## tmin
def tmin():
    pass

## tmax
def tmax():
    pass

## tstd
def tstd():
    pass

## tsem
def tsem():
    pass

## 变异系数
def variation():
    pass

## 查找重复值
def find_repeats():
    pass

## 排序数据
def rankdata():
    pass

## 系数校正
def tiecorrect():
    pass

## 截尾均值
def trim_mean():
    pass

## 几何标准偏差
def gstd():
    pass

## 四分位距
def iqr():
    pass

## 标准误差
def sem():
    pass

## 贝叶斯估计
def bayes_mvs():
    pass

## mvsdist
def mvsdist():
    pass

## 熵
def entropy():
    pass

## 差分熵
def differential_entropy():
    pass

## 中位数绝对偏差
def median_abs_deviation():
    pass


# 频率统计

## 累积频率
def cumfreq():
    pass

## 百分位数得分
def percentileofscore():
    pass

## 给定百分位数的分数
def scoreatpercentile():
    pass

## 相对频率
def relfreq():
    pass


# 二元统计

## 计算数据集的分箱统计
def binned_statistic():
    pass

## 计算二维数据集的分箱统计
def binned_statistic_2d():
    pass

## 计算 d 维数据集的分箱统计
def binned_statistic_dd():
    pass


# 假设检验及相关函数
# SciPy 提供多种假设检验函数，返回检验统计量、p 值，其中一些还返回置信区间或其他相关信息。
# 以下是关于不同统计检验的文档说明，按照功能分类进行组织
# 
# 单样本检验 / 成对样本检验
# --------------------------------------
# 单样本检验通常用于评估单个样本是否来自指定分布或具有指定属性的分布（例如零均值）。
#
# .. autosummary::
#    :toctree: generated/
#
#    ttest_1samp          # 单样本 t 检验
#    binomtest            # 二项分布检验
#    quantile_test        # 分位数检验
#    skewtest             # 偏度检验
#    kurtosistest         # 峰度检验
#    normaltest           # 正态性检验
#    jarque_bera          # Jarque-Bera 正态性检验
#    shapiro              # Shapiro-Wilk 正态性检验
#    anderson             # Anderson-Darling 正态性检验
#    cramervonmises       # Cramér-von Mises 正态性检验
#    ks_1samp             # 单样本 Kolmogorov-Smirnov 检验
#    goodness_of_fit      # 拟合优度检验
#    chisquare            # 卡方检验
#    power_divergence     # 功率差异检验
#
# 成对样本检验通常用于评估两个样本是否来自相同分布；它们与下面的独立样本检验不同，因为每个观察
# 在一个样本中被视为与另一个样本中的密切相关的观察（例如，控制了样本内但未控制样本间的环境因
# 素的情况下）。它们也可以被解释或用作单样本检验（例如，对成对观测之间的差异的均值或中位数的检
# 验）。
#
# .. autosummary::
#    :toctree: generated/
#
#    ttest_rel            # 成对样本 t 检验
#    wilcoxon             # Wilcoxon 符号秩检验
#
# 关联/相关性检验
# -----------------------------
# 这些检验通常用于评估多个样本中或多变量观测中配对观测之间是否存在关系（例如线性关系）。
#
# .. autosummary::
#    :toctree: generated/
#
#    linregress           # 线性回归
#    pearsonr             # Pearson 相关系数及其显著性
#    spearmanr            # Spearman 等级相关系数
#    pointbiserialr       # 点二列相关系数
#    kendalltau           # Kendall Tau 相关系数
#    weightedtau          # 加权 Kendall Tau 相关系数
#    somersd              # Somers' D 关联系数
#    siegelslopes         # Siegel 斜率估计
#    theilslopes          # Theil 斜率估计
#    page_trend_test      # Page 趋势检验
#    multiscale_graphcorr  # 多尺度图相关性检验
#
# 这些关联检验适用于采用列联表形式的样本。支持函数在 `scipy.stats.contingency` 中提供。
#
# .. autosummary::
#    :toctree: generated/
#
#    chi2_contingency     # 卡方独立性检验
#    fisher_exact         # Fisher 精确检验
#    barnard_exact        # Barnard 精确检验
#    boschloo_exact       # Boschloo 精确检验
#
# 独立样本检验
# ------------------------
# 独立样本检验通常用于评估多个样本是否独立地来自相同分布或具有共享属性的不同分布（例如均值相等）。
# 一些测试专门用于比较两个样本。
#
# .. autosummary::
#    :toctree: generated/
#
#    ttest_ind_from_stats # 从统计量进行独立样本 t 检验
#    poisson_means_test   # 泊松分布均值检验
#    ttest_ind            # 独立样本 t 检验
#    mannwhitneyu         # Mann-Whitney U 检验
#    bws_test             # BWS 检验
#    ranksums             # Wilcoxon 符号秩和检验
#    brunnermunzel        # Brunner-Munzel 检验
#    mood                 # Mood 检验
#    ansari               # Ansari-Bradley 检验
#    cramervonmises_2samp # Cramér-von Mises 二样本检验
#    epps_singleton_2samp # Epps-Singleton 二样本检验
#    ks_2samp             # 两样本 Kolmogorov-Smirnov 检验
#    kstest               # Kolmogorov-Smirnov 单样本检验
#
# 其他一些测试广泛适用于多个样本。
#
# .. autosummary::
#    :toctree: generated/
#
#    f_oneway             # 单因素方差分析
#    tukey_hsd            # Tukey HSD 检验
#    dunnett              # Dunnett 多重比较检验
#    kruskal              # Kruskal-Wallis 检验
#    alexandergovern      # Alexander-Govern 检验
#    fligner              # Fligner-Killeen 检验
#    levene               # Levene 方差齐性检验
#    bartlett             # Bartlett 方差齐性检验
#    median_test          # 中位数检验
#    friedmanchisquare    # Friedman 秩和检验
#    anderson_ksamp       # Anderson-Darling K 样本检验
#
# 重抽样和蒙特卡罗方法
# ------------------------
# 这部分将在后续的文档中继续介绍。
#
# 自动摘要文档，包含了多个子模块和函数的说明，每个部分描述了其包含的功能类别和具体函数。
# 这些说明主要用于自动生成文档或帮助用户了解可以在 `scipy.stats` 模块中找到的统计学功能和相关警告信息。

# 用于生成 p-value 和置信区间结果的函数，通常能在更广泛的条件下准确生成结果，但可能需要更大的计算资源和随机结果。
.. autosummary::
   :toctree: generated/

   monte_carlo_test
   permutation_test
   bootstrap
   power

# 这些对象的实例可以传递给一些假设检验函数，用于执行假设检验的重抽样或蒙特卡洛版本。
.. autosummary::
   :toctree: generated/

   MonteCarloMethod
   PermutationMethod
   BootstrapMethod

# 多重假设检验和元分析相关的函数，用于整体评估多个检验结果。
.. autosummary::
   :toctree: generated/

   combine_pvalues
   false_discovery_control

# 不属于以上分类但与上述检验相关的功能。
Quasi-Monte Carlo
=================

# 用于处理列联表的函数。
Contingency Tables
==================

# 用于处理带有掩码的统计函数。
Masked statistics functions
===========================

# 其他统计功能
Other statistical functionality
===============================

# 变换功能
Transformations
---------------

.. autosummary::
   :toctree: generated/

   boxcox
   boxcox_normmax
   boxcox_llf
   yeojohnson
   yeojohnson_normmax
   yeojohnson_llf
   obrientransform
   sigmaclip
   trimboth
   trim1
   zmap
   zscore
   gzscore

# 统计距离
Statistical distances
---------------------

.. autosummary::
   :toctree: generated/

   wasserstein_distance
   wasserstein_distance_nd
   energy_distance

# 抽样
Sampling
--------

# 拟合 / 生存分析
Fitting / Survival Analysis
---------------------------

.. autosummary::
   :toctree: generated/

   fit
   ecdf
   logrank

# 方向性统计函数
Directional statistical functions
---------------------------------

.. autosummary::
   :toctree: generated/

   directional_stats
   circmean
   circvar
   circstd

# 敏感性分析
Sensitivity Analysis
--------------------

.. autosummary::
   :toctree: generated/

   sobol_indices

# 绘图测试
Plot-tests
----------

.. autosummary::
   :toctree: generated/

   ppcc_max
   ppcc_plot
   probplot
   boxcox_normplot
   yeojohnson_normplot

# 单变量和多变量核密度估计
Univariate and multivariate kernel density estimation
-----------------------------------------------------

.. autosummary::
   :toctree: generated/

   gaussian_kde

# scipy.stats 中使用的警告 / 错误
Warnings / Errors used in :mod:`scipy.stats`
--------------------------------------------

.. autosummary::
   :toctree: generated/

   DegenerateDataWarning
   ConstantInputWarning
   NearConstantInputWarning
   FitError
# 导入警告和错误类
from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
                               DegenerateDataWarning, FitError)
# 导入所有 _stats_py 模块中的内容
from ._stats_py import *
# 导入 variation 函数
from ._variation import variation
# 导入 distributions 模块中的所有内容
from .distributions import *
# 导入 _morestats 模块中的所有内容
from ._morestats import *
# 导入 _multicomp 模块中的所有内容
from ._multicomp import *
# 导入 binomtest 函数
from ._binomtest import binomtest
# 导入 _binned_statistic 模块中的所有内容
from ._binned_statistic import *
# 导入 gaussian_kde 函数
from ._kde import gaussian_kde
# 导入 mstats 模块中的所有内容
from . import mstats
# 导入 qmc 模块中的所有内容
from . import qmc
# 导入 _multivariate 模块中的所有内容
from ._multivariate import *
# 导入 contingency 模块中的所有内容
from . import contingency
# 导入 chi2_contingency 函数
from .contingency import chi2_contingency
# 导入 CensoredData 类
from ._censored_data import CensoredData
# 导入 _resampling 模块中的所有内容
from ._resampling import (bootstrap, monte_carlo_test, permutation_test, power,
                          MonteCarloMethod, PermutationMethod, BootstrapMethod)
# 导入 _entropy 模块中的所有内容
from ._entropy import *
# 导入 _hypotests 模块中的所有内容
from ._hypotests import *
# 导入 page_trend_test 函数
from ._page_trend_test import page_trend_test
# 导入 mannwhitneyu 函数
from ._mannwhitneyu import mannwhitneyu
# 导入 bws_test 函数
from ._bws_test import bws_test
# 导入 fit 和 goodness_of_fit 函数
from ._fit import fit, goodness_of_fit
# 导入 Covariance 类
from ._covariance import Covariance
# 导入 _sensitivity_analysis 模块中的所有内容
from ._sensitivity_analysis import *
# 导入 _survival 模块中的所有内容
from ._survival import *
# 导入 multiscale_graphcorr 函数
from ._mgc import multiscale_graphcorr

# 声明 __all__ 变量，包含当前模块中不以下划线开头的所有名称
__all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.

# 导入 PytestTester 类并创建测试对象 test
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，避免污染命名空间
del PytestTester
```