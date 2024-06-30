# `D:\src\scipysrc\scipy\scipy\stats\stats.py`

```
# 以下代码导入 `_sub_module_deprecation` 函数，用于处理模块废弃警告
# 和将来的删除计划通知。此文件不适用于公共使用，将在 SciPy v2.0.0 中删除。
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表 `__all__`，包含了当前模块中所有公开的函数名
__all__ = [  # noqa: F822
    'find_repeats', 'gmean', 'hmean', 'pmean', 'mode', 'tmean', 'tvar',
    'tmin', 'tmax', 'tstd', 'tsem', 'moment',
    'skew', 'kurtosis', 'describe', 'skewtest', 'kurtosistest',
    'normaltest', 'jarque_bera',
    'scoreatpercentile', 'percentileofscore',
    'cumfreq', 'relfreq', 'obrientransform',
    'sem', 'zmap', 'zscore', 'gzscore', 'iqr', 'gstd',
    'median_abs_deviation',
    'sigmaclip', 'trimboth', 'trim1', 'trim_mean',
    'f_oneway',
    'pearsonr', 'fisher_exact',
    'spearmanr', 'pointbiserialr',
    'kendalltau', 'weightedtau', 'multiscale_graphcorr',
    'linregress', 'siegelslopes', 'theilslopes', 'ttest_1samp',
    'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel',
    'kstest', 'ks_1samp', 'ks_2samp',
    'chisquare', 'power_divergence',
    'tiecorrect', 'ranksums', 'kruskal', 'friedmanchisquare',
    'rankdata',
    'combine_pvalues', 'wasserstein_distance', 'energy_distance',
    'brunnermunzel', 'alexandergovern', 'distributions',
    'mstats_basic',
]

# 定义 `__dir__()` 函数，使得模块导入时可以直接获取 `__all__` 中定义的所有函数名
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 函数，用于动态获取模块中的函数，并处理过时警告和废弃提示
def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="stats",
                                   private_modules=["_stats_py", "_mgc"], all=__all__,
                                   attribute=name)
```