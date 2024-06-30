# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\__init__.py`

```
# 导入特征选择算法模块

"""Feature selection algorithms.

These include univariate filter selection methods and the recursive feature elimination
algorithm.
"""

# 从 _base 模块中导入 SelectorMixin 类
from ._base import SelectorMixin
# 从 _from_model 模块中导入 SelectFromModel 类
from ._from_model import SelectFromModel
# 从 _mutual_info 模块中导入 mutual_info_classif 和 mutual_info_regression 函数
from ._mutual_info import mutual_info_classif, mutual_info_regression
# 从 _rfe 模块中导入 RFE 和 RFECV 类
from ._rfe import RFE, RFECV
# 从 _sequential 模块中导入 SequentialFeatureSelector 类
from ._sequential import SequentialFeatureSelector
# 从 _univariate_selection 模块中导入以下内容：
# GenericUnivariateSelect 类
# SelectFdr 类
# SelectFpr 类
# SelectFwe 类
# SelectKBest 类
# SelectPercentile 类
# chi2 函数
# f_classif 函数
# f_oneway 函数
# f_regression 函数
# r_regression 函数
from ._univariate_selection import (
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_oneway,
    f_regression,
    r_regression,
)
# 从 _variance_threshold 模块中导入 VarianceThreshold 类
from ._variance_threshold import VarianceThreshold

# 暴露给外部的所有符号（类、函数等）
__all__ = [
    "GenericUnivariateSelect",
    "SequentialFeatureSelector",
    "RFE",
    "RFECV",
    "SelectFdr",
    "SelectFpr",
    "SelectFwe",
    "SelectKBest",
    "SelectFromModel",
    "SelectPercentile",
    "VarianceThreshold",
    "chi2",
    "f_classif",
    "f_oneway",
    "f_regression",
    "r_regression",
    "mutual_info_classif",
    "mutual_info_regression",
    "SelectorMixin",
]
```