# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\__init__.py`

```
# 集成学习方法的模块，用于分类、回归和异常检测

"""Ensemble-based methods for classification, regression and anomaly detection."""

# 导入集成学习方法所需的类和函数
from ._bagging import BaggingClassifier, BaggingRegressor
from ._base import BaseEnsemble
from ._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from ._gb import GradientBoostingClassifier, GradientBoostingRegressor
from ._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from ._iforest import IsolationForest
from ._stacking import StackingClassifier, StackingRegressor
from ._voting import VotingClassifier, VotingRegressor
from ._weight_boosting import AdaBoostClassifier, AdaBoostRegressor

# 列出模块中公开的类和函数名
__all__ = [
    "BaseEnsemble",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomTreesEmbedding",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "IsolationForest",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
]
```