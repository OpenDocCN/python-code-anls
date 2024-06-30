# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\__init__.py`

```
# k-nearest neighbors 算法相关模块的导入
"""The k-nearest neighbors algorithms."""

# 从 _ball_tree 模块导入 BallTree 类
from ._ball_tree import BallTree
# 从 _base 模块导入 VALID_METRICS, VALID_METRICS_SPARSE 和 sort_graph_by_row_values 函数
from ._base import VALID_METRICS, VALID_METRICS_SPARSE, sort_graph_by_row_values
# 从 _classification 模块导入 KNeighborsClassifier 和 RadiusNeighborsClassifier 类
from ._classification import KNeighborsClassifier, RadiusNeighborsClassifier
# 从 _graph 模块导入 KNeighborsTransformer, RadiusNeighborsTransformer, kneighbors_graph, radius_neighbors_graph 函数
from ._graph import (
    KNeighborsTransformer,
    RadiusNeighborsTransformer,
    kneighbors_graph,
    radius_neighbors_graph,
)
# 从 _kd_tree 模块导入 KDTree 类
from ._kd_tree import KDTree
# 从 _kde 模块导入 KernelDensity 类
from ._kde import KernelDensity
# 从 _lof 模块导入 LocalOutlierFactor 类
from ._lof import LocalOutlierFactor
# 从 _nca 模块导入 NeighborhoodComponentsAnalysis 类
from ._nca import NeighborhoodComponentsAnalysis
# 从 _nearest_centroid 模块导入 NearestCentroid 类
from ._nearest_centroid import NearestCentroid
# 从 _regression 模块导入 KNeighborsRegressor 和 RadiusNeighborsRegressor 类
from ._regression import KNeighborsRegressor, RadiusNeighborsRegressor
# 从 _unsupervised 模块导入 NearestNeighbors 类
from ._unsupervised import NearestNeighbors

# __all__ 列表定义了该模块中公开的符号（变量、类、函数等）
__all__ = [
    "BallTree",
    "KDTree",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KNeighborsTransformer",
    "NearestCentroid",
    "NearestNeighbors",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
    "RadiusNeighborsTransformer",
    "kneighbors_graph",
    "radius_neighbors_graph",
    "KernelDensity",
    "LocalOutlierFactor",
    "NeighborhoodComponentsAnalysis",
    "sort_graph_by_row_values",
    "VALID_METRICS",
    "VALID_METRICS_SPARSE",
]
```