# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_caching_nearest_neighbors.py`

```
"""
=========================
Caching nearest neighbors
=========================

This examples demonstrates how to precompute the k nearest neighbors before
using them in KNeighborsClassifier. KNeighborsClassifier can compute the
nearest neighbors internally, but precomputing them can have several benefits,
such as finer parameter control, caching for multiple use, or custom
implementations.

Here we use the caching property of pipelines to cache the nearest neighbors
graph between multiple fits of KNeighborsClassifier. The first call is slow
since it computes the neighbors graph, while subsequent call are faster as they
do not need to recompute the graph. Here the durations are small since the
dataset is small, but the gain can be more substantial when the dataset grows
larger, or when the grid of parameter to search is large.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入临时目录工具类
from tempfile import TemporaryDirectory

# 导入绘图工具
import matplotlib.pyplot as plt

# 导入数据集加载工具
from sklearn.datasets import load_digits

# 导入网格搜索和K最近邻分类器
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer

# 导入管道工具
from sklearn.pipeline import Pipeline

# 加载手写数字数据集
X, y = load_digits(return_X_y=True)

# 定义邻居数量列表
n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 创建K最近邻图模型，使用网格搜索中可能需要的最大邻居数
graph_model = KNeighborsTransformer(n_neighbors=max(n_neighbors_list), mode="distance")

# 创建K最近邻分类器模型，使用预先计算的距离作为度量
classifier_model = KNeighborsClassifier(metric="precomputed")

# 使用临时目录作为缓存，创建完整的管道模型
with TemporaryDirectory(prefix="sklearn_graph_cache_") as tmpdir:
    full_model = Pipeline(
        steps=[("graph", graph_model), ("classifier", classifier_model)], memory=tmpdir
    )

    # 定义参数网格，用于网格搜索
    param_grid = {"classifier__n_neighbors": n_neighbors_list}

    # 创建网格搜索模型
    grid_model = GridSearchCV(full_model, param_grid)
    
    # 在数据集上拟合网格搜索模型
    grid_model.fit(X, y)

# 绘制网格搜索结果图表
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].errorbar(
    x=n_neighbors_list,
    y=grid_model.cv_results_["mean_test_score"],
    yerr=grid_model.cv_results_["std_test_score"],
)
axes[0].set(xlabel="n_neighbors", title="Classification accuracy")
axes[1].errorbar(
    x=n_neighbors_list,
    y=grid_model.cv_results_["mean_fit_time"],
    yerr=grid_model.cv_results_["std_fit_time"],
    color="r",
)
axes[1].set(xlabel="n_neighbors", title="Fit time (with caching)")
fig.tight_layout()
plt.show()
```