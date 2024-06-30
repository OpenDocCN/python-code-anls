# `D:\src\scipysrc\scikit-learn\examples\compose\plot_feature_union.py`

```
"""
=================================================
Concatenating multiple feature extraction methods
=================================================

In many real-world examples, there are many ways to extract features from a
dataset. Often it is beneficial to combine several methods to obtain good
performance. This example shows how to use ``FeatureUnion`` to combine
features obtained by PCA and univariate selection.

Combining features using this transformer has the benefit that it allows
cross validation and grid searches over the whole process.

The combination used in this example is not particularly helpful on this
dataset and is only used to illustrate the usage of FeatureUnion.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库和模块
from sklearn.datasets import load_iris            # 导入鸢尾花数据集
from sklearn.decomposition import PCA             # 导入PCA算法
from sklearn.feature_selection import SelectKBest # 导入特征选择模块
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证模块
from sklearn.pipeline import FeatureUnion, Pipeline  # 导入特征联合和管道模块
from sklearn.svm import SVC                       # 导入支持向量机模型

iris = load_iris()

X, y = iris.data, iris.target

# This dataset is way too high-dimensional. Better do PCA:
# 对于这个数据集来说，维度太高了，使用PCA进行降维是比较好的选择
pca = PCA(n_components=2)

# Maybe some original features were good, too?
# 也许一些原始特征也很有效？使用SelectKBest选择最好的特征
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:
# 通过PCA和单变量选择构建估计器
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
# 使用组合特征来转换数据集
X_features = combined_features.fit(X, y).transform(X)
print("Combined space has", X_features.shape[1], "features")

svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:
# 对k、n_components和C进行网格搜索
pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(
    features__pca__n_components=[1, 2, 3],
    features__univ_select__k=[1, 2],
    svm__C=[0.1, 1, 10],
)

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
```