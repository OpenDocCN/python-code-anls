# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_feature_transformation.py`

```
"""
===============================================
Feature transformations with ensembles of trees
===============================================

Transform your features into a higher dimensional, sparse space. Then train a
linear model on these features.

First fit an ensemble of trees (totally random trees, a random forest, or
gradient boosted trees) on the training set. Then each leaf of each tree in the
ensemble is assigned a fixed arbitrary feature index in a new feature space.
These leaf indices are then encoded in a one-hot fashion.

Each sample goes through the decisions of each tree of the ensemble and ends up
in one leaf per tree. The sample is encoded by setting feature values for these
leaves to 1 and the other feature values to 0.

The resulting transformer has then learned a supervised, sparse,
high-dimensional categorical embedding of the data.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# First, we will create a large dataset and split it into three sets:
#
# - a set to train the ensemble methods which are later used to as a feature
#   engineering transformer;
# - a set to train the linear model;
# - a set to test the linear model.
#
# It is important to split the data in such way to avoid overfitting by leaking
# data.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=80_000, random_state=10)

X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=10
)
X_train_ensemble, X_train_linear, y_train_ensemble, y_train_linear = train_test_split(
    X_full_train, y_full_train, test_size=0.5, random_state=10
)

# %%
# For each of the ensemble methods, we will use 10 estimators and a maximum
# depth of 3 levels.

n_estimators = 10
max_depth = 3

# %%
# First, we will start by training the random forest and gradient boosting on
# the separated training set

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=10
)
random_forest.fit(X_train_ensemble, y_train_ensemble)

gradient_boosting = GradientBoostingClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=10
)
_ = gradient_boosting.fit(X_train_ensemble, y_train_ensemble)

# %%
# Notice that :class:`~sklearn.ensemble.HistGradientBoostingClassifier` is much
# faster than :class:`~sklearn.ensemble.GradientBoostingClassifier` starting
# with intermediate datasets (`n_samples >= 10_000`), which is not the case of
# the present example.
#
# The :class:`~sklearn.ensemble.RandomTreesEmbedding` is an unsupervised method
# and thus does not required to be trained independently.

from sklearn.ensemble import RandomTreesEmbedding

random_tree_embedding = RandomTreesEmbedding(
    n_estimators=n_estimators, max_depth=max_depth, random_state=0
)

# %%
# 创建三个流水线，使用上述嵌入作为预处理阶段。
# 随机树嵌入可以直接与逻辑回归进行流水线处理，因为它是标准的scikit-learn转换器。
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 创建包含随机树嵌入和逻辑回归的流水线模型
rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
# 使用训练集数据拟合流水线模型
rt_model.fit(X_train_linear, y_train_linear)

# %%
# 然后，我们可以将随机森林或梯度提升与逻辑回归进行流水线处理。然而，特征转换将通过调用 `apply` 方法进行。
# scikit-learn中的流水线期望调用 `transform`。因此，我们将 `apply` 方法的调用包装在 `FunctionTransformer` 中。

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

def rf_apply(X, model):
    return model.apply(X)

# 创建使用随机森林 `apply` 方法的 `FunctionTransformer`
rf_leaves_yielder = FunctionTransformer(rf_apply, kw_args={"model": random_forest})

# 创建包含随机森林 `apply` 方法、OneHotEncoder 和逻辑回归的流水线模型
rf_model = make_pipeline(
    rf_leaves_yielder,
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(max_iter=1000),
)
# 使用训练集数据拟合流水线模型
rf_model.fit(X_train_linear, y_train_linear)

# %%
def gbdt_apply(X, model):
    return model.apply(X)[:, :, 0]

# 创建使用梯度提升 `apply` 方法的 `FunctionTransformer`
gbdt_leaves_yielder = FunctionTransformer(
    gbdt_apply, kw_args={"model": gradient_boosting}
)

# 创建包含梯度提升 `apply` 方法、OneHotEncoder 和逻辑回归的流水线模型
gbdt_model = make_pipeline(
    gbdt_leaves_yielder,
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(max_iter=1000),
)
# 使用训练集数据拟合流水线模型
gbdt_model.fit(X_train_linear, y_train_linear)

# %%
# 最后，我们可以展示所有模型的不同ROC曲线。

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

_, ax = plt.subplots()

# 定义模型列表，包含模型名称和对应的流水线模型
models = [
    ("RT embedding -> LR", rt_model),
    ("RF", random_forest),
    ("RF embedding -> LR", rf_model),
    ("GBDT", gradient_boosting),
    ("GBDT embedding -> LR", gbdt_model),
]

# 创建模型展示对象的字典
model_displays = {}
for name, pipeline in models:
    # 使用流水线模型创建ROC曲线展示对象，并添加到字典中
    model_displays[name] = RocCurveDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, name=name
    )

# 设置图表标题
_ = ax.set_title("ROC curve")

# %%
# 创建另一个图表用于展示缩放后左上角放大的ROC曲线。

_, ax = plt.subplots()
for name, pipeline in models:
    # 绘制缩放后的ROC曲线
    model_displays[name].plot(ax=ax)

# 设置x轴和y轴的显示范围
ax.set_xlim(0, 0.2)
ax.set_ylim(0.8, 1)
# 设置图表标题
_ = ax.set_title("ROC curve (zoomed in at top left)")
```