# `D:\src\scipysrc\scikit-learn\examples\compose\plot_column_transformer_mixed_types.py`

```
"""
===================================
Column Transformer with Mixed Types
===================================

.. currentmodule:: sklearn

This example illustrates how to apply different preprocessing and feature
extraction pipelines to different subsets of features, using
:class:`~compose.ColumnTransformer`. This is particularly handy for the
case of datasets that contain heterogeneous data types, since we may want to
scale the numeric features and one-hot encode the categorical ones.

In this example, the numeric data is standard-scaled after mean-imputation. The
categorical data is one-hot encoded via ``OneHotEncoder``, which
creates a new category for missing values. We further reduce the dimensionality
by selecting categories using a chi-squared test.

In addition, we show two different ways to dispatch the columns to the
particular pre-processor: by column names and by column data types.

Finally, the preprocessing pipeline is integrated in a full prediction pipeline
using :class:`~pipeline.Pipeline`, together with a simple classification
model.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
import numpy as np

from sklearn.compose import ColumnTransformer  # 导入ColumnTransformer类，用于处理不同类型特征的预处理管道
from sklearn.datasets import fetch_openml  # 导入fetch_openml函数，用于获取数据集
from sklearn.feature_selection import SelectPercentile, chi2  # 导入特征选择相关函数
from sklearn.impute import SimpleImputer  # 导入SimpleImputer类，用于填补缺失值
from sklearn.linear_model import LogisticRegression  # 导入LogisticRegression类，用于分类模型
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # 导入模型选择和训练集划分相关函数
from sklearn.pipeline import Pipeline  # 导入Pipeline类，用于创建预测管道
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入数据预处理相关函数

np.random.seed(0)

# %%
# Load data from https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# Alternatively X and y can be obtained directly from the frame attribute:
# X = titanic.frame.drop('survived', axis=1)
# y = titanic.frame['survived']

# %%
# Use ``ColumnTransformer`` by selecting column by names
#
# We will train our classifier with the following features:
#
# Numeric Features:
#
# * ``age``: float;
# * ``fare``: float.
#
# Categorical Features:
#
# * ``embarked``: categories encoded as strings ``{'C', 'S', 'Q'}``;
# * ``sex``: categories encoded as strings ``{'female', 'male'}``;
# * ``pclass``: ordinal integers ``{1, 2, 3}``.
#
# We create the preprocessing pipelines for both numeric and categorical data.
# Note that ``pclass`` could either be treated as a categorical or numeric
# feature.

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),  # 使用OneHotEncoder进行独热编码
        ("selector", SelectPercentile(chi2, percentile=50)),  # 使用SelectPercentile进行特征选择
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),  # 对数值特征应用numeric_transformer预处理管道
        ("cat", categorical_transformer, categorical_features),  # 对分类特征应用categorical_transformer预处理管道
    ]
)
    transformers=[
        # 定义数据转换器列表，包括数值特征和分类特征的转换器
        ("num", numeric_transformer, numeric_features),
        # 数值转换器：用 numeric_transformer 转换 numeric_features 中的特征
        ("cat", categorical_transformer, categorical_features),
        # 分类转换器：用 categorical_transformer 转换 categorical_features 中的特征
    ]
# %%
# 将分类器添加到预处理管道中。
# 现在我们有一个完整的预测管道。
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 在训练数据上拟合管道
clf.fit(X_train, y_train)

# 打印模型评分
print("model score: %.3f" % clf.score(X_test, y_test))

# %%
# ``Pipeline`` 的 HTML 表示（显示图表）
#
# 当在 Jupyter 笔记本中打印 ``Pipeline`` 时，会显示估计器的 HTML 表示：
clf

# %%
# 使用 ``ColumnTransformer`` 按数据类型选择列
#
# 处理清理后的数据集时，可以根据列的数据类型自动进行预处理，
# 使用列的数据类型决定是将列视为数值特征还是分类特征。
# :func:`sklearn.compose.make_column_selector` 提供了这种可能性。
# 首先，让我们只选择一部分列以简化我们的示例。

subset_feature = ["embarked", "sex", "pclass", "age", "fare"]
X_train, X_test = X_train[subset_feature], X_test[subset_feature]

# %%
# 然后，我们查看每列数据类型的信息。

X_train.info()

# %%
# 我们可以看到，当使用 ``fetch_openml`` 加载数据时，`embarked` 和 `sex` 列被标记为 `category` 列。
# 因此，我们可以利用这些信息将分类列分配给 ``categorical_transformer``，将剩余列分配给 ``numerical_transformer``。

# %%
# .. 注意：实际应用中，您需要处理列的数据类型。
#    如果希望某些列被视为 `category`，则需要将它们转换为分类列。
#    如果使用 pandas，可以参考它们关于 `Categorical data` 的文档：
#    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>.

from sklearn.compose import make_column_selector as selector

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
clf

# %%
# 由于基于数据类型的选择器将 ``pclass`` 列视为数值特征，而不是之前的分类特征，
# 因此得到的分数与之前的管道不完全相同：

selector(dtype_exclude="category")(X_train)

# %%

selector(dtype_include="category")(X_train)

# %%
# 在网格搜索中使用预测管道
#
# 可以在 ``ColumnTransformer`` 对象中定义的不同预处理步骤以及分类器的超参数上执行网格搜索。
# 我们将搜索数值预处理的填充策略
# 定义一个参数网格，用于指定预处理器和分类器的超参数范围
param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],  # 数值特征预处理器中填充缺失值的策略
    "preprocessor__cat__selector__percentile": [10, 30, 50, 70],  # 类别特征预处理器中特征选择的百分位数
    "classifier__C": [0.1, 1.0, 10, 100],  # 逻辑回归分类器的正则化参数C
}

# 创建一个随机搜索交叉验证对象，用于找到最佳超参数组合
search_cv = RandomizedSearchCV(clf, param_grid, n_iter=10, random_state=0)
search_cv

# %%
# 调用 'fit' 方法触发交叉验证搜索，寻找最佳超参数组合：
#
search_cv.fit(X_train, y_train)

# 输出最佳参数组合
print("Best params:")
print(search_cv.best_params_)

# %%
# 输出使用最佳参数组合得到的内部交叉验证分数：
print(f"Internal CV score: {search_cv.best_score_:.3f}")

# %%
# 我们也可以将顶部的网格搜索结果作为 pandas 数据帧进行内省：
import pandas as pd

cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[
    [
        "mean_test_score",  # 平均测试分数
        "std_test_score",   # 测试分数的标准差
        "param_preprocessor__num__imputer__strategy",  # 数值特征预处理器的缺失值填充策略
        "param_preprocessor__cat__selector__percentile",  # 类别特征预处理器的特征选择百分位数
        "param_classifier__C",  # 逻辑回归分类器的正则化参数C
    ]
].head(5)

# %%
# 使用最佳超参数在完整的训练集上重新拟合最终模型。
# 我们可以在未用于超参数调整的保留测试数据上评估该最终模型。
#
print(
    "accuracy of the best model from randomized search: "
    f"{search_cv.score(X_test, y_test):.3f}"
)
```