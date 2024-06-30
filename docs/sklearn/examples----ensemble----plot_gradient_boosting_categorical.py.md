# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_gradient_boosting_categorical.py`

```
"""
================================================
Categorical Feature Support in Gradient Boosting
================================================

.. currentmodule:: sklearn

In this example, we will compare the training times and prediction
performances of :class:`~ensemble.HistGradientBoostingRegressor` with
different encoding strategies for categorical features. In
particular, we will evaluate:

- dropping the categorical features
- using a :class:`~preprocessing.OneHotEncoder`
- using an :class:`~preprocessing.OrdinalEncoder` and treat categories as
  ordered, equidistant quantities
- using an :class:`~preprocessing.OrdinalEncoder` and rely on the :ref:`native
  category support <categorical_support_gbdt>` of the
  :class:`~ensemble.HistGradientBoostingRegressor` estimator.

We will work with the Ames Iowa Housing dataset which consists of numerical
and categorical features, where the houses' sales prices is the target.

See :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` for an
example showcasing some other features of
:class:`~ensemble.HistGradientBoostingRegressor`.

"""

# %%
# Load Ames Housing dataset
# -------------------------
# First, we load the Ames Housing data as a pandas dataframe. The features
# are either categorical or numerical:
from sklearn.datasets import fetch_openml

# Fetch the Ames Housing dataset and store it in X, y variables
X, y = fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

# Select a subset of columns from X to speed up the example
categorical_columns_subset = [
    "BldgType",
    "GarageFinish",
    "LotConfig",
    "Functional",
    "MasVnrType",
    "HouseStyle",
    "FireplaceQu",
    "ExterCond",
    "ExterQual",
    "PoolQC",
]

numerical_columns_subset = [
    "3SsnPorch",
    "Fireplaces",
    "BsmtHalfBath",
    "HalfBath",
    "GarageCars",
    "TotRmsAbvGrd",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "GrLivArea",
    "ScreenPorch",
]

# Select only the chosen columns from X
X = X[categorical_columns_subset + numerical_columns_subset]

# Convert selected categorical columns to categorical type
X[categorical_columns_subset] = X[categorical_columns_subset].astype("category")

# Identify all categorical columns in X
categorical_columns = X.select_dtypes(include="category").columns
n_categorical_features = len(categorical_columns)
n_numerical_features = X.select_dtypes(include="number").shape[1]

# Print information about the dataset
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of categorical features: {n_categorical_features}")
print(f"Number of numerical features: {n_numerical_features}")

# %%
# Gradient boosting estimator with dropped categorical features
# -------------------------------------------------------------
# As a baseline, we create an estimator where the categorical features are
# dropped:

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

# Define a transformer to drop categorical columns
dropper = make_column_transformer(
    ("drop", make_column_selector(dtype_include="category")), remainder="passthrough"
)
# 创建一个使用 HistGradientBoostingRegressor 的管道，包含了数据预处理步骤 dropper
# %%
# 使用独热编码处理分类特征的梯度提升估计器
# -------------------------------------------------
# 接下来，我们创建一个管道，将对分类特征进行独热编码处理，
# 并让其余的数值数据直接通过：

from sklearn.preprocessing import OneHotEncoder

# 创建一个列转换器，使用独热编码处理分类特征，其余特征直接通过
one_hot_encoder = make_column_transformer(
    (
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
)

# 创建包含独热编码和 HistGradientBoostingRegressor 的管道
hist_one_hot = make_pipeline(
    one_hot_encoder, HistGradientBoostingRegressor(random_state=42)
)

# %%
# 使用序数编码处理分类特征的梯度提升估计器
# -------------------------------------------------
# 接下来，我们创建一个管道，将分类特征视为有序量来处理，
# 即将分类编码为 0、1、2 等，并将其视为连续特征。

import numpy as np

from sklearn.preprocessing import OrdinalEncoder

# 创建一个列转换器，使用序数编码处理分类特征，其余特征直接通过
ordinal_encoder = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
    # 使用简短的特征名称以便在管道的下一步中更容易指定 HistGradientBoostingRegressor 中的分类变量
    verbose_feature_names_out=False,
)

# 创建包含序数编码和 HistGradientBoostingRegressor 的管道
hist_ordinal = make_pipeline(
    ordinal_encoder, HistGradientBoostingRegressor(random_state=42)
)

# %%
# 具有原生分类特征支持的梯度提升估计器
# -----------------------------------------------------------
# 现在我们创建一个 HistGradientBoostingRegressor 估计器，
# 它将原生支持分类特征。此估计器不会将分类特征视为有序量。
# 我们设置 categorical_features="from_dtype"，以便从 DataFrame 列的 dtypes 中将具有分类 dtype 的特征视为分类特征。

hist_native = HistGradientBoostingRegressor(
    random_state=42, categorical_features="from_dtype"
)

# %%
# 模型比较
# ----------------
# 最后，我们使用交叉验证评估模型。在这里，我们比较模型在
# :func:`~metrics.mean_absolute_percentage_error` 和拟合时间方面的表现。

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate

# 设置评分指标为负均绝对百分比误差
scoring = "neg_mean_absolute_percentage_error"
n_cv_folds = 3

# 使用交叉验证评估 dropper 预处理后的模型性能
dropped_result = cross_validate(hist_dropped, X, y, cv=n_cv_folds, scoring=scoring)

# 使用交叉验证评估 one_hot_encoder 预处理后的模型性能
one_hot_result = cross_validate(hist_one_hot, X, y, cv=n_cv_folds, scoring=scoring)
ordinal_result = cross_validate(hist_ordinal, X, y, cv=n_cv_folds, scoring=scoring)
native_result = cross_validate(hist_native, X, y, cv=n_cv_folds, scoring=scoring)

def plot_results(figure_title):
    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # 设置要绘制的信息列表，包括键名、标题、子图对象和y轴限制
    plot_info = [
        ("fit_time", "Fit times (s)", ax1, None),
        ("test_score", "Mean Absolute Percentage Error", ax2, None),
    ]

    # 设置x轴位置和柱状图宽度
    x, width = np.arange(4), 0.9
    
    # 遍历plot_info列表，对每个子图绘制柱状图
    for key, title, ax, y_limit in plot_info:
        # 从不同结果集中获取对应key的值，用于绘制柱状图
        items = [
            dropped_result[key],
            one_hot_result[key],
            ordinal_result[key],
            native_result[key],
        ]

        # 计算各组数据的均值（MAPE）和标准差
        mape_cv_mean = [np.mean(np.abs(item)) for item in items]
        mape_cv_std = [np.std(item) for item in items]

        # 绘制柱状图，包括均值和误差条，使用颜色区分不同的模型
        ax.bar(
            x=x,
            height=mape_cv_mean,
            width=width,
            yerr=mape_cv_std,
            color=["C0", "C1", "C2", "C3"],
        )

        # 设置子图的标题和标签
        ax.set(
            xlabel="Model",
            title=title,
            xticks=x,
            xticklabels=["Dropped", "One Hot", "Ordinal", "Native"],
            ylim=y_limit,
        )

    # 设置整体标题
    fig.suptitle(figure_title)

plot_results("Gradient Boosting on Ames Housing")

# %%
# 我们可以看到使用独热编码数据的模型明显最慢。这是可以预期的，因为
# 独热编码为每个分类特征值创建一个额外的特征（对每个分类特征而言），从而在
# 拟合过程中需要考虑更多的分裂点。理论上，我们预期本地处理分类特征的速度
# 稍慢于将分类视为有序数量（'Ordinal'），因为本地处理需要对分类进行排序。
# 当分类数目较少时，拟合时间应该是接近的，但在实际中这种情况可能并不总是
# 反映出来。
#
# 在预测性能方面，删除分类特征会导致性能较差。使用分类特征的三个模型具
# 有可比较的误差率，本地处理略有优势。
# 
# %%
# 限制分裂数量
# -----------------------------
# 通常情况下，可以预期从独热编码数据中获得较差的预测结果，特别是当树的
# 深度或节点数量受到限制时：使用独热编码数据时，需要更多的分裂点（即更
# 多的深度），才能恢复相当于本地处理一次分裂所能获得的等效分裂点。
# 
# 这一点对于将分类视为有序数量也同样适用：如果分类为'A..F'，最佳分裂点为
# 'ACF - BDE'，则独热编码模型将需要3个分裂点（左节点中的每个分类一个分
# 裂点），而非本地的有序模型将需要4个分裂点：1个分裂点隔离'A'，1个分裂
# 点隔离'F'，2个分裂点隔离'C'和'BCDE'。
# 
# 在实践中，这些模型性能差异的大小将取决于数据集和树的灵活性。
#
# 循环遍历四种数据处理管道：hist_dropped、hist_one_hot、hist_ordinal、hist_native
for pipe in (hist_dropped, hist_one_hot, hist_ordinal, hist_native):
    # 如果当前管道是 hist_native
    if pipe is hist_native:
        # 对于不使用管道的原生模型，可以直接设置参数
        pipe.set_params(max_depth=3, max_iter=15)
    else:
        # 对于其它管道，设置参数以使用 HistGradientBoostingRegressor
        pipe.set_params(
            histgradientboostingregressor__max_depth=3,
            histgradientboostingregressor__max_iter=15,
        )

# 使用交叉验证评估四种模型的性能并存储结果
dropped_result = cross_validate(hist_dropped, X, y, cv=n_cv_folds, scoring=scoring)
one_hot_result = cross_validate(hist_one_hot, X, y, cv=n_cv_folds, scoring=scoring)
ordinal_result = cross_validate(hist_ordinal, X, y, cv=n_cv_folds, scoring=scoring)
native_result = cross_validate(hist_native, X, y, cv=n_cv_folds, scoring=scoring)

# 绘制 Ames 房屋数据集上梯度提升模型的结果图表（使用少量且浅层的树）
plot_results("Gradient Boosting on Ames Housing (few and small trees)")

# 显示绘制的图表
plt.show()

# %%
# 这些欠拟合模型的结果验证了我们之前的直觉：
# 在分裂预算受限的情况下，原生类别处理策略表现最佳。
# 另外两种策略（独热编码和类别按序数值处理）导致的误差值与完全删除分类特征的基准模型相当。
```