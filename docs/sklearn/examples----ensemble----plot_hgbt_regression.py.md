# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_hgbt_regression.py`

```
"""
==============================================
Features in Histogram Gradient Boosting Trees
==============================================

:ref:`histogram_based_gradient_boosting` (HGBT) models may be one of the most
useful supervised learning models in scikit-learn. They are based on a modern
gradient boosting implementation comparable to LightGBM and XGBoost. As such,
HGBT models are more feature rich than and often outperform alternative models
like random forests, especially when the number of samples is larger than some
ten thousands (see
:ref:`sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`).

The top usability features of HGBT models are:

1. Several available loss functions for mean and quantile regression tasks, see
   :ref:`Quantile loss <quantile_support_hgbdt>`.
2. :ref:`categorical_support_gbdt`, see
   :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_categorical.py`.
3. Early stopping.
4. :ref:`nan_support_hgbt`, which avoids the need for an imputer.
5. :ref:`monotonic_cst_gbdt`.
6. :ref:`interaction_cst_hgbt`.

This example aims at showcasing all points except 2 and 6 in a real life
setting.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Preparing the data
# ==================
# The `electricity dataset <http://www.openml.org/d/151>`_ consists of data
# collected from the Australian New South Wales Electricity Market. In this
# market, prices are not fixed and are affected by supply and demand. They are
# set every five minutes. Electricity transfers to/from the neighboring state of
# Victoria were done to alleviate fluctuations.
#
# The dataset, originally named ELEC2, contains 45,312 instances dated from 7
# May 1996 to 5 December 1998. Each sample of the dataset refers to a period of
# 30 minutes, i.e. there are 48 instances for each time period of one day. Each
# sample on the dataset has 7 columns:
#
# - date: between 7 May 1996 to 5 December 1998. Normalized between 0 and 1;
# - day: day of week (1-7);
# - period: half hour intervals over 24 hours. Normalized between 0 and 1;
# - nswprice/nswdemand: electricity price/demand of New South Wales;
# - vicprice/vicdemand: electricity price/demand of Victoria.
#
# Originally, it is a classification task, but here we use it for the regression
# task to predict the scheduled electricity transfer between states.

from sklearn.datasets import fetch_openml

# Fetch the electricity dataset from openml, version 1, and store it as a pandas DataFrame
electricity = fetch_openml(
    name="electricity", version=1, as_frame=True, parser="pandas"
)
df = electricity.frame

# %%
# This particular dataset has a stepwise constant target for the first 17,760
# samples:

# Retrieve unique values of the "transfer" column for the first 17,760 samples
df["transfer"][:17_760].unique()

# %%
# Let us drop those entries and explore the hourly electricity transfer over
# different days of the week:

import matplotlib.pyplot as plt
import seaborn as sns

# Select the subset of the dataframe starting from index 17,760
df = electricity.frame.iloc[17_760:]
X = df.drop(columns=["transfer", "class"])  # Create features X by dropping "transfer" and "class" columns
y = df["transfer"]  # Target variable y is "transfer"
fig, ax = plt.subplots(figsize=(15, 10))
# 创建一个新的图形和轴对象，设置图形大小为15x10
pointplot = sns.lineplot(x=df["period"], y=df["transfer"], hue=df["day"], ax=ax)
# 使用Seaborn绘制折线图，x轴为df数据框中的"period"列，y轴为"transfer"列，根据"day"列分组绘制不同颜色的线条，并将图形绘制在之前创建的轴对象ax上
handles, lables = ax.get_legend_handles_labels()
# 获取图例的句柄和标签
ax.set(
    title="Hourly energy transfer for different days of the week",
    xlabel="Normalized time of the day",
    ylabel="Normalized energy transfer",
)
# 设置轴的标题、x轴标签和y轴标签
_ = ax.legend(handles, ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
# 使用之前获取的图例句柄和标签，设置图例，显示星期几的标签

# %%
# Notice that energy transfer increases systematically during weekends.
#
# Effect of number of trees and early stopping
# ============================================
# For the sake of illustrating the effect of the (maximum) number of trees, we
# train a :class:`~sklearn.ensemble.HistGradientBoostingRegressor` over the
# daily electricity transfer using the whole dataset. Then we visualize its
# predictions depending on the `max_iter` parameter. Here we don't try to
# evaluate the performance of the model and its capacity to generalize but
# rather its capability to learn from the training data.

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
# 使用train_test_split函数将特征X和目标y按照0.6:0.4的比例划分为训练集和测试集，设置不打乱顺序

print(f"Training sample size: {X_train.shape[0]}")
# 打印训练集样本数量
print(f"Test sample size: {X_test.shape[0]}")
# 打印测试集样本数量
print(f"Number of features: {X_train.shape[1]}")
# 打印特征数量

# %%
max_iter_list = [5, 50]
# 定义max_iter_list列表，包含两个值：5和50
average_week_demand = (
    df.loc[X_test.index].groupby(["day", "period"], observed=False)["transfer"].mean()
)
# 计算测试集上的平均每周需求，根据"day"和"period"分组，观察不包括缺失值
colors = sns.color_palette("colorblind")
# 使用Seaborn调色板生成颜色列表
fig, ax = plt.subplots(figsize=(10, 5))
# 创建新的图形和轴对象，设置图形大小为10x5
average_week_demand.plot(color=colors[0], label="recorded average", linewidth=2, ax=ax)
# 绘制平均每周需求的折线图，使用第一个颜色，并添加标签"recorded average"，线宽为2，绘制在之前创建的轴对象ax上

for idx, max_iter in enumerate(max_iter_list):
    hgbt = HistGradientBoostingRegressor(
        max_iter=max_iter, categorical_features=None, random_state=42
    )
    # 初始化HistGradientBoostingRegressor模型，设置最大迭代次数max_iter，不处理分类特征，设置随机种子为42
    hgbt.fit(X_train, y_train)
    # 使用训练集训练模型

    y_pred = hgbt.predict(X_test)
    # 使用测试集进行预测
    prediction_df = df.loc[X_test.index].copy()
    prediction_df["y_pred"] = y_pred
    # 将预测结果添加到数据框副本中
    average_pred = prediction_df.groupby(["day", "period"], observed=False)[
        "y_pred"
    ].mean()
    # 计算预测的平均值，根据"day"和"period"分组，观察不包括缺失值
    average_pred.plot(
        color=colors[idx + 1], label=f"max_iter={max_iter}", linewidth=2, ax=ax
    )
    # 绘制预测的平均值折线图，使用不同的颜色，添加标签，线宽为2，绘制在之前创建的轴对象ax上

ax.set(
    title="Predicted average energy transfer during the week",
    xticks=[(i + 0.2) * 48 for i in range(7)],
    xticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    xlabel="Time of the week",
    ylabel="Normalized energy transfer",
)
# 设置轴的标题、x轴刻度位置和标签、y轴标签
_ = ax.legend()

# %%
# With just a few iterations, HGBT models can achieve convergence (see
# :ref:`sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`),
# meaning that adding more trees does not improve the model anymore. In the
# figure above, 5 iterations are not enough to get good predictions. With 50
# iterations, we are already able to do a good job.
#
# Setting `max_iter` too high might degrade the prediction quality and cost a lot of
# 定义通用参数字典，包括最大迭代次数、学习率、内部验证集的比例、随机种子、分类特征、评分指标
common_params = {
    "max_iter": 1_000,  # 最大迭代次数
    "learning_rate": 0.3,  # 学习率
    "validation_fraction": 0.2,  # 内部验证集的比例
    "random_state": 42,  # 随机种子
    "categorical_features": None,  # 分类特征
    "scoring": "neg_root_mean_squared_error",  # 评分指标
}

# 创建 HistGradientBoostingRegressor 模型，使用早停策略，并传入通用参数
hgbt = HistGradientBoostingRegressor(early_stopping=True, **common_params)
# 使用训练集 X_train 和 y_train 来拟合模型
hgbt.fit(X_train, y_train)

# 创建图形对象并绘制早停策略下的验证误差曲线
_, ax = plt.subplots()
plt.plot(-hgbt.validation_score_)  # 绘制负的验证分数
_ = ax.set(
    xlabel="number of iterations",  # x轴标签为迭代次数
    ylabel="root mean squared error",  # y轴标签为均方根误差
    title=f"Loss of hgbt with early stopping (n_iter={hgbt.n_iter_})",  # 图表标题包含迭代次数信息
)

# %%
# 然后我们可以将 `max_iter` 的值覆盖为一个合理的值，避免内部验证带来的额外计算成本。
# 将迭代次数的值四舍五入以考虑训练集变异性：

import math

# 将 `max_iter` 设置为 `hgbt.n_iter_` 的值除以100后向上取整再乘以100
common_params["max_iter"] = math.ceil(hgbt.n_iter_ / 100) * 100
# 关闭早停策略
common_params["early_stopping"] = False
# 使用更新后的通用参数创建 HistGradientBoostingRegressor 模型
hgbt = HistGradientBoostingRegressor(**common_params)

# %%
# .. note:: 早停策略中进行的内部验证不适用于时间序列数据。
#
# 缺失值支持
# ==========================
# HGBT 模型原生支持缺失值。在训练过程中，决策树生长器根据潜在增益决定带有缺失值的样本应该进入左子节点还是右子节点。
# 在预测时，这些样本根据训练时的决策送入相应的子节点。如果某个特征在训练时没有缺失值，则在预测时，具有缺失值的样本会送入具有最多样本的子节点（如在拟合期间所见）。
#
# 此示例展示了 HGBT 回归模型如何处理完全随机缺失值（MCAR），即缺失不依赖于观察到的数据或未观察到的数据。
# 我们可以通过随机选择的特征随机替换值来模拟这种情况，将其设为 `nan` 值。

import numpy as np
# 导入根均方误差函数
from sklearn.metrics import root_mean_squared_error

# 使用随机种子创建随机数生成器对象
rng = np.random.RandomState(42)
# 定义测试集中第一周的切片，包含336个时间步，相当于7天*48小时
first_week = slice(0, 336)
# 不同的缺失比例列表
missing_fraction_list = [0, 0.01, 0.03]

# 定义生成缺失值的函数，根据给定的缺失比例随机将数据集中的值设为NaN
def generate_missing_values(X, missing_fraction):
    # 计算数据集中总的单元格数量
    total_cells = X.shape[0] * X.shape[1]
    # 计算需要置为NaN的单元格数量
    num_missing_cells = int(total_cells * missing_fraction)
    # 随机选择行和列的索引
    row_indices = rng.choice(X.shape[0], num_missing_cells, replace=True)
    col_indices = rng.choice(X.shape[1], num_missing_cells, replace=True)
    # 复制数据集并将选定的位置置为NaN，生成新的数据集
    X_missing = X.copy()
    X_missing.iloc[row_indices, col_indices] = np.nan
    return X_missing

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(12, 6))
# 绘制测试集中第一周的实际传输数据
ax.plot(y_test.values[first_week], label="Actual transfer")

# 对于每个缺失比例，生成训练和测试集的缺失数据，训练模型并进行预测
for missing_fraction in missing_fraction_list:
    X_train_missing = generate_missing_values(X_train, missing_fraction)
    X_test_missing = generate_missing_values(X_test, missing_fraction)
    hgbt.fit(X_train_missing, y_train)  # 使用梯度提升回归树模型拟合数据
    y_pred = hgbt.predict(X_test_missing[first_week])  # 对测试集的第一周进行预测
    rmse = root_mean_squared_error(y_test[first_week], y_pred)  # 计算预测结果的根均方误差
    ax.plot(
        y_pred[first_week],
        label=f"missing_fraction={missing_fraction}, RMSE={rmse:.3f}",
        alpha=0.5,
    )

# 设置图形标题、横坐标和纵坐标标签等属性
ax.set(
    title="Daily energy transfer predictions on data with MCAR values",
    xticks=[(i + 0.2) * 48 for i in range(7)],
    xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    xlabel="Time of the week",
    ylabel="Normalized energy transfer",
)
_ = ax.legend(loc="lower right")

# %%
# 预期情况下，随着缺失值比例的增加，模型性能将下降。
#
# 支持分位数损失
# =========================
#
# 在回归中使用分位数损失函数可以反映目标变量的变异性或不确定性。
# 例如，预测第5和第95百分位数可以提供一个90%的预测区间，
# 即我们预计新观测值有90%的概率落在其中的范围内。

# 导入分位数损失函数
from sklearn.metrics import mean_pinball_loss

# 指定要预测的分位数
quantiles = [0.95, 0.05]
predictions = []

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(12, 6))
# 绘制测试集中第一周的实际传输数据
ax.plot(y_test.values[first_week], label="Actual transfer")

# 对于每个分位数，使用梯度提升回归树模型拟合数据并进行预测
for quantile in quantiles:
    hgbt_quantile = HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, **common_params
    )
    hgbt_quantile.fit(X_train, y_train)
    y_pred = hgbt_quantile.predict(X_test[first_week])

    predictions.append(y_pred)
    score = mean_pinball_loss(y_test[first_week], y_pred)  # 计算分位数损失
    ax.plot(
        y_pred[first_week],
        label=f"quantile={quantile}, pinball loss={score:.2f}",
        alpha=0.5,
    )

# 使用不同分位数的预测结果填充预测区间
ax.fill_between(
    range(len(predictions[0][first_week])),
    predictions[0][first_week],
    predictions[1][first_week],
    color=colors[0],
    alpha=0.1,
)
# 设置图形标题、横坐标和纵坐标标签等属性
ax.set(
    title="Daily energy transfer predictions with quantile loss",
    xticks=[(i + 0.2) * 48 for i in range(7)],
    xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    xlabel="Time of the week",
)
    # 设置变量 `ylabel`，用于存储字符串 "Normalized energy transfer"
    ylabel="Normalized energy transfer",
# %%
# We observe a tendence to over-estimate the energy transfer. This could be be
# quantitatively confirmed by computing empirical coverage numbers as done in
# the :ref:`calibration of confidence intervals section <calibration-section>`.
# Keep in mind that those predicted percentiles are just estimations from a
# model. One can still improve the quality of such estimations by:
#
# - collecting more data-points;
# - better tuning of the model hyperparameters, see
#   :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_quantile.py`;
# - engineering more predictive features from the same data, see
#   :ref:`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`.
#
# Monotonic constraints
# =====================
#
# Given specific domain knowledge that requires the relationship between a
# feature and the target to be monotonically increasing or decreasing, one can
# enforce such behaviour in the predictions of a HGBT model using monotonic
# constraints. This makes the model more interpretable and can reduce its
# variance (and potentially mitigate overfitting) at the risk of increasing
# bias. Monotonic constraints can also be used to enforce specific regulatory
# requirements, ensure compliance and align with ethical considerations.
#
# In the present example, the policy of transferring energy from Victoria to New
# South Wales is meant to alleviate price fluctuations, meaning that the model
# predictions have to enforce such goal, i.e. transfer should increase with
# price and demand in New South Wales, but also decrease with price and demand
# in Victoria, in order to benefit both populations.
#
# If the training data has feature names, it’s possible to specify the monotonic
# constraints by passing a dictionary with the convention:
#
# - 1: monotonic increase
# - 0: no constraint
# - -1: monotonic decrease
#
# Alternatively, one can pass an array-like object encoding the above convention by
# position.

# Importing necessary modules
from sklearn.inspection import PartialDependenceDisplay

# Define monotonic constraints for features in the model
monotonic_cst = {
    "date": 0,        # No monotonic constraint for 'date'
    "day": 0,         # No monotonic constraint for 'day'
    "period": 0,      # No monotonic constraint for 'period'
    "nswdemand": 1,   # Monotonic increase constraint for 'nswdemand'
    "nswprice": 1,    # Monotonic increase constraint for 'nswprice'
    "vicdemand": -1,  # Monotonic decrease constraint for 'vicdemand'
    "vicprice": -1,   # Monotonic decrease constraint for 'vicprice'
}

# Initialize HistGradientBoostingRegressor without constraints
hgbt_no_cst = HistGradientBoostingRegressor(
    categorical_features=None, random_state=42
).fit(X, y)

# Initialize HistGradientBoostingRegressor with monotonic constraints
hgbt_cst = HistGradientBoostingRegressor(
    monotonic_cst=monotonic_cst, categorical_features=None, random_state=42
).fit(X, y)

# Create subplots for displaying partial dependence plots
fig, ax = plt.subplots(nrows=2, figsize=(15, 10))

# Plot partial dependence without constraints
disp = PartialDependenceDisplay.from_estimator(
    hgbt_no_cst,
    X,
    features=["nswdemand", "nswprice"],
    line_kw={"linewidth": 2, "label": "unconstrained", "color": "tab:blue"},
    ax=ax[0],
)

# Plot partial dependence with constraints
PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["nswdemand", "nswprice"],
    line_kw={"linewidth": 2, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)

# Display partial dependence plot for another set of features without constraints
disp = PartialDependenceDisplay.from_estimator(
    hgbt_no_cst,
    X,
    # 定义一个包含特征名称的列表，这里是"vicdemand"和"vicprice"
    features=["vicdemand", "vicprice"],
    # 定义一个包含线条属性的字典，包括线宽、标签和颜色
    line_kw={"linewidth": 2, "label": "unconstrained", "color": "tab:blue"},
    # 将当前的图表区域指定为第二个子图(ax[1])
    ax=ax[1],
)
PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["vicdemand", "vicprice"],
    line_kw={"linewidth": 2, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)
_ = plt.legend()

# %%
# 观察到 `nswdemand` 和 `vicdemand` 在无约束条件下似乎已经是单调的。
# 这是一个很好的例子，展示了具有单调性约束的模型是如何“过度约束”的。
#
# 另外，我们可以验证引入单调性约束后，模型的预测质量是否显著下降。
# 为此，我们使用 :class:`~sklearn.model_selection.TimeSeriesSplit`
# 交叉验证来估计测试分数的方差。通过这样做，我们确保训练数据不会超过测试数据，
# 这在处理具有时间关系的数据时至关重要。

from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate

ts_cv = TimeSeriesSplit(n_splits=5, gap=48, test_size=336)  # 一周有 336 个样本
scorer = make_scorer(root_mean_squared_error)

cv_results = cross_validate(hgbt_no_cst, X, y, cv=ts_cv, scoring=scorer)
rmse = cv_results["test_score"]
print(f"RMSE without constraints = {rmse.mean():.3f} +/- {rmse.std():.3f}")

cv_results = cross_validate(hgbt_cst, X, y, cv=ts_cv, scoring=scorer)
rmse = cv_results["test_score"]
print(f"RMSE with constraints    = {rmse.mean():.3f} +/- {rmse.std():.3f}")

# %%
# 也就是说，注意到比较是在两个不同的模型之间进行的，这两个模型可能通过不同的超参数组合进行优化。
# 这就是为什么在本节中我们不像之前那样使用 `common_params` 的原因。
```