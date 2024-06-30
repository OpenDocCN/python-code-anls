# `D:\src\scipysrc\scikit-learn\examples\applications\plot_time_series_lagged_features.py`

```
"""
===========================================
Lagged features for time series forecasting
===========================================

This example demonstrates how Polars-engineered lagged features can be used
for time series forecasting with
:class:`~sklearn.ensemble.HistGradientBoostingRegressor` on the Bike Sharing
Demand dataset.

See the example on
:ref:`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`
for some data exploration on this dataset and a demo on periodic feature
engineering.

"""

# %%
# Analyzing the Bike Sharing Demand dataset
# -----------------------------------------
#
# We start by loading the data from the OpenML repository
# as a pandas dataframe. This will be replaced with Polars
# once `fetch_openml` adds a native support for it.
# We convert to Polars for feature engineering, as it automatically caches
# common subexpressions which are reused in multiple expressions
# (like `pl.col("count").shift(1)` below). See
# https://docs.pola.rs/user-guide/lazy/optimizations/ for more information.

# Importing necessary libraries
import numpy as np  # Importing numpy library for numerical operations
import polars as pl  # Importing Polars library for data manipulation

from sklearn.datasets import fetch_openml  # Importing function to fetch dataset from OpenML

pl.Config.set_fmt_str_lengths(20)  # Setting display format string lengths for Polars

# Fetching Bike Sharing Demand dataset from OpenML and converting it to Polars DataFrame
bike_sharing = fetch_openml(
    "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
)
df = bike_sharing.frame  # Converting fetched data to Pandas DataFrame
df = pl.DataFrame({col: df[col].to_numpy() for col in df.columns})  # Converting to Polars DataFrame

# %%
# Next, we take a look at the statistical summary of the dataset
# so that we can better understand the data that we are working with.
import polars.selectors as cs  # Importing selectors module from Polars for column selection

summary = df.select(cs.numeric()).describe()  # Generating statistical summary of numeric columns in the DataFrame
summary

# %%
# Let us look at the count of the seasons `"fall"`, `"spring"`, `"summer"`
# and `"winter"` present in the dataset to confirm they are balanced.

import matplotlib.pyplot as plt  # Importing matplotlib for plotting

df["season"].value_counts()  # Counting occurrences of each season category in the dataset

# %%
# Generating Polars-engineered lagged features
# --------------------------------------------
# Let's consider the problem of predicting the demand at the
# next hour given past demands. Since the demand is a continuous
# variable, one could intuitively use any regression model. However, we do
# not have the usual `(X_train, y_train)` dataset. Instead, we just have
# the `y_train` demand data sequentially organized by time.

# Creating lagged features using Polars DataFrame operations
lagged_df = df.select(
    "count",  # Selecting the target column "count"
    *[pl.col("count").shift(i).alias(f"lagged_count_{i}h") for i in [1, 2, 3]],  # Creating lagged count features
    lagged_count_1d=pl.col("count").shift(24),  # Lagged count feature for 1 day ago
    lagged_count_1d_1h=pl.col("count").shift(24 + 1),  # Lagged count feature for 1 day and 1 hour ago
    lagged_count_7d=pl.col("count").shift(7 * 24),  # Lagged count feature for 7 days ago
    lagged_count_7d_1h=pl.col("count").shift(7 * 24 + 1),  # Lagged count feature for 7 days and 1 hour ago
    lagged_mean_24h=pl.col("count").shift(1).rolling_mean(24),  # Rolling mean of count for the last 24 hours
    lagged_max_24h=pl.col("count").shift(1).rolling_max(24),  # Rolling max of count for the last 24 hours
    lagged_min_24h=pl.col("count").shift(1).rolling_min(24),  # Rolling min of count for the last 24 hours
    lagged_mean_7d=pl.col("count").shift(1).rolling_mean(7 * 24),  # Rolling mean of count for the last 7 days
    lagged_max_7d=pl.col("count").shift(1).rolling_max(7 * 24),  # Rolling max of count for the last 7 days
    lagged_min_7d=pl.col("count").shift(1).rolling_min(7 * 24),  # Rolling min of count for the last 7 days
)
lagged_df.tail(10)  # Displaying the last 10 rows of the DataFrame

# %%
# Watch out however, the first lines have undefined values because their own
# past is unknown. This depends on how much lag we used:
lagged_df.head(10)



# %%
# We can now separate the lagged features in a matrix `X` and the target variable
# (the counts to predict) in an array of the same first dimension `y`.
lagged_df = lagged_df.drop_nulls()
X = lagged_df.drop("count")
y = lagged_df["count"]
print("X shape: {}\ny shape: {}".format(X.shape, y.shape))



# %%
# Naive evaluation of the next hour bike demand regression
# --------------------------------------------------------
# Let's randomly split our tabularized dataset to train a gradient
# boosting regression tree (GBRT) model and evaluate it using Mean
# Absolute Percentage Error (MAPE). If our model is aimed at forecasting
# (i.e., predicting future data from past data), we should not use training
# data that are ulterior to the testing data. In time series machine learning
# the "i.i.d" (independent and identically distributed) assumption does not
# hold true as the data points are not independent and have a temporal
# relationship.
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = HistGradientBoostingRegressor().fit(X_train, y_train)



# %%
# Taking a look at the performance of the model.
from sklearn.metrics import mean_absolute_percentage_error

y_pred = model.predict(X_test)
mean_absolute_percentage_error(y_test, y_pred)



# %%
# Proper next hour forecasting evaluation
# ---------------------------------------
# Let's use a proper evaluation splitting strategies that takes into account
# the temporal structure of the dataset to evaluate our model's ability to
# predict data points in the future (to avoid cheating by reading values from
# the lagged features in the training set).
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(
    n_splits=3,  # to keep the notebook fast enough on common laptops
    gap=48,  # 2 days data gap between train and test
    max_train_size=10000,  # keep train sets of comparable sizes
    test_size=3000,  # for 2 or 3 digits of precision in scores
)
all_splits = list(ts_cv.split(X, y))



# %%
# Training the model and evaluating its performance based on MAPE.
train_idx, test_idx = all_splits[0]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]

model = HistGradientBoostingRegressor().fit(X_train, y_train)
y_pred = model.predict(X_test)
mean_absolute_percentage_error(y_test, y_pred)



# %%
# The generalization error measured via a shuffled trained test split
# is too optimistic. The generalization via a time-based split is likely to
# be more representative of the true performance of the regression model.
# Let's assess this variability of our error evaluation with proper
# cross-validation:
from sklearn.model_selection import cross_val_score

# 使用交叉验证计算负的平均绝对百分比误差
cv_mape_scores = -cross_val_score(
    model, X, y, cv=ts_cv, scoring="neg_mean_absolute_percentage_error"
)
cv_mape_scores

# %%
# 在不同的交叉验证分割中，得分变化较大！在实际应用中，建议增加分割以更好地评估变化性。
# 从现在开始报告平均CV得分及其标准差。
print(f"CV MAPE: {cv_mape_scores.mean():.3f} ± {cv_mape_scores.std():.3f}")

# %%
# 我们可以计算多种组合的评估指标和损失函数，这些将稍后报告。
from collections import defaultdict

from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_pinball_loss,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_validate


def consolidate_scores(cv_results, scores, metric):
    if metric == "MAPE":
        scores[metric].append(f"{value.mean():.2f} ± {value.std():.2f}")
    else:
        scores[metric].append(f"{value.mean():.1f} ± {value.std():.1f}")

    return scores


# 定义多个评分标准和损失函数
scoring = {
    "MAPE": make_scorer(mean_absolute_percentage_error),
    "RMSE": make_scorer(root_mean_squared_error),
    "MAE": make_scorer(mean_absolute_error),
    "pinball_loss_05": make_scorer(mean_pinball_loss, alpha=0.05),
    "pinball_loss_50": make_scorer(mean_pinball_loss, alpha=0.50),
    "pinball_loss_95": make_scorer(mean_pinball_loss, alpha=0.95),
}
loss_functions = ["squared_error", "poisson", "absolute_error"]
scores = defaultdict(list)

# 对每种损失函数进行模型拟合和交叉验证
for loss_func in loss_functions:
    model = HistGradientBoostingRegressor(loss=loss_func)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=ts_cv,
        scoring=scoring,
        n_jobs=2,
    )
    time = cv_results["fit_time"]
    scores["loss"].append(loss_func)
    scores["fit_time"].append(f"{time.mean():.2f} ± {time.std():.2f} s")

    for key, value in cv_results.items():
        if key.startswith("test_"):
            metric = key.split("test_")[1]
            scores = consolidate_scores(cv_results, scores, metric)


# %%
# 通过分位数回归建模预测不确定性
# -------------------------------------------------------
# 不同于最小二乘和泊松损失函数模型 :math:`Y|X` 的期望值，
# 分位数回归尝试估计条件分布的分位数。
#
# 对于给定的数据点 :math:`x_i`， :math:`Y|X=x_i` 预期是一个随机变量，
# 因为我们预计租车数量无法从特征完全准确预测。它可能受到未能完全捕捉的其他变量的影响，
# 例如未来一小时是否会下雨，这些不能完全从过去几小时的自行车租赁数据中预测出来。
# 这就是我们所说的随机不确定性。
#
# 分位数回归使得能够更精细地描述这种不确定性。
# 定义用于计算分位数回归模型的分位数列表
quantile_list = [0.05, 0.5, 0.95]

# 遍历每个分位数
for quantile in quantile_list:
    # 使用分位数回归模型创建 HistGradientBoostingRegressor 对象
    model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile)
    
    # 使用时间序列交叉验证进行模型评估
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=ts_cv,
        scoring=scoring,
        n_jobs=2,
    )
    
    # 获取拟合时间
    time = cv_results["fit_time"]
    
    # 将拟合时间的平均值和标准差格式化后存入 scores 字典中
    scores["fit_time"].append(f"{time.mean():.2f} ± {time.std():.2f} s")

    # 将损失函数类型和分位数存入 scores 字典中
    scores["loss"].append(f"quantile {int(quantile*100)}")
    
    # 合并评分结果到 scores 字典中
    for key, value in cv_results.items():
        if key.startswith("test_"):
            metric = key.split("test_")[1]
            scores = consolidate_scores(cv_results, scores, metric)

# 将 scores 转换为 Pandas DataFrame
scores_df = pl.DataFrame(scores)
scores_df

# %%
# 查看最小化每个指标的损失值
def min_arg(col):
    col_split = pl.col(col).str.split(" ")
    return pl.arg_sort_by(
        col_split.list.get(0).cast(pl.Float64),
        col_split.list.get(2).cast(pl.Float64),
    ).first()

# 选择最小损失值对应的数据
scores_df.select(
    pl.col("loss").get(min_arg(col_name)).alias(col_name)
    for col_name in scores_df.columns
    if col_name != "loss"
)

# %%
# 即使由于数据集的方差而导致分数分布重叠，
# 当 `loss="squared_error"` 时，平均 RMSE 较低；
# 而当 `loss="absolute_error"` 时，平均 MAPE 较低，符合预期。
# 对于分位数损失，5 和 95 分位数的 Mean Pinball Loss 也是如此。
# 50 分位数损失对应的分数与最小化其他损失函数获得的分数重叠，MAE 也是如此。
#
# 对预测的定性观察
# -------------------
# 现在我们可以可视化模型在第 5 百分位数、中位数和第 95 百分位数的性能：
all_splits = list(ts_cv.split(X, y))
train_idx, test_idx = all_splits[0]

X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]

max_iter = 50

# 使用 Poisson 损失函数的 HistGradientBoostingRegressor 模型进行拟合
gbrt_mean_poisson = HistGradientBoostingRegressor(loss="poisson", max_iter=max_iter)
gbrt_mean_poisson.fit(X_train, y_train)
mean_predictions = gbrt_mean_poisson.predict(X_test)

# 使用分位数为 0.5 的 HistGradientBoostingRegressor 模型进行拟合
gbrt_median = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.5, max_iter=max_iter
)
gbrt_median.fit(X_train, y_train)
median_predictions = gbrt_median.predict(X_test)

# 使用分位数为 0.05 的 HistGradientBoostingRegressor 模型进行拟合
gbrt_percentile_5 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.05, max_iter=max_iter
)
gbrt_percentile_5.fit(X_train, y_train)
percentile_5_predictions = gbrt_percentile_5.predict(X_test)

# 使用分位数为 0.95 的 HistGradientBoostingRegressor 模型进行拟合
gbrt_percentile_95 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.95, max_iter=max_iter
)
gbrt_percentile_95.fit(X_train, y_train)
percentile_95_predictions = gbrt_percentile_95.predict(X_test)

# %%
# 现在我们可以查看回归模型的预测结果：
last_hours = slice(-96, None)
fig, ax = plt.subplots(figsize=(15, 7))
plt.title("Predictions by regression models")
ax.plot(
    y_test[last_hours],
    "x-",
    alpha=0.2,
    label="Actual demand",
    color="black",
)
ax.plot(
    median_predictions[last_hours],
    "^-",
    label="GBRT median",
)
ax.plot(
    mean_predictions[last_hours],
    "x-",
    label="GBRT mean (Poisson)",
)
ax.fill_between(
    np.arange(96),
    percentile_5_predictions[last_hours],
    percentile_95_predictions[last_hours],
    alpha=0.3,
    label="GBRT 90% interval",
)
_ = ax.legend()

# %%
# 这里值得注意的是，蓝色区域在5%和95%百分位估计器之间的宽度随着一天中的时间变化而变化：
#
# - 在夜晚，蓝色带非常窄：模型对自行车租赁数量有较高的确定性预测。而且这些预测似乎是正确的，因为实际需求确实落在这个蓝色带内。
# - 白天，蓝色带较宽：不确定性增加，可能是由于天气变化的不确定性，特别是在周末。
# - 在工作日，我们还可以看到通勤模式在5%和95%估计中仍然可见。
# - 最后，预计有10%的时间，实际需求不在5%和95%百分位估计之间。在这个测试范围内，实际需求似乎更高，尤其是在高峰时段。这可能表明我们的95%百分位估计低估了需求峰值。这可以通过计算如 :ref:`置信区间校准 <calibration-section>` 中所做的经验覆盖率来定量确认。
#
# 通过观察非线性回归模型与最佳模型的性能：
from sklearn.metrics import PredictionErrorDisplay

fig, axes = plt.subplots(ncols=3, figsize=(15, 6), sharey=True)
fig.suptitle("Non-linear regression models")
predictions = [
    median_predictions,
    percentile_5_predictions,
    percentile_95_predictions,
]
labels = [
    "Median",
    "5th percentile",
    "95th percentile",
]
for ax, pred, label in zip(axes, predictions, labels):
    PredictionErrorDisplay.from_predictions(
        y_true=y_test,
        y_pred=pred,
        kind="residual_vs_predicted",
        scatter_kwargs={"alpha": 0.3},
        ax=ax,
    )
    ax.set(xlabel="Predicted demand", ylabel="True demand")
    ax.legend(["Best model", label])

plt.show()

# %%
# 结论
# ----------
# 通过这个例子，我们使用滞后特征探索了时间序列预测。我们比较了一个简单的回归模型（使用标准化的 :class:`~sklearn.model_selection.train_test_split`）与使用 :class:`~sklearn.model_selection.TimeSeriesSplit` 的适当时间序列评估策略。我们观察到，使用默认值 `shuffle=True` 的 :class:`~sklearn.model_selection.train_test_split` 训练的模型产生了过于乐观的平均绝对百分比误差（MAPE）结果。
# produced from the time-based split better represent the performance
# of our time-series regression model. We also analyzed the predictive uncertainty
# of our model via Quantile Regression. Predictions based on the 5th and
# 95th percentile using `loss="quantile"` provide us with a quantitative estimate
# of the uncertainty of the forecasts made by our time series regression model.
# Uncertainty estimation can also be performed
# using `MAPIE <https://mapie.readthedocs.io/en/latest/index.html>`_,
# that provides an implementation based on recent work on conformal prediction
# methods and estimates both aleatoric and epistemic uncertainty at the same time.
# Furthermore, functionalities provided
# by `sktime <https://www.sktime.net/en/latest/users.html>`_
# can be used to extend scikit-learn estimators by making use of recursive time
# series forecasting, that enables dynamic predictions of future values.
```