# `D:\src\scipysrc\scikit-learn\examples\applications\plot_cyclical_feature_engineering.py`

```
"""
================================
Time-related feature engineering
================================

This notebook introduces different strategies to leverage time-related features
for a bike sharing demand regression task that is highly dependent on business
cycles (days, weeks, months) and yearly season cycles.

In the process, we introduce how to perform periodic feature engineering using
the :class:`sklearn.preprocessing.SplineTransformer` class and its
`extrapolation="periodic"` option.

"""

# %%
# Data exploration on the Bike Sharing Demand dataset
# ---------------------------------------------------
#
# We start by loading the data from the OpenML repository.
from sklearn.datasets import fetch_openml

# Fetching the Bike Sharing Demand dataset version 2 as a dataframe
bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
df = bike_sharing.frame

# %%
# To get a quick understanding of the periodic patterns of the data, let us
# have a look at the average demand per hour during a week.
#
# Note that the week starts on a Sunday, during the weekend. We can clearly
# distinguish the commute patterns in the morning and evenings of the work days
# and the leisure use of the bikes on the weekends with a more spread peak
# demand around the middle of the days:
import matplotlib.pyplot as plt

# Creating a figure and axes for plotting the average hourly demand during the week
fig, ax = plt.subplots(figsize=(12, 4))
average_week_demand = df.groupby(["weekday", "hour"])["count"].mean()
average_week_demand.plot(ax=ax)
_ = ax.set(
    title="Average hourly bike demand during the week",
    xticks=[i * 24 for i in range(7)],
    xticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    xlabel="Time of the week",
    ylabel="Number of bike rentals",
)

# %%
#
# The target of the prediction problem is the absolute count of bike rentals on
# a hourly basis:
df["count"].max()

# %%
#
# Let us rescale the target variable (number of hourly bike rentals) to predict
# a relative demand so that the mean absolute error is more easily interpreted
# as a fraction of the maximum demand.
#
# .. note::
#
#     The fit method of the models used in this notebook all minimize the
#     mean squared error to estimate the conditional mean.
#     The absolute error, however, would estimate the conditional median.
#
#     Nevertheless, when reporting performance measures on the test set in
#     the discussion, we choose to focus on the mean absolute error instead
#     of the (root) mean squared error because it is more intuitive to
#     interpret. Note, however, that in this study the best models for one
#     metric are also the best ones in terms of the other metric.
y = df["count"] / df["count"].max()

# %%
fig, ax = plt.subplots(figsize=(12, 4))
# Plotting a histogram of the relative demand values
y.hist(bins=30, ax=ax)
_ = ax.set(
    xlabel="Fraction of rented fleet demand",
    ylabel="Number of hours",
)

# %%
# The input feature data frame is a time annotated hourly log of variables
# describing the weather conditions. It includes both numerical and categorical
# variables. Note that the time information has already been expanded into
# several complementary columns.
#
# Drop the column named "count" from the dataframe `df` to get `X`, which now contains
# all other columns except "count".
X = df.drop("count", axis="columns")
X

# %%
# .. note::
#
#    If the time information was only present as a date or datetime column, we
#    could have expanded it into hour-in-the-day, day-in-the-week,
#    day-in-the-month, month-in-the-year using pandas:
#    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components
#
# We now introspect the distribution of the categorical variables, starting
# with `"weather"`:
#
# Count the occurrences of each unique value in the "weather" column of dataframe `X`.
X["weather"].value_counts()

# %%
# Since there are only 3 `"heavy_rain"` events, we cannot use this category to
# train machine learning models with cross validation. Instead, we simplify the
# representation by collapsing those into the `"rain"` category.
#
# Replace occurrences of "heavy_rain" in the "weather" column of `X` with "rain",
# ensuring the column remains categorical after the replacement.
X["weather"] = (
    X["weather"]
    .astype(object)  # Cast to object to allow string replacement
    .replace(to_replace="heavy_rain", value="rain")  # Replace "heavy_rain" with "rain"
    .astype("category")  # Cast back to categorical dtype
)

# %%
# Display the updated counts of each unique value in the "weather" column of `X`.
X["weather"].value_counts()

# %%
# As expected, the `"season"` variable is well balanced:
#
# Count the occurrences of each unique value in the "season" column of dataframe `X`.
X["season"].value_counts()

# %%
# Time-based cross-validation
# ---------------------------
#
# Since the dataset is a time-ordered event log (hourly demand), we will use a
# time-sensitive cross-validation splitter to evaluate our demand forecasting
# model as realistically as possible. We use a gap of 2 days between the train
# and test side of the splits. We also limit the training set size to make the
# performance of the CV folds more stable.
#
# 1000 test datapoints should be enough to quantify the performance of the
# model. This represents a bit less than a month and a half of contiguous test
# data:
#
# Import `TimeSeriesSplit` from sklearn.model_selection for time-series cross-validation.
from sklearn.model_selection import TimeSeriesSplit

# %%
# Let us manually inspect the various splits to check that the
# `TimeSeriesSplit` works as we expect, starting with the first split:
#
# Generate splits of the data `X` and its target `y` using the `TimeSeriesSplit` object `ts_cv`,
# and retrieve the first split indices.
all_splits = list(ts_cv.split(X, y))
train_0, test_0 = all_splits[0]

# %%
# Display the subset of `X` corresponding to the test set of the first split.
X.iloc[test_0]

# %%
# Display the subset of `X` corresponding to the training set of the first split.
X.iloc[train_0]

# %%
# We now inspect the last split:
#
# Retrieve the indices for the last split of the data into training and test sets.
train_4, test_4 = all_splits[4]

# %%
# Display the subset of `X` corresponding to the test set of the last split.
X.iloc[test_4]

# %%
# Display the subset of `X` corresponding to the training set of the last split.
X.iloc[train_4]

# %%
# All is well. We are now ready to do some predictive modeling!
#
# Gradient Boosting
# -----------------
#
# Gradient Boosting Regression with decision trees is often flexible enough to
# efficiently handle heterogeneous tabular data with a mix of categorical and
# numerical features as long as the number of samples is large enough.
#
# Here, we use the modern
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` with native support
# for categorical features. Therefore, we only need to set
# `categorical_features="from_dtype"` such that features with categorical dtype
# are considered categorical features. For reference, we extract the categorical
# features from the dataframe based on the dtype. The internal trees use a dedicated
# tree splitting rule for these features.
#
# 导入必要的库和模块
from sklearn.compose import ColumnTransformer  # 导入列转换器，用于处理不同列的预处理
from sklearn.ensemble import HistGradientBoostingRegressor  # 导入直方图梯度提升回归模型
from sklearn.model_selection import cross_validate  # 导入交叉验证函数
from sklearn.pipeline import make_pipeline  # 导入管道构建函数

# 创建直方图梯度提升回归模型，并设置随机种子
gbrt = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)

# 确定数据集中的分类特征列
categorical_columns = X.columns[X.dtypes == "category"]
print("Categorical features:", categorical_columns.tolist())

# %%
# 评估梯度提升回归模型的性能，使用平均绝对误差（MAE）来衡量相对需求
import numpy as np

def evaluate(model, X, y, cv, model_prop=None, model_step=None):
    # 执行交叉验证，返回多个性能指标的结果
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_estimator=model_prop is not None,
    )
    if model_prop is not None:
        # 如果指定了模型属性和步骤，则提取每个估计器的指定属性值
        if model_step is not None:
            values = [
                getattr(m[model_step], model_prop) for m in cv_results["estimator"]
            ]
        else:
            values = [getattr(m, model_prop) for m in cv_results["estimator"]]
        print(f"Mean model.{model_prop} = {np.mean(values)}")
    
    # 提取 MAE 和 RMSE 指标的平均值和标准差，并打印出来
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )

# %%
# 我们设置了足够大的 `max_iter` 值，以便触发了早期停止。
#
# 该模型的平均误差约为最大需求的4到5%。这对于没有任何超参数调整的第一次尝试来说非常不错！
# 我们只需明确指定了分类变量。请注意，时间相关特征按原样传递，即无需对其进行处理。
# 但对于树模型来说，这并不是问题，因为它们可以学习输入特征与目标之间的非单调关系。
#
# 线性回归的情况将会有所不同，接下来我们会看到。
#
# 简单的线性回归
# -----------------------
#
# 与线性模型一般情况下一样，分类变量需要进行独热编码。
# 为了保持一致性，我们使用 :class:`~sklearn.preprocessing.MinMaxScaler` 将数值特征缩放到相同的0-1范围内，
# 尽管在这种情况下，它并不会对结果产生太大影响，因为它们已经在可比较的尺度上：
from sklearn.linear_model import RidgeCV  # 导入岭回归模型
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # 导入MinMaxScaler和OneHotEncoder
    # 使用ColumnTransformer进行数据转换，将类别特征进行独热编码，保留其它特征不变
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, categorical_columns),  # 对类别特征进行独热编码的转换器
        ],
        remainder=MinMaxScaler(),  # 对剩余的数值特征进行MinMax缩放的转换器
    ),
    
    # 使用RidgeCV进行岭回归模型的交叉验证，指定不同的alpha值
    RidgeCV(alphas=alphas),
# %%
# 使用 `evaluate` 函数评估 `naive_linear_pipeline` 模型在指定的交叉验证集 `ts_cv` 上的性能，
# 模型参数为 "alpha_"，模型为 RidgeCV。
evaluate(
    naive_linear_pipeline, X, y, cv=ts_cv, model_prop="alpha_", model_step="ridgecv"
)

# %%
# 确认所选的 `alpha_` 是否在我们指定的范围内。
#
# 结果表明性能不佳：平均误差约为最大需求的14%。这比梯度提升模型的平均误差高出三倍以上。
# 我们可以怀疑，周期性时间相关特征的简单原始编码（仅进行了最小-最大缩放）可能阻碍了线性回归模型
# 充分利用时间信息：线性回归模型不能自动建模输入特征和目标之间的非单调关系。需要在输入中
# 工程化非线性项。
#
# 例如，对 `"hour"` 特征的原始数值编码阻止了线性模型识别到，早上6点到8点的小时增加应该对
# 自行车租赁数量有很大的正面影响，而晚上18点到20点的类似增加则应对预测的自行车租赁数量有
# 较大的负面影响。
#
# 将时间步长视为类别变量
# ------------------------
#
# 由于时间特征以整数的形式编码（例如 "hours" 特征有24个唯一值），我们可以决定将其视为类别变量，
# 使用独热编码来处理，从而忽略时间值排序所隐含的任何假设。
#
# 使用独热编码时间特征可以使线性模型具有更大的灵活性，因为我们为每个离散时间级别引入了一个额外的特征。
one_hot_linear_pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, categorical_columns),
            ("one_hot_time", one_hot_encoder, ["hour", "weekday", "month"]),
        ],
        remainder=MinMaxScaler(),
    ),
    RidgeCV(alphas=alphas),
)

# 使用 `evaluate` 函数评估 `one_hot_linear_pipeline` 模型在指定的交叉验证集 `ts_cv` 上的性能。
evaluate(one_hot_linear_pipeline, X, y, cv=ts_cv)

# %%
# 这个模型的平均误差率为10%，比使用时间特征的原始（序数）编码要好得多，验证了我们的直觉，
# 即线性回归模型受益于不将时间进展视为单调的灵活性。
#
# 然而，这引入了大量新的特征。如果将一天中的时间表示为从当天开始的分钟数，而不是小时数，
# 独热编码将引入1440个特征而不是24个。这可能导致显著的过拟合问题。为避免这种情况，我们
# 可以使用 :func:`sklearn.preprocessing.KBinsDiscretizer` 来重新分箱细粒度序数或数值变量的
# 水平数量，同时仍然从独热编码的非单调表现优势中受益。
#
# 最后，我们还观察到，独热编码完全忽略了小时级别的排序，而这可能是一个有趣的归纳偏差。
# 导入所需的库和模块
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV

# 定义一个函数，返回一个将输入数据按照给定周期进行正弦变换的转换器对象
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

# 定义一个函数，返回一个将输入数据按照给定周期进行余弦变换的转换器对象
def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# %%
# 创建一个包含小时数据的DataFrame，并对"hour"特征进行正弦和余弦变换
hour_df = pd.DataFrame(
    np.arange(26).reshape(-1, 1),
    columns=["hour"],
)
hour_df["hour_sin"] = sin_transformer(24).fit_transform(hour_df)["hour"]
hour_df["hour_cos"] = cos_transformer(24).fit_transform(hour_df)["hour"]

# 绘制时钟特征编码后的效果图
hour_df.plot(x="hour")
_ = plt.title("Trigonometric encoding for the 'hour' feature")

# %%
# 使用二维散点图展示时钟特征的正弦和余弦编码，其中颜色表示小时数，类似于24小时模拟时钟的映射效果
fig, ax = plt.subplots(figsize=(7, 5))
sp = ax.scatter(hour_df["hour_sin"], hour_df["hour_cos"], c=hour_df["hour"])
ax.set(
    xlabel="sin(hour)",
    ylabel="cos(hour)",
)
_ = fig.colorbar(sp)

# %%
# 构建特征提取流水线，包括类别特征的独热编码和时间特征的正弦余弦变换
cyclic_cossin_transformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, categorical_columns),  # 对类别特征进行独热编码
        ("month_sin", sin_transformer(12), ["month"]),  # 对月份特征进行周期为12的正弦变换
        ("month_cos", cos_transformer(12), ["month"]),  # 对月份特征进行周期为12的余弦变换
        ("weekday_sin", sin_transformer(7), ["weekday"]),  # 对星期几特征进行周期为7的正弦变换
        ("weekday_cos", cos_transformer(7), ["weekday"]),  # 对星期几特征进行周期为7的余弦变换
        ("hour_sin", sin_transformer(24), ["hour"]),  # 对小时特征进行周期为24的正弦变换
        ("hour_cos", cos_transformer(24), ["hour"]),  # 对小时特征进行周期为24的余弦变换
    ],
    remainder=MinMaxScaler(),  # 对剩余特征进行最小-最大缩放
)

# 构建包括特征提取和线性回归的流水线
cyclic_cossin_linear_pipeline = make_pipeline(
    cyclic_cossin_transformer,
    RidgeCV(alphas=alphas),  # 使用交叉验证选择最优的Ridge回归参数
)

# 使用时间序列交叉验证评估特征提取和线性回归流水线的性能
evaluate(cyclic_cossin_linear_pipeline, X, y, cv=ts_cv)
# 导入需要的库和模块，用于周期性特征的变换和可视化
from sklearn.preprocessing import SplineTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV

# 定义周期性样条变换函数
def periodic_spline_transformer(period, n_splines=None, degree=3):
    # 如果未指定样条数量，则使用周期数作为默认值
    if n_splines is None:
        n_splines = period
    # 根据周期数确定节点数
    n_knots = n_splines + 1
    # 创建并返回样条变换器对象
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

# %%
# 创建一个包含小时数据的DataFrame，用于可视化特征扩展效果
hour_df = pd.DataFrame(
    np.linspace(0, 26, 1000).reshape(-1, 1),
    columns=["hour"],
)
# 对小时数据应用周期性样条变换
splines = periodic_spline_transformer(24, n_splines=12).fit_transform(hour_df)
# 创建包含样条特征的DataFrame
splines_df = pd.DataFrame(
    splines,
    columns=[f"spline_{i}" for i in range(splines.shape[1])],
)
# 组合原始小时数据和样条特征数据，并绘制图表
pd.concat([hour_df, splines_df], axis="columns").plot(x="hour", cmap=plt.cm.tab20b)
_ = plt.title("Periodic spline-based encoding for the 'hour' feature")

# %%
# 通过使用 `extrapolation="periodic"` 参数，观察特征编码在超出午夜后仍保持平滑
#
# 可以使用这种替代周期特征工程策略构建预测管道
#
# 使用比离散级别少的样条可以比独热编码更有效地进行编码，同时保留大部分表达能力
cyclic_spline_transformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, categorical_columns),
        ("cyclic_month", periodic_spline_transformer(12, n_splines=6), ["month"]),
        ("cyclic_weekday", periodic_spline_transformer(7, n_splines=3), ["weekday"]),
        ("cyclic_hour", periodic_spline_transformer(24, n_splines=12), ["hour"]),
    ],
    remainder=MinMaxScaler(),
)
# 创建使用周期性样条编码的线性回归流水线
cyclic_spline_linear_pipeline = make_pipeline(
    cyclic_spline_transformer,
    RidgeCV(alphas=alphas),
)
# 评估线性回归流水线的性能
evaluate(cyclic_spline_linear_pipeline, X, y, cv=ts_cv)

# %%
# 样条特征使得线性模型能够成功利用周期性时间相关特征，将误差从最大需求的 ~14% 降低到 ~10%
#
# 对特征对线性模型预测影响的定性分析
# --------------------------------------------------------------------------
#
# 在这里，我们希望可视化特征工程选择对预测的时间相关形状的影响
#
# 为此，我们考虑一个任意的基于时间的拆分，比较在一系列留出数据点上的预测
# 使用 naive_linear_pipeline 对训练集的第一组进行线性模型训练
naive_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])

# 使用 naive_linear_pipeline 对测试集的第一组进行预测
naive_linear_predictions = naive_linear_pipeline.predict(X.iloc[test_0])

# 使用 one_hot_linear_pipeline 对训练集的第一组进行独热编码特征的线性模型训练
one_hot_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])

# 使用 one_hot_linear_pipeline 对测试集的第一组进行预测
one_hot_linear_predictions = one_hot_linear_pipeline.predict(X.iloc[test_0])

# 使用 cyclic_cossin_linear_pipeline 对训练集的第一组进行三角函数特征的线性模型训练
cyclic_cossin_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])

# 使用 cyclic_cossin_linear_pipeline 对测试集的第一组进行预测
cyclic_cossin_linear_predictions = cyclic_cossin_linear_pipeline.predict(X.iloc[test_0])

# 使用 cyclic_spline_linear_pipeline 对训练集的第一组进行样条函数特征的线性模型训练
cyclic_spline_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])

# 使用 cyclic_spline_linear_pipeline 对测试集的第一组进行预测
cyclic_spline_linear_predictions = cyclic_spline_linear_pipeline.predict(X.iloc[test_0])

# %%
# 我们通过缩放测试集的最后96小时（4天）来可视化这些预测结果，以获取定性洞察:
last_hours = slice(-96, None)
fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle("Predictions by linear models")
ax.plot(
    y.iloc[test_0].values[last_hours],
    "x-",
    alpha=0.2,
    label="Actual demand",
    color="black",
)
ax.plot(naive_linear_predictions[last_hours], "x-", label="Ordinal time features")
ax.plot(
    cyclic_cossin_linear_predictions[last_hours],
    "x-",
    label="Trigonometric time features",
)
ax.plot(
    cyclic_spline_linear_predictions[last_hours],
    "x-",
    label="Spline-based time features",
)
ax.plot(
    one_hot_linear_predictions[last_hours],
    "x-",
    label="One-hot time features",
)
_ = ax.legend()

# %%
# 我们可以从上图中得出以下结论：
#
# - **原始的序数时间相关特征** 存在问题，因为它们未能捕捉到自然的周期性：我们观察到在每天结束时（小时特征从23跳转回0时）预测值出现大幅跳跃。类似的现象可能也会在每周或每年结束时出现。
#
# - 如预期，**三角函数特征**（正弦和余弦）在午夜时并没有这种不连续性，但线性回归模型未能利用这些特征有效地建模一天内的变化。使用更高阶的三角函数特征或具有不同相位的额外三角函数特征可能能够潜在地解决这个问题。
#
# - **周期样条函数特征** 同时解决了这两个问题：它通过使用12个样条函数使线性模型更加表现丰富，可以专注于特定的小时。此外，`extrapolation="periodic"` 选项在 `hour=23` 和 `hour=0` 之间强制实现平滑表示。
#
# - **独热编码特征** 行为与周期样条函数特征类似，但更加尖锐：例如，在工作日的早晨高峰期间，它们可以更好地建模，因为这种高峰持续时间不到一个小时。然而，接下来我们会看到，对于线性模型有利的特性并不一定适用于更加表现丰富的模型。
#
# %%
# 我们还可以比较每种特征工程流水线提取的特征数量：
naive_linear_pipeline[:-1].transform(X).shape
# 使用 naive_linear_pipeline 中除最后一个步骤外的所有转换器对输入 X 进行变换，并返回变换后的形状

# %%
one_hot_linear_pipeline[:-1].transform(X).shape
# 使用 one_hot_linear_pipeline 中除最后一个步骤外的所有转换器对输入 X 进行变换，并返回变换后的形状

# %%
cyclic_cossin_linear_pipeline[:-1].transform(X).shape
# 使用 cyclic_cossin_linear_pipeline 中除最后一个步骤外的所有转换器对输入 X 进行变换，并返回变换后的形状

# %%
cyclic_spline_linear_pipeline[:-1].transform(X).shape
# 使用 cyclic_spline_linear_pipeline 中除最后一个步骤外的所有转换器对输入 X 进行变换，并返回变换后的形状

# %%
# 这段注释解释了通过使用 one-hot 编码和样条编码策略，为时间表示创建了更多特征，
# 比其他策略产生了更高的灵活性（自由度），从而使得下游的线性模型能够避免欠拟合问题。
#
# 最后，我们观察到，无论是基于样条编码还是 one-hot 编码的线性模型都不能很好地
# 预测真实的自行车租赁需求，特别是在高峰期，比如工作日的交通高峰，而在周末这些
# 高峰则相对平缓。基于样条或 one-hot 编码的线性模型往往在周末也预测到与通勤相关
# 的自行车租赁高峰，并且在工作日低估与通勤相关的事件。
#
# 这些系统的预测误差显示出一种欠拟合，这可以通过特征之间缺乏交互项（例如 "工作日"
# 和由 "小时" 派生的特征）来解释。这个问题将在接下来的部分中解决。

# %%
# 使用样条和多项式特征建模成对交互
# -------------------------------------------------------------------
#
# 线性模型不会自动捕捉输入特征之间的交互效应。这一点在通过 `SplineTransformer`
# （或 one-hot 编码或分箱）构造的特征中尤为明显，因为一些特征在边际上是非线性的。
#
# 然而，可以利用 `PolynomialFeatures` 类在粗粒度的样条编码小时上显式地建模
# "工作日" / "小时" 的交互，而不引入过多的新变量：

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures

hour_workday_interaction = make_pipeline(
    ColumnTransformer(
        [
            ("cyclic_hour", periodic_spline_transformer(24, n_splines=8), ["hour"]),
            ("workingday", FunctionTransformer(lambda x: x == "True"), ["workingday"]),
        ]
    ),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
)

# %%
# 然后将这些特征与先前样条基础管道中已计算的特征结合起来。通过显式建模这种成对
# 交互，我们可以观察到性能得到了显著改善：

cyclic_spline_interactions_pipeline = make_pipeline(
    FeatureUnion(
        [
            ("marginal", cyclic_spline_transformer),
            ("interactions", hour_workday_interaction),
        ]
    ),
    RidgeCV(alphas=alphas),
)
evaluate(cyclic_spline_interactions_pipeline, X, y, cv=ts_cv)

# %%
# 使用核函数建模非线性特征交互
# -----------------------------------------------------
#
# 前面的分析强调了模型化输入特征之间交互的必要性
# 导入 Nyström 方法，用于核函数的近似多项式展开
from sklearn.kernel_approximation import Nystroem

# 创建一个包含循环样条变换、Nyström 多项式核函数近似和 RidgeCV 正则化的管道
cyclic_spline_poly_pipeline = make_pipeline(
    cyclic_spline_transformer,  # 使用循环样条变换器进行特征转换
    Nystroem(kernel="poly", degree=2, n_components=300, random_state=0),  # 使用多项式核函数的 Nyström 近似
    RidgeCV(alphas=alphas),  # 使用交叉验证确定的 Ridge 回归正则化参数
)

# 在时间序列交叉验证下评估循环样条多项式管道的性能
evaluate(cyclic_spline_poly_pipeline, X, y, cv=ts_cv)

# %%
# 观察到该模型的表现几乎可以与梯度提升树相媲美，平均误差约为最大需求的 5%
#
# 注意，尽管管道的最终步骤是线性回归模型，但中间步骤如样条特征提取和 Nyström 核逼近都是高度非线性的。
# 因此，这个复合管道比使用原始特征的简单线性回归模型更具表现力。
#
# 为了完整起见，我们还评估了独热编码和核函数近似的组合：
one_hot_poly_pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, categorical_columns),  # 对分类特征使用独热编码
            ("one_hot_time", one_hot_encoder, ["hour", "weekday", "month"]),  # 对时间相关特征使用独热编码
        ],
        remainder="passthrough",  # 保留其余的特征不变
    ),
    Nystroem(kernel="poly", degree=2, n_components=300, random_state=0),  # 使用多项式核函数的 Nyström 近似
    RidgeCV(alphas=alphas),  # 使用交叉验证确定的 Ridge 回归正则化参数
)

# 在时间序列交叉验证下评估独热编码多项式管道的性能
evaluate(one_hot_poly_pipeline, X, y, cv=ts_cv)

# %%
# 当使用线性模型时，独热编码特征在竞争力上与样条特征相当，但使用非线性核的低秩逼近时情况有所不同：
# 这可以解释为样条特征更加平滑，允许核函数逼近找到更具表现力的决策函数。
#
# 现在让我们定性地观察核模型和能够更好地模拟特征之间非线性交互的梯度提升树的预测结果：
gbrt.fit(X.iloc[train_0], y.iloc[train_0])  # 使用梯度提升树拟合训练集
gbrt_predictions = gbrt.predict(X.iloc[test_0])  # 对测试集进行预测

one_hot_poly_pipeline.fit(X.iloc[train_0], y.iloc[train_0])  # 使用独热编码多项式管道拟合训练集
one_hot_poly_predictions = one_hot_poly_pipeline.predict(X.iloc[test_0])  # 对测试集进行预测

cyclic_spline_poly_pipeline.fit(X.iloc[train_0], y.iloc[train_0])  # 使用循环样条多项式管道拟合训练集
cyclic_spline_poly_predictions = cyclic_spline_poly_pipeline.predict(X.iloc[test_0])  # 对测试集进行预测

# %%
# 再次聚焦于测试集的最后四天：
last_hours = slice(-96, None)
fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle("Predictions by non-linear regression models")
ax.plot(
    y.iloc[test_0].values[last_hours],
    "x-",
    alpha=0.2,
    label="Actual demand",
    color="black",
)
ax.plot(
    gbrt_predictions[last_hours],
    "x-",
    label="Gradient Boosted Trees",
)
ax.plot(
    one_hot_poly_predictions[last_hours],
    "x-",
    label="One-hot + polynomial kernel",
)
ax.plot(
    cyclic_spline_poly_predictions[last_hours],
    "x-",
    label="Splines + polynomial kernel",
)
_ = ax.legend()


# %%
# 首先，注意到树可以自然地建模非线性特征交互，因为决策树默认可以超过2级深度。
#
# 在这里，我们可以观察到，样条特征与非线性核心的组合效果相当不错，几乎可以与梯度提升回归树的准确性相媲美。
#
# 相比之下，使用热独编码的时间特征在低秩核模型下表现不佳。特别是在估计低需求小时方面比竞争模型显著高估。
#
# 我们还观察到，这些模型都无法成功预测工作日高峰租车需求的某些峰值。可能需要访问其他特征来进一步改善预测准确性。例如，随时访问车队的地理分布或因需要维修而无法行驶的自行车的比例可能会很有用。
#
# 最后，让我们通过真实与预测需求的散点图，更详细地查看这三个模型的预测误差：
from sklearn.metrics import PredictionErrorDisplay

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 7), sharex=True, sharey="row")
fig.suptitle("Non-linear regression models", y=1.0)
predictions = [
    one_hot_poly_predictions,
    cyclic_spline_poly_predictions,
    gbrt_predictions,
]
labels = [
    "One hot +\npolynomial kernel",
    "Splines +\npolynomial kernel",
    "Gradient Boosted\nTrees",
]
plot_kinds = ["actual_vs_predicted", "residual_vs_predicted"]
for axis_idx, kind in enumerate(plot_kinds):
    for ax, pred, label in zip(axes[axis_idx], predictions, labels):
        disp = PredictionErrorDisplay.from_predictions(
            y_true=y.iloc[test_0],
            y_pred=pred,
            kind=kind,
            scatter_kwargs={"alpha": 0.3},
            ax=ax,
        )
        ax.set_xticks(np.linspace(0, 1, num=5))
        if axis_idx == 0:
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.legend(
                ["Best model", label],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.3),
                ncol=2,
            )
        ax.set_aspect("equal", adjustable="box")
plt.show()
# %%
# 这个可视化结果确认了我们在前面图表中得出的结论。
#
# All models under-estimate the high demand events (working day rush hours),
# but gradient boosting a bit less so. The low demand events are well predicted
# on average by gradient boosting while the one-hot polynomial regression
# pipeline seems to systematically over-estimate demand in that regime. Overall
# the predictions of the gradient boosted trees are closer to the diagonal than
# for the kernel models.
#
# Concluding remarks
# ------------------
#
# We note that we could have obtained slightly better results for kernel models
# by using more components (higher rank kernel approximation) at the cost of
# longer fit and prediction durations. For large values of `n_components`, the
# performance of the one-hot encoded features would even match the spline
# features.
#
# The `Nystroem` + `RidgeCV` regressor could also have been replaced by
# :class:`~sklearn.neural_network.MLPRegressor` with one or two hidden layers
# and we would have obtained quite similar results.
#
# The dataset we used in this case study is sampled on a hourly basis. However
# cyclic spline-based features could model time-within-day or time-within-week
# very efficiently with finer-grained time resolutions (for instance with
# measurements taken every minute instead of every hours) without introducing
# more features. One-hot encoding time representations would not offer this
# flexibility.
#
# Finally, in this notebook we used `RidgeCV` because it is very efficient from
# a computational point of view. However, it models the target variable as a
# Gaussian random variable with constant variance. For positive regression
# problems, it is likely that using a Poisson or Gamma distribution would make
# more sense. This could be achieved by using
# `GridSearchCV(TweedieRegressor(power=2), param_grid({"alpha": alphas}))`
# instead of `RidgeCV`.
```