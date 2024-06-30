# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_tweedie_regression_insurance_claims.py`

```
"""
======================================
Tweedie regression on insurance claims
======================================

This example illustrates the use of Poisson, Gamma and Tweedie regression on
the `French Motor Third-Party Liability Claims dataset
<https://www.openml.org/d/41214>`_, and is inspired by an R tutorial [1]_.

In this dataset, each sample corresponds to an insurance policy, i.e. a
contract within an insurance company and an individual (policyholder).
Available features include driver age, vehicle age, vehicle power, etc.

A few definitions: a *claim* is the request made by a policyholder to the
insurer to compensate for a loss covered by the insurance. The *claim amount*
is the amount of money that the insurer must pay. The *exposure* is the
duration of the insurance coverage of a given policy, in years.

Here our goal is to predict the expected
value, i.e. the mean, of the total claim amount per exposure unit also
referred to as the pure premium.

There are several possibilities to do that, two of which are:

1. Model the number of claims with a Poisson distribution, and the average
   claim amount per claim, also known as severity, as a Gamma distribution
   and multiply the predictions of both in order to get the total claim
   amount.
2. Model the total claim amount per exposure directly, typically with a Tweedie
   distribution of Tweedie power :math:`p \\in (1, 2)`.

In this example we will illustrate both approaches. We start by defining a few
helper functions for loading the data and visualizing results.

.. [1]  A. Noll, R. Salzmann and M.V. Wuthrich, Case Study: French Motor
    Third-Party Liability Claims (November 8, 2018). `doi:10.2139/ssrn.3164764
    <https://doi.org/10.2139/ssrn.3164764>`_
"""

# %%

# 导入必要的库和模块
from functools import partial  # 导入 functools 库中的 partial 函数，用于创建偏函数
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算
import pandas as pd  # 导入 pandas 库，用于数据处理

from sklearn.datasets import fetch_openml  # 导入 fetch_openml 函数，用于从 openml 获取数据集
from sklearn.metrics import (  # 导入 sklearn.metrics 库中的评估指标
    mean_absolute_error,  # 平均绝对误差
    mean_squared_error,  # 均方误差
    mean_tweedie_deviance,  # Tweedie 分布的偏差
)


def load_mtpl2(n_samples=None):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # 从 https://www.openml.org/d/41214 获取 freMTPL2freq 数据集
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)  # 将 IDpol 列转换为整数类型
    df_freq.set_index("IDpol", inplace=True)  # 将 IDpol 列设为索引

    # 从 https://www.openml.org/d/41215 获取 freMTPL2sev 数据集
    df_sev = fetch_openml(data_id=41215, as_frame=True).data

    # 对相同 ID 的 ClaimAmount 进行求和
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")  # 左连接两个数据集
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)  # 将缺失的 ClaimAmount 值填充为 0

    # unquote string fields
    # 遍历数据框中所有数据类型为 object 的列名
    for column_name in df.columns[df.dtypes.values == object]:
        # 对每个列名所对应的列，去除每个元素开头和结尾的单引号字符
        df[column_name] = df[column_name].str.strip("'")
    # 返回数据框的前 n_samples 行
    return df.iloc[:n_samples]
    # 评估一个估计器在训练集和测试集上的性能，使用不同的评估指标
    metrics = [
        ("D² explained", None),  # 如果存在默认评分器，则使用它
        ("mean abs. error", mean_absolute_error),  # 平均绝对误差
        ("mean squared error", mean_squared_error),  # 均方误差
    ]
    if tweedie_powers:
        # 如果提供了 tweedie_powers，则加入 Tweedie 分布相关的评估指标
        metrics += [
            (
                "mean Tweedie dev p={:.4f}".format(power),
                partial(mean_tweedie_deviance, power=power),
            )
            for power in tweedie_powers
        ]

    # 存储每个子集（训练集和测试集）的评估结果
    res = []
    for subset_label, X, df in [
        ("train", X_train, df_train),  # 训练集标签及其数据
        ("test", X_test, df_test),  # 测试集标签及其数据
    # 循环遍历 metrics 列表中的指标和标签对
    ]:
        # 从数据框中获取目标变量 y 和权重变量 _weights
        y, _weights = df[target], df[weights]
        # 遍历 metrics 列表中的每个指标和标签对
        for score_label, metric in metrics:
            # 检查估计器是否是元组且长度为2，表示是频率和严重性模型的乘积
            if isinstance(estimator, tuple) and len(estimator) == 2:
                # 对由频率和严重性模型组成的估计器进行评分
                # 预测结果为频率模型预测值乘以严重性模型预测值
                est_freq, est_sev = estimator
                y_pred = est_freq.predict(X) * est_sev.predict(X)
            else:
                # 否则，直接使用估计器预测 X 的结果
                y_pred = estimator.predict(X)

            # 如果指标为 None，则检查估计器是否有 score 方法，否则跳过
            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                # 计算估计器的评分，考虑样本权重 _weights
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                # 否则，使用给定的 metric 函数计算评分，考虑样本权重 _weights
                score = metric(y, y_pred, sample_weight=_weights)

            # 将评分结果以字典形式加入到结果列表 res 中
            res.append({"subset": subset_label, "metric": score_label, "score": score})

    # 将结果列表 res 转换为 Pandas 数据框
    res = (
        pd.DataFrame(res)
        # 将 "metric" 和 "subset" 列设为索引
        .set_index(["metric", "subset"])
        # 对 "score" 列进行逆透视，以 "metric" 为行索引，"subset" 为列索引
        .score.unstack(-1)
        # 对透视后的结果四舍五入保留四位小数
        .round(4)
        # 仅选择列中包含 "train" 和 "test" 的部分
        .loc[:, ["train", "test"]]
    )
    # 返回最终的结果数据框 res
    return res
# %%
# Loading datasets, basic feature extraction and target definitions
# -----------------------------------------------------------------
#
# We load the freMTPL2 dataset using a function called `load_mtpl2()`.
# This dataset consists of insurance policy information including columns
# such as `ClaimNb` (number of claims), `Exposure` (policy exposure),
# and `ClaimAmount` (amount claimed).
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

# Load the freMTPL2 dataset into DataFrame `df`
df = load_mtpl2()

# Correct for unreasonable observations in `ClaimNb`, `Exposure`, and `ClaimAmount`
# by setting upper limits. Adjust `ClaimNb` to 0 if `ClaimAmount` is 0 but `ClaimNb`
# is greater than or equal to 1, ensuring positive claim amounts for consistency.
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)
df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

# Create a pipeline for logarithmic transformation followed by standard scaling
log_scale_transformer = make_pipeline(
    FunctionTransformer(func=np.log), StandardScaler()
)

# Define a ColumnTransformer for preprocessing different columns in `df`
column_trans = ColumnTransformer(
    [
        (
            "binned_numeric",
            KBinsDiscretizer(n_bins=10, random_state=0),
            ["VehAge", "DrivAge"],
        ),
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
        ),
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),
    ],
    remainder="drop",
)
# Apply the ColumnTransformer to `df` and assign the transformed data to `X`
X = column_trans.fit_transform(df)

# Calculate the Pure Premium as the ratio of `ClaimAmount` to `Exposure` for each policyholder
df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

# Approximate the Pure Premium indirectly by modeling Frequency and Average Claim Amount
df["Frequency"] = df["ClaimNb"] / df["Exposure"]
df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)

# Display a subset of `df` with non-zero `ClaimAmount` using Pandas option context
with pd.option_context("display.max_columns", 15):
    print(df[df.ClaimAmount > 0].head())

# %%
#
# Frequency model -- Poisson distribution
# ---------------------------------------
#
# Model `ClaimNb` (number of claims) using a Poisson distribution. Poisson
# distribution models discrete events occurring with a constant rate over a
# specified interval (`Exposure` in this case, measured in years).
# Here, `y = ClaimNb / Exposure` represents the scaled Poisson distribution,
# with `Exposure` used as `sample_weight`.
from sklearn.linear_model import PoissonRegressor
# 导入所需的库，从sklearn中导入train_test_split函数
from sklearn.model_selection import train_test_split

# 使用train_test_split函数将数据集df和特征X划分为训练集和测试集，并设定随机种子为0
df_train, df_test, X_train, X_test = train_test_split(df, X, random_state=0)

# %%
#
# 让我们记住，尽管这个数据集中似乎有大量数据点，
# 但其中索赔金额非零的评估点数量相当少：
len(df_test)

# %%
len(df_test[df_test["ClaimAmount"] > 0])

# %%
#
# 因此，在随机重新抽样训练测试集时，我们预计会得到显著的评估变异性。
#
# 通过牛顿解法在训练集上最小化泊松偏差来估计模型的参数。由于一些特征是共线的
# （例如因为我们在OneHotEncoder中没有删除任何分类级别），我们使用弱L2惩罚来避免数值问题。
glm_freq = PoissonRegressor(alpha=1e-4, solver="newton-cholesky")
glm_freq.fit(X_train, df_train["Frequency"], sample_weight=df_train["Exposure"])

# 使用score_estimator函数评估PoissonRegressor在训练集和测试集上的性能
scores = score_estimator(
    glm_freq,
    X_train,
    X_test,
    df_train,
    df_test,
    target="Frequency",
    weights="Exposure",
)
print("Evaluation of PoissonRegressor on target Frequency")
print(scores)

# %%
#
# 注意，测试集上的得分竟然比训练集上的好。这可能是由于此次随机的训练测试集划分所特有的。
# 适当的交叉验证可以帮助我们评估这些结果的抽样变异性。
#
# 我们可以通过视觉方式比较观察值和预测值，按驾驶员年龄（“DrivAge”）、
# 车辆年龄（“VehAge”）和保险奖惩（“BonusMalus”）进行聚合比较。

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.2)

# 在训练数据上绘制观察值和预测值的图像，按驾驶员年龄（“DrivAge”）聚合
plot_obs_pred(
    df=df_train,
    feature="DrivAge",
    weight="Exposure",
    observed="Frequency",
    predicted=glm_freq.predict(X_train),
    y_label="Claim Frequency",
    title="train data",
    ax=ax[0, 0],
)

# 在测试数据上绘制观察值和预测值的图像，按驾驶员年龄（“DrivAge”）聚合
plot_obs_pred(
    df=df_test,
    feature="DrivAge",
    weight="Exposure",
    observed="Frequency",
    predicted=glm_freq.predict(X_test),
    y_label="Claim Frequency",
    title="test data",
    ax=ax[0, 1],
    fill_legend=True,
)

# 在测试数据上绘制观察值和预测值的图像，按车辆年龄（“VehAge”）聚合
plot_obs_pred(
    df=df_test,
    feature="VehAge",
    weight="Exposure",
    observed="Frequency",
    predicted=glm_freq.predict(X_test),
    y_label="Claim Frequency",
    title="test data",
    ax=ax[1, 0],
    fill_legend=True,
)

# 在测试数据上绘制观察值和预测值的图像，按保险奖惩（“BonusMalus”）聚合
plot_obs_pred(
    df=df_test,
    feature="BonusMalus",
    weight="Exposure",
    observed="Frequency",
    predicted=glm_freq.predict(X_test),
    y_label="Claim Frequency",
    title="test data",
    ax=ax[1, 1],
    fill_legend=True,
)


# %%
# 根据观察到的数据，年龄低于30岁的驾驶员事故频率较高，并且与“BonusMalus”变量呈正相关。
# 我们的模型能够在很大程度上正确地模拟这种行为。
#
# 严重性模型 - Gamma分布
# ------------------------------------
# The mean claim amount or severity (`AvgClaimAmount`) can be empirically
# shown to follow approximately a Gamma distribution. We fit a GLM model for
# the severity with the same features as the frequency model.

# 导入Gamma回归器
from sklearn.linear_model import GammaRegressor

# 创建用于训练和测试集的掩码，排除ClaimAmount为0的记录
mask_train = df_train["ClaimAmount"] > 0
mask_test = df_test["ClaimAmount"] > 0

# 初始化Gamma回归器对象，设定alpha值和求解器
glm_sev = GammaRegressor(alpha=10.0, solver="newton-cholesky")

# 使用训练数据拟合Gamma回归模型，使用ClaimNb作为样本权重
glm_sev.fit(
    X_train[mask_train.values],  # 训练集特征
    df_train.loc[mask_train, "AvgClaimAmount"],  # 训练集目标变量
    sample_weight=df_train.loc[mask_train, "ClaimNb"],  # 样本权重
)

# 使用自定义函数评估模型性能
scores = score_estimator(
    glm_sev,  # 拟合的Gamma回归器模型
    X_train[mask_train.values],  # 训练集特征
    X_test[mask_test.values],  # 测试集特征
    df_train[mask_train],  # 训练集数据
    df_test[mask_test],  # 测试集数据
    target="AvgClaimAmount",  # 目标变量
    weights="ClaimNb",  # 样本权重
)
# 打印评估结果
print("Evaluation of GammaRegressor on target AvgClaimAmount")
print(scores)

# %%
#
# Those values of the metrics are not necessarily easy to interpret. It can be
# insightful to compare them with a model that does not use any input
# features and always predicts a constant value, i.e. the average claim
# amount, in the same setting:

# 导入虚拟回归器
from sklearn.dummy import DummyRegressor

# 初始化虚拟回归器，使用均值策略
dummy_sev = DummyRegressor(strategy="mean")

# 使用训练数据拟合虚拟回归模型，使用ClaimNb作为样本权重
dummy_sev.fit(
    X_train[mask_train.values],  # 训练集特征
    df_train.loc[mask_train, "AvgClaimAmount"],  # 训练集目标变量
    sample_weight=df_train.loc[mask_train, "ClaimNb"],  # 样本权重
)

# 使用自定义函数评估模型性能
scores = score_estimator(
    dummy_sev,  # 拟合的虚拟回归器模型
    X_train[mask_train.values],  # 训练集特征
    X_test[mask_test.values],  # 测试集特征
    df_train[mask_train],  # 训练集数据
    df_test[mask_test],  # 测试集数据
    target="AvgClaimAmount",  # 目标变量
    weights="ClaimNb",  # 样本权重
)
# 打印评估结果
print("Evaluation of a mean predictor on target AvgClaimAmount")
print(scores)

# %%
#
# We conclude that the claim amount is very challenging to predict. Still, the
# :class:`~sklearn.linear_model.GammaRegressor` is able to leverage some
# information from the input features to slightly improve upon the mean
# baseline in terms of D².
#
# Note that the resulting model is the average claim amount per claim. As such,
# it is conditional on having at least one claim, and cannot be used to predict
# the average claim amount per policy. For this, it needs to be combined with
# a claims frequency model.

# 打印平均索赔金额统计信息
print(
    "Mean AvgClaim Amount per policy:              %.2f "
    % df_train["AvgClaimAmount"].mean()
)
print(
    "Mean AvgClaim Amount | NbClaim > 0:           %.2f"
    % df_train["AvgClaimAmount"][df_train["AvgClaimAmount"] > 0].mean()
)
print(
    "Predicted Mean AvgClaim Amount | NbClaim > 0: %.2f"
    % glm_sev.predict(X_train).mean()
)
print(
    "Predicted Mean AvgClaim Amount (dummy) | NbClaim > 0: %.2f"
    % dummy_sev.predict(X_train).mean()
)

# %%
# We can visually compare observed and predicted values, aggregated for
# the drivers age (``DrivAge``).
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(16, 6))
# 创建一个包含两个子图的图表，每个子图大小为 16x6，垂直排列

plot_obs_pred(
    df=df_train.loc[mask_train],
    feature="DrivAge",
    weight="Exposure",
    observed="AvgClaimAmount",
    predicted=glm_sev.predict(X_train[mask_train.values]),
    y_label="Average Claim Severity",
    title="train data",
    ax=ax[0],
)
# 在第一个子图上绘制观测值和预测值的比较图表，显示训练数据的平均理赔严重性

plot_obs_pred(
    df=df_test.loc[mask_test],
    feature="DrivAge",
    weight="Exposure",
    observed="AvgClaimAmount",
    predicted=glm_sev.predict(X_test[mask_test.values]),
    y_label="Average Claim Severity",
    title="test data",
    ax=ax[1],
    fill_legend=True,
)
# 在第二个子图上绘制观测值和预测值的比较图表，显示测试数据的平均理赔严重性，并填充图例

plt.tight_layout()
# 调整子图布局，使它们更紧凑显示

# %%
# 总体而言，驾驶员年龄（``DrivAge``）对理赔严重性的影响较小，无论是在观察数据还是预测数据中。
#
# 通过产品模型与单个 TweedieRegressor 进行纯保费建模
# --------------------------------------------------------------------
# 如介绍中所述，每单位曝光的总理赔金额可以建模为频率模型预测与严重性模型预测的乘积。
#
# 或者，可以直接使用唯一的复合泊松伽马广义线性模型（带对数链接函数）来建模总损失。
# 此模型是 Tweedie GLM 的特例，其“power”参数为 :math:`p \in (1, 2)`。在这里，我们将 Tweedie
# 模型的 `power` 参数固定为某个有效范围内的任意值（1.9）。理想情况下，应通过网格搜索来选择此值，
# 方法是最小化 Tweedie 模型的负对数似然，但当前实现尚不支持（暂时）。
#
# 我们将比较两种方法的性能。
# 为了量化两个模型的性能，可以计算训练数据和测试数据的均值偏差，假设总理赔金额符合复合泊松-伽马分布。
# 这等同于 Tweedie 分布的 `power` 参数介于 1 和 2 之间。
#
# :func:`sklearn.metrics.mean_tweedie_deviance` 取决于 `power` 参数。
# 由于我们不知道 `power` 参数的真实值，因此我们在一系列可能的值上计算均值偏差，并将模型在相同 `power` 值下进行比较。
# 理想情况下，我们希望无论 `power` 的值如何，一个模型在性能上始终优于另一个模型。
from sklearn.linear_model import TweedieRegressor

glm_pure_premium = TweedieRegressor(power=1.9, alpha=0.1, solver="newton-cholesky")
# 创建 TweedieRegressor 模型对象，使用 `power` 参数为 1.9，`alpha` 参数为 0.1，使用牛顿-乔列斯基方法求解

glm_pure_premium.fit(
    X_train, df_train["PurePremium"], sample_weight=df_train["Exposure"]
)
# 使用训练数据拟合 TweedieRegressor 模型，目标变量为 "PurePremium"，样本权重为 "Exposure"

tweedie_powers = [1.5, 1.7, 1.8, 1.9, 1.99, 1.999, 1.9999]

scores_product_model = score_estimator(
    (glm_freq, glm_sev),
    X_train,
    X_test,
    df_train,
    df_test,
    target="PurePremium",
    weights="Exposure",
    tweedie_powers=tweedie_powers,
)
# 使用指定的 Tweedie 力度参数列表评估频率模型和严重性模型的组合模型在训练集和测试集上的表现

scores_glm_pure_premium = score_estimator(
    glm_pure_premium,
    X_train,
    X_test,
    df_train,
    df_test,
    target="PurePremium",
    weights="Exposure",
    tweedie_powers=tweedie_powers,
)
# 使用 TweedieRegressor 模型评估纯保费模型在训练集和测试集上的表现
    df_test,
    # df_test 是一个数据框，用于存储测试数据集
    target="PurePremium",
    # target 是一个字符串，表示我们模型中的目标变量是纯保费
    weights="Exposure",
    # weights 是一个字符串，表示在模型拟合时考虑的权重，这里是曝光度
    tweedie_powers=tweedie_powers,
    # tweedie_powers 是一个变量，可能是一个数组或列表，表示 Tweedie 模型中的幂次
)

# 将两个数据框按列进行连接，使用产品模型和 Tweedie 回归的分数
scores = pd.concat(
    [scores_product_model, scores_glm_pure_premium],
    axis=1,
    sort=True,
    keys=("Product Model", "TweedieRegressor"),
)

# 打印评估结果标题
print("Evaluation of the Product Model and the Tweedie Regressor on target PurePremium")

# 用上下文管理器设置显示选项，展示连接后的分数数据框
with pd.option_context("display.expand_frame_repr", False):
    print(scores)

# %%
# 在本例中，两种建模方法都表现出相似的性能指标。
# 由于实现原因，产品模型的解释方差百分比 :math:`D^2` 不可用。
#
# 我们还可以通过比较测试和训练子集上的观察和预测的总索赔金额来验证这些模型。
# 我们看到，总体上，两种模型都倾向于低估总索赔（但这种行为取决于正则化的程度）。

res = []
for subset_label, X, df in [
    ("train", X_train, df_train),
    ("test", X_test, df_test),
]:
    exposure = df["Exposure"].values
    res.append(
        {
            "subset": subset_label,
            "observed": df["ClaimAmount"].values.sum(),
            "predicted, frequency*severity model": np.sum(
                exposure * glm_freq.predict(X) * glm_sev.predict(X)
            ),
            "predicted, tweedie, power=%.2f"
            % glm_pure_premium.power: np.sum(exposure * glm_pure_premium.predict(X)),
        }
    )

# 打印结果数据框，以子集标签为索引
print(pd.DataFrame(res).set_index("subset").T)

# %%
#
# 最后，我们可以通过绘制累积索赔来比较两种模型：
# 对于每个模型，基于模型预测将保单持有人从最安全到最危险进行排名，并将观察到的总累积索赔比例绘制在 y 轴上。
# 这种绘图通常被称为模型的有序洛伦兹曲线。
#
# 基于曲线和对角线之间的面积，基尼系数可以用作模型选择指标，用于量化模型排名保单持有人的能力。
# 注意，该指标不反映模型在绝对总索赔金额方面做出准确预测的能力，而只是作为排名指标在相对金额方面的能力。
# 基尼系数的上限为 1.0，但即使是将保单持有人按观察到的索赔金额排序的理想模型也无法达到 1.0 的分数。
#
# 我们观察到，两种模型都能够相对于偶然性更好地排列保单持有人的风险程度，
# 尽管由于预测问题的自然困难，它们与理想模型也都有很大的差距：大多数事故是不可预测的，
# 可能由输入特征完全未描述的环境因素引起。
#
# 需要注意的是，基尼指数只表征模型的排名性能，而不表征其校准性：
# 预测的任何单调变换都不会改变模型的基尼指数。
#
# 最后，应强调的是，复合泊松伽马模型
# 导入必要的库函数 auc 用于计算曲线下面积（Area Under the Curve）
from sklearn.metrics import auc

# 定义绘制洛伦兹曲线的函数，输入真实值 y_true，预测值 y_pred，暴露度 exposure
def lorenz_curve(y_true, y_pred, exposure):
    # 将输入数据转换为 NumPy 数组
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # 按预测风险 y_pred 的大小对样本进行排序
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    
    # 计算累积索赔金额
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]  # 归一化处理，使最终值为 1
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))  # 生成等间隔的累积样本比例
    
    # 返回累积样本比例和累积索赔金额
    return cumulated_samples, cumulated_claim_amount

# 创建一个图形和轴对象
fig, ax = plt.subplots(figsize=(8, 8))

# 使用两种模型预测的产品 y_pred_product 和总纯保费 y_pred_total 进行洛伦兹曲线绘制
for label, y_pred in [
    ("Frequency * Severity model", y_pred_product),
    ("Compound Poisson Gamma", y_pred_total),
]:
    # 调用 lorenz_curve 函数计算洛伦兹曲线的 ordered_samples 和 cum_claims
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    # 计算 Gini 指数并添加到标签中
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += " (Gini index: {:.3f})".format(gini)
    # 绘制洛伦兹曲线
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle 模型：y_pred 等于 y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
# 计算 Oracle 模型的 Gini 指数
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = "Oracle (Gini index: {:.3f})".format(gini)
# 绘制 Oracle 模型的洛伦兹曲线
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# 随机基准线
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
# 设置图形标题和坐标轴标签
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
# 设置图例位置
ax.legend(loc="upper left")
# 显示图形
plt.plot()
```