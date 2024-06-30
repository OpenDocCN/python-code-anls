# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_poisson_regression_non_normal_loss.py`

```
# 导入需要的库
import matplotlib.pyplot as plt  # 导入用于绘图的matplotlib库
import numpy as np  # 导入用于数值计算的numpy库
import pandas as pd  # 导入用于数据处理的pandas库

##############################################################################
# French Motor Third-Party Liability Claims 数据集
# -----------------------------------------------------
#
# 加载来自OpenML的汽车索赔数据集：
# https://www.openml.org/d/41214
from sklearn.datasets import fetch_openml  # 导入用于获取数据集的函数fetch_openml

# 从OpenML加载数据集，并将其转换为DataFrame格式
df = fetch_openml(data_id=41214, as_frame=True).frame
df

# %%
# 索赔数（``ClaimNb``）是一个正整数，可以建模为泊松分布。因此，它被假设为在给定时间间隔内以恒定速率发生的离散事件数（``Exposure``，以年为单位）。
#
# 我们希望通过（缩放后的）泊松分布条件地建模频率 ``y = ClaimNb / Exposure`` 在 ``X`` 上，并使用 ``Exposure`` 作为 ``sample_weight``。

# 计算频率并添加到数据框中
df["Frequency"] = df["ClaimNb"] / df["Exposure"]

# 打印平均频率
print(
    "Average Frequency = {}".format(np.average(df["Frequency"], weights=df["Exposure"]))
)

# 打印没有索赔的曝光时间所占的比例
print(
    "Fraction of exposure with zero claims = {0:.1%}".format(
        df.loc[df["ClaimNb"] == 0, "Exposure"].sum() / df["Exposure"].sum()
    )
)

# 创建一个包含三个子图的图形对象
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 4))

# 设置第一个子图的标题，并绘制索赔数的直方图（对数尺度）
ax0.set_title("Number of claims")
_ = df["ClaimNb"].hist(bins=30, log=True, ax=ax0)

# 设置第二个子图的标题，并绘制曝光时间的直方图（对数尺度）
ax1.set_title("Exposure in years")
_ = df["Exposure"].hist(bins=30, log=True, ax=ax1)
# 设置图表 ax2 的标题为 "Frequency (number of claims per year)"
ax2.set_title("Frequency (number of claims per year)")

# 使用 DataFrame df 的 "Frequency" 列进行直方图绘制，设置 bins=30（条柱数目），log=True（使用对数刻度），将图绘制在 ax2 上
_ = df["Frequency"].hist(bins=30, log=True, ax=ax2)

# %%
# 剩余的列可用于预测索赔事件的频率。
# 这些列非常异质，包含混合的分类变量和数值变量，它们具有不同的尺度，可能分布不均匀。
#
# 因此，为了使用这些预测变量拟合线性模型，需要执行标准的特征转换，如下所示：

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

# 创建一个管道，将对数变换（log）和标准缩放（StandardScaler）结合起来
log_scale_transformer = make_pipeline(
    FunctionTransformer(np.log, validate=False), StandardScaler()
)

# 创建 ColumnTransformer 对象，用于处理不同类型的特征变换
linear_model_preprocessor = ColumnTransformer(
    [
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),  # 不做变换的数值列
        (
            "binned_numeric",
            KBinsDiscretizer(n_bins=10, random_state=0),  # 对 "VehAge" 和 "DrivAge" 列进行分箱处理
            ["VehAge", "DrivAge"],
        ),
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),  # 对 "Density" 列进行对数变换和标准缩放
        (
            "onehot_categorical",
            OneHotEncoder(),  # 对 "VehBrand", "VehPower", "VehGas", "Region", "Area" 列进行独热编码
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
        ),
    ],
    remainder="drop",  # 处理剩余列时丢弃（drop）
)

# %%
# 常数预测基准线
# ------------------------------
#
# 值得注意的是，超过 93% 的保单持有人没有索赔。如果将这个问题转换为二元分类任务，
# 它将显著不平衡，甚至简单地预测均值的模型也可以达到 93% 的准确率。
#
# 为了评估所使用的指标的相关性，我们将考虑一个常数预测器作为基准线，它始终预测训练样本的频率均值。

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 将数据集 df 拆分为训练集和测试集
df_train, df_test = train_test_split(df, test_size=0.33, random_state=0)

# 创建 Pipeline，包括数据预处理和 DummyRegressor 作为回归器
dummy = Pipeline(
    [
        ("preprocessor", linear_model_preprocessor),  # 数据预处理步骤
        ("regressor", DummyRegressor(strategy="mean")),  # 使用均值策略的 DummyRegressor
    ]
).fit(df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"])


##############################################################################
# 计算常数预测基准线在三种不同回归指标下的性能：

from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)


def score_estimator(estimator, df_test):
    """在测试集上评估一个估算器的性能。"""
    y_pred = estimator.predict(df_test)

    # 输出均方误差（MSE）指标
    print(
        "MSE: %.3f"
        % mean_squared_error(
            df_test["Frequency"], y_pred, sample_weight=df_test["Exposure"]
        )
    )
    
    # 输出平均绝对误差（MAE）指标
    print(
        "MAE: %.3f"
        % mean_absolute_error(
            df_test["Frequency"], y_pred, sample_weight=df_test["Exposure"]
        )
    )
    # 忽略非正预测值，因为它们对泊松偏差无效。
    # 创建一个布尔掩码，标记出所有大于零的预测值
    mask = y_pred > 0
    # 如果存在非正预测值
    if (~mask).any():
        # 统计非正预测值的数量和总样本数
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        # 打印警告消息，说明哪些样本的预测值无效且在计算泊松偏差时被忽略
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )

    # 打印均值泊松偏差的结果，格式化输出
    print(
        "mean Poisson deviance: %.3f"
        % mean_poisson_deviance(
            df_test["Frequency"][mask],  # 使用掩码选择非零预测值的实际频次数据
            y_pred[mask],  # 使用掩码选择非零预测值
            sample_weight=df_test["Exposure"][mask],  # 使用掩码选择非零预测值的曝光数据
        )
    )
print("Constant mean frequency evaluation:")
score_estimator(dummy, df_test)
# 打印信息，评估常数均值频率模型在测试数据集上的表现

# %%
# (Generalized) linear models
# ---------------------------
#
# We start by modeling the target variable with the (l2 penalized) least
# squares linear regression model, more commonly known as Ridge regression. We
# use a low penalization `alpha`, as we expect such a linear model to under-fit
# on such a large dataset.

from sklearn.linear_model import Ridge

ridge_glm = Pipeline(
    [
        ("preprocessor", linear_model_preprocessor),  # 数据预处理器
        ("regressor", Ridge(alpha=1e-6)),  # 使用 Ridge 回归器，设置较低的 alpha 参数
    ]
).fit(df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"])

# %%
# The Poisson deviance cannot be computed on non-positive values predicted by
# the model. For models that do return a few non-positive predictions (e.g.
# :class:`~sklearn.linear_model.Ridge`) we ignore the corresponding samples,
# meaning that the obtained Poisson deviance is approximate. An alternative
# approach could be to use :class:`~sklearn.compose.TransformedTargetRegressor`
# meta-estimator to map ``y_pred`` to a strictly positive domain.

print("Ridge evaluation:")
score_estimator(ridge_glm, df_test)

# %%
# Next we fit the Poisson regressor on the target variable. We set the
# regularization strength ``alpha`` to approximately 1e-6 over number of
# samples (i.e. `1e-12`) in order to mimic the Ridge regressor whose L2 penalty
# term scales differently with the number of samples.
#
# Since the Poisson regressor internally models the log of the expected target
# value instead of the expected value directly (log vs identity link function),
# the relationship between X and y is not exactly linear anymore. Therefore the
# Poisson regressor is called a Generalized Linear Model (GLM) rather than a
# vanilla linear model as is the case for Ridge regression.

from sklearn.linear_model import PoissonRegressor

n_samples = df_train.shape[0]

poisson_glm = Pipeline(
    [
        ("preprocessor", linear_model_preprocessor),  # 数据预处理器
        ("regressor", PoissonRegressor(alpha=1e-12, solver="newton-cholesky")),  # 使用 Poisson 回归器，设置 alpha 参数
    ]
)
poisson_glm.fit(
    df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
)

print("PoissonRegressor evaluation:")
score_estimator(poisson_glm, df_test)

# %%
# Gradient Boosting Regression Trees for Poisson regression
# ---------------------------------------------------------
#
# Finally, we will consider a non-linear model, namely Gradient Boosting
# Regression Trees. Tree-based models do not require the categorical data to be
# one-hot encoded: instead, we can encode each category label with an arbitrary
# integer using :class:`~sklearn.preprocessing.OrdinalEncoder`. With this
# encoding, the trees will treat the categorical features as ordered features,
# which might not be always a desired behavior. However this effect is limited
# for deep enough trees which are able to recover the categorical nature of the
# features. The main advantage of the
# 导入需要的库和模块
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 定义一个数据预处理的Pipeline，包括对分类变量的Ordinal编码和对数值变量的保持不变处理
tree_preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            OrdinalEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"],  # 对这些列进行Ordinal编码
        ),
        ("numeric", "passthrough", ["VehAge", "DrivAge", "BonusMalus", "Density"]),  # 数值列保持不变
    ],
    remainder="drop",  # 忽略其余列
)

# 构建Pipeline，包括数据预处理和使用Poisson损失的梯度提升回归器
poisson_gbrt = Pipeline(
    [
        ("preprocessor", tree_preprocessor),  # 数据预处理步骤
        (
            "regressor",
            HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=128),  # 使用Poisson损失的梯度提升回归器
        ),
    ]
)

# 使用训练数据拟合模型，设置样本权重为df_train["Exposure"]
poisson_gbrt.fit(
    df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
)

# 输出模型评估结果的标题
print("Poisson Gradient Boosted Trees evaluation:")

# 调用评分函数评估模型在测试数据上的表现
score_estimator(poisson_gbrt, df_test)

# %%
# 与上面的Poisson GLM一样，梯度提升树模型最小化Poisson偏差。
# 但由于具有更高的预测能力，它可以达到更低的Poisson偏差值。
#
# 使用单一的训练/测试拆分来评估模型容易受到随机波动的影响。
# 如果计算资源允许，应验证交叉验证性能指标是否会得出类似的结论。
#
# 这些模型之间的定性差异也可以通过比较观察目标值的直方图与预测值的直方图来可视化：

# 创建子图，设置子图的布局和共享y轴
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6), sharey=True)
fig.subplots_adjust(bottom=0.2)
n_bins = 20

# 对训练和测试数据集分别进行处理
for row_idx, label, df in zip(range(2), ["train", "test"], [df_train, df_test]):
    # 绘制观察频率的直方图
    df["Frequency"].hist(bins=np.linspace(-1, 30, n_bins), ax=axes[row_idx, 0])

    axes[row_idx, 0].set_title("Data")
    axes[row_idx, 0].set_yscale("log")
    axes[row_idx, 0].set_xlabel("y (observed Frequency)")
    axes[row_idx, 0].set_ylim([1e1, 5e5])
    axes[row_idx, 0].set_ylabel(label + " samples")

    # 对于每个模型，预测目标变量并绘制预测频率的直方图
    for idx, model in enumerate([ridge_glm, poisson_glm, poisson_gbrt]):
        y_pred = model.predict(df)

        pd.Series(y_pred).hist(
            bins=np.linspace(-1, 4, n_bins), ax=axes[row_idx, idx + 1]
        )
        axes[row_idx, idx + 1].set(
            title=model[-1].__class__.__name__,
            yscale="log",
            xlabel="y_pred (predicted expected Frequency)",
        )

# 调整布局，确保子图紧凑显示
plt.tight_layout()

# %%
# 实验数据显示y具有长尾分布。在所有模型中，我们预测随机变量的期望频率，
# 因此我们必然会比观察到的实现值具有更少的极端值。
# Importing necessary function for generating evenly spaced slices of data.
from sklearn.utils import gen_even_slices

# 定义一个函数，用于按风险组比较预测值和观察值的平均频率
def _mean_frequency_by_risk_group(y_true, y_pred, sample_weight=None, n_bins=100):
    """Compare predictions and observations for bins ordered by y_pred.

    We order the samples by ``y_pred`` and split it in bins.
    In each bin the observed mean is compared with the predicted mean.

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred: array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    n_bins: int, optional
        Number of bins to use.

    Returns
    -------
    bin_centers: ndarray of shape (n_bins,)
        bin centers
    y_true_bin: ndarray of shape (n_bins,)
        average y_true for each bin
    y_pred_bin: ndarray of shape (n_bins,)
        average y_pred for each bin
    """
    # 根据 y_pred 的排序索引对样本进行排序
    idx_sort = np.argsort(y_pred)
    # 计算每个 bin 的中心点，等间隔划分
    bin_centers = np.arange(0, 1, 1 / n_bins) + 0.5 / n_bins
    # 初始化每个 bin 的 y_pred 平均值和 y_true 平均值
    y_pred_bin = np.zeros(n_bins)
    y_true_bin = np.zeros(n_bins)
    # 使用 enumerate 函数遍历 gen_even_slices 生成的迭代器，获取索引 n 和切片 sl
    for n, sl in enumerate(gen_even_slices(len(y_true), n_bins)):
        # 根据排序后的索引 idx_sort 对应的样本权重，获取当前切片 sl 的权重
        weights = sample_weight[idx_sort][sl]
        # 计算当前切片 sl 中 y_pred 的加权平均值，存入 y_pred_bin 中的第 n 个位置
        y_pred_bin[n] = np.average(y_pred[idx_sort][sl], weights=weights)
        # 计算当前切片 sl 中 y_true 的加权平均值，存入 y_true_bin 中的第 n 个位置
        y_true_bin[n] = np.average(y_true[idx_sort][sl], weights=weights)
    # 返回计算得到的 bin 中心点、加权平均后的 y_true_bin 和 y_pred_bin
    return bin_centers, y_true_bin, y_pred_bin
# 打印测试集中索赔次数的实际总和
print(f"Actual number of claims: {df_test['ClaimNb'].sum()}")

# 创建一个包含4个子图的画布，每个子图设置为2行2列，尺寸为12x8英寸
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
# 调整子图之间的水平间距
plt.subplots_adjust(wspace=0.3)

# 遍历四个子图及其对应的模型
for axi, model in zip(ax.ravel(), [ridge_glm, poisson_glm, poisson_gbrt, dummy]):
    # 使用当前模型对测试集进行预测
    y_pred = model.predict(df_test)
    # 获取测试集中的实际频率值
    y_true = df_test["Frequency"].values
    # 获取测试集中的暴露值
    exposure = df_test["Exposure"].values
    # 计算按风险组排序后的平均频率及其相应的真实值和预测值
    q, y_true_seg, y_pred_seg = _mean_frequency_by_risk_group(
        y_true, y_pred, sample_weight=exposure, n_bins=10
    )

    # 打印模型名称及其预测的索赔次数总和
    print(f"Predicted number of claims by {model[-1]}: {np.sum(y_pred * exposure):.1f}")

    # 在当前子图上绘制预测值和观测值的折线图
    axi.plot(q, y_pred_seg, marker="x", linestyle="--", label="predictions")
    axi.plot(q, y_true_seg, marker="o", linestyle="--", label="observations")
    axi.set_xlim(0, 1.0)
    axi.set_ylim(0, 0.5)
    axi.set(
        title=model[-1],
        xlabel="Fraction of samples sorted by y_pred",
        ylabel="Mean Frequency (y_pred)",
    )
    axi.legend()
plt.tight_layout()

# %%
# 虚拟回归模型预测常数频率。该模型未将所有样本的排名视为相同，但整体上是
# 良好校准的（用于估计整体人群的平均频率）。
#
# ``Ridge``回归模型可能会预测非常低的预期频率，这与数据不匹配。因此，
# 它可能会严重低估某些保单持有人的风险。
#
# ``PoissonRegressor``和``HistGradientBoostingRegressor``显示出更好的
# 预测和观测目标一致性，特别是对于低预测目标值。
#
# 所有预测的总和也确认了``Ridge``模型的校准问题：它低估了测试集中索赔
# 的总数超过3%，而其他三个模型可以大致恢复测试投资组合的总索赔数。
#
# 排名能力的评估
# -------------------------------
#
# 对于某些业务应用，我们关注模型在排列上最危险和最安全的投保人的能力，而
# 不考虑预测的绝对值。在这种情况下，模型评估会将问题视为排名问题而不是
# 回归问题。
#
# 从这个角度比较三个模型时，可以绘制测试样本按照模型预测的安全到危险的
# 顺序的累积索赔比例与累积暴露比例的曲线图。
#
# 这种图称为洛伦茨曲线，并可以通过基尼系数进行总结：

from sklearn.metrics import auc

# 定义洛伦茨曲线函数，计算并绘制模型的预测风险排序曲线
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # 按预测风险递增排序样本：
    ranking = np.argsort(y_pred)
    ranked_frequencies = y_true[ranking]
    ranked_exposure = exposure[ranking]
    # 计算累积索赔额度，即频率乘以暴露的累积和
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure)
    
    # 将累积索赔额度归一化，使最后一个元素为1
    cumulated_claims /= cumulated_claims[-1]
    
    # 计算累积暴露度
    cumulated_exposure = np.cumsum(ranked_exposure)
    
    # 将累积暴露度归一化，使最后一个元素为1
    cumulated_exposure /= cumulated_exposure[-1]
    
    # 返回归一化后的累积暴露度和累积索赔额度
    return cumulated_exposure, cumulated_claims
# 创建一个图形窗口和一个坐标轴对象，设置图形大小为 8x8 英寸
fig, ax = plt.subplots(figsize=(8, 8))

# 遍历模型列表，对每个模型进行以下操作
for model in [dummy, ridge_glm, poisson_glm, poisson_gbrt]:
    # 对测试数据集进行预测，得到预测值 y_pred
    y_pred = model.predict(df_test)
    # 计算洛伦兹曲线的累积曝光和累积索赔
    cum_exposure, cum_claims = lorenz_curve(
        df_test["Frequency"], y_pred, df_test["Exposure"]
    )
    # 计算基尼系数
    gini = 1 - 2 * auc(cum_exposure, cum_claims)
    # 根据模型名称和基尼系数生成标签
    label = "{} (Gini: {:.2f})".format(model[-1], gini)
    # 绘制洛伦兹曲线
    ax.plot(cum_exposure, cum_claims, linestyle="-", label=label)

# Oracle 模型的特殊情况：y_pred 等于 y_test
cum_exposure, cum_claims = lorenz_curve(
    df_test["Frequency"], df_test["Frequency"], df_test["Exposure"]
)
# 计算基尼系数
gini = 1 - 2 * auc(cum_exposure, cum_claims)
# 根据基尼系数生成标签
label = "Oracle (Gini: {:.2f})".format(gini)
# 绘制洛伦兹曲线，使用虚线和灰色表示
ax.plot(cum_exposure, cum_claims, linestyle="-.", color="gray", label=label)

# 绘制随机基线曲线
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
# 设置坐标轴标题和标签
ax.set(
    title="Lorenz curves by model",
    xlabel="Cumulative proportion of exposure (from safest to riskiest)",
    ylabel="Cumulative proportion of claims",
)
# 设置图例位置
ax.legend(loc="upper left")

# %%
# 与预期相符，虚拟回归器无法正确排列样本，因此在此图上表现最差。
#
# 基于树的模型在按风险对投保人排名方面显著优于两个线性模型。
#
# 三个模型均显著优于随机预测，但也远非完美预测。
#
# 这一点是预期的，因为问题的性质：事故的发生主要由未在数据集列中捕获的偶然因素主导，
# 可以被视为纯粹随机的因素。
#
# 线性模型假设输入变量之间没有交互作用，这可能导致欠拟合。插入多项式特征提取器
# (:func:`~sklearn.preprocessing.PolynomialFeatures`)确实可以提高它们的判别能力，
# 使基尼系数提高 2 个百分点。特别是它提高了模型识别前 5% 风险档案的能力。
#
# 主要要点
# --------------
#
# - 可以通过模型生成的能力来评估模型的性能，包括其生成的预测排名的准确性。
#
# - 可以通过绘制按预测风险分组的测试样本的平均观察值与平均预测值的均方差来评估模型的校准性。
#
# - 岭回归模型的最小二乘损失（以及隐式使用的恒等链接函数）似乎导致该模型校准不良。特别是它倾向于低估风险，甚至可能预测无效的负频率。
#
# - 使用具有对数链接的泊松损失可以纠正这些问题，并导致良好校准的线性模型。
#
# - 基尼系数反映了模型在排名预测能力方面的表现，而不考虑其绝对值，因此仅评估其排名能力。
#
# - 尽管在校准方面有所改进，但这两个线性模型的排名能力都较差。
# models are comparable and well below the ranking power of the Gradient
# Boosting Regression Trees.
#
# - The Poisson deviance computed as an evaluation metric reflects both the
#   calibration and the ranking power of the model. It also makes a linear
#   assumption on the ideal relationship between the expected value and the
#   variance of the response variable. For the sake of conciseness we did not
#   check whether this assumption holds.
#
# - Traditional regression metrics such as Mean Squared Error and Mean Absolute
#   Error are hard to meaningfully interpret on count values with many zeros.

plt.show()
```