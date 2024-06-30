# `D:\src\scipysrc\scikit-learn\examples\compose\plot_transformed_target.py`

```
"""
======================================================
Effect of transforming the targets in regression model
======================================================

In this example, we give an overview of
:class:`~sklearn.compose.TransformedTargetRegressor`. We use two examples
to illustrate the benefit of transforming the targets before learning a linear
regression model. The first example uses synthetic data while the second
example is based on the Ames housing data set.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 打印文档字符串 `__doc__`，该文档字符串包含了示例的概述信息
print(__doc__)

# %%
# Synthetic example
###################
#
# A synthetic random regression dataset is generated. The targets ``y`` are
# modified by:
#
#   1. translating all targets such that all entries are
#      non-negative (by adding the absolute value of the lowest ``y``) and
#   2. applying an exponential function to obtain non-linear
#      targets which cannot be fitted using a simple linear model.
#
# Therefore, a logarithmic (`np.log1p`) and an exponential function
# (`np.expm1`) will be used to transform the targets before training a linear
# regression model and using it for prediction.
import numpy as np

# 生成一个合成的随机回归数据集 `X` 和 `y`
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=10_000, noise=100, random_state=0)

# 将目标 `y` 进行变换：首先平移所有目标，使所有条目都是非负的（通过添加最小 `y` 的绝对值），然后应用指数函数，得到非线性的目标
y = np.expm1((y + abs(y.min())) / 200)
y_trans = np.log1p(y)

# %%
# Below we plot the probability density functions of the target
# before and after applying the logarithmic functions.
import matplotlib.pyplot as plt

# 导入 `train_test_split` 函数用于拆分数据集
from sklearn.model_selection import train_test_split

# 创建一个包含两个子图的图形对象 `f`
f, (ax0, ax1) = plt.subplots(1, 2)

# 在第一个子图 `ax0` 中绘制目标 `y` 的概率密度函数直方图
ax0.hist(y, bins=100, density=True)
ax0.set_xlim([0, 2000])
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
ax0.set_title("Target distribution")

# 在第二个子图 `ax1` 中绘制变换后的目标 `y_trans` 的概率密度函数直方图
ax1.hist(y_trans, bins=100, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
ax1.set_title("Transformed target distribution")

# 设置整个图形对象的标题
f.suptitle("Synthetic data", y=1.05)
plt.tight_layout()

# 使用 `train_test_split` 函数拆分数据集 `X` 和原始目标 `y` 为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
# At first, a linear model will be applied on the original targets. Due to the
# non-linearity, the model trained will not be precise during
# prediction. Subsequently, a logarithmic function is used to linearize the
# targets, allowing better prediction even with a similar linear model as
# reported by the median absolute error (MedAE).
from sklearn.metrics import median_absolute_error, r2_score

# 定义一个函数 `compute_score` 用于计算预测结果的评分，包括 R2 分数和中位绝对误差（MedAE）
def compute_score(y_true, y_pred):
    return {
        "R2": f"{r2_score(y_true, y_pred):.3f}",
        "MedAE": f"{median_absolute_error(y_true, y_pred):.3f}",
    }


# %%
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV

# 创建一个包含两个子图的图形对象 `f`，共享 y 轴
f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

# 使用 RidgeCV 拟合原始目标 `y_train`
ridge_cv = RidgeCV().fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)

# 创建一个 TransformedTargetRegressor 对象，用于将目标 `y_train` 进行对数变换
ridge_cv_with_trans_target = TransformedTargetRegressor(
    regressor = RidgeCV(), func=np.log1p, inverse_func=np.expm1


注释：


# 创建一个元组 (RidgeCV(), func=np.log1p, inverse_func=np.expm1)
# regressor: RidgeCV() 是一个线性回归模型的交叉验证版本
# func: np.log1p 是一个将数据进行 log(1 + x) 转换的函数
# inverse_func: np.expm1 是 np.log1p 的反函数，将经过 log1p 转换的数据还原回原始值
# 使用 RidgeCV 模型拟合训练数据 X_train 和 y_train
ridge_cv = RidgeCV().fit(X_train, y_train)

# 使用拟合好的 RidgeCV 模型对测试数据 X_test 进行预测，得到预测结果 y_pred_ridge
y_pred_ridge = ridge_cv.predict(X_test)

# 使用 PredictionErrorDisplay.from_predictions 方法绘制实际值 y_test 和预测值 y_pred_ridge 的散点图，kind="actual_vs_predicted" 表示实际值与预测值对比，ax=ax0 表示绘制在 ax0 轴上，scatter_kwargs={"alpha": 0.5} 设置散点图透明度为 0.5
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred_ridge,
    kind="actual_vs_predicted",
    ax=ax0,
    scatter_kwargs={"alpha": 0.5},
)

# 使用 PredictionErrorDisplay.from_predictions 方法绘制实际值 y_test 和经过目标变换后的预测值 y_pred_ridge_with_trans_target 的散点图，kind="actual_vs_predicted" 表示实际值与预测值对比，ax=ax1 表示绘制在 ax1 轴上，scatter_kwargs={"alpha": 0.5} 设置散点图透明度为 0.5
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred_ridge_with_trans_target,
    kind="actual_vs_predicted",
    ax=ax1,
    scatter_kwargs={"alpha": 0.5},
)

# 在每个轴上，添加计算得到的分数到图例中
for ax, y_pred in zip([ax0, ax1], [y_pred_ridge, y_pred_ridge_with_trans_target]):
    for name, score in compute_score(y_test, y_pred).items():
        # 绘制一个不可见的点，作为图例的标签，显示每个指标的得分
        ax.plot([], [], " ", label=f"{name}={score}")
    # 在左上角显示图例
    ax.legend(loc="upper left")

# 设置 ax0 的标题为 "Ridge regression \n without target transformation"
ax0.set_title("Ridge regression \n without target transformation")

# 设置 ax1 的标题为 "Ridge regression \n with target transformation"
ax1.set_title("Ridge regression \n with target transformation")

# 设置整体图的标题为 "Synthetic data"，并且将标题位置稍微向上偏移
f.suptitle("Synthetic data", y=1.05)

# 调整图的布局，使子图间的间距更加紧凑
plt.tight_layout()

# %%
# Real-world data set
#####################
#
# In a similar manner, the Ames housing data set is used to show the impact
# of transforming the targets before learning a model. In this example, the
# target to be predicted is the selling price of each house.

# 从 sklearn.datasets 导入 fetch_openml 方法
from sklearn.datasets import fetch_openml
# 从 sklearn.preprocessing 导入 quantile_transform 方法
from sklearn.preprocessing import quantile_transform

# 使用 fetch_openml 方法获取名为 "house_prices" 的数据集，并且将其作为 DataFrame 返回
ames = fetch_openml(name="house_prices", as_frame=True)

# 保留数据集中的数值类型列
X = ames.data.select_dtypes(np.number)

# 删除包含 NaN 或 Inf 值的列
X = X.drop(columns=["LotFrontage", "GarageYrBlt", "MasVnrArea"])

# 将目标值（房屋销售价格）除以 1000，将价格单位调整为千美元
y = ames.target / 1000

# 使用 QuantileTransformer 对 y 进行目标变换，使其分布接近正态分布，n_quantiles=900 表示使用 900 个分位数，output_distribution="normal" 表示输出分布为正态分布，copy=True 表示复制数据
y_trans = quantile_transform(
    y.to_frame(), n_quantiles=900, output_distribution="normal", copy=True
).squeeze()

# %%
# A :class:`~sklearn.preprocessing.QuantileTransformer` is used to normalize
# the target distribution before applying a
# :class:`~sklearn.linear_model.RidgeCV` model.

# 创建一个包含 1 行 2 列的子图布局，并且指定图的大小为 (6.5, 8)
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.5, 8))

# 在 ax0 上绘制 y 的直方图，bins=100 表示使用 100 个条形箱，density=True 表示绘制概率密度直方图
ax0.hist(y, bins=100, density=True)
# 设置 ax0 的 y 轴标签为 "Probability"
ax0.set_ylabel("Probability")
# 设置 ax0 的 x 轴标签为 "Target"
ax0.set_xlabel("Target")
# 设置 ax0 的标题为 "Target distribution"
ax0.set_title("Target distribution")

# 在 ax1 上绘制 y_trans 的直方图，bins=100 表示使用 100 个条形箱，density=True 表示绘制概率密度直方图
ax1.hist(y_trans, bins=100, density=True)
# 设置 ax1 的 y 轴标签为 "Probability"
ax1.set_ylabel("Probability")
# 设置 ax1 的 x 轴标签为 "Target"
ax1.set_xlabel("Target")
# 设置 ax1 的标题为 "Transformed target distribution"
ax1.set_title("Transformed target distribution")

# 设置整体图的标题为 "Ames housing data: selling price"，并且将标题位置稍微向上偏移
f.suptitle("Ames housing data: selling price", y=1.05)

# 调整图的布局，使子图间的间距更加紧凑
plt.tight_layout()

# %%
# 将数据集 X 和目标值 y 分割为训练集和测试集，使用随机种子 1 进行随机分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# %%
# The effect of the transformer is weaker than on the synthetic data. However,
# the transformation results in an increase in :math:`R^2` and large decrease
# of the MedAE. The residual plot (predicted target - true target vs predicted
# target) without target transformation takes on a curved, 'reverse smile'
# shape due to residual values that vary depending on the value of predicted
# target. With target transformation, the shape is more linear indicating
# better model fit.

# 从 sklearn.preprocessing 导入 QuantileTransformer 方法
from sklearn.preprocessing import QuantileTransformer

# 创建一个包含 2 行 2 列的子图布局，并且指定图的大小为 (6.5, 8)
f, (ax0, ax1) = plt.subplots(2, 2, sharey="row", figsize=(6.5, 8))

# 使用 RidgeCV 模型拟合训练数据 X_train 和 y_train，并得到预测结果 y_pred_ridge
ridge_cv = RidgeCV().fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)
# 使用 TransformedTargetRegressor 进行目标变换的岭回归模型训练
ridge_cv_with_trans_target = TransformedTargetRegressor(
    regressor=RidgeCV(),  # 使用 RidgeCV 作为基础回归器
    transformer=QuantileTransformer(n_quantiles=900, output_distribution="normal"),  # 使用分位数转换器进行目标变换
).fit(X_train, y_train)  # 使用训练数据 X_train 和 y_train 进行拟合

# 对测试集 X_test 进行预测
y_pred_ridge_with_trans_target = ridge_cv_with_trans_target.predict(X_test)

# 绘制实际值与预测值的图像
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred_ridge,
    kind="actual_vs_predicted",
    ax=ax0[0],  # 将图像显示在第一个子图 ax0[0] 上
    scatter_kwargs={"alpha": 0.5},  # 设置散点图的透明度为 0.5
)
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred_ridge_with_trans_target,
    kind="actual_vs_predicted",
    ax=ax0[1],  # 将图像显示在第二个子图 ax0[1] 上
    scatter_kwargs={"alpha": 0.5},  # 设置散点图的透明度为 0.5
)

# 在每个图像的图例中添加分数
for ax, y_pred in zip([ax0[0], ax0[1]], [y_pred_ridge, y_pred_ridge_with_trans_target]):
    for name, score in compute_score(y_test, y_pred).items():
        ax.plot([], [], " ", label=f"{name}={score}")  # 添加每个评分项及其分数到图例中
    ax.legend(loc="upper left")  # 将图例放置在左上角

# 设置第一个子图的标题
ax0[0].set_title("Ridge regression \n without target transformation")
# 设置第二个子图的标题
ax0[1].set_title("Ridge regression \n with target transformation")

# 绘制残差与预测值的图像
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred_ridge,
    kind="residual_vs_predicted",
    ax=ax1[0],  # 将图像显示在第一个子图 ax1[0] 上
    scatter_kwargs={"alpha": 0.5},  # 设置散点图的透明度为 0.5
)
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred_ridge_with_trans_target,
    kind="residual_vs_predicted",
    ax=ax1[1],  # 将图像显示在第二个子图 ax1[1] 上
    scatter_kwargs={"alpha": 0.5},  # 设置散点图的透明度为 0.5
)

# 设置第一个子图的标题
ax1[0].set_title("Ridge regression \n without target transformation")
# 设置第二个子图的标题
ax1[1].set_title("Ridge regression \n with target transformation")

# 设置整体图的标题
f.suptitle("Ames housing data: selling price", y=1.05)
# 调整布局，确保图像紧凑显示
plt.tight_layout()
# 展示所有绘制的图像
plt.show()
```