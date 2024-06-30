# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_gradient_boosting_quantile.py`

```
"""
=====================================================
Prediction Intervals for Gradient Boosting Regression
=====================================================

This example shows how quantile regression can be used to create prediction
intervals. See :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py`
for an example showcasing some other features of
:class:`~ensemble.HistGradientBoostingRegressor`.

"""

# %%
# Generate some data for a synthetic regression problem by applying the
# function f to uniformly sampled random inputs.
import numpy as np

from sklearn.model_selection import train_test_split

# 定义用于预测的函数 f
def f(x):
    """The function to predict."""
    return x * np.sin(x)

# 使用随机种子 42 生成 1000 个均匀分布的输入样本 X
rng = np.random.RandomState(42)
X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
# 计算期望的输出值 expected_y，通过函数 f 计算
expected_y = f(X).ravel()

# %%
# To make the problem interesting, we generate observations of the target y as
# the sum of a deterministic term computed by the function f and a random noise
# term that follows a centered `log-normal
# <https://en.wikipedia.org/wiki/Log-normal_distribution>`_. To make this even
# more interesting we consider the case where the amplitude of the noise
# depends on the input variable x (heteroscedastic noise).
#
# The lognormal distribution is non-symmetric and long tailed: observing large
# outliers is likely but it is impossible to observe small outliers.
# 计算噪声的标准差 sigma，这里采用异方差噪声
sigma = 0.5 + X.ravel() / 10
# 生成符合对数正态分布的噪声样本 noise
noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
# 计算最终的观测值 y，为期望输出值 expected_y 加上噪声
y = expected_y + noise

# %%
# Split into train, test datasets:
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
# Fitting non-linear quantile and least squares regressors
# --------------------------------------------------------
#
# Fit gradient boosting models trained with the quantile loss and
# alpha=0.05, 0.5, 0.95.
#
# The models obtained for alpha=0.05 and alpha=0.95 produce a 90% confidence
# interval (95% - 5% = 90%).
#
# The model trained with alpha=0.5 produces a regression of the median: on
# average, there should be the same number of target observations above and
# below the predicted values.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error

# 定义共同的模型参数
common_params = dict(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=2,
    min_samples_leaf=9,
    min_samples_split=9,
)
# 循环创建不同 alpha 值下的梯度提升回归模型
all_models = {}
for alpha in [0.05, 0.5, 0.95]:
    # 使用 quantile 损失函数创建梯度提升回归器
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    # 将模型存入字典 all_models
    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)

# %%
# Notice that :class:`~sklearn.ensemble.HistGradientBoostingRegressor` is much
# faster than :class:`~sklearn.ensemble.GradientBoostingRegressor` starting with
# intermediate datasets (`n_samples >= 10_000`), which is not the case of the
# present example.
#
# For the sake of comparison, we also fit a baseline model trained with the
# usual (mean) squared error (MSE).
# %%
# 使用均方误差作为损失函数，创建一个梯度提升回归器，并将其存储在 all_models 字典中的 "mse" 键下
gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
all_models["mse"] = gbr_ls.fit(X_train, y_train)

# %%
# 创建一个包含 1000 个均匀间隔的输入值的评估集，范围在 [0, 10] 内
xx = np.atleast_2d(np.linspace(0, 10, 1000)).T

# %%
# 绘制真实条件均值函数 f、条件均值预测（损失为平方误差）、条件中位数以及条件90%区间（从第5到第95百分位数）
import matplotlib.pyplot as plt

y_pred = all_models["mse"].predict(xx)
y_lower = all_models["q 0.05"].predict(xx)
y_upper = all_models["q 0.95"].predict(xx)
y_med = all_models["q 0.50"].predict(xx)

fig = plt.figure(figsize=(10, 10))
plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(x)$")
plt.plot(X_test, y_test, "b.", markersize=10, label="Test observations")
plt.plot(xx, y_med, "r-", label="Predicted median")
plt.plot(xx, y_pred, "r-", label="Predicted mean")
plt.plot(xx, y_upper, "k-")
plt.plot(xx, y_lower, "k-")
plt.fill_between(
    xx.ravel(), y_lower, y_upper, alpha=0.4, label="Predicted 90% interval"
)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.ylim(-10, 25)
plt.legend(loc="upper left")
plt.show()

# %%
# 比较预测的中位数和预测的均值，注意到由于噪声向高值（大的异常值）偏斜，中位数平均低于均值。
# 中位数估计看起来也更加平滑，这是由于其对异常值的自然鲁棒性。
#
# 另外观察到，梯度提升树的归纳偏差不幸地阻止了我们的0.05分位数完全捕捉到信号的正弦形状，特别是在 x=8 附近。
# 调整超参数可以减少这种影响，如本笔记本的最后部分所示。
#
# 错误指标的分析
# -----------------------------
#
# 使用训练数据集上的 :func:`~sklearn.metrics.mean_squared_error` 和 :func:`~sklearn.metrics.mean_pinball_loss` 指标来衡量模型。
import pandas as pd

def highlight_min(x):
    x_min = x.min()
    return ["font-weight: bold" if v == x_min else "" for v in x]

results = []
for name, gbr in sorted(all_models.items()):
    metrics = {"model": name}
    y_pred = gbr.predict(X_train)
    for alpha in [0.05, 0.5, 0.95]:
        metrics["pbl=%1.2f" % alpha] = mean_pinball_loss(y_train, y_pred, alpha=alpha)
    metrics["MSE"] = mean_squared_error(y_train, y_pred)
    results.append(metrics)

# 创建并显示结果数据框，使用高亮显示最小值的单元格
pd.DataFrame(results).set_index("model").style.apply(highlight_min)

# %%
# 一列显示了所有模型使用相同指标评估的结果。如果训练收敛，那么在训练集上使用相同指标时，每列的最小值应该得到。
# 如果目标分布是不对称的，期望的条件均值和条件中位数会有显著差异，
# 创建一个空列表用于存储每个模型的评估结果
results = []

# 遍历所有模型，并按名称排序
for name, gbr in sorted(all_models.items()):
    metrics = {"model": name}  # 创建一个字典，存储当前模型名称
    y_pred = gbr.predict(X_test)  # 使用当前模型对测试集进行预测
    # 遍历不同的分位数水平，计算损失指标（分位损失）并存储
    for alpha in [0.05, 0.5, 0.95]:
        metrics["pbl=%1.2f" % alpha] = mean_pinball_loss(y_test, y_pred, alpha=alpha)
    # 计算均方误差（MSE）并存储
    metrics["MSE"] = mean_squared_error(y_test, y_pred)
    # 将当前模型的评估指标字典添加到结果列表中
    results.append(metrics)

# 创建包含结果的数据框，并以模型名称作为索引，并应用样式以突出显示最小值
pd.DataFrame(results).set_index("model").style.apply(highlight_min)
# %%
# 导入必要的库和模块
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from pprint import pprint

# 定义超参数网格
param_grid = dict(
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
)
alpha = 0.05

# 定义评分器，用于衡量负的平均分位损失函数，针对 alpha=0.05
neg_mean_pinball_loss_05p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,  # 最大化负损失
)

# 创建基于梯度提升回归的模型对象，配置为进行分位数回归
gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=0)

# 使用 HalvingRandomSearchCV 进行超参数搜索
search_05p = HalvingRandomSearchCV(
    gbr,
    param_grid,
    resource="n_estimators",
    max_resources=250,
    min_resources=50,
    scoring=neg_mean_pinball_loss_05p_scorer,
    n_jobs=2,
    random_state=0,
).fit(X_train, y_train)

# 打印出最佳的超参数组合
pprint(search_05p.best_params_)

# %%
# 我们观察到为中位数回归器手动调整的超参数与适合第5百分位数回归器的超参数处于相同范围内。
#
# 现在让我们调整适合第95百分位数回归器的超参数。我们需要重新定义用于选择最佳模型的 `scoring` 指标，
# 同时调整内部梯度提升估计器本身的 alpha 参数：
from sklearn.base import clone

# 更新 alpha 参数为 0.95
alpha = 0.95

# 定义新的评分器，用于衡量负的平均分位损失函数，针对 alpha=0.95
neg_mean_pinball_loss_95p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,  # 最大化负损失
)

# 克隆之前的搜索对象，更新适合第95百分位数回归器的配置
search_95p = clone(search_05p).set_params(
    estimator__alpha=alpha,
    scoring=neg_mean_pinball_loss_95p_scorer,
)

# 使用更新后的配置进行训练和搜索
search_95p.fit(X_train, y_train)

# 打印出适合第95百分位数回归器的最佳超参数组合
pprint(search_95p.best_params_)

# %%
# 结果显示，搜索过程确定的适合第95百分位数回归器的超参数与中位数回归器的手动调整超参数
# 以及适合第5百分位数回归器的搜索确定超参数大致处于相同范围。然而，超参数搜索确实
# 导致了一个改进的90%置信区间，该置信区间由这两个调整过的分位数回归器的预测组成。
# 注意，由于异常值的存在，上95百分位数的预测形状比下5百分位数的预测形状更加粗糙：
y_lower = search_05p.predict(xx)  # 预测下5百分位数
y_upper = search_95p.predict(xx)  # 预测上95百分位数

# 创建图表对象，并绘制相关图形
fig = plt.figure(figsize=(10, 10))
plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(x)$")
plt.plot(X_test, y_test, "b.", markersize=10, label="测试观测数据")
plt.plot(xx, y_upper, "k-")
plt.plot(xx, y_lower, "k-")
plt.fill_between(
    xx.ravel(), y_lower, y_upper, alpha=0.4, label="预测的90%置信区间"
)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.ylim(-10, 25)
plt.legend(loc="upper left")
plt.title("使用调整后的超参数进行预测")
plt.show()

# %%
# 在视觉上，这个图比未调优模型的效果要好，特别是在较低分位数的形状方面。

# 现在我们定量评估这对估计器的联合校准度：
coverage_fraction(y_train, search_05p.predict(X_train), search_95p.predict(X_train))
# 计算训练集上的覆盖率，使用搜索得到的5%分位数和95%分位数的预测结果。

# %%
coverage_fraction(y_test, search_05p.predict(X_test), search_95p.predict(X_test))
# 计算测试集上的覆盖率，使用搜索得到的5%分位数和95%分位数的预测结果。

# %%
# 遗憾的是，调优后的这对模型在测试集上的校准仍然不足够好：估计的置信区间宽度仍然太窄。

# 再次强调，我们需要将这项研究嵌入交叉验证循环中，以更好地评估这些估计值的变异性。
```