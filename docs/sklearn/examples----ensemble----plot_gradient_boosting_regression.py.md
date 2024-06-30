# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_gradient_boosting_regression.py`

```
"""
============================
Gradient Boosting regression
============================

This example demonstrates Gradient Boosting to produce a predictive
model from an ensemble of weak predictive models. Gradient boosting can be used
for regression and classification problems. Here, we will train a model to
tackle a diabetes regression task. We will obtain the results from
:class:`~sklearn.ensemble.GradientBoostingRegressor` with least squares loss
and 500 regression trees of depth 4.

Note: For larger datasets (n_samples >= 10000), please refer to
:class:`~sklearn.ensemble.HistGradientBoostingRegressor`. See
:ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` for an example
showcasing some other advantages of
:class:`~ensemble.HistGradientBoostingRegressor`.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy

from sklearn import datasets, ensemble  # 导入数据集和集成学习模块
from sklearn.inspection import permutation_importance  # 导入排列重要性评估工具
from sklearn.metrics import mean_squared_error  # 导入均方误差指标
from sklearn.model_selection import train_test_split  # 导入数据集划分工具

# %%
# Load the data
# -------------------------------------
#
# First we need to load the data.

diabetes = datasets.load_diabetes()  # 加载糖尿病数据集
X, y = diabetes.data, diabetes.target  # 将数据集分为特征 X 和目标 y

# %%
# Data preprocessing
# -------------------------------------
#
# Next, we will split our dataset to use 90% for training and leave the rest
# for testing. We will also set the regression model parameters. You can play
# with these parameters to see how the results change.
#
# `n_estimators` : the number of boosting stages that will be performed.
# Later, we will plot deviance against boosting iterations.
#
# `max_depth` : limits the number of nodes in the tree.
# The best value depends on the interaction of the input variables.
#
# `min_samples_split` : the minimum number of samples required to split an
# internal node.
#
# `learning_rate` : how much the contribution of each tree will shrink.
#
# `loss` : loss function to optimize. The least squares function is  used in
# this case however, there are many other options (see
# :class:`~sklearn.ensemble.GradientBoostingRegressor` ).

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13  # 划分数据集，保留10%作为测试集，设置随机种子
)

params = {
    "n_estimators": 500,  # 提升树的数量
    "max_depth": 4,  # 树的最大深度
    "min_samples_split": 5,  # 分裂内部节点所需的最小样本数
    "learning_rate": 0.01,  # 学习率，控制每棵树的贡献缩减量
    "loss": "squared_error",  # 损失函数为平方误差
}

# %%
# Fit regression model
# --------------------
#
# Now we will initiate the gradient boosting regressors and fit it with our
# training data. Let's also look and the mean squared error on the test data.

reg = ensemble.GradientBoostingRegressor(**params)  # 初始化梯度提升回归器
reg.fit(X_train, y_train)  # 在训练集上拟合模型

mse = mean_squared_error(y_test, reg.predict(X_test))  # 计算测试集上的均方误差
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

# %%
# Plot training deviance
# ----------------------
#
# Finally, we will visualize the results. To do that we will first compute the
# test_score 数组用于存储每次迭代后预测模型在测试集上的均方误差
test_score = np.zeros((params["n_estimators"],), dtype=np.float64)

# 使用 enumerate 函数迭代 reg 模型在测试集 X_test 上的预测结果 y_pred
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    # 计算当前迭代下的均方误差，并存储到 test_score 数组中
    test_score[i] = mean_squared_error(y_test, y_pred)

# 创建一个新的图形窗口，并设置其大小为 6x6 英寸
fig = plt.figure(figsize=(6, 6))

# 在图形窗口中创建一个子图，位于第一行第一列
plt.subplot(1, 1, 1)

# 设置子图的标题为 "Deviance"
plt.title("Deviance")

# 绘制训练集上的 deviance 曲线，颜色为蓝色线条
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)

# 绘制测试集上的 deviance 曲线，颜色为红色线条
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    test_score,
    "r-",
    label="Test Set Deviance"
)

# 添加图例，并设置其位置在右上角
plt.legend(loc="upper right")

# 设置 x 轴的标签为 "Boosting Iterations"
plt.xlabel("Boosting Iterations")

# 设置 y 轴的标签为 "Deviance"
plt.ylabel("Deviance")

# 调整子图的布局，使得内容紧凑显示
fig.tight_layout()

# 显示绘制的图形
plt.show()
# %%

# 绘制特征重要性
# -----------------------
#
# .. warning::
#    注意，基于不纯度的特征重要性对于**高基数**特征（具有许多唯一值的特征）可能具有误导性。
#    作为替代方法，可以在一个预留的测试集上计算 ``reg`` 的排列重要性。详见 :ref:`permutation_importance` 获取更多详情。
#
# 对于本示例，基于不纯度和排列方法识别出相同的两个强预测特征，但顺序不同。第三个最具预测性的特征 "bp" 在两种方法中也是相同的。
# 其余特征预测性较弱，并且排列图的误差条显示它们与 0 重叠。

# 获取模型的特征重要性数组
feature_importance = reg.feature_importances_

# 对特征重要性数组进行排序，返回排序后的索引数组
sorted_idx = np.argsort(feature_importance)

# 生成一个位置数组，用于条形图的水平位置
pos = np.arange(sorted_idx.shape[0]) + 0.5

# 创建一个新的图形窗口，并设置其大小为 12x6 英寸
fig = plt.figure(figsize=(12, 6))

# 在图形窗口中创建一个子图，位于第一行第一列
plt.subplot(1, 2, 1)

# 绘制水平条形图，显示特征重要性，对齐方式为中心
plt.barh(pos, feature_importance[sorted_idx], align="center")

# 设置 y 轴刻度为特征名称，根据排序索引顺序显示
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])

# 设置子图的标题为 "Feature Importance (MDI)"
plt.title("Feature Importance (MDI)")

# 计算模型在测试集上的排列重要性
result = permutation_importance(
    reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

# 对排列重要性的平均值进行排序，返回排序后的索引数组
sorted_idx = result.importances_mean.argsort()

# 在图形窗口中创建一个子图，位于第一行第二列
plt.subplot(1, 2, 2)

# 绘制箱线图，显示排列重要性，水平方向，标签为特征名称，根据排序索引顺序显示
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(diabetes.feature_names)[sorted_idx],
)

# 设置子图的标题为 "Permutation Importance (test set)"
plt.title("Permutation Importance (test set)")

# 调整子图的布局，使得内容紧凑显示
fig.tight_layout()

# 显示绘制的图形
plt.show()
```