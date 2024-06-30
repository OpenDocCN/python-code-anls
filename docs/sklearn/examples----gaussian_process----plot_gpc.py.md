# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_gpc.py`

```
"""
====================================================================
Probabilistic predictions with Gaussian process classification (GPC)
====================================================================

This example illustrates the predicted probability of GPC for an RBF kernel
with different choices of the hyperparameters. The first figure shows the
predicted probability of GPC with arbitrarily chosen hyperparameters and with
the hyperparameters corresponding to the maximum log-marginal-likelihood (LML).

While the hyperparameters chosen by optimizing LML have a considerably larger
LML, they perform slightly worse according to the log-loss on test data. The
figure shows that this is because they exhibit a steep change of the class
probabilities at the class boundaries (which is good) but have predicted
probabilities close to 0.5 far away from the class boundaries (which is bad)
This undesirable effect is caused by the Laplace approximation used
internally by GPC.

The second figure shows the log-marginal-likelihood for different choices of
the kernel's hyperparameters, highlighting the two choices of the
hyperparameters used in the first figure by black dots.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np  # 导入 NumPy 库，用于数值计算
from matplotlib import pyplot as plt  # 导入 Matplotlib 库中的 pyplot 模块，用于绘图

from sklearn.gaussian_process import GaussianProcessClassifier  # 导入高斯过程分类器
from sklearn.gaussian_process.kernels import RBF  # 导入高斯过程分类器中的 RBF 核函数
from sklearn.metrics import accuracy_score, log_loss  # 导入评估指标：准确率和对数损失函数

# Generate data
train_size = 50  # 训练集大小为 50
rng = np.random.RandomState(0)  # 创建一个随机数种子
X = rng.uniform(0, 5, 100)[:, np.newaxis]  # 生成均匀分布的随机数据 X，reshape 成列向量
y = np.array(X[:, 0] > 2.5, dtype=int)  # 根据 X 的条件生成标签 y（大于 2.5 的为 1，否则为 0）

# Specify Gaussian Processes with fixed and optimized hyperparameters
gp_fix = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), optimizer=None)  # 创建一个固定超参数的高斯过程分类器对象
gp_fix.fit(X[:train_size], y[:train_size])  # 使用前 train_size 个样本进行拟合

gp_opt = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))  # 创建一个优化超参数的高斯过程分类器对象
gp_opt.fit(X[:train_size], y[:train_size])  # 使用前 train_size 个样本进行拟合

print(
    "Log Marginal Likelihood (initial): %.3f"
    % gp_fix.log_marginal_likelihood(gp_fix.kernel_.theta)
)
print(
    "Log Marginal Likelihood (optimized): %.3f"
    % gp_opt.log_marginal_likelihood(gp_opt.kernel_.theta)
)

print(
    "Accuracy: %.3f (initial) %.3f (optimized)"
    % (
        accuracy_score(y[:train_size], gp_fix.predict(X[:train_size])),
        accuracy_score(y[:train_size], gp_opt.predict(X[:train_size])),
    )
)
print(
    "Log-loss: %.3f (initial) %.3f (optimized)"
    % (
        log_loss(y[:train_size], gp_fix.predict_proba(X[:train_size])[:, 1]),
        log_loss(y[:train_size], gp_opt.predict_proba(X[:train_size])[:, 1]),
    )
)


# Plot posteriors
plt.figure()  # 创建一个新的图形窗口
plt.scatter(
    X[:train_size, 0], y[:train_size], c="k", label="Train data", edgecolors=(0, 0, 0)
)  # 绘制训练数据散点图
plt.scatter(
    X[train_size:, 0], y[train_size:], c="g", label="Test data", edgecolors=(0, 0, 0)
)  # 绘制测试数据散点图
X_ = np.linspace(0, 5, 100)  # 生成等间距的测试数据
plt.plot(
    X_,
    gp_fix.predict_proba(X_[:, np.newaxis])[:, 1],
    "r",
    # 绘制基于 gp_fix 模型的预测概率
    # 创建一个标签字符串，使用字符串格式化将 gp_fix.kernel_ 的值插入到字符串中
    label="Initial kernel: %s" % gp_fix.kernel_,
# 绘制第一个图：使用优化后的高斯过程模型预测的类别1的概率随特征变化的曲线图
plt.plot(
    X_,  # X 轴数据，特征值
    gp_opt.predict_proba(X_[:, np.newaxis])[:, 1],  # 预测的类别1的概率
    "b",  # 绘制蓝色线条
    label="Optimized kernel: %s" % gp_opt.kernel_  # 图例，显示优化后的核函数信息
)
plt.xlabel("Feature")  # X 轴标签，特征
plt.ylabel("Class 1 probability")  # Y 轴标签，类别1的概率
plt.xlim(0, 5)  # X 轴显示范围
plt.ylim(-0.25, 1.5)  # Y 轴显示范围
plt.legend(loc="best")  # 显示图例，位置最佳位置自动选择

# 绘制第二个图：显示高斯过程模型的对数边际似然在不同参数下的景观
plt.figure()
theta0 = np.logspace(0, 8, 30)  # 参数 theta0 取对数空间的值
theta1 = np.logspace(-1, 1, 29)  # 参数 theta1 取对数空间的值
Theta0, Theta1 = np.meshgrid(theta0, theta1)  # 创建参数空间网格
# 计算对数边际似然值的矩阵
LML = [
    [
        gp_opt.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])
    ]
    for j in range(Theta0.shape[1])
]
LML = np.array(LML).T  # 转置对数边际似然矩阵以便正确显示
plt.plot(
    np.exp(gp_fix.kernel_.theta)[0], np.exp(gp_fix.kernel_.theta)[1], "ko", zorder=10
)  # 绘制固定核函数参数的点
plt.plot(
    np.exp(gp_opt.kernel_.theta)[0], np.exp(gp_opt.kernel_.theta)[1], "ko", zorder=10
)  # 绘制优化后核函数参数的点
plt.pcolor(Theta0, Theta1, LML)  # 绘制对数边际似然的颜色填充图
plt.xscale("log")  # X 轴使用对数尺度
plt.yscale("log")  # Y 轴使用对数尺度
plt.colorbar()  # 添加颜色条
plt.xlabel("Magnitude")  # X 轴标签，参数 theta0 的幅度
plt.ylabel("Length-scale")  # Y 轴标签，参数 theta1 的长度尺度
plt.title("Log-marginal-likelihood")  # 图的标题，对数边际似然

plt.show()  # 显示图形
```