# `D:\src\scipysrc\scikit-learn\examples\svm\plot_svm_regression.py`

```
"""
===================================================================
Support Vector Regression (SVR) using linear and non-linear kernels
===================================================================

Toy example of 1D regression using linear, polynomial and RBF kernels.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

from sklearn.svm import SVR  # 从 scikit-learn 库中导入支持向量回归模型 SVR

# %%
# Generate sample data
# --------------------
X = np.sort(5 * np.random.rand(40, 1), axis=0)  # 生成随机的输入数据 X，按列排序
y = np.sin(X).ravel()  # 计算对应于 X 的正弦值作为目标输出 y，并将其展平为一维数组

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))  # 每隔5个数据点，将目标输出 y 加入噪声

# %%
# Fit regression model
# --------------------
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)  # 创建 RBF 内核的 SVR 模型
svr_lin = SVR(kernel="linear", C=100, gamma="auto")  # 创建线性内核的 SVR 模型
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)  # 创建多项式内核的 SVR 模型

# %%
# Look at the results
# -------------------
lw = 2  # 定义绘图线条的宽度

svrs = [svr_rbf, svr_lin, svr_poly]  # 将三种 SVR 模型放入列表中
kernel_label = ["RBF", "Linear", "Polynomial"]  # SVR 模型对应的内核标签
model_color = ["m", "c", "g"]  # 绘图时使用的颜色

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)  # 创建一个包含三个子图的图形布局
for ix, svr in enumerate(svrs):
    # 绘制每个 SVR 模型的预测结果曲线
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    # 标出每个 SVR 模型的支持向量
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="{} support vectors".format(kernel_label[ix]),
    )
    # 标出其他训练数据点
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
    # 添加图例
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "data", ha="center", va="center")  # 添加 x 轴标签
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")  # 添加 y 轴标签
fig.suptitle("Support Vector Regression", fontsize=14)  # 添加图形的总标题
plt.show()  # 显示绘制的图形
```