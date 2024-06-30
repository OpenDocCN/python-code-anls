# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_ols_3d.py`

```
"""
`
"""
=========================================================
Sparsity Example: Fitting only features 1  and 2
=========================================================

Features 1 and 2 of the diabetes-dataset are fitted and
plotted below. It illustrates that although feature 2
has a strong coefficient on the full model, it does not
give us much regarding `y` when compared to just feature 1.
"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# %%
# First we load the diabetes dataset.

import numpy as np  # 导入 numpy 库，提供数组操作等功能

from sklearn import datasets  # 从 sklearn 库导入 datasets 模块

X, y = datasets.load_diabetes(return_X_y=True)  # 加载糖尿病数据集，返回特征和目标值
indices = (0, 1)  # 定义选择的特征索引元组

X_train = X[:-20, indices]  # 训练数据集中的特征数据，排除最后 20 个样本
X_test = X[-20:, indices]  # 测试数据集中的特征数据，取最后 20 个样本
y_train = y[:-20]  # 训练数据集中的目标值，排除最后 20 个样本
y_test = y[-20:]  # 测试数据集中的目标值，取最后 20 个样本

# %%
# Next we fit a linear regression model.

from sklearn import linear_model  # 从 sklearn 库导入线性回归模型模块

ols = linear_model.LinearRegression()  # 创建线性回归模型对象
_ = ols.fit(X_train, y_train)  # 使用训练数据拟合模型

# %%
# Finally we plot the figure from three different views.

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，绘制图形

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401  # 导入 mpl_toolkits.mplot3d，用于进行 3D 绘图，注释掉未使用

def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(fig_num, figsize=(4, 3))  # 创建图形对象，指定编号和大小
    plt.clf()  # 清空当前图形
    ax = fig.add_subplot(111, projection="3d", elev=elev, azim=azim)  # 添加 3D 坐标轴，指定视角

    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c="k", marker="+")  # 绘制训练数据点，黑色加号标记
    ax.plot_surface(
        np.array([[-0.1, -0.1], [0.15, 0.15]]),  # 绘制平面 X 坐标数据
        np.array([[-0.1, 0.15], [-0.1, 0.15]]),  # 绘制平面 Y 坐标数据
        clf.predict(  # 预测平面上的 Z 值
            np.array([[-0.1, -0.1, 0.15, 0.15], [-0.1, 0.15, -0.1, 0.15]]).T
        ).reshape((2, 2)),  # 预测结果并重塑为 2x2 矩阵
        alpha=0.5,  # 设置平面透明度
    )
    ax.set_xlabel("X_1")  # 设置 X 轴标签
    ax.set_ylabel("X_2")  # 设置 Y 轴标签
    ax.set_zlabel("Y")  # 设置 Z 轴标签
    ax.xaxis.set_ticklabels([])  # 隐藏 X 轴刻度标签
    ax.yaxis.set_ticklabels([])  # 隐藏 Y 轴刻度标签
    ax.zaxis.set_ticklabels([])  # 隐藏 Z 轴刻度标签

# Generate the three different figures from different views
elev = 43.5  # 设置视角的仰角
azim = -110  # 设置视角的方位角
plot_figs(1, elev, azim, X_train, ols)  # 绘制第一个图形，视角为 elev 和 azim

elev = -0.5  # 设置视角的仰角
azim = 0  # 设置视角的方位角
plot_figs(2, elev, azim, X_train, ols)  # 绘制第二个图形，视角为 elev 和 azim

elev = -0.5  # 设置视角的仰角
azim = 90  # 设置视角的方位角
plot_figs(3, elev, azim, X_train, ols)  # 绘制第三个图形，视角为 elev 和 azim

plt.show()  # 显示所有图形
```