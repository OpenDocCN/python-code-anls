# `D:\src\scipysrc\scikit-learn\examples\svm\plot_oneclass.py`

```
"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.

"""

# %%
# 导入必要的库和模块
import numpy as np

from sklearn import svm

# 生成训练数据
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# 生成一些常规的新样本观测数据
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# 生成一些异常的新样本观测数据
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 拟合模型
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

# 预测训练数据的标签
y_pred_train = clf.predict(X_train)

# 预测测试数据的标签
y_pred_test = clf.predict(X_test)

# 预测异常数据的标签
y_pred_outliers = clf.predict(X_outliers)

# 计算训练集中预测为异常的数量
n_error_train = y_pred_train[y_pred_train == -1].size

# 计算测试集中预测为异常的数量
n_error_test = y_pred_test[y_pred_test == -1].size

# 计算异常集中预测为正常的数量
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# %%
# 导入绘图相关的库和模块
import matplotlib.font_manager
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

# 创建图表和子图
_, ax = plt.subplots()

# 为决策边界显示生成网格
xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
X_grid = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

# 使用 DecisionBoundaryDisplay 绘制边界，使用 contourf 方式填充等高线
DecisionBoundaryDisplay.from_estimator(
    clf,
    X_grid,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    cmap="PuBu",
)

# 再次使用 DecisionBoundaryDisplay 绘制边界，指定不同的参数和颜色
DecisionBoundaryDisplay.from_estimator(
    clf,
    X_grid,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    levels=[0, 10000],
    colors="palevioletred",
)

# 第三次使用 DecisionBoundaryDisplay 绘制边界，使用 contour 方式绘制边界线
DecisionBoundaryDisplay.from_estimator(
    clf,
    X_grid,
    response_method="decision_function",
    plot_method="contour",
    ax=ax,
    levels=[0],
    colors="darkred",
    linewidths=2,
)

# 设置散点图的大小
s = 40

# 绘制训练集观测数据的散点图，白色
b1 = ax.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")

# 绘制测试集观测数据的散点图，紫罗兰色
b2 = ax.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")

# 绘制异常集观测数据的散点图，金色
c = ax.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")

# 添加图例
plt.legend(
    [mlines.Line2D([], [], color="darkred"), b1, b2, c],
    [
        "learned frontier",
        "training observations",
        "new regular observations",
        "new abnormal observations",
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)

# 设置坐标轴标签和标题
ax.set(
    xlabel=(
        f"error train: {n_error_train}/200 ; errors novel regular: {n_error_test}/40 ;"
        f" errors novel abnormal: {n_error_outliers}/40"
    ),
    title="Novelty Detection",
    xlim=(-5, 5),
    ylim=(-5, 5),
)

# 显示图表
plt.show()
```