# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgdocsvm_vs_ocsvm.py`

```
"""
====================================================================
One-Class SVM versus One-Class SVM using Stochastic Gradient Descent
====================================================================

This example shows how to approximate the solution of
:class:`sklearn.svm.OneClassSVM` in the case of an RBF kernel with
:class:`sklearn.linear_model.SGDOneClassSVM`, a Stochastic Gradient Descent
(SGD) version of the One-Class SVM. A kernel approximation is first used in
order to apply :class:`sklearn.linear_model.SGDOneClassSVM` which implements a
linear One-Class SVM using SGD.

Note that :class:`sklearn.linear_model.SGDOneClassSVM` scales linearly with
the number of samples whereas the complexity of a kernelized
:class:`sklearn.svm.OneClassSVM` is at best quadratic with respect to the
number of samples. It is not the purpose of this example to illustrate the
benefits of such an approximation in terms of computation time but rather to
show that we obtain similar results on a toy dataset.

"""  # noqa: E501

# %%
# Import necessary libraries for visualization and computation
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# Import specific modules from scikit-learn for SVM and preprocessing
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM

# Set font parameters for matplotlib plots
font = {"weight": "normal", "size": 15}
matplotlib.rc("font", **font)

# Set random seed for reproducibility
random_state = 42
rng = np.random.RandomState(random_state)

# Generate train data
X = 0.3 * rng.randn(500, 2)
X_train = np.r_[X + 2, X - 2]

# Generate some regular novel observations for testing
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# Generate some abnormal novel observations for outlier detection
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# OCSVM hyperparameters
nu = 0.05  # Parameter controlling the proportion of outliers
gamma = 2.0  # Kernel coefficient for 'rbf' kernel

# Fit the One-Class SVM on the training data
clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# Calculate number of errors in predictions
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# Fit the One-Class SVM using a kernel approximation and SGD
transform = Nystroem(gamma=gamma, random_state=random_state)
clf_sgd = SGDOneClassSVM(
    nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=1e-4
)
pipe_sgd = make_pipeline(transform, clf_sgd)
pipe_sgd.fit(X_train)
y_pred_train_sgd = pipe_sgd.predict(X_train)
y_pred_test_sgd = pipe_sgd.predict(X_test)
y_pred_outliers_sgd = pipe_sgd.predict(X_outliers)

# Calculate number of errors in SGD predictions
n_error_train_sgd = y_pred_train_sgd[y_pred_train_sgd == -1].size
n_error_test_sgd = y_pred_test_sgd[y_pred_test_sgd == -1].size
n_error_outliers_sgd = y_pred_outliers_sgd[y_pred_outliers_sgd == 1].size

# %%
# Import DecisionBoundaryDisplay for plotting the decision boundary
from sklearn.inspection import DecisionBoundaryDisplay

# Initialize the plot
_, ax = plt.subplots(figsize=(9, 6))

# Generate a grid of points for plotting decision boundaries
xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
# 创建一个包含所有数据点坐标的矩阵 X，用于绘制决策边界
X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)

# 使用 DecisionBoundaryDisplay 类从分类器 clf 绘制决策边界的填充轮廓图
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    cmap="PuBu",
)

# 使用 DecisionBoundaryDisplay 类从分类器 clf 绘制决策边界的轮廓图
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    plot_method="contour",
    ax=ax,
    linewidths=2,
    colors="darkred",
    levels=[0],
)

# 使用 DecisionBoundaryDisplay 类从分类器 clf 绘制决策边界的填充轮廓图，定制颜色和分界线
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    colors="palevioletred",
    levels=[0, clf.decision_function(X).max()],
)

# 设置散点图的大小为 20，并绘制训练数据点
s = 20
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")

# 绘制测试数据点的散点图
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")

# 绘制异常数据点的散点图
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")

# 设置图形的标题和坐标轴范围，包括训练错误率、测试错误率和异常数据错误率的标签
ax.set(
    title="One-Class SVM",
    xlim=(-4.5, 4.5),
    ylim=(-4.5, 4.5),
    xlabel=(
        f"error train: {n_error_train}/{X_train.shape[0]}; "
        f"errors novel regular: {n_error_test}/{X_test.shape[0]}; "
        f"errors novel abnormal: {n_error_outliers}/{X_outliers.shape[0]}"
    ),
)

# 创建图例，包括学习到的决策边界、训练数据、新的正常数据和新的异常数据
_ = ax.legend(
    [mlines.Line2D([], [], color="darkred", label="learned frontier"), b1, b2, c],
    [
        "learned frontier",
        "training observations",
        "new regular observations",
        "new abnormal observations",
    ],
    loc="upper left",
)

# %%
# 创建新的图形和坐标轴
_, ax = plt.subplots(figsize=(9, 6))

# 生成网格点坐标矩阵 xx 和 yy，并将其展平为一维数组后再合并为矩阵 X
xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)

# 使用 DecisionBoundaryDisplay 类从管道 pipe_sgd 绘制决策边界的填充轮廓图
DecisionBoundaryDisplay.from_estimator(
    pipe_sgd,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    cmap="PuBu",
)

# 使用 DecisionBoundaryDisplay 类从管道 pipe_sgd 绘制决策边界的轮廓图
DecisionBoundaryDisplay.from_estimator(
    pipe_sgd,
    X,
    response_method="decision_function",
    plot_method="contour",
    ax=ax,
    linewidths=2,
    colors="darkred",
    levels=[0],
)

# 使用 DecisionBoundaryDisplay 类从管道 pipe_sgd 绘制决策边界的填充轮廓图，定制颜色和分界线
DecisionBoundaryDisplay.from_estimator(
    pipe_sgd,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    colors="palevioletred",
    levels=[0, pipe_sgd.decision_function(X).max()],
)

# 设置散点图的大小为 20，并绘制训练数据点
s = 20
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")

# 绘制测试数据点的散点图
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")

# 绘制异常数据点的散点图
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")

# 设置图形的标题和坐标轴范围，包括训练错误率、测试错误率和异常数据错误率的标签
ax.set(
    title="Online One-Class SVM",
    xlim=(-4.5, 4.5),
    ylim=(-4.5, 4.5),
    xlabel=(
        f"error train: {n_error_train_sgd}/{X_train.shape[0]}; "
        f"errors novel regular: {n_error_test_sgd}/{X_test.shape[0]}; "
        f"errors novel abnormal: {n_error_outliers_sgd}/{X_outliers.shape[0]}"
    ),
)

# 创建图例，包括学习到的决策边界、训练数据、新的正常数据和新的异常数据
ax.legend(
    [mlines.Line2D([], [], color="darkred", label="learned frontier"), b1, b2, c],
    [
        "learned frontier",
        "training observations",
        "new regular observations",
        "new abnormal observations",
    ],
    loc="upper left",
)
    [
        "learned frontier",  # 标签列表中的第一个标签，表示学习的边界
        "training observations",  # 标签列表中的第二个标签，表示训练观察
        "new regular observations",  # 标签列表中的第三个标签，表示新的正常观察
        "new abnormal observations",  # 标签列表中的第四个标签，表示新的异常观察
    ],
    loc="upper left",
)
plt.show()
```