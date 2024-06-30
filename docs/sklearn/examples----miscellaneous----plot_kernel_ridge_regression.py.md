# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_kernel_ridge_regression.py`

```
# %%
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# 作者信息和许可证声明

# %%
# Generate sample data
# --------------------
# 导入NumPy库
import numpy as np

# 创建随机数生成器
rng = np.random.RandomState(42)

# 生成输入数据X，服从均匀分布
X = 5 * rng.rand(10000, 1)

# 生成目标数据y，为sin(X)的展平结果
y = np.sin(X).ravel()

# 给每五个数据点的目标值加入噪声
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

# 生成用于绘制的输入数据X_plot，从0到5等间距取100000个点
X_plot = np.linspace(0, 5, 100000)[:, None]

# %%
# Construct the kernel-based regression models
# --------------------------------------------

# 导入所需的模块
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# 定义训练集大小
train_size = 100

# 使用GridSearchCV寻找最佳SVR模型
svr = GridSearchCV(
    SVR(kernel="rbf", gamma=0.1),
    param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
)

# 使用GridSearchCV寻找最佳Kernel Ridge模型
kr = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),
    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
)

# %%
# Compare times of SVR and Kernel Ridge Regression
# ------------------------------------------------

# 导入时间模块
import time

# 训练并计时SVR模型
t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print(f"Best SVR with params: {svr.best_params_} and R2 score: {svr.best_score_:.3f}")
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

# 训练并计时Kernel Ridge模型
t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0
print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

# 计算SVR支持向量的比例
sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

# 预测并计时SVR模型的输出
t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" % (X_plot.shape[0], svr_predict))

# 预测并计时Kernel Ridge模型的输出
t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))

# %%
# Look at the results
# -------------------

# 导入绘图库
import matplotlib.pyplot as plt

# 获取SVR支持向量的索引
sv_ind = svr.best_estimator_.support_

# 绘制散点图，标记SVR支持向量
plt.scatter(
    X[sv_ind],
    y[sv_ind],
    c="r",
    s=50,
    label="SVR support vectors",
    zorder=2,
    # 设置散点图中边缘颜色为黑色，RGB颜色值为(0, 0, 0)
    edgecolors=(0, 0, 0),
)
plt.scatter(X[:100], y[:100], c="k", label="data", zorder=1, edgecolors=(0, 0, 0))
plt.plot(
    X_plot,
    y_svr,
    c="r",
    label="SVR (fit: %.3fs, predict: %.3fs)" % (svr_fit, svr_predict),
)
plt.plot(
    X_plot, y_kr, c="g", label="KRR (fit: %.3fs, predict: %.3fs)" % (kr_fit, kr_predict)
)
plt.xlabel("data")
plt.ylabel("target")
plt.title("SVR versus Kernel Ridge")
_ = plt.legend()


# %%
# 上图比较了使用网格搜索优化复杂度/正则化和RBF核带宽时，KRR和SVR的学习模型。
# 这些学习函数非常相似；然而，KRR的拟合速度大约比SVR快3-4倍（都使用了网格搜索）。
#
# 理论上，由于SVR仅使用大约1/3的训练数据点作为支持向量来学习稀疏模型，因此SVR在预测100000个目标值时可能会快大约三倍。
# 然而，在实践中，由于每个模型计算核函数的实现细节，这不一定成立，这可能使得KRR模型与或甚至比SVR模型更快，尽管它计算了更多的算术操作。
#
# Prediction of 100000 target values could be in theory approximately three
# times faster with SVR since it has learned a sparse model using only
# approximately 1/3 of the training datapoints as support vectors. However, in
# practice, this is not necessarily the case because of implementation details
# in the way the kernel function is computed for each model that can make the
# KRR model as fast or even faster despite computing more arithmetic
# operations.


# %%
# 可视化训练和预测时间
# ---------------------------------------

plt.figure()

sizes = np.logspace(1, 3.8, 7).astype(int)
for name, estimator in {
    "KRR": KernelRidge(kernel="rbf", alpha=0.01, gamma=10),
    "SVR": SVR(kernel="rbf", C=1e2, gamma=10),
}.items():
    train_time = []
    test_time = []
    for train_test_size in sizes:
        t0 = time.time()
        estimator.fit(X[:train_test_size], y[:train_test_size])
        train_time.append(time.time() - t0)

        t0 = time.time()
        estimator.predict(X_plot[:1000])
        test_time.append(time.time() - t0)

    plt.plot(
        sizes,
        train_time,
        "o-",
        color="r" if name == "SVR" else "g",
        label="%s (train)" % name,
    )
    plt.plot(
        sizes,
        test_time,
        "o--",
        color="r" if name == "SVR" else "g",
        label="%s (test)" % name,
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title("Execution Time")
_ = plt.legend(loc="best")


# %%
# 该图比较了不同训练集大小时，KRR和SVR拟合和预测所需的时间。
# 对于中等大小的训练集（少于几千个样本），KRR的拟合速度比SVR快；然而，对于更大的训练集，SVR的性能更好。
# 关于预测时间，由于学习了稀疏解决方案，SVR在所有训练集大小上都应该比KRR更快，然而，由于实现细节，实际情况并非总是如此。
# 注意，稀疏度及因此预测时间取决于SVR的参数epsilon和C。
#
# This figure compares the time for fitting and prediction of KRR and SVR for
# different sizes of the training set. Fitting KRR is faster than SVR for
# medium-sized training sets (less than a few thousand samples); however, for
# larger training sets SVR scales better. With regard to prediction time, SVR
# should be faster than KRR for all sizes of the training set because of the
# learned sparse solution, however this is not necessarily the case in practice
# because of implementation details. Note that the degree of sparsity and thus
# the prediction time depends on the parameters epsilon and C of the SVR.


# %%
# 可视化学习曲线
# -----------------------------
from sklearn.model_selection import LearningCurveDisplay
# 创建一个空的图形和轴对象，准备用于绘制学习曲线
_, ax = plt.subplots()

# 初始化支持向量回归器 SVR，使用 RBF 核函数，设置参数 C=10 和 gamma=0.1
svr = SVR(kernel="rbf", C=1e1, gamma=0.1)

# 初始化核岭回归器 KernelRidge，使用 RBF 核函数，设置参数 alpha=0.1 和 gamma=0.1
kr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)

# 定义共享的参数字典，包括训练数据 X 和标签 y 的前 100 个样本，训练大小从 0.1 到 1 变化，评分方式为负均方误差，
# 打开评分的负数值，评分名称为"Mean Squared Error"，评分类型为测试集，标准显示样式为 None，图形使用之前创建的 ax 对象
common_params = {
    "X": X[:100],
    "y": y[:100],
    "train_sizes": np.linspace(0.1, 1, 10),
    "scoring": "neg_mean_squared_error",
    "negate_score": True,
    "score_name": "Mean Squared Error",
    "score_type": "test",
    "std_display_style": None,
    "ax": ax,
}

# 使用 svr 模型创建学习曲线显示对象，并传入共享参数字典
LearningCurveDisplay.from_estimator(svr, **common_params)

# 使用 kr 模型创建学习曲线显示对象，并传入共享参数字典
LearningCurveDisplay.from_estimator(kr, **common_params)

# 设置图形标题为"Learning curves"
ax.set_title("Learning curves")

# 添加图例到轴对象，显示"SVR"和"KRR"对应的标签
ax.legend(handles=ax.get_legend_handles_labels()[0], labels=["SVR", "KRR"])

# 显示图形
plt.show()
```