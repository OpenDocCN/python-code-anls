# `numpy-ml\numpy_ml\plots\nonparametric_plots.py`

```
# 禁用 flake8 检查
# 导入 numpy 库并重命名为 np
import numpy as np

# 导入 matplotlib.pyplot 库并重命名为 plt
import matplotlib.pyplot as plt
# 导入 seaborn 库并重命名为 sns

import seaborn as sns

# 设置 seaborn 图表样式为白色
sns.set_style("white")
# 设置 seaborn 上下文为 paper，字体缩放比例为 0.5
sns.set_context("paper", font_scale=0.5)

# 从 numpy_ml.nonparametric 模块导入 GPRegression、KNN、KernelRegression 类
# 从 numpy_ml.linear_models.lm 模块导入 LinearRegression 类
from numpy_ml.nonparametric import GPRegression, KNN, KernelRegression
from numpy_ml.linear_models.lm import LinearRegression

# 从 sklearn.model_selection 模块导入 train_test_split 函数

from sklearn.model_selection import train_test_split

# 定义函数 random_regression_problem，生成随机回归问题数据集
def random_regression_problem(n_ex, n_in, n_out, d=3, intercept=0, std=1, seed=0):
    # 生成随机系数
    coef = np.random.uniform(0, 50, size=d)
    coef[-1] = intercept

    y = []
    # 生成随机输入数据 X
    X = np.random.uniform(-100, 100, size=(n_ex, n_in))
    for x in X:
        # 计算输出值 y，并加入随机噪声
        val = np.polyval(coef, x) + np.random.normal(0, std)
        y.append(val)
    y = np.array(y)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test, coef

# 定义函数 plot_regression，绘制回归图表
def plot_regression():
    np.random.seed(12345)
    # 创建 4x4 的子图
    fig, axes = plt.subplots(4, 4)
    # 调整子图布局
    plt.tight_layout()
    # 保存图表为 kr_plots.png 文件，分辨率为 300 dpi
    plt.savefig("img/kr_plots.png", dpi=300)
    # 关闭所有图表
    plt.close("all")

# 定义函数 plot_knn，绘制 KNN 图表
def plot_knn():
    np.random.seed(12345)
    # 创建 4x4 的子图
    fig, axes = plt.subplots(4, 4)
    # 调整子图布局
    plt.tight_layout()
    # 保存图表为 knn_plots.png 文件，分辨率为 300 dpi
    plt.savefig("img/knn_plots.png", dpi=300)
    # 关闭所有图表
    plt.close("all")

# 定义函数 plot_gp，绘制高斯过程图表
def plot_gp():
    np.random.seed(12345)
    # 设置 seaborn 上下文为 paper，字体缩放比例为 0.65
    sns.set_context("paper", font_scale=0.65)

    # 生成测试数据 X_test
    X_test = np.linspace(-10, 10, 100)
    X_train = np.array([-3, 0, 7, 1, -9])
    y_train = np.sin(X_train)

    # 创建 2x2 的子图
    fig, axes = plt.subplots(2, 2)
    # 定义 alpha 值列表
    alphas = [0, 1e-10, 1e-5, 1]
    # 遍历 axes 中展平后的子图和 alphas 中的值，同时获取索引 ix 和元组 (ax, alpha)
    for ix, (ax, alpha) in enumerate(zip(axes.flatten(), alphas)):
        # 创建一个高斯过程回归对象 G，使用 RBF 核函数和给定的 alpha 值
        G = GPRegression(kernel="RBFKernel", alpha=alpha)
        # 使用训练数据 X_train 和 y_train 拟合 G 模型
        G.fit(X_train, y_train)
        # 对测试数据 X_test 进行预测，得到预测值 y_pred 和置信区间 conf
        y_pred, conf = G.predict(X_test)

        # 在当前子图 ax 上绘制训练数据点
        ax.plot(X_train, y_train, "rx", label="observed")
        # 在当前子图 ax 上绘制真实函数 np.sin(X_test) 的图像
        ax.plot(X_test, np.sin(X_test), label="true fn")
        # 在当前子图 ax 上绘制预测值 y_pred 的图像，并使用虚线表示
        ax.plot(X_test, y_pred, "--", label="MAP (alpha={})".format(alpha))
        # 在当前子图 ax 上填充置信区间
        ax.fill_between(X_test, y_pred + conf, y_pred - conf, alpha=0.1)
        # 设置 x 轴和 y 轴的刻度为空
        ax.set_xticks([])
        ax.set_yticks([])
        # 移除子图 ax 的上、右边框
        sns.despine()

        # 在子图 ax 上添加图例
        ax.legend()

    # 调整子图布局
    plt.tight_layout()
    # 将当前图保存为图片 gp_alpha.png，分辨率为 300 dpi
    plt.savefig("img/gp_alpha.png", dpi=300)
    # 关闭所有图形窗口
    plt.close("all")
# 定义一个函数用于绘制高斯过程的分布
def plot_gp_dist():
    # 设置随机种子
    np.random.seed(12345)
    # 设置 seaborn 图形的上下文和字体比例
    sns.set_context("paper", font_scale=0.95)

    # 生成测试数据
    X_test = np.linspace(-10, 10, 100)
    # 训练数据
    X_train = np.array([-3, 0, 7, 1, -9])
    y_train = np.sin(X_train)

    # 创建包含3个子图的画布
    fig, axes = plt.subplots(1, 3)
    # 创建高斯过程回归对象
    G = GPRegression(kernel="RBFKernel", alpha=0)
    # 使用训练数据拟合高斯过程
    G.fit(X_train, y_train)

    # 从先验分布中生成3个样本
    y_pred_prior = G.sample(X_test, 3, "prior")
    # 从后验预测分布中生成3个样本
    y_pred_posterior = G.sample(X_test, 3, "posterior_predictive")

    # 绘制先验样本
    for prior_sample in y_pred_prior:
        axes[0].plot(X_test, prior_sample.ravel(), lw=1)
    axes[0].set_title("Prior samples")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # 绘制后验样本
    for post_sample in y_pred_posterior:
        axes[1].plot(X_test, post_sample.ravel(), lw=1)
    axes[1].plot(X_train, y_train, "ko", ms=1.2)
    axes[1].set_title("Posterior samples")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # 预测测试数据的均值和置信区间
    y_pred, conf = G.predict(X_test)

    # 绘制真实函数、MAP 估计和观测数据
    axes[2].plot(X_test, np.sin(X_test), lw=1, label="true function")
    axes[2].plot(X_test, y_pred, lw=1, label="MAP estimate")
    axes[2].fill_between(X_test, y_pred + conf, y_pred - conf, alpha=0.1)
    axes[2].plot(X_train, y_train, "ko", ms=1.2, label="observed")
    axes[2].legend(fontsize="x-small")
    axes[2].set_title("Posterior mean")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # 设置画布大小
    fig.set_size_inches(6, 2)
    # 调整子图布局
    plt.tight_layout()
    # 保存图片
    plt.savefig("img/gp_dist.png", dpi=300)
    # 关闭所有图形
    plt.close("all")
```