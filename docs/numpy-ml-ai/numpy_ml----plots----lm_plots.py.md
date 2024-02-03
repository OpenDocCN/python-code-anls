# `numpy-ml\numpy_ml\plots\lm_plots.py`

```py
# 禁用 flake8 的警告
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 从 sklearn.model_selection 模块中导入 train_test_split 函数
# 从 sklearn.datasets.samples_generator 模块中导入 make_blobs 函数
# 从 sklearn.linear_model 模块中导入 LogisticRegression 类，并使用 LogisticRegression_sk 别名
# 从 sklearn.datasets 模块中导入 make_regression 函数
# 从 sklearn.metrics 模块中导入 zero_one_loss 和 r2_score 函数
import from sklearn.model_selection import train_test_split
import from sklearn.datasets.samples_generator import make_blobs
import from sklearn.linear_model import LogisticRegression as LogisticRegression_sk
import from sklearn.datasets import make_regression
import from sklearn.metrics import zero_one_loss, r2_score

# 导入 matplotlib.pyplot 模块，并使用 plt 别名
import matplotlib.pyplot as plt

# 导入 seaborn 模块，并使用 sns 别名
import seaborn as sns

# 设置 seaborn 图表样式为白色
sns.set_style("white")
# 设置 seaborn 图表上下文为 paper，字体缩放比例为 0.5
sns.set_context("paper", font_scale=0.5)

# 从 numpy_ml.linear_models 模块中导入 RidgeRegression、LinearRegression、BayesianLinearRegressionKnownVariance、BayesianLinearRegressionUnknownVariance、LogisticRegression 类
from numpy_ml.linear_models import (
    RidgeRegression,
    LinearRegression,
    BayesianLinearRegressionKnownVariance,
    BayesianLinearRegressionUnknownVariance,
    LogisticRegression,
)

#######################################################################
#                           Data Generators                           #
#######################################################################

# 定义函数 random_binary_tensor，生成指定形状和稀疏度的随机二进制张量
def random_binary_tensor(shape, sparsity=0.5):
    # 生成随机数组，大于等于 (1 - sparsity) 的元素设为 1，其余设为 0
    X = (np.random.rand(*shape) >= (1 - sparsity)).astype(float)
    return X

# 定义函数 random_regression_problem，生成随机回归问题数据集
def random_regression_problem(n_ex, n_in, n_out, intercept=0, std=1, seed=0):
    # 生成回归问题数据集，包括输入 X、输出 y 和真实系数 coef
    X, y, coef = make_regression(
        n_samples=n_ex,
        n_features=n_in,
        n_targets=n_out,
        bias=intercept,
        noise=std,
        coef=True,
        random_state=seed,
    )
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test, coef

# 定义函数 random_classification_problem，生成随机分类问题数据集
def random_classification_problem(n_ex, n_classes, n_in, seed=0):
    # 生成分类问题数据集，包括输入 X 和标签 y
    X, y = make_blobs(
        n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=seed
    )
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test

#######################################################################
#                                Plots                                #
#######################################################################

# 绘制 logistic 回归的图形
def plot_logistic():
    # 设置随机种子
    np.random.seed(12345)

    # 创建一个包含 4x4 子图的图形
    fig, axes = plt.subplots(4, 4)
    # 遍历每个子图
    for i, ax in enumerate(axes.flatten()):
        # 设置输入特征数和样本数
        n_in = 1
        n_ex = 150
        # 生成随机分类问题的数据集
        X_train, y_train, X_test, y_test = random_classification_problem(
            n_ex, n_classes=2, n_in=n_in, seed=i
        )
        # 创建 LogisticRegression 对象
        LR = LogisticRegression(penalty="l2", gamma=0.2, fit_intercept=True)
        # 使用训练数据拟合模型
        LR.fit(X_train, y_train, lr=0.1, tol=1e-7, max_iter=1e7)
        # 预测测试数据
        y_pred = (LR.predict(X_test) >= 0.5) * 1.0
        # 计算损失
        loss = zero_one_loss(y_test, y_pred) * 100.0

        # 创建 LogisticRegression_sk 对象
        LR_sk = LogisticRegression_sk(
            penalty="l2", tol=0.0001, C=0.8, fit_intercept=True, random_state=i
        )
        # 使用训练数据拟合模型
        LR_sk.fit(X_train, y_train)
        # 预测测试数据
        y_pred_sk = (LR_sk.predict(X_test) >= 0.5) * 1.0
        # 计算损失
        loss_sk = zero_one_loss(y_test, y_pred_sk) * 100.0

        # 设置 x 轴的范围
        xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
        xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
        # 生成用于绘制的数据点
        X_plot = np.linspace(xmin, xmax, 100)
        y_plot = LR.predict(X_plot)
        y_plot_sk = LR_sk.predict_proba(X_plot.reshape(-1, 1))[:, 1]

        # 绘制散点图和曲线
        ax.scatter(X_test[y_pred == 0], y_test[y_pred == 0], alpha=0.5)
        ax.scatter(X_test[y_pred == 1], y_test[y_pred == 1], alpha=0.5)
        ax.plot(X_plot, y_plot, label="mine", alpha=0.75)
        ax.plot(X_plot, y_plot_sk, label="sklearn", alpha=0.75)
        ax.legend()
        ax.set_title("Loss mine: {:.2f} Loss sklearn: {:.2f}".format(loss, loss_sk))

        # 设置 x 轴和 y 轴的刻度标签为空
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    # 调整子图布局
    plt.tight_layout()
    # 保存图形为图片文件
    plt.savefig("plot_logistic.png", dpi=300)
    # 关闭所有图形
    plt.close("all")


# 绘制贝叶斯分类器的图形
def plot_bayes():
    # 设置随机种子
    np.random.seed(12345)
    # 设置输入特征数、输出特征数、样本数、标准差和截距
    n_in = 1
    n_out = 1
    n_ex = 20
    std = 15
    intercept = 10
    # 生成随机的回归问题数据集，包括训练集和测试集，以及系数
    X_train, y_train, X_test, y_test, coefs = random_regression_problem(
        n_ex, n_in, n_out, intercept=intercept, std=std, seed=0
    )

    # 添加一些异常值
    x1, x2 = X_train[0] + 0.5, X_train[6] - 0.3
    y1 = np.dot(x1, coefs) + intercept + 25
    y2 = np.dot(x2, coefs) + intercept - 31
    X_train = np.vstack([X_train, np.array([x1, x2])])
    y_train = np.hstack([y_train, [y1[0], y2[0]])

    # 使用线性回归模型拟合数据
    LR = LinearRegression(fit_intercept=True)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    loss = np.mean((y_test - y_pred) ** 2)

    # 使用岭回归模型拟合数据
    ridge = RidgeRegression(alpha=1, fit_intercept=True)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    loss_ridge = np.mean((y_test - y_pred) ** 2)

    # 使用已知方差的贝叶斯线性回归模型拟合数据
    LR_var = BayesianLinearRegressionKnownVariance(
        mu=np.c_[intercept, coefs][0], sigma=np.sqrt(std), V=None, fit_intercept=True,
    )
    LR_var.fit(X_train, y_train)
    y_pred_var = LR_var.predict(X_test)
    loss_var = np.mean((y_test - y_pred_var) ** 2)

    # 使用未知方差的贝叶斯线性回归模型拟合数据
    LR_novar = BayesianLinearRegressionUnknownVariance(
        alpha=1, beta=2, mu=np.c_[intercept, coefs][0], V=None, fit_intercept=True
    )
    LR_novar.fit(X_train, y_train)
    y_pred_novar = LR_novar.predict(X_test)
    loss_novar = np.mean((y_test - y_pred_novar) ** 2)

    # 计算绘图所需的数据范围
    xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
    xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
    X_plot = np.linspace(xmin, xmax, 100)
    y_plot = LR.predict(X_plot)
    y_plot_ridge = ridge.predict(X_plot)
    y_plot_var = LR_var.predict(X_plot)
    y_plot_novar = LR_novar.predict(X_plot)

    # 计算真实值
    y_true = [np.dot(x, coefs) + intercept for x in X_plot]
    # 创建包含4个子图的图形
    fig, axes = plt.subplots(1, 4)

    # 将子图展平
    axes = axes.flatten()
    # 在第一个子图中绘制散点图和拟合曲线
    axes[0].scatter(X_test, y_test)
    axes[0].plot(X_plot, y_plot, label="MLE")
    axes[0].plot(X_plot, y_true, label="True fn")
    axes[0].set_title("Linear Regression\nMLE Test MSE: {:.2f}".format(loss))
    axes[0].legend()
    # 在第一个子图中绘制 X_plot 和 y_plot 之间的区域，用 error 来填充

    # 在第二个子图中绘制测试数据的散点图
    axes[1].scatter(X_test, y_test)
    # 在第二个子图中绘制 Ridge 回归的预测结果
    axes[1].plot(X_plot, y_plot_ridge, label="MLE")
    # 在第二个子图中绘制真实函数的图像
    axes[1].plot(X_plot, y_true, label="True fn")
    # 设置第二个子图的标题
    axes[1].set_title(
        "Ridge Regression (alpha=1)\nMLE Test MSE: {:.2f}".format(loss_ridge)
    )
    # 添加图例
    axes[1].legend()

    # 在第三个子图中绘制 MAP 的预测结果
    axes[2].plot(X_plot, y_plot_var, label="MAP")
    # 获取后验分布的均值和协方差
    mu, cov = LR_var.posterior["b"].mean, LR_var.posterior["b"].cov
    # 对后验分布进行采样，并绘制采样结果
    for k in range(200):
        b_samp = np.random.multivariate_normal(mu, cov)
        y_samp = [np.dot(x, b_samp[1]) + b_samp[0] for x in X_plot]
        axes[2].plot(X_plot, y_samp, alpha=0.05)
    # 在第三个子图中绘制测试数据的散点图
    axes[2].scatter(X_test, y_test)
    # 在第三个子图中绘制真实函数的图像
    axes[2].plot(X_plot, y_true, label="True fn")
    # 添加图例
    axes[2].legend()
    # 设置第三个子图的标题
    axes[2].set_title(
        "Bayesian Regression (known variance)\nMAP Test MSE: {:.2f}".format(loss_var)
    )

    # 在第四个子图中绘制 MAP 的预测结果
    axes[3].plot(X_plot, y_plot_novar, label="MAP")
    # 获取后验分布的均值和协方差
    mu = LR_novar.posterior["b | sigma**2"].mean
    cov = LR_novar.posterior["b | sigma**2"].cov
    # 对后验分布进行采样，并绘制采样结果
    for k in range(200):
        b_samp = np.random.multivariate_normal(mu, cov)
        y_samp = [np.dot(x, b_samp[1]) + b_samp[0] for x in X_plot]
        axes[3].plot(X_plot, y_samp, alpha=0.05)
    # 在第四个子图中绘制测试数据的散点图
    axes[3].scatter(X_test, y_test)
    # 在第四个子图中绘制真实函数的图像
    axes[3].plot(X_plot, y_true, label="True fn")
    # 添加图例
    axes[3].legend()
    # 设置第四个子图的标题
    axes[3].set_title(
        "Bayesian Regression (unknown variance)\nMAP Test MSE: {:.2f}".format(
            loss_novar
        )
    )

    # 设置所有子图的 x 轴和 y 轴刻度标签为空
    for ax in axes:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    # 设置图形的尺寸
    fig.set_size_inches(7.5, 1.875)
    # 保存图形为文件
    plt.savefig("plot_bayes.png", dpi=300)
    # 关闭所有图形
    plt.close("all")
# 定义一个函数用于绘制回归图
def plot_regression():
    # 设置随机种子，以便结果可重现
    np.random.seed(12345)

    # 创建一个包含4行4列子图的图形对象
    fig, axes = plt.subplots(4, 4)
    # 调整子图之间的间距
    plt.tight_layout()
    # 将图形保存为名为"plot_regression.png"的PNG文件，设置分辨率为300dpi
    plt.savefig("plot_regression.png", dpi=300)
    # 关闭所有图形对象，释放资源
    plt.close("all")
```