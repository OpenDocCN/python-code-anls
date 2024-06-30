# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_bias_variance.py`

```
"""
============================================================
Single estimator versus bagging: bias-variance decomposition
============================================================

This example illustrates and compares the bias-variance decomposition of the
expected mean squared error of a single estimator against a bagging ensemble.

In regression, the expected mean squared error of an estimator can be
decomposed in terms of bias, variance and noise. On average over datasets of
the regression problem, the bias term measures the average amount by which the
predictions of the estimator differ from the predictions of the best possible
estimator for the problem (i.e., the Bayes model). The variance term measures
the variability of the predictions of the estimator when fit over different
random instances of the same problem. Each problem instance is noted "LS", for
"Learning Sample", in the following. Finally, the noise measures the irreducible part
of the error which is due the variability in the data.

The upper left figure illustrates the predictions (in dark red) of a single
decision tree trained over a random dataset LS (the blue dots) of a toy 1d
regression problem. It also illustrates the predictions (in light red) of other
single decision trees trained over other (and different) randomly drawn
instances LS of the problem. Intuitively, the variance term here corresponds to
the width of the beam of predictions (in light red) of the individual
estimators. The larger the variance, the more sensitive are the predictions for
`x` to small changes in the training set. The bias term corresponds to the
difference between the average prediction of the estimator (in cyan) and the
best possible model (in dark blue). On this problem, we can thus observe that
the bias is quite low (both the cyan and the blue curves are close to each
other) while the variance is large (the red beam is rather wide).

The lower left figure plots the pointwise decomposition of the expected mean
squared error of a single decision tree. It confirms that the bias term (in
blue) is low while the variance is large (in green). It also illustrates the
noise part of the error which, as expected, appears to be constant and around
`0.01`.

The right figures correspond to the same plots but using instead a bagging
ensemble of decision trees. In both figures, we can observe that the bias term
is larger than in the previous case. In the upper right figure, the difference
between the average prediction (in cyan) and the best possible model is larger
(e.g., notice the offset around `x=2`). In the lower right figure, the bias
curve is also slightly higher than in the lower left figure. In terms of
variance however, the beam of predictions is narrower, which suggests that the
variance is lower. Indeed, as the lower right figure confirms, the variance
term (in green) is lower than for single decision trees. Overall, the bias-
"""

# 以上内容是一个示例的注释，描述了一个关于单个评估器和装袋集成的偏差-方差分解的例子。
# 代码段本身并没有实际代码，只是一个多行的字符串文本，用于文档说明和注释示例。
# 导入 matplotlib.pyplot 和 numpy 库
import matplotlib.pyplot as plt
import numpy as np

# 导入 BaggingRegressor 和 DecisionTreeRegressor 类
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# 设置参数
n_repeat = 50  # 计算期望的迭代次数
n_train = 50   # 训练集大小
n_test = 1000  # 测试集大小
noise = 0.1    # 噪声的标准差
np.random.seed(0)

# 更改此处来探索其他估计器的偏差-方差分解。
# 对于方差较高的估计器（例如决策树或KNN），这应该工作良好，
# 但对于方差较低的估计器（例如线性模型），效果较差。
estimators = [
    ("Tree", DecisionTreeRegressor()),  # 决策树估计器
    ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor())),  # Bagging 决策树估计器
]

n_estimators = len(estimators)

# 生成数据的函数
def f(x):
    x = x.ravel()
    return np.exp(-(x**2)) + 1.5 * np.exp(-((x - 2) ** 2))

# 生成数据集
def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))
        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y

# 初始化训练集和测试集
X_train = []
y_train = []

# 生成多次训练集
for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

# 生成测试集
X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

# 创建图形窗口
plt.figure(figsize=(10, 8))

# 遍历估计器以进行比较
for n, (name, estimator) in enumerate(estimators):
    # 计算预测值
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)

    # 计算均方误差的偏差^2 + 方差 + 噪声分解
    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2

    y_error /= n_repeat * n_repeat

    # 计算测试集的噪声方差
    y_noise = np.var(y_test, axis=1)
    # 计算偏差平方，即预测值与平均预测值的差异的平方
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    
    # 计算方差，即预测值的方差
    y_var = np.var(y_predict, axis=1)

    # 打印误差分解的详细信息，包括误差、偏差平方、方差和噪声的均值
    print(
        "{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
        " + {3:.4f} (var) + {4:.4f} (noise)".format(
            name, np.mean(y_error), np.mean(y_bias), np.mean(y_var), np.mean(y_noise)
        )
    )

    # 绘制第一个子图：真实函数、训练数据点、重复预测的所有实例、以及预测的期望值
    plt.subplot(2, n_estimators, n + 1)
    plt.plot(X_test, f(X_test), "b", label="$f(x)$")  # 绘制真实函数 f(x)
    plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")  # 绘制训练数据点

    for i in range(n_repeat):
        if i == 0:
            plt.plot(X_test, y_predict[:, i], "r", label=r"$\hat{y}(x)$")  # 绘制第一个重复预测的实例
        else:
            plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)  # 绘制其余重复预测的实例（透明度较低）

    plt.plot(X_test, np.mean(y_predict, axis=1), "c", label=r"$\mathbb{E}_{LS} \hat{y}(x)$")  # 绘制预测的期望值

    plt.xlim([-5, 5])  # 设置 x 轴范围
    plt.title(name)  # 设置子图标题

    if n == n_estimators - 1:
        plt.legend(loc=(1.1, 0.5))  # 如果是最后一个子图，则添加图例

    # 绘制第二个子图：误差、偏差平方、方差和噪声在不同 x 值下的变化
    plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.plot(X_test, y_error, "r", label="$error(x)$")  # 绘制误差
    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),  # 绘制偏差平方
    plt.plot(X_test, y_var, "g", label="$variance(x)$"),  # 绘制方差
    plt.plot(X_test, y_noise, "c", label="$noise(x)$")  # 绘制噪声

    plt.xlim([-5, 5])  # 设置 x 轴范围
    plt.ylim([0, 0.1])  # 设置 y 轴范围

    if n == n_estimators - 1:
        plt.legend(loc=(1.1, 0.5))  # 如果是最后一个子图，则添加图例
# 调整子图布局，右侧留出空间以防止标签被截断
plt.subplots_adjust(right=0.75)
# 显示绘图窗口
plt.show()
```