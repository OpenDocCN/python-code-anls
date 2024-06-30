# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgd_early_stopping.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库
import sys  # 导入系统相关的库
import time  # 导入时间相关的库

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据处理库

from sklearn import linear_model  # 导入线性模型库
from sklearn.datasets import fetch_openml  # 从OpenML下载数据集的函数
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告类
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.utils import shuffle  # 导入数据洗牌函数
from sklearn.utils._testing import ignore_warnings  # 导入忽略警告的函数

# 定义函数：加载MNIST数据集，选择两个特定的类别进行二元分类，并洗牌后返回指定数量的样本
def load_mnist(n_samples=None, class_0="0", class_1="8"):
    """Load MNIST, select two classes, shuffle and return only n_samples."""
    # 从OpenML下载MNIST数据集
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    # 选择指定的两个类别用于二元分类
    mask = np.logical_or(mnist.target == class_0, mnist.target == class_1)

    # 根据掩码选择数据集中的样本和标签，并进行洗牌，使用固定的随机种子
    X, y = shuffle(mnist.data[mask], mnist.target[mask], random_state=42)
    # 如果 n_samples 不为 None，则对 X 和 y 进行切片，保留前 n_samples 个样本
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    # 返回切片后的 X 和 y
    return X, y
@ignore_warnings(category=ConvergenceWarning)
# 使用装饰器忽略收敛警告

def fit_and_score(estimator, max_iter, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    # 设置估算器的最大迭代次数参数
    estimator.set_params(max_iter=max_iter)
    # 设置估算器的随机数种子参数为0
    estimator.set_params(random_state=0)

    # 记录开始时间
    start = time.time()
    # 在训练集上拟合估算器
    estimator.fit(X_train, y_train)

    # 计算拟合时间
    fit_time = time.time() - start
    # 获取迭代次数
    n_iter = estimator.n_iter_
    # 计算在训练集上的评分
    train_score = estimator.score(X_train, y_train)
    # 计算在测试集上的评分
    test_score = estimator.score(X_test, y_test)

    return fit_time, n_iter, train_score, test_score


# 定义要比较的估算器
estimator_dict = {
    "No stopping criterion": linear_model.SGDClassifier(n_iter_no_change=3),
    "Training loss": linear_model.SGDClassifier(
        early_stopping=False, n_iter_no_change=3, tol=0.1
    ),
    "Validation score": linear_model.SGDClassifier(
        early_stopping=True, n_iter_no_change=3, tol=0.0001, validation_fraction=0.2
    ),
}

# 加载数据集
X, y = load_mnist(n_samples=10000)
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 存储结果的列表
results = []
# 对每个估算器进行迭代
for estimator_name, estimator in estimator_dict.items():
    print(estimator_name + ": ", end="")
    # 对不同的最大迭代次数进行迭代
    for max_iter in range(1, 50):
        print(".", end="")
        sys.stdout.flush()

        # 执行拟合和评分，并获取相应的结果
        fit_time, n_iter, train_score, test_score = fit_and_score(
            estimator, max_iter, X_train, X_test, y_train, y_test
        )

        # 将结果添加到结果列表中
        results.append(
            (estimator_name, max_iter, fit_time, n_iter, train_score, test_score)
        )
    print("")

# 转换结果为 pandas 的 DataFrame 以便于绘图
columns = [
    "Stopping criterion",
    "max_iter",
    "Fit time (sec)",
    "n_iter_",
    "Train score",
    "Test score",
]
results_df = pd.DataFrame(results, columns=columns)

# 定义绘图的参数
lines = "Stopping criterion"
x_axis = "max_iter"
styles = ["-.", "--", "-"]

# 第一幅图：训练集和测试集得分
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 4))
for ax, y_axis in zip(axes, ["Train score", "Test score"]):
    for style, (criterion, group_df) in zip(styles, results_df.groupby(lines)):
        group_df.plot(x=x_axis, y=y_axis, label=criterion, ax=ax, style=style)
    ax.set_title(y_axis)
    ax.legend(title=lines)
fig.tight_layout()

# 第二幅图：迭代次数和拟合时间
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
for ax, y_axis in zip(axes, ["n_iter_", "Fit time (sec)"]):
    for style, (criterion, group_df) in zip(styles, results_df.groupby(lines)):
        group_df.plot(x=x_axis, y=y_axis, label=criterion, ax=ax, style=style)
    ax.set_title(y_axis)
    ax.legend(title=lines)
fig.tight_layout()

# 展示绘制的图形
plt.show()
```