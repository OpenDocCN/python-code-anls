# `D:\src\scipysrc\scikit-learn\examples\neural_networks\plot_mlp_alpha.py`

```
"""
================================================
Varying regularization in Multi-layer Perceptron
================================================

A comparison of different values for regularization parameter 'alpha' on
synthetic datasets. The plot shows that different alphas yield different
decision functions.

Alpha is a parameter for regularization term, aka penalty term, that combats
overfitting by constraining the size of the weights. Increasing alpha may fix
high variance (a sign of overfitting) by encouraging smaller weights, resulting
in a decision boundary plot that appears with lesser curvatures.
Similarly, decreasing alpha may fix high bias (a sign of underfitting) by
encouraging larger weights, potentially resulting in a more complicated
decision boundary.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

h = 0.02  # 在网格中的步长大小

alphas = np.logspace(-1, 1, 5)  # 正则化参数alpha的取值范围

classifiers = []  # 存储不同alpha值下的分类器
names = []  # 存储每个分类器对应的名称
for alpha in alphas:
    classifiers.append(
        make_pipeline(
            StandardScaler(),  # 标准化数据
            MLPClassifier(  # 多层感知机分类器
                solver="lbfgs",
                alpha=alpha,  # 设置正则化参数alpha
                random_state=1,
                max_iter=2000,
                early_stopping=True,
                hidden_layer_sizes=[10, 10],  # 设置隐藏层大小
            ),
        )
    )
    names.append(f"alpha {alpha:.2f}")  # 每个分类器的名称，显示对应的alpha值

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0),  # 生成月亮型数据集
    make_circles(noise=0.2, factor=0.5, random_state=1),  # 生成环形数据集
    linearly_separable,  # 线性可分数据集
]

figure = plt.figure(figsize=(17, 9))  # 创建绘图窗口
i = 1
# 遍历数据集
for X, y in datasets:
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 绘制数据集的散点图
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # 绘制训练点
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # 绘制测试点
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # 清空当前子图的 x 轴刻度
    ax.set_xticks(())
    # 清空当前子图的 y 轴刻度
    ax.set_yticks(())
    # 自增计数器 i，用于确定子图位置
    i += 1

    # 遍历每个分类器
    for name, clf in zip(names, classifiers):
        # 在多行子图中创建当前分类器的子图，位置由 i 决定
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # 使用训练数据拟合分类器
        clf.fit(X_train, y_train)
        # 计算分类器在测试集上的准确率
        score = clf.score(X_test, y_test)

        # 绘制决策边界。为此，对网格[x_min, x_max] x [y_min, y_max]中的每个点分配一个颜色
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
        else:
            Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

        # 将结果绘制成颜色图
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # 绘制训练数据点
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="black",
            s=25,
        )
        # 绘制测试数据点
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            alpha=0.6,
            edgecolors="black",
            s=25,
        )

        # 设置子图的 x 轴和 y 轴范围
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # 清空当前子图的 x 轴刻度
        ax.set_xticks(())
        # 清空当前子图的 y 轴刻度
        ax.set_yticks(())
        # 设置子图标题为分类器的名称
        ax.set_title(name)
        # 在子图中添加文本，显示分类器在测试集上的准确率
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            f"{score:.3f}".lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        # 自增计数器 i，用于确定下一个子图的位置
        i += 1
# 调整图形的子图布局，设置左侧边界为0.02，右侧边界为0.98
figure.subplots_adjust(left=0.02, right=0.98)
# 显示当前所有的图形
plt.show()
```