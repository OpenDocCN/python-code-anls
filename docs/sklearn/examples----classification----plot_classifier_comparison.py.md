# `D:\src\scipysrc\scikit-learn\examples\classification\plot_classifier_comparison.py`

```
"""
=====================
Classifier comparison
=====================

A comparison of several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库和模块
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 定义分类器的名称
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

# 初始化各个分类器对象
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# 创建一个合成数据集
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# 定义不同的数据集
datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

# 创建一个大图
figure = plt.figure(figsize=(27, 9))
i = 1
# 迭代处理不同的数据集
for ds_cnt, ds in enumerate(datasets):
    # 预处理数据集，将数据集分割为训练集和测试集
    X, y = ds  # 从数据集中获取特征和标签
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )  # 使用 train_test_split 函数将数据集划分为训练集和测试集，比例为 0.6:0.4，随机种子为 42

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # 计算特征 X 的第一个维度的最小值和最大值，用于设置图表的 x 轴范围
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # 计算特征 X 的第二个维度的最小值和最大值，用于设置图表的 y 轴范围

    # 先绘制数据集的散点图
    cm = plt.cm.RdBu  # 使用红蓝色谱作为颜色映射
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # 定义明亮的颜色映射
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)  # 在子图中创建一个轴对象
    if ds_cnt == 0:
        ax.set_title("Input data")  # 如果是第一个数据集，设置图表标题为 "Input data"
    # 绘制训练数据点
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # 绘制测试数据点
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)  # 设置 x 轴的显示范围
    ax.set_ylim(y_min, y_max)  # 设置 y 轴的显示范围
    ax.set_xticks(())  # 设置不显示 x 轴刻度
    ax.set_yticks(())  # 设置不显示 y 轴刻度
    i += 1

    # 遍历分类器
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)  # 在子图中创建一个轴对象

        clf = make_pipeline(StandardScaler(), clf)  # 创建一个流水线，包括数据标准化和分类器
        clf.fit(X_train, y_train)  # 使用训练集训练分类器
        score = clf.score(X_test, y_test)  # 计算分类器在测试集上的准确率
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )  # 使用 DecisionBoundaryDisplay 显示分类器的决策边界

        # 绘制训练数据点
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # 绘制测试数据点
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)  # 设置 x 轴的显示范围
        ax.set_ylim(y_min, y_max)  # 设置 y 轴的显示范围
        ax.set_xticks(())  # 设置不显示 x 轴刻度
        ax.set_yticks(())  # 设置不显示 y 轴刻度
        if ds_cnt == 0:
            ax.set_title(name)  # 如果是第一个数据集，设置子图标题为分类器的名称
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),  # 在图表上显示分类器的准确率，去除小数点前多余的零
            size=15,
            horizontalalignment="right",
        )
        i += 1
# 调整图表布局使得子图之间不重叠
plt.tight_layout()
# 显示当前所有的图表
plt.show()
```