# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_nca_classification.py`

```
"""
=============================================================================
Comparing Nearest Neighbors with and without Neighborhood Components Analysis
=============================================================================

An example comparing nearest neighbors classification with and without
Neighborhood Components Analysis.

It will plot the class decision boundaries given by a Nearest Neighbors
classifier when using the Euclidean distance on the original features, versus
using the Euclidean distance after the transformation learned by Neighborhood
Components Analysis. The latter aims to find a linear transformation that
maximises the (stochastic) nearest neighbor classification accuracy on the
training set.

"""

# SPDX-License-Identifier: BSD-3-Clause

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库用于绘图
from matplotlib.colors import ListedColormap  # 导入 ListedColormap 用于自定义颜色映射

from sklearn import datasets  # 导入 sklearn 的 datasets 模块用于加载数据集
from sklearn.inspection import DecisionBoundaryDisplay  # 导入 DecisionBoundaryDisplay 用于绘制决策边界的显示
from sklearn.model_selection import train_test_split  # 导入 train_test_split 用于数据集划分
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis  # 导入 KNeighborsClassifier 和 NeighborhoodComponentsAnalysis 用于最近邻分类和邻域成分分析
from sklearn.pipeline import Pipeline  # 导入 Pipeline 用于构建机器学习流水线
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler 用于数据标准化

n_neighbors = 1  # 设置最近邻数量为 1

dataset = datasets.load_iris()  # 加载鸢尾花数据集
X, y = dataset.data, dataset.target  # 分别获取数据集特征和标签

# 仅使用两个特征进行演示，可以通过使用二维数据集来避免这种不优雅的切片
X = X[:, [0, 2]]  # 仅保留数据集的第一列和第三列特征

# 将数据集划分为训练集和测试集，保持类别比例分布，测试集占比 70%，随机种子为 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.7, random_state=42
)

h = 0.05  # 网格中的步长

# 创建颜色映射
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])  # 浅色映射，用于绘制决策边界
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])  # 深色映射，用于绘制数据点颜色

names = ["KNN", "NCA, KNN"]  # 分类器名称列表

classifiers = [
    Pipeline(
        [
            ("scaler", StandardScaler()),  # 数据标准化处理
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),  # K最近邻分类器
        ]
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),  # 数据标准化处理
            ("nca", NeighborhoodComponentsAnalysis()),  # 邻域成分分析
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),  # K最近邻分类器
        ]
    ),
]

# 对每个分类器进行训练和评估
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)  # 使用训练集训练分类器
    score = clf.score(X_test, y_test)  # 在测试集上评估分类器的性能

    _, ax = plt.subplots()  # 创建图形窗口
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=cmap_light,
        alpha=0.8,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
    )

    # 绘制训练集和测试集的数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
    plt.title("{} (k = {})".format(name, n_neighbors))  # 设置图表标题
    plt.text(
        0.9,
        0.1,
        "{:.2f}".format(score),
        size=15,
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

plt.show()  # 显示图形
```