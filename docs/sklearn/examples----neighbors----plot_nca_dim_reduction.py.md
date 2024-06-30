# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_nca_dim_reduction.py`

```
"""
==============================================================
Dimensionality Reduction with Neighborhood Components Analysis
==============================================================

Sample usage of Neighborhood Components Analysis for dimensionality reduction.

This example compares different (linear) dimensionality reduction methods
applied on the Digits data set. The data set contains images of digits from
0 to 9 with approximately 180 samples of each class. Each image is of
dimension 8x8 = 64, and is reduced to a two-dimensional data point.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.

Neighborhood Components Analysis (NCA) tries to find a feature space such
that a stochastic nearest neighbor algorithm will give the best accuracy.
Like LDA, it is a supervised method.

One can see that NCA enforces a clustering of the data that is visually
meaningful despite the large reduction in dimension.

"""

# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy

from sklearn import datasets  # 导入数据集模块
from sklearn.decomposition import PCA  # 导入PCA模块
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 导入LDA模块
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis  # 导入KNN和NCA模块
from sklearn.pipeline import make_pipeline  # 导入管道函数
from sklearn.preprocessing import StandardScaler  # 导入数据标准化函数

n_neighbors = 3  # 设置KNN中的邻居数量
random_state = 0  # 设定随机种子，用于可重复性实验

# Load Digits dataset
X, y = datasets.load_digits(return_X_y=True)  # 载入手写数字数据集

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=random_state
)  # 将数据集分为训练集和测试集，保持类别比例分布

dim = len(X[0])  # 数据集中每个样本的维度
n_classes = len(np.unique(y))  # 数据集中类别的数量

# Reduce dimension to 2 with PCA
pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))
# 使用PCA将数据降至二维

# Reduce dimension to 2 with LinearDiscriminantAnalysis
lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))
# 使用LDA将数据降至二维

# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca = make_pipeline(
    StandardScaler(),
    NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
)
# 使用NCA将数据降至二维

# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
# 使用KNN作为评估各种降维方法的分类器

# Make a list of the methods to be compared
dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]
# 构建一个比较不同降维方法的列表

# plt.figure()
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    # 创建一个新的绘图窗口

    # plt.subplot(1, 3, i + 1, aspect=1)
    # 创建一个1x3的子图布局，当前子图位置为第i+1个（i从0开始）

    # Fit the method's model
    model.fit(X_train, y_train)
    # 对当前方法的模型进行拟合
    # 在嵌入的训练集上训练最近邻分类器
    knn.fit(model.transform(X_train), y_train)

    # 计算嵌入的测试集上最近邻分类器的准确率
    acc_knn = knn.score(model.transform(X_test), y_test)

    # 使用已拟合的模型将数据集嵌入到二维空间中
    X_embedded = model.transform(X)

    # 绘制投影后的点，并显示评估分数
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
    plt.title(
        "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
    )
# 展示 matplotlib 中当前所有图形
plt.show()
```