# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_pca_vs_lda.py`

```
"""
=======================================================
Comparison of LDA and PCA 2D projection of Iris dataset
=======================================================

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.

"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 导入 sklearn 库中的 datasets 模块
from sklearn import datasets
# 导入 sklearn 库中的 PCA 模块
from sklearn.decomposition import PCA
# 导入 sklearn 库中的 LinearDiscriminantAnalysis 模块
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 加载鸢尾花数据集
iris = datasets.load_iris()

# 提取特征数据
X = iris.data
# 提取目标标签
y = iris.target
# 目标标签的名称
target_names = iris.target_names

# 创建 PCA 对象，设置要降到的维度为 2
pca = PCA(n_components=2)
# 对数据进行 PCA 降维处理，并获取转换后的数据
X_r = pca.fit(X).transform(X)

# 创建 LDA 对象，设置要降到的维度为 2
lda = LinearDiscriminantAnalysis(n_components=2)
# 对数据进行 LDA 降维处理，并获取转换后的数据
X_r2 = lda.fit(X, y).transform(X)

# 打印每个主成分解释的方差百分比
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)

# 绘制 PCA 降维后的散点图
plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

# 遍历每个类别，绘制对应颜色的散点图
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")

# 绘制 LDA 降维后的散点图
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset")

# 展示绘制的图形
plt.show()
```