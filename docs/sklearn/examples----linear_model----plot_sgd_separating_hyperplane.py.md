# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgd_separating_hyperplane.py`

```
"""
=========================================
SGD: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a linear Support Vector Machines classifier
trained using SGD.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图
import numpy as np  # 导入 numpy 进行数值计算

from sklearn.datasets import make_blobs  # 导入 make_blobs 用于生成数据集
from sklearn.linear_model import SGDClassifier  # 导入 SGDClassifier 作为分类器

# 创建一个包含50个可分离点的数据集
X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# 使用 SGDClassifier 进行模型拟合
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)

clf.fit(X, Y)

# 绘制分隔超平面、数据点和最近的向量
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ["dashed", "solid", "dashed"]
colors = "k"
# 绘制等高线图，表示分隔超平面
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
# 绘制散点图，显示数据点
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolor="black", s=20)

plt.axis("tight")
plt.show()
```