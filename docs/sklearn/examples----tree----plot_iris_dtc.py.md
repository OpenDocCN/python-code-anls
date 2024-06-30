# `D:\src\scipysrc\scikit-learn\examples\tree\plot_iris_dtc.py`

```
# %%
# 首先加载 scikit-learn 提供的鸢尾花数据集的副本：
from sklearn.datasets import load_iris

iris = load_iris()

# %%
# 显示训练在鸢尾花数据集所有特征对上的决策函数。
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# 参数设置
n_classes = 3  # 类别数量
plot_colors = "ryb"  # 绘图颜色
plot_step = 0.02  # 绘图步长

# 对于每一对特征进行遍历
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # 仅取这两个对应的特征
    X = iris.data[:, pair]
    y = iris.target

    # 训练决策树分类器
    clf = DecisionTreeClassifier().fit(X, y)

    # 绘制决策边界
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # 绘制训练点
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")

# %%
# 显示在所有鸢尾花特征上训练的单棵决策树的结构。
from sklearn.tree import plot_tree

plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.show()
```