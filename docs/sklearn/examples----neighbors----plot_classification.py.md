# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_classification.py`

```
# %%
# Load the data
# -------------
#
# In this example, we use the iris dataset. We split the data into a train and test
# dataset.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载鸢尾花数据集，并将其作为DataFrame对象返回
iris = load_iris(as_frame=True)
# 提取特征数据：萼片长度和萼片宽度
X = iris.data[["sepal length (cm)", "sepal width (cm)"]]
# 提取目标数据（标签）
y = iris.target
# 将数据集拆分为训练集和测试集，保持类分布的均衡性，使用随机种子0进行随机化
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# %%
# K-nearest neighbors classifier
# ------------------------------
#
# We want to use a k-nearest neighbors classifier considering a neighborhood of 11 data
# points. Since our k-nearest neighbors model uses euclidean distance to find the
# nearest neighbors, it is therefore important to scale the data beforehand. Refer to
# the example entitled
# :ref:`sphx_glr_auto_examples_preprocessing_plot_scaling_importance.py` for more
# detailed information.
#
# Thus, we use a :class:`~sklearn.pipeline.Pipeline` to chain a scaler before to use
# our classifier.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 创建一个Pipeline对象，包含数据标准化和K近邻分类器
clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)

# %%
# Decision boundary
# -----------------
#
# Now, we fit two classifiers with different values of the parameter
# `weights`. We plot the decision boundary of each classifier as well as the original
# dataset to observe the difference.
import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

# 创建一个具有两个子图的图形窗口
_, axs = plt.subplots(ncols=2, figsize=(12, 5))

# 对于每个子图和权重值，使用不同权重值（uniform和distance）训练并绘制决策边界
for ax, weights in zip(axs, ("uniform", "distance")):
    # 设置分类器的权重参数并拟合数据
    clf.set_params(knn__weights=weights).fit(X_train, y_train)
    # 通过分类器创建决策边界的可视化显示对象
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_test,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
        shading="auto",
        alpha=0.5,
        ax=ax,
    )
    # 在子图上绘制原始数据的散点图，并添加图例显示类别信息
    scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
    disp.ax_.legend(
        scatter.legend_elements()[0],
        iris.target_names,
        loc="lower left",
        title="Classes",
    )
    # 设置子图标题，显示分类的信息
    _ = disp.ax_.set_title(
        f"3-Class classification\n(k={clf[-1].n_neighbors}, weights={weights!r})"
    )

# 显示图形
plt.show()

# %%
# Conclusion
# ----------
#
# We observe that the parameter `weights` has an impact on the decision boundary. When
# `weights="uniform"` all nearest neighbors will have the same impact on the decision.
# Whereas when `weights="distance"` the weight given to each neighbor is proportional
# to the inverse of its distance.
# 根据邻居到查询点的距离的倒数加权邻居的权重。
# 在某些情况下，考虑距离可能会改善模型。
```