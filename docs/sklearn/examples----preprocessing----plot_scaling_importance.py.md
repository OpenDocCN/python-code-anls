# `D:\src\scipysrc\scikit-learn\examples\preprocessing\plot_scaling_importance.py`

```
"""
=============================
Importance of Feature Scaling
=============================

Feature scaling through standardization, also called Z-score normalization, is
an important preprocessing step for many machine learning algorithms. It
involves rescaling each feature such that it has a standard deviation of 1 and a
mean of 0.

Even if tree based models are (almost) not affected by scaling, many other
algorithms require features to be normalized, often for different reasons: to
ease the convergence (such as a non-penalized logistic regression), to create a
completely different model fit compared to the fit with unscaled data (such as
KNeighbors models). The latter is demoed on the first part of the present
example.

On the second part of the example we show how Principal Component Analysis (PCA)
is impacted by normalization of features. To illustrate this, we compare the
principal components found using :class:`~sklearn.decomposition.PCA` on unscaled
data with those obtained when using a
:class:`~sklearn.preprocessing.StandardScaler` to scale data first.

In the last part of the example we show the effect of the normalization on the
accuracy of a model trained on PCA-reduced data.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Load and prepare data
# =====================
#
# The dataset used is the :ref:`wine_dataset` available at UCI. This dataset has
# continuous features that are heterogeneous in scale due to differing
# properties that they measure (e.g. alcohol content and malic acid).

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the wine dataset
X, y = load_wine(return_X_y=True, as_frame=True)

# Initialize a StandardScaler instance for scaling data
scaler = StandardScaler().set_output(transform="pandas")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Scale the training data using the StandardScaler
scaled_X_train = scaler.fit_transform(X_train)

# %%
# .. _neighbors_scaling:
#
# Effect of rescaling on a k-neighbors models
# ===========================================
#
# For the sake of visualizing the decision boundary of a
# :class:`~sklearn.neighbors.KNeighborsClassifier`, in this section we select a
# subset of 2 features that have values with different orders of magnitude.
#
# Keep in mind that using a subset of the features to train the model may likely
# leave out feature with high predictive impact, resulting in a decision
# boundary that is much worse in comparison to a model trained on the full set
# of features.

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier

# Select the features 'proline' and 'hue' for visualization
X_plot = X[["proline", "hue"]]

# Scale the selected features using StandardScaler
X_plot_scaled = scaler.fit_transform(X_plot)

# Initialize a KNeighborsClassifier with 20 neighbors
clf = KNeighborsClassifier(n_neighbors=20)

# Function to fit the model and plot the decision boundary
def fit_and_plot_model(X_plot, y, clf, ax):
    clf.fit(X_plot, y)
    # 使用给定的分类器和数据点绘制决策边界显示对象
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,                    # 分类器对象
        X_plot,                 # 待绘制的数据点集合
        response_method="predict",  # 使用预测方法进行响应
        alpha=0.5,              # 设置透明度为0.5，用于显示决策边界
        ax=ax,                  # 指定绘图的轴对象
    )
    # 在决策边界显示对象的轴上绘制散点图，用于展示数据点的分布
    disp.ax_.scatter(X_plot["proline"], X_plot["hue"], c=y, s=20, edgecolor="k")
    # 设置决策边界显示对象的轴的 X 轴范围，以确保数据点在合适的范围内显示
    disp.ax_.set_xlim((X_plot["proline"].min(), X_plot["proline"].max()))
    # 设置决策边界显示对象的轴的 Y 轴范围，以确保数据点在合适的范围内显示
    disp.ax_.set_ylim((X_plot["hue"].min(), X_plot["hue"].max()))
    # 返回绘制决策边界后的轴对象，用于进一步的操作或显示
    return disp.ax_
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
# 创建一个包含两个子图的图形窗口，每个子图的尺寸为12x6英寸

fit_and_plot_model(X_plot, y, clf, ax1)
# 在第一个子图(ax1)上绘制使用未经缩放的数据拟合的模型结果，并显示标题为"KNN without scaling"

ax1.set_title("KNN without scaling")
# 设置第一个子图(ax1)的标题为"KNN without scaling"

fit_and_plot_model(X_plot_scaled, y, clf, ax2)
# 在第二个子图(ax2)上绘制使用经过缩放的数据拟合的模型结果

ax2.set_xlabel("scaled proline")
# 设置第二个子图(ax2)的X轴标签为"scaled proline"

ax2.set_ylabel("scaled hue")
# 设置第二个子图(ax2)的Y轴标签为"scaled hue"

_ = ax2.set_title("KNN with scaling")
# 设置第二个子图(ax2)的标题为"KNN with scaling"
    # 在散点图对象 ax2 上绘制散点图
    ax2.scatter(
        # X_train_std_transformed 中目标类别 target_class 的第一列特征值作为 x 轴数据
        x=X_train_std_transformed[y_train == target_class, 0],
        # X_train_std_transformed 中目标类别 target_class 的第二列特征值作为 y 轴数据
        y=X_train_std_transformed[y_train == target_class, 1],
        # 设置散点的颜色
        color=color,
        # 设置图例中显示的标签，显示为 "class {target_class}"
        label=f"class {target_class}",
        # 设置散点的透明度
        alpha=0.5,
        # 设置散点的标记形状
        marker=marker,
    )
ax1.set_title("Unscaled training dataset after PCA")
ax2.set_title("Standardized training dataset after PCA")

for ax in (ax1, ax2):
    ax.set_xlabel("1st principal component")
    ax.set_ylabel("2nd principal component")
    ax.legend(loc="upper right")
    ax.grid()

_ = plt.tight_layout()

# %%
# 从上面的图表中可以观察到，在降维前对特征进行缩放会使得主成分的数量级相似。
# 在这种情况下，它还改善了类别的可分离性。确实，在接下来的部分中，我们确认
# 了更好的可分离性对整体模型性能有良好的影响。
#
# 缩放对模型性能的影响
# ========================
#
# 首先我们展示了在数据缩放或不缩放的情况下，`LogisticRegressionCV`
# 的最佳正则化参数的依赖关系：

import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline

Cs = np.logspace(-5, 5, 20)

unscaled_clf = make_pipeline(pca, LogisticRegressionCV(Cs=Cs))
unscaled_clf.fit(X_train, y_train)

scaled_clf = make_pipeline(scaler, pca, LogisticRegressionCV(Cs=Cs))
scaled_clf.fit(X_train, y_train)

print(f"Optimal C for the unscaled PCA: {unscaled_clf[-1].C_[0]:.4f}\n")
print(f"Optimal C for the standardized data with PCA: {scaled_clf[-1].C_[0]:.2f}")

# %%
# 对于在应用 PCA 前未进行缩放的数据，需要更高的正则化程度（较低的 `C` 值）。
# 现在我们评估了缩放对最优模型的准确率和平均对数损失的影响：

from sklearn.metrics import accuracy_score, log_loss

y_pred = unscaled_clf.predict(X_test)
y_pred_scaled = scaled_clf.predict(X_test)
y_proba = unscaled_clf.predict_proba(X_test)
y_proba_scaled = scaled_clf.predict_proba(X_test)

print("Test accuracy for the unscaled PCA")
print(f"{accuracy_score(y_test, y_pred):.2%}\n")
print("Test accuracy for the standardized data with PCA")
print(f"{accuracy_score(y_test, y_pred_scaled):.2%}\n")
print("Log-loss for the unscaled PCA")
print(f"{log_loss(y_test, y_proba):.3}\n")
print("Log-loss for the standardized data with PCA")
print(f"{log_loss(y_test, y_proba_scaled):.3}")

# %%
# 当数据在应用 `PCA` 前进行缩放时，观察到预测准确率存在明显差异，因此
# 缩放后的版本明显优于未缩放的版本。这与前面部分图表得出的直觉相符，
# 即在使用 `PCA` 前进行缩放后，主成分变得线性可分。
#
# 需要注意的是，在这种情况下，具有缩放特征的模型表现比非缩放特征的模型更好，
# 因为所有变量都预期具有预测性能，我们宁愿避免某些变量相对被忽略。
#
# 如果较小尺度的变量不具有预测性能，可能会出现...
# 在特征缩放后性能下降：在特征缩放后，噪声特征对预测的影响增加，因此缩放会增加过拟合的可能性。
# 最后但同样重要的是，我们观察到通过缩放步骤可以获得更低的对数损失。
```