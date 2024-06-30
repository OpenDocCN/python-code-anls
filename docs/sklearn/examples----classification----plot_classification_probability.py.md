# `D:\src\scipysrc\scikit-learn\examples\classification\plot_classification_probability.py`

```
"""
===============================
Plot classification probability
===============================

Plot the classification probability for different classifiers. We use a 3 class
dataset, and we classify it with a Support Vector classifier, L1 and L2
penalized logistic regression (multinomial multiclass), a One-Vs-Rest version with
logistic regression, and Gaussian process classification.

Linear SVC is not a probabilistic classifier by default but it has a built-in
calibration option enabled in this example (`probability=True`).

The logistic regression with One-Vs-Rest is not a multiclass classifier out of
the box. As a result it has more trouble in separating class 2 and 3 than the
other estimators.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算
from matplotlib import cm  # 导入 matplotlib 的颜色映射模块

from sklearn import datasets  # 导入 sklearn 中的 datasets 模块
from sklearn.gaussian_process import GaussianProcessClassifier  # 导入高斯过程分类器
from sklearn.gaussian_process.kernels import RBF  # 导入高斯过程分类器的径向基核函数
from sklearn.inspection import DecisionBoundaryDisplay  # 导入决策边界显示模块
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import accuracy_score  # 导入准确率评估函数
from sklearn.multiclass import OneVsRestClassifier  # 导入一对多分类器
from sklearn.svm import SVC  # 导入支持向量机分类器

iris = datasets.load_iris()  # 载入鸢尾花数据集
X = iris.data[:, 0:2]  # 只使用前两个特征进行可视化
y = iris.target  # 目标变量

n_features = X.shape[1]  # 特征数量

C = 10  # 正则化参数 C
kernel = 1.0 * RBF([1.0, 1.0])  # GPC 的径向基核函数参数

# 创建不同的分类器
classifiers = {
    "L1 logistic": LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=10000),
    "L2 logistic (Multinomial)": LogisticRegression(
        C=C, penalty="l2", solver="saga", max_iter=10000
    ),
    "L2 logistic (OvR)": OneVsRestClassifier(
        LogisticRegression(C=C, penalty="l2", solver="saga", max_iter=10000)
    ),
    "Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0),
    "GPC": GaussianProcessClassifier(kernel),
}

n_classifiers = len(classifiers)  # 分类器的数量

fig, axes = plt.subplots(  # 创建子图
    nrows=n_classifiers,
    ncols=len(iris.target_names),
    figsize=(3 * 2, n_classifiers * 2),
)

for classifier_idx, (name, classifier) in enumerate(classifiers.items()):  # 遍历每个分类器
    y_pred = classifier.fit(X, y).predict(X)  # 训练分类器并预测结果
    accuracy = accuracy_score(y, y_pred)  # 计算准确率
    print(f"Accuracy (train) for {name}: {accuracy:0.1%}")  # 打印训练集准确率
    # 对于标签中的每一个唯一值，绘制分类器提供的概率估计图
    disp = DecisionBoundaryDisplay.from_estimator(
        classifier,  # 使用给定的分类器对象
        X,  # 输入数据 X
        response_method="predict_proba",  # 使用 predict_proba 方法获取分类概率
        class_of_interest=label,  # 当前感兴趣的类别标签
        ax=axes[classifier_idx, label],  # 绘图的坐标轴对象
        vmin=0,  # 概率值的最小值
        vmax=1,  # 概率值的最大值
    )
    # 设置子图标题为当前类别的名称
    axes[classifier_idx, label].set_title(f"Class {label}")
    
    # 绘制被预测为当前类别的数据点
    mask_y_pred = y_pred == label  # 创建一个布尔掩码，标记预测为当前类别的数据点
    axes[classifier_idx, label].scatter(
        X[mask_y_pred, 0], X[mask_y_pred, 1],  # X 中被预测为当前类别的数据点的坐标
        marker="o",  # 使用圆圈标记
        c="w",  # 标记颜色为白色
        edgecolor="k"  # 边缘颜色为黑色
    )
    # 设置子图的坐标轴标记为空，不显示刻度
    axes[classifier_idx, label].set(xticks=(), yticks=())
    
# 设置当前分类器子图列的 ylabel 为分类器名称
axes[classifier_idx, 0].set_ylabel(name)
# 创建一个新的绘图 axes 对象，定义其位置和大小，这里的 [0.15, 0.04, 0.7, 0.02] 分别表示左边距、底边距、宽度和高度
ax = plt.axes([0.15, 0.04, 0.7, 0.02])

# 设置绘图的标题为 "Probability"
plt.title("Probability")

# 创建一个颜色条，使用 "viridis" 颜色映射，不应用额外的数据标准化(norm=None)，并将其放置在之前创建的 axes 对象上，
# 设置颜色条的方向为水平方向
_ = plt.colorbar(
    cm.ScalarMappable(norm=None, cmap="viridis"), cax=ax, orientation="horizontal"
)

# 显示绘图
plt.show()
```