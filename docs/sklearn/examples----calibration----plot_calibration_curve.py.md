# `D:\src\scipysrc\scikit-learn\examples\calibration\plot_calibration_curve.py`

```
"""
==============================
Probability Calibration curves
==============================

When performing classification one often wants to predict not only the class
label, but also the associated probability. This probability gives some
kind of confidence on the prediction. This example demonstrates how to
visualize how well calibrated the predicted probabilities are using calibration
curves, also known as reliability diagrams. Calibration of an uncalibrated
classifier will also be demonstrated.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# %%
# Dataset
# -------
#
# We will use a synthetic binary classification dataset with 100,000 samples
# and 20 features. Of the 20 features, only 2 are informative, 10 are
# redundant (random combinations of the informative features) and the
# remaining 8 are uninformative (random numbers). Of the 100,000 samples, 1,000
# will be used for model fitting and the rest for testing.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成一个包含 100,000 个样本和 20 个特征的合成二分类数据集，其中有 2 个信息特征，10 个冗余特征，以及 8 个不相关的特征
X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)

# 将数据集划分为训练集和测试集，测试集占比 99%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)

# %%
# Calibration curves
# ------------------
#
# Gaussian Naive Bayes
# ^^^^^^^^^^^^^^^^^^^^
#
# First, we will compare:
#
# * :class:`~sklearn.linear_model.LogisticRegression` (used as baseline
#   since very often, properly regularized logistic regression is well
#   calibrated by default thanks to the use of the log-loss)
# * Uncalibrated :class:`~sklearn.naive_bayes.GaussianNB`
# * :class:`~sklearn.naive_bayes.GaussianNB` with isotonic and sigmoid
#   calibration (see :ref:`User Guide <calibration>`)
#
# Calibration curves for all 4 conditions are plotted below, with the average
# predicted probability for each bin on the x-axis and the fraction of positive
# classes in each bin on the y-axis.

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# 初始化一个 Logistic 回归模型
lr = LogisticRegression(C=1.0)
# 初始化一个高斯朴素贝叶斯模型
gnb = GaussianNB()
# 使用等温校准方法构建一个高斯朴素贝叶斯模型
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
# 使用 Sigmoid 校准方法构建一个高斯朴素贝叶斯模型
gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")

# 将所有分类器及其名称放入列表中
clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (gnb_isotonic, "Naive Bayes + Isotonic"),
    (gnb_sigmoid, "Naive Bayes + Sigmoid"),
]

# %%
# 创建一个 10x10 英寸大小的图形对象
fig = plt.figure(figsize=(10, 10))
# 创建一个 4x2 的网格布局
gs = GridSpec(4, 2)
# 获取颜色映射
colors = plt.get_cmap("Dark2")

# 将校准曲线的坐标轴添加到图形中
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
# 初始化一个字典用于存储校准显示对象
calibration_displays = {}

# 遍历分类器列表，训练模型并绘制校准曲线
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    # 使用给定的分类器（clf）、测试数据集（X_test, y_test）和其他参数创建一个校准显示对象
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,  # 校准曲线的箱数
        name=name,  # 显示名称
        ax=ax_calibration_curve,  # 绘图坐标轴对象
        color=colors(i),  # 根据索引 i 获取对应的颜色
    )
    # 将创建的校准显示对象存储到校准显示字典中，键为名称 name
    calibration_displays[name] = display
# 将校准曲线添加到当前的坐标轴上
ax_calibration_curve.grid()
# 设置标题为“Calibration plots (Naive Bayes)”
ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

# 添加直方图
# 定义网格位置
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
# 对于每个分类器及其名称，在指定的网格位置创建子图并添加直方图
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,  # 使用分类器名称从calibration_displays中获取概率数据
        range=(0, 1),  # 设定直方图的范围为0到1
        bins=10,  # 将数据分成10个箱子
        label=name,  # 添加图例，显示分类器名称
        color=colors(i),  # 使用指定的颜色函数来为直方图着色
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")  # 设置子图的标题、x轴和y轴标签

plt.tight_layout()  # 调整布局，使子图之间没有重叠
plt.show()  # 显示图形

# %%
# 未校准的 GaussianNB 模型由于违反特征独立性假设而校准不佳，
# 这导致了一个过于自信的分类器，表现为典型的S形曲线。通过使用
# isotonic 校准可以修复这个问题，这在几乎对角的校准曲线中可以看出。
# sigmoid 回归也可以稍微改善校准，尽管不如非参数的 isotonic 校准强。
# 这可以归因于我们有大量的校准数据，使得非参数模型的更大灵活性可以得到利用。
#
# 下面我们将进行定量分析，考虑几个分类度量指标：
# brier_score_loss、log_loss、precision、recall、F1 score 和 ROC AUC。

from collections import defaultdict  # 导入 defaultdict 类

import pandas as pd  # 导入 pandas 库

from sklearn.metrics import (  # 从 sklearn.metrics 导入需要的度量函数
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

scores = defaultdict(list)  # 创建一个 defaultdict 列表来存储得分
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)  # 使用训练集对分类器进行拟合
    y_prob = clf.predict_proba(X_test)  # 预测测试集的类概率
    y_pred = clf.predict(X_test)  # 预测测试集的类标签
    scores["Classifier"].append(name)  # 将分类器名称添加到 scores 中

    # 对于每个度量函数，计算其在测试集上的得分并添加到 scores 中
    for metric in [brier_score_loss, log_loss, roc_auc_score]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))

    for metric in [precision_score, recall_score, f1_score]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_pred))

    score_df = pd.DataFrame(scores).set_index("Classifier")  # 创建包含得分的 DataFrame，并以分类器名称为索引
    score_df.round(decimals=3)  # 将得分保留三位小数

score_df  # 显示得分 DataFrame

# %%
# 注意，尽管校准可以改善 brier_score_loss 和 log_loss，但它并不显著改变预测准确度度量（precision、recall 和 F1 score）。
# 这是因为校准不应显著改变决策阈值位置处（图中 x = 0.5 处）的预测概率。
# 然而，校准应该使得预测的概率更为
# accurate and thus more useful for making allocation decisions under
# uncertainty.
# Further, ROC AUC, should not change at all because calibration is a
# monotonic transformation. Indeed, no rank metrics are affected by
# calibration.
#
# Linear support vector classifier
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next, we will compare:
#
# * :class:`~sklearn.linear_model.LogisticRegression` (baseline)
# * Uncalibrated :class:`~sklearn.svm.LinearSVC`. Since SVC does not output
#   probabilities by default, we naively scale the output of the
#   :term:`decision_function` into [0, 1] by applying min-max scaling.
# * :class:`~sklearn.svm.LinearSVC` with isotonic and sigmoid
#   calibration (see :ref:`User Guide <calibration>`)

import numpy as np

from sklearn.svm import LinearSVC


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output for binary classification."""

    def fit(self, X, y):
        super().fit(X, y)
        # Calculate decision function output for training data
        df = self.decision_function(X)
        # Store minimum and maximum values of decision function output
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0, 1]."""
        # Calculate decision function output for input data
        df = self.decision_function(X)
        # Scale decision function output to [0, 1] using min-max scaling
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        # Compute probability for positive class and its complement for negative class
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        # Combine probabilities into a matrix [prob_neg, prob_pos]
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


# %%

lr = LogisticRegression(C=1.0)
# Initialize NaivelyCalibratedLinearSVC with a maximum iteration limit
svc = NaivelyCalibratedLinearSVC(max_iter=10_000)
# Calibrate NaivelyCalibratedLinearSVC using isotonic regression
svc_isotonic = CalibratedClassifierCV(svc, cv=2, method="isotonic")
# Calibrate NaivelyCalibratedLinearSVC using sigmoid regression
svc_sigmoid = CalibratedClassifierCV(svc, cv=2, method="sigmoid")

# List of classifiers to compare
clf_list = [
    (lr, "Logistic"),
    (svc, "SVC"),
    (svc_isotonic, "SVC + Isotonic"),
    (svc_sigmoid, "SVC + Sigmoid"),
]

# %%
# Create a figure for plotting
fig = plt.figure(figsize=(10, 10))
# Define a grid for subplot arrangement
gs = GridSpec(4, 2)

# Add a subplot for calibration curves
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
# Dictionary to store CalibrationDisplay objects
calibration_displays = {}

# Generate calibration plots for each classifier
for i, (clf, name) in enumerate(clf_list):
    # Fit classifier on training data
    clf.fit(X_train, y_train)
    # Generate CalibrationDisplay object for the classifier
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    # Store the display object in dictionary
    calibration_displays[name] = display

# Customize calibration subplot
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (SVC)")

# Add histograms to the remaining grid positions
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    # Determine subplot position
    row, col = grid_positions[i]
    # Add a subplot for histogram
    ax = fig.add_subplot(gs[row, col])

    # Plot histogram of predicted probabilities
    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    # Set subplot title and labels
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

# Adjust layout to prevent overlap
plt.tight_layout()
# Display the plot
plt.show()

# %%
# :class:`~sklearn.svm.LinearSVC` shows the opposite
# behavior to :class:`~sklearn.naive_bayes.GaussianNB`; the calibration
# scores 是一个 defaultdict，用于存储各种分类器的评分数据
scores = defaultdict(list)
# 遍历 clf_list 中的每个分类器 clf，并使用 X_train 和 y_train 进行训练
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    # 获取分类器在 X_test 上的预测概率
    y_prob = clf.predict_proba(X_test)
    # 获取分类器在 X_test 上的预测类别
    y_pred = clf.predict(X_test)
    # 将分类器的名称 name 添加到 scores 字典中的 "Classifier" 键对应的列表中
    scores["Classifier"].append(name)

    # 遍历评分指标列表 [brier_score_loss, log_loss, roc_auc_score]
    for metric in [brier_score_loss, log_loss, roc_auc_score]:
        # 获取评分函数的名称，并进行格式化
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        # 计算评分，并将结果添加到对应指标的列表中
        scores[score_name].append(metric(y_test, y_prob[:, 1]))

    # 遍历评分指标列表 [precision_score, recall_score, f1_score]
    for metric in [precision_score, recall_score, f1_score]:
        # 获取评分函数的名称，并进行格式化
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        # 计算评分，并将结果添加到对应指标的列表中
        scores[score_name].append(metric(y_test, y_pred))

    # 创建包含当前所有评分数据的 DataFrame，以分类器名称作为索引
    score_df = pd.DataFrame(scores).set_index("Classifier")
    # 对 DataFrame 中的数据进行小数点后三位的四舍五入
    score_df.round(decimals=3)

# 返回最终的评分结果 DataFrame
score_df
```