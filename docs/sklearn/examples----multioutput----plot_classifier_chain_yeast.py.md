# `D:\src\scipysrc\scikit-learn\examples\multioutput\plot_classifier_chain_yeast.py`

```
"""
==================================================
Multilabel classification using a classifier chain
==================================================
This example shows how to use :class:`~sklearn.multioutput.ClassifierChain` to solve
a multilabel classification problem.

The most naive strategy to solve such a task is to independently train a binary
classifier on each label (i.e. each column of the target variable). At prediction
time, the ensemble of binary classifiers is used to assemble multitask prediction.

This strategy does not allow to model relationship between different tasks. The
:class:`~sklearn.multioutput.ClassifierChain` is the meta-estimator (i.e. an estimator
taking an inner estimator) that implements a more advanced strategy. The ensemble
of binary classifiers are used as a chain where the prediction of a classifier in the
chain is used as a feature for training the next classifier on a new label. Therefore,
these additional features allow each chain to exploit correlations among labels.

The :ref:`Jaccard similarity <jaccard_similarity_score>` score for chain tends to be
greater than that of the set independent base models.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Loading a dataset
# -----------------
# For this example, we use the `yeast
# <https://www.openml.org/d/40597>`_ dataset which contains
# 2,417 datapoints each with 103 features and 14 possible labels. Each
# data point has at least one label. As a baseline we first train a logistic
# regression classifier for each of the 14 labels. To evaluate the performance of
# these classifiers we predict on a held-out test set and calculate the
# Jaccard similarity for each sample.

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load a multi-label dataset from https://www.openml.org/d/40597
X, Y = fetch_openml("yeast", version=4, return_X_y=True)
Y = Y == "TRUE"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
# Fit models
# ----------
# We fit :class:`~sklearn.linear_model.LogisticRegression` wrapped by
# :class:`~sklearn.multiclass.OneVsRestClassifier` and ensemble of multiple
# :class:`~sklearn.multioutput.ClassifierChain`.
#
# LogisticRegression wrapped by OneVsRestClassifier
# **************************************************
# Since by default :class:`~sklearn.linear_model.LogisticRegression` can't
# handle data with multiple targets, we need to use
# :class:`~sklearn.multiclass.OneVsRestClassifier`.
# After fitting the model we calculate Jaccard similarity.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.multiclass import OneVsRestClassifier

# 创建一个LogisticRegression的基础模型
base_lr = LogisticRegression()
# 使用OneVsRestClassifier将LogisticRegression转化为多标签分类器
ovr = OneVsRestClassifier(base_lr)
# 在训练集上拟合模型
ovr.fit(X_train, Y_train)
# 对测试集进行预测
Y_pred_ovr = ovr.predict(X_test)
# Calculate the Jaccard similarity score between Y_test and Y_pred_ovr
ovr_jaccard_score = jaccard_score(Y_test, Y_pred_ovr, average="samples")

# %%
# Chain of binary classifiers
# ***************************
# 因为每个链中的模型是随机排列的，所以不同链之间的性能存在显著差异。
# 可能存在一种最佳的类别顺序，可以获得最佳性能。然而，我们事先不知道这种排序。
# 相反，我们可以通过对分类器链的二进制预测取平均值，并应用阈值0.5来构建一个投票集成。
# 集成的 Jaccard 相似度分数大于独立模型的分数，并且往往超过集合中每个链的分数（尽管随机排序的链不保证如此）。

from sklearn.multioutput import ClassifierChain

# 创建包含10个随机排列的分类器链列表
chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(10)]
# 在训练集上拟合每个链
for chain in chains:
    chain.fit(X_train, Y_train)

# 对测试集进行预测，得到每个链的预测概率
Y_pred_chains = np.array([chain.predict_proba(X_test) for chain in chains])
# 计算每个链的 Jaccard 相似度分数
chain_jaccard_scores = [
    jaccard_score(Y_test, Y_pred_chain >= 0.5, average="samples")
    for Y_pred_chain in Y_pred_chains
]

# 计算集成预测结果，即所有链预测结果的平均值
Y_pred_ensemble = Y_pred_chains.mean(axis=0)
# 计算集成的 Jaccard 相似度分数
ensemble_jaccard_score = jaccard_score(
    Y_test, Y_pred_ensemble >= 0.5, average="samples"
)

# %%
# Plot results
# ------------
# 绘制独立模型、每个链以及集成的 Jaccard 相似度分数（请注意，此图的纵轴不从0开始）。

# 将所有模型的 Jaccard 相似度分数组合成一个列表
model_scores = [ovr_jaccard_score] + chain_jaccard_scores + [ensemble_jaccard_score]

# 定义模型名称，包括独立模型、每个链以及集成模型
model_names = (
    "Independent",
    "Chain 1",
    "Chain 2",
    "Chain 3",
    "Chain 4",
    "Chain 5",
    "Chain 6",
    "Chain 7",
    "Chain 8",
    "Chain 9",
    "Chain 10",
    "Ensemble",
)

# 设置 x 轴位置
x_pos = np.arange(len(model_names))

# 创建图表和坐标轴
fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True)
ax.set_title("Classifier Chain Ensemble Performance Comparison")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation="vertical")
ax.set_ylabel("Jaccard Similarity Score")
# 设置纵轴范围，留出一定的空间以确保最小值和最大值的显示
ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
# 为每个模型选择颜色，红色表示独立模型，蓝色表示每个链，绿色表示集成模型
colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
# 绘制条形图
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
# 紧凑布局
plt.tight_layout()
# 显示图表
plt.show()

# %%
# Results interpretation
# ----------------------
# 从这个图表中我们可以得出三个主要结论：
#
# - 由 :class:`~sklearn.multiclass.OneVsRestClassifier` 包装的独立模型表现比分类器链的集成和某些单独链要差。
#   这是因为逻辑回归未建模标签之间的关系。
# - :class:`~sklearn.multioutput.ClassifierChain` 利用标签之间的相关性，但由于标签顺序的随机性，可能产生比独立模型更差的结果。
# - 链的集成效果更好，因为它不仅捕捉了标签之间的关系
# 在标签之间进行调整，但也不假设它们的顺序完全正确。
```