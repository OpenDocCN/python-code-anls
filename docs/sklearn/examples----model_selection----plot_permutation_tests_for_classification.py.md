# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_permutation_tests_for_classification.py`

```
"""
=================================================================
Test with permutations the significance of a classification score
=================================================================

This example demonstrates the use of
:func:`~sklearn.model_selection.permutation_test_score` to evaluate the
significance of a cross-validated score using permutations.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Dataset
# -------
#
# We will use the :ref:`iris_dataset`, which consists of measurements taken
# from 3 types of irises.
从 sklearn.datasets 导入 load_iris 函数
加载 iris 数据集
iris = load_iris()
获取特征数据
X = iris.data
获取目标数据
y = iris.target

# %%
# We will also generate some random feature data (i.e., 20 features),
# uncorrelated with the class labels in the iris dataset.
从 numpy 导入 np
定义未相关特征数为 20
使用随机种子 0 创建随机数生成器 rng
生成与 iris 数据集样本数相同且具有 20 个特征的正态分布随机数据 X_rand

# %%
# Permutation test score
# ----------------------
#
# Next, we calculate the
# :func:`~sklearn.model_selection.permutation_test_score` using the original
# iris dataset, which strongly predict the labels and
# the randomly generated features and iris labels, which should have
# no dependency between features and labels. We use the
# :class:`~sklearn.svm.SVC` classifier and :ref:`accuracy_score` to evaluate
# the model at each round.
#
# :func:`~sklearn.model_selection.permutation_test_score` generates a null
# distribution by calculating the accuracy of the classifier
# on 1000 different permutations of the dataset, where features
# remain the same but labels undergo different permutations. This is the
# distribution for the null hypothesis which states there is no dependency
# between the features and labels. An empirical p-value is then calculated as
# the percentage of permutations for which the score obtained is greater
# that the score obtained using the original data.
从 sklearn.model_selection 导入 StratifiedKFold 和 permutation_test_score
从 sklearn.svm 导入 SVC
创建线性核的 SVC 分类器 clf，随机种子设为 7
创建分层 K 折交叉验证器 cv，分成 2 折，打乱数据，随机种子为 0

使用 permutation_test_score 计算 iris 数据集的分类准确率评分 score_iris、置换分数 perm_scores_iris 和 p 值 pvalue_iris
使用 permutation_test_score 计算随机生成特征数据 X_rand 和 iris 标签的分类准确率评分 score_rand、置换分数 perm_scores_rand 和 p 值 pvalue_rand

# %%
# Original data
# ^^^^^^^^^^^^^
#
# Below we plot a histogram of the permutation scores (the null
# distribution). The red line indicates the score obtained by the classifier
# on the original data. The score is much better than those obtained by
# using permuted data and the p-value is thus very low. This indicates that
# there is a low likelihood that this good score would be obtained by chance
# alone. It provides evidence that the iris dataset contains real dependency
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 创建一个新的图形和子图对象
fig, ax = plt.subplots()

# 绘制原始数据的直方图，perm_scores_iris 是经过排列后的得分数据
ax.hist(perm_scores_iris, bins=20, density=True)
# 在直方图上添加一个垂直虚线，表示原始数据的得分值 score_iris
ax.axvline(score_iris, ls="--", color="r")
# 创建一个包含得分信息的标签字符串，显示在图中特定位置
score_label = f"Score on original\ndata: {score_iris:.2f}\n(p-value: {pvalue_iris:.3f})"
ax.text(0.7, 10, score_label, fontsize=12)
# 设置 X 轴和 Y 轴的标签
ax.set_xlabel("Accuracy score")
_ = ax.set_ylabel("Probability density")

# %%
# 随机数据
# ^^^^^^^^^^^
#
# 下面我们绘制了随机数据的空假设分布。排列得分与原始鸢尾花数据的得分相似，
# 因为排列会破坏任何可能存在的特征与标签的依赖关系。然而，这种情况下原始随机数据的得分
# 非常低。这导致了一个较大的 p 值，证实原始数据中不存在特征与标签的依赖关系。

fig, ax = plt.subplots()

# 绘制随机排列数据的直方图，perm_scores_rand 是随机排列后的得分数据
ax.hist(perm_scores_rand, bins=20, density=True)
# 设置 X 轴的范围
ax.set_xlim(0.13)
# 在直方图上添加一个垂直虚线，表示随机排列数据的得分值 score_rand
ax.axvline(score_rand, ls="--", color="r")
# 创建一个包含得分信息的标签字符串，显示在图中特定位置
score_label = f"Score on original\ndata: {score_rand:.2f}\n(p-value: {pvalue_rand:.3f})"
ax.text(0.14, 7.5, score_label, fontsize=12)
# 设置 X 轴和 Y 轴的标签
ax.set_xlabel("Accuracy score")
ax.set_ylabel("Probability density")
# 显示图形
plt.show()

# %%
# 获取较高 p 值的另一个可能原因是分类器无法利用数据中的结构。在这种情况下，只有能够利用
# 存在的依赖关系的分类器才会有低的 p 值。在我们的例子中，数据是随机的，所有分类器的
# p 值都会很高，因为数据中没有结构存在。
#
# 最后，请注意，即使数据中只有微弱的结构，此测试也已被证明能够产生较低的 p 值 [1]_。

# .. rubric:: References
#
# .. [1] Ojala and Garriga. `Permutation Tests for Studying Classifier
#        Performance
#        <http://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf>`_. The
#        Journal of Machine Learning Research (2010) vol. 11
#
```