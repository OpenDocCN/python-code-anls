# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_adjusted_for_chance_measures.py`

```
"""
==========================================================
Adjustment for chance in clustering performance evaluation
==========================================================
This notebook explores the impact of uniformly-distributed random labeling on
the behavior of some clustering evaluation metrics. For such purpose, the
metrics are computed with a fixed number of samples and as a function of the number
of clusters assigned by the estimator. The example is divided into two
experiments:

- a first experiment with fixed "ground truth labels" (and therefore fixed
  number of classes) and randomly "predicted labels";
- a second experiment with varying "ground truth labels", randomly "predicted
  labels". The "predicted labels" have the same number of classes and clusters
  as the "ground truth labels".
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Defining the list of metrics to evaluate
# ----------------------------------------

# Clustering algorithms are fundamentally unsupervised learning methods.
# However, since we assign class labels for the synthetic clusters in this
# example, it is possible to use evaluation metrics that leverage this
# "supervised" ground truth information to quantify the quality of the resulting
# clusters. Examples of such metrics are the following:

# - V-measure, the harmonic mean of completeness and homogeneity;
#
# - Rand index, which measures how frequently pairs of data points are grouped
#   consistently according to the result of the clustering algorithm and the
#   ground truth class assignment;
#
# - Adjusted Rand index (ARI), a chance-adjusted Rand index such that a random
#   cluster assignment has an ARI of 0.0 in expectation;
#
# - Mutual Information (MI) is an information theoretic measure that quantifies
#   how dependent are the two labelings. Note that the maximum value of MI for
#   perfect labelings depends on the number of clusters and samples;
#
# - Normalized Mutual Information (NMI), a Mutual Information defined between 0
#   (no mutual information) in the limit of large number of data points and 1
#   (perfectly matching label assignments, up to a permutation of the labels).
#   It is not adjusted for chance: then the number of clustered data points is
#   not large enough, the expected values of MI or NMI for random labelings can
#   be significantly non-zero;
#
# - Adjusted Mutual Information (AMI), a chance-adjusted Mutual Information.
#   Similarly to ARI, random cluster assignment has an AMI of 0.0 in
#   expectation.
#
# For more information, see the :ref:`clustering_evaluation` module.

from sklearn import metrics

score_funcs = [
    ("V-measure", metrics.v_measure_score),
    ("Rand index", metrics.rand_score),
    ("ARI", metrics.adjusted_rand_score),
    ("MI", metrics.mutual_info_score),
    ("NMI", metrics.normalized_mutual_info_score),
    ("AMI", metrics.adjusted_mutual_info_score),
]

# %%
# First experiment: fixed ground truth labels and growing number of clusters
# --------------------------------------------------------------------------
#
# We first define a function that creates uniformly-distributed random labeling.
import numpy as np

# 创建一个随机数生成器对象，种子值为0
rng = np.random.RandomState(0)

# 定义函数，生成指定数量的随机标签，标签均匀分布在0到n_classes之间
def random_labels(n_samples, n_classes):
    return rng.randint(low=0, high=n_classes, size=n_samples)

# %%
# Another function will use the `random_labels` function to create a fixed set
# of ground truth labels (`labels_a`) distributed in `n_classes` and then score
# several sets of randomly "predicted" labels (`labels_b`) to assess the
# variability of a given metric at a given `n_clusters`.

def fixed_classes_uniform_labelings_scores(
    score_func, n_samples, n_clusters_range, n_classes, n_runs=5
):
    # 创建一个二维数组来存储得分，每行对应一个n_clusters值，每列对应一次运行
    scores = np.zeros((len(n_clusters_range), n_runs))
    # 生成固定数量的随机真实标签
    labels_a = random_labels(n_samples=n_samples, n_classes=n_classes)

    # 遍历不同的n_clusters值和多次运行
    for i, n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            # 生成随机预测标签
            labels_b = random_labels(n_samples=n_samples, n_classes=n_clusters)
            # 计算得分并存储
            scores[i, j] = score_func(labels_a, labels_b)
    return scores

# %%
# In this first example we set the number of classes (true number of clusters) to
# `n_classes=10`. The number of clusters varies over the values provided by
# `n_clusters_range`.

import matplotlib.pyplot as plt
import seaborn as sns

# 定义样本数量、类别数量和聚类数范围
n_samples = 1000
n_classes = 10
n_clusters_range = np.linspace(2, 100, 10).astype(int)
plots = []
names = []

# 设置调色板为色盲模式的颜色
sns.color_palette("colorblind")
# 创建图表对象
plt.figure(1)

# 遍历评分函数列表，每个评分函数使用不同的标记和名称
for marker, (score_name, score_func) in zip("d^vx.,", score_funcs):
    # 计算固定类别的均匀标签得分
    scores = fixed_classes_uniform_labelings_scores(
        score_func, n_samples, n_clusters_range, n_classes=n_classes
    )
    # 绘制误差线图，并将绘图对象添加到列表
    plots.append(
        plt.errorbar(
            n_clusters_range,
            scores.mean(axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=1,
            marker=marker,
        )[0]
    )
    # 添加评分函数的名称
    names.append(score_name)

# 设置图表标题和坐标轴标签
plt.title(
    "Clustering measures for random uniform labeling\n"
    f"against reference assignment with {n_classes} classes"
)
plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
plt.ylabel("Score value")
plt.ylim(bottom=-0.05, top=1.05)
# 添加图例
plt.legend(plots, names, bbox_to_anchor=(0.5, 0.5))
plt.show()

# %%
# The Rand index saturates for `n_clusters` > `n_classes`. Other non-adjusted
# measures such as the V-Measure show a linear dependency between the number of
# clusters and the number of samples.
#
# Adjusted for chance measure, such as ARI and AMI, display some random
# variations centered around a mean score of 0.0, independently of the number of
# samples and clusters.
#
# Second experiment: varying number of classes and clusters
# ---------------------------------------------------------
#
# In this section we define a similar function that uses several metrics to
# 计算两个均匀分布随机标签的分数。在这种情况下，对于每个可能的 `n_clusters_range` 值，类的数量和分配的聚类数是匹配的。

def uniform_labelings_scores(score_func, n_samples, n_clusters_range, n_runs=5):
    scores = np.zeros((len(n_clusters_range), n_runs))  # 创建一个二维数组，用于存储分数

    for i, n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            # 生成两组随机标签，每组包含 `n_samples` 个样本，`n_clusters` 个类别
            labels_a = random_labels(n_samples=n_samples, n_classes=n_clusters)
            labels_b = random_labels(n_samples=n_samples, n_classes=n_clusters)
            # 计算两组标签之间的分数并存储
            scores[i, j] = score_func(labels_a, labels_b)
    return scores  # 返回所有运行的分数数组


# %%
# 在这种情况下，我们使用 `n_samples=100` 来展示类似或等于样本数的聚类数量的效果。

n_samples = 100
n_clusters_range = np.linspace(2, n_samples, 10).astype(int)  # 创建一个包含整数聚类数量的数组

plt.figure(2)  # 创建一个新的图形对象

plots = []  # 用于存储图表的句柄
names = []  # 用于存储图例的名称

# 遍历每个评分函数及其名称
for marker, (score_name, score_func) in zip("d^vx.,", score_funcs):
    # 计算使用当前评分函数得到的分数
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    # 绘制误差条形图，并将返回的图表对象添加到 `plots` 中
    plots.append(
        plt.errorbar(
            n_clusters_range,
            np.median(scores, axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=2,
            marker=marker,
        )[0]
    )
    names.append(score_name)  # 将评分函数名称添加到 `names` 列表中作为图例名称

plt.title(
    "Clustering measures for 2 random uniform labelings\nwith equal number of clusters"
)  # 设置图表标题
plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")  # 设置 x 轴标签
plt.ylabel("Score value")  # 设置 y 轴标签
plt.legend(plots, names)  # 添加图例
plt.ylim(bottom=-0.05, top=1.05)  # 设置 y 轴的范围
plt.show()  # 显示图表

# %%
# 我们观察到与第一个实验类似的结果：调整后的随机标签的度量值保持在接近零的水平，而其他度量值在标签细化时 tend to get larger。
# 当聚类数接近用于计算度量值的样本总数时，随机标签的平均 V-measure 明显增加。
# 此外，原始的互信息不受上限限制，其大小取决于聚类问题的维度和地面真相类别的基数。
# 这就是为什么曲线会超出图表的范围。

# 因此，只有调整后的度量才能安全地用作评估给定 k 值的聚类算法在数据集的各个重叠子样本上的平均稳定性的共识指数。

# 因此，非调整的聚类评估度量可能会误导，因为它们对于细粒度标签输出较大的值，可能会让人误以为标签捕获了有意义的组。
# 特别地，这些非调整的度量不应用于比较输出不同聚类算法的结果，这些算法输出不同数量的聚类。
```