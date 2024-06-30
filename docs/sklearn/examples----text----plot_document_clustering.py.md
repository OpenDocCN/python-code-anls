# `D:\src\scipysrc\scikit-learn\examples\text\plot_document_clustering.py`

```
"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn API can be used to cluster
documents by topics using a `Bag of Words approach
<https://en.wikipedia.org/wiki/Bag-of-words_model>`_.

Two algorithms are demonstrated, namely :class:`~sklearn.cluster.KMeans` and its more
scalable variant, :class:`~sklearn.cluster.MiniBatchKMeans`. Additionally,
latent semantic analysis is used to reduce dimensionality and discover latent
patterns in the data.

This example uses two different text vectorizers: a
:class:`~sklearn.feature_extraction.text.TfidfVectorizer` and a
:class:`~sklearn.feature_extraction.text.HashingVectorizer`. See the example
notebook :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`
for more information on vectorizers and a comparison of their processing times.

For document analysis via a supervised learning approach, see the example script
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Loading text data
# =================
#
# We load data from :ref:`20newsgroups_dataset`, which comprises around 18,000
# newsgroups posts on 20 topics. For illustrative purposes and to reduce the
# computational cost, we select a subset of 4 topics only accounting for around
# 3,400 documents. See the example
# :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
# to gain intuition on the overlap of such topics.
#
# Notice that, by default, the text samples contain some message metadata such
# as `"headers"`, `"footers"` (signatures) and `"quotes"` to other posts. We use
# the `remove` parameter from :func:`~sklearn.datasets.fetch_20newsgroups` to
# strip those features and have a more sensible clustering problem.

import numpy as np

from sklearn.datasets import fetch_20newsgroups

# Define the categories of interest for the newsgroups dataset
categories = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]

# Fetch the 20newsgroups dataset, specifying to remove headers, footers, and quotes
dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    categories=categories,
    shuffle=True,
    random_state=42,
)

# Extract the labels from the dataset
labels = dataset.target

# Calculate unique labels and their counts to determine the number of clusters
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

# Print the number of documents and categories found in the dataset
print(f"{len(dataset.data)} documents - {true_k} categories")

# %%
# Quantifying the quality of clustering results
# =============================================
#
# In this section we define a function to score different clustering pipelines
# using several metrics.
#
# Clustering algorithms are fundamentally unsupervised learning methods.
# However, since we happen to have class labels for this specific dataset, it is
# possible to use evaluation metrics that leverage this "supervised" ground
# truth information to quantify the quality of the resulting clusters. Examples
# 导入必要的库
from collections import defaultdict
from time import time
from sklearn import metrics

# 初始化评估结果存储列表
evaluations = []
evaluations_std = []

# 定义函数，执行K-means聚类并评估性能
def fit_and_evaluate(km, X, name=None, n_runs=5):
    # 如果未指定名称，则使用K-means对象的类名作为名称
    name = km.__class__.__name__ if name is None else name

    # 存储每次运行的训练时间和评分
    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        # 设置随机种子并记录开始时间
        km.set_params(random_state=seed)
        t0 = time()
        # 执行K-means聚类
        km.fit(X)
        # 记录训练时间
        train_times.append(time() - t0)
        # 计算并记录聚类性能指标
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    train_times = np.asarray(train_times)

    # 输出平均训练时间和标准差
    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    # 准备评估结果
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    # 输出每个评分指标的平均值和标准差
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    # 将评估结果和标准差存入列表
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)
# - :class:`~sklearn.feature_extraction.text.HashingVectorizer` hashes word
#   occurrences to a fixed dimensional space, possibly with collisions. The word
#   count vectors are then normalized to each have l2-norm equal to one
#   (projected to the euclidean unit-sphere) which seems to be important for
#   k-means to work in high dimensional space.

# Furthermore it is possible to post-process those extracted features using
# dimensionality reduction. We will explore the impact of those choices on the
# clustering quality in the following.

# Feature Extraction using TfidfVectorizer
# ----------------------------------------

# We first benchmark the estimators using a dictionary vectorizer along with an
# IDF normalization as provided by
# :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_df=0.5,           # 忽略出现在超过50%文档中的词语
    min_df=5,             # 忽略在少于5个文档中出现的词语
    stop_words="english", # 使用英语停用词列表过滤词语
)
t0 = time()               # 记录开始时间
X_tfidf = vectorizer.fit_transform(dataset.data)  # 对数据集进行TF-IDF向量化

print(f"vectorization done in {time() - t0:.3f} s")  # 打印向量化所需时间
print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")  # 打印样本数和特征数

# %%
# After ignoring terms that appear in more than 50% of the documents (as set by
# `max_df=0.5`) and terms that are not present in at least 5 documents (set by
# `min_df=5`), the resulting number of unique terms `n_features` is around
# 8,000. We can additionally quantify the sparsity of the `X_tfidf` matrix as
# the fraction of non-zero entries divided by the total number of elements.

print(f"{X_tfidf.nnz / np.prod(X_tfidf.shape):.3f}")  # 计算并打印TF-IDF矩阵的稀疏度

# %%
# We find that around 0.7% of the entries of the `X_tfidf` matrix are non-zero.

# .. _kmeans_sparse_high_dim:
#
# Clustering sparse data with k-means
# -----------------------------------

# As both :class:`~sklearn.cluster.KMeans` and
# :class:`~sklearn.cluster.MiniBatchKMeans` optimize a non-convex objective
# function, their clustering is not guaranteed to be optimal for a given random
# init. Even further, on sparse high-dimensional data such as text vectorized
# using the Bag of Words approach, k-means can initialize centroids on extremely
# isolated data points. Those data points can stay their own centroids all
# along.

# The following code illustrates how the previous phenomenon can sometimes lead
# to highly imbalanced clusters, depending on the random initialization:

from sklearn.cluster import KMeans

for seed in range(5):  # 针对不同的随机种子执行以下聚类过程
    kmeans = KMeans(
        n_clusters=true_k,   # 设置聚类簇的数量
        max_iter=100,        # 设置最大迭代次数
        n_init=1,            # 设置初始化的次数
        random_state=seed,   # 设置随机种子
    ).fit(X_tfidf)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)  # 获取每个簇的元素数量
    print(f"Number of elements assigned to each cluster: {cluster_sizes}")  # 打印每个簇的元素分布
print()
print(
    "True number of documents in each category according to the class labels: "
    # 根据类别标签，打印每个类别中真实的文档数量
    f"{category_sizes}"


    # 将变量 category_sizes 转换为字符串，并使用 f-string 格式化
# %%
# 为了避免这个问题，一种可能性是增加具有独立随机初始化的运行次数 `n_init`。
# 在这种情况下，将选择具有最佳惯性（k-means的目标函数）的聚类结果。

kmeans = KMeans(
    n_clusters=true_k,   # 设置聚类的数目为 true_k
    max_iter=100,         # 最大迭代次数为100
    n_init=5,             # 运行k-means算法的次数，选择惯性最小的结果
)

fit_and_evaluate(kmeans, X_tfidf, name="KMeans\non tf-idf vectors")

# %%
# 所有这些聚类评估指标的最大值为1.0（表示完美的聚类结果）。值越高越好。
# 调整兰德指数接近0.0表示随机标记。从上面的分数可以看出，聚类分配确实
# 明显高于偶然水平，但整体质量肯定还有提升空间。
#
# 请注意，类标签可能不准确反映文档主题，因此使用标签的度量标准并不一定是
# 评估我们聚类流程质量的最佳选择。
#
# 使用LSA进行降维
# ---------------------------------------------
#
# 即使在将向量化空间的维度降低以使k-means更加稳定的情况下，`n_init=1`仍然
# 可以使用。为此，我们使用 :class:`~sklearn.decomposition.TruncatedSVD`，
# 它适用于词项计数/TF-IDF矩阵。由于SVD结果未经归一化，我们重新进行归一化，
# 以改善 :class:`~sklearn.cluster.KMeans` 的结果。在信息检索和文本挖掘文献中，
# 使用SVD来降低TF-IDF文档向量的维度通常称为 `潜在语义分析 <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`_（LSA）。

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# 创建一个包含降维和归一化步骤的pipeline
lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
# 对TF-IDF向量进行LSA降维处理
X_lsa = lsa.fit_transform(X_tfidf)
# 解释方差，表示SVD步骤的方差解释比例之和
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

# %%
# 使用单个初始化意味着处理时间将减少，无论是对 :class:`~sklearn.cluster.KMeans`
# 还是 :class:`~sklearn.cluster.MiniBatchKMeans` 都是如此。

kmeans = KMeans(
    n_clusters=true_k,   # 设置聚类的数目为 true_k
    max_iter=100,         # 最大迭代次数为100
    n_init=1,             # 仅运行一次k-means算法
)

fit_and_evaluate(kmeans, X_lsa, name="KMeans\nwith LSA on tf-idf vectors")

# %%
# 我们可以观察到，在文档的LSA表示上进行聚类显著更快（因为 `n_init=1` 和 LSA特征空间的维度远低于原始向量空间）。
# 此外，所有聚类评估指标都有所改善。我们使用 :class:`~sklearn.cluster.MiniBatchKMeans` 重复实验。

from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(
    n_clusters=true_k,   # 设置聚类的数目为 true_k
    n_init=1,             # 仅运行一次MiniBatchKMeans算法
    init_size=1000,       # 初始样本大小
    batch_size=1000,      # 小批量K-means的批处理大小
)

fit_and_evaluate(
    minibatch_kmeans,
    X_lsa,
    # 定义一个字符串变量 name，赋值为 "MiniBatchKMeans\nwith LSA on tf-idf vectors"
# %%
# Top terms per cluster
# ---------------------
#
# Since :class:`~sklearn.feature_extraction.text.TfidfVectorizer` can be
# inverted we can identify the cluster centers, which provide an intuition of
# the most influential words **for each cluster**. See the example script
# :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
# for a comparison with the most predictive words **for each target class**.

original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
# 获取原始空间的聚类中心，通过LSA的逆变换得到
order_centroids = original_space_centroids.argsort()[:, ::-1]
# 对聚类中心进行排序，以获取每个聚类中心最重要的特征的索引，降序排列
terms = vectorizer.get_feature_names_out()
# 获取特征的名称列表，即词汇表

for i in range(true_k):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]} ", end="")
    print()
# 打印每个聚类的前十个最重要的特征词汇

# %%
# HashingVectorizer
# -----------------
# An alternative vectorization can be done using a
# :class:`~sklearn.feature_extraction.text.HashingVectorizer` instance, which
# does not provide IDF weighting as this is a stateless model (the fit method
# does nothing). When IDF weighting is needed it can be added by pipelining the
# :class:`~sklearn.feature_extraction.text.HashingVectorizer` output to a
# :class:`~sklearn.feature_extraction.text.TfidfTransformer` instance. In this
# case we also add LSA to the pipeline to reduce the dimension and sparcity of
# the hashed vector space.

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

lsa_vectorizer = make_pipeline(
    HashingVectorizer(stop_words="english", n_features=50_000),
    TfidfTransformer(),
    TruncatedSVD(n_components=100, random_state=0),
    Normalizer(copy=False),
)
# 创建一个包含多个步骤的管道，用于将文本转换为LSA降维后的稀疏矩阵

t0 = time()
X_hashed_lsa = lsa_vectorizer.fit_transform(dataset.data)
print(f"vectorization done in {time() - t0:.3f} s")
# 使用管道进行向量化转换，并计时向量化的时间

# %%
# One can observe that the LSA step takes a relatively long time to fit,
# especially with hashed vectors. The reason is that a hashed space is typically
# large (set to `n_features=50_000` in this example). One can try lowering the
# number of features at the expense of having a larger fraction of features with
# hash collisions as shown in the example notebook
# :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.
#
# We now fit and evaluate the `kmeans` and `minibatch_kmeans` instances on this
# hashed-lsa-reduced data:

fit_and_evaluate(kmeans, X_hashed_lsa, name="KMeans\nwith LSA on hashed vectors")
# 对使用哈希和LSA降维后的数据进行K均值聚类模型的拟合和评估

# %%
fit_and_evaluate(
    minibatch_kmeans,
    X_hashed_lsa,
    name="MiniBatchKMeans\nwith LSA on hashed vectors",
)
# 对使用哈希和LSA降维后的数据进行MiniBatchK均值聚类模型的拟合和评估

# %%
# Both methods lead to good results that are similar to running the same models
# on the traditional LSA vectors (without hashing).
#
# Clustering evaluation summary
# ==============================

import matplotlib.pyplot as plt
import pandas as pd

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 6), sharey=True)

df = pd.DataFrame(evaluations[::-1]).set_index("estimator")
df_std = pd.DataFrame(evaluations_std[::-1]).set_index("estimator")
# 创建用于存储聚类评估结果的DataFrame，并按评估器的逆序排列，并设置索引
# 从数据框中删除 "train_time" 列，并绘制水平条形图，同时显示误差条
# 参数 axis="columns" 表示按列名指定的列进行删除操作
# ax=ax0 指定了图形的绘制区域为 ax0，同时绘制条形图并设置误差条的值为 df_std
df.drop(
    ["train_time"],
    axis="columns",
).plot.barh(ax=ax0, xerr=df_std)

# 设置 ax0 图的 x 轴标签为 "Clustering scores"
ax0.set_xlabel("Clustering scores")
# 设置 ax0 图的 y 轴标签为空字符串，即不显示 y 轴标签
ax0.set_ylabel("")

# 绘制 "train_time" 列的水平条形图，同时显示误差条
# xerr=df_std["train_time"] 设置该列的误差条值为 df_std["train_time"]
df["train_time"].plot.barh(ax=ax1, xerr=df_std["train_time"])

# 设置 ax1 图的 x 轴标签为 "Clustering time (s)"
ax1.set_xlabel("Clustering time (s)")

# 调整图形布局，使得图形紧凑显示
plt.tight_layout()
```