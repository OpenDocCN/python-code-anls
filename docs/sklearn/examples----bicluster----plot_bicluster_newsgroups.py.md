# `D:\src\scipysrc\scikit-learn\examples\bicluster\plot_bicluster_newsgroups.py`

```
"""
================================================================
Biclustering documents with the Spectral Co-clustering algorithm
================================================================

This example demonstrates the Spectral Co-clustering algorithm on the
twenty newsgroups dataset. The 'comp.os.ms-windows.misc' category is
excluded because it contains many posts containing nothing but data.

The TF-IDF vectorized posts form a word frequency matrix, which is
then biclustered using Dhillon's Spectral Co-Clustering algorithm. The
resulting document-word biclusters indicate subsets words used more
often in those subsets documents.

For a few of the best biclusters, its most common document categories
and its ten most important words get printed. The best biclusters are
determined by their normalized cut. The best words are determined by
comparing their sums inside and outside the bicluster.

For comparison, the documents are also clustered using
MiniBatchKMeans. The document clusters derived from the biclusters
achieve a better V-measure than clusters found by MiniBatchKMeans.

"""

import operator
from collections import defaultdict
from time import time

import numpy as np

from sklearn.cluster import MiniBatchKMeans, SpectralCoclustering
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import v_measure_score


def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


# exclude 'comp.os.ms-windows.misc'
categories = [
    "alt.atheism",
    "comp.graphics",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]
# Fetch the 20 newsgroups dataset excluding specified categories
newsgroups = fetch_20newsgroups(categories=categories)
# True labels for each document in the dataset
y_true = newsgroups.target

# Initialize a vectorizer that normalizes numbers and applies TF-IDF
vectorizer = NumberNormalizingVectorizer(stop_words="english", min_df=5)
# Initialize a Spectral Co-clustering object with specified parameters
cocluster = SpectralCoclustering(
    n_clusters=len(categories), svd_method="arpack", random_state=0
)
# Initialize MiniBatchKMeans clustering with specified parameters
kmeans = MiniBatchKMeans(
    n_clusters=len(categories), batch_size=20000, random_state=0, n_init=3
)

# Print status message indicating vectorization process is starting
print("Vectorizing...")
# Transform the input data (list of documents) into a TF-IDF sparse matrix
X = vectorizer.fit_transform(newsgroups.data)

# Print status message indicating co-clustering process is starting
print("Coclustering...")
start_time = time()
# 记录开始时间
cocluster.fit(X)
# 使用协同聚类算法拟合数据 X
y_cocluster = cocluster.row_labels_
# 获取协同聚类的行标签作为结果
print(
    "Done in {:.2f}s. V-measure: {:.4f}".format(
        time() - start_time, v_measure_score(y_cocluster, y_true)
    )
)
# 打印执行时间和评估分数 V-measure

print("MiniBatchKMeans...")
# 输出提示信息
start_time = time()
# 记录开始时间
y_kmeans = kmeans.fit_predict(X)
# 使用 Mini-Batch K-Means 对数据 X 进行聚类预测
print(
    "Done in {:.2f}s. V-measure: {:.4f}".format(
        time() - start_time, v_measure_score(y_kmeans, y_true)
    )
)
# 打印执行时间和评估分数 V-measure

feature_names = vectorizer.get_feature_names_out()
# 获取特征名称

document_names = list(newsgroups.target_names[i] for i in newsgroups.target)
# 从 newsgroups 的目标名称中创建文档名称列表

def bicluster_ncut(i):
    # 定义双向聚类图的归一化割
    rows, cols = cocluster.get_indices(i)
    # 获取第 i 个双向聚类的行和列的索引
    if not (np.any(rows) and np.any(cols)):
        import sys
        return sys.float_info.max
    # 如果行或列索引为空，则返回最大浮点数
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]
    # 计算行和列的补集
    # 注意：以下操作与 X[rows[:, np.newaxis], cols].sum() 相同，但在 scipy <= 0.16 中更快
    weight = X[rows][:, cols].sum()
    # 计算权重
    cut = X[row_complement][:, cols].sum() + X[rows][:, col_complement].sum()
    # 计算割集
    return cut / weight
    # 返回归一化割比率

def most_common(d):
    """Items of a defaultdict(int) with the highest values.

    Like Counter.most_common in Python >=2.7.
    """
    return sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    # 返回值最高的 defaultdict(int) 中的项目

bicluster_ncuts = list(bicluster_ncut(i) for i in range(len(newsgroups.target_names)))
# 创建双向聚类的归一化割列表
best_idx = np.argsort(bicluster_ncuts)[:5]
# 获取归一化割最佳索引的前五个

print()
print("Best biclusters:")
print("----------------")
# 输出提示信息

for idx, cluster in enumerate(best_idx):
    # 遍历最佳索引和集群
    n_rows, n_cols = cocluster.get_shape(cluster)
    # 获取集群形状
    cluster_docs, cluster_words = cocluster.get_indices(cluster)
    # 获取集群的文档和单词的索引

    if not len(cluster_docs) or not len(cluster_words):
        continue
    # 如果文档或单词长度为零，则继续下一次循环

    counter = defaultdict(int)
    # 创建一个默认字典计数器
    for i in cluster_docs:
        counter[document_names[i]] += 1
    # 计算文档名称的计数器

    cat_string = ", ".join(
        "{:.0f}% {}".format(float(c) / n_rows * 100, name)
        for name, c in most_common(counter)[:3]
    )
    # 创建分类字符串

    out_of_cluster_docs = cocluster.row_labels_ != cluster
    # 获取不在集群中的文档
    out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
    # 获取不在集群中的文档索引

    word_col = X[:, cluster_words]
    # 获取列的单词
    word_scores = np.array(
        word_col[cluster_docs, :].sum(axis=0)
        - word_col[out_of_cluster_docs, :].sum(axis=0)
    )
    # 计算单词分数
    word_scores = word_scores.ravel()
    # 将单词分数转为一维数组

    important_words = list(
        feature_names[cluster_words[i]] for i in word_scores.argsort()[:-11:-1]
    )
    # 获取重要单词列表

    print("bicluster {} : {} documents, {} words".format(idx, n_rows, n_cols))
    print("categories   : {}".format(cat_string))
    print("words        : {}\n".format(", ".join(important_words)))
    # 打印双向聚类、文档和单词的信息
```