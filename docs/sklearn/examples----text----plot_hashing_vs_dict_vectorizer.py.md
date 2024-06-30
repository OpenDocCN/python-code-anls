# `D:\src\scipysrc\scikit-learn\examples\text\plot_hashing_vs_dict_vectorizer.py`

```
```python`
"""
===========================================
FeatureHasher and DictVectorizer Comparison
===========================================

In this example we illustrate text vectorization, which is the process of
representing non-numerical input data (such as dictionaries or text documents)
as vectors of real numbers.

We first compare :func:`~sklearn.feature_extraction.FeatureHasher` and
:func:`~sklearn.feature_extraction.DictVectorizer` by using both methods to
vectorize text documents that are preprocessed (tokenized) with the help of a
custom Python function.

Later we introduce and analyze the text-specific vectorizers
:func:`~sklearn.feature_extraction.text.HashingVectorizer`,
:func:`~sklearn.feature_extraction.text.CountVectorizer` and
:func:`~sklearn.feature_extraction.text.TfidfVectorizer` that handle both the
tokenization and the assembling of the feature matrix within a single class.

The objective of the example is to demonstrate the usage of text vectorization
API and to compare their processing time. See the example scripts
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
and :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py` for actual
learning on text documents.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Load Data
# ---------
#
# We load data from :ref:`20newsgroups_dataset`, which comprises around
# 18000 newsgroups posts on 20 topics split in two subsets: one for training and
# one for testing. For the sake of simplicity and reducing the computational
# cost, we select a subset of 7 topics and use the training set only.

from sklearn.datasets import fetch_20newsgroups

# 定义我们感兴趣的类别
categories = [
    "alt.atheism",
    "comp.graphics",
    "comp.sys.ibm.pc.hardware",
    "misc.forsale",
    "rec.autos",
    "sci.space",
    "talk.religion.misc",
]

# 输出加载数据的消息
print("Loading 20 newsgroups training data")
# 从 20 newsgroups 数据集中加载训练数据，返回特征数据和标签
raw_data, _ = fetch_20newsgroups(subset="train", categories=categories, return_X_y=True)
# 计算数据大小，单位为MB
data_size_mb = sum(len(s.encode("utf-8")) for s in raw_data) / 1e6
# 输出数据集的文档数量和数据大小
print(f"{len(raw_data)} documents - {data_size_mb:.3f}MB")

# %%
# Define preprocessing functions
# ------------------------------
#
# A token may be a word, part of a word or anything comprised between spaces or
# symbols in a string. Here we define a function that extracts the tokens using
# a simple regular expression (regex) that matches Unicode word characters. This
# includes most characters that can be part of a word in any language, as well
# as numbers and the underscore:

import re

# 定义分词函数
def tokenize(doc):
    """Extract tokens from doc.

    This uses a simple regex that matches word characters to break strings
    into tokens. For a more principled approach, see CountVectorizer or
    TfidfVectorizer.
    """
    # 使用正则表达式提取单词字符，分割字符串为 tokens
    return (tok.lower() for tok in re.findall(r"\w+", doc))

# 测试分词函数，打印分词结果
list(tokenize("This is a simple example, isn't it?"))

# %%
# We define an additional function that counts the (frequency of) occurrence of
# each token in a given document. It returns a frequency dictionary to be used
# by the vectorizers.
from collections import defaultdict

def token_freqs(doc):
    """Extract a dict mapping tokens from doc to their occurrences."""
    # Initialize a defaultdict to store token frequencies
    freq = defaultdict(int)
    # Tokenize the document and count occurrences of each token
    for tok in tokenize(doc):
        freq[tok] += 1
    return freq

token_freqs("That is one example, but this is another one")

# %%
# Observe in particular that the repeated token `"is"` is counted twice for
# instance.
#
# Breaking a text document into word tokens, potentially losing the order
# information between the words in a sentence is often called a `Bag of Words
# representation <https://en.wikipedia.org/wiki/Bag-of-words_model>`_.

# %%
# DictVectorizer
# --------------
#
# First we benchmark the :func:`~sklearn.feature_extraction.DictVectorizer`,
# then we compare it to :func:`~sklearn.feature_extraction.FeatureHasher` as
# both of them receive dictionaries as input.

from time import time
from sklearn.feature_extraction import DictVectorizer

dict_count_vectorizers = defaultdict(list)

t0 = time()
# Create an instance of DictVectorizer
vectorizer = DictVectorizer()
# Fit and transform the token frequencies into a sparse matrix
vectorizer.fit_transform(token_freqs(d) for d in raw_data)
duration = time() - t0
# Store the vectorizer's class name and processing speed
dict_count_vectorizers["vectorizer"].append(
    vectorizer.__class__.__name__ + "\non freq dicts"
)
dict_count_vectorizers["speed"].append(data_size_mb / duration)
print(f"done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s")
print(f"Found {len(vectorizer.get_feature_names_out())} unique terms")

# %%
# The actual mapping from text token to column index is explicitly stored in
# the `.vocabulary_` attribute which is a potentially very large Python
# dictionary:
type(vectorizer.vocabulary_)

# %%
len(vectorizer.vocabulary_)

# %%
vectorizer.vocabulary_["example"]

# %%
# FeatureHasher
# -------------
#
# Dictionaries take up a large amount of storage space and grow in size as the
# training set grows. Instead of growing the vectors along with a dictionary,
# feature hashing builds a vector of pre-defined length by applying a hash
# function `h` to the features (e.g., tokens), then using the hash values
# directly as feature indices and updating the resulting vector at those
# indices. When the feature space is not large enough, hashing functions tend to
# map distinct values to the same hash code (hash collisions). As a result, it
# is impossible to determine what object generated any particular hash code.
#
# Because of the above it is impossible to recover the original tokens from the
# feature matrix and the best approach to estimate the number of unique terms in
# the original dictionary is to count the number of active columns in the
# encoded feature matrix. For such a purpose we define the following function:

import numpy as np

def n_nonzero_columns(X):
    """Number of columns with at least one non-zero value in a CSR matrix."""
    # 计算在使用FeatureHasher时活跃的特征列数量，即非零元素所对应的唯一列索引数目
    """
    This is useful to count the number of features columns that are effectively
    active when using the FeatureHasher.
    """
    return len(np.unique(X.nonzero()[1]))
# %%
# 默认情况下，`sklearn.feature_extraction.FeatureHasher` 的特征数是 2**20。
# 这里我们设置 `n_features = 2**18` 来演示哈希冲突的情况。
#
# **在频率字典上使用 FeatureHasher**

from sklearn.feature_extraction import FeatureHasher

# 记录开始时间
t0 = time()
# 创建 FeatureHasher 对象，设定特征数为 2**18
hasher = FeatureHasher(n_features=2**18)
# 使用 hasher 对象转换原始数据 raw_data 中的每个文档的词频字典
X = hasher.transform(token_freqs(d) for d in raw_data)
# 计算运行时间
duration = time() - t0
# 将运行速度存入字典中
dict_count_vectorizers["vectorizer"].append(
    hasher.__class__.__name__ + "\non freq dicts"
)
# 计算速度并存入字典中
dict_count_vectorizers["speed"].append(data_size_mb / duration)
# 打印运行时间和速度信息
print(f"done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s")
print(f"Found {n_nonzero_columns(X)} unique tokens")

# %%
# 使用 :func:`~sklearn.feature_extraction.FeatureHasher` 时得到的唯一标记数量比
# 使用 :func:`~sklearn.feature_extraction.DictVectorizer` 得到的数量要少。
# 这是因为哈希冲突的存在。
#
# 通过增加特征空间可以减少冲突的数量。
# 注意，设置较大的特征数并不显著改变向量化器的速度，尽管会导致更大的系数维度，
# 并且需要更多内存来存储这些系数，即使其中大多数是非活跃的。

t0 = time()
# 创建 FeatureHasher 对象，设定特征数为 2**22
hasher = FeatureHasher(n_features=2**22)
# 使用 hasher 对象转换原始数据 raw_data 中的每个文档的词频字典
X = hasher.transform(token_freqs(d) for d in raw_data)
# 计算运行时间
duration = time() - t0

# 打印运行时间信息
print(f"done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s")
print(f"Found {n_nonzero_columns(X)} unique tokens")

# %%
# 我们确认使用 :func:`~sklearn.feature_extraction.FeatureHasher` 时得到的唯一标记数量
# 接近使用 :func:`~sklearn.feature_extraction.DictVectorizer` 得到的数量。
#
# **在原始标记上使用 FeatureHasher**
#
# 或者，可以在 :func:`~sklearn.feature_extraction.FeatureHasher` 中设置 `input_type="string"`，
# 以直接从定制的 `tokenize` 函数输出的字符串进行向量化。
# 这相当于为每个特征名称传递一个隐含的频率为 1 的字典。

t0 = time()
# 创建 FeatureHasher 对象，设定特征数为 2**18，并设置 input_type="string"
hasher = FeatureHasher(n_features=2**18, input_type="string")
# 使用 hasher 对象转换原始数据 raw_data 中的每个文档的标记化结果
X = hasher.transform(tokenize(d) for d in raw_data)
# 计算运行时间
duration = time() - t0
# 将运行速度存入字典中
dict_count_vectorizers["vectorizer"].append(
    hasher.__class__.__name__ + "\non raw tokens"
)
# 计算速度并存入字典中
dict_count_vectorizers["speed"].append(data_size_mb / duration)
# 打印运行时间和速度信息
print(f"done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s")
print(f"Found {n_nonzero_columns(X)} unique tokens")

# %%
# 现在我们绘制上述向量化方法的速度。

import matplotlib.pyplot as plt

# 创建绘图对象和轴对象
fig, ax = plt.subplots(figsize=(12, 6))

# 根据向量化方法的数量，创建纵向条形图
y_pos = np.arange(len(dict_count_vectorizers["vectorizer"]))
ax.barh(y_pos, dict_count_vectorizers["speed"], align="center")
ax.set_yticks(y_pos)
ax.set_yticklabels(dict_count_vectorizers["vectorizer"])
ax.invert_yaxis()
_ = ax.set_xlabel("speed (MB/s)")

# %%
# 在这两种情况下，:func:`~sklearn.feature_extraction.FeatureHasher` 大约快两倍
# ```
# 导入所需的文本特征提取工具包中的 CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# 记录开始时间
t0 = time()
# 创建 CountVectorizer 对象
vectorizer = CountVectorizer()
# 对原始数据进行拟合和转换
vectorizer.fit_transform(raw_data)
# 计算处理时间
duration = time() - t0
# 将 CountVectorizer 对象的类名和速度信息添加到字典中
dict_count_vectorizers["vectorizer"].append(vectorizer.__class__.__name__)
dict_count_vectorizers["speed"].append(data_size_mb / duration)
# 打印处理时间和速度信息
print(f"done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s")
# 打印找到的唯一术语数目
print(f"Found {len(vectorizer.get_feature_names_out())} unique terms")



# %%
# 我们可以看到使用 CountVectorizer 实现大约比使用 DictVectorizer 结合简单函数映射的方法快两倍。这是因为
# CountVectorizer 通过在整个训练集上重复使用编译过的正则表达式进行了优化，而不是像我们的简单分词函数那样每个文档都创建一个。



# 导入 HashingVectorizer 从文本特征提取工具包中
from sklearn.feature_extraction.text import HashingVectorizer

# 记录开始时间
t0 = time()
# 创建 HashingVectorizer 对象，设置特征数为 2**18
vectorizer = HashingVectorizer(n_features=2**18)
# 对原始数据进行拟合和转换
vectorizer.fit_transform(raw_data)
# 计算处理时间
duration = time() - t0
# 将 HashingVectorizer 对象的类名和速度信息添加到字典中
dict_count_vectorizers["vectorizer"].append(vectorizer.__class__.__name__)
dict_count_vectorizers["speed"].append(data_size_mb / duration)
# 打印处理时间和速度信息
print(f"done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s")



# %%
# 我们可以观察到这是目前最快的文本标记化策略，
# assuming that the downstream machine learning task can tolerate a few
# collisions.
#
# TfidfVectorizer
# ---------------
#
# In a large text corpus, some words appear with higher frequency (e.g. "the",
# "a", "is" in English) and do not carry meaningful information about the actual
# contents of a document. If we were to feed the word count data directly to a
# classifier, those very common terms would shadow the frequencies of rarer yet
# more informative terms. In order to re-weight the count features into floating
# point values suitable for usage by a classifier it is very common to use the
# tf-idf transform as implemented by the
# :func:`~sklearn.feature_extraction.text.TfidfTransformer`. TF stands for
# "term-frequency" while "tf-idf" means term-frequency times inverse
# document-frequency.
#
# We now benchmark the :func:`~sklearn.feature_extraction.text.TfidfVectorizer`,
# which is equivalent to combining the tokenization and occurrence counting of
# the :func:`~sklearn.feature_extraction.text.CountVectorizer` along with the
# normalizing and weighting from a
# :func:`~sklearn.feature_extraction.text.TfidfTransformer`.
from sklearn.feature_extraction.text import TfidfVectorizer

# Measure the start time for benchmarking
t0 = time()
# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit the vectorizer to the raw data and transform it
vectorizer.fit_transform(raw_data)
# Calculate the duration of the vectorization process
duration = time() - t0
# Append the vectorizer class name to the dictionary for tracking
dict_count_vectorizers["vectorizer"].append(vectorizer.__class__.__name__)
# Append the speed (data processing rate) to the dictionary
dict_count_vectorizers["speed"].append(data_size_mb / duration)
# Print the duration and processing speed
print(f"done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s")
# Print the number of unique terms found by the vectorizer
print(f"Found {len(vectorizer.get_feature_names_out())} unique terms")

# %%
# Summary
# -------
# Let's conclude this notebook by summarizing all the recorded processing speeds
# in a single plot:
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

# Create horizontal bars for each vectorizer's speed
y_pos = np.arange(len(dict_count_vectorizers["vectorizer"]))
ax.barh(y_pos, dict_count_vectorizers["speed"], align="center")
ax.set_yticks(y_pos)
ax.set_yticklabels(dict_count_vectorizers["vectorizer"])
ax.invert_yaxis()
_ = ax.set_xlabel("speed (MB/s)")

# %%
# Notice from the plot that
# :func:`~sklearn.feature_extraction.text.TfidfVectorizer` is slightly slower
# than :func:`~sklearn.feature_extraction.text.CountVectorizer` because of the
# extra operation induced by the
# :func:`~sklearn.feature_extraction.text.TfidfTransformer`.
#
# Also notice that, by setting the number of features `n_features = 2**18`, the
# :func:`~sklearn.feature_extraction.text.HashingVectorizer` performs better
# than the :func:`~sklearn.feature_extraction.text.CountVectorizer` at the
# expense of inversibility of the transformation due to hash collisions.
#
# We highlight that :func:`~sklearn.feature_extraction.text.CountVectorizer` and
# :func:`~sklearn.feature_extraction.text.HashingVectorizer` perform better than
# their equivalent :func:`~sklearn.feature_extraction.DictVectorizer` and
# :func:`~sklearn.feature_extraction.FeatureHasher` on manually tokenized
# documents since the internal tokenization step of the former vectorizers
# 一次编译正则表达式，然后在所有文档中重复使用它。
```