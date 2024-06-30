# `D:\src\scipysrc\scikit-learn\examples\text\plot_document_classification_20newsgroups.py`

```
"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents by
topics using a `Bag of Words approach
<https://en.wikipedia.org/wiki/Bag-of-words_model>`_. This example uses a
Tf-idf-weighted document-term sparse matrix to encode the features and
demonstrates various classifiers that can efficiently handle sparse matrices.

For document analysis via an unsupervised learning approach, see the example
script :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


# %%
# Loading and vectorizing the 20 newsgroups text dataset
# ======================================================
#
# We define a function to load data from :ref:`20newsgroups_dataset`, which
# comprises around 18,000 newsgroups posts on 20 topics split in two subsets:
# one for training (or development) and the other one for testing (or for
# performance evaluation). Note that, by default, the text samples contain some
# message metadata such as `'headers'`, `'footers'` (signatures) and `'quotes'`
# to other posts. The `fetch_20newsgroups` function therefore accepts a
# parameter named `remove` to attempt stripping such information that can make
# the classification problem "too easy". This is achieved using simple
# heuristics that are neither perfect nor standard, hence disabled by default.

from time import time  # 导入时间模块中的时间函数

from sklearn.datasets import fetch_20newsgroups  # 导入用于获取20个新闻组数据集的函数
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入TF-IDF向量化器

categories = [  # 定义要加载的新闻组主题列表
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]


def size_mb(docs):
    """计算文本数据的总字节数，以兆字节（MB）为单位"""
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def load_dataset(verbose=False, remove=()):
    """加载并向量化20个新闻组数据集"""

    # 从训练集中获取数据
    data_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    # 从测试集中获取数据
    data_test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    # `target_names`中的标签顺序可能与`categories`中不同
    target_names = data_train.target_names

    # 将目标数据拆分为训练集和测试集
    y_train, y_test = data_train.target, data_test.target

    # 使用稀疏向量化器从训练数据中提取特征
    t0 = time()
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    X_train = vectorizer.fit_transform(data_train.data)
    duration_train = time() - t0

    # 使用相同的向量化器从测试数据中提取特征
    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    # 计算运行时长
    duration_test = time() - t0

    # 获取特征名称列表
    feature_names = vectorizer.get_feature_names_out()

    # 如果设置了详细模式
    if verbose:
        # 计算训练数据和测试数据的大小（MB）
        data_train_size_mb = size_mb(data_train.data)
        data_test_size_mb = size_mb(data_test.data)

        # 打印训练集的文档数量和大小
        print(
            f"{len(data_train.data)} documents - "
            f"{data_train_size_mb:.2f}MB (training set)"
        )
        
        # 打印测试集的文档数量和大小
        print(f"{len(data_test.data)} documents - {data_test_size_mb:.2f}MB (test set)")
        
        # 打印类别数量
        print(f"{len(target_names)} categories")
        
        # 打印训练向量化的时间和速度
        print(
            f"vectorize training done in {duration_train:.3f}s "
            f"at {data_train_size_mb / duration_train:.3f}MB/s"
        )
        
        # 打印训练集的样本数量和特征数量
        print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        
        # 打印测试向量化的时间和速度
        print(
            f"vectorize testing done in {duration_test:.3f}s "
            f"at {data_test_size_mb / duration_test:.3f}MB/s"
        )
        
        # 打印测试集的样本数量和特征数量
        print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    # 返回训练集、测试集、训练标签、测试标签、特征名称列表和类别名称列表
    return X_train, X_test, y_train, y_test, feature_names, target_names
# %%
# Analysis of a bag-of-words document classifier
# ==============================================
#
# We will now train a classifier twice, once on the text samples including
# metadata and once after stripping the metadata. For both cases we will analyze
# the classification errors on a test set using a confusion matrix and inspect
# the coefficients that define the classification function of the trained
# models.
#
# Model without metadata stripping
# --------------------------------
#
# We start by using the custom function `load_dataset` to load the data without
# metadata stripping.

X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
    verbose=True
)

# %%
# Our first model is an instance of the
# :class:`~sklearn.linear_model.RidgeClassifier` class. This is a linear
# classification model that uses the mean squared error on {-1, 1} encoded
# targets, one for each possible class. Contrary to
# :class:`~sklearn.linear_model.LogisticRegression`,
# :class:`~sklearn.linear_model.RidgeClassifier` does not
# provide probabilistic predictions (no `predict_proba` method),
# but it is often faster to train.

from sklearn.linear_model import RidgeClassifier

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# %%
# We plot the confusion matrix of this classifier to find if there is a pattern
# in the classification errors.

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)
_ = ax.set_title(
    f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
)

# %%
# The confusion matrix highlights that documents of the `alt.atheism` class are
# often confused with documents with the class `talk.religion.misc` class and
# vice-versa which is expected since the topics are semantically related.
#
# We also observe that some documents of the `sci.space` class can be misclassified as
# `comp.graphics` while the converse is much rarer. A manual inspection of those
# badly classified documents would be required to get some insights on this
# asymmetry. It could be the case that the vocabulary of the space topic could
# be more specific than the vocabulary for computer graphics.
#
# We can gain a deeper understanding of how this classifier makes its decisions
# by looking at the words with the highest average feature effects:

import numpy as np
import pandas as pd

def plot_feature_effects():
    # learned coefficients weighted by frequency of appearance
    average_feature_effects = clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
    # 对目标类别名称列表进行遍历，同时获取每个类别的平均特征影响值的排序后的前五个索引
    for i, label in enumerate(target_names):
        # 获取平均特征影响值数组中排序后的前五个索引，并反转顺序
        top5 = np.argsort(average_feature_effects[i])[-5:][::-1]
        # 如果是第一个类别，创建一个新的DataFrame对象，并设置列名为当前类别的标签
        if i == 0:
            top = pd.DataFrame(feature_names[top5], columns=[label])
            # 记录当前顶部特征的索引
            top_indices = top5
        else:
            # 否则，将当前类别的特征名称添加到已有的DataFrame对象中
            top[label] = feature_names[top5]
            # 合并当前顶部特征的索引数组
            top_indices = np.concatenate((top_indices, top5), axis=None)

    # 确保顶部特征的索引数组中不含重复元素
    top_indices = np.unique(top_indices)
    # 根据顶部特征的索引数组获取相关的预测词汇
    predictive_words = feature_names[top_indices]

    # 绘制特征影响图
    bar_size = 0.25
    padding = 0.75
    # 计算每个条形图的垂直位置
    y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)

    # 创建一个新的图形和坐标轴对象，设置图形大小
    fig, ax = plt.subplots(figsize=(10, 8))
    # 对每个类别绘制水平条形图，显示平均特征影响值
    for i, label in enumerate(target_names):
        ax.barh(
            y_locs + (i - 2) * bar_size,  # 条形图的垂直位置
            average_feature_effects[i, top_indices],  # 平均特征影响值
            height=bar_size,  # 条形图高度
            label=label,  # 类别标签
        )
    # 设置坐标轴的标签和限制
    ax.set(
        yticks=y_locs,  # 设置垂直位置的刻度
        yticklabels=predictive_words,  # 设置垂直位置刻度对应的预测词汇
        ylim=[  # 设置垂直轴的限制
            0 - 4 * bar_size,
            len(top_indices) * (4 * bar_size + padding) - 4 * bar_size,
        ],
    )
    # 在图中添加图例，并设置其位置为右下角
    ax.legend(loc="lower right")

    # 打印每个类别的前5个关键词
    print("top 5 keywords per class:")
    print(top)

    # 返回绘制的坐标轴对象
    return ax
# 设置特征效果图的标题为“原始数据上的平均特征效果”
_ = plot_feature_effects().set_title("Average feature effect on the original data")

# %%
# 我们可以观察到，最具预测性的单词通常与一个类别强烈正相关，与其他所有类别负相关。
# 大多数这些正相关性都很容易解释。然而，像 "god" 和 "people" 这样的单词，与 "talk.misc.religion"
# 和 "alt.atheism" 都呈现出正相关，因为这两个类别预料中共享一些常见词汇。
# 注意，还有一些单词如 "christian" 和 "morality" 只与 "talk.misc.religion" 正相关。
# 此外，在这个数据集版本中，"caltech" 是无神论主义的顶级预测特征之一，这是因为数据集中
# 含有某种形式的元数据，比如讨论中先前电子邮件的发送者的电子邮件地址，可以在下面看到：

data_train = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)

for doc in data_train.data:
    if "caltech" in doc:
        print(doc)
        break

# %%
# 这样的头部、签名脚注（以及来自先前消息的引用元数据）可以被视为副信息，
# 通过识别注册成员，人们更愿意让我们的文本分类器仅从每个文本文档的“主要内容”中学习，
# 而不是依赖于泄露的写作者身份。
#
# 带有元数据剥离的模型
# -----------------------------
#
# scikit-learn 中 20 newsgroups 数据集加载器的 `remove` 选项可以启用启发式方法来过滤掉
# 一些使分类问题变得人为简化的不必要元数据。请注意，这种文本内容的过滤远非完美。
#
# 让我们尝试利用此选项来训练一个文本分类器，它不太依赖这种类型的元数据来做决策：
(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    target_names,
) = load_dataset(remove=("headers", "footers", "quotes"))

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)
_ = ax.set_title(
    f"Confusion Matrix for {clf.__class__.__name__}\non filtered documents"
)

# %%
# 通过查看混淆矩阵，更明显地看出使用元数据训练的模型的分数是过于乐观的。
# 没有访问元数据的分类问题的准确性虽然较低，但更能代表预期的文本分类问题。

# 设置特征效果图的标题为“过滤后文档上的平均特征效果”
_ = plot_feature_effects().set_title("Average feature effects on filtered documents")

# %%
# In the next section we keep the dataset without metadata to compare several
# classifiers.
# %%
# Benchmarking classifiers
# ========================
#
# Scikit-learn provides many different kinds of classification algorithms. In
# this section we will train a selection of those classifiers on the same text
# classification problem and measure both their generalization performance
# (accuracy on the test set) and their computation performance (speed), both at
# training time and testing time. For such purpose we define the following
# benchmarking utilities:

from sklearn import metrics  # 导入用于评估模型性能的指标库
from sklearn.utils.extmath import density  # 导入用于计算稀疏矩阵密度的函数

def benchmark(clf, custom_name=False):
    # 打印分隔线，显示开始训练
    print("_" * 80)
    print("Training: ")
    # 打印分类器信息
    print(clf)
    # 记录训练开始时间
    t0 = time()
    # 训练分类器
    clf.fit(X_train, y_train)
    # 计算训练时间
    train_time = time() - t0
    print(f"train time: {train_time:.3}s")

    # 记录测试开始时间
    t0 = time()
    # 使用训练好的分类器进行预测
    pred = clf.predict(X_test)
    # 计算测试时间
    test_time = time() - t0
    print(f"test time:  {test_time:.3}s")

    # 计算预测准确率
    score = metrics.accuracy_score(y_test, pred)
    print(f"accuracy:   {score:.3}")

    # 如果分类器有 coef_ 属性，表示支持特征权重
    if hasattr(clf, "coef_"):
        # 打印特征维度信息
        print(f"dimensionality: {clf.coef_.shape[1]}")
        # 打印权重矩阵密度信息
        print(f"density: {density(clf.coef_)}")
        print()

    print()
    # 如果指定了自定义名称，使用自定义名称作为分类器描述
    if custom_name:
        clf_descr = str(custom_name)
    else:
        # 否则使用分类器的类名作为描述
        clf_descr = clf.__class__.__name__
    return clf_descr, score, train_time, test_time


# %%
# We now train and test the datasets with 8 different classification models and
# get performance results for each model. The goal of this study is to highlight
# the computation/accuracy tradeoffs of different types of classifiers for
# such a multi-class text classification problem.
#
# Notice that the most important hyperparameters values were tuned using a grid
# search procedure not shown in this notebook for the sake of simplicity. See
# the example script
# :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_text_feature_extraction.py`  # noqa: E501
# for a demo on how such tuning can be done.

from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.linear_model import LogisticRegression, SGDClassifier  # 导入逻辑回归和随机梯度下降分类器
from sklearn.linear_model import RidgeClassifier  # 导入岭回归分类器
from sklearn.naive_bayes import ComplementNB  # 导入补充朴素贝叶斯分类器
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # 导入k近邻和最近质心分类器
from sklearn.svm import LinearSVC  # 导入线性支持向量机分类器

results = []
for clf, name in (
    (LogisticRegression(C=5, max_iter=1000), "Logistic Regression"),
    (RidgeClassifier(alpha=1.0, solver="sparse_cg"), "Ridge Classifier"),
    (KNeighborsClassifier(n_neighbors=100), "kNN"),
    (RandomForestClassifier(), "Random Forest"),
    # L2 penalty Linear SVC
    (LinearSVC(C=0.1, dual=False, max_iter=1000), "Linear SVC"),
    # L2 penalty Linear SGD
    (
        SGDClassifier(
            loss="log_loss", alpha=1e-4, n_iter_no_change=3, early_stopping=True
        ),
        "log-loss SGD",
    ),
    # NearestCentroid (aka Rocchio classifier)
    (NearestCentroid(), "NearestCentroid"),
    # Sparse naive Bayes classifier
    (ComplementNB(), "Complement Naive Bayes"),
):
    # 循环中依次使用不同的分类器进行训练和测试
    (ComplementNB(alpha=0.1), "Complement naive Bayes"),


# 创建一个元组，包含一个ComplementNB分类器对象和一个描述字符串
(ComplementNB(alpha=0.1), "Complement naive Bayes"),


这行代码创建了一个包含两个元素的元组：第一个元素是一个ComplementNB分类器对象，使用了alpha参数设置为0.1；第二个元素是一个字符串，描述了该分类器对象的类型。
```python`
# %%
# Plot accuracy, training and test time of each classifier
# ========================================================
#
# Create scatter plots to visualize the relationship between test accuracy,
# training time, and test time for each classifier.
#
# Initialize indices for the classifiers' results array.
indices = np.arange(len(results))

# Rearrange results into separate arrays for classifier names, scores, training times, and test times.
results = [[x[i] for x in results] for i in range(4)]

# Extract classifier names, scores, training times, and test times from the results.
clf_names, score, training_time, test_time = results

# Convert training_time and test_time into numpy arrays for efficient computation.
training_time = np.array(training_time)
test_time = np.array(test_time)

# Create the first scatter plot for score vs. training time.
fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.scatter(score, training_time, s=60)
ax1.set(
    title="Score-training time trade-off",
    yscale="log",
    xlabel="test accuracy",
    ylabel="training time (s)",
)

# Create the second scatter plot for score vs. test time.
fig, ax2 = plt.subplots(figsize=(10, 8))
ax2.scatter(score, test_time, s=60)
ax2.set(
    title="Score-test time trade-off",
    yscale="log",
    xlabel="test accuracy",
    ylabel="test time (s)",
)

# Annotate each point on the scatter plots with the corresponding classifier name.
for i, txt in enumerate(clf_names):
    ax1.annotate(txt, (score[i], training_time[i]))
    ax2.annotate(txt, (score[i], test_time[i]))

# %%
# The naive Bayes model demonstrates superior trade-offs between score and
# training/testing time compared to other models. Conversely, the Random Forest
# model exhibits prolonged training durations, expensive predictions, and relatively
# inferior accuracy. These observations align with expectations in high-dimensional
# prediction scenarios, where linear models often outperform due to enhanced
# separability of features in expansive dimensions exceeding 10,000.
#
# Disparities in training speed and accuracy among linear models stem from variances
# in optimized loss functions and regularization methodologies. It's noteworthy that
# identical loss functions may yield divergent fitting times and test accuracies
# based on solver or regularization configurations.
#
# Analysis of the second plot reveals uniform prediction speeds across all trained
# linear models, attributable to shared implementation of prediction functions.
#
# KNeighborsClassifier registers diminished accuracy alongside extended testing times,
# as anticipated. The model's computational intensity derives from pairwise distance
# computations between test samples and training set documents, compounded by the
# "curse of dimensionality" effects in text classification tasks.
```