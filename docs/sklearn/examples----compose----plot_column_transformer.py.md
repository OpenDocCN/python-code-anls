# `D:\src\scipysrc\scikit-learn\examples\compose\plot_column_transformer.py`

```
##############################################################################
# 20 newsgroups dataset
# ---------------------
#
# We will use the :ref:`20 newsgroups dataset <20newsgroups_dataset>`, which
# comprises posts from newsgroups on 20 topics. This dataset is split
# into train and test subsets based on messages posted before and after
# a specific date. We will only use posts from 2 categories to speed up running
# time.
import numpy as np

# Import necessary functions and classes from scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

##############################################################################
# Fetching training and testing data subsets from the 20 newsgroups dataset
# -------------------------------------------------------------------------

# Define categories of interest ('sci.med' and 'sci.space')
categories = ["sci.med", "sci.space"]

# Fetch training subset with posts from the specified categories
# remove=("footers", "quotes") removes extra metadata from each post
X_train, y_train = fetch_20newsgroups(
    random_state=1,
    subset="train",
    categories=categories,
    remove=("footers", "quotes"),
    return_X_y=True,
)

# Fetch test subset with posts from the specified categories
# remove=("footers", "quotes") removes extra metadata from each post
X_test, y_test = fetch_20newsgroups(
    random_state=1,
    subset="test",
    categories=categories,
    remove=("footers", "quotes"),
    return_X_y=True,
)

##############################################################################
# Each feature comprises meta information about that post, such as the subject,
# and the body of the news post.

# Print the content of the first training post to inspect its structure
print(X_train[0])

##############################################################################
# Creating transformers
# ---------------------
#
# First, we would like a transformer that extracts the subject and
# body of each post. Since this is a stateless transformation (does not
# require state information from training data), we can define a function that
# performs the data transformation then use
# :class:`~sklearn.preprocessing.FunctionTransformer` to create a scikit-learn
# transformer.

def subject_body_extractor(posts):
    # construct object dtype array with two columns
    # first column = 'subject' and second column = 'body'
    features = np.empty(shape=(len(posts), 2), dtype=object)
    for i, text in enumerate(posts):
        # 使用 enumerate 函数遍历帖子列表 posts，并获取索引 i 和内容 text
        # 使用 partition 方法将 text 按照 '\n\n' 分割为 headers, _, body 三部分
        headers, _, body = text.partition("\n\n")
        # 将 body 存储在 features 数组的第二列
        features[i, 1] = body

        prefix = "Subject:"
        sub = ""
        # 遍历 headers 中的每一行
        # 如果某行以 'Subject:' 开头，则将其后的内容存储在 sub 变量中
        for line in headers.split("\n"):
            if line.startswith(prefix):
                sub = line[len(prefix) :]
                break
        # 将 sub 存储在 features 数组的第一列
        features[i, 0] = sub

    # 返回 features 数组作为函数的输出结果
    return features
subject_body_transformer = FunctionTransformer(subject_body_extractor)

##############################################################################
# 创建一个转换器，使用指定的函数从每篇帖子中提取主题和正文信息。

def text_stats(posts):
    # 对每篇帖子计算长度和句子数，并返回包含这些统计信息的字典列表
    return [{"length": len(text), "num_sentences": text.count(".")} for text in posts]

text_stats_transformer = FunctionTransformer(text_stats)

##############################################################################
# 分类管道
# -----------------------
#
# 下面的管道从每篇帖子中提取主题和正文，使用 ``SubjectBodyExtractor``，生成一个 (n_samples, 2) 的数组。
# 然后，使用 ``ColumnTransformer`` 计算主题和正文的标准词袋特征，以及正文的文本长度和句子数。
# 这些特征被结合起来，使用权重后，对组合特征进行分类器训练。

pipeline = Pipeline(
    [
        # 提取主题和正文
        ("subjectbody", subject_body_transformer),

        # 使用 ColumnTransformer 组合主题和正文特征
        (
            "union",
            ColumnTransformer(
                [
                    # 主题的词袋特征 (列 0)
                    ("subject", TfidfVectorizer(min_df=50), 0),
                    
                    # 正文的词袋特征和分解 (列 1)
                    (
                        "body_bow",
                        Pipeline(
                            [
                                ("tfidf", TfidfVectorizer()),
                                ("best", PCA(n_components=50, svd_solver="arpack")),
                            ]
                        ),
                        1,
                    ),
                    
                    # 从帖子正文提取文本统计的管道
                    (
                        "body_stats",
                        Pipeline(
                            [
                                (
                                    "stats",
                                    text_stats_transformer,
                                ),  # 返回一个字典列表
                                (
                                    "vect",
                                    DictVectorizer(),
                                ),  # 字典列表 -> 特征矩阵
                            ]
                        ),
                        1,
                    ),
                ],
                # ColumnTransformer 特征的权重
                transformer_weights={
                    "subject": 0.8,
                    "body_bow": 0.5,
                    "body_stats": 1.0,
                },
            ),
        ),
        
        # 使用 SVC 分类器对组合特征进行分类
        ("svc", LinearSVC(dual=False)),
    ],
    verbose=True,
)

##############################################################################
# 最后，我们在训练数据上拟合我们的管道，并用它来预测“X_test”的主题。然后打印管道的性能指标报告。

pipeline.fit(X_train, y_train)
# 使用训练数据拟合管道，X_train 是特征数据，y_train 是目标数据

y_pred = pipeline.predict(X_test)
# 使用拟合后的管道预测测试数据 X_test 的目标变量

print("Classification report:\n\n{}".format(classification_report(y_test, y_pred)))
# 打印分类报告，显示测试数据上的预测性能指标
```