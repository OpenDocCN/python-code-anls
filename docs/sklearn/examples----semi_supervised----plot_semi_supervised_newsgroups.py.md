# `D:\src\scipysrc\scikit-learn\examples\semi_supervised\plot_semi_supervised_newsgroups.py`

```
"""
================================================
Semi-supervised Classification on a Text Dataset
================================================

In this example, semi-supervised classifiers are trained on the 20 newsgroups
dataset (which will be automatically downloaded).

You can adjust the number of categories by giving their names to the dataset
loader or setting them to `None` to get all 20 of them.

"""

# 导入必要的库
import numpy as np

# 导入数据集和特征提取工具
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# 导入分类器和评估指标
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

# 导入数据集划分工具和流水线构建工具
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 导入数据预处理工具
from sklearn.preprocessing import FunctionTransformer

# 导入半监督学习相关模型
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

# 加载数据集，指定五个初始类别
data = fetch_20newsgroups(
    subset="train",
    categories=[
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
    ],
)

print("%d documents" % len(data.filenames))  # 打印文档数量
print("%d categories" % len(data.target_names))  # 打印类别数量
print()

# 定义分类器参数
sdg_params = dict(alpha=1e-5, penalty="l2", loss="log_loss")
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)

# 构建监督学习流水线
pipeline = Pipeline(
    [
        ("vect", CountVectorizer(**vectorizer_params)),  # 文本向量化
        ("tfidf", TfidfTransformer()),  # TF-IDF转换
        ("clf", SGDClassifier(**sdg_params)),  # SGD分类器
    ]
)

# 构建自训练半监督学习流水线
st_pipeline = Pipeline(
    [
        ("vect", CountVectorizer(**vectorizer_params)),  # 文本向量化
        ("tfidf", TfidfTransformer()),  # TF-IDF转换
        ("clf", SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True)),  # 自训练分类器
    ]
)

# 构建标签传播半监督学习流水线
ls_pipeline = Pipeline(
    [
        ("vect", CountVectorizer(**vectorizer_params)),  # 文本向量化
        ("tfidf", TfidfTransformer()),  # TF-IDF转换
        # 标签传播不支持稠密矩阵，需要转换为稀疏矩阵
        ("toarray", FunctionTransformer(lambda x: x.toarray())),  # 转换为稀疏矩阵
        ("clf", LabelSpreading()),  # 标签传播分类器
    ]
)


def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    """
    评估并打印分类器性能指标
    """
    print("Number of training samples:", len(X_train))  # 打印训练样本数量
    print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1))  # 打印训练集中未标记样本数量
    clf.fit(X_train, y_train)  # 训练分类器
    y_pred = clf.predict(X_test)  # 预测测试集
    print(
        "Micro-averaged F1 score on test set: %0.3f"
        % f1_score(y_test, y_pred, average="micro")  # 计算并打印测试集的微平均F1分数
    )
    print("-" * 10)
    print()


if __name__ == "__main__":
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Supervised SGDClassifier on 100% of the data:")
    eval_and_print_metrics(pipeline, X_train, y_train, X_test, y_test)

    # 选择一个20%的训练数据子集的掩码
    y_mask = np.random.rand(len(y_train)) < 0.2
    # 使用布尔掩码 `y_mask` 来选择训练数据集的子集 X_20 和 y_20
    X_20, y_20 = map(
        list, zip(*((x, y) for x, y, m in zip(X_train, y_train, y_mask) if m))
    )
    # 输出提示信息，指出在 20% 训练数据上使用监督学习的 SGDClassifier
    print("Supervised SGDClassifier on 20% of the training data:")
    # 调用评估函数，打印和记录 SGDClassifier 在指定数据集上的性能指标
    eval_and_print_metrics(pipeline, X_20, y_20, X_test, y_test)

    # 将非掩码部分的训练数据标记为未标记 (-1 表示未标记)
    y_train[~y_mask] = -1
    # 输出提示信息，指出在 20% 训练数据上使用 SelfTrainingClassifier
    print("SelfTrainingClassifier on 20% of the training data (rest is unlabeled):")
    # 调用评估函数，打印和记录 SelfTrainingClassifier 在指定数据集上的性能指标
    eval_and_print_metrics(st_pipeline, X_train, y_train, X_test, y_test)

    # 输出提示信息，指出在 20% 数据上使用 LabelSpreading
    print("LabelSpreading on 20% of the data (rest is unlabeled):")
    # 调用评估函数，打印和记录 LabelSpreading 在指定数据集上的性能指标
    eval_and_print_metrics(ls_pipeline, X_train, y_train, X_test, y_test)
```