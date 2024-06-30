# `D:\src\scipysrc\scikit-learn\benchmarks\bench_20newsgroups.py`

```
import argparse  # 导入处理命令行参数的模块
from time import time  # 导入计时功能的时间模块

import numpy as np  # 导入数值计算库numpy

from sklearn.datasets import fetch_20newsgroups_vectorized  # 导入用于加载文本数据集的函数
from sklearn.dummy import DummyClassifier  # 导入基于简单规则的分类器
from sklearn.ensemble import (  # 导入集成学习算法中的分类器
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归分类器
from sklearn.metrics import accuracy_score  # 导入评估分类器准确率的函数
from sklearn.naive_bayes import MultinomialNB  # 导入多项式朴素贝叶斯分类器
from sklearn.utils.validation import check_array  # 导入验证数据数组有效性的函数

ESTIMATORS = {
    "dummy": DummyClassifier(),  # 使用基础的占位符分类器
    "random_forest": RandomForestClassifier(max_features="sqrt", min_samples_split=10),  # 使用随机森林分类器
    "extra_trees": ExtraTreesClassifier(max_features="sqrt", min_samples_split=10),  # 使用极端随机树分类器
    "logistic_regression": LogisticRegression(),  # 使用逻辑回归分类器
    "naive_bayes": MultinomialNB(),  # 使用多项式朴素贝叶斯分类器
    "adaboost": AdaBoostClassifier(n_estimators=10, algorithm="SAMME"),  # 使用AdaBoost分类器
}

###############################################################################
# Data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
    parser.add_argument(
        "-e", "--estimators", nargs="+", required=True, choices=ESTIMATORS
    )  # 添加一个命令行参数，接受用户输入的分类器名称
    args = vars(parser.parse_args())  # 解析命令行参数并转换为字典形式

    data_train = fetch_20newsgroups_vectorized(subset="train")  # 加载训练数据集
    data_test = fetch_20newsgroups_vectorized(subset="test")  # 加载测试数据集
    X_train = check_array(data_train.data, dtype=np.float32, accept_sparse="csc")  # 验证并转换训练数据的格式
    X_test = check_array(data_test.data, dtype=np.float32, accept_sparse="csr")  # 验证并转换测试数据的格式
    y_train = data_train.target  # 获取训练数据的目标标签
    y_test = data_test.target  # 获取测试数据的目标标签

    print("20 newsgroups")
    print("=============")
    print(f"X_train.shape = {X_train.shape}")  # 打印训练数据集的形状
    print(f"X_train.format = {X_train.format}")  # 打印训练数据集的格式
    print(f"X_train.dtype = {X_train.dtype}")  # 打印训练数据集的数据类型
    print(f"X_train density = {X_train.nnz / np.prod(X_train.shape)}")  # 计算并打印训练数据集的稀疏性
    print(f"y_train {y_train.shape}")  # 打印训练数据集目标标签的形状
    print(f"X_test {X_test.shape}")  # 打印测试数据集的形状
    print(f"X_test.format = {X_test.format}")  # 打印测试数据集的格式
    print(f"X_test.dtype = {X_test.dtype}")  # 打印测试数据集的数据类型
    print(f"y_test {y_test.shape}")  # 打印测试数据集目标标签的形状
    print()
    print("Classifier Training")
    print("===================")
    accuracy, train_time, test_time = {}, {}, {}  # 初始化用于存储结果的空字典
    for name in sorted(args["estimators"]):  # 遍历用户选择的分类器名称列表
        clf = ESTIMATORS[name]  # 根据名称获取对应的分类器对象
        try:
            clf.set_params(random_state=0)  # 设置分类器的随机状态为0
        except (TypeError, ValueError):
            pass

        print("Training %s ... " % name, end="")  # 打印当前分类器的训练状态信息
        t0 = time()  # 记录当前时间，用于计算训练时间
        clf.fit(X_train, y_train)  # 使用训练数据拟合分类器
        train_time[name] = time() - t0  # 计算并存储训练时间
        t0 = time()  # 记录当前时间，用于计算测试时间
        y_pred = clf.predict(X_test)  # 使用训练后的分类器预测测试数据
        test_time[name] = time() - t0  # 计算并存储测试时间
        accuracy[name] = accuracy_score(y_test, y_pred)  # 计算并存储预测准确率
        print("done")  # 打印训练完成状态信息

    print()
    print("Classification performance:")
    print("===========================")
    print()
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time", "Accuracy"))  # 打印表头信息
    print("-" * 44)  # 打印分隔线
    # 对 accuracy 字典中的键进行排序，排序依据为各键对应的值
    for name in sorted(accuracy, key=accuracy.get):
        # 输出格式化的字符串，包括名称左对齐，训练时间、测试时间和准确率分别居中对齐
        print(
            "%s %s %s %s"
            % (
                name.ljust(16),                      # 将名称左对齐填充到16个字符宽度
                ("%.4fs" % train_time[name]).center(10),  # 训练时间格式化为4位小数，居中填充到10字符宽度
                ("%.4fs" % test_time[name]).center(10),   # 测试时间格式化为4位小数，居中填充到10字符宽度
                ("%.4f" % accuracy[name]).center(10),     # 准确率格式化为4位小数，居中填充到10字符宽度
            )
        )

    # 输出空行
    print()
```