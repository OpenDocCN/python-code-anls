# `D:\src\scipysrc\scikit-learn\benchmarks\bench_covertype.py`

```
"""
===========================
Covertype dataset benchmark
===========================

Benchmark stochastic gradient descent (SGD), Liblinear, and Naive Bayes, CART
(decision tree), RandomForest and Extra-Trees on the forest covertype dataset
of Blackard, Jock, and Dean [1]. The dataset comprises 581,012 samples. It is
low dimensional with 54 features and a sparsity of approx. 23%. Here, we
consider the task of predicting class 1 (spruce/fir). The classification
performance of SGD is competitive with Liblinear while being two orders of
magnitude faster to train::

    [..]
    Classification performance:
    ===========================
    Classifier   train-time test-time error-rate
    --------------------------------------------
    liblinear     15.9744s    0.0705s     0.2305
    GaussianNB    3.0666s     0.3884s     0.4841
    SGD           1.0558s     0.1152s     0.2300
    CART          79.4296s    0.0523s     0.0469
    RandomForest  1190.1620s  0.5881s     0.0243
    ExtraTrees    640.3194s   0.6495s     0.0198

The same task has been used in a number of papers including:

 * :doi:`"SVM Optimization: Inverse Dependence on Training Set Size"
   S. Shalev-Shwartz, N. Srebro - In Proceedings of ICML '08.
   <10.1145/1390156.1390273>`

 * :doi:`"Pegasos: Primal estimated sub-gradient solver for svm"
   S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML '07.
   <10.1145/1273496.1273598>`

 * `"Training Linear SVMs in Linear Time"
   <https://www.cs.cornell.edu/people/tj/publications/joachims_06a.pdf>`_
   T. Joachims - In SIGKDD '06

[1] https://archive.ics.uci.edu/ml/datasets/Covertype

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
from time import time

import numpy as np
from joblib import Memory

from sklearn.datasets import fetch_covtype, get_data_home
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(
    os.path.join(get_data_home(), "covertype_benchmark_data"), mmap_mode="r"
)

# Define a function to load dataset, cache it, and memory map the train/test split
@memory.cache
def load_data(dtype=np.float32, order="C", random_state=13):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_covtype(
        download_if_missing=True, shuffle=True, random_state=random_state
    )
    X = check_array(data["data"], dtype=dtype, order=order)
    y = (data["target"] != 1).astype(int)

    # Create train-test split (as [Joachims, 2006])
    # 打印信息，表示正在创建训练集和测试集的划分
    print("Creating train-test split...")
    # 定义训练集的大小
    n_train = 522911
    # 根据训练集大小划分特征和标签的训练集部分
    X_train = X[:n_train]
    y_train = y[:n_train]
    # 根据训练集大小划分特征和标签的测试集部分
    X_test = X[n_train:]
    y_test = y[n_train:]

    # 标准化前10个特征（数值型特征）
    # 计算训练集前10个特征的均值和标准差
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # 将第10个特征之后的均值标准差设为0和1，即不标准化
    mean[10:] = 0.0
    std[10:] = 1.0
    # 对训练集进行标准化处理
    X_train = (X_train - mean) / std
    # 对测试集使用相同的均值和标准差进行标准化处理
    X_test = (X_test - mean) / std
    # 返回标准化后的训练集特征、测试集特征以及对应的训练集、测试集标签
    return X_train, X_test, y_train, y_test
# 定义一组不同的分类器及其参数配置
ESTIMATORS = {
    "GBRT": GradientBoostingClassifier(n_estimators=250),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=20),
    "RandomForest": RandomForestClassifier(n_estimators=20),
    "CART": DecisionTreeClassifier(min_samples_split=5),
    "SGD": SGDClassifier(alpha=0.001),
    "GaussianNB": GaussianNB(),
    "liblinear": LinearSVC(loss="l2", penalty="l2", C=1000, dual=False, tol=1e-3),
    "SAG": LogisticRegression(solver="sag", max_iter=2, C=1000),
}

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加 --classifiers 参数，可选的分类器列表，默认为 ["liblinear", "GaussianNB", "SGD", "CART"]
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=ESTIMATORS,
        type=str,
        default=["liblinear", "GaussianNB", "SGD", "CART"],
        help="list of classifiers to benchmark.",
    )
    # 添加 --n-jobs 参数，指定并行运行的工作线程数，默认为 1
    parser.add_argument(
        "--n-jobs",
        nargs="?",
        default=1,
        type=int,
        help=(
            "Number of concurrently running workers for "
            "models that support parallelism."
        ),
    )
    # 添加 --order 参数，指定数据的存储顺序，可选 "F" 或 "C"，默认为 "C"
    parser.add_argument(
        "--order",
        nargs="?",
        default="C",
        type=str,
        choices=["F", "C"],
        help="Allow to choose between fortran and C ordered data",
    )
    # 添加 --random-seed 参数，指定随机数生成器的种子，默认为 13
    parser.add_argument(
        "--random-seed",
        nargs="?",
        default=13,
        type=int,
        help="Common seed used by random number generator.",
    )
    # 解析命令行参数为字典
    args = vars(parser.parse_args())

    # 打印脚本的文档字符串
    print(__doc__)

    # 载入数据，根据指定的顺序和随机种子
    X_train, X_test, y_train, y_test = load_data(
        order=args["order"], random_state=args["random_seed"]
    )

    # 打印数据集的统计信息
    print("")
    print("Dataset statistics:")
    print("===================")
    # 打印特征数
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    # 打印类别数
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    # 打印数据类型
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    # 打印训练样本数及其正负样本的数量和大小
    print(
        "%s %d (pos=%d, neg=%d, size=%dMB)"
        % (
            "number of train samples:".ljust(25),
            X_train.shape[0],
            np.sum(y_train == 1),
            np.sum(y_train == 0),
            int(X_train.nbytes / 1e6),
        )
    )
    # 打印测试样本数及其正负样本的数量和大小
    print(
        "%s %d (pos=%d, neg=%d, size=%dMB)"
        % (
            "number of test samples:".ljust(25),
            X_test.shape[0],
            np.sum(y_test == 1),
            np.sum(y_test == 0),
            int(X_test.nbytes / 1e6),
        )
    )

    print()
    # 打印训练分类器的信息
    print("Training Classifiers")
    print("====================")
    # 初始化用于记录错误率、训练时间和测试时间的空字典
    error, train_time, test_time = {}, {}, {}
    # 对分类器名称列表按字母顺序进行排序，并逐个进行训练和评估
    for name in sorted(args["classifiers"]):
        # 打印正在训练的分类器名称
        print("Training %s ... " % name, end="")
        # 获取当前分类器对应的估计器对象
        estimator = ESTIMATORS[name]
        # 获取当前估计器的参数字典
        estimator_params = estimator.get_params()

        # 将所有以 'random_state' 结尾的参数设置为相同的随机种子值
        estimator.set_params(
            **{
                p: args["random_seed"]
                for p in estimator_params
                if p.endswith("random_state")
            }
        )

        # 如果估计器参数中包含 'n_jobs'，设置其并行工作数
        if "n_jobs" in estimator_params:
            estimator.set_params(n_jobs=args["n_jobs"])

        # 开始计时训练时间
        time_start = time()
        # 使用训练集对估计器进行训练
        estimator.fit(X_train, y_train)
        # 计算训练时间并存储
        train_time[name] = time() - time_start

        # 开始计时测试时间
        time_start = time()
        # 使用测试集对估计器进行预测
        y_pred = estimator.predict(X_test)
        # 计算测试时间并存储
        test_time[name] = time() - time_start

        # 计算预测误差并存储
        error[name] = zero_one_loss(y_test, y_pred)

        # 打印训练完成信息
        print("done")

    # 打印空行和分类器性能信息标题
    print()
    print("Classification performance:")
    print("===========================")
    # 打印表头信息：分类器名称、训练时间、测试时间、错误率
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time", "error-rate"))
    # 打印分隔线
    print("-" * 44)
    # 根据分类器的预测误差从低到高对分类器进行排序，并打印性能数据
    for name in sorted(args["classifiers"], key=error.get):
        print(
            "%s %s %s %s"
            % (
                # 打印分类器名称并左对齐
                name.ljust(12),
                # 打印训练时间，并居中对齐
                ("%.4fs" % train_time[name]).center(10),
                # 打印测试时间，并居中对齐
                ("%.4fs" % test_time[name]).center(10),
                # 打印错误率，并居中对齐
                ("%.4f" % error[name]).center(10),
            )
        )

    # 打印空行
    print()
```