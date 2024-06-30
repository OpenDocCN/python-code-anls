# `D:\src\scipysrc\scikit-learn\benchmarks\bench_mnist.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
import argparse     # 用于解析命令行参数
import os           # 提供操作系统相关功能的模块
from time import time   # 导入时间模块中的time函数

import numpy as np      # 数组操作库
from joblib import Memory   # 用于缓存数据的工具

# 从sklearn库中导入需要的函数和类
from sklearn.datasets import fetch_openml, get_data_home   # 数据集加载函数和数据集主目录获取函数
from sklearn.dummy import DummyClassifier   # 用于生成虚拟分类器的类
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier   # 集成方法分类器
from sklearn.kernel_approximation import Nystroem, RBFSampler   # 内核近似方法
from sklearn.linear_model import LogisticRegression   # 逻辑回归模型
from sklearn.metrics import zero_one_loss   # 0-1损失计算函数
from sklearn.neural_network import MLPClassifier   # 多层感知机分类器
from sklearn.pipeline import make_pipeline   # 创建管道的函数
from sklearn.svm import LinearSVC   # 线性支持向量机模型
from sklearn.tree import DecisionTreeClassifier   # 决策树分类器
from sklearn.utils import check_array   # 检查和转换数组的函数

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
# 使用Memory对象缓存数据提取结果，并以只读模式内存映射训练/测试数据分割结果
memory = Memory(os.path.join(get_data_home(), "mnist_benchmark_data"), mmap_mode="r")

# 使用缓存装饰器，加载数据并进行数据类型转换和存储模式设置
@memory.cache
def load_data(dtype=np.float32, order="F"):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    # 从OpenML下载MNIST数据集，并返回为DataFrame
    data = fetch_openml("mnist_784", as_frame=True)
    # 将数据转换为指定数据类型和内存布局的数组
    X = check_array(data["data"], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    # 对特征进行归一化处理，将像素值缩放到0-1范围
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    # 创建训练集和测试集划分，遵循Joachims（2006年）的方法
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test

# 定义一些分类器模型的集合，用于进行性能评估
ESTIMATORS = {
    "dummy": DummyClassifier(),   # 随机预测的虚拟分类器
    "CART": DecisionTreeClassifier(),   # 决策树分类器
    "ExtraTrees": ExtraTreesClassifier(),   # 极端随机树分类器
    "RandomForest": RandomForestClassifier(),   # 随机森林分类器
    "Nystroem-SVM": make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=100)
    ),

# 创建名为"Nystroem-SVM"的机器学习管道，包括特征映射器Nystroem和线性支持向量机LinearSVC。


    "SampledRBF-SVM": make_pipeline(
        RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100)
    ),

# 创建名为"SampledRBF-SVM"的机器学习管道，包括RBF采样器RBFSampler和线性支持向量机LinearSVC。


    "LogisticRegression-SAG": LogisticRegression(solver="sag", tol=1e-1, C=1e4),

# 创建名为"LogisticRegression-SAG"的逻辑回归模型，使用"sag"求解器，容差为1e-1，正则化强度C为1e4。


    "LogisticRegression-SAGA": LogisticRegression(solver="saga", tol=1e-1, C=1e4),

# 创建名为"LogisticRegression-SAGA"的逻辑回归模型，使用"saga"求解器，容差为1e-1，正则化强度C为1e4。


    "MultilayerPerceptron": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="sgd",
        learning_rate_init=0.2,
        momentum=0.9,
        verbose=1,
        tol=1e-4,
        random_state=1,
    ),

# 创建名为"MultilayerPerceptron"的多层感知机分类器，包括两个隐藏层每层100个神经元，最大迭代次数400，学习率初始值为0.2，使用"sgd"求解器，动量为0.9，容差为1e-4，随机状态为1。


    "MLP-adam": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="adam",
        learning_rate_init=0.001,
        verbose=1,
        tol=1e-4,
        random_state=1,
    ),

# 创建名为"MLP-adam"的多层感知机分类器，包括两个隐藏层每层100个神经元，最大迭代次数400，正则化参数alpha为1e-4，使用"adam"求解器，学习率初始值为0.001，容差为1e-4，随机状态为1。
}

# 如果脚本被直接执行（而不是被导入为模块），则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数：classifiers，接受多个参数，从预定义的 ESTIMATORS 中选择，默认为一些分类器
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=ESTIMATORS,
        type=str,
        default=["ExtraTrees", "Nystroem-SVM"],
        help="list of classifiers to benchmark.",
    )
    
    # 添加命令行参数：n-jobs，可选参数，默认为1，用于支持并行处理的模型
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
    
    # 添加命令行参数：order，可选参数，默认为'C'，允许选择 Fortran 或 C 排列的数据
    parser.add_argument(
        "--order",
        nargs="?",
        default="C",
        type=str,
        choices=["F", "C"],
        help="Allow to choose between fortran and C ordered data",
    )
    
    # 添加命令行参数：random-seed，可选参数，默认为0，作为随机数生成器的种子
    parser.add_argument(
        "--random-seed",
        nargs="?",
        default=0,
        type=int,
        help="Common seed used by random number generator.",
    )
    
    # 解析命令行参数，并将其转换为字典形式
    args = vars(parser.parse_args())

    # 打印脚本的文档字符串
    print(__doc__)

    # 加载数据集，根据参数中的 order 进行指定数据排列
    X_train, X_test, y_train, y_test = load_data(order=args["order"])

    # 打印数据集的统计信息
    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print(
        "%s %d (size=%dMB)"
        % (
            "number of train samples:".ljust(25),
            X_train.shape[0],
            int(X_train.nbytes / 1e6),
        )
    )
    print(
        "%s %d (size=%dMB)"
        % (
            "number of test samples:".ljust(25),
            X_test.shape[0],
            int(X_test.nbytes / 1e6),
        )
    )

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    
    # 遍历排序后的分类器列表，依次训练分类器
    for name in sorted(args["classifiers"]):
        print("Training %s ... " % name, end="")
        estimator = ESTIMATORS[name]
        estimator_params = estimator.get_params()

        # 设置分类器的随机种子参数
        estimator.set_params(
            **{
                p: args["random_seed"]
                for p in estimator_params
                if p.endswith("random_state")
            }
        )

        # 如果分类器支持并行运行，设置其并行工作的数量
        if "n_jobs" in estimator_params:
            estimator.set_params(n_jobs=args["n_jobs"])

        # 开始计时，训练分类器
        time_start = time()
        estimator.fit(X_train, y_train)
        train_time[name] = time() - time_start

        # 开始计时，对测试集进行预测
        time_start = time()
        y_pred = estimator.predict(X_test)
        test_time[name] = time() - time_start

        # 计算分类器在测试集上的错误率
        error[name] = zero_one_loss(y_test, y_pred)

        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    # 打印分类器的性能指标：训练时间、测试时间、错误率
    print(
        "{0: <24} {1: >10} {2: >11} {3: >12}".format(
            "Classifier  ", "train-time", "test-time", "error-rate"
        )
    )
    print("-" * 60)
    # 对于传入参数 args 中的 "classifiers" 键所对应的列表，按照 error 字典中相应元素的值进行排序，并遍历
    for name in sorted(args["classifiers"], key=error.get):
        # 打印格式化后的字符串，显示分类器名称、训练时间、测试时间和误差率
        print(
            "{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}".format(
                name, train_time[name], test_time[name], error[name]
            )
        )

    # 输出空行，用于格式化输出
    print()
```