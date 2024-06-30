# `D:\src\scipysrc\scikit-learn\benchmarks\bench_saga.py`

```
"""
Author: Arthur Mensch, Nelle Varoquaux

Benchmarks of sklearn SAGA vs lightning SAGA vs Liblinear. Shows the gain
in using multinomial logistic regression in term of learning time.
"""

# 导入所需的库
import json  # 导入处理 JSON 数据的库
import os  # 导入操作系统相关功能的库
import time  # 导入时间相关功能的库

import matplotlib.pyplot as plt  # 导入绘图库 matplotlib 的 pyplot 模块
import numpy as np  # 导入数值计算库 numpy

from sklearn.datasets import (  # 导入 sklearn 中的数据集模块
    fetch_20newsgroups_vectorized,  # 导入获取新闻数据集的函数
    fetch_rcv1,  # 导入获取 RCV1 数据集的函数
    load_digits,  # 导入获取手写数字数据集的函数
    load_iris,  # 导入获取鸢尾花数据集的函数
)
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import log_loss  # 导入对数损失函数
from sklearn.model_selection import train_test_split  # 导入数据集分割函数
from sklearn.multiclass import OneVsRestClassifier  # 导入一对多分类器
from sklearn.preprocessing import LabelBinarizer, LabelEncoder  # 导入标签编码器
from sklearn.utils.extmath import safe_sparse_dot, softmax  # 导入数学工具函数
from sklearn.utils.parallel import Parallel, delayed  # 导入并行运算相关函数


def fit_single(
    solver,
    X,
    y,
    penalty="l2",
    single_target=True,
    C=1,
    max_iter=10,
    skip_slow=False,
    dtype=np.float64,
):
    # 如果需要跳过使用 lightning 解决 l1 正则化的情况，则打印提示信息并返回
    if skip_slow and solver == "lightning" and penalty == "l1":
        print("skip_slowping l1 logistic regression with solver lightning.")
        return

    # 打印当前正在解决的问题类型和参数设置
    print(
        "Solving %s logistic regression with penalty %s, solver %s."
        % ("binary" if single_target else "multinomial", penalty, solver)
    )

    # 如果 solver 是 lightning，则导入 lightning 库中的 SAGAClassifier
    if solver == "lightning":
        from lightning.classification import SAGAClassifier

    # 根据单目标或多目标分类设置 multi_class 的值
    if single_target or solver not in ["sag", "saga"]:
        multi_class = "ovr"
    else:
        multi_class = "multinomial"

    # 将输入特征 X 和标签 y 转换为指定的数据类型 dtype
    X = X.astype(dtype)
    y = y.astype(dtype)

    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )

    # 计算训练样本数和类别数
    n_samples = X_train.shape[0]
    n_classes = np.unique(y_train).shape[0]

    # 初始化测试分数、训练分数、准确率和时间列表
    test_scores = [1]
    train_scores = [1]
    accuracies = [1 / n_classes]
    times = [0]

    # 根据正则化类型设置 alpha 和 beta 参数
    if penalty == "l2":
        alpha = 1.0 / (C * n_samples)
        beta = 0
        lightning_penalty = None
    else:
        alpha = 0.0
        beta = 1.0 / (C * n_samples)
        lightning_penalty = "l1"
    # 循环迭代，从1到max_iter（包含），步长为2
    for this_max_iter in range(1, max_iter + 1, 2):
        # 打印当前的最大迭代次数和模型参数
        print(
            "[%s, %s, %s] Max iter: %s"
            % (
                "binary" if single_target else "multinomial",  # 根据条件选择分类类型
                penalty,  # 正则化类型
                solver,  # 使用的求解器
                this_max_iter,  # 当前迭代次数
            )
        )
        # 如果求解器是"lightning"，则使用SAGAClassifier模型
        if solver == "lightning":
            lr = SAGAClassifier(
                loss="log",
                alpha=alpha,
                beta=beta,
                penalty=lightning_penalty,
                tol=-1,
                max_iter=this_max_iter,
            )
        else:
            # 否则使用LogisticRegression模型
            lr = LogisticRegression(
                solver=solver,
                C=C,
                penalty=penalty,
                fit_intercept=False,
                tol=0,
                max_iter=this_max_iter,
                random_state=42,
            )
            # 如果多分类设置为"ovr"，则使用OneVsRestClassifier包装
            if multi_class == "ovr":
                lr = OneVsRestClassifier(lr)

        # 计算训练数据的最大值，以缓存 CPU
        X_train.max()
        t0 = time.clock()  # 记录当前时间

        # 拟合模型使用训练集
        lr.fit(X_train, y_train)
        train_time = time.clock() - t0  # 计算训练时间

        scores = []
        # 针对训练集和测试集计算预测概率
        for X, y in [(X_train, y_train), (X_test, y_test)]:
            try:
                y_pred = lr.predict_proba(X)  # 尝试预测概率
            except NotImplementedError:
                # 对于不支持多类别预测概率的情况，使用自定义函数
                y_pred = _predict_proba(lr, X)
            # 如果模型是OneVsRestClassifier，则合并所有估计器的系数
            if isinstance(lr, OneVsRestClassifier):
                coef = np.concatenate([est.coef_ for est in lr.estimators_])
            else:
                coef = lr.coef_
            # 计算对数损失，包括正则化项
            score = log_loss(y, y_pred, normalize=False) / n_samples
            score += 0.5 * alpha * np.sum(coef**2) + beta * np.sum(np.abs(coef))
            scores.append(score)
        train_score, test_score = tuple(scores)  # 分别获取训练集和测试集的评分

        y_pred = lr.predict(X_test)  # 对测试集进行预测
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]  # 计算准确率
        test_scores.append(test_score)  # 将测试集评分添加到列表中
        train_scores.append(train_score)  # 将训练集评分添加到列表中
        accuracies.append(accuracy)  # 将准确率添加到列表中
    return lr, times, train_scores, test_scores, accuracies  # 返回模型对象及相关评估数据
# 定义一个函数，用于预测概率值，适用于多分类问题（n_classes >= 3）
def _predict_proba(lr, X):
    """Predict proba for lightning for n_classes >=3."""
    # 计算预测值，通过稀疏矩阵乘法得到预测结果
    pred = safe_sparse_dot(X, lr.coef_.T)
    # 如果模型具有截距项，则加上截距项
    if hasattr(lr, "intercept_"):
        pred += lr.intercept_
    # 应用 softmax 函数得到概率值并返回
    return softmax(pred)


# 定义一个函数，用于执行实验
def exp(
    solvers,
    penalty,
    single_target,
    n_samples=30000,
    max_iter=20,
    dataset="rcv1",
    n_jobs=1,
    skip_slow=False,
):
    # 数据类型映射字典，指定了数据类型对应的 NumPy 类型
    dtypes_mapping = {
        "float64": np.float64,
        "float32": np.float32,
    }

    # 根据数据集名称加载数据
    if dataset == "rcv1":
        # 获取 RCV1 数据集
        rcv1 = fetch_rcv1()

        # 标签二值化处理
        lbin = LabelBinarizer()
        lbin.fit(rcv1.target_names)

        X = rcv1.data  # 特征数据
        y = rcv1.target  # 目标标签
        y = lbin.inverse_transform(y)  # 反转标签二值化处理
        le = LabelEncoder()
        y = le.fit_transform(y)  # 对标签进行编码
        # 如果是单目标任务，则进一步处理标签
        if single_target:
            y_n = y.copy()
            y_n[y > 16] = 1
            y_n[y <= 16] = 0
            y = y_n

    elif dataset == "digits":
        # 加载 digits 数据集
        X, y = load_digits(return_X_y=True)
        # 如果是单目标任务，则进一步处理标签
        if single_target:
            y_n = y.copy()
            y_n[y < 5] = 1
            y_n[y >= 5] = 0
            y = y_n

    elif dataset == "iris":
        # 加载 iris 数据集
        iris = load_iris()
        X, y = iris.data, iris.target

    elif dataset == "20newspaper":
        # 获取 20newsgroups 数据集
        ng = fetch_20newsgroups_vectorized()
        X = ng.data  # 特征数据
        y = ng.target  # 目标标签
        # 如果是单目标任务，则进一步处理标签
        if single_target:
            y_n = y.copy()
            y_n[y > 4] = 1
            y_n[y <= 16] = 0
            y = y_n

    # 限制样本数量
    X = X[:n_samples]
    y = y[:n_samples]

    # 并行执行拟合任务
    out = Parallel(n_jobs=n_jobs, mmap_mode=None)(
        delayed(fit_single)(
            solver,
            X,
            y,
            penalty=penalty,
            single_target=single_target,
            dtype=dtype,
            C=1,
            max_iter=max_iter,
            skip_slow=skip_slow,
        )
        for solver in solvers
        for dtype in dtypes_mapping.values()
    )

    res = []  # 结果列表
    idx = 0  # 索引计数器
    # 遍历数据类型映射字典的键
    for dtype_name in dtypes_mapping.keys():
        # 遍历求解器列表
        for solver in solvers:
            # 根据条件判断是否跳过慢处理
            if not (skip_slow and solver == "lightning" and penalty == "l1"):
                # 从输出中获取结果数据
                lr, times, train_scores, test_scores, accuracies = out[idx]
                # 组装当前结果的字典
                this_res = dict(
                    solver=solver,
                    penalty=penalty,
                    dtype=dtype_name,
                    single_target=single_target,
                    times=times,
                    train_scores=train_scores,
                    test_scores=test_scores,
                    accuracies=accuracies,
                )
                res.append(this_res)  # 将结果字典添加到结果列表中
            idx += 1  # 索引加一

    # 将结果以 JSON 格式写入文件
    with open("bench_saga.json", "w+") as f:
        json.dump(res, f)


# 定义一个函数，用于绘制图表
def plot(outname=None):
    import pandas as pd

    # 从 JSON 文件中读取数据
    with open("bench_saga.json", "r") as f:
        f = json.load(f)
    res = pd.DataFrame(f)  # 将读取的数据转换为 DataFrame 格式
    res.set_index(["single_target"], inplace=True)  # 将 'single_target' 列设为索引

    grouped = res.groupby(level=["single_target"])  # 按 'single_target' 列进行分组

    colors = {"saga": "C0", "liblinear": "C1", "lightning": "C2"}  # 定义颜色映射字典
    linestyles = {"float32": "--", "float64": "-"}  # 定义线型映射字典
    # 定义一个字典 alpha，包含两个键值对
    alpha = {"float64": 0.5, "float32": 1}
if __name__ == "__main__":
    # 定义使用的求解器列表
    solvers = ["saga", "liblinear", "lightning"]
    # 定义惩罚类型列表
    penalties = ["l1", "l2"]
    # 定义样本数量列表，包括几个特定值和一个空值
    n_samples = [100000, 300000, 500000, 800000, None]
    # 是否只有单一目标的标志
    single_target = True
    
    # 遍历所有惩罚类型
    for penalty in penalties:
        # 遍历所有样本数量
        for n_sample in n_samples:
            # 调用 exp 函数进行实验，设置参数
            exp(
                solvers,         # 求解器列表
                penalty,         # 当前惩罚类型
                single_target,   # 是否单一目标
                n_samples=n_sample,  # 当前样本数量
                n_jobs=1,        # 使用的工作线程数
                dataset="rcv1",  # 数据集名称
                max_iter=10,     # 最大迭代次数
            )
            # 根据当前样本数量生成输出文件名
            if n_sample is not None:
                outname = "figures/saga_%s_%d.png" % (penalty, n_sample)
            else:
                outname = "figures/saga_%s_all.png" % (penalty,)
            
            # 尝试创建目录 "figures"，如果已存在则跳过
            try:
                os.makedirs("figures")
            except OSError:
                pass
            
            # 绘制图表，并保存到指定的输出文件名
            plot(outname)
```