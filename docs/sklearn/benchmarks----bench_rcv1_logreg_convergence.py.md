# `D:\src\scipysrc\scikit-learn\benchmarks\bench_rcv1_logreg_convergence.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库和模块
import gc  # 导入垃圾回收模块
import time  # 导入时间模块

import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy
from joblib import Memory  # 从 joblib 库导入 Memory 类

from sklearn.datasets import fetch_rcv1  # 从 sklearn 中导入数据集加载函数 fetch_rcv1
from sklearn.linear_model import LogisticRegression, SGDClassifier  # 从 sklearn 中导入逻辑回归和随机梯度下降分类器
from sklearn.linear_model._sag import get_auto_step_size  # 从 sklearn 中导入获取自动步长大小函数 get_auto_step_size

try:
    import lightning.classification as lightning_clf  # 尝试导入 lightning 分类模块
except ImportError:
    lightning_clf = None  # 如果导入失败，将 lightning_clf 设为 None

m = Memory(cachedir=".", verbose=0)  # 创建 Memory 对象 m，用于缓存

# 计算逻辑损失函数
def get_loss(w, intercept, myX, myy, C):
    n_samples = myX.shape[0]
    w = w.ravel()
    p = np.mean(np.log(1.0 + np.exp(-myy * (myX.dot(w) + intercept))))
    print("%f + %f" % (p, w.dot(w) / 2.0 / C / n_samples))
    p += w.dot(w) / 2.0 / C / n_samples
    return p

# 使用 joblib 缓存每次拟合结果。注意，这里不将数据集作为参数传递，因为数据集的哈希处理速度太慢，假设数据集不变。
@m.cache()
def bench_one(name, clf_type, clf_params, n_iter):
    clf = clf_type(**clf_params)
    try:
        clf.set_params(max_iter=n_iter, random_state=42)
    except Exception:
        clf.set_params(n_iter=n_iter, random_state=42)

    st = time.time()  # 记录开始时间
    clf.fit(X, y)  # 拟合分类器
    end = time.time()  # 记录结束时间

    try:
        C = 1.0 / clf.alpha / n_samples
    except Exception:
        C = clf.C

    try:
        intercept = clf.intercept_
    except Exception:
        intercept = 0.0

    train_loss = get_loss(clf.coef_, intercept, X, y, C)  # 计算训练损失
    train_score = clf.score(X, y)  # 计算训练分数
    test_score = clf.score(X_test, y_test)  # 计算测试分数
    duration = end - st  # 计算拟合时间

    return train_loss, train_score, test_score, duration

# 对一组分类器进行性能评估
def bench(clfs):
    for (
        name,
        clf,
        iter_range,
        train_losses,
        train_scores,
        test_scores,
        durations,
    ) in clfs:
        print("training %s" % name)  # 打印当前分类器名称
        clf_type = type(clf)
        clf_params = clf.get_params()

        for n_iter in iter_range:
            gc.collect()  # 执行垃圾回收

            train_loss, train_score, test_score, duration = bench_one(
                name, clf_type, clf_params, n_iter
            )  # 进行单次拟合评估

            train_losses.append(train_loss)
            train_scores.append(train_score)
            test_scores.append(test_score)
            durations.append(duration)
            print("classifier: %s" % name)  # 打印分类器名称
            print("train_loss: %.8f" % train_loss)  # 打印训练损失
            print("train_score: %.8f" % train_score)  # 打印训练分数
            print("test_score: %.8f" % test_score)  # 打印测试分数
            print("time for fit: %.8f seconds" % duration)  # 打印拟合时间
            print("")

        print("")

    return clfs

# 绘制训练损失曲线
def plot_train_losses(clfs):
    plt.figure()
    for name, _, _, train_losses, _, _, durations in clfs:
        plt.plot(durations, train_losses, "-o", label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")  # 设置 x 轴标签为“秒”
        plt.ylabel("train loss")  # 设置 y 轴标签为“训练损失”

# 绘制训练分数曲线
def plot_train_scores(clfs):
    plt.figure()
    # 对于给定的 clfs 列表中的每个元组，依次取出元组中的 name, _, _, _, train_scores, _, durations 变量
    for name, _, _, _, train_scores, _, durations in clfs:
        # 绘制训练时长 durations 和训练分数 train_scores 的关系图，以圆点标记每个数据点，连接线形式为 '-o'，并标注线的名称为 name
        plt.plot(durations, train_scores, "-o", label=name)
        # 在图的最佳位置自动添加图例
        plt.legend(loc=0)
        # 设置 x 轴的标签为 "seconds"
        plt.xlabel("seconds")
        # 设置 y 轴的标签为 "train score"
        plt.ylabel("train score")
        # 设置 y 轴的数值范围为 (0.92, 0.96)
        plt.ylim((0.92, 0.96))
def plot_test_scores(clfs):
    """Plot test scores against durations for multiple classifiers.

    Args:
        clfs (list): List of tuples containing classifier information.

    Returns:
        None
    """
    plt.figure()
    for name, _, _, _, _, test_scores, durations in clfs:
        plt.plot(durations, test_scores, "-o", label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("test score")
        plt.ylim((0.92, 0.96))


def plot_dloss(clfs):
    """Plot logarithm of difference between best and train loss against durations.

    Args:
        clfs (list): List of tuples containing classifier information.

    Returns:
        None
    """
    plt.figure()
    pobj_final = []
    for name, _, _, train_losses, _, _, durations in clfs:
        pobj_final.append(train_losses[-1])

    indices = np.argsort(pobj_final)
    pobj_best = pobj_final[indices[0]]

    for name, _, _, train_losses, _, _, durations in clfs:
        log_pobj = np.log(abs(np.array(train_losses) - pobj_best)) / np.log(10)

        plt.plot(durations, log_pobj, "-o", label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("log(best - train_loss)")


def get_max_squared_sum(X):
    """Calculate the maximum row-wise sum of squares of X.

    Args:
        X (numpy.ndarray): Input array

    Returns:
        float: Maximum row-wise sum of squares
    """
    return np.sum(X**2, axis=1).max()


rcv1 = fetch_rcv1()
X = rcv1.data
n_samples, n_features = X.shape

# consider the binary classification problem 'CCAT' vs the rest
ccat_idx = rcv1.target_names.tolist().index("CCAT")
y = rcv1.target.tocsc()[:, ccat_idx].toarray().ravel().astype(np.float64)
y[y == 0] = -1

# parameters
C = 1.0
fit_intercept = True
tol = 1.0e-14

# max_iter range
sgd_iter_range = list(range(1, 121, 10))
newton_iter_range = list(range(1, 25, 3))
lbfgs_iter_range = list(range(1, 242, 12))
liblinear_iter_range = list(range(1, 37, 3))
liblinear_dual_iter_range = list(range(1, 85, 6))
sag_iter_range = list(range(1, 37, 3))

clfs = [
    (
        "LR-liblinear",
        LogisticRegression(
            C=C,
            tol=tol,
            solver="liblinear",
            fit_intercept=fit_intercept,
            intercept_scaling=1,
        ),
        liblinear_iter_range,
        [],  # train_losses placeholder
        [],  # valid_losses placeholder
        [],  # test_scores placeholder
        [],  # durations placeholder
    ),
    (
        "LR-liblinear-dual",
        LogisticRegression(
            C=C,
            tol=tol,
            dual=True,
            solver="liblinear",
            fit_intercept=fit_intercept,
            intercept_scaling=1,
        ),
        liblinear_dual_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "LR-SAG",
        LogisticRegression(C=C, tol=tol, solver="sag", fit_intercept=fit_intercept),
        sag_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "LR-newton-cg",
        LogisticRegression(
            C=C, tol=tol, solver="newton-cg", fit_intercept=fit_intercept
        ),
        newton_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "LR-lbfgs",
        LogisticRegression(C=C, tol=tol, solver="lbfgs", fit_intercept=fit_intercept),
        lbfgs_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        # 使用随机梯度下降（SGD）分类器进行分类任务
        "SGD",
        SGDClassifier(
            # 正则化项的系数，影响正则化强度
            alpha=1.0 / C / n_samples,
            # 惩罚项的类型，这里为L2范数
            penalty="l2",
            # 损失函数类型，这里为逻辑损失函数
            loss="log_loss",
            # 是否计算截距
            fit_intercept=fit_intercept,
            # 控制详细程度的标志，0为不输出进度
            verbose=0,
        ),
        # SGD分类器的迭代次数范围
        sgd_iter_range,
        # 以下列表为占位符，暂无内容
        [],
        [],
        [],
        [],
    ),
# 如果 lightning_clf 不是 None 并且 fit_intercept 为 False，则执行以下操作
if lightning_clf is not None and not fit_intercept:
    # 计算 alpha 参数，用于 SVRGClassifier 和 SAGClassifier
    alpha = 1.0 / C / n_samples
    # 计算与 LR-sag 中相同的步长（step_size）
    max_squared_sum = get_max_squared_sum(X)
    step_size = get_auto_step_size(max_squared_sum, alpha, "log", fit_intercept)

    # 向 clfs 列表中添加 Lightning-SVRG 模型及其参数
    clfs.append(
        (
            "Lightning-SVRG",
            lightning_clf.SVRGClassifier(
                alpha=alpha, eta=step_size, tol=tol, loss="log"
            ),
            sag_iter_range,
            [],
            [],
            [],
            [],
        )
    )
    # 向 clfs 列表中添加 Lightning-SAG 模型及其参数
    clfs.append(
        (
            "Lightning-SAG",
            lightning_clf.SAGClassifier(
                alpha=alpha, eta=step_size, tol=tol, loss="log"
            ),
            sag_iter_range,
            [],
            [],
            [],
            [],
        )
    )

    # 保留仅含有 200 个特征的数据集，以得到一个密集的数据集，
    # 并与稀疏情况下的 Lightning SAG 进行比较，这在稀疏情况下似乎是不正确的。
    X_csc = X.tocsc()
    nnz_in_each_features = X_csc.indptr[1:] - X_csc.indptr[:-1]
    X = X_csc[:, np.argsort(nnz_in_each_features)[-200:]]
    X = X.toarray()
    # 打印数据集大小信息，单位为 MB
    print("dataset: %.3f MB" % (X.nbytes / 1e6))

# 将数据集分为训练集和测试集。相较于 LYRL2004 的分割，交换训练和测试子集，
# 以获得更大的训练数据集。
n = 23149
X_test = X[:n, :]
y_test = y[:n]
X = X[n:, :]
y = y[n:]

# 对 clfs 列表中的模型进行性能评估
clfs = bench(clfs)

# 绘制训练集上的评分图
plot_train_scores(clfs)
# 绘制测试集上的评分图
plot_test_scores(clfs)
# 绘制训练集上的损失图
plot_train_losses(clfs)
# 绘制损失函数的变化图
plot_dloss(clfs)
# 显示所有绘图
plt.show()
```