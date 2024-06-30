# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_sag.py`

```
# 导入所需的库和模块
import math  # 导入数学函数库
import re    # 导入正则表达式模块

import numpy as np     # 导入数值计算库numpy
import pytest           # 导入pytest用于测试
from scipy.special import logsumexp  # 从scipy.special导入logsumexp函数

from sklearn._loss.loss import HalfMultinomialLoss  # 从sklearn._loss.loss模块导入HalfMultinomialLoss类
from sklearn.base import clone                     # 导入sklearn.base模块中的clone函数
from sklearn.datasets import load_iris, make_blobs, make_classification  # 导入加载数据集的函数
from sklearn.linear_model import LogisticRegression, Ridge  # 导入线性模型
from sklearn.linear_model._base import make_dataset  # 导入make_dataset函数
from sklearn.linear_model._linear_loss import LinearModelLoss  # 导入LinearModelLoss类
from sklearn.linear_model._sag import get_auto_step_size  # 导入get_auto_step_size函数
from sklearn.linear_model._sag_fast import _multinomial_grad_loss_all_samples  # 导入_multinomial_grad_loss_all_samples函数
from sklearn.multiclass import OneVsRestClassifier  # 导入OneVsRestClassifier类
from sklearn.preprocessing import LabelBinarizer, LabelEncoder  # 导入数据预处理相关函数
from sklearn.utils import check_random_state, compute_class_weight  # 导入工具函数
from sklearn.utils._testing import (  # 导入用于测试的工具函数
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
)
from sklearn.utils.extmath import row_norms  # 从sklearn.utils.extmath导入row_norms函数
from sklearn.utils.fixes import CSR_CONTAINERS  # 从sklearn.utils.fixes导入CSR_CONTAINERS常量

# 加载鸢尾花数据集
iris = load_iris()


# 用于SAG分类的损失函数的导数
def log_dloss(p, y):
    z = p * y
    # 如果z值大于18.0，使用近似计算避免log的计算
    if z > 18.0:
        return math.exp(-z) * -y
    # 如果z值小于-18.0，直接返回-y
    if z < -18.0:
        return -y
    # 否则，返回-y / (exp(z) + 1.0)
    return -y / (math.exp(z) + 1.0)


# 用于SAG分类的损失函数
def log_loss(p, y):
    return np.mean(np.log(1.0 + np.exp(-y * p)))


# 用于SAG回归的损失函数的导数
def squared_dloss(p, y):
    return p - y


# 用于SAG回归的损失函数
def squared_loss(p, y):
    return np.mean(0.5 * (p - y) * (p - y))


# 计算目标函数（损失函数加上正则项）的值
def get_pobj(w, alpha, myX, myy, loss):
    w = w.ravel()
    pred = np.dot(myX, w)
    p = loss(pred, myy)
    p += alpha * w.dot(w) / 2.0
    return p


# SAG优化算法的实现
def sag(
    X,
    y,
    step_size,
    alpha,
    n_iter=1,
    dloss=None,
    sparse=False,
    sample_weight=None,
    fit_intercept=True,
    saga=False,
):
    n_samples, n_features = X.shape[0], X.shape[1]

    # 初始化权重、梯度总和和梯度内存
    weights = np.zeros(X.shape[1])
    sum_gradient = np.zeros(X.shape[1])
    gradient_memory = np.zeros((n_samples, n_features))

    # 初始化截距项相关变量
    intercept = 0.0
    intercept_sum_gradient = 0.0
    intercept_gradient_memory = np.zeros(n_samples)

    # 初始化随机数生成器和衰减因子
    rng = np.random.RandomState(77)
    decay = 1.0
    seen = set()

    # 对于稀疏数据，设置固定的衰减因子为0.01
    if sparse:
        decay = 0.01
    # 外层循环，迭代训练的次数
    for epoch in range(n_iter):
        # 内层循环，迭代数据集中的样本数
        for k in range(n_samples):
            # 随机选择一个索引
            idx = int(rng.rand() * n_samples)
            # 从数据集 X 中获取对应索引的条目
            entry = X[idx]
            # 将该索引添加到已见索引集合中
            seen.add(idx)
            # 计算线性模型的预测值
            p = np.dot(entry, weights) + intercept
            # 计算损失函数关于预测值的梯度
            gradient = dloss(p, y[idx])
            # 如果有样本权重，则乘以相应的权重
            if sample_weight is not None:
                gradient *= sample_weight[idx]
            # 计算权重的更新量
            update = entry * gradient + alpha * weights
            # 计算梯度修正量
            gradient_correction = update - gradient_memory[idx]
            # 更新总梯度
            sum_gradient += gradient_correction
            # 更新梯度记忆
            gradient_memory[idx] = update
            # 如果使用 SAGA 算法，更新权重
            if saga:
                weights -= gradient_correction * step_size * (1 - 1.0 / len(seen))

            # 如果需要拟合截距
            if fit_intercept:
                # 计算截距的梯度修正量
                gradient_correction = gradient - intercept_gradient_memory[idx]
                # 更新截距的梯度记忆
                intercept_gradient_memory[idx] = gradient
                # 更新截距的总梯度
                intercept_sum_gradient += gradient_correction
                # 计算截距的梯度修正量
                gradient_correction *= step_size * (1.0 - 1.0 / len(seen))
                # 如果使用 SAGA 算法，更新截距
                if saga:
                    intercept -= (
                        step_size * intercept_sum_gradient / len(seen) * decay
                    ) + gradient_correction
                else:
                    intercept -= step_size * intercept_sum_gradient / len(seen) * decay

            # 更新权重
            weights -= step_size * sum_gradient / len(seen)

    # 返回训练后的权重和截距
    return weights, intercept
# 定义稀疏 SAG（Stochastic Average Gradient）优化算法的函数，用于线性模型的拟合
def sag_sparse(
    X,
    y,
    step_size,
    alpha,
    n_iter=1,
    dloss=None,
    sample_weight=None,
    sparse=False,
    fit_intercept=True,
    saga=False,
    random_state=0,
):
    # 如果步长乘以正则化参数 alpha 等于 1.0，则抛出异常，因为稀疏 SAG 无法处理这种情况
    if step_size * alpha == 1.0:
        raise ZeroDivisionError(
            "Sparse sag does not handle the case step_size * alpha == 1"
        )
    
    # 获取样本数量和特征数量
    n_samples, n_features = X.shape[0], X.shape[1]

    # 初始化权重为全零向量
    weights = np.zeros(n_features)
    # 初始化梯度总和为全零向量
    sum_gradient = np.zeros(n_features)
    # 初始化最后更新时间为全零向量
    last_updated = np.zeros(n_features, dtype=int)
    # 初始化梯度记忆为全零向量
    gradient_memory = np.zeros(n_samples)
    # 使用随机状态初始化随机数生成器
    rng = check_random_state(random_state)
    # 初始化截距为 0.0
    intercept = 0.0
    # 初始化截距梯度总和为 0.0
    intercept_sum_gradient = 0.0
    # 权重缩放因子初始化为 1.0
    wscale = 1.0
    # 初始化衰减因子为 1.0
    decay = 1.0
    # 初始化一个空集合 seen
    seen = set()

    # 如果稀疏参数为 True，则将衰减因子设置为固定值 0.01
    if sparse:
        decay = 0.01

    # 计数器初始化为 0
    counter = 0
    # 外层循环，迭代训练次数
    for epoch in range(n_iter):
        # 内层循环，迭代样本数
        for k in range(n_samples):
            # 从随机数生成器中获取一个随机索引
            idx = int(rng.rand() * n_samples)
            # 获取对应索引的样本数据
            entry = X[idx]
            # 将该索引添加到已见过的索引集合中
            seen.add(idx)

            # 如果计数器大于等于1，则执行以下操作
            if counter >= 1:
                # 遍历所有特征
                for j in range(n_features):
                    # 如果上次更新的时间为0，则更新权重
                    if last_updated[j] == 0:
                        weights[j] -= c_sum[counter - 1] * sum_gradient[j]
                    else:
                        # 否则，根据时间差更新权重
                        weights[j] -= (
                            c_sum[counter - 1] - c_sum[last_updated[j] - 1]
                        ) * sum_gradient[j]
                    # 更新最后更新时间为当前计数器值
                    last_updated[j] = counter

            # 计算预测值
            p = (wscale * np.dot(entry, weights)) + intercept
            # 计算损失函数的梯度
            gradient = dloss(p, y[idx])

            # 如果存在样本权重，则将梯度乘以样本权重
            if sample_weight is not None:
                gradient *= sample_weight[idx]

            # 计算权重更新量
            update = entry * gradient
            # 计算梯度校正量
            gradient_correction = update - (gradient_memory[idx] * entry)
            # 更新总梯度
            sum_gradient += gradient_correction

            # 如果采用SAGA优化算法，则更新权重
            if saga:
                for j in range(n_features):
                    weights[j] -= (
                        gradient_correction[j]
                        * step_size
                        * (1 - 1.0 / len(seen))
                        / wscale
                    )

            # 如果拟合截距，则更新截距相关的梯度校正量
            if fit_intercept:
                gradient_correction = gradient - gradient_memory[idx]
                intercept_sum_gradient += gradient_correction
                gradient_correction *= step_size * (1.0 - 1.0 / len(seen))
                # 根据是否采用SAGA算法更新截距
                if saga:
                    intercept -= (
                        step_size * intercept_sum_gradient / len(seen) * decay
                    ) + gradient_correction
                else:
                    intercept -= step_size * intercept_sum_gradient / len(seen) * decay

            # 记录当前样本的梯度
            gradient_memory[idx] = gradient

            # 更新权重缩放因子
            wscale *= 1.0 - alpha * step_size
            # 根据计数器值更新c_sum数组
            if counter == 0:
                c_sum[0] = step_size / (wscale * len(seen))
            else:
                c_sum[counter] = c_sum[counter - 1] + step_size / (wscale * len(seen))

            # 如果计数器大于等于1且权重缩放因子小于1e-9，则执行以下操作
            if counter >= 1 and wscale < 1e-9:
                # 遍历所有特征
                for j in range(n_features):
                    # 如果上次更新时间为0，则更新权重
                    if last_updated[j] == 0:
                        weights[j] -= c_sum[counter] * sum_gradient[j]
                    else:
                        # 否则，根据时间差更新权重
                        weights[j] -= (
                            c_sum[counter] - c_sum[last_updated[j] - 1]
                        ) * sum_gradient[j]
                    # 更新最后更新时间为当前计数器值加1
                    last_updated[j] = counter + 1
                # 将当前计数器位置的c_sum置为0
                c_sum[counter] = 0
                # 对权重乘以权重缩放因子
                weights *= wscale
                # 重置权重缩放因子为1.0
                wscale = 1.0

            # 计数器加1
            counter += 1

    # 循环结束后，进行最后一次权重更新
    for j in range(n_features):
        if last_updated[j] == 0:
            weights[j] -= c_sum[counter - 1] * sum_gradient[j]
        else:
            # 否则，根据时间差更新权重
            weights[j] -= (
                c_sum[counter - 1] - c_sum[last_updated[j] - 1]
            ) * sum_gradient[j]
    # 对权重乘以最终的权重缩放因子
    weights *= wscale
    # 返回更新后的权重和截距
    return weights, intercept
# 计算步长，根据输入的数据 X，学习率 alpha，是否包含截距 fit_intercept，以及是否为分类问题 classification
def get_step_size(X, alpha, fit_intercept, classification=True):
    # 如果是分类问题
    if classification:
        # 返回步长作为四分之一除以（X 中每行平方和的最大值 + 截距 + 4.0 * alpha）
        return 4.0 / (np.max(np.sum(X * X, axis=1)) + fit_intercept + 4.0 * alpha)
    else:
        # 返回步长作为 1.0 除以（X 中每行平方和的最大值 + 截距 + alpha）
        return 1.0 / (np.max(np.sum(X * X, axis=1)) + fit_intercept + alpha)


# 测试分类器匹配函数
def test_classifier_matching():
    # 创建包含 20 个样本的两类数据集 X 和 y
    n_samples = 20
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)
    # 将 y 中值为 0 的样本标签改为 -1
    y[y == 0] = -1
    alpha = 1.1
    fit_intercept = True
    # 调用 get_step_size 函数获取步长
    step_size = get_step_size(X, alpha, fit_intercept)
    # 针对两种求解器 ["sag", "saga"] 分别进行测试
    for solver in ["sag", "saga"]:
        if solver == "sag":
            n_iter = 80
        else:
            # 对于 "saga" 求解器，设定较大的迭代次数
            n_iter = 300
        # 创建 LogisticRegression 分类器对象
        clf = LogisticRegression(
            solver=solver,
            fit_intercept=fit_intercept,
            tol=1e-11,
            C=1.0 / alpha / n_samples,
            max_iter=n_iter,
            random_state=10,
        )
        # 使用分类器拟合数据集 X, y
        clf.fit(X, y)

        # 调用 sag_sparse 函数进行优化
        weights, intercept = sag_sparse(
            X,
            y,
            step_size,
            alpha,
            n_iter=n_iter,
            dloss=log_dloss,
            fit_intercept=fit_intercept,
            saga=solver == "saga",
        )
        # 调用 sag 函数进行优化
        weights2, intercept2 = sag(
            X,
            y,
            step_size,
            alpha,
            n_iter=n_iter,
            dloss=log_dloss,
            fit_intercept=fit_intercept,
            saga=solver == "saga",
        )
        # 将权重和截距至少转换为二维数组和一维数组
        weights = np.atleast_2d(weights)
        intercept = np.atleast_1d(intercept)
        weights2 = np.atleast_2d(weights2)
        intercept2 = np.atleast_1d(intercept2)

        # 断言权重和截距的数组近似相等
        assert_array_almost_equal(weights, clf.coef_, decimal=9)
        assert_array_almost_equal(intercept, clf.intercept_, decimal=9)
        assert_array_almost_equal(weights2, clf.coef_, decimal=9)
        assert_array_almost_equal(intercept2, clf.intercept_, decimal=9)


# 测试回归器匹配函数
def test_regressor_matching():
    # 创建包含 10 个样本和 5 个特征的数据集 X 和真实权重 true_w
    n_samples = 10
    n_features = 5

    rng = np.random.RandomState(10)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    y = X.dot(true_w)

    alpha = 1.0
    n_iter = 100
    fit_intercept = True

    # 调用 get_step_size 函数获取步长
    step_size = get_step_size(X, alpha, fit_intercept, classification=False)
    # 创建 Ridge 回归器对象
    clf = Ridge(
        fit_intercept=fit_intercept,
        tol=0.00000000001,
        solver="sag",
        alpha=alpha * n_samples,
        max_iter=n_iter,
    )
    # 使用回归器拟合数据集 X, y
    clf.fit(X, y)

    # 调用 sag_sparse 函数进行优化
    weights1, intercept1 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=squared_dloss,
        fit_intercept=fit_intercept,
    )
    # 调用 sag 函数进行优化
    weights2, intercept2 = sag(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=squared_dloss,
        fit_intercept=fit_intercept,
    )

    # 断言权重和截距的数组近似相等
    assert_allclose(weights1, clf.coef_)
    assert_allclose(intercept1, clf.intercept_)
    assert_allclose(weights2, clf.coef_)
    # 使用断言检查两个值是否全部近似相等
    assert_allclose(intercept2, clf.intercept_)
# 标记测试以忽略特定的警告信息
@pytest.mark.filterwarnings("ignore:The max_iter was reached")
# 参数化测试函数，使用CSR_CONTAINERS中的不同csr_container作为参数
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_pobj_matches_logistic_regression(csr_container):
    """测试SAG优化器的目标函数与逻辑回归模型是否匹配"""
    # 设置样本数量
    n_samples = 100
    # 正则化参数alpha
    alpha = 1.0
    # 最大迭代次数
    max_iter = 20
    # 生成数据集
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)

    # 创建LogisticRegression对象clf1，使用SAG求解器
    clf1 = LogisticRegression(
        solver="sag",
        fit_intercept=False,
        tol=0.0000001,
        C=1.0 / alpha / n_samples,
        max_iter=max_iter,
        random_state=10,
    )
    # 克隆clf1，创建clf2
    clf2 = clone(clf1)
    # 创建LogisticRegression对象clf3，使用默认求解器
    clf3 = LogisticRegression(
        fit_intercept=False,
        tol=0.0000001,
        C=1.0 / alpha / n_samples,
        max_iter=max_iter,
        random_state=10,
    )

    # 分别拟合三个模型
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    clf3.fit(X, y)

    # 计算目标函数的值（pobj）
    pobj1 = get_pobj(clf1.coef_, alpha, X, y, log_loss)
    pobj2 = get_pobj(clf2.coef_, alpha, X, y, log_loss)
    pobj3 = get_pobj(clf3.coef_, alpha, X, y, log_loss)

    # 断言三个模型的目标函数值近似相等
    assert_array_almost_equal(pobj1, pobj2, decimal=4)
    assert_array_almost_equal(pobj2, pobj3, decimal=4)
    assert_array_almost_equal(pobj3, pobj1, decimal=4)


# 标记测试以忽略特定的警告信息
@pytest.mark.filterwarnings("ignore:The max_iter was reached")
# 参数化测试函数，使用CSR_CONTAINERS中的不同csr_container作为参数
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_pobj_matches_ridge_regression(csr_container):
    """测试SAG优化器的目标函数与岭回归模型是否匹配"""
    # 设置样本数量
    n_samples = 100
    # 特征数量
    n_features = 10
    # 正则化参数alpha
    alpha = 1.0
    # 最大迭代次数
    n_iter = 100
    # 是否拟合截距
    fit_intercept = False
    # 创建随机数发生器
    rng = np.random.RandomState(10)
    # 生成数据集X和真实权重true_w
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    y = X.dot(true_w)

    # 创建Ridge对象clf1，使用SAG求解器
    clf1 = Ridge(
        fit_intercept=fit_intercept,
        tol=0.00000000001,
        solver="sag",
        alpha=alpha,
        max_iter=n_iter,
        random_state=42,
    )
    # 克隆clf1，创建clf2
    clf2 = clone(clf1)
    # 创建Ridge对象clf3，使用LSQR求解器
    clf3 = Ridge(
        fit_intercept=fit_intercept,
        tol=0.00001,
        solver="lsqr",
        alpha=alpha,
        max_iter=n_iter,
        random_state=42,
    )

    # 分别拟合三个模型
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    clf3.fit(X, y)

    # 计算目标函数的值（pobj）
    pobj1 = get_pobj(clf1.coef_, alpha, X, y, squared_loss)
    pobj2 = get_pobj(clf2.coef_, alpha, X, y, squared_loss)
    pobj3 = get_pobj(clf3.coef_, alpha, X, y, squared_loss)

    # 断言三个模型的目标函数值近似相等
    assert_array_almost_equal(pobj1, pobj2, decimal=4)
    assert_array_almost_equal(pobj1, pobj3, decimal=4)
    assert_array_almost_equal(pobj3, pobj2, decimal=4)


# 标记测试以忽略特定的警告信息
@pytest.mark.filterwarnings("ignore:The max_iter was reached")
# 参数化测试函数，使用CSR_CONTAINERS中的不同csr_container作为参数
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_regressor_computed_correctly(csr_container):
    """测试SAG回归器是否计算正确"""
    # 正则化参数alpha
    alpha = 0.1
    # 特征数量
    n_features = 10
    # 样本数量
    n_samples = 40
    # 最大迭代次数
    max_iter = 100
    # 容差
    tol = 0.000001
    # 是否拟合截距
    fit_intercept = True
    # 创建随机数发生器
    rng = np.random.RandomState(0)
    # 生成数据集X和权重w
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w) + 2.0
    # 使用给定参数计算步长，用于优化过程
    step_size = get_step_size(X, alpha, fit_intercept, classification=False)

    # 创建第一个 Ridge 回归器对象
    clf1 = Ridge(
        fit_intercept=fit_intercept,  # 是否拟合截距
        tol=tol,  # 优化的收敛阈值
        solver="sag",  # 使用的优化求解器
        alpha=alpha * n_samples,  # 正则化强度乘以样本数的倍数
        max_iter=max_iter,  # 最大迭代次数
        random_state=rng,  # 随机数种子
    )
    # 克隆第一个 Ridge 回归器对象，创建第二个对象
    clf2 = clone(clf1)

    # 使用 X, y 数据拟合第一个 Ridge 回归器对象
    clf1.fit(X, y)
    # 使用 csr_container(X), y 数据拟合第二个 Ridge 回归器对象
    clf2.fit(csr_container(X), y)

    # 使用 SAG 稀疏优化算法计算稀疏权重和截距，不使用稀疏矩阵
    spweights1, spintercept1 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=max_iter,
        dloss=squared_dloss,
        fit_intercept=fit_intercept,
        random_state=rng,
    )

    # 使用 SAG 稀疏优化算法计算稀疏权重和截距，使用稀疏矩阵
    spweights2, spintercept2 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=max_iter,
        dloss=squared_dloss,
        sparse=True,
        fit_intercept=fit_intercept,
        random_state=rng,
    )

    # 断言第一个回归器的权重与稀疏结果的权重几乎相等，精确到小数点后三位
    assert_array_almost_equal(clf1.coef_.ravel(), spweights1.ravel(), decimal=3)
    # 断言第一个回归器的截距与稀疏结果的截距几乎相等，精确到小数点后一位
    assert_almost_equal(clf1.intercept_, spintercept1, decimal=1)

    # TODO: 当稀疏 Ridge 回归器带截距问题修复后取消注释 (#4710)
    # 断言第二个回归器的权重与稀疏结果的权重几乎相等，精确到小数点后三位
    # assert_array_almost_equal(clf2.coef_.ravel(),
    #                          spweights2.ravel(),
    #                          decimal=3)
    # 断言第二个回归器的截距与稀疏结果的截距几乎相等，精确到小数点后一位
    # assert_almost_equal(clf2.intercept_, spintercept2, decimal=1)
# 定义测试函数，用于测试自动步长计算函数
def test_get_auto_step_size():
    # 创建一个 NumPy 数组 X，包含三个样本，每个样本三个特征，数据类型为 float64
    X = np.array([[1, 2, 3], [2, 3, 4], [2, 3, 2]], dtype=np.float64)
    # 设置学习率 alpha
    alpha = 1.2
    # 设置是否拟合截距
    fit_intercept = False
    # 计算第二个样本的平方和，因为它是最大的
    max_squared_sum = 4 + 9 + 16
    # 调用函数 row_norms 计算 X 的每行的范数的平方，并找出最大值
    max_squared_sum_ = row_norms(X, squared=True).max()
    # 获取样本数目
    n_samples = X.shape[0]
    # 断言 max_squared_sum 与 max_squared_sum_ 几乎相等，精确度为 4 位小数
    assert_almost_equal(max_squared_sum, max_squared_sum_, decimal=4)

    # 迭代 SAGA 求解器和拟合截距的组合
    for saga in [True, False]:
        for fit_intercept in (True, False):
            if saga:
                # 如果使用 SAGA 求解器，计算正则项 L 的平方和和对数形式
                L_sqr = max_squared_sum + alpha + int(fit_intercept)
                L_log = (max_squared_sum + 4.0 * alpha + int(fit_intercept)) / 4.0
                # 计算 mun 的平方和和对数形式
                mun_sqr = min(2 * n_samples * alpha, L_sqr)
                mun_log = min(2 * n_samples * alpha, L_log)
                # 计算步长的平方和和对数形式
                step_size_sqr = 1 / (2 * L_sqr + mun_sqr)
                step_size_log = 1 / (2 * L_log + mun_log)
            else:
                # 如果不使用 SAGA 求解器，计算步长的平方和和对数形式
                step_size_sqr = 1.0 / (max_squared_sum + alpha + int(fit_intercept))
                step_size_log = 4.0 / (
                    max_squared_sum + 4.0 * alpha + int(fit_intercept)
                )

            # 调用函数 get_auto_step_size 计算自动步长，分别针对平方和和对数形式
            step_size_sqr_ = get_auto_step_size(
                max_squared_sum_,
                alpha,
                "squared",
                fit_intercept,
                n_samples=n_samples,
                is_saga=saga,
            )
            step_size_log_ = get_auto_step_size(
                max_squared_sum_,
                alpha,
                "log",
                fit_intercept,
                n_samples=n_samples,
                is_saga=saga,
            )

            # 断言计算得到的步长与函数返回的步长几乎相等，精确度为 4 位小数
            assert_almost_equal(step_size_sqr, step_size_sqr_, decimal=4)
            assert_almost_equal(step_size_log, step_size_log_, decimal=4)

    # 设置错误的损失函数类型，断言调用函数 get_auto_step_size 时抛出 ValueError 异常
    msg = "Unknown loss function for SAG solver, got wrong instead of"
    with pytest.raises(ValueError, match=msg):
        get_auto_step_size(max_squared_sum_, alpha, "wrong", fit_intercept)


# 使用参数化测试，测试 SAG 回归器的性能，测试多个种子和 CSR 容器
@pytest.mark.parametrize("seed", range(3))  # 本地测试了 1000 个种子
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_regressor(seed, csr_container):
    """测试 SAG 回归器的表现"""
    xmin, xmax = -5, 5
    n_samples = 300
    tol = 0.001
    max_iter = 100
    alpha = 0.1
    # 使用给定种子创建随机数生成器
    rng = np.random.RandomState(seed)
    # 创建包含 n_samples 个样本的线性空间
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # 简单的无噪声线性函数
    y = 0.5 * X.ravel()

    # 创建 Ridge 回归器 clf1 和 clf2，使用 SAG 求解器拟合数据
    clf1 = Ridge(
        tol=tol,
        solver="sag",
        max_iter=max_iter,
        alpha=alpha * n_samples,
        random_state=rng,
    )
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    # 计算模型的评分
    score1 = clf1.score(X, y)
    score2 = clf2.score(X, y)
    # 断言评分大于 0.98
    assert score1 > 0.98
    assert score2 > 0.98

    # 含有噪声的简单线性函数
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    # 创建 Ridge 回归器 clf1 和 clf2，使用 SAG 求解器拟合数据
    clf1 = Ridge(tol=tol, solver="sag", max_iter=max_iter, alpha=alpha * n_samples)
    clf2 = clone(clf1)
    # 使用 clf1 模型拟合数据 X 和标签 y
    clf1.fit(X, y)
    # 使用 clf2 模型拟合经过 csr_container 处理后的数据 X 和标签 y
    clf2.fit(csr_container(X), y)
    # 计算 clf1 模型在数据 X 和标签 y 上的预测准确率
    score1 = clf1.score(X, y)
    # 计算 clf2 模型在经过 csr_container 处理后的数据 X 和标签 y 上的预测准确率
    score2 = clf2.score(X, y)
    # 断言 clf1 模型的预测准确率大于 0.45，如果不满足条件将引发 AssertionError
    assert score1 > 0.45
    # 断言 clf2 模型的预测准确率大于 0.45，如果不满足条件将引发 AssertionError
    assert score2 > 0.45
@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_classifier_computed_correctly(csr_container):
    """测试二分类器是否计算正确"""

    alpha = 0.1  # 正则化参数
    n_samples = 50  # 样本数目
    n_iter = 50  # 迭代次数
    tol = 0.00001  # 公差
    fit_intercept = True  # 是否拟合截距

    # 生成用于测试的数据集
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)

    # 计算步长
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)

    # 标签处理，转化为+1和-1
    classes = np.unique(y)
    y_tmp = np.ones(n_samples)
    y_tmp[y != classes[1]] = -1
    y = y_tmp

    # 创建逻辑回归分类器实例
    clf1 = LogisticRegression(
        solver="sag",
        C=1.0 / alpha / n_samples,
        max_iter=n_iter,
        tol=tol,
        random_state=77,
        fit_intercept=fit_intercept,
    )
    clf2 = clone(clf1)

    # 使用普通数据集拟合分类器
    clf1.fit(X, y)
    # 使用csr_container函数处理后的数据集拟合分类器
    clf2.fit(csr_container(X), y)

    # 调用sag_sparse函数计算稀疏权重和截距
    spweights, spintercept = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        fit_intercept=fit_intercept,
    )
    # 使用sparse=True调用sag_sparse函数计算稀疏权重和截距
    spweights2, spintercept2 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        sparse=True,
        fit_intercept=fit_intercept,
    )

    # 断言分类器的权重和稀疏权重在指定精度下是否相等
    assert_array_almost_equal(clf1.coef_.ravel(), spweights.ravel(), decimal=2)
    # 断言分类器的截距和稀疏截距在指定精度下是否相等
    assert_almost_equal(clf1.intercept_, spintercept, decimal=1)

    # 断言另一个分类器的权重和稀疏权重在指定精度下是否相等
    assert_array_almost_equal(clf2.coef_.ravel(), spweights2.ravel(), decimal=2)
    # 断言另一个分类器的截距和稀疏截距在指定精度下是否相等
    assert_almost_equal(clf2.intercept_, spintercept2, decimal=1)


@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_multiclass_computed_correctly(csr_container):
    """测试多分类器是否计算正确"""

    alpha = 0.1  # 正则化参数
    n_samples = 20  # 样本数目
    tol = 1e-5  # 公差
    max_iter = 70  # 最大迭代次数
    fit_intercept = True  # 是否拟合截距

    # 生成用于测试的数据集
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=0, cluster_std=0.1)

    # 计算步长
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)

    # 获取唯一的类别
    classes = np.unique(y)

    # 创建OneVsRestClassifier实例，使用逻辑回归分类器
    clf1 = OneVsRestClassifier(
        LogisticRegression(
            solver="sag",
            C=1.0 / alpha / n_samples,
            max_iter=max_iter,
            tol=tol,
            random_state=77,
            fit_intercept=fit_intercept,
        )
    )
    clf2 = clone(clf1)

    # 使用普通数据集拟合分类器
    clf1.fit(X, y)
    # 使用csr_container函数处理后的数据集拟合分类器
    clf2.fit(csr_container(X), y)

    # 初始化存储权重和截距的列表
    coef1 = []
    intercept1 = []
    coef2 = []
    intercept2 = []
    # 遍历每一个类别
    for cl in classes:
        # 创建编码后的目标变量，如果类别不是当前类，则设置为-1，否则为+1
        y_encoded = np.ones(n_samples)
        y_encoded[y != cl] = -1

        # 使用SAG算法进行稀疏回归训练，返回稀疏权重和截距
        spweights1, spintercept1 = sag_sparse(
            X,
            y_encoded,
            step_size,
            alpha,
            dloss=log_dloss,
            n_iter=max_iter,
            fit_intercept=fit_intercept,
        )
        # 使用SAG算法进行稀疏回归训练，返回稀疏权重和截距（稀疏形式）
        spweights2, spintercept2 = sag_sparse(
            X,
            y_encoded,
            step_size,
            alpha,
            dloss=log_dloss,
            n_iter=max_iter,
            sparse=True,
            fit_intercept=fit_intercept,
        )
        # 将第一个模型的权重系数添加到列表中
        coef1.append(spweights1)
        # 将第一个模型的截距添加到列表中
        intercept1.append(spintercept1)

        # 将第二个模型的权重系数添加到列表中
        coef2.append(spweights2)
        # 将第二个模型的截距添加到列表中
        intercept2.append(spintercept2)

    # 将 coef1、coef2 转换为 numpy 数组形式
    coef1 = np.vstack(coef1)
    coef2 = np.vstack(coef2)
    # 将 intercept1、intercept2 转换为 numpy 数组形式
    intercept1 = np.array(intercept1)
    intercept2 = np.array(intercept2)

    # 对于每个类别，检查第一个分类器的权重系数是否与预期值相近
    for i, cl in enumerate(classes):
        assert_allclose(clf1.estimators_[i].coef_.ravel(), coef1[i], rtol=1e-2)
        # 对于每个类别，检查第一个分类器的截距是否与预期值相近（允许更大的相对误差）
        assert_allclose(clf1.estimators_[i].intercept_, intercept1[i], rtol=1e-1)

        # 对于每个类别，检查第二个分类器的权重系数是否与预期值相近
        assert_allclose(clf2.estimators_[i].coef_.ravel(), coef2[i], rtol=1e-2)
        # 对于每个类别，检查第二个分类器的截距是否与预期值相近（允许更大的相对误差，这里说明了粗略的准确性要求）
        assert_allclose(clf2.estimators_[i].intercept_, intercept2[i], rtol=5e-1)
# 使用 pytest 的 parametrize 装饰器，为每个参数化测试创建多个实例
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_classifier_results(csr_container):
    """测试分类器结果是否与目标匹配"""

    # 定义测试所需的参数
    alpha = 0.1
    n_features = 20
    n_samples = 10
    tol = 0.01
    max_iter = 200
    rng = np.random.RandomState(0)

    # 生成随机正态分布的样本矩阵 X 和权重向量 w
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # 生成样本标签 y，通过计算 X 和 w 的内积，并对其取符号得到
    y = np.dot(X, w)
    y = np.sign(y)

    # 初始化两个 LogisticRegression 分类器实例 clf1 和 clf2
    clf1 = LogisticRegression(
        solver="sag",
        C=1.0 / alpha / n_samples,
        max_iter=max_iter,
        tol=tol,
        random_state=77,
    )
    clf2 = clone(clf1)

    # 分别使用 X, y 拟合 clf1 和 csr_container(X), y 拟合 clf2
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)

    # 使用 clf1 和 clf2 预测 X 的结果 pred1 和 pred2
    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)

    # 断言预测结果 pred1 和 pred2 与真实标签 y 的近似度，精确到小数点后 12 位
    assert_almost_equal(pred1, y, decimal=12)
    assert_almost_equal(pred2, y, decimal=12)


# 使用 pytest 的 filterwarnings 装饰器，忽略特定警告信息
@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_binary_classifier_class_weight(csr_container):
    """测试带有类权重的二元分类器"""

    # 定义测试所需的参数
    alpha = 0.1
    n_samples = 50
    n_iter = 20
    tol = 0.00001
    fit_intercept = True

    # 生成用于测试的样本集 X 和相应的标签 y
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=10, cluster_std=0.1)

    # 获取步长大小
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)

    # 将标签 y 转换为二元标签
    classes = np.unique(y)
    y_tmp = np.ones(n_samples)
    y_tmp[y != classes[1]] = -1
    y = y_tmp

    # 定义类权重
    class_weight = {1: 0.45, -1: 0.55}

    # 初始化两个 LogisticRegression 分类器实例 clf1 和 clf2
    clf1 = LogisticRegression(
        solver="sag",
        C=1.0 / alpha / n_samples,
        max_iter=n_iter,
        tol=tol,
        random_state=77,
        fit_intercept=fit_intercept,
        class_weight=class_weight,
    )
    clf2 = clone(clf1)

    # 分别使用 X, y 拟合 clf1 和 csr_container(X), y 拟合 clf2
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)

    # 计算类权重
    le = LabelEncoder()
    class_weight_ = compute_class_weight(class_weight, classes=np.unique(y), y=y)
    sample_weight = class_weight_[le.fit_transform(y)]

    # 使用 sag_sparse 函数计算稀疏权重和截距 spweights, spintercept 和 spweights2, spintercept2
    spweights, spintercept = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        sample_weight=sample_weight,
        fit_intercept=fit_intercept,
    )
    spweights2, spintercept2 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        sparse=True,
        sample_weight=sample_weight,
        fit_intercept=fit_intercept,
    )

    # 断言 clf1 和 spweights 的系数及截距的近似度，精确到小数点后 2 位
    assert_array_almost_equal(clf1.coef_.ravel(), spweights.ravel(), decimal=2)
    assert_almost_equal(clf1.intercept_, spintercept, decimal=1)

    # 断言 clf2 和 spweights2 的系数及截距的近似度，精确到小数点后 2 位
    assert_array_almost_equal(clf2.coef_.ravel(), spweights2.ravel(), decimal=2)
    assert_almost_equal(clf2.intercept_, spintercept2, decimal=1)


def test_classifier_single_class():
    """测试当数据中仅有一个类别时是否抛出 ValueError 异常"""

    # 定义仅有一个类别的测试数据集 X 和标签 y
    X = [[1, 2], [3, 4]]
    y = [1, 1]

    # 期望捕获的错误消息
    msg = "This solver needs samples of at least 2 classes in the data"

    # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        LogisticRegression(solver="sag").fit(X, y)
def test_multinomial_loss():
    # test if the multinomial loss and gradient computations are consistent

    # 使用鸢尾花数据集的特征和目标值，将目标值转换为浮点数类型
    X, y = iris.data, iris.target.astype(np.float64)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # 使用随机数生成器创建权重矩阵和截距数组
    rng = check_random_state(42)
    weights = rng.randn(n_features, n_classes)
    intercept = rng.randn(n_classes)
    sample_weights = np.abs(rng.randn(n_samples))

    # 计算损失和梯度，类似于多项式 SAG 方法
    dataset, _ = make_dataset(X, y, sample_weights, random_state=42)
    loss_1, grad_1 = _multinomial_grad_loss_all_samples(
        dataset, weights, intercept, n_samples, n_features, n_classes
    )

    # 计算损失和梯度，类似于多项式 LogisticRegression 方法
    loss = LinearModelLoss(
        base_loss=HalfMultinomialLoss(n_classes=n_classes),
        fit_intercept=True,
    )
    weights_intercept = np.vstack((weights, intercept)).T
    loss_2, grad_2 = loss.loss_gradient(
        weights_intercept, X, y, l2_reg_strength=0.0, sample_weight=sample_weights
    )
    grad_2 = grad_2[:, :-1].T

    # 转换为相同的约定，即 LinearModelLoss 使用 average(loss, weight=sw)
    loss_2 *= np.sum(sample_weights)
    grad_2 *= np.sum(sample_weights)

    # 比较计算出的梯度是否一致
    assert_array_almost_equal(grad_1, grad_2)
    # 比较计算出的损失是否接近
    assert_almost_equal(loss_1, loss_2)
    # 将 loss_2 和 grad_2 根据样本权重进行加权平均化
    loss_2 *= np.sum(sample_weights)
    grad_2 *= np.sum(sample_weights)

    # 断言检查 loss_1 和 loss_2 是否几乎相等
    assert_almost_equal(loss_1, loss_2)
    # 断言检查 grad_1 和 grad_2 是否几乎相等
    assert_array_almost_equal(grad_1, grad_2)

    # 真实值（ground truth）
    # 预期的 loss_gt 值
    loss_gt = 11.680360354325961
    # 预期的 grad_gt 值，是一个 2x3 的 numpy 数组
    grad_gt = np.array(
        [[-0.557487, -1.619151, +2.176638], [-0.903942, +5.258745, -4.354803]]
    )
    # 断言检查 loss_1 是否几乎等于预期的 loss_gt
    assert_almost_equal(loss_1, loss_gt)
    # 断言检查 grad_1 是否几乎等于预期的 grad_gt
    assert_array_almost_equal(grad_1, grad_gt)
# 使用 pytest.mark.parametrize 装饰器标记该测试函数，参数化 solver 参数为 "sag" 和 "saga"
@pytest.mark.parametrize("solver", ["sag", "saga"])
# 定义测试函数，验证在 cython sag 中错误处理行为的变化是否符合预期
def test_sag_classifier_raises_error(solver):
    # 以下是对 #13316 的跟踪，cython sag 中的错误处理行为发生了变化。
    # 这仅是一个非回归测试，确保数值错误能够被正确地触发和捕获。

    # 在一个简单的问题上训练分类器
    rng = np.random.RandomState(42)
    X, y = make_classification(random_state=rng)
    # 创建 LogisticRegression 分类器实例，使用指定的 solver 参数和随机种子
    clf = LogisticRegression(solver=solver, random_state=rng, warm_start=True)
    # 使用训练数据拟合分类器
    clf.fit(X, y)

    # 通过以下操作触发数值错误：
    # - 损坏分类器的拟合系数
    # - 利用 warm_start 从当前状态重新拟合分类器
    clf.coef_[:] = np.nan

    # 使用 pytest.raises 断言捕获 ValueError 异常，并验证异常信息中是否包含 "Floating-point under-/overflow"
    with pytest.raises(ValueError, match="Floating-point under-/overflow"):
        clf.fit(X, y)
```