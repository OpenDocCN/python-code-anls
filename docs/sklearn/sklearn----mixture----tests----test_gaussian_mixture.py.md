# `D:\src\scipysrc\scikit-learn\sklearn\mixture\tests\test_gaussian_mixture.py`

```
# 导入必要的库和模块
import copy  # 导入 copy 模块，用于对象的复制
import itertools  # 导入 itertools 模块，用于高效的迭代操作
import re  # 导入 re 模块，用于正则表达式操作
import sys  # 导入 sys 模块，用于系统相关的操作
import warnings  # 导入 warnings 模块，用于警告控制
from io import StringIO  # 从 io 模块中导入 StringIO 类，用于在内存中操作文本数据
from unittest.mock import Mock  # 从 unittest.mock 模块中导入 Mock 类，用于模拟对象

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 库，用于编写和运行测试
from scipy import linalg, stats  # 从 scipy 库中导入 linalg（线性代数）和 stats（统计）模块

import sklearn  # 导入 sklearn（Scikit-Learn）机器学习库
from sklearn.cluster import KMeans  # 从 sklearn.cluster 模块中导入 KMeans 类，用于 K 均值聚类
from sklearn.covariance import EmpiricalCovariance  # 从 sklearn.covariance 模块中导入 EmpiricalCovariance 类，用于计算经验协方差矩阵
from sklearn.datasets import make_spd_matrix  # 从 sklearn.datasets 模块中导入 make_spd_matrix 函数，用于生成正定矩阵
from sklearn.exceptions import ConvergenceWarning, NotFittedError  # 从 sklearn.exceptions 模块中导入特定异常类
from sklearn.metrics.cluster import adjusted_rand_score  # 从 sklearn.metrics.cluster 模块中导入 adjusted_rand_score 函数，用于聚类评估
from sklearn.mixture import GaussianMixture  # 从 sklearn.mixture 模块中导入 GaussianMixture 类，用于高斯混合模型
from sklearn.mixture._gaussian_mixture import (  # 导入 sklearn.mixture._gaussian_mixture 模块中的多个函数和类
    _compute_log_det_cholesky,
    _compute_precision_cholesky,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_spherical,
    _estimate_gaussian_covariances_tied,
    _estimate_gaussian_parameters,
)
from sklearn.utils._testing import (  # 从 sklearn.utils._testing 模块中导入多个函数
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.extmath import fast_logdet  # 从 sklearn.utils.extmath 模块中导入 fast_logdet 函数，用于高效计算对数行列式

# 定义协方差类型的列表常量
COVARIANCE_TYPE = ["full", "tied", "diag", "spherical"]


def generate_data(n_samples, n_features, weights, means, precisions, covariance_type):
    # 创建随机数生成器对象
    rng = np.random.RandomState(0)

    X = []  # 初始化空列表 X，用于存储生成的数据
    # 根据不同的协方差类型生成数据
    if covariance_type == "spherical":
        # 对于球形协方差类型，使用不同权重、均值和精度生成多元正态分布的数据
        for _, (w, m, c) in enumerate(zip(weights, means, precisions["spherical"])):
            X.append(
                rng.multivariate_normal(
                    m, c * np.eye(n_features), int(np.round(w * n_samples))
                )
            )
    if covariance_type == "diag":
        # 对于对角协方差类型，使用不同权重、均值和精度生成多元正态分布的数据
        for _, (w, m, c) in enumerate(zip(weights, means, precisions["diag"])):
            X.append(
                rng.multivariate_normal(m, np.diag(c), int(np.round(w * n_samples)))
            )
    if covariance_type == "tied":
        # 对于共享协方差类型，使用不同权重、均值和共享精度生成多元正态分布的数据
        for _, (w, m) in enumerate(zip(weights, means)):
            X.append(
                rng.multivariate_normal(
                    m, precisions["tied"], int(np.round(w * n_samples))
                )
            )
    if covariance_type == "full":
        # 对于完全协方差类型，使用不同权重、均值和协方差矩阵生成多元正态分布的数据
        for _, (w, m, c) in enumerate(zip(weights, means, precisions["full"])):
            X.append(rng.multivariate_normal(m, c, int(np.round(w * n_samples))))

    # 将生成的数据堆叠成一个 NumPy 数组
    X = np.vstack(X)
    return X  # 返回生成的数据数组


class RandomData:
    # 待实现的 RandomData 类，用于生成随机数据集
    pass  # 占位符，表示该类的实现暂未提供
    # 初始化方法，用于生成一个混合高斯模型对象
    def __init__(self, rng, n_samples=200, n_components=2, n_features=2, scale=50):
        # 设置样本数量
        self.n_samples = n_samples
        # 设置高斯分量数量
        self.n_components = n_components
        # 设置每个样本的特征数量
        self.n_features = n_features

        # 生成指定数量的随机权重并归一化
        self.weights = rng.rand(n_components)
        self.weights = self.weights / self.weights.sum()

        # 生成随机的高斯分量均值并乘以指定的比例尺度
        self.means = rng.rand(n_components, n_features) * scale

        # 生成不同类型的协方差矩阵
        self.covariances = {
            "spherical": 0.5 + rng.rand(n_components),
            "diag": (0.5 + rng.rand(n_components, n_features)) ** 2,
            "tied": make_spd_matrix(n_features, random_state=rng),
            "full": np.array(
                [
                    make_spd_matrix(n_features, random_state=rng) * 0.5
                    for _ in range(n_components)
                ]
            ),
        }

        # 计算每种协方差类型对应的精度矩阵
        self.precisions = {
            "spherical": 1.0 / self.covariances["spherical"],
            "diag": 1.0 / self.covariances["diag"],
            "tied": linalg.inv(self.covariances["tied"]),
            "full": np.array(
                [linalg.inv(covariance) for covariance in self.covariances["full"]]
            ),
        }

        # 使用指定的高斯分布参数生成样本数据
        self.X = dict(
            zip(
                COVARIANCE_TYPE,
                [
                    generate_data(
                        n_samples,
                        n_features,
                        self.weights,
                        self.means,
                        self.covariances,
                        covar_type,
                    )
                    for covar_type in COVARIANCE_TYPE
                ],
            )
        )

        # 生成响应变量 Y，将每个分量对应的样本标签堆叠在一起
        self.Y = np.hstack(
            [
                np.full(int(np.round(w * n_samples)), k, dtype=int)
                for k, w in enumerate(self.weights)
            ]
        )
# 测试高斯混合模型的属性
def test_gaussian_mixture_attributes():
    # 测试不良参数
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)

    # 测试良好参数
    n_components, tol, n_init, max_iter, reg_covar = 2, 1e-4, 3, 30, 1e-1
    covariance_type, init_params = "full", "random"
    # 创建高斯混合模型对象，并使用给定参数进行拟合
    gmm = GaussianMixture(
        n_components=n_components,
        tol=tol,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=reg_covar,
        covariance_type=covariance_type,
        init_params=init_params,
    ).fit(X)

    # 断言各属性值是否符合预期
    assert gmm.n_components == n_components
    assert gmm.covariance_type == covariance_type
    assert gmm.tol == tol
    assert gmm.reg_covar == reg_covar
    assert gmm.max_iter == max_iter
    assert gmm.n_init == n_init
    assert gmm.init_params == init_params


def test_check_weights():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    n_components = rand_data.n_components
    X = rand_data.X["full"]

    g = GaussianMixture(n_components=n_components)

    # 检查权重矩阵形状不良
    weights_bad_shape = rng.rand(n_components, 1)
    g.weights_init = weights_bad_shape
    msg = re.escape(
        "The parameter 'weights' should have the shape of "
        f"({n_components},), but got {str(weights_bad_shape.shape)}"
    )
    # 使用 pytest 断言引发特定异常并匹配消息
    with pytest.raises(ValueError, match=msg):
        g.fit(X)

    # 检查权重值超出范围
    weights_bad_range = rng.rand(n_components) + 1
    g.weights_init = weights_bad_range
    msg = re.escape(
        "The parameter 'weights' should be in the range [0, 1], but got"
        f" max value {np.min(weights_bad_range):.5f}, "
        f"min value {np.max(weights_bad_range):.5f}"
    )
    with pytest.raises(ValueError, match=msg):
        g.fit(X)

    # 检查权重未正规化
    weights_bad_norm = rng.rand(n_components)
    weights_bad_norm = weights_bad_norm / (weights_bad_norm.sum() + 1)
    g.weights_init = weights_bad_norm
    msg = re.escape(
        "The parameter 'weights' should be normalized, "
        f"but got sum(weights) = {np.sum(weights_bad_norm):.5f}"
    )
    with pytest.raises(ValueError, match=msg):
        g.fit(X)

    # 检查良好的权重矩阵
    weights = rand_data.weights
    g = GaussianMixture(weights_init=weights, n_components=n_components)
    g.fit(X)
    # 使用 assert_array_equal 断言两个数组相等
    assert_array_equal(weights, g.weights_init)


def test_check_means():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    n_components, n_features = rand_data.n_components, rand_data.n_features
    X = rand_data.X["full"]

    g = GaussianMixture(n_components=n_components)

    # 检查均值矩阵形状不良
    means_bad_shape = rng.rand(n_components + 1, n_features)
    g.means_init = means_bad_shape
    msg = "The parameter 'means' should have the shape of "
    # 使用 pytest 断言引发特定异常并匹配消息
    with pytest.raises(ValueError, match=msg):
        g.fit(X)

    # 检查良好的均值矩阵
    means = rand_data.means
    g.means_init = means
    g.fit(X)
    # 使用 assert_array_equal 断言两个数组相等
    assert_array_equal(means, g.means_init)


def test_check_precisions():
    # 待实现
    # 创建一个随机数生成器对象，种子为0
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建 RandomData 对象
    rand_data = RandomData(rng)

    # 从 rand_data 对象中获取 n_components 和 n_features 属性值
    n_components, n_features = rand_data.n_components, rand_data.n_features

    # 定义不同协方差类型下的“坏”精度矩阵
    precisions_bad_shape = {
        "full": np.ones((n_components + 1, n_features, n_features)),
        "tied": np.ones((n_features + 1, n_features + 1)),
        "diag": np.ones((n_components + 1, n_features)),
        "spherical": np.ones((n_components + 1)),
    }

    # 定义非正定的精度矩阵
    precisions_not_pos = np.ones((n_components, n_features, n_features))
    precisions_not_pos[0] = np.eye(n_features)
    precisions_not_pos[0, 0, 0] = -1.0

    # 定义非正定的精度矩阵字典
    precisions_not_positive = {
        "full": precisions_not_pos,
        "tied": precisions_not_pos[0],
        "diag": np.full((n_components, n_features), -1.0),
        "spherical": np.full(n_components, -1.0),
    }

    # 不是正定时的错误消息
    not_positive_errors = {
        "full": "symmetric, positive-definite",
        "tied": "symmetric, positive-definite",
        "diag": "positive",
        "spherical": "positive",
    }

    # 遍历每种协方差类型
    for covar_type in COVARIANCE_TYPE:
        # 从 RandomData 对象中获取特定协方差类型的数据集 X
        X = RandomData(rng).X[covar_type]
        # 创建高斯混合模型对象
        g = GaussianMixture(
            n_components=n_components, covariance_type=covar_type, random_state=rng
        )

        # 检查具有错误形状的精度矩阵
        g.precisions_init = precisions_bad_shape[covar_type]
        msg = f"The parameter '{covar_type} precision' should have the shape of"
        # 使用 pytest 检查是否引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            g.fit(X)

        # 检查非正定的精度矩阵
        g.precisions_init = precisions_not_positive[covar_type]
        msg = f"'{covar_type} precision' should be {not_positive_errors[covar_type]}"
        # 使用 pytest 检查是否引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            g.fit(X)

        # 检查正确初始化的精度矩阵
        g.precisions_init = rand_data.precisions[covar_type]
        # 对模型进行拟合
        g.fit(X)
        # 断言 g.precisions_init 是否与 rand_data 对象中的精度矩阵一致
        assert_array_equal(rand_data.precisions[covar_type], g.precisions_init)
def test_suffstat_sk_full():
    # 比较使用 EmpiricalCovariance.covariance 拟合 X*sqrt(resp) 后的精度矩阵
    # 与 _sufficient_sk_full, n_components=1 生成的结果

    rng = np.random.RandomState(0)
    n_samples, n_features = 500, 2

    # 特殊情况1，假设数据已经"居中"
    X = rng.rand(n_samples, n_features)
    resp = rng.rand(n_samples, 1)
    X_resp = np.sqrt(resp) * X
    nk = np.array([n_samples])
    xk = np.zeros((1, n_features))

    # 使用给定的 resp, X, nk, xk, 0 估计高斯协方差
    covars_pred = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)

    # 使用 EmpiricalCovariance 拟合居中的 X_resp
    ecov = EmpiricalCovariance(assume_centered=True)
    ecov.fit(X_resp)

    # 断言计算得到的精度矩阵与预测的协方差之间的误差极小
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm="frobenius"), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm="spectral"), 0)

    # 检查精度计算
    precs_chol_pred = _compute_precision_cholesky(covars_pred, "full")
    precs_pred = np.array([np.dot(prec, prec.T) for prec in precs_chol_pred])
    precs_est = np.array([linalg.inv(cov) for cov in covars_pred])
    assert_array_almost_equal(precs_est, precs_pred)

    # 特殊情况2，假设 resp 全部为1
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean(axis=0).reshape((1, -1))

    # 使用给定的 resp, X, nk, xk, 0 估计高斯协方差
    covars_pred = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)

    # 使用 EmpiricalCovariance 拟合未居中的 X
    ecov = EmpiricalCovariance(assume_centered=False)
    ecov.fit(X)

    # 断言计算得到的精度矩阵与预测的协方差之间的误差极小
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm="frobenius"), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm="spectral"), 0)

    # 检查精度计算
    precs_chol_pred = _compute_precision_cholesky(covars_pred, "full")
    precs_pred = np.array([np.dot(prec, prec.T) for prec in precs_chol_pred])
    precs_est = np.array([linalg.inv(cov) for cov in covars_pred])
    assert_array_almost_equal(precs_est, precs_pred)


def test_suffstat_sk_tied():
    # 使用等式 Nk * Sk / N = S_tied 计算协方差矩阵

    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 500, 2, 2

    # 生成随机的 resp，并归一化
    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)

    # 计算 nk 和 xk
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]

    # 使用给定的 resp, X, nk, xk, 0 估计完全协方差矩阵
    covars_pred_full = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)

    # 计算并合并所有 nk * Sk / N 的结果得到 S_tied
    covars_pred_tied = _estimate_gaussian_covariances_tied(resp, X, nk, xk, 0)

    # 使用 EmpiricalCovariance 类来设置预测的完全协方差矩阵
    ecov = EmpiricalCovariance()
    ecov.covariance_ = covars_pred_full

    # 断言计算得到的 S_tied 与预测的完全协方差矩阵之间的误差极小
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm="frobenius"), 0)
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm="spectral"), 0)

    # 检查精度计算
    precs_chol_pred = _compute_precision_cholesky(covars_pred_tied, "tied")
    precs_pred = np.dot(precs_chol_pred, precs_chol_pred.T)
    precs_est = linalg.inv(covars_pred_tied)
    # 断言预测精度估计值与预测精度实际值几乎相等
    assert_array_almost_equal(precs_est, precs_pred)
def test_suffstat_sk_diag():
    # 对 'full' 情况进行测试
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 500, 2, 2

    # 生成随机响应矩阵
    resp = rng.rand(n_samples, n_components)
    # 归一化响应矩阵
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    # 生成随机样本矩阵 X
    X = rng.rand(n_samples, n_features)
    # 计算每个组件的样本权重和均值
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    # 使用完全协方差矩阵估计函数预测协方差矩阵
    covars_pred_full = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)
    # 使用对角线协方差矩阵估计函数预测协方差矩阵
    covars_pred_diag = _estimate_gaussian_covariances_diag(resp, X, nk, xk, 0)

    # 创建 EmpiricalCovariance 对象
    ecov = EmpiricalCovariance()
    # 对预测的协方差矩阵进行比较
    for cov_full, cov_diag in zip(covars_pred_full, covars_pred_diag):
        # 将完全协方差矩阵转换为对角线协方差矩阵
        ecov.covariance_ = np.diag(np.diag(cov_full))
        # 将对角线协方差矩阵直接赋值
        cov_diag = np.diag(cov_diag)
        # 断言对角线协方差矩阵与计算误差的 Frobenius 范数和谱范数接近于零
        assert_almost_equal(ecov.error_norm(cov_diag, norm="frobenius"), 0)
        assert_almost_equal(ecov.error_norm(cov_diag, norm="spectral"), 0)

    # 检查精度计算
    precs_chol_pred = _compute_precision_cholesky(covars_pred_diag, "diag")
    assert_almost_equal(covars_pred_diag, 1.0 / precs_chol_pred**2)


def test_gaussian_suffstat_sk_spherical():
    # 计算球形协方差等于将数据展平后的一维数据的方差，n_components=1
    rng = np.random.RandomState(0)
    n_samples, n_features = 500, 2

    # 生成随机样本矩阵 X
    X = rng.rand(n_samples, n_features)
    # 减去均值
    X = X - X.mean()
    # 创建全为1的响应矩阵
    resp = np.ones((n_samples, 1))
    # 设置样本权重和均值
    nk = np.array([n_samples])
    xk = X.mean()
    # 使用球形协方差估计函数预测协方差矩阵
    covars_pred_spherical = _estimate_gaussian_covariances_spherical(resp, X, nk, xk, 0)
    # 计算球形协方差矩阵的预测值
    covars_pred_spherical2 = np.dot(X.flatten().T, X.flatten()) / (
        n_features * n_samples
    )
    # 断言球形协方差矩阵与预测值接近
    assert_almost_equal(covars_pred_spherical, covars_pred_spherical2)

    # 检查精度计算
    precs_chol_pred = _compute_precision_cholesky(covars_pred_spherical, "spherical")
    assert_almost_equal(covars_pred_spherical, 1.0 / precs_chol_pred**2)


def test_compute_log_det_cholesky():
    n_features = 2
    rand_data = RandomData(np.random.RandomState(0))

    for covar_type in COVARIANCE_TYPE:
        covariance = rand_data.covariances[covar_type]

        if covar_type == "full":
            # 对完全协方差类型计算行列式
            predected_det = np.array([linalg.det(cov) for cov in covariance])
        elif covar_type == "tied":
            # 对绑定协方差类型计算行列式
            predected_det = linalg.det(covariance)
        elif covar_type == "diag":
            # 对对角线协方差类型计算行列式
            predected_det = np.array([np.prod(cov) for cov in covariance])
        elif covar_type == "spherical":
            # 对球形协方差类型计算行列式
            predected_det = covariance**n_features

        # 计算协方差矩阵的 Cholesky 分解的对数行列式
        expected_det = _compute_log_det_cholesky(
            _compute_precision_cholesky(covariance, covar_type),
            covar_type,
            n_features=n_features,
        )
        # 断言预期的对数行列式与计算的对数行列式接近
        assert_array_almost_equal(expected_det, -0.5 * np.log(predected_det))


def _naive_lmvnpdf_diag(X, means, covars):
    # 创建空的响应矩阵
    resp = np.empty((len(X), len(means)))
    # 计算协方差矩阵的标准差
    stds = np.sqrt(covars)
    # 遍历 means 和 stds 的元素，并且使用索引 i 进行迭代
    for i, (mean, std) in enumerate(zip(means, stds)):
        # 计算 X 每行数据在正态分布 N(mean, std) 下的对数概率密度，并求和
        resp[:, i] = stats.norm.logpdf(X, mean, std).sum(axis=1)
    # 返回计算得到的 resp 数组作为结果
    return resp
# 定义测试函数，用于测试高斯混合模型的对数概率
def test_gaussian_mixture_log_probabilities():
    # 导入所需的函数 _estimate_log_gaussian_prob
    from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob

    # 使用随机种子创建随机数生成器对象
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建 RandomData 对象
    rand_data = RandomData(rng)
    # 设置样本数和特征数
    n_samples = 500
    n_features = rand_data.n_features
    # 获取随机数据对象中的混合成分数
    n_components = rand_data.n_components

    # 获取随机数据对象中的均值
    means = rand_data.means
    # 随机生成对角协方差矩阵
    covars_diag = rng.rand(n_components, n_features)
    # 随机生成样本数据矩阵
    X = rng.rand(n_samples, n_features)
    # 使用 _naive_lmvnpdf_diag 函数计算对数概率
    log_prob_naive = _naive_lmvnpdf_diag(X, means, covars_diag)

    # 计算完整协方差的精度矩阵
    precs_full = np.array([np.diag(1.0 / np.sqrt(x)) for x in covars_diag])

    # 使用 _estimate_log_gaussian_prob 函数计算对数概率
    log_prob = _estimate_log_gaussian_prob(X, means, precs_full, "full")
    # 断言结果是否近似相等
    assert_array_almost_equal(log_prob, log_prob_naive)

    # 对角协方差的精度矩阵
    precs_chol_diag = 1.0 / np.sqrt(covars_diag)
    # 使用 _estimate_log_gaussian_prob 函数计算对数概率
    log_prob = _estimate_log_gaussian_prob(X, means, precs_chol_diag, "diag")
    # 断言结果是否近似相等
    assert_array_almost_equal(log_prob, log_prob_naive)

    # 共享协方差
    covars_tied = np.array([x for x in covars_diag]).mean(axis=0)
    # 计算共享协方差的精度矩阵
    precs_tied = np.diag(np.sqrt(1.0 / covars_tied))

    # 使用 _naive_lmvnpdf_diag 函数计算对数概率
    log_prob_naive = _naive_lmvnpdf_diag(X, means, [covars_tied] * n_components)
    # 使用 _estimate_log_gaussian_prob 函数计算对数概率
    log_prob = _estimate_log_gaussian_prob(X, means, precs_tied, "tied")

    # 断言结果是否近似相等
    assert_array_almost_equal(log_prob, log_prob_naive)

    # 球面协方差
    covars_spherical = covars_diag.mean(axis=1)
    # 计算球面协方差的精度矩阵
    precs_spherical = 1.0 / np.sqrt(covars_diag.mean(axis=1))
    # 使用 _naive_lmvnpdf_diag 函数计算对数概率
    log_prob_naive = _naive_lmvnpdf_diag(
        X, means, [[k] * n_features for k in covars_spherical]
    )
    # 使用 _estimate_log_gaussian_prob 函数计算对数概率
    log_prob = _estimate_log_gaussian_prob(X, means, precs_spherical, "spherical")
    # 断言结果是否近似相等
    assert_array_almost_equal(log_prob, log_prob_naive)


# 跳过对 weighted_log_probabilities, log_weights 的测试


# 定义测试函数，用于测试高斯混合模型的对数概率响应
def test_gaussian_mixture_estimate_log_prob_resp():
    # 测试响应是否被归一化
    # 使用随机种子创建随机数生成器对象
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建 RandomData 对象，设置比例为 5
    rand_data = RandomData(rng, scale=5)
    # 获取随机数据对象中的样本数和特征数
    n_samples = rand_data.n_samples
    n_features = rand_data.n_features
    # 获取随机数据对象中的混合成分数
    n_components = rand_data.n_components

    # 随机生成样本数据矩阵
    X = rng.rand(n_samples, n_features)
    # 对于每种协方差类型
    for covar_type in COVARIANCE_TYPE:
        # 获取随机数据对象中的权重
        weights = rand_data.weights
        # 获取随机数据对象中的均值
        means = rand_data.means
        # 获取随机数据对象中的精度
        precisions = rand_data.precisions[covar_type]
        # 创建高斯混合模型对象
        g = GaussianMixture(
            n_components=n_components,
            random_state=rng,
            weights_init=weights,
            means_init=means,
            precisions_init=precisions,
            covariance_type=covar_type,
        )
        # 使用样本数据 X 拟合模型
        g.fit(X)
        # 预测样本 X 的概率响应
        resp = g.predict_proba(X)
        # 断言概率响应是否被归一化
        assert_array_almost_equal(resp.sum(axis=1), np.ones(n_samples))
        # 断言初始权重是否相等
        assert_array_equal(g.weights_init, weights)
        # 断言初始均值是否相等
        assert_array_equal(g.means_init, means)
        # 断言初始精度是否相等
        assert_array_equal(g.precisions_init, precisions)


# 跳过对 weighted_log_probabilities, log_weights 的测试


# 定义测试函数，用于测试高斯混合模型的预测和预测概率
def test_gaussian_mixture_predict_predict_proba():
    # 使用随机种子创建随机数生成器对象
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建 RandomData 对象
    rand_data = RandomData(rng)
    # 遍历定义的协方差类型列表，例如"full"、"tied"等
    for covar_type in COVARIANCE_TYPE:
        # 从随机生成的数据对象 rand_data 中获取对应协方差类型的特征数据 X
        X = rand_data.X[covar_type]
        # 获取随机生成的数据对象 rand_data 中的标签数据 Y
        Y = rand_data.Y
        # 创建高斯混合模型对象 g
        g = GaussianMixture(
            n_components=rand_data.n_components,  # 混合成分的数量
            random_state=rng,  # 随机数生成器的状态
            weights_init=rand_data.weights,  # 初始权重
            means_init=rand_data.means,  # 初始均值
            precisions_init=rand_data.precisions[covar_type],  # 初始精度矩阵
            covariance_type=covar_type,  # 协方差矩阵的类型
        )

        # 检查在未执行 fit 操作时是否会产生警告消息
        msg = (
            "This GaussianMixture instance is not fitted yet. Call 'fit' "
            "with appropriate arguments before using this estimator."
        )
        # 使用 pytest 断言检查是否会抛出 NotFittedError 异常，并匹配特定的错误消息
        with pytest.raises(NotFittedError, match=msg):
            g.predict(X)

        # 对数据 X 执行拟合操作
        g.fit(X)
        # 使用拟合后的模型预测数据 X 的类标签
        Y_pred = g.predict(X)
        # 计算预测类别的概率并取最大概率对应的类别
        Y_pred_proba = g.predict_proba(X).argmax(axis=1)
        # 使用断言确保预测的类标签 Y_pred 与最大概率的类别 Y_pred_proba 相等
        assert_array_equal(Y_pred, Y_pred_proba)
        # 使用调整兰德指数（Adjusted Rand Score）检查预测结果与真实标签 Y 的相似度是否大于 0.95
        assert adjusted_rand_score(Y, Y_pred) > 0.95
# 设置测试用例的过滤警告，忽略所有“did not converge”警告
@pytest.mark.filterwarnings("ignore:.*did not converge.*")

# 使用参数化装饰器定义多个测试用例，分别测试不同的种子(seed)、最大迭代次数(max_iter)和容忍度(tol)
@pytest.mark.parametrize(
    "seed, max_iter, tol",
    [
        (0, 2, 1e-7),  # 严格的非收敛情况
        (1, 2, 1e-1),  # 松散的非收敛情况
        (3, 300, 1e-7),  # 严格的收敛情况
        (4, 300, 1e-1),  # 松散的收敛情况
    ],
)
# 定义测试函数，测试高斯混合模型的拟合和预测功能
def test_gaussian_mixture_fit_predict(seed, max_iter, tol):
    # 使用给定种子创建随机数生成器
    rng = np.random.RandomState(seed)
    # 使用随机数生成器创建随机数据
    rand_data = RandomData(rng)
    
    # 针对每种协方差类型进行测试
    for covar_type in COVARIANCE_TYPE:
        # 从随机数据中选择相应的特征数据
        X = rand_data.X[covar_type]
        # 获取随机数据的标签
        Y = rand_data.Y
        
        # 创建高斯混合模型对象
        g = GaussianMixture(
            n_components=rand_data.n_components,
            random_state=rng,
            weights_init=rand_data.weights,
            means_init=rand_data.means,
            precisions_init=rand_data.precisions[covar_type],
            covariance_type=covar_type,
            max_iter=max_iter,
            tol=tol,
        )

        # 深拷贝模型对象
        f = copy.deepcopy(g)
        
        # 使用拟合后的模型进行预测，并比较两种预测方法的结果是否一致
        Y_pred1 = f.fit(X).predict(X)
        Y_pred2 = g.fit_predict(X)
        assert_array_equal(Y_pred1, Y_pred2)
        
        # 检查调整后的兰德指数是否大于0.95
        assert adjusted_rand_score(Y, Y_pred2) > 0.95


# 定义测试函数，测试高斯混合模型在n_init大于1时的拟合和预测功能
def test_gaussian_mixture_fit_predict_n_init():
    # 创建随机数据矩阵
    X = np.random.RandomState(0).randn(1000, 5)
    # 创建高斯混合模型对象，设置n_init为5
    gm = GaussianMixture(n_components=5, n_init=5, random_state=0)
    # 使用模型对象进行拟合和预测，并比较两种方法的结果是否一致
    y_pred1 = gm.fit_predict(X)
    y_pred2 = gm.predict(X)
    assert_array_equal(y_pred1, y_pred2)


# 定义测试函数，测试高斯混合模型的拟合功能
def test_gaussian_mixture_fit():
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建随机数据
    rand_data = RandomData(rng)
    # 获取随机数据的特征数和组件数
    n_features = rand_data.n_features
    n_components = rand_data.n_components
    # 遍历每种协方差类型
    for covar_type in COVARIANCE_TYPE:
        # 从随机数据中获取特定协方差类型的数据集
        X = rand_data.X[covar_type]
        # 创建高斯混合模型对象
        g = GaussianMixture(
            n_components=n_components,   # 指定高斯混合模型的组件数量
            n_init=20,                   # 每个组件的初始化次数
            reg_covar=0,                 # 协方差矩阵的正则化参数
            random_state=rng,            # 随机数生成器的状态
            covariance_type=covar_type,  # 指定协方差矩阵的类型
        )
        # 对数据进行拟合
        g.fit(X)

        # 断言：验证模型的权重与随机数据的权重排序后的接近程度
        # 需要更多数据以确保在 rtol=1e-7 的条件下通过测试
        assert_allclose(
            np.sort(g.weights_), np.sort(rand_data.weights), rtol=0.1, atol=1e-2
        )

        # 对均值进行排序，并进行比较
        arg_idx1 = g.means_[:, 0].argsort()
        arg_idx2 = rand_data.means[:, 0].argsort()
        assert_allclose(
            g.means_[arg_idx1], rand_data.means[arg_idx2], rtol=0.1, atol=1e-2
        )

        # 根据协方差类型不同，选择不同的预测精度和测试精度
        if covar_type == "full":
            prec_pred = g.precisions_
            prec_test = rand_data.precisions["full"]
        elif covar_type == "tied":
            prec_pred = np.array([g.precisions_] * n_components)
            prec_test = np.array([rand_data.precisions["tied"]] * n_components)
        elif covar_type == "spherical":
            prec_pred = np.array([np.eye(n_features) * c for c in g.precisions_])
            prec_test = np.array(
                [np.eye(n_features) * c for c in rand_data.precisions["spherical"]]
            )
        elif covar_type == "diag":
            prec_pred = np.array([np.diag(d) for d in g.precisions_])
            prec_test = np.array([np.diag(d) for d in rand_data.precisions["diag"]])

        # 计算迹的排序并进行比较
        arg_idx1 = np.trace(prec_pred, axis1=1, axis2=2).argsort()
        arg_idx2 = np.trace(prec_test, axis1=1, axis2=2).argsort()
        for k, h in zip(arg_idx1, arg_idx2):
            # 创建经验协方差对象
            ecov = EmpiricalCovariance()
            # 设置经验协方差对象的协方差矩阵为测试精度中的第h个
            ecov.covariance_ = prec_test[h]
            # 断言：误差归一化的准确性依赖于数据量和随机性 rng
            assert_allclose(ecov.error_norm(prec_pred[k]), 0, atol=0.15)
def test_gaussian_mixture_fit_best_params():
    # 设置随机数生成器，确保结果可复现
    rng = np.random.RandomState(0)
    # 使用随机数据生成器创建随机数据实例
    rand_data = RandomData(rng)
    # 获取组件数
    n_components = rand_data.n_components
    # 初始化迭代次数
    n_init = 10
    # 遍历不同协方差类型
    for covar_type in COVARIANCE_TYPE:
        # 从随机数据中选择相应协方差类型的数据集
        X = rand_data.X[covar_type]
        # 创建高斯混合模型对象
        g = GaussianMixture(
            n_components=n_components,
            n_init=1,  # 初始拟合次数
            reg_covar=0,  # 协方差矩阵正则化参数
            random_state=rng,  # 随机状态
            covariance_type=covar_type,  # 协方差类型
        )
        # 初始化空列表用于存储每次拟合的对数似然值
        ll = []
        # 进行多次初始化拟合
        for _ in range(n_init):
            g.fit(X)  # 对数据进行拟合
            ll.append(g.score(X))  # 计算并记录对数似然值
        ll = np.array(ll)  # 转换为 NumPy 数组
        # 创建具有多次初始化的高斯混合模型对象
        g_best = GaussianMixture(
            n_components=n_components,
            n_init=n_init,  # 初始拟合次数
            reg_covar=0,  # 协方差矩阵正则化参数
            random_state=rng,  # 随机状态
            covariance_type=covar_type,  # 协方差类型
        )
        # 使用最佳参数进行拟合
        g_best.fit(X)
        # 断言：最小对数似然值应接近最佳拟合模型的对数似然值
        assert_almost_equal(ll.min(), g_best.score(X))


def test_gaussian_mixture_fit_convergence_warning():
    # 设置随机数生成器，确保结果可复现
    rng = np.random.RandomState(0)
    # 使用随机数据生成器创建随机数据实例，设置比例为1
    rand_data = RandomData(rng, scale=1)
    # 获取组件数
    n_components = rand_data.n_components
    # 设置最大迭代次数
    max_iter = 1
    # 遍历不同协方差类型
    for covar_type in COVARIANCE_TYPE:
        # 从随机数据中选择相应协方差类型的数据集
        X = rand_data.X[covar_type]
        # 创建高斯混合模型对象
        g = GaussianMixture(
            n_components=n_components,
            n_init=1,  # 初始拟合次数
            max_iter=max_iter,  # 最大迭代次数
            reg_covar=0,  # 协方差矩阵正则化参数
            random_state=rng,  # 随机状态
            covariance_type=covar_type,  # 协方差类型
        )
        # 定义警告信息
        msg = (
            "Best performing initialization did not converge. "
            "Try different init parameters, or increase max_iter, "
            "tol, or check for degenerate data."
        )
        # 断言：拟合过程中应发出收敛警告，并匹配指定警告信息
        with pytest.warns(ConvergenceWarning, match=msg):
            g.fit(X)


def test_multiple_init():
    # 测试多次初始化是否能够比单次初始化更好
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 5, 2
    X = rng.randn(n_samples, n_features)
    # 遍历不同协方差类型
    for cv_type in COVARIANCE_TYPE:
        # 使用单次初始化拟合的高斯混合模型，并计算评分
        train1 = (
            GaussianMixture(
                n_components=n_components, covariance_type=cv_type, random_state=0
            )
            .fit(X)
            .score(X)
        )
        # 使用多次初始化拟合的高斯混合模型，并计算评分
        train2 = (
            GaussianMixture(
                n_components=n_components,
                covariance_type=cv_type,
                random_state=0,
                n_init=5,  # 多次初始化
            )
            .fit(X)
            .score(X)
        )
        # 断言：多次初始化应不差于单次初始化
        assert train2 >= train1


def test_gaussian_mixture_n_parameters():
    # 测试模型估计的参数数量是否正确
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 5, 2
    X = rng.randn(n_samples, n_features)
    # 预期每种协方差类型对应的参数数量
    n_params = {"spherical": 13, "diag": 21, "tied": 26, "full": 41}
    # 遍历不同协方差类型
    for cv_type in COVARIANCE_TYPE:
        # 创建并拟合高斯混合模型
        g = GaussianMixture(
            n_components=n_components, covariance_type=cv_type, random_state=rng
        ).fit(X)
        # 断言：模型估计的参数数量应与预期一致
        assert g._n_parameters() == n_params[cv_type]


def test_bic_1d_1component():
    # 测试一维数据和单个组件情况下不同协方差类型的 BIC 分数是否相同
    # 使用 NumPy 提供的随机数生成器创建随机状态对象 rng，种子为 0
    rng = np.random.RandomState(0)
    # 定义数据集的样本数、维度和高斯混合模型的组件数
    n_samples, n_dim, n_components = 100, 1, 1
    # 生成符合标准正态分布的数据集 X，大小为 (n_samples, n_dim)
    X = rng.randn(n_samples, n_dim)
    
    # 使用完全协方差类型的高斯混合模型拟合数据 X，并计算其 Bayesian Information Criterion (BIC)
    bic_full = (
        GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=rng
        )
        .fit(X)  # 对数据 X 进行拟合
        .bic(X)  # 计算 BIC 值
    )
    
    # 针对不同的协方差类型 ["tied", "diag", "spherical"] 分别计算对应的 BIC 值，并与完全协方差类型的 BIC 值进行比较
    for covariance_type in ["tied", "diag", "spherical"]:
        bic = (
            GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=rng,
            )
            .fit(X)  # 对数据 X 进行拟合
            .bic(X)  # 计算 BIC 值
        )
        # 断言当前计算得到的 BIC 值与完全协方差类型的 BIC 值近似相等，用于测试模型的一致性
        assert_almost_equal(bic_full, bic)
# 测试高斯混合模型的 AIC 和 BIC 准则
def test_gaussian_mixture_aic_bic():
    # 创建随机数生成器对象，种子为 0
    rng = np.random.RandomState(0)
    # 定义样本数量、特征数量和混合组件数量
    n_samples, n_features, n_components = 50, 3, 2
    # 生成服从标准正态分布的样本数据矩阵 X
    X = rng.randn(n_samples, n_features)
    # 计算标准高斯分布的熵
    sgh = 0.5 * (
        fast_logdet(np.cov(X.T, bias=1)) + n_features * (1 + np.log(2 * np.pi))
    )
    # 遍历协方差类型列表 COVARIANCE_TYPE
    for cv_type in COVARIANCE_TYPE:
        # 创建高斯混合模型对象 g，指定混合组件数、协方差类型和随机数种子等参数
        g = GaussianMixture(
            n_components=n_components,
            covariance_type=cv_type,
            random_state=rng,
            max_iter=200,
        )
        # 使用样本数据 X 拟合高斯混合模型 g
        g.fit(X)
        # 计算 AIC 和 BIC 指标
        aic = 2 * n_samples * sgh + 2 * g._n_parameters()
        bic = 2 * n_samples * sgh + np.log(n_samples) * g._n_parameters()
        # 计算上界 bound
        bound = n_features / np.sqrt(n_samples)
        # 断言 AIC 和 BIC 的相对误差小于 bound
        assert (g.aic(X) - aic) / n_samples < bound
        assert (g.bic(X) - bic) / n_samples < bound


# 测试高斯混合模型的详细输出
def test_gaussian_mixture_verbose():
    # 创建随机数生成器对象，种子为 0
    rng = np.random.RandomState(0)
    # 创建 RandomData 对象 rand_data，使用随机数生成器 rng 初始化
    rand_data = RandomData(rng)
    # 从 RandomData 对象中获取混合组件数量 n_components
    n_components = rand_data.n_components
    # 遍历协方差类型列表 COVARIANCE_TYPE
    for covar_type in COVARIANCE_TYPE:
        # 从 RandomData 对象中获取特定协方差类型的样本数据 X
        X = rand_data.X[covar_type]
        # 创建高斯混合模型对象 g，指定混合组件数、初始化次数、正则化协方差矩阵、随机数种子等参数，开启详细输出
        g = GaussianMixture(
            n_components=n_components,
            n_init=1,
            reg_covar=0,
            random_state=rng,
            covariance_type=covar_type,
            verbose=1,
        )
        # 创建另一个高斯混合模型对象 h，与 g 参数相同，但输出详细程度更高
        h = GaussianMixture(
            n_components=n_components,
            n_init=1,
            reg_covar=0,
            random_state=rng,
            covariance_type=covar_type,
            verbose=2,
        )
        # 保存标准输出对象，用于后续比较
        old_stdout = sys.stdout
        # 重定向标准输出到内存中的 StringIO 对象
        sys.stdout = StringIO()
        try:
            # 使用样本数据 X 拟合模型 g 和 h
            g.fit(X)
            h.fit(X)
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout


# 使用参数化测试忽略警告，测试高斯混合模型的热启动功能
@pytest.mark.filterwarnings("ignore:.*did not converge.*")
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_warm_start(seed):
    # 设置随机数种子
    random_state = seed
    # 创建随机数生成器对象 rng，种子为 seed
    rng = np.random.RandomState(random_state)
    # 定义样本数量、特征数量和混合组件数量
    n_samples, n_features, n_components = 500, 2, 2
    # 生成服从均匀分布的样本数据矩阵 X
    X = rng.rand(n_samples, n_features)

    # 创建高斯混合模型对象 g，指定混合组件数、初始化次数、最大迭代次数、正则化协方差矩阵、随机数种子等参数，关闭热启动
    g = GaussianMixture(
        n_components=n_components,
        n_init=1,
        max_iter=2,
        reg_covar=0,
        random_state=random_state,
        warm_start=False,
    )
    # 创建高斯混合模型对象 h，与 g 参数相同，但开启热启动
    h = GaussianMixture(
        n_components=n_components,
        n_init=1,
        max_iter=1,
        reg_covar=0,
        random_state=random_state,
        warm_start=True,
    )

    # 使用样本数据 X 拟合模型 g
    g.fit(X)
    # 计算 h 拟合后的分数 score1
    score1 = h.fit(X).score(X)
    # 再次拟合 h 并计算分数 score2
    score2 = h.fit(X).score(X)

    # 断言 g 和 h 的权重、均值、精度几乎相等
    assert_almost_equal(g.weights_, h.weights_)
    assert_almost_equal(g.means_, h.means_)
    assert_almost_equal(g.precisions_, h.precisions_)
    # 断言 score2 比 score1 更高
    assert score2 > score1

    # 创建高斯混合模型对象 g，指定混合组件数、初始化次数、最大迭代次数、正则化协方差矩阵、随机数种子等参数，关闭热启动，设置收敛阈值
    g = GaussianMixture(
        n_components=n_components,
        n_init=1,
        max_iter=5,
        reg_covar=0,
        random_state=random_state,
        warm_start=False,
        tol=1e-6,
    )
    # 创建一个高斯混合模型对象 h，用于聚类分析
    h = GaussianMixture(
        n_components=n_components,   # 指定高斯混合模型中的组件数量
        n_init=1,                   # 每个组件运行的初始化数量
        max_iter=5,                 # 每个期间的最大迭代次数
        reg_covar=0,                # 协方差的正则化参数（在这里设为零）
        random_state=random_state,  # 随机数生成器的种子，用于重现结果
        warm_start=True,            # 如果设置为 True，则继续上次调用的解决方案
        tol=1e-6,                   # 迭代停止的容差阈值
    )

    # 使用数据 X 对高斯混合模型 g 进行拟合
    g.fit(X)
    
    # 确保模型 g 没有收敛，即仍然有未收敛的情况
    assert not g.converged_

    # 使用数据 X 对模型 h 进行拟合，尝试使其收敛
    h.fit(X)
    
    # 根据数据的不同，由于数据的完全随机性，可能需要多次重新拟合以实现收敛
    for _ in range(1000):
        h.fit(X)
        # 如果模型 h 已经收敛，则退出循环
        if h.converged_:
            break
    
    # 确保模型 h 已经收敛
    assert h.converged_
# 使用装饰器忽略收敛警告，确保测试函数不会因此而中断
@ignore_warnings(category=ConvergenceWarning)
# 测试函数：检查在 warm_start=True 的情况下是否能检测到收敛
def test_convergence_detected_with_warm_start():
    # 使用种子为0的随机数生成器创建随机数据对象
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    # 获取随机数据的组件数量
    n_components = rand_data.n_components
    # 从随机数据中获取完整数据集 X
    X = rand_data.X["full"]

    # 遍历不同的最大迭代次数
    for max_iter in (1, 2, 50):
        # 创建高斯混合模型对象
        gmm = GaussianMixture(
            n_components=n_components,
            warm_start=True,  # 开启热启动以便迭代时重用上一次拟合结果
            max_iter=max_iter,
            random_state=rng,
        )
        # 进行多次拟合
        for _ in range(100):
            gmm.fit(X)
            # 如果模型已经收敛，则停止迭代
            if gmm.converged_:
                break
        # 断言模型已经收敛
        assert gmm.converged_
        # 断言实际迭代次数不超过指定的最大迭代次数
        assert max_iter >= gmm.n_iter_


# 测试函数：检查 score 方法的功能
def test_score():
    covar_type = "full"
    # 使用种子为0的随机数生成器创建随机数据对象
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    # 获取随机数据的组件数量
    n_components = rand_data.n_components
    # 从随机数据中获取协方差类型为 full 的数据集 X
    X = rand_data.X[covar_type]

    # 检查未调用 fit 方法时的错误消息
    gmm1 = GaussianMixture(
        n_components=n_components,
        n_init=1,
        max_iter=1,
        reg_covar=0,
        random_state=rng,
        covariance_type=covar_type,
    )
    # 期望的错误消息
    msg = (
        "This GaussianMixture instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    # 使用 pytest 断言捕获特定的异常消息
    with pytest.raises(NotFittedError, match=msg):
        gmm1.score(X)

    # 忽略警告，拟合模型
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm1.fit(X)
    # 计算模型得分
    gmm_score = gmm1.score(X)
    # 计算模型得分的概率值
    gmm_score_proba = gmm1.score_samples(X).mean()
    # 断言两个得分值近似相等
    assert_almost_equal(gmm_score, gmm_score_proba)

    # 检查新模型的得分是否增加
    gmm2 = GaussianMixture(
        n_components=n_components,
        n_init=1,
        reg_covar=0,
        random_state=rng,
        covariance_type=covar_type,
    ).fit(X)
    # 断言新模型的得分高于旧模型的得分
    assert gmm2.score(X) > gmm1.score(X)


# 测试函数：检查 score_samples 方法的功能
def test_score_samples():
    covar_type = "full"
    # 使用种子为0的随机数生成器创建随机数据对象
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    # 获取随机数据的组件数量
    n_components = rand_data.n_components
    # 从随机数据中获取协方差类型为 full 的数据集 X
    X = rand_data.X[covar_type]

    # 检查未调用 fit 方法时的错误消息
    gmm = GaussianMixture(
        n_components=n_components,
        n_init=1,
        reg_covar=0,
        random_state=rng,
        covariance_type=covar_type,
    )
    # 期望的错误消息
    msg = (
        "This GaussianMixture instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    # 使用 pytest 断言捕获特定的异常消息
    with pytest.raises(NotFittedError, match=msg):
        gmm.score_samples(X)

    # 拟合模型并计算 score_samples
    gmm_score_samples = gmm.fit(X).score_samples(X)
    # 断言 score_samples 返回的数组形状与随机数据集的样本数相匹配
    assert gmm_score_samples.shape[0] == rand_data.n_samples


# 测试函数：检查 EM 算法每一步的无正则化改善训练集似然性
def test_monotonic_likelihood():
    # 我们检查 EM 算法每一步是否能无正则化地提高训练集的似然性
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    # 获取随机数据的组件数量
    n_components = rand_data.n_components
    # 对每种协方差类型进行迭代
    for covar_type in COVARIANCE_TYPE:
        # 从随机数据中获取对应协方差类型的数据集 X
        X = rand_data.X[covar_type]
        # 创建高斯混合模型对象
        gmm = GaussianMixture(
            n_components=n_components,        # 混合成分的数量
            covariance_type=covar_type,       # 协方差类型
            reg_covar=0,                      # 协方差矩阵的正则化参数
            warm_start=True,                  # 是否热启动
            max_iter=1,                       # 每次迭代的最大次数
            random_state=rng,                 # 随机数种子
            tol=1e-7,                         # 收敛阈值
        )
        # 初始化当前的对数似然为负无穷
        current_log_likelihood = -np.inf
        # 忽略收敛警告，执行训练过程
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            # 通过一次次迭代训练模型，确保每次迭代后训练对数似然都增加
            for _ in range(600):
                # 保存上一次的对数似然值
                prev_log_likelihood = current_log_likelihood
                # 计算当前的对数似然值并更新模型
                current_log_likelihood = gmm.fit(X).score(X)
                # 断言当前的对数似然值大于等于上一次的对数似然值
                assert current_log_likelihood >= prev_log_likelihood

                # 如果模型已经收敛，则结束训练
                if gmm.converged_:
                    break

            # 断言模型已经收敛
            assert gmm.converged_
# 定义测试正则化的函数
def test_regularisation():
    # 使用随机数种子0创建随机数生成器
    rng = np.random.RandomState(0)
    # 设定样本数和特征数
    n_samples, n_features = 10, 5

    # 创建输入数据 X，分为两个簇，一个全为1，一个全为0
    X = np.vstack(
        (np.ones((n_samples // 2, n_features)), np.zeros((n_samples // 2, n_features)))
    )

    # 遍历协方差类型列表 COVARIANCE_TYPE
    for covar_type in COVARIANCE_TYPE:
        # 创建高斯混合模型对象 gmm
        gmm = GaussianMixture(
            n_components=n_samples,
            reg_covar=0,
            covariance_type=covar_type,
            random_state=rng,
        )

        # 忽略运行时警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # 设置匹配错误消息的正则表达式
            msg = re.escape(
                "Fitting the mixture model failed because some components have"
                " ill-defined empirical covariance (for instance caused by "
                "singleton or collapsed samples). Try to decrease the number "
                "of components, or increase reg_covar."
            )
            # 断言拟合过程中抛出 ValueError 异常，并匹配特定消息
            with pytest.raises(ValueError, match=msg):
                gmm.fit(X)

            # 设置 reg_covar 参数为 1e-6，并重新拟合模型
            gmm.set_params(reg_covar=1e-6).fit(X)


# 定义测试属性的函数
def test_property():
    # 使用随机数种子0创建随机数生成器
    rng = np.random.RandomState(0)
    # 创建 RandomData 对象 rand_data，设置数据规模为7
    rand_data = RandomData(rng, scale=7)
    # 获取数据集的组件数量
    n_components = rand_data.n_components

    # 遍历协方差类型列表 COVARIANCE_TYPE
    for covar_type in COVARIANCE_TYPE:
        # 从 RandomData 对象中选择相应协方差类型的数据集 X
        X = rand_data.X[covar_type]
        # 创建高斯混合模型对象 gmm
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covar_type,
            random_state=rng,
            n_init=5,
        )
        # 拟合数据集 X 到高斯混合模型
        gmm.fit(X)
        # 如果协方差类型是 "full"
        if covar_type == "full":
            # 对每个精度矩阵 prec 和协方差矩阵 covar 施行近似相等的数组断言
            for prec, covar in zip(gmm.precisions_, gmm.covariances_):
                assert_array_almost_equal(linalg.inv(prec), covar)
        # 如果协方差类型是 "tied"
        elif covar_type == "tied":
            # 对精度矩阵的逆和协方差矩阵施行近似相等的数组断言
            assert_array_almost_equal(linalg.inv(gmm.precisions_), gmm.covariances_)
        # 其他情况
        else:
            # 对精度矩阵和协方差的倒数施行近似相等的数组断言
            assert_array_almost_equal(gmm.precisions_, 1.0 / gmm.covariances_)


# 定义测试采样的函数
def test_sample():
    # 使用随机数种子0创建随机数生成器
    rng = np.random.RandomState(0)
    # 创建 RandomData 对象 rand_data，设置数据规模为7，组件数量为3
    rand_data = RandomData(rng, scale=7, n_components=3)
    # 获取特征数和组件数量
    n_features, n_components = rand_data.n_features, rand_data.n_components
    # 遍历每种协方差类型
    for covar_type in COVARIANCE_TYPE:
        # 从随机数据中获取对应协方差类型的数据集
        X = rand_data.X[covar_type]

        # 创建一个高斯混合模型对象，设定组件数量、协方差类型和随机种子
        gmm = GaussianMixture(
            n_components=n_components, covariance_type=covar_type, random_state=rng
        )

        # 检查是否需要拟合高斯混合模型才能进行抽样
        msg = "This GaussianMixture instance is not fitted"
        with pytest.raises(NotFittedError, match=msg):
            gmm.sample(0)
        
        # 拟合高斯混合模型
        gmm.fit(X)

        # 检查抽样时样本数量参数是否有效
        msg = "Invalid value for 'n_samples'"
        with pytest.raises(ValueError, match=msg):
            gmm.sample(0)

        # 执行抽样操作，获取生成的样本数据和对应的标签
        n_samples = 20000
        X_s, y_s = gmm.sample(n_samples)

        # 验证每个组件的协方差矩阵是否与抽样数据的协方差矩阵接近
        for k in range(n_components):
            if covar_type == "full":
                assert_array_almost_equal(
                    gmm.covariances_[k], np.cov(X_s[y_s == k].T), decimal=1
                )
            elif covar_type == "tied":
                assert_array_almost_equal(
                    gmm.covariances_, np.cov(X_s[y_s == k].T), decimal=1
                )
            elif covar_type == "diag":
                assert_array_almost_equal(
                    gmm.covariances_[k], np.diag(np.cov(X_s[y_s == k].T)), decimal=1
                )
            else:
                assert_array_almost_equal(
                    gmm.covariances_[k],
                    np.var(X_s[y_s == k] - gmm.means_[k]),
                    decimal=1,
                )

        # 计算生成样本数据的均值并与模型估计的均值进行比较
        means_s = np.array([np.mean(X_s[y_s == k], 0) for k in range(n_components)])
        assert_array_almost_equal(gmm.means_, means_s, decimal=1)

        # 检查生成数据的形状是否符合预期，即 (n_samples, n_features)
        assert X_s.shape == (n_samples, n_features)

        # 检查不同样本数量下生成数据的形状是否正确
        for sample_size in range(1, 100):
            X_s, _ = gmm.sample(sample_size)
            assert X_s.shape == (sample_size, n_features)
@ignore_warnings(category=ConvergenceWarning)
def test_init():
    # 忽略收敛警告的测试函数装饰器

    # 循环15次，测试不同的随机种子
    for random_state in range(15):
        # 使用随机种子创建随机数据对象
        rand_data = RandomData(
            np.random.RandomState(random_state), n_samples=50, scale=1
        )
        # 获取随机数据的组件数和完整数据集
        n_components = rand_data.n_components
        X = rand_data.X["full"]

        # 创建两个高斯混合模型，分别设置不同的n_init值和max_iter=1进行拟合
        gmm1 = GaussianMixture(
            n_components=n_components, n_init=1, max_iter=1, random_state=random_state
        ).fit(X)
        gmm2 = GaussianMixture(
            n_components=n_components, n_init=10, max_iter=1, random_state=random_state
        ).fit(X)

        # 断言第二个模型的下界(lower_bound_)大于等于第一个模型的下界
        assert gmm2.lower_bound_ >= gmm1.lower_bound_


def test_gaussian_mixture_setting_best_params():
    """`GaussianMixture`的最佳参数，`n_iter_`和`lower_bound_`
    在发散情况下必须得到适当设置。

    非回归测试，用于:
    https://github.com/scikit-learn/scikit-learn/issues/18216
    """
    # 使用随机种子0创建随机数据集
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = rnd.uniform(size=(n_samples, 3))

    # 已知这些初始化参数会导致模型发散
    means_init = np.array(
        [
            [0.670637869618158, 0.21038256107384043, 0.12892629765485303],
            [0.09394051075844147, 0.5759464955561779, 0.929296197576212],
            [0.5033230372781258, 0.9569852381759425, 0.08654043447295741],
            [0.18578301420435747, 0.5531158970919143, 0.19388943970532435],
            [0.4548589928173794, 0.35182513658825276, 0.568146063202464],
            [0.609279894978321, 0.7929063819678847, 0.9620097270828052],
        ]
    )
    precisions_init = np.array(
        [
            999999.999604483,
            999999.9990869573,
            553.7603944542167,
            204.78596008931834,
            15.867423501783637,
            85.4595728389735,
        ]
    )
    weights_init = [
        0.03333333333333341,
        0.03333333333333341,
        0.06666666666666674,
        0.06666666666666674,
        0.7000000000000001,
        0.10000000000000007,
    ]

    # 创建高斯混合模型对象，指定各种初始化参数
    gmm = GaussianMixture(
        covariance_type="spherical",
        reg_covar=0,
        means_init=means_init,
        weights_init=weights_init,
        random_state=rnd,
        n_components=len(weights_init),
        precisions_init=precisions_init,
        max_iter=1,
    )
    # 确保在拟合过程中不会抛出错误
    gmm.fit(X)

    # 检查模型是否没有收敛
    assert not gmm.converged_

    # 检查模型参数是否正确设置
    for attr in [
        "weights_",
        "means_",
        "covariances_",
        "precisions_cholesky_",
        "n_iter_",
        "lower_bound_",
    ]:
        assert hasattr(gmm, attr)


@pytest.mark.parametrize(
    "init_params", ["random", "random_from_data", "k-means++", "kmeans"]
)
def test_init_means_not_duplicated(init_params, global_random_seed):
    # 参数化测试，测试不同的初始化方法，用于初始化模型的均值是否重复
    # 使用全局随机种子初始化随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 使用随机数生成器创建 RandomData 对象，指定数据缩放参数为 5
    rand_data = RandomData(rng, scale=5)
    # 获取 RandomData 对象中的成分数量
    n_components = rand_data.n_components
    # 从 RandomData 对象中获取完整数据集 X
    X = rand_data.X["full"]

    # 创建高斯混合模型对象（Gaussian Mixture Model，GMM）
    # 设定成分数量、初始化参数、随机数生成器和最大迭代次数为 0
    gmm = GaussianMixture(
        n_components=n_components, init_params=init_params, random_state=rng, max_iter=0
    )
    # 使用数据集 X 对 GMM 进行拟合
    gmm.fit(X)

    # 获取拟合后的各个成分的均值
    means = gmm.means_
    # 对所有可能的均值组合进行遍历，确保它们不完全相等
    for i_mean, j_mean in itertools.combinations(means, r=2):
        # 使用 np.allclose 检查两个均值数组是否在数值上非常接近
        assert not np.allclose(i_mean, j_mean)
@pytest.mark.parametrize(
    "init_params", ["random", "random_from_data", "k-means++", "kmeans"]
)
# 定义测试函数，用于检查不同初始化方法对高斯混合模型的影响
def test_means_for_all_inits(init_params, global_random_seed):
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 创建随机数据对象，指定数据的尺度为5
    rand_data = RandomData(rng, scale=5)
    # 获取数据中的混合成分数目
    n_components = rand_data.n_components
    # 从随机数据对象中获取完整数据集
    X = rand_data.X["full"]

    # 创建高斯混合模型对象
    gmm = GaussianMixture(
        n_components=n_components, init_params=init_params, random_state=rng
    )
    # 对数据集进行拟合
    gmm.fit(X)

    # 断言检查拟合后的均值数组的形状是否正确
    assert gmm.means_.shape == (n_components, X.shape[1])
    # 断言检查拟合后的均值是否在数据集特征维度的范围内
    assert np.all(X.min(axis=0) <= gmm.means_)
    assert np.all(gmm.means_ <= X.max(axis=0))
    # 断言检查模型是否收敛
    assert gmm.converged_


# 定义测试函数，用于检查当 max_iter=0 时，初始值设定的正确性
def test_max_iter_zero():
    # 使用随机种子创建随机数生成器
    rng = np.random.RandomState(0)
    # 创建随机数据对象，指定数据的尺度为5
    rand_data = RandomData(rng, scale=5)
    # 获取数据中的混合成分数目
    n_components = rand_data.n_components
    # 从随机数据对象中获取完整数据集
    X = rand_data.X["full"]
    # 手动设定初始均值
    means_init = [[20, 30], [30, 25]]
    # 创建高斯混合模型对象，将 max_iter 设为0，以确保返回初始值
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=rng,
        means_init=means_init,
        tol=1e-06,
        max_iter=0,
    )
    # 对数据集进行拟合
    gmm.fit(X)

    # 断言检查拟合后的均值数组是否与初始值一致
    assert_allclose(gmm.means_, means_init)


# 定义测试函数，用于检查当手动提供精度矩阵时是否正确初始化 precision_cholesky_
def test_gaussian_mixture_precisions_init_diag():
    """Check that we properly initialize `precision_cholesky_` when we manually
    provide the precision matrix.

    In this regard, we check the consistency between estimating the precision
    matrix and providing the same precision matrix as initialization. It should
    lead to the same results with the same number of iterations.

    If the initialization is wrong then the number of iterations will increase.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/16944
    """
    # 生成一个玩具数据集
    n_samples = 300
    # 使用随机种子创建随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个平移后的高斯分布数据集
    shifted_gaussian = rng.randn(n_samples, 2) + np.array([20, 20])
    # 创建一个经过线性变换后的高斯分布数据集
    C = np.array([[0.0, -0.7], [3.5, 0.7]])
    stretched_gaussian = np.dot(rng.randn(n_samples, 2), C)
    # 将两个数据集垂直堆叠在一起
    X = np.vstack([shifted_gaussian, stretched_gaussian])

    # 设置检验精度初始化一致性的通用参数
    n_components, covariance_type, reg_covar, random_state = 2, "diag", 1e-6, 0

    # 执行手动初始化以计算精度矩阵:
    # - 运行 KMeans 得到初始猜测
    # - 估计协方差
    # - 从估计的协方差计算精度矩阵
    resp = np.zeros((X.shape[0], n_components))
    label = (
        KMeans(n_clusters=n_components, n_init=1, random_state=random_state)
        .fit(X)
        .labels_
    )
    resp[np.arange(X.shape[0]), label] = 1
    _, _, covariance = _estimate_gaussian_parameters(
        X, resp, reg_covar=reg_covar, covariance_type=covariance_type
    )
    precisions_init = 1 / covariance
    # 创建一个具有初始精度矩阵的高斯混合模型，并使用给定数据 X 进行拟合
    gm_with_init = GaussianMixture(
        n_components=n_components,         # 指定高斯混合模型的组件数量
        covariance_type=covariance_type,   # 指定协方差矩阵的类型（如 'full', 'tied', 'diag', 'spherical'）
        reg_covar=reg_covar,               # 指定用于稳定化协方差估计的正则化因子
        precisions_init=precisions_init,   # 指定初始精度矩阵
        random_state=random_state,         # 指定随机种子以确保结果的可重复性
    ).fit(X)                               # 使用数据 X 拟合模型

    # 创建一个没有初始精度矩阵的高斯混合模型，并使用给定数据 X 进行拟合
    gm_without_init = GaussianMixture(
        n_components=n_components,         # 指定高斯混合模型的组件数量
        covariance_type=covariance_type,   # 指定协方差矩阵的类型（如 'full', 'tied', 'diag', 'spherical'）
        reg_covar=reg_covar,               # 指定用于稳定化协方差估计的正则化因子
        random_state=random_state,         # 指定随机种子以确保结果的可重复性
    ).fit(X)                               # 使用数据 X 拟合模型

    # 断言：验证没有初始精度矩阵和有初始精度矩阵两个模型的迭代次数相同
    assert gm_without_init.n_iter_ == gm_with_init.n_iter_

    # 断言：验证有初始精度矩阵和没有初始精度矩阵两个模型的 Cholesky 分解后的精度矩阵近似相等
    assert_allclose(
        gm_with_init.precisions_cholesky_, gm_without_init.precisions_cholesky_
    )
# 使用给定的种子(seed)随机生成样本和责任(resp)
def _generate_data(seed, n_samples, n_features, n_components):
    rs = np.random.RandomState(seed)
    # 生成一个 n_samples x n_features 的随机样本矩阵
    X = rs.random_sample((n_samples, n_features))
    # 生成一个 n_samples x n_components 的随机责任(resp)矩阵，并进行归一化处理
    resp = rs.random_sample((n_samples, n_components))
    resp /= resp.sum(axis=1)[:, np.newaxis]
    return X, resp


# 计算给定协方差类型下的样本 X 的精度矩阵及其 Cholesky 分解
def _calculate_precisions(X, resp, covariance_type):
    reg_covar = 1e-6  # 正则化协方差
    # 估计高斯模型的参数：权重(weights)、均值(means)、协方差(covariances)
    weights, means, covariances = _estimate_gaussian_parameters(
        X, resp, reg_covar, covariance_type
    )
    # 计算协方差矩阵的 Cholesky 分解
    precisions_cholesky = _compute_precision_cholesky(covariances, covariance_type)

    _, n_components = resp.shape
    # 实例化一个 GaussianMixture 模型，以便使用其 `_set_parameters` 方法来设置
    # `precisions_` 和 `precisions_cholesky_`，以匹配提供的 `covariance_type`
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    params = (weights, means, covariances, precisions_cholesky)
    gmm._set_parameters(params)
    return gmm.precisions_, gmm.precisions_cholesky_


# 使用参数化测试对 GaussianMixture 的精度矩阵初始化进行测试
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPE)
def test_gaussian_mixture_precisions_init(covariance_type, global_random_seed):
    """Non-regression test for #26415."""
    
    # 生成数据
    X, resp = _generate_data(
        seed=global_random_seed,
        n_samples=100,
        n_features=3,
        n_components=4,
    )
    
    # 计算精度矩阵初始化值和期望的 Cholesky 分解
    precisions_init, desired_precisions_cholesky = _calculate_precisions(
        X, resp, covariance_type
    )
    
    # 创建 GaussianMixture 模型，使用给定的精度矩阵初始化值
    gmm = GaussianMixture(
        covariance_type=covariance_type, precisions_init=precisions_init
    )
    # 初始化模型
    gmm._initialize(X, resp)
    # 获取实际的 Cholesky 分解值
    actual_precisions_cholesky = gmm.precisions_cholesky_
    # 断言实际的 Cholesky 分解值与期望的 Cholesky 分解值相近
    assert_allclose(actual_precisions_cholesky, desired_precisions_cholesky)


# 对于只有单个组件的 GaussianMixture，进行稳定性测试
def test_gaussian_mixture_single_component_stable():
    """
    Non-regression test for #23032 ensuring 1-component GM works on only a
    few samples.
    """
    rng = np.random.RandomState(0)
    # 生成一个二维空间中的三个样本，符合多元正态分布
    X = rng.multivariate_normal(np.zeros(2), np.identity(2), size=3)
    # 创建一个只有一个组件的 GaussianMixture 模型
    gm = GaussianMixture(n_components=1)
    # 对模型进行拟合并生成样本
    gm.fit(X).sample()


# 当提供了所有初始化参数时，不对高斯模型参数进行估计
def test_gaussian_mixture_all_init_does_not_estimate_gaussian_parameters(
    monkeypatch,
    global_random_seed,
):
    """When all init parameters are provided, the Gaussian parameters
    are not estimated.

    Non-regression test for gh-26015.
    """
    # 使用 Mock 对象替换 _estimate_gaussian_parameters 函数
    mock = Mock(side_effect=_estimate_gaussian_parameters)
    monkeypatch.setattr(
        sklearn.mixture._gaussian_mixture, "_estimate_gaussian_parameters", mock
    )

    rng = np.random.RandomState(global_random_seed)
    rand_data = RandomData(rng)

    # 创建 GaussianMixture 模型，使用提供的所有初始化参数
    gm = GaussianMixture(
        n_components=rand_data.n_components,
        weights_init=rand_data.weights,
        means_init=rand_data.means,
        precisions_init=rand_data.precisions["full"],
        random_state=rng,
    )
    gm.fit(rand_data.X["full"])
    # 使用高斯混合模型 (GaussianMixture) 对数据进行拟合，输入数据为 rand_data.X["full"]
    # 这里的 gm.fit() 方法用于拟合模型参数，但不估计初始的高斯参数，它们在每次 M 步骤中进行估计。

    assert mock.call_count == gm.n_iter_
    # 断言语句，用于检查模拟对象 mock 的方法调用次数是否等于 gm.n_iter_
    # gm.n_iter_ 表示高斯混合模型的迭代次数，通常是算法迭代的次数。
```