# `D:\src\scipysrc\scikit-learn\sklearn\mixture\tests\test_bayesian_mixture.py`

```
# 导入必要的库和模块
import copy  # 导入深拷贝函数

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架
from scipy.special import gammaln  # 导入gamma函数的对数变换

from sklearn.exceptions import ConvergenceWarning, NotFittedError  # 导入异常类
from sklearn.metrics.cluster import adjusted_rand_score  # 导入调整兰德指数
from sklearn.mixture import BayesianGaussianMixture  # 导入贝叶斯高斯混合模型
from sklearn.mixture._bayesian_mixture import _log_dirichlet_norm, _log_wishart_norm  # 导入内部函数
from sklearn.mixture.tests.test_gaussian_mixture import RandomData  # 导入测试数据生成类
from sklearn.utils._testing import (  # 导入测试工具函数
    assert_almost_equal,
    assert_array_equal,
    ignore_warnings,
)

COVARIANCE_TYPE = ["full", "tied", "diag", "spherical"]  # 高斯混合模型的协方差矩阵类型
PRIOR_TYPE = ["dirichlet_process", "dirichlet_distribution"]  # 高斯混合模型的先验类型


def test_log_dirichlet_norm():
    rng = np.random.RandomState(0)  # 创建随机数生成器对象

    weight_concentration = rng.rand(2)  # 生成两个随机权重集中度
    expected_norm = gammaln(np.sum(weight_concentration)) - np.sum(
        gammaln(weight_concentration)
    )  # 计算预期的狄利克雷分布的对数归一化常数
    predected_norm = _log_dirichlet_norm(weight_concentration)  # 调用计算狄利克雷分布对数归一化常数的函数

    assert_almost_equal(expected_norm, predected_norm)  # 断言预期值与计算值接近


def test_log_wishart_norm():
    rng = np.random.RandomState(0)  # 创建随机数生成器对象

    n_components, n_features = 5, 2  # 定义混合分布的组件数和特征数
    degrees_of_freedom = np.abs(rng.rand(n_components)) + 1.0  # 生成自由度数组
    log_det_precisions_chol = n_features * np.log(range(2, 2 + n_components))  # 计算精度矩阵行列式的对数

    expected_norm = np.empty(5)  # 创建空数组用于存储预期的狄利克雷分布对数归一化常数
    for k, (degrees_of_freedom_k, log_det_k) in enumerate(
        zip(degrees_of_freedom, log_det_precisions_chol)
    ):
        expected_norm[k] = -(
            degrees_of_freedom_k * (log_det_k + 0.5 * n_features * np.log(2.0))
            + np.sum(
                gammaln(
                    0.5
                    * (degrees_of_freedom_k - np.arange(0, n_features)[:, np.newaxis])
                ),
                0,
            )
        ).item()  # 计算每个组件的Wishart分布对数归一化常数
    predected_norm = _log_wishart_norm(
        degrees_of_freedom, log_det_precisions_chol, n_features
    )  # 调用计算Wishart分布对数归一化常数的函数

    assert_almost_equal(expected_norm, predected_norm)  # 断言预期值与计算值接近


def test_bayesian_mixture_weights_prior_initialisation():
    rng = np.random.RandomState(0)  # 创建随机数生成器对象
    n_samples, n_components, n_features = 10, 5, 2  # 定义样本数、组件数和特征数
    X = rng.rand(n_samples, n_features)  # 生成随机数据

    # 检查给定权重集中度的情况下是否正确初始化
    weight_concentration_prior = rng.rand()
    bgmm = BayesianGaussianMixture(
        weight_concentration_prior=weight_concentration_prior, random_state=rng
    ).fit(X)  # 创建贝叶斯高斯混合模型对象并拟合数据
    assert_almost_equal(weight_concentration_prior, bgmm.weight_concentration_prior_)  # 断言权重集中度是否正确初始化

    # 检查默认权重集中度的情况下是否正确初始化
    bgmm = BayesianGaussianMixture(n_components=n_components, random_state=rng).fit(X)  # 创建贝叶斯高斯混合模型对象并拟合数据
    assert_almost_equal(1.0 / n_components, bgmm.weight_concentration_prior_)  # 断言权重集中度是否正确初始化为默认值


def test_bayesian_mixture_mean_prior_initialisation():
    rng = np.random.RandomState(0)  # 创建随机数生成器对象
    n_samples, n_components, n_features = 10, 3, 2  # 定义样本数、组件数和特征数
    X = rng.rand(n_samples, n_features)  # 生成随机数据

    # 检查给定均值精度先验的情况下是否正确初始化
    # 使用随机数生成器 rng 生成一个均匀分布的随机数，作为均值精度的先验值
    mean_precision_prior = rng.rand()
    
    # 使用贝叶斯高斯混合模型（BayesianGaussianMixture）对数据 X 进行拟合
    bgmm = BayesianGaussianMixture(
        mean_precision_prior=mean_precision_prior, random_state=rng
    ).fit(X)
    
    # 断言均值精度的先验值与拟合后的模型中的值几乎相等
    assert_almost_equal(mean_precision_prior, bgmm.mean_precision_prior_)
    
    # 检查默认均值精度先验值的正确初始化
    bgmm = BayesianGaussianMixture(random_state=rng).fit(X)
    assert_almost_equal(1.0, bgmm.mean_precision_prior_)
    
    # 检查给定均值先验值的正确初始化
    mean_prior = rng.rand(n_features)
    bgmm = BayesianGaussianMixture(
        n_components=n_components, mean_prior=mean_prior, random_state=rng
    ).fit(X)
    assert_almost_equal(mean_prior, bgmm.mean_prior_)
    
    # 检查默认均值先验值的正确初始化
    bgmm = BayesianGaussianMixture(n_components=n_components, random_state=rng).fit(X)
    assert_almost_equal(X.mean(axis=0), bgmm.mean_prior_)
# 定义测试函数，用于验证贝叶斯混合模型中的先验精度初始化
def test_bayesian_mixture_precisions_prior_initialisation():
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 设定样本数和特征数
    n_samples, n_features = 10, 2
    # 生成随机数据集
    X = rng.rand(n_samples, n_features)

    # 检查对于不良的自由度先验值是否引发异常
    bad_degrees_of_freedom_prior_ = n_features - 1.0
    # 创建贝叶斯高斯混合模型实例，传入不良的自由度先验值和随机状态
    bgmm = BayesianGaussianMixture(
        degrees_of_freedom_prior=bad_degrees_of_freedom_prior_, random_state=rng
    )
    # 构造异常消息内容
    msg = (
        "The parameter 'degrees_of_freedom_prior' should be greater than"
        f" {n_features - 1}, but got {bad_degrees_of_freedom_prior_:.3f}."
    )
    # 使用 pytest 检查是否引发特定异常和消息
    with pytest.raises(ValueError, match=msg):
        bgmm.fit(X)

    # 检查对于给定的自由度先验值是否正确初始化
    degrees_of_freedom_prior = rng.rand() + n_features - 1.0
    # 创建贝叶斯高斯混合模型实例，传入给定的自由度先验值和随机状态，然后拟合数据
    bgmm = BayesianGaussianMixture(
        degrees_of_freedom_prior=degrees_of_freedom_prior, random_state=rng
    ).fit(X)
    # 断言初始化后的自由度先验值是否与预期值几乎相等
    assert_almost_equal(degrees_of_freedom_prior, bgmm.degrees_of_freedom_prior_)

    # 检查对于默认自由度先验值是否正确初始化
    degrees_of_freedom_prior_default = n_features
    # 创建贝叶斯高斯混合模型实例，传入默认的自由度先验值和随机状态，然后拟合数据
    bgmm = BayesianGaussianMixture(
        degrees_of_freedom_prior=degrees_of_freedom_prior_default, random_state=rng
    ).fit(X)
    # 断言初始化后的自由度先验值是否与预期值几乎相等
    assert_almost_equal(
        degrees_of_freedom_prior_default, bgmm.degrees_of_freedom_prior_
    )

    # 检查对于给定协方差先验值是否正确初始化
    covariance_prior = {
        "full": np.cov(X.T, bias=1) + 10,
        "tied": np.cov(X.T, bias=1) + 5,
        "diag": np.diag(np.atleast_2d(np.cov(X.T, bias=1))) + 3,
        "spherical": rng.rand(),
    }

    bgmm = BayesianGaussianMixture(random_state=rng)
    # 遍历不同的协方差类型
    for cov_type in ["full", "tied", "diag", "spherical"]:
        # 设置当前的协方差类型和对应的先验值，然后拟合数据
        bgmm.covariance_type = cov_type
        bgmm.covariance_prior = covariance_prior[cov_type]
        bgmm.fit(X)
        # 断言初始化后的协方差先验值是否与预期值几乎相等
        assert_almost_equal(covariance_prior[cov_type], bgmm.covariance_prior_)

    # 检查对于默认协方差先验值是否正确初始化
    covariance_prior_default = {
        "full": np.atleast_2d(np.cov(X.T)),
        "tied": np.atleast_2d(np.cov(X.T)),
        "diag": np.var(X, axis=0, ddof=1),
        "spherical": np.var(X, axis=0, ddof=1).mean(),
    }

    bgmm = BayesianGaussianMixture(random_state=0)
    # 遍历不同的协方差类型
    for cov_type in ["full", "tied", "diag", "spherical"]:
        # 设置当前的协方差类型，然后拟合数据
        bgmm.covariance_type = cov_type
        bgmm.fit(X)
        # 断言初始化后的协方差先验值是否与预期值几乎相等
        assert_almost_equal(covariance_prior_default[cov_type], bgmm.covariance_prior_)


# 定义测试函数，用于验证检查贝叶斯混合模型是否已拟合数据
def test_bayesian_mixture_check_is_fitted():
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 设定样本数和特征数
    n_samples, n_features = 10, 2

    # 检查是否引发未拟合异常消息
    bgmm = BayesianGaussianMixture(random_state=rng)
    X = rng.rand(n_samples, n_features)

    msg = "This BayesianGaussianMixture instance is not fitted yet."
    # 使用 pytest 检查是否引发特定异常和消息
    with pytest.raises(ValueError, match=msg):
        bgmm.score(X)


# 定义测试函数，用于验证贝叶斯混合模型中权重初始化
def test_bayesian_mixture_weights():
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 设定样本数和特征数
    n_samples, n_features = 10, 2
    # 使用随机数生成器 rng 生成形状为 (n_samples, n_features) 的随机数据矩阵 X
    X = rng.rand(n_samples, n_features)

    # 使用贝叶斯高斯混合模型拟合数据 X，使用狄利克雷分布作为权重集中先验类型
    bgmm = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_distribution",
        n_components=3,
        random_state=rng,
    ).fit(X)

    # 计算期望的权重，确保权重和为 1
    expected_weights = bgmm.weight_concentration_ / np.sum(bgmm.weight_concentration_)
    assert_almost_equal(expected_weights, bgmm.weights_)
    assert_almost_equal(np.sum(bgmm.weights_), 1.0)

    # 使用贝叶斯高斯混合模型拟合数据 X，使用狄利克雷过程作为权重集中先验类型
    dpgmm = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        n_components=3,
        random_state=rng,
    ).fit(X)
    
    # 计算权重集中参数的和，并进一步计算期望的权重
    weight_dirichlet_sum = (
        dpgmm.weight_concentration_[0] + dpgmm.weight_concentration_[1]
    )
    tmp = dpgmm.weight_concentration_[1] / weight_dirichlet_sum
    expected_weights = (
        dpgmm.weight_concentration_[0]
        / weight_dirichlet_sum
        * np.hstack((1, np.cumprod(tmp[:-1])))
    )
    expected_weights /= np.sum(expected_weights)
    
    # 确保期望的权重和为 1
    assert_almost_equal(expected_weights, dpgmm.weights_)
    assert_almost_equal(np.sum(dpgmm.weights_), 1.0)
# 忽略收敛警告，用于装饰测试函数，确保测试不受收敛警告干扰
@ignore_warnings(category=ConvergenceWarning)
# 测试单调似然性的函数
def test_monotonic_likelihood():
    # 使用随机数种子0创建随机数生成器
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建RandomData对象，设置数据规模为20
    rand_data = RandomData(rng, scale=20)
    # 获取随机数据的成分数
    n_components = rand_data.n_components

    # 对每种先验类型和每种协方差类型进行迭代
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            # 从随机数据中选择特定协方差类型的数据集X
            X = rand_data.X[covar_type]
            # 创建BayesianGaussianMixture对象，设置权重先验类型、成分数、协方差类型等参数
            bgmm = BayesianGaussianMixture(
                weight_concentration_prior_type=prior_type,
                n_components=2 * n_components,
                covariance_type=covar_type,
                warm_start=True,
                max_iter=1,
                random_state=rng,
                tol=1e-3,
            )
            # 初始化当前的较低边界为负无穷
            current_lower_bound = -np.inf

            # 逐步训练，每次只进行一次迭代，以确保每次迭代后训练对数似然性增加
            for _ in range(600):
                # 将前一个较低边界保存为prev_lower_bound
                prev_lower_bound = current_lower_bound
                # 执行一次拟合，并获取当前的较低边界
                current_lower_bound = bgmm.fit(X).lower_bound_
                # 断言当前较低边界大于等于前一个较低边界
                assert current_lower_bound >= prev_lower_bound

                # 如果已经收敛，则终止训练
                if bgmm.converged_:
                    break
            # 断言已经收敛
            assert bgmm.converged_


# 比较不同协方差类型的测试函数
def test_compare_covar_type():
    # 使用随机数种子0创建随机数生成器
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建RandomData对象，设置数据规模为7
    rand_data = RandomData(rng, scale=7)
    # 获取完整协方差类型的数据集X
    X = rand_data.X["full"]
    # 获取随机数据的成分数
    n_components = rand_data.n_components
    for prior_type in PRIOR_TYPE:
        # 对于每种先验类型，使用贝叶斯高斯混合模型进行计算

        # 创建一个BayesianGaussianMixture对象，设置参数
        bgmm = BayesianGaussianMixture(
            weight_concentration_prior_type=prior_type,
            n_components=2 * n_components,
            covariance_type="full",
            max_iter=1,
            random_state=0,
            tol=1e-7,
        )
        # 检查并调整模型参数
        bgmm._check_parameters(X)
        # 使用随机状态初始化模型参数
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        # 计算完整协方差矩阵
        full_covariances = (
            bgmm.covariances_ * bgmm.degrees_of_freedom_[:, np.newaxis, np.newaxis]
        )

        # 检查并验证tied_covariance是否等于full_covariances的均值
        bgmm = BayesianGaussianMixture(
            weight_concentration_prior_type=prior_type,
            n_components=2 * n_components,
            covariance_type="tied",
            max_iter=1,
            random_state=0,
            tol=1e-7,
        )
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        tied_covariance = bgmm.covariances_ * bgmm.degrees_of_freedom_
        assert_almost_equal(tied_covariance, np.mean(full_covariances, 0))

        # 检查并验证diag_covariance是否等于full_covariances的对角线元素
        bgmm = BayesianGaussianMixture(
            weight_concentration_prior_type=prior_type,
            n_components=2 * n_components,
            covariance_type="diag",
            max_iter=1,
            random_state=0,
            tol=1e-7,
        )
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        diag_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_[:, np.newaxis]
        assert_almost_equal(
            diag_covariances, np.array([np.diag(cov) for cov in full_covariances])
        )

        # 检查并验证spherical_covariance是否等于diag_covariances的均值
        bgmm = BayesianGaussianMixture(
            weight_concentration_prior_type=prior_type,
            n_components=2 * n_components,
            covariance_type="spherical",
            max_iter=1,
            random_state=0,
            tol=1e-7,
        )
        bgmm._check_parameters(X)
        bgmm._initialize_parameters(X, np.random.RandomState(0))
        spherical_covariances = bgmm.covariances_ * bgmm.degrees_of_freedom_
        assert_almost_equal(spherical_covariances, np.mean(diag_covariances, 1))
# 忽略收敛警告，装饰器函数，用于测试检查协方差精度
@ignore_warnings(category=ConvergenceWarning)
# 检查协方差和精度矩阵的点积是否为单位矩阵
def test_check_covariance_precision():
    # 使用随机种子创建随机数生成器
    rng = np.random.RandomState(0)
    # 使用随机数据对象创建随机数据，指定数据缩放因子为7
    rand_data = RandomData(rng, scale=7)
    # 设置混合模型的组件数和特征数
    n_components, n_features = 2 * rand_data.n_components, 2

    # 计算完整协方差
    bgmm = BayesianGaussianMixture(
        n_components=n_components, max_iter=100, random_state=rng, tol=1e-3, reg_covar=0
    )
    # 遍历协方差类型列表
    for covar_type in COVARIANCE_TYPE:
        # 设置当前循环的协方差类型
        bgmm.covariance_type = covar_type
        # 使用指定类型的数据拟合模型
        bgmm.fit(rand_data.X[covar_type])

        if covar_type == "full":
            # 对于完整协方差类型，逐个检查协方差和精度矩阵的点积是否为单位矩阵
            for covar, precision in zip(bgmm.covariances_, bgmm.precisions_):
                assert_almost_equal(np.dot(covar, precision), np.eye(n_features))
        elif covar_type == "tied":
            # 对于共享协方差类型，检查协方差矩阵和精度矩阵的点积是否为单位矩阵
            assert_almost_equal(
                np.dot(bgmm.covariances_, bgmm.precisions_), np.eye(n_features)
            )
        elif covar_type == "diag":
            # 对于对角协方差类型，检查每个组件的协方差和精度的逐元素点积是否为单位矩阵
            assert_almost_equal(
                bgmm.covariances_ * bgmm.precisions_,
                np.ones((n_components, n_features)),
            )
        else:
            # 对于球形协方差类型，检查每个组件的协方差和精度的逐元素点积是否为1维单位数组
            assert_almost_equal(
                bgmm.covariances_ * bgmm.precisions_, np.ones(n_components)
            )


# 忽略收敛警告，装饰器函数，用于测试不变性翻译
@ignore_warnings(category=ConvergenceWarning)
def test_invariant_translation():
    # 检查在数据中添加常数后，混合模型参数是否正确变化
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=100)
    n_components = 2 * rand_data.n_components

    # 遍历先验类型和协方差类型列表
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            X = rand_data.X[covar_type]
            # 使用原始数据拟合的混合模型
            bgmm1 = BayesianGaussianMixture(
                weight_concentration_prior_type=prior_type,
                n_components=n_components,
                max_iter=100,
                random_state=0,
                tol=1e-3,
                reg_covar=0,
            ).fit(X)
            # 使用添加常数后的数据拟合的混合模型
            bgmm2 = BayesianGaussianMixture(
                weight_concentration_prior_type=prior_type,
                n_components=n_components,
                max_iter=100,
                random_state=0,
                tol=1e-3,
                reg_covar=0,
            ).fit(X + 100)

            # 检查均值是否正确变化
            assert_almost_equal(bgmm1.means_, bgmm2.means_ - 100)
            # 检查权重是否不变
            assert_almost_equal(bgmm1.weights_, bgmm2.weights_)
            # 检查协方差是否不变
            assert_almost_equal(bgmm1.covariances_, bgmm2.covariances_)


# 使用 pytest 装饰器标记，忽略警告中包含 "did not converge" 的测试
@pytest.mark.filterwarnings("ignore:.*did not converge.*")
# 参数化测试函数，测试不同种子、最大迭代次数和收敛容差的混合模型拟合和预测
@pytest.mark.parametrize(
    "seed, max_iter, tol",
    [
        (0, 2, 1e-7),   # 严格非收敛
        (1, 2, 1e-1),   # 松散非收敛
        (3, 300, 1e-7), # 严格收敛
        (4, 300, 1e-1), # 松散收敛
    ],
)
def test_bayesian_mixture_fit_predict(seed, max_iter, tol):
    # 使用指定种子创建随机数生成器
    rng = np.random.RandomState(seed)
    # 创建随机数据对象，样本数为50，数据缩放因子为7
    rand_data = RandomData(rng, n_samples=50, scale=7)
    # 计算用于高斯混合模型的组件数量，这里是随机数据的两倍
    n_components = 2 * rand_data.n_components

    # 遍历不同的协方差类型
    for covar_type in COVARIANCE_TYPE:
        # 创建第一个贝叶斯高斯混合模型对象
        bgmm1 = BayesianGaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            random_state=rng,
            tol=tol,
            reg_covar=0,
        )
        # 设置当前模型的协方差类型
        bgmm1.covariance_type = covar_type

        # 深拷贝第一个模型，创建第二个贝叶斯高斯混合模型对象
        bgmm2 = copy.deepcopy(bgmm1)

        # 从随机数据中选择特定协方差类型的数据
        X = rand_data.X[covar_type]

        # 使用第一个模型拟合数据并预测类别
        Y_pred1 = bgmm1.fit(X).predict(X)
        
        # 使用第二个模型同时进行拟合和预测
        Y_pred2 = bgmm2.fit_predict(X)
        
        # 断言两个预测结果数组是否相等
        assert_array_equal(Y_pred1, Y_pred2)
# 测试贝叶斯混合模型的 fit_predict 方法在 n_init > 1 时与 fit.predict 方法等价
def test_bayesian_mixture_fit_predict_n_init():
    # 创建一个随机数生成器，并生成一个形状为 (50, 5) 的随机数组 X
    X = np.random.RandomState(0).randn(50, 5)
    # 初始化一个贝叶斯高斯混合模型，设置成分数为 5，初始化次数为 10，随机状态为 0
    gm = BayesianGaussianMixture(n_components=5, n_init=10, random_state=0)
    # 使用 fit_predict 方法拟合模型并预测 X 的类别
    y_pred1 = gm.fit_predict(X)
    # 直接使用 predict 方法预测 X 的类别
    y_pred2 = gm.predict(X)
    # 断言两种预测结果相等
    assert_array_equal(y_pred1, y_pred2)


# 测试贝叶斯混合模型的 predict 和 predict_proba 方法
def test_bayesian_mixture_predict_predict_proba():
    # 使用随机数生成器创建 RandomData 对象
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    # 遍历不同的先验类型和协方差类型
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            # 从随机数据中获取 X 和 Y 数据
            X = rand_data.X[covar_type]
            Y = rand_data.Y
            # 初始化贝叶斯高斯混合模型，设置成分数、随机状态、先验类型和协方差类型
            bgmm = BayesianGaussianMixture(
                n_components=rand_data.n_components,
                random_state=rng,
                weight_concentration_prior_type=prior_type,
                covariance_type=covar_type,
            )

            # 测试如果没有进行 fit 操作，调用 predict 方法是否会抛出 NotFittedError 异常
            msg = (
                "This BayesianGaussianMixture instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this "
                "estimator."
            )
            with pytest.raises(NotFittedError, match=msg):
                bgmm.predict(X)

            # 对 X 进行 fit 操作
            bgmm.fit(X)
            # 使用 fit 后的模型预测 X 的类别
            Y_pred = bgmm.predict(X)
            # 使用 predict_proba 方法获取概率，并取最大概率的类别作为预测结果
            Y_pred_proba = bgmm.predict_proba(X).argmax(axis=1)
            # 断言预测的类别与概率最大类别相等
            assert_array_equal(Y_pred, Y_pred_proba)
            # 断言调整后的兰德指数大于等于 0.95
            assert adjusted_rand_score(Y, Y_pred) >= 0.95
```