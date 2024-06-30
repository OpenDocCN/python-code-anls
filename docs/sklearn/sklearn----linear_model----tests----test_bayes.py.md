# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_bayes.py`

```
# 从 math 模块导入 log 函数
from math import log

# 导入必要的库
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 PyTest 库，用于单元测试

# 导入 scikit-learn 相关模块和函数
from sklearn import datasets  # 导入数据集模块
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge  # 导入线性回归模型
from sklearn.utils import check_random_state  # 导入随机状态检查函数
from sklearn.utils._testing import (  # 导入测试相关函数
    _convert_container,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_less,
)
from sklearn.utils.extmath import fast_logdet  # 导入快速计算对数行列式的函数

# 加载糖尿病数据集
diabetes = datasets.load_diabetes()


def test_bayesian_ridge_scores():
    """Check scores attribute shape"""
    # 获取数据集特征和目标值
    X, y = diabetes.data, diabetes.target

    # 初始化 BayesianRidge 模型并拟合数据
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)

    # 断言模型 scores_ 属性的形状是否符合预期
    assert clf.scores_.shape == (clf.n_iter_ + 1,)


def test_bayesian_ridge_score_values():
    """Check value of score on toy example.

    Compute log marginal likelihood with equation (36) in Sparse Bayesian
    Learning and the Relevance Vector Machine (Tipping, 2001):

    - 0.5 * (log |Id/alpha + X.X^T/lambda| +
             y^T.(Id/alpha + X.X^T/lambda).y + n * log(2 * pi))
    + lambda_1 * log(lambda) - lambda_2 * lambda
    + alpha_1 * log(alpha) - alpha_2 * alpha

    and check equality with the score computed during training.
    """
    # 获取数据集特征和目标值
    X, y = diabetes.data, diabetes.target
    n_samples = X.shape[0]

    # 计算 alpha 和 lambda 的初始值，使用了 y 的方差
    eps = np.finfo(np.float64).eps
    alpha_ = 1.0 / (np.var(y) + eps)
    lambda_ = 1.0

    # Gamma 超先验的参数值
    alpha_1 = 0.1
    alpha_2 = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1

    # 使用文档字符串中的公式计算得分
    score = lambda_1 * log(lambda_) - lambda_2 * lambda_
    score += alpha_1 * log(alpha_) - alpha_2 * alpha_
    M = 1.0 / alpha_ * np.eye(n_samples) + 1.0 / lambda_ * np.dot(X, X.T)
    M_inv_dot_y = np.linalg.solve(M, y)
    score += -0.5 * (
        fast_logdet(M) + np.dot(y.T, M_inv_dot_y) + n_samples * log(2 * np.pi)
    )

    # 使用 BayesianRidge 计算得分
    clf = BayesianRidge(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        max_iter=1,
        fit_intercept=False,
        compute_score=True,
    )
    clf.fit(X, y)

    # 断言 BayesianRidge 计算的得分与手动计算的得分的近似相等性
    assert_almost_equal(clf.scores_[0], score, decimal=9)


def test_bayesian_ridge_parameter():
    # Test correctness of lambda_ and alpha_ parameters (GitHub issue #8224)
    # 创建一个示例数据集 X 和目标值 y
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T

    # 使用 BayesianRidge 模型拟合数据，并计算 lambda_ 和 alpha_ 参数
    br_model = BayesianRidge(compute_score=True).fit(X, y)
    
    # 创建一个 Ridge 模型，其 alpha 值等于 BayesianRidge 模型中 lambda_ 和 alpha_ 的比率
    rr_model = Ridge(alpha=br_model.lambda_ / br_model.alpha_).fit(X, y)
    
    # 断言 Ridge 模型和 BayesianRidge 模型的系数近似相等
    assert_array_almost_equal(rr_model.coef_, br_model.coef_)
    assert_almost_equal(rr_model.intercept_, br_model.intercept_)
    # 定义输入特征 X，每行代表一个样本，每列代表一个特征
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    # 定义目标标签 y，对应每个样本的输出值
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T
    # 定义样本权重 w，用于加权样本在模型训练中的重要性
    w = np.array([4, 3, 3, 1, 1, 2, 3]).T

    # 使用贝叶斯岭回归模型创建 br_model，设置 compute_score=True 来计算模型分数，并使用样本权重 w 进行训练
    br_model = BayesianRidge(compute_score=True).fit(X, y, sample_weight=w)
    # 根据贝叶斯岭回归模型的 lambda_ 和 alpha_ 比率计算得到 alpha 值，用于创建岭回归模型 rr_model
    rr_model = Ridge(alpha=br_model.lambda_ / br_model.alpha_).fit(
        X, y, sample_weight=w
    )
    # 断言岭回归模型的系数与贝叶斯岭回归模型的系数几乎相等
    assert_array_almost_equal(rr_model.coef_, br_model.coef_)
    # 断言岭回归模型的截距与贝叶斯岭回归模型的截距几乎相等
    assert_almost_equal(rr_model.intercept_, br_model.intercept_)
# 测试 BayesianRidge 在简单示例上的表现

# 创建输入特征 X 和目标值 Y
X = np.array([[1], [2], [6], [8], [10]])
Y = np.array([1, 2, 6, 8, 10])

# 初始化 BayesianRidge 回归模型
clf = BayesianRidge(compute_score=True)

# 使用输入数据 X, Y 训练模型
clf.fit(X, Y)

# 检查模型是否能近似学习恒等函数的能力

# 准备测试数据
test = [[1], [3], [4]]

# 使用训练好的模型进行预测
# 检查预测结果与期望值的近似程度
assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)


# 测试带初始参数 (alpha_init, lambda_init) 的 BayesianRidge

# 创建输入特征 X 和目标值 y
X = np.vander(np.linspace(0, 4, 5), 4)
y = np.array([0.0, 1.0, 0.0, -1.0, 0.0])  # y = (x^3 - 6x^2 + 8x) / 3

# 在这种情况下，从默认初始值开始会增加拟合曲线的偏差
# 因此，lambda_init 应该设置得很小
reg = BayesianRidge(alpha_init=1.0, lambda_init=1e-3)

# 训练模型并计算 R2 分数，期望接近于 1
r2 = reg.fit(X, y).score(X, y)
assert_almost_equal(r2, 1.0)


# 测试在常数目标向量下的 BayesianRidge 和 ARDRegression 预测边界情况

# 设置样本数和特征数
n_samples = 4
n_features = 5
random_state = check_random_state(42)
constant_value = random_state.rand()

# 创建随机的输入特征 X 和常数目标值 y
X = random_state.random_sample((n_samples, n_features))
y = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
expected = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)

# 针对 BayesianRidge 和 ARDRegression 进行预测
for clf in [BayesianRidge(), ARDRegression()]:
    y_pred = clf.fit(X, y).predict(X)
    # 检查预测结果是否与期望值几乎相等
    assert_array_almost_equal(y_pred, expected)


# 测试在常数目标向量下的 BayesianRidge 和 ARDRegression 标准差边界情况

# 设置样本数和特征数
n_samples = 10
n_features = 5
random_state = check_random_state(42)
constant_value = random_state.rand()

# 创建随机的输入特征 X 和常数目标值 y
X = random_state.random_sample((n_samples, n_features))
y = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
expected_upper_boundary = 0.01

# 针对 BayesianRidge 和 ARDRegression 进行预测，并返回标准差
for clf in [BayesianRidge(), ARDRegression()]:
    _, y_std = clf.fit(X, y).predict(X, return_std=True)
    # 检查标准差是否小于预期的上界
    assert_array_less(y_std, expected_upper_boundary)


# 测试 ARDRegression 算法中 sigma_ 的更新是否正确

# 创建输入特征 X 和目标值 y
X = np.array([[1, 0], [0, 0]])
y = np.array([0, 0])

# 初始化 ARDRegression 模型，设置最大迭代次数为 1
clf = ARDRegression(max_iter=1)

# 使用输入数据 X, y 训练模型
clf.fit(X, y)

# 针对上述输入，ARDRegression 在第一次迭代中会删除两个系数
# 因此，sigma_ 的预期形状应为 (0, 0)
assert clf.sigma_.shape == (0, 0)

# 确保在预测阶段不会抛出错误
clf.predict(X, return_std=True)


# 测试 BayesianRegression ARD 分类器

# 创建输入特征 X 和目标值 Y
X = np.array([[1], [2], [3]])
Y = np.array([1, 2, 3])
    # 创建一个 ARDRegression 对象，参数 compute_score=True 表示在拟合过程中计算得分
    clf = ARDRegression(compute_score=True)
    
    # 使用给定的训练数据 X 和目标值 Y 来训练回归模型 clf
    clf.fit(X, Y)
    
    # 检查模型是否能够近似学习到身份函数（identity function）
    test = [[1], [3], [4]]
    # 使用 assert_array_almost_equal 函数检验模型预测的结果是否接近给定的目标值 [1, 3, 4]，允许的误差范围为 2
    assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)
@pytest.mark.parametrize("n_samples, n_features", ((10, 100), (100, 10)))
def test_ard_accuracy_on_easy_problem(global_random_seed, n_samples, n_features):
    # 检查在简单问题上 ARD 回归是否能够以合理的精度收敛
    # (Github issue #14055)

    # 创建随机数据集 X 和相应的目标值 y
    X = np.random.RandomState(global_random_seed).normal(size=(250, 3))
    y = X[:, 1]

    # 初始化 ARD 回归器
    regressor = ARDRegression()
    # 使用数据集 X 和目标值 y 进行拟合
    regressor.fit(X, y)

    # 计算第二个系数与真实值的绝对误差
    abs_coef_error = np.abs(1 - regressor.coef_[1])
    # 断言绝对误差小于 1e-10
    assert abs_coef_error < 1e-10


@pytest.mark.parametrize("constructor_name", ["array", "dataframe"])
def test_return_std(constructor_name):
    # 测试 Bayesian 回归器的 return_std 选项

    # 定义线性函数 f(X)
    def f(X):
        return np.dot(X, w) + b

    # 定义带噪声的线性函数 f(X, noise_mult)
    def f_noise(X, noise_mult):
        return f(X) + np.random.randn(X.shape[0]) * noise_mult

    d = 5
    n_train = 50
    n_test = 10

    w = np.array([1.0, 0.0, 1.0, -1.0, 0.0])
    b = 1.0

    # 创建训练集 X
    X = np.random.random((n_train, d))
    X = _convert_container(X, constructor_name)

    # 创建测试集 X_test
    X_test = np.random.random((n_test, d))
    X_test = _convert_container(X_test, constructor_name)

    # 对于不同的 decimal 和 noise_mult 组合进行迭代
    for decimal, noise_mult in enumerate([1, 0.1, 0.01]):
        # 生成带噪声的训练集 y
        y = f_noise(X, noise_mult)

        # 初始化 BayesianRidge 模型并拟合
        m1 = BayesianRidge()
        m1.fit(X, y)
        # 预测测试集上的均值和标准差
        y_mean1, y_std1 = m1.predict(X_test, return_std=True)
        # 断言预测的标准差与预期噪声水平接近
        assert_array_almost_equal(y_std1, noise_mult, decimal=decimal)

        # 初始化 ARDRegression 模型并拟合
        m2 = ARDRegression()
        m2.fit(X, y)
        # 预测测试集上的均值和标准差
        y_mean2, y_std2 = m2.predict(X_test, return_std=True)
        # 断言预测的标准差与预期噪声水平接近
        assert_array_almost_equal(y_std2, noise_mult, decimal=decimal)


def test_update_sigma(global_random_seed):
    # 确保两个 update_sigma() 辅助函数的等效性。当 n_samples < n_features 时使用 Woodbury 公式，
    # 否则使用另一个公式。

    rng = np.random.RandomState(global_random_seed)

    # 设置 n_samples == n_features 以避免在求逆矩阵时的不稳定性问题
    n_samples = n_features = 10
    X = rng.randn(n_samples, n_features)
    alpha = 1
    lmbda = np.arange(1, n_features + 1)
    keep_lambda = np.array([True] * n_features)

    # 初始化 ARDRegression 模型
    reg = ARDRegression()

    # 使用 _update_sigma() 计算协方差矩阵 sigma
    sigma = reg._update_sigma(X, alpha, lmbda, keep_lambda)
    # 使用 _update_sigma_woodbury() 计算协方差矩阵 sigma_woodbury
    sigma_woodbury = reg._update_sigma_woodbury(X, alpha, lmbda, keep_lambda)

    # 断言两种方法计算的结果近似相等
    np.testing.assert_allclose(sigma, sigma_woodbury)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("Estimator", [BayesianRidge, ARDRegression])
def test_dtype_match(dtype, Estimator):
    # 测试当输入数据为 np.float32 时，不会在可能时将其转换为 np.float64

    # 创建输入数据 X 和目标值 y
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]], dtype=dtype)
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T

    # 初始化指定的回归模型
    model = Estimator()
    # 拟合模型
    model.fit(X, y)

    # 检查模型属性的数据类型是否一致
    attributes = ["coef_", "sigma_"]
    for attribute in attributes:
        assert getattr(model, attribute).dtype == X.dtype
    # 使用模型对输入数据 X 进行预测，同时返回预测的均值 y_mean 和标准差 y_std
    y_mean, y_std = model.predict(X, return_std=True)
    # 断言预测的均值 y_mean 的数据类型与输入数据 X 的数据类型相同
    assert y_mean.dtype == X.dtype
    # 断言预测的标准差 y_std 的数据类型与输入数据 X 的数据类型相同
    assert y_std.dtype == X.dtype
# 使用 pytest 的参数化装饰器，指定测试用例参数为 BayesianRidge 和 ARDRegression
@pytest.mark.parametrize("Estimator", [BayesianRidge, ARDRegression])
# 定义测试函数，测试模型的数据类型正确性
def test_dtype_correctness(Estimator):
    # 创建一个示例输入特征矩阵 X，包含七个样本和两个特征
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    # 创建一个示例目标向量 y，包含七个样本的目标值
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T
    # 根据当前的 Estimator 类型创建模型对象
    model = Estimator()
    # 使用 np.float32 类型拟合模型，并获取系数
    coef_32 = model.fit(X.astype(np.float32), y).coef_
    # 使用 np.float64 类型拟合模型，并获取系数
    coef_64 = model.fit(X.astype(np.float64), y).coef_
    # 使用 np.testing.assert_allclose 检查两种数据类型下的系数是否接近
    np.testing.assert_allclose(coef_32, coef_64, rtol=1e-4)
```