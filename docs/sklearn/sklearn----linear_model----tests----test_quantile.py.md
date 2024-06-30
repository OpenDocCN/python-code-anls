# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_quantile.py`

```
# 导入必要的库和模块
import numpy as np  # 导入numpy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试
from pytest import approx  # 导入approx函数，用于测试近似相等
from scipy.optimize import minimize  # 导入minimize函数，用于最小化函数

# 导入scikit-learn相关模块和类
from sklearn.datasets import make_regression  # 导入make_regression函数，用于生成回归数据集
from sklearn.exceptions import ConvergenceWarning  # 导入ConvergenceWarning异常类
from sklearn.linear_model import HuberRegressor, QuantileRegressor  # 导入HuberRegressor和QuantileRegressor类
from sklearn.metrics import mean_pinball_loss  # 导入mean_pinball_loss函数
from sklearn.utils._testing import assert_allclose, skip_if_32bit  # 导入assert_allclose和skip_if_32bit函数
from sklearn.utils.fixes import (  # 导入修复模块中的各种修复
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    parse_version,
    sp_version,
)

# 定义X_y_data的pytest fixture，生成回归数据集
@pytest.fixture
def X_y_data():
    X, y = make_regression(n_samples=10, n_features=1, random_state=0, noise=1)
    return X, y

# 定义默认求解器的pytest fixture，根据SciPy版本选择不同的求解器
@pytest.fixture
def default_solver():
    return "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

# 跳过测试条件：如果SciPy版本大于等于1.11，则跳过测试
@pytest.mark.skipif(
    parse_version(sp_version.base_version) >= parse_version("1.11"),
    reason="interior-point solver is not available in SciPy 1.11",
)
# 参数化测试：测试不兼容的求解器对稀疏输入的影响
@pytest.mark.parametrize("solver", ["interior-point", "revised simplex"])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_incompatible_solver_for_sparse_input(X_y_data, solver, csc_container):
    X, y = X_y_data
    X_sparse = csc_container(X)
    err_msg = (
        f"Solver {solver} does not support sparse X. Use solver 'highs' for example."
    )
    with pytest.raises(ValueError, match=err_msg):
        QuantileRegressor(solver=solver).fit(X_sparse, y)

# 跳过测试条件：如果SciPy版本大于等于1.6.0，则跳过测试
@pytest.mark.parametrize("solver", ("highs-ds", "highs-ipm", "highs"))
@pytest.mark.skipif(
    sp_version >= parse_version("1.6.0"),
    reason="Solvers are available as of scipy 1.6.0",
)
def test_too_new_solver_methods_raise_error(X_y_data, solver):
    """Test that highs solver raises for scipy<1.6.0."""
    X, y = X_y_data
    with pytest.raises(ValueError, match="scipy>=1.6.0"):
        QuantileRegressor(solver=solver).fit(X, y)

# 参数化测试：测试不同分位数和正则化参数对一个简单示例的影响
@pytest.mark.parametrize(
    "quantile, alpha, intercept, coef",
    [
        [0.5, 0, 1, None],       # 50%分位数，无正则化
        [0.51, 0, 1, 10],        # 51%分位数，无正则化
        [0.49, 0, 1, 1],         # 49%分位数，无正则化
        [0.5, 0.01, 1, 1],       # 50%分位数，小的lasso惩罚
        [0.5, 100, 2, 0],        # 50%分位数，大的lasso惩罚
    ],
)
def test_quantile_toy_example(quantile, alpha, intercept, coef, default_solver):
    # 测试不同参数如何影响简单直观示例
    X = [[0], [1], [1]]
    y = [1, 2, 11]
    model = QuantileRegressor(
        quantile=quantile, alpha=alpha, solver=default_solver
    ).fit(X, y)
    assert_allclose(model.intercept_, intercept, atol=1e-2)
    if coef is not None:
        assert_allclose(model.coef_[0], coef, atol=1e-2)
    if alpha < 100:
        assert model.coef_[0] >= 1
    assert model.coef_[0] <= 10
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_quantile_equals_huber_for_low_epsilon(fit_intercept, default_solver):
    # 生成用于测试的随机数据集 X 和 y
    X, y = make_regression(n_samples=100, n_features=20, random_state=0, noise=1.0)
    alpha = 1e-4
    # 使用 HuberRegressor 拟合数据，设置 epsilon 和 alpha 参数
    huber = HuberRegressor(
        epsilon=1 + 1e-4, alpha=alpha, fit_intercept=fit_intercept
    ).fit(X, y)
    # 使用 QuantileRegressor 拟合数据，设置 alpha、fit_intercept 和 solver 参数
    quant = QuantileRegressor(
        alpha=alpha, fit_intercept=fit_intercept, solver=default_solver
    ).fit(X, y)
    # 断言 HuberRegressor 和 QuantileRegressor 的系数相近
    assert_allclose(huber.coef_, quant.coef_, atol=1e-1)
    if fit_intercept:
        # 如果 fit_intercept 为 True，则断言截距也相近
        assert huber.intercept_ == approx(quant.intercept_, abs=1e-1)
        # 检查预测结果中小于真实值的比例是否接近 0.5
        assert np.mean(y < quant.predict(X)) == approx(0.5, abs=1e-1)


@pytest.mark.parametrize("q", [0.5, 0.9, 0.05])
def test_quantile_estimates_calibration(q, default_solver):
    # 测试模型是否准确预测低于预测值的点的百分比
    X, y = make_regression(n_samples=1000, n_features=20, random_state=0, noise=1.0)
    # 使用 QuantileRegressor 拟合数据，设置 quantile、alpha 和 solver 参数
    quant = QuantileRegressor(
        quantile=q,
        alpha=0,
        solver=default_solver,
    ).fit(X, y)
    # 断言真实值小于预测值的比例是否接近预期的 quantile 值
    assert np.mean(y < quant.predict(X)) == approx(q, abs=1e-2)


def test_quantile_sample_weight(default_solver):
    # 测试当样本权重不均等时，模型是否仍能正确估计加权比例
    n = 1000
    X, y = make_regression(n_samples=n, n_features=5, random_state=0, noise=10.0)
    weight = np.ones(n)
    # 当我们增加上部观测的权重时，预测的分位数应该上升
    weight[y > y.mean()] = 100
    # 使用 QuantileRegressor 拟合数据，设置 quantile、alpha 和 solver 参数
    quant = QuantileRegressor(quantile=0.5, alpha=1e-8, solver=default_solver)
    quant.fit(X, y, sample_weight=weight)
    # 断言真实值小于预测值的比例大于 0.5
    assert np.mean(y < quant.predict(X)) > 0.5
    # 使用权重计算加权比例
    weighted_fraction_below = np.average(y < quant.predict(X), weights=weight)
    # 断言加权比例是否接近 0.5
    assert weighted_fraction_below == approx(0.5, abs=3e-2)


@pytest.mark.skipif(
    sp_version < parse_version("1.6.0"),
    reason="The `highs` solver is available from the 1.6.0 scipy version",
)
@pytest.mark.parametrize("quantile", [0.2, 0.5, 0.8])
def test_asymmetric_error(quantile, default_solver):
    """Test quantile regression for asymmetric distributed targets."""
    n_samples = 1000
    rng = np.random.RandomState(42)
    # 创建一个具有特定分布的数据集 X
    X = np.concatenate(
        (
            np.abs(rng.randn(n_samples)[:, None]),
            -rng.randint(2, size=(n_samples, 1)),
        ),
        axis=1,
    )
    intercept = 1.23
    coef = np.array([0.5, -2])
    # 确保 X @ coef + intercept 的最小值大于 0
    assert np.min(X @ coef + intercept) > 0
    # 对于具有指数分布（如 exp(-lambda * x)）的目标，计算指定分位数的量化回归
    # lambda = -(X @ coef + intercept) / log(1 - quantile)
    y = rng.exponential(
        scale=-(X @ coef + intercept) / np.log(1 - quantile), size=n_samples
    )
    # 使用 QuantileRegressor 模型拟合数据，设置分位数为 quantile，alpha 为 0，使用默认的求解器
    model = QuantileRegressor(
        quantile=quantile,
        alpha=0,
        solver=default_solver,
    ).fit(X, y)
    # 这个测试可以通过任何求解器来通过，但为了节省持续集成资源，只使用最快的求解器进行测试。

    # 断言模型的截距与期望的截距相近，相对误差不超过 20%
    assert model.intercept_ == approx(intercept, rel=0.2)
    # 断言模型的系数与期望的系数数组相近，相对误差不超过 60%
    assert_allclose(model.coef_, coef, rtol=0.6)
    # 断言模型预测结果大于真实值的比例的均值接近于预期的分位数 quantile，绝对误差不超过 0.01

    # 现在与使用 Nelder-Mead 优化和 L1 惩罚的结果进行比较
    alpha = 0.01
    # 设置模型参数 alpha 为 0.01，并重新拟合模型
    model.set_params(alpha=alpha).fit(X, y)
    # 模型系数由截距和系数组成的数组
    model_coef = np.r_[model.intercept_, model.coef_]

    # 定义目标函数 func，用于计算损失函数（均值分位损失）和 L1 正则项
    def func(coef):
        loss = mean_pinball_loss(y, X @ coef[1:] + coef[0], alpha=quantile)
        L1 = np.sum(np.abs(coef[1:]))
        return loss + alpha * L1

    # 使用 Nelder-Mead 方法进行最小化优化，初始点为 [1, 0, -1]，容差为 1e-12，最大迭代次数为 2000
    res = minimize(
        fun=func,
        x0=[1, 0, -1],
        method="Nelder-Mead",
        tol=1e-12,
        options={"maxiter": 2000},
    )

    # 断言模型当前系数下的 func 值与优化后得到的 res.x 对应的 func 值相近
    assert func(model_coef) == approx(func(res.x))
    # 断言模型的截距与优化后得到的 res.x[0] 相近
    assert_allclose(model.intercept_, res.x[0])
    # 断言模型的系数与优化后得到的 res.x[1:] 相近
    assert_allclose(model.coef_, res.x[1:])
    # 断言模型预测结果大于真实值的比例的均值接近于预期的分位数 quantile，绝对误差不超过 0.01
@pytest.mark.parametrize("quantile", [0.2, 0.5, 0.8])
def test_equivariance(quantile, default_solver):
    """Test equivariance of quantile regression.

    See Koenker (2005) Quantile Regression, Chapter 2.2.3.
    """
    # 使用种子值 42 初始化随机数生成器
    rng = np.random.RandomState(42)
    # 设定样本数和特征数
    n_samples, n_features = 100, 5
    # 创建具有特定特征的回归数据集
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        noise=0,
        random_state=rng,
        shuffle=False,
    )
    # 使 y 变得不对称
    y += rng.exponential(scale=100, size=y.shape)
    # 设定回归模型的参数
    params = dict(alpha=0, solver=default_solver)
    # 使用 QuantileRegressor 拟合模型，量化分位数为 quantile
    model1 = QuantileRegressor(quantile=quantile, **params).fit(X, y)

    # coef(q; a*y, X) = a * coef(q; y, X)
    a = 2.5
    # 使用 QuantileRegressor 拟合模型，量化分位数为 quantile，乘以常数 a
    model2 = QuantileRegressor(quantile=quantile, **params).fit(X, a * y)
    # 断言拟合后的截距与系数符合预期
    assert model2.intercept_ == approx(a * model1.intercept_, rel=1e-5)
    assert_allclose(model2.coef_, a * model1.coef_, rtol=1e-5)

    # coef(1-q; -a*y, X) = -a * coef(q; y, X)
    # 使用 QuantileRegressor 拟合模型，量化分位数为 1-quantile，乘以常数 -a
    model2 = QuantileRegressor(quantile=1 - quantile, **params).fit(X, -a * y)
    # 断言拟合后的截距与系数符合预期
    assert model2.intercept_ == approx(-a * model1.intercept_, rel=1e-5)
    assert_allclose(model2.coef_, -a * model1.coef_, rtol=1e-5)

    # coef(q; y + X @ g, X) = coef(q; y, X) + g
    # 生成截距和系数的随机向量 g
    g_intercept, g_coef = rng.randn(), rng.randn(n_features)
    # 使用 QuantileRegressor 拟合模型，量化分位数为 quantile，加上 g
    model2 = QuantileRegressor(quantile=quantile, **params)
    model2.fit(X, y + X @ g_coef + g_intercept)
    # 断言拟合后的截距与系数符合预期
    assert model2.intercept_ == approx(model1.intercept_ + g_intercept)
    assert_allclose(model2.coef_, model1.coef_ + g_coef, rtol=1e-6)

    # coef(q; y, X @ A) = A^-1 @ coef(q; y, X)
    # 生成随机矩阵 A
    A = rng.randn(n_features, n_features)
    # 使用 QuantileRegressor 拟合模型，量化分位数为 quantile，数据输入为 X @ A
    model2 = QuantileRegressor(quantile=quantile, **params)
    model2.fit(X @ A, y)
    # 断言拟合后的截距与系数符合预期
    assert model2.intercept_ == approx(model1.intercept_, rel=1e-5)
    assert_allclose(model2.coef_, np.linalg.solve(A, model1.coef_), rtol=1e-5)


@pytest.mark.skipif(
    parse_version(sp_version.base_version) >= parse_version("1.11"),
    reason="interior-point solver is not available in SciPy 1.11",
)
@pytest.mark.filterwarnings("ignore:`method='interior-point'` is deprecated")
def test_linprog_failure():
    """Test that linprog fails."""
    # 创建线性间隔数据
    X = np.linspace(0, 10, num=10).reshape(-1, 1)
    y = np.linspace(0, 10, num=10)
    # 使用 interior-point 方法的 QuantileRegressor，设定最大迭代次数为 1
    reg = QuantileRegressor(
        alpha=0, solver="interior-point", solver_options={"maxiter": 1}
    )

    # 验证线性规划在 QuantileRegressor 中失败的警告
    msg = "Linear programming for QuantileRegressor did not succeed."
    with pytest.warns(ConvergenceWarning, match=msg):
        reg.fit(X, y)


@skip_if_32bit
@pytest.mark.skipif(
    sp_version <= parse_version("1.6.0"),
    reason="Solvers are available as of scipy 1.6.0",
)
@pytest.mark.parametrize(
    "sparse_container", CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS
)
@pytest.mark.parametrize("solver", ["highs", "highs-ds", "highs-ipm"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_sparse_input(sparse_container, solver, fit_intercept, default_solver):
    # 这个测试函数验证稀疏输入是否正常处理
    """Test that sparse and dense X give same results."""
    # 创建一个回归测试，确保稀疏和密集的 X 数据给出相同的结果
    X, y = make_regression(n_samples=100, n_features=20, random_state=1, noise=1.0)
    # 生成稀疏的 X 数据
    X_sparse = sparse_container(X)
    # 设置 alpha 值作为正则化参数
    alpha = 1e-4
    # 使用 QuantileRegressor 对象拟合密集 X 数据
    quant_dense = QuantileRegressor(
        alpha=alpha, fit_intercept=fit_intercept, solver=default_solver
    ).fit(X, y)
    # 使用 QuantileRegressor 对象拟合稀疏 X 数据
    quant_sparse = QuantileRegressor(
        alpha=alpha, fit_intercept=fit_intercept, solver=solver
    ).fit(X_sparse, y)
    # 检查稀疏和密集模型的系数是否非常接近
    assert_allclose(quant_sparse.coef_, quant_dense.coef_, rtol=1e-2)
    # 如果包含截距项，检查稀疏和密集模型的截距项是否接近
    if fit_intercept:
        assert quant_sparse.intercept_ == approx(quant_dense.intercept_)
        # 检查我们仍然能够预测到分数（fraction）
        assert 0.45 <= np.mean(y < quant_sparse.predict(X_sparse)) <= 0.57
def test_error_interior_point_future(X_y_data, monkeypatch):
    """Check that we will raise a proper error when requesting
    `solver='interior-point'` in SciPy >= 1.11.
    """
    X, y = X_y_data  # 解包输入数据 X_y_data 为 X 和 y

    import sklearn.linear_model._quantile  # 导入 sklearn 中的 quantile 回归模块

    with monkeypatch.context() as m:  # 使用 monkeypatch 上下文
        m.setattr(sklearn.linear_model._quantile, "sp_version", parse_version("1.11.0"))
        # 设置 quantile 模块中的 sp_version 属性为 SciPy 版本 1.11.0 的解析版本

        err_msg = "Solver interior-point is not anymore available in SciPy >= 1.11.0."
        # 定义错误消息，说明在 SciPy >= 1.11.0 中 interior-point 求解器不再可用

        with pytest.raises(ValueError, match=err_msg):
            # 使用 pytest 来检查是否会抛出 ValueError 异常，并匹配预期的错误消息
            QuantileRegressor(solver="interior-point").fit(X, y)
            # 创建 QuantileRegressor 实例，使用 solver='interior-point' 参数来尝试拟合数据 X, y
```