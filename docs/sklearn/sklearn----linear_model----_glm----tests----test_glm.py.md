# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_glm\tests\test_glm.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# 导入必要的库和模块
import itertools  # 导入 itertools 模块，用于生成迭代器
import warnings  # 导入 warnings 模块，用于处理警告
from functools import partial  # 导入 partial 函数，用于创建偏函数

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试
import scipy  # 导入 SciPy 库，用于科学计算
from numpy.testing import assert_allclose  # 导入 assert_allclose 函数，用于数值测试
from scipy import linalg  # 导入 linalg 模块，用于线性代数运算
from scipy.optimize import minimize, root  # 导入 minimize 和 root 函数，用于优化和求根

# 导入 scikit-learn 相关模块和类
from sklearn._loss import HalfBinomialLoss, HalfPoissonLoss, HalfTweedieLoss
from sklearn._loss.link import IdentityLink, LogLink
from sklearn.base import clone  # 导入 clone 函数，用于克隆对象
from sklearn.datasets import make_low_rank_matrix, make_regression  # 导入数据生成函数
from sklearn.exceptions import ConvergenceWarning  # 导入 ConvergenceWarning 异常类
from sklearn.linear_model import (
    GammaRegressor,
    PoissonRegressor,
    Ridge,
    TweedieRegressor,
)  # 导入线性模型类
from sklearn.linear_model._glm import _GeneralizedLinearRegressor  # 导入广义线性回归基类
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver  # 导入牛顿法求解器
from sklearn.linear_model._linear_loss import LinearModelLoss  # 导入线性模型损失类
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance  # 导入评估指标
from sklearn.model_selection import train_test_split  # 导入数据集划分函数

SOLVERS = ["lbfgs", "newton-cholesky"]  # 定义求解器列表

# 定义一个自定义的广义线性模型回归器类，继承自 _GeneralizedLinearRegressor
class BinomialRegressor(_GeneralizedLinearRegressor):
    def _get_loss(self):
        return HalfBinomialLoss()  # 返回二项分布损失对象

# 定义一个特殊的优化函数，结合了 Nelder-Mead 方法和牛顿法
def _special_minimize(fun, grad, x, tol_NM, tol):
    # 使用 Nelder-Mead 方法寻找一个良好的起始点
    res_NM = minimize(
        fun, x, method="Nelder-Mead", options={"xatol": tol_NM, "fatol": tol_NM}
    )
    # 然后通过函数梯度的根查找方法对结果进行精细化，这比优化函数本身更精确
    res = root(
        grad,
        res_NM.x,
        method="lm",
        options={"ftol": tol, "xtol": tol, "gtol": tol},
    )
    return res.x

# 定义一个 pytest 的 fixture 函数，用于生成回归数据
@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=107, n_features=10, n_informative=80, noise=0.5, random_state=2
    )
    return X, y

# 定义一个 pytest 的 fixture 函数，用于生成 GLM 数据集和模型参数的组合
@pytest.fixture(
    params=itertools.product(
        ["long", "wide"],
        [
            BinomialRegressor(),
            PoissonRegressor(),
            GammaRegressor(),
            TweedieRegressor(power=1.5),
        ],
    ),
    ids=lambda param: f"{param[0]}-{param[1]}",
)
def glm_dataset(global_random_seed, request):
    """Dataset with GLM solutions, well conditioned X.

    This is inspired by ols_ridge_dataset in test_ridge.py.

    The construction is based on the SVD decomposition of X = U S V'.

    Parameters
    ----------
    type : {"long", "wide"}
        If "long", then n_samples > n_features.
        If "wide", then n_features > n_samples.
    model : a GLM model

    For "wide", we return the minimum norm solution:

        min ||w||_2 subject to w = argmin deviance(X, y, w)

    Note that the deviance is always minimized if y = inverse_link(X w) is possible to
    achieve, which it is in the wide data case. Therefore, we can construct the
    """
    # 返回 GLM 数据集和模型参数的组合
    return request.param
    """
    solution with minimum norm like (wide) OLS:

        min ||w||_2 subject to link(y) = raw_prediction = X w

    Returns
    -------
    model : GLM model
        返回使用的广义线性模型对象
    X : ndarray
        特征矩阵，最后一列为截距项
    y : ndarray
        标签数据
    coef_unpenalized : ndarray
        无惩罚项的最小范数解，即在存在歧义时最小化损失函数和 ||w||_2
        最后一个系数是截距
    coef_penalized : ndarray
        带 L2 正则化 (alpha=l2_reg_strength=1) 的 GLM 解
        最后一个系数是截距
    l2_reg_strength : float
        始终为 1
    """
    data_type, model = request.param
    # 让大维度至少是小维度的两倍，有助于处理类似 (X, X) 的奇异矩阵
    if data_type == "long":
        n_samples, n_features = 12, 4
    else:
        n_samples, n_features = 4, 12
    k = min(n_samples, n_features)
    rng = np.random.RandomState(global_random_seed)
    X = make_low_rank_matrix(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=k,
        tail_strength=0.1,
        random_state=rng,
    )
    X[:, -1] = 1  # 最后一列作为截距项
    U, s, Vt = linalg.svd(X, full_matrices=False)
    assert np.all(s > 1e-3)  # 确保所有奇异值大于1e-3
    assert np.max(s) / np.min(s) < 100  # X 的条件数小于100，确保问题是良好条件的

    if data_type == "long":
        coef_unpenalized = rng.uniform(low=1, high=3, size=n_features)
        coef_unpenalized *= rng.choice([-1, 1], size=n_features)
        raw_prediction = X @ coef_unpenalized
    else:
        raw_prediction = rng.uniform(low=-3, high=3, size=n_samples)
        # 最小范数解 min ||w||_2 使得 raw_prediction = X w:
        # w = X'(XX')^-1 raw_prediction = V s^-1 U' raw_prediction
        coef_unpenalized = Vt.T @ np.diag(1 / s) @ U.T @ raw_prediction

    linear_loss = LinearModelLoss(base_loss=model._get_loss(), fit_intercept=True)
    sw = np.full(shape=n_samples, fill_value=1 / n_samples)
    y = linear_loss.base_loss.link.inverse(raw_prediction)

    # 添加 L2 正则化惩罚 l2_reg_strength * ||coef||_2^2，其中 l2_reg_strength=1，并用优化器求解。
    # 注意问题是良好条件的，我们能得到精确的结果。
    l2_reg_strength = 1
    fun = partial(
        linear_loss.loss,
        X=X[:, :-1],
        y=y,
        sample_weight=sw,
        l2_reg_strength=l2_reg_strength,
    )
    grad = partial(
        linear_loss.gradient,
        X=X[:, :-1],
        y=y,
        sample_weight=sw,
        l2_reg_strength=l2_reg_strength,
    )
    coef_penalized_with_intercept = _special_minimize(
        fun, grad, coef_unpenalized, tol_NM=1e-6, tol=1e-14
    )

    linear_loss = LinearModelLoss(base_loss=model._get_loss(), fit_intercept=False)
    fun = partial(
        linear_loss.loss,
        X=X[:, :-1],
        y=y,
        sample_weight=sw,
        l2_reg_strength=l2_reg_strength,
    )
    # 创建一个梯度函数，作为线性损失函数的梯度函数的偏函数
    grad = partial(
        linear_loss.gradient,  # 使用线性损失函数的梯度函数
        X=X[:, :-1],            # 输入特征矩阵 X 去除最后一列（截距项）
        y=y,                    # 目标变量 y
        sample_weight=sw,       # 样本权重
        l2_reg_strength=l2_reg_strength,  # L2 正则化强度
    )
    
    # 使用特殊的最小化函数 _special_minimize 来计算带有系数惩罚但不含截距项的模型系数
    coef_penalized_without_intercept = _special_minimize(
        fun,                            # 目标函数
        grad,                           # 梯度函数
        coef_unpenalized[:-1],          # 不含截距项的未惩罚系数
        tol_NM=1e-6,                    # 牛顿法的容差
        tol=1e-14                       # 数值优化的容差
    )

    # 断言确保带有截距项的惩罚系数的 L2 范数小于未惩罚系数的 L2 范数
    assert np.linalg.norm(coef_penalized_with_intercept) < np.linalg.norm(
        coef_unpenalized
    )

    # 返回模型、输入特征矩阵 X、目标变量 y、未惩罚系数、带有截距项的惩罚系数、不含截距项的惩罚系数、L2 正则化强度
    return (
        model,
        X,
        y,
        coef_unpenalized,
        coef_penalized_with_intercept,
        coef_penalized_without_intercept,
        l2_reg_strength,
    )
# 使用 pytest.mark.parametrize 装饰器，为 test_glm_regression 函数参数化设置多个 solver 参数
# 和 fit_intercept 参数的组合进行测试。
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_glm_regression(solver, fit_intercept, glm_dataset):
    """Test that GLM converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    """
    # 从数据集中获取模型、特征矩阵 X、目标值 y、拦截系数等信息
    model, X, y, _, coef_with_intercept, coef_without_intercept, alpha = glm_dataset
    # 设置 GLM 模型的参数字典
    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-12,
        max_iter=1000,
    )

    # 克隆模型并设置参数
    model = clone(model).set_params(**params)
    # 移除 X 的最后一列，即拦截列
    X = X[:, :-1]  # remove intercept
    if fit_intercept:
        coef = coef_with_intercept
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        coef = coef_without_intercept
        intercept = 0

    # 使用 X 和 y 拟合模型
    model.fit(X, y)

    # 根据不同的 solver 设置不同的相对容差 rtol
    rtol = 5e-5 if solver == "lbfgs" else 1e-9
    # 断言模型的截距和预期的截距值在容差范围内接近
    assert model.intercept_ == pytest.approx(intercept, rel=rtol)
    # 断言模型的系数和预期的系数值在容差范围内接近
    assert_allclose(model.coef_, coef, rtol=rtol)

    # 使用 sample_weight 进行拟合的相同测试
    model = (
        clone(model).set_params(**params).fit(X, y, sample_weight=np.ones(X.shape[0]))
    )
    assert model.intercept_ == pytest.approx(intercept, rel=rtol)
    assert_allclose(model.coef_, coef, rtol=rtol)


# 使用 pytest.mark.parametrize 装饰器，为 test_glm_regression_hstacked_X 函数参数化设置多个 solver 参数
# 和 fit_intercept 参数的组合进行测试。
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_glm_regression_hstacked_X(solver, fit_intercept, glm_dataset):
    """Test that GLM converges for all solvers to correct solution on hstacked data.

    We work with a simple constructed data set with known solution.
    Fit on [X] with alpha is the same as fit on [X, X]/2 with alpha/2.
    For long X, [X, X] is still a long but singular matrix.
    """
    # 从数据集中获取模型、特征矩阵 X、目标值 y、拦截系数等信息
    model, X, y, _, coef_with_intercept, coef_without_intercept, alpha = glm_dataset
    n_samples, n_features = X.shape
    # 设置 GLM 模型的参数字典，alpha 为原始值的一半
    params = dict(
        alpha=alpha / 2,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-12,
        max_iter=1000,
    )

    # 克隆模型并设置参数
    model = clone(model).set_params(**params)
    # 移除 X 的最后一列，即拦截列
    X = X[:, :-1]  # remove intercept
    # 将 X 水平堆叠自身的一半作为新的特征矩阵 X
    X = 0.5 * np.concatenate((X, X), axis=1)
    # 断言新的特征矩阵 X 的秩不超过最小的维度
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features - 1)
    if fit_intercept:
        coef = coef_with_intercept
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        coef = coef_without_intercept
        intercept = 0

    # 忽略可能出现的收敛警告，继续拟合模型
    with warnings.catch_warnings():
        # XXX: 探索在某些情况下可能出现的收敛警告是否应该被视为 bug。
        # 在这段时间里，如果下面的断言通过，则不管警告是否存在都不失败。
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)

    # 根据不同的 solver 设置不同的相对容差 rtol
    rtol = 2e-4 if solver == "lbfgs" else 5e-9
    # 断言模型的截距和预期的截距值在容差范围内接近
    assert model.intercept_ == pytest.approx(intercept, rel=rtol)
    # 断言模型的系数和预期的系数值在容差范围内接近，通过在末尾堆叠 coef 获得
    assert_allclose(model.coef_, np.r_[coef, coef], rtol=rtol)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_glm_regression_vstacked_X(solver, fit_intercept, glm_dataset):
    """Test that GLM converges for all solvers to correct solution on vstacked data.

    We work with a simple constructed data set with known solution.
    Fit on [X] with alpha is the same as fit on [X], [y]
                                                [X], [y] with 1 * alpha.
    It is the same alpha as the average loss stays the same.
    For wide X, [X', X'] is a singular matrix.
    """
    model, X, y, _, coef_with_intercept, coef_without_intercept, alpha = glm_dataset
    n_samples, n_features = X.shape
    
    # 设置模型参数
    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-12,
        max_iter=1000,
    )

    # 克隆模型并设置参数
    model = clone(model).set_params(**params)
    
    # 移除截距项
    X = X[:, :-1]  # remove intercept
    
    # 将 X 垂直堆叠自身，构成新的 X
    X = np.concatenate((X, X), axis=0)
    
    # 断言 X 的秩不超过最小的样本数和特征数
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)
    
    # 将 y 垂直堆叠自身，构成新的 y
    y = np.r_[y, y]
    
    # 根据是否拟合截距项选择系数和截距
    if fit_intercept:
        coef = coef_with_intercept
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        coef = coef_without_intercept
        intercept = 0
    
    # 拟合模型
    model.fit(X, y)
    
    # 根据求解器选择相对误差容差
    rtol = 3e-5 if solver == "lbfgs" else 5e-9
    
    # 断言模型的截距项接近预期值
    assert model.intercept_ == pytest.approx(intercept, rel=rtol)
    
    # 断言模型的系数接近预期值
    assert_allclose(model.coef_, coef, rtol=rtol)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_glm_regression_unpenalized(solver, fit_intercept, glm_dataset):
    """Test that unpenalized GLM converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    Note: This checks the minimum norm solution for wide X, i.e.
    n_samples < n_features:
        min ||w||_2 subject to w = argmin deviance(X, y, w)
    """
    model, X, y, coef, _, _, _ = glm_dataset
    n_samples, n_features = X.shape
    
    # 设置模型参数
    alpha = 0  # unpenalized
    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-12,
        max_iter=1000,
    )

    # 克隆模型并设置参数
    model = clone(model).set_params(**params)
    
    # 如果拟合截距项，移除 X 的截距项，并设置截距
    if fit_intercept:
        X = X[:, :-1]  # remove intercept
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        intercept = 0

    # 捕获警告以处理特定情况
    with warnings.catch_warnings():
        if solver.startswith("newton") and n_samples < n_features:
            # 对于 newton 类型的求解器在 n_samples < n_features 的情况下会发出警告，
            # 自动切换到 LBFGS 求解器确保模型仍然能够收敛
            warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)
        
        # XXX: 研究是否应将某些情况下出现的收敛警告视为 bug。
        # 在这段时间内，无论下面的断言是否通过，不会因警告而失败。
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        # 拟合模型
        model.fit(X, y)
    # 如果样本数大于特征数，则执行以下操作
    # FIXME: `assert_allclose(model.coef_, coef)` 在大多数情况下正常工作，但在特征数大于样本数时会失败。
    # 大多数当前的广义线性模型求解器在 fit_intercept=True 时不返回最小范数解。
    if n_samples > n_features:
        rtol = 5e-5 if solver == "lbfgs" else 1e-7
        # 断言模型截距与预期截距接近
        assert model.intercept_ == pytest.approx(intercept)
        # 断言模型系数与预期系数接近
        assert_allclose(model.coef_, coef, rtol=rtol)
    else:
        # 如果是欠定问题，预测值等于真实值 y。以下内容展示我们得到了一个解，即目标函数的（非唯一）最小值 ...
        rtol = 5e-5
        if solver == "newton-cholesky":
            rtol = 5e-4
        # 断言模型预测值与真实值 y 接近
        assert_allclose(model.predict(X), y, rtol=rtol)

        # 计算真实解和模型解的范数
        norm_solution = np.linalg.norm(np.r_[intercept, coef])
        norm_model = np.linalg.norm(np.r_[model.intercept_, model.coef_])
        if solver == "newton-cholesky":
            # XXX: 这个求解器表现出随机行为。有时会找到满足 norm_model <= norm_solution 的解！因此我们有条件地进行检查。
            if norm_model < (1 + 1e-12) * norm_solution:
                # 断言模型截距与预期截距接近
                assert model.intercept_ == pytest.approx(intercept)
                # 断言模型系数与预期系数接近
                assert_allclose(model.coef_, coef, rtol=rtol)
        elif solver == "lbfgs" and fit_intercept:
            # 但它不是最小范数解。否则范数将相等。
            # 断言模型解的范数大于真实解的范数
            assert norm_model > (1 + 1e-12) * norm_solution

            # 参见 https://github.com/scikit-learn/scikit-learn/issues/23670。
            # 注意：即使添加微小的惩罚项也无法给出最小范数解。
            # XXX: 我们本可以天真地期望 LBFGS 通过添加一个非常小的惩罚项找到最小范数解，但是由于某种我们当前不完全理解的原因，即使这种方式也会失败。
        else:
            # 当 `fit_intercept=False` 时，LBFGS 在这个问题上自然收敛到最小范数解。
            # XXX: 我们是否有任何理论保证为什么会这样？
            # 断言模型截距与预期截距接近，使用相对容差 rtol
            assert model.intercept_ == pytest.approx(intercept, rel=rtol)
            # 断言模型系数与预期系数接近，使用相对容差 rtol
            assert_allclose(model.coef_, coef, rtol=rtol)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_glm_regression_unpenalized_hstacked_X(solver, fit_intercept, glm_dataset):
    """Test that unpenalized GLM converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    GLM fit on [X] is the same as fit on [X, X]/2.
    For long X, [X, X] is a singular matrix and we check against the minimum norm
    solution:
        min ||w||_2 subject to w = argmin deviance(X, y, w)
    """
    model, X, y, coef, _, _, _ = glm_dataset
    n_samples, n_features = X.shape
    alpha = 0  # unpenalized

    # 设置 GLM 模型参数
    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-12,
        max_iter=1000,
    )

    # 克隆模型并设置参数
    model = clone(model).set_params(**params)

    if fit_intercept:
        intercept = coef[-1]
        coef = coef[:-1]
        if n_samples > n_features:
            # 对于拟合截距，我们需要移除一个截距列并且不除以2
            X = X[:, :-1]  # 移除截距
            X = 0.5 * np.concatenate((X, X), axis=1)
        else:
            # 对于长 X，我们保留一个截距列并且不除以2
            X = np.c_[X[:, :-1], X[:, :-1], X[:, -1]]
    else:
        intercept = 0
        X = 0.5 * np.concatenate((X, X), axis=1)

    # 检查 X 的秩是否小于等于样本数和特征数中较小的那个
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)

    with warnings.catch_warnings():
        if solver.startswith("newton"):
            # 对于 newton 类型的 solver，应该警告并自动切换到 LBFGS，但模型仍应该收敛
            warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)
        # XXX: 调查在某些情况下是否应该将 ConvergenceWarning 视为 bug
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # 拟合模型
        model.fit(X, y)

    if fit_intercept and n_samples < n_features:
        # 在这里我们需要特别注意
        model_intercept = 2 * model.intercept_
        model_coef = 2 * model.coef_[:-1]  # 排除另一个截距项
        # 对于最小范数解，应有 assert model.intercept_ == pytest.approx(model.coef_[-1])
    else:
        model_intercept = model.intercept_
        model_coef = model.coef_

    if n_samples > n_features:
        # 断言截距应该接近预期值
        assert model_intercept == pytest.approx(intercept)
        rtol = 1e-4
        # 断言系数应该接近预期值
        assert_allclose(model_coef, np.r_[coef, coef], rtol=rtol)
    else:
        # 如果不满足上述条件，则进行下面的验证步骤
        # 因为这是一个欠定问题，预测结果等于目标值 y。下面展示我们得到一个解，即目标函数的（非唯一）最小值...
        
        # 根据不同的求解器设置容差值 rtol
        rtol = 1e-6 if solver == "lbfgs" else 5e-6
        # 断言模型预测结果与目标值 y 在容差范围内相等
        assert_allclose(model.predict(X), y, rtol=rtol)
        
        # 如果求解器为 "lbfgs" 且 fit_intercept=True，或者求解器为 "newton-cholesky"
        if (solver == "lbfgs" and fit_intercept) or solver == "newton-cholesky":
            # 与 test_glm_regression_unpenalized 中的相同。
            # 但这不是最小范数解。否则范数将相等。
            # 计算范数解
            norm_solution = np.linalg.norm(
                0.5 * np.r_[intercept, intercept, coef, coef]
            )
            # 计算模型的范数
            norm_model = np.linalg.norm(np.r_[model.intercept_, model.coef_])
            # 断言模型的范数大于 (1 + 1e-12) 倍的范数解
            assert norm_model > (1 + 1e-12) * norm_solution
            # 对于最小范数解，我们会有
            # assert model.intercept_ == pytest.approx(model.coef_[-1])
        else:
            # 断言模型的截距与给定的截距 intercept 在相对容差为 5e-6 的范围内近似相等
            assert model_intercept == pytest.approx(intercept, rel=5e-6)
            # 断言模型的系数与给定的系数 coef 在容差范围内近似相等
            assert_allclose(model_coef, np.r_[coef, coef], rtol=1e-4)
# 使用 pytest 的参数化功能，为每个参数组合执行测试函数
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_glm_regression_unpenalized_vstacked_X(solver, fit_intercept, glm_dataset):
    """Test that unpenalized GLM converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    GLM fit on [X] is the same as fit on [X], [y]
                                         [X], [y].
    For wide X, [X', X'] is a singular matrix and we check against the minimum norm
    solution:
        min ||w||_2 subject to w = argmin deviance(X, y, w)
    """
    model, X, y, coef, _, _, _ = glm_dataset
    n_samples, n_features = X.shape
    alpha = 0  # unpenalized
    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-12,
        max_iter=1000,
    )

    # 克隆模型并设置参数
    model = clone(model).set_params(**params)
    if fit_intercept:
        X = X[:, :-1]  # 移除截距项
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        intercept = 0
    X = np.concatenate((X, X), axis=0)
    # 断言 X 的秩不超过样本数和特征数的较小值
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)
    y = np.r_[y, y]

    with warnings.catch_warnings():
        if solver.startswith("newton") and n_samples < n_features:
            # 对于牛顿法求解器，当样本数小于特征数时，应发出警告并自动切换到 LBFGS 求解器
            # 在这种情况下，模型仍应收敛。
            warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)
        # XXX: 调查 ConvergenceWarning 在某些情况下是否应视为 bug
        # 在此期间，无论是否存在警告，只要下面的断言通过，就不会失败。
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # 使用模型拟合数据
        model.fit(X, y)

    if n_samples > n_features:
        rtol = 5e-5 if solver == "lbfgs" else 1e-6
        # 断言截距项与预期截距项近似相等
        assert model.intercept_ == pytest.approx(intercept)
        # 断言模型系数与预期系数近似相等
        assert_allclose(model.coef_, coef, rtol=rtol)
    else:
        # 如果不是特定的求解器，则设置容差值 rtol
        rtol = 1e-6 if solver == "lbfgs" else 5e-6
        # 断言模型预测结果与真实结果 y 接近，使用容差 rtol
        assert_allclose(model.predict(X), y, rtol=rtol)

        # 计算解的范数和模型参数的范数
        norm_solution = np.linalg.norm(np.r_[intercept, coef])
        norm_model = np.linalg.norm(np.r_[model.intercept_, model.coef_])
        if solver == "newton-cholesky":
            # 对于 newton-cholesky 求解器，检查其随机行为及条件
            # 有时会得到 norm_model <= norm_solution 的解！因此我们有条件地检查。
            if not (norm_model > (1 + 1e-12) * norm_solution):
                # 断言模型截距接近预期值 intercept
                assert model.intercept_ == pytest.approx(intercept)
                # 断言模型系数接近预期值 coef，使用容差 rtol=1e-4
                assert_allclose(model.coef_, coef, rtol=1e-4)
        elif solver == "lbfgs" and fit_intercept:
            # 对于 lbfgs 求解器且 fit_intercept=True 的情况
            # 与 test_glm_regression_unpenalized 中的条件相同。
            # 但它不是最小范数解。否则范数将相等。
            assert norm_model > (1 + 1e-12) * norm_solution
        else:
            # 其他情况下设置不同的容差值 rtol
            rtol = 1e-5 if solver == "newton-cholesky" else 1e-4
            # 断言模型截距接近预期值 intercept，使用相对容差 rtol
            assert model.intercept_ == pytest.approx(intercept, rel=rtol)
            # 断言模型系数接近预期值 coef，使用容差 rtol
            assert_allclose(model.coef_, coef, rtol=rtol)
def test_sample_weights_validation():
    """Test the raised errors in the validation of sample_weight."""

    # scalar value but not positive
    X = [[1]]  # 创建一个包含单个样本的二维数组 X
    y = [1]  # 创建目标变量 y，包含单个元素
    weights = 0  # 初始化权重为标量值 0
    glm = _GeneralizedLinearRegressor()  # 创建广义线性回归器对象

    # Positive weights are accepted
    glm.fit(X, y, sample_weight=1)  # 对广义线性回归器进行拟合，使用权重值为 1

    # 2d array
    weights = [[0]]  # 初始化权重为包含单个元素的二维数组
    with pytest.raises(ValueError, match="must be 1D array or scalar"):
        glm.fit(X, y, weights)  # 对广义线性回归器进行拟合，使用二维数组作为权重，预期会引发 ValueError 异常

    # 1d but wrong length
    weights = [1, 0]  # 初始化权重为包含两个元素的一维数组
    msg = r"sample_weight.shape == \(2,\), expected \(1,\)!"
    with pytest.raises(ValueError, match=msg):
        glm.fit(X, y, weights)  # 对广义线性回归器进行拟合，使用长度错误的一维数组作为权重，预期会引发 ValueError 异常


@pytest.mark.parametrize(
    "glm",
    [
        TweedieRegressor(power=3),
        PoissonRegressor(),
        GammaRegressor(),
        TweedieRegressor(power=1.5),
    ],
)
def test_glm_wrong_y_range(glm):
    """Test GLM regression with wrong y range on various estimators."""
    y = np.array([-1, 2])  # 创建具有非法值的目标变量数组
    X = np.array([[1], [1]])  # 创建简单的特征矩阵
    msg = r"Some value\(s\) of y are out of the valid range of the loss"
    with pytest.raises(ValueError, match=msg):
        glm.fit(X, y)  # 对指定的广义线性模型进行拟合，预期会引发 ValueError 异常


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_glm_identity_regression(fit_intercept):
    """Test GLM regression with identity link on a simple dataset."""
    coef = [1.0, 2.0]  # 创建真实模型的系数向量
    X = np.array([[1, 1, 1, 1, 1], [0, 1, 2, 3, 4]]).T  # 创建特征矩阵
    y = np.dot(X, coef)  # 根据真实模型生成目标变量
    glm = _GeneralizedLinearRegressor(
        alpha=0,
        fit_intercept=fit_intercept,
        tol=1e-12,
    )  # 创建广义线性回归器对象

    if fit_intercept:
        glm.fit(X[:, 1:], y)  # 如果拟合包括截距，对广义线性回归器进行拟合
        assert_allclose(glm.coef_, coef[1:], rtol=1e-10)  # 断言拟合后的系数接近真实系数
        assert_allclose(glm.intercept_, coef[0], rtol=1e-10)  # 断言拟合后的截距接近真实截距
    else:
        glm.fit(X, y)  # 如果拟合不包括截距，对广义线性回归器进行拟合
        assert_allclose(glm.coef_, coef, rtol=1e-12)  # 断言拟合后的系数接近真实系数


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize(
    "GLMEstimator", [_GeneralizedLinearRegressor, PoissonRegressor, GammaRegressor]
)
def test_glm_sample_weight_consistency(fit_intercept, alpha, GLMEstimator):
    """Test that the impact of sample_weight is consistent across GLM estimators."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = rng.rand(n_samples, n_features)  # 创建随机特征矩阵
    y = rng.rand(n_samples)  # 创建随机目标变量数组
    glm_params = dict(alpha=alpha, fit_intercept=fit_intercept)

    glm = GLMEstimator(**glm_params).fit(X, y)  # 创建并拟合指定的广义线性模型
    coef = glm.coef_.copy()  # 复制拟合后的系数向量

    # sample_weight=np.ones(..) should be equivalent to sample_weight=None
    sample_weight = np.ones(y.shape)  # 创建全为 1 的权重数组
    glm.fit(X, y, sample_weight=sample_weight)  # 使用权重数组重新拟合模型
    assert_allclose(glm.coef_, coef, rtol=1e-12)  # 断言拟合后的系数与未加权拟合的系数接近

    # sample_weight are normalized to 1 so, scaling them has no effect
    sample_weight = 2 * np.ones(y.shape)  # 创建所有元素均为 2 的权重数组
    glm.fit(X, y, sample_weight=sample_weight)  # 使用加倍权重数组重新拟合模型
    assert_allclose(glm.coef_, coef, rtol=1e-12)  # 断言拟合后的系数与未加权拟合的系数接近

    # setting one element of sample_weight to 0 is equivalent to removing
    # the corresponding sample
    sample_weight = np.ones(y.shape)  # 创建全为 1 的权重数组
    sample_weight[-1] = 0  # 将最后一个样本的权重设为 0，相当于删除这个样本
    glm.fit(X, y, sample_weight=sample_weight)  # 使用修改后的权重数组重新拟合模型
    coef1 = glm.coef_.copy()  # 复制拟合后的系数向量
    # 使用广义线性模型 (GLM) 对数据进行拟合，使用除最后一个样本外的数据进行训练
    glm.fit(X[:-1], y[:-1])
    # 断言：验证拟合后的系数与预期系数在非常小的相对误差下相等
    assert_allclose(glm.coef_, coef1, rtol=1e-12)

    # 检查通过将样本权重乘以2是否等效于对应样本重复两次
    X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]])
    # 创建样本权重数组，初始值为1，前半部分乘以2
    sample_weight_1 = np.ones(len(y))
    sample_weight_1[: n_samples // 2] = 2

    # 使用指定参数的 GLM 估计器拟合数据，使用自定义的样本权重
    glm1 = GLMEstimator(**glm_params).fit(X, y, sample_weight=sample_weight_1)

    # 使用相同参数的 GLM 估计器拟合扩展后的数据，不使用样本权重
    glm2 = GLMEstimator(**glm_params).fit(X2, y2, sample_weight=None)
    # 断言：验证两种拟合结果的系数在非常小的相对误差下相等
    assert_allclose(glm1.coef_, glm2.coef_)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "estimator",
    [
        PoissonRegressor(),
        GammaRegressor(),
        TweedieRegressor(power=3.0),
        TweedieRegressor(power=0, link="log"),
        TweedieRegressor(power=1.5),
        TweedieRegressor(power=4.5),
    ],
)
def test_glm_log_regression(solver, fit_intercept, estimator):
    """Test GLM regression with log link on a simple dataset."""
    coef = [0.2, -0.1]  # 设置模型的系数
    X = np.array([[0, 1, 2, 3, 4], [1, 1, 1, 1, 1]]).T  # 创建输入特征矩阵 X
    y = np.exp(np.dot(X, coef))  # 生成目标变量 y，使用指数函数构造
    glm = clone(estimator).set_params(  # 克隆给定的回归器，并设置其参数
        alpha=0,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-8,
    )
    if fit_intercept:
        res = glm.fit(X[:, :-1], y)  # 拟合模型，不包括最后一个特征列
        assert_allclose(res.coef_, coef[:-1], rtol=1e-6)  # 断言模型系数接近预期值（不包括截距）
        assert_allclose(res.intercept_, coef[-1], rtol=1e-6)  # 断言截距接近预期值
    else:
        res = glm.fit(X, y)  # 拟合模型，包括所有特征列
        assert_allclose(res.coef_, coef, rtol=2e-6)  # 断言模型系数接近预期值

@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_warm_start(solver, fit_intercept, global_random_seed):
    n_samples, n_features = 100, 10  # 样本数和特征数
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        bias=fit_intercept * 1.0,
        noise=1.0,
        random_state=global_random_seed,
    )  # 生成回归数据集 X 和 y
    y = np.abs(y)  # Poisson 回归要求目标变量非负
    alpha = 1  # 设置正则化参数 alpha
    params = {
        "solver": solver,
        "fit_intercept": fit_intercept,
        "tol": 1e-10,
    }  # 设置模型参数

    glm1 = PoissonRegressor(warm_start=False, max_iter=1000, alpha=alpha, **params)  # 创建不使用热启动的 Poisson 回归模型
    glm1.fit(X, y)  # 拟合第一个 Poisson 回归模型

    glm2 = PoissonRegressor(warm_start=True, max_iter=1, alpha=alpha, **params)  # 创建使用热启动的 Poisson 回归模型，最大迭代次数为 1
    # 因为我们故意设置 max_iter=1，以便求解器应该引发 ConvergenceWarning。
    with pytest.warns(ConvergenceWarning):
        glm2.fit(X, y)  # 拟合第二个 Poisson 回归模型

    linear_loss = LinearModelLoss(
        base_loss=glm1._get_loss(),
        fit_intercept=fit_intercept,
    )  # 创建线性模型损失对象

    sw = np.full_like(y, fill_value=1 / n_samples)  # 创建样本权重向量

    objective_glm1 = linear_loss.loss(
        coef=np.r_[glm1.coef_, glm1.intercept_] if fit_intercept else glm1.coef_,
        X=X,
        y=y,
        sample_weight=sw,
        l2_reg_strength=alpha,
    )  # 计算第一个模型的目标函数值

    objective_glm2 = linear_loss.loss(
        coef=np.r_[glm2.coef_, glm2.intercept_] if fit_intercept else glm2.coef_,
        X=X,
        y=y,
        sample_weight=sw,
        l2_reg_strength=alpha,
    )  # 计算第二个模型的目标函数值

    assert objective_glm1 < objective_glm2  # 断言第一个模型的目标函数值小于第二个模型的目标函数值

    glm2.set_params(max_iter=1000)  # 设置第二个模型的最大迭代次数为 1000
    glm2.fit(X, y)  # 再次拟合第二个模型

    # 由于 lbfgs 求解器从先前迭代中计算近似 Hessian 矩阵，所以在热启动的情况下不会严格相同。
    rtol = 2e-4 if solver == "lbfgs" else 1e-9
    assert_allclose(glm1.coef_, glm2.coef_, rtol=rtol)  # 断言两个模型的系数接近
    # 使用 assert_allclose 函数比较两个广义线性模型 glm1 和 glm2 在给定数据集 X, y 上的得分，并验证它们之间的差异是否在指定的相对容差范围内。
    assert_allclose(glm1.score(X, y), glm2.score(X, y), rtol=1e-5)
# 使用 pytest 的参数化装饰器来定义多组测试参数：n_samples 和 n_features
# 取值分别为 (100, 10), (10, 100)
@pytest.mark.parametrize("n_samples, n_features", [(100, 10), (10, 100)])
# 使用 pytest 的参数化装饰器定义另外的测试参数：fit_intercept
# 取值为 True 和 False
@pytest.mark.parametrize("fit_intercept", [True, False])
# 使用 pytest 的参数化装饰器定义另外的测试参数：sample_weight
# 取值为 None 和 True
@pytest.mark.parametrize("sample_weight", [None, True])
# 定义测试函数 test_normal_ridge_comparison，用于比较普通数据集和 Ridge 回归的表现
def test_normal_ridge_comparison(
    n_samples, n_features, fit_intercept, sample_weight, request
):
    """Compare with Ridge regression for Normal distributions."""
    # 设置测试数据集大小为 10
    test_size = 10
    # 生成符合正态分布的回归数据 X, y
    X, y = make_regression(
        n_samples=n_samples + test_size,
        n_features=n_features,
        n_informative=n_features - 2,
        noise=0.5,
        random_state=42,
    )

    # 根据 n_samples 和 n_features 的关系设置 Ridge 回归的参数
    if n_samples > n_features:
        ridge_params = {"solver": "svd"}
    else:
        ridge_params = {"solver": "saga", "max_iter": 1000000, "tol": 1e-7}

    # 将数据集拆分为训练集和测试集
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=test_size, random_state=0)

    # 设置 Ridge 回归的 alpha 参数
    alpha = 1.0
    if sample_weight is None:
        sw_train = None
        alpha_ridge = alpha * n_samples
    else:
        # 如果有样本权重，则随机生成样本权重
        sw_train = np.random.RandomState(0).rand(len(y_train))
        alpha_ridge = alpha * sw_train.sum()

    # 创建 Ridge 回归对象
    ridge = Ridge(
        alpha=alpha_ridge,
        random_state=42,
        fit_intercept=fit_intercept,
        **ridge_params,
    )
    # 在训练集上拟合 Ridge 回归模型
    ridge.fit(X_train, y_train, sample_weight=sw_train)

    # 创建广义线性回归对象（用于对比）
    glm = _GeneralizedLinearRegressor(
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=300,
        tol=1e-5,
    )
    # 在训练集上拟合广义线性回归模型
    glm.fit(X_train, y_train, sample_weight=sw_train)

    # 断言：检查模型系数形状是否正确
    assert glm.coef_.shape == (X.shape[1],)
    # 断言：检查模型系数是否接近（使用绝对容差）
    assert_allclose(glm.coef_, ridge.coef_, atol=5e-5)
    # 断言：检查截距是否接近（使用相对容差）
    assert_allclose(glm.intercept_, ridge.intercept_, rtol=1e-5)
    # 断言：检查在训练集上预测结果是否接近（使用相对容差）
    assert_allclose(glm.predict(X_train), ridge.predict(X_train), rtol=2e-4)
    # 断言：检查在测试集上预测结果是否接近（使用相对容差）
    assert_allclose(glm.predict(X_test), ridge.predict(X_test), rtol=2e-4)


# 使用 pytest 的参数化装饰器定义测试参数：solver
# 取值为 "lbfgs" 和 "newton-cholesky"
@pytest.mark.parametrize("solver", ["lbfgs", "newton-cholesky"])
# 定义测试函数 test_poisson_glmnet，用于比较 Poisson 回归和 glmnet 的表现
def test_poisson_glmnet(solver):
    """Compare Poisson regression with L2 regularization and LogLink to glmnet"""
    # 准备数据 X 和 y
    X = np.array([[-2, -1, 1, 2], [0, 0, 1, 1]]).T
    y = np.array([0, 1, 1, 2])
    
    # 创建 Poisson 回归对象
    glm = PoissonRegressor(
        alpha=1,
        fit_intercept=True,
        tol=1e-7,
        max_iter=300,
        solver=solver,
    )
    # 在数据集上拟合 Poisson 回归模型
    glm.fit(X, y)

    # 断言：检查截距是否接近预期值（使用相对容差）
    assert_allclose(glm.intercept_, -0.12889386979, rtol=1e-5)
    # 断言：检查系数是否接近预期值（使用相对容差）
    assert_allclose(glm.coef_, [0.29019207995, 0.03741173122], rtol=1e-5)


# 定义测试函数 test_convergence_warning，用于检查收敛警告
def test_convergence_warning(regression_data):
    # 获取回归数据 X, y
    X, y = regression_data

    # 创建广义线性回归对象，设置最大迭代次数为 1，容差为 1e-20
    est = _GeneralizedLinearRegressor(max_iter=1, tol=1e-20)
    # 使用 pytest 模块来检测是否有特定的警告信息，这里是 ConvergenceWarning 警告
    with pytest.warns(ConvergenceWarning):
        # 使用估算器对象（est）对输入数据集（X）进行拟合，使用目标值（y）来训练模型
        est.fit(X, y)
@pytest.mark.parametrize(
    "name, link_class", [("identity", IdentityLink), ("log", LogLink)]
)
# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数，测试 GLM 的 Tweedie link 参数设置为字符串时的情况
def test_tweedie_link_argument(name, link_class):
    """Test GLM link argument set as string."""
    y = np.array([0.1, 0.5])  # in range of all distributions
    X = np.array([[1], [2]])
    # 创建 TweedieRegressor 模型对象，设定 power 和 link 参数，并拟合数据
    glm = TweedieRegressor(power=1, link=name).fit(X, y)
    # 断言模型内部的链接函数是否是预期的 link_class 类型
    assert isinstance(glm._base_loss.link, link_class)


@pytest.mark.parametrize(
    "power, expected_link_class",
    [
        (0, IdentityLink),  # normal
        (1, LogLink),  # poisson
        (2, LogLink),  # gamma
        (3, LogLink),  # inverse-gaussian
    ],
)
# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数，测试 GLM 的 Tweedie link 参数设置为 'auto' 时的情况
def test_tweedie_link_auto(power, expected_link_class):
    """Test that link='auto' delivers the expected link function"""
    y = np.array([0.1, 0.5])  # in range of all distributions
    X = np.array([[1], [2]])
    # 创建 TweedieRegressor 模型对象，设定 power 和 link 参数为 'auto'，并拟合数据
    glm = TweedieRegressor(link="auto", power=power).fit(X, y)
    # 断言模型内部的链接函数是否是预期的 expected_link_class 类型
    assert isinstance(glm._base_loss.link, expected_link_class)


@pytest.mark.parametrize("power", [0, 1, 1.5, 2, 3])
@pytest.mark.parametrize("link", ["log", "identity"])
# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数，测试 TweedieRegressor 模型的 score 方法
def test_tweedie_score(regression_data, power, link):
    """Test that GLM score equals d2_tweedie_score for Tweedie losses."""
    X, y = regression_data
    # 将 y 转为正数
    y = np.abs(y) + 1.0
    # 创建 TweedieRegressor 模型对象，设定 power 和 link 参数，并拟合数据
    glm = TweedieRegressor(power=power, link=link).fit(X, y)
    # 断言模型的 score 方法计算结果与 d2_tweedie_score 函数的结果是否近似相等
    assert glm.score(X, y) == pytest.approx(
        d2_tweedie_score(y, glm.predict(X), power=power)
    )


@pytest.mark.parametrize(
    "estimator, value",
    [
        (PoissonRegressor(), True),
        (GammaRegressor(), True),
        (TweedieRegressor(power=1.5), True),
        (TweedieRegressor(power=0), False),
    ],
)
# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数，测试不同 GLM 模型对象的 _get_tags 方法
def test_tags(estimator, value):
    assert estimator._get_tags()["requires_positive_y"] is value


def test_linalg_warning_with_newton_solver(global_random_seed):
    newton_solver = "newton-cholesky"
    rng = np.random.RandomState(global_random_seed)
    # 使用至少 20 个样本，以减少任何 global_random_seed 下生成退化数据集的可能性。
    X_orig = rng.normal(size=(20, 3))
    y = rng.poisson(
        np.exp(X_orig @ np.ones(X_orig.shape[1])), size=X_orig.shape[0]
    ).astype(np.float64)

    # 同一输入特征的共线变化。
    X_collinear = np.hstack([X_orig] * 10)

    # 考虑在这个问题上的常数基线的偏差。
    baseline_pred = np.full_like(y, y.mean())
    constant_model_deviance = mean_poisson_deviance(y, baseline_pred)
    assert constant_model_deviance > 1.0

    # 在条件良好的设计上，即使没有正则化，也不会引发警告。
    tol = 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 使用 PoissonRegressor 拟合数据，设定 solver 和其他参数
        reg = PoissonRegressor(solver=newton_solver, alpha=0.0, tol=tol).fit(X_orig, y)
    original_newton_deviance = mean_poisson_deviance(y, reg.predict(X_orig))

    # 在这个数据集上，我们应该有足够的数据点，以不使得它
    # 确保原始牛顿偏差大于0.2，以便更容易理解后续断言中的 rtol 的含义：
    assert original_newton_deviance > 0.2

    # 检查模型是否能够在 X_orig 中成功地比常数基线有显著改进（在训练集上评估时）：
    assert constant_model_deviance - original_newton_deviance > 0.1

    # LBFGS 对于共线设计是健壮的，因为其对 Hessian 矩阵的近似构造保证是对称正定的。记录其解：
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reg = PoissonRegressor(solver="lbfgs", alpha=0.0, tol=tol).fit(X_collinear, y)
    collinear_lbfgs_deviance = mean_poisson_deviance(y, reg.predict(X_collinear))

    # 预期 LBFGS 在共线数据上的解应接近于原始数据上牛顿方法的解：
    rtol = 1e-6
    assert collinear_lbfgs_deviance == pytest.approx(original_newton_deviance, rel=rtol)

    # 在共线版本的训练数据上拟合牛顿求解器时，没有正则化应该引发一个信息性警告，并回退到 LBFGS 求解器：
    msg = (
        "The inner solver of .*Newton.*Solver stumbled upon a singular or very "
        "ill-conditioned Hessian matrix"
    )
    with pytest.warns(scipy.linalg.LinAlgWarning, match=msg):
        reg = PoissonRegressor(solver=newton_solver, alpha=0.0, tol=tol).fit(
            X_collinear, y
        )
    # 结果应自动收敛到一个良好的解决方案：
    collinear_newton_deviance = mean_poisson_deviance(y, reg.predict(X_collinear))
    assert collinear_newton_deviance == pytest.approx(
        original_newton_deviance, rel=rtol
    )

    # 稍微增加正则化应该解决问题：
    with warnings.catch_warnings():
        warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
        reg = PoissonRegressor(solver=newton_solver, alpha=1e-10).fit(X_collinear, y)

    # 共线数据上稍微有惩罚的模型应该与原始数据上无惩罚的模型足够接近：
    penalized_collinear_newton_deviance = mean_poisson_deviance(
        y, reg.predict(X_collinear)
    )
    assert penalized_collinear_newton_deviance == pytest.approx(
        original_newton_deviance, rel=rtol
    )
@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_newton_solver_verbosity(capsys, verbose):
    """Test the std output of verbose newton solvers."""

    # 创建一个包含两个元素的浮点数数组
    y = np.array([1, 2], dtype=float)
    # 创建一个2x2的浮点数数组
    X = np.array([[1.0, 0], [0, 1]], dtype=float)
    # 创建一个线性模型损失对象，使用HalfPoissonLoss作为基础损失，不拟合截距
    linear_loss = LinearModelLoss(base_loss=HalfPoissonLoss(), fit_intercept=False)
    
    # 初始化NewtonCholeskySolver对象
    sol = NewtonCholeskySolver(
        coef=linear_loss.init_zero_coef(X),  # 使用线性模型损失对象初始化系数
        linear_loss=linear_loss,  # 指定线性模型损失对象
        l2_reg_strength=0,  # 指定L2正则化强度为0
        verbose=verbose,  # 设定详细程度
    )
    # 使用NewtonCholeskySolver对象解决问题，返回解
    sol.solve(X, y, None)  # 返回 array([0., 0.69314758])
    captured = capsys.readouterr()  # 读取捕获的标准输出和错误

    if verbose == 0:
        assert captured.out == ""  # 断言输出为空字符串
    else:
        msg = [
            "Newton iter=1",  # Newton迭代次数信息
            "Check Convergence",  # 检查收敛性信息
            "1. max |gradient|",  # 最大梯度绝对值信息
            "2. Newton decrement",  # Newton递减信息
            "Solver did converge at loss = ",  # 求解器在损失函数值处收敛信息
        ]
        for m in msg:
            assert m in captured.out  # 断言消息存在于捕获的输出中

    if verbose >= 2:
        msg = ["Backtracking Line Search", "line search iteration="]
        for m in msg:
            assert m in captured.out  # 断言消息存在于捕获的输出中

    # 将Newton求解器设置为一个完全错误的Newton步骤状态。
    sol = NewtonCholeskySolver(
        coef=linear_loss.init_zero_coef(X),
        linear_loss=linear_loss,
        l2_reg_strength=0,
        verbose=verbose,
    )
    # 使用NewtonCholeskySolver对象设置求解环境
    sol.setup(X=X, y=y, sample_weight=None)
    sol.iteration = 1  # 设置迭代次数为1
    sol.update_gradient_hessian(X=X, y=y, sample_weight=None)  # 更新梯度和Hessian矩阵
    sol.coef_newton = np.array([1.0, 0])  # 设置Newton步长向量
    sol.gradient_times_newton = sol.gradient @ sol.coef_newton  # 计算梯度和Newton步长的点积
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sol.line_search(X=X, y=y, sample_weight=None)  # 执行线搜索
        captured = capsys.readouterr()  # 读取捕获的标准输出和错误
    if verbose >= 1:
        assert (
            "Line search did not converge and resorts to lbfgs instead." in captured.out
        )  # 断言特定消息存在于捕获的输出中

    # 将Newton求解器设置为一个具有不良Newton步骤的状态，使得线搜索中的损失改进非常小。
    sol = NewtonCholeskySolver(
        coef=np.array([1e-12, 0.69314758]),
        linear_loss=linear_loss,
        l2_reg_strength=0,
        verbose=verbose,
    )
    # 使用NewtonCholeskySolver对象设置求解环境
    sol.setup(X=X, y=y, sample_weight=None)
    sol.iteration = 1  # 设置迭代次数为1
    sol.update_gradient_hessian(X=X, y=y, sample_weight=None)  # 更新梯度和Hessian矩阵
    sol.coef_newton = np.array([1e-6, 0])  # 设置Newton步长向量
    sol.gradient_times_newton = sol.gradient @ sol.coef_newton  # 计算梯度和Newton步长的点积
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sol.line_search(X=X, y=y, sample_weight=None)  # 执行线搜索
        captured = capsys.readouterr()  # 读取捕获的标准输出和错误
    if verbose >= 2:
        msg = [
            "line search iteration=",  # 线搜索迭代次数信息
            "check loss improvement <= armijo term:",  # 检查损失改进小于阿米约条件信息
            "check loss |improvement| <= eps * |loss_old|:",  # 检查损失改进的绝对值小于eps乘以旧损失的绝对值信息
            "check sum(|gradient|) < sum(|gradient_old|):",  # 检查梯度绝对值之和小于旧梯度绝对值之和信息
        ]
        for m in msg:
            assert m in captured.out  # 断言消息存在于捕获的输出中

    # Test for a case with negative hessian. We badly initialize coef for a Tweedie
    # 使用非标准链接的损失函数，例如带有对数链接的逆高斯分布 Tweedie 损失。
    linear_loss = LinearModelLoss(
        base_loss=HalfTweedieLoss(power=3), fit_intercept=False
    )

    # 初始化一个 Newton-Cholesky 求解器，初始系数为线性损失的零系数加一。
    sol = NewtonCholeskySolver(
        coef=linear_loss.init_zero_coef(X) + 1,
        linear_loss=linear_loss,
        l2_reg_strength=0,
        verbose=verbose,
    )

    # 忽略收敛警告，通过捕获警告避免输出到控制台。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # 使用 Newton-Cholesky 求解器求解线性模型。
        sol.solve(X, y, None)

    # 读取和捕获输出的内容。
    captured = capsys.readouterr()

    # 如果 verbose 大于等于 1，则断言输出包含特定的警告信息。
    if verbose >= 1:
        assert (
            "The inner solver detected a pointwise Hessian with many negative values"
            " and resorts to lbfgs instead." in captured.out
        )
```