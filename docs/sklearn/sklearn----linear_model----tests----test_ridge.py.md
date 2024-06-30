# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_ridge.py`

```
# 导入警告模块，用于管理警告信息的显示
import warnings
# 用于生成迭代器的模块，用于生成多个输入的笛卡尔积
from itertools import product

# 导入第三方数值计算库 numpy，并简写为 np
import numpy as np
# 导入 pytest 模块，用于编写和运行测试用例
import pytest
# 导入 scipy 中的线性代数模块
from scipy import linalg

# 导入 sklearn 中的配置上下文管理器和数据集模块
from sklearn import config_context, datasets
# 导入 sklearn 中的模型克隆函数
from sklearn.base import clone
# 导入 sklearn 中的数据集生成函数
from sklearn.datasets import (
    make_classification,
    make_low_rank_matrix,
    make_multilabel_classification,
    make_regression,
)
# 导入 sklearn 中的异常类，用于处理收敛警告
from sklearn.exceptions import ConvergenceWarning
# 导入 sklearn 中的线性回归模型和岭回归模型
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    ridge_regression,
)
# 导入 sklearn 中岭回归相关的内部函数和类
from sklearn.linear_model._ridge import (
    _check_gcv_mode,
    _RidgeGCV,
    _solve_cholesky,
    _solve_cholesky_kernel,
    _solve_lbfgs,
    _solve_svd,
    _X_CenterStackOp,
)
# 导入 sklearn 中的评估指标函数和类
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
# 导入 sklearn 中的模型选择工具
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneOut,
    cross_val_predict,
)
# 导入 sklearn 中的数据预处理函数
from sklearn.preprocessing import minmax_scale
# 导入 sklearn 中的随机状态检查函数
from sklearn.utils import check_random_state
# 导入 sklearn 中的数组接口函数
from sklearn.utils._array_api import (
    _NUMPY_NAMESPACE_NAMES,
    _atol_for_type,
    _convert_to_numpy,
    yield_namespace_device_dtype_combinations,
    yield_namespaces,
)
# 导入 sklearn 中的测试辅助函数
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
# 导入 sklearn 中的估计器检查函数
from sklearn.utils.estimator_checks import (
    _array_api_for_tests,
    _get_check_estimator_ids,
    check_array_api_input_and_values,
)
# 导入 sklearn 中的修复功能
from sklearn.utils.fixes import (
    _IS_32BIT,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)

# 定义岭回归可用的求解器列表
SOLVERS = ["svd", "sparse_cg", "cholesky", "lsqr", "sag", "saga"]
# 需要截距项的稀疏求解器
SPARSE_SOLVERS_WITH_INTERCEPT = ("sparse_cg", "sag")
# 不需要截距项的稀疏求解器
SPARSE_SOLVERS_WITHOUT_INTERCEPT = ("sparse_cg", "cholesky", "lsqr", "sag", "saga")

# 加载糖尿病数据集
diabetes = datasets.load_diabetes()
# 提取数据和目标值
X_diabetes, y_diabetes = diabetes.data, diabetes.target
# 创建索引数组
ind = np.arange(X_diabetes.shape[0])
# 使用随机状态对索引数组进行洗牌
rng = np.random.RandomState(0)
rng.shuffle(ind)
# 选择前200个索引
ind = ind[:200]
# 根据选定的索引获取部分数据集
X_diabetes, y_diabetes = X_diabetes[ind], y_diabetes[ind]

# 加载鸢尾花数据集
iris = datasets.load_iris()
# 提取数据和目标值
X_iris, y_iris = iris.data, iris.target

# 定义一个函数，用于计算分类准确率的回调函数
def _accuracy_callable(y_test, y_pred, **kwargs):
    return np.mean(y_test == y_pred)

# 定义一个函数，用于计算均方误差的回调函数
def _mean_squared_error_callable(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()

# 定义一个 pytest 的夹具（fixture），生成 OLS 和 Ridge 回归数据集
@pytest.fixture(params=["long", "wide"])
def ols_ridge_dataset(global_random_seed, request):
    """Dataset with OLS and Ridge solutions, well conditioned X.

    The construction is based on the SVD decomposition of X = U S V'.

    Parameters
    ----------
    type : {"long", "wide"}
        If "long", then n_samples > n_features.
        If "wide", then n_features > n_samples.

    For "wide", we return the minimum norm solution w = X' (XX')^-1 y:

        min ||w||_2 subject to X w = y

    Returns
    -------
    X : ndarray
        Last column of 1, i.e. intercept.
    y : ndarray
    """
    # 省略具体数据集生成细节，根据请求参数生成不同形状的数据集
    pass
    # coef_ols 是形状未指定的 ndarray
    # 最小二乘法的解，即 min ||X w - y||_2_2（在存在歧义时，最小化 ||w||_2）
    # 最后一个系数是截距(intercept)。
    coef_ols : ndarray of shape

    # coef_ridge 是形状为 (5,) 的 ndarray
    # alpha=1 时的岭回归解，即 min ||X w - y||_2_2 + ||w||_2^2。
    # 最后一个系数是截距(intercept)。
    coef_ridge : ndarray of shape (5,)

    """
    # 让较大的维度比较小的维度大两倍以上。
    # 这有助于构造奇异矩阵（如 (X, X)）时。
    if request.param == "long":
        n_samples, n_features = 12, 4
    else:
        n_samples, n_features = 4, 12
    
    # 计算矩阵的秩，选择最小的样本数和特征数
    k = min(n_samples, n_features)
    rng = np.random.RandomState(global_random_seed)
    
    # 生成一个低秩矩阵 X
    X = make_low_rank_matrix(
        n_samples=n_samples, n_features=n_features, effective_rank=k, random_state=rng
    )
    
    # 最后一列作为截距(intercept)
    X[:, -1] = 1
    
    # 对 X 进行奇异值分解
    U, s, Vt = linalg.svd(X)
    
    # 确保所有奇异值都大于 1e-3
    assert np.all(s > 1e-3)
    
    # 分割 U 和 Vt 为 U1, U2 和 Vt1
    U1, U2 = U[:, :k], U[:, k:]
    Vt1, _ = Vt[:k, :], Vt[k:, :]

    if request.param == "long":
        # 添加一个在 X'y 乘积中消失的项
        coef_ols = rng.uniform(low=-10, high=10, size=n_features)
        y = X @ coef_ols
        
        # 对 y 添加一个额外的项，这个项的平方服从正态分布
        y += U2 @ rng.normal(size=n_samples - n_features) ** 2
    else:
        # 生成一个服从均匀分布的 y
        y = rng.uniform(low=-10, high=10, size=n_samples)
        
        # 计算最小二乘法的系数 w = X'(XX')^-1 y = V s^-1 U' y
        coef_ols = Vt1.T @ np.diag(1 / s) @ U1.T @ y

    # 添加惩罚项 alpha * ||coef||_2^2，其中 alpha=1，并通过正规方程组求解。
    # 注意问题条件良好，我们能得到精确的结果。
    alpha = 1
    d = alpha * np.identity(n_features)
    d[-1, -1] = 0  # 截距(intercept)不被惩罚
    coef_ridge = linalg.solve(X.T @ X + d, X.T @ y)

    # 确保
    R_OLS = y - X @ coef_ols
    R_Ridge = y - X @ coef_ridge
    
    # 确保最小二乘法的残差范数小于岭回归的残差范数
    assert np.linalg.norm(R_OLS) < np.linalg.norm(R_Ridge)

    # 返回 X, y, coef_ols, coef_ridge
    return X, y, coef_ols, coef_ridge
# 使用 pytest 的 parametrize 装饰器，为 test_ridge_regression 函数参数化测试用例
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_regression(solver, fit_intercept, ols_ridge_dataset, global_random_seed):
    """Test that Ridge converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    """
    # 从 ols_ridge_dataset 解包数据
    X, y, _, coef = ols_ridge_dataset
    alpha = 1.0  # 因为 ols_ridge_dataset 使用这个值作为 alpha

    # 设置 Ridge 模型的参数
    params = dict(
        alpha=alpha,
        fit_intercept=True,
        solver=solver,
        tol=1e-15 if solver in ("sag", "saga") else 1e-10,  # 根据 solver 设置 tol
        random_state=global_random_seed,
    )

    # 计算残差和 R2 值
    res_null = y - np.mean(y)
    res_Ridge = y - X @ coef
    R2_Ridge = 1 - np.sum(res_Ridge**2) / np.sum(res_null**2)

    # 创建 Ridge 模型对象
    model = Ridge(**params)

    # 移除 X 的截距项
    X = X[:, :-1]

    if fit_intercept:
        intercept = coef[-1]
    else:
        # 对 X 和 y 进行中心化处理
        X = X - X.mean(axis=0)
        y = y - y.mean()
        intercept = 0

    # 拟合 Ridge 模型
    model.fit(X, y)
    coef = coef[:-1]  # 移除截距项对应的 coef 值

    # 断言模型的截距项接近预期值
    assert model.intercept_ == pytest.approx(intercept)

    # 断言模型的系数接近预期值
    assert_allclose(model.coef_, coef)

    # 断言模型的 R2 得分接近预期值
    assert model.score(X, y) == pytest.approx(R2_Ridge)

    # 使用 sample_weight 进行同样的测试
    model = Ridge(**params).fit(X, y, sample_weight=np.ones(X.shape[0]))
    assert model.intercept_ == pytest.approx(intercept)
    assert_allclose(model.coef_, coef)
    assert model.score(X, y) == pytest.approx(R2_Ridge)

    # 断言模型的 solver_ 属性与参数中的 solver 相符
    assert model.solver_ == solver


# 使用 pytest 的 parametrize 装饰器，为 test_ridge_regression_hstacked_X 函数参数化测试用例
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_regression_hstacked_X(
    solver, fit_intercept, ols_ridge_dataset, global_random_seed
):
    """Test that Ridge converges for all solvers to correct solution on hstacked data.

    We work with a simple constructed data set with known solution.
    Fit on [X] with alpha is the same as fit on [X, X]/2 with alpha/2.
    For long X, [X, X] is a singular matrix.
    """
    # 从 ols_ridge_dataset 解包数据
    X, y, _, coef = ols_ridge_dataset
    n_samples, n_features = X.shape
    alpha = 1.0  # 因为 ols_ridge_dataset 使用这个值作为 alpha

    # 设置 Ridge 模型的参数，alpha 减半
    model = Ridge(
        alpha=alpha / 2,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-15 if solver in ("sag", "saga") else 1e-10,  # 根据 solver 设置 tol
        random_state=global_random_seed,
    )

    # 移除 X 的截距项，并进行水平堆叠
    X = X[:, :-1]
    X = 0.5 * np.concatenate((X, X), axis=1)

    # 确保堆叠后的 X 是奇异矩阵或者其秩不超过最小的维度
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features - 1)

    if fit_intercept:
        intercept = coef[-1]
    else:
        # 对 X 和 y 进行中心化处理
        X = X - X.mean(axis=0)
        y = y - y.mean()
        intercept = 0

    # 拟合 Ridge 模型
    model.fit(X, y)
    coef = coef[:-1]  # 移除截距项对应的 coef 值

    # 断言模型的截距项接近预期值
    assert model.intercept_ == pytest.approx(intercept)

    # 系数不完全在同一数量级上，添加一个小的公差以减少测试的脆弱性
    assert_allclose(model.coef_, np.r_[coef, coef], atol=1e-8)
# 定义测试函数，用于测试带有堆叠特征的 Ridge 回归的收敛性
def test_ridge_regression_vstacked_X(
    solver, fit_intercept, ols_ridge_dataset, global_random_seed
):
    """Test that Ridge converges for all solvers to correct solution on vstacked data.

    We work with a simple constructed data set with known solution.
    Fit on [X] with alpha is the same as fit on [X], [y]
                                                [X], [y] with 2 * alpha.
    For wide X, [X', X'] is a singular matrix.
    """
    # 从输入数据集中获取 X, y, 忽略其他返回的值，同时获取真实的系数 coef
    X, y, _, coef = ols_ridge_dataset
    # 获取样本数和特征数
    n_samples, n_features = X.shape
    # 设置 alpha 值为 1.0，与 ols_ridge_dataset 中的一致
    alpha = 1.0  # because ols_ridge_dataset uses this.

    # 创建 Ridge 模型对象
    model = Ridge(
        alpha=2 * alpha,  # 设置正则化参数为 2 * alpha
        fit_intercept=fit_intercept,  # 是否拟合截距
        solver=solver,  # 使用的求解器
        tol=1e-15 if solver in ("sag", "saga") else 1e-10,  # 设置收敛容差
        random_state=global_random_seed,  # 随机数种子
    )
    # 移除 X 的最后一列，即拟合截距
    X = X[:, :-1]  # remove intercept
    # 将 X 堆叠一次作为新的输入数据
    X = np.concatenate((X, X), axis=0)
    # 断言堆叠后的 X 的秩不超过样本数和特征数的最小值
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)
    # 将 y 堆叠一次作为新的目标值
    y = np.r_[y, y]
    # 如果拟合截距为真
    if fit_intercept:
        # 获取真实系数的最后一个元素作为拟合截距
        intercept = coef[-1]
    else:
        # 标准化 X 和 y 的平均值为零
        X = X - X.mean(axis=0)
        y = y - y.mean()
        # 拟合截距设为零
        intercept = 0
    # 对模型进行拟合
    model.fit(X, y)
    # 更新真实系数，移除最后一个元素作为拟合截距后的系数
    coef = coef[:-1]

    # 断言模型拟合后的截距与预期一致
    assert model.intercept_ == pytest.approx(intercept)
    # 断言模型拟合后的系数与真实系数在一定容差范围内相等
    # 添加一个小的容差以增强测试的鲁棒性
    assert_allclose(model.coef_, coef, atol=1e-8)
    else:
        # 因为这是一个欠定问题，所以残差应该为0。这表明我们得到了一个解 X w = y ....
        assert_allclose(model.predict(X), y)
        assert_allclose(X @ coef + intercept, y)
        # 但这不是最小范数解。（这两者应该相等。）
        assert np.linalg.norm(np.r_[model.intercept_, model.coef_]) > np.linalg.norm(
            np.r_[intercept, coef]
        )

        # 声明测试预期失败，因为 Ridge 回归不能提供最小范数解。
        pytest.xfail(reason="Ridge does not provide the minimum norm solution.")
        assert model.intercept_ == pytest.approx(intercept)
        assert_allclose(model.coef_, coef)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_regression_unpenalized_hstacked_X(
    solver, fit_intercept, ols_ridge_dataset, global_random_seed
):
    """Test that unpenalized Ridge = OLS converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    OLS fit on [X] is the same as fit on [X, X]/2.
    For long X, [X, X] is a singular matrix and we check against the minimum norm
    solution:
        min ||w||_2 subject to min ||X w - y||_2
    """
    X, y, coef, _ = ols_ridge_dataset
    n_samples, n_features = X.shape
    alpha = 0  # OLS

    # Initialize Ridge regression model with specified parameters
    model = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-15 if solver in ("sag", "saga") else 1e-10,
        random_state=global_random_seed,
    )

    if fit_intercept:
        X = X[:, :-1]  # remove intercept
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        intercept = 0

    # Create a new feature matrix by horizontally stacking X with itself scaled by 0.5
    X = 0.5 * np.concatenate((X, X), axis=1)
    
    # Assert that the rank of the matrix X is less than or equal to the minimum of samples and features
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)
    
    # Fit the model using the modified feature matrix X and target vector y
    model.fit(X, y)

    if n_samples > n_features or not fit_intercept:
        # Assert that the intercept of the model is approximately equal to the expected intercept
        assert model.intercept_ == pytest.approx(intercept)
        
        if solver == "cholesky":
            # Skip the test if using Cholesky solver for a singular X
            pytest.skip()
        
        # Assert that the coefficients of the model are approximately equal to the expected coefficients
        assert_allclose(model.coef_, np.r_[coef, coef])
    else:
        # FIXME: Same as in test_ridge_regression_unpenalized.
        # As it is an underdetermined problem, residuals = 0. This shows that we get
        # a solution to X w = y ....
        assert_allclose(model.predict(X), y)
        
        # But it is not the minimum norm solution. (This should be equal.)
        assert np.linalg.norm(np.r_[model.intercept_, model.coef_]) > np.linalg.norm(
            np.r_[intercept, coef, coef]
        )
        
        # Expect the test to fail as Ridge does not provide the minimum norm solution
        pytest.xfail(reason="Ridge does not provide the minimum norm solution.")
        
        # Assert that the intercept of the model is approximately equal to the expected intercept
        assert model.intercept_ == pytest.approx(intercept)
        
        # Assert that the coefficients of the model are approximately equal to the expected coefficients
        assert_allclose(model.coef_, np.r_[coef, coef])


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_regression_unpenalized_vstacked_X(
    solver, fit_intercept, ols_ridge_dataset, global_random_seed
):
    """Test that unpenalized Ridge = OLS converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    OLS fit on [X] is the same as fit on [X], [y]
                                         [X], [y].
    For wide X, [X', X'] is a singular matrix and we check against the minimum norm
    solution:
        min ||w||_2 subject to X w = y
    """
    X, y, coef, _ = ols_ridge_dataset
    n_samples, n_features = X.shape
    alpha = 0  # OLS
    # 创建 Ridge 回归模型对象，设置模型参数
    model = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-15 if solver in ("sag", "saga") else 1e-10,  # 根据 solver 类型设置收敛容差
        random_state=global_random_seed,
    )

    # 如果需要拟合截距，从特征矩阵 X 中移除最后一列（截距）
    if fit_intercept:
        X = X[:, :-1]  # 移除截距
        intercept = coef[-1]  # 保存拟合的截距
        coef = coef[:-1]  # 更新系数向量

    else:
        intercept = 0  # 如果不需要拟合截距，设截距为 0

    # 将特征矩阵 X 沿垂直方向复制一份
    X = np.concatenate((X, X), axis=0)

    # 断言特征矩阵 X 的秩不超过样本数和特征数的最小值
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)

    # 将目标向量 y 沿着垂直方向复制一份
    y = np.r_[y, y]

    # 使用扩展后的特征矩阵和目标向量拟合 Ridge 回归模型
    model.fit(X, y)

    # 根据问题的不同情况进行断言验证
    if n_samples > n_features or not fit_intercept:
        # 验证拟合模型的截距是否近似于预期的截距
        assert model.intercept_ == pytest.approx(intercept)
        # 验证拟合模型的系数向量是否近似于预期的系数向量
        assert_allclose(model.coef_, coef)
    else:
        # 当问题欠定时，断言预测结果与目标向量 y 的近似性
        assert_allclose(model.predict(X), y)
        # 同时断言拟合模型的范数是否大于预期的范数
        assert np.linalg.norm(np.r_[model.intercept_, model.coef_]) > np.linalg.norm(
            np.r_[intercept, coef]
        )

        # 标记测试失败并说明原因
        pytest.xfail(reason="Ridge 模型未提供最小范数解")
        # 再次验证拟合模型的截距是否近似于预期的截距
        assert model.intercept_ == pytest.approx(intercept)
        # 再次验证拟合模型的系数向量是否近似于预期的系数向量
        assert_allclose(model.coef_, coef)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("alpha", [1.0, 1e-2])
def test_ridge_regression_sample_weights(
    solver,
    fit_intercept,
    sparse_container,
    alpha,
    ols_ridge_dataset,
    global_random_seed,
):
    """Test that Ridge with sample weights gives correct results.

    We use the following trick:
        ||y - Xw||_2 = (z - Aw)' W (z - Aw)
    for z=[y, y], A' = [X', X'] (vstacked), and W[:n/2] + W[n/2:] = 1, W=diag(W)
    """
    # 根据给定的参数进行参数化测试，分别测试不同的求解器、是否拟合截距、稀疏矩阵容器、正则化参数alpha

    if sparse_container is not None:
        # 如果稀疏矩阵容器不为空
        if fit_intercept and solver not in SPARSE_SOLVERS_WITH_INTERCEPT:
            # 如果需要拟合截距，并且求解器不支持带截距的稀疏求解
            pytest.skip()
        elif not fit_intercept and solver not in SPARSE_SOLVERS_WITHOUT_INTERCEPT:
            # 如果不需要拟合截距，并且求解器不支持不带截距的稀疏求解
            pytest.skip()

    X, y, _, coef = ols_ridge_dataset
    # 从数据集中获取特征矩阵X、目标向量y、其他不必要的返回值和真实系数coef
    n_samples, n_features = X.shape
    # 获取样本数和特征数

    sw = rng.uniform(low=0, high=1, size=n_samples)
    # 使用均匀分布生成样本权重sw

    model = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        tol=1e-15 if solver in ["sag", "saga"] else 1e-10,
        max_iter=100_000,
        random_state=global_random_seed,
    )
    # 创建Ridge回归模型，设置正则化参数alpha、是否拟合截距、求解器、收敛阈值和最大迭代次数等参数

    X = X[:, :-1]  # 移除截距项
    X = np.concatenate((X, X), axis=0)
    # 将特征矩阵X复制一份并合并，用于之后的样本加权处理
    y = np.r_[y, y]
    # 将目标向量y复制一份并合并，用于之后的样本加权处理
    sw = np.r_[sw, 1 - sw] * alpha
    # 调整样本权重，使其满足加权的需求

    if fit_intercept:
        intercept = coef[-1]
    else:
        X = X - X.mean(axis=0)
        y = y - y.mean()
        intercept = 0
    # 根据是否需要拟合截距来调整特征矩阵X和目标向量y，并设置截距intercept

    if sparse_container is not None:
        X = sparse_container(X)
    # 如果有稀疏矩阵容器，则将特征矩阵X转换为稀疏格式

    model.fit(X, y, sample_weight=sw)
    # 使用加权样本拟合模型

    coef = coef[:-1]
    # 移除真实系数的最后一个元素，以匹配模型的系数形状

    assert model.intercept_ == pytest.approx(intercept)
    assert_allclose(model.coef_, coef)
    # 断言模型的截距和系数与预期的真实值接近


def test_primal_dual_relationship():
    y = y_diabetes.reshape(-1, 1)
    coef = _solve_cholesky(X_diabetes, y, alpha=[1e-2])
    K = np.dot(X_diabetes, X_diabetes.T)
    dual_coef = _solve_cholesky_kernel(K, y, alpha=[1e-2])
    coef2 = np.dot(X_diabetes.T, dual_coef).T
    assert_array_almost_equal(coef, coef2)
    # 测试原始和对偶问题之间的关系，验证它们的解是否一致


def test_ridge_regression_convergence_fail():
    rng = np.random.RandomState(0)
    y = rng.randn(5)
    X = rng.randn(5, 10)
    warning_message = r"sparse_cg did not converge after" r" [0-9]+ iterations."
    with pytest.warns(ConvergenceWarning, match=warning_message):
        ridge_regression(
            X, y, alpha=1.0, solver="sparse_cg", tol=0.0, max_iter=None, verbose=1
        )
    # 测试岭回归在收敛失败时的警告信息是否正确触发


def test_ridge_shapes_type():
    # Test shape of coef_ and intercept_
    rng = np.random.RandomState(0)
    n_samples, n_features = 5, 10
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    Y1 = y[:, np.newaxis]
    Y = np.c_[y, 1 + y]

    ridge = Ridge()

    ridge.fit(X, y)
    assert ridge.coef_.shape == (n_features,)
    assert ridge.intercept_.shape == ()
    assert isinstance(ridge.coef_, np.ndarray)
    assert isinstance(ridge.intercept_, float)
    # 测试岭回归模型系数和截距的形状和类型是否符合预期

    ridge.fit(X, Y1)
    assert ridge.coef_.shape == (1, n_features)
    # 当目标Y为二维时，测试岭回归模型系数的形状是否符合预期
    # 断言岭回归模型的截距数组形状为 (1,)，用于验证模型属性
    assert ridge.intercept_.shape == (1,)
    # 断言岭回归模型的系数数组类型为 NumPy 数组，用于验证模型属性
    assert isinstance(ridge.coef_, np.ndarray)
    # 再次断言岭回归模型的截距数组形状为 (1,)，用于验证模型属性
    assert isinstance(ridge.intercept_, np.ndarray)
    
    # 使用输入数据 X 和目标数据 Y 训练岭回归模型
    ridge.fit(X, Y)
    # 断言岭回归模型的系数数组形状为 (2, n_features)，其中 n_features 是特征的数量
    assert ridge.coef_.shape == (2, n_features)
    # 断言岭回归模型的截距数组形状为 (2,)，用于验证模型属性
    assert ridge.intercept_.shape == (2,)
    # 再次断言岭回归模型的系数数组类型为 NumPy 数组，用于验证模型属性
    assert isinstance(ridge.coef_, np.ndarray)
    # 再次断言岭回归模型的截距数组类型为 NumPy 数组，用于验证模型属性
    assert isinstance(ridge.intercept_, np.ndarray)
# 测试岭回归拦截器对多目标的影响，参考GitHub问题＃708
def test_ridge_intercept():
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 设置样本数和特征数
    n_samples, n_features = 5, 10
    # 生成随机样本矩阵
    X = rng.randn(n_samples, n_features)
    # 生成随机目标向量
    y = rng.randn(n_samples)
    # 创建包含多个目标的矩阵
    Y = np.c_[y, 1.0 + y]

    # 创建岭回归对象
    ridge = Ridge()

    # 在单目标上拟合岭回归模型
    ridge.fit(X, y)
    # 获取拦截器
    intercept = ridge.intercept_

    # 在多目标上拟合岭回归模型
    ridge.fit(X, Y)
    # 断言拦截器的值与单目标模型的拦截器值相等
    assert_almost_equal(ridge.intercept_[0], intercept)
    # 断言拦截器的值与单目标模型的拦截器值加1.0相等
    assert_almost_equal(ridge.intercept_[1], intercept + 1.0)


# 测试岭回归与最小二乘法（OLS）在alpha=0时的解是否一致
def test_ridge_vs_lstsq():
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 我们需要比特征数更多的样本数
    n_samples, n_features = 5, 4
    # 生成随机目标向量
    y = rng.randn(n_samples)
    # 生成随机样本矩阵
    X = rng.randn(n_samples, n_features)

    # 创建岭回归对象，设置alpha=0且不拟合截距项
    ridge = Ridge(alpha=0.0, fit_intercept=False)
    # 创建普通最小二乘法（OLS）对象，不拟合截距项
    ols = LinearRegression(fit_intercept=False)

    # 在岭回归模型上拟合数据
    ridge.fit(X, y)
    # 在OLS模型上拟合数据
    ols.fit(X, y)
    # 断言岭回归模型和OLS模型的系数近似相等
    assert_almost_equal(ridge.coef_, ols.coef_)

    # 再次在岭回归模型上拟合数据
    ridge.fit(X, y)
    # 再次在OLS模型上拟合数据
    ols.fit(X, y)
    # 断言岭回归模型和OLS模型的系数近似相等
    assert_almost_equal(ridge.coef_, ols.coef_)


# 测试使用单独的惩罚项来配置岭回归对象
def test_ridge_individual_penalties():
    # 创建随机数生成器
    rng = np.random.RandomState(42)

    # 设置样本数、特征数和目标数
    n_samples, n_features, n_targets = 20, 10, 5
    # 生成随机样本矩阵
    X = rng.randn(n_samples, n_features)
    # 生成随机目标矩阵
    y = rng.randn(n_samples, n_targets)

    # 创建一组惩罚项
    penalties = np.arange(n_targets)

    # 使用Cholesky分解求解器，分别拟合每个目标向量，并获取系数
    coef_cholesky = np.array(
        [
            Ridge(alpha=alpha, solver="cholesky").fit(X, target).coef_
            for alpha, target in zip(penalties, y.T)
        ]
    )

    # 使用不同求解器，拟合所有目标向量，并获取系数
    coefs_indiv_pen = [
        Ridge(alpha=penalties, solver=solver, tol=1e-12).fit(X, y).coef_
        for solver in ["svd", "sparse_cg", "lsqr", "cholesky", "sag", "saga"]
    ]
    # 断言Cholesky分解器得到的系数与其他求解器的系数近似相等
    for coef_indiv_pen in coefs_indiv_pen:
        assert_array_almost_equal(coef_cholesky, coef_indiv_pen)

    # 断言在目标数和惩罚项数不匹配时，会引发错误
    ridge = Ridge(alpha=penalties[:-1])
    err_msg = "Number of targets and number of penalties do not correspond: 4 != 5"
    with pytest.raises(ValueError, match=err_msg):
        ridge.fit(X, y)
    # 如果 uniform_weights 为真，则创建一个长度为 X.shape[0] 的全为1的数组作为样本权重 sw
    if uniform_weights:
        sw = np.ones(X.shape[0])
    # 否则，使用随机数生成器 rng 生成形状为 shape[0] 的服从卡方分布的随机数数组作为样本权重 sw
    else:
        sw = rng.chisquare(1, shape[0])
    
    # 对样本权重 sw 求平方根，得到 sqrt_sw
    sqrt_sw = np.sqrt(sw)
    
    # 计算 X 在加权 sw 下的加权平均值，axis=0 表示沿着列的方向求平均值，结果保存在 X_mean 中
    X_mean = np.average(X, axis=0, weights=sw)
    
    # 将 X 根据 X_mean 和 sqrt_sw 进行中心化处理，即每列减去对应的平均值再乘以平方根权重
    X_centered = (X - X_mean) * sqrt_sw[:, None]
    
    # 计算 X_centered 的真实 Gram 矩阵，即 X_centered 与其转置的乘积
    true_gram = X_centered.dot(X_centered.T)
    
    # 使用 csr_container 函数将 X 乘以 sqrt_sw[:, None] 转换为稀疏矩阵 X_sparse
    X_sparse = csr_container(X * sqrt_sw[:, None])
    
    # 创建一个 _RidgeGCV 对象 gcv，设定 fit_intercept=True
    gcv = _RidgeGCV(fit_intercept=True)
    
    # 调用 gcv 对象的 _compute_gram 方法，计算稀疏矩阵 X_sparse 的 Gram 矩阵和均值，结果保存在 computed_gram 和 computed_mean 中
    computed_gram, computed_mean = gcv._compute_gram(X_sparse, sqrt_sw)
    
    # 使用 assert_allclose 函数检查 X_mean 和 computed_mean 是否在数值上非常接近
    assert_allclose(X_mean, computed_mean)
    
    # 使用 assert_allclose 函数检查 true_gram 和 computed_gram 是否在数值上非常接近
    assert_allclose(true_gram, computed_gram)
@pytest.mark.parametrize("shape", [(10, 1), (13, 9), (3, 7), (2, 2), (20, 20)])
@pytest.mark.parametrize("uniform_weights", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数，参数化测试形状、是否均匀权重、稀疏矩阵容器
def test_compute_covariance(shape, uniform_weights, csr_container):
    # 设定随机数种子为0
    rng = np.random.RandomState(0)
    # 生成指定形状的随机数矩阵X
    X = rng.randn(*shape)
    # 根据是否均匀权重选择权重向量sw
    if uniform_weights:
        sw = np.ones(X.shape[0])
    else:
        sw = rng.chisquare(1, shape[0])
    # 计算平方根权重向量
    sqrt_sw = np.sqrt(sw)
    # 计算加权平均值
    X_mean = np.average(X, axis=0, weights=sw)
    # 居中化数据
    X_centered = (X - X_mean) * sqrt_sw[:, None]
    # 计算真实协方差矩阵
    true_covariance = X_centered.T.dot(X_centered)
    # 将X乘以sqrt_sw，并使用稀疏矩阵容器创建稀疏矩阵X_sparse
    X_sparse = csr_container(X * sqrt_sw[:, None])
    # 创建_RidgeGCV对象，设置拟合截距为True
    gcv = _RidgeGCV(fit_intercept=True)
    # 调用_RidgeGCV对象的_compute_covariance方法计算协方差和均值
    computed_cov, computed_mean = gcv._compute_covariance(X_sparse, sqrt_sw)
    # 断言计算得到的均值与真实均值相近
    assert_allclose(X_mean, computed_mean)
    # 断言计算得到的协方差与真实协方差相近
    assert_allclose(true_covariance, computed_cov)


def _make_sparse_offset_regression(
    n_samples=100,
    n_features=100,
    proportion_nonzero=0.5,
    n_informative=10,
    n_targets=1,
    bias=13.0,
    X_offset=30.0,
    noise=30.0,
    shuffle=True,
    coef=False,
    positive=False,
    random_state=None,
):
    # 使用make_regression生成稀疏偏移回归数据集X, y, c
    X, y, c = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        bias=bias,
        noise=noise,
        shuffle=shuffle,
        coef=True,
        random_state=random_state,
    )
    # 如果特征数为1，则将c转换为数组
    if n_features == 1:
        c = np.asarray([c])
    # 将X偏移X_offset
    X += X_offset
    # 根据proportion_nonzero生成掩码mask
    mask = (
        np.random.RandomState(random_state).binomial(1, proportion_nonzero, X.shape) > 0
    )
    # 复制X到removed_X
    removed_X = X.copy()
    # 将mask为False的元素置为0
    X[~mask] = 0.0
    # 将mask为True的元素置为0
    removed_X[mask] = 0.0
    # 调整y使其减去removed_X乘以c的结果
    y -= removed_X.dot(c)
    # 如果positive为True，则调整y使其加上X乘以abs(c)+1-c的结果，并更新c
    if positive:
        y += X.dot(np.abs(c) + 1 - c)
        c = np.abs(c) + 1
    # 如果特征数为1，则将c转换为标量
    if n_features == 1:
        c = c[0]
    # 如果coef为True，则返回X, y, c
    if coef:
        return X, y, c
    # 否则返回X, y
    return X, y


@pytest.mark.parametrize(
    "solver, sparse_container",
    (
        (solver, sparse_container)
        for (solver, sparse_container) in product(
            ["cholesky", "sag", "sparse_cg", "lsqr", "saga", "ridgecv"],
            [None] + CSR_CONTAINERS,
        )
        if sparse_container is None or solver in ["sparse_cg", "ridgecv"]
    ),
)
@pytest.mark.parametrize(
    "n_samples,dtype,proportion_nonzero",
    [(20, "float32", 0.1), (40, "float32", 1.0), (20, "float64", 0.2)],
)
@pytest.mark.parametrize("seed", np.arange(3))
# 定义测试函数，参数化测试求解器的一致性
def test_solver_consistency(
    solver, proportion_nonzero, n_samples, dtype, sparse_container, seed
):
    # 设置alpha为1.0
    alpha = 1.0
    # 如果proportion_nonzero大于0.9，则设置noise为50.0，否则设置为500.0
    noise = 50.0 if proportion_nonzero > 0.9 else 500.0
    # 使用_make_sparse_offset_regression生成稀疏偏移回归数据集X, y
    X, y = _make_sparse_offset_regression(
        bias=10,
        n_features=30,
        proportion_nonzero=proportion_nonzero,
        noise=noise,
        random_state=seed,
        n_samples=n_samples,
    )

    # 手动缩放数据以避免病态情况。在不破坏稀疏模式的情况下使用minmax_scale来处理稀疏案例。
    X = minmax_scale(X)
    # 使用 Ridge 回归模型进行拟合，使用 SVD 方法求解，指定的正则化参数为 alpha
    svd_ridge = Ridge(solver="svd", alpha=alpha).fit(X, y)
    
    # 将特征矩阵 X 和目标向量 y 的数据类型转换为指定的 dtype，且不创建副本
    X = X.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    
    # 如果 sparse_container 参数不为 None，则将特征矩阵 X 转换为稀疏格式，sparse_container 是一个函数或类
    if sparse_container is not None:
        X = sparse_container(X)
    
    # 根据 solver 参数选择合适的 Ridge 回归模型：
    #   - 如果 solver 是 "ridgecv"，则使用 RidgeCV 模型，指定 alpha 参数为 [alpha]
    #   - 否则，使用指定 solver 和 alpha 参数的 Ridge 回归模型，设置公差 tol 为 1e-10
    ridge = RidgeCV(alphas=[alpha]) if solver == "ridgecv" else Ridge(solver=solver, tol=1e-10, alpha=alpha)
    
    # 使用 Ridge 模型对特征矩阵 X 和目标向量 y 进行拟合
    ridge.fit(X, y)
    
    # 断言 ridge 模型的系数（coef_）和 svd_ridge 模型的系数相近，允许的绝对误差为 1e-3，相对误差为 1e-3
    assert_allclose(ridge.coef_, svd_ridge.coef_, atol=1e-3, rtol=1e-3)
    
    # 断言 ridge 模型的截距（intercept_）和 svd_ridge 模型的截距相近，允许的绝对误差为 1e-3，相对误差为 1e-3
    assert_allclose(ridge.intercept_, svd_ridge.intercept_, atol=1e-3, rtol=1e-3)
# 使用 pytest.mark.parametrize 装饰器为 test_ridge_gcv_vs_ridge_loo_cv 函数参数化设定多组测试参数
@pytest.mark.parametrize("gcv_mode", ["svd", "eigen"])
@pytest.mark.parametrize("X_container", [np.asarray] + CSR_CONTAINERS)
@pytest.mark.parametrize("X_shape", [(11, 8), (11, 20)])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "y_shape, noise",
    [
        ((11,), 1.0),     # 定义 y 的形状为 (11,)，噪声为 1.0
        ((11, 1), 30.0),   # 定义 y 的形状为 (11, 1)，噪声为 30.0
        ((11, 3), 150.0),  # 定义 y 的形状为 (11, 3)，噪声为 150.0
    ],
)
# 定义测试函数 test_ridge_gcv_vs_ridge_loo_cv，接收多个参数进行测试
def test_ridge_gcv_vs_ridge_loo_cv(
    gcv_mode, X_container, X_shape, y_shape, fit_intercept, noise
):
    # 从 X_shape 中获取样本数和特征数
    n_samples, n_features = X_shape
    # 从 y_shape 中获取目标数，如果 y_shape 长度为 2 则目标数为最后一个元素，否则为 1
    n_targets = y_shape[-1] if len(y_shape) == 2 else 1
    # 生成稀疏偏移回归数据集 X 和 y
    X, y = _make_sparse_offset_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        shuffle=False,
        noise=noise,
        n_informative=5,
    )
    # 将 y 调整为指定的形状 y_shape
    y = y.reshape(y_shape)

    # 定义岭回归模型的 alpha 值列表
    alphas = [1e-3, 0.1, 1.0, 10.0, 1e3]
    # 创建基于留一法交叉验证的岭回归模型 loo_ridge
    loo_ridge = RidgeCV(
        cv=n_samples,
        fit_intercept=fit_intercept,
        alphas=alphas,
        scoring="neg_mean_squared_error",
    )
    # 创建基于广义交叉验证的岭回归模型 gcv_ridge
    gcv_ridge = RidgeCV(
        gcv_mode=gcv_mode,
        fit_intercept=fit_intercept,
        alphas=alphas,
    )

    # 使用 X, y 对 loo_ridge 进行拟合
    loo_ridge.fit(X, y)

    # 将 X 转换为指定类型 X_container(X) 并对 gcv_ridge 进行拟合
    X_gcv = X_container(X)
    gcv_ridge.fit(X_gcv, y)

    # 断言验证 gcv_ridge 的最优 alpha 与 loo_ridge 的最优 alpha 接近
    assert gcv_ridge.alpha_ == pytest.approx(loo_ridge.alpha_)
    # 断言验证 gcv_ridge 的系数与 loo_ridge 的系数接近
    assert_allclose(gcv_ridge.coef_, loo_ridge.coef_, rtol=1e-3)
    # 断言验证 gcv_ridge 的截距与 loo_ridge 的截距接近
    assert_allclose(gcv_ridge.intercept_, loo_ridge.intercept_, rtol=1e-3)


# 定义测试函数 test_ridge_loo_cv_asym_scoring
def test_ridge_loo_cv_asym_scoring():
    # 检查非对称评分 scoring
    scoring = "explained_variance"
    n_samples, n_features = 10, 5
    n_targets = 1
    # 生成稀疏偏移回归数据集 X 和 y
    X, y = _make_sparse_offset_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        shuffle=False,
        noise=1,
        n_informative=5,
    )

    # 定义岭回归模型的 alpha 值列表
    alphas = [1e-3, 0.1, 1.0, 10.0, 1e3]
    # 创建基于留一法交叉验证的岭回归模型 loo_ridge
    loo_ridge = RidgeCV(
        cv=n_samples, fit_intercept=True, alphas=alphas, scoring=scoring
    )

    # 创建基于广义交叉验证的岭回归模型 gcv_ridge
    gcv_ridge = RidgeCV(fit_intercept=True, alphas=alphas, scoring=scoring)

    # 使用 X, y 对 loo_ridge 进行拟合
    loo_ridge.fit(X, y)
    # 使用 X, y 对 gcv_ridge 进行拟合
    gcv_ridge.fit(X, y)

    # 断言验证 gcv_ridge 的最优 alpha 与 loo_ridge 的最优 alpha 接近
    assert gcv_ridge.alpha_ == pytest.approx(loo_ridge.alpha_)
    # 断言验证 gcv_ridge 的系数与 loo_ridge 的系数接近
    assert_allclose(gcv_ridge.coef_, loo_ridge.coef_, rtol=1e-3)
    # 断言验证 gcv_ridge 的截距与 loo_ridge 的截距接近
    assert_allclose(gcv_ridge.intercept_, loo_ridge.intercept_, rtol=1e-3)


# 使用 pytest.mark.parametrize 装饰器为 test_ridge_gcv_sample_weights 函数参数化设定多组测试参数
@pytest.mark.parametrize("gcv_mode", ["svd", "eigen"])
@pytest.mark.parametrize("X_container", [np.asarray] + CSR_CONTAINERS)
@pytest.mark.parametrize("n_features", [8, 20])
@pytest.mark.parametrize(
    "y_shape, fit_intercept, noise",
    [
        ((11,), True, 1.0),     # 定义 y 的形状为 (11,)，fit_intercept 为 True，噪声为 1.0
        ((11, 1), True, 20.0),  # 定义 y 的形状为 (11, 1)，fit_intercept 为 True，噪声为 20.0
        ((11, 3), True, 150.0), # 定义 y 的形状为 (11, 3)，fit_intercept 为 True，噪声为 150.0
        ((11, 3), False, 30.0), # 定义 y 的形状为 (11, 3)，fit_intercept 为 False，噪声为 30.0
    ],
)
# 定义测试函数 test_ridge_gcv_sample_weights，接收多个参数进行测试
def test_ridge_gcv_sample_weights(
    gcv_mode, X_container, fit_intercept, n_features, y_shape, noise
):
    # 定义岭回归模型的 alpha 值列表
    alphas = [1e-3, 0.1, 1.0, 10.0, 1e3]
    # 使用随机种子为 0 的随机数生成器 rng
    rng = np.random.RandomState(0)
    # 从 y_shape 中获取目标数，如果 y_shape 长度为 2 则目标数为最后一个元素，否则为 1
    n_targets = y_shape[-1] if len(y_shape) == 2 else 1
    # 使用 _make_sparse_offset_regression 函数生成稀疏偏移回归数据集 X 和 y
    X, y = _make_sparse_offset_regression(
        n_samples=11,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        shuffle=False,
        noise=noise,
    )
    # 将 y 重塑为指定形状 y_shape
    y = y.reshape(y_shape)

    # 使用随机数生成器 rng 生成样本权重
    sample_weight = 3 * rng.randn(len(X))
    # 调整样本权重，使其大于等于1，并转换为整数类型
    sample_weight = (sample_weight - sample_weight.min() + 1).astype(int)
    # 根据样本权重创建重复索引数组
    indices = np.repeat(np.arange(X.shape[0]), sample_weight)
    # 将样本权重转换为浮点数类型
    sample_weight = sample_weight.astype(float)
    # 根据重复的索引数组 indices 从 X 和 y 中选择对应的数据创建新的 X_tiled 和 y_tiled
    X_tiled, y_tiled = X[indices], y[indices]

    # 使用 GroupKFold 进行分组交叉验证，将数据集 X_tiled 和 y_tiled 分成 X.shape[0] 组
    cv = GroupKFold(n_splits=X.shape[0])
    splits = cv.split(X_tiled, y_tiled, groups=indices)
    
    # 使用 RidgeCV 进行岭回归交叉验证，设定参数 alphas，评分方式为负均方误差
    kfold = RidgeCV(
        alphas=alphas,
        cv=splits,
        scoring="neg_mean_squared_error",
        fit_intercept=fit_intercept,
    )
    # 在 X_tiled 和 y_tiled 上拟合 RidgeCV 模型
    kfold.fit(X_tiled, y_tiled)

    # 使用交叉验证得到的最佳 alpha 参数创建 Ridge 模型
    ridge_reg = Ridge(alpha=kfold.alpha_, fit_intercept=fit_intercept)
    # 重新使用相同的分组方式进行交叉验证拆分
    splits = cv.split(X_tiled, y_tiled, groups=indices)
    # 使用交叉验证预测 Ridge 模型在 X_tiled 上的 y_tiled
    predictions = cross_val_predict(ridge_reg, X_tiled, y_tiled, cv=splits)
    # 计算交叉验证的误差
    kfold_errors = (y_tiled - predictions) ** 2
    # 将误差按索引分组并求和，创建 kfold_errors 数组
    kfold_errors = [
        np.sum(kfold_errors[indices == i], axis=0) for i in np.arange(X.shape[0])
    ]
    # 将 kfold_errors 转换为 numpy 数组
    kfold_errors = np.asarray(kfold_errors)

    # 使用 X_container 函数封装 X 为 X_gcv
    X_gcv = X_container(X)
    # 使用 RidgeCV 进行广义交叉验证，设定参数 alphas 和 gcv_mode，并保存交叉验证结果
    gcv_ridge = RidgeCV(
        alphas=alphas,
        store_cv_results=True,
        gcv_mode=gcv_mode,
        fit_intercept=fit_intercept,
    )
    # 在 X_gcv 和 y 上使用样本权重 sample_weight 拟合广义交叉验证 RidgeCV 模型
    gcv_ridge.fit(X_gcv, y, sample_weight=sample_weight)
    # 根据 y_shape 的维度选择正确的 gcv_errors
    if len(y_shape) == 2:
        gcv_errors = gcv_ridge.cv_results_[:, :, alphas.index(kfold.alpha_)]
    else:
        gcv_errors = gcv_ridge.cv_results_[:, alphas.index(kfold.alpha_)]

    # 断言交叉验证得到的最佳 alpha 与广义交叉验证得到的最佳 alpha 接近
    assert kfold.alpha_ == pytest.approx(gcv_ridge.alpha_)
    # 断言 gcv_errors 与 kfold_errors 接近，相对误差小于 1e-3
    assert_allclose(gcv_errors, kfold_errors, rtol=1e-3)
    # 断言 gcv_ridge 模型的系数与 kfold 模型的系数接近，相对误差小于 1e-3
    assert_allclose(gcv_ridge.coef_, kfold.coef_, rtol=1e-3)
    # 断言 gcv_ridge 模型的截距与 kfold 模型的截距接近，相对误差小于 1e-3
    assert_allclose(gcv_ridge.intercept_, kfold.intercept_, rtol=1e-3)
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
# 使用 pytest.mark.parametrize 装饰器，对 sparse_container 参数进行参数化测试，包括 None 和 CSR_CONTAINERS 中的内容

@pytest.mark.parametrize(
    "mode, mode_n_greater_than_p, mode_p_greater_than_n",
    [
        (None, "svd", "eigen"),
        ("auto", "svd", "eigen"),
        ("eigen", "eigen", "eigen"),
        ("svd", "svd", "svd"),
    ],
)
# 使用 pytest.mark.parametrize 装饰器，对 mode, mode_n_greater_than_p, mode_p_greater_than_n 参数进行参数化测试，设置了不同的模式组合

def test_check_gcv_mode_choice(
    sparse_container, mode, mode_n_greater_than_p, mode_p_greater_than_n
):
    # 使用 make_regression 生成具有 5 个样本和 2 个特征的数据集 X 和 _
    X, _ = make_regression(n_samples=5, n_features=2)

    if sparse_container is not None:
        # 如果 sparse_container 不为 None，则将 X 转换为稀疏格式
        X = sparse_container(X)

    # 断言 _check_gcv_mode 函数对 X 和 X 的转置的返回值与预期的 mode_n_greater_than_p 和 mode_p_greater_than_n 相等
    assert _check_gcv_mode(X, mode) == mode_n_greater_than_p
    assert _check_gcv_mode(X.T, mode) == mode_p_greater_than_n


def _test_ridge_loo(sparse_container):
    # 测试能够同时处理密集或稀疏矩阵

    n_samples = X_diabetes.shape[0]

    ret = []

    if sparse_container is None:
        # 如果 sparse_container 是 None，则使用 X_diabetes 和 fit_intercept=True
        X, fit_intercept = X_diabetes, True
    else:
        # 否则，使用 sparse_container 转换后的 X_diabetes 和 fit_intercept=False
        X, fit_intercept = sparse_container(X_diabetes), False

    # 创建一个 _RidgeGCV 对象
    ridge_gcv = _RidgeGCV(fit_intercept=fit_intercept)

    # 拟合 ridge_gcv 对象，得到最佳 alpha 值
    ridge_gcv.fit(X, y_diabetes)
    alpha_ = ridge_gcv.alpha_
    ret.append(alpha_)

    # 使用自定义 loss_func 检查是否获得相同的最佳 alpha 值
    f = ignore_warnings
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    ridge_gcv2 = RidgeCV(fit_intercept=False, scoring=scoring)
    f(ridge_gcv2.fit)(X, y_diabetes)
    assert ridge_gcv2.alpha_ == pytest.approx(alpha_)

    # 使用自定义 score_func 检查是否获得相同的最佳 alpha 值
    def func(x, y):
        return -mean_squared_error(x, y)

    scoring = make_scorer(func)
    ridge_gcv3 = RidgeCV(fit_intercept=False, scoring=scoring)
    f(ridge_gcv3.fit)(X, y_diabetes)
    assert ridge_gcv3.alpha_ == pytest.approx(alpha_)

    # 使用 scorer 检查是否获得相同的最佳 alpha 值
    scorer = get_scorer("neg_mean_squared_error")
    ridge_gcv4 = RidgeCV(fit_intercept=False, scoring=scorer)
    ridge_gcv4.fit(X, y_diabetes)
    assert ridge_gcv4.alpha_ == pytest.approx(alpha_)

    # 如果 sparse_container 是 None，则使用样本权重为 np.ones(n_samples) 来拟合 ridge_gcv 对象，并检查是否获得相同的最佳 alpha 值
    if sparse_container is None:
        ridge_gcv.fit(X, y_diabetes, sample_weight=np.ones(n_samples))
        assert ridge_gcv.alpha_ == pytest.approx(alpha_)

    # 模拟多个响应值
    Y = np.vstack((y_diabetes, y_diabetes)).T

    # 拟合 ridge_gcv 对象，并预测 Y 值
    ridge_gcv.fit(X, Y)
    Y_pred = ridge_gcv.predict(X)
    ridge_gcv.fit(X, y_diabetes)
    y_pred = ridge_gcv.predict(X)

    # 断言 np.vstack((y_pred, y_pred)).T 与 Y_pred 在指定的相对误差下是否相等
    assert_allclose(np.vstack((y_pred, y_pred)).T, Y_pred, rtol=1e-5)

    return ret


def _test_ridge_cv(sparse_container):
    # 如果 sparse_container 是 None，则使用 X_diabetes；否则，使用 sparse_container 转换后的 X_diabetes
    X = X_diabetes if sparse_container is None else sparse_container(X_diabetes)

    # 创建 RidgeCV 对象
    ridge_cv = RidgeCV()

    # 拟合 ridge_cv 对象，并预测 X 上的值
    ridge_cv.fit(X, y_diabetes)
    ridge_cv.predict(X)

    # 断言 ridge_cv.coef_ 的形状为一维数组
    assert len(ridge_cv.coef_.shape) == 1
    # 断言 ridge_cv.intercept_ 的类型为 np.float64
    assert type(ridge_cv.intercept_) == np.float64

    # 创建 KFold 对象，设置 5 折交叉验证
    cv = KFold(5)
    ridge_cv.set_params(cv=cv)

    # 拟合 ridge_cv 对象，并预测 X 上的值
    ridge_cv.fit(X, y_diabetes)
    ridge_cv.predict(X)

    # 断言 ridge_cv.coef_ 的形状为一维数组
    assert len(ridge_cv.coef_.shape) == 1
    # 使用断言检查 ridge_cv.intercept_ 的类型是否为 np.float64
    assert type(ridge_cv.intercept_) == np.float64
@pytest.mark.parametrize(
    "ridge, make_dataset",
    [
        (RidgeCV(store_cv_results=False), make_regression),
        (RidgeClassifierCV(store_cv_results=False), make_classification),
    ],
)
# 定义测试函数 test_ridge_gcv_cv_results_not_stored，参数化测试岭回归器和数据生成函数
def test_ridge_gcv_cv_results_not_stored(ridge, make_dataset):
    # 检查当 store_cv_results 参数为 False 时，不存储 cv_results_
    X, y = make_dataset(n_samples=6, random_state=42)
    # 使用 make_dataset 生成数据集 X 和标签 y
    ridge.fit(X, y)
    # 使用岭回归器 ridge 拟合数据
    assert not hasattr(ridge, "cv_results_")
    # 断言岭回归器 ridge 没有 cv_results_ 属性


@pytest.mark.parametrize(
    "ridge, make_dataset",
    [(RidgeCV(), make_regression), (RidgeClassifierCV(), make_classification)],
)
@pytest.mark.parametrize("cv", [None, 3])
# 参数化测试函数 test_ridge_best_score，岭回归器和数据生成函数，以及交叉验证折数
def test_ridge_best_score(ridge, make_dataset, cv):
    # 检查 best_score_ 被正确存储
    X, y = make_dataset(n_samples=6, random_state=42)
    # 使用 make_dataset 生成数据集 X 和标签 y
    ridge.set_params(store_cv_results=False, cv=cv)
    # 设置岭回归器 ridge 的参数 store_cv_results 和 cv
    ridge.fit(X, y)
    # 使用岭回归器 ridge 拟合数据
    assert hasattr(ridge, "best_score_")
    # 断言岭回归器 ridge 有 best_score_ 属性
    assert isinstance(ridge.best_score_, float)
    # 断言 best_score_ 是浮点数类型


def test_ridge_cv_individual_penalties():
    # 测试 ridge_cv 对象在优化每个目标的个别惩罚时的表现

    rng = np.random.RandomState(42)

    # 创建具有多个目标的随机数据集。每个目标应具有不同的最优 alpha 值。
    n_samples, n_features, n_targets = 20, 5, 3
    y = rng.randn(n_samples, n_targets)
    X = (
        np.dot(y[:, [0]], np.ones((1, n_features)))
        + np.dot(y[:, [1]], 0.05 * np.ones((1, n_features)))
        + np.dot(y[:, [2]], 0.001 * np.ones((1, n_features)))
        + rng.randn(n_samples, n_features)
    )

    alphas = (1, 100, 1000)

    # 找到每个目标的最优 alpha
    optimal_alphas = [RidgeCV(alphas=alphas).fit(X, target).alpha_ for target in y.T]

    # 同时为所有目标找到最优 alpha
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True).fit(X, y)
    assert_array_equal(optimal_alphas, ridge_cv.alpha_)

    # 结果的回归权重应包含不同的 alpha 值。
    assert_array_almost_equal(
        Ridge(alpha=ridge_cv.alpha_).fit(X, y).coef_, ridge_cv.coef_
    )

    # 测试 alpha_ 和 cv_results_ 的形状
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, store_cv_results=True).fit(
        X, y
    )
    assert ridge_cv.alpha_.shape == (n_targets,)
    assert ridge_cv.best_score_.shape == (n_targets,)
    assert ridge_cv.cv_results_.shape == (n_samples, len(alphas), n_targets)

    # 测试只有一个 alpha 值的边缘情况
    ridge_cv = RidgeCV(alphas=1, alpha_per_target=True, store_cv_results=True).fit(X, y)
    assert ridge_cv.alpha_.shape == (n_targets,)
    assert ridge_cv.best_score_.shape == (n_targets,)
    assert ridge_cv.cv_results_.shape == (n_samples, n_targets, 1)

    # 测试只有一个目标的边缘情况
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, store_cv_results=True).fit(
        X, y[:, 0]
    )
    assert np.isscalar(ridge_cv.alpha_)
    assert np.isscalar(ridge_cv.best_score_)
    # 断言岭回归交叉验证结果的形状是否符合预期
    assert ridge_cv.cv_results_.shape == (n_samples, len(alphas))

    # 使用自定义评分函数进行岭回归交叉验证
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, scoring="r2").fit(X, y)
    # 断言最优的 alpha 值是否与预期相等
    assert_array_equal(optimal_alphas, ridge_cv.alpha_)
    # 断言使用最优 alpha 值训练的岭回归模型的系数是否与交叉验证得到的系数相近
    assert_array_almost_equal(
        Ridge(alpha=ridge_cv.alpha_).fit(X, y).coef_, ridge_cv.coef_
    )

    # 使用自定义的交叉验证对象和 alpha_per_target=True 会抛出错误
    ridge_cv = RidgeCV(alphas=alphas, cv=LeaveOneOut(), alpha_per_target=True)
    msg = "cv!=None and alpha_per_target=True are incompatible"
    with pytest.raises(ValueError, match=msg):
        ridge_cv.fit(X, y)
    # 使用自定义的交叉验证折数和 alpha_per_target=True 会抛出错误
    ridge_cv = RidgeCV(alphas=alphas, cv=6, alpha_per_target=True)
    with pytest.raises(ValueError, match=msg):
        ridge_cv.fit(X, y)
# 根据是否提供稀疏数据容器，选择性地使用稀疏数据表示或默认的数据表示 X_diabetes
def _test_ridge_diabetes(sparse_container):
    X = X_diabetes if sparse_container is None else sparse_container(X_diabetes)
    # 创建 Ridge 回归器，不包含截距项
    ridge = Ridge(fit_intercept=False)
    # 使用 X 和 y_diabetes 训练 Ridge 回归器
    ridge.fit(X, y_diabetes)
    # 计算并返回模型的 R^2 分数，保留小数点后五位
    return np.round(ridge.score(X, y_diabetes), 5)


def _test_multi_ridge_diabetes(sparse_container):
    # 模拟多个响应
    X = X_diabetes if sparse_container is None else sparse_container(X_diabetes)
    Y = np.vstack((y_diabetes, y_diabetes)).T
    n_features = X_diabetes.shape[1]

    ridge = Ridge(fit_intercept=False)
    ridge.fit(X, Y)
    # 断言 Ridge 回归器的系数形状为 (2, n_features)
    assert ridge.coef_.shape == (2, n_features)
    # 使用 X 预测 Y，并检查预测值
    Y_pred = ridge.predict(X)
    ridge.fit(X, y_diabetes)
    y_pred = ridge.predict(X)
    # 断言预测结果与多响应预测结果 Y_pred 相近，精确度为 0.003
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)


def _test_ridge_classifiers(sparse_container):
    n_classes = np.unique(y_iris).shape[0]
    n_features = X_iris.shape[1]
    X = X_iris if sparse_container is None else sparse_container(X_iris)

    # 遍历 Ridge 分类器及其交叉验证版本
    for reg in (RidgeClassifier(), RidgeClassifierCV()):
        reg.fit(X, y_iris)
        # 断言 Ridge 分类器的系数形状为 (n_classes, n_features)
        assert reg.coef_.shape == (n_classes, n_features)
        # 预测并检查准确率是否大于 0.79
        y_pred = reg.predict(X)
        assert np.mean(y_iris == y_pred) > 0.79

    cv = KFold(5)
    reg = RidgeClassifierCV(cv=cv)
    reg.fit(X, y_iris)
    y_pred = reg.predict(X)
    # 断言使用交叉验证后的准确率是否大于等于 0.8
    assert np.mean(y_iris == y_pred) >= 0.8


@pytest.mark.parametrize("scoring", [None, "accuracy", _accuracy_callable])
@pytest.mark.parametrize("cv", [None, KFold(5)])
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
def test_ridge_classifier_with_scoring(sparse_container, scoring, cv):
    # 非回归测试用例，用于检查 RidgeClassifierCV 在各种评分和交叉验证下的工作情况
    X = X_iris if sparse_container is None else sparse_container(X_iris)
    scoring_ = make_scorer(scoring) if callable(scoring) else scoring
    clf = RidgeClassifierCV(scoring=scoring_, cv=cv)
    # 简单测试确保拟合和预测没有引发错误
    clf.fit(X, y_iris).predict(X)


@pytest.mark.parametrize("cv", [None, KFold(5)])
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
def test_ridge_regression_custom_scoring(sparse_container, cv):
    # 检查自定义评分功能是否按预期工作
    # 检查打破平局策略（保留第一个尝试的 alpha）
    
    def _dummy_score(y_test, y_pred, **kwargs):
        return 0.42

    X = X_iris if sparse_container is None else sparse_container(X_iris)
    alphas = np.logspace(-2, 2, num=5)
    clf = RidgeClassifierCV(alphas=alphas, scoring=make_scorer(_dummy_score), cv=cv)
    clf.fit(X, y_iris)
    # 断言最佳得分近似为 0.42
    assert clf.best_score_ == pytest.approx(0.42)
    # 在平局的情况下，应保留第一个尝试的 alpha
    assert clf.alpha_ == pytest.approx(alphas[0])


def _test_tolerance(sparse_container):
    X = X_diabetes if sparse_container is None else sparse_container(X_diabetes)

    # 创建 Ridge 回归器，设置容差为 1e-5，不包含截距项
    ridge = Ridge(tol=1e-5, fit_intercept=False)
    # 使用 X 和 y_diabetes 训练 Ridge 回归器
    ridge.fit(X, y_diabetes)
    # 计算岭回归模型在数据集 X 和目标变量 y_diabetes 上的得分
    score = ridge.score(X, y_diabetes)

    # 创建一个新的岭回归模型对象 ridge2，并设置参数 tol=1e-3 和 fit_intercept=False
    ridge2 = Ridge(tol=1e-3, fit_intercept=False)
    # 使用数据集 X 和目标变量 y_diabetes 训练 ridge2 模型
    ridge2.fit(X, y_diabetes)
    # 计算 ridge2 模型在数据集 X 和目标变量 y_diabetes 上的得分
    score2 = ridge2.score(X, y_diabetes)

    # 断言第一个岭回归模型的得分大于等于第二个岭回归模型的得分
    assert score >= score2
# 检查数组 API 的属性，用于测试不同的数组命名空间、设备和数据类型
def check_array_api_attributes(name, estimator, array_namespace, device, dtype_name):
    # 根据数组命名空间和设备获取相应的数组 API
    xp = _array_api_for_tests(array_namespace, device)

    # 将 X_iris 转换为指定数据类型的 NumPy 数组
    X_iris_np = X_iris.astype(dtype_name)
    # 将 y_iris 转换为指定数据类型的 NumPy 数组
    y_iris_np = y_iris.astype(dtype_name)

    # 使用数组 API 将 NumPy 数组 X_iris_np 转换为指定设备上的数组
    X_iris_xp = xp.asarray(X_iris_np, device=device)
    # 使用数组 API 将 NumPy 数组 y_iris_np 转换为指定设备上的数组
    y_iris_xp = xp.asarray(y_iris_np, device=device)

    # 使用 NumPy 数组拟合估计器
    estimator.fit(X_iris_np, y_iris_np)
    # 获取 NumPy 数组形式的系数
    coef_np = estimator.coef_
    # 获取 NumPy 数组形式的截距
    intercept_np = estimator.intercept_

    # 切换配置上下文，使用数组 API 进行派发
    with config_context(array_api_dispatch=True):
        # 克隆估计器并使用数组 API 训练
        estimator_xp = clone(estimator).fit(X_iris_xp, y_iris_xp)
        # 获取数组 API 形式的系数
        coef_xp = estimator_xp.coef_
        # 断言系数的形状为 (4,)
        assert coef_xp.shape == (4,)
        # 断言系数的数据类型与 X_iris_xp 相同
        assert coef_xp.dtype == X_iris_xp.dtype

        # 使用 assert_allclose 检查数组 API 形式的系数与 NumPy 形式的系数的近似程度
        assert_allclose(
            _convert_to_numpy(coef_xp, xp=xp),
            coef_np,
            atol=_atol_for_type(dtype_name),
        )
        # 获取数组 API 形式的截距
        intercept_xp = estimator_xp.intercept_
        # 断言截距的形状为空元组
        assert intercept_xp.shape == ()
        # 断言截距的数据类型与 X_iris_xp 相同
        assert intercept_xp.dtype == X_iris_xp.dtype

        # 使用 assert_allclose 检查数组 API 形式的截距与 NumPy 形式的截距的近似程度
        assert_allclose(
            _convert_to_numpy(intercept_xp, xp=xp),
            intercept_np,
            atol=_atol_for_type(dtype_name),
        )


# 使用参数化测试运行多个数组命名空间、设备和数据类型的 Ridge 回归器 API 一致性测试
@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
@pytest.mark.parametrize(
    "check",
    [check_array_api_input_and_values, check_array_api_attributes],
    ids=_get_check_estimator_ids,
)
@pytest.mark.parametrize(
    "estimator",
    [Ridge(solver="svd")],
    ids=_get_check_estimator_ids,
)
def test_ridge_array_api_compliance(
    estimator, check, array_namespace, device, dtype_name
):
    # 获取估计器的类名
    name = estimator.__class__.__name__
    # 执行指定的数组 API 属性检查函数
    check(name, estimator, array_namespace, device=device, dtype_name=dtype_name)


# 使用参数化测试运行多个数组命名空间的错误和警告测试，排除 NumPy 命名空间
@pytest.mark.parametrize(
    "array_namespace", yield_namespaces(include_numpy_namespaces=False)
)
def test_array_api_error_and_warnings_for_solver_parameter(array_namespace):
    # 根据数组命名空间和设备获取相应的数组 API
    xp = _array_api_for_tests(array_namespace, device=None)

    # 使用数组 API 将前5个样本的 X_iris 转换为指定设备上的数组
    X_iris_xp = xp.asarray(X_iris[:5])
    # 使用数组 API 将前5个样本的 y_iris 转换为指定设备上的数组
    y_iris_xp = xp.asarray(y_iris[:5])

    # 获取 Ridge 回归器支持的可用求解器
    available_solvers = Ridge._parameter_constraints["solver"][0].options
    # 遍历除了 "auto" 和 "svd" 外的所有求解器
    for solver in available_solvers - {"auto", "svd"}:
        # 根据求解器创建 Ridge 对象
        ridge = Ridge(solver=solver, positive=solver == "lbfgs")
        # 构建预期的错误消息
        expected_msg = (
            f"Array API dispatch to namespace {xp.__name__} only supports "
            f"solver 'svd'. Got '{solver}'."
        )

        # 使用 pytest.raises 断言捕获预期的 ValueError 异常并检查错误消息
        with pytest.raises(ValueError, match=expected_msg):
            with config_context(array_api_dispatch=True):
                ridge.fit(X_iris_xp, y_iris_xp)

    # 使用 "auto" 求解器和 positive=True 创建 Ridge 对象
    ridge = Ridge(solver="auto", positive=True)
    # 构建预期的错误消息
    expected_msg = (
        "The solvers that support positive fitting do not support "
        f"Array API dispatch to namespace {xp.__name__}. Please "
        "either disable Array API dispatch, or use a numpy-like "
        "namespace, or set `positive=False`."
    )
    # 使用 pytest 模块验证是否引发了特定的 ValueError 异常，并且异常消息与 expected_msg 匹配
    with pytest.raises(ValueError, match=expected_msg):
        # 进入一个配置上下文，设置 array_api_dispatch 为 True
        with config_context(array_api_dispatch=True):
            # 使用 Ridge 回归器拟合数据 X_iris_xp 和标签 y_iris_xp
            ridge.fit(X_iris_xp, y_iris_xp)
    
    # 创建 Ridge 回归器的实例
    ridge = Ridge()
    # 设置期望的警告消息字符串，包含特定的 xp 模块名作为命名空间
    expected_msg = (
        f"Using Array API dispatch to namespace {xp.__name__} with `solver='auto'` "
        "will result in using the solver 'svd'. The results may differ from those "
        "when using a Numpy array, because in that case the preferred solver would "
        "be cholesky. Set `solver='svd'` to suppress this warning."
    )
    # 使用 pytest 模块验证是否引发了 UserWarning 警告，并且警告消息与 expected_msg 匹配
    with pytest.warns(UserWarning, match=expected_msg):
        # 进入一个配置上下文，设置 array_api_dispatch 为 True
        with config_context(array_api_dispatch=True):
            # 使用 Ridge 回归器拟合数据 X_iris_xp 和标签 y_iris_xp
            ridge.fit(X_iris_xp, y_iris_xp)
@pytest.mark.parametrize("array_namespace", sorted(_NUMPY_NAMESPACE_NAMES))
# 使用 pytest 的 parametrize 装饰器，遍历 _NUMPY_NAMESPACE_NAMES 列表中的各个命名空间
def test_array_api_numpy_namespace_no_warning(array_namespace):
    # 调用 _array_api_for_tests 函数，返回特定命名空间的 Array API 对象 xp
    xp = _array_api_for_tests(array_namespace, device=None)

    # 将前五行 X_iris 转换为 xp 命名空间的数组
    X_iris_xp = xp.asarray(X_iris[:5])
    # 将前五行 y_iris 转换为 xp 命名空间的数组
    y_iris_xp = xp.asarray(y_iris[:5])

    # 创建 Ridge 回归对象
    ridge = Ridge()
    # 设置预期的警告信息
    expected_msg = (
        "Results might be different than when Array API dispatch is "
        "disabled, or when a numpy-like namespace is used"
    )

    # 捕获特定警告信息，并且在警告类别为 UserWarning 时抛出错误
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=expected_msg, category=UserWarning)
        # 使用 array_api_dispatch 上下文，开启 Array API 调度
        with config_context(array_api_dispatch=True):
            # 使用 xp 命名空间进行 Ridge 拟合
            ridge.fit(X_iris_xp, y_iris_xp)

    # 所有的 numpy 命名空间都兼容所有的求解器，特别是支持 positive=True 的求解器（如 'lbfgs'）应该可以工作
    with config_context(array_api_dispatch=True):
        Ridge(solver="auto", positive=True).fit(X_iris_xp, y_iris_xp)


@pytest.mark.parametrize(
    "test_func",
    (
        _test_ridge_loo,
        _test_ridge_cv,
        _test_ridge_diabetes,
        _test_multi_ridge_diabetes,
        _test_ridge_classifiers,
        _test_tolerance,
    ),
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，遍历不同的测试函数和 CSR 容器
def test_dense_sparse(test_func, csr_container):
    # 测试稠密矩阵
    ret_dense = test_func(None)
    # 测试稀疏矩阵
    ret_sparse = test_func(csr_container)
    # 检查输出是否相同
    if ret_dense is not None and ret_sparse is not None:
        assert_array_almost_equal(ret_dense, ret_sparse, decimal=3)


def test_class_weights():
    # 测试类别权重
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    # 使用默认的类别权重创建 Ridge 分类器
    reg = RidgeClassifier(class_weight=None)
    reg.fit(X, y)
    # 断言预测结果与预期值相等
    assert_array_equal(reg.predict([[0.2, -1.0]]), np.array([1]))

    # 给类别 1 分配小的权重
    reg = RidgeClassifier(class_weight={1: 0.001})
    reg.fit(X, y)

    # 现在超平面应该顺时针旋转，预测结果应该发生变化
    assert_array_equal(reg.predict([[0.2, -1.0]]), np.array([-1]))

    # 检查 class_weight = 'balanced' 是否可以处理负标签
    reg = RidgeClassifier(class_weight="balanced")
    reg.fit(X, y)
    assert_array_equal(reg.predict([[0.2, -1.0]]), np.array([1]))

    # 当 y 中具有相等数量的所有标签时，class_weight = 'balanced' 和 class_weight = None 应该返回相同的值
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0]])
    y = [1, 1, -1, -1]
    reg = RidgeClassifier(class_weight=None)
    reg.fit(X, y)
    rega = RidgeClassifier(class_weight="balanced")
    rega.fit(X, y)
    assert len(rega.classes_) == 2
    assert_array_almost_equal(reg.coef_, rega.coef_)
    assert_array_almost_equal(reg.intercept_, rega.intercept_)


@pytest.mark.parametrize("reg", (RidgeClassifier, RidgeClassifierCV))
# 使用 pytest 的 parametrize 装饰器，遍历 RidgeClassifier 和 RidgeClassifierCV 类型
def test_class_weight_vs_sample_weight(reg):
    # 检查 class_weights 是否类似于 sample_weights 的行为。
    
    # 创建一个没有使用 class_weight 参数的默认回归器，并用 Iris 数据训练它
    reg1 = reg()
    reg1.fit(iris.data, iris.target)
    
    # 创建一个使用 'balanced' 权重参数的回归器，预期对于平衡的 Iris 数据没有影响
    reg2 = reg(class_weight="balanced")
    reg2.fit(iris.data, iris.target)
    
    # 断言两个回归器的系数近似相等
    assert_almost_equal(reg1.coef_, reg2.coef_)
    
    # 增加类别 1 的重要性，用样本权重（sample_weight）调整，然后与用户定义的类别权重对比
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1.0, 1: 100.0, 2: 1.0}
    
    # 创建一个没有使用权重的回归器，并用样本权重训练它
    reg1 = reg()
    reg1.fit(iris.data, iris.target, sample_weight)
    
    # 创建一个使用用户定义的类别权重参数的回归器
    reg2 = reg(class_weight=class_weight)
    reg2.fit(iris.data, iris.target)
    
    # 断言两个回归器的系数近似相等
    assert_almost_equal(reg1.coef_, reg2.coef_)
    
    # 检查样本权重和类别权重是否是乘法关系
    # 创建一个没有使用权重的回归器，并用样本权重的平方训练它
    reg1 = reg()
    reg1.fit(iris.data, iris.target, sample_weight**2)
    
    # 创建一个使用用户定义的类别权重参数的回归器，并用样本权重训练它
    reg2 = reg(class_weight=class_weight)
    reg2.fit(iris.data, iris.target, sample_weight)
    
    # 断言两个回归器的系数近似相等
    assert_almost_equal(reg1.coef_, reg2.coef_)
def test_class_weights_cv():
    # 测试交叉验证岭回归分类器的类别权重

    # 创建特征矩阵 X 和标签向量 y
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    # 使用默认参数创建岭回归分类器，不使用类别权重
    reg = RidgeClassifierCV(class_weight=None, alphas=[0.01, 0.1, 1])
    reg.fit(X, y)

    # 给类别 1 设置一个较小的权重
    reg = RidgeClassifierCV(class_weight={1: 0.001}, alphas=[0.01, 0.1, 1, 10])
    reg.fit(X, y)

    # 断言预测结果为 [-1]
    assert_array_equal(reg.predict([[-0.2, 2]]), np.array([-1]))


@pytest.mark.parametrize(
    "scoring", [None, "neg_mean_squared_error", _mean_squared_error_callable]
)
def test_ridgecv_store_cv_results(scoring):
    # 测试岭回归交叉验证存储 CV 结果

    # 创建随机数生成器和样本数据
    rng = np.random.RandomState(42)
    n_samples = 8
    n_features = 5
    x = rng.randn(n_samples, n_features)
    alphas = [1e-1, 1e0, 1e1]
    n_alphas = len(alphas)

    # 根据 scoring 参数创建评分器 scoring_
    scoring_ = make_scorer(scoring) if callable(scoring) else scoring

    # 创建 RidgeCV 对象
    r = RidgeCV(alphas=alphas, cv=None, store_cv_results=True, scoring=scoring_)

    # 对于 y 是一维数组的情况
    y = rng.randn(n_samples)
    r.fit(x, y)
    assert r.cv_results_.shape == (n_samples, n_alphas)

    # 对于 y 是二维数组的情况
    n_targets = 3
    y = rng.randn(n_samples, n_targets)
    r.fit(x, y)
    assert r.cv_results_.shape == (n_samples, n_targets, n_alphas)

    # 测试带有指定 CV 参数的 RidgeCV 对象
    r = RidgeCV(cv=3, store_cv_results=True, scoring=scoring)
    with pytest.raises(ValueError, match="cv!=None and store_cv_results"):
        r.fit(x, y)


@pytest.mark.parametrize("scoring", [None, "accuracy", _accuracy_callable])
def test_ridge_classifier_cv_store_cv_results(scoring):
    # 测试岭回归分类器交叉验证存储 CV 结果

    # 创建特征矩阵 X 和标签向量 y
    x = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 1, 1, -1, -1])

    n_samples = x.shape[0]
    alphas = [1e-1, 1e0, 1e1]
    n_alphas = len(alphas)

    # 根据 scoring 参数创建评分器 scoring_
    scoring_ = make_scorer(scoring) if callable(scoring) else scoring

    # 创建 RidgeClassifierCV 对象
    r = RidgeClassifierCV(
        alphas=alphas, cv=None, store_cv_results=True, scoring=scoring_
    )

    # 对于 y 是一维数组的情况
    n_targets = 1
    r.fit(x, y)
    assert r.cv_results_.shape == (n_samples, n_targets, n_alphas)

    # 对于 y 是二维数组的情况
    y = np.array(
        [[1, 1, 1, -1, -1], [1, -1, 1, -1, 1], [-1, -1, 1, -1, -1]]
    ).transpose()
    n_targets = y.shape[1]
    r.fit(x, y)
    assert r.cv_results_.shape == (n_samples, n_targets, n_alphas)


@pytest.mark.parametrize("Estimator", [RidgeCV, RidgeClassifierCV])
def test_ridgecv_alphas_conversion(Estimator):
    # 测试岭回归和岭回归分类器的 alphas 参数转换

    # 创建随机数生成器和样本数据
    rng = np.random.RandomState(0)
    alphas = (0.1, 1.0, 10.0)

    n_samples, n_features = 5, 5
    if Estimator is RidgeCV:
        y = rng.randn(n_samples)
    else:
        y = rng.randint(0, 2, n_samples)
    X = rng.randn(n_samples, n_features)

    # 创建 Estimator 对象
    ridge_est = Estimator(alphas=alphas)
    assert (
        ridge_est.alphas is alphas
    ), f"`alphas` was mutated in `{Estimator.__name__}.__init__`"

    # 拟合数据并断言 alphas 参数的一致性
    ridge_est.fit(X, y)
    assert_array_equal(ridge_est.alphas, np.asarray(alphas))


@pytest.mark.parametrize("cv", [None, 3])
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_ridgecv_alphas_zero
@pytest.mark.parametrize("Estimator", [RidgeCV, RidgeClassifierCV])
def test_ridgecv_alphas_zero(cv, Estimator):
    """Check alpha=0.0 raises error only when `cv=None`."""
    # 设定随机数生成器种子
    rng = np.random.RandomState(0)
    # 设定不同的 alpha 值用于测试
    alphas = (0.0, 1.0, 10.0)

    # 设定样本数和特征数
    n_samples, n_features = 5, 5
    # 根据不同的 Estimator 选择生成目标 y
    if Estimator is RidgeCV:
        y = rng.randn(n_samples)
    else:
        y = rng.randint(0, 2, n_samples)
    # 生成输入特征矩阵 X
    X = rng.randn(n_samples, n_features)

    # 创建 RidgeCV 或 RidgeClassifierCV 对象
    ridge_est = Estimator(alphas=alphas, cv=cv)
    # 如果 cv 为 None，检查是否抛出 ValueError 异常
    if cv is None:
        with pytest.raises(ValueError, match=r"alphas\[0\] == 0.0, must be > 0.0."):
            ridge_est.fit(X, y)
    else:
        ridge_est.fit(X, y)


# 定义测试函数 test_ridgecv_sample_weight
def test_ridgecv_sample_weight():
    # 设定随机数生成器种子
    rng = np.random.RandomState(0)
    # 设定不同的 alpha 值用于测试
    alphas = (0.1, 1.0, 10.0)

    # 遍历不同的样本数和特征数组合
    for n_samples, n_features in ((6, 5), (5, 10)):
        # 生成目标 y
        y = rng.randn(n_samples)
        # 生成输入特征矩阵 X
        X = rng.randn(n_samples, n_features)
        # 生成样本权重
        sample_weight = 1.0 + rng.rand(n_samples)

        # 创建 KFold 交叉验证对象
        cv = KFold(5)
        # 创建 RidgeCV 对象
        ridgecv = RidgeCV(alphas=alphas, cv=cv)
        # 使用样本权重拟合模型
        ridgecv.fit(X, y, sample_weight=sample_weight)

        # 使用 GridSearchCV 直接检查
        parameters = {"alpha": alphas}
        gs = GridSearchCV(Ridge(), parameters, cv=cv)
        gs.fit(X, y, sample_weight=sample_weight)

        # 断言 RidgeCV 得到的最佳 alpha 与 GridSearchCV 得到的最佳 alpha 相等
        assert ridgecv.alpha_ == gs.best_estimator_.alpha
        # 断言 RidgeCV 得到的系数与 GridSearchCV 得到的系数近似相等
        assert_array_almost_equal(ridgecv.coef_, gs.best_estimator_.coef_)


# 定义测试函数 test_raises_value_error_if_sample_weights_greater_than_1d
def test_raises_value_error_if_sample_weights_greater_than_1d():
    # 样本权重必须是标量或一维数组

    # 设定不同的样本数和特征数列表
    n_sampless = [2, 3]
    n_featuress = [3, 2]

    # 设定随机数生成器种子
    rng = np.random.RandomState(42)

    # 遍历不同的样本数和特征数组合
    for n_samples, n_features in zip(n_sampless, n_featuress):
        # 生成输入特征矩阵 X
        X = rng.randn(n_samples, n_features)
        # 生成目标 y
        y = rng.randn(n_samples)
        # 生成符合条件的样本权重
        sample_weights_OK = rng.randn(n_samples) ** 2 + 1
        sample_weights_OK_1 = 1.0
        sample_weights_OK_2 = 2.0
        # 生成不符合条件的样本权重
        sample_weights_not_OK = sample_weights_OK[:, np.newaxis]
        sample_weights_not_OK_2 = sample_weights_OK[np.newaxis, :]

        # 创建 Ridge 对象
        ridge = Ridge(alpha=1)

        # 确保符合条件的样本权重可以正常工作
        ridge.fit(X, y, sample_weights_OK)
        ridge.fit(X, y, sample_weights_OK_1)
        ridge.fit(X, y, sample_weights_OK_2)

        # 定义不能工作的 Ridge 拟合函数
        def fit_ridge_not_ok():
            ridge.fit(X, y, sample_weights_not_OK)

        def fit_ridge_not_ok_2():
            ridge.fit(X, y, sample_weights_not_OK_2)

        # 检查是否抛出 ValueError 异常
        err_msg = "Sample weights must be 1D array or scalar"
        with pytest.raises(ValueError, match=err_msg):
            fit_ridge_not_ok()

        with pytest.raises(ValueError, match=err_msg):
            fit_ridge_not_ok_2()


# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_ridgecv_alphas_zero
@pytest.mark.parametrize("n_samples,n_features", [[2, 3], [3, 2]])
@pytest.mark.parametrize(
    "sparse_container",
    COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS,
)
def test_sparse_design_with_sample_weights(n_samples, n_features, sparse_container):
    # Sample weights must work with sparse matrices

    # 使用种子 42 初始化随机数生成器
    rng = np.random.RandomState(42)

    # 创建稀疏矩阵下的 Ridge 回归模型和密集矩阵下的 Ridge 回归模型
    sparse_ridge = Ridge(alpha=1.0, fit_intercept=False)
    dense_ridge = Ridge(alpha=1.0, fit_intercept=False)

    # 生成随机的数据矩阵 X 和响应变量向量 y
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)

    # 生成随机的样本权重向量 sample_weights
    sample_weights = rng.randn(n_samples) ** 2 + 1

    # 将密集矩阵 X 转换为稀疏矩阵 X_sparse
    X_sparse = sparse_container(X)

    # 在稀疏矩阵上拟合 Ridge 回归模型，使用 sample_weights 作为样本权重
    sparse_ridge.fit(X_sparse, y, sample_weight=sample_weights)

    # 在密集矩阵上拟合 Ridge 回归模型，使用 sample_weights 作为样本权重
    dense_ridge.fit(X, y, sample_weight=sample_weights)

    # 检查稀疏矩阵下的回归系数与密集矩阵下的回归系数是否几乎相等
    assert_array_almost_equal(sparse_ridge.coef_, dense_ridge.coef_, decimal=6)


def test_ridgecv_int_alphas():
    # 创建输入特征矩阵 X 和响应变量向量 y
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    # 使用整数列表作为 alpha 参数创建 RidgeCV 回归交叉验证对象
    ridge = RidgeCV(alphas=(1, 10, 100))
    ridge.fit(X, y)


@pytest.mark.parametrize("Estimator", [RidgeCV, RidgeClassifierCV])
@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        # 测试 alphas 参数的有效性，确保第二个元素大于零
        ({"alphas": (1, -1, -100)}, ValueError, r"alphas\[1\] == -1, must be > 0.0"),
        # 测试 alphas 参数的有效性，确保第一个元素大于零
        (
            {"alphas": (-0.1, -1.0, -10.0)},
            ValueError,
            r"alphas\[0\] == -0.1, must be > 0.0",
        ),
        # 测试 alphas 参数的类型，确保第三个元素是浮点数而不是字符串
        (
            {"alphas": (1, 1.0, "1")},
            TypeError,
            r"alphas\[2\] must be an instance of float, not str",
        ),
    ],
)
def test_ridgecv_alphas_validation(Estimator, params, err_type, err_msg):
    """Check the `alphas` validation in RidgeCV and RidgeClassifierCV."""

    # 创建随机的输入特征矩阵 X 和随机的响应变量向量 y
    n_samples, n_features = 5, 5
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)

    # 测试是否抛出预期的异常类型和消息
    with pytest.raises(err_type, match=err_msg):
        Estimator(**params).fit(X, y)


@pytest.mark.parametrize("Estimator", [RidgeCV, RidgeClassifierCV])
def test_ridgecv_alphas_scalar(Estimator):
    """Check the case when `alphas` is a scalar.
    This case was supported in the past when `alphas` where converted
    into array in `__init__`.
    We add this test to ensure backward compatibility.
    """

    # 创建随机的输入特征矩阵 X 和响应变量向量 y
    n_samples, n_features = 5, 5
    X = rng.randn(n_samples, n_features)
    if Estimator is RidgeCV:
        y = rng.randn(n_samples)
    else:
        y = rng.randint(0, 2, n_samples)

    # 使用标量值作为 alpha 参数创建 RidgeCV 回归交叉验证对象
    Estimator(alphas=1).fit(X, y)


def test_sparse_cg_max_iter():
    # 创建使用 "sparse_cg" 求解器和最大迭代次数为 1 的 Ridge 回归对象
    reg = Ridge(solver="sparse_cg", max_iter=1)
    reg.fit(X_diabetes, y_diabetes)
    # 断言回归系数的形状是否符合预期
    assert reg.coef_.shape[0] == X_diabetes.shape[1]


@ignore_warnings
def test_n_iter():
    # Test that self.n_iter_ is correct.

    # 设置目标数量为 2
    n_targets = 2

    # 创建输入特征矩阵 X 和响应变量向量 y
    X, y = X_diabetes, y_diabetes

    # 将响应变量向量 y 复制成 n_targets 个列，并进行转置
    y_n = np.tile(y, (n_targets, 1)).T

    # 遍历不同的最大迭代次数和求解器类型的组合
    for max_iter in range(1, 4):
        for solver in ("sag", "saga", "lsqr"):
            # 创建 Ridge 回归对象，设置最大迭代次数、容忍度和求解器类型
            reg = Ridge(solver=solver, max_iter=max_iter, tol=1e-12)
            reg.fit(X, y_n)
            # 断言 self.n_iter_ 是否符合预期，应该与最大迭代次数相同
            assert_array_equal(reg.n_iter_, np.tile(max_iter, n_targets))
    # 对于每个 solver 中的值（"sparse_cg", "svd", "cholesky"），依次进行以下操作：
    for solver in ("sparse_cg", "svd", "cholesky"):
        # 创建 Ridge 回归模型，指定求解方法为当前的 solver，最大迭代次数为 1，收敛阈值为 0.1
        reg = Ridge(solver=solver, max_iter=1, tol=1e-1)
        # 使用给定的数据集 X 和目标值 y_n 进行模型训练
        reg.fit(X, y_n)
        # 断言验证：模型未执行任何迭代（迭代次数为 None）
        assert reg.n_iter_ is None
@pytest.mark.parametrize("solver", ["lsqr", "sparse_cg", "lbfgs", "auto"])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_ridge_fit_intercept_sparse(
    solver, with_sample_weight, global_random_seed, csr_container
):
    """Check that ridge finds the same coefs and intercept on dense and sparse input
    in the presence of sample weights.

    For now only sparse_cg and lbfgs can correctly fit an intercept
    with sparse X with default tol and max_iter.
    'sag' is tested separately in test_ridge_fit_intercept_sparse_sag because it
    requires more iterations and should raise a warning if default max_iter is used.
    Other solvers raise an exception, as checked in
    test_ridge_fit_intercept_sparse_error
    """
    # 根据 solver 类型确定是否需要设置 positive 参数
    positive = solver == "lbfgs"
    # 生成稀疏或密集的 X 和对应的 y
    X, y = _make_sparse_offset_regression(
        n_features=20, random_state=global_random_seed, positive=positive
    )

    # 如果需要加权，则生成随机权重
    sample_weight = None
    if with_sample_weight:
        rng = np.random.RandomState(global_random_seed)
        sample_weight = 1.0 + rng.uniform(size=X.shape[0])

    # 根据 solver 类型选择合适的 Ridge 回归器
    dense_solver = "sparse_cg" if solver == "auto" else solver
    # 创建稠密数据下的 Ridge 回归器
    dense_ridge = Ridge(solver=dense_solver, tol=1e-12, positive=positive)
    # 创建稀疏数据下的 Ridge 回归器
    sparse_ridge = Ridge(solver=solver, tol=1e-12, positive=positive)

    # 在稠密数据上拟合 Ridge 回归器
    dense_ridge.fit(X, y, sample_weight=sample_weight)
    # 在稀疏数据上拟合 Ridge 回归器
    sparse_ridge.fit(csr_container(X), y, sample_weight=sample_weight)

    # 断言稠密和稀疏数据下的截距应该相等
    assert_allclose(dense_ridge.intercept_, sparse_ridge.intercept_)
    # 断言稠密和稀疏数据下的系数应该相等
    assert_allclose(dense_ridge.coef_, sparse_ridge.coef_, rtol=5e-7)


@pytest.mark.parametrize("solver", ["saga", "svd", "cholesky"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_ridge_fit_intercept_sparse_error(solver, csr_container):
    """Test that certain solvers raise an error when fitting Ridge regression with sparse data."""
    # 生成稀疏数据和对应的标签
    X, y = _make_sparse_offset_regression(n_features=20, random_state=0)
    # 转换为 CSR 格式的稀疏矩阵
    X_csr = csr_container(X)
    # 创建 Ridge 回归器
    sparse_ridge = Ridge(solver=solver)
    # 预期的错误消息
    err_msg = "solver='{}' does not support".format(solver)
    # 使用 pytest 的断言检查是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        sparse_ridge.fit(X_csr, y)


@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_ridge_fit_intercept_sparse_sag(
    with_sample_weight, global_random_seed, csr_container
):
    """Test ridge fitting with sag solver on sparse data."""
    # 生成稀疏数据和对应的标签
    X, y = _make_sparse_offset_regression(
        n_features=5, n_samples=20, random_state=global_random_seed, X_offset=5.0
    )
    # 如果需要加权，则生成随机权重
    if with_sample_weight:
        rng = np.random.RandomState(global_random_seed)
        sample_weight = 1.0 + rng.uniform(size=X.shape[0])
    else:
        sample_weight = None
    # 转换为 CSR 格式的稀疏矩阵
    X_csr = csr_container(X)
    # 定义参数字典，包括正则化强度、求解器类型、是否拟合截距、收敛容差和最大迭代次数
    params = dict(
        alpha=1.0, solver="sag", fit_intercept=True, tol=1e-10, max_iter=100000
    )
    
    # 创建一个稠密数据情况下的岭回归模型，并使用上述参数进行初始化
    dense_ridge = Ridge(**params)
    
    # 创建一个稀疏数据情况下的岭回归模型，并使用相同的参数进行初始化
    sparse_ridge = Ridge(**params)
    
    # 在稠密数据上拟合岭回归模型，可以传入样本权重参数
    dense_ridge.fit(X, y, sample_weight=sample_weight)
    
    # 在稀疏数据（使用 CSR 格式表示）上拟合岭回归模型，可以传入样本权重参数
    # 使用警告捕获机制，捕获 UserWarning 类型的警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sparse_ridge.fit(X_csr, y, sample_weight=sample_weight)
    
    # 断言稠密数据情况下的岭回归模型的截距与稀疏数据情况下的岭回归模型的截距非常接近
    assert_allclose(dense_ridge.intercept_, sparse_ridge.intercept_, rtol=1e-4)
    
    # 断言稠密数据情况下的岭回归模型的系数与稀疏数据情况下的岭回归模型的系数非常接近
    assert_allclose(dense_ridge.coef_, sparse_ridge.coef_, rtol=1e-4)
    
    # 使用 pytest 的警告断言，检查是否会发出特定类型的 UserWarning 警告，并匹配给定的正则表达式字符串
    with pytest.warns(UserWarning, match='"sag" solver requires.*'):
        # 创建一个使用 "sag" 求解器的岭回归模型，拟合于稀疏数据上
        Ridge(solver="sag", fit_intercept=True, tol=1e-3, max_iter=None).fit(X_csr, y)
@pytest.mark.parametrize("return_intercept", [False, True])
@pytest.mark.parametrize("sample_weight", [None, np.ones(1000)])
@pytest.mark.parametrize("container", [np.array] + CSR_CONTAINERS)
@pytest.mark.parametrize(
    "solver", ["auto", "sparse_cg", "cholesky", "lsqr", "sag", "saga", "lbfgs"]
)
def test_ridge_regression_check_arguments_validity(
    return_intercept, sample_weight, container, solver
):
    """检查所有参数组合是否能够给出有效的估计"""

    # 排除'svd'求解器，因为它对稀疏输入会引发异常

    rng = check_random_state(42)
    # 生成一个随机数生成器对象，种子为42
    X = rng.rand(1000, 3)
    # 生成一个1000行，3列的随机数矩阵
    true_coefs = [1, 2, 0.1]
    y = np.dot(X, true_coefs)
    # 计算真实的因变量值，这里是通过矩阵X和真实系数true_coefs计算得到
    true_intercept = 0.0
    if return_intercept:
        true_intercept = 10000.0
    y += true_intercept
    # 如果return_intercept为True，则加上真实的截距值
    X_testing = container(X)
    # 将生成的随机数矩阵X转换为指定容器类型，如np.array或CSR_CONTAINERS中的一种

    alpha, tol = 1e-3, 1e-6
    atol = 1e-3 if _IS_32BIT else 1e-4

    positive = solver == "lbfgs"

    if solver not in ["sag", "auto"] and return_intercept:
        # 如果求解器不是'sag'或'auto'，并且return_intercept为True
        with pytest.raises(ValueError, match="In Ridge, only 'sag' solver"):
            # 使用pytest断言捕获特定的异常，并匹配错误信息
            ridge_regression(
                X_testing,
                y,
                alpha=alpha,
                solver=solver,
                sample_weight=sample_weight,
                return_intercept=return_intercept,
                positive=positive,
                tol=tol,
            )
        return

    out = ridge_regression(
        X_testing,
        y,
        alpha=alpha,
        solver=solver,
        sample_weight=sample_weight,
        positive=positive,
        return_intercept=return_intercept,
        tol=tol,
    )

    if return_intercept:
        coef, intercept = out
        assert_allclose(coef, true_coefs, rtol=0, atol=atol)
        assert_allclose(intercept, true_intercept, rtol=0, atol=atol)
    else:
        assert_allclose(out, true_coefs, rtol=0, atol=atol)


@pytest.mark.parametrize(
    "solver", ["svd", "sparse_cg", "cholesky", "lsqr", "sag", "saga", "lbfgs"]
)
def test_dtype_match(solver):
    """检查数据类型是否匹配"""

    rng = np.random.RandomState(0)
    alpha = 1.0
    positive = solver == "lbfgs"

    n_samples, n_features = 6, 5
    X_64 = rng.randn(n_samples, n_features)
    y_64 = rng.randn(n_samples)
    X_32 = X_64.astype(np.float32)
    y_32 = y_64.astype(np.float32)

    tol = 2 * np.finfo(np.float32).resolution
    # 检查类型一致性32位
    ridge_32 = Ridge(
        alpha=alpha, solver=solver, max_iter=500, tol=tol, positive=positive
    )
    ridge_32.fit(X_32, y_32)
    coef_32 = ridge_32.coef_

    # 检查类型一致性64位
    ridge_64 = Ridge(
        alpha=alpha, solver=solver, max_iter=500, tol=tol, positive=positive
    )
    ridge_64.fit(X_64, y_64)
    coef_64 = ridge_64.coef_

    # 一次性进行实际检查，便于调试
    assert coef_32.dtype == X_32.dtype
    assert coef_64.dtype == X_64.dtype
    assert ridge_32.predict(X_32).dtype == X_32.dtype
    assert ridge_64.predict(X_64).dtype == X_64.dtype
    # 使用断言检查 ridge_32 和 ridge_64 的 coef_ 属性是否在给定的误差范围内接近
    assert_allclose(ridge_32.coef_, ridge_64.coef_, rtol=1e-4, atol=5e-4)
# 测试函数，用于验证 cholesky 求解器中不同 alpha 值的覆盖率。
def test_dtype_match_cholesky():
    # 随机数生成器，种子为 0
    rng = np.random.RandomState(0)
    # 设置 alpha 数组
    alpha = np.array([1.0, 0.5])

    # 定义样本数、特征数、目标数
    n_samples, n_features, n_target = 6, 7, 2
    # 生成符合正态分布的随机浮点数矩阵 X_64 和 y_64
    X_64 = rng.randn(n_samples, n_features)
    y_64 = rng.randn(n_samples, n_target)
    # 将 X_64 和 y_64 转换为 float32 类型
    X_32 = X_64.astype(np.float32)
    y_32 = y_64.astype(np.float32)

    # 检查类型一致性，使用 32 位
    ridge_32 = Ridge(alpha=alpha, solver="cholesky")
    ridge_32.fit(X_32, y_32)
    coef_32 = ridge_32.coef_

    # 检查类型一致性，使用 64 位
    ridge_64 = Ridge(alpha=alpha, solver="cholesky")
    ridge_64.fit(X_64, y_64)
    coef_64 = ridge_64.coef_

    # 一次性执行所有检查，便于调试
    assert coef_32.dtype == X_32.dtype
    assert coef_64.dtype == X_64.dtype
    assert ridge_32.predict(X_32).dtype == X_32.dtype
    assert ridge_64.predict(X_64).dtype == X_64.dtype
    assert_almost_equal(ridge_32.coef_, ridge_64.coef_, decimal=5)


# 参数化测试，验证 Ridge 回归的数据类型稳定性
@pytest.mark.parametrize(
    "solver", ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
)
@pytest.mark.parametrize("seed", range(1))
def test_ridge_regression_dtype_stability(solver, seed):
    # 设置随机数种子
    random_state = np.random.RandomState(seed)
    n_samples, n_features = 6, 5
    # 生成随机样本 X 和系数 coef
    X = random_state.randn(n_samples, n_features)
    coef = random_state.randn(n_features)
    y = np.dot(X, coef) + 0.01 * random_state.randn(n_samples)
    alpha = 1.0
    positive = solver == "lbfgs"
    results = dict()
    # 对于不同的数据类型（np.float32 和 np.float64），进行 Ridge 回归
    # 设置容差值，sparse_cg 求解器的容差较高
    atol = 1e-3 if solver == "sparse_cg" else 1e-5
    for current_dtype in (np.float32, np.float64):
        results[current_dtype] = ridge_regression(
            X.astype(current_dtype),
            y.astype(current_dtype),
            alpha=alpha,
            solver=solver,
            random_state=random_state,
            sample_weight=None,
            positive=positive,
            max_iter=500,
            tol=1e-10,
            return_n_iter=False,
            return_intercept=False,
        )

    # 断言结果的数据类型正确
    assert results[np.float32].dtype == np.float32
    assert results[np.float64].dtype == np.float64
    # 检查两种数据类型的结果是否接近
    assert_allclose(results[np.float32], results[np.float64], atol=atol)


# 测试 Ridge 回归在使用 SAG 求解器时，处理 Fortran 数组的情况
def test_ridge_sag_with_X_fortran():
    # 使用 make_regression 生成数据集 X, y
    X, y = make_regression(random_state=42)
    # 将 X 转换为 Fortran 格式的数组
    X = np.asfortranarray(X)
    # 仅保留偶数行，但列数不变
    X = X[::2, :]
    y = y[::2]
    # 使用 SAG 求解器拟合 Ridge 模型
    Ridge(solver="sag").fit(X, y)


# 参数化测试，验证 RidgeClassifier 在多标签分类任务中的行为
@pytest.mark.parametrize(
    "Classifier, params",
    [
        (RidgeClassifier, {}),
        (RidgeClassifierCV, {"cv": None}),
        (RidgeClassifierCV, {"cv": 3}),
    ],
)
def test_ridgeclassifier_multilabel(Classifier, params):
    """Check that multilabel classification is supported and give meaningful
    results."""
    # 生成一个多标签分类的数据集，n_classes=1 表示只有一个类别，random_state=0 确保结果可复现
    X, y = make_multilabel_classification(n_classes=1, random_state=0)
    # 调整 y 的形状为列向量
    y = y.reshape(-1, 1)
    # 创建 Y 矩阵，将 y 重复两次作为两列，形成一个二列的 Y 矩阵
    Y = np.concatenate([y, y], axis=1)
    # 使用给定的参数 params 创建一个分类器对象 clf，并使用 X 和 Y 进行拟合
    clf = Classifier(**params).fit(X, Y)
    # 预测 X 的标签，得到预测结果 Y_pred
    Y_pred = clf.predict(X)

    # 断言预测结果的形状与 Y 的形状相同
    assert Y_pred.shape == Y.shape
    # 断言 Y_pred 的第一列与第二列的值相等
    assert_array_equal(Y_pred[:, 0], Y_pred[:, 1])
    # 使用 Ridge 回归器，使用 "sag" 求解器拟合 X 和 y 数据
    Ridge(solver="sag").fit(X, y)
# 使用 pytest.mark.parametrize 装饰器指定参数化测试，测试不同的求解器（solver）
# 和是否拟合截距（fit_intercept）
@pytest.mark.parametrize("solver", ["auto", "lbfgs"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_ridge_positive_regression_test(solver, fit_intercept, alpha):
    """Test that positive Ridge finds true positive coefficients."""
    # 创建一个 4x2 的输入特征矩阵 X
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    # 设置真实的系数向量 coef
    coef = np.array([1, -10])
    # 如果 fit_intercept 为 True，则设置截距为 20，并生成对应的目标值 y
    if fit_intercept:
        intercept = 20
        y = X.dot(coef) + intercept
    else:
        y = X.dot(coef)

    # 创建 Ridge 回归模型对象，根据参数设置 alpha, positive, solver 和 fit_intercept
    model = Ridge(
        alpha=alpha, positive=True, solver=solver, fit_intercept=fit_intercept
    )
    # 使用 X, y 数据拟合模型
    model.fit(X, y)
    # 断言所有的模型系数都大于等于 0
    assert np.all(model.coef_ >= 0)


# 使用 pytest.mark.parametrize 装饰器指定参数化测试，测试不同的拟合截距（fit_intercept）
# 和 alpha 参数
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_ridge_ground_truth_positive_test(fit_intercept, alpha):
    """Test that Ridge w/wo positive converges to the same solution.

    Ridge with positive=True and positive=False must give the same
    when the ground truth coefs are all positive.
    """
    # 使用随机数种子 42 创建 300x100 的随机特征矩阵 X
    rng = np.random.RandomState(42)
    X = rng.randn(300, 100)
    # 生成均匀分布在 0.1 到 1.0 之间的随机系数向量 coef
    coef = rng.uniform(0.1, 1.0, size=X.shape[1])
    # 如果 fit_intercept 为 True，则设置截距为 1，并生成对应的目标值 y
    if fit_intercept:
        intercept = 1
        y = X @ coef + intercept
    else:
        y = X @ coef
    # 向 y 添加服从正态分布的噪声
    y += rng.normal(size=X.shape[0]) * 0.01

    results = []
    # 遍历 positive 参数为 True 和 False 的两种情况
    for positive in [True, False]:
        # 创建 Ridge 回归模型对象，根据参数设置 alpha, positive 和 fit_intercept
        model = Ridge(
            alpha=alpha, positive=positive, fit_intercept=fit_intercept, tol=1e-10
        )
        # 使用 X, y 数据拟合模型，并获取系数
        results.append(model.fit(X, y).coef_)
    # 断言两种情况下得到的系数结果非常接近
    assert_allclose(*results, atol=1e-6, rtol=0)


# 使用 pytest.mark.parametrize 装饰器指定参数化测试，测试不同的求解器（solver）
@pytest.mark.parametrize(
    "solver", ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
)
def test_ridge_positive_error_test(solver):
    """Test input validation for positive argument in Ridge."""
    # 设置 alpha 和 2x2 的输入特征矩阵 X
    alpha = 0.1
    X = np.array([[1, 2], [3, 4]])
    # 设置真实的系数向量 coef
    coef = np.array([1, -1])
    # 生成目标值 y
    y = X @ coef

    # 创建 Ridge 回归模型对象，根据参数设置 alpha, positive, solver 和 fit_intercept
    model = Ridge(alpha=alpha, positive=True, solver=solver, fit_intercept=False)
    # 断言使用不支持 positive 参数时会引发 ValueError 异常
    with pytest.raises(ValueError, match="does not support positive"):
        model.fit(X, y)

    # 断言使用不支持的求解器时会引发 ValueError 异常
    with pytest.raises(ValueError, match="only 'lbfgs' solver can be used"):
        _, _ = ridge_regression(
            X, y, alpha, positive=True, solver=solver, return_intercept=False
        )


# 使用 pytest.mark.parametrize 装饰器指定参数化测试，测试不同的 alpha 参数
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_positive_ridge_loss(alpha):
    """Check ridge loss consistency when positive argument is enabled."""
    # 创建 300x300 的回归样本数据集 X, y
    X, y = make_regression(n_samples=300, n_features=300, random_state=42)
    alpha = 0.10  # 重新设置 alpha 的值为 0.10，之前的参数化测试参数被覆盖

    n_checks = 100

    def ridge_loss(model, random_state=None, noise_scale=1e-8):
        # 获取模型的截距
        intercept = model.intercept_
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            # 获取模型的系数并添加随机噪声
            coef = model.coef_ + rng.uniform(0, noise_scale, size=model.coef_.shape)
        else:
            coef = model.coef_

        # 计算岭回归损失函数
        return 0.5 * np.sum((y - X @ coef - intercept) ** 2) + 0.5 * alpha * np.sum(
            coef**2
        )

    # 在测试函数中没有直接调用 ridge_loss 函数，只是定义了它
    # 使用 Ridge 回归模型拟合数据 X 和目标 y，使用给定的 alpha 参数
    model = Ridge(alpha=alpha).fit(X, y)
    # 使用 Ridge 回归模型拟合数据 X 和目标 y，并强制所有系数为非负数（positive=True）
    model_positive = Ridge(alpha=alpha, positive=True).fit(X, y)

    # Check 1:
    #   使用 Ridge(alpha=alpha) 得到的解的损失值应低于使用 Ridge(alpha=alpha, positive=True) 得到的解的损失值
    loss = ridge_loss(model)
    loss_positive = ridge_loss(model_positive)
    assert loss <= loss_positive

    # Check 2:
    #   使用 Ridge(alpha=alpha, positive=True) 得到的解的损失值应低于对该解进行小幅随机正向扰动后的损失值
    for random_state in range(n_checks):
        loss_perturbed = ridge_loss(model_positive, random_state=random_state)
        assert loss_positive <= loss_perturbed
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_lbfgs_solver_consistency(alpha):
    """Test that LBGFS gets almost the same coef of svd when positive=False."""
    # 创建一个具有300个样本和300个特征的回归数据集
    X, y = make_regression(n_samples=300, n_features=300, random_state=42)
    # 将y转换为列向量
    y = np.expand_dims(y, 1)
    # 将alpha转换为NumPy数组
    alpha = np.asarray([alpha])
    # 配置字典，设置正负标志为False，容差为1e-16，最大迭代次数为500000
    config = {
        "positive": False,
        "tol": 1e-16,
        "max_iter": 500000,
    }

    # 使用LBFGS求解器求解系数
    coef_lbfgs = _solve_lbfgs(X, y, alpha, **config)
    # 使用SVD求解器求解系数
    coef_cholesky = _solve_svd(X, y, alpha)
    # 断言LBFGS求解器得到的系数与SVD求解器得到的系数几乎相同
    assert_allclose(coef_lbfgs, coef_cholesky, atol=1e-4, rtol=0)


def test_lbfgs_solver_error():
    """Test that LBFGS solver raises ConvergenceWarning."""
    # 创建一个包含两个样本的二维数组X和一个二维数组y
    X = np.array([[1, -1], [1, 1]])
    y = np.array([-1e10, 1e10])

    # 创建一个Ridge回归模型对象，使用LBFGS求解器，alpha=0.01，不使用截距项，容差为1e-12，正负标志为True，最大迭代次数为1
    model = Ridge(
        alpha=0.01,
        solver="lbfgs",
        fit_intercept=False,
        tol=1e-12,
        positive=True,
        max_iter=1,
    )
    # 使用pytest检查是否引发了ConvergenceWarning警告，警告信息包含"lbfgs solver did not converge"
    with pytest.warns(ConvergenceWarning, match="lbfgs solver did not converge"):
        model.fit(X, y)


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("data", ["tall", "wide"])
@pytest.mark.parametrize("solver", SOLVERS + ["lbfgs"])
def test_ridge_sample_weight_consistency(
    fit_intercept, sparse_container, data, solver, global_random_seed
):
    """Test that the impact of sample_weight is consistent.

    Note that this test is stricter than the common test
    check_sample_weights_invariance alone.
    """
    # 过滤不支持稀疏输入的求解器
    if sparse_container is not None:
        if solver == "svd" or (solver in ("cholesky", "saga") and fit_intercept):
            pytest.skip("unsupported configuration")

    # XXX: this test is quite sensitive to the seed used to generate the data:
    # ideally we would like the test to pass for any global_random_seed but this is not
    # the case at the moment.
    # 使用随机种子生成随机数发生器rng
    rng = np.random.RandomState(42)
    n_samples = 12
    if data == "tall":
        n_features = n_samples // 2
    else:
        n_features = n_samples * 2

    # 生成n_samples行，n_features列的随机数矩阵X
    X = rng.rand(n_samples, n_features)
    # 生成n_samples长度的随机数向量y
    y = rng.rand(n_samples)
    # 如果sparse_container不为None，将X转换为稀疏矩阵格式
    if sparse_container is not None:
        X = sparse_container(X)
    
    # 配置参数字典，包括fit_intercept、alpha、solver、positive、random_state和tol
    params = dict(
        fit_intercept=fit_intercept,
        alpha=1.0,
        solver=solver,
        positive=(solver == "lbfgs"),
        random_state=global_random_seed,  # for sag/saga
        tol=1e-12,
    )

    # 1) sample_weight=np.ones(..) should be equivalent to sample_weight=None
    # 使用Ridge回归对象reg拟合数据，sample_weight为None时应当与sample_weight=np.ones_like(y)等效
    reg = Ridge(**params).fit(X, y, sample_weight=None)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    # 断言拟合后的系数与初始系数非常接近
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    # 如果 fit_intercept 为 True，则验证回归模型的截距是否与预期相符
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 2) 将 sample_weight 的元素设置为 0 等同于移除这些样本
    # 类似于 check_sample_weights_invariance(name, reg, kind="zeros") 的检查，但我们还测试了稀疏输入的情况
    sample_weight = rng.uniform(low=0.01, high=2, size=X.shape[0])
    sample_weight[-5:] = 0
    y[-5:] *= 1000  # 为了确保排除这些样本的重要性而放大这些样本的影响
    reg.fit(X, y, sample_weight=sample_weight)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    reg.fit(X[:-5, :], y[:-5], sample_weight=sample_weight[:-5])
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 3) 对 sample_weight 进行缩放不应该有任何影响
    # 注意：对于带有惩罚项的模型，缩放惩罚项可能有效。
    reg2 = Ridge(**params).set_params(alpha=np.pi * params["alpha"])
    reg2.fit(X, y, sample_weight=np.pi * sample_weight)
    if solver in ("sag", "saga") and not fit_intercept:
        pytest.xfail(f"Solver {solver} does fail test for scaling of sample_weight.")
    assert_allclose(reg2.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg2.intercept_, intercept)

    # 4) 检查将 sample_weight 乘以 2 是否等同于相应样本重复两次
    if sparse_container is not None:
        X = X.toarray()
    X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]])
    sample_weight_1 = sample_weight.copy()
    sample_weight_1[: n_samples // 2] *= 2
    sample_weight_2 = np.concatenate(
        [sample_weight, sample_weight[: n_samples // 2]], axis=0
    )
    if sparse_container is not None:
        X = sparse_container(X)
        X2 = sparse_container(X2)
    reg1 = Ridge(**params).fit(X, y, sample_weight=sample_weight_1)
    reg2 = Ridge(**params).fit(X2, y2, sample_weight=sample_weight_2)
    assert_allclose(reg1.coef_, reg2.coef_)
    if fit_intercept:
        assert_allclose(reg1.intercept_, reg2.intercept_)
# TODO(1.7): Remove
# 定义一个测试函数，用于验证 `store_cv_values` 参数是否已经被废弃
def test_ridge_store_cv_values_deprecated():
    """Check `store_cv_values` parameter deprecated."""
    # 创建一个具有随机数据的样本集合 X, y
    X, y = make_regression(n_samples=6, random_state=42)
    
    # 创建 RidgeCV 对象，并设置 store_cv_values=True
    ridge = RidgeCV(store_cv_values=True)
    
    # 准备将要显示的警告消息
    msg = "'store_cv_values' is deprecated"
    
    # 使用 pytest 的 warn 断言，检查是否会抛出 FutureWarning 警告，且警告消息匹配预期的消息
    with pytest.warns(FutureWarning, match=msg):
        ridge.fit(X, y)

    # 当同时设置了 store_cv_results 和 store_cv_values 时会出错
    ridge = RidgeCV(store_cv_results=True, store_cv_values=True)
    
    # 准备将要显示的错误消息
    msg = "Both 'store_cv_values' and 'store_cv_results' were"
    
    # 使用 pytest 的 raises 断言，检查是否会抛出 ValueError 异常，且异常消息匹配预期的消息
    with pytest.raises(ValueError, match=msg):
        ridge.fit(X, y)


# Metadata Routing Tests
# ======================

# 使用 pytest.mark.usefixtures 装饰器，为测试函数设置 enable_slep006 修饰器
@pytest.mark.usefixtures("enable_slep006")
# 使用 pytest.mark.parametrize 装饰器，参数化 metaestimator 变量，分别传入 RidgeCV 和 RidgeClassifierCV 作为参数
def test_metadata_routing_with_default_scoring(metaestimator):
    """Test that `RidgeCV` or `RidgeClassifierCV` with default `scoring`
    argument (`None`), don't enter into `RecursionError` when metadata is routed.
    """
    # 创建一个 metaestimator 对象，可能是 RidgeCV 或 RidgeClassifierCV
    metaestimator().get_metadata_routing()


# End of Metadata Routing Tests
# =============================
```