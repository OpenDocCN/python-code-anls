# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_coordinate_descent.py`

```
# 导入警告模块，深拷贝函数等必要库
import warnings
from copy import deepcopy

# 导入joblib用于并行处理，numpy用于数值计算，pytest用于测试
import joblib
import numpy as np
import pytest

# 导入scipy中的插值和稀疏矩阵处理模块
from scipy import interpolate, sparse

# 导入scikit-learn相关模块和类
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    enet_path,
    lars_path,
    lasso_path,
)

# 导入sklearn内部的坐标下降优化函数和相关模块
from sklearn.linear_model._coordinate_descent import _set_order

# 导入交叉验证、网格搜索等模块
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    LeaveOneGroupOut,
)

# 导入用于处理分组数据的混合类
from sklearn.model_selection._split import GroupsConsumerMixin

# 导入创建管道的函数和数据预处理模块
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 导入用于数组检查和测试的辅助函数和类
from sklearn.utils import check_array
from sklearn.utils._testing import (
    TempMemmap,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)

# 导入用于处理稀疏矩阵兼容性的相关修复函数
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("input_order", ["C", "F"])
def test_set_order_dense(order, input_order):
    """检查_set_order是否按照承诺的顺序返回数组。"""
    # 创建按照指定顺序的数组X和y
    X = np.array([[0], [0], [0]], order=input_order)
    y = np.array([0, 0, 0], order=input_order)
    # 调用_set_order函数处理数组X和y，并返回新的数组X2和y2
    X2, y2 = _set_order(X, y, order=order)
    # 如果顺序为"C"，则检查X2和y2是否C连续
    if order == "C":
        assert X2.flags["C_CONTIGUOUS"]
        assert y2.flags["C_CONTIGUOUS"]
    # 如果顺序为"F"，则检查X2和y2是否F连续
    elif order == "F":
        assert X2.flags["F_CONTIGUOUS"]
        assert y2.flags["F_CONTIGUOUS"]

    # 如果输入顺序与处理后的顺序相同，则断言X和X2、y和y2是同一个对象
    if order == input_order:
        assert X is X2
        assert y is y2


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("input_order", ["C", "F"])
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_set_order_sparse(order, input_order, coo_container):
    """检查_set_order是否按照承诺的格式返回稀疏矩阵。"""
    # 创建按照指定容器类型的稀疏矩阵X和y
    X = coo_container(np.array([[0], [0], [0]]))
    y = coo_container(np.array([0, 0, 0]))
    # 将稀疏矩阵按照输入顺序格式化为"csc"或"csr"
    sparse_format = "csc" if input_order == "F" else "csr"
    X = X.asformat(sparse_format)
    y = X.asformat(sparse_format)
    # 调用_set_order函数处理稀疏矩阵X和y，并返回新的稀疏矩阵X2和y2
    X2, y2 = _set_order(X, y, order=order)

    # 根据输出顺序检查X2和y2是否是稀疏矩阵，并且格式正确
    format = "csc" if order == "F" else "csr"
    assert sparse.issparse(X2) and X2.format == format
    assert sparse.issparse(y2) and y2.format == format


def test_lasso_zero():
    """检查Lasso是否能处理零数据而不崩溃。"""
    # 创建全为零的输入数据X和y
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    # _cd_fast.pyx中测试gap < tol，但这里是0.0 < 0.0，应该改为gap <= tol？
    # 这里主要是一个注释，提出对源代码的一个可能改进的建议。
    # 使用 `ignore_warnings` 上下文管理器忽略 `ConvergenceWarning` 警告
    with ignore_warnings(category=ConvergenceWarning):
        # 使用 Lasso 回归模型，alpha 参数设为 0.1，对输入数据 X 和目标数据 y 进行拟合
        clf = Lasso(alpha=0.1).fit(X, y)
    
    # 使用训练好的模型 clf 进行预测，预测输入为 [[1], [2], [3]]
    pred = clf.predict([[1], [2], [3]])
    
    # 断言检查训练后的系数 clf.coef_ 是否接近于 [0]
    assert_array_almost_equal(clf.coef_, [0])
    
    # 断言检查预测结果 pred 是否接近于 [0, 0, 0]
    assert_array_almost_equal(pred, [0, 0, 0])
    
    # 断言检查模型的 dual_gap_ 属性是否接近于 0
    assert_almost_equal(clf.dual_gap_, 0)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
# 标记此测试以忽略与收敛警告相关的所有警告消息

def test_enet_nonfinite_params():
    # 测试 ElasticNet 在处理非有限参数值时是否会引发 ValueError 异常

    rng = np.random.RandomState(0)
    n_samples = 10
    fmax = np.finfo(np.float64).max
    X = fmax * rng.uniform(size=(n_samples, 2))
    y = rng.randint(0, 2, size=n_samples)

    clf = ElasticNet(alpha=0.1)
    # 预期的错误消息
    msg = "Coordinate descent iterations resulted in non-finite parameter values"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_lasso_toy():
    # 使用简单示例测试 Lasso 的不同 alpha 值

    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]  # 一条直线
    T = [[2], [3], [4]]  # 测试样本

    clf = Lasso(alpha=1e-8)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=0.1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


def test_enet_toy():
    # 使用简单示例测试 ElasticNet 的不同 alpha 和 l1_ratio 参数

    X = np.array([[-1.0], [0.0], [1.0]])
    Y = [-1, 0, 1]  # 一条直线
    T = [[2.0], [3.0], [4.0]]  # 测试样本

    # 当 l1_ratio = 1 时，应与 Lasso 相同
    clf = ElasticNet(alpha=1e-8, l1_ratio=1.0)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=100, precompute=False)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf.set_params(max_iter=100, precompute=True)
    clf.fit(X, Y)  # 使用预计算的 Gram 矩阵进行拟合
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)
    # 设置分类器的参数：最大迭代次数为100，使用预计算的 Gram 矩阵 np.dot(X.T, X)
    clf.set_params(max_iter=100, precompute=np.dot(X.T, X))
    # 使用训练数据 X 和标签 Y 进行拟合
    clf.fit(X, Y)  # 使用预计算的 Gram 矩阵进行训练
    # 对测试数据 T 进行预测
    pred = clf.predict(T)
    # 检查分类器拟合后的系数是否与期望值接近，精度为小数点后三位
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    # 检查预测结果是否与期望值接近，精度为小数点后三位
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    # 检查分类器的对偶间隙是否接近零
    assert_almost_equal(clf.dual_gap_, 0)

    # 使用 ElasticNet 模型创建分类器，设置 alpha=0.5 和 l1_ratio=0.5
    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    # 使用训练数据 X 和标签 Y 进行拟合
    clf.fit(X, Y)
    # 对测试数据 T 进行预测
    pred = clf.predict(T)
    # 检查分类器拟合后的系数是否与期望值接近，精度为小数点后三位
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    # 检查预测结果是否与期望值接近，精度为小数点后三位
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    # 检查分类器的对偶间隙是否接近零
    assert_almost_equal(clf.dual_gap_, 0)
# 测试 Lasso 模型的对偶间隙是否与其目标函数的公式匹配，其中数据拟合通过 n_samples 进行了标准化
def test_lasso_dual_gap():
    # 使用 build_dataset 函数生成包含 10 个样本和 30 个特征的数据集
    X, y, _, _ = build_dataset(n_samples=10, n_features=30)
    # 计算样本数
    n_samples = len(y)
    # 计算 alpha 值，为 X.T @ y 的绝对值最大值的 0.01 倍，除以样本数得到
    alpha = 0.01 * np.max(np.abs(X.T @ y)) / n_samples
    # 使用 Lasso 模型拟合数据，设置 alpha 和不拟合截距项
    clf = Lasso(alpha=alpha, fit_intercept=False).fit(X, y)
    # 获取模型的系数向量 w
    w = clf.coef_
    # 计算残差 R
    R = y - X @ w
    # 计算原始问题的值
    primal = 0.5 * np.mean(R**2) + clf.alpha * np.sum(np.abs(w))
    # 对偶问题中 R 的更新：R 除以 (n_samples * alpha) 与 X.T @ R 的最大值的绝对值比较
    R /= np.max(np.abs(X.T @ R) / (n_samples * alpha))
    # 计算对偶问题的值
    dual = 0.5 * (np.mean(y**2) - np.mean((y - R) ** 2))
    # 断言 Lasso 模型的对偶间隙与原始问题值和对偶问题值之差的接近程度
    assert_allclose(clf.dual_gap_, primal - dual)


# 构建一个具有许多噪声特征和相对较少样本的不适定线性回归问题的数据集
def build_dataset(n_samples=50, n_features=200, n_informative_features=10, n_targets=1):
    random_state = np.random.RandomState(0)
    # 使用随机状态生成正态分布的权重向量 w，其中非信息特征的权重置为零
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    # 使用随机状态生成样本矩阵 X
    X = random_state.randn(n_samples, n_features)
    # 计算响应变量 y，y = X @ w
    y = np.dot(X, w)
    # 使用随机状态生成测试样本矩阵 X_test
    X_test = random_state.randn(n_samples, n_features)
    # 计算测试响应变量 y_test，y_test = X_test @ w
    y_test = np.dot(X_test, w)
    # 返回数据集 X, y 和测试集 X_test, y_test
    return X, y, X_test, y_test


# 测试 LassoCV 模型的性能
def test_lasso_cv():
    # 使用 build_dataset 函数生成默认参数的数据集
    X, y, X_test, y_test = build_dataset()
    max_iter = 150
    # 使用 LassoCV 拟合数据，设置 alpha 个数、收敛精度、最大迭代次数和交叉验证折数
    clf = LassoCV(n_alphas=10, eps=1e-3, max_iter=max_iter, cv=3).fit(X, y)
    # 断言模型选择的 alpha 值接近 0.056，精度为小数点后两位
    assert_almost_equal(clf.alpha_, 0.056, 2)

    # 使用预计算选项重新拟合数据集
    clf = LassoCV(n_alphas=10, eps=1e-3, max_iter=max_iter, precompute=True, cv=3)
    clf.fit(X, y)
    # 断言模型选择的 alpha 值接近 0.056，精度为小数点后两位
    assert_almost_equal(clf.alpha_, 0.056, 2)

    # 检查 LassoLarsCV 和坐标下降实现是否选择了相似的 alpha
    lars = LassoLarsCV(max_iter=30, cv=3).fit(X, y)
    # 确保 lars.alpha_ 和 clf.alpha_ 在 alpha 列表中的位置不超过 1
    assert (
        np.abs(
            np.searchsorted(clf.alphas_[::-1], lars.alpha_)
            - np.searchsorted(clf.alphas_[::-1], clf.alpha_)
        )
        <= 1
    )
    # 检查两者是否给出相似的均方误差
    mse_lars = interpolate.interp1d(lars.cv_alphas_, lars.mse_path_.T)
    np.testing.assert_approx_equal(
        mse_lars(clf.alphas_[5]).mean(), clf.mse_path_[5].mean(), significant=2
    )

    # 测试集评分是否大于 0.99
    assert clf.score(X_test, y_test) > 0.99


# 测试带有某些模型选择的 LassoCV 模型
def test_lasso_cv_with_some_model_selection():
    from sklearn import datasets
    from sklearn.model_selection import ShuffleSplit

    # 加载糖尿病数据集
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # 创建管道，包括标准化和 LassoCV 模型，使用 ShuffleSplit 进行交叉验证
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=ShuffleSplit(random_state=0)))
    pipe.fit(X, y)


# 测试 LassoCV 模型带有正约束条件
def test_lasso_cv_positive_constraint():
    # 使用 build_dataset 函数生成默认参数的数据集
    X, y, X_test, y_test = build_dataset()
    max_iter = 500

    # 确保未约束拟合的系数为负数
    clf_unconstrained = LassoCV(n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, n_jobs=1)
    clf_unconstrained.fit(X, y)
    # 断言：无约束条件下的模型系数中最小值应小于0
    assert min(clf_unconstrained.coef_) < 0

    # 在相同的数据上，通过LassoCV进行约束条件拟合后，系数应非负
    clf_constrained = LassoCV(
        n_alphas=3,         # alpha参数的数量
        eps=1e-1,           # alpha的最小比例增量
        max_iter=max_iter,  # 最大迭代次数
        positive=True,      # 系数是否应该是非负的
        cv=2,               # 交叉验证的折数
        n_jobs=1            # 并行运行的作业数
    )
    # 对数据集X和目标y进行拟合
    clf_constrained.fit(X, y)

    # 断言：约束条件下的模型系数中最小值应大于等于0
    assert min(clf_constrained.coef_) >= 0
@pytest.mark.parametrize(
    "alphas, err_type, err_msg",
    [
        ((1, -1, -100), ValueError, r"alphas\[1\] == -1, must be >= 0.0."),
        (
            (-0.1, -1.0, -10.0),
            ValueError,
            r"alphas\[0\] == -0.1, must be >= 0.0.",
        ),
        (
            (1, 1.0, "1"),
            TypeError,
            r"alphas\[2\] must be an instance of float, not str",
        ),
    ],
)
def test_lassocv_alphas_validation(alphas, err_type, err_msg):
    """Check the `alphas` validation in LassoCV."""
    
    # 生成一些随机数据，包括特征矩阵 X 和目标变量 y
    n_samples, n_features = 5, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)
    
    # 创建一个 LassoCV 模型实例，验证 alphas 参数的有效性
    lassocv = LassoCV(alphas=alphas)
    
    # 使用 pytest 检查是否引发了预期的错误类型和错误消息
    with pytest.raises(err_type, match=err_msg):
        lassocv.fit(X, y)


def _scale_alpha_inplace(estimator, n_samples):
    """Rescale the parameter alpha from when the estimator is evoked with
    normalize set to True as if it were evoked in a Pipeline with normalize set
    to False and with a StandardScaler.
    """
    
    # 如果估计器中没有 alpha 或 alphas 参数，则直接返回
    if ("alpha" not in estimator.get_params()) and (
        "alphas" not in estimator.get_params()
    ):
        return
    
    # 根据估计器的类型不同进行 alpha 参数的缩放处理
    if isinstance(estimator, (RidgeCV, RidgeClassifierCV)):
        # alphas 可能是一个列表，将其转换为 np.ndarray 以确保使用广播
        alphas = np.asarray(estimator.alphas) * n_samples
        return estimator.set_params(alphas=alphas)
    
    if isinstance(estimator, (Lasso, LassoLars, MultiTaskLasso)):
        alpha = estimator.alpha * np.sqrt(n_samples)
    
    if isinstance(estimator, (Ridge, RidgeClassifier)):
        alpha = estimator.alpha * n_samples
    
    if isinstance(estimator, (ElasticNet, MultiTaskElasticNet)):
        if estimator.l1_ratio == 1:
            alpha = estimator.alpha * np.sqrt(n_samples)
        elif estimator.l1_ratio == 0:
            alpha = estimator.alpha * n_samples
        else:
            # 如果重构时发生错误，抛出 NotImplementedError 避免潜在的静默错误
            raise NotImplementedError
    
    # 更新估计器的 alpha 参数
    estimator.set_params(alpha=alpha)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize(
    "LinearModel, params",
    [
        (Lasso, {"tol": 1e-16, "alpha": 0.1}),
        (LassoCV, {"tol": 1e-16}),
        (ElasticNetCV, {}),
        (RidgeClassifier, {"solver": "sparse_cg", "alpha": 0.1}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 1, "alpha": 0.01}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 0, "alpha": 0.01}),
        (Ridge, {"solver": "sparse_cg", "tol": 1e-12, "alpha": 0.1}),
        (LinearRegression, {}),
        (RidgeCV, {}),
        (RidgeClassifierCV, {}),
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_model_pipeline_same_dense_and_sparse(LinearModel, params, csr_container):
    """Test that linear model preceded by StandardScaler in the pipeline and
    """
    # 使用 `with_mean=False` 参数可以保持稀疏或密集数据的一致性，使得预测结果 `y_pred` 和 `.coef_` 的值相同。
    # 创建一个标准化后的管道模型，用于处理密集数据
    model_dense = make_pipeline(StandardScaler(with_mean=False), LinearModel(**params))

    # 创建一个标准化后的管道模型，用于处理稀疏数据
    model_sparse = make_pipeline(StandardScaler(with_mean=False), LinearModel(**params))

    # 准备数据
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0  # 将小于0.1的元素设为0，生成稀疏数据

    # 将密集数据转换为稀疏数据格式
    X_sparse = csr_container(X)
    y = rng.rand(n_samples)

    # 如果模型是分类器，则将目标变量 y 转换为 +1 或 -1
    if is_classifier(model_dense):
        y = np.sign(y)

    # 使用密集数据拟合模型
    model_dense.fit(X, y)
    
    # 使用稀疏数据拟合模型
    model_sparse.fit(X_sparse, y)

    # 断言稀疏模型的系数与密集模型的系数非常接近
    assert_allclose(model_sparse[1].coef_, model_dense[1].coef_)

    # 对密集数据进行预测
    y_pred_dense = model_dense.predict(X)
    
    # 对稀疏数据进行预测
    y_pred_sparse = model_sparse.predict(X_sparse)

    # 断言密集预测结果和稀疏预测结果非常接近
    assert_allclose(y_pred_dense, y_pred_sparse)

    # 断言稀疏模型的截距与密集模型的截距非常接近
    assert_allclose(model_dense[1].intercept_, model_sparse[1].intercept_)
# 定义一个测试函数，用于验证 lasso_path 的输出与新输出方式的系数路径是否相同
def test_lasso_path_return_models_vs_new_return_gives_same_coefficients():
    # 测试 lasso_path 使用 lars_path 风格输出是否给出相同结果

    # 创建一些玩具数据
    X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T  # 两列数据的转置数组
    y = np.array([1, 2, 3.1])  # 目标变量数组
    alphas = [5.0, 1.0, 0.5]  # 正则化参数列表

    # 使用 lars_path 和 lasso_path(new output) 计算相同路径的系数
    alphas_lars, _, coef_path_lars = lars_path(X, y, method="lasso")
    coef_path_cont_lars = interpolate.interp1d(
        alphas_lars[::-1], coef_path_lars[:, ::-1]
    )  # 使用线性插值处理 lars_path 的系数路径

    alphas_lasso2, coef_path_lasso2, _ = lasso_path(X, y, alphas=alphas)
    coef_path_cont_lasso = interpolate.interp1d(
        alphas_lasso2[::-1], coef_path_lasso2[:, ::-1]
    )  # 使用线性插值处理 lasso_path(new output) 的系数路径

    # 断言两种插值后的系数路径在指定的 alpha 值处相等，精度为小数点后一位
    assert_array_almost_equal(
        coef_path_cont_lasso(alphas), coef_path_cont_lars(alphas), decimal=1
    )


# 定义一个测试函数，用于测试 ElasticNetCV 的路径选择
def test_enet_path():
    # 使用大量样本和信息特征，以便 l1_ratio 更接近岭回归而不是 Lasso
    X, y, X_test, y_test = build_dataset(
        n_samples=200, n_features=100, n_informative_features=100
    )
    max_iter = 150  # 最大迭代次数

    # 在迭代次数较少的情况下，ElasticNet 可能不会收敛，这样可以加快测试速度
    clf = ElasticNetCV(
        alphas=[0.01, 0.05, 0.1], eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
    )
    ignore_warnings(clf.fit)(X, y)  # 忽略警告并拟合模型

    # 在条件良好的设置下，应该选择最小的惩罚项
    assert_almost_equal(clf.alpha_, min(clf.alphas_))

    # 非稀疏的真实情况：应该选择一个更接近岭回归而不是 Lasso 的 ElasticNet
    assert clf.l1_ratio_ == min(clf.l1_ratio)

    clf = ElasticNetCV(
        alphas=[0.01, 0.05, 0.1],
        eps=2e-3,
        l1_ratio=[0.5, 0.7],
        cv=3,
        max_iter=max_iter,
        precompute=True,
    )
    ignore_warnings(clf.fit)(X, y)

    # 在条件良好的设置下，应该选择最小的惩罚项
    assert_almost_equal(clf.alpha_, min(clf.alphas_))

    # 非稀疏的真实情况：应该选择一个更接近岭回归而不是 Lasso 的 ElasticNet
    assert clf.l1_ratio_ == min(clf.l1_ratio)

    # 在条件良好的设置下，低噪声情况下应该有良好的测试集表现
    assert clf.score(X_test, y_test) > 0.99

    # 多输出/目标情况
    X, y, X_test, y_test = build_dataset(n_features=10, n_targets=3)
    clf = MultiTaskElasticNetCV(
        n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
    )
    ignore_warnings(clf.fit)(X, y)

    # 在条件良好的设置下，低噪声情况下应该有良好的测试集表现
    assert clf.score(X_test, y_test) > 0.99
    assert clf.coef_.shape == (3, 10)

    # 单输出情况应该在两种情况下具有相同的交叉验证 alpha 和 l1_ratio
    X, y, _, _ = build_dataset(n_features=10)
    # 创建一个 ElasticNetCV 对象，用于执行弹性网络回归和交叉验证
    clf1 = ElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    # 使用给定的输入数据 X 和目标数据 y 对 clf1 进行拟合
    clf1.fit(X, y)
    
    # 创建一个 MultiTaskElasticNetCV 对象，用于执行多任务弹性网络回归和交叉验证
    clf2 = MultiTaskElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    # 使用给定的输入数据 X 和目标数据 y 的转置（添加了一个维度）对 clf2 进行拟合
    clf2.fit(X, y[:, np.newaxis])
    
    # 断言验证 clf1 和 clf2 的 l1_ratio 参数的近似相等
    assert_almost_equal(clf1.l1_ratio_, clf2.l1_ratio_)
    # 断言验证 clf1 和 clf2 的 alpha 参数的近似相等
    assert_almost_equal(clf1.alpha_, clf2.alpha_)
# 测试路径参数的函数
def test_path_parameters():
    # 从构建的数据集中获取特征 X 和目标 y
    X, y, _, _ = build_dataset()
    # 设置最大迭代次数为 100
    max_iter = 100

    # 使用 ElasticNetCV 模型，配置参数：50 个 alpha 值，误差容限为 0.001，最大迭代次数为 max_iter，L1 比率为 0.5
    clf = ElasticNetCV(n_alphas=50, eps=1e-3, max_iter=max_iter, l1_ratio=0.5, tol=1e-3)
    # 拟合模型到数据 X, y
    clf.fit(X, y)  # new params
    # 断言检查 L1 比率是否为 0.5
    assert_almost_equal(0.5, clf.l1_ratio)
    # 断言检查 alpha 值的数量是否为 50
    assert 50 == clf.n_alphas
    # 断言检查 alphas_ 数组的长度是否为 50
    assert 50 == len(clf.alphas_)


# 测试热启动的函数
def test_warm_start():
    # 从构建的数据集中获取特征 X 和目标 y
    X, y, _, _ = build_dataset()
    # 使用 ElasticNet 模型，配置参数：alpha 值为 0.1，最大迭代次数为 5，启用热启动
    clf = ElasticNet(alpha=0.1, max_iter=5, warm_start=True)
    # 忽略警告后拟合模型到数据 X, y
    ignore_warnings(clf.fit)(X, y)
    # 再次忽略警告后拟合模型到数据 X, y，进行第二轮迭代（5 次）
    ignore_warnings(clf.fit)(X, y)  # do a second round with 5 iterations

    # 创建另一个 ElasticNet 模型，配置参数：alpha 值为 0.1，最大迭代次数为 10
    clf2 = ElasticNet(alpha=0.1, max_iter=10)
    # 忽略警告后拟合模型到数据 X, y
    ignore_warnings(clf2.fit)(X, y)
    # 断言检查两个模型的系数是否几乎相等
    assert_array_almost_equal(clf2.coef_, clf.coef_)


# 测试 Lasso 模型 alpha 参数警告的函数
def test_lasso_alpha_warning():
    # 设置特征 X 和目标 Y
    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]  # just a straight line

    # 创建 Lasso 模型，配置参数：alpha 值为 0
    clf = Lasso(alpha=0)
    # 使用 pytest 模块捕获警告信息，检查是否出现特定警告信息
    warning_message = (
        "With alpha=0, this algorithm does not "
        "converge well. You are advised to use the "
        "LinearRegression estimator"
    )
    with pytest.warns(UserWarning, match=warning_message):
        # 拟合模型到数据 X, Y
        clf.fit(X, Y)


# 测试 Lasso 模型正值约束的函数
def test_lasso_positive_constraint():
    # 设置特征 X 和目标 y
    X = [[-1], [0], [1]]
    y = [1, 0, -1]  # just a straight line with negative slope

    # 创建 Lasso 模型，配置参数：alpha 值为 0.1，启用正值约束
    lasso = Lasso(alpha=0.1, positive=True)
    # 拟合模型到数据 X, y
    lasso.fit(X, y)
    # 断言检查模型系数的最小值是否大于等于 0
    assert min(lasso.coef_) >= 0

    # 创建另一个 Lasso 模型，配置参数：alpha 值为 0.1，启用预计算，启用正值约束
    lasso = Lasso(alpha=0.1, precompute=True, positive=True)
    # 拟合模型到数据 X, y
    lasso.fit(X, y)
    # 断言检查模型系数的最小值是否大于等于 0
    assert min(lasso.coef_) >= 0


# 测试 ElasticNet 模型正值约束的函数
def test_enet_positive_constraint():
    # 设置特征 X 和目标 y
    X = [[-1], [0], [1]]
    y = [1, 0, -1]  # just a straight line with negative slope

    # 创建 ElasticNet 模型，配置参数：alpha 值为 0.1，启用正值约束
    enet = ElasticNet(alpha=0.1, positive=True)
    # 拟合模型到数据 X, y
    enet.fit(X, y)
    # 断言检查模型系数的最小值是否大于等于 0
    assert min(enet.coef_) >= 0


# 测试 ElasticNetCV 模型正值约束的函数
def test_enet_cv_positive_constraint():
    # 从构建的数据集中获取特征 X, y 和测试集 X_test, y_test
    X, y, X_test, y_test = build_dataset()
    max_iter = 500

    # 确保未约束拟合的模型具有负系数
    enetcv_unconstrained = ElasticNetCV(
        n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, n_jobs=1
    )
    enetcv_unconstrained.fit(X, y)
    # 断言检查模型系数的最小值是否小于 0
    assert min(enetcv_unconstrained.coef_) < 0

    # 在相同的数据上，约束拟合的模型具有非负系数
    enetcv_constrained = ElasticNetCV(
        n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, positive=True, n_jobs=1
    )
    enetcv_constrained.fit(X, y)
    # 断言检查模型系数的最小值是否大于等于 0
    assert min(enetcv_constrained.coef_) >= 0


# 测试多任务模型的函数，设置统一目标值
def test_uniform_targets():
    # 创建 ElasticNetCV 模型和 MultiTaskElasticNetCV 模型，以及 LassoCV 模型和 MultiTaskLassoCV 模型，各配置参数：alpha 值为 3
    enet = ElasticNetCV(n_alphas=3)
    m_enet = MultiTaskElasticNetCV(n_alphas=3)
    lasso = LassoCV(n_alphas=3)
    m_lasso = MultiTaskLassoCV(n_alphas=3)

    # 随机数生成器，种子为 0
    rng = np.random.RandomState(0)

    # 创建训练集 X_train 和测试集 X_test，均为 10 行 3 列的随机数矩阵
    X_train = rng.random_sample(size=(10, 3))
    X_test = rng.random_sample(size=(10, 3))

    # 创建 y1 和 y2，分别为大小为 10 的空数组和大小为 (10, 2) 的空数组
    y1 = np.empty(10)
    y2 = np.empty((10, 2))
    # 遍历单任务模型列表中的每个模型
    for model in models_single_task:
        # 对于每个模型，遍历 y 值为 0 和 5 的情况
        for y_values in (0, 5):
            # 将 y1 数组填充为当前的 y 值
            y1.fill(y_values)
            # 忽略收敛警告，使用 y1 数据拟合模型并预测 X_test
            with ignore_warnings(category=ConvergenceWarning):
                # 断言拟合模型在 X_test 上的预测结果与 y1 相等
                assert_array_equal(model.fit(X_train, y1).predict(X_test), y1)
            # 断言模型的 alpha 参数数组为 [np.finfo(float).resolution] * 3
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)

    # 遍历多任务模型列表中的每个模型
    for model in models_multi_task:
        # 对于每个模型，遍历 y 值为 0 和 5 的情况
        for y_values in (0, 5):
            # 将 y2 的第一列填充为当前的 y 值
            y2[:, 0].fill(y_values)
            # 将 y2 的第二列填充为当前 y 值的两倍
            y2[:, 1].fill(2 * y_values)
            # 忽略收敛警告，使用 y2 数据拟合模型并预测 X_test
            with ignore_warnings(category=ConvergenceWarning):
                # 断言拟合模型在 X_test 上的预测结果与 y2 相等
                assert_array_equal(model.fit(X_train, y2).predict(X_test), y2)
            # 断言模型的 alpha 参数数组为 [np.finfo(float).resolution] * 3
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)
def test_multi_task_lasso_and_enet():
    # 构建数据集 X, y, X_test, y_test
    X, y, X_test, y_test = build_dataset()
    # 构造 Y 矩阵，复制 y 列成为 Y 的两列
    Y = np.c_[y, y]
    # 使用 MultiTaskLasso 拟合数据
    clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
    # 断言 dual_gap_ 在 (0, 1e-5) 之间
    assert 0 < clf.dual_gap_ < 1e-5
    # 断言两个 coef_ 结果几乎相等
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

    # 使用 MultiTaskElasticNet 拟合数据
    clf = MultiTaskElasticNet(alpha=1, tol=1e-8).fit(X, Y)
    # 断言 dual_gap_ 在 (0, 1e-5) 之间
    assert 0 < clf.dual_gap_ < 1e-5
    # 断言两个 coef_ 结果几乎相等
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

    # 使用带有 max_iter=1 的 MultiTaskElasticNet 拟合数据，并发出警告
    clf = MultiTaskElasticNet(alpha=1.0, tol=1e-8, max_iter=1)
    warning_message = (
        "Objective did not converge. You might want to "
        "increase the number of iterations."
    )
    # 使用 pytest.warns 检查是否发出预期的 ConvergenceWarning 警告
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, Y)


def test_lasso_readonly_data():
    # 构建简单的数据集 X, Y, T
    X = np.array([[-1], [0], [1]])
    Y = np.array([-1, 0, 1])  # 简单的直线
    T = np.array([[2], [3], [4]])  # 测试样本
    # 使用临时内存映射（TempMemmap）处理数据 X, Y
    with TempMemmap((X, Y)) as (X, Y):
        # 使用 Lasso 拟合数据
        clf = Lasso(alpha=0.5)
        clf.fit(X, Y)
        # 预测 T 的结果
        pred = clf.predict(T)
        # 断言 coef_ 几乎等于 [0.25]
        assert_array_almost_equal(clf.coef_, [0.25])
        # 断言预测结果与预期几乎相等
        assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
        # 断言 dual_gap_ 几乎为 0
        assert_almost_equal(clf.dual_gap_, 0)


def test_multi_task_lasso_readonly_data():
    # 构建数据集 X, y, X_test, y_test
    X, y, X_test, y_test = build_dataset()
    # 构造 Y 矩阵，复制 y 列成为 Y 的两列
    Y = np.c_[y, y]
    # 使用临时内存映射（TempMemmap）处理数据 X, Y
    with TempMemmap((X, Y)) as (X, Y):
        # 使用 MultiTaskLasso 拟合数据
        clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
        # 断言 dual_gap_ 在 (0, 1e-5) 之间
        assert 0 < clf.dual_gap_ < 1e-5
        # 断言两个 coef_ 结果几乎相等
        assert_array_almost_equal(clf.coef_[0], clf.coef_[1])


def test_enet_multitarget():
    n_targets = 3
    # 构建数据集 X, y
    X, y, _, _ = build_dataset(
        n_samples=10, n_features=8, n_informative_features=10, n_targets=n_targets
    )
    # 使用 ElasticNet 拟合数据
    estimator = ElasticNet(alpha=0.01)
    estimator.fit(X, y)
    # 获取 coef_, intercept_, dual_gap_
    coef, intercept, dual_gap = (
        estimator.coef_,
        estimator.intercept_,
        estimator.dual_gap_,
    )

    # 遍历每个目标，分别拟合并断言结果
    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        # 断言 coef_ 几乎相等
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        # 断言 intercept_ 几乎相等
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        # 断言 dual_gap_ 几乎相等
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)


def test_multioutput_enetcv_error():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    y = rng.randn(10, 2)
    # 使用 ElasticNetCV 拟合数据，预期会引发 ValueError
    clf = ElasticNetCV()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_multitask_enet_and_lasso_cv():
    # 构建数据集 X, y
    X, y, _, _ = build_dataset(n_features=50, n_targets=3)
    # 使用 MultiTaskElasticNetCV 拟合数据
    clf = MultiTaskElasticNetCV(cv=3).fit(X, y)
    # 断言 alpha_ 几乎等于 0.00556
    assert_almost_equal(clf.alpha_, 0.00556, 3)
    # 使用 MultiTaskLassoCV 拟合数据
    clf = MultiTaskLassoCV(cv=3).fit(X, y)
    # 断言 alpha_ 几乎等于 0.00278
    assert_almost_equal(clf.alpha_, 0.00278, 3)

    # 重新构建数据集 X, y
    X, y, _, _ = build_dataset(n_targets=3)
    # 使用 MultiTaskElasticNetCV 拟合数据
    clf = MultiTaskElasticNetCV(
        n_alphas=10, eps=1e-3, max_iter=200, l1_ratio=[0.3, 0.5], tol=1e-3, cv=3
    )
    clf.fit(X, y)
    # 断言 l1_ratio_ 等于 0.5
    assert 0.5 == clf.l1_ratio_
    # 断言 coef_ 的形状为 (3, X.shape[1])
    assert (3, X.shape[1]) == clf.coef_.shape
    # 断言 intercept_ 的形状为 (3,)
    assert (3,) == clf.intercept_.shape
    # 断言 mse_path_ 的形状为 (2, 10, 3)
    assert (2, 10, 3) == clf.mse_path_.shape
    # 确保模型训练后的拉格朗日乘子形状为 (2, 10)
    assert (2, 10) == clf.alphas_.shape
    
    # 构建包含三个目标的数据集 X 和标签 y
    X, y, _, _ = build_dataset(n_targets=3)
    # 初始化 MultiTaskLassoCV 模型，设置参数：10 个拉格朗日乘子、收敛阈值 1e-3、最大迭代次数 500、容忍度 1e-3、交叉验证折数 3
    clf = MultiTaskLassoCV(n_alphas=10, eps=1e-3, max_iter=500, tol=1e-3, cv=3)
    # 使用数据集 X 和标签 y 训练模型
    clf.fit(X, y)
    
    # 确保模型得到的系数形状为 (3, X.shape[1])
    assert (3, X.shape[1]) == clf.coef_.shape
    # 确保模型得到的截距形状为 (3,)
    assert (3,) == clf.intercept_.shape
    # 确保模型的均方误差路径形状为 (10, 3)
    assert (10, 3) == clf.mse_path_.shape
    # 确保模型使用的拉格朗日乘子数量为 10
    assert 10 == len(clf.alphas_)
# 测试使用 ElasticNetCV 和 MultiTaskElasticNetCV 对单输出和多输出回归进行交叉验证

def test_1d_multioutput_enet_and_multitask_enet_cv():
    # 构建包含10个特征的数据集
    X, y, _, _ = build_dataset(n_features=10)
    # 将 y 调整为二维数组
    y = y[:, np.newaxis]
    
    # 使用 ElasticNetCV 模型进行交叉验证
    clf = ElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf.fit(X, y[:, 0])  # 对第一个输出进行拟合

    # 使用 MultiTaskElasticNetCV 模型进行交叉验证
    clf1 = MultiTaskElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)  # 对多个输出进行拟合

    # 检查两个模型的 L1 ratio 是否接近
    assert_almost_equal(clf.l1_ratio_, clf1.l1_ratio_)
    # 检查两个模型的 alpha 是否接近
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    # 检查两个模型第一个输出的系数是否接近
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    # 检查两个模型第一个输出的截距是否接近
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])


# 测试使用 LassoCV 和 MultiTaskLassoCV 对单输出和多输出回归进行交叉验证

def test_1d_multioutput_lasso_and_multitask_lasso_cv():
    # 构建包含10个特征的数据集
    X, y, _, _ = build_dataset(n_features=10)
    # 将 y 调整为二维数组
    y = y[:, np.newaxis]
    
    # 使用 LassoCV 模型进行交叉验证
    clf = LassoCV(n_alphas=5, eps=2e-3)
    clf.fit(X, y[:, 0])  # 对第一个输出进行拟合

    # 使用 MultiTaskLassoCV 模型进行交叉验证
    clf1 = MultiTaskLassoCV(n_alphas=5, eps=2e-3)
    clf1.fit(X, y)  # 对多个输出进行拟合

    # 检查两个模型的 alpha 是否接近
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    # 检查两个模型第一个输出的系数是否接近
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    # 检查两个模型第一个输出的截距是否接近
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])


# 使用不同的稀疏矩阵容器进行 ElasticNetCV 和 LassoCV 的交叉验证

@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_dtype_enet_and_lassocv(csr_container):
    # 构建包含10个特征的数据集
    X, y, _, _ = build_dataset(n_features=10)
    
    # 使用 ElasticNetCV 模型进行交叉验证
    clf = ElasticNetCV(n_alphas=5)
    clf.fit(csr_container(X), y)

    # 使用 ElasticNetCV 模型进行交叉验证（使用 np.float32 类型的数据）
    clf1 = ElasticNetCV(n_alphas=5)
    clf1.fit(csr_container(X, dtype=np.float32), y)

    # 检查两个模型的 alpha 是否非常接近
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    # 检查两个模型的系数是否非常接近
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)

    # 使用 LassoCV 模型进行交叉验证
    clf = LassoCV(n_alphas=5)
    clf.fit(csr_container(X), y)

    # 使用 LassoCV 模型进行交叉验证（使用 np.float32 类型的数据）
    clf1 = LassoCV(n_alphas=5)
    clf1.fit(csr_container(X, dtype=np.float32), y)

    # 检查两个模型的 alpha 是否非常接近
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    # 检查两个模型的系数是否非常接近
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)


# 测试 ElasticNet 模型在传入错误的 Gram 矩阵时是否会抛出异常

def test_elasticnet_precompute_incorrect_gram():
    # 使用 build_dataset() 构建数据集
    X, y, _, _ = build_dataset()

    # 使用 np.random.RandomState(0) 创建随机数生成器
    rng = np.random.RandomState(0)

    # 对 X 进行中心化处理
    X_centered = X - np.average(X, axis=0)

    # 生成一个与 X 维度相同的标准正态分布随机矩阵
    garbage = rng.standard_normal(X.shape)

    # 计算垃圾矩阵的转置与其本身的乘积，得到一个“预计算”的 Gram 矩阵
    precompute = np.dot(garbage.T, garbage)

    # 使用 ElasticNet 模型，传入预计算的 Gram 矩阵
    clf = ElasticNet(alpha=0.01, precompute=precompute)

    # 预期会抛出 ValueError 异常，匹配特定的错误信息
    msg = "Gram matrix.*did not pass validation.*"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X_centered, y)


# 测试 ElasticNet 模型在传入预计算的 Gram 矩阵与使用样本权重进行训练时的等效性

def test_elasticnet_precompute_gram_weighted_samples():
    # 使用 build_dataset() 构建数据集
    X, y, _, _ = build_dataset()

    # 使用 np.random.RandomState(0) 创建随机数生成器
    rng = np.random.RandomState(0)

    # 生成样本权重，服从对数正态分布
    sample_weight = rng.lognormal(size=y.shape)

    # 计算加权样本权重
    w_norm = sample_weight * (y.shape / np.sum(sample_weight))

    # 对 X 进行加权中心化处理
    X_c = X - np.average(X, axis=0, weights=w_norm)

    # 对加权中心化后的 X 进行加权的标准化处理
    X_r = X_c * np.sqrt(w_norm)[:, np.newaxis]

    # 计算加权后的 Gram 矩阵
    gram = np.dot(X_r.T, X_r)

    # 使用 ElasticNet 模型，传入预计算的 Gram 矩阵进行训练
    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    clf1.fit(X_c, y, sample_weight=sample_weight)

    # 使用 ElasticNet 模型，传入默认参数进行训练（内部计算 Gram 矩阵）
    clf2 = ElasticNet(alpha=0.01, precompute=False)
    clf2.fit(X, y, sample_weight=sample_weight)

    # 检查两个模型的系数是否非常接近
    assert_allclose(clf1.coef_, clf2.coef_)


# 测试 ElasticNet 模型在传入预计算的 Gram 矩阵时的行为

def test_elasticnet_precompute_gram():
    # 此测试未提供完整的代码，无法继续添加注释
    pass
    # 使用随机种子为58创建一个随机数生成器对象
    rng = np.random.RandomState(58)
    # 生成一个形状为(1000, 4)的二项分布随机数组，元素类型转换为np.float32
    X = rng.binomial(1, 0.25, (1000, 4)).astype(np.float32)
    # 生成一个长度为1000的随机数组，元素类型转换为np.float32
    y = rng.rand(1000).astype(np.float32)

    # 对 X 进行中心化处理，即每列减去其均值
    X_c = X - np.average(X, axis=0)
    # 计算中心化后 X 的转置与 X_c 的矩阵乘积，得到 Gram 矩阵
    gram = np.dot(X_c.T, X_c)

    # 使用 ElasticNet 模型，设置预计算的 Gram 矩阵作为输入
    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    # 使用中心化后的 X_c 和 y 进行模型拟合
    clf1.fit(X_c, y)

    # 使用 ElasticNet 模型，关闭预计算选项
    clf2 = ElasticNet(alpha=0.01, precompute=False)
    # 使用原始的 X 和 y 进行模型拟合
    clf2.fit(X, y)

    # 检查 clf1 和 clf2 的系数是否在数值上接近
    assert_allclose(clf1.coef_, clf2.coef_)
# 定义一个测试函数，用于验证模型在热启动条件下的收敛性
def test_warm_start_convergence():
    # 构建数据集 X, y，并忽略其他返回值
    X, y, _, _ = build_dataset()
    
    # 使用 ElasticNet 模型进行拟合，设置 alpha 和 tol 参数
    model = ElasticNet(alpha=1e-3, tol=1e-3).fit(X, y)
    
    # 获取模型的迭代次数 n_iter_
    n_iter_reference = model.n_iter_

    # 断言：数据集不够简单，模型需要超过两次迭代才能收敛
    assert n_iter_reference > 2

    # 检查在 warm_start=False 的条件下，多次调用 fit 方法不改变 n_iter_
    model.fit(X, y)
    n_iter_cold_start = model.n_iter_
    assert n_iter_cold_start == n_iter_reference

    # 再次拟合相同的模型，使用热启动（warm start）：优化器只执行一次迭代，然后检查是否已经收敛
    model.set_params(warm_start=True)
    model.fit(X, y)
    n_iter_warm_start = model.n_iter_
    assert n_iter_warm_start == 1


# 定义一个测试函数，验证在降低正则化的情况下，模型的热启动收敛性
def test_warm_start_convergence_with_regularizer_decrement():
    # 加载糖尿病数据集，返回 X 和 y
    X, y = load_diabetes(return_X_y=True)

    # 训练一个在轻度正则化问题上收敛的模型
    final_alpha = 1e-5
    low_reg_model = ElasticNet(alpha=final_alpha).fit(X, y)

    # 使用更高正则化系数重新训练模型，预期它会更快地收敛
    high_reg_model = ElasticNet(alpha=final_alpha * 10).fit(X, y)
    assert low_reg_model.n_iter_ > high_reg_model.n_iter_

    # 使用高度正则化问题的解作为起始点，重新拟合原始、正则化较低的问题
    # 预期此方法比从零开始的原始模型更快收敛
    warm_low_reg_model = deepcopy(high_reg_model)
    warm_low_reg_model.set_params(warm_start=True, alpha=final_alpha)
    warm_low_reg_model.fit(X, y)
    assert low_reg_model.n_iter_ > warm_low_reg_model.n_iter_


# 使用参数化测试，验证随机和循环选择在不同条件下是否给出相同结果
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_random_descent(csr_container):
    # 测试随机选择和循环选择是否给出相同结果
    # 确保测试模型完全收敛，并检查多种条件

    # 使用 gram 矩阵技巧的坐标下降算法
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
    clf_cyclic.fit(X, y)
    clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
    clf_random.fit(X, y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

    # 不使用 gram 矩阵技巧的下降算法
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
    clf_cyclic.fit(X.T, y[:20])
    clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
    clf_random.fit(X.T, y[:20])
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

    # 稀疏情况
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
    # 使用循环选择的 MultiTaskElasticNet 拟合模型，对数据进行训练
    clf_cyclic.fit(csr_container(X), y)
    # 创建一个以随机选择特性的 ElasticNet 模型，设置了随机种子
    clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
    # 使用随机选择的 ElasticNet 拟合模型，对数据进行训练
    clf_random.fit(csr_container(X), y)
    # 断言两个模型的系数近乎相等
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    # 断言两个模型的截距近乎相等
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

    # 多输出情况下的测试。
    # 创建一个新的目标变量，将原始目标变量按列堆叠成两列
    new_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
    # 使用循环选择的 MultiTaskElasticNet 拟合模型，对数据进行训练
    clf_cyclic = MultiTaskElasticNet(selection="cyclic", tol=1e-8)
    clf_cyclic.fit(X, new_y)
    # 创建一个以随机选择特性的 MultiTaskElasticNet 模型，设置了随机种子
    clf_random = MultiTaskElasticNet(selection="random", tol=1e-8, random_state=42)
    # 使用随机选择的 MultiTaskElasticNet 拟合模型，对数据进行训练
    clf_random.fit(X, new_y)
    # 断言两个模型的系数近乎相等
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    # 断言两个模型的截距近乎相等
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)
# 测试正参数情况

def test_enet_path_positive():
    # 构建包含50个样本和50个特征的数据集，并获取其中的输入X和输出Y
    X, Y, _, _ = build_dataset(n_samples=50, n_features=50, n_targets=2)

    # 对于单一输出情况
    # 测试在enet_path中positive=True时，返回的系数是否为正数
    for path in [enet_path, lasso_path]:
        # 调用路径方法（enet_path或lasso_path），并设置positive=True，获取第二个返回值（系数）
        pos_path_coef = path(X, Y[:, 0], positive=True)[1]
        # 断言所有系数均大于等于0
        assert np.all(pos_path_coef >= 0)

    # 对于多输出情况，不允许设置positive参数
    # 测试是否会触发错误
    for path in [enet_path, lasso_path]:
        with pytest.raises(ValueError):
            # 调用路径方法（enet_path或lasso_path），并设置positive=True，应该抛出异常
            path(X, Y, positive=True)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_dense_descent_paths(csr_container):
    # 测试稠密和稀疏输入是否产生相同的下降路径
    # 构建包含50个样本和20个特征的数据集，并获取其中的输入X和输出y
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    # 使用csr_container将X转换为稀疏格式
    csr = csr_container(X)
    for path in [enet_path, lasso_path]:
        # 调用路径方法（enet_path或lasso_path），获取路径、系数和迭代次数
        _, coefs, _ = path(X, y)
        _, sparse_coefs, _ = path(csr, y)
        # 断言稠密和稀疏格式的系数近似相等
        assert_array_almost_equal(coefs, sparse_coefs)


@pytest.mark.parametrize("path_func", [enet_path, lasso_path])
def test_path_unknown_parameter(path_func):
    """检查传递给坐标下降求解器未使用的参数是否会引发错误。"""
    # 构建包含50个样本和20个特征的数据集，并获取其中的输入X和输出y
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    err_msg = "Unexpected parameters in params"
    with pytest.raises(ValueError, match=err_msg):
        # 调用路径函数（enet_path或lasso_path），并传递normalize=True和fit_intercept=True，应该抛出异常
        path_func(X, y, normalize=True, fit_intercept=True)


def test_check_input_false():
    # 构建包含20个样本和10个特征的数据集，并获取其中的输入X和输出y
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    # 检查并确保X是按照"F"顺序排列且数据类型为"float64"
    X = check_array(X, order="F", dtype="float64")
    # 检查并确保y是按照"F"顺序排列且数据类型为"float64"
    y = check_array(X, order="F", dtype="float64")
    # 初始化ElasticNet回归器，设置selection="cyclic"和tol=1e-8
    clf = ElasticNet(selection="cyclic", tol=1e-8)
    # 检查当输入数据格式正确时，是否不会引发错误
    clf.fit(X, y, check_input=False)
    # 当check_input=False时，不会对y进行详尽检查，但其数据类型仍然在_preprocess_data中转换为X的数据类型，所以测试应该通过
    X = check_array(X, order="F", dtype="float32")
    with ignore_warnings(category=ConvergenceWarning):
        # 当没有输入检查时，提供按照"C"顺序排列的X应该导致错误的计算
        X = check_array(X, order="C", dtype="float64")
        with pytest.raises(ValueError):
            clf.fit(X, y, check_input=False)


@pytest.mark.parametrize("check_input", [True, False])
def test_enet_copy_X_True(check_input):
    # 构建数据集并获取输入X和输出y
    X, y, _, _ = build_dataset()
    # 拷贝X并按照"F"顺序排列
    X = X.copy(order="F")

    original_X = X.copy()
    # 初始化ElasticNet回归器，设置copy_X=True
    enet = ElasticNet(copy_X=True)
    # 拟合模型并检查是否不会改变原始数据X
    enet.fit(X, y, check_input=check_input)

    # 断言拟合后的X和原始的X相等
    assert_array_equal(original_X, X)


def test_enet_copy_X_False_check_input_False():
    # 构建数据集并获取输入X和输出y
    X, y, _, _ = build_dataset()
    # 拷贝X并按照"F"顺序排列
    X = X.copy(order="F")

    original_X = X.copy()
    # 初始化ElasticNet回归器，设置copy_X=False
    enet = ElasticNet(copy_X=False)
    # 拟合模型并检查是否不会改变原始数据X
    enet.fit(X, y, check_input=False)

    # 断言至少有一个元素不相等，即X没有被复制，而是被覆盖
    assert np.any(np.not_equal(original_X, X)))
def test`
def test_overrided_gram_matrix():
    # 构建一个包含 20 个样本和 10 个特征的数据集
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    # 计算 X 的转置与 X 的乘积，得到 Gram 矩阵
    Gram = X.T.dot(X)
    # 创建一个 ElasticNet 模型实例，设置选择方法为 "cyclic"，容忍度为 1e-8，并使用 Gram 矩阵进行预计算
    clf = ElasticNet(selection="cyclic", tol=1e-8, precompute=Gram)
    # 定义警告信息，表示提供了 Gram 矩阵但 X 已经中心化，拟合截距时将重新计算 Gram 矩阵
    warning_message = (
        "Gram matrix was provided but X was centered"
        " to fit intercept: recomputing Gram matrix."
    )
    # 使用 pytest 检测是否抛出 UserWarning 类型的警告，并匹配警告信息
    with pytest.warns(UserWarning, match=warning_message):
        clf.fit(X, y)

@pytest.mark.parametrize("model", [ElasticNet, Lasso])
def test_lasso_non_float_y(model):
    # 创建样本数据 X 和 y，y 包含整数值
    X = [[0, 0], [1, 1], [-1, -1]]
    y = [0, 1, 2]
    y_float = [0.0, 1.0, 2.0]

    # 使用指定模型创建并训练模型 clf
    clf = model(fit_intercept=False)
    clf.fit(X, y)
    # 使用相同模型和数据类型为浮点数 y 创建并训练模型 clf_float
    clf_float = model(fit_intercept=False)
    clf_float.fit(X, y_float)
    # 验证 clf 和 clf_float 的系数是否相等
    assert_array_equal(clf.coef_, clf_float.coef_)

def test_enet_float_precision():
    # 生成一个包含 20 个样本和 10 个特征的数据集，以及测试集数据
    X, y, X_test, y_test = build_dataset(n_samples=20, n_features=10)
    # 设置少量迭代次数，确保 ElasticNet 模型能够快速完成测试而不至于未收敛

    for fit_intercept in [True, False]:
        coef = {}
        intercept = {}
        # 测试不同数据类型 np.float64 和 np.float32
        for dtype in [np.float64, np.float32]:
            clf = ElasticNet(
                alpha=0.5,
                max_iter=100,
                precompute=False,
                fit_intercept=fit_intercept,
            )

            # 将 X 和 y 转换为指定数据类型 dtype
            X = dtype(X)
            y = dtype(y)
            ignore_warnings(clf.fit)(X, y)

            coef[("simple", dtype)] = clf.coef_
            intercept[("simple", dtype)] = clf.intercept_

            # 验证模型系数的数据类型是否为 dtype
            assert clf.coef_.dtype == dtype

            # 测试使用 Gram 矩阵进行预计算
            Gram = X.T.dot(X)
            clf_precompute = ElasticNet(
                alpha=0.5,
                max_iter=100,
                precompute=Gram,
                fit_intercept=fit_intercept,
            )
            ignore_warnings(clf_precompute.fit)(X, y)
            # 验证两次训练的系数和截距是否几乎相等
            assert_array_almost_equal(clf.coef_, clf_precompute.coef_)
            assert_array_almost_equal(clf.intercept_, clf_precompute.intercept_)

            # 测试多任务 ElasticNet
            multi_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
            clf_multioutput = MultiTaskElasticNet(
                alpha=0.5,
                max_iter=100,
                fit_intercept=fit_intercept,
            )
            clf_multioutput.fit(X, multi_y)
            coef[("multi", dtype)] = clf_multioutput.coef_
            intercept[("multi", dtype)] = clf_multioutput.intercept_
            assert clf.coef_.dtype == dtype

        # 验证不同数据类型的系数和截距是否相等，精度为小数点后四位
        for v in ["simple", "multi"]:
            assert_array_almost_equal(
                coef[(v, np.float32)], coef[(v, np.float64)], decimal=4
            )
            assert_array_almost_equal(
                intercept[(v, np.float32)], intercept[(v, np.float64)], decimal=4
            )

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_enet_l1_ratio():
    # 测试在使用带有 L1 正则化比率的估计器时，是否抛出错误信息
    # 当 l1_ratio=0 时，禁止自动生成 alpha 网格
    msg = (
        "Automatic alpha grid generation is not supported for l1_ratio=0. "
        "Please supply a grid by providing your estimator with the "
        "appropriate `alphas=` argument."
    )
    # 创建输入特征 X 和目标变量 y 的 NumPy 数组
    X = np.array([[1, 2, 4, 5, 8], [3, 5, 7, 7, 8]]).T
    y = np.array([12, 10, 11, 21, 5])

    # 测试 ElasticNetCV 在 l1_ratio=0 时是否会引发 ValueError，并匹配特定错误信息 msg
    with pytest.raises(ValueError, match=msg):
        ElasticNetCV(l1_ratio=0, random_state=42).fit(X, y)

    # 测试 MultiTaskElasticNetCV 在 l1_ratio=0 时是否会引发 ValueError，并匹配特定错误信息 msg
    with pytest.raises(ValueError, match=msg):
        MultiTaskElasticNetCV(l1_ratio=0, random_state=42).fit(X, y[:, None])

    # 测试当 l1_ratio=0 且 alpha>0 时是否会产生用户警告
    warning_message = (
        "Coordinate descent without L1 regularization may "
        "lead to unexpected results and is discouraged. "
        "Set l1_ratio > 0 to add L1 regularization."
    )
    # 创建 ElasticNetCV 对象，指定 l1_ratio=[0] 和 alphas=[1]
    est = ElasticNetCV(l1_ratio=[0], alphas=[1])
    # 测试是否会产生用户警告，匹配特定警告信息 warning_message
    with pytest.warns(UserWarning, match=warning_message):
        est.fit(X, y)

    # 测试当 l1_ratio=0 时，如果手动提供网格是否允许
    alphas = [0.1, 10]
    estkwds = {"alphas": alphas, "random_state": 42}
    # 创建期望的 ElasticNetCV 对象，指定 l1_ratio=0.00001 和 estkwds 参数
    est_desired = ElasticNetCV(l1_ratio=0.00001, **estkwds)
    # 创建 ElasticNetCV 对象，指定 l1_ratio=0 和 estkwds 参数
    est = ElasticNetCV(l1_ratio=0, **estkwds)
    # 忽略警告进行拟合
    with ignore_warnings():
        est_desired.fit(X, y)
        est.fit(X, y)
    # 断言两个对象的系数数组是否几乎相等，精确度为小数点后五位
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)

    # 创建期望的 MultiTaskElasticNetCV 对象，指定 l1_ratio=0.00001 和 estkwds 参数
    est_desired = MultiTaskElasticNetCV(l1_ratio=0.00001, **estkwds)
    # 创建 MultiTaskElasticNetCV 对象，指定 l1_ratio=0 和 estkwds 参数
    est = MultiTaskElasticNetCV(l1_ratio=0, **estkwds)
    # 忽略警告进行拟合
    with ignore_warnings():
        est.fit(X, y[:, None])
        est_desired.fit(X, y[:, None])
    # 断言两个对象的系数数组是否几乎相等，精确度为小数点后五位
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)
# 测试确保没有截距的 Lasso 回归器的系数形状为 (1,)
def test_coef_shape_not_zero():
    # 创建一个没有截距的 Lasso 回归器实例
    est_no_intercept = Lasso(fit_intercept=False)
    # 使用 np.c_ 将输入数组转换为列堆叠，拟合回归器
    est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
    # 断言回归器的系数形状是否为 (1,)
    assert est_no_intercept.coef_.shape == (1,)


# 测试 MultiTaskLasso 回归器的热启动功能
def test_warm_start_multitask_lasso():
    # 构建数据集
    X, y, X_test, y_test = build_dataset()
    # 创建一个多任务 Lasso 回归器实例，设置 alpha、max_iter 和 warm_start 参数
    clf = MultiTaskLasso(alpha=0.1, max_iter=5, warm_start=True)
    # 忽略警告，拟合回归器并执行第一轮训练
    ignore_warnings(clf.fit)(X, np.c_[y, y])
    # 再次拟合回归器，执行第二轮训练
    ignore_warnings(clf.fit)(X, np.c_[y, y])
    
    # 创建另一个 MultiTaskLasso 回归器实例，设置不同的 max_iter 参数
    clf2 = MultiTaskLasso(alpha=0.1, max_iter=10)
    # 忽略警告，拟合回归器
    ignore_warnings(clf2.fit)(X, np.c_[y, y])
    # 断言两个回归器的系数是否几乎相等
    assert_array_almost_equal(clf2.coef_, clf.coef_)


# 使用参数化测试，测试 ElasticNet 和 Lasso 模型的坐标下降算法
@pytest.mark.parametrize(
    "klass, n_classes, kwargs",
    [
        (Lasso, 1, dict(precompute=True)),
        (Lasso, 1, dict(precompute=False)),
    ],
)
def test_enet_coordinate_descent(klass, n_classes, kwargs):
    """测试模型在不收敛时是否发出警告"""
    # 创建指定类别、参数的回归器实例
    clf = klass(max_iter=2, **kwargs)
    n_samples = 5
    n_features = 2
    # 创建具有极大值的输入数据
    X = np.ones((n_samples, n_features)) * 1e50
    y = np.ones((n_samples, n_classes))
    if klass == Lasso:
        y = y.ravel()
    # 设置警告消息内容
    warning_message = (
        "Objective did not converge. You might want to"
        " increase the number of iterations."
    )
    # 使用 pytest 捕获警告，检查警告消息是否匹配
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, y)


# 测试确保模型在无收敛警告的情况下正常运行
def test_convergence_warnings():
    # 创建随机种子
    random_state = np.random.RandomState(0)
    # 创建标准正态分布的随机数据集
    X = random_state.standard_normal((1000, 500))
    y = random_state.standard_normal((1000, 3))

    # 检查模型在没有收敛警告的情况下是否收敛
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        MultiTaskElasticNet().fit(X, y)


# 使用参数化测试，测试稀疏输入在不收敛时是否发出警告
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_convergence_warning(csr_container):
    # 构建数据集并获取稀疏矩阵
    X, y, _, _ = build_dataset(n_samples=1000, n_features=500)

    # 使用 pytest 捕获警告，检查警告消息是否匹配
    with pytest.warns(ConvergenceWarning):
        ElasticNet(max_iter=1, tol=0).fit(csr_container(X, dtype=np.float32), y)

    # 检查模型在没有收敛警告的情况下是否收敛
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        Lasso().fit(csr_container(X, dtype=np.float32), y)


# 使用参数化测试，测试 LassoCV 模型在设置 precompute 参数时的行为
@pytest.mark.parametrize(
    "precompute, inner_precompute",
    [
        (True, True),
        ("auto", False),
        (False, False),
    ],
)
def test_lassoCV_does_not_set_precompute(monkeypatch, precompute, inner_precompute):
    # 构建数据集
    X, y, _, _ = build_dataset()
    calls = 0

    # 创建自定义的 LassoMock 类，用于检查 precompute 参数
    class LassoMock(Lasso):
        def fit(self, X, y):
            super().fit(X, y)
            nonlocal calls
            calls += 1
            assert self.precompute == inner_precompute

    # 使用 monkeypatch 替换 sklearn.linear_model._coordinate_descent.Lasso 为 LassoMock
    monkeypatch.setattr("sklearn.linear_model._coordinate_descent.Lasso", LassoMock)
    # 创建 LassoCV 回归器实例
    clf = LassoCV(precompute=precompute)
    # 拟合回归器
    clf.fit(X, y)
    # 断言至少进行了一次拟合
    assert calls > 0


# 测试 MultiTaskLassoCV 回归器的 dtype 设置
def test_multi_task_lasso_cv_dtype():
    n_samples, n_features = 10, 3
    rng = np.random.RandomState(42)
    # 使用随机数生成器 rng 生成一个大小为 (n_samples, n_features) 的二项分布随机数组 X
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    # 将 X 的数据类型显式地转换为整数，明确表示 X 是整数类型
    X = X.astype(int)
    # 创建数组 y，复制 X 的第一列作为 y 的两列数据
    y = X[:, [0, 0]].copy()
    # 使用 MultiTaskLassoCV 进行交叉验证，估计参数，使用 5 个 alpha 值，拟合时包含截距项
    est = MultiTaskLassoCV(n_alphas=5, fit_intercept=True).fit(X, y)
    # 断言估计的系数 est.coef_ 与 [[1, 0, 0]] * 2 几乎相等，精确到小数点后三位
    assert_array_almost_equal(est.coef_, [[1, 0, 0]] * 2, decimal=3)
# 使用 pytest.mark.parametrize 装饰器为下面的测试函数提供多组参数组合
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [0.01])
@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
# 定义测试函数，测试样本权重对结果的一致性影响
def test_enet_sample_weight_consistency(
    fit_intercept, alpha, precompute, sparse_container, global_random_seed
):
    """Test that the impact of sample_weight is consistent.

    Note that this test is stricter than the common test
    check_sample_weights_invariance alone and also tests sparse X.
    """
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 设置样本数和特征数
    n_samples, n_features = 10, 5

    # 生成随机的样本数据 X 和 y
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    # 如果 sparse_container 不为空，将 X 转换为稀疏格式
    if sparse_container is not None:
        X = sparse_container(X)
    
    # 设置 ElasticNet 模型的参数
    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        precompute=precompute,
        tol=1e-6,
        l1_ratio=0.5,
    )

    # 创建 ElasticNet 模型并拟合数据
    reg = ElasticNet(**params).fit(X, y)
    # 备份模型的系数
    coef = reg.coef_.copy()
    # 如果 fit_intercept 为 True，备份模型的截距
    if fit_intercept:
        intercept = reg.intercept_

    # 1) sample_weight=np.ones(..) should be equivalent to sample_weight=None
    # 使用相同形状的全为1的 sample_weight 拟合模型，与不使用 sample_weight 应该等效
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 2) sample_weight=None should be equivalent to sample_weight = number
    # 使用数值作为 sample_weight 拟合模型，与不使用 sample_weight 应该等效
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 3) scaling of sample_weight should have no effect, cf. np.average()
    # 对 sample_weight 进行缩放，不应该影响结果，类似于 np.average() 的效果
    sample_weight = rng.uniform(low=0.01, high=2, size=X.shape[0])
    reg = reg.fit(X, y, sample_weight=sample_weight)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_

    # 再次使用缩放后的 sample_weight 拟合模型，结果应该与之前相同
    reg.fit(X, y, sample_weight=np.pi * sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 4) setting elements of sample_weight to 0 is equivalent to removing these samples
    # 将 sample_weight 中的部分元素设为0，等效于移除这些样本
    sample_weight_0 = sample_weight.copy()
    sample_weight_0[-5:] = 0
    y[-5:] *= 1000  # to make excluding those samples important
    # 使用处理后的 sample_weight_0 拟合模型，再与移除相应样本后的拟合结果进行比较
    reg.fit(X, y, sample_weight=sample_weight_0)
    coef_0 = reg.coef_.copy()
    if fit_intercept:
        intercept_0 = reg.intercept_
    reg.fit(X[:-5], y[:-5], sample_weight=sample_weight[:-5])
    assert_allclose(reg.coef_, coef_0, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept_0)

    # 5) check that multiplying sample_weight by 2 is equivalent to repeating
    # corresponding samples twice
    # 检查将 sample_weight 乘以2后，是否等效于相应样本重复两次
    if sparse_container is not None:
        X2 = sparse.vstack([X, X[: n_samples // 2]], format="csc")
    else:
        X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]])
    sample_weight_1 = sample_weight.copy()
    # 将样本权重向量的前一半元素乘以2，增加这些样本的重要性
    sample_weight_1[: n_samples // 2] *= 2
    
    # 将两个样本权重向量拼接起来，以处理两个不同的数据集
    sample_weight_2 = np.concatenate(
        [sample_weight, sample_weight[: n_samples // 2]], axis=0
    )
    
    # 使用给定参数初始化并训练第一个弹性网络模型，其中考虑了样本权重
    reg1 = ElasticNet(**params).fit(X, y, sample_weight=sample_weight_1)
    
    # 使用给定参数初始化并训练第二个弹性网络模型，其中考虑了样本权重
    reg2 = ElasticNet(**params).fit(X2, y2, sample_weight=sample_weight_2)
    
    # 断言第一个模型和第二个模型的系数非常接近，使用相对误差容差为1e-6
    assert_allclose(reg1.coef_, reg2.coef_, rtol=1e-6)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS)
def test_enet_cv_sample_weight_correctness(fit_intercept, sparse_container):
    """Test that ElasticNetCV with sample weights gives correct results."""
    rng = np.random.RandomState(42)
    n_splits, n_samples, n_features = 3, 10, 5
    X = rng.rand(n_splits * n_samples, n_features)
    beta = rng.rand(n_features)
    beta[0:2] = 0
    y = X @ beta + rng.rand(n_splits * n_samples)
    sw = np.ones_like(y)
    if sparse_container is not None:
        X = sparse_container(X)
    params = dict(tol=1e-6)

    # Set alphas, otherwise the two cv models might use different ones.
    if fit_intercept:
        alphas = np.linspace(0.001, 0.01, num=91)
    else:
        alphas = np.linspace(0.01, 0.1, num=91)

    # We weight the first fold 2 times more.
    sw[:n_samples] = 2
    groups_sw = np.r_[
        np.full(n_samples, 0), np.full(n_samples, 1), np.full(n_samples, 2)
    ]
    splits_sw = list(LeaveOneGroupOut().split(X, groups=groups_sw))
    reg_sw = ElasticNetCV(
        alphas=alphas, cv=splits_sw, fit_intercept=fit_intercept, **params
    )
    reg_sw.fit(X, y, sample_weight=sw)

    # We repeat the first fold 2 times and provide splits ourselves
    if sparse_container is not None:
        X = X.toarray()
    X = np.r_[X[:n_samples], X]
    if sparse_container is not None:
        X = sparse_container(X)
    y = np.r_[y[:n_samples], y]
    groups = np.r_[
        np.full(2 * n_samples, 0), np.full(n_samples, 1), np.full(n_samples, 2)
    ]
    splits = list(LeaveOneGroupOut().split(X, groups=groups))
    reg = ElasticNetCV(alphas=alphas, cv=splits, fit_intercept=fit_intercept, **params)
    reg.fit(X, y)

    # ensure that we chose meaningful alphas, i.e. not boundaries
    assert alphas[0] < reg.alpha_ < alphas[-1]
    assert reg_sw.alpha_ == reg.alpha_
    assert_allclose(reg_sw.coef_, reg.coef_)
    assert reg_sw.intercept_ == pytest.approx(reg.intercept_)


@pytest.mark.parametrize("sample_weight", [False, True])
def test_enet_cv_grid_search(sample_weight):
    """Test that ElasticNetCV gives same result as GridSearchCV."""
    n_samples, n_features = 200, 10
    cv = 5
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=10,
        n_informative=n_features - 4,
        noise=10,
        random_state=0,
    )
    if sample_weight:
        sample_weight = np.linspace(1, 5, num=n_samples)
    else:
        sample_weight = None

    alphas = np.logspace(np.log10(1e-5), np.log10(1), num=10)
    l1_ratios = [0.1, 0.5, 0.9]
    reg = ElasticNetCV(cv=cv, alphas=alphas, l1_ratio=l1_ratios)
    reg.fit(X, y, sample_weight=sample_weight)

    param = {"alpha": alphas, "l1_ratio": l1_ratios}
    gs = GridSearchCV(
        estimator=ElasticNet(),
        param_grid=param,
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    ).fit(X, y, sample_weight=sample_weight)
    # 使用网格搜索得到的最佳超参数（l1_ratio 和 alpha）在训练数据集 X, y 上进行拟合
    assert reg.l1_ratio_ == pytest.approx(gs.best_params_["l1_ratio"])
    # 断言确保模型的 l1_ratio 参数接近于网格搜索得到的最佳 l1_ratio 参数值
    assert reg.alpha_ == pytest.approx(gs.best_params_["alpha"])
    # 断言确保模型的 alpha 参数接近于网格搜索得到的最佳 alpha 参数值
# 使用 pytest 的 parametrize 装饰器为 test_enet_cv_sample_weight_consistency 函数提供参数化测试支持
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("l1_ratio", [0, 0.5, 1])
@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS)
def test_enet_cv_sample_weight_consistency(
    fit_intercept, l1_ratio, precompute, sparse_container
):
    """Test that the impact of sample_weight is consistent."""
    
    # 创建一个随机数生成器对象 rng，用于生成确定性随机数
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    # 生成一个随机的二维数组 X，大小为 n_samples x n_features
    X = rng.rand(n_samples, n_features)
    
    # 生成一个随机的一维数组 y，作为回归问题的目标值
    y = X.sum(axis=1) + rng.rand(n_samples)
    
    # 定义模型参数 params，包括 l1_ratio, fit_intercept, precompute 等
    params = dict(
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        precompute=precompute,
        tol=1e-6,
        cv=3,
    )
    
    # 如果 sparse_container 不为空，则对 X 应用 sparse_container 函数
    if sparse_container is not None:
        X = sparse_container(X)
    
    # 根据不同的 l1_ratio，选择不同的回归器进行拟合
    if l1_ratio == 0:
        # 如果 l1_ratio 为 0，则从 params 中移除 l1_ratio 参数，并使用 LassoCV 拟合
        params.pop("l1_ratio", None)
        reg = LassoCV(**params).fit(X, y)
    else:
        # 否则，使用 ElasticNetCV 拟合
        reg = ElasticNetCV(**params).fit(X, y)
    
    # 复制回归器的系数 coef
    coef = reg.coef_.copy()
    
    # 如果 fit_intercept 为 True，则复制回归器的截距 intercept
    if fit_intercept:
        intercept = reg.intercept_
    
    # 测试 sample_weight=np.ones(..) 与 sample_weight=None 的等效性
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    
    # 测试 sample_weight=None 与 sample_weight = number 的等效性
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    
    # 测试 sample_weight 的缩放对结果的影响
    sample_weight = 2 * np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)


# 使用 pytest 的 parametrize 装饰器为 test_linear_models_cv_fit_with_loky 函数提供参数化测试支持
@pytest.mark.parametrize("estimator", [ElasticNetCV, LassoCV])
def test_linear_models_cv_fit_with_loky(estimator):
    # LinearModelsCV.fit 在使用 loky 后端时，对于使用 fancy-indexing 的 memmapped 数据执行就地操作，
    # 这会导致错误，原因是对只读 memmap 的 fancy indexing 行为意外（参见 numpy#14132）。
    
    # 创建一个足够大的问题以触发 memmapping（1MB）
    X, y = make_regression(int(1e6) // 8 + 1, 1)
    assert X.nbytes > 1e6  # 确保 X 的大小大于 1 MB
    
    # 使用 loky 后端执行拟合操作
    with joblib.parallel_backend("loky"):
        estimator(n_jobs=2, cv=3).fit(X, y)


# 使用 pytest 的 parametrize 装饰器为 test_enet_sample_weight_does_not_overwrite_sample_weight 函数提供参数化测试支持
@pytest.mark.parametrize("check_input", [True, False])
def test_enet_sample_weight_does_not_overwrite_sample_weight(check_input):
    """Check that ElasticNet does not overwrite sample_weights."""
    
    # 创建一个随机数生成器对象 rng，用于生成确定性随机数
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5
    
    # 生成一个随机的二维数组 X，大小为 n_samples x n_features
    X = rng.rand(n_samples, n_features)
    
    # 生成一个随机的一维数组 y，作为回归问题的目标值
    y = rng.rand(n_samples)
    
    # 创建一个 sample_weight_1_25，用于保存样本权重的副本
    sample_weight_1_25 = 1.25 * np.ones_like(y)
    # 复制一份 sample_weight 的内容到 sample_weight 变量
    sample_weight = sample_weight_1_25.copy()
    # 创建一个 ElasticNet 回归模型对象
    reg = ElasticNet()
    # 使用给定的输入数据 X 和目标值 y 进行模型训练，可以指定样本权重和输入检查选项
    reg.fit(X, y, sample_weight=sample_weight, check_input=check_input)

    # 断言函数，用于检查两个数组或矩阵是否相等；确保样本权重 sample_weight 和预期的 sample_weight_1_25 相等
    assert_array_equal(sample_weight, sample_weight_1_25)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
# 使用 pytest.mark.filterwarnings 忽略特定警告信息

@pytest.mark.parametrize("ridge_alpha", [1e-1, 1.0, 1e6])
# 使用 pytest.mark.parametrize 注入不同的 ridge_alpha 值作为参数，用于多次运行测试函数

def test_enet_ridge_consistency(ridge_alpha):
    # 检查当 ElasticNet 的 l1_ratio=0 时，它是否与 Ridge 收敛到相同的解
    # 前提是 alpha 的值已经适配好了。
    #
    # XXX: 当弱正则化时（较低的 ridge_alpha 值）这个测试不通过：这可能是 ElasticNet 或 Ridge 的问题（可能性较小），
    # 取决于数据集的统计特性：特别是在 effective_rank 较低时更有问题。

    rng = np.random.RandomState(42)
    n_samples = 300
    X, y = make_regression(
        n_samples=n_samples,
        n_features=100,
        effective_rank=10,
        n_informative=50,
        random_state=rng,
    )
    # 生成一个随机数种子为 42 的随机数生成器对象
    sw = rng.uniform(low=0.01, high=10, size=X.shape[0])
    # 使用随机数生成器生成均匀分布的权重 sw
    alpha = 1.0
    common_params = dict(
        tol=1e-12,
    )
    # 定义通用参数字典 common_params，包括误差容限 tol

    ridge = Ridge(alpha=alpha, **common_params).fit(X, y, sample_weight=sw)
    # 使用 Ridge 模型拟合数据 X, y，并应用样本权重 sw

    alpha_enet = alpha / sw.sum()
    # 计算 ElasticNet 模型的 alpha 值
    enet = ElasticNet(alpha=alpha_enet, l1_ratio=0, **common_params).fit(
        X, y, sample_weight=sw
    )
    # 使用 ElasticNet 模型拟合数据 X, y，l1_ratio=0，应用样本权重 sw

    assert_allclose(ridge.coef_, enet.coef_)
    # 断言 Ridge 和 ElasticNet 模型的系数应该非常接近
    assert_allclose(ridge.intercept_, enet.intercept_)
    # 断言 Ridge 和 ElasticNet 模型的截距应该非常接近


@pytest.mark.parametrize(
    "estimator",
    [
        Lasso(alpha=1.0),
        ElasticNet(alpha=1.0, l1_ratio=0.1),
    ],
)
# 使用 pytest.mark.parametrize 注入不同的 estimator 对象作为参数，用于多次运行测试函数

def test_sample_weight_invariance(estimator):
    rng = np.random.RandomState(42)
    # 生成一个随机数种子为 42 的随机数生成器对象
    X, y = make_regression(
        n_samples=100,
        n_features=300,
        effective_rank=10,
        n_informative=50,
        random_state=rng,
    )
    # 生成具有给定统计特性的回归模型数据 X, y

    sw = rng.uniform(low=0.01, high=2, size=X.shape[0])
    # 使用随机数生成器生成均匀分布的权重 sw
    params = dict(tol=1e-12)
    # 定义参数字典 params，包括误差容限 tol

    # 检查将一些权重设置为 0 是否等同于修剪样本：
    cutoff = X.shape[0] // 3
    sw_with_null = sw.copy()
    sw_with_null[:cutoff] = 0.0
    X_trimmed, y_trimmed = X[cutoff:, :], y[cutoff:]
    sw_trimmed = sw[cutoff:]

    reg_trimmed = (
        clone(estimator)
        .set_params(**params)
        .fit(X_trimmed, y_trimmed, sample_weight=sw_trimmed)
    )
    # 使用 clone 创建一个 estimator 的副本，设置参数并使用修剪后的数据拟合

    reg_null_weighted = (
        clone(estimator).set_params(**params).fit(X, y, sample_weight=sw_with_null)
    )
    # 使用 clone 创建一个 estimator 的副本，设置参数并使用将部分样本权重置为 0 的数据拟合

    assert_allclose(reg_null_weighted.coef_, reg_trimmed.coef_)
    # 断言两种拟合方式下的系数应该非常接近
    assert_allclose(reg_null_weighted.intercept_, reg_trimmed.intercept_)
    # 断言两种拟合方式下的截距应该非常接近

    # 检查复制训练数据集是否等同于将权重乘以 2：
    X_dup = np.concatenate([X, X], axis=0)
    y_dup = np.concatenate([y, y], axis=0)
    sw_dup = np.concatenate([sw, sw], axis=0)

    reg_2sw = clone(estimator).set_params(**params).fit(X, y, sample_weight=2 * sw)
    # 使用 clone 创建一个 estimator 的副本，设置参数并使用样本权重乘以 2 的数据拟合

    reg_dup = (
        clone(estimator).set_params(**params).fit(X_dup, y_dup, sample_weight=sw_dup)
    )
    # 使用 clone 创建一个 estimator 的副本，设置参数并使用复制后的数据拟合

    assert_allclose(reg_2sw.coef_, reg_dup.coef_)
    # 断言两种拟合方式下的系数应该非常接近
    assert_allclose(reg_2sw.intercept_, reg_dup.intercept_)
    # 断言两种拟合方式下的截距应该非常接近
    """Test that sparse coordinate descent works for read-only buffers"""
    
    # 使用随机种子生成器创建一个随机状态对象
    rng = np.random.RandomState(0)
    # 使用ElasticNet模型，设置正则化参数alpha为0.1，复制输入数据X，设置随机种子
    clf = ElasticNet(alpha=0.1, copy_X=True, random_state=rng)
    # 生成一个100行10列的Fortran顺序的数组，并设置为只读
    X = np.asfortranarray(rng.uniform(size=(100, 10)))
    X.setflags(write=False)  # 将数组设置为只读状态
    
    # 生成一个长度为100的随机数组y
    y = rng.rand(100)
    # 使用clf对象对只读的X和y进行拟合
    clf.fit(X, y)
# 使用 pytest 的 mark.parametrize 装饰器定义一个参数化测试函数，测试不同的交叉验证估算器
@pytest.mark.parametrize(
    "EstimatorCV",  # 参数化的参数，包括 ElasticNetCV、LassoCV、MultiTaskElasticNetCV、MultiTaskLassoCV
    [ElasticNetCV, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV],
)
def test_cv_estimators_reject_params_with_no_routing_enabled(EstimatorCV):
    """Check that the models inheriting from class:`LinearModelCV` raise an
    error when any `params` are passed when routing is not enabled.
    """
    # 生成一个回归用的数据集 X 和 y
    X, y = make_regression(random_state=42)
    # 创建分组，用于交叉验证
    groups = np.array([0, 1] * (len(y) // 2))
    # 创建指定的估算器实例
    estimator = EstimatorCV()
    # 准备错误消息，用于检查是否引发 ValueError 异常
    msg = "is only supported if enable_metadata_routing=True"
    # 使用 pytest 的 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y, groups=groups)


# 使用 pytest 的 mark.usefixtures 装饰器，为测试函数设置 slep006 的 fixture
@pytest.mark.usefixtures("enable_slep006")
# 参数化测试函数，测试多任务估算器 MultiTaskElasticNetCV 和 MultiTaskLassoCV
@pytest.mark.parametrize(
    "MultiTaskEstimatorCV",
    [MultiTaskElasticNetCV, MultiTaskLassoCV],
)
def test_multitask_cv_estimators_with_sample_weight(MultiTaskEstimatorCV):
    """Check that for :class:`MultiTaskElasticNetCV` and
    class:`MultiTaskLassoCV` if `sample_weight` is passed and the
    CV splitter does not support `sample_weight` an error is raised.
    On the other hand if the splitter does support `sample_weight`
    while `sample_weight` is passed there is no error and process
    completes smoothly as before.
    """

    # 定义一个自定义的 CVSplitter 类，继承自 GroupsConsumerMixin 和 BaseCrossValidator
    class CVSplitter(GroupsConsumerMixin, BaseCrossValidator):
        # 实现获取切分数量的方法
        def get_n_splits(self, X=None, y=None, groups=None, metadata=None):
            pass  # pragma: nocover

    # 定义一个 CVSplitterSampleWeight 类，继承自 CVSplitter，支持 sample_weight
    class CVSplitterSampleWeight(CVSplitter):
        # 实现数据切分的方法，支持 sample_weight
        def split(self, X, y=None, groups=None, sample_weight=None):
            split_index = len(X) // 2
            train_indices = list(range(0, split_index))
            test_indices = list(range(split_index, len(X)))
            yield test_indices, train_indices
            yield train_indices, test_indices

    # 生成一个多任务回归数据集 X 和 y
    X, y = make_regression(random_state=42, n_targets=2)
    # 设置样本权重
    sample_weight = np.ones(X.shape[0])

    # 如果 CV splitter 不支持 sample_weight，预期引发 ValueError 异常
    splitter = CVSplitter().set_split_request(groups=True)
    estimator = MultiTaskEstimatorCV(cv=splitter)
    msg = "do not support sample weights"
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y, sample_weight=sample_weight)

    # 如果 CV splitter 支持 sample_weight，预期不引发异常
    splitter = CVSplitterSampleWeight().set_split_request(
        groups=True, sample_weight=True
    )
    estimator = MultiTaskEstimatorCV(cv=splitter)
    estimator.fit(X, y, sample_weight=sample_weight)
```