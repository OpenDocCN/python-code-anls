# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_base.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings

# 导入科学计算相关库
import numpy as np
import pytest
from scipy import linalg, sparse

# 导入数据集加载和生成工具
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
# 导入线性回归模型
from sklearn.linear_model import LinearRegression
# 导入数据预处理相关函数
from sklearn.linear_model._base import (
    _preprocess_data,
    _rescale_data,
    make_dataset,
)
# 导入添加虚拟特征的函数
from sklearn.preprocessing import add_dummy_feature
# 导入测试辅助函数
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
# 导入兼容性修复工具
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    LIL_CONTAINERS,
)

# 定义数值比较容差
rtol = 1e-6


def test_linear_regression():
    # 在一个简单数据集上测试线性回归。
    # 定义简单的输入特征和目标值
    X = [[1], [2]]
    Y = [1, 2]

    # 创建线性回归模型
    reg = LinearRegression()
    # 拟合模型
    reg.fit(X, Y)

    # 断言回归系数接近于 [1]
    assert_array_almost_equal(reg.coef_, [1])
    # 断言截距接近于 [0]
    assert_array_almost_equal(reg.intercept_, [0])
    # 断言预测结果接近于 [1, 2]
    assert_array_almost_equal(reg.predict(X), [1, 2])

    # 测试特殊情况下的输入
    X = [[1]]
    Y = [0]

    # 创建线性回归模型
    reg = LinearRegression()
    # 拟合模型
    reg.fit(X, Y)
    # 断言回归系数接近于 [0]
    assert_array_almost_equal(reg.coef_, [0])
    # 断言截距接近于 [0]
    assert_array_almost_equal(reg.intercept_, [0])
    # 断言预测结果接近于 [0]
    assert_array_almost_equal(reg.predict(X), [0])


@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_sample_weights(
    sparse_container, fit_intercept, global_random_seed
):
    # 设置随机数生成器
    rng = np.random.RandomState(global_random_seed)

    # 确定样本数量和特征数量
    n_samples, n_features = 6, 5

    # 生成正态分布的输入特征
    X = rng.normal(size=(n_samples, n_features))
    # 如果指定了稀疏容器，则使用该容器处理输入特征
    if sparse_container is not None:
        X = sparse_container(X)
    # 生成正态分布的目标值
    y = rng.normal(size=n_samples)

    # 生成样本权重
    sample_weight = 1.0 + rng.uniform(size=n_samples)

    # 使用显式样本权重进行线性回归
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y, sample_weight=sample_weight)
    # 获取回归系数和截距
    coefs1 = reg.coef_
    inter1 = reg.intercept_

    # 进行系数形状的断言检查
    assert reg.coef_.shape == (X.shape[1],)  # sanity checks

    # 权重最小二乘法的闭合形式
    # theta = (X^T W X)^(-1) @ X^T W y
    W = np.diag(sample_weight)
    X_aug = X if not fit_intercept else add_dummy_feature(X)

    Xw = X_aug.T @ W @ X_aug
    yw = X_aug.T @ W @ y
    coefs2 = linalg.solve(Xw, yw)

    # 如果不包括截距，断言系数接近
    if not fit_intercept:
        assert_allclose(coefs1, coefs2)
    else:
        # 否则断言系数接近并验证截距
        assert_allclose(coefs1, coefs2[1:])
        assert_allclose(inter1, coefs2[0])


def test_raises_value_error_if_positive_and_sparse():
    # 如果 positive == True 时 X 是稀疏的，则引发 ValueError
    error_msg = "Sparse data was passed for X, but dense data is required."
    # 创建一个稀疏的单位矩阵作为输入特征
    X = sparse.eye(10)
    y = np.ones(10)

    # 创建带有 positive=True 参数的线性回归模型
    reg = LinearRegression(positive=True)

    # 使用 pytest 检查是否引发了预期的 TypeError 异常
    with pytest.raises(TypeError, match=error_msg):
        reg.fit(X, y)


@pytest.mark.parametrize("n_samples, n_features", [(2, 3), (3, 2)])
def test_raises_value_error_if_sample_weights_greater_than_1d(n_samples, n_features):
    # Sample weights must be either scalar or 1D

    # 使用随机数种子创建随机数生成器对象
    rng = np.random.RandomState(0)
    # 生成随机的二维数组作为特征矩阵
    X = rng.randn(n_samples, n_features)
    # 生成随机的目标值数组
    y = rng.randn(n_samples)
    # 生成符合要求的样本权重数组
    sample_weights_OK = rng.randn(n_samples) ** 2 + 1
    sample_weights_OK_1 = 1.0
    sample_weights_OK_2 = 2.0

    # 创建线性回归对象
    reg = LinearRegression()

    # 确保“OK”样本权重正常工作
    reg.fit(X, y, sample_weights_OK)
    reg.fit(X, y, sample_weights_OK_1)
    reg.fit(X, y, sample_weights_OK_2)


def test_fit_intercept():
    # Test assertions on betas shape.

    # 创建两个不同形状的二维数组作为特征矩阵
    X2 = np.array([[0.38349978, 0.61650022], [0.58853682, 0.41146318]])
    X3 = np.array(
        [[0.27677969, 0.70693172, 0.01628859], [0.08385139, 0.20692515, 0.70922346]]
    )
    # 创建目标值数组
    y = np.array([1, 1])

    # 创建带和不带截距的线性回归对象并拟合数据
    lr2_without_intercept = LinearRegression(fit_intercept=False).fit(X2, y)
    lr2_with_intercept = LinearRegression().fit(X2, y)

    lr3_without_intercept = LinearRegression(fit_intercept=False).fit(X3, y)
    lr3_with_intercept = LinearRegression().fit(X3, y)

    # 断言截距的形状是否一致
    assert lr2_with_intercept.coef_.shape == lr2_without_intercept.coef_.shape
    assert lr3_with_intercept.coef_.shape == lr3_without_intercept.coef_.shape
    # 断言不带截距的系数维度是否一致
    assert lr2_without_intercept.coef_.ndim == lr3_without_intercept.coef_.ndim


def test_linear_regression_sparse(global_random_seed):
    # Test that linear regression also works with sparse data

    # 使用全局随机种子创建随机数生成器对象
    rng = np.random.RandomState(global_random_seed)
    n = 100
    # 创建稀疏单位矩阵作为特征矩阵
    X = sparse.eye(n, n)
    beta = rng.rand(n)
    # 计算目标值
    y = X @ beta

    # 创建线性回归对象并拟合稀疏数据
    ols = LinearRegression()
    ols.fit(X, y.ravel())
    # 断言预测值与真实值的接近程度
    assert_array_almost_equal(beta, ols.coef_ + ols.intercept_)
    assert_array_almost_equal(ols.predict(X) - y.ravel(), 0)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_linear_regression_sparse_equal_dense(fit_intercept, csr_container):
    # Test that linear regression agrees between sparse and dense

    # 使用固定的随机种子创建随机数生成器对象
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 2
    # 创建随机的二维数组作为特征矩阵，并将部分值置零
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    # 使用CSR容器包装特征矩阵
    Xcsr = csr_container(X)
    y = rng.rand(n_samples)
    params = dict(fit_intercept=fit_intercept)
    # 创建带和不带截距的线性回归对象，并分别拟合稠密和稀疏数据
    clf_dense = LinearRegression(**params)
    clf_sparse = LinearRegression(**params)
    clf_dense.fit(X, y)
    clf_sparse.fit(Xcsr, y)
    # 断言截距是否近似相等
    assert clf_dense.intercept_ == pytest.approx(clf_sparse.intercept_)
    # 断言系数是否近似相等
    assert_allclose(clf_dense.coef_, clf_sparse.coef_)


def test_linear_regression_multiple_outcome():
    # Test multiple-outcome linear regressions

    # 使用固定的随机种子创建随机数生成器对象
    rng = np.random.RandomState(0)
    # 生成模拟数据
    X, y = make_regression(random_state=rng)

    # 创建多输出目标值
    Y = np.vstack((y, y)).T
    n_features = X.shape[1]

    # 创建线性回归对象并拟合多输出数据
    reg = LinearRegression()
    reg.fit((X), Y)
    # 断言系数的形状是否正确
    assert reg.coef_.shape == (2, n_features)
    # 进行预测并断言单输出和多输出数据的预测结果的形状
    Y_pred = reg.predict(X)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 使用 NumPy 的 assert_array_almost_equal 函数比较预测结果 y_pred 和目标结果 Y_pred，
    # 要求两者的转置（即每列的值应当对应比较）堆叠成的数组几乎相等，精度为小数点后三位。
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)
# 使用pytest的参数化装饰器，用于多次运行同一测试函数，每次使用不同的参数
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
# 测试多输出线性回归在稀疏数据上的表现
def test_linear_regression_sparse_multiple_outcome(global_random_seed, coo_container):
    # 设置全局随机种子
    rng = np.random.RandomState(global_random_seed)
    # 生成稀疏且不相关的数据
    X, y = make_sparse_uncorrelated(random_state=rng)
    # 将输入数据转换为指定的稀疏容器类型
    X = coo_container(X)
    # 创建包含两列y的矩阵Y
    Y = np.vstack((y, y)).T
    # 特征数为X的列数
    n_features = X.shape[1]

    # 创建线性回归对象
    ols = LinearRegression()
    # 拟合多输出线性回归模型
    ols.fit(X, Y)
    # 断言模型系数的形状为(2, n_features)
    assert ols.coef_.shape == (2, n_features)
    # 预测Y的值
    Y_pred = ols.predict(X)
    # 再次拟合模型，但这次只使用单列y
    ols.fit(X, y.ravel())
    # 预测单列y的值
    y_pred = ols.predict(X)
    # 断言两次预测结果的近似性
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)


# 测试正系数约束下的线性回归在简单数据集上的表现
def test_linear_regression_positive():
    X = [[1], [2]]
    y = [1, 2]

    # 创建正系数约束下的线性回归对象
    reg = LinearRegression(positive=True)
    # 拟合模型
    reg.fit(X, y)

    # 断言模型系数接近于 [1]
    assert_array_almost_equal(reg.coef_, [1])
    # 断言截距接近于 [0]
    assert_array_almost_equal(reg.intercept_, [0])
    # 断言预测值接近于 [1, 2]
    assert_array_almost_equal(reg.predict(X), [1, 2])

    # 对退化输入也进行测试
    X = [[1]]
    y = [0]

    reg = LinearRegression(positive=True)
    reg.fit(X, y)
    # 断言模型系数接近于 [0]
    assert_allclose(reg.coef_, [0])
    # 断言截距接近于 [0]
    assert_allclose(reg.intercept_, [0])
    # 断言预测值接近于 [0]
    assert_allclose(reg.predict(X), [0])


# 测试正系数约束下的多输出线性回归
def test_linear_regression_positive_multiple_outcome(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    X, y = make_sparse_uncorrelated(random_state=rng)
    Y = np.vstack((y, y)).T
    n_features = X.shape[1]

    # 创建正系数约束下的线性回归对象
    ols = LinearRegression(positive=True)
    # 拟合多输出线性回归模型
    ols.fit(X, Y)
    # 断言模型系数的形状为(2, n_features)
    assert ols.coef_.shape == (2, n_features)
    # 断言所有系数都大于等于0
    assert np.all(ols.coef_ >= 0.0)
    # 预测Y的值
    Y_pred = ols.predict(X)
    # 再次拟合模型，但这次只使用单列y
    ols.fit(X, y.ravel())
    # 预测单列y的值
    y_pred = ols.predict(X)
    # 断言两次预测结果的近似性
    assert_allclose(np.vstack((y_pred, y_pred)).T, Y_pred)


# 测试正系数与非正系数线性回归的区别
def test_linear_regression_positive_vs_nonpositive(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    X, y = make_sparse_uncorrelated(random_state=rng)

    # 创建正系数约束下的线性回归对象
    reg = LinearRegression(positive=True)
    reg.fit(X, y)
    # 创建无正系数约束下的线性回归对象
    regn = LinearRegression(positive=False)
    regn.fit(X, y)

    # 断言模型系数的平均平方差大于1e-3
    assert np.mean((reg.coef_ - regn.coef_) ** 2) > 1e-3


# 测试在正系数情况下，正系数与非正系数线性回归拟合系数的差异
def test_linear_regression_positive_vs_nonpositive_when_positive(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 4
    X = rng.rand(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 1.5 * X[:, 3]

    # 创建正系数约束下的线性回归对象
    reg = LinearRegression(positive=True)
    reg.fit(X, y)
    # 创建无正系数约束下的线性回归对象
    regn = LinearRegression(positive=False)
    regn.fit(X, y)

    # 断言模型系数的平均平方差小于1e-6
    assert np.mean((reg.coef_ - regn.coef_) ** 2) < 1e-6


@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("use_sw", [True, False])
# 检查线性回归估计器是否不会对数据进行原地修改
def test_inplace_data_preprocessing(sparse_container, use_sw, global_random_seed):
    # 创建一个特定随机种子的随机数生成器实例
    rng = np.random.RandomState(global_random_seed)
    # 生成原始的 X 数据，形状为 (10, 12)
    original_X_data = rng.randn(10, 12)
    # 生成原始的 y 数据，形状为 (10, 2)
    original_y_data = rng.randn(10, 2)
    # 生成原始的 sample weight 数据，形状为 (10)
    orginal_sw_data = rng.rand(10)

    # 如果 sparse_container 不为空，则使用其对原始 X 数据进行处理，否则复制原始 X 数据
    if sparse_container is not None:
        X = sparse_container(original_X_data)
    else:
        X = original_X_data.copy()
    # 复制原始 y 数据
    y = original_y_data.copy()

    # 如果 use_sw 为 True，则复制原始 sample weight 数据，否则设为 None
    if use_sw:
        sample_weight = orginal_sw_data.copy()
    else:
        sample_weight = None

    # 创建一个线性回归对象
    reg = LinearRegression()
    # 使用 X, y 数据进行拟合，可以选择是否使用 sample weight
    reg.fit(X, y, sample_weight=sample_weight)

    # 如果 sparse_container 不为空，则断言 X 的稀疏表示转为稠密后与原始 X 数据相等
    # 否则断言 X 与原始 X 数据相等
    if sparse_container is not None:
        assert_allclose(X.toarray(), original_X_data)
    else:
        assert_allclose(X, original_X_data)

    # 断言 y 与原始 y 数据相等
    assert_allclose(y, original_y_data)

    # 如果 use_sw 为 True，则断言 sample weight 与原始 sample weight 数据相等
    if use_sw:
        assert_allclose(sample_weight, orginal_sw_data)

    # 允许对 X, y 进行原地处理
    reg = LinearRegression(copy_X=False)
    reg.fit(X, y, sample_weight=sample_weight)

    # 如果 sparse_container 不为空，则断言 X 的稀疏表示转为稠密后与原始 X 数据相等
    # 否则，通过 L2 范数判断 X 是否被原地偏移（和可能被权重调整）了，阈值 0.42 是经验值
    if sparse_container is not None:
        # 目前未实现依赖于稀疏输入原地修改的优化
        assert_allclose(X.toarray(), original_X_data)
    else:
        # X 被原地偏移（并且可能被样本权重调整），0.42 阈值是经验性选择的，对任何在可接受范围内的随机种子都是健壮的
        assert np.linalg.norm(X - original_X_data) > 0.42

    # 断言 y 未被 LinearRegression.fit 原地修改
    assert_allclose(y, original_y_data)

    # 如果 use_sw 为 True，则断言 sample weight 没有理由被原地修改
    if use_sw:
        assert_allclose(sample_weight, orginal_sw_data)


# 测试线性回归对 Pandas 的稀疏 DataFrame 是否会引发警告
def test_linear_regression_pd_sparse_dataframe_warning():
    pd = pytest.importorskip("pandas")

    # 只有当某些列是稀疏时才会触发警告
    df = pd.DataFrame({"0": np.random.randn(10)})
    for col in range(1, 4):
        arr = np.random.randn(10)
        arr[:8] = 0
        # 除了第一列以外的所有列都是稀疏的
        if col != 0:
            arr = pd.arrays.SparseArray(arr, fill_value=0)
        df[str(col)] = arr

    msg = "pandas.DataFrame with sparse columns found."

    reg = LinearRegression()
    with pytest.warns(UserWarning, match=msg):
        reg.fit(df.iloc[:, 0:2], df.iloc[:, 3])

    # 当整个 DataFrame 都是稀疏时不会触发警告
    df["0"] = pd.arrays.SparseArray(df["0"], fill_value=0)
    assert hasattr(df, "sparse")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        reg.fit(df.iloc[:, 0:2], df.iloc[:, 3])


# 测试数据预处理功能
def test_preprocess_data(global_random_seed):
    # 这个测试函数暂时没有具体实现内容，仅包含了一个全局随机种子参数
    # 使用给定的全局随机种子创建一个随机数生成器对象
    rng = np.random.RandomState(global_random_seed)
    
    # 定义样本数量和特征数量
    n_samples = 200
    n_features = 2
    
    # 使用随机数生成器生成一个形状为 (n_samples, n_features) 的随机数组 X
    X = rng.rand(n_samples, n_features)
    
    # 使用随机数生成器生成一个长度为 n_samples 的随机数组 y
    y = rng.rand(n_samples)
    
    # 计算 X 的每个特征的期望值，返回长度为 n_features 的数组
    expected_X_mean = np.mean(X, axis=0)
    
    # 计算 y 的期望值，返回一个标量
    expected_y_mean = np.mean(y, axis=0)
    
    # 对数据进行预处理，不考虑截距项
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=False)
    
    # 断言 X_mean 应该接近一个全零数组
    assert_array_almost_equal(X_mean, np.zeros(n_features))
    
    # 断言 y_mean 应该接近零
    assert_array_almost_equal(y_mean, 0)
    
    # 断言 X_scale 应该接近一个全一数组
    assert_array_almost_equal(X_scale, np.ones(n_features))
    
    # 断言 Xt 应该接近 X 本身
    assert_array_almost_equal(Xt, X)
    
    # 断言 yt 应该接近 y 本身
    assert_array_almost_equal(yt, y)
    
    # 对数据进行预处理，考虑截距项
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=True)
    
    # 断言 X_mean 应该接近期望的 X 的每个特征的期望值
    assert_array_almost_equal(X_mean, expected_X_mean)
    
    # 断言 y_mean 应该接近期望的 y 的期望值
    assert_array_almost_equal(y_mean, expected_y_mean)
    
    # 断言 X_scale 应该接近一个全一数组
    assert_array_almost_equal(X_scale, np.ones(n_features))
    
    # 断言 Xt 应该接近 X 减去期望的 X 的每个特征的期望值
    assert_array_almost_equal(Xt, X - expected_X_mean)
    
    # 断言 yt 应该接近 y 减去期望的 y 的期望值
    assert_array_almost_equal(yt, y - expected_y_mean)
# 使用参数化测试框架pytest.mark.parametrize，对sparse_container进行多组测试
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS)
def test_preprocess_data_multioutput(global_random_seed, sparse_container):
    # 创建指定全局随机种子的随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 设置样本数量、特征数量和输出数量
    n_samples = 200
    n_features = 3
    n_outputs = 2
    # 生成随机的输入数据X和输出数据y
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_outputs)
    # 计算期望的y均值
    expected_y_mean = np.mean(y, axis=0)

    # 如果sparse_container不为None，则应用sparse_container处理X
    if sparse_container is not None:
        X = sparse_container(X)

    # 调用_preprocess_data函数进行数据预处理，fit_intercept=False
    _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=False)
    # 断言y均值接近零
    assert_array_almost_equal(y_mean, np.zeros(n_outputs))
    # 断言yt等于y
    assert_array_almost_equal(yt, y)

    # 调用_preprocess_data函数进行数据预处理，fit_intercept=True
    _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=True)
    # 断言y均值接近期望的y均值
    assert_array_almost_equal(y_mean, expected_y_mean)
    # 断言yt等于y减去y均值
    assert_array_almost_equal(yt, y - y_mean)


# 使用参数化测试框架pytest.mark.parametrize，对sparse_container进行多组测试
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
def test_preprocess_data_weighted(sparse_container, global_random_seed):
    # 创建指定全局随机种子的随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 设置样本数量和特征数量
    n_samples = 200
    n_features = 4
    # 生成随机的输入数据X，并使得部分值为零，以确保稀疏性
    X = rng.rand(n_samples, n_features)
    X[X < 0.5] = 0.0

    # 将X的第一个特征放大10倍，以检查特征缩放的影响
    X[:, 0] *= 10

    # 第三个特征设为常数1.0
    X[:, 2] = 1.0

    # 第四个特征设为常数0.0（在稀疏情况下不实际存在）
    X[:, 3] = 0.0
    # 生成随机的输出数据y
    y = rng.rand(n_samples)

    # 生成随机的样本权重
    sample_weight = rng.rand(n_samples)
    # 计算期望的X均值和y均值（加权）
    expected_X_mean = np.average(X, axis=0, weights=sample_weight)
    expected_y_mean = np.average(y, axis=0, weights=sample_weight)

    # 计算加权平均后的X和X的加权方差
    X_sample_weight_avg = np.average(X, weights=sample_weight, axis=0)
    X_sample_weight_var = np.average(
        (X - X_sample_weight_avg) ** 2, weights=sample_weight, axis=0
    )
    # 确定接近常数的特征（不需要缩放）
    constant_mask = X_sample_weight_var < 10 * np.finfo(X.dtype).eps
    assert_array_equal(constant_mask, [0, 0, 1, 1])
    # 计算预期的特征缩放值
    expected_X_scale = np.sqrt(X_sample_weight_var) * np.sqrt(sample_weight.sum())

    # 对X应用sparse_container处理（如果有）
    if sparse_container is not None:
        X = sparse_container(X)

    # 调用_preprocess_data函数进行数据预处理，fit_intercept=True，使用样本权重
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(
        X,
        y,
        fit_intercept=True,
        sample_weight=sample_weight,
    )
    # 断言X均值接近预期的X均值
    assert_array_almost_equal(X_mean, expected_X_mean)
    # 断言y均值接近预期的y均值
    assert_array_almost_equal(y_mean, expected_y_mean)
    # 断言特征缩放接近单位矩阵
    assert_array_almost_equal(X_scale, np.ones(n_features))
    # 如果使用了sparse_container，则断言Xt的稀疏表示与X相同
    if sparse_container is not None:
        assert_array_almost_equal(Xt.toarray(), X.toarray())
    else:
        # 否则断言Xt等于X减去预期的X均值
        assert_array_almost_equal(Xt, X - expected_X_mean)
    # 断言yt等于y减去预期的y均值
    assert_array_almost_equal(yt, y - expected_y_mean)
# 定义一个用于测试稀疏数据处理的函数，使用偏移值来初始化数据
def test_sparse_preprocess_data_offsets(global_random_seed, lil_container):
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 设置样本数和特征数
    n_samples = 200
    n_features = 2
    # 生成稀疏随机矩阵，密度为0.5，使用随机状态rng
    X = sparse.rand(n_samples, n_features, density=0.5, random_state=rng)
    # 转换成lil格式的稀疏矩阵
    X = lil_container(X)
    # 生成随机目标值
    y = rng.rand(n_samples)
    # 将稀疏矩阵X转换为稠密数组XA
    XA = X.toarray()

    # 进行数据预处理，不拟合截距项
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=False)
    # 断言处理后的均值应接近零向量
    assert_array_almost_equal(X_mean, np.zeros(n_features))
    # 断言目标值处理后的均值应接近0
    assert_array_almost_equal(y_mean, 0)
    # 断言特征缩放系数应为全1向量
    assert_array_almost_equal(X_scale, np.ones(n_features))
    # 断言处理后的稀疏矩阵应与原始稠密数组XA接近
    assert_array_almost_equal(Xt.toarray(), XA)
    # 断言处理后的目标值应与原始目标值yt接近
    assert_array_almost_equal(yt, y)

    # 进行数据预处理，拟合截距项
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=True)
    # 断言处理后的均值应与XA按列均值接近
    assert_array_almost_equal(X_mean, np.mean(XA, axis=0))
    # 断言处理后的目标值均值应与y按均值接近
    assert_array_almost_equal(y_mean, np.mean(y, axis=0))
    # 断言特征缩放系数应为全1向量
    assert_array_almost_equal(X_scale, np.ones(n_features))
    # 断言处理后的稀疏矩阵应与原始稠密数组XA接近
    assert_array_almost_equal(Xt.toarray(), XA)
    # 断言处理后的目标值应与y减去均值接近
    assert_array_almost_equal(yt, y - np.mean(y, axis=0))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 测试csr格式数据处理的函数
def test_csr_preprocess_data(csr_container):
    # 生成回归数据
    X, y = make_regression()
    # 将小于2.5的元素设为0
    X[X < 2.5] = 0.0
    # 转换为csr格式的稀疏矩阵
    csr = csr_container(X)
    # 调用_preprocess_data处理数据，拟合截距项
    csr_, y, _, _, _ = _preprocess_data(csr, y, fit_intercept=True)
    # 断言csr_的格式为"csr"
    assert csr_.format == "csr"


@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("to_copy", (True, False))
# 测试处理拷贝数据时的函数，不进行输入检查
def test_preprocess_copy_data_no_checks(sparse_container, to_copy):
    # 生成回归数据
    X, y = make_regression()
    # 将小于2.5的元素设为0
    X[X < 2.5] = 0.0

    # 如果sparse_container不为None，则转换为对应的稀疏格式
    if sparse_container is not None:
        X = sparse_container(X)

    # 调用_preprocess_data处理数据，拟合截距项，指定是否拷贝数据，不进行输入检查
    X_, y_, _, _, _ = _preprocess_data(
        X, y, fit_intercept=True, copy=to_copy, check_input=False
    )

    # 如果需要拷贝数据并且sparse_container不为None，断言X_的数据不与X共享内存
    if to_copy and sparse_container is not None:
        assert not np.may_share_memory(X_.data, X.data)
    # 如果需要拷贝数据并且sparse_container为None，断言X_不与X共享内存
    elif to_copy:
        assert not np.may_share_memory(X_, X)
    # 如果不需要拷贝数据并且sparse_container不为None，断言X_的数据与X的数据共享内存
    elif sparse_container is not None:
        assert np.may_share_memory(X_.data, X.data)
    # 如果不需要拷贝数据并且sparse_container为None，断言X_与X共享内存
    else:
        assert np.may_share_memory(X_, X)


# 测试数据类型处理的函数
def test_dtype_preprocess_data(global_random_seed):
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 设置样本数和特征数
    n_samples = 200
    n_features = 2
    # 生成浮点型随机矩阵X和目标值y
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    # 将X转换为float32类型
    X_32 = np.asarray(X, dtype=np.float32)
    # 将y转换为float32类型
    y_32 = np.asarray(y, dtype=np.float32)
    # 将X转换为float64类型
    X_64 = np.asarray(X, dtype=np.float64)
    # 将y转换为float64类型
    y_64 = np.asarray(y, dtype=np.float64)
    # 对于每个 fit_intercept 值，分别预处理 X_32 和 y_32 数据
    Xt_32, yt_32, X_mean_32, y_mean_32, X_scale_32 = _preprocess_data(
        X_32,
        y_32,
        fit_intercept=fit_intercept,
    )

    # 对于每个 fit_intercept 值，分别预处理 X_64 和 y_64 数据
    Xt_64, yt_64, X_mean_64, y_mean_64, X_scale_64 = _preprocess_data(
        X_64,
        y_64,
        fit_intercept=fit_intercept,
    )

    # 对于每个 fit_intercept 值，分别预处理 X_32 和 y_64 数据
    Xt_3264, yt_3264, X_mean_3264, y_mean_3264, X_scale_3264 = _preprocess_data(
        X_32,
        y_64,
        fit_intercept=fit_intercept,
    )

    # 对于每个 fit_intercept 值，分别预处理 X_64 和 y_32 数据
    Xt_6432, yt_6432, X_mean_6432, y_mean_6432, X_scale_6432 = _preprocess_data(
        X_64,
        y_32,
        fit_intercept=fit_intercept,
    )

    # 断言各种数据的数据类型为 np.float32
    assert Xt_32.dtype == np.float32
    assert yt_32.dtype == np.float32
    assert X_mean_32.dtype == np.float32
    assert y_mean_32.dtype == np.float32
    assert X_scale_32.dtype == np.float32

    # 断言各种数据的数据类型为 np.float64
    assert Xt_64.dtype == np.float64
    assert yt_64.dtype == np.float64
    assert X_mean_64.dtype == np.float64
    assert y_mean_64.dtype == np.float64
    assert X_scale_64.dtype == np.float64

    # 断言各种数据的数据类型为 np.float32
    assert Xt_3264.dtype == np.float32
    assert yt_3264.dtype == np.float32
    assert X_mean_3264.dtype == np.float32
    assert y_mean_3264.dtype == np.float32
    assert X_scale_3264.dtype == np.float32

    # 断言各种数据的数据类型为 np.float64
    assert Xt_6432.dtype == np.float64
    assert yt_6432.dtype == np.float64
    assert X_mean_6432.dtype == np.float64
    assert y_mean_6432.dtype == np.float64
    assert X_scale_6432.dtype == np.float64

    # 断言输入数据 X_32 和 y_32 的数据类型为 np.float32
    assert X_32.dtype == np.float32
    assert y_32.dtype == np.float32

    # 断言输入数据 X_64 和 y_64 的数据类型为 np.float64
    assert X_64.dtype == np.float64
    assert y_64.dtype == np.float64

    # 断言各种预处理后的数据在精度上近似相等
    assert_array_almost_equal(Xt_32, Xt_64)
    assert_array_almost_equal(yt_32, yt_64)
    assert_array_almost_equal(X_mean_32, X_mean_64)
    assert_array_almost_equal(y_mean_32, y_mean_64)
    assert_array_almost_equal(X_scale_32, X_scale_64)
# 使用 pytest.mark.parametrize 装饰器，为 test_rescale_data 函数添加参数化测试
@pytest.mark.parametrize("n_targets", [None, 2])
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
# 定义 test_rescale_data 函数，测试数据重新缩放功能
def test_rescale_data(n_targets, sparse_container, global_random_seed):
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 定义样本数和特征数
    n_samples = 200
    n_features = 2

    # 创建样本权重，使用随机数生成器生成在区间 [1.0, 2.0) 的随机数
    sample_weight = 1.0 + rng.rand(n_samples)
    # 创建样本特征矩阵，使用随机数生成器生成在区间 [0.0, 1.0) 的随机数
    X = rng.rand(n_samples, n_features)
    # 如果 n_targets 为 None，则创建目标值向量，使用随机数生成器生成在区间 [0.0, 1.0) 的随机数
    if n_targets is None:
        y = rng.rand(n_samples)
    else:
        # 否则创建多目标值矩阵，使用随机数生成器生成在区间 [0.0, 1.0) 的随机数
        y = rng.rand(n_samples, n_targets)

    # 计算预期的样本权重的平方根
    expected_sqrt_sw = np.sqrt(sample_weight)
    # 计算预期的重新缩放后的特征矩阵
    expected_rescaled_X = X * expected_sqrt_sw[:, np.newaxis]

    # 如果 n_targets 为 None，则计算预期的重新缩放后的目标值向量
    if n_targets is None:
        expected_rescaled_y = y * expected_sqrt_sw
    else:
        # 否则计算预期的重新缩放后的多目标值矩阵
        expected_rescaled_y = y * expected_sqrt_sw[:, np.newaxis]

    # 如果 sparse_container 不为 None，则将 X 和 y 转换为稀疏表示
    if sparse_container is not None:
        X = sparse_container(X)
        if n_targets is None:
            y = sparse_container(y.reshape(-1, 1))
        else:
            y = sparse_container(y)

    # 调用 _rescale_data 函数进行数据重新缩放
    rescaled_X, rescaled_y, sqrt_sw = _rescale_data(X, y, sample_weight)

    # 断言样本权重的平方根是否与预期相等
    assert_allclose(sqrt_sw, expected_sqrt_sw)

    # 如果 sparse_container 不为 None，则将稀疏表示的 rescaled_X 和 rescaled_y 转换为密集表示
    if sparse_container is not None:
        rescaled_X = rescaled_X.toarray()
        rescaled_y = rescaled_y.toarray()
        if n_targets is None:
            rescaled_y = rescaled_y.ravel()

    # 断言重新缩放后的特征矩阵是否与预期相等
    assert_allclose(rescaled_X, expected_rescaled_X)
    # 断言重新缩放后的目标值向量或多目标值矩阵是否与预期相等
    assert_allclose(rescaled_y, expected_rescaled_y)


# 使用 pytest.mark.parametrize 装饰器，为 test_fused_types_make_dataset 函数添加参数化测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义 test_fused_types_make_dataset 函数，测试生成数据集的不同类型
def test_fused_types_make_dataset(csr_container):
    # 载入鸢尾花数据集
    iris = load_iris()

    # 创建 float32 类型的特征数据和目标值数据
    X_32 = iris.data.astype(np.float32)
    y_32 = iris.target.astype(np.float32)
    # 使用 csr_container 将 float32 类型的特征数据转换为 CSR 稀疏表示
    X_csr_32 = csr_container(X_32)
    # 创建 float32 类型的样本权重数据
    sample_weight_32 = np.arange(y_32.size, dtype=np.float32)

    # 创建 float64 类型的特征数据和目标值数据
    X_64 = iris.data.astype(np.float64)
    y_64 = iris.target.astype(np.float64)
    # 使用 csr_container 将 float64 类型的特征数据转换为 CSR 稀疏表示
    X_csr_64 = csr_container(X_64)
    # 创建 float64 类型的样本权重数据
    sample_weight_64 = np.arange(y_64.size, dtype=np.float64)

    # 创建 array 类型的数据集，分别使用 float32 和 float64 类型的数据
    dataset_32, _ = make_dataset(X_32, y_32, sample_weight_32)
    dataset_64, _ = make_dataset(X_64, y_64, sample_weight_64)
    xi_32, yi_32, _, _ = dataset_32._next_py()
    xi_64, yi_64, _, _ = dataset_64._next_py()
    xi_data_32, _, _ = xi_32
    xi_data_64, _, _ = xi_64

    # 断言 float32 类型的特征数据是否为 float32 类型
    assert xi_data_32.dtype == np.float32
    # 断言 float64 类型的特征数据是否为 float64 类型
    assert xi_data_64.dtype == np.float64
    # 断言 float64 类型的目标值数据与 float32 类型的目标值数据是否近似相等
    assert_allclose(yi_64, yi_32, rtol=rtol)

    # 创建 csr 类型的数据集，分别使用 float32 和 float64 类型的 CSR 稀疏数据
    datasetcsr_32, _ = make_dataset(X_csr_32, y_32, sample_weight_32)
    datasetcsr_64, _ = make_dataset(X_csr_64, y_64, sample_weight_64)
    xicsr_32, yicsr_32, _, _ = datasetcsr_32._next_py()
    xicsr_64, yicsr_64, _, _ = datasetcsr_64._next_py()
    xicsr_data_32, _, _ = xicsr_32
    xicsr_data_64, _, _ = xicsr_64

    # 断言 float32 类型的 CSR 稀疏特征数据是否为 float32 类型
    assert xicsr_data_32.dtype == np.float32
    # 断言 float64 类型的 CSR 稀疏特征数据是否为 float64 类型
    assert xicsr_data_64.dtype == np.float64

    # 断言 float64 类型的 CSR 稀疏特征数据与 float32 类型的 CSR 稀疏特征数据是否近似相等
    assert_allclose(xicsr_data_64, xicsr_data_32, rtol=rtol)
    # 断言 float64 类型的 CSR 稀疏目标值数据与 float32 类型的 CSR 稀疏目标值数据是否近似相等
    assert_allclose(yicsr_64, yicsr_32, rtol=rtol)

    # 断言 array 类型的特征数据与 csr 类型的特征数据（float32）是否完全相等
    assert_array_equal(xi_data_32, xicsr_data_32)
    # 断言 array 类型的特征数据与 csr 类型的特征数据（float64）是否完全相等
    assert_array_equal(xi_data_64, xicsr_data_64)
    # 断言 array 类型的目标值数据与 csr 类型的目标值数据是否完全相等
    assert_array_equal(yi_32, yicsr_32)
    # 使用 assert_array_equal 函数比较 yi_64 和 yicsr_64 两个数组是否相等
    assert_array_equal(yi_64, yicsr_64)
# 使用 pytest.mark.parametrize 装饰器来参数化测试函数，测试稀疏容器和拟合截距的多种情况
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_linear_regression_sample_weight_consistency(
    sparse_container, fit_intercept, global_random_seed
):
    """Test that the impact of sample_weight is consistent.

    Note that this test is stricter than the common test
    check_sample_weights_invariance alone and also tests sparse X.
    It is very similar to test_enet_sample_weight_consistency.
    """
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 定义数据集的样本数和特征数
    n_samples, n_features = 10, 5

    # 生成随机的样本数据 X 和目标值 y
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    # 如果 sparse_container 不为 None，则使用其转换稀疏矩阵 X
    if sparse_container is not None:
        X = sparse_container(X)
    
    # 创建线性回归模型的参数字典
    params = dict(fit_intercept=fit_intercept)

    # 使用 LinearRegression 拟合数据，sample_weight=None 表示无样本权重
    reg = LinearRegression(**params).fit(X, y, sample_weight=None)
    # 复制回归系数以进行后续比较
    coef = reg.coef_.copy()
    # 如果需要拟合截距，则复制拟合得到的截距
    if fit_intercept:
        intercept = reg.intercept_

    # 1) sample_weight=np.ones(..) 必须等效于 sample_weight=None
    # 类似于 check_sample_weights_invariance(name, reg, kind="ones") 的检查，但我们还测试了稀疏输入。
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    # 检查回归系数是否与初始回归的系数相近
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    # 如果需要拟合截距，则检查截距是否相近
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 2) sample_weight=None 应当等效于 sample_weight = number
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    # 再次检查回归系数是否与初始回归的系数相近
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    # 如果需要拟合截距，则再次检查截距是否相近
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 3) sample_weight 的缩放应不影响结果，参考 np.average()
    sample_weight = rng.uniform(low=0.01, high=2, size=X.shape[0])
    # 重新拟合回归模型，使用经过缩放的 sample_weight
    reg = reg.fit(X, y, sample_weight=sample_weight)
    # 更新回归系数以备后续比较
    coef = reg.coef_.copy()
    # 如果需要拟合截距，则更新截距
    if fit_intercept:
        intercept = reg.intercept_

    # 使用 np.pi * sample_weight 重新拟合回归模型
    reg.fit(X, y, sample_weight=np.pi * sample_weight)
    # 如果稀疏容器为 None，则检查回归系数是否相近
    assert_allclose(reg.coef_, coef, rtol=1e-6 if sparse_container is None else 1e-5)
    # 如果需要拟合截距，则再次检查截距是否相近
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 4) 将 sample_weight 中的元素设置为 0 等效于移除这些样本
    sample_weight_0 = sample_weight.copy()
    sample_weight_0[-5:] = 0
    # 增大 y 中后面 5 个样本的目标值，以确保排除这些样本的重要性
    y[-5:] *= 1000
    # 使用设置为 0 的 sample_weight_0 重新拟合回归模型
    reg.fit(X, y, sample_weight=sample_weight_0)
    # 复制回归系数以备后续比较
    coef_0 = reg.coef_.copy()
    # 如果需要拟合截距，则复制拟合得到的截距
    if fit_intercept:
        intercept_0 = reg.intercept_
    # FIXME: https://github.com/scikit-learn/scikit-learn/issues/26164
    # 这个检查通常会失败，例如在调用 SKLEARN_TESTS_GLOBAL_RANDOM_SEED="all" pytest 进行测试时。
    # 主要出现在没有稀疏容器且需要拟合截距的情况下。
    pass
   `
    else:
        # 验证回归系数与预期系数接近，容差为 1e-5
        assert_allclose(reg.coef_, coef_0, rtol=1e-5)
        # 如果启用了截距，验证回归截距与预期截距接近
        if fit_intercept:
            assert_allclose(reg.intercept_, intercept_0)

    # 5) 检查将样本权重乘以 2 是否等价于将对应样本重复两次
    if sparse_container is not None:
        # 对于稀疏矩阵，创建新的稀疏矩阵 X2，包含原始数据 X 和前半部分样本重复的 X
        X2 = sparse.vstack([X, X[: n_samples // 2]], format="csc")
    else:
        # 对于密集矩阵，合并原始数据 X 和前半部分样本重复的 X
        X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    # 合并原始标签 y 和前半部分样本重复的 y
    y2 = np.concatenate([y, y[: n_samples // 2]])
    # 复制样本权重，初始化 sample_weight_1
    sample_weight_1 = sample_weight.copy()
    # 将前半部分样本权重乘以 2
    sample_weight_1[: n_samples // 2] *= 2
    # 合并原始样本权重和前半部分样本权重重复的样本权重，生成 sample_weight_2
    sample_weight_2 = np.concatenate(
        [sample_weight, sample_weight[: n_samples // 2]], axis=0
    )

    # 使用样本数据 X 和样本权重 sample_weight_1 训练一个线性回归模型 reg1
    reg1 = LinearRegression(**params).fit(X, y, sample_weight=sample_weight_1)
    # 使用样本数据 X2 和样本权重 sample_weight_2 训练一个线性回归模型 reg2
    reg2 = LinearRegression(**params).fit(X2, y2, sample_weight=sample_weight_2)
    # 验证 reg1 和 reg2 的回归系数接近，容差为 1e-6
    assert_allclose(reg1.coef_, reg2.coef_, rtol=1e-6)
    # 如果启用了截距，验证 reg1 和 reg2 的截距接近
    if fit_intercept:
        assert_allclose(reg1.intercept_, reg2.intercept_)
```