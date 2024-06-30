# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_sparsefuncs.py`

```
# 导入必要的库和模块
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg

# 导入生成分类数据的函数
from sklearn.datasets import make_classification
# 导入用于测试的断言函数
from sklearn.utils._testing import assert_allclose
# 导入修复的稀疏矩阵容器
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
# 导入稀疏矩阵操作函数
from sklearn.utils.sparsefuncs import (
    _implicit_column_offset,
    count_nonzero,
    csc_median_axis_0,
    incr_mean_variance_axis,
    inplace_column_scale,
    inplace_row_scale,
    inplace_swap_column,
    inplace_swap_row,
    mean_variance_axis,
    min_max_axis,
)
# 导入快速稀疏矩阵操作函数
from sklearn.utils.sparsefuncs_fast import (
    assign_rows_csr,
    csr_row_norms,
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
)

# 使用 pytest 的参数化装饰器来标记测试函数的参数
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_mean_variance_axis0(csc_container, csr_container, lil_container):
    # 生成一个分类数据集 X，_ 变量未使用
    X, _ = make_classification(5, 4, random_state=0)
    
    # 将数组 X 稀疏化一点
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    
    # 使用给定的容器类型构建 X 的 LIL 稀疏矩阵
    X_lil = lil_container(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0
    
    # 使用 pytest 断言会引发 TypeError 异常
    with pytest.raises(TypeError):
        mean_variance_axis(X_lil, axis=0)
    
    # 将 X_lil 转换为 CSR 和 CSC 稀疏矩阵
    X_csr = csr_container(X_lil)
    X_csc = csc_container(X_lil)
    
    # 预期的输入和输出数据类型组合
    expected_dtypes = [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ]
    
    # 遍历预期数据类型组合
    for input_dtype, output_dtype in expected_dtypes:
        # 将 X_test 转换为指定的输入数据类型
        X_test = X.astype(input_dtype)
        
        # 对于每种稀疏矩阵类型（CSR 和 CSC），计算均值和方差
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            
            # 断言均值和方差的数据类型
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            
            # 断言均值和方差与 numpy 计算结果的近似相等性
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


# 使用 pytest 的参数化装饰器来标记测试函数的参数
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
def test_mean_variance_axis0_precision(dtype, sparse_constructor):
    # 检查当真实方差恰好为 0 时，不会出现大的精度损失
    rng = np.random.RandomState(0)
    
    # 创建一个形状为 (1000, 1)、填充值为 100.0 的数组 X
    X = np.full(fill_value=100.0, shape=(1000, 1), dtype=dtype)
    
    # 添加一些应该被忽略的缺失记录
    missing_indices = rng.choice(np.arange(X.shape[0]), 10, replace=False)
    X[missing_indices, 0] = np.nan
    
    # 使用指定的稀疏矩阵构造函数创建稀疏矩阵 X
    X = sparse_constructor(X)
    
    # 创建随机的正权重
    sample_weight = rng.rand(X.shape[0]).astype(dtype)
    
    # 计算均值和方差
    _, var = mean_variance_axis(X, weights=sample_weight, axis=0)
    
    # 断言方差小于指定数据类型的机器精度
    assert var < np.finfo(dtype).eps


# 使用 pytest 的参数化装饰器来标记测试函数的参数
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_implicit_column_offset(csc_container, csr_container):
    # TODO: Add test for _implicit_column_offset function
    pass
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_mean_variance_axis1(csc_container, csr_container, lil_container):
    # 生成一个大小为5x4的分类数据集X，用于测试
    X, _ = make_classification(5, 4, random_state=0)
    # 将数组X稀疏化处理
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    # 使用给定的容器类型将稀疏数组X转换为LIL格式
    X_lil = lil_container(X)
    # 修改LIL格式数组X_lil的一个元素为0
    X_lil[1, 0] = 0
    # 将原始数组X的一个元素设为0
    X[1, 0] = 0

    # 使用pytest断言捕获TypeError异常，确保在axis=1上调用mean_variance_axis函数时抛出异常
    with pytest.raises(TypeError):
        mean_variance_axis(X_lil, axis=1)

    # 使用LIL格式数组X_lil创建CSR格式数组X_csr
    X_csr = csr_container(X_lil)
    # 使用LIL格式数组X_lil创建CSC格式数组X_csc
    X_csc = csc_container(X_lil)

    # 预期的数据类型组合
    expected_dtypes = [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ]

    # 遍历预期的数据类型组合
    for input_dtype, output_dtype in expected_dtypes:
        # 将原始数组X转换为当前的输入数据类型input_dtype
        X_test = X.astype(input_dtype)
        # 对CSR和CSC格式数组进行遍历
        for X_sparse in (X_csr, X_csc):
            # 将稀疏数组X_sparse转换为当前的输入数据类型input_dtype
            X_sparse = X_sparse.astype(input_dtype)
            # 计算X_sparse在axis=0上的均值和方差
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            # 断言计算得到的均值数据类型为output_dtype
            assert X_means.dtype == output_dtype
            # 断言计算得到的方差数据类型为output_dtype
            assert X_vars.dtype == output_dtype
            # 断言计算得到的均值与numpy函数计算得到的均值几乎相等
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            # 断言计算得到的方差与numpy函数计算得到的方差几乎相等
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


@pytest.mark.parametrize(
    ["Xw", "X", "weights"],
    [
        ([[0, 0, 1], [0, 2, 3]], [[0, 0, 1], [0, 2, 3]], [1, 1, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 0, 1], [0, 1, 1, 1]], [1, 2, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]], None),
        (
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [1.0, 1.0, 1.0],
        ),
        (
            [[0, 0], [1, np.nan], [2, 0], [0, 3], [np.nan, np.nan], [np.nan, 2]],
            [
                [0, 0, 0],
                [1, 1, np.nan],
                [2, 2, 0],
                [0, 0, 3],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 2],
            ],
            [2.0, 1.0],
        ),
        (
            [[1, 0, 1], [0, 3, 1]],
            [[1, 0, 0, 0, 1], [0, 3, 3, 3, 1]],
            np.array([1, 3, 1]),
        ),
    ],
)
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incr_mean_variance_axis_weighted_axis1(
    Xw, X, weights, sparse_constructor, dtype
):
    # 定义轴向为1
    axis = 1
    # 将Xw转换为稀疏构造器sparse_constructor创建的稀疏矩阵，并将其转换为dtype类型
    Xw_sparse = sparse_constructor(Xw).astype(dtype)
    # 将X转换为稀疏构造器sparse_constructor创建的稀疏矩阵，并将其转换为dtype类型
    X_sparse = sparse_constructor(X).astype(dtype)

    # 创建大小与Xw的第0轴相同的全零数组last_mean、last_var和last_n，数据类型分别为dtype和int64
    last_mean = np.zeros(np.shape(Xw)[0], dtype=dtype)
    last_var = np.zeros_like(last_mean, dtype=dtype)
    last_n = np.zeros_like(last_mean, dtype=np.int64)

    # 调用incr_mean_variance_axis函数计算X_sparse在axis=1上的增量均值、方差和增量样本数
    means0, vars0, n_incr0 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=None,
    )

    # 调用incr_mean_variance_axis函数计算Xw_sparse在axis=1上的加权增量均值、方差和增量样本数
    means_w0, vars_w0, n_incr_w0 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=weights,
    )
    # 检查 means_w0 的数据类型是否与指定的 dtype 相符
    assert means_w0.dtype == dtype
    # 检查 vars_w0 的数据类型是否与指定的 dtype 相符
    assert vars_w0.dtype == dtype
    # 检查 n_incr_w0 的数据类型是否与指定的 dtype 相符

    assert n_incr_w0.dtype == dtype

    # 使用稀疏矩阵 X_sparse 和指定的轴计算简单平均和方差
    means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

    # 检查计算得到的 means0 是否与 means_w0 几乎相等
    assert_array_almost_equal(means0, means_w0)
    # 检查计算得到的 means0 是否与 means_simple 几乎相等
    assert_array_almost_equal(means0, means_simple)
    # 检查计算得到的 vars0 是否与 vars_w0 几乎相等
    assert_array_almost_equal(vars0, vars_w0)
    # 检查计算得到的 vars0 是否与 vars_simple 几乎相等
    assert_array_almost_equal(vars0, vars_simple)
    # 检查计算得到的 n_incr0 是否与 n_incr_w0 几乎相等
    assert_array_almost_equal(n_incr0, n_incr_w0)

    # 检查增量计算的第二轮结果
    means1, vars1, n_incr1 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=means0,
        last_var=vars0,
        last_n=n_incr0,
        weights=None,
    )

    # 使用增量方式计算第二轮的均值、方差和增量
    means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=means_w0,
        last_var=vars_w0,
        last_n=n_incr_w0,
        weights=weights,
    )

    # 检查增量计算得到的 means1 是否与 means_w1 几乎相等
    assert_array_almost_equal(means1, means_w1)
    # 检查增量计算得到的 vars1 是否与 vars_w1 几乎相等
    assert_array_almost_equal(vars1, vars_w1)
    # 检查增量计算得到的 n_incr1 是否与 n_incr_w1 几乎相等

    assert_array_almost_equal(n_incr1, n_incr_w1)

    # 检查 means_w1 的数据类型是否与指定的 dtype 相符
    assert means_w1.dtype == dtype
    # 检查 vars_w1 的数据类型是否与指定的 dtype 相符
    assert vars_w1.dtype == dtype
    # 检查 n_incr_w1 的数据类型是否与指定的 dtype 相符

    assert n_incr_w1.dtype == dtype
# 使用 pytest 模块的 parametrize 装饰器，为测试用例提供参数化的输入组合
@pytest.mark.parametrize(
    ["Xw", "X", "weights"],
    [  # 定义测试参数，包括 Xw, X 和 weights
        ([[0, 0, 1], [0, 2, 3]], [[0, 0, 1], [0, 2, 3]], [1, 1]),  # 第一组参数
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1], [0, 1, 1]], [1, 2]),  # 第二组参数
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]], None),  # 第三组参数
        (
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [1.0, 1.0],
        ),  # 第四组参数
        (
            [[0, 0, 1, np.nan, 2, 0], [0, 3, np.nan, np.nan, np.nan, 2]],
            [
                [0, 0, 1, np.nan, 2, 0],
                [0, 0, 1, np.nan, 2, 0],
                [0, 3, np.nan, np.nan, np.nan, 2],
            ],
            [2.0, 1.0],
        ),  # 第五组参数
        (
            [[1, 0, 1], [0, 0, 1]],
            [[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            np.array([1, 3]),
        ),  # 第六组参数
    ],
)
# 使用 pytest 的 parametrize 装饰器，为稀疏矩阵构造器提供参数化输入
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，为数据类型提供参数化输入
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 定义测试函数，测试增量计算均值和方差的函数 incr_mean_variance_axis
def test_incr_mean_variance_axis_weighted_axis0(
    Xw, X, weights, sparse_constructor, dtype
):
    axis = 0  # 设置计算的轴向为0
    Xw_sparse = sparse_constructor(Xw).astype(dtype)  # 转换 Xw 到稀疏矩阵并设定数据类型
    X_sparse = sparse_constructor(X).astype(dtype)  # 转换 X 到稀疏矩阵并设定数据类型

    # 初始化上一次计算的均值、方差和计数器
    last_mean = np.zeros(np.size(Xw, 1), dtype=dtype)
    last_var = np.zeros_like(last_mean)
    last_n = np.zeros_like(last_mean, dtype=np.int64)

    # 调用 incr_mean_variance_axis 函数计算不带权重的均值、方差和计数器
    means0, vars0, n_incr0 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=None,
    )

    # 调用 incr_mean_variance_axis 函数计算带权重的均值、方差和计数器
    means_w0, vars_w0, n_incr_w0 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=weights,
    )

    # 断言带权重计算的结果数据类型与指定的数据类型一致
    assert means_w0.dtype == dtype
    assert vars_w0.dtype == dtype
    assert n_incr_w0.dtype == dtype

    # 使用简单方法计算不带权重的均值和方差
    means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

    # 断言增量计算结果与简单计算结果的近似相等性
    assert_array_almost_equal(means0, means_w0)
    assert_array_almost_equal(means0, means_simple)
    assert_array_almost_equal(vars0, vars_w0)
    assert_array_almost_equal(vars0, vars_simple)
    assert_array_almost_equal(n_incr0, n_incr_w0)

    # 进行第二轮增量计算，使用上一轮增量计算的结果作为初始值
    means1, vars1, n_incr1 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=means0,
        last_var=vars0,
        last_n=n_incr0,
        weights=None,
    )

    means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=means_w0,
        last_var=vars_w0,
        last_n=n_incr_w0,
        weights=weights,
    )

    # 断言第二轮增量计算结果与第一轮增量计算结果的近似相等性
    assert_array_almost_equal(means1, means_w1)
    assert_array_almost_equal(vars1, vars_w1)
    assert_array_almost_equal(n_incr1, n_incr_w1)

    # 断言带权重计算第二轮结果的数据类型与指定的数据类型一致
    assert means_w1.dtype == dtype
    assert vars_w1.dtype == dtype
    assert n_incr_w1.dtype == dtype
# 使用 pytest 的参数化功能，为每个测试用例提供多个参数组合
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
# 定义增量均值和方差轴测试函数，接受三种稀疏矩阵容器作为参数
def test_incr_mean_variance_axis(csc_container, csr_container, lil_container):
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
# 测试在 axis=1 且维度不匹配时是否能正确引发错误
def test_incr_mean_variance_axis_dim_mismatch(sparse_constructor):
    """Check that we raise proper error when axis=1 and the dimension mismatch.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18655
    """
    n_samples, n_features = 60, 4
    rng = np.random.RandomState(42)
    X = sparse_constructor(rng.rand(n_samples, n_features))

    # 初始化用于增量计算的数组和字典
    last_mean = np.zeros(n_features)
    last_var = np.zeros_like(last_mean)
    last_n = np.zeros(last_mean.shape, dtype=np.int64)

    kwargs = dict(last_mean=last_mean, last_var=last_var, last_n=last_n)
    # 测试在 axis=0 时增量计算均值和方差，并检查结果是否接近全数据均值和方差
    mean0, var0, _ = incr_mean_variance_axis(X, axis=0, **kwargs)
    assert_allclose(np.mean(X.toarray(), axis=0), mean0)
    assert_allclose(np.var(X.toarray(), axis=0), var0)

    # 测试在 axis=1 时是否引发 ValueError
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X, axis=1, **kwargs)

    # 测试当 last_mean、last_var、last_n 形状不一致时是否引发 ValueError
    kwargs = dict(last_mean=last_mean[:-1], last_var=last_var, last_n=last_n)
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X, axis=0, **kwargs)


@pytest.mark.parametrize(
    "X1, X2",
    [
        (
            sp.random(5, 2, density=0.8, format="csr", random_state=0),
            sp.random(13, 2, density=0.8, format="csr", random_state=0),
        ),
        (
            sp.random(5, 2, density=0.8, format="csr", random_state=0),
            sp.hstack(
                [
                    np.full((13, 1), fill_value=np.nan),
                    sp.random(13, 1, density=0.8, random_state=42),
                ],
                format="csr",
            ),
        ),
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 测试增量均值和方差轴函数在堆叠数据集上的等效性
def test_incr_mean_variance_axis_equivalence_mean_variance(X1, X2, csr_container):
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/16448
    # 检查增量计算的均值和方差是否等同于堆叠数据集上计算的均值和方差
    X1 = csr_container(X1)
    X2 = csr_container(X2)
    axis = 0
    last_mean, last_var = np.zeros(X1.shape[1]), np.zeros(X1.shape[1])
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X2, axis=axis, last_mean=updated_mean, last_var=updated_var, last_n=updated_n
    )
    X = sp.vstack([X1, X2])
    assert_allclose(updated_mean, np.nanmean(X.toarray(), axis=axis))
    # 使用 NumPy 的 assert_allclose 函数检查 updated_var 和 np.nanvar(X.toarray(), axis=axis) 的近似性
    assert_allclose(updated_var, np.nanvar(X.toarray(), axis=axis))
    # 使用 NumPy 的 assert_allclose 函数检查 updated_n 和 np.count_nonzero(~np.isnan(X.toarray()), axis=0) 的近似性
    assert_allclose(updated_n, np.count_nonzero(~np.isnan(X.toarray()), axis=0))
# 定义测试函数，用于测试在没有新样本数的情况下更新均值和方差的行为
def test_incr_mean_variance_no_new_n():
    axis = 0  # 设定操作轴为0
    # 创建稀疏矩阵 X1，密度为0.8，形状为(5, 1)，随机种子为0
    X1 = sp.random(5, 1, density=0.8, random_state=0).tocsr()
    # 创建稀疏矩阵 X2，密度为0.8，形状为(0, 1)，随机种子为0
    X2 = sp.random(0, 1, density=0.8, random_state=0).tocsr()
    # 初始化上一次的均值、方差和样本数为零向量
    last_mean, last_var = np.zeros(X1.shape[1]), np.zeros(X1.shape[1])
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    # 更新 X1 的均值、方差和样本数，返回更新后的结果
    last_mean, last_var, last_n = incr_mean_variance_axis(
        X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    # 使用 X2 更新统计数据，预期 X2 的列将被忽略
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X2, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    # 断言更新后的均值、方差和样本数与上次保持一致
    assert_allclose(updated_mean, last_mean)
    assert_allclose(updated_var, last_var)
    assert_allclose(updated_n, last_n)


# 测试当 last_n 只是一个数字时的行为
def test_incr_mean_variance_n_float():
    axis = 0  # 设定操作轴为0
    # 创建稀疏矩阵 X，密度为0.8，形状为(5, 2)，随机种子为0
    X = sp.random(5, 2, density=0.8, random_state=0).tocsr()
    # 初始化上一次的均值和方差为零向量，上一次的样本数为0
    last_mean, last_var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    last_n = 0
    # 更新 X 的均值、方差和样本数，返回更新后的结果
    _, _, new_n = incr_mean_variance_axis(
        X, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    # 断言更新后的样本数与全为 X 的行数相同
    assert_allclose(new_n, np.full(X.shape[1], X.shape[0]))


# 使用参数化测试，测试在忽略 NaN 值的情况下更新均值和方差
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
def test_incr_mean_variance_axis_ignore_nan(axis, sparse_constructor):
    # 初始化旧的均值、方差和样本数
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_sample_count = np.array([2, 2, 2, 2], dtype=np.int64)

    # 创建稀疏矩阵 X 和 X_nan，包含 NaN 值
    X = sparse_constructor(
        np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]])
    )
    X_nan = sparse_constructor(
        np.array(
            [
                [170, np.nan, 170, 170],
                [np.nan, 170, 430, 430],
                [430, 430, np.nan, 300],
                [300, 300, 300, np.nan],
            ]
        )
    )

    # 如果操作轴为1，将 X 和 X_nan 转置
    if axis:
        X = X.T
        X_nan = X_nan.T

    # 复制一份旧的统计数据，因为它们会被原地修改
    X_means, X_vars, X_sample_count = incr_mean_variance_axis(
        X,
        axis=axis,
        last_mean=old_means.copy(),
        last_var=old_variances.copy(),
        last_n=old_sample_count.copy(),
    )
    X_nan_means, X_nan_vars, X_nan_sample_count = incr_mean_variance_axis(
        X_nan,
        axis=axis,
        last_mean=old_means.copy(),
        last_var=old_variances.copy(),
        last_n=old_sample_count.copy(),
    )

    # 断言包含 NaN 值的 X_nan 的均值、方差和样本数与 X 的保持一致
    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_vars, X_vars)
    assert_allclose(X_nan_sample_count, X_sample_count)


# 使用参数化测试，测试在非法操作轴时的行为
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_mean_variance_illegal_axis(csr_container):
    # 此测试未提供具体的实现代码，在运行时会根据参数化的不同情况自动生成测试用例
    # 使用 make_classification 函数生成一个具有5个样本和4个特征的人工数据集，其中第一个返回值 X 被赋值给 X，第二个返回值被忽略（用 _ 表示）
    X, _ = make_classification(5, 4, random_state=0)
    
    # 将数组 X 中的一些元素设置为零，以增加稀疏性
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    
    # 将稀疏化后的数组 X 转换为 CSR 格式，存储在 X_csr 中
    X_csr = csr_container(X)
    
    # 使用 pytest 的 raises 函数验证在指定轴上调用 mean_variance_axis 函数时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-3)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=2)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-1)
    
    # 使用 pytest 的 raises 函数验证在指定轴上调用 incr_mean_variance_axis 函数时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=-3, last_mean=None, last_var=None, last_n=None
        )
    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=2, last_mean=None, last_var=None, last_n=None
        )
    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=-1, last_mean=None, last_var=None, last_n=None
        )
# 使用 pytest.mark.parametrize 装饰器为 test_densify_rows 函数参数化测试用例，csr_container 为参数之一
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_densify_rows(csr_container):
    # 遍历两种数据类型 np.float32 和 np.float64
    for dtype in (np.float32, np.float64):
        # 创建 CSR 稀疏矩阵 X，数据为特定值，指定数据类型为当前循环的 dtype
        X = csr_container(
            [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=dtype
        )
        # 创建 X_rows 数组，指定特定行的索引
        X_rows = np.array([0, 2, 3], dtype=np.intp)
        # 创建全为 1 的 out 矩阵，形状与 X 相同，数据类型为当前循环的 dtype
        out = np.ones((6, X.shape[1]), dtype=dtype)
        # 创建 out_rows 数组，指定特定行的索引
        out_rows = np.array([1, 3, 4], dtype=np.intp)

        # 创建期望结果 expect，形状与 out 相同，数据类型与 out 相同
        expect = np.ones_like(out)
        # 将 expect 中 out_rows 索引处的行替换为 X 中 X_rows 索引对应行的稀疏矩阵数据
        expect[out_rows] = X[X_rows, :].toarray()

        # 调用 assign_rows_csr 函数，将 X 的指定行数据复制到 out 的指定行
        assign_rows_csr(X, X_rows, out_rows, out)
        # 断言 out 和 expect 矩阵数据近似相等
        assert_array_equal(out, expect)


# 定义测试函数 test_inplace_column_scale
def test_inplace_column_scale():
    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)
    # 创建稀疏矩阵 X，形状为 (100, 200)，稀疏度为 0.05
    X = sp.rand(100, 200, 0.05)
    # 将 X 转换为 CSR 格式，得到 Xr
    Xr = X.tocsr()
    # 将 X 转换为 CSC 格式，得到 Xc
    Xc = X.tocsc()
    # 将 X 转换为普通数组格式，得到 XA
    XA = X.toarray()
    # 创建长度为 200 的随机数数组 scale
    scale = rng.rand(200)
    # 对 XA 进行列乘法运算，将 XA 中的每一列与 scale 中对应位置的值相乘
    XA *= scale

    # 调用 inplace_column_scale 函数，对 Xc 进行原地列乘法操作
    inplace_column_scale(Xc, scale)
    # 调用 inplace_column_scale 函数，对 Xr 进行原地列乘法操作
    inplace_column_scale(Xr, scale)
    # 断言 Xr 和 Xc 的稀疏矩阵数据近似相等
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    # 断言 XA 和 Xc 的稀疏矩阵数据近似相等
    assert_array_almost_equal(XA, Xc.toarray())
    # 断言 XA 和 Xr 的稀疏矩阵数据近似相等
    assert_array_almost_equal(XA, Xr.toarray())
    # 使用 pytest.raises 检查 TypeError 是否被引发，传入 X.tolil() 和 scale 作为参数
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)

    # 将 X 的数据类型转换为 np.float32
    X = X.astype(np.float32)
    # 将 scale 的数据类型转换为 np.float32
    scale = scale.astype(np.float32)
    # 将 X 转换为 CSR 格式，得到 Xr
    Xr = X.tocsr()
    # 将 X 转换为 CSC 格式，得到 Xc
    Xc = X.tocsc()
    # 将 X 转换为普通数组格式，得到 XA
    XA = X.toarray()
    # 对 XA 进行列乘法运算，将 XA 中的每一列与 scale 中对应位置的值相乘
    XA *= scale

    # 调用 inplace_column_scale 函数，对 Xc 进行原地列乘法操作
    inplace_column_scale(Xc, scale)
    # 调用 inplace_column_scale 函数，对 Xr 进行原地列乘法操作
    inplace_column_scale(Xr, scale)
    # 断言 Xr 和 Xc 的稀疏矩阵数据近似相等
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    # 断言 XA 和 Xc 的稀疏矩阵数据近似相等
    assert_array_almost_equal(XA, Xc.toarray())
    # 断言 XA 和 Xr 的稀疏矩阵数据近似相等
    assert_array_almost_equal(XA, Xr.toarray())
    # 使用 pytest.raises 检查 TypeError 是否被引发，传入 X.tolil() 和 scale 作为参数
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)


# 定义测试函数 test_inplace_row_scale
def test_inplace_row_scale():
    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)
    # 创建稀疏矩阵 X，形状为 (100, 200)，稀疏度为 0.05
    X = sp.rand(100, 200, 0.05)
    # 将 X 转换为 CSR 格式，得到 Xr
    Xr = X.tocsr()
    # 将 X 转换为 CSC 格式，得到 Xc
    Xc = X.tocsc()
    # 将 X 转换为普通数组格式，得到 XA
    XA = X.toarray()
    # 创建长度为 100 的随机数数组 scale
    scale = rng.rand(100)
    # 对 XA 进行行乘法运算，将 XA 中的每一行与 scale 中对应位置的值相乘
    XA *= scale.reshape(-1, 1)

    # 调用 inplace_row_scale 函数，对 Xc 进行原地行乘法操作
    inplace_row_scale(Xc, scale)
    # 调用 inplace_row_scale 函数，对 Xr 进行原地行乘法操作
    inplace_row_scale(Xr, scale)
    # 断言 Xr 和 Xc 的稀疏矩阵数据近似相等
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    # 断言 XA 和 Xc 的稀疏矩阵数据近似相等
    assert_array_almost_equal(XA, Xc.toarray())
    # 断言 XA 和 Xr 的稀疏矩阵数据近似相等
    assert_array_almost_equal(XA, Xr.toarray())
    # 使用 pytest.raises 检查 TypeError 是否被引发，传入 X.tolil() 和 scale 作为参数
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)

    # 将 X 的数据类型转换为 np.float32
    X = X.astype(np.float32)
    # 将 scale 的数据类型转换为 np.float32
    scale = scale.astype(np.float32)
    # 将 X 转换为 CSR 格式，得到 Xr
    Xr = X.tocsr()
    # 将 X 转换为 CSC 格式，得到 Xc
    Xc = X.tocsc()
    # 将 X 转换为普通数组格式，得到 XA
    XA = X.toarray()
    # 对 XA 进行行乘法运算，将 XA 中的每一行与 scale 中对应位置的值相乘
    XA *= scale.reshape(-1,
    # 调用函数 inplace_swap_row 对 X_csr 和 X_csc 进行原地交换第一行和最后一行的操作
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    # 断言两种稀疏矩阵表示方式（CSR 和 CSC）的数组形式相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言 X 和 X_csc 的数组形式相等
    assert_array_equal(X, X_csc.toarray())
    # 断言 X 和 X_csr 的数组形式相等
    assert_array_equal(X, X_csr.toarray())

    # 对 X 中第二行和第三行的元素进行交换，并同步在 X_csr 和 X_csc 中进行原地交换
    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    # 断言两种稀疏矩阵表示方式（CSR 和 CSC）的数组形式相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言 X 和 X_csc 的数组形式相等
    assert_array_equal(X, X_csc.toarray())
    # 断言 X 和 X_csr 的数组形式相等
    assert_array_equal(X, X_csr.toarray())
    # 使用 pytest 断言在尝试对 X_csr 进行 to_lil 操作时会引发 TypeError 异常
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())

    # 创建一个新的 numpy 数组 X，指定其形状和数据类型
    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float32
    )
    # 使用 csr_container 函数将 X 转换为 CSR 稀疏矩阵格式
    X_csr = csr_container(X)
    # 使用 csc_container 函数将 X 转换为 CSC 稀疏矩阵格式
    X_csc = csc_container(X)
    # 从 linalg.get_blas_funcs 中获取 swap 函数，并指定 X 作为参数
    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    # 对 X 中第一行和最后一行的元素进行交换，并同步在 X_csr 和 X_csc 中进行原地交换
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    # 断言两种稀疏矩阵表示方式（CSR 和 CSC）的数组形式相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言 X 和 X_csc 的数组形式相等
    assert_array_equal(X, X_csc.toarray())
    # 断言 X 和 X_csr 的数组形式相等
    assert_array_equal(X, X_csr.toarray())
    # 对 X 中第三行和第四行的元素进行交换，并同步在 X_csr 和 X_csc 中进行原地交换
    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    # 断言两种稀疏矩阵表示方式（CSR 和 CSC）的数组形式相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言 X 和 X_csc 的数组形式相等
    assert_array_equal(X, X_csc.toarray())
    # 断言 X 和 X_csr 的数组形式相等
    assert_array_equal(X, X_csr.toarray())
    # 使用 pytest 断言在尝试对 X_csr 进行 to_lil 操作时会引发 TypeError 异常
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())
# 使用参数化测试框架，循环执行每个(csc_container, csr_container)组合的测试用例
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_inplace_swap_column(csc_container, csr_container):
    # 创建一个3x3的浮点64位数组
    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    # 使用csr_container将X转换成稀疏矩阵
    X_csr = csr_container(X)
    # 使用csc_container将X转换成稀疏矩阵
    X_csc = csc_container(X)

    # 获取BLAS函数中的swap函数，并将其赋值给swap变量
    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    # 对X的第一列和最后一列进行元素交换
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    # 对X_csr中的第0列和最后一列进行原地交换
    inplace_swap_column(X_csr, 0, -1)
    # 对X_csc中的第0列和最后一列进行原地交换
    inplace_swap_column(X_csc, 0, -1)
    # 断言X_csr和X_csc的稀疏表示是否相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言X和X_csc的稀疏表示是否相等
    assert_array_equal(X, X_csc.toarray())
    # 断言X和X_csr的稀疏表示是否相等
    assert_array_equal(X, X_csr.toarray())

    # 对X的第一列和第二列进行元素交换
    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    # 对X_csr中的第0列和第1列进行原地交换
    inplace_swap_column(X_csr, 0, 1)
    # 对X_csc中的第0列和第1列进行原地交换
    inplace_swap_column(X_csc, 0, 1)
    # 断言X_csr和X_csc的稀疏表示是否相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言X和X_csc的稀疏表示是否相等
    assert_array_equal(X, X_csc.toarray())
    # 断言X和X_csr的稀疏表示是否相等
    assert_array_equal(X, X_csr.toarray())
    # 使用pytest断言检查调用tolil方法时是否引发TypeError异常
    with pytest.raises(TypeError):
        inplace_swap_column(X_csr.tolil())

    # 使用浮点32位数组重新设置X的值
    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float32
    )
    # 使用csr_container将X转换成稀疏矩阵
    X_csr = csr_container(X)
    # 使用csc_container将X转换成稀疏矩阵
    X_csc = csc_container(X)
    # 获取BLAS函数中的swap函数，并将其赋值给swap变量
    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    # 对X的第一列和最后一列进行元素交换
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    # 对X_csr中的第0列和最后一列进行原地交换
    inplace_swap_column(X_csr, 0, -1)
    # 对X_csc中的第0列和最后一列进行原地交换
    inplace_swap_column(X_csc, 0, -1)
    # 断言X_csr和X_csc的稀疏表示是否相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言X和X_csc的稀疏表示是否相等
    assert_array_equal(X, X_csc.toarray())
    # 断言X和X_csr的稀疏表示是否相等
    assert_array_equal(X, X_csr.toarray())
    # 对X的第一列和第二列进行元素交换
    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    # 对X_csr中的第0列和第1列进行原地交换
    inplace_swap_column(X_csr, 0, 1)
    # 对X_csc中的第0列和第1列进行原地交换
    inplace_swap_column(X_csc, 0, 1)
    # 断言X_csr和X_csc的稀疏表示是否相等
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    # 断言X和X_csc的稀疏表示是否相等
    assert_array_equal(X, X_csc.toarray())
    # 断言X和X_csr的稀疏表示是否相等
    assert_array_equal(X, X_csr.toarray())
    # 使用pytest断言检查调用tolil方法时是否引发TypeError异常
    with pytest.raises(TypeError):
        inplace_swap_column(X_csr.tolil())


# 使用参数化测试框架，循环执行每个测试用例的组合
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("sparse_format", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize(
    "missing_values, min_func, max_func, ignore_nan",
    [(0, np.min, np.max, False), (np.nan, np.nanmin, np.nanmax, True)],
)
@pytest.mark.parametrize("large_indices", [True, False])
def test_min_max(
    dtype,
    axis,
    sparse_format,
    missing_values,
    min_func,
    max_func,
    ignore_nan,
    large_indices,
):
    # 创建一个包含不同数据类型和缺失值的数组X
    X = np.array(
        [
            [0, 3, 0],
            [2, -1, missing_values],
            [0, 0, 0],
            [9, missing_values, 7],
            [4, 0, 5],
        ],
        dtype=dtype,
    )
    # 使用sparse_format将X转换成稀疏矩阵X_sparse
    X_sparse = sparse_format(X)

    # 如果large_indices为True，将X_sparse的indices和indptr属性转换为int64类型
    if large_indices:
        X_sparse.indices = X_sparse.indices.astype("int64")
        X_sparse.indptr = X_sparse.indptr.astype("int64")

    # 调用min_max_axis函数计算稀疏矩阵X_sparse在指定轴上的最小值和最大值
    mins_sparse, maxs_sparse = min_max_axis(X_sparse, axis=axis, ignore_nan=ignore_nan)
    # 断言mins_sparse和maxs_sparse与np.min和np.max函数计算的结果是否相等
    assert_array_equal(mins_sparse, min_func(X, axis=axis))
    assert_array_equal(maxs_sparse, max_func(X, axis=axis))
# 使用 pytest.mark.parametrize 装饰器为每个测试用例参数化 CSC 和 CSR 容器
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_min_max_axis_errors(csc_container, csr_container):
    # 创建一个 5x3 的二维数组 X，包含浮点数元素
    X = np.array(
        [[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    # 使用 CSR 和 CSC 容器封装数组 X
    X_csr = csr_container(X)
    X_csc = csc_container(X)

    # 测试 min_max_axis 函数抛出 TypeError 异常
    with pytest.raises(TypeError):
        min_max_axis(X_csr.tolil(), axis=0)
    
    # 测试 min_max_axis 函数抛出 ValueError 异常
    with pytest.raises(ValueError):
        min_max_axis(X_csr, axis=2)
    
    # 测试 min_max_axis 函数抛出 ValueError 异常
    with pytest.raises(ValueError):
        min_max_axis(X_csc, axis=-3)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_count_nonzero(csc_container, csr_container):
    # 创建一个 5x3 的二维数组 X，包含浮点数元素
    X = np.array(
        [[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    # 使用 CSR 和 CSC 容器封装数组 X
    X_csr = csr_container(X)
    X_csc = csc_container(X)

    # 创建一个布尔数组 X_nonzero，标记 X 中非零元素
    X_nonzero = X != 0
    sample_weight = [0.5, 0.2, 0.3, 0.1, 0.1]
    # 对 X_nonzero 应用样本权重，得到 X_nonzero_weighted
    X_nonzero_weighted = X_nonzero * np.array(sample_weight)[:, None]

    # 遍历多个轴（包括 None），验证 count_nonzero 的输出与 X_nonzero.sum(axis=axis) 的准确性
    for axis in [0, 1, -1, -2, None]:
        assert_array_almost_equal(
            count_nonzero(X_csr, axis=axis), X_nonzero.sum(axis=axis)
        )
        assert_array_almost_equal(
            count_nonzero(X_csr, axis=axis, sample_weight=sample_weight),
            X_nonzero_weighted.sum(axis=axis),
        )

    # 测试 count_nonzero 函数抛出 TypeError 异常
    with pytest.raises(TypeError):
        count_nonzero(X_csc)
    
    # 测试 count_nonzero 函数抛出 ValueError 异常
    with pytest.raises(ValueError):
        count_nonzero(X_csr, axis=2)

    # 验证 count_nonzero 输出的 dtype 一致性
    assert count_nonzero(X_csr, axis=0).dtype == count_nonzero(X_csr, axis=1).dtype
    assert (
        count_nonzero(X_csr, axis=0, sample_weight=sample_weight).dtype
        == count_nonzero(X_csr, axis=1, sample_weight=sample_weight).dtype
    )

    # 检查处理大稀疏矩阵时的 dtype 一致性
    # XXX: 在 32 位系统上（Windows/Linux）测试失败
    try:
        X_csr.indices = X_csr.indices.astype(np.int64)
        X_csr.indptr = X_csr.indptr.astype(np.int64)
        assert count_nonzero(X_csr, axis=0).dtype == count_nonzero(X_csr, axis=1).dtype
        assert (
            count_nonzero(X_csr, axis=0, sample_weight=sample_weight).dtype
            == count_nonzero(X_csr, axis=1, sample_weight=sample_weight).dtype
        )
    except TypeError as e:
        # 捕获类型错误并验证异常消息中的特定内容
        assert "according to the rule 'safe'" in e.args[0] and np.intp().nbytes < 8, e


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csc_row_median(csc_container, csr_container):
    # 测试 csc_row_median 函数是否正确计算中位数

    # 使用随机数种子创建随机数生成器
    rng = np.random.RandomState(0)
    
    # 创建一个大小为 100x50 的随机浮点数数组 X
    X = rng.rand(100, 50)
    
    # 计算 X 的密集表示下每列的中位数
    dense_median = np.median(X, axis=0)
    
    # 使用 CSC 容器封装数组 X
    csc = csc_container(X)
    
    # 计算稀疏表示下每列的中位数
    sparse_median = csc_median_axis_0(csc)
    
    # 验证稀疏表示和密集表示下的中位数是否一致
    assert_array_equal(sparse_median, dense_median)

    # 创建一个大小为 51x100 的随机浮点数数组 X，并将小于 0.7 的元素置为 0
    X = rng.rand(51, 100)
    X[X < 0.7] = 0.0
    # 从指定范围内生成包含10个随机整数的数组下标
    ind = rng.randint(0, 50, 10)
    # 将数组 X 中指定下标的元素取相反数
    X[ind] = -X[ind]
    # 将数组 X 转换为压缩稀疏列 (CSC) 的容器
    csc = csc_container(X)
    # 计算数组 X 沿第0轴的密集中位数
    dense_median = np.median(X, axis=0)
    # 计算压缩稀疏列 (CSC) 中数组 X 沿第0轴的中位数
    sparse_median = csc_median_axis_0(csc)
    # 断言稀疏中位数与密集中位数相等
    assert_array_equal(sparse_median, dense_median)

    # 测试用于简单数据的函数功能
    X = [[0, -2], [-1, -1], [1, 0], [2, 1]]
    # 将数组 X 转换为压缩稀疏列 (CSC) 的容器，并断言其第0轴中位数
    assert_array_equal(csc_median_axis_0(csc_container(X)), np.array([0.5, -0.5]))
    X = [[0, -2], [-1, -5], [1, -3]]
    # 将数组 X 转换为压缩稀疏列 (CSC) 的容器，并断言其第0轴中位数
    assert_array_equal(csc_median_axis_0(csc_container(X)), np.array([0.0, -3]))

    # 测试传入非压缩稀疏列 (CSC) 矩阵时是否引发 TypeError 异常
    with pytest.raises(TypeError):
        csc_median_axis_0(csr_container(X))
@pytest.mark.parametrize(
    "inplace_csr_row_normalize",
    (inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2),
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数 test_inplace_normalize，参数包括 CSR 容器类型和归一化函数
def test_inplace_normalize(csr_container, inplace_csr_row_normalize):
    # 如果 CSR 容器是 sp.csr_matrix 类型
    if csr_container is sp.csr_matrix:
        # 创建一个全为 1 的数组，形状为 (10, 1)
        ones = np.ones((10, 1))
    else:
        # 创建一个全为 1 的数组，形状为 (10,)
        ones = np.ones(10)
    # 创建一个随机状态对象，种子为 10
    rs = RandomState(10)

    # 对于每种数据类型（np.float64 和 np.float32）
    for dtype in (np.float64, np.float32):
        # 生成一个形状为 (10, 5) 的随机数组 X，类型转换为指定的 dtype
        X = rs.randn(10, 5).astype(dtype)
        # 将数组 X 转换为 CSR 格式
        X_csr = csr_container(X)
        # 对于索引数据类型为 np.int32 和 np.int64
        for index_dtype in [np.int32, np.int64]:
            # 如果索引数据类型是 np.int64，则将 X_csr 的 indptr 和 indices 转换为 np.int64 类型
            if index_dtype is np.int64:
                X_csr.indptr = X_csr.indptr.astype(index_dtype)
                X_csr.indices = X_csr.indices.astype(index_dtype)
            # 断言 X_csr 的 indices 和 indptr 的数据类型与 index_dtype 相符
            assert X_csr.indices.dtype == index_dtype
            assert X_csr.indptr.dtype == index_dtype
            # 调用 inplace_csr_row_normalize 函数对 X_csr 进行行归一化操作
            inplace_csr_row_normalize(X_csr)
            # 断言 X_csr 的数据类型为 dtype
            assert X_csr.dtype == dtype
            # 如果 inplace_csr_row_normalize 是 inplace_csr_row_normalize_l2 函数，则对 X_csr 的数据平方
            if inplace_csr_row_normalize is inplace_csr_row_normalize_l2:
                X_csr.data **= 2
            # 断言 X_csr 经过绝对值求和后与 ones 数组相等
            assert_array_almost_equal(np.abs(X_csr).sum(axis=1), ones)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 定义测试函数 test_csr_row_norms，参数为数据类型 dtype
def test_csr_row_norms(dtype):
    # 创建一个稀疏矩阵 X，形状为 (100, 10)，格式为 CSR，数据类型为 dtype，种子为 42
    X = sp.random(100, 10, format="csr", dtype=dtype, random_state=42)

    # 使用 scipy.sparse.linalg.norm 计算 X 的每行范数的平方
    scipy_norms = sp.linalg.norm(X, axis=1) ** 2
    # 调用 csr_row_norms 函数计算 X 的每行范数的平方
    norms = csr_row_norms(X)

    # 断言 norms 的数据类型为 dtype
    assert norms.dtype == dtype
    # 如果 dtype 是 np.float32，则设置相对误差容限为 1e-6；否则为 1e-7
    rtol = 1e-6 if dtype == np.float32 else 1e-7
    # 断言 norms 和 scipy_norms 的近似相等性，相对误差容限为 rtol
    assert_allclose(norms, scipy_norms, rtol=rtol)


@pytest.fixture(scope="module", params=CSR_CONTAINERS + CSC_CONTAINERS)
# 定义 fixture 函数 centered_matrices，参数为 CSR 和 CSC 容器类型
def centered_matrices(request):
    """Returns equivalent tuple[sp.linalg.LinearOperator, np.ndarray]."""
    # 获取请求参数中的稀疏容器类型
    sparse_container = request.param

    # 创建一个随机状态对象，种子为 42
    random_state = np.random.default_rng(42)

    # 创建一个稀疏矩阵 X_sparse，形状为 (500, 100)，稀疏度为 0.1，格式为 CSR，随机状态为 random_state
    X_sparse = sparse_container(
        sp.random(500, 100, density=0.1, format="csr", random_state=random_state)
    )
    # 将 X_sparse 转换为密集矩阵 X_dense
    X_dense = X_sparse.toarray()
    # 计算 X_sparse 每列的均值，转换为一维数组 mu
    mu = np.asarray(X_sparse.mean(axis=0)).ravel()

    # 对 X_sparse 进行隐式列偏移处理，得到 X_sparse_centered
    X_sparse_centered = _implicit_column_offset(X_sparse, mu)
    # 对 X_dense 进行均值中心化处理，得到 X_dense_centered
    X_dense_centered = X_dense - mu

    # 返回中心化后的稀疏矩阵和密集矩阵的元组
    return X_sparse_centered, X_dense_centered


# 定义测试函数 test_implicit_center_matmat，参数包括全局随机种子和中心化矩阵元组 centered_matrices
def test_implicit_center_matmat(global_random_seed, centered_matrices):
    # 获取中心化后的稀疏矩阵和密集矩阵
    X_sparse_centered, X_dense_centered = centered_matrices
    # 创建一个随机数生成器，种子为 global_random_seed
    rng = np.random.default_rng(global_random_seed)
    # 生成一个形状为 (X_dense_centered.shape[1], 50) 的标准正态分布随机矩阵 Y
    Y = rng.standard_normal((X_dense_centered.shape[1], 50))
    # 断言 X_dense_centered 与 X_sparse_centered.matmat(Y) 的近似相等性
    assert_allclose(X_dense_centered @ Y, X_sparse_centered.matmat(Y))
    # 断言 X_dense_centered 与 X_sparse_centered @ Y 的近似相等性
    assert_allclose(X_dense_centered @ Y, X_sparse_centered @ Y)


# 定义测试函数 test_implicit_center_matvec，参数包括全局随机种子和中心化矩阵元组 centered_matrices
def test_implicit_center_matvec(global_random_seed, centered_matrices):
    # 获取中心化后的稀疏矩阵和密集矩阵
    X_sparse_centered, X_dense_centered = centered_matrices
    # 创建一个随机数生成器，种子为 global_random_seed
    rng = np.random.default_rng(global_random_seed)
    # 生成一个形状为 X_dense_centered.shape[1] 的标准正态分布随机向量 y
    y = rng.standard_normal(X_dense_centered.shape[1])
    # 使用 `assert_allclose` 函数比较两个向量的乘积，左边是密集矩阵 `X_dense_centered` 与向量 `y` 的乘积，
    # 右边是稀疏矩阵 `X_sparse_centered` 对向量 `y` 的矩阵向量乘积操作。
    assert_allclose(X_dense_centered @ y, X_sparse_centered.matvec(y))
    
    # 使用 `assert_allclose` 函数再次比较两个向量的乘积，左边是密集矩阵 `X_dense_centered` 与向量 `y` 的乘积，
    # 右边同样是稀疏矩阵 `X_sparse_centered` 与向量 `y` 的乘积操作。
    assert_allclose(X_dense_centered @ y, X_sparse_centered @ y)
# 测试隐式中心化的稀疏和密集矩阵之间的矩阵乘法
def test_implicit_center_rmatmat(global_random_seed, centered_matrices):
    # 解包中心化后的稀疏和密集矩阵
    X_sparse_centered, X_dense_centered = centered_matrices
    # 使用全局随机种子创建随机数生成器
    rng = np.random.default_rng(global_random_seed)
    # 创建一个形状为 (X_dense_centered.shape[0], 50) 的标准正态分布随机矩阵 Y
    Y = rng.standard_normal((X_dense_centered.shape[0], 50))
    # 断言稀疏矩阵的转置与 Y 的乘积近似等于稠密矩阵的转置与 Y 的乘积
    assert_allclose(X_dense_centered.T @ Y, X_sparse_centered.rmatmat(Y))
    # 断言稠密矩阵的转置与 Y 的乘积近似等于稀疏矩阵转置与 Y 的乘积
    assert_allclose(X_dense_centered.T @ Y, X_sparse_centered.T @ Y)


# 测试隐式中心化的稀疏和密集矩阵之间的向量乘法
def test_implit_center_rmatvec(global_random_seed, centered_matrices):
    # 解包中心化后的稀疏和密集矩阵
    X_sparse_centered, X_dense_centered = centered_matrices
    # 使用全局随机种子创建随机数生成器
    rng = np.random.default_rng(global_random_seed)
    # 创建一个形状为 X_dense_centered.shape[0] 的标准正态分布随机向量 y
    y = rng.standard_normal(X_dense_centered.shape[0])
    # 断言稠密矩阵的转置与向量 y 的乘积近似等于稀疏矩阵的 rmatvec(y)
    assert_allclose(X_dense_centered.T @ y, X_sparse_centered.rmatvec(y))
    # 断言稠密矩阵的转置与向量 y 的乘积近似等于稀疏矩阵转置与向量 y 的乘积
    assert_allclose(X_dense_centered.T @ y, X_sparse_centered.T @ y)
```