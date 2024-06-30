# `D:\src\scipysrc\scikit-learn\sklearn\impute\tests\test_impute.py`

```
# 导入所需的模块和函数
import io  # 用于字节流的操作
import re  # 用于正则表达式的处理
import warnings  # 用于警告处理
from itertools import product  # 用于迭代器的操作

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架
from scipy import sparse  # 导入 SciPy 稀疏矩阵模块
from scipy.stats import kstest  # 导入 SciPy 统计模块中的 Kolmogorov-Smirnov 测试

from sklearn import tree  # 导入 Scikit-Learn 决策树模块
from sklearn.datasets import load_diabetes  # 导入 Scikit-Learn 的糖尿病数据集加载函数
from sklearn.dummy import DummyRegressor  # 导入 Scikit-Learn 虚拟回归器
from sklearn.exceptions import ConvergenceWarning  # 导入 Scikit-Learn 收敛警告

# 使 IterativeImputer 可用
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer  # 导入 Scikit-Learn 缺失值处理模块
from sklearn.impute._base import _most_frequent  # 导入 Scikit-Learn 缺失值处理的内部函数
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV  # 导入 Scikit-Learn 线性回归模型
from sklearn.model_selection import GridSearchCV  # 导入 Scikit-Learn 网格搜索交叉验证
from sklearn.pipeline import Pipeline, make_union  # 导入 Scikit-Learn 管道和并集构造函数
from sklearn.random_projection import _sparse_random_matrix  # 导入 Scikit-Learn 随机投影函数
from sklearn.utils._testing import (  # 导入 Scikit-Learn 测试工具函数
    _convert_container,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import (  # 导入 Scikit-Learn 修复功能
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    LIL_CONTAINERS,
)


def _assert_array_equal_and_same_dtype(x, y):
    # 断言两个数组相等且数据类型相同
    assert_array_equal(x, y)
    assert x.dtype == y.dtype


def _assert_allclose_and_same_dtype(x, y):
    # 断言两个数组接近且数据类型相同
    assert_allclose(x, y)
    assert x.dtype == y.dtype


def _check_statistics(
    X, X_true, strategy, statistics, missing_values, sparse_container
):
    """测试给定策略的缺失值处理效果的实用函数。

    用稠密和稀疏数组进行测试。

    检查：
        - 统计值（均值、中位数、众数）是否正确
        - 缺失值是否被正确填充"""

    err_msg = "Parameters: strategy = %s, missing_values = %s, sparse = {0}" % (
        strategy,
        missing_values,
    )

    assert_ae = assert_array_equal

    if X.dtype.kind == "f" or X_true.dtype.kind == "f":
        assert_ae = assert_array_almost_equal

    # 普通矩阵
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    X_trans = imputer.fit(X).transform(X.copy())
    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(False))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(False))

    # 稀疏矩阵
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(sparse_container(X))
    X_trans = imputer.transform(sparse_container(X.copy()))

    if sparse.issparse(X_trans):
        X_trans = X_trans.toarray()

    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(True))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(True))


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_imputation_shape(strategy, csr_container):
    # 验证不同策略下填充后矩阵的形状。
    X = np.random.randn(10, 2)
    X[::2] = np.nan

    imputer = SimpleImputer(strategy=strategy)
    # 使用 imputer 对象对 X 进行拟合和转换，得到填充后的数据 X_imputed
    X_imputed = imputer.fit_transform(csr_container(X))
    # 断言填充后的数据 X_imputed 的形状应为 (10, 2)
    assert X_imputed.shape == (10, 2)
    
    # 使用 imputer 对象对原始 X 进行拟合和转换，覆盖之前的 X_imputed
    X_imputed = imputer.fit_transform(X)
    # 断言再次填充后的数据 X_imputed 的形状应为 (10, 2)
    assert X_imputed.shape == (10, 2)
    
    # 创建 iterative_imputer 对象，使用指定的初始策略对 X 进行拟合和转换
    iterative_imputer = IterativeImputer(initial_strategy=strategy)
    # 使用 iterative_imputer 对象对 X 进行拟合和转换，得到填充后的数据 X_imputed
    X_imputed = iterative_imputer.fit_transform(X)
    # 断言填充后的数据 X_imputed 的形状应为 (10, 2)
    assert X_imputed.shape == (10, 2)
# 使用 pytest 的 mark 来参数化测试函数，测试不同的策略：均值、中位数、最频繁值
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputation_deletion_warning(strategy):
    # 创建一个 3x5 的全为 1 的数组 X
    X = np.ones((3, 5))
    # 将 X 的所有行的第一列设为 NaN
    X[:, 0] = np.nan
    # 使用 SimpleImputer 对象，设定策略并拟合 X
    imputer = SimpleImputer(strategy=strategy).fit(X)

    # 测试是否会发出 UserWarning 并匹配 "Skipping" 提示
    with pytest.warns(UserWarning, match="Skipping"):
        imputer.transform(X)


# 使用 pytest 的 mark 来参数化测试函数，测试不同的策略：均值、中位数、最频繁值
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputation_deletion_warning_feature_names(strategy):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 定义缺失值为 NaN
    missing_values = np.nan
    # 定义特征名称数组为 ["a", "b", "c", "d"]
    feature_names = np.array(["a", "b", "c", "d"], dtype=object)
    # 创建一个包含两行数据的 pandas DataFrame 对象 X
    X = pd.DataFrame(
        [
            [missing_values, missing_values, 1, missing_values],
            [4, missing_values, 2, 10],
        ],
        columns=feature_names,
    )

    # 使用 SimpleImputer 对象，设定策略并拟合 X
    imputer = SimpleImputer(strategy=strategy).fit(X)

    # 检查 SimpleImputer 是否正确返回特征名属性
    assert_array_equal(imputer.feature_names_in_, feature_names)

    # 确保 "Skipping" 提示包含特征名称 'b'
    with pytest.warns(
        UserWarning, match=r"Skipping features without any observed values: \['b'\]"
    ):
        imputer.transform(X)


# 使用 pytest 的 mark 来参数化测试函数，测试不同的策略：均值、中位数、最频繁值、常数
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_error_sparse_0(strategy, csc_container):
    # 检查当 missing_values = 0 且输入为稀疏矩阵时是否引发错误
    X = np.ones((3, 5))
    X[0] = 0
    # 使用 csc_container 将 X 转换为稀疏矩阵
    X = csc_container(X)

    # 创建 SimpleImputer 对象，设定策略和 missing_values=0
    imputer = SimpleImputer(strategy=strategy, missing_values=0)

    # 测试是否引发 ValueError，并匹配 "Provide a dense array"
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.fit(X)

    # 对 X 转换为稠密数组后进行拟合
    imputer.fit(X.toarray())
    # 再次测试是否引发 ValueError，并匹配 "Provide a dense array"
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.transform(X)


# 定义一个安全的中位数计算函数，处理空数组时返回 np.nan
def safe_median(arr, *args, **kwargs):
    # 如果 arr 有 size 属性，则获取其大小，否则获取其长度
    length = arr.size if hasattr(arr, "size") else len(arr)
    # 如果 length 为 0，则返回 np.nan，否则计算中位数
    return np.nan if length == 0 else np.median(arr, *args, **kwargs)


# 定义一个安全的均值计算函数，处理空数组时返回 np.nan
def safe_mean(arr, *args, **kwargs):
    # 如果 arr 有 size 属性，则获取其大小，否则获取其长度
    length = arr.size if hasattr(arr, "size") else len(arr)
    # 如果 length 为 0，则返回 np.nan，否则计算均值
    return np.nan if length == 0 else np.mean(arr, *args, **kwargs)


# 使用 pytest 的 mark 来参数化测试函数，测试不同的 csc_container
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_mean_median(csc_container):
    # 测试使用均值和中位数策略进行插补，当 missing_values != 0 时
    rng = np.random.RandomState(0)

    dim = 10
    dec = 10
    shape = (dim * dim, dim + dec)

    zeros = np.zeros(shape[0])
    values = np.arange(1, shape[0] + 1)
    values[4::2] = -values[4::2]

    # 定义测试用例列表，包括策略名称、缺失值、以及对应的安全计算函数
    tests = [
        ("mean", np.nan, lambda z, v, p: safe_mean(np.hstack((z, v)))),
        ("median", np.nan, lambda z, v, p: safe_median(np.hstack((z, v)))),
    ]
    for strategy, test_missing_values, true_value_fun in tests:
        X = np.empty(shape)
        X_true = np.empty(shape)
        true_statistics = np.empty(shape[1])

        # 对于给定的测试参数，创建空的矩阵 X 和 X_true
        # X 是包含统计数据的矩阵，X_true 是包含真实值的矩阵
        for j in range(shape[1]):
            # 根据当前列 j 的索引计算出各种类型的元素个数
            nb_zeros = (j - dec + 1 > 0) * (j - dec + 1) * (j - dec + 1)
            nb_missing_values = max(shape[0] + dec * dec - (j + dec) * (j + dec), 0)
            nb_values = shape[0] - nb_zeros - nb_missing_values

            # 从预定义的 zeros, test_missing_values 和 values 中取出相应数量的数据
            z = zeros[:nb_zeros]
            p = np.repeat(test_missing_values, nb_missing_values)
            v = values[rng.permutation(len(values))[:nb_values]]

            # 计算真实统计值并存储在 true_statistics 中
            true_statistics[j] = true_value_fun(z, v, p)

            # 创建列数据并填充到 X 的第 j 列
            X[:, j] = np.hstack((v, z, p))

            if 0 == test_missing_values:
                # 当 test_missing_values 为 0 时，应该不会执行到这里
                X_true[:, j] = np.hstack(
                    (v, np.repeat(true_statistics[j], nb_missing_values + nb_zeros))
                )
            else:
                # 将真实值、zeros 和重复的统计值组合填充到 X_true 的第 j 列
                X_true[:, j] = np.hstack(
                    (v, z, np.repeat(true_statistics[j], nb_missing_values))
                )

            # 对 X 和 X_true 的第 j 列进行相同的随机重排列
            np.random.RandomState(j).shuffle(X[:, j])
            np.random.RandomState(j).shuffle(X_true[:, j])

        # 根据策略选择保留不包含 NaN 的列或者所有列
        if strategy == "median":
            cols_to_keep = ~np.isnan(X_true).any(axis=0)
        else:
            cols_to_keep = ~np.isnan(X_true).all(axis=0)

        # 根据 cols_to_keep 对 X_true 进行列筛选
        X_true = X_true[:, cols_to_keep]

        # 检查统计信息的正确性
        _check_statistics(
            X, X_true, strategy, true_statistics, test_missing_values, csc_container
        )
# 使用 pytest 的 @parametrize 装饰器，为 test_imputation_median_special_cases 函数生成多个参数化的测试用例
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_median_special_cases(csc_container):
    # 测试中位数插补在稀疏边界情况下的效果

    # 创建一个包含 NaN 值的 numpy 数组 X，用于测试
    X = np.array(
        [
            [0, np.nan, np.nan],  # 奇数个元素: 隐式零值
            [5, np.nan, np.nan],  # 奇数个元素: 显式非零值
            [0, 0, np.nan],       # 偶数个元素: 平均两个零值
            [-5, 0, np.nan],      # 偶数个元素: 零值与负数的平均值
            [0, 5, np.nan],       # 偶数个元素: 零值与正数的平均值
            [4, 5, np.nan],       # 偶数个元素: 非零值的平均值
            [-4, -5, np.nan],     # 偶数个元素: 负数的平均值
            [-1, 2, np.nan],      # 偶数个元素: 负数与正数的交叉平均值
        ]
    ).transpose()

    # 期望的中位数插补后的结果数组 X_imputed_median
    X_imputed_median = np.array(
        [
            [0, 0, 0],
            [5, 5, 5],
            [0, 0, 0],
            [-5, 0, -2.5],
            [0, 5, 2.5],
            [4, 5, 4.5],
            [-4, -5, -4.5],
            [-1, 2, 0.5],
        ]
    ).transpose()

    # 中位数插补后的统计中位数值
    statistics_median = [0, 5, 0, -2.5, 2.5, 4.5, -4.5, 0.5]

    # 调用 _check_statistics 函数验证中位数插补的结果是否符合预期
    _check_statistics(
        X, X_imputed_median, "median", statistics_median, np.nan, csc_container
    )


# 使用 pytest 的 @parametrize 装饰器，为 test_imputation_mean_median_error_invalid_type 函数生成多个参数化的测试用例
@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("dtype", [None, object, str])
def test_imputation_mean_median_error_invalid_type(strategy, dtype):
    # 创建包含非数值数据的 numpy 数组 X，用于测试均值和中位数插补时的类型错误情况
    X = np.array([["a", "b", 3], [4, "e", 6], ["g", "h", 9]], dtype=dtype)

    # 期望的错误消息
    msg = "non-numeric data:\ncould not convert string to float:"

    # 使用 pytest 的 raises 方法，验证在给定策略下是否会抛出 ValueError 异常并包含特定的错误消息
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


# 使用 pytest 的 @parametrize 装饰器，为 test_imputation_mean_median_error_invalid_type_list_pandas 函数生成多个参数化的测试用例
@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("type", ["list", "dataframe"])
def test_imputation_mean_median_error_invalid_type_list_pandas(strategy, type):
    # 创建包含非数值数据的列表 X，用于测试均值和中位数插补时的类型错误情况
    X = [["a", "b", 3], [4, "e", 6], ["g", "h", 9]]

    # 如果 type 为 'dataframe'，则将列表转换为 pandas 的 DataFrame
    if type == "dataframe":
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)

    # 期望的错误消息
    msg = "non-numeric data:\ncould not convert string to float:"

    # 使用 pytest 的 raises 方法，验证在给定策略下是否会抛出 ValueError 异常并包含特定的错误消息
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


# 使用 pytest 的 @parametrize 装饰器，为 test_imputation_const_mostf_error_invalid_types 函数生成多个参数化的测试用例
@pytest.mark.parametrize("strategy", ["constant", "most_frequent"])
@pytest.mark.parametrize("dtype", [str, np.dtype("U"), np.dtype("S")])
def test_imputation_const_mostf_error_invalid_types(strategy, dtype):
    # 创建包含非数值数据的 numpy 数组 X，用于测试常数和最频繁插补策略时的类型错误情况
    X = np.array(
        [
            [np.nan, np.nan, "a", "f"],
            [np.nan, "c", np.nan, "d"],
            [np.nan, "b", "d", np.nan],
            [np.nan, "c", "d", "h"],
        ],
        dtype=dtype,
    )

    # 期望的错误消息
    err_msg = "SimpleImputer does not support data"

    # 使用 pytest 的 raises 方法，验证在给定策略下是否会抛出 ValueError 异常并包含特定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(X).transform(X)


# 使用 pytest 的 @parametrize 装饰器，为 test_imputation_most_frequent 函数生成多个参数化的测试用例
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_most_frequent(csc_container):
    # 测试使用最频繁值插补策略
    # 创建一个二维数组 X，包含4行4列的数据
    X = np.array(
        [
            [-1, -1, 0, 5],
            [-1, 2, -1, 3],
            [-1, 1, 3, -1],
            [-1, 2, 3, 7],
        ]
    )

    # 创建一个二维数组 X_true，包含4行3列的数据
    X_true = np.array(
        [
            [2, 0, 5],
            [2, 3, 3],
            [1, 3, 3],
            [2, 3, 7],
        ]
    )

    # 注释说明：scipy.stats.mode 在 SimpleImputer 中使用时，与文档中承诺的不同，不返回第一个最频繁出现的值，而是最低频率的值。
    # 当 scipy 更新后，如果这个测试失败，SimpleImputer 需要更新以保持与新的（正确的）行为一致。
    _check_statistics(X, X_true, "most_frequent", [np.nan, 2, 3, 3], -1, csc_container)
@pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
# 定义测试函数 test_imputation_most_frequent_objects，使用参数化测试来测试不同的标记值
def test_imputation_most_frequent_objects(marker):
    # 测试使用最频繁策略进行填充
    X = np.array(
        [
            [marker, marker, "a", "f"],  # 创建一个包含标记值的二维数组
            [marker, "c", marker, "d"],
            [marker, "b", "d", marker],
            [marker, "c", "d", "h"],
        ],
        dtype=object,
    )

    X_true = np.array(
        [
            ["c", "a", "f"],  # 预期的结果数组，标记值被最频繁的非标记值替换
            ["c", "d", "d"],
            ["b", "d", "d"],
            ["c", "d", "h"],
        ],
        dtype=object,
    )

    # 创建一个 SimpleImputer 对象，使用最频繁的策略进行填充
    imputer = SimpleImputer(missing_values=marker, strategy="most_frequent")
    # 对输入数组 X 进行拟合和转换
    X_trans = imputer.fit(X).transform(X)

    # 断言转换后的数组 X_trans 与预期的数组 X_true 相等
    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("dtype", [object, "category"])
# 定义测试函数 test_imputation_most_frequent_pandas，使用参数化测试来测试不同的数据类型
def test_imputation_most_frequent_pandas(dtype):
    # 测试在 pandas 数据框上使用最频繁策略进行填充
    pd = pytest.importorskip("pandas")  # 导入 pytest 并跳过如果导入失败

    # 创建一个包含字符串的文本流对象
    f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")

    # 从文本流中读取数据，并指定数据类型
    df = pd.read_csv(f, dtype=dtype)

    X_true = np.array(
        [["a", "i", "x"], ["a", "j", "y"], ["a", "j", "x"], ["b", "j", "x"]],
        dtype=object,
    )

    # 创建一个 SimpleImputer 对象，使用最频繁的策略进行填充
    imputer = SimpleImputer(strategy="most_frequent")
    # 对数据框 df 进行拟合和转换
    X_trans = imputer.fit_transform(df)

    # 断言转换后的数组 X_trans 与预期的数组 X_true 相等
    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("X_data, missing_value", [(1, 0), (1.0, np.nan)])
# 定义测试函数 test_imputation_constant_error_invalid_type，使用参数化测试来测试不同的填充值和缺失值
def test_imputation_constant_error_invalid_type(X_data, missing_value):
    # 验证在填充值类型不合法时是否引发异常
    X = np.full((3, 5), X_data, dtype=float)  # 创建一个填充了 X_data 的数组

    X[0, 0] = missing_value  # 将指定位置设为缺失值

    fill_value = "x"
    err_msg = f"fill_value={fill_value!r} (of type {type(fill_value)!r}) cannot be cast"
    # 使用 pytest.raises 来检查是否引发了 ValueError 异常，并验证错误消息是否匹配
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer = SimpleImputer(
            missing_values=missing_value, strategy="constant", fill_value=fill_value
        )
        # 对数组 X 进行拟合和转换
        imputer.fit_transform(X)


def test_imputation_constant_integer():
    # 测试在整数数据上使用常数填充策略
    X = np.array([[-1, 2, 3, -1], [4, -1, 5, -1], [6, 7, -1, -1], [8, 9, 0, -1]])

    X_true = np.array([[0, 2, 3, 0], [4, 0, 5, 0], [6, 7, 0, 0], [8, 9, 0, 0]])

    # 创建一个 SimpleImputer 对象，使用常数填充策略
    imputer = SimpleImputer(missing_values=-1, strategy="constant", fill_value=0)
    # 对数组 X 进行拟合和转换
    X_trans = imputer.fit_transform(X)

    # 断言转换后的数组 X_trans 与预期的数组 X_true 相等
    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("array_constructor", CSR_CONTAINERS + [np.asarray])
# 定义测试函数 test_imputation_constant_float，使用参数化测试来测试不同的数组构造函数
def test_imputation_constant_float(array_constructor):
    # 测试在浮点数数据上使用常数填充策略
    X = np.array(
        [
            [np.nan, 1.1, 0, np.nan],
            [1.2, np.nan, 1.3, np.nan],
            [0, 0, np.nan, np.nan],
            [1.4, 1.5, 0, np.nan],
        ]
    )

    X_true = np.array(
        [[-1, 1.1, 0, -1], [1.2, -1, 1.3, -1], [0, 0, -1, -1], [1.4, 1.5, 0, -1]]
    )

    X = array_constructor(X)

    X_true = array_constructor(X_true)
    # 创建一个简单的填充缺失值的对象，使用常数填充策略，缺失值填充为 -1
    imputer = SimpleImputer(strategy="constant", fill_value=-1)
    
    # 使用上述创建的填充器对象对输入数据 X 进行转换处理，将缺失值填充为 -1
    X_trans = imputer.fit_transform(X)
    
    # 对转换后的数据 X_trans 和真实数据 X_true 进行密集与稀疏矩阵的近似比较，确保它们在数值上几乎相等
    assert_allclose_dense_sparse(X_trans, X_true)
@pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
# 使用参数化测试，测试不同的标记值（None, np.nan, "NAN", "", 0）
def test_imputation_constant_object(marker):
    # Test imputation using the constant strategy on objects
    # 测试对对象使用常数策略的填充效果

    X = np.array(
        [
            [marker, "a", "b", marker],
            ["c", marker, "d", marker],
            ["e", "f", marker, marker],
            ["g", "h", "i", marker],
        ],
        dtype=object,
    )

    X_true = np.array(
        [
            ["missing", "a", "b", "missing"],
            ["c", "missing", "d", "missing"],
            ["e", "f", "missing", "missing"],
            ["g", "h", "i", "missing"],
        ],
        dtype=object,
    )

    imputer = SimpleImputer(
        missing_values=marker, strategy="constant", fill_value="missing"
    )
    # 创建一个填充器对象，指定了缺失值和填充策略为常数填充，填充值为"missing"
    X_trans = imputer.fit_transform(X)
    # 对输入数据 X 进行拟合和转换

    assert_array_equal(X_trans, X_true)
    # 断言转换后的数据与预期的真实数据相等


@pytest.mark.parametrize("dtype", [object, "category"])
# 使用参数化测试，测试不同的数据类型（object, "category"）
def test_imputation_constant_pandas(dtype):
    # Test imputation using the constant strategy on pandas df
    # 测试对 Pandas 数据框使用常数策略的填充效果

    pd = pytest.importorskip("pandas")
    # 导入 pytest 并跳过如果导入失败的话

    f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")
    # 创建一个包含 CSV 格式数据的字符串流

    df = pd.read_csv(f, dtype=dtype)
    # 使用 Pandas 读取 CSV 数据并指定数据类型为参数化测试传入的 dtype

    X_true = np.array(
        [
            ["missing_value", "i", "x", "missing_value"],
            ["a", "missing_value", "y", "missing_value"],
            ["a", "j", "missing_value", "missing_value"],
            ["b", "j", "x", "missing_value"],
        ],
        dtype=object,
    )

    imputer = SimpleImputer(strategy="constant")
    # 创建一个填充器对象，使用常数填充策略

    X_trans = imputer.fit_transform(df)
    # 对输入的 Pandas 数据框进行拟合和转换

    assert_array_equal(X_trans, X_true)
    # 断言转换后的数据与预期的真实数据相等


@pytest.mark.parametrize("X", [[[1], [2]], [[1], [np.nan]]])
# 使用参数化测试，测试不同的输入 X
def test_iterative_imputer_one_feature(X):
    # check we exit early when there is a single feature
    # 检查当只有一个特征时，我们能否提前退出

    imputer = IterativeImputer().fit(X)
    # 使用迭代式填充器对 X 进行拟合
    assert imputer.n_iter_ == 0
    # 断言迭代次数为 0

    imputer = IterativeImputer()
    imputer.fit([[1], [2]])
    # 对另一个输入进行拟合
    assert imputer.n_iter_ == 0
    # 断言迭代次数为 0

    imputer.fit([[1], [np.nan]])
    # 对包含 NaN 的输入进行拟合
    assert imputer.n_iter_ == 0
    # 断言迭代次数为 0


def test_imputation_pipeline_grid_search():
    # Test imputation within a pipeline + gridsearch.
    # 测试在管道和网格搜索中的填充效果

    X = _sparse_random_matrix(100, 100, density=0.10)
    # 生成一个稀疏的随机矩阵

    missing_values = X.data[0]
    # 获取稀疏矩阵中的第一个缺失值

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=missing_values)),
            ("tree", tree.DecisionTreeRegressor(random_state=0)),
        ]
    )
    # 创建一个管道，包含一个填充器和一个决策树回归器

    parameters = {"imputer__strategy": ["mean", "median", "most_frequent"]}
    # 设置填充器策略的参数网格

    Y = _sparse_random_matrix(100, 1, density=0.10).toarray()
    # 生成一个稀疏的随机矩阵，并将其转换为稠密数组

    gs = GridSearchCV(pipeline, parameters)
    # 创建一个网格搜索对象，传入管道和参数网格
    gs.fit(X, Y)
    # 对数据进行网格搜索


def test_imputation_copy():
    # Test imputation with copy
    # 测试使用复制功能的填充效果

    X_orig = _sparse_random_matrix(5, 5, density=0.75, random_state=0)
    # 生成一个稀疏的随机矩阵

    # copy=True, dense => copy
    X = X_orig.copy().toarray()
    # 将原始数据复制并转换为稠密数组

    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=True)
    # 创建一个填充器对象，使用平均值填充策略，并设置复制为 True
    Xt = imputer.fit(X).transform(X)
    # 对输入数据进行拟合和转换
    Xt[0, 0] = -1
    # 修改转换后的数据的第一个元素
    assert not np.all(X == Xt)
    # 断言 X 和 Xt 不完全相等

    # copy=True, sparse csr => copy
    X = X_orig.copy()
    # 将原始数据复制
    # 使用 SimpleImputer 类创建一个填充器实例，用于处理缺失值
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=True)
    # 对数据 X 进行拟合和转换，得到填充后的结果 Xt
    Xt = imputer.fit(X).transform(X)
    # 修改 Xt 中的第一个元素为 -1
    Xt.data[0] = -1
    # 断言检查 X 和 Xt 中的数据是否不完全相同
    assert not np.all(X.data == Xt.data)

    # copy=False, dense => no copy
    # 复制 X_orig 并转换为密集数组
    X = X_orig.copy().toarray()
    # 使用 SimpleImputer 填充器处理缺失值，不进行复制
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=False)
    # 对数据 X 进行拟合和转换，得到填充后的结果 Xt
    Xt = imputer.fit(X).transform(X)
    # 修改 Xt 的第一个元素为 -1
    Xt[0, 0] = -1
    # 断言检查 X 和 Xt 是否近似相等
    assert_array_almost_equal(X, Xt)

    # copy=False, sparse csc => no copy
    # 复制 X_orig 并转换为 CSC 格式的稀疏矩阵
    X = X_orig.copy().tocsc()
    # 使用 SimpleImputer 填充器处理缺失值，不进行复制
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    # 对数据 X 进行拟合和转换，得到填充后的结果 Xt
    Xt = imputer.fit(X).transform(X)
    # 修改 Xt 中的第一个元素为 -1
    Xt.data[0] = -1
    # 断言检查 X.data 和 Xt.data 是否近似相等
    assert_array_almost_equal(X.data, Xt.data)

    # copy=False, sparse csr => copy
    # 复制 X_orig 并保持为 CSR 格式的稀疏矩阵
    X = X_orig.copy()
    # 使用 SimpleImputer 填充器处理缺失值，不进行复制
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    # 对数据 X 进行拟合和转换，得到填充后的结果 Xt
    Xt = imputer.fit(X).transform(X)
    # 修改 Xt 中的第一个元素为 -1
    Xt.data[0] = -1
    # 断言检查 X.data 和 Xt.data 是否不完全相同
    assert not np.all(X.data == Xt.data)

    # Note: If X is sparse and if missing_values=0, then a (dense) copy of X is
    # made, even if copy=False.
    # 注意：如果 X 是稀疏矩阵，并且 missing_values=0，即使设置 copy=False，
    # 也会生成 X 的一个（密集）副本。
def test_iterative_imputer_zero_iters():
    # 创建一个随机数生成器对象，种子为0
    rng = np.random.RandomState(0)

    # 设定样本数和特征数
    n = 100
    d = 10

    # 生成稀疏随机矩阵，并转换为稠密数组
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    # 创建一个标志矩阵，标记X中的缺失值位置
    missing_flag = X == 0

    # 将X中的0值替换为NaN
    X[missing_flag] = np.nan

    # 创建一个迭代式填充器对象，最大迭代次数为0
    imputer = IterativeImputer(max_iter=0)

    # 对X进行拟合和转换
    X_imputed = imputer.fit_transform(X)

    # 断言：当max_iter=0时，仅进行初始插补
    assert_allclose(X_imputed, imputer.initial_imputer_.transform(X))

    # 重复上述过程，但强制 n_iter_ 为0
    imputer = IterativeImputer(max_iter=5).fit(X)

    # 断言：转换后的结果不应等于初始插补结果
    assert not np.all(imputer.transform(X) == imputer.initial_imputer_.transform(X))

    # 将 n_iter_ 设置为0
    imputer.n_iter_ = 0

    # 断言：现在它们应该相等，因为只进行了初始插补
    assert_allclose(imputer.transform(X), imputer.initial_imputer_.transform(X))


def test_iterative_imputer_verbose():
    # 创建一个随机数生成器对象，种子为0
    rng = np.random.RandomState(0)

    # 设定样本数和特征数
    n = 100
    d = 3

    # 生成稀疏随机矩阵，并转换为稠密数组
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    # 创建一个迭代式填充器对象，设定missing_values=0, max_iter=1, verbose=1
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=1)

    # 对X进行拟合和转换
    imputer.fit(X)
    imputer.transform(X)

    # 创建一个迭代式填充器对象，设定missing_values=0, max_iter=1, verbose=2
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=2)

    # 对X进行拟合和转换
    imputer.fit(X)
    imputer.transform(X)


def test_iterative_imputer_all_missing():
    # 设定样本数和特征数
    n = 100
    d = 3

    # 创建一个全为0的数组
    X = np.zeros((n, d))

    # 创建一个迭代式填充器对象，设定missing_values=0, max_iter=1
    imputer = IterativeImputer(missing_values=0, max_iter=1)

    # 对X进行拟合和转换
    X_imputed = imputer.fit_transform(X)

    # 断言：拟合后的结果应与初始插补结果相等
    assert_allclose(X_imputed, imputer.initial_imputer_.transform(X))


@pytest.mark.parametrize(
    "imputation_order", ["random", "roman", "ascending", "descending", "arabic"]
)
def test_iterative_imputer_imputation_order(imputation_order):
    # 创建一个随机数生成器对象，种子为0
    rng = np.random.RandomState(0)

    # 设定样本数和特征数
    n = 100
    d = 10

    # 设定最大迭代次数
    max_iter = 2

    # 生成稀疏随机矩阵，并转换为稠密数组
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    # 将第一列所有元素设为1，确保该列不会被迭代式填充器丢弃
    X[:, 0] = 1

    # 创建一个迭代式填充器对象，设定多个参数包括imputation_order
    imputer = IterativeImputer(
        missing_values=0,
        max_iter=max_iter,
        n_nearest_features=5,
        sample_posterior=False,
        skip_complete=True,
        min_value=0,
        max_value=1,
        verbose=1,
        imputation_order=imputation_order,
        random_state=rng,
    )

    # 对X进行拟合和转换
    imputer.fit_transform(X)

    # 获取填充顺序的特征索引列表
    ordered_idx = [i.feat_idx for i in imputer.imputation_sequence_]

    # 断言：每次迭代应该处理的特征数目应等于特征数目减去无缺失值的特征数目
    assert len(ordered_idx) // imputer.n_iter_ == imputer.n_features_with_missing_

    if imputation_order == "roman":
        # 断言：按罗马数字顺序填充的情况下，前d-1个特征应该按顺序为1到d-1
        assert np.all(ordered_idx[: d - 1] == np.arange(1, d))
    elif imputation_order == "arabic":
        # 断言：按阿拉伯数字逆序填充的情况下，前d-1个特征应该按顺序为d-1到1
        assert np.all(ordered_idx[: d - 1] == np.arange(d - 1, 0, -1))
    elif imputation_order == "random":
        # 断言：随机填充的情况下，第一轮和第二轮填充的特征索引列表不应相同
        ordered_idx_round_1 = ordered_idx[: d - 1]
        ordered_idx_round_2 = ordered_idx[d - 1 :]
        assert ordered_idx_round_1 != ordered_idx_round_2
    elif "ending" in imputation_order:
        # 断言：以“ending”结尾的填充顺序，填充的总次数应为最大迭代次数乘以(d-1)
        assert len(ordered_idx) == max_iter * (d - 1)
    # 定义一个包含字符串 "estimator" 和一组回归器对象的列表
    "estimator", [None, DummyRegressor(), BayesianRidge(), ARDRegression(), RidgeCV()]
# 定义一个测试函数，用于测试迭代式填充估算器的行为
def test_iterative_imputer_estimators(estimator):
    # 创建一个随机数生成器，种子为0
    rng = np.random.RandomState(0)

    # 设置样本数量和特征数量
    n = 100
    d = 10
    # 生成稀疏随机矩阵，并将其转换为稠密数组
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    # 创建迭代式填充估算器对象
    imputer = IterativeImputer(
        missing_values=0, max_iter=1, estimator=estimator, random_state=rng
    )
    # 对输入数据进行拟合和转换
    imputer.fit_transform(X)

    # 检查估算器的类型是否正确
    hashes = []
    for triplet in imputer.imputation_sequence_:
        # 确定预期的估算器类型
        expected_type = (
            type(estimator) if estimator is not None else type(BayesianRidge())
        )
        # 断言估算器的类型是否符合预期
        assert isinstance(triplet.estimator, expected_type)
        # 记录估算器对象的 ID
        hashes.append(id(triplet.estimator))

    # 断言所有估算器对象的 ID 都是唯一的
    assert len(set(hashes)) == len(hashes)


# 定义一个测试函数，用于测试迭代式填充器的截断行为
def test_iterative_imputer_clip():
    # 创建一个随机数生成器，种子为0
    rng = np.random.RandomState(0)
    # 设置样本数量和特征数量
    n = 100
    d = 10
    # 生成稀疏随机矩阵，并将其转换为稠密数组
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    # 创建迭代式填充估算器对象，设定最小值和最大值
    imputer = IterativeImputer(
        missing_values=0, max_iter=1, min_value=0.1, max_value=0.2, random_state=rng
    )

    # 对输入数据进行拟合和转换
    Xt = imputer.fit_transform(X)
    # 断言填充后的最小值等于设定的最小值
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    # 断言填充后的最大值等于设定的最大值
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    # 断言非零值保持不变
    assert_allclose(Xt[X != 0], X[X != 0])


# 定义一个测试函数，用于测试带有截断正态后验分布的迭代式填充器行为
def test_iterative_imputer_clip_truncnorm():
    # 创建一个随机数生成器，种子为0
    rng = np.random.RandomState(0)
    # 设置样本数量和特征数量
    n = 100
    d = 10
    # 生成稀疏随机矩阵，并将其转换为稠密数组
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    # 将第一列全部设置为1
    X[:, 0] = 1

    # 创建迭代式填充估算器对象，设定最小值、最大值、随机种子等参数
    imputer = IterativeImputer(
        missing_values=0,
        max_iter=2,
        n_nearest_features=5,
        sample_posterior=True,
        min_value=0.1,
        max_value=0.2,
        verbose=1,
        imputation_order="random",
        random_state=rng,
    )
    # 对输入数据进行拟合和转换
    Xt = imputer.fit_transform(X)
    # 断言填充后的最小值等于设定的最小值
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    # 断言填充后的最大值等于设定的最大值
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    # 断言非零值保持不变
    assert_allclose(Xt[X != 0], X[X != 0])


# 定义一个测试函数，用于测试带有截断正态后验分布的迭代式填充器行为
def test_iterative_imputer_truncated_normal_posterior():
    # 测试使用 `sample_posterior=True` 填充的值是否符合正态分布的 Kolmogorov-Smirnov 检验
    # 注意，如果随机种子设置错误，将导致此测试失败，因为当填充值超出 (min_value, max_value) 范围时，不会发生随机抽样
    rng = np.random.RandomState(42)

    # 生成一个正态分布的随机矩阵，大小为 (5, 5)
    X = rng.normal(size=(5, 5))
    # 将矩阵中的第一个元素设为 NaN
    X[0][0] = np.nan

    # 创建迭代式填充估算器对象，设定最小值、最大值、是否抽样后验、随机种子等参数
    imputer = IterativeImputer(
        min_value=0, max_value=0.5, sample_posterior=True, random_state=rng
    )

    # 对输入数据进行拟合和转换
    imputer.fit_transform(X)
    # 生成多个填充值，对单个缺失值进行多次填充
    imputations = np.array([imputer.transform(X)[0][0] for _ in range(100)])

    # 断言所有填充值大于等于最小值
    assert all(imputations >= 0)
    # 断言所有填充值小于等于最大值
    assert all(imputations <= 0.5)

    # 计算填充值的均值和标准差
    mu, sigma = imputations.mean(), imputations.std()
    # 计算 Kolmogorov-Smirnov 统计量和 p 值
    ks_statistic, p_value = kstest((imputations - mu) / sigma, "norm")
    # 如果标准差为0，加一个小量以避免除以0
    if sigma == 0:
        sigma += 1e-12
    # 对经过标准化后的数据进行 Kolmogorov-Smirnov 检验，检验其是否服从正态分布
    ks_statistic, p_value = kstest((imputations - mu) / sigma, "norm")
    # 我们希望不能拒绝零假设
    # 零假设: 分布是相同的
    # 使用断言来确保 Kolmogorov-Smirnov 统计量小于 0.2 或者 p 值大于 0.1，否则抛出异常信息
    assert ks_statistic < 0.2 or p_value > 0.1, "The posterior does appear to be normal"
# 使用 pytest.mark.parametrize 装饰器，为 test_iterative_imputer_missing_at_transform 函数提供三个不同的策略参数进行参数化测试
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_iterative_imputer_missing_at_transform(strategy):
    # 创建随机数生成器 rng，种子为 0
    rng = np.random.RandomState(0)
    n = 100  # 数据集大小为 100
    d = 10  # 每个数据点的特征数为 10

    # 生成训练集和测试集，都是随机整数矩阵，值域为 [0, 3)
    X_train = rng.randint(low=0, high=3, size=(n, d))
    X_test = rng.randint(low=0, high=3, size=(n, d))

    X_train[:, 0] = 1  # 将训练集的第一列的所有元素设为 1，确保没有缺失值
    X_test[0, 0] = 0  # 将测试集的第一行第一列元素设为 0，表示这里有缺失值

    # 创建 IterativeImputer 对象 imputer，用于填充缺失值，使用给定的策略和随机数生成器 rng 进行训练
    imputer = IterativeImputer(
        missing_values=0, max_iter=1, initial_strategy=strategy, random_state=rng
    ).fit(X_train)

    # 创建 SimpleImputer 对象 initial_imputer，用于填充缺失值，使用给定的策略进行训练
    initial_imputer = SimpleImputer(missing_values=0, strategy=strategy).fit(X_train)

    # 断言：如果在 fit 时没有缺失值，则在 transform 时 imputer 只会使用初始的填充策略
    assert_allclose(
        imputer.transform(X_test)[:, 0], initial_imputer.transform(X_test)[:, 0]
    )


# 测试 IterativeImputer 对象在 transform 过程中的随机性
def test_iterative_imputer_transform_stochasticity():
    rng1 = np.random.RandomState(0)  # 创建第一个随机数生成器 rng1，种子为 0
    rng2 = np.random.RandomState(1)  # 创建第二个随机数生成器 rng2，种子为 1
    n = 100  # 数据集大小为 100
    d = 10  # 每个数据点的特征数为 10

    # 生成稀疏随机矩阵 X，密度为 0.10，使用 rng1 作为随机数生成器
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng1).toarray()

    # 创建 IterativeImputer 对象 imputer，使用 sample_posterior=True 进行训练
    imputer = IterativeImputer(
        missing_values=0, max_iter=1, sample_posterior=True, random_state=rng1
    )
    imputer.fit(X)

    X_fitted_1 = imputer.transform(X)  # 第一次 transform
    X_fitted_2 = imputer.transform(X)  # 第二次 transform

    # 断言：两次 transform 的均值应该不相等
    assert np.mean(X_fitted_1) != pytest.approx(np.mean(X_fitted_2))

    # 创建两个不同的 IterativeImputer 对象 imputer1 和 imputer2，使用相同的参数进行初始化，但使用不同的随机数生成器
    imputer1 = IterativeImputer(
        missing_values=0,
        max_iter=1,
        sample_posterior=False,
        n_nearest_features=None,
        imputation_order="ascending",
        random_state=rng1,
    )

    imputer2 = IterativeImputer(
        missing_values=0,
        max_iter=1,
        sample_posterior=False,
        n_nearest_features=None,
        imputation_order="ascending",
        random_state=rng2,
    )
    imputer1.fit(X)  # 使用 imputer1 对象进行训练
    imputer2.fit(X)  # 使用 imputer2 对象进行训练

    X_fitted_1a = imputer1.transform(X)  # 使用 imputer1 对象进行第一次 transform
    X_fitted_1b = imputer1.transform(X)  # 使用 imputer1 对象进行第二次 transform
    X_fitted_2 = imputer2.transform(X)  # 使用 imputer2 对象进行 transform

    # 断言：对于 sample_posterior=False、n_nearest_features=None、imputation_order="ascending" 的情况，即使使用不同的随机数生成器，两次 transform 的结果应该相等
    assert_allclose(X_fitted_1a, X_fitted_1b)
    assert_allclose(X_fitted_1a, X_fitted_2)


# 测试在没有缺失值的情况下 IterativeImputer 的行为
def test_iterative_imputer_no_missing():
    rng = np.random.RandomState(0)  # 创建随机数生成器 rng，种子为 0
    X = rng.rand(100, 100)  # 生成大小为 100x100 的随机矩阵 X
    X[:, 0] = np.nan  # 将 X 的第一列的所有元素设为 NaN，表示这里有缺失值

    # 创建两个 IterativeImputer 对象 m1 和 m2，使用相同的参数进行初始化，但使用相同的随机数生成器进行训练
    m1 = IterativeImputer(max_iter=10, random_state=rng)
    m2 = IterativeImputer(max_iter=10, random_state=rng)
    pred1 = m1.fit(X).transform(X)  # 使用 m1 对象进行 fit 和 transform
    pred2 = m2.fit_transform(X)  # 使用 m2 对象进行 fit_transform

    # 断言：应该完全排除第一列
    assert_allclose(X[:, 1:], pred1)
    # 断言：fit 和 fit_transform 的结果应该完全相同
    assert_allclose(pred1, pred2)


# 测试在特征维度为一的情况下 IterativeImputer 的行为
def test_iterative_imputer_rank_one():
    rng = np.random.RandomState(0)  # 创建随机数生成器 rng，种子为 0
    d = 50  # 特征维度为 50
    A = rng.rand(d, 1)  # 生成大小为 d x 1 的随机矩阵 A
    # 使用随机数生成器 rng 生成一个形状为 (1, d) 的随机数组 B
    B = rng.rand(1, d)
    
    # 计算矩阵 A 与向量 B 的乘积，得到结果矩阵 X
    X = np.dot(A, B)
    
    # 使用随机数生成器 rng 生成一个形状为 (d, d) 的随机布尔掩码，掩码中的值小于 0.5
    nan_mask = rng.rand(d, d) < 0.5
    
    # 复制矩阵 X 到 X_missing
    X_missing = X.copy()
    
    # 根据 nan_mask 将 X_missing 中对应位置的值设为 NaN
    X_missing[nan_mask] = np.nan
    
    # 创建一个迭代式填充器 imputer，最大迭代次数为 5，启用详细输出，使用随机状态 rng
    imputer = IterativeImputer(max_iter=5, verbose=1, random_state=rng)
    
    # 使用 imputer 对 X_missing 进行填充，得到填充后的结果 X_filled
    X_filled = imputer.fit_transform(X_missing)
    
    # 断言填充后的结果 X_filled 与原始矩阵 X 在容差为 0.02 的情况下近似相等
    assert_allclose(X_filled, X, atol=0.02)
# 使用 pytest.mark.parametrize 装饰器，为 test_iterative_imputer_transform_recovery 函数添加参数化测试
@pytest.mark.parametrize("rank", [3, 5])
def test_iterative_imputer_transform_recovery(rank):
    # 使用随机数种子 0 初始化随机数生成器
    rng = np.random.RandomState(0)
    # 设置矩阵的行数和列数
    n = 70
    d = 70
    # 生成随机矩阵 A 和 B，分别为 n 行 rank 列和 rank 行 d 列
    A = rng.rand(n, rank)
    B = rng.rand(rank, d)
    # 计算完整的矩阵 X_filled，为 A 和 B 的乘积
    X_filled = np.dot(A, B)
    # 创建一个布尔掩码，标记哪些位置将设置为缺失值
    nan_mask = rng.rand(n, d) < 0.5
    # 根据缺失掩码生成缺失值矩阵 X_missing
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # 将数据分为训练集和测试集，取前一半作为训练集
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    # 使用迭代填充器进行填充
    imputer = IterativeImputer(
        max_iter=5, imputation_order="descending", verbose=1, random_state=rng
    ).fit(X_train)
    # 对测试集进行填充操作
    X_test_est = imputer.transform(X_test)
    # 断言填充后的测试集与预期填充值在给定的误差范围内相似
    assert_allclose(X_test_filled, X_test_est, atol=0.1)


# 定义测试函数 test_iterative_imputer_additive_matrix
def test_iterative_imputer_additive_matrix():
    # 使用随机数种子 0 初始化随机数生成器
    rng = np.random.RandomState(0)
    # 设置矩阵的行数和列数
    n = 100
    d = 10
    # 生成随机矩阵 A 和 B，分别为 n 行 d 列的标准正态分布随机数
    A = rng.randn(n, d)
    B = rng.randn(n, d)
    # 创建一个零矩阵 X_filled，形状与 A 相同
    X_filled = np.zeros(A.shape)
    # 使用两层循环计算 X_filled 的每一列
    for i in range(d):
        for j in range(d):
            X_filled[:, (i + j) % d] += (A[:, i] + B[:, j]) / 2
    # 生成一个随机布尔掩码，标记要设置为缺失值的位置
    nan_mask = rng.rand(n, d) < 0.25
    # 根据缺失掩码生成缺失值矩阵 X_missing
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # 将数据分为训练集和测试集，取前一半作为训练集
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    # 使用迭代填充器进行填充
    imputer = IterativeImputer(max_iter=10, verbose=1, random_state=rng).fit(X_train)
    # 对测试集进行填充操作
    X_test_est = imputer.transform(X_test)
    # 断言填充后的测试集与预期填充值在给定的相对和绝对误差范围内相似
    assert_allclose(X_test_filled, X_test_est, rtol=1e-3, atol=0.01)


# 定义测试函数 test_iterative_imputer_early_stopping
def test_iterative_imputer_early_stopping():
    # 使用随机数种子 0 初始化随机数生成器
    rng = np.random.RandomState(0)
    # 设置矩阵的行数和列数
    n = 50
    d = 5
    # 生成随机矩阵 A 和 B，分别为 n 行 1 列和 1 行 d 列
    A = rng.rand(n, 1)
    B = rng.rand(1, d)
    # 计算完整的矩阵 X，为 A 和 B 的乘积
    X = np.dot(A, B)
    # 生成一个随机布尔掩码，标记要设置为缺失值的位置
    nan_mask = rng.rand(n, d) < 0.5
    # 根据缺失掩码生成缺失值矩阵 X_missing
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan

    # 创建迭代填充器对象，设置最大迭代次数、容忍度、采样后验、详细信息和随机数种子
    imputer = IterativeImputer(
        max_iter=100, tol=1e-2, sample_posterior=False, verbose=1, random_state=rng
    )
    # 使用 X_missing 进行填充，并获得填充后的结果 X_filled_100
    X_filled_100 = imputer.fit_transform(X_missing)
    # 断言填充序列的长度等于迭代次数乘以特征数
    assert len(imputer.imputation_sequence_) == d * imputer.n_iter_

    # 使用迭代次数进行早停，得到填充后的结果 X_filled_early
    imputer = IterativeImputer(
        max_iter=imputer.n_iter_, sample_posterior=False, verbose=1, random_state=rng
    )
    X_filled_early = imputer.fit_transform(X_missing)
    # 断言填充后的结果 X_filled_100 与 X_filled_early 在极小的误差范围内相似
    assert_allclose(X_filled_100, X_filled_early, atol=1e-7)

    # 使用零容忍度重新拟合填充器
    imputer = IterativeImputer(
        max_iter=100, tol=0, sample_posterior=False, verbose=1, random_state=rng
    )
    imputer.fit(X_missing)
    # 断言迭代次数等于最大迭代次数
    assert imputer.n_iter_ == imputer.max_iter


# 定义测试函数 test_iterative_imputer_catch_warning
def test_iterative_imputer_catch_warning():
    # 加载糖尿病数据集 X 和 y
    X, y = load_diabetes(return_X_y=True)
    n_samples, n_features = X.shape

    # 模拟数据集中某个特征全为常数的情况
    X[:, 3] = 1

    # 添加一些缺失值
    rng = np.random.RandomState(0)
    missing_rate = 0.15
    # 对于每个特征进行迭代处理
    for feat in range(n_features):
        # 从样本中随机选择一部分索引，用于设置缺失值
        sample_idx = rng.choice(
            np.arange(n_samples), size=int(n_samples * missing_rate), replace=False
        )
        # 将选定的样本索引处的当前特征值设置为 NaN
        X[sample_idx, feat] = np.nan

    # 创建一个迭代式填充器对象，指定参数为每个样本使用5个最近特征，并使用后验样本
    imputer = IterativeImputer(n_nearest_features=5, sample_posterior=True)
    # 设置运行时警告的处理方式，当出现 RuntimeWarning 时抛出异常
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # 使用迭代式填充器拟合并填充 X 和 y，生成填充后的数据 X_fill
        X_fill = imputer.fit_transform(X, y)
    # 断言：确保填充后的数据中不存在任何 NaN 值
    assert not np.any(np.isnan(X_fill))
@pytest.mark.parametrize(
    "min_value, max_value, correct_output",
    [  # 定义测试参数 min_value, max_value 和预期输出 correct_output
        (0, 100, np.array([[0] * 3, [100] * 3])),  # 测试整数范围情况
        (None, None, np.array([[-np.inf] * 3, [np.inf] * 3])),  # 测试默认情况
        (-np.inf, np.inf, np.array([[-np.inf] * 3, [np.inf] * 3])),  # 测试负无穷到正无穷情况
        ([-5, 5, 10], [100, 200, 300], np.array([[-5, 5, 10], [100, 200, 300]])),  # 测试列表范围情况
        (
            [-5, -np.inf, 10],
            [100, 200, np.inf],
            np.array([[-5, -np.inf, 10], [100, 200, np.inf]]),
        ),  # 测试包含负无穷和正无穷的列表情况
    ],
    ids=["scalars", "None-default", "inf", "lists", "lists-with-inf"],  # 为每种情况定义标识符
)
def test_iterative_imputer_min_max_array_like(min_value, max_value, correct_output):
    # 检查在 IterativeImputer 中传递标量或类数组对象作为 min_value 和 max_value 是否有效
    X = np.random.RandomState(0).randn(10, 3)  # 创建随机数据集 X
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)  # 初始化 IterativeImputer 对象
    imputer.fit(X)  # 对数据集 X 进行拟合

    # 断言 min_value 和 max_value 已被转换为 numpy 数组
    assert isinstance(imputer._min_value, np.ndarray) and isinstance(
        imputer._max_value, np.ndarray
    )
    # 断言 min_value 和 max_value 的数组形状与数据集 X 的列数相同
    assert (imputer._min_value.shape[0] == X.shape[1]) and (
        imputer._max_value.shape[0] == X.shape[1]
    )

    # 使用 assert_allclose 检查计算出的最小值和最大值与预期输出的一致性
    assert_allclose(correct_output[0, :], imputer._min_value)
    assert_allclose(correct_output[1, :], imputer._max_value)


@pytest.mark.parametrize(
    "min_value, max_value, err_msg",
    [  # 定义测试参数 min_value, max_value 和预期错误消息 err_msg
        (100, 0, "min_value >= max_value."),  # 测试 min_value 大于 max_value 的情况
        (np.inf, -np.inf, "min_value >= max_value."),  # 测试无穷大和无穷小的情况
        ([-5, 5], [100, 200, 0], "_value' should be of shape"),  # 测试形状不匹配的情况
    ],
)
def test_iterative_imputer_catch_min_max_error(min_value, max_value, err_msg):
    # 检查在 IterativeImputer 中传递不合法的 min_value 和 max_value 是否能捕获 ValueError 异常
    X = np.random.random((10, 3))  # 创建随机数据集 X
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)  # 初始化 IterativeImputer 对象
    with pytest.raises(ValueError, match=err_msg):  # 捕获预期的 ValueError 异常和错误消息
        imputer.fit(X)


@pytest.mark.parametrize(
    "min_max_1, min_max_2",
    [  # 定义测试参数 min_max_1 和 min_max_2
        ([None, None], [-np.inf, np.inf]),  # 测试 None 和无穷的情况
        ([-10, 10], [[-10] * 4, [10] * 4]),  # 测试标量和向量的情况
    ],
    ids=["None-vs-inf", "Scalar-vs-vector"],  # 为每种情况定义标识符
)
def test_iterative_imputer_min_max_array_like_imputation(min_max_1, min_max_2):
    # 测试 None/inf 和标量/向量是否能给出相同的插补结果
    X_train = np.array(
        [
            [np.nan, 2, 2, 1],
            [10, np.nan, np.nan, 7],
            [3, 1, np.nan, 1],
            [np.nan, 4, 2, np.nan],
        ]
    )  # 创建训练数据集 X_train
    X_test = np.array(
        [[np.nan, 2, np.nan, 5], [2, 4, np.nan, np.nan], [np.nan, 1, 10, 1]]
    )  # 创建测试数据集 X_test
    imputer1 = IterativeImputer(
        min_value=min_max_1[0], max_value=min_max_1[1], random_state=0
    )  # 初始化第一个 IterativeImputer 对象
    imputer2 = IterativeImputer(
        min_value=min_max_2[0], max_value=min_max_2[1], random_state=0
    )  # 初始化第二个 IterativeImputer 对象
    X_test_imputed1 = imputer1.fit(X_train).transform(X_test)  # 对 X_test 进行插补并得到结果
    X_test_imputed2 = imputer2.fit(X_train).transform(X_test)  # 对 X_test 进行插补并得到结果
    # 使用 assert_allclose 检查两个插补结果的第一列是否一致
    assert_allclose(X_test_imputed1[:, 0], X_test_imputed2[:, 0])
# 定义测试函数，用于测试在存在缺失数据的情况下，迭代填充器的跳过非缺失值策略
def test_iterative_imputer_skip_non_missing(skip_complete):
    # 检查在测试集中仅存在缺失数据时的填充策略
    # 参考自：https://github.com/scikit-learn/scikit-learn/issues/14383
    rng = np.random.RandomState(0)  # 创建随机数生成器实例，种子为0
    X_train = np.array([[5, 2, 2, 1], [10, 1, 2, 7], [3, 1, 1, 1], [8, 4, 2, 2]])  # 训练集
    X_test = np.array([[np.nan, 2, 4, 5], [np.nan, 4, 1, 2], [np.nan, 1, 10, 1]])  # 测试集
    imputer = IterativeImputer(
        initial_strategy="mean", skip_complete=skip_complete, random_state=rng
    )  # 创建迭代填充器实例，使用均值作为初始策略，并根据skip_complete参数决定是否跳过完整样本，设置随机种子为rng
    X_test_est = imputer.fit(X_train).transform(X_test)  # 对测试集进行拟合和转换
    if skip_complete:
        # 使用初始策略 'mean' 进行填充
        assert_allclose(X_test_est[:, 0], np.mean(X_train[:, 0]))  # 检查填充后第一列的值是否接近训练集第一列的均值
    else:
        assert_allclose(X_test_est[:, 0], [11, 7, 12], rtol=1e-4)  # 检查填充后第一列的值是否接近预期值，设置相对误差容忍度为1e-4


# 使用pytest的参数化装饰器，测试迭代填充器不设置随机状态的情况
@pytest.mark.parametrize("rs_imputer", [None, 1, np.random.RandomState(seed=1)])
@pytest.mark.parametrize("rs_estimator", [None, 1, np.random.RandomState(seed=1)])
def test_iterative_imputer_dont_set_random_state(rs_imputer, rs_estimator):
    # 定义一个零预测器类，用于测试随机状态设置
    class ZeroEstimator:
        def __init__(self, random_state):
            self.random_state = random_state  # 初始化随机状态

        def fit(self, *args, **kgards):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])  # 返回与样本数量相同的零数组作为预测结果

    estimator = ZeroEstimator(random_state=rs_estimator)  # 创建零预测器实例，使用给定的随机状态
    imputer = IterativeImputer(random_state=rs_imputer)  # 创建迭代填充器实例，使用给定的随机状态
    X_train = np.zeros((10, 3))  # 创建一个全零的训练集
    imputer.fit(X_train)  # 对训练集进行拟合
    assert estimator.random_state == rs_estimator  # 检查预测器的随机状态是否与输入的随机状态一致


# 使用pytest的参数化装饰器，测试MissingIndicator的错误情况
@pytest.mark.parametrize(
    "X_fit, X_trans, params, msg_err",
    [
        (
            np.array([[-1, 1], [1, 2]]),
            np.array([[-1, 1], [1, -1]]),
            {"features": "missing-only", "sparse": "auto"},
            "have missing values in transform but have no missing values in fit",
        ),
        (
            np.array([["a", "b"], ["c", "a"]], dtype=str),
            np.array([["a", "b"], ["c", "a"]], dtype=str),
            {},
            "MissingIndicator does not support data with dtype",
        ),
    ],
)
def test_missing_indicator_error(X_fit, X_trans, params, msg_err):
    indicator = MissingIndicator(missing_values=-1)  # 创建缺失指示器实例，指定缺失值为-1
    indicator.set_params(**params)  # 设置指示器的参数
    with pytest.raises(ValueError, match=msg_err):  # 捕获预期的ValueError异常，并检查是否匹配给定的错误消息
        indicator.fit(X_fit).transform(X_trans)  # 对给定的数据进行拟合和转换


# 定义生成MissingIndicator测试用例的辅助函数
def _generate_missing_indicator_cases():
    missing_values_dtypes = [(0, np.int32), (np.nan, np.float64), (-1, np.int32)]  # 不同的缺失值及其数据类型组合
    arr_types = (
        [np.array]  # 包含各种稀疏矩阵格式的数组类型
        + CSC_CONTAINERS
        + CSR_CONTAINERS
        + COO_CONTAINERS
        + LIL_CONTAINERS
        + BSR_CONTAINERS
    )
    return [
        (arr_type, missing_values, dtype)  # 返回数组类型、缺失值及其数据类型的元组列表
        for arr_type, (missing_values, dtype) in product(
            arr_types, missing_values_dtypes
        )
        if not (missing_values == 0 and arr_type is not np.array)  # 排除缺失值为0且数组类型不是np.array的情况
    ]


# 使用pytest的参数化装饰器，测试生成的MissingIndicator测试用例
@pytest.mark.parametrize(
    "arr_type, missing_values, dtype", _generate_missing_indicator_cases()
)
@pytest.mark.parametrize(
    # 定义一个包含元组的列表，每个元组包含三个元素：字符串、整数和NumPy数组
    "param_features, n_features, features_indices",
    [("missing-only", 3, np.array([0, 1, 2])), ("all", 3, np.array([0, 1, 2]))],
def test_missing_indicator_new(
    missing_values, arr_type, dtype, param_features, n_features, features_indices
):
    # 创建一个包含缺失值的二维数组作为模拟数据
    X_fit = np.array([[missing_values, missing_values, 1], [4, 2, missing_values]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    # 期望的结果，即处理后的数组
    X_fit_expected = np.array([[1, 1, 0], [0, 0, 1]])
    X_trans_expected = np.array([[1, 1, 0], [0, 0, 0]])

    # 将输入转换为正确的数组格式并设置正确的数据类型
    X_fit = arr_type(X_fit).astype(dtype)
    X_trans = arr_type(X_trans).astype(dtype)
    X_fit_expected = X_fit_expected.astype(dtype)
    X_trans_expected = X_trans_expected.astype(dtype)

    # 创建一个MissingIndicator对象，设置缺失值、特征和稀疏性参数
    indicator = MissingIndicator(
        missing_values=missing_values, features=param_features, sparse=False
    )
    # 对训练集数据进行拟合并转换
    X_fit_mask = indicator.fit_transform(X_fit)
    # 对测试集数据进行转换
    X_trans_mask = indicator.transform(X_trans)

    # 断言拟合后的数据列数和预期的特征数相等
    assert X_fit_mask.shape[1] == n_features
    assert X_trans_mask.shape[1] == n_features

    # 断言特征索引与预期的特征索引一致
    assert_array_equal(indicator.features_, features_indices)
    # 断言拟合后的结果与预期的结果非常接近
    assert_allclose(X_fit_mask, X_fit_expected[:, features_indices])
    assert_allclose(X_trans_mask, X_trans_expected[:, features_indices])

    # 断言拟合后的结果数据类型为布尔型
    assert X_fit_mask.dtype == bool
    assert X_trans_mask.dtype == bool
    # 断言拟合后的结果类型为numpy数组
    assert isinstance(X_fit_mask, np.ndarray)
    assert isinstance(X_trans_mask, np.ndarray)

    # 将稀疏参数设置为True并重新拟合和转换数据
    indicator.set_params(sparse=True)
    X_fit_mask_sparse = indicator.fit_transform(X_fit)
    X_trans_mask_sparse = indicator.transform(X_trans)

    # 断言稀疏拟合后的结果数据类型为布尔型
    assert X_fit_mask_sparse.dtype == bool
    assert X_trans_mask_sparse.dtype == bool
    # 断言稀疏拟合后的结果格式为"csc"格式
    assert X_fit_mask_sparse.format == "csc"
    assert X_trans_mask_sparse.format == "csc"
    # 断言稀疏拟合后的结果与非稀疏结果非常接近
    assert_allclose(X_fit_mask_sparse.toarray(), X_fit_mask)
    assert_allclose(X_trans_mask_sparse.toarray(), X_trans_mask)
    # 将多个容器类型列表合并，并与 [np.nan] 元素进行笛卡尔积运算，生成一个包含所有可能组合的列表
    list(
        product(
            CSC_CONTAINERS  # 第一个容器类型列表
            + CSR_CONTAINERS  # 第二个容器类型列表
            + COO_CONTAINERS  # 第三个容器类型列表
            + LIL_CONTAINERS  # 第四个容器类型列表
            + BSR_CONTAINERS,  # 第五个容器类型列表
            [np.nan],  # 包含单个元素 np.nan 的列表，用于与各个容器类型列表进行组合
        )
    ),
# 定义测试函数，用于测试 MissingIndicator 类在不同参数设置下的输出格式
def test_missing_indicator_sparse_param(arr_type, missing_values, param_sparse):
    # 创建两个测试数据集，其中包含缺失值，将其转换为特定的数组类型并转换为浮点类型
    X_fit = np.array([[missing_values, missing_values, 1], [4, missing_values, 2]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit = arr_type(X_fit).astype(np.float64)
    X_trans = arr_type(X_trans).astype(np.float64)

    # 创建 MissingIndicator 对象，设置缺失值和稀疏参数
    indicator = MissingIndicator(missing_values=missing_values, sparse=param_sparse)
    
    # 对训练数据集应用 MissingIndicator，并获取结果
    X_fit_mask = indicator.fit_transform(X_fit)
    
    # 对测试数据集应用已经拟合的 MissingIndicator，并获取结果
    X_trans_mask = indicator.transform(X_trans)

    # 根据不同参数设置进行断言验证
    if param_sparse is True:
        assert X_fit_mask.format == "csc"
        assert X_trans_mask.format == "csc"
    elif param_sparse == "auto" and missing_values == 0:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    elif param_sparse is False:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    else:
        # 如果输入数据 X_fit 是稀疏矩阵，则验证结果的格式为 "csc"
        if sparse.issparse(X_fit):
            assert X_fit_mask.format == "csc"
            assert X_trans_mask.format == "csc"
        else:
            # 否则验证结果为 numpy 数组
            assert isinstance(X_fit_mask, np.ndarray)
            assert isinstance(X_trans_mask, np.ndarray)


# 定义测试函数，测试字符串类型数据的 MissingIndicator
def test_missing_indicator_string():
    # 创建包含字符串数据的数组 X
    X = np.array([["a", "b", "c"], ["b", "c", "a"]], dtype=object)
    
    # 创建 MissingIndicator 对象，指定缺失值为 "a"，特征为 "all"
    indicator = MissingIndicator(missing_values="a", features="all")
    
    # 对数据 X 应用 MissingIndicator，并获取转换后的结果
    X_trans = indicator.fit_transform(X)
    
    # 验证转换后的结果是否符合预期
    assert_array_equal(X_trans, np.array([[True, False, False], [False, False, True]]))


# 使用 pytest 的参数化装饰器，定义测试函数，测试不同输入情况下的 MissingIndicator 和 Imputer 组合效果
@pytest.mark.parametrize(
    "X, missing_values, X_trans_exp",
    [
        (
            np.array([["a", "b"], ["b", "a"]], dtype=object),
            "a",
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
        (
            np.array([[np.nan, 1.0], [1.0, np.nan]]),
            np.nan,
            np.array([[1.0, 1.0, True, False], [1.0, 1.0, False, True]]),
        ),
        (
            np.array([[np.nan, "b"], ["b", np.nan]], dtype=object),
            np.nan,
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
        (
            np.array([[None, "b"], ["b", None]], dtype=object),
            None,
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
    ],
)
def test_missing_indicator_with_imputer(X, missing_values, X_trans_exp):
    # 创建 Imputer 和 MissingIndicator 的组合转换器
    trans = make_union(
        SimpleImputer(missing_values=missing_values, strategy="most_frequent"),
        MissingIndicator(missing_values=missing_values),
    )
    
    # 对输入数据 X 应用组合转换器，并获取转换后的结果
    X_trans = trans.fit_transform(X)
    
    # 验证转换后的结果是否符合预期
    assert_array_equal(X_trans, X_trans_exp)


# 使用 pytest 的参数化装饰器，定义测试函数，测试不同 Imputer 构造器和缺失值参数组合下的错误处理
@pytest.mark.parametrize("imputer_constructor", [SimpleImputer, IterativeImputer])
@pytest.mark.parametrize(
    "imputer_missing_values, missing_value, err_msg",
    [
        ("NaN", np.nan, "Input X contains NaN"),
        ("-1", -1, "types are expected to be both numerical."),
    ],



    # 这里是一个列表的结尾，用于闭合之前定义的列表对象或者列表推导式等。
# 定义测试函数，用于测试不同情况下缺失数据处理的行为
def test_inconsistent_dtype_X_missing_values(
    imputer_constructor, imputer_missing_values, missing_value, err_msg
):
    # 回归测试，检查是否能正确处理 X 的数据类型与 missing_values 的不一致情况
    # 设置随机数种子为42
    rng = np.random.RandomState(42)
    # 创建一个 10x10 的随机数矩阵 X
    X = rng.randn(10, 10)
    # 将矩阵中第一个元素设为 missing_value
    X[0, 0] = missing_value

    # 使用给定的 imputer 构造函数创建一个 imputer 对象
    imputer = imputer_constructor(missing_values=imputer_missing_values)

    # 使用 pytest 的 assertRaises 函数验证是否抛出 ValueError，并检查错误消息是否匹配
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(X)


def test_missing_indicator_no_missing():
    # 检查当没有缺失值时，设置 features='missing-only' 时是否所有特征都被丢弃 (#13491)
    X = np.array([[1, 1], [1, 1]])

    # 创建 MissingIndicator 对象 mi，指定特征为 'missing-only'，缺失值为 -1
    mi = MissingIndicator(features="missing-only", missing_values=-1)
    # 对 X 进行转换
    Xt = mi.fit_transform(X)

    # 验证转换后的矩阵 Xt 的列数是否为 0
    assert Xt.shape[1] == 0


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_missing_indicator_sparse_no_explicit_zeros(csr_container):
    # 检查稀疏矩阵 X 时，确保缺失值不会在生成的掩码中成为显式的零 (#13491)
    X = csr_container([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    # 创建 MissingIndicator 对象 mi，指定特征为 'all'，缺失值为 1
    mi = MissingIndicator(features="all", missing_values=1)
    # 对 X 进行转换
    Xt = mi.fit_transform(X)

    # 验证转换后的稀疏矩阵 Xt 的非零元素数量是否等于其总和
    assert Xt.getnnz() == Xt.sum()


@pytest.mark.parametrize("imputer_constructor", [SimpleImputer, IterativeImputer])
def test_imputer_without_indicator(imputer_constructor):
    # 测试没有指示器时的缺失数据填充行为
    X = np.array([[1, 1], [1, 1]])
    # 使用给定的 imputer 构造函数创建一个 imputer 对象
    imputer = imputer_constructor()
    imputer.fit(X)

    # 验证 imputer 对象的 indicator_ 属性是否为 None
    assert imputer.indicator_ is None


@pytest.mark.parametrize(
    "arr_type",
    CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + BSR_CONTAINERS,
)
def test_simple_imputation_add_indicator_sparse_matrix(arr_type):
    # 测试在稀疏矩阵中进行简单的缺失数据填充，并添加指示器列的行为
    X_sparse = arr_type([[np.nan, 1, 5], [2, np.nan, 1], [6, 3, np.nan], [1, 2, 9]])
    X_true = np.array(
        [
            [3.0, 1.0, 5.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 1.0, 0.0],
            [6.0, 3.0, 5.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 9.0, 0.0, 0.0, 0.0],
        ]
    )

    # 使用 SimpleImputer 对象 imputer 进行缺失数据填充，设置 missing_values 为 np.nan，add_indicator 为 True
    imputer = SimpleImputer(missing_values=np.nan, add_indicator=True)
    X_trans = imputer.fit_transform(X_sparse)

    # 验证 X_trans 是否为稀疏矩阵，其形状是否与 X_true 一致，并且转换后的结果是否与 X_true 近似相等
    assert sparse.issparse(X_trans)
    assert X_trans.shape == X_true.shape
    assert_allclose(X_trans.toarray(), X_true)


@pytest.mark.parametrize(
    "strategy, expected", [("most_frequent", "b"), ("constant", "missing_value")]
)
def test_simple_imputation_string_list(strategy, expected):
    # 测试对字符串类型的列表进行简单的缺失数据填充
    X = [["a", "b"], ["c", np.nan]]

    X_true = np.array([["a", "b"], ["c", expected]], dtype=object)

    # 使用 SimpleImputer 对象 imputer 进行缺失数据填充，设置 strategy 为给定的策略
    imputer = SimpleImputer(strategy=strategy)
    X_trans = imputer.fit_transform(X)

    # 验证填充后的结果 X_trans 是否与预期的 X_true 一致
    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize(
    "order, idx_order",
    [("ascending", [3, 4, 2, 0, 1]), ("descending", [1, 0, 2, 4, 3])],
)
def test_imputation_order(order, idx_order):
    # 回归测试，检查填充顺序是否按照预期顺序进行 (#15393)
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    # 将数据集 X 的前50行第1列设置为 NaN（缺失值）
    X[:50, 1] = np.nan
    # 将数据集 X 的前30行第0列设置为 NaN
    X[:30, 0] = np.nan
    # 将数据集 X 的前20行第2列设置为 NaN
    X[:20, 2] = np.nan
    # 将数据集 X 的前10行第4列设置为 NaN
    X[:10, 4] = np.nan
    
    # 使用 pytest 模块的 warns 方法捕获 ConvergenceWarning 警告
    with pytest.warns(ConvergenceWarning):
        # 使用 IterativeImputer 进行数据填充，每次最多迭代1次，指定填充顺序和随机状态
        trs = IterativeImputer(max_iter=1, imputation_order=order, random_state=0).fit(
            X
        )
        # 获取填充顺序的特征索引列表
        idx = [x.feat_idx for x in trs.imputation_sequence_]
        # 断言填充顺序的特征索引列表与预期的 idx_order 相等
        assert idx == idx_order
# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，用于测试简单填充器的逆转换功能
@pytest.mark.parametrize("missing_value", [-1, np.nan])
def test_simple_imputation_inverse_transform(missing_value):
    # 创建第一个测试数据集 X_1，包含缺失值为 missing_value 的数组
    X_1 = np.array(
        [
            [9, missing_value, 3, -1],
            [4, -1, 5, 4],
            [6, 7, missing_value, -1],
            [8, 9, 0, missing_value],
        ]
    )

    # 创建第二个测试数据集 X_2，包含缺失值为 missing_value 的数组
    X_2 = np.array(
        [
            [5, 4, 2, 1],
            [2, 1, missing_value, 3],
            [9, missing_value, 7, 1],
            [6, 4, 2, missing_value],
        ]
    )

    # 创建第三个测试数据集 X_3，包含缺失值为 missing_value 的数组
    X_3 = np.array(
        [
            [1, missing_value, 5, 9],
            [missing_value, 4, missing_value, missing_value],
            [2, missing_value, 7, missing_value],
            [missing_value, 3, missing_value, 8],
        ]
    )

    # 创建第四个测试数据集 X_4，包含缺失值为 missing_value 的数组
    X_4 = np.array(
        [
            [1, 1, 1, 3],
            [missing_value, 2, missing_value, 1],
            [2, 3, 3, 4],
            [missing_value, 4, missing_value, 2],
        ]
    )

    # 创建一个简单填充器对象，指定缺失值和填充策略为均值，并开启指示器功能
    imputer = SimpleImputer(
        missing_values=missing_value, strategy="mean", add_indicator=True
    )

    # 对 X_1 进行拟合和逆转换操作
    X_1_trans = imputer.fit_transform(X_1)
    X_1_inv_trans = imputer.inverse_transform(X_1_trans)

    # 对 X_2 进行转换和逆转换操作（测试新数据）
    X_2_trans = imputer.transform(X_2)
    X_2_inv_trans = imputer.inverse_transform(X_2_trans)

    # 断言逆转换后的数据与原始数据 X_1 和 X_2 相等
    assert_array_equal(X_1_inv_trans, X_1)
    assert_array_equal(X_2_inv_trans, X_2)

    # 对 X_3 和 X_4 进行循环测试：拟合、逆转换并断言逆转换后的数据与原始数据相等
    for X in [X_3, X_4]:
        X_trans = imputer.fit_transform(X)
        X_inv_trans = imputer.inverse_transform(X_trans)
        assert_array_equal(X_inv_trans, X)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，用于测试简单填充器的逆转换功能中的异常情况
@pytest.mark.parametrize("missing_value", [-1, np.nan])
def test_simple_imputation_inverse_transform_exceptions(missing_value):
    # 创建包含缺失值为 missing_value 的测试数据集 X_1
    X_1 = np.array(
        [
            [9, missing_value, 3, -1],
            [4, -1, 5, 4],
            [6, 7, missing_value, -1],
            [8, 9, 0, missing_value],
        ]
    )

    # 创建一个简单填充器对象，指定缺失值和填充策略为均值
    imputer = SimpleImputer(missing_values=missing_value, strategy="mean")

    # 对 X_1 进行拟合和转换操作
    X_1_trans = imputer.fit_transform(X_1)

    # 使用 pytest.raises 断言捕获 ValueError 异常，并匹配特定的异常消息
    with pytest.raises(
        ValueError, match=f"Got 'add_indicator={imputer.add_indicator}'"
    ):
        imputer.inverse_transform(X_1_trans)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，测试在不同情况下最频繁值填充器的行为
@pytest.mark.parametrize(
    "expected,array,dtype,extra_value,n_repeat",
    [
        # object 类型数组
        ("extra_value", ["a", "b", "c"], object, "extra_value", 2),
        (
            "most_frequent_value",
            ["most_frequent_value", "most_frequent_value", "value"],
            object,
            "extra_value",
            1,
        ),
        ("a", ["min_value", "min_valuevalue"], object, "a", 2),
        ("min_value", ["min_value", "min_value", "value"], object, "z", 2),
        # numeric 类型数组
        (10, [1, 2, 3], int, 10, 2),
        (1, [1, 1, 2], int, 10, 1),
        (10, [20, 20, 1], int, 10, 2),
        (1, [1, 1, 20], int, 10, 2),
    ],
)
def test_most_frequent(expected, array, dtype, extra_value, n_repeat):
    # 使用断言来验证预期值 `expected` 是否等于调用 `_most_frequent` 函数后返回的结果
    assert expected == _most_frequent(
        np.array(array, dtype=dtype), extra_value, n_repeat
    )
# 使用 pytest 的参数化装饰器，定义多组测试参数，用于测试不同的初始策略
@pytest.mark.parametrize(
    "initial_strategy", ["mean", "median", "most_frequent", "constant"]
)
# 定义测试函数，测试迭代填充器在保留空特征的情况下的行为
def test_iterative_imputer_keep_empty_features(initial_strategy):
    """Check the behaviour of the iterative imputer with different initial strategy
    and keeping empty features (i.e. features containing only missing values).
    """
    # 创建一个包含 NaN 值的 NumPy 数组作为测试数据
    X = np.array([[1, np.nan, 2], [3, np.nan, np.nan]])

    # 初始化迭代填充器对象，根据参数设置初始策略和保留空特征的选项
    imputer = IterativeImputer(
        initial_strategy=initial_strategy, keep_empty_features=True
    )
    # 使用 fit_transform 方法拟合填充器并转换数据
    X_imputed = imputer.fit_transform(X)
    # 断言检查：验证填充后的数据中第二列的值是否接近于0
    assert_allclose(X_imputed[:, 1], 0)
    # 使用 transform 方法再次转换数据
    X_imputed = imputer.transform(X)
    # 断言检查：验证填充后的数据中第二列的值是否接近于0
    assert_allclose(X_imputed[:, 1], 0)


# 定义测试函数，检查迭代填充器在常数填充值情况下的行为
def test_iterative_imputer_constant_fill_value():
    """Check that we propagate properly the parameter `fill_value`."""
    # 创建一个包含缺失值的 NumPy 数组作为测试数据
    X = np.array([[-1, 2, 3, -1], [4, -1, 5, -1], [6, 7, -1, -1], [8, 9, 0, -1]])

    # 设置常数填充值
    fill_value = 100
    # 初始化迭代填充器对象，根据参数设置缺失值、初始策略、填充值和最大迭代次数
    imputer = IterativeImputer(
        missing_values=-1,
        initial_strategy="constant",
        fill_value=fill_value,
        max_iter=0,
    )
    # 使用 fit_transform 方法拟合填充器并转换数据
    imputer.fit_transform(X)
    # 断言检查：验证初始填充器的统计信息是否与填充值一致
    assert_array_equal(imputer.initial_imputer_.statistics_, fill_value)


# 使用 pytest 的参数化装饰器，定义多组测试参数，用于测试不同的保留空特征选项
@pytest.mark.parametrize("keep_empty_features", [True, False])
# 定义测试函数，测试 KNN 填充器在保留空特征时的行为
def test_knn_imputer_keep_empty_features(keep_empty_features):
    """Check the behaviour of `keep_empty_features` for `KNNImputer`."""
    # 创建一个包含 NaN 值的 NumPy 数组作为测试数据
    X = np.array([[1, np.nan, 2], [3, np.nan, np.nan]])

    # 初始化 KNN 填充器对象，根据参数设置是否保留空特征
    imputer = KNNImputer(keep_empty_features=keep_empty_features)

    # 遍历两种方法：fit_transform 和 transform
    for method in ["fit_transform", "transform"]:
        # 调用填充器对象的方法，对数据进行填充
        X_imputed = getattr(imputer, method)(X)
        # 根据是否保留空特征进行断言检查
        if keep_empty_features:
            # 断言检查：验证填充后的数据形状是否与原始数据相同，并且第二列的值是否为0
            assert X_imputed.shape == X.shape
            assert_array_equal(X_imputed[:, 1], 0)
        else:
            # 断言检查：验证填充后的数据形状是否减少了一个特征列
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)


# 定义测试函数，检查简单填充器在处理 Pandas 数据中 NA 值的行为
def test_simple_impute_pd_na():
    pd = pytest.importorskip("pandas")

    # 创建包含字符串类型的 Pandas DataFrame，其中包含 NA 值
    df = pd.DataFrame({"feature": pd.Series(["abc", None, "de"], dtype="string")})
    # 初始化简单填充器对象，根据参数设置 NA 值、填充策略和填充值
    imputer = SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value="na")
    # 调用 fit_transform 方法拟合填充器并转换数据，进行断言检查
    _assert_array_equal_and_same_dtype(
        imputer.fit_transform(df), np.array([["abc"], ["na"], ["de"]], dtype=object)
    )

    # 创建包含整数类型的 Pandas DataFrame，其中包含 NA 值
    df = pd.DataFrame({"feature": pd.Series([1, None, 3], dtype="Int64")})
    # 初始化简单填充器对象，根据参数设置 NA 值、填充策略和填充值
    imputer = SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value=-1)
    # 调用 fit_transform 方法拟合填充器并转换数据，进行断言检查
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [-1], [3]], dtype="float64")
    )

    # 使用 np.nan 作为 NA 值的情况，也可以正常工作
    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-1)
    # 调用 _assert_allclose_and_same_dtype 函数，验证 imputer.fit_transform(df) 的输出与指定的 numpy 数组是否几乎相等，并且数据类型一致。
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [-1], [3]], dtype="float64")
    )
    
    # 使用 'median' 策略对包含整数类型的 pandas 数组进行填充缺失值处理。
    df = pd.DataFrame({"feature": pd.Series([1, None, 2, 3], dtype="Int64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="median")
    # 验证 imputer.fit_transform(df) 的输出与指定的 numpy 数组是否几乎相等，并且数据类型一致。
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [2], [2], [3]], dtype="float64")
    )
    
    # 使用 'mean' 策略对包含整数类型的 pandas 数组进行填充缺失值处理。
    df = pd.DataFrame({"feature": pd.Series([1, None, 2], dtype="Int64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="mean")
    # 验证 imputer.fit_transform(df) 的输出与指定的 numpy 数组是否几乎相等，并且数据类型一致。
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [1.5], [2]], dtype="float64")
    )
    
    # 使用 'constant' 策略和填充值 -2.0 对包含浮点类型的 pandas 数组进行填充缺失值处理。
    df = pd.DataFrame({"feature": pd.Series([1.0, None, 3.0], dtype="float64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value=-2.0)
    # 验证 imputer.fit_transform(df) 的输出与指定的 numpy 数组是否几乎相等，并且数据类型一致。
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1.0], [-2.0], [3.0]], dtype="float64")
    )
    
    # 使用 'median' 策略对包含浮点类型的 pandas 数组进行填充缺失值处理。
    df = pd.DataFrame({"feature": pd.Series([1.0, None, 2.0, 3.0], dtype="float64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="median")
    # 验证 imputer.fit_transform(df) 的输出与指定的 numpy 数组是否几乎相等，并且数据类型一致。
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df),
        np.array([[1.0], [2.0], [2.0], [3.0]], dtype="float64"),
    )
# 定义一个测试函数，用于测试 MissingIndicator 类的特性
def test_missing_indicator_feature_names_out():
    """Check that missing indicator return the feature names with a prefix."""
    # 导入 pytest 库，并检查是否存在，如果不存在则跳过当前测试
    pd = pytest.importorskip("pandas")

    # 定义缺失值为 np.nan，并创建一个包含缺失值的 DataFrame X
    missing_values = np.nan
    X = pd.DataFrame(
        [
            [missing_values, missing_values, 1, missing_values],
            [4, missing_values, 2, 10],
        ],
        columns=["a", "b", "c", "d"],
    )

    # 使用 MissingIndicator 对象拟合 DataFrame X
    indicator = MissingIndicator(missing_values=missing_values).fit(X)
    # 获取生成的特征名称列表
    feature_names = indicator.get_feature_names_out()
    # 预期的特征名称列表
    expected_names = ["missingindicator_a", "missingindicator_b", "missingindicator_d"]
    # 断言生成的特征名称列表与预期的特征名称列表相等
    assert_array_equal(expected_names, feature_names)


# 定义一个测试函数，用于测试 SimpleImputer 类的 fit_transform 方法
def test_imputer_lists_fit_transform():
    """Check transform uses object dtype when fitted on an object dtype.

    Non-regression test for #19572.
    """
    # 创建一个包含字符串的列表 X
    X = [["a", "b"], ["c", "b"], ["a", "a"]]
    # 使用 SimpleImputer 对象以最频繁值策略拟合列表 X
    imp_frequent = SimpleImputer(strategy="most_frequent").fit(X)
    # 对包含缺失值的新列表进行转换
    X_trans = imp_frequent.transform([[np.nan, np.nan]])
    # 断言转换后的数组类型为 object
    assert X_trans.dtype == object
    # 断言转换后的数组内容与预期相等
    assert_array_equal(X_trans, [["a", "b"]])


# 使用 pytest.mark.parametrize 标记的参数化测试函数，用于测试 SimpleImputer 类的 transform 方法
@pytest.mark.parametrize("dtype_test", [np.float32, np.float64])
def test_imputer_transform_preserves_numeric_dtype(dtype_test):
    """Check transform preserves numeric dtype independent of fit dtype."""
    # 创建一个包含浮点数和缺失值的数组 X
    X = np.asarray(
        [[1.2, 3.4, np.nan], [np.nan, 1.2, 1.3], [4.2, 2, 1]], dtype=np.float64
    )
    # 使用 SimpleImputer 对象拟合数组 X
    imp = SimpleImputer().fit(X)

    # 创建一个包含缺失值的新数组 X_test，类型由参数 dtype_test 指定
    X_test = np.asarray([[np.nan, np.nan, np.nan]], dtype=dtype_test)
    # 对新数组进行转换
    X_trans = imp.transform(X_test)
    # 断言转换后的数组类型与参数 dtype_test 相等
    assert X_trans.dtype == dtype_test


# 使用 pytest.mark.parametrize 标记的参数化测试函数，用于测试 SimpleImputer 类的 constant 策略
@pytest.mark.parametrize("array_type", ["array", "sparse"])
@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_simple_imputer_constant_keep_empty_features(array_type, keep_empty_features):
    """Check the behaviour of `keep_empty_features` with `strategy='constant'.
    For backward compatibility, a column full of missing values will always be
    fill and never dropped.
    """
    # 创建一个包含缺失值的二维数组 X
    X = np.array([[np.nan, 2], [np.nan, 3], [np.nan, 6]])
    # 根据参数 array_type 将数组 X 转换成 array 或 sparse 格式
    X = _convert_container(X, array_type)
    # 设置常量填充值
    fill_value = 10
    # 使用 SimpleImputer 对象以常量填充策略拟合数组 X
    imputer = SimpleImputer(
        strategy="constant",
        fill_value=fill_value,
        keep_empty_features=keep_empty_features,
    )

    # 遍历方法列表 ["fit_transform", "transform"]
    for method in ["fit_transform", "transform"]:
        # 调用 SimpleImputer 对象的方法 method，对数组 X 进行填充转换
        X_imputed = getattr(imputer, method)(X)
        # 断言转换后的数组形状与原始数组 X 相同
        assert X_imputed.shape == X.shape
        # 提取第一列特征，并根据 array_type 判断是否转换成稀疏格式
        constant_feature = (
            X_imputed[:, 0].toarray() if array_type == "sparse" else X_imputed[:, 0]
        )
        # 断言转换后的常量特征与填充值相等
        assert_array_equal(constant_feature, fill_value)


# 使用 pytest.mark.parametrize 标记的参数化测试函数，用于测试 SimpleImputer 类的各种策略
@pytest.mark.parametrize("array_type", ["array", "sparse"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_simple_imputer_keep_empty_features(strategy, array_type, keep_empty_features):
    """Check the behaviour of `keep_empty_features` with all strategies but
    'constant'.
    """
    # 创建一个包含缺失值的二维数组 X
    X = np.array([[np.nan, 2], [np.nan, 3], [np.nan, 6]])
    # 对输入的数据 X 进行类型转换，确保其为指定的 array_type 类型
    X = _convert_container(X, array_type)
    
    # 创建一个 SimpleImputer 对象，指定策略为 strategy，并根据 keep_empty_features 设置是否保留空特征
    imputer = SimpleImputer(strategy=strategy, keep_empty_features=keep_empty_features)

    # 遍历两种方法："fit_transform" 和 "transform"
    for method in ["fit_transform", "transform"]:
        # 使用 getattr 函数动态调用 SimpleImputer 对象的方法（fit_transform 或 transform），对数据 X 进行填充操作
        X_imputed = getattr(imputer, method)(X)
        
        # 如果 keep_empty_features 为 True，则进行以下断言和验证
        if keep_empty_features:
            # 确保填充后的数据 X_imputed 的形状与原始数据 X 的形状相同
            assert X_imputed.shape == X.shape
            
            # 如果 array_type 为 "sparse"，则将第一列转换为稀疏数组并进行断言
            constant_feature = (
                X_imputed[:, 0].toarray() if array_type == "sparse" else X_imputed[:, 0]
            )
            assert_array_equal(constant_feature, 0)  # 确保第一列特征被填充为常数 0
        else:
            # 如果 keep_empty_features 为 False，则断言填充后的数据 X_imputed 的形状为原始数据的行数和减少了一列的列数
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)
# 使用 pytest 的 pytest.mark.parametrize 装饰器来参数化测试函数，参数为 CSC_CONTAINERS 中的每个值
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_custom(csc_container):
    # 创建一个输入矩阵 X，包含浮点数和缺失值（np.nan）
    X = np.array(
        [
            [1.1, 1.1, 1.1],
            [3.9, 1.2, np.nan],
            [np.nan, 1.3, np.nan],
            [0.1, 1.4, 1.4],
            [4.9, 1.5, 1.5],
            [np.nan, 1.6, 1.6],
        ]
    )

    # 创建一个期望的输出矩阵 X_true，对应于通过简单填充器（SimpleImputer）处理后的结果
    X_true = np.array(
        [
            [1.1, 1.1, 1.1],
            [3.9, 1.2, 1.1],
            [0.1, 1.3, 1.1],
            [0.1, 1.4, 1.4],
            [4.9, 1.5, 1.5],
            [0.1, 1.6, 1.6],
        ]
    )

    # 创建一个简单填充器对象，使用 np.nan 作为缺失值，使用 np.min 策略填充缺失值
    imputer = SimpleImputer(missing_values=np.nan, strategy=np.min)
    # 对输入矩阵 X 进行拟合和转换，得到填充后的矩阵 X_trans
    X_trans = imputer.fit_transform(X)
    # 断言填充后的结果 X_trans 与期望的结果 X_true 相等
    assert_array_equal(X_trans, X_true)

    # 测试稀疏矩阵的情况
    imputer = SimpleImputer(missing_values=np.nan, strategy=np.min)
    # 对使用 csc_container 封装的输入矩阵 X 进行拟合和转换，得到填充后的稀疏矩阵 X_trans
    X_trans = imputer.fit_transform(csc_container(X))
    # 断言填充后的稀疏矩阵 X_trans 转为稠密数组后与期望的结果 X_true 相等
    assert_array_equal(X_trans.toarray(), X_true)


def test_simple_imputer_constant_fill_value_casting():
    """检查当无法将填充值转换为输入数据类型时是否能够正确引发错误消息，以及是否能够正确进行类型转换。

    非回归测试:
    https://github.com/scikit-learn/scikit-learn/issues/28309
    """
    # 在拟合阶段无法转换填充值的情况
    fill_value = 1.5
    X_int64 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    # 创建一个常数填充器对象，使用指定的填充值和缺失值标记
    imputer = SimpleImputer(
        strategy="constant", fill_value=fill_value, missing_values=2
    )
    # 准备一个错误消息，指示无法转换填充值的数据类型
    err_msg = f"fill_value={fill_value!r} (of type {type(fill_value)!r}) cannot be cast"
    # 使用 pytest 的 raises 方法，预期引发 ValueError 异常，并匹配预定义的错误消息
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.fit(X_int64)

    # 在转换阶段无法转换填充值的情况
    X_float64 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float64)
    imputer.fit(X_float64)
    # 准备一个错误消息，指示填充值的数据类型无法转换
    err_msg = (
        f"The dtype of the filling value (i.e. {imputer.statistics_.dtype!r}) "
        "cannot be cast"
    )
    # 使用 pytest 的 raises 方法，预期引发 ValueError 异常，并匹配预定义的错误消息
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.transform(X_int64)

    # 检查当填充值与输入数据类型相同时不会引发错误的情况
    fill_value_list = [np.float64(1.5), 1.5, 1]
    X_float32 = X_float64.astype(np.float32)

    for fill_value in fill_value_list:
        # 创建一个常数填充器对象，使用指定的填充值和缺失值标记
        imputer = SimpleImputer(
            strategy="constant", fill_value=fill_value, missing_values=2
        )
        # 对输入矩阵 X_float32 进行拟合和转换，得到填充后的矩阵 X_trans
        X_trans = imputer.fit_transform(X_float32)
        # 断言填充后的结果 X_trans 的数据类型与输入矩阵 X_float32 的数据类型相等
        assert X_trans.dtype == X_float32.dtype
```