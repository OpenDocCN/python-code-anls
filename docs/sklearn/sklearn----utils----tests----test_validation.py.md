# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_validation.py`

```
# 引入必要的模块和库
"""Tests for input validation functions"""

import numbers  # 导入 numbers 模块，用于数值类型的验证
import re  # 导入 re 模块，用于正则表达式操作
import warnings  # 导入 warnings 模块，用于警告处理
from itertools import product  # 从 itertools 模块导入 product 函数，用于迭代器操作
from operator import itemgetter  # 从 operator 模块导入 itemgetter 函数，用于获取对象的特定元素
from tempfile import NamedTemporaryFile  # 从 tempfile 模块导入 NamedTemporaryFile 类，用于创建临时文件

import numpy as np  # 导入 NumPy 库并重命名为 np
import pytest  # 导入 pytest 测试框架
import scipy.sparse as sp  # 导入 SciPy 的稀疏矩阵模块并重命名为 sp
from pytest import importorskip  # 从 pytest 导入 importorskip 函数，用于导入模块时跳过失败

import sklearn  # 导入 scikit-learn 库
from sklearn._config import config_context  # 从 scikit-learn 内部配置模块导入 config_context 函数
from sklearn._min_dependencies import dependent_packages  # 从 scikit-learn 最小依赖模块导入 dependent_packages 函数
from sklearn.base import BaseEstimator  # 从 scikit-learn 的基础模块导入 BaseEstimator 类
from sklearn.datasets import make_blobs  # 从 scikit-learn 的数据集模块导入 make_blobs 函数
from sklearn.ensemble import RandomForestRegressor  # 从 scikit-learn 的集成模块导入 RandomForestRegressor 类
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning  # 从 scikit-learn 的异常模块导入特定异常类
from sklearn.linear_model import ARDRegression  # 从 scikit-learn 的线性模型模块导入 ARDRegression 类

# TODO: add this estimator into the _mocking module in a further refactoring
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit  # 从 scikit-learn 的评估指标测试模块导入 EstimatorWithFit 类
from sklearn.neighbors import KNeighborsClassifier  # 从 scikit-learn 的邻近模块导入 KNeighborsClassifier 类
from sklearn.random_projection import _sparse_random_matrix  # 从 scikit-learn 的随机投影模块导入 _sparse_random_matrix 函数
from sklearn.svm import SVR  # 从 scikit-learn 的支持向量机模块导入 SVR 类
from sklearn.utils import (  # 从 scikit-learn 的实用工具模块导入多个函数和类
    _safe_indexing,
    as_float_array,
    check_array,
    check_symmetric,
    check_X_y,
    deprecated,
)
from sklearn.utils._mocking import (  # 从 scikit-learn 的模拟工具模块导入 MockDataFrame 和 _MockEstimatorOnOffPrediction
    MockDataFrame,
    _MockEstimatorOnOffPrediction,
)
from sklearn.utils._testing import (  # 从 scikit-learn 的测试工具模块导入多个函数和类
    SkipTest,
    TempMemmap,
    _convert_container,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_equal,
    assert_no_warnings,
    create_memmap_backed_data,
    ignore_warnings,
    skip_if_array_api_compat_not_configured,
)
from sklearn.utils.estimator_checks import _NotAnArray  # 从 scikit-learn 的评估器检查模块导入 _NotAnArray 类
from sklearn.utils.fixes import (  # 从 scikit-learn 的修复模块导入多个修复函数和常量
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DIA_CONTAINERS,
    DOK_CONTAINERS,
    parse_version,
)
from sklearn.utils.validation import (  # 从 scikit-learn 的验证模块导入多个验证函数和常量
    FLOAT_DTYPES,
    _allclose_dense_sparse,
    _check_feature_names_in,
    _check_method_params,
    _check_psd_eigenvalues,
    _check_response_method,
    _check_sample_weight,
    _check_y,
    _deprecate_positional_args,
    _get_feature_names,
    _is_fitted,
    _is_pandas_df,
    _is_polars_df,
    _num_features,
    _num_samples,
    _to_object_array,
    assert_all_finite,
    check_consistent_length,
    check_is_fitted,
    check_memory,
    check_non_negative,
    check_random_state,
    check_scalar,
    column_or_1d,
    has_fit_parameter,
)


def test_make_rng():
    # 测试 check_random_state 工具函数的行为
    assert check_random_state(None) is np.random.mtrand._rand  # 检查当输入 None 时，返回 NumPy 随机数生成器的默认实例
    assert check_random_state(np.random) is np.random.mtrand._rand  # 检查当输入 np.random 时，返回 NumPy 随机数生成器的默认实例

    rng_42 = np.random.RandomState(42)
    assert check_random_state(42).randint(100) == rng_42.randint(100)  # 检查当输入种子 42 时，生成的随机整数相等

    rng_42 = np.random.RandomState(42)
    assert check_random_state(rng_42) is rng_42  # 检查当输入已有的随机数生成器实例时，返回该实例本身

    rng_42 = np.random.RandomState(42)
    assert check_random_state(43).randint(100) != rng_42.randint(100)  # 检查当输入不同的种子时，生成的随机整数不相等

    with pytest.raises(ValueError):  # 检查当输入非法种子时，抛出 ValueError 异常
        check_random_state("some invalid seed")


def test_as_float_array():
    # 测试 as_float_array 函数
    X = np.ones((3, 10), dtype=np.int32)  # 创建一个全为 1 的整数型 NumPy 数组，形状为 (3, 10)
    # 将 X 与从0到9的整数数组相加，结果存回 X
    X = X + np.arange(10, dtype=np.int32)
    # 将 X 转换为浮点数数组，不进行复制
    X2 = as_float_array(X, copy=False)
    # 断言 X2 的数据类型为 np.float32
    assert X2.dtype == np.float32
    
    # 另一个测试
    # 将 X 的数据类型转换为 np.int64
    X = X.astype(np.int64)
    # 将 X 转换为浮点数数组，进行复制
    X2 = as_float_array(X, copy=True)
    # 断言不是同一个对象
    assert as_float_array(X, copy=False) is not X
    # 断言 X2 的数据类型为 np.float64
    assert X2.dtype == np.float64
    
    # 测试整数数据类型 <= 32 位
    tested_dtypes = [bool, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]
    for dtype in tested_dtypes:
        # 将 X 的数据类型转换为当前 dtype
        X = X.astype(dtype)
        # 将 X 转换为浮点数数组
        X2 = as_float_array(X)
        # 断言 X2 的数据类型为 np.float32
        assert X2.dtype == np.float32

    # 测试对象数据类型
    # 将 X 的数据类型转换为 object
    X = X.astype(object)
    # 将 X 转换为浮点数数组，进行复制
    X2 = as_float_array(X, copy=True)
    # 断言 X2 的数据类型为 np.float64
    assert X2.dtype == np.float64

    # 此处 X 的类型正确，不应被修改
    # 创建一个形状为 (3, 2) 的全为 1 的 np.float32 数组
    X = np.ones((3, 2), dtype=np.float32)
    # 断言不复制时的浮点数数组与 X 是同一个对象
    assert as_float_array(X, copy=False) is X
    # 测试如果 X 是 Fortran 排序的，则保持不变
    # 将 X 转换为 Fortran 排序数组
    X = np.asfortranarray(X)
    # 断言 as_float_array 处理后的数组是 Fortran 排序的
    assert np.isfortran(as_float_array(X, copy=True))

    # 测试使用 copy 参数处理一些矩阵
    matrices = [
        sp.csc_matrix(np.arange(5)).toarray(),
        _sparse_random_matrix(10, 10, density=0.10).toarray(),
    ]
    for M in matrices:
        # 将 M 转换为浮点数数组，进行复制
        N = as_float_array(M, copy=True)
        # 修改 N 的第一个元素为 NaN
        N[0, 0] = np.nan
        # 断言 M 中不包含 NaN 值
        assert not np.isnan(M).any()
@pytest.mark.parametrize("X", [(np.random.random((10, 2))), (sp.rand(10, 2).tocsr())])
def test_as_float_array_nan(X):
    # 在随机生成的数组 X 中设置一些元素为 NaN
    X[5, 0] = np.nan
    X[6, 1] = np.nan
    # 调用函数 as_float_array 对数组 X 进行转换，允许 NaN 存在
    X_converted = as_float_array(X, force_all_finite="allow-nan")
    # 断言转换后的数组 X_converted 与原始数组 X 在密集或稀疏表示上的近似相等性
    assert_allclose_dense_sparse(X_converted, X)


def test_np_matrix():
    # 确认输入验证代码不会返回 np.matrix 类型
    X = np.arange(12).reshape(3, 4)
    # 断言 as_float_array 函数不会返回 np.matrix 类型
    assert not isinstance(as_float_array(X), np.matrix)
    # 断言 as_float_array 函数不会返回 np.matrix 类型，即使输入是稀疏矩阵
    assert not isinstance(as_float_array(sp.csc_matrix(X)), np.matrix)


def test_memmap():
    # 确认输入验证代码不会复制内存映射数组

    # 定义一个匿名函数 asflt，其参数 x 调用 as_float_array 函数时使用 copy=False
    asflt = lambda x: as_float_array(x, copy=False)

    # 使用 NamedTemporaryFile 创建临时文件，作为内存映射的基础
    with NamedTemporaryFile(prefix="sklearn-test") as tmp:
        # 创建一个形状为 (10, 10) 的内存映射数组 M，数据类型为 np.float32
        M = np.memmap(tmp, shape=(10, 10), dtype=np.float32)
        M[:] = 0  # 将整个内存映射数组 M 的值设置为 0

        # 遍历函数列表，包括 check_array、np.asarray 和 asflt
        for f in (check_array, np.asarray, asflt):
            # 将内存映射数组 M 作为参数传递给函数 f，得到数组 X
            X = f(M)
            X[:] = 1  # 将数组 X 的所有元素设置为 1
            # 断言数组 X 的扁平化表示与内存映射数组 M 的扁平化表示相等
            assert_array_equal(X.ravel(), M.ravel())
            X[:] = 0  # 将数组 X 的所有元素重新设置为 0


def test_ordering():
    # 检查验证工具是否正确地强制执行顺序

    # 创建一个全为 1 的 10x5 的数组 X
    X = np.ones((10, 5))
    # 遍历数组 X 和其转置 X.T
    for A in X, X.T:
        # 对每个 A 和是否复制的选项进行遍历
        for copy in (True, False):
            # 调用 check_array 函数，指定 C 阶或 F 阶，检查结果为 B
            B = check_array(A, order="C", copy=copy)
            assert B.flags["C_CONTIGUOUS"]  # 断言 B 是 C 阶连续的
            B = check_array(A, order="F", copy=copy)
            assert B.flags["F_CONTIGUOUS"]  # 断言 B 是 F 阶连续的
            if copy:
                assert A is not B  # 如果复制了，断言 A 和 B 不是同一个对象

    # 将数组 X 转换为稀疏矩阵，然后反转数据以破坏 C 阶连续性
    X = sp.csr_matrix(X)
    X.data = X.data[::-1]
    assert not X.data.flags["C_CONTIGUOUS"]  # 断言 X 的数据不是 C 阶连续的


@pytest.mark.parametrize(
    "value, force_all_finite", [(np.inf, False), (np.nan, "allow-nan"), (np.nan, False)]
)
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_matrix])
def test_check_array_force_all_finite_valid(value, force_all_finite, retype):
    # 创建一个包含无穷大或 NaN 值的数组 X
    X = retype(np.arange(4).reshape(2, 2).astype(float))
    X[0, 0] = value  # 将数组 X 中的特定位置设置为 value

    # 调用 check_array 函数，验证 force_all_finite 参数
    X_checked = check_array(X, force_all_finite=force_all_finite, accept_sparse=True)
    # 断言 X 和 X_checked 在密集或稀疏表示上的近似相等性
    assert_allclose_dense_sparse(X, X_checked)


@pytest.mark.parametrize(
    "value, input_name, force_all_finite, match_msg",
    [
        (np.inf, "", True, "Input contains infinity"),
        (np.inf, "X", True, "Input X contains infinity"),
        (np.inf, "sample_weight", True, "Input sample_weight contains infinity"),
        (np.inf, "X", "allow-nan", "Input X contains infinity"),
        (np.nan, "", True, "Input contains NaN"),
        (np.nan, "X", True, "Input X contains NaN"),
        (np.nan, "y", True, "Input y contains NaN"),
        (
            np.nan,
            "",
            "allow-inf",
            'force_all_finite should be a bool or "allow-nan"',
        ),
        (np.nan, "", 1, "Input contains NaN"),
    ],
)
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_matrix])
def test_check_array_force_all_finiteinvalid(
    value, input_name, force_all_finite, match_msg, retype
):
    # 创建一个包含无效输入的数组 X
    X = retype(np.arange(4).reshape(2, 2).astype(float))
    X[0, 0] = value  # 将数组 X 中的特定位置设置为 value

    # 调用 check_array 函数，验证 force_all_finite 参数的无效输入
    with pytest.raises(ValueError, match=match_msg):
        check_array(X, force_all_finite=force_all_finite)
    # 使用 retype 函数将一个 2x2 的 np.float64 类型的数组转换成一个 retype 类型的对象 X
    X = retype(np.arange(4).reshape(2, 2).astype(np.float64))
    # 修改 X 中第一行第一列的元素为变量 value 的值
    X[0, 0] = value
    # 使用 pytest 的 raises 方法捕获 ValueError 异常，并验证异常消息与 match_msg 匹配
    with pytest.raises(ValueError, match=match_msg):
        # 调用 check_array 函数，对输入数组 X 进行检查，同时传递额外的参数
        check_array(
            X,
            input_name=input_name,
            force_all_finite=force_all_finite,
            accept_sparse=True,
        )
# 使用 pytest 的参数化功能来定义多个测试用例，其中 input_name 分别为 "X", "y", "sample_weight"
# retype 分别为 np.asarray 和 sp.csr_matrix
@pytest.mark.parametrize("input_name", ["X", "y", "sample_weight"])
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_matrix])
def test_check_array_links_to_imputer_doc_only_for_X(input_name, retype):
    # 创建一个包含 NaN 的数据集，类型由 retype 指定，例如 np.asarray 或 sp.csr_matrix
    data = retype(np.arange(4).reshape(2, 2).astype(np.float64))
    data[0, 0] = np.nan
    # 创建一个 SVR 估算器实例
    estimator = SVR()
    # 定义一个包含详细信息的扩展消息字符串
    extended_msg = (
        f"\n{estimator.__class__.__name__} does not accept missing values"
        " encoded as NaN natively. For supervised learning, you might want"
        " to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor"
        " which accept missing values encoded as NaNs natively."
        " Alternatively, it is possible to preprocess the"
        " data, for instance by using an imputer transformer in a pipeline"
        " or drop samples with missing values. See"
        " https://scikit-learn.org/stable/modules/impute.html"
        " You can find a list of all estimators that handle NaN values"
        " at the following page:"
        " https://scikit-learn.org/stable/modules/impute.html"
        "#estimators-that-handle-nan-values"
    )

    # 断言检查是否引发 ValueError 异常，异常消息应包含输入变量名和 "contains NaN"
    with pytest.raises(ValueError, match=f"Input {input_name} contains NaN") as ctx:
        check_array(
            data,
            estimator=estimator,
            input_name=input_name,
            accept_sparse=True,
        )

    # 如果 input_name 为 "X"，则验证扩展消息是否在异常消息中
    if input_name == "X":
        assert extended_msg in ctx.value.args[0]
    else:
        assert extended_msg not in ctx.value.args[0]

    # 如果 input_name 为 "X"，则验证是否通过 SVR().fit 引发相同的异常
    if input_name == "X":
        # Veriy that _validate_data is automatically called with the right argument
        # to generate the same exception:
        with pytest.raises(ValueError, match=f"Input {input_name} contains NaN") as ctx:
            SVR().fit(data, np.ones(data.shape[0]))
        assert extended_msg in ctx.value.args[0]


# 测试检查包含对象的数组是否强制所有元素为有限值
def test_check_array_force_all_finite_object():
    # 创建一个包含对象类型数据的数组 X，其中包含 NaN
    X = np.array([["a", "b", np.nan]], dtype=object).T

    # 执行 check_array 函数，验证参数设置为允许 NaN，预期 X 与 X_checked 相同
    X_checked = check_array(X, dtype=None, force_all_finite="allow-nan")
    assert X is X_checked

    # 再次执行 check_array 函数，验证参数设置为不强制所有元素为有限值，预期 X 与 X_checked 相同
    X_checked = check_array(X, dtype=None, force_all_finite=False)
    assert X is X_checked

    # 断言检查是否引发 ValueError 异常，异常消息应包含 "Input contains NaN"
    with pytest.raises(ValueError, match="Input contains NaN"):
        check_array(X, dtype=None, force_all_finite=True)


# 使用参数化测试多个输入数据 X 和对应的错误消息 err_msg
@pytest.mark.parametrize(
    "X, err_msg",
    [
        (
            np.array([[1, np.nan]]),
            "Input contains NaN.",
        ),
        (
            np.array([[1, np.nan]]),
            "Input contains NaN.",
        ),
        (
            np.array([[1, np.inf]]),
            "Input contains infinity or a value too large for.*int",
        ),
        (np.array([[1, np.nan]], dtype=object), "cannot convert float NaN to integer"),
    ],
)
# 参数化 force_all_finite 为 True 和 False，测试 check_array 函数强制所有元素为有限值的情况
@pytest.mark.parametrize("force_all_finite", [True, False])
def test_check_array_force_all_finite_object_unsafe_casting(
    X, err_msg, force_all_finite
):
    # 将包含 NaN 或 inf 的浮点数组强制转换为整数 dtype 应该引发异常
    # （测试未完整给出）
    # 使用 pytest 库中的 pytest.raises() 函数捕获 ValueError 异常，同时验证异常信息与 err_msg 匹配
    with pytest.raises(ValueError, match=err_msg):
        # 调用 check_array() 函数，传入参数 X, dtype=int, force_all_finite=force_all_finite，用于检查数组
        check_array(X, dtype=int, force_all_finite=force_all_finite)
def test_check_array_series_err_msg():
    """
    Check that we raise a proper error message when passing a Series and we expect a
    2-dimensional container.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27498
    """
    # 导入 pytest 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个 Pandas Series 对象
    ser = pd.Series([1, 2, 3])
    # 构造错误信息，指示期望的是二维容器，但传入的是 Series 类型
    msg = f"Expected a 2-dimensional container but got {type(ser)} instead."
    # 使用 pytest 断言捕获 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match=msg):
        # 调用 check_array 函数，传入 Series 对象和 ensure_2d=True 参数
        check_array(ser, ensure_2d=True)


@ignore_warnings
def test_check_array():
    # accept_sparse == False
    # 在稀疏输入时引发错误
    X = [[1, 2], [3, 4]]
    # 将 X 转换为 CSR 稀疏矩阵
    X_csr = sp.csr_matrix(X)
    # 使用 pytest 断言捕获 TypeError 异常
    with pytest.raises(TypeError):
        # 调用 check_array 函数，传入 CSR 矩阵
        check_array(X_csr)

    # ensure_2d=False
    # 测试 ensure_2d=False 的情况
    X_array = check_array([0, 1, 2], ensure_2d=False)
    assert X_array.ndim == 1
    # ensure_2d=True with 1d array
    # 测试 ensure_2d=True 且传入一维数组的情况
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead"):
        check_array([0, 1, 2], ensure_2d=True)

    # ensure_2d=True with scalar array
    # 测试 ensure_2d=True 且传入标量数组的情况
    with pytest.raises(ValueError, match="Expected 2D array, got scalar array instead"):
        check_array(10, ensure_2d=True)

    # ensure_2d=True with 1d sparse array
    # 如果存在稀疏数组，则测试 ensure_2d=True 且传入一维稀疏数组的情况
    if hasattr(sp, "csr_array"):
        sparse_row = next(iter(sp.csr_array(X)))
        if sparse_row.ndim == 1:
            # 在 scipy 1.14 及更高版本中，稀疏行是一维的，而之前是二维的
            with pytest.raises(ValueError, match="Expected 2D input, got"):
                check_array(sparse_row, accept_sparse=True, ensure_2d=True)

    # don't allow ndim > 3
    # 不允许超过三维的输入
    X_ndim = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        # 调用 check_array 函数，传入超过三维的数组
        check_array(X_ndim)
    # 使用 allow_nd=True 参数调用 check_array 函数，不应该引发异常
    check_array(X_ndim, allow_nd=True)

    # dtype and order enforcement.
    # dtype 和 order 的强制规定
    X_C = np.arange(4).reshape(2, 2).copy("C")
    X_F = X_C.copy("F")
    X_int = X_C.astype(int)
    X_float = X_C.astype(float)
    Xs = [X_C, X_F, X_int, X_float]
    dtypes = [np.int32, int, float, np.float32, None, bool, object]
    orders = ["C", "F", None]
    copys = [True, False]

    # 遍历 Xs, dtypes, orders, copys 的所有组合
    for X, dtype, order, copy in product(Xs, dtypes, orders, copys):
        # 调用 check_array 函数，传入不同的 dtype, order, copy 参数
        X_checked = check_array(X, dtype=dtype, order=order, copy=copy)
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if order == "C":
            assert X_checked.flags["C_CONTIGUOUS"]
            assert not X_checked.flags["F_CONTIGUOUS"]
        elif order == "F":
            assert X_checked.flags["F_CONTIGUOUS"]
            assert not X_checked.flags["C_CONTIGUOUS"]
        if copy:
            assert X is not X_checked
        else:
            # 如果已经是符合条件的数组，则不会复制
            if (
                X.dtype == X_checked.dtype
                and X_checked.flags["C_CONTIGUOUS"] == X.flags["C_CONTIGUOUS"]
                and X_checked.flags["F_CONTIGUOUS"] == X.flags["F_CONTIGUOUS"]
            ):
                assert X is X_checked
    # 确保 sparse 不为 None

    # 尝试不同类型的稀疏格式
    Xs = []
    Xs.extend(
        [
            sparse_container(X_C)
            for sparse_container in CSR_CONTAINERS
            + CSC_CONTAINERS
            + COO_CONTAINERS
            + DOK_CONTAINERS
        ]
    )
    Xs.extend([Xs[0].astype(np.int64), Xs[0].astype(np.float64)])

    # 允许的稀疏格式
    accept_sparses = [["csr", "coo"], ["coo", "dok"]]
    # scipy 稀疏矩阵不支持对象类型(dtype=object)，因此在此循环中跳过该类型
    non_object_dtypes = [dt for dt in dtypes if dt is not object]
    for X, dtype, accept_sparse, copy in product(
        Xs, non_object_dtypes, accept_sparses, copys
    ):
        # 对输入的 X 进行验证，确保其为数组，并根据指定的 dtype 和 accept_sparse 进行处理
        X_checked = check_array(X, dtype=dtype, accept_sparse=accept_sparse, copy=copy)
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if X.format in accept_sparse:
            # 如果允许的话，X 的格式不应改变
            assert X.format == X_checked.format
        else:
            # 如果发生了转换，应确认 X_checked 的格式与 accept_sparse 中的第一个值相同
            assert X_checked.format == accept_sparse[0]
        if copy:
            assert X is not X_checked
        else:
            # 如果 X 的 dtype 和格式已经符合要求，且不复制，应保持 X 和 X_checked 是同一个对象
            if X.dtype == X_checked.dtype and X.format == X_checked.format:
                assert X is X_checked

    # 其他输入格式
    # 将列表转换为数组
    X_dense = check_array([[1, 2], [3, 4]])
    assert isinstance(X_dense, np.ndarray)
    # 对于太深的列表应该引发 ValueError
    with pytest.raises(ValueError):
        check_array(X_ndim.tolist())
    # 使用 allow_nd=True 时，不应该引发异常
    check_array(X_ndim.tolist(), allow_nd=True)

    # 将奇怪的输入转换为数组
    X_no_array = _NotAnArray(X_dense)
    result = check_array(X_no_array)
    assert isinstance(result, np.ndarray)
# 使用 pytest.mark.parametrize 装饰器为 test_check_array_numeric_error 函数定义多组输入参数 X
@pytest.mark.parametrize(
    "X",
    [
        [["1", "2"], ["3", "4"]],  # 字符串数组的列表
        np.array([["1", "2"], ["3", "4"]], dtype="U"),  # NumPy Unicode 字符串数组
        np.array([["1", "2"], ["3", "4"]], dtype="S"),  # NumPy 字节字符串数组
        [[b"1", b"2"], [b"3", b"4"]],  # 字节字符串数组的列表
        np.array([[b"1", b"2"], [b"3", b"4"]], dtype="V1"),  # NumPy 无符号字节字符串数组
    ],
)
def test_check_array_numeric_error(X):
    """Test that check_array errors when it receives an array of bytes/string
    while a numeric dtype is required."""
    expected_msg = r"dtype='numeric' is not compatible with arrays of bytes/strings"
    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=expected_msg):
        check_array(X, dtype="numeric")


# 使用 pytest.mark.parametrize 装饰器为 test_check_array_pandas_na_support 函数定义多组输入参数 pd_dtype 和 (dtype, expected_dtype)
@pytest.mark.parametrize(
    "pd_dtype", ["Int8", "Int16", "UInt8", "UInt16", "Float32", "Float64"]
)
@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        ([np.float32, np.float64], np.float32),  # 期望结果为 np.float32 的列表
        (np.float64, np.float64),  # dtype 和 expected_dtype 都为 np.float64
        ("numeric", np.float64),  # dtype 为 "numeric"，expected_dtype 为 np.float64
    ],
)
def test_check_array_pandas_na_support(pd_dtype, dtype, expected_dtype):
    # Test pandas numerical extension arrays with pd.NA
    pd = pytest.importorskip("pandas")  # 导入并检查 pandas 是否可用

    if pd_dtype in {"Float32", "Float64"}:
        # Extension dtypes with Floats was added in 1.2
        pd = pytest.importorskip("pandas", minversion="1.2")  # 检查 pandas 版本是否至少为 1.2

    # 创建一个 NumPy 数组 X_np，包含 NaN 值
    X_np = np.array(
        [[1, 2, 3, np.nan, np.nan], [np.nan, np.nan, 8, 4, 6], [1, 2, 3, 4, 5]]
    ).T

    # 使用 pd.DataFrame 创建包含 pd_dtype 类型和列名的 DataFrame X
    X = pd.DataFrame(X_np, dtype=pd_dtype, columns=["a", "b", "c"])
    # 将列 c 转换为 float 类型
    X["c"] = X["c"].astype("float")
    # 使用 check_array 检查 X，强制所有有限的值为允许 NaN，指定 dtype
    X_checked = check_array(X, force_all_finite="allow-nan", dtype=dtype)
    assert_allclose(X_checked, X_np)  # 断言 X_checked 与 X_np 数组近似相等
    assert X_checked.dtype == expected_dtype  # 断言 X_checked 的 dtype 与 expected_dtype 相等

    # 再次使用 check_array 检查 X，不强制所有有限的值为允许 NaN，指定 dtype
    X_checked = check_array(X, force_all_finite=False, dtype=dtype)
    assert_allclose(X_checked, X_np)  # 断言 X_checked 与 X_np 数组近似相等
    assert X_checked.dtype == expected_dtype  # 断言 X_checked 的 dtype 与 expected_dtype 相等

    msg = "Input contains NaN"
    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        check_array(X, force_all_finite=True)


def test_check_array_panadas_na_support_series():
    """Check check_array is correct with pd.NA in a series."""
    pd = pytest.importorskip("pandas")  # 导入并检查 pandas 是否可用

    # 创建一个包含 pd.NA 的 Series X_int64
    X_int64 = pd.Series([1, 2, pd.NA], dtype="Int64")

    msg = "Input contains NaN"
    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        check_array(X_int64, force_all_finite=True, ensure_2d=False)

    # 使用 check_array 检查 X_int64，不强制所有有限的值为允许 NaN，不确保为 2D
    X_out = check_array(X_int64, force_all_finite=False, ensure_2d=False)
    assert_allclose(X_out, [1, 2, np.nan])  # 断言 X_out 与 [1, 2, NaN] 数组近似相等
    assert X_out.dtype == np.float64  # 断言 X_out 的 dtype 为 np.float64

    # 再次使用 check_array 检查 X_int64，不强制所有有限的值为允许 NaN，不确保为 2D，指定 dtype 为 np.float32
    X_out = check_array(
        X_int64, force_all_finite=False, ensure_2d=False, dtype=np.float32
    )
    assert_allclose(X_out, [1, 2, np.nan])  # 断言 X_out 与 [1, 2, NaN] 数组近似相等
    assert X_out.dtype == np.float32  # 断言 X_out 的 dtype 为 np.float32


def test_check_array_pandas_dtype_casting():
    # test that data-frames with homogeneous dtype are not upcast
    pd = pytest.importorskip("pandas")  # 导入并检查 pandas 是否可用
    # 创建一个 dtype 为 np.float32 的 NumPy 数组 X
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    # 创建 X 的 DataFrame 表示 X_df
    X_df = pd.DataFrame(X)
    assert check_array(X_df).dtype == np.float32  # 断言 check_array(X_df) 的 dtype 为 np.float32
    # 检查传入的数据数组 X_df 是否符合指定的浮点数类型
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    # 将 X_df 的第一列数据类型转换为 np.float16
    X_df = X_df.astype({0: np.float16})
    # 断言 X_df 的各列数据类型是否分别为 np.float16, np.float32, np.float32
    assert_array_equal(X_df.dtypes, (np.float16, np.float32, np.float32))
    # 检查 X_df 是否符合默认的浮点数类型 np.float32
    assert check_array(X_df).dtype == np.float32
    # 检查 X_df 是否符合指定的浮点数类型 FLOAT_DTYPES 中的 np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    # 将 X_df 的第一列数据类型转换为 np.int16
    X_df = X_df.astype({0: np.int16})
    # float16, int16, float32 转换后均为 float32
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    # 将 X_df 的第三列数据类型转换为 np.float16
    X_df = X_df.astype({2: np.float16})
    # float16, int16, float16 转换后均为 float32
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    # 将 X_df 所有列的数据类型转换为 np.int16
    X_df = X_df.astype(np.int16)
    # 检查 X_df 是否符合 np.int16 类型
    assert check_array(X_df).dtype == np.int16
    # 因为目前不使用提升规则来确定目标类型，所以转换为默认的 float64 类型
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float64

    # 检查我们是否以半合理的方式处理 pandas 的数据类型
    # 这是复杂的，因为在转换之前我们无法确定这应该是整数。
    cat_df = pd.DataFrame({"cat_col": pd.Categorical([1, 2, 3])})
    # 断言检查处理类别数据的数组是否为 np.int64 类型
    assert check_array(cat_df).dtype == np.int64
    # 断言检查处理类别数据的数组是否为 FLOAT_DTYPES 中的 np.float64 类型
    assert check_array(cat_df, dtype=FLOAT_DTYPES).dtype == np.float64
def test_check_array_on_mock_dataframe():
    # 创建一个NumPy数组作为测试数据
    arr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
    # 使用MockDataFrame类包装数组，创建一个模拟的数据框架对象
    mock_df = MockDataFrame(arr)
    # 调用check_array函数，返回验证后的数组
    checked_arr = check_array(mock_df)
    # 断言验证后的数组的数据类型与原始数组相同
    assert checked_arr.dtype == arr.dtype
    # 再次调用check_array函数，指定数据类型为np.float32，返回验证后的数组
    checked_arr = check_array(mock_df, dtype=np.float32)
    # 断言验证后的数组的数据类型为np.float32
    assert checked_arr.dtype == np.dtype(np.float32)


def test_check_array_dtype_stability():
    # 测试列表包含整数时不会转换为浮点数
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # 断言经过check_array处理后的数组数据类型的kind属性为整数（'i'）
    assert check_array(X).dtype.kind == "i"
    # 断言经过check_array处理后的数组数据类型的kind属性为整数（'i'），且不强制二维化
    assert check_array(X, ensure_2d=False).dtype.kind == "i"


def test_check_array_dtype_warning():
    X_int_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    X_float32 = np.asarray(X_int_list, dtype=np.float32)
    X_int64 = np.asarray(X_int_list, dtype=np.int64)
    X_csr_float32 = sp.csr_matrix(X_float32)
    X_csc_float32 = sp.csc_matrix(X_float32)
    X_csc_int32 = sp.csc_matrix(X_int64, dtype=np.int32)
    integer_data = [X_int64, X_csc_int32]
    float32_data = [X_float32, X_csr_float32, X_csc_float32]
    for X in integer_data:
        # 调用assert_no_warnings确保在不产生警告的情况下调用check_array函数
        X_checked = assert_no_warnings(
            check_array, X, dtype=np.float64, accept_sparse=True
        )
        # 断言验证后的数组数据类型为np.float64
        assert X_checked.dtype == np.float64

    for X in float32_data:
        # 调用assert_no_warnings确保在不产生警告的情况下调用check_array函数
        X_checked = assert_no_warnings(
            check_array, X, dtype=[np.float64, np.float32], accept_sparse=True
        )
        # 断言验证后的数组数据类型为np.float32
        assert X_checked.dtype == np.float32
        # 断言验证后的数组与原始数组对象相同
        assert X_checked is X

        # 调用assert_no_warnings确保在不产生警告的情况下调用check_array函数
        X_checked = assert_no_warnings(
            check_array,
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=["csr", "dok"],
            copy=True,
        )
        # 断言验证后的数组数据类型为np.float32
        assert X_checked.dtype == np.float32
        # 断言验证后的数组与原始数组对象不同
        assert X_checked is not X

    # 调用assert_no_warnings确保在不产生警告的情况下调用check_array函数
    X_checked = assert_no_warnings(
        check_array,
        X_csc_float32,
        dtype=[np.float64, np.float32],
        accept_sparse=["csr", "dok"],
        copy=False,
    )
    # 断言验证后的数组数据类型为np.float32
    assert X_checked.dtype == np.float32
    # 断言验证后的数组与原始数组对象不同
    assert X_checked is not X_csc_float32
    # 断言验证后的数组格式为"csr"
    assert X_checked.format == "csr"


def test_check_array_accept_sparse_type_exception():
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_matrix(X)
    invalid_type = SVR()

    # 检查是否抛出预期的TypeError异常，验证不接受稀疏数据的情况
    msg = (
        "Sparse data was passed, but dense data is required. "
        r"Use '.toarray\(\)' to convert to a dense numpy array."
    )
    with pytest.raises(TypeError, match=msg):
        check_array(X_csr, accept_sparse=False)

    # 检查是否抛出预期的ValueError异常，验证accept_sparse参数类型不正确的情况
    msg = (
        "Parameter 'accept_sparse' should be a string, "
        "boolean or list of strings. You provided 'accept_sparse=.*'."
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=invalid_type)

    # 检查是否抛出预期的ValueError异常，验证accept_sparse参数为空列表或元组的情况
    msg = (
        "When providing 'accept_sparse' as a tuple or list, "
        "it must contain at least one string value."
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=[])
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=())
    # 使用 pytest 框架中的 `pytest.raises` 上下文管理器，检查是否引发了 TypeError 异常，并验证异常消息中是否包含 "SVR"
    with pytest.raises(TypeError, match="SVR"):
        # 调用 `check_array` 函数，传入参数 X_csr 和 accept_sparse=[invalid_type]
        # `check_array` 函数用于验证和转换输入数据，此处验证输入是否为稀疏矩阵格式，其中 accept_sparse 参数包含一个无效类型的列表
        check_array(X_csr, accept_sparse=[invalid_type])
# 定义一个测试函数，验证在接受稀疏矩阵时不会抛出异常
def test_check_array_accept_sparse_no_exception():
    # 创建一个普通的二维列表
    X = [[1, 2], [3, 4]]
    # 将二维列表转换为稀疏矩阵格式的CSR（压缩稀疏行）格式
    X_csr = sp.csr_matrix(X)

    # 使用accept_sparse=True参数调用check_array函数，验证是否接受稀疏矩阵
    check_array(X_csr, accept_sparse=True)
    # 使用accept_sparse="csr"参数调用check_array函数，验证是否接受特定格式的稀疏矩阵
    check_array(X_csr, accept_sparse="csr")
    # 使用accept_sparse=["csr"]参数调用check_array函数，验证是否接受特定格式的稀疏矩阵
    check_array(X_csr, accept_sparse=["csr"])
    # 使用accept_sparse=("csr",)参数调用check_array函数，验证是否接受特定格式的稀疏矩阵
    check_array(X_csr, accept_sparse=("csr",))


# 定义一个参数化的测试fixture，用于生成不同格式的稀疏矩阵
@pytest.fixture(params=["csr", "csc", "coo", "bsr"])
def X_64bit(request):
    # 创建一个大小为20x10的随机稀疏矩阵，格式由request.param指定
    X = sp.rand(20, 10, format=request.param)

    # 如果当前格式为"coo"
    if request.param == "coo":
        # 检查是否具有.coords属性，用于处理Scipy >= 1.13版本的特殊情况
        if hasattr(X, "coords"):
            # 将.coords属性的所有元素转换为int64类型，适应不同版本的Scipy
            X.coords = tuple(v.astype("int64") for v in X.coords)
        else:
            # 对于Scipy < 1.13版本，将.row和.col属性转换为int64类型
            X.row = X.row.astype("int64")
            X.col = X.col.astype("int64")
    else:
        # 对于其他格式的稀疏矩阵，将.indices属性和.indptr属性转换为int64类型
        X.indices = X.indices.astype("int64")
        X.indptr = X.indptr.astype("int64")

    # 返回生成的稀疏矩阵
    yield X


# 定义一个测试函数，验证在接受大稀疏矩阵且允许时不会抛出异常
def test_check_array_accept_large_sparse_no_exception(X_64bit):
    # 当允许大稀疏矩阵时，调用check_array函数验证是否接受X_64bit作为输入
    check_array(X_64bit, accept_large_sparse=True, accept_sparse=True)


# 定义一个测试函数，验证在不允许大稀疏矩阵时会抛出异常
def test_check_array_accept_large_sparse_raise_exception(X_64bit):
    # 准备异常消息，说明不允许使用64位整数索引的大稀疏矩阵
    msg = (
        "Only sparse matrices with 32-bit integer indices "
        "are accepted. Got int64 indices. Please do report"
    )
    # 使用pytest.raises检查是否抛出预期的异常，并匹配异常消息内容
    with pytest.raises(ValueError, match=msg):
        check_array(X_64bit, accept_sparse=True, accept_large_sparse=False)


# 定义一个测试函数，验证在检查最小样本数和特征数时的异常消息
def test_check_array_min_samples_and_features_messages():
    # 空列表默认被视为二维数据，验证是否抛出特定异常消息
    msg = r"0 feature\(s\) \(shape=\(1, 0\)\) while a minimum of 1 is" " required."
    with pytest.raises(ValueError, match=msg):
        check_array([[]])

    # 当ensure_2d=False时，空列表被视为一维数据，验证是否抛出特定异常消息
    msg = r"0 sample\(s\) \(shape=\(0,\)\) while a minimum of 1 is required."
    with pytest.raises(ValueError, match=msg):
        check_array([], ensure_2d=False)

    # 当输入为标量42时，验证是否抛出特定异常消息
    msg = r"Singleton array array\(42\) cannot be considered a valid" " collection."
    with pytest.raises(TypeError, match=msg):
        check_array(42, ensure_2d=False)

    # 创建一个包含1行10列全为1的数组X和一个长度为1的数组y，验证是否抛出特定异常消息
    X = np.ones((1, 10))
    y = np.ones(1)
    msg = r"1 sample\(s\) \(shape=\(1, 10\)\) while a minimum of 2 is" " required."
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2)

    # 验证即使数据为2维，也会抛出相同的异常消息，尽管这不是必需的
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2, ensure_2d=False)
    # 模拟一个需要至少三个特征的模型（例如，SelectKBest 选择 k=3 的情况）
    X = np.ones((10, 2))
    y = np.ones(2)
    # 定义用于匹配错误消息的正则表达式字符串
    msg = r"2 feature\(s\) \(shape=\(10, 2\)\) while a minimum of 3 is" " required."
    # 使用 pytest 来确保调用 check_X_y 函数时会引发 ValueError，并且错误消息符合预期
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3)
    
    # 当维度数量为 2 时，即使允许 allow_nd，也仅启用特征检查：
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3, allow_nd=True)
    
    # 模拟管道阶段修剪了所有特征的 2D 数据集的情况
    X = np.empty(0).reshape(10, 0)
    y = np.ones(10)
    # 定义用于匹配错误消息的正则表达式字符串
    msg = r"0 feature\(s\) \(shape=\(10, 0\)\) while a minimum of 1 is" " required."
    # 使用 pytest 来确保调用 check_X_y 函数时会引发 ValueError，并且错误消息符合预期
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y)
    
    # 默认情况下，不会检查 nd 数据的最小特征数要求：
    X = np.ones((10, 0, 28, 28))
    y = np.ones(10)
    # 调用 check_X_y 函数，使用 allow_nd=True 参数，确保不会引发异常
    X_checked, y_checked = check_X_y(X, y, allow_nd=True)
    # 检查返回的 X 和 y 是否与输入相等
    assert_array_equal(X, X_checked)
    assert_array_equal(y, y_checked)
# 测试函数，用于检查复杂数据的错误处理
def test_check_array_complex_data_error():
    # 创建一个复杂数据的 NumPy 数组 X
    X = np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]])
    # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定错误消息
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # 列表的列表
    X = [[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # 元组的元组
    X = ((1 + 2j, 3 + 4j, 5 + 7j), (2 + 3j, 4 + 5j, 6 + 7j))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # 列表的 NumPy 数组
    X = [np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j])]
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # 元组的 NumPy 数组
    X = (np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j]))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # 模拟的 DataFrame
    X = MockDataFrame(np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # 稀疏矩阵
    X = sp.coo_matrix([[0, 1 + 2j], [0, 0]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # 目标变量通常不会经过 check_array，但也不应接受复杂数据
    y = np.array([1 + 2j, 3 + 4j, 5 + 7j, 2 + 3j, 4 + 5j, 6 + 7j])
    with pytest.raises(ValueError, match="Complex data not supported"):
        _check_y(y)


# 测试函数，用于检查是否有适合参数
def test_has_fit_parameter():
    # 断言 KNeighborsClassifier 类没有 sample_weight 参数
    assert not has_fit_parameter(KNeighborsClassifier, "sample_weight")
    # 断言 RandomForestRegressor 类有 sample_weight 参数
    assert has_fit_parameter(RandomForestRegressor, "sample_weight")
    # 断言 SVR 类有 sample_weight 参数
    assert has_fit_parameter(SVR, "sample_weight")
    # 断言 SVR 实例化对象有 sample_weight 参数
    assert has_fit_parameter(SVR(), "sample_weight")

    # 包含被弃用的 fit 方法的测试类
    class TestClassWithDeprecatedFitMethod:
        @deprecated("Deprecated for the purpose of testing has_fit_parameter")
        def fit(self, X, y, sample_weight=None):
            pass

    # 断言对于包含被弃用 fit 方法的类，has_fit_parameter 方法依然有效
    assert has_fit_parameter(
        TestClassWithDeprecatedFitMethod, "sample_weight"
    ), "has_fit_parameter fails for class with deprecated fit method."


# 测试函数，用于检查对称性
def test_check_symmetric():
    # 创建一个对称数组和不符合要求的数组
    arr_sym = np.array([[0, 1], [1, 2]])
    arr_bad = np.ones(2)
    arr_asym = np.array([[0, 2], [0, 2]])

    # 待测试的数组字典
    test_arrays = {
        "dense": arr_asym,
        "dok": sp.dok_matrix(arr_asym),
        "csr": sp.csr_matrix(arr_asym),
        "csc": sp.csc_matrix(arr_asym),
        "coo": sp.coo_matrix(arr_asym),
        "lil": sp.lil_matrix(arr_asym),
        "bsr": sp.bsr_matrix(arr_asym),
    }

    # 检查对不合格输入的错误处理
    with pytest.raises(ValueError):
        check_symmetric(arr_bad)

    # 检查非对称数组是否被正确处理成对称的
    # 遍历 test_arrays 字典中的每个键值对，其中 arr_format 是格式，arr 是数组数据
    for arr_format, arr in test_arrays.items():
        # 使用 pytest.warns 检查是否有 UserWarning 警告
        with pytest.warns(UserWarning):
            # 调用 check_symmetric 函数检查数组 arr 是否对称
            check_symmetric(arr)
        
        # 使用 pytest.raises 检查是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            # 调用 check_symmetric 函数检查数组 arr 是否对称，并要求抛出异常
            check_symmetric(arr, raise_exception=True)

        # 调用 check_symmetric 函数检查数组 arr 是否对称，但不产生警告
        output = check_symmetric(arr, raise_warning=False)
        
        # 如果输出是稀疏矩阵
        if sp.issparse(output):
            # 断言输出的格式与 arr_format 相同
            assert output.format == arr_format
            # 断言稀疏矩阵转为稠密矩阵后与 arr_sym 相等
            assert_array_equal(output.toarray(), arr_sym)
        else:
            # 断言输出与 arr_sym 相等
            assert_array_equal(output, arr_sym)
def test_check_is_fitted_with_is_fitted():
    # 定义一个继承自 BaseEstimator 的 Estimator 类
    class Estimator(BaseEstimator):
        # 定义 fit 方法用于训练模型
        def fit(self, **kwargs):
            # 将 _is_fitted 标记为 True 表示模型已经拟合
            self._is_fitted = True
            return self

        # 定义 __sklearn_is_fitted__ 方法用于检查模型是否已拟合
        def __sklearn_is_fitted__(self):
            return hasattr(self, "_is_fitted") and self._is_fitted

    # 使用 pytest 检查是否会抛出 NotFittedError 异常
    with pytest.raises(NotFittedError):
        check_is_fitted(Estimator())
    # 检查已拟合的 Estimator 实例不会抛出异常
    check_is_fitted(Estimator().fit())


def test_check_is_fitted():
    # 检查当传入非估计器实例时是否会引发 TypeError
    with pytest.raises(TypeError):
        check_is_fitted(ARDRegression)
    with pytest.raises(TypeError):
        check_is_fitted("SVR")

    # 创建 ARDRegression 和 SVR 实例
    ard = ARDRegression()
    svr = SVR()

    try:
        # 检查未拟合的模型是否会抛出 NotFittedError 异常
        with pytest.raises(NotFittedError):
            check_is_fitted(ard)
        with pytest.raises(NotFittedError):
            check_is_fitted(svr)
    except ValueError:
        assert False, "check_is_fitted failed with ValueError"

    # 使用匹配参数检查 ValueError 是否是 NotFittedError 的子类
    msg = "Random message %(name)s, %(name)s"
    match = "Random message ARDRegression, ARDRegression"
    with pytest.raises(ValueError, match=match):
        check_is_fitted(ard, msg=msg)

    msg = "Another message %(name)s, %(name)s"
    match = "Another message SVR, SVR"
    with pytest.raises(AttributeError, match=match):
        check_is_fitted(svr, msg=msg)

    # 对 ARDRegression 和 SVR 进行拟合操作
    ard.fit(*make_blobs())
    svr.fit(*make_blobs())

    # 断言已拟合的模型不会抛出异常
    assert check_is_fitted(ard) is None
    assert check_is_fitted(svr) is None


def test_check_is_fitted_attributes():
    # 定义一个简单的估计器类 MyEstimator
    class MyEstimator:
        # 定义 fit 方法接受 X, y 参数并返回 self
        def fit(self, X, y):
            return self

    # 设置未拟合时的错误消息
    msg = "not fitted"
    est = MyEstimator()

    # 使用 _is_fitted 函数检查是否已拟合并预期会失败
    assert not _is_fitted(est, attributes=["a_", "b_"])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"])

    # 使用 all_or_any 参数测试多个属性的拟合情况
    assert not _is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    assert not _is_fitted(est, attributes=["a_", "b_"], all_or_any=any)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)

    # 给 est 实例添加属性 a_
    est.a_ = "a"
    # 继续检查未拟合的情况
    assert not _is_fitted(est, attributes=["a_", "b_"])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"])

    # 使用 all_or_any 参数测试部分拟合的情况
    assert not _is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)

    # 部分属性已拟合，部分属性未拟合的情况
    assert _is_fitted(est, attributes=["a_", "b_"], all_or_any=any)
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)

    # 给 est 实例添加属性 b_
    est.b_ = "b"
    # 检查所有属性是否都已拟合
    assert _is_fitted(est, attributes=["a_", "b_"])
    check_is_fitted(est, attributes=["a_", "b_"])
    assert _is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    # 使用断言检查估算器（est）是否已经适配（拟合）到数据上，并且具有指定的属性列表
    assert _is_fitted(est, attributes=["a_", "b_"], all_or_any=any)
    # 检查估算器（est）是否已经适配（拟合）到数据上，并且具有指定的属性列表，如果没有则抛出异常
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)
# 使用 pytest.mark.parametrize 装饰器为 test_check_is_fitted_with_attributes 函数创建多个参数化测试用例，
# 分别测试 wrap 参数为 itemgetter(0)、list 和 tuple 时的情况
@pytest.mark.parametrize(
    "wrap", [itemgetter(0), list, tuple], ids=["single", "list", "tuple"]
)
def test_check_is_fitted_with_attributes(wrap):
    # 创建 ARDRegression 的实例
    ard = ARDRegression()
    # 使用 pytest.raises 检查是否会抛出 NotFittedError 异常，匹配错误消息 "is not fitted yet"
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        # 调用 check_is_fitted 函数，检查 ard 实例是否已经拟合，wrap(["coef_"]) 是属性名称的包装
        check_is_fitted(ard, wrap(["coef_"]))

    # 使用 make_blobs 函数生成样本数据，调用 ARDRegression 实例的 fit 方法进行拟合
    ard.fit(*make_blobs())

    # 检查已拟合的情况，预期不会抛出异常
    check_is_fitted(ard, wrap(["coef_"]))

    # 使用不存在的属性名称 wrap(["coef_bad_"]) 检查是否抛出 NotFittedError 异常
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        check_is_fitted(ard, wrap(["coef_bad_"]))


# test_check_consistent_length 函数测试 check_consistent_length 函数的不同参数组合
def test_check_consistent_length():
    check_consistent_length([1], [2], [3], [4], [5])
    check_consistent_length([[1, 2], [[1, 2]]], [1, 2], ["a", "b"])
    check_consistent_length([1], (2,), np.array([3]), sp.csr_matrix((1, 2)))
    # 检查当参数列表长度不一致时是否抛出 ValueError 异常，匹配错误消息 "inconsistent numbers of samples"
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        check_consistent_length([1, 2], [1])
    # 检查参数类型不匹配时是否抛出 TypeError 异常，匹配类似 "<class 'int'>" 的错误消息
    with pytest.raises(TypeError, match=r"got <\w+ 'int'>"):
        check_consistent_length([1, 2], 1)
    # 检查参数类型不匹配时是否抛出 TypeError 异常，匹配类似 "<class 'object'>" 的错误消息
    with pytest.raises(TypeError, match=r"got <\w+ 'object'>"):
        check_consistent_length([1, 2], object())

    with pytest.raises(TypeError):
        check_consistent_length([1, 2], np.array(1))

    # 测试 RandomForestRegressor 实例作为参数时是否抛出 TypeError 异常，匹配错误消息 "Expected sequence or array-like"
    with pytest.raises(TypeError, match="Expected sequence or array-like"):
        check_consistent_length([1, 2], RandomForestRegressor())
    # XXX: We should have a test with a string, but what is correct behaviour?


# test_check_dataframe_fit_attribute 函数测试 check_consistent_length 函数处理 pandas DataFrame 的情况
def test_check_dataframe_fit_attribute():
    # 检查是否能正确处理包含 'fit' 列的 pandas DataFrame，不应抛出错误
    # 参考：https://github.com/scikit-learn/scikit-learn/issues/8415
    try:
        import pandas as pd

        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X_df = pd.DataFrame(X, columns=["a", "b", "fit"])
        check_consistent_length(X_df)
    except ImportError:
        raise SkipTest("Pandas not found")


# test_suppress_validation 函数测试 assert_all_finite 函数的行为，使用 sklearn.set_config 设置 assume_finite 参数
def test_suppress_validation():
    X = np.array([0, np.inf])
    # 检查包含无限值的数组 X 是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        assert_all_finite(X)
    # 使用 sklearn.set_config 设置 assume_finite=True，再次检查 assert_all_finite 函数
    sklearn.set_config(assume_finite=True)
    assert_all_finite(X)
    # 使用 sklearn.set_config 设置 assume_finite=False，再次检查 assert_all_finite 函数
    sklearn.set_config(assume_finite=False)
    # 检查包含无限值的数组 X 是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        assert_all_finite(X)


# test_check_array_series 函数测试 check_array 函数对 pandas Series 的处理
def test_check_array_series():
    # 测试 check_array 函数能否处理 pandas Series，ensure_2d=False 表示不强制转换为二维数组
    pd = importorskip("pandas")
    res = check_array(pd.Series([1, 2, 3]), ensure_2d=False)
    assert_array_equal(res, np.array([1, 2, 3]))

    # 测试具有分类 dtype 的 pandas Series 的处理，GH12699 问题
    s = pd.Series(["a", "b", "c"]).astype("category")
    res = check_array(s, dtype=None, ensure_2d=False)
    assert_array_equal(res, np.array(["a", "b", "c"], dtype=object))


# test_check_dataframe_mixed_float_dtypes 函数测试 check_dataframe 函数对混合浮点类型的处理
@pytest.mark.parametrize(
    "dtype", ((np.float64, np.float32), np.float64, None, "numeric")
)
@pytest.mark.parametrize("bool_dtype", ("bool", "boolean"))
def test_check_dataframe_mixed_float_dtypes(dtype, bool_dtype):
    # 如果 bool_dtype 是 "boolean"，则使用 pandas 1.0 版本引入的布尔扩展数组
    # 否则使用默认的 pandas 库
    if bool_dtype == "boolean":
        pd = importorskip("pandas", minversion="1.0")
    else:
        pd = importorskip("pandas")

    # 创建一个包含整数、浮点数和布尔类型列的 DataFrame
    df = pd.DataFrame(
        {
            "int": [1, 2, 3],
            "float": [0, 0.1, 2.1],
            "bool": pd.Series([True, False, True], dtype=bool_dtype),
        },
        columns=["int", "float", "bool"],
    )

    # 对 DataFrame 进行类型检查，返回一个数组
    array = check_array(df, dtype=dtype)
    # 断言数组的数据类型是 np.float64
    assert array.dtype == np.float64

    # 创建一个期望的浮点数数组，用于断言检查
    expected_array = np.array(
        [[1.0, 0.0, 1.0], [2.0, 0.1, 0.0], [3.0, 2.1, 1.0]], dtype=float
    )
    # 断言生成的数组与期望的数组在数值上近似
    assert_allclose_dense_sparse(array, expected_array)
def test_check_dataframe_with_only_bool():
    """Check that dataframe with bool return a boolean arrays."""
    pd = importorskip("pandas")  # 导入 pandas 库，如果导入失败则跳过测试
    df = pd.DataFrame({"bool": [True, False, True]})  # 创建包含布尔值列的 DataFrame

    array = check_array(df, dtype=None)  # 调用 check_array 函数处理 DataFrame
    assert array.dtype == np.bool_  # 断言处理后的数组数据类型为布尔型
    assert_array_equal(array, [[True], [False], [True]])  # 断言处理后的数组值正确

    # 创建包含布尔和整数列的 DataFrame
    df = pd.DataFrame(
        {"bool": [True, False, True], "int": [1, 2, 3]},
        columns=["bool", "int"],
    )
    array = check_array(df, dtype="numeric")  # 调用 check_array 函数处理 DataFrame
    assert array.dtype == np.int64  # 断言处理后的数组数据类型为 int64
    assert_array_equal(array, [[1, 1], [0, 2], [1, 3]])  # 断言处理后的数组值正确


def test_check_dataframe_with_only_boolean():
    """Check that dataframe with boolean return a float array with dtype=None"""
    pd = importorskip("pandas", minversion="1.0")  # 导入 pandas 库（版本要求至少为 1.0），如果导入失败则跳过测试
    df = pd.DataFrame({"bool": pd.Series([True, False, True], dtype="boolean")})  # 创建包含布尔值列的 DataFrame

    array = check_array(df, dtype=None)  # 调用 check_array 函数处理 DataFrame
    assert array.dtype == np.float64  # 断言处理后的数组数据类型为 float64
    assert_array_equal(array, [[True], [False], [True]])  # 断言处理后的数组值正确


class DummyMemory:
    def cache(self, func):
        return func


class WrongDummyMemory:
    pass


def test_check_memory():
    memory = check_memory("cache_directory")  # 调用 check_memory 函数，传入字符串参数
    assert memory.location == "cache_directory"  # 断言返回的 memory 对象的 location 属性正确为 "cache_directory"

    memory = check_memory(None)  # 调用 check_memory 函数，传入 None 参数
    assert memory.location is None  # 断言返回的 memory 对象的 location 属性正确为 None

    dummy = DummyMemory()
    memory = check_memory(dummy)  # 调用 check_memory 函数，传入 DummyMemory 实例
    assert memory is dummy  # 断言返回的 memory 对象与传入的 dummy 对象是同一个对象的引用

    # 测试传入不支持的类型，期望引发 ValueError 异常
    msg = (
        "'memory' should be None, a string or have the same interface as"
        " joblib.Memory. Got memory='1' instead."
    )
    with pytest.raises(ValueError, match=msg):
        check_memory(1)

    dummy = WrongDummyMemory()
    msg = (
        "'memory' should be None, a string or have the same interface as"
        " joblib.Memory. Got memory='{}' instead.".format(dummy)
    )
    with pytest.raises(ValueError, match=msg):
        check_memory(dummy)


@pytest.mark.parametrize("copy", [True, False])
def test_check_array_memmap(copy):
    X = np.ones((4, 4))
    with TempMemmap(X, mmap_mode="r") as X_memmap:  # 使用临时的内存映射创建数组 X_memmap
        X_checked = check_array(X_memmap, copy=copy)  # 调用 check_array 函数处理内存映射的数组
        assert np.may_share_memory(X_memmap, X_checked) == (not copy)  # 断言处理后的数组是否与原数组共享内存
        assert X_checked.flags["WRITEABLE"] == copy  # 断言处理后的数组的可写标志与预期一致


@pytest.mark.parametrize(
    "retype",
    [
        np.asarray,
        sp.csr_matrix,
        sp.csc_matrix,
        sp.coo_matrix,
        sp.lil_matrix,
        sp.bsr_matrix,
        sp.dok_matrix,
        sp.dia_matrix,
    ],
)
def test_check_non_negative(retype):
    A = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    X = retype(A)  # 使用给定的类型函数将数组 A 转换成特定类型的矩阵 X
    check_non_negative(X, "")  # 调用 check_non_negative 函数检查矩阵 X 中是否存在负数

    X = retype([[0, 0], [0, 0]])  # 使用给定的类型函数将二维列表转换成特定类型的矩阵 X
    check_non_negative(X, "")  # 调用 check_non_negative 函数检查矩阵 X 中是否存在负数

    A[0, 0] = -1
    X = retype(A)  # 使用给定的类型函数将修改后的数组 A 转换成特定类型的矩阵 X
    with pytest.raises(ValueError, match="Negative "):  # 期望引发 ValueError 异常，异常消息包含 "Negative "
        check_non_negative(X, "")


def test_check_X_y_informative_error():
    X = np.ones((2, 2))
    y = None
    msg = "estimator requires y to be passed, but the target y is None"
    # 使用 pytest 模块来测试是否会抛出 ValueError 异常，并检查异常消息是否匹配指定的正则表达式
    with pytest.raises(ValueError, match=msg):
        # 调用 check_X_y 函数，验证给定的 X 和 y 参数
        check_X_y(X, y)
    
    # 设置异常消息的内容，用于测试 RandomForestRegressor 在 y 为 None 时是否会抛出 ValueError 异常
    msg = "RandomForestRegressor requires y to be passed, but the target y is None"
    # 使用 pytest 模块来测试是否会抛出 ValueError 异常，并检查异常消息是否匹配指定的正则表达式
    with pytest.raises(ValueError, match=msg):
        # 调用 check_X_y 函数，验证给定的 X 和 y 参数以及传入的 estimator 参数
        check_X_y(X, y, estimator=RandomForestRegressor())
def test_retrieve_samples_from_non_standard_shape():
    # 定义一个测试类 TestNonNumericShape，模拟一个非数值形状的对象
    class TestNonNumericShape:
        def __init__(self):
            self.shape = ("not numeric",)  # 初始化方法设置一个非数值形状的元组

        def __len__(self):
            return len([1, 2, 3])  # 返回一个固定列表的长度作为模拟的长度值

    X = TestNonNumericShape()  # 创建一个 TestNonNumericShape 的实例 X
    assert _num_samples(X) == len(X)  # 断言调用 _num_samples 函数的结果与 X 的长度相等

    # 检查如果对象没有定义 __len__ 方法，是否能给出合适的错误信息
    class TestNoLenWeirdShape:
        def __init__(self):
            self.shape = ("not numeric",)  # 初始化方法设置一个非数值形状的元组

    # 使用 pytest 断言检查 _num_samples 函数对 TestNoLenWeirdShape 类型的对象会抛出 TypeError，且错误信息包含 "Expected sequence or array-like"
    with pytest.raises(TypeError, match="Expected sequence or array-like"):
        _num_samples(TestNoLenWeirdShape())


@pytest.mark.parametrize("x", [2, 3, 2.5, 5])
def test_check_scalar_valid(x):
    """Test that check_scalar returns no error/warning if valid inputs are
    provided"""
    with warnings.catch_warnings():  # 捕获警告信息
        warnings.simplefilter("error")  # 设置警告处理方式为抛出异常
        # 调用 check_scalar 函数，检查是否对于给定的有效输入不会产生错误或警告
        scalar = check_scalar(
            x,
            "test_name",
            target_type=numbers.Real,
            min_val=2,
            max_val=5,
            include_boundaries="both",
        )
    assert scalar == x  # 断言 scalar 的值与输入的 x 相等


@pytest.mark.parametrize(
    "x, target_name, target_type, min_val, max_val, include_boundaries, err_msg",
    [
        (
            1,
            "test_name1",
            float,
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of float, not int."),
        ),
        (
            None,
            "test_name1",
            numbers.Real,
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of float, not NoneType."),
        ),
        (
            None,
            "test_name1",
            numbers.Integral,
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of int, not NoneType."),
        ),
        (
            1,
            "test_name1",
            (float, bool),
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of {float, bool}, not int."),
        ),
        (
            1,
            "test_name2",
            int,
            2,
            4,
            "neither",
            ValueError("test_name2 == 1, must be > 2."),
        ),
        (
            5,
            "test_name3",
            int,
            2,
            4,
            "neither",
            ValueError("test_name3 == 5, must be < 4."),
        ),
        (
            2,
            "test_name4",
            int,
            2,
            4,
            "right",
            ValueError("test_name4 == 2, must be > 2."),
        ),
        (
            4,
            "test_name5",
            int,
            2,
            4,
            "left",
            ValueError("test_name5 == 4, must be < 4."),
        ),
        (
            4,
            "test_name6",
            int,
            2,
            4,
            "bad parameter value",
            ValueError(
                "Unknown value for `include_boundaries`: 'bad parameter value'. "
                "Possible values are: ('left', 'right', 'both', 'neither')."
            ),
        ),
        (
            4,
            "test_name7",
            int,
            None,
            4,
            "left",
            ValueError(
                "`include_boundaries`='left' without specifying explicitly `min_val` "
                "is inconsistent."
            ),
        ),
        (
            4,
            "test_name8",
            int,
            2,
            None,
            "right",
            ValueError(
                "`include_boundaries`='right' without specifying explicitly `max_val` "
                "is inconsistent."
            ),
        ),
    ]
    
    
    注释：
    
    
    # 列表包含多个元组，每个元组描述了一个测试条件和对应的异常情况
    
    (
        1,                   # 第一个参数: 测试条件的值
        "test_name1",        # 第二个参数: 测试条件的名称
        float,               # 第三个参数: 期望的类型或类型的元组
        2,                   # 第四个参数: 最小值
        4,                   # 第五个参数: 最大值
        "neither",           # 第六个参数: `include_boundaries` 的值
        TypeError("test_name1 must be an instance of float, not int."),  # 异常: 期望的异常信息
    )
    # ... 后续元组的注释结构类似，描述了不同的测试条件和期望的异常情况
# 定义一个测试函数 `test_check_scalar_invalid`，用于验证 `check_scalar` 函数对于错误输入是否能正确引发异常
def test_check_scalar_invalid(
    x, target_name, target_type, min_val, max_val, include_boundaries, err_msg
):
    """Test that check_scalar returns the right error if a wrong input is
    given"""
    # 使用 pytest.raises 来捕获期望的异常，将其存储在 raised_error 中
    with pytest.raises(Exception) as raised_error:
        check_scalar(
            x,
            target_name,
            target_type=target_type,
            min_val=min_val,
            max_val=max_val,
            include_boundaries=include_boundaries,
        )
    # 断言捕获的异常消息字符串与预期的错误消息字符串相同
    assert str(raised_error.value) == str(err_msg)
    # 断言捕获的异常类型与预期的错误消息类型相同
    assert type(raised_error.value) == type(err_msg)


# 定义一组有效的测试用例 `_psd_cases_valid`，用于测试 `_check_psd_eigenvalues` 函数的正常输入
_psd_cases_valid = {
    "nominal": ((1, 2), np.array([1, 2]), None, ""),
    "nominal_np_array": (np.array([1, 2]), np.array([1, 2]), None, ""),
    "insignificant_imag": (
        (5, 5e-5j),
        np.array([5, 0]),
        PositiveSpectrumWarning,
        "There are imaginary parts in eigenvalues \\(1e\\-05 of the maximum real part",
    ),
    "insignificant neg": ((5, -5e-5), np.array([5, 0]), PositiveSpectrumWarning, ""),
    "insignificant neg float32": (
        np.array([1, -1e-6], dtype=np.float32),
        np.array([1, 0], dtype=np.float32),
        PositiveSpectrumWarning,
        "There are negative eigenvalues \\(1e\\-06 of the maximum positive",
    ),
    "insignificant neg float64": (
        np.array([1, -1e-10], dtype=np.float64),
        np.array([1, 0], dtype=np.float64),
        PositiveSpectrumWarning,
        "There are negative eigenvalues \\(1e\\-10 of the maximum positive",
    ),
    "insignificant pos": (
        (5, 4e-12),
        np.array([5, 0]),
        PositiveSpectrumWarning,
        "the largest eigenvalue is more than 1e\\+12 times the smallest",
    ),
}

# 使用 pytest.mark.parametrize 注册 `_psd_cases_valid` 中的测试用例，针对 `test_check_psd_eigenvalues_valid` 函数进行参数化测试
@pytest.mark.parametrize(
    "lambdas, expected_lambdas, w_type, w_msg",
    list(_psd_cases_valid.values()),
    ids=list(_psd_cases_valid.keys()),
)
# 再次使用 pytest.mark.parametrize 对 `enable_warnings` 参数进行参数化，分别测试启用和禁用警告时的行为
@pytest.mark.parametrize("enable_warnings", [True, False])
def test_check_psd_eigenvalues_valid(
    lambdas, expected_lambdas, w_type, w_msg, enable_warnings
):
    # Test that ``_check_psd_eigenvalues`` returns the right output for valid
    # input, possibly raising the right warning

    # 根据 `enable_warnings` 参数的值确定是否期望引发警告
    if not enable_warnings:
        w_type = None

    # 根据 `w_type` 是否为 None，选择不同的上下文管理器来验证函数 `_check_psd_eigenvalues` 的行为
    if w_type is None:
        # 没有警告时，验证函数不应引发任何警告
        with warnings.catch_warnings():
            warnings.simplefilter("error", PositiveSpectrumWarning)
            lambdas_fixed = _check_psd_eigenvalues(
                lambdas, enable_warnings=enable_warnings
            )
    else:
        # 有警告时，验证函数应引发特定类型的警告，并匹配警告消息
        with pytest.warns(w_type, match=w_msg):
            lambdas_fixed = _check_psd_eigenvalues(
                lambdas, enable_warnings=enable_warnings
            )

    # 断言函数的输出与预期输出 `expected_lambdas` 在数值上的近似性
    assert_allclose(expected_lambdas, lambdas_fixed)


# 定义一组无效的测试用例 `_psd_cases_invalid`，用于测试 `_check_psd_eigenvalues` 函数的异常输入
_psd_cases_invalid = {
    "significant_imag": (
        (5, 5j),
        ValueError,
        "There are significant imaginary parts in eigenv",
    ),
    "all negative": (
        (-5, -1),
        ValueError,
        "All eigenvalues are negative \\(maximum is -1",
    ),
    # 键为 "significant neg"，值是一个元组，包含三个元素：
    #   - 第一个元素是包含两个整数的元组 (5, -1)
    #   - 第二个元素是 ValueError 类型的异常对象
    #   - 第三个元素是描述性字符串 "There are significant negative eigenvalues"
    "significant neg": (
        (5, -1),  # 元组中的第一个元素是整数 5，第二个元素是整数 -1
        ValueError,  # 元组中的第二个元素是 ValueError 类型的异常对象
        "There are significant negative eigenvalues",  # 元组中的第三个元素是描述性字符串
    ),

    # 键为 "significant neg float32"，值是一个元组，包含三个元素：
    #   - 第一个元素是一个 numpy 数组，包含两个 float32 类型的浮点数 [3e-4, -2e-6]
    #   - 第二个元素是 ValueError 类型的异常对象
    #   - 第三个元素是描述性字符串 "There are significant negative eigenvalues"
    "significant neg float32": (
        np.array([3e-4, -2e-6], dtype=np.float32),  # 元组中的第一个元素是一个包含两个 float32 类型浮点数的 numpy 数组
        ValueError,  # 元组中的第二个元素是 ValueError 类型的异常对象
        "There are significant negative eigenvalues",  # 元组中的第三个元素是描述性字符串
    ),

    # 键为 "significant neg float64"，值是一个元组，包含三个元素：
    #   - 第一个元素是一个 numpy 数组，包含两个 float64 类型的浮点数 [1e-5, -2e-10]
    #   - 第二个元素是 ValueError 类型的异常对象
    #   - 第三个元素是描述性字符串 "There are significant negative eigenvalues"
    "significant neg float64": (
        np.array([1e-5, -2e-10], dtype=np.float64),  # 元组中的第一个元素是一个包含两个 float64 类型浮点数的 numpy 数组
        ValueError,  # 元组中的第二个元素是 ValueError 类型的异常对象
        "There are significant negative eigenvalues",  # 元组中的第三个元素是描述性字符串
    ),
}

# 使用 pytest 的参数化装饰器，对 _psd_cases_invalid 字典中的值进行参数化测试，
# lambdas 是测试函数的参数，err_type 是期望引发的错误类型，err_msg 是期望的错误消息
@pytest.mark.parametrize(
    "lambdas, err_type, err_msg",
    list(_psd_cases_invalid.values()),
    ids=list(_psd_cases_invalid.keys()),
)
def test_check_psd_eigenvalues_invalid(lambdas, err_type, err_msg):
    # 测试 _check_psd_eigenvalues 在输入无效时是否引发正确的错误

    with pytest.raises(err_type, match=err_msg):
        _check_psd_eigenvalues(lambdas)


def test_check_sample_weight():
    # 检查数组的顺序
    sample_weight = np.ones(10)[::2]
    assert not sample_weight.flags["C_CONTIGUOUS"]
    # 调用 _check_sample_weight 函数，检查返回的 sample_weight 是否是 C 连续的
    sample_weight = _check_sample_weight(sample_weight, X=np.ones((5, 1)))
    assert sample_weight.flags["C_CONTIGUOUS"]

    # 检查 None 输入
    sample_weight = _check_sample_weight(None, X=np.ones((5, 2)))
    assert_allclose(sample_weight, np.ones(5))

    # 检查数字输入
    sample_weight = _check_sample_weight(2.0, X=np.ones((5, 2)))
    assert_allclose(sample_weight, 2 * np.ones(5))

    # 检查错误的维度数量
    with pytest.raises(ValueError, match="Sample weights must be 1D array or scalar"):
        _check_sample_weight(np.ones((2, 4)), X=np.ones((2, 2)))

    # 检查不正确的 n_samples
    msg = r"sample_weight.shape == \(4,\), expected \(2,\)!"
    with pytest.raises(ValueError, match=msg):
        _check_sample_weight(np.ones(4), X=np.ones((2, 2)))

    # float32 类型被保留
    X = np.ones((5, 2))
    sample_weight = np.ones(5, dtype=np.float32)
    sample_weight = _check_sample_weight(sample_weight, X)
    assert sample_weight.dtype == np.float32

    # int 类型将会被转换为 float64
    X = np.ones((5, 2), dtype=int)
    sample_weight = _check_sample_weight(None, X, dtype=X.dtype)
    assert sample_weight.dtype == np.float64

    # 当 only_non_negative=True 时，检查负权重
    X = np.ones((5, 2))
    sample_weight = np.ones(_num_samples(X))
    sample_weight[-1] = -10
    err_msg = "Negative values in data passed to `sample_weight`"
    with pytest.raises(ValueError, match=err_msg):
        _check_sample_weight(sample_weight, X, only_non_negative=True)


@pytest.mark.parametrize("toarray", [np.array, sp.csr_matrix, sp.csc_matrix])
def test_allclose_dense_sparse_equals(toarray):
    base = np.arange(9).reshape(3, 3)
    x, y = toarray(base), toarray(base)
    assert _allclose_dense_sparse(x, y)


@pytest.mark.parametrize("toarray", [np.array, sp.csr_matrix, sp.csc_matrix])
def test_allclose_dense_sparse_not_equals(toarray):
    base = np.arange(9).reshape(3, 3)
    x, y = toarray(base), toarray(base + 1)
    assert not _allclose_dense_sparse(x, y)


@pytest.mark.parametrize("toarray", [sp.csr_matrix, sp.csc_matrix])
def test_allclose_dense_sparse_raise(toarray):
    x = np.arange(9).reshape(3, 3)
    y = toarray(x + 1)

    msg = "Can only compare two sparse matrices, not a sparse matrix and an array"
    with pytest.raises(ValueError, match=msg):
        _allclose_dense_sparse(x, y)


def test_deprecate_positional_args_warns_for_function():
    # 使用装饰器 @_deprecate_positional_args 来标记函数 f1，表明其接受的位置参数在未来版本中会被弃用
    def f1(a, b, *, c=1, d=1):
        pass
    
    # 使用 pytest.warns 来检测 FutureWarning 警告，确保在调用 f1 函数时传递 c=3 作为关键字参数
    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        f1(1, 2, 3)
    
    # 同样使用 pytest.warns 检测 FutureWarning 警告，确保在调用 f1 函数时传递 c=3, d=4 作为关键字参数
    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        f1(1, 2, 3, 4)
    
    # 使用装饰器 @_deprecate_positional_args 来标记函数 f2，表明其接受的位置参数在未来版本中会被弃用
    def f2(a=1, *, b=1, c=1, d=1):
        pass
    
    # 使用 pytest.warns 来检测 FutureWarning 警告，确保在调用 f2 函数时传递 b=2 作为关键字参数
    with pytest.warns(FutureWarning, match=r"Pass b=2 as keyword args"):
        f2(1, 2)
    
    # 使用装饰器 @_deprecate_positional_args 来标记函数 f3，表明其接受的位置参数在未来版本中会被弃用
    # 函数 f3 的第一个参数 a 是一个位置参数，后面的 b 是一个只能通过关键字传递的参数，其余的 c 和 d 是默认参数
    def f3(a, *, b, c=1, d=1):
        pass
    
    # 使用 pytest.warns 来检测 FutureWarning 警告，确保在调用 f3 函数时传递 b=2 作为关键字参数
    with pytest.warns(FutureWarning, match=r"Pass b=2 as keyword args"):
        f3(1, 2)
def test_deprecate_positional_args_warns_for_function_version():
    # 定义一个测试函数，测试函数装饰器_deprecate_positional_args对函数版本进行警告
    @_deprecate_positional_args(version="1.1")
    def f1(a, *, b):
        pass

    # 使用pytest的warns断言检测是否触发FutureWarning，匹配特定的警告信息
    with pytest.warns(
        FutureWarning, match=r"From version 1.1 passing these as positional"
    ):
        # 调用带有过时参数的函数，预期触发警告
        f1(1, 2)


def test_deprecate_positional_args_warns_for_class():
    # 定义一个测试类A1
    class A1:
        # 类的初始化方法，使用_deprecate_positional_args装饰器
        @_deprecate_positional_args
        def __init__(self, a, b, *, c=1, d=1):
            pass

    # 使用pytest的warns断言检测是否触发FutureWarning，匹配特定的警告信息
    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        # 实例化类A1，传递额外的位置参数，预期触发警告
        A1(1, 2, 3)

    # 使用pytest的warns断言检测是否触发FutureWarning，匹配特定的警告信息
    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        # 实例化类A1，传递额外的位置参数，预期触发警告
        A1(1, 2, 3, 4)

    # 定义另一个测试类A2
    class A2:
        # 类的初始化方法，使用_deprecate_positional_args装饰器
        @_deprecate_positional_args
        def __init__(self, a=1, b=1, *, c=1, d=1):
            pass

    # 使用pytest的warns断言检测是否触发FutureWarning，匹配特定的警告信息
    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        # 实例化类A2，传递额外的位置参数，预期触发警告
        A2(1, 2, 3)

    # 使用pytest的warns断言检测是否触发FutureWarning，匹配特定的警告信息
    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        # 实例化类A2，传递额外的位置参数，预期触发警告
        A2(1, 2, 3, 4)


@pytest.mark.parametrize("indices", [None, [1, 3]])
def test_check_method_params(indices):
    # 生成一个4x2的随机数组X
    X = np.random.randn(4, 2)
    # 定义一个参数字典_params，包含多种数据类型和None值
    _params = {
        "list": [1, 2, 3, 4],
        "array": np.array([1, 2, 3, 4]),
        "sparse-col": sp.csc_matrix([1, 2, 3, 4]).T,
        "sparse-row": sp.csc_matrix([1, 2, 3, 4]),
        "scalar-int": 1,
        "scalar-str": "xxx",
        "None": None,
    }
    # 调用_check_method_params函数，传递参数X和_params，以及indices参数
    result = _check_method_params(X, params=_params, indices=indices)
    # 根据indices是否为None确定indices_的值
    indices_ = indices if indices is not None else list(range(X.shape[0]))

    # 遍历特定的键进行断言，确保结果中的值和_params中的对应值一致
    for key in ["sparse-row", "scalar-int", "scalar-str", "None"]:
        assert result[key] is _params[key]

    # 断言结果中的"list"键对应的值与_safe_indexing函数处理后的_params["list"]一致
    assert result["list"] == _safe_indexing(_params["list"], indices_)
    # 断言结果中的"array"键对应的值与_safe_indexing函数处理后的_params["array"]一致
    assert_array_equal(result["array"], _safe_indexing(_params["array"], indices_))
    # 断言结果中的"sparse-col"键对应的值与_safe_indexing函数处理后的_params["sparse-col"]一致
    assert_allclose_dense_sparse(
        result["sparse-col"], _safe_indexing(_params["sparse-col"], indices_)
    )


@pytest.mark.parametrize("sp_format", [True, "csr", "csc", "coo", "bsr"])
def test_check_sparse_pandas_sp_format(sp_format):
    # 导入pytest模块中的pandas对象
    pd = pytest.importorskip("pandas")
    # 生成一个大小为10x3的稀疏随机矩阵
    sp_mat = _sparse_random_matrix(10, 3)

    # 使用稀疏矩阵生成一个稀疏DataFrame对象sdf
    sdf = pd.DataFrame.sparse.from_spmatrix(sp_mat)
    # 调用check_array函数，检查对sdf的处理结果
    result = check_array(sdf, accept_sparse=sp_format)

    # 如果sp_format为True，则pandas默认转换为coo格式
    if sp_format is True:
        sp_format = "coo"

    # 断言结果是稀疏矩阵类型
    assert sp.issparse(result)
    # 断言结果的格式与sp_format匹配
    assert result.format == sp_format
    # 断言结果与原始稀疏矩阵sp_mat在密集形式下的近似一致性
    assert_allclose_dense_sparse(sp_mat, result)


@pytest.mark.parametrize(
    "ntype1, ntype2",
    [
        ("longdouble", "float16"),
        ("float16", "float32"),
        ("float32", "double"),
        ("int16", "int32"),
        ("int32", "long"),
        ("byte", "uint16"),
        ("ushort", "uint32"),
        ("uint32", "uint64"),
        ("uint8", "int8"),
    ],
)
def test_check_pandas_sparse_invalid(ntype1, ntype2):
    """检查在数据框架中使用不支持的数据类型时是否引发错误"""
    # 这里省略部分代码，用于测试DataFrame包含的情况
    # 导入 pytest 库，并要求至少能导入 pandas 库，否则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建一个 Pandas DataFrame 对象，包含两列稀疏扩展数组
    df = pd.DataFrame(
        {
            # 第一列稀疏数组，使用指定的数据类型 ntype1 和填充值 0
            "col1": pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0),
            # 第二列稀疏数组，使用指定的数据类型 ntype2 和填充值 0
            "col2": pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0),
        }
    )

    # 如果 pandas 版本低于 1.1，则检查抛出错误
    if parse_version(pd.__version__) < parse_version("1.1"):
        err_msg = "Pandas DataFrame with mixed sparse extension arrays"
        # 使用 pytest 检查是否会抛出值错误，并匹配特定的错误消息
        with pytest.raises(ValueError, match=err_msg):
            check_array(df, accept_sparse=["csr", "csc"])
    else:
        # 从 pandas 1.1 开始，已经修复了此问题，因此不会再抛出错误
        check_array(df, accept_sparse=["csr", "csc"])
@pytest.mark.parametrize(
    "ntype1, ntype2, expected_subtype",
    [
        ("double", "longdouble", np.floating),  # 定义不同数据类型对应的预期子类型
        ("single", "float32", np.floating),
        ("double", "float64", np.floating),
        ("int8", "byte", np.integer),
        ("short", "int16", np.integer),
        ("intc", "int32", np.integer),
        ("intp", "long", np.integer),
        ("int", "long", np.integer),
        ("int64", "longlong", np.integer),
        ("int_", "intp", np.integer),
        ("ubyte", "uint8", np.unsignedinteger),
        ("uint16", "ushort", np.unsignedinteger),
        ("uintc", "uint32", np.unsignedinteger),
        ("uint", "uint64", np.unsignedinteger),
        ("uintp", "ulonglong", np.unsignedinteger),
    ],
)
def test_check_pandas_sparse_valid(ntype1, ntype2, expected_subtype):
    """测试确保支持稀疏DataFrame的混合类型转换安全进行。

    使用pytest.importorskip确保导入pandas模块成功。
    创建包含稀疏数据的DataFrame，并调用check_array函数进行转换。
    最后断言转换后的数组的dtype是否符合预期的子类型。
    """
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "col1": pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0),
            "col2": pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0),
        }
    )
    arr = check_array(df, accept_sparse=["csr", "csc"])
    assert np.issubdtype(arr.dtype, expected_subtype)


@pytest.mark.parametrize(
    "constructor_name",
    ["list", "tuple", "array", "dataframe", "sparse_csr", "sparse_csc"],
)
def test_num_features(constructor_name):
    """检查对于类数组的_num_features函数。

    创建一个简单的二维数组X，并使用_convert_container函数将其转换为指定类型。
    确保转换后的对象的特征数量为3。
    """
    X = [[1, 2, 3], [4, 5, 6]]
    X = _convert_container(X, constructor_name)
    assert _num_features(X) == 3


@pytest.mark.parametrize(
    "X",
    [
        [1, 2, 3],
        ["a", "b", "c"],
        [False, True, False],
        [1.0, 3.4, 4.0],
        [{"a": 1}, {"b": 2}, {"c": 3}],
    ],
    ids=["int", "str", "bool", "float", "dict"],
)
@pytest.mark.parametrize("constructor_name", ["list", "tuple", "array", "series"])
def test_num_features_errors_1d_containers(X, constructor_name):
    """测试针对一维容器的_num_features函数的错误情况。

    使用_convert_container函数将输入X转换为指定类型的数据结构。
    根据不同的constructor_name，定义期望的错误消息内容。
    使用pytest.raises确保调用_num_features时会抛出TypeError，并匹配预期的错误消息。
    """
    X = _convert_container(X, constructor_name)
    if constructor_name == "array":
        expected_type_name = "numpy.ndarray"
    elif constructor_name == "series":
        expected_type_name = "pandas.core.series.Series"
    else:
        expected_type_name = constructor_name
    message = (
        f"Unable to find the number of features from X of type {expected_type_name}"
    )
    if hasattr(X, "shape"):
        message += " with shape (3,)"
    elif isinstance(X[0], str):
        message += " where the samples are of type str"
    elif isinstance(X[0], dict):
        message += " where the samples are of type dict"
    with pytest.raises(TypeError, match=re.escape(message)):
        _num_features(X)


@pytest.mark.parametrize("X", [1, "b", False, 3.0], ids=["int", "str", "bool", "float"])
def test_num_features_errors_scalars(X):
    """测试针对标量输入的_num_features函数的错误情况。

    根据X的类型定义期望的错误消息内容。
    使用pytest.raises确保调用_num_features时会抛出TypeError，并匹配预期的错误消息。
    """
    msg = f"Unable to find the number of features from X of type {type(X).__qualname__}"
    with pytest.raises(TypeError, match=re.escape(msg)):
        _num_features(X)
    # 使用 pytest 中的断言来测试 _num_features 函数是否会引发 TypeError，并检查异常消息是否匹配给定的 msg
    with pytest.raises(TypeError, match=msg):
        # 调用 _num_features 函数，期望它引发 TypeError 异常，并验证异常消息是否匹配 msg
        _num_features(X)
@pytest.mark.parametrize(
    "names",
    [list(range(2)), range(2), None, [["a", "b"], ["c", "d"]]],
    ids=["list-int", "range", "default", "MultiIndex"],
)
def test_get_feature_names_pandas_with_ints_no_warning(names):
    """Get feature names with pandas dataframes without warning.

    Column names with consistent dtypes will not warn, such as int or MultiIndex.
    """
    # 导入 pytest 库，并跳过如果没有安装 pandas 的话
    pd = pytest.importorskip("pandas")
    # 创建一个 pandas DataFrame，使用给定的列名 names
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)

    # 使用 warnings 模块捕获警告
    with warnings.catch_warnings():
        # 设置当 FutureWarning 发生时抛出异常
        warnings.simplefilter("error", FutureWarning)
        # 调用 _get_feature_names 函数获取特征名
        names = _get_feature_names(X)
    # 断言特征名为 None
    assert names is None


def test_get_feature_names_pandas():
    """Get feature names with pandas dataframes."""
    # 导入 pytest 库，并跳过如果没有安装 pandas 的话
    pd = pytest.importorskip("pandas")
    # 创建一个 pandas DataFrame，包含三列，列名为 col_0, col_1, col_2
    columns = [f"col_{i}" for i in range(3)]
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)
    # 调用 _get_feature_names 函数获取特征名
    feature_names = _get_feature_names(X)

    # 断言特征名数组与列名数组相等
    assert_array_equal(feature_names, columns)


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [("pyarrow", "12.0.0"), ("dataframe", "1.5.0"), ("polars", "0.18.2")],
)
def test_get_feature_names_dataframe_protocol(constructor_name, minversion):
    """Uses the dataframe exchange protocol to get feature names."""
    # 定义数据和列名
    data = [[1, 4, 2], [3, 3, 6]]
    columns = ["col_0", "col_1", "col_2"]
    # 使用 _convert_container 函数将数据转换为特定类型的数据结构
    df = _convert_container(
        data, constructor_name, columns_name=columns, minversion=minversion
    )
    # 调用 _get_feature_names 函数获取特征名
    feature_names = _get_feature_names(df)

    # 断言特征名数组与列名数组相等
    assert_array_equal(feature_names, columns)


@pytest.mark.parametrize("constructor_name", ["pyarrow", "dataframe", "polars"])
def test_is_pandas_df_other_libraries(constructor_name):
    # 使用 _convert_container 函数将数据转换为特定类型的数据结构
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    # 根据 constructor_name 判断是否为 pandas DataFrame
    if constructor_name in ("pyarrow", "polars"):
        assert not _is_pandas_df(df)
    else:
        assert _is_pandas_df(df)


def test_is_pandas_df():
    """Check behavior of is_pandas_df when pandas is installed."""
    # 导入 pytest 库，并跳过如果没有安装 pandas 的话
    pd = pytest.importorskip("pandas")
    # 创建一个 pandas DataFrame
    df = pd.DataFrame([[1, 2, 3]])
    # 断言是否为 pandas DataFrame
    assert _is_pandas_df(df)
    # 断言不是 pandas DataFrame
    assert not _is_pandas_df(np.asarray([1, 2, 3]))
    # 断言不是 pandas DataFrame
    assert not _is_pandas_df(1)


def test_is_pandas_df_pandas_not_installed(hide_available_pandas):
    """Check _is_pandas_df when pandas is not installed."""
    # 断言不是 pandas DataFrame
    assert not _is_pandas_df(np.asarray([1, 2, 3]))
    # 断言不是 pandas DataFrame
    assert not _is_pandas_df(1)


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        ("pyarrow", dependent_packages["pyarrow"][0]),
        ("dataframe", dependent_packages["pandas"][0]),
        ("polars", dependent_packages["polars"][0]),
    ],
)
def test_is_polars_df_other_libraries(constructor_name, minversion):
    # 使用 _convert_container 函数将数据转换为特定类型的数据结构
    df = _convert_container(
        [[1, 4, 2], [3, 3, 6]],
        constructor_name,
        minversion=minversion,
    )
    # 根据 constructor_name 判断是否为 polars DataFrame
    if constructor_name in ("pyarrow", "dataframe"):
        assert not _is_polars_df(df)
    else:
        assert _is_polars_df(df)


def test_is_polars_df_for_duck_typed_polars_dataframe():
    # 这个测试函数尚未实现，因此没有内容
    pass
    """Check _is_polars_df for object that looks like a polars dataframe"""

    # 定义一个名为 NotAPolarsDataFrame 的类，用于模拟不是 Polars DataFrame 的对象
    class NotAPolarsDataFrame:
        def __init__(self):
            # 初始化对象属性 columns，设置为一个包含整数的列表
            self.columns = [1, 2, 3]
            # 初始化对象属性 schema，设置为字符串 "my_schema"
            self.schema = "my_schema"

    # 创建一个 NotAPolarsDataFrame 的实例
    not_a_polars_df = NotAPolarsDataFrame()
    # 使用断言验证 _is_polars_df 函数对于 not_a_polars_df 返回 False
    assert not _is_polars_df(not_a_polars_df)
def test_get_feature_names_numpy():
    """Get feature names return None for numpy arrays."""
    # 创建一个包含两行三列的 NumPy 数组
    X = np.array([[1, 2, 3], [4, 5, 6]])
    # 调用 _get_feature_names 函数获取特征名
    names = _get_feature_names(X)
    # 断言特征名为 None
    assert names is None


@pytest.mark.parametrize(
    "names, dtypes",
    [
        (["a", 1], "['int', 'str']"),
        (["pizza", ["a", "b"]], "['list', 'str']"),
    ],
    ids=["int-str", "list-str"],
)
def test_get_feature_names_invalid_dtypes(names, dtypes):
    """Get feature names errors when the feature names have mixed dtypes"""
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个包含三行两列的 DataFrame，列名为 names 中的值
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)

    # 构建错误消息的正则表达式，用于匹配特定格式的错误信息
    msg = re.escape(
        "Feature names are only supported if all input features have string names, "
        f"but your input has {dtypes} as feature name / column name types. "
        "If you want feature names to be stored and validated, you must convert "
        "them all to strings, by using X.columns = X.columns.astype(str) for "
        "example. Otherwise you can remove feature / column names from your input "
        "data, or convert them all to a non-string data type."
    )
    # 断言调用 _get_feature_names 函数会抛出 TypeError，并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        names = _get_feature_names(X)


class PassthroughTransformer(BaseEstimator):
    def fit(self, X, y=None):
        # 使用 _validate_data 方法验证输入数据 X
        self._validate_data(X, reset=True)
        return self

    def transform(self, X):
        # 返回输入数据 X，即为"透传"转换器
        return X

    def get_feature_names_out(self, input_features=None):
        # 调用 _check_feature_names_in 函数处理输入特征名 input_features
        return _check_feature_names_in(self, input_features)


def test_check_feature_names_in():
    """Check behavior of check_feature_names_in for arrays."""
    # 创建一个包含一行三列的 NumPy 数组
    X = np.array([[0.0, 1.0, 2.0]])
    # 创建 PassthroughTransformer 实例并拟合数据 X
    est = PassthroughTransformer().fit(X)

    # 获取转换后的特征名
    names = est.get_feature_names_out()
    # 断言特征名列表与预期的列表相等
    assert_array_equal(names, ["x0", "x1", "x2"])

    # 准备一个长度不匹配的特征名列表
    incorrect_len_names = ["x10", "x1"]
    # 断言调用 get_feature_names_out 时会抛出 ValueError，并匹配预期的错误消息
    with pytest.raises(ValueError, match="input_features should have length equal to"):
        est.get_feature_names_out(incorrect_len_names)

    # 删除 n_feature_in_ 属性后，再次调用 get_feature_names_out 函数
    del est.n_features_in_
    # 断言调用 get_feature_names_out 时会抛出 ValueError，并匹配预期的错误消息
    with pytest.raises(ValueError, match="Unable to generate feature names"):
        est.get_feature_names_out()


def test_check_feature_names_in_pandas():
    """Check behavior of check_feature_names_in for pandas dataframes."""
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个包含一行三列的 DataFrame，列名为 ["a", "b", "c"]
    names = ["a", "b", "c"]
    df = pd.DataFrame([[0.0, 1.0, 2.0]], columns=names)
    # 创建 PassthroughTransformer 实例并拟合数据 df
    est = PassthroughTransformer().fit(df)

    # 获取转换后的特征名
    names = est.get_feature_names_out()
    # 断言特征名列表与预期的列表相等
    assert_array_equal(names, ["a", "b", "c"])

    # 准备一个不匹配的特征名列表
    with pytest.raises(ValueError, match="input_features is not equal to"):
        est.get_feature_names_out(["x1", "x2", "x3"])


def test_check_response_method_unknown_method():
    """Check the error message when passing an unknown response method."""
    # 定义错误消息字符串
    err_msg = (
        "RandomForestRegressor has none of the following attributes: unknown_method."
    )
    # 断言调用 _check_response_method 函数会抛出 AttributeError，并匹配预期的错误消息
    with pytest.raises(AttributeError, match=err_msg):
        _check_response_method(RandomForestRegressor(), "unknown_method")
# 使用 pytest 的 parametrize 装饰器，定义一个参数化测试函数，测试不支持的响应方法
@pytest.mark.parametrize(
    "response_method", ["decision_function", "predict_proba", "predict"]
)
def test_check_response_method_not_supported_response_method(response_method):
    """Check the error message when a response method is not supported by the
    estimator."""
    # 构造错误消息，指出估计器缺少指定的响应方法
    err_msg = (
        f"EstimatorWithFit has none of the following attributes: {response_method}."
    )
    # 断言调用 _check_response_method 函数时抛出 AttributeError 异常，并匹配错误消息
    with pytest.raises(AttributeError, match=err_msg):
        _check_response_method(EstimatorWithFit(), response_method)


# 测试检查 _check_response_method 函数能够接受方法名称列表
def test_check_response_method_list_str():
    """Check that we can pass a list of ordered method."""
    # 定义一个模拟估计器，该估计器只支持 predict_proba 方法
    method_implemented = ["predict_proba"]
    my_estimator = _MockEstimatorOnOffPrediction(method_implemented)

    X = "mocking_data"

    # 当估计器不支持定义的方法时，应该抛出 AttributeError 异常
    response_method = ["decision_function", "predict"]
    err_msg = (
        "_MockEstimatorOnOffPrediction has none of the following attributes: "
        f"{', '.join(response_method)}."
    )
    with pytest.raises(AttributeError, match=err_msg):
        _check_response_method(my_estimator, response_method)(X)

    # 检查当估计器支持其中一个方法时不会出现问题
    response_method = ["decision_function", "predict_proba"]
    # 调用 _check_response_method 函数并断言返回的方法名为 predict_proba
    method_name_predicting = _check_response_method(my_estimator, response_method)(X)
    assert method_name_predicting == "predict_proba"

    # 检查返回的方法顺序是否符合预期
    method_implemented = ["predict_proba", "predict"]
    my_estimator = _MockEstimatorOnOffPrediction(method_implemented)
    response_method = ["decision_function", "predict", "predict_proba"]
    # 调用 _check_response_method 函数并断言返回的方法名为 predict
    method_name_predicting = _check_response_method(my_estimator, response_method)(X)
    assert method_name_predicting == "predict"


# 回归测试，检查 pandas Series 的布尔值保持不变
def test_boolean_series_remains_boolean():
    """Regression test for gh-25145"""
    pd = importorskip("pandas")
    # 调用 check_array 函数，检查是否能正确处理布尔值 Series
    res = check_array(pd.Series([True, False]), ensure_2d=False)
    expected = np.array([True, False])

    # 断言结果的数据类型和值与期望相符
    assert res.dtype == expected.dtype
    assert_array_equal(res, expected)


# 参数化测试函数，测试 pandas 数组返回 ndarray
@pytest.mark.parametrize("input_values", [[0, 1, 0, 1, 0, np.nan], [0, 1, 0, 1, 0, 1]])
def test_pandas_array_returns_ndarray(input_values):
    """Check pandas array with extensions dtypes returns a numeric ndarray.

    Non-regression test for gh-25637.
    """
    pd = importorskip("pandas")
    # 使用指定的扩展 dtype 创建 pandas array
    input_series = pd.array(input_values, dtype="Int32")
    # 调用 check_array 函数，并断言返回结果的数据类型是浮点数
    result = check_array(
        input_series,
        dtype=None,
        ensure_2d=False,
        allow_nd=False,
        force_all_finite=False,
    )
    assert np.issubdtype(result.dtype.kind, np.floating)
    # 断言返回的结果与输入值相近
    assert_allclose(result, input_values)


# 如果数组 API 兼容未配置，则跳过测试
@skip_if_array_api_compat_not_configured
# 参数化测试函数，检查数组 API 数组是否正确处理非有限值
@pytest.mark.parametrize("array_namespace", ["array_api_strict", "cupy.array_api"])
def test_check_array_array_api_has_non_finite(array_namespace):
    """Checks that Array API arrays checks non-finite correctly."""
    # 导入指定的数组 API
    xp = pytest.importorskip(array_namespace)
    # 创建一个包含 NaN 值的二维数组，数据类型为 float32
    X_nan = xp.asarray([[xp.nan, 1, 0], [0, xp.nan, 3]], dtype=xp.float32)
    # 使用 array_api_dispatch 配置上下文，确保在 XP 数组 API 中调度
    with config_context(array_api_dispatch=True):
        # 使用 pytest 检查是否引发 ValueError 异常，并匹配指定的错误信息字符串
        with pytest.raises(ValueError, match="Input contains NaN."):
            # 调用 check_array 函数检查 X_nan 数组，预期会引发包含 "Input contains NaN." 的 ValueError 异常
            check_array(X_nan)

    # 创建一个包含 inf 值的二维数组，数据类型为 float32
    X_inf = xp.asarray([[xp.inf, 1, 0], [0, xp.inf, 3]], dtype=xp.float32)
    # 使用 array_api_dispatch 配置上下文，确保在 XP 数组 API 中调度
    with config_context(array_api_dispatch=True):
        # 使用 pytest 检查是否引发 ValueError 异常，并匹配指定的错误信息字符串
        with pytest.raises(ValueError, match="infinity or a value too large"):
            # 调用 check_array 函数检查 X_inf 数组，预期会引发包含 "infinity or a value too large" 的 ValueError 异常
            check_array(X_inf)
@pytest.mark.parametrize(
    "extension_dtype, regular_dtype",
    [
        ("boolean", "bool"),
        ("Int64", "int64"),
        ("Float64", "float64"),
        ("category", "object"),
    ],
)
@pytest.mark.parametrize("include_object", [True, False])
def test_check_array_multiple_extensions(
    extension_dtype, regular_dtype, include_object
):
    """Check pandas extension arrays give the same result as non-extension arrays."""
    pd = pytest.importorskip("pandas")
    # 创建一个普通的 Pandas DataFrame，包含不同数据类型的列
    X_regular = pd.DataFrame(
        {
            "a": pd.Series([1, 0, 1, 0], dtype=regular_dtype),
            "c": pd.Series([9, 8, 7, 6], dtype="int64"),
        }
    )
    # 如果 include_object 为 True，则添加一个 object 类型的列
    if include_object:
        X_regular["b"] = pd.Series(["a", "b", "c", "d"], dtype="object")

    # 创建扩展后的 DataFrame，将列 'a' 的数据类型转换为 extension_dtype
    X_extension = X_regular.assign(a=X_regular["a"].astype(extension_dtype))

    # 对普通和扩展后的 DataFrame 进行数组检查
    X_regular_checked = check_array(X_regular, dtype=None)
    X_extension_checked = check_array(X_extension, dtype=None)
    # 断言两个数组检查结果应该一致
    assert_array_equal(X_regular_checked, X_extension_checked)


def test_num_samples_dataframe_protocol():
    """Use the DataFrame interchange protocol to get n_samples from polars."""
    pl = pytest.importorskip("polars")

    # 创建一个 Polars DataFrame
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 断言从 DataFrame 中获取样本数应该为 3
    assert _num_samples(df) == 3


@pytest.mark.parametrize(
    "sparse_container",
    CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + DIA_CONTAINERS,
)
@pytest.mark.parametrize("output_format", ["csr", "csc", "coo"])
def test_check_array_dia_to_int32_indexed_csr_csc_coo(sparse_container, output_format):
    """Check the consistency of the indices dtype with sparse matrices/arrays."""
    # 使用 sparse_container 创建一个稀疏矩阵 X
    X = sparse_container([[0, 1], [1, 0]], dtype=np.float64)

    # 显式设置索引数组的 dtype
    if hasattr(X, "offsets"):  # DIA 矩阵
        X.offsets = X.offsets.astype(np.int32)
    elif hasattr(X, "row") and hasattr(X, "col"):  # COO 矩阵
        X.row = X.row.astype(np.int32)
    elif hasattr(X, "indices") and hasattr(X, "indptr"):  # CSR 或 CSC 矩阵
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int32)

    # 对稀疏矩阵 X 进行数组检查，接受指定的稀疏格式 output_format
    X_checked = check_array(X, accept_sparse=output_format)
    if output_format == "coo":
        # 断言 COO 格式下的行和列索引 dtype 应该为 np.int32
        assert X_checked.row.dtype == np.int32
        assert X_checked.col.dtype == np.int32
    else:  # output_format in ["csr", "csc"]
        # 断言 CSR 或 CSC 格式下的索引和指针 dtype 应该为 np.int32
        assert X_checked.indices.dtype == np.int32
        assert X_checked.indptr.dtype == np.int32


@pytest.mark.parametrize("sequence", [[np.array(1), np.array(2)], [[1, 2], [3, 4]]])
def test_to_object_array(sequence):
    # 转换输入序列为对象类型的 NumPy 数组
    out = _to_object_array(sequence)
    # 断言输出是一个 NumPy 数组，并且 dtype 是对象类型
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "O"
    # 断言输出数组的维度为 1
    assert out.ndim == 1


def test_column_or_1d():
    # 这个测试函数尚未完整提供
    pass
    EXAMPLES = [
        ("binary", ["spam", "egg", "spam"]),  # 示例列表，包含二元分类的字符串数据
        ("binary", [0, 1, 0, 1]),             # 示例列表，包含二元分类的整数数据
        ("continuous", np.arange(10) / 20.0), # 示例列表，包含连续型数据
        ("multiclass", [1, 2, 3]),             # 示例列表，包含多类分类的整数数据
        ("multiclass", [0, 1, 2, 2, 0]),       # 示例列表，包含多类分类的整数数据
        ("multiclass", [[1], [2], [3]]),       # 示例列表，包含多类分类的嵌套列表数据
        ("multilabel-indicator", [[0, 1, 0], [0, 0, 1]]),  # 示例列表，包含多标签指示器的二维列表数据
        ("multiclass-multioutput", [[1, 2, 3]]),  # 示例列表，包含多类多输出的二维列表数据
        ("multiclass-multioutput", [[1, 1], [2, 2], [3, 1]]),  # 示例列表，包含多类多输出的二维列表数据
        ("multiclass-multioutput", [[5, 1], [4, 2], [3, 1]]),  # 示例列表，包含多类多输出的二维列表数据
        ("multiclass-multioutput", [[1, 2, 3]]),  # 示例列表，包含多类多输出的二维列表数据
        ("continuous-multioutput", np.arange(30).reshape((-1, 3))),  # 示例列表，包含连续型多输出的二维 NumPy 数组数据
    ]

    for y_type, y in EXAMPLES:
        if y_type in ["binary", "multiclass", "continuous"]:
            # 如果示例类型为二元、多类分类或连续型，则验证 y 是一维数组或可以展平为一维的数组
            assert_array_equal(column_or_1d(y), np.ravel(y))
        else:
            # 如果示例类型不在上述三种类型中，预期会抛出 ValueError 异常
            with pytest.raises(ValueError):
                column_or_1d(y)
# 定义一个测试函数，用于检查 _is_polars_df 对非数据框对象返回 False 的行为
def test__is_polars_df():
    """Check that _is_polars_df return False for non-dataframe objects."""

    # 定义一个类 LooksLikePolars，模拟数据框的结构
    class LooksLikePolars:
        def __init__(self):
            self.columns = ["a", "b"]
            self.schema = ["a", "b"]

    # 断言调用 _is_polars_df 函数返回 False
    assert not _is_polars_df(LooksLikePolars())


# 定义一个测试函数，检查在请求不进行复制的情况下，check_array 在 numpy 数组上的行为
def test_check_array_writeable_np():
    """Check the behavior of check_array when a writeable array is requested
    without copy if possible, on numpy arrays.
    """
    # 生成一个大小为 (10, 10) 的随机 numpy 数组
    X = np.random.uniform(size=(10, 10))

    # 调用 check_array 函数，要求不进行复制并且要求可写
    out = check_array(X, copy=False, force_writeable=True)
    # 断言 out 与 X 可能共享内存
    assert np.may_share_memory(out, X)
    # 断言 out 可写
    assert out.flags.writeable

    # 设置 X 不可写
    X.flags.writeable = False

    # 再次调用 check_array 函数，要求不进行复制并且要求可写
    out = check_array(X, copy=False, force_writeable=True)
    # 断言 out 与 X 不共享内存
    assert not np.may_share_memory(out, X)
    # 断言 out 可写


# 定义一个测试函数，检查在请求不进行复制的情况下，check_array 在内存映射上的行为
def test_check_array_writeable_mmap():
    """Check the behavior of check_array when a writeable array is requested
    without copy if possible, on a memory-map.

    A common situation is when a meta-estimators run in parallel using multiprocessing
    with joblib, which creates read-only memory-maps of large arrays.
    """
    # 生成一个大小为 (10, 10) 的随机 numpy 数组
    X = np.random.uniform(size=(10, 10))

    # 创建一个写入模式的内存映射
    mmap = create_memmap_backed_data(X, mmap_mode="w+")
    # 调用 check_array 函数，要求不进行复制并且要求可写
    out = check_array(mmap, copy=False, force_writeable=True)
    # 断言 out 与 mmap 可能共享内存
    assert np.may_share_memory(out, mmap)
    # 断言 out 可写

    # 创建一个只读模式的内存映射
    mmap = create_memmap_backed_data(X, mmap_mode="r")
    # 再次调用 check_array 函数，要求不进行复制并且要求可写
    out = check_array(mmap, copy=False, force_writeable=True)
    # 断言 out 与 mmap 不共享内存
    assert not np.may_share_memory(out, mmap)
    # 断言 out 可写


# 定义一个测试函数，检查在请求不进行复制的情况下，check_array 在数据框上的行为
def test_check_array_writeable_df():
    """Check the behavior of check_array when a writeable array is requested
    without copy if possible, on a dataframe.
    """
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 生成一个大小为 (10, 10) 的随机 numpy 数组
    X = np.random.uniform(size=(10, 10))
    # 使用 numpy 数组创建一个数据框，要求不进行复制
    df = pd.DataFrame(X, copy=False)

    # 调用 check_array 函数，要求不进行复制并且要求可写
    out = check_array(df, copy=False, force_writeable=True)
    # 断言 out 与 df 可能共享内存
    assert np.may_share_memory(out, df)
    # 断言 out 可写

    # 设置 X 不可写，并再次使用 X 创建一个数据框
    X.flags.writeable = False
    df = pd.DataFrame(X, copy=False)

    # 再次调用 check_array 函数，要求不进行复制并且要求可写
    out = check_array(df, copy=False, force_writeable=True)
    # 断言 out 与 df 不共享内存
    assert not np.may_share_memory(out, df)
    # 断言 out 可写
```