# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_indexing.py`

```
# 导入警告模块，用于显示警告信息
import warnings
# 从标准库中导入深拷贝函数 copy
from copy import copy
# 从单元测试模块中导入 SkipTest 异常类
from unittest import SkipTest

# 导入第三方库 numpy，并重命名为 np
import numpy as np
# 导入 pytest 测试框架
import pytest

# 导入 scikit-learn 库
import sklearn
# 从 scikit-learn 库的 externals._packaging.version 模块中导入 parse 函数，并重命名为 parse_version
from sklearn.externals._packaging.version import parse as parse_version
# 从 scikit-learn 的 utils 模块中导入 _safe_indexing、resample、shuffle 函数
from sklearn.utils import _safe_indexing, resample, shuffle
# 从 scikit-learn 的 utils._array_api 模块中导入 yield_namespace_device_dtype_combinations 函数
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
# 从 scikit-learn 的 utils._indexing 模块中导入 _determine_key_type、_get_column_indices、_safe_assign 函数
from sklearn.utils._indexing import (
    _determine_key_type,
    _get_column_indices,
    _safe_assign,
)
# 从 scikit-learn 的 utils._mocking 模块中导入 MockDataFrame 类
from sklearn.utils._mocking import MockDataFrame
# 从 scikit-learn 的 utils._testing 模块中导入多个函数和类
from sklearn.utils._testing import (
    _array_api_for_tests,
    _convert_container,
    assert_allclose_dense_sparse,
    assert_array_equal,
    skip_if_array_api_compat_not_configured,
)
# 从 scikit-learn 的 utils.fixes 模块中导入 CSC_CONTAINERS 和 CSR_CONTAINERS 常量
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS

# 创建一个测试用的数组 X_toy，包含 0 到 8 共 9 个元素，reshape 成 3 行 3 列的二维数组
X_toy = np.arange(9).reshape((3, 3))


def test_polars_indexing():
    """Check _safe_indexing for polars as expected."""
    # 导入 polars 库，如果版本小于 0.18.2 则引发 SkipTest 异常
    pl = pytest.importorskip("polars", minversion="0.18.2")
    # 创建一个 polars 的 DataFrame 对象 df，包含三列数据
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": [4, 5, 6, 8], "c": [1, 4, 1, 10]}, orient="row"
    )

    # 导入 polars.testing 模块中的 assert_frame_equal 函数，用于比较 DataFrame 是否相等

    # 字符串类型的键列表
    str_keys = [["b"], ["a", "b"], ["b", "a", "c"], ["c"], ["a"]]
    # 遍历字符串类型的键列表
    for key in str_keys:
        # 使用 _safe_indexing 函数从 DataFrame df 中安全获取指定键的数据
        out = _safe_indexing(df, key, axis=1)
        # 断言从 DataFrame df 中获取的数据与预期数据相等
        assert_frame_equal(df[key], out)

    # 布尔类型的键列表
    bool_keys = [([True, False, True], ["a", "c"]), ([False, False, True], ["c"])]
    # 遍历布尔类型的键列表
    for bool_key, str_key in bool_keys:
        # 使用 _safe_indexing 函数从 DataFrame df 中安全获取指定键的数据
        out = _safe_indexing(df, bool_key, axis=1)
        # 断言从 DataFrame df 中获取的数据与预期数据相等
        assert_frame_equal(df[:, str_key], out)

    # 整数类型的键列表
    int_keys = [([0, 1], ["a", "b"]), ([2], ["c"])]
    # 遍历整数类型的键列表
    for int_key, str_key in int_keys:
        # 使用 _safe_indexing 函数从 DataFrame df 中安全获取指定键的数据
        out = _safe_indexing(df, int_key, axis=1)
        # 断言从 DataFrame df 中获取的数据与预期数据相等
        assert_frame_equal(df[:, str_key], out)

    # axis=0 方向的键列表
    axis_0_keys = [[0, 1], [1, 3], [3, 2]]
    # 遍历 axis=0 方向的键列表
    for key in axis_0_keys:
        # 使用 _safe_indexing 函数从 DataFrame df 中安全获取指定键的数据
        out = _safe_indexing(df, key, axis=0)
        # 断言从 DataFrame df 中获取的数据与预期数据相等
        assert_frame_equal(df[key], out)


@pytest.mark.parametrize(
    "key, dtype",
    [
        (0, "int"),
        ("0", "str"),
        (True, "bool"),
        (np.bool_(True), "bool"),
        ([0, 1, 2], "int"),
        (["0", "1", "2"], "str"),
        ((0, 1, 2), "int"),
        (("0", "1", "2"), "str"),
        (slice(None, None), None),
        (slice(0, 2), "int"),
        (np.array([0, 1, 2], dtype=np.int32), "int"),
        (np.array([0, 1, 2], dtype=np.int64), "int"),
        (np.array([0, 1, 2], dtype=np.uint8), "int"),
        ([True, False], "bool"),
        ((True, False), "bool"),
        (np.array([True, False]), "bool"),
        ("col_0", "str"),
        (["col_0", "col_1", "col_2"], "str"),
        (("col_0", "col_1", "col_2"), "str"),
        (slice("begin", "end"), "str"),
        (np.array(["col_0", "col_1", "col_2"]), "str"),
        (np.array(["col_0", "col_1", "col_2"], dtype=object), "str"),
    ],
)
def test_determine_key_type(key, dtype):
    # 断言 _determine_key_type 函数返回的键类型与预期类型相等
    assert _determine_key_type(key) == dtype


def test_determine_key_type_error():
    # 使用 pytest 检查 _determine_key_type 函数在输入错误时是否会引发 ValueError 异常，并匹配指定的错误信息
    with pytest.raises(ValueError, match="No valid specification of the"):
        _determine_key_type(1.0)
# 测试确定键类型函数对切片错误的处理
def test_determine_key_type_slice_error():
    # 使用 pytest 的断言检查是否抛出预期的 TypeError 异常，并匹配特定的错误信息
    with pytest.raises(TypeError, match="Only array-like or scalar are"):
        _determine_key_type(slice(0, 2, 1), accept_slice=False)


# 跳过如果未配置兼容的数组 API
@skip_if_array_api_compat_not_configured
# 使用参数化装饰器，提供多组参数给测试函数
@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
# 测试确定键类型函数对数组 API 的不同输入情况的处理
def test_determine_key_type_array_api(array_namespace, device, dtype_name):
    # 根据提供的 array_namespace 和 device 获取适合测试的数组 API
    xp = _array_api_for_tests(array_namespace, device)

    # 在 sklearn 的上下文中启用数组 API 调度
    with sklearn.config_context(array_api_dispatch=True):
        # 创建整数数组作为键，断言确定键类型函数的返回值为 "int"
        int_array_key = xp.asarray([1, 2, 3])
        assert _determine_key_type(int_array_key) == "int"

        # 创建布尔数组作为键，断言确定键类型函数的返回值为 "bool"
        bool_array_key = xp.asarray([True, False, True])
        assert _determine_key_type(bool_array_key) == "bool"

        # 尝试创建复数数组作为键，捕获 TypeError 异常（因为不是所有的数组 API 都支持复数）
        try:
            complex_array_key = xp.asarray([1 + 1j, 2 + 2j, 3 + 3j])
        except TypeError:
            # 复数数值在所有数组 API 库中不受支持的情况下，设为 None
            complex_array_key = None

        # 如果成功创建了复数数组键，则使用 pytest 的断言检查是否抛出预期的 ValueError 异常
        if complex_array_key is not None:
            with pytest.raises(ValueError, match="No valid specification of the"):
                _determine_key_type(complex_array_key)


# 参数化测试，测试在二维容器中安全索引操作在 axis=0 轴上的处理
@pytest.mark.parametrize(
    "array_type", ["list", "array", "sparse", "dataframe", "polars"]
)
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series", "slice"])
def test_safe_indexing_2d_container_axis_0(array_type, indices_type):
    # 设置索引
    indices = [1, 2]
    # 如果索引类型为 slice 且第二个索引是整数，将第二个索引增加1
    if indices_type == "slice" and isinstance(indices[1], int):
        indices[1] += 1
    # 转换容器类型为测试所需的类型
    array = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type)
    indices = _convert_container(indices, indices_type)
    # 执行安全索引操作
    subset = _safe_indexing(array, indices, axis=0)
    # 使用 assert_allclose_dense_sparse 函数断言子集与预期结果的近似性
    assert_allclose_dense_sparse(
        subset, _convert_container([[4, 5, 6], [7, 8, 9]], array_type)
    )


# 参数化测试，测试在一维容器中安全索引操作的处理
@pytest.mark.parametrize("array_type", ["list", "array", "series", "polars_series"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series", "slice"])
def test_safe_indexing_1d_container(array_type, indices_type):
    # 设置索引
    indices = [1, 2]
    # 如果索引类型为 slice 且第二个索引是整数，将第二个索引增加1
    if indices_type == "slice" and isinstance(indices[1], int):
        indices[1] += 1
    # 转换容器类型为测试所需的类型
    array = _convert_container([1, 2, 3, 4, 5, 6, 7, 8, 9], array_type)
    indices = _convert_container(indices, indices_type)
    # 执行安全索引操作
    subset = _safe_indexing(array, indices, axis=0)
    # 使用 assert_allclose_dense_sparse 函数断言子集与预期结果的近似性
    assert_allclose_dense_sparse(subset, _convert_container([2, 3], array_type))


# 参数化测试，测试在二维容器中安全索引操作在 axis=1 轴上的处理
@pytest.mark.parametrize("array_type", ["array", "sparse", "dataframe", "polars"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series", "slice"])
@pytest.mark.parametrize("indices", [[1, 2], ["col_1", "col_2"]])
def test_safe_indexing_2d_container_axis_1(array_type, indices_type, indices):
    # 验证索引，制作一份副本因为 indices 是可变的且在测试之间共享
    indices_converted = copy(indices)
    # 如果索引类型为 slice 且第二个索引是整数，将第二个索引增加1
    if indices_type == "slice" and isinstance(indices[1], int):
        indices_converted[1] += 1
    # 定义列名列表
    columns_name = ["col_0", "col_1", "col_2"]
    # 调用_convert_container函数，将二维数组[[1, 2, 3], [4, 5, 6], [7, 8, 9]]转换为指定类型的数据结构，并指定列名
    array = _convert_container(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name
    )
    # 调用_convert_container函数，将indices_converted转换为指定类型的数据结构，indices_type为数据类型参数
    
    indices_converted = _convert_container(indices_converted, indices_type)
    
    # 如果indices的第一个元素是字符串，并且array_type不是"dataframe"或"polars"
    if isinstance(indices[0], str) and array_type not in ("dataframe", "polars"):
        # 设置错误消息
        err_msg = (
            "Specifying the columns using strings is only supported for dataframes"
        )
        # 使用pytest的raises方法，检测是否抛出ValueError异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=err_msg):
            # 调用_safe_indexing函数，对array在axis=1维度上进行安全索引操作
            _safe_indexing(array, indices_converted, axis=1)
    else:
        # 调用_safe_indexing函数，对array在axis=1维度上进行安全索引操作，并赋值给subset
        subset = _safe_indexing(array, indices_converted, axis=1)
        # 使用assert_allclose_dense_sparse函数，断言subset与[[2, 3], [5, 6], [8, 9]]经_convert_container处理后的数据在稠密和稀疏表示上的接近性
        assert_allclose_dense_sparse(
            subset, _convert_container([[2, 3], [5, 6], [8, 9]], array_type)
        )
# 使用 pytest.mark.parametrize 装饰器定义测试参数化，array_read_only 参数为 True 或 False
# indices_read_only 参数为 True 或 False
# array_type 参数为 "array", "sparse", "dataframe", "polars"
# indices_type 参数为 "array", "series"
# axis 和 expected_array 参数对应 [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])] 的元组
@pytest.mark.parametrize("array_read_only", [True, False])
@pytest.mark.parametrize("indices_read_only", [True, False])
@pytest.mark.parametrize("array_type", ["array", "sparse", "dataframe", "polars"])
@pytest.mark.parametrize("indices_type", ["array", "series"])
@pytest.mark.parametrize(
    "axis, expected_array", [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])]
)
def test_safe_indexing_2d_read_only_axis_1(
    array_read_only, indices_read_only, array_type, indices_type, axis, expected_array
):
    # 创建二维数组
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 如果 array_read_only 为 True，则设置 array 不可写
    if array_read_only:
        array.setflags(write=False)
    # 将 array 转换为指定类型（array_type）
    array = _convert_container(array, array_type)
    # 创建索引数组
    indices = np.array([1, 2])
    # 如果 indices_read_only 为 True，则设置 indices 不可写
    if indices_read_only:
        indices.setflags(write=False)
    # 将 indices 转换为指定类型（indices_type）
    indices = _convert_container(indices, indices_type)
    # 调用 _safe_indexing 函数，从 array 中安全获取 subset，根据指定的 axis
    subset = _safe_indexing(array, indices, axis=axis)
    # 使用 assert_allclose_dense_sparse 断言 subset 与预期的 expected_array 近似相等
    assert_allclose_dense_sparse(subset, _convert_container(expected_array, array_type))


# 使用 pytest.mark.parametrize 装饰器定义测试参数化，array_type 参数为 "list", "array", "series", "polars_series"
# indices_type 参数为 "list", "tuple", "array", "series"
@pytest.mark.parametrize("array_type", ["list", "array", "series", "polars_series"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series"])
def test_safe_indexing_1d_container_mask(array_type, indices_type):
    # 创建索引列表
    indices = [False] + [True] * 2 + [False] * 6
    # 创建容器数组
    array = _convert_container([1, 2, 3, 4, 5, 6, 7, 8, 9], array_type)
    # 将 indices 转换为指定类型（indices_type）
    indices = _convert_container(indices, indices_type)
    # 调用 _safe_indexing 函数，从 array 中安全获取 subset，根据 axis=0
    subset = _safe_indexing(array, indices, axis=0)
    # 使用 assert_allclose_dense_sparse 断言 subset 与预期的 [2, 3] 近似相等
    assert_allclose_dense_sparse(subset, _convert_container([2, 3], array_type))


# 使用 pytest.mark.parametrize 装饰器定义测试参数化，array_type 参数为 "array", "sparse", "dataframe", "polars"
# indices_type 参数为 "list", "tuple", "array", "series"
# axis 和 expected_subset 参数对应 [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])] 的元组
@pytest.mark.parametrize("array_type", ["array", "sparse", "dataframe", "polars"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series"])
@pytest.mark.parametrize(
    "axis, expected_subset",
    [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])],
)
def test_safe_indexing_2d_mask(array_type, indices_type, axis, expected_subset):
    # 创建列名列表
    columns_name = ["col_0", "col_1", "col_2"]
    # 创建二维容器数组
    array = _convert_container(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name
    )
    # 创建索引列表
    indices = [False, True, True]
    # 将 indices 转换为指定类型（indices_type）
    indices = _convert_container(indices, indices_type)

    # 调用 _safe_indexing 函数，从 array 中安全获取 subset，根据指定的 axis
    subset = _safe_indexing(array, indices, axis=axis)
    # 使用 assert_allclose_dense_sparse 断言 subset 与预期的 expected_subset 近似相等
    assert_allclose_dense_sparse(
        subset, _convert_container(expected_subset, array_type)
    )


# 使用 pytest.mark.parametrize 装饰器定义测试参数化，array_type 参数为 "list", "array", "series", "polars_series"
@pytest.mark.parametrize(
    "array_type, expected_output_type",
    [
        ("list", "list"),
        ("array", "array"),
        ("sparse", "sparse"),
        ("dataframe", "series"),
        ("polars", "polars_series"),
    ],
)
def test_safe_indexing_2d_scalar_axis_0(array_type, expected_output_type):
    # 创建二维容器数组
    array = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type)
    # 创建索引值
    indices = 2
    # 调用 _safe_indexing 函数，从 array 中安全获取 subset，根据 axis=0
    subset = _safe_indexing(array, indices, axis=0)
    # 根据 expected_output_type 创建预期数组
    expected_array = _convert_container([7, 8, 9], expected_output_type)
    # 使用 assert_allclose_dense_sparse 断言 subset 与预期的 expected_array 近似相等
    assert_allclose_dense_sparse(subset, expected_array)
# 测试针对一维标量索引的安全索引功能
def test_safe_indexing_1d_scalar(array_type):
    # 将列表 [1, 2, 3, 4, 5, 6, 7, 8, 9] 转换为指定类型的容器对象
    array = _convert_container([1, 2, 3, 4, 5, 6, 7, 8, 9], array_type)
    # 索引值为标量 2
    indices = 2
    # 在 axis=0 轴向上进行安全索引操作
    subset = _safe_indexing(array, indices, axis=0)
    # 断言子集等于预期值 3
    assert subset == 3


# 参数化测试，测试针对二维标量索引在 axis=1 轴向上的安全索引功能
@pytest.mark.parametrize(
    "array_type, expected_output_type",
    [
        ("array", "array"),
        ("sparse", "sparse"),
        ("dataframe", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.parametrize("indices", [2, "col_2"])
def test_safe_indexing_2d_scalar_axis_1(array_type, expected_output_type, indices):
    # 列名列表
    columns_name = ["col_0", "col_1", "col_2"]
    # 将二维列表 [[1, 2, 3], [4, 5, 6], [7, 8, 9]] 转换为指定类型的容器对象
    array = _convert_container(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name
    )

    # 如果 indices 是字符串且 array_type 不在 ("dataframe", "polars") 中
    if isinstance(indices, str) and array_type not in ("dataframe", "polars"):
        # 抛出值错误，错误信息匹配 err_msg
        err_msg = "Specifying the columns using strings is only supported for dataframes"
        with pytest.raises(ValueError, match=err_msg):
            _safe_indexing(array, indices, axis=1)
    else:
        # 在 axis=1 轴向上进行安全索引操作
        subset = _safe_indexing(array, indices, axis=1)
        # 预期输出为 [3, 6, 9]
        expected_output = [3, 6, 9]
        # 如果预期输出类型为 sparse，则调整输出结构为 [[3], [6], [9]]
        if expected_output_type == "sparse":
            expected_output = [[3], [6], [9]]
        # 将预期输出转换为指定类型的容器对象
        expected_array = _convert_container(expected_output, expected_output_type)
        # 断言 subset 与 expected_array 在数值上近似相等
        assert_allclose_dense_sparse(subset, expected_array)


# 参数化测试，测试在 axis=0 轴向上对 None 的安全索引功能
@pytest.mark.parametrize("array_type", ["list", "array", "sparse"])
def test_safe_indexing_None_axis_0(array_type):
    # 将二维列表 [[1, 2, 3], [4, 5, 6], [7, 8, 9]] 转换为指定类型的容器对象
    X = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type)
    # 在 axis=0 轴向上进行安全索引操作
    X_subset = _safe_indexing(X, None, axis=0)
    # 断言 X_subset 与 X 在数值上近似相等
    assert_allclose_dense_sparse(X_subset, X)


# 测试当 Pandas DataFrame 中没有匹配列时，会引发错误
def test_safe_indexing_pandas_no_matching_cols_error():
    # 导入 Pandas，如果导入失败则跳过测试
    pd = pytest.importorskip("pandas")
    # 错误信息
    err_msg = "No valid specification of the columns."
    # 创建 Pandas DataFrame X
    X = pd.DataFrame(X_toy)
    # 使用 pytest 引发值错误，错误信息匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X, [1.0], axis=1)


# 参数化测试，测试错误的 axis 参数值
@pytest.mark.parametrize("axis", [None, 3])
def test_safe_indexing_error_axis(axis):
    # 使用 pytest 引发值错误，错误信息匹配 "'axis' should be either 0"
    with pytest.raises(ValueError, match="'axis' should be either 0"):
        _safe_indexing(X_toy, [0, 1], axis=axis)


# 参数化测试，测试对 1 维数组类型错误的安全索引操作
@pytest.mark.parametrize("X_constructor", ["array", "series", "polars_series"])
def test_safe_indexing_1d_array_error(X_constructor):
    # 检查如果传递的类似数组是 1 维的，并且尝试在第二维上进行索引时是否引发错误
    X = list(range(5))
    if X_constructor == "array":
        X_constructor = np.asarray(X)
    elif X_constructor == "series":
        pd = pytest.importorskip("pandas")
        X_constructor = pd.Series(X)
    elif X_constructor == "polars_series":
        pl = pytest.importorskip("polars")
        X_constructor = pl.Series(values=X)

    # 错误信息
    err_msg = "'X' should be a 2D NumPy array, 2D sparse matrix or dataframe"
    # 使用 pytest 引发值错误，错误信息匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X_constructor, [0, 1], axis=1)
# 测试在不支持的类型上使用安全索引，即列表索引
def test_safe_indexing_container_axis_0_unsupported_type():
    # 列表索引
    indices = ["col_1", "col_2"]
    # 二维数组
    array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # 错误消息
    err_msg = "String indexing is not supported with 'axis=0'"
    # 断言引发 ValueError 异常并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(array, indices, axis=0)


def test_safe_indexing_pandas_no_settingwithcopy_warning():
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 解析 pandas 版本号
    pd_version = parse_version(pd.__version__)
    pd_base_version = parse_version(pd_version.base_version)

    # 如果 pandas 的基础版本大于等于 3，则跳过测试
    if pd_base_version >= parse_version("3"):
        raise SkipTest("SettingWithCopyWarning has been removed in pandas 3.0.0.dev")

    # 创建 DataFrame 对象 X
    X = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    # 使用安全索引函数 _safe_indexing，以列表 [0, 1] 作为索引，axis=0
    subset = _safe_indexing(X, [0, 1], axis=0)
    
    # 如果 pandas 版本的错误模块中存在 SettingWithCopyWarning，则设置为该变量
    if hasattr(pd.errors, "SettingWithCopyWarning"):
        SettingWithCopyWarning = pd.errors.SettingWithCopyWarning
    else:
        # 向后兼容性处理，针对 pandas 版本小于 1.5
        SettingWithCopyWarning = pd.core.common.SettingWithCopyWarning
    
    # 使用警告捕获机制，捕获 SettingWithCopyWarning 异常并转换为错误
    with warnings.catch_warnings():
        warnings.simplefilter("error", SettingWithCopyWarning)
        # 在子集的第一行第一列进行赋值操作
        subset.iloc[0, 0] = 10
    # 断言原始 DataFrame 中第一行第一列的值未被赋值操作所影响
    assert X.iloc[0, 0] == 1


@pytest.mark.parametrize("indices", [0, [0, 1], slice(0, 2), np.array([0, 1])])
def test_safe_indexing_list_axis_1_unsupported(indices):
    """Check that we raise a ValueError when axis=1 with input as list."""
    # 输入为列表时，检查是否引发 ValueError 异常
    X = [[1, 2], [4, 5], [7, 8]]
    # 错误消息
    err_msg = "axis=1 is not supported for lists"
    # 断言引发 ValueError 异常并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X, indices, axis=1)


@pytest.mark.parametrize("array_type", ["array", "sparse", "dataframe"])
def test_safe_assign(array_type):
    """Check that `_safe_assign` works as expected."""
    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)
    # 生成随机数组 X_array
    X_array = rng.randn(10, 5)

    # 行索引器
    row_indexer = [1, 2]
    # 创建与行索引器对应的随机数组 values
    values = rng.randn(len(row_indexer), X_array.shape[1])
    # 将 X_array 转换为指定类型的容器 X
    X = _convert_container(X_array, array_type)
    # 使用安全赋值函数 _safe_assign
    _safe_assign(X, values, row_indexer=row_indexer)

    # 使用安全索引函数 _safe_indexing，以行索引器作为索引，axis=0
    assigned_portion = _safe_indexing(X, row_indexer, axis=0)
    # 断言所有密集或稀疏的值近似相等
    assert_allclose_dense_sparse(
        assigned_portion, _convert_container(values, array_type)
    )

    # 列索引器
    column_indexer = [1, 2]
    # 创建与列索引器对应的随机数组 values
    values = rng.randn(X_array.shape[0], len(column_indexer))
    # 将 X_array 转换为指定类型的容器 X
    X = _convert_container(X_array, array_type)
    # 使用安全赋值函数 _safe_assign
    _safe_assign(X, values, column_indexer=column_indexer)

    # 使用安全索引函数 _safe_indexing，以列索引器作为索引，axis=1
    assigned_portion = _safe_indexing(X, column_indexer, axis=1)
    # 断言所有密集或稀疏的值近似相等
    assert_allclose_dense_sparse(
        assigned_portion, _convert_container(values, array_type)
    )

    # 行索引器和列索引器均为 None
    row_indexer, column_indexer = None, None
    # 创建与 X_array 相同形状的随机数组 values
    values = rng.randn(*X.shape)
    # 将 X_array 转换为指定类型的容器 X
    X = _convert_container(X_array, array_type)
    # 使用安全赋值函数 _safe_assign
    _safe_assign(X, values, column_indexer=column_indexer)

    # 断言所有密集或稀疏的值近似相等
    assert_allclose_dense_sparse(X, _convert_container(values, array_type))
# 使用 pytest 的 parametrize 装饰器来多次运行同一个测试函数，测试 get_column_indices 函数在不同情况下的错误处理能力
@pytest.mark.parametrize(
    "key, err_msg",
    [
        (10, r"all features must be in \[0, 2\]"),  # 当 key 为整数 10 时，预期抛出特定错误消息
        ("whatever", "A given column is not a column of the dataframe"),  # 当 key 为字符串 "whatever" 时，预期抛出特定错误消息
        (object(), "No valid specification of the columns"),  # 当 key 为对象时，预期抛出特定错误消息
    ],
)
def test_get_column_indices_error(key, err_msg):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个名为 X_df 的 DataFrame，用于测试
    X_df = pd.DataFrame(X_toy, columns=["col_0", "col_1", "col_2"])

    # 使用 pytest 的 raises 方法检查调用 _get_column_indices 函数时是否抛出 ValueError，并匹配特定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        _get_column_indices(X_df, key)


# 使用 pytest 的 parametrize 装饰器来多次运行同一个测试函数，测试 get_column_indices 函数在 pandas 非唯一列名情况下的错误处理能力
@pytest.mark.parametrize(
    "key", [["col1"], ["col2"], ["col1", "col2"], ["col1", "col3"], ["col2", "col3"]]
)
def test_get_column_indices_pandas_nonunique_columns_error(key):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个名为 X 的 DataFrame，包含列名重复的情况
    toy = np.zeros((1, 5), dtype=int)
    columns = ["col1", "col1", "col2", "col3", "col2"]
    X = pd.DataFrame(toy, columns=columns)

    # 构造错误消息
    err_msg = "Selected columns, {}, are not unique in dataframe".format(key)
    # 使用 pytest 的 raises 方法检查调用 _get_column_indices 函数时是否抛出 ValueError，并获取异常信息对象
    with pytest.raises(ValueError) as exc_info:
        _get_column_indices(X, key)
    # 断言异常信息的字符串形式与预期的错误消息相匹配
    assert str(exc_info.value) == err_msg


# 测试函数，检查 _get_column_indices 函数在边缘情况下的交换处理
def test_get_column_indices_interchange():
    """Check _get_column_indices for edge cases with the interchange"""
    # 导入 pandas 库，如果版本低于 1.5 则跳过测试
    pd = pytest.importorskip("pandas", minversion="1.5")

    # 创建一个简单的 DataFrame
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])

    # 创建一个模拟的 DataFrame 对象，用于触发 dataframe 协议代码路径
    class MockDataFrame:
        def __init__(self, df):
            self._df = df

        def __getattr__(self, name):
            return getattr(self._df, name)

    # 使用 MockDataFrame 包装原始 DataFrame
    df_mocked = MockDataFrame(df)

    # 预定义的测试键和预期结果
    key_results = [
        (slice(1, None), [1, 2]),       # 切片键测试，预期返回索引 1 和 2 的列
        (slice(None, 2), [0, 1]),       # 切片键测试，预期返回索引 0 和 1 的列
        (slice(1, 2), [1]),             # 切片键测试，预期返回索引 1 的列
        (["b", "c"], [1, 2]),           # 列名列表测试，预期返回 "b" 和 "c" 列的索引
        (slice("a", "b"), [0, 1]),      # 切片键测试，预期返回 "a" 到 "b" 列的索引
        (slice("a", None), [0, 1, 2]),  # 切片键测试，预期返回 "a" 到最后列的索引
        (slice(None, "a"), [0]),        # 切片键测试，预期返回从开头到 "a" 列的索引
        (["c", "a"], [2, 0]),           # 列名列表测试，预期返回 "c" 和 "a" 列的索引
        ([], []),                       # 空列表测试，预期返回空列表
    ]

    # 遍历测试键和预期结果，逐一断言调用 _get_column_indices 函数返回的结果是否符合预期
    for key, result in key_results:
        assert _get_column_indices(df_mocked, key) == result

    # 测试不存在的列名时是否抛出特定的 ValueError 异常
    msg = "A given column is not a column of the dataframe"
    with pytest.raises(ValueError, match=msg):
        _get_column_indices(df_mocked, ["not_a_column"])

    # 测试不支持的切片步长时是否抛出特定的 NotImplementedError 异常
    msg = "key.step must be 1 or None"
    with pytest.raises(NotImplementedError, match=msg):
        _get_column_indices(df_mocked, slice("a", None, 2))


# 测试函数，检查 resample 函数的边界情况
def test_resample():
    # 边界情况测试，确认 resample 返回 None
    assert resample() is None

    # 检查使用无效参数时是否引发 ValueError 异常
    with pytest.raises(ValueError):
        resample([0], [0, 1])
    with pytest.raises(ValueError):
        resample([0, 1], [0, 1], replace=False, n_samples=3)

    # 检查 Issue:6581，当 replace 为 True 时，n_samples 可能大于预期值
    assert len(resample([1, 2], n_samples=5)) == 5


# 测试函数，检查 resample 函数的分层抽样能力
def test_resample_stratified():
    # 确保 resample 可以进行分层抽样
    rng = np.random.RandomState(0)
    n_samples = 100
    p = 0.9
    X = rng.normal(size=(n_samples, 1))
    y = rng.binomial(1, p, size=n_samples)
    # 无需使用变量保存未分层重新采样后的样本，使用下划线 `_` 表示忽略此变量
    _, y_not_stratified = resample(X, y, n_samples=10, random_state=0, stratify=None)
    # 断言：验证未分层重新采样后所有标签值均为1
    assert np.all(y_not_stratified == 1)
    
    # 使用分层重新采样方法重新采样数据
    _, y_stratified = resample(X, y, n_samples=10, random_state=0, stratify=y)
    # 断言：验证分层重新采样后不是所有标签值都为1
    assert not np.all(y_stratified == 1)
    # 断言：验证分层重新采样后标签值为1的数量为9（总数为10，其中一个为0）
    assert np.sum(y_stratified) == 9  # all 1s, one 0
def test_resample_stratified_replace():
    # 确保分层重采样支持 replace 参数
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 1))  # 生成正态分布的随机数据
    y = rng.randint(0, 2, size=n_samples)  # 生成随机的0/1标签

    X_replace, _ = resample(  # 使用 replace=True 进行重采样
        X, y, replace=True, n_samples=50, random_state=rng, stratify=y
    )
    X_no_replace, _ = resample(  # 使用 replace=False 进行重采样
        X, y, replace=False, n_samples=50, random_state=rng, stratify=y
    )
    assert np.unique(X_replace).shape[0] < 50  # 确保有重复数据的情况下，唯一值的数量小于50
    assert np.unique(X_no_replace).shape[0] == 50  # 确保无重复数据的情况下，唯一值的数量为50

    # 确保当使用 replacement 时，n_samples 可以大于 X.shape[0]
    X_replace, _ = resample(
        X, y, replace=True, n_samples=1000, random_state=rng, stratify=y
    )
    assert X_replace.shape[0] == 1000  # 确保结果的样本数为1000
    assert np.unique(X_replace).shape[0] == 100  # 确保结果中的唯一值数量为100


def test_resample_stratify_2dy():
    # 确保当 y 是二维时，仍能进行分层采样
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 1))  # 生成正态分布的随机数据
    y = rng.randint(0, 2, size=(n_samples, 2))  # 生成随机的0/1标签，形状为 (n_samples, 2)
    X, y = resample(X, y, n_samples=50, random_state=rng, stratify=y)  # 进行分层重采样
    assert y.ndim == 2  # 确保结果中 y 的维度为2


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_resample_stratify_sparse_error(csr_container):
    # 确保 resample 接受 ndarray 类型的输入
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 2))  # 生成正态分布的随机数据
    y = rng.randint(0, 2, size=n_samples)  # 生成随机的0/1标签
    stratify = csr_container(y.reshape(-1, 1))  # 使用 csr_container 将 y 转换为稀疏矩阵
    with pytest.raises(TypeError, match="Sparse data was passed"):  # 确保抛出类型错误异常
        X, y = resample(X, y, n_samples=50, random_state=rng, stratify=stratify)


def test_shuffle_on_ndim_equals_three():
    def to_tuple(A):  # 将内部数组转换为元组以便哈希化
        return tuple(tuple(tuple(C) for C in B) for B in A)

    A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 创建三维数组 A，形状为 (2, 2, 2)
    S = set(to_tuple(A))  # 将数组 A 转换为元组集合 S
    shuffle(A)  # 对数组 A 进行打乱操作，确保不会因维度为3而引发 ValueError
    assert set(to_tuple(A)) == S  # 确保打乱后的数组 A 的元组集合与原始的 S 相同


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_shuffle_dont_convert_to_array(csc_container):
    # 确保 shuffle 不会尝试将数据转换为 numpy 数组，并允许任何可索引的数据结构通过
    a = ["a", "b", "c"]  # 创建列表 a
    b = np.array(["a", "b", "c"], dtype=object)  # 创建对象类型的 numpy 数组 b
    c = [1, 2, 3]  # 创建列表 c
    d = MockDataFrame(np.array([["a", 0], ["b", 1], ["c", 2]], dtype=object))  # 创建 MockDataFrame 对象 d
    e = csc_container(np.arange(6).reshape(3, 2))  # 使用 csc_container 创建稀疏矩阵 e
    a_s, b_s, c_s, d_s, e_s = shuffle(a, b, c, d, e, random_state=0)  # 对 a, b, c, d, e 进行打乱操作

    assert a_s == ["c", "b", "a"]  # 确保列表 a_s 被正确打乱
    assert type(a_s) == list  # 确保 a_s 的类型为列表

    assert_array_equal(b_s, ["c", "b", "a"])  # 确保 numpy 数组 b_s 被正确打乱
    assert b_s.dtype == object  # 确保 b_s 的数据类型为对象类型

    assert c_s == [3, 2, 1]  # 确保列表 c_s 被正确打乱
    assert type(c_s) == list  # 确保 c_s 的类型为列表

    assert_array_equal(d_s, np.array([["c", 2], ["b", 1], ["a", 0]], dtype=object))  # 确保 MockDataFrame d_s 被正确打乱
    assert type(d_s) == MockDataFrame  # 确保 d_s 的类型为 MockDataFrame

    assert_array_equal(e_s.toarray(), np.array([[4, 5], [2, 3], [0, 1]]))  # 确保稀疏矩阵 e_s 被正确打乱
```