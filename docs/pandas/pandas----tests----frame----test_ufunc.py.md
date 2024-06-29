# `D:\src\scipysrc\pandas\pandas\tests\frame\test_ufunc.py`

```
# 导入必要的模块和函数
from functools import partial  # 导入 functools 模块的 partial 函数，用于创建偏函数
import re  # 导入 re 模块，用于正则表达式操作

import numpy as np  # 导入 numpy 库并命名为 np，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 pandas 库并命名为 pd，用于数据处理和分析
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块
from pandas.api.types import is_extension_array_dtype  # 从 pandas 的 api.types 模块中导入 is_extension_array_dtype 函数

# 不同的数据类型用于测试
dtypes = [
    "int64",  # 标准整数数据类型
    "Int64",  # 支持缺失值的整数数据类型
    {"A": "int64", "B": "Int64"},  # 指定每列数据类型的字典
]

# 使用 pytest 的 parametrize 装饰器对以下测试函数进行参数化
@pytest.mark.parametrize("dtype", dtypes)
def test_unary_unary(dtype):
    # 测试一元操作符的行为，输入和输出都是一元的
    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    # 创建一个 DataFrame 对象，包含指定数据类型，并进行类型转换
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    # 对 DataFrame 执行 numpy 的正数操作
    result = np.positive(df)
    # 创建预期结果的 DataFrame，类型与输入相同
    expected = pd.DataFrame(
        np.positive(values), index=df.index, columns=df.columns
    ).astype(dtype)
    # 使用 pandas._testing 模块的 assert_frame_equal 函数比较结果和预期值
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_unary_binary(request, dtype):
    # 测试一元操作符的行为，输入是一元的，输出是二元的
    if is_extension_array_dtype(dtype) or isinstance(dtype, dict):
        # 如果数据类型是扩展类型或者是字典，则标记为预期失败
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple outputs not implemented."
            )
        )

    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    # 创建一个 DataFrame 对象，包含指定数据类型，并进行类型转换
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    # 对 DataFrame 执行 numpy 的浮点数分解操作
    result_pandas = np.modf(df)
    # 断言返回结果是一个元组
    assert isinstance(result_pandas, tuple)
    # 断言元组长度为 2
    assert len(result_pandas) == 2
    # 创建 numpy 预期结果
    expected_numpy = np.modf(values)

    # 遍历结果元组和预期结果，使用 pandas._testing 模块的 assert_frame_equal 函数比较每个部分
    for result, b in zip(result_pandas, expected_numpy):
        expected = pd.DataFrame(b, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_binary_input_dispatch_binop(dtype):
    # 测试二元输入分派到我们的特殊方法
    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    # 创建一个 DataFrame 对象，包含指定数据类型，并进行类型转换
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    # 对 DataFrame 执行 numpy 的加法操作
    result = np.add(df, df)
    # 创建预期结果的 DataFrame，类型与输入相同
    expected = pd.DataFrame(
        np.add(values, values), index=df.index, columns=df.columns
    ).astype(dtype)
    # 使用 pandas._testing 模块的 assert_frame_equal 函数比较结果和预期值
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func,arg,expected",
    [
        (np.add, 1, [2, 3, 4, 5]),  # 测试 np.add 函数的单参数情况
        (
            partial(np.add, where=[[False, True], [True, False]]),
            np.array([[1, 1], [1, 1]]),
            [0, 3, 4, 0],
        ),  # 测试部分应用 np.add 函数，并传递额外参数的情况
        (np.power, np.array([[1, 1], [2, 2]]), [1, 2, 9, 16]),  # 测试 np.power 函数
        (np.subtract, 2, [-1, 0, 1, 2]),  # 测试 np.subtract 函数
        (
            partial(np.negative, where=np.array([[False, True], [True, False]])),
            None,
            [0, -2, -3, 0],
        ),  # 测试部分应用 np.negative 函数，并传递额外参数的情况
    ],
)
def test_ufunc_passes_args(func, arg, expected):
    # 测试 numpy 通用函数传递参数的行为
    arr = np.array([[1, 2], [3, 4]])
    df = pd.DataFrame(arr)
    result_inplace = np.zeros_like(arr)
    # 如果参数 arg 是 None，则使用 func 对 DataFrame 执行 in-place 操作
    if arg is None:
        result = func(df, out=result_inplace)
    else:
        # 否则，使用 func 对 DataFrame 和参数 arg 执行 in-place 操作
        result = func(df, arg, out=result_inplace)

    expected = np.array(expected).reshape(2, 2)
    # 使用 pandas._testing 模块的 assert_numpy_array_equal 函数比较结果和预期值
    tm.assert_numpy_array_equal(result_inplace, expected)

    expected = pd.DataFrame(expected)  # 创建预期结果的 DataFrame
    # 使用测试框架中的 assert_frame_equal 函数比较 result 和 expected 两个数据帧是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("dtype_a", dtypes)
@pytest.mark.parametrize("dtype_b", dtypes)
# 使用 pytest 的 parametrize 标记，对 dtype_a 和 dtype_b 进行参数化测试

def test_binary_input_aligns_columns(request, dtype_a, dtype_b):
    if (
        is_extension_array_dtype(dtype_a)
        or isinstance(dtype_a, dict)
        or is_extension_array_dtype(dtype_b)
        or isinstance(dtype_b, dict)
    ):
        # 如果 dtype_a 或 dtype_b 是扩展数组类型或字典类型，则标记测试为预期失败
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple inputs not implemented."
            )
        )

    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}).astype(dtype_a)
    # 创建 DataFrame df1，其中列类型转换为 dtype_a

    if isinstance(dtype_a, dict) and isinstance(dtype_b, dict):
        dtype_b = dtype_b.copy()
        dtype_b["C"] = dtype_b.pop("B")
    df2 = pd.DataFrame({"A": [1, 2], "C": [3, 4]}).astype(dtype_b)
    # 如果 dtype_a 和 dtype_b 都是字典类型，则复制 dtype_b 并调整其键，创建 DataFrame df2

    # 从 Pandas DataFrame df1 和 df2 调用 np.heaviside 函数
    result = np.heaviside(df1, df2)

    # 期望结果，对 np.heaviside 函数的预期输出
    expected = np.heaviside(
        np.array([[1, 3, np.nan], [2, 4, np.nan]]),
        np.array([[1, np.nan, 3], [2, np.nan, 4]]),
    )
    # 创建期望的 Pandas DataFrame，指定索引和列名
    expected = pd.DataFrame(expected, index=[0, 1], columns=["A", "B", "C"])
    # 使用 Pandas 的 tm 模块比较结果和期望值的 DataFrame
    tm.assert_frame_equal(result, expected)

    # 使用 np.heaviside 函数，但 df2 使用其值而不是 DataFrame
    result = np.heaviside(df1, df2.values)
    # 创建期望的 Pandas DataFrame，指定索引和列名
    expected = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"])
    # 使用 Pandas 的 tm 模块比较结果和期望值的 DataFrame
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
# 使用 pytest 的 parametrize 标记，对 dtype 进行参数化测试
def test_binary_input_aligns_index(request, dtype):
    if is_extension_array_dtype(dtype) or isinstance(dtype, dict):
        # 如果 dtype 是扩展数组类型或字典类型，则标记测试为预期失败
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple inputs not implemented."
            )
        )
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).astype(dtype)
    # 创建 DataFrame df1，指定索引和列类型为 dtype
    df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "c"]).astype(dtype)
    # 创建 DataFrame df2，指定索引和列类型为 dtype
    result = np.heaviside(df1, df2)
    # 调用 np.heaviside 函数，传入两个 DataFrame
    expected = np.heaviside(
        np.array([[1, 3], [3, 4], [np.nan, np.nan]]),
        np.array([[1, 3], [np.nan, np.nan], [3, 4]]),
    )
    # 创建期望的 Pandas DataFrame，指定索引和列名
    expected = pd.DataFrame(expected, index=["a", "b", "c"], columns=["A", "B"])
    # 使用 Pandas 的 tm 模块比较结果和期望值的 DataFrame
    tm.assert_frame_equal(result, expected)

    # 使用 np.heaviside 函数，但 df2 使用其值而不是 DataFrame
    result = np.heaviside(df1, df2.values)
    # 创建期望的 Pandas DataFrame，指定索引和列名
    expected = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"], index=["a", "b"]
    )
    # 使用 Pandas 的 tm 模块比较结果和期望值的 DataFrame
    tm.assert_frame_equal(result, expected)


def test_binary_frame_series_raises():
    # 当前不支持的操作
    df = pd.DataFrame({"A": [1, 2]})
    # 使用 pytest 的 raises 断言，检查是否抛出 NotImplementedError 异常，匹配给定的消息字符串
    with pytest.raises(NotImplementedError, match="logaddexp"):
        np.logaddexp(df, df["A"])

    # 同上，检查是否抛出 NotImplementedError 异常，匹配给定的消息字符串
    with pytest.raises(NotImplementedError, match="logaddexp"):
        np.logaddexp(df["A"], df)


def test_unary_accumulate_axis():
    # https://github.com/pandas-dev/pandas/issues/39259
    df = pd.DataFrame({"a": [1, 3, 2, 4]})
    # 使用 np.maximum.accumulate 对 DataFrame df 进行累积计算
    result = np.maximum.accumulate(df)
    # 创建期望的 Pandas DataFrame
    expected = pd.DataFrame({"a": [1, 3, 3, 4]})
    # 使用 Pandas 的 tm 模块比较结果和期望值的 DataFrame
    tm.assert_frame_equal(result, expected)

    df = pd.DataFrame({"a": [1, 3, 2, 4], "b": [0.1, 4.0, 3.0, 2.0]})
    # 计算 DataFrame 中每列的累积最大值，返回结果
    result = np.maximum.accumulate(df)
    # 在理论上可以保持整数数据类型（dtype）默认情况下对 axis=0 操作
    expected = pd.DataFrame({"a": [1.0, 3.0, 3.0, 4.0], "b": [0.1, 4.0, 4.0, 4.0]})
    # 使用测试工具比较计算结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)
    
    # 计算 DataFrame 中每行的累积最大值，返回结果
    result = np.maximum.accumulate(df, axis=0)
    # 使用测试工具比较计算结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)
    
    # 计算 DataFrame 中每行的累积最大值，返回结果
    result = np.maximum.accumulate(df, axis=1)
    expected = pd.DataFrame({"a": [1.0, 3.0, 2.0, 4.0], "b": [1.0, 4.0, 3.0, 4.0]})
    # 使用测试工具比较计算结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)
def test_frame_outer_disallowed():
    df = pd.DataFrame({"A": [1, 2]})
    # 使用 pytest 的 assert_raises 来测试是否会抛出 NotImplementedError 异常，且异常消息为空字符串
    with pytest.raises(NotImplementedError, match=""):
        # 在 2.0 版本中实施弃用
        np.subtract.outer(df, df)


def test_alignment_deprecation_enforced():
    # 在 2.0 版本中实施强制执行
    # https://github.com/pandas-dev/pandas/issues/39184
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    s1 = pd.Series([1, 2], index=["a", "b"])
    s2 = pd.Series([1, 2], index=["b", "c"])

    # 二进制操作，DataFrame 和 DataFrame
    expected = pd.DataFrame({"a": [2, 4, 6], "b": [8, 10, 12]})
    with tm.assert_produces_warning(None):
        # 对齐的情况下不会有警告！
        result = np.add(df1, df1)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, df2.values)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, df2)
    expected = pd.DataFrame({"a": [np.nan] * 3, "b": [5, 7, 9], "c": [np.nan] * 3})
    tm.assert_frame_equal(result, expected)

    result = np.add(df1.values, df2)
    expected = pd.DataFrame({"b": [2, 4, 6], "c": [8, 10, 12]})
    tm.assert_frame_equal(result, expected)

    # 二进制操作，DataFrame 和 Series
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [6, 7, 8]})
    with tm.assert_produces_warning(None):
        # 对齐的情况下不会有警告！
        result = np.add(df1, s1)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, s2.values)
    tm.assert_frame_equal(result, expected)

    expected = pd.DataFrame(
        {"a": [np.nan] * 3, "b": [5.0, 6.0, 7.0], "c": [np.nan] * 3}
    )
    result = np.add(df1, s2)
    tm.assert_frame_equal(result, expected)

    msg = "Cannot apply ufunc <ufunc 'add'> to mixed DataFrame and Series inputs."
    with pytest.raises(NotImplementedError, match=msg):
        # 尝试在 Series 和 DataFrame 之间应用 ufunc <ufunc 'add'>，期望抛出特定错误消息的异常
        np.add(s2, df1)


@pytest.mark.single_cpu
def test_alignment_deprecation_many_inputs_enforced():
    # 在 2.0 版本中实施强制执行
    # https://github.com/pandas-dev/pandas/issues/39184
    # 测试在大于 2 个输入时是否会弃用 -> 使用 numba 编写的 ufunc 处理，因为 numpy 本身没有这样的 ufuncs
    numba = pytest.importorskip("numba")

    @numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
    def my_ufunc(x, y, z):
        return x + y + z

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    df3 = pd.DataFrame({"a": [1, 2, 3], "c": [4, 5, 6]})

    result = my_ufunc(df1, df2, df3)
    expected = pd.DataFrame(np.full((3, 3), np.nan), columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)

    # 全部对齐的情况下不会有警告
    with tm.assert_produces_warning(None):
        result = my_ufunc(df1, df1, df1)
    expected = pd.DataFrame([[3.0, 12.0], [6.0, 15.0], [9.0, 18.0]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # 混合 DataFrame / arrays
    # 定义错误消息，用于匹配 pytest 抛出的 ValueError 异常
    msg = (
        r"operands could not be broadcast together with shapes \(3,3\) \(3,3\) \(3,2\)"
    )
    # 使用 pytest 的 raises 断言，验证调用 my_ufunc 函数时是否抛出预期的异常
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1, df2, df3.values)

    # 单个数据框 -> 没有警告
    # 使用 tm.assert_produces_warning(None) 断言，确保调用 my_ufunc 函数时不会产生警告
    with tm.assert_produces_warning(None):
        # 调用 my_ufunc 函数，并将结果赋给 result
        result = my_ufunc(df1, df2.values, df3.values)
    # 使用 tm.assert_frame_equal 验证 result 是否与期望的结果 expected 相等
    tm.assert_frame_equal(result, expected)

    # 使用第一个数据框的索引
    # 定义错误消息，用于匹配 pytest 抛出的 ValueError 异常
    msg = (
        r"operands could not be broadcast together with shapes \(3,2\) \(3,3\) \(3,3\)"
    )
    # 使用 pytest 的 raises 断言，验证调用 my_ufunc 函数时是否抛出预期的异常
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1.values, df2, df3)
def test_array_ufuncs_for_many_arguments():
    # 定义一个函数 add3，用于将三个参数相加
    def add3(x, y, z):
        return x + y + z

    # 从 Python 函数 add3 创建一个通用函数（ufunc），它可以处理三个输入并返回一个输出
    ufunc = np.frompyfunc(add3, 3, 1)

    # 创建一个包含整数的 DataFrame
    df = pd.DataFrame([[1, 2], [3, 4]])

    # 使用 ufunc 处理 DataFrame df 与自身及标量 1 的组合，返回结果 DataFrame result
    result = ufunc(df, df, 1)

    # 创建一个期望的 DataFrame，包含预期的结果
    expected = pd.DataFrame([[3, 5], [7, 9]], dtype=object)

    # 断言 result 和 expected 的 DataFrame 结果是否相等
    tm.assert_frame_equal(result, expected)

    # 创建一个包含整数的 Series
    ser = pd.Series([1, 2])

    # 准备一个错误消息，指出无法将 ufunc 应用于混合的 DataFrame 和 Series 输入
    msg = (
        "Cannot apply ufunc <ufunc 'add3 (vectorized)'> "
        "to mixed DataFrame and Series inputs."
    )

    # 使用 pytest 来检查是否会引发 NotImplementedError，并匹配预期的错误消息
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        ufunc(df, df, ser)
```