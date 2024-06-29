# `D:\src\scipysrc\pandas\pandas\tests\tools\test_to_numeric.py`

```
import decimal  # 导入 decimal 模块

import numpy as np  # 导入 NumPy 库，并将其命名为 np
from numpy import iinfo  # 从 NumPy 中导入 iinfo 函数
import pytest  # 导入 pytest 测试框架

import pandas.util._test_decorators as td  # 导入 pandas 内部测试工具模块

import pandas as pd  # 导入 pandas 库，并将其命名为 pd
from pandas import (  # 从 pandas 中导入多个类和函数
    ArrowDtype,
    DataFrame,
    Index,
    Series,
    option_context,
    to_numeric,
)
import pandas._testing as tm  # 导入 pandas 测试工具模块


@pytest.fixture(params=[None, "raise", "coerce"])  # 定义名为 errors 的测试夹具，参数为 None, "raise", "coerce"
def errors(request):
    return request.param  # 返回参数化的值


@pytest.fixture(params=[True, False])  # 定义名为 signed 的测试夹具，参数为 True 和 False
def signed(request):
    return request.param  # 返回参数化的值


@pytest.fixture(params=[lambda x: x, str], ids=["identity", "str"])  # 定义名为 transform 的测试夹具，参数化为 lambda 函数和 str，指定 ids
def transform(request):
    return request.param  # 返回参数化的值


@pytest.fixture(params=[47393996303418497800, 100000000000000000000])  # 定义名为 large_val 的测试夹具，参数化为两个大数值
def large_val(request):
    return request.param  # 返回参数化的值


@pytest.fixture(params=[True, False])  # 定义名为 multiple_elts 的测试夹具，参数为 True 和 False
def multiple_elts(request):
    return request.param  # 返回参数化的值


@pytest.fixture(  # 定义名为 transform_assert_equal 的测试夹具，参数化为元组列表
    params=[
        (lambda x: Index(x, name="idx"), tm.assert_index_equal),  # 元组包含 lambda 函数和相应的断言函数
        (lambda x: Series(x, name="ser"), tm.assert_series_equal),  # 元组包含 lambda 函数和相应的断言函数
        (lambda x: np.array(Index(x).values), tm.assert_numpy_array_equal),  # 元组包含 lambda 函数和相应的断言函数
    ]
)
def transform_assert_equal(request):
    return request.param  # 返回参数化的值


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器参数化测试函数 test_empty 的输入和预期输出
    "input_kwargs,result_kwargs",
    [
        ({}, {"dtype": np.int64}),  # 空字典作为输入，输出为指定 dtype 的 Series
        ({"errors": "coerce", "downcast": "integer"}, {"dtype": np.int8}),  # 指定了 errors 和 downcast 参数，输出为指定 dtype 的 Series
    ],
)
def test_empty(input_kwargs, result_kwargs):
    # see gh-16302
    ser = Series([], dtype=object)  # 创建一个空的 Series 对象，指定数据类型为 object

    result = to_numeric(ser, **input_kwargs)  # 调用 to_numeric 函数处理空的 Series 对象

    expected = Series([], **result_kwargs)  # 创建一个期望的 Series 对象，指定参数化的输出参数
    tm.assert_series_equal(result, expected)  # 断言处理结果与期望结果相等


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器参数化测试函数 test_series 的 infer_string 和 last_val 参数
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("last_val", ["7", 7])  # 参数化 last_val 参数为字符串 "7" 和整数 7
def test_series(last_val, infer_string):
    with option_context("future.infer_string", infer_string):  # 设置上下文，用于控制特定选项的值
        ser = Series(["1", "-3.14", last_val])  # 创建一个包含字符串和 last_val 的 Series 对象
        result = to_numeric(ser)  # 调用 to_numeric 函数将 Series 转换为数值类型

    expected = Series([1, -3.14, 7])  # 创建一个期望的 Series 对象，期望值已经转换为数值类型
    tm.assert_series_equal(result, expected)  # 断言处理结果与期望结果相等


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器参数化测试函数 test_series_numeric 的 data 参数
    "data",
    [
        [1, 3, 4, 5],  # 整数列表
        [1.0, 3.0, 4.0, 5.0],  # 浮点数列表
        # Bool is regarded as numeric.
        [True, False, True, True],  # 布尔值列表，被视为数值类型
    ],
)
def test_series_numeric(data):
    ser = Series(data, index=list("ABCD"), name="EFG")  # 创建一个 Series 对象，包含指定数据和索引

    result = to_numeric(ser)  # 调用 to_numeric 函数将 Series 转换为数值类型
    tm.assert_series_equal(result, ser)  # 断言处理结果与原始 Series 对象相等


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器参数化测试函数 test_error 的 data 和 msg 参数
    "data,msg",
    [
        ([1, -3.14, "apple"], 'Unable to parse string "apple" at position 2'),  # 包含无法解析的字符串的列表
        (
            ["orange", 1, -3.14, "apple"],  # 包含无法解析的字符串的列表
            'Unable to parse string "orange" at position 0',  # 期望的错误消息
        ),
    ],
)
def test_error(data, msg):
    ser = Series(data)  # 创建一个包含指定数据的 Series 对象

    with pytest.raises(ValueError, match=msg):  # 断言调用 to_numeric 函数时抛出 ValueError 异常，并匹配特定的错误消息
        to_numeric(ser, errors="raise")


def test_ignore_error():
    ser = Series([1, -3.14, "apple"])  # 创建一个包含指定数据的 Series 对象
    result = to_numeric(ser, errors="coerce")  # 调用 to_numeric 函数，并忽略错误

    expected = Series([1, -3.14, np.nan])  # 创建一个期望的 Series 对象，忽略错误后的结果
    tm.assert_series_equal(result, expected)  # 断言处理结果与期望结果相等
    [
        # 抛出异常，指示在位置 2 无法解析字符串 "apple"
        ("raise", 'Unable to parse string "apple" at position 2'),
        # 强制转换为浮点数
        ("coerce", [1.0, 0.0, np.nan]),
    ],
def test_bool_handling(errors, exp):
    # 创建一个包含 True、False 和 "apple" 的 Pandas Series 对象
    ser = Series([True, False, "apple"])

    # 如果 exp 是字符串，预期会引发 ValueError 异常，并且异常信息要匹配 exp
    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        # 否则，将 Series 转换为数值型，使用 errors 参数控制错误处理方式
        result = to_numeric(ser, errors=errors)
        # 创建预期结果的 Series 对象
        expected = Series(exp)

        # 断言转换后的结果与预期结果相等
        tm.assert_series_equal(result, expected)


def test_list():
    # 创建一个包含字符串 "1", "-3.14", "7" 的列表
    ser = ["1", "-3.14", "7"]
    # 将列表中的字符串元素转换为相应的数值类型
    res = to_numeric(ser)

    # 创建预期的 NumPy 数组
    expected = np.array([1, -3.14, 7])
    # 断言转换后的结果与预期结果相等
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "data,arr_kwargs",
    [
        # 使用参数 {"dtype": np.int64} 创建包含整数的列表
        ([1, 3, 4, 5], {"dtype": np.int64}),
        # 使用默认参数创建包含浮点数的列表
        ([1.0, 3.0, 4.0, 5.0], {}),
        # 布尔值被视为数值型
        ([True, False, True, True], {}),
    ],
)
def test_list_numeric(data, arr_kwargs):
    # 将列表中的元素转换为数值类型
    result = to_numeric(data)
    # 使用指定的参数创建预期的 NumPy 数组
    expected = np.array(data, **arr_kwargs)
    # 断言转换后的结果与预期结果相等
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"dtype": "O"}, {}])
def test_numeric(kwargs):
    # 创建一个包含整数、浮点数和字符串的列表
    data = [1, -3.14, 7]

    # 使用 kwargs 参数创建 Pandas Series 对象
    ser = Series(data, **kwargs)
    # 将 Series 对象中的元素转换为数值类型
    result = to_numeric(ser)

    # 创建预期的 Series 对象
    expected = Series(data)
    # 断言转换后的结果与预期结果相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "columns",
    [
        # 单列
        "a",
        # 多列
        ["a", "b"],
    ],
)
def test_numeric_df_columns(columns):
    # 见 issue gh-14827
    # 创建一个包含不同类型数据的 DataFrame 对象
    df = DataFrame(
        {
            "a": [1.2, decimal.Decimal(3.14), decimal.Decimal("infinity"), "0.1"],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )

    # 创建预期的 DataFrame 对象
    expected = DataFrame({"a": [1.2, 3.14, np.inf, 0.1], "b": [1.0, 2.0, 3.0, 4.0]})
    # 对 DataFrame 中指定的列应用 to_numeric 函数进行类型转换
    df[columns] = df[columns].apply(to_numeric)

    # 断言转换后的 DataFrame 与预期的 DataFrame 相等
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "data,exp_data",
    [
        # 测试包含 Decimal 对象和嵌套列表的情况
        (
            [[decimal.Decimal(3.14), 1.0], decimal.Decimal(1.6), 0.1],
            [[3.14, 1.0], 1.6, 0.1],
        ),
        # 测试包含 Decimal 对象和嵌套数组的情况
        ([np.array([decimal.Decimal(3.14), 1.0]), 0.1], [[3.14, 1.0], 0.1]),
    ],
)
def test_numeric_embedded_arr_likes(data, exp_data):
    # 测试带有嵌套列表和数组的 to_numeric 函数
    # 创建包含 data 的 DataFrame 对象
    df = DataFrame({"a": data})
    # 对 DataFrame 中指定列应用 to_numeric 函数进行类型转换
    df["a"] = df["a"].apply(to_numeric)

    # 创建预期的 DataFrame 对象
    expected = DataFrame({"a": exp_data})
    # 断言转换后的 DataFrame 与预期的 DataFrame 相等
    tm.assert_frame_equal(df, expected)


def test_all_nan():
    # 创建一个包含字符串 "a", "b", "c" 的 Series 对象
    ser = Series(["a", "b", "c"])
    # 将 Series 中的元素转换为数值类型，错误值被强制转换为 NaN
    result = to_numeric(ser, errors="coerce")

    # 创建预期的 Series 对象
    expected = Series([np.nan, np.nan, np.nan])
    # 断言转换后的结果与预期结果相等
    tm.assert_series_equal(result, expected)


def test_type_check(errors):
    # 见 issue gh-11776
    # 创建一个包含整数和字符串的 DataFrame 对象
    df = DataFrame({"a": [1, -3.14, 7], "b": ["4", "5", "6"]})
    # 如果 errors 不为 None，则将其包装在 kwargs 参数中
    kwargs = {"errors": errors} if errors is not None else {}
    # 预期会引发 TypeError 异常，异常信息要匹配 "1-d array"
    with pytest.raises(TypeError, match="1-d array"):
        to_numeric(df, **kwargs)


@pytest.mark.parametrize("val", [1, 1.1, 20001])
def test_scalar(val, signed, transform):
    # 如果 signed 为 True，则将 val 取负数
    val = -val if signed else val
    # 断言转换后的结果与预期结果相等
    assert to_numeric(transform(val)) == float(val)


def test_really_large_scalar(large_val, signed, transform, errors):
    # 见 issue gh-24910
    # 根据是否传入 errors 参数来决定 kwargs 字典的内容，如果传入了 errors 则设置为 {"errors": errors}，否则为空字典 {}
    kwargs = {"errors": errors} if errors is not None else {}

    # 根据 signed 变量的布尔值，决定 val 的值是 -large_val 还是 large_val
    val = -large_val if signed else large_val

    # 对 val 进行 transform 函数的处理
    val = transform(val)

    # 检查 val 是否是字符串类型
    val_is_string = isinstance(val, str)

    # 如果 val 是字符串并且 errors 的值是 None
# 测试一个函数，用于处理包含大量数据的数组，针对不同情况进行验证
def test_really_large_in_arr(large_val, signed, transform, multiple_elts, errors):
    # 见问题 gh-24910
    # 根据是否有错误参数，设置关键字参数字典
    kwargs = {"errors": errors} if errors is not None else {}
    # 根据是否有符号，选择对大值取负或正
    val = -large_val if signed else large_val
    # 对取得的值进行转换操作
    val = transform(val)

    # 添加额外元素到数组中
    extra_elt = "string"
    arr = [val] + multiple_elts * [extra_elt]

    # 检查值是否为字符串类型
    val_is_string = isinstance(val, str)
    # 检查是否进行强制类型转换
    coercing = errors == "coerce"

    # 如果没有错误或者错误为 "raise"，且值为字符串或存在多个元素
    if errors in (None, "raise") and (val_is_string or multiple_elts):
        if val_is_string:
            msg = "Integer out of range. at position 0"
        else:
            msg = 'Unable to parse string "string" at position 1'

        # 使用 pytest 断言抛出 ValueError 异常，并匹配指定消息
        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        # 否则，调用 to_numeric 函数进行转换
        result = to_numeric(arr, **kwargs)

        # 根据情况设置期望值
        exp_val = float(val) if (coercing and val_is_string) else val
        expected = [exp_val]

        # 如果存在多个元素
        if multiple_elts:
            if coercing:
                expected.append(np.nan)
                exp_dtype = float
            else:
                expected.append(extra_elt)
                exp_dtype = object
        else:
            exp_dtype = float if isinstance(exp_val, (int, float)) else object

        # 使用测试工具包中的函数验证结果近似等于期望值的数组
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


# 测试处理包含大量数据的数组的一致性函数
def test_really_large_in_arr_consistent(large_val, signed, multiple_elts, errors):
    # 见问题 gh-24910
    #
    # 即使我们确定需要保留浮点数，也不意味着对后续无法转换为整数的元素应宽容。
    kwargs = {"errors": errors} if errors is not None else {}
    # 创建包含一个元素的数组，该元素为大值的字符串形式
    arr = [str(-large_val if signed else large_val)]

    # 如果存在多个元素，将大值插入数组的开头
    if multiple_elts:
        arr.insert(0, large_val)

    # 如果没有错误或错误为 "raise"
    if errors in (None, "raise"):
        # 计算索引位置，将在该位置上抛出指定消息的 ValueError 异常
        index = int(multiple_elts)
        msg = f"Integer out of range. at position {index}"

        # 使用 pytest 断言抛出 ValueError 异常，并匹配指定消息
        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        # 否则，调用 to_numeric 函数进行转换
        result = to_numeric(arr, **kwargs)
        # 设置期望值为数组中各元素的浮点数形式
        expected = [float(i) for i in arr]
        exp_dtype = float

        # 使用测试工具包中的函数验证结果等于期望值的数组
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


# 使用参数化测试框架 pytest.mark.parametrize 进行多个测试用例的执行
@pytest.mark.parametrize(
    "errors,checker",
    [
        # 测试错误为 "raise" 时的情况，期望抛出指定消息的 ValueError 异常
        ("raise", 'Unable to parse string "fail" at position 0'),
        # 测试错误为 "coerce" 时的情况，期望转换结果为 NaN
        ("coerce", lambda x: np.isnan(x)),
    ],
)
# 测试处理单个标量失败的情况
def test_scalar_fail(errors, checker):
    scalar = "fail"

    # 如果检查器为字符串类型
    if isinstance(checker, str):
        # 使用 pytest 断言抛出 ValueError 异常，并匹配指定消息
        with pytest.raises(ValueError, match=checker):
            to_numeric(scalar, errors=errors)
    else:
        # 否则，使用 assert 断言检查转换结果是否符合期望的条件
        assert checker(to_numeric(scalar, errors=errors))


# 使用参数化测试框架 pytest.mark.parametrize 进行多个测试用例的执行
@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
# 测试对数值类型的数据进行转换和验证
def test_numeric_dtypes(data, transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    # 对数据进行转换操作
    data = transform(data)

    # 调用 to_numeric 函数进行转换
    result = to_numeric(data)
    # 使用测试工具包中的函数验证结果是否等于原始数据
    assert_equal(result, data)


# 使用参数化测试框架 pytest.mark.parametrize 进行多个测试用例的执行
@pytest.mark.parametrize(
    "data,exp",
    [
        # 测试将字符串数组转换为整数数组的情况
        (["1", "2", "3"], np.array([1, 2, 3], dtype="int64")),
        # 测试将字符串数组转换为浮点数数组的情况
        (["1.5", "2.7", "3.4"], np.array([1.5, 2.7, 3.4])),
    ],
)
# 定义一个测试函数，用于检查转换函数对字符串数据的处理是否符合预期
def test_str(data, exp, transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    # 对输入数据进行转换为数值类型
    result = to_numeric(transform(data))

    # 将期望结果转换为数值类型
    expected = transform(exp)
    # 断言结果与期望值相等
    assert_equal(result, expected)


# 定义一个测试函数，验证转换函数对类似日期时间的数据的处理是否正确
def test_datetime_like(tz_naive_fixture, transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    # 创建一个带时区信息的日期时间索引
    idx = pd.date_range("20130101", periods=3, tz=tz_naive_fixture)

    # 对日期时间索引进行转换为数值类型
    result = to_numeric(transform(idx))
    # 将日期时间索引转换为其整数表示，并作为期望结果
    expected = transform(idx.asi8)
    # 断言结果与期望值相等
    assert_equal(result, expected)


# 定义一个测试函数，验证转换函数对时间间隔数据的处理是否正确
def test_timedelta(transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    # 创建一个时间间隔索引
    idx = pd.timedelta_range("1 days", periods=3, freq="D")

    # 对时间间隔索引进行转换为数值类型
    result = to_numeric(transform(idx))
    # 将时间间隔索引转换为其整数表示，并作为期望结果
    expected = transform(idx.asi8)
    # 断言结果与期望值相等
    assert_equal(result, expected)


# 定义一个测试函数，验证转换函数对时期数据的处理是否正确
def test_period(request, transform_assert_equal):
    transform, assert_equal = transform_assert_equal

    # 创建一个时期索引
    idx = pd.period_range("2011-01", periods=3, freq="M", name="")
    # 对时期索引进行转换
    inp = transform(idx)

    # 如果转换结果不是索引类型，应用一个标记，说明不支持期类型转换
    if not isinstance(inp, Index):
        request.applymarker(
            pytest.mark.xfail(reason="Missing PeriodDtype support in to_numeric")
        )
    # 对转换后的结果进行数值类型转换
    result = to_numeric(inp)
    # 将时期索引转换为其整数表示，并作为期望结果
    expected = transform(idx.asi8)
    # 断言结果与期望值相等
    assert_equal(result, expected)


# 定义一个参数化测试，验证在处理非可哈希对象时转换函数的行为是否符合预期
@pytest.mark.parametrize(
    "errors,expected",
    [
        ("raise", "Invalid object type at position 0"),  # 如果期望抛出特定异常则设置对应的异常信息
        ("coerce", Series([np.nan, 1.0, np.nan])),  # 否则期望得到一个特定的 Series 结果
    ],
)
def test_non_hashable(errors, expected):
    # 创建一个包含多种数据类型的 Series 对象
    ser = Series([[10.0, 2], 1.0, "apple"])

    # 如果期望结果是字符串，则期望在执行转换时抛出指定的类型错误异常
    if isinstance(expected, str):
        with pytest.raises(TypeError, match=expected):
            to_numeric(ser, errors=errors)
    else:
        # 否则，期望转换后的结果与指定的 Series 结果相等
        result = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, expected)


# 定义一个测试函数，验证当提供无效的下转换方法时转换函数是否能够正确抛出异常
def test_downcast_invalid_cast():
    # 创建一个包含多种数据类型的列表
    data = ["1", 2, 3]
    invalid_downcast = "unsigned-integer"
    msg = "invalid downcasting method provided"

    # 期望在执行转换时抛出指定的值错误异常
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, downcast=invalid_downcast)


# 定义一个测试函数，验证当提供无效的错误处理值时转换函数是否能够正确抛出异常
def test_errors_invalid_value():
    # 创建一个包含多种数据类型的列表
    data = ["1", 2, 3]
    invalid_error_value = "invalid"
    msg = "invalid error value specified"

    # 期望在执行转换时抛出指定的值错误异常
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, errors=invalid_error_value)


# 定义一个参数化测试，验证转换函数在基本情况下对数据的下转换行为是否符合预期
@pytest.mark.parametrize(
    "data",
    [
        ["1", 2, 3],  # 包含字符串和整数的列表
        [1, 2, 3],  # 只包含整数的列表
        np.array(["1970-01-02", "1970-01-03", "1970-01-04"], dtype="datetime64[D]"),  # 包含日期时间的 NumPy 数组
    ],
)
@pytest.mark.parametrize(
    "kwargs,exp_dtype",
    [
        # 基本的函数测试
        ({}, np.int64),
        ({"downcast": None}, np.int64),
        # 支持 np.float32 以下的下转换选项
        ({"downcast": "float"}, np.dtype(np.float32).char),
        # 基本的数据类型支持
        ({"downcast": "unsigned"}, np.dtype(np.typecodes["UnsignedInteger"][0])),
    ],
)
def test_downcast_basic(data, kwargs, exp_dtype):
    # 期望在执行转换时得到指定的数据类型结果
    result = to_numeric(data, **kwargs)
    # 创建一个预期的 NumPy 数组，内容为 [1, 2, 3]，数据类型为 exp_dtype
    expected = np.array([1, 2, 3], dtype=exp_dtype)
    # 使用测试框架中的函数，比较 result 和 expected 是否相等的 NumPy 数组
    tm.assert_numpy_array_equal(result, expected)
@pytest.mark.parametrize("signed_downcast", ["integer", "signed"])
@pytest.mark.parametrize(
    "data",
    [
        ["1", 2, 3],
        [1, 2, 3],
        np.array(["1970-01-02", "1970-01-03", "1970-01-04"], dtype="datetime64[D]"),
    ],
)
def test_signed_downcast(data, signed_downcast):
    # 用于测试数据类型转换函数 to_numeric() 的 signed_downcast 参数
    # 期望的结果是将 data 转换为最小整数类型 smallest_int_dtype，并进行比较
    smallest_int_dtype = np.dtype(np.typecodes["Integer"][0])
    expected = np.array([1, 2, 3], dtype=smallest_int_dtype)

    res = to_numeric(data, downcast=signed_downcast)
    tm.assert_numpy_array_equal(res, expected)


def test_ignore_downcast_neg_to_unsigned():
    # 测试当 downcast 参数设置为 "unsigned" 时的数据类型转换函数 to_numeric()
    # 由于数据中包含负数，因此不能转换为无符号整数
    data = ["-1", 2, 3]
    expected = np.array([-1, 2, 3], dtype=np.int64)

    res = to_numeric(data, downcast="unsigned")
    tm.assert_numpy_array_equal(res, expected)


# Warning in 32 bit platforms
@pytest.mark.parametrize("downcast", ["integer", "signed", "unsigned"])
@pytest.mark.parametrize(
    "data,expected",
    [
        (["1.1", 2, 3], np.array([1.1, 2, 3], dtype=np.float64)),
        (
            [10000.0, 20000, 3000, 40000.36, 50000, 50000.00],
            np.array(
                [10000.0, 20000, 3000, 40000.36, 50000, 50000.00], dtype=np.float64
            ),
        ),
    ],
)
def test_ignore_downcast_cannot_convert_float(data, expected, downcast):
    # 测试当 downcast 参数为 "integer"、"signed" 或 "unsigned" 时的数据类型转换函数 to_numeric()
    # 由于数据中包含浮点数，因此不能转换为整数类型
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "downcast,expected_dtype",
    [("integer", np.int16), ("signed", np.int16), ("unsigned", np.uint16)],
)
def test_downcast_not8bit(downcast, expected_dtype):
    # 测试当 downcast 参数为 "integer"、"signed" 或 "unsigned" 时的数据类型转换函数 to_numeric()
    # 检查最小整数类型不一定是 np.(u)int8
    data = ["256", 257, 258]

    expected = np.array([256, 257, 258], dtype=expected_dtype)
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "dtype,downcast,min_max",
    [
        # 定义一个包含多个元组的列表，每个元组描述了一种数据类型的信息
        ("int8", "integer", [iinfo(np.int8).min, iinfo(np.int8).max]),
    
        ("int16", "integer", [iinfo(np.int16).min, iinfo(np.int16).max]),
    
        ("int32", "integer", [iinfo(np.int32).min, iinfo(np.int32).max]),
    
        ("int64", "integer", [iinfo(np.int64).min, iinfo(np.int64).max]),
    
        ("uint8", "unsigned", [iinfo(np.uint8).min, iinfo(np.uint8).max]),
    
        ("uint16", "unsigned", [iinfo(np.uint16).min, iinfo(np.uint16).max]),
    
        ("uint32", "unsigned", [iinfo(np.uint32).min, iinfo(np.uint32).max]),
    
        ("uint64", "unsigned", [iinfo(np.uint64).min, iinfo(np.uint64).max]),
    
        ("int16", "integer", [iinfo(np.int8).min, iinfo(np.int8).max + 1]),
    
        ("int32", "integer", [iinfo(np.int16).min, iinfo(np.int16).max + 1]),
    
        ("int64", "integer", [iinfo(np.int32).min, iinfo(np.int32).max + 1]),
    
        ("int16", "integer", [iinfo(np.int8).min - 1, iinfo(np.int16).max]),
    
        ("int32", "integer", [iinfo(np.int16).min - 1, iinfo(np.int32).max]),
    
        ("int64", "integer", [iinfo(np.int32).min - 1, iinfo(np.int64).max]),
    
        ("uint16", "unsigned", [iinfo(np.uint8).min, iinfo(np.uint8).max + 1]),
    
        ("uint32", "unsigned", [iinfo(np.uint16).min, iinfo(np.uint16).max + 1]),
    
        ("uint64", "unsigned", [iinfo(np.uint32).min, iinfo(np.uint32).max + 1]),
    ],
def test_downcast_limits(dtype, downcast, min_max):
    # see gh-14404: test the limits of each downcast.
    series = to_numeric(Series(min_max), downcast=downcast)
    # 断言转换后的数据类型是否符合预期
    assert series.dtype == dtype


def test_downcast_float64_to_float32():
    # GH-43693: Check float64 preservation when >= 16,777,217
    series = Series([16777217.0, np.finfo(np.float64).max, np.nan], dtype=np.float64)
    result = to_numeric(series, downcast="float")
    # 断言原始系列和转换后的结果系列数据类型相同
    assert series.dtype == result.dtype


def test_downcast_uint64():
    # see gh-14422:
    # BUG: to_numeric doesn't work uint64 numbers
    ser = Series([0, 9223372036854775808])
    result = to_numeric(ser, downcast="unsigned")
    expected = Series([0, 9223372036854775808], dtype=np.uint64)
    # 断言转换后的结果与预期结果相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data,exp_data",
    [
        (
            [200, 300, "", "NaN", 30000000000000000000],
            [200, 300, np.nan, np.nan, 30000000000000000000],
        ),
        (
            ["12345678901234567890", "1234567890", "ITEM"],
            [12345678901234567890, 1234567890, np.nan],
        ),
    ],
)
def test_coerce_uint64_conflict(data, exp_data):
    # see gh-17007 and gh-17125
    #
    # Still returns float despite the uint64-nan conflict,
    # which would normally force the casting to object.
    result = to_numeric(Series(data), errors="coerce")
    expected = Series(exp_data, dtype=float)
    # 断言转换后的结果与预期结果相等
    tm.assert_series_equal(result, expected)


def test_non_coerce_uint64_conflict():
    # see gh-17007 and gh-17125
    #
    # For completeness.
    ser = Series(["12345678901234567890", "1234567890", "ITEM"])

    with pytest.raises(ValueError, match="Unable to parse string"):
        # 断言抛出特定的 ValueError 异常
        to_numeric(ser, errors="raise")


@pytest.mark.parametrize("dc1", ["integer", "float", "unsigned"])
@pytest.mark.parametrize("dc2", ["integer", "float", "unsigned"])
def test_downcast_empty(dc1, dc2):
    # GH32493

    tm.assert_numpy_array_equal(
        # 断言两个空数组经过不同的 downcast 后结果相等
        to_numeric([], downcast=dc1),
        to_numeric([], downcast=dc2),
        check_dtype=False,
    )


def test_failure_to_convert_uint64_string_to_NaN():
    # GH 32394
    result = to_numeric("uint64", errors="coerce")
    # 断言转换后的结果为 NaN
    assert np.isnan(result)

    ser = Series([32, 64, np.nan])
    result = to_numeric(Series(["32", "64", "uint64"]), errors="coerce")
    # 断言转换后的结果系列与预期系列相等
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize(
    "strrep",
    # 以下是一个包含多个字符串的列表，每个字符串代表一个数值
    [
        "243.164",
        "245.968",
        "249.585",
        "259.745",
        "265.742",
        "272.567",
        "279.196",
        "280.366",
        "275.034",
        "271.351",
        "272.889",
        "270.627",
        "280.828",
        "290.383",
        "308.153",
        "319.945",
        "336.0",
        "344.09",
        "351.385",
        "356.178",
        "359.82",
        "361.03",
        "367.701",
        "380.812",
        "387.98",
        "391.749",
        "391.171",
        "385.97",
        "385.345",
        "386.121",
        "390.996",
        "399.734",
        "413.073",
        "421.532",
        "430.221",
        "437.092",
        "439.746",
        "446.01",
        "451.191",
        "460.463",
        "469.779",
        "472.025",
        "479.49",
        "474.864",
        "467.54",
        "471.978",
    ]
# GH 31364
# 测试精度和浮点数转换
def test_precision_float_conversion(strrep):
    result = to_numeric(strrep)  # 调用 to_numeric 函数将字符串转换为数值

    assert result == float(strrep)  # 断言转换后的结果与原始浮点数字符串相等


@pytest.mark.parametrize(
    "values, expected",
    [
        (["1", "2", None], Series([1, 2, np.nan], dtype="Int64")),  # 测试处理包含空值的字符串数组的转换
        (["1", "2", "3"], Series([1, 2, 3], dtype="Int64")),  # 测试处理整数字符串数组的转换
        (["1", "2", 3], Series([1, 2, 3], dtype="Int64")),  # 测试处理混合类型数组的转换
        (["1", "2", 3.5], Series([1, 2, 3.5], dtype="Float64")),  # 测试处理包含浮点数的字符串数组的转换
        (["1", None, 3.5], Series([1, np.nan, 3.5], dtype="Float64")),  # 测试处理包含空值和浮点数的字符串数组的转换
        (["1", "2", "3.5"], Series([1, 2, 3.5], dtype="Float64")),  # 测试处理包含浮点数字符串数组的转换
    ],
)
def test_to_numeric_from_nullable_string(values, nullable_string_dtype, expected):
    # https://github.com/pandas-dev/pandas/issues/37262
    # 根据 GitHub 上的 issue 讨论，测试处理可空字符串转换为数值的功能
    s = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(s)
    tm.assert_series_equal(result, expected)  # 断言转换后的结果与期望的 Series 对象相等


def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype):
    # GH#52146
    # 测试强制处理可空字符串转换时的功能
    values = ["a", "1"]
    ser = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(ser, errors="coerce")
    expected = Series([pd.NA, 1], dtype="Int64")
    tm.assert_series_equal(result, expected)  # 断言转换后的结果与期望的 Series 对象相等


@pytest.mark.parametrize(
    "data, input_dtype, downcast, expected_dtype",
    (
        ([1, 1], "Int64", "integer", "Int8"),  # 测试整数类型数据的下转型为 Int8 类型
        ([1.0, pd.NA], "Float64", "integer", "Int8"),  # 测试浮点数数据的下转型为 Int8 类型
        ([1.0, 1.1], "Float64", "integer", "Float64"),  # 测试浮点数数据的下转型为 Float64 类型
        ([1, pd.NA], "Int64", "integer", "Int8"),  # 测试整数类型数据的下转型为 Int8 类型
        ([450, 300], "Int64", "integer", "Int16"),  # 测试整数类型数据的下转型为 Int16 类型
        ([1, 1], "Float64", "integer", "Int8"),  # 测试浮点数数据的下转型为 Int8 类型
        ([np.iinfo(np.int64).max - 1, 1], "Int64", "integer", "Int64"),  # 测试整数类型数据的下转型为 Int64 类型
        ([1, 1], "Int64", "signed", "Int8"),  # 测试整数类型数据的有符号下转型为 Int8 类型
        ([1.0, 1.0], "Float32", "signed", "Int8"),  # 测试浮点数数据的有符号下转型为 Int8 类型
        ([1.0, 1.1], "Float64", "signed", "Float64"),  # 测试浮点数数据的有符号下转型为 Float64 类型
        ([1, pd.NA], "Int64", "signed", "Int8"),  # 测试整数类型数据的有符号下转型为 Int8 类型
        ([450, -300], "Int64", "signed", "Int16"),  # 测试整数类型数据的有符号下转型为 Int16 类型
        ([np.iinfo(np.uint64).max - 1, 1], "UInt64", "signed", "UInt64"),  # 测试无符号整数类型数据的有符号下转型为 UInt64 类型
        ([1, 1], "Int64", "unsigned", "UInt8"),  # 测试整数类型数据的无符号下转型为 UInt8 类型
        ([1.0, 1.0], "Float32", "unsigned", "UInt8"),  # 测试浮点数数据的无符号下转型为 UInt8 类型
        ([1.0, 1.1], "Float64", "unsigned", "Float64"),  # 测试浮点数数据的无符号下转型为 Float64 类型
        ([1, pd.NA], "Int64", "unsigned", "UInt8"),  # 测试整数类型数据的无符号下转型为 UInt8 类型
        ([450, -300], "Int64", "unsigned", "Int64"),  # 测试整数类型数据的无符号下转型为 Int64 类型
        ([-1, -1], "Int32", "unsigned", "Int32"),  # 测试整数类型数据的无符号下转型为 Int32 类型
        ([1, 1], "Float64", "float", "Float32"),  # 测试浮点数数据的浮点型转型为 Float32 类型
        ([1, 1.1], "Float64", "float", "Float32"),  # 测试浮点数数据的浮点型转型为 Float32 类型
        ([1, 1], "Float32", "float", "Float32"),  # 测试浮点数数据的浮点型转型为 Float32 类型
        ([1, 1.1], "Float32", "float", "Float32"),  # 测试浮点数数据的浮点型转型为 Float32 类型
    ),
)
def test_downcast_nullable_numeric(data, input_dtype, downcast, expected_dtype):
    arr = pd.array(data, dtype=input_dtype)
    result = to_numeric(arr, downcast=downcast)  # 调用 to_numeric 函数进行数据类型转换
    expected = pd.array(data, dtype=expected_dtype)
    tm.assert_extension_array_equal(result, expected)  # 断言转换后的结果与期望的 ExtensionArray 对象相等


def test_downcast_nullable_mask_is_copied():
    # GH38974
    # 测试复制带有空值的可空数组进行下转型时的功能
    arr = pd.array([1, 2, pd.NA], dtype="Int64")

    result = to_numeric(arr, downcast="integer")  # 调用 to_numeric 函数将数组下转型为整数类型
    expected = pd.array([1, 2, pd.NA], dtype="Int8")
    # 使用测试工具函数来比较 `result` 和 `expected` 是否相等
    tm.assert_extension_array_equal(result, expected)
    
    # 将数组 `arr` 的第二个元素设置为缺失值 `pd.NA`，预期不会修改 `result`
    # 再次使用测试工具函数来比较 `result` 和 `expected` 是否相等
    tm.assert_extension_array_equal(result, expected)
# 测试函数，用于验证科学计数法转换功能
def test_to_numeric_scientific_notation():
    # GH 15898：GitHub 上的 issue 编号，用于跟踪和管理相关问题
    result = to_numeric("1.7e+308")  # 将字符串转换为数值类型
    expected = np.float64(1.7e308)   # 期望的数值类型为 np.float64
    assert result == expected        # 断言结果与期望相符


# 使用参数化测试，验证大浮点数不会降级为 float32 类型的情况
@pytest.mark.parametrize("val", [9876543210.0, 2.0**128])
def test_to_numeric_large_float_not_downcast_to_float_32(val):
    # GH 19729：GitHub 上的 issue 编号，用于跟踪和管理相关问题
    expected = Series([val])                               # 创建一个包含期望值的 Series 对象
    result = to_numeric(expected, downcast="float")        # 执行 to_numeric 函数进行转换
    tm.assert_series_equal(result, expected)               # 断言结果与期望的 Series 对象相等


# 使用参数化测试，验证不同数据类型下的 dtype_backend 设置对转换结果的影响
@pytest.mark.parametrize(
    "val, dtype", [(1, "Int64"), (1.5, "Float64"), (True, "boolean")]
)
def test_to_numeric_dtype_backend(val, dtype):
    # GH#50505：GitHub 上的 issue 编号，用于跟踪和管理相关问题
    ser = Series([val], dtype=object)                      # 创建一个对象 dtype 为 object 的 Series 对象
    result = to_numeric(ser, dtype_backend="numpy_nullable")  # 使用 numpy_nullable 进行数据类型转换
    expected = Series([val], dtype=dtype)                  # 创建一个期望的 Series 对象
    tm.assert_series_equal(result, expected)               # 断言结果与期望的 Series 对象相等


# 使用参数化测试，验证 dtype_backend 设置对缺失值处理的影响
@pytest.mark.parametrize(
    "val, dtype",
    [
        (1, "Int64"),
        (1.5, "Float64"),
        (True, "boolean"),
        (1, "int64[pyarrow]"),
        (1.5, "float64[pyarrow]"),
        (True, "bool[pyarrow]"),
    ],
)
def test_to_numeric_dtype_backend_na(val, dtype):
    # GH#50505：GitHub 上的 issue 编号，用于跟踪和管理相关问题
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")                     # 如果使用 pyarrow，则检查并导入相应的模块
        dtype_backend = "pyarrow"
    else:
        dtype_backend = "numpy_nullable"                   # 否则使用 numpy_nullable
    ser = Series([val, None], dtype=object)                # 创建一个对象 dtype 为 object 的 Series 对象，包含一个 None 值
    result = to_numeric(ser, dtype_backend=dtype_backend)  # 执行数据类型转换
    expected = Series([val, pd.NA], dtype=dtype)           # 创建一个期望的 Series 对象，包含 pd.NA
    tm.assert_series_equal(result, expected)               # 断言结果与期望的 Series 对象相等


# 使用参数化测试，验证 dtype_backend 和 downcast 设置对数据类型转换的影响
@pytest.mark.parametrize(
    "val, dtype, downcast",
    [
        (1, "Int8", "integer"),
        (1.5, "Float32", "float"),
        (1, "Int8", "signed"),
        (1, "int8[pyarrow]", "integer"),
        (1.5, "float[pyarrow]", "float"),
        (1, "int8[pyarrow]", "signed"),
    ],
)
def test_to_numeric_dtype_backend_downcasting(val, dtype, downcast):
    # GH#50505：GitHub 上的 issue 编号，用于跟踪和管理相关问题
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")                     # 如果使用 pyarrow，则检查并导入相应的模块
        dtype_backend = "pyarrow"
    else:
        dtype_backend = "numpy_nullable"                   # 否则使用 numpy_nullable
    ser = Series([val, None], dtype=object)                # 创建一个对象 dtype 为 object 的 Series 对象，包含一个 None 值
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast=downcast)  # 执行数据类型转换
    expected = Series([val, pd.NA], dtype=dtype)           # 创建一个期望的 Series 对象，包含 pd.NA
    tm.assert_series_equal(result, expected)               # 断言结果与期望的 Series 对象相等


# 使用参数化测试，验证对 unsigned 类型数据进行 downcast 的影响
@pytest.mark.parametrize(
    "smaller, dtype_backend",
    [["UInt8", "numpy_nullable"], ["uint8[pyarrow]", "pyarrow"]],
)
def test_to_numeric_dtype_backend_downcasting_uint(smaller, dtype_backend):
    # GH#50505：GitHub 上的 issue 编号，用于跟踪和管理相关问题
    if dtype_backend == "pyarrow":
        pytest.importorskip("pyarrow")                     # 如果使用 pyarrow，则检查并导入相应的模块
    ser = Series([1, pd.NA], dtype="UInt64")               # 创建一个 UInt64 类型的 Series 对象，包含一个 pd.NA 值
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast="unsigned")  # 执行数据类型转换
    expected = Series([1, pd.NA], dtype=smaller)           # 创建一个期望的 Series 对象，包含小数据类型
    tm.assert_series_equal(result, expected)               # 断言结果与期望的 Series 对象相等


# 使用参数化测试，验证已经是 nullable 的数据类型转换不会受到影响
@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        "UInt64",
        "Float64",
        "boolean",
        "int64[pyarrow]",
        "uint64[pyarrow]",
        "float64[pyarrow]",
        "bool[pyarrow]",
    ],
)
def test_to_numeric_dtype_backend_already_nullable(dtype):
    # GH#50505：GitHub 上的 issue 编号，用于跟踪和管理相关问题
    # 如果 "pyarrow" 在 dtype 中存在，则检查是否可以导入 "pyarrow"，否则跳过测试
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    
    # 创建一个 Series 对象，包含两个元素：1 和 pd.NA（缺失值），并指定数据类型为 dtype
    ser = Series([1, pd.NA], dtype=dtype)
    
    # 调用 to_numeric 函数，将 Series 对象转换为数值类型，使用 numpy_nullable 作为后端
    result = to_numeric(ser, dtype_backend="numpy_nullable")
    
    # 创建一个期望的 Series 对象，与 result 结果进行比较
    expected = Series([1, pd.NA], dtype=dtype)
    
    # 使用测试框架中的 assert_series_equal 函数，断言 result 和 expected 应当相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试将 Series 对象转换为数值类型时的后端错误情况
def test_to_numeric_dtype_backend_error(dtype_backend):
    # 创建一个包含字符串的 Series 对象
    ser = Series(["a", "b", ""])
    # 创建一个期望的 Series 对象，与原始 Series 对象一致
    expected = ser.copy()
    # 使用 pytest 来验证调用 to_numeric 函数时是否会引发 ValueError 异常，且异常信息匹配指定字符串
    with pytest.raises(ValueError, match="Unable to parse string"):
        to_numeric(ser, dtype_backend=dtype_backend)

    # 调用 to_numeric 函数，使用 "coerce" 错误处理模式
    result = to_numeric(ser, dtype_backend=dtype_backend, errors="coerce")
    
    # 根据 dtype_backend 的值确定期望的 Series 对象的数据类型
    if dtype_backend == "pyarrow":
        dtype = "double[pyarrow]"
    else:
        dtype = "Float64"
    
    # 创建一个期望的 Series 对象，包含 np.nan 值，并指定数据类型
    expected = Series([np.nan, np.nan, np.nan], dtype=dtype)
    
    # 使用 assert_series_equal 函数验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试传递给 to_numeric 函数的 dtype_backend 参数为无效值的情况
def test_invalid_dtype_backend():
    # 创建一个包含整数的 Series 对象
    ser = Series([1, 2, 3])
    # 定义一个错误信息字符串
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    # 使用 pytest 来验证调用 to_numeric 函数时是否会引发 ValueError 异常，且异常信息匹配指定字符串
    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, dtype_backend="numpy")


# 定义一个测试函数，用于测试在 pyarrow 后端情况下使用 "coerce" 错误处理模式的情况
def test_coerce_pyarrow_backend():
    # 引入 pytest，如果 pyarrow 不可用，则跳过这个测试
    pa = pytest.importorskip("pyarrow")
    # 创建一个包含字符的 Series 对象，使用 ArrowDtype 指定数据类型
    ser = Series(list("12x"), dtype=ArrowDtype(pa.string()))
    # 调用 to_numeric 函数，使用 "coerce" 错误处理模式和 pyarrow 后端
    result = to_numeric(ser, errors="coerce", dtype_backend="pyarrow")
    # 创建一个期望的 Series 对象，包含整数和 None 值，并指定数据类型
    expected = Series([1, 2, None], dtype=ArrowDtype(pa.int64()))
    # 使用 assert_series_equal 函数验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```