# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_ints.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需的模块和库
from io import StringIO  # 导入StringIO用于在内存中操作字符串
import numpy as np  # 导入numpy库
import pytest  # 导入pytest库用于单元测试

from pandas import (  # 导入pandas库中的DataFrame和Series
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入pandas内部测试工具模块

# 忽略特定警告的pytest标记
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")  # pytest标记，用于处理pyarrow预期失败的情况
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")    # pytest标记，用于跳过pyarrow相关测试

# 测试函数：测试整数类型转换
def test_int_conversion(all_parsers):
    data = """A,B
1.0,1
2.0,2
3.0,3
"""
    parser = all_parsers  # 使用传入的解析器对象
    result = parser.read_csv(StringIO(data))  # 使用解析器读取CSV数据

    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=["A", "B"])  # 预期的DataFrame结果
    tm.assert_frame_equal(result, expected)  # 使用测试工具模块验证结果DataFrame是否与预期相等


# 参数化测试函数：测试布尔类型解析
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            "A,B\nTrue,1\nFalse,2\nTrue,3",
            {},
            [[True, 1], [False, 2], [True, 3]],
        ),
        (
            "A,B\nYES,1\nno,2\nyes,3\nNo,3\nYes,3",
            {"true_values": ["yes", "Yes", "YES"], "false_values": ["no", "NO", "No"]},
            [[True, 1], [False, 2], [True, 3], [False, 3], [True, 3]],
        ),
        (
            "A,B\nTRUE,1\nFALSE,2\nTRUE,3",
            {},
            [[True, 1], [False, 2], [True, 3]],
        ),
        (
            "A,B\nfoo,bar\nbar,foo",
            {"true_values": ["foo"], "false_values": ["bar"]},
            [[True, False], [False, True]],
        ),
    ],
)
def test_parse_bool(all_parsers, data, kwargs, expected):
    parser = all_parsers  # 使用传入的解析器对象
    result = parser.read_csv(StringIO(data), **kwargs)  # 使用解析器读取CSV数据，传入额外的参数

    expected = DataFrame(expected, columns=["A", "B"])  # 预期的DataFrame结果
    tm.assert_frame_equal(result, expected)  # 使用测试工具模块验证结果DataFrame是否与预期相等


# 测试函数：测试大整数在浮点精度以上的情况
def test_parse_integers_above_fp_precision(all_parsers):
    data = """Numbers
17007000002000191
17007000002000191
17007000002000191
17007000002000191
17007000002000192
17007000002000192
17007000002000192
17007000002000192
17007000002000192
17007000002000194"""
    parser = all_parsers  # 使用传入的解析器对象
    result = parser.read_csv(StringIO(data))  # 使用解析器读取CSV数据

    expected = DataFrame(  # 预期的DataFrame结果，包含大整数列
        {
            "Numbers": [
                17007000002000191,
                17007000002000191,
                17007000002000191,
                17007000002000191,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000194,
            ]
        }
    )
    tm.assert_frame_equal(result, expected)  # 使用测试工具模块验证结果DataFrame是否与预期相等


# 参数化测试函数：测试整数溢出的Bug
@pytest.mark.parametrize("sep", [" ", r"\s+"])
def test_integer_overflow_bug(all_parsers, sep):
    # see gh-2601
    data = "65248E10 11\n55555E55 22\n"
    parser = all_parsers  # 使用传入的解析器对象
    if parser.engine == "pyarrow" and sep != " ":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):  # 预期抛出特定的值错误异常
            parser.read_csv(StringIO(data), header=None, sep=sep)  # 使用解析器读取CSV数据，传入额外的参数
        return
    # 使用 parser 对象读取 CSV 数据，数据来源是内存中的字符串，不含表头，使用指定的分隔符进行解析
    result = parser.read_csv(StringIO(data), header=None, sep=sep)
    # 预期的数据框 DataFrame，包含两行两列，值分别为 6.5248e14, 11 和 5.5555e59, 22
    expected = DataFrame([[6.5248e14, 11], [5.5555e59, 22]])
    # 断言两个数据框是否相等
    tm.assert_frame_equal(result, expected)
# 测试处理整数溢出问题，涉及 GitHub issue 2599
def test_int64_min_issues(all_parsers):
    # 选择所有解析器中的一个
    parser = all_parsers
    # 定义包含逗号分隔数据的字符串
    data = "A,B\n0,0\n0,"
    # 使用解析器读取 CSV 数据并返回结果
    result = parser.read_csv(StringIO(data))

    # 期望的数据帧，包含列 A 和 B，其中 B 列的第二行是 NaN
    expected = DataFrame({"A": [0, 0], "B": [0, np.nan]})
    # 断言读取结果与期望结果相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conv", [None, np.int64, np.uint64])
def test_int64_overflow(all_parsers, conv, request):
    # 包含超出 int64 范围的 ID 数据的字符串
    data = """ID
00013007854817840016671868
00013007854817840016749251
00013007854817840016754630
00013007854817840016781876
00013007854817840017028824
00013007854817840017963235
00013007854817840018860166"""
    # 选择所有解析器中的一个
    parser = all_parsers

    if conv is None:
        # 如果未指定转换器，使用默认设置进行解析
        # 由于 ID 超出 UINT64_MAX，因此将返回 object 类型的数据帧
        if parser.engine == "pyarrow":
            # 如果使用 pyarrow 引擎，标记为预期失败（预期解析为 float64）
            mark = pytest.mark.xfail(reason="parses to float64")
            request.applymarker(mark)

        # 使用解析器读取 CSV 数据并返回结果
        result = parser.read_csv(StringIO(data))
        # 期望的数据帧，只包含一列 ID
        expected = DataFrame(
            [
                "00013007854817840016671868",
                "00013007854817840016749251",
                "00013007854817840016754630",
                "00013007854817840016781876",
                "00013007854817840017028824",
                "00013007854817840017963235",
                "00013007854817840018860166",
            ],
            columns=["ID"],
        )
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)
    else:
        # 如果指定了转换器（conv），尝试将 ID 转换为 int64 或 uint64 会引发 OverflowError
        # 设置错误消息字符串
        msg = "|".join(
            [
                "Python int too large to convert to C long",
                "long too big to convert",
                "int too big to convert",
            ]
        )
        err = OverflowError
        if parser.engine == "pyarrow":
            # 如果使用 pyarrow 引擎，抛出 ValueError，因为 'converters' 选项不支持
            err = ValueError
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"

        # 使用 pytest 断言引发特定错误和匹配的错误消息
        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), converters={"ID": conv})


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "val", [np.iinfo(np.uint64).max, np.iinfo(np.int64).max, np.iinfo(np.int64).min]
)
def test_int64_uint64_range(all_parsers, val):
    # 这些数值正好位于 int64-uint64 范围内，应解析为字符串
    # 选择所有解析器中的一个
    parser = all_parsers
    # 使用解析器读取包含单个数值的 CSV 数据并返回结果
    result = parser.read_csv(StringIO(str(val)), header=None)

    # 期望的数据帧，包含一行，数值为 val
    expected = DataFrame([val])
    # 断言读取结果与期望结果相等
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "val", [np.iinfo(np.uint64).max + 1, np.iinfo(np.int64).min - 1]
)
def test_outside_int64_uint64_range(all_parsers, val):
    # 这些数值刚好位于 int64-uint64 范围之外，应解析为字符串
    # 选择所有解析器中的一个
    parser = all_parsers
    # 使用解析器读取包含单个数值的 CSV 数据并返回结果
    result = parser.read_csv(StringIO(str(val)), header=None)
    # 创建一个包含单个字符串值的 DataFrame，并将其转换为字符串形式
    expected = DataFrame([str(val)])
    # 使用测试工具包中的函数来比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest 的 xfail 标记，标记该测试用例预期失败（在 pyarrow 下得到 float64 而不是 object 类型）
@xfail_pyarrow  # gets float64 dtype instead of object
# 使用 pytest 的 parametrize 标记，参数化测试数据 exp_data 包含两组字符串列表，每组列表包含两个字符串：一个负数和一个 2^63 次方
@pytest.mark.parametrize("exp_data", [[str(-1), str(2**63)], [str(2**63), str(-1)]])
def test_numeric_range_too_wide(all_parsers, exp_data):
    # 没有数值类型可以同时容纳负数和 uint64 值，因此它们应被转换为字符串。
    # 获取测试用例中的解析器对象
    parser = all_parsers
    # 将 exp_data 转换为一个包含换行符的字符串
    data = "\n".join(exp_data)
    # 用 exp_data 创建预期的 DataFrame 对象
    expected = DataFrame(exp_data)

    # 使用 parser 对象读取 CSV 数据，并返回结果
    result = parser.read_csv(StringIO(data), header=None)
    # 使用 pytest 的 assert_frame_equal 断言函数比较结果和预期的 DataFrame 对象
    tm.assert_frame_equal(result, expected)


def test_integer_precision(all_parsers):
    # Gh 7072
    # 定义包含两行数据的 CSV 格式字符串 s
    s = """1,1;0;0;0;1;1;3844;3844;3844;1;1;1;1;1;1;0;0;1;1;0;0,,,4321583677327450765
5,1;0;0;0;1;1;843;843;843;1;1;1;1;1;1;0;0;1;1;0;0,64.0,;,4321113141090630389"""
    # 获取测试用例中的解析器对象
    parser = all_parsers
    # 使用 parser 对象读取 CSV 数据，并从结果中选择第五列
    result = parser.read_csv(StringIO(s), header=None)[4]
    # 创建预期的 Series 对象，包含两个大整数作为元素
    expected = Series([4321583677327450765, 4321113141090630389], name=4)
    # 使用 pytest 的 assert_series_equal 断言函数比较结果和预期的 Series 对象
    tm.assert_series_equal(result, expected)
```