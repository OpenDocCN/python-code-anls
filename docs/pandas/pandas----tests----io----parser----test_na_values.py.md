# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_na_values.py`

```
"""
Tests that NA values are properly handled during
parsing for all of the parsers defined in parsers.py
"""

# 导入所需的模块和库
from io import StringIO  # 导入字符串IO模块中的StringIO类

import numpy as np  # 导入NumPy库，并用np作为别名
import pytest  # 导入pytest测试框架

from pandas._libs.parsers import STR_NA_VALUES  # 从pandas._libs.parsers模块导入STR_NA_VALUES常量

from pandas import (  # 从pandas库中导入DataFrame、Index、MultiIndex类
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm  # 导入pandas._testing模块，并使用tm作为别名

# 设置pytest标记，忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 标记使用xfail_pyarrow装饰器，用于pyarrow引擎的测试
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
# 标记使用skip_pyarrow装饰器，用于跳过pyarrow引擎的测试
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


# 测试函数：测试处理字符串类型的缺失值
def test_string_nas(all_parsers):
    parser = all_parsers  # 获取传入的解析器对象
    data = """A,B,C
a,b,c
d,,f
,g,h
"""
    # 使用给定的解析器解析CSV数据，返回DataFrame对象
    result = parser.read_csv(StringIO(data))
    # 期望的DataFrame对象，包含预期的数据及缺失值
    expected = DataFrame(
        [["a", "b", "c"], ["d", np.nan, "f"], [np.nan, "g", "h"]],
        columns=["A", "B", "C"],
    )
    # 根据解析器类型，在需要的位置设置None值
    if parser.engine == "pyarrow":
        expected.loc[2, "A"] = None
        expected.loc[1, "B"] = None
    # 使用测试框架断言实际输出与期望输出相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试检测字符串型缺失值
def test_detect_string_na(all_parsers):
    parser = all_parsers  # 获取传入的解析器对象
    data = """A,B
foo,bar
NA,baz
NaN,nan
"""
    # 期望的DataFrame对象，包含预期的数据及缺失值
    expected = DataFrame(
        [["foo", "bar"], [np.nan, "baz"], [np.nan, np.nan]], columns=["A", "B"]
    )
    # 根据解析器类型，在需要的位置设置None值
    if parser.engine == "pyarrow":
        expected.loc[[1, 2], "A"] = None
        expected.loc[2, "B"] = None
    # 使用给定的解析器解析CSV数据，返回DataFrame对象
    result = parser.read_csv(StringIO(data))
    # 使用测试框架断言实际输出与期望输出相等
    tm.assert_frame_equal(result, expected)


# 参数化测试函数：测试非字符串型缺失值的处理
@pytest.mark.parametrize(
    "na_values",  # 参数化：缺失值列表
    [
        ["-999.0", "-999"],
        [-999, -999.0],
        [-999.0, -999],
        ["-999.0"],
        ["-999"],
        [-999.0],
        [-999],
    ],
)
@pytest.mark.parametrize(
    "data",  # 参数化：CSV数据列表
    [
        """A,B
-999,1.2
2,-999
3,4.5
""",
        """A,B
-999,1.200
2,-999.000
3,4.500
""",
    ],
)
def test_non_string_na_values(all_parsers, data, na_values, request):
    # see gh-3611: with an odd float format, we can't match
    # the string "999.0" exactly but still need float matching
    parser = all_parsers  # 获取传入的解析器对象
    expected = DataFrame([[np.nan, 1.2], [2.0, np.nan], [3.0, 4.5]], columns=["A", "B"])

    # 如果是pyarrow引擎，并且na_values中不全是字符串，则抛出TypeError异常
    if parser.engine == "pyarrow" and not all(isinstance(x, str) for x in na_values):
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    # 如果是pyarrow引擎，并且数据中包含"-999.000"字符串，则标记为xfail，说明pyarrow引擎不能正确识别等效浮点数
    elif parser.engine == "pyarrow" and "-999.000" in data:
        mark = pytest.mark.xfail(
            reason="pyarrow engined does not recognize equivalent floats"
        )
        request.applymarker(mark)

    # 使用给定的解析器解析CSV数据，返回DataFrame对象
    result = parser.read_csv(StringIO(data), na_values=na_values)
    # 使用测试框架断言实际输出与期望输出相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试默认缺失值处理
def test_default_na_values(all_parsers):
    # 定义一个集合，包含各种表示缺失值的字符串
    _NA_VALUES = {
        "-1.#IND",
        "1.#QNAN",
        "1.#IND",
        "-1.#QNAN",
        "#N/A",
        "N/A",
        "n/a",
        "NA",
        "<NA>",
        "#NA",
        "NULL",
        "null",
        "NaN",
        "nan",
        "-NaN",
        "-nan",
        "#N/A N/A",
        "",
        "None",
    }
    
    # 使用断言确保定义的集合和另一个变量 STR_NA_VALUES 相等
    assert _NA_VALUES == STR_NA_VALUES
    
    # 将所有解析器赋值给变量 parser
    parser = all_parsers
    
    # 计算 _NA_VALUES 集合的长度
    nv = len(_NA_VALUES)
    
    # 定义函数 f，用于生成缺失值的字符串
    def f(i, v):
        # 如果 i 等于 0，则初始化 buf 为空字符串
        if i == 0:
            buf = ""
        # 如果 i 大于 0，则生成包含 i 个逗号的字符串
        elif i > 0:
            buf = "".join([","] * i)
    
        # 将参数 v 追加到 buf
        buf = f"{buf}{v}"
    
        # 如果 i 小于 nv - 1，则生成包含 nv - i - 1 个逗号的字符串并追加到 buf
        if i < nv - 1:
            joined = "".join([","] * (nv - i - 1))
            buf = f"{buf}{joined}"
    
        return buf
    
    # 使用列表推导式和函数 f，生成包含所有 _NA_VALUES 的字符串的数据流对象
    data = StringIO("\n".join([f(i, v) for i, v in enumerate(_NA_VALUES)]))
    
    # 使用 numpy 创建一个与 _NA_VALUES 长度相同的列，每个元素的值为 NaN，组成一个预期的 DataFrame
    expected = DataFrame(np.nan, columns=range(nv), index=range(nv))
    
    # 使用 parser 的 read_csv 方法读取 data 中的内容，不使用 header 行
    result = parser.read_csv(data, header=None)
    
    # 使用测试工具（tm）确保 result 和 expected 的 DataFrame 结构相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("na_values", ["baz", ["baz"]])
# 使用 pytest 的参数化装饰器，定义参数 na_values 为 "baz" 和 ["baz"] 两种情况
def test_custom_na_values(all_parsers, na_values):
    # 从 all_parsers 获取数据解析器
    parser = all_parsers
    # 定义测试数据
    data = """A,B,C
ignore,this,row
1,NA,3
-1.#IND,5,baz
7,8,NaN
"""
    # 期望的 DataFrame 结果
    expected = DataFrame(
        [[1.0, np.nan, 3], [np.nan, 5, np.nan], [7, 8, np.nan]], columns=["A", "B", "C"]
    )
    # 如果数据解析器的引擎是 "pyarrow"
    if parser.engine == "pyarrow":
        # 抛出 ValueError 异常，匹配给定的错误信息
        msg = "skiprows argument must be an integer when using engine='pyarrow'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
        return

    # 使用数据解析器读取 CSV 数据，应用给定的 na_values 和 skiprows 参数
    result = parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
    # 断言结果与期望的 DataFrame 结果相等
    tm.assert_frame_equal(result, expected)


def test_bool_na_values(all_parsers):
    # 定义测试数据
    data = """A,B,C
True,False,True
NA,True,False
False,NA,True"""
    # 从 all_parsers 获取数据解析器
    parser = all_parsers
    # 使用数据解析器读取 CSV 数据
    result = parser.read_csv(StringIO(data))
    # 期望的 DataFrame 结果
    expected = DataFrame(
        {
            "A": np.array([True, np.nan, False], dtype=object),
            "B": np.array([False, True, np.nan], dtype=object),
            "C": [True, False, True],
        }
    )
    # 如果数据解析器的引擎是 "pyarrow"
    if parser.engine == "pyarrow":
        # 调整期望结果中特定位置的值为 None
        expected.loc[1, "A"] = None
        expected.loc[2, "B"] = None
    # 断言结果与期望的 DataFrame 结果相等
    tm.assert_frame_equal(result, expected)


def test_na_value_dict(all_parsers):
    # 定义测试数据
    data = """A,B,C
foo,bar,NA
bar,foo,foo
foo,bar,NA
bar,foo,foo"""
    # 从 all_parsers 获取数据解析器
    parser = all_parsers

    # 如果数据解析器的引擎是 "pyarrow"
    if parser.engine == "pyarrow":
        # 抛出 ValueError 异常，匹配给定的错误信息
        msg = "pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
        return

    # 使用数据解析器读取 CSV 数据，应用给定的 na_values 参数
    df = parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
    # 期望的 DataFrame 结果
    expected = DataFrame(
        {
            "A": [np.nan, "bar", np.nan, "bar"],
            "B": [np.nan, "foo", np.nan, "foo"],
            "C": [np.nan, "foo", np.nan, "foo"],
        }
    )
    # 断言结果与期望的 DataFrame 结果相等
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "index_col,expected",
    [
        (
            [0],
            DataFrame({"b": [np.nan], "c": [1], "d": [5]}, index=Index([0], name="a")),
        ),
        (
            [0, 2],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
        (
            ["a", "c"],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
    ],
)
# 使用 pytest 的参数化装饰器，定义 index_col 和 expected 参数的多个测试情况
def test_na_value_dict_multi_index(all_parsers, index_col, expected):
    # 定义测试数据
    data = """\
a,b,c,d
0,NA,1,5
"""
    # 从 all_parsers 获取数据解析器
    parser = all_parsers
    # 使用数据解析器读取 CSV 数据，应用给定的 na_values 和 index_col 参数
    result = parser.read_csv(StringIO(data), na_values=set(), index_col=index_col)
    # 断言结果与期望的 DataFrame 结果相等
    tm.assert_frame_equal(result, expected)
    # 创建一个包含四个元组的列表，每个元组包含两个字典：
    [
        # 第一个元组
        (
            # 第一个字典，空字典
            {},
            # 第二个字典，包含键值对 "A"、"B"、"C"
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],  # 字符串列表，包含 NaN 值
                "B": [1, 2, 3, 4, 5, 6, 7],  # 整数列表
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],  # 字符串列表，包含 NaN 值
            },
        ),
        # 第二个元组
        (
            # 第一个字典，包含键 "na_values" 和 "keep_default_na"
            {"na_values": {"A": [], "C": []}, "keep_default_na": False},
            # 第二个字典，包含键值对 "A"、"B"、"C"
            {
                "A": ["a", "b", "", "d", "e", "nan", "g"],  # 字符串列表，包含空字符串和 "nan"
                "B": [1, 2, 3, 4, 5, 6, 7],  # 整数列表
                "C": ["one", "two", "three", "nan", "five", "", "seven"],  # 字符串列表，包含空字符串和 "nan"
            },
        ),
        # 第三个元组
        (
            # 第一个字典，包含键 "na_values" 和 "keep_default_na"
            {"na_values": ["a"], "keep_default_na": False},
            # 第二个字典，包含键值对 "A"、"B"、"C"
            {
                "A": [np.nan, "b", "", "d", "e", "nan", "g"],  # 字符串列表，包含 NaN 值和空字符串
                "B": [1, 2, 3, 4, 5, 6, 7],  # 整数列表
                "C": ["one", "two", "three", "nan", "five", "", "seven"],  # 字符串列表，包含 "nan" 和空字符串
            },
        ),
        # 第四个元组
        (
            # 第一个字典，包含键 "na_values" 和 "keep_default_na"
            {"na_values": {"A": [], "C": []}},
            # 第二个字典，包含键值对 "A"、"B"、"C"
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],  # 字符串列表，包含 NaN 值
                "B": [1, 2, 3, 4, 5, 6, 7],  # 整数列表
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],  # 字符串列表，包含 NaN 值
            },
        ),
    ],
def test_na_values_keep_default(all_parsers, kwargs, expected, request):
    # 定义包含空值的测试数据
    data = """\
A,B,C
a,1,one
b,2,two
,3,three
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    # 选择当前测试使用的解析器
    parser = all_parsers
    # 如果解析器使用 pyarrow 引擎
    if parser.engine == "pyarrow":
        # 如果传递了字典类型的 na_values 参数，引发 ValueError 异常
        if "na_values" in kwargs and isinstance(kwargs["na_values"], dict):
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), **kwargs)
            return
        # 标记该测试为预期失败
        mark = pytest.mark.xfail()
        request.applymarker(mark)

    # 使用给定参数解析 CSV 数据
    result = parser.read_csv(StringIO(data), **kwargs)
    # 创建预期结果的 DataFrame 对象
    expected = DataFrame(expected)
    # 断言解析结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_no_na_values_no_keep_default(all_parsers):
    # 查看 GitHub 问题 #4318：传递 na_values=None 和 keep_default_na=False 时，'None' 作为 na_value
    data = """\
A,B,C
a,1,None
b,2,two
,3,None
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    # 选择当前测试使用的解析器
    parser = all_parsers
    # 使用给定参数解析 CSV 数据，禁用默认的 na_values
    result = parser.read_csv(StringIO(data), keep_default_na=False)

    # 创建预期结果的 DataFrame 对象
    expected = DataFrame(
        {
            "A": ["a", "b", "", "d", "e", "nan", "g"],
            "B": [1, 2, 3, 4, 5, 6, 7],
            "C": ["None", "two", "None", "nan", "five", "", "seven"],
        }
    )
    # 断言解析结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_values(all_parsers):
    # 查看 GitHub 问题 #19227
    data = "a,b\n,2"
    # 选择当前测试使用的解析器
    parser = all_parsers

    # 如果解析器使用 pyarrow 引擎
    if parser.engine == "pyarrow":
        # 如果传递了字典类型的 na_values 参数，引发 ValueError 异常
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
            )
        return

    # 使用给定参数解析 CSV 数据，禁用默认的 na_values
    result = parser.read_csv(
        StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
    )
    # 创建预期结果的 DataFrame 对象
    expected = DataFrame({"a": [""], "b": [np.nan]})
    # 断言解析结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_scalar_values(all_parsers):
    # 查看 GitHub 问题 #19227
    #
    # 标量值不应导致解析崩溃或失败。
    data = "a,b\n1,2"
    # 选择当前测试使用的解析器
    parser = all_parsers

    # 如果解析器使用 pyarrow 引擎
    if parser.engine == "pyarrow":
        # 如果传递了字典类型的 na_values 参数，引发 ValueError 异常
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
        return

    # 使用给定参数解析 CSV 数据，禁用默认的 na_values
    df = parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
    # 创建预期结果的 DataFrame 对象
    expected = DataFrame({"a": [1], "b": [np.nan]})
    # 断言解析结果与预期结果相等
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("col_zero_na_values", [113125, "113125"])
def test_no_keep_default_na_dict_na_values_diff_reprs(all_parsers, col_zero_na_values):
    # 查看 GitHub 问题 #19227
    data = """\
113125,"blah","/blaha",kjsdkj,412.166,225.874,214.008
729639,"qwer","",asdfkj,466.681,,252.373
"""
    # 选择当前测试使用的解析器
    parser = all_parsers
    # 创建一个期望的DataFrame对象，包含特定的数据和NaN值
    expected = DataFrame(
        {
            0: [np.nan, 729639.0],
            1: [np.nan, "qwer"],
            2: ["/blaha", np.nan],
            3: ["kjsdkj", "asdfkj"],
            4: [412.166, 466.681],
            5: ["225.874", ""],
            6: [np.nan, 252.373],
        }
    )

    # 检查解析器使用的引擎是否为"pyarrow"
    if parser.engine == "pyarrow":
        # 如果是"pyarrow"引擎，抛出值错误并匹配特定的错误消息
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            # 尝试使用read_csv方法，传入StringIO对象和特定参数，预期抛出错误
            parser.read_csv(
                StringIO(data),
                header=None,
                keep_default_na=False,
                na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
            )
        return

    # 使用解析器的read_csv方法读取数据，预期得到的DataFrame应该和期望的DataFrame相等
    result = parser.read_csv(
        StringIO(data),
        header=None,
        keep_default_na=False,
        na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
    )
    # 使用tm.assert_frame_equal方法比较result和expected，确认它们相等
    tm.assert_frame_equal(result, expected)
@xfail_pyarrow  # 标记为在 pyarrow 下预期会失败的测试用例，原因是数据类型不匹配和未来警告
@pytest.mark.parametrize(
    "na_filter,row_data",
    [
        (True, [[1, "A"], [np.nan, np.nan], [3, "C"]]),  # 使用真值过滤 NaN，期望返回的数据
        (False, [["1", "A"], ["nan", "B"], ["3", "C"]]),  # 禁用 NaN 过滤，期望返回的数据
    ],
)
def test_na_values_na_filter_override(all_parsers, na_filter, row_data):
    data = """\
A,B
1,A
nan,B
3,C
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), na_values=["B"], na_filter=na_filter)  # 读取 CSV 数据并处理 NaN 值

    expected = DataFrame(row_data, columns=["A", "B"])  # 期望得到的 DataFrame 结果
    tm.assert_frame_equal(result, expected)  # 断言实际结果与期望结果相等


@skip_pyarrow  # 跳过在 pyarrow 引擎下的测试，因为 CSV 解析错误：期望 8 列，得到了 5 列
def test_na_trailing_columns(all_parsers):
    parser = all_parsers
    data = """Date,Currency,Symbol,Type,Units,UnitPrice,Cost,Tax
2012-03-14,USD,AAPL,BUY,1000
2012-05-12,USD,SBUX,SELL,500"""

    # 尾部多余的列应该全部是 NaN
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [
            ["2012-03-14", "USD", "AAPL", "BUY", 1000, np.nan, np.nan, np.nan],
            ["2012-05-12", "USD", "SBUX", "SELL", 500, np.nan, np.nan, np.nan],
        ],
        columns=[
            "Date",
            "Currency",
            "Symbol",
            "Type",
            "Units",
            "UnitPrice",
            "Cost",
            "Tax",
        ],
    )
    tm.assert_frame_equal(result, expected)  # 断言实际结果与期望结果相等


@pytest.mark.parametrize(
    "na_values,row_data",
    [
        (1, [[np.nan, 2.0], [2.0, np.nan]]),  # 使用标量 1 作为 NaN 值，期望返回的数据
        ({"a": 2, "b": 1}, [[1.0, 2.0], [np.nan, np.nan]]),  # 使用字典作为 NaN 值，期望返回的数据
    ],
)
def test_na_values_scalar(all_parsers, na_values, row_data):
    # 见 gh-12224
    parser = all_parsers
    names = ["a", "b"]
    data = "1,2\n2,1"

    if parser.engine == "pyarrow" and isinstance(na_values, dict):
        if isinstance(na_values, dict):
            err = ValueError
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
        else:
            err = TypeError
            msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    elif parser.engine == "pyarrow":
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return

    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)  # 读取 CSV 数据并处理 NaN 值
    expected = DataFrame(row_data, columns=names)  # 期望得到的 DataFrame 结果
    tm.assert_frame_equal(result, expected)  # 断言实际结果与期望结果相等


def test_na_values_dict_aliasing(all_parsers):
    parser = all_parsers
    na_values = {"a": 2, "b": 1}
    na_values_copy = na_values.copy()

    names = ["a", "b"]
    data = "1,2\n2,1"

    expected = DataFrame([[1.0, 2.0], [np.nan, np.nan]], columns=names)  # 期望得到的 DataFrame 结果
    # 如果使用的解析引擎是 "pyarrow"
    if parser.engine == "pyarrow":
        # 准备错误信息字符串，用于匹配 pytest 抛出的 ValueError 异常
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        # 使用 pytest 的断言来验证是否抛出预期的 ValueError 异常，并匹配错误信息字符串
        with pytest.raises(ValueError, match=msg):
            # 调用 parser 的 read_csv 方法解析 CSV 数据，并传入数据流、列名和 na_values 参数
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        # 如果抛出异常则直接返回，不再执行后续代码
        return

    # 使用 parser 的 read_csv 方法解析 CSV 数据，并传入数据流、列名和 na_values 参数，将结果保存在 result 中
    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)

    # 使用 assert_frame_equal 断言比较 result 和期望的数据框 expected 是否相等
    tm.assert_frame_equal(result, expected)
    # 使用 assert_dict_equal 断言比较 na_values 和其副本 na_values_copy 是否相等
    tm.assert_dict_equal(na_values, na_values_copy)
# 测试处理空值字典和空列名
def test_na_values_dict_null_column_name(all_parsers):
    # 见问题 gh-57547
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 定义包含空值的 CSV 数据字符串
    data = ",x,y\n\nMA,1,2\nNA,2,1\nOA,,3"
    # 定义列名列表，包含一个空列名
    names = [None, "x", "y"]
    # 创建一个空值字典，用于指定每个列的空值
    na_values = {name: STR_NA_VALUES for name in names}
    # 定义数据类型字典，将空列和其他列的数据类型指定为对象和浮点数
    dtype = {None: "object", "x": "float64", "y": "float64"}

    # 如果解析器引擎是 pyarrow，则抛出错误，因为不支持传递空值字典
    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                index_col=0,
                header=0,
                dtype=dtype,
                names=names,
                na_values=na_values,
                keep_default_na=False,
            )
        return

    # 创建预期的数据帧对象，包含预期的数据和空值处理
    expected = DataFrame(
        {None: ["MA", "NA", "OA"], "x": [1.0, 2.0, np.nan], "y": [2.0, 1.0, 3.0]}
    )

    # 将预期的数据帧对象设置为空列索引
    expected = expected.set_index(None)

    # 使用解析器对象读取 CSV 数据，进行空值处理，并获取结果
    result = parser.read_csv(
        StringIO(data),
        index_col=0,
        header=0,
        dtype=dtype,
        names=names,
        na_values=na_values,
        keep_default_na=False,
    )

    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


# 测试处理空值字典和列索引
def test_na_values_dict_col_index(all_parsers):
    # 见问题 gh-14203
    # 定义包含数据的 CSV 字符串
    data = "a\nfoo\n1"
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 创建空值字典，指定第一列的空值
    na_values = {0: "foo"}

    # 如果解析器引擎是 pyarrow，则抛出错误，因为不支持传递空值字典
    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return

    # 使用解析器对象读取 CSV 数据，进行空值处理，并获取结果
    result = parser.read_csv(StringIO(data), na_values=na_values)
    # 创建预期的数据帧对象，包含预期的数据和空值处理
    expected = DataFrame({"a": [np.nan, 1]})
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


# 参数化测试，测试处理 uint64 类型的空值
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            str(2**63) + "\n" + str(2**63 + 1),
            {"na_values": [2**63]},
            [str(2**63), str(2**63 + 1)],
        ),
        (str(2**63) + ",1" + "\n,2", {}, [[str(2**63), 1], ["", 2]]),
        (str(2**63) + "\n1", {"na_values": [2**63]}, [np.nan, 1]),
    ],
)
# 测试处理 uint64 类型的空值列表
def test_na_values_uint64(all_parsers, data, kwargs, expected, request):
    # 见问题 gh-14983
    # 从参数中获取所有解析器对象
    parser = all_parsers

    # 如果解析器引擎是 pyarrow 并且参数中包含空值字典，则引发类型错误
    if parser.engine == "pyarrow" and "na_values" in kwargs:
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), header=None, **kwargs)
        return
    elif parser.engine == "pyarrow":
        # 标记该测试预计会失败，因为 pyarrow 会返回 float64 而不是 object 类型
        mark = pytest.mark.xfail(reason="Returns float64 instead of object")
        request.applymarker(mark)

    # 使用解析器对象读取 CSV 数据，进行空值处理，并获取结果
    result = parser.read_csv(StringIO(data), header=None, **kwargs)
    # 创建预期的数据帧对象，包含预期的数据和空值处理
    expected = DataFrame(expected)
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


# 测试处理空值字典和默认索引列
def test_empty_na_values_no_default_with_index(all_parsers):
    # 见问题 gh-15835
    # 定义包含数据的 CSV 字符串
    data = "a,1\nb,2"
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 创建预期的数据帧对象，包含预期的数据和索引
    expected = DataFrame({"1": [2]}, index=Index(["b"], name="a"))

    # 使用解析器对象读取 CSV 数据，进行空值处理，并获取结果
    result = parser.read_csv(StringIO(data), index_col=0, keep_default_na=False)
    # 使用测试框架中的函数 `assert_frame_equal` 来比较 `result` 和 `expected` 两个数据框是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "na_filter,index_data", [(False, ["", "5"]), (True, [np.nan, 5.0])]
)
def test_no_na_filter_on_index(all_parsers, na_filter, index_data, request):
    # 标记此测试函数用途为解决 GitHub 问题 #5239
    #
    # 当 na_filter=False 时，不在索引中解析 NA 值
    parser = all_parsers
    data = "a,b,c\n1,,3\n4,5,6"

    if parser.engine == "pyarrow" and na_filter is False:
        # 标记测试为预期失败，原因是索引结果不匹配
        mark = pytest.mark.xfail(reason="mismatched index result")
        request.applymarker(mark)

    # 期望的 DataFrame 结果，指定了索引数据
    expected = DataFrame({"a": [1, 4], "c": [3, 6]}, index=Index(index_data, name="b"))
    result = parser.read_csv(StringIO(data), index_col=[1], na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


def test_inf_na_values_with_int_index(all_parsers):
    # 标记此测试函数用途为解决 GitHub 问题 #17128
    parser = all_parsers
    data = "idx,col1,col2\n1,3,4\n2,inf,-inf"

    # 处理整数索引列中的 inf 值，不应导致 OverflowError
    out = parser.read_csv(StringIO(data), index_col=[0], na_values=["inf", "-inf"])
    expected = DataFrame(
        {"col1": [3, np.nan], "col2": [4, np.nan]}, index=Index([1, 2], name="idx")
    )
    tm.assert_frame_equal(out, expected)


@xfail_pyarrow  # mismatched shape
@pytest.mark.parametrize("na_filter", [True, False])
def test_na_values_with_dtype_str_and_na_filter(all_parsers, na_filter):
    # 标记此测试函数用途为解决 GitHub 问题 #20377
    parser = all_parsers
    data = "a,b,c\n1,,3\n4,5,6"

    # 根据 na_filter 的值，处理缺失值，将缺失值转换为 NaN 或保留为空字符串
    empty = np.nan if na_filter else ""
    expected = DataFrame({"a": ["1", "4"], "b": [empty, "5"], "c": ["3", "6"]})

    result = parser.read_csv(StringIO(data), na_filter=na_filter, dtype=str)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # mismatched exception message
@pytest.mark.parametrize(
    "data, na_values",
    [
        ("false,1\n,1\ntrue", None),
        ("false,1\nnull,1\ntrue", None),
        ("false,1\nnan,1\ntrue", None),
        ("false,1\nfoo,1\ntrue", "foo"),
        ("false,1\nfoo,1\ntrue", ["foo"]),
        ("false,1\nfoo,1\ntrue", {"a": "foo"}),
    ],
)
def test_cast_NA_to_bool_raises_error(all_parsers, data, na_values):
    parser = all_parsers
    msg = "|".join(
        [
            "Bool column has NA values in column [0a]",
            "cannot safely convert passed user dtype of "
            "bool for object dtyped data in column 0",
        ]
    )

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(
            StringIO(data),
            header=None,
            names=["a", "b"],
            dtype={"a": "bool"},
            na_values=na_values,
        )


# TODO: this test isn't about the na_values keyword, it is about the empty entries
#  being returned with NaN entries, whereas the pyarrow engine returns "nan"
@xfail_pyarrow  # mismatched shapes
def test_str_nan_dropped(all_parsers):
    # 标记此测试函数用途为解决 GitHub 问题 #21131
    parser = all_parsers

    data = """File: small.csv,,
10010010233,0123,654
foo,,bar
# 测试读取 CSV 数据并验证处理结果的一致性

# 定义测试函数，使用 pytest 框架
def test_read_csv(all_parsers):
    # 获取所需的解析器对象
    parser = all_parsers
    # 准备测试数据，包含三列数据的 CSV 格式字符串
    data = """01001000155,4530,898"""

    # 调用解析器的读取 CSV 方法，并进行数据处理：读取为 DataFrame 后去除空值行
    result = parser.read_csv(
        StringIO(data),
        header=None,
        names=["col1", "col2", "col3"],
        dtype={"col1": str, "col2": str, "col3": str},
    ).dropna()

    # 预期的 DataFrame 结果，包含两行数据
    expected = DataFrame(
        {
            "col1": ["10010010233", "01001000155"],
            "col2": ["0123", "4530"],
            "col3": ["654", "898"],
        },
        index=[1, 3],
    )

    # 使用 pandas 提供的 assert_frame_equal 方法验证结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 测试读取 CSV 数据，并验证特定条件下的处理行为

# 定义测试函数，使用 pytest 框架
def test_nan_multi_index(all_parsers):
    # 获取所需的解析器对象
    parser = all_parsers
    # 准备测试数据，包含多层索引的 CSV 格式字符串
    data = "A,B,B\nX,Y,Z\n1,2,inf"

    # 对于 pyarrow 引擎，验证特定的异常情况是否抛出
    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), header=list(range(2)), na_values={("B", "Z"): "inf"}
            )
        return

    # 对于其他引擎，验证读取 CSV 后的结果是否符合预期，包含缺失值处理
    result = parser.read_csv(
        StringIO(data), header=list(range(2)), na_values={("B", "Z"): "inf"}
    )

    # 预期的 DataFrame 结果，包含一列缺失值
    expected = DataFrame(
        {
            ("A", "X"): [1],
            ("B", "Y"): [2],
            ("B", "Z"): [np.nan],
        }
    )

    # 使用 pandas 提供的 assert_frame_equal 方法验证结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 测试处理布尔值和 NaN 转换为布尔类型的情况

# 定义测试函数，使用 pytest 框架，此测试预期会失败
@xfail_pyarrow  # 标记该测试预期失败，因为异常类型不符合预期
def test_bool_and_nan_to_bool(all_parsers):
    # 获取所需的解析器对象
    parser = all_parsers
    # 准备测试数据，包含布尔值和 NaN 的 CSV 格式字符串
    data = """0
NaN
True
False
"""
    # 验证是否抛出特定异常，此处预期 ValueError 异常
    with pytest.raises(ValueError, match="NA values"):
        parser.read_csv(StringIO(data), dtype="bool")


# 测试处理布尔值和 NaN 转换为整数类型的情况

# 定义测试函数，使用 pytest 框架
def test_bool_and_nan_to_int(all_parsers):
    # 获取所需的解析器对象
    parser = all_parsers
    # 准备测试数据，包含布尔值和 NaN 的 CSV 格式字符串
    data = """0
NaN
True
False
"""
    # 验证是否抛出特定异常，此处预期 ValueError 异常
    with pytest.raises(ValueError, match="convert|NoneType"):
        parser.read_csv(StringIO(data), dtype="int")


# 测试处理布尔值和 NaN 转换为浮点数类型的情况

# 定义测试函数，使用 pytest 框架
def test_bool_and_nan_to_float(all_parsers):
    # 获取所需的解析器对象
    parser = all_parsers
    # 准备测试数据，包含布尔值和 NaN 的 CSV 格式字符串
    data = """0
NaN
True
False
"""
    # 验证读取 CSV 后的结果是否符合预期，包含 NaN 转换
    result = parser.read_csv(StringIO(data), dtype="float")
    # 预期的 DataFrame 结果，包含一列 NaN 值
    expected = DataFrame.from_dict({"0": [np.nan, 1.0, 0.0]})
    # 使用 pandas 提供的 assert_frame_equal 方法验证结果与预期是否一致
    tm.assert_frame_equal(result, expected)
```