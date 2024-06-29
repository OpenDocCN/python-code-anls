# `D:\src\scipysrc\pandas\pandas\tests\io\parser\dtypes\test_dtypes_basic.py`

```
"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""

# 导入所需的模块和库
from collections import defaultdict
from io import StringIO

import numpy as np  # 导入 NumPy 库并命名为 np
import pytest  # 导入 pytest 测试框架

from pandas.errors import ParserWarning  # 导入 ParserWarning 异常类

import pandas as pd  # 导入 Pandas 库并命名为 pd
from pandas import (  # 从 Pandas 中导入 DataFrame 和 Timestamp 类
    DataFrame,
    Timestamp,
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具

from pandas.core.arrays import (  # 导入 Pandas 的数组类型
    ArrowStringArray,
    IntegerArray,
    StringArray,
)

# 忽略特定警告信息的 pytest 标记
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.mark.parametrize("dtype", [str, object])  # 参数化测试，测试 str 和 object 类型
@pytest.mark.parametrize("check_orig", [True, False])  # 参数化测试，测试 True 和 False
@pytest.mark.usefixtures("pyarrow_xfail")  # 使用 pytest 的 usefixtures 标记，处理 pyarrow 失败情况
def test_dtype_all_columns(all_parsers, dtype, check_orig):
    # 测试用例：gh-3795, gh-6607
    parser = all_parsers  # 获取所有解析器的实例

    df = DataFrame(  # 创建一个 DataFrame 对象
        np.random.default_rng(2).random((5, 2)).round(4),  # 使用随机数填充数据
        columns=list("AB"),  # 指定列名为 A 和 B
        index=["1A", "1B", "1C", "1D", "1E"],  # 指定索引
    )

    with tm.ensure_clean("__passing_str_as_dtype__.csv") as path:  # 使用 tm.ensure_clean 确保路径的清理
        df.to_csv(path)  # 将 DataFrame 写入 CSV 文件

        result = parser.read_csv(path, dtype=dtype, index_col=0)  # 读取 CSV 文件，指定数据类型和索引列

        if check_orig:  # 如果 check_orig 为 True
            expected = df.copy()  # 复制原始 DataFrame 到 expected
            result = result.astype(float)  # 将结果 DataFrame 转换为 float 类型
        else:  # 如果 check_orig 为 False
            expected = df.astype(str)  # 将原始 DataFrame 转换为 str 类型

        tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具比较 result 和 expected 的一致性


@pytest.mark.usefixtures("pyarrow_xfail")  # 使用 pytest 的 usefixtures 标记，处理 pyarrow 失败情况
def test_dtype_per_column(all_parsers):
    parser = all_parsers  # 获取所有解析器的实例
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""
    expected = DataFrame(  # 创建预期的 DataFrame
        [[1, "2.5"], [2, "3.5"], [3, "4.5"], [4, "5.5"]], columns=["one", "two"]
    )
    expected["one"] = expected["one"].astype(np.float64)  # 将列 'one' 转换为 np.float64 类型
    expected["two"] = expected["two"].astype(object)  # 将列 'two' 转换为 object 类型

    result = parser.read_csv(StringIO(data), dtype={"one": np.float64, 1: str})  # 读取 CSV 数据，并指定列的数据类型
    tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具比较 result 和 expected 的一致性


def test_invalid_dtype_per_column(all_parsers):
    parser = all_parsers  # 获取所有解析器的实例
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""

    with pytest.raises(TypeError, match="data type [\"']foo[\"'] not understood"):  # 检查是否抛出预期的 TypeError 异常
        parser.read_csv(StringIO(data), dtype={"one": "foo", 1: "int"})  # 尝试读取 CSV 数据，并指定错误的数据类型


def test_raise_on_passed_int_dtype_with_nas(all_parsers):
    # 测试用例：gh-2631
    parser = all_parsers  # 获取所有解析器的实例
    data = """YEAR, DOY, a
    2001,106380451,10
    2001,,11
    2001,106380451,67"""

    if parser.engine == "c":  # 根据解析器的引擎类型设置不同的错误消息
        msg = "Integer column has NA values"
    elif parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
    else:
        msg = "Unable to convert column DOY"

    with pytest.raises(ValueError, match=msg):  # 检查是否抛出预期的 ValueError 异常
        parser.read_csv(StringIO(data), dtype={"DOY": np.int64}, skipinitialspace=True)  # 尝试读取 CSV 数据，并指定列 'DOY' 的数据类型


def test_dtype_with_converters(all_parsers):
    parser = all_parsers  # 获取所有解析器的实例
    data = """a,b
    1.1,2.2
    1.2,2.3"""
    # 如果使用的解析引擎是 "pyarrow"
    if parser.engine == "pyarrow":
        # 抛出值错误，指定匹配的消息
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            # 试图使用 'pyarrow' 引擎解析 CSV 数据，指定了 converters 选项，预期抛出错误
            parser.read_csv(
                StringIO(data), dtype={"a": "i8"}, converters={"a": lambda x: str(x)}
            )
        # 如果抛出异常，直接返回，后续代码不执行
        return

    # 如果指定了 converter，dtype 规范将被忽略
    result = parser.read_csv_check_warnings(
        ParserWarning,
        "Both a converter and dtype were specified for column a "
        "- only the converter will be used.",
        StringIO(data),
        dtype={"a": "i8"},
        converters={"a": lambda x: str(x)},
    )
    # 预期的数据框结果，将转换列 'a' 的值为字符串
    expected = DataFrame({"a": ["1.1", "1.2"], "b": [2.2, 2.3]})
    # 检查解析结果与预期是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试解析器对于数值数据类型的处理
def test_numeric_dtype(all_parsers, any_real_numpy_dtype):
    # 准备测试数据，一个包含数字的字符串
    data = "0\n1"
    # 选择指定的解析器
    parser = all_parsers
    # 准备期望的数据框，数据类型由参数指定
    expected = DataFrame([0, 1], dtype=any_real_numpy_dtype)

    # 使用解析器读取 CSV 格式的数据，设置头部为 None，数据类型由参数指定
    result = parser.read_csv(StringIO(data), header=None, dtype=any_real_numpy_dtype)
    # 断言结果与期望相同
    tm.assert_frame_equal(expected, result)


# 使用 pytest 的 fixture 功能标记测试函数，测试布尔值数据类型的处理
@pytest.mark.usefixtures("pyarrow_xfail")
def test_boolean_dtype(all_parsers):
    # 选择指定的解析器
    parser = all_parsers
    # 准备包含布尔值的字符串数据
    data = "\n".join(
        [
            "a",
            "True",
            "TRUE",
            "true",
            "1",
            "1.0",
            "False",
            "FALSE",
            "false",
            "0",
            "0.0",
            "NaN",
            "nan",
            "NA",
            "null",
            "NULL",
        ]
    )

    # 使用解析器读取 CSV 格式的数据，指定数据类型为布尔型
    result = parser.read_csv(StringIO(data), dtype="boolean")
    # 准备期望的数据框，包含布尔值列
    expected = DataFrame(
        {
            "a": pd.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                dtype="boolean",
            )
        }
    )

    # 断言结果与期望相同
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 fixture 功能标记测试函数，测试带有指定分隔符、选择列和解析日期的处理
@pytest.mark.usefixtures("pyarrow_xfail")
def test_delimiter_with_usecols_and_parse_dates(all_parsers):
    # GH#35873
    # 使用解析器读取 CSV 格式的数据，指定引擎为 Python，指定列名、使用的列、解析日期列和小数点分隔符
    result = all_parsers.read_csv(
        StringIO('"dump","-9,1","-9,1",20101010'),
        engine="python",
        names=["col", "col1", "col2", "col3"],
        usecols=["col1", "col2", "col3"],
        parse_dates=["col3"],
        decimal=",",
    )
    # 准备期望的数据框，包含特定列和解析后的日期
    expected = DataFrame(
        {"col1": [-9.1], "col2": [-9.1], "col3": [Timestamp("2010-10-10")]}
    )
    # 断言结果与期望相同
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 功能，测试十进制和指数的处理
@pytest.mark.parametrize("thousands", ["_", None])
def test_decimal_and_exponential(
    request, python_parser_only, numeric_decimal, thousands
):
    # GH#31920
    # 调用具体的十进制数检查函数，验证解析器对于数字和千位分隔符的处理
    decimal_number_check(request, python_parser_only, numeric_decimal, thousands, None)


# 使用 pytest 的 parametrize 功能，测试在不同浮点精度下的处理
@pytest.mark.parametrize("thousands", ["_", None])
@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
def test_1000_sep_decimal_float_precision(
    request, c_parser_only, numeric_decimal, float_precision, thousands
):
    # test decimal and thousand sep handling in across 'float_precision'
    # parsers
    # 调用具体的十进制数检查函数，验证解析器在不同浮点精度下对于数字和千位分隔符的处理
    decimal_number_check(
        request, c_parser_only, numeric_decimal, thousands, float_precision
    )
    text, value = numeric_decimal
    text = " " + text + " "
    if isinstance(value, str):  # the negative cases (parse as text)
        value = " " + value + " "
    # 再次调用具体的十进制数检查函数，验证特定情况下解析器对于数字和千位分隔符的处理
    decimal_number_check(
        request, c_parser_only, (text, value), thousands, float_precision
    )
# GH#31920
def decimal_number_check(request, parser, numeric_decimal, thousands, float_precision):
    # 从 numeric_decimal 中获取第一个值作为待处理的数值字符串
    value = numeric_decimal[0]
    # 如果 thousands 为 None 并且 value 符合指定格式，则标记为预期失败
    if thousands is None and value in ("1_,", "1_234,56", "1_234,56e0"):
        request.applymarker(
            pytest.mark.xfail(reason=f"thousands={thousands} and sep is in {value}")
        )
    # 使用 parser 对象读取以 value 为内容的 CSV 数据，设置各种参数
    df = parser.read_csv(
        StringIO(value),
        float_precision=float_precision,
        sep="|",
        thousands=thousands,
        decimal=",",
        header=None,
    )
    # 获取第一行第一列的值
    val = df.iloc[0, 0]
    # 断言获取的值与 numeric_decimal 中的第二个值相等
    assert val == numeric_decimal[1]


@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
def test_skip_whitespace(c_parser_only, float_precision):
    DATA = """id\tnum\t
1\t1.2 \t
1\t 2.1\t
2\t 1\t
2\t 1.2 \t
"""
    # 使用 c_parser_only 对象读取上述数据，并设置各种参数
    df = c_parser_only.read_csv(
        StringIO(DATA),
        float_precision=float_precision,
        sep="\t",
        header=0,
        dtype={1: np.float64},
    )
    # 断言第二列的值与预期的 Series 相等
    tm.assert_series_equal(df.iloc[:, 1], pd.Series([1.2, 2.1, 1.0, 1.2], name="num"))


@pytest.mark.usefixtures("pyarrow_xfail")
def test_true_values_cast_to_bool(all_parsers):
    # GH#34655
    # 定义包含特定文本数据的字符串
    text = """a,b
yes,xxx
no,yyy
1,zzz
0,aaa
    """
    parser = all_parsers
    # 使用 parser 对象读取文本数据，并将特定的字符串值转换为布尔值
    result = parser.read_csv(
        StringIO(text),
        true_values=["yes"],
        false_values=["no"],
        dtype={"a": "boolean"},
    )
    # 预期的 DataFrame 结果
    expected = DataFrame(
        {"a": [True, False, True, False], "b": ["xxx", "yyy", "zzz", "aaa"]}
    )
    # 将结果中的布尔类型列转换为 boolean 类型
    expected["a"] = expected["a"].astype("boolean")
    # 断言读取的结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("dtypes, exp_value", [({}, "1"), ({"a.1": "int64"}, 1)])
def test_dtype_mangle_dup_cols(all_parsers, dtypes, exp_value):
    # GH#35211
    parser = all_parsers
    # 定义包含重复列名数据的字符串
    data = """a,a\n1,1"""
    # 构建 dtype 字典，指定数据类型
    dtype_dict = {"a": str, **dtypes}
    # GH#42462
    # 创建 dtype 字典的副本
    dtype_dict_copy = dtype_dict.copy()
    # 使用 parser 对象读取上述数据，并按指定的 dtype 解析
    result = parser.read_csv(StringIO(data), dtype=dtype_dict)
    # 预期的 DataFrame 结果
    expected = DataFrame({"a": ["1"], "a.1": [exp_value]})
    # 检查 dtype 字典是否未被修改
    assert dtype_dict == dtype_dict_copy, "dtype dict changed"
    # 断言读取的结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_mangle_dup_cols_single_dtype(all_parsers):
    # GH#42022
    parser = all_parsers
    # 定义包含重复列名数据的字符串
    data = """a,a\n1,1"""
    # 使用 parser 对象读取上述数据，并将所有列解析为字符串类型
    result = parser.read_csv(StringIO(data), dtype=str)
    # 预期的 DataFrame 结果
    expected = DataFrame({"a": ["1"], "a.1": ["1"]})
    # 断言读取的结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_multi_index(all_parsers):
    # GH 42446
    parser = all_parsers
    # 定义包含多级标题的数据字符串
    data = "A,B,B\nX,Y,Z\n1,2,3"

    # 使用 parser 对象读取上述数据，并按指定的 dtype 解析
    result = parser.read_csv(
        StringIO(data),
        header=list(range(2)),
        dtype={
            ("A", "X"): np.int32,
            ("B", "Y"): np.int32,
            ("B", "Z"): np.float32,
        },
    )
    # 创建一个预期的DataFrame对象，包含如下列和对应的数据类型：
    # 列 ("A", "X") 包含一个 numpy.int32 类型的数组，数据为 [1]
    # 列 ("B", "Y") 包含一个 numpy.int32 类型的数组，数据为 [2]
    # 列 ("B", "Z") 包含一个 numpy.float32 类型的数组，数据为 [3]
    expected = DataFrame(
        {
            ("A", "X"): np.int32([1]),
            ("B", "Y"): np.int32([2]),
            ("B", "Z"): np.float32([3]),
        }
    )
    
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数，
    # 检查变量 result 和预期的 DataFrame expected 是否相等。
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，测试可空整数数据类型的处理
def test_nullable_int_dtype(all_parsers, any_int_ea_dtype):
    # GH 25472
    # 使用传入的解析器和数据类型
    parser = all_parsers
    dtype = any_int_ea_dtype

    # 定义测试数据字符串
    data = """a,b,c
,3,5
1,,6
2,4,"""
    
    # 期望结果，创建一个数据帧，包含三列，每列使用指定的数据类型处理缺失值
    expected = DataFrame(
        {
            "a": pd.array([pd.NA, 1, 2], dtype=dtype),
            "b": pd.array([3, pd.NA, 4], dtype=dtype),
            "c": pd.array([5, 6, pd.NA], dtype=dtype),
        }
    )
    
    # 使用解析器读取CSV数据，期望结果与预期数据帧相等
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


# 使用pytest标记，参数化测试函数，测试默认数据类型字典的处理
@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("default", ["float", "float64"])
def test_dtypes_defaultdict(all_parsers, default):
    # GH#41574
    # 定义测试数据字符串
    data = """a,b
1,2
"""
    
    # 定义数据类型字典，默认使用指定的默认数据类型处理未指定的列
    dtype = defaultdict(lambda: default, a="int64")
    parser = all_parsers
    
    # 使用解析器读取CSV数据，期望结果与预期数据帧相等
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({"a": [1], "b": 2.0})
    tm.assert_frame_equal(result, expected)


# 使用pytest标记，测试处理重复列名的数据类型字典的功能
@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtypes_defaultdict_mangle_dup_cols(all_parsers):
    # GH#41574
    # 定义测试数据字符串
    data = """a,b,a,b,b.1
1,2,3,4,5
"""
    
    # 定义数据类型字典，指定特定列使用特定数据类型处理，避免列名冲突
    dtype = defaultdict(lambda: "float64", a="int64")
    dtype["b.1"] = "int64"
    parser = all_parsers
    
    # 使用解析器读取CSV数据，期望结果与预期数据帧相等
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({"a": [1], "b": [2.0], "a.1": [3], "b.2": [4.0], "b.1": [5]})
    tm.assert_frame_equal(result, expected)


# 使用pytest标记，测试无法识别数据类型的数据类型字典的处理
@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtypes_defaultdict_invalid(all_parsers):
    # GH#41574
    # 定义测试数据字符串
    data = """a,b
1,2
"""
    
    # 定义数据类型字典，将无法识别的数据类型应用于指定列
    dtype = defaultdict(lambda: "invalid_dtype", a="int64")
    parser = all_parsers
    
    # 使用解析器读取CSV数据，期望抛出类型错误异常
    with pytest.raises(TypeError, match="not understood"):
        parser.read_csv(StringIO(data), dtype=dtype)


# 定义测试函数，测试数据类型后端处理与日期解析
def test_dtype_backend(all_parsers):
    # GH#36712

    # 使用传入的解析器
    parser = all_parsers

    # 定义测试数据字符串
    data = """a,b,c,d,e,f,g,h,i,j
1,2.5,True,a,,,,,12-31-2019,
3,4.5,False,b,6,7.5,True,a,12-31-2019,
"""
    
    # 使用解析器读取CSV数据，设置数据类型后端为numpy_nullable，同时解析日期列
    result = parser.read_csv(
        StringIO(data), dtype_backend="numpy_nullable", parse_dates=["i"]
    )
    
    # 期望结果，创建一个数据帧，每列根据数据内容自动推断数据类型
    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="Int64"),
            "b": pd.Series([2.5, 4.5], dtype="Float64"),
            "c": pd.Series([True, False], dtype="boolean"),
            "d": pd.Series(["a", "b"], dtype="string"),
            "e": pd.Series([pd.NA, 6], dtype="Int64"),
            "f": pd.Series([pd.NA, 7.5], dtype="Float64"),
            "g": pd.Series([pd.NA, True], dtype="boolean"),
            "h": pd.Series([pd.NA, "a"], dtype="string"),
            "i": pd.Series([Timestamp("2019-12-31")] * 2),
            "j": pd.Series([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试数据类型后端处理与指定数据类型
def test_dtype_backend_and_dtype(all_parsers):
    # GH#36712

    # 使用传入的解析器
    parser = all_parsers

    # 定义测试数据字符串
    data = """a,b
1,2.5
,
"""
    
    # 使用解析器读取CSV数据，设置数据类型后端为numpy_nullable，同时指定数据类型
    result = parser.read_csv(
        StringIO(data), dtype_backend="numpy_nullable", dtype="float64"
    )
    
    # 期望结果，创建一个数据帧，每列使用指定的浮点数据类型
    expected = DataFrame({"a": [1.0, np.nan], "b": [2.5, np.nan]})
    tm.assert_frame_equal(result, expected)
# 测试读取字符串存储模式的数据类型后端
def test_dtype_backend_string(all_parsers, string_storage):
    # 导入或跳过导入 pyarrow 库，如果导入失败则跳过该测试
    pa = pytest.importorskip("pyarrow")

    # 设置 pandas 上下文环境，指定字符串存储模式
    with pd.option_context("mode.string_storage", string_storage):
        # 使用给定的解析器
        parser = all_parsers

        # 定义测试数据
        data = """a,b
a,x
b,
"""

        # 使用解析器读取 CSV 数据，并指定数据类型后端为 "numpy_nullable"
        result = parser.read_csv(StringIO(data), dtype_backend="numpy_nullable")

        # 根据字符串存储模式不同，设置预期结果
        if string_storage == "python":
            expected = DataFrame(
                {
                    "a": StringArray(np.array(["a", "b"], dtype=np.object_)),
                    "b": StringArray(np.array(["x", pd.NA], dtype=np.object_)),
                }
            )
        else:
            expected = DataFrame(
                {
                    "a": ArrowStringArray(pa.array(["a", "b"])),
                    "b": ArrowStringArray(pa.array(["x", None])),
                }
            )

        # 使用测试工具函数检查结果是否与预期相同
        tm.assert_frame_equal(result, expected)


# 测试指定数据类型的数据类型后端
def test_dtype_backend_ea_dtype_specified(all_parsers):
    # GH#491496
    # 定义测试数据
    data = """a,b
1,2
"""

    # 使用给定的解析器
    parser = all_parsers

    # 使用解析器读取 CSV 数据，指定数据类型为 "Int64"，数据类型后端为 "numpy_nullable"
    result = parser.read_csv(
        StringIO(data), dtype="Int64", dtype_backend="numpy_nullable"
    )

    # 设置预期结果
    expected = DataFrame({"a": [1], "b": 2}, dtype="Int64")

    # 使用测试工具函数检查结果是否与预期相同
    tm.assert_frame_equal(result, expected)


# 测试 pyarrow 数据类型后端
def test_dtype_backend_pyarrow(all_parsers, request):
    # GH#36712
    # 导入或跳过导入 pyarrow 库，如果导入失败则跳过该测试
    pa = pytest.importorskip("pyarrow")

    # 使用给定的解析器
    parser = all_parsers

    # 定义包含日期解析的测试数据
    data = """a,b,c,d,e,f,g,h,i,j
1,2.5,True,a,,,,,12-31-2019,
3,4.5,False,b,6,7.5,True,a,12-31-2019,
"""

    # 使用解析器读取 CSV 数据，指定数据类型后端为 "pyarrow"，并解析指定列为日期
    result = parser.read_csv(StringIO(data), dtype_backend="pyarrow", parse_dates=["i"])

    # 设置预期结果，使用 pyarrow 数组类型表示
    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="int64[pyarrow]"),
            "b": pd.Series([2.5, 4.5], dtype="float64[pyarrow]"),
            "c": pd.Series([True, False], dtype="bool[pyarrow]"),
            "d": pd.Series(["a", "b"], dtype=pd.ArrowDtype(pa.string())),
            "e": pd.Series([pd.NA, 6], dtype="int64[pyarrow]"),
            "f": pd.Series([pd.NA, 7.5], dtype="float64[pyarrow]"),
            "g": pd.Series([pd.NA, True], dtype="bool[pyarrow]"),
            "h": pd.Series(
                [pd.NA, "a"],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "i": pd.Series([Timestamp("2019-12-31")] * 2),
            "j": pd.Series([pd.NA, pd.NA], dtype="null[pyarrow]"),
        }
    )

    # 使用测试工具函数检查结果是否与预期相同
    tm.assert_frame_equal(result, expected)


# 避免整数溢出的测试
@pytest.mark.usefixtures("pyarrow_xfail")
def test_ea_int_avoid_overflow(all_parsers):
    # GH#32134
    # 使用给定的解析器
    parser = all_parsers

    # 定义包含整数溢出情况的测试数据
    data = """a,b
1,1
,1
1582218195625938945,1
"""

    # 使用解析器读取 CSV 数据，指定列 "a" 的数据类型为 "Int64"
    result = parser.read_csv(StringIO(data), dtype={"a": "Int64"})

    # 设置预期结果，使用 IntegerArray 表示 "a" 列的数据
    expected = DataFrame(
        {
            "a": IntegerArray(
                np.array([1, 1, 1582218195625938945]), np.array([False, True, False])
            ),
            "b": 1,
        }
    )

    # 使用测试工具函数检查结果是否与预期相同
    tm.assert_frame_equal(result, expected)


# 测试字符串类型推断
def test_string_inference(all_parsers):
    # GH#54430
    # 尝试导入 pyarrow 库，如果导入失败则跳过当前测试
    pytest.importorskip("pyarrow")
    # 定义数据类型为 "string[pyarrow_numpy]"
    dtype = "string[pyarrow_numpy]"

    # 定义包含两列数据的字符串，每列由字母a和b组成
    data = """a,b
# GH#54868
# 使用给定的所有解析器对象进行CSV数据解析
parser = all_parsers
# 定义包含CSV格式数据的字符串
data = """a,b,c
1,2,3
4,5,6"""
# 使用解析器对象解析CSV数据，仅选择"a"和"c"列，并将"a"列指定为对象类型
result = parser.read_csv(StringIO(data), usecols=["a", "c"], dtype={"a": object})
# 根据解析器引擎选择合适的数值或字符串值
if parser.engine == "pyarrow":
    values = [1, 4]
else:
    values = ["1", "4"]
# 构建预期的DataFrame对象
expected = DataFrame({"a": pd.Series(values, dtype=object), "c": [3, 6]})
# 检查解析结果是否与预期一致
tm.assert_frame_equal(result, expected)
```