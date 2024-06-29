# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_parse_dates.py`

```
"""
Tests date parsing functionality for all of the
parsers defined in parsers.py
"""

# 导入所需模块和库
from datetime import (
    datetime,            # 导入 datetime 类型
    timedelta,           # 导入 timedelta 类型
    timezone,            # 导入 timezone 类型
)
from io import StringIO  # 导入 StringIO 类

import numpy as np       # 导入 numpy 库
import pytest            # 导入 pytest 测试框架

import pandas as pd                   # 导入 pandas 库
from pandas import (                  # 导入 pandas 中的多个模块和类
    DataFrame,                         # DataFrame 类
    DatetimeIndex,                     # DatetimeIndex 类
    Index,                             # Index 类
    MultiIndex,                        # MultiIndex 类
    Series,                            # Series 类
    Timestamp,                         # Timestamp 类
)
import pandas._testing as tm           # 导入 pandas 内部测试模块
from pandas.core.indexes.datetimes import date_range      # 导入日期范围相关模块
from pandas.core.tools.datetimes import start_caching_at   # 导入日期工具模块

from pandas.io.parsers import read_csv    # 导入 pandas 中的 read_csv 函数

# 设置 pytest 标记，忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 设置 pytest 标记，对 pyarrow 进行标记，标记为期望失败
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
# 设置 pytest 标记，对 pyarrow 进行标记，标记为跳过
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


# 定义测试函数：测试将日期列作为索引列
def test_date_col_as_index_col(all_parsers):
    # 准备测试数据
    data = """\
KORD,19990127 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
"""
    parser = all_parsers   # 使用传入的所有解析器
    kwds = {
        "header": None,                     # 无标题行
        "parse_dates": [1],                 # 解析第二列作为日期时间
        "index_col": 1,                     # 将第二列作为索引列
        "names": ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"],  # 列名定义
    }
    # 执行 CSV 数据读取
    result = parser.read_csv(StringIO(data), **kwds)

    # 准备预期的 DataFrame 索引
    index = Index(
        [
            datetime(1999, 1, 27, 19, 0),
            datetime(1999, 1, 27, 20, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 22, 0),
        ],
        dtype="M8[s]",
        name="X1",
    )
    # 准备预期的 DataFrame 数据
    expected = DataFrame(
        [
            ["KORD", " 18:56:00", 0.81, 2.81, 7.2, 0.0, 280.0],
            ["KORD", " 19:56:00", 0.01, 2.21, 7.2, 0.0, 260.0],
            ["KORD", " 20:56:00", -0.59, 2.21, 5.7, 0.0, 280.0],
            ["KORD", " 21:18:00", -0.99, 2.01, 3.6, 0.0, 270.0],
            ["KORD", " 21:56:00", -0.59, 1.71, 5.1, 0.0, 290.0],
        ],
        columns=["X0", "X2", "X3", "X4", "X5", "X6", "X7"],
        index=index,
    )
    # 如果使用 pyarrow 引擎，则调整预期中的时间数据类型
    if parser.engine == "pyarrow":
        # https://github.com/pandas-dev/pandas/issues/44231
        # pyarrow 6.0 开始推断时间类型
        expected["X2"] = pd.to_datetime("1970-01-01" + expected["X2"]).dt.time

    # 使用测试模块中的方法验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)


# 标记为期望失败的测试函数：测试自然日期解析
@xfail_pyarrow
def test_nat_parse(all_parsers):
    # 见 gh-3062
    parser = all_parsers
    # 准备测试 DataFrame
    df = DataFrame(
        {
            "A": np.arange(10, dtype="float64"),
            "B": Timestamp("20010101"),
        }
    )
    # 将部分行设置为 NaN
    df.iloc[3:6, :] = np.nan

    # 使用测试模块中的方法，确保生成临时 CSV 文件并读取
    with tm.ensure_clean("__nat_parse_.csv") as path:
        df.to_csv(path)

        # 读取 CSV 文件，并解析日期列
        result = parser.read_csv(path, index_col=0, parse_dates=["B"])
        # 使用测试模块中的方法验证结果与预期是否相等
        tm.assert_frame_equal(result, df)


# 标记为跳过的测试函数：测试隐式将第一列作为日期解析
@skip_pyarrow
def test_parse_dates_implicit_first_col(all_parsers):
    data = """A,B,C
# 使用不同的解析器读取 CSV 数据并进行日期解析，比较结果是否符合预期
parser = all_parsers
# 使用默认参数解析 CSV 数据，并尝试解析日期
result = parser.read_csv(StringIO(data), parse_dates=True)

# 指定索引列为第一列，并解析日期
expected = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
tm.assert_frame_equal(result, expected)


@xfail_pyarrow
# 测试解析日期功能，使用不同的解析器
def test_parse_dates_string(all_parsers):
    data = """date,A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    # 读取 CSV 数据并将 "date" 列解析为日期类型，忽略频率差异
    result = parser.read_csv(StringIO(data), index_col="date", parse_dates=["date"])
    index = date_range("1/1/2009", periods=3, name="date", unit="s")._with_freq(None)

    # 构建预期的 DataFrame
    expected = DataFrame(
        {"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]}, index=index
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize("parse_dates", [[0, 2], ["a", "c"]])
# 测试解析指定列作为日期的功能，使用不同的解析器和参数化的日期列
def test_parse_dates_column_list(all_parsers, parse_dates):
    data = "a,b,c\n01/01/2010,1,15/02/2010"
    parser = all_parsers

    # 构建预期的 DataFrame
    expected = DataFrame(
        {"a": [datetime(2010, 1, 1)], "b": [1], "c": [datetime(2010, 2, 15)]}
    )
    expected["a"] = expected["a"].astype("M8[s]")
    expected["c"] = expected["c"].astype("M8[s]")
    expected = expected.set_index(["a", "b"])

    # 读取 CSV 数据并根据参数化的日期列索引进行日期解析
    result = parser.read_csv(
        StringIO(data), index_col=[0, 1], parse_dates=parse_dates, dayfirst=True
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize("index_col", [[0, 1], [1, 0]])
# 测试多级索引和日期解析功能，使用不同的解析器和参数化的索引列
def test_multi_index_parse_dates(all_parsers, index_col):
    data = """index1,index2,A,B,C
20090101,one,a,1,2
20090101,two,b,3,4
20090101,three,c,4,5
20090102,one,a,1,2
20090102,two,b,3,4
20090102,three,c,4,5
20090103,one,a,1,2
20090103,two,b,3,4
20090103,three,c,4,5
"""
    parser = all_parsers
    dti = date_range("2009-01-01", periods=3, freq="D", unit="s")
    index = MultiIndex.from_product(
        [
            dti,
            ("one", "two", "three"),
        ],
        names=["index1", "index2"],
    )

    # 如果索引列是 [1, 0]，则交换索引顺序
    if index_col == [1, 0]:
        index = index.swaplevel(0, 1)

    # 构建预期的 DataFrame
    expected = DataFrame(
        [
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
        ],
        columns=["A", "B", "C"],
        index=index,
    )
    # 读取 CSV 数据并根据参数化的索引列和日期解析
    result = parser.read_csv_check_warnings(
        UserWarning,
        "Could not infer format",
        StringIO(data),
        index_col=index_col,
        parse_dates=True,
    )
    tm.assert_frame_equal(result, expected)


def test_parse_tz_aware(all_parsers):
    # 测试时区感知的日期解析功能，参考 gh-1693
    parser = all_parsers
    data = "Date,x\n2012-06-13T01:39:00Z,0.5"

    # 读取 CSV 数据并根据索引列解析日期
    result = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    expected = DataFrame(
        {"x": [0.5]}, index=Index([Timestamp("2012-06-13 01:39:00+00:00")], name="Date")
    )
    # 如果解析器的引擎是 "pyarrow"，则导入 pytest 并检查是否有 pytz 模块
    if parser.engine == "pyarrow":
        pytz = pytest.importorskip("pytz")
        # 预期时区设为 UTC
        expected_tz = pytz.utc
    else:
        # 否则，预期时区设为 UTC
        expected_tz = timezone.utc
    # 使用测试工具比较结果和期望值的数据框，确保它们相等
    tm.assert_frame_equal(result, expected)
    # 断言结果数据框的索引时区与预期时区一致
    assert result.index.tz is expected_tz
@pytest.mark.parametrize("kwargs", [{}, {"index_col": "C"}])
# 使用 pytest 的参数化装饰器，为函数 test_read_with_parse_dates_scalar_non_bool 提供两组参数：空字典和{"index_col": "C"}
def test_read_with_parse_dates_scalar_non_bool(all_parsers, kwargs):
    # 测试函数，用于验证 parse_dates 参数只接受布尔值和列表类型
    # 参考 GitHub issue #5636
    parser = all_parsers
    msg = "Only booleans and lists " "are accepted for the 'parse_dates' parameter"
    # 测试数据，包含一个日期字符串
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        # 使用 pytest 的 raises 断言检查是否会抛出 TypeError 异常，并匹配指定的错误消息
        parser.read_csv(StringIO(data), parse_dates="C", **kwargs)


@pytest.mark.parametrize("parse_dates", [(1,), np.array([4, 5]), {1, 3}])
# 使用 pytest 的参数化装饰器，为函数 test_read_with_parse_dates_invalid_type 提供三组参数：(1,)、numpy 数组 [4, 5]、集合 {1, 3}
def test_read_with_parse_dates_invalid_type(all_parsers, parse_dates):
    # 测试函数，用于验证 parse_dates 参数只接受布尔值和列表类型
    parser = all_parsers
    msg = "Only booleans and lists " "are accepted for the 'parse_dates' parameter"
    # 测试数据，包含一个日期字符串
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        # 使用 pytest 的 raises 断言检查是否会抛出 TypeError 异常，并匹配指定的错误消息
        parser.read_csv(StringIO(data), parse_dates=parse_dates)


@pytest.mark.parametrize("value", ["nan", ""])
# 使用 pytest 的参数化装饰器，为函数 test_bad_date_parse 提供两组参数："nan" 和 ""
def test_bad_date_parse(all_parsers, cache, value):
    # 测试函数，用于验证处理无效日期时，是否正确处理缓存设置
    parser = all_parsers
    s = StringIO((f"{value},\n") * (start_caching_at + 1))

    parser.read_csv(
        s,
        header=None,
        names=["foo", "bar"],
        parse_dates=["foo"],
        cache_dates=cache,
    )


@pytest.mark.parametrize("value", ["0"])
# 使用 pytest 的参数化装饰器，为函数 test_bad_date_parse_with_warning 提供参数 "0"
def test_bad_date_parse_with_warning(all_parsers, cache, value):
    # 测试函数，用于验证处理无效日期时，是否正确处理警告和缓存设置
    parser = all_parsers
    s = StringIO((f"{value},\n") * 50000)

    if parser.engine == "pyarrow":
        # 如果使用 pyarrow 引擎，它将 "0" 读取为整数 0（类型为 int64），因此 pandas 不会尝试猜测日期时间格式
        # TODO: 在 pyarrow 中直接解析日期，参见 GitHub issue https://github.com/pandas-dev/pandas/issues/48017
        warn = None
    elif cache:
        # 注意：如果设置了 'cache_dates'，则不会发出警告，因为这里只有一个唯一的日期，所以不存在不一致解析的风险。
        warn = None
    else:
        warn = UserWarning
    parser.read_csv_check_warnings(
        warn,
        "Could not infer format",
        s,
        header=None,
        names=["foo", "bar"],
        parse_dates=["foo"],
        cache_dates=cache,
        raise_on_extra_warnings=False,
    )


def test_parse_dates_empty_string(all_parsers):
    # 测试函数，用于验证解析空字符串日期的行为
    # 参考 GitHub issue #2263
    parser = all_parsers
    data = "Date,test\n2012-01-01,1\n,2"
    result = parser.read_csv(StringIO(data), parse_dates=["Date"], na_filter=False)

    expected = DataFrame(
        [[datetime(2012, 1, 1), 1], [pd.NaT, 2]], columns=["Date", "test"]
    )
    expected["Date"] = expected["Date"].astype("M8[s]")
    # 使用 pandas 的 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            "a\n04.15.2016",  # 第一个示例数据：包含日期字符串的单列DataFrame
            {"parse_dates": ["a"]},  # 解析列 'a' 作为日期列
            DataFrame([datetime(2016, 4, 15)], columns=["a"], dtype="M8[s]"),  # 创建日期为2016年4月15日的DataFrame，列名为'a'，数据类型为datetime64[s]
        ),
        (
            "a\n04.15.2016",  # 第二个示例数据：包含日期字符串的单列DataFrame
            {"parse_dates": True, "index_col": 0},  # 解析所有列作为日期列，并将第一列作为索引列
            DataFrame(  # 创建一个空列的DataFrame，索引为日期2016年4月15日，数据类型为datetime64[s]，列为空
                index=DatetimeIndex(["2016-04-15"], dtype="M8[s]", name="a"), columns=[]
            ),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",  # 第三个示例数据：包含两列日期字符串的DataFrame
            {"parse_dates": ["a", "b"]},  # 解析列 'a' 和 'b' 作为日期列
            DataFrame(  # 创建包含日期为2016年4月15日和2013年9月16日的DataFrame，列名为'a'和'b'，数据类型为datetime64[s]
                [[datetime(2016, 4, 15), datetime(2013, 9, 16)]],
                dtype="M8[s]",
                columns=["a", "b"],
            ),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",  # 第四个示例数据：包含两列日期字符串的DataFrame
            {"parse_dates": True, "index_col": [0, 1]},  # 解析所有列作为日期列，并将第一列和第二列作为层次化索引
            DataFrame(  # 创建一个空列的DataFrame，层次化索引为日期2016年4月15日和2013年9月16日，数据类型为datetime64[s]，列为空
                index=MultiIndex.from_tuples(
                    [
                        (
                            Timestamp(2016, 4, 15).as_unit("s"),  # 将2016年4月15日转换为时间戳（以秒为单位）
                            Timestamp(2013, 9, 16).as_unit("s"),  # 将2013年9月16日转换为时间戳（以秒为单位）
                        )
                    ],
                    names=["a", "b"],  # 层次化索引的名称分别为'a'和'b'
                ),
                columns=[],
            ),
        ),
    ],
@pytest.mark.parametrize(
    "date_string,dayfirst,expected",
    [
        ("32/32/2019", False, "32/32/2019"),   # Test case with an invalid date format
        ("02/30/2019", False, "02/30/2019"),   # Test case with an invalid date format
        ("13/13/2019", False, "13/13/2019"),   # Test case with an invalid date format
        ("13/2019", False, "13/2019"),         # Test case with an invalid date format
        ("a3/11/2018", False, "a3/11/2018"),   # Test case with an invalid date format
        ("10/11/2o17", False, "10/11/2o17"),   # Test case with an invalid date format
    ],
)
@skip_pyarrow  # Skip this test for pyarrow due to CSV parse error
def test_invalid_parse_delimited_date(all_parsers, date_string):
    # see gh-2697
    #
    # Testing various invalid date formats to ensure they are correctly parsed as strings.
    parser = all_parsers

    # Creating expected DataFrame with the invalid date string as the single element
    expected = DataFrame({0: [date_string]}, dtype="object")

    # Parsing the invalid date string with pandas read_csv
    result = parser.read_csv(
        StringIO(date_string),
        header=None,
        parse_dates=[0],
    )

    # Asserting that the result matches the expected DataFrame
    tm.assert_frame_equal(result, expected)
    [
        # 第一个元组: "13/02/2019"，格式为 %d/%m/%Y；由于月份大于 12，会进行替换，预期结果为 True，对应的 datetime 为 2019年2月13日
        ("13/02/2019", True, datetime(2019, 2, 13)),
        # 第二个元组: "02/13/2019"，格式为 %m/%d/%Y；由于日期大于 12，不会进行替换，预期结果为 False，对应的 datetime 为 2019年2月13日
        ("02/13/2019", False, datetime(2019, 2, 13)),
        # 第三个元组: "04/02/2019"，格式为 %d/%m/%Y 且 dayfirst==True，会进行替换，预期结果为 True，对应的 datetime 为 2019年2月4日
        ("04/02/2019", True, datetime(2019, 2, 4)),
    ],
# ArrowInvalid: CSV parse error: Empty CSV file or block: cannot infer number of columns
@skip_pyarrow
# 使用 pytest 装饰器跳过 pyarrow 引擎的测试，因为它无法处理 dayfirst 选项
@pytest.mark.parametrize(
    "date_string,dayfirst,expected",
    [
        # %d/%m/%Y; month > 12
        ("13/02/2019", False, datetime(2019, 2, 13)),
        # %m/%d/%Y; day > 12
        ("02/13/2019", True, datetime(2019, 2, 13)),
    ],
)
# 测试解析带有日期交换的 CSV 数据，期望引发特定错误或生成预期的日期时间对象
def test_parse_delimited_date_swap_with_warning(
    all_parsers, date_string, dayfirst, expected
):
    # 使用给定的所有解析器
    parser = all_parsers
    # 创建预期结果的 DataFrame，以确保日期时间数据为 datetime64[s] 类型
    expected = DataFrame({0: [expected]}, dtype="datetime64[s]")
    # 设置警告消息的正则表达式模式，用于检查警告内容是否符合预期
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )
    # 使用 parser.read_csv_check_warnings 方法解析 CSV 数据，同时检查警告消息
    result = parser.read_csv_check_warnings(
        UserWarning,
        warning_msg,
        StringIO(date_string),
        header=None,
        dayfirst=dayfirst,
        parse_dates=[0],
    )
    # 断言解析结果与预期结果相等
    tm.assert_frame_equal(result, expected)


# ArrowKeyError: Column 'fdate1' in include_columns does not exist in CSV file
@skip_pyarrow
# 使用 pytest 装饰器跳过 pyarrow 引擎的测试，因为列 'fdate1' 在 CSV 文件中不存在
@pytest.mark.parametrize(
    "names, usecols, parse_dates, missing_cols",
    [
        (None, ["val"], ["date", "time"], "date, time"),
        (None, ["val"], [0, "time"], "time"),
        (["date1", "time1", "temperature"], None, ["date", "time"], "date, time"),
        (
            ["date1", "time1", "temperature"],
            ["date1", "temperature"],
            ["date1", "time"],
            "time",
        ),
    ],
)
# 测试当指定的解析日期列不存在时是否会引发 ValueError
def test_missing_parse_dates_column_raises(
    all_parsers, names, usecols, parse_dates, missing_cols
):
    # 使用给定的所有解析器
    parser = all_parsers
    # 准备包含特定内容的内存缓冲区对象，模拟 CSV 文件
    content = StringIO("date,time,val\n2020-01-31,04:20:32,32\n")
    # 构造错误信息，指示缺失的列在 'parse_dates' 参数中
    msg = f"Missing column provided to 'parse_dates': '{missing_cols}'"

    # 使用 pytest 模块的 raises 函数，验证是否抛出 ValueError 异常，并检查异常消息是否匹配预期
    with pytest.raises(ValueError, match=msg):
        # 调用 parser 对象的 read_csv 方法，解析给定的 CSV 内容
        parser.read_csv(
            content, sep=",", names=names, usecols=usecols, parse_dates=parse_dates
        )
@xfail_pyarrow  # 标记此测试为预期失败，原因是形状不匹配
def test_date_parser_and_names(all_parsers):
    # GH#33699
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 创建包含数据的字符串流对象
    data = StringIO("""x,y\n1,2""")
    # 设置警告默认为 UserWarning 类型
    warn = UserWarning
    # 如果解析器的引擎是 "pyarrow"，则更新 warn 变量为元组 (UserWarning, DeprecationWarning)
    if parser.engine == "pyarrow":
        warn = (UserWarning, DeprecationWarning)
    # 使用解析器读取 CSV 数据并检查警告信息，设置结果到 result 变量中
    result = parser.read_csv_check_warnings(
        warn,
        "Could not infer format",
        data,
        parse_dates=["B"],
        names=["B"],
    )
    # 创建预期结果的数据框架对象
    expected = DataFrame({"B": ["y", "2"]}, index=["x", "1"])
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 标记此测试为预期失败，原因是需要整数类型
def test_date_parser_multiindex_columns(all_parsers):
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 设置包含数据的字符串
    data = """a,b
1,2
2019-12-31,6"""
    # 使用解析器读取 CSV 数据，设置结果到 result 变量中
    result = parser.read_csv(StringIO(data), parse_dates=[("a", "1")], header=[0, 1])
    # 创建预期结果的数据框架对象
    expected = DataFrame({("a", "1"): Timestamp("2019-12-31"), ("b", "2"): [6]})
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)


def test_date_parser_usecols_thousands(all_parsers):
    # GH#39365
    # 设置包含数据的字符串
    data = """A,B,C
    1,3,20-09-01-01
    2,4,20-09-01-01
    """
    # 从参数中获取所有解析器对象
    parser = all_parsers

    # 如果解析器的引擎是 "pyarrow"，则抛出 ValueError 异常，验证是否包含特定消息
    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                parse_dates=[1],
                usecols=[1, 2],
                thousands="-",
            )
        return

    # 使用解析器读取 CSV 数据并检查警告信息，设置结果到 result 变量中
    result = parser.read_csv_check_warnings(
        UserWarning,
        "Could not infer format",
        StringIO(data),
        parse_dates=[1],
        usecols=[1, 2],
        thousands="-",
    )
    # 创建预期结果的数据框架对象
    expected = DataFrame({"B": [3, 4], "C": [Timestamp("20-09-2001 01:00:00")] * 2})
    expected["C"] = expected["C"].astype("M8[s]")
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)


def test_dayfirst_warnings():
    # GH 12585

    # CASE 1: valid input
    # 设置包含日期数据的输入字符串
    input = "date\n31/12/2014\n10/03/2011"
    # 创建预期结果的日期索引对象
    expected = DatetimeIndex(
        ["2014-12-31", "2011-03-10"], dtype="datetime64[s]", freq=None, name="date"
    )
    # 设置警告消息的正则表达式模式
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )

    # A. dayfirst 参数正确，不应产生警告
    # 使用 read_csv 函数读取 CSV 数据，并设置日期解析与索引列
    res1 = read_csv(
        StringIO(input), parse_dates=["date"], dayfirst=True, index_col="date"
    ).index
    # 使用测试框架验证结果与预期是否相等
    tm.assert_index_equal(expected, res1)

    # B. dayfirst 参数错误，应产生警告
    # 使用 read_csv 函数读取 CSV 数据，并设置日期解析与索引列，验证是否产生特定警告消息
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res2 = read_csv(
            StringIO(input), parse_dates=["date"], dayfirst=False, index_col="date"
        ).index
    # 使用测试框架验证结果与预期是否相等
    tm.assert_index_equal(expected, res2)

    # CASE 2: invalid input
    # 无法一致处理单一格式的输入，直接返回给用户
    input = "date\n31/12/2014\n03/30/2011"
    expected = Index(["31/12/2014", "03/30/2011"], dtype="object", name="date")

    # A. 使用 dayfirst=True 参数进行日期解析
    res5 = read_csv(
        StringIO(input), parse_dates=["date"], dayfirst=True, index_col="date"
    ).index
    # 验证解析后的索引与预期结果是否相等
    tm.assert_index_equal(expected, res5)

    # B. 使用 dayfirst=False 参数进行日期解析，并带有警告
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res6 = read_csv(
            StringIO(input), parse_dates=["date"], dayfirst=False, index_col="date"
        ).index
    # 验证解析后的索引与预期结果是否相等
    tm.assert_index_equal(expected, res6)
@pytest.mark.parametrize(
    "date_string, dayfirst",
    [  # 参数化测试数据，包括日期字符串和是否 dayfirst 的标志位
        pytest.param(
            "31/1/2014",
            False,
            id="second date is single-digit",  # 参数化测试的标识符
        ),
        pytest.param(
            "1/31/2014",
            True,
            id="first date is single-digit",  # 参数化测试的标识符
        ),
    ],
)
def test_dayfirst_warnings_no_leading_zero(date_string, dayfirst):
    # GH47880
    initial_value = f"date\n{date_string}"  # 构造初始数据字符串
    expected = DatetimeIndex(
        ["2014-01-31"], dtype="datetime64[s]", freq=None, name="date"
    )  # 期望的日期索引对象
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )  # 期望的警告消息正则表达式
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res = read_csv(
            StringIO(initial_value),
            parse_dates=["date"],  # 指定要解析为日期的列
            index_col="date",  # 指定索引列
            dayfirst=dayfirst,  # 是否优先考虑日期中的天
        ).index  # 获取读取数据后的索引
    tm.assert_index_equal(expected, res)  # 断言期望的索引和实际的索引相等


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
def test_infer_first_column_as_index(all_parsers):
    # GH#11019
    parser = all_parsers  # 使用所有的解析器
    data = "a,b,c\n1970-01-01,2,3,4"  # 输入的数据字符串
    result = parser.read_csv(
        StringIO(data),
        parse_dates=["a"],  # 指定要解析为日期的列
    )  # 执行 CSV 解析
    expected = DataFrame({"a": "2", "b": 3, "c": 4}, index=["1970-01-01"])  # 期望的数据框架
    tm.assert_frame_equal(result, expected)  # 断言期望的数据框架和实际的数据框架相等


@xfail_pyarrow  # pyarrow engine doesn't support passing a dict for na_values
def test_replace_nans_before_parsing_dates(all_parsers):
    # GH#26203
    parser = all_parsers  # 使用所有的解析器
    data = """Test
2012-10-01
0
2015-05-15
#
2017-09-09
"""  # 输入的数据字符串，包含 NaN 值
    result = parser.read_csv(
        StringIO(data),
        na_values={"Test": ["#", "0"]},  # 指定 NaN 值的替换字典
        parse_dates=["Test"],  # 指定要解析为日期的列
        date_format="%Y-%m-%d",  # 指定日期的格式
    )  # 执行 CSV 解析
    expected = DataFrame(
        {
            "Test": [
                Timestamp("2012-10-01"),
                pd.NaT,
                Timestamp("2015-05-15"),
                pd.NaT,
                Timestamp("2017-09-09"),
            ]
        },
        dtype="M8[s]",  # 期望的数据框架，指定数据类型为 datetime64[s]
    )
    tm.assert_frame_equal(result, expected)  # 断言期望的数据框架和实际的数据框架相等


@xfail_pyarrow  # string[python] instead of dt64[ns]
def test_parse_dates_and_string_dtype(all_parsers):
    # GH#34066
    parser = all_parsers  # 使用所有的解析器
    data = """a,b
1,2019-12-31
"""  # 输入的数据字符串
    result = parser.read_csv(StringIO(data), dtype="string", parse_dates=["b"])  # 执行 CSV 解析，指定数据类型和解析为日期的列
    expected = DataFrame({"a": ["1"], "b": [Timestamp("2019-12-31")]})  # 期望的数据框架
    expected["a"] = expected["a"].astype("string")  # 将列 'a' 转换为字符串类型
    expected["b"] = expected["b"].astype("M8[s]")  # 将列 'b' 转换为 datetime64[s] 类型
    tm.assert_frame_equal(result, expected)  # 断言期望的数据框架和实际的数据框架相等


def test_parse_dot_separated_dates(all_parsers):
    # https://github.com/pandas-dev/pandas/issues/2586
    parser = all_parsers  # 使用所有的解析器
    data = """a,b
27.03.2003 14:55:00.000,1
03.08.2003 15:20:00.000,2"""  # 输入的数据字符串，包含日期和时间
    # 如果解析器引擎为 "pyarrow"，则使用 Index 类来创建预期的索引对象
    if parser.engine == "pyarrow":
        expected_index = Index(
            ["27.03.2003 14:55:00.000", "03.08.2003 15:20:00.000"],
            dtype="object",
            name="a",
        )
        # 不生成警告
        warn = None
    # 如果解析器引擎不是 "pyarrow"，则使用 DatetimeIndex 类来创建预期的索引对象
    else:
        expected_index = DatetimeIndex(
            ["2003-03-27 14:55:00", "2003-08-03 15:20:00"],
            dtype="datetime64[ms]",
            name="a",
        )
        # 生成 UserWarning 类型的警告
        warn = UserWarning
    # 消息内容，用于在读取 CSV 时生成警告
    msg = r"when dayfirst=False \(the default\) was specified"
    # 调用解析器的 read_csv_check_warnings 方法进行 CSV 文件读取，并检查警告
    result = parser.read_csv_check_warnings(
        warn,
        msg,
        StringIO(data),
        parse_dates=True,
        index_col=0,
        raise_on_extra_warnings=False,
    )
    # 创建预期的 DataFrame 对象，用于与实际读取的数据进行比较
    expected = DataFrame({"b": [1, 2]}, index=expected_index)
    # 使用测试工具函数 tm.assert_frame_equal 检查结果是否与预期一致
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试解析 CSV 并处理日期格式的情况，通过给定的所有解析器进行测试
def test_parse_dates_dict_format(all_parsers):
    # 标记：GH#51240
    # 使用给定的解析器对象
    parser = all_parsers
    # 准备包含日期格式数据的 CSV 字符串
    data = """a,b
2019-12-31,31-12-2019
2020-12-31,31-12-2020"""

    # 使用解析器对象读取 CSV 数据，并指定日期格式字典进行日期解析
    result = parser.read_csv(
        StringIO(data),
        date_format={"a": "%Y-%m-%d", "b": "%d-%m-%Y"},
        parse_dates=["a", "b"],
    )
    # 期望的数据帧，包含日期时间戳，并指定数据类型为秒级的日期时间
    expected = DataFrame(
        {
            "a": [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
            "b": [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
        },
        dtype="M8[s]",
    )
    # 断言结果与期望是否相等
    tm.assert_frame_equal(result, expected)


# 标记为暂时失败的测试函数，用于测试带有日期格式字典和索引列的 CSV 解析
@xfail_pyarrow  # object dtype index
def test_parse_dates_dict_format_index(all_parsers):
    # 标记：GH#51240
    # 使用给定的解析器对象
    parser = all_parsers
    # 准备包含日期格式数据的 CSV 字符串
    data = """a,b
2019-12-31,31-12-2019
2020-12-31,31-12-2020"""

    # 使用解析器对象读取 CSV 数据，并指定日期格式字典进行日期解析，同时将第一列设置为索引
    result = parser.read_csv(
        StringIO(data), date_format={"a": "%Y-%m-%d"}, parse_dates=True, index_col=0
    )
    # 期望的数据帧，仅包含索引为日期时间戳的列'b'
    expected = DataFrame(
        {
            "b": ["31-12-2019", "31-12-2020"],
        },
        index=Index([Timestamp("2019-12-31"), Timestamp("2020-12-31")], name="a"),
    )
    # 断言结果与期望是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，用于测试使用 Arrow 引擎解析 CSV 文件并处理日期数据
def test_parse_dates_arrow_engine(all_parsers):
    # 标记：GH#53295
    # 使用给定的解析器对象
    parser = all_parsers
    # 准备包含日期时间数据的 CSV 字符串
    data = """a,b
2000-01-01 00:00:00,1
2000-01-01 00:00:01,1"""

    # 使用解析器对象读取 CSV 数据，并指定 'a' 列为日期时间解析
    result = parser.read_csv(StringIO(data), parse_dates=["a"])
    # 期望的数据帧，包含 'a' 列为日期时间戳，'b' 列为整数值
    expected = DataFrame(
        {
            "a": [
                Timestamp("2000-01-01 00:00:00"),
                Timestamp("2000-01-01 00:00:01"),
            ],
            "b": 1,
        }
    )
    # 断言结果与期望是否相等
    tm.assert_frame_equal(result, expected)


# 标记为暂时失败的测试函数，用于测试包含混合偏移的 CSV 文件解析
@xfail_pyarrow  # object dtype index
def test_from_csv_with_mixed_offsets(all_parsers):
    # 使用给定的解析器对象
    parser = all_parsers
    # 准备包含混合偏移日期时间数据的 CSV 字符串
    data = "a\n2020-01-01T00:00:00+01:00\n2020-01-01T00:00:00+00:00"
    # 使用解析器对象读取 CSV 数据，并指定 'a' 列为日期时间解析，然后选择 'a' 列作为结果
    result = parser.read_csv(StringIO(data), parse_dates=["a"])["a"]
    # 期望的 Series，包含 'a' 列为混合偏移的日期时间字符串，指定索引为 [0, 1]
    expected = Series(
        [
            "2020-01-01T00:00:00+01:00",
            "2020-01-01T00:00:00+00:00",
        ],
        name="a",
        index=[0, 1],
    )
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)
```