# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_loc.py`

```
# 导入必要的模块和库
from collections import namedtuple  # 导入namedtuple模块，用于创建命名元组
from datetime import (  # 导入多个datetime模块中的类和函数
    date,
    datetime,
    time,
    timedelta,
)
import re  # 导入re模块，用于正则表达式操作

from dateutil.tz import gettz  # 导入gettz函数，用于获取时区信息
import numpy as np  # 导入numpy库，并简写为np
import pytest  # 导入pytest库，用于单元测试

from pandas._config import using_pyarrow_string_dtype  # 导入using_pyarrow_string_dtype函数
from pandas._libs import index as libindex  # 导入libindex模块作为_pandas._libs.index别名
from pandas.compat.numpy import np_version_gt2  # 导入np_version_gt2函数
from pandas.errors import IndexingError  # 导入IndexingError异常类

import pandas as pd  # 导入pandas库，并简写为pd
from pandas import (  # 从pandas库中导入多个类和函数
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IndexSlice,
    MultiIndex,
    Period,
    PeriodIndex,
    Series,
    SparseDtype,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm  # 导入pandas._testing模块，并简写为tm
from pandas.api.types import is_scalar  # 导入is_scalar函数
from pandas.core.indexing import _one_ellipsis_message  # 导入_one_ellipsis_message函数
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises  # 导入check_indexing_smoketest_or_raises函数

@pytest.mark.parametrize(  # 使用pytest.mark.parametrize装饰器标记参数化测试
    "series, new_series, expected_ser",  # 参数化的变量名称
    [  # 参数化的测试数据
        [[np.nan, np.nan, "b"], ["a", np.nan, np.nan], [False, True, True]],  # 第一组参数化数据
        [[np.nan, "b"], ["a", np.nan], [False, True]],  # 第二组参数化数据
    ],
)
def test_not_change_nan_loc(series, new_series, expected_ser):
    # 测试函数：验证loc方法对NaN值的处理
    df = DataFrame({"A": series})  # 创建DataFrame对象df，列名为"A"，数据为series
    df.loc[:, "A"] = new_series  # 使用loc方法将"A"列的值设置为new_series
    expected = DataFrame({"A": expected_ser})  # 创建预期的DataFrame对象expected，列名为"A"，数据为expected_ser
    tm.assert_frame_equal(df.isna(), expected)  # 使用tm.assert_frame_equal断言方法比较df.isna()和expected是否相等
    tm.assert_frame_equal(df.notna(), ~expected)  # 使用tm.assert_frame_equal断言方法比较df.notna()和~expected是否相等


class TestLoc:
    def test_none_values_on_string_columns(self):
        # 测试函数：验证字符串列中的None值处理
        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")  # 创建dtype为"str"的DataFrame对象df，列名为"a"

        assert df.loc[2, "a"] is None  # 使用loc方法检索第2行、"a"列的值，并断言其为None

    def test_loc_getitem_int(self, frame_or_series):
        # 测试函数：验证loc方法对整数标签的处理
        # int标签
        obj = frame_or_series(range(3), index=Index(list("abc"), dtype=object))  # 创建对象obj，数据为range(3)，索引为Index(list("abc"))
        check_indexing_smoketest_or_raises(obj, "loc", 2, fails=KeyError)  # 调用check_indexing_smoketest_or_raises函数，验证obj的loc[2]是否引发KeyError异常

    def test_loc_getitem_label(self, frame_or_series):
        # 测试函数：验证loc方法对标签的处理
        # 标签
        obj = frame_or_series()  # 创建对象obj
        check_indexing_smoketest_or_raises(obj, "loc", "c", fails=KeyError)  # 调用check_indexing_smoketest_or_raises函数，验证obj的loc["c"]是否引发KeyError异常

    @pytest.mark.parametrize("key", ["f", 20])  # 参数化测试，key为["f", 20]
    @pytest.mark.parametrize(  # 参数化测试
        "index",  # 参数化的变量名称
        [  # 参数化的测试数据
            Index(list("abcd"), dtype=object),  # Index对象，数据为["a", "b", "c", "d"]
            Index([2, 4, "null", 8], dtype=object),  # Index对象，数据为[2, 4, "null", 8]
            date_range("20130101", periods=4),  # DatetimeIndex对象，包含从20130101开始的4个日期
            Index(range(0, 8, 2), dtype=np.float64),  # Index对象，数据为[0.0, 2.0, 4.0, 6.0]，数据类型为np.float64
            Index([]),  # 空的Index对象
        ],
    )
    def test_loc_getitem_label_out_of_range(self, key, index, frame_or_series):
        obj = frame_or_series(range(len(index)), index=index)  # 创建对象obj，数据为range(len(index))，索引为index
        # 超出范围的标签
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)  # 调用check_indexing_smoketest_or_raises函数，验证obj的loc[key]是否引发KeyError异常

    @pytest.mark.parametrize("key", [[0, 1, 2], [1, 3.0, "A"]])  # 参数化测试，key为[[0, 1, 2], [1, 3.0, "A"]]
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])  # 参数化测试，dtype为[np.int64, np.uint64, np.float64]
    @pytest.mark.parametrize(
        "slc, indexes, axes, fails",
        [  # 参数化测试用例的参数，包括切片、索引、轴和预期失败类型
            [
                slice(1, 3),  # 切片对象，表示索引范围为1到3
                [
                    Index(list("abcd"), dtype=object),  # 对象索引列表，包含字符列表和对象类型
                    Index([2, 4, "null", 8], dtype=object),  # 对象索引列表，包含整数、字符串和对象类型
                    None,  # 空索引
                    date_range("20130101", periods=4),  # 日期范围索引
                    Index(range(0, 12, 3), dtype=np.float64),  # 对象索引列表，包含浮点数类型
                ],
                None,  # 轴为空
                TypeError,  # 预期的异常类型为TypeError
            ],
            [
                slice("20130102", "20130104"),  # 字符串范围切片
                [date_range("20130101", periods=4)],  # 日期范围索引
                1,  # 轴为1
                TypeError,  # 预期的异常类型为TypeError
            ],
            [
                slice(2, 8),  # 整数范围切片
                [Index([2, 4, "null", 8], dtype=object)],  # 对象索引列表，包含整数、字符串和对象类型
                0,  # 轴为0
                TypeError,  # 预期的异常类型为TypeError
            ],
            [
                slice(2, 8),  # 整数范围切片
                [Index([2, 4, "null", 8], dtype=object)],  # 对象索引列表，包含整数、字符串和对象类型
                1,  # 轴为1
                KeyError,  # 预期的异常类型为KeyError
            ],
            [
                slice(2, 4, 2),  # 步长为2的整数范围切片
                [Index([2, 4, "null", 8], dtype=object)],  # 对象索引列表，包含整数、字符串和对象类型
                0,  # 轴为0
                TypeError,  # 预期的异常类型为TypeError
            ],
        ],
    )
    def test_loc_getitem_label_slice(self, slc, indexes, axes, fails, frame_or_series):
        # label slices (with ints)
        # 标签切片（包含整数）

        # real label slices
        # 真实的标签切片

        # GH 14316
        # GitHub issue 14316

        for index in indexes:
            if index is None:
                obj = frame_or_series()
            else:
                obj = frame_or_series(range(len(index)), index=index)
            # 检查索引操作的基本功能或引发异常
            check_indexing_smoketest_or_raises(
                obj,
                "loc",
                slc,
                axes=axes,
                fails=fails,
            )
    def test_setitem_from_duplicate_axis(self):
        # GH#34034
        # 创建一个包含重复索引的DataFrame对象
        df = DataFrame(
            [[20, "a"], [200, "a"], [200, "a"]],
            columns=["col1", "col2"],
            index=[10, 1, 1],
        )
        # 使用.loc方法修改特定位置的值为数组
        df.loc[1, "col1"] = np.arange(2)
        # 创建预期的DataFrame对象，以比较测试结果
        expected = DataFrame(
            [[20, "a"], [0, "a"], [1, "a"]], columns=["col1", "col2"], index=[10, 1, 1]
        )
        # 使用assert_frame_equal进行DataFrame对象的比较
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self):
        # GH 26779
        # 创建一个包含不同数据类型的DataFrame对象
        df = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", np.nan, "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    pd.NaT,
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        # 创建第二个DataFrame对象
        df2 = DataFrame(
            data={"A": ["String 3"], "B": [Timestamp("2019-06-11 12:00:00")]}
        )
        # 在原始DataFrame中，当'A'列的值为NaN时，使用df2的值进行替换
        df.loc[df["A"].isna(), ["A", "B"]] = df2.values
        # 创建预期的DataFrame对象，以比较测试结果
        expected = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", "String 3", "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        # 使用assert_frame_equal进行DataFrame对象的比较
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "obj, key, exp",
        [
            (
                DataFrame([[1]], columns=Index([False])),
                IndexSlice[:, False],
                Series([1], name=False),
            ),
            (Series([1], index=Index([False])), False, [1]),
            (DataFrame([[1]], index=Index([False])), False, Series([1], name=False)),
        ],
    )
    def test_loc_getitem_single_boolean_arg(self, obj, key, exp):
        # GH 44322
        # 使用.loc进行基于布尔值的索引操作
        res = obj.loc[key]
        # 根据exp的类型，使用tm.assert_equal或assert进行结果比较
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp
class TestLocBaseIndependent:
    # Tests for loc that do not depend on subclassing Base

    # Test case for loc with numpy string array indexing
    def test_loc_npstr(self):
        # GH#45580
        # Create a DataFrame with datetime index from "2021" to "2022"
        df = DataFrame(index=date_range("2021", "2022"))
        # Perform loc operation using numpy array with a date string
        result = df.loc[np.array(["2021/6/1"])[0] :]
        # Expected result using iloc to retrieve rows from index position 151 onwards
        expected = df.iloc[151:]
        # Assert equality between result and expected DataFrame
        tm.assert_frame_equal(result, expected)

    # Parameterized test case for various key-value pairs
    @pytest.mark.parametrize(
        "msg, key",
        [
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'bar'", (Period(2019), "foo", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'y1', 'bar'", (Period(2019), "y1", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'z1'", (Period(2019), "foo", "z1")),
            (
                r"Period\('2018', 'Y-DEC'\), Period\('2016', 'Y-DEC'\), 'bar'",
                (Period(2018), Period(2016), "bar"),
            ),
            (r"Period\('2018', 'Y-DEC'\), 'foo', 'y1'", (Period(2018), "foo", "y1")),
            (
                r"Period\('2017', 'Y-DEC'\), 'foo', Period\('2015', 'Y-DEC'\)",
                (Period(2017), "foo", Period(2015)),
            ),
            (r"Period\('2017', 'Y-DEC'\), 'z1', 'bar'", (Period(2017), "z1", "bar")),
        ],
    )
    def test_contains_raise_error_if_period_index_is_in_multi_index(self, msg, key):
        # GH#20684
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        # Create a DataFrame with multi-index based on Period objects
        df = DataFrame(
            {
                "A": [Period(2019), "x1", "x2"],
                "B": [Period(2018), Period(2016), "y1"],
                "C": [Period(2017), "z1", Period(2015)],
                "V1": [1, 2, 3],
                "V2": [10, 20, 30],
            }
        ).set_index(["A", "B", "C"])
        # Assert KeyError is raised with specified message when accessing loc with key
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    # Test case to check loc behavior with Unicode key
    def test_loc_getitem_missing_unicode_key(self):
        # Create a DataFrame with a column "a" containing [1]
        df = DataFrame({"a": [1]})
        # Assert KeyError is raised with Unicode key "\u05d0"
        with pytest.raises(KeyError, match="\u05d0"):
            df.loc[:, "\u05d0"]  # should not raise UnicodeEncodeError

    # Test case to check loc behavior with duplicated index
    def test_loc_getitem_dups(self):
        # GH 5678
        # Create a DataFrame with random values and duplicated index
        df = DataFrame(
            np.random.default_rng(2).random((20, 5)),
            index=["ABCDE"[x % 5] for x in range(20)],
        )
        # Expected result of loc operation on duplicated index "A" and column 0
        expected = df.loc["A", 0]
        # Result of repeated loc operations on column 0 and index "A"
        result = df.loc[:, 0].loc["A"]
        # Assert equality between result and expected Series
        tm.assert_series_equal(result, expected)
    def test_loc_getitem_dups2(self):
        # 测试用例：处理重复索引的iloc/loc操作
        df = DataFrame(
            [[1, 2, "foo", "bar", Timestamp("20130101")]],
            columns=["a", "a", "a", "a", "a"],
            index=[1],
        )
        expected = Series(
            [1, 2, "foo", "bar", Timestamp("20130101")],
            index=["a", "a", "a", "a", "a"],
            name=1,
        )

        # 使用iloc获取第一行数据，应与预期的Series相等
        result = df.iloc[0]
        tm.assert_series_equal(result, expected)

        # 使用loc获取索引为1的数据，应与预期的Series相等
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self):
        # 测试用例：处理重复索引的loc设置项
        df_orig = DataFrame(
            {
                "me": list("rttti"),
                "foo": list("aaade"),
                "bar": np.arange(5, dtype="float64") * 1.34 + 2,
                "bar2": np.arange(5, dtype="float64") * -0.34 + 2,
            }
        ).set_index("me")

        # 定义索引器，指定行为'r'，列为['bar', 'bar2']，对应数据乘以2.0
        indexer = (
            "r",
            ["bar", "bar2"],
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

        # 定义索引器，指定行为'r'，列为'bar'，对应数据乘以2.0
        indexer = (
            "r",
            "bar",
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]

        # 定义索引器，指定行为't'，列为['bar', 'bar2']，对应数据乘以2.0
        indexer = (
            "t",
            ["bar", "bar2"],
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self):
        # 测试用例：处理loc切片设置项
        # 分配相同类型的数据不应改变类型
        df1 = DataFrame({"a": [0, 1, 1], "b": Series([100, 200, 300], dtype="uint32")})
        ix = df1["a"] == 1
        newb1 = df1.loc[ix, "b"] + 1
        df1.loc[ix, "b"] = newb1
        expected = DataFrame(
            {"a": [0, 1, 1], "b": Series([100, 201, 301], dtype="uint32")}
        )
        tm.assert_frame_equal(df1, expected)

        # 分配新类型的数据应获取推断的类型
        df2 = DataFrame({"a": [0, 1, 1], "b": [100, 200, 300]}, dtype="uint64")
        ix = df1["a"] == 1
        newb2 = df2.loc[ix, "b"]
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df1.loc[ix, "b"] = newb2
        expected = DataFrame({"a": [0, 1, 1], "b": [100, 200, 300]}, dtype="uint64")
        tm.assert_frame_equal(df2, expected)
    def test_loc_setitem_dtype(self):
        # GH31340
        # 创建一个包含"id", "a", "b", "c"列的DataFrame对象
        df = DataFrame({"id": ["A"], "a": [1.2], "b": [0.0], "c": [-2.5]})
        # 列表cols包含要处理的列名
        cols = ["a", "b", "c"]
        # 将DataFrame中cols列的数据类型转换为float32
        df.loc[:, cols] = df.loc[:, cols].astype("float32")

        # 在2.0版本之前，这种设置会交换新的数组，现在在2.0中是正确地原地操作，与非分割路径一致
        # 创建一个期望的DataFrame对象，包含与之前对应列相同的数据，数据类型为float64
        expected = DataFrame(
            {
                "id": ["A"],
                "a": np.array([1.2], dtype="float64"),
                "b": np.array([0.0], dtype="float64"),
                "c": np.array([-2.5], dtype="float64"),
            }
        )  # id被推断为object类型

        # 使用测试工具比较df和期望值expected的DataFrame对象是否相等
        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self):
        # 创建一个Series对象，索引为["a", "b", "c"]，数据为[0, 1, 2]
        s = Series(range(3), index=["a", "b", "c"])

        # 一致性检查，预期会抛出KeyError异常，匹配错误信息"not in index"
        with pytest.raises(KeyError, match="not in index"):
            s[["a", "d"]]

        # 重新赋值s为一个索引为[0, 1, 2]的Series对象
        s = Series(range(3))
        # 预期会抛出KeyError异常，匹配错误信息"not in index"
        with pytest.raises(KeyError, match="not in index"):
            s[[0, 3]]

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index):
        # GH26658
        # 创建一个Series对象，数据为[1, 2, 3]
        s = Series([1, 2, 3])
        # 构造错误信息字符串，指出布尔索引的长度错误
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        # 预期会抛出IndexError异常，匹配错误信息msg
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self):
        # TODO: test something here?
        # 暂时没有具体的测试内容

    def test_loc_to_fail(self):
        # GH3449
        # 创建一个3x3的DataFrame对象，数据为随机数
        df = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            index=["a", "b", "c"],
            columns=["e", "f", "g"],
        )

        # 构造错误信息字符串，指出索引[1, 2]不在列索引中
        msg = (
            rf"\"None of \[Index\(\[1, 2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        # 预期会抛出KeyError异常，匹配错误信息msg
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self):
        # GH  7496
        # loc不应该回退

        # 创建一个dtype为object的Series对象
        s = Series(dtype=object)
        # 向索引为1的位置插入值1
        s.loc[1] = 1
        # 向索引为"a"的位置插入值2
        s.loc["a"] = 2

        # 预期会抛出KeyError异常，匹配错误信息"-1"
        with pytest.raises(KeyError, match=r"^-1$"):
            s.loc[-1]

        # 构造错误信息字符串，指出索引[-1, -2]不在索引中
        msg = (
            rf"\"None of \[Index\(\[-1, -2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        # 预期会抛出KeyError异常，匹配错误信息msg
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]

        # 构造错误信息字符串，指出索引['4']不在索引中
        msg = r"\"None of \[Index\(\['4'\], dtype='object'\)\] are in the \[index\]\""
        # 预期会抛出KeyError异常，匹配错误信息msg
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(["4"], dtype=object)]

        # 向索引为-1的位置插入值3
        s.loc[-1] = 3
        # 预期会抛出KeyError异常，匹配错误信息"not in index"
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[-1, -2]]

        # 向索引为"a"的位置插入值2
        s["a"] = 2
        # 构造错误信息字符串，指出索引[-2]不在索引中
        msg = (
            rf"\"None of \[Index\(\[-2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        # 预期会抛出KeyError异常，匹配错误信息msg
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]

        # 删除索引为"a"的元素
        del s["a"]

        # 预期会抛出KeyError异常，匹配错误信息msg
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0
    def test_loc_to_fail3(self):
        # 检查.loc[values]和.loc[values,:]之间的不一致性
        # GH 7999
        # 创建一个包含单列"value"的DataFrame，索引为[1, 2]
        df = DataFrame([["a"], ["b"]], index=[1, 2], columns=["value"])

        # 准备用于匹配的错误消息，包含特定格式的数据类型信息
        msg = (
            rf"\"None of \[Index\(\[3\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        # 期望捕获KeyError，并匹配预设的错误消息
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]

        # 再次捕获KeyError，并匹配相同的错误消息
        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self):
        # GH 15747
        # 如果有*任何*缺失标签，应该引发KeyError

        # 创建一个包含整数序列[1, 2, 3]的Series
        s = Series([1, 2, 3])

        # 使用.loc[[2]]获取特定标签的值

        # 准备用于匹配的错误消息，包含特定格式的RangeIndex信息
        msg = "None of [RangeIndex(start=3, stop=4, step=1)] are in the [index]"
        # 期望捕获KeyError，并匹配预设的错误消息
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]

        # 期望捕获KeyError，错误消息包含"not in index"
        # 一次测试中有非匹配和匹配的情况
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[2, 3]]

    def test_loc_index(self):
        # GH 17131
        # 布尔索引应该像布尔numpy数组一样进行索引

        # 创建一个5行10列的DataFrame，包含随机生成的数据，索引名包含"alpha"和"beta"
        df = DataFrame(
            np.random.default_rng(2).random(size=(5, 10)),
            index=["alpha_0", "alpha_1", "alpha_2", "beta_0", "beta_1"],
        )

        # 根据索引名包含"alpha"的布尔值索引创建期望的DataFrame
        mask = df.index.map(lambda x: "alpha" in x)
        expected = df.loc[np.array(mask)]

        # 使用布尔值索引直接索引DataFrame，并与期望结果进行比较
        result = df.loc[mask]
        tm.assert_frame_equal(result, expected)

        # 使用布尔值索引的.values属性索引DataFrame，并与期望结果进行比较
        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)

        # 使用pd.array创建布尔索引，索引DataFrame，并与期望结果进行比较
        result = df.loc[pd.array(mask, dtype="boolean")]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self):
        # 创建一个4行4列的DataFrame，包含随机生成的数据，列名为["A", "B", "C", "D"]，索引名为["A", "B", "C", "D"]
        df = DataFrame(
            np.random.default_rng(2).random((4, 4)),
            columns=["A", "B", "C", "D"],
            index=["A", "B", "C", "D"],
        )

        # 测试.loc[:, "A":"B"]的操作，希望结果包含列["A", "B"]，并且行索引为["A", "B"]
        result = df.loc[:, "A":"B"].iloc[0:2, :]
        assert (result.columns == ["A", "B"]).all()
        assert (result.index == ["A", "B"]).all()

        # 混合类型的测试
        # 创建一个包含时间戳和整数的DataFrame，并与期望的Series进行比较
        result = DataFrame({"a": [Timestamp("20130101")], "b": [1]}).iloc[0]
        expected = Series([Timestamp("20130101"), 1], index=["a", "b"], name=0)
        tm.assert_series_equal(result, expected)
        assert result.dtype == object

    @pytest.fixture
    def frame_for_consistency(self):
        # 返回一个DataFrame，包含"date"列和"val"列，分别为日期范围和整数序列
        return DataFrame(
            {
                "date": date_range("2000-01-01", "2000-01-5"),
                "val": Series(range(5), dtype=np.int64),
            }
        )

    @pytest.mark.parametrize(
        "val",
        [0, np.array(0, dtype=np.int64), np.array([0, 0, 0, 0, 0], dtype=np.int64)],
    )
    def test_loc_setitem_consistency(self, frame_for_consistency, val):
        # GH 6149
        # 当行具有空切片时，对于 setitem 和 loc，进行类似的强制转换
        expected = DataFrame(
            {
                "date": Series(0, index=range(5), dtype=np.int64),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 对 df 中的 "date" 列进行设置为 val，使用 loc 方法
            df.loc[:, "date"] = val
        # 检查 df 是否等于预期的 DataFrame
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency):
        # GH 6149
        # 当行具有空切片时，对于 setitem 和 loc，进行类似的强制转换
        expected = DataFrame(
            {
                "date": Series("foo", index=range(5)),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 对 df 中的 "date" 列进行设置为字符串 "foo"，使用 loc 方法
            df.loc[:, "date"] = "foo"
        # 检查 df 是否等于预期的 DataFrame
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency):
        # GH 6149
        # 当行具有空切片时，对于 setitem 和 loc，进行类似的强制转换
        expected = DataFrame(
            {
                "date": Series(1.0, index=range(5)),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 对 df 中的 "date" 列进行设置为浮点数 1.0，使用 loc 方法
            df.loc[:, "date"] = 1.0
        # 检查 df 是否等于预期的 DataFrame
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_single_row(self):
        # GH 15494
        # 在只有单行的 DataFrame 上进行设置
        df = DataFrame({"date": Series([Timestamp("20180101")])})
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 对 df 中的 "date" 列进行设置为字符串 "string"，使用 loc 方法
            df.loc[:, "date"] = "string"
        expected = DataFrame({"date": Series(["string"])})
        # 检查 df 是否等于预期的 DataFrame
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_empty(self):
        # empty (essentially noops)
        # 在空 DataFrame 上进行设置
        # 在 2.0 版本之前，#45333 强制生效前，这里的 loc.setitem 会改变 df.x 的 dtype 至 int64
        expected = DataFrame(columns=["x", "y"])
        df = DataFrame(columns=["x", "y"])
        with tm.assert_produces_warning(None):
            # 对 df 中的 "x" 列进行设置为整数 1，使用 loc 方法
            df.loc[:, "x"] = 1
        # 检查 df 是否等于预期的 DataFrame
        tm.assert_frame_equal(df, expected)

        # 使用 setitem 进行设置会引入新的数组，因此会改变 dtype
        df = DataFrame(columns=["x", "y"])
        df["x"] = 1
        expected["x"] = expected["x"].astype(np.int64)
        # 检查 df 是否等于预期的 DataFrame
        tm.assert_frame_equal(df, expected)
    def test_loc_setitem_consistency_slice_column_len(self):
        # .loc[:,column] setting with slice == len of the column
        # GH10408
        levels = [
            ["Region_1"] * 4,  # 创建包含四个相同值的列表
            ["Site_1", "Site_1", "Site_2", "Site_2"],  # 不同的站点名称列表
            [3987227376, 3980680971, 3977723249, 3977723089],  # RespondentID 列表
        ]
        mi = MultiIndex.from_arrays(levels, names=["Region", "Site", "RespondentID"])  # 用给定的数组创建多级索引对象

        clevels = [
            ["Respondent", "Respondent", "Respondent", "OtherCat", "OtherCat"],  # 第一级和第二级列名列表
            ["Something", "StartDate", "EndDate", "Yes/No", "SomethingElse"],  # 第三级和第四级列名列表
        ]
        cols = MultiIndex.from_arrays(clevels, names=["Level_0", "Level_1"])  # 用给定的数组创建多级列索引对象

        values = [
            ["A", "5/25/2015 10:59", "5/25/2015 11:22", "Yes", np.nan],  # 数据值列表
            ["A", "5/21/2015 9:40", "5/21/2015 9:52", "Yes", "Yes"],  # 数据值列表
            ["A", "5/20/2015 8:27", "5/20/2015 8:41", "Yes", np.nan],  # 数据值列表
            ["A", "5/20/2015 8:33", "5/20/2015 9:09", "Yes", "No"],  # 数据值列表
        ]
        df = DataFrame(values, index=mi, columns=cols)  # 使用给定的值、索引和列创建 DataFrame 对象

        df.loc[:, ("Respondent", "StartDate")] = to_datetime(
            df.loc[:, ("Respondent", "StartDate")]  # 将 "Respondent" 列中的 "StartDate" 列转换为 datetime 类型
        )
        df.loc[:, ("Respondent", "EndDate")] = to_datetime(
            df.loc[:, ("Respondent", "EndDate")]  # 将 "Respondent" 列中的 "EndDate" 列转换为 datetime 类型
        )
        df = df.infer_objects()  # 推断 DataFrame 中的对象类型

        # Adding a new key
        df.loc[:, ("Respondent", "Duration")] = (
            df.loc[:, ("Respondent", "EndDate")]  # 计算 "Duration" 列
            - df.loc[:, ("Respondent", "StartDate")]  # 通过减去 "StartDate" 列来计算
        )

        # timedelta64[m] -> float, so this cannot be done inplace, so
        #  no warning
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[:, ("Respondent", "Duration")] = df.loc[
                :, ("Respondent", "Duration")  # 将 "Duration" 列转换为浮点数
            ] / Timedelta(60_000_000_000)  # 将 "Duration" 列的时间差转换为分钟

        expected = Series(
            [23.0, 12.0, 14.0, 36.0], index=df.index, name=("Respondent", "Duration")  # 期望的 Series 对象
        )
        tm.assert_series_equal(df[("Respondent", "Duration")], expected)  # 断言 "Respondent" 列的 "Duration" 列与期望的 Series 对象相等

    @pytest.mark.parametrize("unit", ["Y", "M", "D", "h", "m", "s", "ms", "us"])
    def test_loc_assign_non_ns_datetime(self, unit):
        # GH 27395, non-ns dtype assignment via .loc should work
        # and return the same result when using simple assignment
        df = DataFrame(
            {
                "timestamp": [
                    np.datetime64("2017-02-11 12:41:29"),
                    np.datetime64("1991-11-07 04:22:37"),
                ]
            }
        )

        df.loc[:, unit] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")  # 将 "timestamp" 列的值转换为指定单位的 datetime64 类型
        df["expected"] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")  # 将 "timestamp" 列的值转换为指定单位的 datetime64 类型
        expected = Series(df.loc[:, "expected"], name=unit)  # 期望的 Series 对象
        tm.assert_series_equal(df.loc[:, unit], expected)  # 断言指定单位的列与期望的 Series 对象相等
    def test_loc_modify_datetime(self):
        # 修正日期时间测试，参见 GitHub issue 28837
        df = DataFrame.from_dict(
            {"date": [1485264372711, 1485265925110, 1540215845888, 1540282121025]}
        )

        # 将毫秒时间戳转换为日期时间格式，并缓存结果
        df["date_dt"] = to_datetime(df["date"], unit="ms", cache=True).dt.as_unit("ms")

        # 使用 loc 方法在 DataFrame 中设置新的列 "date_dt_cp"，并赋值为 "date_dt" 列的值
        df.loc[:, "date_dt_cp"] = df.loc[:, "date_dt"]

        # 使用 loc 方法在 DataFrame 中选择特定行并设置 "date_dt_cp" 列的值为 "date_dt" 列的值
        df.loc[[2, 3], "date_dt_cp"] = df.loc[[2, 3], "date_dt"]

        # 期望的结果 DataFrame
        expected = DataFrame(
            [
                [1485264372711, "2017-01-24 13:26:12.711", "2017-01-24 13:26:12.711"],
                [1485265925110, "2017-01-24 13:52:05.110", "2017-01-24 13:52:05.110"],
                [1540215845888, "2018-10-22 13:44:05.888", "2018-10-22 13:44:05.888"],
                [1540282121025, "2018-10-23 08:08:41.025", "2018-10-23 08:08:41.025"],
            ],
            columns=["date", "date_dt", "date_dt_cp"],
        )

        # 将 "date_dt" 和 "date_dt_cp" 列的值转换为日期时间格式
        columns = ["date_dt", "date_dt_cp"]
        expected[columns] = expected[columns].apply(to_datetime)

        # 使用 assert_frame_equal 检查 df 和期望的结果 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex(self):
        # GitHub issue 6254 设置问题
        # 创建一个具有浮点数类型列 "A" 的 DataFrame，索引为 [3, 5, 4]
        df = DataFrame(index=[3, 5, 4], columns=["A"], dtype=float)
        
        # 使用 loc 方法在 DataFrame 中设置指定索引和列 "A" 的值为整数类型数组
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")

        # 创建预期结果 DataFrame，其中列 "A" 包含一个 Series
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({"A": ser})

        # 使用 assert_frame_equal 检查 df 和期望的结果 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self):
        # GitHub issue 40480
        # 创建一个具有浮点数类型列 "A" 和字符串类型列 "B" 的 DataFrame，索引为 [3, 5, 4]
        df = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        
        # 设置列 "B" 的值为字符串 "string"
        df["B"] = "string"
        
        # 使用 loc 方法在 DataFrame 中设置指定索引和列 "A" 的值为整数类型数组，并将 Series 类型索引转换为浮点数
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        
        # 创建预期结果 DataFrame，其中列 "A" 包含一个整数类型 Series，列 "B" 为字符串 "string"
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        expected = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"

        # 使用 assert_frame_equal 检查 df 和期望的结果 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self):
        # GitHub issue 40480
        # 创建一个具有浮点数类型列 "A" 和字符串类型列 "B" 的 DataFrame，索引为 [1, 2, 3]
        df = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        
        # 设置列 "B" 的值为字符串 "string"
        df["B"] = "string"
        
        # 使用 loc 方法在 DataFrame 中使用逆序切片设置列 "A" 的值为整数类型数组，并保持索引顺序
        df.loc[slice(3, 0, -1), "A"] = np.array([1, 2, 3], dtype="int64")
        
        # 创建预期结果 DataFrame，其中列 "A" 包含一个浮点数类型列表，列 "B" 为字符串 "string"
        expected = DataFrame({"A": [3.0, 2.0, 1.0], "B": "string"}, index=[1, 2, 3])

        # 使用 assert_frame_equal 检查 df 和期望的结果 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
    def test_loc_setitem_empty_frame(self):
        # GH#6252 setting with an empty frame
        # 创建一组字符串键 ["@0", "@1", "@2", "@3", "@4"]
        keys1 = ["@" + str(i) for i in range(5)]
        # 创建一个包含整数 0 到 4 的 numpy 数组，数据类型为 int64
        val1 = np.arange(5, dtype="int64")

        # 创建另一组字符串键 ["@0", "@1", "@2", "@3"]
        keys2 = ["@" + str(i) for i in range(4)]
        # 创建一个包含整数 0 到 3 的 numpy 数组，数据类型为 int64
        val2 = np.arange(4, dtype="int64")

        # 将 keys1 和 keys2 的并集作为索引创建一个空的 DataFrame 对象 df
        index = list(set(keys1).union(keys2))
        df = DataFrame(index=index)
        
        # 初始化 df 中的列 "A" 和 "B" 为 NaN
        df["A"] = np.nan
        df["B"] = np.nan
        
        # 使用 loc 方法将 val1 中的值分配给 df 的列 "A" 中的对应位置
        df.loc[keys1, "A"] = val1
        
        # 使用 loc 方法将 val2 中的值分配给 df 的列 "B" 中的对应位置
        df.loc[keys2, "B"] = val2

        # 因为 df["A"] 最初被初始化为 float64，设置值时是就地修改，因此数据类型保持不变
        # 创建一个 Series 对象 sera，索引为 keys1，数据为 val1，数据类型为 np.float64
        sera = Series(val1, index=keys1, dtype=np.float64)
        # 创建一个 Series 对象 serb，索引为 keys2，数据为 val2，数据类型为默认的 dtype
        serb = Series(val2, index=keys2)
        
        # 创建一个预期的 DataFrame 对象 expected，包含列 "A" 和 "B"，并重新索引为 index
        expected = DataFrame(
            {"A": sera, "B": serb}, columns=Index(["A", "B"], dtype=object)
        ).reindex(index=index)
        
        # 使用 pytest 的 assert_frame_equal 方法比较 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self):
        # 创建一个 4x4 的 DataFrame 对象 df，数据为标准正态分布随机数
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=list("abcd"),
            columns=list("ABCD"),
        )

        # 获取 df 的第一个元素，保存在 result 中
        result = df.iloc[0, 0]

        # 使用 loc 方法将值 1 分配给 df 中 "a" 行 "A" 列的位置
        df.loc["a", "A"] = 1
        # 获取 df 中 "a" 行 "A" 列的值，保存在 result 中
        result = df.loc["a", "A"]
        # 使用 assert 语句检查 result 是否等于 1
        assert result == 1

        # 再次获取 df 的第一个元素，保存在 result 中
        result = df.iloc[0, 0]
        # 使用 assert 语句检查 result 是否等于 1
        assert result == 1

        # 使用 loc 方法将值 0 分配给 df 中所有行的 "B" 到 "D" 列的位置
        df.loc[:, "B":"D"] = 0
        # 获取 df 中所有行的 "B" 到 "D" 列的子 DataFrame，保存在 expected 中
        expected = df.loc[:, "B":"D"]
        # 获取 df 中所有行的第 1 列到最后一列的子 DataFrame，保存在 result 中
        result = df.iloc[:, 1:]
        # 使用 pytest 的 assert_frame_equal 方法比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self):
        # GH 8669
        # invalid coercion of nan -> int
        # 创建一个包含两列 "A" 和 "B" 的 DataFrame 对象 df，其中 "B" 列包含 NaN 值
        df = DataFrame({"A": [1, 2, 3], "B": np.nan})
        
        # 使用 loc 方法根据条件 df.B > df.A，将 df 中 "B" 列的值设置为 df 中对应行的 "A" 列的值
        df.loc[df.B > df.A, "B"] = df.A
        
        # 创建一个预期的 DataFrame 对象 expected，与 df 结构一致，但值不变
        expected = DataFrame({"A": [1, 2, 3], "B": np.nan})
        
        # 使用 pytest 的 assert_frame_equal 方法比较 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self):
        # GH 6546
        # setting with mixed labels
        # 创建一个包含三列的 DataFrame 对象 df，列名分别为 1, 2, 'a'
        df = DataFrame({1: [1, 2], 2: [3, 4], "a": ["a", "b"]})

        # 使用 loc 方法获取 df 中第 0 行的列 1 和 2 对应的 Series，保存在 result 中
        result = df.loc[0, [1, 2]]
        # 创建一个预期的 Series 对象 expected，数据为 [1, 3]，索引为 [1, 2]
        expected = Series(
            [1, 3], index=Index([1, 2], dtype=object), dtype=object, name=0
        )
        # 使用 pytest 的 assert_series_equal 方法比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个预期的 DataFrame 对象 expected，将 df 中第 0 行的列 1 和 2 的值设置为 [5, 6]
        expected = DataFrame({1: [5, 2], 2: [6, 4], "a": ["a", "b"]})
        # 使用 loc 方法将 df 中第 0 行的列 1 和 2 的值设置为 [5, 6]
        df.loc[0, [1, 2]] = [5, 6]
        # 使用 pytest 的 assert_frame_equal 方法比较 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)
    def test_loc_setitem_frame_multiples(self):
        # 定义测试方法：测试 loc 设置多行数据

        # 创建一个 DataFrame 对象，包含两列数据
        df = DataFrame(
            {"A": ["foo", "bar", "baz"], "B": Series(range(3), dtype=np.int64)}
        )
        
        # 使用 df.loc[1:2] 获取索引为 1 到 2 的行作为 rhs
        rhs = df.loc[1:2]
        
        # 设置 rhs 的索引为 df 的索引的前两个值
        rhs.index = df.index[0:2]
        
        # 用 rhs 替换 df 中索引为 0 到 1 的行
        df.loc[0:1] = rhs
        
        # 创建期望的 DataFrame 对象，验证 df 的期望结果
        expected = DataFrame(
            {"A": ["bar", "baz", "baz"], "B": Series([1, 2, 2], dtype=np.int64)}
        )
        
        # 使用 assert_frame_equal 检查 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建第二个测试场景：设置多行数据，rhs 是一个 DataFrame 对象
        df = DataFrame(
            {
                "date": date_range("2000-01-01", "2000-01-5"),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        
        # 创建期望的 DataFrame 对象，带有 Timestamp 类型的日期列
        expected = DataFrame(
            {
                "date": [
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000103"),
                ],
                "val": Series([0, 1, 0, 1, 2], dtype=np.int64),
            }
        )
        
        # 将 expected 的 "date" 列转换为 'M8[ns]' 类型
        expected["date"] = expected["date"].astype("M8[ns]")
        
        # 使用 df.loc[0:2] 获取索引为 0 到 2 的行作为 rhs
        rhs = df.loc[0:2]
        
        # 设置 rhs 的索引为 df 的索引的第三到第五个值
        rhs.index = df.index[2:5]
        
        # 用 rhs 替换 df 中索引为 2 到 4 的行
        df.loc[2:4] = rhs
        
        # 使用 assert_frame_equal 检查 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "indexer", [["A"], slice(None, "A", None), np.array(["A"])]
    )
    @pytest.mark.parametrize("value", [["Z"], np.array(["Z"])])
    def test_loc_setitem_with_scalar_index(self, indexer, value):
        # GH #19474
        # 当像 "df.loc[0, ['A']] = ['Z']" 这样赋值时应该逐个元素地评估，
        # 而不是使用 "setter('A', ['Z'])"。

        # 创建一个 DataFrame 对象，确保 'A' 列的 dtype 是 object 类型
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"]).astype({"A": object})
        
        # 使用 df.loc[0, indexer] 将 value 赋值给指定索引位置
        df.loc[0, indexer] = value
        
        # 获取 df.loc[0, 'A'] 的结果
        result = df.loc[0, "A"]
        
        # 断言结果是标量且等于 "Z"
        assert is_scalar(result) and result == "Z"
    # 使用 pytest 的 parametrize 装饰器，为测试方法 test_loc_setitem_missing_columns 参数化多组输入
    @pytest.mark.parametrize(
        "index,box,expected",
        [
            (
                # 第一组输入参数：index 为列表 [0, 2]，box 为列表 ["A", "B", "C", "D"]，expected 为指定的 DataFrame
                ([0, 2], ["A", "B", "C", "D"]),
                7,
                DataFrame(
                    [[7, 7, 7, 7], [3, 4, np.nan, np.nan], [7, 7, 7, 7]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                # 第二组输入参数：index 为 1，box 为列表 ["C", "D"]，expected 为指定的 DataFrame
                (1, ["C", "D"]),
                [7, 8],
                DataFrame(
                    [[1, 2, np.nan, np.nan], [3, 4, 7, 8], [5, 6, np.nan, np.nan]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                # 第三组输入参数：index 为 1，box 为列表 ["A", "B", "C"]，expected 为指定的 DataFrame
                (1, ["A", "B", "C"]),
                np.array([7, 8, 9], dtype=np.int64),
                DataFrame(
                    [[1, 2, np.nan], [7, 8, 9], [5, 6, np.nan]], columns=["A", "B", "C"]
                ),
            ),
            (
                # 第四组输入参数：index 为切片 slice(1, 3, None)，box 为列表 ["B", "C", "D"]，expected 为指定的 DataFrame
                (slice(1, 3, None), ["B", "C", "D"]),
                [[7, 8, 9], [10, 11, 12]],
                DataFrame(
                    [[1, 2, np.nan, np.nan], [3, 7, 8, 9], [5, 10, 11, 12]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                # 第五组输入参数：index 为切片 slice(1, 3, None)，box 为列表 ["C", "A", "D"]，expected 为指定的 DataFrame
                (slice(1, 3, None), ["C", "A", "D"]),
                np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64),
                DataFrame(
                    [[1, 2, np.nan, np.nan], [8, 4, 7, 9], [11, 6, 10, 12]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                # 第六组输入参数：index 为切片 slice(None, None, None)，box 为指定的 DataFrame，expected 为指定的 DataFrame
                (slice(None, None, None), ["A", "C"]),
                DataFrame([[7, 8], [9, 10], [11, 12]], columns=["A", "C"]),
                DataFrame(
                    [[7, 2, 8], [9, 4, 10], [11, 6, 12]], columns=["A", "B", "C"]
                ),
            ),
        ],
    )
    # 定义测试方法 test_loc_setitem_missing_columns，测试 DataFrame 的 loc 赋值操作
    def test_loc_setitem_missing_columns(self, index, box, expected):
        # GH 29334
        # 创建一个测试用的 DataFrame df，包含特定的数据和列名
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "B"])

        # 在 df 上使用 loc 方法对指定的 index 范围赋值为 box
        df.loc[index] = box
        # 使用 pytest 的 assert_frame_equal 方法验证 df 是否与 expected 相等
        tm.assert_frame_equal(df, expected)

    # 定义测试方法 test_loc_coercion，测试 DataFrame 的 iloc 方法和数据类型转换
    def test_loc_coercion(self):
        # GH#12411
        # 创建一个包含日期的 DataFrame df
        df = DataFrame({"date": [Timestamp("20130101").tz_localize("UTC"), pd.NaT]})
        # 获取 df 的数据类型
        expected = df.dtypes

        # 使用 iloc 方法选取索引为 [0] 的行，保存结果为 result
        result = df.iloc[[0]]
        # 使用 pytest 的 assert_series_equal 方法验证 result 的数据类型是否与 expected 相等
        tm.assert_series_equal(result.dtypes, expected)

        # 使用 iloc 方法选取索引为 [1] 的行，保存结果为 result
        result = df.iloc[[1]]
        # 使用 pytest 的 assert_series_equal 方法验证 result 的数据类型是否与 expected 相等
        tm.assert_series_equal(result.dtypes, expected)

    # 定义测试方法 test_loc_coercion2，测试 DataFrame 的 iloc 方法和数据类型转换
    def test_loc_coercion2(self):
        # GH#12045
        # 创建一个包含日期的 DataFrame df
        df = DataFrame({"date": [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        # 获取 df 的数据类型
        expected = df.dtypes

        # 使用 iloc 方法选取索引为 [0] 的行，保存结果为 result
        result = df.iloc[[0]]
        # 使用 pytest 的 assert_series_equal 方法验证 result 的数据类型是否与 expected 相等
        tm.assert_series_equal(result.dtypes, expected)

        # 使用 iloc 方法选取索引为 [1] 的行，保存结果为 result
        result = df.iloc[[1]]
        # 使用 pytest 的 assert_series_equal 方法验证 result 的数据类型是否与 expected 相等
        tm.assert_series_equal(result.dtypes, expected)
    def test_loc_coercion3(self):
        # 测试iloc方法中的位置强制转换
        # GH#11594
        # 创建一个DataFrame对象，包含一个字符串列和一些空值
        df = DataFrame({"text": ["some words"] + [None] * 9})
        # 获取预期的数据类型
        expected = df.dtypes

        # 使用iloc方法选择第0到1行（不包括第2行）
        result = df.iloc[0:2]
        # 断言结果的数据类型与预期相等
        tm.assert_series_equal(result.dtypes, expected)

        # 使用iloc方法选择第3行到最后一行
        result = df.iloc[3:]
        # 断言结果的数据类型与预期相等
        tm.assert_series_equal(result.dtypes, expected)

    def test_setitem_new_key_tz(self, indexer_sl):
        # 测试设置具有时区的新键值对应的方法
        # GH#12862 不应在分配第二个值时引发异常
        # 创建一个包含两个时区本地化的日期时间对象的Series
        vals = [
            to_datetime(42).tz_localize("UTC"),
            to_datetime(666).tz_localize("UTC"),
        ]
        # 创建一个预期的Series对象，指定索引为对象类型
        expected = Series(vals, index=Index(["foo", "bar"], dtype=object))

        # 创建一个空的Series对象，数据类型为对象
        ser = Series(dtype=object)
        # 使用索引器函数分配第一个值给键"foo"
        indexer_sl(ser)["foo"] = vals[0]
        # 使用索引器函数分配第二个值给键"bar"
        indexer_sl(ser)["bar"] = vals[1]

        # 断言Series对象与预期相等
        tm.assert_series_equal(ser, expected)

    def test_loc_non_unique(self):
        # 测试loc方法对非唯一索引的处理
        # GH3659
        # 使用非单调索引进行loc切片时会引发异常
        # 创建一个DataFrame对象，包含两列"A"和"B"，以及非单调的索引
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]
        )
        # 准备错误信息，指示无法获取非唯一标签的左切片边界
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        # 使用pytest断言应引发KeyError异常，并匹配预期的错误消息
        with pytest.raises(KeyError, match=msg):
            df.loc[1:]
        msg = "'Cannot get left slice bound for non-unique label: 0'"
        with pytest.raises(KeyError, match=msg):
            df.loc[0:]
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:2]

        # 对单调索引则不会引发异常
        # 对DataFrame对象按索引进行排序
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]
        ).sort_index(axis=0)
        # 使用loc方法选择索引大于等于1的行
        result = df.loc[1:]
        # 创建预期的DataFrame对象
        expected = DataFrame({"A": [2, 4, 5, 6], "B": [4, 6, 7, 8]}, index=[1, 1, 2, 3])
        # 断言结果DataFrame与预期相等
        tm.assert_frame_equal(result, expected)

        # 使用loc方法选择所有行
        result = df.loc[0:]
        # 断言结果DataFrame与原始DataFrame相等
        tm.assert_frame_equal(result, df)

        # 使用loc方法选择索引介于1和2之间的行
        result = df.loc[1:2]
        # 创建预期的DataFrame对象
        expected = DataFrame({"A": [2, 4, 5], "B": [4, 6, 7]}, index=[1, 1, 2])
        # 断言结果DataFrame与预期相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("length, l2", [[900, 100], [900000, 100000]])
    def test_loc_non_unique_memory_error(self, length, l2):
        # GH 4280
        # 大选择下非唯一索引触发内存错误

        # 定义列名列表
        columns = list("ABCDEFG")

        # 创建一个 DataFrame，包含随机标准正态分布数据和全为1的数据
        df = pd.concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((length, len(columns))),
                    index=np.arange(length),
                    columns=columns,
                ),
                DataFrame(np.ones((l2, len(columns))), index=[0] * l2, columns=columns),
            ]
        )

        # 断言索引不唯一
        assert df.index.is_unique is False

        # 创建索引数组 mask
        mask = np.arange(l2)
        # 使用 loc 根据 mask 选择数据
        result = df.loc[mask]
        # 期望结果为将 df.take([0])、全为1的 DataFrame 和 df.take(mask[1:]) 合并
        expected = pd.concat(
            [
                df.take([0]),
                DataFrame(
                    np.ones((len(mask), len(columns))),
                    index=[0] * len(mask),
                    columns=columns,
                ),
                df.take(mask[1:]),
            ]
        )
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(result, expected)

    def test_loc_name(self):
        # GH 3880
        # 设置 DataFrame 的索引名为 "index_name"
        df = DataFrame([[1, 1], [1, 1]])
        df.index.name = "index_name"
        # 使用 iloc 选择索引为 [0, 1] 的数据，检查索引名
        result = df.iloc[[0, 1]].index.name
        # 断言结果为 "index_name"
        assert result == "index_name"

        # 使用 loc 选择索引为 [0, 1] 的数据，检查索引名
        result = df.loc[[0, 1]].index.name
        # 断言结果为 "index_name"
        assert result == "index_name"

    def test_loc_empty_list_indexer_is_ok(self):
        # 创建一个具有自定义索引和列的 DataFrame
        df = DataFrame(
            np.ones((5, 2)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(2)], name="a"),
        )
        # 使用 loc 选择所有行但不选择列，预期与使用 iloc 选择的结果相同
        tm.assert_frame_equal(
            df.loc[:, []], df.iloc[:, :0], check_index_type=True, check_column_type=True
        )
        # 使用 loc 选择所有列但不选择行，预期与使用 iloc 选择的结果相同
        tm.assert_frame_equal(
            df.loc[[], :], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )
        # 使用 loc 选择所有行但不选择列，预期与使用 iloc 选择的结果相同
        tm.assert_frame_equal(
            df.loc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )
    def test_identity_slice_returns_new_object(self):
        # GH13873
        # 创建原始的 DataFrame 对象，包含一个列 'a'，值为 [1, 2, 3]
        original_df = DataFrame({"a": [1, 2, 3]})
        # 使用 .loc[:] 对原始 DataFrame 进行切片，得到一个新的 DataFrame 对象 sliced_df
        sliced_df = original_df.loc[:]
        # 断言切片后的 DataFrame 与原始 DataFrame 不是同一个对象
        assert sliced_df is not original_df
        # 断言对原始 DataFrame 使用 [:] 切片仍然不是同一个对象
        assert original_df[:] is not original_df
        # 断言对原始 DataFrame 使用 .loc[:, :] 切片同样不是同一个对象
        assert original_df.loc[:, :] is not original_df

        # 应该是浅拷贝
        # 断言原始 DataFrame 的列 'a' 的值和切片后的 DataFrame 的列 'a' 的值共享内存
        assert np.shares_memory(original_df["a"]._values, sliced_df["a"]._values)

        # 使用 .loc[:, "a"] 设置原始 DataFrame 中的数据，会改变原始和切片两者的值，取决于是否是 CoW
        original_df.loc[:, "a"] = [4, 4, 4]
        # 断言切片后的 DataFrame 的列 'a' 的值没有被修改
        assert (sliced_df["a"] == [1, 2, 3]).all()

        # 这些操作不应返回拷贝
        # 创建一个随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 断言获取列 0 的数据不是 .loc[:, 0] 的拷贝
        assert df[0] is not df.loc[:, 0]

        # 同样的测试适用于 Series
        # 创建原始的 Series 对象
        original_series = Series([1, 2, 3, 4, 5, 6])
        # 使用 .loc[:] 对原始 Series 进行切片，得到一个新的 Series 对象 sliced_series
        sliced_series = original_series.loc[:]
        # 断言切片后的 Series 与原始 Series 不是同一个对象
        assert sliced_series is not original_series
        # 断言对原始 Series 使用 [:] 切片仍然不是同一个对象
        assert original_series[:] is not original_series

        # 使用切片对原始 Series 进行赋值
        original_series[:3] = [7, 8, 9]
        # 断言切片后的 Series 的前三个元素没有被修改
        assert all(sliced_series[:3] == [1, 2, 3])


    def test_loc_copy_vs_view(self, request):
        # GH 15631
        # 创建一个 DataFrame 对象 x，包含两列 'a' 和 'b'，数据为 [[0, 1, 2], [0, 1, 2]]
        x = DataFrame(zip(range(3), range(3)), columns=["a", "b"])

        # 创建 x 的一个深拷贝 y
        y = x.copy()
        # 使用 .loc[:, "a"] 获取列 'a' 的视图 q，并对其值加上 2
        q = y.loc[:, "a"]
        q += 2

        # 断言修改后的 y 与原始 x 的内容相同
        tm.assert_frame_equal(x, y)

        # 创建 x 的另一个深拷贝 z
        z = x.copy()
        # 使用 .loc[x.index, "a"] 获取 'a' 列的视图 q，并对其值加上 2
        q = z.loc[x.index, "a"]
        q += 2

        # 断言修改后的 z 与原始 x 的内容相同
        tm.assert_frame_equal(x, z)


    def test_loc_uint64(self):
        # GH20722
        # 测试 loc 是否接受 uint64 的最大值作为索引
        umax = np.iinfo("uint64").max
        # 创建一个 Series 对象 ser，包含两个元素 [1, 2]，索引分别为 umax-1 和 umax
        ser = Series([1, 2], index=[umax - 1, umax])

        # 使用 loc[umax-1] 获取 ser 中 umax-1 对应的值
        result = ser.loc[umax - 1]
        expected = ser.iloc[0]
        # 断言 loc 返回的结果与 iloc 返回的结果相同
        assert result == expected

        # 使用 loc[[umax-1]] 获取 ser 中 umax-1 对应的子 Series
        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        # 断言 loc 返回的子 Series 与 iloc 返回的子 Series 相同
        tm.assert_series_equal(result, expected)

        # 使用 loc[[umax-1, umax]] 获取整个 Series
        result = ser.loc[[umax - 1, umax]]
        # 断言 loc 返回的 Series 与原始 ser 相同
        tm.assert_series_equal(result, ser)


    def test_loc_uint64_disallow_negative(self):
        # GH#41775
        # 测试 loc 是否不接受负数作为索引
        umax = np.iinfo("uint64").max
        # 创建一个 Series 对象 ser，包含两个元素 [1, 2]，索引分别为 umax-1 和 umax
        ser = Series([1, 2], index=[umax - 1, umax])

        # 使用 pytest 检查 loc[-1] 是否会抛出 KeyError 异常
        with pytest.raises(KeyError, match="-1"):
            # 不允许使用负数索引
            ser.loc[-1]

        # 使用 pytest 检查 loc[[-1]] 是否会抛出 KeyError 异常
        with pytest.raises(KeyError, match="-1"):
            # 不允许使用负数索引
            ser.loc[[-1]]


    def test_loc_setitem_empty_append_expands_rows(self):
        # GH6173, 向空 DataFrame 进行多次追加操作

        data = [1, 2, 3]
        # 创建一个预期结果的 DataFrame
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )

        # 创建一个空的 DataFrame df，列名为 ['x', 'y']
        df = DataFrame(columns=["x", "y"])
        # 使用 .loc[:, "x"] 对 df 进行赋值，追加数据
        df.loc[:, "x"] = data
        # 断言 df 和预期的结果 expected 相等
        tm.assert_frame_equal(df, expected)
    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self):
        # 定义一个测试方法，测试在混合数据类型的情况下，使用 loc 设置空数据框的扩展行操作
        # GH#37932 与 test_loc_setitem_empty_append_expands_rows 相同，但因为存在混合数据类型，所以需要经过 take_split_path
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )

        # 创建一个空的数据框 df，包含列 "x" 和 "y"
        df = DataFrame(columns=["x", "y"])
        # 将 "x" 列的数据类型转换为 np.int64
        df["x"] = df["x"].astype(np.int64)
        # 使用 loc 方法设置所有行的 "x" 列数据为 data
        df.loc[:, "x"] = data
        # 断言数据框 df 与预期结果 expected 相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self):
        # 定义一个测试方法，测试向空数据框中追加单个值的操作
        # 只追加一个值
        expected = DataFrame({"x": [1.0], "y": [np.nan]})
        # 创建一个空的数据框 df，包含列 "x" 和 "y"，数据类型为 float
        df = DataFrame(columns=["x", "y"], dtype=float)
        # 使用 loc 方法将索引为 0 的行的 "x" 列设置为预期结果中的值
        df.loc[0, "x"] = expected.loc[0, "x"]
        # 断言数据框 df 与预期结果 expected 相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self):
        # 定义一个测试方法，测试向空数据框中进行多种追加操作时可能引发的异常
        # GH6173，向空数据框进行多种追加操作

        data = [1, 2]
        # 创建一个空的数据框 df，包含列 "x" 和 "y"
        df = DataFrame(columns=["x", "y"])
        # 将索引类型转换为 np.int64
        df.index = df.index.astype(np.int64)
        # 预期会抛出 KeyError 异常，匹配特定的错误消息
        msg = r"None of .*Index.* are in the \[index\]"
        with pytest.raises(KeyError, match=msg):
            # 使用 loc 方法向索引为 [0, 1] 的行的 "x" 列设置数据为 data
            df.loc[[0, 1], "x"] = data

        # 预期会抛出 ValueError 异常，匹配特定的错误消息
        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            # 使用 loc 方法向索引为 0 到 2 的行的 "x" 列设置数据为 data
            df.loc[0:2, "x"] = data

    def test_indexing_zerodim_np_array(self):
        # 定义一个测试方法，测试使用零维 np 数组进行索引操作的情况
        # GH24924
        # 创建一个数据框 df
        df = DataFrame([[1, 2], [3, 4]])
        # 使用 loc 方法，通过零维 np 数组索引获取结果
        result = df.loc[np.array(0)]
        # 创建一个序列 s，用于与结果进行比较
        s = Series([1, 2], name=0)
        # 断言结果与序列 s 相等
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self):
        # 定义一个测试方法，测试使用零维 np 数组进行序列索引操作的情况
        # GH24924
        # 创建一个序列 s
        s = Series([1, 2])
        # 使用 loc 方法，通过零维 np 数组索引获取结果
        result = s.loc[np.array(0)]
        # 断言结果为 1
        assert result == 1

    def test_loc_reverse_assignment(self):
        # 定义一个测试方法，测试使用 loc 方法进行逆向赋值操作的情况
        # GH26939
        # 创建一个数据序列 data
        data = [1, 2, 3, 4, 5, 6] + [None] * 4
        # 创建一个预期结果序列 expected，指定索引为 2010 到 2019
        expected = Series(data, index=range(2010, 2020))

        # 创建一个结果序列 result，数据类型为 np.float64，指定索引为 2010 到 2019
        result = Series(index=range(2010, 2020), dtype=np.float64)
        # 使用 loc 方法，逆向设置索引为 2015 到 2010 的数据为 [6, 5, 4, 3, 2, 1]
        result.loc[2015:2010:-1] = [6, 5, 4, 3, 2, 1]

        # 断言结果序列 result 与预期结果序列 expected 相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set int into string")
    # 定义一个测试函数，用于测试 loc 操作设置字符串转换为小浮点数的类型转换
    def test_loc_setitem_str_to_small_float_conversion_type(self):
        # GH#20388

        # 创建包含随机小浮点数字符串的列数据
        col_data = [str(np.random.default_rng(2).random() * 1e-12) for _ in range(5)]
        # 创建结果 DataFrame，指定列名为 "A"
        result = DataFrame(col_data, columns=["A"])
        # 创建期望的 DataFrame，指定列名为 "A"，数据类型为 object
        expected = DataFrame(col_data, columns=["A"], dtype=object)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 loc/iloc 进行赋值操作，尝试在原地设置值，本例中成功
        result.loc[result.index, "A"] = [float(x) for x in col_data]
        # 更新期望的 DataFrame，数据类型为 float 类型，并转换为 object 类型
        expected = DataFrame(col_data, columns=["A"], dtype=float).astype(object)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 __setitem__ 赋值整列，将新数组交换进来
        # GH#???
        result["A"] = [float(x) for x in col_data]
        # 更新期望的 DataFrame，数据类型为 float 类型
        expected = DataFrame(col_data, columns=["A"], dtype=float)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，用于测试 loc 操作获取时间对象
    def test_loc_getitem_time_object(self, frame_or_series):
        # 创建日期范围，频率为每 5 分钟
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        # 创建掩码，筛选出小时为 9，分钟为 30 的时间点
        mask = (rng.hour == 9) & (rng.minute == 30)

        # 创建随机标准正态分布数据的 DataFrame，索引为 rng
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )
        # 获取特定类型的对象，可能是 DataFrame 或 Series
        obj = tm.get_obj(obj, frame_or_series)

        # 使用 loc 获取时间为 9:30 的数据
        result = obj.loc[time(9, 30)]
        # 使用掩码 loc 获取符合条件的数据
        exp = obj.loc[mask]
        # 断言两个对象是否相等
        tm.assert_equal(result, exp)

        # 使用 loc 获取 "1/4/2000" 之后的数据块
        chunk = obj.loc["1/4/2000":]
        # 再次使用 loc 获取时间为 9:30 的数据
        result = chunk.loc[time(9, 30)]
        # 期望结果为最后一行的数据
        expected = result[-1:]

        # 不重置频率的情况下，这些索引分别是 5 分钟和 1440 分钟
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        # 断言两个对象是否相等
        tm.assert_equal(result, expected)

    # 使用 pytest 的参数化标记定义测试函数，测试 loc 操作从稀疏矩阵获取范围
    @pytest.mark.parametrize("spmatrix_t", ["coo_matrix", "csc_matrix", "csr_matrix"])
    @pytest.mark.parametrize("dtype", [np.complex128, np.float64, np.int64, bool])
    def test_loc_getitem_range_from_spmatrix(self, spmatrix_t, dtype):
        # 导入 scipy.sparse，如果不存在则跳过测试
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 根据 spmatrix_t 字符串获取对应的稀疏矩阵类型
        spmatrix_t = getattr(sp_sparse, spmatrix_t)

        # 创建一个稀疏矩阵，生成一个大小为 (5, 7) 的方阵，对角线元素为 1，其余为 0
        rows, cols = 5, 7
        spmatrix = spmatrix_t(np.eye(rows, cols, dtype=dtype), dtype=dtype)
        # 从稀疏矩阵创建 DataFrame
        df = DataFrame.sparse.from_spmatrix(spmatrix)

        # 对 GH#34526 的回归测试
        # 创建索引范围为 itr_idx 的 DataFrame 子集，将 NaN 替换为 0
        itr_idx = range(2, rows)
        result = np.nan_to_num(df.loc[itr_idx].values)
        # 从稀疏矩阵中获取预期的数组
        expected = spmatrix.toarray()[itr_idx]
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 对 GH#34540 的回归测试
        # 获取 itr_idx 范围内的 DataFrame 子集的数据类型
        result = df.loc[itr_idx].dtypes.values
        # 创建一个填充值为 SparseDtype(dtype) 的预期数组
        expected = np.full(cols, SparseDtype(dtype))
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)
    def test_loc_getitem_listlike_all_retains_sparse(self):
        # 创建一个包含稀疏类型数据的 DataFrame
        df = DataFrame({"A": pd.array([0, 0], dtype=SparseDtype("int64"))})
        # 使用 loc 方法获取指定索引的结果
        result = df.loc[[0, 1]]
        # 断言获取的结果与原始 DataFrame 相同
        tm.assert_frame_equal(result, df)

    def test_loc_getitem_sparse_frame(self):
        # GH34687
        # 导入 scipy.sparse 模块，如果导入失败则跳过测试
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 从稀疏矩阵创建 DataFrame
        df = DataFrame.sparse.from_spmatrix(sp_sparse.eye(5, dtype=np.int64))
        # 使用 loc 方法获取指定范围的结果
        result = df.loc[range(2)]
        # 期望的 DataFrame 结果
        expected = DataFrame(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=SparseDtype(np.int64),
        )
        tm.assert_frame_equal(result, expected)

        # 连续使用 loc 方法获取结果的子集
        result = df.loc[range(2)].loc[range(1)]
        expected = DataFrame([[1, 0, 0, 0, 0]], dtype=SparseDtype(np.int64))
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sparse_series(self):
        # GH34687
        # 创建一个包含稀疏类型数据的 Series
        s = Series([1.0, 0.0, 0.0, 0.0, 0.0], dtype=SparseDtype("float64", 0.0))

        # 使用 loc 方法获取指定范围的结果
        result = s.loc[range(2)]
        # 期望的 Series 结果
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)

        # 连续使用 loc 方法获取结果的子集
        result = s.loc[range(3)].loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("indexer", ["loc", "iloc"])
    def test_getitem_single_row_sparse_df(self, indexer):
        # GH#46406
        # 创建一个包含稀疏类型数据的 DataFrame
        df = DataFrame([[1.0, 0.0, 1.5], [0.0, 2.0, 0.0]], dtype=SparseDtype(float))
        # 使用 getattr 动态获取 loc 或 iloc 方法，并获取指定索引的结果
        result = getattr(df, indexer)[0]
        # 期望的 Series 结果
        expected = Series([1.0, 0.0, 1.5], dtype=SparseDtype(float), name=0)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("key_type", [iter, np.array, Series, Index])
    def test_loc_getitem_iterable(self, float_frame, key_type):
        # 使用不同类型的索引键创建索引对象
        idx = key_type(["A", "B", "C"])
        # 使用 loc 方法获取 DataFrame 的列子集
        result = float_frame.loc[:, idx]
        # 期望的 DataFrame 结果
        expected = float_frame.loc[:, ["A", "B", "C"]]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_timedelta_0seconds(self):
        # GH#10583
        # 创建一个随机数填充的 DataFrame
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 4)))
        # 使用时间间隔索引范围创建 DataFrame 的索引
        df.index = timedelta_range(start="0s", periods=10, freq="s")
        # 使用 loc 方法获取从指定时间间隔开始的结果
        expected = df.loc[Timedelta("0s") :, :]
        result = df.loc["0s":, :]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("val,expected", [(2**63 - 1, 1), (2**63, 2)])
    def test_loc_getitem_uint64_scalar(self, val, expected):
        # see GH#19399
        # 创建一个包含稀疏类型数据的 Series
        df = DataFrame([1, 2], index=[2**63 - 1, 2**63])
        # 使用 loc 方法获取指定标量索引的结果
        result = df.loc[val]
        # 期望的 Series 结果
        expected = Series([expected])
        expected.name = val
        tm.assert_series_equal(result, expected)
    def test_loc_setitem_int_label_with_float_index(self, float_numpy_dtype):
        # 标注：标签是浮点数
        dtype = float_numpy_dtype
        # 创建一个 Series，包含字符串 "a", "b", "c"，使用浮点数索引
        ser = Series(["a", "b", "c"], index=Index([0, 0.5, 1], dtype=dtype))
        # 创建一个预期的 Series 副本
        expected = ser.copy()

        # 使用 loc 方法将值 "zoo" 赋给索引为 1 的位置
        ser.loc[1] = "zoo"
        # 在预期中，使用 iloc 方法将值 "zoo" 赋给索引为 2 的位置
        expected.iloc[2] = "zoo"

        # 断言 ser 和 expected 是否相等
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "indexer, expected",
        [
            # 在 index 为 0 的情况下，这个测试名称有误，因为 df.index[indexer] 是一个标量。
            (0, [20, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]),
            ([3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9]),
        ],
    )
    def test_loc_setitem_listlike_with_timedelta64index(self, indexer, expected):
        # GH#16637
        # 创建一个时间增量索引对象 tdi，单位为秒
        tdi = to_timedelta(range(10), unit="s")
        # 创建一个 DataFrame，包含一列名为 "x"，值为 0 到 9，索引为 tdi
        df = DataFrame({"x": range(10)}, dtype="int64", index=tdi)

        # 使用 loc 方法将 df.index[indexer] 对应位置的 "x" 列的值设置为 20
        df.loc[df.index[indexer], "x"] = 20

        # 创建预期的 DataFrame
        expected = DataFrame(
            expected,
            index=tdi,
            columns=["x"],
            dtype="int64",
        )

        # 断言预期结果和实际结果是否相等
        tm.assert_frame_equal(expected, df)

    def test_loc_setitem_categorical_values_partial_column_slice(self):
        # 将分类值分配给整数/... 列的部分使用分类的值
        # 创建一个 DataFrame，包含两列 "a" 和 "b"
        df = DataFrame({"a": [1, 1, 1, 1, 1], "b": list("aaaaa")})
        # 创建预期的 DataFrame
        exp = DataFrame({"a": [1, "b", "b", 1, 1], "b": list("aabba")})
        # 检查是否会产生 FutureWarning，匹配 "item of incompatible dtype"
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            # 使用 loc 方法将分类变量 ["b", "b"] 分配给 df 中索引为 1 到 2 的 "a" 列
            df.loc[1:2, "a"] = Categorical(["b", "b"], categories=["a", "b"])
            # 使用 loc 方法将分类变量 ["b", "b"] 分配给 df 中索引为 2 到 3 的 "b" 列
            df.loc[2:3, "b"] = Categorical(["b", "b"], categories=["a", "b"])
        # 断言 df 和 exp 是否相等
        tm.assert_frame_equal(df, exp)

    def test_loc_setitem_single_row_categorical(self, using_infer_string):
        # GH#25495
        # 创建一个 DataFrame，包含两列 "Alpha" 和 "Numeric"
        df = DataFrame({"Alpha": ["a"], "Numeric": [0]})
        # 创建一个分类对象，使用 df["Alpha"] 的值，指定类别为 ["a", "b", "c"]
        categories = Categorical(df["Alpha"], categories=["a", "b", "c"])

        # 在 2.0 之前，这将交换一个新数组，2.0 中它是原地操作，与非分割路径一致
        # 使用 loc 方法将 categories 分配给 df 中的 "Alpha" 列
        df.loc[:, "Alpha"] = categories

        # 获取结果 Series
        result = df["Alpha"]
        # 创建预期的 Series
        expected = Series(categories, index=df.index, name="Alpha").astype(
            object if not using_infer_string else "string[pyarrow_numpy]"
        )
        # 断言结果和预期是否相等
        tm.assert_series_equal(result, expected)

        # 再次检查非 loc 设置是否保留分类性
        # 使用 categories 分配给 df 中的 "Alpha" 列
        df["Alpha"] = categories
        # 断言 df["Alpha"] 和 Series(categories, name="Alpha") 是否相等
        tm.assert_series_equal(df["Alpha"], Series(categories, name="Alpha"))
    # 测试函数，用于验证在设置数据框 DataFrame 的 loc 中的日期时间类型键时的类型转换问题
    def test_loc_setitem_datetime_coercion(self):
        # 创建包含一个日期时间戳的数据框 DataFrame
        df = DataFrame({"c": [Timestamp("2010-10-01")] * 3})
        # 将索引为 0 和 1 的 "c" 列设置为 np.datetime64 类型的日期时间
        df.loc[0:1, "c"] = np.datetime64("2008-08-08")
        # 断言索引为 0 和 1 的 "c" 列确实被设置为预期的时间戳
        assert Timestamp("2008-08-08") == df.loc[0, "c"]
        assert Timestamp("2008-08-08") == df.loc[1, "c"]
        # 使用 assert_produces_warning 确保设置索引为 2 的 "c" 列为 date 类型会产生 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[2, "c"] = date(2005, 5, 5)
        # 断言索引为 2 的 "c" 列确实被设置为预期的日期
        assert Timestamp("2005-05-05").date() == df.loc[2, "c"]

    @pytest.mark.parametrize("idxer", ["var", ["var"]])
    # 测试函数，用于验证在设置包含时区的 DatetimeIndex 时的 dtype 问题
    def test_loc_setitem_datetimeindex_tz(self, idxer, tz_naive_fixture):
        # 获取时区信息
        tz = tz_naive_fixture
        # 创建一个包含时区信息的 DatetimeIndex
        idx = date_range(start="2015-07-12", periods=3, freq="h", tz=tz)
        # 创建一个预期的数据框，其中索引为 idx，列名为 "var"，数据为 1.2
        expected = DataFrame(1.2, index=idx, columns=["var"])
        # 创建一个初始 dtype 为 np.float64 的数据框 result
        result = DataFrame(index=idx, columns=["var"], dtype=np.float64)
        # 使用 assert_produces_warning 确保设置 idxer 为预期值时会产生 FutureWarning 警告
        with tm.assert_produces_warning(
            FutureWarning if idxer == "var" else None, match="incompatible dtype"
        ):
            # 设置 result 中的列 idxer 为 expected
            result.loc[:, idxer] = expected
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，用于验证在设置时间键时的行为
    def test_loc_setitem_time_key(self):
        # 创建一个包含时间间隔的 DatetimeIndex
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        # 创建一个随机数据框 df，数据为标准正态分布
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        # 创建两个时间键 akey 和 bkey
        akey = time(12, 0, 0)
        bkey = slice(time(13, 0, 0), time(14, 0, 0))
        # 创建两组索引值 ainds 和 binds
        ainds = [24, 72, 120, 168]
        binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]

        # 复制 df 到 result
        result = df.copy()
        # 将 result 中时间键为 akey 的行设置为 0
        result.loc[akey] = 0
        # 将 result 设置为只包含 akey 时间键的行
        result = result.loc[akey]
        # 复制 df 中时间键为 akey 的行到 expected
        expected = df.loc[akey].copy()
        # 将 expected 所有值设置为 0
        expected.loc[:] = 0
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(result, expected)

        # 复制 df 到 result
        result = df.copy()
        # 将 result 中时间键为 akey 的行设置为 0
        result.loc[akey] = 0
        # 将 result 中时间键为 akey 的行设置为 df 中 ainds 索引对应的行
        result.loc[akey] = df.iloc[ainds]
        # 断言 result 和 df 相等
        tm.assert_frame_equal(result, df)

        # 复制 df 到 result
        result = df.copy()
        # 将 result 中时间键为 bkey 的行设置为 0
        result.loc[bkey] = 0
        # 将 result 设置为只包含 bkey 时间键的行
        result = result.loc[bkey]
        # 复制 df 中时间键为 bkey 的行到 expected
        expected = df.loc[bkey].copy()
        # 将 expected 所有值设置为 0
        expected.loc[:] = 0
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(result, expected)

        # 复制 df 到 result
        result = df.copy()
        # 将 result 中时间键为 bkey 的行设置为 0
        result.loc[bkey] = 0
        # 将 result 中时间键为 bkey 的行设置为 df 中 binds 索引对应的行
        result.loc[bkey] = df.iloc[binds]
        # 断言 result 和 df 相等
        tm.assert_frame_equal(result, df)
    def test_loc_setitem_unsorted_multiindex_columns(self, key):
        # GH#38601
        # 创建一个多级索引对象 mi，包含元组 ("A", 4), ("B", "3"), ("A", "2")
        mi = MultiIndex.from_tuples([("A", 4), ("B", "3"), ("A", "2")])
        # 创建一个 DataFrame 对象 df，包含两行三列数据，并使用 mi 作为列索引
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        # 复制 df 并赋值给 obj
        obj = df.copy()
        # 使用 loc 方法将 obj 的所有行、列为 key 的位置赋值为 int64 类型的零数组
        obj.loc[:, key] = np.zeros((2, 2), dtype="int64")
        # 创建一个期望的 DataFrame 对象 expected
        expected = DataFrame([[0, 2, 0], [0, 5, 0]], columns=mi)
        # 检查 obj 和 expected 是否相等
        tm.assert_frame_equal(obj, expected)

        # 对 df 按照列索引排序
        df = df.sort_index(axis=1)
        # 使用 loc 方法将 df 的所有行、列为 key 的位置赋值为 int64 类型的零数组
        df.loc[:, key] = np.zeros((2, 2), dtype="int64")
        # 对期望的 DataFrame 对象 expected 按照列索引排序
        expected = expected.sort_index(axis=1)
        # 检查 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_uint_drop(self, any_int_numpy_dtype):
        # see GH#18311
        # 创建一个 Series 对象 series，包含整数数据，并指定数据类型为 any_int_numpy_dtype
        series = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        # 使用 loc 方法将 series 的索引为 0 的位置赋值为 4
        series.loc[0] = 4
        # 创建一个期望的 Series 对象 expected
        expected = Series([4, 2, 3], dtype=any_int_numpy_dtype)
        # 检查 series 和 expected 是否相等
        tm.assert_series_equal(series, expected)

    def test_loc_setitem_td64_non_nano(self):
        # GH#14155
        # 创建一个 Series 对象 ser，包含 10 个 np.timedelta64(10, "m") 元素
        ser = Series(10 * [np.timedelta64(10, "m")])
        # 使用 loc 方法将 ser 的索引为 [1, 2, 3] 的位置赋值为 np.timedelta64(20, "m")
        ser.loc[[1, 2, 3]] = np.timedelta64(20, "m")
        # 创建一个期望的 Series 对象 expected
        expected = Series(10 * [np.timedelta64(10, "m")])
        # 使用 loc 方法将 expected 的索引为 [1, 2, 3] 的位置赋值为 Timedelta(np.timedelta64(20, "m"))
        expected.loc[[1, 2, 3]] = Timedelta(np.timedelta64(20, "m"))
        # 检查 ser 和 expected 是否相等
        tm.assert_series_equal(ser, expected)

    def test_loc_setitem_2d_to_1d_raises(self):
        # 创建一个随机数据数组 data
        data = np.random.default_rng(2).standard_normal((2, 2))
        # 创建一个浮点数类型的 Series 对象 ser，包含范围为 [0, 1] 的元素
        ser = Series(range(2), dtype="float64")

        # 准备捕获的异常消息
        msg = "setting an array element with a sequence."
        # 使用 pytest 的 assert_raises 方法捕获 ValueError 异常，并检查是否包含指定消息 msg
        with pytest.raises(ValueError, match=msg):
            # 使用 loc 方法将 ser 的索引为 range(2) 的位置赋值为 data
            ser.loc[range(2)] = data

        with pytest.raises(ValueError, match=msg):
            # 使用 loc 方法将 ser 的所有位置赋值为 data
            ser.loc[:] = data

    def test_loc_getitem_interval_index(self):
        # GH#19977
        # 创建一个区间索引对象 index，起始于 0，包含 3 个区间
        index = pd.interval_range(start=0, periods=3)
        # 创建一个 DataFrame 对象 df，包含三行三列数据，并使用 index 作为行索引，列名为 ["A", "B", "C"]
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"])

        # 准备期望的结果 expected
        expected = 1
        # 使用 loc 方法获取 df 中索引为 0.5，列为 "A" 的元素
        result = df.loc[0.5, "A"]
        # 检查 result 和 expected 是否近似相等
        tm.assert_almost_equal(result, expected)

    def test_loc_getitem_interval_index2(self):
        # GH#19977
        # 创建一个区间索引对象 index，起始于 0，包含 3 个区间，闭合方式为 "both"
        index = pd.interval_range(start=0, periods=3, closed="both")
        # 创建一个 DataFrame 对象 df，包含三行三列数据，并使用 index 作为行索引，列名为 ["A", "B", "C"]
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"])

        # 创建一个区间索引对象 index_exp，起始于 0，包含 2 个区间，频率为 1，闭合方式为 "both"
        index_exp = pd.interval_range(start=0, periods=2, freq=1, closed="both")
        # 准备期望的 Series 对象 expected
        expected = Series([1, 4], index=index_exp, name="A")
        # 使用 loc 方法获取 df 中索引为 1，列为 "A" 的元素
        result = df.loc[1, "A"]
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tpl", [(1,), (1, 2)])
    # 测试使用单个双元组和双元组列表来获取索引位置的情况
    def test_loc_getitem_index_single_double_tuples(self, tpl):
        # 创建包含元组 (1,) 和 (1, 2) 的索引对象
        idx = Index(
            [(1,), (1, 2)],
            name="A",
            tupleize_cols=False,
        )
        # 使用创建的索引对象作为 DataFrame 的索引
        df = DataFrame(index=idx)

        # 通过 loc 方法使用元组 tpl 获取数据
        result = df.loc[[tpl]]
        # 创建包含单个元组 tpl 的索引对象
        idx = Index([tpl], name="A", tupleize_cols=False)
        # 期望的结果 DataFrame
        expected = DataFrame(index=idx)
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试使用命名元组作为索引的情况
    def test_loc_getitem_index_namedtuple(self):
        # 定义命名元组类型 IndexType
        IndexType = namedtuple("IndexType", ["a", "b"])
        # 创建两个命名元组对象 idx1 和 idx2
        idx1 = IndexType("foo", "bar")
        idx2 = IndexType("baz", "bof")
        # 创建包含 idx1 和 idx2 的索引对象
        index = Index([idx1, idx2], name="composite_index", tupleize_cols=False)
        # 创建包含元组数据的 DataFrame，设置列名为 ["A", "B"]
        df = DataFrame([(1, 2), (3, 4)], index=index, columns=["A", "B"])

        # 使用命名元组 IndexType("foo", "bar") 获取列 "A" 的数据
        result = df.loc[IndexType("foo", "bar")]["A"]
        # 断言 result 是否等于 1
        assert result == 1

    # 测试使用混合类型单列设置的情况
    def test_loc_setitem_single_column_mixed(self, using_infer_string):
        # 创建一个随机数填充的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )
        # 向 DataFrame 添加一个名为 "str" 的新列，所有行的值为 "qux"
        df["str"] = "qux"
        # 使用 loc 方法将 df.index 中每隔一个元素对应行的 "str" 列值设为 np.nan
        df.loc[df.index[::2], "str"] = np.nan
        # 期望的结果是一个 Series，其中每隔一个元素为 np.nan，其余为 "qux"
        expected = Series(
            [np.nan, "qux", np.nan, "qux", np.nan],
            dtype=object if not using_infer_string else "string[pyarrow_numpy]",
        ).values
        # 断言 df["str"] 的值与 expected 是否几乎相等
        tm.assert_almost_equal(df["str"].values, expected)

    # 测试在设置时进行数据类型转换的情况
    def test_loc_setitem_cast2(self):
        # 创建一个随机数填充的 DataFrame，列名为 "A", "B", "C"，其中 "event" 列初始化为 np.nan
        df = DataFrame(np.random.default_rng(2).random((30, 3)), columns=tuple("ABC"))
        df["event"] = np.nan
        # 使用 loc 方法将索引为 10 的行的 "event" 列值设为字符串 "foo"
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df.loc[10, "event"] = "foo"
        # 获取 DataFrame 的列数据类型
        result = df.dtypes
        # 期望的结果是一个 Series，其中前三列为 float64，最后一列 "event" 为 object 类型
        expected = Series(
            [np.dtype("float64")] * 3 + [np.dtype("object")],
            index=["A", "B", "C", "event"],
        )
        # 断言 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 测试数据类型保持不变的情况
    def test_loc_setitem_cast3(self):
        # 创建一个包含整数类型的 DataFrame 列 "one"
        df = DataFrame({"one": np.arange(6, dtype=np.int8)})
        # 使用 loc 方法将索引为 1 的行的 "one" 列值设为 6
        df.loc[1, "one"] = 6
        # 断言 df 的 "one" 列数据类型仍然是 np.int8
        assert df.dtypes.one == np.dtype(np.int8)
        # 直接设置 df 的 "one" 列为 np.int8 类型的 7
        df.one = np.int8(7)
        # 断言 df 的 "one" 列数据类型仍然是 np.int8
        assert df.dtypes.one == np.dtype(np.int8)

    # 测试不将范围键视为位置参数的情况
    def test_loc_setitem_range_key(self, frame_or_series):
        # 根据参数 frame_or_series 创建对象 obj，索引为 [3, 4, 1, 0, 2]
        obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])

        # 根据 obj 的维度确定 values 的形状
        values = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]

        # 使用 loc 方法将范围键 range(3) 对应的行设置为 values
        obj.loc[range(3)] = values

        # 期望的结果是一个与 obj 索引相同的对象，其值为 [0, 1, 10, 9, 11]
        expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        # 断言 obj 与 expected 是否相等
        tm.assert_equal(obj, expected)
    # 定义一个测试函数 test_loc_setitem_numpy_frame_categorical_value，用于测试特定功能
    def test_loc_setitem_numpy_frame_categorical_value():
        # GH#52927: GitHub issue编号，指示此测试用例的背景或相关问题
        # 创建一个包含两列的DataFrame，其中列"a"包含五个值为1的整数，列"b"包含五个值为"a"的字符串
        df = DataFrame({"a": [1, 1, 1, 1, 1], "b": ["a", "a", "a", "a", "a"]})
        
        # 使用loc方法将DataFrame的索引1到2（包括）处的"a"列值设为Categorical类型的对象
        # Categorical对象包含值为2和2的类别数据，类别为1和2
        df.loc[1:2, "a"] = Categorical([2, 2], categories=[1, 2])
        
        # 创建一个预期结果的DataFrame，其中第一列"a"的第二和第三行的值被设置为2
        expected = DataFrame({"a": [1, 2, 2, 1, 1], "b": ["a", "a", "a", "a", "a"]})
        
        # 使用assert_frame_equal函数比较实际DataFrame df和预期结果DataFrame expected是否相等
        tm.assert_frame_equal(df, expected)
class TestLocWithEllipsis:
    @pytest.fixture
    def indexer(self, indexer_li):
        # 在这里测试 iloc
        return indexer_li

    @pytest.fixture
    def obj(self, series_with_simple_index, frame_or_series):
        obj = series_with_simple_index
        if frame_or_series is not Series:
            obj = obj.to_frame()
        return obj

    def test_loc_iloc_getitem_ellipsis(self, obj, indexer):
        # 使用省略号对对象进行索引操作，预期结果应该与原对象相同
        result = indexer(obj)[...]
        tm.assert_equal(result, obj)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_loc_iloc_getitem_leading_ellipses(self, series_with_simple_index, indexer):
        obj = series_with_simple_index
        key = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]

        if indexer is tm.loc and obj.index.inferred_type == "boolean":
            # 当使用布尔类型索引时，传递 [False] 会被解释为布尔掩码
            # TODO: 应该吗？当长度不匹配时不明确
            return
        if indexer is tm.loc and isinstance(obj.index, MultiIndex):
            msg = "MultiIndex does not support indexing with Ellipsis"
            # 多重索引不支持使用省略号进行索引，抛出未实现的错误
            with pytest.raises(NotImplementedError, match=msg):
                result = indexer(obj)[..., [key]]

        elif len(obj) != 0:
            result = indexer(obj)[..., [key]]
            expected = indexer(obj)[[key]]
            # 验证预期结果与实际结果是否相等
            tm.assert_series_equal(result, expected)

        key2 = 0 if indexer is tm.iloc else obj.name
        df = obj.to_frame()
        result = indexer(df)[..., [key2]]
        expected = indexer(df)[:, [key2]]
        # 验证预期结果与实际结果是否相等
        tm.assert_frame_equal(result, expected)

    def test_loc_iloc_getitem_ellipses_only_one_ellipsis(self, obj, indexer):
        # GH37750
        key = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ...]
            # 使用一个省略号时会引发索引错误，匹配 _one_ellipsis_message 消息

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., [key], ...]
            # 使用省略号和其他索引时会引发索引错误，匹配 _one_ellipsis_message 消息

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ..., key]
            # 使用两个省略号和其他索引时会引发索引错误，匹配 _one_ellipsis_message 消息

        # 当第一个键是省略号时，one_ellipsis_message 优先于 "Too many indexers" 消息
        with pytest.raises(IndexingError, match="Too many indexers"):
            indexer(obj)[key, ..., ...]
            # 当第一个键是省略号时，会引发 "Too many indexers" 消息


class TestLocWithMultiIndex:
    @pytest.mark.parametrize(
        "keys, expected",
        [
            (["b", "a"], [["b", "b", "a", "a"], [1, 2, 1, 2]]),
            (["a", "b"], [["a", "a", "b", "b"], [1, 2, 1, 2]]),
            ((["a", "b"], [1, 2]), [["a", "a", "b", "b"], [1, 2, 1, 2]]),
            ((["a", "b"], [2, 1]), [["a", "a", "b", "b"], [2, 1, 2, 1]]),
            ((["b", "a"], [2, 1]), [["b", "b", "a", "a"], [2, 1, 2, 1]]),
            ((["b", "a"], [1, 2]), [["b", "b", "a", "a"], [1, 2, 1, 2]]),
            ((["c", "a"], [2, 1]), [["c", "a", "a"], [1, 2, 1]]),
        ],
    )
    @pytest.mark.parametrize("dim", ["index", "columns"])
    # 使用 pytest 的参数化装饰器，定义测试函数参数 dim，分别测试 "index" 和 "columns"
    def test_loc_getitem_multilevel_index_order(self, dim, keys, expected):
        # GH#22797
        # 修复多级索引下使用 MultiIndex.loc 时键顺序的问题
        kwargs = {dim: [["c", "a", "a", "b", "b"], [1, 1, 2, 1, 2]]}
        # 创建一个 DataFrame，指定多级索引的键和数据为从 0 到 24 的数组
        df = DataFrame(np.arange(25).reshape(5, 5), **kwargs)
        # 根据预期的索引数组创建 MultiIndex 对象
        exp_index = MultiIndex.from_arrays(expected)
        if dim == "index":
            # 使用 MultiIndex.loc 获取行，与预期的索引比较
            res = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == "columns":
            # 使用 MultiIndex.loc 获取列，与预期的索引比较
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

    def test_loc_preserve_names(self, multiindex_year_month_day_dataframe_random_data):
        # 使用 MultiIndex 的名称测试 MultiIndex.loc 的名称保留
        ymd = multiindex_year_month_day_dataframe_random_data

        # 测试单个索引值时，保留索引名称
        result = ymd.loc[2000]
        result2 = ymd["A"].loc[2000]
        assert result.index.names == ymd.index.names[1:]
        assert result2.index.names == ymd.index.names[1:]

        # 测试多个索引值时，保留索引名称
        result = ymd.loc[2000, 2]
        result2 = ymd["A"].loc[2000, 2]
        assert result.index.name == ymd.index.names[2]
        assert result2.index.name == ymd.index.names[2]

    def test_loc_getitem_multiindex_nonunique_len_zero(self):
        # GH#13691
        # 测试 MultiIndex.loc 在索引长度为零且非唯一的情况下的行为
        mi = MultiIndex.from_product([[0], [1, 1]])
        ser = Series(0, index=mi)

        # 使用空列表作为索引，期望结果为空 Series
        res = ser.loc[[]]
        expected = ser[:0]
        tm.assert_series_equal(res, expected)

        # 使用 iloc 获取空切片作为索引，期望结果同样为空 Series
        res2 = ser.loc[ser.iloc[0:0]]
        tm.assert_series_equal(res2, expected)

    def test_loc_getitem_access_none_value_in_multiindex(self):
        # GH#34318: 测试通过 MultiIndex 访问 None 值的情况
        # 在 MultiIndex 中使用 .loc 访问 None 值
        ser = Series([None], MultiIndex.from_arrays([["Level1"], ["Level2"]]))
        result = ser.loc[("Level1", "Level2")]
        assert result is None

        # 在 MultiIndex 中使用 .loc 访问 None 值和非 None 值
        midx = MultiIndex.from_product([["Level1"], ["Level2_a", "Level2_b"]])

        # 测试在索引中使用 None 值，期望结果为 None
        ser = Series([None] * len(midx), dtype=object, index=midx)
        result = ser.loc[("Level1", "Level2_a")]
        assert result is None

        # 测试在索引中使用非 None 值，期望结果为 1
        ser = Series([1] * len(midx), dtype=object, index=midx)
        result = ser.loc[("Level1", "Level2_a")]
        assert result == 1

    def test_loc_setitem_multiindex_slice(self):
        # GH 34870
        # 测试 MultiIndex.loc 用于设置切片的行为
        index = MultiIndex.from_tuples(
            zip(
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
            ),
            names=["first", "second"],
        )

        # 创建带有 MultiIndex 的 Series 对象
        result = Series([1, 1, 1, 1, 1, 1, 1, 1], index=index)
        # 使用 MultiIndex.loc 设置切片的值为 100
        result.loc[("baz", "one") : ("foo", "two")] = 100

        # 期望结果为指定位置值为 100 的 Series 对象
        expected = Series([1, 1, 100, 100, 100, 100, 1, 1], index=index)

        tm.assert_series_equal(result, expected)
    def test_loc_getitem_slice_datetime_objs_with_datetimeindex(self):
        # 创建时间序列，从 '2000-01-01' 开始，每隔 10 分钟一个时间点，总共 100000 个时间点
        times = date_range("2000-01-01", freq="10min", periods=100000)
        # 创建一个序列，包含 0 到 99999 的整数，以时间序列 times 作为索引
        ser = Series(range(100000), times)
        # 使用 loc 方法，选取从 datetime(1900, 1, 1) 到 datetime(2100, 1, 1) 之间的数据
        result = ser.loc[datetime(1900, 1, 1) : datetime(2100, 1, 1)]
        # 断言选取的结果与原序列相等
        tm.assert_series_equal(result, ser)

    def test_loc_getitem_datetime_string_with_datetimeindex(self):
        # GH 16710
        # 创建一个数据框，包含两列 'a' 和 'b'，索引为从 '2010-01-01' 到 '2010-01-10' 的日期范围
        df = DataFrame(
            {"a": range(10), "b": range(10)},
            index=date_range("2010-01-01", "2010-01-10"),
        )
        # 使用 loc 方法，选取索引为 ['2010-01-01', '2010-01-05']，列为 ['a', 'b'] 的数据
        result = df.loc[["2010-01-01", "2010-01-05"], ["a", "b"]]
        # 创建预期的数据框，索引为 ['2010-01-01', '2010-01-05']，列为 ['a', 'b']
        expected = DataFrame(
            {"a": [0, 4], "b": [0, 4]},
            index=DatetimeIndex(["2010-01-01", "2010-01-05"]).as_unit("ns"),
        )
        # 断言选取的结果与预期的数据框相等
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sorted_index_level_with_duplicates(self):
        # GH#4516 对具有重复值和多种数据类型的多级索引进行排序
        # 创建一个多级索引 mi，包含重复值和两个级别名称 'A' 和 'B'
        mi = MultiIndex.from_tuples(
            [
                ("foo", "bar"),
                ("foo", "bar"),
                ("bah", "bam"),
                ("bah", "bam"),
                ("foo", "bar"),
                ("bah", "bam"),
            ],
            names=["A", "B"],
        )
        # 创建一个数据框 df，包含 6 行 2 列的数据，使用 mi 作为索引，列名为 ['C', 'D']
        df = DataFrame(
            [
                [1.0, 1],
                [2.0, 2],
                [3.0, 3],
                [4.0, 4],
                [5.0, 5],
                [6.0, 6],
            ],
            index=mi,
            columns=["C", "D"],
        )
        # 根据第一级别的索引排序数据框 df
        df = df.sort_index(level=0)

        # 创建预期的数据框，包含 ['C', 'D'] 列的数据，索引为 mi 的第 0、1、4 行
        expected = DataFrame(
            [[1.0, 1], [2.0, 2], [5.0, 5]], columns=["C", "D"], index=mi.take([0, 1, 4])
        )

        # 使用 loc 方法，选取索引为 ('foo', 'bar') 的数据
        result = df.loc[("foo", "bar")]
        # 断言选取的结果与预期的数据框相等
        tm.assert_frame_equal(result, expected)

    def test_additional_element_to_categorical_series_loc(self):
        # GH#47677
        # 创建一个分类类型的序列 result，包含字符串 'a', 'b', 'c'
        result = Series(["a", "b", "c"], dtype="category")
        # 使用 loc 方法，在索引为 3 的位置插入整数 0
        result.loc[3] = 0
        # 创建预期的分类类型序列 expected，包含字符串 'a', 'b', 'c', 0
        expected = Series(["a", "b", "c", 0], dtype="object")
        # 断言序列 result 与预期的序列 expected 相等
        tm.assert_series_equal(result, expected)

    def test_additional_categorical_element_loc(self):
        # GH#47677
        # 创建一个分类类型的序列 result，包含字符串 'a', 'b', 'c'
        result = Series(["a", "b", "c"], dtype="category")
        # 使用 loc 方法，在索引为 3 的位置插入字符串 'a'
        result.loc[3] = "a"
        # 创建预期的分类类型序列 expected，包含字符串 'a', 'b', 'c', 'a'
        expected = Series(["a", "b", "c", "a"], dtype="category")
        # 断言序列 result 与预期的序列 expected 相等
        tm.assert_series_equal(result, expected)
    def test_loc_set_nan_in_categorical_series(self, any_numeric_ea_dtype):
        # GH#47677
        # 创建一个包含数值的系列对象，使用特定的分类数据类型
        srs = Series(
            [1, 2, 3],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        
        # 将索引位置为3的值设置为NaN
        srs.loc[3] = np.nan
        # 创建期望的系列对象，包含NaN值
        expected = Series(
            [1, 2, 3, np.nan],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        # 断言两个系列对象是否相等
        tm.assert_series_equal(srs, expected)
        
        # 将索引位置为1的值设置为NaN
        srs.loc[1] = np.nan
        # 创建期望的系列对象，更新后包含NaN值
        expected = Series(
            [1, np.nan, 3, np.nan],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        # 断言两个系列对象是否相等
        tm.assert_series_equal(srs, expected)

    @pytest.mark.parametrize("na", (np.nan, pd.NA, None, pd.NaT))
    def test_loc_consistency_series_enlarge_set_into(self, na):
        # GH#47677
        # 创建一个分类类型的系列对象，并在索引位置为3处设置为na值（np.nan, pd.NA, None, pd.NaT中的一种）
        srs_enlarge = Series(["a", "b", "c"], dtype="category")
        srs_enlarge.loc[3] = na

        # 创建另一个分类类型的系列对象，扩展长度，并在索引位置为3处设置为na值
        srs_setinto = Series(["a", "b", "c", "a"], dtype="category")
        srs_setinto.loc[3] = na

        # 断言两个系列对象是否相等
        tm.assert_series_equal(srs_enlarge, srs_setinto)
        # 创建期望的系列对象，更新后包含na值
        expected = Series(["a", "b", "c", na], dtype="category")
        # 断言扩展后的系列对象是否符合预期
        tm.assert_series_equal(srs_enlarge, expected)

    def test_loc_getitem_preserves_index_level_category_dtype(self):
        # GH#15166
        # 创建一个数据框，使用多级索引，其中一个级别使用分类索引
        df = DataFrame(
            data=np.arange(2, 22, 2),
            index=MultiIndex(
                levels=[CategoricalIndex(["a", "b"]), range(10)],
                codes=[[0] * 5 + [1] * 5, range(10)],
                names=["Index1", "Index2"],
            ),
        )

        # 创建期望的分类索引对象
        expected = CategoricalIndex(
            ["a", "b"],
            categories=["a", "b"],
            ordered=False,
            name="Index1",
            dtype="category",
        )

        # 获取数据框索引的第一个级别
        result = df.index.levels[0]
        # 断言结果与期望的分类索引对象相等
        tm.assert_index_equal(result, expected)

        # 获取数据框中索引为"a"的子集，再次获取第一个级别
        result = df.loc[["a"]].index.levels[0]
        # 断言结果与期望的分类索引对象相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("lt_value", [30, 10])
    def test_loc_multiindex_levels_contain_values_not_in_index_anymore(self, lt_value):
        # GH#41170
        # 创建一个数据框，使用多级索引
        df = DataFrame({"a": [12, 23, 34, 45]}, index=[list("aabb"), [0, 1, 2, 3]])
        # 使用断言检查是否会引发预期的KeyError异常
        with pytest.raises(KeyError, match=r"\['b'\] not in index"):
            df.loc[df["a"] < lt_value, :].loc[["b"], :]

    def test_loc_multiindex_null_slice_na_level(self):
        # GH#42055
        # 创建两个数组，一个包含NaN值，一个包含字符串
        lev1 = np.array([np.nan, np.nan])
        lev2 = ["bar", "baz"]
        # 使用数组创建一个多级索引对象
        mi = MultiIndex.from_arrays([lev1, lev2])
        # 使用多级索引创建一个系列对象
        ser = Series([0, 1], index=mi)
        # 获取索引为"bar"的切片结果
        result = ser.loc[:, "bar"]

        # TODO: should we have name="bar"?
        # 创建期望的系列对象，索引为NaN
        expected = Series([0], index=[np.nan])
        # 断言结果与期望的系列对象相等
        tm.assert_series_equal(result, expected)
    def test_loc_drops_level(self):
        # 测试函数：验证 loc 方法是否正确去除指定级别
        # 基于 test_series_varied_multiindex_alignment 的测试情况，
        # 以前在删除第一个级别时会失败
        # 创建一个多级索引 mi，包含三个级别：ab、xy、num
        mi = MultiIndex.from_product(
            [list("ab"), list("xy"), [1, 2]], names=["ab", "xy", "num"]
        )
        # 创建一个 Series 对象 ser，使用 mi 作为索引，值为 range(8)
        ser = Series(range(8), index=mi)

        # 使用 loc 方法获取索引中以 "a" 开头的所有数据
        loc_result = ser.loc["a", :, :]
        # 期望的结果是将索引的第一个级别删除后的前四个条目
        expected = ser.index.droplevel(0)[:4]
        # 断言 loc_result 的索引与期望的索引相等
        tm.assert_index_equal(loc_result.index, expected)
    # 定义一个测试类 TestLocSetitemWithExpansion
    class TestLocSetitemWithExpansion:
        # 测试大型数据框架的 loc 设置项与扩展
        def test_loc_setitem_with_expansion_large_dataframe(self, monkeypatch):
            # GH#10692
            # 设置大小截断为 50
            size_cutoff = 50
            # 在 monkeypatch 上下文中执行以下操作
            with monkeypatch.context():
                # 设置 libindex 模块中的 _SIZE_CUTOFF 属性为 size_cutoff
                monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
                # 创建一个包含 "x" 列和范围为 0 到 size_cutoff 的整数的 DataFrame 对象
                result = DataFrame({"x": range(size_cutoff)}, dtype="int64")
                # 使用 loc 将一个新的值 size_cutoff 插入到 DataFrame 中
                result.loc[size_cutoff] = size_cutoff
            # 创建预期的 DataFrame，将范围扩展到 size_cutoff + 1
            expected = DataFrame({"x": range(size_cutoff + 1)}, dtype="int64")
            # 断言两个 DataFrame 对象是否相等
            tm.assert_frame_equal(result, expected)

        # 测试空 Series 对象的 loc 设置项
        def test_loc_setitem_empty_series(self):
            # GH#5226

            # 使用空的 object 类型的 Series 部分设置
            ser = Series(dtype=object)
            ser.loc[1] = 1
            # 断言 Series 对象是否与指定的 Series 对象相等
            tm.assert_series_equal(ser, Series([1], index=[1]))
            ser.loc[3] = 3
            # 再次断言 Series 对象是否与指定的 Series 对象相等
            tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

        # 测试空 Series 对象的 loc 设置项，数据类型为 float
        def test_loc_setitem_empty_series_float(self):
            # GH#5226

            # 使用空的 object 类型的 Series 部分设置
            ser = Series(dtype=object)
            ser.loc[1] = 1.0
            # 断言 Series 对象是否与指定的 Series 对象相等
            tm.assert_series_equal(ser, Series([1.0], index=[1]))
            ser.loc[3] = 3.0
            # 再次断言 Series 对象是否与指定的 Series 对象相等
            tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

        # 测试空 Series 对象的 loc 设置项，索引为字符串类型
        def test_loc_setitem_empty_series_str_idx(self):
            # GH#5226

            # 使用空的 object 类型的 Series 部分设置
            ser = Series(dtype=object)
            ser.loc["foo"] = 1
            # 断言 Series 对象是否与指定的 Series 对象相等
            tm.assert_series_equal(ser, Series([1], index=Index(["foo"], dtype=object)))
            ser.loc["bar"] = 3
            # 再次断言 Series 对象是否与指定的 Series 对象相等
            tm.assert_series_equal(
                ser, Series([1, 3], index=Index(["foo", "bar"], dtype=object))
            )
            ser.loc[3] = 4
            # 最后断言 Series 对象是否与指定的 Series 对象相等
            tm.assert_series_equal(
                ser, Series([1, 3, 4], index=Index(["foo", "bar", 3], dtype=object))
            )

        # 测试逐步增加的 Series 对象的 loc 设置项
        def test_loc_setitem_incremental_with_dst(self):
            # GH#20724
            # 创建一个基础时间为 '2015-11-01' 的 datetime 对象，时区设置为 'US/Pacific'
            base = datetime(2015, 11, 1, tzinfo=gettz("US/Pacific"))
            # 生成一系列时间索引，每隔 900 秒增加一次
            idxs = [base + timedelta(seconds=i * 900) for i in range(16)]
            # 创建一个初始值为 0，索引为 idxs[0] 的 Series 对象
            result = Series([0], index=[idxs[0]])
            # 逐步在 Series 对象中插入值为 1 的时间戳
            for ts in idxs:
                result.loc[ts] = 1
            # 创建预期的 Series 对象，所有索引对应的值都为 1
            expected = Series(1, index=idxs)
            # 断言两个 Series 对象是否相等
            tm.assert_series_equal(result, expected)

        # 使用 pytest 的参数化装饰器测试 datetime 键的 loc 设置项的类型转换
        @pytest.mark.parametrize(
            "conv",
            [
                lambda x: x,
                lambda x: x.to_datetime64(),
                lambda x: x.to_pydatetime(),
                lambda x: np.datetime64(x),
            ],
            ids=["self", "to_datetime64", "to_pydatetime", "np.datetime64"],
        )
        def test_loc_setitem_datetime_keys_cast(self, conv):
            # GH#9516, GH#51363 changed in 3.0 to not cast on Index.insert
            # 创建两个时间戳对象 dt1 和 dt2
            dt1 = Timestamp("20130101 09:00:00")
            dt2 = Timestamp("20130101 10:00:00")
            # 创建一个空的 DataFrame 对象
            df = DataFrame()
            # 使用 loc 将时间戳 dt1 转换为 conv(dt1) 作为索引，插入值为 100 到列 "one"
            df.loc[conv(dt1), "one"] = 100
            # 使用 loc 将时间戳 dt2 转换为 conv(dt2) 作为索引，插入值为 200 到列 "one"
            df.loc[conv(dt2), "one"] = 200

            # 创建预期的 DataFrame 对象，包含一列 "one"，两行索引为 conv(dt1) 和 conv(dt2)
            expected = DataFrame(
                {"one": [100.0, 200.0]},
                index=Index([conv(dt1), conv(dt2)], dtype=object),
                columns=Index(["one"], dtype=object),
            )
            # 断言两个 DataFrame 对象是否相等
            tm.assert_frame_equal(df, expected)
    # 测试函数，用于验证设置分类列后数据类型保持不变
    def test_loc_setitem_categorical_column_retains_dtype(self, ordered):
        # GH16360
        # 创建包含单列"A"的DataFrame
        result = DataFrame({"A": [1]})
        # 使用.loc方法设置列"B"为有序分类数据
        result.loc[:, "B"] = Categorical(["b"], ordered=ordered)
        # 创建预期的DataFrame，包含"A"列和"B"列，B列为有序分类数据
        expected = DataFrame({"A": [1], "B": Categorical(["b"], ordered=ordered)})
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，验证.loc设置扩展和现有目标的情况
    def test_loc_setitem_with_expansion_and_existing_dst(self):
        # GH#18308
        # 定义时间戳变量
        start = Timestamp("2017-10-29 00:00:00+0200", tz="Europe/Madrid")
        end = Timestamp("2017-10-29 03:00:00+0100", tz="Europe/Madrid")
        ts = Timestamp("2016-10-10 03:00:00", tz="Europe/Madrid")
        # 创建时间范围索引
        idx = date_range(start, end, inclusive="left", freq="h")
        # 断言时间戳不在索引中
        assert ts not in idx  # i.e. result.loc setitem is with-expansion

        # 创建带有索引和"value"列的DataFrame
        result = DataFrame(index=idx, columns=["value"])
        # 使用.loc方法设置时间戳对应的"value"列为12
        result.loc[ts, "value"] = 12
        # 创建预期的DataFrame，扩展了一个包含时间戳的索引和"value"列
        expected = DataFrame(
            [np.nan] * len(idx) + [12],
            index=idx.append(DatetimeIndex([ts])),
            columns=["value"],
            dtype=object,
        )
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，验证带有扩展的设置元素操作
    def test_setitem_with_expansion(self):
        # indexing - setting an element
        # 创建包含时间数据的DataFrame
        df = DataFrame(
            data=to_datetime(["2015-03-30 20:12:32", "2015-03-12 00:11:11"]),
            columns=["time"],
        )
        # 添加新列"new_col"
        df["new_col"] = ["new", "old"]
        # 设置"time"列为索引后重新赋值为UTC时区
        df.time = df.set_index("time").index.tz_localize("UTC")
        # 根据条件筛选并设置索引为"US/Pacific"时区
        v = df[df.new_col == "new"].set_index("time").index.tz_convert("US/Pacific")

        # 在不同时区对象的旧版本尝试设置单个元素；在2.0版本中保持数据类型
        df2 = df.copy()
        df2.loc[df2.new_col == "new", "time"] = v

        # 创建预期的Series，包含经过时区转换后的第一个时间戳和原DataFrame的第二个时间戳
        expected = Series([v[0].tz_convert("UTC"), df.loc[1, "time"]], name="time")
        # 断言两个Series是否相等
        tm.assert_series_equal(df2.time, expected)

        # 计算新时间戳值并赋给对应条件的"time"列
        v = df.loc[df.new_col == "new", "time"] + Timedelta("1s").as_unit("s")
        df.loc[df.new_col == "new", "time"] = v
        # 断言条件后"time"列是否与新时间戳值相等
        tm.assert_series_equal(df.loc[df.new_col == "new", "time"], v)

    # 测试函数，验证带有扩展的设置元素操作和空数据框的无穷大上溢
    def test_loc_setitem_with_expansion_inf_upcast_empty(self):
        # Test with np.inf in columns
        # 创建空的DataFrame
        df = DataFrame()
        # 设置特定位置的元素为1和2
        df.loc[0, 0] = 1
        df.loc[1, 1] = 2
        df.loc[0, np.inf] = 3

        # 获取结果的列索引
        result = df.columns
        # 创建预期的Index，包含0、1和np.inf
        expected = Index([0, 1, np.inf], dtype=np.float64)
        # 断言两个Index是否相等
        tm.assert_index_equal(result, expected)

    # 使用pytest标记，忽略超出lexsort深度的索引警告
    @pytest.mark.filterwarnings("ignore:indexing past lexsort depth")
    # 测试函数：test_loc_setitem_with_expansion_nonunique_index，用于测试在非唯一索引情况下的 loc 设置项功能
    def test_loc_setitem_with_expansion_nonunique_index(self, index):
        # GH#40096：引用 GitHub 问题编号 40096
        if not len(index):
            pytest.skip("Not relevant for empty Index")  # 如果索引为空，则跳过测试

        index = index.repeat(2)  # 确保索引非唯一
        N = len(index)
        arr = np.arange(N).astype(np.int64)

        orig = DataFrame(arr, index=index, columns=[0])

        # key that will requiring object-dtype casting in the index
        key = "kapow"
        assert key not in index  # 断言确保 key 不在索引中，否则测试无效
        # TODO: using a tuple key breaks here in many cases
        # TODO: 使用元组键在许多情况下会导致问题

        exp_index = index.insert(len(index), key)  # 将 key 插入到索引的末尾
        if isinstance(index, MultiIndex):
            assert exp_index[-1][0] == key  # 如果是 MultiIndex，检查最后一个元素的第一个级别是否为 key
        else:
            assert exp_index[-1] == key  # 否则，检查最后一个元素是否为 key
        exp_data = np.arange(N + 1).astype(np.float64)
        expected = DataFrame(exp_data, index=exp_index, columns=[0])

        # Add new row, but no new columns
        df = orig.copy()
        df.loc[key, 0] = N  # 在索引为 key，列为 0 的位置设置新的值 N
        tm.assert_frame_equal(df, expected)  # 断言 df 和 expected 的内容是否相等

        # add new row on a Series
        ser = orig.copy()[0]
        ser.loc[key] = N
        # the series machinery lets us preserve int dtype instead of float
        # 系列机制使我们可以保持整数类型而不是浮点类型
        expected = expected[0].astype(np.int64)
        tm.assert_series_equal(ser, expected)  # 断言 ser 和 expected 的内容是否相等

        # add new row and new column
        df = orig.copy()
        df.loc[key, 1] = N  # 在索引为 key，列为 1 的位置设置新的值 N
        expected = DataFrame(
            {0: list(arr) + [np.nan], 1: [np.nan] * N + [float(N)]},
            index=exp_index,
        )
        tm.assert_frame_equal(df, expected)  # 断言 df 和 expected 的内容是否相等

    # 测试函数：test_loc_setitem_with_expansion_preserves_nullable_int，测试 loc 设置项在保留可空整数类型时的行为
    def test_loc_setitem_with_expansion_preserves_nullable_int(
        self, any_numeric_ea_dtype
    ):
        # GH#42099：引用 GitHub 问题编号 42099
        ser = Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype)
        df = DataFrame({"data": ser})

        result = DataFrame(index=df.index)
        result.loc[df.index, "data"] = ser

        tm.assert_frame_equal(result, df, check_column_type=False)  # 断言 result 和 df 的内容是否相等，忽略列类型的检查

        result = DataFrame(index=df.index)
        result.loc[df.index, "data"] = ser._values
        tm.assert_frame_equal(result, df, check_column_type=False)  # 断言 result 和 df 的内容是否相等，忽略列类型的检查

    # 测试函数：test_loc_setitem_ea_not_full_column，测试 loc 设置项在不完整列情况下的行为
    def test_loc_setitem_ea_not_full_column(self):
        # GH#39163：引用 GitHub 问题编号 39163
        df = DataFrame({"A": range(5)})

        val = date_range("2016-01-01", periods=3, tz="US/Pacific")

        df.loc[[0, 1, 2], "B"] = val  # 在索引为 [0, 1, 2]，列为 'B' 的位置设置新的值 val

        bex = val.append(DatetimeIndex([pd.NaT, pd.NaT], dtype=val.dtype))
        expected = DataFrame({"A": range(5), "B": bex})
        assert expected.dtypes["B"] == val.dtype  # 断言 'B' 列的数据类型与 val 的数据类型相同
        tm.assert_frame_equal(df, expected)  # 断言 df 和 expected 的内容是否相等
class TestLocCallable:
    # 定义测试类 TestLocCallable

    def test_frame_loc_getitem_callable(self):
        # 定义测试方法 test_frame_loc_getitem_callable

        # 创建一个 DataFrame 包含三列数据
        df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})
        
        # 使用 callable 对象作为 loc 的索引器，选择满足条件 x.A > 2 的行
        res = df.loc[lambda x: x.A > 2]
        # 断言返回的 DataFrame 等于通过布尔索引选择的行
        tm.assert_frame_equal(res, df.loc[df.A > 2])

        # 使用 callable 对象作为 loc 的索引器，选择满足条件 x.B == "b" 的行
        res = df.loc[lambda x: x.B == "b", :]
        # 断言返回的 DataFrame 等于通过布尔索引选择的行
        tm.assert_frame_equal(res, df.loc[df.B == "b", :])

        # 使用 callable 对象作为 loc 的索引器，选择满足条件 x.A > 2 的行和列名为 "B" 的列
        res = df.loc[lambda x: x.A > 2, lambda x: x.columns == "B"]
        # 断言返回的 DataFrame 等于通过布尔索引选择的行和列
        tm.assert_frame_equal(res, df.loc[df.A > 2, [False, True, False]])

        # 使用 callable 对象作为 loc 的索引器，选择满足条件 x.A > 2 的行和列名为 "B" 的列
        res = df.loc[lambda x: x.A > 2, lambda x: "B"]
        # 断言返回的 Series 等于通过布尔索引选择的列 "B"
        tm.assert_series_equal(res, df.loc[df.A > 2, "B"])

        # 使用 callable 对象作为 loc 的索引器，选择满足条件 x.A > 2 的行和列名为 ["A", "B"] 的列
        res = df.loc[lambda x: x.A > 2, lambda x: ["A", "B"]]
        # 断言返回的 DataFrame 等于通过布尔索引选择的列 ["A", "B"]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])

        # 使用 callable 对象作为 loc 的索引器，选择满足条件 x.A == 2 的行和列名为 ["A", "B"] 的列
        res = df.loc[lambda x: x.A == 2, lambda x: ["A", "B"]]
        # 断言返回的 DataFrame 等于通过布尔索引选择的行和列 ["A", "B"]
        tm.assert_frame_equal(res, df.loc[df.A == 2, ["A", "B"]])

        # 使用 callable 对象作为 loc 的索引器，选择 scalar 值 1 和列名为 "A" 的列
        res = df.loc[lambda x: 1, lambda x: "A"]
        # 断言返回的值等于通过位置索引选择的值 df.loc[1, "A"]
        assert res == df.loc[1, "A"]

    def test_frame_loc_getitem_callable_mixture(self):
        # 定义测试方法 test_frame_loc_getitem_callable_mixture

        # 创建一个 DataFrame 包含三列数据
        df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})

        # 使用 callable 对象作为 loc 的索引器，选择满足条件 x.A > 2 的行和列名为 ["A", "B"] 的列
        res = df.loc[lambda x: x.A > 2, ["A", "B"]]
        # 断言返回的 DataFrame 等于通过布尔索引选择的行和列 ["A", "B"]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])

        # 使用列表作为行索引和 callable 对象作为列索引器，选择行索引为 [2, 3] 和列名为 ["A", "B"] 的列
        res = df.loc[[2, 3], lambda x: ["A", "B"]]
        # 断言返回的 DataFrame 等于通过位置索引选择的行和列 ["A", "B"]
        tm.assert_frame_equal(res, df.loc[[2, 3], ["A", "B"]])

        # 使用标量值 3 作为行索引和 callable 对象作为列索引器，选择行索引为 3 和列名为 ["A", "B"] 的列
        res = df.loc[3, lambda x: ["A", "B"]]
        # 断言返回的 Series 等于通过位置索引选择的行和列 ["A", "B"]
        tm.assert_series_equal(res, df.loc[3, ["A", "B"]])

    def test_frame_loc_getitem_callable_labels(self):
        # 定义测试方法 test_frame_loc_getitem_callable_labels

        # 创建一个 DataFrame 包含两列数据，行索引为 ["A", "B", "C", "D"]
        df = DataFrame({"X": [1, 2, 3, 4], "Y": list("aabb")}, index=list("ABCD"))

        # 使用 callable 对象作为 loc 的行索引器，选择行索引为 ["A", "C"] 的行
        res = df.loc[lambda x: ["A", "C"]]
        # 断言返回的 DataFrame 等于通过标签索引选择的行
        tm.assert_frame_equal(res, df.loc[["A", "C"]])

        # 使用 callable 对象作为 loc 的行索引器，选择行索引为 ["A", "C"] 和所有列的数据
        res = df.loc[lambda x: ["A", "C"], :]
        # 断言返回的 DataFrame 等于通过标签索引选择的行和所有列
        tm.assert_frame_equal(res, df.loc[["A", "C"], :])

        # 使用 callable 对象作为 loc 的行索引器和列索引器，选择行索引为 ["A", "C"] 和列名为 "X" 的列
        res = df.loc[lambda x: ["A", "C"], lambda x: "X"]
        # 断言返回的 Series 等于通过标签索引选择的行和列 "X"
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        # 使用 callable 对象作为 loc 的行索引器和列索引器，选择行索引为 ["A", "C"] 和列名为 ["X"] 的列
        res = df.loc[lambda x: ["A", "C"], lambda x: ["X"]]
        # 断言返回的 DataFrame 等于通过标签索引选择的行和列 ["X"]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

        # 使用 callable 对象作为 loc 的行索引器和标签 "X"，选择行索引为 ["A", "C"] 和列名为 "X" 的列
        res = df.loc[["A", "C"], lambda x: "X"]
        # 断言返回的 Series 等于通过标签索引选择的行和列 "X"
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        # 使用 callable 对象作为 loc 的行索引器和列索引器，选择行索引为 ["A", "C"] 和列名为 ["X"] 的列
        res = df.loc[["A", "C"], lambda x: ["X"]]
        # 断言返回的 DataFrame 等于通过标签索引选择的行和列 ["X"]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

        # 使用 callable 对象作为 loc 的行索引器，选择行索引为 ["A", "C"] 和标签 "X" 的列
        res = df.loc[lambda x: ["A", "C"], "X"]
        # 断言返回的 Series 等于通过标签索引选择的行和列 "X"
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        # 使用 callable 对象作为 loc 的行索引器，选择行索引为 ["A", "C"] 和标签 ["X"] 的列
        res = df.loc[lambda x: ["A", "C"], ["X"]]
        # 断言返回的 DataFrame 等于通过标签索引选择的行和列 ["X
    # 定义一个测试方法，用于测试DataFrame的loc属性的可调用方式设置元素
    def test_frame_loc_setitem_callable(self):
        # 创建一个DataFrame对象df，包含两列：'X'列和'Y'列，其中'Y'列是包含字符列表的Series对象，数据类型为对象
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": Series(list("aabb"), dtype=object)},
            index=list("ABCD"),
        )

        # 测试用例1: 使用可调用函数设置标签为'A'和'C'的行为-20
        res = df.copy()
        res.loc[lambda x: ["A", "C"]] = -20
        exp = df.copy()
        exp.loc[["A", "C"]] = -20
        tm.assert_frame_equal(res, exp)

        # 测试用例2: 使用可调用函数设置标签为'A'和'C'的行和所有列为20
        res = df.copy()
        res.loc[lambda x: ["A", "C"], :] = 20
        exp = df.copy()
        exp.loc[["A", "C"], :] = 20
        tm.assert_frame_equal(res, exp)

        # 测试用例3: 使用可调用函数设置标签为'A'和'C'的行和列'X'为-1
        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: "X"] = -1
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = -1
        tm.assert_frame_equal(res, exp)

        # 测试用例4: 使用可调用函数设置标签为'A'和'C'的行和列'X'为数组[5, 10]
        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: ["X"]] = [5, 10]
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = [5, 10]
        tm.assert_frame_equal(res, exp)

        # 测试用例5: 使用可调用函数设置标签为'A'和'C'的行和列'X'为数组[-1, -2]
        res = df.copy()
        res.loc[["A", "C"], lambda x: "X"] = np.array([-1, -2])
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = np.array([-1, -2])
        tm.assert_frame_equal(res, exp)

        # 测试用例6: 使用可调用函数设置标签为'A'和'C'的行和列'X'为10
        res = df.copy()
        res.loc[["A", "C"], lambda x: ["X"]] = 10
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = 10
        tm.assert_frame_equal(res, exp)

        # 测试用例7: 使用可调用函数设置标签为'A'和'C'的行为'X'列为-2
        res = df.copy()
        res.loc[lambda x: ["A", "C"], "X"] = -2
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = -2
        tm.assert_frame_equal(res, exp)

        # 测试用例8: 使用可调用函数设置标签为'A'和'C'的行和列'X'为-4
        res = df.copy()
        res.loc[lambda x: ["A", "C"], ["X"]] = -4
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = -4
        tm.assert_frame_equal(res, exp)
class TestPartialStringSlicing:
    def test_loc_getitem_partial_string_slicing_datetimeindex(self):
        # 创建一个 DataFrame 包含两列，其中一列是字符串列表，另一列是整数列表，使用日期时间索引
        df = DataFrame(
            {"col1": ["a", "b", "c"], "col2": [1, 2, 3]},
            index=to_datetime(["2020-08-01", "2020-07-02", "2020-08-05"]),
        )
        # 创建预期的 DataFrame，选择指定日期时间的行
        expected = DataFrame(
            {"col1": ["a", "c"], "col2": [1, 3]},
            index=to_datetime(["2020-08-01", "2020-08-05"]),
        )
        # 使用 loc 方法根据部分字符串选择行，并存储结果
        result = df.loc["2020-08"]
        # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_periodindex(self):
        # 创建一个周期范围，转换为 Series
        pi = pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")
        ser = pi.to_series()
        # 使用 loc 方法根据部分字符串选择项，存储结果
        result = ser.loc[:"2017-12"]
        # 创建预期的 Series，使用 iloc 方法选择除最后一项之外的所有项
        expected = ser.iloc[:-1]
        # 使用测试工具比较结果 Series 和预期 Series 是否相等
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_timedeltaindex(self):
        # 创建一个时间差范围，转换为 Series
        ix = timedelta_range(start="1 day", end="2 days", freq="1h")
        ser = ix.to_series()
        # 使用 loc 方法根据部分字符串选择项，存储结果
        result = ser.loc[:"1 days"]
        # 创建预期的 Series，使用 iloc 方法选择除最后一项之外的所有项
        expected = ser.iloc[:-1]
        # 使用测试工具比较结果 Series 和预期 Series 是否相等
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_str_timedeltaindex(self):
        # 创建一个 DataFrame，使用时间差作为索引
        df = DataFrame({"x": range(3)}, index=to_timedelta(range(3), unit="days"))
        # 创建预期的 Series，选择第一行
        expected = df.iloc[0]
        # 使用 loc 方法根据时间差字符串选择行，并存储结果
        sliced = df.loc["0 days"]
        # 使用测试工具比较结果 Series 和预期 Series 是否相等
        tm.assert_series_equal(sliced, expected)

    @pytest.mark.parametrize("indexer_end", [None, "2020-01-02 23:59:59.999999999"])
    def test_loc_getitem_partial_slice_non_monotonicity(
        self, tz_aware_fixture, indexer_end, frame_or_series
    ):
        # 创建一个对象（DataFrame 或 Series），使用日期时间索引，包含时区信息
        obj = frame_or_series(
            [1] * 5,
            index=DatetimeIndex(
                [
                    Timestamp("2019-12-30"),
                    Timestamp("2020-01-01"),
                    Timestamp("2019-12-25"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                    Timestamp("2019-12-19"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        # 创建预期的对象（DataFrame 或 Series），选择特定日期时间范围的项
        expected = frame_or_series(
            [1] * 2,
            index=DatetimeIndex(
                [
                    Timestamp("2020-01-01"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        # 创建索引切片对象
        indexer = slice("2020-01-01", indexer_end)

        # 使用切片对象选择对象的数据，并存储结果
        result = obj[indexer]
        # 使用测试工具比较结果和预期是否相等
        tm.assert_equal(result, expected)

        # 使用 loc 方法根据切片对象选择对象的数据，并存储结果
        result = obj.loc[indexer]
        # 使用测试工具比较结果和预期是否相等
        tm.assert_equal(result, expected)
    def test_loc_getitem_slicing_datetimes_frame(self):
        # GH#7523
        # 测试用例函数，用于测试 Pandas DataFrame 的日期时间切片操作

        # 创建唯一索引的 DataFrame
        df_unique = DataFrame(
            np.arange(4.0, dtype="float64"),
            index=[datetime(2001, 1, i, 10, 00) for i in [1, 2, 3, 4]],
        )

        # 创建有重复索引的 DataFrame
        df_dups = DataFrame(
            np.arange(5.0, dtype="float64"),
            index=[datetime(2001, 1, i, 10, 00) for i in [1, 2, 2, 3, 4]],
        )

        # 遍历两种 DataFrame 进行测试
        for df in [df_unique, df_dups]:
            # 测试从特定日期时间开始到最后的切片是否返回整个 DataFrame
            result = df.loc[datetime(2001, 1, 1, 10) :]
            tm.assert_frame_equal(result, df)
            # 测试从开始到指定日期时间结束的切片是否返回整个 DataFrame
            result = df.loc[: datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)
            # 测试从指定日期时间开始到指定日期时间结束的切片是否返回整个 DataFrame
            result = df.loc[datetime(2001, 1, 1, 10) : datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)

            # 测试从指定日期时间之后的切片是否按预期返回
            result = df.loc[datetime(2001, 1, 1, 11) :]
            expected = df.iloc[1:]  # 预期是从第二行开始的 DataFrame
            tm.assert_frame_equal(result, expected)
            # 以字符串形式指定日期时间进行切片，预期与上一行相同
            result = df.loc["20010101 11":]
            tm.assert_frame_equal(result, expected)

    def test_loc_getitem_label_slice_across_dst(self):
        # GH#21846
        # 测试用例函数，测试跨越夏令时的 Pandas Series 的标签切片操作

        # 创建具有时区信息的日期时间索引
        idx = date_range(
            "2017-10-29 01:30:00", tz="Europe/Berlin", periods=5, freq="30 min"
        )
        series2 = Series([0, 1, 2, 3, 4], index=idx)

        # 创建两个带时区的 Timestamp 对象
        t_1 = Timestamp("2017-10-29 02:30:00+02:00", tz="Europe/Berlin")
        t_2 = Timestamp("2017-10-29 02:00:00+01:00", tz="Europe/Berlin")

        # 测试从 t_1 到 t_2 之间的标签切片操作是否返回预期结果
        result = series2.loc[t_1:t_2]
        expected = Series([2, 3], index=idx[2:4])
        tm.assert_series_equal(result, expected)

        # 测试直接使用 t_1 进行索引是否返回预期的单个值
        result = series2[t_1]
        expected = 2
        assert result == expected

    @pytest.mark.parametrize(
        "index",
        [
            pd.period_range(start="2017-01-01", end="2018-01-01", freq="M"),
            timedelta_range(start="1 day", end="2 days", freq="1h"),
        ],
    )
    def test_loc_getitem_label_slice_period_timedelta(self, index):
        # 测试用例函数，测试 Pandas Series 的标签切片操作对于 Period 和 Timedelta 索引的情况

        # 将索引转换为 Series 对象
        ser = index.to_series()

        # 测试从开头到倒数第二个索引进行标签切片操作是否返回预期结果
        result = ser.loc[: index[-2]]
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_floats_inexact(self):
        # 测试用例函数，测试 Pandas DataFrame 的浮点数索引切片操作（精度不精确的情况）

        # 创建浮点数索引的 DataFrame
        index = [52195.504153, 52196.303147, 52198.369883]
        df = DataFrame(np.random.default_rng(2).random((3, 2)), index=index)

        # 测试从 52195.1 到 52196.5 的浮点数切片操作是否返回预期长度的 Series
        s1 = df.loc[52195.1:52196.5]
        assert len(s1) == 2

        # 测试从 52195.1 到 52196.6 的浮点数切片操作是否返回预期长度的 Series
        s1 = df.loc[52195.1:52196.6]
        assert len(s1) == 2

        # 测试从 52195.1 到 52198.9 的浮点数切片操作是否返回预期长度的 Series
        s1 = df.loc[52195.1:52198.9]
        assert len(s1) == 3

    def test_loc_getitem_float_slice_floatindex(self, float_numpy_dtype):
        # 测试用例函数，测试 Pandas Series 的浮点数索引切片操作（具有浮点数索引的情况）

        dtype = float_numpy_dtype
        # 创建具有浮点数索引的 Series 对象
        ser = Series(
            np.random.default_rng(2).random(10), index=np.arange(10, 20, dtype=dtype)
        )

        # 测试从 12.0 开始到末尾的切片操作是否返回预期长度
        assert len(ser.loc[12.0:]) == 8
        # 测试从 12.5 开始到末尾的切片操作是否返回预期长度
        assert len(ser.loc[12.5:]) == 7

        # 修改索引中的一个值为 12.2，测试从 12.0 开始到末尾的切片操作是否返回预期长度
        idx = np.arange(10, 20, dtype=dtype)
        idx[2] = 12.2
        ser.index = idx
        assert len(ser.loc[12.0:]) == 8
        # 测试从 12.5 开始到末尾的切片操作是否返回预期长度
        assert len(ser.loc[12.5:]) == 7
    @pytest.mark.parametrize(
        "start,stop, expected_slice",
        [
            [np.timedelta64(0, "ns"), None, slice(0, 11)],
            [np.timedelta64(1, "D"), np.timedelta64(6, "D"), slice(1, 7)],
            [None, np.timedelta64(4, "D"), slice(0, 5)],
        ],
    )
    # 定义参数化测试函数，用于测试 loc 操作对时间间隔索引的切片操作
    def test_loc_getitem_slice_label_td64obj(self, start, stop, expected_slice):
        # GH#20393
        # 创建一个 Series 对象，索引为时间间隔范围从 "0 days" 到 "10 days"
        ser = Series(range(11), timedelta_range("0 days", "10 days"))
        # 执行 loc 切片操作，并存储结果
        result = ser.loc[slice(start, stop)]
        # 根据预期切片索引，从原始 Series 中获取预期结果
        expected = ser.iloc[expected_slice]
        # 使用测试工具比较结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("start", ["2018", "2020"])
    # 定义参数化测试函数，测试 loc 操作在非单调日期时间索引上的部分切片
    def test_loc_getitem_slice_unordered_dt_index(self, frame_or_series, start):
        # 创建一个 DataFrame 或 Series 对象，索引为时间戳列表
        obj = frame_or_series(
            [1, 2, 3],
            index=[Timestamp("2016"), Timestamp("2019"), Timestamp("2017")],
        )
        # 使用 pytest 检查是否会引发 KeyError，并匹配特定错误信息
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            # 执行 loc 切片操作，尝试切片从 start 到 "2022" 的部分
            obj.loc[start:"2022"]

    @pytest.mark.parametrize("value", [1, 1.5])
    # 定义参数化测试函数，测试 loc 操作在对象索引中使用整数标签切片
    def test_loc_getitem_slice_labels_int_in_object_index(self, frame_or_series, value):
        # 创建一个 DataFrame 或 Series 对象，索引为混合类型标签列表
        obj = frame_or_series(range(4), index=[value, "first", 2, "third"])
        # 执行 loc 切片操作，根据整数标签值从 value 到 "third" 进行切片
        result = obj.loc[value:"third"]
        # 创建一个预期的 DataFrame 或 Series 对象，用于比较结果
        expected = frame_or_series(range(4), index=[value, "first", 2, "third"])
        # 使用测试工具比较结果和预期结果是否相等
        tm.assert_equal(result, expected)

    # 定义测试函数，测试 loc 操作在列名为混合数据类型时的切片操作
    def test_loc_getitem_slice_columns_mixed_dtype(self):
        # 创建一个 DataFrame 对象，列包含整数和字符串作为列名
        df = DataFrame({"test": 1, 1: 2, 2: 3}, index=[0])
        # 创建一个预期的 DataFrame 对象，包含从列名为 1 开始的部分数据
        expected = DataFrame(
            data=[[2, 3]], index=[0], columns=Index([1, 2], dtype=object)
        )
        # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
        tm.assert_frame_equal(df.loc[:, 1:], expected)
# 定义一个测试类 TestLocBooleanLabelsAndSlices，用于测试 loc 方法在布尔索引和切片中的行为
class TestLocBooleanLabelsAndSlices:
    
    # 使用 pytest 的参数化装饰器，测试布尔值为 True 和 False 时的场景
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_loc_bool_incompatible_index_raises(
        self, index, frame_or_series, bool_value
    ):
        # GH20432
        # 准备错误消息，指示布尔标签在没有布尔索引的情况下不能使用
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        
        # 如果索引的推断类型不是布尔型
        if index.inferred_type != "boolean":
            # 创建一个 DataFrame 或 Series 对象，传入指定的索引和数据类型为对象
            obj = frame_or_series(index=index, dtype="object")
            # 使用 pytest 断言抛出 KeyError 异常，并匹配指定消息
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

    # 使用 pytest 的参数化装饰器，测试布尔值为 True 和 False 时的场景
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_loc_bool_should_not_raise(self, frame_or_series, bool_value):
        # 创建一个 DataFrame 或 Series 对象，传入布尔类型的索引和对象数据类型
        obj = frame_or_series(
            index=Index([True, False], dtype="boolean"), dtype="object"
        )
        # 使用 loc 方法，传入布尔值进行索引操作
        obj.loc[bool_value]

    # 测试 loc 方法在布尔切片时的行为
    def test_loc_bool_slice_raises(self, index, frame_or_series):
        # GH20432
        # 准备错误消息，指示布尔值在切片中不能使用
        message = (
            r"slice\(True, False, None\): boolean values can not be used in a slice"
        )
        # 创建一个 DataFrame 或 Series 对象，传入指定的索引和数据类型为对象
        obj = frame_or_series(index=index, dtype="object")
        # 使用 pytest 断言抛出 TypeError 异常，并匹配指定消息
        with pytest.raises(TypeError, match=message):
            obj.loc[True:False]


# 定义一个测试类 TestLocBooleanMask，用于测试 loc 方法在布尔掩码时的行为
class TestLocBooleanMask:
    
    # 测试 loc 方法在布尔掩码与 TimedeltaIndex 时的行为
    def test_loc_setitem_bool_mask_timedeltaindex(self):
        # GH#14946
        # 创建一个 DataFrame，包含一列 x，其值为 0 到 9
        df = DataFrame({"x": range(10)})
        # 将索引设置为以秒为单位的 TimedeltaIndex
        df.index = to_timedelta(range(10), unit="s")
        # 定义条件列表，每个条件为 df["x"] 的比较结果
        conditions = [df["x"] > 3, df["x"] == 3, df["x"] < 3]
        # 期望的数据结果，根据条件进行修改后的值
        expected_data = [
            [0, 1, 2, 3, 10, 10, 10, 10, 10, 10],
            [0, 1, 2, 10, 4, 5, 6, 7, 8, 9],
            [10, 10, 10, 3, 4, 5, 6, 7, 8, 9],
        ]
        
        # 遍历条件和期望数据
        for cond, data in zip(conditions, expected_data):
            # 复制原始 DataFrame
            result = df.copy()
            # 使用 loc 方法根据条件修改列 x 的值为 10
            result.loc[cond, "x"] = 10
            
            # 创建期望的 DataFrame，传入预期的数据和 TimedeltaIndex
            expected = DataFrame(
                data,
                index=to_timedelta(range(10), unit="s"),
                columns=["x"],
                dtype="int64",
            )
            # 使用 pandas 的测试工具比较结果和期望值是否相等
            tm.assert_frame_equal(expected, result)

    # 使用 pytest 的参数化装饰器，测试时区为 None 和 "UTC" 时的行为
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_loc_setitem_mask_with_datetimeindex_tz(self, tz):
        # GH#16889
        # 支持具有对齐和时区感知的 DatetimeIndex 的 loc 方法
        # 创建一个布尔掩码数组
        mask = np.array([True, False, True, False])
        
        # 创建一个 DateTimeIndex，从 "20010101" 开始，共四个周期，指定时区为 tz
        idx = date_range("20010101", periods=4, tz=tz)
        # 创建一个 DataFrame，包含一列 a，其值为 0 到 3，并转换为浮点数
        df = DataFrame({"a": np.arange(4)}, index=idx).astype("float64")

        # 复制原始 DataFrame
        result = df.copy()
        # 使用 loc 方法根据布尔掩码 mask 修改所有列的值为与原始 DataFrame 相同的值
        result.loc[mask, :] = df.loc[mask, :]
        # 使用 pandas 的测试工具比较结果和原始 DataFrame 是否相等
        tm.assert_frame_equal(result, df)

        # 复制原始 DataFrame
        result = df.copy()
        # 使用 loc 方法根据布尔掩码 mask 修改 DataFrame 的值为与原始 DataFrame 相同的值
        result.loc[mask] = df.loc[mask]
        # 使用 pandas 的测试工具比较结果和原始 DataFrame 是否相等
        tm.assert_frame_equal(result, df)
    def test_loc_setitem_mask_and_label_with_datetimeindex(self):
        # GH#9478
        # 解决了在日期时间索引部分设置时的对齐问题
        df = DataFrame(
            np.arange(6.0).reshape(3, 2),
            columns=list("AB"),
            index=date_range("1/1/2000", periods=3, freq="1h"),
        )
        expected = df.copy()
        # 在 DataFrame 副本上添加新列"C"，并在第一行设置日期时间索引值，其它行设置为 NaT
        expected["C"] = [expected.index[0]] + [pd.NaT, pd.NaT]

        mask = df.A < 1
        # 使用 mask 来选择符合条件的行，并将其对应的"C"列设置为行的日期时间索引值
        df.loc[mask, "C"] = df.loc[mask].index
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_mask_td64_series_value(self):
        # GH#23462 用布尔键列表，值为 Series 的情况
        td1 = Timedelta(0)
        td2 = Timedelta(28767471428571405)
        df = DataFrame({"col": Series([td1, td2])})
        df_copy = df.copy()
        ser = Series([td1])

        expected = df["col"].iloc[1]._value
        # 使用布尔列表选择行，并将其替换为 Series 中的值
        df.loc[[True, False]] = ser
        result = df["col"].iloc[1]._value

        assert expected == result
        tm.assert_frame_equal(df, df_copy)

    def test_loc_setitem_boolean_and_column(self, float_frame):
        expected = float_frame.copy()
        mask = float_frame["A"] > 0

        # 根据条件 mask 选择行，并将"B"列设置为0
        float_frame.loc[mask, "B"] = 0

        values = expected.values.copy()
        values[mask.values, 1] = 0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(float_frame, expected)

    def test_loc_setitem_ndframe_values_alignment(self):
        # GH#45501
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 使用 DataFrame 替换选择的行和列
        df.loc[[False, False, True], ["a"]] = DataFrame(
            {"a": [10, 20, 30]}, index=[2, 1, 0]
        )

        expected = DataFrame({"a": [1, 2, 10], "b": [4, 5, 6]})
        tm.assert_frame_equal(df, expected)

        # 使用 Series 替换选择的行和列
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], ["a"]] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)

        # 使用 Series 替换选择的行和单独的列"a"
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], "a"] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)

        # 使用 Series 替换选择的行的一部分，但保留 DataFrame 的原始状态
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()
        ser = df["a"]
        ser.loc[[False, False, True]] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, df_orig)

    def test_loc_indexer_empty_broadcast(self):
        # GH#51450 处理空索引广播的问题
        df = DataFrame({"a": [], "b": []}, dtype=object)
        expected = df.copy()
        # 使用空布尔数组选择并广播"a"列的值
        df.loc[np.array([], dtype=np.bool_), ["a"]] = df["a"].copy()
        tm.assert_frame_equal(df, expected)
    # 定义测试方法，验证 loc 索引器中所有值为 False 时的广播行为
    def test_loc_indexer_all_false_broadcast(self):
        # 使用 DataFrame 创建包含一个字符串列 'a' 和一个字符串列 'b' 的对象
        df = DataFrame({"a": ["x"], "b": ["y"]}, dtype=object)
        # 复制 df，准备作为预期输出结果
        expected = df.copy()
        # 将 df 中所有值为 False 的行的 'a' 列设置为 'b' 列的复制
        df.loc[np.array([False], dtype=np.bool_), ["a"]] = df["b"].copy()
        # 使用测试框架验证 df 和预期结果的相等性
        tm.assert_frame_equal(df, expected)

    # 定义测试方法，验证 loc 索引器中长度为一的情况
    def test_loc_indexer_length_one(self):
        # 使用 DataFrame 创建包含一个字符串列 'a' 和一个字符串列 'b' 的对象
        df = DataFrame({"a": ["x"], "b": ["y"]}, dtype=object)
        # 创建预期结果，将 'a' 列设置为 'b' 列的内容
        expected = DataFrame({"a": ["y"], "b": ["y"]}, dtype=object)
        # 将 df 中索引为 True 的行的 'a' 列设置为 'b' 列的复制
        df.loc[np.array([True], dtype=np.bool_), ["a"]] = df["b"].copy()
        # 使用测试框架验证 df 和预期结果的相等性
        tm.assert_frame_equal(df, expected)
    # 定义一个测试类 TestLocListlike
    class TestLocListlike:
        
        # 使用 pytest.mark.parametrize 装饰器，为 test_loc_getitem_list_of_labels_categoricalindex_with_na 方法参数化测试用例
        @pytest.mark.parametrize("box", [lambda x: x, np.asarray, list])
        # 测试用例：验证在包含 NA 值的分类索引中，通过传入列表获取元素是否正常工作
        def test_loc_getitem_list_of_labels_categoricalindex_with_na(self, box):
            # 创建一个包含 ["A", "B", np.nan] 的分类索引对象 ci
            ci = CategoricalIndex(["A", "B", np.nan])
            # 创建一个 Series 对象 ser，索引为 ci，值为 range(3)
            ser = Series(range(3), index=ci)

            # 测试 ser.loc[box(ci)] 的结果是否与 ser 相等
            result = ser.loc[box(ci)]
            tm.assert_series_equal(result, ser)

            # 测试 ser[box(ci)] 的结果是否与 ser 相等
            result = ser[box(ci)]
            tm.assert_series_equal(result, ser)

            # 测试 ser.to_frame().loc[box(ci)] 的结果是否与 ser.to_frame() 相等
            result = ser.to_frame().loc[box(ci)]
            tm.assert_frame_equal(result, ser.to_frame())

            # 创建 ser2 为 ser 的部分切片，ci2 为 ci 的部分切片
            ser2 = ser[:-1]
            ci2 = ci[1:]

            # 如果在 ser2 中没有 NA 值存在，应该抛出 KeyError 异常
            msg = "not in index"
            with pytest.raises(KeyError, match=msg):
                ser2.loc[box(ci2)]

            with pytest.raises(KeyError, match=msg):
                ser2[box(ci2)]

            with pytest.raises(KeyError, match=msg):
                ser2.to_frame().loc[box(ci2)]

        # 测试用例：验证在使用日期标签列表获取 Series 中的元素时，缺少特定值是否会引发 KeyError
        def test_loc_getitem_series_label_list_missing_values(self):
            # 创建一个日期标签的 numpy 数组 key
            key = np.array(
                ["2001-01-04", "2001-01-02", "2001-01-04", "2001-01-14"], dtype="datetime64"
            )
            # 创建一个 Series 对象 ser，索引为日期范围从 "2001-01-01" 开始的四个日期，频率为每天一次
            ser = Series([2, 5, 8, 11], date_range("2001-01-01", freq="D", periods=4))
            
            # 当使用 ser.loc[key] 获取元素时，如果 key 中的某些值不在 ser 的索引中，预期会抛出 KeyError 异常
            with pytest.raises(KeyError, match="not in index"):
                ser.loc[key]

        # 测试用例：验证在使用整数标签列表获取 Series 中的元素时，缺少特定值是否会引发 KeyError
        def test_loc_getitem_series_label_list_missing_integer_values(self):
            # 创建一个 Series 对象 ser，其索引为包含两个整数的 numpy 数组，数据也为相同的整数数组
            ser = Series(
                index=np.array([9730701000001104, 10049011000001109]),
                data=np.array([999000011000001104, 999000011000001104]),
            )
            
            # 当使用 ser.loc[np.array([...])] 获取元素时，如果传入的数组中的某些整数值不在 ser 的索引中，预期会抛出 KeyError 异常
            with pytest.raises(KeyError, match="not in index"):
                ser.loc[np.array([9730701000001104, 10047311000001102])]

        # 使用 pytest.mark.parametrize 装饰器，为 to_period 参数化测试用例
        @pytest.mark.parametrize("to_period", [True, False])
    # 测试函数，用于测试 loc 方法在处理日期时间键列表时的行为
    def test_loc_getitem_listlike_of_datetimelike_keys(self, to_period):
        # GH#11497

        # 创建一个日期范围索引，从 "2011-01-01" 到 "2011-01-02"，频率为每日，命名为 "idx"
        idx = date_range("2011-01-01", "2011-01-02", freq="D", name="idx")
        # 如果参数 to_period 为 True，则将索引转换为周期索引
        if to_period:
            idx = idx.to_period("D")
        # 创建一个 Series 对象，包含值 [0.1, 0.2]，使用上述索引 idx，命名为 "s"
        ser = Series([0.1, 0.2], index=idx, name="s")

        # 创建一个包含两个 Timestamp 对象的键列表
        keys = [Timestamp("2011-01-01"), Timestamp("2011-01-02")]
        # 如果参数 to_period 为 True，则将键列表中的每个 Timestamp 对象转换为周期对象
        if to_period:
            keys = [x.to_period("D") for x in keys]
        # 使用 loc 方法根据键列表获取对应的值，并赋给 result
        result = ser.loc[keys]
        # 创建一个期望的 Series 对象 exp，包含值 [0.1, 0.2]，使用索引 idx，命名为 "s"
        exp = Series([0.1, 0.2], index=idx, name="s")
        # 如果 to_period 不为 True，则清除 exp 的索引频率信息
        if not to_period:
            exp.index = exp.index._with_freq(None)
        # 使用 assert_series_equal 检查 result 和 exp 是否相等，包括索引类型
        tm.assert_series_equal(result, exp, check_index_type=True)

        # 创建一个包含三个 Timestamp 对象的键列表
        keys = [
            Timestamp("2011-01-02"),
            Timestamp("2011-01-02"),
            Timestamp("2011-01-01"),
        ]
        # 如果参数 to_period 为 True，则将键列表中的每个 Timestamp 对象转换为周期对象
        if to_period:
            keys = [x.to_period("D") for x in keys]
        # 创建一个期望的 Series 对象 exp，包含值 [0.2, 0.2, 0.1]，使用索引 keys，命名为 "idx"，数据类型与 idx 相同
        exp = Series(
            [0.2, 0.2, 0.1], index=Index(keys, name="idx", dtype=idx.dtype), name="s"
        )
        # 使用 loc 方法根据键列表获取对应的值，并赋给 result
        result = ser.loc[keys]
        # 使用 assert_series_equal 检查 result 和 exp 是否相等，包括索引类型
        tm.assert_series_equal(result, exp, check_index_type=True)

        # 创建一个包含三个 Timestamp 对象的键列表
        keys = [
            Timestamp("2011-01-03"),
            Timestamp("2011-01-02"),
            Timestamp("2011-01-03"),
        ]
        # 如果参数 to_period 为 True，则将键列表中的每个 Timestamp 对象转换为周期对象
        if to_period:
            keys = [x.to_period("D") for x in keys]

        # 使用 pytest 的断言来检查在使用 loc 方法时是否引发 KeyError 异常，且异常消息包含 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[keys]

    # 测试函数，用于测试 loc 方法在处理具有命名索引的情况下的行为
    def test_loc_named_index(self):
        # GH 42790
        # 创建一个 DataFrame 对象 df，包含数据、命名索引和列名
        df = DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        # 从 df 中选择前两行，并将索引名命名为 "foo"，赋给 expected
        expected = df.iloc[:2]
        expected.index.name = "foo"
        # 使用 loc 方法根据具有命名索引的 Index 对象获取对应的 DataFrame 部分，并赋给 result
        result = df.loc[Index(["cobra", "viper"], name="foo")]
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "columns, column_key, expected_columns",
    [
        ([2011, 2012, 2013], [2011, 2012], [0, 1]),  # 定义测试参数：列名、列关键字和期望的列索引
        ([2011, 2012, "All"], [2011, 2012], [0, 1]),  # 定义测试参数：列名包含字符串和期望的列索引
        ([2011, 2012, "All"], [2011, "All"], [0, 2]),  # 定义测试参数：列名包含字符串和期望的列索引
    ],
)
def test_loc_getitem_label_list_integer_labels(columns, column_key, expected_columns):
    # gh-14836
    # 创建一个 DataFrame 对象，使用随机数填充，指定列和索引
    df = DataFrame(
        np.random.default_rng(2).random((3, 3)), columns=columns, index=list("ABC")
    )
    # 根据期望的列索引获取 DataFrame 的子集
    expected = df.iloc[:, expected_columns]
    # 使用 loc 方法获取指定行和列的子集
    result = df.loc[["A", "B", "C"], column_key]

    # 使用 pytest 框架的 assert_frame_equal 断言方法比较结果和期望值，检查列类型
    tm.assert_frame_equal(result, expected, check_column_type=True)


def test_loc_setitem_float_intindex():
    # GH 8720
    # 创建一个 DataFrame 对象，使用标准正态分布的随机数填充
    rand_data = np.random.default_rng(2).standard_normal((8, 4))
    result = DataFrame(rand_data)
    # 在新列（浮点数索引）中设置 NaN 值
    result.loc[:, 0.5] = np.nan
    # 期望的 DataFrame 对象，增加了一个新列，包含 NaN 值
    expected_data = np.hstack((rand_data, np.array([np.nan] * 8).reshape(8, 1)))
    expected = DataFrame(expected_data, columns=[0.0, 1.0, 2.0, 3.0, 0.5])
    tm.assert_frame_equal(result, expected)

    result = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    tm.assert_frame_equal(result, expected)


def test_loc_axis_1_slice():
    # GH 10586
    # 创建一个 DataFrame 对象，填充为全 1 的数组，指定行和多级列索引
    cols = [(yr, m) for yr in [2014, 2015] for m in [7, 8, 9, 10]]
    df = DataFrame(
        np.ones((10, 8)),
        index=tuple("ABCDEFGHIJ"),
        columns=MultiIndex.from_tuples(cols),
    )
    # 使用 loc(axis=1) 方法对列进行切片，选择特定范围内的列
    result = df.loc(axis=1)[(2014, 9) : (2015, 8)]
    # 期望的 DataFrame 对象，包含选择范围内的列
    expected = DataFrame(
        np.ones((10, 4)),
        index=tuple("ABCDEFGHIJ"),
        columns=MultiIndex.from_tuples([(2014, 9), (2014, 10), (2015, 7), (2015, 8)]),
    )
    tm.assert_frame_equal(result, expected)


def test_loc_set_dataframe_multiindex():
    # GH 14592
    # 创建一个 DataFrame 对象，所有元素均为字符 "a"，指定行和多级列索引
    expected = DataFrame(
        "a", index=range(2), columns=MultiIndex.from_product([range(2), range(2)])
    )
    result = expected.copy()
    # 使用 loc 方法设置特定位置的元素，目标位置与源位置相同
    result.loc[0, [(0, 1)]] = result.loc[0, [(0, 1)]]
    tm.assert_frame_equal(result, expected)


def test_loc_mixed_int_float():
    # GH#19456
    # 创建一个 Series 对象，使用对象索引类型的索引
    ser = Series(range(2), Index([1, 2.0], dtype=object))

    # 使用 loc 方法获取索引为 1 的元素
    result = ser.loc[1]
    assert result == 0


def test_loc_with_positional_slice_raises():
    # GH#31840
    # 创建一个 Series 对象，指定索引
    ser = Series(range(4), index=["A", "B", "C", "D"])

    # 使用 .loc 方法对位置切片进行访问，预期引发 TypeError 异常
    with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
        ser.loc[:3] = 2


def test_loc_slice_disallows_positional():
    # GH#16121, GH#24612, GH#31810
    # 创建一个日期范围索引的 DataFrame 对象，填充随机数
    dti = date_range("2016-01-01", periods=3)
    df = DataFrame(np.random.default_rng(2).random((3, 2)), index=dti)

    # 创建一个 Series 对象，选择 DataFrame 的第一列
    ser = df[0]

    msg = (
        "cannot do slice indexing on DatetimeIndex with these "
        r"indexers \[1\] of type int"
    )

    # 针对 DataFrame 和 Series 对象，使用 .loc 方法进行切片访问，预期引发 TypeError 异常
    for obj in [df, ser]:
        with pytest.raises(TypeError, match=msg):
            obj.loc[1:3]

        with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
            # GH#31840 强制执行不正确的行为
            obj.loc[1:3] = 1

    with pytest.raises(TypeError, match=msg):
        df.loc[1:3, 1]
    # 使用 pytest 的上下文管理器检测是否引发了 TypeError 异常，并匹配异常信息中是否包含特定字符串 "Slicing a positional slice with .loc"
    with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
        # 在数据框 df 上进行切片操作，但使用了 .loc 方法在位置切片上的不正确行为
        df.loc[1:3, 1] = 2
def test_loc_datetimelike_mismatched_dtypes():
    # GH#32650 dont mix and match datetime/timedelta/period dtypes

    # 创建一个 DataFrame，其中包含随机数据，列名为["a", "b", "c"]，索引为 DatetimeIndex
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 3)),
        columns=["a", "b", "c"],
        index=date_range("2012", freq="h", periods=5),
    )
    # 从 df 中选择非唯一的 DatetimeIndex 来创建一个新的 DataFrame
    df = df.iloc[[0, 2, 2, 3]].copy()

    # 获取 DataFrame 的索引
    dti = df.index
    # 使用索引的 asi8 属性创建 TimedeltaIndex，确保匹配整数值
    tdi = pd.TimedeltaIndex(dti.asi8)  # matching i8 values

    # 设置匹配的错误信息模式
    msg = r"None of \[TimedeltaIndex.* are in the \[index\]"
    # 测试是否会抛出 KeyError，并检查错误信息是否匹配
    with pytest.raises(KeyError, match=msg):
        df.loc[tdi]

    # 对列 "a" 进行索引，并测试是否会抛出 KeyError
    with pytest.raises(KeyError, match=msg):
        df["a"].loc[tdi]


def test_loc_with_period_index_indexer():
    # GH#4125
    # 创建一个 PeriodIndex，从 "2002-01" 到 "2003-12"，频率为每月
    idx = pd.period_range("2002-01", "2003-12", freq="M")
    # 创建一个 DataFrame，包含随机数据，行索引为 idx
    df = DataFrame(np.random.default_rng(2).standard_normal((24, 10)), index=idx)
    # 使用 loc 方法并传入 idx 来测试是否与 df 相等
    tm.assert_frame_equal(df, df.loc[idx])
    # 将 idx 转换为列表传入 loc 方法，并测试是否与 df 相等
    tm.assert_frame_equal(df, df.loc[list(idx)])
    # 同上，但传入的是一个列表的列表
    tm.assert_frame_equal(df, df.loc[list(idx)])
    # 使用切片索引 idx 的前 5 行，并测试是否与 df 相等
    tm.assert_frame_equal(df.iloc[0:5], df.loc[idx[0:5]])
    # 再次测试 df 是否与 loc 方法的结果相等
    tm.assert_frame_equal(df, df.loc[list(idx)])


def test_loc_setitem_multiindex_timestamp():
    # GH#13831
    # 创建一个包含随机数据的 DataFrame，行索引为日期范围从 "1/1/2000" 开始的 8 天
    vals = np.random.default_rng(2).standard_normal((8, 6))
    idx = date_range("1/1/2000", periods=8)
    cols = ["A", "B", "C", "D", "E", "F"]
    exp = DataFrame(vals, index=idx, columns=cols)
    # 将 exp 的某个索引位置的 ("A", "B") 值设为 NaN
    exp.loc[exp.index[1], ("A", "B")] = np.nan
    # 将 vals 的相应位置设为 NaN
    vals[1][0:2] = np.nan
    # 创建一个新的 DataFrame res，用 vals 替换 vals 中的 NaN 值
    res = DataFrame(vals, index=idx, columns=cols)
    # 测试 res 和 exp 是否相等
    tm.assert_frame_equal(res, exp)


def test_loc_getitem_multiindex_tuple_level():
    # GH#27591
    lev1 = ["a", "b", "c"]
    lev2 = [(0, 1), (1, 0)]
    lev3 = [0, 1]
    cols = MultiIndex.from_product([lev1, lev2, lev3], names=["x", "y", "z"])
    # 创建一个 DataFrame，所有元素初始化为 6，列使用 MultiIndex
    df = DataFrame(6, index=range(5), columns=cols)

    # 使用 loc 方法选择列 "lev1[0]", "lev2[0]", "lev3[0]"，并测试是否与期望值相等
    result = df.loc[:, (lev1[0], lev2[0], lev3[0])]

    # TODO: i think this actually should drop levels
    # 创建期望的 DataFrame expected，选择第一列
    expected = df.iloc[:, :1]
    # 测试 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 使用 xs 方法选择相同的索引，测试是否与 expected 相等
    alt = df.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=1)
    tm.assert_frame_equal(alt, expected)

    # 在 Series 上进行相同的操作
    ser = df.iloc[0]
    # 期望的 Series，选择第一个元素
    expected2 = ser.iloc[:1]

    # 使用 xs 方法在 Series 上选择相同的索引，测试是否与 expected2 相等
    alt2 = ser.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=0)
    tm.assert_series_equal(alt2, expected2)

    # 使用 loc 方法在 Series 上选择 lev1[0], lev2[0], lev3[0]，并断言结果是否等于 6
    result2 = ser.loc[lev1[0], lev2[0], lev3[0]]
    assert result2 == 6


def test_loc_getitem_nullable_index_with_duplicates():
    # GH#34497
    # 创建一个 DataFrame，数据包括一个带有重复值的 Int64 列
    df = DataFrame(
        data=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, np.nan, np.nan]]).T,
        columns=["a", "b", "c"],
        dtype="Int64",
    )
    # 将 "c" 列设置为索引，并检查索引的 dtype 是否为 "Int64"
    df2 = df.set_index("c")
    assert df2.index.dtype == "Int64"

    # 使用 loc 方法选择索引为 1 的行，测试结果是否与期望的 Series 相等
    res = df2.loc[1]
    expected = Series([1, 5], index=df2.columns, dtype="Int64", name=1)
    tm.assert_series_equal(res, expected)

    # 将索引转换为 object 类型，并使用 loc 方法选择索引为 1 的行，测试结果是否与期望的 Series 相等
    df2.index = df2.index.astype(object)
    res = df2.loc[1]
    # 使用测试工具函数 tm.assert_series_equal() 比较 res 和 expected 两个对象是否相等
    tm.assert_series_equal(res, expected)
@pytest.mark.parametrize("value", [300, np.uint16(300), np.int16(300)])
# 使用 pytest 的参数化装饰器，针对不同的数值类型进行测试
def test_loc_setitem_uint8_upcast(value):
    # GH#26049
    # 标记：GitHub 问题编号 26049

    df = DataFrame([1, 2, 3, 4], columns=["col1"], dtype="uint8")
    # 创建一个 DataFrame，包含一列数据，数据类型为 uint8

    with tm.assert_produces_warning(FutureWarning, match="item of incompatible dtype"):
        # 使用 tm.assert_produces_warning 检查是否产生 FutureWarning 警告，匹配警告消息 "item of incompatible dtype"
        df.loc[2, "col1"] = value  # 将无法保存在 uint8 中的值赋给 DataFrame 的某个位置

    if np_version_gt2 and isinstance(value, np.int16):
        # 如果 numpy 版本大于 2，并且 value 是 np.int16 类型
        # 注意，uint8 + int16 的结果类型是 int16
        # 在 numpy < 2 中，numpy 会检查值并确认其可以容纳在 uint16 中，因此结果类型是 uint16
        dtype = "int16"
    else:
        dtype = "uint16"
    
    expected = DataFrame([1, 2, 300, 4], columns=["col1"], dtype=dtype)
    # 创建预期的 DataFrame，包含修改后的预期结果数据
    tm.assert_frame_equal(df, expected)
    # 使用 tm.assert_frame_equal 检查 df 和 expected 是否相等


@pytest.mark.parametrize(
    "fill_val,exp_dtype",
    [
        (Timestamp("2022-01-06"), "datetime64[ns]"),
        (Timestamp("2022-01-07", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
    ],
)
# 使用 pytest 的参数化装饰器，针对不同的日期时间值和预期的数据类型进行测试
def test_loc_setitem_using_datetimelike_str_as_index(fill_val, exp_dtype):
    data = ["2022-01-02", "2022-01-03", "2022-01-04", fill_val.date()]
    # 创建日期时间数据列表，包括 fill_val 的日期部分
    index = DatetimeIndex(data, tz=fill_val.tz, dtype=exp_dtype)
    # 使用日期时间数据创建索引，包括时区信息和预期的数据类型
    df = DataFrame([10, 11, 12, 14], columns=["a"], index=index)
    # 创建一个 DataFrame，包含一列数据，使用上述索引作为行索引
    df.loc["2022-01-08", "a"] = 13
    # 使用未存在的日期时间字符串索引添加新行

    data.append("2022-01-08")
    expected_index = DatetimeIndex(data, dtype=exp_dtype)
    # 创建预期的索引，包括添加新行后的日期时间数据和预期的数据类型
    tm.assert_index_equal(df.index, expected_index, exact=True)
    # 使用 tm.assert_index_equal 检查 df 的索引和 expected_index 是否相等


def test_loc_set_int_dtype():
    # GH#23326
    # 标记：GitHub 问题编号 23326
    df = DataFrame([list("abc")])
    # 创建一个 DataFrame，包含一行字符数据
    df.loc[:, "col1"] = 5
    # 将整数值 5 赋给 DataFrame 的 "col1" 列

    expected = DataFrame({0: ["a"], 1: ["b"], 2: ["c"], "col1": [5]})
    # 创建预期的 DataFrame，包含修改后的预期结果数据
    tm.assert_frame_equal(df, expected)
    # 使用 tm.assert_frame_equal 检查 df 和 expected 是否相等


@pytest.mark.filterwarnings(r"ignore:Period with BDay freq is deprecated:FutureWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
# 忽略特定类型的警告消息
def test_loc_periodindex_3_levels():
    # GH#24091
    # 标记：GitHub 问题编号 24091
    p_index = PeriodIndex(
        ["20181101 1100", "20181101 1200", "20181102 1300", "20181102 1400"],
        name="datetime",
        freq="B",
    )
    # 创建一个 PeriodIndex，包含日期时间字符串列表，设置名称和频率为工作日
    mi_series = DataFrame(
        [["A", "B", 1.0], ["A", "C", 2.0], ["Z", "Q", 3.0], ["W", "F", 4.0]],
        index=p_index,
        columns=["ONE", "TWO", "VALUES"],
    )
    # 创建一个 DataFrame，使用上述 PeriodIndex 作为索引，包含多层索引和列名
    mi_series = mi_series.set_index(["ONE", "TWO"], append=True)["VALUES"]
    # 将 "ONE" 和 "TWO" 列设置为额外的索引层，并选择 "VALUES" 列作为结果

    assert mi_series.loc[(p_index[0], "A", "B")] == 1.0
    # 使用 loc 定位并断言特定索引元组的值是否等于 1.0


def test_loc_setitem_pyarrow_strings():
    # GH#52319
    # 标记：GitHub 问题编号 52319
    pytest.importorskip("pyarrow")
    # 导入 pyarrow 库，如果导入失败则跳过此测试
    df = DataFrame(
        {
            "strings": Series(["A", "B", "C"], dtype="string[pyarrow]"),
            "ids": Series([True, True, False]),
        }
    )
    # 创建一个 DataFrame，包含字符串列和布尔列，字符串列的数据类型为 pyarrow 字符串
    new_value = Series(["X", "Y"])
    # 创建一个新的 Series 对象，包含字符串数据

    df.loc[df.ids, "strings"] = new_value
    # 使用布尔条件选择并更新 "strings" 列的值为 new_value

    expected_df = DataFrame(
        {
            "strings": Series(["X", "Y", "C"], dtype="string[pyarrow]"),
            "ids": Series([True, True, False]),
        }
    )
    # 创建预期的 DataFrame，包含修改后的预期结果数据

    tm.assert_frame_equal(df, expected_df)
    # 使用 tm.assert_frame_equal 检查 df 和 expected_df 是否相等
    @pytest.mark.parametrize("val,expected", [(2**63 - 1, 3), (2**63, 4)])
    # 使用 pytest 的 parametrize 装饰器，定义测试参数化输入，验证测试函数的多组输入输出情况
    def test_loc_uint64(self, val, expected):
        # 查看 GitHub issue #19399
        ser = Series({2**63 - 1: 3, 2**63: 4})
        # 创建 Series 对象 ser，使用字典初始化，包含 2**63-1 和 2**63 作为索引，3 和 4 作为对应的值
        assert ser.loc[val] == expected
        # 断言 ser.loc[val] 的值等于 expected

    def test_loc_getitem(self, string_series, datetime_series):
        # 从 string_series 中选择索引为 [3, 4, 7] 的位置，进行 loc 操作
        inds = string_series.index[[3, 4, 7]]
        tm.assert_series_equal(string_series.loc[inds], string_series.reindex(inds))
        # 使用 tm.assert_series_equal 断言 loc 操作后的结果与 reindex 操作的结果一致

        # 从 datetime_series 中选择切片操作，验证 loc 与 truncate 方法的一致性
        d1, d2 = datetime_series.index[[5, 15]]
        result = datetime_series.loc[d1:d2]
        expected = datetime_series.truncate(d1, d2)
        tm.assert_series_equal(result, expected)

        # 使用布尔掩码进行 loc 操作，验证结果与使用掩码后的原始 series 一致
        mask = string_series > string_series.median()
        tm.assert_series_equal(string_series.loc[mask], string_series[mask])

        # 单独索引 datetime_series 的值，验证 loc 操作的正确性
        assert datetime_series.loc[d1] == datetime_series[d1]
        assert datetime_series.loc[d2] == datetime_series[d2]

    def test_loc_getitem_not_monotonic(self, datetime_series):
        # 选择 datetime_series 中索引为 [5, 15] 的位置
        d1, d2 = datetime_series.index[[5, 15]]

        # 创建 ts2，包含 datetime_series 选取每隔两个取值后乱序的结果
        ts2 = datetime_series[::2].iloc[[1, 2, 0]]

        # 使用 pytest 的 raises 方法验证在非单调序列中使用 loc[d1:d2] 会抛出 KeyError
        msg = r"Timestamp\('2000-01-10 00:00:00'\)"
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2]
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2] = 0

    def test_loc_getitem_setitem_integer_slice_keyerrors(self):
        # 创建一个带有索引的 Series 对象 ser，索引为 [0, 2, 4, ..., 18]
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )

        # 对 ser 进行切片并赋值为 0，验证切片和赋值操作的正确性
        cp = ser.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).all()

        cp = ser.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()

        # 使用 loc 和 reindex 验证 loc 操作后的结果与预期的 reindex 结果一致
        result = ser.iloc[2:6]
        result2 = ser.loc[3:11]
        expected = ser.reindex([4, 6, 8, 10])
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        # 对非单调序列进行 loc 操作，验证在索引不连续的情况下会抛出 KeyError
        s2 = ser.iloc[list(range(5)) + list(range(9, 4, -1))]
        with pytest.raises(KeyError, match=r"^3$"):
            s2.loc[3:11]
        with pytest.raises(KeyError, match=r"^3$"):
            s2.loc[3:11] = 0

    def test_loc_getitem_iterator(self, string_series):
        # 创建一个迭代器 idx，选取 string_series 的前 10 个索引位置
        idx = iter(string_series.index[:10])
        result = string_series.loc[idx]
        tm.assert_series_equal(result, string_series[:10])
        # 使用 tm.assert_series_equal 验证 loc 操作后的结果与 string_series 的切片结果一致

    def test_loc_setitem_boolean(self, string_series):
        # 创建一个布尔掩码 mask，选取 string_series 中大于中位数的部分
        mask = string_series > string_series.median()

        # 对 string_series 使用 loc 操作，并将满足 mask 的位置设置为 0
        result = string_series.copy()
        result.loc[mask] = 0
        expected = string_series
        expected[mask] = 0
        tm.assert_series_equal(result, expected)
        # 使用 tm.assert_series_equal 验证 loc 操作后的结果与预期的结果一致
    def test_loc_setitem_corner(self, string_series):
        # 获取指定索引位置的列表
        inds = list(string_series.index[[5, 8, 12]])
        # 在指定的索引位置设置值为5
        string_series.loc[inds] = 5
        # 准备一个错误消息
        msg = r"\['foo'\] not in index"
        # 使用 pytest 检查是否会引发 KeyError，并匹配预期的错误消息
        with pytest.raises(KeyError, match=msg):
            # 在指定的索引位置添加新值5，预期会引发 KeyError
            string_series.loc[inds + ["foo"]] = 5

    def test_basic_setitem_with_labels(self, datetime_series):
        # 获取指定索引位置的列表
        indices = datetime_series.index[[5, 10, 15]]

        # 复制原始的时间序列
        cp = datetime_series.copy()
        exp = datetime_series.copy()
        # 在指定的索引位置设置值为0
        cp[indices] = 0
        # 使用 .loc 方法在指定的索引位置设置值为0
        exp.loc[indices] = 0
        # 使用测试工具检查两个序列是否相等
        tm.assert_series_equal(cp, exp)

        cp = datetime_series.copy()
        exp = datetime_series.copy()
        # 在切片范围内的索引位置设置值为0
        cp[indices[0] : indices[2]] = 0
        # 使用 .loc 方法在切片范围内的索引位置设置值为0
        exp.loc[indices[0] : indices[2]] = 0
        # 使用测试工具检查两个序列是否相等
        tm.assert_series_equal(cp, exp)

    def test_loc_setitem_listlike_of_ints(self):
        # 创建一个具有整数索引的 Series 对象
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )
        # 准备一个整数索引列表
        inds = [0, 4, 6]
        # 准备一个整数索引的 NumPy 数组
        arr_inds = np.array([0, 4, 6])

        cp = ser.copy()
        exp = ser.copy()
        # 在整数索引位置设置值为0
        ser[inds] = 0
        # 使用 .loc 方法在整数索引位置设置值为0
        ser.loc[inds] = 0
        # 使用测试工具检查两个序列是否相等
        tm.assert_series_equal(cp, exp)

        cp = ser.copy()
        exp = ser.copy()
        # 在整数索引数组位置设置值为0
        ser[arr_inds] = 0
        # 使用 .loc 方法在整数索引数组位置设置值为0
        ser.loc[arr_inds] = 0
        # 使用测试工具检查两个序列是否相等
        tm.assert_series_equal(cp, exp)

        # 准备一个未找到的整数索引列表
        inds_notfound = [0, 4, 5, 6]
        # 准备一个未找到的整数索引的 NumPy 数组
        arr_inds_notfound = np.array([0, 4, 5, 6])
        # 准备一个错误消息
        msg = r"\[5\] not in index"
        # 使用 pytest 检查是否会引发 KeyError，并匹配预期的错误消息
        with pytest.raises(KeyError, match=msg):
            # 在未找到的整数索引位置设置值为0，预期会引发 KeyError
            ser[inds_notfound] = 0
        # 使用 pytest 检查是否会引发异常，并匹配预期的错误消息
        with pytest.raises(Exception, match=msg):
            # 在未找到的整数索引数组位置设置值为0，预期会引发异常
            ser[arr_inds_notfound] = 0

    def test_loc_setitem_dt64tz_values(self):
        # GH#12089
        # 创建一个带有时区的日期序列
        ser = Series(
            date_range("2011-01-01", periods=3, tz="US/Eastern"),
            index=["a", "b", "c"],
        )
        # 复制原始的序列
        s2 = ser.copy()
        # 准备一个预期的时间戳
        expected = Timestamp("2011-01-03", tz="US/Eastern")
        # 使用 .loc 方法在指定的索引位置设置值为预期的时间戳
        s2.loc["a"] = expected
        # 获取设置后的结果值
        result = s2.loc["a"]
        # 断言结果值与预期值相等
        assert result == expected

        s2 = ser.copy()
        # 使用 .iloc 方法在指定的整数位置设置值为预期的时间戳
        s2.iloc[0] = expected
        # 获取设置后的结果值
        result = s2.iloc[0]
        # 断言结果值与预期值相等
        assert result == expected

        s2 = ser.copy()
        # 直接通过索引名设置值为预期的时间戳
        s2["a"] = expected
        # 获取设置后的结果值
        result = s2["a"]
        # 断言结果值与预期值相等
        assert result == expected

    @pytest.mark.parametrize("array_fn", [np.array, pd.array, list, tuple])
    @pytest.mark.parametrize("size", [0, 4, 5, 6])
    def test_loc_iloc_setitem_with_listlike(self, size, array_fn):
        # GH37748
        # 测试将一个列表对象插入到大小为 N（这里为5）的 Series 中
        # 插入对象的大小为 0, N-1, N, N+1

        # 根据指定的 array_fn 创建一个数组对象
        arr = array_fn([0] * size)
        # 准备预期的 Series 对象，包含插入的数组对象
        expected = Series([arr, 0, 0, 0, 0], index=list("abcde"), dtype=object)

        # 创建一个初始值为0的 Series 对象，索引为 'abcde'
        ser = Series(0, index=list("abcde"), dtype=object)
        # 使用 .loc 方法在索引 'a' 的位置设置值为数组对象 arr
        ser.loc["a"] = arr
        # 使用测试工具检查两个序列是否相等
        tm.assert_series_equal(ser, expected)

        # 创建一个初始值为0的 Series 对象，索引为 'abcde'
        ser = Series(0, index=list("abcde"), dtype=object)
        # 使用 .iloc 方法在索引位置 0 的位置设置值为数组对象 arr
        ser.iloc[0] = arr
        # 使用测试工具检查两个序列是否相等
        tm.assert_series_equal(ser, expected)
    # 使用 pytest 的参数化装饰器，为 test_loc_series_getitem_too_many_dimensions 方法提供多个参数化的测试用例
    @pytest.mark.parametrize("indexer", [IndexSlice["A", :], ("A", slice(None))])
    def test_loc_series_getitem_too_many_dimensions(self, indexer):
        # GH#35349
        # 创建一个包含多级索引的 Series 对象，用于测试
        ser = Series(
            index=MultiIndex.from_tuples([("A", "0"), ("A", "1"), ("B", "0")]),
            data=[21, 22, 23],
        )
        # 定义错误消息，用于断言在索引时引发 IndexingError 异常
        msg = "Too many indexers"
        # 测试使用 .loc 进行索引时是否抛出 IndexingError 异常
        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :]

        # 测试使用 .loc 进行赋值时是否抛出 IndexingError 异常
        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :] = 1

    # 测试 Series 对象的 .loc 属性用于设置值
    def test_loc_setitem(self, string_series):
        # 获取索引的子集
        inds = string_series.index[[3, 4, 7]]

        # 复制 Series 对象
        result = string_series.copy()
        # 使用 .loc 根据索引 inds 设置值为 5
        result.loc[inds] = 5

        # 期望的结果
        expected = string_series.copy()
        expected.iloc[[3, 4, 7]] = 5
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 使用 .iloc 设置切片范围内的值为 10
        result.iloc[5:10] = 10
        expected[5:10] = 10
        # 再次断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 使用索引值范围设置切片的值
        d1, d2 = string_series.index[[5, 15]]
        result.loc[d1:d2] = 6
        expected[5:16] = 6  # 因为是包含关系，所以设置的范围是 [5, 16]
        # 最后再次断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 设置单个索引值的值
        string_series.loc[d1] = 4
        string_series.loc[d2] = 6
        # 断言设置成功
        assert string_series[d1] == 4
        assert string_series[d2] == 6

    # 使用 pytest 的参数化装饰器，为 test_loc_assign_dict_to_row 方法提供多个参数化的测试用例
    @pytest.mark.parametrize("dtype", ["object", "string"])
    def test_loc_assign_dict_to_row(self, dtype):
        # GH41044
        # 创建一个 DataFrame 对象，包含两列 A 和 B，用于测试
        df = DataFrame({"A": ["abc", "def"], "B": ["ghi", "jkl"]}, dtype=dtype)
        # 使用 .loc 设置第一行的值为字典 {"A": "newA", "B": "newB"}
        df.loc[0, :] = {"A": "newA", "B": "newB"}

        # 期望的结果 DataFrame
        expected = DataFrame({"A": ["newA", "def"], "B": ["newB", "jkl"]}, dtype=dtype)

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 测试 DataFrame 对象使用 .loc 设置多个条目的值为字典，包括 timedelta 类型
    def test_loc_setitem_dict_timedelta_multiple_set(self):
        # GH 16309
        # 创建一个空的 DataFrame 对象，列名为 ["time", "value"]
        result = DataFrame(columns=["time", "value"])
        # 使用 .loc 设置索引为 1 的行的值为字典 {"time": Timedelta(6, unit="s"), "value": "foo"}
        result.loc[1] = {"time": Timedelta(6, unit="s"), "value": "foo"}
        # 再次设置同一个索引的行的值为字典
        result.loc[1] = {"time": Timedelta(6, unit="s"), "value": "foo"}
        # 期望的结果 DataFrame
        expected = DataFrame(
            [[Timedelta(6, unit="s"), "foo"]], columns=["time", "value"], index=[1]
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试 DataFrame 对象使用 .loc 设置多个条目的值，包括新列
    def test_loc_set_multiple_items_in_multiple_new_columns(self):
        # GH 25594
        # 创建一个具有指定索引和列名的空 DataFrame 对象
        df = DataFrame(index=[1, 2], columns=["a"])
        # 使用 .loc 设置索引为 1 的行的 ["b", "c"] 列的值为 [6, 7]
        df.loc[1, ["b", "c"]] = [6, 7]

        # 期望的结果 DataFrame
        expected = DataFrame(
            {
                "a": Series([np.nan, np.nan], dtype="object"),
                "b": [6, np.nan],
                "c": [7, np.nan],
            },
            index=[1, 2],
        )

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 测试 Series 对象使用 .loc 进行索引
    def test_getitem_loc_str_periodindex(self):
        # GH#33964
        # 预期的警告消息
        msg = "Period with BDay freq is deprecated"
        # 使用 assert_produces_warning 上下文管理器来断言警告消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 创建一个 PeriodIndex 对象，用于测试
            index = pd.period_range(start="2000", periods=20, freq="B")
            series = Series(range(20), index=index)
            # 使用 .loc 根据字符串索引获取值，并断言其值为 9
            assert series.loc["2000-01-14"] == 9
    # 定义一个测试函数，测试处理非唯一掩码索引的情况
    def test_loc_nonunique_masked_index(self):
        # GH 57027，引用 GitHub 上的 issue 编号，用以跟踪问题来源
        ids = list(range(11))  # 创建一个包含 0 到 10 的列表作为索引标识符
        index = Index(ids * 1000, dtype="Int64")  # 创建一个索引对象，包含重复的标识符，每个重复 1000 次
        df = DataFrame({"val": np.arange(len(index), dtype=np.intp)}, index=index)
        # 创建一个数据帧 df，其中包含一个列 "val"，数据为从 0 到索引长度减一的整数，数据类型为 np.intp
        result = df.loc[ids]  # 使用给定的 ids 列表对数据帧进行 loc 操作，得到结果数据帧 result
        expected = DataFrame(
            {"val": index.argsort(kind="stable").astype(np.intp)},
            index=Index(np.array(ids).repeat(1000), dtype="Int64"),
        )
        # 创建期望的数据帧 expected，其中包含一个 "val" 列，值为按稳定排序后的索引值，数据类型为 np.intp
        # expected 数据帧的索引是一个重复 ids 1000 次的数组，数据类型为 "Int64"
        tm.assert_frame_equal(result, expected)
        # 使用测试框架中的 assert_frame_equal 函数比较 result 和 expected 数据帧，确保它们相等
```