# `D:\src\scipysrc\pandas\pandas\tests\frame\test_block_internals.py`

```
# 引入 datetime 和 timedelta 两个类
from datetime import (
    datetime,
    timedelta,
)
# 引入 itertools 模块，用于迭代工具函数
import itertools

# 引入 numpy 并使用 np 别名
import numpy as np
# 引入 pytest 模块，用于编写测试用例
import pytest

# 引入 pandas 库，并从中引入多个类和函数
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    date_range,
    option_context,
)
# 引入 pandas 内部测试模块
import pandas._testing as tm
# 从 pandas 内核中引入 NumpyBlock 类
from pandas.core.internals.blocks import NumpyBlock

# 用于存放需要使用 BlockManager 内部数据结构的方法集合的类
class TestDataFrameBlockInternals:
    def test_setitem_invalidates_datetime_index_freq(self):
        # 修改 datetime64tz 类型的列会导致底层 DatetimeIndex 的 `freq` 属性无效
        dti = date_range("20130101", periods=3, tz="US/Eastern")
        ts = dti[1]

        # 创建包含 datetime64tz 列的 DataFrame
        df = DataFrame({"B": dti})
        # 断言 DataFrame 列 'B' 的 _values 的 freq 属性为 None
        assert df["B"]._values.freq is None

        # 在 DataFrame 中使用 pd.NaT（Not a Time）替换某个位置的值
        df.iloc[1, 0] = pd.NaT
        # 再次断言 DataFrame 列 'B' 的 _values 的 freq 属性为 None
        assert df["B"]._values.freq is None

        # 检查 DatetimeIndex 是否未被原地修改
        assert dti.freq == "D"
        assert dti[1] == ts

    def test_cast_internals(self, float_frame):
        # 测试类型转换相关的内部操作
        msg = "Passing a BlockManager to DataFrame"
        
        # 使用断言检查是否会产生 DeprecationWarning 警告信息
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            # 将 float_frame 的 BlockManager 转换为整型 DataFrame
            casted = DataFrame(float_frame._mgr, dtype=int)
        expected = DataFrame(float_frame._series, dtype=int)
        tm.assert_frame_equal(casted, expected)

        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            # 将 float_frame 的 BlockManager 转换为 np.int32 类型的 DataFrame
            casted = DataFrame(float_frame._mgr, dtype=np.int32)
        expected = DataFrame(float_frame._series, dtype=np.int32)
        tm.assert_frame_equal(casted, expected)

    def test_consolidate(self, float_frame):
        # 测试 DataFrame 的 conslidate 方法
        float_frame["E"] = 7.0
        # 执行 DataFrame 的内部 conslidate 操作
        consolidated = float_frame._consolidate()
        # 断言 conslidate 后的 BlockManager 中 blocks 的数量为 1
        assert len(consolidated._mgr.blocks) == 1

        # 确保进行了复制操作，而不是原地修改
        recons = consolidated._consolidate()
        assert recons is not consolidated
        tm.assert_frame_equal(recons, consolidated)

        float_frame["F"] = 8.0
        assert len(float_frame._mgr.blocks) == 3

        # 返回值应为 None，表示原地 conslidate 操作成功执行
        return_value = float_frame._consolidate_inplace()
        assert return_value is None
        assert len(float_frame._mgr.blocks) == 1

    def test_consolidate_inplace(self, float_frame):
        # 触发原地 conslidate 操作的测试
        for letter in range(ord("A"), ord("Z")):
            float_frame[chr(letter)] = chr(letter)

    def test_modify_values(self, float_frame):
        # 测试修改 DataFrame 值的方法
        with pytest.raises(ValueError, match="read-only"):
            # 尝试修改 float_frame 的 values 的第 5 个元素为 5，预期会引发 ValueError 异常
            float_frame.values[5] = 5
        # 断言第 5 个元素的值没有改变
        assert (float_frame.values[5] != 5).all()

    def test_boolean_set_uncons(self, float_frame):
        # 测试使用布尔索引设置 DataFrame 值并取消 conslidate 的影响
        float_frame["E"] = 7.0

        expected = float_frame.values.copy()
        expected[expected > 1] = 2

        # 使用布尔索引将大于 1 的元素设置为 2
        float_frame[float_frame > 1] = 2
        tm.assert_almost_equal(expected, float_frame.values)
    def test_constructor_with_convert(self):
        # 这个测试主要是测试 lib.maybe_convert_objects 的功能
        # #2845

        # 创建一个包含一个最大整数的 DataFrame
        df = DataFrame({"A": [2**63 - 1]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 int64
        expected = Series(np.asarray([2**63 - 1], np.int64), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含一个大整数的 DataFrame
        df = DataFrame({"A": [2**63]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 uint64
        expected = Series(np.asarray([2**63], np.uint64), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含日期时间和布尔值的 DataFrame
        df = DataFrame({"A": [datetime(2005, 1, 1), True]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 object
        expected = Series(
            np.asarray([datetime(2005, 1, 1), True], np.object_), name="A"
        )
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含 None 和整数的 DataFrame
        df = DataFrame({"A": [None, 1]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 float64
        expected = Series(np.asarray([np.nan, 1], np.float64), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含浮点数的 DataFrame
        df = DataFrame({"A": [1.0, 2]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 float64
        expected = Series(np.asarray([1.0, 2], np.float64), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含复数和整数的 DataFrame
        df = DataFrame({"A": [1.0 + 2.0j, 3]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 complex128
        expected = Series(np.asarray([1.0 + 2.0j, 3], np.complex128), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含复数和浮点数的 DataFrame
        df = DataFrame({"A": [1.0 + 2.0j, 3.0]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 complex128
        expected = Series(np.asarray([1.0 + 2.0j, 3.0], np.complex128), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含复数和布尔值的 DataFrame
        df = DataFrame({"A": [1.0 + 2.0j, True]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 object
        expected = Series(np.asarray([1.0 + 2.0j, True], np.object_), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含浮点数和 None 的 DataFrame
        df = DataFrame({"A": [1.0, None]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 float64
        expected = Series(np.asarray([1.0, np.nan], np.float64), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含复数和 None 的 DataFrame
        df = DataFrame({"A": [1.0 + 2.0j, None]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 complex128
        expected = Series(np.asarray([1.0 + 2.0j, np.nan], np.complex128), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含浮点数、整数、布尔值和 None 的 DataFrame
        df = DataFrame({"A": [2.0, 1, True, None]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 object
        expected = Series(np.asarray([2.0, 1, True, None], np.object_), name="A")
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含浮点数、整数、日期时间和 None 的 DataFrame
        df = DataFrame({"A": [2.0, 1, datetime(2006, 1, 1), None]})
        # 从 DataFrame 中取出 "A" 列数据
        result = df["A"]
        # 创建预期的 Series，数据类型为 object
        expected = Series(
            np.asarray([2.0, 1, datetime(2006, 1, 1), None], np.object_), name="A"
        )
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
    # 定义测试函数，测试混合类型情况下的构建

    # 创建包含日期时间和空值的二维列表
    data = [
        [datetime(2001, 1, 5), np.nan, datetime(2001, 1, 2)],
        [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 1)],
    ]
    # 根据数据创建数据帧（DataFrame）
    df = DataFrame(data)

    # 检查数据帧的数据类型（dtypes）
    result = df.dtypes
    # 期望的数据类型为包含一个 datetime64[us] 类型的 Series 对象
    expected = Series({"datetime64[us]": 3})

    # 混合类型帧操作

    # 将当前时间添加到浮点数与字符串混合的数据帧中的 'datetime' 列
    float_string_frame["datetime"] = datetime.now()
    # 将 timedelta 添加到 'timedelta' 列，间隔为一天一秒钟
    float_string_frame["timedelta"] = timedelta(days=1, seconds=1)
    # 断言 'datetime' 列的数据类型为 "M8[us]"
    assert float_string_frame["datetime"].dtype == "M8[us]"
    # 断言 'timedelta' 列的数据类型为 "m8[us]"
    assert float_string_frame["timedelta"].dtype == "m8[us]"

    # 检查数据帧的数据类型（dtypes）
    result = float_string_frame.dtypes
    # 期望的数据类型为一个包含多种类型的 Series 对象
    expected = Series(
        [np.dtype("float64")] * 4
        + [
            np.dtype("object") if not using_infer_string else "string",
            np.dtype("datetime64[us]"),
            np.dtype("timedelta64[us]"),
        ],
        index=list("ABCD") + ["foo", "datetime", "timedelta"],
    )
    # 使用测试工具库（tm）断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 测试带转换的构建函数

    # 创建一个 numpy 数组，其元素为非 ns 纳秒级 timedelta64 类型
    arr = np.array([1, 2, 3], dtype="timedelta64[s]")
    # 创建一个索引为 [0, 1, 2] 的数据帧（DataFrame）
    df = DataFrame(index=range(3))
    # 将数组 arr 赋值给数据帧 df 的 'A' 列
    df["A"] = arr
    # 期望的结果数据帧
    expected = DataFrame(
        {"A": pd.timedelta_range("00:00:01", periods=3, freq="s")}, index=range(3)
    )
    # 使用测试工具库（tm）断言 df["A"].to_numpy() 与 arr 数组是否相等
    tm.assert_numpy_array_equal(df["A"].to_numpy(), arr)

    # 创建期望的数据帧
    expected = DataFrame(
        {
            "dt1": Timestamp("20130101"),
            "dt2": date_range("20130101", periods=3).astype("M8[s]"),
            # 'dt3' : date_range('20130101 00:00:01',periods=3,freq='s'),
            # FIXME: don't leave commented-out
        },
        index=range(3),
    )
    # 断言数据帧 expected 的 'dt1' 列数据类型为 "M8[s]"
    assert expected.dtypes["dt1"] == "M8[s]"
    # 断言数据帧 expected 的 'dt2' 列数据类型为 "M8[s]"

    # 创建一个索引为 [0, 1, 2] 的数据帧（DataFrame）
    df = DataFrame(index=range(3))
    # 在数据帧 df 中添加 'dt1' 列，其值为 np.datetime64("2013-01-01")
    df["dt1"] = np.datetime64("2013-01-01")
    # 在数据帧 df 中添加 'dt2' 列，其值为包含日期字符串的 numpy 数组
    df["dt2"] = np.array(
        ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
    )

    # df['dt3'] = np.array(['2013-01-01 00:00:01','2013-01-01
    # 00:00:02','2013-01-01 00:00:03'],dtype='datetime64[s]')
    # FIXME: don't leave commented-out

    # 使用测试工具库（tm）断言两个数据帧 df 和 expected 是否相等
    tm.assert_frame_equal(df, expected)
    # 测试构造函数对复合数据类型的处理
    def test_constructor_compound_dtypes(self):
        # GH 5191
        # 对于复合数据类型应该抛出 NotImplementedError

        # 定义一个函数 f，接受 dtype 参数
        def f(dtype):
            # 创建一个包含相同元素的列表，每个元素为 (datetime 对象, "aa", 20)
            data = list(itertools.repeat((datetime(2001, 1, 1), "aa", 20), 9))
            # 使用 DataFrame 构造函数创建 DataFrame 对象，指定列名和 dtype
            return DataFrame(data=data, columns=["A", "B", "C"], dtype=dtype)

        # 准备错误信息
        msg = "compound dtypes are not implemented in the DataFrame constructor"
        # 断言调用 f 函数时抛出 NotImplementedError 异常，并检查异常信息是否匹配预期
        with pytest.raises(NotImplementedError, match=msg):
            f([("A", "datetime64[h]"), ("B", "str"), ("C", "int32")])

        # 在 2.0 之前版本，以下代码可能有效（尽管结果可能出乎意料）
        # 检查使用 "int64" 和 "float64" 作为 dtype 时是否抛出 TypeError 异常
        with pytest.raises(TypeError, match="argument must be"):
            f("int64")
        with pytest.raises(TypeError, match="argument must be"):
            f("float64")

        # 10822
        # 准备错误信息，用于匹配 ValueError 异常的消息
        msg = "^Unknown datetime string format, unable to parse: aa, at position 0$"
        # 断言使用 "M8[ns]" 作为 dtype 时抛出 ValueError 异常，并检查异常信息是否匹配预期
        with pytest.raises(ValueError, match=msg):
            f("M8[ns]")

    # 测试 pickle 功能对包含浮点数和字符串的 DataFrame 的处理
    def test_pickle_float_string_frame(self, float_string_frame):
        # 序列化和反序列化测试，确保数据一致性
        unpickled = tm.round_trip_pickle(float_string_frame)
        # 断言反序列化后的 DataFrame 和原始的 DataFrame 在内容上完全一致
        tm.assert_frame_equal(float_string_frame, unpickled)

        # buglet
        # 访问 float_string_frame 对象的 _mgr.ndim 属性

    # 测试 pickle 功能对空 DataFrame 的处理
    def test_pickle_empty(self):
        # 创建一个空的 DataFrame 对象
        empty_frame = DataFrame()
        # 序列化和反序列化测试空 DataFrame
        unpickled = tm.round_trip_pickle(empty_frame)
        # 调用 repr 函数打印反序列化后的对象的字符串表示形式

    # 测试 pickle 功能对包含时区信息的 DataFrame 的处理
    def test_pickle_empty_tz_frame(self, timezone_frame):
        # 序列化和反序列化测试带时区信息的 DataFrame
        unpickled = tm.round_trip_pickle(timezone_frame)
        # 断言反序列化后的 DataFrame 和原始的 DataFrame 在内容上完全一致
        tm.assert_frame_equal(timezone_frame, unpickled)
    # 测试处理 datetime64 类型数据合并的方法
    def test_consolidate_datetime64(self):
        # 创建包含时间序列和测量值的数据框
        df = DataFrame(
            {
                "starting": pd.to_datetime(
                    [
                        "2012-06-21 00:00",
                        "2012-06-23 07:00",
                        "2012-06-23 16:30",
                        "2012-06-25 08:00",
                        "2012-06-26 12:00",
                    ]
                ),
                "ending": pd.to_datetime(
                    [
                        "2012-06-23 07:00",
                        "2012-06-23 16:30",
                        "2012-06-25 08:00",
                        "2012-06-26 12:00",
                        "2012-06-27 08:00",
                    ]
                ),
                "measure": [77, 65, 77, 0, 77],
            }
        )

        # 提取并设置起始时间序列
        ser_starting = df.starting
        ser_starting.index = ser_starting.values
        # 将起始时间序列本地化为 "US/Eastern" 时区
        ser_starting = ser_starting.tz_localize("US/Eastern")
        # 转换起始时间序列时区为 UTC
        ser_starting = ser_starting.tz_convert("UTC")
        # 设置起始时间序列的索引名称为 "starting"
        ser_starting.index.name = "starting"

        # 提取并设置结束时间序列
        ser_ending = df.ending
        ser_ending.index = ser_ending.values
        # 将结束时间序列本地化为 "US/Eastern" 时区
        ser_ending = ser_ending.tz_localize("US/Eastern")
        # 转换结束时间序列时区为 UTC
        ser_ending = ser_ending.tz_convert("UTC")
        # 设置结束时间序列的索引名称为 "ending"
        ser_ending.index.name = "ending"

        # 更新数据框中的起始和结束时间列
        df.starting = ser_starting.index
        df.ending = ser_ending.index

        # 断言数据框中的起始和结束时间列索引与提取的序列相同
        tm.assert_index_equal(pd.DatetimeIndex(df.starting), ser_starting.index)
        tm.assert_index_equal(pd.DatetimeIndex(df.ending), ser_ending.index)

    # 测试混合类型检测的方法
    def test_is_mixed_type(self, float_frame, float_string_frame):
        # 断言 float_frame 不是混合类型
        assert not float_frame._is_mixed_type
        # 断言 float_string_frame 是混合类型
        assert float_string_frame._is_mixed_type

    # 测试修复缓存的 Series Bug 473
    def test_stale_cached_series_bug_473(self):
        # 使用 option_context 设置 chained_assignment 参数为 None
        with option_context("chained_assignment", None):
            # 创建一个随机数据框 Y
            Y = DataFrame(
                np.random.default_rng(2).random((4, 4)),
                index=("a", "b", "c", "d"),
                columns=("e", "f", "g", "h"),
            )
            # 打印 Y 的表现形式
            repr(Y)
            # 将 Y 的 "e" 列转换为对象类型
            Y["e"] = Y["e"].astype("object")
            # 使用 chained_assignment_error 上下文，尝试修改 Y 的 "g" 列中的 "c" 行为 NaN
            with tm.raises_chained_assignment_error():
                Y["g"]["c"] = np.nan
            # 再次打印 Y 的表现形式
            repr(Y)
            # 计算 Y 的总和
            Y.sum()
            # 计算 Y 的 "g" 列总和
            Y["g"].sum()
            # 断言 Y 的 "g" 列中 "c" 行不是 NaN
            assert not pd.isna(Y["g"]["c"])
    # 测试处理奇怪的列损坏问题
    def test_strange_column_corruption_issue(self, performance_warning):
        # TODO(wesm): 不清楚这与内部事务的确切关系如何
        # 创建一个具有两行的空 DataFrame
        df = DataFrame(index=[0, 1])
        # 在第一列中插入 NaN 值
        df[0] = np.nan
        # 初始化一个空字典用于跟踪已处理的列
        wasCol = {}

        # 使用断言来验证是否生成了特定性能警告
        with tm.assert_produces_warning(
            performance_warning, raise_on_extra_warnings=False
        ):
            # 遍历 DataFrame 的索引和值
            for i, dt in enumerate(df.index):
                # 遍历从 100 到 199 的列范围
                for col in range(100, 200):
                    # 如果列不在 wasCol 字典中，则将其添加并设置为 NaN
                    if col not in wasCol:
                        wasCol[col] = 1
                        df[col] = np.nan
                    # 在指定的行和列位置设置值为 i
                    df.loc[dt, col] = i

        # 设置一个列标签
        myid = 100

        # 使用 pd.isna 检查指定列中的 NaN 值数量
        first = len(df.loc[pd.isna(df[myid]), [myid]])
        # 再次检查指定列中的 NaN 值数量
        second = len(df.loc[pd.isna(df[myid]), [myid]])
        # 断言两次检查结果都为 0
        assert first == second == 0

    # 测试构造函数不接受 Pandas 数组
    def test_constructor_no_pandas_array(self):
        # 确保在 Series 内部不允许使用 NumpyExtensionArray
        # 参见 https://github.com/pandas-dev/pandas/issues/23995 了解更多信息
        # 创建一个包含 Series 的数组
        arr = Series([1, 2, 3]).array
        # 构建一个 DataFrame，其中一个列使用上述数组
        result = DataFrame({"A": arr})
        # 创建一个预期的 DataFrame
        expected = DataFrame({"A": [1, 2, 3]})
        # 使用断言验证两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
        # 断言结果 DataFrame 的第一个块是 NumpyBlock 类型
        assert isinstance(result._mgr.blocks[0], NumpyBlock)
        # 断言结果 DataFrame 的第一个块是数值型的
        assert result._mgr.blocks[0].is_numeric

    # 测试添加包含 Pandas 数组的列
    def test_add_column_with_pandas_array(self):
        # GH 26390
        # 创建一个 DataFrame，包含两列和一个额外的 Pandas 数组列
        df = DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
        # 在 DataFrame 中添加一个使用 NumpyExtensionArray 的列
        df["c"] = pd.arrays.NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
        # 创建一个预期的 DataFrame，包含相同的列设置
        df2 = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": ["a", "b", "c", "d"],
                "c": pd.arrays.NumpyExtensionArray(
                    np.array([1, 2, None, 3], dtype=object)
                ),
            }
        )
        # 断言新列的数据块类型是 NumpyBlock
        assert type(df["c"]._mgr.blocks[0]) == NumpyBlock
        # 断言新列的数据块是对象类型
        assert df["c"]._mgr.blocks[0].is_object
        # 对比两个 DataFrame 是否相等
        tm.assert_frame_equal(df, df2)
# 设置一个测试函数，验证在原地更新时设置有效的块值
def test_update_inplace_sets_valid_block_values():
    # 创建一个包含分类数据类型的DataFrame，列'a'包含值为1, 2和None的Series对象
    df = DataFrame({"a": Series([1, 2, None], dtype="category")})

    # 在原地更新单个列
    with tm.raises_chained_assignment_error():
        df["a"].fillna(1, inplace=True)

    # 检查没有将Series对象放入任何块的值中
    assert isinstance(df._mgr.blocks[0].values, Categorical)


# 测试非合并项缓存的获取操作
def test_nonconsolidated_item_cache_take():
    # 创建一个包含对象数据类型列的非合并DataFrame
    df = DataFrame()
    df["col1"] = Series(["a"], dtype=object)
    df["col2"] = Series([0], dtype=object)

    # 访问列（项缓存）
    df["col1"] == "A"
    
    # 执行take操作
    # （之前的问题是这里合并了但没有重置项缓存，
    # 导致缓存无效，使得.at操作无法正常工作）
    df[df["col2"] == 0]

    # 现在设置值应该更新实际的DataFrame
    df.at[0, "col1"] = "A"

    # 预期结果DataFrame
    expected = DataFrame({"col1": ["A"], "col2": [0]}, dtype=object)
    # 断言DataFrame相等
    tm.assert_frame_equal(df, expected)
    # 断言DataFrame中指定位置的值为"A"
    assert df.at[0, "col1"] == "A"
```