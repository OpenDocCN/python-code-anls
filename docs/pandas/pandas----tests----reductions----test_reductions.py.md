# `D:\src\scipysrc\pandas\pandas\tests\reductions\test_reductions.py`

```
# 导入所需模块和类
from datetime import (
    datetime,        # 导入 datetime 类
    timedelta,       # 导入 timedelta 类
)
from decimal import Decimal  # 导入 Decimal 类

import numpy as np  # 导入 numpy 库并重命名为 np
import pytest  # 导入 pytest 库

import pandas as pd  # 导入 pandas 库并重命名为 pd
from pandas import (
    Categorical,           # 导入 Categorical 类
    DataFrame,             # 导入 DataFrame 类
    DatetimeIndex,         # 导入 DatetimeIndex 类
    Index,                 # 导入 Index 类
    NaT,                   # 导入 NaT 对象
    Period,                # 导入 Period 类
    PeriodIndex,           # 导入 PeriodIndex 类
    RangeIndex,            # 导入 RangeIndex 类
    Series,                # 导入 Series 类
    Timedelta,             # 导入 Timedelta 类
    TimedeltaIndex,        # 导入 TimedeltaIndex 类
    Timestamp,             # 导入 Timestamp 类
    date_range,            # 导入 date_range 函数
    isna,                  # 导入 isna 函数
    period_range,          # 导入 period_range 函数
    timedelta_range,       # 导入 timedelta_range 函数
    to_timedelta,          # 导入 to_timedelta 函数
)
import pandas._testing as tm  # 导入 pandas 测试模块作为 tm
from pandas.core import nanops  # 导入 pandas 核心模块中的 nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics  # 导入 ArrowStringArrayNumpySemantics 类


def get_objs():
    # 创建多个不同类型的 Index 对象并存储在列表中
    indexes = [
        Index([True, False] * 5, name="a"),  # 布尔类型的 Index 对象
        Index(np.arange(10), dtype=np.int64, name="a"),  # 整数类型的 Index 对象
        Index(np.arange(10), dtype=np.float64, name="a"),  # 浮点数类型的 Index 对象
        DatetimeIndex(date_range("2020-01-01", periods=10), name="a"),  # 日期时间类型的 DatetimeIndex 对象
        DatetimeIndex(date_range("2020-01-01", periods=10), name="a").tz_localize(
            tz="US/Eastern"
        ),  # 本地化时区后的日期时间类型的 DatetimeIndex 对象
        PeriodIndex(period_range("2020-01-01", periods=10, freq="D"), name="a"),  # 周期类型的 PeriodIndex 对象
        Index([str(i) for i in range(10)], name="a"),  # 字符串类型的 Index 对象
    ]

    # 创建一个包含随机数据的 numpy 数组
    arr = np.random.default_rng(2).standard_normal(10)
    # 根据不同的 Index 对象创建 Series 对象，并存储在列表中
    series = [Series(arr, index=idx, name="a") for idx in indexes]

    # 将 Index 对象和 Series 对象合并成一个列表并返回
    objs = indexes + series
    return objs


class TestReductions:
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("obj", get_objs())
    def test_ops(self, opname, obj):
        # 调用 getattr 函数获取对象的最大值或最小值
        result = getattr(obj, opname)()
        if not isinstance(obj, PeriodIndex):
            if isinstance(obj.values, ArrowStringArrayNumpySemantics):
                # 如果是 ArrowStringArrayNumpySemantics 对象，则调用 numpy 数组的相应方法
                expected = getattr(np.array(obj.values), opname)()
            else:
                # 否则，调用对象的 values 属性的相应方法
                expected = getattr(obj.values, opname)()
        else:
            # 对于 PeriodIndex 对象，调用 asi8 属性的相应方法生成 Period 对象
            expected = Period(ordinal=getattr(obj.asi8, opname)(), freq=obj.freq)

        if getattr(obj, "tz", None) is not None:
            # 如果对象有时区信息，则将期望结果转换为纳秒精度的整数
            expected = expected.astype("M8[ns]").astype("int64")
            # 断言实际结果与期望结果相等
            assert result._value == expected
        else:
            # 否则，直接断言实际结果与期望结果相等
            assert result == expected

    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize(
        "dtype, val",
        [
            ("object", 2.0),                 # 对象类型的参数
            ("float64", 2.0),                # 浮点数类型的参数
            ("datetime64[ns]", datetime(2011, 11, 1)),  # 日期时间类型的参数
            ("Int64", 2),                    # Int64 类型的参数
            ("boolean", True),               # 布尔类型的参数
        ],
    )
    # 定义测试方法，用于测试特定操作（最小值或最大值）在不同数据类型和输入情况下的行为
    def test_nanminmax(self, opname, dtype, val, index_or_series):
        # GH#7261
        # 从参数中获取待测试的数据结构类别
        klass = index_or_series

        # 定义检查结果中缺失值的函数
        def check_missing(res):
            # 如果数据类型为日期时间类型，则检查结果是否为 NaT（Not a Time）
            if dtype == "datetime64[ns]":
                return res is NaT
            # 如果数据类型为 Int64 或 boolean，则检查结果是否为 pd.NA（Pandas的缺失值）
            elif dtype in ["Int64", "boolean"]:
                return res is pd.NA
            # 否则，使用 pandas 的 isna 函数检查结果是否为缺失值
            else:
                return isna(res)

        # 创建一个包含一个 None 值的对象实例
        obj = klass([None], dtype=dtype)
        # 断言调用指定操作后的结果中是否存在缺失值
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))

        # 创建一个空对象实例
        obj = klass([], dtype=dtype)
        # 断言调用指定操作后的结果中是否存在缺失值
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))

        # 对于数据类型为对象类型的情况，直接返回，因为通用测试只对空对象或全为 NaN 的情况有效
        if dtype == "object":
            return

        # 创建包含一个 None 和一个指定值的对象实例
        obj = klass([None, val], dtype=dtype)
        # 断言调用指定操作后的结果是否与指定值相等
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

        # 创建包含两个 None 和一个指定值的对象实例
        obj = klass([None, val, None], dtype=dtype)
        # 断言调用指定操作后的结果是否与指定值相等
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

    # 使用参数化测试框架定义测试方法，测试带有 NaN 值的最大值和最小值操作
    @pytest.mark.parametrize("opname", ["max", "min"])
    def test_nanargminmax(self, opname, index_or_series):
        # GH#7261
        # 从参数中获取待测试的数据结构类别
        klass = index_or_series
        # 根据类别选择正确的操作名称
        arg_op = "arg" + opname if klass is Index else "idx" + opname

        # 创建一个包含 NaT 和指定日期时间的对象实例
        obj = klass([NaT, datetime(2011, 11, 1)])
        # 断言调用指定操作后的结果是否符合预期
        assert getattr(obj, arg_op)() == 1

        # 使用 pytest 的异常断言，确保在跳过 NaN 值时会引发 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            getattr(obj, arg_op)(skipna=False)

        # 创建一个包含 NaT、指定日期时间和 NaT 的对象实例
        obj = klass([NaT, datetime(2011, 11, 1), NaT])
        # 检查针对非单调日期时间索引的路径
        assert getattr(obj, arg_op)() == 1
        with pytest.raises(ValueError, match="Encountered an NA value"):
            getattr(obj, arg_op)(skipna=False)

    # 使用参数化测试框架定义测试方法，测试空对象的最大值和最小值操作
    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "datetime64[ns, UTC]"])
    def test_nanops_empty_object(self, opname, index_or_series, dtype):
        # 从参数中获取待测试的数据结构类别
        klass = index_or_series
        # 根据类别选择正确的操作名称
        arg_op = "arg" + opname if klass is Index else "idx" + opname

        # 创建一个空对象实例，并指定数据类型
        obj = klass([], dtype=dtype)

        # 断言调用指定操作后的结果是否为 NaT
        assert getattr(obj, opname)() is NaT
        assert getattr(obj, opname)(skipna=False) is NaT

        # 使用 pytest 的异常断言，确保在空序列时会引发 ValueError 异常
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)()
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)(skipna=False)
    # 定义测试方法，用于测试 Index 对象的 argmin 和 argmax 方法
    def test_argminmax(self):
        # 创建一个 Index 对象，包含整数数组 [0, 1, 2, 3, 4]
        obj = Index(np.arange(5, dtype="int64"))
        # 断言 Index 对象中最小值的索引为 0
        assert obj.argmin() == 0
        # 断言 Index 对象中最大值的索引为 4
        assert obj.argmax() == 4

        # 创建一个 Index 对象，包含 [NaN, 1, NaN, 2]
        obj = Index([np.nan, 1, np.nan, 2])
        # 断言 Index 对象中最小值的索引为 1
        assert obj.argmin() == 1
        # 断言 Index 对象中最大值的索引为 3
        assert obj.argmax() == 3
        # 使用 pytest 验证当 skipna=False 时，argmin 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        # 使用 pytest 验证当 skipna=False 时，argmax 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

        # 创建一个 Index 对象，包含 [NaN]
        obj = Index([np.nan])
        # 使用 pytest 验证当所有值为 NaN 时，argmin 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmin()
        # 使用 pytest 验证当所有值为 NaN 时，argmax 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmax()
        # 使用 pytest 验证当 skipna=False 且值为 NaN 时，argmin 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        # 使用 pytest 验证当 skipna=False 且值为 NaN 时，argmax 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

        # 创建一个 Index 对象，包含 [NaT, datetime(2011, 11, 1), datetime(2011, 11, 2), NaT]
        obj = Index([NaT, datetime(2011, 11, 1), datetime(2011, 11, 2), NaT])
        # 断言 Index 对象中最小值的索引为 1
        assert obj.argmin() == 1
        # 断言 Index 对象中最大值的索引为 2
        assert obj.argmax() == 2
        # 使用 pytest 验证当 skipna=False 且值为 NaT 时，argmin 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        # 使用 pytest 验证当 skipna=False 且值为 NaT 时，argmax 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

        # 创建一个 Index 对象，包含 [NaT]
        obj = Index([NaT])
        # 使用 pytest 验证当所有值为 NaT 时，argmin 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmin()
        # 使用 pytest 验证当所有值为 NaT 时，argmax 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmax()
        # 使用 pytest 验证当 skipna=False 且值为 NaT 时，argmin 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        # 使用 pytest 验证当 skipna=False 且值为 NaT 时，argmax 方法抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

    # 使用 pytest.mark.parametrize 进行参数化测试，测试 DataFrame 对象的最大和最小值方法
    @pytest.mark.parametrize("op, expected_col", [["max", "a"], ["min", "b"]])
    def test_same_tz_min_max_axis_1(self, op, expected_col):
        # GH 10390
        # 创建一个带有时区信息的 DataFrame 对象，包含列 'a' 和 'b'
        df = DataFrame(
            date_range("2016-01-01 00:00:00", periods=3, tz="UTC"), columns=["a"]
        )
        # 在 DataFrame 中添加一列 'b'，其值为列 'a' 减去 3600 秒后的结果
        df["b"] = df.a.subtract(Timedelta(seconds=3600))
        # 调用 DataFrame 对象的 op 操作（max 或 min），指定 axis=1 进行操作
        result = getattr(df, op)(axis=1)
        # 期望结果为 DataFrame 中的列 expected_col，并去除其名称
        expected = df[expected_col].rename(None)
        # 使用 tm.assert_series_equal 验证结果与期望结果是否相等
        tm.assert_series_equal(result, expected)

    # 使用 pytest.mark.parametrize 进行参数化测试，测试 numpy 的 maximum 和 minimum 方法
    @pytest.mark.parametrize("func", ["maximum", "minimum"])
    def test_numpy_reduction_with_tz_aware_dtype(self, tz_aware_fixture, func):
        # GH 15552
        # 获取时区感知的 fixture
        tz = tz_aware_fixture
        # 创建一个包含单个日期的 Series 对象，并进行时区本地化
        arg = pd.to_datetime(["2019"]).tz_localize(tz)
        # 期望结果为一个包含 arg 的 Series 对象
        expected = Series(arg)
        # 调用 numpy 的 func 方法（maximum 或 minimum），对两个相同的 Series 进行操作
        result = getattr(np, func)(expected, expected)
        # 使用 tm.assert_series_equal 验证结果与期望结果是否相等
        tm.assert_series_equal(result, expected)

    # 测试 DataFrame 对象中包含 NaN、整数和 timedelta 类型数据时的 sum 方法
    def test_nan_int_timedelta_sum(self):
        # GH 27185
        # 创建一个 DataFrame 对象，包含 'A' 列和 'B' 列
        df = DataFrame(
            {
                "A": Series([1, 2, NaT], dtype="timedelta64[ns]"),
                "B": Series([1, 2, np.nan], dtype="Int64"),
            }
        )
        # 期望结果为包含键 'A' 和 'B' 的 Series 对象，其值分别为 Timedelta(3) 和 3
        expected = Series({"A": Timedelta(3), "B": 3})
        # 调用 DataFrame 对象的 sum 方法，计算各列的和
        result = df.sum()
        # 使用 tm.assert_series_equal 验证结果与期望结果是否相等
        tm.assert_series_equal(result, expected)
class TestIndexReductions:
    # Note: the name TestIndexReductions indicates these tests
    #  were moved from a Index-specific test file, _not_ that these tests are
    #  intended long-term to be Index-specific

    @pytest.mark.parametrize(
        "start,stop,step",
        [
            (0, 400, 3),       # 参数化测试：起始值0，终止值400，步长3
            (500, 0, -6),      # 参数化测试：起始值500，终止值0，步长-6
            (-(10**6), 10**6, 4),  # 参数化测试：起始值-1000000，终止值1000000，步长4
            (10**6, -(10**6), -4),  # 参数化测试：起始值1000000，终止值-1000000，步长-4
            (0, 10, 20),       # 参数化测试：起始值0，终止值10，步长20
        ],
    )
    def test_max_min_range(self, start, stop, step):
        # GH#17607
        # 创建一个 RangeIndex 对象，以提供给测试的起始、终止和步长参数
        idx = RangeIndex(start, stop, step)
        # 期望结果是 RangeIndex 对象内部值的最大值
        expected = idx._values.max()
        # 计算 RangeIndex 对象的最大值
        result = idx.max()
        # 断言计算结果与期望值相等
        assert result == expected

        # skipna 应该是无关紧要的，因为 RangeIndex 不应该包含 NA 值
        # 计算最大值时不考虑 NA 值，结果应该与期望值相同
        result2 = idx.max(skipna=False)
        assert result2 == expected

        # 期望结果是 RangeIndex 对象内部值的最小值
        expected = idx._values.min()
        # 计算 RangeIndex 对象的最小值
        result = idx.min()
        # 断言计算结果与期望值相等
        assert result == expected

        # skipna 应该是无关紧要的，因为 RangeIndex 不应该包含 NA 值
        # 计算最小值时不考虑 NA 值，结果应该与期望值相同
        result2 = idx.min(skipna=False)
        assert result2 == expected

        # empty
        # 创建一个反向步长的 RangeIndex 对象
        idx = RangeIndex(start, stop, -step)
        # 断言这个 RangeIndex 对象的最大值应该是 NA
        assert isna(idx.max())
        # 断言这个 RangeIndex 对象的最小值应该是 NA
        assert isna(idx.min())

    def test_minmax_timedelta64(self):
        # monotonic
        # 创建一个 TimedeltaIndex 对象，包含 ["1 days", "2 days", "3 days"]，应该是单调递增的
        idx1 = TimedeltaIndex(["1 days", "2 days", "3 days"])
        assert idx1.is_monotonic_increasing

        # non-monotonic
        # 创建一个 TimedeltaIndex 对象，包含 ["1 days", np.nan, "3 days", "NaT"]，应该不是单调递增的
        idx2 = TimedeltaIndex(["1 days", np.nan, "3 days", "NaT"])
        assert not idx2.is_monotonic_increasing

        for idx in [idx1, idx2]:
            # 断言 TimedeltaIndex 对象的最小值是 "1 days"
            assert idx.min() == Timedelta("1 days")
            # 断言 TimedeltaIndex 对象的最大值是 "3 days"
            assert idx.max() == Timedelta("3 days")
            # 断言 TimedeltaIndex 对象的最小值的索引是 0
            assert idx.argmin() == 0
            # 断言 TimedeltaIndex 对象的最大值的索引是 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_timedelta_empty_or_na(self, op):
        # Return NaT
        # 创建一个空的 TimedeltaIndex 对象
        obj = TimedeltaIndex([])
        # 断言调用 min 或 max 方法时返回 NaT
        assert getattr(obj, op)() is NaT

        # 创建一个包含 NaT 的 TimedeltaIndex 对象
        obj = TimedeltaIndex([NaT])
        # 断言调用 min 或 max 方法时返回 NaT
        assert getattr(obj, op)() is NaT

        # 创建一个全是 NaT 的 TimedeltaIndex 对象
        obj = TimedeltaIndex([NaT, NaT, NaT])
        # 断言调用 min 或 max 方法时返回 NaT
        assert getattr(obj, op)() is NaT

    def test_numpy_minmax_timedelta64(self):
        # 创建一个 timedelta 对象数组 td，从 "16815 days" 到 "16820 days"，频率为每天一次
        td = timedelta_range("16815 days", "16820 days", freq="D")

        # 断言 td 数组的最小值是 "16815 days"
        assert np.min(td) == Timedelta("16815 days")
        # 断言 td 数组的最大值是 "16820 days"
        assert np.max(td) == Timedelta("16820 days")

        # 尝试在 np.min 和 np.max 方法中使用不支持的 'out' 参数，应该引发 ValueError 异常
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(td, out=0)

        # 断言 td 数组的最小值的索引是 0
        assert np.argmin(td) == 0
        # 断言 td 数组的最大值的索引是 5
        assert np.argmax(td) == 5

        # 尝试在 np.argmin 和 np.argmax 方法中使用不支持的 'out' 参数，应该引发 ValueError 异常
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(td, out=0)
    def test_timedelta_ops(self):
        # GH#4984
        # 确保操作返回 Timedelta 类型的对象

        # 创建一个包含日期时间的 Series，每个元素都是从 "20130101" 开始，每个元素之间相差 i*i 秒的时间间隔
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )

        # 计算时间差（每个元素与前一个元素的时间差）
        td = s.diff()

        # 计算时间差的均值
        result = td.mean()
        expected = to_timedelta(timedelta(seconds=9))
        assert result == expected

        # 将时间差转换为 DataFrame，并计算其均值
        result = td.to_frame().mean()
        assert result[0] == expected

        # 计算时间差的分位数（这里是第10%的分位数）
        result = td.quantile(0.1)
        expected = Timedelta(np.timedelta64(2600, "ms"))
        assert result == expected

        # 计算时间差的中位数
        result = td.median()
        expected = to_timedelta("00:00:09")
        assert result == expected

        # 将时间差转换为 DataFrame，并计算其中位数
        result = td.to_frame().median()
        assert result[0] == expected

        # 计算时间差的总和
        result = td.sum()
        expected = to_timedelta("00:01:21")
        assert result == expected

        # 将时间差转换为 DataFrame，并计算其总和
        result = td.to_frame().sum()
        assert result[0] == expected

        # 计算时间差的标准差
        result = td.std()
        expected = to_timedelta(Series(td.dropna().values).std())
        assert result == expected

        # 将时间差转换为 DataFrame，并计算其标准差
        result = td.to_frame().std()
        assert result[0] == expected

        # GH#10040
        # 确保 median() 正确处理 NaT（不是一个时间的情况）
        
        # 创建一个包含时间戳的 Series，计算其时间差，并计算时间差的中位数
        s = Series([Timestamp("2015-02-03"), Timestamp("2015-02-07")])
        assert s.diff().median() == timedelta(days=4)

        # 创建一个包含多个时间戳的 Series，计算其时间差，并计算时间差的中位数
        s = Series(
            [Timestamp("2015-02-03"), Timestamp("2015-02-07"), Timestamp("2015-02-15")]
        )
        assert s.diff().median() == timedelta(days=6)

    @pytest.mark.parametrize("opname", ["skew", "kurt", "sem", "prod", "var"])
    def test_invalid_td64_reductions(self, opname):
        # 测试对于 timedelta64 类型的对象，不支持的操作是否会引发 TypeError 异常
        
        # 创建一个包含日期时间的 Series，每个元素都是从 "20130101" 开始，每个元素之间相差 i*i 秒的时间间隔
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )
        
        # 计算时间差（每个元素与前一个元素的时间差）
        td = s.diff()

        # 构造用于匹配错误消息的正则表达式模式
        msg = "|".join(
            [
                f"reduction operation '{opname}' not allowed for this dtype",
                rf"cannot perform {opname} with type timedelta64\[ns\]",
                f"does not support operation '{opname}'",
            ]
        )

        # 断言调用不支持的操作时会抛出 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            getattr(td, opname)()

        # 断言将时间差转换为 DataFrame 后，调用不支持的操作时会抛出 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            getattr(td.to_frame(), opname)(numeric_only=False)

    def test_minmax_tz(self, tz_naive_fixture):
        # 测试带有时区信息的最小值和最大值的计算

        # 获取时区信息
        tz = tz_naive_fixture

        # 创建一个带有时区信息的日期时间索引，确保索引是单调递增的
        idx1 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz=tz)
        assert idx1.is_monotonic_increasing

        # 创建一个带有时区信息的日期时间索引，确保索引不是单调递增的
        idx2 = DatetimeIndex(
            ["2011-01-01", NaT, "2011-01-03", "2011-01-02", NaT], tz=tz
        )
        assert not idx2.is_monotonic_increasing

        # 对两个索引分别进行如下断言：最小值、最大值、最小值的索引位置、最大值的索引位置
        for idx in [idx1, idx2]:
            assert idx.min() == Timestamp("2011-01-01", tz=tz)
            assert idx.max() == Timestamp("2011-01-03", tz=tz)
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_nat_datetime64(self, op):
        # 测试对空的DatetimeIndex返回NaT
        obj = DatetimeIndex([])
        assert isna(getattr(obj, op)())

        # 测试对只含有NaT的DatetimeIndex返回NaT
        obj = DatetimeIndex([NaT])
        assert isna(getattr(obj, op)())

        # 测试对多个NaT的DatetimeIndex返回NaT
        obj = DatetimeIndex([NaT, NaT, NaT])
        assert isna(getattr(obj, op)())

    def test_numpy_minmax_integer(self):
        # GH#26125：测试numpy中对整数Index的最大最小值计算

        # 创建一个Index对象
        idx = Index([1, 2, 3])

        # 计算预期的最大值并与numpy的结果比较
        expected = idx.values.max()
        result = np.max(idx)
        assert result == expected

        # 计算预期的最小值并与numpy的结果比较
        expected = idx.values.min()
        result = np.min(idx)
        assert result == expected

        # 测试带有'out'参数时的错误处理
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

        # 计算预期的最大索引并与numpy的结果比较
        expected = idx.values.argmax()
        result = np.argmax(idx)
        assert result == expected

        # 计算预期的最小索引并与numpy的结果比较
        expected = idx.values.argmin()
        result = np.argmin(idx)
        assert result == expected

        # 再次测试带有'out'参数时的错误处理
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(idx, out=0)

    def test_numpy_minmax_range(self):
        # GH#26125：测试numpy中对RangeIndex的最大最小值计算

        # 创建一个RangeIndex对象
        idx = RangeIndex(0, 10, 3)

        # 计算预期的最大值并与numpy的结果比较
        result = np.max(idx)
        assert result == 9

        # 计算预期的最小值并与numpy的结果比较
        result = np.min(idx)
        assert result == 0

        # 测试带有'out'参数时的错误处理
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

        # 不需要再次测试argmax/argmin的兼容性，因为实现与基本整数索引相同

    def test_numpy_minmax_datetime64(self):
        # 测试numpy中对DatetimeIndex的最大最小值计算

        # 创建一个日期范围对象
        dr = date_range(start="2016-01-15", end="2016-01-20")

        # 断言计算的最小日期等于预期结果
        assert np.min(dr) == Timestamp("2016-01-15 00:00:00")
        # 断言计算的最大日期等于预期结果
        assert np.max(dr) == Timestamp("2016-01-20 00:00:00")

        # 测试带有'out'参数时的错误处理
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.max(dr, out=0)

        # 断言计算的最小日期索引等于预期结果
        assert np.argmin(dr) == 0
        # 断言计算的最大日期索引等于预期结果
        assert np.argmax(dr) == 5

        # 再次测试带有'out'参数时的错误处理
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.argmax(dr, out=0)
    def test_minmax_period(self):
        # 创建一个周期性索引 idx1，包含 NaT、"2011-01-01"、"2011-01-02"、"2011-01-03" 四个日期，频率为每天
        idx1 = PeriodIndex([NaT, "2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        # 断言 idx1 不是单调递增的
        assert not idx1.is_monotonic_increasing
        # 断言 idx1 从索引 1 开始是单调递增的
        assert idx1[1:].is_monotonic_increasing

        # 创建一个周期性索引 idx2，包含 "2011-01-01"、NaT、"2011-01-03"、"2011-01-02"、NaT 五个日期，频率为每天
        idx2 = PeriodIndex(
            ["2011-01-01", NaT, "2011-01-03", "2011-01-02", NaT], freq="D"
        )
        # 断言 idx2 不是单调递增的
        assert not idx2.is_monotonic_increasing

        # 对于 idx1 和 idx2，断言最小值和最大值的日期符合预期
        for idx in [idx1, idx2]:
            assert idx.min() == Period("2011-01-01", freq="D")
            assert idx.max() == Period("2011-01-03", freq="D")
        # 断言 idx1 的最小值索引是 1
        assert idx1.argmin() == 1
        # 断言 idx2 的最小值索引是 0
        assert idx2.argmin() == 0
        # 断言 idx1 的最大值索引是 3
        assert idx1.argmax() == 3
        # 断言 idx2 的最大值索引是 2
        assert idx2.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    @pytest.mark.parametrize("data", [[], [NaT], [NaT, NaT, NaT]])
    def test_minmax_period_empty_nat(self, op, data):
        # 创建一个周期性索引 obj，数据为 data，频率为每月，用于测试空数据或包含 NaT 的情况
        obj = PeriodIndex(data, freq="M")
        # 获取 op（min 或 max）操作后的结果，预期结果是 NaT
        result = getattr(obj, op)()
        assert result is NaT

    def test_numpy_minmax_period(self):
        # 创建一个日期范围 pr，从 "2016-01-15" 到 "2016-01-20"
        pr = period_range(start="2016-01-15", end="2016-01-20")

        # 断言 pr 的最小值是 Period("2016-01-15", freq="D")
        assert np.min(pr) == Period("2016-01-15", freq="D")
        # 断言 pr 的最大值是 Period("2016-01-20", freq="D")
        assert np.max(pr) == Period("2016-01-20", freq="D")

        # 预期出现错误消息 "the 'out' parameter is not supported"，测试 np.min 和 np.max 的 out 参数不支持情况
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(pr, out=0)

        # 断言 np.argmin(pr) 的结果是 0
        assert np.argmin(pr) == 0
        # 断言 np.argmax(pr) 的结果是 5
        assert np.argmax(pr) == 5

        # 再次测试 np.argmin 和 np.argmax 的 out 参数不支持情况
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(pr, out=0)

    def test_min_max_categorical(self):
        # 创建一个非有序的分类索引 ci，数据为 "aabbca"，分类为 "cab"，用于测试非有序情况下的 min 和 max 操作
        ci = pd.CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)
        # 预期出现 TypeError，提示非有序分类无法执行 min 操作
        msg = (
            r"Categorical is not ordered for operation min\n"
            r"you can use .as_ordered\(\) to change the Categorical to an ordered one\n"
        )
        with pytest.raises(TypeError, match=msg):
            ci.min()
        # 预期出现 TypeError，提示非有序分类无法执行 max 操作
        msg = (
            r"Categorical is not ordered for operation max\n"
            r"you can use .as_ordered\(\) to change the Categorical to an ordered one\n"
        )
        with pytest.raises(TypeError, match=msg):
            ci.max()

        # 创建一个有序的分类索引 ci，数据为 "aabbca"，分类为 "cab"，用于测试有序情况下的 min 和 max 操作
        ci = pd.CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=True)
        # 断言 ci 的最小值是 "c"
        assert ci.min() == "c"
        # 断言 ci 的最大值是 "b"
        assert ci.max() == "b"
# 定义一个测试类 TestSeriesReductions，用于测试 Series 对象的各种约简操作
class TestSeriesReductions:
    # 注意：TestSeriesReductions 的命名表明这些测试被移动自一个特定于系列的测试文件，
    # 并非长期用于系列特定的测试

    # 测试对无穷大的求和操作
    def test_sum_inf(self):
        # 创建一个包含 10 个标准正态随机数的 Series 对象
        s = Series(np.random.default_rng(2).standard_normal(10))
        # 复制 Series 对象 s
        s2 = s.copy()

        # 将索引为 5 到 7 的元素设置为无穷大
        s[5:8] = np.inf
        # 将索引为 5 到 7 的元素设置为 NaN
        s2[5:8] = np.nan

        # 断言 s 的总和是否为无穷大
        assert np.isinf(s.sum())

        # 创建一个 100x100 的随机浮点数数组，并将第三列设置为无穷大
        arr = np.random.default_rng(2).standard_normal((100, 100)).astype("f4")
        arr[:, 2] = np.inf

        # 对数组 arr 按行求和，返回结果中是否全部为无穷大
        res = nanops.nansum(arr, axis=1)
        assert np.isinf(res).all()

    # 参数化测试：针对空或可空数据类型的操作一致性
    @pytest.mark.parametrize(
        "dtype", ["float64", "Float32", "Int64", "boolean", "object"]
    )
    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("method, unit", [("sum", 0.0), ("prod", 1.0)])
    @pytest.mark.parametrize("method", ["mean", "var"])
    @pytest.mark.parametrize("dtype", ["Float64", "Int64", "boolean"])
    def test_ops_consistency_on_empty_nullable(self, method, dtype):
        # GH#34814
        # 空或全为 NA 时对可空数据类型的一致性

        # 空 Series
        eser = Series([], dtype=dtype)
        # 对空 Series 调用指定方法，断言结果为 NA
        result = getattr(eser, method)()
        assert result is pd.NA

        # 全为 NA 的 Series
        nser = Series([np.nan], dtype=dtype)
        # 对全为 NA 的 Series 调用指定方法，断言结果为 NA
        result = getattr(nser, method)()
        assert result is pd.NA

    # 参数化测试：针对空数据的操作一致性
    @pytest.mark.parametrize("method", ["mean", "median", "std", "var"])
    def test_ops_consistency_on_empty(self, method):
        # GH#7869
        # 空数据时的操作一致性

        # 浮点数
        result = getattr(Series(dtype=float), method)()
        # 断言结果是否为缺失值
        assert isna(result)

        # timedelta64[ns]
        tdser = Series([], dtype="m8[ns]")
        if method == "var":
            msg = "|".join(
                [
                    "operation 'var' not allowed",
                    r"cannot perform var with type timedelta64\[ns\]",
                    "does not support operation 'var'",
                ]
            )
            # 对 timedelta64[ns] 类型调用 var 方法时，断言引发 TypeError，匹配指定的消息
            with pytest.raises(TypeError, match=msg):
                getattr(tdser, method)()
        else:
            # 对 timedelta64[ns] 类型调用指定方法，断言结果为 NaT
            result = getattr(tdser, method)()
            assert result is NaT

    # 测试 nansum 的 buglet
    def test_nansum_buglet(self):
        # 创建一个包含 NaN 的 Series 对象
        ser = Series([1.0, np.nan], index=[0, 1])
        # 对 Series 对象进行 nansum 操作，断言结果近似等于 1
        result = np.nansum(ser)
        tm.assert_almost_equal(result, 1)

    # 参数化测试：针对整数类型的操作一致性
    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("dtype", ["int32", "int64"])
    def test_sum_overflow_int(self, use_bottleneck, dtype):
        # 使用指定的设置上下文，设置是否使用瓶颈优化
        with pd.option_context("use_bottleneck", use_bottleneck):
            # GH#6915
            # 在较小的整数类型上可能发生溢出
            # 创建一个包含5000000个元素的NumPy数组，数据类型为dtype
            v = np.arange(5000000, dtype=dtype)
            # 创建一个Series对象，使用上述数组作为数据
            s = Series(v)

            # 计算Series的总和，不跳过NaN值
            result = s.sum(skipna=False)
            # 断言：将结果转换为整数后，应与数组v的总和相等（使用int64类型）
            assert int(result) == v.sum(dtype="int64")
            # 计算Series的最小值，不跳过NaN值
            result = s.min(skipna=False)
            # 断言：将结果转换为整数后，应为0
            assert int(result) == 0
            # 计算Series的最大值，不跳过NaN值
            result = s.max(skipna=False)
            # 断言：将结果转换为整数后，应与数组v的最后一个元素相等
            assert int(result) == v[-1]

    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_sum_overflow_float(self, use_bottleneck, dtype):
        # 使用指定的设置上下文，设置是否使用瓶颈优化
        with pd.option_context("use_bottleneck", use_bottleneck):
            # 创建一个包含5000000个元素的NumPy数组，数据类型为dtype
            v = np.arange(5000000, dtype=dtype)
            # 创建一个Series对象，使用上述数组作为数据
            s = Series(v)

            # 计算Series的总和，不跳过NaN值
            result = s.sum(skipna=False)
            # 断言：结果应与数组v的总和相等，使用相同的数据类型dtype
            assert result == v.sum(dtype=dtype)
            # 计算Series的最小值，不跳过NaN值
            result = s.min(skipna=False)
            # 断言：使用np.allclose检查结果是否接近0.0
            assert np.allclose(float(result), 0.0)
            # 计算Series的最大值，不跳过NaN值
            result = s.max(skipna=False)
            # 断言：使用np.allclose检查结果是否接近数组v的最后一个元素
            assert np.allclose(float(result), v[-1])

    def test_mean_masked_overflow(self):
        # GH#48378
        # 定义一个大数值
        val = 100_000_000_000_000_000
        # 创建一个包含100个元素的NumPy数组，所有元素均为val
        n_elements = 100
        na = np.array([val] * n_elements)
        # 创建一个Series对象，使用上述数组作为数据，数据类型为"Int64"
        ser = Series([val] * n_elements, dtype="Int64")

        # 使用NumPy计算数组na的均值
        result_numpy = np.mean(na)
        # 计算Series对象ser的均值
        result_masked = ser.mean()
        # 断言：Series对象的均值应与NumPy数组的均值相等
        assert result_masked - result_numpy == 0
        # 断言：Series对象的均值应等于1e17
        assert result_masked == 1e17

    @pytest.mark.parametrize("ddof, exp", [(1, 2.5), (0, 2.0)])
    def test_var_masked_array(self, ddof, exp):
        # GH#48379
        # 创建一个包含整数1到5的Series对象，数据类型为"Int64"
        ser = Series([1, 2, 3, 4, 5], dtype="Int64")
        # 创建一个包含整数1到5的Series对象，数据类型为"int64"
        ser_numpy_dtype = Series([1, 2, 3, 4, 5], dtype="int64")
        # 计算Series对象ser的方差，使用自由度参数ddof
        result = ser.var(ddof=ddof)
        # 计算Series对象ser_numpy_dtype的方差，使用自由度参数ddof
        result_numpy_dtype = ser_numpy_dtype.var(ddof=ddof)
        # 断言：Series对象ser的方差应与ser_numpy_dtype的方差相等
        assert result == result_numpy_dtype
        # 断言：Series对象ser的方差应等于期望值exp
        assert result == exp

    @pytest.mark.parametrize("dtype", ("m8[ns]", "M8[ns]", "M8[ns, UTC]"))
    def test_empty_timeseries_reductions_return_nat(self, dtype, skipna):
        # covers GH#11245
        # 断言：空时间序列的最小值应为NaT（不适用跳过NaN值设置）
        assert Series([], dtype=dtype).min(skipna=skipna) is NaT
        # 断言：空时间序列的最大值应为NaT（不适用跳过NaN值设置）
        assert Series([], dtype=dtype).max(skipna=skipna) is NaT

    def test_numpy_argmin(self):
        # See GH#16830
        # 创建一个包含整数1到10的NumPy数组
        data = np.arange(1, 11)

        # 创建一个Series对象，使用上述数组作为数据和索引
        s = Series(data, index=data)
        # 使用NumPy计算Series对象s的最小值的索引
        result = np.argmin(s)

        # 计算预期的最小值索引
        expected = np.argmin(data)
        # 断言：Series对象s的最小值索引应与预期相等
        assert result == expected

        # 使用Series对象的argmin方法计算最小值的索引
        result = s.argmin()

        # 断言：Series对象s的最小值索引应与预期相等
        assert result == expected

        # 检查使用'out'参数时是否引发了ValueError异常
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmin(s, out=data)
    # 定义一个测试方法，用于测试 numpy 的 argmax 函数
    def test_numpy_argmax(self):
        # 设置数据为从 1 到 10 的数组
        data = np.arange(1, 11)

        # 创建一个 Pandas 的 Series 对象，使用上面的数据作为值和索引
        ser = Series(data, index=data)
        
        # 使用 numpy 的 argmax 函数计算 Series 中最大值的索引
        result = np.argmax(ser)
        
        # 使用 numpy 的 argmax 函数计算原始数据中的最大值索引
        expected = np.argmax(data)
        
        # 断言结果与预期相符
        assert result == expected

        # 调用 Series 对象自带的 argmax 方法
        result = ser.argmax()

        # 再次断言结果与预期相符
        assert result == expected

        # 准备错误消息字符串
        msg = "the 'out' parameter is not supported"
        
        # 使用 pytest 的 raises 方法检测是否抛出 ValueError 异常，异常消息需匹配指定的字符串
        with pytest.raises(ValueError, match=msg):
            np.argmax(ser, out=data)

    # 定义一个测试方法，用于测试在 datetime64 索引上的 idxmin 方法
    def test_idxmin_dt64index(self, unit):
        # 准备 DatetimeIndex，其中包含字符串 "NaT" 和一个有效日期
        dti = DatetimeIndex(["NaT", "2015-02-08", "NaT"]).as_unit(unit)
        
        # 创建一个 Pandas 的 Series 对象，包含数值和 NaN 值，使用上面的 DatetimeIndex 作为索引
        ser = Series([1.0, 2.0, np.nan], index=dti)
        
        # 使用 pytest 的 raises 方法检测是否抛出 ValueError 异常，异常消息需匹配指定的字符串
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 调用 idxmin 方法，禁用跳过 NaN 值的功能
            ser.idxmin(skipna=False)
        
        # 同上，检测是否抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 调用 idxmax 方法，禁用跳过 NaN 值的功能
            ser.idxmax(skipna=False)

        # 将 Series 对象转换为 DataFrame 对象
        df = ser.to_frame()
        
        # 同上，检测是否抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 在 DataFrame 上调用 idxmin 方法，禁用跳过 NaN 值的功能
            df.idxmin(skipna=False)
        
        # 同上，检测是否抛出 ValueError 异常
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 在 DataFrame 上调用 idxmax 方法，禁用跳过 NaN 值的功能
            df.idxmax(skipna=False)

    # 定义一个测试方法，用于测试 Series 对象的 idxmin 方法
    def test_idxmin(self):
        # 添加一些 NaN 值到 Series 对象中
        string_series = Series(range(20), dtype=np.float64, name="series")
        string_series[5:15] = np.nan

        # 断言跳过 NaN 值后，索引最小值对应的值等于 Series 的最小值
        assert string_series[string_series.idxmin()] == string_series.min()
        
        # 使用 pytest 的 raises 方法检测是否抛出 ValueError 异常，异常消息需匹配指定的字符串
        with pytest.raises(ValueError, match="Encountered an NA value"):
            # 调用 idxmin 方法，禁用跳过 NaN 值的功能
            string_series.idxmin(skipna=False)

        # 创建一个没有 NaN 值的新 Series 对象
        nona = string_series.dropna()
        
        # 断言在没有 NaN 值的 Series 中，索引最小值对应的值等于 Series 的最小值
        assert nona[nona.idxmin()] == nona.min()
        
        # 使用列表的索引方法，检查索引最小值在 Series 中的位置是否等于值最小的位置
        assert nona.index.values.tolist().index(nona.idxmin()) == nona.values.argmin()

        # 创建一个所有值为 NaN 的 Series 对象
        allna = string_series * np.nan
        
        # 准备错误消息字符串
        msg = "Encountered all NA values"
        
        # 使用 pytest 的 raises 方法检测是否抛出 ValueError 异常，异常消息需匹配指定的字符串
        with pytest.raises(ValueError, match=msg):
            # 调用 idxmin 方法
            allna.idxmin()

        # 创建一个 datetime64[ns] 类型的 Series 对象
        s = Series(date_range("20130102", periods=6))
        
        # 使用 idxmin 方法找到 Series 中最小值的索引
        result = s.idxmin()
        assert result == 0

        # 将第一个值设置为 NaN
        s[0] = np.nan
        
        # 再次使用 idxmin 方法找到 Series 中最小值的索引
        result = s.idxmin()
        assert result == 1
    def test_idxmax(self):
        # 测试 idxmax 方法

        # 创建一个包含20个元素的 Series，数据类型为 float64
        string_series = Series(range(20), dtype=np.float64, name="series")

        # 添加一些 NaN 值
        string_series[5:15] = np.nan

        # 测试 skipna 参数为 True 时的情况
        # 断言：索引出的最大值应该等于 Series 的最大值
        assert string_series[string_series.idxmax()] == string_series.max()

        # 使用 pytest 来断言当 skipna 参数为 False 时抛出 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(ValueError, match="Encountered an NA value"):
            assert isna(string_series.idxmax(skipna=False))

        # 删除 NaN 值后的 Series
        nona = string_series.dropna()

        # 断言：非 NaN 值的最大值应该等于 Series 的最大值
        assert nona[nona.idxmax()] == nona.max()

        # 断言：非 NaN 值的最大值的索引在非 NaN 值列表中的索引等于非 NaN 值数组的最大值索引
        assert nona.index.values.tolist().index(nona.idxmax()) == nona.values.argmax()

        # 全部为 NaN 值的 Series
        allna = string_series * np.nan

        # 断言：当 Series 全部为 NaN 值时，调用 idxmax 方法会抛出 ValueError 异常
        msg = "Encountered all NA values"
        with pytest.raises(ValueError, match=msg):
            allna.idxmax()

        # 创建一个包含日期的 Series
        s = Series(date_range("20130102", periods=6))

        # 断言：日期 Series 的最大值的索引应该是 5
        result = s.idxmax()
        assert result == 5

        # 将第五个位置的值设为 NaN
        s[5] = np.nan

        # 断言：修改后的日期 Series 的最大值的索引应该是 4
        result = s.idxmax()
        assert result == 4

        # 创建一个具有 float64 数据类型索引的 Series
        s = Series([1, 2, 3], [1.1, 2.1, 3.1])

        # 断言：具有 float64 数据类型索引的 Series 的最大值的索引应该是 3.1
        result = s.idxmax()
        assert result == 3.1

        # 断言：具有 float64 数据类型索引的 Series 的最小值的索引应该是 1.1
        result = s.idxmin()
        assert result == 1.1

        # 使用索引作为值创建 Series
        s = Series(s.index, s.index)

        # 断言：使用索引作为值的 Series 的最大值的索引应该是 3.1
        result = s.idxmax()
        assert result == 3.1

        # 断言：使用索引作为值的 Series 的最小值的索引应该是 1.1
        result = s.idxmin()
        assert result == 1.1

    def test_all_any(self):
        # 测试 all 和 any 方法

        # 创建一个包含从 0 到 9 的浮点数的 Series，索引从 "2020-01-01" 开始
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        # 创建一个布尔 Series，表示是否大于 0
        bool_series = ts > 0

        # 断言：布尔 Series 中不是所有值都为 True
        assert not bool_series.all()

        # 断言：布尔 Series 中至少有一个值为 True
        assert bool_series.any()

        # 创建一个包含字符串和布尔值的 Series
        s = Series(["abc", True])

        # 断言：Series 中至少有一个值为 True
        assert s.any()

    def test_numpy_all_any(self, index_or_series):
        # 测试 numpy 的 all 和 any 方法

        # 从 index_or_series 中创建一个 Index
        idx = index_or_series([0, 1, 2])

        # 断言：Index 中不是所有值都为 True
        assert not np.all(idx)

        # 断言：Index 中至少有一个值为 True
        assert np.any(idx)

        # 创建一个 Index
        idx = Index([1, 2, 3])

        # 断言：Index 中所有值都为 True
        assert np.all(idx)

    def test_all_any_skipna(self):
        # 测试 all 和 any 方法的 skipna 参数

        # 创建一个包含 NaN 和 True 的 Series
        s1 = Series([np.nan, True])

        # 断言：当 skipna 参数为 False 时，NaN 和 True 的 Series 的 all 方法结果为 True
        assert s1.all(skipna=False)

        # 断言：当 skipna 参数为 True 时，NaN 和 True 的 Series 的 all 方法结果为 False
        assert s1.all(skipna=True)

        # 创建一个包含 NaN 和 False 的 Series
        s2 = Series([np.nan, False])

        # 断言：NaN 和 False 的 Series 的 any 方法结果为 True
        assert s2.any(skipna=False)

        # 断言：NaN 和 False 的 Series 的 any 方法结果为 False
        assert not s2.any(skipna=True)

    def test_all_any_bool_only(self):
        # 测试 all 和 any 方法的 bool_only 参数

        # 创建一个包含布尔值的 Series，同时有重复的索引
        s = Series([False, False, True, True, False, True], index=[0, 0, 1, 1, 2, 2])

        # 断言：布尔值的 Series 的 any 方法结果为 True
        assert s.any(bool_only=True)

        # 断言：布尔值的 Series 的 all 方法结果为 False
        assert not s.all(bool_only=True)

    def test_any_all_object_dtype(self, all_boolean_reductions, skipna):
        # 测试 object 类型的 Series 的 all 和 any 方法

        # 创建一个包含字符串的 Series
        ser = Series(["a", "b", "c", "d", "e"], dtype=object)

        # 调用 all_boolean_reductions 指定的方法，验证结果是否为 True
        result = getattr(ser, all_boolean_reductions)(skipna=skipna)
        expected = True

        # 断言：调用指定方法后的结果应该等于预期值 True
        assert result == expected
    # 使用 pytest.mark.parametrize 装饰器，为 test_any_all_object_dtype_missing 方法参数化测试数据
    @pytest.mark.parametrize(
        "data", [[False, None], [None, False], [False, np.nan], [np.nan, False]]
    )
    # 测试方法，验证对缺失值的处理
    def test_any_all_object_dtype_missing(self, data, all_boolean_reductions):
        # GH#27709
        # 创建一个 Series 对象，使用传入的 data 初始化
        ser = Series(data)
        # 调用 Series 对象的 all_boolean_reductions 方法，设置 skipna=False
        result = getattr(ser, all_boolean_reductions)(skipna=False)
    
        # None 被视为 False，而 np.nan 被视为 True
        # 根据 all_boolean_reductions 的值和 data 中是否包含 None 来设置期望的结果
        expected = all_boolean_reductions == "any" and None not in data
        # 断言方法的返回值与期望值相等
        assert result == expected
    
    # 参数化测试方法，测试不同的数据类型和 skipna 参数
    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "UInt64", "Float64"])
    @pytest.mark.parametrize(
        # 期望的数据结构，分为 skipna=False/any, skipna=False/all, skipna=True/any, skipna=True/all
        "data,expected_data",
        [
            ([0, 0, 0], [[False, False], [False, False]]),
            ([1, 1, 1], [[True, True], [True, True]]),
            ([pd.NA, pd.NA, pd.NA], [[pd.NA, pd.NA], [False, True]]),
            ([0, pd.NA, 0], [[pd.NA, False], [False, False]]),
            ([1, pd.NA, 1], [[True, pd.NA], [True, True]]),
            ([1, pd.NA, 0], [[True, False], [True, False]]),
        ],
    )
    # 测试方法，验证对可空数据的处理逻辑
    def test_any_all_nullable_kleene_logic(
        self, all_boolean_reductions, skipna, data, dtype, expected_data
    ):
        # GH-37506, GH-41967
        # 创建一个 Series 对象，使用传入的 data 和 dtype 初始化
        ser = Series(data, dtype=dtype)
        # 根据 expected_data 和 skipna、all_boolean_reductions 的值确定期望的结果
        expected = expected_data[skipna][all_boolean_reductions == "all"]
    
        # 调用 Series 对象的 all_boolean_reductions 方法，设置 skipna 参数
        result = getattr(ser, all_boolean_reductions)(skipna=skipna)
        # 断言方法的返回值与期望值相等，或者都为 pd.NA
        assert (result is pd.NA and expected is pd.NA) or result == expected
    
    # 测试方法，验证在 axis=1 且仅考虑布尔值的情况下，DataFrame 的 any 方法的行为
    def test_any_axis1_bool_only(self):
        # GH#32432
        # 创建一个 DataFrame 对象，包含两列 A 和 B
        df = DataFrame({"A": [True, False], "B": [1, 2]})
        # 调用 DataFrame 对象的 any 方法，设置 axis=1 和 bool_only=True
        result = df.any(axis=1, bool_only=True)
        # 创建一个期望的 Series 对象，包含预期的布尔值
        expected = Series([True, False])
        # 使用测试模块中的 assert 函数比较 result 和 expected，保证它们相等
        tm.assert_series_equal(result, expected)
    def test_any_all_datetimelike(self):
        # GH#38723 这些可能不是期望的长期行为（GH#34479）
        # 但在此期间应该是内部一致的
        # 从指定日期开始创建一个时间范围，包括3个时间点，返回其内部数据
        dta = date_range("1995-01-02", periods=3)._data
        # 使用内部数据创建一个Series对象
        ser = Series(dta)
        # 使用Series对象创建一个DataFrame对象
        df = DataFrame(ser)

        # GH#34479
        # 检查是否引发TypeError异常，异常信息应匹配给定的消息
        msg = "datetime64 type does not support operation '(any|all)'"
        with pytest.raises(TypeError, match=msg):
            dta.all()
        with pytest.raises(TypeError, match=msg):
            dta.any()

        with pytest.raises(TypeError, match=msg):
            ser.all()
        with pytest.raises(TypeError, match=msg):
            ser.any()

        with pytest.raises(TypeError, match=msg):
            df.any().all()
        with pytest.raises(TypeError, match=msg):
            df.all().all()

        # 将日期时间数据转换为UTC时区
        dta = dta.tz_localize("UTC")
        ser = Series(dta)
        df = DataFrame(ser)
        # GH#34479
        with pytest.raises(TypeError, match=msg):
            dta.all()
        with pytest.raises(TypeError, match=msg):
            dta.any()

        with pytest.raises(TypeError, match=msg):
            ser.all()
        with pytest.raises(TypeError, match=msg):
            ser.any()

        with pytest.raises(TypeError, match=msg):
            df.any().all()
        with pytest.raises(TypeError, match=msg):
            df.all().all()

        # 计算时间差，以第一个时间点为基准
        tda = dta - dta[0]
        ser = Series(tda)
        df = DataFrame(ser)

        # 断言至少有一个时间差不为零
        assert tda.any()
        # 断言所有时间差都为零
        assert not tda.all()

        # 断言至少有一个时间序列元素不为零
        assert ser.any()
        # 断言所有时间序列元素都为零
        assert not ser.all()

        # 断言DataFrame中至少有一个元素为True
        assert df.any().all()
        # 断言DataFrame中所有元素中至少有一个为False
        assert not df.all().any()

    def test_any_all_pyarrow_string(self):
        # GH#54591
        # 确保pyarrow库可导入，否则跳过测试
        pytest.importorskip("pyarrow")
        # 创建一个包含两个字符串的Series对象，dtype指定为"string[pyarrow_numpy]"
        ser = Series(["", "a"], dtype="string[pyarrow_numpy]")
        # 断言至少有一个非空字符串
        assert ser.any()
        # 断言所有字符串不都为空
        assert not ser.all()

        # 创建一个包含一个空值和一个非空字符串的Series对象，dtype指定为"string[pyarrow_numpy]"
        ser = Series([None, "a"], dtype="string[pyarrow_numpy]")
        # 断言至少有一个非空值
        assert ser.any()
        # 断言所有值都非空
        assert ser.all()
        # 断言所有值至少有一个非空（忽略NaN）
        assert not ser.all(skipna=False)

        # 创建一个包含两个空值的Series对象，dtype指定为"string[pyarrow_numpy]"
        ser = Series([None, ""], dtype="string[pyarrow_numpy]")
        # 断言所有值都为空
        assert not ser.any()
        # 断言所有值都为空

        # 创建一个包含两个非空字符串的Series对象，dtype指定为"string[pyarrow_numpy]"
        ser = Series(["a", "b"], dtype="string[pyarrow_numpy]")
        # 断言至少有一个非空字符串
        assert ser.any()
        # 断言所有字符串都非空
        assert ser.all()
    # 定义一个测试方法，用于测试 Timedelta64 数据类型的分析功能
    def test_timedelta64_analytics(self):
        # 创建一个日期范围，包含三天，每天的频率为一天
        dti = date_range("2012-1-1", periods=3, freq="D")
        # 创建一个时间序列，每个日期减去固定时间戳的时间差
        td = Series(dti) - Timestamp("20120101")

        # 计算时间差序列中最小值的索引
        result = td.idxmin()
        # 断言最小值的索引应为 0
        assert result == 0

        # 计算时间差序列中最大值的索引
        result = td.idxmax()
        # 断言最大值的索引应为 2
        assert result == 2

        # GH#2982
        # 测试带有 NaT（Not a Time）的情况
        td[0] = np.nan

        # 再次计算时间差序列中最小值的索引
        result = td.idxmin()
        # 断言最小值的索引应为 1
        assert result == 1

        # 再次计算时间差序列中最大值的索引
        result = td.idxmax()
        # 断言最大值的索引应为 2
        assert result == 2

        # 计算两个日期序列的绝对时间差
        s1 = Series(date_range("20120101", periods=3))
        s2 = Series(date_range("20120102", periods=3))
        expected = Series(s2 - s1)

        result = np.abs(s1 - s2)
        # 断言计算得到的绝对时间差应与预期的一致
        tm.assert_series_equal(result, expected)

        result = (s1 - s2).abs()
        # 断言计算得到的绝对时间差应与预期的一致
        tm.assert_series_equal(result, expected)

        # 计算时间差序列中的最大值和最小值
        result = td.max()
        expected = Timedelta("2 days")
        # 断言计算得到的最大值应与预期的一致
        assert result == expected

        result = td.min()
        expected = Timedelta("1 days")
        # 断言计算得到的最小值应与预期的一致
        assert result == expected

    # 定义一个测试方法，用于测试空序列情况下 idxmin 和 idxmax 方法应引发异常的情况
    def test_assert_idxminmax_empty_raises(self):
        """
        Cases where ``Series.argmax`` and related should raise an exception
        """
        # 创建一个空的浮点数序列
        test_input = Series([], dtype="float64")
        msg = "attempt to get argmin of an empty sequence"
        # 断言调用空序列的 idxmin 方法应引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin(skipna=False)
        msg = "attempt to get argmax of an empty sequence"
        # 断言调用空序列的 idxmax 方法应引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax(skipna=False)

    # 定义一个测试方法，用于测试对象类型序列的 idxmin 和 idxmax 方法
    def test_idxminmax_object_dtype(self, using_infer_string):
        # 在早于版本 2.1 的情况下，对象类型的序列不支持 argmin/max 方法
        ser = Series(["foo", "bar", "baz"])
        # 断言对象类型序列的最大值索引应为 0
        assert ser.idxmax() == 0
        assert ser.idxmax(skipna=False) == 0
        # 断言对象类型序列的最小值索引应为 1
        assert ser.idxmin() == 1
        assert ser.idxmin(skipna=False) == 1

        ser2 = Series([(1,), (2,)])
        # 断言对象类型序列的最大值索引应为 1
        assert ser2.idxmax() == 1
        assert ser2.idxmax(skipna=False) == 1
        # 断言对象类型序列的最小值索引应为 0
        assert ser2.idxmin() == 0
        assert ser2.idxmin(skipna=False) == 0

        if not using_infer_string:
            # 尝试将 np.nan 与字符串进行比较会引发 TypeError 异常
            ser3 = Series(["foo", "foo", "bar", "bar", None, np.nan, "baz"])
            msg = "'>' not supported between instances of 'float' and 'str'"
            # 断言尝试比较会引发 TypeError 异常
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax(skipna=False)
            msg = "'<' not supported between instances of 'float' and 'str'"
            # 断言尝试比较会引发 TypeError 异常
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin(skipna=False)
    # 测试DataFrame对象的idxmax方法，查找每列中的最大值索引
    def test_idxminmax_object_frame(self):
        # 创建包含姓名和数值的DataFrame
        df = DataFrame([["zimm", 2.5], ["biff", 1.0], ["bid", 12.0]])
        # 调用idxmax方法，返回每列最大值的索引
        res = df.idxmax()
        # 创建期望的Series对象，包含每列最大值的索引
        exp = Series([0, 2])
        # 断言返回的结果与期望的结果相等
        tm.assert_series_equal(res, exp)

    # 测试Series对象的idxmax和idxmin方法，处理包含元组的Series对象
    def test_idxminmax_object_tuples(self):
        # 创建包含元组的Series对象
        ser = Series([(1, 3), (2, 2), (3, 1)])
        # 断言调用idxmax方法返回的最大值索引为2
        assert ser.idxmax() == 2
        # 断言调用idxmin方法返回的最小值索引为0
        assert ser.idxmin() == 0
        # 断言调用idxmax方法（skipna=False）返回的最大值索引为2
        assert ser.idxmax(skipna=False) == 2
        # 断言调用idxmin方法（skipna=False）返回的最小值索引为0
        assert ser.idxmin(skipna=False) == 0

    # 测试DataFrame对象的idxmax和idxmin方法，处理包含Decimal对象的DataFrame
    def test_idxminmax_object_decimals(self):
        # 创建包含Decimal对象的DataFrame
        df = DataFrame(
            {
                "idx": [0, 1],
                "x": [Decimal("8.68"), Decimal("42.23")],
                "y": [Decimal("7.11"), Decimal("79.61")],
            }
        )
        # 调用idxmax方法，返回每列最大值的索引
        res = df.idxmax()
        # 创建期望的Series对象，包含每列最大值的索引
        exp = Series({"idx": 1, "x": 1, "y": 1})
        # 断言返回的结果与期望的结果相等
        tm.assert_series_equal(res, exp)

        # 调用idxmin方法，返回每列最小值的索引
        res2 = df.idxmin()
        # 创建期望的Series对象，包含每列最小值的索引
        exp2 = exp - 1
        # 断言返回的结果与期望的结果相等
        tm.assert_series_equal(res2, exp2)

    # 测试Series对象的argmax和argmin方法，处理包含整数对象的Series
    def test_argminmax_object_ints(self):
        # 创建包含整数对象的Series
        ser = Series([0, 1], dtype="object")
        # 断言调用argmax方法返回的最大值索引为1
        assert ser.argmax() == 1
        # 断言调用argmin方法返回的最小值索引为0
        assert ser.argmin() == 0
        # 断言调用argmax方法（skipna=False）返回的最大值索引为1
        assert ser.argmax(skipna=False) == 1
        # 断言调用argmin方法（skipna=False）返回的最小值索引为0
        assert ser.argmin(skipna=False) == 0

    # 测试Series对象的idxmax和idxmin方法，处理包含Inf值和NA值的数值Series
    def test_idxminmax_with_inf(self):
        # 创建包含Inf值和NA值的Series
        s = Series([0, -np.inf, np.inf, np.nan])

        # 断言调用idxmin方法返回的最小值索引为1
        assert s.idxmin() == 1
        # 使用pytest检查，调用idxmin方法（skipna=False）时会引发ValueError异常，匹配指定的错误消息
        with pytest.raises(ValueError, match="Encountered an NA value"):
            s.idxmin(skipna=False)

        # 断言调用idxmax方法返回的最大值索引为2
        assert s.idxmax() == 2
        # 使用pytest检查，调用idxmax方法（skipna=False）时会引发ValueError异常，匹配指定的错误消息
        with pytest.raises(ValueError, match="Encountered an NA value"):
            s.idxmax(skipna=False)

    # 测试Series对象的sum方法，处理uint64类型的数据
    def test_sum_uint64(self):
        # 创建包含uint64类型数据的Series
        s = Series([10000000000000000000], dtype="uint64")
        # 调用sum方法求和
        result = s.sum()
        # 期望的求和结果为np.uint64(10000000000000000000)
        expected = np.uint64(10000000000000000000)
        # 使用tm.assert_almost_equal函数断言结果与期望相等
        tm.assert_almost_equal(result, expected)

    # 测试Series对象的sum方法，验证在求和后保留有符号性质
    def test_signedness_preserved_after_sum(self):
        # 创建包含整数的Series
        ser = Series([1, 2, 3, 4])
        # 将Series对象转换为uint8类型后，调用sum方法，检查结果的数据类型是否为uint64
        assert ser.astype("uint8").sum().dtype == "uint64"
class TestDatetime64SeriesReductions:
    # Note: the name TestDatetime64SeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize(
        "nat_ser",
        [
            Series([NaT, NaT]),  # 创建包含两个 NaT 值的 Series 对象
            Series([NaT, Timedelta("nat")]),  # 创建包含一个 NaT 和一个 Timedelta('nat') 值的 Series 对象
            Series([Timedelta("nat"), Timedelta("nat")]),  # 创建包含两个 Timedelta('nat') 值的 Series 对象
        ],
    )
    def test_minmax_nat_series(self, nat_ser):
        # GH#23282
        assert nat_ser.min() is NaT  # 断言 Series 的最小值为 NaT
        assert nat_ser.max() is NaT  # 断言 Series 的最大值为 NaT
        assert nat_ser.min(skipna=False) is NaT  # 断言 Series 的最小值（不跳过 NA 值）为 NaT
        assert nat_ser.max(skipna=False) is NaT  # 断言 Series 的最大值（不跳过 NA 值）为 NaT

    @pytest.mark.parametrize(
        "nat_df",
        [
            [NaT, NaT],  # 创建包含两个 NaT 值的列表
            [NaT, Timedelta("nat")],  # 创建包含一个 NaT 和一个 Timedelta('nat') 值的列表
            [Timedelta("nat"), Timedelta("nat")],  # 创建包含两个 Timedelta('nat') 值的列表
        ],
    )
    def test_minmax_nat_dataframe(self, nat_df):
        # GH#23282
        nat_df = DataFrame(nat_df)  # 使用列表创建 DataFrame 对象
        assert nat_df.min()[0] is NaT  # 断言 DataFrame 的第一列的最小值为 NaT
        assert nat_df.max()[0] is NaT  # 断言 DataFrame 的第一列的最大值为 NaT
        assert nat_df.min(skipna=False)[0] is NaT  # 断言 DataFrame 的第一列的最小值（不跳过 NA 值）为 NaT
        assert nat_df.max(skipna=False)[0] is NaT  # 断言 DataFrame 的第一列的最大值（不跳过 NA 值）为 NaT

    def test_min_max(self):
        rng = date_range("1/1/2000", "12/31/2000")  # 生成一个日期范围
        rng2 = rng.take(np.random.default_rng(2).permutation(len(rng)))  # 打乱顺序后的日期范围

        the_min = rng2.min()  # 获取打乱顺序后日期范围的最小值
        the_max = rng2.max()  # 获取打乱顺序后日期范围的最大值
        assert isinstance(the_min, Timestamp)  # 断言最小值是 Timestamp 类型
        assert isinstance(the_max, Timestamp)  # 断言最大值是 Timestamp 类型
        assert the_min == rng[0]  # 断言最小值等于原始日期范围的第一个日期
        assert the_max == rng[-1]  # 断言最大值等于原始日期范围的最后一个日期

        assert rng.min() == rng[0]  # 断言原始日期范围的最小值等于第一个日期
        assert rng.max() == rng[-1]  # 断言原始日期范围的最大值等于最后一个日期

    def test_min_max_series(self):
        rng = date_range("1/1/2000", periods=10, freq="4h")  # 生成一个包含 10 个日期的日期范围，频率为 4 小时
        lvls = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]  # 级别列表
        df = DataFrame(
            {
                "TS": rng,  # 时间序列列
                "V": np.random.default_rng(2).standard_normal(len(rng)),  # 随机生成的标准正态分布值列
                "L": lvls,  # 级别列
            }
        )

        result = df.TS.max()  # 获取时间序列列的最大值
        exp = Timestamp(df.TS.iat[-1])  # 预期的最大时间戳
        assert isinstance(result, Timestamp)  # 断言结果是 Timestamp 类型
        assert result == exp  # 断言结果等于预期值

        result = df.TS.min()  # 获取时间序列列的最小值
        exp = Timestamp(df.TS.iat[0])  # 预期的最小时间戳
        assert isinstance(result, Timestamp)  # 断言结果是 Timestamp 类型
        assert result == exp  # 断言结果等于预期值


class TestCategoricalSeriesReductions:
    # Note: the name TestCategoricalSeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_unordered_raises(self, function):
        # unordered cats have no min/max
        cat = Series(Categorical(["a", "b", "c", "d"], ordered=False))  # 创建一个无序分类 Series 对象
        msg = f"Categorical is not ordered for operation {function}"  # 错误消息字符串
        with pytest.raises(TypeError, match=msg):  # 断言操作未定义时会抛出 TypeError 异常，并匹配错误消息
            getattr(cat, function)()  # 调用对应的函数进行操作
    @pytest.mark.parametrize(
        "values, categories",
        [
            (list("abc"), list("abc")),  # 参数化测试数据，值为 ['a', 'b', 'c']，类别也为 ['a', 'b', 'c']
            (list("abc"), list("cba")),  # 参数化测试数据，值为 ['a', 'b', 'c']，类别为 ['c', 'b', 'a']
            (list("abc") + [np.nan], list("cba")),  # 参数化测试数据，值为 ['a', 'b', 'c', np.nan]，类别为 ['c', 'b', 'a']
            ([1, 2, 3], [3, 2, 1]),  # 参数化测试数据，值为 [1, 2, 3]，类别为 [3, 2, 1]
            ([1, 2, 3, np.nan], [3, 2, 1]),  # 参数化测试数据，值为 [1, 2, 3, np.nan]，类别为 [3, 2, 1]
        ],
    )
    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_ordered(self, values, categories, function):
        # GH 25303
        # 创建带有顺序的分类数据系列对象
        cat = Series(Categorical(values, categories=categories, ordered=True))
        # 调用对象的 min 或 max 方法，跳过 NaN 值
        result = getattr(cat, function)(skipna=True)
        # 根据函数类型选择预期结果
        expected = categories[0] if function == "min" else categories[2]
        # 断言测试结果与预期结果相符
        assert result == expected

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_ordered_with_nan_only(self, function, skipna):
        # https://github.com/pandas-dev/pandas/issues/33450
        # 创建只包含 NaN 的分类数据系列对象
        cat = Series(Categorical([np.nan], categories=[1, 2], ordered=True))
        # 调用对象的 min 或 max 方法，根据 skipna 参数决定是否跳过 NaN 值
        result = getattr(cat, function)(skipna=skipna)
        # 断言结果为 NaN
        assert result is np.nan

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_skipna(self, function, skipna):
        # 创建带有 NaN 值的分类数据系列对象
        cat = Series(
            Categorical(["a", "b", np.nan, "a"], categories=["b", "a"], ordered=True)
        )
        # 调用对象的 min 或 max 方法，根据 skipna 参数决定是否跳过 NaN 值
        result = getattr(cat, function)(skipna=skipna)

        if skipna is True:
            # 如果 skipna 为 True，选择预期结果为 "b" 或 "a"
            expected = "b" if function == "min" else "a"
            # 断言测试结果与预期结果相符
            assert result == expected
        else:
            # 如果 skipna 为 False，预期结果为 NaN
            assert result is np.nan
class TestSeriesMode:
    # Note: the name TestSeriesMode indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    # Test case for calculating mode of an empty Series
    def test_mode_empty(self, dropna):
        # Create an empty Series with float64 dtype
        s = Series([], dtype=np.float64)
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Assert that the result is equal to the original Series (which should be empty)
        tm.assert_series_equal(result, s)

    # Parameterized test cases for numerical data
    @pytest.mark.parametrize(
        "dropna, data, expected",
        [
            (True, [1, 1, 1, 2], [1]),  # Test mode with dropna=True
            (True, [1, 1, 1, 2, 3, 3, 3], [1, 3]),  # Test mode with dropna=True
            (False, [1, 1, 1, 2], [1]),  # Test mode with dropna=False
            (False, [1, 1, 1, 2, 3, 3, 3], [1, 3]),  # Test mode with dropna=False
        ],
    )
    def test_mode_numerical(self, dropna, data, expected, any_real_numpy_dtype):
        # Create a Series with numerical data
        s = Series(data, dtype=any_real_numpy_dtype)
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Create an expected Series based on the expected mode
        expected = Series(expected, dtype=any_real_numpy_dtype)
        # Assert that the result matches the expected mode
        tm.assert_series_equal(result, expected)

    # Parameterized test cases for numerical data including NaN
    @pytest.mark.parametrize("dropna, expected", [(True, [1.0]), (False, [1, np.nan])])
    def test_mode_numerical_nan(self, dropna, expected):
        # Create a Series with numerical data including NaN
        s = Series([1, 1, 2, np.nan, np.nan])
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Create an expected Series based on the expected mode
        expected = Series(expected)
        # Assert that the result matches the expected mode
        tm.assert_series_equal(result, expected)

    # Parameterized test cases for string and object types
    @pytest.mark.parametrize(
        "dropna, expected1, expected2, expected3",
        [(True, ["b"], ["bar"], ["nan"]), (False, ["b"], [np.nan], ["nan"])],
    )
    def test_mode_str_obj(self, dropna, expected1, expected2, expected3):
        # Test case for string and object types
        data = ["a"] * 2 + ["b"] * 3

        # Create a Series with string dtype
        s = Series(data, dtype="c")
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Create an expected Series based on the expected mode
        expected1 = Series(expected1, dtype="c")
        # Assert that the result matches the expected mode
        tm.assert_series_equal(result, expected1)

        # Another test case for object dtype including NaN
        data = ["foo", "bar", "bar", np.nan, np.nan, np.nan]

        # Create a Series with object dtype
        s = Series(data, dtype=object)
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Create an expected Series based on the expected mode
        expected2 = Series(expected2, dtype=None if expected2 == ["bar"] else object)
        # Assert that the result matches the expected mode
        tm.assert_series_equal(result, expected2)

        # Another test case for object dtype converted to string
        data = ["foo", "bar", "bar", np.nan, np.nan, np.nan]

        # Create a Series with object dtype, converted to string
        s = Series(data, dtype=object).astype(str)
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Create an expected Series based on the expected mode
        expected3 = Series(expected3)
        # Assert that the result matches the expected mode
        tm.assert_series_equal(result, expected3)

    # Parameterized test cases for mixed dtype
    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [(True, ["foo"], ["foo"]), (False, ["foo"], [np.nan])],
    )
    def test_mode_mixeddtype(self, dropna, expected1, expected2):
        # Test case for mixed dtype (numeric and string)
        s = Series([1, "foo", "foo"])
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Create an expected Series based on the expected mode
        expected = Series(expected1)
        # Assert that the result matches the expected mode
        tm.assert_series_equal(result, expected)

        # Another test case for mixed dtype including NaN
        s = Series([1, "foo", "foo", np.nan, np.nan, np.nan])
        # Compute the mode of the Series
        result = s.mode(dropna)
        # Create an expected Series based on the expected mode
        expected = Series(expected2, dtype=None if expected2 == ["foo"] else object)
        # Assert that the result matches the expected mode
        tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [  # 参数化测试用例，分别测试 dropna 为 True 和 False 的情况
            (
                True,  # 当 dropna 为 True 时的预期结果
                ["1900-05-03", "2011-01-03", "2013-01-02"],  # 预期结果列表1
                ["2011-01-03", "2013-01-02"],  # 预期结果列表2
            ),
            (
                False,  # 当 dropna 为 False 时的预期结果
                [np.nan],  # 预期结果列表1
                [np.nan, "2011-01-03", "2013-01-02"],  # 预期结果列表2
            ),
        ],
    )
    def test_mode_datetime(self, dropna, expected1, expected2):
        # 测试日期时间类型的 mode 方法

        s = Series(
            ["2011-01-03", "2013-01-02", "1900-05-03", "nan", "nan"], dtype="M8[ns]"
        )
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="M8[ns]")
        tm.assert_series_equal(result, expected1)

        s = Series(
            [
                "2011-01-03",
                "2013-01-02",
                "1900-05-03",
                "2011-01-03",
                "2013-01-02",
                "nan",
                "nan",
            ],
            dtype="M8[ns]",
        )
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="M8[ns]")
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [  # 参数化测试用例，分别测试 dropna 为 True 和 False 的情况
            (True, ["-1 days", "0 days", "1 days"], ["2 min", "1 day"]),  # 预期结果列表1
            (False, [np.nan], [np.nan, "2 min", "1 day"]),  # 预期结果列表2
        ],
    )
    def test_mode_timedelta(self, dropna, expected1, expected2):
        # 测试时间间隔类型的 mode 方法
        # gh-5986: Test timedelta types.

        s = Series(
            ["1 days", "-1 days", "0 days", "nan", "nan"], dtype="timedelta64[ns]"
        )
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected1)

        s = Series(
            [
                "1 day",
                "1 day",
                "-1 day",
                "-1 day 2 min",
                "2 min",
                "2 min",
                "nan",
                "nan",
            ],
            dtype="timedelta64[ns]",
        )
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2, expected3",
        [  # 参数化测试用例，分别测试 dropna 为 True 和 False 的情况
            (
                True,  # 当 dropna 为 True 时的预期结果
                Categorical([1, 2], categories=[1, 2]),  # 预期结果1
                Categorical(["a"], categories=[1, "a"]),  # 预期结果2
                Categorical([3, 1], categories=[3, 2, 1], ordered=True),  # 预期结果3
            ),
            (
                False,  # 当 dropna 为 False 时的预期结果
                Categorical([np.nan], categories=[1, 2]),  # 预期结果1
                Categorical([np.nan, "a"], categories=[1, "a"]),  # 预期结果2
                Categorical([np.nan, 3, 1], categories=[3, 2, 1], ordered=True),  # 预期结果3
            ),
        ],
    )
    # 定义一个测试方法，用于测试 Series 对象的 mode 方法在不同条件下的行为
    def test_mode_category(self, dropna, expected1, expected2, expected3):
        # 创建一个包含类别数据的 Series 对象，计算其 mode，并与预期结果 expected1 进行比较
        s = Series(Categorical([1, 2, np.nan, np.nan]))
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="category")
        tm.assert_series_equal(result, expected1)

        # 创建另一个包含类别数据的 Series 对象，计算其 mode，并与预期结果 expected2 进行比较
        s = Series(Categorical([1, "a", "a", np.nan, np.nan]))
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="category")
        tm.assert_series_equal(result, expected2)

        # 创建第三个包含有序类别数据的 Series 对象，计算其 mode，并与预期结果 expected3 进行比较
        s = Series(
            Categorical(
                [1, 1, 2, 3, 3, np.nan, np.nan], categories=[3, 2, 1], ordered=True
            )
        )
        result = s.mode(dropna)
        expected3 = Series(expected3, dtype="category")
        tm.assert_series_equal(result, expected3)

    # 使用 pytest 的 parametrize 装饰器，定义一个测试方法，用于测试 uint64 整数溢出情况下的 mode 方法
    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [(True, [2**63], [1, 2**63]), (False, [2**63], [1, 2**63])],
    )
    def test_mode_intoverflow(self, dropna, expected1, expected2):
        # 创建一个包含 uint64 数据的 Series 对象，计算其 mode，并与预期结果 expected1 进行比较
        s = Series([1, 2**63, 2**63], dtype=np.uint64)
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype=np.uint64)
        tm.assert_series_equal(result, expected1)

        # 创建另一个包含 uint64 数据的 Series 对象，计算其 mode，并与预期结果 expected2 进行比较
        s = Series([1, 2**63], dtype=np.uint64)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=np.uint64)
        tm.assert_series_equal(result, expected2)

    # 定义一个测试方法，用于测试 mode 方法在结果无法排序时触发警告的情况
    def test_mode_sortwarning(self):
        # 创建一个包含数据和 NaN 的 Series 对象
        expected = Series(["foo", np.nan])
        s = Series([1, "foo", "foo", np.nan, np.nan])

        # 使用 assert_produces_warning 上下文管理器捕获警告信息，测试 mode 方法的结果排序问题
        with tm.assert_produces_warning(UserWarning, match="Unable to sort modes"):
            result = s.mode(dropna=False)
            # 对结果进行排序以使其符合预期，并重置索引
            result = result.sort_values().reset_index(drop=True)

        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，用于测试布尔类型数据包含 NA 值时的 mode 方法行为
    def test_mode_boolean_with_na(self):
        # 创建一个包含 True、False、NA 的布尔类型 Series 对象
        ser = Series([True, False, True, pd.NA], dtype="boolean")
        result = ser.mode()
        expected = Series({0: True}, dtype="boolean")
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器，定义一个测试方法，用于测试复数类型数据的 mode 方法
    @pytest.mark.parametrize(
        "array,expected,dtype",
        [
            (
                [0, 1j, 1, 1, 1 + 1j, 1 + 2j],
                [1],
                np.complex128,
            ),
            (
                [0, 1j, 1, 1, 1 + 1j, 1 + 2j],
                [1],
                np.complex64,
            ),
            (
                [1 + 1j, 2j, 1 + 1j],
                [1 + 1j],
                np.complex128,
            ),
        ],
    )
    def test_single_mode_value_complex(self, array, expected, dtype):
        # 创建一个复数类型的 Series 对象，计算其 mode，并与预期结果 expected 进行比较
        result = Series(array, dtype=dtype).mode()
        expected = Series(expected, dtype=dtype)
        tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize(
        "array,expected,dtype",
        [
            (
                # no modes
                [0, 1j, 1, 1 + 1j, 1 + 2j],  # 输入数组1，包含复数和实数
                [0j, 1j, 1 + 0j, 1 + 1j, 1 + 2j],  # 期望输出数组1，复数排序后的结果
                np.complex128,  # 数组元素的数据类型
            ),
            (
                [1 + 1j, 2j, 1 + 1j, 2j, 3],  # 输入数组2，包含复数和实数
                [2j, 1 + 1j],  # 期望输出数组2，复数排序后的结果
                np.complex64,  # 数组元素的数据类型
            ),
        ],
    )
    def test_multimode_complex(self, array, expected, dtype):
        # GH 17927
        # mode tries to sort multimodal series.
        # Complex numbers are sorted by their magnitude
        result = Series(array, dtype=dtype).mode()  # 调用 Series 类创建对象并计算众数
        expected = Series(expected, dtype=dtype)  # 创建期望的 Series 对象
        tm.assert_series_equal(result, expected)  # 断言计算结果与期望值是否相等
```