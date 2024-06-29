# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_index_new.py`

```
"""
Tests for the Index constructor conducting inference.
"""

# 导入所需的模块和类
from datetime import (
    datetime,            # 导入日期时间类
    timedelta,           # 导入时间差类
    timezone,            # 导入时区类
)
from decimal import Decimal  # 导入 Decimal 类

import numpy as np        # 导入 NumPy 库并重命名为 np
import pytest             # 导入 pytest 测试框架

from pandas._libs.tslibs.timezones import maybe_get_tz  # 导入 pandas 时间相关模块

from pandas import (      # 导入多个 pandas 类和函数
    NA,                   # 导入 NA 常量
    Categorical,          # 导入分类数据类型
    CategoricalIndex,     # 导入分类索引类型
    DatetimeIndex,        # 导入日期时间索引类型
    Index,                # 导入通用索引类型
    IntervalIndex,        # 导入区间索引类型
    MultiIndex,           # 导入多级索引类型
    NaT,                  # 导入 NaT 常量
    PeriodIndex,          # 导入周期索引类型
    Series,               # 导入序列类型
    TimedeltaIndex,       # 导入时间差索引类型
    Timestamp,            # 导入时间戳类型
    array,                # 导入 array 函数
    date_range,           # 导入 date_range 函数
    period_range,         # 导入 period_range 函数
    timedelta_range,      # 导入 timedelta_range 函数
)
import pandas._testing as tm  # 导入 pandas 测试模块

# 定义测试类 TestIndexConstructorInference
class TestIndexConstructorInference:
    def test_object_all_bools(self):
        # 测试 ndarray[object] 类型全为布尔值时的行为
        arr = np.array([True, False], dtype=object)
        res = Index(arr)
        assert res.dtype == object

        # 检查与 Series 的行为是否一致
        assert Series(arr).dtype == object

    def test_object_all_complex(self):
        # 测试 ndarray[object] 类型全为复数时的行为
        arr = np.array([complex(1), complex(2)], dtype=object)
        res = Index(arr)
        assert res.dtype == object

        # 检查与 Series 的行为是否一致
        assert Series(arr).dtype == object

    @pytest.mark.parametrize("val", [NaT, None, np.nan, float("nan")])
    def test_infer_nat(self, val):
        # 测试在包含 NaT/None/nan 且至少一个 NaT 的情况下的推断行为
        values = [NaT, val]

        idx = Index(values)
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

        idx = Index(values[::-1])
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

        idx = Index(np.array(values, dtype=object))
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

        idx = Index(np.array(values, dtype=object)[::-1])
        assert idx.dtype == "datetime64[s]" and idx.isna().all()

    @pytest.mark.parametrize("na_value", [None, np.nan])
    @pytest.mark.parametrize("vtype", [list, tuple, iter])
    def test_construction_list_tuples_nan(self, na_value, vtype):
        # 测试包含 NaN 的有效元组的构建行为
        values = [(1, "two"), (3.0, na_value)]
        result = Index(vtype(values))
        expected = MultiIndex.from_tuples(values)
        tm.assert_index_equal(result, expected)

    def test_constructor_int_dtype_float(self, any_int_numpy_dtype):
        # 测试在给定整数类型的情况下，传入浮点数的构造行为
        expected = Index([0, 1, 2, 3], dtype=any_int_numpy_dtype)
        result = Index([0.0, 1.0, 2.0, 3.0], dtype=any_int_numpy_dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize(
        "vals", [[True, False, True], np.array([True, False, True], dtype=bool)]
    )
    # 测试构造函数，将给定的索引和值用布尔类型处理后存储到 Index 对象中
    def test_constructor_dtypes_to_object(self, cast_index, vals):
        if cast_index:
            # 如果需要强制转换索引为布尔类型，则创建 Index 对象
            index = Index(vals, dtype=bool)
        else:
            # 否则直接创建 Index 对象
            index = Index(vals)

        # 断言 index 是 Index 类型的对象
        assert type(index) is Index
        # 断言 index 的数据类型是 bool
        assert index.dtype == bool

    # 测试构造函数，处理分类索引并返回 object 类型的 Index 对象
    def test_constructor_categorical_to_object(self):
        # 创建分类索引对象
        ci = CategoricalIndex(range(5))
        # 使用 object 数据类型创建 Index 对象
        result = Index(ci, dtype=object)
        # 断言 result 不是 CategoricalIndex 类型的对象
        assert not isinstance(result, CategoricalIndex)

    # 测试构造函数，推断 PeriodIndex 类型并返回对应的 Index 对象
    def test_constructor_infer_periodindex(self):
        # 创建 PeriodIndex 对象
        xp = period_range("2012-1-1", freq="M", periods=3)
        # 使用 PeriodIndex 对象创建 Index 对象
        rs = Index(xp)
        # 断言 rs 和 xp 相等
        tm.assert_index_equal(rs, xp)
        # 断言 rs 是 PeriodIndex 类型的对象
        assert isinstance(rs, PeriodIndex)

    # 测试构造函数，从周期列表创建 Index 对象并返回
    def test_from_list_of_periods(self):
        # 创建周期范围对象
        rng = period_range("1/1/2000", periods=20, freq="D")
        # 将周期范围转换为列表
        periods = list(rng)

        # 使用周期列表创建 Index 对象
        result = Index(periods)
        # 断言 result 是 PeriodIndex 类型的对象
        assert isinstance(result, PeriodIndex)

    # 测试构造函数，推断出 np.NaT 类型并将其转换为对应的 Index 对象
    @pytest.mark.parametrize("pos", [0, 1])
    @pytest.mark.parametrize(
        "klass,dtype,ctor",
        [
            (DatetimeIndex, "datetime64[ns]", np.datetime64("nat")),
            (TimedeltaIndex, "timedelta64[ns]", np.timedelta64("nat")),
        ],
    )
    def test_constructor_infer_nat_dt_like(
        self, pos, klass, dtype, ctor, nulls_fixture, request
    ):
        if isinstance(nulls_fixture, Decimal):
            # 如果 nulls_fixture 是 Decimal 类型，则跳过这个测试用例
            pytest.skip(
                f"We don't cast {type(nulls_fixture).__name__} to "
                "datetime64/timedelta64"
            )

        # 创建包含 NaT 的特定类型对象
        expected = klass([NaT, NaT])
        if dtype[0] == "d":
            # 如果 dtype 是以 'd' 开头，则转换为 M8[ns] 类型
            expected = expected.astype("M8[ns]")
        # 断言 expected 的数据类型是指定的 dtype
        assert expected.dtype == dtype
        # 构造数据列表，将 nulls_fixture 插入到指定位置
        data = [ctor]
        data.insert(pos, nulls_fixture)

        # 如果 nulls_fixture 是 NA，则期望结果包含 NA 和 NaT
        if nulls_fixture is NA:
            expected = Index([NA, NaT])
            # 对该测试用例打上失败标记，原因是与 np.NaT 构造函数存在问题，参见 GH 31884
            mark = pytest.mark.xfail(reason="Broken with np.NaT ctor; see GH 31884")
            request.applymarker(mark)

        # 使用 data 创建 Index 对象
        result = Index(data)

        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

        # 使用 object 类型的 np.array(data) 创建 Index 对象
        result = Index(np.array(data, dtype=object))

        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

    # 测试构造函数，混合 np.datetime64/timedelta64 nat 类型数据推断为 object 类型的 Index 对象
    @pytest.mark.parametrize("swap_objs", [True, False])
    def test_constructor_mixed_nat_objs_infers_object(self, swap_objs):
        # 创建包含 np.datetime64/timedelta64 nat 类型数据的列表
        data = [np.datetime64("nat"), np.timedelta64("nat")]
        if swap_objs:
            # 如果 swap_objs 为 True，则交换数据顺序
            data = data[::-1]

        # 期望结果是 object 类型的 Index 对象
        expected = Index(data, dtype=object)
        # 断言使用 data 创建的 Index 对象与期望的 expected 相等
        tm.assert_index_equal(Index(data), expected)
        # 断言使用 object 数据类型的 np.array(data) 创建的 Index 对象与期望的 expected 相等
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)
    # 定义一个测试方法，用于测试日期时间和datetime64的构造函数
    def test_constructor_datetime_and_datetime64(self, swap_objs):
        # 创建包含两个日期时间对象的列表：Timestamp对象和当前时间的np.datetime64对象
        data = [Timestamp(2021, 6, 8, 9, 42), np.datetime64("now")]
        
        # 如果swap_objs为True，反转data列表中的对象顺序
        if swap_objs:
            data = data[::-1]
        
        # 创建预期的DatetimeIndex对象，使用data列表作为参数
        expected = DatetimeIndex(data)

        # 断言使用Index类构造的对象与预期的DatetimeIndex相等
        tm.assert_index_equal(Index(data), expected)
        
        # 断言使用Index类构造的对象（强制指定dtype为object）与预期的DatetimeIndex相等
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    # 定义一个测试方法，用于测试混合时区的日期时间构造函数
    def test_constructor_datetimes_mixed_tzs(self):
        # 获取"US/Central"时区信息，返回时区对象
        tz = maybe_get_tz("US/Central")
        
        # 创建带有指定时区信息的datetime对象
        dt1 = datetime(2020, 1, 1, tzinfo=tz)
        
        # 创建带有UTC时区信息的datetime对象
        dt2 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        # 使用包含dt1和dt2的列表创建Index对象
        result = Index([dt1, dt2])
        
        # 创建预期的Index对象，使用包含dt1和dt2的列表，并指定dtype为object
        expected = Index([dt1, dt2], dtype=object)
        
        # 断言result和expected对象相等
        tm.assert_index_equal(result, expected)
class TestDtypeEnforced:
    # 检查不应默默忽略 dtype 关键字

    def test_constructor_object_dtype_with_ea_data(self, any_numeric_ea_dtype):
        # GH#45206
        # 创建一个包含单个元素的数组，指定任意数值型 ea 类型作为 dtype
        arr = array([0], dtype=any_numeric_ea_dtype)

        # 使用对象类型（dtype=object）创建索引对象 idx
        idx = Index(arr, dtype=object)
        # 断言索引对象 idx 的 dtype 为 object
        assert idx.dtype == object

    @pytest.mark.parametrize("dtype", [object, "float64", "uint64", "category"])
    def test_constructor_range_values_mismatched_dtype(self, dtype):
        # 创建一个包含 0 到 4 的整数范围的索引 rng
        rng = Index(range(5))

        # 使用给定的 dtype 创建索引对象 result
        result = Index(rng, dtype=dtype)
        # 断言索引对象 result 的 dtype 与给定的 dtype 相符
        assert result.dtype == dtype

        # 用给定的 dtype 创建索引对象 result
        result = Index(range(5), dtype=dtype)
        # 断言索引对象 result 的 dtype 与给定的 dtype 相符
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [object, "float64", "uint64", "category"])
    def test_constructor_categorical_values_mismatched_non_ea_dtype(self, dtype):
        # 创建一个分类对象 cat，包含整数 1、2、3
        cat = Categorical([1, 2, 3])

        # 使用给定的 dtype 创建索引对象 result
        result = Index(cat, dtype=dtype)
        # 断言索引对象 result 的 dtype 与给定的 dtype 相符
        assert result.dtype == dtype

    def test_constructor_categorical_values_mismatched_dtype(self):
        # 创建一个日期范围对象 dti，从 "2016-01-01" 开始，包含 3 个周期
        dti = date_range("2016-01-01", periods=3)
        # 使用 dti 的 dtype 创建分类对象 cat
        cat = Categorical(dti)
        # 使用 dti 的 dtype 创建索引对象 result
        result = Index(cat, dti.dtype)
        # 断言索引对象 result 与 dti 相等
        tm.assert_index_equal(result, dti)

        # 为 dti 对象设置时区 "Asia/Tokyo"
        dti2 = dti.tz_localize("Asia/Tokyo")
        # 使用 dti2 的 dtype 创建分类对象 cat2
        cat2 = Categorical(dti2)
        # 使用 dti2 的 dtype 创建索引对象 result
        result = Index(cat2, dti2.dtype)
        # 断言索引对象 result 与 dti2 相等
        tm.assert_index_equal(result, dti2)

        # 创建一个区间索引对象 ii，从 dti 的断点创建
        ii = IntervalIndex.from_breaks(range(5))
        # 使用 ii 的 dtype 创建分类对象 cat3
        cat3 = Categorical(ii)
        # 使用 ii 的 dtype 创建索引对象 result
        result = Index(cat3, dtype=ii.dtype)
        # 断言索引对象 result 与 ii 相等
        tm.assert_index_equal(result, ii)

    def test_constructor_ea_values_mismatched_categorical_dtype(self):
        # 创建一个日期范围对象 dti，从 "2016-01-01" 开始，包含 3 个周期
        dti = date_range("2016-01-01", periods=3)
        # 使用 "category" dtype 创建索引对象 result
        result = Index(dti, dtype="category")
        # 创建一个预期的分类索引对象 expected
        expected = CategoricalIndex(dti)
        # 断言索引对象 result 与预期的分类索引对象 expected 相等
        tm.assert_index_equal(result, expected)

        # 创建一个日期范围对象 dti2，从 "2016-01-01" 开始，包含 3 个周期，时区设置为 "US/Pacific"
        dti2 = date_range("2016-01-01", periods=3, tz="US/Pacific")
        # 使用 "category" dtype 创建索引对象 result
        result = Index(dti2, dtype="category")
        # 创建一个预期的分类索引对象 expected
        expected = CategoricalIndex(dti2)
        # 断言索引对象 result 与预期的分类索引对象 expected 相等
        tm.assert_index_equal(result, expected)

    def test_constructor_period_values_mismatched_dtype(self):
        # 创建一个周期范围对象 pi，从 "2016-01-01" 开始，包含 3 个周期，频率为 "D"（天）
        pi = period_range("2016-01-01", periods=3, freq="D")
        # 使用 "category" dtype 创建索引对象 result
        result = Index(pi, dtype="category")
        # 创建一个预期的分类索引对象 expected
        expected = CategoricalIndex(pi)
        # 断言索引对象 result 与预期的分类索引对象 expected 相等
        tm.assert_index_equal(result, expected)

    def test_constructor_timedelta64_values_mismatched_dtype(self):
        # 检查不应默默忽略 dtype 关键字
        # 创建一个时间间隔范围对象 tdi，从 "4 Days" 开始，包含 5 个周期
        tdi = timedelta_range("4 Days", periods=5)
        # 使用 "category" dtype 创建索引对象 result
        result = Index(tdi, dtype="category")
        # 创建一个预期的分类索引对象 expected
        expected = CategoricalIndex(tdi)
        # 断言索引对象 result 与预期的分类索引对象 expected 相等
        tm.assert_index_equal(result, expected)

    def test_constructor_interval_values_mismatched_dtype(self):
        # 创建一个日期范围对象 dti，从 "2016-01-01" 开始，包含 3 个周期
        dti = date_range("2016-01-01", periods=3)
        # 从 dti 的断点创建一个区间索引对象 ii
        ii = IntervalIndex.from_breaks(dti)
        # 使用 "category" dtype 创建索引对象 result
        result = Index(ii, dtype="category")
        # 创建一个预期的分类索引对象 expected
        expected = CategoricalIndex(ii)
        # 断言索引对象 result 与预期的分类索引对象 expected 相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试方法，用于验证在 period 数据类型与 datetime64 数据类型之间不匹配时的行为
    def test_constructor_datetime64_values_mismatched_period_dtype(self):
        # 创建一个日期范围，从 "2016-01-01" 开始，包含3个周期
        dti = date_range("2016-01-01", periods=3)
        # 用指定的数据类型 "Period[D]" 创建一个 Index 对象
        result = Index(dti, dtype="Period[D]")
        # 将 dti 转换为期望的 "D" 周期的 Period[D] 类型
        expected = dti.to_period("D")
        # 使用 pytest 的 assert_index_equal 断言方法比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "uint64"])
    # 定义一个测试方法，用于验证在 int64 或 uint64 数据类型且包含 NaN 值时的行为
    def test_constructor_int_dtype_nan_raises(self, dtype):
        # 创建包含 NaN 值的数据列表
        data = [np.nan]
        # 准备匹配的错误消息
        msg = "cannot convert"
        # 使用 pytest.raises 断言捕获 ValueError 异常，并匹配错误消息 msg
        with pytest.raises(ValueError, match=msg):
            # 使用指定的数据类型 dtype 创建 Index 对象，预期会触发 ValueError 异常
            Index(data, dtype=dtype)

    @pytest.mark.parametrize(
        "vals",
        [
            [1, 2, 3],  # 普通的整数列表
            np.array([1, 2, 3]),  # 整数类型的 NumPy 数组
            np.array([1, 2, 3], dtype=int),  # 指定为整数类型的 NumPy 数组
            # 下面的数据应该会被强制转换为 int 类型
            [1.0, 2.0, 3.0],  # 浮点数列表
            np.array([1.0, 2.0, 3.0], dtype=float),  # 浮点数类型的 NumPy 数组
        ],
    )
    # 定义一个测试方法，用于验证在不同输入数据和任意整数 NumPy 数据类型的情况下的行为
    def test_constructor_dtypes_to_int(self, vals, any_int_numpy_dtype):
        # 从参数中获取任意整数的 NumPy 数据类型
        dtype = any_int_numpy_dtype
        # 使用指定的数据类型 dtype 创建 Index 对象
        index = Index(vals, dtype=dtype)
        # 使用 assert 断言 index 对象的数据类型是否与指定的 dtype 一致
        assert index.dtype == dtype

    @pytest.mark.parametrize(
        "vals",
        [
            [1, 2, 3],  # 普通的整数列表
            [1.0, 2.0, 3.0],  # 浮点数列表
            np.array([1.0, 2.0, 3.0]),  # 浮点数类型的 NumPy 数组
            np.array([1, 2, 3], dtype=int),  # 整数类型的 NumPy 数组
            np.array([1.0, 2.0, 3.0], dtype=float),  # 指定为浮点数类型的 NumPy 数组
        ],
    )
    # 定义一个测试方法，用于验证在不同输入数据和任意浮点数 NumPy 数据类型的情况下的行为
    def test_constructor_dtypes_to_float(self, vals, float_numpy_dtype):
        # 从参数中获取任意浮点数的 NumPy 数据类型
        dtype = float_numpy_dtype
        # 使用指定的数据类型 dtype 创建 Index 对象
        index = Index(vals, dtype=dtype)
        # 使用 assert 断言 index 对象的数据类型是否与指定的 dtype 一致
        assert index.dtype == dtype

    @pytest.mark.parametrize(
        "vals",
        [
            [1, 2, 3],  # 普通的整数列表
            np.array([1, 2, 3], dtype=int),  # 整数类型的 NumPy 数组
            np.array(["2011-01-01", "2011-01-02"], dtype="datetime64[ns]"),  # datetime64 类型的 NumPy 数组
            [datetime(2011, 1, 1), datetime(2011, 1, 2)],  # Python datetime 对象列表
        ],
    )
    # 定义一个测试方法，用于验证在不同输入数据下以及将数据类型强制转换为分类类型时的行为
    def test_constructor_dtypes_to_categorical(self, vals):
        # 使用 "category" 数据类型创建 Index 对象
        index = Index(vals, dtype="category")
        # 使用 assert 断言 index 对象是否为 CategoricalIndex 类型
        assert isinstance(index, CategoricalIndex)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize(
        "vals",
        [
            np.array([np.datetime64("2011-01-01"), np.datetime64("2011-01-02")]),  # datetime64 类型的 NumPy 数组
            [datetime(2011, 1, 1), datetime(2011, 1, 2)],  # Python datetime 对象列表
        ],
    )
    # 定义一个测试方法，用于验证在不同输入数据下以及根据 cast_index 参数决定是否进行索引类型转换时的行为
    def test_constructor_dtypes_to_datetime(self, cast_index, vals):
        # 将 vals 转换为 Index 对象
        vals = Index(vals)
        if cast_index:
            # 如果 cast_index 为 True，则使用对象类型（object）创建 Index 对象
            index = Index(vals, dtype=object)
            # 使用 assert 断言 index 对象是否为 Index 类型，并且数据类型为 object
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            # 如果 cast_index 为 False，则直接创建 Index 对象
            index = Index(vals)
            # 使用 assert 断言 index 对象是否为 DatetimeIndex 类型
            assert isinstance(index, DatetimeIndex)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize(
        "vals",
        [
            np.array([np.timedelta64(1, "D"), np.timedelta64(1, "D")]),  # timedelta64 类型的 NumPy 数组
            [timedelta(1), timedelta(1)],  # Python timedelta 对象列表
        ],
    )
    # 测试函数：测试在将对象作为参数传递给 Index 构造函数时，构造不同类型的 Index 对象
    def test_constructor_dtypes_to_timedelta(self, cast_index, vals):
        # 如果 cast_index 为 True，则创建一个 dtype 为 object 的 Index 对象
        if cast_index:
            index = Index(vals, dtype=object)
            # 断言 index 是 Index 类的实例
            assert isinstance(index, Index)
            # 断言 index 的数据类型为 object
            assert index.dtype == object
        else:
            # 否则，创建一个普通的 Index 对象
            index = Index(vals)
            # 断言 index 是 TimedeltaIndex 类的实例
            assert isinstance(index, TimedeltaIndex)

    # 测试函数：测试将 TimedeltaIndex 对象作为参数传递给 Index 构造函数时的行为
    def test_pass_timedeltaindex_to_index(self):
        # 创建一个时间间隔范围
        rng = timedelta_range("1 days", "10 days")
        # 使用 dtype 为 object 创建一个 Index 对象
        idx = Index(rng, dtype=object)

        # 期望的 Index 对象，将时间间隔转换为 Python 原生 timedelta 类型
        expected = Index(rng.to_pytimedelta(), dtype=object)

        # 使用测试工具函数，断言 idx 和 expected 的值数组相等
        tm.assert_numpy_array_equal(idx.values, expected.values)

    # 测试函数：测试将 DatetimeIndex 对象作为参数传递给 Index 构造函数时的行为
    def test_pass_datetimeindex_to_index(self):
        # 创建一个日期范围
        rng = date_range("1/1/2000", "3/1/2000")
        # 使用 dtype 为 object 创建一个 Index 对象
        idx = Index(rng, dtype=object)

        # 期望的 Index 对象，将日期转换为 Python 原生 datetime 类型
        expected = Index(rng.to_pydatetime(), dtype=object)

        # 使用测试工具函数，断言 idx 和 expected 的值数组相等
        tm.assert_numpy_array_equal(idx.values, expected.values)
# 定义一个测试类 TestIndexConstructorUnwrapping，用于测试 pd.Index 的构造方式
class TestIndexConstructorUnwrapping:

    # 在参数化测试中，测试将不同的 arraylike 值传递给 pd.Index
    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_from_series_dt64(self, klass):
        # 创建时间戳列表 stamps
        stamps = [Timestamp("20110101"), Timestamp("20120101"), Timestamp("20130101")]
        # 创建预期的 DatetimeIndex 对象
        expected = DatetimeIndex(stamps)
        # 根据时间戳列表创建 Series 对象 ser
        ser = Series(stamps)
        # 使用 klass 构造函数创建索引 result
        result = klass(ser)
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

    # 测试不使用 pandas 数组直接构造索引
    def test_constructor_no_pandas_array(self):
        # 创建包含整数的 Series 对象 ser
        ser = Series([1, 2, 3])
        # 使用 Index 构造函数从 ser 的数组属性创建索引 result
        result = Index(ser.array)
        # 创建预期的 Index 对象 expected
        expected = Index([1, 2, 3])
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

    # 在参数化测试中，测试将 ndarray like 对象传递给 Index 构造函数
    @pytest.mark.parametrize(
        "array",
        [
            np.arange(5),
            np.array(["a", "b", "c"]),
            date_range("2000-01-01", periods=3).values,
        ],
    )
    def test_constructor_ndarray_like(self, array):
        # 用于模拟 ndarray like 对象的类 ArrayLike
        class ArrayLike:
            def __init__(self, array) -> None:
                self.array = array

            # 定义 __array__ 方法，返回包含 array 的 np.ndarray
            def __array__(self, dtype=None, copy=None) -> np.ndarray:
                return self.array

        # 使用 Index 构造函数从 ArrayLike 类的实例创建索引 result
        result = Index(ArrayLike(array))
        # 创建预期的 Index 对象 expected
        expected = Index(array)
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)


# 定义一个测试类 TestIndexConstructionErrors，用于测试 Index 构造时的错误处理
class TestIndexConstructionErrors:

    # 测试构造函数中的 int64 溢出情况
    def test_constructor_overflow_int64(self):
        # 设置错误信息
        msg = (
            "The elements provided in the data cannot "
            "all be casted to the dtype int64"
        )
        # 使用 pytest 的断言，期望抛出 OverflowError，并匹配特定的错误信息 msg
        with pytest.raises(OverflowError, match=msg):
            Index([np.iinfo(np.uint64).max - 1], dtype="int64")
```