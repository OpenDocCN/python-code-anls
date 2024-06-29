# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\methods\test_astype.py`

```
from datetime import timedelta  # 导入 timedelta 类，用于处理时间差

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 导入 Pandas 模块的特定部分
    Index,  # 索引对象
    NaT,  # 表示不确定或缺失的时间戳值
    Timedelta,  # 时间差对象
    TimedeltaIndex,  # 时间差索引对象
    timedelta_range,  # 创建时间差范围
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具
from pandas.core.arrays import TimedeltaArray  # 导入 Pandas 时间差数组对象


class TestTimedeltaIndex:  # 定义测试类 TestTimedeltaIndex
    def test_astype_object(self):  # 定义测试方法 test_astype_object
        idx = timedelta_range(start="1 days", periods=4, freq="D", name="idx")  # 创建时间差索引
        expected_list = [  # 预期的时间差列表
            Timedelta("1 days"),
            Timedelta("2 days"),
            Timedelta("3 days"),
            Timedelta("4 days"),
        ]
        result = idx.astype(object)  # 将时间差索引转换为 object 类型
        expected = Index(expected_list, dtype=object, name="idx")  # 创建预期的索引对象
        tm.assert_index_equal(result, expected)  # 使用测试工具比较结果和预期
        assert idx.tolist() == expected_list  # 断言时间差索引的列表形式与预期列表相同

    def test_astype_object_with_nat(self):  # 定义测试方法 test_astype_object_with_nat
        idx = TimedeltaIndex(  # 创建时间差索引对象
            [timedelta(days=1), timedelta(days=2), NaT, timedelta(days=4)], name="idx"
        )
        expected_list = [  # 预期的时间差列表
            Timedelta("1 days"),
            Timedelta("2 days"),
            NaT,
            Timedelta("4 days"),
        ]
        result = idx.astype(object)  # 将时间差索引转换为 object 类型
        expected = Index(expected_list, dtype=object, name="idx")  # 创建预期的索引对象
        tm.assert_index_equal(result, expected)  # 使用测试工具比较结果和预期
        assert idx.tolist() == expected_list  # 断言时间差索引的列表形式与预期列表相同

    def test_astype(self):  # 定义测试方法 test_astype
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan], name="idx")  # 创建时间差索引对象

        result = idx.astype(object)  # 将时间差索引转换为 object 类型
        expected = Index(  # 创建预期的索引对象
            [Timedelta("1 days 03:46:40")] + [NaT] * 3, dtype=object, name="idx"
        )
        tm.assert_index_equal(result, expected)  # 使用测试工具比较结果和预期

        result = idx.astype(np.int64)  # 将时间差索引转换为 np.int64 类型
        expected = Index(  # 创建预期的索引对象
            [100000000000000] + [-9223372036854775808] * 3, dtype=np.int64, name="idx"
        )
        tm.assert_index_equal(result, expected)  # 使用测试工具比较结果和预期

        result = idx.astype(str)  # 将时间差索引转换为字符串类型
        expected = Index([str(x) for x in idx], name="idx", dtype=object)  # 创建预期的索引对象
        tm.assert_index_equal(result, expected)  # 使用测试工具比较结果和预期

        rng = timedelta_range("1 days", periods=10)  # 创建时间差范围
        result = rng.astype("i8")  # 将时间差范围转换为 int64 类型
        tm.assert_index_equal(result, Index(rng.asi8))  # 使用测试工具比较结果和预期
        tm.assert_numpy_array_equal(rng.asi8, result.values)  # 使用测试工具比较 NumPy 数组是否相等

    def test_astype_uint(self):  # 定义测试方法 test_astype_uint
        arr = timedelta_range("1h", periods=2)  # 创建时间差范围

        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):  # 断言抛出特定类型错误异常
            arr.astype("uint64")  # 尝试将时间差范围转换为 uint64 类型
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):  # 断言抛出特定类型错误异常
            arr.astype("uint32")  # 尝试将时间差范围转换为 uint32 类型
    def test_astype_timedelta64(self):
        # 测试 timedelta64 数据类型转换功能
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan])

        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        # 断言 ValueError 异常被正确抛出，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            idx.astype("timedelta64")

        # 测试将 idx 转换为 timedelta64[ns] 是否成功
        result = idx.astype("timedelta64[ns]")
        tm.assert_index_equal(result, idx)
        # 断言 result 与 idx 不是同一个对象
        assert result is not idx

        # 测试将 idx 转换为 timedelta64[ns] 是否成功（使用 copy=False）
        result = idx.astype("timedelta64[ns]", copy=False)
        tm.assert_index_equal(result, idx)
        # 断言 result 与 idx 是同一个对象
        assert result is idx

    def test_astype_to_td64d_raises(self, index_or_series):
        # 测试将 timedelta64[ns] 转换为 timedelta64[D] 是否引发异常
        # 使用 index_or_series 函数创建包含 Timedelta 对象的数据结构 td
        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )
        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64\[D\]. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        # 断言 ValueError 异常被正确抛出，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            td.astype("timedelta64[D]")

    def test_astype_ms_to_s(self, index_or_series):
        # 测试将 m8[ns] 类型的数据转换为 m8[s] 类型的数据
        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )

        # 预期的 exp_values 是将 td 转换为 m8[s] 类型的 numpy 数组
        exp_values = np.asarray(td).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(exp_values, dtype=exp_values.dtype)
        expected = index_or_series(exp_tda)

        # 断言 result 与 expected 相等
        tm.assert_equal(result, expected)

    def test_astype_freq_conversion(self):
        # 测试 timedelta_range 转换为不同频率的 TimedeltaArray 对象
        tdi = timedelta_range("1 Day", periods=30)

        # 将 tdi 转换为 m8[s] 类型
        res = tdi.astype("m8[s]")
        exp_values = np.asarray(tdi).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(
            exp_values, dtype=exp_values.dtype, freq=tdi.freq
        )
        expected = Index(exp_tda)

        # 断言 res 与 expected 相等
        assert expected.dtype == "m8[s]"
        tm.assert_index_equal(res, expected)

        # 检查此转换是否与 Series 和 TimedeltaArray 匹配
        res = tdi._data.astype("m8[s]")
        tm.assert_equal(res, expected._values)

        res = tdi.to_series().astype("m8[s]")
        tm.assert_equal(res._values, expected._values._with_freq(None))

    @pytest.mark.parametrize("dtype", [float, "datetime64", "datetime64[ns]"])
    def test_astype_raises(self, dtype):
        # 测试将 TimedeltaIndex 转换为指定 dtype 是否引发 TypeError 异常
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan])
        msg = "Cannot cast TimedeltaIndex to dtype"
        # 断言 TypeError 异常被正确抛出，并匹配特定错误消息
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)
    # 定义一个测试方法，用于测试时间增量范围对象的类型转换为分类类型的功能
    def test_astype_category(self):
        # 创建一个时间增量范围对象，包含两个小时频率的时间增量
        obj = timedelta_range("1h", periods=2, freq="h")

        # 将时间增量范围对象转换为分类类型
        result = obj.astype("category")
        # 生成预期的分类索引对象，包含两个时间增量值
        expected = pd.CategoricalIndex([Timedelta("1h"), Timedelta("2h")])
        # 断言转换后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 将时间增量范围对象的底层数据转换为分类类型
        result = obj._data.astype("category")
        # 获取预期的分类数据，即预期分类索引对象的值数组
        expected = expected.values
        # 断言转换后的结果与预期结果相等
        tm.assert_categorical_equal(result, expected)

    # 定义一个测试方法，用于测试时间增量范围对象数据的数组类型转换功能
    def test_astype_array_fallback(self):
        # 创建一个时间增量范围对象，从"1h"开始，包含两个时间增量
        obj = timedelta_range("1h", periods=2)
        # 将时间增量范围对象的数据转换为布尔类型
        result = obj.astype(bool)
        # 生成预期的索引对象，包含布尔数组 [True, True]
        expected = Index(np.array([True, True]))
        # 断言转换后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 将时间增量范围对象的底层数据转换为布尔类型数组
        result = obj._data.astype(bool)
        # 生成预期的布尔数组 [True, True]
        expected = np.array([True, True])
        # 断言转换后的结果与预期结果相等
        tm.assert_numpy_array_equal(result, expected)
```