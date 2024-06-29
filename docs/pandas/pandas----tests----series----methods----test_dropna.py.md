# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_dropna.py`

```
    # 导入 numpy 库，并使用 np 作为别名
    import numpy as np
    # 导入 pytest 库，用于编写和运行测试
    import pytest

    # 从 pandas 库中导入多个类和函数
    from pandas import (
        DatetimeIndex,    # 日期时间索引类
        IntervalIndex,    # 区间索引类
        NaT,              # "Not a Time" 表示缺失日期时间值的特殊对象
        Period,           # 时间段类
        Series,           # 序列类，类似于数组，但具有标签
        Timestamp,        # 时间戳类
    )
    # 导入 pandas 内部测试工具模块
    import pandas._testing as tm

    # 定义一个测试类 TestDropna
    class TestDropna:
        # 测试空序列的 dropna 方法
        def test_dropna_empty(self):
            # 创建一个空的 Series 对象
            ser = Series([], dtype=object)

            # 断言经过 dropna 后序列长度为0
            assert len(ser.dropna()) == 0
            # 使用 inplace=True 参数调用 dropna 方法，验证返回值为 None
            return_value = ser.dropna(inplace=True)
            assert return_value is None
            # 再次断言经过 dropna(inplace=True) 后序列长度为0
            assert len(ser) == 0

            # 对于无效的 axis 参数，验证抛出 ValueError 异常
            msg = "No axis named 1 for object type Series"
            with pytest.raises(ValueError, match=msg):
                ser.dropna(axis=1)

        # 测试保留序列名称的 dropna 方法
        def test_dropna_preserve_name(self, datetime_series):
            # 将 datetime_series 的前五个值设为 NaN
            datetime_series[:5] = np.nan
            # 调用 dropna 方法
            result = datetime_series.dropna()
            # 断言结果序列的名称与原始序列相同
            assert result.name == datetime_series.name
            # 保存原始序列的名称
            name = datetime_series.name
            # 复制原始序列
            ts = datetime_series.copy()
            # 使用 inplace=True 参数调用 dropna 方法，验证返回值为 None
            return_value = ts.dropna(inplace=True)
            assert return_value is None
            # 再次断言处理后序列的名称与原始序列相同
            assert ts.name == name

        # 测试无 NaN 值的 dropna 方法
        def test_dropna_no_nan(self):
            # 遍历不同类型的序列
            for ser in [
                Series([1, 2, 3], name="x"),              # 整数序列
                Series([False, True, False], name="x"),   # 布尔序列
            ]:
                # 调用 dropna 方法
                result = ser.dropna()
                # 使用测试工具方法检查结果序列与原始序列相等
                tm.assert_series_equal(result, ser)
                # 断言结果序列与原始序列不是同一个对象
                assert result is not ser

                # 复制原始序列
                s2 = ser.copy()
                # 使用 inplace=True 参数调用 dropna 方法，验证返回值为 None
                return_value = s2.dropna(inplace=True)
                assert return_value is None
                # 再次使用测试工具方法检查处理后序列与原始序列相等
                tm.assert_series_equal(s2, ser)

        # 测试包含区间索引的 dropna 方法
        def test_dropna_intervals(self):
            # 创建一个包含区间索引的 Series 对象
            ser = Series(
                [np.nan, 1, 2, 3],
                IntervalIndex.from_arrays([np.nan, 0, 1, 2], [np.nan, 1, 2, 3]),
            )

            # 调用 dropna 方法
            result = ser.dropna()
            # 期望的结果是去除 NaN 值后的 Series 对象
            expected = ser.iloc[1:]
            # 使用测试工具方法检查结果与期望结果是否相等
            tm.assert_series_equal(result, expected)

        # 测试包含时间段数据类型的 dropna 方法
        def test_dropna_period_dtype(self):
            # 创建一个包含时间段对象的 Series 对象
            ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])
            # 调用 dropna 方法
            result = ser.dropna()
            # 期望的结果是去除 NaN 值后的 Series 对象
            expected = Series([Period("2011-01", freq="M")])

            # 使用测试工具方法检查结果与期望结果是否相等
            tm.assert_series_equal(result, expected)
    # 定义一个测试方法，测试处理 datetime64 类型数据的 dropna 方法，参数 unit 为时间单位
    def test_datetime64_tz_dropna(self, unit):
        # 创建一个 Series 对象，包含多个时间戳和 NaT（Not a Time）值，数据类型为指定时间单位的 datetime-like
        ser = Series(
            [
                Timestamp("2011-01-01 10:00"),
                NaT,
                Timestamp("2011-01-03 10:00"),
                NaT,
            ],
            dtype=f"M8[{unit}]",
        )
        # 调用 dropna 方法，删除 Series 中的 NaN 值，得到处理后的结果
        result = ser.dropna()
        # 创建预期的 Series 对象，只包含非 NaN 值，数据类型与原始数据相同
        expected = Series(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-03 10:00")],
            index=[0, 2],
            dtype=f"M8[{unit}]",
        )
        # 使用测试工具函数验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个 DatetimeIndex 对象，包含多个时间戳和 NaT 值，带有时区信息
        idx = DatetimeIndex(
            ["2011-01-01 10:00", NaT, "2011-01-03 10:00", NaT], tz="Asia/Tokyo"
        ).as_unit(unit)
        # 创建一个 Series 对象，使用上述 DatetimeIndex 作为数据源
        ser = Series(idx)
        # 断言 Series 的数据类型是否符合预期，包含指定单位和亚洲/东京时区信息
        assert ser.dtype == f"datetime64[{unit}, Asia/Tokyo]"
        # 调用 dropna 方法，删除 Series 中的 NaN 值，得到处理后的结果
        result = ser.dropna()
        # 创建预期的 Series 对象，只包含非 NaN 值，数据类型与原始数据相同
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-03 10:00", tz="Asia/Tokyo"),
            ],
            index=[0, 2],
            dtype=f"datetime64[{unit}, Asia/Tokyo]",
        )
        # 断言处理后的结果的数据类型是否符合预期
        assert result.dtype == f"datetime64[{unit}, Asia/Tokyo]"
        # 使用测试工具函数验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 标记此测试方法，参数化测试 val 取值为 1 和 1.5
    @pytest.mark.parametrize("val", [1, 1.5])
    def test_dropna_ignore_index(self, val):
        # 创建一个 Series 对象，包含整数和浮点数值，指定索引
        ser = Series([1, 2, val], index=[3, 2, 1])
        # 调用 dropna 方法，删除 Series 中的 NaN 值，并忽略索引重排，得到处理后的结果
        result = ser.dropna(ignore_index=True)
        # 创建预期的 Series 对象，只包含非 NaN 值，索引重排后的结果
        expected = Series([1, 2, val])
        # 使用测试工具函数验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 在原始 Series 上直接调用 dropna 方法，删除 NaN 值，并在原地修改
        ser.dropna(ignore_index=True, inplace=True)
        # 使用测试工具函数验证修改后的 ser 是否与 expected 相等
        tm.assert_series_equal(ser, expected)
```