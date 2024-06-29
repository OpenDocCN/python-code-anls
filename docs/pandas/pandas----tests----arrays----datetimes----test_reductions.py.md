# `D:\src\scipysrc\pandas\pandas\tests\arrays\datetimes\test_reductions.py`

```
# 导入需要的库
import numpy as np
import pytest  # 导入 pytest 库

from pandas.core.dtypes.dtypes import DatetimeTZDtype  # 导入 DatetimeTZDtype 类

import pandas as pd  # 导入 pandas 库
from pandas import NaT  # 导入 NaT（Not a Time）对象
import pandas._testing as tm  # 导入 pandas 测试模块
from pandas.core.arrays import DatetimeArray  # 导入 DatetimeArray 类


class TestReductions:
    @pytest.fixture
    def arr1d(self, tz_naive_fixture):
        """返回带参数化时区的 DatetimeArray 的 Fixture"""
        tz = tz_naive_fixture
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        # 创建 DatetimeArray 对象，从日期序列中生成
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-03",
                "2000-01-03",
                "NaT",
                "2000-01-02",
                "2000-01-05",
                "2000-01-04",
            ],
            dtype=dtype,
        )
        return arr

    def test_min_max(self, arr1d, unit):
        arr = arr1d
        arr = arr.as_unit(unit)  # 将 DatetimeArray 对象转换为指定单位的时间单位
        tz = arr.tz  # 获取 DatetimeArray 对象的时区信息

        # 测试 DatetimeArray 对象的最小值方法
        result = arr.min()
        expected = pd.Timestamp("2000-01-02", tz=tz).as_unit(unit)
        assert result == expected
        assert result.unit == expected.unit

        # 测试 DatetimeArray 对象的最大值方法
        result = arr.max()
        expected = pd.Timestamp("2000-01-05", tz=tz).as_unit(unit)
        assert result == expected
        assert result.unit == expected.unit

        # 测试在不跳过 NaN 值情况下的最小值方法
        result = arr.min(skipna=False)
        assert result is NaT

        # 测试在不跳过 NaN 值情况下的最大值方法
        result = arr.max(skipna=False)
        assert result is NaT

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_min_max_empty(self, skipna, tz):
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        # 创建一个空的 DatetimeArray 对象
        arr = DatetimeArray._from_sequence([], dtype=dtype)
        # 测试空 DatetimeArray 对象的最小值方法
        result = arr.min(skipna=skipna)
        assert result is NaT

        # 测试空 DatetimeArray 对象的最大值方法
        result = arr.max(skipna=skipna)
        assert result is NaT

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_median_empty(self, skipna, tz):
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        # 创建一个空的 DatetimeArray 对象
        arr = DatetimeArray._from_sequence([], dtype=dtype)
        # 测试空 DatetimeArray 对象的中位数方法
        result = arr.median(skipna=skipna)
        assert result is NaT

        # 将 DatetimeArray 对象重塑为 0 行 3 列的数组
        arr = arr.reshape(0, 3)
        # 测试在指定轴上的中位数方法，期望结果是 NaT 的 DatetimeArray 对象
        result = arr.median(axis=0, skipna=skipna)
        expected = type(arr)._from_sequence([NaT, NaT, NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected)

        # 测试在指定轴上的中位数方法，期望结果是空的 DatetimeArray 对象
        result = arr.median(axis=1, skipna=skipna)
        expected = type(arr)._from_sequence([], dtype=arr.dtype)
        tm.assert_equal(result, expected)

    def test_median(self, arr1d):
        arr = arr1d

        # 测试 DatetimeArray 对象的中位数方法
        result = arr.median()
        assert result == arr[0]
        # 测试在不跳过 NaN 值情况下的中位数方法
        result = arr.median(skipna=False)
        assert result is NaT

        # 测试在删除 NaN 值后的中位数方法，期望结果与数组的第一个元素相同
        result = arr.dropna().median(skipna=False)
        assert result == arr[0]

        # 测试在指定轴上的中位数方法，期望结果与数组的第一个元素相同
        result = arr.median(axis=0)
        assert result == arr[0]
    # 测试一维数组的中位数计算功能
    def test_median_axis(self, arr1d):
        # 将输入的一维数组赋值给变量 arr
        arr = arr1d
        # 断言：计算沿 axis=0 方向的中位数应与计算整体中位数结果相同
        assert arr.median(axis=0) == arr.median()
        # 断言：当 skipna=False 时，沿 axis=0 方向的中位数应为 NaT
        assert arr.median(axis=0, skipna=False) is NaT

        # 准备匹配的错误信息
        msg = r"abs\(axis\) must be less than ndim"
        # 使用 pytest 检查是否会引发 ValueError，匹配错误信息 msg
        with pytest.raises(ValueError, match=msg):
            arr.median(axis=1)

    # 标记 pytest 忽略 RuntimeWarning: All-NaN slice encountered
    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    # 测试二维数组的中位数计算功能
    def test_median_2d(self, arr1d):
        # 将输入的一维数组 arr1d 重塑为形状为 (1, -1) 的二维数组 arr
        arr = arr1d.reshape(1, -1)

        # axis = None
        # 断言：计算整体中位数应与原始一维数组的中位数结果相同
        assert arr.median() == arr1d.median()
        # 断言：当 skipna=False 时，计算整体中位数应为 NaT
        assert arr.median(skipna=False) is NaT

        # axis = 0
        # 计算沿 axis=0 方向的中位数
        result = arr.median(axis=0)
        expected = arr1d
        # 使用 pandas 测试工具 tm 进行结果比较
        tm.assert_equal(result, expected)

        # 由于第三列全为 NaT，无论是否 skipna，该位置的结果都应为 NaT
        result = arr.median(axis=0, skipna=False)
        expected = arr1d
        tm.assert_equal(result, expected)

        # axis = 1
        # 计算沿 axis=1 方向的中位数
        result = arr.median(axis=1)
        expected = type(arr)._from_sequence([arr1d.median()], dtype=arr.dtype)
        tm.assert_equal(result, expected)

        result = arr.median(axis=1, skipna=False)
        expected = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected)

    # 测试一维数组的均值计算功能
    def test_mean(self, arr1d):
        arr = arr1d

        # 手动验证结果
        expected = arr[0] + 0.4 * pd.Timedelta(days=1)

        # 计算整体均值并断言结果与预期相等
        result = arr.mean()
        assert result == expected
        # 计算整体均值，skipna=False 时结果应为 NaT
        result = arr.mean(skipna=False)
        assert result is NaT

        # 删除 NaN 值后再计算均值，skipna=False 时结果仍与整体均值相等
        result = arr.dropna().mean(skipna=False)
        assert result == expected

        # 计算沿 axis=0 方向的均值
        result = arr.mean(axis=0)
        assert result == expected

    # 测试二维数组的均值计算功能
    def test_mean_2d(self):
        # 创建一个时区为 US/Pacific 的日期时间索引
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        # 将日期时间索引的数据重塑为形状为 (3, 2) 的二维数组 dta
        dta = dti._data.reshape(3, 2)

        # 计算沿 axis=0 方向的均值
        result = dta.mean(axis=0)
        expected = dta[1]
        tm.assert_datetime_array_equal(result, expected)

        # 计算沿 axis=1 方向的均值
        result = dta.mean(axis=1)
        expected = dta[:, 0] + pd.Timedelta(hours=12)
        tm.assert_datetime_array_equal(result, expected)

        # 计算整体均值
        result = dta.mean(axis=None)
        expected = dti.mean()
        assert result == expected

    # 测试空数组的均值计算功能
    def test_mean_empty(self, arr1d, skipna):
        arr = arr1d[:0]

        # 空数组的均值应为 NaT
        assert arr.mean(skipna=skipna) is NaT

        # 将空数组 arr 重塑为形状为 (0, 3) 的二维数组 arr2d
        arr2d = arr.reshape(0, 3)

        # 计算沿 axis=0 方向的均值，skipna=skipna
        result = arr2d.mean(axis=0, skipna=skipna)
        expected = DatetimeArray._from_sequence([NaT, NaT, NaT], dtype=arr.dtype)
        tm.assert_datetime_array_equal(result, expected)

        # 计算沿 axis=1 方向的均值，skipna=skipna
        result = arr2d.mean(axis=1, skipna=skipna)
        expected = arr  # 即一维数组，为空
        tm.assert_datetime_array_equal(result, expected)

        # 计算整体均值，skipna=skipna
        result = arr2d.mean(axis=None, skipna=skipna)
        assert result is NaT
```