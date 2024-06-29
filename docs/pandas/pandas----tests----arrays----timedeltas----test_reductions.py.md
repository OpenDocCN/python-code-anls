# `D:\src\scipysrc\pandas\pandas\tests\arrays\timedeltas\test_reductions.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，并简写为 pd
from pandas import Timedelta  # 从 Pandas 中导入 Timedelta 类
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.core import nanops  # 导入 Pandas 核心模块中的 nanops
from pandas.core.arrays import TimedeltaArray  # 从 Pandas 核心数组中导入 TimedeltaArray 类


class TestReductions:
    @pytest.mark.parametrize("name", ["std", "min", "max", "median", "mean"])
    def test_reductions_empty(self, name, skipna):
        tdi = pd.TimedeltaIndex([])  # 创建空的 TimedeltaIndex 对象
        arr = tdi.array  # 从 TimedeltaIndex 中获取其数组表示

        result = getattr(tdi, name)(skipna=skipna)  # 调用指定的统计方法（如 std、min 等），跳过 NaN 值
        assert result is pd.NaT  # 断言结果为 NaT（Not a Time）

        result = getattr(arr, name)(skipna=skipna)  # 在数组上调用相同的统计方法
        assert result is pd.NaT  # 断言结果为 NaT

    def test_sum_empty(self, skipna):
        tdi = pd.TimedeltaIndex([])  # 创建空的 TimedeltaIndex 对象
        arr = tdi.array  # 获取其数组表示

        result = tdi.sum(skipna=skipna)  # 计算 TimedeltaIndex 的总和，跳过 NaN 值
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == Timedelta(0)  # 断言结果等于零时间间隔

        result = arr.sum(skipna=skipna)  # 在数组上进行相同的总和计算
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == Timedelta(0)  # 断言结果等于零时间间隔

    def test_min_max(self, unit):
        dtype = f"m8[{unit}]"  # 根据给定的单位创建 Timedelta 类型的 dtype
        arr = TimedeltaArray._from_sequence(
            ["3h", "3h", "NaT", "2h", "5h", "4h"], dtype=dtype
        )  # 使用给定的 dtype 创建 TimedeltaArray 对象

        result = arr.min()  # 计算数组中的最小值
        expected = Timedelta("2h")  # 预期的最小时间间隔为 2 小时
        assert result == expected  # 断言结果与预期相同

        result = arr.max()  # 计算数组中的最大值
        expected = Timedelta("5h")  # 预期的最大时间间隔为 5 小时
        assert result == expected  # 断言结果与预期相同

        result = arr.min(skipna=False)  # 在不跳过 NaN 值的情况下计算最小值
        assert result is pd.NaT  # 断言结果为 NaT

        result = arr.max(skipna=False)  # 在不跳过 NaN 值的情况下计算最大值
        assert result is pd.NaT  # 断言结果为 NaT

    def test_sum(self):
        tdi = pd.TimedeltaIndex(["3h", "3h", "NaT", "2h", "5h", "4h"])  # 创建 TimedeltaIndex 对象
        arr = tdi.array  # 获取其数组表示

        result = arr.sum(skipna=True)  # 计算数组的总和，跳过 NaN 值
        expected = Timedelta(hours=17)  # 预期的总和为 17 小时
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == expected  # 断言结果与预期相同

        result = tdi.sum(skipna=True)  # 计算 TimedeltaIndex 的总和，跳过 NaN 值
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == expected  # 断言结果与预期相同

        result = arr.sum(skipna=False)  # 在不跳过 NaN 值的情况下计算数组的总和
        assert result is pd.NaT  # 断言结果为 NaT

        result = tdi.sum(skipna=False)  # 在不跳过 NaN 值的情况下计算 TimedeltaIndex 的总和
        assert result is pd.NaT  # 断言结果为 NaT

        result = arr.sum(min_count=9)  # 计算数组的总和，要求至少有 9 个非 NaN 值
        assert result is pd.NaT  # 断言结果为 NaT

        result = tdi.sum(min_count=9)  # 计算 TimedeltaIndex 的总和，要求至少有 9 个非 NaN 值
        assert result is pd.NaT  # 断言结果为 NaT

        result = arr.sum(min_count=1)  # 计算数组的总和，要求至少有 1 个非 NaN 值
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == expected  # 断言结果与预期相同

        result = tdi.sum(min_count=1)  # 计算 TimedeltaIndex 的总和，要求至少有 1 个非 NaN 值
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == expected  # 断言结果与预期相同

    def test_npsum(self):
        # GH#25282, GH#25335 np.sum should return a Timedelta, not timedelta64
        tdi = pd.TimedeltaIndex(["3h", "3h", "2h", "5h", "4h"])  # 创建 TimedeltaIndex 对象
        arr = tdi.array  # 获取其数组表示

        result = np.sum(tdi)  # 使用 NumPy 计算 TimedeltaIndex 的总和
        expected = Timedelta(hours=17)  # 预期的总和为 17 小时
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == expected  # 断言结果与预期相同

        result = np.sum(arr)  # 使用 NumPy 计算数组的总和
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result == expected  # 断言结果与预期相同
    # 测试函数，计算二维数组的元素和，不跳过 NaN 值
    def test_sum_2d_skipna_false(self):
        # 创建一个二维 NumPy 数组，转换为 TimedeltaArray
        arr = np.arange(8).astype(np.int64).view("m8[s]").astype("m8[ns]").reshape(4, 2)
        # 将最后一个元素设为 "Nat"，表示不可用的时间增量
        arr[-1, -1] = "Nat"

        # 从序列创建 TimedeltaArray 对象
        tda = TimedeltaArray._from_sequence(arr)

        # 计算 TimedeltaArray 的总和，不跳过 NaN 值
        result = tda.sum(skipna=False)
        assert result is pd.NaT

        # 沿着 axis=0 计算 TimedeltaArray 的总和，不跳过 NaN 值
        result = tda.sum(axis=0, skipna=False)
        expected = pd.TimedeltaIndex([Timedelta(seconds=12), pd.NaT])._values
        tm.assert_timedelta_array_equal(result, expected)

        # 沿着 axis=1 计算 TimedeltaArray 的总和，不跳过 NaN 值
        result = tda.sum(axis=1, skipna=False)
        expected = pd.TimedeltaIndex(
            [
                Timedelta(seconds=1),
                Timedelta(seconds=5),
                Timedelta(seconds=9),
                pd.NaT,
            ]
        )._values
        tm.assert_timedelta_array_equal(result, expected)

    # 添加 Timestamp 后，测试 DatetimeArray 的标准差
    @pytest.mark.parametrize(
        "add",
        [
            Timedelta(0),
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-01-01", tz="Asia/Tokyo"),
        ],
    )
    def test_std(self, add):
        # 创建 TimedeltaIndex 对象并加上指定的 Timestamp
        tdi = pd.TimedeltaIndex(["0h", "4h", "NaT", "4h", "0h", "2h"]) + add
        # 获取其内部的数组
        arr = tdi.array

        # 计算数组的标准差，跳过 NaN 值
        result = arr.std(skipna=True)
        expected = Timedelta(hours=2)
        assert isinstance(result, Timedelta)
        assert result == expected

        # 计算 TimedeltaIndex 的标准差，跳过 NaN 值
        result = tdi.std(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        # 如果数组没有时区信息，则使用 NumPy 进行标准差计算，跳过 NaN 值
        if getattr(arr, "tz", None) is None:
            result = nanops.nanstd(np.asarray(arr), skipna=True)
            assert isinstance(result, np.timedelta64)
            assert result == expected

        # 计算数组的标准差，不跳过 NaN 值
        result = arr.std(skipna=False)
        assert result is pd.NaT

        # 计算 TimedeltaIndex 的标准差，不跳过 NaN 值
        result = tdi.std(skipna=False)
        assert result is pd.NaT

        # 如果数组没有时区信息，则使用 NumPy 进行标准差计算，不跳过 NaN 值
        if getattr(arr, "tz", None) is None:
            result = nanops.nanstd(np.asarray(arr), skipna=False)
            assert isinstance(result, np.timedelta64)
            assert np.isnat(result)

    # 测试 TimedeltaIndex 数组的中位数计算
    def test_median(self):
        # 创建 TimedeltaIndex 对象
        tdi = pd.TimedeltaIndex(["0h", "3h", "NaT", "5h06m", "0h", "2h"])
        # 获取其内部的数组
        arr = tdi.array

        # 计算数组的中位数，跳过 NaN 值
        result = arr.median(skipna=True)
        expected = Timedelta(hours=2)
        assert isinstance(result, Timedelta)
        assert result == expected

        # 计算 TimedeltaIndex 的中位数，跳过 NaN 值
        result = tdi.median(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        # 计算数组的中位数，不跳过 NaN 值
        result = arr.median(skipna=False)
        assert result is pd.NaT

        # 计算 TimedeltaIndex 的中位数，不跳过 NaN 值
        result = tdi.median(skipna=False)
        assert result is pd.NaT
    # 定义测试函数，用于测试 TimedeltaIndex 类的均值计算功能
    def test_mean(self):
        # 创建一个 TimedeltaIndex 对象，包含时间增量字符串和 NaT（Not a Time）值
        tdi = pd.TimedeltaIndex(["0h", "3h", "NaT", "5h06m", "0h", "2h"])
        # 获取该 TimedeltaIndex 对象内部的底层数组
        arr = tdi._data

        # 手动验证的预期结果，创建一个 Timedelta 对象，其值为数组中非空值的均值
        expected = Timedelta(arr.dropna()._ndarray.mean())

        # 计算数组的均值
        result = arr.mean()
        # 断言计算结果是否与预期相等
        assert result == expected

        # 计算数组的均值，不跳过 NaN 值（NaT）
        result = arr.mean(skipna=False)
        # 断言结果是否为 pd.NaT
        assert result is pd.NaT

        # 计算数组去除 NaN 值后的均值，不跳过 NaN 值
        result = arr.dropna().mean(skipna=False)
        # 断言计算结果是否与预期相等
        assert result == expected

        # 沿着 axis=0 方向计算数组的均值
        result = arr.mean(axis=0)
        # 断言计算结果是否与预期相等
        assert result == expected

    # 定义测试函数，用于测试二维时间增量数组的均值计算功能
    def test_mean_2d(self):
        # 创建一个时间增量范围，包含从 "14 days" 开始的 6 个时间增量
        tdi = pd.timedelta_range("14 days", periods=6)
        # 将时间增量数组重塑为 3x2 的二维数组
        tda = tdi._data.reshape(3, 2)

        # 沿着 axis=0 方向计算二维数组的均值
        result = tda.mean(axis=0)
        # 预期的结果为第二行的元素
        expected = tda[1]
        # 使用测试工具函数断言时间增量数组的均值结果与预期结果相等
        tm.assert_timedelta_array_equal(result, expected)

        # 沿着 axis=1 方向计算二维数组的均值
        result = tda.mean(axis=1)
        # 预期的结果为第一列加上 12 小时的时间增量
        expected = tda[:, 0] + Timedelta(hours=12)
        # 使用测试工具函数断言时间增量数组的均值结果与预期结果相等
        tm.assert_timedelta_array_equal(result, expected)

        # 计算二维数组所有元素的均值（展平数组）
        result = tda.mean(axis=None)
        # 使用 TimedeltaIndex 对象的均值作为预期结果
        expected = tdi.mean()
        # 断言计算结果是否与预期相等
        assert result == expected
```