# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_factorize.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入以下模块：
    DatetimeIndex,   # 日期时间索引类
    Index,           # 索引类
    date_range,      # 日期范围生成函数
    factorize,       # 因子化函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestDatetimeIndexFactorize:
    def test_factorize(self):
        idx1 = DatetimeIndex(  # 创建日期时间索引对象 idx1
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"]
        )

        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)  # 期望的因子化数组
        exp_idx = DatetimeIndex(["2014-01", "2014-02", "2014-03"])  # 期望的日期时间索引对象

        arr, idx = idx1.factorize()  # 对 idx1 进行因子化操作，返回因子化数组和索引对象
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证因子化后的数组是否与期望数组相等
        tm.assert_index_equal(idx, exp_idx)  # 断言：验证因子化后的索引对象是否与期望索引对象相等
        assert idx.freq == exp_idx.freq  # 断言：验证索引对象的频率是否与期望频率相等

        arr, idx = idx1.factorize(sort=True)  # 对 idx1 进行排序后的因子化操作
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证排序后因子化后的数组是否与期望数组相等
        tm.assert_index_equal(idx, exp_idx)  # 断言：验证排序后因子化后的索引对象是否与期望索引对象相等
        assert idx.freq == exp_idx.freq  # 断言：验证索引对象的频率是否与期望频率相等

        # 时区信息必须被保留
        idx1 = idx1.tz_localize("Asia/Tokyo")  # 将 idx1 本地化到 "Asia/Tokyo" 时区
        exp_idx = exp_idx.tz_localize("Asia/Tokyo")  # 将期望的索引对象本地化到 "Asia/Tokyo" 时区

        arr, idx = idx1.factorize()  # 再次对本地化后的 idx1 进行因子化操作
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证本地化后因子化后的数组是否与期望数组相等
        tm.assert_index_equal(idx, exp_idx)  # 断言：验证本地化后因子化后的索引对象是否与期望索引对象相等
        assert idx.freq == exp_idx.freq  # 断言：验证索引对象的频率是否与期望频率相等

        idx2 = DatetimeIndex(  # 创建新的日期时间索引对象 idx2
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"]
        )

        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.intp)  # 更新期望的因子化数组
        exp_idx = DatetimeIndex(["2014-01", "2014-02", "2014-03"])  # 更新期望的日期时间索引对象
        arr, idx = idx2.factorize(sort=True)  # 对 idx2 进行排序后的因子化操作
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证排序后因子化后的数组是否与期望数组相等
        tm.assert_index_equal(idx, exp_idx)  # 断言：验证排序后因子化后的索引对象是否与期望索引对象相等
        assert idx.freq == exp_idx.freq  # 断言：验证索引对象的频率是否与期望频率相等

        exp_arr = np.array([0, 0, 1, 2, 0, 2], dtype=np.intp)  # 更新期望的因子化数组
        exp_idx = DatetimeIndex(["2014-03", "2014-02", "2014-01"])  # 更新期望的日期时间索引对象
        arr, idx = idx2.factorize()  # 对 idx2 进行因子化操作
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证因子化后的数组是否与期望数组相等
        tm.assert_index_equal(idx, exp_idx)  # 断言：验证因子化后的索引对象是否与期望索引对象相等
        assert idx.freq == exp_idx.freq  # 断言：验证索引对象的频率是否与期望频率相等

    def test_factorize_preserves_freq(self):
        # GH#38120 freq should be preserved
        idx3 = date_range("2000-01", periods=4, freq="ME", tz="Asia/Tokyo")  # 创建包含时区信息的日期范围对象 idx3
        exp_arr = np.array([0, 1, 2, 3], dtype=np.intp)  # 期望的因子化数组

        arr, idx = idx3.factorize()  # 对 idx3 进行因子化操作
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证因子化后的数组是否与期望数组相等
        tm.assert_index_equal(idx, idx3)  # 断言：验证因子化后的索引对象是否与原始索引对象相等
        assert idx.freq == idx3.freq  # 断言：验证索引对象的频率是否与原始索引对象的频率相等

        arr, idx = factorize(idx3)  # 对 idx3 进行因子化操作（另一种方法）
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证因子化后的数组是否与期望数组相等
        tm.assert_index_equal(idx, idx3)  # 断言：验证因子化后的索引对象是否与原始索引对象相等
        assert idx.freq == idx3.freq  # 断言：验证索引对象的频率是否与原始索引对象的频率相等

    def test_factorize_tz(self, tz_naive_fixture, index_or_series):
        tz = tz_naive_fixture  # 获取时区 fixture
        # GH#13750
        base = date_range("2016-11-05", freq="h", periods=100, tz=tz)  # 创建带有时区信息的日期范围对象 base
        idx = base.repeat(5)  # 对 base 进行重复操作生成新的索引对象 idx

        exp_arr = np.arange(100, dtype=np.intp).repeat(5)  # 期望的因子化数组

        obj = index_or_series(idx)  # 根据 idx 创建索引或系列对象

        arr, res = obj.factorize()  # 对 obj 进行因子化操作，返回因子化数组和索引对象
        tm.assert_numpy_array_equal(arr, exp_arr)  # 断言：验证因子化后的数组是否与期望数组相等
        expected = base._with_freq(None)  # 获取 base 对象的频率为 None 的预期结果
        tm.assert_index_equal(res, expected)  # 断言：验证因子化后的索引对象是否与预期结果相等
        assert res.freq == expected.freq  # 断言：验证索引对象的频率是否与预期结果的频率相等
    # 定义测试方法，用于测试 factorize 方法的行为，对传入的索引或系列对象进行处理
    def test_factorize_dst(self, index_or_series):
        # GH#13750
        # 创建一个时间范围索引，从 "2016-11-06" 开始，每小时频率，共12个时间点，时区为 "US/Eastern"
        idx = date_range("2016-11-06", freq="h", periods=12, tz="US/Eastern")
        # 根据传入的索引或系列对象创建对象实例
        obj = index_or_series(idx)

        # 对对象调用 factorize 方法，获取返回的数组 arr 和结果 res
        arr, res = obj.factorize()
        # 断言 numpy 数组 arr 等于 [0, 1, 2, ..., 11]
        tm.assert_numpy_array_equal(arr, np.arange(12, dtype=np.intp))
        # 断言索引 res 等于初始创建的时间范围索引 idx
        tm.assert_index_equal(res, idx)
        # 如果传入的是 Index 类型，则断言结果索引 res 的频率与初始索引 idx 的频率相同
        if index_or_series is Index:
            assert res.freq == idx.freq

        # 重复以上步骤，使用不同的起始日期 "2016-06-13" 创建时间范围索引
        idx = date_range("2016-06-13", freq="h", periods=12, tz="US/Eastern")
        obj = index_or_series(idx)

        arr, res = obj.factorize()
        tm.assert_numpy_array_equal(arr, np.arange(12, dtype=np.intp))
        tm.assert_index_equal(res, idx)
        if index_or_series is Index:
            assert res.freq == idx.freq

    # 使用 pytest 的参数化装饰器，测试 factorize 方法在不同情况下的行为
    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize_no_freq_non_nano(self, tz_naive_fixture, sort):
        # GH#51978 case that does not go through the fastpath based on
        #  non-None freq
        # 获取时区信息的测试固件
        tz = tz_naive_fixture
        # 创建一个时间范围索引，从 "2016-11-06" 开始，每小时频率，共5个时间点，选择其中的特定顺序 [0, 4, 1, 3, 2]
        idx = date_range("2016-11-06", freq="h", periods=5, tz=tz)[[0, 4, 1, 3, 2]]
        # 对索引调用 factorize 方法，获取期望的编码和唯一值
        exp_codes, exp_uniques = idx.factorize(sort=sort)

        # 将索引转换为秒单位后再调用 factorize 方法，获取结果编码和唯一值
        res_codes, res_uniques = idx.as_unit("s").factorize(sort=sort)

        # 断言结果编码和唯一值与期望相同
        tm.assert_numpy_array_equal(res_codes, exp_codes)
        tm.assert_index_equal(res_uniques, exp_uniques.as_unit("s"))

        # 将索引转换为秒单位后转换为系列，再调用 factorize 方法，获取结果编码和唯一值
        res_codes, res_uniques = idx.as_unit("s").to_series().factorize(sort=sort)
        # 断言结果编码和唯一值与期望相同
        tm.assert_numpy_array_equal(res_codes, exp_codes)
        tm.assert_index_equal(res_uniques, exp_uniques.as_unit("s"))
```