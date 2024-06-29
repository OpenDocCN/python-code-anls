# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_factorize.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据

from pandas import PeriodIndex  # 从 pandas 库中导入 PeriodIndex 类，用于处理时间周期索引
import pandas._testing as tm  # 导入 pandas 测试模块，用于测试工具函数


class TestFactorize:
    def test_factorize_period(self):
        # 创建一个时间周期索引 idx1，包含了三个月份的数据，频率为每月 ("M")
        idx1 = PeriodIndex(
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
            freq="M",
        )

        # 期望的数组 exp_arr，包含了时间周期索引的因子化结果
        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
        # 期望的时间周期索引 exp_idx，包含了三个月份的数据，频率为每月 ("M")
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")

        # 调用 idx1 的 factorize() 方法进行因子化计算，返回结果数组 arr 和索引 idx
        arr, idx = idx1.factorize()
        # 使用测试工具函数 tm.assert_numpy_array_equal() 断言 arr 与 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试工具函数 tm.assert_index_equal() 断言 idx 与 exp_idx 相等
        tm.assert_index_equal(idx, exp_idx)

        # 调用 idx1 的 factorize(sort=True) 方法进行排序后的因子化计算
        arr, idx = idx1.factorize(sort=True)
        # 使用测试工具函数 tm.assert_numpy_array_equal() 断言 arr 与 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试工具函数 tm.assert_index_equal() 断言 idx 与 exp_idx 相等
        tm.assert_index_equal(idx, exp_idx)

    def test_factorize_period_nonmonotonic(self):
        # 创建一个时间周期索引 idx2，包含了三个月份的数据，但是顺序是非单调的
        idx2 = PeriodIndex(
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"],
            freq="M",
        )
        # 期望的时间周期索引 exp_idx，包含了三个月份的数据，频率为每月 ("M")
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")

        # 期望的数组 exp_arr，包含了排序后的时间周期索引的因子化结果
        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.intp)
        # 调用 idx2 的 factorize(sort=True) 方法进行排序后的因子化计算
        arr, idx = idx2.factorize(sort=True)
        # 使用测试工具函数 tm.assert_numpy_array_equal() 断言 arr 与 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试工具函数 tm.assert_index_equal() 断言 idx 与 exp_idx 相等
        tm.assert_index_equal(idx, exp_idx)

        # 期望的数组 exp_arr，包含了原始顺序的时间周期索引的因子化结果
        exp_arr = np.array([0, 0, 1, 2, 0, 2], dtype=np.intp)
        # 期望的时间周期索引 exp_idx，包含了三个月份的数据，频率为每月 ("M")，原始顺序
        exp_idx = PeriodIndex(["2014-03", "2014-02", "2014-01"], freq="M")
        # 调用 idx2 的 factorize() 方法进行原始顺序的因子化计算
        arr, idx = idx2.factorize()
        # 使用测试工具函数 tm.assert_numpy_array_equal() 断言 arr 与 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试工具函数 tm.assert_index_equal() 断言 idx 与 exp_idx 相等
        tm.assert_index_equal(idx, exp_idx)
```