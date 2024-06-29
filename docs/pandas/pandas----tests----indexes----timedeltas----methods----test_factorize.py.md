# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\methods\test_factorize.py`

```
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 从 pandas 库中导入以下模块
from pandas import (
    TimedeltaIndex,      # 导入 TimedeltaIndex 类
    factorize,           # 导入 factorize 函数
    timedelta_range,     # 导入 timedelta_range 函数
)

# 导入 pandas._testing 模块，并使用 tm 别名
import pandas._testing as tm

# 定义一个测试类 TestTimedeltaIndexFactorize
class TestTimedeltaIndexFactorize:
    # 定义测试方法 test_factorize
    def test_factorize(self):
        # 创建一个 TimedeltaIndex 对象 idx1，包含特定时间间隔字符串
        idx1 = TimedeltaIndex(["1 day", "1 day", "2 day", "2 day", "3 day", "3 day"])

        # 期望的结果数组 exp_arr，包含了 idx1 中各时间间隔字符串对应的索引
        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
        # 期望的 TimedeltaIndex 对象 exp_idx，包含了 idx1 中唯一的时间间隔字符串
        exp_idx = TimedeltaIndex(["1 day", "2 day", "3 day"])

        # 调用 idx1 的 factorize 方法，返回结果数组 arr 和索引对象 idx
        arr, idx = idx1.factorize()
        # 使用测试模块中的函数检查 arr 是否与期望的 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试模块中的函数检查 idx 是否与期望的 exp_idx 相等
        tm.assert_index_equal(idx, exp_idx)
        # 断言 idx 的频率与 exp_idx 的频率相等
        assert idx.freq == exp_idx.freq

        # 调用 idx1 的 factorize 方法，传入参数 sort=True，返回结果数组 arr 和索引对象 idx
        arr, idx = idx1.factorize(sort=True)
        # 使用测试模块中的函数检查 arr 是否与期望的 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试模块中的函数检查 idx 是否与期望的 exp_idx 相等
        tm.assert_index_equal(idx, exp_idx)
        # 断言 idx 的频率与 exp_idx 的频率相等
        assert idx.freq == exp_idx.freq

    # 定义测试方法 test_factorize_preserves_freq
    def test_factorize_preserves_freq(self):
        # 创建一个 timedelta_range 对象 idx3，从 "1 day" 开始，每秒钟创建一个时间点，共计 4 个时间点
        idx3 = timedelta_range("1 day", periods=4, freq="s")
        # 期望的结果数组 exp_arr，包含了 idx3 中各时间点对应的索引
        exp_arr = np.array([0, 1, 2, 3], dtype=np.intp)
        
        # 调用 idx3 的 factorize 方法，返回结果数组 arr 和索引对象 idx
        arr, idx = idx3.factorize()
        # 使用测试模块中的函数检查 arr 是否与期望的 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试模块中的函数检查 idx 是否与 idx3 相等
        tm.assert_index_equal(idx, idx3)
        # 断言 idx 的频率与 idx3 的频率相等
        assert idx.freq == idx3.freq

        # 调用 factorize 函数，传入 idx3 对象作为参数，返回结果数组 arr 和索引对象 idx
        arr, idx = factorize(idx3)
        # 使用测试模块中的函数检查 arr 是否与期望的 exp_arr 相等
        tm.assert_numpy_array_equal(arr, exp_arr)
        # 使用测试模块中的函数检查 idx 是否与 idx3 相等
        tm.assert_index_equal(idx, idx3)
        # 断言 idx 的频率与 idx3 的频率相等
        assert idx.freq == idx3.freq
```