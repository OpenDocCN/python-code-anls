# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_npy_units.py`

```
# 导入 NumPy 库，使用 np 别名
import numpy as np

# 从 pandas 库中导入相关的模块和函数
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas._libs.tslibs.vectorized import is_date_array_normalized

# 创建一个 datetime64 类型的 ndarray，其中的日期已经标准化
day_arr = np.arange(10, dtype="i8").view("M8[D]")

# 定义一个测试类 TestIsDateArrayNormalized
class TestIsDateArrayNormalized:
    
    # 测试日期数组 day_arr 是否标准化
    def test_is_date_array_normalized_day(self):
        arr = day_arr
        abbrev = "D"
        # 将缩写转换为 NumPy 的单位表示
        unit = abbrev_to_npy_unit(abbrev)
        # 调用 is_date_array_normalized 函数，检查日期数组是否标准化
        result = is_date_array_normalized(arr.view("i8"), None, unit)
        # 断言结果为 True
        assert result is True

    # 测试秒级日期数组是否标准化
    def test_is_date_array_normalized_seconds(self):
        abbrev = "s"
        arr = day_arr.astype(f"M8[{abbrev}]")
        unit = abbrev_to_npy_unit(abbrev)
        # 调用 is_date_array_normalized 函数，检查日期数组是否标准化
        result = is_date_array_normalized(arr.view("i8"), None, unit)
        # 断言结果为 True
        assert result is True

        # 修改数组中第一个元素，增加 1 秒
        arr[0] += np.timedelta64(1, abbrev)
        # 再次调用 is_date_array_normalized 函数，检查日期数组是否标准化
        result2 = is_date_array_normalized(arr.view("i8"), None, unit)
        # 断言结果为 False
        assert result2 is False
```