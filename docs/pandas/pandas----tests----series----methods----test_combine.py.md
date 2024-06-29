# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_combine.py`

```
from pandas import Series  # 导入 Series 类
import pandas._testing as tm  # 导入 pandas 测试模块


class TestCombine:
    def test_combine_scalar(self):
        # GH#21248
        # 注意 - combine() 与另一个 Series 的组合在其他地方进行了测试，
        # 因为它在测试运算符时被使用
        # 创建一个包含 0 到 4 的 Series，每个元素乘以 10
        ser = Series([i * 10 for i in range(5)])
        # 使用 combine 方法将每个元素与标量 3 进行加法运算
        result = ser.combine(3, lambda x, y: x + y)
        # 创建预期的 Series，每个元素乘以 10 后加上 3
        expected = Series([i * 10 + 3 for i in range(5)])
        # 断言 result 和 expected 的 Series 相等
        tm.assert_series_equal(result, expected)

        # 使用 combine 方法将每个元素与标量 22 进行比较取最小值
        result = ser.combine(22, lambda x, y: min(x, y))
        # 创建预期的 Series，每个元素乘以 10 后与 22 比较取最小值
        expected = Series([min(i * 10, 22) for i in range(5)])
        # 断言 result 和 expected 的 Series 相等
        tm.assert_series_equal(result, expected)
```