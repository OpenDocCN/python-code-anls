# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_pop.py`

```
# 从 pandas 库导入 Series 类
from pandas import Series
# 导入 pandas 测试模块
import pandas._testing as tm

# 定义一个测试函数 test_pop
def test_pop():
    # 注释: 测试 GH#6600 的特性
    # 创建一个 Series 对象，包含 [0, 4, 0] 数据，索引为 ["A", "B", "C"]，名称为 4
    ser = Series([0, 4, 0], index=["A", "B", "C"], name=4)

    # 弹出索引为 "B" 的元素，返回弹出的值 4
    result = ser.pop("B")
    # 断言弹出的结果为 4
    assert result == 4

    # 创建预期的 Series 对象，包含 [0, 0] 数据，索引为 ["A", "C"]，名称为 4
    expected = Series([0, 0], index=["A", "C"], name=4)
    # 断言 ser 和 expected Series 对象相等
    tm.assert_series_equal(ser, expected)
```