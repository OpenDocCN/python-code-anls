# `D:\src\scipysrc\pandas\pandas\tests\scalar\interval\test_formats.py`

```
# 从 pandas 库中导入 Interval 类，用于表示数值区间
from pandas import Interval


# 定义测试函数，用于测试 Interval 类的字符串表示方法
def test_interval_repr():
    # 创建一个闭区间 [0, 1) 的 Interval 对象
    interval = Interval(0, 1)
    # 断言 Interval 对象的字符串表示应为 "Interval(0, 1, closed='right')"
    assert repr(interval) == "Interval(0, 1, closed='right')"
    # 断言 Interval 对象的 str 方法返回值应为 "(0, 1]"
    assert str(interval) == "(0, 1]"

    # 创建一个左闭右开区间 [0, 1) 的 Interval 对象
    interval_left = Interval(0, 1, closed="left")
    # 断言 Interval 对象的字符串表示应为 "Interval(0, 1, closed='left')"
    assert repr(interval_left) == "Interval(0, 1, closed='left')"
    # 断言 Interval 对象的 str 方法返回值应为 "[0, 1)"
    assert str(interval_left) == "[0, 1)"
```