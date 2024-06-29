# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_equals.py`

```
import numpy as np  # 导入 NumPy 库，用于数组操作

from pandas import (  # 从 Pandas 库中导入以下模块：
    IntervalIndex,   # 区间索引模块
    date_range,      # 日期范围生成模块
)


class TestEquals:
    def test_equals(self, closed):
        expected = IntervalIndex.from_breaks(np.arange(5), closed=closed)  # 创建一个预期的区间索引对象
        assert expected.equals(expected)  # 断言：预期的对象应该等于自身
        assert expected.equals(expected.copy())  # 断言：预期的对象应该等于其复制品

        assert not expected.equals(expected.astype(object))  # 断言：预期的对象不应该等于类型转换为对象类型的对象
        assert not expected.equals(np.array(expected))  # 断言：预期的对象不应该等于转换为 NumPy 数组的对象
        assert not expected.equals(list(expected))  # 断言：预期的对象不应该等于转换为列表的对象

        assert not expected.equals([1, 2])  # 断言：预期的对象不应该等于列表 [1, 2]
        assert not expected.equals(np.array([1, 2]))  # 断言：预期的对象不应该等于 NumPy 数组 [1, 2]
        assert not expected.equals(date_range("20130101", periods=2))  # 断言：预期的对象不应该等于生成的日期范围对象

        expected_name1 = IntervalIndex.from_breaks(
            np.arange(5), closed=closed, name="foo"
        )  # 创建一个具有命名的预期区间索引对象
        expected_name2 = IntervalIndex.from_breaks(
            np.arange(5), closed=closed, name="bar"
        )  # 创建另一个具有不同命名的预期区间索引对象
        assert expected.equals(expected_name1)  # 断言：预期的对象应该等于具有相同数据但不同名称的对象
        assert expected_name1.equals(expected_name2)  # 断言：具有不同名称但相同数据的两个对象应该相等

        for other_closed in {"left", "right", "both", "neither"} - {closed}:
            expected_other_closed = IntervalIndex.from_breaks(
                np.arange(5), closed=other_closed
            )  # 创建具有不同关闭方式的预期区间索引对象
            assert not expected.equals(expected_other_closed)  # 断言：预期的对象不应该等于具有不同关闭方式的对象
```