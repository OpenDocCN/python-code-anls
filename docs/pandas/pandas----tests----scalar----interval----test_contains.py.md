# `D:\src\scipysrc\pandas\pandas\tests\scalar\interval\test_contains.py`

```
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入以下对象：
    Interval,  # 区间对象
    Timedelta,  # 时间增量对象
    Timestamp,  # 时间戳对象
)


class TestContains:  # 定义测试类 TestContains
    def test_contains(self):  # 定义测试方法 test_contains
        interval = Interval(0, 1)  # 创建一个区间对象 [0, 1)
        assert 0.5 in interval  # 断言 0.5 在区间内
        assert 1 in interval  # 断言 1 在区间内
        assert 0 not in interval  # 断言 0 不在区间内

        interval_both = Interval(0, 1, "both")  # 创建一个两端闭合的区间对象 [0, 1]
        assert 0 in interval_both  # 断言 0 在区间内
        assert 1 in interval_both  # 断言 1 在区间内

        interval_neither = Interval(0, 1, closed="neither")  # 创建一个两端开放的区间对象 (0, 1)
        assert 0 not in interval_neither  # 断言 0 不在区间内
        assert 0.5 in interval_neither  # 断言 0.5 在区间内
        assert 1 not in interval_neither  # 断言 1 不在区间内

    def test_contains_interval(self, inclusive_endpoints_fixture):  # 定义测试方法 test_contains_interval，使用 fixture inclusive_endpoints_fixture
        interval1 = Interval(0, 1, "both")  # 创建一个两端闭合的区间对象 [0, 1]
        interval2 = Interval(0, 1, inclusive_endpoints_fixture)  # 使用 fixture 创建另一个区间对象
        assert interval1 in interval1  # 断言区间对象 interval1 包含自身
        assert interval2 in interval2  # 断言区间对象 interval2 包含自身
        assert interval2 in interval1  # 断言区间对象 interval1 包含 interval2
        assert interval1 not in interval2 or inclusive_endpoints_fixture == "both"  # 断言 interval1 不在 interval2 中或 inclusive_endpoints_fixture 为 "both"

    def test_contains_infinite_length(self):  # 定义测试方法 test_contains_infinite_length
        interval1 = Interval(0, 1, "both")  # 创建一个两端闭合的区间对象 [0, 1]
        interval2 = Interval(float("-inf"), float("inf"), "neither")  # 创建一个无限长区间对象 (-∞, +∞)
        assert interval1 in interval2  # 断言 interval1 在 interval2 中
        assert interval2 not in interval1  # 断言 interval2 不在 interval1 中

    def test_contains_zero_length(self):  # 定义测试方法 test_contains_zero_length
        interval1 = Interval(0, 1, "both")  # 创建一个两端闭合的区间对象 [0, 1]
        interval2 = Interval(-1, -1, "both")  # 创建一个长度为零的区间对象 [-1, -1]
        interval3 = Interval(0.5, 0.5, "both")  # 创建一个长度为零的区间对象 [0.5, 0.5]
        assert interval2 not in interval1  # 断言 interval2 不在 interval1 中
        assert interval3 in interval1  # 断言 interval3 在 interval1 中
        assert interval2 not in interval3 and interval3 not in interval2  # 断言 interval2 不在 interval3 中且 interval3 不在 interval2 中
        assert interval1 not in interval2 and interval1 not in interval3  # 断言 interval1 不在 interval2 中且 interval1 不在 interval3 中

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，参数类型为 type1
        "type1",  # 参数名为 type1
        [  # 参数取值列表
            (0, 1),  # 第一组参数 (0, 1)
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),  # 第二组参数（时间戳对象）
            (Timedelta("0h"), Timedelta("1h")),  # 第三组参数（时间增量对象）
        ],
    )
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，参数类型为 type2
        "type2",  # 参数名为 type2
        [  # 参数取值列表，与 type1 类似
            (0, 1),
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),
            (Timedelta("0h"), Timedelta("1h")),
        ],
    )
    def test_contains_mixed_types(self, type1, type2):  # 定义测试方法 test_contains_mixed_types，使用 type1 和 type2 作为参数
        interval1 = Interval(*type1)  # 根据 type1 创建一个区间对象
        interval2 = Interval(*type2)  # 根据 type2 创建一个区间对象
        if type1 == type2:  # 如果 type1 等于 type2
            assert interval1 in interval2  # 断言 interval1 在 interval2 中
        else:  # 否则
            msg = "^'<=' not supported between instances of"
            with pytest.raises(TypeError, match=msg):  # 使用 pytest 断言捕获预期的 TypeError 异常
                interval1 in interval2  # 断言 interval1 在 interval2 中
```