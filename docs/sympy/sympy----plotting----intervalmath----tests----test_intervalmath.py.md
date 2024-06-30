# `D:\src\scipysrc\sympy\sympy\plotting\intervalmath\tests\test_intervalmath.py`

```
from sympy.plotting.intervalmath import interval  # 导入 interval 函数从 intervalmath 模块
from sympy.testing.pytest import raises  # 导入 raises 函数从 pytest 模块


def test_interval():  # 定义测试函数 test_interval，用于测试 interval 函数的不同情况
    assert (interval(1, 1) == interval(1, 1, is_valid=True)) == (True, True)  # 断言判断两个 interval 对象是否相等
    assert (interval(1, 1) == interval(1, 1, is_valid=False)) == (True, False)  # 断言判断两个 interval 对象是否相等
    assert (interval(1, 1) == interval(1, 1, is_valid=None)) == (True, None)  # 断言判断两个 interval 对象是否相等
    assert (interval(1, 1.5) == interval(1, 2)) == (None, True)  # 断言判断两个 interval 对象是否相等
    assert (interval(0, 1) == interval(2, 3)) == (False, True)  # 断言判断两个 interval 对象是否相等
    assert (interval(0, 1) == interval(1, 2)) == (None, True)  # 断言判断两个 interval 对象是否相等
    assert (interval(1, 2) != interval(1, 2)) == (False, True)  # 断言判断两个 interval 对象是否不相等
    assert (interval(1, 3) != interval(2, 3)) == (None, True)  # 断言判断两个 interval 对象是否不相等
    assert (interval(1, 3) != interval(-5, -3)) == (True, True)  # 断言判断两个 interval 对象是否不相等
    assert (interval(1, 3, is_valid=False) != interval(-5, -3)) == (True, False)  # 断言判断两个 interval 对象是否不相等
    assert (interval(1, 3, is_valid=None) != interval(-5, 3)) == (None, None)  # 断言判断两个 interval 对象是否不相等
    assert (interval(4, 4) != 4) == (False, True)  # 断言判断 interval 对象与数值是否不相等
    assert (interval(1, 1) == 1) == (True, True)  # 断言判断 interval 对象与数值是否相等
    assert (interval(1, 3, is_valid=False) == interval(1, 3)) == (True, False)  # 断言判断两个 interval 对象是否相等
    assert (interval(1, 3, is_valid=None) == interval(1, 3)) == (True, None)  # 断言判断两个 interval 对象是否相等
    inter = interval(-5, 5)  # 创建一个 interval 对象 inter，表示从 -5 到 5 的区间
    assert (interval(inter) == interval(-5, 5)) == (True, True)  # 断言判断 interval 对象的复制是否成功
    assert inter.width == 10  # 断言判断 interval 对象的宽度是否为 10
    assert 0 in inter  # 断言判断数值 0 是否在 interval 对象中
    assert -5 in inter  # 断言判断数值 -5 是否在 interval 对象中
    assert 5 in inter  # 断言判断数值 5 是否在 interval 对象中
    assert interval(0, 3) in inter  # 断言判断区间 [0, 3] 是否完全包含在 interval 对象中
    assert interval(-6, 2) not in inter  # 断言判断区间 [-6, 2] 是否完全不包含在 interval 对象中
    assert -5.05 not in inter  # 断言判断数值 -5.05 是否不在 interval 对象中
    assert 5.3 not in inter  # 断言判断数值 5.3 是否不在 interval 对象中
    interb = interval(-float('inf'), float('inf'))  # 创建一个 interval 对象 interb，表示全体实数
    assert 0 in inter  # 断言判断数值 0 是否在 interval 对象中
    assert inter in interb  # 断言判断 interval 对象 inter 是否包含在 interval 对象 interb 中
    assert interval(0, float('inf')) in interb  # 断言判断区间 [0, ∞) 是否包含在 interval 对象 interb 中
    assert interval(-float('inf'), 5) in interb  # 断言判断区间 (-∞, 5] 是否包含在 interval 对象 interb 中
    assert interval(-1e50, 1e50) in interb  # 断言判断区间 [-10^50, 10^50] 是否包含在 interval 对象 interb 中
    assert (-interval(-1, -2, is_valid=False) == interval(1, 2)) == (True, False)  # 断言判断 interval 对象的取反是否正确
    raises(ValueError, lambda: interval(1, 2, 3))  # 断言验证调用 interval 函数时是否会引发 ValueError 异常


def test_interval_add():  # 定义测试函数 test_interval_add，用于测试 interval 对象的加法运算
    assert (interval(1, 2) + interval(2, 3) == interval(3, 5)) == (True, True)  # 断言判断两个 interval 对象相加的结果是否正确
    assert (1 + interval(1, 2) == interval(2, 3)) == (True, True)  # 断言判断数值与 interval 对象相加的结果是否正确
    assert (interval(1, 2) + 1 == interval(2, 3)) == (True, True)  # 断言判断 interval 对象与数值相加的结果是否正确
    compare = (1 + interval(0, float('inf')) == interval(1, float('inf')))  # 将数值与区间 [0, ∞) 相加，并与区间 [1, ∞) 进行比较
    assert compare == (True, True)  # 断言判断比较结果是否正确
    a = 1 + interval(2, 5, is_valid=False)  # 将数值与无效的 interval 对象 [2, 5] 相加
    assert a.is_valid is False  # 断言判断相加后的结果是否为无效的 interval 对象
    a = 1 + interval(2, 5, is_valid=None)  # 将数值与未定义有效性的 interval 对象 [2, 5] 相加
    assert a.is_valid is None  # 断言判断相加后的结果是否为未定义有效性的 interval 对象
    a = interval(2, 5, is_valid=False) + interval(3, 5, is_valid=None)  # 将无效的 interval 对象 [2, 5] 与未定义有效性的 interval 对象 [3, 5] 相加
    assert a.is_valid is False  # 断言判断相加后的结果是否为无效的 interval 对象
    a = interval(3, 5) + interval(-1, 1, is_valid=None)  # 将 interval 对象 [3, 5] 与未定义有效性的 interval 对象 [-1, 1] 相加
    assert a.is_valid is None  # 断言判断相加后的结果是否为未定义有效性的 interval 对象
    a = interval(2, 5, is_valid=False) + 1  # 将无效的 interval 对象 [2, 5] 与数值相加
    assert a.is_valid is False  # 断言判断相加后的结果是否为无效的 interval 对象


def test_interval_sub():  # 定义测试函数 test_interval_sub，用于测试 interval 对象的减法运算
    assert (interval(1, 2) - interval(1, 5) == interval(-4, 1)) == (True, True)  # 断言判断两个 interval 对象相减的结果是否正确
    assert (interval(1, 2) - 1 == interval(0, 1)) == (True, True)  # 断言
    # 创建一个区间对象a，表示区间(1, 3)，但不进行有效性检查
    a = interval(1, 3, is_valid=False) - interval(1, 3)
    # 断言区间对象a的有效性为False
    assert a.is_valid is False
    
    # 创建一个新的区间对象a，表示区间(1, 3)，并且有效性未指定
    a = interval(1, 3, is_valid=None) - interval(1, 3)
    # 断言区间对象a的有效性为None
    assert a.is_valid is None
def test_interval_inequality():
    # 测试区间的不等式比较运算
    assert (interval(1, 2) < interval(3, 4)) == (True, True)
    # 区间(1, 2)不小于区间(2, 4)，第一个比较结果为None，第二个比较结果为True
    assert (interval(1, 2) < interval(2, 4)) == (None, True)
    # 区间(1, 2)不小于区间(-2, 0)，第一个比较结果为False，第二个比较结果为True
    assert (interval(1, 2) < interval(-2, 0)) == (False, True)
    # 区间(1, 2)小于等于区间(2, 4)，两个比较结果均为True
    assert (interval(1, 2) <= interval(2, 4)) == (True, True)
    # 区间(1, 2)小于等于区间(1.5, 6)，第一个比较结果为None，第二个比较结果为True
    assert (interval(1, 2) <= interval(1.5, 6)) == (None, True)
    # 区间(2, 3)小于等于区间(1, 2)，第一个比较结果为None，第二个比较结果为True
    assert (interval(2, 3) <= interval(1, 2)) == (None, True)
    # 区间(2, 3)小于等于区间(1, 1.5)，第一个比较结果为False，第二个比较结果为True
    assert (interval(2, 3) <= interval(1, 1.5)) == (False, True)
    # 区间(1, 2)在is_valid=False条件下，小于等于区间(-2, 0)，两个比较结果均为False
    assert (interval(1, 2, is_valid=False) <= interval(-2, 0)) == (False, False)
    # 区间(1, 2)在is_valid=None条件下，小于等于区间(-2, 0)，第一个比较结果为False，第二个比较结果为None
    assert (interval(1, 2, is_valid=None) <= interval(-2, 0)) == (False, None)
    # 区间(1, 2)小于等于数值1.5，第一个比较结果为None，第二个比较结果为True
    assert (interval(1, 2) <= 1.5) == (None, True)
    # 区间(1, 2)小于等于数值3，两个比较结果均为True
    assert (interval(1, 2) <= 3) == (True, True)
    # 区间(1, 2)小于等于数值0，第一个比较结果为False，第二个比较结果为True
    assert (interval(1, 2) <= 0) == (False, True)
    # 区间(5, 8)大于区间(2, 3)，两个比较结果均为True
    assert (interval(5, 8) > interval(2, 3)) == (True, True)
    # 区间(2, 5)大于区间(1, 3)，第一个比较结果为None，第二个比较结果为True
    assert (interval(2, 5) > interval(1, 3)) == (None, True)
    # 区间(2, 3)大于区间(3.1, 5)，第一个比较结果为False，第二个比较结果为True
    assert (interval(2, 3) > interval(3.1, 5)) == (False, True)

    # 区间(-1, 1)等于数值0，第一个比较结果为None，第二个比较结果为True
    assert (interval(-1, 1) == 0) == (None, True)
    # 区间(-1, 1)等于数值2，第一个比较结果为False，第二个比较结果为True
    assert (interval(-1, 1) == 2) == (False, True)
    # 区间(-1, 1)不等于数值0，第一个比较结果为None，第二个比较结果为True
    assert (interval(-1, 1) != 0) == (None, True)
    # 区间(-1, 1)不等于数值2，第一个比较结果为True，第二个比较结果为True
    assert (interval(-1, 1) != 2) == (True, True)

    # 区间(3, 5)大于数值2，两个比较结果均为True
    assert (interval(3, 5) > 2) == (True, True)
    # 区间(3, 5)小于数值2，第一个比较结果为False，第二个比较结果为True
    assert (interval(3, 5) < 2) == (False, True)
    # 区间(1, 5)小于数值2，第一个比较结果为None，第二个比较结果为True
    assert (interval(1, 5) < 2) == (None, True)
    # 区间(1, 5)大于数值2，第一个比较结果为None，第二个比较结果为True
    assert (interval(1, 5) > 2) == (None, True)
    # 区间(0, 1)大于数值2，第一个比较结果为False，第二个比较结果为True
    assert (interval(0, 1) > 2) == (False, True)
    # 区间(1, 2)大于等于区间(0, 1)，两个比较结果均为True
    assert (interval(1, 2) >= interval(0, 1)) == (True, True)
    # 区间(1, 2)大于等于区间(0, 1.5)，第一个比较结果为None，第二个比较结果为True
    assert (interval(1, 2) >= interval(0, 1.5)) == (None, True)
    # 区间(1, 2)大于等于区间(3, 4)，第一个比较结果为False，第二个比较结果为True
    assert (interval(1, 2) >= interval(3, 4)) == (False, True)
    # 区间(1, 2)大于等于数值0，两个比较结果均为True
    assert (interval(1, 2) >= 0) == (True, True)
    # 区间(1, 2)大于等于数值1.2，第一个比较结果为None，第二个比较结果为True
    assert (interval(1, 2) >= 1.2) == (None, True)
    # 区间(1, 2)大于等于数值3，第一个比较结果为False，第二个比较结果为True
    assert (interval(1, 2) >= 3) == (False, True)
    # 数值2大于区间(0, 1)，两个比较结果均为True
    assert (2 > interval(0, 1)) == (True, True)
    # 区间(-1, 1)在is_valid=False条件下，小于区间(2, 5)，第一个比较结果为True，第二个比较结果为False
    a = interval(-1, 1, is_valid=False) < interval(2, 5, is_valid=None)
    assert a == (True, False)
    # 区间(-1, 1)在is_valid=None条件下，小于区间(2, 5)，第一个比较结果为True，第二个比较结果为False
    a = interval(-1, 1, is_valid=None) < interval(2, 5, is_valid=False)
    assert a == (True, False)
    # 区间(-1, 1)在is_valid=None条件下，小于区间(2, 5)，第
    # 断言检查对象 a 的 is_valid 属性是否为 None
    assert a.is_valid is None
    
    # 创建两个 interval 对象的交集，其中第一个对象设置 is_valid=False，第二个对象设置 is_valid=None
    a = interval(1, 5, is_valid=False) * interval(1, 2, is_valid=None)
    
    # 断言检查对象 a 的 is_valid 属性是否为 False
    assert a.is_valid is False
# 定义用于测试的函数，测试除法操作
def test_interval_div():
    # 使用 interval 对象进行除法运算，期望结果为全体实数范围，包括无效值
    div = interval(1, 2, is_valid=False) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=False)

    # 使用 interval 对象进行除法运算，期望结果为全体实数范围，包括未确定的有效性
    div = interval(1, 2, is_valid=None) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=None)

    # 使用 interval 对象进行除法运算，期望结果为全体实数范围，包括未确定的有效性
    div = 3 / interval(1, 2, is_valid=None)
    assert div == interval(-float('inf'), float('inf'), is_valid=None)

    # 测试除以零的情况，期望结果 is_valid 属性为 False
    a = interval(1, 2) / 0
    assert a.is_valid is False

    # 使用 interval 对象进行除法运算，期望结果 is_valid 属性为 None
    a = interval(0.5, 1) / interval(-1, 0)
    assert a.is_valid is None

    # 使用 interval 对象进行除法运算，期望结果 is_valid 属性为 None
    a = interval(0, 1) / interval(0, 1)
    assert a.is_valid is None

    # 使用 interval 对象进行除法运算，期望结果 is_valid 属性为 None
    a = interval(-1, 1) / interval(-1, 1)
    assert a.is_valid is None

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-1, 2) / interval(0.5, 1) == interval(-2.0, 4.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(0, 1) / interval(0.5, 1) == interval(0.0, 2.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-1, 0) / interval(0.5, 1) == interval(-2.0, 0.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-0.5, -0.25) / interval(0.5, 1) == interval(-1.0, -0.25)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(0.5, 1) / interval(0.5, 1) == interval(0.5, 2.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(0.5, 4) / interval(0.5, 1) == interval(0.5, 8.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-1, -0.5) / interval(0.5, 1) == interval(-2.0, -0.5)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-4, -0.5) / interval(0.5, 1) == interval(-8.0, -0.5)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-1, 2) / interval(-2, -0.5) == interval(-4.0, 2.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(0, 1) / interval(-2, -0.5) == interval(-2.0, 0.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-1, 0) / interval(-2, -0.5) == interval(0.0, 2.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-0.5, -0.25) / interval(-2, -0.5) == interval(0.125, 1.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(0.5, 1) / interval(-2, -0.5) == interval(-2.0, -0.25)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(0.5, 4) / interval(-2, -0.5) == interval(-8.0, -0.25)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-1, -0.5) / interval(-2, -0.5) == interval(0.25, 2.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果为具体的数值区间对象
    a = interval(-4, -0.5) / interval(-2, -0.5) == interval(0.25, 8.0)
    assert a == (True, True)

    # 使用 interval 对象进行除法运算，期望结果 is_valid 属性为 False
    a = interval(-5, 5, is_valid=False) / 2
    assert a.is_valid is False

# 定义测试哈希功能的函数
def test_hashable():
    '''
    测试 interval 对象是否可哈希。
    这是为了能够将其放入缓存中，这在 py3k 中绘图时是必要的。
    详细信息，请参见：

    https://github.com/sympy/sympy/pull/2101
    https://github.com/sympy/sympy/issues/6533
    '''
    # 对 interval 对象进行哈希运算
    hash(interval(1, 1))
    hash(interval(1, 1, is_valid=True))
    hash(interval(-4, -0.5))
    hash(interval(-2, -0.5))
    hash(interval(0.25, 8.0))
```