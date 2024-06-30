# `D:\src\scipysrc\sympy\sympy\strategies\branch\tests\test_traverse.py`

```
# 导入必要的模块和函数
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.strategies.branch.traverse import top_down, sall
from sympy.strategies.branch.core import do_one, identity


# 定义一个生成器函数，用于增加整数对象的值
def inc(x):
    # 如果 x 是整数类型，生成 x+1 的值
    if isinstance(x, Integer):
        yield x + 1


# 测试 top_down 策略的简单用例
def test_top_down_easy():
    # 创建一个 Basic 对象，包含整数 S(1) 和 S(2)
    expr = Basic(S(1), S(2))
    # 期望的结果是将每个整数对象增加 1
    expected = Basic(S(2), S(3))
    # 创建一个 top_down 策略，应用 inc 函数
    brl = top_down(inc)

    # 断言结果与期望相同
    assert set(brl(expr)) == {expected}


# 测试 top_down 策略的复杂用例
def test_top_down_big_tree():
    # 创建一个复杂的 Basic 对象树
    expr = Basic(S(1), Basic(S(2)), Basic(S(3), Basic(S(4)), S(5)))
    # 期望的结果是将每个整数对象增加 1
    expected = Basic(S(2), Basic(S(3)), Basic(S(4), Basic(S(5)), S(6)))
    # 创建一个 top_down 策略，应用 inc 函数
    brl = top_down(inc)

    # 断言结果与期望相同
    assert set(brl(expr)) == {expected}


# 测试更复杂的 top_down 策略与函数的结合
def test_top_down_harder_function():
    # 定义一个更复杂的生成器函数 split5
    def split5(x):
        # 如果 x 等于 5，生成 x-1 和 x+1 的值
        if x == 5:
            yield x - 1
            yield x + 1

    # 创建一个 Basic 对象树，包含多层嵌套
    expr = Basic(Basic(S(5), S(6)), S(1))
    # 期望的结果是对含有 5 的部分进行分割，生成多个变体
    expected = {Basic(Basic(S(4), S(6)), S(1)), Basic(Basic(S(6), S(6)), S(1))}
    # 创建一个 top_down 策略，应用 split5 函数
    brl = top_down(split5)

    # 断言结果与期望相同
    assert set(brl(expr)) == expected


# 测试 sall 策略的简单用例
def test_sall():
    # 创建一个 Basic 对象，包含整数 S(1) 和 S(2)
    expr = Basic(S(1), S(2))
    # 期望的结果是将每个整数对象增加 1
    expected = Basic(S(2), S(3))
    # 创建一个 sall 策略，应用 inc 函数
    brl = sall(inc)

    # 断言结果与期望相同
    assert list(brl(expr)) == [expected]

    # 创建一个更复杂的 Basic 对象树
    expr = Basic(S(1), S(2), Basic(S(3), S(4)))
    # 期望的结果是将每个整数对象增加 1，但仅应用于最顶层的元素
    expected = Basic(S(2), S(3), Basic(S(3), S(4)))
    # 创建一个 sall 策略，应用 do_one(inc, identity) 函数
    brl = sall(do_one(inc, identity))

    # 断言结果与期望相同
    assert list(brl(expr)) == [expected]
```