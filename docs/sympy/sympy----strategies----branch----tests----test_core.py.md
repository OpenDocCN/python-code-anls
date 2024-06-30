# `D:\src\scipysrc\sympy\sympy\strategies\branch\tests\test_core.py`

```
# 导入从 sympy.strategies.branch.core 中导入的多个函数和类
from sympy.strategies.branch.core import (
    exhaust, debug, multiplex, condition, notempty, chain, onaction, sfilter,
    yieldify, do_one, identity)


# 递减生成器函数，如果输入大于0则返回 x-1，否则返回 x
def posdec(x):
    if x > 0:
        yield x - 1
    else:
        yield x


# 分支函数，根据不同的条件返回不同的值
def branch5(x):
    if 0 < x < 5:
        yield x - 1
    elif 5 < x < 10:
        yield x + 1
    elif x == 5:
        yield x + 1
        yield x - 1
    else:
        yield x


# 判断是否为偶数的函数
def even(x):
    return x % 2 == 0


# 返回 x+1 的生成器函数
def inc(x):
    yield x + 1


# 生成从 0 到 n-1 的整数序列
def one_to_n(n):
    yield from range(n)


# 测试 exhaust 函数的单元测试
def test_exhaust():
    brl = exhaust(branch5)
    assert set(brl(3)) == {0}
    assert set(brl(7)) == {10}
    assert set(brl(5)) == {0, 10}


# 测试 debug 函数的单元测试
def test_debug():
    from io import StringIO
    file = StringIO()
    rl = debug(posdec, file)
    list(rl(5))
    log = file.getvalue()
    file.close()

    assert posdec.__name__ in log
    assert '5' in log
    assert '4' in log


# 测试 multiplex 函数的单元测试
def test_multiplex():
    brl = multiplex(posdec, branch5)
    assert set(brl(3)) == {2}
    assert set(brl(7)) == {6, 8}
    assert set(brl(5)) == {4, 6}


# 测试 condition 函数的单元测试
def test_condition():
    brl = condition(even, branch5)
    assert set(brl(4)) == set(branch5(4))
    assert set(brl(5)) == set()


# 测试 sfilter 函数的单元测试
def test_sfilter():
    brl = sfilter(even, one_to_n)
    assert set(brl(10)) == {0, 2, 4, 6, 8}


# 测试 notempty 函数的单元测试
def test_notempty():
    # 如果输入为偶数则返回该数，否则返回空集合的生成器函数
    def ident_if_even(x):
        if even(x):
            yield x

    brl = notempty(ident_if_even)
    assert set(brl(4)) == {4}
    assert set(brl(5)) == {5}


# 测试 chain 函数的单元测试
def test_chain():
    assert list(chain()(2)) == [2]  # identity
    assert list(chain(inc, inc)(2)) == [4]
    assert list(chain(branch5, inc)(4)) == [4]
    assert set(chain(branch5, inc)(5)) == {5, 7}
    assert list(chain(inc, branch5)(5)) == [7]


# 测试 onaction 函数的单元测试
def test_onaction():
    L = []

    # 记录输入、输出的函数
    def record(fn, input, output):
        L.append((input, output))

    list(onaction(inc, record)(2))
    assert L == [(2, 3)]

    list(onaction(identity, record)(2))
    assert L == [(2, 3)]


# 测试 yieldify 函数的单元测试
def test_yieldify():
    yinc = yieldify(lambda x: x + 1)
    assert list(yinc(3)) == [4]


# 测试 do_one 函数的单元测试
def test_do_one():
    # 抛出 ValueError 的函数
    def bad(expr):
        raise ValueError

    assert list(do_one(inc)(3)) == [4]
    assert list(do_one(inc, bad)(3)) == [4]
    assert list(do_one(inc, posdec)(3)) == [4]
```