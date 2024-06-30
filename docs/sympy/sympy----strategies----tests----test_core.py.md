# `D:\src\scipysrc\sympy\sympy\strategies\tests\test_core.py`

```
# 导入未来版本兼容的注解模块
from __future__ import annotations
# 导入 Sympy 中的单例模块 S
from sympy.core.singleton import S
# 导入 Sympy 中的基础模块 Basic
from sympy.core.basic import Basic
# 导入 Sympy 策略核心模块，包括多个策略函数
from sympy.strategies.core import (
    null_safe, exhaust, memoize, condition,
    chain, tryit, do_one, debug, switch, minimize)
# 导入字符串 IO 模块
from io import StringIO


# 定义一个函数 posdec，将输入的整数减 1，如果输入为非正数则返回原值
def posdec(x: int) -> int:
    if x > 0:
        return x - 1
    return x


# 定义一个函数 inc，将输入的整数加 1
def inc(x: int) -> int:
    return x + 1


# 定义一个函数 dec，将输入的整数减 1
def dec(x: int) -> int:
    return x - 1


# 测试函数 test_null_safe
def test_null_safe():
    # 定义一个函数 rl，如果输入为 1，则返回 2，否则返回 None
    def rl(expr: int) -> int | None:
        if expr == 1:
            return 2
        return None
    
    # 使用 null_safe 函数包装 rl 函数，使其对输入参数进行安全处理
    safe_rl = null_safe(rl)
    assert rl(1) == safe_rl(1)
    assert rl(3) is None
    assert safe_rl(3) == 3


# 测试函数 test_exhaust
def test_exhaust():
    # 使用 exhaust 函数包装 posdec 函数，返回一个总是返回 0 的函数 sink
    sink = exhaust(posdec)
    assert sink(5) == 0
    assert sink(10) == 0


# 测试函数 test_memoize
def test_memoize():
    # 使用 memoize 函数包装 posdec 函数，返回一个记忆化的函数 rl
    rl = memoize(posdec)
    assert rl(5) == posdec(5)
    assert rl(5) == posdec(5)
    assert rl(-2) == posdec(-2)


# 测试函数 test_condition
def test_condition():
    # 使用 condition 函数，根据输入函数 lambda x: x % 2 == 0 条件选择 posdec 函数或原函数
    rl = condition(lambda x: x % 2 == 0, posdec)
    assert rl(5) == 5
    assert rl(4) == 3


# 测试函数 test_chain
def test_chain():
    # 使用 chain 函数，依次应用两次 posdec 函数
    rl = chain(posdec, posdec)
    assert rl(5) == 3
    assert rl(1) == 0


# 测试函数 test_tryit
def test_tryit():
    # 定义一个函数 rl，断言总是失败
    def rl(expr: Basic) -> Basic:
        assert False
    
    # 使用 tryit 函数包装 rl 函数，使其能处理 AssertionError 异常
    safe_rl = tryit(rl, AssertionError)
    assert safe_rl(S(1)) == S(1)


# 测试函数 test_do_one
def test_do_one():
    # 使用 do_one 函数，尝试应用 posdec 函数两次
    rl = do_one(posdec, posdec)
    assert rl(5) == 4
    
    # 定义两个测试规则函数 rl1 和 rl2
    def rl1(x: int) -> int:
        if x == 1:
            return 2
        return x
    
    def rl2(x: int) -> int:
        if x == 2:
            return 3
        return x
    
    # 使用 do_one 函数，根据输入值选择应用 rl1 或 rl2 函数
    rule = do_one(rl1, rl2)
    assert rule(1) == 2
    assert rule(rule(1)) == 3


# 测试函数 test_debug
def test_debug():
    # 创建一个 StringIO 对象作为文件
    file = StringIO()
    # 使用 debug 函数，将 posdec 函数的调试信息写入文件
    rl = debug(posdec, file)
    rl(5)
    log = file.getvalue()
    file.close()
    
    # 断言文件中包含 posdec 函数名、输入值 5 和输出值 4 的调试信息
    assert posdec.__name__ in log
    assert '5' in log
    assert '4' in log


# 测试函数 test_switch
def test_switch():
    # 定义一个返回输入值模 3 的余数的函数 key
    def key(x: int) -> int:
        return x % 3
    
    # 使用 switch 函数，根据 key 函数返回值选择应用 inc 或 dec 函数
    rl = switch(key, {0: inc, 1: dec})
    assert rl(3) == 4
    assert rl(4) == 3
    assert rl(5) == 5


# 测试函数 test_minimize
def test_minimize():
    # 定义一个返回相反数的函数 key
    def key(x: int) -> int:
        return -x
    
    # 使用 minimize 函数，从两个函数 inc 和 dec 中选择最小化的函数 rl
    rl = minimize(inc, dec)
    assert rl(4) == 3
    
    # 使用 minimize 函数，从两个函数 inc 和 dec 中选择最小化的函数 rl，并指定目标函数 key
    rl = minimize(inc, dec, objective=key)
    assert rl(4) == 5
```