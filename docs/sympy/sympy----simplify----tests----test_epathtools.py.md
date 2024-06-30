# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_epathtools.py`

```
# 导入需要的模块和函数
"""Tests for tools for manipulation of expressions using paths. """
from sympy.simplify.epathtools import epath, EPath  # 导入路径操作相关的函数和类
from sympy.testing.pytest import raises  # 导入用于测试的 pytest 的 raises 函数

from sympy.core.numbers import E  # 导入常数 E
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入三角函数
from sympy.abc import x, y, z, t  # 导入符号变量 x, y, z, t

# 定义测试函数 test_epath_select
def test_epath_select():
    expr = [((x, 1, t), 2), ((3, y, 4), z)]  # 定义测试用的表达式列表

    # 测试 epath 函数的不同路径表达式
    assert epath("/*", expr) == [((x, 1, t), 2), ((3, y, 4), z)]
    assert epath("/*/*", expr) == [(x, 1, t), 2, (3, y, 4), z]
    assert epath("/*/*/*", expr) == [x, 1, t, 3, y, 4]
    assert epath("/*/*/*/*", expr) == []

    assert epath("/[:]", expr) == [((x, 1, t), 2), ((3, y, 4), z)]
    assert epath("/[:]/[:]", expr) == [(x, 1, t), 2, (3, y, 4), z]
    assert epath("/[:]/[:]/[:]", expr) == [x, 1, t, 3, y, 4]
    assert epath("/[:]/[:]/[:]/[:]", expr) == []

    assert epath("/*/[:]", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/[0]", expr) == [(x, 1, t), (3, y, 4)]
    assert epath("/*/[1]", expr) == [2, z]
    assert epath("/*/[2]", expr) == []

    assert epath("/*/int", expr) == [2]
    assert epath("/*/Symbol", expr) == [z]
    assert epath("/*/tuple", expr) == [(x, 1, t), (3, y, 4)]
    assert epath("/*/__iter__?", expr) == [(x, 1, t), (3, y, 4)]

    assert epath("/*/int|tuple", expr) == [(x, 1, t), 2, (3, y, 4)]
    assert epath("/*/Symbol|tuple", expr) == [(x, 1, t), (3, y, 4), z]
    assert epath("/*/int|Symbol|tuple", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/int|__iter__?", expr) == [(x, 1, t), 2, (3, y, 4)]
    assert epath("/*/Symbol|__iter__?", expr) == [(x, 1, t), (3, y, 4), z]
    assert epath("/*/int|Symbol|__iter__?", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/[0]/int", expr) == [1, 3, 4]
    assert epath("/*/[0]/Symbol", expr) == [x, t, y]

    assert epath("/*/[0]/int[1:]", expr) == [1, 4]
    assert epath("/*/[0]/Symbol[1:]", expr) == [t, y]

    assert epath("/Symbol", x + y + z + 1) == [x, y, z]
    assert epath("/*/*/Symbol", t + sin(x + 1) + cos(x + y + E)) == [x, x, y]

# 定义测试函数 test_epath_apply
def test_epath_apply():
    expr = [((x, 1, t), 2), ((3, y, 4), z)]  # 定义测试用的表达式列表
    func = lambda expr: expr**2  # 定义一个平方函数

    assert epath("/*", expr, list) == [[(x, 1, t), 2], [(3, y, 4), z]]

    assert epath("/*/[0]", expr, list) == [([x, 1, t], 2), ([3, y, 4], z)]
    assert epath("/*/[1]", expr, func) == [((x, 1, t), 4), ((3, y, 4), z**2)]
    assert epath("/*/[2]", expr, list) == expr

    assert epath("/*/[0]/int", expr, func) == [((x, 1, t), 2), ((9, y, 16), z)]
    assert epath("/*/[0]/Symbol", expr, func) == [((x**2, 1, t**2), 2), ((3, y**2, 4), z)]
    assert epath("/*/[0]/int[1:]", expr, func) == [((x, 1, t), 2), ((3, y, 16), z)]
    assert epath("/*/[0]/Symbol[1:]", expr, func) == [((x, 1, t**2), 2), ((3, y**2, 4), z)]

    assert epath("/Symbol", x + y + z + 1, func) == x**2 + y**2 + z**2 + 1
    # 断言语句，用于检查表达式是否为真
    assert epath("/*/*/Symbol", t + sin(x + 1) + cos(x + y + E), func) == \
        # 构造表达式 epath("/*/*/Symbol", t + sin(x**2 + 1) + cos(x**2 + y**2 + E), func)，断言其结果与右侧表达式相等
        t + sin(x**2 + 1) + cos(x**2 + y**2 + E)
# 定义一个测试函数 test_EPath
def test_EPath():
    # 断言：检查 EPath 对象实例化时内部 _path 属性的值是否符合预期 "/*/[0]"
    assert EPath("/*/[0]")._path == "/*/[0]"
    # 断言：检查将 EPath 对象作为参数传入 EPath 构造函数时，内部 _path 属性的值是否保持 "/*/[0]"
    assert EPath(EPath("/*/[0]"))._path == "/*/[0]"
    # 断言：检查 epath 函数返回的对象是否为 EPath 类型
    assert isinstance(epath("/*/[0]"), EPath) is True

    # 断言：检查 EPath 对象的 repr 方法返回的字符串是否正确
    assert repr(EPath("/*/[0]")) == "EPath('/*/[0]')"

    # 引发 ValueError 异常，测试空路径时是否抛出异常
    raises(ValueError, lambda: EPath(""))
    # 引发 ValueError 异常，测试根路径 "/" 时是否抛出异常
    raises(ValueError, lambda: EPath("/"))
    # 引发 ValueError 异常，测试路径包含非法字符 "|x" 时是否抛出异常
    raises(ValueError, lambda: EPath("/|x"))
    # 引发 ValueError 异常，测试路径包含不完整的字符组 "["
    raises(ValueError, lambda: EPath("/["))
    # 引发 ValueError 异常，测试路径包含不正确的字符 "%"
    raises(ValueError, lambda: EPath("/[0]%"))

    # 引发 NotImplementedError 异常，测试传入符号 "Symbol" 时是否抛出异常
    raises(NotImplementedError, lambda: EPath("Symbol"))
```