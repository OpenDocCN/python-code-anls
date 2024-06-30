# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_function.py`

```
""" Unit tests for Hyper_Function"""
# 导入必要的模块和函数
from sympy.core import symbols, Dummy, Tuple, S, Rational
from sympy.functions import hyper
from sympy.simplify.hyperexpand import Hyper_Function

# 测试函数，验证 Hyper_Function 的属性设置是否正确
def test_attrs():
    # 创建虚拟符号 a 和 b
    a, b = symbols('a, b', cls=Dummy)
    # 创建 Hyper_Function 对象
    f = Hyper_Function([2, a], [b])
    # 断言属性 ap 正确
    assert f.ap == Tuple(2, a)
    # 断言属性 bq 正确
    assert f.bq == Tuple(b)
    # 断言 args 属性正确
    assert f.args == (Tuple(2, a), Tuple(b))
    # 断言 sizes 属性正确
    assert f.sizes == (2, 1)

# 测试函数，验证 Hyper_Function 对象的调用结果是否正确
def test_call():
    # 创建虚拟符号 a, b, x
    a, b, x = symbols('a, b, x', cls=Dummy)
    # 创建 Hyper_Function 对象
    f = Hyper_Function([2, a], [b])
    # 断言调用结果是否正确
    assert f(x) == hyper([2, a], [b], x)

# 测试函数，验证 Hyper_Function 对象是否包含特定符号或元组
def test_has():
    # 创建虚拟符号 a, b, c
    a, b, c = symbols('a, b, c', cls=Dummy)
    # 创建 Hyper_Function 对象
    f = Hyper_Function([2, -a], [b])
    # 断言是否包含符号 a
    assert f.has(a)
    # 断言是否包含元组 (b)
    assert f.has(Tuple(b))
    # 断言不包含符号 c
    assert not f.has(c)

# 测试函数，验证 Hyper_Function 对象的相等性
def test_eq():
    # 断言相等的 Hyper_Function 对象
    assert Hyper_Function([1], []) == Hyper_Function([1], [])
    # 断言不相等的 Hyper_Function 对象
    assert (Hyper_Function([1], []) != Hyper_Function([1], [])) is False
    # 断言不相等的 Hyper_Function 对象（不同的参数列表）
    assert Hyper_Function([1], []) != Hyper_Function([2], [])
    # 断言不相等的 Hyper_Function 对象（不同的参数数量）
    assert Hyper_Function([1], []) != Hyper_Function([1, 2], [])
    # 断言不相等的 Hyper_Function 对象（不同的 bq 参数）
    assert Hyper_Function([1], []) != Hyper_Function([1], [2])

# 测试函数，验证 Hyper_Function 对象的 gamma 方法
def test_gamma():
    # 验证特定参数下的 gamma 方法返回值
    assert Hyper_Function([2, 3], [-1]).gamma == 0
    assert Hyper_Function([-2, -3], [-1]).gamma == 2
    n = Dummy(integer=True)
    assert Hyper_Function([-1, n, 1], []).gamma == 1
    assert Hyper_Function([-1, -n, 1], []).gamma == 1
    p = Dummy(integer=True, positive=True)
    assert Hyper_Function([-1, p, 1], []).gamma == 1
    assert Hyper_Function([-1, -p, 1], []).gamma == 2

# 测试函数，验证 Hyper_Function 对象的 _is_suitable_origin 方法
def test_suitable_origin():
    # 验证不同情况下 _is_suitable_origin 方法的返回值
    assert Hyper_Function((S.Half,), (Rational(3, 2),))._is_suitable_origin() is True
    assert Hyper_Function((S.Half,), (S.Half,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (Rational(-1, 2),))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (0,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (-1, 1,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 0), (1,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 1),
            (2, Rational(-2, 3)))._is_suitable_origin() is True
    assert Hyper_Function((S.Half, 1),
            (2, Rational(-2, 3), Rational(3, 2)))._is_suitable_origin() is True
```