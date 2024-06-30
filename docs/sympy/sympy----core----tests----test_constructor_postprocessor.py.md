# `D:\src\scipysrc\sympy\sympy\core\tests\test_constructor_postprocessor.py`

```
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import (Symbol, symbols)
from sympy.testing.pytest import XFAIL

class SymbolInMulOnce(Symbol):
    # 定义一个符号类，只能在乘法表达式 `Mul` 中出现一次
    pass

# 将自定义符号类 `SymbolInMulOnce` 注册到 SymPy 的构造后处理映射中
Basic._constructor_postprocessor_mapping[SymbolInMulOnce] = {
    "Mul": [lambda x: x],  # 在乘法表达式中保持不变
    "Pow": [lambda x: x.base if isinstance(x.base, SymbolInMulOnce) else x],  # 如果指数的基数是 `SymbolInMulOnce` 类型，则保留基数部分
    "Add": [lambda x: x],  # 在加法表达式中保持不变
}

# 定义一个符号类，可以从乘法表达式 `Mul` 中移除其他符号
def _postprocess_SymbolRemovesOtherSymbols(expr):
    args = tuple(i for i in expr.args if not isinstance(i, Symbol) or isinstance(i, SymbolRemovesOtherSymbols))
    if args == expr.args:
        return expr
    return Mul.fromiter(args)

class SymbolRemovesOtherSymbols(Symbol):
    # 定义一个符号类，用于在乘法表达式 `Mul` 中移除其他符号
    pass

# 将自定义符号类 `SymbolRemovesOtherSymbols` 注册到 SymPy 的构造后处理映射中
Basic._constructor_postprocessor_mapping[SymbolRemovesOtherSymbols] = {
    "Mul": [_postprocess_SymbolRemovesOtherSymbols],  # 使用定义的处理函数来处理乘法表达式 `Mul`
}

class SubclassSymbolInMulOnce(SymbolInMulOnce):
    # `SymbolInMulOnce` 类的子类
    pass

class SubclassSymbolRemovesOtherSymbols(SymbolRemovesOtherSymbols):
    # `SymbolRemovesOtherSymbols` 类的子类
    pass

# 测试函数：验证 `SymbolInMulOnce` 和 `SymbolRemovesOtherSymbols` 的行为
def test_constructor_postprocessors1():
    x = SymbolInMulOnce("x")
    y = SymbolInMulOnce("y")
    assert isinstance(3*x, Mul)  # 验证乘法表达式中的乘法运算
    assert (3*x).args == (3, x)  # 验证乘法表达式中的参数
    assert x*x == x  # 验证乘法表达式中的乘法运算
    assert 3*x*x == 3*x  # 验证乘法表达式中的乘法运算
    assert 2*x*x + x == 3*x  # 验证乘法表达式中的加法运算
    assert x**3*y*y == x*y  # 验证乘方运算
    assert x**5 + y*x**3 == x + x*y  # 验证乘方运算和加法运算

    w = SymbolRemovesOtherSymbols("w")
    assert x*w == w  # 验证乘法表达式中移除其他符号的行为
    assert (3*w).args == (3, w)  # 验证乘法表达式中移除其他符号的行为
    assert set((w + x).args) == {x, w}  # 验证加法表达式中移除其他符号的行为

# 测试函数：验证 `SubclassSymbolInMulOnce` 和 `SubclassSymbolRemovesOtherSymbols` 的行为
def test_constructor_postprocessors2():
    x = SubclassSymbolInMulOnce("x")
    y = SubclassSymbolInMulOnce("y")
    assert isinstance(3*x, Mul)  # 验证乘法表达式中的乘法运算
    assert (3*x).args == (3, x)  # 验证乘法表达式中的参数
    assert x*x == x  # 验证乘法表达式中的乘法运算
    assert 3*x*x == 3*x  # 验证乘法表达式中的乘法运算
    assert 2*x*x + x == 3*x  # 验证乘法表达式中的加法运算
    assert x**3*y*y == x*y  # 验证乘方运算
    assert x**5 + y*x**3 == x + x*y  # 验证乘方运算和加法运算

    w = SubclassSymbolRemovesOtherSymbols("w")
    assert x*w == w  # 验证乘法表达式中移除其他符号的行为
    assert (3*w).args == (3, w)  # 验证乘法表达式中移除其他符号的行为
    assert set((w + x).args) == {x, w}  # 验证加法表达式中移除其他符号的行为

@XFAIL
def test_subexpression_postprocessors():
    # 该函数用于测试子表达式的后处理，但该功能已移除，参见 issue #15948
    a = symbols("a")
    x = SymbolInMulOnce("x")
    w = SymbolRemovesOtherSymbols("w")
    assert 3*a*w**2 == 3*w**2
    assert 3*a*x**3*w**2 == 3*w**2

    x = SubclassSymbolInMulOnce("x")
    w = SubclassSymbolRemovesOtherSymbols("w")
    assert 3*a*w**2 == 3*w**2
    assert 3*a*x**3*w**2 == 3*w**2
```