# `D:\src\scipysrc\sympy\sympy\core\tests\test_sympify.py`

```
# 导入 sympy 库中的具体类和函数
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, pi, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, Or, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.parsing.sympy_parser import null
from sympy.polys.polytools import Poly
from sympy.printing.repr import srepr
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.abc import x, y
from sympy.core.sympify import (sympify, _sympify, SympifyError, kernS,
    CantSympify, converter)
from sympy.core.decorators import _sympifyit
from sympy.external import import_module
from sympy.testing.pytest import raises, XFAIL, skip
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.geometry import Point, Line
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.abc import _clash, _clash1, _clash2
from sympy.external.gmpy import gmpy as _gmpy, flint as _flint
from sympy.sets import FiniteSet, EmptySet
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray

# 导入 mpmath 库
import mpmath
# 导入 collections 库中的 defaultdict 和 OrderedDict 类
from collections import defaultdict, OrderedDict

# 尝试导入 numpy 库，如果失败则为 numpy 赋值 None
numpy = import_module('numpy')

# 定义测试函数 test_issue_3538，测试 sympify 函数的行为
def test_issue_3538():
    # 使用 sympify 函数将字符串 "exp(x)" 转换为表达式对象 v
    v = sympify("exp(x)")
    # 断言 v 等于 sympy 中的 exp(x) 表达式
    assert v == exp(x)
    # 断言 v 的类型与 exp(x) 的类型相同
    assert type(v) == type(exp(x))
    # 断言 v 的类型的字符串表示与 exp(x) 的类型的字符串表示相同
    assert str(type(v)) == str(type(exp(x)))

# 定义测试函数 test_sympify1，测试 sympify 函数在不同输入下的行为
def test_sympify1():
    # 断言 sympify("x") 返回符号对象 Symbol("x")
    assert sympify("x") == Symbol("x")
    # 断言 sympify("   x") 返回符号对象 Symbol("x")
    assert sympify("   x") == Symbol("x")
    # 断言 sympify("   x   ") 返回符号对象 Symbol("x")
    assert sympify("   x   ") == Symbol("x")
    # issue 4877
    # 测试 sympify 函数处理不同格式的数字字符串的情况
    assert sympify('--.5') == 0.5
    assert sympify('-1/2') == -S.Half
    assert sympify('-+--.5') == -0.5
    assert sympify('-.[3]') == Rational(-1, 3)
    assert sympify('.[3]') == Rational(1, 3)
    assert sympify('+.[3]') == Rational(1, 3)
    assert sympify('+0.[3]*10**-2') == Rational(1, 300)
    assert sympify('.[052631578947368421]') == Rational(1, 19)
    assert sympify('.0[526315789473684210]') == Rational(1, 19)
    assert sympify('.034[56]') == Rational(1711, 49500)
    # options to make reals into rationals
    # 测试 sympify 函数在有理化选项为真时的行为
    assert sympify('1.22[345]', rational=True) == \
        1 + Rational(22, 100) + Rational(345, 99900)
    assert sympify('2/2.6', rational=True) == Rational(10, 13)
    assert sympify('2.6/2', rational=True) == Rational(13, 10)
    assert sympify('2.6e2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e+2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e-2/17', rational=True) == Rational(26, 17000)
    assert sympify('2.1+3/4', rational=True) == \
        Rational(21, 10) + Rational(3, 4)
    # 使用 sympify 函数将字符串转换为 SymPy 的表达式对象，并使用 rational=True 选项尝试将其转换为有理数形式。
    assert sympify('2.234456', rational=True) == Rational(279307, 125000)
    
    # 使用 sympify 函数将科学计数法表示的字符串转换为 SymPy 的表达式对象，并尝试转换为有理数形式。
    assert sympify('2.234456e23', rational=True) == 223445600000000000000000
    
    # 使用 sympify 函数将科学计数法表示的字符串转换为 SymPy 的表达式对象，并尝试转换为有理数形式。
    assert sympify('2.234456e-23', rational=True) == \
        Rational(279307, 12500000000000000000000000000)
    
    # 使用 sympify 函数将负科学计数法表示的字符串转换为 SymPy 的表达式对象，并尝试转换为有理数形式。
    assert sympify('-2.234456e-23', rational=True) == \
        Rational(-279307, 12500000000000000000000000000)
    
    # 使用 sympify 函数将分数表示的字符串转换为 SymPy 的表达式对象，并尝试转换为有理数形式。
    assert sympify('12345678901/17', rational=True) == \
        Rational(12345678901, 17)
    
    # 使用 sympify 函数将包含变量 x 的表达式字符串转换为 SymPy 的表达式对象，并尝试转换为有理数形式。
    assert sympify('1/.3 + x', rational=True) == Rational(10, 3) + x
    
    # 确保长分数工作正常
    assert sympify('222222222222/11111111111') == \
        Rational(222222222222, 11111111111)
    
    # 确保从循环小数表示转换的分数也可以正确处理
    assert sympify('1/.2[123456789012]') == Rational(333333333333, 70781892967)
    
    # 确保从高精度实数表示转换的分数也可以正确处理
    assert sympify('.1234567890123456', rational=True) == \
        Rational(19290123283179, 156250000000000)
# 定义一个测试函数，用于测试 sympify 函数能否正确处理 fractions 模块中的 Fraction 对象
def test_sympify_Fraction():
    try:
        import fractions  # 尝试导入 fractions 模块
    except ImportError:
        pass  # 如果导入失败则忽略，继续执行后续代码
    else:
        # 使用 sympify 函数将 Fraction(101, 127) 转换为 SymPy 中的 Rational 对象，并进行断言检查
        value = sympify(fractions.Fraction(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational


# 定义一个测试函数，用于测试 sympify 函数能否正确处理 gmpy2 模块中的 mpz 和 mpq 对象
def test_sympify_gmpy():
    if _gmpy is not None:
        import gmpy2  # 如果 _gmpy 可用，则导入 gmpy2 模块

        # 使用 sympify 函数将 gmpy2.mpz(1000001) 转换为 SymPy 中的 Integer 对象，并进行断言检查
        value = sympify(gmpy2.mpz(1000001))
        assert value == Integer(1000001) and type(value) is Integer

        # 使用 sympify 函数将 gmpy2.mpq(101, 127) 转换为 SymPy 中的 Rational 对象，并进行断言检查
        value = sympify(gmpy2.mpq(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational


# 定义一个测试函数，用于测试 sympify 函数能否正确处理 flint 模块中的 fmpz 和 fmpq 对象
def test_sympify_flint():
    if _flint is not None:
        import flint  # 如果 _flint 可用，则导入 flint 模块

        # 使用 sympify 函数将 flint.fmpz(1000001) 转换为 SymPy 中的 Integer 对象，并进行断言检查
        value = sympify(flint.fmpz(1000001))
        assert value == Integer(1000001) and type(value) is Integer

        # 使用 sympify 函数将 flint.fmpq(101, 127) 转换为 SymPy 中的 Rational 对象，并进行断言检查
        value = sympify(flint.fmpq(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational


# 定义一个修饰器函数，用于在测试 sympify 函数时保留 mpmath 的当前小数位设置
@conserve_mpmath_dps
def test_sympify_mpmath():
    # 使用 sympify 函数将 mpmath.mpf(1.0) 转换为 SymPy 中的 Float 对象，并进行断言检查
    value = sympify(mpmath.mpf(1.0))
    assert value == Float(1.0) and type(value) is Float

    # 修改 mpmath 的小数位精度为 12
    mpmath.mp.dps = 12
    # 对比 sympify(mpmath.pi) 是否在给定的精度范围内等于预期值，进行断言检查
    assert sympify(mpmath.pi).epsilon_eq(Float("3.14159265359"), Float("1e-12")) == True
    assert sympify(mpmath.pi).epsilon_eq(Float("3.14159265359"), Float("1e-13")) == False

    # 恢复 mpmath 的小数位精度为 6
    mpmath.mp.dps = 6
    # 对比 sympify(mpmath.pi) 是否在给定的精度范围内等于预期值，进行断言检查
    assert sympify(mpmath.pi).epsilon_eq(Float("3.14159"), Float("1e-5")) == True
    assert sympify(mpmath.pi).epsilon_eq(Float("3.14159"), Float("1e-6")) == False

    # 修改 mpmath 的小数位精度为 15
    mpmath.mp.dps = 15
    # 使用 sympify 函数将 mpmath.mpc(1.0 + 2.0j) 转换为 SymPy 中的复数对象，并进行断言检查
    assert sympify(mpmath.mpc(1.0 + 2.0j)) == Float(1.0) + Float(2.0)*I


# 定义一个测试函数，用于测试 sympify 函数对自定义类 A 的处理
def test_sympify2():
    class A:
        def _sympy_(self):
            return Symbol("x")**3  # 定义 _sympy_ 方法返回 Symbol("x") 的三次方

    a = A()

    # 使用 _sympify 函数将自定义类 A 的实例 a 转换为 SymPy 表达式，并进行断言检查
    assert _sympify(a) == x**3
    # 使用 sympify 函数将自定义类 A 的实例 a 转换为 SymPy 表达式，并进行断言检查
    assert sympify(a) == x**3
    # 进行 a 和 x**3 的相等性断言检查
    assert a == x**3


# 定义一个测试函数，用于测试 sympify 函数对字符串表达式的处理
def test_sympify3():
    # 使用 sympify 函数将字符串表达式 "x**3" 转换为 SymPy 表达式，并进行断言检查
    assert sympify("x**3") == x**3
    # 使用 sympify 函数将字符串表达式 "x^3" 转换为 SymPy 表达式，并进行断言检查
    assert sympify("x^3") == x**3
    # 使用 sympify 函数将字符串表达式 "1/2" 转换为 SymPy 表达式，并进行断言检查
    assert sympify("1/2") == Integer(1)/2

    # 使用 raises 函数验证当使用 _sympify 函数处理非法字符串表达式时是否会引发 SympifyError
    raises(SympifyError, lambda: _sympify('x**3'))
    raises(SympifyError, lambda: _sympify('1/2'))


# 定义一个测试函数，用于测试 sympify 函数对关键字的处理
def test_sympify_keywords():
    # 使用 raises 函数验证当使用 sympify 函数处理包含关键字的字符串表达式时是否会引发 SympifyError
    raises(SympifyError, lambda: sympify('if'))
    raises(SympifyError, lambda: sympify('for'))
    raises(SympifyError, lambda: sympify('while'))
    raises(SympifyError, lambda: sympify('lambda'))


# 定义一个测试函数，用于测试 sympify 函数对浮点数字符串的处理
def test_sympify_float():
    # 使用 sympify 函数将字符串表达式 "1e-64" 转换为 SymPy 表达式，并进行非零断言检查
    assert sympify("1e-64") != 0
    # 使用 sympify 函数将字符串表达式 "1e-20000" 转换为 SymPy 表达式，并进行非零断言检查
    assert sympify("1e-20000") != 0


# 定义一个测试函数，用于测试 sympify 函数对布尔值的处理
def test_sympify_bool():
    # 使用 sympify 函数将布尔值 True 转换为 SymPy 中的 true 对象，并进行断言检查
    assert sympify(True) is true
    # 使用 sympify 函数将布尔值 False 转换为 SymPy 中的 false 对象，并进行断言检查
    assert sympify(False) is false


# 定义一个测试函数，用于测试 sympify 函数对可迭代对象的处理
def test_sympyify_iterables():
    ans = [Rational(3, 10), Rational(1, 5)]
    # 使用 sympify 函数将列表中的字符串表达式转换为 SymPy 中的有理数对象，并进行断言检查
    assert sympify(['.3', '.2'], rational=True) == ans
    # 使用 sympify 函数将字典中的字符串键和值转换为 SymPy 中的符号，并进行断言检查
    assert sympify({"x": 0, "y": 1}) ==
# 定义用于测试问题 16859 的函数
def test_issue_16859():
    # 定义一个类 no，继承自 float 和 CantSympify，但没有实现 _sympy_ 方法
    class no(float, CantSympify):
        pass
    # 使用 lambda 函数测试 sympify 函数是否能够捕获 SympifyError 异常
    raises(SympifyError, lambda: sympify(no(1.2)))


# 定义用于测试 sympify4 的函数
def test_sympify4():
    # 定义类 A，实现 _sympy_ 方法返回符号 "x"
    class A:
        def _sympy_(self):
            return Symbol("x")

    # 创建类 A 的实例
    a = A()

    # 断言使用 _sympify 函数对 a 的立方进行计算并与 x 的立方比较
    assert _sympify(a)**3 == x**3
    # 断言使用 sympify 函数对 a 的立方进行计算并与 x 的立方比较
    assert sympify(a)**3 == x**3
    # 断言 a 等于符号 "x"
    assert a == x


# 定义用于测试 sympify_text 的函数
def test_sympify_text():
    # 断言 sympify 函数能正确将字符串 'some' 转换为符号 'some'
    assert sympify('some') == Symbol('some')
    # 断言 sympify 函数能正确将字符串 'core' 转换为符号 'core'
    assert sympify('core') == Symbol('core')

    # 断言 sympify 函数能正确识别字符串 'True' 并返回 Python 中的 True 值
    assert sympify('True') is True
    # 断言 sympify 函数能正确识别字符串 'False' 并返回 Python 中的 False 值
    assert sympify('False') is False

    # 断言 sympify 函数能正确识别字符串 'Poly' 并返回 sympy 中的 Poly 对象
    assert sympify('Poly') == Poly
    # 断言 sympify 函数能正确识别字符串 'sin' 并返回 sympy 中的 sin 函数
    assert sympify('sin') == sin


# 定义用于测试 sympify_function 的函数
def test_sympify_function():
    # 断言 sympify 函数能正确处理包含表达式的字符串，例如 'factor(x**2-1, x)'
    assert sympify('factor(x**2-1, x)') == -(1 - x)*(x + 1)
    # 断言 sympify 函数能正确处理包含数学常数的表达式，例如 'sin(pi/2)*cos(pi)'
    assert sympify('sin(pi/2)*cos(pi)') == -Integer(1)


# 定义用于测试 sympify_poly 的函数
def test_sympify_poly():
    # 创建一个 Poly 对象 p = x^2 + x + 1
    p = Poly(x**2 + x + 1, x)

    # 断言 _sympify 函数能正确识别 Poly 对象并返回其自身
    assert _sympify(p) is p
    # 断言 sympify 函数能正确识别 Poly 对象并返回其自身
    assert sympify(p) is p


# 定义用于测试 sympify_factorial 的函数
def test_sympify_factorial():
    # 断言 sympify 函数能正确识别字符串 'x!' 并返回阶乘 factorial(x)
    assert sympify('x!') == factorial(x)
    # 断言 sympify 函数能正确识别字符串 '(x+1)!' 并返回阶乘 factorial(x + 1)
    assert sympify('(x+1)!') == factorial(x + 1)
    # 断言 sympify 函数能正确识别复杂的阶乘表达式，例如 '(1 + y*(x + 1))!' 并返回其阶乘
    assert sympify('(1 + y*(x + 1))!') == factorial(1 + y*(x + 1))
    # 断言 sympify 函数能正确识别带有阶乘的表达式并计算其平方，例如 '(1 + y*(x + 1)!)^2'
    assert sympify('(1 + y*(x + 1)!)^2') == (1 + y*factorial(x + 1))**2
    # 断言 sympify 函数能正确识别带有变量和阶乘的表达式，例如 'y*x!'
    assert sympify('y*x!') == y*factorial(x)
    # 断言 sympify 函数能正确识别双阶乘表达式，例如 'x!!'
    assert sympify('x!!') == factorial2(x)
    # 断言 sympify 函数能正确识别双阶乘表达式，例如 '(x+1)!!'
    assert sympify('(x+1)!!') == factorial2(x + 1)
    # 断言 sympify 函数能正确识别复杂的双阶乘表达式，例如 '(1 + y*(x + 1))!!'
    assert sympify('(1 + y*(x + 1))!!') == factorial2(1 + y*(x + 1))
    # 断言 sympify 函数能正确识别带有双阶乘的表达式并计算其平方，例如 '(1 + y*(x + 1)!!)^2'
    assert sympify('(1 + y*(x + 1)!!)^2') == (1 + y*factorial2(x + 1))**2
    # 断言 sympify 函数能正确识别带有变量和双阶乘的表达式，例如 'y*x!!'
    assert sympify('y*x!!') == y*factorial2(x)
    # 断言 sympify 函数能正确识别嵌套阶乘表达式并返回其阶乘 factorial(factorial2(x))
    assert sympify('factorial2(x)!') == factorial(factorial2(x))

    # 使用 lambda 函数测试 sympify 函数能否捕获 SympifyError 异常
    raises(SympifyError, lambda: sympify("+!!"))
    raises(SympifyError, lambda: sympify(")!!"))
    raises(SympifyError, lambda: sympify("!"))
    raises(SympifyError, lambda: sympify("(!)"))
    raises(SympifyError, lambda: sympify("x!!!"))


# 定义用于测试问题 3595 的函数
def test_issue_3595():
    # 断言 sympify 函数能正确识别变量名 'a_' 并返回相应的符号对象 Symbol("a_")
    assert sympify("a_") == Symbol("a_")
    # 断言 sympify 函数能正确识别变量名 '_a' 并返回相应的符号对象 Symbol("_a")


# 定义用于测试 lambda 表达式的函数
def test_lambda():
    # 创建一个符号 'x'
    x = Symbol('x')
    # 断言 sympify 函数能正确识别字符串 'lambda: 1' 并返回 Lambda((), 1)
    assert sympify('lambda: 1') == Lambda((), 1)
    # 断言 sympify 函数能正确识别字符串 'lambda x: x' 并返回 Lambda(x, x)
    assert sympify('lambda x: x') == Lambda(x, x)
    # 断言 sympify 函数能正确识别字符串 'lambda x: 2*x' 并返回 Lambda(x, 2*x)
    assert sympify('lambda x: 2*x') == Lambda(x, 2*x)
    # 断言 sympify 函数能正确识别字符串 'lambda x, y: 2*x+y' 并返回 Lambda((x, y), 2*x + y)


# 定义用于测试 lambda 表达式异常情况的函数
def test_lambda_raises():
    # 使用 lambda 函数测试 sympify 函数能否捕获 SympifyError 异常，对于带有 *args 的 lambda 表达式
    raises(SympifyError, lambda: sympify("lambda *args: args")) # args 参数错误
    # 使用 lambda 函数测试 sympify 函数能否捕获 SympifyError 异常，对于带有 **kwargs 的 lambda 表达式
    raises(SympifyError, lambda: sympify("lambda **kwargs: kwargs[0]"))
    # 使用 sympify 函数确保对象能被 Sympy 解析，但不适用于 _sympify 函数。
    # 参考 GitHub 问题链接：https://github.com/sympy/sympy/issues/20124
    assert sympify(f) is f
    # 断言调用 _sympify 函数会引发 SympifyError 异常
    raises(SympifyError, lambda: _sympify(f))

    # 定义类 A
    class A:
        # 定义 _sympy_ 方法，返回整数 5
        def _sympy_(self):
            return Integer(5)

    # 创建类 A 的实例 a
    a = A()
    # 断言调用 _sympify 函数将实例 a 转换为整数 5
    assert _sympify(a) == Integer(5)

    # 负面测试 _sympify 函数
    # 断言调用 _sympify 函数分别会引发 SympifyError 异常，参数为字符串 '1' 和列表 [1, 2, 3]
    raises(SympifyError, lambda: _sympify('1'))
    raises(SympifyError, lambda: _sympify([1, 2, 3]))
def test_sympifyit():
    x = Symbol('x')  # 创建符号变量 x
    y = Symbol('y')  # 创建符号变量 y

    @_sympifyit('b', NotImplemented)
    def add(a, b):
        return a + b  # 定义一个函数 add，将 a 和 b 相加

    assert add(x, 1) == x + 1  # 测试 add 函数，确保能正确计算 x + 1
    assert add(x, 0.5) == x + Float('0.5')  # 测试 add 函数，确保能正确处理浮点数
    assert add(x, y) == x + y  # 测试 add 函数，确保能正确计算 x + y

    assert add(x, '1') == NotImplemented  # 测试 add 函数，确保能处理非法的输入 '1'

    @_sympifyit('b')
    def add_raises(a, b):
        return a + b  # 定义另一个函数 add_raises，将 a 和 b 相加

    assert add_raises(x, 1) == x + 1  # 测试 add_raises 函数，确保能正确计算 x + 1
    assert add_raises(x, 0.5) == x + Float('0.5')  # 测试 add_raises 函数，确保能正确处理浮点数
    assert add_raises(x, y) == x + y  # 测试 add_raises 函数，确保能正确计算 x + y

    raises(SympifyError, lambda: add_raises(x, '1'))  # 测试 add_raises 函数，确保能正确抛出异常 SympifyError


def test_int_float():
    class F1_1:
        def __float__(self):
            return 1.1

    class F1_1b:
        """
        This class is still a float, even though it also implements __int__().
        """
        def __float__(self):
            return 1.1

        def __int__(self):
            return 1

    class F1_1c:
        """
        This class is still a float, because it implements _sympy_()
        """
        def __float__(self):
            return 1.1

        def __int__(self):
            return 1

        def _sympy_(self):
            return Float(1.1)

    class I5:
        def __int__(self):
            return 5

    class I5b:
        """
        This class implements both __int__() and __float__(), so it will be
        treated as Float in SymPy. One could change this behavior, by using
        float(a) == int(a), but deciding that integer-valued floats represent
        exact numbers is arbitrary and often not correct, so we do not do it.
        If, in the future, we decide to do it anyway, the tests for I5b need to
        be changed.
        """
        def __float__(self):
            return 5.0

        def __int__(self):
            return 5

    class I5c:
        """
        This class implements both __int__() and __float__(), but also
        a _sympy_() method, so it will be Integer.
        """
        def __float__(self):
            return 5.0

        def __int__(self):
            return 5

        def _sympy_(self):
            return Integer(5)

    i5 = I5()  # 创建一个 I5 类的实例 i5
    i5b = I5b()  # 创建一个 I5b 类的实例 i5b
    i5c = I5c()  # 创建一个 I5c 类的实例 i5c
    f1_1 = F1_1()  # 创建一个 F1_1 类的实例 f1_1
    f1_1b = F1_1b()  # 创建一个 F1_1b 类的实例 f1_1b
    f1_1c = F1_1c()  # 创建一个 F1_1c 类的实例 f1_1c

    assert sympify(i5) == 5  # 测试 sympify 函数，确保能正确将 i5 转换为整数类型
    assert isinstance(sympify(i5), Integer)  # 确保 sympify 函数返回的对象是 Integer 类型
    assert sympify(i5b) == 5.0  # 测试 sympify 函数，确保能正确将 i5b 转换为浮点数类型
    assert isinstance(sympify(i5b), Float)  # 确保 sympify 函数返回的对象是 Float 类型
    assert sympify(i5c) == 5  # 测试 sympify 函数，确保能正确将 i5c 转换为整数类型
    assert isinstance(sympify(i5c), Integer)  # 确保 sympify 函数返回的对象是 Integer 类型
    assert abs(sympify(f1_1) - 1.1) < 1e-5  # 测试 sympify 函数，确保能正确将 f1_1 转换为接近 1.1 的浮点数
    assert abs(sympify(f1_1b) - 1.1) < 1e-5  # 测试 sympify 函数，确保能正确将 f1_1b 转换为接近 1.1 的浮点数
    assert abs(sympify(f1_1c) - 1.1) < 1e-5  # 测试 sympify 函数，确保能正确将 f1_1c 转换为接近 1.1 的浮点数

    assert _sympify(i5) == 5  # 测试 _sympify 函数，确保能正确将 i5 转换为整数类型
    assert isinstance(_sympify(i5), Integer)  # 确保 _sympify 函数返回的对象是 Integer 类型
    assert _sympify(i5b) == 5.0  # 测试 _sympify 函数，确保能正确将 i5b 转换为浮点数类型
    assert isinstance(_sympify(i5b), Float)  # 确保 _sympify 函数返回的对象是 Float 类型
    assert _sympify(i5c) == 5  # 测试 _sympify 函数，确保能正确将 i5c 转换为整数类型
    assert isinstance(_sympify(i5c), Integer)  # 确保 _sympify 函数返回的对象是 Integer 类型
    assert abs(_sympify(f1_1) - 1.1) < 1e-5  # 测试 _sympify 函数，确保能正确将 f1_1 转换为接近 1.1 的浮点数
    assert abs(_sympify(f1_1b) - 1.1) < 1e-5  # 测试 _sympify 函数，确保能正确将 f1_1b 转换为接近 1.1 的浮点数
    assert abs(_sympify(f1_1c) - 1.1) < 1e-5  # 测试 _sympify 函数，确保能正确将 f1_1c 转换为接近 1.1 的浮点数
    # 定义一个包含不同数学表达式和对应的未计算形式的字典
    cases = {
        '2 + 3': Add(2, 3, evaluate=False),  # 表达式 "2 + 3" 的未计算形式，不进行计算
        '2**2 / 3': Mul(Pow(2, 2, evaluate=False), Pow(3, -1, evaluate=False), evaluate=False),
        # 表达式 "2**2 / 3" 的未计算形式，包括乘方和除法，不进行计算
        '2 + 3 * 5': Add(2, Mul(3, 5, evaluate=False), evaluate=False),
        # 表达式 "2 + 3 * 5" 的未计算形式，包括加法和乘法，不进行计算
        '2 - 3 * 5': Add(2, Mul(-1, Mul(3, 5, evaluate=False), evaluate=False), evaluate=False),
        # 表达式 "2 - 3 * 5" 的未计算形式，包括减法和乘法，不进行计算
        '1 / 3': Mul(1, Pow(3, -1, evaluate=False), evaluate=False),
        # 表达式 "1 / 3" 的未计算形式，包括除法，不进行计算
        'True | False': Or(True, False, evaluate=False),  # 逻辑表达式 "True | False" 的未计算形式，不进行计算
        '1 + 2 + 3 + 5*3 + integrate(x)': Add(1, 2, 3, Mul(5, 3, evaluate=False), x**2/2, evaluate=False),
        # 复合表达式 "1 + 2 + 3 + 5*3 + integrate(x)" 的未计算形式，不进行计算
        '2 * 4 * 6 + 8': Add(Mul(2, 4, 6, evaluate=False), 8, evaluate=False),
        # 表达式 "2 * 4 * 6 + 8" 的未计算形式，包括乘法和加法，不进行计算
        '2 - 8 / 4': Add(2, Mul(-1, Mul(8, Pow(4, -1, evaluate=False), evaluate=False), evaluate=False), evaluate=False),
        # 表达式 "2 - 8 / 4" 的未计算形式，包括减法、乘法和除法，不进行计算
        '2 - 2**2': Add(2, Mul(-1, Pow(2, 2, evaluate=False), evaluate=False), evaluate=False),
        # 表达式 "2 - 2**2" 的未计算形式，包括减法和乘方，不进行计算
    }
    
    # 遍历每个测试用例，并断言未计算形式是否等于给定的结果
    for case, result in cases.items():
        assert sympify(case, evaluate=False) == result
# 定义测试函数，用于检查 sympify 函数的各种用例
def test_issue_4133():
    # 使用 sympify 将字符串 'Integer(4)' 转换为 sympy 的表达式对象
    a = sympify('Integer(4)')

    # 断言 a 应该等于 Integer(4)
    assert a == Integer(4)
    # 断言 a 是一个整数类型的对象
    assert a.is_Integer


# 定义测试函数，用于检查 sympify 函数对列表、元组、集合的处理
def test_issue_3982():
    # 创建列表 a
    a = [3, 2.0]
    # 断言 sympify 对列表 a 的处理结果应该是 [Integer(3), Float(2.0)]
    assert sympify(a) == [Integer(3), Float(2.0)]
    # 断言 sympify 对元组 tuple(a) 的处理结果应该是 Tuple(Integer(3), Float(2.0))
    assert sympify(tuple(a)) == Tuple(Integer(3), Float(2.0))
    # 断言 sympify 对集合 set(a) 的处理结果应该是 FiniteSet(Integer(3), Float(2.0))
    assert sympify(set(a)) == FiniteSet(Integer(3), Float(2.0))


# 定义测试函数，用于检查 sympify 函数与 S 符号的交互
def test_S_sympify():
    # 断言 S(1)/2、sympify(1)/2 和 S.Half 应该相等
    assert S(1)/2 == sympify(1)/2 == S.Half
    # 断言 (-2)**(S(1)/2) 应该等于 sqrt(2)*I
    assert (-2)**(S(1)/2) == sqrt(2)*I


# 定义测试函数，用于检查 sympify 函数在处理复数时的情况
def test_issue_4788():
    # 断言对于复数 S(1.0 + 0J)，其 srepr 结果应该等于 srepr(S(1.0)) 和 srepr(Float(1.0))
    assert srepr(S(1.0 + 0J)) == srepr(S(1.0)) == srepr(Float(1.0))


# 定义测试函数，用于检查 sympify 函数对 None 的处理
def test_issue_4798_None():
    # 断言 sympify(S(None)) 应该返回 None
    assert S(None) is None


# 定义测试函数，用于检查 sympify 函数在处理字符串表达式时的情况
def test_issue_3218():
    # 断言 sympify("x+\ny") 应该等于 x + y
    assert sympify("x+\ny") == x + y


# 定义测试函数，用于检查 sympify 函数与 numpy 结合使用时的情况
def test_issue_19399():
    # 如果没有安装 numpy，则跳过这个测试
    if not numpy:
        skip("numpy not installed.")

    # 创建 numpy 数组 a
    a = numpy.array(Rational(1, 2))
    # 创建有理数 b
    b = Rational(1, 3)
    # 断言 a*b 的结果及其类型应该与 b*a 的结果及其类型相同
    assert (a * b, type(a * b)) == (b * a, type(b * a))


# 定义测试函数，用于检查 sympify 函数对内置符号的处理
def test_issue_4988_builtins():
    # 创建符号 C
    C = Symbol('C')
    # 创建变量字典 vars 包含符号 C
    vars = {'C': C}
    # 使用 sympify 将字符串 'C' 转换为符号表达式 exp1
    exp1 = sympify('C')
    # 断言 exp1 应该等于 C，确保没有与 sympy.C 混淆
    assert exp1 == C

    # 使用 vars 变量字典将字符串 'C' 转换为符号表达式 exp2
    exp2 = sympify('C', vars)
    # 断言 exp2 应该等于 C，确保没有与 sympy.C 混淆
    assert exp2 == C


# 定义测试函数，用于检查 sympify 函数在处理几何对象时的情况
def test_geometry():
    # 使用 sympify 将 Point(0, 1) 转换为 sympy 的点对象 p
    p = sympify(Point(0, 1))
    # 断言 p 应该等于 Point(0, 1) 并且是 Point 类的实例
    assert p == Point(0, 1) and isinstance(p, Point)

    # 使用 sympify 将 Line(p, (1, 0)) 转换为 sympy 的线对象 L
    L = sympify(Line(p, (1, 0)))
    # 断言 L 应该等于 Line((0, 1), (1, 0)) 并且是 Line 类的实例
    assert L == Line((0, 1), (1, 0)) and isinstance(L, Line)


# 定义测试函数，用于检查 kernS 函数的多个问题
def test_kernS():
    # 定义字符串 s
    s = '-1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x)))'
    # 第一个断言，确保表达式求值正确
    assert -1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) == -1

    # 调用 kernS 函数处理字符串 s，并赋值给 ss
    ss = kernS(s)
    # 第二个断言，确保 ss 不等于 -1，且 ss 简化后应该等于 -1
    assert ss != -1 and ss.simplify() == -1

    # 修改字符串 s 中的 'x' 为 '_kern'，再次调用 kernS 函数处理
    s = '-1 - 2*(-(-_kern + 1/_kern)/(_kern*(_kern - 1/_kern)**2) - 1/(_kern*(_kern - 1/_kern)))'
    ss = kernS(s)
    # 第三个断言，确保 ss 不等于 -1，且 ss 简化后应该等于 -1
    assert ss != -1 and ss.simplify() == -1

    # issue 6687 测试
    assert (kernS('Interval(-1,-2 - 4*(-3))')
        == Interval(-1, Add(-2, Mul(12, 1, evaluate=False), evaluate=False)))
    # 断言 kernS 处理 '_kern' 字符串应该返回 Symbol('_kern')
    assert kernS('_kern') == Symbol('_kern')
    # 断言 kernS 处理 'E**-(x)' 字符串应该返回 exp(-x)
    assert kernS('E**-(x)') == exp(-x)

    # 定义表达式 e
    e = 2*(x + y)*y
    # 断言 kernS 处理多种表达方式时应该返回相同的结果列表
    assert kernS(['2*(x + y)*y', ('2*(x + y)*y',)]) == [e, (e,)]
    # 断言 kernS 处理 '-(2*sin(x)**2 + 2*sin(x)*cos(x))*y/2' 字符串应该返回符号表达式
    assert kernS('-(2*sin(x)**2 + 2*sin(x)*cos(x))*y/2') == \
        -y*(2*sin(x)**2 + 2*sin(x)*cos(x))/2

    # issue 15132 测试
    assert kernS('(1 - x)/(1 - x*(1-y))') == kernS('(1-x)/(1-(1-y)*x)')
    assert kernS('(1-2**-(4+1)*(1-y)*x)') == (1 - x*(1 - y)/32)
    assert kernS('(1-2**(4+1)*(1-y)*x)') == (1 - 32*x*(1 - y))
    assert kernS('(1-2.*(1-y)*x)') == 1 - 2.*x*(1 - y)

    # 断言 kernS 处理表达式 "(2*x)/(x-1)" 应该得到正确的结果
    one = kernS('x - (x - 1)')
    # 确保使用 S 函数计算 "Q & C" 的字符串表示，并验证结果为 'C & Q'
    assert str(S("Q & C", locals=_clash1)) == 'C & Q'
    # 确保使用 S 函数计算 'pi(x)' 的字符串表示，并验证结果为 'pi(x)'
    assert str(S('pi(x)', locals=_clash2)) == 'pi(x)'
    # 初始化一个空的 locals 字典
    locals = {}
    # 在 locals 字典中执行动态代码 "from sympy.abc import Q, C"，导入 Q 和 C 符号
    exec("from sympy.abc import Q, C", locals)
    # 确保使用 S 函数计算 'C&Q' 的字符串表示，并验证结果为 'C & Q'
    assert str(S('C&Q', locals)) == 'C & Q'
    # 使用 _clash 作为 locals 参数，确保使用 S 函数计算 'pi(C, Q)' 的字符串表示，并验证结果为 'pi(C, Q)'
    assert str(S('pi(C, Q)', locals=_clash)) == 'pi(C, Q)'
    # 使用 _clash2 作为 locals 参数，确保计算 'pi + x' 表达式的自由符号个数为 2
    assert len(S('pi + x', locals=_clash2).free_symbols) == 2
    # 使用 _clash2 作为 locals 参数，确保计算 'pi + pi(x)' 引发 TypeError 异常
    raises(TypeError, lambda: S('pi + pi(x)', locals=_clash2))
    # 确保所有 _clash, _clash1, _clash2 的值均为 {null}
    assert all(set(i.values()) == {null} for i in (
        _clash, _clash1, _clash2))
# 测试函数：验证从字符串中高精度计算出的 pi 值，并检查其正弦值是否接近于零
def test_issue_8821_highprec_from_str():
    # 使用 SymPy 计算 pi 的高精度值并转换为字符串
    s = str(pi.evalf(128))
    # 将字符串表示的数学表达式转换为 SymPy 表达式
    p = sympify(s)
    # 断言计算出的 sin(p) 的绝对值小于给定的误差限
    assert Abs(sin(p)) < 1e-127


# 测试函数：验证 SymPy 对 numpy 数组的符号化处理
def test_issue_10295():
    # 如果 numpy 模块不存在，则跳过这个测试
    if not numpy:
        skip("numpy not installed.")

    # 创建一个 numpy 数组 A
    A = numpy.array([[1, 3, -1],
                     [0, 1, 7]])
    # 将 numpy 数组 A 转换为 SymPy 的矩阵表示
    sA = S(A)
    # 断言 sA 的形状是否与原始数组 A 的形状相同
    assert sA.shape == (2, 3)
    # 使用 numpy.ndenumerate 遍历 A 中的元素及其索引
    for (ri, ci), val in numpy.ndenumerate(A):
        # 断言 sA 中的对应元素与原始数组 A 中的元素相等
        assert sA[ri, ci] == val

    # 创建一个包含符号变量的 numpy 数组 B
    B = numpy.array([-7, x, 3*y**2])
    # 将 numpy 数组 B 转换为 SymPy 的向量表示
    sB = S(B)
    # 断言 sB 的形状是否与原始数组 B 的形状相同
    assert sB.shape == (3,)
    # 逐个断言 sB 中的元素与原始数组 B 中的元素相等
    assert B[0] == sB[0] == -7
    assert B[1] == sB[1] == x
    assert B[2] == sB[2] == 3*y**2

    # 创建一个连续整数的 numpy 数组 C，并改变其形状为 (2, 3, 4)
    C = numpy.arange(0, 24)
    C.resize(2,3,4)
    # 将 numpy 数组 C 转换为 SymPy 的多维数组表示
    sC = S(C)
    # 断言 sC 的某个元素是整数
    assert sC[0, 0, 0].is_integer
    # 断言 sC 的某个元素确实等于 0
    assert sC[0, 0, 0] == 0

    # 创建两个 numpy 数组 a1 和 a2，并进行形状调整
    a1 = numpy.array([1, 2, 3])
    a2 = numpy.array(list(range(24)))
    a2.resize(2, 4, 3)
    # 断言 sympify 函数对 a1 和 a2 的符号化结果是否符合预期
    assert sympify(a1) == ImmutableDenseNDimArray([1, 2, 3])
    assert sympify(a2) == ImmutableDenseNDimArray(list(range(24)), (2, 4, 3))


# 测试函数：验证 SymPy 对 Python 3 中 range 函数的符号化处理
def test_Range():
    # 断言 sympify 函数对 range(10) 的符号化结果是否与 Range(10) 相等
    assert sympify(range(10)) == Range(10)
    # 断言 _sympify 函数对 range(10) 的符号化结果是否与 Range(10) 相等
    assert _sympify(range(10)) == Range(10)


# 测试函数：验证 SymPy 对集合的符号化处理
def test_sympify_set():
    # 创建一个符号变量 n
    n = Symbol('n')
    # 断言 sympify 函数对集合 {n} 的符号化结果是否为 FiniteSet(n)
    assert sympify({n}) == FiniteSet(n)
    # 断言 sympify 函数对空集的符号化结果是否为 EmptySet


# 测试函数：验证 SymPy 对 numpy 数组的符号化处理
def test_sympify_numpy():
    # 如果 numpy 模块不存在，则跳过这个测试
    if not numpy:
        skip('numpy not installed. Abort numpy tests.')
    np = numpy

    # 定义一个函数用于比较两个对象是否相等
    def equal(x, y):
        return x == y and type(x) == type(y)

    # 断言 sympify 函数对 numpy 不同数据类型的符号化结果是否符合预期
    assert sympify(np.bool_(1)) is S(True)
    try:
        assert equal(
            sympify(np.int_(1234567891234567891)), S(1234567891234567891))
        assert equal(
            sympify(np.intp(1234567891234567891)), S(1234567891234567891))
    except OverflowError:
        # 在 32 位系统上可能会出现 OverflowError
        pass
    assert equal(sympify(np.intc(1234567891)), S(1234567891))
    assert equal(sympify(np.int8(-123)), S(-123))
    assert equal(sympify(np.int16(-12345)), S(-12345))
    assert equal(sympify(np.int32(-1234567891)), S(-1234567891))
    assert equal(
        sympify(np.int64(-1234567891234567891)), S(-1234567891234567891))
    assert equal(sympify(np.uint8(123)), S(123))
    assert equal(sympify(np.uint16(12345)), S(12345))
    assert equal(sympify(np.uint32(1234567891)), S(1234567891))
    assert equal(
        sympify(np.uint64(1234567891234567891)), S(1234567891234567891))
    assert equal(sympify(np.float32(1.123456)), Float(1.123456, precision=24))
    assert equal(sympify(np.float64(1.1234567891234)),
                Float(1.1234567891234, precision=53))

    # 对于 np.longdouble 和其他扩展精度数据类型，精确的测试结果取决于平台
    ldprec = np.finfo(np.longdouble(1)).nmant + 1
    assert equal(sympify(np.longdouble(1.123456789)),
                 Float(1.123456789, precision=ldprec))

    assert equal(sympify(np.complex64(1 + 2j)), S(1.0 + 2.0*I))
    assert equal(sympify(np.complex128(1 + 2j)), S(1.0 + 2.0*I))

    lcprec = np.finfo(np.clongdouble(1)).nmant + 1
    # 断言：验证 sympify(np.clongdouble(1 + 2j)) 是否等于 Float(1.0, precision=lcprec) + Float(2.0, precision=lcprec)*I
    assert equal(sympify(np.clongdouble(1 + 2j)),
                Float(1.0, precision=lcprec) + Float(2.0, precision=lcprec)*I)

    # 如果 np.float96 在当前平台上存在
    if hasattr(np, 'float96'):
        # 计算 float96 的精度：通过创建 np.float96 类型的实例获取尾数位数 + 1
        f96prec = np.finfo(np.float96(1)).nmant + 1
        # 断言：验证 sympify(np.float96(1.123456789)) 是否等于 Float(1.123456789, precision=f96prec)
        assert equal(sympify(np.float96(1.123456789)),
                    Float(1.123456789, precision=f96prec))

    # 如果 np.float128 在当前平台上存在
    if hasattr(np, 'float128'):
        # 计算 float128 的精度：通过创建 np.float128 类型的实例获取尾数位数 + 1
        f128prec = np.finfo(np.float128(1)).nmant + 1
        # 断言：验证 sympify(np.float128(1.123456789123)) 是否等于 Float(1.123456789123, precision=f128prec)
        assert equal(sympify(np.float128(1.123456789123)),
                    Float(1.123456789123, precision=f128prec))
def test_sympify_rational_numbers_set():
    # 标记此测试为预期失败的测试
    ans = [Rational(3, 10), Rational(1, 5)]
    # 断言 sympify 函数将 {'.3', '.2'} 转换为有理数集合，与预期结果 ans 相同
    assert sympify({'.3', '.2'}, rational=True) == FiniteSet(*ans)


def test_sympify_mro():
    """Tests the resolution order for classes that implement _sympy_"""
    # 定义类 a，实现 _sympy_ 方法返回整数 1
    class a:
        def _sympy_(self):
            return Integer(1)
    # 定义类 b，继承自 a，实现 _sympy_ 方法返回整数 2
    class b(a):
        def _sympy_(self):
            return Integer(2)
    # 定义类 c，继承自 a，未实现 _sympy_ 方法

    # 断言 sympify 函数对不同类的实例化返回相应的整数
    assert sympify(a()) == Integer(1)
    assert sympify(b()) == Integer(2)
    assert sympify(c()) == Integer(1)


def test_sympify_converter():
    """Tests the resolution order for classes in converter"""
    # 定义类 a、b、c，无具体实现

    # 设置 converter[a] 和 converter[b]，分别使用 lambda 函数返回整数 1 和 2
    converter[a] = lambda x: Integer(1)
    converter[b] = lambda x: Integer(2)

    # 断言 sympify 函数对不同类的实例化返回相应的整数
    assert sympify(a()) == Integer(1)
    assert sympify(b()) == Integer(2)
    assert sympify(c()) == Integer(1)

    # 定义类 MyInteger 继承自 Integer

    # 如果 int 在 converter 中，则备份其转换器函数；否则 int_converter 为 None
    if int in converter:
        int_converter = converter[int]
    else:
        int_converter = None

    try:
        # 将 converter[int] 设置为 MyInteger 类型，断言 sympify(1) 返回 MyInteger(1)
        converter[int] = MyInteger
        assert sympify(1) == MyInteger(1)
    finally:
        # 最终恢复或删除 converter[int] 的设置
        if int_converter is None:
            del converter[int]
        else:
            converter[int] = int_converter


def test_issue_13924():
    # 如果 numpy 模块未安装，跳过测试
    if not numpy:
        skip("numpy not installed.")

    # 使用 sympify 函数处理 numpy 数组，断言结果为 ImmutableDenseNDimArray 类型
    a = sympify(numpy.array([1]))
    assert isinstance(a, ImmutableDenseNDimArray)
    assert a[0] == 1


def test_numpy_sympify_args():
    # 检查 sympify 函数是否能处理 numpy 类型参数的问题（例如 numpy.str_）
    # 如果 numpy 模块未安装，跳过测试
    if not numpy:
        skip("numpy not installed.")

    # 使用 sympify 函数处理 numpy.str_('a')，断言结果为 Symbol 类型且与预期 Symbol('a') 相同
    a = sympify(numpy.str_('a'))
    assert type(a) is Symbol
    assert a == Symbol('a')

    # 定义 CustomSymbol 类，继承自 Symbol

    # 使用 sympify 函数处理 numpy.str_('a')，传入自定义符号类参数，断言结果为自定义符号类型
    a = sympify(numpy.str_('a'), {"Symbol": CustomSymbol})
    assert isinstance(a, CustomSymbol)

    # 使用 sympify 函数处理 numpy.str_('x^y')，并禁用默认的 XOR 转换，断言结果为 x^y 表达式
    a = sympify(numpy.str_('x^y'))
    assert a == x**y
    # 使用 sympify 函数处理 numpy.str_('x^y')，并保留为未评估状态，断言结果为 Add(x, x, evaluate=False) 类型
    a = sympify(numpy.str_('x^y'), evaluate=False)
    assert a == Xor(x, y)

    # 使用 sympify 函数处理 numpy.str_('x')，设置 strict=True，预期抛出 SympifyError 异常
    raises(SympifyError, lambda: sympify(numpy.str_('x'), strict=True))

    # 使用 sympify 函数处理 numpy.str_('1.1')，断言结果为 Float 类型且与预期结果 1.1 相同
    a = sympify(numpy.str_('1.1'))
    assert isinstance(a, Float)
    assert a == 1.1

    # 使用 sympify 函数处理 numpy.str_('1.1')，设置 rational=True，断言结果为 Rational 类型且与预期结果 Rational(11, 10) 相同
    a = sympify(numpy.str_('1.1'), rational=True)
    assert isinstance(a, Rational)
    assert a == Rational(11, 10)

    # 使用 sympify 函数处理 numpy.str_('x + x')，断言结果为 Mul 类型且与预期结果 2*x 相同
    a = sympify(numpy.str_('x + x'))
    assert isinstance(a, Mul)
    assert a == 2*x

    # 使用 sympify 函数处理 numpy.str_('x + x')，设置 evaluate=False，断言结果为 Add 类型且与预期结果 Add(x, x, evaluate=False) 相同
    a = sympify(numpy.str_('x + x'), evaluate=False)
    assert isinstance(a, Add)
    assert a == Add(x, x, evaluate=False)


def test_issue_5939():
     # 定义符号 a 和 b
     a = Symbol('a')
     b = Symbol('b')
     # 使用 sympify 函数处理字符串 'a+\nb'，断言结果为 a + b
     assert sympify('''a+\nb''') == a + b


def test_issue_16759():
    # 使用 sympify 函数处理浮点数键的字典 {.5: 1}，断言结果中 S.Half 不在键集合中
    d = sympify({.5: 1})
    assert S.Half not in d
    # 断言浮点数键的字典中 Float(.5) 在键集合中，且对应值为 S.One
    assert Float(.5) in d
    assert d[.5] is S.One

    # 使用 sympify 函数处理有序字典 OrderedDict({.5: 1})，断言结果中 S.Half 不在键集合中
    d = sympify(OrderedDict({.5: 1}))
    assert S.Half not in d
    # 断言有序字典中 Float(.5) 在键集合中，且对应值为 S.One
    assert Float(.5) in d
    assert d[.5] is S.One

    # 使用 sympify 函数处理默认字典 defaultdict(int, {.5: 1})，断言结果中 S.Half 不在键集合中
    d = sympify(defaultdict(int, {.5: 1}))
    assert S.Half not in d
    # 断言默认字典中 Float(.5) 在键集合中，且对应值为 S.One
    assert Float(.5) in d
    assert d[.5] is S.One
# 定义一个测试函数，用于验证 GitHub 上的问题 #17811
def test_issue_17811():
    # 创建一个代表函数 'a' 的符号对象
    a = Function('a')
    # 断言语法解析器能正确识别表达式 'a(x)*5'，并且不进行求值
    assert sympify('a(x)*5', evaluate=False) == Mul(a(x), 5, evaluate=False)


# 定义一个测试函数，用于验证 GitHub 上的问题 #8439
def test_issue_8439():
    # 断言无穷大的浮点数能被正确解析为正无穷大 oo
    assert sympify(float('inf')) == oo
    # 断言表达式 x + 无穷大 能被正确解析为 x + oo
    assert x + float('inf') == x + oo
    # 断言将 float('inf') 转换为符号 oo 能够成功
    assert S(float('inf')) == oo


# 定义一个测试函数，用于验证 GitHub 上的问题 #14706
def test_issue_14706():
    # 如果 numpy 模块未安装，则跳过测试
    if not numpy:
        skip("numpy not installed.")

    # 创建不同形状和类型的零数组和单位数组
    z1 = numpy.zeros((1, 1), dtype=numpy.float64)
    z2 = numpy.zeros((2, 2), dtype=numpy.float64)
    z3 = numpy.zeros((), dtype=numpy.float64)

    y1 = numpy.ones((1, 1), dtype=numpy.float64)
    y2 = numpy.ones((2, 2), dtype=numpy.float64)
    y3 = numpy.ones((), dtype=numpy.float64)

    # 断言 numpy 数组加法的结果与预期的全局数组相等
    assert numpy.all(x + z1 == numpy.full((1, 1), x))
    assert numpy.all(x + z2 == numpy.full((2, 2), x))
    assert numpy.all(z1 + x == numpy.full((1, 1), x))
    assert numpy.all(z2 + x == numpy.full((2, 2), x))

    # 对于不同的零和单位数组 z3, 0, 0.0, 0j，执行一系列断言
    for z in [z3,
              numpy.int64(0),
              numpy.float64(0),
              numpy.complex64(0)]:
        assert x + z == x
        assert z + x == x
        # 断言 x + z 的结果是符号对象 Symbol
        assert isinstance(x + z, Symbol)
        # 断言 z + x 的结果是符号对象 Symbol
        assert isinstance(z + x, Symbol)

    # 断言当 numpy 数组 y1 和 y2 加上 x 时，结果与预期的全局数组相等
    assert numpy.all(x + y1 == numpy.full((1, 1), x + 1.0))
    assert numpy.all(x + y2 == numpy.full((2, 2), x + 1.0))
    assert numpy.all(y1 + x == numpy.full((1, 1), x + 1.0))
    assert numpy.all(y2 + x == numpy.full((2, 2), x + 1.0))

    # 对于不同的单位数组 y3, 1, 1.0, 1j，执行一系列断言
    for y_ in [y3,
              numpy.int64(1),
              numpy.float64(1),
              numpy.complex64(1)]:
        assert x + y_ == y_ + x
        # 断言 x + y_ 的结果是加法对象 Add
        assert isinstance(x + y_, Add)
        # 断言 y_ + x 的结果是加法对象 Add
        assert isinstance(y_ + x, Add)

    # 断言 x 加上一个 numpy 数组 x 的结果是 2*x
    assert x + numpy.array(x) == 2 * x
    # 断言 x 加上 numpy 数组 [x] 的结果是 numpy 数组 [2*x]，dtype 为 object
    assert x + numpy.array([x]) == numpy.array([2*x], dtype=object)

    # 断言 sympify 能正确将 numpy 数组转换为不可变的多维数组
    assert sympify(numpy.array([1])) == ImmutableDenseNDimArray([1], 1)
    assert sympify(numpy.array([[[1]]])) == ImmutableDenseNDimArray([1], (1, 1, 1))
    assert sympify(z1) == ImmutableDenseNDimArray([0.0], (1, 1))
    assert sympify(z2) == ImmutableDenseNDimArray([0.0, 0.0, 0.0, 0.0], (2, 2))
    assert sympify(z3) == ImmutableDenseNDimArray([0.0], ())
    assert sympify(z3, strict=True) == 0.0

    # 断言当 strict=True 时，将 numpy 数组转换为 sympify 会引发 SympifyError 异常
    raises(SympifyError, lambda: sympify(numpy.array([1]), strict=True))
    raises(SympifyError, lambda: sympify(z1, strict=True))
    raises(SympifyError, lambda: sympify(z2, strict=True))


# 定义一个测试函数，用于验证 GitHub 上的问题 #21536
def test_issue_21536():
    # 测试在 iterable 输入情况下 evaluate=False 的行为
    u = sympify("x+3*x+2", evaluate=False)
    v = sympify("2*x+4*x+2+4", evaluate=False)

    # 断言 u 是一个加法表达式，并且其参数集合包含 {x, 3*x, 2}
    assert u.is_Add and set(u.args) == {x, 3*x, 2}
    # 断言 v 是一个加法表达式，并且其参数集合包含 {2*x, 4*x, 2, 4}
    assert v.is_Add and set(v.args) == {2*x, 4*x, 2, 4}
    # 使用 sympify 函数将输入的字符串或字符串列表转换为 SymPy 表达式对象，并进行求值
    assert sympify(["x+3*x+2", "2*x+4*x+2+4"], evaluate=False) == [u, v]
    
    # 对单个字符串输入进行求值，返回 SymPy 表达式对象 u 和 v
    u = sympify("x+3*x+2", evaluate=True)
    v = sympify("2*x+4*x+2+4", evaluate=True)
    
    # 断言语句，验证 u 是否为 Add 类型，并检查其参数集合是否为 {4*x, 2}
    assert u.is_Add and set(u.args) == {4*x, 2}
    # 断言语句，验证 v 是否为 Add 类型，并检查其参数集合是否为 {6*x, 6}
    assert v.is_Add and set(v.args) == {6*x, 6}
    # 对字符串列表输入进行求值，并与之前计算得到的 u 和 v 进行比较
    assert sympify(["x+3*x+2", "2*x+4*x+2+4"], evaluate=True) == [u, v]
    
    # 对单个字符串输入进行 sympify，不指定 evaluate 参数，默认为 False，不进行求值
    u = sympify("x+3*x+2")
    v = sympify("2*x+4*x+2+4")
    
    # 断言语句，验证 u 是否为 Add 类型，并检查其参数集合是否为 {4*x, 2}
    assert u.is_Add and set(u.args) == {4*x, 2}
    # 断言语句，验证 v 是否为 Add 类型，并检查其参数集合是否为 {6*x, 6}
    assert v.is_Add and set(v.args) == {6*x, 6}
    # 对字符串列表输入进行 sympify，并与之前计算得到的 u 和 v 进行比较
    assert sympify(["x+3*x+2", "2*x+4*x+2+4"]) == [u, v]
```