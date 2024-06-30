# `D:\src\scipysrc\sympy\sympy\printing\tests\test_repr.py`

```
from __future__ import annotations
# 引入未来的 annotations 特性，支持类型提示中的自引用类型

from typing import Any
# 引入 Any 类型，表示可以是任意类型的对象

from sympy.external.gmpy import GROUND_TYPES
# 从 sympy.external.gmpy 模块导入 GROUND_TYPES

from sympy.testing.pytest import raises, warns_deprecated_sympy
# 从 sympy.testing.pytest 模块导入 raises 和 warns_deprecated_sympy 函数

from sympy.assumptions.ask import Q
# 从 sympy.assumptions.ask 模块导入 Q 对象

from sympy.core.function import (Function, WildFunction)
# 从 sympy.core.function 模块导入 Function 和 WildFunction 类

from sympy.core.numbers import (AlgebraicNumber, Float, Integer, Rational)
# 从 sympy.core.numbers 模块导入 AlgebraicNumber, Float, Integer, Rational 类

from sympy.core.singleton import S
# 从 sympy.core.singleton 模块导入 S 对象

from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
# 从 sympy.core.symbol 模块导入 Dummy, Symbol, Wild, symbols 函数

from sympy.core.sympify import sympify
# 从 sympy.core.sympify 模块导入 sympify 函数

from sympy.functions.elementary.complexes import Abs
# 从 sympy.functions.elementary.complexes 模块导入 Abs 函数

from sympy.functions.elementary.miscellaneous import (root, sqrt)
# 从 sympy.functions.elementary.miscellaneous 模块导入 root 和 sqrt 函数

from sympy.functions.elementary.trigonometric import sin
# 从 sympy.functions.elementary.trigonometric 模块导入 sin 函数

from sympy.functions.special.delta_functions import Heaviside
# 从 sympy.functions.special.delta_functions 模块导入 Heaviside 函数

from sympy.logic.boolalg import (false, true)
# 从 sympy.logic.boolalg 模块导入 false 和 true 对象

from sympy.matrices.dense import (Matrix, ones)
# 从 sympy.matrices.dense 模块导入 Matrix 和 ones 函数

from sympy.matrices.expressions.matexpr import MatrixSymbol
# 从 sympy.matrices.expressions.matexpr 模块导入 MatrixSymbol 类

from sympy.matrices.immutable import ImmutableDenseMatrix
# 从 sympy.matrices.immutable 模块导入 ImmutableDenseMatrix 类

from sympy.combinatorics import Cycle, Permutation
# 从 sympy.combinatorics 模块导入 Cycle 和 Permutation 类

from sympy.core.symbol import Str
# 从 sympy.core.symbol 模块导入 Str 类

from sympy.geometry import Point, Ellipse
# 从 sympy.geometry 模块导入 Point 和 Ellipse 类

from sympy.printing import srepr
# 从 sympy.printing 模块导入 srepr 函数

from sympy.polys import ring, field, ZZ, QQ, lex, grlex, Poly
# 从 sympy.polys 模块导入 ring, field, ZZ, QQ, lex, grlex, Poly 函数

from sympy.polys.polyclasses import DMP
# 从 sympy.polys.polyclasses 模块导入 DMP 类

from sympy.polys.agca.extensions import FiniteExtension
# 从 sympy.polys.agca.extensions 模块导入 FiniteExtension 类

x, y = symbols('x,y')
# 使用 sympy.core.symbol 模块中的 symbols 函数创建符号对象 x 和 y

# eval(srepr(expr)) == expr has to succeed in the right environment. The right
# environment is the scope of "from sympy import *" for most cases.
ENV: dict[str, Any] = {"Str": Str}
# 创建一个字典 ENV，其键为字符串类型，值为任意类型，初始化包含键为 "Str"，值为 Str 类型的条目

exec("from sympy import *", ENV)
# 在 ENV 环境中执行 "from sympy import *" 语句，导入 sympy 中的所有内容

def sT(expr, string, import_stmt=None, **kwargs):
    """
    sT := sreprTest

    Tests that srepr delivers the expected string and that
    the condition eval(srepr(expr))==expr holds.
    """
    if import_stmt is None:
        ENV2 = ENV
    else:
        ENV2 = ENV.copy()
        exec(import_stmt, ENV2)
    # 根据是否提供 import_stmt，确定使用 ENV 还是复制的 ENV 并执行 import_stmt 后的环境 ENV2

    assert srepr(expr, **kwargs) == string
    # 断言 srepr 函数返回的字符串与给定的 string 相等

    assert eval(string, ENV2) == expr
    # 断言将 string 在 ENV2 环境中求值后得到的结果与 expr 相等

def test_printmethod():
    class R(Abs):
        def _sympyrepr(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert srepr(R(x)) == "foo(Symbol('x'))"
    # 断言对 R(x) 使用 srepr 函数得到的字符串与 "foo(Symbol('x'))" 相等

def test_Add():
    sT(x + y, "Add(Symbol('x'), Symbol('y'))")
    # 调用 sT 函数测试 x + y 的 srepr 结果是否为 "Add(Symbol('x'), Symbol('y'))"

    assert srepr(x**2 + 1, order='lex') == "Add(Pow(Symbol('x'), Integer(2)), Integer(1))"
    # 断言 x**2 + 1 在 lex 排序下的 srepr 结果与 "Add(Pow(Symbol('x'), Integer(2)), Integer(1))" 相等

    assert srepr(x**2 + 1, order='old') == "Add(Integer(1), Pow(Symbol('x'), Integer(2)))"
    # 断言 x**2 + 1 在 old 排序下的 srepr 结果与 "Add(Integer(1), Pow(Symbol('x'), Integer(2)))" 相等

    assert srepr(sympify('x + 3 - 2', evaluate=False), order='none') == "Add(Symbol('x'), Integer(3), Mul(Integer(-1), Integer(2)))"
    # 断言对 sympify('x + 3 - 2', evaluate=False) 使用 srepr 函数得到的结果与 "Add(Symbol('x'), Integer(3), Mul(Integer(-1), Integer(2)))" 相等

def test_more_than_255_args_issue_10259():
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    for op in (Add, Mul):
        expr = op(*symbols('x:256'))
        assert eval(srepr(expr)) == expr
        # 断言对 op(*symbols('x:256')) 使用 srepr 函数得到的结果在当前环境中求值后与原始表达式 expr 相等

def test_Function():
    sT(Function("f")(x), "Function('f')(Symbol('x'))")
    # 调用 sT 函数测试 Function("f")(x) 的 srepr 结果是否为 "Function('f')(Symbol('x'))"

    # test unapplied Function
    sT(Function('f'), "Function('f')")
    # 调用 sT 函数测试未应用的 Function('f') 的 srepr 结果是否为 "Function('f')"

def test_Heaviside():
    sT(Heaviside(x), "Heaviside(Symbol('x'))")
    # 调用 sT 函数测试 Heaviside(x) 的 srepr 结果是否为 "Heaviside(Symbol('x'))"

    sT(Heaviside(x, 1), "Heaviside(Symbol('x'), Integer(1))")
    # 调用 sT 函数测试 Heaviside(x, 1) 的 srepr 结果是否为 "Heaviside(Symbol('x'), Integer(1))"
# 定义一个函数用于测试 Geometry 模块的功能
def test_Geometry():
    # 测试 Point 类的构造函数，期望输出指定的字符串表示
    sT(Point(0, 0), "Point2D(Integer(0), Integer(0))")
    # 测试 Ellipse 类的构造函数，期望输出指定的字符串表示
    sT(Ellipse(Point(0, 0), 5, 1),
       "Ellipse(Point2D(Integer(0), Integer(0)), Integer(5), Integer(1))")
    # TODO more tests  # 还需要添加更多的测试


# 定义一个函数用于测试 Singletons 模块的功能
def test_Singletons():
    # 测试 Catalan 常数的值，期望输出指定的字符串
    sT(S.Catalan, 'Catalan')
    # 测试 ComplexInfinity 的值，期望输出指定的字符串
    sT(S.ComplexInfinity, 'zoo')
    # 测试 EulerGamma 常数的值，期望输出指定的字符串
    sT(S.EulerGamma, 'EulerGamma')
    # 测试 Exp1 常数的值，期望输出指定的字符串
    sT(S.Exp1, 'E')
    # 测试 GoldenRatio 常数的值，期望输出指定的字符串
    sT(S.GoldenRatio, 'GoldenRatio')
    # 测试 TribonacciConstant 常数的值，期望输出指定的字符串
    sT(S.TribonacciConstant, 'TribonacciConstant')
    # 测试 Half 常数的值，期望输出指定的字符串
    sT(S.Half, 'Rational(1, 2)')
    # 测试 ImaginaryUnit 常数的值，期望输出指定的字符串
    sT(S.ImaginaryUnit, 'I')
    # 测试 Infinity 常数的值，期望输出指定的字符串
    sT(S.Infinity, 'oo')
    # 测试 NaN（Not a Number）的值，期望输出指定的字符串
    sT(S.NaN, 'nan')
    # 测试 NegativeInfinity 常数的值，期望输出指定的字符串
    sT(S.NegativeInfinity, '-oo')
    # 测试 NegativeOne 常数的值，期望输出指定的字符串
    sT(S.NegativeOne, 'Integer(-1)')
    # 测试 One 常数的值，期望输出指定的字符串
    sT(S.One, 'Integer(1)')
    # 测试 Pi 常数的值，期望输出指定的字符串
    sT(S.Pi, 'pi')
    # 测试 Zero 常数的值，期望输出指定的字符串
    sT(S.Zero, 'Integer(0)')
    # 测试 Complexes 常数的值，期望输出指定的字符串
    sT(S.Complexes, 'Complexes')
    # 测试 EmptySequence 常数的值，期望输出指定的字符串
    sT(S.EmptySequence, 'EmptySequence')
    # 测试 EmptySet 常数的值，期望输出指定的字符串
    sT(S.EmptySet, 'EmptySet')
    # sT(S.IdentityFunction, 'Lambda(_x, _x)')
    # 测试 Naturals 常数的值，期望输出指定的字符串
    sT(S.Naturals, 'Naturals')
    # 测试 Naturals0 常数的值，期望输出指定的字符串
    sT(S.Naturals0, 'Naturals0')
    # 测试 Rationals 常数的值，期望输出指定的字符串
    sT(S.Rationals, 'Rationals')
    # 测试 Reals 常数的值，期望输出指定的字符串
    sT(S.Reals, 'Reals')
    # 测试 UniversalSet 常数的值，期望输出指定的字符串
    sT(S.UniversalSet, 'UniversalSet')


# 定义一个函数用于测试 Integer 类的功能
def test_Integer():
    # 测试 Integer 类的构造函数，期望输出指定的字符串表示
    sT(Integer(4), "Integer(4)")


# 定义一个函数用于测试 list 类型的功能
def test_list():
    # 测试 list 的构造函数，期望输出指定的字符串表示
    sT([x, Integer(4)], "[Symbol('x'), Integer(4)]")


# 定义一个函数用于测试 Matrix 模块的功能
def test_Matrix():
    # 使用循环遍历两种不同类型的矩阵，进行测试
    for cls, name in [(Matrix, "MutableDenseMatrix"), (ImmutableDenseMatrix, "ImmutableDenseMatrix")]:
        # 测试矩阵类的构造函数，期望输出指定的字符串表示
        sT(cls([[x**+1, 1], [y, x + y]]),
           "%s([[Symbol('x'), Integer(1)], [Symbol('y'), Add(Symbol('x'), Symbol('y'))]])" % name)

        # 测试空矩阵类的构造函数，期望输出指定的字符串表示
        sT(cls(), "%s([])" % name)

        # 再次测试矩阵类的构造函数，期望输出指定的字符串表示
        sT(cls([[x**+1, 1], [y, x + y]]), "%s([[Symbol('x'), Integer(1)], [Symbol('y'), Add(Symbol('x'), Symbol('y'))]])" % name)


# 定义一个函数用于测试空矩阵的功能
def test_empty_Matrix():
    # 测试创建指定行和列数的全为 1 的矩阵，期望输出指定的字符串表示
    sT(ones(0, 3), "MutableDenseMatrix(0, 3, [])")
    # 测试创建指定行和列数的全为 1 的矩阵，期望输出指定的字符串表示
    sT(ones(4, 0), "MutableDenseMatrix(4, 0, [])")
    # 测试创建指定行和列数的全为 1 的矩阵，期望输出指定的字符串表示
    sT(ones(0, 0), "MutableDenseMatrix([])")


# 定义一个函数用于测试 Rational 类的功能
def test_Rational():
    # 测试 Rational 类的构造函数，期望输出指定的字符串表示
    sT(Rational(1, 3), "Rational(1, 3)")
    # 测试 Rational 类的构造函数，期望输出指定的字符串表示
    sT(Rational(-1, 3), "Rational(-1, 3)")


# 定义一个函数用于测试 Float 类的功能
def test_Float():
    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('1.23', dps=3), "Float('1.22998', precision=13)")
    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('1.23456789', dps=9), "Float('1.23456788994', precision=33)")
    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('1.234567890123456789', dps=19),
       "Float('1.234567890123456789013', precision=66)")
    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('0.60038617995049726', dps=15),
       "Float('0.60038617995049726', precision=53)")

    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('1.23', precision=13), "Float('1.22998', precision=13)")
    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('1.23456789', precision=33),
       "Float('1.23456788994', precision=33)")
    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('1.234567890123456789', precision=66),
       "Float('1.234567890123456789013', precision=66)")
    # 测试 Float 类的构造函数，期望输出指定的字符串表示
    sT(Float('0.600386179950497
    s2 = "Symbol('x', negative=False, integer=True)"
    # 定义字符串 s2，表示一个带有特定参数的符号对象 'x'
    assert srepr(x) in (s1, s2)
    # 断言调用 srepr(x) 的结果在元组 (s1, s2) 中，用于验证 x 的符号表达式是否在预期范围内
    assert eval(srepr(x), ENV) == x
    # 断言将 srepr(x) 表达式在环境变量 ENV 下进行求值得到的结果等于 x 本身
def test_Symbol_no_special_commutative_treatment():
    # 测试不带特殊交换处理的符号对象
    sT(Symbol('x'), "Symbol('x')")
    # 测试带有 commutative=False 参数的符号对象
    sT(Symbol('x', commutative=False), "Symbol('x', commutative=False)")
    # 测试带有 commutative=0 参数的符号对象，0 被解释为 False
    sT(Symbol('x', commutative=0), "Symbol('x', commutative=False)")
    # 测试带有 commutative=True 参数的符号对象
    sT(Symbol('x', commutative=True), "Symbol('x', commutative=True)")
    # 测试带有 commutative=1 参数的符号对象，1 被解释为 True
    sT(Symbol('x', commutative=1), "Symbol('x', commutative=True)")


def test_Wild():
    # 测试带有 even=True 参数的 Wild 对象
    sT(Wild('x', even=True), "Wild('x', even=True)")


def test_Dummy():
    # 创建一个 Dummy 对象 'd'
    d = Dummy('d')
    # 测试 Dummy 对象的字符串表示，包含 dummy_index
    sT(d, "Dummy('d', dummy_index=%s)" % str(d.dummy_index))


def test_Dummy_assumption():
    # 创建一个带有 nonzero=True 参数的 Dummy 对象 'd'
    d = Dummy('d', nonzero=True)
    # 断言 Dummy 对象和其表达式的字符串表示相等
    assert d == eval(srepr(d))
    # 构造两种可能的 Dummy 对象的字符串表示
    s1 = "Dummy('d', dummy_index=%s, nonzero=True)" % str(d.dummy_index)
    s2 = "Dummy('d', nonzero=True, dummy_index=%s)" % str(d.dummy_index)
    # 断言 Dummy 对象的字符串表示符合其中一种可能的格式
    assert srepr(d) in (s1, s2)


def test_Dummy_from_Symbol():
    # 测试从 Symbol 对象创建 Dummy 对象，不应该返回完整的假设字典
    n = Symbol('n', integer=True)
    d = n.as_dummy()
    # 断言 Dummy 对象的字符串表示
    assert srepr(d) == "Dummy('n', dummy_index=%s)" % str(d.dummy_index)


def test_tuple():
    # 测试创建包含单个符号对象的元组的字符串表示
    sT((x,), "(Symbol('x'),)")
    # 测试创建包含两个符号对象的元组的字符串表示
    sT((x, y), "(Symbol('x'), Symbol('y'))")


def test_WildFunction():
    # 测试创建 WildFunction 对象 'w' 的字符串表示
    sT(WildFunction('w'), "WildFunction('w')")


def test_settins():
    # 断言调用 srepr 函数时引发 TypeError 异常，因为方法参数为 "garbage" 是无效的
    raises(TypeError, lambda: srepr(x, method="garbage"))


def test_Mul():
    # 测试 Mul 对象的字符串表示，包含乘法操作数和幂操作
    sT(3*x**3*y, "Mul(Integer(3), Pow(Symbol('x'), Integer(3)), Symbol('y'))")
    # 断言使用 order='old' 参数时 Mul 对象的字符串表示
    assert srepr(3*x**3*y, order='old') == "Mul(Integer(3), Symbol('y'), Pow(Symbol('x'), Integer(3)))"
    # 断言使用 evaluate=False 参数时 Mul 对象的字符串表示
    assert srepr(sympify('(x+4)*2*x*7', evaluate=False), order='none') == "Mul(Add(Symbol('x'), Integer(4)), Integer(2), Symbol('x'), Integer(7))"


def test_AlgebraicNumber():
    # 创建一个 AlgebraicNumber 对象，表示平方根
    a = AlgebraicNumber(sqrt(2))
    # 测试 AlgebraicNumber 对象的字符串表示
    sT(a, "AlgebraicNumber(Pow(Integer(2), Rational(1, 2)), [Integer(1), Integer(0)])")
    # 创建一个 AlgebraicNumber 对象，表示立方根
    a = AlgebraicNumber(root(-2, 3))
    # 测试 AlgebraicNumber 对象的字符串表示
    sT(a, "AlgebraicNumber(Pow(Integer(-2), Rational(1, 3)), [Integer(1), Integer(0)])")


def test_PolyRing():
    # 断言 PolyRing 对象的字符串表示
    assert srepr(ring("x", ZZ, lex)[0]) == "PolyRing((Symbol('x'),), ZZ, lex)"
    assert srepr(ring("x,y", QQ, grlex)[0]) == "PolyRing((Symbol('x'), Symbol('y')), QQ, grlex)"
    assert srepr(ring("x,y,z", ZZ["t"], lex)[0]) == "PolyRing((Symbol('x'), Symbol('y'), Symbol('z')), ZZ[t], lex)"


def test_FracField():
    # 断言 FracField 对象的字符串表示
    assert srepr(field("x", ZZ, lex)[0]) == "FracField((Symbol('x'),), ZZ, lex)"
    assert srepr(field("x,y", QQ, grlex)[0]) == "FracField((Symbol('x'), Symbol('y')), QQ, grlex)"
    assert srepr(field("x,y,z", ZZ["t"], lex)[0]) == "FracField((Symbol('x'), Symbol('y'), Symbol('z')), ZZ[t], lex)"


def test_PolyElement():
    # 创建一个 PolyRing 对象和一个多项式元素，并断言其字符串表示
    R, x, y = ring("x,y", ZZ)
    assert srepr(3*x**2*y + 1) == "PolyElement(PolyRing((Symbol('x'), Symbol('y')), ZZ, lex), [((2, 1), 3), ((0, 0), 1)])"


def test_FracElement():
    # 创建一个 FracField 对象和一个分式元素，并断言其字符串表示
    F, x, y = field("x,y", ZZ)
    assert srepr((3*x**2*y + 1)/(x - y**2)) == "FracElement(FracField((Symbol('x'), Symbol('y')), ZZ, lex), [((2, 1), 3), ((0, 0), 1)], [((1, 0), 1), ((0, 2), -1)])"


def test_FractionField():
    pass  # 未提供该函数的实现
    # 断言语句，验证 QQ.frac_field(x) 返回的字符串表示是否与期望值相同
    assert srepr(QQ.frac_field(x)) == \
        "FractionField(FracField((Symbol('x'),), QQ, lex))"
    # 断言语句，验证 QQ.frac_field(x, y, order=grlex) 返回的字符串表示是否与期望值相同
    assert srepr(QQ.frac_field(x, y, order=grlex)) == \
        "FractionField(FracField((Symbol('x'), Symbol('y')), QQ, grlex))"
# 定义测试函数 test_PolynomialRingBase
def test_PolynomialRingBase():
    # 断言 ZZ.old_poly_ring(x) 的字符串表示等于指定字符串
    assert srepr(ZZ.old_poly_ring(x)) == \
        "GlobalPolynomialRing(ZZ, Symbol('x'))"
    # 断言 ZZ[x].old_poly_ring(y) 的字符串表示等于指定字符串
    assert srepr(ZZ[x].old_poly_ring(y)) == \
        "GlobalPolynomialRing(ZZ[x], Symbol('y'))"
    # 断言 QQ.frac_field(x).old_poly_ring(y) 的字符串表示等于指定字符串
    assert srepr(QQ.frac_field(x).old_poly_ring(y)) == \
        "GlobalPolynomialRing(FractionField(FracField((Symbol('x'),), QQ, lex)), Symbol('y'))"

# 定义测试函数 test_DMP
def test_DMP():
    # 创建多项式 p1，使用 DMP 类初始化
    p1 = DMP([1, 2], ZZ)
    # 创建多项式 p2，使用 ZZ.old_poly_ring(x) 初始化
    p2 = ZZ.old_poly_ring(x)([1, 2])
    # 根据 GROUND_TYPES 的值进行不同的断言
    if GROUND_TYPES != 'flint':
        # 当 GROUND_TYPES 不为 'flint' 时，断言 p1 的字符串表示为指定字符串
        assert srepr(p1) == "DMP_Python([1, 2], ZZ)"
        # 当 GROUND_TYPES 不为 'flint' 时，断言 p2 的字符串表示为指定字符串
        assert srepr(p2) == "DMP_Python([1, 2], ZZ)"
    else:
        # 当 GROUND_TYPES 为 'flint' 时，断言 p1 的字符串表示为指定字符串
        assert srepr(p1) == "DUP_Flint([1, 2], ZZ)"
        # 当 GROUND_TYPES 为 'flint' 时，断言 p2 的字符串表示为指定字符串
        assert srepr(p2) == "DUP_Flint([1, 2], ZZ)"

# 定义测试函数 test_FiniteExtension
def test_FiniteExtension():
    # 断言 FiniteExtension(Poly(x**2 + 1, x)) 的字符串表示等于指定字符串
    assert srepr(FiniteExtension(Poly(x**2 + 1, x))) == \
        "FiniteExtension(Poly(x**2 + 1, x, domain='ZZ'))"

# 定义测试函数 test_ExtensionElement
def test_ExtensionElement():
    # 创建 FiniteExtension 对象 A
    A = FiniteExtension(Poly(x**2 + 1, x))
    # 根据 GROUND_TYPES 的值选择不同的 ans 字符串表示
    if GROUND_TYPES != 'flint':
        ans = "ExtElem(DMP_Python([1, 0], ZZ), FiniteExtension(Poly(x**2 + 1, x, domain='ZZ')))"
    else:
        ans = "ExtElem(DUP_Flint([1, 0], ZZ), FiniteExtension(Poly(x**2 + 1, x, domain='ZZ')))"
    # 断言 A.generator 的字符串表示等于 ans
    assert srepr(A.generator) == ans

# 定义测试函数 test_BooleanAtom
def test_BooleanAtom():
    # 断言 true 的字符串表示为 "true"
    assert srepr(true) == "true"
    # 断言 false 的字符串表示为 "false"
    assert srepr(false) == "false"

# 定义测试函数 test_Integers
def test_Integers():
    # 调用 sT 函数，断言 S.Integers 的字符串表示等于指定字符串
    sT(S.Integers, "Integers")

# 定义测试函数 test_Naturals
def test_Naturals():
    # 调用 sT 函数，断言 S.Naturals 的字符串表示等于指定字符串
    sT(S.Naturals, "Naturals")

# 定义测试函数 test_Naturals0
def test_Naturals0():
    # 调用 sT 函数，断言 S.Naturals0 的字符串表示等于指定字符串
    sT(S.Naturals0, "Naturals0")

# 定义测试函数 test_Reals
def test_Reals():
    # 调用 sT 函数，断言 S.Reals 的字符串表示等于指定字符串
    sT(S.Reals, "Reals")

# 定义测试函数 test_matrix_expressions
def test_matrix_expressions():
    # 创建符号 n，假设为整数
    n = symbols('n', integer=True)
    # 创建 n x n 的矩阵符号 A 和 B
    A = MatrixSymbol("A", n, n)
    B = MatrixSymbol("B", n, n)
    # 调用 sT 函数，断言矩阵符号 A 的字符串表示等于指定字符串
    sT(A, "MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True))")
    # 调用 sT 函数，断言矩阵乘法 A*B 的字符串表示等于指定字符串
    sT(A*B, "MatMul(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")
    # 调用 sT 函数，断言矩阵加法 A + B 的字符串表示等于指定字符串
    sT(A + B, "MatAdd(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")

# 定义测试函数 test_Cycle
def test_Cycle():
    # TODO: 由于 Cycle 不是不可变的，sT 函数会失败，因此暂时跳过测试
    # import_stmt = "from sympy.combinatorics import Cycle"
    # sT(Cycle(1, 2), "Cycle(1, 2)", import_stmt)
    # 断言 Cycle(1, 2) 的字符串表示为 "Cycle(1, 2)"
    assert srepr(Cycle(1, 2)) == "Cycle(1, 2)"

# 定义测试函数 test_Permutation
def test_Permutation():
    # 导入语句
    import_stmt = "from sympy.combinatorics import Permutation"
    # 调用 sT 函数，断言 Permutation(1, 2)(3, 4) 的字符串表示等于指定字符串，不考虑循环表示
    sT(Permutation(1, 2)(3, 4), "Permutation([0, 2, 1, 4, 3])", import_stmt, perm_cyclic=False)
    # 调用 sT 函数，断言 Permutation(1, 2)(3, 4) 的字符串表示等于指定字符串，考虑循环表示
    sT(Permutation(1, 2)(3, 4), "Permutation(1, 2)(3, 4)", import_stmt, perm_cyclic=True)

    # 弃用警告上下文
    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        # 设置 Permutation.print_cyclic 为 False
        Permutation.print_cyclic = False
        # 调用 sT 函数，断言 Permutation(1, 2)(3, 4) 的字符串表示等于指定字符串，不考虑循环表示
        sT(Permutation(1, 2)(3, 4), "Permutation([0, 2, 1, 4, 3])", import_stmt)
        # 恢复 Permutation.print_cyclic 的原始值
        Permutation.print_cyclic = old_print_cyclic

# 定义测试函数 test_dict
def test_dict():
    # 从 sympy.abc 中导入符号 x, y, z
    from sympy.abc import x, y, z
    # 创建一个空字典 d
    d = {}
    # 断言，验证表达式 srepr(d) == "{}" 是否为真
    assert srepr(d) == "{}"
    
    # 创建一个字典 d，包含一个键值对 {x: y}
    d = {x: y}
    
    # 断言，验证表达式 srepr(d) == "{Symbol('x'): Symbol('y')}" 是否为真
    assert srepr(d) == "{Symbol('x'): Symbol('y')}"
    
    # 创建一个字典 d，包含两个键值对 {x: y, y: z}
    d = {x: y, y: z}
    
    # 断言，验证表达式 srepr(d) 是否在给定的两个可能性中的任意一个中
    assert srepr(d) in (
        "{Symbol('x'): Symbol('y'), Symbol('y'): Symbol('z')}",
        "{Symbol('y'): Symbol('z'), Symbol('x'): Symbol('y')}",
    )
    
    # 创建一个字典 d，包含一个键值对 {x: {y: z}}
    d = {x: {y: z}}
    
    # 断言，验证表达式 srepr(d) == "{Symbol('x'): {Symbol('y'): Symbol('z')}}" 是否为真
    assert srepr(d) == "{Symbol('x'): {Symbol('y'): Symbol('z')}}"
# 定义一个函数用于测试集合操作
def test_set():
    # 从 sympy.abc 模块中导入符号 x 和 y
    from sympy.abc import x, y
    # 创建一个空集合 s
    s = set()
    # 断言空集合 s 的字符串表示应该是 "set()"
    assert srepr(s) == "set()"
    
    # 将集合 s 更新为包含符号 x 和 y 的集合
    s = {x, y}
    # 断言集合 s 的字符串表示应该是其中一个可能的形式
    assert srepr(s) in ("{Symbol('x'), Symbol('y')}", "{Symbol('y'), Symbol('x')}")

# 定义一个函数用于测试 Predicate
def test_Predicate():
    # 调用 sT 函数，检查 Q.even 的字符串表示是否正确
    sT(Q.even, "Q.even")

# 定义一个函数用于测试 AppliedPredicate
def test_AppliedPredicate():
    # 调用 sT 函数，检查 Q.even(Symbol('z')) 的字符串表示是否正确
    sT(Q.even(Symbol('z')), "AppliedPredicate(Q.even, Symbol('z'))")
```