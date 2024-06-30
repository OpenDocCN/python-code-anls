# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_printing.py`

```
# -*- encoding: utf-8 -*-
"""
TODO:
* Address Issue 2251, printing of spin states
"""
# 引入未来的注释语法支持
from __future__ import annotations
# 引入类型提示中的 Any 类型
from typing import Any

# 导入量子力学相关模块和类
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import CGate, CNotGate, IdentityGate, UGate, XGate
from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace, HilbertSpace, L2
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import Operator, OuterProduct, DifferentialOperator
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.qubit import Qubit, IntQubit
from sympy.physics.quantum.spin import Jz, J2, JzBra, JzBraCoupled, JzKet, JzKetCoupled, Rotation, WignerD
from sympy.physics.quantum.state import Bra, Ket, TimeDepBra, TimeDepKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.sho1d import RaisingOp

# 导入核心功能模块和类
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL

# 用于 srepr 字符串的导入
from sympy.physics.quantum.spin import JzOp

# 导入打印相关函数
from sympy.printing import srepr
from sympy.printing.pretty import pretty as xpretty
from sympy.printing.latex import latex

# 为了避免名称冲突，重命名 Matrix 类
MutableDenseMatrix = Matrix

# 创建一个空的环境字典
ENV: dict[str, Any] = {}
# 在环境中执行语句，导入 sympy 的所有内容
exec('from sympy import *', ENV)
# 在环境中执行语句，导入量子力学相关内容
exec('from sympy.physics.quantum import *', ENV)
# 在环境中执行语句，导入 CG 相关内容
exec('from sympy.physics.quantum.cg import *', ENV)
# 在环境中执行语句，导入 spin 相关内容
exec('from sympy.physics.quantum.spin import *', ENV)
# 在环境中执行语句，导入 hilbert 相关内容
exec('from sympy.physics.quantum.hilbert import *', ENV)
# 在环境中执行语句，导入 qubit 相关内容
exec('from sympy.physics.quantum.qubit import *', ENV)
# 在环境中执行语句，导入 qexpr 相关内容
exec('from sympy.physics.quantum.qexpr import *', ENV)
# 在环境中执行语句，导入 gate 相关内容
exec('from sympy.physics.quantum.gate import *', ENV)
# 在环境中执行语句，导入常量相关内容
exec('from sympy.physics.quantum.constants import *', ENV)


def sT(expr, string):
    """
    sT := sreprTest
    from sympy/printing/tests/test_repr.py
    """
    # 断言 srepr 函数的输出与预期的字符串相等
    assert srepr(expr) == string
    # 断言字符串表达式在当前环境中的求值结果与原始表达式相等
    assert eval(string, ENV) == expr


def pretty(expr):
    """ASCII pretty-printing"""
    # 使用 ASCII 字符输出美化的表达式
    return xpretty(expr, use_unicode=False, wrap_line=False)


def upretty(expr):
    """Unicode pretty-printing"""
    # 使用 Unicode 字符输出美化的表达式
    return xpretty(expr, use_unicode=True, wrap_line=False)


def test_anticommutator():
    # 创建两个操作符 A 和 B
    A = Operator('A')
    B = Operator('B')
    # 创建 A 和 B 的反对易子
    ac = AntiCommutator(A, B)
    # 创建 A^2 和 B 的反对易子
    ac_tall = AntiCommutator(A**2, B)
    # 断言反对易子的字符串表示与预期相等
    assert str(ac) == '{A,B}'
    # 断言 ASCII 美化输出与预期相等
    assert pretty(ac) == '{A,B}'
    # 断言 Unicode 美化输出与预期相等
    assert upretty(ac) == '{A,B}'
    # 断言 LaTeX 输出与预期相等
    assert latex(ac) == r'\left\{A,B\right\}'
    # 使用 sT 函数进行 srepr 测试
    sT(ac, "AntiCommutator(Operator(Symbol('A')),Operator(Symbol('B')))")
    # 断言 A^2 和 B 的反对易子的字符串表示与预期相等
    assert str(ac_tall) == '{A**2,B}'
    # 创建一个多行字符串，包含 ASCII 字符集中的可打印字符
"""\
/ 2  \\\n\
<A ,B>\n\
\\    /\
"""
# ASCII艺术风格的字符串，描述一个特定形式的图案

ucode_str = \
"""\
⎧ 2  ⎫\n\
⎨A ,B⎬\n\
⎩    ⎭\
"""
# Unicode艺术风格的字符串，描述与ASCII艺术风格相同的图案，但使用Unicode字符

assert pretty(ac_tall) == ascii_str
# 断言检查函数pretty(ac_tall)生成的字符串是否与ASCII艺术风格的字符串ascii_str相等

assert upretty(ac_tall) == ucode_str
# 断言检查函数upretty(ac_tall)生成的字符串是否与Unicode艺术风格的字符串ucode_str相等

assert latex(ac_tall) == r'\left\{A^{2},B\right\}'
# 断言检查函数latex(ac_tall)生成的LaTeX表达式是否符合预期的格式

sT(ac_tall, "AntiCommutator(Pow(Operator(Symbol('A')), Integer(2)),Operator(Symbol('B')))")
# 调用函数sT，传递参数ac_tall和其对应的字符串描述，用于测试或其他目的
    # 断言，验证 pretty 函数处理后的结果是否等于 ascii_str
    assert pretty(c_tall) == ascii_str
    # 断言，验证 upretty 函数处理后的结果是否等于 ucode_str
    assert upretty(c_tall) == ucode_str
    # 断言，验证 latex 函数处理后的结果是否等于预期的 LaTeX 字符串
    assert latex(c_tall) == r'\left[A^{2},B\right]'
    # 调用 sT 函数，输出 c_tall 对应的字符串表示和给定的描述信息
    sT(c_tall, "Commutator(Pow(Operator(Symbol('A')), Integer(2)),Operator(Symbol('B')))")
# 测试常数和函数是否按预期工作
def test_constants():
    # 检查字符串表示是否正确
    assert str(hbar) == 'hbar'
    # 检查漂亮的字符串表示是否正确
    assert pretty(hbar) == 'hbar'
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(hbar) == 'ℏ'
    # 检查LaTeX表示是否正确
    assert latex(hbar) == r'\hbar'
    # 测试字符串表示转换
    sT(hbar, "HBar()")


# 测试Dagger操作的功能
def test_dagger():
    # 创建符号变量 x
    x = symbols('x')
    # 创建Dagger表达式
    expr = Dagger(x)
    # 检查字符串表示是否正确
    assert str(expr) == 'Dagger(x)'
    # ASCII漂亮的字符串表示
    ascii_str = \
"""\
 +\n\
x \
"""
    # Unicode漂亮的字符串表示
    ucode_str = \
"""\
 †\n\
x \
"""
    # 检查漂亮的字符串表示是否正确
    assert pretty(expr) == ascii_str
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(expr) == ucode_str
    # 检查LaTeX表示是否正确
    assert latex(expr) == r'x^{\dagger}'
    # 测试字符串表示转换
    sT(expr, "Dagger(Symbol('x'))")


# 以下函数标记为XFAIL，测试门操作失败的情况
@XFAIL
def test_gate_failing():
    # 创建符号变量 a, b, c, d
    a, b, c, d = symbols('a,b,c,d')
    # 创建矩阵 uMat
    uMat = Matrix([[a, b], [c, d]])
    # 创建 UGate 门
    g = UGate((0,), uMat)
    # 检查字符串表示是否正确
    assert str(g) == 'U(0)'


# 测试门操作的功能
def test_gate():
    # 创建符号变量 a, b, c, d
    a, b, c, d = symbols('a,b,c,d')
    # 创建矩阵 uMat
    uMat = Matrix([[a, b], [c, d]])
    # 创建 Qubit 对象
    q = Qubit(1, 0, 1, 0, 1)
    # 创建 IdentityGate 对象 g1
    g1 = IdentityGate(2)
    # 创建 CGate 对象 g2
    g2 = CGate((3, 0), XGate(1))
    # 创建 CNotGate 对象 g3
    g3 = CNotGate(1, 0)
    # 创建 UGate 对象 g4
    g4 = UGate((0,), uMat)
    # 检查字符串表示是否正确
    assert str(g1) == '1(2)'
    # 检查漂亮的字符串表示是否正确
    assert pretty(g1) == '1 \n 2'
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(g1) == '1 \n 2'
    # 检查LaTeX表示是否正确
    assert latex(g1) == r'1_{2}'
    # 测试字符串表示转换
    sT(g1, "IdentityGate(Integer(2))")
    # 检查字符串表示是否正确
    assert str(g1*q) == '1(2)*|10101>'
    # ASCII漂亮的字符串表示
    ascii_str = \
"""\
1 *|10101>\n\
 2        \
"""
    # Unicode漂亮的字符串表示
    ucode_str = \
"""\
1 ⋅❘10101⟩\n\
 2        \
"""
    # 检查漂亮的字符串表示是否正确
    assert pretty(g1*q) == ascii_str
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(g1*q) == ucode_str
    # 检查LaTeX表示是否正确
    assert latex(g1*q) == r'1_{2} {\left|10101\right\rangle }'
    # 测试字符串表示转换
    sT(g1*q, "Mul(IdentityGate(Integer(2)), Qubit(Integer(1),Integer(0),Integer(1),Integer(0),Integer(1)))")
    # 检查字符串表示是否正确
    assert str(g2) == 'C((3,0),X(1))'
    # ASCII漂亮的字符串表示
    ascii_str = \
"""\
C   /X \\\n\
 3,0\\ 1/\
"""
    # Unicode漂亮的字符串表示
    ucode_str = \
"""\
C   ⎛X ⎞\n\
 3,0⎝ 1⎠\
"""
    # 检查漂亮的字符串表示是否正确
    assert pretty(g2) == ascii_str
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(g2) == ucode_str
    # 检查LaTeX表示是否正确
    assert latex(g2) == r'C_{3,0}{\left(X_{1}\right)}'
    # 测试字符串表示转换
    sT(g2, "CGate(Tuple(Integer(3), Integer(0)),XGate(Integer(1)))")
    # 检查字符串表示是否正确
    assert str(g3) == 'CNOT(1,0)'
    # ASCII漂亮的字符串表示
    ascii_str = \
"""\
CNOT   \n\
    1,0\
"""
    # Unicode漂亮的字符串表示
    ucode_str = \
"""\
CNOT   \n\
    1,0\
"""
    # 检查漂亮的字符串表示是否正确
    assert pretty(g3) == ascii_str
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(g3) == ucode_str
    # 检查LaTeX表示是否正确
    assert latex(g3) == r'\text{CNOT}_{1,0}'
    # 测试字符串表示转换
    sT(g3, "CNotGate(Integer(1),Integer(0))")
    # ASCII漂亮的字符串表示
    ascii_str = \
"""\
U \n\
 0\
"""
    # Unicode漂亮的字符串表示
    ucode_str = \
"""\
U \n\
 0\
"""
    # 检查字符串表示是否正确
    assert str(g4) == \
"""\
U((0,),Matrix([\n\
[a, b],\n\
[c, d]]))\
"""
    # 检查漂亮的字符串表示是否正确
    assert pretty(g4) == ascii_str
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(g4) == ucode_str
    # 检查LaTeX表示是否正确
    assert latex(g4) == r'U_{0}'
    # 测试字符串表示转换
    sT(g4, "UGate(Tuple(Integer(0)),ImmutableDenseMatrix([[Symbol('a'), Symbol('b')], [Symbol('c'), Symbol('d')]]))")


# 测试Hilbert空间对象的功能
def test_hilbert():
    # 创建HilbertSpace对象 h1
    h1 = HilbertSpace()
    # 创建ComplexSpace对象 h2
    h2 = ComplexSpace(2)
    # 创建FockSpace对象 h3
    h3 = FockSpace()
    # 创建L2对象 h4
    h4 = L2(Interval(0, oo))
    # 检查字符串表示是否正确
    assert str(h1) == 'H'
    # 检查漂亮的字符串表示是否正确
    assert pretty(h1) == 'H'
    # 检查Unicode漂亮的字符串表示是否正确
    assert upretty(h1) == 'H'
    # 检查LaTeX表示是否正确
    assert latex(h1) == r'\mathcal{H}'
    # 测试字符串表示转换
    sT(h1, "HilbertSpace()")
    # 检查字符串表示是否正确
    assert str(h2) == 'C(2)'
    # ASCII漂亮的字符串表示
    ascii_str = \
"""\
 2\n\
C \
"""
    # Unicode漂亮的字符串表示
    ucode_str = \
"""\
 2\n\
C \
"""
    # 检查漂亮的字符串表示是否正确
    assert pretty(h2) == ascii_str
    # 检查Unicode漂亮的字符串表示是否正确
    # 断言检查对象 h3 的字符串表示是否等于 'F'
    assert str(h3) == 'F'
    # 断言检查对象 h3 的美化输出是否等于 'F'
    assert pretty(h3) == 'F'
    # 断言检查对象 h3 的 Unicode 美化输出是否等于 'F'
    assert upretty(h3) == 'F'
    # 断言检查对象 h3 的 LaTeX 表示是否等于 '\mathcal{F}'
    assert latex(h3) == r'\mathcal{F}'
    # 将对象 h3 的类型信息输出为字符串 "FockSpace()"
    sT(h3, "FockSpace()")
    # 断言检查对象 h4 的字符串表示是否等于 'L2(Interval(0, oo))'
    assert str(h4) == 'L2(Interval(0, oo))'
    # 给变量 ascii_str 赋值，继续下一行输入
    ascii_str = \
"""
Define symbolic expressions and perform tests on their representations.
"""

# 定义 Unicode 字符串表示的符号和其 ASCII 等效形式
ascii_str = \
"""\
 2\n\
L \
"""
ucode_str = \
"""\
 2\n\
L \
"""

# 断言: 使用预定义的 pretty 函数生成的字符串应与 ASCII 形式的字符串相等
assert pretty(h4) == ascii_str

# 断言: 使用预定义的 upretty 函数生成的字符串应与 Unicode 形式的字符串相等
assert upretty(h4) == ucode_str

# 断言: 使用预定义的 latex 函数生成的字符串应与 LaTeX 格式的字符串相等
assert latex(h4) == r'{\mathcal{L}^2}\left( \left[0, \infty\right) \right)'

# 调用 sT 函数，将 h4 的字符串表示和特定字符串进行比较
sT(h4, "L2(Interval(Integer(0), oo, false, true))")

# 断言: 将 h1 + h2 转换为字符串后应与指定字符串 'H+C(2)' 相等
assert str(h1 + h2) == 'H+C(2)'

# 定义 ASCII 和 Unicode 格式的字符串表示 h1 + h2
ascii_str = \
"""\
     2\n\
H + C \
"""
ucode_str = \
"""\
     2\n\
H ⊕ C \
"""

# 断言: 使用预定义的 pretty 函数生成的字符串应与 ASCII 格式的字符串相等
assert pretty(h1 + h2) == ascii_str

# 断言: 使用预定义的 upretty 函数生成的字符串应与 Unicode 格式的字符串相等
assert upretty(h1 + h2) == ucode_str

# 断言: 使用预定义的 latex 函数生成的字符串应存在（非空字符串）
assert latex(h1 + h2)

# 调用 sT 函数，将 h1 + h2 的字符串表示和特定字符串进行比较
sT(h1 + h2, "DirectSumHilbertSpace(HilbertSpace(),ComplexSpace(Integer(2)))")

# 断言: 将 h1 * h2 转换为字符串后应与指定字符串 'H*C(2)' 相等
assert str(h1*h2) == "H*C(2)"

# 定义 ASCII 和 Unicode 格式的字符串表示 h1 * h2
ascii_str = \
"""\
     2\n\
H x C \
"""
ucode_str = \
"""\
     2\n\
H ⨂ C \
"""

# 断言: 使用预定义的 pretty 函数生成的字符串应与 ASCII 格式的字符串相等
assert pretty(h1*h2) == ascii_str

# 断言: 使用预定义的 upretty 函数生成的字符串应与 Unicode 格式的字符串相等
assert upretty(h1*h2) == ucode_str

# 断言: 使用预定义的 latex 函数生成的字符串应存在（非空字符串）
assert latex(h1*h2)

# 调用 sT 函数，将 h1*h2 的字符串表示和特定字符串进行比较
sT(h1*h2, "TensorProductHilbertSpace(HilbertSpace(),ComplexSpace(Integer(2)))")

# 断言: 将 h1**2 转换为字符串后应与指定字符串 'H**2' 相等
assert str(h1**2) == 'H**2'

# 定义 ASCII 和 Unicode 格式的字符串表示 h1**2
ascii_str = \
"""\
 x2\n\
H  \
"""
ucode_str = \
"""\
 ⨂2\n\
H  \
"""

# 断言: 使用预定义的 pretty 函数生成的字符串应与 ASCII 格式的字符串相等
assert pretty(h1**2) == ascii_str

# 断言: 使用预定义的 upretty 函数生成的字符串应与 Unicode 格式的字符串相等
assert upretty(h1**2) == ucode_str

# 断言: 使用预定义的 latex 函数生成的字符串应与 LaTeX 格式的字符串相等
assert latex(h1**2) == r'{\mathcal{H}}^{\otimes 2}'

# 调用 sT 函数，将 h1**2 的字符串表示和特定字符串进行比较
sT(h1**2, "TensorPowerHilbertSpace(HilbertSpace(),Integer(2))")


def test_innerproduct():
    # 定义符号变量 x
    x = symbols('x')

    # 创建内积对象 ip1 至 ip4 和 ip_tall1 至 ip_tall3
    ip1 = InnerProduct(Bra(), Ket())
    ip2 = InnerProduct(TimeDepBra(), TimeDepKet())
    ip3 = InnerProduct(JzBra(1, 1), JzKet(1, 1))
    ip4 = InnerProduct(JzBraCoupled(1, 1, (1, 1)), JzKetCoupled(1, 1, (1, 1)))
    ip_tall1 = InnerProduct(Bra(x/2), Ket(x/2))
    ip_tall2 = InnerProduct(Bra(x), Ket(x/2))
    ip_tall3 = InnerProduct(Bra(x/2), Ket(x))

    # 断言: 将 ip1 转换为字符串后应与指定字符串 '<psi|psi>' 相等
    assert str(ip1) == '<psi|psi>'

    # 断言: 使用预定义的 pretty 函数生成的字符串应与 ASCII 格式的字符串相等
    assert pretty(ip1) == '<psi|psi>'

    # 断言: 使用预定义的 upretty 函数生成的字符串应与 Unicode 格式的字符串相等
    assert upretty(ip1) == '⟨ψ❘ψ⟩'

    # 断言: 使用预定义的 latex 函数生成的字符串应与 LaTeX 格式的字符串相等
    assert latex(ip1) == r'\left\langle \psi \right. {\left|\psi\right\rangle }'

    # 调用 sT 函数，将 ip1 的字符串表示和特定字符串进行比较
    sT(ip1, "InnerProduct(Bra(Symbol('psi')),Ket(Symbol('psi')))")

    # 继续对 ip2 至 ip4 进行类似的断言和测试，每个对象的字符串表示都有对应的测试和比较
    # 调用函数 sT，传入两个参数:
    # - 第一个参数是 ip4
    # - 第二个参数是一个字符串，表示内积操作的详细描述
    sT(ip4, "InnerProduct(JzBraCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))),JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))))")

    # 使用 assert 断言检查变量 ip_tall1 的字符串表示是否等于 '<x/2|x/2>'
    assert str(ip_tall1) == '<x/2|x/2>'

    # 继续赋值操作，将 '\' 后面的内容赋给 ascii_str 变量
    ascii_str = \
"""
创建一个包含 ASCII 艺术风格的字符串，代表一个图形。
/ | \ 
/ x|x \
\ -|- /
 \2|2/ 
"""
ucode_str = \
"""
创建一个包含 Unicode 艺术风格的字符串，代表同一个图形。
╱ │ ╲ 
╱ x│x ╲
╲ ─│─ ╱
 ╲2│2╱ 
"""
assert pretty(ip_tall1) == ascii_str
assert upretty(ip_tall1) == ucode_str
assert latex(ip_tall1) == \
    r'\left\langle \frac{x}{2} \right. {\left|\frac{x}{2}\right\rangle }'
sT(ip_tall1, "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Mul(Rational(1, 2), Symbol('x'))))")
assert str(ip_tall2) == '<x|x/2>'
ascii_str = \
"""
创建一个 ASCII 艺术风格的字符串，代表一个不同的图形。
/ | \ 
/  |x \
\ x|- /
 \ |2/ 
"""
ucode_str = \
"""
创建一个 Unicode 艺术风格的字符串，代表同一个图形。
╱ │ ╲ 
╱  │x ╲
╲ x│─ ╱
 ╲ │2╱ 
"""
assert pretty(ip_tall2) == ascii_str
assert upretty(ip_tall2) == ucode_str
assert latex(ip_tall2) == \
    r'\left\langle x \right. {\left|\frac{x}{2}\right\rangle }'
sT(ip_tall2,
   "InnerProduct(Bra(Symbol('x')),Ket(Mul(Rational(1, 2), Symbol('x'))))")
assert str(ip_tall3) == '<x/2|x>'
ascii_str = \
"""
创建一个 ASCII 艺术风格的字符串，代表另一个不同的图形。
/ | \ 
/ x|  \
\ -|x /
 \2| / 
"""
ucode_str = \
"""
创建一个 Unicode 艺术风格的字符串，代表同一个图形。
╱ │ ╲ 
╱ x│  ╲
╲ ─│x ╱
 ╲2│ ╱ 
"""
assert pretty(ip_tall3) == ascii_str
assert upretty(ip_tall3) == ucode_str
assert latex(ip_tall3) == \
    r'\left\langle \frac{x}{2} \right. {\left|x\right\rangle }'
sT(ip_tall3,
   "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Symbol('x')))")

def test_operator():
    a = Operator('A')
    b = Operator('B', Symbol('t'), S.Half)
    inv = a.inv()
    f = Function('f')
    x = symbols('x')
    d = DifferentialOperator(Derivative(f(x), x), f(x))
    op = OuterProduct(Ket(), Bra())
    assert str(a) == 'A'
    assert pretty(a) == 'A'
    assert upretty(a) == 'A'
    assert latex(a) == 'A'
    sT(a, "Operator(Symbol('A'))")
    assert str(inv) == 'A**(-1)'
    ascii_str = \
"""
创建一个 ASCII 艺术风格的字符串，表示一个逆运算。
 -1
A  
"""
ucode_str = \
"""
创建一个 Unicode 艺术风格的字符串，表示同一个逆运算。
 -1
A  
"""
assert pretty(inv) == ascii_str
assert upretty(inv) == ucode_str
assert latex(inv) == r'A^{-1}'
sT(inv, "Pow(Operator(Symbol('A')), Integer(-1))")
assert str(d) == 'DifferentialOperator(Derivative(f(x), x),f(x))'
ascii_str = \
"""
创建一个 ASCII 艺术风格的字符串，表示一个微分操作符。
                    /d            \\
DifferentialOperator|--(f(x)),f(x)|
                    \\dx           / 
"""
ucode_str = \
"""
创建一个 Unicode 艺术风格的字符串，表示同一个微分操作符。
                    ⎛d            ⎞
DifferentialOperator⎜──(f(x)),f(x)⎟
                    ⎝dx           ⎠ 
"""
assert pretty(d) == ascii_str
assert upretty(d) == ucode_str
assert latex(d) == \
    r'DifferentialOperator\left(\frac{d}{d x} f{\left(x \right)},f{\left(x \right)}\right)'
sT(d, "DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))")
assert str(b) == 'Operator(B,t,1/2)'
assert pretty(b) == 'Operator(B,t,1/2)'
assert upretty(b) == 'Operator(B,t,1/2)'
assert latex(b) == r'Operator\left(B,t,\frac{1}{2}\right)'
sT(b, "Operator(Symbol('B'),Symbol('t'),Rational(1, 2))")
assert str(op) == '|psi><psi|'
    # 断言：验证操作 pretty(op) 返回的结果是否等于 '|psi><psi|'
    assert pretty(op) == '|psi><psi|'
    
    # 断言：验证操作 upretty(op) 返回的结果是否等于 '❘ψ⟩⟨ψ❘'
    assert upretty(op) == '❘ψ⟩⟨ψ❘'
    
    # 断言：验证操作 latex(op) 返回的结果是否等于 r'{\left|\psi\right\rangle }{\left\langle \psi\right|}'
    assert latex(op) == r'{\left|\psi\right\rangle }{\left\langle \psi\right|}'
    
    # 调用函数 sT(op, "OuterProduct(Ket(Symbol('psi')),Bra(Symbol('psi')))")
    sT(op, "OuterProduct(Ket(Symbol('psi')),Bra(Symbol('psi')))")
# 定义名为 test_qexpr 的测试函数
def test_qexpr():
    # 创建一个名为 q 的 QExpr 对象，表示符号 'q'
    q = QExpr('q')
    # 断言 QExpr 对象的字符串表示为 'q'
    assert str(q) == 'q'
    # 断言通过 pretty 函数美化后的字符串为 'q'
    assert pretty(q) == 'q'
    # 断言通过 upretty 函数 Unicode 美化后的字符串为 'q'
    assert upretty(q) == 'q'
    # 断言通过 latex 函数转换为 LaTeX 格式后的字符串为 r'q'
    assert latex(q) == r'q'
    # 调用 sT 函数，验证 q 对象与字符串 "QExpr(Symbol('q'))" 相符

    # 创建一个名为 q1 的 Qubit 对象，表示量子比特 '0101'
    q1 = Qubit('0101')
    # 创建一个名为 q2 的 IntQubit 对象，表示整数量子比特 8
    q2 = IntQubit(8)
    # 断言 Qubit 对象 q1 的字符串表示为 '|0101>'
    assert str(q1) == '|0101>'
    # 断言通过 pretty 函数美化后的字符串为 '|0101>'
    assert pretty(q1) == '|0101>'
    # 断言通过 upretty 函数 Unicode 美化后的字符串为 '❘0101⟩'
    assert upretty(q1) == '❘0101⟩'
    # 断言通过 latex 函数转换为 LaTeX 格式后的字符串为 r'{\left|0101\right\rangle }'
    assert latex(q1) == r'{\left|0101\right\rangle }'
    # 调用 sT 函数，验证 q1 对象与字符串 "Qubit(Integer(0),Integer(1),Integer(0),Integer(1))" 相符

    # 断言 IntQubit 对象 q2 的字符串表示为 '|8>'
    assert str(q2) == '|8>'
    # 断言通过 pretty 函数美化后的字符串为 '|8>'
    assert pretty(q2) == '|8>'
    # 断言通过 upretty 函数 Unicode 美化后的字符串为 '❘8⟩'
    assert upretty(q2) == '❘8⟩'
    # 断言通过 latex 函数转换为 LaTeX 格式后的字符串为 r'{\left|8\right\rangle }'
    assert latex(q2) == r'{\left|8\right\rangle }'
    # 调用 sT 函数，验证 q2 对象与字符串 "IntQubit(8)" 相符


# 定义名为 test_spin 的测试函数
def test_spin():
    # 创建一个名为 lz 的 JzOp 对象，表示角动量 z 分量 'L'
    lz = JzOp('L')
    # 断言 JzOp 对象 lz 的字符串表示为 'Lz'
    assert str(lz) == 'Lz'
    # 定义 ASCII 格式的字符串 ascii_str
    ascii_str = \
"""\
L \n\
 z\
"""
    # 断言通过 pretty 函数美化后的字符串与 ascii_str 相符
    assert pretty(lz) == ascii_str
    # 定义 Unicode 格式的字符串 ucode_str
    ucode_str = \
"""\
L \n\
 z\
"""
    # 断言通过 upretty 函数 Unicode 美化后的字符串与 ucode_str 相符
    assert upretty(lz) == ucode_str
    # 断言通过 latex 函数转换为 LaTeX 格式后的字符串为 'L_z'
    assert latex(lz) == 'L_z'
    # 调用 sT 函数，验证 lz 对象与字符串 "JzOp(Symbol('L'))" 相符

    # 以下类似地测试其他对象（略去不重复的注释）
    assert str(J2) == 'J2'
    ascii_str = \
"""\
 2\n\
J \
"""
    assert pretty(J2) == ascii_str
    assert upretty(J2) == ucode_str
    assert latex(J2) == r'J^2'
    sT(J2, "J2Op(Symbol('J'))")

    assert str(Jz) == 'Jz'
    ascii_str = \
"""\
J \n\
 z\
"""
    assert pretty(Jz) == ascii_str
    assert upretty(Jz) == ucode_str
    assert latex(Jz) == 'J_z'
    sT(Jz, "JzOp(Symbol('J'))")

    assert str(ket) == '|1,0>'
    assert pretty(ket) == '|1,0>'
    assert upretty(ket) == '❘1,0⟩'
    assert latex(ket) == r'{\left|1,0\right\rangle }'
    sT(ket, "JzKet(Integer(1),Integer(0))")

    assert str(bra) == '<1,0|'
    assert pretty(bra) == '<1,0|>'
    assert upretty(bra) == '⟨1,0❘'
    assert latex(bra) == r'{\left\langle 1,0\right|}'
    sT(bra, "JzBra(Integer(1),Integer(0))")

    assert str(cket) == '|1,0,j1=1,j2=2>'
    assert pretty(cket) == '|1,0,j1=1,j2=2>'
    assert upretty(cket) == '❘1,0,j₁=1,j₂=2⟩'
    assert latex(cket) == r'{\left|1,0,j_{1}=1,j_{2}=2\right\rangle }'
    sT(cket, "JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))")

    assert str(cbra) == '<1,0,j1=1,j2=2|'
    assert pretty(cbra) == '<1,0,j1=1,j2=2|>'
    assert upretty(cbra) == '⟨1,0,j₁=1,j₂=2❘'
    assert latex(cbra) == r'{\left\langle 1,0,j_{1}=1,j_{2}=2\right|}'
    sT(cbra, "JzBraCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))")

    assert str(cket_big) == '|1,0,j1=1,j2=2,j3=3,j(1,2)=3>'
    # TODO: Fix non-unicode pretty printing
    # i.e. j1,2 -> j(1,2)
    assert pretty(cket_big) == '|1,0,j1=1,j2=2,j3=3,j1,2=3>'
    # 断言：验证 `cket_big` 的可打印表示是否为 '❘1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3⟩'
    assert upretty(cket_big) == '❘1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3⟩'

    # 断言：验证 `cket_big` 的 LaTeX 表示是否为指定格式
    assert latex(cket_big) == \
        r'{\left|1,0,j_{1}=1,j_{2}=2,j_{3}=3,j_{1,2}=3\right\rangle }'

    # 对 `cket_big` 应用自定义转换函数 `sT`
    sT(cket_big, "JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2), Integer(3)),Tuple(Tuple(Integer(1), Integer(2), Integer(3)), Tuple(Integer(1), Integer(3), Integer(1))))")

    # 断言：验证 `cbra_big` 的字符串表示是否为 '<1,0,j1=1,j2=2,j3=3,j(1,2)=3|'
    assert str(cbra_big) == '<1,0,j1=1,j2=2,j3=3,j(1,2)=3|'

    # 断言：验证 `cbra_big` 的美观输出是否为 '<1,0,j1=1,j2=2,j3=3,j1,2=3|'
    assert pretty(cbra_big) == '<1,0,j1=1,j2=2,j3=3,j1,2=3|'

    # 断言：验证 `cbra_big` 的 Unicode 表示是否为 '⟨1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3❘'
    assert upretty(cbra_big) == '⟨1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3❘'

    # 断言：验证 `cbra_big` 的 LaTeX 表示是否为指定格式
    assert latex(cbra_big) == \
        r'{\left\langle 1,0,j_{1}=1,j_{2}=2,j_{3}=3,j_{1,2}=3\right|}'

    # 对 `cbra_big` 应用自定义转换函数 `sT`
    sT(cbra_big, "JzBraCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2), Integer(3)),Tuple(Tuple(Integer(1), Integer(2), Integer(3)), Tuple(Integer(1), Integer(3), Integer(1))))")

    # 断言：验证 `rot` 的字符串表示是否为 'R(1,2,3)'
    assert str(rot) == 'R(1,2,3)'

    # 断言：验证 `rot` 的美观输出是否为 'R (1,2,3)'
    assert pretty(rot) == 'R (1,2,3)'

    # 断言：验证 `rot` 的 Unicode 表示是否为 'ℛ (1,2,3)'
    assert upretty(rot) == 'ℛ (1,2,3)'

    # 断言：验证 `rot` 的 LaTeX 表示是否为指定格式
    assert latex(rot) == r'\mathcal{R}\left(1,2,3\right)'

    # 对 `rot` 应用自定义转换函数 `sT`
    sT(rot, "Rotation(Integer(1),Integer(2),Integer(3))")

    # 断言：验证 `bigd` 的字符串表示是否为 'WignerD(1, 2, 3, 4, 5, 6)'
    assert str(bigd) == 'WignerD(1, 2, 3, 4, 5, 6)'

    # 赋值：设置 ASCII 字符串的多行赋值
    ascii_str = \
"""\
 1         \n\
D   (4,5,6)\n\
 2,3       \
"""
ucode_str = \
"""\
 1         \n\
D   (4,5,6)\n\
 2,3       \
"""
assert pretty(bigd) == ascii_str
assert upretty(bigd) == ucode_str
assert latex(bigd) == r'D^{1}_{2,3}\left(4,5,6\right)'
sT(bigd, "WignerD(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
assert str(smalld) == 'WignerD(1, 2, 3, 0, 4, 0)'
ascii_str = \
"""\
 1     \n\
d   (4)\n\
 2,3   \
"""
ucode_str = \
"""\
 1     \n\
d   (4)\n\
 2,3   \
"""
assert pretty(smalld) == ascii_str
assert upretty(smalld) == ucode_str
assert latex(smalld) == r'd^{1}_{2,3}\left(4\right)'
sT(smalld, "WignerD(Integer(1), Integer(2), Integer(3), Integer(0), Integer(4), Integer(0))")


def test_state():
    x = symbols('x')
    # 创建 Bra 和 Ket 对象
    bra = Bra()
    ket = Ket()
    # 创建具有符号参数的 Bra 和 Ket 对象
    bra_tall = Bra(x/2)
    ket_tall = Ket(x/2)
    # 创建 TimeDepBra 和 TimeDepKet 对象
    tbra = TimeDepBra()
    tket = TimeDepKet()
    # 验证 Bra 对象的字符串表示
    assert str(bra) == '<psi|'
    # 验证 Bra 对象的美观字符串表示
    assert pretty(bra) == '<psi|'
    # 验证 Bra 对象的 Unicode 美观字符串表示
    assert upretty(bra) == '⟨ψ❘'
    # 验证 Bra 对象的 LaTeX 表示
    assert latex(bra) == r'{\left\langle \psi\right|}'
    # 斯皮诺（sT）函数调用，记录 Bra 对象
    sT(bra, "Bra(Symbol('psi'))")
    # 验证 Ket 对象的字符串表示
    assert str(ket) == '|psi>'
    # 验证 Ket 对象的美观字符串表示
    assert pretty(ket) == '|psi>'
    # 验证 Ket 对象的 Unicode 美观字符串表示
    assert upretty(ket) == '❘ψ⟩'
    # 验证 Ket 对象的 LaTeX 表示
    assert latex(ket) == r'{\left|\psi\right\rangle }'
    # 斯皮诺函数调用，记录 Ket 对象
    sT(ket, "Ket(Symbol('psi'))")
    # 验证 Bra 对象（含符号参数）的字符串表示
    assert str(bra_tall) == '<x/2|'
    ascii_str = \
"""\
 / |\n\
/ x|\n\
\\ -|\n\
 \\2|\
"""
    ucode_str = \
"""\
 ╱ │\n\
╱ x│\n\
╲ ─│\n\
 ╲2│\
"""
    # 验证 Bra 对象（含符号参数）的美观字符串表示
    assert pretty(bra_tall) == ascii_str
    # 验证 Bra 对象（含符号参数）的 Unicode 美观字符串表示
    assert upretty(bra_tall) == ucode_str
    # 验证 Bra 对象（含符号参数）的 LaTeX 表示
    assert latex(bra_tall) == r'{\left\langle \frac{x}{2}\right|}'
    # 斯皮诺函数调用，记录 Bra 对象（含符号参数）
    sT(bra_tall, "Bra(Mul(Rational(1, 2), Symbol('x')))")
    # 验证 Ket 对象（含符号参数）的字符串表示
    assert str(ket_tall) == '|x/2>'
    ascii_str = \
"""\
| \\ \n\
|x \\\n\
|- /\n\
|2/ \
"""
    ucode_str = \
"""\
│ ╲ \n\
│x ╲\n\
│─ ╱\n\
│2╱ \
"""
    # 验证 Ket 对象（含符号参数）的美观字符串表示
    assert pretty(ket_tall) == ascii_str
    # 验证 Ket 对象（含符号参数）的 Unicode 美观字符串表示
    assert upretty(ket_tall) == ucode_str
    # 验证 Ket 对象（含符号参数）的 LaTeX 表示
    assert latex(ket_tall) == r'{\left|\frac{x}{2}\right\rangle }'
    # 斯皮诺函数调用，记录 Ket 对象（含符号参数）
    sT(ket_tall, "Ket(Mul(Rational(1, 2), Symbol('x')))")
    # 验证 TimeDepBra 对象的字符串表示
    assert str(tbra) == '<psi;t|'
    # 验证 TimeDepBra 对象的美观字符串表示
    assert pretty(tbra) == '<psi;t|'
    # 验证 TimeDepBra 对象的 Unicode 美观字符串表示
    assert upretty(tbra) == '⟨ψ;t❘'
    # 验证 TimeDepBra 对象的 LaTeX 表示
    assert latex(tbra) == r'{\left\langle \psi;t\right|}'
    # 斯皮诺函数调用，记录 TimeDepBra 对象
    sT(tbra, "TimeDepBra(Symbol('psi'),Symbol('t'))")
    # 验证 TimeDepKet 对象的字符串表示
    assert str(tket) == '|psi;t>'
    # 验证 TimeDepKet 对象的美观字符串表示
    assert pretty(tket) == '|psi;t>'
    # 验证 TimeDepKet 对象的 Unicode 美观字符串表示
    assert upretty(tket) == '❘ψ;t⟩'
    # 验证 TimeDepKet 对象的 LaTeX 表示
    assert latex(tket) == r'{\left|\psi;t\right\rangle }'
    # 斯皮诺函数调用，记录 TimeDepKet 对象
    sT(tket, "TimeDepKet(Symbol('psi'),Symbol('t'))")


def test_tensorproduct():
    # 创建 JzKet 对象并将其张量积
    tp = TensorProduct(JzKet(1, 1), JzKet(1, 0))
    # 验证 TensorProduct 对象的字符串表示
    assert str(tp) == '|1,1>x|1,0>'
    # 验证 TensorProduct 对象的美观字符串表示
    assert pretty(tp) == '|1,1>x |1,0>'
    # 验证 TensorProduct 对象的 Unicode 美观字符串表示
    assert upretty(tp) == '❘1,1⟩⨂ ❘1,0⟩'
    # 验证 TensorProduct 对象的 LaTeX 表示
    assert latex(tp) == \
        r'{{\left|1,1\right\rangle }}\otimes {{\left|1,0\right\rangle }}'
    # 斯皮诺函数调用，记录 TensorProduct 对象
    sT(tp, "TensorProduct(JzKet(Integer(1),Integer(1)), JzKet(Integer(1),Integer(0)))")


def test_big_expr():
    # 创建函数对象和符号对象
    f = Function('f')
    x = symbols('x')
    # 计算表达式 e1，包括算符的反对易子、微分算符的幂、张量积以及 Jz 的操作
    e1 = Dagger(AntiCommutator(Operator('A') + Operator('B'), Pow(DifferentialOperator(Derivative(f(x), x), f(x)), 3))*TensorProduct(Jz**2, Operator('A') + Operator('B')))*(JzBra(1, 0) + JzBra(1, 1))*(JzKet(0, 0) + JzKet(1, -1))
    
    # 计算表达式 e2，包括 Jz 的对易子、算符的反对易子、逆算符的平方、以及 Jz 的对易子的共轭
    e2 = Commutator(Jz**2, Operator('A') + Operator('B'))*AntiCommutator(Dagger(Operator('C')*Operator('D')), Operator('E').inv()**2)*Dagger(Commutator(Jz, J2))
    
    # 计算表达式 e3，包括 Wigner 3j 符号、张量积、对易子、算符的共轭、张量积的外积等操作
    e3 = Wigner3j(1, 2, 3, 4, 5, 6)*TensorProduct(Commutator(Operator('A') + Dagger(Operator('B')), Operator('C') + Operator('D')), Jz - J2)*Dagger(OuterProduct(Dagger(JzBra(1, 1)), JzBra(1, 0)))*TensorProduct(JzKetCoupled(1, 1, (1, 1)) + JzKetCoupled(1, 0, (1, 1)), JzKetCoupled(1, -1, (1, 1)))
    
    # 计算表达式 e4，包括复数空间、费克空间、L2 空间、希尔伯特空间的和
    e4 = (ComplexSpace(1)*ComplexSpace(2) + FockSpace()**2)*(L2(Interval(
        0, oo)) + HilbertSpace())
    
    # 断言表达式 e1 的字符串表示符合特定的格式
    assert str(e1) == '(Jz**2)x(Dagger(A) + Dagger(B))*{Dagger(DifferentialOperator(Derivative(f(x), x),f(x)))**3,Dagger(A) + Dagger(B)}*(<1,0| + <1,1|)*(|0,0> + |1,-1>)'
    
    # 定义一个 ASCII 字符串，可能在后续的代码中使用
    ascii_str = \
# 定义一个长字符串，包含特定格式的数学表达式，使用反斜杠转义换行符连接多行字符串
ucode_str = \
"""\
                 ⎧                                      3        ⎫                                 \n\
                 ⎪⎛                                   †⎞         ⎪                                 \n\
    2  ⎛ †    †⎞ ⎨⎜                    ⎛d            ⎞ ⎟   †    †⎬                                 \n\
⎛J ⎞ ⨂ ⎝A  + B ⎠⋅⎪⎜DifferentialOperator⎜──(f(x)),f(x)⎟ ⎟ ,A  + B ⎪⋅(⟨1,0❘ + ⟨1,1❘)⋅(❘0,0⟩ + ❘1,-1⟩)\n\
⎝ z⎠             ⎩⎝                    ⎝dx           ⎠ ⎠         ⎭                                 \
"""
# 使用断言测试 pretty 函数对 e1 的输出是否与 ascii_str 相等
assert pretty(e1) == ascii_str
# 使用断言测试 upretty 函数对 e1 的输出是否与 ucode_str 相等
assert upretty(e1) == ucode_str
# 使用断言测试 latex 函数对 e1 的输出是否与指定的 LaTeX 字符串相等
assert latex(e1) == \
    r'{J_z^{2}}\otimes \left({A^{\dagger} + B^{\dagger}}\right) \left\{\left(DifferentialOperator\left(\frac{d}{d x} f{\left(x \right)},f{\left(x \right)}\right)^{\dagger}\right)^{3},A^{\dagger} + B^{\dagger}\right\} \left({\left\langle 1,0\right|} + {\left\langle 1,1\right|}\right) \left({\left|0,0\right\rangle } + {\left|1,-1\right\rangle }\right)'
# 调用 sT 函数，测试其对 e1 的输出
sT(e1, "Mul(TensorProduct(Pow(JzOp(Symbol('J')), Integer(2)), Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), AntiCommutator(Pow(Dagger(DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))), Integer(3)),Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), Add(JzBra(Integer(1),Integer(0)), JzBra(Integer(1),Integer(1))), Add(JzKet(Integer(0),Integer(0)), JzKet(Integer(1),Integer(-1))))")

# 定义 ascii_str，包含特定格式的数学表达式，使用反斜杠转义换行符连接多行字符串
ascii_str = \
"""\
[    2      ] / -2  +  +\\ [ 2   ]\n\
[/J \\ ,A + B]*<E  ,D *C >*[J ,J ]\n\
[\\ z/       ] \\         / [    z]\
"""
# 定义 ucode_str，包含特定格式的数学表达式，使用反斜杠转义换行符连接多行字符串
ucode_str = \
"""\
⎡    2      ⎤ ⎧ -2  †  †⎫ ⎡ 2   ⎤\n\
⎢⎛J ⎞ ,A + B⎥⋅⎨E  ,D ⋅C ⎬⋅⎢J ,J ⎥\n\
⎣⎝ z⎠       ⎦ ⎩         ⎭ ⎣    z⎦\
"""
# 使用断言测试 pretty 函数对 e2 的输出是否与 ascii_str 相等
assert pretty(e2) == ascii_str
# 使用断言测试 upretty 函数对 e2 的输出是否与 ucode_str 相等
assert upretty(e2) == ucode_str
# 使用断言测试 latex 函数对 e2 的输出是否与指定的 LaTeX 字符串相等
assert latex(e2) == \
    r'\left[J_z^{2},A + B\right] \left\{E^{-2},D^{\dagger} C^{\dagger}\right\} \left[J^2,J_z\right]'
# 调用 sT 函数，测试其对 e2 的输出
sT(e2, "Mul(Commutator(Pow(JzOp(Symbol('J')), Integer(2)),Add(Operator(Symbol('A')), Operator(Symbol('B')))), AntiCommutator(Pow(Operator(Symbol('E')), Integer(-2)),Mul(Dagger(Operator(Symbol('D'))), Dagger(Operator(Symbol('C'))))), Commutator(J2Op(Symbol('J')),JzOp(Symbol('J'))))")
    # 断言，验证 e3 对象转换为字符串后是否等于指定的字符串
    assert str(e3) == \
        "Wigner3j(1, 2, 3, 4, 5, 6)*[Dagger(B) + A,C + D]x(-J2 + Jz)*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x|1,-1,j1=1,j2=1>"
    
    # 创建 ascii_str 变量，继续下一行
    ascii_str = \
"""\
          [ +          ]  /   2     \\                                                                 \n\
/1  3  5\\*[B  + A,C + D]x |- J  + J |*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x |1,-1,j1=1,j2=1>\n\
|       |                 \\        z/                                                                 \n\
\\2  4  6/                                                                                             \
"""
# 定义 ASCII 艺术风格的字符串，表示一个数学表达式
ucode_str = \
"""\
          ⎡ †          ⎤  ⎛   2     ⎞                                                                 \n\
⎛1  3  5⎞⋅⎣B  + A,C + D⎦⨂ ⎜- J  + J ⎟⋅❘1,0⟩⟨1,1❘⋅(❘1,0,j₁=1,j₂=1⟩ + ❘1,1,j₁=1,j₂=1⟩)⨂ ❘1,-1,j₁=1,j₂=1⟩\n\
⎜       ⎟                 ⎝        z⎠                                                                 \n\
⎝2  4  6⎠                                                                                             \
"""
# 定义 Unicode 艺术风格的字符串，表示同一个数学表达式

assert pretty(e3) == ascii_str
assert upretty(e3) == ucode_str
# 断言：使用定义的 ASCII 和 Unicode 表达式进行 Pretty 打印时应该得到相同的结果

assert latex(e3) == \
    r'\left(\begin{array}{ccc} 1 & 3 & 5 \\ 2 & 4 & 6 \end{array}\right) {\left[B^{\dagger} + A,C + D\right]}\otimes \left({- J^2 + J_z}\right) {\left|1,0\right\rangle }{\left\langle 1,1\right|} \left({{\left|1,0,j_{1}=1,j_{2}=1\right\rangle } + {\left|1,1,j_{1}=1,j_{2}=1\right\rangle }}\right)\otimes {{\left|1,-1,j_{1}=1,j_{2}=1\right\rangle }}'
# 断言：使用 LaTeX 打印时，表达式 e3 应该与给定的 LaTeX 字符串相匹配

sT(e3, "Mul(Wigner3j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6)), TensorProduct(Commutator(Add(Dagger(Operator(Symbol('B'))), Operator(Symbol('A'))),Add(Operator(Symbol('C')), Operator(Symbol('D')))), Add(Mul(Integer(-1), J2Op(Symbol('J'))), JzOp(Symbol('J')))), OuterProduct(JzKet(Integer(1),Integer(0)),JzBra(Integer(1),Integer(1))), TensorProduct(Add(JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))), JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))), JzKetCoupled(Integer(1),Integer(-1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))))")
# 调用 sT 函数对 e3 进行特定操作的测试，参数是一个复杂的表达式字符串

assert str(e4) == '(C(1)*C(2)+F**2)*(L2(Interval(0, oo))+H)'
# 断言：将 e4 转换为字符串应该与给定的字符串相匹配

ascii_str = \
"""\
// 1    2\\    x2\\   / 2    \\\n\
\\\\C  x C / + F  / x \\L  + H/\
"""
# 定义 ASCII 艺术风格的字符串，表示另一个数学表达式

ucode_str = \
"""\
⎛⎛ 1    2⎞    ⨂2⎞   ⎛ 2    ⎞\n\
⎝⎝C  ⨂ C ⎠ ⊕ F  ⎠ ⨂ ⎝L  ⊕ H⎠\
"""
# 定义 Unicode 艺术风格的字符串，表示同一个数学表达式

assert pretty(e4) == ascii_str
assert upretty(e4) == ucode_str
# 断言：使用定义的 ASCII 和 Unicode 表达式进行 Pretty 打印时应该得到相同的结果

assert latex(e4) == \
    r'\left(\left(\mathcal{C}^{1}\otimes \mathcal{C}^{2}\right)\oplus {\mathcal{F}}^{\otimes 2}\right)\otimes \left({\mathcal{L}^2}\left( \left[0, \infty\right) \right)\oplus \mathcal{H}\right)'
# 断言：使用 LaTeX 打印时，表达式 e4 应该与给定的 LaTeX 字符串相匹配

sT(e4, "TensorProductHilbertSpace((DirectSumHilbertSpace(TensorProductHilbertSpace(ComplexSpace(Integer(1)),ComplexSpace(Integer(2))),TensorPowerHilbertSpace(FockSpace(),Integer(2)))),(DirectSumHilbertSpace(L2(Interval(Integer(0), oo, false, true)),HilbertSpace())))")
# 调用 sT 函数对 e4 进行特定操作的测试，参数是一个复杂的表达式字符串

def _test_sho1d():
    # 创建一个上升算子 'a'
    ad = RaisingOp('a')
    assert pretty(ad) == ' \N{DAGGER}\na '
    # 断言：使用 pretty 函数打印 ad 应该得到给定的字符串
    # 断言语句，用于确保变量 ad 被转换为 LaTeX 格式后等于字符串 'a^{\\dagger}'
    assert latex(ad) == 'a^{\\dagger}'
```