# `D:\src\scipysrc\sympy\sympy\core\tests\test_noncommutative.py`

```
"""Tests for noncommutative symbols and expressions."""

# 导入符号代数相关模块和函数
from sympy.core.function import expand
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import (cancel, factor)
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.radsimp import (collect, radsimp, rcollect)
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import (posify, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.abc import x, y, z
from sympy.testing.pytest import XFAIL

# 定义非交换符号和表达式的测试用例
A, B, C = symbols("A B C", commutative=False)
X = symbols("X", commutative=False, hermitian=True)
Y = symbols("Y", commutative=False, antihermitian=True)

# 测试求共轭转置、伴随
def test_adjoint():
    assert adjoint(A).is_commutative is False  # 检查伴随操作后符号是否为可交换的
    assert adjoint(A*A) == adjoint(A)**2  # 检查伴随操作后的平方是否等于平方的伴随
    assert adjoint(A*B) == adjoint(B)*adjoint(A)  # 检查伴随操作后的乘积是否等于乘积的逆序伴随
    assert adjoint(A*B**2) == adjoint(B)**2*adjoint(A)  # 检查伴随操作后的幂次乘积是否符合伴随的幂次法则
    assert adjoint(A*B - B*A) == adjoint(B)*adjoint(A) - adjoint(A)*adjoint(B)  # 检查伴随操作后的差是否等于差的逆序伴随
    assert adjoint(A + I*B) == adjoint(A) - I*adjoint(B)  # 检查伴随操作后的和是否符合伴随的和法则

    assert adjoint(X) == X  # 检查对厄米特符号的伴随操作是否等于其自身
    assert adjoint(-I*X) == I*X  # 检查对厄米特符号乘以虚数单位的伴随操作是否符合伴随的线性法则
    assert adjoint(Y) == -Y  # 检查对反厄米特符号的伴随操作是否等于其相反数
    assert adjoint(-I*Y) == -I*Y  # 检查对反厄米特符号乘以虚数单位的伴随操作是否符合伴随的线性法则

    assert adjoint(X) == conjugate(transpose(X))  # 检查伴随操作是否等于共轭转置
    assert adjoint(Y) == conjugate(transpose(Y))  # 检查伴随操作是否等于共轭转置
    assert adjoint(X) == transpose(conjugate(X))  # 检查伴随操作是否等于转置共轭
    assert adjoint(Y) == transpose(conjugate(Y))  # 检查伴随操作是否等于转置共轭

# 测试取消化简
def test_cancel():
    assert cancel(A*B - B*A) == A*B - B*A  # 检查取消化简对于非交换乘积是否不影响结果
    assert cancel(A*B*(x - 1)) == A*B*(x - 1)  # 检查取消化简对于非交换乘积含有变量的表达式是否不影响结果
    assert cancel(A*B*(x**2 - 1)/(x + 1)) == A*B*(x - 1)  # 检查取消化简对于非交换乘积含有变量的有理表达式是否得到正确的结果
    assert cancel(A*B*(x**2 - 1)/(x + 1) - B*A*(x - 1)) == A*B*(x - 1) + (1 - x)*B*A  # 检查取消化简对于包含多项式和有理函数的表达式是否得到正确的结果

@XFAIL
def test_collect():
    assert collect(A*B - B*A, A) == A*B - B*A  # 检查收集操作对于非交换乘积是否不影响结果
    assert collect(A*B - B*A, B) == A*B - B*A  # 检查收集操作对于非交换乘积是否不影响结果
    assert collect(A*B - B*A, x) == A*B - B*A  # 检查收集操作对于非交换乘积是否不影响结果

# 测试组合化简
def test_combsimp():
    assert combsimp(A*B - B*A) == A*B - B*A  # 检查组合化简对于非交换乘积是否不影响结果

# 测试伽马函数化简
def test_gammasimp():
    assert gammasimp(A*B - B*A) == A*B - B*A  # 检查伽马函数化简对于非交换乘积是否不影响结果

# 测试共轭
def test_conjugate():
    assert conjugate(A).is_commutative is False  # 检查共轭操作后符号是否为可交换的
    assert (A*A).conjugate() == conjugate(A)**2  # 检查共轭操作后的平方是否等于平方的共轭
    assert (A*B).conjugate() == conjugate(A)*conjugate(B)  # 检查共轭操作后的乘积是否等于乘积的共轭乘积
    assert (A*B**2).conjugate() == conjugate(A)*conjugate(B)**2  # 检查共轭操作后的幂次乘积是否符合共轭的幂次法则
    assert (A*B - B*A).conjugate() == \
        conjugate(A)*conjugate(B) - conjugate(B)*conjugate(A)  # 检查共轭操作后的差是否等于差的逆序共轭
    assert (A*B).conjugate() - (B*A).conjugate() == \
        conjugate(A)*conjugate(B) - conjugate(B)*conjugate(A)  # 检查共轭操作后的差的共轭是否等于差的逆序共轭

    assert (A + I*B).conjugate() == conjugate(A) - I*conjugate(B)  # 检查共轭操作后的和是否符合共轭的和法则

# 测试展开
def test_expand():
    assert expand((A*B)**2) == A*B*A*B  # 检查展开操作是否正确展开乘方的非交换乘积
    assert expand(A*B - B*A) == A*B - B*A  # 检查展开操作对于非交换乘积是否不影响结果
    assert expand((A*B/A)**2) == A*B*B/A  # 检查展开操作是否正确展开含有除法的非交换乘积的乘方
    assert expand(B*A*(A + B)*B) == B*A**2*B + B*A*B**2  # 检查展开操作是否正确展开包含多项式的非交换乘积

# 测试因式分解
def test_factor():
    assert factor(A*B - B*A) == A*B - B*A  # 检查因式分解操作对于非交换乘积是否不影响结果

# 测试正数化简
def test_posify():
    # 断言A的正则表达式的第一个元素是不可交换的
    assert posify(A)[0].is_commutative is False
    
    # 对于元组中的每个表达式q：
    for q in (A*B/A, (A*B/A)**2, (A*B)**2, A*B - B*A):
        # 对q进行正则表达式
        p = posify(q)
        # 断言对于正则表达式的第一个元素，将第二个元素代入后等于q
        assert p[0].subs(p[1]) == q
# 定义一个测试函数，用于验证 radsimp 函数的行为是否符合预期
def test_radsimp():
    # 断言 radsimp(A*B - B*A) 的结果应该等于 A*B - B*A
    assert radsimp(A*B - B*A) == A*B - B*A


# 标记为 XFAIL 的测试函数，预期其失败
@XFAIL
def test_ratsimp():
    # 断言 ratsimp(A*B - B*A) 的结果应该等于 A*B - B*A
    assert ratsimp(A*B - B*A) == A*B - B*A


# 标记为 XFAIL 的测试函数，预期其失败
@XFAIL
def test_rcollect():
    # 断言 rcollect(A*B - B*A, A) 的结果应该等于 A*B - B*A
    assert rcollect(A*B - B*A, A) == A*B - B*A
    # 断言 rcollect(A*B - B*A, B) 的结果应该等于 A*B - B*A
    assert rcollect(A*B - B*A, B) == A*B - B*A
    # 断言 rcollect(A*B - B*A, x) 的结果应该等于 A*B - B*A
    assert rcollect(A*B - B*A, x) == A*B - B*A


# 定义一个测试函数，用于验证 simplify 函数的行为是否符合预期
def test_simplify():
    # 断言 simplify(A*B - B*A) 的结果应该等于 A*B - B*A
    assert simplify(A*B - B*A) == A*B - B*A


# 定义一个测试函数，用于验证 subs 方法的行为是否符合预期
def test_subs():
    # 断言 (x*y*A).subs(x*y, z) 的结果应该等于 A*z
    assert (x*y*A).subs(x*y, z) == A*z
    # 断言 (x*A*B).subs(x*A, C) 的结果应该等于 C*B
    assert (x*A*B).subs(x*A, C) == C*B
    # 断言 (x*A*x*x).subs(x**2*A, C) 的结果应该等于 x*C
    assert (x*A*x*x).subs(x**2*A, C) == x*C
    # 断言 (x*A*x*B).subs(x**2*A, C) 的结果应该等于 C*B
    assert (x*A*x*B).subs(x**2*A, C) == C*B
    # 断言 (A**2*B**2).subs(A*B**2, C) 的结果应该等于 A*C
    assert (A**2*B**2).subs(A*B**2, C) == A*C
    # 断言 (A*A*A + A*B*A).subs(A*A*A, C) 的结果应该等于 C + A*B*A
    assert (A*A*A + A*B*A).subs(A*A*A, C) == C + A*B*A


# 定义一个测试函数，用于验证 transpose 函数的行为是否符合预期
def test_transpose():
    # 断言 transpose(A).is_commutative 的结果应该为 False
    assert transpose(A).is_commutative is False
    # 断言 transpose(A*A) 的结果应该等于 transpose(A)**2
    assert transpose(A*A) == transpose(A)**2
    # 断言 transpose(A*B) 的结果应该等于 transpose(B)*transpose(A)
    assert transpose(A*B) == transpose(B)*transpose(A)
    # 断言 transpose(A*B**2) 的结果应该等于 transpose(B)**2*transpose(A)
    assert transpose(A*B**2) == transpose(B)**2*transpose(A)
    # 断言 transpose(A*B - B*A) 的结果应该等于 transpose(B)*transpose(A) - transpose(A)*transpose(B)
    assert transpose(A*B - B*A) == \
        transpose(B)*transpose(A) - transpose(A)*transpose(B)
    # 断言 transpose(A + I*B) 的结果应该等于 transpose(A) + I*transpose(B)
    assert transpose(A + I*B) == transpose(A) + I*transpose(B)

    # 断言 transpose(X) 的结果应该等于 conjugate(X)
    assert transpose(X) == conjugate(X)
    # 断言 transpose(-I*X) 的结果应该等于 -I*conjugate(X)
    assert transpose(-I*X) == -I*conjugate(X)
    # 断言 transpose(Y) 的结果应该等于 -conjugate(Y)
    assert transpose(Y) == -conjugate(Y)
    # 断言 transpose(-I*Y) 的结果应该等于 I*conjugate(Y)
    assert transpose(-I*Y) == I*conjugate(Y)


# 定义一个测试函数，用于验证 trigsimp 函数的行为是否符合预期
def test_trigsimp():
    # 断言 trigsimp(A*sin(x)**2 + A*cos(x)**2) 的结果应该等于 A
    assert trigsimp(A*sin(x)**2 + A*cos(x)**2) == A
```