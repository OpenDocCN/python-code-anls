# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_complexes.py`

```
from sympy.core.expr import Expr  # 导入 Expr 类
from sympy.core.function import (Derivative, Function, Lambda, expand)  # 导入多个函数和类
from sympy.core.numbers import (E, I, Rational, comp, nan, oo, pi, zoo)  # 导入多个常数和函数
from sympy.core.relational import Eq  # 导入 Eq 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol 类和 symbols 函数
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, sign, transpose)  # 导入多个复数函数
from sympy.functions.elementary.exponential import (exp, exp_polar, log)  # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入 Piecewise 类
from sympy.functions.elementary.trigonometric import (acos, atan, atan2, cos, sin)  # 导入三角函数
from sympy.functions.elementary.hyperbolic import sinh  # 导入双曲正弦函数
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)  # 导入特殊函数
from sympy.integrals.integrals import Integral  # 导入 Integral 类
from sympy.matrices.dense import Matrix  # 导入稠密矩阵类
from sympy.matrices.expressions.funcmatrix import FunctionMatrix  # 导入函数矩阵类
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号类
from sympy.matrices.immutable import (ImmutableMatrix, ImmutableSparseMatrix)  # 导入不可变矩阵类
from sympy.matrices import SparseMatrix  # 导入稀疏矩阵类
from sympy.sets.sets import Interval  # 导入区间类
from sympy.core.expr import unchanged  # 导入 unchanged 函数
from sympy.core.function import ArgumentIndexError  # 导入 ArgumentIndexError 异常
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow  # 导入测试函数和装饰器


def N_equals(a, b):
    """检查两个复数是否在数值上接近"""
    return comp(a.n(), b.n(), 1.e-6)


def test_re():
    x, y = symbols('x,y')  # 创建符号变量 x 和 y
    a, b = symbols('a,b', real=True)  # 创建实数符号变量 a 和 b

    r = Symbol('r', real=True)  # 创建一个实数符号 r
    i = Symbol('i', imaginary=True)  # 创建一个虚数符号 i

    assert re(nan) is nan  # 测试 re 函数对 nan 的处理

    assert re(oo) is oo  # 测试 re 函数对正无穷的处理
    assert re(-oo) is -oo  # 测试 re 函数对负无穷的处理

    assert re(0) == 0  # 测试 re 函数对整数 0 的处理

    assert re(1) == 1  # 测试 re 函数对整数 1 的处理
    assert re(-1) == -1  # 测试 re 函数对整数 -1 的处理

    assert re(E) == E  # 测试 re 函数对自然常数 E 的处理
    assert re(-E) == -E  # 测试 re 函数对负自然常数 E 的处理

    assert unchanged(re, x)  # 测试 re 函数对符号变量 x 的处理
    assert re(x*I) == -im(x)  # 测试 re 函数对虚数 x*I 的处理
    assert re(r*I) == 0  # 测试 re 函数对实数 r*I 的处理
    assert re(r) == r  # 测试 re 函数对实数 r 的处理
    assert re(i*I) == I * i  # 测试 re 函数对虚数 i*I 的处理
    assert re(i) == 0  # 测试 re 函数对虚数 i 的处理

    assert re(x + y) == re(x) + re(y)  # 测试 re 函数的加法性质
    assert re(x + r) == re(x) + r  # 测试 re 函数对实数 r 加 x 的处理

    assert re(re(x)) == re(x)  # 测试 re 函数的幂等性质

    assert re(2 + I) == 2  # 测试 re 函数对复数 2+I 的处理
    assert re(x + I) == re(x)  # 测试 re 函数对 x+I 的处理

    assert re(x + y*I) == re(x) - im(y)  # 测试 re 函数对复数 x+y*I 的处理
    assert re(x + r*I) == re(x)  # 测试 re 函数对 x+r*I 的处理

    assert re(log(2*I)) == log(2)  # 测试 re 函数对 log(2*I) 的处理

    assert re((2 + I)**2).expand(complex=True) == 3  # 测试 re 函数对 (2+I)^2 的处理

    assert re(conjugate(x)) == re(x)  # 测试 re 函数对 conjugate(x) 的处理
    assert conjugate(re(x)) == re(x)  # 测试 conjugate 函数对 re(x) 的处理

    assert re(x).as_real_imag() == (re(x), 0)  # 测试 re 函数的实部-虚部分解

    assert re(i*r*x).diff(r) == re(i*x)  # 测试 re 函数对 i*r*x 求 r 的偏导数
    assert re(i*r*x).diff(i) == I*r*im(x)  # 测试 re 函数对 i*r*x 求 i 的偏导数

    assert re(
        sqrt(a + b*I)) == (a**2 + b**2)**Rational(1, 4)*cos(atan2(b, a)/2)  # 测试 re 函数对 sqrt(a + b*I) 的处理
    assert re(a * (2 + b*I)) == 2*a  # 测试 re 函数对 a * (2 + b*I) 的处理

    assert re((1 + sqrt(a + b*I))/2) == \
        (a**2 + b**2)**Rational(1, 4)*cos(atan2(b, a)/2)/2 + S.Half  # 测试 re 函数对 (1 + sqrt(a + b*I))/2 的处理

    assert re(x).rewrite(im) == x - S.ImaginaryUnit*im(x)  # 测试 re 函数在 im 域下的重写
    assert (x + re(y)).rewrite(re, im) == x + y - S.ImaginaryUnit*im(y)  # 测试 re 函数在 re, im 域下的重写

    a = Symbol('a', algebraic=True)  # 创建一个代数符号 a
    t = Symbol('t', transcendental=True)  # 创建一个超越符号 t
    x = Symbol('x')  # 创建一个通用符号 x
    assert re(a).is_algebraic  # 测试 re 函数对代数符号 a 的属性检查
    # 断言检查关于复数实部的计算是否正确
    assert re(x).is_algebraic is None
    # 断言检查关于复数实部的计算是否返回 False
    assert re(t).is_algebraic is False
    
    # 断言检查复数无穷大的实部是否为 NaN
    assert re(S.ComplexInfinity) is S.NaN
    
    # 定义符号变量 n, m, l
    n, m, l = symbols('n m l')
    # 创建一个 n x m 的矩阵符号 A
    A = MatrixSymbol('A', n, m)
    # 断言检查矩阵 A 的实部是否等于 (1/2) * (A + 共轭矩阵(A))
    assert re(A) == (S.Half) * (A + conjugate(A))
    
    # 定义一个具体的复数矩阵 A
    A = Matrix([[1 + 4*I, 2], [0, -3*I]])
    # 断言检查矩阵 A 的实部是否等于给定的实数部分矩阵
    assert re(A) == Matrix([[1, 2], [0, 0]])
    
    # 定义一个不可变复数矩阵 A
    A = ImmutableMatrix([[1 + 3*I, 3 - 2*I], [0, 2*I]])
    # 断言检查不可变复数矩阵 A 的实部是否等于给定的实数部分矩阵
    assert re(A) == ImmutableMatrix([[1, 3], [0, 0]])
    
    # 创建一个稀疏矩阵 X，其实部由特定规则生成
    X = SparseMatrix([[2*j + i*I for i in range(5)] for j in range(5)])
    # 断言检查稀疏矩阵 X 的实部是否等于给定的稀疏矩阵
    assert re(X) - Matrix([[0, 0, 0, 0, 0],
                           [2, 2, 2, 2, 2],
                           [4, 4, 4, 4, 4],
                           [6, 6, 6, 6, 6],
                           [8, 8, 8, 8, 8]]) == Matrix.zeros(5)
    
    # 断言检查稀疏矩阵 X 的虚部是否等于给定的稀疏矩阵
    assert im(X) - Matrix([[0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4]]) == Matrix.zeros(5)
    
    # 创建一个函数矩阵 X，每个元素为 Lambda((n, m), n + m*I)
    X = FunctionMatrix(3, 3, Lambda((n, m), n + m*I))
    # 断言检查函数矩阵 X 的实部是否等于给定的矩阵
    assert re(X) == Matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
# 定义测试函数 test_im
def test_im():
    # 定义符号变量 x, y 和 a, b
    x, y = symbols('x,y')
    a, b = symbols('a,b', real=True)

    # 定义实部为实数的符号变量 r 和虚部为虚数的符号变量 i
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)

    # 断言：对于 nan，其虚部应为 nan
    assert im(nan) is nan

    # 断言：对于 oo*I，其虚部应为 oo
    assert im(oo*I) is oo
    # 断言：对于 -oo*I，其虚部应为 -oo
    assert im(-oo*I) is -oo

    # 断言：对于 0，其虚部应为 0
    assert im(0) == 0

    # 断言：对于 1，其虚部应为 0
    assert im(1) == 0
    # 断言：对于 -1，其虚部应为 0
    assert im(-1) == 0

    # 断言：对于 E*I，其虚部应为 E
    assert im(E*I) == E
    # 断言：对于 -E*I，其虚部应为 -E
    assert im(-E*I) == -E

    # 断言：未改变函数 im 的参数 x，其虚部应与其实部相等
    assert unchanged(im, x)
    # 断言：对于 x*I，其虚部应为 x 的实部
    assert im(x*I) == re(x)
    # 断言：对于 r*I，其虚部应为 r
    assert im(r*I) == r
    # 断言：对于 r，其虚部应为 0
    assert im(r) == 0
    # 断言：对于 i*I，其虚部应为 0
    assert im(i*I) == 0
    # 断言：对于 i，其虚部应为 -I*i
    assert im(i) == -I * i

    # 断言：对于 x + y，其虚部应为 x 和 y 的虚部之和
    assert im(x + y) == im(x) + im(y)
    # 断言：对于 x + r，其虚部应为 x 的虚部
    assert im(x + r) == im(x)
    # 断言：对于 x + r*I，其虚部应为 x 的虚部加上 r
    assert im(x + r*I) == im(x) + r

    # 断言：对于 im(x)*I，其虚部应为 x 的虚部
    assert im(im(x)*I) == im(x)

    # 断言：对于 2 + I，其虚部应为 1
    assert im(2 + I) == 1
    # 断言：对于 x + I，其虚部应为 x 的虚部加上 1
    assert im(x + I) == im(x) + 1

    # 断言：对于 x + y*I，其虚部应为 x 的虚部加上 y 的实部
    assert im(x + y*I) == im(x) + re(y)
    # 断言：对于 x + r*I，其虚部应为 x 的虚部加上 r
    assert im(x + r*I) == im(x) + r

    # 断言：对于 log(2*I)，其虚部应为 pi/2
    assert im(log(2*I)) == pi/2

    # 断言：对于 (2 + I)**2 的虚部，应展开为 4
    assert im((2 + I)**2).expand(complex=True) == 4

    # 断言：对于 conjugate(x) 的虚部，应为 -x 的虚部
    assert im(conjugate(x)) == -im(x)
    # 断言：对于 im(x) 的共轭，其虚部应等于 x 的虚部
    assert conjugate(im(x)) == im(x)

    # 断言：对于 im(x)，其实部和虚部构成的元组应为 (im(x), 0)
    assert im(x).as_real_imag() == (im(x), 0)

    # 断言：对于 im(i*r*x) 对 r 的导数，应为 im(i*x)
    assert im(i*r*x).diff(r) == im(i*x)
    # 断言：对于 im(i*r*x) 对 i 的导数，应为 -I * re(r*x)
    assert im(i*r*x).diff(i) == -I * re(r*x)

    # 断言：对于 sqrt(a + b*I) 的虚部，应为 (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)
    assert im(sqrt(a + b*I)) == (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)
    # 断言：对于 a * (2 + b*I) 的虚部，应为 a*b
    assert im(a * (2 + b*I)) == a*b

    # 断言：对于 (1 + sqrt(a + b*I))/2 的虚部，应为 (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)/2
    assert im((1 + sqrt(a + b*I))/2) == \
        (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)/2

    # 断言：对于 im(x) 重写为 re，应为 -S.ImaginaryUnit * (x - re(x))
    assert im(x).rewrite(re) == -S.ImaginaryUnit * (x - re(x))
    # 断言：对于 (x + im(y)) 重写为 im, re，应为 x - S.ImaginaryUnit * (y - re(y))
    assert (x + im(y)).rewrite(im, re) == x - S.ImaginaryUnit * (y - re(y))

    # 定义 algebraic 和 transcendental 符号变量
    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    # 断言：对于 re(a)，其是否是代数的断言应为 True
    assert re(a).is_algebraic
    # 断言：对于 re(x)，其是否是代数的断言应为 None
    assert re(x).is_algebraic is None
    # 断言：对于 re(t)，其是否是代数的断言应为 False
    assert re(t).is_algebraic is False

    # 断言：对于 im(S.ComplexInfinity)，其虚部应为 S.NaN
    assert im(S.ComplexInfinity) is S.NaN

    # 定义符号变量 n, m, l 和矩阵符号 A
    n, m, l = symbols('n m l')
    A = MatrixSymbol('A', n, m)

    # 断言：对于 A 的虚部，应为 (S.One/(2*I)) * (A - conjugate(A))
    assert im(A) == (S.One/(2*I)) * (A - conjugate(A))

    # 定义矩阵 A，其元素为复数
    A = Matrix([[1 + 4*I, 2],[0, -3*I]])
    # 断言：对于 A 的虚部，应为 Matrix([[4, 0],[0, -3]])
    assert im(A) == Matrix([[4, 0],[0, -3]])

    # 定义不可变矩阵 A，其元素为复数
    A = ImmutableMatrix([[1 + 3*I, 3-2*I],[0, 2*I]])
    # 断言：对于 A 的虚部，应为 ImmutableMatrix([[3, -2],[0, 2]])
    assert im(A) == ImmutableMatrix([[3, -2],[0, 2]])

    # 定义稀疏不可变矩阵 X 和普通矩阵 Y
    X = ImmutableSparseMatrix(
            [[i*I + i for i in range(5)] for i in range(5)])
    Y = SparseMatrix([list(range(5)) for i in range(5)])
    # 断言：对于 X 的虚部（转
    # 断言符号函数应该是实数
    assert sign(x).is_real is None
    # 断言符号函数不应该是零
    assert sign(x).is_zero is None
    # 断言符号函数的运算结果等于符号函数本身
    assert sign(x).doit() == sign(x)
    # 断言对于常数倍的符号函数，符号不变
    assert sign(1.2*x) == sign(x)
    assert sign(2*x) == sign(x)
    # 断言对于复数倍的符号函数，符号乘以虚数单位 I
    assert sign(I*x) == I*sign(x)
    assert sign(-2*I*x) == -I*sign(x)
    # 断言符号函数对于共轭的结果，等于共轭的符号函数
    assert sign(conjugate(x)) == conjugate(sign(x))

    # 创建一个正数符号的符号变量 p
    p = Symbol('p', positive=True)
    # 创建一个负数符号的符号变量 n 和 m
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    # 断言对于正数倍的符号函数，符号不变
    assert sign(2*p*x) == sign(x)
    # 断言对于负数倍的符号函数，符号取反
    assert sign(n*x) == -sign(x)
    # 断言对于多个负数倍的符号函数，符号不变
    assert sign(n*m*x) == sign(x)

    # 将符号变量 x 定义为虚数
    x = Symbol('x', imaginary=True)
    # 断言符号函数是虚数
    assert sign(x).is_imaginary is True
    # 断言符号函数不是整数
    assert sign(x).is_integer is False
    # 断言符号函数不是实数
    assert sign(x).is_real is False
    # 断言符号函数不是零
    assert sign(x).is_zero is False
    # 断言符号函数对 x 的导数等于 2*DiracDelta(-I*x)
    assert sign(x).diff(x) == 2*DiracDelta(-I*x)
    # 断言符号函数的计算结果等于 x 的绝对值的倒数
    assert sign(x).doit() == x / Abs(x)
    # 断言符号函数的共轭等于其负值
    assert conjugate(sign(x)) == -sign(x)

    # 将符号变量 x 定义为实数
    x = Symbol('x', real=True)
    # 断言符号函数不是虚数
    assert sign(x).is_imaginary is False
    # 断言符号函数是整数
    assert sign(x).is_integer is True
    # 断言符号函数是实数
    assert sign(x).is_real is True
    # 断言符号函数是否为零的属性未定义
    assert sign(x).is_zero is None
    # 断言符号函数对 x 的导数等于 2*DiracDelta(x)
    assert sign(x).diff(x) == 2*DiracDelta(x)
    # 断言符号函数的计算结果等于符号函数本身
    assert sign(x).doit() == sign(x)
    # 断言符号函数的共轭等于其本身
    assert conjugate(sign(x)) == sign(x)

    # 将符号变量 x 定义为非零数
    x = Symbol('x', nonzero=True)
    # 断言符号函数不是虚数
    assert sign(x).is_imaginary is False
    # 断言符号函数是整数
    assert sign(x).is_integer is True
    # 断言符号函数是实数
    assert sign(x).is_real is True
    # 断言符号函数不是零
    assert sign(x).is_zero is False
    # 断言符号函数的计算结果等于 x 的绝对值的倒数
    assert sign(x).doit() == x / Abs(x)
    # 断言符号函数对于绝对值函数的结果等于 1
    assert sign(Abs(x)) == 1
    # 断言符号函数的绝对值等于 1
    assert Abs(sign(x)) == 1

    # 将符号变量 x 定义为正数
    x = Symbol('x', positive=True)
    # 断言符号函数不是虚数
    assert sign(x).is_imaginary is False
    # 断言符号函数是整数
    assert sign(x).is_integer is True
    # 断言符号函数是实数
    assert sign(x).is_real is True
    # 断言符号函数不是零
    assert sign(x).is_zero is False
    # 断言符号函数的计算结果等于 x 的绝对值的倒数
    assert sign(x).doit() == x / Abs(x)
    # 断言符号函数对于绝对值函数的结果等于 1
    assert sign(Abs(x)) == 1
    # 断言符号函数的绝对值等于 1
    assert Abs(sign(x)) == 1

    # 将符号变量 x 定义为零
    x = 0
    # 断言符号函数不是虚数
    assert sign(x).is_imaginary is False
    # 断言符号函数是整数
    assert sign(x).is_integer is True
    # 断言符号函数是实数
    assert sign(x).is_real is True
    # 断言符号函数是零
    assert sign(x).is_zero is True
    # 断言符号函数的计算结果等于零
    assert sign(x).doit() == 0
    # 断言符号函数对于绝对值函数的结果等于零
    assert sign(Abs(x)) == 0
    # 断言符号函数的绝对值等于零
    assert Abs(sign(x)) == 0

    # 创建一个非零整数符号变量 nz
    nz = Symbol('nz', nonzero=True, integer=True)
    # 断言符号函数不是虚数
    assert sign(nz).is_imaginary is False
    # 断言符号函数是整数
    assert sign(nz).is_integer is True
    # 断言符号函数是实数
    assert sign(nz).is_real is True
    # 断言符号函数不是零
    assert sign(nz).is_zero is False
    # 断言符号函数的平方等于 1
    assert sign(nz)**2 == 1
    # 断言符号函数的立方等于 (符号函数, 3)
    assert (sign(nz)**3).args == (sign(nz), 3)

    # 断言符号函数对于非负数的符号变量是非负数
    assert sign(Symbol('x', nonnegative=True)).is_nonnegative
    # 断言符号函数对于非负数的符号变量不是非正数
    assert sign(Symbol('x', nonnegative=True)).is_nonpositive is None
    # 断言符号函数对于非正数的符号变量不是非负数
    assert sign(Symbol('x', nonpositive=True)).is_nonnegative is None
    # 断言符号函数对于非正数的符号变量是非正数
    assert sign(Symbol('x', nonpositive=True)).is_nonpositive
    # 断言符号函数对于实数的符号变量不是非负数
    assert sign(Symbol('x', real=True)).is_nonnegative is None
    # 断言符号函数对于实数的符号变量不是非正数
    assert sign(Symbol('x', real=True)).is_nonpositive is None
    # 断言符号函数对于实数的符号变量不是非正数，且不是零
    assert sign(Symbol('x', real=True, zero=False)).is_nonpositive is None

    # 将符号变量 x 和 y 定义为实数和符号变量
    x, y = Symbol('x', real=True), Symbol('y')
    f = Function('f')
    # 断言符号函数用分段函数重写的结果
    assert sign(x).rewrite(Piecewise) == \
        Piecewise((1, x > 0), (-
    # 断言：对 y 的符号函数应用 Heaviside 变换后仍等于原始符号函数
    assert sign(y).rewrite(Heaviside) == sign(y)
    # 断言：对 y 的符号函数应用 Abs 变换后使用分段函数表示
    assert sign(y).rewrite(Abs) == Piecewise((0, Eq(y, 0)), (y/Abs(y), True))
    # 断言：对 f(y) 的符号函数应用 Abs 变换后使用分段函数表示
    assert sign(f(y)).rewrite(Abs) == Piecewise((0, Eq(f(y), 0)), (f(y)/Abs(f(y)), True))

    # 评估可以评估的表达式
    assert sign(exp_polar(I*pi)*pi) is S.NegativeOne

    # 定义表达式 eq
    eq = -sqrt(10 + 6*sqrt(3)) + sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3))
    # 如果有一种快速的方式可以知道何时可以证明这样的表达式为零，则将其与零相比较是可以接受的
    assert sign(eq).func is sign or sign(eq) == 0
    # 但有时候这很困难，因此最好不要用测试拖慢 abs 函数的运行速度
    q = 1 + sqrt(2) - 2*sqrt(3) + 1331*sqrt(6)
    p = expand(q**3)**Rational(1, 3)
    d = p - q
    # 断言：对 d 的符号函数应用函数检查是否是符号函数或者是否为零
    assert sign(d).func is sign or sign(d) == 0
def test_as_real_imag():
    n = pi**1000
    # 计算数值 n 的实部和虚部，如果虚部为零，应返回 (n, 0)
    assert n.as_real_imag() == (n, 0)

    # 解决问题 6261
    x = Symbol('x')
    # 计算 sqrt(x) 的实部和虚部
    assert sqrt(x).as_real_imag() == \
        ((re(x)**2 + im(x)**2)**Rational(1, 4)*cos(atan2(im(x), re(x))/2),
         (re(x)**2 + im(x)**2)**Rational(1, 4)*sin(atan2(im(x), re(x))/2))

    # 解决问题 3853
    a, b = symbols('a,b', real=True)
    # 计算 ((1 + sqrt(a + b*I))/2) 的实部和虚部
    assert ((1 + sqrt(a + b*I))/2).as_real_imag() == \
           (
               (a**2 + b**2)**Rational(1, 4)*cos(atan2(b, a)/2)/2 + S.Half,
               (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)/2)

    # 计算 sqrt(a**2) 的实部和虚部
    assert sqrt(a**2).as_real_imag() == (sqrt(a**2), 0)

    i = symbols('i', imaginary=True)
    # 计算 sqrt(i**2) 的实部和虚部
    assert sqrt(i**2).as_real_imag() == (0, abs(i))

    # 计算 ((1 + I)/(1 - I)) 的实部和虚部
    assert ((1 + I)/(1 - I)).as_real_imag() == (0, 1)

    # 计算 ((1 + I)**3/(1 - I)) 的实部和虚部
    assert ((1 + I)**3/(1 - I)).as_real_imag() == (-2, 0)


@XFAIL
def test_sign_issue_3068():
    n = pi**1000
    i = int(n)
    x = Symbol('x')
    # 检查 n - i 的结果四舍五入后是否等于 1
    assert (n - i).round() == 1  # 不会挂起
    # 检查 n - i 的符号是否为正
    assert sign(n - i) == 1
    # 当请求的位数为 1 时，可能无法正确获取符号
    # 当请求的位数为 2 时，可以正常工作
    assert (n - x).n(1, subs={x: i}) > 0
    assert (n - x).n(2, subs={x: i}) > 0


def test_Abs():
    raises(TypeError, lambda: Abs(Interval(2, 3)))  # 解决问题 8717

    x, y = symbols('x,y')
    # 检查 sign(sign(x)) 是否等于 sign(x)
    assert sign(sign(x)) == sign(x)
    # 检查 x*y 的函数是否为 sign
    assert sign(x*y).func is sign
    # 检查 Abs(0) 是否等于 0
    assert Abs(0) == 0
    # 检查 Abs(1) 是否等于 1
    assert Abs(1) == 1
    # 检查 Abs(-1) 是否等于 1
    assert Abs(-1) == 1
    # 检查 Abs(I) 是否等于 1
    assert Abs(I) == 1
    # 检查 Abs(-I) 是否等于 1
    assert Abs(-I) == 1
    # 检查 Abs(nan) 是否为 nan
    assert Abs(nan) is nan
    # 检查 Abs(zoo) 是否为 oo
    assert Abs(zoo) is oo
    # 检查 Abs(I * pi) 是否等于 pi
    assert Abs(I * pi) == pi
    # 检查 Abs(-I * pi) 是否等于 pi
    assert Abs(-I * pi) == pi
    # 检查 Abs(I * x) 是否等于 Abs(x)
    assert Abs(I * x) == Abs(x)
    # 检查 Abs(-I * x) 是否等于 Abs(x)
    assert Abs(-I * x) == Abs(x)
    # 检查 Abs(-2*x) 是否等于 2*Abs(x)
    assert Abs(-2*x) == 2*Abs(x)
    # 检查 Abs(-2.0*x) 是否等于 2.0*Abs(x)
    assert Abs(-2.0*x) == 2.0*Abs(x)
    # 检查 Abs(2*pi*x*y) 是否等于 2*pi*Abs(x*y)
    assert Abs(2*pi*x*y) == 2*pi*Abs(x*y)
    # 检查 Abs(conjugate(x)) 是否等于 Abs(x)
    assert Abs(conjugate(x)) == Abs(x)
    # 检查 conjugate(Abs(x)) 是否等于 Abs(x)
    assert conjugate(Abs(x)) == Abs(x)
    # 检查 Abs(x).expand(complex=True) 是否等于 sqrt(re(x)**2 + im(x)**2)
    assert Abs(x).expand(complex=True) == sqrt(re(x)**2 + im(x)**2)

    a = Symbol('a', positive=True)
    # 检查 Abs(2*pi*x*a) 是否等于 2*pi*a*Abs(x)
    assert Abs(2*pi*x*a) == 2*pi*a*Abs(x)
    # 检查 Abs(2*pi*I*x*a) 是否等于 2*pi*a*Abs(x)
    assert Abs(2*pi*I*x*a) == 2*pi*a*Abs(x)

    x = Symbol('x', real=True)
    n = Symbol('n', integer=True)
    # 检查 Abs((-1)**n) 是否等于 1
    assert Abs((-1)**n) == 1
    # 检查 x**(2*n) 是否等于 Abs(x)**(2*n)
    assert x**(2*n) == Abs(x)**(2*n)
    # 检查 Abs(x).diff(x) 是否等于 sign(x)
    assert Abs(x).diff(x) == sign(x)
    # 检查 abs(x) 是否等于 Abs(x)（Python 内置）
    assert abs(x) == Abs(x)
    # 检查 Abs(x)**3 是否等于 x**2*Abs(x)
    assert Abs(x)**3 == x**2*Abs(x)
    # 检查 Abs(x)**4 是否等于 x**4
    assert Abs(x)**4 == x**4
    # 检查 Abs(x)**(3*n) 的参数是否为 (Abs(x), 3*n)（保持符号奇数不变）
    assert (Abs(x)**(3*n)).args == (Abs(x), 3*n)
    # 检查 1/Abs(x) 的参数是否为 (Abs(x), -1)
    assert (1/Abs(x)).args == (Abs(x), -1)
    # 检查 1/Abs(x)**3 是否等于 1/(x**2*Abs(x))
    assert 1/Abs(x)**3 == 1/(x**2*Abs(x))
    # 检查 Abs(x)**-3 是否等于 Abs(x)/(x**4)
    assert Abs(x)**-3 == Abs(x)/(x**4)
    # 检查 Abs(x**3) 是否等于 x**2*Abs(x)
    assert Abs(x**3) == x**2*Abs(x)
    # 检查 Abs(I**I) 是否等于 exp(-pi/2)
    assert Abs(I**I) == exp(-pi/2)
    # 检查 Abs((4 + 5*I)**(6 + 7*I)) 是否等于 68921*exp(-7*atan(Rational(5, 4)))
    assert Abs((4 + 5*I)**(6 + 7*I)) == 68921*exp(-7*atan(Rational(5, 4)))
    y = Symbol('y', real=True)
    # 检查 Abs(I**y) 是否等于 1
    assert Abs(I**y) == 1
    y = Symbol('y')
    # 断言表达式验证绝对值和指数函数的关系
    assert Abs(I**y) == exp(-pi*im(y)/2)

    # 创建一个带有虚部的符号变量 x
    x = Symbol('x', imaginary=True)
    # 断言绝对值对 x 的导数等于其符号函数的负值
    assert Abs(x).diff(x) == -sign(x)

    # 定义一个复杂的表达式 eq
    eq = -sqrt(10 + 6*sqrt(3)) + sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3))
    # 对表达式的绝对值进行断言，验证其是否为绝对值函数或者是否为零
    # 附注：有时很难判断一个表达式是否为零，因此最好不要用慢速测试负担 abs 函数
    assert abs(eq).func is Abs or abs(eq) == 0

    # 定义两个表达式 q 和 p，并计算它们的差值 d
    q = 1 + sqrt(2) - 2*sqrt(3) + 1331*sqrt(6)
    p = expand(q**3)**Rational(1, 3)
    d = p - q
    # 对差值 d 的绝对值进行断言，验证其是否为绝对值函数或者是否为零
    assert abs(d).func is Abs or abs(d) == 0

    # 多个绝对值断言示例
    assert Abs(4*exp(pi*I/4)) == 4
    assert Abs(3**(2 + I)) == 9
    assert Abs((-3)**(1 - I)) == 3*exp(pi)

    # 对无穷大的绝对值进行断言
    assert Abs(oo) is oo
    assert Abs(-oo) is oo
    assert Abs(oo + I) is oo
    assert Abs(oo + I*oo) is oo

    # 创建三个符号变量 a, t, x，分别指定其代数属性和超越属性
    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    # 验证符号变量的实部是否为代数数
    assert re(a).is_algebraic
    # 验证符号变量的实部是否为代数数，但未知
    assert re(x).is_algebraic is None
    # 验证符号变量的实部是否为超越数
    assert re(t).is_algebraic is False
    # 对符号变量 x 的绝对值进行导数断言，验证其等于符号函数
    assert Abs(x).fdiff() == sign(x)
    # 断言绝对值函数的二阶导数引发参数索引错误
    raises(ArgumentIndexError, lambda: Abs(x).fdiff(2))

    # 不会引发递归错误的 abs 断言
    arg = sqrt(acos(1 - I)*acos(1 + I))
    assert abs(arg) == arg

    # 对分母中包含绝对值的特殊处理进行断言
    assert abs(1/x) == 1/Abs(x)
    # 验证表达式 e 是否为乘法表达式，并且等于特定形式的分式
    e = abs(2/x**2)
    assert e.is_Mul and e == 2/Abs(x**2)
    # 对 Abs 函数在特定表达式中不变性进行断言
    assert unchanged(Abs, y/x)
    assert unchanged(Abs, x/(x + 1))
    assert unchanged(Abs, x*y)
    # 对符号变量 x 除以正数 p 的绝对值进行断言
    p = Symbol('p', positive=True)
    assert abs(x/p) == abs(x)/p

    # 对不同类型的符号变量和函数进行不变性断言
    assert unchanged(Abs, Symbol('x', real=True)**y)
    # issue 19627 的问题断言
    f = Function('f', positive=True)
    assert sqrt(f(x)**2) == f(x)
    # issue 21625 的问题断言
    assert unchanged(Abs, S("im(acos(-i + acosh(-g + i)))"))
def test_Abs_rewrite():
    # 定义符号变量 x，并指定为实数
    x = Symbol('x', real=True)
    # 对 Abs(x) 应用 Heaviside 重写，并展开
    a = Abs(x).rewrite(Heaviside).expand()
    # 断言重写后的结果与预期相等
    assert a == x*Heaviside(x) - x*Heaviside(-x)
    # 遍历一些值，断言在这些值上的替换结果等于 abs(i)
    for i in [-2, -1, 0, 1, 2]:
        assert a.subs(x, i) == abs(i)
    
    # 定义符号变量 y
    y = Symbol('y')
    # 断言 Abs(y) 应用 Heaviside 重写的结果等于 Abs(y)
    assert Abs(y).rewrite(Heaviside) == Abs(y)

    # 定义符号变量 x, y，并分别指定 x 为实数
    x, y = Symbol('x', real=True), Symbol('y')
    # 断言 Abs(x) 应用 Piecewise 重写的结果等于 Piecewise((x, x >= 0), (-x, True))
    assert Abs(x).rewrite(Piecewise) == Piecewise((x, x >= 0), (-x, True))
    # 断言 Abs(y) 应用 Piecewise 重写的结果等于 Abs(y)
    assert Abs(y).rewrite(Piecewise) == Abs(y)
    # 断言 Abs(y) 应用 sign 重写的结果等于 y/sign(y)
    assert Abs(y).rewrite(sign) == y/sign(y)

    # 定义符号变量 i，并指定为虚数
    i = Symbol('i', imaginary=True)
    # 断言 abs(i) 应用 Piecewise 重写的结果等于 Piecewise((I*i, I*i >= 0), (-I*i, True))
    assert abs(i).rewrite(Piecewise) == Piecewise((I*i, I*i >= 0), (-I*i, True))

    # 断言 Abs(y) 应用 conjugate 重写的结果等于 sqrt(y*conjugate(y))
    assert Abs(y).rewrite(conjugate) == sqrt(y*conjugate(y))
    # 断言 Abs(i) 应用 conjugate 重写的结果等于 sqrt(-i**2)
    assert Abs(i).rewrite(conjugate) == sqrt(-i**2)  # == -I*i

    # 定义符号变量 y，并指定为扩展实数
    y = Symbol('y', extended_real=True)
    # 断言 Abs(exp(-I*x)-exp(-I*y))**2 应用 conjugate 重写的结果等于 -exp(I*x)*exp(-I*y) + 2 - exp(-I*x)*exp(I*y)
    assert (Abs(exp(-I*x)-exp(-I*y))**2).rewrite(conjugate) == \
        -exp(I*x)*exp(-I*y) + 2 - exp(-I*x)*exp(I*y)


def test_Abs_real():
    # 测试仅适用于实数的 abs 的一些性质
    x = Symbol('x', complex=True)
    assert sqrt(x**2) != Abs(x)
    assert Abs(x**2) != x**2

    # 定义符号变量 x，并指定为实数
    x = Symbol('x', real=True)
    # 断言 sqrt(x**2) 等于 Abs(x)
    assert sqrt(x**2) == Abs(x)
    # 断言 Abs(x**2) 等于 x**2
    assert Abs(x**2) == x**2

    # 如果符号为零，以下仍然适用
    nn = Symbol('nn', nonnegative=True, real=True)
    np = Symbol('np', nonpositive=True, real=True)
    # 断言 Abs(nn) 等于 nn
    assert Abs(nn) == nn
    # 断言 Abs(np) 等于 -np
    assert Abs(np) == -np


def test_Abs_properties():
    x = Symbol('x')
    # 断言 Abs(x) 的 is_real 属性为 None
    assert Abs(x).is_real is None
    # 断言 Abs(x) 的 is_extended_real 属性为 True
    assert Abs(x).is_extended_real is True
    # 断言 Abs(x) 的 is_rational 属性为 None
    assert Abs(x).is_rational is None
    # 断言 Abs(x) 的 is_positive 属性为 None
    assert Abs(x).is_positive is None
    # 断言 Abs(x) 的 is_nonnegative 属性为 None
    assert Abs(x).is_nonnegative is None
    # 断言 Abs(x) 的 is_extended_positive 属性为 None
    assert Abs(x).is_extended_positive is None
    # 断言 Abs(x) 的 is_extended_nonnegative 属性为 True
    assert Abs(x).is_extended_nonnegative is True

    f = Symbol('x', finite=True)
    # 断言 Abs(f) 的 is_real 属性为 True
    assert Abs(f).is_real is True
    # 断言 Abs(f) 的 is_extended_real 属性为 True
    assert Abs(f).is_extended_real is True
    # 断言 Abs(f) 的 is_rational 属性为 None
    assert Abs(f).is_rational is None
    # 断言 Abs(f) 的 is_positive 属性为 None
    assert Abs(f).is_positive is None
    # 断言 Abs(f) 的 is_nonnegative 属性为 True
    assert Abs(f).is_nonnegative is True
    # 断言 Abs(f) 的 is_extended_positive 属性为 None
    assert Abs(f).is_extended_positive is None
    # 断言 Abs(f) 的 is_extended_nonnegative 属性为 True
    assert Abs(f).is_extended_nonnegative is True

    z = Symbol('z', complex=True, zero=False)
    # 断言 Abs(z) 的 is_real 属性为 True（因为复数意味着有限）
    assert Abs(z).is_real is True
    # 断言 Abs(z) 的 is_extended_real 属性为 True
    assert Abs(z).is_extended_real is True
    # 断言 Abs(z) 的 is_rational 属性为 None
    assert Abs(z).is_rational is None
    # 断言 Abs(z) 的 is_positive 属性为 True
    assert Abs(z).is_positive is True
    # 断言 Abs(z) 的 is_extended_positive 属性为 True
    assert Abs(z).is_extended_positive is True
    # 断言 Abs(z) 的 is_zero 属性为 False
    assert Abs(z).is_zero is False

    p = Symbol('p', positive=True)
    # 断言 Abs(p) 的 is_real 属性为 True
    assert Abs(p).is_real is True
    # 断言 Abs(p) 的 is_extended_real 属性为 True
    assert Abs(p).is_extended_real is True
    # 断言 Abs(p) 的 is_rational 属性为 None
    assert Abs(p).is_rational is None
    # 断言 Abs(p) 的 is_positive 属性为 True
    assert Abs(p).is_positive is True
    # 断言 Abs(p) 的 is_zero 属性为 False
    assert Abs(p).is_zero is False

    q = Symbol('q', rational=True)
    # 断言 Abs(q) 的 is_real 属性为 True
    assert Abs(q).is_real is True
    # 断言 Abs(q) 的 is_rational 属性为 True
    assert Abs(q).is_rational is True
    # 断言 Abs(q) 的 is_integer 属性为 None
    assert Abs(q).is_integer is None
    # 断言 Abs(q) 的 is_positive 属性为 None
    assert Abs(q).is_positive is None
    # 断言 Abs(q) 的 is_nonnegative 属性为 True
    assert Abs(q).is_nonnegative is True

    i = Symbol('i', integer=True)
    # 断言 Abs(i) 的 is_real 属性为 True
    assert Abs(i).is_real is True
    # 断言 Abs(i) 的 is_integer 属性为 True
    assert Abs(i).is_integer is True
    # 断言 Abs(i) 的 is_positive 属性为 None
    assert Abs(i).is_positive is None
    # 断言 i 的绝对值是否非负
    assert Abs(i).is_nonnegative is True

    # 创建一个偶数符号 'e'
    e = Symbol('n', even=True)
    # 创建一个不是偶数的实数符号 'ne'
    ne = Symbol('ne', real=True, even=False)
    # 断言 'e' 的绝对值是否是偶数
    assert Abs(e).is_even is True
    # 断言 'ne' 的绝对值是否是偶数
    assert Abs(ne).is_even is False
    # 断言 i 的绝对值是否是偶数
    assert Abs(i).is_even is None

    # 创建一个奇数符号 'o'
    o = Symbol('n', odd=True)
    # 创建一个不是奇数的实数符号 'no'
    no = Symbol('no', real=True, odd=False)
    # 断言 'o' 的绝对值是否是奇数
    assert Abs(o).is_odd is True
    # 断言 'no' 的绝对值是否是奇数
    assert Abs(no).is_odd is False
    # 断言 i 的绝对值是否是奇数
    assert Abs(i).is_odd is None
def test_abs():
    # 测试 abs 函数是否调用 Abs 函数；不要将其重命名为 test_Abs，因为上面已经有这个测试
    a = Symbol('a', positive=True)
    # 断言绝对值操作对复数值的计算是否正确
    assert abs(I*(1 + a)**2) == (1 + a)**2


def test_arg():
    # 断言对于 arg 函数的多个输入值，返回的辐角是否正确
    assert arg(0) is nan
    assert arg(1) == 0
    assert arg(-1) == pi
    assert arg(I) == pi/2
    assert arg(-I) == -pi/2
    assert arg(1 + I) == pi/4
    assert arg(-1 + I) == pi*Rational(3, 4)
    assert arg(1 - I) == -pi/4
    assert arg(exp_polar(4*pi*I)) == 4*pi
    assert arg(exp_polar(-7*pi*I)) == -7*pi
    assert arg(exp_polar(5 - 3*pi*I/4)) == pi*Rational(-3, 4)

    assert arg(exp(I*pi/7)) == pi/7     # issue 17300
    assert arg(exp(16*I)) == 16 - 6*pi
    assert arg(exp(13*I*pi/12)) == -11*pi/12
    assert arg(exp(123 - 5*I)) == -5 + 2*pi
    assert arg(exp(sin(1 + 3*I))) == -2*pi + cos(1)*sinh(3)
    r = Symbol('r', real=True)
    assert arg(exp(r - 2*I)) == -2

    f = Function('f')
    assert not arg(f(0) + I*f(1)).atoms(re)

    # 检查嵌套调用
    x = Symbol('x')
    assert arg(arg(arg(x))) is not S.NaN
    assert arg(arg(arg(arg(x)))) is S.NaN
    r = Symbol('r', extended_real=True)
    assert arg(arg(r)) is not S.NaN
    assert arg(arg(arg(r))) is S.NaN

    p = Function('p', extended_positive=True)
    assert arg(p(x)) == 0
    assert arg((3 + I)*p(x)) == arg(3  + I)

    p = Symbol('p', positive=True)
    assert arg(p) == 0
    assert arg(p*I) == pi/2

    n = Symbol('n', negative=True)
    assert arg(n) == pi
    assert arg(n*I) == -pi/2

    x = Symbol('x')
    assert conjugate(arg(x)) == arg(x)

    e = p + I*p**2
    assert arg(e) == arg(1 + p*I)
    # 确保符号不会交换
    e = -2*p + 4*I*p**2
    assert arg(e) == arg(-1 + 2*p*I)
    # 确保符号不会丢失
    x = symbols('x', real=True)  # 可能为零
    e = x + I*x
    assert arg(e) == arg(x*(1 + I))
    assert arg(e/p) == arg(x*(1 + I))
    e = p*cos(p) + I*log(p)*exp(p)
    assert arg(e).args[0] == e
    # 保持简单 -- 让用户进行更高级的消除操作
    e = (p + 1) + I*(p**2 - 1)
    assert arg(e).args[0] == e

    f = Function('f')
    e = 2*x*(f(0) - 1) - 2*x*f(0)
    assert arg(e) == arg(-2*x)
    assert arg(f(0)).func == arg and arg(f(0)).args == (f(0),)


def test_arg_rewrite():
    # 断言对于 arg 函数的重写，是否与 atan2 函数的计算结果相同
    assert arg(1 + I) == atan2(1, 1)

    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert arg(x + I*y).rewrite(atan2) == atan2(y, x)


def test_adjoint():
    a = Symbol('a', antihermitian=True)
    b = Symbol('b', hermitian=True)
    # 断言共轭转置操作对于反自共轭矩阵的计算是否正确
    assert adjoint(a) == -a
    assert adjoint(I*a) == I*a
    assert adjoint(b) == b
    assert adjoint(I*b) == -I*b
    assert adjoint(a*b) == -b*a
    assert adjoint(I*a*b) == I*b*a

    x, y = symbols('x y')
    assert adjoint(adjoint(x)) == x
    assert adjoint(x + y) == adjoint(x) + adjoint(y)
    assert adjoint(x - y) == adjoint(x) - adjoint(y)
    assert adjoint(x * y) == adjoint(x) * adjoint(y)
    assert adjoint(x / y) == adjoint(x) / adjoint(y)
    # 断言对称伴随算子对负数的应用应该返回其相反数的伴随算子
    assert adjoint(-x) == -adjoint(x)
    
    # 创建两个非可交换符号变量 x 和 y
    x, y = symbols('x y', commutative=False)
    
    # 断言对称伴随算子两次作用于变量 x 应该返回原变量 x
    assert adjoint(adjoint(x)) == x
    
    # 断言对称伴随算子应用于两个变量 x 和 y 的和应该等于分别对它们应用对称伴随算子后的和
    assert adjoint(x + y) == adjoint(x) + adjoint(y)
    
    # 断言对称伴随算子应用于两个变量 x 和 y 的差应该等于分别对它们应用对称伴随算子后的差
    assert adjoint(x - y) == adjoint(x) - adjoint(y)
    
    # 断言对称伴随算子应用于两个变量 x 和 y 的乘积应该等于它们对称伴随算子的乘积的反序
    assert adjoint(x * y) == adjoint(y) * adjoint(x)
    
    # 断言对称伴随算子应用于两个变量 x 和 y 的除法应该等于分别对它们应用对称伴随算子后的商
    assert adjoint(x / y) == 1 / adjoint(y) * adjoint(x)
    
    # 断言对称伴随算子对负数的应用应该返回其相反数的伴随算子
    assert adjoint(-x) == -adjoint(x)
# 定义一个测试函数，用于测试复数和矩阵操作的共轭（conjugate）函数
def test_conjugate():
    # 创建一个实数符号 a 和一个虚数符号 b
    a = Symbol('a', real=True)
    b = Symbol('b', imaginary=True)
    # 检验共轭函数对实数符号的作用，应返回其本身
    assert conjugate(a) == a
    # 检验共轭函数对纯虚数乘以虚数单位的作用，应返回相反数
    assert conjugate(I*a) == -I*a
    # 检验共轭函数对虚数的作用，应返回其相反数
    assert conjugate(b) == -b
    # 检验共轭函数对纯虚数乘以虚数单位的作用，应返回虚数单位乘以自身
    assert conjugate(I*b) == I*b
    # 检验共轭函数对实数乘以虚数的乘积的作用，应返回其相反数
    assert conjugate(a*b) == -a*b
    # 检验共轭函数对纯虚数乘以实数和虚数单位的乘积的作用，应返回虚数单位乘以乘积的自身
    assert conjugate(I*a*b) == I*a*b

    # 创建两个符号变量 x 和 y
    x, y = symbols('x y')
    # 检验共轭函数对共轭函数的作用，应返回其自身
    assert conjugate(conjugate(x)) == x
    # 检验共轭函数的逆函数的作用，应返回共轭函数
    assert conjugate(x).inverse() == conjugate
    # 检验共轭函数对加法操作的作用，应返回加法操作的共轭
    assert conjugate(x + y) == conjugate(x) + conjugate(y)
    # 检验共轭函数对减法操作的作用，应返回减法操作的共轭
    assert conjugate(x - y) == conjugate(x) - conjugate(y)
    # 检验共轭函数对乘法操作的作用，应返回乘法操作的共轭
    assert conjugate(x * y) == conjugate(x) * conjugate(y)
    # 检验共轭函数对除法操作的作用，应返回除法操作的共轭
    assert conjugate(x / y) == conjugate(x) / conjugate(y)
    # 检验共轭函数对负数的作用，应返回其相反数的共轭
    assert conjugate(-x) == -conjugate(x)

    # 创建一个具有代数特性的符号变量 a 和一个具有超越特性的符号变量 t
    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    # 检验实部函数对代数符号的作用，应表明其为代数类型
    assert re(a).is_algebraic
    # 检验实部函数对普通符号的作用，应表明其不是代数类型
    assert re(x).is_algebraic is None
    # 检验实部函数对超越符号的作用，应表明其不是代数类型
    assert re(t).is_algebraic is False


# 定义一个测试函数，用于测试转置（transpose）和共轭转置（adjoint）函数的操作
def test_conjugate_transpose():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 检验共轭转置函数对转置函数的作用，应返回共轭转置函数
    assert conjugate(transpose(x)) == adjoint(x)
    # 检验转置函数对共轭函数的作用，应返回共轭转置函数
    assert transpose(conjugate(x)) == adjoint(x)
    # 检验共轭转置函数对转置函数的作用，应返回共轭函数
    assert adjoint(transpose(x)) == conjugate(x)
    # 检验转置函数对共轭转置函数的作用，应返回共轭函数
    assert transpose(adjoint(x)) == conjugate(x)
    # 检验共轭函数对共轭转置函数的作用，应返回转置函数
    assert adjoint(conjugate(x)) == transpose(x)
    # 检验共轭函数对转置共轭函数的作用，应返回转置函数
    assert conjugate(adjoint(x)) == transpose(x)

    # 创建一个自定义类 Symmetric，继承自 Expr 类
    class Symmetric(Expr):
        # 定义 _eval_adjoint 方法，返回 None
        def _eval_adjoint(self):
            return None

        # 定义 _eval_conjugate 方法，返回 None
        def _eval_conjugate(self):
            return None

        # 定义 _eval_transpose 方法，返回自身
        def _eval_transpose(self):
            return self

    # 创建一个 Symmetric 类的实例 x
    x = Symmetric()
    # 检验共轭函数对 Symmetric 类的实例的作用，应返回其共轭转置
    assert conjugate(x) == adjoint(x)
    # 检验转置函数对 Symmetric 类的实例的作用，应返回其自身
    assert transpose(x) == x


# 定义一个测试函数，用于测试转置（transpose）函数的操作
def test_transpose():
    # 创建一个复数类型的符号变量 a
    a = Symbol('a', complex=True)
    # 检验转置函数对复数符号的作用，应返回其自身
    assert transpose(a) == a
    # 检验转置函数对虚数单位乘以复数符号的作用，应返回虚数单位乘以复数符号
    assert transpose(I*a) == I*a

    # 创建两个符号变量 x 和 y
    x, y = symbols('x y')
    # 检验转置函数对转置函数的作用，应返回其自身
    assert transpose(transpose(x)) == x
    # 检验转置函数对加法操作的作用，应返回加法操作的转置
    assert transpose(x + y) == transpose(x) + transpose(y)
    # 检验转置函数对减法操作的作用，应返回减法操作的转置
    assert transpose(x - y) == transpose(x) - transpose(y)
    # 检验转置函数对乘法操作的作用，应返回乘法操作的转置
    assert transpose(x * y) == transpose(x) * transpose(y)
    # 检验转置函数对除法操作的作用，应返回除法操作的转置
    assert transpose(x / y) == transpose(x) / transpose(y)
    # 检验转置函数对负数的作用，应返回其负数的转置
    assert transpose(-x) == -transpose(x)

    # 创建两个符号变量 x 和 y，指定其为非交换类型
    x, y = symbols('x y', commutative=False)
    # 检验转置函数对转置函数的作用，应返回其自身
    assert transpose(transpose(x)) == x
    # 检验转置函数对非交换类型的加法操作的作用，应返回加法操作的转置
    assert transpose(x + y) == transpose(x) + transpose(y)
    # 检验转置函数对非交换类型的减法操作的作用，应返回减法操作的转置
    assert transpose(x - y) == transpose(x) - transpose(y)
    # 检验转置函数对非交换类型的乘法操作的作用，应返回乘法操作的转置
    assert transpose(x * y) == transpose(y) * transpose(x)
    # 检验转置函数对非交换类型的除法操作的作用，应返回除法操作的转置
    assert transpose(x / y) == 1 / transpose(y) * transpose(x)
    # 检验转置函数对非交换类型的负数的作用，应返回其负数的转置
    assert transpose(-x) == -transpose(x)


# 定
    # 断言：确保 polarify 函数对表达式 1 + x 进行极化，lift=True 时结果与 polar_lift(1 + x) 相等
    assert polarify(1 + x, lift=True) == polar_lift(1 + x)

    # 断言：确保 polarify 函数对表达式 1 + f(x) 进行极化，lift=True 时结果与 polar_lift(1 + f(polar_lift(x))) 相等
    assert polarify(1 + f(x), lift=True) == polar_lift(1 + f(polar_lift(x)))

    # 对表达式 f(x) + z 进行极化，返回极化后的表达式和替换字典
    newex, subs = polarify(f(x) + z)
    # 断言：确保用替换字典 subs 替换 newex 后结果与 f(x) + z 相等
    assert newex.subs(subs) == f(x) + z

    # 创建符号 mu 和 sigma，其中 sigma 被指定为正数
    mu = Symbol("mu")
    sigma = Symbol("sigma", positive=True)

    # 确保 polarify(lift=True) 不会尝试对积分变量进行极化
    assert polarify(
        Integral(sqrt(2)*x*exp(-(-mu + x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma),
        (x, -oo, oo)), lift=True) == Integral(sqrt(2)*(sigma*exp_polar(0))**exp_polar(I*pi)*
        exp((sigma*exp_polar(0))**(2*exp_polar(I*pi))*exp_polar(I*pi)*polar_lift(-mu + x)**
        (2*exp_polar(0))/2)*exp_polar(0)*polar_lift(x)/(2*sqrt(pi)), (x, -oo, oo))
# 定义函数 test_unpolarify，用于测试 sympy 库中的 unpolarify 函数
def test_unpolarify():
    # 导入必要的 sympy 模块和函数
    from sympy.functions.elementary.complexes import (polar_lift, principal_branch, unpolarify)
    from sympy.core.relational import Ne
    from sympy.functions.elementary.hyperbolic import tanh
    from sympy.functions.special.error_functions import erf
    from sympy.functions.special.gamma_functions import (gamma, uppergamma)
    from sympy.abc import x

    # 定义变量 p 和 u，分别为带极坐标和不带极坐标的表达式
    p = exp_polar(7*I) + 1
    u = exp(7*I) + 1

    # 断言语句，测试 unpolarify 函数的各种用例

    # 基本用例
    assert unpolarify(1) == 1
    assert unpolarify(p) == u
    assert unpolarify(p**2) == u**2
    assert unpolarify(p**x) == p**x
    assert unpolarify(p*x) == u*x
    assert unpolarify(p + x) == u + x
    assert unpolarify(sqrt(sin(p))) == sqrt(sin(u))

    # 测试归约到主分支 2*pi
    t = principal_branch(x, 2*pi)
    assert unpolarify(t) == x
    assert unpolarify(sqrt(t)) == sqrt(t)

    # 测试仅处理指数的情况
    assert unpolarify(p**p, exponents_only=True) == p**u
    assert unpolarify(uppergamma(x, p**p)) == uppergamma(x, p**u)

    # 测试函数的处理
    assert unpolarify(sin(p)) == sin(u)
    assert unpolarify(tanh(p)) == tanh(u)
    assert unpolarify(gamma(p)) == gamma(u)
    assert unpolarify(erf(p)) == erf(u)
    assert unpolarify(uppergamma(x, p)) == uppergamma(x, p)

    # 复杂表达式的测试
    assert unpolarify(uppergamma(sin(p), sin(p + exp_polar(0)))) == \
        uppergamma(sin(u), sin(u + 1))
    assert unpolarify(uppergamma(polar_lift(0), 2*exp_polar(0))) == \
        uppergamma(0, 2)

    # 测试等式和不等式
    assert unpolarify(Eq(p, 0)) == Eq(u, 0)
    assert unpolarify(Ne(p, 0)) == Ne(u, 0)
    assert unpolarify(polar_lift(x) > 0) == (x > 0)

    # 测试布尔值
    assert unpolarify(True) is True


# 定义函数 test_issue_4035，测试 sympy 库中的一个问题
def test_issue_4035():
    from sympy import Symbol
    from sympy.functions.elementary.trigonometric import Abs, sign, arg
    x = Symbol('x')

    # 断言语句，测试绝对值、符号和幅角的展开
    assert Abs(x).expand(trig=True) == Abs(x)
    assert sign(x).expand(trig=True) == sign(x)
    assert arg(x).expand(trig=True) == arg(x)


# 定义函数 test_issue_3206，测试 sympy 库中的一个问题
def test_issue_3206():
    from sympy import Symbol, Abs
    x = Symbol('x')

    # 断言语句，测试绝对值函数的双重应用
    assert Abs(Abs(x)) == Abs(x)


# 定义函数 test_issue_4754_derivative_conjugate，测试 sympy 库中的一个问题
def test_issue_4754_derivative_conjugate():
    from sympy import Symbol, Function, I
    from sympy.core.relational import Eq, Ne
    from sympy.matrices import Matrix
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    f = Function('f')

    # 断言语句，测试导数和共轭的关系
    assert (f(x).conjugate()).diff(x) == (f(x).diff(x)).conjugate()
    assert (f(y).conjugate()).diff(y) == -(f(y).diff(y)).conjugate()


# 定义函数 test_derivatives_issue_4757，测试 sympy 库中的一个问题
def test_derivatives_issue_4757():
    from sympy import Symbol, Function, I, sqrt
    from sympy.core.relational import Eq
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    f = Function('f')

    # 断言语句，测试实部、虚部和绝对值的导数
    assert re(f(x)).diff(x) == re(f(x).diff(x))
    assert im(f(x)).diff(x) == im(f(x).diff(x))
    assert re(f(y)).diff(y) == -I*im(f(y).diff(y))
    assert im(f(y)).diff(y) == -I*re(f(y).diff(y))
    assert Abs(f(x)).diff(x).subs(f(x), 1 + I*x).doit() == x/sqrt(1 + x**2)
    assert arg(f(x)).diff(x).subs(f(x), 1 + I*x**2).doit() == 2*x/(1 + x**4)
    assert Abs(f(y)).diff(y).subs(f(y), 1 + y).doit() == -y/sqrt(1 - y**2)
    assert arg(f(y)).diff(y).subs(f(y), I + y**2).doit() == 2*y/(1 + y**4)


# 定义函数 test_issue_11413，测试 sympy 库中的一个问题
def test_issue_11413():
    from sympy import Symbol, Function, Matrix
    v0 = Symbol('v0')
    v1 = Symbol('v1')
    v2 = Symbol('v2')
    V = Matrix([[v0],[v1],[v2]])

    # 该函数用于测试一个矩阵 V 的创建和处理
    # 对向量进行单位化，使其成为单位向量
    U = V.normalized()
    
    # 断言向量U应该等于由v0, v1, v2组成的列向量的单位向量
    assert U == Matrix([
        [v0/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)],
        [v1/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)],
        [v2/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)]
    ])
    
    # 计算向量U的范数，应该得到1
    U.norm = sqrt(v0**2/(v0**2 + v1**2 + v2**2) + v1**2/(v0**2 + v1**2 + v2**2) + v2**2/(v0**2 + v1**2 + v2**2))
    assert simplify(U.norm) == 1
# 定义测试函数 test_periodic_argument，用于测试周期性参数函数
def test_periodic_argument():
    # 从 sympy 库导入相关函数和符号
    from sympy.functions.elementary.complexes import (periodic_argument, polar_lift, principal_branch, unbranched_argument)
    # 定义符号变量 x 和 p，其中 p 为正数
    x = Symbol('x')
    p = Symbol('p', positive=True)

    # 断言：未分支参数函数在复数 2 + I 上的值等于给定周期参数的值
    assert unbranched_argument(2 + I) == periodic_argument(2 + I, oo)
    # 断言：未分支参数函数在表达式 1 + x 上的值等于给定周期参数的值
    assert unbranched_argument(1 + x) == periodic_argument(1 + x, oo)
    # 断言：未分支参数函数在 (1 + I)^2 复数的值约等于 π/2
    assert N_equals(unbranched_argument((1 + I)**2), pi/2)
    # 断言：未分支参数函数在 (1 - I)^2 复数的值约等于 -π/2
    assert N_equals(unbranched_argument((1 - I)**2), -pi/2)
    # 断言：给定周期参数函数在 (1 + I)^2 复数的值约等于 π/2
    assert N_equals(periodic_argument((1 + I)**2, 3*pi), pi/2)
    # 断言：给定周期参数函数在 (1 - I)^2 复数的值约等于 -π/2
    assert N_equals(periodic_argument((1 - I)**2, 3*pi), -pi/2)

    # 断言：未分支参数函数在 principal_branch(x, π) 中的值等于给定周期参数的值
    assert unbranched_argument(principal_branch(x, pi)) == \
        periodic_argument(x, pi)

    # 断言：未分支参数函数在 polar_lift(2 + I) 的值等于未分支参数函数在 2 + I 的值
    assert unbranched_argument(polar_lift(2 + I)) == unbranched_argument(2 + I)
    # 断言：给定周期参数函数在 polar_lift(2 + I) 中的值等于给定周期参数函数在 2 + I 中的值
    assert periodic_argument(polar_lift(2 + I), 2*pi) == \
        periodic_argument(2 + I, 2*pi)
    # 断言：给定周期参数函数在 polar_lift(2 + I) 中的值等于给定周期参数函数在 2 + I 中的值
    assert periodic_argument(polar_lift(2 + I), 3*pi) == \
        periodic_argument(2 + I, 3*pi)
    # 断言：给定周期参数函数在 polar_lift(2 + I) 中的值等于给定周期参数函数在 2 + I 中的值
    assert periodic_argument(polar_lift(2 + I), pi) == \
        periodic_argument(polar_lift(2 + I), pi)

    # 断言：未分支参数函数在 polar_lift(1 + I) 中的值约等于 π/4
    assert unbranched_argument(polar_lift(1 + I)) == pi/4
    # 断言：给定周期参数函数在 2*p 中的值等于给定周期参数函数在 p 中的值
    assert periodic_argument(2*p, p) == periodic_argument(p, p)
    # 断言：给定周期参数函数在 π*p 中的值等于给定周期参数函数在 p 中的值
    assert periodic_argument(pi*p, p) == periodic_argument(p, p)

    # 断言：极坐标提升 polar_lift(1 + I) 的绝对值等于 1 + I 的绝对值
    assert Abs(polar_lift(1 + I)) == Abs(1 + I)


# 定义测试函数 test_principal_branch，用于测试主分支函数
def test_principal_branch():
    # 从 sympy 库导入相关函数和符号
    from sympy.functions.elementary.complexes import (polar_lift, principal_branch)
    # 定义符号变量 p 和 x
    p = Symbol('p', positive=True)
    x = Symbol('x')
    # 定义负数符号变量 neg
    neg = Symbol('x', negative=True)

    # 断言：主分支函数在 polar_lift(x) 中的值等于主分支函数在 x 中的值
    assert principal_branch(polar_lift(x), p) == principal_branch(x, p)
    # 断言：主分支函数在 polar_lift(2 + I) 中的值等于主分支函数在 2 + I 中的值
    assert principal_branch(polar_lift(2 + I), p) == principal_branch(2 + I, p)
    # 断言：主分支函数在 2*x 中的值等于 2 * 主分支函数在 x 中的值
    assert principal_branch(2*x, p) == 2*principal_branch(x, p)
    # 断言：主分支函数在 1 中的值等于 exp_polar(0)
    assert principal_branch(1, pi) == exp_polar(0)
    # 断言：主分支函数在 -1 中的值等于 exp_polar(I * π)
    assert principal_branch(-1, 2*pi) == exp_polar(I*pi)
    # 断言：主分支函数在 -1 中的值等于 exp_polar(0)
    assert principal_branch(-1, pi) == exp_polar(0)
    # 断言：主分支函数在 exp_polar(3 * I * π) * x 中的值等于主分支函数在 exp_polar(I * π) * x 中的值
    assert principal_branch(exp_polar(3*pi*I)*x, 2*pi) == \
        principal_branch(exp_polar(I*pi)*x, 2*pi)
    # 断言：主分支函数在 neg * exp_polar(π * I) 中的值等于 neg * exp_polar(-I * π)
    assert principal_branch(neg*exp_polar(pi*I), 2*pi) == neg*exp_polar(-I*pi)
    # 断言：与问题 #14692 相关
    assert principal_branch(exp_polar(-I*pi/2)/polar_lift(neg), 2*pi) == \
        exp_polar(-I*pi/2)/neg

    # 断言：主分支函数在 (1 + I)^2 中的值约等于 2 * I
    assert N_equals(principal_branch((1 + I)**2, 2*pi), 2*I)
    # 断言：主分支函数在 (1 + I)^2 中的值约等于 2 * I
    assert N_equals(principal_branch((1 + I)**2, 3*pi), 2*I)
    # 断言：主分支函数在 (1 + I)^2 中的值约等于 2 * I
    assert N_equals(principal_branch((1 + I)**2, 1*pi), 2*I)

    # 测试参数的卫生问题
    # 断言：主分支函数在 x 和 I 之间的值是主分支函数
    assert principal_branch(x, I).func is principal_branch
    # 断言：主分支函数在 x 和 -4 之间的值是主分支函数
    assert principal_branch(x, -4).func is principal_branch
    # 断言：主分支函数在 x 和 -oo 之间的值是主分支函数
    assert principal_branch(x, -oo).func is principal_branch
    # 断言：主分支函数在 x 和 zoo 之间的值是主分支函数
    assert principal_branch(x, zoo).func is principal_branch
    # 断言：验证 n - i 的符号为正
    assert sign(n - i) == 1
    # 断言：验证 n - i 的绝对值等于 n - i 自身
    assert abs(n - i) == n - i
    # 创建符号变量 x
    x = Symbol('x')
    # 计算一个极小的值 eps，等于 pi 的负1500次方
    eps = pi**-1500
    # 计算一个极大的值 big，等于 pi 的1000次方
    big = pi**1000
    # 计算 one，等于 cos(x)**2 + sin(x)**2，这个值应当恒等于 1
    one = cos(x)**2 + sin(x)**2
    # 计算 e，使用上面计算的 big 和 one，加上 eps
    e = big*one - big + eps
    # 导入 sympy 库中的 simplify 函数
    from sympy.simplify.simplify import simplify
    # 断言：验证简化后的 e 的符号为正
    assert sign(simplify(e)) == 1
    # 遍历 xi 中的每个值：111, 11, 1, 1/10，验证 e 在这些值下的符号为正
    for xi in (111, 11, 1, Rational(1, 10)):
        assert sign(e.subs(x, xi)) == 1
# 定义测试函数，检验问题编号 14216
def test_issue_14216():
    # 从 sympy 库中导入复数函数 unpolarify
    from sympy.functions.elementary.complexes import unpolarify
    # 创建一个 2x2 的矩阵符号 A
    A = MatrixSymbol("A", 2, 2)
    # 断言：对矩阵 A 的第 (0, 0) 元素应用 unpolarify 函数后结果应与 A[0, 0] 相等
    assert unpolarify(A[0, 0]) == A[0, 0]
    # 断言：对矩阵 A 的第 (0, 0) 元素乘以第 (1, 0) 元素应用 unpolarify 函数后结果应与 A[0, 0]*A[1, 0] 相等
    assert unpolarify(A[0, 0]*A[1, 0]) == A[0, 0]*A[1, 0]


# 定义测试函数，检验问题编号 14238
def test_issue_14238():
    # 符号 r，被声明为实数
    r = Symbol('r', real=True)
    # 断言：当 r 大于 0 时，Abs(r + Piecewise((0, r > 0), (1 - r, True))) 的值为 0
    assert Abs(r + Piecewise((0, r > 0), (1 - r, True)))


# 定义测试函数，检验问题编号 22189
def test_issue_22189():
    # 符号 x
    x = Symbol('x')
    # 对于迭代中的每个 a
    for a in (sqrt(7 - 2*x) - 2, 1 - x):
        # 断言：Abs(a) - Abs(-a) 的值应为 0，同时打印当前的 a 值
        assert Abs(a) - Abs(-a) == 0, a


# 定义测试函数，检验零假设
def test_zero_assumptions():
    # 符号 nr，被声明为非实数但有限
    nr = Symbol('nonreal', real=False, finite=True)
    # 符号 ni，被声明为非虚数
    ni = Symbol('nonimaginary', imaginary=False)
    # 符号 nzni，被声明为非零且非虚数
    nzni = Symbol('nonzerononimaginary', zero=False, imaginary=False)

    # 断言：nr 的实部是否为零应为未知
    assert re(nr).is_zero is None
    # 断言：nr 的虚部是否为零应为 False
    assert im(nr).is_zero is False

    # 断言：ni 的实部是否为零应为未知
    assert re(ni).is_zero is None
    # 断言：ni 的虚部是否为零应为未知
    assert im(ni).is_zero is None

    # 断言：nzni 的实部是否为零应为 False
    assert re(nzni).is_zero is False
    # 断言：nzni 的虚部是否为零应为未知
    assert im(nzni).is_zero is None


# 使用修饰符，定义测试函数，检验问题编号 15893
@_both_exp_pow
def test_issue_15893():
    # 函数 f 被声明为实数
    f = Function('f', real=True)
    # 符号 x 被声明为实数
    x = Symbol('x', real=True)
    # 构建方程 Abs(f(x)) 对 f(x) 的导数
    eq = Derivative(Abs(f(x)), f(x))
    # 断言：对方程进行求值应等于 f(x) 的符号函数值
    assert eq.doit() == sign(f(x))
```