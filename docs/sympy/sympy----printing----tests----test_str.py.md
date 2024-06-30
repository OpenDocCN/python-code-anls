# `D:\src\scipysrc\sympy\sympy\printing\tests\test_str.py`

```
from sympy import MatAdd  # 导入 MatAdd 类
from sympy.algebras.quaternion import Quaternion  # 导入 Quaternion 类
from sympy.assumptions.ask import Q  # 导入 Q 对象，用于符号假设
from sympy.calculus.accumulationbounds import AccumBounds  # 导入 AccumBounds 类，积分累积边界
from sympy.combinatorics.partitions import Partition  # 导入 Partition 类，数学分区对象
from sympy.concrete.summations import (Sum, summation)  # 导入 Sum 和 summation 函数，用于求和
from sympy.core.add import Add  # 导入 Add 类
from sympy.core.containers import (Dict, Tuple)  # 导入 Dict 和 Tuple 类，容器类型
from sympy.core.expr import UnevaluatedExpr, Expr  # 导入 UnevaluatedExpr 和 Expr 类
from sympy.core.function import (Derivative, Function, Lambda, Subs, WildFunction)  # 导入不同的函数类
from sympy.core.mul import Mul  # 导入 Mul 类，乘法表达式
from sympy.core import (Catalan, EulerGamma, GoldenRatio, TribonacciConstant)  # 导入常数如 Catalan、EulerGamma
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi, zoo)  # 导入各种数学常数
from sympy.core.parameters import _exp_is_pow  # 导入 _exp_is_pow 参数
from sympy.core.power import Pow  # 导入 Pow 类，幂函数
from sympy.core.relational import (Eq, Rel, Ne)  # 导入关系运算符类
from sympy.core.singleton import S  # 导入 S 单例对象
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)  # 导入符号类和创建符号的函数
from sympy.functions.combinatorial.factorials import (factorial, factorial2, subfactorial)  # 导入阶乘相关函数
from sympy.functions.elementary.complexes import Abs  # 导入绝对值函数 Abs
from sympy.functions.elementary.exponential import exp  # 导入指数函数 exp
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数 sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入三角函数 cos 和 sin
from sympy.functions.special.delta_functions import Heaviside  # 导入 Heaviside 阶跃函数
from sympy.functions.special.zeta_functions import zeta  # 导入 Riemann zeta 函数
from sympy.integrals.integrals import Integral  # 导入积分函数 Integral
from sympy.logic.boolalg import (Equivalent, false, true, Xor)  # 导入逻辑运算函数
from sympy.matrices.dense import Matrix  # 导入密集矩阵类 Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号表达式类 MatrixSymbol
from sympy.matrices.expressions import Identity  # 导入单位矩阵 Identity 类
from sympy.matrices.expressions.slice import MatrixSlice  # 导入矩阵切片类 MatrixSlice
from sympy.matrices import SparseMatrix  # 导入稀疏矩阵类 SparseMatrix
from sympy.polys.polytools import factor  # 导入因式分解函数 factor
from sympy.series.limits import Limit  # 导入极限函数 Limit
from sympy.series.order import O  # 导入 O 符号，用于渐近展开
from sympy.sets.sets import (Complement, FiniteSet, Interval, SymmetricDifference)  # 导入集合操作类
from sympy.stats import (Covariance, Expectation, Probability, Variance)  # 导入统计相关类
from sympy.stats.rv import RandomSymbol  # 导入随机变量符号类
from sympy.external import import_module  # 导入外部模块导入函数
from sympy.physics.control.lti import (TransferFunction, Series, Parallel,  # 导入控制系统函数和类
    Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback)
from sympy.physics.units import second, joule  # 导入物理单位
from sympy.polys import (Poly, rootof, RootSum, groebner, ring, field, ZZ, QQ,  # 导入多项式相关类和函数
    ZZ_I, QQ_I, lex, grlex)
from sympy.geometry import Point, Circle, Polygon, Ellipse, Triangle  # 导入几何图形类
from sympy.tensor import NDimArray  # 导入多维数组类
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement  # 导入数组表达式类

from sympy.testing.pytest import raises, warns_deprecated_sympy  # 导入测试框架相关函数

from sympy.printing import sstr, sstrrepr, StrPrinter  # 导入打印函数和打印相关类
from sympy.physics.quantum.trace import Tr  # 导入量子跟踪函数 Tr

x, y, z, w, t = symbols('x,y,z,w,t')  # 创建符号变量 x, y, z, w, t
d = Dummy('d')  # 创建虚拟符号 d

def test_printmethod():  # 定义测试函数 test_printmethod
    class R(Abs):  # 定义 R 类，继承自 Abs 类
        def _sympystr(self, printer):  # 定义 _sympystr 方法，用于自定义打印格式
            return "foo(%s)" % printer._print(self.args[0])  # 返回格式化的字符串
    assert sstr(R(x)) == "foo(x)"  # 断言 R(x) 的打印字符串为 "foo(x)"
    # 定义一个名为 R 的类，该类继承自 Abs 类
    class R(Abs):
        # 重写 _sympystr 方法，接受一个打印机对象 printer 作为参数，返回字符串 "foo"
        def _sympystr(self, printer):
            return "foo"
    
    # 使用断言来验证 sstr(R(x)) 的输出是否等于字符串 "foo"
    assert sstr(R(x)) == "foo"
# 测试 Abs 函数
def test_Abs():
    # 断言 Abs(x) 的字符串表示为 "Abs(x)"
    assert str(Abs(x)) == "Abs(x)"
    # 断言 Abs(1/6) 的字符串表示为 "1/6"
    assert str(Abs(Rational(1, 6))) == "1/6"
    # 断言 Abs(-1/6) 的字符串表示为 "1/6"
    assert str(Abs(Rational(-1, 6))) == "1/6"


# 测试 Add 函数
def test_Add():
    # 断言 x + y 的字符串表示为 "x + y"
    assert str(x + y) == "x + y"
    # 断言 x + 1 的字符串表示为 "x + 1"
    assert str(x + 1) == "x + 1"
    # 断言 x + x**2 的字符串表示为 "x**2 + x"
    assert str(x + x**2) == "x**2 + x"
    # 断言 Add(0, 1, evaluate=False) 的字符串表示为 "0 + 1"
    assert str(Add(0, 1, evaluate=False)) == "0 + 1"
    # 断言 Add(0, 0, 1, evaluate=False) 的字符串表示为 "0 + 0 + 1"
    assert str(Add(0, 0, 1, evaluate=False)) == "0 + 0 + 1"
    # 断言 1.0*x 的字符串表示为 "1.0*x"
    assert str(1.0*x) == "1.0*x"
    # 断言 5 + x + y + x*y + x**2 + y**2 的字符串表示为 "x**2 + x*y + x + y**2 + y + 5"
    assert str(5 + x + y + x*y + x**2 + y**2) == "x**2 + x*y + x + y**2 + y + 5"
    # 断言 1 + x + x**2/2 + x**3/3 的字符串表示为 "x**3/3 + x**2/2 + x + 1"
    assert str(1 + x + x**2/2 + x**3/3) == "x**3/3 + x**2/2 + x + 1"
    # 断言 2*x - 7*x**2 + 2 + 3*y 的字符串表示为 "-7*x**2 + 2*x + 3*y + 2"
    assert str(2*x - 7*x**2 + 2 + 3*y) == "-7*x**2 + 2*x + 3*y + 2"
    # 断言 x - y 的字符串表示为 "x - y"
    assert str(x - y) == "x - y"
    # 断言 2 - x 的字符串表示为 "2 - x"
    assert str(2 - x) == "2 - x"
    # 断言 x - 2 的字符串表示为 "x - 2"
    assert str(x - 2) == "x - 2"
    # 断言 x - y - z - w 的字符串表示为 "-w + x - y - z"
    assert str(x - y - z - w) == "-w + x - y - z"
    # 断言 x - z*y**2*z*w 的字符串表示为 "-w*y**2*z**2 + x"
    assert str(x - z*y**2*z*w) == "-w*y**2*z**2 + x"
    # 断言 x - 1*y*x*y 的字符串表示为 "-x*y**2 + x"
    assert str(x - 1*y*x*y) == "-x*y**2 + x"
    # 断言 sin(x).series(x, 0, 15) 的字符串表示
    assert str(sin(x).series(x, 0, 15)) == "x - x**3/6 + x**5/120 - x**7/5040 + x**9/362880 - x**11/39916800 + x**13/6227020800 + O(x**15)"
    # 断言 Add(Add(-w, x, evaluate=False), Add(-y, z, evaluate=False), evaluate=False) 的字符串表示为 "(-w + x) + (-y + z)"
    assert str(Add(Add(-w, x, evaluate=False), Add(-y, z,  evaluate=False),  evaluate=False)) == "(-w + x) + (-y + z)"
    # 断言 Add(Add(-x, -y, evaluate=False), -z, evaluate=False) 的字符串表示为 "-z + (-x - y)"
    assert str(Add(Add(-x, -y, evaluate=False), -z, evaluate=False)) == "-z + (-x - y)"
    # 断言 Add(Add(Add(-x, -y, evaluate=False), -z, evaluate=False), -t, evaluate=False) 的字符串表示为 "-t + (-z + (-x - y))"
    assert str(Add(Add(Add(-x, -y, evaluate=False), -z, evaluate=False), -t, evaluate=False)) == "-t + (-z + (-x - y))"


# 测试 Catalan 常数
def test_Catalan():
    # 断言 Catalan 的字符串表示为 "Catalan"
    assert str(Catalan) == "Catalan"


# 测试 ComplexInfinity
def test_ComplexInfinity():
    # 断言 ComplexInfinity 的字符串表示为 "zoo"
    assert str(zoo) == "zoo"


# 测试 Derivative 导数函数
def test_Derivative():
    # 断言 Derivative(x, y) 的字符串表示为 "Derivative(x, y)"
    assert str(Derivative(x, y)) == "Derivative(x, y)"
    # 断言 Derivative(x**2, x, evaluate=False) 的字符串表示为 "Derivative(x**2, x)"
    assert str(Derivative(x**2, x, evaluate=False)) == "Derivative(x**2, x)"
    # 断言 Derivative(x**2/y, x, y, evaluate=False) 的字符串表示为 "Derivative(x**2/y, x, y)"
    assert str(Derivative(x**2/y, x, y, evaluate=False)) == "Derivative(x**2/y, x, y)"


# 测试 dict 字典
def test_dict():
    # 断言 {1: 1 + x} 的字符串表示为 "{1: x + 1}"
    assert str({1: 1 + x}) == sstr({1: 1 + x}) == "{1: x + 1}"
    # 断言 {1: x**2, 2: y*x} 的字符串表示为 "{1: x**2, 2: x*y}" 或 "{2: x*y, 1: x**2}"
    assert str({1: x**2, 2: y*x}) in ("{1: x**2, 2: x*y}", "{2: x*y, 1: x**2}")
    # 断言 sstr({1: x**2, 2: y*x}) 的字符串表示为 "{1: x**2, 2: x*y}"
    assert sstr({1: x**2, 2: y*x}) == "{1: x**2, 2: x*y}"


# 测试 Dict 字典
def test_Dict():
    # 断言 Dict({1: 1 + x}) 的字符串表示为 "{1: x + 1}"
    assert str(Dict({1: 1 + x})) == sstr({1: 1 + x}) == "{1: x + 1}"
    # 断言 Dict({1: x**2, 2: y*x}) 的字符串表示为 "{1: x**2, 2: x*y}" 或 "{2: x*y, 1: x**2}"
    assert str(Dict({1: x**2, 2: y*x})) in ("{1: x**2, 2: x*y}", "{2: x*y, 1: x**2}")
    # 断言 sstr(Dict({1: x**2, 2: y*x})) 的字符串表示为 "{1: x**2, 2: x*y}"
    assert sstr(Dict({1: x**2, 2: y*x})) == "{1: x**2, 2: x*y}"


# 测试 Dummy
def test_Dummy():
    # 断言 d 的字符串表示为 "_d"
    assert str(d) == "_d"
    # 断言 d + x 的字符串表示为 "_d + x"
    assert str(d + x) == "_d + x"


# 测试 EulerGamma
def test_EulerGamma():
    # 断言 EulerGamma 的字符串表示为 "EulerGamma"
    assert str(EulerGamma) == "EulerGamma"


# 测试 Exp 指数函数
def test_Exp():
    # 断言 E 的字符串表示为 "
    # 断言：计算 subfactorial 函数对于参数 3 的返回值是否为字符串 "2"
    assert str(subfactorial(3)) == "2"
    
    # 断言：计算 subfactorial 函数对于变量 n 的返回值是否为字符串 "subfactorial(n)"
    assert str(subfactorial(n)) == "subfactorial(n)"
    
    # 断言：计算 subfactorial 函数对于参数 2*n 的返回值是否为字符串 "subfactorial(2*n)"
    assert str(subfactorial(2*n)) == "subfactorial(2*n)"
def test_Function():
    # 创建一个名为 'f' 的函数对象
    f = Function('f')
    # 对函数 'f' 应用变量 'x'，得到表达式 f(x)
    fx = f(x)
    # 创建一个名为 'w' 的通配符函数对象
    w = WildFunction('w')
    # 断言函数对象 'f' 的字符串表示为 "f"
    assert str(f) == "f"
    # 断言表达式 f(x) 的字符串表示为 "f(x)"
    assert str(fx) == "f(x)"
    # 断言通配符函数 'w' 的字符串表示为 "w_"
    assert str(w) == "w_"


def test_Geometry():
    # 断言点 (0, 0) 的字符串表示为 'Point2D(0, 0)'
    assert sstr(Point(0, 0)) == 'Point2D(0, 0)'
    # 断言以点 (0, 0) 为圆心、半径为 3 的圆的字符串表示为 'Circle(Point2D(0, 0), 3)'
    assert sstr(Circle(Point(0, 0), 3)) == 'Circle(Point2D(0, 0), 3)'
    # 断言以点 (1, 2) 为中心、长轴为 3、短轴为 4 的椭圆的字符串表示为 'Ellipse(Point2D(1, 2), 3, 4)'
    assert sstr(Ellipse(Point(1, 2), 3, 4)) == 'Ellipse(Point2D(1, 2), 3, 4)'
    # 断言以点 (1, 1)、(7, 8)、(0, -1) 为顶点的三角形的字符串表示
    assert sstr(Triangle(Point(1, 1), Point(7, 8), Point(0, -1))) == \
        'Triangle(Point2D(1, 1), Point2D(7, 8), Point2D(0, -1))'
    # 断言以点 (5, 6)、(-2, -3)、(0, 0)、(4, 7) 为顶点的多边形的字符串表示
    assert sstr(Polygon(Point(5, 6), Point(-2, -3), Point(0, 0), Point(4, 7))) == \
        'Polygon(Point2D(5, 6), Point2D(-2, -3), Point2D(0, 0), Point2D(4, 7))'
    # 断言以点 (0, 0)、(1, 0)、(0, 1) 为顶点的三角形的字符串表示，使用 sympy_integers=True
    assert sstr(Triangle(Point(0, 0), Point(1, 0), Point(0, 1)), sympy_integers=True) == \
        'Triangle(Point2D(S(0), S(0)), Point2D(S(1), S(0)), Point2D(S(0), S(1)))'
    # 断言以点 (1, 2) 为中心、长轴为 3、短轴为 4 的椭圆的字符串表示，使用 sympy_integers=True
    assert sstr(Ellipse(Point(1, 2), 3, 4), sympy_integers=True) == \
        'Ellipse(Point2D(S(1), S(2)), S(3), S(4))'


def test_GoldenRatio():
    # 断言黄金比例的字符串表示为 "GoldenRatio"
    assert str(GoldenRatio) == "GoldenRatio"


def test_Heaviside():
    # 断言 Heaviside 函数在变量 'x' 上的字符串表示为 "Heaviside(x)"，默认阈值 S.Half
    assert str(Heaviside(x)) == str(Heaviside(x, S.Half)) == "Heaviside(x)"
    # 断言 Heaviside 函数在变量 'x' 上、阈值为 1 的字符串表示为 "Heaviside(x, 1)"
    assert str(Heaviside(x, 1)) == "Heaviside(x, 1)"


def test_TribonacciConstant():
    # 断言 Tribonacci 常数的字符串表示为 "TribonacciConstant"
    assert str(TribonacciConstant) == "TribonacciConstant"


def test_ImaginaryUnit():
    # 断言虚数单位 'I' 的字符串表示为 "I"
    assert str(I) == "I"


def test_Infinity():
    # 断言无穷大 'oo' 的字符串表示为 "oo"
    assert str(oo) == "oo"
    # 断言无穷大乘虚数单位 'oo*I' 的字符串表示为 "oo*I"
    assert str(oo*I) == "oo*I"


def test_Integer():
    # 断言整数对象的字符串表示
    assert str(Integer(-1)) == "-1"
    assert str(Integer(1)) == "1"
    assert str(Integer(-3)) == "-3"
    assert str(Integer(0)) == "0"
    assert str(Integer(25)) == "25"


def test_Integral():
    # 断言积分表达式的字符串表示
    assert str(Integral(sin(x), y)) == "Integral(sin(x), y)"
    assert str(Integral(sin(x), (y, 0, 1))) == "Integral(sin(x), (y, 0, 1))"


def test_Interval():
    # 测试 Interval 对象的字符串表示
    n = (S.NegativeInfinity, 1, 2, S.Infinity)
    for i in range(len(n)):
        for j in range(i + 1, len(n)):
            for l in (True, False):
                for r in (True, False):
                    ival = Interval(n[i], n[j], l, r)
                    assert S(str(ival)) == ival


def test_AccumBounds():
    # 断言 AccumBounds 对象的字符串表示
    a = Symbol('a', real=True)
    assert str(AccumBounds(0, a)) == "AccumBounds(0, a)"
    assert str(AccumBounds(0, 1)) == "AccumBounds(0, 1)"


def test_Lambda():
    # 断言 Lambda 表达式的字符串表示
    assert str(Lambda(d, d**2)) == "Lambda(_d, _d**2)"
    # issue 2908
    assert str(Lambda((), 1)) == "Lambda((), 1)"
    assert str(Lambda((), x)) == "Lambda((), x)"
    assert str(Lambda((x, y), x+y)) == "Lambda((x, y), x + y)"
    assert str(Lambda(((x, y),), x+y)) == "Lambda(((x, y),), x + y)"


def test_Limit():
    # 断言极限表达式的字符串表示
    assert str(Limit(sin(x)/x, x, y)) == "Limit(sin(x)/x, x, y, dir='+')"
    assert str(Limit(1/x, x, 0)) == "Limit(1/x, x, 0, dir='+')"
    assert str(Limit(sin(x)/x, x, y, dir="-")) == "Limit(sin(x)/x, x, y, dir='-')"


def test_list():
    # 断言列表对象的字符串表示
    assert str([x]) == sstr([x]) == "[x]"
    assert str([x**2, x*y + 1]) == sstr([x**2, x*y + 1]) == "[x**2, x*y + 1]"
    # 使用断言验证两个表达式的字符串表示是否相等
    assert str([x**2, [y + x]]) == sstr([x**2, [y + x]]) == "[x**2, [x + y]]"
# 定义一个测试函数，用于测试 Matrix 类的字符串表示方法
def test_Matrix_str():
    # 创建一个 Matrix 对象 M，包含表达式 [[x**+1, 1], [y, x + y]]
    M = Matrix([[x**+1, 1], [y, x + y]])
    # 断言 Matrix 对象 M 的字符串表示与指定字符串相等
    assert str(M) == "Matrix([[x, 1], [y, x + y]])"
    # 断言调用 sstr 函数后 Matrix 对象 M 的字符串表示与指定格式化字符串相等
    assert sstr(M) == "Matrix([\n[x,     1],\n[y, x + y]])"

    # 重新赋值 Matrix 对象 M
    M = Matrix([[1]])
    # 断言 Matrix 对象 M 的字符串表示与调用 sstr 函数后的结果相等，并且都等于 "Matrix([[1]])"
    assert str(M) == sstr(M) == "Matrix([[1]])"

    # 重新赋值 Matrix 对象 M
    M = Matrix([[1, 2]])
    # 断言 Matrix 对象 M 的字符串表示与调用 sstr 函数后的结果相等，并且都等于 "Matrix([[1, 2]])"
    assert str(M) == sstr(M) ==  "Matrix([[1, 2]])"

    # 重新赋值 Matrix 对象 M
    M = Matrix()
    # 断言 Matrix 对象 M 的字符串表示与调用 sstr 函数后的结果相等，并且都等于 "Matrix(0, 0, [])"
    assert str(M) == sstr(M) == "Matrix(0, 0, [])"

    # 使用特定函数构造 Matrix 对象 M
    M = Matrix(0, 1, lambda i, j: 0)
    # 断言 Matrix 对象 M 的字符串表示与调用 sstr 函数后的结果相等，并且都等于 "Matrix(0, 1, [])"
    assert str(M) == sstr(M) == "Matrix(0, 1, [])"


# 定义一个测试函数，用于测试 Mul（乘法表达式）的字符串表示方法
def test_Mul():
    # 断言 x/y 的字符串表示为 "x/y"
    assert str(x/y) == "x/y"
    # 断言 y/x 的字符串表示为 "y/x"
    assert str(y/x) == "y/x"
    # 断言 x/y/z 的字符串表示为 "x/(y*z)"
    assert str(x/y/z) == "x/(y*z)"
    # 断言 (x + 1)/(y + 2) 的字符串表示为 "(x + 1)/(y + 2)"
    assert str((x + 1)/(y + 2)) == "(x + 1)/(y + 2)"
    # 断言 2*x/3 的字符串表示为 '2*x/3'
    assert str(2*x/3) == '2*x/3'
    # 断言 -2*x/3 的字符串表示为 '-2*x/3'
    assert str(-2*x/3) == '-2*x/3'
    # 断言 -1.0*x 的字符串表示为 '-1.0*x'
    assert str(-1.0*x) == '-1.0*x'
    # 断言 1.0*x 的字符串表示为 '1.0*x'
    assert str(1.0*x) == '1.0*x'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '0*1'
    assert str(Mul(0, 1, evaluate=False)) == '0*1'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*0'
    assert str(Mul(1, 0, evaluate=False)) == '1*0'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*1'
    assert str(Mul(1, 1, evaluate=False)) == '1*1'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*1*1'
    assert str(Mul(1, 1, 1, evaluate=False)) == '1*1*1'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*2'
    assert str(Mul(1, 2, evaluate=False)) == '1*2'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*(1/2)'
    assert str(Mul(1, S.Half, evaluate=False)) == '1*(1/2)'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*1*(1/2)'
    assert str(Mul(1, 1, S.Half, evaluate=False)) == '1*1*(1/2)'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*1*2*3*x'
    assert str(Mul(1, 1, 2, 3, x, evaluate=False)) == '1*1*2*3*x'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '1*(-1)'
    assert str(Mul(1, -1, evaluate=False)) == '1*(-1)'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '-1*1'
    assert str(Mul(-1, 1, evaluate=False)) == '-1*1'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '4*3*2*1*0*y*x'
    assert str(Mul(4, 3, 2, 1, 0, y, x, evaluate=False)) == '4*3*2*1*0*y*x'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '4*3*2*(z + 1)*0*y*x'
    assert str(Mul(4, 3, 2, 1+z, 0, y, x, evaluate=False)) == '4*3*2*(z + 1)*0*y*x'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '(2/3)*(5/7)'
    assert str(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == '(2/3)*(5/7)'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 '-2*x/(y*y)'
    assert str(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2*x/(y*y)'
    # 断言使用 evaluate=False 构造的 Mul 对象的字符串表示为 'x/(1/y)'
    assert str(Mul(x, Pow(1/y, -1, evaluate=False), evaluate=False)) == 'x/(1/y)'

    # Issue 24108
    # 断言在 evaluate=False 环境下，表达式的字符串表示为 "(-1 - 1*1)/2"
    from sympy.core.parameters import evaluate
    with evaluate(False):
        assert str(Mul(Pow(Integer(2), Integer(-1)), Add(Integer(-1), Mul(Integer(-1), Integer(1))))) == "(-1 - 1*1)/2"

    # 定义两个自定义类
    class CustomClass1(Expr):
        is_commutative = True

    class CustomClass2(Expr):
        is_commutative = True
    # 创建两个自定义类的实例
    cc1 = CustomClass1()
    cc2 = CustomClass2()
    # 断言 Rational(2)*cc1 的字符串表示为 '2*CustomClass1()'
    assert str(Rational(2)*cc1) == '2*CustomClass1()'
    # 断言 cc1*Rational(2) 的字符串表示为 '2*CustomClass1()'
    assert str(cc1*Rational(2)) == '2*CustomClass1()'
    # 断言 cc1*Float("1.5") 的字符串表示为 '1.5*CustomClass1()'
    assert str(cc1*Float("1.5")) == '1.5*CustomClass1()'
    # 断言 cc2*Rational(2) 的字符串表示为 '2*CustomClass2()'
    assert str(cc2*Rational(2)) == '2*CustomClass2()'
    # 断言 cc2*Rational(2)*cc1 的字符串表示为 '2*CustomClass1()*CustomClass2()'
    assert str(cc2*Rational(2)*cc1) == '2*CustomClass1()*CustomClass2()'
    # 断言 cc1*Rational(2)*cc2 的字符串表示为 '2*Custom
    # 断言：确保 O 函数返回的字符串表示与指定的字符串相同
    assert str(O(x, x, y)) == "O(x, x, y)"
    # 断言：确保 O 函数返回的字符串表示与指定的字符串相同
    assert str(O(x, x, y)) == "O(x, x, y)"
    # 断言：确保 O 函数返回的字符串表示与指定的字符串相同
    assert str(O(x, (x, oo), (y, oo))) == "O(x, (x, oo), (y, oo))"
# 定义测试函数 test_Permutation_Cycle，用于测试置换（permutation）和循环（cycle）相关功能
def test_Permutation_Cycle():
    # 从 sympy.combinatorics 导入 Permutation 和 Cycle 类
    from sympy.combinatorics import Permutation, Cycle

    # 对于每个置换和预期结果的元组进行迭代
    for p, s in [
        (Cycle(),           # 创建一个空循环置换
        '()'),              # 预期字符串表示为空置换
        (Cycle(2),          # 创建一个置换，将 2 映射到 2 自身
        '(2)'),             # 预期字符串表示为 (2)
        (Cycle(2, 1),       # 创建一个置换，将 1 映射到 2，2 映射到 1
        '(1 2)'),           # 预期字符串表示为 (1 2)
        (Cycle(1, 2)(5)(6, 7)(10),  # 创建一个复合置换，多个循环
        '(1 2)(6 7)(10)'),  # 预期字符串表示为 (1 2)(6 7)(10)
        (Cycle(3, 4)(1, 2)(3, 4),   # 创建一个复合置换，包含重叠的循环
        '(1 2)(4)'),        # 预期字符串表示为 (1 2)(4)
    ]:
        # 使用 sstr 函数将置换对象 p 转换为字符串，并断言其与预期字符串 s 相同
        assert sstr(p) == s

    # 对于每个置换和预期结果的元组进行迭代，测试非循环表示
    for p, s in [
        (Permutation([]),       # 创建一个空置换
        'Permutation([])'),     # 预期字符串表示为 Permutation([])
        (Permutation([], size=1),   # 创建一个空置换，大小为 1
        'Permutation([0])'),        # 预期字符串表示为 Permutation([0])
        (Permutation([], size=2),   # 创建一个空置换，大小为 2
        'Permutation([0, 1])'),     # 预期字符串表示为 Permutation([0, 1])
        (Permutation([], size=10),  # 创建一个空置换，大小为 10
        'Permutation([], size=10)'),# 预期字符串表示为 Permutation([], size=10)
        (Permutation([1, 0, 2]),    # 创建一个置换，重新排列元素
        'Permutation([1, 0, 2])'),  # 预期字符串表示为 Permutation([1, 0, 2])
        (Permutation([1, 0, 2, 3, 4, 5]),   # 创建一个置换，部分循环
        'Permutation([1, 0], size=6)'),    # 预期字符串表示为 Permutation([1, 0], size=6)
        (Permutation([1, 0, 2, 3, 4, 5], size=10),   # 创建一个置换，部分循环，大小为 10
        'Permutation([1, 0], size=10)'),  # 预期字符串表示为 Permutation([1, 0], size=10)
    ]:
        # 使用 sstr 函数将置换对象 p 转换为字符串，禁用循环表示，并断言其与预期字符串 s 相同
        assert sstr(p, perm_cyclic=False) == s

    # 对于每个置换和预期结果的元组进行迭代，测试循环表示
    for p, s in [
        (Permutation([]),       # 创建一个空置换
        '()'),                  # 预期字符串表示为空置换
        (Permutation([], size=1),   # 创建一个空置换，大小为 1
        '(0)'),                     # 预期字符串表示为 (0)
        (Permutation([], size=2),   # 创建一个空置换，大小为 2
        '(1)'),                     # 预期字符串表示为 (1)
        (Permutation([], size=10),  # 创建一个空置换，大小为 10
        '(9)'),                     # 预期字符串表示为 (9)
        (Permutation([1, 0, 2]),    # 创建一个置换，重新排列元素
        '(2)(0 1)'),                # 预期字符串表示为 (2)(0 1)
        (Permutation([1, 0, 2, 3, 4, 5]),   # 创建一个置换，部分循环
        '(5)(0 1)'),                    # 预期字符串表示为 (5)(0 1)
        (Permutation([1, 0, 2, 3, 4, 5], size=10),   # 创建一个置换，部分循环，大小为 10
        '(9)(0 1)'),                            # 预期字符串表示为 (9)(0 1)
        (Permutation([0, 1, 3, 2, 4, 5], size=10),   # 创建一个置换，重新排列元素，大小为 10
        '(9)(2 3)'),                                # 预期字符串表示为 (9)(2 3)
    ]:
        # 使用 sstr 函数将置换对象 p 转换为字符串，并断言其与预期字符串 s 相同
        assert sstr(p) == s

    # 使用 warns_deprecated_sympy 上下文管理器，测试废弃警告
    with warns_deprecated_sympy():
        # 存储旧的打印循环标志
        old_print_cyclic = Permutation.print_cyclic
        # 禁用置换对象的循环打印
        Permutation.print_cyclic = False
        # 断言置换对象 [1, 0, 2] 的字符串表示为 'Permutation([1, 0, 2])'
        assert sstr(Permutation([1, 0, 2])) == 'Permutation([1, 0, 2])'
        # 恢复原始的打印循环标志
        Permutation.print_cyclic = old_print_cyclic

# 定义测试函数 test_Pi，测试符号 π 的字符串表示
def test_Pi():
    # 断言符号 π 的字符串表示为 "pi"
    assert str(pi) == "pi"

# 定义测试函数 test_Poly，测试多项式对象的字符串表示
def test_Poly():
    # 断言多项式 Poly(0, x, domain='ZZ') 的字符串表示为 "Poly(0, x, domain='ZZ')"
    assert str(Poly(0, x)) == "Poly(0, x, domain='ZZ')"
    # 断言多项式 Poly(1, x, domain='ZZ') 的字符串表示为 "Poly(1, x, domain='ZZ')"
    assert str(Poly(1, x)) == "Poly(1, x, domain='ZZ')"
    # 断言多项式 Poly(x, x, domain='ZZ') 的字符串表示为 "Poly(x, x, domain='ZZ')"
    assert str(Poly(x, x)) == "Poly(x, x, domain='ZZ')"

    # 断言多项式 Poly(2*x + 1, x, domain='ZZ') 的字符串表示为 "Poly(2*x + 1, x, domain='ZZ')"
    assert str(Poly(2*x + 1, x)) == "Poly(2*x + 1, x, domain='ZZ')"
    # 断言多项式 Poly(2*x - 1, x, domain='ZZ') 的字符串表示为 "Poly(2*x - 1, x, domain='ZZ')"
    assert str(Poly(2*x - 1, x)) == "Poly(2*x - 1, x, domain='ZZ')"

    # 断言多项式 Poly(-1, x, domain='ZZ') 的字符串表示为 "Poly(-1, x, domain='ZZ')"
    assert str(Poly(-1, x)) == "Poly(-1, x, domain='ZZ')"
    # 断言多项式 Poly(-x, x, domain='ZZ') 的字符串表示为 "Poly(-x, x, domain='ZZ')"
    assert str(Poly(-x, x)) == "Poly(-x, x, domain='ZZ')"

    # 断言多项式 Poly(-2*x + 1, x, domain='ZZ') 的字符串表示为 "Poly(-2*x + 1, x, domain
    # 断言：验证多项式对象的字符串表示是否符合预期
    assert str(
        Poly(x**2 + 1 + y, x)) == "Poly(x**2 + y + 1, x, domain='ZZ[y]')"
    
    # 断言：验证多项式对象的字符串表示是否符合预期
    assert str(
        Poly(x**2 - 1 + y, x)) == "Poly(x**2 + y - 1, x, domain='ZZ[y]')"
    
    # 断言：验证多项式对象的字符串表示是否符合预期（包含虚数单位 I）
    assert str(Poly(x**2 + I*x, x)) == "Poly(x**2 + I*x, x, domain='ZZ_I')"
    
    # 断言：验证多项式对象的字符串表示是否符合预期（包含虚数单位 I）
    assert str(Poly(x**2 - I*x, x)) == "Poly(x**2 - I*x, x, domain='ZZ_I')"
    
    # 断言：验证多变量多项式对象的字符串表示是否符合预期
    assert str(Poly(-x*y*z + x*y - 1, x, y, z)) == "Poly(-x*y*z + x*y - 1, x, y, z, domain='ZZ')"
    
    # 断言：验证多变量多项式对象的字符串表示是否符合预期（包含符号 w）
    assert str(Poly(-w*x**21*y**7*z + (1 + w)*z**3 - 2*x*z + 1, x, y, z)) == \
        "Poly(-w*x**21*y**7*z - 2*x*z + (w + 1)*z**3 + 1, x, y, z, domain='ZZ[w]')"
    
    # 断言：验证多项式对象的字符串表示是否符合预期（使用模数为 2）
    assert str(Poly(x**2 + 1, x, modulus=2)) == "Poly(x**2 + 1, x, modulus=2)"
    
    # 断言：验证多项式对象的字符串表示是否符合预期（使用模数为 17）
    assert str(Poly(2*x**2 + 3*x + 4, x, modulus=17)) == "Poly(2*x**2 + 3*x + 4, x, modulus=17)"
# 定义测试函数 test_PolyRing，用于测试多项式环的创建和字符串表示
def test_PolyRing():
    # 断言验证多项式环的字符串表示是否符合预期
    assert str(ring("x", ZZ, lex)[0]) == "Polynomial ring in x over ZZ with lex order"
    assert str(ring("x,y", QQ, grlex)[0]) == "Polynomial ring in x, y over QQ with grlex order"
    assert str(ring("x,y,z", ZZ["t"], lex)[0]) == "Polynomial ring in x, y, z over ZZ[t] with lex order"


# 定义测试函数 test_FracField，用于测试有理函数域的创建和字符串表示
def test_FracField():
    # 断言验证有理函数域的字符串表示是否符合预期
    assert str(field("x", ZZ, lex)[0]) == "Rational function field in x over ZZ with lex order"
    assert str(field("x,y", QQ, grlex)[0]) == "Rational function field in x, y over QQ with grlex order"
    assert str(field("x,y,z", ZZ["t"], lex)[0]) == "Rational function field in x, y, z over ZZ[t] with lex order"


# 定义测试函数 test_PolyElement，用于测试多项式元素的创建和字符串表示
def test_PolyElement():
    # 创建多项式环及元素
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)
    Rx_zzi, xz = ring("x", ZZ_I)

    # 断言验证多项式元素的字符串表示是否符合预期
    assert str(x - x) == "0"
    assert str(x - 1) == "x - 1"
    assert str(x + 1) == "x + 1"
    assert str(x**2) == "x**2"
    assert str(x**(-2)) == "x**(-2)"
    assert str(x**QQ(1, 2)) == "x**(1/2)"

    assert str((u**2 + 3*u*v + 1)*x**2*y + u + 1) == "(u**2 + 3*u*v + 1)*x**2*y + u + 1"
    assert str((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x) == "(u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x"
    assert str((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1) == "(u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1"
    assert str((-u**2 + 3*u*v - 1)*x**2*y - (u + 1)*x - 1) == "-(u**2 - 3*u*v + 1)*x**2*y - (u + 1)*x - 1"

    assert str(-(v**2 + v + 1)*x + 3*u*v + 1) == "-(v**2 + v + 1)*x + 3*u*v + 1"
    assert str(-(v**2 + v + 1)*x - 3*u*v + 1) == "-(v**2 + v + 1)*x - 3*u*v + 1"

    assert str((1+I)*xz + 2) == "(1 + 1*I)*x + (2 + 0*I)"


# 定义测试函数 test_FracElement，用于测试有理函数元素的创建和字符串表示
def test_FracElement():
    # 创建有理函数域及元素
    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)
    Rx_zzi, xz = field("x", QQ_I)
    i = QQ_I(0, 1)

    # 断言验证有理函数元素的字符串表示是否符合预期
    assert str(x - x) == "0"
    assert str(x - 1) == "x - 1"
    assert str(x + 1) == "x + 1"

    assert str(x/3) == "x/3"
    assert str(x/z) == "x/z"
    assert str(x*y/z) == "x*y/z"
    assert str(x/(z*t)) == "x/(z*t)"
    assert str(x*y/(z*t)) == "x*y/(z*t)"

    assert str((x - 1)/y) == "(x - 1)/y"
    assert str((x + 1)/y) == "(x + 1)/y"
    assert str((-x - 1)/y) == "(-x - 1)/y"
    assert str((x + 1)/(y*z)) == "(x + 1)/(y*z)"
    assert str(-y/(x + 1)) == "-y/(x + 1)"
    assert str(y*z/(x + 1)) == "y*z/(x + 1)"

    assert str(((u + 1)*x*y + 1)/((v - 1)*z - 1)) == "((u + 1)*x*y + 1)/((v - 1)*z - 1)"
    assert str(((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)) == "((u + 1)*x*y + 1)/((v - 1)*z - u*v*t - 1)"

    assert str((1+i)/xz) == "(1 + 1*I)/x"
    assert str(((1+i)*xz - i)/xz) == "((1 + 1*I)*x + (0 + -1*I))/x"


# 定义测试函数 test_GaussianInteger，用于测试高斯整数的创建和字符串表示
def test_GaussianInteger():
    # 断言验证高斯整数的字符串表示是否符合预期
    assert str(ZZ_I(1, 0)) == "1"
    assert str(ZZ_I(-1, 0)) == "-1"
    assert str(ZZ_I(0, 1)) == "I"
    assert str(ZZ_I(0, -1)) == "-I"
    assert str(ZZ_I(0, 2)) == "2*I"
    assert str(ZZ_I(0, -2)) == "-2*I"
    assert str(ZZ_I(1, 1)) == "1 + I"
    assert str(ZZ_I(-1, -1)) == "-1 - I"
    assert str(ZZ_I(-1, -2)) == "-1 - 2*I"
    # 确保 QQ_I 函数返回值为字符串 "1"，断言判断
    assert str(QQ_I(1, 0)) == "1"
    # 确保 QQ_I 函数返回值为字符串 "2/3"，断言判断
    assert str(QQ_I(QQ(2, 3), 0)) == "2/3"
    # 确保 QQ_I 函数返回值为字符串 "2*I/3"，断言判断
    assert str(QQ_I(0, QQ(2, 3))) == "2*I/3"
    # 确保 QQ_I 函数返回值为字符串 "1/2 - 2*I/3"，断言判断
    assert str(QQ_I(QQ(1, 2), QQ(-2, 3))) == "1/2 - 2*I/3"
# 定义测试函数 test_Pow，用于测试幂运算的字符串表示是否正确
def test_Pow():
    # 断言 x 的负一次幂的字符串表示是否为 "1/x"
    assert str(x**-1) == "1/x"
    # 断言 x 的负二次幂的字符串表示是否为 "x**(-2)"
    assert str(x**-2) == "x**(-2)"
    # 断言 x 的二次幂的字符串表示是否为 "x**2"
    assert str(x**2) == "x**2"
    # 断言 (x + y) 的负一次幂的字符串表示是否为 "1/(x + y)"
    assert str((x + y)**-1) == "1/(x + y)"
    # 断言 (x + y) 的负二次幂的字符串表示是否为 "(x + y)**(-2)"
    assert str((x + y)**-2) == "(x + y)**(-2)"
    # 断言 (x + y) 的二次幂的字符串表示是否为 "(x + y)**2"
    assert str((x + y)**2) == "(x + y)**2"
    # 断言 (x + y) 的 (1 + x) 次幂的字符串表示是否为 "(x + y)**(x + 1)"
    assert str((x + y)**(1 + x)) == "(x + y)**(x + 1)"
    # 断言 x 的 Rational(1, 3) 次幂的字符串表示是否为 "x**(1/3)"
    assert str(x**Rational(1, 3)) == "x**(1/3)"
    # 断言 1 除以 x 的 Rational(1, 3) 次幂的字符串表示是否为 "x**(-1/3)"
    assert str(1/x**Rational(1, 3)) == "x**(-1/3)"
    # 断言 sqrt(sqrt(x)) 的字符串表示是否为 "x**(1/4)"
    assert str(sqrt(sqrt(x))) == "x**(1/4)"
    # 断言 x 的负一点零次幂的字符串表示是否为 'x**(-1.0)'
    assert str(x**-1.0) == 'x**(-1.0)'
    # 断言 Pow(S(2), -1.0, evaluate=False) 的字符串表示是否为 '2**(-1.0)'
    # 查看问题号为 2860 的问题
    assert str(Pow(S(2), -1.0, evaluate=False)) == '2**(-1.0)'


# 定义测试函数 test_sqrt，用于测试平方根运算的字符串表示是否正确
def test_sqrt():
    # 断言 sqrt(x) 的字符串表示是否为 "sqrt(x)"
    assert str(sqrt(x)) == "sqrt(x)"
    # 断言 sqrt(x**2) 的字符串表示是否为 "sqrt(x**2)"
    assert str(sqrt(x**2)) == "sqrt(x**2)"
    # 断言 1 除以 sqrt(x) 的字符串表示是否为 "1/sqrt(x)"
    assert str(1/sqrt(x)) == "1/sqrt(x)"
    # 断言 1 除以 sqrt(x**2) 的字符串表示是否为 "1/sqrt(x**2)"
    assert str(1/sqrt(x**2)) == "1/sqrt(x**2)"
    # 断言 y 除以 sqrt(x) 的字符串表示是否为 "y/sqrt(x)"
    assert str(y/sqrt(x)) == "y/sqrt(x)"
    # 断言 x 的 0.5 次幂的字符串表示是否为 "x**0.5"
    assert str(x**0.5) == "x**0.5"
    # 断言 1 除以 x 的 0.5 次幂的字符串表示是否为 "x**(-0.5)"
    assert str(1/x**0.5) == "x**(-0.5)"


# 定义测试函数 test_Rational，用于测试有理数运算的字符串表示是否正确
def test_Rational():
    # 创建有理数对象 n1、n2、n3、n4、n5、n7、n8
    n1 = Rational(1, 4)
    n2 = Rational(1, 3)
    n3 = Rational(2, 4)
    n4 = Rational(2, -4)
    n5 = Rational(0)
    n7 = Rational(3)
    n8 = Rational(-3)
    # 断言 n1 乘以 n2 的字符串表示是否为 "1/12"
    assert str(n1*n2) == "1/12"
    # 断言 n1 乘以 n2 的字符串表示是否为 "1/12"（重复的断言）
    assert str(n1*n2) == "1/12"
    # 断言 n3 的字符串表示是否为 "1/2"
    assert str(n3) == "1/2"
    # 断言 n1 乘以 n3 的字符串表示是否为 "1/8"
    assert str(n1*n3) == "1/8"
    # 断言 n1 加上 n3 的字符串表示是否为 "3/4"
    assert str(n1 + n3) == "3/4"
    # 断言 n1 加上 n2 的字符串表示是否为 "7/12"
    assert str(n1 + n2) == "7/12"
    # 断言 n1 加上 n4 的字符串表示是否为 "-1/4"
    assert str(n1 + n4) == "-1/4"
    # 断言 n4 乘以 n4 的字符串表示是否为 "1/4"
    assert str(n4*n4) == "1/4"
    # 断言 n4 加上 n2 的字符串表示是否为 "-1/6"
    assert str(n4 + n2) == "-1/6"
    # 断言 n4 加上 n5 的字符串表示是否为 "-1/2"
    assert str(n4 + n5) == "-1/2"
    # 断言 n4 乘以 n5 的字符串表示是否为 "0"
    assert str(n4*n5) == "0"
    # 断言 n3 加上 n4 的字符串表示是否为 "0"
    assert str(n3 + n4) == "0"
    # 断言 n1 的 n7 次幂的字符串表示是否为 "1/64"
    assert str(n1**n7) == "1/64"
    # 断言 n2 的 n7 次幂的字符串表示是否为 "1/27"
    assert str(n2**n7) == "1/27"
    # 断言 n2 的 n8 次幂的字符串表示是否为 "27"
    assert str(n2**n8) == "27"
    # 断言 n7 的 n8 次幂的字符串表示是否为 "1/27"
    assert str(n7**n8) == "1/27"
    # 断言 Rational("-25") 的字符串表示是否为 "-25"
    assert str(Rational("-25")) == "-25"
    # 断言 Rational("1.25") 的字符串表示是否为 "5/4"
    assert str(Rational("1.25")) == "5/4"
    # 断言 Rational("-2.6e-2") 的字符串表示是否为 "-13/500"
    assert str(Rational("-2.6e-2")) == "-13/500"
    # 断言 S("25/7") 的字符串表示是否为 "25/7"
    assert str(S("25/7")) == "25/7"
    # 断言 S("-123/569") 的字符串表示是否为 "-123/569"
    assert str(S("-123/569")) == "-123/569"
    # 断言 S("0.1[23]", rational=1) 的字符串表示是否为 "61/495"
    assert str(S("0.1[23]", rational=1)) == "61/495"
    # 断言 S("5.1[666]", rational=1) 的字符串表示是否为 "31/6"
    assert str(S("5.1[666]", rational=1)) == "31/6"
    # 断言 S("-5.1[666]", rational=1) 的字符串表示是否为 "-31/6"
    assert str(S("-5.1[666]", rational=1)) == "-31/6"
    # 断言 S("0.[9]", rational=1) 的字符串表示是否为 "1"
    assert str(S("0.[9]", rational=1)) == "1"
    # 断言 S("-0.[9]", rational=1) 的字符串表示是否为 "-1"

    # 断言 sqrt(Rational(1, 4)) 的字符串表示是否为 "
    # NOTE dps is the whole number of decimal digits
    # 使用 assert 语句验证浮点数对象 Float 的字符串表示是否正确
    assert str(Float('1.23', dps=1 + 2)) == '1.23'
    assert str(Float('1.23456789', dps=1 + 8)) == '1.23456789'
    assert str(
        Float('1.234567890123456789', dps=1 + 18)) == '1.234567890123456789'
    # 使用 evalf 方法计算 pi 的近似值，并验证其字符串表示是否正确
    assert str(pi.evalf(1 + 2)) == '3.14'
    assert str(pi.evalf(1 + 14)) == '3.14159265358979'
    # 验证对于更高精度的计算，pi 的字符串表示是否正确
    assert str(pi.evalf(1 + 64)) == ('3.141592653589793238462643383279'
                                     '5028841971693993751058209749445923')
    # 使用 round 方法对 pi 进行舍入，并验证其字符串表示是否正确
    assert str(pi.round(-1)) == '0.0'
    # 使用 n 方法计算 pi 的幂次运算，并验证其字符串表示是否正确
    assert str((pi**400 - (pi**400).round(1)).n(2)) == '-0.e+88'
    # 使用 sstr 函数将浮点数对象 Float 转换为字符串表示，并验证其结果是否正确
    assert sstr(Float("100"), full_prec=False, min=-2, max=2) == '1.0e+2'
    assert sstr(Float("100"), full_prec=False, min=-2, max=3) == '100.0'
    assert sstr(Float("0.1"), full_prec=False, min=-2, max=3) == '0.1'
    assert sstr(Float("0.099"), min=-2, max=3) == '9.90000000000000e-2'
# 测试 Relational 类的字符串表示是否正确
def test_Relational():
    # 检查 Relational 表达式 x < y 的字符串表示
    assert str(Rel(x, y, "<")) == "x < y"
    # 检查 Relational 表达式 x + y == y 的字符串表示
    assert str(Rel(x + y, y, "==")) == "Eq(x + y, y)"
    # 检查 Relational 表达式 x != y 的字符串表示
    assert str(Rel(x, y, "!=")) == "Ne(x, y)"
    # 检查 Logical 表达式 Eq(x, 1) | Eq(x, 2) 的字符串表示
    assert str(Eq(x, 1) | Eq(x, 2)) == "Eq(x, 1) | Eq(x, 2)"
    # 检查 Logical 表达式 Ne(x, 1) & Ne(x, 2) 的字符串表示
    assert str(Ne(x, 1) & Ne(x, 2)) == "Ne(x, 1) & Ne(x, 2)"


# 测试 AppliedBinaryRelation 类的字符串表示是否正确
def test_AppliedBinaryRelation():
    # 检查 AppliedBinaryRelation 类 Q.eq(x, y) 的字符串表示
    assert str(Q.eq(x, y)) == "Q.eq(x, y)"
    # 检查 AppliedBinaryRelation 类 Q.ne(x, y) 的字符串表示
    assert str(Q.ne(x, y)) == "Q.ne(x, y)"


# 测试 CRootOf 类的字符串表示是否正确
def test_CRootOf():
    # 检查 CRootOf 表达式 rootof(x**5 + 2*x - 1, 0) 的字符串表示
    assert str(rootof(x**5 + 2*x - 1, 0)) == "CRootOf(x**5 + 2*x - 1, 0)"


# 测试 RootSum 类的字符串表示是否正确
def test_RootSum():
    f = x**5 + 2*x - 1

    # 检查 RootSum 表达式 RootSum(x**5 + 2*x - 1, Lambda(z, z), auto=False) 的字符串表示
    assert str(RootSum(f, Lambda(z, z), auto=False)) == "RootSum(x**5 + 2*x - 1)"
    # 检查 RootSum 表达式 RootSum(x**5 + 2*x - 1, Lambda(z, z**2), auto=False) 的字符串表示
    assert str(RootSum(f, Lambda(z, z**2), auto=False)) == "RootSum(x**5 + 2*x - 1, Lambda(z, z**2))"


# 测试 GroebnerBasis 类的字符串表示是否正确
def test_GroebnerBasis():
    # 检查 GroebnerBasis 表达式 groebner([], x, y) 的字符串表示
    assert str(groebner([], x, y)) == "GroebnerBasis([], x, y, domain='ZZ', order='lex')"

    F = [x**2 - 3*y - x + 1, y**2 - 2*x + y - 1]

    # 检查 GroebnerBasis 表达式 groebner(F, order='grlex') 的字符串表示
    assert str(groebner(F, order='grlex')) == \
        "GroebnerBasis([x**2 - x - 3*y + 1, y**2 - 2*x + y - 1], x, y, domain='ZZ', order='grlex')"
    # 检查 GroebnerBasis 表达式 groebner(F, order='lex') 的字符串表示
    assert str(groebner(F, order='lex')) == \
        "GroebnerBasis([2*x - y**2 - y + 1, y**4 + 2*y**3 - 3*y**2 - 16*y + 7], x, y, domain='ZZ', order='lex')"


# 测试 set 函数的字符串表示是否正确
def test_set():
    # 检查空集合 set() 的字符串表示
    assert sstr(set()) == 'set()'
    # 检查空不可变集合 frozenset() 的字符串表示
    assert sstr(frozenset()) == 'frozenset()'

    # 检查单元素集合 {1} 的字符串表示
    assert sstr({1}) == '{1}'
    # 检查单元素不可变集合 frozenset({1}) 的字符串表示
    assert sstr(frozenset([1])) == 'frozenset({1})'
    # 检查多元素集合 {1, 2, 3} 的字符串表示
    assert sstr({1, 2, 3}) == '{1, 2, 3}'
    # 检查多元素不可变集合 frozenset({1, 2, 3}) 的字符串表示
    assert sstr(frozenset([1, 2, 3])) == 'frozenset({1, 2, 3})'

    # 检查复杂集合 {1, x, x**2, x**3, x**4} 的字符串表示
    assert sstr({1, x, x**2, x**3, x**4}) == '{1, x, x**2, x**3, x**4}'
    # 检查复杂不可变集合 frozenset({1, x, x**2, x**3, x**4}) 的字符串表示
    assert sstr(frozenset([1, x, x**2, x**3, x**4])) == 'frozenset({1, x, x**2, x**3, x**4})'


# 测试 SparseMatrix 类的字符串表示是否正确
def test_SparseMatrix():
    M = SparseMatrix([[x**+1, 1], [y, x + y]])
    # 检查 SparseMatrix 对象 M 的字符串表示
    assert str(M) == "Matrix([[x, 1], [y, x + y]])"
    # 检查 SparseMatrix 对象 M 的详细字符串表示
    assert sstr(M) == "Matrix([\n[x,     1],\n[y, x + y]])"


# 测试 Sum 类的字符串表示是否正确
def test_Sum():
    # 检查 Sum 表达式 Sum(cos(3*z), (z, x, y)) 的字符串表示
    assert str(summation(cos(3*z), (z, x, y))) == "Sum(cos(3*z), (z, x, y))"
    # 检查 Sum 表达式 Sum(x*y**2, (x, -2, 2), (y, -5, 5)) 的字符串表示
    assert str(Sum(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        "Sum(x*y**2, (x, -2, 2), (y, -5, 5))"


# 测试 Symbol 类的字符串表示是否正确
def test_Symbol():
    # 检查符号 y 的字符串表示
    assert str(y) == "y"
    # 检查符号 x 的字符串表示
    assert str(x) == "x"
    e = x
    # 检查变量 e 的字符串表示
    assert str(e) == "x"


# 测试 tuple 函数的字符串表示是否正确
def test_tuple():
    # 检查元组 (x,) 的字符串表示和简略字符串表示
    assert str((x,)) == sstr((x,)) == "(x,)"
    # 检查元组 (x + y, 1 + x) 的字符串表示和简略字符串表示
    assert str((x + y, 1 + x)) == sstr((x + y, 1 + x)) == "(x + y, x + 1)"
    # 检查元组 (x + y, (1 + x, x**2)) 的字符串表示和简略字符串表示
    assert str((x + y, (1 + x, x**2))) == sstr((x + y, (1 + x, x**2))) == "(x + y, (x + 1, x**2))"


# 测试 Series_str 函数的字符串表示是否正确
def test_Series_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    # 检查 Series 类 Series(tf1, tf2) 的字符串表示
    assert str(Series(tf1, tf2)) ==
    # 断言语句：验证 Series 对象转换为字符串后是否等于特定的字符串
    assert str(Series(-tf2, tf1)) == \
        "Series(TransferFunction(-x + y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y))"
# 定义测试函数，测试 MIMOSeries 类的字符串表示
def test_MIMOSeries_str():
    # 创建两个传递函数对象 tf1 和 tf2
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    # 创建传递函数矩阵对象 tfm_1 和 tfm_2
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    # 断言 MIMOSeries 对象的字符串表示符合预期
    assert str(MIMOSeries(tfm_1, tfm_2)) == \
        "MIMOSeries(TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), " \
        "(TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)))), " \
        "TransferFunctionMatrix(((TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)), " \
        "(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)))))"


# 定义测试函数，测试 TransferFunction 类的字符串表示
def test_TransferFunction_str():
    # 创建传递函数对象 tf1
    tf1 = TransferFunction(x - 1, x + 1, x)
    # 断言 tf1 对象的字符串表示符合预期
    assert str(tf1) == "TransferFunction(x - 1, x + 1, x)"
    # 创建传递函数对象 tf2
    tf2 = TransferFunction(x + 1, 2 - y, x)
    # 断言 tf2 对象的字符串表示符合预期
    assert str(tf2) == "TransferFunction(x + 1, 2 - y, x)"
    # 创建传递函数对象 tf3
    tf3 = TransferFunction(y, y**2 + 2*y + 3, y)
    # 断言 tf3 对象的字符串表示符合预期
    assert str(tf3) == "TransferFunction(y, y**2 + 2*y + 3, y)"


# 定义测试函数，测试 Parallel 类的字符串表示
def test_Parallel_str():
    # 创建传递函数对象 tf1 和 tf2
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    # 断言 Parallel 对象的字符串表示符合预期
    assert str(Parallel(tf1, tf2)) == \
        "Parallel(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y))"
    # 创建传递函数对象 tf3
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    # 断言 Parallel 对象包含三个传递函数对象时的字符串表示符合预期
    assert str(Parallel(tf1, tf2, tf3)) == \
        "Parallel(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y), " \
        "TransferFunction(t*x**2 - t**w*x + w, t - y, y))"
    # 断言 Parallel 对象包含 tf2 和 tf1 的负传递函数对象时的字符串表示符合预期
    assert str(Parallel(-tf2, tf1)) == \
        "Parallel(TransferFunction(-x + y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y))"


# 定义测试函数，测试 MIMOParallel 类的字符串表示
def test_MIMOParallel_str():
    # 创建传递函数对象 tf1 和 tf2
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    # 创建传递函数矩阵对象 tfm_1 和 tfm_2
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    # 断言 MIMOParallel 对象的字符串表示符合预期
    assert str(MIMOParallel(tfm_1, tfm_2)) == \
        "MIMOParallel(TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), " \
        "(TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)))), " \
        "TransferFunctionMatrix(((TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)), " \
        "(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)))))"


# 定义测试函数，测试 Feedback 类的字符串表示
def test_Feedback_str():
    # 创建传递函数对象 tf1、tf2 和 tf3
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    # 断言 Feedback 对象的字符串表示符合预期
    assert str(Feedback(tf1*tf2, tf3)) == \
        "Feedback(Series(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), " \
        "TransferFunction(t*x**2 - t**w*x + w, t - y, y), -1)"
    # 使用断言来验证给定表达式的字符串表示是否与预期相符
    assert str(Feedback(tf1, TransferFunction(1, 1, y), 1)) == \
        "Feedback(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(1, 1, y), 1)"
def test_MIMOFeedback_str():
    # 创建第一个传递函数对象 tf1，使用表达式 x**2 - y**3 作为分子，y - z 作为分母，x 作为变量
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    # 创建第二个传递函数对象 tf2，使用表达式 y - x 作为分子，z + y 作为分母，x 作为变量
    tf2 = TransferFunction(y - x, z + y, x)
    # 创建第一个传递函数矩阵 tfm_1，包含两个传递函数对象 tf1 和 tf2 的二维矩阵
    tfm_1 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    # 创建第二个传递函数矩阵 tfm_2，包含两个传递函数对象 tf1 和 tf2 的二维矩阵，顺序稍有不同
    tfm_2 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    # 验证 MIMOFeedback 对象的字符串表示是否符合预期
    assert (str(MIMOFeedback(tfm_1, tfm_2)) \
            == "MIMOFeedback(TransferFunctionMatrix(((TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x))," \
            " (TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)))), " \
            "TransferFunctionMatrix(((TransferFunction(x**2 - y**3, y - z, x), " \
            "TransferFunction(-x + y, y + z, x)), (TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)))), -1)")
    # 验证带有增益参数的 MIMOFeedback 对象的字符串表示是否符合预期
    assert (str(MIMOFeedback(tfm_1, tfm_2, 1)) \
            == "MIMOFeedback(TransferFunctionMatrix(((TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)), " \
            "(TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)))), " \
            "TransferFunctionMatrix(((TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)), "\
            "(TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)))), 1)")


def test_TransferFunctionMatrix_str():
    # 创建三个不同的传递函数对象 tf1, tf2, tf3
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    # 验证单个传递函数矩阵对象的字符串表示是否符合预期
    assert str(TransferFunctionMatrix([[tf1], [tf2]])) == \
        "TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y),), (TransferFunction(x - y, x + y, y),)))"
    # 验证包含多个传递函数对象的传递函数矩阵对象的字符串表示是否符合预期
    assert str(TransferFunctionMatrix([[tf1, tf2], [tf3, tf2]])) == \
        "TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), (TransferFunction(t*x**2 - t**w*x + w, t - y, y), TransferFunction(x - y, x + y, y))))"


def test_Quaternion_str_printer():
    # 创建四元数对象 q
    q = Quaternion(x, y, z, t)
    # 验证四元数对象的字符串表示是否符合预期
    assert str(q) == "x + y*i + z*j + t*k"
    # 创建另一个四元数对象 q，验证其字符串表示是否符合预期
    q = Quaternion(x, y, z, x*t)
    assert str(q) == "x + y*i + z*j + t*x*k"
    # 创建另一个四元数对象 q，验证其字符串表示是否符合预期
    q = Quaternion(x, y, z, x + t)
    assert str(q) == "x + y*i + z*j + (t + x)*k"


def test_Quantity_str():
    # 验证带有缩写的单位 second 的字符串表示是否符合预期
    assert sstr(second, abbrev=True) == "s"
    # 验证带有缩写的单位 joule 的字符串表示是否符合预期
    assert sstr(joule, abbrev=True) == "J"
    # 验证单位 second 的字符串表示是否符合预期
    assert str(second) == "second"
    # 验证单位 joule 的字符串表示是否符合预期
    assert str(joule) == "joule"


def test_wild_str():
    # 验证包含 Wild 的表达式的字符串表示是否符合预期，不会导致无限递归
    w = Wild('x')
    assert str(w + 1) == 'x_ + 1'
    assert str(exp(2**w) + 5) == 'exp(2**x_) + 5'
    assert str(3*w + 1) == '3*x_ + 1'
    assert str(1/w + 1) == '1 + 1/x_'
    assert str(w**2 + 1) == 'x_**2 + 1'
    assert str(1/(1 - w)) == '1/(1 - x_)'


def test_wild_matchpy():
    # 导入需要的模块和类
    from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar

    # 尝试导入 matchpy，若无法导入则返回
    matchpy = import_module("matchpy")

    if matchpy is None:
        return

    # 创建不同类型的 Wild 对象 wd, wp, ws，并验证其字符串表示是否符合预期
    wd = WildDot('w_')
    wp = WildPlus('w__')
    ws = WildStar('w___')

    assert str(wd) == 'w_'
    # 断言，验证 wp 对象的字符串表示应为 'w__'
    assert str(wp) == 'w__'
    # 断言，验证 ws 对象的字符串表示应为 'w___'
    assert str(ws) == 'w___'
    
    # 断言，验证 wp/ws + 2**wd 的字符串表示应为 '2**w_ + w__/w___'
    assert str(wp/ws + 2**wd) == '2**w_ + w__/w___'
    # 断言，验证 sin(wd)*cos(wp)*sqrt(ws) 的字符串表示应为 'sqrt(w___)*sin(w_)*cos(w__)'
    assert str(sin(wd)*cos(wp)*sqrt(ws)) == 'sqrt(w___)*sin(w_)*cos(w__)'
def test_zeta():
    # 检验 zeta(3) 的字符串表示是否等于 "zeta(3)"
    assert str(zeta(3)) == "zeta(3)"


def test_issue_3101():
    e = x - y
    a = str(e)
    b = str(e)
    # 检验两次字符串化的结果是否相等
    assert a == b


def test_issue_3103():
    e = -2*sqrt(x) - y/sqrt(x)/2
    # 检验 e 的字符串表示是否不在指定的列表中
    assert str(e) not in ["(-2)*x**1/2(-1/2)*x**(-1/2)*y",
            "-2*x**1/2(-1/2)*x**(-1/2)*y", "-2*x**1/2-1/2*x**-1/2*w"]
    # 检验 e 的字符串表示是否等于 "-2*sqrt(x) - y/(2*sqrt(x))"
    assert str(e) == "-2*sqrt(x) - y/(2*sqrt(x))"


def test_issue_4021():
    e = Integral(x, x) + 1
    # 检验 e 的字符串表示是否等于 'Integral(x, x) + 1'
    assert str(e) == 'Integral(x, x) + 1'


def test_sstrrepr():
    assert sstr('abc') == 'abc'
    assert sstrrepr('abc') == "'abc'"

    e = ['a', 'b', 'c', x]
    # 检验 e 的字符串表示是否等于 "[a, b, c, x]"
    assert sstr(e) == "[a, b, c, x]"
    # 检验 e 的字符串表示（使用 repr）是否等于 "['a', 'b', 'c', x]"
    assert sstrrepr(e) == "['a', 'b', 'c', x]"


def test_infinity():
    # 检验无穷乘虚数单位的字符串表示是否等于 "oo*I"
    assert sstr(oo*I) == "oo*I"


def test_full_prec():
    assert sstr(S("0.3"), full_prec=True) == "0.300000000000000"
    assert sstr(S("0.3"), full_prec="auto") == "0.300000000000000"
    assert sstr(S("0.3"), full_prec=False) == "0.3"
    # 检验 S("0.3")*x 的全精度字符串表示是否在给定列表中
    assert sstr(S("0.3")*x, full_prec=True) in [
        "0.300000000000000*x",
        "x*0.300000000000000"
    ]
    assert sstr(S("0.3")*x, full_prec="auto") in [
        "0.3*x",
        "x*0.3"
    ]
    assert sstr(S("0.3")*x, full_prec=False) in [
        "0.3*x",
        "x*0.3"
    ]


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    # 检验非交换符号 A*B*C**-1 的字符串表示是否等于 "A*B*C**(-1)"
    assert sstr(A*B*C**-1) == "A*B*C**(-1)"
    # 检验非交换符号 C**-1*A*B 的字符串表示是否等于 "C**(-1)*A*B"
    assert sstr(C**-1*A*B) == "C**(-1)*A*B"
    # 检验非交换符号 A*C**-1*B 的字符串表示是否等于 "A*C**(-1)*B"
    assert sstr(A*C**-1*B) == "A*C**(-1)*B"
    # 检验 sqrt(A) 的字符串表示是否等于 "sqrt(A)"
    assert sstr(sqrt(A)) == "sqrt(A)"
    # 检验 1/sqrt(A) 的字符串表示是否等于 "A**(-1/2)"
    assert sstr(1/sqrt(A)) == "A**(-1/2)"


def test_empty_printer():
    str_printer = StrPrinter()
    # 检验空打印器对字符串 "foo" 的处理结果是否等于 "foo"
    assert str_printer.emptyPrinter("foo") == "foo"
    # 检验空打印器对 x*y 的处理结果是否等于 "x*y"
    assert str_printer.emptyPrinter(x*y) == "x*y"
    # 检验空打印器对整数 32 的处理结果是否等于 "32"
    assert str_printer.emptyPrinter(32) == "32"


def test_settings():
    # 检验设置方法为 "garbage" 时是否会引发 TypeError 异常
    raises(TypeError, lambda: sstr(S(4), method="garbage"))


def test_RandomDomain():
    from sympy.stats import Normal, Die, Exponential, pspace, where
    X = Normal('x1', 0, 1)
    # 检验 X > 0 的定义域字符串表示是否等于 "Domain: (0 < x1) & (x1 < oo)"
    assert str(where(X > 0)) == "Domain: (0 < x1) & (x1 < oo)"

    D = Die('d1', 6)
    # 检验 D > 4 的定义域字符串表示是否等于 "Domain: Eq(d1, 5) | Eq(d1, 6)"
    assert str(where(D > 4)) == "Domain: Eq(d1, 5) | Eq(d1, 6)"

    A = Exponential('a', 1)
    B = Exponential('b', 1)
    # 检验 Exponential('a', 1) 和 Exponential('b', 1) 的元组的概率空间的定义域字符串表示是否等于 "Domain: (0 <= a) & (0 <= b) & (a < oo) & (b < oo)"
    assert str(pspace(Tuple(A, B)).domain) == "Domain: (0 <= a) & (0 <= b) & (a < oo) & (b < oo)"


def test_FiniteSet():
    assert str(FiniteSet(*range(1, 51))) == (
        '{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,'
        ' 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,'
        ' 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50}'
    )
    assert str(FiniteSet(*range(1, 6))) == '{1, 2, 3, 4, 5}'
    # 检验包含 x*y 和 x**2 的 FiniteSet 的字符串表示是否等于 '{x**2, x*y}'
    assert str(FiniteSet(*[x*y, x**2])) == '{x**2, x*y}'
    # 检验嵌套 FiniteSet 的字符串表示是否等于 'FiniteSet(5, FiniteSet(5, {x, y}), {x, y})'
    assert str(FiniteSet(FiniteSet(FiniteSet(x, y), 5), FiniteSet(x,y), 5)
               ) == 'FiniteSet(5, FiniteSet(5, {x, y}), {x, y})'


def test_Partition():
    # 检验 Partition(FiniteSet(x, y), {z}) 的字符串表示是否等于 'Partition({z}, {x, y})'
    assert str(Partition(FiniteSet(x, y), {z})) == 'Partition({z}, {x, y})'


def test_UniversalSet():
    # 检验 S.UniversalSet 的字符串表示是否等于 'UniversalSet'
    assert str(S.UniversalSet) == 'UniversalSet'
def test_PrettyPoly():
    F = QQ.frac_field(x, y)
    R = QQ[x, y]
    assert sstr(F.convert(x/(x + y))) == sstr(x/(x + y))
    assert sstr(R.convert(x + y)) == sstr(x + y)


def test_categories():
    from sympy.categories import (Object, NamedMorphism,
        IdentityMorphism, Category)

    # 创建两个对象 A 和 B
    A = Object("A")
    B = Object("B")

    # 创建从 A 到 B 的命名态射 f 和 A 的恒等态射
    f = NamedMorphism(A, B, "f")
    id_A = IdentityMorphism(A)

    # 创建一个分类 K
    K = Category("K")

    # 断言对象 A、态射 f 和恒等态射 id_A 的字符串表示
    assert str(A) == 'Object("A")'
    assert str(f) == 'NamedMorphism(Object("A"), Object("B"), "f")'
    assert str(id_A) == 'IdentityMorphism(Object("A"))'

    # 断言分类 K 的字符串表示
    assert str(K) == 'Category("K")'


def test_Tr():
    A, B = symbols('A B', commutative=False)
    t = Tr(A*B)
    assert str(t) == 'Tr(A*B)'


def test_issue_6387():
    assert str(factor(-3.0*z + 3)) == '-3.0*(1.0*z - 1.0)'


def test_MatMul_MatAdd():
    X, Y = MatrixSymbol("X", 2, 2), MatrixSymbol("Y", 2, 2)

    # 断言矩阵表达式的字符串表示
    assert str(2*(X + Y)) == "2*X + 2*Y"
    assert str(I*X) == "I*X"
    assert str(-I*X) == "-I*X"
    assert str((1 + I)*X) == '(1 + I)*X'
    assert str(-(1 + I)*X) == '(-1 - I)*X'
    assert str(MatAdd(MatAdd(X, Y), MatAdd(X, Y))) == '(X + Y) + (X + Y)'


def test_MatrixSlice():
    n = Symbol('n', integer=True)
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)

    # 断言矩阵切片的字符串表示
    assert str(MatrixSlice(X, (None, None, None), (None, None, None))) == 'X[:, :]'
    assert str(X[x:x + 1, y:y + 1]) == 'X[x:x + 1, y:y + 1]'
    assert str(X[x:x + 1:2, y:y + 1:2]) == 'X[x:x + 1:2, y:y + 1:2]'
    assert str(X[:x, y:]) == 'X[:x, y:]'
    assert str(X[:x, y:]) == 'X[:x, y:]'
    assert str(X[x:, :y]) == 'X[x:, :y]'
    assert str(X[x:y, z:w]) == 'X[x:y, z:w]'
    assert str(X[x:y:t, w:t:x]) == 'X[x:y:t, w:t:x]'
    assert str(X[x::y, t::w]) == 'X[x::y, t::w]'
    assert str(X[:x:y, :t:w]) == 'X[:x:y, :t:w]'
    assert str(X[::x, ::y]) == 'X[::x, ::y]'
    assert str(MatrixSlice(X, (0, None, None), (0, None, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (None, n, None), (None, n, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (0, n, None), (0, n, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (0, n, 2), (0, n, 2))) == 'X[::2, ::2]'
    assert str(X[1:2:3, 4:5:6]) == 'X[1:2:3, 4:5:6]'
    assert str(X[1:3:5, 4:6:8]) == 'X[1:3:5, 4:6:8]'
    assert str(X[1:10:2]) == 'X[1:10:2, :]'
    assert str(Y[:5, 1:9:2]) == 'Y[:5, 1:9:2]'
    assert str(Y[:5, 1:10:2]) == 'Y[:5, 1::2]'
    assert str(Y[5, :5:2]) == 'Y[5:6, :5:2]'
    assert str(X[0:1, 0:1]) == 'X[:1, :1]'
    assert str(X[0:1:2, 0:1:2]) == 'X[:1:2, :1:2]'
    assert str((Y + Z)[2:, 2:]) == '(Y + Z)[2:, 2:]'


def test_true_false():
    assert str(true) == repr(true) == sstr(true) == "True"
    assert str(false) == repr(false) == sstr(false) == "False"


def test_Equivalent():
    assert str(Equivalent(y, x)) == "Equivalent(x, y)"


def test_Xor():
    assert str(Xor(y, x, evaluate=False)) == "x ^ y"


def test_Complement():
    # 待补充
    # 断言语句，用于检查表达式是否为真
    assert str(Complement(S.Reals, S.Naturals)) == 'Complement(Reals, Naturals)'
    # 上述断言验证是否计算得到的字符串表示与预期的字符串 'Complement(Reals, Naturals)' 相等
def test_SymmetricDifference():
    # 断言测试 SymmetricDifference 函数，检查其返回的字符串表示是否符合预期
    assert str(SymmetricDifference(Interval(2, 3), Interval(3, 4), evaluate=False)) == \
           'SymmetricDifference(Interval(2, 3), Interval(3, 4))'


def test_UnevaluatedExpr():
    # 定义符号变量 a, b
    a, b = symbols("a b")
    # 创建一个未评估表达式，乘以2
    expr1 = 2 * UnevaluatedExpr(a + b)
    # 断言表达式的字符串表示是否正确
    assert str(expr1) == "2*(a + b)"


def test_MatrixElement_printing():
    # 为问题 #11821 编写的测试用例
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    # 断言矩阵元素的字符串表示是否正确
    assert(str(A[0, 0]) == "A[0, 0]")
    assert(str(3 * A[0, 0]) == "3*A[0, 0]")

    # 对 C[0, 0] 应用 C 替换为 A - B 的表达式
    F = C[0, 0].subs(C, A - B)
    # 断言表达式的字符串表示是否正确
    assert str(F) == "(A - B)[0, 0]"


def test_MatrixSymbol_printing():
    # 创建 3x3 的矩阵符号 A, B
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)

    # 断言矩阵表达式的字符串表示是否正确
    assert str(A - A*B - B) == "A - A*B - B"
    assert str(A*B - (A+B)) == "-A + A*B - B"
    assert str(A**(-1)) == "A**(-1)"
    assert str(A**3) == "A**3"


def test_MatrixExpressions():
    n = Symbol('n', integer=True)
    X = MatrixSymbol('X', n, n)

    # 断言矩阵符号 X 的字符串表示是否正确
    assert str(X) == "X"

    # 对 X.T*X 应用 sin 函数，元素级应用
    expr = (X.T*X).applyfunc(sin)
    # 断言表达式的字符串表示是否正确
    assert str(expr) == 'Lambda(_d, sin(_d)).(X.T*X)'

    # 创建 Lambda 函数 lamda(x, 1/x)
    lamda = Lambda(x, 1/x)
    # 对 n*X 应用元素级应用
    expr = (n*X).applyfunc(lamda)
    # 断言表达式的字符串表示是否正确
    assert str(expr) == 'Lambda(x, 1/x).(n*X)'


def test_Subs_printing():
    # 断言 Subs 对象的字符串表示是否正确
    assert str(Subs(x, (x,), (1,))) == 'Subs(x, x, 1)'
    assert str(Subs(x + y, (x, y), (1, 2))) == 'Subs(x + y, (x, y), (1, 2))'


def test_issue_15716():
    # 创建阶乘表达式的积分
    e = Integral(factorial(x), (x, -oo, oo))
    # 断言其作为项的表示是否正确
    assert e.as_terms() == ([(e, ((1.0, 0.0), (1,), ()))], [e])


def test_str_special_matrices():
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    # 断言特殊矩阵的字符串表示是否正确
    assert str(Identity(4)) == 'I'
    assert str(ZeroMatrix(2, 2)) == '0'
    assert str(OneMatrix(2, 2)) == '1'


def test_issue_14567():
    # 断言对于给定表达式不会引发错误
    assert factorial(Sum(-1, (x, 0, 0))) + y  # doesn't raise an error


def test_issue_21823():
    # 断言 Partition 对象的字符串表示是否正确
    assert str(Partition([1, 2])) == 'Partition({1, 2})'
    assert str(Partition({1, 2})) == 'Partition({1, 2})'


def test_issue_22689():
    # 断言 Mul 对象的字符串表示是否正确
    assert str(Mul(Pow(x,-2, evaluate=False), Pow(3,-1,evaluate=False), evaluate=False)) == "1/(x**2*3)"


def test_issue_21119_21460():
    ss = lambda x: str(S(x, evaluate=False))
    # 断言字符串表示是否符合预期
    assert ss('4/2') == '4/2'
    assert ss('4/-2') == '4/(-2)'
    assert ss('-4/2') == '-4/2'
    assert ss('-4/-2') == '-4/(-2)'
    assert ss('-2*3/-1') == '-2*3/(-1)'
    assert ss('-2*3/-1/2') == '-2*3/(-1*2)'
    assert ss('4/2/1') == '4/(2*1)'
    assert ss('-2/-1/2') == '-2/(-1*2)'
    assert ss('2*3*4**(-2*3)') == '2*3/4**(2*3)'
    assert ss('2*3*1*4**(-2*3)') == '2*3*1/4**(2*3)'


def test_Str():
    from sympy.core.symbol import Str
    # 断言 Str 对象的字符串表示是否符合预期
    assert str(Str('x')) == 'x'
    assert sstrrepr(Str('x')) == "Str('x')"


def test_diffgeom():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
    x,y = symbols('x y', real=True)
    # 创建流形对象 M
    m = Manifold('M', 2)
    # 断言其字符串表示是否正确
    assert str(m) == "M"
    # 创建 patch 对象 P 绑定到流形 m
    p = Patch('P', m)
    # 断言其字符串表示是否正确
    assert str(p) == "P"
    # 创建一个名为 rect 的坐标系对象，类型为 'rect'，位置为 p，坐标轴范围为 [x, y]
    rect = CoordSystem('rect', p, [x, y])
    
    # 断言 rect 对象的字符串表示应为 "rect"
    assert str(rect) == "rect"
    
    # 使用 rect 坐标系创建一个基本标量场对象 b，初始值为 0
    b = BaseScalarField(rect, 0)
    
    # 断言 b 对象的字符串表示应为 "x"
    assert str(b) == "x"
def test_printing_stats():
    # 定义随机符号 x 和 y
    x = RandomSymbol("x")
    y = RandomSymbol("y")
    # 创建 z1-z4，分别为表达式 Probability(x > 0)*Identity(2)，Expectation(x)*Identity(2)，Variance(x)*Identity(2)，Covariance(x, y)*Identity(2)
    z1 = Probability(x > 0)*Identity(2)
    z2 = Expectation(x)*Identity(2)
    z3 = Variance(x)*Identity(2)
    z4 = Covariance(x, y) * Identity(2)

    # 断言 z1-z4 的字符串表示
    assert str(z1) == "Probability(x > 0)*I"
    assert str(z2) == "Expectation(x)*I"
    assert str(z3) == "Variance(x)*I"
    assert str(z4) == "Covariance(x, y)*I"
    # 断言 z1-z4 不是可交换的
    assert z1.is_commutative == False
    assert z2.is_commutative == False
    assert z3.is_commutative == False
    assert z4.is_commutative == False
    # 断言 z2-z4 的可交换性评估结果均为 False
    assert z2._eval_is_commutative() == False
    assert z3._eval_is_commutative() == False
    assert z4._eval_is_commutative() == False
```