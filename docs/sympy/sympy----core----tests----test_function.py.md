# `D:\src\scipysrc\sympy\sympy\core\tests\test_function.py`

```
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于表示求和
from sympy.core.basic import Basic, _aresame  # 导入 Basic 类和 _aresame 函数
from sympy.core.cache import clear_cache  # 导入 clear_cache 函数，用于清除缓存
from sympy.core.containers import Dict, Tuple  # 导入 Dict 和 Tuple 类，用于容器操作
from sympy.core.expr import Expr, unchanged  # 导入 Expr 类和 unchanged 函数
from sympy.core.function import (Subs, Function, diff, Lambda, expand,  # 导入多个函数和类
    nfloat, Derivative)
from sympy.core.numbers import E, Float, zoo, Rational, pi, I, oo, nan  # 导入各种数学常数和特殊值
from sympy.core.power import Pow  # 导入 Pow 类，用于表示幂运算
from sympy.core.relational import Eq  # 导入 Eq 类，用于表示等式
from sympy.core.singleton import S  # 导入 S 类，表示 SymPy 的单例
from sympy.core.symbol import symbols, Dummy, Symbol  # 导入符号相关的类和函数
from sympy.functions.elementary.complexes import im, re  # 导入虚部和实部函数
from sympy.functions.elementary.exponential import log, exp  # 导入对数和指数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数类
from sympy.functions.elementary.trigonometric import sin, cos, acos  # 导入三角函数及反三角函数
from sympy.functions.special.error_functions import expint  # 导入指数积分函数
from sympy.functions.special.gamma_functions import loggamma, polygamma  # 导入 Gamma 函数相关
from sympy.matrices.dense import Matrix  # 导入密集矩阵类
from sympy.printing.str import sstr  # 导入 sstr 函数，用于生成可打印字符串表示
from sympy.series.order import O  # 导入 O 类，用于表示高阶无穷小
from sympy.tensor.indexed import Indexed  # 导入 Indexed 类，用于表示张量索引
from sympy.core.function import (PoleError, _mexpand, arity,  # 导入多个函数和类
        BadSignatureError, BadArgumentsError)
from sympy.core.parameters import _exp_is_pow  # 导入 _exp_is_pow 参数
from sympy.core.sympify import sympify, SympifyError  # 导入 sympify 函数及其异常
from sympy.matrices import MutableMatrix, ImmutableMatrix  # 导入可变和不可变矩阵类
from sympy.sets.sets import FiniteSet  # 导入有限集类
from sympy.solvers.solveset import solveset  # 导入 solveset 函数，用于求解方程
from sympy.tensor.array import NDimArray  # 导入多维数组类
from sympy.utilities.iterables import subsets, variations  # 导入子集和排列组合函数
from sympy.testing.pytest import XFAIL, raises, warns_deprecated_sympy, _both_exp_pow  # 导入测试相关函数

from sympy.abc import t, w, x, y, z  # 导入符号变量 t, w, x, y, z
f, g, h = symbols('f g h', cls=Function)  # 定义符号函数 f, g, h
_xi_1, _xi_2, _xi_3 = [Dummy() for i in range(3)]  # 创建三个虚拟变量对象

def test_f_expand_complex():
    x = Symbol('x', real=True)  # 创建一个名为 x 的实数符号变量

    assert f(x).expand(complex=True) == I*im(f(x)) + re(f(x))  # 断言 f(x) 在复数域下的展开结果
    assert exp(x).expand(complex=True) == exp(x)  # 断言 exp(x) 在复数域下的展开结果
    assert exp(I*x).expand(complex=True) == cos(x) + I*sin(x)  # 断言 exp(I*x) 在复数域下的展开结果
    assert exp(z).expand(complex=True) == cos(im(z))*exp(re(z)) + \
        I*sin(im(z))*exp(re(z))  # 断言 exp(z) 在复数域下的展开结果

def test_bug1():
    e = sqrt(-log(w))  # 创建表达式 e = sqrt(-log(w))
    assert e.subs(log(w), -x) == sqrt(x)  # 断言替换 log(w) 为 -x 后的结果为 sqrt(x)

    e = sqrt(-5*log(w))  # 创建表达式 e = sqrt(-5*log(w))
    assert e.subs(log(w), -x) == sqrt(5*x)  # 断言替换 log(w) 为 -x 后的结果为 sqrt(5*x)

def test_general_function():
    nu = Function('nu')  # 创建函数符号 nu

    e = nu(x)  # 创建表达式 e = nu(x)
    edx = e.diff(x)  # 求 e 对 x 的导数
    edy = e.diff(y)  # 求 e 对 y 的导数
    edxdx = e.diff(x).diff(x)  # 求 e 对 x 的二阶导数
    edxdy = e.diff(x).diff(y)  # 求 e 先对 x 后对 y 的混合导数
    assert e == nu(x)  # 断言 e 等于 nu(x)
    assert edx != nu(x)  # 断言 e 对 x 的导数不等于 nu(x)
    assert edx == diff(nu(x), x)  # 断言 e 对 x 的导数等于 diff(nu(x), x)
    assert edy == 0  # 断言 e 对 y 的导数为 0
    assert edxdx == diff(diff(nu(x), x), x)  # 断言 e 对 x 的二阶导数等于 diff(diff(nu(x), x), x)
    assert edxdy == 0  # 断言 e 对 x 后对 y 的混合导数为 0

def test_general_function_nullary():
    nu = Function('nu')  # 创建函数符号 nu

    e = nu()  # 创建表达式 e = nu()
    edx = e.diff(x)  # 求 e 对 x 的导数
    edxdx = e.diff(x).diff(x)  # 求 e 对 x 的二阶导数
    assert e == nu()  # 断言 e 等于 nu()
    assert edx != nu()  # 断言 e 对 x 的导数不等于 nu()
    assert edx == 0  # 断言 e 对 x 的导数为 0
    assert edxdx == 0  # 断言 e 对 x 的二阶导数为 0

def test_derivative_subs_bug():
    e = diff(g(x), x)  # 创建 g(x) 对 x 的导数
    assert e.subs(g(x), f(x)) != e  # 断言将 g(x) 替换为 f(x) 后的结果不等于 e
    assert e.subs(g(x), f(x)) == Derivative(f(x), x)  # 断言将 g(x) 替换为 f(x) 后的结果等于 Derivative(f(x), x)
    # 断言：验证 e.subs(g(x), -f(x)) 的结果是否等于 Derivative(-f(x), x)
    assert e.subs(g(x), -f(x)) == Derivative(-f(x), x)
    
    # 断言：验证 e.subs(x, y) 的结果是否等于 Derivative(g(y), y)
    assert e.subs(x, y) == Derivative(g(y), y)
# 定义一个测试函数，用于测试微分运算中的自身替换错误
def test_derivative_subs_self_bug():
    # 计算 f(x) 对 x 的导数
    d = diff(f(x), x)

    # 断言对 d 进行自身替换为 y 后应该得到 y
    assert d.subs(d, y) == y


# 定义一个测试函数，用于测试微分的线性性质
def test_derivative_linearity():
    # 断言 -f(x) 对 x 的导数等于 -f(x) 的导数
    assert diff(-f(x), x) == -diff(f(x), x)
    # 断言 8*f(x) 对 x 的导数等于 8 乘以 f(x) 的导数
    assert diff(8*f(x), x) == 8*diff(f(x), x)
    # 断言 8*f(x) 对 x 的导数不等于 7 乘以 f(x) 的导数
    assert diff(8*f(x), x) != 7*diff(f(x), x)
    # 断言 8*f(x)*x 对 x 的导数等于 8*f(x) 加上 8*x 乘以 f(x) 的导数
    assert diff(8*f(x)*x, x) == 8*f(x) + 8*x*diff(f(x), x)
    # 断言 8*f(x)*y*x 对 x 的导数在展开后等于 8*y*f(x) 加上 8*y*x 乘以 f(x) 的导数
    assert diff(8*f(x)*y*x, x).expand() == 8*y*f(x) + 8*y*x*diff(f(x), x)


# 定义一个测试函数，用于测试微分的求值行为
def test_derivative_evaluate():
    # 断言使用 Derivative 类来表示 sin(x) 对 x 的导数不等于 diff 函数直接计算的结果
    assert Derivative(sin(x), x) != diff(sin(x), x)
    # 断言使用 Derivative 类来计算 sin(x) 对 x 的导数并化简得到 diff 函数的结果
    assert Derivative(sin(x), x).doit() == diff(sin(x), x)

    # 断言 Derivative 类对 f(x) 连续求两次导数等于 diff 函数直接计算的结果
    assert Derivative(Derivative(f(x), x), x) == diff(f(x), x, x)
    # 断言 Derivative 类对 sin(x) 连续求导 0 次等于 sin(x)
    assert Derivative(sin(x), x, 0) == sin(x)
    # 断言 Derivative 类对 sin(x) 在 (x, y) 和 (x, -y) 处求导等于 sin(x)
    assert Derivative(sin(x), (x, y), (x, -y)) == sin(x)


# 定义一个测试函数，用于测试符号变量的多阶偏导数计算
def test_diff_symbols():
    # 断言计算 f(x, y, z) 对 x, y, z 三个变量的偏导数结果与 Derivative 类计算结果相等
    assert diff(f(x, y, z), x, y, z) == Derivative(f(x, y, z), x, y, z)
    # 断言计算 f(x, y, z) 对 x 连续求三次导数的结果与 Derivative 类计算结果相等，并且都等于 Derivative(f(x, y, z), (x, 3))
    assert diff(f(x, y, z), x, x, x) == Derivative(f(x, y, z), x, x, x) == Derivative(f(x, y, z), (x, 3))
    # 断言计算 f(x, y, z) 对 x 连续求三次导数的结果与 Derivative 类计算结果相等
    assert diff(f(x, y, z), x, 3) == Derivative(f(x, y, z), x, 3)

    # issue 5028 的测试案例
    # 断言对表达式 -z + x/y 求 z, x, y 三个变量的偏导数分别得到 [-1, 1/y, -x/y**2] 的结果
    assert [diff(-z + x/y, sym) for sym in (z, x, y)] == [-1, 1/y, -x/y**2]
    # 断言计算 f(x, y, z) 对 x, y, z 连续求三次导数的结果与 Derivative 类计算结果相等，并且都等于 Derivative(f(x, y, z), x, y, z, z)
    assert diff(f(x, y, z), x, y, z, 2) == Derivative(f(x, y, z), x, y, z, z)
    # 断言计算 f(x, y, z) 对 x, y, z 连续求三次导数的结果与 Derivative 类计算结果相等，且评估标记为 False
    assert diff(f(x, y, z), x, y, z, 2, evaluate=False) == Derivative(f(x, y, z), x, y, z, z)
    # 断言对 Derivative 类对象应用 _eval_derivative 方法得到 Derivative(f(x, y, z), x, y, z, z) 的结果
    assert Derivative(f(x, y, z), x, y, z)._eval_derivative(z) == Derivative(f(x, y, z), x, y, z, z)
    # 断言对 Derivative 类对象连续求两次导数并应用 _eval_derivative 方法得到 Derivative(f(x, y, z), x, y, z, z) 的结果
    assert Derivative(Derivative(f(x, y, z), x), y)._eval_derivative(z) == Derivative(f(x, y, z), x, y, z)

    # 验证是否抛出 TypeError 异常，检查 cos(x).diff((x, y)).variables 是否有 variables 属性
    raises(TypeError, lambda: cos(x).diff((x, y)).variables)
    # 断言 cos(x).diff((x, y))._wrt_variables 结果为 [x]
    assert cos(x).diff((x, y))._wrt_variables == [x]

    # issue 23222 的测试案例
    # 断言 sympify("a*x+b") 对 "x" 求导数得到 sympify("a")
    assert sympify("a*x+b").diff("x") == sympify("a")


# 定义一个测试函数，用于测试自定义函数类中的 nargs 属性
def test_Function():
    # 定义一个无参数的自定义函数类 myfunc
    class myfunc(Function):
        @classmethod
        def eval(cls):  # 零个参数的情况
            return

    # 断言无参数的 myfunc.nargs 是 FiniteSet(0)
    assert myfunc.nargs == FiniteSet(0)
    # 断言调用无参数的 myfunc 实例的 nargs 也是 FiniteSet(0)
    assert myfunc().nargs == FiniteSet(0)
    # 断言带有参数的 myfunc 实例调用 nargs 会引发 TypeError 异常
    raises(TypeError, lambda: myfunc(x).nargs)

    # 定义一个带有一个参数的自定义函数类 myfunc
    class myfunc(Function):
        @classmethod
        def eval(cls, x):  # 一个参数的情况
            return

    # 断言一个参数的 myfunc.nargs 是 FiniteSet(1)
    assert myfunc.nargs == FiniteSet(1)
    # 断言带有一个参数的 myfunc 实例的 nargs 也是 FiniteSet(1)
    assert myfunc(x).nargs == FiniteSet(1)
    # 断言带有两个参数的 myfunc 实例调用 nargs 会引发 TypeError 异常
    raises(TypeError, lambda: myfunc(x, y).nargs)

    # 定义一个带有可变参数的自定义函数类 myfunc
    class myfunc(Function):
        @classmethod
        def eval(cls, *x):  # 可变参数的情况
            return

    # 断言可变参数的 myfunc.nargs 是 S.Naturals0
    assert myfunc.nargs == S.Naturals0
    # 断言带有一个参数的 myfunc 实例的 nargs 也是 S.Naturals0
    assert myfunc(x).nargs == S.Naturals0


# 定义一个测试函数，用于测试内置函数的 nargs
def test_nargs_inheritance():
    # 定义多个函数类，演示 nargs 属性的继承和设置
    class f1(Function):
        nargs = 2
    class f2(f1):
        pass
    class f3(f2):
        pass
    class f4(f3):
        nargs = 1,2  # 设置 nargs 属性为元组 (1, 2)
    class f5(f4):
        pass
    class f6(f5):
        pass
    class f7(f6):
        nargs=None  # 设置 nargs 属性为 None
    class f8(f7):
        pass
    class f9(f8):
        pass
    class f10(f9):
        nargs = 1  # 设置 nargs 属性为 1
    class f11(f10):
        pass
    # 对各个函数类的 nargs 属性进行断言验证
    assert f1.nargs == FiniteSet(2)
    assert f2.nargs == FiniteSet(2)
    assert f3.nargs == FiniteSet(2)
    assert f4.nargs == FiniteSet(1, 2)
    assert f5.nargs == FiniteSet(1, 2)
    assert f6.nargs == FiniteSet(1, 2)
    assert f7.nargs == S.Naturals0  # 断言 nargs 属性为自然数集合 S.Naturals0
    assert f8.nargs == S.Naturals0
    assert f9.nargs == S.Naturals0
    assert f10.nargs == FiniteSet(1)
    assert f11.nargs == FiniteSet(1)

def test_arity():
    # 测试 arity 函数的不同用例
    f = lambda x, y: 1
    assert arity(f) == 2  # 断言 f 的参数个数为 2
    def f(x, y, z=None):
        pass
    assert arity(f) == (2, 3)  # 断言 f 的参数个数为 (2, 3)
    assert arity(lambda *x: x) is None  # 断言匿名函数的参数个数为 None
    assert arity(log) == (1, 2)  # 断言 log 函数的参数个数为 (1, 2)

def test_Lambda():
    # 测试 Lambda 类的各种功能和方法
    e = Lambda(x, x**2)
    assert e(4) == 16  # 断言 Lambda 函数对参数 4 计算结果为 16
    assert e(x) == x**2  # 断言 Lambda 函数对参数 x 计算结果为 x**2
    assert e(y) == y**2  # 断言 Lambda 函数对参数 y 计算结果为 y**2

    assert Lambda((), 42)() == 42  # 断言 Lambda 函数无参数时返回 42
    assert unchanged(Lambda, (), 42)  # 断言 unchanged 函数对 Lambda 与 ((), 42) 参数计算结果不变
    assert Lambda((), 42) != Lambda((), 43)  # 断言两个 Lambda 对象不相等
    assert Lambda((), f(x))() == f(x)  # 断言 Lambda 函数对 f(x) 的计算结果为 f(x)
    assert Lambda((), 42).nargs == FiniteSet(0)  # 断言 Lambda 函数的 nargs 属性为 {0}

    assert unchanged(Lambda, (x,), x**2)  # 断言 unchanged 函数对 Lambda 与 (x,) 参数计算结果不变
    assert Lambda(x, x**2) == Lambda((x,), x**2)  # 断言两种不同 Lambda 对象相等
    assert Lambda(x, x**2) != Lambda(x, x**2 + 1)  # 断言两个 Lambda 对象不相等
    assert Lambda((x, y), x**y) != Lambda((y, x), y**x)  # 断言两个 Lambda 对象不相等
    assert Lambda((x, y), x**y) != Lambda((x, y), y**x)  # 断言两个 Lambda 对象不相等

    assert Lambda((x, y), x**y)(x, y) == x**y  # 断言 Lambda 函数对 (x, y) 计算结果为 x**y
    assert Lambda((x, y), x**y)(3, 3) == 3**3  # 断言 Lambda 函数对 (3, 3) 计算结果为 3**3
    assert Lambda((x, y), x**y)(x, 3) == x**3  # 断言 Lambda 函数对 (x, 3) 计算结果为 x**3
    assert Lambda((x, y), x**y)(3, y) == 3**y  # 断言 Lambda 函数对 (3, y) 计算结果为 3**y
    assert Lambda(x, f(x))(x) == f(x)  # 断言 Lambda 函数对 f(x) 计算结果为 f(x)
    assert Lambda(x, x**2)(e(x)) == x**4  # 断言 Lambda 函数对 e(x) 的计算结果为 x**4
    assert e(e(x)) == x**4  # 断言对 e(e(x)) 的计算结果为 x**4

    x1, x2 = (Indexed('x', i) for i in (1, 2))
    assert Lambda((x1, x2), x1 + x2)(x, y) == x + y  # 断言 Lambda 函数对 (x, y) 计算结果为 x + y

    assert Lambda((x, y), x + y).nargs == FiniteSet(2)  # 断言 Lambda 函数的 nargs 属性为 {2}

    p = x, y, z, t
    assert Lambda(p, t*(x + y + z))(*p) == t * (x + y + z)  # 断言 Lambda 函数对 p 的计算结果为 t * (x + y + z)

    eq = Lambda(x, 2*x) + Lambda(y, 2*y)
    assert eq != 2*Lambda(x, 2*x)  # 断言 Lambda 函数组合后不等于单独计算的结果
    assert eq.as_dummy() == 2*Lambda(x, 2*x).as_dummy()  # 断言 Lambda 函数 as_dummy 后相等
    assert Lambda(x, 2*x) not in [ Lambda(x, x) ]  # 断言 Lambda 函数不在列表中
    raises(BadSignatureError, lambda: Lambda(1, x))  # 断言 Lambda 函数参数类型错误会引发 BadSignatureError
    assert Lambda(x, 1)(1) is S.One  # 断言 Lambda 函数对参数 1 计算结果为 S.One

    raises(BadSignatureError, lambda: Lambda((x, x), x + 2))  # 断言 Lambda 函数参数错误会引发 BadSignatureError
    raises(BadSignatureError, lambda: Lambda(((x, x), y), x))  # 断言 Lambda 函数参数错误会引发 BadSignatureError
    raises(BadSignatureError, lambda: Lambda(((y, x), x), x))  # 断言 Lambda 函数参数错误会引发 BadSignatureError
    raises(BadSignatureError, lambda: Lambda(((y, 1), 2), x))  # 断言 Lambda 函数参数错误会引发 BadSignatureError

    with warns_deprecated_sympy():
        assert Lambda([x, y], x+y) == Lambda((x, y), x+y)

    flam = Lambda(((x, y),), x + y)
    assert flam((2, 3)) == 5  # 断言 Lambda 函数对 (2, 3) 计算结果为 5
    flam = Lambda(((x, y), z), x + y + z)
    assert flam((2, 3), 1) == 6  # 断言 Lambda 函数对 ((2, 3), 1) 计算结果为 6
    flam = Lambda((((x, y), z),), x + y + z)
    assert flam(((2, 3), 1)) == 6  # 断言 Lambda 函数对 (((2, 3), 1)) 计算结果为 6
    raises(BadArgumentsError, lambda: flam(1, 2, 3))  # 断言 Lambda 函数参数错误会引发 BadArgumentsError
    # 创建一个 Lambda 函数对象，接收一个参数 x，返回元组 (x, x)
    flam = Lambda( (x,), (x, x))
    # 断言 Lambda 函数对参数 (1,) 的调用结果为 ((1,), (1,))
    assert flam(1,) == (1, 1)
    # 断言 Lambda 函数对参数 ((1,),) 的调用结果为 (((1,),), ((1,),))
    assert flam((1,)) == ((1,), (1,))

    # 创建另一个 Lambda 函数对象，接收一个参数 (x,)，返回元组 (x, x)
    flam = Lambda( ((x,),), (x, x))
    # 断言 Lambda 函数对参数 1 的调用会引发 BadArgumentsError 异常
    raises(BadArgumentsError, lambda: flam(1))
    # 断言 Lambda 函数对参数 ((1,),) 的调用结果为 (1, 1)
    assert flam((1,)) == (1, 1)

    # 断言 BadSignatureError 是 TypeError 的子类，用于向后兼容性
    assert issubclass(BadSignatureError, TypeError)
    # 断言 BadArgumentsError 是 TypeError 的子类，用于向后兼容性
    assert issubclass(BadArgumentsError, TypeError)

    # 测试 Lambda 函数的哈希值，确保不引发异常
    hash(Lambda(x, 2*x))
    hash(Lambda(x, x))  # IdentityFunction 的子类
# 测试Lambda函数是否返回IdentityFunction
def test_IdentityFunction():
    assert Lambda(x, x) is Lambda(y, y) is S.IdentityFunction
    assert Lambda(x, 2*x) is not S.IdentityFunction
    assert Lambda((x, y), x) is not S.IdentityFunction


# 测试Lambda函数的free_symbols属性
def test_Lambda_symbols():
    assert Lambda(x, 2*x).free_symbols == set()
    assert Lambda(x, x*y).free_symbols == {y}
    assert Lambda((), 42).free_symbols == set()
    assert Lambda((), x*y).free_symbols == {x, y}


# 测试函数对象f的free_symbols属性
def test_functionclas_symbols():
    assert f.free_symbols == set()


# 测试Lambda函数的参数传递是否引发TypeError异常
def test_Lambda_arguments():
    raises(TypeError, lambda: Lambda(x, 2*x)(x, y))
    raises(TypeError, lambda: Lambda((x, y), x + y)(x))
    raises(TypeError, lambda: Lambda((), 42)(x))


# 测试Lambda函数的相等性
def test_Lambda_equality():
    assert Lambda((x, y), 2*x) == Lambda((x, y), 2*x)
    # 以下两个Lambda对象显然不相等
    assert Lambda(x, 2*x) != Lambda((x, y), 2*x)
    assert Lambda(x, 2*x) != 2*x
    # 虽然有时我们希望不同绑定符号的表达式比较相同，
    # 但Python的`==`操作符不会这样做；
    # 相等的两个对象意味着它们是不可区分的且缓存到相同的值。
    # 我们不希望在数学上相同但变量名不同的表达式可以互换，
    # 否则为什么要允许不同的变量名呢？
    assert Lambda(x, 2*x) != Lambda(y, 2*y)


# 测试Subs类的不同功能
def test_Subs():
    assert Subs(1, (), ()) is S.One
    # 检查空替换对哈希值的影响
    assert Subs(x, y, z) != Subs(x, y, 1)
    # 空替换有效
    assert Subs(x, x, 1).subs(x, y).has(y)
    # 自我映射变量/点
    assert Subs(Derivative(f(x), (x, 2)), x, x).doit() == f(x).diff(x, x)
    assert Subs(x, x, 0).has(x)  # 这是一个结构性答案
    assert not Subs(x, x, 0).free_symbols
    assert Subs(Subs(x + y, x, 2), y, 1) == Subs(x + y, (x, y), (2, 1))
    assert Subs(x, (x,), (0,)) == Subs(x, x, 0)
    assert Subs(x, x, 0) == Subs(y, y, 0)
    assert Subs(x, x, 0).subs(x, 1) == Subs(x, x, 0)
    assert Subs(y, x, 0).subs(y, 1) == Subs(1, x, 0)
    assert Subs(f(x), x, 0).doit() == f(0)
    assert Subs(f(x**2), x**2, 0).doit() == f(0)
    assert Subs(f(x, y, z), (x, y, z), (0, 1, 1)) != \
        Subs(f(x, y, z), (x, y, z), (0, 0, 1))
    assert Subs(x, y, 2).subs(x, y).doit() == 2
    assert Subs(f(x, y), (x, y, z), (0, 1, 1)) != \
        Subs(f(x, y) + z, (x, y, z), (0, 1, 0))
    assert Subs(f(x, y), (x, y), (0, 1)).doit() == f(0, 1)
    assert Subs(Subs(f(x, y), x, 0), y, 1).doit() == f(0, 1)
    raises(ValueError, lambda: Subs(f(x, y), (x, y), (0, 0, 1)))
    raises(ValueError, lambda: Subs(f(x, y), (x, x, y), (0, 0, 1)))

    assert len(Subs(f(x, y), (x, y), (0, 1)).variables) == 2
    assert Subs(f(x, y), (x, y), (0, 1)).point == Tuple(0, 1)

    assert Subs(f(x), x, 0) == Subs(f(y), y, 0)
    assert Subs(f(x, y), (x, y), (0, 1)) == Subs(f(x, y), (y, x), (1, 0))
    # 断言：使用 Subs 类进行符号替换，验证两个表达式是否相等
    assert Subs(f(x)*y, (x, y), (0, 1)) == Subs(f(y)*x, (y, x), (0, 1))
    # 断言：使用 Subs 类进行符号替换，验证两个表达式是否相等
    assert Subs(f(x)*y, (x, y), (1, 1)) == Subs(f(y)*x, (x, y), (1, 1))

    # 断言：对 Subs 类进行连续替换和计算，验证结果是否等于 f(0)
    assert Subs(f(x), x, 0).subs(x, 1).doit() == f(0)
    # 断言：使用 Subs 类替换符号并验证结果是否与给定的 Subs 对象相等
    assert Subs(f(x), x, y).subs(y, 0) == Subs(f(x), x, 0)
    # 断言：对 Subs 类进行符号替换，验证结果是否与预期的 Subs 对象相等
    assert Subs(y*f(x), x, y).subs(y, 2) == Subs(2*f(x), x, 2)
    # 断言：使用 Subs 类进行符号替换和乘法操作，验证结果是否等于给定的乘积
    assert (2 * Subs(f(x), x, 0)).subs(Subs(f(x), x, 0), y) == 2*y

    # 断言：验证 Subs 类对象的 free_symbols 属性是否为空集合
    assert Subs(f(x), x, 0).free_symbols == set()
    # 断言：验证 Subs 类对象的 free_symbols 属性是否包含指定的符号集合
    assert Subs(f(x, y), x, z).free_symbols == {y, z}

    # 断言：使用 Subs 类计算导数并进行计算，验证结果是否正确
    assert Subs(f(x).diff(x), x, 0).doit(), Subs(f(x).diff(x), x, 0)
    # 断言：使用 Subs 类进行符号替换和加法操作，验证结果是否正确
    assert Subs(1 + f(x).diff(x), x, 0).doit(), 1 + Subs(f(x).diff(x), x, 0)
    # 断言：使用 Subs 类进行符号替换和导数计算，验证结果是否等于给定表达式
    assert Subs(y*f(x, y).diff(x), (x, y), (0, 2)).doit() == \
        2*Subs(Derivative(f(x, 2), x), x, 0)
    # 断言：使用 Subs 类进行符号替换和导数计算，验证结果是否等于给定表达式
    assert Subs(y**2*f(x), x, 0).diff(y) == 2*y*f(0)

    # 创建 Subs 类对象并赋值给变量 e
    e = Subs(y**2*f(x), x, y)
    # 断言：验证 Subs 类对象的导数计算是否正确
    assert e.diff(y) == e.doit().diff(y) == y**2*Derivative(f(y), y) + 2*y*f(y)

    # 断言：使用 Subs 类进行加法操作，验证结果是否等于两倍的第一个表达式
    assert Subs(f(x), x, 0) + Subs(f(x), x, 0) == 2*Subs(f(x), x, 0)
    # 创建 Subs 类对象并赋值给变量 e1 和 e2
    e1 = Subs(z*f(x), x, 1)
    e2 = Subs(z*f(y), y, 1)
    # 断言：使用 Subs 类进行加法操作，验证结果是否等于两倍的第一个对象
    assert e1 + e2 == 2*e1
    # 断言：验证两个 Subs 类对象的哈希值是否相等
    assert e1.__hash__() == e2.__hash__()
    # 断言：验证 Subs 类对象是否不在给定列表中
    assert Subs(z*f(x + 1), x, 1) not in [ e1, e2 ]
    # 断言：使用 Subs 类进行符号替换，验证结果是否等于预期的导数替换
    assert Derivative(f(x), x).subs(x, g(x)) == Derivative(f(g(x)), g(x))
    # 断言：使用 Subs 类进行符号替换，验证结果是否等于预期的 Subs 对象
    assert Derivative(f(x), x).subs(x, x + y) == Subs(Derivative(f(x), x),
        x, x + y)
    # 断言：使用 Subs 类进行符号替换和数值计算，验证结果是否等于给定的数值
    assert Subs(f(x)*cos(y) + z, (x, y), (0, pi/3)).n(2) == \
        Subs(f(x)*cos(y) + z, (x, y), (0, pi/3)).evalf(2) == \
        z + Rational('1/2').n(2)*f(0)

    # 断言：使用 Subs 类进行导数计算和符号替换，验证结果是否等于预期的导数替换
    assert f(x).diff(x).subs(x, 0).subs(x, y) == f(x).diff(x).subs(x, 0)
    # 断言：使用 Subs 类进行符号替换和乘法操作，验证结果是否等于预期的乘积
    assert (x*f(x).diff(x).subs(x, 0)).subs(x, y) == y*f(x).diff(x).subs(x, 0)
    # 断言：使用 Subs 类进行导数计算和符号替换，验证结果是否等于预期的导数替换
    assert Subs(Derivative(g(x)**2, g(x), x), g(x), exp(x)
        ).doit() == 2*exp(x)
    # 断言：使用 Subs 类进行导数计算和符号替换，验证结果是否等于预期的导数替换
    assert Subs(Derivative(g(x)**2, g(x), x), g(x), exp(x)
        ).doit(deep=False) == 2*Derivative(exp(x), x)
    # 断言：使用 Subs 类进行导数计算和符号替换，验证结果是否等于预期的导数替换
    assert Derivative(f(x, g(x)), x).doit() == Derivative(
        f(x, g(x)), g(x))*Derivative(g(x), x) + Subs(Derivative(
        f(y, g(x)), y), y, x)
def test_doitdoit():
    # 对 Derivative 对象应用 doit() 方法，计算其导数
    done = Derivative(f(x, g(x)), x, g(x)).doit()
    # 断言计算得到的结果等于再次应用 doit() 方法后的结果
    assert done == done.doit()


@XFAIL
def test_Subs2():
    # 这反映了 subs() 方法的一个限制，可能不会修复
    # 断言使用 subs() 方法替换后的表达式等于 f(sqrt(x))
    assert Subs(f(x), x**2, x).doit() == f(sqrt(x))


def test_expand_function():
    # 测试 expand() 函数的基本用法
    assert expand(x + y) == x + y
    # 测试带有复数参数的 expand() 函数
    assert expand(x + y, complex=True) == I*im(x) + I*im(y) + re(x) + re(y)
    # 测试使用 modulus 参数的 expand() 函数
    assert expand((x + y)**11, modulus=11) == x**11 + y**11


def test_function_comparable():
    # 测试 sin(x) 和 cos(x) 的 is_comparable 属性
    assert sin(x).is_comparable is False
    assert cos(x).is_comparable is False

    # 测试数值类型的 sin 和 cos 函数是否可比较
    assert sin(Float('0.1')).is_comparable is True
    assert cos(Float('0.1')).is_comparable is True

    # 测试常数 E 和有理数 1/3 的 sin 和 cos 是否可比较
    assert sin(E).is_comparable is True
    assert cos(E).is_comparable is True

    assert sin(Rational(1, 3)).is_comparable is True
    assert cos(Rational(1, 3)).is_comparable is True


def test_function_comparable_infinities():
    # 测试无穷大、负无穷大、NaN 和复数的 sin 函数是否可比较
    assert sin(oo).is_comparable is False
    assert sin(-oo).is_comparable is False
    assert sin(zoo).is_comparable is False
    assert sin(nan).is_comparable is False


def test_deriv1():
    # 这些测试需要在特定点求导数（问题编号 4719）才能正确工作
    # 参见问题编号 4624
    assert f(2*x).diff(x) == 2*Subs(Derivative(f(x), x), x, 2*x)
    assert (f(x)**3).diff(x) == 3*f(x)**2*f(x).diff(x)
    assert (f(2*x)**3).diff(x) == 6*f(2*x)**2*Subs(
        Derivative(f(x), x), x, 2*x)

    assert f(2 + x).diff(x) == Subs(Derivative(f(x), x), x, x + 2)
    assert f(2 + 3*x).diff(x) == 3*Subs(
        Derivative(f(x), x), x, 3*x + 2)
    assert f(3*sin(x)).diff(x) == 3*cos(x)*Subs(
        Derivative(f(x), x), x, 3*sin(x))

    # 参见问题编号 8510
    assert f(x, x + z).diff(x) == (
        Subs(Derivative(f(y, x + z), y), y, x) +
        Subs(Derivative(f(x, y), y), y, x + z))
    assert f(x, x**2).diff(x) == (
        2*x*Subs(Derivative(f(x, y), y), y, x**2) +
        Subs(Derivative(f(y, x**2), y), y, x))
    # 但是 Subs 并不总是必需的
    assert f(x, g(y)).diff(g(y)) == Derivative(f(x, g(y)), g(y))


def test_deriv2():
    assert (x**3).diff(x) == 3*x**2
    assert (x**3).diff(x, evaluate=False) != 3*x**2
    assert (x**3).diff(x, evaluate=False) == Derivative(x**3, x)

    assert diff(x**3, x) == 3*x**2
    assert diff(x**3, x, evaluate=False) != 3*x**2
    assert diff(x**3, x, evaluate=False) == Derivative(x**3, x)


def test_func_deriv():
    assert f(x).diff(x) == Derivative(f(x), x)
    # 问题编号 4534
    assert f(x, y).diff(x, y) - f(x, y).diff(y, x) == 0
    assert Derivative(f(x, y), x, y).args[1:] == ((x, 1), (y, 1))
    assert Derivative(f(x, y), y, x).args[1:] == ((y, 1), (x, 1))
    assert (Derivative(f(x, y), x, y) - Derivative(f(x, y), y, x)).doit() == 0


def test_suppressed_evaluation():
    a = sin(0, evaluate=False)
    assert a != 0
    assert a.func is sin
    assert a.args == (0,)


def test_function_evalf():
    def eq(a, b, eps):
        return abs(a - b) < eps
    assert eq(sin(1).evalf(15), Float("0.841470984807897"), 1e-13)
    # 断言：验证 sin(2) 的数值近似是否等于给定精度下的浮点数值，误差容限为 1e-23
    assert eq(
        sin(2).evalf(25), Float("0.9092974268256816953960199", 25), 1e-23)
    
    # 断言：验证 sin(1 + I) 的数值近似是否等于给定精度下的复数值，实部误差容限为 1e-13
    # 虚部为 Float("0.634963914784736")*I
    assert eq(sin(1 + I).evalf(
        15), Float("1.29845758141598") + Float("0.634963914784736")*I, 1e-13)
    
    # 断言：验证 exp(1 + I) 的数值近似是否等于给定精度下的复数值，实部误差容限为 1e-13
    # 虚部为 Float("2.28735528717884239")*I
    assert eq(exp(1 + I).evalf(15), Float(
        "1.46869393991588") + Float("2.28735528717884239")*I, 1e-13)
    
    # 断言：验证 exp(-0.5 + 1.5*I) 的数值近似是否等于给定精度下的复数值，实部误差容限为 1e-13
    # 虚部为 Float("0.605011292285002")*I
    assert eq(exp(-0.5 + 1.5*I).evalf(15), Float(
        "0.0429042815937374") + Float("0.605011292285002")*I, 1e-13)
    
    # 断言：验证 log(pi + sqrt(2)*I) 的数值近似是否等于给定精度下的复数值，实部误差容限为 1e-13
    # 虚部为 Float("0.422985442737893")*I
    assert eq(log(pi + sqrt(2)*I).evalf(
        15), Float("1.23699044022052") + Float("0.422985442737893")*I, 1e-13)
    
    # 断言：验证 cos(100) 的数值近似是否等于给定精度下的浮点数值，误差容限为 1e-13
    assert eq(cos(100).evalf(15), Float("0.86231887228768"), 1e-13)
# 定义一个名为 test_extensibility_eval 的测试函数
def test_extensibility_eval():
    # 定义一个名为 MyFunc 的类，继承自 Function 类
    class MyFunc(Function):
        # 类方法 eval，接受任意数量的参数并返回元组 (0, 0, 0)
        @classmethod
        def eval(cls, *args):
            return (0, 0, 0)
    # 断言 MyFunc(0) 的返回值等于 (0, 0, 0)
    assert MyFunc(0) == (0, 0, 0)


# 装饰一个函数，添加了 _both_exp_pow 装饰器
@_both_exp_pow
# 定义一个名为 test_function_non_commutative 的测试函数
def test_function_non_commutative():
    # 创建一个名为 x 的符号，并声明它为非交换的
    x = Symbol('x', commutative=False)
    # 断言 f(x) 的交换性为 False
    assert f(x).is_commutative is False
    # 断言 sin(x) 的交换性为 False
    assert sin(x).is_commutative is False
    # 断言 exp(x) 的交换性为 False
    assert exp(x).is_commutative is False
    # 断言 log(x) 的交换性为 False
    assert log(x).is_commutative is False
    # 断言 f(x) 的复性为 False
    assert f(x).is_complex is False
    # 断言 sin(x) 的复性为 False
    assert sin(x).is_complex is False
    # 断言 exp(x) 的复性为 False
    assert exp(x).is_complex is False
    # 断言 log(x) 的复性为 False
    assert log(x).is_complex is False


# 定义一个名为 test_function_complex 的测试函数
def test_function_complex():
    # 创建一个名为 x 的符号，并声明它为复数
    x = Symbol('x', complex=True)
    # 创建一个名为 xzf 的符号，声明它为复数且非零
    xzf = Symbol('x', complex=True, zero=False)
    # 断言 f(x) 的交换性为 True
    assert f(x).is_commutative is True
    # 断言 sin(x) 的交换性为 True
    assert sin(x).is_commutative is True
    # 断言 exp(x) 的交换性为 True
    assert exp(x).is_commutative is True
    # 断言 log(x) 的交换性为 True
    assert log(x).is_commutative is True
    # 断言 f(x) 的复性为 None
    assert f(x).is_complex is None
    # 断言 sin(x) 的复性为 True
    assert sin(x).is_complex is True
    # 断言 exp(x) 的复性为 True
    assert exp(x).is_complex is True
    # 断言 log(x) 的复性为 None
    assert log(x).is_complex is None
    # 断言 log(xzf) 的复性为 True
    assert log(xzf).is_complex is True


# 定义一个名为 test_function__eval_nseries 的测试函数
def test_function__eval_nseries():
    # 创建一个名为 n 的符号
    n = Symbol('n')

    # 断言 sin(x)._eval_nseries(x, 2, None) 的结果为 x + O(x**2)
    assert sin(x)._eval_nseries(x, 2, None) == x + O(x**2)
    # 断言 sin(x + 1)._eval_nseries(x, 2, None) 的结果为 x*cos(1) + sin(1) + O(x**2)
    assert sin(x + 1)._eval_nseries(x, 2, None) == x*cos(1) + sin(1) + O(x**2)
    # 断言 sin(pi*(1 - x))._eval_nseries(x, 2, None) 的结果为 pi*x + O(x**2)
    assert sin(pi*(1 - x))._eval_nseries(x, 2, None) == pi*x + O(x**2)
    # 断言 acos(1 - x**2)._eval_nseries(x, 2, None) 的结果为 sqrt(2)*sqrt(x**2) + O(x**2)
    assert acos(1 - x**2)._eval_nseries(x, 2, None) == sqrt(2)*sqrt(x**2) + O(x**2)
    # 断言 polygamma(n, x + 1)._eval_nseries(x, 2, None) 的结果为 polygamma(n, 1) + polygamma(n + 1, 1)*x + O(x**2)
    assert polygamma(n, x + 1)._eval_nseries(x, 2, None) == \
        polygamma(n, 1) + polygamma(n + 1, 1)*x + O(x**2)
    # 断言 lambda: sin(1/x)._eval_nseries(x, 2, None) 抛出 PoleError 异常
    raises(PoleError, lambda: sin(1/x)._eval_nseries(x, 2, None))
    # 断言 acos(1 - x)._eval_nseries(x, 2, None) 的结果为 sqrt(2)*sqrt(x) + sqrt(2)*x**(S(3)/2)/12 + O(x**2)
    assert acos(1 - x)._eval_nseries(x, 2, None) == sqrt(2)*sqrt(x) + sqrt(2)*x**(S(3)/2)/12 + O(x**2)
    # 断言 acos(1 + x)._eval_nseries(x, 2, None) 的结果为 sqrt(2)*sqrt(-x) + sqrt(2)*(-x)**(S(3)/2)/12 + O(x**2)
    assert acos(1 + x)._eval_nseries(x, 2, None) == sqrt(2)*sqrt(-x) + sqrt(2)*(-x)**(S(3)/2)/12 + O(x**2)
    # 断言 loggamma(1/x)._eval_nseries(x, 0, None) 的结果为 log(x)/2 - log(x)/x - 1/x + O(1, x)
    assert loggamma(1/x)._eval_nseries(x, 0, None) == \
        log(x)/2 - log(x)/x - 1/x + O(1, x)
    # 断言 loggamma(log(1/x)).nseries(x, n=1, logx=y) 的结果为 loggamma(-y)
    assert loggamma(log(1/x)).nseries(x, n=1, logx=y) == loggamma(-y)

    # issue 6725:
    # 创建 f(x,y) 的级数展开，并命名为 s1
    s1 = f(x,y).series(y, n=2)
    # 断言 s1 所有符号的名称为 {'x', 'xi', 'y'}
    assert {i.name for i in s1.atoms(Symbol)} == {'x', 'xi', 'y'}
    # 创建符号 xi，并将 f(xi, y) 的级数展开命名为 s2
    xi = Symbol('xi')
    s2 = f(xi, y).series(y, n=2)
    # 断言 s2 所有符号的名称为 {'xi', 'xi0', 'y'}
    assert {i.name for i in s2.atoms(Symbol)} == {'xi', 'xi0', 'y'}


# 定义一个名为 test_doit 的测试函数
def test_doit():
    # 创建一个名为 n 的整数符号
    n = Symbol('n', integer=True)
    # 创建一个求和表达式 f = Sum(2 * n * x, (n, 1, 3))
    f = Sum(2 * n * x, (n, 1, 3))
    # 创建一个对 f 对 x 求导的表达式，并命名为 d
    d = Derivative(f, x)
    # 断言 d.doit() 的结果为 12
    assert d.doit() == 12
    # 断言 d.doit(deep=False) 的结果为 Sum(2*n, (n, 1, 3))
    assert d.doit(deep=False) == Sum(2*n, (n, 1, 3))


# 定义一个名为 test_evalf_default 的测试函数
def test_evalf_default():
    # 导入 polygamma 函数
    from sympy.functions.special.gamma_functions import polygamma
    # 断言 type(sin(4.0)) 的结果为 Float 类型
    assert type(sin(4.0)) == Float
    # 断言 type(re(sin(I + 1.0))) 的结果为 Float 类型
    assert type(re(sin(I +
    # 断言：验证 sin(4) 的返回类型是否为 sin 类型
    assert type(sin(4)) == sin
    
    # 断言：验证 polygamma(2.0, 4.0) 的返回类型是否为 Float 类型
    assert type(polygamma(2.0, 4.0)) == Float
    
    # 断言：验证 sin(Rational(1, 4)) 的返回类型是否为 sin 类型
    assert type(sin(Rational(1, 4))) == sin
def test_issue_5399():
    # 定义测试函数，输入参数为 x, y, 2, 1/2
    args = [x, y, S(2), S.Half]

    def ok(a):
        """Return True if the input args for diff are ok"""
        # 如果 a 为空，则返回 False
        if not a:
            return False
        # 如果第一个元素不是 Symbol 对象，则返回 False
        if a[0].is_Symbol is False:
            return False
        # 获取所有是 Symbol 的元素的索引列表
        s_at = [i for i in range(len(a)) if a[i].is_Symbol]
        # 获取所有不是 Symbol 的元素的索引列表
        n_at = [i for i in range(len(a)) if not a[i].is_Symbol]
        # 每个 Symbol 元素后面要么是 Symbol 要么是 Integer
        # 每个 Number 元素后面必须是 Symbol
        return (all(a[i + 1].is_Symbol or a[i + 1].is_Integer
                    for i in s_at if i + 1 < len(a)) and
                all(a[i + 1].is_Symbol
                    for i in n_at if i + 1 < len(a)))

    # 定义一个表达式 eq = x**10 * y**8
    eq = x**10 * y**8
    # 对于 args 的所有子集 a
    for a in subsets(args):
        # 对于 a 的所有排列 v
        for v in variations(a, len(a)):
            # 如果 ok(v) 返回 True，则执行 eq.diff(*v)，不会引发异常
            if ok(v):
                eq.diff(*v) # does not raise
            # 否则，期望引发 ValueError 异常
            else:
                raises(ValueError, lambda: eq.diff(*v))


def test_derivative_numerically():
    # 随机生成一个 z0
    z0 = x._random()
    # 断言数值微分结果与 cos(z0) 的差小于 1e-15
    assert abs(Derivative(sin(x), x).doit_numerically(z0) - cos(z0)) < 1e-15


def test_fdiff_argument_index_error():
    # 导入 ArgumentIndexError 异常类
    from sympy.core.function import ArgumentIndexError

    # 定义一个自定义函数 myfunc
    class myfunc(Function):
        nargs = 1  # 定义 nargs 为 1，因为没有 eval 方法

        def fdiff(self, idx):
            # 抛出 ArgumentIndexError 异常
            raise ArgumentIndexError

    # 创建 myfunc 的实例 mf
    mf = myfunc(x)
    # 断言 myfunc 对 x 的求导结果等于 Derivative(mf, x)
    assert mf.diff(x) == Derivative(mf, x)
    # 断言调用 myfunc 时传递两个参数会引发 TypeError 异常
    raises(TypeError, lambda: myfunc(x, x))


def test_deriv_wrt_function():
    # 定义一个关于 t 的函数 x = f(t)
    x = f(t)
    # 计算 x 对 t 的导数 xd
    xd = diff(x, t)
    # 计算 xd 对 t 的导数 xdd
    xdd = diff(xd, t)
    # 定义另一个关于 t 的函数 y = g(t)
    y = g(t)
    # 计算 y 对 t 的导数 yd
    yd = diff(y, t)

    # 断言 x 对 t 的导数等于 xd
    assert diff(x, t) == xd
    # 断言对 2 * x + 4 对 t 的导数等于 2 * xd
    assert diff(2 * x + 4, t) == 2 * xd
    # 断言对 2 * x + 4 + y 对 t 的导数等于 2 * xd + yd
    assert diff(2 * x + 4 + y, t) == 2 * xd + yd
    # 断言对 2 * x + 4 + y * x 对 t 的导数等于 2 * xd + x * yd + xd * y
    assert diff(2 * x + 4 + y * x, t) == 2 * xd + x * yd + xd * y
    # 断言对 2 * x + 4 + y * x 对 x 的导数等于 2 + y
    assert diff(2 * x + 4 + y * x, x) == 2 + y
    # 断言对 4 * x**2 + 3 * x + x * y 对 t 的导数等于 3 * xd + x * yd + xd * y + 8 * x * xd
    assert (diff(4 * x**2 + 3 * x + x * y, t) == 3 * xd + x * yd + xd * y +
            8 * x * xd)
    # 断言对 4 * x**2 + 3 * xd + x * y 对 t 的导数等于 3 * xdd + x * yd + xd * y + 8 * x * xd
    assert (diff(4 * x**2 + 3 * xd + x * y, t) == 3 * xdd + x * yd + xd * y +
            8 * x * xd)
    # 断言对 4 * x**2 + 3 * xd + x * y 对 xd 的导数等于 3
    assert diff(4 * x**2 + 3 * xd + x * y, xd) == 3
    # 断言对 4 * x**2 + 3 * xd + x * y 对 xdd 的导数等于 0
    assert diff(4 * x**2 + 3 * xd + x * y, xdd) == 0
    # 断言对 sin(x) 对 t 的导数等于 xd * cos(x)
    assert diff(sin(x), t) == xd * cos(x)
    # 断言对 exp(x) 对 t 的导数等于 xd * exp(x)
    assert diff(exp(x), t) == xd * exp(x)
    # 断言对 sqrt(x) 对 t 的导数等于 xd / (2 * sqrt(x))
    assert diff(sqrt(x), t) == xd / (2 * sqrt(x))


def test_diff_wrt_value():
    # 断言 Expr 对象不支持微分
    assert Expr()._diff_wrt is False
    # 断言 x 支持微分
    assert x._diff_wrt is True
    # 断言 f(x) 支持微分
    assert f(x)._diff_wrt is True
    # 断言 Derivative(f(x), x) 支持微分
    assert Derivative(f(x), x)._diff_wrt is True
    # 断言 Derivative(x**2, x) 不支持微分
    assert Derivative(x**2, x)._diff_wrt is False


def test_diff_wrt():
    # 定义一个关于 x 的函数 fx = f(x)
    fx = f(x)
    # 计算 fx 对 x 的导数 dfx
    dfx = diff(f(x), x)
    # 计算 dfx 对 x 的导数 ddfx
    ddfx = diff(f(x), x, x)

    # 断言对 sin(fx) + fx**2 对 fx 的导数等于 cos(fx) + 2*fx
    assert diff(sin(fx) + fx**2, fx) == cos(fx) + 2*fx
    # 断言对 sin(dfx) + dfx**2 对 dfx 的导数等于 cos(dfx) + 2*dfx
    assert diff(sin(dfx) + dfx**2, dfx) == cos(dfx) + 2*dfx
    # 断言对 sin(ddfx) + ddfx**2 对 ddfx 的导数等于 cos(ddfx) + 2*ddfx
    assert diff(sin(ddfx) + ddfx**2, ddfx) == cos(ddfx) + 2*ddfx
    # 断言对 fx**2 对 dfx 的导数等于 0
    assert diff(fx**2, dfx) == 0
    # 断言对 fx**2 对 ddfx 的导数等于 0
    assert diff(fx**2, ddfx) == 0
    # 断言对 dfx**2 对 fx 的导数等于 0
    assert diff(dfx**2, fx) == 0
    # 断言对 dfx**2 对 ddfx 的导数等于 0
    assert diff(dfx**2, ddfx) == 0
    # 断言对 ddfx
    # 断言：f(x) 对 x 的导数再对 f(x) 的导数应为 0
    assert diff(f(x), x).diff(f(x)) == 0
    
    # 断言：sin(f(x)) - cos(f'(x)) 对 f(x) 的导数应为 cos(f(x))
    assert (sin(f(x)) - cos(diff(f(x), x))).diff(f(x)) == cos(f(x))
    
    # 断言：对 sin(fx) 在 fx 和 x 的偏导数应与在 x 和 fx 的偏导数相等
    assert diff(sin(fx), fx, x) == diff(sin(fx), x, fx)
    
    # Chain rule cases
    # 断言：复合函数 f(g(x)) 对 x 的导数等于 g(x) 对 x 的导数乘以 f(g(x)) 对 g(x) 的导数
    assert f(g(x)).diff(x) == \
        Derivative(g(x), x)*Derivative(f(g(x)), g(x))
    
    # 断言：复合函数 f(g(x), h(y)) 对 x 的导数等于 g(x) 对 x 的导数乘以 f(g(x), h(y)) 对 g(x) 的导数
    assert diff(f(g(x), h(y)), x) == \
        Derivative(g(x), x)*Derivative(f(g(x), h(y)), g(x))
    
    # 断言：复合函数 f(g(x), h(x)) 对 x 的导数等于 g(x) 对 x 的导数乘以 f 对 g(x) 的导数，加上 h(x) 对 x 的导数乘以 f 对 h(x) 的导数
    assert diff(f(g(x), h(x)), x) == (
        Derivative(f(g(x), h(x)), g(x))*Derivative(g(x), x) +
        Derivative(f(g(x), h(x)), h(x))*Derivative(h(x), x))
    
    # 断言：f(sin(x)) 对 x 的导数应为 cos(x) 乘以 f(x) 对 x 在 x=sin(x) 处的代数
    assert f(
        sin(x)).diff(x) == cos(x)*Subs(Derivative(f(x), x), x, sin(x))
    
    # 断言：f(g(x)) 对 g(x) 的导数应为 f(g(x)) 对 g(x) 的导数
    assert diff(f(g(x)), g(x)) == Derivative(f(g(x)), g(x))
# 测试函数，验证复合函数求导与函数替换的结果是否正确
def test_diff_wrt_func_subs():
    # 断言复合函数 f(g(x)) 对 x 求导后，替换 g(x) 为 Lambda 表达式 Lambda(x, 2*x)，再求值等于 f(2*x) 对 x 求导的结果
    assert f(g(x)).diff(x).subs(g, Lambda(x, 2*x)).doit() == f(2*x).diff(x)


# 测试在导数中的符号替换操作
def test_subs_in_derivative():
    expr = sin(x*exp(y))
    u = Function('u')
    v = Function('v')
    # 断言对表达式 expr 求关于 y 的导数后，用 expr 替换 y 后结果等于对 y 求导数的结果
    assert Derivative(expr, y).subs(expr, y) == Derivative(y, y)
    # 断言对表达式 expr 求关于 y 的导数后，将 y 替换为 x 后再求值等于直接将 y 替换为 x 后再求导数的结果
    assert Derivative(expr, y).subs(y, x).doit() == Derivative(expr, y).doit().subs(y, x)
    # 断言对 f(x, y) 关于 y 求导数后，将 y 替换为 x 后结果等于 Subs 对象，用 Subs 替换 Derivative 的 y 参数
    assert Derivative(f(x, y), y).subs(y, x) == Subs(Derivative(f(x, y), y), y, x)
    # 断言对 f(x, y) 关于 y 求导数后，将 x 替换为 y 后结果等于 Subs 对象，用 Subs 替换 Derivative 的 x 参数
    assert Derivative(f(x, y), y).subs(x, y) == Subs(Derivative(f(x, y), y), x, y)
    # 断言对 f(x, y) 关于 y 求导数后，将 y 替换为 g(x, y) 后结果等于对 Subs(Derivative(f(x, y), y), y, g(x, y)) 进行求值
    assert Derivative(f(x, y), y).subs(y, g(x, y)) == Subs(Derivative(f(x, y), y), y, g(x, y)).doit()
    # 断言对 f(x, y) 关于 y 求导数后，将 x 替换为 g(x, y) 后结果等于 Subs 对象，用 Subs 替换 Derivative 的 x 参数
    assert Derivative(f(x, y), y).subs(x, g(x, y)) == Subs(Derivative(f(x, y), y), x, g(x, y))
    # 断言对 f(u(x), h(y)) 关于 h(y) 求导数后，将 h(y) 替换为 g(x, y) 后结果等于对 Subs(Derivative(f(u(x), h(y)), h(y)), h(y), g(x, y)) 进行求值
    assert Derivative(f(u(x), h(y)), h(y)).subs(h(y), g(x, y)) == Subs(Derivative(f(u(x), h(y)), h(y)), h(y), g(x, y)).doit()
    # 断言对 f(x, y) 关于 y 求导数后，将 y 替换为 z 后结果等于 Derivative(f(x, z), z)
    assert Derivative(f(x, y), y).subs(y, z) == Derivative(f(x, z), z)
    # 断言对 f(x, y) 关于 y 求导数后，将 y 替换为 g(y) 后结果等于 Derivative(f(x, g(y)), g(y))
    assert Derivative(f(x, y), y).subs(y, g(y)) == Derivative(f(x, g(y)), g(y))
    # 断言对 f(g(x), h(y)) 关于 h(y) 求导数后，将 h(y) 替换为 u(y) 后结果等于 Derivative(f(g(x), u(y)), u(y))
    assert Derivative(f(g(x), h(y)), h(y)).subs(h(y), u(y)) == Derivative(f(g(x), u(y)), u(y))
    # 断言对 f(f(x)) 关于 x 求导数后，将 f 替换为 Lambda 表达式 Lambda((x, y), x + y) 后结果等于 Subs(Derivative(z + x, z), z, 2*x)
    assert Derivative(f(f(x)), x).subs(f, Lambda((x, y), x + y)) == Subs(Derivative(z + x, z), z, 2*x)
    # 断言对 Derivative(f(f(x)), f(x)) 的 Subs 替换 f 为 cos 后求值等于 sin(x)*sin(cos(x))
    assert Subs(Derivative(f(f(x)), x), f, cos).doit() == sin(x)*sin(cos(x))
    # 断言对 Derivative(f(f(x)), f(x)) 的 Subs 替换 f 为 cos 后求导数等于 -sin(cos(x))
    assert Subs(Derivative(f(f(x)), f(x)), f, cos).doit() == -sin(cos(x))
    # 断言对 v(x, y, u(x, y)) 多重求导操作的结果是 Expr 类型
    assert isinstance(v(x, y, u(x, y)).diff(y).diff(x).diff(y), Expr)
    # 断言对 Lambda 表达式 F 进行对 f(f(x)) 的 Subs 替换 f 为 F 后求值结果为 0
    F = Lambda((x, y), exp(2*x + 3*y))
    abstract = f(x, f(x, x)).diff(x, 2)
    concrete = F(x, F(x, x)).diff(x, 2)
    assert (abstract.subs(f, F).doit() - concrete).simplify() == 0
    # 断言在 f(x) 关于 x 求导后，将 x 替换为 0 后的结果中，包含常数 S.One
    assert x in f(x).diff(x).subs(x, 0).atoms()
    # 断言对 Derivative(f(x,f(x,y)), x, y) 的 Subs 替换 x 为 g(y) 后结果等于 Subs(Derivative(f(x, f(x, y)), x, y), x, g(y))
    assert Derivative(f(x, f(x, y)), x, y).subs(x, g(y)) == Subs(Derivative(f(x, f(x, y)), x, y), x, g(y))
    # 断言对 Derivative(f(x, x), x) 的 Subs 替换 x 为 0 后结果等于 Subs(Derivative(f(x, x), x), x, 0)
    assert Derivative(f(x, x), x).subs(x, 0) == Subs(Derivative(f(x, x), x), x, 0)
    # 断言对 Derivative(f(y, g(x)), (x, z)) 的 Subs 替换 z 为 x 后结果等于 Derivative(f(y, g(x)), (x, x))
    assert Derivative(f(y, g(x)), (x, z)).subs(z, x) == Derivative(f(y, g(x)), (x, x))
    
    # 定义 df 为 f(x) 关于 x 的导数
    df = f(x).diff(x)
    # 断言对 df 的 Subs 替换 df 为 1 后结果为 S.One
    assert df.subs(df, 1) is S.One
    # 断言对 df 的二阶导数为 S.One
    assert df.diff(df) is S.One
    
    # 定义 dxy 和 dyx 为 f(x, y) 关于 x, y 的混合导数
    dxy = Derivative(f(x, y), x, y)
    dyx = Derivative(f(x, y), y, x)
    # 断言对 dxy 的 Subs 替换 dyx 为 1 后结果为 S.One
    assert dxy.subs(Derivative(f(x, y), y, x), 1) is S.One
    # 断言对 dxy 的混合导数为 S.One
    assert dxy.diff(dyx) is S.One
    
    # 断言对 Derivative(f(x, y), x, 2, y, 3) 的 Subs 替换 dyx 为 g(x, y) 后结果等于 Derivative(g(x, y), x, 1, y, 2
    # 对于每个表达式 wrt，依次进行求导并检查是否引发 ValueError 异常
    for wrt in (
            cos(x), re(x), x**2, x*y, 1 + x,
            Derivative(cos(x), x), Derivative(f(f(x)), x)):
        raises(ValueError, lambda: diff(f(x), wrt))
    
    # 如果我们不对 wrt 进行求导，那么不会引发错误
    # 断言：对 exp(x*y) 求关于 x*y 的 0 阶导数应该等于 exp(x*y)
    assert diff(exp(x*y), x*y, 0) == exp(x*y)
def test_diff_wrt_intlike():
    class Two:
        def __int__(self):
            return 2

    # 定义一个名为 Two 的类，实现了 __int__ 方法，返回整数 2
    assert cos(x).diff(x, Two()) == -cos(x)


def test_klein_gordon_lagrangian():
    m = Symbol('m')
    phi = f(x, t)

    # 定义拉格朗日量 L
    L = -(diff(phi, t)**2 - diff(phi, x)**2 - m**2*phi**2)/2
    # 定义方程 eqna 和 eqnb，分别表示拉格朗日方程
    eqna = Eq(
        diff(L, phi) - diff(L, diff(phi, x), x) - diff(L, diff(phi, t), t), 0)
    eqnb = Eq(diff(phi, t, t) - diff(phi, x, x) + m**2*phi, 0)
    # 断言拉格朗日方程 eqna 和 eqnb 相等
    assert eqna == eqnb


def test_sho_lagrangian():
    m = Symbol('m')
    k = Symbol('k')
    x = f(t)

    # 定义简谐振动的拉格朗日量 L
    L = m*diff(x, t)**2/2 - k*x**2/2
    # 定义方程 eqna 和 eqnb，分别表示拉格朗日方程
    eqna = Eq(diff(L, x), diff(L, diff(x, t), t))
    eqnb = Eq(-k*x, m*diff(x, t, t))
    # 断言拉格朗日方程 eqna 和 eqnb 相等
    assert eqna == eqnb

    # 断言关于偏导数的对称性质
    assert diff(L, x, t) == diff(L, t, x)
    assert diff(L, diff(x, t), t) == m*diff(x, t, 2)
    assert diff(L, t, diff(x, t)) == -k*x + m*diff(x, t, 2)


def test_straight_line():
    F = f(x)
    Fd = F.diff(x)
    # 定义直线路径的长度 L
    L = sqrt(1 + Fd**2)
    # 断言关于 L 对 F 和 Fd 的偏导数
    assert diff(L, F) == 0
    assert diff(L, Fd) == Fd/sqrt(1 + Fd**2)


def test_sort_variable():
    vsort = Derivative._sort_variable_count
    def vsort0(*v, reverse=False):
        return [i[0] for i in vsort([(i, 0) for i in (
            reversed(v) if reverse else v)])]

    # 对不同的变量排序进行测试
    for R in range(2):
        assert vsort0(y, x, reverse=R) == [x, y]
        assert vsort0(f(x), x, reverse=R) == [x, f(x)]
        assert vsort0(f(y), f(x), reverse=R) == [f(x), f(y)]
        assert vsort0(g(x), f(y), reverse=R) == [f(y), g(x)]
        assert vsort0(f(x, y), f(x), reverse=R) == [f(x), f(x, y)]
        fx = f(x).diff(x)
        assert vsort0(fx, y, reverse=R) == [y, fx]
        fy = f(y).diff(y)
        assert vsort0(fy, fx, reverse=R) == [fx, fy]
        fxx = fx.diff(x)
        assert vsort0(fxx, fx, reverse=R) == [fx, fxx]
        assert vsort0(Basic(x), f(x), reverse=R) == [f(x), Basic(x)]
        assert vsort0(Basic(y), Basic(x), reverse=R) == [Basic(x), Basic(y)]
        assert vsort0(Basic(y, z), Basic(x), reverse=R) == [
            Basic(x), Basic(y, z)]
        assert vsort0(fx, x, reverse=R) == [
            x, fx] if R else [fx, x]
        assert vsort0(Basic(x), x, reverse=R) == [
            x, Basic(x)] if R else [Basic(x), x]
        assert vsort0(Basic(f(x)), f(x), reverse=R) == [
            f(x), Basic(f(x))] if R else [Basic(f(x)), f(x)]
        assert vsort0(Basic(x, z), Basic(x), reverse=R) == [
            Basic(x), Basic(x, z)] if R else [Basic(x, z), Basic(x)]
    # 断言空列表情况
    assert vsort([]) == []
    assert _aresame(vsort([(x, 1)]), [Tuple(x, 1)])
    assert vsort([(x, y), (x, z)]) == [(x, y + z)]
    assert vsort([(y, 1), (x, 1 + y)]) == [(x, 1 + y), (y, 1)]
    # 完成对排序函数的测试覆盖，以下为旧版测试
    assert vsort([(x, 3), (y, 2), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
    assert vsort([(h(x), 1), (g(x), 1), (f(x), 1)]) == [
        (f(x), 1), (g(x), 1), (h(x), 1)]
    assert vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1),
        (f(x), 1)]) == [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1),
        (h(x), 1)]
    # 第一条断言，验证 vsort 函数对列表进行排序，使得元组的第一个元素按字典序升序排列，第二个元素按原顺序排列
    assert vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)]) == [(x, 1),
        (y, 1), (f(x), 1), (f(y), 1)]
    
    # 第二条断言，验证 vsort 函数对列表进行排序，同样遵循字典序升序和原顺序的规则
    assert vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1),
        (h(x), 1), (y, 2), (x, 1)]) == [(x, 3), (y, 3), (z, 1),
        (f(x), 1), (g(x), 1), (h(x), 1)]
    
    # 第三条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1),
        (g(x), 1)]) == [(x, 1), (y, 1), (z, 1), (f(x), 2), (g(x), 1)]
    
    # 第四条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2),
        (g(x), 1), (z, 2), (z, 1), (y, 1), (x, 1)]) == [(x, 3), (y, 3),
        (z, 4), (f(x), 3), (g(x), 1)]
    
    # 第五条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort(((y, 2), (x, 1), (y, 1), (x, 1))) == [(x, 2), (y, 3)]
    
    # 第六条断言，验证 vsort 函数对列表进行排序，并验证排序后的第一个元素是否为元组
    assert isinstance(vsort([(x, 3), (y, 2), (z, 1)])[0], Tuple)
    
    # 第七条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(x, 1), (f(x), 1), (x, 1)]) == [(x, 2), (f(x), 1)]
    
    # 第八条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(y, 2), (x, 3), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
    
    # 第九条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(h(y), 1), (g(x), 1), (f(x), 1)]) == [
        (f(x), 1), (g(x), 1), (h(y), 1)]
    
    # 第十条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(x, 1), (y, 1), (x, 1)]) == [(x, 2), (y, 1)]
    
    # 第十一条断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(f(x), 1), (f(y), 1), (f(x), 1)]) == [
        (f(x), 2), (f(y), 1)]
    
    # 对 dfx 进行操作，计算 f(x) 对 x 的导数
    dfx = f(x).diff(x)
    # 构建 self 列表
    self = [(dfx, 1), (x, 1)]
    # 最后一条断言，验证 vsort 函数对 self 列表进行排序，验证是否正确按照规则排序
    assert vsort(self) == self
    
    # 断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([
        (dfx, 1), (y, 1), (f(x), 1), (x, 1), (f(y), 1), (x, 1)]) == [
        (y, 1), (f(x), 1), (f(y), 1), (dfx, 1), (x, 2)]
    
    # 对 dfy 进行操作，计算 f(y) 对 y 的导数
    dfy = f(y).diff(y)
    # 断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(dfy, 1), (dfx, 1)]) == [(dfx, 1), (dfy, 1)]
    
    # 对 d2fx 进行操作，计算 dfx 对 x 的二阶导数
    d2fx = dfx.diff(x)
    # 断言，验证 vsort 函数对列表进行排序，验证是否正确按照规则排序
    assert vsort([(d2fx, 1), (dfx, 1)]) == [(dfx, 1), (d2fx, 1)]
def test_nfloat():
    # 导入必要的库和符号
    from sympy.core.basic import _aresame
    from sympy.polys.rootoftools import rootof

    # 定义符号变量 x
    x = Symbol("x")

    # 创建一个有理数幂的表达式 eq
    eq = x**Rational(4, 3) + 4*x**(S.One/3)/3
    # 断言 nfloat 函数应用后的结果与预期值相等，不考虑指数形式
    assert _aresame(nfloat(eq), x**Rational(4, 3) + (4.0/3)*x**(S.One/3))
    
    # 断言 nfloat 函数应用后的结果与预期值相等，考虑指数形式
    assert _aresame(nfloat(eq, exponent=True), x**(4.0/3) + (4.0/3)*x**(1.0/3))
    
    # 创建一个带变量 x 的幂的表达式 eq
    eq = x**Rational(4, 3) + 4*x**(x/3)/3
    # 断言 nfloat 函数应用后的结果与预期值相等，不考虑指数形式
    assert _aresame(nfloat(eq), x**Rational(4, 3) + (4.0/3)*x**(x/3))
    
    # 创建一个大整数 big
    big = 12345678901234567890
    # 根据 nfloat 函数的精度要求创建一个对应的浮点数对象 Float_big
    Float_big = Float(big, 15)
    # 断言 nfloat 函数应用后的结果与预期值相等
    assert _aresame(nfloat(big), Float_big)
    # 断言 nfloat 函数应用后的结果与预期值相等，包含变量 x
    assert _aresame(nfloat(big*x), Float_big*x)
    # 断言 nfloat 函数应用后的结果与预期值相等，考虑指数形式
    assert _aresame(nfloat(x**big, exponent=True), x**Float_big)
    # 断言 nfloat 函数应用后的结果与预期值相等，包含余弦函数
    assert nfloat(cos(x + sqrt(2))) == cos(x + nfloat(sqrt(2)))

    # 解决问题 6342
    # 对 f 进行求解，检查其是否有自由符号
    f = S('x*lamda + lamda**3*(x/2 + 1/2) + lamda**2 + 1/4')
    assert not any(a.free_symbols for a in solveset(f.subs(x, -0.139)))

    # 解决问题 6632
    # 断言 nfloat 函数应用后的结果与预期值相等
    assert nfloat(-100000*sqrt(2500000001) + 5000000001) == \
        9.99999999800000e-11

    # 解决问题 7122
    # 创建一个表达式 eq
    eq = cos(3*x**4 + y)*rootof(x**5 + 3*x**3 + 1, 0)
    # 断言 nfloat 函数应用后的结果的字符串形式与预期值相等，不考虑指数形式，保留一位小数
    assert str(nfloat(eq, exponent=False, n=1)) == '-0.7*cos(3.0*x**4 + y)'

    # 解决问题 10933
    # 遍历两种类型的字典数据结构 dict 和 Dict
    for ti in (dict, Dict):
        # 创建一个具有有理数键和值的字典 d
        d = ti({S.Half: S.Half})
        # 对字典 d 应用 nfloat 函数，并断言结果的类型与预期相同
        n = nfloat(d)
        assert isinstance(n, ti)
        # 断言 nfloat 函数应用后的结果与预期值相等，不考虑指数形式
        assert _aresame(list(n.items()).pop(), (S.Half, Float(.5)))
    for ti in (dict, Dict):
        # 创建一个具有有理数键和值的字典 d
        d = ti({S.Half: S.Half})
        # 对字典 d 应用 nfloat 函数，考虑键的类型转换为浮点数，断言结果的类型与预期相同
        n = nfloat(d, dkeys=True)
        assert isinstance(n, ti)
        # 断言 nfloat 函数应用后的结果与预期值相等，不考虑指数形式
        assert _aresame(list(n.items()).pop(), (Float(.5), Float(.5)))
    # 创建一个包含有理数的列表 d
    d = [S.Half]
    # 对列表 d 应用 nfloat 函数，并断言结果的类型与预期相同
    n = nfloat(d)
    assert type(n) is list
    # 断言 nfloat 函数应用后的结果与预期值相等，不考虑指数形式
    assert _aresame(n[0], Float(.5))
    # 断言 nfloat 函数应用后的结果的右侧与预期值相等，不考虑指数形式
    assert _aresame(nfloat(Eq(x, S.Half)).rhs, Float(.5))
    # 断言 nfloat 函数应用后的结果与预期值相等
    assert _aresame(nfloat(S(True)), S(True))
    # 断言 nfloat 函数应用后的结果的第一个元素与预期值相等，不考虑指数形式
    assert _aresame(nfloat(Tuple(S.Half))[0], Float(.5))
    # 断言 nfloat 函数应用后的结果与预期值相等
    assert nfloat(Eq((3 - I)**2/2 + I, 0)) == S.false
    # 断言 nfloat 函数应用后的结果与预期值相等，传递额外的关键字参数
    assert nfloat([{S.Half: x}], dkeys=True) == [{Float(0.5): x}]

    # 解决问题 17706
    # 创建一个可变的矩阵 A
    A = MutableMatrix([[1, 2], [3, 4]])
    # 创建一个浮点数精度为 53 的可变矩阵 B，用于与 nfloat 函数应用后的结果比较
    B = MutableMatrix(
        [[Float('1.0', precision=53), Float('2.0', precision=53)],
        [Float('3.0', precision=53), Float('4.0', precision=53)]])
    # 断言 nfloat 函数应用后的结果与预期值相等
    assert _aresame(nfloat(A), B)
    
    # 创建一个不可变的矩阵 A
    A = ImmutableMatrix([[1, 2], [3, 4]])
    # 创建一个不可变的矩阵 B，其中包含两行两列的浮点数值
    B = ImmutableMatrix(
        [[Float('1.0', precision=53), Float('2.0', precision=53)],
        [Float('3.0', precision=53), Float('4.0', precision=53)]])
    # 使用 _aresame 函数检查 A 的数值近似是否与 B 相同
    assert _aresame(nfloat(A), B)
    
    # 处理 issue 22524 的问题
    # 创建一个名为 f 的函数对象
    f = Function('f')
    # 使用 nfloat 函数转换 f(2) 的结果，并检查其是否不包含浮点数原子
    assert not nfloat(f(2)).atoms(Float)
def test_issue_7068():
    # 导入 sympy 的符号 a, b
    from sympy.abc import a, b
    # 定义函数 f
    f = Function('f')
    # 创建两个虚拟变量 y1 和 y2
    y1 = Dummy('y')
    y2 = Dummy('y')
    # 计算两个函数表达式 func1 和 func2
    func1 = f(a + y1 * b)
    func2 = f(a + y2 * b)
    # 分别对 func1 和 func2 求关于 y1 和 y2 的偏导数
    func1_y = func1.diff(y1)
    func2_y = func2.diff(y2)
    # 断言 func1_y 和 func2_y 不相等
    assert func1_y != func2_y
    # 创建 Subs 对象 z1 和 z2，表示 f(a) 在 a 处的替代值
    z1 = Subs(f(a), a, y1)
    z2 = Subs(f(a), a, y2)
    # 断言 z1 和 z2 不相等
    assert z1 != z2


def test_issue_7231():
    # 导入 sympy 的符号 a
    from sympy.abc import a
    # 计算 f(x) 在 x 点的级数展开 ans1
    ans1 = f(x).series(x, a)
    # 计算级数展开的结果 res
    res = (f(a) + (-a + x)*Subs(Derivative(f(y), y), y, a) +
           (-a + x)**2*Subs(Derivative(f(y), y, y), y, a)/2 +
           (-a + x)**3*Subs(Derivative(f(y), y, y, y),
                            y, a)/6 +
           (-a + x)**4*Subs(Derivative(f(y), y, y, y, y),
                            y, a)/24 +
           (-a + x)**5*Subs(Derivative(f(y), y, y, y, y, y),
                            y, a)/120 + O((-a + x)**6, (x, a)))
    # 断言 res 等于 ans1
    assert res == ans1
    # 再次计算 f(x) 在 x 点的级数展开 ans2
    ans2 = f(x).series(x, a)
    # 断言 res 等于 ans2
    assert res == ans2


def test_issue_7687():
    # 导入 sympy 的函数类 Function 和符号 x
    from sympy.core.function import Function
    from sympy.abc import x
    # 创建函数 f(x)
    f = Function('f')(x)
    # 创建另一个 f(x) 实例 ff
    ff = Function('f')(x)
    # 使用 matches 方法比较 ff 和 f
    match_with_cache = ff.matches(f)
    # 断言 f 和 ff 的类型相同
    assert isinstance(f, type(ff))
    # 清空缓存
    clear_cache()
    # 重新创建函数 f(x)
    ff = Function('f')(x)
    # 断言 f 和 ff 的类型相同
    assert isinstance(f, type(ff))
    # 断言 match_with_cache 与 ff.matches(f) 的结果相同
    assert match_with_cache == ff.matches(f)


def test_issue_7688():
    # 导入 sympy 的函数类 Function 和 UndefinedFunction
    from sympy.core.function import Function, UndefinedFunction
    # 创建一个名为 f 的 UndefinedFunction 实例
    f = Function('f')  # 实际上是一个 UndefinedFunction
    # 清空缓存
    clear_cache()
    # 创建一个类 A，继承自 UndefinedFunction
    class A(UndefinedFunction):
        pass
    # 创建 A 的实例 a，表示函数 'f'
    a = A('f')
    # 断言 a 的类型与 f 的类型相同
    assert isinstance(a, type(f))


def test_mexpand():
    # 导入 sympy 的符号 x
    from sympy.abc import x
    # 断言 _mexpand(None) 的结果是 None
    assert _mexpand(None) is None
    # 断言 _mexpand(1) 的结果是 S.One
    assert _mexpand(1) is S.One
    # 断言 _mexpand(x*(x + 1)**2) 的结果与 (x*(x + 1)**2).expand() 相同
    assert _mexpand(x*(x + 1)**2) == (x*(x + 1)**2).expand()


def test_issue_8469():
    # 这个测试不应该运行太久
    N = 40
    # 定义函数 g(w, theta)
    def g(w, theta):
        return 1/(1+exp(w-theta))

    # 创建符号 ws
    ws = symbols(['w%i'%i for i in range(N)])
    # 导入 functools
    import functools
    # 使用 functools.reduce 计算表达式 expr
    expr = functools.reduce(g, ws)
    # 断言 expr 的类型是 Pow
    assert isinstance(expr, Pow)


def test_issue_12996():
    # foo=True 模拟 Derivative 从 Integral 获取的参数类型
    # 当它传递 doit 到表达式时
    # 断言 Derivative(im(x), x).doit(foo=True) 等于 Derivative(im(x), x)
    assert Derivative(im(x), x).doit(foo=True) == Derivative(im(x), x)


def test_should_evalf():
    # 这个测试不应该运行太久 (参见 #8506)
    # 断言 sin((1.0 + 1.0*I)**10000 + 1) 的类型是 sin
    assert isinstance(sin((1.0 + 1.0*I)**10000 + 1), sin)


def test_Derivative_as_finite_difference():
    # 中心差分法计算在网格点的一阶导数
    x, h = symbols('x h', real=True)
    dfdx = f(x).diff(x)
    # 断言 dfdx 的有限差分与中心差分法公式的结果相等
    assert (dfdx.as_finite_difference([x-2, x-1, x, x+1, x+2]) -
            (S.One/12*(f(x-2)-f(x+2)) + Rational(2, 3)*(f(x+1)-f(x-1)))).simplify() == 0

    # 中心差分法计算在 "中间" 的一阶导数
    assert (dfdx.as_finite_difference() -
            (f(x + S.Half)-f(x - S.Half))).simplify() == 0
    assert (dfdx.as_finite_difference(h) -
            (f(x + h/S(2))-f(x - h/S(2)))/h).simplify() == 0
    # 断言：使用有限差分近似计算函数的一阶导数在网格点处的值，验证结果是否为零
    assert (dfdx.as_finite_difference([x - 3*h, x-h, x+h, x + 3*h]) -
            (S(9)/(8*2*h)*(f(x+h) - f(x-h)) +
             S.One/(24*2*h)*(f(x - 3*h) - f(x + 3*h)))).simplify() == 0

    # 断言：使用有限差分近似计算一阶导数在网格点处的值（单边）
    assert (dfdx.as_finite_difference([0, 1, 2], 0) -
            (Rational(-3, 2)*f(0) + 2*f(1) - f(2)/2)).simplify() == 0

    # 断言：使用有限差分近似计算一阶导数在网格点处的值（单边）
    assert (dfdx.as_finite_difference([x, x+h], x) -
            (f(x+h) - f(x))/h).simplify() == 0

    # 断言：使用有限差分近似计算一阶导数在网格点处的值（单边）
    assert (dfdx.as_finite_difference([x-h, x, x+h], x-h) -
            (-S(3)/(2*h)*f(x-h) + 2/h*f(x) -
             S.One/(2*h)*f(x+h))).simplify() == 0

    # 断言：使用有限差分近似计算一阶导数在网格点处的值（单边），中间位置
    assert (dfdx.as_finite_difference([x-h, x+h, x + 3*h, x + 5*h, x + 7*h])
            - 1/(2*h)*(-S(11)/(12)*f(x-h) + S(17)/(24)*f(x+h)
                       + Rational(3, 8)*f(x + 3*h) - Rational(5, 24)*f(x + 5*h)
                       + S.One/24*f(x + 7*h))).simplify() == 0

    # 计算二阶导数
    d2fdx2 = f(x).diff(x, 2)
    # 断言：使用有限差分近似计算二阶导数在网格点处的值（中心差分）
    assert (d2fdx2.as_finite_difference([x-h, x, x+h]) -
            h**-2 * (f(x-h) + f(x+h) - 2*f(x))).simplify() == 0

    # 断言：使用有限差分近似计算二阶导数在网格点处的值（中心差分）
    assert (d2fdx2.as_finite_difference([x - 2*h, x-h, x, x+h, x + 2*h]) -
            h**-2 * (Rational(-1, 12)*(f(x - 2*h) + f(x + 2*h)) +
                     Rational(4, 3)*(f(x+h) + f(x-h)) - Rational(5, 2)*f(x))).simplify() == 0

    # 断言：使用有限差分近似计算二阶导数在网格点处的值（中心差分），中间位置
    assert (d2fdx2.as_finite_difference([x - 3*h, x-h, x+h, x + 3*h]) -
            (2*h)**-2 * (S.Half*(f(x - 3*h) + f(x + 3*h)) -
                         S.Half*(f(x+h) + f(x-h)))).simplify() == 0

    # 断言：使用有限差分近似计算二阶导数在网格点处的值（单边）
    assert (d2fdx2.as_finite_difference([x, x+h, x + 2*h, x + 3*h]) -
            h**-2 * (2*f(x) - 5*f(x+h) +
                     4*f(x+2*h) - f(x+3*h))).simplify() == 0

    # 断言：使用有限差分近似计算二阶导数在网格点处的值（单边），中间位置
    assert (d2fdx2.as_finite_difference([x-h, x+h, x + 3*h, x + 5*h]) -
            (2*h)**-2 * (Rational(3, 2)*f(x-h) - Rational(7, 2)*f(x+h) + Rational(5, 2)*f(x + 3*h) -
                         S.Half*f(x + 5*h))).simplify() == 0

    # 计算三阶导数
    d3fdx3 = f(x).diff(x, 3)
    # 断言：使用有限差分近似计算三阶导数在网格点处的值（中心差分）
    assert (d3fdx3.as_finite_difference() -
            (-f(x - Rational(3, 2)) + 3*f(x - S.Half) -
             3*f(x + S.Half) + f(x + Rational(3, 2)))).simplify() == 0

    # 断言：使用有限差分近似计算三阶导数在网格点处的值（中心差分）
    assert (d3fdx3.as_finite_difference(
        [x - 3*h, x - 2*h, x-h, x, x+h, x + 2*h, x + 3*h]) -
        h**-3 * (S.One/8*(f(x - 3*h) - f(x + 3*h)) - f(x - 2*h) +
                 f(x + 2*h) + Rational(13, 8)*(f(x-h) - f(x+h)))).simplify() == 0

    # 断言：使用有限差分近似计算三阶导数在网格点处的值（中心差分），中间位置
    assert (d3fdx3.as_finite_difference([x - 3*h, x-h, x+h, x + 3*h]) -
            (2*h)**-3 * (f(x + 3*h)-f(x - 3*h) +
                         3*(f(x-h)-f(x+h)))).simplify() == 0

    # 断言：使用有限差分近似计算三阶导数在网格点处的值（单边）
    # 断言语句，验证三阶有限差分公式的准确性
    assert (d3fdx3.as_finite_difference([x, x+h, x + 2*h, x + 3*h]) -
            h**-3 * (f(x + 3*h)-f(x) + 3*(f(x+h)-f(x + 2*h)))).simplify() == 0

    # 断言语句，验证在"half-way"处的单边三阶导数的准确性
    assert (d3fdx3.as_finite_difference([x-h, x+h, x + 3*h, x + 5*h]) -
            (2*h)**-3 * (f(x + 5*h)-f(x-h) +
                         3*(f(x+h)-f(x + 3*h)))).simplify() == 0

    # 解决问题11007
    # 创建一个实数符号变量y
    y = Symbol('y', real=True)
    # 计算函数f(x, y)对x和y的二阶偏导数
    d2fdxdy = f(x, y).diff(x, y)

    # 定义参考值ref0，用于计算在x +- S.Half处对y的导数
    ref0 = Derivative(f(x + S.Half, y), y) - Derivative(f(x - S.Half, y), y)
    # 断言语句，验证二阶混合偏导数的有限差分公式的准确性
    assert (d2fdxdy.as_finite_difference(wrt=x) - ref0).simplify() == 0

    # 定义S.Half为half，计算x和y的四个"half"相邻点
    half = S.Half
    xm, xp, ym, yp = x-half, x+half, y-half, y+half
    # 定义参考值ref2，用于计算在不同方向上的二阶混合偏导数的有限差分
    ref2 = f(xm, ym) + f(xp, yp) - f(xp, ym) - f(xm, yp)
    # 断言语句，验证二阶混合偏导数的有限差分公式的准确性
    assert (d2fdxdy.as_finite_difference() - ref2).simplify() == 0
# 测试问题 11159 的函数
def test_issue_11159():
    # 使用 _exp_is_pow(False) 上下文进行测试，确认 Application._eval_subs 的行为
    with _exp_is_pow(False):
        expr1 = E  # 设定 expr1 为常数 E
        expr0 = expr1 * expr1  # 计算 expr0 = E^2
        expr1 = expr0.subs(expr1, expr0)  # 在 expr0 中用 expr1 替换为 expr0 自身
        assert expr0 == expr1  # 断言 expr0 等于 expr1
    # 使用 _exp_is_pow(True) 上下文进行测试
    with _exp_is_pow(True):
        expr1 = E  # 设定 expr1 为常数 E
        expr0 = expr1 * expr1  # 计算 expr0 = E^2
        expr2 = expr0.subs(expr1, expr0)  # 在 expr0 中用 expr1 替换为 expr0 自身
        assert expr2 == E ** 4  # 断言 expr2 等于 E 的四次方


# 测试问题 12005 的函数
def test_issue_12005():
    # 创建 e1 对象，表示对 f(x) 对 x 求导后再对 x 求二阶导数的求值
    e1 = Subs(Derivative(f(x), x), x, x)
    assert e1.diff(x) == Derivative(f(x), x, x)  # 断言 e1 对 x 的导数等于 f(x) 的二阶导数

    # 创建 e2 对象，表示对 f(x) 对 x 求导后，在 x = x^2 + 1 处求导数
    e2 = Subs(Derivative(f(x), x), x, x**2 + 1)
    assert e2.diff(x) == 2*x*Subs(Derivative(f(x), x, x), x, x**2 + 1)  # 断言 e2 对 x 的导数

    # 创建 e3 对象，表示对 f(x) + y^2 - y 对 y 求导
    e3 = Subs(Derivative(f(x) + y**2 - y, y), y, y**2)
    assert e3.diff(y) == 4*y  # 断言 e3 对 y 的导数为 4*y

    # 创建 e4 对象，表示对 f(x + y) 对 y 求导，在 y = x^2 处求导数
    e4 = Subs(Derivative(f(x + y), y), y, x**2)
    assert e4.diff(y) is S.Zero  # 断言 e4 对 y 的导数为零

    # 创建 e5 对象，表示对 f(x) 对 x 求导，将 (y, z) 替换为 (y, z)
    e5 = Subs(Derivative(f(x), x), (y, z), (y, z))
    assert e5.diff(x) == Derivative(f(x), x, x)  # 断言 e5 对 x 的导数等于 f(x) 的二阶导数

    # 断言 f(g(x)) 对 g(x) 和 g(x) 的求导结果
    assert f(g(x)).diff(g(x), g(x)) == Derivative(f(g(x)), g(x), g(x))


# 测试问题 13843 的函数
def test_issue_13843():
    x = symbols('x')  # 定义符号变量 x
    f = Function('f')  # 定义函数 f
    m, n = symbols('m n', integer=True)  # 定义整数符号变量 m 和 n
    # 断言对 f(x) 在 x 上连续求导 m 次，再连续求导 n 次的结果
    assert Derivative(Derivative(f(x), (x, m)), (x, n)) == Derivative(f(x), (x, m + n))
    # 断言对 f(x) 在 x 上连续求导 m+5 次，再连续求导 n+3 次的结果
    assert Derivative(Derivative(f(x), (x, m+5)), (x, n+3)) == Derivative(f(x), (x, m + n + 8))
    # 断言对 f(x) 在 x 上连续求导 n 次的结果
    assert Derivative(f(x), (x, n)).doit() == Derivative(f(x), (x, n))


# 测试 order 可以为零的函数
def test_order_could_be_zero():
    x, y = symbols('x, y')  # 定义符号变量 x 和 y
    n = symbols('n', integer=True, nonnegative=True)  # 定义非负整数符号变量 n
    m = symbols('m', integer=True, positive=True)  # 定义正整数符号变量 m
    # 断言对 y 在 x 上连续求导 n 次的结果
    assert diff(y, (x, n)) == Piecewise((y, Eq(n, 0)), (0, True))
    # 断言对 y 在 x 上连续求导 n+1 次的结果为零
    assert diff(y, (x, n + 1)) is S.Zero
    # 断言对 y 在 x 上连续求导 m 次的结果为零
    assert diff(y, (x, m)) is S.Zero


# 测试未定义函数的等式问题
def test_undefined_function_eq():
    f = Function('f')  # 定义函数 f
    f2 = Function('f')  # 定义另一个同名函数 f2
    g = Function('g')  # 定义函数 g
    f_real = Function('f', is_real=True)  # 定义实数函数 f_real

    # 断言 f 等于 f2
    assert f == f2
    # 断言 f 和 f2 的哈希值相等
    assert hash(f) == hash(f2)
    # 断言 f 等于 f 自身
    assert f == f

    # 断言 f 不等于 g
    assert f != g

    # 断言 f 不等于 f_real
    assert f != f_real


# 测试函数的假设条件
def test_function_assumptions():
    x = Symbol('x')  # 定义符号变量 x
    f = Function('f')  # 定义函数 f
    f_real = Function('f', real=True)  # 定义实数函数 f_real
    f_real1 = Function('f', real=1)  # 定义实数函数 f_real1
    f_real_inherit = Function(Symbol('f', real=True))  # 定义继承了实数属性的函数 f_real_inherit

    # 断言 f_real 等于 f_real1，即假设被规范化
    assert f_real == f_real1
    # 断言 f 不等于 f_real
    assert f != f_real
    # 断言 f(x) 不等于 f_real(x)
    assert f(x) != f_real(x)

    # 断言 f(x) 的 is_real 属性为 None
    assert f(x).is_real is None
    # 断言 f_real(x) 的 is_real 属性为 True
    assert f_real(x).is_real is True
    # 断言 f_real_inherit(x) 的 is_real 属性为 True，并且名称为 'f'
    assert f_real_inherit(x).is_real is True and f_real_inherit.name == 'f'

    # 另一种方式定义 f_real2，但由于 UndefinedFunction.__new__ 的工作方式，它不等于 f_real
    f_real2 = Function('f', is_real=True)
    assert f_real2(x).is_real is True


# 测试未定义函数的浮点数问题 6938
def test_undef_fcn_float_issue_6938():
    f = Function('ceil')  # 定义函数 ceil
    # 断言 f(0.3) 不是一个数值
    assert not f(0.3).is_number
    f = Function('sin')  # 定义函数 sin
    # 断言 f(0.3) 不是一个数值
    assert not f(0.3).is_number
    # 断言 f(pi) 的 evalf 结果不是一个数值
    assert not f(pi).evalf().is_number
    x = Symbol('x')  # 定义符号变量 x
    # 断言 f(x) 在 x = 1.2 处的 evalf 结
    # Issue 15170. Make sure UndefinedFunction with eval defined works
    # properly.

    # 定义一个匿名函数 fdiff，接受一个参数 argindex，默认为 1，返回 self.args[argindex - 1] 的余弦值
    fdiff = lambda self, argindex=1: cos(self.args[argindex - 1])

    # 定义一个类方法 eval，接受一个参数 t，直接返回 None
    eval = classmethod(lambda cls, t: None)

    # 定义一个类方法 _imp_，接受一个参数 t，返回 sin(t)
    _imp_ = classmethod(lambda cls, t: sin(t))

    # 使用 Function 类创建一个新函数 temp，传入参数 fdiff、eval、_imp_，并赋值给 temp 变量
    temp = Function('temp', fdiff=fdiff, eval=eval, _imp_=_imp_)

    # 创建一个表达式 expr，调用 temp 函数并传入参数 t
    expr = temp(t)

    # 断言 sympify 函数处理后的 expr 等于原始的 expr
    assert sympify(expr) == expr

    # 断言 sympify 处理后的 expr 的类型的 fdiff 属性的名称为 "<lambda>"
    assert type(sympify(expr)).fdiff.__name__ == "<lambda>"

    # 断言 expr 对 t 的导数等于 cos(t)
    assert expr.diff(t) == cos(t)
def test_issue_15241():
    # 定义函数 F 为 x 的函数
    F = f(x)
    # 对 F 关于 x 求导数
    Fx = F.diff(x)
    # 断言：对表达式 (F + x*Fx) 关于 x 求 Fx 的二阶导数应为 2
    assert (F + x*Fx).diff(x, Fx) == 2
    # 断言：对表达式 (F + x*Fx) 关于 Fx 求 x 的导数应为 1
    assert (F + x*Fx).diff(Fx, x) == 1
    # 断言：对表达式 (x*F + x*Fx*F) 关于 F 求 x 的导数应为 x*Fx 关于 x 的导数 + Fx + 1
    assert (x*F + x*Fx*F).diff(F, x) == x*Fx.diff(x) + Fx + 1
    # 断言：对表达式 (x*F + x*Fx*F) 关于 x 求 F 的导数应为 x*Fx 关于 x 的导数 + Fx + 1
    assert (x*F + x*Fx*F).diff(x, F) == x*Fx.diff(x) + Fx + 1


def test_issue_15226():
    # 断言：Subs(Derivative(f(y), x, y), y, g(x)) 的求导结果不等于 0
    assert Subs(Derivative(f(y), x, y), y, g(x)).doit() != 0


def test_issue_7027():
    # 对于 wrt 中的每个值（cos(x), re(x), Derivative(cos(x), x)），断言调用 diff(f(x), wrt) 会引发 ValueError
    for wrt in (cos(x), re(x), Derivative(cos(x), x)):
        raises(ValueError, lambda: diff(f(x), wrt))


def test_derivative_quick_exit():
    # 断言：f(x) 关于 y 的导数为 0
    assert f(x).diff(y) == 0
    # 断言：f(x) 关于 y 和 f(x) 的导数为 0
    assert f(x).diff(y, f(x)) == 0
    # 断言：f(x) 关于 x 和 f(y) 的导数为 0
    assert f(x).diff(x, f(y)) == 0
    # 断言：f(f(x)) 关于 x, f(x), f(y) 的导数为 0
    assert f(f(x)).diff(x, f(x), f(y)) == 0
    # 断言：f(f(x)) 关于 x, f(x), y 的导数为 0
    assert f(f(x)).diff(x, f(x), y) == 0
    # 断言：f(x) 关于 g(x) 的导数为 0
    assert f(x).diff(g(x)) == 0
    # 断言：f(x) 关于 x 和 f(x) 关于 x 的导数为 1
    assert f(x).diff(x, f(x).diff(x)) == 1
    # 定义 df 为 f(x) 关于 x 的导数
    df = f(x).diff(x)
    # 断言：f(x) 关于 df 的导数为 0
    assert f(x).diff(df) == 0
    # 定义 dg 为 g(x) 关于 x 的导数
    dg = g(x).diff(x)
    # 断言：dg 关于 df 的导数为 0
    assert dg.diff(df).doit() == 0


def test_issue_15084_13166():
    # 定义 eq 为 f(x, g(x))
    eq = f(x, g(x))
    # 断言：eq 关于 (g(x), y) 的偏导数应为 Derivative(f(x, g(x)), (g(x), y))
    assert eq.diff((g(x), y)) == Derivative(f(x, g(x)), (g(x), y))
    # issue 13166
    # 断言：eq 关于 x 的二阶导数应为给定的复杂表达式
    assert eq.diff(x, 2).doit() == (
        (Derivative(f(x, g(x)), (g(x), 2))*Derivative(g(x), x) +
        Subs(Derivative(f(x, _xi_2), _xi_2, x), _xi_2, g(x)))*Derivative(g(x),
        x) + Derivative(f(x, g(x)), g(x))*Derivative(g(x), (x, 2)) +
        Derivative(g(x), x)*Subs(Derivative(f(_xi_1, g(x)), _xi_1, g(x)),
        _xi_1, x) + Subs(Derivative(f(_xi_1, g(x)), (_xi_1, 2)), _xi_1, x))
    # issue 6681
    # 断言：diff(f(x, t, g(x, t)), x) 关于 x 的导数应为给定的复杂表达式
    assert diff(f(x, t, g(x, t)), x).doit() == (
        Derivative(f(x, t, g(x, t)), g(x, t))*Derivative(g(x, t), x) +
        Subs(Derivative(f(_xi_1, t, g(x, t)), _xi_1), _xi_1, x))
    # 断言：eq 关于 x, g(x) 的导数应等价于 eq 关于 g(x), x 的导数
    assert eq.diff(x, g(x)) == eq.diff(g(x), x)


def test_negative_counts():
    # issue 13873
    # 断言：sin(x) 关于 x 的导数为负数会引发 ValueError
    raises(ValueError, lambda: sin(x).diff(x, -1))


def test_Derivative__new__():
    # 断言：调用 f(x).diff((x, 2), 0) 会引发 TypeError
    raises(TypeError, lambda: f(x).diff((x, 2), 0))
    # 断言：f(x, y) 关于 [(x, y), 0] 的偏导数应为 f(x, y)
    assert f(x, y).diff([(x, y), 0]) == f(x, y)
    # 断言：f(x, y) 关于 [(x, y), 1] 的偏导数应为 NDimArray([Derivative(f(x, y), x), Derivative(f(x, y), y)])
    assert f(x, y).diff([(x, y), 1]) == NDimArray([
        Derivative(f(x, y), x), Derivative(f(x, y), y)])
    # 断言：f(x,y) 关于 y, (x, z), y, x 的偏导数应为 Derivative(f(x, y), (x, z + 1), (y, 2))
    assert f(x,y).diff(y, (x, z), y, x) == Derivative(
        f(x, y), (x, z + 1), (y, 2))
    # 断言：Matrix([x]) 关于 x 的二阶导数应为 Matrix([0])
    assert Matrix([x]).diff(x, 2) == Matrix([0])  # is_zero exit


def test_issue_14719_10150():
    class V(Expr):
        _diff_wrt = True
        is_scalar = False
    # 断言：V() 关于 V() 的导数应为 Derivative(V(), V())
    assert V().diff(V()) == Derivative(V(), V())
    # 断言：(2*V()) 关于 V() 的导数应为 2*Derivative(V(), V())
    assert (2*V()).diff(V()) == 2*Derivative(V(), V())
    class X(Expr):
        _diff_wrt = True
    # 断言：X() 关于 X() 的导数应为 1
    assert X().diff(X()) == 1
    # 断言：(2*X()) 关于 X() 的导数应为 2
    assert (2*X()).diff(X()) == 2


def test_noncommutative_issue_15131():
    # 定义符号 x 和 t，使其不可交换
    x = Symbol('x', commutative=False)
    t = Symbol('t', commutative=False)
    # 定义 fx 和 ft 为不可交换函数 Fx(x) 和 Ft(t)
    fx = Function('Fx', commutative=False)(x)
    ft = Function('Ft', commutative=False)(t)
    # 创建一个符号变量 'A'，并指定其为非交换的
    A = Symbol('A', commutative=False)
    
    # 构建一个表达式 eq，等于 fx * A * ft，其中 fx 和 ft 是其他变量或表达式
    eq = fx * A * ft
    
    # 对表达式 eq 关于变量 t 进行求导
    eqdt = eq.diff(t)
    
    # 断言：验证 eqdt 表达式中最后一个参数是否等于 ft 关于 t 的导数
    assert eqdt.args[-1] == ft.diff(t)
# 定义测试函数，用于测试 Derivative 对象的创建和操作
def test_Subs_Derivative():
    # 创建 Derivative 对象 a，代表 f(g(x), h(x)) 对 g(x), h(x), x 的偏导数
    a = Derivative(f(g(x), h(x)), g(x), h(x), x)
    # 创建 Derivative 对象 b，代表 Derivative(f(g(x), h(x)), g(x), h(x)) 对 x 的偏导数
    b = Derivative(Derivative(f(g(x), h(x)), g(x), h(x)), x)
    # 创建 c，代表 f(g(x), h(x)) 对 g(x), h(x), x 的偏导数，使用 diff 方法计算
    c = f(g(x), h(x)).diff(g(x), h(x), x)
    # 创建 d，代表 f(g(x), h(x)) 对 g(x), h(x) 的偏导数后再对 x 的偏导数
    d = f(g(x), h(x)).diff(g(x), h(x)).diff(x)
    # 创建 Derivative 对象 e，代表 f(g(x), h(x)) 对 x 的偏导数
    e = Derivative(f(g(x), h(x)), x)
    # 将上述所有 Derivative 对象放入元组 eqs 中
    eqs = (a, b, c, d, e)
    # 定义 lambda 函数 subs，对其参数应用一系列代换，最终返回结果
    subs = lambda arg: arg.subs(f, Lambda((x, y), exp(x + y))
        ).subs(g(x), 1/x).subs(h(x), x**3)
    # 计算预期的 ans 值
    ans = 3*x**2*exp(1/x)*exp(x**3) - exp(1/x)*exp(x**3)/x**2
    # 断言所有 subs(i).doit().expand() 值都等于 ans
    assert all(subs(i).doit().expand() == ans for i in eqs)
    # 断言所有 subs(i.doit()).doit().expand() 值都等于 ans
    assert all(subs(i.doit()).doit().expand() == ans for i in eqs)

# 测试 Function 对象的名称属性
def test_issue_15360():
    f = Function('f')
    assert f.name == 'f'

# 测试 Function 对象的一些特定问题
def test_issue_15947():
    # 断言 f._diff_wrt 属性为 False
    assert f._diff_wrt is False
    # 断言 lambda 函数调用时传递 f(f) 引发 TypeError
    raises(TypeError, lambda: f(f))
    # 断言 f(x).diff(f) 引发 TypeError
    raises(TypeError, lambda: f(x).diff(f))

# 测试 diff 函数的 free_symbols 方法
def test_Derivative_free_symbols():
    f = Function('f')
    n = Symbol('n', integer=True, positive=True)
    # 断言 diff(f(x), (x, n)).free_symbols 返回 {n, x}
    assert diff(f(x), (x, n)).free_symbols == {n, x}

# 测试 Derivative 对象的计算和结果
def test_issue_20683():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    # 计算 Derivative(z, x) 在 x=0 处的值
    y = Derivative(z, x).subs(x,0)
    # 断言 y.doit() 等于 0
    assert y.doit() == 0
    # 计算 Derivative(8, x) 在 x=0 处的值
    y = Derivative(8, x).subs(x,0)
    # 断言 y.doit() 等于 0
    assert y.doit() == 0

# 测试 exp 和 cos 函数的级数展开
def test_issue_10503():
    f = exp(x**3)*cos(x**6)
    # 断言 f 在 x=0 处展开到 14 阶的级数
    assert f.series(x, 0, 14) == 1 + x**3 + x**6/2 + x**9/6 - 11*x**12/24 + O(x**14)

# 测试 solveset 函数的数值计算精度
def test_issue_17382():
    # 定义 NS 函数，将 solveset(2 * cos(x) * cos(2 * x) - 1, x, S.Reals) 的数值结果转化为字符串
    def NS(e, n=15, **options):
        return sstr(sympify(e).evalf(n, **options), full_prec=True)
    
    x = Symbol('x')
    # 解方程 2 * cos(x) * cos(2 * x) - 1 = 0，预期结果为数值集合
    expr = solveset(2 * cos(x) * cos(2 * x) - 1, x, S.Reals)
    expected = "Union(" \
               "ImageSet(Lambda(_n, 6.28318530717959*_n + 5.79812359592087), Integers), " \
               "ImageSet(Lambda(_n, 6.28318530717959*_n + 0.485061711258717), Integers))"
    # 断言 NS(expr) 的结果等于预期的 expected
    assert NS(expr) == expected

# 测试 Function 对象的 eval 方法
def test_eval_sympified():
    # 定义 F 类，继承自 Function，实现 eval 方法返回常数 1
    class F(Function):
        @classmethod
        def eval(cls, x):
            assert x is S.One
            return 1

    # 断言 F(1) 等于 S.One
    assert F(1) is S.One

    # 定义 F2 类，继承自 Function，实现 eval 方法，在 x=0 时返回字符串 '1'
    class F2(Function):
        @classmethod
        def eval(cls, x):
            if x == 0:
                return '1'

    # 断言 lambda 函数调用时传递 F2(0) 引发 SympifyError
    raises(SympifyError, lambda: F2(0))
    # 调用 F2(1)，不应引发异常
    F2(1) # Doesn't raise

# 测试 Function 对象的 eval 方法要求
def test_eval_classmethod_check():
    # 使用 raises 断言语句，检查定义 eval 方法的类 F 是否引发 TypeError
    with raises(TypeError):
        class F(Function):
            def eval(self, x):
                pass
```