# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_pickling.py`

```
# 导入 inspect 模块，用于检查对象
import inspect
# 导入 copy 模块，用于对象的浅拷贝和深拷贝
import copy
# 导入 pickle 模块，用于对象的序列化和反序列化
import pickle

# 导入 sympy.physics.units 中的 meter 单位
from sympy.physics.units import meter

# 导入 sympy.testing.pytest 中的 XFAIL, raises, ignore_warnings 装饰器
from sympy.testing.pytest import XFAIL, raises, ignore_warnings

# 导入 sympy.core.basic 中的 Atom, Basic 类
from sympy.core.basic import Atom, Basic
# 导入 sympy.core.singleton 中的 SingletonRegistry 类
from sympy.core.singleton import SingletonRegistry
# 导入 sympy.core.symbol 中的 Str, Dummy, Symbol, Wild 类
from sympy.core.symbol import Str, Dummy, Symbol, Wild
# 导入 sympy.core.numbers 中的各种数学常数和类型
from sympy.core.numbers import (E, I, pi, oo, zoo, nan, Integer,
        Rational, Float, AlgebraicNumber)
# 导入 sympy.core.relational 中的比较关系类
from sympy.core.relational import (Equality, GreaterThan, LessThan, Relational,
        StrictGreaterThan, StrictLessThan, Unequality)
# 导入 sympy.core.add 中的 Add 类
from sympy.core.add import Add
# 导入 sympy.core.mul 中的 Mul 类
from sympy.core.mul import Mul
# 导入 sympy.core.power 中的 Pow 类
from sympy.core.power import Pow
# 导入 sympy.core.function 中的各种函数类
from sympy.core.function import Derivative, Function, FunctionClass, Lambda, \
    WildFunction
# 导入 sympy.sets.sets 中的 Interval 类
from sympy.sets.sets import Interval
# 导入 sympy.core.multidimensional 中的 vectorize 函数
from sympy.core.multidimensional import vectorize

# 导入 sympy.external.gmpy 中的 gmpy 作为 _gmpy 别名
from sympy.external.gmpy import gmpy as _gmpy
# 导入 sympy.utilities.exceptions 中的 SymPyDeprecationWarning 异常
from sympy.utilities.exceptions import SymPyDeprecationWarning

# 从 sympy.core.singleton 中导入 S 单例对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 中导入 symbols 函数
from sympy.core.symbol import symbols

# 导入 sympy.external 中的 import_module 函数，并将 cloudpickle 模块导入为 cloudpickle
cloudpickle = import_module('cloudpickle')

# 定义一个集合，包含对象在进行对象比较时不相等的属性
not_equal_attrs = {
    '_assumptions',  # 缓存属性，对象创建时并不会自动填充
    '_mhash',   # 调用 __hash__ 方法后缓存属性，但在创建后设置为 None
}

# 定义一个集合，包含已经废弃的属性名称
deprecated_attrs = {
    'is_EmptySet',  # 自 SymPy 1.5 开始已废弃
    'expr_free_symbols',  # 自 SymPy 1.9 开始已废弃
}

def check(a, exclude=[], check_attr=True, deprecated=()):
    """ Check that pickling and copying round-trips.
    """
    # 对于 Basic 类的实例，禁用协议 0 和 1 的 pickle 操作
    if isinstance(a, Basic):
        for protocol in [0, 1]:
            raises(NotImplementedError, lambda: pickle.dumps(a, protocol))

    # 定义协议列表，用于测试对象的序列化和复制操作
    protocols = [2, copy.copy, copy.deepcopy, 3, 4]
    # 如果 cloudpickle 可用，则添加 cloudpickle 到协议列表中
    if cloudpickle:
        protocols.extend([cloudpickle])
    # 对于给定的协议列表，逐个进行处理
    for protocol in protocols:
        # 如果当前协议在排除列表中，则跳过处理
        if protocol in exclude:
            continue

        # 如果当前协议是可调用的函数
        if callable(protocol):
            # 如果a是一个类的实例
            if isinstance(a, type):
                # 类无法被复制，但这是可以接受的情况，因此跳过处理
                continue
            # 使用当前协议对a进行处理，生成b
            b = protocol(a)
        # 如果当前协议是一个模块
        elif inspect.ismodule(protocol):
            # 使用当前协议的loads和dumps函数进行序列化和反序列化处理
            b = protocol.loads(protocol.dumps(a))
        else:
            # 使用pickle模块对a进行序列化和反序列化处理
            b = pickle.loads(pickle.dumps(a, protocol))

        # 获取a和b对象的属性列表
        d1 = dir(a)
        d2 = dir(b)
        # 断言a和b的属性集合应该完全相同
        assert set(d1) == set(d2)

        # 如果不需要检查属性，则继续下一个协议的处理
        if not check_attr:
            continue

        # 定义函数c，用于比较a和b对象的属性
        def c(a, b, d):
            # 遍历属性列表d
            for i in d:
                # 如果属性i在不相等属性列表中
                if i in not_equal_attrs:
                    # 如果a对象有属性i，则断言b对象也应该有属性i
                    if hasattr(a, i):
                        assert hasattr(b, i), i
                # 如果属性i在被弃用的属性列表中
                elif i in deprecated_attrs or i in deprecated:
                    # 忽略SymPyDeprecationWarning，断言a和b对象的属性i相等
                    with ignore_warnings(SymPyDeprecationWarning):
                        assert getattr(a, i) == getattr(b, i), i
                # 对于其他属性
                elif not hasattr(a, i):
                    # 如果a对象没有属性i，则跳过处理
                    continue
                else:
                    # 获取属性i的值
                    attr = getattr(a, i)
                    # 如果属性i不是一个函数
                    if not hasattr(attr, "__call__"):
                        # 断言b对象有属性i，并且其值与a对象的相同
                        assert hasattr(b, i), i
                        assert getattr(b, i) == attr, "%s != %s, protocol: %s" % (getattr(b, i), attr, protocol)

        # 调用函数c，比较a和b对象的属性
        c(a, b, d1)
        c(b, a, d2)
#================== core =========================

# 测试基础核心类和实例
def test_core_basic():
    for c in (Atom, Atom(), Basic, Basic(), SingletonRegistry, S):
        check(c)

# 测试 Str 类
def test_core_Str():
    check(Str('x'))

# 测试 symbol 相关类
def test_core_symbol():
    # 设置一个唯一的名称作为 Symbol 的名字，以避免与文件中的其他测试变量冲突
    # 在此测试后，相同名称的符号将被缓存为非交换的
    for c in (Dummy, Dummy("x", commutative=False), Symbol,
              Symbol("_issue_3130", commutative=False), Wild, Wild("x")):
        check(c)

# 测试数字类
def test_core_numbers():
    for c in (Integer(2), Rational(2, 3), Float("1.2")):
        check(c)
    for c in (AlgebraicNumber, AlgebraicNumber(sqrt(3))):
        check(c, check_attr=False)

# 测试浮点数复制的特性
def test_core_float_copy():
    # 参见 gh-7457
    y = Symbol("x") + 1.0
    check(y)  # 不会引发 TypeError（"argument is not an mpz"）

# 测试关系类
def test_core_relational():
    x = Symbol("x")
    y = Symbol("y")
    for c in (Equality, Equality(x, y), GreaterThan, GreaterThan(x, y),
              LessThan, LessThan(x, y), Relational, Relational(x, y),
              StrictGreaterThan, StrictGreaterThan(x, y), StrictLessThan,
              StrictLessThan(x, y), Unequality, Unequality(x, y)):
        check(c)

# 测试加法类
def test_core_add():
    x = Symbol("x")
    for c in (Add, Add(x, 4)):
        check(c)

# 测试乘法类
def test_core_mul():
    x = Symbol("x")
    for c in (Mul, Mul(x, 4)):
        check(c)

# 测试幂运算类
def test_core_power():
    x = Symbol("x")
    for c in (Pow, Pow(x, 4)):
        check(c)

# 测试函数相关类
def test_core_function():
    x = Symbol("x")
    for f in (Derivative, Derivative(x), Function, FunctionClass, Lambda,
              WildFunction):
        check(f)

# 测试未定义函数类
def test_core_undefinedfunctions():
    f = Function("f")
    # 下面的完全失败测试
    exclude = list(range(5))
    # https://github.com/cloudpipe/cloudpickle/issues/65
    # https://github.com/cloudpipe/cloudpickle/issues/190
    exclude.append(cloudpickle)
    check(f, exclude=exclude)

# 标记为预期失败的测试用例
@XFAIL
def test_core_undefinedfunctions_fail():
    # 这个失败是因为 f 被假定为 sympy.basic.function.f 类
    f = Function("f")
    check(f)

# 测试区间类
def test_core_interval():
    for c in (Interval, Interval(0, 2)):
        check(c)

# 测试多维度函数
def test_core_multidimensional():
    for c in (vectorize, vectorize(0)):
        check(c)

# 测试单例对象
def test_Singletons():
    protocols = [0, 1, 2, 3, 4]
    copiers = [copy.copy, copy.deepcopy]
    copiers += [lambda x: pickle.loads(pickle.dumps(x, proto))
                for proto in protocols]
    if cloudpickle:
        copiers += [lambda x: cloudpickle.loads(cloudpickle.dumps(x))]

    for obj in (Integer(-1), Integer(0), Integer(1), Rational(1, 2), pi, E, I,
                oo, -oo, zoo, nan, S.GoldenRatio, S.TribonacciConstant,
                S.EulerGamma, S.Catalan, S.EmptySet, S.IdentityFunction):
        for func in copiers:
            assert func(obj) is obj

#================== functions ===================
```python`
# 导入 sympy 库中的特定函数
from sympy.functions import (Piecewise, lowergamma, acosh, chebyshevu,
        chebyshevt, ln, chebyshevt_root, legendre, Heaviside, bernoulli, coth,
        tanh, assoc_legendre, sign, arg, asin, DiracDelta, re, rf, Abs,
        uppergamma, binomial, sinh, cos, cot, acos, acot, gamma, bell,
        hermite, harmonic, LambertW, zeta, log, factorial, asinh, acoth, cosh,
        dirichlet_eta, Eijk, loggamma, erf, ceiling, im, fibonacci,
        tribonacci, conjugate, tan, chebyshevu_root, floor, atanh, sqrt, sin,
        atan, ff, lucas, atan2, polygamma, exp)

# 定义一个函数用于测试 sympy 中的各种数学函数
def test_functions():
    # 定义包含单变量函数的元组
    one_var = (acosh, ln, Heaviside, factorial, bernoulli, coth, tanh,
            sign, arg, asin, DiracDelta, re, Abs, sinh, cos, cot, acos, acot,
            gamma, bell, harmonic, LambertW, zeta, log, factorial, asinh,
            acoth, cosh, dirichlet_eta, loggamma, erf, ceiling, im, fibonacci,
            tribonacci, conjugate, tan, floor, atanh, sin, atan, lucas, exp)
    # 定义包含双变量函数的元组
    two_var = (rf, ff, lowergamma, chebyshevu, chebyshevt, binomial,
            atan2, polygamma, hermite, legendre, uppergamma)
    # 使用 sympy 的符号函数创建符号变量 x, y, z
    x, y, z = symbols("x,y,z")
    # 定义其他函数或对象的元组
    others = (chebyshevt_root, chebyshevu_root, Eijk(x, y, z),
            Piecewise( (0, x < -1), (x**2, x <= 1), (x**3, True)),
            assoc_legendre)
    # 遍历单变量函数元组，并检查它们
    for cls in one_var:
        check(cls)
        # 创建一个单变量函数实例并检查它
        c = cls(x)
        check(c)
    # 遍历双变量函数元组，并检查它们
    for cls in two_var:
        check(cls)
        # 创建一个双变量函数实例并检查它
        c = cls(x, y)
        check(c)
    # 遍历其他函数或对象的元组，并检查它们
    for cls in others:
        check(cls)

#================== geometry ====================
# 导入 sympy 几何模块中的实体和对象
from sympy.geometry.entity import GeometryEntity
from sympy.geometry.point import Point
from sympy.geometry.ellipse import Circle, Ellipse
from sympy.geometry.line import Line, LinearEntity, Ray, Segment
from sympy.geometry.polygon import Polygon, RegularPolygon, Triangle

# 定义一个函数用于测试 sympy 几何模块中的各种几何实体和对象
def test_geometry():
    # 创建几何点对象
    p1 = Point(1, 2)
    p2 = Point(2, 3)
    p3 = Point(0, 0)
    p4 = Point(0, 1)
    # 遍历并检查各种几何实体和对象
    for c in (
        GeometryEntity, GeometryEntity(), Point, p1, Circle, Circle(p1, 2),
        Ellipse, Ellipse(p1, 3, 4), Line, Line(p1, p2), LinearEntity,
        LinearEntity(p1, p2), Ray, Ray(p1, p2), Segment, Segment(p1, p2),
        Polygon, Polygon(p1, p2, p3, p4), RegularPolygon,
            RegularPolygon(p1, 4, 5), Triangle, Triangle(p1, p2, p3)):
        # 检查每个几何对象，并禁用属性检查
        check(c, check_attr=False)

#================== integrals ====================
# 导入 sympy 积分模块中的积分对象
from sympy.integrals.integrals import Integral

# 定义一个函数用于测试 sympy 积分模块中的积分对象
def test_integrals():
    # 创建符号变量 x
    x = Symbol("x")
    # 遍历并检查积分对象
    for c in (Integral, Integral(x)):
        check(c)

#==================== logic =====================
# 导入 sympy 核心逻辑模块中的逻辑对象
from sympy.core.logic import Logic

# 定义一个函数用于测试 sympy 核心逻辑模块中的逻辑对象
def test_logic():
    # 遍历并检查逻辑对象
    for c in (Logic, Logic(1)):
        check(c)

#================== matrices ====================
# 导入 sympy 矩阵模块中的矩阵对象
from sympy.matrices import Matrix, SparseMatrix

# 定义一个函数用于测试 sympy 矩阵模块中的矩阵对象
def test_matrices():
    # 遍历并检查矩阵对象
    for c in (Matrix, Matrix([1, 2, 3]), SparseMatrix, SparseMatrix([[1, 2], [3, 4]])):
        # 检查每个矩阵对象，并禁用一些不推荐使用的属性
        check(c, deprecated=['_smat', '_mat'])

#================== ntheory =====================

*Did you know* that the concept of elliptic curves plays a fundamental role in modern cryptography, especially in the field of public key cryptography?
# 从 sympy.ntheory.generate 模块导入 Sieve 类
from sympy.ntheory.generate import Sieve

# 定义测试函数 test_ntheory
def test_ntheory():
    # 对于 Sieve 和 Sieve()，分别执行 check 函数
    for c in (Sieve, Sieve()):
        check(c)

#================== physics =====================
# 从 sympy.physics.paulialgebra 模块导入 Pauli 类
from sympy.physics.paulialgebra import Pauli
# 从 sympy.physics.units 模块导入 Unit 类
from sympy.physics.units import Unit

# 定义测试函数 test_physics
def test_physics():
    # 对于 Unit, meter, Pauli 和 Pauli(1)，分别执行 check 函数
    for c in (Unit, meter, Pauli, Pauli(1)):
        check(c)

#================== plotting ====================
# XXX: These tests are not complete, so XFAIL them

# 定义装饰器为 XFAIL 的测试函数 test_plotting
@XFAIL
def test_plotting():
    # 从 sympy.plotting.pygletplot.color_scheme 模块导入 ColorGradient 和 ColorScheme 类
    from sympy.plotting.pygletplot.color_scheme import ColorGradient, ColorScheme
    # 从 sympy.plotting.pygletplot.managed_window 模块导入 ManagedWindow 类
    from sympy.plotting.pygletplot.managed_window import ManagedWindow
    # 从 sympy.plotting.plot 模块导入 Plot 和 ScreenShot 类
    from sympy.plotting.plot import Plot, ScreenShot
    # 从 sympy.plotting.pygletplot.plot_axes 模块导入 PlotAxes, PlotAxesBase, PlotAxesFrame, PlotAxesOrdinate 类
    from sympy.plotting.pygletplot.plot_axes import PlotAxes, PlotAxesBase, PlotAxesFrame, PlotAxesOrdinate
    # 从 sympy.plotting.pygletplot.plot_camera 模块导入 PlotCamera 类
    from sympy.plotting.pygletplot.plot_camera import PlotCamera
    # 从 sympy.plotting.pygletplot.plot_controller 模块导入 PlotController 类
    from sympy.plotting.pygletplot.plot_controller import PlotController
    # 从 sympy.plotting.pygletplot.plot_curve 模块导入 PlotCurve 类
    from sympy.plotting.pygletplot.plot_curve import PlotCurve
    # 从 sympy.plotting.pygletplot.plot_interval 模块导入 PlotInterval 类
    from sympy.plotting.pygletplot.plot_interval import PlotInterval
    # 从 sympy.plotting.pygletplot.plot_mode 模块导入 PlotMode 类
    from sympy.plotting.pygletplot.plot_mode import PlotMode
    # 从 sympy.plotting.pygletplot.plot_modes 模块导入 Cartesian2D, Cartesian3D, Cylindrical, \
    # ParametricCurve2D, ParametricCurve3D, ParametricSurface, Polar, Spherical 类
    from sympy.plotting.pygletplot.plot_modes import Cartesian2D, Cartesian3D, Cylindrical, \
        ParametricCurve2D, ParametricCurve3D, ParametricSurface, Polar, Spherical
    # 从 sympy.plotting.pygletplot.plot_object 模块导入 PlotObject 类
    from sympy.plotting.pygletplot.plot_object import PlotObject
    # 从 sympy.plotting.pygletplot.plot_surface 模块导入 PlotSurface 类
    from sympy.plotting.pygletplot.plot_surface import PlotSurface
    # 从 sympy.plotting.pygletplot.plot_window 模块导入 PlotWindow 类
    from sympy.plotting.pygletplot.plot_window import PlotWindow
    # 对于 ColorGradient, ColorGradient(0.2, 0.4), ColorScheme, ManagedWindow,
    # Plot, ScreenShot, PlotAxes, PlotAxesBase, PlotAxesFrame, PlotAxesOrdinate,
    # PlotCamera, PlotController, PlotCurve, PlotInterval, PlotMode, Cartesian2D,
    # Cartesian3D, Cylindrical, ParametricCurve2D, ParametricCurve3D, ParametricSurface,
    # Polar, Spherical, PlotObject, PlotSurface, PlotWindow，分别执行 check 函数
    for c in (
        ColorGradient, ColorGradient(0.2, 0.4), ColorScheme, ManagedWindow,
        Plot, ScreenShot, PlotAxes, PlotAxesBase,
        PlotAxesFrame, PlotAxesOrdinate, PlotCamera, PlotController,
        PlotCurve, PlotInterval, PlotMode, Cartesian2D, Cartesian3D,
        Cylindrical, ParametricCurve2D, ParametricCurve3D,
        ParametricSurface, Polar, Spherical, PlotObject, PlotSurface,
        PlotWindow):
        check(c)

# 定义装饰器为 XFAIL 的测试函数 test_plotting2
@XFAIL
def test_plotting2():
    # 从 sympy.plotting.pygletplot.color_scheme 模块导入 ColorScheme 类
    from sympy.plotting.pygletplot.color_scheme import ColorScheme
    # 从 sympy.plotting.plot 模块导入 Plot 类
    from sympy.plotting.plot import Plot
    # 从 sympy.plotting.pygletplot.plot_axes 模块导入 PlotAxes 类
    from sympy.plotting.pygletplot.plot_axes import PlotAxes
    # 对于 ColorScheme, PlotAxes，分别执行 check 函数
    for c in (
        ColorScheme, ManagedWindow, Plot):
        check(c)
    # 导入 sympy.plotting.plot_window 模块中的 PlotWindow 类
    # from sympy.plotting.plot_window import PlotWindow
    # 检查彩虹色彩方案，并应用于绘图
    check(ColorScheme("rainbow"))
    # 创建一个不可见的绘图对象，参数设置为不可见
    check(Plot(1, visible=False))
    # 检查绘图坐标轴，添加默认的绘图坐标轴
    check(PlotAxes())
#================== polys =======================
# 从 sympy.polys.domains.integerring 模块导入 ZZ 类
from sympy.polys.domains.integerring import ZZ
# 从 sympy.polys.domains.rationalfield 模块导入 QQ 类
from sympy.polys.domains.rationalfield import QQ
# 从 sympy.polys.orderings 模块导入 lex 函数
from sympy.polys.orderings import lex
# 从 sympy.polys.polytools 模块导入 Poly 类
from sympy.polys.polytools import Poly

# 定义测试函数 test_pickling_polys_polytools
def test_pickling_polys_polytools():
    # 从 sympy.polys.polytools 模块导入 PurePoly 类
    from sympy.polys.polytools import PurePoly
    # 导入 Symbol 类
    x = Symbol('x')

    # 对于 Poly 和 Poly(x, x)，分别进行检查
    for c in (Poly, Poly(x, x)):
        check(c)

    # 对于 PurePoly 和 PurePoly(x)，分别进行检查
    for c in (PurePoly, PurePoly(x)):
        check(c)

    # TODO: 修复 Options 类的序列化问题（参见 GroebnerBasis._options）
    # 以下代码目前被注释掉，不执行
    # for c in (GroebnerBasis, GroebnerBasis([x**2 - 1], x, order=lex)):
    #     check(c)

# 定义测试函数 test_pickling_polys_polyclasses
def test_pickling_polys_polyclasses():
    # 从 sympy.polys.polyclasses 模块导入 DMP, DMF, ANP 类
    from sympy.polys.polyclasses import DMP, DMF, ANP

    # 对于 DMP 类和其实例化参数，分别进行检查
    for c in (DMP, DMP([[ZZ(1)], [ZZ(2)], [ZZ(3)]], ZZ)):
        check(c, deprecated=['rep'])
    # 对于 DMF 类和其实例化参数，分别进行检查
    for c in (DMF, DMF(([ZZ(1), ZZ(2)], [ZZ(1), ZZ(3)]), ZZ)):
        check(c)
    # 对于 ANP 类和其实例化参数，分别进行检查
    for c in (ANP, ANP([QQ(1), QQ(2)], [QQ(1), QQ(2), QQ(3)], QQ)):
        check(c)

# 定义 XFAIL 装饰器修饰的测试函数 test_pickling_polys_rings，用于标记预期失败的测试
@XFAIL
def test_pickling_polys_rings():
    # 注意: 不能使用协议 < 2，因为必须执行 __new__ 以确保环的缓存正常工作。
    
    # 从 sympy.polys.rings 模块导入 PolyRing 类
    from sympy.polys.rings import PolyRing

    # 创建 PolyRing 对象，定义环 "x,y,z"，使用整数环 ZZ 和 lex 排序
    ring = PolyRing("x,y,z", ZZ, lex)

    # 对于 PolyRing 类和其实例化参数，分别进行检查，但排除索引 0 和 1
    for c in (PolyRing, ring):
        check(c, exclude=[0, 1])

    # 对于 ring.dtype 和 ring.one，分别进行检查，但排除索引 0 和 1，并且不检查属性
    for c in (ring.dtype, ring.one):
        check(c, exclude=[0, 1], check_attr=False) # TODO: Py3k

# 定义测试函数 test_pickling_polys_fields
def test_pickling_polys_fields():
    # 注意: 不能使用协议 < 2，因为必须执行 __new__ 以确保字段的缓存正常工作。
    
    # 以下代码被注释掉，不执行
    # from sympy.polys.fields 模块导入 FracField 类
    # from sympy.polys.fields import FracField

    # 创建 FracField 对象，定义字段 "x,y,z"，使用整数环 ZZ 和 lex 排序
    # field = FracField("x,y,z", ZZ, lex)

    # TODO: AssertionError: assert id(obj) not in self.memo
    # for c in (FracField, field):
    #     check(c, exclude=[0, 1])

    # TODO: AssertionError: assert id(obj) not in self.memo
    # for c in (field.dtype, field.one):
    #     check(c, exclude=[0, 1])

# 定义测试函数 test_pickling_polys_elements
def test_pickling_polys_elements():
    # 从 sympy.polys.domains.pythonrational 模块导入 PythonRational 类
    from sympy.polys.domains.pythonrational import PythonRational
    # 以下代码被注释掉，不执行
    #from sympy.polys.domains.pythonfinitefield 模块导入 PythonFiniteField 类
    #from sympy.polys.domains.pythonfinitefield import PythonFiniteField
    #from sympy.polys.domains.mpelements 模块导入 MPContext 类
    #from sympy.polys.domains.mpelements import MPContext

    # 对于 PythonRational 类和其实例化参数，分别进行检查
    for c in (PythonRational, PythonRational(1, 7)):
        check(c)

    # 以下代码被注释掉，不执行
    # gf = PythonFiniteField(17)

    # TODO: fix pickling of ModularInteger
    # for c in (gf.dtype, gf(5)):
    #     check(c)

    # TODO: fix pickling of RealElement
    # mp = MPContext()
    # for c in (mp.mpf, mp.mpf(1.0)):
    #     check(c)

    # TODO: fix pickling of ComplexElement
    # for c in (mp.mpc, mp.mpc(1.0, -1.5)):
    #     check(c)

# 定义测试函数 test_pickling_polys_domains
def test_pickling_polys_domains():
    # 从 sympy.polys.domains.pythonintegerring 模块导入 PythonIntegerRing 类
    from sympy.polys.domains.pythonintegerring import PythonIntegerRing
    # 从 sympy.polys.domains.pythonrationalfield 模块导入 PythonRationalField 类
    from sympy.polys.domains.pythonrationalfield import PythonRationalField

    # TODO: fix pickling of ModularInteger
    # for c in (PythonFiniteField, PythonFiniteField(17)):
    #     check(c)

    # 对于 PythonIntegerRing 类和其实例化参数，分别进行检查
    for c in (PythonIntegerRing, PythonIntegerRing()):
        check(c)
    # 对于PythonIntegerRing和PythonRationalField类的实例化，分别执行check函数
    for c in (PythonIntegerRing, PythonIntegerRing()):
        check(c, check_attr=False)

    # 对于PythonRationalField和PythonRationalField类的实例化，分别执行check函数
    for c in (PythonRationalField, PythonRationalField()):
        check(c, check_attr=False)

    # 如果_gmpy模块可用，则执行以下操作
    if _gmpy is not None:
        # 导入GMPYIntegerRing和GMPYRationalField类
        from sympy.polys.domains.gmpyintegerring import GMPYIntegerRing
        from sympy.polys.domains.gmpyrationalfield import GMPYRationalField

        # 对GMPYIntegerRing类和其实例化的对象分别执行check函数
        for c in (GMPYIntegerRing, GMPYIntegerRing()):
            check(c, check_attr=False)

        # 对GMPYRationalField类和其实例化的对象分别执行check函数
        for c in (GMPYRationalField, GMPYRationalField()):
            check(c, check_attr=False)

    # 导入AlgebraicField和ExpressionDomain类
    from sympy.polys.domains.algebraicfield import AlgebraicField
    from sympy.polys.domains.expressiondomain import ExpressionDomain

    # 对AlgebraicField类和其实例化的对象分别执行check函数
    for c in (AlgebraicField, AlgebraicField(QQ, sqrt(3))):
        check(c, check_attr=False)

    # 对ExpressionDomain类和其实例化的对象分别执行check函数
    for c in (ExpressionDomain, ExpressionDomain()):
        check(c, check_attr=False)
# 定义测试函数 test_pickling_polys_orderings，用于测试 sympy.polys.orderings 模块中的排序类
def test_pickling_polys_orderings():
    # 从 sympy.polys.orderings 模块中导入排序类 LexOrder, GradedLexOrder, ReversedGradedLexOrder, InverseOrder
    from sympy.polys.orderings import (LexOrder, GradedLexOrder,
        ReversedGradedLexOrder, InverseOrder)
    
    # 对 LexOrder 类及其实例进行测试
    for c in (LexOrder, LexOrder()):
        # 调用 check 函数进行测试
        check(c)
    
    # 对 GradedLexOrder 类及其实例进行测试
    for c in (GradedLexOrder, GradedLexOrder()):
        # 调用 check 函数进行测试
        check(c)
    
    # 对 ReversedGradedLexOrder 类及其实例进行测试
    for c in (ReversedGradedLexOrder, ReversedGradedLexOrder()):
        # 调用 check 函数进行测试
        check(c)
    
    # TODO: 哎呀，Python 太天真了。在 pickling 模块中不支持 lambda 函数或内部函数。也许有人能想出如何处理这个问题。
    #
    # for c in (ProductOrder, ProductOrder((LexOrder(),       lambda m: m[:2]),
    #                                      (GradedLexOrder(), lambda m: m[2:]))):
    #     check(c)
    
    # 对 InverseOrder 类及其实例进行测试
    for c in (InverseOrder, InverseOrder(LexOrder())):
        # 调用 check 函数进行测试
        check(c)

# 定义测试函数 test_pickling_polys_monomials，用于测试 sympy.polys.monomials 模块中的单项式类
def test_pickling_polys_monomials():
    # 从 sympy.polys.monomials 模块中导入 MonomialOps 和 Monomial 类
    from sympy.polys.monomials import MonomialOps, Monomial
    # 导入符号变量 x, y, z
    x, y, z = symbols("x,y,z")
    
    # 对 MonomialOps 类及其实例进行测试
    for c in (MonomialOps, MonomialOps(3)):
        # 调用 check 函数进行测试
        check(c)
    
    # 对 Monomial 类及其实例进行测试
    for c in (Monomial, Monomial((1, 2, 3), (x, y, z))):
        # 调用 check 函数进行测试
        check(c)

# 定义测试函数 test_pickling_polys_errors，用于测试 sympy.polys.polyerrors 模块中的异常类
def test_pickling_polys_errors():
    # 从 sympy.polys.polyerrors 模块中导入多个异常类
    from sympy.polys.polyerrors import (HeuristicGCDFailed,
        HomomorphismFailed, IsomorphismFailed, ExtraneousFactors,
        EvaluationFailed, RefinementFailed, CoercionFailed, NotInvertible,
        NotReversible, NotAlgebraic, DomainError, PolynomialError,
        UnificationFailed, GeneratorsError, GeneratorsNeeded,
        UnivariatePolynomialError, MultivariatePolynomialError, OptionError,
        FlagError)
    # 导入 ExactQuotientFailed 异常类
    # from sympy.polys.polyerrors import (ExactQuotientFailed,
    #         OperationNotSupported, ComputationFailed, PolificationFailed)

    # 定义符号变量 x
    # x = Symbol('x')

    # TODO: TypeError: __init__() takes at least 3 arguments (1 given)
    # for c in (ExactQuotientFailed, ExactQuotientFailed(x, 3*x, ZZ)):
    #    check(c)

    # TODO: TypeError: can't pickle instancemethod objects
    # for c in (OperationNotSupported, OperationNotSupported(Poly(x), Poly.gcd)):
    #    check(c)

    # 对 HeuristicGCDFailed 类及其实例进行测试
    for c in (HeuristicGCDFailed, HeuristicGCDFailed()):
        # 调用 check 函数进行测试
        check(c)

    # 对 HomomorphismFailed 类及其实例进行测试
    for c in (HomomorphismFailed, HomomorphismFailed()):
        # 调用 check 函数进行测试
        check(c)

    # 对 IsomorphismFailed 类及其实例进行测试
    for c in (IsomorphismFailed, IsomorphismFailed()):
        # 调用 check 函数进行测试
        check(c)

    # 对 ExtraneousFactors 类及其实例进行测试
    for c in (ExtraneousFactors, ExtraneousFactors()):
        # 调用 check 函数进行测试
        check(c)

    # 对 EvaluationFailed 类及其实例进行测试
    for c in (EvaluationFailed, EvaluationFailed()):
        # 调用 check 函数进行测试
        check(c)

    # 对 RefinementFailed 类及其实例进行测试
    for c in (RefinementFailed, RefinementFailed()):
        # 调用 check 函数进行测试
        check(c)

    # 对 CoercionFailed 类及其实例进行测试
    for c in (CoercionFailed, CoercionFailed()):
        # 调用 check 函数进行测试
        check(c)

    # 对 NotInvertible 类及其实例进行测试
    for c in (NotInvertible, NotInvertible()):
        # 调用 check 函数进行测试
        check(c)

    # 对 NotReversible 类及其实例进行测试
    for c in (NotReversible, NotReversible()):
        # 调用 check 函数进行测试
        check(c)

    # 对 NotAlgebraic 类及其实例进行测试
    for c in (NotAlgebraic, NotAlgebraic()):
        # 调用 check 函数进行测试
        check(c)

    # 对 DomainError 类及其实例进行测试
    for c in (DomainError, DomainError()):
        # 调用 check 函数进行测试
        check(c)

    # 对 PolynomialError 类及其实例进行测试
    for c in (PolynomialError, PolynomialError()):
        # 调用 check 函数进行测试
        check(c)

    # 对 UnificationFailed 类及其实例进行测试
    for c in (UnificationFailed, UnificationFailed()):
        # 调用 check 函数进行测试
        check(c)
    # 对于 GeneratorsError 和 GeneratorsError()，分别进行检查
    for c in (GeneratorsError, GeneratorsError()):
        check(c)

    # 对于 GeneratorsNeeded 和 GeneratorsNeeded()，分别进行检查
    for c in (GeneratorsNeeded, GeneratorsNeeded()):
        check(c)

    # TODO: PicklingError: Can't pickle <function <lambda> at 0x38578c0>: it's not found as __main__.<lambda>
    # 对于 ComputationFailed 和 ComputationFailed(lambda t: t, 3, None)，分别进行检查
    # 由于涉及到 lambda 函数，可能会引发 PicklingError
    # for c in (ComputationFailed, ComputationFailed(lambda t: t, 3, None)):
    #    check(c)

    # 对于 UnivariatePolynomialError 和 UnivariatePolynomialError()，分别进行检查
    for c in (UnivariatePolynomialError, UnivariatePolynomialError()):
        check(c)

    # 对于 MultivariatePolynomialError 和 MultivariatePolynomialError()，分别进行检查
    for c in (MultivariatePolynomialError, MultivariatePolynomialError()):
        check(c)

    # TODO: TypeError: __init__() takes at least 3 arguments (1 given)
    # 对于 PolificationFailed 和 PolificationFailed({}, x, x, False)，分别进行检查
    # 由于参数不匹配，可能会引发 TypeError
    # for c in (PolificationFailed, PolificationFailed({}, x, x, False)):
    #    check(c)

    # 对于 OptionError 和 OptionError()，分别进行检查
    for c in (OptionError, OptionError()):
        check(c)

    # 对于 FlagError 和 FlagError()，分别进行检查
    for c in (FlagError, FlagError()):
        check(c)
# TODO: 定义测试函数test_pickling_polys_options()
#    从sympy.polys.polyoptions模块导入Options类
#    TODO: 修复`symbols`标志的序列化问题
#    遍历Options类的实例和另一个具有指定参数（domain='ZZ', polys=False）的Options类的实例
#        调用check函数检查每个实例

# TODO: 定义测试函数test_pickling_polys_rootisolation()
#    从sympy.polys.rootoftools模块导入CRootOf和RootSum类
#    创建符号变量x
#    定义方程f = x**3 + x + 3
#    遍历CRootOf类及其f和0作为参数的实例
#        调用check函数检查每个实例
#    遍历RootSum类及其f和exp作为参数的实例
#        调用check函数检查每个实例

#================== printing ====================
# 从sympy.printing.latex模块导入LatexPrinter类
# 从sympy.printing.mathml模块导入MathMLContentPrinter和MathMLPresentationPrinter类
# 从sympy.printing.pretty.pretty模块导入PrettyPrinter类
# 从sympy.printing.pretty.stringpict模块导入prettyForm和stringPict类
# 从sympy.printing.printer模块导入Printer类
# 从sympy.printing.python模块导入PythonPrinter类

# 定义测试函数test_printing()
#    遍历LatexPrinter类、LatexPrinter的实例、MathMLContentPrinter类、
#    MathMLPresentationPrinter类、PrettyPrinter类、prettyForm类、
#    stringPict类及其"a"作为参数的实例、Printer类、Printer的实例、
#    PythonPrinter类和PythonPrinter的实例
#        调用check函数检查每个实例

# 定义装饰器为XFAIL的测试函数test_printing1()
#    调用check函数检查MathMLContentPrinter类的实例

# 定义装饰器为XFAIL的测试函数test_printing2()
#    调用check函数检查MathMLPresentationPrinter类的实例

# 定义装饰器为XFAIL的测试函数test_printing3()
#    调用check函数检查PrettyPrinter类的实例

#================== series ======================
# 从sympy.series.limits模块导入Limit类
# 从sympy.series.order模块导入Order类

# 定义测试函数test_series()
#    创建符号变量e和x
#    遍历Limit类、Limit类的实例e, x, 1作为参数、Order类和Order类的实例e作为参数
#        调用check函数检查每个实例

#================== concrete ==================
# 从sympy.concrete.products模块导入Product类
# 从sympy.concrete.summations模块导入Sum类

# 定义测试函数test_concrete()
#    创建符号变量x
#    遍历Product类、Product类的实例x, (x, 2, 4)作为参数、Sum类和Sum类的实例x, (x, 2, 4)作为参数
#        调用check函数检查每个实例

# 定义测试函数test_deprecation_warning()
#    创建SymPyDeprecationWarning类的实例w，包括消息、自1.0版本起弃用以及“active-deprecations”作为活跃弃用的目标
#    调用check函数检查w实例

#================= old pickles =================
# 定义测试函数test_unpickle_from_older_versions()
#    定义数据data，包括二进制序列化数据
#    断言反序列化data后得到的对象与sqrt(2)相等
```