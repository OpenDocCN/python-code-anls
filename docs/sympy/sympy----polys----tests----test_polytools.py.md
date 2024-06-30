# `D:\src\scipysrc\sympy\sympy\polys\tests\test_polytools.py`

```
"""Tests for user-friendly public interface to polynomial functions. """

# 导入pickle模块，用于序列化和反序列化对象
import pickle

# 导入sympy库中的多项式相关模块和函数
from sympy.polys.polytools import (
    Poly, PurePoly, poly,  # 多项式相关工具
    parallel_poly_from_expr,  # 并行多项式创建
    degree, degree_list,  # 多项式的次数计算
    total_degree,  # 多项式的总次数计算
    LC, LM, LT,  # 多项式的首项系数、首项、最高次项
    pdiv, prem, pquo, pexquo,  # 多项式的除法和余数计算
    div, rem, quo, exquo,  # 多项式的除法运算
    half_gcdex, gcdex, invert,  # 多项式的半扩展欧几里得算法、扩展欧几里得算法、求逆元
    subresultants,  # 多项式的子结果列
    resultant, discriminant,  # 多项式的结果式、判别式
    terms_gcd, cofactors,  # 多项式的最大公因式、因子分解
    gcd, gcd_list,  # 多项式的最大公因数、最大公因数列表
    lcm, lcm_list,  # 多项式的最小公倍数、最小公倍数列表
    trunc,  # 多项式的截断
    monic, content, primitive,  # 多项式的首一化、内容、原始多项式
    compose, decompose,  # 多项式的复合、分解
    sturm,  # 多项式的Sturm链
    gff_list, gff,  # 多项式的Galois域因子分解、Galois域因子分解
    sqf_norm, sqf_part, sqf_list, sqf,  # 多项式的平方因子、因子分解、平方因子分解、因子分解
    factor_list, factor,  # 多项式的因子列表、因子分解
    intervals, refine_root, count_roots,  # 多项式的区间、根的精细化、根的计数
    all_roots, real_roots, nroots, ground_roots,  # 多项式的所有根、实根、数值根、整数根
    nth_power_roots_poly,  # 多项式的n次方根多项式
    cancel, reduced, groebner,  # 多项式的取消、约简、Groebner基
    GroebnerBasis, is_zero_dimensional,  # Groebner基、判断是否为零维
    _torational_factor_list,  # 将多项式转换为有理系数
    to_rational_coeffs  # 多项式的有理系数
)

# 导入sympy中的多项式相关错误类
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,  # 多元多项式错误
    ExactQuotientFailed,  # 精确除法失败
    PolificationFailed,  # 多项式化失败
    ComputationFailed,  # 计算失败
    UnificationFailed,  # 统一失败
    RefinementFailed,  # 精细化失败
    GeneratorsNeeded,  # 需要生成器
    GeneratorsError,  # 生成器错误
    PolynomialError,  # 多项式错误
    CoercionFailed,  # 强制转换失败
    DomainError,  # 域错误
    OptionError,  # 选项错误
    FlagError  # 标志错误
)

# 导入sympy中的多项式类
from sympy.polys.polyclasses import DMP

# 导入sympy中的多项式领域
from sympy.polys.fields import field
from sympy.polys.domains import (
    FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX  # 有限域、整数环、有理数环、高精度整数环、高精度有理数环、实数环、表达式环
)
from sympy.polys.domains.realfield import RealField  # 实数域
from sympy.polys.domains.complexfield import ComplexField  # 复数域
from sympy.polys.orderings import lex, grlex, grevlex  # 词典序、逆词典序、反词典序

# 导入sympy中的Galois群相关模块
from sympy.combinatorics.galois import S4TransitiveSubgroups

# 导入sympy中的基本数学运算类和函数
from sympy.core.add import Add  # 加法运算
from sympy.core.basic import _aresame  # 比较对象是否相同
from sympy.core.containers import Tuple  # 元组
from sympy.core.expr import Expr  # 表达式
from sympy.core.function import (Derivative, diff, expand)  # 导数、求导、展开
from sympy.core.mul import _keep_coeff, Mul  # 保持系数、乘法
from sympy.core.numbers import (
    Float, I, Integer, Rational, oo, pi  # 浮点数、虚数单位、整数、有理数、无穷大、圆周率
)
from sympy.core.power import Pow  # 幂运算
from sympy.core.relational import Eq  # 等式
from sympy.core.singleton import S  # 单例
from sympy.core.symbol import Symbol  # 符号
from sympy.functions.elementary.complexes import (im, re)  # 复数的虚部、实部
from sympy.functions.elementary.exponential import exp  # 指数函数
from sympy.functions.elementary.hyperbolic import tanh  # 双曲正切函数
from sympy.functions.elementary.miscellaneous import sqrt  # 平方根
from sympy.functions.elementary.piecewise import Piecewise  # 分段函数
from sympy.functions.elementary.trigonometric import sin  # 正弦函数
from sympy.matrices.dense import Matrix  # 矩阵
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 矩阵符号
from sympy.polys.rootoftools import rootof  # 根式工具
from sympy.simplify.simplify import signsimp  # 符号化简
from sympy.utilities.iterables import iterable  # 可迭代对象检查
from sympy.utilities.exceptions import SymPyDeprecationWarning  # SymPy弃用警告

# 导入sympy测试模块
from sympy.testing.pytest import raises, warns_deprecated_sympy, warns  # 测试异常抛出、警告

# 导入sympy中的全局符号变量
from sympy.abc import a, b, c, d, p, q, t, w, x, y, z  # 符号变量

# 定义_epsilon_eq函数，比较两个数组是否在给定精度内相等
def _epsilon_eq(a, b):
    for u, v in zip(a, b):
        if abs(u - v) > 1e-10:
            return False
    return True

# 定义_strict_eq函数，严格比较两个数组是否相等
def _strict_eq(a, b):
    # 检查变量a和b的类型是否相同
    if type(a) == type(b):
        # 如果变量a是可迭代的
        if iterable(a):
            # 检查a和b的长度是否相同
            if len(a) == len(b):
                # 使用_zip_函数逐对比较a和b中的元素是否严格相等，并返回比较结果
                return all(_strict_eq(c, d) for c, d in zip(a, b))
            else:
                # 如果a和b的长度不相同，则返回False
                return False
        else:
            # 如果a不是可迭代的，检查a是否是Poly类型，并且调用a对象的eq方法进行比较，要求严格比较(strict=True)
            return isinstance(a, Poly) and a.eq(b, strict=True)
    else:
        # 如果变量a和b的类型不相同，则返回False
        return False
def test_Poly_mixed_operations():
    # 创建一个多项式 p = x
    p = Poly(x, x)
    # 测试用例：p * exp(x)，检测是否会发出弃用警告
    with warns_deprecated_sympy():
        p * exp(x)
    # 测试用例：p + exp(x)，检测是否会发出弃用警告
    with warns_deprecated_sympy():
        p + exp(x)
    # 测试用例：p - exp(x)，检测是否会发出弃用警告
    with warns_deprecated_sympy():
        p - exp(x)


def test_Poly_from_dict():
    # 定义有限域 K = FF(3)
    K = FF(3)

    # 测试用例：根据字典创建多项式，测试 0:1, 1:2 的情况
    assert Poly.from_dict({0: 1, 1: 2}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_dict({0: 1, 1: 5}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)

    # 测试用例：根据字典创建多项式，测试 (0,):1, (1,):2 的情况
    assert Poly.from_dict({(0,): 1, (1,): 2}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_dict({(0,): 1, (1,): 5}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)

    # 测试用例：根据字典创建多项式，测试 {(0, 0): 1, (1, 1): 2} 的情况
    assert Poly.from_dict({(0, 0): 1, (1, 1): 2}, gens=(x, y), domain=K).rep == DMP([[K(2), K(0)], [K(1)]], K)

    # 测试用例：根据字典创建多项式，测试 0:1, 1:2 的情况，域为整数环 ZZ
    assert Poly.from_dict({0: 1, 1: 2}, gens=x).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    # 测试用例：根据字典创建多项式，测试 0:1, 1:2 的情况，域为有理数域 QQ
    assert Poly.from_dict({0: 1, 1: 2}, gens=x, field=True).rep == DMP([QQ(2), QQ(1)], QQ)

    # 测试用例：根据字典创建多项式，测试 0:1, 1:2 的情况，域为整数环 ZZ
    assert Poly.from_dict({0: 1, 1: 2}, gens=x, domain=ZZ).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    # 测试用例：根据字典创建多项式，测试 0:1, 1:2 的情况，域为有理数域 QQ
    assert Poly.from_dict({0: 1, 1: 2}, gens=x, domain=QQ).rep == DMP([QQ(2), QQ(1)], QQ)

    # 测试用例：根据字典创建多项式，测试 (0,):1, (1,):2 的情况，域为整数环 ZZ
    assert Poly.from_dict({(0,): 1, (1,): 2}, gens=x).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    # 测试用例：根据字典创建多项式，测试 (0,):1, (1,):2 的情况，域为有理数域 QQ
    assert Poly.from_dict({(0,): 1, (1,): 2}, gens=x, field=True).rep == DMP([QQ(2), QQ(1)], QQ)

    # 测试用例：根据字典创建多项式，测试 (0,):1, (1,):2 的情况，域为整数环 ZZ
    assert Poly.from_dict({(0,): 1, (1,): 2}, gens=x, domain=ZZ).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    # 测试用例：根据字典创建多项式，测试 (0,):1, (1,):2 的情况，域为有理数域 QQ
    assert Poly.from_dict({(0,): 1, (1,): 2}, gens=x, domain=QQ).rep == DMP([QQ(2), QQ(1)], QQ)

    # 测试用例：根据字典创建多项式，测试 (1,): sin(y) 的情况，复合标志为 False
    assert Poly.from_dict({(1,): sin(y)}, gens=x, composite=False) == Poly(sin(y)*x, x, domain='EX')
    # 测试用例：根据字典创建多项式，测试 (1,): y 的情况，复合标志为 False
    assert Poly.from_dict({(1,): y}, gens=x, composite=False) == Poly(y*x, x, domain='EX')
    # 测试用例：根据字典创建多项式，测试 (1, 1): 1 的情况，生成器为 (x, y)，复合标志为 False
    assert Poly.from_dict({(1, 1): 1}, gens=(x, y), composite=False) == Poly(x*y, x, y, domain='ZZ')
    # 测试用例：根据字典创建多项式，测试 (1, 0): y 的情况，生成器为 (x, z)，复合标志为 False
    assert Poly.from_dict({(1, 0): y}, gens=(x, z), composite=False) == Poly(y*x, x, z, domain='EX')


def test_Poly_from_list():
    # 定义有限域 K = FF(3)
    K = FF(3)

    # 测试用例：根据列表创建多项式，测试 [2, 1] 的情况，域为有限域 K
    assert Poly.from_list([2, 1], gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_list([5, 1], gens=x, domain=K).rep == DMP([K(2), K(1)], K)

    # 测试用例：根据列表创建多项式，测试 [2, 1] 的情况，域为整数环 ZZ
    assert Poly.from_list([2, 1], gens=x).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    # 测试用例：根据列表创建多项式，测试 [2, 1] 的情况，域为有理数域 QQ
    assert Poly.from_list([2, 1], gens=x, field=True).rep == DMP([QQ(2), QQ(1)], QQ)

    # 测试用例：根据列表创建多项式，测试 [2, 1] 的情况，域为整数环 ZZ
    assert Poly.from_list([2, 1], gens=x, domain=ZZ).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    # 测试用例：根据列表创建多项式，测试 [2, 1] 的情况，域为有理数域 QQ
    assert Poly.from_list([2, 1], gens=x, domain=QQ).rep == DMP([QQ(2), QQ(1)], QQ)

    # 测试用例：根据列表创建多项式，测试 [0, 1.0] 的情况，域为实数域 RR
    assert Poly.from_list([0, 1.0], gens=x).rep == DMP([RR(1.0)], RR)
    # 测试用例：根据列表创建多项式，测试 [1.0, 0] 的情况，域为实数域 RR
    assert Poly.from_list([1.0, 0], gens=x).rep == DMP([RR(1.0), RR(0.0)], RR)

    # 测试用例：尝试使用空列表作为输入，预期会抛出 MultivariatePolynomialError 异
    # 使用 Poly 类的 from_poly 方法创建多项式对象，并验证其表示形式与预期相符
    
    assert Poly.from_poly(f, domain=K).rep == DMP([K(1), K(1)], K)
    # 使用给定的域 K，将 f 转换为多项式对象，并验证其内部表示是否为 DMP([K(1), K(1)], K)
    
    assert Poly.from_poly(f, domain=ZZ).rep == DMP([ZZ(1), ZZ(7)], ZZ)
    # 使用整数环 ZZ，将 f 转换为多项式对象，并验证其内部表示是否为 DMP([ZZ(1), ZZ(7)], ZZ)
    
    assert Poly.from_poly(f, domain=QQ).rep == DMP([QQ(1), QQ(7)], QQ)
    # 使用有理数域 QQ，将 f 转换为多项式对象，并验证其内部表示是否为 DMP([QQ(1), QQ(7)], QQ)
    
    assert Poly.from_poly(f, gens=x) == f
    # 使用生成器 x，将 f 转换为多项式对象，并验证是否与原始多项式 f 相同
    
    assert Poly.from_poly(f, gens=x, domain=K).rep == DMP([K(1), K(1)], K)
    # 使用生成器 x 和域 K，将 f 转换为多项式对象，并验证其内部表示是否为 DMP([K(1), K(1)], K)
    
    assert Poly.from_poly(f, gens=x, domain=ZZ).rep == DMP([ZZ(1), ZZ(7)], ZZ)
    # 使用生成器 x 和整数环 ZZ，将 f 转换为多项式对象，并验证其内部表示是否为 DMP([ZZ(1), ZZ(7)], ZZ)
    
    assert Poly.from_poly(f, gens=x, domain=QQ).rep == DMP([QQ(1), QQ(7)], QQ)
    # 使用生成器 x 和有理数域 QQ，将 f 转换为多项式对象，并验证其内部表示是否为 DMP([QQ(1), QQ(7)], QQ)
    
    assert Poly.from_poly(f, gens=y) == Poly(x + 7, y, domain='ZZ[x]')
    # 使用生成器 y，将 f 转换为多项式对象，并验证其结果与 Poly(x + 7, y, domain='ZZ[x]') 相同
    # 注意：这里生成了一个新的多项式对象 Poly(x + 7, y, domain='ZZ[x]')
    
    raises(CoercionFailed, lambda: Poly.from_poly(f, gens=y, domain=K))
    # 使用生成器 y 和域 K，试图将 f 转换为多项式对象，但由于类型转换失败抛出 CoercionFailed 异常
    
    raises(CoercionFailed, lambda: Poly.from_poly(f, gens=y, domain=ZZ))
    # 使用生成器 y 和整数环 ZZ，试图将 f 转换为多项式对象，但由于类型转换失败抛出 CoercionFailed 异常
    
    raises(CoercionFailed, lambda: Poly.from_poly(f, gens=y, domain=QQ))
    # 使用生成器 y 和有理数域 QQ，试图将 f 转换为多项式对象，但由于类型转换失败抛出 CoercionFailed 异常
    
    assert Poly.from_poly(f, gens=(x, y)) == Poly(x + 7, x, y, domain='ZZ')
    # 使用生成器 (x, y)，将 f 转换为多项式对象，并验证其结果与 Poly(x + 7, x, y, domain='ZZ') 相同
    
    assert Poly.from_poly(f, gens=(x, y), domain=ZZ) == Poly(x + 7, x, y, domain='ZZ')
    # 使用生成器 (x, y) 和整数环 ZZ，将 f 转换为多项式对象，并验证其结果与 Poly(x + 7, x, y, domain='ZZ') 相同
    
    assert Poly.from_poly(f, gens=(x, y), domain=QQ) == Poly(x + 7, x, y, domain='QQ')
    # 使用生成器 (x, y) 和有理数域 QQ，将 f 转换为多项式对象，并验证其结果与 Poly(x + 7, x, y, domain='QQ') 相同
    
    assert Poly.from_poly(f, gens=(x, y), modulus=3) == Poly(x + 7, x, y, domain='FF(3)')
    # 使用生成器 (x, y) 和模数 3，将 f 转换为多项式对象，并验证其结果与 Poly(x + 7, x, y, domain='FF(3)') 相同
    
    K = FF(2)
    
    assert Poly.from_poly(g) == g
    # 使用默认域，将 g 转换为多项式对象，并验证其结果与 g 相同
    
    assert Poly.from_poly(g, domain=ZZ).rep == DMP([ZZ(1), ZZ(-1)], ZZ)
    # 使用整数环 ZZ，将 g 转换为多项式对象，并验证其内部表示是否为 DMP([ZZ(1), ZZ(-1)], ZZ)
    
    raises(CoercionFailed, lambda: Poly.from_poly(g, domain=QQ))
    # 使用有理数域 QQ，试图将 g 转换为多项式对象，但由于类型转换失败抛出 CoercionFailed 异常
    
    assert Poly.from_poly(g, domain=K).rep == DMP([K(1), K(0)], K)
    # 使用域 K，将 g 转换为多项式对象，并验证其内部表示是否为 DMP([K(1), K(0)], K)
    
    assert Poly.from_poly(g, gens=x) == g
    # 使用生成器 x，将 g 转换为多项式对象，并验证其结果与 g 相同
    
    assert Poly.from_poly(g, gens=x, domain=ZZ).rep == DMP([ZZ(1), ZZ(-1)], ZZ)
    # 使用生成器 x 和整数环 ZZ，将 g 转换为多项式对象，并验证其内部表示是否为 DMP([ZZ(1), ZZ(-1)], ZZ)
    
    raises(CoercionFailed, lambda: Poly.from_poly(g, gens=x, domain=QQ))
    # 使用生成器 x 和有理数域 QQ，试图将 g 转换为多项式对象，但由于类型转换失败抛出 CoercionFailed 异常
    
    assert Poly.from_poly(g, gens=x, domain=K).rep == DMP([K(1), K(0)], K)
    # 使用生成器 x 和域 K，将 g 转换为多项式对象，并验证其内部表示是否为 DMP([K(1), K(0)], K)
    
    K = FF(3)
    
    assert Poly.from_poly(h) == h
    # 使用默认域，将 h 转换为多项式对象，并验证其结果与 h 相同
    
    assert Poly.from_poly(h, domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    # 使用整数环 ZZ，将 h 转换为多项式对象，并验证其内部表示是否为 DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    
    assert Poly.from_poly(h, domain=QQ).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    # 使用有理数域 QQ，将 h 转换为多项式对象，并验证其内部表示是否为 DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    
    assert Poly.from_poly(h, domain=K).rep == DMP([[K(1)], [K(1), K(0)]], K)
    # 使用域 K，将 h 转换为多项式对象，并验证其内部表示是否为 DMP([[K(1)], [K(1), K(0)]], K)
    
    assert Poly.from_poly(h, gens=x) == Poly(x + y, x, domain=ZZ[y])
    # 使用生成器 x，将 h 转换为多项式对象，并验证其结果与 Poly(x + y, x, domain=ZZ[y]) 相同
    
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=x, domain=ZZ))
    # 使用生成器 x 和整数环 ZZ，试图将 h 转换为多项式对象，但由于类型转换失败抛出 CoercionFailed 异常
    
    assert Poly.from_poly(h, gens=x, domain=ZZ[y]) == Poly(x + y, x, domain=ZZ[y])
    # 使用生成器 x 和整数环 ZZ[y]，将 h 转换为多项式对象，并验证其结果与 Poly(x + y, x, domain=ZZ[y]) 相同
    
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=x, domain=QQ))
    # 使用生成器
    # 使用多项式类 Poly 的静态方法 from_poly 创建多项式对象，并进行断言测试
    
    # 断言1：验证使用默认环境 ZZ（整数环）创建的多项式对象与预期的多项式表示 DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ) 相等
    assert Poly.from_poly(h, gens=(x, y), domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    
    # 断言2：验证使用有理数环 QQ 创建的多项式对象与预期的多项式表示 DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ) 相等
    assert Poly.from_poly(h, gens=(x, y), domain=QQ).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    
    # 断言3：验证使用自定义环 K 创建的多项式对象与预期的多项式表示 DMP([[K(1)], [K(1), K(0)]], K) 相等
    assert Poly.from_poly(h, gens=(x, y), domain=K).rep == DMP([[K(1)], [K(1), K(0)]], K)
    
    # 断言4：验证使用不同顺序 (y, x) 创建的多项式对象与预期的多项式表示 DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ) 相等
    assert Poly.from_poly(h, gens=(y, x)).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    
    # 断言5：验证使用不同顺序 (y, x) 和整数环 ZZ 创建的多项式对象与预期的多项式表示 DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ) 相等
    assert Poly.from_poly(h, gens=(y, x), domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    
    # 断言6：验证使用不同顺序 (y, x) 和有理数环 QQ 创建的多项式对象与预期的多项式表示 DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ) 相等
    assert Poly.from_poly(h, gens=(y, x), domain=QQ).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    
    # 断言7：验证使用不同顺序 (y, x) 和自定义环 K 创建的多项式对象与预期的多项式表示 DMP([[K(1)], [K(1), K(0)]], K) 相等
    assert Poly.from_poly(h, gens=(y, x), domain=K).rep == DMP([[K(1)], [K(1), K(0)]], K)
    
    # 断言8：验证使用域字段属性创建的多项式对象与预期的多项式表示 DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ) 相等
    assert Poly.from_poly(h, gens=(x, y), field=True).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    
    # 断言9：验证再次使用域字段属性创建的多项式对象与预期的多项式表示 DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ) 相等
    assert Poly.from_poly(h, gens=(x, y), field=True).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
# 测试函数：检查从表达式创建多项式的各种情况

def test_Poly_from_expr():
    # 测试异常情况：尝试从零表达式创建多项式应引发 GeneratorsNeeded 异常
    raises(GeneratorsNeeded, lambda: Poly.from_expr(S.Zero))
    # 测试异常情况：尝试从整数表达式创建多项式应引发 GeneratorsNeeded 异常
    raises(GeneratorsNeeded, lambda: Poly.from_expr(S(7)))

    F3 = FF(3)

    # 检查从表达式 x + 5 创建多项式，使用域 F3
    assert Poly.from_expr(x + 5, domain=F3).rep == DMP([F3(1), F3(2)], F3)
    # 检查从表达式 y + 5 创建多项式，使用域 F3
    assert Poly.from_expr(y + 5, domain=F3).rep == DMP([F3(1), F3(2)], F3)

    # 检查从表达式 x + 5 创建多项式，指定生成器 x 和域 F3
    assert Poly.from_expr(x + 5, x, domain=F3).rep == DMP([F3(1), F3(2)], F3)
    # 检查从表达式 y + 5 创建多项式，指定生成器 y 和域 F3
    assert Poly.from_expr(y + 5, y, domain=F3).rep == DMP([F3(1), F3(2)], F3)

    # 检查从表达式 x + y 创建多项式，使用域 F3
    assert Poly.from_expr(x + y, domain=F3).rep == DMP([[F3(1)], [F3(1), F3(0)]], F3)
    # 检查从表达式 x + y 创建多项式，指定生成器 x 和 y，使用域 F3
    assert Poly.from_expr(x + y, x, y, domain=F3).rep == DMP([[F3(1)], [F3(1), F3(0)]], F3)

    # 检查从表达式 x + 5 创建多项式，使用默认整数域 ZZ
    assert Poly.from_expr(x + 5).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    # 检查从表达式 y + 5 创建多项式，使用默认整数域 ZZ
    assert Poly.from_expr(y + 5).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    # 检查从表达式 x + 5 创建多项式，指定生成器 x 和使用默认整数域 ZZ
    assert Poly.from_expr(x + 5, x).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    # 检查从表达式 y + 5 创建多项式，指定生成器 y 和使用默认整数域 ZZ
    assert Poly.from_expr(y + 5, y).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    # 检查从表达式 x + 5 创建多项式，指定生成器 x 和整数域 ZZ
    assert Poly.from_expr(x + 5, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    # 检查从表达式 y + 5 创建多项式，指定生成器 y 和整数域 ZZ
    assert Poly.from_expr(y + 5, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    # 检查从表达式 x + 5 创建多项式，指定生成器 x 和整数域 ZZ
    assert Poly.from_expr(x + 5, x, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    # 检查从表达式 y + 5 创建多项式，指定生成器 y 和整数域 ZZ
    assert Poly.from_expr(y + 5, y, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    # 检查从表达式 x + 5 创建多项式，指定生成器 x 和 y，整数域 ZZ
    assert Poly.from_expr(x + 5, x, y, domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(5)]], ZZ)
    # 检查从表达式 y + 5 创建多项式，指定生成器 x 和 y，整数域 ZZ
    assert Poly.from_expr(y + 5, x, y, domain=ZZ).rep == DMP([[ZZ(1), ZZ(5)]], ZZ)


def test_poly_from_domain_element():
    # 测试从域元素创建多项式的各种情况

    dom = ZZ[x]
    # 检查从域元素 dom(x+1) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)
    # 将域转换为其字段
    dom = dom.get_field()
    # 检查从域元素 dom(x+1) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)

    dom = QQ[x]
    # 检查从域元素 dom(x+1) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)
    # 将域转换为其字段
    dom = dom.get_field()
    # 检查从域元素 dom(x+1) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)

    dom = ZZ.old_poly_ring(x)
    # 检查从域元素 dom([ZZ(1), ZZ(1)]) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom([ZZ(1), ZZ(1)]), y, domain=dom).rep == DMP([dom([ZZ(1), ZZ(1)])], dom)
    # 将域转换为其字段
    dom = dom.get_field()
    # 检查从域元素 dom([ZZ(1), ZZ(1)]) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom([ZZ(1), ZZ(1)]), y, domain=dom).rep == DMP([dom([ZZ(1), ZZ(1)])], dom)

    dom = QQ.old_poly_ring(x)
    # 检查从域元素 dom([QQ(1), QQ(1)]) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom([QQ(1), QQ(1)]), y, domain=dom).rep == DMP([dom([QQ(1), QQ(1)])], dom)
    # 将域转换为其字段
    dom = dom.get_field()
    # 检查从域元素 dom([QQ(1), QQ(1)]) 创建多项式，使用生成器 y 和域 dom
    assert Poly(dom([QQ(1), QQ(1)]), y, domain=dom).rep == DMP([dom([QQ(1), QQ(1)])], dom)

    dom = QQ.algebraic_field(I)
    # 检查从域元素 dom([1, 1]) 创建多项式，使用生成器 x 和域 dom
    assert Poly(dom([1, 1]), x, domain=dom).rep == DMP([dom([1, 1])], dom)


def test_Poly__new__():
    # 测试多项式构造函数 Poly.__new__ 的各种选项错误情况

    # 测试异常情况：指定相同生成器多次应引发 GeneratorsError 异常
    raises(GeneratorsError, lambda: Poly(x + 1, x, x))

    # 测试异常情况：指定不同生成器应引发 GeneratorsError 异常
    raises(GeneratorsError, lambda: Poly(x + y, x, y, domain=ZZ[x]))
    raises(GeneratorsError, lambda: Poly(x + y, x, y, domain=ZZ[y]))

    # 测试异常情况：使用 symmetric=True 应引发 OptionError 异常
    raises(OptionError, lambda: Poly(x, x, symmetric=True))
    # 测试异常情况：同时指定 modulus 和 domain=QQ 应
    # 使用 lambda 函数测试在设置 modulus=3 和 extension=[sqrt(3)] 时 Poly 对象抛出 OptionError 异常
    raises(OptionError, lambda: Poly(x + 2, x, modulus=3, extension=[sqrt(3)]))

    # 使用 lambda 函数测试在设置 domain=ZZ 和 extension=True 时 Poly 对象抛出 OptionError 异常
    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, extension=True))
    # 使用 lambda 函数测试在设置 modulus=3 和 extension=True 时 Poly 对象抛出 OptionError 异常
    raises(OptionError, lambda: Poly(x + 2, x, modulus=3, extension=True))

    # 使用 lambda 函数测试在设置 domain=ZZ 和 greedy=True 时 Poly 对象抛出 OptionError 异常
    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, greedy=True))
    # 使用 lambda 函数测试在设置 domain=QQ 和 field=True 时 Poly 对象抛出 OptionError 异常
    raises(OptionError, lambda: Poly(x + 2, x, domain=QQ, field=True))

    # 使用 lambda 函数测试在设置 domain=ZZ 和 greedy=False 时 Poly 对象抛出 OptionError 异常
    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, greedy=False))
    # 使用 lambda 函数测试在设置 domain=QQ 和 field=False 时 Poly 对象抛出 OptionError 异常
    raises(OptionError, lambda: Poly(x + 2, x, domain=QQ, field=False))

    # 使用 lambda 函数测试在设置 modulus=3 和 order='grlex' 时 Poly 对象抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: Poly(x + 1, x, modulus=3, order='grlex'))
    # 使用 lambda 函数测试在设置 order='grlex' 时 Poly 对象抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: Poly(x + 1, x, order='grlex'))

    # 使用 lambda 函数测试在传递字典作为参数时 Poly 对象抛出 GeneratorsNeeded 异常
    raises(GeneratorsNeeded, lambda: Poly({1: 2, 0: 1}))
    # 使用 lambda 函数测试在传递列表作为参数时 Poly 对象抛出 GeneratorsNeeded 异常
    raises(GeneratorsNeeded, lambda: Poly([2, 1]))
    # 使用 lambda 函数测试在传递元组作为参数时 Poly 对象抛出 GeneratorsNeeded 异常
    raises(GeneratorsNeeded, lambda: Poly((2, 1)))

    # 使用 lambda 函数测试在不传递生成器时 Poly 对象抛出 GeneratorsNeeded 异常
    raises(GeneratorsNeeded, lambda: Poly(1))

    # 使用断言验证 Poly('x-x') 等于 Poly(0, x)
    assert Poly('x-x') == Poly(0, x)

    # 创建多项式 f = a*x**2 + b*x + c
    f = a*x**2 + b*x + c

    # 使用断言验证不同方式初始化的 Poly 对象等于 f
    assert Poly({2: a, 1: b, 0: c}, x) == f
    assert Poly(iter([a, b, c]), x) == f
    assert Poly([a, b, c], x) == f
    assert Poly((a, b, c), x) == f

    # 创建一个空的多项式对象 f
    f = Poly({}, x, y, z)

    # 使用断言验证 f 的生成器为 (x, y, z)，并且其表达式表示为 0
    assert f.gens == (x, y, z) and f.as_expr() == 0

    # 使用断言验证在传递 Poly 对象作为参数时，Poly 对象等于原始表达式的 Poly 对象
    assert Poly(Poly(a*x + b*y, x, y), x) == Poly(a*x + b*y, x)

    # 使用断言验证在指定不同域下创建的多项式的系数列表
    assert Poly(3*x**2 + 2*x + 1, domain='ZZ').all_coeffs() == [3, 2, 1]
    assert Poly(3*x**2 + 2*x + 1, domain='QQ').all_coeffs() == [3, 2, 1]
    assert Poly(3*x**2 + 2*x + 1, domain='RR').all_coeffs() == [3.0, 2.0, 1.0]

    # 使用 lambda 函数测试在传递非整数系数时 Poly 对象抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: Poly(3*x**2/5 + x*Rational(2, 5) + 1, domain='ZZ'))
    # 使用断言验证在有理数域下创建的多项式的系数列表
    assert Poly(3*x**2/5 + x*Rational(2, 5) + 1, domain='QQ').all_coeffs() == [Rational(3, 5), Rational(2, 5), 1]
    # 使用自定义函数 _epsilon_eq 验证在实数域下创建的多项式的系数列表
    assert _epsilon_eq(
        Poly(3*x**2/5 + x*Rational(2, 5) + 1, domain='RR').all_coeffs(), [0.6, 0.4, 1.0])

    # 使用断言验证在传递浮点数系数时创建的多项式的系数列表
    assert Poly(3.0*x**2 + 2.0*x + 1, domain='ZZ').all_coeffs() == [3, 2, 1]
    assert Poly(3.0*x**2 + 2.0*x + 1, domain='QQ').all_coeffs() == [3, 2, 1]
    assert Poly(3.0*x**2 + 2.0*x + 1, domain='RR').all_coeffs() == [3.0, 2.0, 1.0]

    # 使用 lambda 函数测试在传递非整数浮点数系数时 Poly 对象抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: Poly(3.1*x**2 + 2.1*x + 1, domain='ZZ'))
    # 使用断言验证在有理数域下创建的多项式的系数列表
    assert Poly(3.1*x**2 + 2.1*x + 1, domain='QQ').all_coeffs() == [Rational(31, 10), Rational(21, 10), 1]
    # 使用断言验证在实数域下创建的多项式的系数列表
    assert Poly(3.1*x**2 + 2.1*x + 1, domain='RR').all_coeffs() == [3.1, 2.1, 1.0]

    # 使用断言验证在指定多变量情况下创建的多项式对象等于对应的多项式表达式
    assert Poly({(2, 1): 1, (1, 2): 2, (1, 1): 3}, x, y) == \
        Poly(x**2*y + 2*x*y**2 + 3*x*y, x, y)

    # 使用断言验证在指定扩展参数时，Poly 对象的域为 QQ 的代数扩张域
    assert Poly(x**2 + 1, extension=I).get_domain() == QQ.algebraic_field(I)

    # 创建多项式 f = 3*x**5 - x**4 + x**3 - x**2 + 65538
    f = 3*x**5 - x**4 + x**3 - x** 2 + 65538

    # 使用断言验证在设置 modulus=65537 和 symmetric=True 时创建的 Poly 对象等于设定系数后的多项式
    assert Poly(f, x, modulus=65537, symmetric=True) == \
        Poly(3*x**5 - x**4 + x**3 - x** 2 + 1, x, modulus=65537,
             symmetric=True)
    # 使用断言验证在设置 modulus=65537 和 symmetric=False 时创建的 Poly 对象等于设定系数后的多项式
    assert Poly(f, x, modulus=65537, symmetric=False) == \
        Poly(3*x**5 + 65536*x**4 + x**3 + 65536*x** 2 + 1, x,
             modulus=65537, symmetric=False)

    # 使用断言验证创建的 Poly 对象的域
    # 断言语句，用于检查表达式返回的结果是否为 ComplexField 类型
    assert isinstance(Poly(x**2 + x + I + 1.0).get_domain(), ComplexField)
def test_Poly__args():
    assert Poly(x**2 + 1).args == (x**2 + 1, x)
    # 确保 Poly 类中 args 属性返回元组，包含多项式表达式和变量 x


def test_Poly__gens():
    assert Poly((x - p)*(x - q), x).gens == (x,)
    # 当指定 x 作为变量时，gens 属性应返回元组 (x,)，表明多项式中的生成器变量
    assert Poly((x - p)*(x - q), p).gens == (p,)
    # 当指定 p 作为变量时，gens 属性应返回元组 (p,)
    assert Poly((x - p)*(x - q), q).gens == (q,)
    # 当指定 q 作为变量时，gens 属性应返回元组 (q,)

    assert Poly((x - p)*(x - q), x, p).gens == (x, p)
    # 指定 x 和 p 作为变量时，gens 属性应返回元组 (x, p)
    assert Poly((x - p)*(x - q), x, q).gens == (x, q)
    # 指定 x 和 q 作为变量时，gens 属性应返回元组 (x, q)

    assert Poly((x - p)*(x - q), x, p, q).gens == (x, p, q)
    # 指定 x, p 和 q 作为变量时，gens 属性应返回元组 (x, p, q)
    assert Poly((x - p)*(x - q), p, x, q).gens == (p, x, q)
    # 指定 p, x 和 q 作为变量时，gens 属性应返回元组 (p, x, q)
    assert Poly((x - p)*(x - q), p, q, x).gens == (p, q, x)
    # 指定 p, q 和 x 作为变量时，gens 属性应返回元组 (p, q, x)

    assert Poly((x - p)*(x - q)).gens == (x, p, q)
    # 未指定变量顺序时，默认按照 x, p, q 的顺序返回 gens 属性

    assert Poly((x - p)*(x - q), sort='x > p > q').gens == (x, p, q)
    # 按照 'x > p > q' 的顺序返回 gens 属性
    assert Poly((x - p)*(x - q), sort='p > x > q').gens == (p, x, q)
    # 按照 'p > x > q' 的顺序返回 gens 属性
    assert Poly((x - p)*(x - q), sort='p > q > x').gens == (p, q, x)
    # 按照 'p > q > x' 的顺序返回 gens 属性

    assert Poly((x - p)*(x - q), x, p, q, sort='p > q > x').gens == (x, p, q)
    # 指定变量顺序为 'p > q > x'，gens 属性应按照 (x, p, q) 的顺序返回

    assert Poly((x - p)*(x - q), wrt='x').gens == (x, p, q)
    # 指定 wrt='x' 时，gens 属性应按照 (x, p, q) 的顺序返回
    assert Poly((x - p)*(x - q), wrt='p').gens == (p, x, q)
    # 指定 wrt='p' 时，gens 属性应按照 (p, x, q) 的顺序返回
    assert Poly((x - p)*(x - q), wrt='q').gens == (q, x, p)
    # 指定 wrt='q' 时，gens 属性应按照 (q, x, p) 的顺序返回

    assert Poly((x - p)*(x - q), wrt=x).gens == (x, p, q)
    # 指定 wrt=x 时，gens 属性应按照 (x, p, q) 的顺序返回
    assert Poly((x - p)*(x - q), wrt=p).gens == (p, x, q)
    # 指定 wrt=p 时，gens 属性应按照 (p, x, q) 的顺序返回
    assert Poly((x - p)*(x - q), wrt=q).gens == (q, x, p)
    # 指定 wrt=q 时，gens 属性应按照 (q, x, p) 的顺序返回

    assert Poly((x - p)*(x - q), x, p, q, wrt='p').gens == (x, p, q)
    # 指定 x, p, q 和 wrt='p' 时，gens 属性应按照 (x, p, q) 的顺序返回

    assert Poly((x - p)*(x - q), wrt='p', sort='q > x').gens == (p, q, x)
    # 指定 wrt='p' 和 sort='q > x' 时，gens 属性应按照 (p, q, x) 的顺序返回
    assert Poly((x - p)*(x - q), wrt='q', sort='p > x').gens == (q, p, x)
    # 指定 wrt='q' 和 sort='p > x' 时，gens 属性应按照 (q, p, x) 的顺序返回


def test_Poly_zero():
    assert Poly(x).zero == Poly(0, x, domain=ZZ)
    # 确保 Poly 类中 zero 属性返回多项式 0，指定变量为 x，定义域为整数环 ZZ
    assert Poly(x/2).zero == Poly(0, x, domain=QQ)
    # 确保 Poly 类中 zero 属性返回多项式 0，指定变量为 x，定义域为有理数域 QQ


def test_Poly_one():
    assert Poly(x).one == Poly(1, x, domain=ZZ)
    # 确保 Poly 类中 one 属性返回多项式 1，指定变量为 x，定义域为整数环 ZZ
    assert Poly(x/2).one == Poly(1, x, domain=QQ)
    # 确保 Poly 类中 one 属性返回多项式 1，指定变量为 x，定义域为有理数域 QQ


def test_Poly__unify():
    raises(UnificationFailed, lambda: Poly(x)._unify(y))
    # 确保当尝试对不同变量进行统一时会引发 UnificationFailed 异常

    F3 = FF(3)

    assert Poly(x, x, modulus=3)._unify(Poly(y, y, modulus=3))[2:] == (
        DMP([[F3(1)], []], F3), DMP([[F3(1), F3(0)]], F3))
    # 确保在模数为 3 的情况下，Poly 类中 _unify 方法能够成功统一多项式 x 和 y
    raises(UnificationFailed, lambda: Poly(x, x, modulus=3)._unify(Poly(y, y, modulus=5)))
    # 确保当模数不同导致统一失败时会引发 UnificationFailed 异常

    raises(UnificationFailed, lambda: Poly(y, x, y)._unify(Poly(x, x, modulus=3)))
    # 确保当多项式变量顺序不同导致统一失败时会引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: Poly(x, x, modulus=3)._unify(Poly(y, x, y)))
    # 确保当多项式变量顺序不同导致统一失败时会引发 UnificationFailed 异常

    assert Poly(x + 1, x)._unify(Poly(x + 2, x))[2:] ==\
        (DMP([ZZ(1), ZZ(1)], ZZ), DMP([ZZ(1), ZZ(2)], ZZ))
    # 确保在指定变量 x 下，Poly 类中 _unify 方法能够成功统一 x+1 和 x+2 的多项式表示
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, x))[2:] ==\
        (DMP([QQ(1), QQ(1)], QQ), DMP([QQ(1), QQ(
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[ZZ(1)], [ZZ(1)]], ZZ), DMP([[ZZ(1)], [ZZ(2)]], ZZ))
    # 断言：验证两个多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))
    # 断言：验证带有有理数域的多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))

    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[ZZ(1)], [ZZ(1)]], ZZ), DMP([[ZZ(1)], [ZZ(2)]], ZZ))
    # 断言：验证两个多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))
    # 断言：验证带有有理数域的多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))

    assert Poly(x + 1, x)._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[ZZ(1), ZZ(1)]], ZZ), DMP([[ZZ(1), ZZ(2)]], ZZ))
    # 断言：验证两个多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))
    # 断言：验证带有有理数域的多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x)._unify(Poly(x + 2, y, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))

    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[ZZ(1), ZZ(1)]], ZZ), DMP([[ZZ(1), ZZ(2)]], ZZ))
    # 断言：验证两个多项式对象的合一结果是否符合预期
    assert Poly(x + 1, y, x, domain='QQ')._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))
    # 断言：验证带有有理数域的多项式对象的合一结果是否符合预期
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))

    assert Poly(x + 1, x, y)._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[ZZ(1)], [ZZ(1)]], ZZ), DMP([[ZZ(1)], [ZZ(2)]], ZZ))
    # 断言：验证两个多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))
    # 断言：验证带有有理数域的多项式对象的合一结果是否符合预期
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, y, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))

    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[ZZ(1), ZZ(1)]], ZZ), DMP([[ZZ(1), ZZ(2)]], ZZ))
    # 断言：验证两个多项式对象的合一结果是否符合预期
    assert Poly(x + 1, y, x, domain='QQ')._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))
    # 断言：验证带有有理数域的多项式对象的合一结果是否符合预期
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))

    assert Poly(x**2 + I, x, domain=ZZ_I).unify(Poly(x**2 + sqrt(2), x, extension=True)) == \
            (Poly(x**2 + I, x, domain='QQ<sqrt(2) + I>'), Poly(x**2 + sqrt(2), x, domain='QQ<sqrt(2) + I>'))
    # 断言：验证具有特定域的多项式对象的合一结果是否符合预期

    F, A, B = field("a,b", ZZ)

    assert Poly(a*x, x, domain='ZZ[a]')._unify(Poly(a*b*x, x, domain='ZZ(a,b)'))[2:] == \
        (DMP([A, F(0)], F.to_domain()), DMP([A*B, F(0)], F.to_domain()))
    # 断言：验证两个具有不同域的多项式对象的合一结果是否符合预期
    # 断言：使用 Poly 类创建两个多项式对象，分别为 a*x 和 a*b*x，确保它们在特定域中统一后的结果等于 (DMP([A, F(0)], F.to_domain()), DMP([A*B, F(0)], F.to_domain())) 的后两个元素。
    assert Poly(a*x, x, domain='ZZ(a)')._unify(Poly(a*b*x, x, domain='ZZ(a,b)'))[2:] == \
        (DMP([A, F(0)], F.to_domain()), DMP([A*B, F(0)], F.to_domain()))

    # 断言：使用 Poly 类尝试创建一个多项式对象，传递另一个多项式作为参数，此处应该引发 CoercionFailed 异常，因为多项式的定义域 'ZZ(x)' 不兼容于所传递的多项式对象。
    raises(CoercionFailed, lambda: Poly(Poly(x**2 + x**2*z, y, field=True), domain='ZZ(x)'))

    # 创建 Poly 类的实例 f 和 g，分别代表定义域为 QQ(x) 和 QQ[x] 的多项式 t**2 + t/3 + x。
    f = Poly(t**2 + t/3 + x, t, domain='QQ(x)')
    g = Poly(t**2 + t/3 + x, t, domain='QQ[x]')

    # 断言：对多项式 f 和 g 进行统一操作，确保它们的返回结果的后两个元素分别是 f.rep 和 g.rep。
    assert f._unify(g)[2:] == (f.rep, f.rep)
# 定义测试函数 test_Poly_free_symbols，用于测试 Poly 类的 free_symbols 方法
def test_Poly_free_symbols():
    # 断言：计算 x**2 + 1 的自由符号集合应为 {x}
    assert Poly(x**2 + 1).free_symbols == {x}
    # 断言：计算 x**2 + y*z 的自由符号集合应为 {x, y, z}
    assert Poly(x**2 + y*z).free_symbols == {x, y, z}
    # 断言：指定变量 x，计算 x**2 + y*z 的自由符号集合应为 {x, y, z}
    assert Poly(x**2 + y*z, x).free_symbols == {x, y, z}
    # 断言：计算 x**2 + sin(y*z) 的自由符号集合应为 {x, y, z}
    assert Poly(x**2 + sin(y*z)).free_symbols == {x, y, z}
    # 断言：指定变量 x，计算 x**2 + sin(y*z) 的自由符号集合应为 {x, y, z}
    assert Poly(x**2 + sin(y*z), x).free_symbols == {x, y, z}
    # 断言：指定域为 EX，计算 x**2 + sin(y*z) 的自由符号集合应为 {x, y, z}
    assert Poly(x**2 + sin(y*z), x, domain=EX).free_symbols == {x, y, z}
    # 断言：指定多个变量 x, y, z，计算 1 + x + x**2 的自由符号集合应为 {x}
    assert Poly(1 + x + x**2, x, y, z).free_symbols == {x}
    # 断言：指定变量 z，计算 x + sin(y) 的自由符号集合应为 {x, y}
    assert Poly(x + sin(y), z).free_symbols == {x, y}


# 定义测试函数 test_PurePoly_free_symbols，用于测试 PurePoly 类的 free_symbols 方法
def test_PurePoly_free_symbols():
    # 断言：计算 x**2 + 1 的自由符号集合应为空集
    assert PurePoly(x**2 + 1).free_symbols == set()
    # 断言：计算 x**2 + y*z 的自由符号集合应为空集
    assert PurePoly(x**2 + y*z).free_symbols == set()
    # 断言：指定变量 x，计算 x**2 + y*z 的自由符号集合应为 {y, z}
    assert PurePoly(x**2 + y*z, x).free_symbols == {y, z}
    # 断言：计算 x**2 + sin(y*z) 的自由符号集合应为空集
    assert PurePoly(x**2 + sin(y*z)).free_symbols == set()
    # 断言：指定变量 x，计算 x**2 + sin(y*z) 的自由符号集合应为 {y, z}
    assert PurePoly(x**2 + sin(y*z), x).free_symbols == {y, z}
    # 断言：指定域为 EX，计算 x**2 + sin(y*z) 的自由符号集合应为 {y, z}
    assert PurePoly(x**2 + sin(y*z), x, domain=EX).free_symbols == {y, z}


# 定义测试函数 test_Poly__eq__，用于测试 Poly 类的 __eq__ 方法
def test_Poly__eq__():
    # 断言：比较两个相同的 Poly 对象，结果应为 True
    assert (Poly(x, x) == Poly(x, x)) is True
    # 断言：比较带有不同域的 Poly 对象，结果应为 False
    assert (Poly(x, x, domain=QQ) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, domain=QQ)) is False
    # 断言：比较带有不同域的 Poly 对象，结果应为 False
    assert (Poly(x, x, domain=ZZ[a]) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, domain=ZZ[a])) is False
    # 断言：比较不同形式的 Poly 对象，结果应为 False
    assert (Poly(x*y, x, y) == Poly(x, x)) is False
    assert (Poly(x, x, y) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, y)) is False
    # 断言：比较不同形式的 Poly 对象，结果应为 False
    assert (Poly(x**2 + 1, x) == Poly(y**2 + 1, y)) is False
    assert (Poly(y**2 + 1, y) == Poly(x**2 + 1, x)) is False
    # 创建两个 Poly 对象进行比较，并断言结果为 False
    f = Poly(x, x, domain=ZZ)
    g = Poly(x, x, domain=QQ)
    assert f.eq(g) is False
    assert f.ne(g) is True
    assert f.eq(g, strict=True) is False
    assert f.ne(g, strict=True) is True
    # 创建两个不同域的 Poly 对象进行比较，并断言结果为 False
    t0 = Symbol('t0')
    f =  Poly((t0/2 + x**2)*t**2 - x**2*t, t, domain='QQ[x,t0]')
    g =  Poly((t0/2 + x**2)*t**2 - x**2*t, t, domain='ZZ(x,t0)')
    assert (f == g) is False


# 定义测试函数 test_PurePoly__eq__，用于测试 PurePoly 类的 __eq__ 方法
def test_PurePoly__eq__():
    # 断言：比较两个相同的 PurePoly 对象，结果应为 True
    assert (PurePoly(x, x) == PurePoly(x, x)) is True
    assert (PurePoly(x, x, domain=QQ) == PurePoly(x, x)) is True
    assert (PurePoly(x, x) == PurePoly(x, x, domain=QQ)) is True
    # 断言：比较带有不同域的 PurePoly 对象，结果应为 True
    assert (PurePoly(x, x, domain=ZZ[a]) == PurePoly(x, x)) is True
    assert (PurePoly(x, x) == PurePoly(x, x, domain=ZZ[a])) is True
    # 断言：比较不同形式的 PurePoly 对象，结果应为 False
    assert (PurePoly(x*y, x, y) == PurePoly(x, x)) is False
    assert (PurePoly(x, x, y) == PurePoly(x, x)) is False
    assert (PurePoly(x, x) == PurePoly(x, x, y)) is False
    # 断言：比较不同形式的 PurePoly 对象，结果应为 True
    assert (PurePoly(x**2 + 1, x) == PurePoly(y**2 + 1, y)) is True
    assert (PurePoly(y**2 + 1, y) == PurePoly(x**2 + 1, x)) is True
    # 创建两个 PurePoly 对象进行比较，并断言结果为 True
    f = PurePoly(x, x, domain=ZZ)
    g = PurePoly(x, x, domain=QQ)
    assert f.eq(g) is True
    assert f.ne(g) is False
    assert f.eq(g, strict=True) is False
    assert f.ne(g, strict=True) is True
    # 创建两个不同符号的 PurePoly 对象进行比较，并断言结果为 True
    f = PurePoly(x, x, domain=ZZ)
    g = PurePoly(y, y, domain=QQ)
    assert f.eq(g) is True
    assert f.ne(g) is False
    assert f.eq(g, strict=True) is False
    assert f.ne(g, strict=True) is True
    # 断言：确保将 x^2 + 1 封装成 PurePoly 对象后，返回的对象类型是 PurePoly
    assert isinstance(PurePoly(Poly(x**2 + 1)), PurePoly) is True
    # 断言：确保将 x^2 + 1 封装成 Poly 对象后，返回的对象类型是 Poly
    assert isinstance(Poly(PurePoly(x**2 + 1)), Poly) is True
# 定义测试函数 test_Poly_get_domain，测试 Poly 类的 get_domain 方法
def test_Poly_get_domain():
    # 断言 Poly(2*x) 的域为整数环 ZZ
    assert Poly(2*x).get_domain() == ZZ

    # 断言 Poly(2*x, domain='ZZ') 的域为整数环 ZZ
    assert Poly(2*x, domain='ZZ').get_domain() == ZZ
    # 断言 Poly(2*x, domain='QQ') 的域为有理数域 QQ
    assert Poly(2*x, domain='QQ').get_domain() == QQ

    # 断言 Poly(x/2) 的域为有理数域 QQ
    assert Poly(x/2).get_domain() == QQ

    # 使用 lambda 函数断言 Poly(x/2, domain='ZZ') 抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: Poly(x/2, domain='ZZ'))
    # 断言 Poly(x/2, domain='QQ') 的域为有理数域 QQ
    assert Poly(x/2, domain='QQ').get_domain() == QQ

    # 断言 Poly(0.2*x).get_domain() 返回的类型是实数域 RealField 的实例
    assert isinstance(Poly(0.2*x).get_domain(), RealField)


# 定义测试函数 test_Poly_set_domain，测试 Poly 类的 set_domain 方法
def test_Poly_set_domain():
    # 断言 Poly(2*x + 1).set_domain(ZZ) 返回与输入相同的多项式对象
    assert Poly(2*x + 1).set_domain(ZZ) == Poly(2*x + 1)
    # 断言 Poly(2*x + 1).set_domain('ZZ') 返回与输入相同的多项式对象
    assert Poly(2*x + 1).set_domain('ZZ') == Poly(2*x + 1)

    # 断言 Poly(2*x + 1).set_domain(QQ) 返回带有 'QQ' 域的多项式对象
    assert Poly(2*x + 1).set_domain(QQ) == Poly(2*x + 1, domain='QQ')
    # 断言 Poly(2*x + 1).set_domain('QQ') 返回带有 'QQ' 域的多项式对象
    assert Poly(2*x + 1).set_domain('QQ') == Poly(2*x + 1, domain='QQ')

    # 断言 Poly(Rational(2, 10)*x + Rational(1, 10)).set_domain('RR') 返回与 0.2*x + 0.1 相同的多项式对象
    assert Poly(Rational(2, 10)*x + Rational(1, 10)).set_domain('RR') == Poly(0.2*x + 0.1)
    # 断言 Poly(0.2*x + 0.1).set_domain('QQ') 返回与 Rational(2, 10)*x + Rational(1, 10) 相同的多项式对象
    assert Poly(0.2*x + 0.1).set_domain('QQ') == Poly(Rational(2, 10)*x + Rational(1, 10))

    # 使用 lambda 函数断言 Poly(x/2 + 1).set_domain(ZZ) 抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: Poly(x/2 + 1).set_domain(ZZ))
    # 使用 lambda 函数断言 Poly(x + 1, modulus=2).set_domain(QQ) 抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: Poly(x + 1, modulus=2).set_domain(QQ))

    # 使用 lambda 函数断言 Poly(x*y, x, y).set_domain(ZZ[y]) 抛出 GeneratorsError 异常
    raises(GeneratorsError, lambda: Poly(x*y, x, y).set_domain(ZZ[y]))


# 定义测试函数 test_Poly_get_modulus，测试 Poly 类的 get_modulus 方法
def test_Poly_get_modulus():
    # 断言 Poly(x**2 + 1, modulus=2).get_modulus() 返回 2
    assert Poly(x**2 + 1, modulus=2).get_modulus() == 2
    # 使用 lambda 函数断言 Poly(x**2 + 1).get_modulus() 抛出 PolynomialError 异常
    raises(PolynomialError, lambda: Poly(x**2 + 1).get_modulus())


# 定义测试函数 test_Poly_set_modulus，测试 Poly 类的 set_modulus 方法
def test_Poly_set_modulus():
    # 断言 Poly(x**2 + 1, modulus=2).set_modulus(7) 返回与输入相同的多项式对象，但 modulus 为 7
    assert Poly(x**2 + 1, modulus=2).set_modulus(7) == Poly(x**2 + 1, modulus=7)
    # 断言 Poly(x**2 + 5, modulus=7).set_modulus(2) 返回与 x**2 + 1, modulus=2 相同的多项式对象
    assert Poly(x**2 + 5, modulus=7).set_modulus(2) == Poly(x**2 + 1, modulus=2)

    # 断言 Poly(x**2 + 1).set_modulus(2) 返回与输入相同的多项式对象，但 modulus 为 2
    assert Poly(x**2 + 1).set_modulus(2) == Poly(x**2 + 1, modulus=2)

    # 使用 lambda 函数断言 Poly(x/2 + 1).set_modulus(2) 抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: Poly(x/2 + 1).set_modulus(2))


# 定义测试函数 test_Poly_add_ground，测试 Poly 类的 add_ground 方法
def test_Poly_add_ground():
    # 断言 Poly(x + 1).add_ground(2) 返回 x + 3 的多项式对象
    assert Poly(x + 1).add_ground(2) == Poly(x + 3)


# 定义测试函数 test_Poly_sub_ground，测试 Poly 类的 sub_ground 方法
def test_Poly_sub_ground():
    # 断言 Poly(x + 1).sub_ground(2) 返回 x - 1 的多项式对象
    assert Poly(x + 1).sub_ground(2) == Poly(x - 1)


# 定义测试函数 test_Poly_mul_ground，测试 Poly 类的 mul_ground 方法
def test_Poly_mul_ground():
    # 断言 Poly(x + 1).mul_ground(2) 返回 2*x + 2 的多项式对象
    assert Poly(x + 1).mul_ground(2) == Poly(2*x + 2)


# 定义测试函数 test_Poly_quo_ground，测试 Poly 类的 quo_ground 方法
def test_Poly_quo_ground():
    # 断言 Poly(2*x + 4).quo_ground(2) 返回 x + 2 的多项式对象
    assert Poly(2*x + 4).quo_ground(2) == Poly(x + 2)
    # 断言 Poly(2*x + 3).quo_ground(2) 返回 x + 1 的多项式对象
    assert Poly(2*x + 3).quo_ground(2) == Poly(x + 1)


# 定义测试函数 test_Poly_exquo_ground，测试 Poly 类的 exquo_ground 方法
def test_Poly_exquo_ground():
    # 断言 Poly(2*x + 4).exquo_ground(2) 返回 x + 2 的多项式对象
    assert Poly(2*x + 4).exquo_ground(2) == Poly(x + 2)
    # 使用 lambda 函数断言 Poly(2*x + 3).exquo_ground(2) 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: Poly(2*x + 3).exquo_ground(2))


# 定义测试函数 test_Poly_abs，测试 Poly 类的 abs 方法
def test_Poly_abs():
    # 断言 Poly(-x + 1, x).abs() 返回与 abs(Poly(-x + 1, x)) 相同的多项式对象，即 Poly(x + 1, x)
    assert Poly(-x + 1, x).abs() == abs(Poly(-x + 1, x)) == Poly(x + 1, x)


# 定义测试函数 test_Poly_neg，测试 Poly 类的 neg 方法
def test_Poly_neg():
    # 断言 Poly(-x + 1, x).neg() 返回与 -Poly(-x + 1, x) 相同的多项式对象，即 Poly(x - 1, x)
    assert Poly(-x + 1, x).neg() == -Poly(-x + 1, x) == Poly(x - 1
    # 断言：创建一个一元多项式对象 Poly(1, x)，并从中减去 Poly(0, x)，预期结果是 Poly(1, x)
    assert Poly(1, x).sub(Poly(0, x)) == Poly(1, x)
    
    # 断言：创建一个二元多项式对象 Poly(1, x, y)，从中减去 Poly(0, x)，预期结果是 Poly(1, x, y)
    assert Poly(1, x, y) - Poly(0, x) == Poly(1, x, y)
    
    # 断言：创建一个二元多项式对象 Poly(0, x)，并从中减去 Poly(1, x, y)，预期结果是 Poly(-1, x, y)
    assert Poly(0, x).sub(Poly(1, x, y)) == Poly(-1, x, y)
    
    # 断言：创建一个二元多项式对象 Poly(0, x, y)，从中减去 Poly(1, x, y)，预期结果是 Poly(-1, x, y)
    assert Poly(0, x, y) - Poly(1, x, y) == Poly(-1, x, y)
    
    # 断言：创建一个一元多项式对象 Poly(1, x)，并从中减去 x，预期结果是 Poly(1 - x, x)
    assert Poly(1, x) - x == Poly(1 - x, x)
    
    # 使用警告：Poly(1, x) 减去 sin(x) 的操作已被 sympy 废弃，此处可能会产生警告
    with warns_deprecated_sympy():
        Poly(1, x) - sin(x)
    
    # 断言：创建一个二元多项式对象 Poly(x, x)，从中减去 1，预期结果是 Poly(x - 1, x)
    assert Poly(x, x) - 1 == Poly(x - 1, x)
    
    # 断言：创建一个一元多项式对象 1，从中减去 Poly(x, x)，预期结果是 Poly(1 - x, x)
    assert 1 - Poly(x, x) == Poly(1 - x, x)
# 定义一个测试函数，用于测试多项式类的乘法功能
def test_Poly_mul():
    # 断言：零多项式乘以任何多项式都等于零多项式
    assert Poly(0, x).mul(Poly(0, x)) == Poly(0, x)
    # 断言：使用运算符*进行的乘法也应该等同于零多项式
    assert Poly(0, x) * Poly(0, x) == Poly(0, x)

    # 断言：常数多项式相乘
    assert Poly(2, x).mul(Poly(4, x)) == Poly(8, x)
    # 断言：不同变量数目的多项式相乘
    assert Poly(2, x, y) * Poly(4, x) == Poly(8, x, y)
    assert Poly(4, x).mul(Poly(2, x, y)) == Poly(8, x, y)
    # 断言：相同变量数目的多项式相乘
    assert Poly(4, x, y) * Poly(2, x, y) == Poly(8, x, y)

    # 断言：多项式与单个变量相乘
    assert Poly(1, x) * x == Poly(x, x)
    # 使用警告函数检查不推荐使用的函数操作
    with warns_deprecated_sympy():
        Poly(1, x) * sin(x)

    # 断言：多项式乘以常数
    assert Poly(x, x) * 2 == Poly(2*x, x)
    assert 2 * Poly(x, x) == Poly(2*x, x)

# 测试一个特定问题的函数
def test_issue_13079():
    # 断言：多项式与变量的乘法应生成一个新的多项式
    assert Poly(x)*x == Poly(x**2, x, domain='ZZ')
    assert x*Poly(x) == Poly(x**2, x, domain='ZZ')
    assert -2*Poly(x) == Poly(-2*x, x, domain='ZZ')
    assert S(-2)*Poly(x) == Poly(-2*x, x, domain='ZZ')
    assert Poly(x)*S(-2) == Poly(-2*x, x, domain='ZZ')

# 测试多项式类的平方操作
def test_Poly_sqr():
    assert Poly(x*y, x, y).sqr() == Poly(x**2*y**2, x, y)

# 测试多项式类的乘方操作
def test_Poly_pow():
    assert Poly(x, x).pow(10) == Poly(x**10, x)
    assert Poly(x, x).pow(Integer(10)) == Poly(x**10, x)

    assert Poly(2*y, x, y).pow(4) == Poly(16*y**4, x, y)
    assert Poly(2*y, x, y).pow(Integer(4)) == Poly(16*y**4, x, y)

    assert Poly(7*x*y, x, y)**3 == Poly(343*x**3*y**3, x, y)

    # 检查不支持的操作是否引发异常
    raises(TypeError, lambda: Poly(x*y + 1, x, y)**(-1))
    raises(TypeError, lambda: Poly(x*y + 1, x, y)**x)

# 测试多项式类的除法和取模操作
def test_Poly_divmod():
    f, g = Poly(x**2), Poly(x)
    q, r = g, Poly(0, x)

    assert divmod(f, g) == (q, r)
    assert f // g == q
    assert f % g == r

    assert divmod(f, x) == (q, r)
    assert f // x == q
    assert f % x == r

    q, r = Poly(0, x), Poly(2, x)

    assert divmod(2, g) == (q, r)
    assert 2 // g == q
    assert 2 % g == r

    assert Poly(x)/Poly(x) == 1
    assert Poly(x**2)/Poly(x) == x
    assert Poly(x)/Poly(x**2) == 1/x

# 测试多项式类的相等和不相等操作
def test_Poly_eq_ne():
    assert (Poly(x + y, x, y) == Poly(x + y, x, y)) is True
    assert (Poly(x + y, x) == Poly(x + y, x, y)) is False
    assert (Poly(x + y, x, y) == Poly(x + y, x)) is False
    assert (Poly(x + y, x) == Poly(x + y, x)) is True
    assert (Poly(x + y, y) == Poly(x + y, y)) is True

    assert (Poly(x + y, x, y) == x + y) is True
    assert (Poly(x + y, x) == x + y) is True
    assert (Poly(x + y, x, y) == x + y) is True
    assert (Poly(x + y, x) == x + y) is True
    assert (Poly(x + y, y) == x + y) is True

    assert (Poly(x + y, x, y) != Poly(x + y, x, y)) is False
    assert (Poly(x + y, x) != Poly(x + y, x, y)) is True
    assert (Poly(x + y, x, y) != Poly(x + y, x)) is True
    assert (Poly(x + y, x) != Poly(x + y, x)) is False
    assert (Poly(x + y, y) != Poly(x + y, y)) is False

    assert (Poly(x + y, x, y) != x + y) is False
    assert (Poly(x + y, x) != x + y) is False
    assert (Poly(x + y, x, y) != x + y) is False
    assert (Poly(x + y, x) != x + y) is False
    assert (Poly(x + y, y) != x + y) is False

    assert (Poly(x, x) == sin(x)) is False
    assert (Poly(x, x) != sin(x)) is True
    # 断言：验证多项式 Poly(0, x) 的布尔值为 False
    assert not bool(Poly(0, x)) is True
    
    # 断言：验证多项式 Poly(1, x) 的布尔值为 True
    assert not bool(Poly(1, x)) is False
# 定义测试函数 test_Poly_properties
def test_Poly_properties():
    # 断言 Poly(0, x) 是否为零多项式
    assert Poly(0, x).is_zero is True
    # 断言 Poly(1, x) 是否不为零多项式
    assert Poly(1, x).is_zero is False

    # 断言 Poly(1, x) 是否为单位多项式
    assert Poly(1, x).is_one is True
    # 断言 Poly(2, x) 是否不为单位多项式
    assert Poly(2, x).is_one is False

    # 断言 Poly(x - 1, x) 是否为平方自由因式多项式
    assert Poly(x - 1, x).is_sqf is True
    # 断言 Poly((x - 1)**2, x) 是否不为平方自由因式多项式
    assert Poly((x - 1)**2, x).is_sqf is False

    # 断言 Poly(x - 1, x) 是否为首一多项式
    assert Poly(x - 1, x).is_monic is True
    # 断言 Poly(2*x - 1, x) 是否不为首一多项式
    assert Poly(2*x - 1, x).is_monic is False

    # 断言 Poly(3*x + 2, x) 是否为原始多项式
    assert Poly(3*x + 2, x).is_primitive is True
    # 断言 Poly(4*x + 2, x) 是否不为原始多项式
    assert Poly(4*x + 2, x).is_primitive is False

    # 断言 Poly(1, x) 是否为常数多项式
    assert Poly(1, x).is_ground is True
    # 断言 Poly(x, x) 是否不为常数多项式
    assert Poly(x, x).is_ground is False

    # 断言 Poly(x + y + z + 1) 是否为一次多项式
    assert Poly(x + y + z + 1).is_linear is True
    # 断言 Poly(x*y*z + 1) 是否不为一次多项式
    assert Poly(x*y*z + 1).is_linear is False

    # 断言 Poly(x*y + z + 1) 是否为二次多项式
    assert Poly(x*y + z + 1).is_quadratic is True
    # 断言 Poly(x*y*z + 1) 是否不为二次多项式
    assert Poly(x*y*z + 1).is_quadratic is False

    # 断言 Poly(x*y) 是否为单项式
    assert Poly(x*y).is_monomial is True
    # 断言 Poly(x*y + 1) 是否不为单项式
    assert Poly(x*y + 1).is_monomial is False

    # 断言 Poly(x**2 + x*y) 是否为齐次多项式
    assert Poly(x**2 + x*y).is_homogeneous is True
    # 断言 Poly(x**3 + x*y) 是否不为齐次多项式
    assert Poly(x**3 + x*y).is_homogeneous is False

    # 断言 Poly(x) 是否为一元多项式
    assert Poly(x).is_univariate is True
    # 断言 Poly(x*y) 是否不为一元多项式
    assert Poly(x*y).is_univariate is False

    # 断言 Poly(x*y) 是否为多元多项式
    assert Poly(x*y).is_multivariate is True
    # 断言 Poly(x) 是否不为多元多项式
    assert Poly(x).is_multivariate is False

    # 断言 Poly(x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1) 是否为旋回多项式
    assert Poly(x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1).is_cyclotomic is False
    # 断言 Poly(x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1) 是否为旋回多项式
    assert Poly(x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1).is_cyclotomic is True


# 定义测试函数 test_Poly_is_irreducible
def test_Poly_is_irreducible():
    # 断言 Poly(x**2 + x + 1) 是否为不可约多项式
    assert Poly(x**2 + x + 1).is_irreducible is True
    # 断言 Poly(x**2 + 2*x + 1) 是否不为不可约多项式
    assert Poly(x**2 + 2*x + 1).is_irreducible is False

    # 断言 Poly(7*x + 3, modulus=11) 是否为不可约多项式
    assert Poly(7*x + 3, modulus=11).is_irreducible is True
    # 断言 Poly(7*x**2 + 3*x + 1, modulus=11) 是否不为不可约多项式
    assert Poly(7*x**2 + 3*x + 1, modulus=11).is_irreducible is False


# 定义测试函数 test_Poly_subs
def test_Poly_subs():
    # 断言 Poly(x + 1) 在 x = 0 处的代换值为 1
    assert Poly(x + 1).subs(x, 0) == 1

    # 断言 Poly(x + 1) 在 x = x 处的代换值为 Poly(x + 1)
    assert Poly(x + 1).subs(x, x) == Poly(x + 1)
    # 断言 Poly(x + 1) 在 x = y 处的代换值为 Poly(y + 1)
    assert Poly(x + 1).subs(x, y) == Poly(y + 1)

    # 断言 Poly(x*y, x) 在 y = x 处的代换值为 x**2
    assert Poly(x*y, x).subs(y, x) == x**2
    # 断言 Poly(x*y, x) 在 x = y 处的代换值为 y**2
    assert Poly(x*y, x).subs(x, y) == y**2


# 定义测试函数 test_Poly_replace
def test_Poly_replace():
    # 断言 Poly(x + 1) 替换 x 后结果仍为 Poly(x + 1)
    assert Poly(x + 1).replace(x) == Poly(x + 1)
    # 断言 Poly(x + 1) 替换 y 后结果为 Poly(y + 1)
    assert Poly(x + 1).replace(y) == Poly(y + 1)

    # 使用 lambda 函数检测抛出异常 PolynomialError 是否正确
    raises(PolynomialError, lambda: Poly(x + y).replace(z))

    # 断言 Poly(x + 1) 替换 x 为 x 后结果仍为 Poly(x + 1)
    assert Poly(x + 1).replace(x, x) == Poly(x + 1)
    # 断言 Poly(x + 1) 替换 x 为 y 后结果为 Poly(y + 1)
    assert Poly(x + 1).replace(x, y) == Poly(y + 1)

    # 断言 Poly(x + y) 替换 x 为 x 后结果仍为 Poly(x + y)
    assert Poly(x + y).replace(x, x) == Poly(x + y)
    # 断言 Poly(x + y) 替换 x 为 z 后结果为 Poly(z + y, z, y)
    assert Poly(x + y).replace(x, z) == Poly(z + y, z, y)

    # 断言 Poly(x + y) 替换 y 为 y 后结果仍为 Poly(x + y)
    assert Poly(x + y).replace(y, y) == Poly(x + y)
    # 断言 Poly(x + y) 替换 y 为 z 后结果为 Poly(x + z, x, z)
    assert Poly(x + y).replace(y, z) == Poly(x + z, x, z)
    # 断言 Poly(x + y) 替换 z 为 t 后结果仍为 Poly(x + y)
    assert Poly(x + y).replace(z, t) == Poly(x + y)

    # 使用 lambda 函数检测抛出异常 PolynomialError 是否正确
    raises(PolynomialError, lambda: Poly(x + y).replace(x, y))

    # 断言 Poly(x + y, x) 替换 x 为 z 后结果为 Poly(z + y, z)
    assert Poly(x + y,
    # 断言：测试多项式对象的重新排序功能，验证是否不改变多项式的表达式结果。
    assert Poly(x + y, y, x).reorder(y, x) == Poly(x + y, y, x)
    
    # 断言：测试多项式对象在指定变量顺序为 x 时的重新排序功能，验证是否不改变多项式的表达式结果。
    assert Poly(x + y, x, y).reorder(wrt=x) == Poly(x + y, x, y)
    
    # 断言：测试多项式对象在指定变量顺序为 y 时的重新排序功能，验证是否将变量 y 排在变量 x 前面，而不改变表达式的结果。
    assert Poly(x + y, x, y).reorder(wrt=y) == Poly(x + y, y, x)
# 定义测试函数 test_Poly_ltrim，用于测试多项式对象的 ltrim 方法
def test_Poly_ltrim():
    # 创建 Poly 对象 f，将多项式 y**2 + y*z**2 作为参数传入，并调用 ltrim 方法去除 y 变量
    f = Poly(y**2 + y*z**2, x, y, z).ltrim(y)
    # 断言 f 的表达式形式与 y**2 + y*z**2 相等，并且生成器 gens 为 (y, z)
    assert f.as_expr() == y**2 + y*z**2 and f.gens == (y, z)
    # 断言 Poly 对象 x*y - x 的结果调用 ltrim 方法去除 1 后仍为其本身
    assert Poly(x*y - x, z, x, y).ltrim(1) == Poly(x*y - x, x, y)

    # 断言 Poly 对象 x*y**2 + y**2 调用 ltrim 方法去除 y 抛出 PolynomialError 异常
    raises(PolynomialError, lambda: Poly(x*y**2 + y**2, x, y).ltrim(y))
    # 断言 Poly 对象 x*y - x 调用 ltrim 方法去除 -1 抛出 PolynomialError 异常
    raises(PolynomialError, lambda: Poly(x*y - x, x, y).ltrim(-1))


# 定义测试函数 test_Poly_has_only_gens，测试多项式对象的 has_only_gens 方法
def test_Poly_has_only_gens():
    # 断言 Poly 对象 x*y + 1 的生成器是否仅为 x 和 y，返回 True
    assert Poly(x*y + 1, x, y, z).has_only_gens(x, y) is True
    # 断言 Poly 对象 x*y + z 的生成器是否仅为 x 和 y，返回 False
    assert Poly(x*y + z, x, y, z).has_only_gens(x, y) is False

    # 断言 Poly 对象 x*y**2 + y**2 调用 has_only_gens 方法抛出 GeneratorsError 异常
    raises(GeneratorsError, lambda: Poly(x*y**2 + y**2, x, y).has_only_gens(t))


# 定义测试函数 test_Poly_to_ring，测试多项式对象的 to_ring 方法
def test_Poly_to_ring():
    # 断言 Poly 对象 2*x + 1，指定 domain='ZZ' 后调用 to_ring 方法结果与原多项式相等
    assert Poly(2*x + 1, domain='ZZ').to_ring() == Poly(2*x + 1, domain='ZZ')
    # 断言 Poly 对象 2*x + 1，指定 domain='QQ' 后调用 to_ring 方法结果与 Poly(2*x + 1, domain='ZZ') 相等
    assert Poly(2*x + 1, domain='QQ').to_ring() == Poly(2*x + 1, domain='ZZ')

    # 断言 Poly 对象 x/2 + 1 调用 to_ring 方法抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: Poly(x/2 + 1).to_ring())
    # 断言 Poly 对象 2*x + 1，指定 modulus=3 后调用 to_ring 方法抛出 DomainError 异常
    raises(DomainError, lambda: Poly(2*x + 1, modulus=3).to_ring())


# 定义测试函数 test_Poly_to_field，测试多项式对象的 to_field 方法
def test_Poly_to_field():
    # 断言 Poly 对象 2*x + 1，指定 domain='ZZ' 后调用 to_field 方法结果与 Poly(2*x + 1, domain='QQ') 相等
    assert Poly(2*x + 1, domain='ZZ').to_field() == Poly(2*x + 1, domain='QQ')
    # 断言 Poly 对象 2*x + 1，指定 domain='QQ' 后调用 to_field 方法结果与 Poly(2*x + 1, domain='QQ') 相等
    assert Poly(2*x + 1, domain='QQ').to_field() == Poly(2*x + 1, domain='QQ')

    # 断言 Poly 对象 x/2 + 1，指定 domain='QQ' 后调用 to_field 方法结果与原多项式相等
    assert Poly(x/2 + 1, domain='QQ').to_field() == Poly(x/2 + 1, domain='QQ')
    # 断言 Poly 对象 2*x + 1，指定 modulus=3 后调用 to_field 方法结果与 Poly(2*x + 1, modulus=3) 相等
    assert Poly(2*x + 1, modulus=3).to_field() == Poly(2*x + 1, modulus=3)

    # 断言 Poly 对象 2.0*x + 1.0 调用 to_field 方法结果与原多项式相等
    assert Poly(2.0*x + 1.0).to_field() == Poly(2.0*x + 1.0)


# 定义测试函数 test_Poly_to_exact，测试多项式对象的 to_exact 方法
def test_Poly_to_exact():
    # 断言 Poly 对象 2*x 调用 to_exact 方法结果与原多项式相等
    assert Poly(2*x).to_exact() == Poly(2*x)
    # 断言 Poly 对象 x/2 调用 to_exact 方法结果与原多项式相等
    assert Poly(x/2).to_exact() == Poly(x/2)

    # 断言 Poly 对象 0.1*x 调用 to_exact 方法结果与 Poly(x/10) 相等
    assert Poly(0.1*x).to_exact() == Poly(x/10)


# 定义测试函数 test_Poly_retract，测试多项式对象的 retract 方法
def test_Poly_retract():
    # 创建 Poly 对象 f，多项式为 x**2 + 1，指定 domain=QQ[y] 后调用 retract 方法
    f = Poly(x**2 + 1, x, domain=QQ[y])

    # 断言 f 调用 retract 方法结果与 Poly(x**2 + 1, x, domain='ZZ') 相等
    assert f.retract() == Poly(x**2 + 1, x, domain='ZZ')
    # 断言 f 调用 retract 方法，指定 field=True 后结果与 Poly(x**2 + 1, x, domain='QQ') 相等
    assert f.retract(field=True) == Poly(x**2 + 1, x, domain='QQ')

    # 断言 Poly 对象 0，指定变量 x, y 后调用 retract 方法结果与 Poly(0, x, y) 相等
    assert Poly(0, x, y).retract() == Poly(0, x, y)


# 定义测试函数 test_Poly_slice，测试多项式对象的 slice 方法
def test_Poly_slice():
    # 创建 Poly 对象 f，多项式为 x**3 + 2*x**2 + 3*x + 4
    f = Poly(x**3 + 2*x**2 + 3*x + 4)

    # 断言 f 调用 slice 方法，区间 [0, 0] 结果与 Poly(0, x) 相等
    assert f.slice(0, 0) == Poly(0, x)
    # 断言 f 调用 slice 方法，区间 [0, 1] 结果与 Poly(4, x) 相等
    assert f.slice(0, 1) == Poly(4, x)
    # 断言 f 调用 slice 方法，区间 [0, 2] 结果与 Poly(3*x + 4, x) 相等
    assert f.slice(0, 2) == Poly(3*x + 4, x)
    # 断言 f 调用 slice 方法，区间 [0, 3] 结果与 Poly(2*x**2 + 3*x + 4, x) 相等
    assert f.slice(0, 3) == Poly(2*x**2 + 3*x + 4, x)
    # 断言 f 调用 slice 方法，区间 [0, 4] 结果与 Poly(x**3 + 2*x**2 + 3*x + 4, x) 相等
    assert f.slice(0, 4) == Poly(x**3 + 2*x**2 + 3*x + 4, x)

    # 断言 f 调用 slice 方法，指定切片变量 x，区间 [0, 0] 结果与
    # 断言：验证多项式对象的 monoms() 方法返回的单项式列表与预期列表是否相等
    assert Poly(7*x**4 + 2*x + 1, x).monoms() == [(4,), (1,), (0,)]
    
    # 断言：验证多项式对象在使用 'lex' 排序后的 monoms() 方法返回的单项式列表与预期列表是否相等
    assert Poly(x*y**7 + 2*x**2*y**3).monoms('lex') == [(2, 3), (1, 7)]
    
    # 断言：验证多项式对象在使用 'grlex' 排序后的 monoms() 方法返回的单项式列表与预期列表是否相等
    assert Poly(x*y**7 + 2*x**2*y**3).monoms('grlex') == [(1, 7), (2, 3)]
# 定义一个名为 test_Poly_terms 的测试函数
def test_Poly_terms():
    # 断言：Poly(0, x) 的 terms 方法返回 [((0,), 0)]
    assert Poly(0, x).terms() == [((0,), 0)]
    # 断言：Poly(1, x) 的 terms 方法返回 [((0,), 1)]
    assert Poly(1, x).terms() == [((0,), 1)]

    # 断言：Poly(2*x + 1, x) 的 terms 方法返回 [((1,), 2), ((0,), 1)]
    assert Poly(2*x + 1, x).terms() == [((1,), 2), ((0,), 1)]

    # 断言：Poly(7*x**2 + 2*x + 1, x) 的 terms 方法返回 [((2,), 7), ((1,), 2), ((0,), 1)]
    assert Poly(7*x**2 + 2*x + 1, x).terms() == [((2,), 7), ((1,), 2), ((0,), 1)]
    
    # 断言：Poly(7*x**4 + 2*x + 1, x) 的 terms 方法返回 [((4,), 7), ((1,), 2), ((0,), 1)]
    assert Poly(7*x**4 + 2*x + 1, x).terms() == [((4,), 7), ((1,), 2), ((0,), 1)]

    # 断言：Poly(x*y**7 + 2*x**2*y**3).terms('lex') 的 terms 方法返回 [((2, 3), 2), ((1, 7), 1)]
    assert Poly(x*y**7 + 2*x**2*y**3).terms('lex') == [((2, 3), 2), ((1, 7), 1)]
    
    # 断言：Poly(x*y**7 + 2*x**2*y**3).terms('grlex') 的 terms 方法返回 [((1, 7), 1), ((2, 3), 2)]
    assert Poly(x*y**7 + 2*x**2*y**3).terms('grlex') == [((1, 7), 1), ((2, 3), 2)]


# 定义一个名为 test_Poly_all_coeffs 的测试函数
def test_Poly_all_coeffs():
    # 断言：Poly(0, x) 的 all_coeffs 方法返回 [0]
    assert Poly(0, x).all_coeffs() == [0]
    # 断言：Poly(1, x) 的 all_coeffs 方法返回 [1]
    assert Poly(1, x).all_coeffs() == [1]

    # 断言：Poly(2*x + 1, x) 的 all_coeffs 方法返回 [2, 1]
    assert Poly(2*x + 1, x).all_coeffs() == [2, 1]

    # 断言：Poly(7*x**2 + 2*x + 1, x) 的 all_coeffs 方法返回 [7, 2, 1]
    assert Poly(7*x**2 + 2*x + 1, x).all_coeffs() == [7, 2, 1]
    
    # 断言：Poly(7*x**4 + 2*x + 1, x) 的 all_coeffs 方法返回 [7, 0, 0, 2, 1]
    assert Poly(7*x**4 + 2*x + 1, x).all_coeffs() == [7, 0, 0, 2, 1]


# 定义一个名为 test_Poly_all_monoms 的测试函数
def test_Poly_all_monoms():
    # 断言：Poly(0, x) 的 all_monoms 方法返回 [(0,)]
    assert Poly(0, x).all_monoms() == [(0,)]
    # 断言：Poly(1, x) 的 all_monoms 方法返回 [(0,)]
    assert Poly(1, x).all_monoms() == [(0,)]

    # 断言：Poly(2*x + 1, x) 的 all_monoms 方法返回 [(1,), (0,)]
    assert Poly(2*x + 1, x).all_monoms() == [(1,), (0,)]

    # 断言：Poly(7*x**2 + 2*x + 1, x) 的 all_monoms 方法返回 [(2,), (1,), (0,)]
    assert Poly(7*x**2 + 2*x + 1, x).all_monoms() == [(2,), (1,), (0,)]
    
    # 断言：Poly(7*x**4 + 2*x + 1, x) 的 all_monoms 方法返回 [(4,), (3,), (2,), (1,), (0,)]
    assert Poly(7*x**4 + 2*x + 1, x).all_monoms() == [(4,), (3,), (2,), (1,), (0,)]


# 定义一个名为 test_Poly_all_terms 的测试函数
def test_Poly_all_terms():
    # 断言：Poly(0, x) 的 all_terms 方法返回 [((0,), 0)]
    assert Poly(0, x).all_terms() == [((0,), 0)]
    # 断言：Poly(1, x) 的 all_terms 方法返回 [((0,), 1)]
    assert Poly(1, x).all_terms() == [((0,), 1)]

    # 断言：Poly(2*x + 1, x) 的 all_terms 方法返回 [((1,), 2), ((0,), 1)]
    assert Poly(2*x + 1, x).all_terms() == [((1,), 2), ((0,), 1)]

    # 断言：Poly(7*x**2 + 2*x + 1, x) 的 all_terms 方法返回 [((2,), 7), ((1,), 2), ((0,), 1)]
    assert Poly(7*x**2 + 2*x + 1, x).all_terms() == [((2,), 7), ((1,), 2), ((0,), 1)]
    
    # 断言：Poly(7*x**4 + 2*x + 1, x) 的 all_terms 方法返回 [((4,), 7), ((3,), 0), ((2,), 0), ((1,), 2), ((0,), 1)]
    assert Poly(7*x**4 + 2*x + 1, x).all_terms() == [((4,), 7), ((3,), 0), ((2,), 0), ((1,), 2), ((0,), 1)]


# 定义一个名为 test_Poly_termwise 的测试函数
def test_Poly_termwise():
    # 创建 Poly(x**2 + 20*x + 400) 并赋值给 f
    f = Poly(x**2 + 20*x + 400)
    # 创建 Poly(x**2 + 2*x + 4) 并赋值给 g
    g = Poly(x**2 + 2*x + 4)

    # 定义 func 函数，用于 termwise 方法的参数
    def func(monom, coeff):
        (k,) = monom
        return coeff//10**(2 - k)

    # 断言：f 的 termwise 方法应用 func 函数后结果等于 g
    assert f.termwise(func) == g

    # 重新定义 func 函数，返回值为元组 (k,) 和 coeff//10**(2 - k)
    def func(monom, coeff):
        (k,) = monom
        return (k,), coeff//10**(2 - k)

    # 断言：f 的 termwise 方法应用重新定义的 func 函数后结果等于 g
    assert f.termwise(func) == g


# 定义一个名为 test_Poly_length 的测试函数
def test_Poly_length():
    # 断言：Poly(0, x) 的 length 方法返回 0
    assert Poly(0, x).length() == 0
    # 断言：Poly(1, x) 的 length 方法返回 1
    assert Poly(1, x).length() == 1
    # 断言：Poly(x, x) 的 length 方法返回 1
    assert Poly(x, x).length() == 1

    # 断言：Poly(x + 1, x) 的 length 方法返回 2
    assert Poly(x + 1, x).length() == 2
    # 断言：Poly(x**2 + 1, x) 的 length 方法返回 2
    assert Poly(x**2 + 1, x).length() == 2
    # 断言：Poly(x**2 + x + 1, x) 的 length 方法返回 3
    assert Poly(x**2 + x + 1, x).length() == 3


# 定义一个名为 test_Poly_as_dict 的测试函数
def test_Poly_as_dict():
    # 断言：Poly(0, x) 的 as_dict 方法返回 {}
    assert Poly(0, x).
    # 断言：将多项式 Poly(3*x**2*y*z**3 + 4*x*y + 5*x*z) 转换为表达式后与原多项式比较，确认相等性
    assert Poly(3*x**2*y*z**3 + 4*x*y + 5*x*z).as_expr() == 3*x**2*y*z**3 + 4*x*y + 5*x*z
    
    # 创建多项式对象 f = Poly(x**2 + 2*x*y**2 - y, x, y)
    f = Poly(x**2 + 2*x*y**2 - y, x, y)
    
    # 断言：将多项式 f 转换为表达式后与预期表达式 -y + x**2 + 2*x*y**2 比较，确认相等性
    assert f.as_expr() == -y + x**2 + 2*x*y**2
    
    # 断言：将多项式 f 中 x 变量替换为 5 后，将表达式计算得到结果与预期表达式 25 - y + 10*y**2 比较，确认相等性
    assert f.as_expr({x: 5}) == 25 - y + 10*y**2
    
    # 断言：将多项式 f 中 y 变量替换为 6 后，将表达式计算得到结果与预期表达式 -6 + 72*x + x**2 比较，确认相等性
    assert f.as_expr({y: 6}) == -6 + 72*x + x**2
    
    # 断言：将多项式 f 中 x 和 y 变量分别替换为 5 和 6 后，将表达式计算得到结果与预期值 379 比较，确认相等性
    assert f.as_expr({x: 5, y: 6}) == 379
    
    # 断言：将多项式 f 中 x 和 y 变量分别替换为 5 和 6 后，直接计算表达式结果与预期值 379 比较，确认相等性
    assert f.as_expr(5, 6) == 379
    
    # 断言：使用 lambda 表达式尝试将多项式 f 中未定义的变量 z 替换为 7，预期会引发 GeneratorsError 异常
    raises(GeneratorsError, lambda: f.as_expr({z: 7}))
# 测试函数：测试 Poly 类的 lift 方法
def test_Poly_lift():
    # 断言 Poly 对象的 lift 方法返回的结果与给定的多项式相等
    assert Poly(x**4 - I*x + 17*I, x, gaussian=True).lift() == \
        Poly(x**16 + 2*x**10 + 578*x**8 + x**4 - 578*x**2 + 83521,
             x, domain='QQ')


# 测试函数：测试 Poly 类的 deflate 方法
def test_Poly_deflate():
    # 检查 Poly 对象系数为 0 时的 deflate 结果
    assert Poly(0, x).deflate() == ((1,), Poly(0, x))
    # 检查 Poly 对象系数为 1 时的 deflate 结果
    assert Poly(1, x).deflate() == ((1,), Poly(1, x))
    # 检查 Poly 对象为一次多项式时的 deflate 结果
    assert Poly(x, x).deflate() == ((1,), Poly(x, x))

    # 检查二次多项式的 deflate 结果
    assert Poly(x**2, x).deflate() == ((2,), Poly(x, x))
    # 检查十七次多项式的 deflate 结果
    assert Poly(x**17, x).deflate() == ((17,), Poly(x, x))

    # 检查包含多个变量的多项式的 deflate 结果
    assert Poly(
        x**2*y*z**11 + x**4*z**11).deflate() == ((2, 1, 11), Poly(x*y*z + x**2*z))


# 测试函数：测试 Poly 类的 inject 方法
def test_Poly_inject():
    # 创建 Poly 对象 f
    f = Poly(x**2*y + x*y**3 + x*y + 1, x)

    # 断言使用 inject 方法保持变量 x 不变
    assert f.inject() == Poly(x**2*y + x*y**3 + x*y + 1, x, y)
    # 断言使用 inject 方法将 y 插入到变量 x 之前
    assert f.inject(front=True) == Poly(y**3*x + y*x**2 + y*x + 1, y, x)


# 测试函数：测试 Poly 类的 eject 方法
def test_Poly_eject():
    # 创建 Poly 对象 f
    f = Poly(x**2*y + x*y**3 + x*y + 1, x, y)

    # 断言使用 eject 方法将 x 从多项式中排除
    assert f.eject(x) == Poly(x*y**3 + (x**2 + x)*y + 1, y, domain='ZZ[x]')
    # 断言使用 eject 方法将 y 从多项式中排除
    assert f.eject(y) == Poly(y*x**2 + (y**3 + y)*x + 1, x, domain='ZZ[y]')

    # 创建包含多个变量的多项式 g
    ex = x + y + z + t + w
    g = Poly(ex, x, y, z, t, w)

    # 断言使用 eject 方法将 x 从多项式 g 中排除
    assert g.eject(x) == Poly(ex, y, z, t, w, domain='ZZ[x]')
    # 断言使用 eject 方法依次将 x 和 y 从多项式 g 中排除
    assert g.eject(x, y) == Poly(ex, z, t, w, domain='ZZ[x, y]')
    # 断言使用 eject 方法依次将 x、y 和 z 从多项式 g 中排除
    assert g.eject(x, y, z) == Poly(ex, t, w, domain='ZZ[x, y, z]')
    # 断言使用 eject 方法将 w 从多项式 g 中排除
    assert g.eject(w) == Poly(ex, x, y, z, t, domain='ZZ[w]')
    # 断言使用 eject 方法依次将 t 和 w 从多项式 g 中排除
    assert g.eject(t, w) == Poly(ex, x, y, z, domain='ZZ[t, w]')
    # 断言使用 eject 方法依次将 z、t 和 w 从多项式 g 中排除
    assert g.eject(z, t, w) == Poly(ex, x, y, domain='ZZ[z, t, w]')

    # 检查不支持的 eject 情况
    raises(DomainError, lambda: Poly(x*y, x, y, domain=ZZ[z]).eject(y))
    raises(NotImplementedError, lambda: Poly(x*y, x, y, z).eject(y))


# 测试函数：测试 Poly 类的 exclude 方法
def test_Poly_exclude():
    # 断言排除 Poly 对象中的第二个变量 y 后的结果
    assert Poly(x, x, y).exclude() == Poly(x, x)
    # 断言排除 Poly 对象中的所有变量后的结果
    assert Poly(x*y, x, y).exclude() == Poly(x*y, x, y)
    # 断言排除 Poly 对象中的常数项后的结果
    assert Poly(1, x, y).exclude() == Poly(1, x, y)


# 测试函数：测试 Poly 类的 _gen_to_level 方法
def test_Poly__gen_to_level():
    # 断言返回 -2 对应的 _gen_to_level 结果为 0
    assert Poly(1, x, y)._gen_to_level(-2) == 0
    # 断言返回 -1 对应的 _gen_to_level 结果为 1
    assert Poly(1, x, y)._gen_to_level(-1) == 1
    # 断言返回 0 对应的 _gen_to_level 结果为 0
    assert Poly(1, x, y)._gen_to_level( 0) == 0
    # 断言返回 1 对应的 _gen_to_level 结果为 1
    assert Poly(1, x, y)._gen_to_level( 1) == 1

    # 检查不支持的 _gen_to_level 调用
    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level(-3))
    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level( 2))

    # 断言返回变量 x 对应的 _gen_to_level 结果为 0
    assert Poly(1, x, y)._gen_to_level(x) == 0
    # 断言返回变量 y 对应的 _gen_to_level 结果为 1
    assert Poly(1, x, y)._gen_to_level(y) == 1

    # 断言返回字符串 'x' 对应的 _gen_to_level 结果为 0
    assert Poly(1, x, y)._gen_to_level('x') == 0
    # 断言返回字符串 'y' 对应的 _gen_to_level 结果为 1
    assert Poly(1, x, y)._gen_to_level('y') == 1

    # 检查不支持的 _gen_to_level 调用
    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level(z))
    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level('z'))


# 测试函数：测试 Poly 类的 degree 方法
def test_Poly_degree():
    # 断言 Poly 对象为常数 0 时，其 degree 为负无穷
    assert Poly(0, x).degree() is -oo
    # 断言 Poly 对象为常数 1 时，其 degree 为 0
    assert Poly(1, x).degree() == 0
    # 断言 Poly 对象为一次多项式 x 时，其 degree 为 1
    assert Poly(x, x).degree() == 1

    # 断言 Poly 对象为常数 0 时，针对变量 0 的 degree 为负无穷
    assert Poly(0, x).degree(gen=0) is -oo
    # 断言 Poly 对象为常数 1 时，针对变量 0 的 degree 为 0
    assert Poly(1, x).degree(gen=0) == 0
    # 断言 Poly 对象为一次多项式 x 时，针对变量 0 的 degree 为 1
    assert Poly(x, x).degree(gen=0) == 1

    # 断言 Poly 对象为常数 0 时，针对变量 x 的 degree 为负无穷
    assert Poly(0, x).degree(gen=x) is -oo
    # 断言：使用 Poly 类创建一个多项式 Poly(x, x)，并指定变量 'x' 计算其在 'x' 变量下的次数，预期结果为 1
    assert Poly(x, x).degree(gen='x') == 1
    
    # 使用 lambda 表达式和 raises 函数，验证在创建常数多项式 Poly(1, x) 时抛出 PolynomialError 异常，因为 'gen' 参数不是有效的生成器
    raises(PolynomialError, lambda: Poly(1, x).degree(gen=1))
    raises(PolynomialError, lambda: Poly(1, x).degree(gen=y))
    raises(PolynomialError, lambda: Poly(1, x).degree(gen='y'))
    
    # 断言：创建多项式 Poly(1, x, y)，并计算其次数，预期结果为 0，因为多项式只包含常数项
    assert Poly(1, x, y).degree() == 0
    
    # 断言：创建多项式 Poly(2*y, x, y)，并计算其次数，预期结果为 0，因为多项式只包含 y 的一次项
    assert Poly(2*y, x, y).degree() == 0
    
    # 断言：创建多项式 Poly(x*y, x, y)，并计算其次数，预期结果为 1，因为多项式包含 x*y 的一次项
    assert Poly(x*y, x, y).degree() == 1
    
    # 断言：创建多项式 Poly(1, x, y)，并指定变量 'x' 计算其在 'x' 变量下的次数，预期结果为 0，因为多项式只包含常数项
    assert Poly(1, x, y).degree(gen=x) == 0
    
    # 断言：创建多项式 Poly(2*y, x, y)，并指定变量 'x' 计算其在 'x' 变量下的次数，预期结果为 0，因为多项式只包含 y 的一次项
    assert Poly(2*y, x, y).degree(gen=x) == 0
    
    # 断言：创建多项式 Poly(x*y, x, y)，并指定变量 'x' 计算其在 'x' 变量下的次数，预期结果为 1，因为多项式包含 x*y 的一次项
    assert Poly(x*y, x, y).degree(gen=x) == 1
    
    # 断言：创建多项式 Poly(1, x, y)，并指定变量 'y' 计算其在 'y' 变量下的次数，预期结果为 0，因为多项式只包含常数项
    assert Poly(1, x, y).degree(gen=y) == 0
    
    # 断言：创建多项式 Poly(2*y, x, y)，并指定变量 'y' 计算其在 'y' 变量下的次数，预期结果为 1，因为多项式包含 y 的一次项
    assert Poly(2*y, x, y).degree(gen=y) == 1
    
    # 断言：创建多项式 Poly(x*y, x, y)，并指定变量 'y' 计算其在 'y' 变量下的次数，预期结果为 1，因为多项式包含 x*y 的一次项
    assert Poly(x*y, x, y).degree(gen=y) == 1
    
    # 断言：对于常数 0，以变量 'x' 计算其次数，预期结果为负无穷
    assert degree(0, x) is -oo
    
    # 断言：对于常数 1，以变量 'x' 计算其次数，预期结果为 0
    assert degree(1, x) == 0
    
    # 断言：对于变量 'x'，计算其在自身变量下的次数，预期结果为 1
    assert degree(x, x) == 1
    
    # 断言：对于多项式 x*y**2，以变量 'x' 计算其次数，预期结果为 1
    assert degree(x*y**2, x) == 1
    
    # 断言：对于多项式 x*y**2，以变量 'y' 计算其次数，预期结果为 2
    assert degree(x*y**2, y) == 2
    
    # 断言：对于多项式 x*y**2，以变量 'z' 计算其次数，预期结果为 0，因为 z 不在多项式中
    assert degree(x*y**2, z) == 0
    
    # 断言：对于常数 pi，计算其次数，预期结果为 1
    assert degree(pi) == 1
    
    # 使用 lambda 表达式和 raises 函数，验证对于多项式 y**2 + x**3，抛出 TypeError 异常，因为没有指定变量进行计算
    raises(TypeError, lambda: degree(y**2 + x**3))
    
    # 使用 lambda 表达式和 raises 函数，验证对于多项式 y**2 + x**3，抛出 TypeError 异常，因为 'gen' 参数不是有效的生成器
    raises(TypeError, lambda: degree(y**2 + x**3, 1))
    
    # 使用 lambda 表达式和 raises 函数，验证对于多项式 x，抛出 PolynomialError 异常，因为 'gen' 参数不是有效的生成器
    raises(PolynomialError, lambda: degree(x, 1.1))
    
    # 使用 lambda 表达式和 raises 函数，验证对于多项式 x**2/(x**3 + 1)，抛出 PolynomialError 异常，因为无法计算次数
    raises(PolynomialError, lambda: degree(x**2/(x**3 + 1), x))
    
    # 断言：使用 Poly 类创建多项式 Poly(0, x)，并指定变量 'z' 计算其次数，预期结果为负无穷，因为多项式为零
    assert degree(Poly(0,x),z) is -oo
    
    # 断言：使用 Poly 类创建多项式 Poly(1, x)，并指定变量 'z' 计算其次数，预期结果为 0，因为多项式只包含常数项
    assert degree(Poly(1,x),z) == 0
    
    # 断言：使用 Poly 类创建多项式 Poly(x**2+y**3,y)，计算其在 'y' 变量下的次数，预期结果为 3
    assert degree(Poly(x**2+y**3,y)) == 3
    
    # 断言：使用 Poly 类创建多项式 Poly(y**2 + x**3, y, x)，并指定 '1' 变量计算其次数，预期结果为 3
    assert degree(Poly(y**2 + x**3, y, x), 1) == 3
    
    # 断言：使用 Poly 类创建多项式 Poly(y**2 + x**3, x)，并指定 'z' 变量计算其次数，预期结果为 0，因为 z 不在多项式中
    assert degree(Poly(y**2 + x**3, x), z) == 0
    
    # 断言：使用 Poly 类创建多项式 Poly(y**2 + x**3 + z**4, x)，并指定 'z' 变量计算其次数，预期结果为 4
    assert degree(Poly(y**2 + x**3 + z**4, x), z) == 4
def test_Poly_degree_list():
    # 检查多项式在各变量上的次数列表，当多项式为常数0时，次数为负无穷
    assert Poly(0, x).degree_list() == (-oo,)
    assert Poly(0, x, y).degree_list() == (-oo, -oo)
    assert Poly(0, x, y, z).degree_list() == (-oo, -oo, -oo)

    # 检查多项式在各变量上的次数列表，对于常数1，次数为0
    assert Poly(1, x).degree_list() == (0,)
    assert Poly(1, x, y).degree_list() == (0, 0)
    assert Poly(1, x, y, z).degree_list() == (0, 0, 0)

    # 检查多项式在各变量上的次数列表，给定具体的多项式表达式
    assert Poly(x**2*y + x**3*z**2 + 1).degree_list() == (3, 1, 2)

    # 检查非多项式对象时是否引发异常
    assert degree_list(1, x) == (0,)
    raises(ComputationFailed, lambda: degree_list(1))


def test_Poly_total_degree():
    # 检查多项式的总次数计算
    assert Poly(x**2*y + x**3*z**2 + 1).total_degree() == 5
    assert Poly(x**2 + z**3).total_degree() == 3
    assert Poly(x*y*z + z**4).total_degree() == 4
    assert Poly(x**3 + x + 1).total_degree() == 3

    # 检查在指定变量上的多项式总次数计算
    assert total_degree(x*y + z**3) == 3
    assert total_degree(x*y + z**3, x, y) == 2
    assert total_degree(1) == 0
    assert total_degree(Poly(y**2 + x**3 + z**4)) == 4
    assert total_degree(Poly(y**2 + x**3 + z**4, x)) == 3
    assert total_degree(Poly(y**2 + x**3 + z**4, x), z) == 4
    assert total_degree(Poly(x**9 + x*z*y + x**3*z**2 + z**7, x), z) == 7


def test_Poly_homogenize():
    # 检查多项式的齐次化
    assert Poly(x**2+y).homogenize(z) == Poly(x**2+y*z)
    assert Poly(x+y).homogenize(z) == Poly(x+y, x, y, z)
    assert Poly(x+y**2).homogenize(y) == Poly(x*y+y**2)


def test_Poly_homogeneous_order():
    # 检查多项式的齐次阶数计算
    assert Poly(0, x, y).homogeneous_order() is -oo
    assert Poly(1, x, y).homogeneous_order() == 0
    assert Poly(x, x, y).homogeneous_order() == 1
    assert Poly(x*y, x, y).homogeneous_order() == 2

    # 检查非齐次多项式的情况
    assert Poly(x + 1, x, y).homogeneous_order() is None
    assert Poly(x*y + x, x, y).homogeneous_order() is None

    # 检查具体多项式的齐次阶数计算
    assert Poly(x**5 + 2*x**3*y**2 + 9*x*y**4).homogeneous_order() == 5
    assert Poly(x**5 + 2*x**3*y**3 + 9*x*y**4).homogeneous_order() is None


def test_Poly_LC():
    # 检查多项式的主导系数（leading coefficient）
    assert Poly(0, x).LC() == 0
    assert Poly(1, x).LC() == 1
    assert Poly(2*x**2 + x, x).LC() == 2

    # 检查多项式的主导系数在指定排序方式下的计算
    assert Poly(x*y**7 + 2*x**2*y**3).LC('lex') == 2
    assert Poly(x*y**7 + 2*x**2*y**3).LC('grlex') == 1

    # 检查函数接口LC在指定排序方式下的计算
    assert LC(x*y**7 + 2*x**2*y**3, order='lex') == 2
    assert LC(x*y**7 + 2*x**2*y**3, order='grlex') == 1


def test_Poly_TC():
    # 检查多项式的尾项系数（trailing coefficient）
    assert Poly(0, x).TC() == 0
    assert Poly(1, x).TC() == 1
    assert Poly(2*x**2 + x, x).TC() == 0


def test_Poly_EC():
    # 检查多项式的额外系数（extra coefficient）
    assert Poly(0, x).EC() == 0
    assert Poly(1, x).EC() == 1
    assert Poly(2*x**2 + x, x).EC() == 1

    # 检查多项式的额外系数在指定排序方式下的计算
    assert Poly(x*y**7 + 2*x**2*y**3).EC('lex') == 1
    assert Poly(x*y**7 + 2*x**2*y**3).EC('grlex') == 2


def test_Poly_coeff():
    # 检查多项式的单项式系数（coefficient of a monomial）
    assert Poly(0, x).coeff_monomial(1) == 0
    assert Poly(0, x).coeff_monomial(x) == 0

    assert Poly(1, x).coeff_monomial(1) == 1
    assert Poly(1, x).coeff_monomial(x) == 0

    assert Poly(x**8, x).coeff_monomial(1) == 0
    assert Poly(x**8, x).coeff_monomial(x**7) == 0
    assert Poly(x**8, x).coeff_monomial(x**8) == 1
    # 断言：检查多项式 Poly(x**8, x) 中 x**9 的系数是否为 0
    assert Poly(x**8, x).coeff_monomial(x**9) == 0
    
    # 断言：检查多项式 Poly(3*x*y**2 + 1, x, y) 中常数项 1 的系数是否为 1
    assert Poly(3*x*y**2 + 1, x, y).coeff_monomial(1) == 1
    # 断言：检查多项式 Poly(3*x*y**2 + 1, x, y) 中 x*y**2 的系数是否为 3
    assert Poly(3*x*y**2 + 1, x, y).coeff_monomial(x*y**2) == 3
    
    # 创建多项式 p = Poly(24*x*y*exp(8) + 23*x, x, y)
    p = Poly(24*x*y*exp(8) + 23*x, x, y)
    
    # 断言：检查多项式 p 中 x 的系数是否为 23
    assert p.coeff_monomial(x) == 23
    # 断言：检查多项式 p 中 y 的系数是否为 0
    assert p.coeff_monomial(y) == 0
    # 断言：检查多项式 p 中 x*y 的系数是否为 24*exp(8)
    assert p.coeff_monomial(x*y) == 24*exp(8)
    
    # 断言：将多项式 p 转换为表达式并检查其在 x 上的系数是否为 24*y*exp(8) + 23
    assert p.as_expr().coeff(x) == 24*y*exp(8) + 23
    # 断言：检查 Poly 对象 p 的 coeff 方法在参数为 x 时是否会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: p.coeff(x))
    
    # 断言：检查 Poly(x + 1) 对象在尝试计算非单项式的系数时是否会抛出 ValueError 异常
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(0))
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(3*x))
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(3*x*y))
# 定义一个测试函数，测试多项式对象的 nth 方法
def test_Poly_nth():
    # 断言对零多项式在给定位置的值为零
    assert Poly(0, x).nth(0) == 0
    assert Poly(0, x).nth(1) == 0

    # 断言对常数多项式在给定位置的值等于常数本身
    assert Poly(1, x).nth(0) == 1
    assert Poly(1, x).nth(1) == 0

    # 断言对 x^8 多项式在指定位置的系数
    assert Poly(x**8, x).nth(0) == 0
    assert Poly(x**8, x).nth(7) == 0
    assert Poly(x**8, x).nth(8) == 1
    assert Poly(x**8, x).nth(9) == 0

    # 断言对于二变量多项式在指定位置的系数
    assert Poly(3*x*y**2 + 1, x, y).nth(0, 0) == 1
    assert Poly(3*x*y**2 + 1, x, y).nth(1, 2) == 3

    # 检验异常情况，验证抛出值错误异常
    raises(ValueError, lambda: Poly(x*y + 1, x, y).nth(1))


# 定义一个测试函数，测试多项式对象的 LM 方法
def test_Poly_LM():
    # 断言零多项式的 LM 为空元组
    assert Poly(0, x).LM() == (0,)
    assert Poly(1, x).LM() == (0,)
    assert Poly(2*x**2 + x, x).LM() == (2,)

    # 断言按字典序 'lex' 和 'grlex' 计算的 LM 值
    assert Poly(x*y**7 + 2*x**2*y**3).LM('lex') == (2, 3)
    assert Poly(x*y**7 + 2*x**2*y**3).LM('grlex') == (1, 7)

    # 断言使用自定义排序函数的 LM 值
    assert LM(x*y**7 + 2*x**2*y**3, order='lex') == x**2*y**3
    assert LM(x*y**7 + 2*x**2*y**3, order='grlex') == x*y**7


# 定义一个测试函数，测试多项式对象的 LM 方法和自定义排序
def test_Poly_LM_custom_order():
    # 定义多项式并使用逆字典序的函数进行 LM 计算
    f = Poly(x**2*y**3*z + x**2*y*z**3 + x*y*z + 1)
    rev_lex = lambda monom: tuple(reversed(monom))

    # 断言按 'lex' 排序计算的 LM 值
    assert f.LM(order='lex') == (2, 3, 1)
    # 断言按自定义排序函数计算的 LM 值
    assert f.LM(order=rev_lex) == (2, 1, 3)


# 定义一个测试函数，测试多项式对象的 EM 方法
def test_Poly_EM():
    # 断言零多项式的 EM 为空元组
    assert Poly(0, x).EM() == (0,)
    assert Poly(1, x).EM() == (0,)
    assert Poly(2*x**2 + x, x).EM() == (1,)

    # 断言按字典序 'lex' 和 'grlex' 计算的 EM 值
    assert Poly(x*y**7 + 2*x**2*y**3).EM('lex') == (1, 7)
    assert Poly(x*y**7 + 2*x**2*y**3).EM('grlex') == (2, 3)


# 定义一个测试函数，测试多项式对象的 LT 方法
def test_Poly_LT():
    # 断言零多项式的 LT 是 ((0,), 0)
    assert Poly(0, x).LT() == ((0,), 0)
    assert Poly(1, x).LT() == ((0,), 1)
    assert Poly(2*x**2 + x, x).LT() == ((2,), 2)

    # 断言按字典序 'lex' 和 'grlex' 计算的 LT 值
    assert Poly(x*y**7 + 2*x**2*y**3).LT('lex') == ((2, 3), 2)
    assert Poly(x*y**7 + 2*x**2*y**3).LT('grlex') == ((1, 7), 1)

    # 断言使用指定排序方式计算 LT 值
    assert LT(x*y**7 + 2*x**2*y**3, order='lex') == 2*x**2*y**3
    assert LT(x*y**7 + 2*x**2*y**3, order='grlex') == x*y**7


# 定义一个测试函数，测试多项式对象的 ET 方法
def test_Poly_ET():
    # 断言零多项式的 ET 是 ((0,), 0)
    assert Poly(0, x).ET() == ((0,), 0)
    assert Poly(1, x).ET() == ((0,), 1)
    assert Poly(2*x**2 + x, x).ET() == ((1,), 1)

    # 断言按字典序 'lex' 和 'grlex' 计算的 ET 值
    assert Poly(x*y**7 + 2*x**2*y**3).ET('lex') == ((1, 7), 1)
    assert Poly(x*y**7 + 2*x**2*y**3).ET('grlex') == ((2, 3), 2)


# 定义一个测试函数，测试多项式对象的 max_norm 方法
def test_Poly_max_norm():
    # 断言对于常数多项式的 max_norm
    assert Poly(-1, x).max_norm() == 1
    assert Poly(0, x).max_norm() == 0
    assert Poly(1, x).max_norm() == 1


# 定义一个测试函数，测试多项式对象的 l1_norm 方法
def test_Poly_l1_norm():
    # 断言对于常数多项式的 l1_norm
    assert Poly(-1, x).l1_norm() == 1
    assert Poly(0, x).l1_norm() == 0
    assert Poly(1, x).l1_norm() == 1


# 定义一个测试函数，测试多项式对象的 clear_denoms 方法
def test_Poly_clear_denoms():
    # 断言对于整数系数的多项式，清除分母后系数为 1
    coeff, poly = Poly(x + 2, x).clear_denoms()
    assert coeff == 1 and poly == Poly(
        x + 2, x, domain='ZZ') and poly.get_domain() == ZZ

    # 断言对于有理数系数的多项式，清除分母后系数为 2
    coeff, poly = Poly(x/2 + 1, x).clear_denoms()
    assert coeff == 2 and poly == Poly(
        x + 2, x, domain='QQ') and poly.get_domain() == QQ

    # 断言对于有理数系数的多项式，且转换为整数系数后，清除分母后系数为 2
    coeff, poly = Poly(x/2 + 1, x).clear_denoms(convert=True)
    assert coeff == 2 and poly == Poly(
        x + 2, x, domain='ZZ') and poly.get_domain() == ZZ

    # 断言对于有理数系数的多项式，且转换为整数系数后，清除分母后系数为 y
    coeff, poly = Poly(x/y + 1, x).clear_denoms(convert=True)
    assert coeff == y and poly == Poly(
        x + y, x, domain='ZZ[y]') and poly.get_domain() == ZZ[y]
    # 将给定表达式 x/3 + sqrt(2) 视作一个多项式，使用符号变量 x，并在表达式域中明确指定为表达式 (EX)。
    # 清除分母，返回系数 coeff 和多项式对象 poly
    coeff, poly = Poly(x/3 + sqrt(2), x, domain='EX').clear_denoms()
    # 使用断言确认计算结果是否符合预期：
    # coeff 应为 3，poly 应为 x + 3*sqrt(2)，并且 poly 对象的域应为 EX
    assert coeff == 3 and poly == Poly(x + 3*sqrt(2), x, domain='EX') and poly.get_domain() == EX
    
    # 将给定表达式 x/3 + sqrt(2) 视作一个多项式，使用符号变量 x，并在表达式域中明确指定为表达式 (EX)。
    # 在清除分母时，使用 convert=True 将分母转换为整数，返回系数 coeff 和多项式对象 poly
    coeff, poly = Poly(x/3 + sqrt(2), x, domain='EX').clear_denoms(convert=True)
    # 使用断言确认计算结果是否符合预期：
    # coeff 应为 3，poly 应为 x + 3*sqrt(2)，并且 poly 对象的域应为 EX
    assert coeff == 3 and poly == Poly(x + 3*sqrt(2), x, domain='EX') and poly.get_domain() == EX
def test_Poly_rat_clear_denoms():
    # 创建多项式 f 和 g，f 是 x^2/y + 1，g 是 x^3 + y 的多项式
    f = Poly(x**2/y + 1, x)
    g = Poly(x**3 + y, x)

    # 使用 f 的 rat_clear_denoms 方法处理 g，返回处理后的两个多项式
    assert f.rat_clear_denoms(g) == \
        (Poly(x**2 + y, x), Poly(y*x**3 + y**2, x))

    # 将 f 和 g 设置为域 EX（表达式域）
    f = f.set_domain(EX)
    g = g.set_domain(EX)

    # 断言处理后的 f 和 g 仍然与原始 f 和 g 相同
    assert f.rat_clear_denoms(g) == (f, g)


def test_issue_20427():
    # 创建包含大数学表达式的多项式 f，并断言其应该等于多项式 0
    f = Poly(-117968192370600*18**(S(1)/3)/(217603955769048*(24201 +
        253*sqrt(9165))**(S(1)/3) + 2273005839412*sqrt(9165)*(24201 +
        253*sqrt(9165))**(S(1)/3)) - 15720318185*2**(S(2)/3)*3**(S(1)/3)*(24201
        + 253*sqrt(9165))**(S(2)/3)/(217603955769048*(24201 + 253*sqrt(9165))**
        (S(1)/3) + 2273005839412*sqrt(9165)*(24201 + 253*sqrt(9165))**(S(1)/3))
        + 15720318185*12**(S(1)/3)*(24201 + 253*sqrt(9165))**(S(2)/3)/(
        217603955769048*(24201 + 253*sqrt(9165))**(S(1)/3) + 2273005839412*
        sqrt(9165)*(24201 + 253*sqrt(9165))**(S(1)/3)) + 117968192370600*2**(
        S(1)/3)*3**(S(2)/3)/(217603955769048*(24201 + 253*sqrt(9165))**(S(1)/3)
        + 2273005839412*sqrt(9165)*(24201 + 253*sqrt(9165))**(S(1)/3)), x)
    assert f == Poly(0, x, domain='EX')


def test_Poly_integrate():
    # 断言多项式的不同积分操作结果
    assert Poly(x + 1).integrate() == Poly(x**2/2 + x)
    assert Poly(x + 1).integrate(x) == Poly(x**2/2 + x)
    assert Poly(x + 1).integrate((x, 1)) == Poly(x**2/2 + x)

    assert Poly(x*y + 1).integrate(x) == Poly(x**2*y/2 + x)
    assert Poly(x*y + 1).integrate(y) == Poly(x*y**2/2 + y)

    assert Poly(x*y + 1).integrate(x, x) == Poly(x**3*y/6 + x**2/2)
    assert Poly(x*y + 1).integrate(y, y) == Poly(x*y**3/6 + y**2/2)

    assert Poly(x*y + 1).integrate((x, 2)) == Poly(x**3*y/6 + x**2/2)
    assert Poly(x*y + 1).integrate((y, 2)) == Poly(x*y**3/6 + y**2/2)

    assert Poly(x*y + 1).integrate(x, y) == Poly(x**2*y**2/4 + x*y)
    assert Poly(x*y + 1).integrate(y, x) == Poly(x**2*y**2/4 + x*y)


def test_Poly_diff():
    # 断言多项式的不同求导操作结果
    assert Poly(x**2 + x).diff() == Poly(2*x + 1)
    assert Poly(x**2 + x).diff(x) == Poly(2*x + 1)
    assert Poly(x**2 + x).diff((x, 1)) == Poly(2*x + 1)

    assert Poly(x**2*y**2 + x*y).diff(x) == Poly(2*x*y**2 + y)
    assert Poly(x**2*y**2 + x*y).diff(y) == Poly(2*x**2*y + x)

    assert Poly(x**2*y**2 + x*y).diff(x, x) == Poly(2*y**2, x, y)
    assert Poly(x**2*y**2 + x*y).diff(y, y) == Poly(2*x**2, x, y)

    assert Poly(x**2*y**2 + x*y).diff((x, 2)) == Poly(2*y**2, x, y)
    assert Poly(x**2*y**2 + x*y).diff((y, 2)) == Poly(2*x**2, x, y)

    assert Poly(x**2*y**2 + x*y).diff(x, y) == Poly(4*x*y + 1)
    assert Poly(x**2*y**2 + x*y).diff(y, x) == Poly(4*x*y + 1)


def test_issue_9585():
    # 断言多项式的不同求导操作结果
    assert diff(Poly(x**2 + x)) == Poly(2*x + 1)
    assert diff(Poly(x**2 + x), x, evaluate=False) == \
        Derivative(Poly(x**2 + x), x)
    assert Derivative(Poly(x**2 + x), x).doit() == Poly(2*x + 1)


def test_Poly_eval():
    # 断言多项式在给定点或点组合处的求值结果
    assert Poly(0, x).eval(7) == 0
    assert Poly(1, x).eval(7) == 1
    assert Poly(x, x).eval(7) == 7

    assert Poly(0, x).eval(0, 7) == 0
    assert Poly(1, x).eval(0, 7) == 1
    # 检查多项式对象在 x = 0 时的求值是否为 7
    assert Poly(x, x).eval(0, 7) == 7

    # 检查多项式对象为常数 0 时在 x = 7 时的求值是否为 0
    assert Poly(0, x).eval(x, 7) == 0
    # 检查多项式对象为常数 1 时在 x = 7 时的求值是否为 1
    assert Poly(1, x).eval(x, 7) == 1
    # 检查多项式对象为 x 时在 x = 7 时的求值是否为 7
    assert Poly(x, x).eval(x, 7) == 7

    # 检查多项式对象为常数 0 时在 x = 'x' 时的求值是否为 0
    assert Poly(0, x).eval('x', 7) == 0
    # 检查多项式对象为常数 1 时在 x = 'x' 时的求值是否为 1
    assert Poly(1, x).eval('x', 7) == 1
    # 检查多项式对象为 x 时在 x = 'x' 时的求值是否为 7
    assert Poly(x, x).eval('x', 7) == 7

    # 检查多项式对象为常数 1 时在非 x 参数（1）上的求值是否引发多项式错误
    raises(PolynomialError, lambda: Poly(1, x).eval(1, 7))
    # 检查多项式对象为常数 1 时在未定义的变量 y 上的求值是否引发多项式错误
    raises(PolynomialError, lambda: Poly(1, x).eval(y, 7))
    # 检查多项式对象为常数 1 时在字符串 'y' 上的求值是否引发多项式错误
    raises(PolynomialError, lambda: Poly(1, x).eval('y', 7))

    # 检查多项式对象为常数 123 * x^0 * y^0 时在 x = 7 时的求值是否为常数 123 * y^0
    assert Poly(123, x, y).eval(7) == Poly(123, y)
    # 检查多项式对象为常数 2 * y * x^0 * y^1 时在 x = 7 时的求值是否为常数 2 * y * y^1
    assert Poly(2*y, x, y).eval(7) == Poly(2*y, y)
    # 检查多项式对象为 x * y * x^1 * y^1 时在 x = 7 时的求值是否为常数 7 * y * y^1
    assert Poly(x*y, x, y).eval(7) == Poly(7*y, y)

    # 检查多项式对象为常数 123 * x^0 * y^0 时在 x = 7 时的求值是否为常数 123 * y^0
    assert Poly(123, x, y).eval(x, 7) == Poly(123, y)
    # 检查多项式对象为常数 2 * y * x^0 * y^1 时在 x = 7 时的求值是否为常数 2 * y * y^1
    assert Poly(2*y, x, y).eval(x, 7) == Poly(2*y, y)
    # 检查多项式对象为 x * y * x^1 * y^1 时在 x = 7 时的求值是否为常数 7 * y * y^1
    assert Poly(x*y, x, y).eval(x, 7) == Poly(7*y, y)

    # 检查多项式对象为常数 123 * x^0 * y^0 时在 y = 7 时的求值是否为常数 123 * x^0
    assert Poly(123, x, y).eval(y, 7) == Poly(123, x)
    # 检查多项式对象为常数 2 * y * x^0 * y^1 时在 y = 7 时的求值是否为常数 14 * x^0
    assert Poly(2*y, x, y).eval(y, 7) == Poly(14, x)
    # 检查多项式对象为 x * y * x^1 * y^1 时在 y = 7 时的求值是否为常数 7 * x * x^0
    assert Poly(x*y, x, y).eval(y, 7) == Poly(7*x, x)

    # 检查多项式对象为 x * y + y 时在 x = 7 时的求值是否为常数 8 * y
    assert Poly(x*y + y, x, y).eval({x: 7}) == Poly(8*y, y)
    # 检查多项式对象为 x * y + y 时在 y = 7 时的求值是否为常数 7 * x + 7
    assert Poly(x*y + y, x, y).eval({y: 7}) == Poly(7*x + 7, x)

    # 检查多项式对象为 x * y + y 时在 x = 6, y = 7 时的求值是否为 49
    assert Poly(x*y + y, x, y).eval({x: 6, y: 7}) == 49
    # 检查多项式对象为 x * y + y 时在 x = 7, y = 6 时的求值是否为 48
    assert Poly(x*y + y, x, y).eval({x: 7, y: 6}) == 48

    # 检查多项式对象为 x * y + y 时在输入为元组 (6, 7) 时的求值是否为 49
    assert Poly(x*y + y, x, y).eval((6, 7)) == 49
    # 检查多项式对象为 x * y + y 时在输入为列表 [6, 7] 时的求值是否为 49
    assert Poly(x*y + y, x, y).eval([6, 7]) == 49

    # 检查多项式对象为 x + 1 在整数环 'ZZ' 上，求值在 S.Half 时是否为有理数 3/2
    assert Poly(x + 1, domain='ZZ').eval(S.Half) == Rational(3, 2)
    # 检查多项式对象为 x + 1 在整数环 'ZZ' 上，求值在 sqrt(2) 时是否为 sqrt(2) + 1
    assert Poly(x + 1, domain='ZZ').eval(sqrt(2)) == sqrt(2) + 1

    # 检查多项式对象为 x * y + y 时在输入为元组 (6, 7, 8) 时是否引发值错误
    raises(ValueError, lambda: Poly(x*y + y, x, y).eval((6, 7, 8)))
    # 检查多项式对象为 x + 1 在整数环 'ZZ' 上，求值在 S.Half 时是否引发域错误
    raises(DomainError, lambda: Poly(x + 1, domain='ZZ').eval(S.Half, auto=False))

    # 问题 6344：创建符号 'alpha'
    alpha = Symbol('alpha')
    # 计算多项式结果 (2*alpha*z - 2*alpha + z**2 + 3)/(z**2 - 2*z + 1)
    result = (2*alpha*z - 2*alpha + z**2 + 3)/(z**2 - 2*z + 1)

    # 创建在整数环 'ZZ[alpha]' 上的多项式对象，并检查其在 (z + 1)/(z - 1) 时的求值是否为 result
    f = Poly(x**2 + (alpha - 1)*x - alpha + 1, x, domain='ZZ[alpha]')
    assert f.eval((z + 1)/(z - 1)) == result

    # 创建在整数环 'ZZ[alpha,z]' 上的多项式对象 g，并检查其在 (z + 1)/(z - 1) 时的求值是否为多项式对象 result
    g = Poly(x**2 + (alpha - 1
# 定义一个测试函数，用于测试 Poly 类的 __call__ 方法
def test_Poly___call__():
    # 创建一个 Poly 对象 f，其表达式为 2*x*y + 3*x + y + 2*z
    f = Poly(2*x*y + 3*x + y + 2*z)

    # 断言 f(2) 的结果与 Poly(5*y + 2*z + 6) 相等
    assert f(2) == Poly(5*y + 2*z + 6)
    # 断言 f(2, 5) 的结果与 Poly(2*z + 31) 相等
    assert f(2, 5) == Poly(2*z + 31)
    # 断言 f(2, 5, 7) 的结果为 45
    assert f(2, 5, 7) == 45


# 定义一个测试函数，用于测试 parallel_poly_from_expr 函数
def test_parallel_poly_from_expr():
    # 断言将 [x - 1, x**2 - 1] 转化为 Poly 对象列表时的结果
    assert parallel_poly_from_expr([x - 1, x**2 - 1], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([Poly(x - 1, x), x**2 - 1], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([x - 1, Poly(x**2 - 1, x)], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([Poly(x - 1, x), Poly(x**2 - 1, x)], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]

    # 断言将 [x - 1, x**2 - 1] 转化为 Poly 对象列表时的结果，带有额外的变量 y
    assert parallel_poly_from_expr([x - 1, x**2 - 1], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]
    assert parallel_poly_from_expr([Poly(x - 1, x), x**2 - 1], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]
    assert parallel_poly_from_expr([x - 1, Poly(x**2 - 1, x)], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]
    assert parallel_poly_from_expr([Poly(x - 1, x), Poly(x**2 - 1, x)], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]

    # 断言将 [x - 1, x**2 - 1] 转化为 Poly 对象列表时的结果，自动推断变量 x
    assert parallel_poly_from_expr([x - 1, x**2 - 1])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([Poly(x - 1, x), x**2 - 1])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([x - 1, Poly(x**2 - 1, x)])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([Poly(x - 1, x), Poly(x**2 - 1, x)])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]

    # 断言将 [1, x**2 - 1] 转化为 Poly 对象列表时的结果
    assert parallel_poly_from_expr([1, x**2 - 1])[0] == [Poly(1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([1, Poly(x**2 - 1, x)])[0] == [Poly(1, x), Poly(x**2 - 1, x)]

    # 断言将 [x**2 - 1, 1] 转化为 Poly 对象列表时的结果
    assert parallel_poly_from_expr([x**2 - 1, 1])[0] == [Poly(x**2 - 1, x), Poly(1, x)]
    assert parallel_poly_from_expr([Poly(x**2 - 1, x), 1])[0] == [Poly(x**2 - 1, x), Poly(1, x)]

    # 断言将 [Poly(x, x, y), Poly(y, x, y)] 转化为 Poly 对象列表时的结果，指定 lex 排序
    assert parallel_poly_from_expr([Poly(x, x, y), Poly(y, x, y)], x, y, order='lex')[0] == \
        [Poly(x, x, y, domain='ZZ'), Poly(y, x, y, domain='ZZ')]

    # 断言当传入 [0, 1] 时，会抛出 PolificationFailed 异常
    raises(PolificationFailed, lambda: parallel_poly_from_expr([0, 1]))


# 定义一个测试函数，用于测试 pdiv 和 prem 函数
def test_pdiv():
    # 定义多项式 f = x**2 - y**2 和 g = x - y
    f, g = x**2 - y**2, x - y
    # 定义商 q = x + y 和余数 r = 0
    q, r = x + y, 0

    # 将 f、g、q、r 转化为 Poly 对象
    F, G, Q, R = [Poly(h, x, y) for h in (f, g, q, r)]

    # 断言使用 Poly 对象的 pdiv 方法求商和余数的结果
    assert F.pdiv(G) == (Q, R)
    assert F.prem(G) == R
    assert F.pquo(G) == Q
    assert F.pexquo(G) == Q

    # 断言使用顶层函数 pdiv 和 prem 求商和余数的结果
    assert pdiv(f, g) == (q, r)
    assert prem(f, g) == r
    # 断言：调用函数 pquo(f, g)，断言其返回值等于 q
    assert pquo(f, g) == q
    # 断言：调用函数 pexquo(f, g)，断言其返回值等于 q
    assert pexquo(f, g) == q
    
    # 断言：调用函数 pdiv(f, g, x, y)，断言其返回值等于 (q, r)
    assert pdiv(f, g, x, y) == (q, r)
    # 断言：调用函数 prem(f, g, x, y)，断言其返回值等于 r
    assert prem(f, g, x, y) == r
    # 断言：调用函数 pquo(f, g, x, y)，断言其返回值等于 q
    assert pquo(f, g, x, y) == q
    # 断言：调用函数 pexquo(f, g, x, y)，断言其返回值等于 q
    assert pexquo(f, g, x, y) == q
    
    # 断言：调用函数 pdiv(f, g, (x, y))，断言其返回值等于 (q, r)
    assert pdiv(f, g, (x, y)) == (q, r)
    # 断言：调用函数 prem(f, g, (x, y))，断言其返回值等于 r
    assert prem(f, g, (x, y)) == r
    # 断言：调用函数 pquo(f, g, (x, y))，断言其返回值等于 q
    assert pquo(f, g, (x, y)) == q
    # 断言：调用函数 pexquo(f, g, (x, y))，断言其返回值等于 q
    assert pexquo(f, g, (x, y)) == q
    
    # 断言：调用函数 pdiv(F, G)，断言其返回值等于 (Q, R)
    assert pdiv(F, G) == (Q, R)
    # 断言：调用函数 prem(F, G)，断言其返回值等于 R
    assert prem(F, G) == R
    # 断言：调用函数 pquo(F, G)，断言其返回值等于 Q
    assert pquo(F, G) == Q
    # 断言：调用函数 pexquo(F, G)，断言其返回值等于 Q
    assert pexquo(F, G) == Q
    
    # 断言：调用函数 pdiv(f, g, polys=True)，断言其返回值等于 (Q, R)
    assert pdiv(f, g, polys=True) == (Q, R)
    # 断言：调用函数 prem(f, g, polys=True)，断言其返回值等于 R
    assert prem(f, g, polys=True) == R
    # 断言：调用函数 pquo(f, g, polys=True)，断言其返回值等于 Q
    assert pquo(f, g, polys=True) == Q
    # 断言：调用函数 pexquo(f, g, polys=True)，断言其返回值等于 Q
    assert pexquo(f, g, polys=True) == Q
    
    # 断言：调用函数 pdiv(F, G, polys=False)，断言其返回值等于 (q, r)
    assert pdiv(F, G, polys=False) == (q, r)
    # 断言：调用函数 prem(F, G, polys=False)，断言其返回值等于 r
    assert prem(F, G, polys=False) == r
    # 断言：调用函数 pquo(F, G, polys=False)，断言其返回值等于 q
    assert pquo(F, G, polys=False) == q
    # 断言：调用函数 pexquo(F, G, polys=False)，断言其返回值等于 q
    assert pexquo(F, G, polys=False) == q
    
    # 断言：调用函数 raises(ComputationFailed, lambda: pdiv(4, 2))，验证抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: pdiv(4, 2))
    # 断言：调用函数 raises(ComputationFailed, lambda: prem(4, 2))，验证抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: prem(4, 2))
    # 断言：调用函数 raises(ComputationFailed, lambda: pquo(4, 2))，验证抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: pquo(4, 2))
    # 断言：调用函数 raises(ComputationFailed, lambda: pexquo(4, 2))，验证抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: pexquo(4, 2))
# 定义一个函数用于测试多项式的除法操作
def test_div():
    # 定义多项式 f 和 g
    f, g = x**2 - y**2, x - y
    # 定义商和余数的初值
    q, r = x + y, 0

    # 创建多项式对象 F, G, Q, R 分别对应 f, g, q, r 的多项式表示
    F, G, Q, R = [ Poly(h, x, y) for h in (f, g, q, r) ]

    # 断言多项式 F 除以 G 的结果为商 Q 和余数 R
    assert F.div(G) == (Q, R)
    # 断言多项式 F 除以 G 的余数为 R
    assert F.rem(G) == R
    # 断言多项式 F 除以 G 的商为 Q
    assert F.quo(G) == Q
    # 断言多项式 F 除以 G 的精确商为 Q
    assert F.exquo(G) == Q

    # 断言函数 div 返回多项式 f 除以 g 的结果为商 q 和余数 r
    assert div(f, g) == (q, r)
    # 断言函数 rem 返回多项式 f 除以 g 的余数为 r
    assert rem(f, g) == r
    # 断言函数 quo 返回多项式 f 除以 g 的商为 q
    assert quo(f, g) == q
    # 断言函数 exquo 返回多项式 f 除以 g 的精确商为 q
    assert exquo(f, g) == q

    # 断言带有变量参数的 div 函数返回多项式 f 除以 g 的结果为商 q 和余数 r
    assert div(f, g, x, y) == (q, r)
    # 断言带有变量参数的 rem 函数返回多项式 f 除以 g 的余数为 r
    assert rem(f, g, x, y) == r
    # 断言带有变量参数的 quo 函数返回多项式 f 除以 g 的商为 q
    assert quo(f, g, x, y) == q
    # 断言带有变量参数的 exquo 函数返回多项式 f 除以 g 的精确商为 q
    assert exquo(f, g, x, y) == q

    # 断言带有元组参数的 div 函数返回多项式 f 除以 g 的结果为商 q 和余数 r
    assert div(f, g, (x, y)) == (q, r)
    # 断言带有元组参数的 rem 函数返回多项式 f 除以 g 的余数为 r
    assert rem(f, g, (x, y)) == r
    # 断言带有元组参数的 quo 函数返回多项式 f 除以 g 的商为 q
    assert quo(f, g, (x, y)) == q
    # 断言带有元组参数的 exquo 函数返回多项式 f 除以 g 的精确商为 q
    assert exquo(f, g, (x, y)) == q

    # 断言多项式对象 F 除以 G 的结果为商 Q 和余数 R
    assert div(F, G) == (Q, R)
    # 断言多项式对象 F 除以 G 的余数为 R
    assert rem(F, G) == R
    # 断言多项式对象 F 除以 G 的商为 Q
    assert quo(F, G) == Q
    # 断言多项式对象 F 除以 G 的精确商为 Q
    assert exquo(F, G) == Q

    # 断言带有 polys=True 参数的 div 函数返回多项式 f 除以 g 的结果为商 Q 和余数 R
    assert div(f, g, polys=True) == (Q, R)
    # 断言带有 polys=True 参数的 rem 函数返回多项式 f 除以 g 的余数为 R
    assert rem(f, g, polys=True) == R
    # 断言带有 polys=True 参数的 quo 函数返回多项式 f 除以 g 的商为 Q
    assert quo(f, g, polys=True) == Q
    # 断言带有 polys=True 参数的 exquo 函数返回多项式 f 除以 g 的精确商为 Q
    assert exquo(f, g, polys=True) == Q

    # 断言带有 polys=False 参数的 div 函数返回多项式对象 F 除以 G 的结果为商 q 和余数 r
    assert div(F, G, polys=False) == (q, r)
    # 断言带有 polys=False 参数的 rem 函数返回多项式对象 F 除以 G 的余数为 r
    assert rem(F, G, polys=False) == r
    # 断言带有 polys=False 参数的 quo 函数返回多项式对象 F 除以 G 的商为 q
    assert quo(F, G, polys=False) == q
    # 断言带有 polys=False 参数的 exquo 函数返回多项式对象 F 除以 G 的精确商为 q
    assert exquo(F, G, polys=False) == q

    # 断言调用 div 函数时抛出 ComputationFailed 异常，lambda 表达式输入参数为 4 和 2
    raises(ComputationFailed, lambda: div(4, 2))
    # 断言调用 rem 函数时抛出 ComputationFailed 异常，lambda 表达式输入参数为 4 和 2
    raises(ComputationFailed, lambda: rem(4, 2))
    # 断言调用 quo 函数时抛出 ComputationFailed 异常，lambda 表达式输入参数为 4 和 2
    raises(ComputationFailed, lambda: quo(4, 2))
    # 断言调用 exquo 函数时抛出 ComputationFailed 异常，lambda 表达式输入参数为 4 和 2
    raises(ComputationFailed, lambda: exquo(4, 2))

    # 重新定义多项式 f 和 g
    f, g = x**2 + 1, 2*x - 4

    # 定义期望的商和余数值
    qz, rz = 0, x**2 + 1
    qq, rq = x/2 + 1, 5

    # 断言函数 div 返回多项式 f 除以 g 的结果为商 qq 和余数 rq
    assert div(f, g) == (qq, rq)
    # 断言函数 div 使用 auto=True 时返回多项式 f 除以 g 的结果为商 qq 和余数 rq
    assert div(f, g, auto=True) == (qq, rq)
    # 断言函数 div 使用 auto=False 时返回多项式 f 除以 g 的结果为商 qz 和余数 rz
    assert div(f, g, auto=False) == (qz, rz)
    # 断言函数 div 使用 domain=ZZ 时返回多项式 f 除以 g 的结果为商 qz 和余数 rz
    assert div(f, g, domain=ZZ) == (qz, rz)
    # 断言函数 div 使用 domain=QQ 时返回多项式 f 除以 g 的结果为商 qq 和余数 rq
    assert div(f, g, domain=QQ) == (qq, rq)
    # 断言函数 div 使用 domain=ZZ 和 auto=True 时返回多项式 f 除以 g 的结果为商 qq 和余数 rq
    assert div(f, g, domain=ZZ, auto=True) == (qq, rq)
    # 断言函数 div 使用 domain=ZZ 和 auto=False 时返回多项式 f 除以 g 的结果为商 qz 和余数 rz
    assert div(f, g, domain=ZZ, auto=False) == (qz, rz)
    # 断言函数 div 使用 domain=QQ 和 auto=True 时返回多项式 f 除以 g 的结果为商 qq 和余数 rq
    assert div(f, g, domain=QQ, auto=True) == (qq, rq)
    # 断言函数 div 使用 domain=QQ 和 auto=False 时返回多项式 f 除以 g 的结果为商 qq 和余数 rq
    assert div(f, g, domain=QQ, auto=False) == (qq, rq)

    # 断言函数 rem 返回多项式 f 除以 g 的余数为 rq
    assert rem(f, g) == rq
    # 断
    # 断言：使用 exquo 函数计算 f 除以 g 在有理数域 QQ 上的商等于 q
    assert exquo(f, g, domain=QQ, auto=False) == q

    # 创建多项式 f = x^2 和 g = x
    f, g = Poly(x**2), Poly(x)

    # 使用 div 函数计算 f 除以 g 的商和余数
    q, r = f.div(g)
    # 断言：q 和 r 的定义域为整数域 ZZ
    assert q.get_domain().is_ZZ and r.get_domain().is_ZZ
    
    # 计算 f 除以 g 的余数
    r = f.rem(g)
    # 断言：r 的定义域为整数域 ZZ
    assert r.get_domain().is_ZZ
    
    # 计算 f 除以 g 的商
    q = f.quo(g)
    # 断言：q 的定义域为整数域 ZZ
    assert q.get_domain().is_ZZ
    
    # 使用 exquo 函数计算 f 除以 g 在扩展有理数域上的商
    q = f.exquo(g)
    # 断言：q 的定义域为整数域 ZZ
    assert q.get_domain().is_ZZ

    # 创建多项式 f = x + y 和 g = 2*x + y
    f, g = Poly(x+y, x), Poly(2*x+y, x)
    # 使用 div 函数计算 f 除以 g 的商和余数
    q, r = f.div(g)
    # 断言：q 的定义域为分数域 Frac，r 的定义域为分数域 Frac
    assert q.get_domain().is_Frac and r.get_domain().is_Frac

    # 创建多项式 p = 2 + 3*I 和 q = 1 - I，定义域为整数域扩展 ZZ_I
    p = Poly(2+3*I, x, domain=ZZ_I)
    q = Poly(1-I, x, domain=ZZ_I)
    # 断言：使用 div 函数计算 p 除以 q 的结果
    assert p.div(q, auto=False) == \
        (Poly(0, x, domain='ZZ_I'), Poly(2 + 3*I, x, domain='ZZ_I'))
    # 断言：使用 div 函数计算 p 除以 q 的结果，自动转换为有理数域 QQ_I
    assert p.div(q, auto=True) == \
        (Poly(-S(1)/2 + 5*I/2, x, domain='QQ_I'), Poly(0, x, domain='QQ_I'))

    # 创建多项式 f = 5*x^2 + 10*x + 3 和 g = 2*x + 2
    f = 5*x**2 + 10*x + 3
    g = 2*x + 2
    # 断言：使用 div 函数计算 f 除以 g 在整数域 ZZ 上的商和余数
    assert div(f, g, domain=ZZ) == (0, f)
# 测试函数，用于验证 div 函数的行为是否符合预期
def test_issue_7864():
    # 使用 div 函数计算两个值的商和余数
    q, r = div(a, .408248290463863*a)
    # 断言商的绝对值与预期值的差小于指定精度
    assert abs(q - 2.44948974278318) < 1e-14
    # 断言余数为 0
    assert r == 0


# 测试 gcdex 相关函数
def test_gcdex():
    # 定义多项式 f 和 g
    f, g = 2*x, x**2 - 16
    # 定义 s, t, h
    s, t, h = x/32, Rational(-1, 16), 1

    # 将 f, g, s, t, h 转换为有理数多项式对象
    F, G, S, T, H = [ Poly(u, x, domain='QQ') for u in (f, g, s, t, h) ]

    # 断言 half_gcdex 函数的返回结果符合预期
    assert F.half_gcdex(G) == (S, H)
    assert F.gcdex(G) == (S, T, H)
    assert F.invert(G) == S

    # 断言全局作用域下的函数调用结果
    assert half_gcdex(f, g) == (s, h)
    assert gcdex(f, g) == (s, t, h)
    assert invert(f, g) == s

    # 断言指定 x 作为符号对象的函数调用结果
    assert half_gcdex(f, g, x) == (s, h)
    assert gcdex(f, g, x) == (s, t, h)
    assert invert(f, g, x) == s

    # 断言使用元组 (x,) 作为符号对象的函数调用结果
    assert half_gcdex(f, g, (x,)) == (s, h)
    assert gcdex(f, g, (x,)) == (s, t, h)
    assert invert(f, g, (x,)) == s

    # 断言多项式对象作为参数的函数调用结果
    assert half_gcdex(F, G) == (S, H)
    assert gcdex(F, G) == (S, T, H)
    assert invert(F, G) == S

    # 断言使用 polys=True 参数的函数调用结果
    assert half_gcdex(f, g, polys=True) == (S, H)
    assert gcdex(f, g, polys=True) == (S, T, H)
    assert invert(f, g, polys=True) == S

    # 断言使用 polys=False 参数的函数调用结果
    assert half_gcdex(F, G, polys=False) == (s, h)
    assert gcdex(F, G, polys=False) == (s, t, h)
    assert invert(F, G, polys=False) == s

    # 断言基本整数的函数调用结果
    assert half_gcdex(100, 2004) == (-20, 4)
    assert gcdex(100, 2004) == (-20, 1, 4)
    assert invert(3, 7) == 5

    # 断言 DomainError 异常被正确引发
    raises(DomainError, lambda: half_gcdex(x + 1, 2*x + 1, auto=False))
    raises(DomainError, lambda: gcdex(x + 1, 2*x + 1, auto=False))
    raises(DomainError, lambda: invert(x + 1, 2*x + 1, auto=False))


# 测试 Poly 对象的 revert 方法
def test_revert():
    # 定义多项式 f
    f = Poly(1 - x**2/2 + x**4/24 - x**6/720)
    # 定义多项式 g
    g = Poly(61*x**6/720 + 5*x**4/24 + x**2/2 + 1)

    # 断言 revert 方法的返回结果符合预期
    assert f.revert(8) == g


# 测试 subresultants 相关函数
def test_subresultants():
    # 定义多项式 f, g, h
    f, g, h = x**2 - 2*x + 1, x**2 - 1, 2*x - 2
    # 将 f, g, h 转换为多项式对象
    F, G, H = Poly(f), Poly(g), Poly(h)

    # 断言 subresultants 函数的返回结果符合预期
    assert F.subresultants(G) == [F, G, H]
    assert subresultants(f, g) == [f, g, h]
    assert subresultants(f, g, x) == [f, g, h]
    assert subresultants(f, g, (x,)) == [f, g, h]
    assert subresultants(F, G) == [F, G, H]
    assert subresultants(f, g, polys=True) == [F, G, H]
    assert subresultants(F, G, polys=False) == [f, g, h]

    # 断言 ComputationFailed 异常被正确引发
    raises(ComputationFailed, lambda: subresultants(4, 2))


# 测试 resultant 相关函数
def test_resultant():
    # 定义多项式 f, g, h
    f, g, h = x**2 - 2*x + 1, x**2 - 1, 0
    # 将 f, g 转换为多项式对象
    F, G = Poly(f), Poly(g)

    # 断言 resultant 函数的返回结果符合预期
    assert F.resultant(G) == h
    assert resultant(f, g) == h
    assert resultant(f, g, x) == h
    assert resultant(f, g, (x,)) == h
    assert resultant(F, G) == h
    assert resultant(f, g, polys=True) == h
    assert resultant(F, G, polys=False) == h
    assert resultant(f, g, includePRS=True) == (h, [f, g, 2*x - 2])

    # 定义另外一组多项式 f, g, h
    f, g, h = x - a, x - b, a - b
    # 将 f, g, h 转换为多项式对象
    F, G, H = Poly(f), Poly(g), Poly(h)

    # 断言 resultant 函数的返回结果符合预期
    assert F.resultant(G) == H
    assert resultant(f, g) == h
    assert resultant(f, g, x) == h
    assert resultant(f, g, (x,)) == h
    assert resultant(F, G) == H
    assert resultant(f, g, polys=True) == H
    assert resultant(F, G, polys=False) == h

    # 断言 ComputationFailed 异常被正确引发
    raises(ComputationFailed, lambda: resultant(4, 2))


# 测试 discriminant 相关函数
def test_discriminant():
    # 定义多项式 f, g
    f, g = x**3 + 3*x**2 + 9*x - 13, -11664
    # 将 f 转换为多项式对象
    F = Poly(f)

    # 此处省略 assert 语句，因为测试函数未完
    # 断言语句：检查多项式 F 的判别式是否等于 g
    assert F.discriminant() == g
    # 断言语句：检查函数 discriminant 计算的 f 的判别式是否等于 g
    assert discriminant(f) == g
    # 断言语句：检查函数 discriminant 在给定 x 的情况下计算的 f 的判别式是否等于 g
    assert discriminant(f, x) == g
    # 断言语句：检查函数 discriminant 在给定 (x,) 的情况下计算的 f 的判别式是否等于 g
    assert discriminant(f, (x,)) == g
    # 断言语句：检查函数 discriminant 计算的多项式 F 的判别式是否等于 g
    assert discriminant(F) == g
    # 断言语句：检查函数 discriminant 在 polys=True 时计算的 f 的判别式是否等于 g
    assert discriminant(f, polys=True) == g
    # 断言语句：检查函数 discriminant 在 polys=False 时计算的多项式 F 的判别式是否等于 g
    assert discriminant(F, polys=False) == g

    # 创建多项式 f 和其判别式 g
    f, g = a*x**2 + b*x + c, b**2 - 4*a*c
    # 将 f 和 g 转化为多项式对象 F 和 G
    F, G = Poly(f), Poly(g)

    # 断言语句：检查多项式 F 的判别式是否等于 G
    assert F.discriminant() == G
    # 断言语句：检查函数 discriminant 计算的 f 的判别式是否等于 g
    assert discriminant(f) == g
    # 断言语句：检查函数 discriminant 在给定 x, a, b, c 的情况下计算的 f 的判别式是否等于 g
    assert discriminant(f, x, a, b, c) == g
    # 断言语句：检查函数 discriminant 在给定 (x, a, b, c) 的情况下计算的 f 的判别式是否等于 g
    assert discriminant(f, (x, a, b, c)) == g
    # 断言语句：检查函数 discriminant 计算的多项式 F 的判别式是否等于 G
    assert discriminant(F) == G
    # 断言语句：检查函数 discriminant 在 polys=True 时计算的 f 的判别式是否等于 G
    assert discriminant(f, polys=True) == G
    # 断言语句：检查函数 discriminant 在 polys=False 时计算的多项式 F 的判别式是否等于 g
    assert discriminant(F, polys=False) == g

    # 断言语句：检查调用 discriminant 函数时是否抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: discriminant(4))
def test_dispersion():
    # We test only the API here. For more mathematical
    # tests see the dedicated test file.
    
    # 创建一个多项式对象 fp，表示 (x + 1)*(x + 2)
    fp = poly((x + 1)*(x + 2), x)
    # 断言 fp 的分散集排序后应为 [0, 1]
    assert sorted(fp.dispersionset()) == [0, 1]
    # 断言 fp 的分散指数应为 1
    assert fp.dispersion() == 1

    # 创建一个多项式对象 fp，表示 x**4 - 3*x**2 + 1
    fp = poly(x**4 - 3*x**2 + 1, x)
    # 创建一个多项式对象 gp，表示 fp 向左平移 -3
    gp = fp.shift(-3)
    # 断言 fp 和 gp 的共同分散集排序后应为 [2, 3, 4]
    assert sorted(fp.dispersionset(gp)) == [2, 3, 4]
    # 断言 fp 相对于 gp 的分散指数应为 4
    assert fp.dispersion(gp) == 4


def test_gcd_list():
    # 创建一个多项式列表 F
    F = [x**3 - 1, x**2 - 1, x**2 - 3*x + 2]

    # 断言多项式列表 F 的最大公因子应为 x - 1
    assert gcd_list(F) == x - 1
    # 断言多项式列表 F 的最大公因子，返回为多项式对象 Poly(x - 1)
    assert gcd_list(F, polys=True) == Poly(x - 1)

    # 断言空列表的最大公因子应为 0
    assert gcd_list([]) == 0
    # 断言数字列表 [1, 2] 的最大公因子应为 1
    assert gcd_list([1, 2]) == 1
    # 断言数字列表 [4, 6, 8] 的最大公因子应为 2
    assert gcd_list([4, 6, 8]) == 2

    # 断言包含单个多项式表达式的列表的最大公因子应为 0
    assert gcd_list([x*(y + 42) - x*y - x*42]) == 0

    # 断言空列表的最大公因子（带 x 变量）为 S.Zero
    gcd = gcd_list([], x)
    assert gcd.is_Number and gcd is S.Zero

    # 断言空列表的最大公因子（带 x 变量，返回多项式对象）为 Poly(0, x)
    gcd = gcd_list([], x, polys=True)
    assert gcd.is_Poly and gcd.is_zero

    # 断言包含 sqrt(2) 和 -sqrt(2) 的列表的最大公因子应为 sqrt(2)
    a = sqrt(2)
    assert gcd_list([a, -a]) == gcd_list([-a, a]) == a

    # 断言调用抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: gcd_list([], polys=True))


def test_lcm_list():
    # 创建一个多项式列表 F
    F = [x**3 - 1, x**2 - 1, x**2 - 3*x + 2]

    # 断言多项式列表 F 的最小公倍数应为 x**5 - x**4 - 2*x**3 - x**2 + x + 2
    assert lcm_list(F) == x**5 - x**4 - 2*x**3 - x**2 + x + 2
    # 断言多项式列表 F 的最小公倍数，返回为多项式对象 Poly(x**5 - x**4 - 2*x**3 - x**2 + x + 2)
    assert lcm_list(F, polys=True) == Poly(x**5 - x**4 - 2*x**3 - x**2 + x + 2)

    # 断言空列表的最小公倍数应为 1
    assert lcm_list([]) == 1
    # 断言数字列表 [1, 2] 的最小公倍数应为 2
    assert lcm_list([1, 2]) == 2
    # 断言数字列表 [4, 6, 8] 的最小公倍数应为 24
    assert lcm_list([4, 6, 8]) == 24

    # 断言包含单个多项式表达式的列表的最小公倍数应为 0
    assert lcm_list([x*(y + 42) - x*y - x*42]) == 0

    # 断言空列表的最小公倍数（带 x 变量）为 S.One
    lcm = lcm_list([], x)
    assert lcm.is_Number and lcm is S.One

    # 断言空列表的最小公倍数（带 x 变量，返回多项式对象）为 Poly(1, x)
    lcm = lcm_list([], x, polys=True)
    assert lcm.is_Poly and lcm.is_one

    # 断言调用抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: lcm_list([], polys=True))


def test_gcd():
    # 定义多项式 f 和 g
    f, g = x**3 - 1, x**2 - 1
    s, t = x**2 + x + 1, x + 1
    h, r = x - 1, x**4 + x**3 - x - 1

    # 创建多项式对象 F, G, S, T, H, R
    F, G, S, T, H, R = [ Poly(u) for u in (f, g, s, t, h, r) ]

    # 断言多项式 F 与 G 的因式分解应为 (H, S, T)
    assert F.cofactors(G) == (H, S, T)
    # 断言多项式 F 与 G 的最大公因子应为 H
    assert F.gcd(G) == H
    # 断言多项式 F 与 G 的最小公倍数应为 R
    assert F.lcm(G) == R

    # 断言 f 和 g 的因式分解应为 (h, s, t)
    assert cofactors(f, g) == (h, s, t)
    # 断言 f 和 g 的最大公因子应为 h
    assert gcd(f, g) == h
    # 断言 f 和 g 的最小公倍数应为 r
    assert lcm(f, g) == r

    # 断言 f 和 g 的因式分解（带 x 变量）应为 (h, s, t)
    assert cofactors(f, g, x) == (h, s, t)
    # 断言 f 和 g 的最大公因子（带 x 变量）应为 h
    assert gcd(f, g, x) == h
    # 断言 f 和 g 的最小公倍数（带 x 变量）应为 r
    assert lcm(f, g, x) == r

    # 断言 f 和 g 的因式分解（带 (x,) 变量）应为 (h, s, t)
    assert cofactors(f, g, (x,)) == (h, s, t)
    # 断言 f 和 g 的最大公因子（带 (x,) 变量）应为 h
    assert gcd(f, g, (x,)) == h
    # 断言 f 和 g 的最小公倍数（带 (x,) 变量）应为 r
    assert lcm(f, g, (x,)) == r

    # 断言多项式 F 和 G 的因式分解应为 (H, S, T)
    assert cofactors(F, G) == (H, S, T)
    # 断言多项式 F 和 G 的最大公因子应为 H
    assert gcd(F, G) == H
    # 断言多项式 F 和 G 的最小公倍数应为 R
    assert lcm(F, G) == R

    # 断言 f 和 g 的因式分解（带 polys=True）应为 (H, S, T)
    assert cofactors(f, g, polys=True) == (H, S, T)
    # 断言 f 和 g 的最大公因子（带 polys=True）应为 H
    assert gcd(f, g, polys=True) == H
    # 断言 f 和 g 的最小公倍
    # 初始化变量 h, s, t 分别为 x - 4, x + 1, x^2 + 1
    h, s, t = x - 4, x + 1, x**2 + 1

    # 使用 cofactors 函数检查 f 和 g 在模数为 11 时的结果是否等于 (h, s, t)
    assert cofactors(f, g, modulus=11) == (h, s, t)
    # 使用 gcd 函数检查 f 和 g 在模数为 11 时的最大公约数是否等于 h
    assert gcd(f, g, modulus=11) == h
    # 使用 lcm 函数检查 f 和 g 在模数为 11 时的最小公倍数是否等于 l
    assert lcm(f, g, modulus=11) == l

    # 重新定义 f 和 g
    f, g = x**2 + 8*x + 7, x**3 + 7*x**2 + x + 7
    # 重新定义 l 为 x^4 + 8*x^3 + 8*x^2 + 8*x + 7
    l = x**4 + 8*x**3 + 8*x**2 + 8*x + 7
    # 重新定义 h, s, t 分别为 x + 7, x + 1, x^2 + 1
    h, s, t = x + 7, x + 1, x**2 + 1

    # 使用 cofactors 函数检查 f 和 g 在模数为 11 时的结果是否等于 (h, s, t)，并指定 symmetric=False
    assert cofactors(f, g, modulus=11, symmetric=False) == (h, s, t)
    # 使用 gcd 函数检查 f 和 g 在模数为 11 时的最大公约数是否等于 h，指定 symmetric=False
    assert gcd(f, g, modulus=11, symmetric=False) == h
    # 使用 lcm 函数检查 f 和 g 在模数为 11 时的最小公倍数是否等于 l，指定 symmetric=False
    assert lcm(f, g, modulus=11, symmetric=False) == l

    # 定义 a, b 为 sqrt(2) 和 -sqrt(2)
    a, b = sqrt(2), -sqrt(2)
    # 使用 gcd 函数检查 a 和 b 的最大公约数是否等于 gcd(b, a) 等于 sqrt(2)
    assert gcd(a, b) == gcd(b, a) == sqrt(2)

    # 定义 a, b 为 sqrt(-2) 和 -sqrt(-2)
    a, b = sqrt(-2), -sqrt(-2)
    # 使用 gcd 函数检查 a 和 b 的最大公约数是否等于 gcd(b, a) 等于 sqrt(2)
    assert gcd(a, b) == gcd(b, a) == sqrt(2)

    # 使用 gcd 函数检查 Poly(x - 2, x) 和 Poly(I*x, x) 的最大公约数是否等于 Poly(1, x, domain=ZZ_I)
    assert gcd(Poly(x - 2, x), Poly(I*x, x)) == Poly(1, x, domain=ZZ_I)

    # 使用 raises 函数确保调用 gcd(x) 抛出 TypeError 异常
    raises(TypeError, lambda: gcd(x))
    # 使用 raises 函数确保调用 lcm(x) 抛出 TypeError 异常
    raises(TypeError, lambda: lcm(x))

    # 重新定义 f 和 g
    f = Poly(-1, x)
    g = Poly(1, x)
    # 使用 lcm 函数检查 f 和 g 的最小公倍数是否等于 Poly(1, x)
    assert lcm(f, g) == Poly(1, x)

    # 重新定义 f 和 g
    f = Poly(0, x)
    g = Poly([1, 1], x)
    # 对于 f 和 g 的每一个，确保以下等式成立
    for i in (f, g):
        assert lcm(i, 0) == 0
        assert lcm(0, i) == 0
        assert lcm(i, f) == 0
        assert lcm(f, i) == 0

    # 重新定义 f
    f = 4*x**2 + x + 2
    # 创建 Poly 对象 pfz 和 pfq
    pfz = Poly(f, domain=ZZ)
    pfq = Poly(f, domain=QQ)

    # 使用 pfz 对象的 gcd 方法检查自身的最大公约数是否等于自身
    assert pfz.gcd(pfz) == pfz
    # 使用 pfz 对象的 lcm 方法检查自身的最小公倍数是否等于自身
    assert pfz.lcm(pfz) == pfz
    # 使用 pfq 对象的 gcd 方法检查自身的最大公约数是否等于 pfq 的首一形式
    assert pfq.gcd(pfq) == pfq.monic()
    # 使用 pfq 对象的 lcm 方法检查自身的最小公倍数是否等于 pfq 的首一形式
    assert pfq.lcm(pfq) == pfq.monic()
    # 使用 gcd 函数检查 f 和 f 的最大公约数是否等于 f
    assert gcd(f, f) == f
    # 使用 lcm 函数检查 f 和 f 的最小公倍数是否等于 f
    assert lcm(f, f) == f
    # 使用 gcd 函数检查 f 和 f 在 domain=QQ 时的最大公约数是否等于 f 的首一形式
    assert gcd(f, f, domain=QQ) == monic(f)
    # 使用 lcm 函数检查 f 和 f 在 domain=QQ 时的最小公倍数是否等于 f 的首一形式
    assert lcm(f, f, domain=QQ) == monic(f)
# 测试函数，用于验证 gcd 函数在不同类型和参数下的行为
def test_gcd_numbers_vs_polys():
    # 检查 gcd 函数对于整数的返回类型是否为 Integer
    assert isinstance(gcd(3, 9), Integer)
    # 检查 gcd 函数对于多项式表达式的返回类型是否为 Integer
    assert isinstance(gcd(3*x, 9), Integer)

    # 检查 gcd 函数对于整数的计算结果是否正确
    assert gcd(3, 9) == 3
    # 检查 gcd 函数对于多项式表达式的计算结果是否正确
    assert gcd(3*x, 9) == 3

    # 检查 gcd 函数对于有理数的返回类型是否为 Rational
    assert isinstance(gcd(Rational(3, 2), Rational(9, 4)), Rational)
    # 检查 gcd 函数对于有理数的多项式表达式的返回类型是否为 Rational
    assert isinstance(gcd(Rational(3, 2)*x, Rational(9, 4)), Rational)

    # 检查 gcd 函数对于有理数的计算结果是否正确
    assert gcd(Rational(3, 2), Rational(9, 4)) == Rational(3, 4)
    # 检查 gcd 函数对于有理数的多项式表达式的计算结果是否正确
    assert gcd(Rational(3, 2)*x, Rational(9, 4)) == 1

    # 检查 gcd 函数对于浮点数的返回类型是否为 Float
    assert isinstance(gcd(3.0, 9.0), Float)
    # 检查 gcd 函数对于浮点数的多项式表达式的返回类型是否为 Float
    assert isinstance(gcd(3.0*x, 9.0), Float)

    # 检查 gcd 函数对于浮点数的计算结果是否正确
    assert gcd(3.0, 9.0) == 1.0
    # 检查 gcd 函数对于浮点数的多项式表达式的计算结果是否正确
    assert gcd(3.0*x, 9.0) == 1.0

    # 针对 issue 20597 的部分修复验证
    assert gcd(Mul(2, 3, evaluate=False), 2) == 2


# 测试 terms_gcd 函数，验证其对多项式的最大公约数计算是否正确
def test_terms_gcd():
    # 验证 terms_gcd 函数对于单个参数的计算结果是否正确
    assert terms_gcd(1) == 1
    # 验证 terms_gcd 函数对于包含变量的单个参数的计算结果是否正确
    assert terms_gcd(1, x) == 1

    # 验证 terms_gcd 函数对于单项式的计算结果是否正确
    assert terms_gcd(x - 1) == x - 1
    assert terms_gcd(-x - 1) == -x - 1

    # 验证 terms_gcd 函数对于一般多项式的计算结果是否正确
    assert terms_gcd(2*x + 3) == 2*x + 3
    assert terms_gcd(6*x + 4) == Mul(2, 3*x + 2, evaluate=False)

    # 验证 terms_gcd 函数对于多项式的计算结果是否正确，包含同类项合并
    assert terms_gcd(x**3*y + x*y**3) == x*y*(x**2 + y**2)
    assert terms_gcd(2*x**3*y + 2*x*y**3) == 2*x*y*(x**2 + y**2)
    assert terms_gcd(x**3*y/2 + x*y**3/2) == x*y/2*(x**2 + y**2)

    # 验证 terms_gcd 函数对于不同系数的多项式的计算结果是否正确
    assert terms_gcd(x**3*y + 2*x*y**3) == x*y*(x**2 + 2*y**2)
    assert terms_gcd(2*x**3*y + 4*x*y**3) == 2*x*y*(x**2 + 2*y**2)
    assert terms_gcd(2*x**3*y/3 + 4*x*y**3/5) == x*y*Rational(2, 15)*(5*x**2 + 6*y**2)

    # 验证 terms_gcd 函数对于包含浮点数的多项式的计算结果是否正确
    assert terms_gcd(2.0*x**3*y + 4.1*x*y**3) == x*y*(2.0*x**2 + 4.1*y**2)
    assert _aresame(terms_gcd(2.0*x + 3), 2.0*x + 3)

    # 验证 terms_gcd 函数对于包含复杂表达式的多项式的计算结果是否正确
    assert terms_gcd((3 + 3*x)*(x + x*y), expand=False) == \
        (3*x + 3)*(x*y + x)
    assert terms_gcd((3 + 3*x)*(x + x*sin(3 + 3*y)), expand=False, deep=True) == \
        3*x*(x + 1)*(sin(Mul(3, y + 1, evaluate=False)) + 1)
    assert terms_gcd(sin(x + x*y), deep=True) == \
        sin(x*(y + 1))

    # 验证 terms_gcd 函数对于方程式的计算结果是否正确
    eq = Eq(2*x, 2*y + 2*z*y)
    assert terms_gcd(eq) == Eq(2*x, 2*y*(z + 1))
    assert terms_gcd(eq, deep=True) == Eq(2*x, 2*y*(z + 1))

    # 验证 terms_gcd 函数对于非法输入的异常处理是否正确
    raises(TypeError, lambda: terms_gcd(x < 2))


# 测试 trunc 函数，验证其在多项式截断操作中的行为是否符合预期
def test_trunc():
    # 创建两个多项式对象
    f, g = x**5 + 2*x**4 + 3*x**3 + 4*x**2 + 5*x + 6, x**5 - x**4 + x**2 - x
    F, G = Poly(f), Poly(g)

    # 验证 Poly 类的 trunc 方法在给定模数下的截断结果是否正确
    assert F.trunc(3) == G
    # 验证 trunc 函数在给定模数下的截断结果是否正确
    assert trunc(f, 3) == g
    assert trunc(f, 3, x) == g
    assert trunc(f, 3, (x,)) == g
    assert trunc(F, 3) == G
    assert trunc(f, 3, polys=True) == G
    assert trunc(F, 3, polys=False) == g

    # 创建另外两个多项式对象
    f, g = 6*x**5 + 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1, -x**4 + x**3 - x + 1
    F, G = Poly(f), Poly(g)

    # 再次验证 Poly 类的 trunc 方法在给定模数下的截断结果是否正确
    assert F.trunc(3) == G
    assert trunc(f, 3) == g
    assert trunc(f, 3, x) == g
    assert trunc(f, 3, (x,)) == g
    assert trunc(F, 3) == G
    assert trunc(f, 3, polys=True) == G
    assert trunc(F, 3, polys=False) == g

    # 验证 Poly 类的 trunc 方法在给定模数下的截断结果是否正确
    f = Poly(x**2 + 2*x + 3, modulus=5)
    assert f.trunc(2) == Poly(x**2 + 1, modulus=5)


# 测试 monic 函数，验证其对多项式的首项系数归一化操作是否正确
def test_monic():
    # 创建两个多项式对象，一个是有理数域下的多项式，一个是普通多项式
    f, g = 2*x - 1, x - S.Half
    F, G = Poly(f, domain='QQ'), Poly(g)

    # 验证 Poly 类的 monic 方法在有理
    # 断言：对于多项式 f，要求其首项系数为单位元 G，并且多项式类型为多项式
    assert monic(f, polys=True) == G
    # 断言：对于多项式 F，要求其首项系数为单位元 g，并且多项式类型为非多项式
    assert monic(F, polys=False) == g

    # 断言：调用 monic 函数传入一个整数，预期抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: monic(4))

    # 断言：对于多项式 2*x**2 + 6*x + 4，关闭自动模式，要求其首项系数为单位元 x**2 + 3*x + 2
    assert monic(2*x**2 + 6*x + 4, auto=False) == x**2 + 3*x + 2
    # 断言：对于多项式 2*x + 6*x + 1，关闭自动模式，预期抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: monic(2*x + 6*x + 1, auto=False))

    # 断言：对于浮点数多项式 2.0*x**2 + 6.0*x + 4.0，要求其首项系数为单位元 1.0*x**2 + 3.0*x + 2.0
    assert monic(2.0*x**2 + 6.0*x + 4.0) == 1.0*x**2 + 3.0*x + 2.0
    # 断言：对于多项式 2*x**2 + 3*x + 4，使用模数 5，要求其首项系数为单位元 x**2 - x + 2
    assert monic(2*x**2 + 3*x + 4, modulus=5) == x**2 - x + 2
def test_content():
    # 定义多项式 f 和 Poly 对象 F
    f, F = 4*x + 2, Poly(4*x + 2)

    # 断言 Poly 对象 F 的内容为 2
    assert F.content() == 2
    # 断言函数 content 计算多项式 f 的内容为 2
    assert content(f) == 2

    # 断言对于整数 4，调用 content 函数会引发 ComputationFailed 异常
    raises(ComputationFailed, lambda: content(4))

    # 使用 modulus=3 创建多项式 f
    f = Poly(2*x, modulus=3)

    # 断言 Poly 对象 f 的内容为 1
    assert f.content() == 1


def test_primitive():
    # 定义多项式 f 和 g，以及对应的 Poly 对象 F 和 G
    f, g = 4*x + 2, 2*x + 1
    F, G = Poly(f), Poly(g)

    # 断言 Poly 对象 F 的原始分解为 (2, G)
    assert F.primitive() == (2, G)
    # 断言函数 primitive 计算多项式 f 的原始分解为 (2, g)
    assert primitive(f) == (2, g)
    assert primitive(f, x) == (2, g)
    assert primitive(f, (x,)) == (2, g)
    assert primitive(F) == (2, G)
    assert primitive(f, polys=True) == (2, G)
    assert primitive(F, polys=False) == (2, g)

    # 断言对于整数 4，调用 primitive 函数会引发 ComputationFailed 异常
    raises(ComputationFailed, lambda: primitive(4))

    # 使用 modulus=3 创建多项式 f 和 g
    f = Poly(2*x, modulus=3)
    g = Poly(2.0*x, domain=RR)

    # 断言 Poly 对象 f 的原始分解为 (1, f)
    assert f.primitive() == (1, f)
    # 断言 Poly 对象 g 的原始分解为 (1.0, g)
    assert g.primitive() == (1.0, g)

    # 断言对于字符串表示的表达式进行原始分解的正确性
    assert primitive(S('-3*x/4 + y + 11/8')) == \
        S('(1/8, -6*x + 8*y + 11)')


def test_compose():
    # 定义多项式 f, g, h 和对应的 Poly 对象 F, G, H
    f = x**12 + 20*x**10 + 150*x**8 + 500*x**6 + 625*x**4 - 2*x**3 - 10*x + 9
    g = x**4 - 2*x + 9
    h = x**3 + 5*x

    F, G, H = map(Poly, (f, g, h))

    # 断言 Poly 对象 G 对 H 的合成结果为 F
    assert G.compose(H) == F
    # 断言函数 compose 计算 g 对 h 的合成结果为 f
    assert compose(g, h) == f
    assert compose(g, h, x) == f
    assert compose(g, h, (x,)) == f
    assert compose(G, H) == F
    assert compose(g, h, polys=True) == F
    assert compose(G, H, polys=False) == f

    # 断言 Poly 对象 F 的分解结果为 [G, H]
    assert F.decompose() == [G, H]
    # 断言函数 decompose 计算多项式 f 的分解结果为 [g, h]
    assert decompose(f) == [g, h]
    assert decompose(f, x) == [g, h]
    assert decompose(f, (x,)) == [g, h]
    assert decompose(F) == [G, H]
    assert decompose(f, polys=True) == [G, H]
    assert decompose(F, polys=False) == [g, h]

    # 断言对于整数参数调用 compose 和 decompose 函数会引发 ComputationFailed 异常
    raises(ComputationFailed, lambda: compose(4, 2))
    raises(ComputationFailed, lambda: decompose(4))

    # 断言对于不同的参数顺序，函数 compose 的结果正确性
    assert compose(x**2 - y**2, x - y, x, y) == x**2 - 2*x*y
    assert compose(x**2 - y**2, x - y, y, x) == -y**2 + 2*x*y


def test_shift():
    # 断言多项式 x**2 - 2*x + 1 在 x 方向移动 2 个单位后的结果
    assert Poly(x**2 - 2*x + 1, x).shift(2) == Poly(x**2 + 2*x + 1, x)


def test_shift_list():
    # 断言多项式 x*y 在 x, y 方向分别移动 [1, 2] 个单位后的结果
    assert Poly(x*y, [x,y]).shift_list([1,2]) == Poly((x+1)*(y+2), [x,y])


def test_transform():
    # 断言多项式 x**2 - 2*x + 1 在三重单元化下的变换结果
    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1), Poly(x - 1)) == \
        Poly(4, x) == \
        cancel((x - 1)**2*(x**2 - 2*x + 1).subs(x, (x + 1)/(x - 1)))

    assert Poly(x**2 - x/2 + 1, x).transform(Poly(x + 1), Poly(x - 1)) == \
        Poly(3*x**2/2 + Rational(5, 2), x) == \
        cancel((x - 1)**2*(x**2 - x/2 + 1).subs(x, (x + 1)/(x - 1)))

    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + S.Half), Poly(x - 1)) == \
        Poly(Rational(9, 4), x) == \
        cancel((x - 1)**2*(x**2 - 2*x + 1).subs(x, (x + S.Half)/(x - 1)))

    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1), Poly(x - S.Half)) == \
        Poly(Rational(9, 4), x) == \
        cancel((x - S.Half)**2*(x**2 - 2*x + 1).subs(x, (x + 1)/(x - S.Half)))

    # 统一 ZZ, QQ 和 RR 的结果
    # 断言：将多项式 x**2 - 2*x + 1 转换为以 x + 1.0 和 x - S.Half 为基础的多项式，验证结果是否等于 Rational(9, 4)，且定义域为实数域（'RR'）
    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1.0), Poly(x - S.Half)) == \
        Poly(Rational(9, 4), x, domain='RR') == \
        cancel((x - S.Half)**2*(x**2 - 2*x + 1).subs(x, (x + 1.0)/(x - S.Half)))
    
    # 引发异常：尝试用不匹配的基础多项式进行多项式变换
    raises(ValueError, lambda: Poly(x*y).transform(Poly(x + 1), Poly(x - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(y + 1), Poly(x - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(x + 1), Poly(y - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(x*y + 1), Poly(x - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(x + 1), Poly(x*y - 1)))
# 定义一个测试函数 test_sturm，用于测试多项式的 Sturm 序列相关功能
def test_sturm():
    # 设置两个多项式 f 和 F，分别为 x 和 Poly(x, domain='QQ')
    f, F = x, Poly(x, domain='QQ')
    # 设置两个常数 g 和 G，分别为 1 和 Poly(1, x, domain='QQ')

    # 断言 Poly 对象的 sturm 方法返回期望的列表 [F, G]
    assert F.sturm() == [F, G]
    # 断言全局函数 sturm 对 f 执行期望的操作，返回 [f, g]
    assert sturm(f) == [f, g]
    # 断言全局函数 sturm 对 f 使用指定变量 x，返回 [f, g]
    assert sturm(f, x) == [f, g]
    # 断言全局函数 sturm 对 f 使用元组 (x,)，返回 [f, g]
    assert sturm(f, (x,)) == [f, g]
    # 断言全局函数 sturm 对 Poly 对象 F 执行期望的操作，返回 [F, G]
    assert sturm(F) == [F, G]
    # 断言全局函数 sturm 对 f 使用 polys=True，返回 [F, G]
    assert sturm(f, polys=True) == [F, G]
    # 断言全局函数 sturm 对 Poly 对象 F 使用 polys=False，返回 [f, g]
    assert sturm(F, polys=False) == [f, g]

    # 测试错误处理：使用 lambda 函数检查 sturm(4) 是否引发 ComputationFailed 异常
    raises(ComputationFailed, lambda: sturm(4))
    # 测试错误处理：使用 lambda 函数检查 sturm(f, auto=False) 是否引发 DomainError 异常
    raises(DomainError, lambda: sturm(f, auto=False))

    # 设置一个复杂的多项式 f
    f = Poly(S(1024)/(15625*pi**8)*x**5
           - S(4096)/(625*pi**8)*x**4
           + S(32)/(15625*pi**4)*x**3
           - S(128)/(625*pi**4)*x**2
           + Rational(1, 625)*x
           - Rational(1, 625), x, domain='ZZ(pi)')

    # 断言全局函数 sturm 对 f 执行期望的操作，返回一个列表，每个元素是一个多项式
    assert sturm(f) == [
        Poly(x**3 - 100*x**2 + pi**4/64*x - 25*pi**4/16, x, domain='ZZ(pi)'),
        Poly(3*x**2 - 200*x + pi**4/64, x, domain='ZZ(pi)'),
        Poly((Rational(20000, 9) - pi**4/96)*x + 25*pi**4/18, x, domain='ZZ(pi)'),
        Poly((-3686400000000*pi**4 - 11520000*pi**8 - 9*pi**12)/(26214400000000 - 245760000*pi**4 + 576*pi**8), x, domain='ZZ(pi)')
    ]


# 定义一个测试函数 test_gff，用于测试多项式的 GCD 分解相关功能
def test_gff():
    # 设置一个简单的多项式 f
    f = x**5 + 2*x**4 - x**3 - 2*x**2

    # 断言 Poly 对象的 gff_list 方法返回期望的列表 [(Poly(x), 1), (Poly(x + 2), 4)]
    assert Poly(f).gff_list() == [(Poly(x), 1), (Poly(x + 2), 4)]
    # 断言全局函数 gff_list 对 f 执行期望的操作，返回 [(x, 1), (x + 2, 4)]
    assert gff_list(f) == [(x, 1), (x + 2, 4)]

    # 测试错误处理：使用 lambda 函数检查 gff(f) 是否引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: gff(f))

    # 设置一个复杂的多项式 f
    f = x*(x - 1)**3*(x - 2)**2*(x - 4)**2*(x - 5)

    # 断言 Poly 对象的 gff_list 方法返回期望的列表
    assert Poly(f).gff_list() == [
        (Poly(x**2 - 5*x + 4), 1),
        (Poly(x**2 - 5*x + 4), 2),
        (Poly(x), 3)
    ]
    # 断言全局函数 gff_list 对 f 执行期望的操作，返回 [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]
    assert gff_list(f) == [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]

    # 测试错误处理：使用 lambda 函数检查 gff(f) 是否引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: gff(f))


# 定义一个测试函数 test_norm，用于测试多项式的 norm 方法
def test_norm():
    # 设置两个常数 a 和 b，分别为 sqrt(2) 和 sqrt(3)
    a, b = sqrt(2), sqrt(3)
    # 设置一个多项式 f，使用 Poly 构造函数，指定域为 QQ
    f = Poly(a*x + b*y, x, y, extension=(a, b))
    # 断言多项式对象的 norm 方法返回期望的多项式
    assert f.norm() == Poly(4*x**4 - 12*x**2*y**2 + 9*y**4, x, y, domain='QQ')


# 定义一个测试函数 test_sqf_norm，用于测试多项式的 sqf_norm 方法
def test_sqf_norm():
    # 断言全局函数 sqf_norm 对 x**2 - 2 使用 sqrt(3) 作为扩展域，返回期望的结果元组
    assert sqf_norm(x**2 - 2, extension=sqrt(3)) == \
        ([1], x**2 - 2*sqrt(3)*x + 1, x**4 - 10*x**2 + 1)
    # 断言全局函数 sqf_norm 对 x**2 - 3 使用 sqrt(2) 作为扩展域，返回期望的结果元组
    assert sqf_norm(x**2 - 3, extension=sqrt(2)) == \
        ([1], x**2 - 2*sqrt(2)*x - 1, x**4 - 10*x**2 + 1)

    # 断言 Poly 对象的 sqf_norm 方法返回期望的结果元组
    assert Poly(x**2 - 2, extension=sqrt(3)).sqf_norm() == \
        ([1], Poly(x**2 - 2*sqrt(3)*x + 1, x, extension=sqrt(3)),
              Poly(x**4 - 10*x**2 + 1, x, domain='QQ'))

    # 断言 Poly 对象的 sqf_norm 方法返回期望的结果元组
    assert Poly(x**2 - 3, extension=sqrt(2)).sqf_norm() == \
        ([1], Poly(x**2 - 2*sqrt(2)*x - 1, x, extension=sqrt(2)),
              Poly(x**4 - 10*x**2 + 1, x, domain='QQ'))


# 定义一个测试函数 test_sqf，用于测试多项式的 sqf 相关方法
def test_sqf():
    # 设置几个简单的多项式 f, g, h
    f = x**5 - x**3 - x**2 + 1
    g = x**3 + 2*x**2 + 2*x + 1
    h = x - 1

    # 设置一个复杂的多项式 p
    p = x**4 + x**3 - x - 1

    # 将 f, g, h, p 映射为 Poly 对象 F, G, H, P
    F, G, H, P = map(Poly, (f, g, h, p))

    # 断言 Poly 对象的 sqf_part 方法返回期望的多项式 P
    assert F.sqf_part() == P
    # 断言全局函数 sqf_part 对 f 执行期望的操作，返回 p
    assert sqf_part(f) == p
    #
    # 断言sqf_list(F)返回的结果是否等于(1, [(G, 1), (H, 2)])
    assert sqf_list(F) == (1, [(G, 1), (H, 2)])
    # 断言sqf_list(f, polys=True)返回的结果是否等于(1, [(G, 1), (H, 2)])
    assert sqf_list(f, polys=True) == (1, [(G, 1), (H, 2)])
    # 断言sqf_list(F, polys=False)返回的结果是否等于(1, [(g, 1), (h, 2)])
    assert sqf_list(F, polys=False) == (1, [(g, 1), (h, 2)])

    # 断言F.sqf_list_include()返回的结果是否等于[(G, 1), (H, 2)]
    assert F.sqf_list_include() == [(G, 1), (H, 2)]

    # 使用lambda函数检查sqf_part(4)是否引发ComputationFailed异常
    raises(ComputationFailed, lambda: sqf_part(4))

    # 断言sqf(1)的返回值是否等于1
    assert sqf(1) == 1
    # 断言sqf_list(1)的返回值是否等于(1, [])
    assert sqf_list(1) == (1, [])

    # 断言sqf((2*x**2 + 2)**7)的返回值是否等于128*(x**2 + 1)**7
    assert sqf((2*x**2 + 2)**7) == 128*(x**2 + 1)**7

    # 断言sqf(f)的返回值是否等于g*h**2
    assert sqf(f) == g*h**2
    # 断言sqf(f, x)的返回值是否等于g*h**2
    assert sqf(f, x) == g*h**2
    # 断言sqf(f, (x,))的返回值是否等于g*h**2
    assert sqf(f, (x,)) == g*h**2

    # 定义d = x**2 + y**2
    d = x**2 + y**2

    # 断言sqf(f/d)的返回值是否等于(g*h**2)/d
    assert sqf(f/d) == (g*h**2)/d
    # 断言sqf(f/d, x)的返回值是否等于(g*h**2)/d
    assert sqf(f/d, x) == (g*h**2)/d
    # 断言sqf(f/d, (x,))的返回值是否等于(g*h**2)/d
    assert sqf(f/d, (x,)) == (g*h**2)/d

    # 断言sqf(x - 1)的返回值是否等于x - 1
    assert sqf(x - 1) == x - 1
    # 断言sqf(-x - 1)的返回值是否等于-x - 1
    assert sqf(-x - 1) == -x - 1

    # 断言sqf(x - 1)的返回值是否等于x - 1
    assert sqf(x - 1) == x - 1
    # 断言sqf(6*x - 10)的返回值是否等于Mul(2, 3*x - 5, evaluate=False)
    assert sqf(6*x - 10) == Mul(2, 3*x - 5, evaluate=False)

    # 断言sqf((6*x - 10)/(3*x - 6))的返回值是否等于Rational(2, 3)*((3*x - 5)/(x - 2))
    assert sqf((6*x - 10)/(3*x - 6)) == Rational(2, 3)*((3*x - 5)/(x - 2))
    # 断言sqf(Poly(x**2 - 2*x + 1))的返回值是否等于(x - 1)**2
    assert sqf(Poly(x**2 - 2*x + 1)) == (x - 1)**2

    # 重新定义f = 3 + x - x*(1 + x) + x**2
    f = 3 + x - x*(1 + x) + x**2

    # 断言sqf(f)的返回值是否等于3
    assert sqf(f) == 3

    # 重新定义f = (x**2 + 2*x + 1)**20000000000
    f = (x**2 + 2*x + 1)**20000000000

    # 断言sqf(f)的返回值是否等于(x + 1)**40000000000
    assert sqf(f) == (x + 1)**40000000000
    # 断言sqf_list(f)的返回值是否等于(1, [(x + 1, 40000000000)])
    assert sqf_list(f) == (1, [(x + 1, 40000000000)])

    # 测试链接的GitHub问题：https://github.com/sympy/sympy/issues/26497
    # 断言sqf(expand(((y - 2)**2 * (y + 2) * (x + 1))))的返回值是否等于(y - 2)**2 * expand((y + 2) * (x + 1))
    assert sqf(expand(((y - 2)**2 * (y + 2) * (x + 1)))) == (y - 2)**2 * expand((y + 2) * (x + 1))
    # 断言sqf(expand(((y - 2)**2 * (y + 2) * (z + 1))))的返回值是否等于(y - 2)**2 * expand((y + 2) * (z + 1))
    assert sqf(expand(((y - 2)**2 * (y + 2) * (z + 1)))) == (y - 2)**2 * expand((y + 2) * (z + 1))
    # 断言sqf(expand(((y - I)**2 * (y + I) * (x + 1))))的返回值是否等于(y - I)**2 * expand((y + I) * (x + 1))
    assert sqf(expand(((y - I)**2 * (y + I) * (x + 1)))) == (y - I)**2 * expand((y + I) * (x + 1))
    # 断言sqf(expand(((y - I)**2 * (y + I) * (z + 1))))的返回值是否等于(y - I)**2 * expand((y + I) * (z + 1))
    assert sqf(expand(((y - I)**2 * (y + I) * (z + 1)))) == (y - I)**2 * expand((y + I) * (z + 1))

    # 检查因子是否组合并排序。
    p = (x - 2)**2*(x - 1)*(x + y)**2*(y - 2)**2*(y - 1)
    # 断言Poly(p).sqf_list()的返回值是否等于(1, [(Poly(x*y - x - y + 1), 1), (Poly(x**2*y - 2*x**2 + x*y**2 - 4*x*y + 4*x - 2*y**2 + 4*y), 2)])
    assert Poly(p).sqf_list() == (1, [
        (Poly(x*y - x - y + 1), 1),
        (Poly(x**2*y - 2*x**2 + x*y**2 - 4*x*y + 4*x - 2*y**2 + 4*y), 2)
    ])
def test_factor():
    # 定义多项式 f(x) = x^5 - x^3 - x^2 + 1
    f = x**5 - x**3 - x**2 + 1

    # 定义多项式 u(x) = x + 1
    u = x + 1
    # 定义多项式 v(x) = x - 1
    v = x - 1
    # 定义多项式 w(x) = x^2 + x + 1
    w = x**2 + x + 1

    # 将 f, u, v, w 转换为多项式对象 F, U, V, W
    F, U, V, W = map(Poly, (f, u, v, w))

    # 断言 F 的因式分解结果为 (1, [(U, 1), (V, 2), (W, 1)])
    assert F.factor_list() == (1, [(U, 1), (V, 2), (W, 1)])
    # 断言调用 factor_list 函数得到的结果等同于上一条断言
    assert factor_list(f) == (1, [(u, 1), (v, 2), (w, 1)])
    # 断言指定变量 x 后调用 factor_list 函数的结果等同于上一条断言
    assert factor_list(f, x) == (1, [(u, 1), (v, 2), (w, 1)])
    # 断言指定变量 (x,) 后调用 factor_list 函数的结果等同于上一条断言
    assert factor_list(f, (x,)) == (1, [(u, 1), (v, 2), (w, 1)])
    # 断言调用 factor_list 函数得到的结果等同于上一条断言，其中输入参数为多项式对象 F
    assert factor_list(F) == (1, [(U, 1), (V, 2), (W, 1)])
    # 断言调用 factor_list 函数，传入 polys=True 参数得到的结果等同于上一条断言
    assert factor_list(f, polys=True) == (1, [(U, 1), (V, 2), (W, 1)])
    # 断言调用 factor_list 函数，传入 polys=False 参数得到的结果等同于上一条断言
    assert factor_list(F, polys=False) == (1, [(u, 1), (v, 2), (w, 1)])

    # 断言 F 的 factor_list_include 方法返回 [(U, 1), (V, 2), (W, 1)]
    assert F.factor_list_include() == [(U, 1), (V, 2), (W, 1)]

    # 断言对于整数 1 的因式分解结果为 (1, [])
    assert factor_list(1) == (1, [])
    # 断言对于整数 6 的因式分解结果为 (6, [])
    assert factor_list(6) == (6, [])
    # 断言对于 sqrt(3) 在变量 x 上的因式分解结果为 (sqrt(3), [])
    assert factor_list(sqrt(3), x) == (sqrt(3), [])
    # 断言对于 (-1)^x 在变量 x 上的因式分解结果为 (1, [(-1, x)])
    assert factor_list((-1)**x, x) == (1, [(-1, x)])
    # 断言对于 (2*x)^y 在变量 x 上的因式分解结果为 (1, [(2, y), (x, y)])
    assert factor_list((2*x)**y, x) == (1, [(2, y), (x, y)])
    # 断言对于 sqrt(x*y) 在变量 x 上的因式分解结果为 (1, [(x*y, S.Half)])
    assert factor_list(sqrt(x*y), x) == (1, [(x*y, S.Half)])

    # 断言对于 3*x 的因式分解结果为 3*x
    assert factor(3*x) == 3*x
    # 断言对于 3*x^2 的因式分解结果为 3*x^2
    assert factor(3*x**2) == 3*x**2

    # 断言对于 (2*x^2 + 2)^7 的因式分解结果为 128*(x^2 + 1)^7
    assert factor((2*x**2 + 2)**7) == 128*(x**2 + 1)**7

    # 断言对于 f 的因式分解结果为 u*v^2*w
    assert factor(f) == u*v**2*w
    # 断言对于 f 在变量 x 上的因式分解结果等同于上一条断言
    assert factor(f, x) == u*v**2*w
    # 断言对于 f 在变量 (x,) 上的因式分解结果等同于上一条断言
    assert factor(f, (x,)) == u*v**2*w

    # 定义多项式 g, p, q, r
    g, p, q, r = x**2 - y**2, x - y, x + y, x**2 + 1

    # 断言对于 f/g 的因式分解结果为 (u*v^2*w)/(p*q)
    assert factor(f/g) == (u*v**2*w)/(p*q)
    # 断言对于 f/g 在变量 x 上的因式分解结果等同于上一条断言
    assert factor(f/g, x) == (u*v**2*w)/(p*q)
    # 断言对于 f/g 在变量 (x,) 上的因式分解结果等同于上一条断言
    assert factor(f/g, (x,)) == (u*v**2*w)/(p*q)

    # 定义符号 p, i, r 的属性
    p = Symbol('p', positive=True)
    i = Symbol('i', integer=True)
    r = Symbol('r', real=True)

    # 断言对于 sqrt(x*y) 的因式分解结果是一个 Pow 对象
    assert factor(sqrt(x*y)).is_Pow is True

    # 断言对于 sqrt(3*x^2 - 3) 的因式分解结果为 sqrt(3)*sqrt((x - 1)*(x + 1))
    assert factor(sqrt(3*x**2 - 3)) == sqrt(3)*sqrt((x - 1)*(x + 1))
    # 断言对于 sqrt(3*x^2 + 3) 的因式分解结果为 sqrt(3)*sqrt(x^2 + 1)
    assert factor(sqrt(3*x**2 + 3)) == sqrt(3)*sqrt(x**2 + 1)

    # 断言对于 (y*x^2 - y)^i 的因式分解结果为 y^i*(x - 1)^i*(x + 1)^i
    assert factor((y*x**2 - y)**i) == y**i*(x - 1)**i*(x + 1)**i
    # 断言对于 (y*x^2 + y)^i 的因式分解结果为 y^i*(x^2 + 1)^i
    assert factor((y*x**2 + y)**i) == y**i*(x**2 + 1)**i

    # 断言对于 (y*x^2 - y)^t 的因式分解结果为 (y*(x - 1)*(x + 1))^t
    assert factor((y*x**2 - y)**t) == (y*(x - 1)*(x + 1))**t
    # 断言对于 (y*x^2 + y)^t 的因式分解结果为 (y*(x^2 + 1))^t
    assert factor((y*x**2 + y)**t) == (y*(x**2 + 1))**t

    # 定义多项式 f 和 g
    f = sqrt(expand((r**2 + 1)*(p + 1)*(p - 1)*(p - 2)**3))
    g = sqrt((p - 2)**3*(p - 1))*sqrt(p + 1)*sqrt(r**2 + 1)

    # 断言对于 f 的因式分解结果为 g
    assert factor(f) == g
    # 断言对于 g 的因式分解结果等同于上一条断言
    assert factor(g) == g

    # 定义多项式 g 和 f
    g = (x - 1)**5*(r**2 + 1)
    f = sqrt(expand(g))

    # 断言
    # 断言：使用 QQ_I 域对多项式 f 进行因式分解，并断言其结果等于 f_qqi
    assert factor(f, domain=QQ_I) == f_qqi

    # 定义多项式 f
    f = x**2 + 2*sqrt(2)*x + 2

    # 断言：使用扩展域 sqrt(2) 对多项式 f 进行因式分解，并断言其结果为 (x + sqrt(2))**2
    assert factor(f, extension=sqrt(2)) == (x + sqrt(2))**2
    # 断言：使用扩展域 sqrt(2) 对多项式 f 的立方进行因式分解，并断言其结果为 (x + sqrt(2))**6
    assert factor(f**3, extension=sqrt(2)) == (x + sqrt(2))**6

    # 断言：使用扩展域 sqrt(2) 对多项式 x**2 - 2*y**2 进行因式分解，并断言其结果为 (x + sqrt(2)*y)*(x - sqrt(2)*y)
    assert factor(x**2 - 2*y**2, extension=sqrt(2)) == \
        (x + sqrt(2)*y)*(x - sqrt(2)*y)
    # 断言：使用扩展域 sqrt(2) 对多项式 2*x**2 - 4*y**2 进行因式分解，并断言其结果为 2*((x + sqrt(2)*y)*(x - sqrt(2)*y))
    assert factor(2*x**2 - 4*y**2, extension=sqrt(2)) == \
        2*((x + sqrt(2)*y)*(x - sqrt(2)*y))

    # 断言：对多项式 x - 1 进行因式分解，并断言其结果为 x - 1
    assert factor(x - 1) == x - 1
    # 断言：对多项式 -x - 1 进行因式分解，并断言其结果为 -x - 1
    assert factor(-x - 1) == -x - 1

    # 断言：对多项式 x - 1 进行因式分解，并断言其结果为 x - 1
    assert factor(x - 1) == x - 1

    # 断言：对多项式 6*x - 10 进行因式分解，并断言其结果为 Mul(2, 3*x - 5, evaluate=False)
    assert factor(6*x - 10) == Mul(2, 3*x - 5, evaluate=False)

    # 断言：对多项式 x**11 + x + 1 使用模数 65537 和对称选项进行因式分解，并断言其结果为 (x**2 + x + 1)*(x**9 - x**8 + x**6 - x**5 + x**3 - x**2 + 1)
    assert factor(x**11 + x + 1, modulus=65537, symmetric=True) == \
        (x**2 + x + 1)*(x**9 - x**8 + x**6 - x**5 + x**3 - x**2 + 1)
    # 断言：对多项式 x**11 + x + 1 使用模数 65537 和非对称选项进行因式分解，并断言其结果为 (x**2 + x + 1)*(x**9 + 65536*x**8 + x**6 + 65536*x**5 + x**3 + 65536*x**2 + 1)
    assert factor(x**11 + x + 1, modulus=65537, symmetric=False) == \
        (x**2 + x + 1)*(x**9 + 65536*x**8 + x**6 + 65536*x**5 +
         x**3 + 65536*x** 2 + 1)

    # 定义多项式 f 和 g
    f = x/pi + x*sin(x)/pi
    g = y/(pi**2 + 2*pi + 1) + y*sin(x)/(pi**2 + 2*pi + 1)

    # 断言：对多项式 f 进行因式分解，并断言其结果为 x*(sin(x) + 1)/pi
    assert factor(f) == x*(sin(x) + 1)/pi
    # 断言：对多项式 g 进行因式分解，并断言其结果为 y*(sin(x) + 1)/(pi + 1)**2
    assert factor(g) == y*(sin(x) + 1)/(pi + 1)**2

    # 断言：对方程 x**2 + 2*x + 1 = x**3 + 1 进行因式分解，并断言其结果为 Eq((x + 1)**2, (x + 1)*(x**2 - x + 1))
    assert factor(Eq(
        x**2 + 2*x + 1, x**3 + 1)) == Eq((x + 1)**2, (x + 1)*(x**2 - x + 1))

    # 定义多项式 f
    f = (x**2 - 1)/(x**2 + 4*x + 4)

    # 断言：对多项式 f 进行因式分解，并断言其结果为 (x + 1)*(x - 1)/(x + 2)**2
    assert factor(f) == (x + 1)*(x - 1)/(x + 2)**2
    # 断言：对多项式 f 进行关于变量 x 的因式分解，并断言其结果为 (x + 1)*(x - 1)/(x + 2)**2
    assert factor(f, x) == (x + 1)*(x - 1)/(x + 2)**2

    # 定义多项式 f
    f = 3 + x - x*(1 + x) + x**2

    # 断言：对多项式 f 进行因式分解，并断言其结果为 3
    assert factor(f) == 3
    # 断言：对多项式 f 关于变量 x 进行因式分解，并断言其结果为 3
    assert factor(f, x) == 3

    # 断言：对多项式 1/(x**2 + 2*x + 1/x) - 1 进行因式分解，并断言其结果为 -((1 - x + 2*x**2 + x**3)/(1 + 2*x**2 + x**3))
    assert factor(1/(x**2 + 2*x + 1/x) - 1) == -((1 - x + 2*x**2 +
                  x**3)/(1 + 2*x**2 + x**3))

    # 断言：对多项式 f 使用 expand=False 选项进行因式分解，并断言其结果为 f
    assert factor(f, expand=False) == f
    # 使用 lambda 表达式断言：对多项式 f 使用 expand=False 和变量 x 进行因式分解，应抛出 PolynomialError 异常
    raises(PolynomialError, lambda: factor(f, x, expand=False))

    # 使用 lambda 表达式断言：对多项式 x**2 - 1 进行因式分解，应抛出 FlagError 异常
    raises(FlagError, lambda: factor(x**2 - 1, polys=True))

    # 断言：对列表 [x, Eq(x**2 - y**2, Tuple(x**2 - z**2, 1/x + 1/y))] 进行因式分解，并断言其结果为 [x, Eq((x - y)*(x + y), Tuple((x - z)*(x + z), (x + y)/x/y))]
    assert factor([x, Eq(x**2 - y**2, Tuple(x**2 - z**2, 1/x + 1/y))]) == \
        [x, Eq((x - y)*(x + y), Tuple((x - z)*(x + z), (x + y)/x/y))]

    # 断言：使用 Poly 类对 x**3 + x + 1 进行因式分解，检查结果中第一项的类型不是 PurePoly
    assert not isinstance(
        Poly(x**3 + x + 1).factor_list()[1][0][0], PurePoly) is True
    # 断言：使用 PurePoly 类对 x**3 + x + 1 进行因式分解，检查结果中第一项的类型是 PurePoly
    assert isinstance(
        PurePoly(x**3 + x + 1).factor_list()[1][0
    # 断言：验证表达式的因式分解结果是否正确
    assert factor(eq, y, deep=True) == (y + 5)*(y + 6)*(x**2 + 7*x + 12)
    
    # 使用 fraction=True 选项确保因式分解中包含有理数的展开
    f = 5*x + 3*exp(2 - 7*x)
    assert factor(f, deep=True) == factor(f, deep=True, fraction=True)
    
    # 使用 fraction=False 选项确保因式分解中不包含有理数的展开
    assert factor(f, deep=True, fraction=False) == 5*x + 3*exp(2)*exp(-7*x)
    
    # 使用给定的变量顺序对多元因式进行因式分解，返回系数和因子列表
    assert factor_list(x**3 - x*y**2, t, w, x) == (
        1, [(x, 1), (x - y, 1), (x + y, 1)])
    
    # 验证特定数学表达式的因式分解结果是否正确
    s2, s2p, s2n = sqrt(2), 1 + sqrt(2), 1 - sqrt(2)
    pip, pin = 1 + pi, 1 - pi
    assert factor_list(s2p*s2n) == (-1, [(-s2n, 1), (s2p, 1)])
    assert factor_list(pip*pin) == (-1, [(-pin, 1), (pip, 1)])
    assert factor_list(s2*s2n) == (-s2, [(-s2n, 1)])
    assert factor_list(pi*pin) == (-1, [(-pin, 1), (pi, 1)])
    assert factor_list(s2p*s2n, x) == (s2p*s2n, [])
    assert factor_list(pip*pin, x) == (pip*pin, [])
    assert factor_list(s2*s2n, x) == (s2*s2n, [])
    assert factor_list(pi*pin, x) == (pi*pin, [])
    assert factor_list((x - sqrt(2)*pi)*(x + sqrt(2)*pi), x) == (
        1, [(x - sqrt(2)*pi, 1), (x + sqrt(2)*pi, 1)])
    
    # 验证特定复杂表达式的因式分解结果是否正确
    p = ((y - I)**2 * (y + I) * (x + 1))
    assert factor(expand(p)) == p
    
    p = ((x - I)**2 * (x + I) * (y + 1))
    assert factor(expand(p)) == p
    
    p = (y + 1)**2*(y + sqrt(2))**2*(x**2 + x + 2 + 3*sqrt(2))**2
    assert factor(expand(p), extension=True) == p
    
    # 验证复杂表达式的因式分解结果是否正确
    e = (
        -x**2*y**4/(y**2 + 1) + 2*I*x**2*y**3/(y**2 + 1) + 2*I*x**2*y/(y**2 + 1) +
        x**2/(y**2 + 1) - 2*x*y**4/(y**2 + 1) + 4*I*x*y**3/(y**2 + 1) +
        4*I*x*y/(y**2 + 1) + 2*x/(y**2 + 1) - y**4 - y**4/(y**2 + 1) + 2*I*y**3 +
        2*I*y**3/(y**2 + 1) + 2*I*y + 2*I*y/(y**2 + 1) + 1 + 1/(y**2 + 1)
    )
    assert factor(e) == -(y - I)**3*(y + I)*(x**2 + 2*x + y**2 + 2)/(y**2 + 1)
    
    
    这段代码是一系列用于验证 SymPy 库中因式分解函数 `factor` 和 `factor_list` 的测试断言。每个断言都有不同的用例和选项，用来确保因式分解结果的正确性和符合预期的展开方式。
# 定义一个测试函数，用于测试大整数的因式分解
def test_factor_large():
    # 计算第一个多项式 f
    f = (x**2 + 4*x + 4)**10000000*(x**2 + 1)*(x**2 + 2*x + 1)**1234567
    # 计算第二个多项式 g
    g = ((x**2 + 2*x + 1)**3000*y**2 + (x**2 + 2*x + 1)**3000*2*y + (x**2 + 2*x + 1)**3000)

    # 断言：使用 factor 函数对 f 进行因式分解，应得到的结果
    assert factor(f) == (x + 2)**20000000*(x**2 + 1)*(x + 1)**2469134
    # 断言：使用 factor 函数对 g 进行因式分解，应得到的结果
    assert factor(g) == (x + 1)**6000*(y + 1)**2

    # 断言：使用 factor_list 函数对 f 进行因式分解，应得到的结果
    assert factor_list(f) == (1, [(x + 1, 2469134), (x + 2, 20000000), (x**2 + 1, 1)])
    # 断言：使用 factor_list 函数对 g 进行因式分解，应得到的结果
    assert factor_list(g) == (1, [(y + 1, 2), (x + 1, 6000)])

    # 重新赋值 f 和 g，准备进行下一组因式分解测试
    f = (x**2 - y**2)**200000*(x**7 + 1)
    g = (x**2 + y**2)**200000*(x**7 + 1)

    # 断言：使用 factor 函数对 f 进行因式分解，应得到的结果
    assert factor(f) == (x + 1)*(x - y)**200000*(x + y)**200000*(x**6 - x**5 + x**4 - x**3 + x**2 - x + 1)
    # 断言：使用 factor 函数对 g 进行因式分解（高斯整数），应得到的结果
    assert factor(g, gaussian=True) == (x + 1)*(x - I*y)**200000*(x + I*y)**200000*(x**6 - x**5 + x**4 - x**3 + x**2 - x + 1)

    # 断言：使用 factor_list 函数对 f 进行因式分解，应得到的结果
    assert factor_list(f) == (1, [(x + 1, 1), (x - y, 200000), (x + y, 200000), (x**6 - x**5 + x**4 - x**3 + x**2 - x + 1, 1)])
    # 断言：使用 factor_list 函数对 g 进行因式分解（高斯整数），应得到的结果
    assert factor_list(g, gaussian=True) == (1, [(x + 1, 1), (x - I*y, 200000), (x + I*y, 200000), (x**6 - x**5 + x**4 - x**3 + x**2 - x + 1, 1)])


# 定义一个测试函数，用于测试不进行求值的因式分解
def test_factor_noeval():
    # 断言：对 6*x - 10 进行因式分解，应得到的结果
    assert factor(6*x - 10) == Mul(2, 3*x - 5, evaluate=False)
    # 断言：对 (6*x - 10)/(3*x - 6) 进行因式分解，应得到的结果
    assert factor((6*x - 10)/(3*x - 6)) == Mul(Rational(2, 3), 3*x - 5, 1/(x - 2))


# 定义一个测试函数，用于测试 intervals 函数的不同输入
def test_intervals():
    # 断言：对 intervals(0) 的计算结果应为空列表
    assert intervals(0) == []
    # 断言：对 intervals(1) 的计算结果应为空列表
    assert intervals(1) == []

    # 断言：对 intervals(x, sqf=True) 的计算结果
    assert intervals(x, sqf=True) == [(0, 0)]
    # 断言：对 intervals(x) 的计算结果
    assert intervals(x) == [((0, 0), 1)]

    # 断言：对 intervals(x**128) 的计算结果
    assert intervals(x**128) == [((0, 0), 128)]
    # 断言：对 intervals([x**2, x**4]) 的计算结果
    assert intervals([x**2, x**4]) == [((0, 0), {0: 2, 1: 4})]

    # 创建一个多项式 f
    f = Poly((x*Rational(2, 5) - Rational(17, 3))*(4*x + Rational(1, 257)))

    # 断言：对 f.intervals(sqf=True) 的计算结果
    assert f.intervals(sqf=True) == [(-1, 0), (14, 15)]
    # 断言：对 f.intervals() 的计算结果
    assert f.intervals() == [((-1, 0), 1), ((14, 15), 1)]

    # 断言：对 f.intervals(fast=True, sqf=True) 的计算结果
    assert f.intervals(fast=True, sqf=True) == [(-1, 0), (14, 15)]
    # 断言：对 f.intervals(fast=True) 的计算结果
    assert f.intervals(fast=True) == [((-1, 0), 1), ((14, 15), 1)]

    # 断言：对 f.intervals(eps=Rational(1, 10)) 的计算结果
    assert f.intervals(eps=Rational(1, 10)) == f.intervals(eps=0.1) == [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    # 断言：对 f.intervals(eps=Rational(1, 100)) 的计算结果
    assert f.intervals(eps=Rational(1, 100)) == f.intervals(eps=0.01) == [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    # 断言：对 f.intervals(eps=Rational(1, 1000)) 的计算结果
    assert f.intervals(eps=Rational(1, 1000)) == f.intervals(eps=0.001) == [((Rational(-1, 1002), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    # 断言：对 f.intervals(eps=Rational(1, 10000)) 的计算结果
    assert f.intervals(eps=Rational(1, 10000)) == f.intervals(eps=0.0001) == [((Rational(-1, 1028), Rational(-1, 1028)), 1), ((Rational(85, 6), Rational(85, 6)), 1)]

    # 将 f 赋值给新的变量，重新进行测试
    f = (x*Rational(2, 5) - Rational(17, 3))*(4*x + Rational(1, 257))

    # 断言：对 intervals(f, sqf=True) 的计算结果
    assert intervals(f, sqf=True) == [(-1, 0), (14, 15)]
    # 断言：对 intervals(f) 的计算结果
    assert intervals(f) == [((-1, 0), 1), ((14, 15), 1)]

    # 断言：对 intervals(f, eps=Rational(1, 10)) 的计算结果
    assert intervals(f, eps=Rational(1, 10)) == intervals(f, eps=0.1) == [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    # 断言：使用 intervals 函数计算 f 函数的区间，并比较多种精度设置的结果是否相等
    assert intervals(f, eps=Rational(1, 100)) == intervals(f, eps=0.01) == \
        [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    
    # 断言：使用 intervals 函数计算 f 函数的区间，并比较多种精度设置的结果是否相等
    assert intervals(f, eps=Rational(1, 1000)) == intervals(f, eps=0.001) == \
        [((Rational(-1, 1002), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    
    # 断言：使用 intervals 函数计算 f 函数的区间，并比较多种精度设置的结果是否相等
    assert intervals(f, eps=Rational(1, 10000)) == intervals(f, eps=0.0001) == \
        [((Rational(-1, 1028), Rational(-1, 1028)), 1), ((Rational(85, 6), Rational(85, 6)), 1)]

    # 定义多项式 f
    f = Poly((x**2 - 2)*(x**2 - 3)**7*(x + 1)*(7*x + 3)**3)
    
    # 断言：使用 f 的 intervals 方法计算其区间
    assert f.intervals() == \
        [((-2, Rational(-3, 2)), 7), ((Rational(-3, 2), -1), 1),
         ((-1, -1), 1), ((-1, 0), 3),
         ((1, Rational(3, 2)), 1), ((Rational(3, 2), 2), 7)]

    # 断言：使用 intervals 函数计算多项式列表的区间
    assert intervals([x**5 - 200, x**5 - 201]) == \
        [((Rational(75, 26), Rational(101, 35)), {0: 1}), ((Rational(309, 107), Rational(26, 9)), {1: 1})]

    # 断言：使用 intervals 函数计算多项式列表的区间，启用快速模式
    assert intervals([x**5 - 200, x**5 - 201], fast=True) == \
        [((Rational(75, 26), Rational(101, 35)), {0: 1}), ((Rational(309, 107), Rational(26, 9)), {1: 1})]

    # 断言：使用 intervals 函数计算多项式列表的区间
    assert intervals([x**2 - 200, x**2 - 201]) == \
        [((Rational(-71, 5), Rational(-85, 6)), {1: 1}), ((Rational(-85, 6), -14), {0: 1}),
         ((14, Rational(85, 6)), {0: 1}), ((Rational(85, 6), Rational(71, 5)), {1: 1})]

    # 断言：使用 intervals 函数计算多项式列表的区间
    assert intervals([x + 1, x + 2, x - 1, x + 1, 1, x - 1, x - 1, (x - 2)**2]) == \
        [((-2, -2), {1: 1}), ((-1, -1), {0: 1, 3: 1}), ((1, 1), {2: 1, 5: 1, 6: 1}), ((2, 2), {7: 2})]

    # 定义多项式 f, g, h
    f, g, h = x**2 - 2, x**4 - 4*x**2 + 4, x - 1

    # 断言：使用 intervals 函数计算 f 函数的区间，限制下界和要求平方因子
    assert intervals(f, inf=Rational(7, 4), sqf=True) == []

    # 断言：使用 intervals 函数计算 f 函数的区间，限制下界和要求平方因子
    assert intervals(f, inf=Rational(7, 5), sqf=True) == [(Rational(7, 5), Rational(3, 2))]

    # 断言：使用 intervals 函数计算 f 函数的区间，限制上界和要求平方因子
    assert intervals(f, sup=Rational(7, 4), sqf=True) == [(-2, -1), (1, Rational(3, 2))]

    # 断言：使用 intervals 函数计算 f 函数的区间，限制上界和要求平方因子
    assert intervals(f, sup=Rational(7, 5), sqf=True) == [(-2, -1)]

    # 断言：使用 intervals 函数计算 g 函数的区间，限制下界
    assert intervals(g, inf=Rational(7, 4)) == []

    # 断言：使用 intervals 函数计算 g 函数的区间，限制下界
    assert intervals(g, inf=Rational(7, 5)) == [((Rational(7, 5), Rational(3, 2)), 2)]

    # 断言：使用 intervals 函数计算 g 函数的区间，限制上界
    assert intervals(g, sup=Rational(7, 4)) == [((-2, -1), 2), ((1, Rational(3, 2)), 2)]

    # 断言：使用 intervals 函数计算 g 函数的区间，限制上界
    assert intervals(g, sup=Rational(7, 5)) == [((-2, -1), 2)]

    # 断言：使用 intervals 函数计算多项式列表的区间，限制下界
    assert intervals([g, h], inf=Rational(7, 4)) == []

    # 断言：使用 intervals 函数计算多项式列表的区间，限制下界
    assert intervals([g, h], inf=Rational(7, 5)) == [((Rational(7, 5), Rational(3, 2)), {0: 2})]

    # 断言：使用 intervals 函数计算多项式列表的区间，限制上界
    assert intervals([g, h], sup=S(7)/4) == [((-2, -1), {0: 2}), ((1, 1), {1: 1}), ((1, Rational(3, 2)), {0: 2})]

    # 断言：使用 intervals 函数计算多项式列表的区间，限制上界
    assert intervals([g, h], sup=Rational(7, 5)) == [((-2, -1), {0: 2}), ((1, 1), {1: 1})]

    # 断言：使用 intervals 函数计算多项式列表的区间
    assert intervals([x + 2, x**2 - 2]) == [((-2, -2), {0: 1}), ((-2, -1), {1: 1}), ((1, 2), {1: 1})]

    # 断言：使用 intervals 函数计算多项式列表的区间，启用严格模式
    assert intervals([x + 2, x**2 - 2], strict=True) == [((-2, -2), {0: 1}), ((Rational(-3, 2), -1), {1: 1}), ((1, 2), {1: 1})]

    # 定义多项式 f
    f = 7*z**4 - 19*z**3 + 20*z**2 + 17*z + 20

    # 断言：使用 intervals 函数计算 f 函数的区间
    assert intervals(f) == []

    # 使用 intervals 函数计算 f 函数的区间，同时获取实部和复部分
    real_part, complex_part = intervals(f, all=True, sqf=True)

    # 断言：验证实部区间为空
    assert real_part == []
    # 断言：验证所有复数部分的区间，确保每个根 r 满足 a < r < b 的实部和虚部条件
    assert all(re(a) < re(r) < re(b) and im(a) < im(r) < im(b) for (a, b), r in zip(complex_part, nroots(f)))
    
    # 断言：验证复数部分是否等于指定的四元组列表
    assert complex_part == [(Rational(-40, 7) - I*40/7, 0),
                            (Rational(-40, 7), I*40/7),
                            (I*Rational(-40, 7), Rational(40, 7)),
                            (0, Rational(40, 7) + I*40/7)]
    
    # 调用 intervals 函数，返回多项式 f 的实部和复数部分的区间列表
    real_part, complex_part = intervals(f, all=True, sqf=True, eps=Rational(1, 10))
    
    # 断言：验证实部区间是否为空列表
    assert real_part == []
    
    # 断言：验证所有复数部分的区间，确保每个根 r 满足 a < r < b 的实部和虚部条件
    assert all(re(a) < re(r) < re(b) and im(a) < im(r) < im(b) for (a, b), r in zip(complex_part, nroots(f)))
    
    # 引发 ValueError 异常，lambda 函数用于测试 intervals 函数对极小值 eps 的异常处理
    raises(ValueError, lambda: intervals(x**2 - 2, eps=10**-100000))
    
    # 引发 ValueError 异常，lambda 函数用于测试 Poly 类的 intervals 方法对极小值 eps 的异常处理
    raises(ValueError, lambda: Poly(x**2 - 2).intervals(eps=10**-100000))
    
    # 引发 ValueError 异常，lambda 函数用于测试 intervals 函数对包含多个多项式的列表的异常处理
    raises(ValueError, lambda: intervals([x**2 - 2, x**2 - 3], eps=10**-100000))
def test_refine_root():
    # 创建一个二次多项式对象 f = x^2 - 2
    f = Poly(x**2 - 2)

    # 测试精确到步数为0时的根的细化操作
    assert f.refine_root(1, 2, steps=0) == (1, 2)
    assert f.refine_root(-2, -1, steps=0) == (-2, -1)

    # 测试使用步数为None时的根的细化操作
    assert f.refine_root(1, 2, steps=None) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=None) == (Rational(-3, 2), -1)

    # 测试使用指定步数进行根的细化操作
    assert f.refine_root(1, 2, steps=1) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=1) == (Rational(-3, 2), -1)

    # 测试使用步数为1且启用快速模式时的根的细化操作
    assert f.refine_root(1, 2, steps=1, fast=True) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=1, fast=True) == (Rational(-3, 2), -1)

    # 测试使用指定的精度 eps 进行根的细化操作
    assert f.refine_root(1, 2, eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert f.refine_root(1, 2, eps=1e-2) == (Rational(24, 17), Rational(17, 12))

    # 测试对非方程的多项式使用 refine_root 时抛出 PolynomialError 异常
    raises(PolynomialError, lambda: (f**2).refine_root(1, 2, check_sqf=True))

    # 测试在无法细化根的情况下是否抛出 RefinementFailed 异常
    raises(RefinementFailed, lambda: (f**2).refine_root(1, 2))
    raises(RefinementFailed, lambda: (f**2).refine_root(2, 3))

    # 将 f 设置为普通的多项式 x^2 - 2
    f = x**2 - 2

    # 测试使用 refine_root 函数进行根的细化操作
    assert refine_root(f, 1, 2, steps=1) == (1, Rational(3, 2))
    assert refine_root(f, -2, -1, steps=1) == (Rational(-3, 2), -1)

    # 测试使用 refine_root 函数且启用快速模式时进行根的细化操作
    assert refine_root(f, 1, 2, steps=1, fast=True) == (1, Rational(3, 2))
    assert refine_root(f, -2, -1, steps=1, fast=True) == (Rational(-3, 2), -1)

    # 测试使用指定的精度 eps 进行根的细化操作
    assert refine_root(f, 1, 2, eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert refine_root(f, 1, 2, eps=1e-2) == (Rational(24, 17), Rational(17, 12))

    # 测试对精度过低的情况下是否抛出 ValueError 异常
    raises(PolynomialError, lambda: refine_root(1, 7, 8, eps=Rational(1, 100)))
    raises(ValueError, lambda: Poly(f).refine_root(1, 2, eps=10**-100000))
    raises(ValueError, lambda: refine_root(f, 1, 2, eps=10**-100000))


def test_count_roots():
    # 测试计算 x^2 - 2 的根的个数
    assert count_roots(x**2 - 2) == 2

    # 测试在无穷大区间 (-oo, +oo) 内计算 x^2 - 2 的根的个数
    assert count_roots(x**2 - 2, inf=-oo) == 2
    assert count_roots(x**2 - 2, sup=+oo) == 2
    assert count_roots(x**2 - 2, inf=-oo, sup=+oo) == 2

    # 测试在指定区间内计算 x^2 - 2 的根的个数
    assert count_roots(x**2 - 2, inf=-2) == 2
    assert count_roots(x**2 - 2, inf=-1) == 1

    assert count_roots(x**2 - 2, sup=1) == 1
    assert count_roots(x**2 - 2, sup=2) == 2

    assert count_roots(x**2 - 2, inf=-1, sup=1) == 0
    assert count_roots(x**2 - 2, inf=-2, sup=2) == 2

    assert count_roots(x**2 + 2) == 0
    assert count_roots(x**2 + 2, inf=-2*I) == 2
    assert count_roots(x**2 + 2, sup=+2*I) == 2
    assert count_roots(x**2 + 2, inf=-2*I, sup=+2*I) == 2

    assert count_roots(x**2 + 2, inf=0) == 0
    assert count_roots(x**2 + 2, sup=0) == 0

    assert count_roots(x**2 + 2, inf=-I) == 1
    assert count_roots(x**2 + 2, sup=+I) == 1

    assert count_roots(x**2 + 2, inf=+I/2, sup=+I) == 0
    assert count_roots(x**2 + 2, inf=-I, sup=-I/2) == 0

    # 测试对于常数多项式是否抛出 PolynomialError 异常
    raises(PolynomialError, lambda: count_roots(1))


def test_Poly_root():
    # 创建一个三次多项式对象 f = 2*x^3 - 7*x^2 + 4*x + 4
    f = Poly(2*x**3 - 7*x**2 + 4*x + 4)

    # 测试计算 f 的根在指定索引处的值
    assert f.root(0) == Rational(-1, 2)
    assert f.root(1) == 2
    assert f.root(2) == 2
    # 调用函数 raises 来测试是否会抛出 IndexError 异常，使用 lambda 匿名函数调用 f.root(3)
    raises(IndexError, lambda: f.root(3))
    
    # 使用 assert 语句来验证多项式 Poly(x**5 + x + 1) 在给定根索引下的值是否等于方程 rootof(x**3 - x**2 + 1, 0) 的根
    assert Poly(x**5 + x + 1).root(0) == rootof(x**3 - x**2 + 1, 0)
def test_real_roots():
    # 检查对于 x，返回其实根为 [0]
    assert real_roots(x) == [0]
    # 检查对于 x，使用 multiple=False 参数，返回其实根为 [(0, 1)]
    assert real_roots(x, multiple=False) == [(0, 1)]

    # 检查对于 x^3，返回其实根为 [0, 0, 0]
    assert real_roots(x**3) == [0, 0, 0]
    # 检查对于 x^3，使用 multiple=False 参数，返回其实根为 [(0, 3)]
    assert real_roots(x**3, multiple=False) == [(0, 3)]

    # 检查对于 x*(x^3 + x + 3)，返回其实根为 [rootof(x^3 + x + 3, 0), 0]
    assert real_roots(x*(x**3 + x + 3)) == [rootof(x**3 + x + 3, 0), 0]
    # 检查对于 x*(x^3 + x + 3)，使用 multiple=False 参数，返回其实根为 [(rootof(x^3 + x + 3, 0), 1), (0, 1)]
    assert real_roots(x*(x**3 + x + 3), multiple=False) == [(rootof(x**3 + x + 3, 0), 1), (0, 1)]

    # 检查对于 x^3*(x^3 + x + 3)，返回其实根为 [rootof(x^3 + x + 3, 0), 0, 0, 0]
    assert real_roots(x**3*(x**3 + x + 3)) == [rootof(x**3 + x + 3, 0), 0, 0, 0]
    # 检查对于 x^3*(x^3 + x + 3)，使用 multiple=False 参数，返回其实根为 [(rootof(x^3 + x + 3, 0), 1), (0, 3)]
    assert real_roots(x**3*(x**3 + x + 3), multiple=False) == [(rootof(x**3 + x + 3, 0), 1), (0, 3)]

    # 检查对于 x^2 - 2，不使用根式求解，返回其实根为 [-sqrt(2), sqrt(2)]
    assert real_roots(x**2 - 2, radicals=False) == [
            rootof(x**2 - 2, 0, radicals=False),
            rootof(x**2 - 2, 1, radicals=False),
        ]

    f = 2*x**3 - 7*x**2 + 4*x + 4
    g = x**3 + x + 1

    # 检查对于 Poly(f).real_roots()，返回其实根为 [Rational(-1, 2), 2, 2]
    assert Poly(f).real_roots() == [Rational(-1, 2), 2, 2]
    # 检查对于 Poly(g).real_roots()，返回其实根为 [rootof(g, 0)]
    assert Poly(g).real_roots() == [rootof(g, 0)]


def test_all_roots():
    f = 2*x**3 - 7*x**2 + 4*x + 4
    froots = [Rational(-1, 2), 2, 2]
    # 检查对于 all_roots(f)，返回其所有根与 Poly(f).all_roots() 相同，均为 froots
    assert all_roots(f) == Poly(f).all_roots() == froots

    g = x**3 + x + 1
    groots = [rootof(g, 0), rootof(g, 1), rootof(g, 2)]
    # 检查对于 all_roots(g)，返回其所有根与 Poly(g).all_roots() 相同，均为 groots
    assert all_roots(g) == Poly(g).all_roots() == groots

    # 检查对于 x^2 - 2，返回其所有根为 [-sqrt(2), sqrt(2)]
    assert all_roots(x**2 - 2) == [-sqrt(2), sqrt(2)]
    # 检查对于 x^2 - 2，使用 multiple=False 参数，返回其所有根为 [(-sqrt(2), 1), (sqrt(2), 1)]
    assert all_roots(x**2 - 2, multiple=False) == [(-sqrt(2), 1), (sqrt(2), 1)]
    # 检查对于 x^2 - 2，不使用根式求解，返回其所有根为 [rootof(x^2 - 2, 0, radicals=False), rootof(x^2 - 2, 1, radicals=False)]
    assert all_roots(x**2 - 2, radicals=False) == [
        rootof(x**2 - 2, 0, radicals=False),
        rootof(x**2 - 2, 1, radicals=False),
    ]

    p = x**5 - x - 1
    # 检查对于 all_roots(p)，返回其所有根为 [rootof(p, 0), rootof(p, 1), rootof(p, 2), rootof(p, 3), rootof(p, 4)]
    assert all_roots(p) == [
        rootof(p, 0), rootof(p, 1), rootof(p, 2), rootof(p, 3), rootof(p, 4)
    ]


def test_nroots():
    # 检查对于 Poly(0, x).nroots()，返回空列表
    assert Poly(0, x).nroots() == []
    # 检查对于 Poly(1, x).nroots()，返回空列表
    assert Poly(1, x).nroots() == []

    # 检查对于 Poly(x^2 - 1, x).nroots()，返回其数值根为 [-1.0, 1.0]
    assert Poly(x**2 - 1, x).nroots() == [-1.0, 1.0]
    # 检查对于 Poly(x^2 + 1, x).nroots()，返回其复数根为 [-1.0*I, 1.0*I]
    assert Poly(x**2 + 1, x).nroots() == [-1.0*I, 1.0*I]

    roots = Poly(x**2 - 1, x).nroots()
    # 检查 roots == [-1.0, 1.0]
    assert roots == [-1.0, 1.0]

    roots = Poly(x**2 + 1, x).nroots()
    # 检查 roots == [-1.0*I, 1.0*I]
    assert roots == [-1.0*I, 1.0*I]

    roots = Poly(x**2/3 - Rational(1, 3), x).nroots()
    # 检查 roots == [-1.0, 1.0]
    assert roots == [-1.0, 1.0]

    roots = Poly(x**2/3 + Rational(1, 3), x).nroots()
    # 检查 roots == [-1.0*I, 1.0*I]
    assert roots == [-1.0*I, 1.0*I]

    # 检查对于 Poly(x^2 + 2*I, x).nroots()，返回其数值根为 [-1.0 + 1.0*I, 1.0 - 1.0*I]
    assert Poly(x**2 + 2*I, x).nroots() == [-1.0 + 1.0*I, 1.0 - 1.0*I]
    # 检查对于 Poly(x^2 + 2*I, x, extension=I).nroots()，返回其数值根为 [-1.0 + 1.0*I, 1.0 - 1.0*I]
    assert Poly(x**2 + 2*I, x, extension=I).nroots() == [-1.0 + 1.0*I, 1.0 - 1.0*I]

    # 检查对于 Poly(0.2*x + 0.1).nroots()，返回其数值根为 [-0.5]
    assert Poly(0.2*x + 0.1).nroots() == [-0.5]

    roots = nroots(x**5 + x + 1, n=5)
    eps = Float("1e-5")

    # 检查 re(roots[0]).epsilon_eq(-0.754
    # 对第一个根进行断言，检查其实部接近于 -0.75487，误差小于 eps，并且虚部为假
    assert re(roots[0]).epsilon_eq(-0.75487, eps) is S.false
    # 对第一个根进行断言，检查其虚部为 0.0
    assert im(roots[0]) == 0.0
    # 对第二个根进行断言，检查其实部在精度为 5 时等于 -0.5
    assert re(roots[1]) == Float(-0.5, 5)
    # 对第二个根进行断言，检查其虚部接近于 -0.86602，误差小于 eps，并且虚部为假
    assert im(roots[1]).epsilon_eq(-0.86602, eps) is S.false
    # 对第三个根进行断言，检查其实部在精度为 5 时等于 -0.5
    assert re(roots[2]) == Float(-0.5, 5)
    # 对第三个根进行断言，检查其虚部接近于 +0.86602，误差小于 eps，并且虚部为假
    assert im(roots[2]).epsilon_eq(+0.86602, eps) is S.false
    # 对第四个根进行断言，检查其实部接近于 +0.87743，误差小于 eps，并且虚部为假
    assert re(roots[3]).epsilon_eq(+0.87743, eps) is S.false
    # 对第四个根进行断言，检查其虚部接近于 -0.74486，误差小于 eps，并且虚部为假
    assert im(roots[3]).epsilon_eq(-0.74486, eps) is S.false
    # 对第五个根进行断言，检查其实部接近于 +0.87743，误差小于 eps，并且虚部为假
    assert re(roots[4]).epsilon_eq(+0.87743, eps) is S.false
    # 对第五个根进行断言，检查其虚部接近于 +0.74486，误差小于 eps，并且虚部为假
    assert im(roots[4]).epsilon_eq(+0.74486, eps) is S.false

    # 断言在多项式定义域上抛出 DomainError 异常，当尝试使用 x + y 进行多项式根查找时
    raises(DomainError, lambda: Poly(x + y, x).nroots())
    # 断言在多变量多项式上抛出 MultivariatePolynomialError 异常，当尝试使用 x + y 进行多项式根查找时
    raises(MultivariatePolynomialError, lambda: Poly(x + y).nroots())

    # 断言对 x^2 - 1 进行数值根查找得到的结果为 [-1.0, 1.0]
    assert nroots(x**2 - 1) == [-1.0, 1.0]

    # 将 x^2 - 1 的数值根存储在 roots 变量中，并断言其值为 [-1.0, 1.0]
    roots = nroots(x**2 - 1)
    assert roots == [-1.0, 1.0]

    # 断言对 x + I 进行数值根查找得到的结果为 [-1.0*I]
    assert nroots(x + I) == [-1.0*I]
    # 断言对 x + 2*I 进行数值根查找得到的结果为 [-2.0*I]
    assert nroots(x + 2*I) == [-2.0*I]

    # 断言在多项式定义域上抛出 PolynomialError 异常，当尝试对常数 0 进行数值根查找时
    raises(PolynomialError, lambda: nroots(0))

    # issue 8296 的问题：创建多项式 x^4 - 1，并对其进行根查找和数值计算，断言结果与预期一致
    f = Poly(x**4 - 1)
    assert f.nroots(2) == [w.n(2) for w in f.all_roots()]

    # 断言对大数值多项式的数值根查找结果的字符串表示与预期一致
    assert str(Poly(x**16 + 32*x**14 + 508*x**12 + 5440*x**10 +
        39510*x**8 + 204320*x**6 + 755548*x**4 + 1434496*x**2 +
        877969).nroots(2)) == ('[-1.7 - 1.9*I, -1.7 + 1.9*I, -1.7 '
        '- 2.5*I, -1.7 + 2.5*I, -1.0*I, 1.0*I, -1.7*I, 1.7*I, -2.8*I, '
        '2.8*I, -3.4*I, 3.4*I, 1.7 - 1.9*I, 1.7 + 1.9*I, 1.7 - 2.5*I, '
        '1.7 + 2.5*I]')

    # 断言对非常小的多项式 1e-15*x^2 - 1 进行数值根查找的结果的字符串表示与预期一致
    assert str(Poly(1e-15*x**2 -1).nroots()) == ('[-31622776.6016838, 31622776.6016838]')
# 定义测试函数 test_ground_roots，用于测试多项式的根的计算
def test_ground_roots():
    # 定义一个多项式 f = x^6 - 4*x^4 + 4*x^3 - x^2
    f = x**6 - 4*x**4 + 4*x**3 - x**2

    # 断言使用 Poly 类的 ground_roots 方法计算多项式的地根，应该得到 {1: 2, 0: 2} 的结果
    assert Poly(f).ground_roots() == {S.One: 2, S.Zero: 2}
    # 断言直接调用 ground_roots 函数计算多项式的地根，应该得到 {1: 2, 0: 2} 的结果
    assert ground_roots(f) == {S.One: 2, S.Zero: 2}


# 定义测试函数 test_nth_power_roots_poly，用于测试多项式的 n 次幂根分解
def test_nth_power_roots_poly():
    # 定义一个多项式 f = x^4 - x^2 + 1
    f = x**4 - x**2 + 1

    # 定义几个预期的 n 次幂根分解结果
    f_2 = (x**2 - x + 1)**2
    f_3 = (x**2 + 1)**2
    f_4 = (x**2 + x + 1)**2
    f_12 = (x - 1)**4

    # 断言计算 f 的一次幂根分解应该得到 f 本身
    assert nth_power_roots_poly(f, 1) == f

    # 断言传递非法的 n 值应该引发 ValueError 异常
    raises(ValueError, lambda: nth_power_roots_poly(f, 0))
    raises(ValueError, lambda: nth_power_roots_poly(f, x))

    # 断言计算 f 的不同 n 值的幂根分解结果应该与预期的结果相等
    assert factor(nth_power_roots_poly(f, 2)) == f_2
    assert factor(nth_power_roots_poly(f, 3)) == f_3
    assert factor(nth_power_roots_poly(f, 4)) == f_4
    assert factor(nth_power_roots_poly(f, 12)) == f_12

    # 断言对于多变量多项式应该引发 MultivariatePolynomialError 异常
    raises(MultivariatePolynomialError, lambda: nth_power_roots_poly(
        x + y, 2, x, y))


# 定义测试函数 test_same_root，用于测试多项式的相同根检测
def test_same_root():
    # 定义一个多项式 f = x^4 + x^3 + x^2 + x + 1
    f = Poly(x**4 + x**3 + x**2 + x + 1)
    # 获取 f 的 same_root 方法
    eq = f.same_root
    # 定义一个复数根 r0
    r0 = exp(2 * I * pi / 5)
    # 断言返回所有根中与 r0 相同的根的索引应该是 [3]
    assert [i for i, r in enumerate(f.all_roots()) if eq(r, r0)] == [3]

    # 断言对于有理数域的多项式应该引发 PolynomialError 异常
    raises(PolynomialError,
           lambda: Poly(x + 1, domain=QQ).same_root(0, 0))
    # 断言对于有限域的多项式应该引发 DomainError 异常
    raises(DomainError,
           lambda: Poly(x**2 + 1, domain=FF(7)).same_root(0, 0))
    raises(DomainError,
           lambda: Poly(x ** 2 + 1, domain=ZZ_I).same_root(0, 0))
    raises(DomainError,
           lambda: Poly(y * x**2 + 1, domain=ZZ[y]).same_root(0, 0))
    # 断言对于多变量多项式应该引发 MultivariatePolynomialError 异常
    raises(MultivariatePolynomialError,
           lambda: Poly(x * y + 1, domain=ZZ).same_root(0, 0))


# 定义测试函数 test_torational_factor_list，用于测试有理因式列表的计算
def test_torational_factor_list():
    # 计算多项式 p = ((x^2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}) 的有理因式列表
    p = expand(((x**2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}))
    # 断言计算出的有理因式列表应该与预期结果 (-2, [(-x*(1 + sqrt(2))/2 + 1, 1), (-x*(1 + sqrt(2)) - 1, 1), (-x*(1 + sqrt(2)) + 1, 1)]) 相等
    assert _torational_factor_list(p, x) == (-2, [
        (-x*(1 + sqrt(2))/2 + 1, 1),
        (-x*(1 + sqrt(2)) - 1, 1),
        (-x*(1 + sqrt(2)) + 1, 1)])

    # 计算多项式 p = ((x^2-1)*(x-2)).subs({x:x*(1 + 2**(1/4))}) 的有理因式列表
    # 断言计算出的有理因式列表应该为 None
    assert _torational_factor_list(p, x) is None


# 定义测试函数 test_cancel，用于测试多项式的因式消除
def test_cancel():
    # 断言对于 0，7，x 的因式消除结果应该分别为 0，7，x
    assert cancel(0) == 0
    assert cancel(7) == 7
    assert cancel(x) == x

    # 断言对于无穷大 oo 的因式消除结果应该为 oo
    assert cancel(oo) is oo

    # 断言对于元组 (2, 3) 的因式消除结果应该为 (1, 2, 3)
    assert cancel((2, 3)) == (1, 2, 3)

    # 断言对于元组 (1, 0) 的因式消除结果应该为 (1, 1, 0)
    assert cancel((1, 0), x) == (1, 1, 0)
    assert cancel((0, 1), x) == (1, 0, 1)

    # 定义一些多项式对象
    f, g, p, q = 4*x**2 - 4, 2*x - 2, 2*x + 2, 1
    F, G, P, Q = [ Poly(u, x) for u in (f, g, p, q) ]

    # 断言多项式对象的因式消除结果应该与预期结果相等
    assert F.cancel(G) == (1, P, Q)
    assert cancel((f, g)) == (1, p, q)
    assert cancel((f, g), x) == (1, p, q)
    assert cancel((f, g), (x,)) == (1, p, q)
    assert cancel((F, G)) == (1, P, Q)
    assert cancel((f, g), polys=True) == (1, P, Q)
    assert cancel((F, G), polys=False) == (1, p, q)

    # 定义有理函数 f = (x^2 - 2)/(x + sqrt(2))
    f = (x**2 - 2)/(x + sqrt(2))

    # 断言对于 f 的因式消除结果应该是 f 本身
    assert cancel(f) == f
    # 断言对于 f 的非贪婪因式消除结果应该是 x - sqrt(2)
    assert cancel(f, greedy=False) == x - sqrt(2)

    # 定义有理函数 f = (x^2 - 2)/(x - sqrt(2))
    f = (x**2 - 2)/(x - sqrt(2))

    # 断言对于 f 的因式消除结果应该是 f 本身
    assert cancel(f) == f
    # 断言对于 f 的非贪婪因式消除结果应该是 x + sqrt(2)
    assert cancel(f
    # 断言语句，验证 cancel 函数对表达式 (x**2 - y**2)/(x - y) 的处理结果是否等于 x + y
    assert cancel((x**2 - y**2)/(x - y)) == x + y

    # 断言语句，验证 cancel 函数对表达式 (x**3 - 1)/(x**2 - 1) 的处理结果是否等于 (x**2 + x + 1)/(x + 1)
    assert cancel((x**3 - 1)/(x**2 - 1)) == (x**2 + x + 1)/(x + 1)

    # 断言语句，验证 cancel 函数对表达式 (exp(2*x) + 2*exp(x) + 1)/(exp(x) + 1) 的处理结果是否等于 exp(x) + 1
    assert cancel((exp(2*x) + 2*exp(x) + 1)/(exp(x) + 1)) == exp(x) + 1

    # 创建多项式对象 f 和 g
    f = Poly(x**2 - a**2, x)
    g = Poly(x - a, x)

    # 创建在整数环 ZZ[a] 上的多项式对象 F 和 G
    F = Poly(x + a, x, domain='ZZ[a]')
    G = Poly(1, x, domain='ZZ[a]')

    # 断言语句，验证 cancel 函数对 (f, g) 元组的处理结果是否等于 (1, F, G)
    assert cancel((f, g)) == (1, F, G)

    # 创建多项式对象 f 和 g，包含根号 2 的常数
    f = x**3 + (sqrt(2) - 2)*x**2 - (2*sqrt(2) + 3)*x - 3*sqrt(2)
    g = x**2 - 2

    # 断言语句，验证 cancel 函数对带有扩展参数的 (f, g) 元组的处理结果是否符合预期
    assert cancel((f, g), extension=True) == (1, x**2 - 2*x - 3, x - sqrt(2))

    # 创建多项式对象 f 和 g
    f = Poly(-2*x + 3, x)
    g = Poly(-x**9 + x**8 + x**6 - x**5 + 2*x**2 - 3*x + 1, x)

    # 断言语句，验证 cancel 函数对 (f, g) 元组的处理结果是否等于 (1, -f, -g)
    assert cancel((f, g)) == (1, -f, -g)

    # 创建有理数环 QQ 上的多项式对象 f 和 g
    f = Poly(x/3 + 1, x)
    g = Poly(x/7 + 1, x)

    # 断言语句，验证 f.cancel(g) 方法的处理结果是否等于给定的元组
    assert f.cancel(g) == (S(7)/3,
        Poly(x + 3, x, domain=QQ),
        Poly(x + 7, x, domain=QQ))

    # 断言语句，验证 f.cancel(g, include=True) 方法的处理结果是否等于给定的元组
    assert f.cancel(g, include=True) == (
        Poly(7*x + 21, x, domain=QQ),
        Poly(3*x + 21, x, domain=QQ))

    # 创建符号 y 上的多项式对象 f 和 g
    f = Poly(y, y, domain='ZZ(x)')
    g = Poly(1, y, domain='ZZ[x]')

    # 断言语句，验证 f.cancel(g) 方法的处理结果是否等于给定的元组
    assert f.cancel(g) == (1, Poly(y, y, domain='ZZ(x)'), Poly(1, y, domain='ZZ(x)'))

    # 断言语句，验证 f.cancel(g, include=True) 方法的处理结果是否等于给定的元组
    assert f.cancel(g, include=True) == (
        Poly(y, y, domain='ZZ(x)'),
        Poly(1, y, domain='ZZ(x)'))

    # 创建在 ZZ(x) 上的多项式对象 f 和 g
    f = Poly(5*x*y + x, y, domain='ZZ(x)')
    g = Poly(2*x**2*y, y, domain='ZZ(x)')

    # 断言语句，验证 f.cancel(g, include=True) 方法的处理结果是否等于给定的元组
    assert f.cancel(g, include=True) == (
        Poly(5*y + 1, y, domain='ZZ(x)'),
        Poly(2*x*y, y, domain='ZZ(x)'))

    # 创建数学表达式 f
    f = -(-2*x - 4*y + 0.005*(z - y)**2)/((z - y)*(-z + y + 2))

    # 断言语句，验证 cancel 函数对 f 的处理结果的 is_Mul 属性是否为 True
    assert cancel(f).is_Mul == True

    # 创建 tanh 函数表达式 P 和 Q
    P = tanh(x - 3.0)
    Q = tanh(x + 3.0)

    # 创建复杂的数学表达式 f
    f = ((-2*P**2 + 2)*(-P**2 + 1)*Q**2/2 + (-2*P**2 + 2)*(-2*Q**2 + 2)*P*Q - (-2*P**2 + 2)*P**2*Q**2 + (-2*Q**2 + 2)*(-Q**2 + 1)*P**2/2 - (-2*Q**2 + 2)*P**2*Q**2)/(2*sqrt(P**2*Q**2 + 0.0001)) \
      + (-(-2*P**2 + 2)*P*Q**2/2 - (-2*Q**2 + 2)*P**2*Q/2)*((-2*P**2 + 2)*P*Q**2/2 + (-2*Q**2 + 2)*P**2*Q/2)/(2*(P**2*Q**2 + 0.0001)**Rational(3, 2))

    # 断言语句，验证 cancel 函数对 f 的处理结果的 is_Mul 属性是否为 True
    assert cancel(f).is_Mul == True

    # 创建非交换符号 A
    A = Symbol('A', commutative=False)

    # 创建分段函数 p1 和 p2
    p1 = Piecewise((A*(x**2 - 1)/(x + 1), x > 1), ((x + 2)/(x**2 + 2*x), True))
    p2 = Piecewise((A*(x - 1), x > 1), (1/x, True))

    # 断言语句，验证 cancel 函数对 p1 的处理结果是否等于 p2
    assert cancel(p1) == p2

    # 断言语句，验证 cancel 函数对 2*p1 的处理结果是否等于 2*p2
    assert cancel(2*p1) == 2*p2

    # 断言语句，验证 cancel 函数对 1 + p1 的处理结果是否等于 1 + p2
    assert cancel(1 + p1) == 1 + p2

    # 断言语句，验证 cancel 函数对 (x**2 - 1)/(x + 1)*p1 的处理结果是否等于 (x - 1)*p2
    assert cancel((x**2 - 1)/(x + 1)*p1) == (x - 1)*p2

    # 断言语句，验证 cancel 函数对 (x**2 - 1)/(x + 1) + p1 的处理结果是否等于 (x - 1) + p2
    assert cancel((x**2 - 1)/(x + 1) + p1) == (x - 1) + p2
    # 创建分段函数 p3 和 p4
    p3 = Piecewise(((x**2 - 1)/(x + 1), x > 1), ((x + 2)/(x**2 + 2*x), True))
    p4 = Piecewise(((x - 1), x > 1), (1/x, True))
    
    # 使用 cancel 函数验证 p3 是否能化简为 p4
    assert cancel(p3) == p4
    
    # 使用 cancel 函数验证 2*p3 是否能化简为 2*p4
    assert cancel(2*p3) == 2*p4
    
    # 使用 cancel 函数验证 1 + p3 是否能化简为 1 + p4
    assert cancel(1 + p3) == 1 + p4
    
    # 使用 cancel 函数验证 (x**2 - 1)/(x + 1)*p3 是否能化简为 (x - 1)*p4
    assert cancel((x**2 - 1)/(x + 1)*p3) == (x - 1)*p4
    
    # 使用 cancel 函数验证 (x**2 - 1)/(x + 1) + p3 是否能化简为 (x - 1) + p4
    assert cancel((x**2 - 1)/(x + 1) + p3) == (x - 1) + p4
    
    # issue 4077 的长表达式，使用 S 函数创建符号表达式 q，不进行求值
    q = S('''(2*1*(x - 1/x)/(x*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x -
        1/x)) - 2/x)) - 2*1*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))*((-x + 1/x)*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x - 1/x)))/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) -
        2/x) + 1)*((x - 1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x - 1/x)**2)) -
        1/(x*(x - 1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x
        - 1/x)) - 2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) -
        1/(x**2*(x - 1/x)) - 2/x)/x - 1/x)*(((-x + 1/x)/((x*(x - 1/x)**2)) +
        1/(x*(x - 1/x)))*((-(x - 1/x)/(x*(x - 1/x)) - 1/x)*((x - 1/x)/((x*(x -
        1/x)**2)) - 1/(x*(x - 1/x)))/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) -
        1/(x**2*(x - 1/x)) - 2/x) - 1 + (x - 1/x)/(x - 1/x))/((x*((x -
        1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) -
        2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x
        - 1/x)) - 2/x))) + ((x - 1/x)/((x*(x - 1/x))) + 1/x)/((x*(2*x - (-x +
        1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x))) + 1/x)/(2*x +
        2*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x - 1/x)))*((-(x - 1/x)/(x*(x
        - 1/x)) - 1/x)*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x - 1/x)))/(2*x -
        (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x) - 1 + (x -
        1/x)/(x - 1/x))/((x*((x - 1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x -
        1/x)**2)) - 1/(x*(x - 1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2)
        - 1/(x**2*(x - 1/x)) - 2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x
        - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x))) - 2*((x - 1/x)/((x*(x -
        1/x))) + 1/x)/(x*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x -
        1/x)) - 2/x)) - 2/x) - ((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))*((-x + 1/x)*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) -
        2/x) + 1)/(x*((x - 1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x - 1/x)**2))
        - 1/(x*(x - 1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) -
        1/(x**2*(x - 1/x)) - 2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x -
        1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x)) + (x - 1/x)/((x*(2*x - (-x +
        1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x))) - 1/x''',
        evaluate=False)
    
    # 使用 cancel 函数验证 q 在不进行符号简化时是否为 S.NaN
    assert cancel(q, _signsimp=False) is S.NaN
    
    # 使用 subs 方法验证 q 在 x=2 处的值是否为 S.NaN
    assert q.subs(x, 2) is S.NaN
    
    # 使用 signsimp 函数验证 q 是否能被简化为 S.NaN
    assert signsimp(q) is S.NaN
    
    # issue 9363 中创建一个符号矩阵 M，形状为 5x5
    M = MatrixSymbol('M', 5, 5)
    # 断言：验证 cancel 函数对 M 矩阵第一个元素加 7 的结果是否与原值相等
    assert cancel(M[0,0] + 7) == M[0,0] + 7
    
    # 创建数学表达式，包括 sin 函数和乘法运算，其中 M 矩阵的特定元素参与计算
    expr = sin(M[1, 4] + M[2, 1] * 5 * M[4, 0]) - 5 * M[1, 2] / z
    
    # 断言：验证 cancel 函数对 expr 表达式的化简是否与预期相等
    assert cancel(expr) == (z*sin(M[1, 4] + M[2, 1] * 5 * M[4, 0]) - 5 * M[1, 2]) / z
    
    # 断言：验证 cancel 函数对复杂表达式 (x**2 + 1)/(x - I) 的化简是否正确
    assert cancel((x**2 + 1)/(x - I)) == x + I
# 定义一个测试函数，用于测试多项式在整数上通过缩放根变成首一多项式的功能
def test_make_monic_over_integers_by_scaling_roots():
    # 创建一个整数环上的二次多项式 f = x**2 + 3*x + 4
    f = Poly(x**2 + 3*x + 4, x, domain='ZZ')
    # 调用函数使 f 变为整数环上的首一多项式，返回值 g 和缩放因子 c
    g, c = f.make_monic_over_integers_by_scaling_roots()
    # 断言 g 应该等于 f
    assert g == f
    # 断言缩放因子 c 应该为整数环的单位元素
    assert c == ZZ.one

    # 创建一个有理数环上的二次多项式 f = x**2 + 3*x + 4
    f = Poly(x**2 + 3*x + 4, x, domain='QQ')
    # 调用函数使 f 变为整数环上的首一多项式，返回值 g 和缩放因子 c
    g, c = f.make_monic_over_integers_by_scaling_roots()
    # 断言 g 应该等于 f 转换为整数环上的多项式
    assert g == f.to_ring()
    # 断言缩放因子 c 应该为整数环的单位元素
    assert c == ZZ.one

    # 创建一个有理数环上的二次多项式 f = x**2/2 + 1/4*x + 1/8
    f = Poly(x**2/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')
    # 调用函数使 f 变为整数环上的首一多项式，返回值 g 和缩放因子 c
    g, c = f.make_monic_over_integers_by_scaling_roots()
    # 断言 g 应该等于整数环上的三次多项式 x**2 + 2*x + 4
    assert g == Poly(x**2 + 2*x + 4, x, domain='ZZ')
    # 断言缩放因子 c 应该为 4
    assert c == 4

    # 创建一个有理数环上的三次多项式 f = x**3/2 + 1/4*x + 1/8
    f = Poly(x**3/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')
    # 调用函数使 f 变为整数环上的首一多项式，返回值 g 和缩放因子 c
    g, c = f.make_monic_over_integers_by_scaling_roots()
    # 断言 g 应该等于整数环上的四次多项式 x**3 + 8*x + 16
    assert g == Poly(x**3 + 8*x + 16, x, domain='ZZ')
    # 断言缩放因子 c 应该为 4
    assert c == 4

    # 创建一个关于 x 和 y 的多项式 f = x*y
    f = Poly(x*y, x, y)
    # 断言调用此函数会抛出 ValueError 异常
    raises(ValueError, lambda: f.make_monic_over_integers_by_scaling_roots())

    # 创建一个关于 x 的多项式 f = x，定义在实数域上
    f = Poly(x, domain='RR')
    # 断言调用此函数会抛出 ValueError 异常
    raises(ValueError, lambda: f.make_monic_over_integers_by_scaling_roots())


# 定义一个测试函数，测试多项式的 Galois 群
def test_galois_group():
    # 创建一个四次多项式 f = x**4 - 2
    f = Poly(x ** 4 - 2)
    # 调用函数计算 f 的 Galois 群 G 和备选值 alt
    G, alt = f.galois_group(by_name=True)
    # 断言计算得到的 Galois 群 G 应该是 S4 中的一个转置子群 D4
    assert G == S4TransitiveSubgroups.D4
    # 断言备选值 alt 应该为 False


# 定义一个测试函数，测试多项式的约化函数
def test_reduced():
    # 创建一个多项式 f = 2*x**4 + y**2 - x**2 + y**3
    f = 2*x**4 + y**2 - x**2 + y**3
    # 定义一个基底 G = [x**3 - x, y**3 - y]
    G = [x**3 - x, y**3 - y]

    # 定义预期的商 Q 和余式 r
    Q = [2*x, 1]
    r = x**2 + y**2 + y

    # 断言调用 reduced 函数得到的结果与预期相符
    assert reduced(f, G) == (Q, r)
    assert reduced(f, G, x, y) == (Q, r)

    # 计算基底 G 的格罗布纳基 H
    H = groebner(G)

    # 断言使用 H 对 f 进行约化的结果与预期相符
    assert H.reduce(f) == (Q, r)

    # 定义使用多项式形式的预期结果 Q 和 r
    Q = [Poly(2*x, x, y), Poly(1, x, y)]
    r = Poly(x**2 + y**2 + y, x, y)

    # 断言使用 reduced 函数得到的多项式形式结果与预期相符
    assert _strict_eq(reduced(f, G, polys=True), (Q, r))
    assert _strict_eq(reduced(f, G, x, y, polys=True), (Q, r))

    # 计算基底 G 的多项式形式的格罗布纳基 H
    H = groebner(G, polys=True)

    # 断言使用 H 对 f 进行约化的多项式形式结果与预期相符
    assert _strict_eq(H.reduce(f), (Q, r))

    # 创建一个多项式 f = 2*x**3 + y**3 + 3*y
    f = 2*x**3 + y**3 + 3*y
    # 计算基底 G = groebner([x**2 + y**2 - 1, x*y - 2])
    G = groebner([x**2 + y**2 - 1, x*y - 2])

    # 定义预期的商 Q 和余式 r
    Q = [x**2 - x*y**3/2 + x*y/2 + y**6/4 - y**4/2 + y**2/4, -y**5/4 + y**3/2 + y*Rational(3, 4)]
    r = 0

    # 断言使用 reduced 函数得到的结果与预期相符
    assert reduced(f, G) == (Q, r)
    assert G.reduce(f) == (Q, r)

    # 断言使用 reduced 函数设置 auto=False 时得到的余式不为零
    assert reduced(f, G, auto=False)[1] != 0
    assert G.reduce(f, auto=False)[1] != 0

    # 断言基底 G 包含多项式 f
    assert G.contains(f) is True
    assert G.contains(f + 1) is False

    # 断言对于常数 1 使用 reduced 函数的结果
    assert reduced(1, [1], x) == ([1], 0)
    # 断言调用 reduced 函数时会抛出 ComputationFailed 异常
    raises(ComputationFailed, lambda: reduced(1, [1]))


# 定义一个测试函数，测试格罗布纳基的计算
def test_groebner():
    # 断言对于空基底，计算得到的格罗布纳基应该为空列表
    assert groebner([], x, y, z) == []

    # 断言使用 lex 排序计算的格罗布纳基结果与预期相符
    assert groebner([x**2 + 1, y**4*x + x**3], x, y, order='lex') == [1 + x**2, -1 + y**4]
    # 断言使用 grevlex 排序计算的格罗布
    # 使用给定的多项式 F，变量 x, y, z，以及 modulus=7 和 symmetric=False 参数调用 groebner 函数，返回 Groebner 基 G
    G = groebner(F, x, y, z, modulus=7, symmetric=False)

    # 断言检查计算得到的 Groebner 基 G 是否符合预期的多项式列表
    assert G == [1 + x + y + 3*z + 2*z**2 + 2*z**3 + 6*z**4 + z**5,
                 1 + 3*y + y**2 + 6*z**2 + 3*z**3 + 3*z**4 + 3*z**5 + 4*z**6,
                 1 + 4*y + 4*z + y*z + 4*z**3 + z**4 + z**6,
                 6 + 6*z + z**2 + 4*z**3 + 3*z**4 + 6*z**5 + 3*z**6 + z**7]

    # 使用给定的多项式 f, Groebner 基 G, 变量 x, y, z, 以及 modulus=7, symmetric=False, polys=True 参数调用 reduced 函数
    Q, r = reduced(f, G, x, y, z, modulus=7, symmetric=False, polys=True)

    # 断言检查 reduced 函数的计算结果是否符合预期的形式
    assert sum([ q*g for q, g in zip(Q, G.polys)], r) == Poly(f, modulus=7)

    # 重新定义 F，设定新的多项式列表
    F = [x*y - 2*y, 2*y**2 - x**2]

    # 断言检查使用 grevlex 排序得到的 Groebner 基是否符合预期的多项式列表
    assert groebner(F, x, y, order='grevlex') == \
        [y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y]

    # 断言检查使用 grevlex 排序得到的 Groebner 基（y, x 顺序）是否符合预期的多项式列表
    assert groebner(F, y, x, order='grevlex') == \
        [x**3 - 2*x**2, -x**2 + 2*y**2, x*y - 2*y]

    # 断言检查在 field=True 的情况下使用 grevlex 排序得到的 Groebner 基是否符合预期的多项式列表
    assert groebner(F, order='grevlex', field=True) == \
        [y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y]

    # 断言检查仅有常数项的情况下调用 groebner 函数是否返回预期的结果
    assert groebner([1], x) == [1]

    # 断言检查包含浮点数的多项式调用 groebner 函数是否返回预期的结果
    assert groebner([x**2 + 2.0*y], x, y) == [1.0*x**2 + 2.0*y]

    # 使用 lambda 函数和 raises 检查调用 groebner 函数时引发的异常是否符合预期
    raises(ComputationFailed, lambda: groebner([1]))

    # 断言检查使用不同计算方法（'buchberger'）处理给定多项式是否返回预期的结果
    assert groebner([x**2 - 1, x**3 + 1], method='buchberger') == [x + 1]

    # 断言检查使用不同计算方法（'f5b'）处理给定多项式是否返回预期的结果
    assert groebner([x**2 - 1, x**3 + 1], method='f5b') == [x + 1]

    # 使用 lambda 函数和 raises 检查调用 groebner 函数时引发的异常是否符合预期（值错误）
    raises(ValueError, lambda: groebner([x, y], method='unknown'))
def test_fglm():
    # 定义多项式 F
    F = [a + b + c + d, a*b + a*d + b*c + b*d, a*b*c + a*b*d + a*c*d + b*c*d, a*b*c*d - 1]
    # 计算 F 的 Groebner 基，使用 grlex 排序
    G = groebner(F, a, b, c, d, order=grlex)

    # 定义多项式 B
    B = [
        4*a + 3*d**9 - 4*d**5 - 3*d,
        4*b + 4*c - 3*d**9 + 4*d**5 + 7*d,
        4*c**2 + 3*d**10 - 4*d**6 - 3*d**2,
        4*c*d**4 + 4*c - d**9 + 4*d**5 + 5*d,
        d**12 - d**8 - d**4 + 1,
    ]

    # 断言使用 lex 排序计算的 Groebner 基等于 B
    assert groebner(F, a, b, c, d, order=lex) == B
    # 断言 G 对象的 fglm 方法使用 lex 排序得到的结果等于 B
    assert G.fglm(lex) == B

    # 定义新的多项式 F
    F = [9*x**8 + 36*x**7 - 32*x**6 - 252*x**5 - 78*x**4 + 468*x**3 + 288*x**2 - 108*x + 9,
        -72*t*x**7 - 252*t*x**6 + 192*t*x**5 + 1260*t*x**4 + 312*t*x**3 - 404*t*x**2 - 576*t*x + \
        108*t - 72*x**7 - 256*x**6 + 192*x**5 + 1280*x**4 + 312*x**3 - 576*x + 96]
    # 计算 F 的 Groebner 基，使用 grlex 排序
    G = groebner(F, t, x, order=grlex)

    # 定义多项式 B
    B = [
        203577793572507451707*t + 627982239411707112*x**7 - 666924143779443762*x**6 - \
        10874593056632447619*x**5 + 5119998792707079562*x**4 + 72917161949456066376*x**3 + \
        20362663855832380362*x**2 - 142079311455258371571*x + 183756699868981873194,
        9*x**8 + 36*x**7 - 32*x**6 - 252*x**5 - 78*x**4 + 468*x**3 + 288*x**2 - 108*x + 9,
    ]

    # 断言使用 lex 排序计算的 Groebner 基等于 B
    assert groebner(F, t, x, order=lex) == B
    # 断言 G 对象的 fglm 方法使用 lex 排序得到的结果等于 B
    assert G.fglm(lex) == B

    # 定义新的多项式 F
    F = [x**2 - x - 3*y + 1, -2*x + y**2 + y - 1]
    # 计算 F 的 Groebner 基，使用 lex 排序
    G = groebner(F, x, y, order=lex)

    # 定义多项式 B
    B = [
        x**2 - x - 3*y + 1,
        y**2 - 2*x + y - 1,
    ]

    # 断言使用 grlex 排序计算的 Groebner 基等于 B
    assert groebner(F, x, y, order=grlex) == B
    # 断言 G 对象的 fglm 方法使用 grlex 排序得到的结果等于 B
    assert G.fglm(grlex) == B


def test_is_zero_dimensional():
    # 断言 [x, y] 是零维的
    assert is_zero_dimensional([x, y], x, y) is True
    # 断言 [x**3 + y**2] 不是零维的
    assert is_zero_dimensional([x**3 + y**2], x, y) is False

    # 断言 [x, y, z] 是零维的
    assert is_zero_dimensional([x, y, z], x, y, z) is True
    # 断言 [x, y, z] 在加入 t 后不是零维的
    assert is_zero_dimensional([x, y, z], x, y, z, t) is False

    # 定义多项式 F
    F = [x*y - z, y*z - x, x*y - y]
    # 断言 F 是零维的
    assert is_zero_dimensional(F, x, y, z) is True

    # 定义多项式 F
    F = [x**2 - 2*x*z + 5, x*y**2 + y*z**3, 3*y**2 - 8*z**2]
    # 断言 F 是零维的
    assert is_zero_dimensional(F, x, y, z) is True


def test_GroebnerBasis():
    # 定义多项式 F
    F = [x*y - 2*y, 2*y**2 - x**2]

    # 计算 F 的 Groebner 基，使用 grevlex 排序
    G = groebner(F, x, y, order='grevlex')
    # 定义多项式 H
    H = [y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y]
    # 将 H 转换为 Poly 对象的列表 P
    P = [ Poly(h, x, y) for h in H ]

    # 断言在添加 0 后，使用 grevlex 排序计算的 Groebner 基等于 G
    assert groebner(F + [0], x, y, order='grevlex') == G
    # 断言 G 是 GroebnerBasis 类的实例
    assert isinstance(G, GroebnerBasis) is True

    # 断言 G 的长度为 3
    assert len(G) == 3

    # 断言 G 的第一个元素等于 H 的第一个元素且不是 Poly 对象
    assert G[0] == H[0] and not G[0].is_Poly
    # 断言 G 的第二个元素等于 H 的第二个元素且不是 Poly 对象
    assert G[1] == H[1] and not G[1].is_Poly
    # 断言 G 的第三个元素等于 H 的第三个元素且不是 Poly 对象
    assert G[2] == H[2] and not G[2].is_Poly

    # 断言 G 的后两个元素与 H 的后两个元素相等且不是任何 Poly 对象
    assert G[1:] == H[1:] and not any(g.is_Poly for g in G[1:])
    # 断言 G 的前两个元素与 H 的前两个元素相等且不是任何 Poly 对象
    assert G[:2] == H[:2] and not any(g.is_Poly for g in G[1:])

    # 断言 G 的表达式列表与 H 相等
    assert G.exprs == H
    # 断言 G 的多项式列表与 P 相等
    assert G.polys == P
    # 断言 G 的生成器为 (x, y)
    assert G.gens == (x, y)
    # 断言 G 的域为 ZZ (整数环)
    assert G.domain == ZZ
    # 断言 G 的排序方式为 grevlex
    assert G.order == grevlex

    # 断言 G 等于 H
    assert G == H
    # 断言 G 等于 tuple(H)
    assert G == tuple(H)
    # 断言 G 等于 P
    assert G == P
    # 断言 G 等于 tuple(P)
    assert G == tuple(P)

    # 断言 G 不等于空列表
    assert G != []

    # 重新计算 F 的 Groebner 基，指定 polys=True
    G = groebner(F, x, y, order='
    # 断言条件：检查列表 G 和 P 的前两个元素是否相等，并且检查列表 G 的其余元素是否都为多项式对象。
    assert G[:2] == P[:2] and all(g.is_Poly for g in G[1:])
# 定义一个函数用于测试多项式操作
def test_poly():
    # 断言多项式poly(x)等于Poly(x, x)
    assert poly(x) == Poly(x, x)
    # 断言多项式poly(y)等于Poly(y, y)
    assert poly(y) == Poly(y, y)

    # 断言多项式poly(x + y)等于Poly(x + y, x, y)
    assert poly(x + y) == Poly(x + y, x, y)
    # 断言多项式poly(x + sin(x))等于Poly(x + sin(x), x, sin(x))
    assert poly(x + sin(x)) == Poly(x + sin(x), x, sin(x))

    # 断言多项式poly(x + y, wrt=y)等于Poly(x + y, y, x)
    assert poly(x + y, wrt=y) == Poly(x + y, y, x)
    # 断言多项式poly(x + sin(x), wrt=sin(x))等于Poly(x + sin(x), sin(x), x)
    assert poly(x + sin(x), wrt=sin(x)) == Poly(x + sin(x), sin(x), x)

    # 断言多项式poly(x*y + 2*x*z**2 + 17)等于Poly(x*y + 2*x*z**2 + 17, x, y, z)
    assert poly(x*y + 2*x*z**2 + 17) == Poly(x*y + 2*x*z**2 + 17, x, y, z)

    # 断言多项式poly(2*(y + z)**2 - 1)等于Poly(2*y**2 + 4*y*z + 2*z**2 - 1, y, z)
    assert poly(2*(y + z)**2 - 1) == Poly(2*y**2 + 4*y*z + 2*z**2 - 1, y, z)
    # 断言多项式poly(x*(y + z)**2 - 1)等于Poly(x*y**2 + 2*x*y*z + x*z**2 - 1, x, y, z)
    assert poly(x*(y + z)**2 - 1) == Poly(x*y**2 + 2*x*y*z + x*z**2 - 1, x, y, z)
    # 断言多项式poly(2*x*(y + z)**2 - 1)等于Poly(2*x*y**2 + 4*x*y*z + 2*x*z**2 - 1, x, y, z)
    assert poly(2*x*(y + z)**2 - 1) == Poly(2*x*y**2 + 4*x*y*z + 2*x*z**2 - 1, x, y, z)

    # 断言多项式poly(2*(y + z)**2 - x - 1)等于Poly(2*y**2 + 4*y*z + 2*z**2 - x - 1, x, y, z)
    assert poly(2*(y + z)**2 - x - 1) == Poly(2*y**2 + 4*y*z + 2*z**2 - x - 1, x, y, z)
    # 断言多项式poly(x*(y + z)**2 - x - 1)等于Poly(x*y**2 + 2*x*y*z + x*z**2 - x - 1, x, y, z)
    assert poly(x*(y + z)**2 - x - 1) == Poly(x*y**2 + 2*x*y*z + x*z**2 - x - 1, x, y, z)
    # 断言多项式poly(2*x*(y + z)**2 - x - 1)等于Poly(2*x*y**2 + 4*x*y*z + 2*x*z**2 - x - 1, x, y, z)
    assert poly(2*x*(y + z)**2 - x - 1) == Poly(2*x*y**2 + 4*x*y*z + 2*x*z**2 - x - 1, x, y, z)

    # 断言多项式poly(x*y + (x + y)**2 + (x + z)**2)等于Poly(2*x*z + 3*x*y + y**2 + z**2 + 2*x**2, x, y, z)
    assert poly(x*y + (x + y)**2 + (x + z)**2) == Poly(2*x*z + 3*x*y + y**2 + z**2 + 2*x**2, x, y, z)
    # 断言多项式poly(x*y*(x + y)*(x + z)**2)等于Poly(x**3*y**2 + x*y**2*z**2 + y*x**2*z**2 + 2*z*x**2*y**2 + 2*y*z*x**3 + y*x**4, x, y, z)
    assert poly(x*y*(x + y)*(x + z)**2) == Poly(x**3*y**2 + x*y**2*z**2 + y*x**2*z**2 + 2*z*x**2*y**2 + 2*y*z*x**3 + y*x**4, x, y, z)

    # 断言多项式poly(Poly(x + y + z, y, x, z))等于Poly(x + y + z, y, x, z)
    assert poly(Poly(x + y + z, y, x, z)) == Poly(x + y + z, y, x, z)

    # 断言多项式poly((x + y)**2, x)等于Poly(x**2 + 2*x*y + y**2, x, domain=ZZ[y])
    assert poly((x + y)**2, x) == Poly(x**2 + 2*x*y + y**2, x, domain=ZZ[y])
    # 断言多项式poly((x + y)**2, y)等于Poly(x**2 + 2*x*y + y**2, y, domain=ZZ[x])
    assert poly((x + y)**2, y) == Poly(x**2 + 2*x*y + y**2, y, domain=ZZ[x])

    # 断言多项式poly(1, x)等于Poly(1, x)
    assert poly(1, x) == Poly(1, x)
    # 使用lambda表达式断言生成器需求异常
    raises(GeneratorsNeeded, lambda: poly(1))

    # 断言多项式poly(x + y, x, y)等于Poly(x + y, x, y)
    assert poly(x + y, x, y) == Poly(x + y, x, y)
    # 断言多项式poly(x + y, y, x)等于Poly(x + y, y, x)
    assert poly(x + y, y, x) == Poly(x + y, y, x)

    # issue 6184
    # 断言表达式x + (2*x + 3)**2/5 + S(6)/5的多项式化简等于其展开形式
    expr1 = x + (2*x + 3)**2/5 + S(6)/5
    assert poly(expr1).as_expr() == expr1.expand()
    # 断言表达式y*(y+1) + S(1)/3的多项式化简等于其展开形式
    expr2 = y*(y+1) + S(1)/3
    assert poly(expr2).as_expr() == expr2.expand()


# 定义一个函数用于测试_coeff函数
def test_keep_coeff():
    # 使用Mul类构建u对象
    u = Mul(2, x + 1, evaluate=False)
    # 断言_keep_coeff(S.One, x)等于x
    assert _keep_coeff(S.One, x) == x
    # 断言_keep_coeff(S.NegativeOne, x)等于-x
    assert _keep_coeff(S.NegativeOne, x) == -x
    # 断言_keep_coeff(S(1.0), x)等于1.0*x
    assert _keep_coeff(S(1.0), x) == 1.0*x
    # 断言_keep_coeff(S(-1.0), x)等于-1.0*x
    assert _keep_coeff(S(-1.0), x) == -1.0*x
    # 断言_keep_coeff(S.One, 2*x)等于2*x
    assert _keep_coeff(S.One, 2*x) == 2*x
    # 断言_keep_coeff(S(2), x/2)等于x
    assert _keep_coeff(S(2), x/2) == x
    # 断言_keep_coeff(S(2), sin(x))等于2*sin(x)
    assert _keep_coeff(S(2), sin(x)) == 2*sin(x)
    # 断言_keep_coeff(S(2), x + 1)等
    # 断言：验证表达式的展开和因式分解
    assert expand(factor(expand(
        # 外部展开：展开 (x - I*y)*(z - I*t)
        (x - I*y)*(z - I*t)), extension=[I])) == -I*t*x - t*y + x*z - I*y*z
def test_noncommutative():
    class foo(Expr):
        is_commutative=False  # 定义一个非交换的表达式类
    e = x/(x + x*y)  # 创建一个表达式 e
    c = 1/( 1 + y)  # 创建一个表达式 c
    assert cancel(foo(e)) == foo(c)  # 使用 foo 类对 e 和 c 进行化简并比较结果
    assert cancel(e + foo(e)) == c + foo(c)  # 对 e 和 foo(e) 进行化简并比较结果
    assert cancel(e*foo(c)) == c*foo(c)  # 对 e 乘以 foo(c) 进行化简并比较结果


def test_to_rational_coeffs():
    assert to_rational_coeffs(
        Poly(x**3 + y*x**2 + sqrt(y), x, domain='EX')) is None  # 对多项式进行有理系数转换
    # issue 21268
    assert to_rational_coeffs(
        Poly(y**3 + sqrt(2)*y**2*sin(x) + 1, y)) is None  # 对多项式进行有理系数转换

    assert to_rational_coeffs(Poly(x, y)) is None  # 对多项式进行有理系数转换
    assert to_rational_coeffs(Poly(sqrt(2)*y)) is None  # 对多项式进行有理系数转换


def test_factor_terms():
    # issue 7067
    assert factor_list(x*(x + y)) == (1, [(x, 1), (x + y, 1)])  # 对表达式进行因式分解
    assert sqf_list(x*(x + y)) == (1, [(x**2 + x*y, 1)])  # 对表达式进行平方自由分解


def test_as_list():
    # issue 14496
    assert Poly(x**3 + 2, x, domain='ZZ').as_list() == [1, 0, 0, 2]  # 将多项式转换为列表形式
    assert Poly(x**2 + y + 1, x, y, domain='ZZ').as_list() == [[1], [], [1, 1]]  # 将多项式转换为列表形式
    assert Poly(x**2 + y + 1, x, y, z, domain='ZZ').as_list() == \
                                                    [[[1]], [[]], [[1], [1]]]  # 将多项式转换为列表形式


def test_issue_11198():
    assert factor_list(sqrt(2)*x) == (sqrt(2), [(x, 1)])  # 对表达式进行因式分解
    assert factor_list(sqrt(2)*sin(x), sin(x)) == (sqrt(2), [(sin(x), 1)])  # 对表达式进行因式分解


def test_Poly_precision():
    # Make sure Poly doesn't lose precision
    p = Poly(pi.evalf(100)*x)  # 创建一个高精度的多项式
    assert p.as_expr() == pi.evalf(100)*x  # 将多项式转换回表达式并进行比较


def test_issue_12400():
    # Correction of check for negative exponents
    assert poly(1/(1+sqrt(2)), x) == \
            Poly(1/(1+sqrt(2)), x, domain='EX')  # 对多项式进行创建和比较，处理负指数的修正


def test_issue_14364():
    assert gcd(S(6)*(1 + sqrt(3))/5, S(3)*(1 + sqrt(3))/10) == Rational(3, 10) * (1 + sqrt(3))  # 求两个数的最大公约数
    assert gcd(sqrt(5)*Rational(4, 7), sqrt(5)*Rational(2, 3)) == sqrt(5)*Rational(2, 21)  # 求两个数的最大公约数

    assert lcm(Rational(2, 3)*sqrt(3), Rational(5, 6)*sqrt(3)) == S(10)*sqrt(3)/3  # 求两个数的最小公倍数
    assert lcm(3*sqrt(3), 4/sqrt(3)) == 12*sqrt(3)  # 求两个数的最小公倍数
    assert lcm(S(5)*(1 + 2**Rational(1, 3))/6, S(3)*(1 + 2**Rational(1, 3))/8) == Rational(15, 2) * (1 + 2**Rational(1, 3))  # 求两个数的最小公倍数

    assert gcd(Rational(2, 3)*sqrt(3), Rational(5, 6)/sqrt(3)) == sqrt(3)/18  # 求两个数的最大公约数
    assert gcd(S(4)*sqrt(13)/7, S(3)*sqrt(13)/14) == sqrt(13)/14  # 求两个数的最大公约数

    # gcd_list and lcm_list
    assert gcd([S(2)*sqrt(47)/7, S(6)*sqrt(47)/5, S(8)*sqrt(47)/5]) == sqrt(47)*Rational(2, 35)  # 求列表中多个数的最大公约数
    assert gcd([S(6)*(1 + sqrt(7))/5, S(2)*(1 + sqrt(7))/7, S(4)*(1 + sqrt(7))/13]) ==  (1 + sqrt(7))*Rational(2, 455)  # 求列表中多个数的最大公约数
    assert lcm((Rational(7, 2)/sqrt(15), Rational(5, 6)/sqrt(15), Rational(5, 8)/sqrt(15))) == Rational(35, 2)/sqrt(15)  # 求列表中多个数的最小公倍数
    assert lcm([S(5)*(2 + 2**Rational(5, 7))/6, S(7)*(2 + 2**Rational(5, 7))/2, S(13)*(2 + 2**Rational(5, 7))/4]) == Rational(455, 2) * (2 + 2**Rational(5, 7))  # 求列表中多个数的最小公倍数


def test_issue_15669():
    x = Symbol("x", positive=True)
    expr = (16*x**3/(-x**2 + sqrt(8*x**2 + (x**2 - 2)**2) + 2)**2 -
        2*2**Rational(4, 5)*x*(-x**2 + sqrt(8*x**2 + (x**2 - 2)**2) + 2)**Rational(3, 5) + 10*x)
    assert factor(expr, deep=True) == x*(x**2 + 2)  # 对表达式进行因式分解


def test_issue_17988():
    x = Symbol('x')
    # 使用 poly 函数创建一个多项式 p，其参数为 x - 1
    p = poly(x - 1)
    
    # 使用 warns_deprecated_sympy 上下文管理器，捕获 SymPy 的过时警告
    with warns_deprecated_sympy():
        # 创建一个 1x2 的矩阵 M，其元素为两个相同的多项式 poly(x + 1)
        M = Matrix([[poly(x + 1), poly(x + 1)]])
    
    # 使用 warns 上下文管理器，捕获 SymPyDeprecationWarning 警告，不进行测试堆栈层级的检查
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        # 断言 p * M 等于 M * p，结果为一个 1x2 的矩阵，其元素为两个多项式 poly(x**2 - 1)
        assert p * M == M * p == Matrix([[poly(x**2 - 1), poly(x**2 - 1)]])
# 测试问题 18205 的函数，验证取消操作的正确性
def test_issue_18205():
    # 验证取消操作对复数 (2 + I)*(3 - I) 的结果是否为 7 + I
    assert cancel((2 + I)*(3 - I)) == 7 + I
    # 验证取消操作对复数 (2 + I)*(2 - I) 的结果是否为 5
    assert cancel((2 + I)*(2 - I)) == 5


# 测试问题 8695 的函数，验证多项式的平方因式分解列表是否正确
def test_issue_8695():
    # 定义多项式 p
    p = (x**2 + 1) * (x - 1)**2 * (x - 2)**3 * (x - 3)**3
    # 预期的平方因式分解结果
    result = (1, [(x**2 + 1, 1), (x - 1, 2), (x**2 - 5*x + 6, 3)])
    # 验证 sqf_list 函数对 p 的输出是否等于预期结果 result
    assert sqf_list(p) == result


# 测试问题 19113 的函数，验证多项式根的细化和计数操作是否引发多项式错误异常
def test_issue_19113():
    # 定义多项式 eq
    eq = sin(x)**3 - sin(x) + 1
    # 预期在指定区间上对根的细化操作应引发 PolynomialError 异常
    raises(PolynomialError, lambda: refine_root(eq, 1, 2, 1e-2))
    # 预期在指定区间上计算根数操作应引发 PolynomialError 异常
    raises(PolynomialError, lambda: count_roots(eq, -1, 1))
    # 预期实数根计算操作应引发 PolynomialError 异常
    raises(PolynomialError, lambda: real_roots(eq))
    # 预期数值根计算操作应引发 PolynomialError 异常
    raises(PolynomialError, lambda: nroots(eq))
    # 预期地面根计算操作应引发 PolynomialError 异常
    raises(PolynomialError, lambda: ground_roots(eq))
    # 预期二次方根多项式计算操作应引发 PolynomialError 异常
    raises(PolynomialError, lambda: nth_power_roots_poly(eq, 2))


# 测试问题 19360 的函数，验证多项式因式分解是否正确
def test_issue_19360():
    # 定义多项式 f
    f = 2*x**2 - 2*sqrt(2)*x*y + y**2
    # 验证 factor 函数对 f 进行因式分解时的输出是否符合预期
    assert factor(f, extension=sqrt(2)) == 2*(x - (sqrt(2)*y/2))**2

    # 重新定义多项式 f
    f = -I*t*x - t*y + x*z - I*y*z
    # 验证 factor 函数对 f 进行因式分解时的输出是否符合预期
    assert factor(f, extension=I) == (x - I*y)*(-I*t + z)


# 测试多项式复制是否与原始多项式相等
def test_poly_copy_equals_original():
    # 创建多项式 poly
    poly = Poly(x + y, x, y, z)
    # 复制多项式 poly
    copy = poly.copy()
    # 验证复制的多项式与原始多项式是否相等
    assert poly == copy, (
        "Copied polynomial not equal to original.")
    # 验证复制的多项式是否使用了相同的生成器
    assert poly.gens == copy.gens, (
        "Copied polynomial has different generators than original.")


# 测试反序列化后的多项式是否与原始多项式相等
def test_deserialized_poly_equals_original():
    # 创建多项式 poly
    poly = Poly(x + y, x, y, z)
    # 反序列化多项式 poly
    deserialized = pickle.loads(pickle.dumps(poly))
    # 验证反序列化后的多项式是否与原始多项式相等
    assert poly == deserialized, (
        "Deserialized polynomial not equal to original.")
    # 验证反序列化后的多项式是否使用了相同的生成器
    assert poly.gens == deserialized.gens, (
        "Deserialized polynomial has different generators than original.")


# 测试问题 20389 的函数，验证多项式的阶数计算是否正确
def test_issue_20389():
    # 计算多项式 x * (x + 1) - x ** 2 - x 的阶数
    result = degree(x * (x + 1) - x ** 2 - x, x)
    # 验证计算结果是否为负无穷
    assert result == -oo


# 测试问题 20985 的函数，验证多项式的生成和阶数计算是否正确
def test_issue_20985():
    # 导入符号变量模块
    from sympy.core.symbol import symbols
    # 定义符号变量 w 和 R
    w, R = symbols('w R')
    # 创建多项式 poly
    poly = Poly(1.0 + I*w/R, w, 1/R)
    # 验证多项式 poly 的阶数是否等于 1
    assert poly.degree() == S(1)
```