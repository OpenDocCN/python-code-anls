# `D:\src\scipysrc\sympy\sympy\sets\tests\test_setexpr.py`

```
# 导入需要的类和函数
from sympy.sets.setexpr import SetExpr
from sympy.sets import Interval, FiniteSet, Intersection, ImageSet, Union

from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.sets.sets import Set

# 定义符号变量和虚拟变量
a, x = symbols("a, x")
_d = Dummy("d")

# 测试函数：测试 SetExpr 类的基本功能
def test_setexpr():
    se = SetExpr(Interval(0, 1))
    assert isinstance(se.set, Set)  # 断言 se.set 是 Set 类的实例
    assert isinstance(se, Expr)  # 断言 se 是 Expr 类的实例

# 测试函数：测试指数和对数函数在 SetExpr 上的应用
def test_scalar_funcs():
    assert SetExpr(Interval(0, 1)).set == Interval(0, 1)
    a, b = Symbol('a', real=True), Symbol('b', real=True)
    a, b = 1, 2
    # TODO: add support for more functions in the future:
    for f in [exp, log]:
        input_se = f(SetExpr(Interval(a, b)))
        output = input_se.set
        expected = Interval(Min(f(a), f(b)), Max(f(a), f(b)))
        assert output == expected

# 测试函数：测试加法和乘法运算在 SetExpr 上的应用
def test_Add_Mul():
    assert (SetExpr(Interval(0, 1)) + 1).set == Interval(1, 2)
    assert (SetExpr(Interval(0, 1))*2).set == Interval(0, 2)

# 测试函数：测试幂运算在 SetExpr 上的应用
def test_Pow():
    assert (SetExpr(Interval(0, 2))**2).set == Interval(0, 4)

# 测试函数：测试复合函数在 SetExpr 上的应用
def test_compound():
    assert (exp(SetExpr(Interval(0, 1))*2 + 1)).set == \
           Interval(exp(1), exp(3))

# 测试函数：测试两个 Interval 类型 SetExpr 的加法和乘法运算
def test_Interval_Interval():
    assert (SetExpr(Interval(1, 2)) + SetExpr(Interval(10, 20))).set == \
           Interval(11, 22)
    assert (SetExpr(Interval(1, 2))*SetExpr(Interval(10, 20))).set == \
           Interval(10, 40)

# 测试函数：测试两个 FiniteSet 类型 SetExpr 的加法和乘法运算
def test_FiniteSet_FiniteSet():
    assert (SetExpr(FiniteSet(1, 2, 3)) + SetExpr(FiniteSet(1, 2))).set == \
           FiniteSet(2, 3, 4, 5)
    assert (SetExpr(FiniteSet(1, 2, 3))*SetExpr(FiniteSet(1, 2))).set == \
           FiniteSet(1, 2, 3, 4, 6)

# 测试函数：测试 Interval 类型和 FiniteSet 类型 SetExpr 的加法运算
def test_Interval_FiniteSet():
    assert (SetExpr(FiniteSet(1, 2)) + SetExpr(Interval(0, 10))).set == \
           Interval(1, 12)

# 测试函数：测试多个 SetExpr 的加法运算
def test_Many_Sets():
    assert (SetExpr(Interval(0, 1)) +
            SetExpr(Interval(2, 3)) +
            SetExpr(FiniteSet(10, 11, 12))).set == Interval(12, 16)

# 测试函数：测试两个相同 FiniteSet 类型 SetExpr 的加法运算
def test_same_setexprs_are_not_identical():
    a = SetExpr(FiniteSet(0, 1))
    b = SetExpr(FiniteSet(0, 1))
    assert (a + b).set == FiniteSet(0, 1, 2)

    # 无法检测到两个 SetExpr 相同的情况：
    # assert (a + a).set == FiniteSet(0, 2)

# 测试函数：测试 Interval 类型 SetExpr 的各种子类的算术运算
def test_Interval_arithmetic():
    i12cc = SetExpr(Interval(1, 2))
    i12lo = SetExpr(Interval.Lopen(1, 2))
    i12ro = SetExpr(Interval.Ropen(1, 2))
    i12o = SetExpr(Interval.open(1, 2))

    n23cc = SetExpr(Interval(-2, 3))
    n23lo = SetExpr(Interval.Lopen(-2, 3))
    n23ro = SetExpr(Interval.Ropen(-2, 3))
    n23o = SetExpr(Interval.open(-2, 3))

    n3n2cc = SetExpr(Interval(-3, -2))

    assert i12cc + i12cc == SetExpr(Interval(2, 4))
    # 断言：两个 Interval 对象相减，结果应为包含 [-1, 1] 的集合表达式
    assert i12cc - i12cc == SetExpr(Interval(-1, 1))
    # 断言：两个 Interval 对象相乘，结果应为包含 [1, 4] 的集合表达式
    assert i12cc*i12cc == SetExpr(Interval(1, 4))
    # 断言：两个 Interval 对象相除，结果应为包含 [1/2, 2] 的集合表达式
    assert i12cc/i12cc == SetExpr(Interval(S.Half, 2))
    # 断言：Interval 对象的平方，结果应为包含 [1, 4] 的集合表达式
    assert i12cc**2 == SetExpr(Interval(1, 4))
    # 断言：Interval 对象的立方，结果应为包含 [1, 8] 的集合表达式
    assert i12cc**3 == SetExpr(Interval(1, 8))

    # 断言：两个 Interval 对象相加，结果应为开区间 (2, 4) 的集合表达式
    assert i12lo + i12ro == SetExpr(Interval.open(2, 4))
    # 断言：两个 Interval 对象相减，结果应为左开右闭区间 (-1, 1] 的集合表达式
    assert i12lo - i12ro == SetExpr(Interval.Lopen(-1, 1))
    # 断言：两个 Interval 对象相乘，结果应为开区间 (1, 4) 的集合表达式
    assert i12lo*i12ro == SetExpr(Interval.open(1, 4))
    # 断言：两个 Interval 对象相除，结果应为左开右闭区间 (1/2, 2] 的集合表达式
    assert i12lo/i12ro == SetExpr(Interval.Lopen(S.Half, 2))
    # 断言：两个 Interval 对象相加，结果应为左开右闭区间 (2, 4] 的集合表达式
    assert i12lo + i12lo == SetExpr(Interval.Lopen(2, 4))
    # 断言：两个 Interval 对象相减，结果应为开区间 (-1, 1) 的集合表达式
    assert i12lo - i12lo == SetExpr(Interval.open(-1, 1))
    # 断言：两个 Interval 对象相乘，结果应为左开右闭区间 (1, 4) 的集合表达式
    assert i12lo*i12lo == SetExpr(Interval.Lopen(1, 4))
    # 断言：两个 Interval 对象相除，结果应为开区间 (1/2, 2) 的集合表达式
    assert i12lo/i12lo == SetExpr(Interval.open(S.Half, 2))
    # 断言：一个 Interval 对象与一个数值对象相加，结果应为左开右闭区间 (2, 4] 的集合表达式
    assert i12lo + i12cc == SetExpr(Interval.Lopen(2, 4))
    # 断言：一个 Interval 对象与一个数值对象相减，结果应为左开右闭区间 (-1, 1] 的集合表达式
    assert i12lo - i12cc == SetExpr(Interval.Lopen(-1, 1))
    # 断言：一个 Interval 对象与一个数值对象相乘，结果应为左开右闭区间 (1, 4) 的集合表达式
    assert i12lo*i12cc == SetExpr(Interval.Lopen(1, 4))
    # 断言：一个 Interval 对象与一个数值对象相除，结果应为左开右闭区间 (1/2, 2] 的集合表达式
    assert i12lo/i12cc == SetExpr(Interval.Lopen(S.Half, 2))
    # 断言：一个 Interval 对象的平方，结果应为左开右闭区间 (1, 4) 的集合表达式
    assert i12lo**2 == SetExpr(Interval.Lopen(1, 4))
    # 断言：一个 Interval 对象的立方，结果应为左开右闭区间 (1, 8) 的集合表达式
    assert i12lo**3 == SetExpr(Interval.Lopen(1, 8))

    # 断言：两个 Interval 对象相加，结果应为右开区间 (2, 4) 的集合表达式
    assert i12ro + i12ro == SetExpr(Interval.Ropen(2, 4))
    # 断言：两个 Interval 对象相减，结果应为开区间 (-1, 1) 的集合表达式
    assert i12ro - i12ro == SetExpr(Interval.open(-1, 1))
    # 断言：两个 Interval 对象相乘，结果应为右开区间 (1, 4) 的集合表达式
    assert i12ro*i12ro == SetExpr(Interval.Ropen(1, 4))
    # 断言：两个 Interval 对象相除，结果应为开区间 (1/2, 2) 的集合表达式
    assert i12ro/i12ro == SetExpr(Interval.open(S.Half, 2))
    # 断言：一个 Interval 对象与一个数值对象相加，结果应为右开区间 (2, 4) 的集合表达式
    assert i12ro + i12cc == SetExpr(Interval.Ropen(2, 4))
    # 断言：一个 Interval 对象与一个数值对象相减，结果应为右开区间 (-1, 1) 的集合表达式
    assert i12ro - i12cc == SetExpr(Interval.Ropen(-1, 1))
    # 断言：一个 Interval 对象与一个数值对象相乘，结果应为右开区间 (1, 4) 的集合表达式
    assert i12ro*i12cc == SetExpr(Interval.Ropen(1, 4))
    # 断言：一个 Interval 对象与一个数值对象相除，结果应为右开区间 (1/2, 2) 的集合表达式
    assert i12ro/i12cc == SetExpr(Interval.Ropen(S.Half, 2))
    # 断言：一个 Interval 对象与一个数值对象相加，结果应为开区间 (2, 4) 的集合表达式
    assert i12ro + i12o == SetExpr(Interval.open(2, 4))
    # 断言：一个 Interval 对象与一个数值对象相减，结果应为开区间 (-1, 1) 的集合表达式
    assert i12ro - i12o == SetExpr(Interval.open(-1, 1))
    # 断言：一个 Interval 对象与一个数值对象相乘，结果应为开区间 (1, 4) 的集合表达式
    assert i12ro*i12o == SetExpr(Interval.open(1, 4))
    # 断言：一个 Interval 对象与一个数值对象相除，结果应为开区间 (1/2, 2) 的集合表达式
    assert i12ro/i12o == SetExpr(Interval.open(S.Half, 2))
    # 断言：一个 Interval 对象的平方，结果应为右开区间 (1, 4) 的集合表达式
    assert i12ro**2 == SetExpr(Interval.Ropen(1, 4))
    # 断言：一个 Interval 对象的立方，结果应为右开区间 (1, 8) 的集合表达式
    assert i12ro**3 == SetExpr(Interval.Ropen(1, 8))

    # 断言：一个 Interval 对象与一个数值对象相加，结果应为开区间 (2, 4) 的集合表达式
    assert i12o + i12lo == SetExpr(Interval.open(2, 4))
    # 断言：一个 Interval 对象与一个数值对象相减，结果应为开区间 (-1, 1) 的集合表达式
    assert i12o - i12lo == SetExpr(Interval.open(-1, 1))
    # 断言：一个 Interval 对象与一个数值对象相乘，结果应为开区间 (1, 4) 的集合表达式
    assert i12o*i
    # 断言：n23cc 乘以自身等于闭区间 [-6, 9]
    assert n23cc*n23cc == SetExpr(Interval(-6, 9))
    
    # 断言：n23cc 除以自身等于开区间 (-∞, ∞)
    assert n23cc/n23cc == SetExpr(Interval.open(-oo, oo))
    
    # 断言：n23cc 加上 n23ro 等于右开区间 (-4, 6)
    assert n23cc + n23ro == SetExpr(Interval.Ropen(-4, 6))
    
    # 断言：n23cc 减去 n23ro 等于左开区间 (-5, 5)
    assert n23cc - n23ro == SetExpr(Interval.Lopen(-5, 5))
    
    # 断言：n23cc 乘以 n23ro 等于右开区间 (-6, 9)
    assert n23cc*n23ro == SetExpr(Interval.Ropen(-6, 9))
    
    # 断言：n23cc 除以 n23ro 等于左开区间 (-∞, ∞)
    assert n23cc/n23ro == SetExpr(Interval.Lopen(-oo, oo))
    
    # 断言：n23cc 加上 n23lo 等于左开区间 (-4, 6)
    assert n23cc + n23lo == SetExpr(Interval.Lopen(-4, 6))
    
    # 断言：n23cc 减去 n23lo 等于右开区间 (-5, 5)
    assert n23cc - n23lo == SetExpr(Interval.Ropen(-5, 5))
    
    # 断言：n23cc 乘以 n23lo 等于闭区间 [-6, 9)
    assert n23cc*n23lo == SetExpr(Interval(-6, 9))
    
    # 断言：n23cc 除以 n23lo 等于开区间 (-∞, ∞)
    assert n23cc/n23lo == SetExpr(Interval.open(-oo, oo))
    
    # 断言：n23cc 加上 n23o 等于开区间 (-4, 6)
    assert n23cc + n23o == SetExpr(Interval.open(-4, 6))
    
    # 断言：n23cc 减去 n23o 等于开区间 (-5, 5)
    assert n23cc - n23o == SetExpr(Interval.open(-5, 5))
    
    # 断言：n23cc 乘以 n23o 等于开区间 (-6, 9)
    assert n23cc*n23o == SetExpr(Interval.open(-6, 9))
    
    # 断言：n23cc 的平方等于闭区间 [0, 9]
    assert n23cc**2 == SetExpr(Interval(0, 9))
    
    # 断言：n23cc 的立方等于闭区间 [-8, 27]
    assert n23cc**3 == SetExpr(Interval(-8, 27))

    # 定义新的变量 n32cc, n32lo, n32ro 分别表示区间 [-3, 2], (-3, 2], [-3, 2)
    n32cc = SetExpr(Interval(-3, 2))
    n32lo = SetExpr(Interval.Lopen(-3, 2))
    n32ro = SetExpr(Interval.Ropen(-3, 2))
    
    # 断言：n32cc 乘以 n32lo 等于右开区间 (-6, 9)
    assert n32cc*n32lo == SetExpr(Interval.Ropen(-6, 9))
    
    # 断言：n32cc 乘以 n32cc 等于闭区间 [-6, 9]
    assert n32cc*n32cc == SetExpr(Interval(-6, 9))
    
    # 断言：n32lo 乘以 n32cc 等于右开区间 (-6, 9)
    assert n32lo*n32cc == SetExpr(Interval.Ropen(-6, 9))
    
    # 断言：n32cc 乘以 n32ro 等于闭区间 [-6, 9]
    assert n32cc*n32ro == SetExpr(Interval(-6, 9))
    
    # 断言：n32lo 乘以 n32ro 等于右开区间 (-6, 9)
    assert n32lo*n32ro == SetExpr(Interval.Ropen(-6, 9))
    
    # 断言：n32cc 除以 n32lo 等于右开区间 (-∞, ∞)
    assert n32cc/n32lo == SetExpr(Interval.Ropen(-oo, oo))
    
    # 断言：i12cc 除以 n32lo 等于右开区间 (-∞, ∞)，注意这里可能有变量定义缺失

    # 断言：n3n2cc 的平方等于闭区间 [4, 9]
    assert n3n2cc**2 == SetExpr(Interval(4, 9))
    
    # 断言：n3n2cc 的立方等于闭区间 [-27, -8]
    assert n3n2cc**3 == SetExpr(Interval(-27, -8))
    
    # 断言：n23cc 加上 i12cc 等于闭区间 [-1, 5]
    assert n23cc + i12cc == SetExpr(Interval(-1, 5))
    
    # 断言：n23cc 减去 i12cc 等于闭区间 [-4, 2]
    assert n23cc - i12cc == SetExpr(Interval(-4, 2))
    
    # 断言：n23cc 乘以 i12cc 等于闭区间 [-4, 6]
    assert n23cc*i12cc == SetExpr(Interval(-4, 6))
    
    # 断言：n23cc 除以 i12cc 等于闭区间 [-2, 3]
    assert n23cc/i12cc == SetExpr(Interval(-2, 3))
def test_SetExpr_Intersection():
    # 定义符号变量 x, y, z, w
    x, y, z, w = symbols("x y z w")
    # 创建区间 set1 和 set2
    set1 = Interval(x, y)
    set2 = Interval(w, z)
    # 计算 set1 和 set2 的交集
    inter = Intersection(set1, set2)
    # 将交集 inter 封装为 SetExpr 对象 se
    se = SetExpr(inter)
    # 断言表达式 exp(se).set 等于两个 ImageSet 的交集
    assert exp(se).set == Intersection(
        ImageSet(Lambda(x, exp(x)), set1),
        ImageSet(Lambda(x, exp(x)), set2))
    # 断言表达式 cos(se).set 等于 ImageSet(Lambda(x, cos(x)), inter)
    assert cos(se).set == ImageSet(Lambda(x, cos(x)), inter)


def test_SetExpr_Interval_div():
    # TODO: 由于存在 bug，某些表达式当前被注释掉，无法计算:
    # 断言 SetExpr(Interval(-3, -2)) / SetExpr(Interval(-2, 1)) 等于 SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-3, -2))/SetExpr(Interval(-2, 1)) == SetExpr(Interval(-oo, oo))
    # 断言 SetExpr(Interval(2, 3)) / SetExpr(Interval(-2, 2)) 等于 SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(2, 3))/SetExpr(Interval(-2, 2)) == SetExpr(Interval(-oo, oo))

    # 断言 SetExpr(Interval(-3, -2)) / SetExpr(Interval(0, 4)) 等于 SetExpr(Interval(-oo, -1/2))
    assert SetExpr(Interval(-3, -2))/SetExpr(Interval(0, 4)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    # 断言 SetExpr(Interval(2, 4)) / SetExpr(Interval(-3, 0)) 等于 SetExpr(Interval(-oo, -2/3))
    assert SetExpr(Interval(2, 4))/SetExpr(Interval(-3, 0)) == SetExpr(Interval(-oo, Rational(-2, 3)))
    # 断言 SetExpr(Interval(2, 4)) / SetExpr(Interval(0, 3)) 等于 SetExpr(Interval(2/3, oo))
    assert SetExpr(Interval(2, 4))/SetExpr(Interval(0, 3)) == SetExpr(Interval(Rational(2, 3), oo))

    # 断言 1 / SetExpr(Interval(-1, 2)) 等于 Union(Interval(-oo, -1), Interval(1/2, oo))
    assert 1/SetExpr(Interval(-1, 2)) == SetExpr(Union(Interval(-oo, -1), Interval(S.Half, oo)))

    # 断言 1 / SetExpr(Interval(0, 2)) 等于 SetExpr(Interval(1/2, oo))
    assert 1/SetExpr(Interval(0, 2)) == SetExpr(Interval(S.Half, oo))
    # 断言 (-1) / SetExpr(Interval(0, 2)) 等于 SetExpr(Interval(-oo, -1/2))
    assert (-1)/SetExpr(Interval(0, 2)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    # 断言 1 / SetExpr(Interval(-oo, 0)) 等于 SetExpr(Interval.open(-oo, 0))
    assert 1/SetExpr(Interval(-oo, 0)) == SetExpr(Interval.open(-oo, 0))
    # 断言 1 / SetExpr(Interval(-1, 0)) 等于 SetExpr(Interval(-oo, -1))
    assert 1/SetExpr(Interval(-1, 0)) == SetExpr(Interval(-oo, -1))
    # 断言 SetExpr(Interval(-1, 2)) / a 等于 Mul(SetExpr(Interval(1, 2)), 1/a, evaluate=False)
    # assert SetExpr(Interval(1, 2))/a == Mul(SetExpr(Interval(1, 2)), 1/a, evaluate=False)

    # 断言 SetExpr(Interval(1, 2)) / 0 等于 SetExpr(Interval(1, 2)) * zoo
    # assert SetExpr(Interval(1, 2))/0 == SetExpr(Interval(1, 2))*zoo
    # 断言 SetExpr(Interval(1, oo)) / oo 等于 SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(1, oo))/oo == SetExpr(Interval(0, oo))
    # 断言 SetExpr(Interval(1, oo)) / (-oo) 等于 SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(1, oo))/(-oo) == SetExpr(Interval(-oo, 0))
    # 断言 SetExpr(Interval(-oo, -1)) / oo 等于 SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, -1))/oo == SetExpr(Interval(-oo, 0))
    # 断言 SetExpr(Interval(-oo, -1)) / (-oo) 等于 SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-oo, -1))/(-oo) == SetExpr(Interval(0, oo))
    # 断言 SetExpr(Interval(-oo, oo)) / oo 等于 SetExpr(Interval(-oo, oo))
    # assert SetExpr(Interval(-oo, oo))/oo == SetExpr(Interval(-oo, oo))
    # 断言 SetExpr(Interval(-oo, oo)) / (-oo) 等于 SetExpr(Interval(-oo, oo))
    # assert SetExpr(Interval(-oo, oo))/(-oo) == SetExpr(Interval(-oo, oo))
    # 断言 SetExpr(Interval(-1, oo)) / oo 等于 SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-1, oo))/oo == SetExpr(Interval(0, oo))
    # 断言 SetExpr(Interval(-1, oo)) / (-oo) 等于 SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-1, oo))/(-oo) == SetExpr(Interval(-oo, 0))
    # 断言 SetExpr(Interval(-oo, 1)) / oo 等于 SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, 1))/oo == SetExpr(Interval(-oo, 0))
    # 断言 SetExpr(Interval(-oo, 1)) / (-oo) 等于 SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-oo, 1))/(-oo) == SetExpr(Interval(0, oo))


def test_SetExpr_Interval_pow():
    # 断言 SetExpr(Interval(0, 2)) ** 2 等于 SetExpr(Interval(0, 4))
    assert SetExpr(Interval(0, 2))**2 == SetExpr(Interval(0, 4))
    # 断言 SetExpr(Interval(-1, 1)) ** 2 等于 SetExpr(Interval(0, 1))
    assert SetExpr(Interval(-1, 1))**2 == SetExpr(Interval(0, 1))
    # 断言 SetExpr(Interval(1, 2)) ** 2 等于 SetExpr(Interval(1, 4))
    assert SetExpr(Interval(1, 2))**2 == SetExpr(Interval(1, 4))
    # 断言 SetExpr(Interval(-1, 2)) ** 3 等于 SetExpr(Interval(-1, 8))
    assert SetExpr(Interval(-1, 2))**3 == SetExpr(Interval(-1, 8))
    # 断言：区间 [-1, 1] 的幂运算结果应为 {1} 的集合表达式
    assert SetExpr(Interval(-1, 1))**0 == SetExpr(FiniteSet(1))

    # 断言：区间 [1, 2] 的 5/2 次幂运算结果应为区间 [1, 4*sqrt(2)]
    assert SetExpr(Interval(1, 2))**Rational(5, 2) == SetExpr(Interval(1, 4*sqrt(2)))

    # 注释：以下断言被注释掉，因为它们涉及到复杂的幂运算，需要进一步验证其正确性
    # assert SetExpr(Interval(-1, 2))**Rational(1, 3) == SetExpr(Interval(-1, 2**Rational(1, 3)))
    # assert SetExpr(Interval(0, 2))**S.Half == SetExpr(Interval(0, sqrt(2)))

    # 注释：以下断言被注释掉，因为它们涉及到复杂的幂运算，需要进一步验证其正确性
    # assert SetExpr(Interval(-4, 2))**Rational(2, 3) == SetExpr(Interval(0, 2*2**Rational(1, 3)))

    # 注释：以下断言被注释掉，因为它们涉及到复杂的幂运算，需要进一步验证其正确性
    # assert SetExpr(Interval(-1, 5))**S.Half == SetExpr(Interval(0, sqrt(5)))
    # assert SetExpr(Interval(-oo, 2))**S.Half == SetExpr(Interval(0, sqrt(2)))
    # assert SetExpr(Interval(-2, 3))**(Rational(-1, 4)) == SetExpr(Interval(0, oo))

    # 断言：区间 [1, 5] 的 -2 次幂运算结果应为区间 [1/25, 1]
    assert SetExpr(Interval(1, 5))**(-2) == SetExpr(Interval(Rational(1, 25), 1))

    # 断言：区间 [-1, 3] 的 -2 次幂运算结果应为区间 [0, oo]
    assert SetExpr(Interval(-1, 3))**(-2) == SetExpr(Interval(0, oo))

    # 断言：区间 [0, 2] 的 -2 次幂运算结果应为区间 [1/4, oo]
    assert SetExpr(Interval(0, 2))**(-2) == SetExpr(Interval(Rational(1, 4), oo))

    # 断言：区间 [-1, 2] 的 -3 次幂运算结果应为区间 (-oo, -1) ∪ [1/8, oo]
    assert SetExpr(Interval(-1, 2))**(-3) == SetExpr(Union(Interval(-oo, -1), Interval(Rational(1, 8), oo)))

    # 断言：区间 [-3, -2] 的 -3 次幂运算结果应为区间 [-1/8, -1/27]
    assert SetExpr(Interval(-3, -2))**(-3) == SetExpr(Interval(Rational(-1, 8), Rational(-1, 27)))

    # 断言：区间 [-3, -2] 的 -2 次幂运算结果应为区间 [1/9, 1/4]
    assert SetExpr(Interval(-3, -2))**(-2) == SetExpr(Interval(Rational(1, 9), Rational(1, 4)))

    # 注释：以下断言被注释掉，因为它们涉及到复杂的幂运算，需要进一步验证其正确性
    # assert SetExpr(Interval(0, oo))**S.Half == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-oo, -1))**Rational(1, 3) == SetExpr(Interval(-oo, -1))
    # assert SetExpr(Interval(-2, 3))**(Rational(-1, 3)) == SetExpr(Interval(-oo, oo))

    # 断言：区间 (-oo, 0) 的 -2 次幂运算结果应为开区间 (0, oo)
    assert SetExpr(Interval(-oo, 0))**(-2) == SetExpr(Interval.open(0, oo))

    # 断言：区间 [-2, 0) 的 -2 次幂运算结果应为区间 [1/4, oo)
    assert SetExpr(Interval(-2, 0))**(-2) == SetExpr(Interval(Rational(1, 4), oo))

    # 断言：区间 (1/3, 1/2] 的无穷大次幂运算结果应为有限集 {0}
    assert SetExpr(Interval(Rational(1, 3), S.Half))**oo == SetExpr(FiniteSet(0))

    # 断言：区间 [0, 1/2] 的无穷大次幂运算结果应为有限集 {0}
    assert SetExpr(Interval(0, S.Half))**oo == SetExpr(FiniteSet(0))

    # 断言：区间 (1/2, 1] 的无穷大次幂运算结果应为区间 [0, oo)
    assert SetExpr(Interval(S.Half, 1))**oo == SetExpr(Interval(0, oo))

    # 断言：区间 [0, 1] 的无穷大次幂运算结果应为区间 [0, oo)
    assert SetExpr(Interval(0, 1))**oo == SetExpr(Interval(0, oo))

    # 断言：区间 [2, 3] 的无穷大次幂运算结果应为有限集 {oo}
    assert SetExpr(Interval(2, 3))**oo == SetExpr(FiniteSet(oo))

    # 断言：区间 [1, 2] 的无穷大次幂运算结果应为区间 [0, oo)
    assert SetExpr(Interval(1, 2))**oo == SetExpr(Interval(0, oo))

    # 断言：区间 (1/2, 3] 的无穷大次幂运算结果应为区间 [0, oo)
    assert SetExpr(Interval(S.Half, 3))**oo == SetExpr(Interval(0, oo))

    # 断言：区间 (-1/3, -1/4] 的无穷大次幂运算结果应为有限集 {0}
    assert SetExpr(Interval(Rational(-1, 3), Rational(-1, 4)))**oo == SetExpr(FiniteSet(0))

    # 断言：区间 [-1, -1/2) 的无穷大次幂运算结果应为整个实数轴的区间 [-oo, oo)
    assert SetExpr(Interval(-1, Rational(-1, 2)))**oo == SetExpr(Interval(-oo, oo))

    # 断言：区间 [-3, -2] 的无穷大次幂运算结果应为有限集 {-oo, oo}
    assert SetExpr(Interval(-3, -2))**oo == SetExpr(FiniteSet(-oo, oo))

    # 断言：区间 [-2, -1] 的无穷大次幂运算结果应为整个实数轴的区间 [-oo, oo)
    assert SetExpr(Interval(-2, -1))**oo == SetExpr(Interval(-oo, oo))

    # 断言：区间 [-2, -1/2) 的无穷大次幂运算结果应为整个实数轴的区间 [-oo, oo)
    assert SetExpr(Interval(-2, Rational(-1, 2)))**oo == SetExpr(Interval(-oo, oo))

    # 断言：区间 (-1/2, 1] 的无穷大次幂运算结果应为有限集 {0}
    assert SetExpr(Interval(Rational(-1, 2), S.Half))**oo == SetExpr(FiniteSet(0))

    # 断言：区间 (-1/2, 1] 的无穷大次幂运算结果应为区间 [0, oo)
    assert SetExpr(Interval(Rational(-1, 2), 1))**oo == SetExpr(Interval(0, oo))

    # 断言：区
    # 断言：对表达式 SetExpr(Interval(-2, S.Half))**oo 进行检查，验证其等于 SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-2, S.Half))**oo == SetExpr(Interval(-oo, oo))
    
    # 断言：对表达式 (SetExpr(Interval(1, 2))**x).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**x), Interval(1, 2)))) 进行检查
    assert (SetExpr(Interval(1, 2))**x).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**x), Interval(1, 2))))
    
    # 断言：对表达式 SetExpr(Interval(2, 3))**(-oo) 进行检查，验证其等于 SetExpr(FiniteSet(0))
    assert SetExpr(Interval(2, 3))**(-oo) == SetExpr(FiniteSet(0))
    
    # 断言：对表达式 SetExpr(Interval(0, 2))**(-oo) 进行检查，验证其等于 SetExpr(Interval(0, oo))
    assert SetExpr(Interval(0, 2))**(-oo) == SetExpr(Interval(0, oo))
    
    # 断言：对表达式 (SetExpr(Interval(-1, 2))**(-oo)).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**(-oo)), Interval(-1, 2)))) 进行检查
    assert (SetExpr(Interval(-1, 2))**(-oo)).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**(-oo)), Interval(-1, 2))))
def test_SetExpr_Integers():
    # 断言：整数集合加1仍为整数集合
    assert SetExpr(S.Integers) + 1 == SetExpr(S.Integers)
    
    # 断言：整数集合加虚数单位I后，应与映射整数集合到 Lambda(_d, _d + I) 的结果相等
    assert (SetExpr(S.Integers) + I).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, _d + I), S.Integers)))
    
    # 断言：整数集合乘以-1仍为整数集合
    assert SetExpr(S.Integers)*(-1) == SetExpr(S.Integers)
    
    # 断言：整数集合乘以2后，应与映射整数集合到 Lambda(_d, 2*_d) 的结果相等
    assert (SetExpr(S.Integers)*2).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, 2*_d), S.Integers)))
    
    # 断言：整数集合乘以虚数单位I后，应与映射整数集合到 Lambda(_d, I*_d) 的结果相等
    assert (SetExpr(S.Integers)*I).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, I*_d), S.Integers)))
    
    # issue #18050:
    # 断言：对整数集合应用 Lambda(x, I*x + 1) 函数后，应与映射整数集合到 Lambda(_d, I*_d + 1) 的结果相等
    assert SetExpr(S.Integers)._eval_func(Lambda(x, I*x + 1)).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, I*_d + 1), S.Integers)))
    
    # needs improvement:
    # 断言：整数集合乘以虚数单位I后加1，应与映射整数集合到 Lambda(x, x + 1) 以及再映射到 Lambda(_d, _d*I) 的结果相等
    assert (SetExpr(S.Integers)*I + 1).dummy_eq(
        SetExpr(ImageSet(Lambda(x, x + 1),
                         ImageSet(Lambda(_d, _d*I), S.Integers))))
```