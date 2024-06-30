# `D:\src\scipysrc\sympy\sympy\sets\tests\test_conditionset.py`

```
from sympy.core.expr import unchanged  # 导入 unchanged 函数
from sympy.sets import (ConditionSet, Intersection, FiniteSet,  # 导入多个集合操作相关模块
    EmptySet, Union, Contains, ImageSet)
from sympy.sets.sets import SetKind  # 导入集合种类相关模块
from sympy.core.function import (Function, Lambda)  # 导入函数和Lambda表达式相关模块
from sympy.core.mod import Mod  # 导入取模运算相关模块
from sympy.core.kind import NumberKind  # 导入数值种类相关模块
from sympy.core.numbers import (oo, pi)  # 导入无穷大和π等常数
from sympy.core.relational import (Eq, Ne)  # 导入等式和不等式判断相关模块
from sympy.core.singleton import S  # 导入单例集合S
from sympy.core.symbol import (Symbol, symbols)  # 导入符号和符号生成函数
from sympy.functions.elementary.complexes import Abs  # 导入绝对值函数
from sympy.functions.elementary.trigonometric import (asin, sin)  # 导入反正弦和正弦函数
from sympy.logic.boolalg import And  # 导入逻辑与操作
from sympy.matrices.dense import Matrix  # 导入密集矩阵相关模块
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号相关模块
from sympy.sets.sets import Interval  # 导入区间操作相关模块
from sympy.testing.pytest import raises, warns_deprecated_sympy  # 导入测试相关函数

w = Symbol('w')  # 创建符号 w
x = Symbol('x')  # 创建符号 x
y = Symbol('y')  # 创建符号 y
z = Symbol('z')  # 创建符号 z
f = Function('f')  # 创建函数符号 f

# 定义测试函数 test_CondSet
def test_CondSet():
    # 创建一个条件集，满足 sin(x) = 0，x ∈ [0, 2π)
    sin_sols_principal = ConditionSet(x, Eq(sin(x), 0),
                                      Interval(0, 2*pi, False, True))
    assert pi in sin_sols_principal  # 断言 π 在 sin_sols_principal 中
    assert pi/2 not in sin_sols_principal  # 断言 π/2 不在 sin_sols_principal 中
    assert 3*pi not in sin_sols_principal  # 断言 3π 不在 sin_sols_principal 中
    assert oo not in sin_sols_principal  # 断言 无穷大不在 sin_sols_principal 中
    assert 5 in ConditionSet(x, x**2 > 4, S.Reals)  # 断言 5 在满足 x^2 > 4 的实数集合中
    assert 1 not in ConditionSet(x, x**2 > 4, S.Reals)  # 断言 1 不在满足 x^2 > 4 的实数集合中
    # 对于这种情况，0 不在基本集合中，因此它不可能在任何满足条件的子集中
    assert 0 not in ConditionSet(x, y > 5, Interval(1, 7))
    # 由于 'in' 需要一个真/假值，下面的断言会引发错误，因为给定的值无法提供条件评估所需的信息
    # （因为条件不依赖于虚拟符号）：结果是 `y > 5`。
    # 在这种情况下，ConditionSet 就像 Piecewise((Interval(1, 7), y > 5), (S.EmptySet, True)) 一样运行。
    raises(TypeError, lambda: 6 in ConditionSet(x, y > 5,
        Interval(1, 7)))

    X = MatrixSymbol('X', 2, 2)  # 创建一个2x2的矩阵符号 X
    matrix_set = ConditionSet(X, Eq(X*Matrix([[1, 1], [1, 1]]), X))  # 创建一个矩阵集合，满足 X * [[1, 1], [1, 1]] = X
    Y = Matrix([[0, 0], [0, 0]])  # 创建一个2x2的零矩阵 Y
    assert matrix_set.contains(Y).doit() is S.true  # 断言 matrix_set 包含 Y
    Z = Matrix([[1, 2], [3, 4]])  # 创建一个非条件满足矩阵 Z
    assert matrix_set.contains(Z).doit() is S.false  # 断言 matrix_set 不包含 Z

    assert isinstance(ConditionSet(x, x < 1, {x, y}).base_set,
        FiniteSet)  # 断言 ConditionSet(x, x < 1, {x, y}) 的基础集合是有限集
    raises(TypeError, lambda: ConditionSet(x, x + 1, {x, y}))  # 断言 ConditionSet(x, x + 1, {x, y}) 引发 TypeError
    raises(TypeError, lambda: ConditionSet(x, x, 1))  # 断言 ConditionSet(x, x, 1) 引发 TypeError

    I = S.Integers  # 设置整数集合 I
    U = S.UniversalSet  # 设置全集 U
    C = ConditionSet  # 别名 C 代表 ConditionSet
    assert C(x, False, I) is S.EmptySet  # 断言 ConditionSet(x, False, I) 等于空集
    assert C(x, True, I) is I  # 断言 ConditionSet(x, True, I) 等于整数集合 I
    assert C(x, x < 1, C(x, x < 2, I)
        ) == C(x, (x < 1) & (x < 2), I)  # 断言嵌套的 ConditionSet 与条件 (x < 1) & (x < 2) 等效
    assert C(y, y < 1, C(x, y < 2, I)
        ) == C(x, (x < 1) & (y < 2), I), C(y, y < 1, C(x, y < 2, I))  # 断言嵌套的 ConditionSet 与条件 (x < 1) & (y < 2) 等效
    assert C(y, y < 1, C(x, x < 2, I)
        ) == C(y, (y < 1) & (y < 2), I)  # 断言嵌套的 ConditionSet 与条件 (y < 1) & (y < 2) 等效
    assert C(y, y < 1, C(x, y < x, I)
        ) == C(x, (x < 1) & (y < x), I)  # 断言嵌套的 ConditionSet 与条件 (x < 1) & (y < x) 等效
    assert unchanged(C, y, x < 1, C(x, y < x, I))  # 断言未改变的 ConditionSet
    assert ConditionSet(x, x < 1).base_set is U  # 断言 ConditionSet(x, x < 1).base_set 是全集
    # 在实例化时未进行参数检查，但在测试包含性时会引发错误
    assert ConditionSet((x,), x < 1).base_set is U

    # 创建条件集合对象 c，其中包含条件 (x < y) 和结果 I**2
    c = ConditionSet((x, y), x < y, I**2)
    # 检查元组 (1, 2) 是否在条件集合 c 中
    assert (1, 2) in c
    # 检查元组 (1, pi) 是否不在条件集合 c 中
    assert (1, pi) not in c

    # 使用 lambda 表达式尝试创建带有超出预期参数数量的条件对象，应引发 TypeError
    raises(TypeError, lambda: C(x, x > 1, C((x, y), x > 1, I**2)))
    # 由于参数个数不匹配，应引发 TypeError 错误，因为只接受 3 个参数
    raises(TypeError, lambda: C((x, y), x + y < 2, U, U))
# 定义测试函数 `test_CondSet_intersect()`，用于测试 ConditionSet 类的交集操作
def test_CondSet_intersect():
    # 创建输入条件集 input_conditionset，包含条件 x**2 > 4 和区间 [1, 4)
    input_conditionset = ConditionSet(x, x**2 > 4, Interval(1, 4, False, False))
    # 创建其他区间对象 other_domain，区间为 (0, 3)
    other_domain = Interval(0, 3, False, False)
    # 创建期望输出的条件集 output_conditionset，包含条件 x**2 > 4 和区间 [1, 3)
    output_conditionset = ConditionSet(x, x**2 > 4, Interval(1, 3, False, False))
    # 断言 ConditionSet 的交集操作结果等于期望的 output_conditionset
    assert Intersection(input_conditionset, other_domain) == output_conditionset


# 定义测试函数 `test_issue_9849()`，用于测试 ConditionSet 类在特定条件下的行为
def test_issue_9849():
    # 断言满足 Eq(x, x) 条件的 ConditionSet 对象返回 S.Naturals 对象
    assert ConditionSet(x, Eq(x, x), S.Naturals) is S.Naturals
    # 断言满足 Eq(Abs(sin(x)), -1) 条件的 ConditionSet 对象返回 S.EmptySet 空集合对象
    assert ConditionSet(x, Eq(Abs(sin(x)), -1), S.Naturals) == S.EmptySet


# 定义测试函数 `test_simplified_FiniteSet_in_CondSet()`，测试 ConditionSet 类与 FiniteSet 的交互
def test_simplified_FiniteSet_in_CondSet():
    # 断言满足 And(x < 1, x > -3) 条件的 ConditionSet 返回包含 0 的 FiniteSet 对象
    assert ConditionSet(x, And(x < 1, x > -3), FiniteSet(0, 1, 2)) == FiniteSet(0)
    # 断言满足 x < 0 条件的 ConditionSet 返回 EmptySet 空集合对象
    assert ConditionSet(x, x < 0, FiniteSet(0, 1, 2)) == EmptySet
    # 断言满足 And(x < -3) 条件的 ConditionSet 也返回 EmptySet 空集合对象
    assert ConditionSet(x, And(x < -3), EmptySet) == EmptySet
    # 定义新的符号 y
    y = Symbol('y')
    # 断言满足 Eq(Mod(x, 3), 1) 条件的 ConditionSet 返回包含 1 和 4 的 FiniteSet 对象
    assert (ConditionSet(x, Eq(Mod(x, 3), 1), FiniteSet(1, 4, 2, y)) ==
            Union(FiniteSet(1, 4), ConditionSet(x, Eq(Mod(x, 3), 1), FiniteSet(y))))


# 定义测试函数 `test_free_symbols()`，测试 ConditionSet 对象的 free_symbols 属性
def test_free_symbols():
    # 断言 ConditionSet 对象以 Eq(y, 0) 为条件时，其 free_symbols 包含符号 y 和 z
    assert ConditionSet(x, Eq(y, 0), FiniteSet(z)).free_symbols == {y, z}
    # 断言 ConditionSet 对象以 Eq(x, 0) 为条件时，其 free_symbols 包含符号 z
    assert ConditionSet(x, Eq(x, 0), FiniteSet(z)).free_symbols == {z}
    # 断言 ConditionSet 对象以 Eq(x, 0) 和 FiniteSet(x, z) 为条件时，其 free_symbols 包含符号 x 和 z
    assert ConditionSet(x, Eq(x, 0), FiniteSet(x, z)).free_symbols == {x, z}
    # 断言 ConditionSet 对象以 Eq(x, 0) 和 ImageSet(Lambda(y, y**2), S.Integers) 为条件时，其 free_symbols 为空集合
    assert ConditionSet(x, Eq(x, 0), ImageSet(Lambda(y, y**2), S.Integers)).free_symbols == set()


# 定义测试函数 `test_bound_symbols()`，测试 ConditionSet 对象的 bound_symbols 属性
def test_bound_symbols():
    # 断言 ConditionSet 对象以 Eq(y, 0) 为条件时，其 bound_symbols 包含符号 x
    assert ConditionSet(x, Eq(y, 0), FiniteSet(z)).bound_symbols == [x]
    # 断言 ConditionSet 对象以 Eq(x, 0) 和 FiniteSet(x, y) 为条件时，其 bound_symbols 包含符号 x
    assert ConditionSet(x, Eq(x, 0), FiniteSet(x, y)).bound_symbols == [x]
    # 断言 ConditionSet 对象以 x < 10 和 ImageSet(Lambda(y, y**2), S.Integers) 为条件时，其 bound_symbols 包含符号 x
    assert ConditionSet(x, x < 10, ImageSet(Lambda(y, y**2), S.Integers)).bound_symbols == [x]
    # 断言 ConditionSet 对象以 x < 10 和 ConditionSet(y, y > 1, S.Integers) 为条件时，其 bound_symbols 包含符号 x
    assert ConditionSet(x, x < 10, ConditionSet(y, y > 1, S.Integers)).bound_symbols == [x]


# 定义测试函数 `test_as_dummy()`，测试 ConditionSet 对象的 as_dummy() 方法
def test_as_dummy():
    # 定义符号 _0 和 _1
    _0, _1 = symbols('_0 _1')
    # 断言 ConditionSet 对象以 x < 1 和 Interval(y, oo) 为条件时，其 as_dummy() 结果为 ConditionSet(_0, _0 < 1, Interval(y, oo))
    assert ConditionSet(x, x < 1, Interval(y, oo)).as_dummy() == ConditionSet(_0, _0 < 1, Interval(y, oo))
    # 断言 ConditionSet 对象以 x < 1 和 Interval(x, oo) 为条件时，其 as_dummy() 结果为 ConditionSet(_0, _0 < 1, Interval(x, oo))
    assert ConditionSet(x, x < 1, Interval(x, oo)).as_dummy() == ConditionSet(_0, _0 < 1, Interval(x, oo))
    # 断言 ConditionSet 对象以 x < 1 和 ImageSet(Lambda(y, y**2), S.Integers) 为条件时，其 as_dummy() 结果为 ConditionSet(_0, _0 < 1, ImageSet(Lambda(_0, _0**2), S.Integers))
    assert ConditionSet(x, x < 1, ImageSet(Lambda(y, y**2), S.Integers)).as_dummy() == ConditionSet(_0, _0 < 1, ImageSet(Lambda(_0, _0**2), S.Integers))
    # 创建 ConditionSet 对象 e，以 (x, y) <= S.Reals**2 为条件
    e = ConditionSet((x, y), x <= y, S.Reals**2)
    # 断言 e 的 bound_symbols 属性返回 [x, y]
    assert e.bound_symbols == [x, y]
    # 断言 e 的 as_dummy() 结果等于 ConditionSet((_0, _1), _0 <= _1, S.Reals**2)
    assert e.as_dummy() == ConditionSet((_0, _1), _0 <= _1, S.Reals**2)
    # 再次断言 e 的 as_dummy() 结果等于 ConditionSet((y, x), y <= x, S.Reals**2)，说明 as_dummy() 方法能够正确交换变量顺序
    assert e.as_dummy() == ConditionSet((y, x), y <= x, S.Reals**2).as_dummy()


# 定义测试函数 `test_subs_CondSet()`，测试 ConditionSet 对象的 subs 和 xreplace 方法
def test_subs_CondSet():
    # 创建包含符号 z 和 y 的 FiniteSet 对象 s
    s = FiniteSet(z, y)
    # 创建条件为 x < 2 和 s 的 ConditionSet 对象 c
    c = ConditionSet(x, x < 2, s)
    # 断言 c.subs(x, y) 返回原 ConditionSet 对象 c，因为变量 x 在条件中没有被替换
    assert c.subs(x, y) == c
    # 断言 c.subs(z, y) 返回 ConditionSet(x, x < 2, FiniteSet(y))，因为成功替换了符号 z
    assert c.subs(z, y) == ConditionSet(x, x < 2, FiniteSet(y))
    # 断言 c.xreplace({x: y
    # 定义一个负数符号对象 n
    n = Symbol('n', negative=True)
    # 断言条件集合，验证当 n 是负整数时，条件集合为空集
    assert ConditionSet(n, 0 < n, S.Integers) is S.EmptySet

    # 定义一个正数符号对象 p
    p = Symbol('p', positive=True)
    # 断言条件集合，验证在 n < y 的条件下，用 x 替换 n 后条件集合不变
    assert ConditionSet(n, n < y, S.Integers).subs(n, x) == ConditionSet(n, n < y, S.Integers)

    # 断言条件集合构造函数会引发 ValueError 异常，因为 x + 1 不满足 x < 1 的条件
    raises(ValueError, lambda: ConditionSet(x + 1, x < 1, S.Integers))

    # 断言条件集合，验证在 n < x 的条件下，用 p 替换 x 后条件集合等同于区间 (-5, 5)
    assert ConditionSet(p, n < x, Interval(-5, 5)).subs(x, p) == Interval(-5, 5), ConditionSet(p, n < x, Interval(-5, 5)).subs(x, p)

    # 断言条件集合，验证在 n < x 的条件下，用 p 替换 x 后条件集合等同于区间 (-oo, 0)
    assert ConditionSet(n, n < x, Interval(-oo, 0)).subs(x, p) == Interval(-oo, 0)

    # 断言条件集合，验证在 f(x) < 1 的条件下，用 y 替换 f(x) 后条件集合等同于原始条件集合
    assert ConditionSet(f(x), f(x) < 1, {w, z}).subs(f(x), y) == ConditionSet(f(x), f(x) < 1, {w, z})

    # issue 17341 的问题验证
    # 定义符号对象 k
    k = Symbol('k')
    # 定义图像集 img1 和 img2
    img1 = ImageSet(Lambda(k, 2*k*pi + asin(y)), S.Integers)
    img2 = ImageSet(Lambda(k, 2*k*pi + asin(S.One/3)), S.Integers)
    # 断言条件集合，验证在包含 y 在区间 (-1, 1) 的条件下，用 S.One/3 替换 y 后条件集合与 img2 相等
    assert ConditionSet(x, Contains(y, Interval(-1,1)), img1).subs(y, S.One/3).dummy_eq(img2)

    # 断言条件集合，验证在 x + y < 3 的条件下，(0, 1) 是否属于条件集合的元素 (x, y)
    assert (0, 1) in ConditionSet((x, y), x + y < 3, S.Integers**2)

    # 断言条件集合构造函数会引发 TypeError 异常，因为 n < -10 不符合 Interval(0, 10) 的条件
    raises(TypeError, lambda: ConditionSet(n, n < -10, Interval(0, 10)))
# 定义一个名为 test_subs_CondSet_tebr 的测试函数
def test_subs_CondSet_tebr():
    # 使用 warns_deprecated_sympy 上下文管理器，检查是否有关于 sympy 库的已弃用警告
    with warns_deprecated_sympy():
        # 断言：条件集合 ConditionSet 的结果应与另一个条件集合相等
        assert ConditionSet((x, y), {x + 1, x + y}, S.Reals**2) == \
            ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals**2)


# 定义一个名为 test_dummy_eq 的测试函数
def test_dummy_eq():
    # 将 ConditionSet 缩写为 C，将 S.Integers 缩写为 I
    C = ConditionSet
    I = S.Integers
    # 创建一个条件集合 c，条件为 x 小于 1，集合为整数集合 I
    c = C(x, x < 1, I)
    # 断言：条件集合 c 与另一个条件集合（y 小于 1，集合为整数集合 I）相等
    assert c.dummy_eq(C(y, y < 1, I))
    # 断言：条件集合 c 与整数 1 不相等
    assert c.dummy_eq(1) == False
    # 断言：条件集合 c 与条件集合（x 小于 1，集合为实数集合 S.Reals）不相等
    assert c.dummy_eq(C(x, x < 1, S.Reals)) == False

    # 创建条件集合 c1，条件为 x+1 等于 0 并且 x+y 等于 0，集合为 S.Reals 的二维
    c1 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals**2)
    # 创建条件集合 c2，条件与 c1 相同，集合也为 S.Reals 的二维
    c2 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals**2)
    # 创建条件集合 c3，条件与 c1 相同，但集合为 S.Complexes 的二维
    c3 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Complexes**2)
    # 断言：条件集合 c1 与条件集合 c2 相等
    assert c1.dummy_eq(c2)
    # 断言：条件集合 c1 与条件集合 c3 不相等
    assert c1.dummy_eq(c3) is False
    # 断言：条件集合 c 与条件集合 c1 不相等
    assert c.dummy_eq(c1) is False
    # 断言：条件集合 c1 与条件集合 c 不相等
    assert c1.dummy_eq(c) is False

    # issue 19496
    # 创建符号 m、n、a
    m = Symbol('m')
    n = Symbol('n')
    a = Symbol('a')
    # 创建图像集 d1，Lambda 表达式为 m*pi，集合为整数集合 S.Integers
    d1 = ImageSet(Lambda(m, m*pi), S.Integers)
    # 创建图像集 d2，Lambda 表达式为 n*pi，集合为整数集合 S.Integers
    d2 = ImageSet(Lambda(n, n*pi), S.Integers)
    # 创建条件集合 c1，条件为 a 不等于 0，集合为 d1
    c1 = ConditionSet(x, Ne(a, 0), d1)
    # 创建条件集合 c2，条件为 a 不等于 0，集合为 d2
    c2 = ConditionSet(x, Ne(a, 0), d2)
    # 断言：条件集合 c1 与条件集合 c2 相等
    assert c1.dummy_eq(c2)


# 定义一个名为 test_contains 的测试函数
def test_contains():
    # 断言：整数 6 是否包含在条件集合中，条件为 x 大于 5，集合为区间 [1, 7]
    assert 6 in ConditionSet(x, x > 5, Interval(1, 7))
    # 断言：整数 8 是否包含在条件集合中，条件为 y 大于 5，集合为区间 [1, 7]
    assert (8 in ConditionSet(x, y > 5, Interval(1, 7))) is False
    # 抛出 TypeError 异常，lambda 表达式中条件不足以给出 True 或 False 结果
    raises(TypeError,
           lambda: 6 in ConditionSet(x, y > 5, Interval(1, 7)))
    # 抛出 TypeError 异常，lambda 表达式中比较未定义
    raises(TypeError, lambda: 0 in ConditionSet(x, 1/x >= 0, S.Reals))
    # 断言：条件集合中是否包含整数 6，结果应为 y 大于 5
    assert ConditionSet(x, y > 5, Interval(1, 7)).contains(6) == (y > 5)
    # 断言：条件集合中是否包含整数 8，结果应为 False
    assert ConditionSet(x, y > 5, Interval(1, 7)).contains(8) is S.false
    # 断言：条件集合中是否包含符号 w，结果应为包含 w 在区间 [1, 7] 且 y 大于 5
    assert ConditionSet(x, y > 5, Interval(1, 7)).contains(w) == And(Contains(w, Interval(1, 7)), y > 5)
    # 断言：条件集合中是否包含整数 0，Lambda 表达式为 1/x 大于等于 0，集合为实数集合 S.Reals
    assert ConditionSet(x, 1/x >= 0, S.Reals).contains(0) == \
        Contains(0, ConditionSet(x, 1/x >= 0, S.Reals), evaluate=False)
    # 创建条件集合 c，条件为 x+y 大于 1，集合为整数集合的二维
    c = ConditionSet((x, y), x + y > 1, S.Integers**2)
    # 断言：条件集合 c 不包含整数 1
    assert not c.contains(1)
    # 断言：条件集合 c 包含元组 (2, 1)
    assert c.contains((2, 1))
    # 断言：条件集合 c 不包含元组 (0, 1)
    assert not c.contains((0, 1))
    # 创建条件集合 c，条件为 w+x+y 大于 1，集合为整数集合与整数集合的二维的笛卡尔积
    c = ConditionSet((w, (x, y)), w + x + y > 1, S.Integers * S.Integers**2)
    # 断言：条件集合 c 不包含整数 1
    assert not c.contains(1)
    # 断言：条件集合 c 不包含元组 (1, 2)
    assert not c.contains((1, 2))
    # 断言：条件集合 c 不包含元组 ((1, 2), 3)
    assert not c.contains(((1, 2), 3))
    # 断言：条件集合 c 不包含元组 ((1, 2), (3, 4))
    assert not c.contains(((1, 2), (3, 4)))
    # 断言：条件集合 c 包含元组 (1, (3, 4))


# 定义一个名为 test_as_relational 的测试函数
def test_as_relational():
    # 断言：条件集合中的 (x, y)，条件为 x 大于 1，集合为整数集合的二维，转换成关系表达式
    assert ConditionSet((x, y), x > 1, S.Integers**2).as_relational((x, y)) == \
        (x > 1) & Contains(x, S.Integers) & Contains(y, S.Integers)
    # 断言：条件集合中的 x，条件为 x 大于 1，集合为整数集合，转换成关系表达式
    assert ConditionSet
    # 断言语句，验证 outer 是否等于满足给定条件的 ConditionSet
    assert outer == ConditionSet(x, sin(x) + x > 0, S.Reals)
    
    # 创建一个内部的 ConditionSet，定义 y 满足 sin(y) + y > 0 的条件
    inner = ConditionSet(y, sin(y) + y > 0)
    
    # 使用 inner 条件作为外部 ConditionSet 的一部分，定义 x 满足包含在 inner 中，并且 x 属于实数集 S.Reals
    outer = ConditionSet(x, Contains(y, inner), S.Reals)
    
    # 断言语句，验证 outer 不等于满足给定条件的 ConditionSet
    assert outer != ConditionSet(x, sin(x) + x > 0, S.Reals)
    
    # 创建一个内部的 ConditionSet，定义 x 满足 sin(x) + x > 0，并且在区间 [-1, 1] 内
    inner = ConditionSet(x, sin(x) + x > 0).intersect(Interval(-1, 1))
    
    # 使用 inner 条件作为外部 ConditionSet 的一部分，定义 x 满足包含在 inner 中，并且 x 属于实数集 S.Reals
    outer = ConditionSet(x, Contains(x, inner), S.Reals)
    
    # 断言语句，验证 outer 等于满足给定条件的 ConditionSet
    assert outer == ConditionSet(x, sin(x) + x > 0, Interval(-1, 1))
# 测试函数，用于检测符号重复问题
def test_duplicate():
    # 导入异常类 BadSignatureError
    from sympy.core.function import BadSignatureError
    # 创建重复符号 'a,a' 的符号集合
    dup = symbols('a,a')
    # 断言在条件集合 ConditionSet 中使用重复符号会引发 BadSignatureError 异常
    raises(BadSignatureError, lambda: ConditionSet(dup, x < 0))


# 测试函数，验证 ConditionSet 类中的不同条件下的 SetKind
def test_SetKind_ConditionSet():
    # 断言在条件为 Eq(sin(x), 0) 的情况下，ConditionSet 返回的 kind 为 NumberKind
    assert ConditionSet(x, Eq(sin(x), 0), Interval(0, 2*pi)).kind is SetKind(NumberKind)
    # 断言在条件为 x < 0 的情况下，ConditionSet 返回的 kind 为 NumberKind
    assert ConditionSet(x, x < 0).kind is SetKind(NumberKind)
```