# `D:\src\scipysrc\sympy\sympy\tensor\tests\test_indexed.py`

```
# 从 sympy.core 模块中导入符号、符号、元组、无穷大和虚拟对象
from sympy.core import symbols, Symbol, Tuple, oo, Dummy
# 导入异常处理类 IndexException
from sympy.tensor.indexed import IndexException
# 导入 pytest 中的 raises 函数用于测试异常情况
from sympy.testing.pytest import raises
# 导入 iterable 函数，用于检查对象是否可迭代
from sympy.utilities.iterables import iterable

# 从 sympy.concrete.summations 模块中导入 Sum 类
from sympy.concrete.summations import Sum
# 从 sympy.core.function 模块中导入 Function、Subs 和 Derivative 类
from sympy.core.function import Function, Subs, Derivative
# 从 sympy.core.relational 模块中导入比较运算符类
from sympy.core.relational import (StrictLessThan, GreaterThan,
    StrictGreaterThan, LessThan)
# 从 sympy.core.singleton 模块中导入常量 S
from sympy.core.singleton import S
# 从 sympy.functions.elementary.exponential 模块中导入指数和对数函数 exp 和 log
from sympy.functions.elementary.exponential import exp, log
# 从 sympy.functions.elementary.trigonometric 模块中导入三角函数 cos 和 sin
from sympy.functions.elementary.trigonometric import cos, sin
# 从 sympy.functions.special.tensor_functions 模块中导入 KroneckerDelta 函数
from sympy.functions.special.tensor_functions import KroneckerDelta
# 从 sympy.series.order 模块中导入 Order 类
from sympy.series.order import Order
# 从 sympy.sets.fancysets 模块中导入 Range 类
from sympy.sets.fancysets import Range
# 从 sympy.tensor.indexed 模块中导入 IndexedBase、Idx 和 Indexed 类
from sympy.tensor.indexed import IndexedBase, Idx, Indexed


def test_Idx_construction():
    # 定义整数符号 i, a, b
    i, a, b = symbols('i a b', integer=True)
    # 检查 Idx(i) 和 Idx(i, 1) 不相等
    assert Idx(i) != Idx(i, 1)
    # 检查 Idx(i, a) 和 Idx(i, (0, a - 1)) 相等
    assert Idx(i, a) == Idx(i, (0, a - 1))
    # 检查 Idx(i, oo) 和 Idx(i, (0, oo)) 相等
    assert Idx(i, oo) == Idx(i, (0, oo))

    # 定义非整数符号 x
    x = symbols('x', integer=False)
    # 测试对非整数符号的 Idx 调用会抛出 TypeError 异常
    raises(TypeError, lambda: Idx(x))
    raises(TypeError, lambda: Idx(0.5))
    raises(TypeError, lambda: Idx(i, x))
    raises(TypeError, lambda: Idx(i, 0.5))
    raises(TypeError, lambda: Idx(i, (x, 5)))
    raises(TypeError, lambda: Idx(i, (2, x)))
    raises(TypeError, lambda: Idx(i, (2, 3.5)))


def test_Idx_properties():
    # 定义整数符号 i, a, b
    i, a, b = symbols('i a b', integer=True)
    # 检查 Idx(i).is_integer 是否为 True
    assert Idx(i).is_integer
    # 检查 Idx(i).name 是否为 'i'
    assert Idx(i).name == 'i'
    # 检查 Idx(i + 2).name 是否为 'i + 2'
    assert Idx(i + 2).name == 'i + 2'
    # 检查 Idx('foo').name 是否为 'foo'
    assert Idx('foo').name == 'foo'


def test_Idx_bounds():
    # 定义整数符号 i, a, b
    i, a, b = symbols('i a b', integer=True)
    # 检查 Idx(i).lower 是否为 None
    assert Idx(i).lower is None
    # 检查 Idx(i).upper 是否为 None
    assert Idx(i).upper is None
    # 检查 Idx(i, a).lower 是否为 0
    assert Idx(i, a).lower == 0
    # 检查 Idx(i, a).upper 是否为 a - 1
    assert Idx(i, a).upper == a - 1
    # 检查 Idx(i, 5).lower 是否为 0
    assert Idx(i, 5).lower == 0
    # 检查 Idx(i, 5).upper 是否为 4
    assert Idx(i, 5).upper == 4
    # 检查 Idx(i, oo).lower 是否为 0
    assert Idx(i, oo).lower == 0
    # 检查 Idx(i, oo).upper 是否为 oo
    assert Idx(i, oo).upper is oo
    # 检查 Idx(i, (a, b)).lower 是否为 a
    assert Idx(i, (a, b)).lower == a
    # 检查 Idx(i, (a, b)).upper 是否为 b
    assert Idx(i, (a, b)).upper == b
    # 检查 Idx(i, (1, 5)).lower 是否为 1
    assert Idx(i, (1, 5)).lower == 1
    # 检查 Idx(i, (1, 5)).upper 是否为 5
    assert Idx(i, (1, 5)).upper == 5
    # 检查 Idx(i, (-oo, oo)).lower 是否为 -oo
    assert Idx(i, (-oo, oo)).lower is -oo
    # 检查 Idx(i, (-oo, oo)).upper 是否为 oo
    assert Idx(i, (-oo, oo)).upper is oo


def test_Idx_fixed_bounds():
    # 定义整数符号 i, a, b, x
    i, a, b, x = symbols('i a b x', integer=True)
    # 检查 Idx(x).lower 是否为 None
    assert Idx(x).lower is None
    # 检查 Idx(x).upper 是否为 None
    assert Idx(x).upper is None
    # 检查 Idx(x, a).lower 是否为 0
    assert Idx(x, a).lower == 0
    # 检查 Idx(x, a).upper 是否为 a - 1
    assert Idx(x, a).upper == a - 1
    # 检查 Idx(x, 5).lower 是否为 0
    assert Idx(x, 5).lower == 0
    # 检查 Idx(x, 5).upper 是否为 4
    assert Idx(x, 5).upper == 4
    # 检查 Idx(x, oo).lower 是否为 0
    assert Idx(x, oo).lower == 0
    # 检查 Idx(x, oo).upper 是否为 oo
    assert Idx(x, oo).upper is oo
    # 检查 Idx(x, (a, b)).lower 是否为 a
    assert Idx(x, (a, b)).lower == a
    # 检查 Idx(x, (a, b)).upper 是否为 b
    assert Idx(x, (a, b)).upper == b
    # 检查 Idx(x, (1, 5)).lower 是否为 1
    assert Idx(x, (1, 5)).lower == 1
    # 检查 Idx(x, (1, 5)).upper 是否为 5
    assert Idx(x, (1, 5)).upper == 5
    # 检查 Idx(x, (-oo, oo)).lower 是否为 -oo
    assert Idx(x, (-oo, oo)).lower is -oo
    # 检查 Idx(x, (-oo, oo)).upper 是否为 oo
    assert Idx(x, (-oo, oo)).upper is oo


def test_Idx_inequalities():
    # 创建具有边界 (1, 4) 的 Idx 对象 i14, (7, 9) 的 Idx 对象 i79
    # (4, 6) 的 Idx 对象 i46 和 (3, 5) 的 Idx 对象 i35
    i14 = Idx("i14", (1, 4))
    i79 = Idx("i
    # 断言 i14 小于 5
    assert StrictLessThan(i14, 5)
    # 断言 i14 不大于 5
    assert not GreaterThan(i14, 5)
    # 断言 i14 不严格大于 5
    assert not StrictGreaterThan(i14, 5)

    # 断言 i14 小于等于 4
    assert i14 <= 4
    # 断言 i14 < 4 返回 StrictLessThan 类型
    assert isinstance(i14 < 4, StrictLessThan)
    # 断言 i14 >= 4 返回 GreaterThan 类型
    assert isinstance(i14 >= 4, GreaterThan)
    # 断言 i14 不大于 4
    assert not (i14 > 4)

    # 断言 i14 小于等于 1 返回 LessThan 类型
    assert isinstance(i14 <= 1, LessThan)
    # 断言 i14 不小于 1
    assert not (i14 < 1)
    # 断言 i14 大于等于 1
    assert i14 >= 1
    # 断言 i14 > 1 返回 StrictGreaterThan 类型
    assert isinstance(i14 > 1, StrictGreaterThan)

    # 断言 i14 不小于等于 0
    assert not (i14 <= 0)
    # 断言 i14 不小于 0
    assert not (i14 < 0)
    # 断言 i14 大于等于 0
    assert i14 >= 0
    # 断言 i14 大于 0
    assert i14 > 0

    # 导入 sympy.abc 中的 x 符号
    from sympy.abc import x

    # 断言 i14 < x 返回 StrictLessThan 类型
    assert isinstance(i14 < x, StrictLessThan)
    # 断言 i14 > x 返回 StrictGreaterThan 类型
    assert isinstance(i14 > x, StrictGreaterThan)
    # 断言 i14 小于等于 x 返回 LessThan 类型
    assert isinstance(i14 <= x, LessThan)
    # 断言 i14 大于等于 x 返回 GreaterThan 类型
    assert isinstance(i14 >= x, GreaterThan)

    # 断言 i14 小于 i79
    assert i14 < i79
    # 断言 i14 小于等于 i79
    assert i14 <= i79
    # 断言 i14 不大于 i79
    assert not (i14 > i79)
    # 断言 i14 不大于等于 i79
    assert not (i14 >= i79)

    # 断言 i14 小于等于 i46 返回 StrictLessThan 类型
    assert i14 <= i46
    # 断言 i14 小于 i46 返回 StrictLessThan 类型
    assert isinstance(i14 < i46, StrictLessThan)
    # 断言 i14 大于等于 i46 返回 GreaterThan 类型
    assert isinstance(i14 >= i46, GreaterThan)
    # 断言 i14 不大于 i46
    assert not (i14 > i46)

    # 断言 i14 小于 i35 返回 StrictLessThan 类型
    assert isinstance(i14 < i35, StrictLessThan)
    # 断言 i14 大于 i35 返回 StrictGreaterThan 类型
    assert isinstance(i14 > i35, StrictGreaterThan)
    # 断言 i14 小于等于 i35 返回 LessThan 类型
    assert isinstance(i14 <= i35, LessThan)
    # 断言 i14 大于等于 i35 返回 GreaterThan 类型
    assert isinstance(i14 >= i35, GreaterThan)

    # 创建 Idx 类型的变量 iNone1 和 iNone2
    iNone1 = Idx("iNone1")
    iNone2 = Idx("iNone2")

    # 断言 iNone1 小于 iNone2 返回 StrictLessThan 类型
    assert isinstance(iNone1 < iNone2, StrictLessThan)
    # 断言 iNone1 大于 iNone2 返回 StrictGreaterThan 类型
    assert isinstance(iNone1 > iNone2, StrictGreaterThan)
    # 断言 iNone1 小于等于 iNone2 返回 LessThan 类型
    assert isinstance(iNone1 <= iNone2, LessThan)
    # 断言 iNone1 大于等于 iNone2 返回 GreaterThan 类型
    assert isinstance(iNone1 >= iNone2, GreaterThan)
def test_Idx_inequalities_current_fails():
    # 创建一个名为 i14 的 Idx 对象，表示索引从 1 到 4 的范围
    i14 = Idx("i14", (1, 4))

    # 断言 5 大于等于 i14
    assert S(5) >= i14
    # 断言 5 大于 i14
    assert S(5) > i14
    # 断言 5 不小于等于 i14
    assert not (S(5) <= i14)
    # 断言 5 不小于 i14
    assert not (S(5) < i14)


def test_Idx_func_args():
    # 定义整数符号变量 i, a, b
    i, a, b = symbols('i a b', integer=True)
    # 创建一个不带范围的 Idx 对象 ii
    ii = Idx(i)
    # 断言 Idx 对象的 func 方法返回原始对象
    assert ii.func(*ii.args) == ii
    # 创建一个带有单一起始索引 a 的 Idx 对象 ii
    ii = Idx(i, a)
    # 断言 Idx 对象的 func 方法返回原始对象
    assert ii.func(*ii.args) == ii
    # 创建一个带有范围 (a, b) 的 Idx 对象 ii
    ii = Idx(i, (a, b))
    # 断言 Idx 对象的 func 方法返回原始对象
    assert ii.func(*ii.args) == ii


def test_Idx_subs():
    # 定义整数符号变量 i, a, b
    i, a, b = symbols('i a b', integer=True)
    # 断言替换 Idx(i, a) 中的 a 为 b，得到 Idx(i, b)
    assert Idx(i, a).subs(a, b) == Idx(i, b)
    # 断言替换 Idx(i, a) 中的 i 为 b，得到 Idx(b, a)
    assert Idx(i, a).subs(i, b) == Idx(b, a)

    # 断言替换 Idx(i) 中的 i 为 2，得到 Idx(2)
    assert Idx(i).subs(i, 2) == Idx(2)
    # 断言替换 Idx(i, a) 中的 a 为 2，得到 Idx(i, 2)
    assert Idx(i, a).subs(a, 2) == Idx(i, 2)
    # 断言替换带有范围 (a, b) 的 Idx(i, (a, b)) 中的 i 为 2，得到 Idx(2, (a, b))
    assert Idx(i, (a, b)).subs(i, 2) == Idx(2, (a, b))


def test_IndexedBase_sugar():
    # 定义整数符号变量 i, j
    i, j = symbols('i j', integer=True)
    # 定义符号变量 a
    a = symbols('a')
    # 创建 Indexed 对象 A1，等同于 IndexedBase(a)[i, j]
    A1 = Indexed(a, i, j)
    # 创建 IndexedBase 对象 A2
    A2 = IndexedBase(a)
    # 断言 A1 与 A2[i, j] 相等
    assert A1 == A2[i, j]
    # 断言 A1 与 A2[(i, j)] 相等
    assert A1 == A2[(i, j)]
    # 断言 A1 与 A2[[i, j]] 相等
    assert A1 == A2[[i, j]]
    # 断言 A1 与 A2[Tuple(i, j)] 相等
    assert A1 == A2[Tuple(i, j)]
    # 断言 A2[1, 0] 中的所有参数都是整数
    assert all(a.is_Integer for a in A2[1, 0].args[1:])


def test_IndexedBase_subs():
    # 定义整数符号变量 i
    i = symbols('i', integer=True)
    # 定义符号变量 a, b
    a, b = symbols('a b')
    # 创建 IndexedBase 对象 A 和 B
    A = IndexedBase(a)
    B = IndexedBase(b)
    # 断言 A[i] 等于 B[i].subs(b, a)
    assert A[i] == B[i].subs(b, a)
    # 定义字典 C
    C = {1: 2}
    # 断言 C[1] 等于 A[1].subs(A, C)
    assert C[1] == A[1].subs(A, C)


def test_IndexedBase_shape():
    # 定义整数符号变量 i, j, m, n
    i, j, m, n = symbols('i j m n', integer=True)
    # 创建形状为 (m, m) 的 IndexedBase 对象 a 和形状为 (m, n) 的 IndexedBase 对象 b
    a = IndexedBase('a', shape=(m, m))
    b = IndexedBase('a', shape=(m, n))
    # 断言 b 的形状是 Tuple(m, n)
    assert b.shape == Tuple(m, n)
    # 断言 a[i, j] 不等于 b[i, j]
    assert a[i, j] != b[i, j]
    # 断言 a[i, j] 等于 b[i, j].subs(n, m)
    assert a[i, j] == b[i, j].subs(n, m)
    # 断言 b 的 func 方法返回原始对象
    assert b.func(*b.args) == b
    # 断言 b[i, j] 的 func 方法返回原始对象
    assert b[i, j].func(*b[i, j].args) == b[i, j]
    # 断言调用 b[i] 和 b[i, i, j] 会引发 IndexException 异常
    raises(IndexException, lambda: b[i])
    raises(IndexException, lambda: b[i, i, j])
    # 创建形状为 m 的 IndexedBase 对象 F
    F = IndexedBase("F", shape=m)
    # 断言 F 的形状是 Tuple(m)
    assert F.shape == Tuple(m)
    # 断言 F[i].subs(i, j) 等于 F[j]
    assert F[i].subs(i, j) == F[j]
    # 断言调用 F[i, j] 会引发 IndexException 异常
    raises(IndexException, lambda: F[i, j])


def test_IndexedBase_assumptions():
    # 定义符号变量 i
    i = Symbol('i', integer=True)
    # 定义符号变量 a
    a = Symbol('a')
    # 创建具有 positive=True 属性的 IndexedBase 对象 A
    A = IndexedBase(a, positive=True)
    # 遍历 A 和 A[i]
    for c in (A, A[i]):
        # 断言 c 是实数
        assert c.is_real
        # 断言 c 是复数
        assert c.is_complex
        # 断言 c 不是虚数
        assert not c.is_imaginary
        # 断言 c 是非负数
        assert c.is_nonnegative
        # 断言 c 是非零数
        assert c.is_nonzero
        # 断言 c 是可交换的
        assert c.is_commutative
        # 断言 log(exp(c)) 等于 c
        assert log(exp(c)) == c

    # 断言 A 不等于 IndexedBase(a)
    assert A != IndexedBase(a)
    # 断言 A 等于具有 positive=True 和 real=True 属性的 IndexedBase(a)
    assert A == IndexedBase(a, positive=True, real=True)
    # 断言 A[i] 不等于 Indexed(a, i)
    assert A[i] != Indexed(a, i)


def test_IndexedBase_assumptions_inheritance():
    # 定义整数符号变量 I
    I = Symbol('I', integer=True)
    # 创建 IndexedBase 对象 I_inherit 和 I_explicit
    I_inherit = IndexedBase(I)
    I_explicit = IndexedBase('I', integer=True)

    # 断言 I_inherit 是整数
    assert I_inherit.is_integer
    # 断言 I_explicit 是整数
    assert I_explicit.is_integer
    # 断言 I_inherit 的标签是整数
    assert I_inherit.label.is_integer
    # 断言 I_explicit 的标签是整数
    assert I_explicit.label.is_integer
    #
def test_Indexed_constructor():
    # 定义符号变量 i, j，均为整数
    i, j = symbols('i j', integer=True)
    # 创建 Indexed 对象 A，表示符号 'A' 的索引为 (i, j)
    A = Indexed('A', i, j)
    # 断言 A 等于 Indexed(Symbol('A'), i, j)
    assert A == Indexed(Symbol('A'), i, j)
    # 断言 A 等于 Indexed(IndexedBase('A'), i, j)
    assert A == Indexed(IndexedBase('A'), i, j)
    # 检查是否引发 TypeError 异常，因为第一个参数 A 不是字符串或 IndexedBase 对象
    raises(TypeError, lambda: Indexed(A, i, j))
    # 检查是否引发 IndexException 异常，因为没有提供索引
    raises(IndexException, lambda: Indexed("A"))
    # 断言 A 的自由符号集合为 {A, A.base.label, i, j}
    assert A.free_symbols == {A, A.base.label, i, j}


def test_Indexed_func_args():
    # 定义符号变量 i, j，均为整数
    i, j = symbols('i j', integer=True)
    # 定义符号变量 a
    a = symbols('a')
    # 创建 Indexed 对象 A，表示符号 a 的索引为 (i, j)
    A = Indexed(a, i, j)
    # 断言 A 等于 A.func(*A.args)，即 A 本身与它的函数参数一致
    assert A == A.func(*A.args)


def test_Indexed_subs():
    # 定义符号变量 i, j, k，均为整数
    i, j, k = symbols('i j k', integer=True)
    # 定义符号变量 a, b
    a, b = symbols('a b')
    # 创建 IndexedBase 对象 A 和 B
    A = IndexedBase(a)
    B = IndexedBase(b)
    # 断言 A[i, j] 等于 B[i, j].subs(b, a)，即 B[i, j] 中的 b 替换为 a 后与 A[i, j] 相等
    assert A[i, j] == B[i, j].subs(b, a)
    # 断言 A[i, j] 等于 A[i, k].subs(k, j)，即 A[i, k] 中的 k 替换为 j 后与 A[i, j] 相等


def test_Indexed_properties():
    # 定义符号变量 i, j，均为整数
    i, j = symbols('i j', integer=True)
    # 创建 Indexed 对象 A，表示符号 'A' 的索引为 (i, j)
    A = Indexed('A', i, j)
    # 断言 A 的名称为 'A[i, j]'
    assert A.name == 'A[i, j]'
    # 断言 A 的秩为 2
    assert A.rank == 2
    # 断言 A 的索引为 (i, j)
    assert A.indices == (i, j)
    # 断言 A 的基础为 IndexedBase('A')
    assert A.base == IndexedBase('A')
    # 断言 A 的范围为 [None, None]
    assert A.ranges == [None, None]
    # 检查是否引发 IndexException 异常，因为没有提供形状信息
    raises(IndexException, lambda: A.shape)

    # 定义符号变量 n, m，均为整数
    n, m = symbols('n m', integer=True)
    # 断言 Indexed('A', Idx(i, m), Idx(j, n)).ranges 为 [Tuple(0, m - 1), Tuple(0, n - 1)]
    assert Indexed('A', Idx(i, m), Idx(j, n)).ranges == [Tuple(0, m - 1), Tuple(0, n - 1)]
    # 检查是否引发 IndexException 异常，因为没有提供形状信息
    raises(IndexException, lambda: Indexed("A", Idx(i, m), Idx(j)).shape)


def test_Indexed_shape_precedence():
    # 定义符号变量 i, j，均为整数
    i, j = symbols('i j', integer=True)
    # 定义符号变量 o, p，均为整数
    o, p = symbols('o p', integer=True)
    # 定义符号变量 n, m，均为整数
    n, m = symbols('n m', integer=True)
    # 创建形状为 (o, p) 的 IndexedBase 对象 a
    a = IndexedBase('a', shape=(o, p))
    # 断言 a 的形状为 Tuple(o, p)
    assert a.shape == Tuple(o, p)
    # 断言 Indexed(a, Idx(i, m), Idx(j, n)).ranges 为 [Tuple(0, m - 1), Tuple(0, n - 1)]
    assert Indexed(a, Idx(i, m), Idx(j, n)).ranges == [Tuple(0, m - 1), Tuple(0, n - 1)]
    # 断言 Indexed(a, Idx(i, m), Idx(j, n)).shape 为 Tuple(o, p)
    assert Indexed(a, Idx(i, m), Idx(j, n)).shape == Tuple(o, p)
    # 断言 Indexed(a, Idx(i, m), Idx(j)).ranges 为 [Tuple(0, m - 1), (None, None)]
    assert Indexed(a, Idx(i, m), Idx(j)).ranges == [Tuple(0, m - 1), (None, None)]
    # 断言 Indexed(a, Idx(i, m), Idx(j)).shape 为 Tuple(o, p)
    assert Indexed(a, Idx(i, m), Idx(j)).shape == Tuple(o, p)


def test_complex_indices():
    # 定义符号变量 i, j，均为整数
    i, j = symbols('i j', integer=True)
    # 创建 Indexed 对象 A，表示符号 'A' 的索引为 (i, i + j)
    A = Indexed('A', i, i + j)
    # 断言 A 的秩为 2
    assert A.rank == 2
    # 断言 A 的索引为 (i, i + j)
    assert A.indices == (i, i + j)


def test_not_interable():
    # 定义符号变量 i, j，均为整数
    i, j = symbols('i j', integer=True)
    # 创建 Indexed 对象 A，表示符号 'A' 的索引为 (i, i + j)
    A = Indexed('A', i, i + j)
    # 断言 A 不可迭代
    assert not iterable(A)


def test_Indexed_coeff():
    # 定义符号 N，为整数
    N = Symbol('N', integer=True)
    # 定义 len_y 为符号 N 的值
    len_y = N
    # 定义索引 i，其范围为 0 到 len_y-1
    i = Idx('i', len_y-1)
    # 创建形状为 (len_y,) 的 IndexedBase 对象 y
    y = IndexedBase('y', shape=(len_y,))
    # 计算 (1/y[i+1]*y[i]) 中 y[i] 的系数
    a = (1/y[i+1]*y[i]).coeff(y[i])
    # 计算 (y[i]/y[i+1]) 中 y[i] 的系数
    b = (y[i]/y[i+1]).coeff(y[i])
    # 断言 a 等于 b
    assert a == b


def test_differentiation():
    # 导入 KroneckerDelta 函数
    from sympy.functions.special.tensor_functions import KroneckerDelta
    # 定义符号变量 i, j, k, l，均为 Idx 类型
    i, j, k, l = symbols('i j k l', cls=Idx)
    # 定义符号变量 a
    a = symbols('a')
    # 定义符号变量 m, n，均为整数且有限
    m, n = symbols("m, n", integer=True, finite=True)
    # 断言 m 是实数
    assert m.is_real
    # 定义 IndexedBase 对象 h 和 L
    h, L = symbols('h L', cls=IndexedBase)
    # 定义 hi 和 hj 分别为 h[i] 和 h[j]
    hi, hj = h[i], h[j]

    #
    # 断言：求和表达式关于变量 hj 的偏导数等于对应的 Kronecker delta 函数的求和
    assert Sum(expr, (i, -oo, oo)).diff(hj) == Sum(2*KroneckerDelta(i, j), (i, -oo, oo))
    
    # 断言：求和表达式关于变量 hj 的偏导数的求和等于常数乘以 Kronecker delta 函数的求和
    assert Sum(expr.diff(hj), (i, -oo, oo)) == Sum(2*KroneckerDelta(i, j), (i, -oo, oo))
    
    # 断言：求和表达式关于变量 hj 的偏导数的求和并求值等于 2
    assert Sum(expr, (i, -oo, oo)).diff(hj).doit() == 2

    # 断言：求和表达式关于变量 hi 的偏导数并求值等于常数乘以求和的求值
    assert Sum(expr.diff(hi), (i, -oo, oo)).doit() == Sum(2, (i, -oo, oo)).doit()

    # 断言：求和表达式关于变量 hi 的偏导数的求值是正无穷大
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() is oo

    # 设置表达式为 a * hj * hj / 2
    expr = a * hj * hj / S(2)

    # 断言：表达式关于变量 hi 的偏导数等于给定的乘积
    assert expr.diff(hi) == a * h[j] * KroneckerDelta(i, j)

    # 断言：表达式关于变量 a 的偏导数等于表达式的一部分
    assert expr.diff(a) == hj * hj / S(2)

    # 断言：对表达式进行两次 a 的偏导数得到零
    assert expr.diff(a, 2) is S.Zero

    # 断言：求和表达式关于变量 hi 的偏导数等于给定的乘积的求和
    assert Sum(expr, (i, -oo, oo)).diff(hi) == Sum(a*KroneckerDelta(i, j)*h[j], (i, -oo, oo))

    # 断言：求和表达式关于变量 hi 的偏导数的求和等于给定的乘积的求和
    assert Sum(expr.diff(hi), (i, -oo, oo)) == Sum(a*KroneckerDelta(i, j)*h[j], (i, -oo, oo))

    # 断言：求和表达式关于变量 hi 的偏导数的求和并求值等于给定的乘积
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() == a*h[j]

    # 断言：求和表达式关于变量 hi 的偏导数的求和等于给定的乘积的求和（对 j 进行求和）
    assert Sum(expr, (j, -oo, oo)).diff(hi) == Sum(a*KroneckerDelta(i, j)*h[j], (j, -oo, oo))

    # 断言：求和表达式关于变量 hi 的偏导数的求和并求值等于给定的乘积（对 j 进行求和）
    assert Sum(expr.diff(hi), (j, -oo, oo)) == Sum(a*KroneckerDelta(i, j)*h[j], (j, -oo, oo))

    # 断言：求和表达式关于变量 hi 的偏导数的求和并求值等于给定的乘积的求和（对 j 进行求和）的结果
    assert Sum(expr, (j, -oo, oo)).diff(hi).doit() == a*h[i]

    # 设置表达式为 a * sin(hj * hj)
    expr = a * sin(hj * hj)

    # 断言：表达式关于变量 hi 的偏导数等于给定的乘积
    assert expr.diff(hi) == 2*a*cos(hj * hj) * hj * KroneckerDelta(i, j)

    # 断言：表达式关于变量 hj 的偏导数等于给定的乘积
    assert expr.diff(hj) == 2*a*cos(hj * hj) * hj

    # 设置表达式为 a * L[i, j] * h[j]
    expr = a * L[i, j] * h[j]

    # 断言：表达式关于变量 hi 的偏导数等于给定的乘积
    assert expr.diff(hi) == a*L[i, j]*KroneckerDelta(i, j)

    # 断言：表达式关于变量 hj 的偏导数等于给定的乘积
    assert expr.diff(hj) == a*L[i, j]

    # 断言：表达式关于变量 L[i, j] 的偏导数等于给定的乘积
    assert expr.diff(L[i, j]) == a*h[j]

    # 断言：表达式关于变量 L[k, l] 的偏导数等于给定的乘积
    assert expr.diff(L[k, l]) == a*KroneckerDelta(i, k)*KroneckerDelta(j, l)*h[j]

    # 断言：表达式关于变量 L[i, l] 的偏导数等于给定的乘积
    assert expr.diff(L[i, l]) == a*KroneckerDelta(j, l)*h[j]

    # 断言：求和表达式关于变量 L[k, l] 的偏导数等于给定的乘积的求和
    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]) == Sum(a * KroneckerDelta(i, k) * KroneckerDelta(j, l) * h[j], (j, -oo, oo))

    # 断言：求和表达式关于变量 L[k, l] 的偏导数的求和并求值等于给定的乘积的求和的结果
    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]).doit() == a * KroneckerDelta(i, k) * h[l]

    # 断言：变量 h[m] 关于其自身的偏导数等于 1
    assert h[m].diff(h[m]) == 1

    # 断言：变量 h[m] 关于变量 h[n] 的偏导数等于 Kronecker delta 函数
    assert h[m].diff(h[n]) == KroneckerDelta(m, n)

    # 断言：求和表达式关于变量 h[m] 的偏导数等于给定的乘积的求和
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[n]) == Sum(a*KroneckerDelta(m, n), (m, -oo, oo))

    # 断言：求和表达式关于变量 h[m] 的偏导数的求和并求值等于给定的乘积的求和的结果
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[n]).doit() == a

    # 断言：求和表达式关于变量 h[m] 的偏导数等于正无穷大乘以 a
    assert Sum(a*h[m], (n, -oo, oo)).diff(h[m]).doit() == oo*a
def test_indexed_series():
    # 创建一个 IndexedBase 对象 A，用于表示索引化的符号表达式
    A = IndexedBase("A")
    # 创建一个整数符号 i，用于索引 A 中的元素
    i = symbols("i", integer=True)
    # 断言：计算 sin(A[i]) 的 Taylor 展开，并验证结果
    assert sin(A[i]).series(A[i]) == A[i] - A[i]**3/6 + A[i]**5/120 + Order(A[i]**6, A[i])


def test_indexed_is_constant():
    # 创建一个 IndexedBase 对象 A，用于表示索引化的符号表达式
    A = IndexedBase("A")
    # 创建整数符号 i, j, k，用于索引 A 中的元素
    i, j, k = symbols("i,j,k")
    # 断言：A[i] 是否是常数
    assert not A[i].is_constant()
    # 断言：A[i] 在 j 方向是否是常数
    assert A[i].is_constant(j)
    # 断言：A[1+2*i, k] 是否是常数
    assert not A[1+2*i, k].is_constant()
    # 断言：A[1+2*i, k] 在 i 方向是否是常数
    assert not A[1+2*i, k].is_constant(i)
    # 断言：A[1+2*i, k] 在 j 方向是否是常数
    assert A[1+2*i, k].is_constant(j)
    # 断言：Indexed(Range(5), 2) 是否等于 2
    assert Indexed(Range(5), 2) == 2


def test_issue_12533():
    # 创建一个 IndexedBase 对象 d，用于表示索引化的符号表达式
    d = IndexedBase('d')
    # 断言：IndexedBase(range(5)) 是否等于 Range(0, 5, 1)
    assert IndexedBase(range(5)) == Range(0, 5, 1)
    # 断言：用符号替换 d[0] 的符号 d 为 range(5)，并验证结果为 0
    assert d[0].subs(Symbol("d"), range(5)) == 0
    # 断言：用 d 替换 d[0] 的符号 d 为 range(5)，并验证结果为 0
    assert d[0].subs(d, range(5)) == 0
    # 断言：用 d 替换 d[1] 的符号 d 为 range(5)，并验证结果为 1
    assert d[1].subs(d, range(5)) == 1
    # 断言：Indexed(Range(5), 2) 是否等于 2
    assert Indexed(Range(5), 2) == 2


def test_issue_12780():
    # 创建符号 n
    n = symbols("n")
    # 创建一个索引对象 i，其范围为 (0, n)
    i = Idx("i", (0, n))
    # 验证：调用 i.subs(n, 1.5) 是否会引发 TypeError 异常
    raises(TypeError, lambda: i.subs(n, 1.5))


def test_issue_18604():
    # 创建符号 m
    m = symbols("m")
    # 断言：Idx("i", m) 的名称是否为 'i'
    assert Idx("i", m).name == 'i'
    # 断言：Idx("i", m) 的下限是否为 0
    assert Idx("i", m).lower == 0
    # 断言：Idx("i", m) 的上限是否为 m - 1
    assert Idx("i", m).upper == m - 1
    # 创建一个非实数符号 m
    m = symbols("m", real=False)
    # 验证：调用 Idx("i", m) 是否会引发 TypeError 异常
    raises(TypeError, lambda: Idx("i", m))


def test_Subs_with_Indexed():
    # 创建一个 IndexedBase 对象 A，用于表示索引化的符号表达式
    A = IndexedBase("A")
    # 创建整数符号 i, j, k，用于索引 A 中的元素
    i, j, k = symbols("i,j,k")
    # 创建符号 x, y, z
    x, y, z = symbols("x,y,z")
    # 创建一个函数符号 f
    f = Function("f")

    # 断言：对 Subs(A[i], A[i], A[j]) 求 A[j] 的导数
    assert Subs(A[i], A[i], A[j]).diff(A[j]) == 1
    # 断言：对 Subs(A[i], A[i], x) 求 A[i] 的导数
    assert Subs(A[i], A[i], x).diff(A[i]) == 0
    # 断言：对 Subs(A[i], A[i], x) 求 A[j] 的导数
    assert Subs(A[i], A[i], x).diff(A[j]) == 0
    # 断言：对 Subs(A[i], A[i], x) 求 x 的导数
    assert Subs(A[i], A[i], x).diff(x) == 1
    # 断言：对 Subs(A[i], A[i], x) 求 y 的导数
    assert Subs(A[i], A[i], x).diff(y) == 0
    # 断言：对 Subs(A[i], A[i], A[j]) 求 A[k] 的导数
    assert Subs(A[i], A[i], A[j]).diff(A[k]) == KroneckerDelta(j, k)
    # 断言：对 Subs(x, x, A[i]) 求 A[j] 的导数
    assert Subs(x, x, A[i]).diff(A[j]) == KroneckerDelta(i, j)
    # 断言：对 Subs(f(A[i]), A[i], x) 求 A[j] 的导数
    assert Subs(f(A[i]), A[i], x).diff(A[j]) == 0
    # 断言：对 Subs(f(A[i]), A[i], A[k]) 求 A[j] 的导数
    assert Subs(f(A[i]), A[i], A[k]).diff(A[j]) == Derivative(f(A[k]), A[k])*KroneckerDelta(j, k)
    # 断言：对 Subs(x, x, A[i]**2) 求 A[j] 的导数
    assert Subs(x, x, A[i]**2).diff(A[j]) == 2*KroneckerDelta(i, j)*A[i]
    # 断言：对 Subs(A[i], A[i], A[j]**2) 求 A[k] 的导数
    assert Subs(A[i], A[i], A[j]**2).diff(A[k]) == 2*KroneckerDelta(j, k)*A[j]

    # 断言：对 Subs(A[i]*x, x, A[i]) 求 A[i] 的导数
    assert Subs(A[i]*x, x, A[i]).diff(A[i]) == 2*A[i]
    # 断言：对 Subs(A[i]*x, x, A[i]) 求 A[j] 的导数
    assert Subs(A[i]*x, x, A[i]).diff(A[j]) == 2*A[i]*KroneckerDelta(i, j)
    # 断言：对 Subs(A[i]*x, x, A[j]) 求 A[i] 的导数
    assert Subs(A[i]*x, x, A[j]).diff(A[i]) == A[j] + A[i]*KroneckerDelta(i, j)
    # 断言：对 Subs(A[i]*x, x, A[j]) 求 A[j] 的导数
    assert Subs(A[i]*x, x, A[j]).diff(A[j]) == A[i] + A[j]*KroneckerDelta(i, j)
    # 断言：对 Subs(A[i]*x, x, A[i]) 求 A[k] 的导数
    assert Subs(A[i]*x, x, A[i]).diff(A[k]) == 2*A[i]*KroneckerDelta(i, k)
    # 断言：对 Subs(A[i]*x, x, A[j]) 求 A[k] 的导数
    assert Subs(A[i]*x, x, A[j]).diff(A[k]) == KroneckerDelta(i, k)*A[j] + KroneckerDelta(j, k)*A[i]

    # 断言：对 Subs(A[i]*x, A[i], x) 求 A[i] 的导数
    assert Subs(A[i]*x
    # 定义一个名为 _xi_1 的符号，用于符号计算
    _xi_1 = symbols("xi_1", cls=Dummy)
    # 断言表达式关于 x[m0] 的偏导数与给定表达式相等
    assert expr.diff(x[m0]).dummy_eq(
        # 表达式的偏导数，根据链式法则和符号替换规则计算得出
        (x[i] - y[i]) * KroneckerDelta(i, m0) * \
        2 * Subs(
            # 对 f(_xi_1) 求关于 _xi_1 的导数，并在 (_xi_1,) 处进行符号替换
            Derivative(f(_xi_1), _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2 / sigma,)
        ) / sigma
    )
    # 断言表达式关于 x[m0] 和 x[m1] 的二阶偏导数与给定表达式相等
    assert expr.diff(x[m0]).diff(x[m1]).dummy_eq(
        2 * KroneckerDelta(i, m0) * \
        KroneckerDelta(i, m1) * Subs(
            # 对 f(_xi_1) 求关于 _xi_1 的导数，并在 (_xi_1,) 处进行符号替换
            Derivative(f(_xi_1), _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2 / sigma,)
         ) / sigma + \
        4 * (x[i] - y[i])**2 * KroneckerDelta(i, m0) * KroneckerDelta(i, m1) * \
        Subs(
            # 对 f(_xi_1) 求关于 _xi_1 的二阶导数，并在 (_xi_1,) 处进行符号替换
            Derivative(f(_xi_1), _xi_1, _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2 / sigma,)
        ) / sigma**2
    )
# 定义一个测试函数，用于验证 IndexedBase 对象的交换性
def test_IndexedBase_commutative():
    # 创建 IndexedBase 对象 t，指定为非交换的
    t = IndexedBase('t', commutative=False)
    # 创建 IndexedBase 对象 u，指定为非交换的
    u = IndexedBase('u', commutative=False)
    # 创建 IndexedBase 对象 v，默认为交换的
    v = IndexedBase('v')
    # 断言：t[0]*v[0] 应该等于 v[0]*t[0]，验证交换性
    assert t[0]*v[0] == v[0]*t[0]
    # 断言：t[0]*u[0] 不应该等于 u[0]*t[0]，验证非交换性
    assert t[0]*u[0] != u[0]*t[0]
```