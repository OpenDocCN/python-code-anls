# `D:\src\scipysrc\sympy\sympy\core\tests\test_containers.py`

```
# 导入必要的模块和类
from collections import defaultdict

from sympy.core.basic import Basic  # 导入基本类 Basic
from sympy.core.containers import (Dict, Tuple)  # 导入容器类 Dict 和 Tuple
from sympy.core.numbers import Integer  # 导入整数类 Integer
from sympy.core.kind import NumberKind  # 导入数字种类类 NumberKind
from sympy.matrices.kind import MatrixKind  # 导入矩阵种类类 MatrixKind
from sympy.core.singleton import S  # 导入单例类 S
from sympy.core.symbol import symbols  # 导入符号类 symbols
from sympy.core.sympify import sympify  # 导入 sympify 函数
from sympy.matrices.dense import Matrix  # 导入密集矩阵类 Matrix
from sympy.sets.sets import FiniteSet  # 导入有限集类 FiniteSet
from sympy.core.containers import tuple_wrapper, TupleKind  # 导入元组包装器和元组种类类 TupleKind
from sympy.core.expr import unchanged  # 导入未改变表达式类 unchanged
from sympy.core.function import Function, Lambda  # 导入函数类 Function 和 Lambda
from sympy.core.relational import Eq  # 导入相等关系类 Eq
from sympy.testing.pytest import raises  # 导入 pytest 的 raises 函数
from sympy.utilities.iterables import is_sequence, iterable  # 导入检测可迭代性的函数 is_sequence 和 iterable

from sympy.abc import x, y  # 导入符号 x 和 y


def test_Tuple():
    t = (1, 2, 3, 4)
    st = Tuple(*t)  # 创建一个元组对象 st，包含元组 t 中的元素
    assert set(sympify(t)) == set(st)  # 检查 sympify 后的 t 和 st 的集合是否相同
    assert len(t) == len(st)  # 检查 t 和 st 的长度是否相同
    assert set(sympify(t[:2])) == set(st[:2])  # 检查 t 和 st 的前两个元素的集合是否相同
    assert isinstance(st[:], Tuple)  # 检查 st[:] 是否是 Tuple 类的实例
    assert st == Tuple(1, 2, 3, 4)  # 检查 st 是否等于 Tuple(1, 2, 3, 4)
    assert st.func(*st.args) == st  # 检查通过 st.func(*st.args) 是否能够重新构造出 st
    p, q, r, s = symbols('p q r s')
    t2 = (p, q, r, s)
    st2 = Tuple(*t2)  # 创建另一个元组对象 st2，包含元组 t2 中的元素
    assert st2.atoms() == set(t2)  # 检查 st2 的原子集合是否与 t2 的集合相同
    assert st == st2.subs({p: 1, q: 2, r: 3, s: 4})  # 检查将 p, q, r, s 替换为 1, 2, 3, 4 后 st 是否与 st2 相等
    # issue 5505
    assert all(isinstance(arg, Basic) for arg in st.args)  # 检查 st 的每个参数是否都是 Basic 类的实例
    assert Tuple(p, 1).subs(p, 0) == Tuple(0, 1)  # 检查将 p 替换为 0 后 Tuple(p, 1) 是否等于 Tuple(0, 1)
    assert Tuple(p, Tuple(p, 1)).subs(p, 0) == Tuple(0, Tuple(0, 1))  # 检查将 p 替换为 0 后 Tuple(p, Tuple(p, 1)) 是否等于 Tuple(0, Tuple(0, 1))

    assert Tuple(t2) == Tuple(Tuple(*t2))  # 检查通过两种方式构造的 Tuple(t2) 是否相等
    assert Tuple.fromiter(t2) == Tuple(*t2)  # 检查通过迭代器构造的 Tuple 是否与直接传入元组的 Tuple 相等
    assert Tuple.fromiter(x for x in range(4)) == Tuple(0, 1, 2, 3)  # 检查通过迭代器构造的 Tuple 是否与预期的 Tuple 相等
    assert st2.fromiter(st2.args) == st2  # 检查通过 st2.args 构造的 Tuple 是否与 st2 相等


def test_Tuple_contains():
    t1, t2 = Tuple(1), Tuple(2)
    assert t1 in Tuple(1, 2, 3, t1, Tuple(t2))  # 检查 t1 是否在包含 t1 和 t2 的 Tuple 中
    assert t2 not in Tuple(1, 2, 3, t1, Tuple(t2))  # 检查 t2 是否不在包含 t1 和 t2 的 Tuple 中


def test_Tuple_concatenation():
    assert Tuple(1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)  # 检查元组连接操作是否正确
    assert (1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)  # 检查元组与普通元组连接操作是否正确
    assert Tuple(1, 2) + (3, 4) == Tuple(1, 2, 3, 4)  # 检查元组与普通元组连接操作是否正确
    raises(TypeError, lambda: Tuple(1, 2) + 3)  # 检查当与非元组对象连接时是否会引发 TypeError
    raises(TypeError, lambda: 1 + Tuple(2, 3))  # 检查当非元组对象与元组连接时是否会引发 TypeError

    # the Tuple case in __radd__ is only reached when a subclass is involved
    class Tuple2(Tuple):
        def __radd__(self, other):
            return Tuple.__radd__(self, other + other)
    assert Tuple(1, 2) + Tuple2(3, 4) == Tuple(1, 2, 1, 2, 3, 4)  # 检查自定义 __radd__ 方法的正确性
    assert Tuple2(1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)  # 检查自定义 __radd__ 方法的正确性


def test_Tuple_equality():
    assert not isinstance(Tuple(1, 2), tuple)  # 检查 Tuple(1, 2) 是否不是 Python 内置的元组类的实例
    assert (Tuple(1, 2) == (1, 2)) is True  # 检查 Tuple(1, 2) 是否等于 (1, 2)
    assert (Tuple(1, 2) != (1, 2)) is False  # 检查 Tuple(1, 2) 是否不等于 (1, 2)
    assert (Tuple(1, 2) == (1, 3)) is False  # 检查 Tuple(1, 2) 是否不等于 (1, 3)
    assert (Tuple(1, 2) != (1, 3)) is True  # 检查 Tuple(1, 2) 是否不等于 (1, 3)
    assert (Tuple(1, 2) == Tuple(1, 2)) is True  # 检查 Tuple(1, 2) 是否等于 Tuple(1, 2)
    assert (Tuple(1, 2) != Tuple(1, 2)) is False  # 检查 Tuple(1, 2) 是否不等于 Tuple(1, 2)
    assert (Tuple(1, 2) == Tuple(1, 3)) is False  # 检查 Tuple(1, 2) 是否不等于 Tuple(1, 3)
    assert (Tuple(1, 2) != Tuple(1, 3)) is True  # 检查 Tuple(1, 2) 是否不等于 Tuple(1, 3)


def test_Tuple_Eq():
    assert Eq(Tuple(), Tuple()) is S.true  # 检查空元组是否等于空元组
    assert Eq(Tuple(1), 1) is S.false
    # 断言：比较两个元组是否相等，期望结果是逻辑假（False）
    assert Eq(Tuple(1, 2), Tuple(1, 3)) is S.false
    
    # 断言：比较两个元组是否相等，期望结果是逻辑真（True）
    assert Eq(Tuple(1, 2), Tuple(1, 2)) is S.true
    
    # 断言：检查未改变的函数应用结果是否与预期的相等
    assert unchanged(Eq, Tuple(1, x), Tuple(1, 2))
    
    # 断言：在将变量 x 替换为 2 后，比较两个元组是否相等，期望结果是逻辑真（True）
    assert Eq(Tuple(1, x), Tuple(1, 2)).subs(x, 2) is S.true
    
    # 断言：检查未改变的函数应用结果是否与预期的相等
    assert unchanged(Eq, Tuple(1, 2), x)
    
    # 创建一个函数符号 'f'
    f = Function('f')
    
    # 断言：检查未改变的函数应用结果是否与预期的相等
    assert unchanged(Eq, Tuple(1), f(x))
    
    # 断言：在将变量 x 替换为 1，并将函数 f 替换为返回元组的 Lambda 函数后，比较结果是否为逻辑真（True）
    assert Eq(Tuple(1), f(x)).subs(x, 1).subs(f, Lambda(y, (y,))) is S.true
def test_Tuple_comparision():
    # 断言元组比较，检查元组 (1, 3) 是否大于等于 (-10, 30)，返回值为 S.true
    assert (Tuple(1, 3) >= Tuple(-10, 30)) is S.true
    # 断言元组比较，检查元组 (1, 3) 是否小于等于 (-10, 30)，返回值为 S.false
    assert (Tuple(1, 3) <= Tuple(-10, 30)) is S.false
    # 断言元组比较，检查元组 (1, 3) 是否大于等于 (1, 3)，返回值为 S.true
    assert (Tuple(1, 3) >= Tuple(1, 3)) is S.true
    # 断言元组比较，检查元组 (1, 3) 是否小于等于 (1, 3)，返回值为 S.true
    assert (Tuple(1, 3) <= Tuple(1, 3)) is S.true


def test_Tuple_tuple_count():
    # 断言 Tuple(0, 1, 2, 3) 中数字 4 出现的次数为 0
    assert Tuple(0, 1, 2, 3).tuple_count(4) == 0
    # 断言 Tuple(0, 4, 1, 2, 3) 中数字 4 出现的次数为 1
    assert Tuple(0, 4, 1, 2, 3).tuple_count(4) == 1
    # 断言 Tuple(0, 4, 1, 4, 2, 3) 中数字 4 出现的次数为 2
    assert Tuple(0, 4, 1, 4, 2, 3).tuple_count(4) == 2
    # 断言 Tuple(0, 4, 1, 4, 2, 4, 3) 中数字 4 出现的次数为 3
    assert Tuple(0, 4, 1, 4, 2, 4, 3).tuple_count(4) == 3


def test_Tuple_index():
    # 断言在 Tuple(4, 0, 1, 2, 3) 中查找数字 4 的索引为 0
    assert Tuple(4, 0, 1, 2, 3).index(4) == 0
    # 断言在 Tuple(0, 4, 1, 2, 3) 中查找数字 4 的索引为 1
    assert Tuple(0, 4, 1, 2, 3).index(4) == 1
    # 断言在 Tuple(0, 1, 4, 2, 3) 中查找数字 4 的索引为 2
    assert Tuple(0, 1, 4, 2, 3).index(4) == 2
    # 断言在 Tuple(0, 1, 2, 4, 3) 中查找数字 4 的索引为 3
    assert Tuple(0, 1, 2, 4, 3).index(4) == 3
    # 断言在 Tuple(0, 1, 2, 3, 4) 中查找数字 4 的索引为 4
    assert Tuple(0, 1, 2, 3, 4).index(4) == 4

    # 使用 lambda 函数检查以下情况会触发 ValueError 异常
    raises(ValueError, lambda: Tuple(0, 1, 2, 3).index(4))
    raises(ValueError, lambda: Tuple(4, 0, 1, 2, 3).index(4, 1))
    raises(ValueError, lambda: Tuple(0, 1, 2, 3, 4).index(4, 1, 4))


def test_Tuple_mul():
    # 断言 Tuple(1, 2, 3) 乘以 2 得到 Tuple(1, 2, 3, 1, 2, 3)
    assert Tuple(1, 2, 3)*2 == Tuple(1, 2, 3, 1, 2, 3)
    # 断言 2 乘以 Tuple(1, 2, 3) 得到 Tuple(1, 2, 3, 1, 2, 3)
    assert 2*Tuple(1, 2, 3) == Tuple(1, 2, 3, 1, 2, 3)
    # 断言 Tuple(1, 2, 3) 乘以 Integer(2) 得到 Tuple(1, 2, 3, 1, 2, 3)
    assert Tuple(1, 2, 3)*Integer(2) == Tuple(1, 2, 3, 1, 2, 3)
    # 断言 Integer(2) 乘以 Tuple(1, 2, 3) 得到 Tuple(1, 2, 3, 1, 2, 3)
    assert Integer(2)*Tuple(1, 2, 3) == Tuple(1, 2, 3, 1, 2, 3)

    # 使用 lambda 函数检查以下情况会触发 TypeError 异常
    raises(TypeError, lambda: Tuple(1, 2, 3)*S.Half)
    raises(TypeError, lambda: S.Half*Tuple(1, 2, 3))


def test_tuple_wrapper():
    # 定义一个装饰器，将输入的参数转换为元组形式并返回
    @tuple_wrapper
    def wrap_tuples_and_return(*t):
        return t

    p = symbols('p')
    # 断言 wrap_tuples_and_return(p, 1) 返回值为 (p, 1)
    assert wrap_tuples_and_return(p, 1) == (p, 1)
    # 断言 wrap_tuples_and_return((p, 1)) 返回值为 (Tuple(p, 1),)
    assert wrap_tuples_and_return((p, 1)) == (Tuple(p, 1),)
    # 断言 wrap_tuples_and_return(1, (p, 2), 3) 返回值为 (1, Tuple(p, 2), 3)
    assert wrap_tuples_and_return(1, (p, 2), 3) == (1, Tuple(p, 2), 3)


def test_iterable_is_sequence():
    ordered = [[], (), Tuple(), Matrix([[]])]
    unordered = [set()]
    not_sympy_iterable = [{}, '', '']
    # 断言 ordered 中的元素都是序列
    assert all(is_sequence(i) for i in ordered)
    # 断言 unordered 中的元素都不是序列
    assert all(not is_sequence(i) for i in unordered)
    # 断言 ordered + unordered 中的所有元素都是可迭代的
    assert all(iterable(i) for i in ordered + unordered)
    # 断言 not_sympy_iterable 中的所有元素都不是可迭代的
    assert all(not iterable(i) for i in not_sympy_iterable)
    # 断言 not_sympy_iterable 中的所有元素都不是 SymPy 的可迭代对象
    assert all(iterable(i, exclude=None) for i in not_sympy_iterable)


def test_TupleKind():
    # 定义一个 TupleKind 对象，包含 NumberKind 和 MatrixKind(NumberKind)
    kind = TupleKind(NumberKind, MatrixKind(NumberKind))
    # 断言 Tuple(1, Matrix([1, 2])) 的 kind 与 kind 相同
    assert Tuple(1, Matrix([1, 2])).kind is kind
    # 断言 Tuple(1, 2) 的 kind 是 TupleKind(NumberKind, NumberKind)
    assert Tuple(1, 2).kind is TupleKind(NumberKind, NumberKind)
    # 断言 Tuple(1, 2) 的 kind.element_kind 是 (NumberKind, NumberKind)
    assert Tuple(1, 2).kind.element_kind == (NumberKind, NumberKind)


def test_Dict():
    x, y, z = symbols('x y z')
    # 创建一个字典对象 d，包含键值对 {x: 1, y: 2, z: 3}
    d = Dict({x: 1, y: 2, z: 3})
    # 断言 d[x] 的值为 1
    assert d[x] == 1
    # 断言 d[y] 的值为 2
    assert d[y] == 2
    # 使用 lambda 函数检查以下情况会触发 KeyError
    # 使用 raises 函数检查是否抛出 TypeError 异常，传入的 lambda 表达式调用 Dict 构造函数尝试创建字典
    raises(TypeError, lambda: Dict(((x, 1), (y, 2), (z, 3))))

    # 使用 with 语句和 raises 函数检查是否抛出 NotImplementedError 异常，尝试在字典 d 上执行 d[5] = 6 操作以验证其不可变性
    with raises(NotImplementedError):
        d[5] = 6  # assert immutability

    # 断言字典 d 的项以集合形式等于预期的 Tuple 对象集合，表示字典内容与预期的键值对一致
    assert set(d.items()) == {Tuple(x, S.One), Tuple(y, S(2)), Tuple(z, S(3))}
    
    # 断言集合 d 的键以集合形式等于预期的 {x, y, z}，验证字典的键集合是否与预期一致
    assert set(d) == {x, y, z}
    
    # 断言字典 d 的字符串表示形式与预期的 '{x: 1, y: 2, z: 3}' 相等
    assert str(d) == '{x: 1, y: 2, z: 3}'
    
    # 断言字典 d 的 __repr__() 方法返回的字符串与预期的 '{x: 1, y: 2, z: 3}' 相等
    assert d.__repr__() == '{x: 1, y: 2, z: 3}'

    # 测试通过现有字典创建新的 Dict 对象，并断言它与原始字典相等
    d = Dict({x: 1, y: 2, z: 3})
    assert d == Dict(d)

    # 测试支持从 defaultdict 创建 Dict 对象，并断言默认值为 int 类型的键的值为 0
    d = defaultdict(int)
    assert d[x] == 0
    assert d[y] == 0
    assert d[z] == 0
    
    # 断言可以通过 Dict 函数将 defaultdict 转换为普通的 Dict 对象
    assert Dict(d)
    
    # 将 defaultdict 转换为 Dict 对象 d，并断言其长度为 3
    d = Dict(d)
    assert len(d) == 3
    
    # 断言 Dict 对象 d 的键集合与预期的 {x, y, z} 相等
    assert set(d.keys()) == {x, y, z}
    
    # 断言 Dict 对象 d 的值集合与预期的 {S.Zero, S.Zero, S.Zero} 相等，即默认值集合
    assert set(d.values()) == {S.Zero, S.Zero, S.Zero}
def test_issue_5788():
    args = [(1, 2), (2, 1)]
    # 对于每个类 o 在 [Dict, Tuple, FiniteSet] 中进行迭代
    for o in [Dict, Tuple, FiniteSet]:
        # __eq__ 和参数处理
        # 如果 o 不是 Tuple 类型，则进行下面的断言
        if o != Tuple:
            # 断言 o 类型对象使用 args 和其反转参数构造的对象相等
            assert o(*args) == o(*reversed(args))
        # 构造包含 o 类型对象和其反转对象的列表 pair
        pair = [o(*args), o(*reversed(args))]
        # 断言对 pair 列表进行排序后的结果相等
        assert sorted(pair) == sorted(pair)
        # 断言使用 args 构造的 o 类型对象能够成功转换为集合
        assert set(o(*args))  # doesn't fail
```