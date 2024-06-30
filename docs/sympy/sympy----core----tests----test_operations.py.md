# `D:\src\scipysrc\sympy\sympy\core\tests\test_operations.py`

```
# 导入所需的 Sympy 模块和类
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul

# 创建一个最简单的 LatticeOp 类 join
class join(LatticeOp):
    zero = Integer(0)   # 定义零元素
    identity = Integer(1)   # 定义单位元素

# 测试简单的 lattice 操作
def test_lattice_simple():
    assert join(join(2, 3), 4) == join(2, join(3, 4))   # 结合性测试
    assert join(2, 3) == join(3, 2)   # 交换性测试
    assert join(0, 2) == 0   # 零元素测试
    assert join(1, 2) == 2   # 单位元素测试
    assert join(2, 2) == 2   # 自反性测试

    assert join(join(2, 3), 4) == join(2, 3, 4)   # 额外结合性测试
    assert join() == 1   # 空操作测试
    assert join(4) == 4   # 单元素测试
    assert join(1, 4, 2, 3, 1, 3, 2) == join(2, 3, 4)   # 多参数测试

# 测试 lattice 的短路行为
def test_lattice_shortcircuit():
    raises(SympifyError, lambda: join(object))   # 抛出异常的测试
    assert join(0, object) == 0   # 零元素的短路测试

# 测试 lattice 的字符串表示
def test_lattice_print():
    assert str(join(5, 4, 3, 2)) == 'join(2, 3, 4, 5)'   # 字符串表示测试

# 测试构造 lattice 的参数
def test_lattice_make_args():
    assert join.make_args(join(2, 3, 4)) == {S(2), S(3), S(4)}   # 构造参数测试
    assert join.make_args(0) == {0}   # 构造零元素参数测试
    assert list(join.make_args(0))[0] is S.Zero   # 零元素构造测试
    assert Add.make_args(0)[0] is S.Zero   # Add 操作的构造零元素测试

# 测试特定问题的解决方案
def test_issue_14025():
    a, b, c, d = symbols('a,b,c,d', commutative=False)
    assert Mul(a, b, c).has(c*b) == False   # Mul 操作的特定问题测试
    assert Mul(a, b, c).has(b*c) == True
    assert Mul(a, b, c, d).has(b*c*d) == True

# 测试 AssocOp 的展开行为
def test_AssocOp_flatten():
    a, b, c, d = symbols('a,b,c,d')

    # 定义自定义的 AssocOp 类 MyAssoc
    class MyAssoc(AssocOp):
        identity = S.One

    assert MyAssoc(a, MyAssoc(b, c)).args == \
        MyAssoc(MyAssoc(a, b), c).args == \
        MyAssoc(MyAssoc(a, b, c)).args == \
        MyAssoc(a, b, c).args == \
            (a, b, c)   # 展开行为测试
    u = MyAssoc(b, c)
    v = MyAssoc(u, d, evaluate=False)
    assert v.args == (u, d)
    # 类似于 Add，任何未评估的外部调用都将展平内部参数
    assert MyAssoc(a, v).args == (a, b, c, d)

# 测试 add 的调度器
def test_add_dispatcher():

    class NewBase(Expr):
        @property
        def _add_handler(self):
            return NewAdd
    class NewAdd(NewBase, Add):
        pass
    add.register_handlerclass((Add, NewAdd), NewAdd)

    a, b = Symbol('a'), NewBase()

    # 当作为回退调用时，使用 Add
    assert add(1, 2) == Add(1, 2)
    assert add(a, a) == Add(a, a)

    # 通过注册的优先级进行选择
    assert add(a,b,a) == NewAdd(2*a, b)

# 测试 mul 的调度器
def test_mul_dispatcher():

    class NewBase(Expr):
        @property
        def _mul_handler(self):
            return NewMul
    class NewMul(NewBase, Mul):
        pass
    mul.register_handlerclass((Mul, NewMul), NewMul)

    a, b = Symbol('a'), NewBase()

    # 当作为回退调用时，使用 Mul
    assert mul(1, 2) == Mul(1, 2)
    assert mul(a, a) == Mul(a, a)

    # 通过注册的优先级进行选择
    assert mul(a,b,a) == NewMul(a**2, b)
```