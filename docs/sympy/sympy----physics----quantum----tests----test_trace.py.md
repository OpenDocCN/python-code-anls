# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_trace.py`

```
# 导入所需模块中的类和函数
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.trace import Tr
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 定义测试函数 test_trace_new
def test_trace_new():
    # 定义符号变量
    a, b, c, d, Y = symbols('a b c d Y')
    A, B, C, D = symbols('A B C D', commutative=False)

    # 检查对于 a + b 的迹是 a + b
    assert Tr(a + b) == a + b
    # 检查对于 A + B 的迹是 Tr(A) + Tr(B)
    assert Tr(A + B) == Tr(A) + Tr(B)

    # 检查迹的参数不隐式置换
    assert Tr(C*D*A*B).args[0].args == (C, D, A, B)

    # 检查对于 a*b + c*d 的迹是 a*b + c*d
    assert Tr((a*b) + (c*d)) == (a*b) + (c*d)
    # 检查 Tr(scalar*A) = scalar*Tr(A)
    assert Tr(a*A) == a*Tr(A)
    assert Tr(a*A*B*b) == a*b*Tr(A*B)

    # 因为 A 是符号且非交换的
    assert isinstance(Tr(A), Tr)

    # 检查对于 pow(a, b) 的迹是 a**b
    assert Tr(pow(a, b)) == a**b
    assert isinstance(Tr(pow(A, a)), Tr)

    # 检查对于矩阵 M 的迹是 3
    M = Matrix([[1, 1], [2, 2]])
    assert Tr(M) == 3

    ## 测试不同形式的索引
    # 没有索引
    t = Tr(A)
    assert t.args[1] == Tuple()

    # 单个索引
    t = Tr(A, 0)
    assert t.args[1] == Tuple(0)

    # 索引在列表中
    t = Tr(A, [0])
    assert t.args[1] == Tuple(0)

    t = Tr(A, [0, 1, 2])
    assert t.args[1] == Tuple(0, 1, 2)

    # 索引是元组
    t = Tr(A, (0))
    assert t.args[1] == Tuple(0)

    t = Tr(A, (1, 2))
    assert t.args[1] == Tuple(1, 2)

    # 追踪索引测试
    t = Tr((A + B), [2])
    assert t.args[0].args[1] == Tuple(2) and t.args[1].args[1] == Tuple(2)

    t = Tr(a*A, [2, 3])
    assert t.args[1].args[1] == Tuple(2, 3)

    # 类中定义了 trace 方法
    # 模拟 numpy 对象
    class Foo:
        def trace(self):
            return 1
    assert Tr(Foo()) == 1

    # 参数测试
    # 当未提供一个或两个参数时，检查值错误
    raises(ValueError, lambda: Tr())
    raises(ValueError, lambda: Tr(A, 1, 2))

# 定义测试函数 test_trace_doit
def test_trace_doit():
    # 定义符号变量
    a, b, c, d = symbols('a b c d')
    A, B, C, D = symbols('A B C D', commutative=False)

    # TODO: 在测试减少密度操作等时需要

# 定义测试函数 test_permute
def test_permute():
    # 定义符号变量
    A, B, C, D, E, F, G = symbols('A B C D E F G', commutative=False)
    t = Tr(A*B*C*D*E*F*G)

    # 检查各种排列方式下的参数顺序
    assert t.permute(0).args[0].args == (A, B, C, D, E, F, G)
    assert t.permute(2).args[0].args == (F, G, A, B, C, D, E)
    assert t.permute(4).args[0].args == (D, E, F, G, A, B, C)
    assert t.permute(6).args[0].args == (B, C, D, E, F, G, A)
    assert t.permute(8).args[0].args == t.permute(1).args[0].args

    assert t.permute(-1).args[0].args == (B, C, D, E, F, G, A)
    assert t.permute(-3).args[0].args == (D, E, F, G, A, B, C)
    assert t.permute(-5).args[0].args == (F, G, A, B, C, D, E)
    assert t.permute(-8).args[0].args == t.permute(-1).args[0].args

    t = Tr((A + B)*(B*B)*C*D)
    assert t.permute(2).args[0].args == (C, D, (A + B), (B**2))

    t1 = Tr(A*B)
    t2 = t1.permute(1)
    assert id(t1) != id(t2) and t1 == t2

# 定义测试函数 test_deprecated_core_trace
def test_deprecated_core_trace():
    # 留空，用于测试已弃用的追踪方法
    # 使用 warns_deprecated_sympy() 上下文管理器来捕获 Sympy 废弃警告
    with warns_deprecated_sympy():
        # 从 sympy.core.trace 模块导入 Tr，并忽略 F401 类型的导入警告
        from sympy.core.trace import Tr # noqa:F401
```