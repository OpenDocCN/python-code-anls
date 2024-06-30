# `D:\src\scipysrc\sympy\sympy\sandbox\tests\test_indexed_integrals.py`

```
# 导入需要的模块和函数，从 sympy.sandbox.indexed_integrals 模块导入 IndexedIntegral 类
# 从 sympy.core.symbol 模块导入 symbols 函数，从 sympy.functions.elementary.trigonometric 模块导入 cos 和 sin 函数
# 从 sympy.tensor.indexed 模块导入 Idx 和 IndexedBase 类
from sympy.sandbox.indexed_integrals import IndexedIntegral
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.indexed import (Idx, IndexedBase)


# 定义测试函数 test_indexed_integrals
def test_indexed_integrals():
    # 创建 IndexedBase 对象 A，用于表示索引的基础
    A = IndexedBase('A')
    # 创建符号变量 i 和 j，并指定它们为整数
    i, j = symbols('i j', integer=True)
    # 创建符号变量 a1 和 a2，这些变量是索引符号 Idx 的实例
    a1, a2 = symbols('a1:3', cls=Idx)
    # 断言 a1 是 Idx 类的实例
    assert isinstance(a1, Idx)

    # 下面是一系列断言语句，用于测试 IndexedIntegral 类的不同用法和结果
    # 断言对于简单的积分，结果应为 A[i]
    assert IndexedIntegral(1, A[i]).doit() == A[i]
    # 断言对于 A[i]^2 的积分，结果应为 A[i]^2 / 2
    assert IndexedIntegral(A[i], A[i]).doit() == A[i] ** 2 / 2
    # 断言对于 A[i]*A[j] 的积分，结果应为 A[i]*A[j]
    assert IndexedIntegral(A[j], A[i]).doit() == A[i] * A[j]
    # 断言对于 A[i]*A[j]*A[i] 的积分，结果应为 A[i]^2 * A[j] / 2
    assert IndexedIntegral(A[i] * A[j], A[i]).doit() == A[i] ** 2 * A[j] / 2
    # 断言对于 sin(A[i]) 的积分，结果应为 -cos(A[i])
    assert IndexedIntegral(sin(A[i]), A[i]).doit() == -cos(A[i])
    # 断言对于 sin(A[j])*A[i] 的积分，结果应为 sin(A[j])*A[i]
    assert IndexedIntegral(sin(A[j]), A[i]).doit() == sin(A[j]) * A[i]

    # 以下是针对带有 Idx 符号的情况的断言语句
    # 断言对于简单的积分，结果应为 A[a1]
    assert IndexedIntegral(1, A[a1]).doit() == A[a1]
    # 断言对于 A[a1]^2 的积分，结果应为 A[a1]^2 / 2
    assert IndexedIntegral(A[a1], A[a1]).doit() == A[a1] ** 2 / 2
    # 断言对于 A[a1]*A[a2] 的积分，结果应为 A[a1]*A[a2]
    assert IndexedIntegral(A[a2], A[a1]).doit() == A[a1] * A[a2]
    # 断言对于 A[a1]*A[a2]*A[a1] 的积分，结果应为 A[a1]^2 * A[a2] / 2
    assert IndexedIntegral(A[a1] * A[a2], A[a1]).doit() == A[a1] ** 2 * A[a2] / 2
    # 断言对于 sin(A[a1]) 的积分，结果应为 -cos(A[a1])
    assert IndexedIntegral(sin(A[a1]), A[a1]).doit() == -cos(A[a1])
    # 断言对于 sin(A[a2])*A[a1] 的积分，结果应为 sin(A[a2])*A[a1]
    assert IndexedIntegral(sin(A[a2]), A[a1]).doit() == sin(A[a2]) * A[a1]
```