# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_type_C.py`

```
# 导入 CartanType 类从 sympy.liealgebras.cartan_type 模块和 Matrix 类从 sympy.matrices 模块
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

# 定义测试函数 test_type_C
def test_type_C():
    # 创建 CartanType 对象 c，表示 Cartan 类型 "C4"
    c = CartanType("C4")
    # 创建一个 4x4 的矩阵对象 m，表示特定的 Cartan 矩阵
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -2, 2])
    # 断言 c 的 Cartan 矩阵与 m 相等
    assert c.cartan_matrix() == m
    # 断言 c 的维度为 4
    assert c.dimension() == 4
    # 断言 c 的第 4 个简单根为 [0, 0, 0, 2]
    assert c.simple_root(4) == [0, 0, 0, 2]
    # 断言 c 的根的数量为 32
    assert c.roots() == 32
    # 断言 c 的基底数量为 36
    assert c.basis() == 36
    # 断言 c 的李代数为 "sp(8)"
    assert c.lie_algebra() == "sp(8)"
    # 创建 CartanType 对象 t，表示 Cartan 类型 "C3"
    t = CartanType(['C', 3])
    # 断言 t 的维度为 3
    assert t.dimension() == 3
    # 定义字符串 diag，表示一个特定的 Dynkin 图形式
    diag = "0---0---0=<=0\n1   2   3   4"
    # 断言 c 的 Dynkin 图与 diag 相等
    assert c.dynkin_diagram() == diag
    # 断言 c 的正根集合
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0],
            3: [1, 0, -1, 0], 4: [1, 0, 1, 0], 5: [1, 0, 0, -1],
            6: [1, 0, 0, 1], 7: [0, 1, -1, 0], 8: [0, 1, 1, 0],
            9: [0, 1, 0, -1], 10: [0, 1, 0, 1], 11: [0, 0, 1, -1],
            12: [0, 0, 1, 1], 13: [2, 0, 0, 0], 14: [0, 2, 0, 0], 15: [0, 0, 2, 0],
            16: [0, 0, 0, 2]}
```