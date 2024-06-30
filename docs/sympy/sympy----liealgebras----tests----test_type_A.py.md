# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_type_A.py`

```
# 从 sympy.liealgebras.cartan_type 模块中导入 CartanType 类
# 从 sympy.matrices 模块中导入 Matrix 类
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

# 定义测试函数 test_type_A
def test_type_A():
    # 创建 CartanType 对象，表示 A3 类型的 Cartan 矩阵
    c = CartanType("A3")
    # 创建一个 3x3 的矩阵 m，填充特定的数据
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -1, 0, -1, 2])
    # 断言 m 与 CartanType 对象的 Cartan 矩阵相等
    assert m == c.cartan_matrix()
    # 断言 CartanType 对象的基数为 8
    assert c.basis() == 8
    # 断言 CartanType 对象的根数为 12
    assert c.roots() == 12
    # 断言 CartanType 对象的维数为 4
    assert c.dimension() == 4
    # 断言 CartanType 对象的第一个简单根为 [1, -1, 0, 0]
    assert c.simple_root(1) == [1, -1, 0, 0]
    # 断言 CartanType 对象的最高根为 [1, 0, 0, -1]
    assert c.highest_root() == [1, 0, 0, -1]
    # 断言 CartanType 对象的李代数表示为 "su(4)"
    assert c.lie_algebra() == "su(4)"
    # 定义一个字符串 diag，表示 Dynkin 图表达式
    diag = "0---0---0\n1   2   3"
    # 断言 CartanType 对象的 Dynkin 图表达式与 diag 相等
    assert c.dynkin_diagram() == diag
    # 断言 CartanType 对象的正根集合符合预期的字典形式
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 0, -1, 0],
            3: [1, 0, 0, -1], 4: [0, 1, -1, 0], 5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
```