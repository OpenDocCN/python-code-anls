# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_type_B.py`

```
# 导入 CartanType 类，该类位于 sympy.liealgebras.cartan_type 模块中
from sympy.liealgebras.cartan_type import CartanType
# 导入 Matrix 类，该类位于 sympy.matrices 模块中
from sympy.matrices import Matrix

# 定义测试函数 test_type_B
def test_type_B():
    # 创建 CartanType 对象，参数为 "B3"
    c = CartanType("B3")
    # 创建一个 3x3 的矩阵对象 m，填充具体数值
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -2, 0, -1, 2])
    # 断言 m 等于 CartanType 对象 c 的 Cartan 矩阵
    assert m == c.cartan_matrix()
    # 断言 CartanType 对象 c 的维度为 3
    assert c.dimension() == 3
    # 断言 CartanType 对象 c 的根的数量为 18
    assert c.roots() == 18
    # 断言 CartanType 对象 c 的第三个简单根为 [0, 0, 1]
    assert c.simple_root(3) == [0, 0, 1]
    # 断言 CartanType 对象 c 的基数为 3
    assert c.basis() == 3
    # 断言 CartanType 对象 c 的李代数名称为 "so(6)"
    assert c.lie_algebra() == "so(6)"
    # 设置一个字符串 diag 作为预期的 Dynkin 图表达式
    diag = "0---0=>=0\n1   2   3"
    # 断言 CartanType 对象 c 的 Dynkin 图表达式与预期 diag 相等
    assert c.dynkin_diagram() == diag
    # 断言 CartanType 对象 c 的正根集合与预期字典相等
    assert c.positive_roots() ==  {1: [1, -1, 0], 2: [1, 1, 0], 3: [1, 0, -1],
            4: [1, 0, 1], 5: [0, 1, -1], 6: [0, 1, 1], 7: [1, 0, 0],
            8: [0, 1, 0], 9: [0, 0, 1]}
```