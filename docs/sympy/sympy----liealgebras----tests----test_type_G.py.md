# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_type_G.py`

```
# 导入 sympy 库中的 CartanType 类和 Matrix 类
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

# 定义测试函数 test_type_G
def test_type_G():
    # 创建 CartanType 对象，参数为 "G2"
    c = CartanType("G2")
    # 创建一个 2x2 的矩阵对象 m
    m = Matrix(2, 2, [2, -1, -3, 2])
    # 断言 CartanType 对象的 cartan_matrix 方法返回的矩阵与 m 相等
    assert c.cartan_matrix() == m
    # 断言 CartanType 对象的 simple_root 方法返回索引为 2 的简单根
    assert c.simple_root(2) == [1, -2, 1]
    # 断言 CartanType 对象的 basis 方法返回值为 14
    assert c.basis() == 14
    # 断言 CartanType 对象的 roots 方法返回值为 12
    assert c.roots() == 12
    # 断言 CartanType 对象的 dimension 方法返回值为 3
    assert c.dimension() == 3
    # 定义一个字符串 diag，用于存储预期的 Dynkin 图表达式
    diag = "0≡<≡0\n1   2"
    # 断言 CartanType 对象的 dynkin_diagram 方法返回的图表达式与 diag 相等
    assert diag == c.dynkin_diagram()
    # 断言 CartanType 对象的 positive_roots 方法返回的字典
    assert c.positive_roots() == {1: [0, 1, -1], 2: [1, -2, 1], 3: [1, -1, 0],
                                  4: [1, 0, 1], 5: [1, 1, -2], 6: [2, -1, -1]}
```