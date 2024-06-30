# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_type_D.py`

```
# 导入 sympy 库中的 CartanType 类
from sympy.liealgebras.cartan_type import CartanType
# 导入 sympy 库中的 Matrix 类
from sympy.matrices import Matrix

# 定义函数 test_type_D，用于测试 CartanType 类中与 "D4" 相关的方法和属性
def test_type_D():
    # 创建 CartanType 对象，表示 Cartan 类型为 "D4"
    c = CartanType("D4")
    # 创建一个 4x4 的矩阵对象 m，表示 Cartan 矩阵
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -1, -1, 0, -1, 2, 0, 0, -1, 0, 2])
    # 断言 CartanType 对象的 cartan_matrix 方法返回的结果与 m 相等
    assert c.cartan_matrix() == m
    # 断言 CartanType 对象的 basis 方法返回值为 6
    assert c.basis() == 6
    # 断言 CartanType 对象的 lie_algebra 方法返回值为 "so(8)"
    assert c.lie_algebra() == "so(8)"
    # 断言 CartanType 对象的 roots 方法返回值为 24
    assert c.roots() == 24
    # 断言 CartanType 对象的 simple_root 方法返回索引为 3 的简单根
    assert c.simple_root(3) == [0, 0, 1, -1]
    # 定义一个字符串 diag，表示 CartanType 对象的 Dynkin 图
    diag = "    3\n    0\n    |\n    |\n0---0---0\n1   2   4"
    # 断言 diag 等于 CartanType 对象的 dynkin_diagram 方法返回的字符串
    assert diag == c.dynkin_diagram()
    # 断言 CartanType 对象的 positive_roots 方法返回的字典与指定的字典相等
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0],
            3: [1, 0, -1, 0], 4: [1, 0, 1, 0], 5: [1, 0, 0, -1], 6: [1, 0, 0, 1],
            7: [0, 1, -1, 0], 8: [0, 1, 1, 0], 9: [0, 1, 0, -1], 10: [0, 1, 0, 1],
            11: [0, 0, 1, -1], 12: [0, 0, 1, 1]}
```