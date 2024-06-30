# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_type_E.py`

```
# 导入 CartanType 类，该类用于处理李代数的 Cartan 类型
# 以及 Matrix 类，用于创建和操作矩阵
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

# 定义测试函数 test_type_E
def test_type_E():
    # 创建 CartanType 对象，指定类型为 "E6"
    c = CartanType("E6")
    # 创建一个 6x6 的矩阵 m，填充指定的数据
    m = Matrix(6, 6, [2, 0, -1, 0, 0, 0,
                      0, 2, 0, -1, 0, 0,
                      -1, 0, 2, -1, 0, 0,
                      0, -1, -1, 2, -1, 0,
                      0, 0, -1, 2, -1, 0,
                      0, 0, 0, -1, 2, 0])
    # 断言 CartanType 对象的 cartan_matrix 方法返回的矩阵与 m 相等
    assert c.cartan_matrix() == m
    # 断言 CartanType 对象的 dimension 方法返回值为 8
    assert c.dimension() == 8
    # 断言 CartanType 对象的 simple_root 方法返回索引为 6 的简单根
    assert c.simple_root(6) == [0, 0, 0, -1, 1, 0, 0, 0]
    # 断言 CartanType 对象的 roots 方法返回 72
    assert c.roots() == 72
    # 断言 CartanType 对象的 basis 方法返回 78
    assert c.basis() == 78
    # 构建一个表示动金图的字符串 diag
    diag = " "*8 + "2\n" + " "*8 + "0\n" + " "*8 + "|\n" + " "*8 + "|\n"
    diag += "---".join("0" for i in range(1, 6)) + "\n"
    diag += "1   " + "   ".join(str(i) for i in range(3, 7))
    # 断言 CartanType 对象的 dynkin_diagram 方法返回的字符串与 diag 相等
    assert c.dynkin_diagram() == diag
    # 获取 CartanType 对象的 positive_roots 方法返回的正根列表
    posroots = c.positive_roots()
    # 断言正根列表中索引为 8 的元素与预期的列表相等
    assert posroots[8] == [1, 0, 0, 0, 1, 0, 0, 0]
```