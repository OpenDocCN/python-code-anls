# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_type_F.py`

```
# 从 sympy.liealgebras.cartan_type 模块中导入 CartanType 类
# 从 sympy.matrices 模块中导入 Matrix 类
# 从 sympy.core.backend 模块中导入 S 对象
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix
from sympy.core.backend import S

# 定义一个名为 test_type_F 的函数，用于测试 CartanType 类的功能
def test_type_F():
    # 创建一个 CartanType 类型的对象，表示 Lie 代数的类型为 "F4"
    c = CartanType("F4")
    
    # 创建一个 4x4 的 Matrix 对象，表示一个特定的矩阵
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -2, 0, 0, -1, 2, -1, 0, 0, -1, 2])
    
    # 断言：返回 Cartan 矩阵是否与预期的 m 相等
    assert c.cartan_matrix() == m
    
    # 断言：返回 Lie 代数的维度是否为 4
    assert c.dimension() == 4
    
    # 断言：返回第一个简单根的系数列表是否正确
    assert c.simple_root(1) == [1, -1, 0, 0]
    
    # 断言：返回第二个简单根的系数列表是否正确
    assert c.simple_root(2) == [0, 1, -1, 0]
    
    # 断言：返回第三个简单根的系数列表是否正确
    assert c.simple_root(3) == [0, 0, 0, 1]
    
    # 断言：返回第四个简单根的系数列表是否正确
    assert c.simple_root(4) == [-S.Half, -S.Half, -S.Half, -S.Half]
    
    # 断言：返回根的数量是否为 48
    assert c.roots() == 48
    
    # 断言：返回基的数量是否为 52
    assert c.basis() == 52
    
    # 创建一个字符串 diag，表示动金图的一部分
    diag = "0---0=>=0---0\n" + "   ".join(str(i) for i in range(1, 5))
    
    # 断言：返回动金图是否与预期的 diag 字符串相等
    assert c.dynkin_diagram() == diag
    
    # 断言：返回正根的字典是否正确
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0], 3: [1, 0, -1, 0],
            4: [1, 0, 1, 0], 5: [1, 0, 0, -1], 6: [1, 0, 0, 1], 7: [0, 1, -1, 0],
            8: [0, 1, 1, 0], 9: [0, 1, 0, -1], 10: [0, 1, 0, 1], 11: [0, 0, 1, -1],
            12: [0, 0, 1, 1], 13: [1, 0, 0, 0], 14: [0, 1, 0, 0], 15: [0, 0, 1, 0],
            16: [0, 0, 0, 1], 17: [S.Half, S.Half, S.Half, S.Half], 18: [S.Half, -S.Half, S.Half, S.Half],
            19: [S.Half, S.Half, -S.Half, S.Half], 20: [S.Half, S.Half, S.Half, -S.Half], 21: [S.Half, S.Half, -S.Half, -S.Half],
            22: [S.Half, -S.Half, S.Half, -S.Half], 23: [S.Half, -S.Half, -S.Half, S.Half], 24: [S.Half, -S.Half, -S.Half, -S.Half]}
```