# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_root_system.py`

```
# 导入所需的类和函数
from sympy.liealgebras.root_system import RootSystem
from sympy.liealgebras.type_a import TypeA
from sympy.matrices import Matrix

# 定义测试函数 test_root_system
def test_root_system():
    # 创建根系统对象 c，类型为 "A3"
    c = RootSystem("A3")
    # 断言根系统的 Cartan 类型为 TypeA(3)
    assert c.cartan_type == TypeA(3)
    # 断言根系统的简单根系为 {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}
    assert c.simple_roots() == {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}
    # 断言根空间的表示形式为 "alpha[1] + alpha[2] + alpha[3]"
    assert c.root_space() == "alpha[1] + alpha[2] + alpha[3]"
    # 断言根系统的 Cartan 矩阵为 Matrix([[ 2, -1,  0], [-1,  2, -1], [ 0, -1,  2]])
    assert c.cartan_matrix() == Matrix([[ 2, -1,  0], [-1,  2, -1], [ 0, -1,  2]])
    # 断言根系统的 Dynkin 图形式为 "0---0---0\n1   2   3"
    assert c.dynkin_diagram() == "0---0---0\n1   2   3"
    # 断言在简单根系中增加根 [1, 2] 的结果为 [1, 0, -1, 0]
    assert c.add_simple_roots(1, 2) == [1, 0, -1, 0]
    # 断言所有根系的集合
    assert c.all_roots() == {
        1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1],
        4: [0, 1, -1, 0], 5: [0, 1, 0, -1], 6: [0, 0, 1, -1],
        7: [-1, 1, 0, 0], 8: [-1, 0, 1, 0], 9: [-1, 0, 0, 1],
        10: [0, -1, 1, 0], 11: [0, -1, 0, 1], 12: [0, 0, -1, 1]
    }
    # 断言在根系中增加给定根 [1, 0, -1, 0] 和 [0, 0, 1, -1] 的结果为 [1, 0, 0, -1]
    assert c.add_as_roots([1, 0, -1, 0], [0, 0, 1, -1]) == [1, 0, 0, -1]
```