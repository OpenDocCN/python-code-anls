# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_weyl_group.py`

```
# 导入从 sympy.liealgebras.weyl_group 模块中导入 WeylGroup 类
# 导入从 sympy.matrices 模块中导入 Matrix 类
from sympy.liealgebras.weyl_group import WeylGroup
from sympy.matrices import Matrix

# 定义测试函数 test_weyl_group
def test_weyl_group():
    # 创建 WeylGroup 类的实例 c，表示 A3 类型的 Weyl 群
    c = WeylGroup("A3")
    # 断言 c 对象的 matrix_form 方法对 'r1*r2' 返回的矩阵形式
    assert c.matrix_form('r1*r2') == Matrix([[0, 0, 1, 0], [1, 0, 0, 0],
        [0, 1, 0, 0], [0, 0, 0, 1]])
    # 断言 c 对象的 generators 方法返回的生成器列表
    assert c.generators() == ['r1', 'r2', 'r3']
    # 断言 c 对象的 group_order 方法返回的群的阶
    assert c.group_order() == 24.0
    # 断言 c 对象的 group_name 方法返回的群的名称
    assert c.group_name() == "S4: the symmetric group acting on 4 elements."
    # 断言 c 对象的 coxeter_diagram 方法返回的 Coxeter 图形表示
    assert c.coxeter_diagram() == "0---0---0\n1   2   3"
    # 断言 c 对象的 element_order 方法计算 'r1*r2*r3' 的元素阶
    assert c.element_order('r1*r2*r3') == 4
    # 断言 c 对象的 element_order 方法计算 'r1*r3*r2*r3' 的元素阶
    assert c.element_order('r1*r3*r2*r3') == 3

    # 创建 WeylGroup 类的实例 d，表示 B5 类型的 Weyl 群
    d = WeylGroup("B5")
    # 断言 d 对象的 group_order 方法返回的群的阶
    assert d.group_order() == 3840
    # 断言 d 对象的 element_order 方法计算 'r1*r2*r4*r5' 的元素阶
    assert d.element_order('r1*r2*r4*r5') == 12
    # 断言 d 对象的 matrix_form 方法对 'r2*r3' 返回的矩阵形式
    assert d.matrix_form('r2*r3') ==  Matrix([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    # 断言 d 对象的 element_order 方法计算 'r1*r2*r1*r3*r5' 的元素阶
    assert d.element_order('r1*r2*r1*r3*r5') == 6

    # 创建 WeylGroup 类的实例 e，表示 D5 类型的 Weyl 群
    e = WeylGroup("D5")
    # 断言 e 对象的 element_order 方法计算 'r2*r3*r5' 的元素阶
    assert e.element_order('r2*r3*r5') == 4
    # 断言 e 对象的 matrix_form 方法对 'r2*r3*r5' 返回的矩阵形式
    assert e.matrix_form('r2*r3*r5') == Matrix([[1, 0, 0, 0, 0], [0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0]])

    # 创建 WeylGroup 类的实例 f，表示 G2 类型的 Weyl 群
    f = WeylGroup("G2")
    # 断言 f 对象的 element_order 方法计算 'r1*r2*r1*r2' 的元素阶
    assert f.element_order('r1*r2*r1*r2') == 3
    # 断言 f 对象的 element_order 方法计算 'r2*r1*r1*r2' 的元素阶
    assert f.element_order('r2*r1*r1*r2') == 1
    # 断言 f 对象的 matrix_form 方法对 'r1*r2*r1*r2' 返回的矩阵形式
    assert f.matrix_form('r1*r2*r1*r2') == Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # 创建 WeylGroup 类的实例 g，表示 F4 类型的 Weyl 群
    g = WeylGroup("F4")
    # 断言 g 对象的 matrix_form 方法对 'r2*r3' 返回的矩阵形式
    assert g.matrix_form('r2*r3') == Matrix([[1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, -1], [0, 0, 1, 0]])
    # 断言 g 对象的 element_order 方法计算 'r2*r3' 的元素阶
    assert g.element_order('r2*r3') == 4

    # 创建 WeylGroup 类的实例 h，表示 E6 类型的 Weyl 群
    h = WeylGroup("E6")
    # 断言 h 对象的 group_order 方法返回的群的阶
    assert h.group_order() == 51840
```