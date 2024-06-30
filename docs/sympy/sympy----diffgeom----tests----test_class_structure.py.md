# `D:\src\scipysrc\sympy\sympy\diffgeom\tests\test_class_structure.py`

```
from sympy.diffgeom import Manifold, Patch, CoordSystem, Point  # 导入符号计算相关的模块
from sympy.core.function import Function  # 导入函数相关的模块
from sympy.core.symbol import symbols  # 导入符号相关的模块
from sympy.testing.pytest import warns_deprecated_sympy  # 导入用于测试的模块

m = Manifold('m', 2)  # 创建一个二维流形对象 'm'
p = Patch('p', m)  # 在流形 'm' 上创建一个块 'p'
a, b = symbols('a b')  # 创建符号 'a' 和 'b'
cs = CoordSystem('cs', p, [a, b])  # 在块 'p' 上创建一个坐标系 'cs'，使用符号 'a' 和 'b' 作为坐标
x, y = symbols('x y')  # 创建符号 'x' 和 'y'
f = Function('f')  # 创建一个函数 'f'
s1, s2 = cs.coord_functions()  # 获取坐标系 'cs' 的坐标函数
v1, v2 = cs.base_vectors()  # 获取坐标系 'cs' 的基向量
f1, f2 = cs.base_oneforms()  # 获取坐标系 'cs' 的基一形式

def test_point():
    point = Point(cs, [x, y])  # 创建一个坐标系 'cs' 上的点，坐标为 [x, y]
    assert point != Point(cs, [2, y])  # 断言点不等于另一个点，坐标为 [2, y]
    # TODO 断言点在替换 x 为 2 后的坐标等于 [2, y]
    # TODO 断言点的自由符号集合等于 {x, y}

def test_subs():
    assert s1.subs(s1, s2) == s2  # 断言替换 s1 为 s2 后的结果等于 s2
    assert v1.subs(v1, v2) == v2  # 断言替换 v1 为 v2 后的结果等于 v2
    assert f1.subs(f1, f2) == f2  # 断言替换 f1 为 f2 后的结果等于 f2
    assert (x*f(s1) + y).subs(s1, s2) == x*f(s2) + y  # 断言替换 s1 为 s2 后的表达式结果正确
    assert (f(s1)*v1).subs(v1, v2) == f(s1)*v2  # 断言替换 v1 为 v2 后的表达式结果正确
    assert (y*f(s1)*f1).subs(f1, f2) == y*f(s1)*f2  # 断言替换 f1 为 f2 后的表达式结果正确

def test_deprecated():
    with warns_deprecated_sympy():  # 使用 deprecated 警告上下文
        cs_wname = CoordSystem('cs', p, ['a', 'b'])  # 创建坐标系 'cs'，使用符号 'a' 和 'b' 作为坐标
        assert cs_wname == cs_wname.func(*cs_wname.args)  # 断言坐标系与其函数形式相同
```