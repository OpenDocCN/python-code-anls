# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_output.py`

```
# 导入必要的模块和函数
from sympy.core.singleton import S  # 导入 sympy 的单例 S
from sympy.physics.vector import Vector, ReferenceFrame, Dyadic  # 导入矢量、参考系和二阶张量
from sympy.testing.pytest import raises  # 导入用于测试的 raises 函数

A = ReferenceFrame('A')  # 创建一个名为 A 的参考系对象

# 定义一个测试函数，用于测试向量和张量的运算结果类型
def test_output_type():
    A = ReferenceFrame('A')  # 创建一个名为 A 的参考系对象
    v = A.x + A.y  # 定义一个向量 v，由参考系 A 的 x 和 y 方向向量相加组成
    d = v | v  # 定义一个二阶张量 d，使用向量 v 进行外积操作得到

    zerov = Vector(0)  # 创建一个零向量对象
    zerod = Dyadic(0)  # 创建一个零二阶张量对象

    # 断言各种运算的结果类型是否符合预期

    # 点乘操作
    assert isinstance(d & d, Dyadic)
    assert isinstance(d & zerod, Dyadic)
    assert isinstance(zerod & d, Dyadic)
    assert isinstance(d & v, Vector)
    assert isinstance(v & d, Vector)
    assert isinstance(d & zerov, Vector)
    assert isinstance(zerov & d, Vector)
    raises(TypeError, lambda: d & S.Zero)
    raises(TypeError, lambda: S.Zero & d)
    raises(TypeError, lambda: d & 0)
    raises(TypeError, lambda: 0 & d)
    assert not isinstance(v & v, (Vector, Dyadic))
    assert not isinstance(v & zerov, (Vector, Dyadic))
    assert not isinstance(zerov & v, (Vector, Dyadic))
    raises(TypeError, lambda: v & S.Zero)
    raises(TypeError, lambda: S.Zero & v)
    raises(TypeError, lambda: v & 0)
    raises(TypeError, lambda: 0 & v)

    # 叉乘操作
    raises(TypeError, lambda: d ^ d)
    raises(TypeError, lambda: d ^ zerod)
    raises(TypeError, lambda: zerod ^ d)
    assert isinstance(d ^ v, Dyadic)
    assert isinstance(v ^ d, Dyadic)
    assert isinstance(d ^ zerov, Dyadic)
    assert isinstance(zerov ^ d, Dyadic)
    raises(TypeError, lambda: d ^ S.Zero)
    raises(TypeError, lambda: S.Zero ^ d)
    raises(TypeError, lambda: d ^ 0)
    raises(TypeError, lambda: 0 ^ d)
    assert isinstance(v ^ v, Vector)
    assert isinstance(v ^ zerov, Vector)
    assert isinstance(zerov ^ v, Vector)
    raises(TypeError, lambda: v ^ S.Zero)
    raises(TypeError, lambda: S.Zero ^ v)
    raises(TypeError, lambda: v ^ 0)
    raises(TypeError, lambda: 0 ^ v)

    # 外积操作
    raises(TypeError, lambda: d | d)
    raises(TypeError, lambda: d | zerod)
    raises(TypeError, lambda: zerod | d)
    raises(TypeError, lambda: d | v)
    raises(TypeError, lambda: v | d)
    raises(TypeError, lambda: d | zerov)
    raises(TypeError, lambda: zerov | d)
    raises(TypeError, lambda: d | S.Zero)
    raises(TypeError, lambda: S.Zero | d)
    raises(TypeError, lambda: d | 0)
    raises(TypeError, lambda: 0 | d)
    assert isinstance(v | v, Dyadic)
    assert isinstance(v | zerov, Dyadic)
    assert isinstance(zerov | v, Dyadic)
    raises(TypeError, lambda: v | S.Zero)
    raises(TypeError, lambda: S.Zero | v)
    raises(TypeError, lambda: v | 0)
    raises(TypeError, lambda: 0 | v)
```