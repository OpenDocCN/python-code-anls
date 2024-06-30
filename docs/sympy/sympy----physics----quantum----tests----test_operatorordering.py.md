# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_operatorordering.py`

```
# 从 sympy.physics.quantum 模块中导入 Dagger 类，用于创建算符的共轭转置
from sympy.physics.quantum import Dagger
# 从 sympy.physics.quantum.boson 模块中导入 BosonOp 类，用于创建玻色子算符
from sympy.physics.quantum.boson import BosonOp
# 从 sympy.physics.quantum.fermion 模块中导入 FermionOp 类，用于创建费米子算符
from sympy.physics.quantum.fermion import FermionOp
# 从 sympy.physics.quantum.operatorordering 模块中导入 normal_order 和 normal_ordered_form 函数，
# 用于算符的正常排序和正常排序形式
from sympy.physics.quantum.operatorordering import (normal_order,
                                                 normal_ordered_form)


# 定义测试函数 test_normal_order，用于测试正常排序函数 normal_order 的功能
def test_normal_order():
    # 创建一个名为 a 的玻色子算符对象
    a = BosonOp('a')
    # 创建一个名为 c 的费米子算符对象
    c = FermionOp('c')

    # 断言表达式，验证对玻色子算符的正常排序是否得到期望的结果
    assert normal_order(a * Dagger(a)) == Dagger(a) * a
    assert normal_order(Dagger(a) * a) == Dagger(a) * a
    assert normal_order(a * Dagger(a) ** 2) == Dagger(a) ** 2 * a

    # 断言表达式，验证对费米子算符的正常排序是否得到期望的结果
    assert normal_order(c * Dagger(c)) == - Dagger(c) * c
    assert normal_order(Dagger(c) * c) == Dagger(c) * c
    assert normal_order(c * Dagger(c) ** 2) == Dagger(c) ** 2 * c


# 定义测试函数 test_normal_ordered_form，用于测试正常排序形式函数 normal_ordered_form 的功能
def test_normal_ordered_form():
    # 创建名为 a 和 b 的两个玻色子算符对象
    a = BosonOp('a')
    b = BosonOp('b')

    # 创建名为 c 和 d 的两个费米子算符对象
    c = FermionOp('c')
    d = FermionOp('d')

    # 断言表达式，验证对玻色子算符的正常排序形式是否得到期望的结果
    assert normal_ordered_form(Dagger(a) * a) == Dagger(a) * a
    assert normal_ordered_form(a * Dagger(a)) == 1 + Dagger(a) * a
    assert normal_ordered_form(a ** 2 * Dagger(a)) == \
        2 * a + Dagger(a) * a ** 2
    assert normal_ordered_form(a ** 3 * Dagger(a)) == \
        3 * a ** 2 + Dagger(a) * a ** 3

    # 断言表达式，验证对费米子算符的正常排序形式是否得到期望的结果
    assert normal_ordered_form(Dagger(c) * c) == Dagger(c) * c
    assert normal_ordered_form(c * Dagger(c)) == 1 - Dagger(c) * c
    assert normal_ordered_form(c ** 2 * Dagger(c)) == Dagger(c) * c ** 2
    assert normal_ordered_form(c ** 3 * Dagger(c)) == \
        c ** 2 - Dagger(c) * c ** 3

    # 断言表达式，验证带有强制排序的正常排序形式函数是否得到期望的结果
    assert normal_ordered_form(a * Dagger(b), True) == Dagger(b) * a
    assert normal_ordered_form(Dagger(a) * b, True) == Dagger(a) * b
    assert normal_ordered_form(b * a, True) == a * b
    assert normal_ordered_form(Dagger(b) * Dagger(a), True) == Dagger(a) * Dagger(b)

    assert normal_ordered_form(c * Dagger(d), True) == -Dagger(d) * c
    assert normal_ordered_form(Dagger(c) * d, True) == Dagger(c) * d
    assert normal_ordered_form(d * c, True) == -c * d
    assert normal_ordered_form(Dagger(d) * Dagger(c), True) == -Dagger(c) * Dagger(d)
```