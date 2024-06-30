# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_tensor_functions.py`

```
from sympy.core.relational import Ne  # 导入 sympy 中的不等式符号 Ne
from sympy.core.symbol import (Dummy, Symbol, symbols)  # 导入 sympy 中的符号相关模块
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)  # 导入复数相关函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.special.tensor_functions import (Eijk, KroneckerDelta, LeviCivita)  # 导入张量函数

from sympy.physics.secondquant import evaluate_deltas, F  # 导入量子物理相关模块

x, y = symbols('x y')  # 定义符号变量 x 和 y


def test_levicivita():
    # 测试 LeviCivita 符号函数的各种情况
    assert Eijk(1, 2, 3) == LeviCivita(1, 2, 3)
    assert LeviCivita(1, 2, 3) == 1
    assert LeviCivita(int(1), int(2), int(3)) == 1
    assert LeviCivita(1, 3, 2) == -1
    assert LeviCivita(1, 2, 2) == 0
    i, j, k = symbols('i j k')  # 定义符号变量 i, j, k
    assert LeviCivita(i, j, k) == LeviCivita(i, j, k, evaluate=False)
    assert LeviCivita(i, j, i) == 0
    assert LeviCivita(1, i, i) == 0
    assert LeviCivita(i, j, k).doit() == (j - i)*(k - i)*(k - j)/2
    assert LeviCivita(1, 2, 3, 1) == 0
    assert LeviCivita(4, 5, 1, 2, 3) == 1
    assert LeviCivita(4, 5, 2, 1, 3) == -1

    assert LeviCivita(i, j, k).is_integer is True  # 检查 LeviCivita 符号函数是否是整数

    assert adjoint(LeviCivita(i, j, k)) == LeviCivita(i, j, k)  # 测试 LeviCivita 符号函数的共轭转置
    assert conjugate(LeviCivita(i, j, k)) == LeviCivita(i, j, k)  # 测试 LeviCivita 符号函数的共轭
    assert transpose(LeviCivita(i, j, k)) == LeviCivita(i, j, k)  # 测试 LeviCivita 符号函数的转置


def test_kronecker_delta():
    i, j = symbols('i j')  # 定义符号变量 i 和 j
    k = Symbol('k', nonzero=True)  # 定义非零符号变量 k
    assert KroneckerDelta(1, 1) == 1
    assert KroneckerDelta(1, 2) == 0
    assert KroneckerDelta(k, 0) == 0
    assert KroneckerDelta(x, x) == 1
    assert KroneckerDelta(x**2 - y**2, x**2 - y**2) == 1
    assert KroneckerDelta(i, i) == 1
    assert KroneckerDelta(i, i + 1) == 0
    assert KroneckerDelta(0, 0) == 1
    assert KroneckerDelta(0, 1) == 0
    assert KroneckerDelta(i + k, i) == 0
    assert KroneckerDelta(i + k, i + k) == 1
    assert KroneckerDelta(i + k, i + 1 + k) == 0
    assert KroneckerDelta(i, j).subs({"i": 1, "j": 0}) == 0
    assert KroneckerDelta(i, j).subs({"i": 3, "j": 3}) == 1

    assert KroneckerDelta(i, j)**0 == 1
    for n in range(1, 10):
        assert KroneckerDelta(i, j)**n == KroneckerDelta(i, j)
        assert KroneckerDelta(i, j)**-n == 1/KroneckerDelta(i, j)

    assert KroneckerDelta(i, j).is_integer is True  # 检查 KroneckerDelta 符号函数是否是整数

    assert adjoint(KroneckerDelta(i, j)) == KroneckerDelta(i, j)  # 测试 KroneckerDelta 符号函数的共轭转置
    assert conjugate(KroneckerDelta(i, j)) == KroneckerDelta(i, j)  # 测试 KroneckerDelta 符号函数的共轭
    assert transpose(KroneckerDelta(i, j)) == KroneckerDelta(i, j)  # 测试 KroneckerDelta 符号函数的转置
    # 测试是否为标准形式
    assert (KroneckerDelta(i, j) == KroneckerDelta(j, i)) == True

    assert KroneckerDelta(i, j).rewrite(Piecewise) == Piecewise((0, Ne(i, j)), (1, True))  # 使用分段函数重写 KroneckerDelta 符号函数

    # 带范围的测试：
    assert KroneckerDelta(i, j, (0, i)).args == (i, j, (0, i))
    assert KroneckerDelta(i, j, (-j, i)).delta_range == (-j, i)

    # 如果索引超出范围，返回零：
    assert KroneckerDelta(i, j, (0, i-1)) == 0
    assert KroneckerDelta(-1, j, (0, i-1)) == 0
    assert KroneckerDelta(j, -1, (0, i-1)) == 0
    assert KroneckerDelta(j, i, (0, i-1)) == 0
def test_kronecker_delta_secondquant():
    """secondquant-specific methods"""
    # 引入 KroneckerDelta 类作为 D 的别名
    D = KroneckerDelta
    # 定义符号变量 i, j, v, w，其中 i, j 是 below_fermi 类型的虚拟变量，v, w 是通常的虚拟变量
    i, j, v, w = symbols('i j v w', below_fermi=True, cls=Dummy)
    # 定义符号变量 a, b, t, u，其中 a, b 是 above_fermi 类型的虚拟变量，t, u 是通常的虚拟变量
    a, b, t, u = symbols('a b t u', above_fermi=True, cls=Dummy)
    # 定义符号变量 p, q, r, s，这些是通常类型的虚拟变量
    p, q, r, s = symbols('p q r s', cls=Dummy)

    # 下面的断言测试 KroneckerDelta 对象的属性和方法

    # 断言 D(i, a) 等于 0
    assert D(i, a) == 0
    # 断言 D(i, t) 等于 0
    assert D(i, t) == 0

    # 断言 D(i, j) 的 is_above_fermi 属性为 False
    assert D(i, j).is_above_fermi is False
    # 断言 D(a, b) 的 is_above_fermi 属性为 True
    assert D(a, b).is_above_fermi is True
    # 断言 D(p, q) 的 is_above_fermi 属性为 True
    assert D(p, q).is_above_fermi is True
    # 断言 D(i, q) 的 is_above_fermi 属性为 False
    assert D(i, q).is_above_fermi is False
    # 断言 D(q, i) 的 is_above_fermi 属性为 False
    assert D(q, i).is_above_fermi is False
    # 断言 D(q, v) 的 is_above_fermi 属性为 False
    assert D(q, v).is_above_fermi is False
    # 断言 D(a, q) 的 is_above_fermi 属性为 True
    assert D(a, q).is_above_fermi is True

    # 断言 D(i, j) 的 is_below_fermi 属性为 True
    assert D(i, j).is_below_fermi is True
    # 断言 D(a, b) 的 is_below_fermi 属性为 False
    assert D(a, b).is_below_fermi is False
    # 断言 D(p, q) 的 is_below_fermi 属性为 True
    assert D(p, q).is_below_fermi is True
    # 断言 D(p, j) 的 is_below_fermi 属性为 True
    assert D(p, j).is_below_fermi is True
    # 断言 D(q, b) 的 is_below_fermi 属性为 False
    assert D(q, b).is_below_fermi is False

    # 断言 D(i, j) 的 is_only_above_fermi 属性为 False
    assert D(i, j).is_only_above_fermi is False
    # 断言 D(a, b) 的 is_only_above_fermi 属性为 True
    assert D(a, b).is_only_above_fermi is True
    # 断言 D(p, q) 的 is_only_above_fermi 属性为 False
    assert D(p, q).is_only_above_fermi is False
    # 断言 D(i, q) 的 is_only_above_fermi 属性为 False
    assert D(i, q).is_only_above_fermi is False
    # 断言 D(q, i) 的 is_only_above_fermi 属性为 False
    assert D(q, i).is_only_above_fermi is False
    # 断言 D(a, q) 的 is_only_above_fermi 属性为 True
    assert D(a, q).is_only_above_fermi is True

    # 断言 D(i, j) 的 is_only_below_fermi 属性为 True
    assert D(i, j).is_only_below_fermi is True
    # 断言 D(a, b) 的 is_only_below_fermi 属性为 False
    assert D(a, b).is_only_below_fermi is False
    # 断言 D(p, q) 的 is_only_below_fermi 属性为 False
    assert D(p, q).is_only_below_fermi is False
    # 断言 D(p, j) 的 is_only_below_fermi 属性为 True
    assert D(p, j).is_only_below_fermi is True
    # 断言 D(q, b) 的 is_only_below_fermi 属性为 False
    assert D(q, b).is_only_below_fermi is False

    # 断言 D(i, q).indices_contain_equal_information 为 False
    assert not D(i, q).indices_contain_equal_information
    # 断言 D(a, q).indices_contain_equal_information 为 False
    assert not D(a, q).indices_contain_equal_information
    # 断言 D(p, q).indices_contain_equal_information 为 True
    assert D(p, q).indices_contain_equal_information
    # 断言 D(a, b).indices_contain_equal_information 为 True
    assert D(a, b).indices_contain_equal_information
    # 断言 D(i, j).indices_contain_equal_information 为 True
    assert D(i, j).indices_contain_equal_information

    # 断言 D(q, b).preferred_index 等于 b
    assert D(q, b).preferred_index == b
    # 断言 D(q, b).killable_index 等于 q
    assert D(q, b).killable_index == q
    # 断言 D(q, t).preferred_index 等于 t
    assert D(q, t).preferred_index == t
    # 断言 D(q, t).killable_index 等于 q
    assert D(q, t).killable_index == q
    # 断言 D(q, i).preferred_index 等于 i
    assert D(q, i).preferred_index == i
    # 断言 D(q, i).killable_index 等于 q
    assert D(q, i).killable_index == q
    # 断言 D(q, v).preferred_index 等于 v
    assert D(q, v).preferred_index == v
    # 断言 D(q, v).killable_index 等于 q
    assert D(q, v).killable_index == q
    # 断言 D(q, p).preferred_index 等于 p
    assert D(q, p).preferred_index == p
    # 断言 D(q, p).killable_index 等于 q
    assert D(q, p).killable_index == q

    # 引入 evaluate_deltas 函数作为 EV 的别名
    EV = evaluate_deltas
    # 断言 EV(D(a, q)*F(q)) 等于 F(a)
    assert EV(D(a, q)*F(q)) == F(a)
    # 断言 EV(D(i, q)*F(q)) 等于 F(i)
    assert EV(D(i, q)*F(q)) == F(i)
    # 断言 EV(D(a, q)*F(a)) 等于 D(a, q)*F(a)
    assert EV(D(a, q)*F(a)) == D(a, q)*F(a)
    # 断言 EV(D(i, q)*F(i)) 等于 D(i, q)*F(i)
    assert EV(D(i, q)*F(i)) == D(i, q)*F(i)
    # 断言 EV(D(a, b)*F(a)) 等于 F(b)
    assert EV(D(a, b)*F(a)) == F(b)
    # 断言 EV(D(a, b)*F(b)) 等于 F(a)
    assert EV(D(a, b)*F(b)) == F(a)
    # 断言 EV(D(i, j)*F(i)) 等于 F(j)
    assert EV(D(i, j)*F(i)) == F(j)
    # 断言 EV(D(i, j)*F(j)) 等于 F(i)
    assert EV(D(i, j)*F(j)) == F(i)
    # 断言 EV(D(p, q)*F(q)) 等于 F(p)
    assert EV(D(p, q)*F(q)) == F(p)
    # 断言 EV(D(p, q)*F(p)) 等于 F(q)
    assert EV(D(p, q)*F(p)) == F(q)
```