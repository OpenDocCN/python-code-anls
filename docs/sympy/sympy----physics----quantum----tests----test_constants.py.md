# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_constants.py`

```
# 导入浮点数类型的实现
from sympy.core.numbers import Float

# 从量子物理常量模块中导入约化普朗克常数 hbar
from sympy.physics.quantum.constants import hbar

# 定义测试函数 test_hbar，用于验证 hbar 的属性和计算准确性
def test_hbar():
    # 断言 hbar 是可交换的（符号运算）
    assert hbar.is_commutative is True
    # 断言 hbar 是实数
    assert hbar.is_real is True
    # 断言 hbar 是正数
    assert hbar.is_positive is True
    # 断言 hbar 不是负数
    assert hbar.is_negative is False
    # 断言 hbar 是无理数
    assert hbar.is_irrational is True

    # 断言 hbar 的数值近似等于给定的浮点数
    assert hbar.evalf() == Float(1.05457162e-34)
```