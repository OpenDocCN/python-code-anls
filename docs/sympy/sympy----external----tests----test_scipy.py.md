# `D:\src\scipysrc\sympy\sympy\external\tests\test_scipy.py`

```
# 导入外部模块 import_module，用于动态导入 scipy
from sympy.external import import_module

# 检查是否成功导入 scipy 模块，如果未成功导入，则禁用测试
scipy = import_module('scipy')
if not scipy:
    # 禁用测试的标志位
    disabled = True

# 导入 sympy 函数库中的贝塞尔函数零点函数 jn_zeros
from sympy.functions.special.bessel import jn_zeros


# 定义一个函数 eq，用于比较两个列表 a 和 b 是否在给定的误差范围内相等
def eq(a, b, tol=1e-6):
    # 使用 zip 函数将列表 a 和 b 的对应元素进行比较
    for x, y in zip(a, b):
        # 如果两个元素之差的绝对值超过了指定的误差范围，则返回 False
        if not (abs(x - y) < tol):
            return False
    # 如果所有对应元素的差值都在误差范围内，则返回 True
    return True


# 定义一个测试函数 test_jn_zeros，用于测试 jn_zeros 函数的功能
def test_jn_zeros():
    # 断言语句，用于验证 jn_zeros 返回的结果是否符合预期
    assert eq(jn_zeros(0, 4, method="scipy"),
            [3.141592, 6.283185, 9.424777, 12.566370])
    assert eq(jn_zeros(1, 4, method="scipy"),
            [4.493409, 7.725251, 10.904121, 14.066193])
    assert eq(jn_zeros(2, 4, method="scipy"),
            [5.763459, 9.095011, 12.322940, 15.514603])
    assert eq(jn_zeros(3, 4, method="scipy"),
            [6.987932, 10.417118, 13.698023, 16.923621])
    assert eq(jn_zeros(4, 4, method="scipy"),
            [8.182561, 11.704907, 15.039664, 18.301255])
```