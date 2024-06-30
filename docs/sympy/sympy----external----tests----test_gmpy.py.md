# `D:\src\scipysrc\sympy\sympy\external\tests\test_gmpy.py`

```
# 从 sympy.external.gmpy 导入 LONG_MAX 和 iroot 函数
from sympy.external.gmpy import LONG_MAX, iroot
# 从 sympy.testing.pytest 导入 raises 函数，用于测试预期的异常情况


# 定义测试函数 test_iroot，用于测试 iroot 函数的各种情况
def test_iroot():
    # 断言调用 iroot 函数计算给定数的平方根，预期返回 (1, False)
    assert iroot(2, LONG_MAX) == (1, False)
    # 断言调用 iroot 函数计算给定数的平方根，预期返回 (1, False)
    assert iroot(2, LONG_MAX + 1) == (1, False)
    # 对于范围内的每个整数 x 进行测试，调用 iroot 函数计算 x 的 1 次方根，预期返回 (x, True)
    for x in range(3):
        assert iroot(x, 1) == (x, True)
    # 断言调用 iroot 函数处理负数情况，预期引发 ValueError 异常
    raises(ValueError, lambda: iroot(-1, 1))
    # 断言调用 iroot 函数处理 0 的情况，预期引发 ValueError 异常
    raises(ValueError, lambda: iroot(0, 0))
    # 断言调用 iroot 函数处理负指数情况，预期引发 ValueError 异常
    raises(ValueError, lambda: iroot(0, -1))
```