# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_method.py`

```
# 导入 Sympy 中的 `_Methods` 方法，用于物理力学的计算方法
from sympy.physics.mechanics.method import _Methods
# 导入 Sympy 的测试框架中的 raises 函数，用于测试是否抛出指定类型的异常
from sympy.testing.pytest import raises

# 定义测试函数 `test_method`
def test_method():
    # 断言调用 `_Methods` 类时会引发 `TypeError` 异常，使用 lambda 函数延迟执行
    raises(TypeError, lambda: _Methods())
```