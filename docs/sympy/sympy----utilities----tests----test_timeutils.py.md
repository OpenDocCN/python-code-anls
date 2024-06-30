# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_timeutils.py`

```
"""Tests for simple tools for timing functions' execution. """

# 从 sympy.utilities.timeutils 中导入 timed 工具函数
from sympy.utilities.timeutils import timed

# 定义一个测试函数 test_timed
def test_timed():
    # 使用 timed 函数测试 lambda 函数的执行时间，限制为 100000 循环
    result = timed(lambda: 1 + 1, limit=100000)
    # 断言结果中的第一个元素为 100000，第四个元素为 "ns"
    assert result[0] == 100000 and result[3] == "ns", str(result)

    # 使用 timed 函数测试字符串表达式 "1 + 1" 的执行时间，限制为 100000 循环
    result = timed("1 + 1", limit=100000)
    # 断言结果中的第一个元素为 100000，第四个元素为 "ns"
    assert result[0] == 100000 and result[3] == "ns"
```