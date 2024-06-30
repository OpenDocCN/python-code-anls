# `D:\src\scipysrc\sympy\sympy\assumptions\tests\test_assumptions_2.py`

```
"""
rename this to test_assumptions.py when the old assumptions system is deleted
"""

# 从 sympy.abc 模块导入符号 x 和 y
from sympy.abc import x, y
# 导入全局假设系统的相关组件
from sympy.assumptions.assume import global_assumptions
# 导入判断假设的工具 Q
from sympy.assumptions.ask import Q
# 导入美化打印输出的函数
from sympy.printing import pretty


def test_equal():
    """Test for equality"""
    # 断言两次 Q.positive(x) 是否相等
    assert Q.positive(x) == Q.positive(x)
    # 断言 Q.positive(x) 和其否定是否不相等
    assert Q.positive(x) != ~Q.positive(x)
    # 断言 Q.positive(x) 的否定和其本身的否定相等
    assert ~Q.positive(x) == ~Q.positive(x)


def test_pretty():
    # 断言美化输出 Q.positive(x) 的结果是否符合预期
    assert pretty(Q.positive(x)) == "Q.positive(x)"
    # 断言美化输出 Q.positive 和 Q.integer 的集合是否符合预期
    assert pretty({Q.positive, Q.integer}) == "{Q.integer, Q.positive}"


def test_global():
    """Test for global assumptions"""
    # 添加 x >
```