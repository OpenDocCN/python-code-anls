# `D:\src\scipysrc\sympy\sympy\strategies\branch\tests\test_tools.py`

```
# 从 sympy.strategies.branch.tools 中导入 canon 函数
from sympy.strategies.branch.tools import canon
# 从 sympy.core.basic 中导入 Basic 类
from sympy.core.basic import Basic
# 从 sympy.core.numbers 中导入 Integer 类
from sympy.core.numbers import Integer
# 从 sympy.core.singleton 中导入 S 单例对象
from sympy.core.singleton import S

# 定义一个生成器函数 posdec，用于处理整数 x
def posdec(x):
    # 如果 x 是 Integer 类型且大于 0
    if isinstance(x, Integer) and x > 0:
        # 生成 x-1 的值
        yield x - 1
    else:
        # 否则生成原始 x 的值
        yield x

# 定义一个函数 branch5，用于处理整数 x
def branch5(x):
    # 如果 x 是 Integer 类型
    if isinstance(x, Integer):
        # 如果 x 在 0 到 5 之间（不包括 5）
        if 0 < x < 5:
            # 生成 x-1 的值
            yield x - 1
        # 如果 x 在 5 到 10 之间（不包括 10）
        elif 5 < x < 10:
            # 生成 x+1 的值
            yield x + 1
        # 如果 x 等于 5
        elif x == 5:
            # 生成 x+1 的值
            yield x + 1
            # 生成 x-1 的值
            yield x - 1
        else:
            # 其他情况生成原始 x 的值
            yield x

# 定义一个测试函数 test_zero_ints
def test_zero_ints():
    # 创建一个表达式 expr
    expr = Basic(S(2), Basic(S(5), S(3)), S(8))
    # 创建预期结果集合 expected
    expected = {Basic(S(0), Basic(S(0), S(0)), S(0))}

    # 使用 canon 函数对 posdec 函数进行规范化
    brl = canon(posdec)
    # 断言规范化后的结果集合与预期结果集合相等
    assert set(brl(expr)) == expected

# 定义一个测试函数 test_split5
def test_split5():
    # 创建一个表达式 expr
    expr = Basic(S(2), Basic(S(5), S(3)), S(8))
    # 创建预期结果集合 expected
    expected = {
        Basic(S(0), Basic(S(0), S(0)), S(10)),
        Basic(S(0), Basic(S(10), S(0)), S(10))}

    # 使用 canon 函数对 branch5 函数进行规范化
    brl = canon(branch5)
    # 断言规范化后的结果集合与预期结果集合相等
    assert set(brl(expr)) == expected
```