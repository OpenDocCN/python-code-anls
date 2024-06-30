# `D:\src\scipysrc\sympy\sympy\core\tests\test_parameters.py`

```
# 从 sympy.abc 模块中导入符号变量 x, y
from sympy.abc import x, y
# 从 sympy.core.parameters 模块中导入 evaluate 函数
from sympy.core.parameters import evaluate
# 从 sympy.core 模块中导入 Mul, Add, Pow, S 类
from sympy.core import Mul, Add, Pow, S
# 从 sympy.core.numbers 模块中导入 oo（无穷大）常量
from sympy.core.numbers import oo
# 从 sympy.functions.elementary.miscellaneous 模块中导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt

# 定义测试函数 test_add
def test_add():
    # 进入 evaluate(False) 上下文
    with evaluate(False):
        # 检查 oo - oo 的结果
        p = oo - oo
        assert isinstance(p, Add) and p.args == (oo, -oo)
        # 检查 5 - oo 的结果
        p = 5 - oo
        assert isinstance(p, Add) and p.args == (-oo, 5)
        # 检查 oo - 5 的结果
        p = oo - 5
        assert isinstance(p, Add) and p.args == (oo, -5)
        # 检查 oo + 5 的结果
        p = oo + 5
        assert isinstance(p, Add) and p.args == (oo, 5)
        # 检查 5 + oo 的结果
        p = 5 + oo
        assert isinstance(p, Add) and p.args == (oo, 5)
        # 检查 -oo + 5 的结果
        p = -oo + 5
        assert isinstance(p, Add) and p.args == (-oo, 5)
        # 检查 -5 - oo 的结果
        p = -5 - oo
        assert isinstance(p, Add) and p.args == (-oo, -5)

    # 再次进入 evaluate(False) 上下文
    with evaluate(False):
        # 检查 x + x 的结果
        expr = x + x
        assert isinstance(expr, Add)
        assert expr.args == (x, x)

        # 再次进入 evaluate(True) 上下文
        with evaluate(True):
            # 断言 (x + x).args 的结果为 (2, x)
            assert (x + x).args == (2, x)

        # 断言 (x + x).args 的结果为 (x, x)
        assert (x + x).args == (x, x)

    # 断言 x + x 的结果为 Mul 类型
    assert isinstance(x + x, Mul)

    # 进入 evaluate(False) 上下文
    with evaluate(False):
        # 断言 S.One + 1 的结果为 Add(1, 1)
        assert S.One + 1 == Add(1, 1)
        # 断言 1 + S.One 的结果为 Add(1, 1)

        assert 1 + S.One == Add(1, 1)
        # 断言 S(4) - 3 的结果为 Add(4, -3)
        assert S(4) - 3 == Add(4, -3)
        # 断言 -3 + S(4) 的结果为 Add(4, -3)
        assert -3 + S(4) == Add(4, -3)

        # 断言 S(2) * 4 的结果为 Mul(2, 4)
        assert S(2) * 4 == Mul(2, 4)
        # 断言 4 * S(2) 的结果为 Mul(2, 4)
        assert 4 * S(2) == Mul(2, 4)

        # 断言 S(6) / 3 的结果为 Mul(6, Pow(3, -1))
        assert S(6) / 3 == Mul(6, Pow(3, -1))
        # 断言 S.One / 3 * 6 的结果为 Mul(S.One / 3, 6)
        assert S.One / 3 * 6 == Mul(S.One / 3, 6)

        # 断言 9 ** S(2) 的结果为 Pow(9, 2)
        assert 9 ** S(2) == Pow(9, 2)
        # 断言 S(2) ** 9 的结果为 Pow(2, 9)
        assert S(2) ** 9 == Pow(2, 9)

        # 断言 S(2) / 2 的结果为 Mul(2, Pow(2, -1))
        assert S(2) / 2 == Mul(2, Pow(2, -1))
        # 断言 S.One / 2 * 2 的结果为 Mul(S.One / 2, 2)
        assert S.One / 2 * 2 == Mul(S.One / 2, 2)

        # 断言 S(2) / 3 + 1 的结果为 Add(S(2) / 3, 1)
        assert S(2) / 3 + 1 == Add(S(2) / 3, 1)
        # 断言 1 + S(2) / 3 的结果为 Add(1, S(2) / 3)
        assert 1 + S(2) / 3 == Add(1, S(2) / 3)

        # 断言 S(4) / 7 - 3 的结果为 Add(S(4) / 7, -3)
        assert S(4) / 7 - 3 == Add(S(4) / 7, -3)
        # 断言 -3 + S(4) / 7 的结果为 Add(-3, S(4) / 7)

        assert -3 + S(4) / 7 == Add(-3, S(4) / 7)
        # 断言 S(2) / 4 * 4 的结果为 Mul(S(2) / 4, 4)
        assert S(2) / 4 * 4 == Mul(S(2) / 4, 4)

        # 断言 4 * (S(2) / 4) 的结果为 Mul(4, S(2) / 4)
        assert 4 * (S(2) / 4) == Mul(4, S(2) / 4)

        # 断言 S(6) / 3 的结果为 Mul(6, Pow(3, -1))
        assert S(6) / 3 == Mul(6, Pow(3, -1))
        # 断言 S.One / 3 * 6 的结果为 Mul(S.One / 3, 6)
        assert S.One / 3 * 6 == Mul(S.One / 3, 6)

        # 断言 S.One / 3 + sqrt(3) 的结果为 Add(S.One / 3, sqrt(3))
        assert S.One / 3 + sqrt(3) == Add(S.One / 3, sqrt(3))
        # 断言 sqrt(3) + S.One / 3 的结果为 Add(sqrt(3), S.One / 3)
        assert sqrt(3) + S.One / 3 == Add(sqrt(3), S.One / 3)

        # 断言 S.One / 2 * 10.333 的结果为 Mul(S.One / 2, 10.333)
        assert S.One / 2 * 10.333 == Mul(S.One / 2, 10.333)
        # 断言 10.333 * (S.One / 2) 的结果为 Mul(10.333, S.One / 2)
        assert 10.333 * (S.One / 2) == Mul(10.333, S.One / 2)

        # 断言 sqrt(2) * sqrt(2) 的结果为 Mul(sqrt(2), sqrt(2))
        assert sqrt(2) * sqrt(2) == Mul(sqrt(2), sqrt(2))

        # 断言 S.One / 2 + x 的结果为 Add(S.One / 2, x)
        assert S.One / 2 + x == Add(S.One / 2, x)
        # 断言 x + S.One / 2 的结果为 Add(x, S.One / 2)

        assert x + S.One / 2 == Add(x, S.One / 2)
        # 断言 S.One / x * x 的结果为 Mul(S.One / x, x)

        assert S.One / x * x == Mul(S.One / x, x)
        # 断言 x * (S.One / x) 的结果为 Mul(x, Pow(x, -1))

        assert x * (S.One / x) == Mul(x, Pow(x, -1))

        # 断言 S.One / 3 的结果为 Pow(3, -1)
        assert S.One / 3 == Pow(3, -1)
        # 断言 S.One / x 的结果为 Pow(x, -1
    # 进入禁用符号表达式求值的上下文
    with evaluate(False):
        # 创建一个表达式 `x + x`，它的参数应为 `(x, x)`
        expr = x + x
        # 断言表达式的参数是否为 `(x, x)`
        assert expr.args == (x, x)
        
        # 进入启用符号表达式求值的上下文
        with evaluate(True):
            # 创建一个表达式 `x + x`，期望它的参数为 `(2, x)`
            expr = x + x
            # 断言表达式的参数是否为 `(2, x)`
            assert expr.args == (2, x)
        
        # 恢复到禁用符号表达式求值的上下文
        expr = x + x
        # 断言表达式的参数是否为 `(x, x)`，此时应与最初的情况相同
        assert expr.args == (x, x)
# 定义一个测试函数，用于验证表达式的重用性
def test_reusability():
    # 调用 evaluate 函数，传入 False 参数，返回一个上下文对象 f
    f = evaluate(False)

    # 进入上下文 f
    with f:
        # 创建一个表达式 x + x
        expr = x + x
        # 断言表达式的参数应为 (x, x)
        assert expr.args == (x, x)

    # 创建一个新的表达式 x + x
    expr = x + x
    # 断言表达式的参数应为 (2, x)
    assert expr.args == (2, x)

    # 再次进入上下文 f
    with f:
        # 创建一个表达式 x + x
        expr = x + x
        # 断言表达式的参数应为 (x, x)
        assert expr.args == (x, x)

    # 确保重入性与可重用性
    # 创建一个新的上下文对象 ctx
    ctx = evaluate(False)
    with ctx:
        # 创建一个表达式 x + x
        expr = x + x
        # 断言表达式的参数应为 (x, x)
        assert expr.args == (x, x)
        with ctx:
            # 再次进入上下文 ctx
            expr = x + x
            # 断言表达式的参数应为 (x, x)
            assert expr.args == (x, x)

    # 创建一个新的表达式 x + x
    expr = x + x
    # 断言表达式的参数应为 (2, x)
    assert expr.args == (2, x)
```