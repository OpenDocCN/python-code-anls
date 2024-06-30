# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_ast_parser.py`

```
# 从 sympy.core.singleton 模块导入 S（单例对象）
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 symbols（符号变量）
from sympy.core.symbol import symbols
# 从 sympy.parsing.ast_parser 模块导入 parse_expr（表达式解析函数）
from sympy.parsing.ast_parser import parse_expr
# 从 sympy.testing.pytest 模块导入 raises（用于测试时引发异常）
from sympy.testing.pytest import raises
# 从 sympy.core.sympify 模块导入 SympifyError（符号化错误）
from sympy.core.sympify import SympifyError
# 导入 warnings 模块（用于警告处理）
import warnings

# 定义测试函数 test_parse_expr
def test_parse_expr():
    # 使用 symbols 函数创建符号变量 a 和 b
    a, b = symbols('a, b')

    # 测试 issue_16393
    assert parse_expr('a + b', {}) == a + b
    # 断言解析 'a +' 表达式时引发 SympifyError 异常
    raises(SympifyError, lambda: parse_expr('a + ', {}))

    # 测试 Transform.visit_Constant
    assert parse_expr('1 + 2', {}) == S(3)
    assert parse_expr('1 + 2.0', {}) == S(3.0)

    # 测试 Transform.visit_Name
    assert parse_expr('Rational(1, 2)', {}) == S(1)/2
    # 断言解析 'a' 表达式时，使用给定的符号变量字典 {'a': a}
    assert parse_expr('a', {'a': a}) == a

    # 测试 issue_23092
    # 使用 warnings 模块捕获警告
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # 断言解析 '6 * 7' 表达式为 S(42)，并且不产生警告
        assert parse_expr('6 * 7', {}) == S(42)
```