# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_latex_deps.py`

```
# 从 sympy.external 模块中导入 import_module 函数
from sympy.external import import_module
# 从 sympy.testing.pytest 模块中导入 ignore_warnings 和 raises 函数
from sympy.testing.pytest import ignore_warnings, raises

# 尝试导入 antlr4 模块，如果未安装则不发出警告
antlr4 = import_module("antlr4", warn_not_installed=False)

# 如果成功导入了 antlr4 模块，则将 disabled 设为 True
if antlr4:
    disabled = True

# 定义名为 test_no_import 的测试函数
def test_no_import():
    # 从 sympy.parsing.latex 模块中导入 parse_latex 函数
    from sympy.parsing.latex import parse_latex

    # 忽略 UserWarning 并期望抛出 ImportError 异常
    with ignore_warnings(UserWarning):
        with raises(ImportError):
            # 调用 parse_latex 函数尝试解析 LaTeX 表达式 '1 + 1'
            parse_latex('1 + 1')
```