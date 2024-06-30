# `D:\src\scipysrc\sympy\sympy\printing\tests\test_gtk.py`

```
# 导入 SymPy 库中的特定模块和函数，分别为 sin 函数、print_gtk 函数以及测试相关的装饰器 XFAIL 和 raises
from sympy.functions.elementary.trigonometric import sin
from sympy.printing.gtk import print_gtk
from sympy.testing.pytest import XFAIL, raises

# XFAIL 装饰的测试函数，表示当 python-lxml 未安装时测试会失败，不希望依赖 SymPy 中的任何组件

@XFAIL
def test_1():
    # 从 sympy.abc 模块导入符号 x
    from sympy.abc import x
    # 打印 x**2 的图形，不启动视图器
    print_gtk(x**2, start_viewer=False)
    # 打印 x**2 + sin(x)/4 的图形，不启动视图器
    print_gtk(x**2 + sin(x)/4, start_viewer=False)

# 测试函数 test_settings，验证 print_gtk 函数的参数设置是否能够引发 TypeError 异常

def test_settings():
    # 从 sympy.abc 模块导入符号 x
    from sympy.abc import x
    # 使用 lambda 函数调用 print_gtk，验证 method 参数设置为 "garbage" 是否会引发 TypeError 异常
    raises(TypeError, lambda: print_gtk(x, method="garbage"))
```