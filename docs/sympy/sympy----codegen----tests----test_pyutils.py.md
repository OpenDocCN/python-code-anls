# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_pyutils.py`

```
# 从 sympy.codegen.ast 模块中导入 Print 类
from sympy.codegen.ast import Print
# 从 sympy.codegen.pyutils 模块中导入 render_as_module 函数

def test_standard():
    # 创建一个 Print 对象，打印 'x' 和 'y' 两个变量，格式化输出坐标信息
    ast = Print('x y'.split(), r"coordinate: %12.5g %12.5g\n")
    # 断言使用 render_as_module 函数将 ast 对象渲染为 Python3 标准下的字符串
    assert render_as_module(ast, standard='python3') == \
        '\n\nprint("coordinate: %12.5g %12.5g\\n" % (x, y), end="")'
```