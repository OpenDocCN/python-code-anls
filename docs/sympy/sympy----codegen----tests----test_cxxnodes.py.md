# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_cxxnodes.py`

```
# 导入必要的符号和类型相关模块
from sympy.core.symbol import Symbol
from sympy.codegen.ast import Type
from sympy.codegen.cxxnodes import using
from sympy.printing.codeprinter import cxxcode

# 创建一个符号对象 x
x = Symbol('x')

# 定义一个测试函数 test_using
def test_using():
    # 创建一个类型对象，表示 std::vector
    v = Type('std::vector')

    # 使用 using 函数创建一个使用声明，返回一个 using 对象 u1
    u1 = using(v)
    # 断言该 using 对象的 C++ 代码输出应为 'using std::vector'
    assert cxxcode(u1) == 'using std::vector'

    # 使用 using 函数创建一个使用声明，设置别名为 'vec'，返回一个 using 对象 u2
    u2 = using(v, 'vec')
    # 断言该 using 对象的 C++ 代码输出应为 'using vec = std::vector'
    assert cxxcode(u2) == 'using vec = std::vector'
```