# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_pynodes.py`

```
# 从 sympy.core.symbol 模块导入 symbols 符号函数
# 从 sympy.codegen.pynodes 模块导入 List 类
from sympy.core.symbol import symbols
from sympy.codegen.pynodes import List

# 定义测试函数 test_List
def test_List():
    # 创建一个 List 对象 l，包含元素 2, 3, 4
    l = List(2, 3, 4)
    # 断言 l 应该与包含元素 2, 3, 4 的 List 对象相等
    assert l == List(2, 3, 4)
    # 断言 l 的字符串表示应该为 "[2, 3, 4]"
    assert str(l) == "[2, 3, 4]"
    # 使用 symbols 函数创建符号变量 x, y, z
    x, y, z = symbols('x y z')
    # 创建一个 List 对象 l，包含符号表达式 x**2, y**3, z**4
    l = List(x**2, y**3, z**4)
    # 对 List 对象 l 调用 replace 方法，传入两个 lambda 表达式作为参数
    # 第一个 lambda 表达式判断参数是否为幂运算并且指数大于 2
    # 第二个 lambda 表达式对符合条件的参数执行替换操作
    m = l.replace(lambda arg: arg.is_Pow and arg.exp > 2, lambda p: p.base - p.exp)
    # 断言 m 应该与包含符号表达式 x**2, y-3, z-4 的列表相等
    assert m == [x**2, y - 3, z - 4]
```