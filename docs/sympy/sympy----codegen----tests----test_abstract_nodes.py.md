# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_abstract_nodes.py`

```
# 导入符号和列表模块
from sympy.core.symbol import symbols
from sympy.codegen.abstract_nodes import List

# 定义测试函数 test_List
def test_List():
    # 创建一个 List 对象，包含元素 2, 3, 4
    l = List(2, 3, 4)
    # 断言 l 等于 List(2, 3, 4)
    assert l == List(2, 3, 4)
    # 断言 l 的字符串表示为 "[2, 3, 4]"
    assert str(l) == "[2, 3, 4]"
    
    # 创建符号变量 x, y, z
    x, y, z = symbols('x y z')
    # 创建一个新的 List 对象，包含 x 的平方, y 的立方, z 的四次方
    l = List(x**2, y**3, z**4)
    
    # 对 List 进行操作，替换其中符合条件的元素
    # 这里与 Python 内置的列表不同，我们可以在 List 上调用 "replace" 方法
    m = l.replace(lambda arg: arg.is_Pow and arg.exp > 2, lambda p: p.base - p.exp)
    
    # 断言 m 等于 [x**2, y-3, z-4]
    assert m == [x**2, y-3, z-4]
    
    # 计算 m 的哈希值
    hash(m)
```