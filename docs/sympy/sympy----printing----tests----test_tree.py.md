# `D:\src\scipysrc\sympy\sympy\printing\tests\test_tree.py`

```
# 从 sympy.printing.tree 模块导入 tree 函数
# 从 sympy.testing.pytest 模块导入 XFAIL 装饰器
from sympy.printing.tree import tree
from sympy.testing.pytest import XFAIL


# 在使 _assumptions 缓存确定性之后移除此标记。
@XFAIL
# 定义一个测试函数，用于测试 MatrixSymbol 的加法表达式的树形打印
def test_print_tree_MatAdd():
    # 从 sympy.matrices.expressions 模块导入 MatrixSymbol 类
    from sympy.matrices.expressions import MatrixSymbol
    # 创建名为 A 和 B 的 MatrixSymbol 实例，各自为 3x3 的矩阵符号
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)

    # 断言语句，比较 A + B 的树形打印结果与指定的字符串是否相等
    assert tree(A + B) == "".join(test_str)


# 定义一个测试函数，用于测试 MatrixSymbol 的加法表达式在没有假设的情况下的树形打印
def test_print_tree_MatAdd_noassumptions():
    # 从 sympy.matrices.expressions 模块导入 MatrixSymbol 类
    from sympy.matrices.expressions import MatrixSymbol
    # 创建名为 A 和 B 的 MatrixSymbol 实例，各自为 3x3 的矩阵符号
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)

    # 指定的树形打印结果字符串，展示 MatAdd: A + B 的结构
    test_str = \
"""MatAdd: A + B
+-MatrixSymbol: A
| +-Str: A
| +-Integer: 3
| +-Integer: 3
+-MatrixSymbol: B
  +-Str: B
  +-Integer: 3
  +-Integer: 3
"""

    # 断言语句，比较 A + B 在 assumptions=False 情况下的树形打印结果与预期的字符串是否相等
    assert tree(A + B, assumptions=False) == test_str
```