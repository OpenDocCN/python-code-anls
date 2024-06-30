# `D:\src\scipysrc\sympy\sympy\strategies\tests\test_tools.py`

```
# 导入 sympy 库中特定的模块和函数：subs, typed
from sympy.strategies.tools import subs, typed
# 导入 sympy 库中的 rl 模块，从中导入 rm_id 函数
from sympy.strategies.rl import rm_id
# 从 sympy 库中导入 core 模块的 Basic 类
from sympy.core.basic import Basic
# 从 sympy 库中导入 core.singleton 模块中的 S 对象
from sympy.core.singleton import S


# 定义测试函数 test_subs
def test_subs():
    # 从 sympy.core.symbol 模块中导入 symbols 函数
    from sympy.core.symbol import symbols
    # 使用 symbols 函数定义符号变量 a, b, c, d, e, f
    a, b, c, d, e, f = symbols('a,b,c,d,e,f')
    # 定义映射字典 mapping，将 a 映射到 d，将 d 映射到 a，将 Basic(e) 映射到 Basic(f)
    mapping = {a: d, d: a, Basic(e): Basic(f)}
    # 创建表达式 expr，使用 Basic 类进行嵌套，结构为 Basic(a, Basic(b, c), Basic(d, Basic(e)))
    expr = Basic(a, Basic(b, c), Basic(d, Basic(e)))
    # 创建预期结果 result，结构为 Basic(d, Basic(b, c), Basic(a, Basic(f)))
    result = Basic(d, Basic(b, c), Basic(a, Basic(f)))
    # 断言替换操作 subs(mapping)(expr) 的结果等于预期结果 result
    assert subs(mapping)(expr) == result


# 定义测试函数 test_subs_empty
def test_subs_empty():
    # 断言使用空字典进行替换操作 subs({})(Basic(S(1), S(2))) 等于 Basic(S(1), S(2))
    assert subs({})(Basic(S(1), S(2))) == Basic(S(1), S(2))


# 定义测试函数 test_typed
def test_typed():
    # 定义类 A，继承自 Basic 类
    class A(Basic):
        pass

    # 定义类 B，继承自 Basic 类
    class B(Basic):
        pass

    # 定义 rmzeros 函数，使用 rm_id 匿名函数移除等于 S(0) 的元素
    rmzeros = rm_id(lambda x: x == S(0))
    # 定义 rmones 函数，使用 rm_id 匿名函数移除等于 S(1) 的元素
    rmones = rm_id(lambda x: x == S(1))
    # 创建 remove_something 字典，键为 A 类型，值为 rmzeros 函数；键为 B 类型，值为 rmones 函数
    remove_something = typed({A: rmzeros, B: rmones})

    # 断言 remove_something(A(S(0), S(1))) 的结果等于 A(S(1))
    assert remove_something(A(S(0), S(1))) == A(S(1))
    # 断言 remove_something(B(S(0), S(1))) 的结果等于 B(S(0))
    assert remove_something(B(S(0), S(1))) == B(S(0))
```