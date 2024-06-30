# `D:\src\scipysrc\sympy\sympy\core\tests\test_var.py`

```
# 导入所需模块和函数
from sympy.core.function import (Function, FunctionClass)
from sympy.core.symbol import (Symbol, var)
from sympy.testing.pytest import raises

# 定义测试函数：测试单个变量的创建
def test_var():
    # 准备命名空间，包括 var 和 raises 函数
    ns = {"var": var, "raises": raises}
    # 使用 eval 函数执行 var('a')，将变量名 'a' 加入命名空间
    eval("var('a')", ns)
    # 断言命名空间中 'a' 对应的符号是 Symbol('a')
    assert ns["a"] == Symbol("a")

    # 使用 eval 函数执行 var('b bb cc zz _x')，将多个变量名加入命名空间
    eval("var('b bb cc zz _x')", ns)
    # 逐一断言命名空间中各变量对应的符号名称是否符合预期
    assert ns["b"] == Symbol("b")
    assert ns["bb"] == Symbol("bb")
    assert ns["cc"] == Symbol("cc")
    assert ns["zz"] == Symbol("zz")
    assert ns["_x"] == Symbol("_x")

    # 使用 eval 函数执行 var(['d', 'e', 'fg'])，将列表中的变量名加入命名空间
    v = eval("var(['d', 'e', 'fg'])", ns)
    # 逐一断言命名空间中各变量对应的符号是否与预期的 Symbol 对象相等
    assert ns['d'] == Symbol('d')
    assert ns['e'] == Symbol('e')
    assert ns['fg'] == Symbol('fg')

    # 检查返回值 v 不应该是原始列表 ['d', 'e', 'fg']，而应该是 Symbol 对象组成的列表
    assert v != ['d', 'e', 'fg']
    assert v == [Symbol('d'), Symbol('e'), Symbol('fg')]


# 定义测试函数：测试 var 函数的返回值
def test_var_return():
    # 准备命名空间，包括 var 和 raises 函数
    ns = {"var": var, "raises": raises}
    # "raises(ValueError, lambda: var(''))" 这行字符串并无实际作用，应忽略

    # 使用 eval 函数执行 var('q')，将变量名 'q' 加入命名空间
    v2 = eval("var('q')", ns)
    # 使用 eval 函数执行 var('q p')，将多个变量名加入命名空间
    v3 = eval("var('q p')", ns)

    # 断言命名空间中 'q' 对应的符号是 Symbol('q')
    assert v2 == Symbol('q')
    # 断言命名空间中 'q' 和 'p' 对应的符号是符合预期的 Symbol 对象
    assert v3 == (Symbol('q'), Symbol('p'))


# 定义测试函数：测试 var 函数接受逗号分隔的变量名
def test_var_accepts_comma():
    # 准备命名空间，包括 var 函数
    ns = {"var": var}
    # 使用 eval 函数执行 var('x y z')，将多个变量名加入命名空间
    v1 = eval("var('x y z')", ns)
    # 使用 eval 函数执行 var('x,y,z')，将逗号分隔的多个变量名加入命名空间
    v2 = eval("var('x,y,z')", ns)
    # 使用 eval 函数执行 var('x,y z')，将包含逗号和空格的多个变量名加入命名空间
    v3 = eval("var('x,y z')", ns)

    # 断言各种方式定义的变量名生成的符号对象是一致的
    assert v1 == v2
    assert v1 == v3


# 定义测试函数：测试 var 函数的关键字参数
def test_var_keywords():
    # 准备命名空间，包括 var 函数
    ns = {"var": var}
    # 使用 eval 函数执行 var('x y', real=True)，创建两个实数变量 'x' 和 'y'
    eval("var('x y', real=True)", ns)
    # 断言命名空间中 'x' 和 'y' 对应的符号对象具有实数属性
    assert ns['x'].is_real and ns['y'].is_real


# 定义测试函数：测试 var 函数的 cls 参数
def test_var_cls():
    # 准备命名空间，包括 var 和 Function 函数
    ns = {"var": var, "Function": Function}
    # 使用 eval 函数执行 var('f', cls=Function)，创建函数符号 'f'
    eval("var('f', cls=Function)", ns)

    # 断言命名空间中 'f' 对应的对象是 FunctionClass 类的实例
    assert isinstance(ns['f'], FunctionClass)

    # 使用 eval 函数执行 var('g,h', cls=Function)，创建多个函数符号 'g' 和 'h'
    eval("var('g,h', cls=Function)", ns)

    # 断言命名空间中 'g' 和 'h' 对应的对象都是 FunctionClass 类的实例
    assert isinstance(ns['g'], FunctionClass)
    assert isinstance(ns['h'], FunctionClass)
```