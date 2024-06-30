# `D:\src\scipysrc\sympy\sympy\multipledispatch\tests\test_core.py`

```
# 导入必要的模块和类型声明
from __future__ import annotations
from typing import Any

# 导入 sympy.multipledispatch 中的相关组件
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns

# 导入 functools 模块的 partial 函数
from functools import partial

# 创建一个空的命名空间字典，用于存储分派函数
test_namespace: dict[str, Any] = {}

# 备份原始的 dispatch 函数
orig_dispatch = dispatch

# 将 dispatch 函数修改为其部分应用，绑定 namespace 参数到 test_namespace
dispatch = partial(dispatch, namespace=test_namespace)


# 定义测试单一分派函数的函数
def test_singledispatch():
    @dispatch(int)
    def f(x):  # noqa:F811
        return x + 1

    @dispatch(int)
    def g(x):  # noqa:F811
        return x + 2

    @dispatch(float)  # noqa:F811
    def f(x):  # noqa:F811
        return x - 1

    # 断言不同输入类型的分派函数计算正确
    assert f(1) == 2
    assert g(1) == 3
    assert f(1.0) == 0

    # 断言对于不支持的输入类型会引发 NotImplementedError
    assert raises(NotImplementedError, lambda: f('hello'))


# 定义测试多重分派函数的函数
def test_multipledispatch():
    @dispatch(int, int)
    def f(x, y):  # noqa:F811
        return x + y

    @dispatch(float, float)  # noqa:F811
    def f(x, y):  # noqa:F811
        return x - y

    # 断言不同输入类型的分派函数计算正确
    assert f(1, 2) == 3
    assert f(1.0, 2.0) == -1.0


# 定义用于继承关系测试的类
class A: pass
class B: pass
class C(A): pass
class D(C): pass
class E(C): pass


# 定义测试继承关系中的分派函数
def test_inheritance():
    @dispatch(A)
    def f(x):  # noqa:F811
        return 'a'

    @dispatch(B)  # noqa:F811
    def f(x):  # noqa:F811
        return 'b'

    # 断言分派函数根据对象类型返回正确的结果
    assert f(A()) == 'a'
    assert f(B()) == 'b'
    assert f(C()) == 'a'


# 定义测试继承关系和多重分派的函数
def test_inheritance_and_multiple_dispatch():
    @dispatch(A, A)
    def f(x, y):  # noqa:F811
        return type(x), type(y)

    @dispatch(A, B)  # noqa:F811
    def f(x, y):  # noqa:F811
        return 0

    # 断言不同输入类型的分派函数计算正确
    assert f(A(), A()) == (A, A)
    assert f(A(), C()) == (A, C)
    assert f(A(), B()) == 0
    assert f(C(), B()) == 0
    assert raises(NotImplementedError, lambda: f(B(), B()))


# 定义测试存在竞争解的函数
def test_competing_solutions():
    @dispatch(A)
    def h(x):  # noqa:F811
        return 1

    @dispatch(C)  # noqa:F811
    def h(x):  # noqa:F811
        return 2

    # 断言竞争解函数对特定对象类型返回正确的结果
    assert h(D()) == 2


# 定义测试存在竞争解和多重分派的函数
def test_competing_multiple():
    @dispatch(A, B)
    def h(x, y):  # noqa:F811
        return 1

    @dispatch(C, B)  # noqa:F811
    def h(x, y):  # noqa:F811
        return 2

    # 断言竞争解和多重分派函数对特定对象类型返回正确的结果
    assert h(D(), B()) == 2


# 定义测试存在模糊匹配的函数
def test_competing_ambiguous():
    test_namespace = {}
    dispatch = partial(orig_dispatch, namespace=test_namespace)

    @dispatch(A, C)
    def f(x, y):  # noqa:F811
        return 2

    with warns(AmbiguityWarning, test_stacklevel=False):
        @dispatch(C, A)  # noqa:F811
        def f(x, y):  # noqa:F811
            return 2

    # 断言模糊匹配函数对特定对象类型返回正确的结果
    assert f(A(), C()) == f(C(), A()) == 2
    # assert raises(Warning, lambda : f(C(), C()))


# 定义测试分派函数缓存正确行为的函数
def test_caching_correct_behavior():
    @dispatch(A)
    def f(x):  # noqa:F811
        return 1

    # 断言分派函数对特定对象类型的缓存行为表现正确
    assert f(C()) == 1

    @dispatch(C)
    def f(x):  # noqa:F811
        return 2

    # 断言更新后的分派函数对特定对象类型返回正确的结果
    assert f(C()) == 2


# 定义测试并集类型的函数
def test_union_types():
    @dispatch((A, C))
    def f(x):  # noqa:F811
        return 1

    # 断言并集类型的分派函数对不同类型对象返回正确的结果
    assert f(A()) == 1
    assert f(C()) == 1


# 定义测试命名空间的函数
def test_namespaces():
    ns1 = {}
    ns2 = {}

    def foo(x):
        return 1
    # 使用 orig_dispatch 函数对 foo 函数进行装饰，并将结果赋给 foo1
    foo1 = orig_dispatch(int, namespace=ns1)(foo)

    # 定义一个简单的函数 foo，返回固定的整数 2
    def foo(x):
        return 2
    
    # 使用 orig_dispatch 函数对 foo 函数进行装饰，并将结果赋给 foo2
    foo2 = orig_dispatch(int, namespace=ns2)(foo)

    # 断言调用 foo1(0) 应该返回 1
    assert foo1(0) == 1
    # 断言调用 foo2(0) 应该返回 2
    assert foo2(0) == 2
"""
Fails
def test_dispatch_on_dispatch():
    @dispatch(A)
    @dispatch(C)
    def q(x): # noqa:F811
        return 1

    assert q(A()) == 1
    assert q(C()) == 1
"""

# 定义测试函数 test_methods，测试方法的多态性
def test_methods():
    # 定义类 Foo
    class Foo:
        # 注册方法 f，当参数 x 为 float 类型时执行
        @dispatch(float)
        def f(self, x): # noqa:F811
            return x - 1

        # 注册方法 f，当参数 x 为 int 类型时执行
        @dispatch(int) # noqa:F811
        def f(self, x): # noqa:F811
            return x + 1

        # 注册方法 g，当参数 x 为 int 类型时执行
        @dispatch(int)
        def g(self, x): # noqa:F811
            return x + 3

    # 创建 Foo 类的实例 foo
    foo = Foo()
    # 断言调用 foo.f(1) 返回 2
    assert foo.f(1) == 2
    # 断言调用 foo.f(1.0) 返回 0.0
    assert foo.f(1.0) == 0.0
    # 断言调用 foo.g(1) 返回 4
    assert foo.g(1) == 4


# 定义测试函数 test_methods_multiple_dispatch，测试方法的多重分派
def test_methods_multiple_dispatch():
    # 定义类 Foo
    class Foo:
        # 注册方法 f，当参数 x 和 y 都为 A 类型时执行
        @dispatch(A, A)
        def f(x, y): # noqa:F811
            return 1

        # 注册方法 f，当参数 x 为 A 类型，y 为 C 类型时执行
        @dispatch(A, C) # noqa:F811
        def f(x, y): # noqa:F811
            return 2

    # 创建 Foo 类的实例 foo
    foo = Foo()
    # 断言调用 foo.f(A(), A()) 返回 1
    assert foo.f(A(), A()) == 1
    # 断言调用 foo.f(A(), C()) 返回 2
    assert foo.f(A(), C()) == 2
    # 断言调用 foo.f(C(), C()) 返回 2
    assert foo.f(C(), C()) == 2
```