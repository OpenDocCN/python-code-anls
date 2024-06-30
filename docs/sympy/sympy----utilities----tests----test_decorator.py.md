# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_decorator.py`

```
# 导入必要的装饰器和测试辅助函数
from functools import wraps
from sympy.utilities.decorator import threaded, xthreaded, memoize_property, deprecated
from sympy.testing.pytest import warns_deprecated_sympy

# 导入必要的符号和类定义
from sympy.core.basic import Basic
from sympy.core.relational import Eq
from sympy.matrices.dense import Matrix
from sympy.abc import x, y

# 定义测试函数，测试 @threaded 装饰器的功能
def test_threaded():
    # 带有 @threaded 装饰器的函数定义
    @threaded
    def function(expr, *args):
        return 2*expr + sum(args)

    # 测试不同类型的输入参数
    assert function(Matrix([[x, y], [1, x]]), 1, 2) == \
        Matrix([[2*x + 3, 2*y + 3], [5, 2*x + 3]])
    assert function(Eq(x, y), 1, 2) == Eq(2*x + 3, 2*y + 3)
    assert function([x, y], 1, 2) == [2*x + 3, 2*y + 3]
    assert function((x, y), 1, 2) == (2*x + 3, 2*y + 3)
    assert function({x, y}, 1, 2) == {2*x + 3, 2*y + 3}

    # 测试另一个带 @threaded 装饰器的函数定义
    @threaded
    def function(expr, n):
        return expr**n

    assert function(x + y, 2) == x**2 + y**2
    assert function(x, 2) == x**2

# 定义测试函数，测试 @xthreaded 装饰器的功能
def test_xthreaded():
    @xthreaded
    def function(expr, n):
        return expr**n

    assert function(x + y, 2) == (x + y)**2

# 定义测试函数，测试 @wraps 装饰器的功能
def test_wraps():
    # 定义一个简单的函数
    def my_func(x):
        """My function. """

    # 添加一个自定义属性
    my_func.is_my_func = True

    # 使用 @threaded 和 @wraps 装饰器装饰函数
    new_my_func = threaded(my_func)
    new_my_func = wraps(my_func)(new_my_func)

    # 断言装饰后的函数名称、文档字符串、自定义属性等正确
    assert new_my_func.__name__ == 'my_func'
    assert new_my_func.__doc__ == 'My function. '
    assert hasattr(new_my_func, 'is_my_func')
    assert new_my_func.is_my_func is True

# 定义测试函数，测试 @memoize_property 装饰器的功能
def test_memoize_property():
    # 定义一个测试用的类
    class TestMemoize(Basic):
        @memoize_property
        def prop(self):
            return Basic()

    # 创建类的实例，检查 memoize_property 装饰器的缓存效果
    member = TestMemoize()
    obj1 = member.prop
    obj2 = member.prop
    assert obj1 is obj2

# 定义测试函数，测试 @deprecated 装饰器的功能
def test_deprecated():
    # 带有 @deprecated 装饰器的函数定义
    @deprecated('deprecated_function is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    def deprecated_function(x):
        return x

    # 测试函数调用时是否会发出弃用警告
    with warns_deprecated_sympy():
        assert deprecated_function(1) == 1

    # 带有 @deprecated 装饰器的类定义
    @deprecated('deprecated_class is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    class deprecated_class:
        pass

    # 测试类的实例化是否会发出弃用警告
    with warns_deprecated_sympy():
        assert isinstance(deprecated_class(), deprecated_class)

    # 测试带有 @deprecated 装饰器的类，在实例化时返回自身是否正常工作
    @deprecated('deprecated_class_new is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    class deprecated_class_new:
        def __new__(cls, arg):
            return arg

    with warns_deprecated_sympy():
        assert deprecated_class_new(1) == 1

    # 以下代码未完成，被省略
    @deprecated('deprecated_class_init is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    # 定义一个名为 deprecated_class_init 的类，表示一个被弃用的类
    class deprecated_class_init:
        # 初始化方法，接受一个参数 arg，并将其设置为实例属性
        def __init__(self, arg):
            self.arg = 1

    # 使用 warns_deprecated_sympy 上下文管理器捕获被弃用警告
    with warns_deprecated_sympy():
        # 断言：创建 deprecated_class_init 类的实例，并验证其 arg 属性值为 1
        assert deprecated_class_init(1).arg == 1

    # 使用装饰器 @deprecated 对类 deprecated_class_new_init 进行标记为弃用，
    # 提供了相应的弃用信息和版本信息
    @deprecated('deprecated_class_new_init is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    # 定义一个名为 deprecated_class_new_init 的类，表示一个被弃用的类
    class deprecated_class_new_init:
        # 自定义 __new__ 方法，根据参数值 arg 返回相应对象
        def __new__(cls, arg):
            if arg == 0:
                return arg
            return object.__new__(cls)

        # 初始化方法，接受一个参数 arg，并将其设置为实例属性
        def __init__(self, arg):
            self.arg = 1

    # 使用 warns_deprecated_sympy 上下文管理器捕获被弃用警告
    with warns_deprecated_sympy():
        # 断言：创建 deprecated_class_new_init 类的实例，并验证当 arg 为 0 时返回 0
        assert deprecated_class_new_init(0) == 0

    # 使用 warns_deprecated_sympy 上下文管理器捕获被弃用警告
    with warns_deprecated_sympy():
        # 断言：创建 deprecated_class_new_init 类的实例，并验证其 arg 属性值为 1
        assert deprecated_class_new_init(1).arg == 1
```