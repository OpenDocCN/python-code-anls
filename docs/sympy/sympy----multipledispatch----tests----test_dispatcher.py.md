# `D:\src\scipysrc\sympy\sympy\multipledispatch\tests\test_dispatcher.py`

```
# 导入 sympy.multipledispatch.dispatcher 模块中的几个类和函数
# Dispatcher: 用于多重分派的主类
# MDNotImplementedError: 多重分派中的未实现错误类
# MethodDispatcher: 用于处理方法的多重分派的类
# halt_ordering: 用于停止分派顺序的函数
# restart_ordering: 用于重启分派顺序的函数
# ambiguity_register_error_ignore_dup: 处理重复注册时的歧义注册错误的函数
from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
                                         MethodDispatcher, halt_ordering,
                                         restart_ordering,
                                         ambiguity_register_error_ignore_dup)
# 导入 sympy.testing.pytest 模块中的 raises 和 warns 函数
from sympy.testing.pytest import raises, warns


# 定义一个返回输入值本身的函数
def identity(x):
    return x

# 定义一个将整数加一的函数
def inc(x):
    return x + 1

# 定义一个将整数减一的函数
def dec(x):
    return x - 1


# 定义测试函数 test_dispatcher
def test_dispatcher():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')
    # 向 f 中注册一个接受整数参数并调用 inc 函数的分派规则
    f.add((int,), inc)
    # 向 f 中注册一个接受浮点数参数并调用 dec 函数的分派规则
    f.add((float,), dec)

    # 使用 warns 上下文管理器捕获 DeprecationWarning，测试 stacklevel 设置为 False
    with warns(DeprecationWarning, test_stacklevel=False):
        # 断言当传入 (int,) 时，f 解析到 inc 函数
        assert f.resolve((int,)) == inc
    # 断言当传入整数参数时，f 分派到 inc 函数
    assert f.dispatch(int) is inc

    # 断言 f(1) 返回 2
    assert f(1) == 2
    # 断言 f(1.0) 返回 0.0
    assert f(1.0) == 0.0


# 定义测试函数 test_union_types
def test_union_types():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')
    # 注册一个接受 int 或 float 参数的分派规则，并调用 inc 函数
    f.register((int, float))(inc)

    # 断言 f(1) 返回 2
    assert f(1) == 2
    # 断言 f(1.0) 返回 2.0
    assert f(1.0) == 2.0


# 定义测试函数 test_dispatcher_as_decorator
def test_dispatcher_as_decorator():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 使用装饰器语法注册一个接受整数参数的分派规则，并调用 inc 函数
    @f.register(int)
    def inc(x): # noqa:F811
        return x + 1

    # 使用装饰器语法注册一个接受浮点数参数的分派规则，并调用 dec 函数
    @f.register(float) # noqa:F811
    def inc(x): # noqa:F811
        return x - 1

    # 断言 f(1) 返回 2
    assert f(1) == 2
    # 断言 f(1.0) 返回 0.0
    assert f(1.0) == 0.0


# 定义测试函数 test_register_instance_method
def test_register_instance_method():

    # 定义一个测试类 Test
    class Test:
        # 使用 MethodDispatcher 来定义 __init__ 方法的分派
        __init__ = MethodDispatcher('f')

        # 使用装饰器语法注册一个接受 list 参数的分派规则，并将数据保存在 self.data 中
        @__init__.register(list)
        def _init_list(self, data):
            self.data = data

        # 使用装饰器语法注册一个接受 object 参数的分派规则，并将数据保存在 self.data 中
        @__init__.register(object)
        def _init_obj(self, datum):
            self.data = [datum]

    # 创建 Test 类的两个实例 a 和 b
    a = Test(3)
    b = Test([3])
    # 断言 a.data 等于 b.data，即均为 [3]
    assert a.data == b.data


# 定义测试函数 test_on_ambiguity
def test_on_ambiguity():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 定义一个返回输入值本身的函数 identity
    def identity(x): return x

    # 定义一个列表 ambiguities，用于记录是否发生了歧义
    ambiguities = [False]

    # 定义一个处理歧义的回调函数 on_ambiguity
    def on_ambiguity(dispatcher, amb):
        ambiguities[0] = True

    # 向 f 中注册一个接受两个 object 类型参数的分派规则，并调用 identity 函数，同时指定 on_ambiguity 回调
    f.add((object, object), identity, on_ambiguity=on_ambiguity)
    # 断言 ambiguities[0] 为 False，即未发生歧义
    assert not ambiguities[0]
    # 向 f 中注册一个接受 (object, float) 参数的分派规则，并调用 identity 函数，同时指定 on_ambiguity 回调
    f.add((object, float), identity, on_ambiguity=on_ambiguity)
    # 断言 ambiguities[0] 为 False，仍未发生歧义
    assert not ambiguities[0]
    # 向 f 中注册一个接受 (float, object) 参数的分派规则，并调用 identity 函数，同时指定 on_ambiguity 回调
    f.add((float, object), identity, on_ambiguity=on_ambiguity)
    # 断言 ambiguities[0] 为 True，发生了歧义
    assert ambiguities[0]


# 定义测试函数 test_raise_error_on_non_class
def test_raise_error_on_non_class():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')
    # 断言调用 f.add((1,), inc) 时会抛出 TypeError 异常
    assert raises(TypeError, lambda: f.add((1,), inc))


# 定义测试函数 test_docstring
def test_docstring():

    # 定义三个带有文档字符串的函数
    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x + y

    def three(x, y):
        return x + y

    # 定义一个字符串 master_doc，表示多重方法本身的文档
    master_doc = 'Doc of the multimethod itself'

    # 创建一个名为 f 的 Dispatcher 对象，并指定 master_doc 作为其文档字符串
    f = Dispatcher('f', doc=master_doc)
    # 向 f 中注册一个接受 (object, object) 参数的分派规则，并调用 one 函数
    f.add((object, object), one)
    # 向 f 中注册一个接受 (int, int) 参数的分派规则，并调用 two 函数
    f.add((int, int), two)
    # 向 f 中注册一个接受 (float, float) 参数的分派规则，并调用 three 函数
    f.add((float, float), three)

    # 断言 one 函数的文档字符串在 f 的文档字符串中
    assert one.__doc__.strip() in f.__doc__
    # 断言 two 函数的文档字符串在 f 的文档字符串中
    assert two.__doc__.strip() in f.__doc__
    # 断言 one 函数的文档字符串在 two 函数的文档字符串之前
    assert f.__doc__.find(one.__doc__.strip()) < \
        f.__doc__.find(two.__doc__.strip())
    # 断言 'object, object' 在 f 的文档字符串中
    assert 'object, object' in f.__doc__
    # 断言 master_doc 在 f 的文档字符串中
    assert master_doc in f.__doc__


# 定义
    # 定义字符串变量，用于描述多方法本身的文档说明
    master_doc = 'Doc of the multimethod itself'
    
    # 创建一个名为 'f' 的分发器对象，指定其文档为 master_doc
    f = Dispatcher('f', doc=master_doc)
    
    # 向分发器对象 'f' 中添加方法：
    # - 当参数类型为 (object, object) 时，调用函数 one
    f.add((object, object), one)
    # - 当参数类型为 (int, int) 时，调用函数 two
    f.add((int, int), two)
    # - 当参数类型为 (float, float) 时，调用函数 three
    f.add((float, float), three)
    
    # 使用断言检验，当给定参数 (1, 1) 时，调用分发器对象 'f' 的 _help 方法应返回函数 two 的文档
    assert f._help(1, 1) == two.__doc__
    # 使用断言检验，当给定参数 (1.0, 2.0) 时，调用分发器对象 'f' 的 _help 方法应返回函数 three 的文档
    assert f._help(1.0, 2.0) == three.__doc__
def test_source():
    # 定义函数 one，接受两个参数 x 和 y，返回它们的和
    def one(x, y):
        """ Docstring number one """
        return x + y

    # 定义函数 two，接受两个参数 x 和 y，返回 x 减去 y 的结果
    def two(x, y):
        """ Docstring number two """
        return x - y

    # 设置主文档字符串，描述多方法本身
    master_doc = 'Doc of the multimethod itself'

    # 创建名为 f 的 Dispatcher 对象，传入文档字符串作为参数
    f = Dispatcher('f', doc=master_doc)
    # 将 (int, int) 类型的参数组合与函数 one 绑定，添加到 Dispatcher 对象 f 中
    f.add((int, int), one)
    # 将 (float, float) 类型的参数组合与函数 two 绑定，添加到 Dispatcher 对象 f 中
    f.add((float, float), two)

    # 断言 'x + y' 存在于调用 f._source(1, 1) 的结果中
    assert 'x + y' in f._source(1, 1)
    # 断言 'x - y' 存在于调用 f._source(1.0, 1.0) 的结果中
    assert 'x - y' in f._source(1.0, 1.0)


def test_source_raises_on_missing_function():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 断言调用 f.source(1) 会引发 TypeError 异常
    assert raises(TypeError, lambda: f.source(1))


def test_halt_method_resolution():
    # 创建一个列表 g，其包含一个整数元素 0
    g = [0]

    # 定义函数 on_ambiguity，接受两个参数 a 和 b，在函数内部将列表 g 的第一个元素加 1
    def on_ambiguity(a, b):
        g[0] += 1

    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 调用 halt_ordering 函数，暂停方法解析
    halt_ordering()

    # 定义一个名为 func 的函数，接受任意数量的参数，但函数体为空
    def func(*args):
        pass

    # 将 (int, object) 类型的参数组合与函数 func 绑定，添加到 Dispatcher 对象 f 中
    f.add((int, object), func)
    # 将 (object, int) 类型的参数组合与函数 func 绑定，添加到 Dispatcher 对象 f 中
    f.add((object, int), func)

    # 断言列表 g 的值为 [0]
    assert g == [0]

    # 恢复方法解析，传入 on_ambiguity 函数作为参数
    restart_ordering(on_ambiguity=on_ambiguity)

    # 断言列表 g 的值为 [1]
    assert g == [1]

    # 断言 Dispatcher 对象 f 的排序集合为 {(int, object), (object, int)}
    assert set(f.ordering) == {(int, object), (object, int)}


def test_no_implementations():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 断言调用 f('hello') 会引发 NotImplementedError 异常
    assert raises(NotImplementedError, lambda: f('hello'))


def test_register_stacking():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 使用装饰器语法将函数 rev 注册为接受 list 类型参数的处理函数，并将其添加到 Dispatcher 对象 f 中
    @f.register(list)
    # 使用装饰器语法将函数 rev 注册为接受 tuple 类型参数的处理函数，并将其添加到 Dispatcher 对象 f 中
    @f.register(tuple)
    def rev(x):
        return x[::-1]

    # 断言调用 f((1, 2, 3)) 的结果为 (3, 2, 1)
    assert f((1, 2, 3)) == (3, 2, 1)
    # 断言调用 f([1, 2, 3]) 的结果为 [3, 2, 1]
    assert f([1, 2, 3]) == [3, 2, 1]

    # 断言调用 f('hello') 会引发 NotImplementedError 异常
    assert raises(NotImplementedError, lambda: f('hello'))
    # 断言调用 rev('hello') 的结果为 'olleh'
    assert rev('hello') == 'olleh'


def test_dispatch_method():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 使用装饰器语法将函数 rev 注册为接受 list 类型参数的处理函数，并将其添加到 Dispatcher 对象 f 中
    @f.register(list)
    def rev(x):
        return x[::-1]

    # 使用装饰器语法将函数 add 注册为接受两个 int 类型参数的处理函数，并将其添加到 Dispatcher 对象 f 中
    @f.register(int, int)
    def add(x, y):
        return x + y

    # 定义一个名为 MyList 的类，继承自 list 类
    class MyList(list):
        pass

    # 断言调用 f.dispatch(list) 的结果为 rev 函数
    assert f.dispatch(list) is rev
    # 断言调用 f.dispatch(MyList) 的结果为 rev 函数
    assert f.dispatch(MyList) is rev
    # 断言调用 f.dispatch(int, int) 的结果为 add 函数
    assert f.dispatch(int, int) is add


def test_not_implemented():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 使用装饰器语法将函数 _ 注册为接受任意对象参数的处理函数，并将其添加到 Dispatcher 对象 f 中
    @f.register(object)
    def _(x):
        return 'default'

    # 使用装饰器语法将函数 _ 注册为接受 int 类型参数的处理函数，并将其添加到 Dispatcher 对象 f 中
    @f.register(int)
    def _(x):
        if x % 2 == 0:
            return 'even'
        else:
            raise MDNotImplementedError()

    # 断言调用 f('hello') 的结果为 'default'
    assert f('hello') == 'default'  # default behavior
    # 断言调用 f(2) 的结果为 'even'
    assert f(2) == 'even'          # specialized behavior
    # 断言调用 f(3) 的结果为 'default'
    assert f(3) == 'default'       # fall back to default behavior
    # 断言调用 f(1, 2) 会引发 NotImplementedError 异常
    assert raises(NotImplementedError, lambda: f(1, 2))


def test_not_implemented_error():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 使用装饰器语法将函数 _ 注册为接受 float 类型参数的处理函数，并将其添加到 Dispatcher 对象 f 中
    @f.register(float)
    def _(a):
        raise MDNotImplementedError()

    # 断言调用 f(1.0) 会引发 NotImplementedError 异常
    assert raises(NotImplementedError, lambda: f(1.0))


def test_ambiguity_register_error_ignore_dup():
    # 创建一个名为 f 的 Dispatcher 对象
    f = Dispatcher('f')

    # 定义类 A
    class A:
        pass
    # 定义类 B，继承自 A
    class B(A):
        pass
    # 定义类 C，继承自 A
    class C(A):
        pass

    # 使用 ambiguity_register_error_ignore_dup 标志，将 (A, B) 类型的参数组合与匿名函数 lambda x, y: None 绑定，并将其添加到 Dispatcher 对象 f 中
    f.add((A, B), lambda x, y: None, ambiguity_register_error_ignore_dup)
    # 使用 ambiguity_register_error_ignore_dup 标志，将 (B, A) 类型的参数组合与匿名函数 lambda x, y: None 绑定，并将其添加到 Dispatcher 对象 f 中
    f.add((B, A), lambda x, y: None, ambiguity_register_error_ignore_dup)
    # 使用 ambiguity_register_error_ignore_dup 标志，将 (A, C) 类型的参数组合与匿名函数 lambda x, y: None 绑定
```