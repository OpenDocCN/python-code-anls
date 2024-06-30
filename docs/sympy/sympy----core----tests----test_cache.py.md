# `D:\src\scipysrc\sympy\sympy\core\tests\test_cache.py`

```
# 导入系统模块 sys
import sys
# 从 sympy.core.cache 模块中导入 cacheit, cached_property, lazy_function
from sympy.core.cache import cacheit, cached_property, lazy_function
# 从 sympy.testing.pytest 模块中导入 raises 函数
from sympy.testing.pytest import raises

# 定义测试函数 test_cacheit_doc
def test_cacheit_doc():
    # 定义被 cacheit 装饰的测试函数 testfn
    @cacheit
    def testfn():
        "test docstring"  # 函数的文档字符串
        pass  # 占位语句，无具体操作

    # 断言测试函数的文档字符串为 "test docstring"
    assert testfn.__doc__ == "test docstring"
    # 断言测试函数的名称为 "testfn"
    assert testfn.__name__ == "testfn"

# 定义测试函数 test_cacheit_unhashable
def test_cacheit_unhashable():
    # 定义被 cacheit 装饰的测试函数 testit，接受参数 x
    @cacheit
    def testit(x):
        return x  # 返回参数 x 的值

    # 断言调用 testit 函数返回值为 1
    assert testit(1) == 1
    # 再次断言调用 testit 函数返回值为 1，表明对相同参数的结果已缓存
    assert testit(1) == 1
    a = {}  # 创建空字典 a
    # 断言调用 testit 函数返回空字典
    assert testit(a) == {}
    a[1] = 2  # 将键值对添加到字典 a 中
    # 断言调用 testit 函数返回更新后的字典 {1: 2}
    assert testit(a) == {1: 2}

# 定义测试函数 test_cachit_exception
def test_cachit_exception():
    # 确保缓存在函数引发 TypeError 时不会多次调用函数

    a = []  # 创建空列表 a

    # 定义被 cacheit 装饰的测试函数 testf，接受参数 x
    @cacheit
    def testf(x):
        a.append(0)  # 将 0 添加到列表 a
        raise TypeError  # 抛出 TypeError 异常

    # 断言调用 testf(1) 函数引发 TypeError 异常
    raises(TypeError, lambda: testf(1))
    # 断言列表 a 的长度为 1
    assert len(a) == 1

    a.clear()  # 清空列表 a
    # 对不可哈希类型引发 TypeError 异常
    raises(TypeError, lambda: testf([]))
    # 断言列表 a 的长度为 1

    # 定义另一个被 cacheit 装饰的测试函数 testf2，接受参数 x
    @cacheit
    def testf2(x):
        a.append(0)  # 将 0 添加到列表 a
        raise TypeError("Error")  # 抛出带有消息的 TypeError 异常

    a.clear()  # 再次清空列表 a
    raises(TypeError, lambda: testf2(1))  # 断言调用 testf2(1) 函数引发 TypeError 异常
    assert len(a) == 1  # 断言列表 a 的长度为 1

    a.clear()  # 再次清空列表 a
    # 对不可哈希类型引发 TypeError 异常
    raises(TypeError, lambda: testf2([]))
    # 断言列表 a 的长度为 1

# 定义测试函数 test_cached_property
def test_cached_property():
    # 定义类 A
    class A:
        def __init__(self, value):
            self.value = value  # 初始化属性 value
            self.calls = 0  # 初始化属性 calls 为 0

        @cached_property  # 使用 cached_property 装饰器
        def prop(self):
            self.calls = self.calls + 1  # 每次调用增加 calls 计数
            return self.value  # 返回属性 value

    a = A(2)  # 创建类 A 的实例 a，初始化值为 2
    assert a.calls == 0  # 断言 calls 初始为 0
    assert a.prop == 2  # 断言访问 prop 属性返回 2
    assert a.calls == 1  # 断言 calls 累计为 1
    assert a.prop == 2  # 再次断言访问 prop 属性返回 2
    assert a.calls == 1  # calls 仍然为 1
    b = A(None)  # 创建类 A 的另一个实例 b，初始化值为 None
    assert b.prop == None  # 断言访问 prop 属性返回 None

# 定义测试函数 test_lazy_function
def test_lazy_function():
    module_name='xmlrpc.client'  # 模块名称为 'xmlrpc.client'
    function_name = 'gzip_decode'  # 函数名称为 'gzip_decode'
    lazy = lazy_function(module_name, function_name)  # 获取 lazy 函数对象
    assert lazy(b'') == b''  # 断言调用 lazy 函数返回空字节串
    assert module_name in sys.modules  # 断言模块名称在 sys.modules 中
    assert function_name in str(lazy)  # 断言函数名称在 lazy 的字符串表示中
    repr_lazy = repr(lazy)  # 获取 lazy 的 repr 字符串表示
    assert 'LazyFunction' in repr_lazy  # 断言 LazyFunction 出现在 repr_lazy 中
    assert function_name in repr_lazy  # 断言函数名称也在 repr_lazy 中

    lazy = lazy_function('sympy.core.cache', 'cheap')  # 获取另一个 lazy 函数对象
```