# `.\numpy\numpy\_core\tests\test_overrides.py`

```
# 导入inspect模块，用于获取对象的信息
import inspect
# 导入sys模块，提供与解释器交互的功能
import sys
# 导入os模块，提供与操作系统交互的功能
import os
# 导入tempfile模块，用于创建临时文件和目录
import tempfile
# 从io模块导入StringIO类，用于在内存中读写str对象
from io import StringIO
# 导入unittest模块中的mock类，用于模拟对象行为
from unittest import mock
# 导入pickle模块，用于序列化和反序列化Python对象

import pickle

# 导入pytest模块，用于编写和运行测试
import pytest

# 导入numpy模块，并将其重命名为np，提供对数组和数学函数的支持
import numpy as np

# 从numpy.testing模块导入断言函数，用于测试numpy数组的期望行为
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_raises_regex)

# 从numpy._core.overrides模块导入特定函数和类
from numpy._core.overrides import (
    _get_implementing_args, array_function_dispatch,
    verify_matching_signatures)


# 定义一个函数，在调用数组函数时返回NotImplemented
def _return_not_implemented(self, *args, **kwargs):
    return NotImplemented


# 使用array_function_dispatch装饰器注册一个函数，指定一个参数的数组函数分发
@array_function_dispatch(lambda array: (array,))
def dispatched_one_arg(array):
    """Docstring."""
    return 'original'


# 使用array_function_dispatch装饰器注册一个函数，指定两个参数的数组函数分发
@array_function_dispatch(lambda array1, array2: (array1, array2))
def dispatched_two_arg(array1, array2):
    """Docstring."""
    return 'original'


# 测试类TestGetImplementingArgs，用于测试_get_implementing_args函数
class TestGetImplementingArgs:

    # 测试_ndarray方法，用于测试数组类型的行为
    def test_ndarray(self):
        array = np.array(1)

        # 调用_get_implementing_args函数，检查返回的参数是否与数组一致
        args = _get_implementing_args([array])
        assert_equal(list(args), [array])

        # 调用_get_implementing_args函数，检查返回的参数是否与数组一致
        args = _get_implementing_args([array, array])
        assert_equal(list(args), [array])

        # 调用_get_implementing_args函数，检查返回的参数是否与数组一致
        args = _get_implementing_args([array, 1])
        assert_equal(list(args), [array])

        # 调用_get_implementing_args函数，检查返回的参数是否与数组一致
        args = _get_implementing_args([1, array])
        assert_equal(list(args), [array])

    # 测试_ndarray_subclasses方法，用于测试数组的子类行为
    def test_ndarray_subclasses(self):

        # 定义一个继承自np.ndarray的类OverrideSub，覆盖__array_function__方法
        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        # 定义一个继承自np.ndarray的类NoOverrideSub，未覆盖__array_function__方法
        class NoOverrideSub(np.ndarray):
            pass

        array = np.array(1).view(np.ndarray)
        override_sub = np.array(1).view(OverrideSub)
        no_override_sub = np.array(1).view(NoOverrideSub)

        # 调用_get_implementing_args函数，检查返回的参数是否按预期排列
        args = _get_implementing_args([array, override_sub])
        assert_equal(list(args), [override_sub, array])

        # 调用_get_implementing_args函数，检查返回的参数是否按预期排列
        args = _get_implementing_args([array, no_override_sub])
        assert_equal(list(args), [no_override_sub, array])

        # 调用_get_implementing_args函数，检查返回的参数是否按预期排列
        args = _get_implementing_args([override_sub, no_override_sub])
        assert_equal(list(args), [override_sub, no_override_sub])

    # 测试_ndarray_and_duck_array方法，用于测试数组和鸭子类型对象的行为
    def test_ndarray_and_duck_array(self):

        # 定义一个类Other，覆盖__array_function__方法
        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        other = Other()

        # 调用_get_implementing_args函数，检查返回的参数是否按预期排列
        args = _get_implementing_args([other, array])
        assert_equal(list(args), [other, array])

        # 调用_get_implementing_args函数，检查返回的参数是否按预期排列
        args = _get_implementing_args([array, other])
        assert_equal(list(args), [array, other])

    # 测试_ndarray_subclass_and_duck_array方法，用于测试数组子类和鸭子类型对象的行为
    def test_ndarray_subclass_and_duck_array(self):

        # 定义一个继承自np.ndarray的类OverrideSub，覆盖__array_function__方法
        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        # 定义一个类Other，覆盖__array_function__方法
        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        subarray = np.array(1).view(OverrideSub)
        other = Other()

        # 调用_get_implementing_args函数，检查返回的参数是否按预期排列
        assert_equal(_get_implementing_args([array, subarray, other]),
                     [subarray, array, other])
        # 调用_get_implementing_args函数，检查返回的参数是否按预期排列
        assert_equal(_get_implementing_args([array, other, subarray]),
                     [subarray, array, other])
    # 定义一个测试方法，用于测试多个“鸭子类型”数组的情况
    def test_many_duck_arrays(self):

        # 定义类 A，设置 __array_function__ 属性为 _return_not_implemented 函数
        class A:
            __array_function__ = _return_not_implemented

        # 定义类 B，继承自 A，设置 __array_function__ 属性为 _return_not_implemented 函数
        class B(A):
            __array_function__ = _return_not_implemented

        # 定义类 C，继承自 A，设置 __array_function__ 属性为 _return_not_implemented 函数
        class C(A):
            __array_function__ = _return_not_implemented

        # 定义类 D，没有继承自 A，设置 __array_function__ 属性为 _return_not_implemented 函数
        class D:
            __array_function__ = _return_not_implemented

        # 创建类 A、B、C 和 D 的实例
        a = A()
        b = B()
        c = C()
        d = D()

        # 使用 _get_implementing_args 函数测试不同参数组合下的返回结果是否符合预期
        assert_equal(_get_implementing_args([1]), [])  # 测试传入整数 1 的情况
        assert_equal(_get_implementing_args([a]), [a])  # 测试传入实例 a 的情况
        assert_equal(_get_implementing_args([a, 1]), [a])  # 测试传入实例 a 和整数 1 的情况
        assert_equal(_get_implementing_args([a, a, a]), [a])  # 测试传入多个实例 a 的情况
        assert_equal(_get_implementing_args([a, d, a]), [a, d])  # 测试传入实例 a、d 和 a 的情况
        assert_equal(_get_implementing_args([a, b]), [b, a])  # 测试传入实例 a 和 b 的情况
        assert_equal(_get_implementing_args([b, a]), [b, a])  # 测试传入实例 b 和 a 的情况
        assert_equal(_get_implementing_args([a, b, c]), [b, c, a])  # 测试传入实例 a、b 和 c 的情况
        assert_equal(_get_implementing_args([a, c, b]), [c, b, a])  # 测试传入实例 a、c 和 b 的情况

    # 定义一个测试方法，用于测试过多“鸭子类型”数组的情况
    def test_too_many_duck_arrays(self):
        # 创建一个命名空间字典，其中 __array_function__ 属性为 _return_not_implemented 函数
        namespace = dict(__array_function__=_return_not_implemented)
        # 创建 65 个动态生成的类，每个类都有相同的命名空间
        types = [type('A' + str(i), (object,), namespace) for i in range(65)]
        # 创建一个包含 65 个实例的列表
        relevant_args = [t() for t in types]

        # 使用 _get_implementing_args 函数测试前 64 个实例是否返回预期的结果
        actual = _get_implementing_args(relevant_args[:64])
        assert_equal(actual, relevant_args[:64])

        # 使用 assert_raises_regex 断言捕获 TypeError 异常，检查当传入超过 64 个不同类型实例时是否引发异常
        with assert_raises_regex(TypeError, 'distinct argument types'):
            _get_implementing_args(relevant_args)
class TestNDArrayArrayFunction:
    
    def test_method(self):
        
        class Other:
            __array_function__ = _return_not_implemented  # 定义一个没有实现的特殊方法 __array_function__

        class NoOverrideSub(np.ndarray):  # 定义一个继承自 np.ndarray 的子类，没有覆盖 __array_function__
            pass
        
        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented  # 定义一个继承自 np.ndarray 的子类，并覆盖 __array_function__

        array = np.array([1])  # 创建一个包含单个元素的 numpy 数组
        other = Other()  # 创建 Other 类的实例
        no_override_sub = array.view(NoOverrideSub)  # 将 array 视图转换为 NoOverrideSub 类的实例
        override_sub = array.view(OverrideSub)  # 将 array 视图转换为 OverrideSub 类的实例

        # 调用 array 对象的 __array_function__ 方法，传入函数、类型元组、参数和关键字参数，期望返回 'original'
        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray,),
                                          args=(array, 1.), kwargs={})
        assert_equal(result, 'original')  # 断言结果为 'original'

        # 调用 array 对象的 __array_function__ 方法，传入函数、类型元组、参数和关键字参数，期望返回 NotImplemented
        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, Other),
                                          args=(array, other), kwargs={})
        assert_(result is NotImplemented)  # 断言结果为 NotImplemented

        # 调用 array 对象的 __array_function__ 方法，传入函数、类型元组、参数和关键字参数，期望返回 'original'
        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, NoOverrideSub),
                                          args=(array, no_override_sub),
                                          kwargs={})
        assert_equal(result, 'original')  # 断言结果为 'original'

        # 调用 array 对象的 __array_function__ 方法，传入函数、类型元组、参数和关键字参数，期望返回 'original'
        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, OverrideSub),
                                          args=(array, override_sub),
                                          kwargs={})
        assert_equal(result, 'original')  # 断言结果为 'original'

        # 使用 assert_raises_regex 上下文管理器断言 TypeError，错误信息中包含 'no implementation found'
        with assert_raises_regex(TypeError, 'no implementation found'):
            np.concatenate((array, other))  # 尝试使用 np.concatenate() 合并 array 和 other

        expected = np.concatenate((array, array))  # 预期的合并结果，两个相同的 array 数组
        result = np.concatenate((array, no_override_sub))  # 使用 np.concatenate() 合并 array 和 no_override_sub
        assert_equal(result, expected.view(NoOverrideSub))  # 断言结果与预期的 NoOverrideSub 类型的视图相等
        result = np.concatenate((array, override_sub))  # 使用 np.concatenate() 合并 array 和 override_sub
        assert_equal(result, expected.view(OverrideSub))  # 断言结果与预期的 OverrideSub 类型的视图相等

    def test_no_wrapper(self):
        # 这段代码不应该执行，除非用户有意调用具有无效参数的 __array_function__ 方法，
        # 但我们仍然检查确保能够正确引发适当的错误。
        array = np.array(1)  # 创建一个包含单个元素的 numpy 数组
        func = lambda x: x  # 定义一个简单的 lambda 函数
        with assert_raises_regex(AttributeError, '_implementation'):
            array.__array_function__(func=func, types=(np.ndarray,),
                                     args=(array,), kwargs={})  # 尝试调用 array 对象的 __array_function__ 方法
    def test_interface(self):
        # 定义一个内部类 MyArray，用于测试 __array_function__ 方法的调用
        class MyArray:
            # 自定义 __array_function__ 方法，返回传入的参数元组
            def __array_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)

        # 创建 MyArray 的实例 original
        original = MyArray()
        # 调用 dispatched_one_arg 函数，并解构返回的元组
        (obj, func, types, args, kwargs) = dispatched_one_arg(original)
        # 断言 original 和 obj 是同一个对象
        assert_(obj is original)
        # 断言 func 是 dispatched_one_arg 函数本身
        assert_(func is dispatched_one_arg)
        # 断言 types 是包含 MyArray 类型的集合
        assert_equal(set(types), {MyArray})
        # args 应包含 original 这个实例
        # assert_equal 内部使用了重载的 np.iscomplexobj() 方法
        assert_(args == (original,))
        # kwargs 应该为空字典
        assert_equal(kwargs, {})

    def test_not_implemented(self):
        # 定义一个内部类 MyArray，覆盖 __array_function__ 方法返回 NotImplemented
        class MyArray:
            def __array_function__(self, func, types, args, kwargs):
                return NotImplemented

        # 创建 MyArray 的实例 array
        array = MyArray()
        # 使用 assert_raises_regex 断言抛出 TypeError 异常，且异常信息包含 'no implementation found'
        with assert_raises_regex(TypeError, 'no implementation found'):
            # 调用 dispatched_one_arg 函数
            dispatched_one_arg(array)

    def test_where_dispatch(self):
        # 定义一个内部类 DuckArray，覆盖 __array_function__ 方法返回固定字符串 "overridden"
        class DuckArray:
            def __array_function__(self, ufunc, method, *inputs, **kwargs):
                return "overridden"

        # 创建一个 numpy 数组 array
        array = np.array(1)
        # 创建 DuckArray 的实例 duck_array
        duck_array = DuckArray()

        # 调用 numpy 的 std 函数，传入 where 参数为 duck_array
        result = np.std(array, where=duck_array)

        # 断言 result 等于 "overridden"
        assert_equal(result, "overridden")
class TestVerifyMatchingSignatures:

    def test_verify_matching_signatures(self):

        # 调用 verify_matching_signatures 函数，传入两个相同的匿名函数，预期不会引发异常
        verify_matching_signatures(lambda x: 0, lambda x: 0)
        verify_matching_signatures(lambda x=None: 0, lambda x=None: 0)
        verify_matching_signatures(lambda x=1: 0, lambda x=None: 0)

        # 使用 assert_raises 断言预期运行时错误将被引发
        with assert_raises(RuntimeError):
            # 传入两个参数名称不同的匿名函数，预期引发 RuntimeError
            verify_matching_signatures(lambda a: 0, lambda b: 0)
        with assert_raises(RuntimeError):
            # 传入一个参数有默认值，另一个参数无默认值的匿名函数，预期引发 RuntimeError
            verify_matching_signatures(lambda x: 0, lambda x=None: 0)
        with assert_raises(RuntimeError):
            # 传入两个参数名称相同但默认值类型不同的匿名函数，预期引发 RuntimeError
            verify_matching_signatures(lambda x=None: 0, lambda y=None: 0)
        with assert_raises(RuntimeError):
            # 传入两个参数名称相同且默认值类型相同但值不同的匿名函数，预期引发 RuntimeError
            verify_matching_signatures(lambda x=1: 0, lambda y=1: 0)

    def test_array_function_dispatch(self):

        # 使用 assert_raises 断言预期运行时错误将被引发
        with assert_raises(RuntimeError):
            # 带有 array_function_dispatch 装饰器的函数，传入的参数类型与声明的不匹配，预期引发 RuntimeError
            @array_function_dispatch(lambda x: (x,))
            def f(y):
                pass

        # 不应该引发异常
        @array_function_dispatch(lambda x: (x,), verify=False)
        def f(y):
            pass


def _new_duck_type_and_implements():
    """创建一个鸭子类型的数组和实现函数。"""
    HANDLED_FUNCTIONS = {}

    class MyArray:
        def __array_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all(issubclass(t, MyArray) for t in types):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(numpy_function):
        """注册一个 __array_function__ 实现函数。"""
        def decorator(func):
            HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator

    return (MyArray, implements)


class TestArrayFunctionImplementation:

    def test_one_arg(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(dispatched_one_arg)
        def _(array):
            return 'myarray'

        # 断言 dispatch_one_arg 函数对不同类型参数的行为
        assert_equal(dispatched_one_arg(1), 'original')
        assert_equal(dispatched_one_arg(MyArray()), 'myarray')
    def test_optional_args(self):
        # 获取 MyArray 类型和其实现的 duck-typing 函数
        MyArray, implements = _new_duck_type_and_implements()

        # 定义带可选参数的函数 func_with_option，通过装饰器进行数组函数分派
        @array_function_dispatch(lambda array, option=None: (array,))
        def func_with_option(array, option='default'):
            return option

        # 将 func_with_option 函数实现为 my_array_func_with_option 函数
        @implements(func_with_option)
        def my_array_func_with_option(array, new_option='myarray'):
            return new_option

        # 断言 func_with_option 函数的不同调用方式返回预期结果
        assert_equal(func_with_option(1), 'default')
        assert_equal(func_with_option(1, option='extra'), 'extra')
        assert_equal(func_with_option(MyArray()), 'myarray')
        # 使用未实现的选项调用 func_with_option 应抛出 TypeError 异常
        with assert_raises(TypeError):
            func_with_option(MyArray(), option='extra')

        # 断言 my_array_func_with_option 函数的调用结果
        result = my_array_func_with_option(MyArray(), new_option='yes')
        assert_equal(result, 'yes')
        # 使用未实现的选项调用 func_with_option 应抛出 TypeError 异常
        with assert_raises(TypeError):
            func_with_option(MyArray(), new_option='no')

    def test_not_implemented(self):
        # 获取 MyArray 类型和其实现的 duck-typing 函数
        MyArray, implements = _new_duck_type_and_implements()

        # 定义函数 func，通过装饰器指定数组函数分派的模块为 'my'
        @array_function_dispatch(lambda array: (array,), module='my')
        def func(array):
            return array

        # 创建一个数组 array
        array = np.array(1)
        # 断言 func 函数返回的是输入的数组 array
        assert_(func(array) is array)
        # 断言 func 函数的模块名为 'my'
        assert_equal(func.__module__, 'my')

        # 使用未实现的 MyArray 类型调用 func 应抛出 TypeError 异常
        with assert_raises_regex(
                TypeError, "no implementation found for 'my.func'"):
            func(MyArray())

    @pytest.mark.parametrize("name", ["concatenate", "mean", "asarray"])
    def test_signature_error_message_simple(self, name):
        # 根据参数 name 获取 numpy 模块中对应的函数 func
        func = getattr(np, name)
        try:
            # 调用 func 函数，期望抛出 TypeError 异常
            func()
        except TypeError as e:
            exc = e

        # 断言异常信息以 name + '()' 开头
        assert exc.args[0].startswith(f"{name}()")

    def test_signature_error_message(self):
        # 定义一个分派函数 _dispatcher，返回空元组
        # func 函数的分派器使用 _dispatcher
        def _dispatcher():
            return ()

        # 定义函数 func，并使用 _dispatcher 作为其数组函数分派器
        @array_function_dispatch(_dispatcher)
        def func():
            pass

        try:
            # 调用 func 的 _implementation 方法，传入错误的参数名 bad_arg
            func._implementation(bad_arg=3)
        except TypeError as e:
            expected_exception = e

        try:
            # 调用 func 函数，传入错误的参数名 bad_arg
            func(bad_arg=3)
            raise AssertionError("must fail")
        except TypeError as exc:
            # 如果异常信息以 '_dispatcher' 开头，跳过此测试（Python 版本不使用 __qualname__ 格式化 TypeError）
            if exc.args[0].startswith("_dispatcher"):
                pytest.skip("Python version is not using __qualname__ for "
                            "TypeError formatting.")

            # 断言异常信息与预期异常相同
            assert exc.args == expected_exception.args

    @pytest.mark.parametrize("value", [234, "this func is not replaced"])
    def test_dispatcher_error(self, value):
        # 如果调度程序引发错误，则不应尝试进行变异
        # 创建一个 TypeError 的实例，用于后续引发异常
        error = TypeError(value)

        # 定义一个内部函数 dispatcher，用于引发之前创建的错误
        def dispatcher():
            raise error

        # 使用 array_function_dispatch 装饰器将 func 关联到 dispatcher 函数
        @array_function_dispatch(dispatcher)
        def func():
            return 3

        try:
            # 调用 func 函数，期望它引发之前定义的 TypeError 异常
            func()
            # 如果没有引发异常，则抛出 AssertionError
            raise AssertionError("must fail")
        except TypeError as exc:
            # 断言捕获的异常对象和之前创建的 error 对象相同
            assert exc is error  # unmodified exception

    def test_properties(self):
        # 检查 str 和 repr 方法的行为是否合理
        # 获取 dispatched_two_arg 函数的引用
        func = dispatched_two_arg
        # 断言函数的字符串表示应该与其实现对象的字符串表示相同
        assert str(func) == str(func._implementation)
        # 获取函数的 repr，去除其中的对象地址信息
        repr_no_id = repr(func).split("at ")[0]
        repr_no_id_impl = repr(func._implementation).split("at ")[0]
        # 断言函数的 repr 和其实现对象的 repr 应该相同（去除对象地址信息后）
        assert repr_no_id == repr_no_id_impl

    @pytest.mark.parametrize("func", [
            lambda x, y: 0,  # 没有 like 参数
            lambda like=None: 0,  # 不是仅关键字参数
            lambda *, like=None, a=3: 0,  # 不是最后一个参数（尽管不影响测试）
        ])
    def test_bad_like_sig(self, func):
        # 我们对函数签名进行合理性检查，这些应该失败
        # 使用 pytest.raises 检查 array_function_dispatch 装饰器对于 func 的应用是否引发 RuntimeError
        with pytest.raises(RuntimeError):
            array_function_dispatch()(func)

    def test_bad_like_passing(self):
        # 测试通过位置参数将 like 传递给装饰函数的内部一致性检查
        # 定义一个函数 func，接受一个关键字参数 like
        def func(*, like=None):
            pass

        # 使用 array_function_dispatch 装饰 func 函数
        func_with_like = array_function_dispatch()(func)
        # 使用 pytest.raises 检查调用 func_with_like 函数时传递 like 参数是否引发 TypeError
        with pytest.raises(TypeError):
            func_with_like()
        with pytest.raises(TypeError):
            func_with_like(like=234)

    def test_too_many_args(self):
        # 主要是为了增加代码覆盖率的单元测试
        # 创建一个包含多个 MyArr 类的实例对象列表
        objs = []
        for i in range(80):
            class MyArr:
                def __array_function__(self, *args, **kwargs):
                    return NotImplemented

            objs.append(MyArr())

        # 定义一个 _dispatch 函数，用于作为 array_function_dispatch 的参数
        def _dispatch(*args):
            return args

        # 使用 array_function_dispatch 装饰 func 函数
        @array_function_dispatch(_dispatch)
        def func(*args):
            pass

        # 使用 pytest.raises 检查调用 func 函数时传递太多参数是否引发 TypeError，并匹配指定的错误信息
        with pytest.raises(TypeError, match="maximum number"):
            func(*objs)
class TestNDArrayMethods:

    def test_repr(self):
        # 用于测试：即使 __array_function__ 没有实现 np.array_repr()，
        # 这个方法仍应该被定义
        class MyArray(np.ndarray):
            def __array_function__(*args, **kwargs):
                return NotImplemented

        # 创建一个视图为 MyArray 的数组对象
        array = np.array(1).view(MyArray)
        # 断言该数组的 repr 应为 'MyArray(1)'
        assert_equal(repr(array), 'MyArray(1)')
        # 断言该数组的 str 应为 '1'
        assert_equal(str(array), '1')


class TestNumPyFunctions:

    def test_set_module(self):
        # 断言 np.sum 函数的模块应为 'numpy'
        assert_equal(np.sum.__module__, 'numpy')
        # 断言 np.char.equal 函数的模块应为 'numpy.char'
        assert_equal(np.char.equal.__module__, 'numpy.char')
        # 断言 np.fft.fft 函数的模块应为 'numpy.fft'
        assert_equal(np.fft.fft.__module__, 'numpy.fft')
        # 断言 np.linalg.solve 函数的模块应为 'numpy.linalg'
        assert_equal(np.linalg.solve.__module__, 'numpy.linalg')

    def test_inspect_sum(self):
        # 获取 np.sum 函数的签名
        signature = inspect.signature(np.sum)
        # 断言签名中包含 'axis' 参数
        assert_('axis' in signature.parameters)

    def test_override_sum(self):
        # 创建一个自定义数组类 MyArray 和其对应的接口实现
        MyArray, implements = _new_duck_type_and_implements()

        # 使用装饰器实现对 np.sum 的重载
        @implements(np.sum)
        def _(array):
            return 'yes'

        # 断言对 MyArray 类型的数组调用 np.sum 应返回 'yes'
        assert_equal(np.sum(MyArray()), 'yes')

    def test_sum_on_mock_array(self):
        # 由于 __array_function__ 只在类字典中查找，因此需要为模拟对象创建一个代理
        class ArrayProxy:
            def __init__(self, value):
                self.value = value

            # 重定向 __array_function__ 至实际值对象
            def __array_function__(self, *args, **kwargs):
                return self.value.__array_function__(*args, **kwargs)

            # 重定向 __array__ 至实际值对象
            def __array__(self, *args, **kwargs):
                return self.value.__array__(*args, **kwargs)

        # 创建一个使用 ArrayProxy 的模拟对象
        proxy = ArrayProxy(mock.Mock(spec=ArrayProxy))
        # 设置模拟对象的 __array_function__ 返回值为 1
        proxy.value.__array_function__.return_value = 1
        # 调用 np.sum(proxy)，期望结果为 1
        result = np.sum(proxy)
        assert_equal(result, 1)
        # 断言 np.sum 调用时传入的参数符合预期
        proxy.value.__array_function__.assert_called_once_with(
            np.sum, (ArrayProxy,), (proxy,), {})
        # 断言未调用 proxy.value.__array__ 方法
        proxy.value.__array__.assert_not_called()

    def test_sum_forwarding_implementation(self):
        # 创建一个继承自 np.ndarray 的自定义数组类 MyArray
        class MyArray(np.ndarray):

            # 自定义 sum 方法
            def sum(self, axis, out):
                return 'summed'

            # 重定向 __array_function__ 至父类实现
            def __array_function__(self, func, types, args, kwargs):
                return super().__array_function__(func, types, args, kwargs)

        # 创建一个视图为 MyArray 的数组对象
        array = np.array(1).view(MyArray)
        # 断言 np.sum(array) 的结果为 'summed'
        assert_equal(np.sum(array), 'summed')


class TestArrayLike:
    def setup_method(self):
        # 定义内部类 MyArray
        class MyArray():
            # MyArray 类的初始化函数
            def __init__(self, function=None):
                self.function = function

            # 定义 __array_function__ 方法，用于处理数组函数
            def __array_function__(self, func, types, args, kwargs):
                # 确保 func 是 numpy 中的函数
                assert func is getattr(np, func.__name__)
                try:
                    # 获取 MyArray 实例中与 func 同名的方法
                    my_func = getattr(self, func.__name__)
                except AttributeError:
                    # 如果找不到同名方法，则返回 NotImplemented
                    return NotImplemented
                # 调用找到的方法并返回结果
                return my_func(*args, **kwargs)

        # 将 MyArray 类赋值给 self.MyArray
        self.MyArray = MyArray

        # 定义内部类 MyNoArrayFunctionArray
        class MyNoArrayFunctionArray():
            # MyNoArrayFunctionArray 类的初始化函数
            def __init__(self, function=None):
                self.function = function

        # 将 MyNoArrayFunctionArray 类赋值给 self.MyNoArrayFunctionArray
        self.MyNoArrayFunctionArray = MyNoArrayFunctionArray

    def add_method(self, name, arr_class, enable_value_error=False):
        # 定义内部函数 _definition，用于向 arr_class 动态添加方法
        def _definition(*args, **kwargs):
            # 检查 kwargs 中是否包含 'like' 键
            assert 'like' not in kwargs

            # 如果 enable_value_error 为 True 并且 kwargs 中包含 'value_error' 键，则抛出 ValueError
            if enable_value_error and 'value_error' in kwargs:
                raise ValueError

            # 调用 arr_class 中 name 方法，并返回结果
            return arr_class(getattr(arr_class, name))

        # 将 _definition 方法添加为 arr_class 的 name 方法
        setattr(arr_class, name, _definition)

    # 定义函数 func_args，返回传入的 args 和 kwargs
    def func_args(*args, **kwargs):
        return args, kwargs

    # 定义测试方法 test_array_like_not_implemented
    def test_array_like_not_implemented(self):
        # 向 self.MyArray 添加 'array' 方法
        self.add_method('array', self.MyArray)

        # 创建一个 self.MyArray 的实例 ref
        ref = self.MyArray.array()

        # 使用 assert_raises_regex 断言捕获 TypeError 异常，并检查异常消息中是否包含 'no implementation found'
        with assert_raises_regex(TypeError, 'no implementation found'):
            # 调用 np.asarray 函数，使用 ref 作为参数，但不应包含 'like' 参数
            array_like = np.asarray(1, like=ref)

    # 定义一个测试用例参数列表 _array_tests
    _array_tests = [
        ('array', *func_args((1,))),
        ('asarray', *func_args((1,))),
        ('asanyarray', *func_args((1,))),
        ('ascontiguousarray', *func_args((2, 3))),
        ('asfortranarray', *func_args((2, 3))),
        ('require', *func_args((np.arange(6).reshape(2, 3),),
                               requirements=['A', 'F'])),
        ('empty', *func_args((1,))),
        ('full', *func_args((1,), 2)),
        ('ones', *func_args((1,))),
        ('zeros', *func_args((1,))),
        ('arange', *func_args(3)),
        ('frombuffer', *func_args(b'\x00' * 8, dtype=int)),
        ('fromiter', *func_args(range(3), dtype=int)),
        ('fromstring', *func_args('1,2', dtype=int, sep=',')),
        ('loadtxt', *func_args(lambda: StringIO('0 1\n2 3'))),
        ('genfromtxt', *func_args(lambda: StringIO('1,2.1'),
                                  dtype=[('int', 'i8'), ('float', 'f8')],
                                  delimiter=',')),
    ]

    # 使用 pytest.mark.parametrize 装饰器为 test_array_like_not_implemented 方法参数化
    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('numpy_ref', [True, False])
    # 定义一个测试方法，用于测试类似数组的行为
    def test_array_like(self, function, args, kwargs, numpy_ref):
        # 添加自定义方法'array'，将其与self.MyArray关联
        self.add_method('array', self.MyArray)
        # 添加指定的函数作为自定义方法，并将其与self.MyArray关联
        self.add_method(function, self.MyArray)
        # 从numpy中获取指定函数的引用
        np_func = getattr(np, function)
        # 从self.MyArray中获取指定函数的引用
        my_func = getattr(self.MyArray, function)

        # 如果numpy_ref为True，则将ref设为numpy数组，否则为self.MyArray的array方法返回值
        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()

        # 处理参数，将可调用的参数进行调用，保证参数为元组形式
        like_args = tuple(a() if callable(a) else a for a in args)
        # 使用指定的参数和kwargs，以及like参数为ref，调用np_func函数
        array_like = np_func(*like_args, **kwargs, like=ref)

        # 如果numpy_ref为True
        if numpy_ref is True:
            # 断言array_like的类型为np.ndarray
            assert type(array_like) is np.ndarray

            # 重新处理参数，保证参数为元组形式
            np_args = tuple(a() if callable(a) else a for a in args)
            # 使用指定的参数和kwargs，调用np_func函数
            np_arr = np_func(*np_args, **kwargs)

            # 特殊情况处理np.empty以确保数值匹配
            if function == "empty":
                np_arr.fill(1)
                array_like.fill(1)

            # 断言array_like与np_arr相等
            assert_equal(array_like, np_arr)
        else:
            # 断言array_like的类型为self.MyArray
            assert type(array_like) is self.MyArray
            # 断言array_like的函数属性为my_func
            assert array_like.function is my_func

    # 使用pytest标记参数化测试
    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    # 使用pytest标记参数化测试
    @pytest.mark.parametrize('ref', [1, [1], "MyNoArrayFunctionArray"])
    # 定义一个测试方法，用于测试不支持数组函数like行为的情况
    def test_no_array_function_like(self, function, args, kwargs, ref):
        # 添加自定义方法'array'，将其与self.MyNoArrayFunctionArray关联
        self.add_method('array', self.MyNoArrayFunctionArray)
        # 添加指定的函数作为自定义方法，并将其与self.MyNoArrayFunctionArray关联
        self.add_method(function, self.MyNoArrayFunctionArray)
        # 从numpy中获取指定函数的引用
        np_func = getattr(np, function)

        # 如果ref为"MyNoArrayFunctionArray"字符串，则将ref设为self.MyNoArrayFunctionArray的array方法返回值
        if ref == "MyNoArrayFunctionArray":
            ref = self.MyNoArrayFunctionArray.array()

        # 处理参数，将可调用的参数进行调用，保证参数为元组形式
        like_args = tuple(a() if callable(a) else a for a in args)

        # 使用指定的参数和kwargs，以及like参数为ref，调用np_func函数，期望抛出TypeError异常
        with assert_raises_regex(TypeError,
                'The `like` argument must be an array-like that implements'):
            np_func(*like_args, **kwargs, like=ref)

    # 使用pytest标记参数化测试
    @pytest.mark.parametrize('numpy_ref', [True, False])
    # 定义一个测试方法，用于测试从文件中读取类似数组的行为
    def test_array_like_fromfile(self, numpy_ref):
        # 添加自定义方法'array'，将其与self.MyArray关联
        self.add_method('array', self.MyArray)
        # 添加自定义方法'fromfile'，将其与self.MyArray关联
        self.add_method("fromfile", self.MyArray)

        # 如果numpy_ref为True，则将ref设为numpy数组，否则为self.MyArray的array方法返回值
        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()

        # 生成一个包含5个随机数的numpy数组
        data = np.random.random(5)

        # 使用临时目录作为文件存储的位置
        with tempfile.TemporaryDirectory() as tmpdir:
            # 构造文件名，文件位于临时目录下，名为"testfile"
            fname = os.path.join(tmpdir, "testfile")
            # 将数据写入文件
            data.tofile(fname)

            # 使用np.fromfile函数从文件中读取数据，like参数为ref
            array_like = np.fromfile(fname, like=ref)
            # 如果numpy_ref为True
            if numpy_ref is True:
                # 断言array_like的类型为np.ndarray
                assert type(array_like) is np.ndarray
                # 使用np.fromfile函数再次从文件中读取数据，like参数为ref
                np_res = np.fromfile(fname, like=ref)
                # 断言np_res与data相等
                assert_equal(np_res, data)
                # 断言array_like与np_res相等
                assert_equal(array_like, np_res)
            else:
                # 断言array_like的类型为self.MyArray
                assert type(array_like) is self.MyArray
                # 断言array_like的函数属性为self.MyArray的fromfile方法
                assert array_like.function is self.MyArray.fromfile
    # 定义一个测试方法，用于测试异常处理
    def test_exception_handling(self):
        # 向自定义数组类添加一个名为 'array' 的方法，并启用值错误异常处理选项
        self.add_method('array', self.MyArray, enable_value_error=True)

        # 调用自定义数组类的 array() 方法，返回一个引用
        ref = self.MyArray.array()

        # 使用 assert_raises 来断言会抛出 TypeError 异常
        with assert_raises(TypeError):
            # 预期此处会首先引发关于 'value_error' 参数无效的错误
            np.array(1, value_error=True, like=ref)

    # 使用 pytest 的参数化标记来定义一个测试方法，测试函数为 _array_tests 中指定的函数
    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    def test_like_as_none(self, function, args, kwargs):
        # 向自定义数组类添加一个名为 'array' 的方法
        self.add_method('array', self.MyArray)
        # 向自定义数组类添加指定的 function 方法
        self.add_method(function, self.MyArray)
        # 获取 numpy 模块中的 function 函数
        np_func = getattr(np, function)

        # 将 args 中的每个元素如果是可调用的则调用，形成新的元组 like_args
        like_args = tuple(a() if callable(a) else a for a in args)
        # 对于 loadtxt 和 genfromtxt 函数，初始化时需要这样做以避免错误
        like_args_exp = tuple(a() if callable(a) else a for a in args)

        # 调用 numpy 模块中的 function 函数，like 参数设为 None
        array_like = np_func(*like_args, **kwargs, like=None)
        # 调用 numpy 模块中的 function 函数，使用 like_args_exp 和 kwargs
        expected = np_func(*like_args_exp, **kwargs)

        # 特殊处理 np.empty 函数，确保其生成的数组值与期望值一致
        if function == "empty":
            array_like.fill(1)
            expected.fill(1)

        # 使用 assert_equal 断言 array_like 与 expected 相等
        assert_equal(array_like, expected)
def test_function_like():
    # 检查 np.mean 的类型是否为特定的 ArrayFunctionDispatcher 类型
    assert type(np.mean) is np._core._multiarray_umath._ArrayFunctionDispatcher 

    # 定义一个 MyClass 类
    class MyClass:
        # 实现 __array__ 方法以支持 mean 函数
        def __array__(self, dtype=None, copy=None):
            # 返回一个 ndarray 以供 np.mean 使用
            return np.arange(3)

        # 静态方法 func1 使用 np.mean
        func1 = staticmethod(np.mean)
        # 实例方法 func2 使用 np.mean
        func2 = np.mean
        # 类方法 func3 使用 np.mean
        func3 = classmethod(np.mean)

    # 创建 MyClass 实例 m
    m = MyClass()
    # 断言 func1 在 m 实例上调用时返回 10
    assert m.func1([10]) == 10
    # 断言 func2 在 m 实例上调用时返回 1（np.arange(3) 的均值）
    assert m.func2() == 1
    # 使用 pytest 断言 func3 在类上调用时引发 TypeError 异常，匹配给定的错误信息
    with pytest.raises(TypeError, match="unsupported operand type"):
        m.func3()

    # 手动绑定 np.mean 方法也可以正常工作（上面的方法可能是一个捷径）：
    # 绑定 np.mean 方法到 MyClass 实例 m 上
    bound = np.mean.__get__(m, MyClass)
    # 断言绑定后调用返回 1
    assert bound() == 1

    # 未绑定的方法实际上是静态调用，这里传入 None
    bound = np.mean.__get__(None, MyClass)
    # 断言未绑定调用时返回 10
    assert bound([10]) == 10

    # 绑定 np.mean 方法作为类方法
    bound = np.mean.__get__(MyClass)
    # 使用 pytest 断言调用时引发 TypeError 异常，匹配给定的错误信息
    with pytest.raises(TypeError, match="unsupported operand type"):
        bound()
```