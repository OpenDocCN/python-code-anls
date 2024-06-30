# `D:\src\scipysrc\sympy\sympy\core\cache.py`

```
""" Caching facility for SymPy """
# 导入模块 import_module 用于动态导入模块，Callable 用于类型提示
from importlib import import_module
from typing import Callable

# _cache 类，继承自 list，用于管理缓存的函数列表
class _cache(list):
    """ List of cached functions """

    # 打印缓存信息的方法
    def print_cache(self):
        """print cache info"""

        # 遍历缓存列表中的每个函数
        for item in self:
            # 获取函数名
            name = item.__name__
            myfunc = item
            # 循环检查函数是否被包装过
            while hasattr(myfunc, '__wrapped__'):
                # 如果函数有 cache_info 方法，则获取缓存信息
                if hasattr(myfunc, 'cache_info'):
                    info = myfunc.cache_info()
                    break
                else:
                    myfunc = myfunc.__wrapped__
            else:
                info = None

            # 打印函数名及其缓存信息
            print(name, info)

    # 清除缓存内容的方法
    def clear_cache(self):
        """clear cache content"""
        # 遍历缓存列表中的每个函数
        for item in self:
            myfunc = item
            # 循环检查函数是否被包装过
            while hasattr(myfunc, '__wrapped__'):
                # 如果函数有 cache_clear 方法，则清除缓存
                if hasattr(myfunc, 'cache_clear'):
                    myfunc.cache_clear()
                    break
                else:
                    myfunc = myfunc.__wrapped__


# 全局缓存注册表:
CACHE = _cache()
# 使 print_cache 和 clear_cache 方法可用
print_cache = CACHE.print_cache
clear_cache = CACHE.clear_cache

# 导入 lru_cache 和 wraps 函数
from functools import lru_cache, wraps

# 缓存装饰器函数 __cacheit，接受最大缓存大小 maxsize 参数
def __cacheit(maxsize):
    """caching decorator.

        important: the result of cached function must be *immutable*


        Examples
        ========

        >>> from sympy import cacheit
        >>> @cacheit
        ... def f(a, b):
        ...    return a+b

        >>> @cacheit
        ... def f(a, b): # noqa: F811
        ...    return [a, b] # <-- WRONG, returns mutable object

        to force cacheit to check returned results mutability and consistency,
        set environment variable SYMPY_USE_CACHE to 'debug'
    """
    # 内部函数，接受被装饰的函数 func
    def func_wrapper(func):
        # 使用 lru_cache 创建缓存版本的函数 cfunc
        cfunc = lru_cache(maxsize, typed=True)(func)

        # 包装器函数 wrapper，保留原始函数的元数据
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 尝试从缓存中获取结果
                retval = cfunc(*args, **kwargs)
            except TypeError as e:
                # 捕获 TypeError 异常，处理不可哈希类型的情况
                if not e.args or not e.args[0].startswith('unhashable type:'):
                    raise
                # 如果出现问题，则调用原始函数
                retval = func(*args, **kwargs)
            return retval

        # 添加缓存信息和清除缓存的方法到 wrapper 函数
        wrapper.cache_info = cfunc.cache_info
        wrapper.cache_clear = cfunc.cache_clear

        # 将 wrapper 函数添加到 CACHE 缓存列表中
        CACHE.append(wrapper)
        return wrapper

    return func_wrapper

########################################

# 未缓存的版本，直接返回传入的函数
def __cacheit_nocache(func):
    return func

# 缓存调试版本，包含检查缓存一致性的代码
def __cacheit_debug(maxsize):
    """cacheit + code to check cache consistency"""
    def func_wrapper(func):
        # 使用缓存函数 __cacheit(maxsize) 包装传入的函数 func，返回新的函数 cfunc
        cfunc = __cacheit(maxsize)(func)

        @wraps(func)
        def wrapper(*args, **kw_args):
            # 总是调用原始函数 func 并将其结果与缓存版本进行比较
            r1 = func(*args, **kw_args)
            r2 = cfunc(*args, **kw_args)

            # 尝试查看结果是否是不可变的
            #
            # 这段代码的原理是：
            #
            # hash([1,2,3])         -> 抛出 TypeError
            # hash({'a':1, 'b':2})  -> 抛出 TypeError
            # hash((1,[2,3]))       -> 抛出 TypeError
            #
            # hash((1,2,3))         -> 只是计算哈希值
            hash(r1), hash(r2)

            # 同时检查返回的值是否相同
            if r1 != r2:
                raise RuntimeError("Returned values are not the same")
            return r1
        return wrapper
    return func_wrapper
# 获取环境变量中指定键的值，默认为给定的默认值
def _getenv(key, default=None):
    from os import getenv
    return getenv(key, default)

# 从环境变量中读取 SYMPY_USE_CACHE 的设置，转换为小写
USE_CACHE = _getenv('SYMPY_USE_CACHE', 'yes').lower()

# 从环境变量中读取 SYMPY_CACHE_SIZE 的设置，转换为小写
scs = _getenv('SYMPY_CACHE_SIZE', '1000')

# 如果 SYMPY_CACHE_SIZE 的值为 'none'，则将 SYMPY_CACHE_SIZE 设为 None，表示不进行缓存
if scs.lower() == 'none':
    SYMPY_CACHE_SIZE = None
else:
    try:
        # 尝试将 SYMPY_CACHE_SIZE 的值转换为整数
        SYMPY_CACHE_SIZE = int(scs)
    except ValueError:
        # 若无法转换为整数，则抛出错误
        raise RuntimeError(
            'SYMPY_CACHE_SIZE must be a valid integer or None. ' + \
            'Got: %s' % SYMPY_CACHE_SIZE)

# 根据 USE_CACHE 的不同值，选择不同的缓存策略
if USE_CACHE == 'no':
    # 不使用缓存
    cacheit = __cacheit_nocache
elif USE_CACHE == 'yes':
    # 使用标准缓存策略，并设置缓存大小为 SYMPY_CACHE_SIZE
    cacheit = __cacheit(SYMPY_CACHE_SIZE)
elif USE_CACHE == 'debug':
    # 使用调试模式的缓存策略，但会明显降低性能
    cacheit = __cacheit_debug(SYMPY_CACHE_SIZE)   # a lot slower
else:
    # 若未识别的 USE_CACHE 值，则抛出错误
    raise RuntimeError(
        'unrecognized value for SYMPY_USE_CACHE: %s' % USE_CACHE)


def cached_property(func):
    '''Decorator to cache property method'''

    # 属性名，用于存储缓存结果
    attrname = '__' + func.__name__
    _cached_property_sentinel = object()

    def propfunc(self):
        # 获取已缓存的值，如果不存在则调用函数计算并缓存结果
        val = getattr(self, attrname, _cached_property_sentinel)
        if val is _cached_property_sentinel:
            val = func(self)
            setattr(self, attrname, val)
        return val

    return property(propfunc)


def lazy_function(module : str, name : str) -> Callable:
    """Create a lazy proxy for a function in a module.

    The module containing the function is not imported until the function is used.

    """

    func = None

    def _get_function():
        nonlocal func
        if func is None:
            # 延迟加载函数所在的模块，直到首次调用函数时才导入模块
            func = getattr(import_module(module), name)
        return func

    # 使用元类以便 help() 函数能显示文档字符串
    class LazyFunctionMeta(type):

        @property
        def __doc__(self):
            # 获取函数的文档字符串，并添加额外信息说明这是一个 LazyFunction 的包装器
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

    class LazyFunction(metaclass=LazyFunctionMeta):

        def __call__(self, *args, **kwargs):
            # 内联获取函数对象以提升性能
            nonlocal func
            if func is None:
                func = getattr(import_module(module), name)
            return func(*args, **kwargs)

        @property
        def __doc__(self):
            # 获取函数的文档字符串，并添加额外信息说明这是一个 LazyFunction 的包装器
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

        def __str__(self):
            return _get_function().__str__()

        def __repr__(self):
            return f"<{__class__.__name__} object at 0x{id(self):x}>: wrapping '{module}.{name}'"

    return LazyFunction()
```