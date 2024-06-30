# `D:\src\scipysrc\sympy\sympy\core\decorators.py`

```
"""
SymPy core decorators.

The purpose of this module is to expose decorators without any other
dependencies, so that they can be easily imported anywhere in sympy/core.
"""

from functools import wraps  # 导入 functools 库中的 wraps 函数
from .sympify import SympifyError, sympify  # 导入当前目录下的 sympify 模块及其异常 SympifyError


def _sympifyit(arg, retval=None):
    """
    decorator to smartly _sympify function arguments

    Explanation
    ===========

    @_sympifyit('other', NotImplemented)
    def add(self, other):
        ...

    In add, other can be thought of as already being a SymPy object.

    If it is not, the code is likely to catch an exception, then other will
    be explicitly _sympified, and the whole code restarted.

    if _sympify(arg) fails, NotImplemented will be returned

    See also
    ========

    __sympifyit
    """
    def deco(func):
        return __sympifyit(func, arg, retval)

    return deco


def __sympifyit(func, arg, retval=None):
    """Decorator to _sympify `arg` argument for function `func`.

       Do not use directly -- use _sympifyit instead.
    """

    # we support f(a,b) only
    if not func.__code__.co_argcount:
        raise LookupError("func not found")
    # only b is _sympified
    assert func.__code__.co_varnames[1] == arg  # 断言第二个参数名必须为 arg
    if retval is None:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            return func(a, sympify(b, strict=True))  # 将参数 b 严格 _sympify 后调用 func

    else:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                # If an external class has _op_priority, it knows how to deal
                # with SymPy objects. Otherwise, it must be converted.
                if not hasattr(b, '_op_priority'):
                    b = sympify(b, strict=True)  # 如果 b 没有 _op_priority 属性，严格 _sympify b
                return func(a, b)
            except SympifyError:
                return retval  # 捕获 SympifyError 异常时返回指定的 retval

    return __sympifyit_wrapper


def call_highest_priority(method_name):
    """A decorator for binary special methods to handle _op_priority.

    Explanation
    ===========

    Binary special methods in Expr and its subclasses use a special attribute
    '_op_priority' to determine whose special method will be called to
    handle the operation. In general, the object having the highest value of
    '_op_priority' will handle the operation. Expr and subclasses that define
    custom binary special methods (__mul__, etc.) should decorate those
    methods with this decorator to add the priority logic.

    The ``method_name`` argument is the name of the method of the other class
    that will be called.  Use this decorator in the following manner::

        # Call other.__rmul__ if other._op_priority > self._op_priority
        @call_highest_priority('__rmul__')
        def __mul__(self, other):
            ...

        # Call other.__mul__ if other._op_priority > self._op_priority
        @call_highest_priority('__mul__')
        def __rmul__(self, other):
        ...
    """
    # 定义一个装饰器函数 priority_decorator，接受一个函数 func 作为参数
    def priority_decorator(func):
        # 使用 functools 模块中的 wraps 装饰器，保留原始函数 func 的元数据
        @wraps(func)
        # 定义一个内部函数 binary_op_wrapper，接受 self 和 other 两个参数
        def binary_op_wrapper(self, other):
            # 检查 other 对象是否具有 _op_priority 属性
            if hasattr(other, '_op_priority'):
                # 检查 other 对象的 _op_priority 是否大于 self 对象的 _op_priority
                if other._op_priority > self._op_priority:
                    # 获取 other 对象中名为 method_name 的属性 f
                    f = getattr(other, method_name, None)
                    # 如果 f 不为空，则调用 f(self) 并返回结果
                    if f is not None:
                        return f(self)
            # 如果 other 对象不满足上述条件，则调用原始函数 func(self, other) 并返回结果
            return func(self, other)
        # 返回内部函数 binary_op_wrapper，用作装饰后的函数
        return binary_op_wrapper
    # 返回装饰器函数 priority_decorator，用于装饰其他函数或方法
    return priority_decorator
# 定义一个装饰器函数，用于处理需要 sympify（符号化）参数的方法
def sympify_method_args(cls):
    '''
    为方法带有 sympify（符号化）参数的类提供装饰器。

    说明
    ===========

    sympify_method_args 装饰器与 sympify_return 装饰器一起使用，用于自动对方法参数进行符号化。
    这适用于常见的写法，比如：

    示例
    ========

    >>> from sympy import Basic, SympifyError, S
    >>> from sympy.core.sympify import _sympify

    >>> class MyTuple(Basic):
    ...     def __add__(self, other):
    ...         try:
    ...             other = _sympify(other)
    ...         except SympifyError:
    ...             return NotImplemented
    ...         if not isinstance(other, MyTuple):
    ...             return NotImplemented
    ...         return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    在上面的示例中，重要的是当 other 无法符号化或符号化结果不是预期类型时返回 NotImplemented。
    这使得 MyTuple 类可以与重载 __add__ 并希望与 Tuple 实例进行合作的其他类协作使用。

    使用本装饰器，上述代码可以写成

    >>> from sympy.core.decorators import sympify_method_args, sympify_return

    >>> @sympify_method_args
    ... class MyTuple(Basic):
    ...     @sympify_return([('other', 'MyTuple')], NotImplemented)
    ...     def __add__(self, other):
    ...          return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    这里的思路是装饰器负责在每个可能需要接受非符号化参数的方法中处理此类样板代码。
    然后，例如 __add__ 方法的主体可以编写而不必担心调用 _sympify 或检查生成对象的类型。

    sympify_return 的参数是形如 (parameter_name, expected_type) 的元组列表和要返回的值（例如 NotImplemented）。
    expected_type 参数可以是一个类型，例如 Tuple，或者是一个字符串 'Tuple'。
    使用字符串对于在其类体内指定类型很有用（如上面的示例所示）。

    注意：目前 sympify_return 仅适用于接受单个参数（不包括 self）的方法。
    将 expected_type 指定为字符串仅适用于定义方法的类。
    '''
    # 从 sympify_return 装饰器创建的每个包装对象中提取包装方法，并将 cls 参数传递给它们，以便进行前向字符串引用。
    for attrname, obj in cls.__dict__.items():
        if isinstance(obj, _SympifyWrapper):
            setattr(cls, attrname, obj.make_wrapped(cls))
    # 返回经过装饰的类
    return cls
    '''Function/method decorator to sympify arguments automatically
    
    See the docstring of sympify_method_args for explanation.
    '''
    # 定义一个装饰器函数，用于自动将参数转换为 sympy 对象
    # 接受一个函数作为参数，并返回一个包装了原函数的 _SympifyWrapper 对象
    def wrapper(func):
        return _SympifyWrapper(func, args)
    # 返回装饰器函数本身，以便将其应用于其他函数或方法
    return wrapper
class _SympifyWrapper:
    '''Internal class used by sympify_return and sympify_method_args'''

    def __init__(self, func, args):
        # 初始化函数，接收函数对象和参数元组作为参数
        self.func = func
        self.args = args

    def make_wrapped(self, cls):
        # 根据给定的类创建包装函数
        func = self.func
        parameters, retval = self.args

        # XXX: Handle more than one parameter?
        # 处理多于一个参数的情况？
        [(parameter, expectedcls)] = parameters

        # Handle forward references to the current class using strings
        # 处理使用字符串进行对当前类的前向引用
        if expectedcls == cls.__name__:
            expectedcls = cls

        # Raise RuntimeError since this is a failure at import time and should
        # not be recoverable.
        # 由于在导入时出现失败且无法恢复，因此引发 RuntimeError。
        nargs = func.__code__.co_argcount
        # we support f(a, b) only
        # 只支持形如 f(a, b) 的函数
        if nargs != 2:
            raise RuntimeError('sympify_return can only be used with 2 argument functions')
        # only b is _sympified
        # 只有参数 b 会被 sympify 化
        if func.__code__.co_varnames[1] != parameter:
            raise RuntimeError('parameter name mismatch "%s" in %s' %
                    (parameter, func.__name__))

        @wraps(func)
        def _func(self, other):
            # XXX: The check for _op_priority here should be removed. It is
            # needed to stop mutable matrices from being sympified to
            # immutable matrices which breaks things in quantum...
            # 此处应该移除对 _op_priority 的检查。它用于阻止可变矩阵被 sympify 化为不可变矩阵，这会导致量子计算中出现问题。
            if not hasattr(other, '_op_priority'):
                try:
                    other = sympify(other, strict=True)
                except SympifyError:
                    return retval
            if not isinstance(other, expectedcls):
                return retval
            return func(self, other)

        return _func
```