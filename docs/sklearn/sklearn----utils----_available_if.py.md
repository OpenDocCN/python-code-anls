# `D:\src\scipysrc\scikit-learn\sklearn\utils\_available_if.py`

```
# 导入需要的函数和类
from functools import update_wrapper, wraps
from types import MethodType


class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        # 存储被装饰的方法、检查函数以及属性名称
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # 更新描述器的文档字符串
        update_wrapper(self, fn)

    def _check(self, obj, owner):
        # 如果检查函数返回假值，抛出属性错误
        attr_err_msg = (
            f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
        )
        try:
            check_result = self.check(obj)
        except Exception as e:
            raise AttributeError(attr_err_msg) from e

        if not check_result:
            raise AttributeError(attr_err_msg)

    def __get__(self, obj, owner=None):
        # 当从实例中获取属性时调用此方法
        if obj is not None:
            # 只在实例上代理，不在类上代理，以允许访问文档字符串
            self._check(obj, owner=owner)
            out = MethodType(self.fn, obj)

        else:
            # 当从类中获取属性时，返回一个未绑定方法
            @wraps(self.fn)
            def out(*args, **kwargs):
                self._check(args[0], owner=owner)
                return self.fn(*args, **kwargs)

        return out


def available_if(check):
    """An attribute that is available only if check returns a truthy value.

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Returns
    -------
    callable
        Callable makes the decorated method available if `check` returns
        a truthy value, otherwise the decorated method is unavailable.

    Examples
    --------
    >>> from sklearn.utils.metaestimators import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    # 返回一个 lambda 函数，该函数创建 _AvailableIfDescriptor 实例
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)
```