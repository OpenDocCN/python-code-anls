# `D:\src\scipysrc\scikit-learn\sklearn\utils\deprecation.py`

```
# 导入 functools 和 warnings 模块
import functools
import warnings

# 定义 __all__ 列表，指明模块中公开的对象
__all__ = ["deprecated"]

# 定义一个装饰器类 deprecated，用于标记已弃用的函数或类
class deprecated:
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    Examples
    --------
    >>> from sklearn.utils import deprecated
    >>> deprecated()
    <sklearn.utils.deprecation.deprecated object at ...>
    >>> @deprecated()
    ... def some_function(): pass

    Parameters
    ----------
    extra : str, default=''
          To be added to the deprecation messages.
    """

    # Adapted from https://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    # 初始化方法，接受一个可选的额外消息参数
    def __init__(self, extra=""):
        self.extra = extra

    # 实现 __call__ 方法，使得实例对象可以像函数一样调用，用来装饰函数或类
    def __call__(self, obj):
        """Call method

        Parameters
        ----------
        obj : object
        """
        # 如果被装饰的对象是类，则调用 _decorate_class 方法
        if isinstance(obj, type):
            return self._decorate_class(obj)
        # 如果被装饰的对象是属性，则调用 _decorate_property 方法
        elif isinstance(obj, property):
            return self._decorate_property(obj)
        # 否则，调用 _decorate_fun 方法（装饰函数）
        else:
            return self._decorate_fun(obj)

    # 装饰类的方法，生成一个新的包含警告的类
    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # 保存原始的 __new__ 方法
        new = cls.__new__

        # 定义包装函数，用来替换原始的 __new__ 方法
        def wrapped(cls, *args, **kwargs):
            warnings.warn(msg, category=FutureWarning)
            # 如果原始的 __new__ 方法是 object.__new__，则返回 object.__new__(cls)
            if new is object.__new__:
                return object.__new__(cls)
            # 否则，调用原始的 __new__ 方法
            return new(cls, *args, **kwargs)

        # 将包装函数赋值给类的 __new__ 方法
        cls.__new__ = wrapped

        # 设置包装函数的名称为 "__new__"，并保存原始的 __new__ 方法
        wrapped.__name__ = "__new__"
        wrapped.deprecated_original = new

        return cls

    # 装饰函数的方法，生成一个新的包含警告的函数
    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # 定义包装函数，用来替换原始的函数
        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=FutureWarning)
            # 调用原始的函数
            return fun(*args, **kwargs)

        # 在包装函数上添加一个指向原始函数的引用，以便在 Python 2 中可以检查函数参数（在 Python 3 中已经支持）
        wrapped.__wrapped__ = fun

        return wrapped

    # 装饰属性的方法，生成一个新的包含警告的属性
    def _decorate_property(self, prop):
        msg = self.extra

        # 定义包装的属性，用来替换原始属性的 getter 方法
        @property
        @functools.wraps(prop)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=FutureWarning)
            # 调用原始属性的 getter 方法
            return prop.fget(*args, **kwargs)

        return wrapped

# 定义一个内部函数 _is_deprecated，用来判断函数是否已经被弃用
def _is_deprecated(func):
    """Helper to check if func is wrapped by our deprecated decorator"""
    # 获取函数的闭包列表，如果不存在闭包则置为空列表
    closures = getattr(func, "__closure__", [])
    # 如果闭包为 None，则将其置为空列表
    if closures is None:
        closures = []
    # 检查闭包中是否存在字符串类型的 cell_contents，如果存在 "deprecated" 字符串则表示函数被标记为 deprecated
    is_deprecated = "deprecated" in "".join(
        [c.cell_contents for c in closures if isinstance(c.cell_contents, str)]
    )
    # 返回标记结果，True 表示函数被标记为 deprecated，否则为 False
    return is_deprecated
# TODO: remove in 1.7
# 定义一个函数 `_deprecate_Xt_in_inverse_transform`，用于处理在 `inverse_transform` 方法中弃用 `Xt` 参数，推荐使用 `X` 参数。
def _deprecate_Xt_in_inverse_transform(X, Xt):
    """Helper to deprecate the `Xt` argument in favor of `X` in inverse_transform."""
    # 如果 `X` 和 `Xt` 都不为 None，则抛出类型错误，因为不能同时使用两者，只能选择其中之一使用。
    if X is not None and Xt is not None:
        raise TypeError("Cannot use both X and Xt. Use X only.")

    # 如果 `X` 和 `Xt` 都为 None，则抛出类型错误，因为需要至少提供一个参数。
    if X is None and Xt is None:
        raise TypeError("Missing required positional argument: X.")

    # 如果 `Xt` 不为 None，则发出警告说明在版本 1.5 中 `Xt` 已更名为 `X`，并在版本 1.7 中将删除。
    if Xt is not None:
        warnings.warn(
            "Xt was renamed X in version 1.5 and will be removed in 1.7.",
            FutureWarning,
        )
        # 返回 `Xt` 参数
        return Xt

    # 如果 `Xt` 为 None，则返回 `X` 参数
    return X
```