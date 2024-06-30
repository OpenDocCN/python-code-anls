# `D:\src\scipysrc\sympy\sympy\core\singleton.py`

```
# 导入需要的模块和类
from .core import Registry  # 从当前包中导入 Registry 类
from .sympify import sympify  # 从当前包中导入 sympify 函数

# 定义 SingletonRegistry 类，继承自 Registry 类
class SingletonRegistry(Registry):
    """
    The registry for the singleton classes (accessible as ``S``).

    Explanation
    ===========

    This class serves as two separate things.

    The first thing it is is the ``SingletonRegistry``. Several classes in
    SymPy appear so often that they are singletonized, that is, using some
    metaprogramming they are made so that they can only be instantiated once
    (see the :class:`sympy.core.singleton.Singleton` class for details). For
    instance, every time you create ``Integer(0)``, this will return the same
    instance, :class:`sympy.core.numbers.Zero`. All singleton instances are
    attributes of the ``S`` object, so ``Integer(0)`` can also be accessed as
    ``S.Zero``.

    Singletonization offers two advantages: it saves memory, and it allows
    fast comparison. It saves memory because no matter how many times the
    singletonized objects appear in expressions in memory, they all point to
    the same single instance in memory. The fast comparison comes from the
    fact that you can use ``is`` to compare exact instances in Python
    (usually, you need to use ``==`` to compare things). ``is`` compares
    objects by memory address, and is very fast.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> a = Integer(0)
    >>> a is S.Zero
    True

    For the most part, the fact that certain objects are singletonized is an
    implementation detail that users should not need to worry about. In SymPy
    library code, ``is`` comparison is often used for performance purposes
    The primary advantage of ``S`` for end users is the convenient access to
    certain instances that are otherwise difficult to type, like ``S.Half``
    (instead of ``Rational(1, 2)``).

    When using ``is`` comparison, make sure the argument is sympified. For
    instance,

    >>> x = 0
    >>> x is S.Zero
    False

    This problem is not an issue when using ``==``, which is recommended for
    most use-cases:

    >>> 0 == S.Zero
    True

    The second thing ``S`` is is a shortcut for
    :func:`sympy.core.sympify.sympify`. :func:`sympy.core.sympify.sympify` is
    the function that converts Python objects such as ``int(1)`` into SymPy
    objects such as ``Integer(1)``. It also converts the string form of an
    expression into a SymPy expression, like ``sympify("x**2")`` ->
    ``Symbol("x")**2``. ``S(1)`` is the same thing as ``sympify(1)``
    (basically, ``S.__call__`` has been defined to call ``sympify``).

    This is for convenience, since ``S`` is a single letter. It's mostly
    useful for defining rational numbers. Consider an expression like ``x +
    1/2``. If you enter this directly in Python, it will evaluate the ``1/2``
    and give ``0.5``, because both arguments are ints (see also
    :ref:`tutorial-gotchas-final-notes`). However, in SymPy, you usually want
    """
    pass  # SingletonRegistry 类暂时没有额外的实现，使用了 pass 表示空操作
    # 空的元组，用于限制实例的动态属性赋值，以提高性能和安全性
    __slots__ = ()
    
    # 将 sympify 方法设置为静态方法 __call__，允许使用 S() 创建 SymPy 对象的快捷方式
    __call__ = staticmethod(sympify)
    
    # 初始化方法，创建一个空字典 _classes_to_install 用于注册尚未安装的类
    def __init__(self):
        self._classes_to_install = {}
        # _classes_to_install 字典存储了注册但尚未安装为属性的类，
        # 其作用是延迟类的创建直到首次访问属性时进行安装，
        # 这有助于控制对象创建时机，避免过早的对象创建。
    
    # 注册方法，用于向 SingletonRegistry 注册类，并确保重复的类可以覆盖旧的注册
    def register(self, cls):
        if hasattr(self, cls.__name__):
            delattr(self, cls.__name__)
        self._classes_to_install[cls.__name__] = cls
        # 如果已经存在同名属性，先删除旧属性，然后将新的类添加到 _classes_to_install 字典中
    
    # __getattr__ 方法，当尝试访问不存在的属性时调用，用于动态加载注册但尚未安装的类
    def __getattr__(self, name):
        if name not in self._classes_to_install:
            raise AttributeError(
                "Attribute '%s' was not installed on SymPy registry %s" % (
                name, self))
        class_to_install = self._classes_to_install[name]
        value_to_install = class_to_install()
        self.__setattr__(name, value_to_install)
        del self._classes_to_install[name]
        return value_to_install
        # 如果请求的属性名不在 _classes_to_install 中，抛出 AttributeError；
        # 否则，获取并安装对应的类，并返回其实例化后的值，完成属性的延迟安装和访问。
    
    # __repr__ 方法，返回对象的字符串表示形式 "S"
    def __repr__(self):
        return "S"
        # 返回对象的字符串表示形式 "S"
# 创建单例注册表的实例，用于存储所有单例类的实例
S = SingletonRegistry()

# 定义单例元类
class Singleton(type):
    """
    单例模式的元类。

    Explanation
    ===========

    单例类只有一个实例，每次实例化该类时返回同一个实例。此外，可以通过全局注册对象 ``S`` 来访问这个实例，
    形式为 ``S.<class_name>``。

    Examples
    ========

        >>> from sympy import S, Basic
        >>> from sympy.core.singleton import Singleton
        >>> class MySingleton(Basic, metaclass=Singleton):
        ...     pass
        >>> Basic() is Basic()
        False
        >>> MySingleton() is MySingleton()
        True
        >>> S.MySingleton is MySingleton()
        True

    Notes
    =====

    实例的创建被延迟到第一次访问该值时。
    （在 SymPy 1.0 之前的版本中，实例在类创建时被创建，这可能会导致导入循环。）
    """
    
    # 初始化方法
    def __init__(cls, *args, **kwargs):
        # 使用 Basic 类的实例化方法创建一个实例
        cls._instance = obj = Basic.__new__(cls)
        # 将类的 __new__ 方法重写为直接返回该实例
        cls.__new__ = lambda cls: obj
        # 将类的 __getnewargs__ 方法重写为返回空元组
        cls.__getnewargs__ = lambda obj: ()
        # 将类的 __getstate__ 方法重写为返回 None
        cls.__getstate__ = lambda obj: None
        # 在单例注册表 S 中注册该类
        S.register(cls)


# 延迟导入以避免循环导入
from .basic import Basic
```