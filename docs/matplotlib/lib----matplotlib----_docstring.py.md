# `D:\src\scipysrc\matplotlib\lib\matplotlib\_docstring.py`

```
# 导入 inspect 模块
import inspect

# 从当前目录下的 _api 模块中导入所有内容
from . import _api

# 定义一个装饰器，用于定义艺术家属性的 kwdoc 文档
def kwarg_doc(text):
    """
    Decorator for defining the kwdoc documentation of artist properties.

    This decorator can be applied to artist property setter methods.
    The given text is stored in a private attribute ``_kwarg_doc`` on
    the method.  It is used to overwrite auto-generated documentation
    in the *kwdoc list* for artists. The kwdoc list is used to document
    ``**kwargs`` when they are properties of an artist. See e.g. the
    ``**kwargs`` section in `.Axes.text`.

    The text should contain the supported types, as well as the default
    value if applicable, e.g.:

        @_docstring.kwarg_doc("bool, default: :rc:`text.usetex`")
        def set_usetex(self, usetex):

    See Also
    --------
    matplotlib.artist.kwdoc

    """
    def decorator(func):
        # 将文本存储在方法的私有属性 _kwarg_doc 中
        func._kwarg_doc = text
        return func
    return decorator

# 定义一个 Substitution 类，用于在对象的文档字符串上执行 % 替换
class Substitution:
    """
    A decorator that performs %-substitution on an object's docstring.

    This decorator should be robust even if ``obj.__doc__`` is None (for
    example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Substitution with a sequence or dictionary
    suitable for performing substitution; then decorate a suitable function
    with the constructed object, e.g.::

        sub_author_name = Substitution(author='Jason')

        @sub_author_name
        def some_function(x):
            "%(author)s wrote this function"

        # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments::

        sub_first_last_names = Substitution('Edgar Allen', 'Poe')

        @sub_first_last_names
        def some_function(x):
            "%s %s wrote the Raven"
    """
    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise TypeError("Only positional or keyword args are allowed")
        self.params = args or kwargs

    def __call__(self, func):
        # 如果函数有文档字符串，则对其进行 % 替换
        if func.__doc__:
            func.__doc__ = inspect.cleandoc(func.__doc__) % self.params
        return func

    def update(self, *args, **kwargs):
        """
        Update ``self.params`` (which must be a dict) with the supplied args.
        """
        self.params.update(*args, **kwargs)

# 定义一个继承自 dict 的 _ArtistKwdocLoader 类
class _ArtistKwdocLoader(dict):
    def __missing__(self, key):
        # 如果缺少指定的键且不是以 ":kwdoc" 结尾，则引发 KeyError
        if not key.endswith(":kwdoc"):
            raise KeyError(key)
        name = key[:-len(":kwdoc")]
        from matplotlib.artist import Artist, kwdoc
        try:
            # 查找 Artist 的子类中与指定名称匹配的类
            cls, = [cls for cls in _api.recursive_subclasses(Artist)
                    if cls.__name__ == name]
        except ValueError as e:
            raise KeyError(key) from e
        return self.setdefault(key, kwdoc(cls))

# 定义一个继承自 Substitution 的 _ArtistPropertiesSubstitution 类
class _ArtistPropertiesSubstitution(Substitution):
    """
    A `.Substitution` with two additional features:
    # 初始化方法，用于创建一个新的实例
    def __init__(self):
        # 初始化一个参数对象，使用 _ArtistKwdocLoader 类
        self.params = _ArtistKwdocLoader()

    # 调用对象时触发的方法
    def __call__(self, obj):
        # 调用父类的 __call__ 方法
        super().__call__(obj)
        # 如果对象是类且其初始化方法不是 object 类的初始化方法
        if isinstance(obj, type) and obj.__init__ != object.__init__:
            # 递归调用自身，处理对象的初始化方法
            self(obj.__init__)
        # 返回对象本身
        return obj
# 定义一个函数copy，用于从另一个源函数中复制文档字符串（如果存在）。
def copy(source):
    """Copy a docstring from another source function (if present)."""
    # 定义内部函数do_copy，接收一个目标函数作为参数
    def do_copy(target):
        # 如果源函数source有文档字符串，则将其复制给目标函数target的文档字符串
        if source.__doc__:
            target.__doc__ = source.__doc__
        # 返回带有可能复制了文档字符串的目标函数target
        return target
    # 返回内部函数do_copy，用于后续复制文档字符串到其他函数
    return do_copy


# 创建一个装饰器，用于容纳在Matplotlib中重复使用的各种文档字符串片段。
# dedent_interpd和interpd都指向_ArtistPropertiesSubstitution()的实例
dedent_interpd = interpd = _ArtistPropertiesSubstitution()
```