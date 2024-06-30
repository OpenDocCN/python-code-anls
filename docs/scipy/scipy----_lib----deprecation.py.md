# `D:\src\scipysrc\scipy\scipy\_lib\deprecation.py`

```
# 从inspect模块中导入Parameter和signature对象
from inspect import Parameter, signature
# 从functools模块中导入所有内容
import functools
# 从warnings模块中导入所有内容
import warnings
# 从importlib模块中导入import_module函数
from importlib import import_module

# 定义模块级别变量__all__，用于指定在使用import *时应导入的符号列表
__all__ = ["_deprecated"]

# 定义一个特殊的对象_NoValue，用作将被弃用的参数的默认值。应该使用它而不是None，
# 因为用户可能会将None解析为位置参数。
_NoValue = object()

# 定义_deprecated函数，用于发出函数被弃用的警告消息
def _deprecated(msg, stacklevel=2):
    """Deprecate a function by emitting a warning on use."""
    # 此处函数体为空，用于将来添加发出警告的具体逻辑
    def wrap(fun):
        # 如果传入的参数 fun 是一个类对象，则发出运行时警告，指出试图废弃这个类
        if isinstance(fun, type):
            warnings.warn(
                f"Trying to deprecate class {fun!r}",
                category=RuntimeWarning, stacklevel=2)
            # 直接返回传入的类对象
            return fun

        # 如果传入的参数 fun 是一个函数或方法，则进行装饰处理
        @functools.wraps(fun)
        def call(*args, **kwargs):
            # 发出废弃警告，提示使用者函数或方法已经废弃
            warnings.warn(msg, category=DeprecationWarning,
                          stacklevel=stacklevel)
            # 调用原始的函数或方法，并返回其结果
            return fun(*args, **kwargs)
        
        # 将装饰后的函数的文档字符串设置为原始函数的文档字符串
        call.__doc__ = fun.__doc__
        return call

    # 返回 wrap 函数本身，使其可以被调用来进行装饰
    return wrap
class _DeprecationHelperStr:
    """
    Helper class used by deprecate_cython_api
    """
    # 初始化方法，接收内容和消息作为参数
    def __init__(self, content, message):
        self._content = content
        self._message = message

    # 定义哈希方法，用于对象的哈希化处理
    def __hash__(self):
        return hash(self._content)

    # 定义相等方法，比较对象内容是否相等，如果相等则发出警告消息
    def __eq__(self, other):
        res = (self._content == other)
        if res:
            warnings.warn(self._message, category=DeprecationWarning,
                          stacklevel=2)
        return res


def deprecate_cython_api(module, routine_name, new_name=None, message=None):
    """
    Deprecate an exported cdef function in a public Cython API module.

    Only functions can be deprecated; typedefs etc. cannot.

    Parameters
    ----------
    module : module
        Public Cython API module (e.g. scipy.linalg.cython_blas).
    routine_name : str
        Name of the routine to deprecate. May also be a fused-type
        routine (in which case its all specializations are deprecated).
    new_name : str
        New name to include in the deprecation warning message
    message : str
        Additional text in the deprecation warning message

    Examples
    --------
    Usually, this function would be used in the top-level of the
    module ``.pyx`` file:

    >>> from scipy._lib.deprecation import deprecate_cython_api
    >>> import scipy.linalg.cython_blas as mod
    >>> deprecate_cython_api(mod, "dgemm", "dgemm_new",
    ...                      message="Deprecated in Scipy 1.5.0")
    >>> del deprecate_cython_api, mod

    After this, Cython modules that use the deprecated function emit a
    deprecation warning when they are imported.

    """
    # 构建旧函数名，格式为模块名.函数名
    old_name = f"{module.__name__}.{routine_name}"

    # 构建警告消息内容，若有新函数名，则提示建议使用新函数名
    if new_name is None:
        depdoc = "`%s` is deprecated!" % old_name
    else:
        depdoc = f"`{old_name}` is deprecated, use `{new_name}` instead!"

    # 若有附加消息，添加到警告消息内容中
    if message is not None:
        depdoc += "\n" + message

    # 获取模块的 Cython API 字典
    d = module.__pyx_capi__

    # 检查函数是否为融合类型函数，并对其进行处理
    j = 0
    has_fused = False
    while True:
        fused_name = f"__pyx_fuse_{j}{routine_name}"
        if fused_name in d:
            has_fused = True
            # 使用自定义的帮助类作为键，将旧融合函数名映射到新的警告消息
            d[_DeprecationHelperStr(fused_name, depdoc)] = d.pop(fused_name)
            j += 1
        else:
            break

    # 若不是融合类型函数，则直接使用函数名进行处理
    if not has_fused:
        # 使用自定义的帮助类作为键，将旧函数名映射到新的警告消息
        d[_DeprecationHelperStr(routine_name, depdoc)] = d.pop(routine_name)


# taken from scikit-learn, see
# https://github.com/scikit-learn/scikit-learn/blob/1.3.0/sklearn/utils/validation.py#L38
def _deprecate_positional_args(func=None, *, version=None):
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    """
    version : callable, default=None
        The version when positional arguments will result in error.
    """
    # 如果未指定版本号，抛出数值错误，提示需要指定版本号来更改签名
    if version is None:
        msg = "Need to specify a version where signature will be changed"
        raise ValueError(msg)

    # 定义内部函数 _inner_deprecate_positional_args，接收一个函数 f 作为参数
    def _inner_deprecate_positional_args(f):
        # 获取函数 f 的参数签名
        sig = signature(f)
        # 初始化关键字参数列表和全部参数列表
        kwonly_args = []
        all_args = []

        # 遍历函数 f 的参数签名中的每个参数
        for name, param in sig.parameters.items():
            # 如果参数类型是 POSITIONAL_OR_KEYWORD，将其添加到全部参数列表中
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            # 如果参数类型是 KEYWORD_ONLY，将其添加到关键字参数列表中
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        # 定义内部函数 inner_f，接收任意位置参数 args 和关键字参数 kwargs
        @functools.wraps(f)
        def inner_f(*args, **kwargs):
            # 计算多余的位置参数个数
            extra_args = len(args) - len(all_args)
            # 如果多余的位置参数个数小于等于 0，则直接调用原函数 f
            if extra_args <= 0:
                return f(*args, **kwargs)

            # 如果多余的位置参数个数大于 0
            args_msg = ", ".join(kwonly_args[:extra_args])  # 提取超出的位置参数名
            # 发出警告，提示用户正在使用位置参数调用函数，建议改用关键字参数
            warnings.warn(
                (
                    f"You are passing as positional arguments: {args_msg}. "
                    "Please change your invocation to use keyword arguments. "
                    f"From SciPy {version}, passing these as positional "
                    "arguments will result in an error."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            # 将位置参数和对应的参数名组成的字典更新到关键字参数 kwargs 中
            kwargs.update(zip(sig.parameters, args))
            # 调用原函数 f，传入更新后的关键字参数
            return f(**kwargs)

        # 返回内部函数 inner_f
        return inner_f

    # 如果 func 参数不为 None，则返回 _inner_deprecate_positional_args 函数的调用结果
    if func is not None:
        return _inner_deprecate_positional_args(func)

    # 否则返回 _inner_deprecate_positional_args 函数本身
    return _inner_deprecate_positional_args
```