# `D:\src\scipysrc\pandas\pandas\_config\config.py`

```
# 引入将要使用的模块和类型声明
from __future__ import annotations

# 引入上下文管理器和其他需要的模块
from contextlib import contextmanager
import re
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Callable,
)

# 引入警告模块，用于处理已废弃选项的警告
import warnings

# 引入 Pandas 内部模块和异常处理函数
from pandas._typing import F
from pandas.util._exceptions import find_stack_level

# 如果是类型检查模式，引入额外的类型定义
if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

# 定义已废弃选项的具名元组结构
class DeprecatedOption(NamedTuple):
    key: str             # 选项键名
    msg: str | None      # 弃用消息（可选）
    rkey: str | None     # 重定向键名（可选）
    removal_ver: str | None  # 移除版本（可选）

# 定义已注册选项的具名元组结构
class RegisteredOption(NamedTuple):
    key: str                                    # 选项键名
    defval: Any                                 # 默认值
    doc: str                                    # 文档描述
    validator: Callable[[object], Any] | None    # 验证函数（可选）
    cb: Callable[[str], Any] | None              # 回调函数（可选）

# 用于存储已废弃选项元数据的字典
_deprecated_options: dict[str, DeprecatedOption] = {}

# 用于存储已注册选项元数据的字典
_registered_options: dict[str, RegisteredOption] = {}

# 用于存储已注册选项当前值的字典
_global_config: dict[str, Any] = {}
# keys which have a special meaning
# 定义具有特殊含义的键列表
_reserved_keys: list[str] = ["all"]


class OptionError(AttributeError, KeyError):
    """
    Exception raised for pandas.options.

    Backwards compatible with KeyError checks.

    Examples
    --------
    >>> pd.options.context
    Traceback (most recent call last):
    OptionError: No such option
    """
    # OptionError 类，继承自 AttributeError 和 KeyError，用于处理 pandas.options 中的异常情况


#
# User API
# 用户接口


def _get_single_key(pat: str) -> str:
    # 从选项中选择符合模式 pat 的键
    keys = _select_options(pat)
    # 如果没有匹配到任何键
    if len(keys) == 0:
        # 如果模式被废弃，发出警告
        _warn_if_deprecated(pat)
        # 抛出 OptionError 异常，指示没有找到符合模式的键
        raise OptionError(f"No such keys(s): {pat!r}")
    # 如果匹配到多个键
    if len(keys) > 1:
        # 抛出 OptionError 异常，指示模式匹配到多个键
        raise OptionError("Pattern matched multiple keys")
    # 获取唯一匹配的键
    key = keys[0]

    # 如果键被废弃，发出警告
    _warn_if_deprecated(key)

    # 将键进行翻译处理，返回翻译后的键
    key = _translate_key(key)

    # 返回处理后的键
    return key


def get_option(pat: str) -> Any:
    """
    Retrieve the value of the specified option.

    Parameters
    ----------
    pat : str
        Regexp which should match a single option.

        .. warning::

            Partial matches are supported for convenience, but unless you use the
            full option name (e.g. x.y.z.option_name), your code may break in future
            versions if new options with similar names are introduced.

    Returns
    -------
    Any
        The value of the option.

    Raises
    ------
    OptionError : if no such option exists

    See Also
    --------
    set_option : Set the value of the specified option or options.
    reset_option : Reset one or more options to their default value.
    describe_option : Print the description for one or more registered options.

    Notes
    -----
    For all available options, please view the :ref:`User Guide <options.available>`
    or use ``pandas.describe_option()``.

    Examples
    --------
    >>> pd.get_option("display.max_columns")  # doctest: +SKIP
    4
    """
    # 获取单一选项的键
    key = _get_single_key(pat)

    # 遍历嵌套字典，返回对应选项的值
    root, k = _get_root(key)
    return root[k]


def set_option(*args) -> None:
    """
    Set the value of the specified option or options.

    Parameters
    ----------
    *args : str | object
        Arguments provided in pairs, which will be interpreted as (pattern, value)
        pairs.
        pattern: str
        Regexp which should match a single option
        value: object
        New value of option

        .. warning::

            Partial pattern matches are supported for convenience, but unless you
            use the full option name (e.g. x.y.z.option_name), your code may break in
            future versions if new options with similar names are introduced.

    Returns
    -------
    None
        No return value.

    Raises
    ------
    ValueError if odd numbers of non-keyword arguments are provided
    TypeError if keyword arguments are provided
    OptionError if no such option exists

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    reset_option : Reset one or more options to their default value.
    """
    # 设置指定选项的值或选项的值

    # 如果提供的非关键字参数为奇数个数，抛出 ValueError 异常
    if len(args) % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    # 对每一对参数执行设置操作
    for i in range(0, len(args), 2):
        pattern = args[i]
        value = args[i + 1]

        # 获取单一选项的键
        key = _get_single_key(pattern)

        # 设置选项的新值
        _set_option(key, value)
    """
    describe_option : 打印一个或多个已注册选项的描述。
    option_context : 在 `with` 语句中临时设置选项的上下文管理器。

    Notes
    -----
    若要查看所有可用选项，请参阅 :ref:`用户指南 <options.available>` 或使用 `pandas.describe_option()`。

    Examples
    --------
    >>> pd.set_option("display.max_columns", 4)
    >>> df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> df
    0  1  ...  3   4
    0  1  2  ...  4   5
    1  6  7  ...  9  10
    [2 rows x 5 columns]
    >>> pd.reset_option("display.max_columns")
    """
    # 计算传入参数的个数
    nargs = len(args)
    # 如果没有参数或者参数个数为奇数，则抛出值错误异常
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    # 遍历每对参数
    for k, v in zip(args[::2], args[1::2]):
        # 获取单个键名
        key = _get_single_key(k)

        # 获取注册的选项对象
        opt = _get_registered_option(key)
        # 如果选项存在且有验证器，则调用验证器验证参数值
        if opt and opt.validator:
            opt.validator(v)

        # 获取键名的根和根键名
        root, k_root = _get_root(key)
        # 将参数值设置到嵌套字典中对应位置
        root[k_root] = v

        # 如果有回调函数，则调用回调函数
        if opt.cb:
            opt.cb(key)
# 打印一个或多个已注册选项的描述信息。
# 如果没有参数调用，则返回所有已注册选项的列表。

def describe_option(pat: str = "", _print_desc: bool = True) -> str | None:
    """
    Print the description for one or more registered options.

    Call with no arguments to get a listing for all registered options.

    Parameters
    ----------
    pat : str, default ""
        String or string regexp pattern.
        Empty string will return all options.
        For regexp strings, all matching keys will have their description displayed.
    _print_desc : bool, default True
        If True (default) the description(s) will be printed to stdout.
        Otherwise, the description(s) will be returned as a string
        (for testing).

    Returns
    -------
    None
        If ``_print_desc=True``.
    str
        If the description(s) as a string if ``_print_desc=False``.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option or options.
    reset_option : Reset one or more options to their default value.

    Notes
    -----
    For all available options, please view the
    :ref:`User Guide <options.available>`.

    Examples
    --------
    >>> pd.describe_option("display.max_columns")  # doctest: +SKIP
    display.max_columns : int
        If max_cols is exceeded, switch to truncate view...
    """
    # 根据模式字符串获取选项键列表
    keys = _select_options(pat)
    
    # 如果未找到匹配的选项键，引发选项错误异常
    if len(keys) == 0:
        raise OptionError(f"No such keys(s) for {pat=}")

    # 构建描述字符串
    s = "\n".join([_build_option_description(k) for k in keys])

    # 如果_print_desc为True，则打印描述字符串并返回None；否则返回描述字符串
    if _print_desc:
        print(s)
        return None
    return s


# 将一个或多个选项重置为它们的默认值。
def reset_option(pat: str) -> None:
    """
    Reset one or more options to their default value.

    Parameters
    ----------
    pat : str/regex
        If specified only options matching ``pat*`` will be reset.
        Pass ``"all"`` as argument to reset all options.

        .. warning::

            Partial matches are supported for convenience, but unless you
            use the full option name (e.g. x.y.z.option_name), your code may break
            in future versions if new options with similar names are introduced.

    Returns
    -------
    None
        No return value.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option or options.
    describe_option : Print the description for one or more registered options.

    Notes
    -----
    For all available options, please view the
    :ref:`User Guide <options.available>`.

    Examples
    --------
    >>> pd.reset_option("display.max_columns")  # doctest: +SKIP
    """
    # 根据模式字符串获取选项键列表
    keys = _select_options(pat)

    # 如果未找到匹配的选项键，引发选项错误异常
    if len(keys) == 0:
        raise OptionError(f"No such keys(s) for {pat=}")
    # 如果满足条件：keys 的长度大于 1，且 pat 的长度小于 4，并且 pat 不等于 "all"
    # 则抛出 ValueError 异常，提醒必须至少指定 4 个字符来重置多个选项，
    # 可以使用特殊关键字 "all" 将所有选项重置为它们的默认值。
    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when "
            "resetting multiple keys, use the special keyword "
            '"all" to reset all the options to their default value'
        )
    
    # 遍历 keys 中的每个键 k
    for k in keys:
        # 调用 set_option 函数，重置键 k 对应的选项为其注册选项中定义的默认值（_registered_options[k].defval）
        set_option(k, _registered_options[k].defval)
def get_default_val(pat: str):
    # 调用辅助函数获取单一键值
    key = _get_single_key(pat)
    # 返回注册选项中指定键的默认值
    return _get_registered_option(key).defval


class DictWrapper:
    """提供对嵌套字典进行属性访问的功能"""

    d: dict[str, Any]

    def __init__(self, d: dict[str, Any], prefix: str = "") -> None:
        # 初始化方法，将传入的字典和可选前缀存储为对象的属性
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        # 设置属性值的方法，检查是否存在相同键，且不允许覆盖子树
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        if key in self.d and not isinstance(self.d[key], dict):
            # 如果键已存在且不是字典类型，则设置选项的值
            set_option(prefix, val)
        else:
            # 否则，抛出选项错误，不能设置不存在的选项
            raise OptionError("You can only set the value of existing options")

    def __getattr__(self, key: str):
        # 获取属性值的方法，构建完整的键名，尝试从字典中获取值
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        try:
            v = object.__getattribute__(self, "d")[key]
        except KeyError as err:
            # 如果键不存在，则抛出选项错误
            raise OptionError("No such option") from err
        if isinstance(v, dict):
            # 如果值是字典，则返回一个新的DictWrapper对象
            return DictWrapper(v, prefix)
        else:
            # 否则，返回指定键的选项值
            return get_option(prefix)

    def __dir__(self) -> list[str]:
        # 返回字典的所有键的列表作为对象的属性
        return list(self.d.keys())


options = DictWrapper(_global_config)

@contextmanager
def option_context(*args) -> Generator[None, None, None]:
    """
    用于在with语句中临时设置选项的上下文管理器。

    参数
    ----------
    *args : str | object
        以成对出现的参数提供模式和值，如(pat, val, pat, val...)。

    返回
    -------
    None
        没有返回值。

    参见
    --------
    get_option : 获取指定选项的值。
    set_option : 设置指定选项的值。
    reset_option : 将一个或多个选项重置为其默认值。
    describe_option : 打印一个或多个已注册选项的描述信息。

    注意
    -----
    查看所有可用选项，请参阅 :ref:`用户指南 <options.available>` 或使用 ``pandas.describe_option()``。

    示例
    --------
    >>> from pandas import option_context
    >>> with option_context("display.max_rows", 10, "display.max_columns", 5):
    ...     pass
    """
    if len(args) % 2 != 0 or len(args) < 2:
        # 如果参数个数不是偶数或小于2个，抛出值错误异常
        raise ValueError(
            "Provide an even amount of arguments as "
            "option_context(pat, val, pat, val...)."
        )

    # 将参数成对组成元组列表
    ops = tuple(zip(args[::2], args[1::2]))
    try:
        # 为每对参数设置新值，并存储旧值以备恢复
        undo = tuple((pat, get_option(pat)) for pat, val in ops)
        for pat, val in ops:
            set_option(pat, val)
        yield
    finally:
        # 在最终处理块中恢复所有选项的旧值
        for pat, val in undo:
            set_option(pat, val)


def register_option(
    key: str,
    defval: object,
    doc: str = "",
    ...
    validator: Callable[[object], Any] | None = None,
    cb: Callable[[str], Any] | None = None,


# 声明两个参数：validator 和 cb
# validator 是一个可调用对象（callable），接受一个 object 类型参数，返回任意类型（Any），或者为 None
# cb 是一个可调用对象（callable），接受一个 str 类型参数，返回任意类型（Any），或者为 None
def register_option(
    key: str,
    defval: object,
    doc: str,
    validator: Callable = None,
    cb: Callable = None
) -> None:
    """
    Register an option in the package-wide pandas config object

    Parameters
    ----------
    key : str
        Fully-qualified key, e.g. "x.y.option - z".
    defval : object
        Default value of the option.
    doc : str
        Description of the option.
    validator : Callable, optional
        Function of a single argument, should raise `ValueError` if
        called with a value which is not a legal value for the option.
    cb : Callable, optional
        A function of a single argument "key", which is called
        immediately after an option value is set/reset. key is
        the full name of the option.

    Raises
    ------
    OptionError
        If the specified `key` has already been registered or is a reserved key.
    ValueError
        If `validator` is specified and `defval` is not a valid value, or if any part of
        `key` is not a valid identifier or is a Python keyword.

    """
    import keyword  # 导入关键字模块
    import tokenize  # 导入 tokenize 模块用于处理 Python 代码的标记化

    key = key.lower()  # 将 key 转换为小写字母

    if key in _registered_options:
        raise OptionError(f"Option '{key}' has already been registered")  # 如果 key 已经注册，则抛出 OptionError
    if key in _reserved_keys:
        raise OptionError(f"Option '{key}' is a reserved key")  # 如果 key 是保留关键字，则抛出 OptionError

    # 如果指定了 validator，则验证 defval 是否合法
    if validator:
        validator(defval)

    # 拆分 key 成为路径列表，每一级作为字典的键
    path = key.split(".")

    # 验证 key 的每一级是否合法
    for k in path:
        if not re.match("^" + tokenize.Name + "$", k):
            raise ValueError(f"{k} is not a valid identifier")  # 如果任何部分不是有效标识符，则抛出 ValueError
        if keyword.iskeyword(k):
            raise ValueError(f"{k} is a python keyword")  # 如果任何部分是 Python 关键字，则抛出 ValueError

    cursor = _global_config  # 设置游标从全局配置开始
    msg = "Path prefix to option '{option}' is already an option"  # 提示消息模板

    # 逐级遍历路径，创建必要的字典
    for i, p in enumerate(path[:-1]):
        if not isinstance(cursor, dict):
            raise OptionError(msg.format(option=".".join(path[:i])))  # 如果路径前缀已经是一个选项，则抛出 OptionError
        if p not in cursor:
            cursor[p] = {}  # 如果当前级别不存在，则创建空字典
        cursor = cursor[p]  # 移动游标到下一级

    if not isinstance(cursor, dict):
        raise OptionError(msg.format(option=".".join(path[:-1])))

    cursor[path[-1]] = defval  # 在最后一级设置默认值

    # 保存选项的元数据
    _registered_options[key] = RegisteredOption(
        key=key, defval=defval, doc=doc, validator=validator, cb=cb
    )
    # 将 `key` 参数转换为小写，以便统一处理大小写敏感性
    key = key.lower()
    
    # 检查是否已经将 `key` 添加到已弃用选项字典 `_deprecated_options` 中
    if key in _deprecated_options:
        # 如果已经添加，抛出选项错误异常，指明该选项已经被定义为弃用
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")
    
    # 将 `key` 添加到 `_deprecated_options` 字典中，并创建一个 `DeprecatedOption` 对象
    # `DeprecatedOption` 对象包含 `key`、`msg`、`rkey` 和 `removal_ver` 参数的信息
    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)
# functions internal to the module

def _select_options(pat: str) -> list[str]:
    """
    返回与 `pat` 匹配的键列表

    如果 pat=="all"，则返回所有注册选项
    """
    # 如果 pat 是精确匹配的已注册选项，则直接返回列表
    if pat in _registered_options:
        return [pat]

    # 否则遍历所有注册选项的键，按需返回匹配的结果
    keys = sorted(_registered_options.keys())
    if pat == "all":  # 预留的关键字
        return keys

    # 使用正则表达式（忽略大小写）匹配 pat，在已注册选项中找到相应的键
    return [k for k in keys if re.search(pat, k, re.I)]


def _get_root(key: str) -> tuple[dict[str, Any], str]:
    """
    根据 key 获取其对应的根配置和末端键名

    返回一个元组，包含根配置字典和末端键名
    """
    # 将 key 按点分割，以获取配置路径
    path = key.split(".")
    cursor = _global_config

    # 遍历路径中的每个部分，以获取最终的根配置和末端键名
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _get_deprecated_option(key: str):
    """
    获取已弃用选项的元数据，如果 `key` 是已弃用的话

    返回
    -------
    如果 key 已弃用，则返回 DeprecatedOption（命名元组），否则返回 None
    """
    try:
        d = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d


def _get_registered_option(key: str):
    """
    获取注册选项的元数据，如果 `key` 是已注册选项的话

    返回
    -------
    如果 key 是已注册选项，则返回 RegisteredOption（命名元组），否则返回 None
    """
    return _registered_options.get(key)


def _translate_key(key: str) -> str:
    """
    如果 key 是已弃用且有替代键定义，则返回替代键，否则原样返回 `key`
    """
    # 获取已弃用选项的元数据
    d = _get_deprecated_option(key)
    if d:
        return d.rkey or key  # 如果有替代键则返回替代键，否则返回原键
    else:
        return key


def _warn_if_deprecated(key: str) -> bool:
    """
    检查 `key` 是否为已弃用选项，如果是则打印警告信息

    返回
    -------
    bool - 如果 `key` 是已弃用选项则返回 True，否则返回 False
    """
    # 获取已弃用选项的元数据
    d = _get_deprecated_option(key)
    if d:
        if d.msg:
            warnings.warn(
                d.msg,
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        else:
            msg = f"'{key}' is deprecated"
            if d.removal_ver:
                msg += f" and will be removed in {d.removal_ver}"
            if d.rkey:
                msg += f", please use '{d.rkey}' instead."
            else:
                msg += ", please refrain from using it."

            warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())
        return True
    return False


def _build_option_description(k: str) -> str:
    """
    构建已注册选项的格式化描述并打印出来
    """
    # 获取已注册选项和已弃用选项的元数据
    o = _get_registered_option(k)
    d = _get_deprecated_option(k)

    s = f"{k} "  # 初始化描述字符串

    # 如果有选项文档，则加入描述字符串
    if o.doc:
        s += "\n".join(o.doc.strip().split("\n"))
    else:
        s += "No description available."

    # 如果有注册选项，则加入默认值和当前值的描述
    if o:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            s += f"\n    [default: {o.defval}] [currently: {get_option(k)}]"
    # 如果条件变量 d 存在且不为空（即为真）
    if d:
        # 从对象 d 中获取属性 rkey，如果不存在则使用空字符串
        rkey = d.rkey or ""
        # 将字符串 s 后面添加一行文本，指示该功能已被弃用
        s += "\n    (Deprecated"
        # 将字符串 s 后面添加一行文本，指示应使用 rkey 替代当前功能
        s += f", use `{rkey}` instead."
        # 将字符串 s 后面添加一行文本，表示该功能已被标记为弃用
        s += ")"
        
    # 返回最终构建好的字符串 s
    return s
# 定义一个上下文管理器，用于在一个共同前缀下多次调用 API

@contextmanager
def config_prefix(prefix: str) -> Generator[None, None, None]:
    """
    上下文管理器，用于在具有共同前缀的多次 API 调用中使用

    支持的 API 函数：(register / get / set )__option

    警告：这不是线程安全的，如果使用 "from x import y" 导入 API 函数到模块中将无法正常工作。

    示例
    -------
    import pandas._config.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option(size, " 6 pt")
        cf.get_option(size)
        ...

        等等

    将注册选项 "display.font.color"、"display.font.size"，设置 "display.font.size" 的值...等等。
    """
    # 注意：reset_option 依赖于 set_option，并且直接依赖于键，不适合这种猴子补丁方案

    global register_option, get_option, set_option

    def wrap(func: F) -> F:
        def inner(key: str, *args, **kwds):
            pkey = f"{prefix}.{key}"
            return func(pkey, *args, **kwds)

        return cast(F, inner)

    _register_option = register_option
    _get_option = get_option
    _set_option = set_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    try:
        yield
    finally:
        set_option = _set_option
        get_option = _get_option
        register_option = _register_option


# 以下的工厂函数和方法可以作为 register_option 的验证器参数

def is_type_factory(_type: type[Any]) -> Callable[[Any], None]:
    """
    返回一个验证函数，用于检查参数的类型是否为指定类型

    参数
    ----------
    `_type` - 要比较的类型（例如 type(x) == `_type`）

    返回
    -------
    validator - 接受单个参数 x 的函数，如果 type(x) 不等于 `_type` 则抛出 ValueError
    """

    def inner(x) -> None:
        if type(x) != _type:
            raise ValueError(f"值必须是类型 '{_type}'")

    return inner


def is_instance_factory(_type: type | tuple[type, ...]) -> Callable[[Any], None]:
    """
    返回一个验证函数，用于检查参数是否是指定类型的实例

    参数
    ----------
    `_type` - 要检查的类型

    返回
    -------
    validator - 接受单个参数 x 的函数，如果 x 不是 `_type` 的实例则抛出 ValueError
    """
    if isinstance(_type, tuple):
        type_repr = "|".join(map(str, _type))
    else:
        type_repr = f"'{_type}'"

    def inner(x) -> None:
        if not isinstance(x, _type):
            raise ValueError(f"值必须是 {type_repr} 的实例")

    return inner


def is_one_of_factory(legal_values: Sequence) -> Callable[[Any], None]:
    callables = [c for c in legal_values if callable(c)]
    legal_values = [c for c in legal_values if not callable(c)]
    # 定义一个内部函数 inner，参数为 x，返回类型为 None
    def inner(x) -> None:
        # 检查 x 是否在合法值列表 legal_values 中
        if x not in legal_values:
            # 如果 x 不在合法值列表中，则进入条件判断
            # 检查是否有任何一个可调用对象 c(x) 返回 True
            if not any(c(x) for c in callables):
                # 如果没有可调用对象返回 True，则将合法值列表转换为字符串列表
                uvals = [str(lval) for lval in legal_values]
                # 使用竖线分隔符连接合法值列表中的值，形成可打印的字符串
                pp_values = "|".join(uvals)
                # 构造错误消息，指出 x 的值必须是合法值列表中的一个
                msg = f"Value must be one of {pp_values}"
                # 如果存在可调用对象，添加信息说明也可以是可调用的
                if len(callables):
                    msg += " or a callable"
                # 抛出 ValueError 异常，将错误消息作为异常信息
                raise ValueError(msg)

    # 返回内部函数 inner 的引用
    return inner
# 验证值是否为 None 或者是非负整数。
# 如果值为 None，则直接返回。
# 如果值为整数，则检查其是否大于等于零，是则返回，否则抛出 ValueError 异常。
def is_nonnegative_int(value: object) -> None:
    """
    Verify that value is None or a positive int.

    Parameters
    ----------
    value : None or int
            The `value` to be checked.

    Raises
    ------
    ValueError
        When the value is not None or is a negative integer
    """
    if value is None:
        return

    elif isinstance(value, int):
        if value >= 0:
            return

    msg = "Value must be a nonnegative integer or None"
    raise ValueError(msg)


# 常见类型验证器，为了方便使用
# 用法示例：register_option(... , validator = is_int)
# 这些函数使用 is_type_factory 和 is_instance_factory 工厂函数创建。
is_int = is_type_factory(int)
is_bool = is_type_factory(bool)
is_float = is_type_factory(float)
is_str = is_type_factory(str)
is_text = is_instance_factory((str, bytes))


def is_callable(obj: object) -> bool:
    """
    检查对象是否可调用（callable）。

    Parameters
    ----------
    `obj` - 要检查的对象

    Returns
    -------
    validator - 如果对象是可调用的则返回 True，否则抛出 ValueError 异常。

    Raises
    ------
    ValueError
        如果对象不可调用
    """
    if not callable(obj):
        raise ValueError("Value must be a callable")
    return True
```