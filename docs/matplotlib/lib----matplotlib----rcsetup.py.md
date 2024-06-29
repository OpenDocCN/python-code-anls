# `D:\src\scipysrc\matplotlib\lib\matplotlib\rcsetup.py`

```py
"""
The rcsetup module contains the validation code for customization using
Matplotlib's rc settings.

Each rc setting is assigned a function used to validate any attempted changes
to that setting.  The validation functions are defined in the rcsetup module,
and are used to construct the rcParams global object which stores the settings
and is referenced throughout Matplotlib.

The default values of the rc settings are set in the default matplotlibrc file.
Any additions or deletions to the parameter set listed here should also be
propagated to the :file:`lib/matplotlib/mpl-data/matplotlibrc` in Matplotlib's
root source directory.
"""

import ast  # 导入 ast 模块，用于抽象语法树相关操作
from functools import lru_cache, reduce  # 导入 functools 模块中的 lru_cache 和 reduce 函数
from numbers import Real  # 导入 numbers 模块中的 Real 类
import operator  # 导入 operator 模块，用于操作符函数
import os  # 导入 os 模块，提供操作系统相关功能
import re  # 导入 re 模块，用于正则表达式操作

import numpy as np  # 导入 NumPy 库，并将其命名为 np

from matplotlib import _api, cbook  # 从 matplotlib 中导入 _api 和 cbook 模块
from matplotlib.backends import BackendFilter, backend_registry  # 从 matplotlib.backends 中导入 BackendFilter 和 backend_registry
from matplotlib.cbook import ls_mapper  # 从 matplotlib.cbook 中导入 ls_mapper 函数
from matplotlib.colors import Colormap, is_color_like  # 从 matplotlib.colors 中导入 Colormap 和 is_color_like 函数
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern  # 从 matplotlib._fontconfig_pattern 中导入 parse_fontconfig_pattern 函数
from matplotlib._enums import JoinStyle, CapStyle  # 从 matplotlib._enums 中导入 JoinStyle 和 CapStyle 枚举

# Don't let the original cycler collide with our validating cycler
from cycler import Cycler, cycler as ccycler  # 从 cycler 中导入 Cycler 类和 cycler 函数


@_api.caching_module_getattr
class __getattr__:  # 定义 __getattr__ 类，用于获取属性
    @_api.deprecated(
        "3.9",
        alternative="``matplotlib.backends.backend_registry.list_builtin"
            "(matplotlib.backends.BackendFilter.INTERACTIVE)``")
    @property
    def interactive_bk(self):  # 定义 interactive_bk 属性方法
        return backend_registry.list_builtin(BackendFilter.INTERACTIVE)  # 返回交互式后端的列表

    @_api.deprecated(
        "3.9",
        alternative="``matplotlib.backends.backend_registry.list_builtin"
            "(matplotlib.backends.BackendFilter.NON_INTERACTIVE)``")
    @property
    def non_interactive_bk(self):  # 定义 non_interactive_bk 属性方法
        return backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)  # 返回非交互式后端的列表

    @_api.deprecated(
        "3.9",
        alternative="``matplotlib.backends.backend_registry.list_builtin()``")
    @property
    def all_backends(self):  # 定义 all_backends 属性方法
        return backend_registry.list_builtin()  # 返回所有后端的列表


class ValidateInStrings:  # 定义 ValidateInStrings 类，用于验证字符串是否在指定列表中
    def __init__(self, key, valid, ignorecase=False, *,
                 _deprecated_since=None):
        """*valid* is a list of legal strings."""
        self.key = key  # 设置键值
        self.ignorecase = ignorecase  # 是否忽略大小写
        self._deprecated_since = _deprecated_since  # 弃用提示时间点

        def func(s):  # 内部函数，根据 ignorecase 决定是否转换为小写
            if ignorecase:
                return s.lower()
            else:
                return s
        self.valid = {func(k): k for k in valid}  # 使用 func 函数构建验证字典
    # 定义一个特殊方法 __call__()，使对象可以被调用
    def __call__(self, s):
        # 如果对象被标记为已弃用
        if self._deprecated_since:
            # 查找全局变量中与当前对象相同的名称
            name, = (k for k, v in globals().items() if v is self)
            # 发出弃用警告，指定弃用版本和名称，并说明对象类型为函数
            _api.warn_deprecated(
                self._deprecated_since, name=name, obj_type="function")
        
        # 如果忽略大小写并且输入是字符串，则转换输入为小写
        if self.ignorecase and isinstance(s, str):
            s = s.lower()
        
        # 如果输入在有效值列表中，则返回对应的有效值
        if s in self.valid:
            return self.valid[s]
        
        # 构造错误消息，说明输入不是有效值，列出支持的所有有效值
        msg = (f"{s!r} is not a valid value for {self.key}; supported values "
               f"are {[*self.valid.values()]}")
        
        # 如果输入是字符串，并且被双引号或单引号包围，并且去除引号后是有效值，则提醒去除引号
        if (isinstance(s, str)
                and (s.startswith('"') and s.endswith('"')
                     or s.startswith("'") and s.endswith("'"))
                and s[1:-1] in self.valid):
            msg += "; remove quotes surrounding your string"
        
        # 抛出值错误，包含详细的错误消息
        raise ValueError(msg)
# 基于 LRU 缓存的装饰器，用于将标量验证器转换为列表验证器
@lru_cache
def _listify_validator(scalar_validator, allow_stringlist=False, *,
                       n=None, doc=None):
    # 定义内部函数 f，接受字符串 s 作为输入
    def f(s):
        # 如果输入是字符串类型
        if isinstance(s, str):
            try:
                # 尝试将逗号分隔的字符串转换为列表，并使用 scalar_validator 进行每个元素验证
                val = [scalar_validator(v.strip()) for v in s.split(',')
                       if v.strip()]
            except Exception:
                if allow_stringlist:
                    # 有时，颜色列表可能是单个包含单字母颜色名称的字符串。因此尝试这种情况。
                    val = [scalar_validator(v.strip()) for v in s if v.strip()]
                else:
                    raise
        # 允许任何有序的序列类型 -- generators, np.ndarray, pd.Series
        # -- 但不允许集合类型，因为其迭代顺序是非确定性的。
        elif np.iterable(s) and not isinstance(s, (set, frozenset)):
            # 此列表推导式的条件将保留过滤掉空字符串的行为（该行为来自原始的 validate_stringlist()），
            # 同时允许任何非字符串/文本标量值，如数字和数组。
            val = [scalar_validator(v) for v in s
                   if not isinstance(v, str) or v]
        else:
            # 如果既不是字符串也不是可迭代类型，则抛出 ValueError
            raise ValueError(
                f"Expected str or other non-set iterable, but got {s}")
        # 如果 n 不为 None，并且列表 val 的长度与 n 不相等，则抛出 ValueError
        if n is not None and len(val) != n:
            raise ValueError(
                f"Expected {n} values, but there are {len(val)} values in {s}")
        return val

    try:
        # 设置内部函数 f 的名称为标量验证器的名称 + 'list'
        f.__name__ = f"{scalar_validator.__name__}list"
    except AttributeError:  # 对于类实例
        f.__name__ = f"{type(scalar_validator).__name__}List"
    # 设置内部函数 f 的全限定名称，保留其父函数的限定部分
    f.__qualname__ = f.__qualname__.rsplit(".", 1)[0] + "." + f.__name__
    # 如果提供了文档字符串，则将其设置为内部函数 f 的文档
    f.__doc__ = doc if doc is not None else scalar_validator.__doc__
    return f


# 定义一个简单的验证函数，接受任何输入并直接返回
def validate_any(s):
    return s

# 创建一个 validate_any 函数的列表版本，使用 _listify_validator 装饰器
validate_anylist = _listify_validator(validate_any)


# 定义一个验证日期的函数，尝试将输入转换为 np.datetime64 对象
def _validate_date(s):
    try:
        np.datetime64(s)
        return s
    except ValueError:
        raise ValueError(
            f'{s!r} should be a string that can be parsed by numpy.datetime64')


# 定义一个验证布尔值的函数，将输入转换为布尔值或引发 ValueError 异常
def validate_bool(b):
    """Convert b to ``bool`` or raise."""
    if isinstance(b, str):
        b = b.lower()
    # 判断输入是否为布尔值的不同表示形式，若是则返回 True 或 False
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        # 若无法转换为布尔值，则引发 ValueError 异常
        raise ValueError(f'Cannot convert {b!r} to bool')


# 定义一个验证轴位置的函数，尝试将输入转换为布尔值，或者是特定字符串 'line'
def validate_axisbelow(s):
    try:
        return validate_bool(s)
    except ValueError:
        if isinstance(s, str):
            if s == 'line':
                return 'line'
    # 若无法解释为 True, False, 或 "line"，则引发 ValueError 异常
    raise ValueError(f'{s!r} cannot be interpreted as'
                     ' True, False, or "line"')


# 定义一个验证 DPI（每英寸点数）的函数，确认输入为字符串 'figure' 或者尝试将输入转换为浮点数
def validate_dpi(s):
    """Confirm s is string 'figure' or convert s to float or raise."""
    if s == 'figure':
        return s
    try:
        return float(s)
    # 若无法转换为浮点数，则保持引发的异常不变
    # 如果捕获到 ValueError 异常，则抛出一个新的 ValueError 异常。
    # 抛出的异常信息包括原始字符串 s 的表达形式以及说明 s 不是字符串 "figure"，
    # 且无法将其转换为浮点数的原因。
    except ValueError as e:
        raise ValueError(f'{s!r} is not string "figure" and '
                         f'could not convert {s!r} to float') from e
def _make_type_validator(cls, *, allow_none=False):
    """
    Return a validator that converts inputs to *cls* or raises (and possibly
    allows ``None`` as well).
    """

    def validator(s):
        # 如果允许值为 None 并且输入是 None 或者字符串 "none"（忽略大小写），则返回 None
        if (allow_none and
                (s is None or cbook._str_lower_equal(s, "none"))):
            return None
        # 如果 cls 是 str 类型但输入不是字符串，则抛出 ValueError 异常
        if cls is str and not isinstance(s, str):
            raise ValueError(f'Could not convert {s!r} to str')
        try:
            # 尝试将输入 s 转换为 cls 类型，若转换失败则抛出相应异常
            return cls(s)
        except (TypeError, ValueError) as e:
            # 若转换失败，则抛出包含详细信息的 ValueError 异常
            raise ValueError(
                f'Could not convert {s!r} to {cls.__name__}') from e

    # 设置 validator 函数的名称为 validate_{cls.__name__}，若允许 None 还要追加 "_or_None"
    validator.__name__ = f"validate_{cls.__name__}"
    if allow_none:
        validator.__name__ += "_or_None"
    # 设置 validator 函数的限定名称以保证其在给定作用域中唯一
    validator.__qualname__ = (
        validator.__qualname__.rsplit(".", 1)[0] + "." + validator.__name__)
    return validator


# 创建一个字符串类型的验证器
validate_string = _make_type_validator(str)
# 创建一个允许 None 的字符串类型的验证器
validate_string_or_None = _make_type_validator(str, allow_none=True)
# 创建一个验证器，将单个字符串转换为字符串列表，并附带文档说明
validate_stringlist = _listify_validator(
    validate_string, doc='return a list of strings')
# 创建一个整数类型的验证器
validate_int = _make_type_validator(int)
# 创建一个允许 None 的整数类型的验证器
validate_int_or_None = _make_type_validator(int, allow_none=True)
# 创建一个浮点数类型的验证器
validate_float = _make_type_validator(float)
# 创建一个允许 None 的浮点数类型的验证器
validate_float_or_None = _make_type_validator(float, allow_none=True)
# 创建一个验证器，将单个浮点数转换为浮点数列表，并附带文档说明
validate_floatlist = _listify_validator(
    validate_float, doc='return a list of floats')


def _validate_marker(s):
    try:
        # 尝试将输入 s 转换为整数类型
        return validate_int(s)
    except ValueError as e:
        try:
            # 转换失败时尝试将输入 s 转换为字符串类型
            return validate_string(s)
        except ValueError as e:
            # 若均转换失败，则抛出详细的 ValueError 异常
            raise ValueError('Supported markers are [string, int]') from e


# 创建一个验证器，将单个标记（marker）转换为标记列表，并附带文档说明
_validate_markerlist = _listify_validator(
    _validate_marker, doc='return a list of markers')


def _validate_pathlike(s):
    if isinstance(s, (str, os.PathLike)):
        # 若输入 s 是字符串或 os.PathLike 类型，则解码并返回其字符串表示形式
        # 这是因为 savefig.directory 需要区分 ""（当前工作目录）和 "."（当前工作目录，但可由用户选择更新）
        return os.fsdecode(s)
    else:
        # 否则，将输入 s 当作字符串进行验证
        return validate_string(s)


def validate_fonttype(s):
    """
    Confirm that this is a Postscript or PDF font type that we know how to
    convert to.
    """
    # 定义支持的字体类型及其对应的值
    fonttypes = {'type3':    3,
                 'truetype': 42}
    try:
        # 尝试将输入 s 转换为整数类型
        fonttype = validate_int(s)
    except ValueError:
        try:
            # 转换失败时，将输入 s 转换为小写后查找对应的字体类型值
            return fonttypes[s.lower()]
        except KeyError as e:
            # 若查找失败，则抛出详细的 ValueError 异常
            raise ValueError('Supported Postscript/PDF font types are %s'
                             % list(fonttypes)) from e
    else:
        # 若转换成功，但结果不在支持的字体类型值列表中，则抛出详细的 ValueError 异常
        if fonttype not in fonttypes.values():
            raise ValueError(
                'Supported Postscript/PDF font types are %s' %
                list(fonttypes.values()))
        return fonttype


# 定义一个特殊的对象作为后端验证器的标志
_auto_backend_sentinel = object()


def validate_backend(s):
    # 若输入 s 是 _auto_backend_sentinel 对象或者在后端注册表中有效，则返回输入 s
    if s is _auto_backend_sentinel or backend_registry.is_valid_backend(s):
        return s
    else:
        # 构建错误消息，指出给定的 's' 不是有效的后端值，列出所有支持的后端值
        msg = (f"'{s}' is not a valid value for backend; supported values are "
               f"{backend_registry.list_all()}")
        # 抛出值错误，将错误消息作为异常信息
        raise ValueError(msg)
def _validate_toolbar(s):
    # 使用 ValidateInStrings 类验证 s 是否在指定的字符串列表中，并忽略大小写
    s = ValidateInStrings(
        'toolbar', ['None', 'toolbar2', 'toolmanager'], ignorecase=True)(s)
    # 如果 s 是 'toolmanager'，则发出警告信息
    if s == 'toolmanager':
        _api.warn_external(
            "Treat the new Tool classes introduced in v1.5 as experimental "
            "for now; the API and rcParam may change in future versions.")
    return s


def validate_color_or_inherit(s):
    """Return a valid color arg."""
    # 如果 s 是 'inherit'，则直接返回 'inherit'
    if cbook._str_equal(s, 'inherit'):
        return s
    # 否则调用 validate_color 函数处理 s
    return validate_color(s)


def validate_color_or_auto(s):
    # 如果 s 是 'auto'，则直接返回 'auto'
    if cbook._str_equal(s, 'auto'):
        return s
    # 否则调用 validate_color 函数处理 s
    return validate_color(s)


def validate_color_for_prop_cycle(s):
    # 如果 s 是字符串且匹配 "^C[0-9]$" 的正则表达式，则抛出异常
    if isinstance(s, str) and re.match("^C[0-9]$", s):
        raise ValueError(f"Cannot put cycle reference ({s!r}) in prop_cycler")
    # 否则调用 validate_color 函数处理 s
    return validate_color(s)


def _validate_color_or_linecolor(s):
    # 如果 s 是 'linecolor'，则直接返回 'linecolor'
    if cbook._str_equal(s, 'linecolor'):
        return s
    # 如果 s 是 'mfc' 或 'markerfacecolor'，则返回 'markerfacecolor'
    elif cbook._str_equal(s, 'mfc') or cbook._str_equal(s, 'markerfacecolor'):
        return 'markerfacecolor'
    # 如果 s 是 'mec' 或 'markeredgecolor'，则返回 'markeredgecolor'
    elif cbook._str_equal(s, 'mec') or cbook._str_equal(s, 'markeredgecolor'):
        return 'markeredgecolor'
    # 如果 s 是 None 或者是长度为 6 或 8 的字符串，则尝试转换为颜色值
    elif s is None or isinstance(s, str) and len(s) == 6 or len(s) == 8:
        stmp = '#' + s
        # 如果 stmp 是颜色值，则返回 stmp
        if is_color_like(stmp):
            return stmp
        # 如果 s 是 'none'，则返回 None
        if s.lower() == 'none':
            return None
    # 如果 s 是颜色值，则直接返回 s
    elif is_color_like(s):
        return s

    # 如果以上条件都不满足，则抛出 ValueError 异常
    raise ValueError(f'{s!r} does not look like a color arg')


def validate_color(s):
    """Return a valid color arg."""
    # 如果 s 是字符串，并且是 'none'，则返回 'none'
    if isinstance(s, str):
        if s.lower() == 'none':
            return 'none'
        # 如果 s 的长度为 6 或 8，则加上 '#' 并尝试转换为颜色值
        if len(s) == 6 or len(s) == 8:
            stmp = '#' + s
            if is_color_like(stmp):
                return stmp

    # 如果 s 是颜色值，则直接返回 s
    if is_color_like(s):
        return s

    # 如果仍然有效，则 s 必须是 matplotlibrc 中作为字符串的元组
    try:
        color = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        pass
    else:
        if is_color_like(color):
            return color

    # 如果以上条件都不满足，则抛出 ValueError 异常
    raise ValueError(f'{s!r} does not look like a color arg')


validate_colorlist = _listify_validator(
    validate_color, allow_stringlist=True, doc='return a list of colorspecs')


def _validate_cmap(s):
    # 检查 s 是 str 类型或 Colormap 类型
    _api.check_isinstance((str, Colormap), cmap=s)
    return s


def validate_aspect(s):
    # 如果 s 是 'auto' 或 'equal'，则直接返回 s
    if s in ('auto', 'equal'):
        return s
    # 否则尝试将 s 转换为 float 类型，如果失败则抛出异常
    try:
        return float(s)
    except ValueError as e:
        raise ValueError('not a valid aspect specification') from e


def validate_fontsize_None(s):
    # 如果 s 是 None 或 'None'，则返回 None
    if s is None or s == 'None':
        return None
    else:
        # 否则调用 validate_fontsize 函数处理 s
        return validate_fontsize(s)


def validate_fontsize(s):
    fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
                 'x-large', 'xx-large', 'smaller', 'larger']
    # 如果 s 是字符串，则转换为小写
    if isinstance(s, str):
        s = s.lower()
    # 如果 s 在 fontsizes 列表中，则返回 s
    if s in fontsizes:
        return s
    # 否则尝试将 s 转换为 float 类型，如果成功则返回转换后的值
    try:
        return float(s)
    except ValueError:
        # 如果转换失败，则抛出 ValueError 异常
        raise ValueError(f'{s!r} does not look like a fontsize')
    # 如果捕获到 ValueError 异常，则重新抛出 ValueError 异常，并附加详细错误信息
    except ValueError as e:
        raise ValueError("%s is not a valid font size. Valid font sizes "
                         "are %s." % (s, ", ".join(fontsizes))) from e
# 将 validate_fontsize 函数应用于列表的每个元素，返回验证后的列表
validate_fontsizelist = _listify_validator(validate_fontsize)


def validate_fontweight(s):
    # 定义标准的字体粗细列表
    weights = [
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black']
    # 注意：在 Matplotlib 中，历史上字体粗细是区分大小写的
    if s in weights:
        return s
    try:
        return int(s)  # 尝试将 s 转换为整数
    except (ValueError, TypeError) as e:
        raise ValueError(f'{s} is not a valid font weight.') from e  # 抛出值错误，指出 s 不是有效的字体粗细


def validate_fontstretch(s):
    # 定义标准的字体拉伸值列表
    stretchvalues = [
        'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed',
        'normal', 'semi-expanded', 'expanded', 'extra-expanded',
        'ultra-expanded']
    # 注意：在 Matplotlib 中，历史上字体拉伸值是区分大小写的
    if s in stretchvalues:
        return s
    try:
        return int(s)  # 尝试将 s 转换为整数
    except (ValueError, TypeError) as e:
        raise ValueError(f'{s} is not a valid font stretch.') from e  # 抛出值错误，指出 s 不是有效的字体拉伸值


def validate_font_properties(s):
    parse_fontconfig_pattern(s)  # 解析字体配置模式
    return s


def _validate_mathtext_fallback(s):
    _fallback_fonts = ['cm', 'stix', 'stixsans']  # 可接受的数学文本回退字体列表
    if isinstance(s, str):
        s = s.lower()  # 如果 s 是字符串，转换为小写
    if s is None or s == 'none':
        return None  # 如果 s 是 None 或者 'none'，返回 None
    elif s.lower() in _fallback_fonts:
        return s  # 如果 s 在回退字体列表中，则返回 s
    else:
        raise ValueError(
            f"{s} is not a valid fallback font name. Valid fallback font "
            f"names are {','.join(_fallback_fonts)}. Passing 'None' will turn "
            "fallback off.")  # 抛出值错误，指出 s 不是有效的回退字体名称


def validate_whiskers(s):
    try:
        return _listify_validator(validate_float, n=2)(s)  # 使用 validate_float 函数验证列表中的两个值
    except (TypeError, ValueError):
        try:
            return float(s)  # 尝试将 s 转换为浮点数
        except ValueError as e:
            raise ValueError("Not a valid whisker value [float, "
                             "(float, float)]") from e  # 抛出值错误，指出 s 不是有效的“whisker”值


def validate_ps_distiller(s):
    if isinstance(s, str):
        s = s.lower()  # 如果 s 是字符串，转换为小写
    if s in ('none', None, 'false', False):
        return None  # 如果 s 是 'none', None, 'false' 或 False，则返回 None
    else:
        return ValidateInStrings('ps.usedistiller', ['ghostscript', 'xpdf'])(s)  # 使用 ValidateInStrings 进行验证


def _validate_papersize(s):
    # 重新内联此验证器，当 'auto' 不再被推荐使用时
    s = ValidateInStrings("ps.papersize",
                          ["figure", "auto", "letter", "legal", "ledger",
                           *[f"{ab}{i}" for ab in "ab" for i in range(11)]],
                          ignorecase=True)(s)  # 使用 ValidateInStrings 进行验证，忽略大小写
    if s == "auto":
        _api.warn_deprecated("3.8", name="ps.papersize='auto'",
                             addendum="Pass an explicit paper type, figure, or omit "
                             "the *ps.papersize* rcParam entirely.")  # 发出警告，推荐传递明确的纸张类型或者省略 ps.papersize
    return s


# 专门用于命名线型验证的验证器，基于 ls_mapper 中的条目，以及从 Line2D.set_linestyle 读取的可能字符串列表
_validate_named_linestyle = ValidateInStrings(
    'linestyle',
    [*ls_mapper.keys(), *ls_mapper.values(), 'None', 'none', ' ', ''],
    ignorecase=True)  # 使用 ValidateInStrings 进行验证，忽略大小写
    # 设置 ignorecase 参数为 True，表示在比较字符串时忽略大小写
    ignorecase=True
def _validate_linestyle(ls):
    """
    A validator for all possible line styles, the named ones *and*
    the on-off ink sequences.
    """
    # 如果输入的 ls 是字符串类型
    if isinstance(ls, str):
        try:  # 首先尝试验证是否为有效的命名线条样式，如 '--' 或 'solid'。
            return _validate_named_linestyle(ls)
        except ValueError:
            pass
        try:
            ls = ast.literal_eval(ls)  # 解析 matplotlibrc。
        except (SyntaxError, ValueError):
            pass  # 最终会抛出 ValueError 异常。

    def _is_iterable_not_string_like(x):
        # 明确排除 bytes/bytearrays，以免将它们错误地解释为数字序列（码点）。
        return np.iterable(x) and not isinstance(x, (str, bytes, bytearray))

    # 如果 ls 是可迭代且不类似于字符串
    if _is_iterable_not_string_like(ls):
        if len(ls) == 2 and _is_iterable_not_string_like(ls[1]):
            # (offset, (on, off, on, off, ...))
            offset, onoff = ls
        else:
            # 对于向后兼容性：(on, off, on, off, ...)；偏移量是隐含的。
            offset = 0
            onoff = ls

        if (isinstance(offset, Real)
                and len(onoff) % 2 == 0
                and all(isinstance(elem, Real) for elem in onoff)):
            return (offset, onoff)

    # 如果无法验证为有效的线条样式，抛出 ValueError 异常
    raise ValueError(f"linestyle {ls!r} is not a valid on-off ink sequence.")


validate_fillstyle = ValidateInStrings(
    'markers.fillstyle', ['full', 'left', 'right', 'bottom', 'top', 'none'])

# 对填充样式进行验证，应在指定的字符串集合中
validate_fillstylelist = _listify_validator(validate_fillstyle)


def validate_markevery(s):
    """
    Validate the markevery property of a Line2D object.

    Parameters
    ----------
    s : None, int, (int, int), slice, float, (float, float), or list[int]

    Returns
    -------
    None, int, (int, int), slice, float, (float, float), or list[int]
    """
    # 验证 s 是否为 slice、float、int 或 None 类型
    if isinstance(s, (slice, float, int, type(None))):
        return s
    # 验证 s 是否为 tuple 类型
    if isinstance(s, tuple):
        if (len(s) == 2
                and (all(isinstance(e, int) for e in s)
                     or all(isinstance(e, float) for e in s))):
            return s
        else:
            raise TypeError(
                "'markevery' tuple must be pair of ints or of floats")
    # 验证 s 是否为 list 类型
    if isinstance(s, list):
        if all(isinstance(e, int) for e in s):
            return s
        else:
            raise TypeError(
                "'markevery' list must have all elements of type int")
    # 如果类型不在预期范围内，抛出 TypeError 异常
    raise TypeError("'markevery' is of an invalid type")


validate_markeverylist = _listify_validator(validate_markevery)


def validate_bbox(s):
    # 如果 s 是字符串类型
    if isinstance(s, str):
        s = s.lower()
        # 如果 s 是 'tight'，返回 s
        if s == 'tight':
            return s
        # 如果 s 是 'standard'，返回 None
        if s == 'standard':
            return None
        # 否则抛出 ValueError 异常
        raise ValueError("bbox should be 'tight' or 'standard'")
    elif s is not None:
        # 如果 s 不是 None，则执行以下语句，这段代码用于向后兼容。
        # 在此情况下，None 等同于 'standard'。
        raise ValueError("bbox should be 'tight' or 'standard'")
    # 返回变量 s 的值
    return s
def validate_sketch(s):
    # 如果输入是字符串，转换为小写并去除首尾空格
    if isinstance(s, str):
        s = s.lower().strip()
        # 如果字符串以 '(' 开头且以 ')' 结尾，去除首尾括号
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
    # 如果字符串是 'none' 或者 None，则返回 None
    if s == 'none' or s is None:
        return None
    try:
        # 使用 validate_float 函数验证 s，并将其转换为包含三个元素的元组
        return tuple(_listify_validator(validate_float, n=3)(s))
    except ValueError as exc:
        # 如果验证失败，则抛出异常
        raise ValueError("Expected a (scale, length, randomness) tuple") from exc


def _validate_greaterthan_minushalf(s):
    # 验证 s 是一个浮点数，并且大于 -0.5
    s = validate_float(s)
    if s > -0.5:
        return s
    else:
        raise RuntimeError(f'Value must be >-0.5; got {s}')


def _validate_greaterequal0_lessequal1(s):
    # 验证 s 是一个浮点数，并且在 [0, 1] 范围内
    s = validate_float(s)
    if 0 <= s <= 1:
        return s
    else:
        raise RuntimeError(f'Value must be >=0 and <=1; got {s}')


def _validate_int_greaterequal0(s):
    # 验证 s 是一个整数，并且大于等于 0
    s = validate_int(s)
    if s >= 0:
        return s
    else:
        raise RuntimeError(f'Value must be >=0; got {s}')


def validate_hatch(s):
    r"""
    Validate a hatch pattern.
    A hatch pattern string can have any sequence of the following
    characters: ``\ / | - + * . x o O``.
    """
    # 验证输入是否为字符串
    if not isinstance(s, str):
        raise ValueError("Hatch pattern must be a string")
    # 检查输入字符串中是否包含未知的图案字符
    _api.check_isinstance(str, hatch_pattern=s)
    unknown = set(s) - {'\\', '/', '|', '-', '+', '*', '.', 'x', 'o', 'O'}
    if unknown:
        raise ValueError("Unknown hatch symbol(s): %s" % list(unknown))
    return s


validate_hatchlist = _listify_validator(validate_hatch)
validate_dashlist = _listify_validator(validate_floatlist)


def _validate_minor_tick_ndivs(n):
    """
    Validate ndiv parameter related to the minor ticks.
    It controls the number of minor ticks to be placed between
    two major ticks.
    """
    # 如果 n 是字符串 'auto'，直接返回
    if cbook._str_lower_equal(n, 'auto'):
        return n
    try:
        # 否则，验证 n 是非负整数
        n = _validate_int_greaterequal0(n)
        return n
    except (RuntimeError, ValueError):
        pass

    # 如果验证失败，则抛出异常
    raise ValueError("'tick.minor.ndivs' must be 'auto' or non-negative int")


_prop_validators = {
    'color': _listify_validator(validate_color_for_prop_cycle,
                                allow_stringlist=True),
    'linewidth': validate_floatlist,
    'linestyle': _listify_validator(_validate_linestyle),
    'facecolor': validate_colorlist,
    'edgecolor': validate_colorlist,
    'joinstyle': _listify_validator(JoinStyle),
    'capstyle': _listify_validator(CapStyle),
    'fillstyle': validate_fillstylelist,
    'markerfacecolor': validate_colorlist,
    'markersize': validate_floatlist,
    'markeredgewidth': validate_floatlist,
    'markeredgecolor': validate_colorlist,
    'markevery': validate_markeverylist,
    'alpha': validate_floatlist,
    'marker': _validate_markerlist,
    'hatch': validate_hatchlist,  # 使用 validate_hatchlist 验证 'hatch' 属性
    'dashes': validate_dashlist,  # 使用 validate_dashlist 验证 'dashes' 属性
}
# 属性名称的别名映射字典，用于简化常见属性名的输入
_prop_aliases = {
    'c': 'color',                   # 'c' 表示 'color'
    'lw': 'linewidth',              # 'lw' 表示 'linewidth'
    'ls': 'linestyle',              # 'ls' 表示 'linestyle'
    'fc': 'facecolor',              # 'fc' 表示 'facecolor'
    'ec': 'edgecolor',              # 'ec' 表示 'edgecolor'
    'mfc': 'markerfacecolor',       # 'mfc' 表示 'markerfacecolor'
    'mec': 'markeredgecolor',       # 'mec' 表示 'markeredgecolor'
    'mew': 'markeredgewidth',       # 'mew' 表示 'markeredgewidth'
    'ms': 'markersize',             # 'ms' 表示 'markersize'
}


def cycler(*args, **kwargs):
    """
    Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,
    but includes input validation.

    Call signatures::

      cycler(cycler)
      cycler(label=values[, label2=values2[, ...]])
      cycler(label, values)

    Form 1 copies a given `~cycler.Cycler` object.

    Form 2 creates a `~cycler.Cycler` which cycles over one or more
    properties simultaneously. If multiple properties are given, their
    value lists must have the same length.

    Form 3 creates a `~cycler.Cycler` for a single property. This form
    exists for compatibility with the original cycler. Its use is
    discouraged in favor of the kwarg form, i.e. ``cycler(label=values)``.

    Parameters
    ----------
    cycler : Cycler
        Copy constructor for Cycler.

    label : str
        The property key. Must be a valid `.Artist` property.
        For example, 'color' or 'linestyle'. Aliases are allowed,
        such as 'c' for 'color' and 'lw' for 'linewidth'.

    values : iterable
        Finite-length iterable of the property values. These values
        are validated and will raise a ValueError if invalid.

    Returns
    -------
    Cycler
        A new :class:`~cycler.Cycler` for the given properties.

    Examples
    --------
    Creating a cycler for a single property:

    >>> c = cycler(color=['red', 'green', 'blue'])

    Creating a cycler for simultaneously cycling over multiple properties
    (e.g. red circle, green plus, blue cross):

    >>> c = cycler(color=['red', 'green', 'blue'],
    ...            marker=['o', '+', 'x'])

    """
    # 检查参数的组合方式是否符合要求
    if args and kwargs:
        raise TypeError("cycler() can only accept positional OR keyword "
                        "arguments -- not both.")
    elif not args and not kwargs:
        raise TypeError("cycler() must have positional OR keyword arguments")

    # 处理只有一个参数的情况，参数必须是 Cycler 实例，返回验证后的 Cycler 对象
    if len(args) == 1:
        if not isinstance(args[0], Cycler):
            raise TypeError("If only one positional argument given, it must "
                            "be a Cycler instance.")
        return validate_cycler(args[0])
    # 处理有两个参数的情况，将其作为键值对列表存储
    elif len(args) == 2:
        pairs = [(args[0], args[1])]
    # 处理超过两个参数的情况，引发参数数量错误
    elif len(args) > 2:
        raise _api.nargs_error('cycler', '0-2', len(args))
    else:
        # 否则将关键字参数转换为键值对列表
        pairs = kwargs.items()

    # 存储经过验证的属性键值对列表
    validated = []
    # 遍历属性和对应的值对
    for prop, vals in pairs:
        # 根据属性名获取标准化后的属性名，如果没有别名则返回原属性名
        norm_prop = _prop_aliases.get(prop, prop)
        # 根据标准化后的属性名获取验证器函数，如果不存在则抛出类型错误异常
        validator = _prop_validators.get(norm_prop, None)
        if validator is None:
            raise TypeError("Unknown artist property: %s" % prop)
        # 使用获取到的验证器函数对属性值进行验证和处理
        vals = validator(vals)
        # 将标准化后的属性名和验证后的属性值组成元组，并添加到验证后的列表中
        validated.append((norm_prop, vals))

    # 使用生成器表达式将验证后的列表中的元组展开成键值对，并使用 reduce 函数将它们合并为单个列表
    return reduce(operator.add, (ccycler(k, v) for k, v in validated))
# 定义一个类，用于检查AST节点中的属性，如果属性以双下划线开头和结尾，则引发异常
class _DunderChecker(ast.NodeVisitor):
    def visit_Attribute(self, node):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise ValueError("cycler strings with dunders are forbidden")
        self.generic_visit(node)


# 创建一个专门验证 legend.loc 参数的验证器
_validate_named_legend_loc = ValidateInStrings(
    'legend.loc',
    [
        "best",
        "upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center",
        "center"],
    ignorecase=True)


def _validate_legend_loc(loc):
    """
    确认 loc 参数是否是 rc.Params["legend.loc"] 支持的类型。

    .. versionadded:: 3.8

    Parameters
    ----------
    loc : str | int | (float, float) | str((float, float))
        图例位置的表示形式。

    Returns
    -------
    loc : str | int | (float, float) or raise ValueError exception
        图例位置。
    """
    if isinstance(loc, str):
        try:
            return _validate_named_legend_loc(loc)
        except ValueError:
            pass
        try:
            loc = ast.literal_eval(loc)
        except (SyntaxError, ValueError):
            pass
    if isinstance(loc, int):
        if 0 <= loc <= 10:
            return loc
    if isinstance(loc, tuple):
        if len(loc) == 2 and all(isinstance(e, Real) for e in loc):
            return loc
    # 如果 loc 类型不在支持的范围内，引发 ValueError 异常
    raise ValueError(f"{loc} is not a valid legend location.")


def validate_cycler(s):
    """从字符串表示或对象本身返回一个 Cycler 对象。"""
    if isinstance(s, str):
        # TODO: 我们可能需要重新考虑这一点...
        # 虽然我认为已经很安全了，但执行未经过清洗的任意代码是危险的。
        # 考虑到 rcparams 可能来自互联网（未来计划），这可能非常危险。
        # 我通过仅允许 'cycler()' 函数来锁定了它。
        # 更新：部分修补了一个安全漏洞。
        # 我真的应该读过这篇文章：
        # https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
        # 我们应该用 PyParsing 和 ast.literal_eval() 的组合替换这个 eval。
        try:
            _DunderChecker().visit(ast.parse(s))
            s = eval(s, {'cycler': cycler, '__builtins__': {}})
        except BaseException as e:
            raise ValueError(f"{s!r} is not a valid cycler construction: {e}"
                             ) from e
    # 确保上述 eval() 返回的是一个 Cycler 对象。
    if isinstance(s, Cycler):
        cycler_inst = s
    else:
        # 如果不是字符串或 Cycler 实例，则引发 ValueError 异常。
        raise ValueError(f"Object is not a string or Cycler instance: {s!r}")

    # 检查 cycler_inst 中的未知属性是否在已知的属性验证器中，否则引发异常。
    unknowns = cycler_inst.keys - (set(_prop_validators) | set(_prop_aliases))
    if unknowns:
        raise ValueError("Unknown artist properties: %s" % unknowns)
    # Not a full validation, but it'll at least normalize property names
    # A fuller validation would require v0.10 of cycler.
    # 创建一个空的集合用于检查属性名的唯一性和规范化
    checker = set()
    
    # 遍历循环对象的所有属性名
    for prop in cycler_inst.keys:
        # 根据别名字典尝试规范化属性名
        norm_prop = _prop_aliases.get(prop, prop)
        
        # 如果规范化后的属性名与原属性名不同且已经存在于属性集合中，则抛出异常
        if norm_prop != prop and norm_prop in cycler_inst.keys:
            raise ValueError(f"Cannot specify both {norm_prop!r} and alias "
                             f"{prop!r} in the same prop_cycle")
        
        # 如果规范化后的属性名已经在检查器集合中存在，则抛出异常
        if norm_prop in checker:
            raise ValueError(f"Another property was already aliased to "
                             f"{norm_prop!r}. Collision normalizing {prop!r}.")
        
        # 将规范化后的属性名添加到检查器集合中
        checker.update([norm_prop])

    # 确保检查器集合中的元素数量与循环对象属性名数量相同，以确保所有属性名都是唯一的
    assert len(checker) == len(cycler_inst.keys)

    # 现在可以安全地修改循环对象的属性名
    # 遍历循环对象的所有属性名
    for prop in cycler_inst.keys:
        # 根据别名字典尝试规范化属性名
        norm_prop = _prop_aliases.get(prop, prop)
        
        # 修改循环对象的属性名为规范化后的属性名
        cycler_inst.change_key(prop, norm_prop)

    # 遍历循环对象按属性分组的键值对，对每组属性值进行验证
    for key, vals in cycler_inst.by_key().items():
        # 使用属性验证器函数验证属性值
        _prop_validators[key](vals)

    # 返回已经修改后的循环对象
    return cycler_inst
def validate_hist_bins(s):
    # 定义有效的字符串列表
    valid_strs = ["auto", "sturges", "fd", "doane", "scott", "rice", "sqrt"]
    # 如果输入是字符串且在有效字符串列表中，则直接返回该字符串
    if isinstance(s, str) and s in valid_strs:
        return s
    # 尝试将输入转换为整数，如果成功则返回整数值
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    # 尝试调用 validate_floatlist 函数，如果成功则返回其结果
    try:
        return validate_floatlist(s)
    except ValueError:
        pass
    # 如果无法转换成有效的类型，抛出 ValueError 异常
    raise ValueError(f"'hist.bins' must be one of {valid_strs}, an int or"
                     " a sequence of floats")


class _ignorecase(list):
    """A marker class indicating that a list-of-str is case-insensitive."""


def _convert_validator_spec(key, conv):
    # 如果 conv 是列表，则检查是否是 _ignorecase 类型，然后返回相应的验证器
    if isinstance(conv, list):
        ignorecase = isinstance(conv, _ignorecase)
        return ValidateInStrings(key, conv, ignorecase=ignorecase)
    else:
        # 如果 conv 不是列表，则直接返回其本身
        return conv


# Mapping of rcParams to validators.
# Converters given as lists or _ignorecase are converted to ValidateInStrings
# immediately below.
# The rcParams defaults are defined in lib/matplotlib/mpl-data/matplotlibrc, which
# gets copied to matplotlib/mpl-data/matplotlibrc by the setup script.
_validators = {
    "backend":           validate_backend,  # 验证后端
    "backend_fallback":  validate_bool,  # 验证后备后端的布尔值
    "figure.hooks":      validate_stringlist,  # 验证字符串列表
    "toolbar":           _validate_toolbar,  # 验证工具栏
    "interactive":       validate_bool,  # 验证交互模式的布尔值
    "timezone":          validate_string,  # 验证时区字符串

    "webagg.port":            validate_int,  # 验证整数端口号
    "webagg.address":         validate_string,  # 验证字符串地址
    "webagg.open_in_browser": validate_bool,  # 验证打开浏览器的布尔值
    "webagg.port_retries":    validate_int,  # 验证整数端口重试次数

    # line props
    "lines.linewidth":       validate_float,  # 验证线宽（点数）
    "lines.linestyle":       _validate_linestyle,  # 验证线型样式（实线）
    "lines.color":           validate_color,  # 验证颜色循环中的第一个颜色
    "lines.marker":          _validate_marker,  # 验证标记名称
    "lines.markerfacecolor": validate_color_or_auto,  # 验证标记面颜色或自动
    "lines.markeredgecolor": validate_color_or_auto,  # 验证标记边缘颜色或自动
    "lines.markeredgewidth": validate_float,  # 验证标记边缘宽度
    "lines.markersize":      validate_float,  # 验证标记大小（点数）
    "lines.antialiased":     validate_bool,  # 验证抗锯齿效果的布尔值
    "lines.dash_joinstyle":  JoinStyle,  # 验证虚线连接样式
    "lines.solid_joinstyle": JoinStyle,  # 验证实线连接样式
    "lines.dash_capstyle":   CapStyle,  # 验证虚线端点样式
    "lines.solid_capstyle":  CapStyle,  # 验证实线端点样式
    "lines.dashed_pattern":  validate_floatlist,  # 验证虚线样式模式列表
    "lines.dashdot_pattern": validate_floatlist,  # 验证点划线样式模式列表
    "lines.dotted_pattern":  validate_floatlist,  # 验证点线样式模式列表
    "lines.scale_dashes":    validate_bool,  # 验证是否缩放虚线间隔

    # marker props
    "markers.fillstyle": validate_fillstyle,  # 验证填充样式

    ## pcolor(mesh) props:
    "pcolor.shading": ["auto", "flat", "nearest", "gouraud"],  # 验证填充颜色渐变方式
    "pcolormesh.snap": validate_bool,  # 验证是否对网格进行快照

    ## patch props
    "patch.linewidth":       validate_float,  # 验证路径宽度（点数）
    "patch.edgecolor":       validate_color,  # 验证边缘颜色
    "patch.force_edgecolor": validate_bool,  # 验证是否强制使用边缘颜色
    "patch.facecolor":       validate_color,  # 验证填充颜色循环中的第一个颜色
    "patch.antialiased":     validate_bool,  # 验证抗锯齿效果的布尔值
    # hatch props
    "hatch.color":     validate_color,   # Validate and set the hatch color
    "hatch.linewidth": validate_float,   # Validate and set the hatch linewidth

    # Histogram properties
    "hist.bins": validate_hist_bins,     # Validate and set the histogram bin count

    # Boxplot properties
    "boxplot.notch":       validate_bool,       # Validate and set whether to notch the boxplot
    "boxplot.vertical":    validate_bool,       # Validate and set orientation of the boxplot
    "boxplot.whiskers":    validate_whiskers,   # Validate and set style of whiskers in boxplot
    "boxplot.bootstrap":   validate_int_or_None,# Validate and set whether to bootstrap in boxplot
    "boxplot.patchartist": validate_bool,       # Validate and set whether to use patch artist in boxplot
    "boxplot.showmeans":   validate_bool,       # Validate and set whether to show means in boxplot
    "boxplot.showcaps":    validate_bool,       # Validate and set whether to show caps in boxplot
    "boxplot.showbox":     validate_bool,       # Validate and set whether to show box in boxplot
    "boxplot.showfliers":  validate_bool,       # Validate and set whether to show fliers in boxplot
    "boxplot.meanline":    validate_bool,       # Validate and set whether to show mean line in boxplot

    "boxplot.flierprops.color":           validate_color,        # Validate and set flier color in boxplot
    "boxplot.flierprops.marker":          _validate_marker,      # Validate and set marker style for fliers in boxplot
    "boxplot.flierprops.markerfacecolor": validate_color_or_auto,# Validate and set marker face color for fliers in boxplot
    "boxplot.flierprops.markeredgecolor": validate_color,        # Validate and set marker edge color for fliers in boxplot
    "boxplot.flierprops.markeredgewidth": validate_float,        # Validate and set marker edge width for fliers in boxplot
    "boxplot.flierprops.markersize":      validate_float,        # Validate and set marker size for fliers in boxplot
    "boxplot.flierprops.linestyle":       _validate_linestyle,   # Validate and set linestyle for fliers in boxplot
    "boxplot.flierprops.linewidth":       validate_float,        # Validate and set linewidth for fliers in boxplot

    "boxplot.boxprops.color":     validate_color,      # Validate and set color for box in boxplot
    "boxplot.boxprops.linewidth": validate_float,      # Validate and set linewidth for box in boxplot
    "boxplot.boxprops.linestyle": _validate_linestyle, # Validate and set linestyle for box in boxplot

    "boxplot.whiskerprops.color":     validate_color,      # Validate and set color for whiskers in boxplot
    "boxplot.whiskerprops.linewidth": validate_float,      # Validate and set linewidth for whiskers in boxplot
    "boxplot.whiskerprops.linestyle": _validate_linestyle,# Validate and set linestyle for whiskers in boxplot

    "boxplot.capprops.color":     validate_color,      # Validate and set color for caps in boxplot
    "boxplot.capprops.linewidth": validate_float,      # Validate and set linewidth for caps in boxplot
    "boxplot.capprops.linestyle": _validate_linestyle,# Validate and set linestyle for caps in boxplot

    "boxplot.medianprops.color":     validate_color,      # Validate and set color for median line in boxplot
    "boxplot.medianprops.linewidth": validate_float,      # Validate and set linewidth for median line in boxplot
    "boxplot.medianprops.linestyle": _validate_linestyle,# Validate and set linestyle for median line in boxplot

    "boxplot.meanprops.color":           validate_color,        # Validate and set color for mean marker in boxplot
    "boxplot.meanprops.marker":          _validate_marker,      # Validate and set marker style for mean marker in boxplot
    "boxplot.meanprops.markerfacecolor": validate_color,        # Validate and set marker face color for mean marker in boxplot
    "boxplot.meanprops.markeredgecolor": validate_color,        # Validate and set marker edge color for mean marker in boxplot
    "boxplot.meanprops.markersize":      validate_float,        # Validate and set marker size for mean marker in boxplot
    "boxplot.meanprops.linestyle":       _validate_linestyle,   # Validate and set linestyle for mean marker in boxplot
    "boxplot.meanprops.linewidth":       validate_float,        # Validate and set linewidth for mean marker in boxplot

    # font props
    "font.family":     validate_stringlist,  # Validate and set font family list
    "font.style":      validate_string,      # Validate and set font style
    "font.variant":    validate_string,      # Validate and set font variant
    "font.stretch":    validate_fontstretch,# Validate and set font stretch
    "font.weight":     validate_fontweight,  # Validate and set font weight
    "font.size":       validate_float,       # Validate and set base font size in points
    "font.serif":      validate_stringlist,  # Validate and set serif font family list
    "font.sans-serif": validate_stringlist,  # Validate and set sans-serif font family list
    "font.cursive":    validate_stringlist,  # Validate and set cursive font family list
    "font.fantasy":    validate_stringlist,  # Validate and set fantasy font family list
    "font.monospace":  validate_stringlist,  # Validate and set monospace font family list

    # text props
    "text.color":          validate_color,      # Validate and set text color
    "text.usetex":         validate_bool,       # Validate and set whether to use LaTeX for text rendering
    "text.latex.preamble": validate_string,     # Validate and set LaTeX preamble for text rendering
    "text.hinting":        [                    # Validate and set text hinting options
                            "default", 
                            "no_autohint", 
                            "force_autohint",
                            "no_hinting", 
                            "auto", 
                            "native", 
                            "either", 
                            "none"
                            ],
    "text.hinting_factor": validate_int,  # 验证整数类型的文本提示因子
    "text.kerning_factor": validate_int,  # 验证整数类型的字距调整因子
    "text.antialiased":    validate_bool,  # 验证布尔类型的文本抗锯齿设置
    "text.parse_math":     validate_bool,  # 验证布尔类型的数学表达式解析设置

    "mathtext.cal":            validate_font_properties,  # 验证数学文本字体属性
    "mathtext.rm":             validate_font_properties,  # 验证数学文本字体属性
    "mathtext.tt":             validate_font_properties,  # 验证数学文本字体属性
    "mathtext.it":             validate_font_properties,  # 验证数学文本字体属性
    "mathtext.bf":             validate_font_properties,  # 验证数学文本字体属性
    "mathtext.bfit":           validate_font_properties,  # 验证数学文本字体属性
    "mathtext.sf":             validate_font_properties,  # 验证数学文本字体属性
    "mathtext.fontset":        ["dejavusans", "dejavuserif", "cm", "stix",
                                "stixsans", "custom"],  # 数学文本字体集合验证
    "mathtext.default":        ["rm", "cal", "bfit", "it", "tt", "sf", "bf", "default",
                                "bb", "frak", "scr", "regular"],  # 数学文本默认设置验证
    "mathtext.fallback":       _validate_mathtext_fallback,  # 验证数学文本回退设置

    "image.aspect":              validate_aspect,  # 图像纵横比验证函数
    "image.interpolation":       validate_string,  # 图像插值方法验证
    "image.interpolation_stage": ["data", "rgba"],  # 图像插值阶段验证
    "image.cmap":                _validate_cmap,  # 图像色彩映射验证
    "image.lut":                 validate_int,  # 图像查找表验证
    "image.origin":              ["upper", "lower"],  # 图像原点位置验证
    "image.resample":            validate_bool,  # 图像重新采样验证
    "image.composite_image": validate_bool,  # 向量图形后端是否将所有图像组合成单个复合图像的验证

    # contour props
    "contour.negative_linestyle": _validate_linestyle,  # 等高线负值线条样式验证
    "contour.corner_mask":        validate_bool,  # 等高线角点遮罩验证
    "contour.linewidth":          validate_float_or_None,  # 等高线线宽验证
    "contour.algorithm":          ["mpl2005", "mpl2014", "serial", "threaded"],  # 等高线计算算法验证

    # errorbar props
    "errorbar.capsize": validate_float,  # 误差条端标尺尺寸验证

    # axis props
    # x/y 轴标题对齐方式
    "xaxis.labellocation": ["left", "center", "right"],  # X 轴标题位置验证
    "yaxis.labellocation": ["bottom", "center", "top"],  # Y 轴标题位置验证

    # Axes props
    "axes.axisbelow":        validate_axisbelow,  # 坐标轴位于数据之下验证
    "axes.facecolor":        validate_color,  # 背景颜色验证
    "axes.edgecolor":        validate_color,  # 边框颜色验证
    "axes.linewidth":        validate_float,  # 边框线宽验证

    "axes.spines.left":      validate_bool,  # 设置坐标轴脊柱的可见性
    "axes.spines.right":     validate_bool,
    "axes.spines.bottom":    validate_bool,
    "axes.spines.top":       validate_bool,

    "axes.titlesize":     validate_fontsize,  # 坐标轴标题字体大小验证
    "axes.titlelocation": ["left", "center", "right"],  # 坐标轴标题位置验证
    "axes.titleweight":   validate_fontweight,  # 坐标轴标题字体粗细验证
    "axes.titlecolor":    validate_color_or_auto,  # 坐标轴标题颜色验证
    "axes.titley":        validate_float_or_None,  # 坐标轴标题位置验证
    "axes.titlepad":      validate_float,  # 从坐标轴顶部装饰到标题的间距验证
    # 设置是否显示网格
    "axes.grid":          validate_bool,  # display grid or not

    # 设置绘制哪些网格（次要、主要或全部）
    "axes.grid.which":    ["minor", "both", "major"],  # which grids are drawn

    # 设置网格类型（x 轴、y 轴或两者）
    "axes.grid.axis":     ["x", "y", "both"],  # grid type

    # 设置 x 和 y 标签的字体大小
    "axes.labelsize":     validate_fontsize,  # fontsize of x & y labels

    # 设置标签与轴之间的间距
    "axes.labelpad":      validate_float,  # space between label and axis

    # 设置 x 和 y 标签的字体粗细
    "axes.labelweight":   validate_fontweight,  # fontsize of x & y labels

    # 设置轴标签的颜色
    "axes.labelcolor":    validate_color,  # color of axis label

    # 如果轴范围的对数小于第一个或大于第二个值，则使用科学计数法
    "axes.formatter.limits": _listify_validator(validate_int, n=2),

    # 使用当前区域设置格式化刻度
    "axes.formatter.use_locale": validate_bool,

    # 使用数学文本格式化刻度
    "axes.formatter.use_mathtext": validate_bool,

    # 在科学计数法中格式化的最小指数
    "axes.formatter.min_exponent": validate_int,

    # 是否使用偏移来格式化刻度
    "axes.formatter.useoffset": validate_bool,

    # 偏移量的阈值
    "axes.formatter.offset_threshold": validate_int,

    # 设置是否显示负号（unicode_minus）
    "axes.unicode_minus": validate_bool,

    # 设置属性循环（可以是循环对象或其字符串表示形式）
    "axes.prop_cycle": validate_cycler,

    # 自动设置轴限制模式（data 或 round_numbers）
    "axes.autolimit_mode": ["data", "round_numbers"],

    # 在 x 轴上添加的边距
    "axes.xmargin": _validate_greaterthan_minushalf,

    # 在 y 轴上添加的边距
    "axes.ymargin": _validate_greaterthan_minushalf,

    # 在 z 轴上添加的边距
    "axes.zmargin": _validate_greaterthan_minushalf,

    # 设置是否显示极坐标网格
    "polaraxes.grid":    validate_bool,

    # 设置是否显示三维图的网格
    "axes3d.grid":       validate_bool,

    # 当手动设置三维轴限制时，是否自动添加边距
    "axes3d.automargin": validate_bool,

    # 设置三维图 x 轴背景面板颜色
    "axes3d.xaxis.panecolor":    validate_color,

    # 设置三维图 y 轴背景面板颜色
    "axes3d.yaxis.panecolor":    validate_color,

    # 设置三维图 z 轴背景面板颜色
    "axes3d.zaxis.panecolor":    validate_color,

    # 设置散点图的标记样式
    "scatter.marker":     _validate_marker,

    # 设置散点图边缘颜色
    "scatter.edgecolors": validate_string,

    # 设置日期的起始时间（epoch）
    "date.epoch": _validate_date,

    # 自动格式化器设置：年、月、日、小时、分钟、秒、微秒
    "date.autoformatter.year":        validate_string,
    "date.autoformatter.month":       validate_string,
    "date.autoformatter.day":         validate_string,
    "date.autoformatter.hour":        validate_string,
    "date.autoformatter.minute":      validate_string,
    "date.autoformatter.second":      validate_string,
    "date.autoformatter.microsecond": validate_string,

    # 日期转换器设置：自动或简洁
    'date.converter':          ['auto', 'concise'],

    # 对于自动日期定位器，选择间隔倍数
    'date.interval_multiples': validate_bool,

    # 设置图例是否使用圆角边框
    "legend.fancybox": validate_bool,

    # 设置图例的位置
    "legend.loc": _validate_legend_loc
    "legend.numpoints":      validate_int,  # 验证整数，图例线上的点数，用于散点图
    "legend.scatterpoints":  validate_int,  # 验证整数，散点图中图例线上的点数
    "legend.fontsize":       validate_fontsize,  # 验证字体大小，图例字体大小
    "legend.title_fontsize": validate_fontsize_None,  # 验证字体大小或None，图例标题字体大小
    "legend.labelcolor":     _validate_color_or_linecolor,  # 验证颜色或线条颜色，图例标签颜色
    "legend.markerscale":    validate_float,  # 验证浮点数，图例标记大小相对于原始大小的比例
    "legend.shadow":         validate_bool,  # 验证布尔值，图例是否带阴影
    "legend.frameon":        validate_bool,  # 验证布尔值，图例周围是否画边框
    "legend.framealpha":     validate_float_or_None,  # 验证浮点数或None，图例边框的透明度值

    ## 下列维度均为字体大小的比例
    "legend.borderpad":      validate_float,  # 验证浮点数，边框填充大小，单位为字体大小
    "legend.labelspacing":   validate_float,  # 验证浮点数，图例条目之间的垂直间距
    "legend.handlelength":   validate_float,  # 验证浮点数，图例线的长度
    "legend.handleheight":   validate_float,  # 验证浮点数，图例线的高度
    "legend.handletextpad":  validate_float,  # 验证浮点数，图例线与文本间的间距
    "legend.borderaxespad":  validate_float,  # 验证浮点数，图例与坐标轴边缘的间距
    "legend.columnspacing":  validate_float,  # 验证浮点数，列与列之间的间距
    "legend.facecolor":      validate_color_or_inherit,  # 验证颜色或继承值，图例背景颜色
    "legend.edgecolor":      validate_color_or_inherit,  # 验证颜色或继承值，图例边框颜色

    # 刻度属性
    "xtick.top":             validate_bool,  # 验证布尔值，上方是否绘制刻度
    "xtick.bottom":          validate_bool,  # 验证布尔值，下方是否绘制刻度
    "xtick.labeltop":        validate_bool,  # 验证布尔值，上方是否显示刻度标签
    "xtick.labelbottom":     validate_bool,  # 验证布尔值，下方是否显示刻度标签
    "xtick.major.size":      validate_float,  # 验证浮点数，主刻度的大小（点数）
    "xtick.minor.size":      validate_float,  # 验证浮点数，次刻度的大小（点数）
    "xtick.major.width":     validate_float,  # 验证浮点数，主刻度的宽度（点数）
    "xtick.minor.width":     validate_float,  # 验证浮点数，次刻度的宽度（点数）
    "xtick.major.pad":       validate_float,  # 验证浮点数，刻度与标签之间的距离（点数）
    "xtick.minor.pad":       validate_float,  # 验证浮点数，次刻度与标签之间的距离（点数）
    "xtick.color":           validate_color,  # 验证颜色，刻度的颜色
    "xtick.labelcolor":      validate_color_or_inherit,  # 验证颜色或继承值，刻度标签的颜色
    "xtick.minor.visible":   validate_bool,  # 验证布尔值，是否显示次刻度
    "xtick.minor.top":       validate_bool,  # 验证布尔值，上方是否显示次刻度
    "xtick.minor.bottom":    validate_bool,  # 验证布尔值，下方是否显示次刻度
    "xtick.major.top":       validate_bool,  # 验证布尔值，上方是否显示主刻度
    "xtick.major.bottom":    validate_bool,  # 验证布尔值，下方是否显示主刻度
    "xtick.minor.ndivs":     _validate_minor_tick_ndivs,  # 验证整数，次刻度的数目
    "xtick.labelsize":       validate_fontsize,  # 验证字体大小，刻度标签的字体大小
    # x轴刻度方向
    "xtick.direction":     ["out", "in", "inout"],  # direction of xticks
    # x轴刻度对齐方式
    "xtick.alignment":     ["center", "right", "left"],

    # 左侧y轴刻度是否显示
    "ytick.left":          validate_bool,      # draw ticks on left side
    # 右侧y轴刻度是否显示
    "ytick.right":         validate_bool,      # draw ticks on right side
    # 左侧y轴刻度标签是否显示
    "ytick.labelleft":     validate_bool,      # draw tick labels on left side
    # 右侧y轴刻度标签是否显示
    "ytick.labelright":    validate_bool,      # draw tick labels on right side
    # 主要y轴刻度大小（单位：点）
    "ytick.major.size":    validate_float,     # major ytick size in points
    # 次要y轴刻度大小（单位：点）
    "ytick.minor.size":    validate_float,     # minor ytick size in points
    # 主要y轴刻度宽度（单位：点）
    "ytick.major.width":   validate_float,     # major ytick width in points
    # 次要y轴刻度宽度（单位：点）
    "ytick.minor.width":   validate_float,     # minor ytick width in points
    # 主要y轴刻度标签到刻度的距离（单位：点）
    "ytick.major.pad":     validate_float,     # distance to label in points
    # 次要y轴刻度标签到刻度的距离（单位：点）
    "ytick.minor.pad":     validate_float,     # distance to label in points
    # y轴刻度颜色
    "ytick.color":         validate_color,     # color of yticks
    # y轴刻度标签颜色或继承颜色
    "ytick.labelcolor":    validate_color_or_inherit,  # color of ytick labels
    # 是否显示次要y轴刻度
    "ytick.minor.visible": validate_bool,      # visibility of minor yticks
    # 左侧是否绘制次要y轴刻度
    "ytick.minor.left":    validate_bool,      # draw left minor yticks
    # 右侧是否绘制次要y轴刻度
    "ytick.minor.right":   validate_bool,      # draw right minor yticks
    # 左侧是否绘制主要y轴刻度
    "ytick.major.left":    validate_bool,      # draw left major yticks
    # 右侧是否绘制主要y轴刻度
    "ytick.major.right":   validate_bool,      # draw right major yticks
    # 次要y轴刻度数量
    # number of minor yticks
    "ytick.minor.ndivs":   _validate_minor_tick_ndivs,
    # y轴刻度标签字体大小
    "ytick.labelsize":     validate_fontsize,  # fontsize of ytick labels
    # y轴刻度方向
    "ytick.direction":     ["out", "in", "inout"],  # direction of yticks
    # y轴刻度对齐方式
    "ytick.alignment":     [
        "center", "top", "bottom", "baseline", "center_baseline"],

    # 网格线颜色
    "grid.color":        validate_color,  # grid color
    # 网格线样式
    "grid.linestyle":    _validate_linestyle,  # solid
    # 网格线宽度（单位：点）
    "grid.linewidth":    validate_float,     # in points
    # 网格线透明度
    "grid.alpha":        validate_float,

    ## 图形属性
    # 图形标题字体大小
    "figure.titlesize":   validate_fontsize,
    # 图形标题字体粗细
    "figure.titleweight": validate_fontweight,

    # 图形标签字体大小
    "figure.labelsize":   validate_fontsize,
    # 图形标签字体粗细
    "figure.labelweight": validate_fontweight,

    # 图形大小（单位：英寸），宽度乘高度
    "figure.figsize":          _listify_validator(validate_float, n=2),
    # 图形分辨率（单位：点每英寸）
    "figure.dpi":              validate_float,
    # 图形背景色
    "figure.facecolor":        validate_color,
    # 图形边缘颜色
    "figure.edgecolor":        validate_color,
    # 是否显示图形边框
    "figure.frameon":          validate_bool,
    # 是否自动调整图形布局
    "figure.autolayout":       validate_bool,
    # 最大打开警告数
    "figure.max_open_warning": validate_int,
    # 是否提升窗口
    "figure.raise_window":     validate_bool,
    # macOS窗口模式
    "macosx.window_mode":      ["system", "tab", "window"],

    # 子图左边距（单位：相对值）
    "figure.subplot.left":   validate_float,
    # 子图右边距（单位：相对值）
    "figure.subplot.right":  validate_float,
    # 子图底边距（单位：相对值）
    "figure.subplot.bottom": validate_float,
    # 子图顶边距（单位：相对值）
    "figure.subplot.top":    validate_float,
    # 子图水平间距（单位：相对值）
    "figure.subplot.wspace": validate_float,
    # 子图垂直间距（单位：相对值）
    "figure.subplot.hspace": validate_float,
    "figure.constrained_layout.use": validate_bool,  # 是否启用 constrained_layout？
    # wspace 和 hspace 是相邻子图用于空间的分数。
    # 比上面的要小得多，因为我们不需要为文本留出空间。
    "figure.constrained_layout.hspace": validate_float,
    "figure.constrained_layout.wspace": validate_float,
    # Axes 周围的缓冲区，单位为英寸。
    "figure.constrained_layout.h_pad": validate_float,
    "figure.constrained_layout.w_pad": validate_float,

    ## 保存图形属性
    'savefig.dpi':          validate_dpi,  # 分辨率
    'savefig.facecolor':    validate_color_or_auto,  # 背景颜色或自动
    'savefig.edgecolor':    validate_color_or_auto,  # 边框颜色或自动
    'savefig.orientation':  ['landscape', 'portrait'],  # 方向（横向或纵向）
    "savefig.format":       validate_string,  # 保存格式
    "savefig.bbox":         validate_bbox,  # "tight" 或 "standard"（= None）
    "savefig.pad_inches":   validate_float,  # 边距（单位英寸）
    # 保存对话框中的默认目录
    "savefig.directory":    _validate_pathlike,
    "savefig.transparent":  validate_bool,  # 是否透明保存

    "tk.window_focus": validate_bool,  # 维持 TkAgg 的 shell 焦点

    # 设置纸张大小/类型
    "ps.papersize":       _validate_papersize,
    "ps.useafm":          validate_bool,
    # 使用 ghostscript 或 xpdf 来蒸馏 ps 输出
    "ps.usedistiller":    validate_ps_distiller,
    "ps.distiller.res":   validate_int,  # dpi
    "ps.fonttype":        validate_fonttype,  # 字体类型（3 - Type3 或 42 - Truetype）
    "pdf.compression":    validate_int,  # 压缩级别（0-9；0 表示禁用）
    "pdf.inheritcolor":   validate_bool,  # 是否跳过颜色设置命令
    # 仅使用每个 PDF 查看应用程序中内嵌的 14 种 PDF 核心字体
    "pdf.use14corefonts": validate_bool,
    "pdf.fonttype":       validate_fonttype,  # 字体类型（3 - Type3 或 42 - Truetype）

    "pgf.texsystem": ["xelatex", "lualatex", "pdflatex"],  # 使用的 LaTeX 变体
    "pgf.rcfonts":   validate_bool,  # 是否使用 mpl 的 rc 设置进行字体配置
    "pgf.preamble":  validate_string,  # 自定义 LaTeX 引言

    # 将光栅图像数据写入 svg 文件
    "svg.image_inline": validate_bool,
    "svg.fonttype": ["none", "path"],  # 将文本保存为文本 ("none") 或 "路径"
    "svg.hashsalt": validate_string_or_None,

    # 设置此项以生成硬拷贝文档字符串
    "docstring.hardcopy": validate_bool,

    "path.simplify":           validate_bool,  # 是否简化路径
    "path.simplify_threshold": _validate_greaterequal0_lessequal1,  # 简化阈值
    "path.snap":               validate_bool,  # 是否捕捉路径
    "path.sketch":             validate_sketch,  # 草图效果
    "path.effects":            validate_anylist,  # 路径效果
    "agg.path.chunksize":      validate_int,  # 切片路径的大小（0 禁用切片）

    # 键映射（多字符映射应为列表/元组）
    "keymap.fullscreen": validate_stringlist,
    "keymap.home":       validate_stringlist,
    "keymap.back":       validate_stringlist,
    "keymap.forward":    validate_stringlist,
    "keymap.pan":        validate_stringlist,
    "keymap.zoom":       validate_stringlist,
    # 定义键映射设置，每个键表示一个设置项，值为验证函数
    "keymap.save":       validate_stringlist,
    "keymap.quit":       validate_stringlist,
    "keymap.quit_all":   validate_stringlist,  # 例如: "W", "cmd+W", "Q"
    "keymap.grid":       validate_stringlist,
    "keymap.grid_minor": validate_stringlist,
    "keymap.yscale":     validate_stringlist,
    "keymap.xscale":     validate_stringlist,
    "keymap.help":       validate_stringlist,
    "keymap.copy":       validate_stringlist,

    # 动画设置
    "animation.html":         ["html5", "jshtml", "none"],
    # HTML 中以 base64 编码的动画的大小限制，单位为 MB
    "animation.embed_limit":  validate_float,
    # 动画写入器的选择
    "animation.writer":       validate_string,
    # 动画编解码器的选择
    "animation.codec":        validate_string,
    # 动画比特率设置
    "animation.bitrate":      validate_int,
    # 帧写入到磁盘时的图像格式选择
    "animation.frame_format": ["png", "jpeg", "tiff", "raw", "rgba", "ppm",
                               "sgi", "bmp", "pbm", "svg"],
    # ffmpeg 二进制文件路径设置，如果仅为二进制名称，则使用系统 PATH
    "animation.ffmpeg_path":  _validate_pathlike,
    # ffmpeg 电影写入器的额外参数设置（使用管道）
    "animation.ffmpeg_args":  validate_stringlist,
    # convert 二进制文件路径设置，如果仅为二进制名称，则使用系统 PATH
    "animation.convert_path": _validate_pathlike,
    # convert 电影写入器的额外参数设置（使用管道）
    "animation.convert_args": validate_stringlist,

    # 经典模式（2.0 之前）兼容模式
    # 用于一些无法通过 rcParam 合理实现向后兼容的设置
    # 这不会完全启用经典模式。要完全启用，使用 `matplotlib.style.use("classic")`
    "_internal.classic_mode": validate_bool
}
# _hardcoded_defaults 字典存储了一些默认配置项，这些项不是从 matplotlibrc 文件推断出来的。
# 其中 "_internal.classic_mode" 是一个私有配置项，设置为 False。
# 在这里没有当前的弃用项。
# backend 是在构建 rcParamsDefault 时单独处理的。
_validators = {
    # _validators 字典通过调用 _convert_validator_spec 函数来为每个键值对应用验证器转换。
    # _convert_validator_spec 函数被用于每个 _validators.items() 中的键值对 k 和 conv。
    k: _convert_validator_spec(k, conv)
    for k, conv in _validators.items()
}
```