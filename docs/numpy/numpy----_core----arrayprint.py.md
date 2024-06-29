# `.\numpy\numpy\_core\arrayprint.py`

```py
"""Array printing function

$Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $

"""
__all__ = ["array2string", "array_str", "array_repr",
           "set_printoptions", "get_printoptions", "printoptions",
           "format_float_positional", "format_float_scientific"]
__docformat__ = 'restructuredtext'

#
# Written by Konrad Hinsen <hinsenk@ere.umontreal.ca>
# last revision: 1996-3-13
# modified by Jim Hugunin 1997-3-3 for repr's and str's (and other details)
# and by Perry Greenfield 2000-4-1 for numarray
# and by Travis Oliphant  2005-8-22 for numpy


# Note: Both scalartypes.c.src and arrayprint.py implement strs for numpy
# scalars but for different purposes. scalartypes.c.src has str/reprs for when
# the scalar is printed on its own, while arrayprint.py has strs for when
# scalars are printed inside an ndarray. Only the latter strs are currently
# user-customizable.

import functools
import numbers
import sys
try:
    from _thread import get_ident
except ImportError:
    from _dummy_thread import get_ident

import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
                         datetime_as_string, datetime_data, ndarray,
                         set_legacy_print_mode)
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float64, complex128,
                           flexible)
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib

_format_options = {
    'edgeitems': 3,  # repr N leading and trailing items of each dimension
    'threshold': 1000,  # total items > triggers array summarization
    'floatmode': 'maxprec',
    'precision': 8,  # precision of floating point representations
    'suppress': False,  # suppress printing small floating values in exp format
    'linewidth': 75,
    'nanstr': 'nan',
    'infstr': 'inf',
    'sign': '-',
    'formatter': None,
    # Internally stored as an int to simplify comparisons; converted from/to
    # str/False on the way in/out.
    'legacy': sys.maxsize,
    'override_repr': None,
}

def _make_options_dict(precision=None, threshold=None, edgeitems=None,
                       linewidth=None, suppress=None, nanstr=None, infstr=None,
                       sign=None, formatter=None, floatmode=None, legacy=None,
                       override_repr=None):
    """
    Make a dictionary out of the non-None arguments, plus conversion of
    *legacy* and sanity checks.
    """

    options = {k: v for k, v in list(locals().items()) if v is not None}

    if suppress is not None:
        options['suppress'] = bool(suppress)

    modes = ['fixed', 'unique', 'maxprec', 'maxprec_equal']

    # Return the constructed options dictionary
    return options
    # 检查 floatmode 是否在 modes 列表或者为 None，否则引发数值错误
    if floatmode not in modes + [None]:
        raise ValueError("floatmode option must be one of " +
                         ", ".join('"{}"'.format(m) for m in modes))

    # 检查 sign 是否为 None, '-', '+', 或者 ' ' 中的一个，否则引发数值错误
    if sign not in [None, '-', '+', ' ']:
        raise ValueError("sign option must be one of ' ', '+', or '-'")

    # 根据 legacy 的不同取值设置 options 字典中的 legacy 键
    if legacy == False:
        options['legacy'] = sys.maxsize
    elif legacy == '1.13':
        options['legacy'] = 113
    elif legacy == '1.21':
        options['legacy'] = 121
    elif legacy == '1.25':
        options['legacy'] = 125
    elif legacy is None:
        pass  # 如果 legacy 为 None，则什么都不做
    else:
        # 如果 legacy 取值不符合预期，发出警告
        warnings.warn(
            "legacy printing option can currently only be '1.13', '1.21', "
            "'1.25', or `False`", stacklevel=3)

    # 如果 threshold 不为 None，检查其类型是否为数值类型，否则引发类型错误
    if threshold is not None:
        # 拒绝来自 stack overflow 建议的不良 threshold 参数
        if not isinstance(threshold, numbers.Number):
            raise TypeError("threshold must be numeric")
        # 检查 threshold 是否为 NaN，如果是，则引发数值错误
        if np.isnan(threshold):
            raise ValueError("threshold must be non-NAN, try "
                             "sys.maxsize for untruncated representation")

    # 如果 precision 不为 None，尝试将其转换为整数并设置 options 字典中的 precision 键
    if precision is not None:
        # 拒绝来自 issue #18254 建议的不良 precision 参数
        try:
            options['precision'] = operator.index(precision)
        except TypeError as e:
            raise TypeError('precision must be an integer') from e

    # 返回处理后的 options 字典
    return options
# 设置模块为 'numpy'
@set_module('numpy')
# 定义设置打印选项的函数
def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None, nanstr=None,
                     infstr=None, formatter=None, sign=None, floatmode=None,
                     *, legacy=None, override_repr=None):
    """
    Set printing options.

    These options determine the way floating point numbers, arrays and
    other NumPy objects are displayed.

    Parameters
    ----------
    precision : int or None, optional
        浮点数输出的有效数字位数（默认为 8）。
        如果 `floatmode` 不是 `fixed`，可以为 None，此时将打印足够多的位数以唯一确定值。
    threshold : int, optional
        触发总结而不是完整 repr 的数组元素总数阈值（默认为 1000）。
        若要始终使用完整 repr 而不进行总结，请传递 `sys.maxsize`。
    edgeitems : int, optional
        每个维度开头和结尾处摘要中的数组项数（默认为 3）。
    linewidth : int, optional
        用于插入换行符的每行字符数（默认为 75）。
    suppress : bool, optional
        如果为 True，则始终使用固定点表示法打印浮点数，此时在当前精度下等于零的数将打印为零。
        如果为 False，则当最小数的绝对值 < 1e-4 或最大绝对值与最小绝对值的比值 > 1e3 时使用科学计数法。
        默认为 False。
    nanstr : str, optional
        浮点数 NaN 的字符串表示（默认为 'nan'）。
    infstr : str, optional
        浮点数无穷大的字符串表示（默认为 'inf'）。
    sign : string, either '-', '+', or ' ', optional
        控制浮点类型的符号打印。如果为 '+'，始终打印正值的符号。
        如果为 ' '，在正值的符号位置打印一个空格字符。
        如果为 '-'，省略正值的符号字符。（默认为 '-'）

        .. versionchanged:: 2.0
             现在 sign 参数可以是整数类型，以前只能是浮点类型。
    """
    # formatter 是一个字典，其中键是数据类型或者数据类型组合的字符串，值是对应的格式化函数（可调用对象）
    # 如果为 None，则使用默认的格式化函数处理相应类型
    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:
        
        - 'bool'
            # 布尔型数据格式化函数
        - 'int'
            # 整数型数据格式化函数
        - 'timedelta' : a `numpy.timedelta64`
            # 时间增量数据格式化函数，对应 numpy.timedelta64 类型
        - 'datetime' : a `numpy.datetime64`
            # 日期时间数据格式化函数，对应 numpy.datetime64 类型
        - 'float'
            # 浮点数数据格式化函数
        - 'longfloat' : 128-bit floats
            # 长浮点数（128位）数据格式化函数
        - 'complexfloat'
            # 复数浮点数数据格式化函数
        - 'longcomplexfloat' : composed of two 128-bit floats
            # 由两个 128 位浮点数组成的长复数浮点数数据格式化函数
        - 'numpystr' : types `numpy.bytes_` and `numpy.str_`
            # numpy.bytes_ 和 numpy.str_ 类型数据格式化函数
        - 'object' : `np.object_` arrays
            # 对象数组数据格式化函数
        
        Other keys that can be used to set a group of types at once are:
        
        - 'all' : sets all types
            # 设置所有类型的格式化函数
        - 'int_kind' : sets 'int'
            # 设置整数类型的格式化函数
        - 'float_kind' : sets 'float' and 'longfloat'
            # 设置浮点数和长浮点数类型的格式化函数
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
            # 设置复数浮点数和长复数浮点数类型的格式化函数
        - 'str_kind' : sets 'numpystr'
            # 设置 numpy 字符串类型的格式化函数
    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types. Can take the following values
        (default maxprec_equal):
        
        * 'fixed': Always print exactly `precision` fractional digits,
                even if this would print more or fewer digits than
                necessary to specify the value uniquely.
            # 固定模式：始终打印精确的 `precision` 位小数，即使这可能比唯一指定值所需的位数多或少。
        * 'unique': Print the minimum number of fractional digits necessary
                to represent each value uniquely. Different elements may
                have a different number of digits. The value of the
                `precision` option is ignored.
            # 唯一模式：打印表示每个值所需的最少小数位数，以确保每个值都能唯一表示。不同元素可能具有不同位数的小数部分。忽略 `precision` 选项的值。
        * 'maxprec': Print at most `precision` fractional digits, but if
                an element can be uniquely represented with fewer digits
                only print it with that many.
            # 最大精度模式：打印至多 `precision` 位小数，但如果某个元素可以用更少的位数唯一表示，则只打印该数目。
        * 'maxprec_equal': Print at most `precision` fractional digits,
                but if every element in the array can be uniquely
                represented with an equal number of fewer digits, use that
                many digits for all elements.
            # 最大精度等值模式：打印至多 `precision` 位小数，但如果数组中的每个元素都可以用相等或更少位数唯一表示，则对所有元素使用该数目的小数位数。
    legacy : string or `False`, optional
        # 控制是否启用旧版本打印模式的选项，可以是字符串 `'1.13'`, `'1.21'`, `'1.25'` 或 `False`
        # `'1.13'` 模式近似于 numpy 1.13 的打印输出，浮点数的符号位置包含空格，对于0维数组有不同行为
        # `'1.21'` 模式近似于 numpy 1.21 的打印输出，特别是复杂结构的dtype的打印
        # `'1.25'` 模式近似于 numpy 1.25 的打印输出，主要是数字标量打印时不包含类型信息
        # 设置为 `False` 则禁用旧版本打印模式
        # 未识别的字符串会被忽略，并显示警告以保持向前兼容性
        .. versionadded:: 1.14.0
        .. versionchanged:: 1.22.0
        .. versionchanged:: 2.0

    override_repr: callable, optional
        # 如果设置，将使用传递的函数生成数组的repr（表示字符串），忽略其他选项
        # 参见 `get_printoptions`, `printoptions`, `array2string`
        # `formatter` 总是通过调用 `set_printoptions` 来重置

    Notes
    -----
    # 作为上下文管理器使用 `printoptions` 可以临时设置打印选项
    # 示例中展示了如何设置浮点精度、数组概述、抑制小结果以及自定义格式化器的用法
    # 还展示了如何恢复默认选项和临时覆盖选项的方法
    # 使用给定的参数创建选项字典opt
    opt = _make_options_dict(precision, threshold, edgeitems, linewidth,
                             suppress, nanstr, infstr, sign, formatter,
                             floatmode, legacy, override_repr)
    # formatter 和 override_repr 总是被重设
    opt['formatter'] = formatter
    opt['override_repr'] = override_repr
    # 更新全局的格式选项 _format_options
    _format_options.update(opt)

    # 根据 legacy 模式设置 C 变量
    if _format_options['legacy'] == 113:
        # 如果 legacy 模式为 113，则设置打印模式为 113
        set_legacy_print_mode(113)
        # 在 legacy 模式下重设 sign 选项，以避免混淆
        _format_options['sign'] = '-'
    elif _format_options['legacy'] == 121:
        # 如果 legacy 模式为 121，则设置打印模式为 121
        set_legacy_print_mode(121)
    elif _format_options['legacy'] == 125:
        # 如果 legacy 模式为 125，则设置打印模式为 125
        set_legacy_print_mode(125)
    elif _format_options['legacy'] == sys.maxsize:
        # 如果 legacy 模式为系统支持的最大值，则设置打印模式为 0
        set_legacy_print_mode(0)
# 设置模块名称为'numpy'
@set_module('numpy')
# 返回当前的打印选项
def get_printoptions():
    """
    Return the current print options.

    Returns
    -------
    print_opts : dict
        Dictionary of current print options with keys

        - precision : int
        - threshold : int
        - edgeitems : int
        - linewidth : int
        - suppress : bool
        - nanstr : str
        - infstr : str
        - formatter : dict of callables
        - sign : str

        For a full description of these options, see `set_printoptions`.

    See Also
    --------
    set_printoptions, printoptions

    Examples
    --------

    >>> np.get_printoptions()
    {'edgeitems': 3, 'threshold': 1000, ..., 'override_repr': None}

    >>> np.get_printoptions()['linewidth']
    75
    >>> np.set_printoptions(linewidth=100)
    >>> np.get_printoptions()['linewidth']
    100

    """
    # 复制格式化选项并包含'legacy'键
    opts = _format_options.copy()
    # 根据当前'legacy'键的值选择对应的版本字符串或者False
    opts['legacy'] = {
        113: '1.13', 121: '1.21', 125: '1.25', sys.maxsize: False,
    }[opts['legacy']]
    return opts


# 返回作为整数的旧打印模式
def _get_legacy_print_mode():
    """Return the legacy print mode as an int."""
    return _format_options['legacy']


# 设置模块名称为'numpy'，并创建一个上下文管理器来设置打印选项
@set_module('numpy')
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for setting print options.

    Set print options for the scope of the `with` block, and restore the old
    options at the end. See `set_printoptions` for the full description of
    available options.

    Examples
    --------

    >>> from numpy.testing import assert_equal
    >>> with np.printoptions(precision=2):
    ...     np.array([2.0]) / 3
    array([0.67])

    The `as`-clause of the `with`-statement gives the current print options:

    >>> with np.printoptions(precision=2) as opts:
    ...      assert_equal(opts, np.get_printoptions())

    See Also
    --------
    set_printoptions, get_printoptions

    """
    # 获取当前的打印选项
    opts = np.get_printoptions()
    try:
        # 设置新的打印选项
        np.set_printoptions(*args, **kwargs)
        # 在with块中返回新的打印选项
        yield np.get_printoptions()
    finally:
        # 恢复到旧的打印选项
        np.set_printoptions(**opts)


# 保留数组的N-D角（前导和尾随边缘）
def _leading_trailing(a, edgeitems, index=()):
    """
    Keep only the N-D corners (leading and trailing edges) of an array.

    Should be passed a base-class ndarray, since it makes no guarantees about
    preserving subclasses.
    """
    # 确定当前处理的轴数
    axis = len(index)
    # 如果轴数等于数组维度，则返回指定索引的元素
    if axis == a.ndim:
        return a[index]

    # 如果当前轴的长度大于2倍的边缘项数
    if a.shape[axis] > 2 * edgeitems:
        # 连接前导和尾随边缘的元素
        return concatenate((
            _leading_trailing(a, edgeitems, index + np.index_exp[:edgeitems]),
            _leading_trailing(a, edgeitems, index + np.index_exp[-edgeitems:])
        ), axis=axis)
    else:
        # 递归地处理边缘项
        return _leading_trailing(a, edgeitems, index + np.index_exp[:])


# 格式化对象数组以打印不含歧义的列表
def _object_format(o):
    """ Object arrays containing lists should be printed unambiguously """
    # 如果对象类型为列表，则返回格式化字符串
    if type(o) is list:
        fmt = 'list({!r})'
    else:
        fmt = '{!r}'
    return fmt.format(o)


# 格式化表示形式
def repr_format(x):
    # 如果x是字符串或字节字符串类型，则返回其表示形式
    if isinstance(x, (np.str_, np.bytes_)):
        return repr(x.item())
    return repr(x)
def str_format(x):
    # 如果 x 是 numpy 的字符串或字节类型，返回其字符串表示
    if isinstance(x, (np.str_, np.bytes_)):
        return str(x.item())
    # 否则，返回 x 的普通字符串表示
    return str(x)

def _get_formatdict(data, *, precision, floatmode, suppress, sign, legacy,
                    formatter, **kwargs):
    # 注意：kwargs 中的额外参数将被忽略

    # 定义一个格式化字典，根据数据类型选择相应的格式化函数
    formatdict = {
        'bool': lambda: BoolFormat(data),  # 布尔型数据的格式化函数
        'int': lambda: IntegerFormat(data, sign),  # 整数数据的格式化函数
        'float': lambda: FloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),  # 浮点数数据的格式化函数
        'longfloat': lambda: FloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),  # 长浮点数数据的格式化函数
        'complexfloat': lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),  # 复数浮点数数据的格式化函数
        'longcomplexfloat': lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),  # 长复数浮点数数据的格式化函数
        'datetime': lambda: DatetimeFormat(data, legacy=legacy),  # 日期时间数据的格式化函数
        'timedelta': lambda: TimedeltaFormat(data),  # 时间间隔数据的格式化函数
        'object': lambda: _object_format,  # 对象类型数据的格式化函数
        'void': lambda: str_format,  # 未知类型数据的格式化函数
        'numpystr': lambda: repr_format  # numpy 字符串数据的格式化函数
    }

    # 如果传入了 formatter 参数，则根据其内容更新 formatdict 中的格式化函数
    def indirect(x):
        return lambda: x

    if formatter is not None:
        fkeys = [k for k in formatter.keys() if formatter[k] is not None]
        if 'all' in fkeys:
            for key in formatdict.keys():
                formatdict[key] = indirect(formatter['all'])
        if 'int_kind' in fkeys:
            for key in ['int']:
                formatdict[key] = indirect(formatter['int_kind'])
        if 'float_kind' in fkeys:
            for key in ['float', 'longfloat']:
                formatdict[key] = indirect(formatter['float_kind'])
        if 'complex_kind' in fkeys:
            for key in ['complexfloat', 'longcomplexfloat']:
                formatdict[key] = indirect(formatter['complex_kind'])
        if 'str_kind' in fkeys:
            formatdict['numpystr'] = indirect(formatter['str_kind'])
        for key in formatdict.keys():
            if key in fkeys:
                formatdict[key] = indirect(formatter[key])

    # 返回构建好的格式化字典
    return formatdict

def _get_format_function(data, **options):
    """
    找到适合数据类型的格式化函数
    """
    dtype_ = data.dtype
    dtypeobj = dtype_.type
    formatdict = _get_formatdict(data, **options)  # 获取格式化函数字典
    if dtypeobj is None:
        return formatdict["numpystr"]()  # 若数据类型对象为空，则返回默认格式化函数
    elif issubclass(dtypeobj, _nt.bool):
        return formatdict['bool']()  # 若数据类型为布尔型，则返回布尔型数据格式化函数
    elif issubclass(dtypeobj, _nt.integer):
        if issubclass(dtypeobj, _nt.timedelta64):
            return formatdict['timedelta']()  # 若数据类型为时间间隔型整数，则返回时间间隔数据格式化函数
        else:
            return formatdict['int']()  # 否则返回整数数据格式化函数
    # 如果数据类型是浮点数类型的子类
    elif issubclass(dtypeobj, _nt.floating):
        # 如果是长双精度浮点数类型的子类
        if issubclass(dtypeobj, _nt.longdouble):
            # 返回长双精度浮点数的格式化对象
            return formatdict['longfloat']()
        else:
            # 返回普通浮点数的格式化对象
            return formatdict['float']()
    
    # 如果数据类型是复数浮点数类型的子类
    elif issubclass(dtypeobj, _nt.complexfloating):
        # 如果是长复数浮点数类型的子类
        if issubclass(dtypeobj, _nt.clongdouble):
            # 返回长复数浮点数的格式化对象
            return formatdict['longcomplexfloat']()
        else:
            # 返回普通复数浮点数的格式化对象
            return formatdict['complexfloat']()
    
    # 如果数据类型是字符串或字节类型的子类
    elif issubclass(dtypeobj, (_nt.str_, _nt.bytes_)):
        # 返回NumPy字符串的格式化对象
        return formatdict['numpystr']()
    
    # 如果数据类型是datetime64类型的子类
    elif issubclass(dtypeobj, _nt.datetime64):
        # 返回日期时间的格式化对象
        return formatdict['datetime']()
    
    # 如果数据类型是object类型的子类
    elif issubclass(dtypeobj, _nt.object_):
        # 如果数据类型具有字段名（不为None），则返回根据数据创建的结构化void格式化对象
        if dtype_.names is not None:
            return StructuredVoidFormat.from_data(data, **options)
        else:
            # 否则返回void类型的格式化对象
            return formatdict['void']()
    
    # 默认情况下返回NumPy字符串的格式化对象
    else:
        return formatdict['numpystr']()
# 定义装饰器函数 _recursive_guard，用于处理递归调用的保护机制
def _recursive_guard(fillvalue='...'):
    """
    Like the python 3.2 reprlib.recursive_repr, but forwards *args and **kwargs

    Decorates a function such that if it calls itself with the same first
    argument, it returns `fillvalue` instead of recursing.

    Largely copied from reprlib.recursive_repr
    """

    # 装饰函数，用于实际包装被修饰的函数
    def decorating_function(f):
        # 用于跟踪正在运行中的函数调用的集合
        repr_running = set()

        # 实际的包装器函数，实现递归调用的保护
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            # 生成一个唯一标识符来代表当前调用的对象和线程标识符的元组
            key = id(self), get_ident()
            # 如果当前调用的对象和线程标识符已经在运行中集合中，直接返回填充值
            if key in repr_running:
                return fillvalue
            # 将当前调用的对象和线程标识符加入到运行中集合中
            repr_running.add(key)
            try:
                # 调用原始函数，并传递相同的参数
                return f(self, *args, **kwargs)
            finally:
                # 当函数调用完成时，从运行中集合中移除当前调用的对象和线程标识符
                repr_running.discard(key)

        return wrapper

    return decorating_function


# 使用 _recursive_guard 装饰器，处理数组包含自身的递归调用情况
@_recursive_guard()
def _array2string(a, options, separator=' ', prefix=""):
    # 将输入数组转换为 ndarray 对象
    data = asarray(a)
    # 如果数组是标量（shape 为 ()），使用 data 重新赋值 a
    if a.shape == ():
        a = data

    # 如果数组大小超过阈值选项中的值，进行摘要处理
    if a.size > options['threshold']:
        summary_insert = "..."
        # 对数据进行前导和尾随修剪，以便显示边缘项
        data = _leading_trailing(data, options['edgeitems'])
    else:
        summary_insert = ""

    # 根据数据类型找到适当的格式化函数
    format_function = _get_format_function(data, **options)

    # 设置下一行的前缀，用于多行输出格式处理
    next_line_prefix = " "
    next_line_prefix += " "*len(prefix)

    # 调用 _formatArray 函数，将数组转换为格式化后的字符串列表
    lst = _formatArray(a, format_function, options['linewidth'],
                       next_line_prefix, separator, options['edgeitems'],
                       summary_insert, options['legacy'])
    return lst


# 定义一个分派函数，用于处理 array2string 的参数分发
def _array2string_dispatcher(
        a, max_line_width=None, precision=None,
        suppress_small=None, separator=None, prefix=None,
        style=None, formatter=None, threshold=None,
        edgeitems=None, sign=None, floatmode=None, suffix=None,
        *, legacy=None):
    return (a,)


# 使用 array_function_dispatch 装饰器，将 _array2string_dispatcher 与 array2string 绑定
@array_function_dispatch(_array2string_dispatcher, module='numpy')
def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ', prefix="",
                 style=np._NoValue, formatter=None, threshold=None,
                 edgeitems=None, sign=None, floatmode=None, suffix="",
                 *, legacy=None):
    """
    Return a string representation of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int or None, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    """
    # 是否将非常接近零的数表示为零；默认为 False
    suppress_small : bool, optional
        # 接近零的定义取决于精度：例如，如果精度为 8，则绝对值小于 5e-9 的数字被表示为零
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        # 默认为 ``numpy.get_printoptions()['suppress']``
    separator : str, optional
        # 元素之间的分隔符
    prefix : str, optional
    suffix : str, optional
        # 前缀和后缀字符串的长度分别用于对齐和包装输出。数组通常打印为::
        The length of the prefix and suffix strings are used to respectively
        align and wrap the output. An array is typically printed as::

          prefix + array2string(a) + suffix

        # 输出左侧由前缀字符串的长度填充，并在列 ``max_line_width - len(suffix)`` 处强制换行。
        It should be noted that the content of prefix and suffix strings are
        not included in the output.
    style : _NoValue, optional
        # 没有效果，不要使用。
        Has no effect, do not use.

        # .. deprecated:: 1.14.0
    formatter : dict of callables, optional
        # 如果不是 None，则键应指示相应格式化函数适用的类型。
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.

        # 可以设置格式化函数的个别类型包括：
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:

        - 'bool'
        - 'int'
        - 'timedelta' : a `numpy.timedelta64`
        - 'datetime' : a `numpy.datetime64`
        - 'float'
        - 'longfloat' : 128-bit floats
        - 'complexfloat'
        - 'longcomplexfloat' : composed of two 128-bit floats
        - 'void' : type `numpy.void`
        - 'numpystr' : types `numpy.bytes_` and `numpy.str_`

        # 其他可用于一次设置一组类型的键包括：
        Other keys that can be used to set a group of types at once are:

        - 'all' : sets all types
        - 'int_kind' : sets 'int'
        - 'float_kind' : sets 'float' and 'longfloat'
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
        - 'str_kind' : sets 'numpystr'
    threshold : int, optional
        # 触发汇总而不是完整 repr 的数组元素总数。
        Total number of array elements which trigger summarization
        rather than full repr.
        # 默认为 ``numpy.get_printoptions()['threshold']``
    edgeitems : int, optional
        # 每个维度开头和结尾的摘要中的数组项数。
        Number of array items in summary at beginning and end of
        each dimension.
        # 默认为 ``numpy.get_printoptions()['edgeitems']``
    sign : string, either '-', '+', or ' ', optional
        # 控制浮点类型的符号打印。如果是 '+', 总是打印正值的符号。如果是 ' ', 在正值的符号位置总是打印一个空格（空白字符）。如果是 '-', 则省略正值的符号字符。
        Controls printing of the sign of floating-point types. If '+', always
        print the sign of positive values. If ' ', always prints a space
        (whitespace character) in the sign position of positive values.  If
        '-', omit the sign character of positive values.
        # 默认为 ``numpy.get_printoptions()['sign']``.

        # .. versionchanged:: 2.0
             The sign parameter can now be an integer type, previously
             types were floating-point types.
    # 控制浮点类型的 `precision` 选项的解释方式
    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types.
        Defaults to ``numpy.get_printoptions()['floatmode']``.
        Can take the following values:

        - 'fixed': Always print exactly `precision` fractional digits,
          even if this would print more or fewer digits than
          necessary to specify the value uniquely.
        - 'unique': Print the minimum number of fractional digits necessary
          to represent each value uniquely. Different elements may
          have a different number of digits.  The value of the
          `precision` option is ignored.
        - 'maxprec': Print at most `precision` fractional digits, but if
          an element can be uniquely represented with fewer digits
          only print it with that many.
        - 'maxprec_equal': Print at most `precision` fractional digits,
          but if every element in the array can be uniquely
          represented with an equal number of fewer digits, use that
          many digits for all elements.
    
    # 控制是否启用 1.13 版本的遗留打印模式
    legacy : string or `False`, optional
        If set to the string ``'1.13'`` enables 1.13 legacy printing mode. This
        approximates numpy 1.13 print output by including a space in the sign
        position of floats and different behavior for 0d arrays. If set to
        `False`, disables legacy mode. Unrecognized strings will be ignored
        with a warning for forward compatibility.

        .. versionadded:: 1.14.0

    # 返回数组的字符串表示形式
    Returns
    -------
    array_str : str
        String representation of the array.

    # 如果 `formatter` 中的可调用对象未返回字符串，则引发 TypeError
    Raises
    ------
    TypeError
        if a callable in `formatter` does not return a string.

    # 参见其他相关函数和选项设置
    See Also
    --------
    array_str, array_repr, set_printoptions, get_printoptions

    # 注意事项，关于特定类型的格式化程序覆盖了 `precision` 关键字的说明
    Notes
    -----
    If a formatter is specified for a certain type, the `precision` keyword is
    ignored for that type.

    This is a very flexible function; `array_repr` and `array_str` are using
    `array2string` internally so keywords with the same name should work
    identically in all three functions.

    # 示例说明，展示了不同情况下的使用方式和输出结果
    Examples
    --------
    >>> x = np.array([1e-16,1,2,3])
    >>> np.array2string(x, precision=2, separator=',',
    ...                       suppress_small=True)
    '[0.,1.,2.,3.]'

    >>> x  = np.arange(3.)
    >>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
    '[0.00 1.00 2.00]'

    >>> x  = np.arange(3)
    >>> np.array2string(x, formatter={'int':lambda x: hex(x)})
    '[0x0 0x1 0x2]'

    """

    # 根据参数生成选项字典
    overrides = _make_options_dict(precision, threshold, edgeitems,
                                   max_line_width, suppress_small, None, None,
                                   sign, formatter, floatmode, legacy)
    # 复制格式化选项字典并更新为新生成的覆盖选项
    options = _format_options.copy()
    options.update(overrides)
    # 如果选项中的'legacy'值小于等于113，则执行以下操作
    if options['legacy'] <= 113:
        # 如果样式参数为默认值 np._NoValue，则将其设定为 repr 函数
        if style is np._NoValue:
            style = repr

        # 如果数组 a 的形状为 ()（即空元组）且没有结构化字段名
        if a.shape == () and a.dtype.names is None:
            # 返回数组中唯一元素的样式化表示
            return style(a.item())
    
    # 如果'legacy'值大于113，则执行以下操作
    elif style is not np._NoValue:
        # 发出警告提示，说明'style'参数在非1.13 'legacy'模式下已被弃用且不再起作用
        warnings.warn("'style' argument is deprecated and no longer functional"
                      " except in 1.13 'legacy' mode",
                      DeprecationWarning, stacklevel=2)

    # 如果'legacy'值大于113，则调整选项中的'linewidth'值减去后缀长度
    if options['legacy'] > 113:
        options['linewidth'] -= len(suffix)

    # 如果数组 a 的大小为0，则将其视为空数组表示
    if a.size == 0:
        return "[]"

    # 调用 _array2string 函数处理数组 a，并返回结果字符串
    return _array2string(a, options, separator, prefix)
# 定义一个函数 `_extendLine`，用于扩展行 `line`，根据需要在其末尾添加单词 `word`，并根据指定的条件进行换行处理。
def _extendLine(s, line, word, line_width, next_line_prefix, legacy):
    # 判断是否需要换行，条件是当前行加上单词长度超过了指定的行宽
    needs_wrap = len(line) + len(word) > line_width
    # 如果 legacy 大于 113，则不进行换行，除非当前行的长度小于或等于下一行的前缀长度
    if legacy > 113:
        if len(line) <= len(next_line_prefix):
            needs_wrap = False

    # 如果需要换行，则将当前行加入字符串 s，然后重置 line 为下一行的前缀
    if needs_wrap:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    # 将单词 word 添加到当前行的末尾
    line += word
    # 返回更新后的字符串 s 和当前行 line
    return s, line

# 定义一个函数 `_extendLine_pretty`，扩展行 `line`，处理格式化良好的（可能是多行的）字符串 `word`。
def _extendLine_pretty(s, line, word, line_width, next_line_prefix, legacy):
    """
    Extends line with nicely formatted (possibly multi-line) string ``word``.
    """
    # 将字符串 word 按行分割为列表 words
    words = word.splitlines()
    # 如果 word 只有一行，或者 legacy 小于等于 113，则调用 `_extendLine` 函数处理
    if len(words) == 1 or legacy <= 113:
        return _extendLine(s, line, word, line_width, next_line_prefix, legacy)

    # 计算所有单词中最长的长度
    max_word_length = max(len(word) for word in words)
    # 如果当前行加上最长单词长度超过了行宽，并且当前行长度大于下一行前缀的长度，则进行换行处理
    if (len(line) + max_word_length > line_width and
            len(line) > len(next_line_prefix)):
        s += line.rstrip() + '\n'
        line = next_line_prefix + words[0]
        indent = next_line_prefix
    else:
        # 否则，设置缩进为当前行的长度的空格
        indent = len(line)*' '
        line += words[0]

    # 遍历除第一个单词外的所有单词，添加到字符串 s 中并设置正确的缩进
    for word in words[1::]:
        s += line.rstrip() + '\n'
        line = indent + word

    # 计算最后一个单词的长度差，用空格填充到当前行末尾
    suffix_length = max_word_length - len(words[-1])
    line += suffix_length*' '

    # 返回更新后的字符串 s 和当前行 line
    return s, line

# 定义一个函数 `_formatArray`，用于格式化数组 `a`，使用指定的格式化函数 `format_function`，并根据条件进行换行和缩进处理。
def _formatArray(a, format_function, line_width, next_line_prefix,
                 separator, edge_items, summary_insert, legacy):
    """formatArray is designed for two modes of operation:

    1. Full output

    2. Summarized output
    """
    try:
        # 调用递归函数 recurser，传入初始索引为空元组和指定的 hanging_indent
        return recurser(index=(),
                        hanging_indent=next_line_prefix,
                        curr_width=line_width)
    finally:
        # 递归闭包会有自身的循环引用，需要通过垃圾回收来解决（gh-10620）。为了性能和 PyPy 的友好性，我们打破循环引用：
        recurser = None

# 定义一个函数 `_none_or_positive_arg`，用于检查参数 x 是否为 None 或正数，如果是 None 则返回 -1，如果是负数则抛出 ValueError 异常。
def _none_or_positive_arg(x, name):
    if x is None:
        return -1
    if x < 0:
        raise ValueError("{} must be >= 0".format(name))
    return x

# 定义一个类 `FloatingFormat`，用于处理 np.floating 的子类型的格式化。
class FloatingFormat:
    """ Formatter for subtypes of np.floating """
    def __init__(self, data, precision, floatmode, suppress_small, sign=False,
                 *, legacy=None):
        # 用于向后兼容，接受布尔值并将其转换为符号字符串
        if isinstance(sign, bool):
            sign = '+' if sign else '-'

        self._legacy = legacy
        # 如果遗留版本小于等于113，则根据条件设置符号
        if self._legacy <= 113:
            # 当数据不是0维时，遗留版本不支持符号'-'
            if data.shape != () and sign == '-':
                sign = ' '

        self.floatmode = floatmode
        # 根据浮点模式设置精度，若为'unique'则精度为None
        if floatmode == 'unique':
            self.precision = None
        else:
            self.precision = precision

        # 确保精度参数为None或正数
        self.precision = _none_or_positive_arg(self.precision, 'precision')

        self.suppress_small = suppress_small
        self.sign = sign
        self.exp_format = False
        self.large_exponent = False

        # 调用fillFormat方法处理数据
        self.fillFormat(data)

    def __call__(self, x):
        # 如果x不是有限数
        if not np.isfinite(x):
            with errstate(invalid='ignore'):
                if np.isnan(x):
                    # 对于NaN，根据符号返回格式化后的字符串
                    sign = '+' if self.sign == '+' else ''
                    ret = sign + _format_options['nanstr']
                else:  # isinf
                    # 对于无穷大数，根据符号返回格式化后的字符串
                    sign = '-' if x < 0 else '+' if self.sign == '+' else ''
                    ret = sign + _format_options['infstr']
                # 根据格式化后字符串的长度，返回填充空格后的结果字符串
                return ' '*(
                    self.pad_left + self.pad_right + 1 - len(ret)
                ) + ret

        # 如果采用指数格式化
        if self.exp_format:
            # 调用dragon4_scientific函数进行科学计数法格式化
            return dragon4_scientific(x,
                                      precision=self.precision,
                                      min_digits=self.min_digits,
                                      unique=self.unique,
                                      trim=self.trim,
                                      sign=self.sign == '+',
                                      pad_left=self.pad_left,
                                      exp_digits=self.exp_size)
        else:
            # 否则，调用dragon4_positional函数进行定点格式化
            return dragon4_positional(x,
                                      precision=self.precision,
                                      min_digits=self.min_digits,
                                      unique=self.unique,
                                      fractional=True,
                                      trim=self.trim,
                                      sign=self.sign == '+',
                                      pad_left=self.pad_left,
                                      pad_right=self.pad_right)
# 设置当前模块为 'numpy'
@set_module('numpy')
# 定义函数 format_float_scientific，用于将浮点数格式化为科学计数法的字符串表示
def format_float_scientific(x, precision=None, unique=True, trim='k',
                            sign=False, pad_left=None, exp_digits=None,
                            min_digits=None):
    """
    Format a floating-point scalar as a decimal string in scientific notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        is given fewer digits than necessary can be printed. If `min_digits`
        is given more can be printed, in which cases the last digit is rounded
        with unbiased rounding.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value with unbiased rounding
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    exp_digits : non-negative integer, optional
        Pad the exponent with zeros until it contains at least this
        many digits. If omitted, the exponent will be at least 2 digits.
    min_digits : non-negative integer or None, optional
        Minimum number of digits to print. This only has an effect for
        `unique=True`. In that case more digits than necessary to uniquely
        identify the value may be printed and rounded unbiased.

        .. versionadded:: 1.21.0

    Returns
    -------
    rep : string
        The string representation of the floating point value

    See Also
    --------
    format_float_positional

    Examples
    --------
    >>> np.format_float_scientific(np.float32(np.pi))
    '3.1415927e+00'
    >>> s = np.float32(1.23e24)
    >>> np.format_float_scientific(s, unique=False, precision=15)
    '1.230000071797338e+24'
    >>> np.format_float_scientific(s, exp_digits=4)
    '1.23e+0024'
    """
    # 实现将浮点数 x 格式化为科学计数法的字符串表示，根据参数控制精度、舍入、填充等操作
    pass
    """
    确保精度参数非空或正数，否则抛出异常
    precision = _none_or_positive_arg(precision, 'precision')
    确保左侧填充参数非空或正数，否则抛出异常
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    确保指数数字参数非空或正数，否则抛出异常
    exp_digits = _none_or_positive_arg(exp_digits, 'exp_digits')
    确保最小数字参数非空或正数，否则抛出异常
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    如果最小数字大于0并且精度大于0且最小数字大于精度，则抛出值错误异常
    if min_digits > 0 and precision > 0 and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    调用 dragon4_scientific 函数，传入参数 x、precision、unique、trim、sign、pad_left、exp_digits、min_digits
    返回 dragon4_scientific 函数的结果
    return dragon4_scientific(x, precision=precision, unique=unique,
                              trim=trim, sign=sign, pad_left=pad_left,
                              exp_digits=exp_digits, min_digits=min_digits)
    """
# 设置模块为 'numpy'，这个装饰器函数将定义在这个模块中
@set_module('numpy')
# 定义一个函数，将浮点数 x 格式化为十进制字符串，采用位置表示法
def format_float_positional(x, precision=None, unique=True,
                            fractional=True, trim='k', sign=False,
                            pad_left=None, pad_right=None, min_digits=None):
    """
    Format a floating-point scalar as a decimal string in positional notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        is given fewer digits than necessary can be printed, or if `min_digits`
        is given more can be printed, in which cases the last digit is rounded
        with unbiased rounding.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value with unbiased rounding
    fractional : boolean, optional
        If `True`, the cutoffs of `precision` and `min_digits` refer to the
        total number of digits after the decimal point, including leading
        zeros.
        If `False`, `precision` and `min_digits` refer to the total number of
        significant digits, before or after the decimal point, ignoring leading
        zeros.
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    pad_right : non-negative integer, optional
        Pad the right side of the string with whitespace until at least that
        many characters are to the right of the decimal point.
    min_digits : non-negative integer or None, optional
        Minimum number of digits to print. Only has an effect if `unique=True`
        in which case additional digits past those necessary to uniquely
        identify the value may be printed, rounding the last additional digit.

        .. versionadded:: 1.21.0

    Returns
    -------
    """
    # rep : string
    # The string representation of the floating point value

    # See Also
    # --------
    # format_float_scientific

    # Examples
    # --------
    # >>> np.format_float_positional(np.float32(np.pi))
    # '3.1415927'
    # >>> np.format_float_positional(np.float16(np.pi))
    # '3.14'
    # >>> np.format_float_positional(np.float16(0.3))
    # '0.3'
    # >>> np.format_float_positional(np.float16(0.3), unique=False, precision=10)
    # '0.3000488281'
    """
    precision = _none_or_positive_arg(precision, 'precision')
    # 将 precision 参数转换为非空或正值
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    # 将 pad_left 参数转换为非空或正值
    pad_right = _none_or_positive_arg(pad_right, 'pad_right')
    # 将 pad_right 参数转换为非空或正值
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    # 将 min_digits 参数转换为非空或正值
    if not fractional and precision == 0:
        # 如果不是小数且精度为0，则引发值错误异常
        raise ValueError("precision must be greater than 0 if "
                         "fractional=False")
    if min_digits > 0 and precision > 0 and min_digits > precision:
        # 如果 min_digits 大于0且精度大于0且 min_digits 大于精度，则引发值错误异常
        raise ValueError("min_digits must be less than or equal to precision")
    # 调用 dragon4_positional 函数，返回浮点数的字符串表示
    return dragon4_positional(x, precision=precision, unique=unique,
                              fractional=fractional, trim=trim,
                              sign=sign, pad_left=pad_left,
                              pad_right=pad_right, min_digits=min_digits)
class IntegerFormat:
    def __init__(self, data, sign='-'):
        # 如果数据大小大于0，则计算数据的最大值和最小值
        if data.size > 0:
            data_max = np.max(data)
            data_min = np.min(data)
            # 计算数据的最大值的字符串长度
            data_max_str_len = len(str(data_max))
            # 如果符号为' '且数据最小值小于0，则将符号设为'-'
            if sign == ' ' and data_min < 0:
                sign = '-'
            # 如果数据的最大值大于等于0且符号在"+ "中，则最大字符串长度加1
            if data_max >= 0 and sign in "+ ":
                data_max_str_len += 1
            # 计算最终的最大字符串长度，考虑最大值和最小值的字符串长度
            max_str_len = max(data_max_str_len,
                              len(str(data_min)))
        else:
            max_str_len = 0
        # 根据最大字符串长度和符号创建格式化字符串
        self.format = f'{{:{sign}{max_str_len}d}}'

    def __call__(self, x):
        # 使用格式化字符串将x格式化为字符串
        return self.format.format(x)


class BoolFormat:
    def __init__(self, data, **kwargs):
        # 如果数据不是0维数组，为了使" True"和"False"对齐，添加额外的空格
        self.truestr = ' True' if data.shape != () else 'True'

    def __call__(self, x):
        # 根据x的值返回对应的字符串表示
        return self.truestr if x else "False"


class ComplexFloatingFormat:
    """ Formatter for subtypes of np.complexfloating """
    def __init__(self, x, precision, floatmode, suppress_small,
                 sign=False, *, legacy=None):
        # 为了向后兼容，接受布尔值作为符号
        if isinstance(sign, bool):
            sign = '+' if sign else '-'

        # 根据实部和虚部分别创建浮点数格式化对象
        floatmode_real = floatmode_imag = floatmode
        if legacy <= 113:
            floatmode_real = 'maxprec_equal'
            floatmode_imag = 'maxprec'

        self.real_format = FloatingFormat(
            x.real, precision, floatmode_real, suppress_small,
            sign=sign, legacy=legacy
        )
        self.imag_format = FloatingFormat(
            x.imag, precision, floatmode_imag, suppress_small,
            sign='+', legacy=legacy
        )

    def __call__(self, x):
        # 使用实部和虚部的格式化对象分别格式化x，最后将虚部结果中添加'j'
        r = self.real_format(x.real)
        i = self.imag_format(x.imag)
        sp = len(i.rstrip())
        i = i[:sp] + 'j' + i[sp:]
        return r + i


class _TimelikeFormat:
    def __init__(self, data):
        # 选择出非 NaT 的元素，并计算它们的最大最小值的字符串长度
        non_nat = data[~isnat(data)]
        if len(non_nat) > 0:
            max_str_len = max(len(self._format_non_nat(np.max(non_nat))),
                              len(self._format_non_nat(np.min(non_nat))))
        else:
            max_str_len = 0
        # 如果数据中包含 NaT，则最大字符串长度至少为5
        if len(non_nat) < data.size:
            max_str_len = max(max_str_len, 5)
        # 根据最大字符串长度创建格式化字符串和 NaT 的表示
        self._format = '%{}s'.format(max_str_len)
        self._nat = "'NaT'".rjust(max_str_len)

    def _format_non_nat(self, x):
        # 在子类中实现，用于格式化非 NaT 元素
        raise NotImplementedError

    def __call__(self, x):
        # 根据是否为 NaT，返回对应的字符串表示
        if isnat(x):
            return self._nat
        else:
            return self._format % self._format_non_nat(x)


class DatetimeFormat(_TimelikeFormat):
    # DatetimeFormat 类继承自 _TimelikeFormat 类，因此不需要重复注释
    # 初始化函数，用于初始化对象的各个属性
    def __init__(self, x, unit=None, timezone=None, casting='same_kind',
                 legacy=False):
        # 如果未提供单位参数，则根据数据类型推断时间单位
        if unit is None:
            if x.dtype.kind == 'M':
                unit = datetime_data(x.dtype)[0]
            else:
                unit = 's'

        # 如果未提供时区参数，默认为'naive'（即无时区）
        if timezone is None:
            timezone = 'naive'
        
        # 设置对象的属性
        self.timezone = timezone  # 设置时区
        self.unit = unit  # 设置时间单位
        self.casting = casting  # 设置类型转换方式
        self.legacy = legacy  # 设置是否使用旧版本特性

        # 调用父类的初始化方法，必须在上述属性配置之后调用
        super().__init__(x)

    # 对象的调用方法，根据对象的属性来处理输入数据 x
    def __call__(self, x):
        # 如果 legacy 属性小于等于 113，则调用特定的格式化方法 _format_non_nat
        if self.legacy <= 113:
            return self._format_non_nat(x)
        # 否则调用父类的调用方法
        return super().__call__(x)

    # 格式化非 NaT（非时间点缺失值）的方法
    def _format_non_nat(self, x):
        # 调用 datetime_as_string 函数，将时间 x 格式化成字符串
        return "'%s'" % datetime_as_string(x,
                                    unit=self.unit,
                                    timezone=self.timezone,
                                    casting=self.casting)
class TimedeltaFormat(_TimelikeFormat):
    # 定义一个TimedeltaFormat类，继承自_TimelikeFormat类
    def _format_non_nat(self, x):
        # 定义一个方法_format_non_nat，返回x的字符串形式，数据类型转换为'i8'
        return str(x.astype('i8'))


class SubArrayFormat:
    # 定义一个SubArrayFormat类
    def __init__(self, format_function, **options):
        # 构造方法，接收格式化函数和额外的参数
        self.format_function = format_function
        self.threshold = options['threshold']
        self.edge_items = options['edgeitems']

    def __call__(self, a):
        # 实现括号运算符，接收数组a作为输入
        self.summary_insert = "..." if a.size > self.threshold else ""
        return self.format_array(a)

    def format_array(self, a):
        # 格式化数组a
        if np.ndim(a) == 0:
            return self.format_function(a)

        if self.summary_insert and a.shape[0] > 2*self.edge_items:
            formatted = (
                [self.format_array(a_) for a_ in a[:self.edge_items]]
                + [self.summary_insert]
                + [self.format_array(a_) for a_ in a[-self.edge_items:]]
            )
        else:
            formatted = [self.format_array(a_) for a_ in a]

        return "[" + ", ".join(formatted) + "]"


class StructuredVoidFormat:
    """
    Formatter for structured np.void objects.

    This does not work on structured alias types like
    np.dtype(('i4', 'i2,i2')), as alias scalars lose their field information,
    and the implementation relies upon np.void.__getitem__.
    """
    # 结构化np.void对象的格式化器
    def __init__(self, format_functions):
        # 初始化方法，接收格式化函数列表
        self.format_functions = format_functions

    @classmethod
    def from_data(cls, data, **options):
        """
        This is a second way to initialize StructuredVoidFormat,
        using the raw data as input. Added to avoid changing
        the signature of __init__.
        """
        # 通过数据初始化StructuredVoidFormat的另一种方式，接收原始数据和额外参数
        format_functions = []
        for field_name in data.dtype.names:
            format_function = _get_format_function(data[field_name], **options)
            if data.dtype[field_name].shape != ():
                format_function = SubArrayFormat(format_function, **options)
            format_functions.append(format_function)
        return cls(format_functions)

    def __call__(self, x):
        # 实现括号运算符，接收x作为输入
        str_fields = [
            format_function(field)
            for field, format_function in zip(x, self.format_functions)
        ]
        if len(str_fields) == 1:
            return "({},)".format(str_fields[0])
        else:
            return "({})".format(", ".join(str_fields))


def _void_scalar_to_string(x, is_repr=True):
    """
    Implements the repr for structured-void scalars. It is called from the
    scalartypes.c.src code, and is placed here because it uses the elementwise
    formatters defined above.
    """
    # 实现结构化void标量的repr。被scalartypes.c.src代码调用，放在这里是因为它使用了上面定义的按元素格式化器。
    options = _format_options.copy()

    if options["legacy"] <= 125:
        return StructuredVoidFormat.from_data(array(x), **_format_options)(x)

    if options.get('formatter') is None:
        options['formatter'] = {}
    options['formatter'].setdefault('float_kind', str)
    val_repr = StructuredVoidFormat.from_data(array(x), **options)(x)
    if not is_repr:
        return val_repr
    cls = type(x)
    # 构建完整的类全限定名，将类所在模块中的 "numpy" 替换为 "np"，并添加类名
    cls_fqn = cls.__module__.replace("numpy", "np") + "." + cls.__name__

    # 根据数组 x 的数据类型创建一个 void 类型的 NumPy 数据类型对象
    void_dtype = np.dtype((np.void, x.dtype))

    # 返回一个格式化的字符串，表示实例化类 cls_fqn，包括 val_repr 的表达形式和 void_dtype 的数据类型
    return f"{cls_fqn}({val_repr}, dtype={void_dtype!s})"
# 定义一个列表，包含几种数据类型：int_, float64, complex128, _nt.bool
_typelessdata = [int_, float64, complex128, _nt.bool]

# 判断给定的数据类型是否由其值的表示隐含
def dtype_is_implied(dtype):
    """
    根据其值的表示确定给定的数据类型是否隐含。

    Parameters
    ----------
    dtype : dtype
        数据类型

    Returns
    -------
    implied : bool
        如果数据类型由其值的表示隐含，则返回True。

    Examples
    --------
    >>> np._core.arrayprint.dtype_is_implied(int)
    True
    >>> np.array([1, 2, 3], int)
    array([1, 2, 3])
    >>> np._core.arrayprint.dtype_is_implied(np.int8)
    False
    >>> np.array([1, 2, 3], np.int8)
    array([1, 2, 3], dtype=int8)
    """
    dtype = np.dtype(dtype)
    # 如果格式选项中的'legacy'小于等于113且数据类型是np.bool，则返回False
    if _format_options['legacy'] <= 113 and dtype.type == np.bool:
        return False

    # 如果数据类型具有结构或者有命名字段，则返回False
    if dtype.names is not None:
        return False

    # 如果数据类型不是本机类型（即可能涉及字节顺序），除非大小为1（例如，int8, bool），否则返回False
    if not dtype.isnative:
        return False

    # 如果数据类型的类型在_typelessdata列表中，则返回True
    return dtype.type in _typelessdata


# 将数据类型转换为其等效的短形式表示
def dtype_short_repr(dtype):
    """
    将数据类型转换为一个短形式表示，该表示在评估时等同于原始数据类型。

    The intent is roughly that the following holds

    >>> from numpy import *
    >>> dt = np.int64([1, 2]).dtype
    >>> assert eval(dtype_short_repr(dt)) == dt
    """
    if type(dtype).__repr__ != np.dtype.__repr__:
        # 对于用户定义的数据类型，可能需要自定义repr，这里可能需要移动逻辑
        return repr(dtype)
    if dtype.names is not None:
        # 结构化数据类型给出一个列表或元组的repr
        return str(dtype)
    elif issubclass(dtype.type, flexible):
        # 分开处理这些，以免给出如str256这样的垃圾表示
        return "'%s'" % str(dtype)

    typename = dtype.name
    if not dtype.isnative:
        # 处理像dtype('<u2')这样的情况，它与已建立的数据类型相同（在本例中是uint16）
        # 但它们具有不同的字节顺序。
        return "'%s'" % str(dtype)
    # 对于不能表示为Python变量名的类型名称进行引号处理
    if typename and not (typename[0].isalpha() and typename.isalnum()):
        typename = repr(typename)
    return typename


# 内部版本的array_repr()函数的实现，允许覆盖array2string
def _array_repr_implementation(
        arr, max_line_width=None, precision=None, suppress_small=None,
        array2string=array2string):
    """
    内部版本的array_repr()函数的实现，允许覆盖array2string。
    """
    override_repr = _format_options["override_repr"]
    if override_repr is not None:
        return override_repr(arr)

    if max_line_width is None:
        max_line_width = _format_options['linewidth']

    if type(arr) is not ndarray:
        class_name = type(arr).__name__
    else:
        class_name = "array"

    skipdtype = dtype_is_implied(arr.dtype) and arr.size > 0

    prefix = class_name + "("
    suffix = ")" if skipdtype else ","
    # 检查是否满足遗留格式选项小于等于 113，并且数组是标量且没有结构化类型
    if (_format_options['legacy'] <= 113 and
            arr.shape == () and not arr.dtype.names):
        # 将数组的单个元素转换为字符串表示形式
        lst = repr(arr.item())
    # 如果数组不为空，或者数组的形状为 (0,)
    elif arr.size > 0 or arr.shape == (0,):
        # 将数组转换为字符串表示形式，使用指定的参数
        lst = array2string(arr, max_line_width, precision, suppress_small,
                           ', ', prefix, suffix=suffix)
    else:  # 如果数组长度为零，但形状不是 (0,)
        # 显示空列表，以及数组的形状的字符串表示形式
        lst = "[], shape=%s" % (repr(arr.shape),)

    # 拼接数组的字符串表示形式和前缀后缀
    arr_str = prefix + lst + suffix

    # 如果需要跳过数据类型信息，则直接返回拼接好的数组字符串表示形式
    if skipdtype:
        return arr_str

    # 获取数组数据类型的简短表示形式
    dtype_str = "dtype={})".format(dtype_short_repr(arr.dtype))

    # 计算是否需要将数据类型信息放在新的一行：如果加上数据类型信息会超过最大行宽，则需要换行显示
    last_line_len = len(arr_str) - (arr_str.rfind('\n') + 1)
    spacer = " "
    # 如果遗留格式选项小于等于 113
    if _format_options['legacy'] <= 113:
        # 如果数组的数据类型是 flexible 类型的子类，则在数据类型信息之前换行显示
        if issubclass(arr.dtype.type, flexible):
            spacer = '\n' + ' '*len(class_name + "(")
    # 否则，如果加上数据类型信息会超过最大行宽，则在数据类型信息之前换行显示
    elif last_line_len + len(dtype_str) + 1 > max_line_width:
        spacer = '\n' + ' '*len(class_name + "(")

    # 返回拼接好的数组字符串表示形式和数据类型信息，以适当的空格分隔
    return arr_str + spacer + dtype_str
# 定义一个辅助函数，用于派发数组的字符串表示函数
def _array_repr_dispatcher(
        arr, max_line_width=None, precision=None, suppress_small=None):
    return (arr,)

# 使用装饰器将_array_repr_dispatcher函数注册为array_repr函数的派发函数，指定模块为'numpy'
@array_function_dispatch(_array_repr_dispatcher, module='numpy')
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """
    返回数组的字符串表示形式。

    Parameters
    ----------
    arr : ndarray
        输入的数组。
    max_line_width : int, optional
        如果文本长度超过`max_line_width`，则插入换行符。
        默认为 ``numpy.get_printoptions()['linewidth']``。
    precision : int, optional
        浮点数精度。
        默认为 ``numpy.get_printoptions()['precision']``。
    suppress_small : bool, optional
        将接近零的数字表示为零；默认为False。
        接近零的定义取决于precision：例如，如果精度为8，
        绝对值小于5e-9的数字将被表示为零。
        默认为 ``numpy.get_printoptions()['suppress']``。

    Returns
    -------
    string : str
      数组的字符串表示形式。

    See Also
    --------
    array_str, array2string, set_printoptions

    Examples
    --------
    >>> np.array_repr(np.array([1,2]))
    'array([1, 2])'
    >>> np.array_repr(np.ma.array([0.]))
    'MaskedArray([0.])'
    >>> np.array_repr(np.array([], np.int32))
    'array([], dtype=int32)'

    >>> x = np.array([1e-6, 4e-7, 2, 3])
    >>> np.array_repr(x, precision=6, suppress_small=True)
    'array([0.000001,  0.      ,  2.      ,  3.      ])'

    """
    # 调用_array_repr_implementation函数来实现array_repr的具体逻辑
    return _array_repr_implementation(
        arr, max_line_width, precision, suppress_small)

# 递归保护装饰器函数，确保处理字节类型的变量时不会出错
@_recursive_guard()
def _guarded_repr_or_str(v):
    if isinstance(v, bytes):
        return repr(v)
    return str(v)

# 实现数组字符串表示的内部函数，允许重写array2string
def _array_str_implementation(
        a, max_line_width=None, precision=None, suppress_small=None,
        array2string=array2string):
    """Internal version of array_str() that allows overriding array2string."""
    # 如果_legacy为113以下，且数组a是0维且无字段名，则返回其item的字符串表示
    if (_format_options['legacy'] <= 113 and
            a.shape == () and not a.dtype.names):
        return str(a.item())

    # 对于0维数组的情况，返回其标量值的字符串表示
    if a.shape == ():
        # 获取标量并调用其str方法，避免对子类出现问题，其中使用ndarray的getindex来索引
        return _guarded_repr_or_str(np.ndarray.__getitem__(a, ()))

    # 调用array2string函数来生成数组的字符串表示
    return array2string(a, max_line_width, precision, suppress_small, ' ', "")

# 定义一个辅助函数，用于派发数组的字符串表示函数
def _array_str_dispatcher(
        a, max_line_width=None, precision=None, suppress_small=None):
    return (a,)

# 使用装饰器将_array_str_dispatcher函数注册为array_str函数的派发函数，指定模块为'numpy'
@array_function_dispatch(_array_str_dispatcher, module='numpy')
# 定义函数 array_str，用于返回数组的字符串表示形式
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.

    See Also
    --------
    array2string, array_repr, set_printoptions

    Examples
    --------
    >>> np.array_str(np.arange(3))
    '[0 1 2]'

    """
    # 调用内部函数 _array_str_implementation 处理数组的字符串表示形式，并返回结果
    return _array_str_implementation(
        a, max_line_width, precision, suppress_small)


# 如果 __array_function__ 被禁用，需要定义以下两个变量
# 获取 array2string 的原始实现，如果有的话
_array2string_impl = getattr(array2string, '__wrapped__', array2string)
# 创建默认的 array_str 函数，使用 functools.partial 函数预设部分参数
_default_array_str = functools.partial(_array_str_implementation,
                                       array2string=_array2string_impl)
# 创建默认的 array_repr 函数，使用 functools.partial 函数预设部分参数
_default_array_repr = functools.partial(_array_repr_implementation,
                                        array2string=_array2string_impl)
```