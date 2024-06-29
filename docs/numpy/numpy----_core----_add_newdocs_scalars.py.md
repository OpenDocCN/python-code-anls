# `.\numpy\numpy\_core\_add_newdocs_scalars.py`

```
"""
This file is separate from ``_add_newdocs.py`` so that it can be mocked out by
our sphinx ``conf.py`` during doc builds, where we want to avoid showing
platform-dependent information.
"""

# 导入系统操作模块和操作系统模块
import sys
import os
# 导入numpy的数据类型模块
from numpy._core import dtype
# 导入numpy的数据类型别名模块
from numpy._core import numerictypes as _numerictypes
# 导入numpy的函数基础模块中的add_newdoc函数
from numpy._core.function_base import add_newdoc

##############################################################################
#
# Documentation for concrete scalar classes
#
##############################################################################

# 定义一个函数，用于生成类型别名的生成器
def numeric_type_aliases(aliases):
    def type_aliases_gen():
        # 遍历提供的别名列表
        for alias, doc in aliases:
            try:
                # 尝试获取_numerictypes模块中的别名对应的类型
                alias_type = getattr(_numerictypes, alias)
            except AttributeError:
                # 如果别名在_numerictypes模块中不存在，则跳过
                # 不同平台可能存在的别名集合不同
                pass
            else:
                # 如果获取成功，则生成一个元组(alias_type, alias, doc)
                yield (alias_type, alias, doc)
    # 返回类型别名生成器生成的列表
    return list(type_aliases_gen())

# 定义一个可能的类型别名列表
possible_aliases = numeric_type_aliases([
    ('int8', '8-bit signed integer (``-128`` to ``127``)'),
    ('int16', '16-bit signed integer (``-32_768`` to ``32_767``)'),
    ('int32', '32-bit signed integer (``-2_147_483_648`` to ``2_147_483_647``)'),
    ('int64', '64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``)'),
    ('intp', 'Signed integer large enough to fit pointer, compatible with C ``intptr_t``'),
    ('uint8', '8-bit unsigned integer (``0`` to ``255``)'),
    ('uint16', '16-bit unsigned integer (``0`` to ``65_535``)'),
    ('uint32', '32-bit unsigned integer (``0`` to ``4_294_967_295``)'),
    ('uint64', '64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``)'),
    ('uintp', 'Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``'),
    ('float16', '16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa'),
    ('float32', '32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa'),
    ('float64', '64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa'),
    ('float96', '96-bit extended-precision floating-point number type'),
    ('float128', '128-bit extended-precision floating-point number type'),
    ('complex64', 'Complex number type composed of 2 32-bit-precision floating-point numbers'),
    ('complex128', 'Complex number type composed of 2 64-bit-precision floating-point numbers'),
    ('complex192', 'Complex number type composed of 2 96-bit extended-precision floating-point numbers'),
    ('complex256', 'Complex number type composed of 2 128-bit extended-precision floating-point numbers'),
    ])

# 定义一个函数，尝试获取平台和机器信息
def _get_platform_and_machine():
    try:
        # 尝试获取当前操作系统的信息
        system, _, _, _, machine = os.uname()
    # 捕获 AttributeError 异常
    except AttributeError:
        # 获取当前操作系统平台信息
        system = sys.platform
        # 如果系统为 'win32'，获取处理器架构信息
        if system == 'win32':
            machine = os.environ.get('PROCESSOR_ARCHITEW6432', '') \
                    or os.environ.get('PROCESSOR_ARCHITECTURE', '')
        else:
            # 对于非 'win32' 系统，设置处理器架构为 'unknown'
            machine = 'unknown'
    # 返回系统平台信息和处理器架构信息
    return system, machine
_system, _machine = _get_platform_and_machine()
_doc_alias_string = f":Alias on this platform ({_system} {_machine}):"

# 定义函数，用于为给定的标量类型添加新的文档
def add_newdoc_for_scalar_type(obj, fixed_aliases, doc):
    # 获取 _numerictypes 模块中名为 obj 的对象
    o = getattr(_numerictypes, obj)

    # 获取对象 o 的字符编码
    character_code = dtype(o).char

    # 如果 obj 不等于 o 的名称，则生成规范名称文档
    canonical_name_doc = "" if obj == o.__name__ else \
                        f":Canonical name: `numpy.{obj}`\n    "

    # 根据 fixed_aliases 生成别名文档
    if fixed_aliases:
        alias_doc = ''.join(f":Alias: `numpy.{alias}`\n    "
                            for alias in fixed_aliases)
    else:
        alias_doc = ''

    # 根据 possible_aliases 中的信息生成别名文档，并结合 _doc_alias_string
    alias_doc += ''.join(f"{_doc_alias_string} `numpy.{alias}`: {doc}.\n    "
                         for (alias_type, alias, doc) in possible_aliases if alias_type is o)

    # 构建完整的文档字符串，包括传入的 doc 参数以及其他详细信息
    docstring = f"""
    {doc.strip()}  # 去除首尾空白字符

    :Character code: ``'{character_code}'``
    {canonical_name_doc}{alias_doc}
    """

    # 调用 add_newdoc 函数，将文档添加到 'numpy._core.numerictypes' 模块的 obj 对象上
    add_newdoc('numpy._core.numerictypes', obj, docstring)

# 布尔类型的文档字符串
_bool_docstring = (
    """
    Boolean type (True or False), stored as a byte.

    .. warning::

       The :class:`bool` type is not a subclass of the :class:`int_` type
       (the :class:`bool` is not even a number type). This is different
       than Python's default implementation of :class:`bool` as a
       sub-class of :class:`int`.
    """
)

# 分别为 'bool', 'bool_', 'byte', 'short', 'intc' 等类型添加新的文档
add_newdoc_for_scalar_type('bool', [], _bool_docstring)
add_newdoc_for_scalar_type('bool_', [], _bool_docstring)
add_newdoc_for_scalar_type('byte', [],
    """
    Signed integer type, compatible with C ``char``.
    """)
add_newdoc_for_scalar_type('short', [],
    """
    Signed integer type, compatible with C ``short``.
    """)
add_newdoc_for_scalar_type('intc', [],
    """
    Signed integer type, compatible with C ``int``.
    """)

# TODO: These docs probably need an if to highlight the default rather than
#       the C-types (and be correct).

# 为 'int_' 类型添加新的文档，描述其作为默认有符号整数类型的特性
add_newdoc_for_scalar_type('int_', [],
    """
    Default signed integer type, 64bit on 64bit systems and 32bit on 32bit
    systems.
    """)

# 为 'longlong', 'ubyte', 'ushort', 'uintc' 等类型添加新的文档
add_newdoc_for_scalar_type('longlong', [],
    """
    Signed integer type, compatible with C ``long long``.
    """)
add_newdoc_for_scalar_type('ubyte', [],
    """
    Unsigned integer type, compatible with C ``unsigned char``.
    """)
add_newdoc_for_scalar_type('ushort', [],
    """
    Unsigned integer type, compatible with C ``unsigned short``.
    """)
add_newdoc_for_scalar_type('uintc', [],
    """
    Unsigned integer type, compatible with C ``unsigned int``.
    """)

# 为 'uint', 'ulonglong', 'half', 'single' 等类型添加新的文档
add_newdoc_for_scalar_type('uint', [],
    """
    Unsigned signed integer type, 64bit on 64bit systems and 32bit on 32bit
    systems.
    """)
add_newdoc_for_scalar_type('ulonglong', [],
    """
    Signed integer type, compatible with C ``unsigned long long``.
    """)
add_newdoc_for_scalar_type('half', [],
    """
    Half-precision floating-point number type.
    """)
add_newdoc_for_scalar_type('single', [],
    """
    Single-precision floating-point number type.
    """)
    # 定义一个新的类`Single-precision floating-point number type`，它兼容C语言中的`float`。
    Single-precision floating-point number type, compatible with C ``float``.
    """
# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('double', [],
    """
    双精度浮点数类型，与 Python 中的 :class:`float` 和 C 的 ``double`` 兼容。
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('longdouble', [],
    """
    扩展精度浮点数类型，与 C 的 ``long double`` 兼容，但不一定与 IEEE 754 四倍精度兼容。
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('csingle', [],
    """
    复数类型，由两个单精度浮点数组成。
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('cdouble', [],
    """
    复数类型，由两个双精度浮点数组成，与 Python 的 :class:`complex` 兼容。
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('clongdouble', [],
    """
    复数类型，由两个扩展精度浮点数组成。
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('object_', [],
    """
    任意 Python 对象。
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('str_', [],
    r"""
    Unicode 字符串。

    此类型会去除尾部的空字符。

    >>> s = np.str_("abc\x00")
    >>> s
    'abc'

    不同于内置的 :class:`str`，此类型支持
    :ref:`python:bufferobjects`，以 UCS4 的形式展示其内容：

    >>> m = memoryview(np.str_("abc"))
    >>> m.format
    '3w'
    >>> m.tobytes()
    b'a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00'
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('bytes_', [],
    r"""
    字节串。

    在数组中使用时，此类型会去除尾部的空字节。
    """)

# 为指定的标量类型添加新的文档字符串
add_newdoc_for_scalar_type('void', [],
    r"""
    np.void(length_or_data, /, dtype=None)

    创建一个新的结构化或非结构化的空标量。

    参数
    ----------
    length_or_data : int, array-like, bytes-like, object
       长度或字节数据，用于创建非结构化的空标量。当指定 dtype 时，可以是要存储在新标量中的数据。
       这可以是一个类似数组的对象，此时可能返回一个数组。
    dtype : dtype, optional
       如果提供，则为新标量的数据类型。该数据类型必须是 "void" 类型（即结构化或非结构化的空标量）。

       .. versionadded:: 1.24

    注意
    -----
    由于历史原因和空标量可以表示任意字节数据和结构化数据类型的特性，
    空构造函数有三种调用约定：

    1. ``np.void(5)`` 创建一个填充有五个 ``\0`` 字节的 ``dtype="V5"`` 标量。其中的 5 可以是 Python 或 NumPy 的整数。
    2. ``np.void(b"bytes-like")`` 从字节串创建一个空标量。数据类型的项大小将匹配字节串长度，这里是 ``"V10"``。
    3. 当传递 ``dtype=`` 时，调用与创建数组类似。但是返回的是空标量而不是数组。

    请参阅示例，展示了所有三种不同的约定。

    示例
    """)
    # 创建一个空的 NumPy void 对象，参数为整数 5，默认使用 8 字节填充
    np.void(5)
    
    # 创建一个 NumPy void 对象，参数为字节序列 b'abcd'，使用对应的 ASCII 码填充
    np.void(b'\x00\x00\x00\x00\x00')
    
    # 创建一个 NumPy void 对象，参数为元组 (3.2, b'eggs')，指定数据类型为浮点数和字节串，分别对应字段 'f0' 和 'f1'
    np.void((3.2, b'eggs'), dtype=[('f0', '<f8'), ('f1', 'S5')])
    
    # 创建一个 NumPy void 对象，参数为整数 3，指定数据类型为带有字段 'x' 和 'y' 的字节串
    np.void((3, 3), dtype=[('x', 'i1'), ('y', 'i1')])
# 为 datetime64 类型添加新的文档字符串
add_newdoc_for_scalar_type('datetime64', [],
    """
    如果从 64 位整数创建，则表示相对于 ``1970-01-01T00:00:00`` 的偏移量。
    如果从字符串创建，则字符串可以是 ISO 8601 日期或日期时间格式。
    
    当解析包含时区的字符串以创建 datetime 对象时（以 'Z' 或时区偏移量结尾），将丢弃时区并给出用户警告。
    
    Datetime64 对象应被视为 UTC，因此偏移量为 +0000。

    >>> np.datetime64(10, 'Y')
    np.datetime64('1980')
    >>> np.datetime64('1980', 'Y')
    np.datetime64('1980')
    >>> np.datetime64(10, 'D')
    np.datetime64('1970-01-11')

    更多信息请参见 :ref:`arrays.datetime`。
    """)

# 为 timedelta64 类型添加新的文档字符串
add_newdoc_for_scalar_type('timedelta64', [],
    """
    作为 64 位整数存储的 timedelta。

    更多信息请参见 :ref:`arrays.datetime`。
    """)

# 为 integer 类型的 is_integer 方法添加新的文档字符串
add_newdoc('numpy._core.numerictypes', "integer", ('is_integer',
    """
    integer.is_integer() -> bool

    如果数是有限的整数值，则返回 ``True``。

    .. versionadded:: 1.22

    示例
    --------
    >>> np.int64(-2).is_integer()
    True
    >>> np.uint32(5).is_integer()
    True
    """))

# 为浮点类型（half, single, double, longdouble）的 as_integer_ratio 方法添加新的文档字符串
for float_name in ('half', 'single', 'double', 'longdouble'):
    add_newdoc('numpy._core.numerictypes', float_name, ('as_integer_ratio',
        """
        {ftype}.as_integer_ratio() -> (int, int)

        返回一对整数，其比率恰好等于原始浮点数，并且具有正的分母。
        对无穷大返回 `OverflowError`，对 NaN 返回 `ValueError`。

        >>> np.{ftype}(10.0).as_integer_ratio()
        (10, 1)
        >>> np.{ftype}(0.0).as_integer_ratio()
        (0, 1)
        >>> np.{ftype}(-.25).as_integer_ratio()
        (-1, 4)
        """.format(ftype=float_name)))

    # 为浮点类型（half, single, double, longdouble）的 is_integer 方法添加新的文档字符串
    add_newdoc('numpy._core.numerictypes', float_name, ('is_integer',
        f"""
        {float_name}.is_integer() -> bool

        如果浮点数是有限的整数值，则返回 ``True``；否则返回 ``False``。

        .. versionadded:: 1.22

        示例
        --------
        >>> np.{float_name}(-2.0).is_integer()
        True
        >>> np.{float_name}(3.2).is_integer()
        False
        """))

# 为整数类型（int8, uint8, int16, uint16, int32, uint32, int64, uint64）添加新的文档字符串
for int_name in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
        'int64', 'uint64'):
    # 通过检查类型码，为有符号情况添加负数示例
    # 使用 add_newdoc 函数添加文档字符串到指定的 numpy._core.numerictypes 模块的 int_name 中
    add_newdoc('numpy._core.numerictypes', int_name, ('bit_count',
        # 定义 bit_count 方法的文档字符串，描述其作用和用法
        f"""
        {int_name}.bit_count() -> int

        Computes the number of 1-bits in the absolute value of the input.
        Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

        Examples
        --------
        >>> np.{int_name}(127).bit_count()
        7""" +
        # 如果 int_name 对应的数据类型是小写，则添加额外的例子说明
        (f"""
        >>> np.{int_name}(-127).bit_count()
        7
        """ if dtype(int_name).char.islower() else "")))
```