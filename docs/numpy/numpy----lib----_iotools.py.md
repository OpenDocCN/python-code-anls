# `.\numpy\numpy\lib\_iotools.py`

```py
"""A collection of functions designed to help I/O with ascii files.

"""
__docformat__ = "restructuredtext en"

import numpy as np
import numpy._core.numeric as nx
from numpy._utils import asbytes, asunicode


def _decode_line(line, encoding=None):
    """Decode bytes from binary input streams.

    Defaults to decoding from 'latin1'. That differs from the behavior of
    np.compat.asunicode that decodes from 'ascii'.

    Parameters
    ----------
    line : str or bytes
         Line to be decoded.
    encoding : str
         Encoding used to decode `line`.

    Returns
    -------
    decoded_line : str

    """
    # 如果输入是字节流，则根据指定的编码或默认使用'latin1'进行解码成字符串
    if type(line) is bytes:
        if encoding is None:
            encoding = "latin1"
        line = line.decode(encoding)

    return line


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def _is_bytes_like(obj):
    """
    Check whether obj behaves like a bytes object.
    """
    try:
        obj + b''
    except (TypeError, ValueError):
        return False
    return True


def has_nested_fields(ndtype):
    """
    Returns whether one or several fields of a dtype are nested.

    Parameters
    ----------
    ndtype : dtype
        Data-type of a structured array.

    Raises
    ------
    AttributeError
        If `ndtype` does not have a `names` attribute.

    Examples
    --------
    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float)])
    >>> np.lib._iotools.has_nested_fields(dt)
    False

    """
    # 检查结构化数据类型是否包含嵌套字段
    for name in ndtype.names or ():
        if ndtype[name].names is not None:
            return True
    return False


def flatten_dtype(ndtype, flatten_base=False):
    """
    Unpack a structured data-type by collapsing nested fields and/or fields
    with a shape.

    Note that the field names are lost.

    Parameters
    ----------
    ndtype : dtype
        The datatype to collapse
    flatten_base : bool, optional
       If True, transform a field with a shape into several fields. Default is
       False.

    Examples
    --------
    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
    ...                ('block', int, (2, 3))])
    >>> np.lib._iotools.flatten_dtype(dt)
    [dtype('S4'), dtype('float64'), dtype('float64'), dtype('int64')]
    >>> np.lib._iotools.flatten_dtype(dt, flatten_base=True)
    [dtype('S4'),
     dtype('float64'),
     dtype('float64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64')]

    """
    # 展开结构化数据类型，将嵌套字段和带有形状的字段展开为单一字段，返回结果中不包含字段名
    names = ndtype.names
    if names is None:
        if flatten_base:
            return [ndtype.base] * int(np.prod(ndtype.shape))
        return [ndtype.base]
    else:
        # 如果条件不满足，则执行以下代码块
        types = []
        # 初始化一个空列表 types，用于存储数据类型信息

        for field in names:
            # 遍历 names 列表中的每一个字段名
            info = ndtype.fields[field]
            # 获取字段名在 ndtype 结构中的信息
            flat_dt = flatten_dtype(info[0], flatten_base)
            # 调用 flatten_dtype 函数，将字段信息展开成扁平化的数据类型列表
            types.extend(flat_dt)
            # 将扁平化的数据类型列表添加到 types 列表中

        # 返回最终的 types 列表，其中包含了所有字段名对应的扁平化数据类型
        return types
    """
    Object to split a string at a given delimiter or at given places.

    Parameters
    ----------
    delimiter : str, int, or sequence of ints, optional
        If a string, character used to delimit consecutive fields.
        If an integer or a sequence of integers, width(s) of each field.
    comments : str, optional
        Character used to mark the beginning of a comment. Default is '#'.
    autostrip : bool, optional
        Whether to strip each individual field. Default is True.

    """

    def autostrip(self, method):
        """
        Wrapper to strip each member of the output of `method`.

        Parameters
        ----------
        method : function
            Function that takes a single argument and returns a sequence of
            strings.

        Returns
        -------
        wrapped : function
            The result of wrapping `method`. `wrapped` takes a single input
            argument and returns a list of strings that are stripped of
            white-space.

        """
        return lambda input: [_.strip() for _ in method(input)]

    def __init__(self, delimiter=None, comments='#', autostrip=True,
                 encoding=None):
        # Convert delimiter and comments to string format
        delimiter = _decode_line(delimiter)
        comments = _decode_line(comments)

        # Set the comments attribute
        self.comments = comments

        # Determine the type of splitter based on delimiter
        # Delimiter is a character
        if (delimiter is None) or isinstance(delimiter, str):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        # Delimiter is a list of field widths
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            # Calculate slice indices based on cumulative sum of widths
            idx = np.cumsum([0] + list(delimiter))
            delimiter = [slice(i, j) for (i, j) in zip(idx[:-1], idx[1:])]
        # Delimiter is a single integer
        elif int(delimiter):
            (_handyman, delimiter) = (
                    self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)

        # Set the delimiter attribute
        self.delimiter = delimiter

        # Set the appropriate splitter method
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman

        # Set the encoding attribute
        self.encoding = encoding

    def _delimited_splitter(self, line):
        """Chop off comments, strip, and split at delimiter. """
        # Remove comments if specified
        if self.comments is not None:
            line = line.split(self.comments)[0]

        # Strip leading and trailing whitespace and newlines
        line = line.strip(" \r\n")

        # Return an empty list if line is empty
        if not line:
            return []

        # Split the line at the delimiter and return the list of fields
        return line.split(self.delimiter)

    def _fixedwidth_splitter(self, line):
        # Remove comments if specified
        if self.comments is not None:
            line = line.split(self.comments)[0]

        # Strip trailing newline characters
        line = line.strip("\r\n")

        # Return an empty list if line is empty
        if not line:
            return []

        # Split the line into fixed width segments defined by delimiter
        fixed = self.delimiter
        slices = [slice(i, i + fixed) for i in range(0, len(line), fixed)]
        return [line[s] for s in slices]
    # 定义一个方法 `_variablewidth_splitter`，接受一个字符串 `line` 作为参数
    def _variablewidth_splitter(self, line):
        # 如果存在注释符号，则将 line 按照注释符号分割，并取第一个分割部分作为新的 line
        if self.comments is not None:
            line = line.split(self.comments)[0]
        # 如果经过注释处理后的 line 为空字符串，则返回一个空列表
        if not line:
            return []
        # 从对象的 `delimiter` 属性中获取切片列表 `slices`
        slices = self.delimiter
        # 返回一个列表，包含根据切片列表 `slices` 从 `line` 中提取的各部分字符
        return [line[s] for s in slices]
    
    # 定义一个方法 `__call__`，接受一个字符串 `line` 作为参数
    def __call__(self, line):
        # 使用 `self.encoding` 解码输入的 `line`，并传递给 `_decode_line` 方法进行处理
        decoded_line = _decode_line(line, self.encoding)
        # 调用对象的 `_handyman` 方法处理解码后的 `line`，并返回结果
        return self._handyman(decoded_line)
# 定义一个名称验证器类 NameValidator
class NameValidator:
    """
    Object to validate a list of strings to use as field names.

    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by '_'. During instantiation, the user can define a list
    of names to exclude, as well as a list of invalid characters. Names in
    the exclusion list are appended a '_' character.

    Once an instance has been created, it can be called with a list of
    names, and a list of valid names will be created.  The `__call__`
    method accepts an optional keyword "default" that sets the default name
    in case of ambiguity. By default this is 'f', so that names will
    default to `f0`, `f1`, etc.

    Parameters
    ----------
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default
        list ['return', 'file', 'print']. Excluded names are appended an
        underscore: for example, `file` becomes `file_` if supplied.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        * If True, field names are case-sensitive.
        * If False or 'upper', field names are converted to upper case.
        * If 'lower', field names are converted to lower case.

        The default value is True.
    replace_space : '_', optional
        Character(s) used in replacement of white spaces.

    Notes
    -----
    Calling an instance of `NameValidator` is the same as calling its
    method `validate`.

    Examples
    --------
    >>> validator = np.lib._iotools.NameValidator()
    >>> validator(['file', 'field2', 'with space', 'CaSe'])
    ('file_', 'field2', 'with_space', 'CaSe')

    >>> validator = np.lib._iotools.NameValidator(excludelist=['excl'],
    ...                                           deletechars='q',
    ...                                           case_sensitive=False)
    >>> validator(['excl', 'field2', 'no_q', 'with space', 'CaSe'])
    ('EXCL', 'FIELD2', 'NO_Q', 'WITH_SPACE', 'CASE')

    """

    # 默认排除列表，包括一些默认不允许作为字段名的关键词
    defaultexcludelist = ['return', 'file', 'print']
    # 默认的无效字符集合，用于从名称中删除无效字符
    defaultdeletechars = set(r"""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")
    # 初始化函数，用于初始化对象的属性和状态
    def __init__(self, excludelist=None, deletechars=None,
                 case_sensitive=None, replace_space='_'):
        # 处理排除列表的逻辑
        if excludelist is None:
            excludelist = []
        # 将默认排除列表与传入的排除列表合并
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        
        # 处理要删除的字符列表的逻辑
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        # 向要删除的字符集合中添加双引号字符
        delete.add('"')
        self.deletechars = delete
        
        # 处理大小写敏感选项的逻辑
        if (case_sensitive is None) or (case_sensitive is True):
            # 如果大小写敏感为None或True，则不改变字符大小写
            self.case_converter = lambda x: x
        elif (case_sensitive is False) or case_sensitive.startswith('u'):
            # 如果大小写敏感为False或以'u'开头，则将字符转换为大写
            self.case_converter = lambda x: x.upper()
        elif case_sensitive.startswith('l'):
            # 如果大小写敏感以'l'开头，则将字符转换为小写
            self.case_converter = lambda x: x.lower()
        else:
            # 如果大小写敏感选项无法识别，则抛出值错误异常
            msg = 'unrecognized case_sensitive value %s.' % case_sensitive
            raise ValueError(msg)
        
        # 设定替换空格的字符
        self.replace_space = replace_space
    # 对字段名称列表进行验证，确保其符合结构化数组的要求

    def validate(self, names, defaultfmt="f%i", nbfields=None):
        """
        Validate a list of strings as field names for a structured array.

        Parameters
        ----------
        names : sequence of str
            Strings to be validated.
        defaultfmt : str, optional
            Default format string, used if validating a given string
            reduces its length to zero.
        nbfields : integer, optional
            Final number of validated names, used to expand or shrink the
            initial list of names.

        Returns
        -------
        validatednames : list of str
            The list of validated field names.

        Notes
        -----
        A `NameValidator` instance can be called directly, which is the
        same as calling `validate`. For examples, see `NameValidator`.

        """
        # 初始检查，处理可能的空输入情况
        if (names is None):
            if (nbfields is None):
                return None
            names = []
        # 如果输入的是单个字符串，转换为单元素列表
        if isinstance(names, str):
            names = [names, ]
        # 如果指定了 nbfields，调整 names 的长度
        if nbfields is not None:
            nbnames = len(names)
            if (nbnames < nbfields):
                names = list(names) + [''] * (nbfields - nbnames)
            elif (nbnames > nbfields):
                names = names[:nbfields]
        
        # 设置一些快捷方式和变量初始化
        deletechars = self.deletechars  # 删除字符列表
        excludelist = self.excludelist  # 排除列表
        case_converter = self.case_converter  # 大小写转换函数
        replace_space = self.replace_space  # 空格替换字符
        
        # 初始化一些变量
        validatednames = []  # 存放验证后的名称
        seen = dict()  # 记录已经出现过的名称及其出现次数
        nbempty = 0  # 记录空名称的数量

        # 遍历输入的每个名称进行验证处理
        for item in names:
            item = case_converter(item).strip()  # 大小写转换并去除首尾空白
            if replace_space:
                item = item.replace(' ', replace_space)  # 替换空格字符
            item = ''.join([c for c in item if c not in deletechars])  # 删除指定字符
            if item == '':
                item = defaultfmt % nbempty  # 如果名称为空，则使用默认格式字符串
                while item in names:
                    nbempty += 1
                    item = defaultfmt % nbempty
                nbempty += 1
            elif item in excludelist:
                item += '_'  # 如果名称在排除列表中，则添加下划线
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)  # 如果名称已经存在，则加上计数后缀
            else:
                validatednames.append(item)  # 否则直接加入验证通过的名称列表
            seen[item] = cnt + 1  # 更新名称计数
        
        return tuple(validatednames)  # 返回验证后的名称列表（元组形式）

    def __call__(self, names, defaultfmt="f%i", nbfields=None):
        return self.validate(names, defaultfmt=defaultfmt, nbfields=nbfields)
    """
    Factory class for function transforming a string into another object
    (int, float).

    After initialization, an instance can be called to transform a string
    into another object. If the string is recognized as representing a
    missing value, a default value is returned.

    Attributes
    ----------
    func : function
        Function used for the conversion.
    default : any
        Default value to return when the input corresponds to a missing
        value.
    type : type
        Type of the output.
    _status : int
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in
        order.
    _locked : bool
        Holds `locked` parameter.

    Parameters
    ----------
    dtype_or_func : {None, dtype, function}, optional
        If a `dtype`, specifies the input data type, used to define a basic
        function and a default value for missing data. For example, when
        `dtype` is float, the `func` attribute is set to `float` and the
        default value to `np.nan`.  If a function, this function is used to
        convert a string to another object. In this case, it is recommended
        to give an associated default value as input.
    default : any, optional
        Value to return by default, that is, when the string to be
        converted is flagged as missing. If not given, `StringConverter`
        tries to supply a reasonable default value.
    """

class ConverterError(Exception):
    """
    Exception raised when an error occurs in a converter for string values.
    """
    pass

class ConverterLockError(ConverterError):
    """
    Exception raised when an attempt is made to upgrade a locked converter.
    """
    pass

class ConversionWarning(UserWarning):
    """
    Warning issued when a string converter has a problem.

    Notes
    -----
    In `genfromtxt` a `ConversionWarning` is issued if raising exceptions
    is explicitly suppressed with the "invalid_raise" keyword.
    """
    pass
    missing_values : {None, sequence of str}, optional
        ``None`` 或者字符串序列，表示缺失值。如果是 ``None``，则缺失值用空条目表示。默认为 ``None``.
    locked : bool, optional
        是否锁定 StringConverter，防止自动升级。默认为 False。

    """
    _mapper = [(nx.bool, str2bool, False),
               (nx.int_, int, -1),]

    # On 32-bit systems, we need to make sure that we explicitly include
    # nx.int64 since ns.int_ is nx.int32.
    # 在32位系统上，需要确保显式包含 nx.int64，因为 ns.int_ 是 nx.int32。
    if nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize:
        _mapper.append((nx.int64, int, -1))

    _mapper.extend([(nx.float64, float, nx.nan),
                    (nx.complex128, complex, nx.nan + 0j),
                    (nx.longdouble, nx.longdouble, nx.nan),
                    # If a non-default dtype is passed, fall back to generic
                    # ones (should only be used for the converter)
                    # 如果传递了非默认的 dtype，则回退到通用的类型（仅用于转换器）
                    (nx.integer, int, -1),
                    (nx.floating, float, nx.nan),
                    (nx.complexfloating, complex, nx.nan + 0j),
                    # Last, try with the string types (must be last, because
                    # `_mapper[-1]` is used as default in some cases)
                    # 最后，尝试使用字符串类型（必须放在最后，因为在某些情况下 `_mapper[-1]` 用作默认值）
                    (nx.str_, asunicode, '???'),
                    (nx.bytes_, asbytes, '???'),
                    ])

    @classmethod
    def _getdtype(cls, val):
        """Returns the dtype of the input variable."""
        # 返回输入变量的 dtype。
        return np.array(val).dtype

    @classmethod
    def _getsubdtype(cls, val):
        """Returns the type of the dtype of the input variable."""
        # 返回输入变量的 dtype 的类型。
        return np.array(val).dtype.type

    @classmethod
    def _dtypeortype(cls, dtype):
        """Returns dtype for datetime64 and type of dtype otherwise."""
        # 对于 datetime64 返回 dtype，否则返回 dtype 的类型。

        # This is a bit annoying. We want to return the "general" type in most
        # cases (ie. "string" rather than "S10"), but we want to return the
        # specific type for datetime64 (ie. "datetime64[us]" rather than
        # "datetime64").
        # 这有点烦人。大多数情况下我们想返回“一般”类型（例如“字符串”而不是“S10”），但我们想对 datetime64 返回具体类型（例如“datetime64[us]”而不是“datetime64”）。
        if dtype.type == np.datetime64:
            return dtype
        return dtype.type

    @classmethod
    def upgrade_mapper(cls, func, default=None):
        """
        Upgrade the mapper of a StringConverter by adding a new function and
        its corresponding default.

        The input function (or sequence of functions) and its associated
        default value (if any) is inserted in penultimate position of the
        mapper.  The corresponding type is estimated from the dtype of the
        default value.

        Parameters
        ----------
        func : var
            Function, or sequence of functions

        Examples
        --------
        >>> import dateutil.parser
        >>> import datetime
        >>> dateparser = dateutil.parser.parse
        >>> defaultdate = datetime.date(2000, 1, 1)
        >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
        """
        # Func is a single function
        if hasattr(func, '__call__'):
            # Insert a tuple containing dtype, func, and default into the penultimate position of _mapper
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            # Func is a sequence of functions
            if isinstance(func[0], (tuple, list)):
                # Insert each tuple in func into the penultimate position of _mapper
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            # Determine default value if not provided
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func) - len(default)))
            # Insert dtype, function, and default values into penultimate position of _mapper for each function in func
            for fct, dft in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))

    @classmethod
    def _find_map_entry(cls, dtype):
        # Search for a converter entry matching the specific dtype
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if dtype.type == deftype:
                return i, (deftype, func, default_def)

        # If no exact match, search for an inexact match
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if np.issubdtype(dtype.type, deftype):
                return i, (deftype, func, default_def)

        # Raise an exception if no matching entry is found
        raise LookupError
    # 初始化函数，接受多个参数：dtype_or_func（数据类型或函数，默认为None）、default（默认值，默认为None）、
    # missing_values（缺失值，默认为None）、locked（是否锁定，默认为False）
    def __init__(self, dtype_or_func=None, default=None, missing_values=None,
                 locked=False):
        # 设置一个锁，用于升级
        self._locked = bool(locked)
        
        # 如果没有传入数据类型（dtype_or_func为None），进行最小化初始化
        if dtype_or_func is None:
            # 设置默认处理函数为str2bool
            self.func = str2bool
            # 设置状态为0
            self._status = 0
            # 如果没有指定默认值，则设置为False
            self.default = default or False
            # 将数据类型设置为布尔型
            dtype = np.dtype('bool')
        else:
            # 如果输入是一个np.dtype类型
            try:
                # 不设置处理函数
                self.func = None
                # 将数据类型设置为输入的np.dtype
                dtype = np.dtype(dtype_or_func)
            except TypeError:
                # 如果dtype_or_func必须是一个函数
                if not hasattr(dtype_or_func, '__call__'):
                    errmsg = ("The input argument `dtype` is neither a"
                              " function nor a dtype (got '%s' instead)")
                    # 抛出类型错误异常
                    raise TypeError(errmsg % type(dtype_or_func))
                # 设置处理函数为dtype_or_func
                self.func = dtype_or_func
                # 如果没有默认值，则尝试猜测或者设置为None
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                # 获取数据类型
                dtype = self._getdtype(default)

            # 在映射器中找到最佳匹配
            try:
                # 查找dtype在映射器中的匹配项
                self._status, (_, func, default_def) = self._find_map_entry(dtype)
            except LookupError:
                # 如果找不到匹配项，则使用传入的默认值
                self.default = default
                # 获取最后一个映射项的函数
                _, func, _ = self._mapper[-1]
                # 设置状态为0
                self._status = 0
            else:
                # 如果没有指定默认值，则使用找到的默认值
                if default is None:
                    self.default = default_def
                else:
                    self.default = default

            # 如果输入是一个数据类型，则将函数设置为最后一次看到的函数
            if self.func is None:
                self.func = func

            # 如果函数状态为1（整数），则更改函数为更健壮的选项
            if self.func == self._mapper[1][1]:
                if issubclass(dtype.type, np.uint64):
                    self.func = np.uint64
                elif issubclass(dtype.type, np.int64):
                    self.func = np.int64
                else:
                    self.func = lambda x: int(float(x))
        
        # 存储与缺失值对应的字符串列表
        if missing_values is None:
            # 如果缺失值为空，则设置为{''}
            self.missing_values = {''}
        else:
            # 如果缺失值是字符串，则将其按逗号分隔成列表
            if isinstance(missing_values, str):
                missing_values = missing_values.split(",")
            # 将缺失值列表与{''}合并为集合
            self.missing_values = set(list(missing_values) + [''])

        # 设置调用函数为_strict_call
        self._callingfunction = self._strict_call
        # 确定类型为_dtypeortype返回的类型
        self.type = self._dtypeortype(dtype)
        # 设置检查标志为False
        self._checked = False
        # 存储初始默认值
        self._initial_default = default
    def _loose_call(self, value):
        try:
            # 尝试使用 func 对值进行转换
            return self.func(value)
        except ValueError:
            # 如果值转换失败，则返回默认值
            return self.default

    def _strict_call(self, value):
        try:
            # 检查是否可以使用当前函数转换值
            new_value = self.func(value)

            # 对于整数类型，除了检查 func 是否能转换外，还需确保不会发生溢出错误
            if self.func is int:
                try:
                    np.array(value, dtype=self.type)
                except OverflowError:
                    raise ValueError

            # 如果能成功转换，则返回新值
            return new_value

        except ValueError:
            # 如果值转换失败
            if value.strip() in self.missing_values:
                # 如果值属于缺失值列表，并且状态未锁定，则将_checked设置为False
                if not self._status:
                    self._checked = False
                return self.default
            # 如果转换失败且不属于缺失值列表，则抛出详细错误信息
            raise ValueError("Cannot convert string '%s'" % value)

    def __call__(self, value):
        # 调用_callingfunction方法来处理值
        return self._callingfunction(value)

    def _do_upgrade(self):
        # 如果已锁定转换器，则抛出异常
        if self._locked:
            errmsg = "Converter is locked and cannot be upgraded"
            raise ConverterLockError(errmsg)
        # 获取映射的最大状态值
        _statusmax = len(self._mapper)
        # 如果状态达到最大值，则无法升级
        _status = self._status
        if _status == _statusmax:
            errmsg = "Could not find a valid conversion function"
            raise ConverterError(errmsg)
        elif _status < _statusmax - 1:
            _status += 1
        # 更新转换器的类型、函数和默认值
        self.type, self.func, default = self._mapper[_status]
        self._status = _status
        # 如果设置了初始默认值，则使用该值作为默认值；否则使用映射中的默认值
        if self._initial_default is not None:
            self.default = self._initial_default
        else:
            self.default = default

    def upgrade(self, value):
        """
        Find the best converter for a given string, and return the result.

        The supplied string `value` is converted by testing different
        converters in order. First the `func` method of the
        `StringConverter` instance is tried, if this fails other available
        converters are tried.  The order in which these other converters
        are tried is determined by the `_status` attribute of the instance.

        Parameters
        ----------
        value : str
            The string to convert.

        Returns
        -------
        out : any
            The result of converting `value` with the appropriate converter.

        """
        # 标记检查状态为True
        self._checked = True
        try:
            # 尝试严格调用转换器处理值
            return self._strict_call(value)
        except ValueError:
            # 如果转换失败，则执行升级操作
            self._do_upgrade()
            # 递归调用upgrade方法，尝试再次转换值
            return self.upgrade(value)
    def iterupgrade(self, value):
        # 将对象的_checked属性设置为True，表示已经进行了检查
        self._checked = True
        
        # 如果value不可迭代，将其转换为包含单个元素的元组
        if not hasattr(value, '__iter__'):
            value = (value,)
        
        # 将对象的_strict_call方法缓存到本地变量_strict_call中
        _strict_call = self._strict_call
        
        # 尝试对value中的每个元素调用_strict_call方法
        try:
            for _m in value:
                _strict_call(_m)
        
        # 如果出现ValueError异常，执行对象的_do_upgrade方法，并递归调用iterupgrade方法
        except ValueError:
            self._do_upgrade()
            self.iterupgrade(value)

    def update(self, func, default=None, testing_value=None,
               missing_values='', locked=False):
        """
        直接设置StringConverter的属性。

        Parameters
        ----------
        func : function
            转换函数。
        default : any, optional
            默认返回的值，即当待转换的字符串标记为缺失时使用的值。如果未提供，
            StringConverter会尝试提供一个合理的默认值。
        testing_value : str, optional
            表示转换器的标准输入值的字符串。此字符串用于帮助定义一个合理的默认值。
        missing_values : {sequence of str, None}, optional
            指示缺失值的字符串序列。如果为``None``，则清除现有的`missing_values`。默认为``''``。
        locked : bool, optional
            是否锁定StringConverter以防止自动升级。默认为False。

        Notes
        -----
        `update`接受与`StringConverter`构造函数相同的参数，不同之处在于`func`不接受`dtype`，
        而构造函数中的`dtype_or_func`接受。

        """
        # 将func赋值给对象的func属性
        self.func = func
        
        # 将locked赋值给对象的_locked属性
        self._locked = locked

        # 如果default不为None，将其赋值给对象的default属性，并根据testing_value测试和设置self.type
        if default is not None:
            self.default = default
            tester = func(testing_value or '1')
            self.type = self._dtypeortype(self._getdtype(tester))
        else:
            try:
                # 使用testing_value测试func，将结果设置给tester
                tester = func(testing_value or '1')
            except (TypeError, ValueError):
                tester = None
            # 根据tester获取其dtype，然后根据dtype设置self.type
            self.type = self._dtypeortype(self._getdtype(tester))

        # 将missing_values添加到现有集合或清除它
        if missing_values is None:
            # 如果missing_values为None，清空self.missing_values
            self.missing_values = set()
        else:
            # 如果missing_values不可迭代，将其转换为列表
            if not np.iterable(missing_values):
                missing_values = [missing_values]
            # 检查missing_values中所有元素是否为字符串，如果不是，抛出TypeError异常
            if not all(isinstance(v, str) for v in missing_values):
                raise TypeError("missing_values must be strings or unicode")
            # 将missing_values添加到self.missing_values中
            self.missing_values.update(missing_values)
# 定义便捷函数以创建 `np.dtype` 对象
def easy_dtype(ndtype, names=None, defaultfmt="f%i", **validationargs):
    """
    Convenience function to create a `np.dtype` object.

    The function processes the input `dtype` and matches it with the given
    names.

    Parameters
    ----------
    ndtype : var
        Definition of the dtype. Can be any string or dictionary recognized
        by the `np.dtype` function, or a sequence of types.
    names : str or sequence, optional
        Sequence of strings to use as field names for a structured dtype.
        For convenience, `names` can be a string of a comma-separated list
        of names.
    defaultfmt : str, optional
        Format string used to define missing names, such as ``"f%i"``
        (default) or ``"fields_%02i"``.
    validationargs : optional
        A series of optional arguments used to initialize a
        `NameValidator`.

    Examples
    --------
    >>> np.lib._iotools.easy_dtype(float)
    dtype('float64')
    >>> np.lib._iotools.easy_dtype("i4, f8")
    dtype([('f0', '<i4'), ('f1', '<f8')])
    >>> np.lib._iotools.easy_dtype("i4, f8", defaultfmt="field_%03i")
    dtype([('field_000', '<i4'), ('field_001', '<f8')])

    >>> np.lib._iotools.easy_dtype((int, float, float), names="a,b,c")
    dtype([('a', '<i8'), ('b', '<f8'), ('c', '<f8')])
    >>> np.lib._iotools.easy_dtype(float, names="a,b,c")
    dtype([('a', '<f8'), ('b', '<f8'), ('c', '<f8')])

    """
    # 尝试将 `ndtype` 转换为 `np.dtype` 对象
    try:
        ndtype = np.dtype(ndtype)
    except TypeError:
        # 如果无法转换，初始化一个 `NameValidator` 实例用于验证字段名
        validate = NameValidator(**validationargs)
        nbfields = len(ndtype)
        # 如果 `names` 为空，创建一个空列表以匹配 `ndtype` 的长度
        if names is None:
            names = [''] * len(ndtype)
        # 如果 `names` 是字符串，将其分割为列表
        elif isinstance(names, str):
            names = names.split(",")
        # 使用 `validate` 对 `names` 进行验证，确保其数量与 `ndtype` 中字段数量一致，并使用 `defaultfmt` 表示缺失字段的格式
        names = validate(names, nbfields=nbfields, defaultfmt=defaultfmt)
        # 根据验证后的 `names` 和 `ndtype` 创建一个结构化 `np.dtype` 对象
        ndtype = np.dtype(dict(formats=ndtype, names=names))
    else:
        # 如果没有隐式名称的情况下，处理显式名称
        if names is not None:
            # 使用给定的验证参数创建名称验证器
            validate = NameValidator(**validationargs)
            # 如果 `names` 是字符串，则将其拆分为列表
            if isinstance(names, str):
                names = names.split(",")
            # 如果数据类型没有字段名称，则重复数据类型以匹配名称的数量
            if ndtype.names is None:
                formats = tuple([ndtype.type] * len(names))
                # 使用验证器验证并设置名称，默认格式为 `defaultfmt`
                names = validate(names, defaultfmt=defaultfmt)
                # 创建结构化数据类型，名称与格式一一对应
                ndtype = np.dtype(list(zip(names, formats)))
            # 如果数据类型已经有字段名称，则只需根据需要验证名称
            else:
                # 使用验证器验证名称，并根据字段数设置默认格式
                ndtype.names = validate(names, nbfields=len(ndtype.names),
                                        defaultfmt=defaultfmt)
        # 如果没有隐式名称，且数据类型也没有字段名称
        elif ndtype.names is not None:
            # 使用给定的验证参数创建名称验证器
            validate = NameValidator(**validationargs)
            # 默认初始名称为数字命名的元组
            numbered_names = tuple("f%i" % i for i in range(len(ndtype.names)))
            # 如果初始名称是数字命名且默认格式不是 "f%i"，则改变格式
            if ((ndtype.names == numbered_names) and (defaultfmt != "f%i")):
                # 使用验证器验证并设置空名称列表，格式为 `defaultfmt`
                ndtype.names = validate([''] * len(ndtype.names),
                                        defaultfmt=defaultfmt)
            # 如果初始名称是显式命名，则只需验证名称
            else:
                # 使用验证器验证并设置名称，格式为 `defaultfmt`
                ndtype.names = validate(ndtype.names, defaultfmt=defaultfmt)
    # 返回处理后的数据类型
    return ndtype
```