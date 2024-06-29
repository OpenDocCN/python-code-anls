# `.\numpy\numpy\_core\records.py`

```
"""
This module contains a set of functions for record arrays.
"""
# 导入必要的模块和包
import os
import warnings
from collections import Counter
from contextlib import nullcontext

# 从模块_utils中导入set_module函数
from .._utils import set_module
# 从当前目录的子模块中导入numeric作为别名sb，导入numerictypes作为别名nt
from . import numeric as sb
from . import numerictypes as nt
# 从当前目录的子模块中导入arrayprint模块中的_get_legacy_print_mode函数
from .arrayprint import _get_legacy_print_mode

# 定义可导出的函数和类名列表
__all__ = [
    'record', 'recarray', 'format_parser', 'fromarrays', 'fromrecords',
    'fromstring', 'fromfile', 'array', 'find_duplicate',
]

# 将sb.ndarray赋值给ndarray，简化代码中的引用
ndarray = sb.ndarray

# 字节顺序转换字典，用于将字符表示的字节顺序映射为标准符号
_byteorderconv = {'b': '>',
                  'l': '<',
                  'n': '=',
                  'B': '>',
                  'L': '<',
                  'N': '=',
                  'S': 's',
                  's': 's',
                  '>': '>',
                  '<': '<',
                  '=': '=',
                  '|': '|',
                  'I': '|',
                  'i': '|'}

# 数字格式的正则表达式字典，使用numerictypes模块中的sctypeDict函数获取
# 允许多维度规范与元组语法的结合，例如 '(2,3)f4' 和 ' (  2 ,  3  )  f4  ' 都是合法的
numfmt = nt.sctypeDict


# 使用装饰器set_module将函数find_duplicate置于numpy.rec模块下
@set_module('numpy.rec')
def find_duplicate(list):
    """Find duplication in a list, return a list of duplicated elements"""
    return [
        item
        for item, counts in Counter(list).items()
        if counts > 1
    ]


# format_parser类的定义，用于将格式、名称和标题描述转换为dtype
@set_module('numpy.rec')
class format_parser:
    """
    Class to convert formats, names, titles description to a dtype.

    After constructing the format_parser object, the dtype attribute is
    the converted data-type:
    ``dtype = format_parser(formats, names, titles).dtype``

    Attributes
    ----------
    dtype : dtype
        The converted data-type.

    Parameters
    ----------
    formats : str or list of str
        The format description, either specified as a string with
        comma-separated format descriptions in the form ``'f8, i4, S5'``, or
        a list of format description strings  in the form
        ``['f8', 'i4', 'S5']``.
    names : str or list/tuple of str
        The field names, either specified as a comma-separated string in the
        form ``'col1, col2, col3'``, or as a list or tuple of strings in the
        form ``['col1', 'col2', 'col3']``.
        An empty list can be used, in that case default field names
        ('f0', 'f1', ...) are used.
    titles : sequence
        Sequence of title strings. An empty list can be used to leave titles
        out.
    aligned : bool, optional
        If True, align the fields by padding as the C-compiler would.
        Default is False.
    byteorder : str, optional
        If specified, all the fields will be changed to the
        provided byte-order.  Otherwise, the default byte-order is
        used. For all available string specifiers, see `dtype.newbyteorder`.

    See Also
    --------
    numpy.dtype, numpy.typename

    Examples
    --------
    >>> np.rec.format_parser(['<f8', '<i4'], ['col1', 'col2'],
    """
    """
    `names` and/or `titles` can be empty lists. If `titles` is an empty list,
    titles will simply not appear. If `names` is empty, default field names
    will be used.

    >>> np.rec.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
    ...                      []).dtype
    dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '<S5')])
    >>> np.rec.format_parser(['<f8', '<i4', '<a5'], [], []).dtype
    dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', 'S5')])

    """

    def __init__(self, formats, names, titles, aligned=False, byteorder=None):
        """ Initialize the record parser object with formats, field names, titles, alignment, and byte order """

        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdtype(byteorder)

    def _parseFormats(self, formats, aligned=False):
        """ Parse the field formats based on input formats and alignment """

        if formats is None:
            raise ValueError("Need formats argument")
        if isinstance(formats, list):
            # Create structured dtype from list of formats
            dtype = sb.dtype(
                [
                    ('f{}'.format(i), format_) 
                    for i, format_ in enumerate(formats)
                ],
                aligned,
            )
        else:
            # Create structured dtype from single format string
            dtype = sb.dtype(formats, aligned)
        
        fields = dtype.fields
        if fields is None:
            # If fields are not parsed correctly, create a default dtype
            dtype = sb.dtype([('f1', dtype)], aligned)
            fields = dtype.fields
        
        keys = dtype.names
        # Store the formats of each field
        self._f_formats = [fields[key][0] for key in keys]
        # Store the offsets of each field
        self._offsets = [fields[key][1] for key in keys]
        # Store the number of fields
        self._nfields = len(keys)

    def _setfieldnames(self, names, titles):
        """ Convert input field names into a list and assign to the _names attribute """

        if names:
            if type(names) in [list, tuple]:
                pass
            elif isinstance(names, str):
                # Split comma-separated string into list of names
                names = names.split(',')
            else:
                raise NameError("illegal input names %s" % repr(names))

            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []

        # Assign default names if not enough names are specified
        self._names += ['f%d' % i for i in range(len(self._names),
                                                 self._nfields)]
        
        # Check for duplicate names
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError("Duplicate field names: %s" % _dup)

        if titles:
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []

        # Fill titles with None for fields without specific titles
        if self._nfields > len(titles):
            self._titles += [None] * (self._nfields - len(titles))
    # 定义一个私有方法 _createdtype，用于创建一个新的数据类型 dtype
    def _createdtype(self, byteorder):
        # 根据给定的属性创建一个结构化数据类型 dtype
        dtype = sb.dtype({
            'names': self._names,       # 使用 self._names 定义字段名
            'formats': self._f_formats, # 使用 self._f_formats 定义字段格式
            'offsets': self._offsets,   # 使用 self._offsets 定义字段偏移量
            'titles': self._titles,     # 使用 self._titles 定义字段标题
        })
        # 如果传入了 byteorder 参数，则将其转换为对应的字节顺序，并更新 dtype
        if byteorder is not None:
            byteorder = _byteorderconv[byteorder[0]]  # 获取对应的字节顺序
            dtype = dtype.newbyteorder(byteorder)     # 调整 dtype 的字节顺序为新的顺序

        # 将创建的数据类型 dtype 赋值给当前对象的 self.dtype 属性
        self.dtype = dtype
class record(nt.void):
    """A data-type scalar that allows field access as attribute lookup.
    """

    # manually set name and module so that this class's type shows up
    # as numpy.record when printed
    __name__ = 'record'
    __module__ = 'numpy'

    def __repr__(self):
        # 如果打印模式小于或等于 113，返回对象的字符串表示形式
        if _get_legacy_print_mode() <= 113:
            return self.__str__()
        # 否则调用父类的 __repr__ 方法
        return super().__repr__()

    def __str__(self):
        # 如果打印模式小于或等于 113，返回对象单个元素的字符串表示形式
        if _get_legacy_print_mode() <= 113:
            return str(self.item())
        # 否则调用父类的 __str__ 方法
        return super().__str__()

    def __getattribute__(self, attr):
        # 如果属性名为 'setfield', 'getfield', 'dtype' 中的一个，直接从父类获取该属性
        if attr in ('setfield', 'getfield', 'dtype'):
            return nt.void.__getattribute__(self, attr)
        try:
            # 否则尝试从父类获取属性
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        # 获取字段字典
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        # 尝试获取属性在字段字典中的信息
        res = fielddict.get(attr, None)
        if res:
            # 如果存在字段信息，获取字段值
            obj = self.getfield(*res[:2])
            # 如果字段值有子字段，则返回一个 record 类型的视图，否则返回字段值
            try:
                dt = obj.dtype
            except AttributeError:
                # 如果字段是 Object 类型，直接返回字段值
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        else:
            # 如果字段信息不存在，抛出 AttributeError 异常
            raise AttributeError("'record' object has no "
                    "attribute '%s'" % attr)

    def __setattr__(self, attr, val):
        # 如果属性名为 'setfield', 'getfield', 'dtype' 中的一个，抛出 AttributeError 异常
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError("Cannot set '%s' attribute" % attr)
        # 获取字段字典
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        # 尝试获取属性在字段字典中的信息
        res = fielddict.get(attr, None)
        if res:
            # 如果存在字段信息，设置字段值
            return self.setfield(val, *res[:2])
        else:
            # 如果字段信息不存在且属性存在，则设置属性值，否则抛出 AttributeError 异常
            if getattr(self, attr, None):
                return nt.void.__setattr__(self, attr, val)
            else:
                raise AttributeError("'record' object has no "
                        "attribute '%s'" % attr)

    def __getitem__(self, indx):
        # 获取指定索引位置的元素
        obj = nt.void.__getitem__(self, indx)

        # 复制 record.__getattribute__ 的行为
        if isinstance(obj, nt.void) and obj.dtype.names is not None:
            # 如果元素是 record 类型且有子字段，则返回一个 record 类型的视图
            return obj.view((self.__class__, obj.dtype))
        else:
            # 否则返回单个元素
            return obj

    def pprint(self):
        """Pretty-print all fields."""
        # 漂亮打印所有字段
        names = self.dtype.names
        # 计算字段名的最大长度
        maxlen = max(len(name) for name in names)
        fmt = '%% %ds: %%s' % maxlen
        # 格式化每个字段名及其值
        rows = [fmt % (name, getattr(self, name)) for name in names]
        # 将格式化后的字符串连接成一个字符串并返回
        return "\n".join(rows)
# 设置模块名称为 "numpy.rec"
@set_module("numpy.rec")
# 定义一个类 recarray，继承自 ndarray
class recarray(ndarray):
    """Construct an ndarray that allows field access using attributes.

    Arrays may have a data-types containing fields, analogous
    to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
    where each entry in the array is a pair of ``(int, float)``.  Normally,
    these attributes are accessed using dictionary lookups such as ``arr['x']``
    and ``arr['y']``.  Record arrays allow the fields to be accessed as members
    of the array, using ``arr.x`` and ``arr.y``.

    Parameters
    ----------
    shape : tuple
        Shape of output array.
    dtype : data-type, optional
        The desired data-type.  By default, the data-type is determined
        from `formats`, `names`, `titles`, `aligned` and `byteorder`.
    formats : list of data-types, optional
        A list containing the data-types for the different columns, e.g.
        ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
        convention of using types directly, i.e. ``(int, float, int)``.
        Note that `formats` must be a list, not a tuple.
        Given that `formats` is somewhat limited, we recommend specifying
        `dtype` instead.
    names : tuple of str, optional
        The name of each column, e.g. ``('x', 'y', 'z')``.
    buf : buffer, optional
        By default, a new array is created of the given shape and data-type.
        If `buf` is specified and is an object exposing the buffer interface,
        the array will use the memory from the existing buffer.  In this case,
        the `offset` and `strides` keywords are available.

    Other Parameters
    ----------------
    titles : tuple of str, optional
        Aliases for column names.  For example, if `names` were
        ``('x', 'y', 'z')`` and `titles` is
        ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
        ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
    byteorder : {'<', '>', '='}, optional
        Byte-order for all fields.
    aligned : bool, optional
        Align the fields in memory as the C-compiler would.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    offset : int, optional
        Start reading buffer (`buf`) from this offset onwards.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    rec : recarray
        Empty array of the given shape and type.

    See Also
    --------
    numpy.rec.fromrecords : Construct a record array from data.
    numpy.record : fundamental data-type for `recarray`.
    numpy.rec.format_parser : determine data-type from formats, names, titles.

    Notes
    -----
    This constructor can be compared to ``empty``: it creates a new record
    """
    # 构造函数初始化
    def __init__(self, shape, dtype=None, formats=None, names=None,
                 buf=None, **kwargs):
        # 调用父类的构造函数，初始化数组
        super(recarray, self).__new__(recarray, shape, dtype, buffer=buf)
        # 如果指定了字段格式（formats）和字段名称（names），则设置相关属性
        if formats is not None and names is not None:
            # 确保 formats 是列表形式
            if not isinstance(formats, list):
                raise TypeError("formats must be a list")
            # 确保 names 是元组形式
            if not isinstance(names, tuple):
                raise TypeError("names must be a tuple")
            # 设置字段格式和名称
            self._fieldnames = names
            self._formats = formats

            # 确保字段格式和字段名称长度一致
            if len(self._fieldnames) != len(self._formats):
                raise ValueError("Length of formats and names must be equal")
        # 如果没有指定 dtype，则根据其他参数推断 dtype
        elif dtype is None:
            self._fieldnames = []
            self._formats = []
        # 如果没有指定 names，则抛出错误
        else:
            raise ValueError("Both formats and names must be specified if dtype is provided")

    def __setattr__(self, attr, value):
        # 如果属性名存在于字段名称中，设置属性值
        if attr in self._fieldnames:
            idx = self._fieldnames.index(attr)
            self[idx] = value
        else:
            # 否则，调用父类的属性设置方法
            super(recarray, self).__setattr__(attr, value)

    def __getattr__(self, attr):
        # 如果属性名存在于字段名称中，返回相应的值
        if attr in self._fieldnames:
            idx = self._fieldnames.index(attr)
            return self[idx]
        else:
            # 否则，调用父类的属性获取方法
            return super(recarray, self).__getattribute__(attr)

    def __getitem__(self, key):
        # 支持字段名和索引作为键值
        if isinstance(key, str):
            # 如果 key 是字符串，则返回对应字段的值
            if key in self._fieldnames:
                idx = self._fieldnames.index(key)
                return self[idx]
            else:
                raise KeyError(f"Field '{key}' not found in recarray")
        else:
            # 否则，调用父类的获取方法
            return super(recarray, self).__getitem__(key)

    def __setitem__(self, key, value):
        # 支持字段名和索引作为键值
        if isinstance(key, str):
            # 如果 key 是字符串，则设置对应字段的值
            if key in self._fieldnames:
                idx = self._fieldnames.index(key)
                self[idx] = value
            else:
                raise KeyError(f"Field '{key}' not found in recarray")
        else:
            # 否则，调用父类的设置方法
            super(recarray, self).__setitem__(key, value)

    def __reduce__(self):
        # 返回可序列化对象的元组表示
        pickled_state = super(recarray, self).__reduce__()
        new_state = pickled_state[2] + (self._fieldnames, self._formats)
        return pickled_state[0], pickled_state[1], new_state

    def __repr__(self):
        # 返回对象的字符串表示形式
        return "recarray(shape={}, dtype={}, formats={}, names={})".format(
            self.shape, self.dtype, self._formats, self._fieldnames)

    def __str__(self):
        # 返回对象的可打印字符串表示形式
        return self.__repr__()

# 设置模块名称为 "numpy.rec"
@set_module("numpy.rec")
# 定义一个类 recarray，继承自 ndarray
class recarray(ndarray):
    """Construct an ndarray that allows field access using attributes.

    Arrays may have a data-types containing fields, analogous
    to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
    where each entry in the array is a pair of ``(int, float)``.  Normally,
    these attributes are accessed using dictionary lookups such as ``arr['x']``
    and ``arr['y']``.  Record arrays allow the fields to be accessed as members
    of the array, using ``arr.x`` and ``arr.y``.

    Parameters
    ----------
    shape : tuple
        Shape of output array.
    dtype : data-type, optional
        The desired data-type.  By default, the data-type is determined
        from `formats`, `names`, `titles`, `aligned` and `byteorder`.
    formats : list of data-types, optional
        A list containing the data-types for the different columns, e.g.
        ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
        convention of using types directly, i.e. ``(int, float, int)``.
        Note that `formats` must be a list, not a tuple.
        Given that `formats` is somewhat limited, we recommend specifying
        `dtype` instead.
    names : tuple of str, optional
        The name of each column, e.g. ``('x', 'y', 'z')``.
    buf : buffer, optional
        By default, a new array is created of the given shape and data-type.
        If `buf` is specified and is an object exposing the buffer interface,
        the array will use the memory from the existing buffer.  In this case,
        the `offset` and `strides` keywords are available.

    Other Parameters
    ----------------
    titles : tuple of str, optional
        Aliases for column names.  For example, if `names` were
        ``('x', 'y', 'z')`` and `titles` is
        ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
        ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
    byteorder : {'<', '>', '='}, optional
        Byte-order for all fields.
    aligned : bool, optional
        Align the fields in memory as the C-compiler would.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    offset : int, optional
        Start reading buffer (`buf`) from this offset onwards.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    rec : recarray
        Empty array of the given shape and type.

    See Also
    --------
    numpy.rec.fromrecords : Construct a record array from data.
    numpy.record : fundamental data-type for `recarray`.
    numpy.rec.format_parser : determine data-type from formats, names, titles.

    Notes
    -----
    This constructor can be compared to ``empty``: it creates a new record
    """
    # 构造函数初始化
    def __init__(self, shape, dtype=None, formats=None, names=None,
                 buf=None, **kwargs):
        # 调用父类的构造函数，初始化数组
        super(recarray, self).__new__(recarray, shape, dtype, buffer=buf)
        # 如果指定了字段格式（formats）和字段名称（names），则设置相关属性
        if formats is not None and names is not None:
            # 确保 formats 是列表形式
            if not isinstance(formats, list):
                raise TypeError("formats must be a list")
            # 确保 names 是元组形式
            if not isinstance(names, tuple):
                raise TypeError("names must be a tuple")
            # 设置字段格式和名称
            self._fieldnames = names
            self._formats = formats
            # 确保字段格式和字段名称长度一致
            if len(self._fieldnames) != len(self._formats):
                raise ValueError("Length of formats and names must be equal")
        # 如果没有指定 dtype，则根据其他参数推断 dtype
        elif dtype is None:
            self._fieldnames = []
            self._formats = []
        # 如果没有指定 names，则抛出错误
        else:
            raise ValueError("Both formats and names must be specified if dtype is provided")

    def __setattr__(self, attr, value):
        # 如果属性名存在于字段名称中，设置属性值
        if attr in self._fieldnames:
            idx = self._fieldnames.index(attr)
            self[idx] = value
        else:
            # 否则，调用父类的属性设置方法
            super(recarray, self).__setattr__(attr, value)

    def __getattr__(self, attr):
        # 如果属性名存在于字段名称中，返回相应的值
        if attr in self._fieldnames:
            idx = self._fieldnames.index(attr)
            return self[idx]
        else:
            # 否则，调用父类的属性获取方法
            return super(recarray, self).__getattribute__(attr)

    def __getitem__(self, key):
        # 支持字段名和索引作为键值
        if isinstance(key, str):
            # 如果 key 是字符串，则返回对应字段的值
            if key in self._fieldnames:
                idx = self._fieldnames.index(key)
                return self[idx]
            else:
                raise KeyError(f"Field '{key}' not found in recarray")
        else:
            # 否则，调用父类的获取方法
            return super(recarray, self).__getitem__(key)

    def __setitem__(self, key, value):
        # 支持字段名和索引作为键值
        if isinstance(key, str):
            # 如果 key 是字符串，则设置对应字段的值
            if key
    """
    Create a new subclass of ndarray for structured arrays, optionally filled
    with data from a buffer.

    Parameters
    ----------
    subtype : type
        The subclass type.
    shape : tuple
        Shape of the new array.
    dtype : dtype, optional
        Data type descriptor for the array. If not provided, it is inferred
        from other parameters.
    buf : buffer-like, optional
        Object exposing buffer interface for data storage.
    offset : int, optional
        Offset in bytes from the start of the buffer to the beginning of the
        array data.
    strides : tuple, optional
        Strides of the array data in memory.
    formats : sequence, optional
        Format descriptors for structured data.
    names : sequence, optional
        Names for the fields of structured data.
    titles : sequence, optional
        Titles for the fields of structured data.
    byteorder : {'=', '|', '>', '<', 'little', 'big'}, optional
        Byte order of the data. Default is native byte order.
    aligned : bool, optional
        Whether the data should be aligned.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order.

    Returns
    -------
    self : ndarray
        A new instance of the structured array subclass.

    Notes
    -----
    This constructor initializes a structured array subclass similar to
    ndarray, but with additional support for structured data types. It can
    create an empty array structure or initialize it from provided data.

    See Also
    --------
    np.recarray : View a standard ndarray as a record array.
    """

    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False, order='C'):
        """
        Create a new instance of the structured array subclass.

        Parameters
        ----------
        subtype : type
            The subclass type.
        shape : tuple
            Shape of the new array.
        dtype : dtype, optional
            Data type descriptor for the array. If not provided, it is inferred
            from other parameters.
        buf : buffer-like, optional
            Object exposing buffer interface for data storage.
        offset : int, optional
            Offset in bytes from the start of the buffer to the beginning of the
            array data.
        strides : tuple, optional
            Strides of the array data in memory.
        formats : sequence, optional
            Format descriptors for structured data.
        names : sequence, optional
            Names for the fields of structured data.
        titles : sequence, optional
            Titles for the fields of structured data.
        byteorder : {'=', '|', '>', '<', 'little', 'big'}, optional
            Byte order of the data. Default is native byte order.
        aligned : bool, optional
            Whether the data should be aligned.
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in row-major (C-style) or
            column-major (Fortran-style) order.

        Returns
        -------
        self : ndarray
            A new instance of the structured array subclass.

        Notes
        -----
        This method is responsible for creating a new structured array
        instance based on the provided parameters. It initializes the array
        either from a specified buffer or as an empty structure if no buffer
        is provided.

        If `dtype` is specified, it converts it to a data type descriptor.
        If `buf` is provided, it initializes the array using the buffer's
        data.

        """
        if dtype is not None:
            descr = sb.dtype(dtype)  # Convert dtype to a structured dtype
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder
            ).dtype  # Parse formats, names, titles to infer dtype

        if buf is None:
            self = ndarray.__new__(
                subtype, shape, (record, descr), order=order
            )  # Create ndarray instance without buffer
        else:
            self = ndarray.__new__(
                subtype, shape, (record, descr), buffer=buf,
                offset=offset, strides=strides, order=order
            )  # Create ndarray instance with buffer

        return self

    def __array_finalize__(self, obj):
        """
        Finalizes the creation of a structured array instance.

        Parameters
        ----------
        self : ndarray
            The newly created structured array instance.
        obj : ndarray or None
            An object from which the structured array instance was derived.

        Notes
        -----
        This method is called after the instance has been created and is
        responsible for finalizing its initialization. It checks if the
        dtype of the array is a record dtype and ensures that if it has
        names defined, it properly sets the dtype.

        If `self.dtype` is not a record dtype but has names, it invokes
        `__setattr__` to convert it to a record dtype.
        """
        if self.dtype.type is not record and self.dtype.names is not None:
            # Convert to a record dtype if dtype is not np.record
            self.dtype = self.dtype
    # 当尝试获取对象的属性时调用的特殊方法，用于获取对象的属性值
    def __getattribute__(self, attr):
        # 检查 ndarray 是否具有这个属性，并返回属性值（注意，如果字段与 ndarray 属性同名，将无法通过属性访问）
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # 如果 attr 是字段名
            pass

        # 查找具有此名称的字段
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            # 获取字段的偏移量和数据类型
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            # 抛出 AttributeError 如果没有找到对应的字段
            raise AttributeError("recarray has no attribute %s" % attr) from e
        # 获取字段的值
        obj = self.getfield(*res)

        # 在此时，obj 总是一个 recarray，因为（参见 PyArray_GetField）obj 的类型是继承的。
        # 如果 obj.dtype 是非结构化的，将其转换为 ndarray。
        # 如果 obj 是结构化的并且是 void 类型，则将其转换为相同的 dtype.type（例如保留 numpy.record 类型），
        # 因为嵌套的结构化字段不会继承类型。但是对于非 void 结构化类型，不执行此操作。
        if obj.dtype.names is not None:
            if issubclass(obj.dtype.type, nt.void):
                return obj.view(dtype=(self.dtype.type, obj.dtype))
            return obj
        else:
            return obj.view(ndarray)

    # 当尝试设置对象的属性时调用的特殊方法，用于设置对象的属性值
    def __setattr__(self, attr, val):

        # 自动将（void）结构化类型转换为记录类型
        # （但不包括非 void 结构、子数组或非结构化的 void）
        if (
            attr == 'dtype' and 
            issubclass(val.type, nt.void) and 
            val.names is not None
        ):
            val = sb.dtype((record, val))

        # 检查属性是否是新属性
        newattr = attr not in self.__dict__
        try:
            # 尝试设置属性值
            ret = object.__setattr__(self, attr, val)
        except Exception:
            # 获取字段字典或空字典
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                # 如果属性不在字段字典中，则抛出异常
                raise
        else:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                # 如果属性不在字段字典中，返回设置结果
                return ret
            if newattr:
                # 如果是新属性或此 setattr 在内部属性上起作用
                try:
                    # 尝试删除新属性
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        try:
            # 获取字段的偏移量和数据类型
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            # 抛出 AttributeError 如果没有找到对应的字段
            raise AttributeError(
                "record array has no attribute %s" % attr
            ) from e
        # 设置字段的值
        return self.setfield(val, *res)
    #`
    # 覆盖了父类的 __getitem__ 方法，根据索引获取元素
    def __getitem__(self, indx):
        # 调用父类的 __getitem__ 方法获取元素
        obj = super().__getitem__(indx)

        # 模仿 getattr 的行为，但这里可能返回单个元素
        if isinstance(obj, ndarray):  # 如果返回的是 ndarray 类型的对象
            if obj.dtype.names is not None:  # 如果 ndarray 具有字段名
                # 将 obj 视图转换为当前类的类型
                obj = obj.view(type(self))
                # 如果 obj 的 dtype 是 numpy.void 的子类
                if issubclass(obj.dtype.type, np.void):
                    # 返回一个新的视图，其 dtype 为 (self.dtype.type, obj.dtype)
                    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj  # 返回转换后的 obj
            else:
                return obj.view(type=ndarray)  # 返回 obj 的 ndarray 视图
        else:
            # 返回单个元素
            return obj

    # 返回对象的字符串表示形式
    def __repr__(self):

        repr_dtype = self.dtype
        if (
            self.dtype.type is np.record or 
            not issubclass(self.dtype.type, np.void)
        ):
            # 如果是完整的记录数组（具有 numpy.record dtype），
            # 或者如果它具有标量（非 void）dtype 且没有记录，
            # 使用 rec.array 函数表示。由于 rec.array 会将 dtype 转换为 numpy.record，
            # 所以在打印前需要转换回非记录形式。
            if repr_dtype.type is np.record:
                repr_dtype = np.dtype((np.void, repr_dtype))
            prefix = "rec.array("
            fmt = 'rec.array(%s,%sdtype=%s)'
        else:
            # 否则使用 np.array 加一个视图表示
            # 这种情况只会在用户处理 dtype 时可能出现奇怪的情况。
            prefix = "array("
            fmt = 'array(%s,%sdtype=%s).view(numpy.recarray)'

        # 获取数据/形状的字符串表示。逻辑取自 numeric.array_repr
        if self.size > 0 or self.shape == (0,):
            lst = np.array2string(
                self, separator=', ', prefix=prefix, suffix=',')
        else:
            # 显示长度为零的形状，除非它是 (0,)
            lst = "[], shape=%s" % (repr(self.shape),)

        lf = '\n'+' '*len(prefix)
        if _get_legacy_print_mode() <= 113:
            lf = ' ' + lf  # 尾随空格
        return fmt % (lst, lf, repr_dtype)

    # 返回字段的值或设置字段的值
    def field(self, attr, val=None):
        if isinstance(attr, int):
            # 如果 attr 是整数，获取 dtype 的字段名
            names = ndarray.__getattribute__(self, 'dtype').names
            attr = names[attr]

        # 获取 dtype 的字段字典
        fielddict = ndarray.__getattribute__(self, 'dtype').fields

        # 获取字段的偏移量和形状
        res = fielddict[attr][:2]

        if val is None:
            # 如果 val 为 None，获取字段的值
            obj = self.getfield(*res)
            if obj.dtype.names is not None:
                return obj  # 如果 obj 具有字段名，返回 obj
            return obj.view(ndarray)  # 否则返回 obj 的 ndarray 视图
        else:
            # 否则设置字段的值并返回结果
            return self.setfield(val, *res)
def _deprecate_shape_0_as_None(shape):
    # 如果 shape 为 0，则发出未来警告，并返回 None
    if shape == 0:
        warnings.warn(
            "Passing `shape=0` to have the shape be inferred is deprecated, "
            "and in future will be equivalent to `shape=(0,)`. To infer "
            "the shape and suppress this warning, pass `shape=None` instead.",
            FutureWarning, stacklevel=3)
        return None
    else:
        # 否则直接返回 shape
        return shape


@set_module("numpy.rec")
def fromarrays(arrayList, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    """Create a record array from a (flat) list of arrays

    Parameters
    ----------
    arrayList : list or tuple
        List of array-like objects (such as lists, tuples,
        and ndarrays).
    dtype : data-type, optional
        valid dtype for all arrays
    shape : int or tuple of ints, optional
        Shape of the resulting array. If not provided, inferred from
        ``arrayList[0]``.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.rec.format_parser` to construct a dtype. See that function for
        detailed documentation.

    Returns
    -------
    np.recarray
        Record array consisting of given arrayList columns.

    Examples
    --------
    >>> x1=np.array([1,2,3,4])
    >>> x2=np.array(['a','dd','xyz','12'])
    >>> x3=np.array([1.1,2,3,4])
    >>> r = np.rec.fromarrays([x1,x2,x3],names='a,b,c')
    >>> print(r[1])
    (2, 'dd', 2.0) # may vary
    >>> x1[1]=34
    >>> r.a
    array([1, 2, 3, 4])

    >>> x1 = np.array([1, 2, 3, 4])
    >>> x2 = np.array(['a', 'dd', 'xyz', '12'])
    >>> x3 = np.array([1.1, 2, 3,4])
    >>> r = np.rec.fromarrays(
    ...     [x1, x2, x3],
    ...     dtype=np.dtype([('a', np.int32), ('b', 'S3'), ('c', np.float32)]))
    >>> r
    rec.array([(1, b'a', 1.1), (2, b'dd', 2. ), (3, b'xyz', 3. ),
               (4, b'12', 4. )],
              dtype=[('a', '<i4'), ('b', 'S3'), ('c', '<f4')])
    """

    arrayList = [sb.asarray(x) for x in arrayList]

    # NumPy 1.19.0, 2020-01-01
    # 调用 _deprecate_shape_0_as_None 函数，处理 shape 为 0 的情况
    shape = _deprecate_shape_0_as_None(shape)

    if shape is None:
        # 如果 shape 为 None，则从第一个数组推断出 shape
        shape = arrayList[0].shape
    elif isinstance(shape, int):
        # 如果 shape 是 int 类型，则转换为元组
        shape = (shape,)

    if formats is None and dtype is None:
        # 如果未提供 dtype，则通过遍历 arrayList 列表中的对象来确定 formats
        formats = [obj.dtype for obj in arrayList]

    if dtype is not None:
        # 如果提供了 dtype，则将其转换为 dtype 对象
        descr = sb.dtype(dtype)
    else:
        # 否则，通过 format_parser 函数获取 dtype 描述符
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    _names = descr.names

    # 根据 dtype 的长度确定形状
    if len(descr) != len(arrayList):
        raise ValueError("mismatch between the number of fields "
                "and the number of arrays")

    d0 = descr[0].shape
    nn = len(d0)
    if nn > 0:
        shape = shape[:-nn]

    _array = recarray(shape, descr)
    # 遍历数组列表并填充记录数组（创建副本）
    for k, obj in enumerate(arrayList):
        # 获取描述符对象的维度
        nn = descr[k].ndim
        # 获取对象的形状，去掉描述符对象的维度后剩余的部分
        testshape = obj.shape[:obj.ndim - nn]
        # 获取当前对象的名称
        name = _names[k]
        # 检查当前对象的形状是否与整体形状相匹配
        if testshape != shape:
            # 如果形状不匹配，则抛出值错误异常
            raise ValueError(f'array-shape mismatch in array {k} ("{name}")')

        # 将当前对象存入记录数组中的对应名称位置
        _array[name] = obj

    # 返回填充后的记录数组
    return _array
# 设置函数的模块为 "numpy.rec"
@set_module("numpy.rec")
def fromrecords(recList, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None):
    """从文本形式的记录列表创建一个 recarray。

    Parameters
    ----------
    recList : sequence
        包含记录的列表，可以是异构的数据 - 将会提升到最高的数据类型。
    dtype : data-type, optional
        所有数组的有效数据类型。
    shape : int or tuple of ints, optional
        每个数组的形状。
    formats, names, titles, aligned, byteorder :
        如果 `dtype` 是 ``None``，这些参数将传递给 `numpy.format_parser` 来构建数据类型。
        详细文档请参考该函数。

        如果 `formats` 和 `dtype` 都是 `None`，则会自动检测格式。使用元组列表而不是列表列表可以提高处理速度。

    Returns
    -------
    np.recarray
        包含给定 recList 行的记录数组。

    Examples
    --------
    >>> r=np.rec.fromrecords([(456,'dbe',1.2),(2,'de',1.3)],
    ... names='col1,col2,col3')
    >>> print(r[0])
    (456, 'dbe', 1.2)
    >>> r.col1
    array([456,   2])
    >>> r.col2
    array(['dbe', 'de'], dtype='<U3')
    >>> import pickle
    >>> pickle.loads(pickle.dumps(r))
    rec.array([(456, 'dbe', 1.2), (  2, 'de', 1.3)],
              dtype=[('col1', '<i8'), ('col2', '<U3'), ('col3', '<f8')])
    """

    if formats is None and dtype is None:  # 如果没有指定格式和数据类型，则执行以下代码（较慢的方式）
        # 将 recList 转换为对象数组
        obj = sb.array(recList, dtype=object)
        # 生成每列的数组列表
        arrlist = [
            sb.array(obj[..., i].tolist()) for i in range(obj.shape[-1])
        ]
        # 调用 fromarrays 函数，返回结果
        return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                          titles=titles, aligned=aligned, byteorder=byteorder)

    if dtype is not None:
        # 如果指定了数据类型，则创建描述符
        descr = sb.dtype((record, dtype))
    else:
        # 否则，通过 format_parser 函数获取描述符
        descr = format_parser(
            formats, names, titles, aligned, byteorder
        ).dtype

    try:
        # 尝试使用描述符创建数组
        retval = sb.array(recList, dtype=descr)
    except (TypeError, ValueError):
        # 处理可能的类型错误或数值错误
        shape = _deprecate_shape_0_as_None(shape)
        if shape is None:
            shape = len(recList)
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) > 1:
            raise ValueError("Can only deal with 1-d array.")
        # 创建 recarray 对象并填充数据
        _array = recarray(shape, descr)
        for k in range(_array.size):
            _array[k] = tuple(recList[k])
        # 提出警告，因为传入的可能是列表列表而不是列表元组
        warnings.warn(
            "fromrecords expected a list of tuples, may have received a list "
            "of lists instead. In the future that will raise an error",
            FutureWarning, stacklevel=2)
        return _array
    else:
        # 如果指定了形状且结果数组的形状与之不同，则修改形状
        if shape is not None and retval.shape != shape:
            retval.shape = shape

    # 将结果数组视图转换为 recarray 类型并返回
    res = retval.view(recarray)

    return res


@set_module("numpy.rec")
# 创建一个从二进制数据中生成记录数组的函数
def fromstring(datastring, dtype=None, shape=None, offset=0, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    r"""Create a record array from binary data

    Note that despite the name of this function it does not accept `str`
    instances.

    Parameters
    ----------
    datastring : bytes-like
        Buffer of binary data
    dtype : data-type, optional
        Valid dtype for all arrays
    shape : int or tuple of ints, optional
        Shape of each array.
    offset : int, optional
        Position in the buffer to start reading from.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.

    Returns
    -------
    np.recarray
        Record array view into the data in datastring. This will be readonly
        if `datastring` is readonly.

    See Also
    --------
    numpy.frombuffer

    Examples
    --------
    >>> a = b'\x01\x02\x03abc'
    >>> np.rec.fromstring(a, dtype='u1,u1,u1,S3')
    rec.array([(1, 2, 3, b'abc')],
            dtype=[('f0', 'u1'), ('f1', 'u1'), ('f2', 'u1'), ('f3', 'S3')])

    >>> grades_dtype = [('Name', (np.str_, 10)), ('Marks', np.float64),
    ...                 ('GradeLevel', np.int32)]
    >>> grades_array = np.array([('Sam', 33.3, 3), ('Mike', 44.4, 5),
    ...                         ('Aadi', 66.6, 6)], dtype=grades_dtype)
    >>> np.rec.fromstring(grades_array.tobytes(), dtype=grades_dtype)
    rec.array([('Sam', 33.3, 3), ('Mike', 44.4, 5), ('Aadi', 66.6, 6)],
            dtype=[('Name', '<U10'), ('Marks', '<f8'), ('GradeLevel', '<i4')])

    >>> s = '\x01\x02\x03abc'
    >>> np.rec.fromstring(s, dtype='u1,u1,u1,S3')
    Traceback (most recent call last):
       ...
    TypeError: a bytes-like object is required, not 'str'
    """

    # 如果未提供 dtype 和 formats 参数，则抛出 TypeError 异常
    if dtype is None and formats is None:
        raise TypeError("fromstring() needs a 'dtype' or 'formats' argument")

    # 使用 dtype 参数创建描述符
    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype

    # 计算每个记录的字节大小
    itemsize = descr.itemsize

    # 处理 NumPy 1.19.0 中的 shape 参数
    shape = _deprecate_shape_0_as_None(shape)

    # 如果 shape 参数为 None 或 -1，则根据数据长度和偏移计算 shape
    if shape in (None, -1):
        shape = (len(datastring) - offset) // itemsize

    # 创建记录数组对象并返回
    _array = recarray(shape, descr, buf=datastring, offset=offset)
    return _array

# 获取文件对象中剩余的字节数
def get_remaining_size(fd):
    # 记录当前文件指针位置
    pos = fd.tell()
    try:
        # 将文件指针移到文件末尾并计算文件总大小
        fd.seek(0, 2)
        return fd.tell() - pos  # 返回剩余的字节数
    finally:
        # 恢复文件指针到原始位置
        fd.seek(pos, 0)

@set_module("numpy.rec")
def fromfile(fd, dtype=None, shape=None, offset=0, formats=None,
             names=None, titles=None, aligned=False, byteorder=None):
    """Create an array from binary file data

    Parameters
    ----------
    fd : file-like object
        File object containing binary data
    dtype : data-type, optional
        Valid dtype for all arrays
    shape : int or tuple of ints, optional
        Shape of each array.
    offset : int, optional
        Position in the file to start reading from.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.

    """
    fd : str or file type
        # 参数fd可以是字符串或文件对象，如果是字符串或路径对象，则会打开文件；否则假定它是文件对象。
        # 文件对象必须支持随机访问（即必须具有tell和seek方法）。

    dtype : data-type, optional
        # 数据类型，可选参数。

    shape : int or tuple of ints, optional
        # 数组的形状，可以是整数或整数元组，可选参数。

    offset : int, optional
        # 文件中开始读取的位置，可选参数。

    formats, names, titles, aligned, byteorder :
        # 如果`dtype`为None，则这些参数会传递给`numpy.format_parser`来构造dtype。
        # 参见该函数的详细文档说明。

    Returns
    -------
    np.recarray
        # 返回一个包含文件中数据的记录数组。

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> a = np.empty(10,dtype='f8,i4,a5')
    >>> a[5] = (0.5,10,'abcde')
    >>>
    >>> fd=TemporaryFile()
    >>> a = a.view(a.dtype.newbyteorder('<'))
    >>> a.tofile(fd)
    >>>
    >>> _ = fd.seek(0)
    >>> r=np.rec.fromfile(fd, formats='f8,i4,a5', shape=10,
    ... byteorder='<')
    >>> print(r[5])
    (0.5, 10, b'abcde')
    >>> r.shape
    (10,)
    """

    if dtype is None and formats is None:
        raise TypeError("fromfile() needs a 'dtype' or 'formats' argument")

    # NumPy 1.19.0, 2020-01-01
    shape = _deprecate_shape_0_as_None(shape)
        # 使用_deprecate_shape_0_as_None函数处理shape参数，将形状为0的情况处理为None。

    if shape is None:
        shape = (-1,)
    elif isinstance(shape, int):
        shape = (shape,)
        # 如果shape为None，则设置为(-1,)；如果shape是整数，则转换为元组。

    if hasattr(fd, 'readinto'):
        # GH issue 2504. fd supports io.RawIOBase or io.BufferedIOBase
        # interface. Example of fd: gzip, BytesIO, BufferedReader
        # file already opened
        ctx = nullcontext(fd)
        # 如果fd支持io.RawIOBase或io.BufferedIOBase接口，则使用nullcontext来创建上下文。
    else:
        # open file
        ctx = open(os.fspath(fd), 'rb')
        # 否则，通过os.fspath将fd转换为路径字符串，并以二进制只读模式打开文件。

    with ctx as fd:
        if offset > 0:
            fd.seek(offset, 1)
            # 如果offset大于0，则在文件中移动读取位置。

        size = get_remaining_size(fd)
        # 获取文件中剩余的字节数。

        if dtype is not None:
            descr = sb.dtype(dtype)
            # 如果dtype不为None，则使用sb.dtype处理dtype参数。
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder
            ).dtype
            # 否则，使用format_parser函数根据formats、names、titles、aligned、byteorder构造dtype。

        itemsize = descr.itemsize
        # 获取dtype的每个元素的字节大小。

        shapeprod = sb.array(shape).prod(dtype=nt.intp)
        # 计算shape中所有元素的乘积，并将结果的数据类型设为nt.intp。

        shapesize = shapeprod * itemsize
        # 计算总的形状大小（字节数）。

        if shapesize < 0:
            shape = list(shape)
            shape[shape.index(-1)] = size // -shapesize
            shape = tuple(shape)
            shapeprod = sb.array(shape).prod(dtype=nt.intp)
            # 如果shapesize小于0，则根据文件中剩余的字节数调整shape的值。

        nbytes = shapeprod * itemsize
        # 计算总的字节数。

        if nbytes > size:
            raise ValueError(
                    "Not enough bytes left in file for specified "
                    "shape and type."
                )
            # 如果需要读取的字节数大于文件中剩余的字节数，则引发ValueError异常。

        # create the array
        _array = recarray(shape, descr)
        # 创建一个形状为shape、数据类型为descr的记录数组。

        nbytesread = fd.readinto(_array.data)
        # 将文件中的数据读取到记录数组的数据部分。

        if nbytesread != nbytes:
            raise OSError("Didn't read as many bytes as expected")
            # 如果实际读取的字节数与预期的不符，则引发OSError异常。

    return _array
# 设置模块名称为 "numpy.rec"
@set_module("numpy.rec")
# 定义一个名为 array 的函数，用于构建记录数组
def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
          names=None, titles=None, aligned=False, byteorder=None, copy=True):
    """
    Construct a record array from a wide-variety of objects.

    A general-purpose record array constructor that dispatches to the
    appropriate `recarray` creation function based on the inputs (see Notes).

    Parameters
    ----------
    obj : any
        Input object. See Notes for details on how various input types are
        treated.
    dtype : data-type, optional
        Valid dtype for array.
    shape : int or tuple of ints, optional
        Shape of each array.
    offset : int, optional
        Position in the file or buffer to start reading from.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.
    copy : bool, optional
        Whether to copy the input object (True), or to use a reference instead.
        This option only applies when the input is an ndarray or recarray.
        Defaults to True.

    Returns
    -------
    np.recarray
        Record array created from the specified object.

    Notes
    -----
    If `obj` is ``None``, then call the `~numpy.recarray` constructor. If
    `obj` is a string, then call the `fromstring` constructor. If `obj` is a
    list or a tuple, then if the first object is an `~numpy.ndarray`, call
    `fromarrays`, otherwise call `fromrecords`. If `obj` is a
    `~numpy.recarray`, then make a copy of the data in the recarray
    (if ``copy=True``) and use the new formats, names, and titles. If `obj`
    is a file, then call `fromfile`. Finally, if obj is an `ndarray`, then
    return ``obj.view(recarray)``, making a copy of the data if ``copy=True``.

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.rec.array(a)
    rec.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]],
              dtype=int64)

    >>> b = [(1, 1), (2, 4), (3, 9)]
    >>> c = np.rec.array(b, formats = ['i2', 'f2'], names = ('x', 'y'))
    >>> c
    rec.array([(1, 1.), (2, 4.), (3, 9.)],
              dtype=[('x', '<i2'), ('y', '<f2')])

    >>> c.x
    array([1, 2, 3], dtype=int16)

    >>> c.y
    array([1.,  4.,  9.], dtype=float16)

    >>> r = np.rec.array(['abc','def'], names=['col1','col2'])
    >>> print(r.col1)
    abc

    >>> r.col1
    array('abc', dtype='<U3')

    >>> r.col2
    array('def', dtype='<U3')
    """
    # 检查对象是否为None、字符串或具有'readinto'属性，并且formats和dtype都未定义时，抛出值错误异常
    if ((isinstance(obj, (type(None), str)) or hasattr(obj, 'readinto')) and
           formats is None and dtype is None):
        raise ValueError("Must define formats (or dtype) if object is "
                         "None, string, or an open file")

    # 初始化一个空字典用于存储关键字参数
    kwds = {}

    # 如果定义了dtype，则将其转换为dtype对象
    if dtype is not None:
        dtype = sb.dtype(dtype)
    
    # 如果formats不为None，则使用format_parser函数解析formats参数，并获取其dtype
    elif formats is not None:
        dtype = format_parser(formats, names, titles,
                              aligned, byteorder).dtype
    
    # 如果formats和dtype都未定义，则将关键字参数设置为包含这些参数的字典
    else:
        kwds = {'formats': formats,
                'names': names,
                'titles': titles,
                'aligned': aligned,
                'byteorder': byteorder
                }

    # 如果obj为None，则根据指定的shape创建一个recarray对象，使用给定的dtype、buf、offset和strides参数
    if obj is None:
        if shape is None:
            raise ValueError("Must define a shape if obj is None")
        return recarray(shape, dtype, buf=obj, offset=offset, strides=strides)

    # 如果obj是字节串，则调用fromstring函数解析为数组对象，使用给定的dtype、shape和kwds参数
    elif isinstance(obj, bytes):
        return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

    # 如果obj是列表或元组，则根据其第一个元素的类型判断调用fromrecords或fromarrays函数，使用给定的dtype、shape和kwds参数
    elif isinstance(obj, (list, tuple)):
        if isinstance(obj[0], (tuple, list)):
            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
        else:
            return fromarrays(obj, dtype=dtype, shape=shape, **kwds)

    # 如果obj是recarray类型，则根据情况进行视图转换和复制操作，并返回新的对象
    elif isinstance(obj, recarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new

    # 如果obj具有'readinto'属性，则调用fromfile函数从文件对象创建数组对象，使用给定的dtype、shape和offset参数
    elif hasattr(obj, 'readinto'):
        return fromfile(obj, dtype=dtype, shape=shape, offset=offset)

    # 如果obj是ndarray类型，则根据情况进行视图转换和复制操作，并返回新的recarray视图对象
    elif isinstance(obj, ndarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new.view(recarray)

    # 对于其他情况，尝试获取obj的__array_interface__属性，如果未定义或者不是字典类型，则抛出值错误异常
    else:
        interface = getattr(obj, "__array_interface__", None)
        if interface is None or not isinstance(interface, dict):
            raise ValueError("Unknown input type")
        # 将obj转换为数组对象，并根据情况进行视图转换操作，返回recarray视图对象
        obj = sb.array(obj)
        if dtype is not None and (obj.dtype != dtype):
            obj = obj.view(dtype)
        return obj.view(recarray)
```