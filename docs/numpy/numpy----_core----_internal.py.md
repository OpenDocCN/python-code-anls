# `.\numpy\numpy\_core\_internal.py`

```
"""
A place for internal code

Some things are more easily handled Python.

"""
# 导入必要的模块和异常
import ast                 # AST（抽象语法树）模块，用于处理 Python 的语法分析树
import re                  # 正则表达式模块，用于字符串匹配和操作
import sys                 # sys 模块，提供对解释器相关的操作
import warnings            # 警告模块，用于发出警告信息

# 从 numpy 中导入特定模块和类
from ..exceptions import DTypePromotionError   # 导入自定义的数据类型提升异常类
from .multiarray import dtype, array, ndarray, promote_types, StringDType  # 导入多维数组相关类和函数
from numpy import _NoValue   # 导入 numpy 中的特定值

try:
    import ctypes   # 尝试导入 ctypes 模块，用于处理 C 语言数据类型
except ImportError:
    ctypes = None

# 判断当前 Python 解释器是否为 PyPy
IS_PYPY = sys.implementation.name == 'pypy'

# 根据系统的字节顺序设置数据格式化字符串
if sys.byteorder == 'little':
    _nbo = '<'
else:
    _nbo = '>'

# 定义一个函数用于处理字段列表，根据给定的对齐方式生成相应的字段信息
def _makenames_list(adict, align):
    allfields = []

    for fname, obj in adict.items():
        n = len(obj)
        if not isinstance(obj, tuple) or n not in (2, 3):
            raise ValueError("entry not a 2- or 3- tuple")
        if n > 2 and obj[2] == fname:
            continue
        num = int(obj[1])
        if num < 0:
            raise ValueError("invalid offset.")
        format = dtype(obj[0], align=align)
        if n > 2:
            title = obj[2]
        else:
            title = None
        allfields.append((fname, format, num, title))
    # 按偏移量排序
    allfields.sort(key=lambda x: x[2])
    names = [x[0] for x in allfields]
    formats = [x[1] for x in allfields]
    offsets = [x[2] for x in allfields]
    titles = [x[3] for x in allfields]

    return names, formats, offsets, titles

# 当一个字典在数据类型描述符中被使用时，在 PyArray_DescrConverter 函数中调用
# 如果字典缺少 "names" 和 "formats" 字段，则使用此函数生成数据类型
def _usefields(adict, align):
    try:
        names = adict[-1]
    except KeyError:
        names = None
    if names is None:
        names, formats, offsets, titles = _makenames_list(adict, align)
    else:
        formats = []
        offsets = []
        titles = []
        for name in names:
            res = adict[name]
            formats.append(res[0])
            offsets.append(res[1])
            if len(res) > 2:
                titles.append(res[2])
            else:
                titles.append(None)

    return dtype({"names": names,
                  "formats": formats,
                  "offsets": offsets,
                  "titles": titles}, align)


# 构建一个数组协议描述符列表
# 从描述符的字段属性中递归调用自身，直到最终找到没有字段的描述符，然后返回一个简单的类型字符串
def _array_descr(descriptor):
    fields = descriptor.fields
    if fields is None:
        subdtype = descriptor.subdtype
        if subdtype is None:
            if descriptor.metadata is None:
                return descriptor.str
            else:
                new = descriptor.metadata.copy()
                if new:
                    return (descriptor.str, new)
                else:
                    return descriptor.str
        else:
            return (_array_descr(subdtype[0]), subdtype[1])

    names = descriptor.names
    ordered_fields = [fields[x] + (x,) for x in names]
    result = []
    offset = 0
    # 对于给定的有序字段列表，依次处理每个字段
    for field in ordered_fields:
        # 检查字段的偏移量是否大于当前偏移量
        if field[1] > offset:
            # 计算当前字段的偏移量增量
            num = field[1] - offset
            # 将空字符串和偏移量增量信息追加到结果列表中
            result.append(('', f'|V{num}'))
            # 更新当前偏移量
            offset += num
        # 如果字段的偏移量小于当前偏移量，抛出数值错误异常
        elif field[1] < offset:
            raise ValueError(
                "dtype.descr is not defined for types with overlapping or "
                "out-of-order fields")
        
        # 如果字段列表的长度大于3，将字段名称设为元组(field[2], field[3])
        if len(field) > 3:
            name = (field[2], field[3])
        else:
            # 否则，将字段名称设为field[2]
            name = field[2]
        
        # 如果字段的子数据类型存在
        if field[0].subdtype:
            # 创建一个元组，包含字段名称、数组描述和子数据类型的信息
            tup = (name, _array_descr(field[0].subdtype[0]),
                   field[0].subdtype[1])
        else:
            # 否则，创建一个元组，包含字段名称和数组描述的信息
            tup = (name, _array_descr(field[0]))
        
        # 更新当前偏移量，加上当前字段的数据项大小
        offset += field[0].itemsize
        # 将处理后的字段信息元组追加到结果列表中
        result.append(tup)
    
    # 如果描述符的数据项大小大于当前偏移量
    if descriptor.itemsize > offset:
        # 计算描述符的数据项大小与当前偏移量之间的差值
        num = descriptor.itemsize - offset
        # 将空字符串和差值信息追加到结果列表中
        result.append(('', f'|V{num}'))
    
    # 返回最终的结果列表
    return result
# 正则表达式，用于解析格式字符串中的各个部分
format_re = re.compile(r'(?P<order1>[<>|=]?)'  # 匹配第一个字节顺序标识符（可选）
                       r'(?P<repeats> *[(]?[ ,0-9]*[)]? *)'  # 匹配重复次数或格式字段
                       r'(?P<order2>[<>|=]?)'  # 匹配第二个字节顺序标识符（可选）
                       r'(?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)')  # 匹配数据类型描述

# 正则表达式，用于匹配逗号分隔的字符串中的逗号
sep_re = re.compile(r'\s*,\s*')

# 正则表达式，用于匹配字符串末尾的空格
space_re = re.compile(r'\s+$')

# 用于处理等号对应的转换字典
_convorder = {'=': _nbo}

def _commastring(astr):
    # 开始索引初始化为0
    startindex = 0
    # 结果列表
    result = []
    # 是否为列表
    islist = False

    # 当开始索引小于字符串长度时循环
    while startindex < len(astr):
        # 使用 format_re 正则表达式匹配字符串 astr，从 startindex 开始匹配
        mo = format_re.match(astr, pos=startindex)
        try:
            # 尝试解构正则匹配结果为四个部分：order1, repeats, order2, dtype
            (order1, repeats, order2, dtype) = mo.groups()
        except (TypeError, AttributeError):
            # 捕获可能的异常，抛出包含详细信息的 ValueError
            raise ValueError(
                f'format number {len(result)+1} of "{astr}" is not recognized'
                ) from None
        startindex = mo.end()

        # 如果开始索引小于字符串长度
        if startindex < len(astr):
            # 如果是末尾的空格，将开始索引设为字符串长度
            if space_re.match(astr, pos=startindex):
                startindex = len(astr)
            else:
                # 否则尝试使用 sep_re 匹配逗号分隔符
                mo = sep_re.match(astr, pos=startindex)
                if not mo:
                    # 如果不匹配，抛出异常
                    raise ValueError(
                        'format number %d of "%s" is not recognized' %
                        (len(result)+1, astr))
                startindex = mo.end()
                # 设置为列表标记为 True
                islist = True

        # 根据 order1 和 order2 确定顺序标识符 order
        if order2 == '':
            order = order1
        elif order1 == '':
            order = order2
        else:
            # 使用 _convorder 获取对应的顺序标识符，如果不一致则抛出异常
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if (order1 != order2):
                raise ValueError(
                    'inconsistent byte-order specification %s and %s' %
                    (order1, order2))
            order = order1

        # 如果顺序标识符为 '|', '=', _nbo 中的一种，则置为空字符串
        if order in ('|', '=', _nbo):
            order = ''
        # 将顺序标识符与 dtype 组合成新的数据类型描述
        dtype = order + dtype

        # 如果 repeats 为空，则新项为 dtype；否则将 repeats 转换为字面量并与 dtype 组合成元组
        if repeats == '':
            newitem = dtype
        else:
            if (repeats[0] == "(" and repeats[-1] == ")"
                    and repeats[1:-1].strip() != ""
                    and "," not in repeats):
                # 警告信息，提示传入的重复次数格式已过时
                warnings.warn(
                    'Passing in a parenthesized single number for repeats '
                    'is deprecated; pass either a single number or indicate '
                    'a tuple with a comma, like "(2,)".', DeprecationWarning,
                    stacklevel=2)
            newitem = (dtype, ast.literal_eval(repeats))

        # 将新项添加到结果列表中
        result.append(newitem)

    # 如果 islist 为 True，返回结果列表；否则返回结果列表的第一个元素
    return result if islist else result[0]

# dummy_ctype 类的定义，用于模拟 C 类型
class dummy_ctype:

    def __init__(self, cls):
        self._cls = cls

    def __mul__(self, other):
        # 乘法运算符重载，返回 self
        return self

    def __call__(self, *other):
        # 调用运算符重载，返回传入参数的类对象
        return self._cls(other)

    def __eq__(self, other):
        # 等于运算符重载，比较两个对象的类是否相等
        return self._cls == other._cls

    def __ne__(self, other):
        # 不等于运算符重载，比较两个对象的类是否不相等
        return self._cls != other._cls

def _getintp_ctype():
    # 获取 _getintp_ctype 的缓存值，如果不为 None 则直接返回
    val = _getintp_ctype.cache
    if val is not None:
        return val
    # 检查 ctypes 是否可用，如果不可用则导入 numpy 并使用 dummy_ctype 处理 np.intp 类型
    if ctypes is None:
        import numpy as np
        val = dummy_ctype(np.intp)
    else:
        # 获取 dtype 对象的字符表示
        char = dtype('n').char
        # 根据字符表示选择对应的 ctypes 类型
        if char == 'i':
            val = ctypes.c_int
        elif char == 'l':
            val = ctypes.c_long
        elif char == 'q':
            val = ctypes.c_longlong
        else:
            val = ctypes.c_long
    # 将确定的 ctypes 类型缓存到 _getintp_ctype.cache 中
    _getintp_ctype.cache = val
    # 返回确定的 ctypes 类型
    return val
# Reset the cache attribute `_getintp_ctype.cache` to None
_getintp_ctype.cache = None

# Define a class `_missing_ctypes` used to emulate ctypes functionality if ctypes is unavailable
class _missing_ctypes:
    # Define a method `cast` that returns the value of `num`
    def cast(self, num, obj):
        return num.value
    
    # Define a nested class `c_void_p` that initializes with a pointer `ptr` and sets `value` attribute to `ptr`
    class c_void_p:
        def __init__(self, ptr):
            self.value = ptr

# Define a class `_ctypes` used for managing ctypes-related operations
class _ctypes:
    # Initialize with an array `array` and optional pointer `ptr`
    def __init__(self, array, ptr=None):
        self._arr = array
        
        # Check if ctypes module is available
        if ctypes:
            self._ctypes = ctypes  # Use standard ctypes if available
            self._data = self._ctypes.c_void_p(ptr)  # Initialize with c_void_p object
        else:
            # Emulate ctypes functionality using `_missing_ctypes` if ctypes is unavailable
            self._ctypes = _missing_ctypes()
            self._data = self._ctypes.c_void_p(ptr)
            self._data._objects = array  # Attach array reference to `_data`

        # Determine if array is zero-dimensional
        if self._arr.ndim == 0:
            self._zerod = True
        else:
            self._zerod = False

    # Method to return data pointer cast to a specified ctypes object `obj`
    def data_as(self, obj):
        """
        Return the data pointer cast to a particular c-types object.
        For example, calling ``self._as_parameter_`` is equivalent to
        ``self.data_as(ctypes.c_void_p)``. Perhaps you want to use
        the data as a pointer to a ctypes array of floating-point data:
        ``self.data_as(ctypes.POINTER(ctypes.c_double))``.

        The returned pointer will keep a reference to the array.
        """
        # Workaround for CPython bug causing circular reference with `_ctypes.cast`
        ptr = self._ctypes.cast(self._data, obj)  # Cast `_data` to `obj`
        ptr._arr = self._arr  # Attach `_arr` to `ptr` to maintain array reference
        return ptr

    # Method to return shape tuple as an array of a specified c-types type `obj`
    def shape_as(self, obj):
        """
        Return the shape tuple as an array of some other c-types
        type. For example: ``self.shape_as(ctypes.c_short)``.
        """
        if self._zerod:
            return None  # Return None if array is zero-dimensional
        return (obj * self._arr.ndim)(*self._arr.shape)  # Create array of shape using `obj`

    # Method to return strides tuple as an array of a specified c-types type `obj`
    def strides_as(self, obj):
        """
        Return the strides tuple as an array of some other
        c-types type. For example: ``self.strides_as(ctypes.c_longlong)``.
        """
        if self._zerod:
            return None  # Return None if array is zero-dimensional
        return (obj * self._arr.ndim)(*self._arr.strides)  # Create array of strides using `obj`

    @property
    def data(self):
        """
        返回数组内存区域的指针作为 Python 整数。
        这个内存区域可能包含未对齐或不正确字节顺序的数据。
        甚至这个内存区域可能是不可写的。
        当将这个属性传递给任意的 C 代码时，应该尊重数组的标志和数据类型，
        以避免可能导致 Python 崩溃的问题。用户注意！
        此属性的值与 `self._array_interface_['data'][0]` 完全相同。

        与 `data_as` 不同，注意不会保留对数组的引用：
        例如 `ctypes.c_void_p((a + b).ctypes.data)` 将导致指向已释放数组的指针，
        应该写作 `(a + b).ctypes.data_as(ctypes.c_void_p)`
        """
        return self._data.value

    @property
    def shape(self):
        """
        (c_intp*self.ndim): 长度为 self.ndim 的 ctypes 数组，
        基本类型是对应于此平台上 `dtype('p')` 的 C 整数（参见 `~numpy.ctypeslib.c_intp`）。
        这个基本类型可能是 `ctypes.c_int`、`ctypes.c_long` 或 `ctypes.c_longlong`，具体取决于平台。
        这个 ctypes 数组包含底层数组的形状信息。
        """
        return self.shape_as(_getintp_ctype())

    @property
    def strides(self):
        """
        (c_intp*self.ndim): 长度为 self.ndim 的 ctypes 数组，
        基本类型与 shape 属性相同。这个 ctypes 数组包含底层数组的跨步信息。
        跨步信息很重要，因为它显示了在数组中跳转到下一个元素需要跳过多少字节。
        """
        return self.strides_as(_getintp_ctype())

    @property
    def _as_parameter_(self):
        """
        覆盖 ctypes 的半神奇方法

        允许 `c_func(some_array.ctypes)`
        """
        return self.data_as(ctypes.c_void_p)

    # Numpy 1.21.0, 2021-05-18

    def get_data(self):
        """已弃用的 `_ctypes.data` 属性的获取器。

        .. deprecated:: 1.21
        """
        warnings.warn('"get_data" 已弃用。请使用 "data"',
                      DeprecationWarning, stacklevel=2)
        return self.data

    def get_shape(self):
        """已弃用的 `_ctypes.shape` 属性的获取器。

        .. deprecated:: 1.21
        """
        warnings.warn('"get_shape" 已弃用。请使用 "shape"',
                      DeprecationWarning, stacklevel=2)
        return self.shape

    def get_strides(self):
        """已弃用的 `_ctypes.strides` 属性的获取器。

        .. deprecated:: 1.21
        """
        warnings.warn('"get_strides" 已弃用。请使用 "strides"',
                      DeprecationWarning, stacklevel=2)
        return self.strides
    # 定义一个方法 `get_as_parameter`，用于获取 `_ctypes._as_parameter_` 属性值。
    # 此方法已弃用，用于返回 `_ctypes._as_parameter_` 属性的值。
    def get_as_parameter(self):
        """Deprecated getter for the `_ctypes._as_parameter_` property.

        .. deprecated:: 1.21
        """
        # 发出警告，提醒使用者方法已弃用，建议使用 `_as_parameter_` 替代
        warnings.warn(
            '"get_as_parameter" is deprecated. Use "_as_parameter_" instead',
            DeprecationWarning, stacklevel=2,
        )
        # 返回 `_ctypes._as_parameter_` 的值
        return self._as_parameter_
# 给定数据类型和排序对象，返回新的字段名元组，按指定顺序排列
def _newnames(datatype, order):
    # 获取旧字段名元组
    oldnames = datatype.names
    # 转换为列表进行操作
    nameslist = list(oldnames)
    # 如果 order 是字符串，转换为列表
    if isinstance(order, str):
        order = [order]
    seen = set()
    # 如果 order 是列表或元组，则遍历处理
    if isinstance(order, (list, tuple)):
        for name in order:
            try:
                nameslist.remove(name)
            except ValueError:
                # 如果字段名重复，抛出异常
                if name in seen:
                    raise ValueError(f"duplicate field name: {name}") from None
                else:
                    raise ValueError(f"unknown field name: {name}") from None
            seen.add(name)
        # 返回按顺序排列后的新字段名元组
        return tuple(list(order) + nameslist)
    # 如果 order 类型不支持，抛出异常
    raise ValueError(f"unsupported order value: {order}")

# 返回结构化数组的副本，移除字段之间的填充字节
def _copy_fields(ary):
    """Return copy of structured array with padding between fields removed.

    Parameters
    ----------
    ary : ndarray
       Structured array from which to remove padding bytes

    Returns
    -------
    ary_copy : ndarray
       Copy of ary with padding bytes removed
    """
    # 获取数组的数据类型
    dt = ary.dtype
    # 创建副本数据类型，包括字段名和格式
    copy_dtype = {'names': dt.names,
                  'formats': [dt.fields[name][0] for name in dt.names]}
    # 返回副本数组，使用副本数据类型，并确保复制数据
    return array(ary, dtype=copy_dtype, copy=True)

# 对两个结构化数据类型执行类型提升
def _promote_fields(dt1, dt2):
    """ Perform type promotion for two structured dtypes.

    Parameters
    ----------
    dt1 : structured dtype
        First dtype.
    dt2 : structured dtype
        Second dtype.

    Returns
    -------
    out : dtype
        The promoted dtype

    Notes
    -----
    If one of the inputs is aligned, the result will be.  The titles of
    both descriptors must match (point to the same field).
    """
    # 必须都是结构化的，并且字段名必须相同且顺序一致
    if (dt1.names is None or dt2.names is None) or dt1.names != dt2.names:
        raise DTypePromotionError(
                f"field names `{dt1.names}` and `{dt2.names}` mismatch.")

    # 如果两者完全相同，则可能可以直接返回相同的数据类型
    identical = dt1 is dt2
    new_fields = []
    for name in dt1.names:
        field1 = dt1.fields[name]
        field2 = dt2.fields[name]
        # 获取提升后的新描述符
        new_descr = promote_types(field1[0], field2[0])
        identical = identical and new_descr is field1[0]

        # 检查标题是否匹配（如果有的话）
        if field1[2:] != field2[2:]:
            raise DTypePromotionError(
                    f"field titles of field '{name}' mismatch")
        # 如果长度为2，则直接添加，否则添加元组（包含标题和字段名）
        if len(field1) == 2:
            new_fields.append((name, new_descr))
        else:
            new_fields.append(((field1[2], name), new_descr))

    # 返回新的数据类型，保持对齐性
    res = dtype(new_fields, align=dt1.isalignedstruct or dt2.isalignedstruct)

    # 如果数据类型完全相同，则保留标识（和元数据）
    # 如果 itemsize 和偏移量也没有修改，则应该保留标识
    # 可能可以加快速度，但也可能完全删除。
    # 如果条件 identical 为真且 res 的元素大小与 dt1 的元素大小相同，则执行以下操作
    if identical and res.itemsize == dt1.itemsize:
        # 遍历 dt1 的字段名
        for name in dt1.names:
            # 如果 dt1 的字段 name 对应的第二个元素类型不等于 res 中相同字段 name 对应的第二个元素类型
            if dt1.fields[name][1] != res.fields[name][1]:
                # 返回 res，表示 dtype 发生了变化。
                return res  # the dtype changed.
        # 如果以上条件都没有触发返回，返回 dt1，表示 dtype 未发生变化。
        return dt1

    # 如果条件不满足，返回 res。
    return res
# 定义一个函数，用于检查对对象数组进行 getfield 操作的安全性
def _getfield_is_safe(oldtype, newtype, offset):
    """ Checks safety of getfield for object arrays.

    As in _view_is_safe, we need to check that memory containing objects is not
    reinterpreted as a non-object datatype and vice versa.

    Parameters
    ----------
    oldtype : data-type
        Data type of the original ndarray.
    newtype : data-type
        Data type of the field being accessed by ndarray.getfield
    offset : int
        Offset of the field being accessed by ndarray.getfield

    Raises
    ------
    TypeError
        If the field access is invalid

    """
    # 检查是否涉及对象类型，如果涉及则进行安全性检查
    if newtype.hasobject or oldtype.hasobject:
        # 如果偏移量为 0 并且新类型与旧类型相同，则返回
        if offset == 0 and newtype == oldtype:
            return
        # 如果旧类型具有命名字段，则遍历每个字段
        if oldtype.names is not None:
            for name in oldtype.names:
                # 检查字段偏移量和类型是否匹配
                if (oldtype.fields[name][1] == offset and
                        oldtype.fields[name][0] == newtype):
                    return
        # 若不满足上述条件，则表示无法进行字段的 get/set 操作，抛出异常
        raise TypeError("Cannot get/set field of an object array")
    return


# 定义一个函数，用于检查涉及对象数组的视图操作的安全性
def _view_is_safe(oldtype, newtype):
    """ Checks safety of a view involving object arrays, for example when
    doing::

        np.zeros(10, dtype=oldtype).view(newtype)

    Parameters
    ----------
    oldtype : data-type
        Data type of original ndarray
    newtype : data-type
        Data type of the view

    Raises
    ------
    TypeError
        If the new type is incompatible with the old type.

    """

    # 若类型相同，则视图操作安全无需检查
    if oldtype == newtype:
        return

    # 若涉及到对象类型，则视图操作不安全，抛出异常
    if newtype.hasobject or oldtype.hasobject:
        raise TypeError("Cannot change data-type for array of references.")
    return


# PEP 3118 格式转换映射表，用于构造 NumPy 的 dtype
_pep3118_native_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'h',
    'H': 'H',
    'i': 'i',
    'I': 'I',
    'l': 'l',
    'L': 'L',
    'q': 'q',
    'Q': 'Q',
    'e': 'e',
    'f': 'f',
    'd': 'd',
    'g': 'g',
    'Zf': 'F',
    'Zd': 'D',
    'Zg': 'G',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # 填充
}
_pep3118_native_typechars = ''.join(_pep3118_native_map.keys())

# PEP 3118 标准格式转换映射表
_pep3118_standard_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'i2',
    'H': 'u2',
    'i': 'i4',
    'I': 'u4',
    'l': 'i4',
    'L': 'u4',
    'q': 'i8',
    'Q': 'u8',
    'e': 'f2',
    'f': 'f',
    'd': 'd',
    'Zf': 'F',
    'Zd': 'D',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # 填充
}
_pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())

# PEP 3118 不支持的格式映射表
_pep3118_unsupported_map = {
    'u': 'UCS-2 strings',
    '&': 'pointers',
    't': 'bitfields',
    'X': 'function pointers',
}

# _Stream 类定义，用于处理字节流
class _Stream:
    def __init__(self, s):
        self.s = s
        self.byteorder = '@'

    # 前进方法，用于从流中读取指定数量的字节并返回
    def advance(self, n):
        res = self.s[:n]
        self.s = self.s[n:]
        return res
    # 检查字符串起始是否与给定字符串 c 匹配，若匹配则消耗相同长度的字符串并返回 True，否则返回 False
    def consume(self, c):
        if self.s[:len(c)] == c:
            self.advance(len(c))  # 消耗与 c 相同长度的字符串
            return True
        return False

    # 消耗当前字符串直到遇到符合条件的字符或函数 c，返回消耗的部分字符串
    def consume_until(self, c):
        if callable(c):  # 如果 c 是可调用对象（函数）
            i = 0
            while i < len(self.s) and not c(self.s[i]):
                i = i + 1  # 查找第一个使得 c 返回 True 的位置
            return self.advance(i)  # 消耗 i 长度的字符串并返回
        else:
            i = self.s.index(c)  # 找到字符串中第一次出现字符 c 的位置
            res = self.advance(i)  # 消耗 i 长度的字符串并返回
            self.advance(len(c))  # 再消耗 c 的长度
            return res

    # 返回当前字符串的第一个字符
    @property
    def next(self):
        return self.s[0]

    # 检查当前字符串是否为真（非空）
    def __bool__(self):
        return bool(self.s)
# 从 PEP3118 规范中解析数据类型，返回解析后的数据类型对象
def _dtype_from_pep3118(spec):
    # 创建一个流对象来处理规范
    stream = _Stream(spec)
    # 调用内部函数来解析 PEP3118 规范，获取数据类型和对齐方式
    dtype, align = __dtype_from_pep3118(stream, is_subdtype=False)
    # 返回解析后的数据类型对象
    return dtype

# 内部函数：从流中解析 PEP3118 规范，返回数据类型和对齐方式
def __dtype_from_pep3118(stream, is_subdtype):
    # 定义字段规范的初始状态
    field_spec = dict(
        names=[],
        formats=[],
        offsets=[],
        itemsize=0
    )
    offset = 0
    common_alignment = 1
    is_padding = False

    # 解析规范
    # 对齐类型的最后额外填充
    if stream.byteorder == '@':
        field_spec['itemsize'] += (-offset) % common_alignment

    # 检查是否为简单的单项类型，如果是则展开它
    if (field_spec['names'] == [None]
            and field_spec['offsets'][0] == 0
            and field_spec['itemsize'] == field_spec['formats'][0].itemsize
            and not is_subdtype):
        ret = field_spec['formats'][0]
    else:
        # 修正字段名称
        _fix_names(field_spec)
        # 创建数据类型对象
        ret = dtype(field_spec)

    # 返回解析后的数据类型对象和通用对齐方式
    return ret, common_alignment

# 修正字段名称函数：将为 None 的字段名称替换为下一个未使用的 f%d 名称
def _fix_names(field_spec):
    names = field_spec['names']
    for i, name in enumerate(names):
        if name is not None:
            continue

        j = 0
        while True:
            name = f'f{j}'
            if name not in names:
                break
            j = j + 1
        names[i] = name

# 在数据类型的末尾注入指定数量的填充字节
def _add_trailing_padding(value, padding):
    if value.fields is None:
        # 创建只包含一个字段的字段规范
        field_spec = dict(
            names=['f0'],
            formats=[value],
            offsets=[0],
            itemsize=value.itemsize
        )
    else:
        # 从现有数据类型中创建字段规范
        fields = value.fields
        names = value.names
        field_spec = dict(
            names=names,
            formats=[fields[name][0] for name in names],
            offsets=[fields[name][1] for name in names],
            itemsize=value.itemsize
        )

    # 增加填充字节到字段规范的 itemsize
    field_spec['itemsize'] += padding
    return dtype(field_spec)

# 计算列表中所有元素的乘积
def _prod(a):
    p = 1
    for x in a:
        p *= x
    return p

# 计算 a 和 b 的最大公约数
def _gcd(a, b):
    """Calculate the greatest common divisor of a and b"""
    while b:
        a, b = b, a % b
    return a

# 计算 a 和 b 的最小公倍数
def _lcm(a, b):
    return a // _gcd(a, b) * b

# 格式化当 __array_ufunc__ 放弃时的错误消息
def array_ufunc_errmsg_formatter(dummy, ufunc, method, *inputs, **kwargs):
    args_string = ', '.join(['{!r}'.format(arg) for arg in inputs] +
                            ['{}={!r}'.format(k, v)
                             for k, v in kwargs.items()])
    args = inputs + kwargs.get('out', ())
    types_string = ', '.join(repr(type(arg).__name__) for arg in args)
    return ('operand type(s) all returned NotImplemented from '
            '__array_ufunc__({!r}, {!r}, {}): {}'
            .format(ufunc, method, args_string, types_string))

# 格式化当 __array_ufunc__ 放弃时的错误消息
def array_function_errmsg_formatter(public_api, types):
    """ Format the error message for when __array_ufunc__ gives up. """
    # 构造函数名，格式化为字符串，包含模块名和函数名
    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    # 返回一条错误消息，指明没有在实现 '__array_function__' 的类型中找到对应函数的实现
    return ("no implementation found for '{}' on types that implement "
            '__array_function__: {}'.format(func_name, list(types)))
# 构建函数签名字符串，类似于 PEP 457 的格式
def _ufunc_doc_signature_formatter(ufunc):
    # 如果输入参数个数为1，则输入参数为 'x'
    if ufunc.nin == 1:
        in_args = 'x'
    else:
        # 否则，输入参数为 'x1, x2, ...' 的格式，根据输入参数个数生成
        in_args = ', '.join(f'x{i+1}' for i in range(ufunc.nin))

    # 输出参数可以是关键字参数或位置参数
    if ufunc.nout == 0:
        out_args = ', /, out=()'
    elif ufunc.nout == 1:
        out_args = ', /, out=None'
    else:
        # 如果有多个输出参数，则输出参数为 '[, out1, out2, ...]' 的格式
        out_args = '[, {positional}], / [, out={default}]'.format(
            positional=', '.join(
                'out{}'.format(i+1) for i in range(ufunc.nout)),
            default=repr((None,)*ufunc.nout)
        )

    # 关键字参数包括一些固定的参数和可变的参数
    kwargs = (
        ", casting='same_kind'"
        ", order='K'"
        ", dtype=None"
        ", subok=True"
    )

    # 注意：gufunc 可能支持 `axis` 参数，也可能不支持
    if ufunc.signature is None:
        kwargs = f", where=True{kwargs}[, signature]"
    else:
        kwargs += "[, signature, axes, axis]"

    # 拼接所有部分成为完整的函数签名字符串
    return '{name}({in_args}{out_args}, *{kwargs})'.format(
        name=ufunc.__name__,
        in_args=in_args,
        out_args=out_args,
        kwargs=kwargs
    )


# 检查一个类是否来自 ctypes，以解决针对这些对象的缓冲区协议中的 bug，参见 bpo-10746
def npy_ctypes_check(cls):
    try:
        # ctypes 类是新式类，因此有 __mro__ 属性。对于具有多重继承的 ctypes 类，这可能失败。
        if IS_PYPY:
            # (..., _ctypes.basics._CData, Bufferable, object)
            ctype_base = cls.__mro__[-3]
        else:
            # # (..., _ctypes._CData, object)
            ctype_base = cls.__mro__[-2]
        # 检查基类是否属于 _ctypes 模块
        return '_ctypes' in ctype_base.__module__
    except Exception:
        # 出现异常时返回 False
        return False


# 用于处理 na_object 参数在 stringdtype 的 C 实现中 __reduce__ 方法的 _NoValue 默认参数
def _convert_to_stringdtype_kwargs(coerce, na_object=_NoValue):
    if na_object is _NoValue:
        # 如果 na_object 是 _NoValue，默认创建一个 StringDType 对象
        return StringDType(coerce=coerce)
    else:
        # 否则，使用指定的 na_object 创建 StringDType 对象
        return StringDType(coerce=coerce, na_object=na_object)
```