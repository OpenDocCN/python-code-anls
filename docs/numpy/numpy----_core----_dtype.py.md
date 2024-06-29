# `.\numpy\numpy\_core\_dtype.py`

```
"""
A place for code to be called from the implementation of np.dtype

String handling is much easier to do correctly in python.
"""
# 导入 numpy 库，并使用别名 np
import numpy as np

# 将数据类型的种类映射为相应的基本类型名称
_kind_to_stem = {
    'u': 'uint',    # 无符号整数
    'i': 'int',     # 有符号整数
    'c': 'complex', # 复数
    'f': 'float',   # 浮点数
    'b': 'bool',    # 布尔值
    'V': 'void',    # void（空类型）
    'O': 'object',  # Python 对象
    'M': 'datetime',# 日期时间
    'm': 'timedelta', # 时间间隔
    'S': 'bytes',   # 字节串
    'U': 'str',     # Unicode 字符串
}

# 根据数据类型返回其种类对应的基本类型名称
def _kind_name(dtype):
    try:
        return _kind_to_stem[dtype.kind]
    except KeyError as e:
        raise RuntimeError(
            "internal dtype error, unknown kind {!r}"
            .format(dtype.kind)
        ) from None

# 根据数据类型返回其字符串表示形式
def __str__(dtype):
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=True)  # 返回结构化数据类型的字符串表示
    elif dtype.subdtype:
        return _subarray_str(dtype)  # 返回子数组数据类型的字符串表示
    elif issubclass(dtype.type, np.flexible) or not dtype.isnative:
        return dtype.str  # 返回灵活数据类型或非本机字节顺序数据类型的字符串表示
    else:
        return dtype.name  # 返回数据类型的名称

# 返回数据类型的详细表示形式
def __repr__(dtype):
    arg_str = _construction_repr(dtype, include_align=False)  # 获取构造数据类型的字符串表示形式
    if dtype.isalignedstruct:
        arg_str = arg_str + ", align=True"  # 如果数据类型是对齐结构，则添加 align=True 参数
    return "dtype({})".format(arg_str)  # 返回数据类型的构造函数表示形式

# 解包数据类型字段的帮助函数，返回标准化后的字段信息
def _unpack_field(dtype, offset, title=None):
    """
    Helper function to normalize the items in dtype.fields.

    Call as:

    dtype, offset, title = _unpack_field(*dtype.fields[name])
    """
    return dtype, offset, title

# 检查数据类型是否为未定义大小的数据类型
def _isunsized(dtype):
    # PyDataType_ISUNSIZED
    return dtype.itemsize == 0  # 返回数据类型的元素大小是否为零

# 构造数据类型的字符串表示形式，不包括 'dtype()' 部分
def _construction_repr(dtype, include_align=False, short=False):
    """
    Creates a string repr of the dtype, excluding the 'dtype()' part
    surrounding the object. This object may be a string, a list, or
    a dict depending on the nature of the dtype. This
    is the object passed as the first parameter to the dtype
    constructor, and if no additional constructor parameters are
    given, will reproduce the exact memory layout.

    Parameters
    ----------
    short : bool
        If true, this creates a shorter repr using 'kind' and 'itemsize',
        instead of the longer type name.

    include_align : bool
        If true, this includes the 'align=True' parameter
        inside the struct dtype construction dict when needed. Use this flag
        if you want a proper repr string without the 'dtype()' part around it.

        If false, this does not preserve the
        'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for
        struct arrays like the regular repr does, because the 'align'
        flag is not part of first dtype constructor parameter. This
        mode is intended for a full 'repr', where the 'align=True' is
        provided as the second parameter.
    """
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=include_align)  # 返回结构化数据类型的字符串表示
    elif dtype.subdtype:
        return _subarray_str(dtype)  # 返回子数组数据类型的字符串表示
    else:
        return _scalar_str(dtype, short=short)  # 返回标量数据类型的字符串表示

# 构造标量数据类型的字符串表示形式
def _scalar_str(dtype, short):
    byteorder = _byte_order_str(dtype)
    # 如果数据类型是布尔型(np.bool)
    if dtype.type == np.bool:
        # 如果需要简短表示(short为True)，返回问号字符'?'作为表示
        if short:
            return "'?'"
        else:
            # 否则返回完整表示'bool'
            return "'bool'"

    # 如果数据类型是对象(np.object_)
    elif dtype.type == np.object_:
        # 对象引用在不同平台可能有不同大小，所以此处不应包含itemsize大小信息
        return "'O'"

    # 如果数据类型是字节串(np.bytes_)
    elif dtype.type == np.bytes_:
        # 如果是无大小限制的字节串，返回'S'
        if _isunsized(dtype):
            return "'S'"
        else:
            # 否则返回'S%d'，其中%d是字节串的itemsize大小
            return "'S%d'" % dtype.itemsize

    # 如果数据类型是字符串(np.str_)
    elif dtype.type == np.str_:
        # 如果是无大小限制的字符串，返回'字节序U'格式的字符串
        if _isunsized(dtype):
            return "'%sU'" % byteorder
        else:
            # 否则返回'字节序U%d'，其中%d是字符串的itemsize大小除以4的结果
            return "'%sU%d'" % (byteorder, dtype.itemsize / 4)

    # 如果数据类型是普通字符串(str)
    elif dtype.type == str:
        return "'T'"

    # 如果数据类型不是传统的数据类型
    elif not type(dtype)._legacy:
        # 返回形如'字节序类名大小位'的字符串表示
        return f"'{byteorder}{type(dtype).__name__}{dtype.itemsize * 8}'"

    # 对于void的子类，保持原样，但是历史上的repr实际上并不显示子类
    elif issubclass(dtype.type, np.void):
        # 如果是无大小限制的void类型，返回'V'
        if _isunsized(dtype):
            return "'V'"
        else:
            # 否则返回'V%d'，其中%d是void类型的itemsize大小
            return "'V%d'" % dtype.itemsize

    # 如果数据类型是日期时间类型(np.datetime64)
    elif dtype.type == np.datetime64:
        # 返回形如'字节序M8元数据'的日期时间字符串表示
        return "'%sM8%s'" % (byteorder, _datetime_metadata_str(dtype))

    # 如果数据类型是时间间隔类型(np.timedelta64)
    elif dtype.type == np.timedelta64:
        # 返回形如'字节序m8元数据'的时间间隔字符串表示
        return "'%sm8%s'" % (byteorder, _datetime_metadata_str(dtype))

    # 如果数据类型是数值类型的子类型(np.number)
    elif np.issubdtype(dtype, np.number):
        # 如果需要简短表示(short为True)，返回形如'字节序类型大小'的简短字符串表示
        if short or dtype.byteorder not in ('=', '|'):
            return "'%s%c%d'" % (byteorder, dtype.kind, dtype.itemsize)
        else:
            # 否则返回形如'类型名大小'的长字符串表示
            return "'%s%d'" % (_kind_name(dtype), 8*dtype.itemsize)

    # 如果数据类型是内置类型的标志(dtype.isbuiltin == 2)
    elif dtype.isbuiltin == 2:
        # 返回该内置类型的名称
        return dtype.type.__name__

    # 如果以上条件都不满足，抛出运行时错误
    else:
        raise RuntimeError(
            "Internal error: NumPy dtype unrecognized type number")
# 将数据类型的字节顺序规范化为 '<' 或 '>'
def _byte_order_str(dtype):
    # 创建一个新的数据类型对象，其字节顺序被交换
    swapped = np.dtype(int).newbyteorder('S')
    # 获取原生数据类型对象
    native = swapped.newbyteorder('S')

    # 获取当前数据类型的字节顺序
    byteorder = dtype.byteorder
    # 如果字节顺序为 '='，返回原生数据类型的字节顺序
    if byteorder == '=':
        return native.byteorder
    # 如果字节顺序为 'S'，这条路径永远不会被执行
    if byteorder == 'S':
        # TODO: this path can never be reached
        return swapped.byteorder
    # 如果字节顺序为 '|'，返回空字符串
    elif byteorder == '|':
        return ''
    else:
        # 否则返回当前字节顺序
        return byteorder


def _datetime_metadata_str(dtype):
    # TODO: this duplicates the C metastr_to_unicode functionality
    # 获取日期时间数据类型的单位和计数
    unit, count = np.datetime_data(dtype)
    # 如果单位为 'generic'，返回空字符串
    if unit == 'generic':
        return ''
    # 如果计数为 1，返回格式化后的单位字符串，例如 '[unit]'
    elif count == 1:
        return '[{}]'.format(unit)
    # 否则返回格式化后的计数和单位字符串，例如 '[countunit]'
    else:
        return '[{}{}]'.format(count, unit)


def _struct_dict_str(dtype, includealignedflag):
    # 将字段名解包到列表 ls 中
    names = dtype.names
    fld_dtypes = []
    offsets = []
    titles = []
    # 遍历字段名列表
    for name in names:
        # 解包字段信息
        fld_dtype, offset, title = _unpack_field(*dtype.fields[name])
        fld_dtypes.append(fld_dtype)
        offsets.append(offset)
        titles.append(title)

    # 构建字典字符串

    # 根据打印模式确定冒号和字段分隔符
    if np._core.arrayprint._get_legacy_print_mode() <= 121:
        colon = ":"
        fieldsep = ","
    else:
        colon = ": "
        fieldsep = ", "

    # 首先添加字段名
    ret = "{'names'%s[" % colon
    ret += fieldsep.join(repr(name) for name in names)

    # 其次添加格式
    ret += "], 'formats'%s[" % colon
    ret += fieldsep.join(
        _construction_repr(fld_dtype, short=True) for fld_dtype in fld_dtypes)

    # 然后添加偏移量
    ret += "], 'offsets'%s[" % colon
    ret += fieldsep.join("%d" % offset for offset in offsets)

    # 如果存在标题，则添加标题信息
    if any(title is not None for title in titles):
        ret += "], 'titles'%s[" % colon
        ret += fieldsep.join(repr(title) for title in titles)

    # 最后添加项目大小信息
    ret += "], 'itemsize'%s%d" % (colon, dtype.itemsize)

    # 如果包含对齐标志并且数据类型是对齐的结构体，则添加对齐标志
    if (includealignedflag and dtype.isalignedstruct):
        ret += ", 'aligned'%sTrue}" % colon
    else:
        ret += "}"

    return ret


def _aligned_offset(offset, alignment):
    # 将偏移量向上舍入到最接近的对齐边界
    return - (-offset // alignment) * alignment


def _is_packed(dtype):
    """
    检查结构化数据类型 'dtype' 是否具有简单的布局，即所有字段按顺序排列，
    且没有额外的对齐填充。
    
    当返回 True 时，可以从字段名和数据类型的列表重建数据类型，而不需要额外的参数。

    复制了 C 中的 `is_dtype_struct_simple_unaligned_layout` 函数。
    """
    # 获取结构体的对齐属性
    align = dtype.isalignedstruct
    max_alignment = 1
    total_offset = 0
    # 遍历结构体类型(dtype)的每个字段名
    for name in dtype.names:
        # 解包字段信息，获取字段类型、偏移量和标题
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])

        # 如果需要按照字段对齐要求调整偏移量
        if align:
            total_offset = _aligned_offset(total_offset, fld_dtype.alignment)
            # 更新最大对齐值
            max_alignment = max(max_alignment, fld_dtype.alignment)

        # 检查字段的偏移量是否与累计偏移量相等
        if fld_offset != total_offset:
            return False
        
        # 更新累计偏移量以包括当前字段的大小
        total_offset += fld_dtype.itemsize

    # 如果需要按照字段对齐要求调整最终的总偏移量
    if align:
        total_offset = _aligned_offset(total_offset, max_alignment)

    # 检查最终累计偏移量是否与结构体类型的总大小相等
    if total_offset != dtype.itemsize:
        return False
    
    # 结构体布局验证通过，返回True
    return True
def _struct_list_str(dtype):
    items = []
    # 遍历结构化数据类型中的字段名
    for name in dtype.names:
        # 解包字段元组
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])

        item = "("
        # 如果存在标题，则格式化为特定字符串
        if title is not None:
            item += "({!r}, {!r}), ".format(title, name)
        else:
            item += "{!r}, ".format(name)
        # 处理特殊情况下的子数组
        if fld_dtype.subdtype is not None:
            base, shape = fld_dtype.subdtype
            item += "{}, {}".format(
                _construction_repr(base, short=True),
                shape
            )
        else:
            item += _construction_repr(fld_dtype, short=True)

        item += ")"
        # 将格式化后的字段信息添加到列表中
        items.append(item)

    # 返回包含所有字段信息的字符串列表
    return "[" + ", ".join(items) + "]"


def _struct_str(dtype, include_align):
    # 如果不需要包含对齐信息并且结构体是紧凑的，使用列表形式的字符串表示
    if not (include_align and dtype.isalignedstruct) and _is_packed(dtype):
        sub = _struct_list_str(dtype)
    else:
        # 否则使用字典形式的字符串表示
        sub = _struct_dict_str(dtype, include_align)

    # 如果数据类型不是默认的 void 类型，则返回带有模块名和数据类型的字符串表示
    if dtype.type != np.void:
        return "({t.__module__}.{t.__name__}, {f})".format(t=dtype.type, f=sub)
    else:
        # 否则返回结构体的字符串表示
        return sub


def _subarray_str(dtype):
    # 获取子数组的字符串表示，包括基础类型和形状信息
    base, shape = dtype.subdtype
    return "({}, {})".format(
        _construction_repr(base, short=True),
        shape
    )


def _name_includes_bit_suffix(dtype):
    if dtype.type == np.object_:
        # 对象类型不包含位后缀
        return False
    elif dtype.type == np.bool:
        # 布尔类型不包含位后缀
        return False
    elif dtype.type is None:
        # 未知类型包含位后缀
        return True
    elif np.issubdtype(dtype, np.flexible) and _isunsized(dtype):
        # 未指定类型不包含位后缀
        return False
    else:
        # 其他类型包含位后缀
        return True


def _name_get(dtype):
    # 返回数据类型的名称，考虑到是否包含位后缀和日期时间元数据
    if dtype.isbuiltin == 2:
        # 用户自定义数据类型返回其类型名称
        return dtype.type.__name__

    if not type(dtype)._legacy:
        name = type(dtype).__name__

    elif issubclass(dtype.type, np.void):
        # void 类型保留其名称，如 `record64`
        name = dtype.type.__name__
    else:
        name = _kind_name(dtype)

    # 如果需要，添加位数信息
    if _name_includes_bit_suffix(dtype):
        name += "{}".format(dtype.itemsize * 8)

    # 如果是日期时间类型，添加日期时间元数据
    if dtype.type in (np.datetime64, np.timedelta64):
        name += _datetime_metadata_str(dtype)

    return name
```