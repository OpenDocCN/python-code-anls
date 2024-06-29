# `.\numpy\numpy\lib\recfunctions.py`

```
"""
Collection of utilities to manipulate structured arrays.

Most of these functions were initially implemented by John Hunter for
matplotlib.  They have been rewritten and extended for convenience.

"""
import itertools  # 导入 itertools 库，用于高效的迭代工具
import numpy as np  # 导入 NumPy 库，用于数组操作
import numpy.ma as ma  # 导入 NumPy 的 masked array 模块，用于处理带掩码的数组
from numpy import ndarray  # 导入 NumPy 的 ndarray 类型
from numpy.ma import MaskedArray  # 导入 NumPy 的 MaskedArray 类型
from numpy.ma.mrecords import MaskedRecords  # 导入 NumPy 的 MaskedRecords 类型
from numpy._core.overrides import array_function_dispatch  # 导入 NumPy 的函数调度装饰器
from numpy._core.records import recarray  # 导入 NumPy 的 recarray 类型
from numpy.lib._iotools import _is_string_like  # 导入 NumPy 的内部 I/O 工具函数

_check_fill_value = np.ma.core._check_fill_value  # 设置变量 _check_fill_value 为 NumPy 中 masked array 的填充值检查函数


__all__ = [
    'append_fields', 'apply_along_fields', 'assign_fields_by_name',
    'drop_fields', 'find_duplicates', 'flatten_descr',
    'get_fieldstructure', 'get_names', 'get_names_flat',
    'join_by', 'merge_arrays', 'rec_append_fields',
    'rec_drop_fields', 'rec_join', 'recursive_fill_fields',
    'rename_fields', 'repack_fields', 'require_fields',
    'stack_arrays', 'structured_to_unstructured', 'unstructured_to_structured',
    ]


def _recursive_fill_fields_dispatcher(input, output):
    return (input, output)  # 返回 input 和 output 作为元组


@array_function_dispatch(_recursive_fill_fields_dispatcher)
def recursive_fill_fields(input, output):
    """
    Fills fields from output with fields from input,
    with support for nested structures.

    Parameters
    ----------
    input : ndarray
        Input array.
    output : ndarray
        Output array.

    Notes
    -----
    * `output` should be at least the same size as `input`

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.array([(1, 10.), (2, 20.)], dtype=[('A', np.int64), ('B', np.float64)])
    >>> b = np.zeros((3,), dtype=a.dtype)
    >>> rfn.recursive_fill_fields(a, b)
    array([(1, 10.), (2, 20.), (0,  0.)], dtype=[('A', '<i8'), ('B', '<f8')])

    """
    newdtype = output.dtype  # 获取输出数组的数据类型
    for field in newdtype.names:  # 遍历输出数组的字段名
        try:
            current = input[field]  # 尝试从输入数组中获取当前字段的数据
        except ValueError:
            continue  # 如果出错，继续下一个字段
        if current.dtype.names is not None:  # 如果当前字段是结构化数组
            recursive_fill_fields(current, output[field])  # 递归地填充子结构的字段
        else:
            output[field][:len(current)] = current  # 否则直接将当前字段的数据填充到输出数组的对应字段中
    return output  # 返回填充好数据的输出数组


def _get_fieldspec(dtype):
    """
    Produce a list of name/dtype pairs corresponding to the dtype fields

    Similar to dtype.descr, but the second item of each tuple is a dtype, not a
    string. As a result, this handles subarray dtypes

    Can be passed to the dtype constructor to reconstruct the dtype, noting that
    this (deliberately) discards field offsets.

    Examples
    --------
    >>> dt = np.dtype([(('a', 'A'), np.int64), ('b', np.double, 3)])
    >>> dt.descr
    [(('a', 'A'), '<i8'), ('b', '<f8', (3,))]
    >>> _get_fieldspec(dt)
    [(('a', 'A'), dtype('int64')), ('b', dtype(('<f8', (3,))))]

    """
    if dtype.names is None:
        # .descr returns a nameless field, so we should too
        return [('', dtype)]  # 如果 dtype 没有字段名，返回一个名字为空的字段和其类型的列表
    else:
        # 为每个字段生成一个生成器，返回字段名及其描述信息
        fields = ((name, dtype.fields[name]) for name in dtype.names)
        # 如果字段有标题，保留标题；否则使用字段名作为标题，并返回字段的数据类型
        return [
            (name if len(f) == 2 else (f[2], name), f[0])
            for name, f in fields
        ]
# 返回给定数据类型的字段名作为元组。如果输入的数据类型没有字段，则会引发错误。

def get_names(adtype):
    """
    Returns the field names of the input datatype as a tuple. Input datatype
    must have fields otherwise error is raised.

    Parameters
    ----------
    adtype : dtype
        Input datatype

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> rfn.get_names(np.empty((1,), dtype=[('A', int)]).dtype)
    ('A',)
    >>> rfn.get_names(np.empty((1,), dtype=[('A',int), ('B', float)]).dtype)
    ('A', 'B')
    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
    >>> rfn.get_names(adtype)
    ('a', ('b', ('ba', 'bb')))
    """
    listnames = []  # 初始化一个空列表，用于存储字段名
    names = adtype.names  # 获取输入数据类型的字段名列表
    for name in names:  # 遍历每个字段名
        current = adtype[name]  # 获取当前字段的类型描述
        if current.names is not None:  # 如果当前字段是结构化类型
            listnames.append((name, tuple(get_names(current))))  # 递归获取子字段名并添加到列表
        else:  # 如果当前字段不是结构化类型
            listnames.append(name)  # 直接将字段名添加到列表
    return tuple(listnames)  # 返回存储字段名的元组


# 返回扁平化后的给定数据类型的字段名作为元组。如果输入的数据类型没有字段，则会引发错误。

def get_names_flat(adtype):
    """
    Returns the field names of the input datatype as a tuple. Input datatype
    must have fields otherwise error is raised.
    Nested structure are flattened beforehand.

    Parameters
    ----------
    adtype : dtype
        Input datatype

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> rfn.get_names_flat(np.empty((1,), dtype=[('A', int)]).dtype) is None
    False
    >>> rfn.get_names_flat(np.empty((1,), dtype=[('A',int), ('B', str)]).dtype)
    ('A', 'B')
    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
    >>> rfn.get_names_flat(adtype)
    ('a', 'b', 'ba', 'bb')
    """
    listnames = []  # 初始化一个空列表，用于存储字段名
    names = adtype.names  # 获取输入数据类型的字段名列表
    for name in names:  # 遍历每个字段名
        listnames.append(name)  # 将字段名添加到列表
        current = adtype[name]  # 获取当前字段的类型描述
        if current.names is not None:  # 如果当前字段是结构化类型
            listnames.extend(get_names_flat(current))  # 递归获取扁平化后的子字段名并扩展到列表
    return tuple(listnames)  # 返回存储字段名的元组


# 将结构化数据类型描述扁平化成一个元组的形式，用于描述字段和其类型。

def flatten_descr(ndtype):
    """
    Flatten a structured data-type description.

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> ndtype = np.dtype([('a', '<i4'), ('b', [('ba', '<f8'), ('bb', '<i4')])])
    >>> rfn.flatten_descr(ndtype)
    (('a', dtype('int32')), ('ba', dtype('float64')), ('bb', dtype('int32')))

    """
    names = ndtype.names  # 获取结构化数据类型的字段名列表
    if names is None:  # 如果字段名列表为空
        return (('', ndtype),)  # 返回一个元组，包含一个空字符串和数据类型描述
    else:
        descr = []  # 初始化一个空列表，用于存储扁平化后的描述
        for field in names:  # 遍历每个字段名
            (typ, _) = ndtype.fields[field]  # 获取字段的类型描述
            if typ.names is not None:  # 如果字段的类型是结构化类型
                descr.extend(flatten_descr(typ))  # 递归扁平化子结构类型描述并扩展到列表
            else:  # 如果字段的类型不是结构化类型
                descr.append((field, typ))  # 直接添加字段名和类型描述到列表
        return tuple(descr)  # 返回存储字段名和类型描述的元组


# 将序列数组的数据类型描述扁平化为一个列表，用于描述字段和其类型。

def _zip_dtype(seqarrays, flatten=False):
    newdtype = []  # 初始化一个空列表，用于存储扁平化后的数据类型描述
    if flatten:  # 如果指定了扁平化标志
        for a in seqarrays:  # 遍历每个序列数组
            newdtype.extend(flatten_descr(a.dtype))  # 扁平化数据类型描述并扩展到列表
    else:
        # 遍历 seqarrays 中的每个数组
        for a in seqarrays:
            # 获取当前数组的数据类型
            current = a.dtype
            # 检查当前数据类型是否有字段名，并且只有一个字段
            if current.names is not None and len(current.names) == 1:
                # 特殊情况 - 只有一个字段的数据类型会被展平
                newdtype.extend(_get_fieldspec(current))
            else:
                # 将当前数据类型作为一个元组添加到 newdtype 中
                newdtype.append(('', current))
    # 返回新创建的数据类型对象
    return np.dtype(newdtype)
# 返回多个数组的字段描述合并后的数据类型描述符
def _zip_descr(seqarrays, flatten=False):
    # 调用 _zip_dtype 函数获取数据类型描述符，根据 flatten 参数决定是否展开嵌套描述
    return _zip_dtype(seqarrays, flatten=flatten).descr


# 返回包含字段索引其父字段列表的字典
def get_fieldstructure(adtype, lastname=None, parents=None,):
    # 如果 parents 为空，初始化为一个空字典
    if parents is None:
        parents = {}
    # 获取 adtype 的字段名列表
    names = adtype.names
    # 遍历每个字段名
    for name in names:
        # 获取当前字段的数据类型
        current = adtype[name]
        # 如果当前字段还有子字段
        if current.names is not None:
            # 如果有上一个处理过的字段名 lastname，将当前字段名添加到其父字段列表中
            if lastname:
                parents[name] = [lastname, ]
            else:
                parents[name] = []
            # 递归调用 get_fieldstructure 处理当前字段的子类型
            parents.update(get_fieldstructure(current, name, parents))
        else:
            # 获取上一个处理过的父字段列表
            lastparent = [_ for _ in (parents.get(lastname, []) or [])]
            # 如果存在上一个父字段，将当前字段名添加到其父字段列表中
            if lastparent:
                lastparent.append(lastname)
            elif lastname:
                lastparent = [lastname, ]
            # 更新当前字段的父字段列表
            parents[name] = lastparent or []
    # 返回所有字段的父字段字典
    return parents


# 返回一个展平后的字段迭代器，从一个数组序列中获取字段
def _izip_fields_flat(iterable):
    # 遍历可迭代对象中的每一个元素
    for element in iterable:
        # 如果元素是一个 NumPy 的 void 类型
        if isinstance(element, np.void):
            # 递归展开该 void 类型的元组
            yield from _izip_fields_flat(tuple(element))
        else:
            yield element


# 返回一个字段迭代器，从一个数组序列中获取字段
def _izip_fields(iterable):
    # 遍历可迭代对象中的每一个元素
    for element in iterable:
        # 如果元素是可迭代的，并且不是字符串类型
        if (hasattr(element, '__iter__') and
                not isinstance(element, str)):
            # 递归调用 _izip_fields 函数处理嵌套结构的元素
            yield from _izip_fields(element)
        elif isinstance(element, np.void) and len(tuple(element)) == 1:
            # 如果元素是 NumPy 的 void 类型且长度为1
            # 与上一个表达式相同的语句
            yield from _izip_fields(element)
        else:
            yield element


# 返回一个记录迭代器，从一个数组序列中获取连接的项目
def _izip_records(seqarrays, fill_value=None, flatten=True):
    """
    Returns an iterator of concatenated items from a sequence of arrays.

    Parameters
    ----------
    seqarrays : sequence of arrays
        The sequence of arrays to iterate over.
    fill_value : any, optional
        The value to use for missing fields in records.
    flatten : bool, optional
        Whether to flatten nested structures.

    """
    # 这个函数的余下部分在下一个代码块中继续注释
    # seqarrays: 序列的数组
    #         输入的多个数组构成的序列。
    # fill_value: {None, integer}
    #         用于填充较短可迭代对象的值。
    # flatten: {True, False}
    #         是否展平项目，或者采用嵌套方式。

    # 根据 flatten 参数选择合适的 zip 函数
    if flatten:
        zipfunc = _izip_fields_flat  # 如果 flatten 为 True，使用展平版本的 zip 函数
    else:
        zipfunc = _izip_fields       # 如果 flatten 为 False，使用嵌套版本的 zip 函数

    # 使用 itertools.zip_longest 函数将 seqarrays 中的数组按照 fill_value 扩展为等长元组
    for tup in itertools.zip_longest(*seqarrays, fillvalue=fill_value):
        # 调用选定的 zipfunc 函数来处理每个元组，并使用 yield 生成器返回结果
        yield tuple(zipfunc(tup))
# 定义一个私有函数，根据输入参数返回 recarray、ndarray、MaskedArray 或 MaskedRecords
def _fix_output(output, usemask=True, asrecarray=False):
    # 如果 output 不是 MaskedArray 类型，则不使用掩码
    if not isinstance(output, MaskedArray):
        usemask = False
    # 如果使用了掩码
    if usemask:
        # 如果需要返回 MaskedRecords，则将 output 转换为 MaskedRecords 类型
        if asrecarray:
            output = output.view(MaskedRecords)
    else:
        # 如果不使用掩码，则将 output 转换为填充后的数组
        output = ma.filled(output)
        # 如果需要返回 recarray，则将 output 转换为 recarray 类型
        if asrecarray:
            output = output.view(recarray)
    # 返回处理后的 output
    return output

# 更新 output 的默认填充值和掩码数据，根据默认值字典 defaults
def _fix_defaults(output, defaults=None):
    # 获取 output 的列名
    names = output.dtype.names
    # 分别获取 output 的数据、掩码和填充值
    (data, mask, fill_value) = (output.data, output.mask, output.fill_value)
    # 遍历默认值字典 defaults，更新 fill_value 和 data
    for (k, v) in (defaults or {}).items():
        # 如果列名在 names 中，则更新 fill_value 和 data
        if k in names:
            fill_value[k] = v
            data[k][mask[k]] = v
    # 返回更新后的 output
    return output

# 分发器函数，用于选择使用哪个函数来合并数组
def _merge_arrays_dispatcher(seqarrays, fill_value=None, flatten=None, usemask=None, asrecarray=None):
    return seqarrays

# merge_arrays 函数，逐列合并数组
@array_function_dispatch(_merge_arrays_dispatcher)
def merge_arrays(seqarrays, fill_value=-1, flatten=False, usemask=False, asrecarray=False):
    """
    Merge arrays field by field.

    Parameters
    ----------
    seqarrays : sequence of ndarrays
        Sequence of arrays
    fill_value : {float}, optional
        Filling value used to pad missing data on the shorter arrays.
    flatten : {False, True}, optional
        Whether to collapse nested fields.
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : {False, True}, optional
        Whether to return a recarray (MaskedRecords) or not.

    Examples
    --------
    ... (示例代码，略)

    Notes
    -----
    * Without a mask, the missing value will be filled with something,
      depending on what its corresponding type:

      * ``-1``      for integers
      * ``-1.0``    for floating point numbers
      * ``'-'``     for characters
      * ``'-1'``    for strings
      * ``True``    for boolean values
    * XXX: I just obtained these values empirically
    """
    # 只有一个条目在输入序列中？
    if (len(seqarrays) == 1):
        # 将 seqarrays 转换为数组
        seqarrays = np.asanyarray(seqarrays[0])
    # 判断输入的 seqarrays 是否为单个 ndarray 或 np.void 类型
    if isinstance(seqarrays, (ndarray, np.void)):
        # 获取 seqarrays 的数据类型
        seqdtype = seqarrays.dtype
        # 确保 seqdtype 有命名字段
        if seqdtype.names is None:
            seqdtype = np.dtype([('', seqdtype)])
        # 如果不需要扁平化或者扁平化后的数据类型与 seqdtype 相同，则执行最小处理
        if not flatten or _zip_dtype((seqarrays,), flatten=True) == seqdtype:
            # 将 seqarrays 扁平化处理
            seqarrays = seqarrays.ravel()
            # 确定返回的数组类型
            if usemask:
                if asrecarray:
                    seqtype = MaskedRecords
                else:
                    seqtype = MaskedArray
            elif asrecarray:
                seqtype = recarray
            else:
                seqtype = ndarray
            # 返回视图，使用指定的 dtype 和 type
            return seqarrays.view(dtype=seqdtype, type=seqtype)
        else:
            # 如果类型不匹配，则将 seqarrays 转换为元组
            seqarrays = (seqarrays,)
    else:
        # 确保输入序列中的每个元素都是数组，并将其转换为 asanyarray
        seqarrays = [np.asanyarray(_m) for _m in seqarrays]

    # 获取输入序列中每个数组的大小，并找出最大的大小
    sizes = tuple(a.size for a in seqarrays)
    maxlength = max(sizes)

    # 获取输出的 dtype（如果需要扁平化则进行扁平化处理）
    newdtype = _zip_dtype(seqarrays, flatten=flatten)

    # 初始化数据和掩码的序列
    seqdata = []
    seqmask = []

    # 如果需要 MaskedArray，进行特殊处理的循环
    if usemask:
        for (a, n) in zip(seqarrays, sizes):
            # 计算缺失的数据量
            nbmissing = (maxlength - n)
            # 获取数据和掩码
            data = a.ravel().__array__()
            mask = ma.getmaskarray(a).ravel()

            # 如果存在缺失数据，获取填充值
            if nbmissing:
                fval = _check_fill_value(fill_value, a.dtype)
                if isinstance(fval, (ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                        fmsk = True
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
                        fmsk = np.ones((1,), dtype=mask.dtype)
                else:
                    fval = None
                    fmsk = True

            # 将填充后的数据作为迭代器存储到 seqdata 和 seqmask 中
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
            seqmask.append(itertools.chain(mask, [fmsk] * nbmissing))

        # 创建数据的迭代器
        data = tuple(_izip_records(seqdata, flatten=flatten))
        # 创建 MaskedArray
        output = ma.array(np.fromiter(data, dtype=newdtype, count=maxlength),
                          mask=list(_izip_records(seqmask, flatten=flatten)))
        # 如果需要返回的是 MaskedRecords，则转换为 MaskedRecords
        if asrecarray:
            output = output.view(MaskedRecords)
    else:
        # Same as before, without the mask we don't need...
        # 对于没有掩码的情况，与之前相同的处理流程

        for (a, n) in zip(seqarrays, sizes):
            # 计算需要填充的数量
            nbmissing = (maxlength - n)
            
            # 将数组展平并转换为数组
            data = a.ravel().__array__()

            if nbmissing:
                # 检查并获取填充值
                fval = _check_fill_value(fill_value, a.dtype)
                
                if isinstance(fval, (ndarray, np.void)):
                    if len(fval.dtype) == 1:
                        fval = fval.item()[0]
                    else:
                        fval = np.array(fval, dtype=a.dtype, ndmin=1)
                else:
                    fval = None
            else:
                fval = None
            
            # 将数据与填充值的重复链表附加到序列数据中
            seqdata.append(itertools.chain(data, [fval] * nbmissing))
        
        # 从迭代器中创建新的 NumPy 数组
        output = np.fromiter(tuple(_izip_records(seqdata, flatten=flatten)),
                             dtype=newdtype, count=maxlength)
        
        if asrecarray:
            # 如果需要返回记录数组，则转换输出为记录数组
            output = output.view(recarray)
    
    # 处理完成，返回输出结果
    return output
# 返回一个分派器，它始终返回包含 base 的元组
def _drop_fields_dispatcher(base, drop_names, usemask=None, asrecarray=None):
    return (base,)

# 使用 array_function_dispatch 装饰器注册的函数，根据输入的参数来决定分派给特定函数
@array_function_dispatch(_drop_fields_dispatcher)
def drop_fields(base, drop_names, usemask=True, asrecarray=False):
    """
    返回一个删除了指定字段的新数组。

    支持嵌套字段。

    .. versionchanged:: 1.18.0
        如果删除了所有字段，则返回一个字段数为 0 的数组，而不是像之前一样返回 ``None``。

    Parameters
    ----------
    base : array
        输入的数组
    drop_names : string 或 sequence
        要删除的字段名或字段名列表。
    usemask : {False, True}, optional
        是否返回掩码数组。
    asrecarray : string 或 sequence, optional
        是否返回 recarray 或 mrecarray (`asrecarray=True`)，或者是一个普通的 ndarray 或带有灵活 dtype 的掩码数组。默认为 False。

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
    ...   dtype=[('a', np.int64), ('b', [('ba', np.double), ('bb', np.int64)])])
    >>> rfn.drop_fields(a, 'a')
    array([((2., 3),), ((5., 6),)],
          dtype=[('b', [('ba', '<f8'), ('bb', '<i8')])])
    >>> rfn.drop_fields(a, 'ba')
    array([(1, (3,)), (4, (6,))], dtype=[('a', '<i8'), ('b', [('bb', '<i8')])])
    >>> rfn.drop_fields(a, ['ba', 'bb'])
    array([(1,), (4,)], dtype=[('a', '<i8')])
    """
    # 如果 drop_names 是字符串，转换为列表
    if _is_string_like(drop_names):
        drop_names = [drop_names]
    else:
        drop_names = set(drop_names)

    # 定义递归函数来处理数据类型描述符
    def _drop_descr(ndtype, drop_names):
        names = ndtype.names
        newdtype = []
        for name in names:
            current = ndtype[name]
            # 如果当前字段名在 drop_names 中，则跳过
            if name in drop_names:
                continue
            if current.names is not None:
                # 如果当前字段是结构化类型，则递归处理
                descr = _drop_descr(current, drop_names)
                if descr:
                    newdtype.append((name, descr))
            else:
                # 否则将当前字段添加到新数据类型描述符中
                newdtype.append((name, current))
        return newdtype

    # 使用 _drop_descr 函数生成新的数据类型描述符
    newdtype = _drop_descr(base.dtype, drop_names)

    # 创建一个空数组来存储结果
    output = np.empty(base.shape, dtype=newdtype)
    # 使用递归填充字段值到输出数组中
    output = recursive_fill_fields(base, output)
    # 根据参数设置修正输出结果
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)


def _keep_fields(base, keep_names, usemask=True, asrecarray=False):
    """
    返回一个仅包含指定字段并保持这些字段顺序的新数组。

    Parameters
    ----------
    base : array
        输入的数组
    keep_names : string 或 sequence
        要保留的字段名或字段名列表。保持的顺序将与输入中的顺序一致。
    usemask : {False, True}, optional
        是否返回掩码数组。
    """
    # 定义一个参数 `asrecarray`，可以是字符串或序列，控制返回类型为 recarray 或 mrecarray
    # (`asrecarray=True`)，或者是带有灵活数据类型的普通 ndarray 或 masked array。默认为 False。
    newdtype = [(n, base.dtype[n]) for n in keep_names]
    # 创建一个空的 ndarray，使用指定的数据类型 `newdtype`，形状与 `base` 相同
    output = np.empty(base.shape, dtype=newdtype)
    # 调用函数 `recursive_fill_fields`，将 `base` 填充到创建的 `output` 中
    output = recursive_fill_fields(base, output)
    # 返回修正后的输出，根据参数 `usemask` 和 `asrecarray` 进行调整
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)
# 定义一个调度器函数 _rec_drop_fields_dispatcher，用于分发删除字段操作的基础函数和参数
def _rec_drop_fields_dispatcher(base, drop_names):
    # 返回一个包含 base 的元组
    return (base,)

# 使用 array_function_dispatch 装饰器，将 _rec_drop_fields_dispatcher 函数与 rec_drop_fields 绑定
@array_function_dispatch(_rec_drop_fields_dispatcher)
def rec_drop_fields(base, drop_names):
    """
    Returns a new numpy.recarray with fields in `drop_names` dropped.
    """
    # 调用 drop_fields 函数，从 base 中删除指定的字段名 drop_names，返回一个新的 recarray 对象
    return drop_fields(base, drop_names, usemask=False, asrecarray=True)


# 定义一个调度器函数 _rename_fields_dispatcher，用于分发重命名字段操作的基础函数和参数
def _rename_fields_dispatcher(base, namemapper):
    # 返回一个包含 base 的元组
    return (base,)

# 使用 array_function_dispatch 装饰器，将 _rename_fields_dispatcher 函数与 rename_fields 绑定
@array_function_dispatch(_rename_fields_dispatcher)
def rename_fields(base, namemapper):
    """
    Rename the fields from a flexible-datatype ndarray or recarray.

    Nested fields are supported.

    Parameters
    ----------
    base : ndarray
        Input array whose fields must be modified.
    namemapper : dictionary
        Dictionary mapping old field names to their new version.

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
    ...   dtype=[('a', int),('b', [('ba', float), ('bb', (float, 2))])])
    >>> rfn.rename_fields(a, {'a':'A', 'bb':'BB'})
    array([(1, (2., [ 3., 30.])), (4, (5., [ 6., 60.]))],
          dtype=[('A', '<i8'), ('b', [('ba', '<f8'), ('BB', '<f8', (2,))])])

    """
    # 定义一个递归函数 _recursive_rename_fields，用于迭代修改字段名
    def _recursive_rename_fields(ndtype, namemapper):
        newdtype = []
        # 遍历当前 dtype 的所有字段名
        for name in ndtype.names:
            # 根据 namemapper 映射新的字段名，如果不存在映射，则保留原名
            newname = namemapper.get(name, name)
            current = ndtype[name]
            # 如果当前字段是复合类型（含有子字段），递归调用 _recursive_rename_fields 处理
            if current.names is not None:
                newdtype.append(
                    (newname, _recursive_rename_fields(current, namemapper))
                    )
            else:
                newdtype.append((newname, current))
        return newdtype
    
    # 根据 namemapper 对 base 的 dtype 进行递归重命名字段
    newdtype = _recursive_rename_fields(base.dtype, namemapper)
    # 返回一个视图，使得 base 按照新的 dtype 进行重命名后的结果
    return base.view(newdtype)


# 定义一个调度器函数 _append_fields_dispatcher，用于分发向现有数组添加字段操作的基础函数和参数
def _append_fields_dispatcher(base, names, data, dtypes=None,
                              fill_value=None, usemask=None, asrecarray=None):
    # 返回一个包含 base 和 data 中所有元素的生成器
    yield base
    yield from data

# 使用 array_function_dispatch 装饰器，将 _append_fields_dispatcher 函数与 append_fields 绑定
@array_function_dispatch(_append_fields_dispatcher)
def append_fields(base, names, data, dtypes=None,
                  fill_value=-1, usemask=True, asrecarray=False):
    """
    Add new fields to an existing array.

    The names of the fields are given with the `names` arguments,
    the corresponding values with the `data` arguments.
    If a single field is appended, `names`, `data` and `dtypes` do not have
    to be lists but just values.

    Parameters
    ----------
    base : array
        Input array to extend.
    names : string, sequence
        String or sequence of strings corresponding to the names
        of the new fields.
    data : array or sequence of arrays
        Array or sequence of arrays storing the fields to add to the base.
    dtypes : sequence of datatypes, optional
        Datatype or sequence of datatypes.
        If None, the datatypes are estimated from the `data`.
    fill_value : {float}, optional
        Filling value used to pad missing data on the shorter arrays.

    """
    # 添加新字段到现有数组 base 中
    # 如果 names, data, dtypes 只是单一值而不是列表，则添加单一字段
    # 返回扩展后的数组
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : {False, True}, optional
        Whether to return a recarray (MaskedRecords) or not.

    """
    # 检查传入的字段名是否合法
    if isinstance(names, (tuple, list)):
        # 如果传入的字段名是一个元组或列表
        if len(names) != len(data):
            # 如果字段名的数量与数据数组的数量不匹配，抛出数值错误异常
            msg = "The number of arrays does not match the number of names"
            raise ValueError(msg)
    elif isinstance(names, str):
        # 如果传入的字段名是字符串，则将其转换为单元素列表
        names = [names, ]
        data = [data, ]
    #
    if dtypes is None:
        # 如果未指定数据类型
        data = [np.array(a, copy=None, subok=True) for a in data]
        # 将每个数据数组转换为 NumPy 数组
        data = [a.view([(name, a.dtype)]) for (name, a) in zip(names, data)]
    else:
        # 如果指定了数据类型
        if not isinstance(dtypes, (tuple, list)):
            # 如果数据类型不是元组或列表，则转换为单元素列表
            dtypes = [dtypes, ]
        if len(data) != len(dtypes):
            # 如果数据数组数量与数据类型数量不匹配
            if len(dtypes) == 1:
                # 如果数据类型数量为1，则复制以匹配数据数组数量
                dtypes = dtypes * len(data)
            else:
                msg = "The dtypes argument must be None, a dtype, or a list."
                raise ValueError(msg)
        data = [np.array(a, copy=None, subok=True, dtype=d).view([(n, d)])
                for (a, n, d) in zip(data, names, dtypes)]
    #
    base = merge_arrays(base, usemask=usemask, fill_value=fill_value)
    # 将基本数据与指定参数合并为一个数组
    if len(data) > 1:
        # 如果数据数组数量大于1
        data = merge_arrays(data, flatten=True, usemask=usemask,
                            fill_value=fill_value)
        # 合并所有数据数组为一个数组
    else:
        data = data.pop()
        # 否则取出单个数据数组
    #
    output = ma.masked_all(
        max(len(base), len(data)),
        dtype=_get_fieldspec(base.dtype) + _get_fieldspec(data.dtype))
    # 创建一个全遮蔽的数组，大小为基本数据与合并数据数组的最大长度，并指定数据类型
    output = recursive_fill_fields(base, output)
    # 递归填充输出数组的字段
    output = recursive_fill_fields(data, output)
    # 递归填充数据数组的字段到输出数组
    #
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)
    # 返回修正后的输出数组，根据参数决定是否使用掩码数组或返回记录数组
# 定义一个生成器函数，用于分发基本参数和数据，实现迭代功能
def _rec_append_fields_dispatcher(base, names, data, dtypes=None):
    yield base
    # 使用生成器将数据添加到生成器函数中
    yield from data


# 将array_function_dispatch装饰器应用于_rec_append_fields_dispatcher函数，用于分派相应的功能
@array_function_dispatch(_rec_append_fields_dispatcher)
# 定义rec_append_fields函数，用于向现有数组添加新字段
def rec_append_fields(base, names, data, dtypes=None):
    """
    Add new fields to an existing array.

    The names of the fields are given with the `names` arguments,
    the corresponding values with the `data` arguments.
    If a single field is appended, `names`, `data` and `dtypes` do not have
    to be lists but just values.

    Parameters
    ----------
    base : array
        Input array to extend.
    names : string, sequence
        String or sequence of strings corresponding to the names
        of the new fields.
    data : array or sequence of arrays
        Array or sequence of arrays storing the fields to add to the base.
    dtypes : sequence of datatypes, optional
        Datatype or sequence of datatypes.
        If None, the datatypes are estimated from the `data`.

    See Also
    --------
    append_fields

    Returns
    -------
    appended_array : np.recarray
        返回扩展后的数组，作为结构化数组（np.recarray）返回。
    """
    # 调用append_fields函数，将新字段添加到基础数组中，并返回扩展后的结果
    return append_fields(base, names, data=data, dtypes=dtypes,
                         asrecarray=True, usemask=False)


# 定义一个分发器函数，用于分派_repack_fields_dispatcher函数
def _repack_fields_dispatcher(a, align=None, recurse=None):
    # 返回传入的参数a，作为迭代结果
    return (a,)


# 将array_function_dispatch装饰器应用于_repack_fields_dispatcher函数，用于分派相应的功能
@array_function_dispatch(_repack_fields_dispatcher)
# 定义repack_fields函数，用于重新打包结构化数组或数据类型在内存中的字段
def repack_fields(a, align=False, recurse=False):
    """
    Re-pack the fields of a structured array or dtype in memory.

    The memory layout of structured datatypes allows fields at arbitrary
    byte offsets. This means the fields can be separated by padding bytes,
    their offsets can be non-monotonically increasing, and they can overlap.

    This method removes any overlaps and reorders the fields in memory so they
    have increasing byte offsets, and adds or removes padding bytes depending
    on the `align` option, which behaves like the `align` option to
    `numpy.dtype`.

    If `align=False`, this method produces a "packed" memory layout in which
    each field starts at the byte the previous field ended, and any padding
    bytes are removed.

    If `align=True`, this methods produces an "aligned" memory layout in which
    each field's offset is a multiple of its alignment, and the total itemsize
    is a multiple of the largest alignment, by adding padding bytes as needed.

    Parameters
    ----------
    a : ndarray or dtype
       array or dtype for which to repack the fields.
    align : boolean
       If true, use an "aligned" memory layout, otherwise use a "packed" layout.
    recurse : boolean
       If True, also repack nested structures.

    Returns
    -------
    repacked : ndarray or dtype
       Copy of `a` with fields repacked, or `a` itself if no repacking was
       needed.

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> def print_offsets(d):
    ...     print("offsets:", [d.fields[name][1] for name in d.names])
    ...     print("itemsize:", d.itemsize)
    ...
    """
    # 返回传入的参数a，作为重新打包后的结果
    return (a,)
    # 创建一个 NumPy 数据类型对象 `dt`，包含三个字段：'f0', 'f1', 'f2'，分别对应于无符号字节（u1）、小端序64位整数（<i8）、小端序双精度浮点数（<f8）
    dt = np.dtype('u1, <i8, <f8', align=True)
    # 打印出创建的数据类型对象 `dt`
    dt
    # dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['u1', '<i8', '<f8'], \
# 定义函数 print_offsets，用于打印偏移量和数据类型的大小
def print_offsets(dt):
    """
    打印数据类型的偏移量和每个字段的大小

    Parameters:
    dt : np.dtype
        NumPy 数据类型对象，描述了字段的布局和大小
    """
    # 获取字段名和偏移量的列表
    offsets = [f[1] for f in dt.fields.values()]
    # 计算数据类型对象的字节大小
    itemsize = dt.itemsize
    # 打印字段偏移量列表和数据类型的大小
    print(f"offsets: {offsets}")
    print(f"itemsize: {itemsize}")

# 重新组织数据类型的字段布局，使得字段紧凑排列
def repack_fields(a, align=True, recurse=False):
    """
    重新组织数据类型的字段布局，以减少内存占用

    Parameters:
    a : np.dtype or np.ndarray
        要重新组织字段布局的数据类型或数组
    align : bool, optional
        是否按照字节对齐，默认为 True
    recurse : bool, optional
        是否递归处理嵌套的数据类型，默认为 False

    Returns:
    np.dtype
        重新组织后的数据类型对象
    """
    # 如果输入不是 np.dtype 对象，则递归调用 repack_fields 处理其 dtype 属性
    if not isinstance(a, np.dtype):
        dt = repack_fields(a.dtype, align=align, recurse=recurse)
        return a.astype(dt, copy=False)

    # 如果数据类型没有字段名，则直接返回
    if a.names is None:
        return a

    # 存储字段信息的列表
    fieldinfo = []
    # 遍历每个字段名
    for name in a.names:
        # 获取字段元组 (dtype, offset, title)
        tup = a.fields[name]
        # 如果指定递归，则重新组织字段的 dtype
        if recurse:
            fmt = repack_fields(tup[0], align=align, recurse=True)
        else:
            fmt = tup[0]

        # 如果字段元组包含标题信息，则重新组织为 (title, name)
        if len(tup) == 3:
            name = (tup[2], name)

        # 将字段信息添加到列表中
        fieldinfo.append((name, fmt))

    # 创建并返回重新组织后的数据类型对象
    dt = np.dtype(fieldinfo, align=align)
    return np.dtype((a.type, dt))

# 获取数据类型的所有标量字段的列表，包括嵌套字段，按照从左到右的顺序排列
def _get_fields_and_offsets(dt, offset=0):
    """
    返回数据类型 "dt" 中所有标量字段的列表，包括嵌套字段，按照从左到右的顺序排列

    Parameters:
    dt : np.dtype
        NumPy 数据类型对象，描述了字段的布局和大小
    offset : int, optional
        偏移量，用于计算字段的绝对偏移，默认为 0

    Returns:
    list
        包含 (dtype, count, offset) 元组的列表，描述了所有标量字段
    """
    # 计算元素数和子数组中的元素数，返回基本 dtype 和元素数
    def count_elem(dt):
        count = 1
        while dt.shape != ():
            for size in dt.shape:
                count *= size
            dt = dt.base
        return dt, count

    # 存储字段列表的列表
    fields = []
    # 遍历每个字段名
    for name in dt.names:
        # 获取字段元组 (dtype, offset)
        field = dt.fields[name]
        f_dt, f_offset = field[0], field[1]
        # 计算字段的元素数和基本 dtype
        f_dt, n = count_elem(f_dt)

        # 如果字段没有字段名，则直接添加到 fields 列表中
        if f_dt.names is None:
            fields.append((np.dtype((f_dt, (n,))), n, f_offset + offset))
        else:
            # 递归调用 _get_fields_and_offsets 处理子字段
            subfields = _get_fields_and_offsets(f_dt, f_offset + offset)
            size = f_dt.itemsize

            # 扩展 fields 列表，处理子数组的情况
            for i in range(n):
                if i == 0:
                    fields.extend(subfields)
                else:
                    fields.extend([(d, c, o + i*size) for d, c, o in subfields])
    return fields

# 计算字段之间的公共步幅，如果步幅不是常数则返回 None
def _common_stride(offsets, counts, itemsize):
    """
    返回字段之间的步幅，如果步幅不是常数则返回 None。counts 中的值指定子数组的长度，
    子数组被视为许多连续字段，始终为正步幅。

    Parameters:
    offsets : list
        字段的偏移量列表
    counts : list
        子数组的长度列表
    itemsize : int
        数据类型对象的字节大小

    Returns:
    int or None
        字段之间的步幅，如果步幅不是常数则返回 None
    """
    if len(offsets) <= 1:
        return itemsize

    # 检查是否存在负步幅
    negative = offsets[1] < offsets[0]
    if negative:
        # 反转列表，使得偏移量升序排列
        it = zip(reversed(offsets), reversed(counts))
    else:
        it = zip(offsets, counts)

    prev_offset = None
    stride = None
    # 遍历迭代器中的偏移量和计数
    for offset, count in it:
        # 如果计数不为1，表示子数组总是 C 连续的
        if count != 1:
            # 如果需要负步长，则返回 None，因为子数组不可能有负步长
            if negative:
                return None
            # 如果步长未指定，则设为元素大小
            if stride is None:
                stride = itemsize
            # 如果步长与元素大小不同，则返回 None
            if stride != itemsize:
                return None
            # 计算子数组的结束偏移量
            end_offset = offset + (count - 1) * itemsize
        else:
            # 如果计数为1，直接将结束偏移量设为当前偏移量
            end_offset = offset

        # 如果存在前一个偏移量，则计算新的步长
        if prev_offset is not None:
            new_stride = offset - prev_offset
            # 如果步长未指定，则设为新计算的步长
            if stride is None:
                stride = new_stride
            # 如果当前步长与新计算的步长不同，则返回 None
            if stride != new_stride:
                return None

        # 更新前一个偏移量为当前的结束偏移量
        prev_offset = end_offset

    # 如果需要负步长，则返回负的当前步长
    if negative:
        return -stride
    # 否则返回当前步长
    return stride
# 定义一个私有函数 _structured_to_unstructured_dispatcher，返回元组 (arr,)
def _structured_to_unstructured_dispatcher(arr, dtype=None, copy=None,
                                           casting=None):
    return (arr,)

# 使用装饰器 array_function_dispatch 将下面的函数注册为 arr 参数的处理函数
@array_function_dispatch(_structured_to_unstructured_dispatcher)
def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    """
    Converts an n-D structured array into an (n+1)-D unstructured array.

    The new array will have a new last dimension equal in size to the
    number of field-elements of the input array. If not supplied, the output
    datatype is determined from the numpy type promotion rules applied to all
    the field datatypes.

    Nested fields, as well as each element of any subarray fields, all count
    as a single field-elements.

    Parameters
    ----------
    arr : ndarray
       Structured array or dtype to convert. Cannot contain object datatype.
    dtype : dtype, optional
       The dtype of the output unstructured array.
    copy : bool, optional
        If true, always return a copy. If false, a view is returned if
        possible, such as when the `dtype` and strides of the fields are
        suitable and the array subtype is one of `numpy.ndarray`,
        `numpy.recarray` or `numpy.memmap`.

        .. versionchanged:: 1.25.0
            A view can now be returned if the fields are separated by a
            uniform stride.

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `numpy.ndarray.astype`. Controls what kind of
        data casting may occur.

    Returns
    -------
    unstructured : ndarray
       Unstructured array with one more dimension.

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
    >>> a
    array([(0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.]),
           (0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.])],
          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])
    >>> rfn.structured_to_unstructured(a)
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
    ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
    >>> np.mean(rfn.structured_to_unstructured(b[['x', 'z']]), axis=-1)
    array([ 3. ,  5.5,  9. , 11. ])

    """
    # 检查 arr 是否为结构化数组，如果不是则抛出 ValueError 异常
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')

    # 调用 _get_fields_and_offsets 函数获取字段信息
    fields = _get_fields_and_offsets(arr.dtype)
    n_fields = len(fields)
    # 如果 arr 中没有字段且未指定 dtype，则抛出 ValueError 异常
    if n_fields == 0 and dtype is None:
        raise ValueError("arr has no fields. Unable to guess dtype")
    # 如果 arr 中没有字段，抛出 NotImplementedError 异常（这段代码暂时无法工作）
    elif n_fields == 0:
        raise NotImplementedError("arr with no fields is not supported")

    # 解构 fields 元组，获取各个字段的 dtype、元素数量和偏移量
    dts, counts, offsets = zip(*fields)
    # 为每个字段生成一个默认名称
    names = ['f{}'.format(n) for n in range(n_fields)]
    # 如果未指定 dtype 参数，则根据 dts 中所有元素的基本数据类型推断输出的 dtype
    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        # 否则，使用指定的 dtype
        out_dtype = np.dtype(dtype)

    # 使用一系列视图和类型转换将数组转换为非结构化数组：

    # 首先使用展平的字段视图（不适用于对象数组）
    # 注意：dts 可能包含子数组的形状信息
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': arr.dtype.itemsize})
    arr = arr.view(flattened_fields)

    # 我们只允许少数几种类型通过调整步幅转换为非结构化数组，因为我们知道它对于 np.matrix 或 np.ma.MaskedArray 是不起作用的。
    can_view = type(arr) in (np.ndarray, np.recarray, np.memmap)
    if (not copy) and can_view and all(dt.base == out_dtype for dt in dts):
        # 所有元素已经具有正确的 dtype；如果它们有一个公共步幅，我们可以返回一个视图
        common_stride = _common_stride(offsets, counts, out_dtype.itemsize)
        if common_stride is not None:
            wrap = arr.__array_wrap__

            new_shape = arr.shape + (sum(counts), out_dtype.itemsize)
            new_strides = arr.strides + (abs(common_stride), 1)

            arr = arr[..., np.newaxis].view(np.uint8)  # 视图为字节
            arr = arr[..., min(offsets):]  # 移除前导未使用数据
            arr = np.lib.stride_tricks.as_strided(arr,
                                                  new_shape,
                                                  new_strides,
                                                  subok=True)

            # 转换并再次去除最后一个维度
            arr = arr.view(out_dtype)[..., 0]

            if common_stride < 0:
                arr = arr[..., ::-1]  # 如果步幅为负数，则反转数组
            if type(arr) is not type(wrap.__self__):
                # 有些类型（如 recarray）在中间过程中转换为 ndarray，因此我们必须再次包装以匹配 copy=True 的行为。
                arr = wrap(arr)
            return arr

    # 然后将所有字段转换为新的 dtype，并封装为紧凑格式
    packed_fields = np.dtype({'names': names,
                              'formats': [(out_dtype, dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)

    # 最后安全地将紧凑格式视为非结构化类型
    return arr.view((out_dtype, (sum(counts),)))
# 定义一个分派器函数，返回传入的数组作为元组的第一个元素
def _unstructured_to_structured_dispatcher(arr, dtype=None, names=None,
                                           align=None, copy=None, casting=None):
    return (arr,)

# 使用array_function_dispatch装饰器，将_unstructured_to_structured_dispatcher函数注册为unstructured_to_structured的分派器
@array_function_dispatch(_unstructured_to_structured_dispatcher)
def unstructured_to_structured(arr, dtype=None, names=None, align=False,
                               copy=False, casting='unsafe'):
    """
    Converts an n-D unstructured array into an (n-1)-D structured array.

    The last dimension of the input array is converted into a structure, with
    number of field-elements equal to the size of the last dimension of the
    input array. By default all output fields have the input array's dtype, but
    an output structured dtype with an equal number of fields-elements can be
    supplied instead.

    Nested fields, as well as each element of any subarray fields, all count
    towards the number of field-elements.

    Parameters
    ----------
    arr : ndarray
       Unstructured array or dtype to convert.
    dtype : dtype, optional
       The structured dtype of the output array
    names : list of strings, optional
       If dtype is not supplied, this specifies the field names for the output
       dtype, in order. The field dtypes will be the same as the input array.
    align : boolean, optional
       Whether to create an aligned memory layout.
    copy : bool, optional
        See copy argument to `numpy.ndarray.astype`. If true, always return a
        copy. If false, and `dtype` requirements are satisfied, a view is
        returned.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `numpy.ndarray.astype`. Controls what kind of
        data casting may occur.

    Returns
    -------
    structured : ndarray
       Structured array with fewer dimensions.

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> dt = np.dtype([('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
    >>> a = np.arange(20).reshape((4,5))
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    >>> rfn.unstructured_to_structured(a, dt)
    array([( 0, ( 1.,  2), [ 3.,  4.]), ( 5, ( 6.,  7), [ 8.,  9.]),
           (10, (11., 12), [13., 14.]), (15, (16., 17), [18., 19.])],
          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])

    """
    # 如果数组是零维的，则抛出值错误
    if arr.shape == ():
        raise ValueError('arr must have at least one dimension')
    # 获取数组最后一个维度的大小
    n_elem = arr.shape[-1]
    # 如果最后一个维度的大小为0，抛出未实现错误
    if n_elem == 0:
        # 太多其他地方的bug，现在无法工作
        raise NotImplementedError("last axis with size 0 is not supported")

    # 如果未提供dtype，则根据names创建一个默认的dtype，字段类型与输入数组的dtype相同
    if dtype is None:
        if names is None:
            names = ['f{}'.format(n) for n in range(n_elem)]
        # 根据names创建dtype，并根据align参数创建对齐的内存布局
        out_dtype = np.dtype([(n, arr.dtype) for n in names], align=align)
        # 获取结构化dtype的字段、计数和偏移量
        fields = _get_fields_and_offsets(out_dtype)
        dts, counts, offsets = zip(*fields)
    else:
        if names is not None:
            raise ValueError("don't supply both dtype and names")
        # 如果 dtype 参数不为 None，则将其转换为 np.dtype 对象
        dtype = np.dtype(dtype)
        # 对输入的 dtype 进行合理性检查，获取字段和偏移量
        fields = _get_fields_and_offsets(dtype)
        # 如果没有字段，则设置为空列表
        if len(fields) == 0:
            dts, counts, offsets = [], [], []
        else:
            # 解压字段元组为分离的数据类型、计数和偏移量
            dts, counts, offsets = zip(*fields)

        # 如果输入数组 arr 的最后一个维度的长度不等于字段数之和，则引发 ValueError 异常
        if n_elem != sum(counts):
            raise ValueError('The length of the last dimension of arr must '
                             'be equal to the number of fields in dtype')
        # 将输出的数据类型设为输入的 dtype
        out_dtype = dtype
        # 如果 align 为 True 且输出数据类型不是对齐的结构体，则引发 ValueError 异常
        if align and not out_dtype.isalignedstruct:
            raise ValueError("align was True but dtype is not aligned")

    # 为字段生成名称列表，如 ['f0', 'f1', ...]
    names = ['f{}'.format(n) for n in range(len(fields))]

    # 使用一系列视图和类型转换来转换为结构化数组：

    # 第一步将 arr 视图转换为一个打包的结构化数组，使用一个统一的数据类型
    packed_fields = np.dtype({'names': names,
                              'formats': [(arr.dtype, dt.shape) for dt in dts]})
    arr = np.ascontiguousarray(arr).view(packed_fields)

    # 接下来将其转换为一个展开但是扁平化的格式，各字段具有不同的数据类型
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': out_dtype.itemsize})
    arr = arr.astype(flattened_fields, copy=copy, casting=casting)

    # 最后将其视图转换为最终的嵌套数据类型，并移除最后一个轴
    return arr.view(out_dtype)[..., 0]
# 定义函数 _apply_along_fields_dispatcher，接收函数和数组作为参数，返回元组 (arr,)
def _apply_along_fields_dispatcher(func, arr):
    return (arr,)

# 使用装饰器 array_function_dispatch 将 apply_along_fields 函数注册为 func 和 arr 参数组合的处理器
@array_function_dispatch(_apply_along_fields_dispatcher)
def apply_along_fields(func, arr):
    """
    Apply function 'func' as a reduction across fields of a structured array.

    This is similar to `numpy.apply_along_axis`, but treats the fields of a
    structured array as an extra axis. The fields are all first cast to a
    common type following the type-promotion rules from `numpy.result_type`
    applied to the field's dtypes.

    Parameters
    ----------
    func : function
       Function to apply on the "field" dimension. This function must
       support an `axis` argument, like `numpy.mean`, `numpy.sum`, etc.
    arr : ndarray
       Structured array for which to apply func.

    Returns
    -------
    out : ndarray
       Result of the reduction operation

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
    ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
    >>> rfn.apply_along_fields(np.mean, b)
    array([ 2.66666667,  5.33333333,  8.66666667, 11.        ])
    >>> rfn.apply_along_fields(np.mean, b[['x', 'z']])
    array([ 3. ,  5.5,  9. , 11. ])

    """
    # 检查结构化数组是否有字段名，如果没有则抛出 ValueError 异常
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')

    # 将结构化数组 arr 转换为非结构化数组 uarr
    uarr = structured_to_unstructured(arr)
    # 调用 func 函数，沿着最后一个轴（字段轴）对 uarr 进行操作，返回结果
    return func(uarr, axis=-1)
    # 下面的方式可以工作并且避免了对轴的要求，但是非常慢：
    #return np.apply_along_axis(func, -1, uarr)

# 定义函数 _assign_fields_by_name_dispatcher，接收 dst、src 和 zero_unassigned 参数，返回元组 (dst, src)
def _assign_fields_by_name_dispatcher(dst, src, zero_unassigned=None):
    return dst, src

# 使用装饰器 array_function_dispatch 将 assign_fields_by_name 函数注册为 dst、src 和 zero_unassigned 参数组合的处理器
@array_function_dispatch(_assign_fields_by_name_dispatcher)
def assign_fields_by_name(dst, src, zero_unassigned=True):
    """
    Assigns values from one structured array to another by field name.

    Normally in numpy >= 1.14, assignment of one structured array to another
    copies fields "by position", meaning that the first field from the src is
    copied to the first field of the dst, and so on, regardless of field name.

    This function instead copies "by field name", such that fields in the dst
    are assigned from the identically named field in the src. This applies
    recursively for nested structures. This is how structure assignment worked
    in numpy >= 1.6 to <= 1.13.

    Parameters
    ----------
    dst : ndarray
    src : ndarray
        The source and destination arrays during assignment.
    zero_unassigned : bool, optional
        If True, fields in the dst for which there was no matching
        field in the src are filled with the value 0 (zero). This
        was the behavior of numpy <= 1.13. If False, those fields
        are not modified.
    """

    # 如果目标结构化数组 dst 没有字段名，则直接将整个 src 复制到 dst
    if dst.dtype.names is None:
        dst[...] = src
        return
    # 遍历目标数据结构的字段名列表
    for name in dst.dtype.names:
        # 检查当前字段名是否存在于源数据结构的字段名列表中
        if name not in src.dtype.names:
            # 如果目标字段名不存在于源字段名列表中
            # 并且 zero_unassigned 参数为 True，则将目标字段置为 0
            if zero_unassigned:
                dst[name] = 0
        else:
            # 如果目标字段名存在于源字段名列表中，则调用函数将对应字段的值赋给目标字段
            assign_fields_by_name(dst[name], src[name],
                                  zero_unassigned)
# 返回输入数组作为元组的单元素元组
def _require_fields_dispatcher(array, required_dtype):
    return (array,)

# 使用 array_function_dispatch 装饰器来声明 require_fields 函数的调度
@array_function_dispatch(_require_fields_dispatcher)
def require_fields(array, required_dtype):
    """
    Casts a structured array to a new dtype using assignment by field-name.

    This function assigns from the old to the new array by name, so the
    value of a field in the output array is the value of the field with the
    same name in the source array. This has the effect of creating a new
    ndarray containing only the fields "required" by the required_dtype.

    If a field name in the required_dtype does not exist in the
    input array, that field is created and set to 0 in the output array.

    Parameters
    ----------
    a : ndarray
       array to cast
    required_dtype : dtype
       datatype for output array

    Returns
    -------
    out : ndarray
        array with the new dtype, with field values copied from the fields in
        the input array with the same name

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.ones(4, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
    >>> rfn.require_fields(a, [('b', 'f4'), ('c', 'u1')])
    array([(1., 1), (1., 1), (1., 1), (1., 1)],
      dtype=[('b', '<f4'), ('c', 'u1')])
    >>> rfn.require_fields(a, [('b', 'f4'), ('newf', 'u1')])
    array([(1., 0), (1., 0), (1., 0), (1., 0)],
      dtype=[('b', '<f4'), ('newf', 'u1')])

    """
    # 使用 required_dtype 创建一个形状与输入数组相同的空数组
    out = np.empty(array.shape, dtype=required_dtype)
    # 调用 assign_fields_by_name 函数，通过字段名从输入数组复制值到输出数组
    assign_fields_by_name(out, array)
    # 返回新创建的输出数组
    return out


# 返回输入数组作为元组的单元素元组
def _stack_arrays_dispatcher(arrays, defaults=None, usemask=None,
                             asrecarray=None, autoconvert=None):
    return arrays

# 使用 array_function_dispatch 装饰器来声明 stack_arrays 函数的调度
@array_function_dispatch(_stack_arrays_dispatcher)
def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False,
                 autoconvert=False):
    """
    Superposes arrays fields by fields

    Parameters
    ----------
    arrays : array or sequence
        Sequence of input arrays.
    defaults : dictionary, optional
        Dictionary mapping field names to the corresponding default values.
    usemask : {True, False}, optional
        Whether to return a MaskedArray (or MaskedRecords is
        `asrecarray==True`) or a ndarray.
    asrecarray : {False, True}, optional
        Whether to return a recarray (or MaskedRecords if `usemask==True`)
        or just a flexible-type ndarray.
    autoconvert : {False, True}, optional
        Whether automatically cast the type of the field to the maximum.

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> x = np.array([1, 2,])
    >>> rfn.stack_arrays(x) is x
    True
    >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])
    >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
    ...   dtype=[('A', '|S3'), ('B', np.double), ('C', np.double)])
    >>> test = rfn.stack_arrays((z,zz))
    >>> test

    """
    # 返回输入的 arrays 参数
    return arrays
    # 创建一个带有数据、掩码和填充值的结构化数组
    masked_array(data=[(b'A', 1.0, --), (b'B', 2.0, --), (b'a', 10.0, 100.0),
                       (b'b', 20.0, 200.0), (b'c', 30.0, 300.0)],
                 mask=[(False, False,  True), (False, False,  True),
                       (False, False, False), (False, False, False),
                       (False, False, False)],
           fill_value=(b'N/A', 1e+20, 1e+20),
                dtype=[('A', 'S3'), ('B', '<f8'), ('C', '<f8')])

    """
    # 检查输入参数 arrays 是否为 ndarray 类型
    if isinstance(arrays, ndarray):
        return arrays
    # 如果 arrays 只包含一个元素，则直接返回该元素
    elif len(arrays) == 1:
        return arrays[0]
    # 将 arrays 中的每个元素转换为任意数组并展平
    seqarrays = [np.asanyarray(a).ravel() for a in arrays]
    # 计算每个序列数组的长度
    nrecords = [len(a) for a in seqarrays]
    # 获取每个序列数组的数据类型
    ndtype = [a.dtype for a in seqarrays]
    # 获取每个字段的名称
    fldnames = [d.names for d in ndtype]
    #
    # 从第一个数据类型获取字段描述符
    dtype_l = ndtype[0]
    newdescr = _get_fieldspec(dtype_l)
    # 提取字段的名称列表
    names = [n for n, d in newdescr]
    # 遍历其他数据类型的字段描述符
    for dtype_n in ndtype[1:]:
        # 遍历每个字段及其类型的元组
        for fname, fdtype in _get_fieldspec(dtype_n):
            # 如果字段名称不在当前名称列表中，则添加新的字段描述符
            if fname not in names:
                newdescr.append((fname, fdtype))
                names.append(fname)
            else:
                # 如果字段名称已存在，检查类型是否兼容
                nameidx = names.index(fname)
                _, cdtype = newdescr[nameidx]
                if autoconvert:
                    # 如果自动转换为真，则更新字段的最大类型
                    newdescr[nameidx] = (fname, max(fdtype, cdtype))
                elif fdtype != cdtype:
                    # 如果类型不兼容且禁止自动转换，则引发类型错误异常
                    raise TypeError("Incompatible type '%s' <> '%s'" %
                                    (cdtype, fdtype))
    # 如果只有一个字段，则使用 concatenate 连接数组
    if len(newdescr) == 1:
        output = ma.concatenate(seqarrays)
    else:
        #
        # 创建一个具有指定形状和数据类型的掩码数组
        output = ma.masked_all((np.sum(nrecords),), newdescr)
        # 计算偏移量，用于确定每个序列数组的位置范围
        offset = np.cumsum(np.r_[0, nrecords])
        seen = []
        # 遍历序列数组、字段名称和偏移量，填充输出数组
        for (a, n, i, j) in zip(seqarrays, fldnames, offset[:-1], offset[1:]):
            names = a.dtype.names
            if names is None:
                output['f%i' % len(seen)][i:j] = a
            else:
                for name in n:
                    output[name][i:j] = a[name]
                    if name not in seen:
                        seen.append(name)
    #
    # 返回修正后的输出结果
    return _fix_output(_fix_defaults(output, defaults),
                       usemask=usemask, asrecarray=asrecarray)
# 定义一个私有函数 _find_duplicates_dispatcher，用于作为 find_duplicates 函数的分发器
def _find_duplicates_dispatcher(
        a, key=None, ignoremask=None, return_index=None):
    # 返回输入数组 a，作为分发器的返回结果
    return (a,)


# 使用装饰器将 _find_duplicates_dispatcher 注册为 find_duplicates 函数的分发器
@array_function_dispatch(_find_duplicates_dispatcher)
def find_duplicates(a, key=None, ignoremask=True, return_index=False):
    """
    Find the duplicates in a structured array along a given key

    Parameters
    ----------
    a : array-like
        Input array
    key : {string, None}, optional
        Name of the fields along which to check the duplicates.
        If None, the search is performed by records
    ignoremask : {True, False}, optional
        Whether masked data should be discarded or considered as duplicates.
    return_index : {False, True}, optional
        Whether to return the indices of the duplicated values.

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> ndtype = [('a', int)]
    >>> a = np.ma.array([1, 1, 1, 2, 2, 3, 3],
    ...         mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
    >>> rfn.find_duplicates(a, ignoremask=True, return_index=True)
    (masked_array(data=[(1,), (1,), (2,), (2,)],
                 mask=[(False,), (False,), (False,), (False,)],
           fill_value=(999999,),
                dtype=[('a', '<i8')]), array([0, 1, 3, 4]))
    """
    # 将输入数组转换为一个扁平化的 NumPy 数组
    a = np.asanyarray(a).ravel()
    # 获取数组元素的字段结构的字典
    fields = get_fieldstructure(a.dtype)
    # 设置基础数据为输入数组
    base = a
    # 如果指定了 key，则根据 key 所指定的字段进行排序
    if key:
        for f in fields[key]:
            base = base[f]
        base = base[key]
    # 获取排序后的索引和排序后的数据
    sortidx = base.argsort()
    sortedbase = base[sortidx]
    sorteddata = sortedbase.filled()
    # 比较排序后的数据，找到重复项
    flag = (sorteddata[:-1] == sorteddata[1:])
    # 如果 ignoremask 为 True，则在需要时将 flag 设置为 False
    if ignoremask:
        sortedmask = sortedbase.recordmask
        flag[sortedmask[1:]] = False
    # 将 flag 向左扩展一个位置，以便包括左边的点
    flag = np.concatenate(([False], flag))
    # 需要将左边的点也包含进去，否则会遗漏
    flag[:-1] = flag[:-1] + flag[1:]
    # 根据排序后的索引找到重复项的值
    duplicates = a[sortidx][flag]
    # 如果 return_index 为 True，则返回重复项和它们的索引
    if return_index:
        return (duplicates, sortidx[flag])
    else:
        # 否则，只返回重复项的值
        return duplicates


# 定义一个私有函数 _join_by_dispatcher，用于作为 join_by 函数的分发器
def _join_by_dispatcher(
        key, r1, r2, jointype=None, r1postfix=None, r2postfix=None,
        defaults=None, usemask=None, asrecarray=None):
    # 返回 r1 和 r2，作为分发器的返回结果
    return (r1, r2)


# 使用装饰器将 _join_by_dispatcher 注册为 join_by 函数的分发器
@array_function_dispatch(_join_by_dispatcher)
def join_by(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
            defaults=None, usemask=True, asrecarray=False):
    """
    Join arrays `r1` and `r2` on key `key`.

    The key should be either a string or a sequence of string corresponding
    to the fields used to join the array.  An exception is raised if the
    `key` field cannot be found in the two input arrays.  Neither `r1` nor
    `r2` should have any duplicates along `key`: the presence of duplicates
    will make the output quite unreliable. Note that duplicates are not
    """
    # 函数 join_by 的文档字符串，描述了函数的作用和参数信息，但不包括代码功能的详细解释
    # 检查连接类型参数 jointype 是否合法
    if jointype not in ('inner', 'outer', 'leftouter'):
        raise ValueError(
                "The 'jointype' argument should be in 'inner', "
                "'outer' or 'leftouter' (got '%s' instead)" % jointype
                )

    # 如果 key 是字符串，则转换为包含一个元素的元组
    if isinstance(key, str):
        key = (key,)

    # 检查 key 是否有重复项
    if len(set(key)) != len(key):
        # 如果有重复项，抛出 ValueError 异常
        dup = next(x for n,x in enumerate(key) if x in key[n+1:])
        raise ValueError("duplicate join key %r" % dup)

    # 检查 key 中的字段名是否存在于 r1 和 r2 的数据类型中
    for name in key:
        if name not in r1.dtype.names:
            # 如果 key 中的字段名在 r1 中不存在，抛出 ValueError 异常
            raise ValueError('r1 does not have key field %r' % name)
        if name not in r2.dtype.names:
            # 如果 key 中的字段名在 r2 中不存在，抛出 ValueError 异常
            raise ValueError('r2 does not have key field %r' % name)

    # 将 r1 和 r2 转换为扁平化数组
    r1 = r1.ravel()
    r2 = r2.ravel()

    # 获取 r1 和 r2 的字段名，并赋值给 r1names 和 r2names
    (r1names, r2names) = (r1.dtype.names, r2.dtype.names)

    # 检查字段名是否有冲突
    collisions = (set(r1names) & set(r2names)) - set(key)
    # 如果发生冲突并且 r1postfix 和 r2postfix 都为空，则抛出 ValueError 异常
    if collisions and not (r1postfix or r2postfix):
        msg = "r1 and r2 contain common names, r1postfix and r2postfix "
        msg += "can't both be empty"
        raise ValueError(msg)

    # 创建仅包含键的临时数组
    # （使用 `r1` 中键的顺序保持向后兼容性）
    key1 = [n for n in r1names if n in key]
    # 从 `r1` 和 `r2` 中仅保留与 `key1` 相关的字段
    r1k = _keep_fields(r1, key1)
    r2k = _keep_fields(r2, key1)

    # 将两个数组连接起来进行比较
    aux = ma.concatenate((r1k, r2k))
    # 根据 `key` 的顺序对 `aux` 进行排序并返回排序后的索引
    idx_sort = aux.argsort(order=key)
    aux = aux[idx_sort]

    # 获取共同的键
    flag_in = ma.concatenate(([False], aux[1:] == aux[:-1]))
    flag_in[:-1] = flag_in[1:] + flag_in[:-1]
    idx_in = idx_sort[flag_in]
    idx_1 = idx_in[(idx_in < nb1)]
    idx_2 = idx_in[(idx_in >= nb1)] - nb1
    (r1cmn, r2cmn) = (len(idx_1), len(idx_2))

    # 根据联接类型进行处理
    if jointype == 'inner':
        (r1spc, r2spc) = (0, 0)
    elif jointype == 'outer':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
        idx_2 = np.concatenate((idx_2, idx_out[(idx_out >= nb1)] - nb1))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, len(idx_2) - r2cmn)
    elif jointype == 'leftouter':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
        (r1spc, r2spc) = (len(idx_1) - r1cmn, 0)

    # 从每个输入中选择条目
    (s1, s2) = (r1[idx_1], r2[idx_2])

    # 构建输出数组的新描述......
    # 从键字段开始
    ndtype = _get_fieldspec(r1k.dtype)

    # 添加来自 `r1` 的字段
    for fname, fdtype in _get_fieldspec(r1.dtype):
        if fname not in key:
            ndtype.append((fname, fdtype))

    # 添加来自 `r2` 的字段
    for fname, fdtype in _get_fieldspec(r2.dtype):
        # 我们之前是否已经见过当前的名称？
        # 我们每次都需要重建这个列表
        names = list(name for name, dtype in ndtype)
        try:
            nameidx = names.index(fname)
        except ValueError:
            # ... 我们之前没有见过：将描述添加到当前列表中
            ndtype.append((fname, fdtype))
        else:
            # 发生冲突
            _, cdtype = ndtype[nameidx]
            if fname in key:
                # 当前字段是键的一部分：取最大的 dtype
                ndtype[nameidx] = (fname, max(fdtype, cdtype))
            else:
                # 当前字段不是键的一部分：添加后缀，并将新字段放置在旧字段的旁边
                ndtype[nameidx:nameidx + 1] = [
                    (fname + r1postfix, cdtype),
                    (fname + r2postfix, fdtype)
                ]

    # 从新字段重新构建 dtype
    ndtype = np.dtype(ndtype)

    # 找到最大的共同字段数：
    # r1cmn 和 r2cmn 应该相等，但...
    cmn = max(r1cmn, r2cmn)

    # 构建一个空数组
    # 创建一个所有元素都是掩码值的数组，形状为 (cmn + r1spc + r2spc,)，数据类型为 ndtype
    output = ma.masked_all((cmn + r1spc + r2spc,), dtype=ndtype)
    # 获取输出数组的字段名列表
    names = output.dtype.names
    # 遍历 r1names 中的字段名
    for f in r1names:
        # 从 s1 中获取名为 f 的字段数据
        selected = s1[f]
        # 如果字段名 f 不在 names 中，或者在 r2names 中但没有 r2postfix 且不在 key 中，则添加 r1postfix 后缀
        if f not in names or (f in r2names and not r2postfix and f not in key):
            f += r1postfix
        # 获取当前输出数组中名为 f 的字段
        current = output[f]
        # 将 selected 中前 r1cmn 个元素复制到 current 的前 r1cmn 个位置
        current[:r1cmn] = selected[:r1cmn]
        # 如果 jointype 是 'outer' 或 'leftouter'，将 selected 中 r1cmn 之后的元素复制到 current 的 cmn 到 cmn + r1spc 位置
        if jointype in ('outer', 'leftouter'):
            current[cmn:cmn + r1spc] = selected[r1cmn:]
    # 再次遍历 r2names 中的字段名
    for f in r2names:
        # 从 s2 中获取名为 f 的字段数据
        selected = s2[f]
        # 如果字段名 f 不在 names 中，或者在 r1names 中但没有 r1postfix 且不在 key 中，则添加 r2postfix 后缀
        if f not in names or (f in r1names and not r1postfix and f not in key):
            f += r2postfix
        # 获取当前输出数组中名为 f 的字段
        current = output[f]
        # 将 selected 中前 r2cmn 个元素复制到 current 的前 r2cmn 个位置
        current[:r2cmn] = selected[:r2cmn]
        # 如果 jointype 是 'outer' 并且 r2spc 不为零，则将 selected 中最后 r2spc 个元素复制到 current 的末尾 r2spc 个位置
        if (jointype == 'outer') and r2spc:
            current[-r2spc:] = selected[r2cmn:]
    # 对输出数组按照指定的 key 进行排序
    output.sort(order=key)
    # 构建关键字参数字典
    kwargs = dict(usemask=usemask, asrecarray=asrecarray)
    # 调用 _fix_defaults 函数处理默认值，然后调用 _fix_output 函数修正输出
    return _fix_output(_fix_defaults(output, defaults), **kwargs)
# 使用 `_rec_join_dispatcher` 函数作为分派函数的装饰器，用于分发不同情况下的数组连接操作
def _rec_join_dispatcher(
        key, r1, r2, jointype=None, r1postfix=None, r2postfix=None,
        defaults=None):
    # 返回 r1 和 r2 这两个参数的元组
    return (r1, r2)


# 使用 `array_function_dispatch` 装饰器，将 `_rec_join_dispatcher` 函数与 `rec_join` 函数关联
@array_function_dispatch(_rec_join_dispatcher)
def rec_join(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
             defaults=None):
    """
    Join arrays `r1` and `r2` on keys.
    Alternative to join_by, that always returns a np.recarray.

    See Also
    --------
    join_by : equivalent function
    """
    # 设置关键字参数的默认值，并创建关键字参数字典 `kwargs`
    kwargs = dict(jointype=jointype, r1postfix=r1postfix, r2postfix=r2postfix,
                  defaults=defaults, usemask=False, asrecarray=True)
    # 调用 `join_by` 函数进行数组连接操作，返回一个 `np.recarray` 类型的对象
    return join_by(key, r1, r2, **kwargs)
```