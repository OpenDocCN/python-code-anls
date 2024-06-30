# `D:\src\scipysrc\scipy\scipy\io\_idl.py`

```
def _read_bytes(f, n):
    '''从文件对象 `f` 中读取 `n` 字节的数据并返回'''

    # 从文件对象 `f` 中读取 `n` 字节的数据
    return f.read(n)
    # 读取下一个 `n` 字节的数据并返回
    return f.read(n)
# 读取一个单字节
def _read_byte(f):
    '''Read a single byte'''
    # 从文件对象 f 中读取 4 字节数据，取其中的第一个字节，并转换为无符号 8 位整数
    return np.uint8(struct.unpack('>B', f.read(4)[:1])[0])


# 读取一个带符号的 32 位整数
def _read_long(f):
    '''Read a signed 32-bit integer'''
    # 从文件对象 f 中读取 4 字节数据，按大端格式解析为带符号 32 位整数
    return np.int32(struct.unpack('>l', f.read(4))[0])


# 读取一个带符号的 16 位整数
def _read_int16(f):
    '''Read a signed 16-bit integer'''
    # 从文件对象 f 中读取 4 字节数据，取其中第 3 和第 4 字节，并解析为带符号 16 位整数
    return np.int16(struct.unpack('>h', f.read(4)[2:4])[0])


# 读取一个带符号的 32 位整数
def _read_int32(f):
    '''Read a signed 32-bit integer'''
    # 从文件对象 f 中读取 4 字节数据，按大端格式解析为带符号 32 位整数
    return np.int32(struct.unpack('>i', f.read(4))[0])


# 读取一个带符号的 64 位整数
def _read_int64(f):
    '''Read a signed 64-bit integer'''
    # 从文件对象 f 中读取 8 字节数据，按大端格式解析为带符号 64 位整数
    return np.int64(struct.unpack('>q', f.read(8))[0])


# 读取一个无符号的 16 位整数
def _read_uint16(f):
    '''Read an unsigned 16-bit integer'''
    # 从文件对象 f 中读取 4 字节数据，取其中第 3 和第 4 字节，并解析为无符号 16 位整数
    return np.uint16(struct.unpack('>H', f.read(4)[2:4])[0])


# 读取一个无符号的 32 位整数
def _read_uint32(f):
    '''Read an unsigned 32-bit integer'''
    # 从文件对象 f 中读取 4 字节数据，按大端格式解析为无符号 32 位整数
    return np.uint32(struct.unpack('>I', f.read(4))[0])


# 读取一个无符号的 64 位整数
def _read_uint64(f):
    '''Read an unsigned 64-bit integer'''
    # 从文件对象 f 中读取 8 字节数据，按大端格式解析为无符号 64 位整数
    return np.uint64(struct.unpack('>Q', f.read(8))[0])


# 读取一个 32 位浮点数
def _read_float32(f):
    '''Read a 32-bit float'''
    # 从文件对象 f 中读取 4 字节数据，按大端格式解析为 32 位浮点数
    return np.float32(struct.unpack('>f', f.read(4))[0])


# 读取一个 64 位浮点数
def _read_float64(f):
    '''Read a 64-bit float'''
    # 从文件对象 f 中读取 8 字节数据，按大端格式解析为 64 位浮点数
    return np.float64(struct.unpack('>d', f.read(8))[0])


class Pointer:
    '''Class used to define pointers'''

    def __init__(self, index):
        # 初始化指针对象，设置其索引值
        self.index = index
        return


class ObjectPointer(Pointer):
    '''Class used to define object pointers'''
    # ObjectPointer 类继承自 Pointer 类，用于定义对象指针
    pass


# 读取一个字符串
def _read_string(f):
    '''Read a string'''
    # 读取字符串的长度
    length = _read_long(f)
    if length > 0:
        # 如果长度大于 0，则读取指定长度的字节数据，并以 Latin-1 编码解码为字符串
        chars = _read_bytes(f, length).decode('latin1')
        # 对文件进行 32 位对齐
        _align_32(f)
    else:
        chars = ''
    return chars


# 读取一个数据字符串（长度被指定两次）
def _read_string_data(f):
    '''Read a data string (length is specified twice)'''
    # 读取字符串的长度
    length = _read_long(f)
    if length > 0:
        # 如果长度大于 0，则再次读取长度，并读取相应长度的字节数据
        string_data = _read_bytes(f, length)
        # 对文件进行 32 位对齐
        _align_32(f)
    else:
        string_data = ''
    return string_data


# 读取指定数据类型的变量
def _read_data(f, dtype):
    '''Read a variable with a specified data type'''
    if dtype == 1:
        # 如果数据类型为 1，检查读取的 32 位整数是否为 1，否则抛出异常
        if _read_int32(f) != 1:
            raise Exception("Error occurred while reading byte variable")
        return _read_byte(f)
    elif dtype == 2:
        # 如果数据类型为 2，读取带符号 16 位整数
        return _read_int16(f)
    elif dtype == 3:
        # 如果数据类型为 3，读取带符号 32 位整数
        return _read_int32(f)
    elif dtype == 4:
        # 如果数据类型为 4，读取 32 位浮点数
        return _read_float32(f)
    elif dtype == 5:
        # 如果数据类型为 5，读取 64 位浮点数
        return _read_float64(f)
    elif dtype == 6:
        # 如果数据类型为 6，读取两个 32 位浮点数构成的复数
        real = _read_float32(f)
        imag = _read_float32(f)
        return np.complex64(real + imag * 1j)
    elif dtype == 7:
        # 如果数据类型为 7，读取数据字符串
        return _read_string_data(f)
    elif dtype == 8:
        # 如果数据类型为 8，抛出异常，不应该出现在这里
        raise Exception("Should not be here - please report this")
    elif dtype == 9:
        # 如果数据类型为 9，读取两个 64 位浮点数构成的复数
        real = _read_float64(f)
        imag = _read_float64(f)
        return np.complex128(real + imag * 1j)
    elif dtype == 10:
        # 如果数据类型为 10，读取一个指针对象
        return Pointer(_read_int32(f))
    elif dtype == 11:
        # 如果数据类型为 11，读取一个对象指针对象
        return ObjectPointer(_read_int32(f))
    elif dtype == 12:
        # 如果数据类型为 12，读取无符号 16 位整数
        return _read_uint16(f)
    elif dtype == 13:
        # 如果数据类型为 13，读取无符号 32 位整数
        return _read_uint32(f)
    # 如果 dtype 等于 14，则调用 _read_int64 函数并返回其结果
    elif dtype == 14:
        return _read_int64(f)
    # 如果 dtype 等于 15，则调用 _read_uint64 函数并返回其结果
    elif dtype == 15:
        return _read_uint64(f)
    # 如果以上条件都不满足，则抛出异常，表示遇到未知的 IDL 类型
    else:
        raise Exception("Unknown IDL type: %i - please report this" % dtype)
# 读取给定文件中的结构数据，使用提供的数组描述符和结构描述符作为参数
def _read_structure(f, array_desc, struct_desc):
    # 从数组描述符中获取行数
    nrows = array_desc['nelements']
    # 获取结构描述符中的列信息
    columns = struct_desc['tagtable']

    # 初始化数据类型列表
    dtype = []
    # 遍历列信息
    for col in columns:
        # 如果列是结构或数组
        if col['structure'] or col['array']:
            # 添加对象类型到数据类型列表
            dtype.append(((col['name'].lower(), col['name']), np.object_))
        else:
            # 如果列的类型代码在预定义的类型字典中
            if col['typecode'] in DTYPE_DICT:
                # 添加指定类型到数据类型列表
                dtype.append(((col['name'].lower(), col['name']),
                              DTYPE_DICT[col['typecode']]))
            else:
                # 抛出异常，指出未实现的变量类型
                raise Exception("Variable type %i not implemented" % col['typecode'])

    # 使用数据类型列表创建结构化的 numpy 数组
    structure = np.rec.recarray((nrows, ), dtype=dtype)

    # 遍历每一行数据
    for i in range(nrows):
        # 遍历每一列信息
        for col in columns:
            # 获取列的类型代码
            dtype = col['typecode']
            # 如果列是结构类型
            if col['structure']:
                # 递归调用 _read_structure() 读取结构数据并赋值给结构字段
                structure[col['name']][i] = _read_structure(f,
                                      struct_desc['arrtable'][col['name']],
                                      struct_desc['structtable'][col['name']])
            elif col['array']:
                # 调用 _read_array() 读取数组数据并赋值给数组字段
                structure[col['name']][i] = _read_array(f, dtype,
                                      struct_desc['arrtable'][col['name']])
            else:
                # 调用 _read_data() 读取普通数据并赋值给字段
                structure[col['name']][i] = _read_data(f, dtype)

    # 如果数组的维度大于1，则根据维度信息重塑结构
    if array_desc['ndims'] > 1:
        dims = array_desc['dims'][:int(array_desc['ndims'])]
        dims.reverse()
        structure = structure.reshape(dims)

    # 返回结构化数组
    return structure


# 读取给定类型码和数组描述符的数组数据
def _read_array(f, typecode, array_desc):
    # 根据类型码选择不同的读取方式
    if typecode in [1, 3, 4, 5, 6, 9, 13, 14, 15]:
        # 对于特定的类型码，根据数组描述符中的字节数读取数据
        if typecode == 1:
            nbytes = _read_int32(f)
            if nbytes != array_desc['nbytes']:
                warnings.warn("Not able to verify number of bytes from header",
                              stacklevel=3)

        # 将读取的字节作为 numpy 数组
        array = np.frombuffer(f.read(array_desc['nbytes']),
                              dtype=DTYPE_DICT[typecode])

    elif typecode in [2, 12]:
        # 对于2字节类型，需要跳过每两个数据，因为它们不是紧凑排列的
        array = np.frombuffer(f.read(array_desc['nbytes']*2),
                              dtype=DTYPE_DICT[typecode])[1::2]

    else:
        # 否则，将字节读取为列表
        array = []
        for i in range(array_desc['nelements']):
            dtype = typecode
            data = _read_data(f, dtype)
            array.append(data)

        # 将列表转换为 numpy 数组，类型为对象
        array = np.array(array, dtype=np.object_)

    # 如果数组的维度大于1，则根据维度信息重塑数组
    if array_desc['ndims'] > 1:
        dims = array_desc['dims'][:int(array_desc['ndims'])]
        dims.reverse()
        array = array.reshape(dims)

    # 调整到下一个对齐位置
    _align_32(f)
    return array


注释：


    # 返回变量 array 的值作为函数的结果
def _read_record(f):
    '''Function to read in a full record'''

    # 读取记录类型，调用 _read_long 函数获取整数值
    record = {'rectype': _read_long(f)}

    # 读取下一个记录位置，由两个 32 位整数组成
    nextrec = _read_uint32(f)
    nextrec += _read_uint32(f).astype(np.int64) * 2**32

    # 跳过 4 字节
    _skip_bytes(f, 4)

    # 检查记录类型是否在预定义的字典中
    if record['rectype'] not in RECTYPE_DICT:
        raise Exception("Unknown RECTYPE: %i" % record['rectype'])

    # 将记录类型转换为字符串描述
    record['rectype'] = RECTYPE_DICT[record['rectype']]

    # 处理特定类型为 "VARIABLE" 或 "HEAP_DATA" 的记录
    if record['rectype'] in ["VARIABLE", "HEAP_DATA"]:

        # 如果记录类型是 "VARIABLE"
        if record['rectype'] == "VARIABLE":
            # 读取变量名字符串
            record['varname'] = _read_string(f)
        else:
            # 否则，读取堆索引并跳过额外的 4 字节
            record['heap_index'] = _read_long(f)
            _skip_bytes(f, 4)

        # 读取类型描述信息
        rectypedesc = _read_typedesc(f)

        # 如果类型码为 0
        if rectypedesc['typecode'] == 0:

            # 如果下一个记录位置等于当前文件指针位置，数据置为 None，表示空值
            if nextrec == f.tell():
                record['data'] = None  # Indicates NULL value
            else:
                raise ValueError("Unexpected type code: 0")

        else:

            # 读取变量起始位置，应为 7
            varstart = _read_long(f)
            if varstart != 7:
                raise Exception("VARSTART is not 7")

            # 如果类型描述具有结构体，则读取结构体数据
            if rectypedesc['structure']:
                record['data'] = _read_structure(f, rectypedesc['array_desc'],
                                                    rectypedesc['struct_desc'])
            # 如果类型描述具有数组，则读取数组数据
            elif rectypedesc['array']:
                record['data'] = _read_array(f, rectypedesc['typecode'],
                                                rectypedesc['array_desc'])
            # 否则，按类型码读取数据
            else:
                dtype = rectypedesc['typecode']
                record['data'] = _read_data(f, dtype)

    # 如果记录类型为 "TIMESTAMP"
    elif record['rectype'] == "TIMESTAMP":

        # 跳过 4 * 256 字节，读取日期、用户和主机信息
        _skip_bytes(f, 4*256)
        record['date'] = _read_string(f)
        record['user'] = _read_string(f)
        record['host'] = _read_string(f)

    # 如果记录类型为 "VERSION"
    elif record['rectype'] == "VERSION":

        # 依次读取格式、架构、操作系统和版本信息
        record['format'] = _read_long(f)
        record['arch'] = _read_string(f)
        record['os'] = _read_string(f)
        record['release'] = _read_string(f)

    # 如果记录类型为 "IDENTIFICATON"
    elif record['rectype'] == "IDENTIFICATON":

        # 依次读取作者、标题和 ID 代码
        record['author'] = _read_string(f)
        record['title'] = _read_string(f)
        record['idcode'] = _read_string(f)

    # 如果记录类型为 "NOTICE"
    elif record['rectype'] == "NOTICE":

        # 读取通知信息字符串
        record['notice'] = _read_string(f)

    # 如果记录类型为 "DESCRIPTION"
    elif record['rectype'] == "DESCRIPTION":

        # 读取描述信息的字符串数据
        record['description'] = _read_string_data(f)

    # 如果记录类型为 "HEAP_HEADER"
    elif record['rectype'] == "HEAP_HEADER":

        # 读取堆头信息的值数量和索引列表
        record['nvalues'] = _read_long(f)
        record['indices'] = [_read_long(f) for _ in range(record['nvalues'])]

    # 如果记录类型为 "COMMONBLOCK"
    elif record['rectype'] == "COMMONBLOCK":

        # 读取公共块的变量数量、块名称和变量名列表
        record['nvars'] = _read_long(f)
        record['name'] = _read_string(f)
        record['varnames'] = [_read_string(f) for _ in range(record['nvars'])]

    # 如果记录类型为 "END_MARKER"
    elif record['rectype'] == "END_MARKER":

        # 标记为结束记录
        record['end'] = True

    # 如果记录类型为 "UNKNOWN"
    elif record['rectype'] == "UNKNOWN":

        # 发出警告，跳过未知记录
        warnings.warn("Skipping UNKNOWN record", stacklevel=3)

    # 如果记录类型为 "SYSTEM_VARIABLE"
    elif record['rectype'] == "SYSTEM_VARIABLE":

        # 发出警告，跳过系统变量记录
        warnings.warn("Skipping SYSTEM_VARIABLE record", stacklevel=3)
    else:
        # 如果 record['rectype'] 的值不在预期范围内，抛出异常
        raise Exception(f"record['rectype']={record['rectype']} not implemented")

    # 将文件指针移动到下一个记录的位置
    f.seek(nextrec)

    # 返回解析出的 record 对象
    return record
# 读取类型描述符的函数
def _read_typedesc(f):
    '''Function to read in a type descriptor'''

    # 从文件对象 f 中读取类型码和变量标志，并组成字典 typedesc
    typedesc = {'typecode': _read_long(f), 'varflags': _read_long(f)}

    # 检查是否存在系统变量，如果有则抛出异常
    if typedesc['varflags'] & 2 == 2:
        raise Exception("System variables not implemented")

    # 检查是否为数组，设置 typedesc 中的 array 字段
    typedesc['array'] = typedesc['varflags'] & 4 == 4
    # 检查是否为结构体，设置 typedesc 中的 structure 字段
    typedesc['structure'] = typedesc['varflags'] & 32 == 32

    # 如果是结构体，进一步读取数组描述符和结构描述符
    if typedesc['structure']:
        typedesc['array_desc'] = _read_arraydesc(f)
        typedesc['struct_desc'] = _read_structdesc(f)
    # 如果是数组，只读取数组描述符
    elif typedesc['array']:
        typedesc['array_desc'] = _read_arraydesc(f)

    # 返回构建好的类型描述符字典
    return typedesc


# 读取数组描述符的函数
def _read_arraydesc(f):
    '''Function to read in an array descriptor'''

    # 创建空的数组描述符字典 arraydesc，并读取 arrstart 字段
    arraydesc = {'arrstart': _read_long(f)}

    # 根据 arrstart 的值选择不同的处理分支
    if arraydesc['arrstart'] == 8:
        # 如果 arrstart 为 8，按照特定格式读取剩余字段
        _skip_bytes(f, 4)
        arraydesc['nbytes'] = _read_long(f)
        arraydesc['nelements'] = _read_long(f)
        arraydesc['ndims'] = _read_long(f)
        _skip_bytes(f, 8)
        arraydesc['nmax'] = _read_long(f)
        # 读取数组的维度信息列表
        arraydesc['dims'] = [_read_long(f) for _ in range(arraydesc['nmax'])]

    elif arraydesc['arrstart'] == 18:
        # 如果 arrstart 为 18，发出警告并按照 64 位数组读取实验性方式处理
        warnings.warn("Using experimental 64-bit array read", stacklevel=3)
        _skip_bytes(f, 8)
        arraydesc['nbytes'] = _read_uint64(f)
        arraydesc['nelements'] = _read_uint64(f)
        arraydesc['ndims'] = _read_long(f)
        _skip_bytes(f, 8)
        arraydesc['nmax'] = 8
        # 读取数组的维度信息列表
        arraydesc['dims'] = []
        for d in range(arraydesc['nmax']):
            v = _read_long(f)
            if v != 0:
                raise Exception("Expected a zero in ARRAY_DESC")
            arraydesc['dims'].append(_read_long(f))

    else:
        # 如果 arrstart 的值未知，则抛出异常
        raise Exception("Unknown ARRSTART: %i" % arraydesc['arrstart'])

    # 返回构建好的数组描述符字典
    return arraydesc


# 读取结构体描述符的函数
def _read_structdesc(f):
    '''Function to read in a structure descriptor'''

    # 创建空的结构体描述符字典 structdesc
    structdesc = {}

    # 从文件对象 f 中读取结构体的起始码 structstart
    structstart = _read_long(f)
    # 如果 structstart 不等于 9，则抛出异常
    if structstart != 9:
        raise Exception("STRUCTSTART should be 9")

    # 读取结构体名称，并存入 structdesc 中的 name 字段
    structdesc['name'] = _read_string(f)
    # 读取预定义字段，存入 predef 字段
    predef = _read_long(f)
    # 读取标签数、字节长度，并存入对应字段
    structdesc['ntags'] = _read_long(f)
    structdesc['nbytes'] = _read_long(f)

    # 解析预定义字段中的各位信息，并存入相应字段
    structdesc['predef'] = predef & 1
    structdesc['inherits'] = predef & 2
    structdesc['is_super'] = predef & 4

    # 返回构建好的结构体描述符字典
    return structdesc
    # 如果结构描述中的 'predef' 键为假值（False 或者空），则执行以下操作
    if not structdesc['predef']:
        # 读取 'ntags' 次数的标签描述，并将其存入 'tagtable' 列表
        structdesc['tagtable'] = [_read_tagdesc(f)
                                  for _ in range(structdesc['ntags'])]

        # 遍历 'tagtable' 中的每个标签，为每个标签读取其名称并存入 'name' 键
        for tag in structdesc['tagtable']:
            tag['name'] = _read_string(f)

        # 创建包含数组描述的字典 'arrtable'，键为标签名称，值为对应的数组描述
        structdesc['arrtable'] = {tag['name']: _read_arraydesc(f)
                                  for tag in structdesc['tagtable']
                                  if tag['array']}

        # 创建包含结构描述的字典 'structtable'，键为标签名称，值为对应的结构描述
        structdesc['structtable'] = {tag['name']: _read_structdesc(f)
                                     for tag in structdesc['tagtable']
                                     if tag['structure']}

        # 如果结构描述中有继承或者是超类标志，则执行以下操作
        if structdesc['inherits'] or structdesc['is_super']:
            # 读取类名字符串并存入 'classname'
            structdesc['classname'] = _read_string(f)
            # 读取并存储超类数量到 'nsupclasses'
            structdesc['nsupclasses'] = _read_long(f)
            # 读取 'nsupclasses' 个超类名称，并存入 'supclassnames' 列表
            structdesc['supclassnames'] = [
                _read_string(f) for _ in range(structdesc['nsupclasses'])]
            # 读取 'nsupclasses' 个超类的结构描述，并存入 'supclasstable' 列表
            structdesc['supclasstable'] = [
                _read_structdesc(f) for _ in range(structdesc['nsupclasses'])]

        # 将当前结构描述存入全局的 STRUCT_DICT 字典，键为结构名
        STRUCT_DICT[structdesc['name']] = structdesc

    else:
        # 如果 'predef' 键为真值（True 或者非空），并且结构描述的名称不在 STRUCT_DICT 中，则引发异常
        if structdesc['name'] not in STRUCT_DICT:
            raise Exception("PREDEF=1 but can't find definition")

        # 将结构描述设置为在 STRUCT_DICT 中找到的对应名称的结构描述
        structdesc = STRUCT_DICT[structdesc['name']]

    # 返回最终的结构描述
    return structdesc
# 定义一个函数，用于读取标签描述符
def _read_tagdesc(f):
    '''Function to read in a tag descriptor'''
    # 初始化标签描述字典，包含偏移量的长整型数据
    tagdesc = {'offset': _read_long(f)}

    # 如果偏移量为-1，则使用无符号64位整数读取偏移量
    if tagdesc['offset'] == -1:
        tagdesc['offset'] = _read_uint64(f)

    # 读取标签类型码
    tagdesc['typecode'] = _read_long(f)
    # 读取标签标志位
    tagflags = _read_long(f)

    # 检查标志位，设置标签是否为数组、结构体或标量
    tagdesc['array'] = tagflags & 4 == 4
    tagdesc['structure'] = tagflags & 32 == 32
    tagdesc['scalar'] = tagdesc['typecode'] in DTYPE_DICT
    # 假设 '10'x 表示标量

    # 返回读取的标签描述字典
    return tagdesc


# 定义一个函数，用于替换堆中的变量
def _replace_heap(variable, heap):
    # 如果变量是指针类型
    if isinstance(variable, Pointer):
        # 循环直到找到非指针变量或者变量为None
        while isinstance(variable, Pointer):
            # 如果指针索引为0，则将变量设为None
            if variable.index == 0:
                variable = None
            else:
                # 如果指针索引在堆中存在，则将变量设为堆中对应的值
                if variable.index in heap:
                    variable = heap[variable.index]
                else:
                    # 如果指针引用的变量未在堆中找到，则发出警告，并将变量设为None
                    warnings.warn("Variable referenced by pointer not found "
                                  "in heap: variable will be set to None",
                                  stacklevel=3)
                    variable = None

        # 递归调用_replace_heap函数替换变量，直到不再是指针类型
        replace, new = _replace_heap(variable, heap)

        # 如果需要替换，则将变量设为新值
        if replace:
            variable = new

        # 返回替换标志和变量
        return True, variable

    # 如果变量是np.rec.recarray类型
    elif isinstance(variable, np.rec.recarray):
        # 遍历记录
        for ir, record in enumerate(variable):
            # 递归调用_replace_heap函数替换记录中的变量
            replace, new = _replace_heap(record, heap)
            # 如果需要替换，则更新记录
            if replace:
                variable[ir] = new

        # 返回替换标志和变量
        return False, variable

    # 如果变量是np.record类型
    elif isinstance(variable, np.record):
        # 遍历值
        for iv, value in enumerate(variable):
            # 递归调用_replace_heap函数替换值
            replace, new = _replace_heap(value, heap)
            # 如果需要替换，则更新值
            if replace:
                variable[iv] = new

        # 返回替换标志和变量
        return False, variable

    # 如果变量是np.ndarray类型
    elif isinstance(variable, np.ndarray):
        # 如果数组的dtype是np.object_
        if variable.dtype.type is np.object_:
            # 遍历数组中的值
            for iv in range(variable.size):
                # 递归调用_replace_heap函数替换数组中的对象
                replace, new = _replace_heap(variable.item(iv), heap)
                # 如果需要替换，则更新数组中的对象
                if replace:
                    variable.reshape(-1)[iv] = new

        # 返回替换标志和变量
        return False, variable

    # 如果变量不是以上任何一种类型，则直接返回变量和替换标志
    else:
        return False, variable


# 定义一个自定义字典类AttrDict，实现大小写不敏感的字典功能
class AttrDict(dict):
    '''
    A case-insensitive dictionary with access via item, attribute, and call
    notations:

        >>> from scipy.io._idl import AttrDict
        >>> d = AttrDict()
        >>> d['Variable'] = 123
        >>> d['Variable']
        123
        >>> d.Variable
        123
        >>> d.variable
        123
        >>> d('VARIABLE')
        123
        >>> d['missing']
        Traceback (most recent error last):
        ...
        KeyError: 'missing'
        >>> d.missing
        Traceback (most recent error last):
        ...
        AttributeError: 'AttrDict' object has no attribute 'missing'
    '''

    # 初始化方法，接受一个初始字典init作为参数
    def __init__(self, init={}):
        # 调用父类dict的初始化方法，将init字典转换为小写键的形式存储
        dict.__init__(self, init)

    # 重写__getitem__方法，通过小写键获取字典中的值
    def __getitem__(self, name):
        return super().__getitem__(name.lower())

    # 重写__setitem__方法，通过小写键设置字典中的值
    def __setitem__(self, key, value):
        return super().__setitem__(key.lower(), value)
    # 当访问不存在的属性时，尝试调用 __getitem__ 方法获取属性值
    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        # 如果属性不存在，抛出 AttributeError 异常，提供详细的错误信息
        except KeyError:
            raise AttributeError(
                f"'{type(self)}' object has no attribute '{name}'") from None

    # 将 __setitem__ 方法绑定给 __setattr__ 方法，使得属性设置操作调用 __setitem__ 方法
    __setattr__ = __setitem__
    # 将 __getitem__ 方法绑定给 __call__ 方法，使得对象可以被调用，调用时调用 __getitem__ 方法
    __call__ = __getitem__
    # 初始化记录和变量容器
    records = []

    # 如果 python_dict 为 True 或者 idict 不为 None，则使用标准的 Python 字典存储变量
    if python_dict or idict:
        variables = {}
    else:
        # 否则使用具有项目、属性和调用访问权限的不区分大小写的字典存储变量
        variables = AttrDict()

    # 打开 IDL 文件以二进制只读模式
    f = open(file_name, 'rb')

    # 读取文件的签名，预期为 'SR'
    signature = _read_bytes(f, 2)
    # 检查文件签名是否为 b'SR'，如果不是则抛出异常
    if signature != b'SR':
        raise Exception("Invalid SIGNATURE: %s" % signature)

    # 读取记录格式，通常为 b'\x00\x04' 表示普通 .sav 文件，b'\x00\x06' 表示压缩 .sav 文件
    recfmt = _read_bytes(f, 2)

    # 处理普通 .sav 文件
    if recfmt == b'\x00\x04':
        pass

    # 处理压缩 .sav 文件
    elif recfmt == b'\x00\x06':

        # 如果 verbose 参数为 True，则打印信息表示文件是压缩的
        if verbose:
            print("IDL Save file is compressed")

        # 如果提供了未压缩文件名，则将解压后的数据写入该文件中，否则创建一个临时文件
        if uncompressed_file_name:
            fout = open(uncompressed_file_name, 'w+b')
        else:
            fout = tempfile.NamedTemporaryFile(suffix='.sav')

        # 如果 verbose 参数为 True，则打印信息表示解压后的文件名
        if verbose:
            print(" -> expanding to %s" % fout.name)

        # 写入文件头部信息 'SR\x00\x04'
        fout.write(b'SR\x00\x04')

        # 循环处理记录
        while True:

            # 读取记录类型
            rectype = _read_long(f)
            fout.write(struct.pack('>l', int(rectype)))

            # 读取下一个记录位置，并将其作为 int 返回
            nextrec = _read_uint32(f)
            nextrec += _read_uint32(f).astype(np.int64) * 2**32

            # 读取未知的 4 字节数据
            unknown = f.read(4)

            # 检查是否到达文件末尾
            if RECTYPE_DICT[rectype] == 'END_MARKER':
                modval = np.int64(2**32)
                fout.write(struct.pack('>I', int(nextrec) % modval))
                fout.write(
                    struct.pack('>I', int((nextrec - (nextrec % modval)) / modval))
                )
                fout.write(unknown)
                break

            # 获取当前位置
            pos = f.tell()

            # 解压记录字符串
            rec_string = zlib.decompress(f.read(nextrec-pos))

            # 计算下一个记录的位置
            nextrec = fout.tell() + len(rec_string) + 12

            # 写出记录
            fout.write(struct.pack('>I', int(nextrec % 2**32)))
            fout.write(struct.pack('>I', int((nextrec - (nextrec % 2**32)) / 2**32)))
            fout.write(unknown)
            fout.write(rec_string)

        # 关闭原始压缩文件
        f.close()

        # 将 f 设置为解压后的文件，并跳过前四个字节
        f = fout
        f.seek(4)

    else:
        # 如果记录格式不是预期的 b'\x00\x04' 或 b'\x00\x06'，则抛出异常
        raise Exception("Invalid RECFMT: %s" % recfmt)

    # 循环处理记录，将其添加到列表中
    while True:
        r = _read_record(f)
        records.append(r)
        # 如果记录中包含 'end' 字段且为 True，则退出循环
        if 'end' in r:
            if r['end']:
                break

    # 关闭文件
    f.close()

    # 查找堆数据变量
    heap = {}
    for r in records:
        if r['rectype'] == "HEAP_DATA":
            heap[r['heap_index']] = r['data']

    # 查找所有变量
    for r in records:
        if r['rectype'] == "VARIABLE":
            # 替换变量中的堆数据引用为实际数据
            replace, new = _replace_heap(r['data'], heap)
            if replace:
                r['data'] = new
            # 将变量名转换为小写并存储到 variables 字典中
            variables[r['varname'].lower()] = r['data']
    if verbose:
        # 如果 verbose 参数为 True，执行以下代码块

        # 打印文件的时间戳信息
        for record in records:
            if record['rectype'] == "TIMESTAMP":
                # 打印分隔线
                print("-"*50)
                # 打印日期信息
                print("Date: %s" % record['date'])
                # 打印用户信息
                print("User: %s" % record['user'])
                # 打印主机信息
                print("Host: %s" % record['host'])
                # 中断循环
                break

        # 打印文件的版本信息
        for record in records:
            if record['rectype'] == "VERSION":
                # 打印分隔线
                print("-"*50)
                # 打印文件格式信息
                print("Format: %s" % record['format'])
                # 打印架构信息
                print("Architecture: %s" % record['arch'])
                # 打印操作系统信息
                print("Operating System: %s" % record['os'])
                # 打印IDL版本信息
                print("IDL Version: %s" % record['release'])
                # 中断循环
                break

        # 打印文件的标识信息
        for record in records:
            if record['rectype'] == "IDENTIFICATON":
                # 打印分隔线
                print("-"*50)
                # 打印作者信息
                print("Author: %s" % record['author'])
                # 打印标题信息
                print("Title: %s" % record['title'])
                # 打印ID代码信息
                print("ID Code: %s" % record['idcode'])
                # 中断循环
                break

        # 打印文件中保存的描述信息
        for record in records:
            if record['rectype'] == "DESCRIPTION":
                # 打印分隔线
                print("-"*50)
                # 打印描述信息
                print("Description: %s" % record['description'])
                # 中断循环
                break

        # 打印成功读取的记录总数
        print("-"*50)
        print("Successfully read %i records of which:" %
                                            (len(records)))

        # 创建记录类型的便捷列表
        rectypes = [r['rectype'] for r in records]

        # 打印各个记录类型的数量
        for rt in set(rectypes):
            if rt != 'END_MARKER':
                print(" - %i are of type %s" % (rectypes.count(rt), rt))
        print("-"*50)

        # 如果记录类型列表中包含 'VARIABLE'，打印可用的变量信息
        if 'VARIABLE' in rectypes:
            print("Available variables:")
            for var in variables:
                print(f" - {var} [{type(variables[var])}]")
            print("-"*50)

    if idict:
        # 如果 idict 参数不为空，将变量字典中的内容更新到 idict 中并返回 idict
        for var in variables:
            idict[var] = variables[var]
        return idict
    else:
        # 如果 idict 参数为空，直接返回变量字典 variables
        return variables
```