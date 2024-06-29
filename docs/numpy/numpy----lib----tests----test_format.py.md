# `.\numpy\numpy\lib\tests\test_format.py`

```
# doctest
r''' Test the .npy file format.

Set up:

    >>> import sys   # 导入 sys 模块，用于系统相关操作
    >>> from io import BytesIO   # 导入 BytesIO 类，用于操作二进制数据流
    >>> from numpy.lib import format   # 从 numpy 库中导入 format 模块，用于处理数据格式
    >>>
    >>> scalars = [   # 定义标量类型的列表
    ...     np.uint8,   # 8 位无符号整数
    ...     np.int8,    # 8 位有符号整数
    ...     np.uint16,  # 16 位无符号整数
    ...     np.int16,   # 16 位有符号整数
    ...     np.uint32,  # 32 位无符号整数
    ...     np.int32,   # 32 位有符号整数
    ...     np.uint64,  # 64 位无符号整数
    ...     np.int64,   # 64 位有符号整数
    ...     np.float32, # 单精度浮点数
    ...     np.float64, # 双精度浮点数
    ...     np.complex64,   # 复数，使用单精度浮点数表示
    ...     np.complex128,  # 复数，使用双精度浮点数表示
    ...     object,     # Python 对象类型
    ... ]
    >>>
    >>> basic_arrays = []   # 定义空列表用于存储基本数组
    >>>
    >>> for scalar in scalars:   # 遍历标量类型列表
    ...     for endian in '<>':   # 遍历大小端字符 '<' 和 '>'
    ...         dtype = np.dtype(scalar).newbyteorder(endian)   # 创建指定字节顺序的数据类型对象
    ...         basic = np.arange(15).astype(dtype)   # 生成一个基本数组，使用指定数据类型
    ...         basic_arrays.extend([   # 将不同的基本数组类型扩展到基本数组列表中
    ...             np.array([], dtype=dtype),   # 空数组
    ...             np.array(10, dtype=dtype),   # 包含一个元素 10 的数组
    ...             basic,   # 原始数组
    ...             basic.reshape((3,5)),   # 重塑为 3x5 的数组
    ...             basic.reshape((3,5)).T,   # 转置后的 5x3 数组
    ...             basic.reshape((3,5))[::-1,::2],   # 切片操作后的数组
    ...         ])
    ...
    >>>
    >>> Pdescr = [   # 定义结构化数组的描述符列表
    ...     ('x', 'i4', (2,)),   # 字段名为 'x'，数据类型为 int32，形状为 (2,)
    ...     ('y', 'f8', (2, 2)),   # 字段名为 'y'，数据类型为 float64，形状为 (2,2)
    ...     ('z', 'u1')]   # 字段名为 'z'，数据类型为 uint8
    >>>
    >>>
    >>> PbufferT = [   # 定义缓冲区类型列表
    ...     ([3,2], [[6.,4.],[6.,4.]], 8),   # 包含多个元素的元组
    ...     ([4,3], [[7.,5.],[7.,5.]], 9),   # 包含多个元素的元组
    ...     ]
    >>>
    >>>
    >>> Ndescr = [   # 定义结构化数组的描述符列表
    ...     ('x', 'i4', (2,)),   # 字段名为 'x'，数据类型为 int32，形状为 (2,)
    ...     ('Info', [   # 字段名为 'Info'，包含多个子字段的列表
    ...         ('value', 'c16'),   # 子字段 'value'，数据类型为 complex128
    ...         ('y2', 'f8'),   # 子字段 'y2'，数据类型为 float64
    ...         ('Info2', [   # 子字段 'Info2'，包含多个子字段的列表
    ...             ('name', 'S2'),   # 子字段 'name'，数据类型为 string，长度为 2
    ...             ('value', 'c16', (2,)),   # 子字段 'value'，数据类型为 complex128，形状为 (2,)
    ...             ('y3', 'f8', (2,)),   # 子字段 'y3'，数据类型为 float64，形状为 (2,)
    ...             ('z3', 'u4', (2,))]),   # 子字段 'z3'，数据类型为 uint32，形状为 (2,)
    ...         ('name', 'S2'),   # 子字段 'name'，数据类型为 string，长度为 2
    ...         ('z2', 'b1')]),   # 子字段 'z2'，数据类型为 boolean
    ...     ('color', 'S2'),   # 字段名为 'color'，数据类型为 string，长度为 2
    ...     ('info', [   # 字段名为 'info'，包含多个子字段的列表
    ...         ('Name', 'U8'),   # 子字段 'Name'，数据类型为 Unicode 字符串，长度为 8
    ...         ('Value', 'c16')]),   # 子字段 'Value'，数据类型为 complex128
    ...     ('y', 'f8', (2, 2)),   # 字段名为 'y'，数据类型为 float64，形状为 (2,2)
    ...     ('z', 'u1')]   # 字段名为 'z'，数据类型为 uint8
    >>>
    >>>
    >>> NbufferT = [   # 定义缓冲区类型列表
    ...     ([3,2], (6j, 6., ('nn', [6j,4j], [6.,4.], [1,2]), 'NN', True), 'cc', ('NN', 6j), [[6.,4.],[6.,4.]], 8),   # 包含多个元素的元组
    ...     ([4,3], (7j, 7., ('oo', [7j,5j], [7.,5.], [2,1]), 'OO', False), 'dd', ('OO', 7j), [[7.,5.],[7.,5.]], 9),   # 包含多个元素的元组
    ...     ]
    >>>
    >>>
    >>> record_arrays = [   # 定义记录数组列表
    ...     np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('<')),   # 使用指定描述符创建小端序的结构化数组
    ...     np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('<')),   # 使用指定描述符创建小端序的结构化数组
    ...     np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('>')),   # 使用指定描述符创建大端序的结构化数组
    ...     np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('>')),   # 使用指定描述符创建大端序的结构化数组
    ... ]

Test the magic string writing.

    >>> format.magic(1, 0)   # 测试生成魔术字符串，版本号 (1, 0)
    '\x93NUMPY\x01\x00'
    >>> format.magic(0, 0)   # 测试生成魔术字符串，版本号 (0, 0)
    '\x93NUMPY\x00\x00'
    >>> format.magic(255, 255)   # 测试生成魔术字符串，版本号 (255, 255)
    '\x93NUMPY\xff\xff'
    >>> format.magic(2, 5)   # 测试生成魔术字符串，版本号 (2, 5)
    '\x93NUMPY\x02\x05'

Test the magic string reading.

    >>> format.read_magic(BytesIO(format.magic(1, 0)))   # 测试读取魔术字符串，版本号 (1, 0)
    (1, 0)
    >>> format.read_magic(BytesIO(format.magic(0, 0)))   # 测试读取魔术字符串，版本号 (0,
    # 使用 format 模块中的 read_magic 函数来读取通过 magic 函数生成的魔术数元组
    >>> format.read_magic(BytesIO(format.magic(2, 5)))
    (2, 5)
# 测试头部写入功能

# 对于每个数组 arr，包括 basic_arrays 和 record_arrays 中的数组
for arr in basic_arrays + record_arrays:
    # 创建一个字节流对象 f
    f = BytesIO()
    # 调用 format.write_array_header_1_0 方法，将数组 arr 的头部信息写入字节流 f
    format.write_array_header_1_0(f, arr)   # XXX: arr is not a dict, items gets called on it
    # 打印字节流 f 的值的可打印表示
    print(repr(f.getvalue()))


这段代码的作用是测试头部写入功能，对给定的数组列表（basic_arrays 和 record_arrays）中的每个数组 arr，调用 `format.write_array_header_1_0` 方法将其头部信息写入一个字节流对象，并打印字节流对象的可打印表示。
    # 下面是一系列字符串，每个字符串描述了一个NumPy数组的数据类型和布局信息
    # 字符串格式为 "F\x00{'descr': '类型描述符', 'fortran_order': 是否Fortran顺序, 'shape': 数组形状}"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (15,)}             \n"
    # 下面是一系列字符串，每个字符串描述了一个 NumPy 数组的数据类型和形状
    # 这些字符串看起来是以某种格式保存的元数据信息，描述了数组的 dtype、存储顺序（是否按 Fortran 顺序存储）、形状等信息
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<f4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (15,)}             \n"
    # 下面是一系列字符串，每个字符串描述了一个 NumPy 数组的数据类型和形状
    # 这些字符串看起来像是从某种格式中提取的元数据
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>f4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<f8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>f8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<c8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>c8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (0,)}             \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': ()}               \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (15,)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (3, 5)}           \n"
    "F\x00{'descr': '<c16', 'fortran_order': True, 'shape': (5, 3)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (3, 3)}           \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (0,)}             \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': ()}               \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (15,)}            \n"
    # 定义多行字符串，包含了多个 NumPy dtype 的描述信息，每行描述了一个 dtype 的结构
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (3, 5)}           \n"
    "F\x00{'descr': '>c16', 'fortran_order': True, 'shape': (5, 3)}            \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (3, 3)}           \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "v\x00{'descr': [('x', '<i4', (2,)), ('y', '<f8', (2, 2)), ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}         \n"
    "\x16\x02{'descr': [('x', '<i4', (2,)),\n           ('Info',\n            [('value', '<c16'),\n             ('y2', '<f8'),\n             ('Info2',\n              [('name', '|S2'),\n               ('value', '<c16', (2,)),\n               ('y3', '<f8', (2,)),\n               ('z3', '<u4', (2,))]),\n             ('name', '|S2'),\n             ('z2', '|b1')]),\n           ('color', '|S2'),\n           ('info', [('Name', '<U8'), ('Value', '<c16')]),\n           ('y', '<f8', (2, 2)),\n           ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}      \n"
    "v\x00{'descr': [('x', '>i4', (2,)), ('y', '>f8', (2, 2)), ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}         \n"
    "\x16\x02{'descr': [('x', '>i4', (2,)),\n           ('Info',\n            [('value', '>c16'),\n             ('y2', '>f8'),\n             ('Info2',\n              [('name', '|S2'),\n               ('value', '>c16', (2,)),\n               ('y3', '>f8', (2,)),\n               ('z3', '>u4', (2,))]),\n             ('name', '|S2'),\n             ('z2', '|b1')]),\n           ('color', '|S2'),\n           ('info', [('Name', '>U8'), ('Value', '>c16')]),\n           ('y', '>f8', (2, 2)),\n           ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}      \n"
'''
# 导入所需的库和模块
import sys                  # 导入sys模块，用于系统相关操作
import os                   # 导入os模块，用于操作系统相关功能
import warnings             # 导入warnings模块，用于处理警告
import pytest               # 导入pytest模块，用于编写和运行测试用例
from io import BytesIO      # 从io模块中导入BytesIO类，用于操作二进制数据流

import numpy as np          # 导入NumPy库，并用np作为别名
from numpy.testing import (
    assert_, assert_array_equal, assert_raises, assert_raises_regex,
    assert_warns, IS_PYPY, IS_WASM
    )                       # 从numpy.testing模块导入多个断言函数和变量
from numpy.testing._private.utils import requires_memory  # 导入requires_memory函数
from numpy.lib import format  # 导入format模块中的函数和类


# Generate some basic arrays to test with.
# 生成一些基本的数组用于测试
scalars = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    object,
]
basic_arrays = []
for scalar in scalars:
    for endian in '<>':
        dtype = np.dtype(scalar).newbyteorder(endian)
        basic = np.arange(1500).astype(dtype)
        basic_arrays.extend([
            # Empty
            np.array([], dtype=dtype),  # 创建空数组，指定dtype
            # Rank-0
            np.array(10, dtype=dtype),  # 创建标量数组
            # 1-D
            basic,  # 创建一维数组
            # 2-D C-contiguous
            basic.reshape((30, 50)),  # 创建C顺序的二维数组
            # 2-D F-contiguous
            basic.reshape((30, 50)).T,  # 创建Fortran顺序的二维数组
            # 2-D non-contiguous
            basic.reshape((30, 50))[::-1, ::2],  # 创建非连续的二维数组
        ])

# More complicated record arrays.
# 更复杂的记录数组

# Structure of a plain array description:
# 简单数组描述的结构
Pdescr = [
    ('x', 'i4', (2,)),   # 元组结构描述字段名为'x'，数据类型为'i4'（32位整数），形状为(2,)
    ('y', 'f8', (2, 2)),  # 元组结构描述字段名为'y'，数据类型为'f8'（64位浮点数），形状为(2, 2)
    ('z', 'u1')          # 元组结构描述字段名为'z'，数据类型为'u1'（无符号字节）
]

# A plain list of tuples with values for testing:
# 用于测试的元组值的简单列表
PbufferT = [
    # x     y                  z
    ([3, 2], [[6., 4.], [6., 4.]], 8),  # 包含测试数据的元组列表
    ([4, 3], [[7., 5.], [7., 5.]], 9),
    ]


# This is the structure of the table used for nested objects (DON'T PANIC!):
# 这是用于嵌套对象的表的结构（不要惊慌！）

# The corresponding nested array description:
# 相应的嵌套数组描述
Ndescr = [
    ('x', 'i4', (2,)),   # 元组结构描述字段名为'x'，数据类型为'i4'（32位整数），形状为(2,)
    ('Info', [
        ('value', 'c16'),  # Info字段下的value，数据类型为'c16'（复数浮点数）
        ('y2', 'f8'),      # Info字段下的y2，数据类型为'f8'（64位浮点数）
        ('Info2', [
            ('name', 'S2'),  # Info2字段下的name，数据类型为'S2'（字符串，长度为2）
            ('value', 'c16', (2,)),  # Info2字段下的value，数据类型为'c16'（复数浮点数），形状为(2,)
            ('y3', 'f8', (2,)),      # Info2字段下的y3，数据类型为'f8'（64位浮点数），形状为(2,)
            ('z3', 'u4', (2,))]),   # Info2字段下的z3，数据类型为'u4'（32位无符号整数），形状为(2,)
        ('name', 'S2'),     # Info字段下的name，数据类型为'S2'（字符串，长度为2）
        ('z2', 'b1')]),     # Info字段下的z2，数据类型为'b1'（布尔值，占1位）
    ('color', 'S2'),       # 元组结构描述字段名为'color'，数据类型为'S2'（字符串，长度为2）
    ('info', [
        ('Name', 'U8'),    # info字段下的Name，数据类型为'U8'（Unicode字符串，最大长度为8）
        ('Value', 'c16')]), # info字段下的Value，数据类型为'c16'（复数浮点数）
    ('y', 'f8', (2, 2)),   # 元组结构描述字段名为'y'，数据类型为'f8'（64位浮点数），形状为(2, 2)
    ('z', 'u1')            # 元组结构描述字段名为'z'，数据类型为'u1'（无符号字节）
]

NbufferT = [
    # x     Info                                                color info        y                  z
    #       value y2 Info2                            name z2         Name Value
    #                name   value    y3       z3
    ([3, 2], (6j, 6., ('nn', [6j, 4j], [6., 4.], [1, 2]), 'NN', True),
     'cc', ('NN', 6j), [[6., 4.], [6., 4.]], 8),  # 包含测试数据的元组列表
    # 定义一个包含复杂结构的元组
    ([
        # 列表 [4, 3]
        4, 3
    ],
        # 元组 (7j, 7., ('oo', [7j, 5j], [7., 5.], [2, 1]), 'OO', False)
        (
            # 复数 7j
            7j,
            # 浮点数 7.0
            7.0,
            # 元组 ('oo', [7j, 5j], [7., 5.], [2, 1])
            (
                # 字符串 'oo'
                'oo',
                # 列表 [7j, 5j]
                [7j, 5j],
                # 列表 [7.0, 5.0]
                [7.0, 5.0],
                # 列表 [2, 1]
                [2, 1]
            ),
            # 字符串 'OO'
            'OO',
            # 布尔值 False
            False
        ),
        # 字符串 'dd'
        'dd',
        # 元组 ('OO', 7j)
        ('OO', 7j),
        # 列表 [[7.0, 5.0], [7.0, 5.0]]
        [
            [7.0, 5.0],
            [7.0, 5.0]
        ],
        # 整数 9
        9
    )
# 定义包含多个记录数组的列表
record_arrays = [
    # 创建一个新的 NumPy 数组，使用小端序的 Pdescr 数据类型
    np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('<')),
    # 创建一个新的 NumPy 数组，使用小端序的 Ndescr 数据类型
    np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('<')),
    # 创建一个新的 NumPy 数组，使用大端序的 Pdescr 数据类型
    np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('>')),
    # 创建一个新的 NumPy 数组，使用大端序的 Ndescr 数据类型
    np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('>')),
    # 创建一个包含单个零元素的 NumPy 结构化数组
    np.zeros(1, dtype=[('c', ('<f8', (5,)), (2,))])
]


# 继承自 BytesIO 的类，随机读取不同大小的字节流
class BytesIOSRandomSize(BytesIO):
    def read(self, size=None):
        import random
        # 随机选择读取的字节数量，范围在1到size之间
        size = random.randint(1, size)
        return super().read(size)


def roundtrip(arr):
    # 创建一个新的字节流对象
    f = BytesIO()
    # 将数组 arr 写入字节流 f 中
    format.write_array(f, arr)
    # 创建一个新的字节流对象，将 f 的内容作为初始值
    f2 = BytesIO(f.getvalue())
    # 从字节流 f2 中读取数组数据，允许使用 pickle
    arr2 = format.read_array(f2, allow_pickle=True)
    return arr2


def roundtrip_randsize(arr):
    # 创建一个新的字节流对象
    f = BytesIO()
    # 将数组 arr 写入字节流 f 中
    format.write_array(f, arr)
    # 创建一个继承自 BytesIO 的 BytesIOSRandomSize 对象，将 f 的内容作为初始值
    f2 = BytesIOSRandomSize(f.getvalue())
    # 从字节流 f2 中读取数组数据
    arr2 = format.read_array(f2)
    return arr2


def roundtrip_truncated(arr):
    # 创建一个新的字节流对象
    f = BytesIO()
    # 将数组 arr 写入字节流 f 中
    format.write_array(f, arr)
    # 创建一个新的字节流对象，其内容比 f 的内容少一个字节
    f2 = BytesIO(f.getvalue()[0:-1])
    # 从字节流 f2 中读取数组数据
    arr2 = format.read_array(f2)
    return arr2


def assert_equal_(o1, o2):
    # 断言 o1 和 o2 相等
    assert_(o1 == o2)


def test_roundtrip():
    # 遍历基本数组和记录数组
    for arr in basic_arrays + record_arrays:
        # 将数组 arr 进行序列化和反序列化，并断言原数组和反序列化后的数组相等
        arr2 = roundtrip(arr)
        assert_array_equal(arr, arr2)


def test_roundtrip_randsize():
    # 遍历基本数组和记录数组
    for arr in basic_arrays + record_arrays:
        # 如果数组的数据类型不是 object，则进行随机大小字节流的序列化和反序列化，并断言结果相等
        if arr.dtype != object:
            arr2 = roundtrip_randsize(arr)
            assert_array_equal(arr, arr2)


def test_roundtrip_truncated():
    # 遍历基本数组
    for arr in basic_arrays:
        # 如果数组的数据类型不是 object，则测试字节流长度比实际需要少一个字节的情况，预期抛出 ValueError 异常
        if arr.dtype != object:
            assert_raises(ValueError, roundtrip_truncated, arr)


def test_long_str():
    # 检查长于内部缓冲区大小的数组元素，gh-4027
    long_str_arr = np.ones(1, dtype=np.dtype((str, format.BUFFER_SIZE + 1)))
    # 将长字符串数组进行序列化和反序列化，并断言原数组和反序列化后的数组相等
    long_str_arr2 = roundtrip(long_str_arr)
    assert_array_equal(long_str_arr, long_str_arr2)


# 标记为跳过测试，在 WASM 环境下 memmap 不工作正常
@pytest.mark.skipif(IS_WASM, reason="memmap doesn't work correctly")
# 标记为慢速测试
@pytest.mark.slow
def test_memmap_roundtrip(tmpdir):
    # 对于 basic_arrays 和 record_arrays 合并后的数组进行迭代，同时获取索引 i 和数组 arr
    for i, arr in enumerate(basic_arrays + record_arrays):
        # 如果数组的数据类型包含对象类型，跳过这些数组，因为无法进行内存映射
        if arr.dtype.hasobject:
            # 跳过这些数组，因为无法使用内存映射处理它们
            continue
        
        # 将普通和内存映射方式分别写入文件
        nfn = os.path.join(tmpdir, f'normal{i}.npy')  # 普通方式存储的文件路径
        mfn = os.path.join(tmpdir, f'memmap{i}.npy')  # 内存映射方式存储的文件路径
        
        # 使用普通方式打开文件，并将数组 arr 写入其中
        with open(nfn, 'wb') as fp:
            format.write_array(fp, arr)
        
        # 检查数组的存储顺序是否是 Fortran 风格
        fortran_order = (
            arr.flags.f_contiguous and not arr.flags.c_contiguous)
        
        # 使用内存映射方式打开文件，并将数组 arr 写入其中
        ma = format.open_memmap(mfn, mode='w+', dtype=arr.dtype,
                                shape=arr.shape, fortran_order=fortran_order)
        ma[...] = arr  # 将数组 arr 写入内存映射文件中
        ma.flush()  # 刷新内存映射文件，确保数据被写入
        
        # 检查普通和内存映射方式存储的文件内容是否一致
        with open(nfn, 'rb') as fp:
            normal_bytes = fp.read()  # 读取普通文件的内容
        with open(mfn, 'rb') as fp:
            memmap_bytes = fp.read()  # 读取内存映射文件的内容
        assert_equal_(normal_bytes, memmap_bytes)  # 断言普通文件和内存映射文件内容相等
        
        # 检查通过内存映射方式读取文件的正确性
        ma = format.open_memmap(nfn, mode='r')  # 使用内存映射方式打开普通文件
        ma.flush()  # 刷新内存映射文件，确保数据可用
# 定义一个带有参数的测试函数，用于测试压缩数据的往返过程
def test_compressed_roundtrip(tmpdir):
    # 创建一个形状为 (200, 200) 的随机数组
    arr = np.random.rand(200, 200)
    # 在临时目录下创建一个名为 'compressed.npz' 的压缩文件
    npz_file = os.path.join(tmpdir, 'compressed.npz')
    # 将数组 arr 保存为压缩的 npz 文件
    np.savez_compressed(npz_file, arr=arr)
    # 使用 np.load 加载压缩的 npz 文件，并将其赋值给变量 npz
    with np.load(npz_file) as npz:
        # 从 npz 中读取键为 'arr' 的数组，赋值给 arr1
        arr1 = npz['arr']
    # 断言数组 arr 和 arr1 相等
    assert_array_equal(arr, arr1)

# 创建一个数据类型对象 dt1，包含一个字节对齐的结构
dt1 = np.dtype('i1, i4, i1', align=True)
# 创建一个数据类型对象 dt2，包含非对齐和显式偏移的结构
dt2 = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'i4'],
                'offsets': [1, 6]})
# 创建一个数据类型对象 dt3，包含嵌套的结构内结构
dt3 = np.dtype({'names': ['c', 'd'], 'formats': ['i4', dt2]})
# 创建一个数据类型对象 dt4，包含空字符串命名的字段
dt4 = np.dtype({'names': ['a', '', 'b'], 'formats': ['i4']*3})
# 创建一个数据类型对象 dt5，包含标题信息的结构
dt5 = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'i4'],
                'offsets': [1, 6], 'titles': ['aa', 'bb']})
# 创建一个数据类型对象 dt6，空的结构
dt6 = np.dtype({'names': [], 'formats': [], 'itemsize': 8})

# 使用 pytest.mark.parametrize 装饰器，对 test_load_padded_dtype 函数参数化
@pytest.mark.parametrize("dt", [dt1, dt2, dt3, dt4, dt5, dt6])
# 定义测试加载填充数据类型的函数，使用临时目录和参数化的数据类型 dt
def test_load_padded_dtype(tmpdir, dt):
    # 创建一个包含三个元素的数组 arr，数据类型为 dt
    arr = np.zeros(3, dt)
    for i in range(3):
        # 将 arr 的每个元素设置为 i + 5
        arr[i] = i + 5
    # 在临时目录下创建一个名为 'aligned.npz' 的 npz 文件
    npz_file = os.path.join(tmpdir, 'aligned.npz')
    # 将数组 arr 保存为 npz 文件
    np.savez(npz_file, arr=arr)
    # 使用 np.load 加载 npz 文件，并将其赋值给变量 npz
    with np.load(npz_file) as npz:
        # 从 npz 中读取键为 'arr' 的数组，赋值给 arr1
        arr1 = npz['arr']
    # 断言数组 arr 和 arr1 相等
    assert_array_equal(arr, arr1)

# 使用 pytest.mark.skipif 装饰器，根据条件跳过测试
@pytest.mark.skipif(sys.version_info >= (3, 12), reason="see gh-23988")
# 使用 pytest.mark.xfail 装饰器，标记测试为预期失败
@pytest.mark.xfail(IS_WASM, reason="Emscripten NODEFS has a buggy dup")
# 定义测试 Python 2 和 Python 3 互操作性的函数
def test_python2_python3_interoperability():
    # 定义文件名 fname
    fname = 'win64python2.npy'
    # 组合路径，找到包含 fname 的文件路径
    path = os.path.join(os.path.dirname(__file__), 'data', fname)
    # 使用 pytest.warns 检查是否有 UserWarning，匹配包含 "Reading.*this warning." 的警告
    with pytest.warns(UserWarning, match="Reading.*this warning\\."):
        # 加载路径 path 下的数据，赋值给 data
        data = np.load(path)
    # 断言 data 和一个包含两个值为 1 的数组相等
    assert_array_equal(data, np.ones(2))

# 定义测试 Python 2 和 Python 3 之间 pickle 互操作性的函数
def test_pickle_python2_python3():
    # 测试在 Python 2 和 Python 3 上加载保存的对象数组是否有效
    # 定义数据目录路径
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # 创建一个对象数组 expected，包含多种类型的对象
    expected = np.array([None, range, '\u512a\u826f',
                         b'\xe4\xb8\x8d\xe8\x89\xaf'],
                        dtype=object)
    # 遍历给定的文件名列表
    for fname in ['py2-np0-objarr.npy', 'py2-objarr.npy', 'py2-objarr.npz',
                  'py3-objarr.npy', 'py3-objarr.npz']:
        # 构建文件的完整路径
        path = os.path.join(data_dir, fname)

        # 遍历两种编码方式：bytes 和 latin1
        for encoding in ['bytes', 'latin1']:
            # 使用指定编码方式加载 NumPy 数据文件
            data_f = np.load(path, allow_pickle=True, encoding=encoding)

            # 如果文件名以 '.npz' 结尾，提取其中的 'x' 数据
            if fname.endswith('.npz'):
                data = data_f['x']
                data_f.close()
            else:
                data = data_f

            # 根据不同的编码方式和文件名前缀进行断言检查
            if encoding == 'latin1' and fname.startswith('py2'):
                # 对于 Latin1 编码和以 'py2' 开头的文件名，进行断言
                assert_(isinstance(data[3], str))
                assert_array_equal(data[:-1], expected[:-1])
                # 验证是否存在乱码
                assert_array_equal(data[-1].encode(encoding), expected[-1])
            else:
                # 对于其他情况，检查数据类型和整体数组是否匹配
                assert_(isinstance(data[3], bytes))
                assert_array_equal(data, expected)

        # 如果文件名以 'py2' 开头
        if fname.startswith('py2'):
            # 如果文件名以 '.npz' 结尾
            if fname.endswith('.npz'):
                # 加载数据文件，并断言 UnicodeError 异常
                data = np.load(path, allow_pickle=True)
                assert_raises(UnicodeError, data.__getitem__, 'x')
                data.close()
                # 加载数据文件，并断言 ImportError 异常，指定 Latin1 编码
                data = np.load(path, allow_pickle=True, fix_imports=False,
                               encoding='latin1')
                assert_raises(ImportError, data.__getitem__, 'x')
                data.close()
            else:
                # 对于非 '.npz' 结尾的文件名，断言 UnicodeError 异常
                assert_raises(UnicodeError, np.load, path,
                              allow_pickle=True)
                # 对于非 '.npz' 结尾的文件名，断言 ImportError 异常，指定 Latin1 编码
                assert_raises(ImportError, np.load, path,
                              allow_pickle=True, fix_imports=False,
                              encoding='latin1')
# 定义一个测试函数，用于测试禁止使用 pickle 时的情况
def test_pickle_disallow(tmpdir):
    # 指定数据目录为当前文件目录下的 'data' 子目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # 测试对于包含对象数组的 numpy 文件，禁止 pickle 序列化时是否引发 ValueError
    path = os.path.join(data_dir, 'py2-objarr.npy')
    assert_raises(ValueError, np.load, path,
                  allow_pickle=False, encoding='latin1')

    # 测试对于包含对象数组的 numpy 压缩文件，禁止 pickle 序列化时是否引发 ValueError
    path = os.path.join(data_dir, 'py2-objarr.npz')
    with np.load(path, allow_pickle=False, encoding='latin1') as f:
        assert_raises(ValueError, f.__getitem__, 'x')

    # 测试禁止 pickle 时保存 numpy 数组是否引发 ValueError
    path = os.path.join(tmpdir, 'pickle-disabled.npy')
    assert_raises(ValueError, np.save, path, np.array([None], dtype=object),
                  allow_pickle=False)

# 标记化测试，测试不同的 dtype 的转换
@pytest.mark.parametrize('dt', [
    # 第一种 dtype
    np.dtype(np.dtype([('a', np.int8),
                       ('b', np.int16),
                       ('c', np.int32),
                      ], align=True),
             (3,)),
    # 第二种 dtype
    np.dtype([('x', np.dtype({'names':['a','b'],
                              'formats':['i1','i1'],
                              'offsets':[0,4],
                              'itemsize':8,
                             },
                    (3,)),
               (4,),
             )]),
    # 第三种 dtype
    np.dtype([('x',
                   ('<f8', (5,)),
                   (2,),
               )]),
    # 第四种 dtype
    np.dtype([('x', np.dtype((
        np.dtype((
            np.dtype({'names':['a','b'],
                      'formats':['i1','i1'],
                      'offsets':[0,4],
                      'itemsize':8}),
            (3,)
            )),
        (4,)
        )))
        ]),
    # 第五种 dtype
    np.dtype([
        ('a', np.dtype((
            np.dtype((
                np.dtype((
                    np.dtype([
                        ('a', int),
                        ('b', np.dtype({'names':['a','b'],
                                        'formats':['i1','i1'],
                                        'offsets':[0,4],
                                        'itemsize':8})),
                    ]),
                    (3,),
                )),
                (4,),
            )),
            (5,),
        )))
        ]),
    ])

# 测试函数，将描述转换为 dtype，并检查转换后的结果是否与原始描述相等
def test_descr_to_dtype(dt):
    dt1 = format.descr_to_dtype(dt.descr)
    assert_equal_(dt1, dt)
    arr1 = np.zeros(3, dt)
    arr2 = roundtrip(arr1)
    assert_array_equal(arr1, arr2)

# 测试函数，版本 2.0 的特性
def test_version_2_0():
    f = BytesIO()
    # 创建一个需要超过 2 字节头部的数据类型
    dt = [(("%d" % i) * 100, float) for i in range(500)]
    d = np.ones(1000, dtype=dt)

    # 使用版本 2.0 写入数组到字节流中
    format.write_array(f, d, version=(2, 0))
    
    # 检查是否生成 UserWarning 警告
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        format.write_array(f, d)
        assert_(w[0].category is UserWarning)

    # 检查数据部分的对齐
    f.seek(0)
    header = f.readline()
    assert_(len(header) % format.ARRAY_ALIGN == 0)

    # 读取写入的数组数据并与原始数组比较
    f.seek(0)
    n = format.read_array(f, max_header_size=200000)
    assert_array_equal(d, n)

    # 请求版本 1.0 但数据无法以此方式保存时，检查是否引发 ValueError
    assert_raises(ValueError, format.write_array, f, d, (1, 0))
@pytest.mark.skipif(IS_WASM, reason="memmap doesn't work correctly")
# 定义一个测试函数，如果在WebAssembly环境下，跳过此测试
def test_version_2_0_memmap(tmpdir):
    # requires more than 2 byte for header
    # 创建一个复杂的数据类型列表，每个元素包含一个100位的数字字符串和一个浮点数
    dt = [(("%d" % i) * 100, float) for i in range(500)]
    # 创建一个包含500个这种复杂类型的数组
    d = np.ones(1000, dtype=dt)
    # 生成两个临时文件路径
    tf1 = os.path.join(tmpdir, f'version2_01.npy')
    tf2 = os.path.join(tmpdir, f'version2_02.npy')

    # 1.0 requested but data cannot be saved this way
    # 断言，试图以版本1.0保存数据会引发值错误异常
    assert_raises(ValueError, format.open_memmap, tf1, mode='w+', dtype=d.dtype,
                            shape=d.shape, version=(1, 0))

    # 创建一个内存映射数组对象，以写入模式打开，版本为2.0
    ma = format.open_memmap(tf1, mode='w+', dtype=d.dtype,
                            shape=d.shape, version=(2, 0))
    # 将数据写入内存映射数组
    ma[...] = d
    # 刷新数据到磁盘
    ma.flush()
    # 重新打开内存映射数组以只读模式，并指定较大的最大头部大小
    ma = format.open_memmap(tf1, mode='r', max_header_size=200000)
    # 断言，重新打开的内存映射数组与原始数据相等
    assert_array_equal(ma, d)

    # 使用警告捕获上下文，检查是否会引发用户警告
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        # 创建一个内存映射数组对象，以写入模式打开，版本为None
        ma = format.open_memmap(tf2, mode='w+', dtype=d.dtype,
                                shape=d.shape, version=None)
        # 断言，捕获的第一个警告是用户警告
        assert_(w[0].category is UserWarning)
        # 将数据写入内存映射数组
        ma[...] = d
        # 刷新数据到磁盘
        ma.flush()

    # 重新打开内存映射数组以只读模式，并指定较大的最大头部大小
    ma = format.open_memmap(tf2, mode='r', max_header_size=200000)
    # 断言，重新打开的内存映射数组与原始数据相等
    assert_array_equal(ma, d)

@pytest.mark.parametrize("mmap_mode", ["r", None])
# 定义一个参数化测试函数，测试内存映射模式为'r'和None的情况
def test_huge_header(tmpdir, mmap_mode):
    # 创建一个具有巨大头部的数组
    f = os.path.join(tmpdir, f'large_header.npy')
    arr = np.array(1, dtype="i,"*10000+"i")

    # 使用警告捕获上下文，检查是否会引发用户警告
    with pytest.warns(UserWarning, match=".*format 2.0"):
        # 保存数组到文件
        np.save(f, arr)
    
    # 断言，加载文件时如果头部太大，会引发值错误异常
    with pytest.raises(ValueError, match="Header.*large"):
        np.load(f, mmap_mode=mmap_mode)

    # 断言，加载文件时如果头部太大，会引发值错误异常，指定最大头部大小为20000
    with pytest.raises(ValueError, match="Header.*large"):
        np.load(f, mmap_mode=mmap_mode, max_header_size=20000)

    # 加载文件，并允许反序列化对象
    res = np.load(f, mmap_mode=mmap_mode, allow_pickle=True)
    # 断言，加载的数组与原始数组相等
    assert_array_equal(res, arr)

    # 加载文件，并指定较大的最大头部大小为180000
    res = np.load(f, mmap_mode=mmap_mode, max_header_size=180000)
    # 断言，加载的数组与原始数组相等
    assert_array_equal(res, arr)

# 定义一个测试函数，测试写入不同版本的数组
def test_write_version():
    # 创建一个字节流对象
    f = BytesIO()
    arr = np.arange(1)
    # 这些应该全部通过。
    # 使用版本1.0写入数组到字节流
    format.write_array(f, arr, version=(1, 0))
    # 写入数组到字节流，默认版本
    format.write_array(f, arr)

    # 使用None版本写入数组到字节流
    format.write_array(f, arr, version=None)
    # 写入数组到字节流，默认版本

    format.write_array(f, arr)

    # 使用版本2.0写入数组到字节流
    format.write_array(f, arr, version=(2, 0))
    # 写入数组到字节流，默认版本

    # 这些应该全部失败。
    # 创建一个不合法的版本列表，包括(1, 1), (0, 0), (0, 1), (2, 2), (255, 255)
    ]
    # 遍历 bad_versions 列表中的每个版本号
    for version in bad_versions:
        # 使用 assert_raises_regex 上下文管理器来捕获 ValueError 异常，并检查错误消息中是否包含特定文本
        with assert_raises_regex(ValueError,
                                 'we only support format version.*'):
            # 调用 format.write_array 函数，传入文件对象 f、数组 arr，以及当前迭代的版本号作为参数
            format.write_array(f, arr, version=version)
# 定义可能导致版本错误的魔术数列表
bad_version_magic = [
    b'\x93NUMPY\x01\x01',
    b'\x93NUMPY\x00\x00',
    b'\x93NUMPY\x00\x01',
    b'\x93NUMPY\x02\x00',
    b'\x93NUMPY\x02\x02',
    b'\x93NUMPY\xff\xff',
]

# 定义可能格式错误的魔术数列表
malformed_magic = [
    b'\x92NUMPY\x01\x00',
    b'\x00NUMPY\x01\x00',
    b'\x93numpy\x01\x00',
    b'\x93MATLB\x01\x00',
    b'\x93NUMPY\x01',
    b'\x93NUMPY',
    b'',
]

# 测试读取魔术数的函数
def test_read_magic():
    # 创建两个字节流对象
    s1 = BytesIO()
    s2 = BytesIO()

    # 创建一个包含浮点数的3x6的NumPy数组
    arr = np.ones((3, 6), dtype=float)

    # 向s1和s2写入不同版本的数组数据
    format.write_array(s1, arr, version=(1, 0))
    format.write_array(s2, arr, version=(2, 0))

    # 将文件指针移至字节流的开头
    s1.seek(0)
    s2.seek(0)

    # 读取并获取s1和s2的版本号
    version1 = format.read_magic(s1)
    version2 = format.read_magic(s2)

    # 断言s1和s2的版本号与预期的一致
    assert_(version1 == (1, 0))
    assert_(version2 == (2, 0))

    # 断言s1和s2的当前位置为魔术数长度
    assert_(s1.tell() == format.MAGIC_LEN)
    assert_(s2.tell() == format.MAGIC_LEN)

# 测试读取带有错误魔术数的情况
def test_read_magic_bad_magic():
    # 对每个错误魔术数进行测试
    for magic in malformed_magic:
        # 创建一个包含错误魔术数的字节流对象
        f = BytesIO(magic)
        # 断言调用读取数组函数时会引发值错误异常
        assert_raises(ValueError, format.read_array, f)

# 测试读取1.0版本错误魔术数的情况
def test_read_version_1_0_bad_magic():
    # 对每个可能导致1.0版本错误的魔术数和错误魔术数进行测试
    for magic in bad_version_magic + malformed_magic:
        # 创建一个包含魔术数或错误魔术数的字节流对象
        f = BytesIO(magic)
        # 断言调用读取数组函数时会引发值错误异常
        assert_raises(ValueError, format.read_array, f)

# 测试不良魔术数参数的情况
def test_bad_magic_args():
    # 断言调用魔术数函数时会引发值错误异常（负数或超过255的参数）
    assert_raises(ValueError, format.magic, -1, 1)
    assert_raises(ValueError, format.magic, 256, 1)
    assert_raises(ValueError, format.magic, 1, -1)
    assert_raises(ValueError, format.magic, 1, 256)

# 测试大型头部的情况
def test_large_header():
    # 创建一个字节流对象
    s = BytesIO()
    # 定义包含头部信息的字典
    d = {'shape': tuple(), 'fortran_order': False, 'descr': '<i8'}
    # 向s写入数组头部信息
    format.write_array_header_1_0(s, d)

    # 重新创建一个字节流对象
    s = BytesIO()
    # 修改头部信息字典的'descr'项，使其包含超长的字符串
    d['descr'] = [('x'*256*256, '<i8')]
    # 断言调用写入数组头部信息函数时会引发值错误异常
    assert_raises(ValueError, format.write_array_header_1_0, s, d)

# 测试读取1.0版本数组头部信息的情况
def test_read_array_header_1_0():
    # 创建一个字节流对象
    s = BytesIO()

    # 创建一个包含浮点数的3x6的NumPy数组
    arr = np.ones((3, 6), dtype=float)
    # 向s写入数组数据
    format.write_array(s, arr, version=(1, 0))

    # 将文件指针移至魔术数后面
    s.seek(format.MAGIC_LEN)
    # 读取并获取数组头部信息
    shape, fortran, dtype = format.read_array_header_1_0(s)

    # 断言当前位置为数组对齐长度的整数倍
    assert_(s.tell() % format.ARRAY_ALIGN == 0)
    # 断言数组头部信息与预期的一致
    assert_((shape, fortran, dtype) == ((3, 6), False, float))

# 测试读取2.0版本数组头部信息的情况
def test_read_array_header_2_0():
    # 创建一个字节流对象
    s = BytesIO()

    # 创建一个包含浮点数的3x6的NumPy数组
    arr = np.ones((3, 6), dtype=float)
    # 向s写入数组数据
    format.write_array(s, arr, version=(2, 0))

    # 将文件指针移至魔术数后面
    s.seek(format.MAGIC_LEN)
    # 读取并获取数组头部信息
    shape, fortran, dtype = format.read_array_header_2_0(s)

    # 断言当前位置为数组对齐长度的整数倍
    assert_(s.tell() % format.ARRAY_ALIGN == 0)
    # 断言数组头部信息与预期的一致
    assert_((shape, fortran, dtype) == ((3, 6), False, float))

# 测试错误头部的情况
def test_bad_header():
    # 断言调用读取1.0版本数组头部信息函数时会引发值错误异常（头部长度小于2）
    s = BytesIO()
    assert_raises(ValueError, format.read_array_header_1_0, s)
    s = BytesIO(b'1')
    assert_raises(ValueError, format.read_array_header_1_0, s)

    # 断言调用读取1.0版本数组头部信息函数时会引发值错误异常（头部短于指定的大小）
    s = BytesIO(b'\x01\x00')
    assert_raises(ValueError, format.read_array_header_1_0, s)

    # 断言调用读取1.0版本数组头部信息函数时会引发值错误异常（缺少所需的确切键）
    s = BytesIO(
        b"\x93NUMPY\x01\x006\x00{'descr': 'x', 'shape': (1, 2), }" +
        b"                    \n"
    )
    # 使用 assert_raises 函数测试调用 format.read_array_header_1_0 函数时是否会引发 ValueError 异常
    assert_raises(ValueError, format.read_array_header_1_0, s)
    
    # 创建一个包含数组头信息的字典 d，包括形状 (1, 2)，顺序为非 Fortran，描述符为 "x"，以及额外的键 "extrakey" 值为 -1
    d = {"shape": (1, 2),
         "fortran_order": False,
         "descr": "x",
         "extrakey": -1}
    
    # 创建一个字节流对象 s
    s = BytesIO()
    
    # 调用 format.write_array_header_1_0 函数，将字典 d 的内容写入字节流 s 中
    format.write_array_header_1_0(s, d)
    
    # 使用 assert_raises 函数测试调用 format.read_array_header_1_0 函数时是否会引发 ValueError 异常，传入的参数是字节流 s
    assert_raises(ValueError, format.read_array_header_1_0, s)
# 测试大文件支持函数，跳过 Windows 和 Cygwin 平台
def test_large_file_support(tmpdir):
    if (sys.platform == 'win32' or sys.platform == 'cygwin'):
        pytest.skip("Unknown if Windows has sparse filesystems")
    
    # 尝试创建一个大的稀疏文件
    tf_name = os.path.join(tmpdir, 'sparse_file')
    try:
        # 使用 subprocess 调用 truncate 命令创建一个大小为 5GB 的文件
        import subprocess as sp
        sp.check_call(["truncate", "-s", "5368709120", tf_name])
    except Exception:
        pytest.skip("Could not create 5GB large file")
    
    # 在文件末尾写入一个小数组
    with open(tf_name, "wb") as f:
        f.seek(5368709120)
        d = np.arange(5)
        np.save(f, d)
    
    # 读取写入的数据
    with open(tf_name, "rb") as f:
        f.seek(5368709120)
        r = np.load(f)
    
    # 断言读取的数据与写入的数据一致
    assert_array_equal(r, d)


# 标记：如果在 PyPy 上运行则跳过
@pytest.mark.skipif(IS_PYPY, reason="flaky on PyPy")
# 标记：如果 intp 类型的大小小于 8 字节则跳过
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8,
                    reason="test requires 64-bit system")
# 标记：慢速测试，需要至少 2GB 空闲内存
@pytest.mark.slow
@requires_memory(free_bytes=2 * 2**30)
# 测试大归档函数
def test_large_archive(tmpdir):
    # 回归测试：测试数组维度乘积超出 int32 范围的保存
    shape = (2**30, 2)
    try:
        a = np.empty(shape, dtype=np.uint8)
    except MemoryError:
        pytest.skip("Could not create large file")

    # 创建大归档文件
    fname = os.path.join(tmpdir, "large_archive")
    with open(fname, "wb") as f:
        np.savez(f, arr=a)

    del a

    # 从文件中读取数组
    with open(fname, "rb") as f:
        new_a = np.load(f)["arr"]

    # 断言新数组的形状与原数组形状一致
    assert new_a.shape == shape


# 测试空的 npz 文件
def test_empty_npz(tmpdir):
    # 测试 gh-9989
    fname = os.path.join(tmpdir, "nothing.npz")
    np.savez(fname)
    with np.load(fname) as nps:
        pass


# 测试 Unicode 字段名
def test_unicode_field_names(tmpdir):
    # gh-7391
    arr = np.array([
        (1, 3),
        (1, 2),
        (1, 3),
        (1, 2)
    ], dtype=[
        ('int', int),
        ('\N{CJK UNIFIED IDEOGRAPH-6574}\N{CJK UNIFIED IDEOGRAPH-5F62}', int)
    ])
    fname = os.path.join(tmpdir, "unicode.npy")
    with open(fname, 'wb') as f:
        format.write_array(f, arr, version=(3, 0))
    with open(fname, 'rb') as f:
        arr2 = format.read_array(f)
    
    # 断言两个数组是否相等
    assert_array_equal(arr, arr2)

    # 通知用户选择了版本 3.0
    with open(fname, 'wb') as f:
        with assert_warns(UserWarning):
            format.write_array(f, arr, version=None)


# 测试 header_growth_axis 函数
def test_header_growth_axis():
    for is_fortran_array, dtype_space, expected_header_length in [
        [False, 22, 128], [False, 23, 192], [True, 23, 128], [True, 24, 192]
        for size in [10**i for i in range(format.GROWTH_AXIS_MAX_DIGITS)]:
            # 创建一个字节流对象
            fp = BytesIO()
            # 调用 format 模块的 write_array_header_1_0 函数，向字节流中写入数组的头信息
            format.write_array_header_1_0(fp, {
                'shape': (2, size) if is_fortran_array else (size, 2),
                'fortran_order': is_fortran_array,
                'descr': np.dtype([(' '*dtype_space, int)])
            })

            # 断言确保写入的头信息长度符合预期
            assert len(fp.getvalue()) == expected_header_length
# 使用 pytest.mark.parametrize 装饰器，定义了测试用例参数化，每组参数包括数据类型 dt 和是否失败标志 fail
@pytest.mark.parametrize('dt, fail', [
    # 第一组参数：定义一个结构化数据类型，包含两个字段 'a' 和 'b'，分别是 float 和长度为 3 的字符串类型
    (np.dtype({'names': ['a', 'b'], 'formats': [float, np.dtype('S3', metadata={'some': 'stuff'})]}), True),
    # 第二组参数：定义一个整数数据类型，并附加 metadata 字典
    (np.dtype(int, metadata={'some': 'stuff'}), False),
    # 第三组参数：定义一个包含子数组的数据类型，子数组包含两个整数
    (np.dtype([('subarray', (int, (2,)))], metadata={'some': 'stuff'}), False),
    # 第四组参数：递归定义数据类型，其中一个字段 'b' 包含一个数据类型，该数据类型包含一个字段 'c'，为整数类型且没有 metadata
    (np.dtype({'names': ['a', 'b'], 'formats': [float, np.dtype({'names': ['c'], 'formats': [np.dtype(int, metadata={})]})]}), False)
])
# 使用 pytest.mark.skipif 装饰器，设置条件跳过测试，当满足条件 IS_PYPY 为真且 Python 解释器版本 <= (7, 3, 8) 时跳过，原因是 PyPy 在错误格式化上存在 bug
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
        reason="PyPy bug in error formatting")
# 定义测试函数 test_metadata_dtype，用于测试 numpy 中结构化数据类型的 metadata 功能
def test_metadata_dtype(dt, fail):
    # gh-14142 注释
    # 创建一个包含 10 个元素的数组 arr，数据类型为 dt
    arr = np.ones(10, dtype=dt)
    # 创建一个字节流对象 buf
    buf = BytesIO()
    # 使用 assert_warns 检测是否会发出 UserWarning
    with assert_warns(UserWarning):
        # 将数组 arr 保存到 buf 中
        np.save(buf, arr)
    # 将 buf 的读取指针移动到开头
    buf.seek(0)
    # 如果 fail 为真，则期望抛出 ValueError 异常
    if fail:
        with assert_raises(ValueError):
            # 加载 buf 中的数据
            np.load(buf)
    else:
        # 否则，加载 buf 中的数据到 arr2
        arr2 = np.load(buf)
        # BUG: assert_array_equal 不会检查 metadata
        # 导入 drop_metadata 函数，该函数位于 numpy.lib._utils_impl 模块中
        from numpy.lib._utils_impl import drop_metadata
        # 断言 arr 和 arr2 相等
        assert_array_equal(arr, arr2)
        # 断言 drop_metadata 函数对 arr.dtype 的返回值不等于 arr.dtype 自身
        assert drop_metadata(arr.dtype) is not arr.dtype
        # 断言 drop_metadata 函数对 arr2.dtype 的返回值等于 arr2.dtype 自身
        assert drop_metadata(arr2.dtype) is arr2.dtype
```