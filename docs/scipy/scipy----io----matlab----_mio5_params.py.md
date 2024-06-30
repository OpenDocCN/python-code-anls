# `D:\src\scipysrc\scipy\scipy\io\matlab\_mio5_params.py`

```
''' Constants and classes for matlab 5 read and write

See also mio5_utils.pyx where these same constants arise as c enums.

If you make changes in this file, don't forget to change mio5_utils.pyx
'''

# 导入 NumPy 库，将其命名为 np
import numpy as np

# 从 _miobase 模块中导入 convert_dtypes 函数
from ._miobase import convert_dtypes

# 定义 __all__ 列表，包含公开的变量和类名
__all__ = [
    'MDTYPES', 'MatlabFunction', 'MatlabObject', 'MatlabOpaque',
    'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'OPAQUE_DTYPE', 'codecs_template',
    'mat_struct', 'mclass_dtypes_template', 'mclass_info', 'mdtypes_template',
    'miCOMPRESSED', 'miDOUBLE', 'miINT16', 'miINT32', 'miINT64', 'miINT8',
    'miMATRIX', 'miSINGLE', 'miUINT16', 'miUINT32', 'miUINT64', 'miUINT8',
    'miUTF16', 'miUTF32', 'miUTF8', 'mxCELL_CLASS', 'mxCHAR_CLASS',
    'mxDOUBLE_CLASS', 'mxFUNCTION_CLASS', 'mxINT16_CLASS', 'mxINT32_CLASS',
    'mxINT64_CLASS', 'mxINT8_CLASS', 'mxOBJECT_CLASS',
    'mxOBJECT_CLASS_FROM_MATRIX_H', 'mxOPAQUE_CLASS', 'mxSINGLE_CLASS',
    'mxSPARSE_CLASS', 'mxSTRUCT_CLASS', 'mxUINT16_CLASS', 'mxUINT32_CLASS',
    'mxUINT64_CLASS', 'mxUINT8_CLASS'
]

# 定义 MATLAB 数据类型的常量
miINT8 = 1
miUINT8 = 2
miINT16 = 3
miUINT16 = 4
miINT32 = 5
miUINT32 = 6
miSINGLE = 7
miDOUBLE = 9
miINT64 = 12
miUINT64 = 13
miMATRIX = 14
miCOMPRESSED = 15
miUTF8 = 16
miUTF16 = 17
miUTF32 = 18

# 定义 MATLAB 类型的类别常量
mxCELL_CLASS = 1
mxSTRUCT_CLASS = 2
mxOBJECT_CLASS = 3  # MATLAB 7 MAT-File Format 中对象的类别
mxCHAR_CLASS = 4
mxSPARSE_CLASS = 5
mxDOUBLE_CLASS = 6
mxSINGLE_CLASS = 7
mxINT8_CLASS = 8
mxUINT8_CLASS = 9
mxINT16_CLASS = 10
mxUINT16_CLASS = 11
mxINT32_CLASS = 12
mxUINT32_CLASS = 13
mxINT64_CLASS = 14
mxUINT64_CLASS = 15
mxFUNCTION_CLASS = 16  # MATLAB 函数的类别
mxOPAQUE_CLASS = 17  # MATLAB 不透明对象的类别
mxOBJECT_CLASS_FROM_MATRIX_H = 18  # 基于 matrix.h 定义的对象类别

# 定义 MATLAB 数据类型与 NumPy 数据类型的映射关系模板
mdtypes_template = {
    miINT8: 'i1',
    miUINT8: 'u1',
    miINT16: 'i2',
    miUINT16: 'u2',
    miINT32: 'i4',
    miUINT32: 'u4',
    miSINGLE: 'f4',
    miDOUBLE: 'f8',
    miINT64: 'i8',
    miUINT64: 'u8',
    miUTF8: 'u1',
    miUTF16: 'u2',
    miUTF32: 'u4',
    'file_header': [('description', 'S116'),
                    ('subsystem_offset', 'i8'),
                    ('version', 'u2'),
                    ('endian_test', 'S2')],
    'tag_full': [('mdtype', 'u4'), ('byte_count', 'u4')],
    'tag_smalldata':[('byte_count_mdtype', 'u4'), ('data', 'S4')],
}
    'array_flags': [('data_type', 'u4'),
                    ('byte_count', 'u4'),
                    ('flags_class','u4'),
                    ('nzmax', 'u4')],
    'U1': 'U1',
    }
# 定义一个模板字典，将 MATLAB 数组类型映射为 NumPy 数据类型的字符串表示
mclass_dtypes_template = {
    mxINT8_CLASS: 'i1',         # MATLAB int8 类型对应 NumPy 的 'i1'
    mxUINT8_CLASS: 'u1',        # MATLAB uint8 类型对应 NumPy 的 'u1'
    mxINT16_CLASS: 'i2',        # MATLAB int16 类型对应 NumPy 的 'i2'
    mxUINT16_CLASS: 'u2',       # MATLAB uint16 类型对应 NumPy 的 'u2'
    mxINT32_CLASS: 'i4',        # MATLAB int32 类型对应 NumPy 的 'i4'
    mxUINT32_CLASS: 'u4',       # MATLAB uint32 类型对应 NumPy 的 'u4'
    mxINT64_CLASS: 'i8',        # MATLAB int64 类型对应 NumPy 的 'i8'
    mxUINT64_CLASS: 'u8',       # MATLAB uint64 类型对应 NumPy 的 'u8'
    mxSINGLE_CLASS: 'f4',       # MATLAB single 类型对应 NumPy 的 'f4'
    mxDOUBLE_CLASS: 'f8',       # MATLAB double 类型对应 NumPy 的 'f8'
}

# 定义一个字典，将 MATLAB 类型映射为其字符串表示
mclass_info = {
    mxINT8_CLASS: 'int8',       # MATLAB int8 类型对应字符串 'int8'
    mxUINT8_CLASS: 'uint8',     # MATLAB uint8 类型对应字符串 'uint8'
    mxINT16_CLASS: 'int16',     # MATLAB int16 类型对应字符串 'int16'
    mxUINT16_CLASS: 'uint16',   # MATLAB uint16 类型对应字符串 'uint16'
    mxINT32_CLASS: 'int32',     # MATLAB int32 类型对应字符串 'int32'
    mxUINT32_CLASS: 'uint32',   # MATLAB uint32 类型对应字符串 'uint32'
    mxINT64_CLASS: 'int64',     # MATLAB int64 类型对应字符串 'int64'
    mxUINT64_CLASS: 'uint64',   # MATLAB uint64 类型对应字符串 'uint64'
    mxSINGLE_CLASS: 'single',   # MATLAB single 类型对应字符串 'single'
    mxDOUBLE_CLASS: 'double',   # MATLAB double 类型对应字符串 'double'
    mxCELL_CLASS: 'cell',       # MATLAB cell 类型对应字符串 'cell'
    mxSTRUCT_CLASS: 'struct',   # MATLAB struct 类型对应字符串 'struct'
    mxOBJECT_CLASS: 'object',   # MATLAB object 类型对应字符串 'object'
    mxCHAR_CLASS: 'char',       # MATLAB char 类型对应字符串 'char'
    mxSPARSE_CLASS: 'sparse',   # MATLAB sparse 类型对应字符串 'sparse'
    mxFUNCTION_CLASS: 'function',# MATLAB function 类型对应字符串 'function'
    mxOPAQUE_CLASS: 'opaque',   # MATLAB opaque 类型对应字符串 'opaque'
}

# 将 NumPy 数据类型字符串映射为 MATLAB 数组类型常量
NP_TO_MTYPES = {
    'f8': miDOUBLE,             # NumPy 'f8' 对应 MATLAB miDOUBLE
    'c32': miDOUBLE,            # NumPy 'c32' 对应 MATLAB miDOUBLE
    'c24': miDOUBLE,            # NumPy 'c24' 对应 MATLAB miDOUBLE
    'c16': miDOUBLE,            # NumPy 'c16' 对应 MATLAB miDOUBLE
    'f4': miSINGLE,             # NumPy 'f4' 对应 MATLAB miSINGLE
    'c8': miSINGLE,             # NumPy 'c8' 对应 MATLAB miSINGLE
    'i8': miINT64,              # NumPy 'i8' 对应 MATLAB miINT64
    'i4': miINT32,              # NumPy 'i4' 对应 MATLAB miINT32
    'i2': miINT16,              # NumPy 'i2' 对应 MATLAB miINT16
    'i1': miINT8,               # NumPy 'i1' 对应 MATLAB miINT8
    'u8': miUINT64,             # NumPy 'u8' 对应 MATLAB miUINT64
    'u4': miUINT32,             # NumPy 'u4' 对应 MATLAB miUINT32
    'u2': miUINT16,             # NumPy 'u2' 对应 MATLAB miUINT16
    'u1': miUINT8,              # NumPy 'u1' 对应 MATLAB miUINT8
    'S1': miUINT8,              # NumPy 'S1' 对应 MATLAB miUINT8
    'U1': miUTF16,              # NumPy 'U1' 对应 MATLAB miUTF16
    'b1': miUINT8,              # NumPy 'b1' 对应 MATLAB miUINT8 (虽然不标准，但 MATLAB 似乎使用这个)
}

# 将 NumPy 数据类型字符串映射为 MATLAB 数组类别常量
NP_TO_MXTYPES = {
    'f8': mxDOUBLE_CLASS,       # NumPy 'f8' 对应 MATLAB mxDOUBLE_CLASS
    'c32': mxDOUBLE_CLASS,      # NumPy 'c32' 对应 MATLAB mxDOUBLE_CLASS
    'c24': mxDOUBLE_CLASS,      # NumPy 'c24' 对应 MATLAB mxDOUBLE_CLASS
    'c16': mxDOUBLE_CLASS,      # NumPy 'c16' 对应 MATLAB mxDOUBLE_CLASS
    'f4': mxSINGLE_CLASS,       # NumPy 'f4' 对应 MATLAB mxSINGLE_CLASS
    'c8': mxSINGLE_CLASS,       # NumPy 'c8' 对应 MATLAB mxSINGLE_CLASS
    'i8': mxINT64_CLASS,        # NumPy 'i8' 对应 MATLAB mxINT64_CLASS
    'i4': mxINT32_CLASS,        # NumPy 'i4' 对应 MATLAB mxINT32_CLASS
    'i2': mxINT16_CLASS,        # NumPy 'i2' 对应 MATLAB mxINT16_CLASS
    'i1': mxINT8_CLASS,         # NumPy 'i1' 对应 MATLAB mxINT8_CLASS
    'u8': mxUINT64_CLASS,       # NumPy 'u8' 对应 MATLAB mxUINT64_CLASS
    'u4': mxUINT32_CLASS,       # NumPy 'u4' 对应 MATLAB mxUINT32_CLASS
    'u2': mxUINT16_CLASS,       # NumPy 'u2' 对应 MATLAB mxUINT16_CLASS
    'u1': mxUINT8_CLASS,        # NumPy 'u1' 对应 MATLAB mxUINT8_CLASS
    'S1': mxUINT8_CLASS,        # NumPy 'S1' 对应 MATLAB mxUINT8_CLASS
    'b1': mxUINT8_CLASS,        # NumPy 'b1' 对应 MATLAB mxUINT8_CLASS (虽然不标准，但 MATLAB 似乎使用这个)
}

''' Before release v7.1 (release 14) matlab (TM) used the system
default character encoding scheme padded out to 16-bits. Release 14
and later use Unicode. When saving character data, R14 checks if it
can be encoded in 7-bit ascii, and saves in that format if so.'''

# 定义一个编解码器模板字典，将 MATLAB 字符编码类型映射为相应的编解码器和宽度
codecs_template = {
    miUTF8: {'codec': 'utf_8', 'width': 1},     # MATLAB miUTF8 对应 UTF-8 编码，宽度为 1
    miUTF16: {'codec': 'utf_16', 'width': 2},   # MATLAB miUTF16 对应 UTF-16 编码，宽度为 2
    miUTF32: {'codec': 'utf_32','width': 4},    # MATLAB miUTF32 对应 UTF-32 编码，宽度为 4
}


def _convert_codecs(template, byte_order):
    ''' Convert codec template mapping to byte order

    将编解码器模板映射转换为指定字节序的编解码器映射

    Parameters
    ----------
    template : mapping
       key, value are respectively codec name, and root name for codec
       (without byte order suffix)
       键值对分别为编解码器名称和不带字节序后缀的根名称
    byte_order : {'<', '>'}
       code for little or big endian
       小端或大端的代码标识

    Returns
    -------
    codecs : dict
       key, value are name, codec (as in .encode(codec))
       键值对分别为名称和编解码器（如 .encode(codec) 中的 codec）
    '''
    codecs = {}
    postfix = byte_order == '<' and '_le'
    # 定义一个字典 `_def` 包含三个键值对，每个值是通过函数转换后的结果：
    # - 'dtypes': 使用函数 `convert_dtypes` 将 `mdtypes_template` 和 `_bytecode` 转换得到的数据类型
    # - 'classes': 使用函数 `convert_dtypes` 将 `mclass_dtypes_template` 和 `_bytecode` 转换得到的数据类型
    # - 'codecs': 使用函数 `_convert_codecs` 将 `codecs_template` 和 `_bytecode` 转换得到的编解码器
    
    _def = {
        'dtypes': convert_dtypes(mdtypes_template, _bytecode),
        'classes': convert_dtypes(mclass_dtypes_template, _bytecode),
        'codecs': _convert_codecs(codecs_template, _bytecode)
    }
    
    # 将 `_def` 字典赋值给全局变量 `MDTYPES` 的 `_bytecode` 键
    MDTYPES[_bytecode] = _def
# 定义一个名为 mat_struct 的类，用于存储从结构体中读取的数据的占位符

class mat_struct:
    """Placeholder for holding read data from structs.
    
    We use instances of this class when the user passes False as a value to the
    ``struct_as_record`` parameter of the :func:`scipy.io.loadmat` function.
    """
    pass


# 定义 MatlabObject 类，它是 numpy.ndarray 的子类，用于表示这是一个 MATLAB 对象

class MatlabObject(np.ndarray):
    """Subclass of ndarray to signal this is a matlab object.
    
    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be instantiated directly.
    """

    def __new__(cls, input_array, classname=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.classname = classname
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.classname = getattr(obj, 'classname', None)
        # We do not need to return anything


# 定义 MatlabFunction 类，它是 numpy.ndarray 的子类，用于表示一个 MATLAB 函数

class MatlabFunction(np.ndarray):
    """Subclass for a MATLAB function.
    
    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


# 定义 MatlabOpaque 类，它是 numpy.ndarray 的子类，用于表示一个 MATLAB 不透明矩阵

class MatlabOpaque(np.ndarray):
    """Subclass for a MATLAB opaque matrix.
    
    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


# 定义 OPAQUE_DTYPE，它是一个 numpy.dtype 对象，描述了 MATLAB 不透明类型的结构
OPAQUE_DTYPE = np.dtype(
    [('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])
```