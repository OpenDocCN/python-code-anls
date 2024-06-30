# `D:\src\scipysrc\scipy\scipy\io\matlab\_mio5_utils.pyx`

```
''' Cython mio5 utility routines (-*- python -*- like)

'''

# Programmer's notes
# ------------------
# Routines here have been reasonably optimized.

# The char matrix reading is not very fast, but it's not usually a
# bottleneck. See comments in ``read_char`` for possible ways to go if you
# want to optimize.

# 导入系统模块
import sys

# 导入 copy 模块中的 copy 函数并重命名为 pycopy
from copy import copy as pycopy

# 导入 Cython 的声明
cimport cython

# 导入标准库中的 calloc 和 free 函数声明
from libc.stdlib cimport calloc, free

# 导入标准库中的 strcmp 函数声明
from libc.string cimport strcmp

# 从 CPython 的 API 中导入相关函数和结构体声明
from cpython cimport Py_INCREF, PyObject
cdef extern from "Python.h":
    unicode PyUnicode_FromString(const char *u)
    ctypedef struct PyTypeObject:
        pass

# 从 CPython 的 numpy/arrayobject.h 头文件中导入相关声明
from cpython cimport PyBytes_Size
import numpy as np
cimport numpy as cnp

# 从 numpy/arrayobject.h 头文件中导入 PyArray_Type 和 PyArray_NewFromDescr 函数声明
cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyArray_Type
    cnp.ndarray PyArray_NewFromDescr(PyTypeObject *subtype,
                                     cnp.dtype newdtype,
                                     int nd,
                                     cnp.npy_intp* dims,
                                     cnp.npy_intp* strides,
                                     void* data,
                                     int flags,
                                     object parent)

# 从 numpy_rephrasing.h 头文件中导入 PyArray_Set_BASE 函数声明
cdef extern from "numpy_rephrasing.h":
    void PyArray_Set_BASE(cnp.ndarray arr, object obj)

# 初始化 numpy C-API
cnp.import_array()

# 定义常量 _MAT_MAXDIMS，并设置为 32
DEF _MAT_MAXDIMS = 32
# 定义常量 _N_MIS，并设置为 20
DEF _N_MIS = 20
# 定义常量 _N_MXS，并设置为 20
DEF _N_MXS = 20

# 从当前目录下导入 _streams 模块
from . cimport _streams

# 导入 scipy.io.matlab._mio_utils 模块中的 squeeze_element 和 chars_to_strings 函数
from scipy.io.matlab._mio_utils import squeeze_element, chars_to_strings

# 导入 scipy.io.matlab._mio5_params 模块并重命名为 mio5p
import scipy.io.matlab._mio5_params as mio5p

# 从 scipy.sparse 模块中导入 csc_matrix 类型
from scipy.sparse import csc_matrix

# 定义枚举类型 miINT8 等，对应 MATLAB 数据类型
cdef enum:
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

# 定义枚举类型 mxCELL_CLASS 等，对应 MATLAB 类型标识
cdef enum: # see comments in mio5_params
    mxCELL_CLASS = 1
    mxSTRUCT_CLASS = 2
    mxOBJECT_CLASS = 3
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
    mxFUNCTION_CLASS = 16
    mxOPAQUE_CLASS = 17 # This appears to be a function workspace
    mxOBJECT_CLASS_FROM_MATRIX_H = 18

# 检查系统的字节序，并存储在 sys_is_le 变量中
cdef bint sys_is_le = sys.byteorder == 'little'
# 如果系统是小端序，swapped_code 设为 '>'，否则设为 '<'
swapped_code = '>' if sys_is_le else '<'

# 定义 OPAQUE_DTYPE 和 BOOL_DTYPE 类型
cdef cnp.dtype OPAQUE_DTYPE = mio5p.OPAQUE_DTYPE
cdef cnp.dtype BOOL_DTYPE = np.dtype(np.bool_)

# 定义一个 Cython 公共函数 byteswap_u4，用于对 cnp.uint32_t 类型数据进行字节交换
cpdef cnp.uint32_t byteswap_u4(cnp.uint32_t u4) noexcept:
    return ((u4 << 24) |
           ((u4 << 8) & 0xff0000U) |
           ((u4 >> 8 & 0xff00u)) |
           (u4 >> 24))
cdef class VarHeader5:
    # 定义 VarHeader5 类，用于表示 MATLAB 文件变量头部信息

    cdef readonly object name
    # 只读属性，表示变量名

    cdef readonly int mclass
    # 只读属性，表示 MATLAB 数据类型

    cdef readonly object dims
    # 只读属性，表示变量的维度信息

    cdef cnp.int32_t dims_ptr[_MAT_MAXDIMS]
    # 使用 C 的 int32_t 类型数组存储维度指针

    cdef int n_dims
    # 表示维度数量的整数变量

    cdef int check_stream_limit
    # 检查流限制的整数变量

    cdef int is_complex
    # 表示变量是否为复数的整数变量

    cdef readonly int is_logical
    # 只读属性，表示变量是否为逻辑型

    cdef public int is_global
    # 公共整数变量，表示变量是否为全局变量

    cdef readonly size_t nzmax
    # 只读属性，表示非零元素的最大数量

    def set_dims(self, dims):
        """ Allow setting of dimensions from python

        This is for constructing headers for tests
        """
        # 允许从 Python 设置变量的维度信息

        self.dims = dims
        # 设置对象的 dims 属性为传入的 dims 参数

        self.n_dims = len(dims)
        # 设置对象的 n_dims 属性为 dims 参数的长度

        for i, dim in enumerate(dims):
            self.dims_ptr[i] = <cnp.int32_t>int(dim)
            # 将 dims 数组中的每个维度转换为 int32_t 类型并存储在 dims_ptr 数组中


cdef class VarReader5:
    """Initialize from file reader object

    preader needs the following fields defined:

    * mat_stream (file-like)
    * byte_order (str)
    * uint16_codec (str)
    * struct_as_record (bool)
    * chars_as_strings (bool)
    * mat_dtype (bool)
    * squeeze_me (bool)
    """

    # VarReader5 类，从文件读取器对象初始化

    cdef public int is_swapped, little_endian
    # 公共整数变量，表示是否进行了字节交换和小端模式

    cdef int struct_as_record
    # 结构作为记录的整数变量

    cdef object codecs, uint16_codec
    # 对象变量，用于存储编解码器和 uint16 编解码器

    # c-optimized version of reading stream
    cdef _streams.GenericStream cstream
    # C 优化版本的流读取器对象 cstream

    # pointers to stuff in preader.dtypes
    cdef PyObject* dtypes[_N_MIS]
    # 指向 preader.dtypes 中内容的指针数组

    # pointers to stuff in preader.class_dtypes
    cdef PyObject* class_dtypes[_N_MXS]
    # 指向 preader.class_dtypes 中内容的指针数组

    # element processing options
    cdef:
        int mat_dtype
        # 表示是否处理 MATLAB 数据类型的整数变量

        int squeeze_me
        # 表示是否压缩维度的整数变量

        int chars_as_strings
        # 表示是否将字符处理为字符串的整数变量
    def __cinit__(self, preader):
        # 获取输入的字节顺序信息
        byte_order = preader.byte_order
        # 检查是否需要交换字节序
        self.is_swapped = byte_order == swapped_code
        # 根据字节序确定小端序情况
        if self.is_swapped:
            self.little_endian = not sys_is_le
        else:
            self.little_endian = sys_is_le
        # 用于影响读取 Matlab 结构数组的选项
        self.struct_as_record = preader.struct_as_record
        # 存储用于文本矩阵读取的编解码器
        self.codecs = mio5p.MDTYPES[byte_order]['codecs'].copy()
        # 存储用于 uint16 类型的编解码器
        self.uint16_codec = preader.uint16_codec
        uint16_codec = self.uint16_codec
        # 设置 miUINT16 字符编码的长度
        self.codecs['uint16_len'] = len("  ".encode(uint16_codec)) \
                - len(" ".encode(uint16_codec))
        self.codecs['uint16_codec'] = uint16_codec
        # 根据 Python 文件类对象创建 C 优化的流对象
        self.cstream = _streams.make_stream(preader.mat_stream)
        # 元素处理选项
        self.mat_dtype = preader.mat_dtype
        self.chars_as_strings = preader.chars_as_strings
        self.squeeze_me = preader.squeeze_me
        # 将整数键的 dtype 引用复制到对象指针数组中
        for key, dt in mio5p.MDTYPES[byte_order]['dtypes'].items():
            if isinstance(key, str):
                continue
            self.dtypes[key] = <PyObject*>dt
        # 将 class_dtypes 的引用复制到对象指针数组中
        for key, dt in mio5p.MDTYPES[byte_order]['classes'].items():
            if isinstance(key, str):
                continue
            self.class_dtypes[key] = <PyObject*>dt

    def set_stream(self, fobj):
        ''' 
        从类似文件的对象 `fobj` 中设置最佳类型的流

        当初始化变量读取时从 Python 调用
        '''
        self.cstream = _streams.make_stream(fobj)

    def read_tag(self):
        ''' 
        读取标签 mdtype 和 byte_count

        进行必要的交换并考虑 SDE 格式。

        参见 ``read_full_tag`` 方法。

        返回
        -------
        mdtype : int
           Matlab 数据类型代码
        byte_count : int
           后续包含数据的字节数
        tag_data : None 或 str
           标签本身的任何数据。对于完整标签为 None，对于小数据元素为长度为 `byte_count` 的字符串。
        '''
        cdef cnp.uint32_t mdtype, byte_count
        cdef char tag_ptr[4]
        cdef int tag_res
        cdef object tag_data = None
        tag_res = self.cread_tag(&mdtype, &byte_count, tag_ptr)
        if tag_res == 2: # SDE 格式
            tag_data = tag_ptr[:byte_count]
        return (mdtype, byte_count, tag_data)
    # 定义一个 Cython 函数 read_element，用于读取数据元素并返回缓冲区

    ''' Read data element into string buffer, return buffer

    The element is the atom of the matlab file format.

    Parameters
    ----------
    mdtype_ptr : uint32_t*
       指向 uint32_t 值的指针，用于存储 mdtype 值
    byte_count_ptr : uint32_t*
       指向 uint32_t 值的指针，用于存储字节计数
    pp : void**
       void* 的指针。pp[0] 将被设置为指向返回的字符串内存的起始位置
    copy : int
       如果非零，则执行必要的复制操作，以便可以自由更改内存，而不会干扰其他对象。
       否则返回不应写入的字符串，从而避免不必要的复制

    Return
    ------
    data : str
       包含读取数据的 Python 字符串对象

    Notes
    -----
    参见 read_element_into，用于将元素读入预分配的内存块。
    '''
    
    # 定义 Cython 变量 byte_count 和 tag_data 数组
    cdef cnp.uint32_t byte_count
    cdef char tag_data[4]
    # 定义 Python 对象变量 data
    cdef object data
    # 定义整数变量 mod8 和 tag_res，并调用 self.cread_tag 方法
    cdef int mod8
    cdef int tag_res = self.cread_tag(mdtype_ptr,
                                      byte_count_ptr,
                                      tag_data)
    # 从 byte_count_ptr 中获取 byte_count 值
    byte_count = byte_count_ptr[0]
    
    # 根据 tag_res 的值进行条件判断
    if tag_res == 1: # full format
        # 调用 self.cstream.read_string 方法读取 byte_count 长度的字符串数据
        data = self.cstream.read_string(
            byte_count,
            pp,
            copy)
        # 将文件指针移动到下一个 64 位边界
        mod8 = byte_count % 8
        if mod8:
            self.cstream.seek(8 - mod8, 1)
    else: # SDE format, make safer home for data
        # 使用 tag_data 的前 byte_count 长度作为 data 数据
        data = tag_data[:byte_count]
        # 将 pp[0] 设置为指向 data 的 char* 指针
        pp[0] = <char *>data
    
    # 返回读取到的 data 数据
    return data
    # 定义 Cython 函数，从数据流中读取一个元素并存入预分配的内存空间中
    # 参数说明：
    # mdtype_ptr : uint32_t*
    #    指向 uint32_t 类型值的指针，用于写入 mdtype 值
    # byte_count_ptr : uint32_t*
    #    指向 uint32_t 类型值的指针，用于写入字节计数
    # ptr : void*
    #    内存缓冲区，用于存储读取的数据
    # max_byte_count : uint32_t
    #    ptr 指向的缓冲区的大小

    ''' Read element into pre-allocated memory in `ptr`

    Parameters
    ----------
    mdtype_ptr : uint32_t*
       pointer to uint32_t value to which we write the mdtype value
    byte_count_ptr : uint32_t*
       pointer to uint32_t value to which we write the byte count
    ptr : void*
       memory buffer into which to read
    max_byte_count : uint32_t
       size of the buffer pointed to by ptr

    Returns
    -------
    void

    Notes
    -----
    Compare ``read_element``.
    '''
    # 定义变量 mod8
    cdef:
       int mod8

    # 如果最大字节计数小于4，抛出数值错误异常
    if max_byte_count < 4:
        raise ValueError('Unexpected amount of data to read (malformed input file?)')

    # 调用 self.cread_tag 方法读取标签数据
    cdef int res = self.cread_tag(
        mdtype_ptr,
        byte_count_ptr,
        <char *>ptr)

    # 从 byte_count_ptr 中读取字节计数
    cdef cnp.uint32_t byte_count = byte_count_ptr[0]

    # 如果 res 为 1，表示完整格式
    if res == 1: # full format
        # 如果字节计数大于最大字节计数，抛出数值错误异常
        if byte_count > max_byte_count:
            raise ValueError('Unexpected amount of data to read (malformed input file?)')

        # 调用 self.cstream.read_into 方法读取数据到 ptr 中
        res = self.cstream.read_into(ptr, byte_count)

        # 将位置移到下一个 64 位边界
        mod8 = byte_count % 8
        if mod8:
            self.cstream.seek(8 - mod8, 1)

    # 返回 0 表示成功
    return 0
    # 定义一个 Cython cdef 函数，用于读取数值数据并返回一个 ndarray 对象
    cpdef cnp.ndarray read_numeric(self, int copy=True, size_t nnz=-1):
        ''' Read numeric data element into ndarray

        Reads element, then casts to ndarray.

        The type of the array is usually given by the ``mdtype`` returned via
        ``read_element``.  Sparse logical arrays are an exception, where the
        type of the array may be ``np.bool_`` even if the ``mdtype`` claims the
        data is of float64 type.

        Parameters
        ----------
        copy : bool, optional
            Whether to copy the array before returning.  If False, return array
            backed by bytes read from file.
        nnz : int, optional
            Number of non-zero values when reading numeric data from sparse
            matrices.  -1 if not reading sparse matrices, or to disable check
            for bytes data instead of declared data type (see Notes).

        Returns
        -------
        arr : array
            Numeric array

        Notes
        -----
        MATLAB apparently likes to store sparse logical matrix data as bytes
        instead of miDOUBLE (float64) data type, even though the data element
        still declares its type as miDOUBLE.  We can guess this has happened by
        looking for the length of the data compared to the expected number of
        elements, using the `nnz` input parameter.
        '''
        # 定义 Cython 变量 mdtype、byte_count、data_ptr
        cdef cnp.uint32_t mdtype, byte_count
        # 定义 Cython 变量 el_count 和 el，el 用于存储返回的 ndarray
        cdef void *data_ptr
        cdef cnp.npy_intp el_count
        cdef cnp.ndarray el
        # 调用对象的 read_element 方法读取数据，并返回结果存储在 data 中
        cdef object data = self.read_element(
            &mdtype, &byte_count, <void **>&data_ptr, copy)
        # 从预定义的 dtypes 中获取对应 mdtype 的数据类型描述符 dt
        cdef cnp.dtype dt = <cnp.dtype>self.dtypes[mdtype]
        # 如果数据类型的字节大小不为1且 nnz 不等于-1且 byte_count 等于 nnz，则 el_count 设置为 nnz，dt 设置为 BOOL_DTYPE
        if dt.itemsize != 1 and nnz != -1 and byte_count == nnz:
            el_count = <cnp.npy_intp> nnz
            dt = BOOL_DTYPE
        else:
            # 否则 el_count 设置为 byte_count 除以 dt 的字节大小
            el_count = byte_count // dt.itemsize
        # 设置标志 flags
        cdef int flags = 0
        if copy:
            flags = cnp.NPY_ARRAY_WRITEABLE
        # 增加 dt 的引用计数
        Py_INCREF(<object> dt)
        # 使用 PyArray_NewFromDescr 创建一个 ndarray el，使用 data_ptr 作为数据源
        el = PyArray_NewFromDescr(&PyArray_Type,
                                   dt,
                                   1,
                                   &el_count,
                                   NULL,
                                   <void*>data_ptr,
                                   flags,
                                   <object>NULL)
        # 增加 data 的引用计数
        Py_INCREF(<object> data)
        # 将 el 的基础对象设置为 data
        PyArray_Set_BASE(el, data)
        # 返回 ndarray el
        return el
    # 定义一个内联函数，用于读取并返回 int8 类型的字符串
    cdef inline object read_int8_string(self):
        ''' Read, return int8 type string
        
        int8 类型的字符串用于变量名、类名、结构体和对象的字段名。
        
        Specializes ``read_element``
        特化了 ``read_element`` 方法
        '''
        cdef:
            cnp.uint32_t mdtype, byte_count, i  # 定义存储数据类型、字节计数和循环变量的变量
            void* ptr  # 定义一个指针变量
            unsigned char* byte_ptr  # 定义一个无符号字符指针变量
            object data  # 定义一个 Python 对象变量，用于存储读取的数据
        # 调用 read_element 方法读取数据，并将返回的值存储在 data 变量中
        data = self.read_element(&mdtype, &byte_count, &ptr)
        # 如果数据类型是 miUTF8，说明可能是一些格式不正确的 .mat 文件
        if mdtype == miUTF8:  # Some badly-formed .mat files have utf8 here
            byte_ptr = <unsigned char*> ptr  # 将 ptr 转换为无符号字符指针类型
            # 遍历字节，并检查是否有大于 127 的字节，如果有则抛出 ValueError
            for i in range(byte_count):
                if byte_ptr[i] > 127:
                    raise ValueError('Non ascii int8 string')
        # 如果数据类型不是 miINT8，则抛出 TypeError
        elif mdtype != miINT8:
            raise TypeError('Expecting miINT8 as data type')
        # 返回读取的数据
        return data

    # 定义一个函数，用于将 int32 值读取到预先分配的内存中
    cdef int read_into_int32s(self, cnp.int32_t *int32p, cnp.uint32_t max_byte_count) except -1:
        ''' Read int32 values into pre-allocated memory
        
        将 int32 值读取到预先分配的内存中。需要注意的是，根据需要进行字节交换。
        
        Specializes ``read_element_into``
        特化了 ``read_element_into`` 方法
        
        Parameters
        ----------
        int32p : int32 指针
            指向预先分配内存的指针
        max_count : uint32_t
            最大字节计数
        
        Returns
        -------
        n_ints : int
            读取的整数个数
        '''
        cdef:
            cnp.uint32_t mdtype, byte_count, n_ints  # 定义存储数据类型、字节计数和整数个数的变量
            int i, check_ints=0  # 定义循环变量和检查整数标志变量
        # 调用 read_element_into 方法读取数据到 int32p 指向的内存中
        self.read_element_into(&mdtype, &byte_count, <void *>int32p, max_byte_count)
        # 如果数据类型是 miUINT32，则设置检查整数标志为 1
        if mdtype == miUINT32:
            check_ints = 1
        # 如果数据类型不是 miINT32，则抛出 TypeError
        elif mdtype != miINT32:
            raise TypeError('Expecting miINT32 as data type')
        # 计算读取的整数个数，每个 int32 占据 4 字节
        n_ints = byte_count // 4
        # 如果数据需要进行字节交换，则对每个整数进行交换
        if self.is_swapped:
            for i in range(n_ints):
                int32p[i] = byteswap_u4(int32p[i])
        # 如果需要检查整数类型，并且发现有负值的 miUINT32 类型数据，则抛出 ValueError
        if check_ints:
            for i in range(n_ints):
                if int32p[i] < 0:
                    raise ValueError('Expecting miINT32, got miUINT32 with '
                                     'negative values')
        # 返回读取的整数个数
        return n_ints

    # 定义一个 Python 方法，用于从流中读取完整的 u4, u4 标签
    def read_full_tag(self):
        ''' Python method for reading full u4, u4 tag from stream
        
        从流中读取完整的 u4, u4 标签，并返回数据类型代码和后续数据字节数
        
        Returns
        -------
        mdtype : int32
            Matlab 数据类型代码
        byte_count : int32
            后续数据的字节数
        
        Notes
        -----
        假设标签确实是完整的，即不是小数据元素。这意味着可以跳过一些检查，使其比 ``read_tag`` 稍快。
        '''
        cdef cnp.uint32_t mdtype, byte_count  # 定义存储数据类型代码和字节计数的变量
        # 调用 cread_full_tag 方法读取数据类型代码和字节计数
        self.cread_full_tag(&mdtype, &byte_count)
        # 返回数据类型代码和字节计数
        return mdtype, byte_count
    # 定义一个 C 方法，用于从流中读取完整的 u4, u4 标签
    cdef int cread_full_tag(self,
                            cnp.uint32_t* mdtype,
                            cnp.uint32_t* byte_count) except -1:
        ''' C method for reading full u4, u4 tag from stream'''
        # 声明一个用于存储两个 uint32_t 值的数组
        cdef cnp.uint32_t u4s[2]
        # 从流中读取8字节的数据到 u4s 数组中
        self.cstream.read_into(<void *>u4s, 8)
        # 如果需要字节交换，对读取的数据进行字节交换处理，并赋值给 mdtype 和 byte_count
        if self.is_swapped:
            mdtype[0] = byteswap_u4(u4s[0])
            byte_count[0] = byteswap_u4(u4s[1])
        else:
            # 否则直接将读取的数据赋值给 mdtype 和 byte_count
            mdtype[0] = u4s[0]
            byte_count[0] = u4s[1]
        # 返回操作成功的标志
        return 0

    # 定义一个 CPython 方法，用于读取当前流位置的变量头部信息
    cpdef VarHeader5 read_header(self, int check_stream_limit):
        ''' Return matrix header for current stream position

        Returns matrix headers at top level and sub levels

        Parameters
        ----------
        check_stream_limit : if True, then if the returned header
        is passed to array_from_header, it will be verified that
        the length of the uncompressed data is not overlong (which
        can indicate .mat file corruption)
        '''
        cdef:
            cdef cnp.uint32_t u4s[2]  # 用于存储两个 uint32_t 值的数组
            cnp.uint32_t flags_class, nzmax  # 存储 flags_class 和 nzmax 的变量
            cnp.uint16_t mc  # 存储 uint16_t 类型的 mc
            int i  # 循环使用的整数变量
            VarHeader5 header  # 创建 VarHeader5 类型的变量 header

        # 读取并丢弃 mdtype 和 byte_count 数据
        self.cstream.read_into(<void *>u4s, 8)
        
        # 读取 array flags 和 nzmax 数据
        self.cstream.read_into(<void *>u4s, 8)
        
        # 如果需要字节交换，对读取的数据进行字节交换处理，并赋值给 flags_class 和 nzmax
        if self.is_swapped:
            flags_class = byteswap_u4(u4s[0])
            nzmax = byteswap_u4(u4s[1])
        else:
            # 否则直接将读取的数据赋值给 flags_class 和 nzmax
            flags_class = u4s[0]
            nzmax = u4s[1]
        
        # 创建一个 VarHeader5 类型的对象 header
        header = VarHeader5()
        
        # 从 flags_class 中提取出 mc 的值
        mc = flags_class & 0xFF
        header.mclass = mc
        
        # 根据 flags_class 中的位域信息设置 header 的各个属性
        header.check_stream_limit = check_stream_limit
        header.is_logical = flags_class >> 9 & 1
        header.is_global = flags_class >> 10 & 1
        header.is_complex = flags_class >> 11 & 1
        header.nzmax = nzmax
        
        # 对于所有的 miMATRIX 类型，除了 mxOPAQUE_CLASS，都有 dims 和 name 属性
        if mc == mxOPAQUE_CLASS:
            header.name = None
            header.dims = None
            return header
        
        # 读取 dims 的数量并将其存储到 header 对象中
        header.n_dims = self.read_into_int32s(header.dims_ptr, sizeof(header.dims_ptr))
        
        # 如果 dims 的数量超过 _MAT_MAXDIMS，抛出 ValueError 异常
        if header.n_dims > _MAT_MAXDIMS:
            raise ValueError('Too many dimensions (%d) for numpy arrays'
                             % header.n_dims)
        
        # 将 dims_ptr 转换为一个列表并存储到 header 的 dims 属性中
        header.dims = [header.dims_ptr[i] for i in range(header.n_dims)]
        
        # 读取 int8 类型的字符串作为 header 的 name 属性
        header.name = self.read_int8_string()
        
        # 返回读取到的 header 对象
        return header
    # 定义一个内联函数，用于从头部计算数组的大小
    # 通过直接访问头部中的整数来计算数组大小，而不使用 Python 列表 header.dims
    cdef inline size_t size_from_header(self, VarHeader5 header) noexcept:
        ''' Supporting routine for calculating array sizes from header

        Probably unnecessary optimization that uses integers stored in
        header rather than ``header.dims`` that is a python list.

        Parameters
        ----------
        header : VarHeader5
           array header

        Returns
        -------
        size : size_t
           size of array referenced by header (product of dims)
        '''
        # 计算数组中项目的数量，通过维数的乘积
        cdef size_t size = 1
        cdef int i
        for i in range(header.n_dims):
            size *= header.dims_ptr[i]
        return size

    # 定义一个函数，用于读取包含在子级中的矩阵头部
    # 结合了 ``read_header`` 和 ``array_from_header`` 的功能，根据 self 中的选项处理数组
    cdef read_mi_matrix(self, int process=1):
        ''' Read header with matrix at sub-levels

        Combines ``read_header`` and functionality of
        ``array_from_header``.  Applies standard processing of array
        given options set in self.

        Parameters
        ----------
        process : int, optional
           If not zero, apply post-processing on returned array

        Returns
        -------
        arr : ndarray or sparse matrix
        '''
        cdef:
            VarHeader5 header
            cnp.uint32_t mdtype, byte_count
        
        # 读取完整的标签信息
        self.cread_full_tag(&mdtype, &byte_count)
        if mdtype != miMATRIX:
            raise TypeError('Expecting matrix here')
        if byte_count == 0: # 空矩阵
            if process and self.squeeze_me:
                return np.array([])
            else:
                return np.array([[]])
        
        # 读取头部信息
        header = self.read_header(False)
        return self.array_from_header(header, process)
    # 使用 `cpdef` 声明一个 CPython 可调用的函数，可以在 Python 和 C/C++ 中调用
    cpdef array_from_header(self, VarHeader5 header, int process=1):
        ''' Read array of any class, given matrix `header`

        Parameters
        ----------
        header : VarHeader5
           array header object
        process : int, optional
           If not zero, apply post-processing on returned array

        Returns
        -------
        arr : array or sparse array
           read array
        '''
        # 声明变量 `arr` 和 `mat_dtype`
        cdef:
            object arr
            cnp.dtype mat_dtype
        # 将 `header.mclass` 赋值给变量 `mc`
        cdef int mc = header.mclass
        # 检查 `mc` 是否属于以下数值矩阵类别之一
        if (mc == mxDOUBLE_CLASS
            or mc == mxSINGLE_CLASS
            or mc == mxINT8_CLASS
            or mc == mxUINT8_CLASS
            or mc == mxINT16_CLASS
            or mc == mxUINT16_CLASS
            or mc == mxINT32_CLASS
            or mc == mxUINT32_CLASS
            or mc == mxINT64_CLASS
            or mc == mxUINT64_CLASS): # numeric matrix
            # 调用 `self.read_real_complex` 方法读取实数或复数矩阵
            arr = self.read_real_complex(header)
            # 如果需要进行处理且 `self.mat_dtype` 存在
            if process and self.mat_dtype: # might need to recast
                # 如果 `header.is_logical` 为真，则 `mat_dtype` 为 `BOOL_DTYPE`，否则为 `self.class_dtypes[mc]`
                if header.is_logical:
                    mat_dtype = BOOL_DTYPE
                else:
                    mat_dtype = <object>self.class_dtypes[mc]
                # 将 `arr` 转换为 `mat_dtype` 类型
                arr = arr.astype(mat_dtype)
        # 如果 `mc` 属于稀疏矩阵类别 `mxSPARSE_CLASS`
        elif mc == mxSPARSE_CLASS:
            # 调用 `self.read_sparse` 方法读取稀疏矩阵
            arr = self.read_sparse(header)
            # 对稀疏矩阵不进行任何处理
            process = False
        # 如果 `mc` 属于字符数组类别 `mxCHAR_CLASS`
        elif mc == mxCHAR_CLASS:
            # 调用 `self.read_char` 方法读取字符数组
            arr = self.read_char(header)
            # 如果需要进行处理且 `self.chars_as_strings` 为真
            if process and self.chars_as_strings:
                # 将字符数组转换为字符串数组
                arr = chars_to_strings(arr)
        # 如果 `mc` 属于单元数组类别 `mxCELL_CLASS`
        elif mc == mxCELL_CLASS:
            # 调用 `self.read_cells` 方法读取单元数组
            arr = self.read_cells(header)
        # 如果 `mc` 属于结构体数组类别 `mxSTRUCT_CLASS`
        elif mc == mxSTRUCT_CLASS:
            # 调用 `self.read_struct` 方法读取结构体数组
            arr = self.read_struct(header)
        # 如果 `mc` 属于对象数组类别 `mxOBJECT_CLASS`
        elif mc == mxOBJECT_CLASS: # like structs, but with classname
            # 读取对象类名并解码为字符串
            classname = self.read_int8_string().decode('latin1')
            # 读取结构体数组并封装为 MatlabObject 对象
            arr = self.read_struct(header)
            arr = mio5p.MatlabObject(arr, classname)
        # 如果 `mc` 属于函数句柄数组类别 `mxFUNCTION_CLASS`
        elif mc == mxFUNCTION_CLASS: # just a matrix of struct type
            # 读取函数句柄数组并封装为 MatlabFunction 对象
            arr = self.read_mi_matrix()
            arr = mio5p.MatlabFunction(arr)
            # 不压缩以提高可重写性
            process = 0
        # 如果 `mc` 属于不透明对象数组类别 `mxOPAQUE_CLASS`
        elif mc == mxOPAQUE_CLASS:
            # 读取不透明对象数组并封装为 MatlabOpaque 对象
            arr = self.read_opaque(header)
            arr = mio5p.MatlabOpaque(arr)
            # 不压缩以提高可重写性
            process = 0
        # 确保已读取完校验和
        read_ok = self.cstream.all_data_read()
        # 如果 `header.check_stream_limit` 为真且未完全读取数据，则抛出异常
        if header.check_stream_limit and not read_ok:
            raise ValueError('Did not fully consume compressed contents' +
                             ' of an miCOMPRESSED element. This can' +
                             ' indicate that the .mat file is corrupted.')
        # 如果需要处理且 `self.squeeze_me` 为真，则返回压缩数组
        if process and self.squeeze_me:
            return squeeze_element(arr)
        # 否则返回原始数组
        return arr
    # 使用 VarHeader5 类型的 header 参数获取形状信息
    def shape_from_header(self, VarHeader5 header):
        # 从 header 中获取 mclass 属性值
        cdef int mc = header.mclass
        # 定义 tuple 类型的 shape 变量
        cdef tuple shape
        # 根据 mclass 的值进行判断和赋值
        if mc == mxSPARSE_CLASS:
            # 如果 mclass 是 mxSPARSE_CLASS，使用 header.dims 创建 shape
            shape = tuple(header.dims)
        elif mc == mxCHAR_CLASS:
            # 如果 mclass 是 mxCHAR_CLASS，也使用 header.dims 创建 shape
            shape = tuple(header.dims)
            # 如果 self.chars_as_strings 为真，去掉 shape 的最后一个维度
            if self.chars_as_strings:
                shape = shape[:-1]
        else:
            # 对于其他情况，仍然使用 header.dims 创建 shape
            shape = tuple(header.dims)
        # 如果 self.squeeze_me 为真，去除 shape 中值为 1 的维度
        if self.squeeze_me:
            shape = tuple([x for x in shape if x != 1])
        # 返回最终确定的 shape
        return shape

    # 使用 VarHeader5 类型的 header 参数读取实数和复数矩阵
    cpdef cnp.ndarray read_real_complex(self, VarHeader5 header):
        ''' 从流中读取实数/复数矩阵 '''
        # 定义 res 和 res_j 两个 cnp.ndarray 类型的变量
        cdef:
            cnp.ndarray res, res_j
        # 如果 header.is_complex 为真
        if header.is_complex:
            # 避免复制数组以节省内存，读取实数部分和虚数部分
            res = self.read_numeric(False)
            res_j = self.read_numeric(False)
            # 使用 c8 来表示 f4 类型的数据，使用 c16 来表示 f8 类型的数据
            # 通过 res = res + res_j * 1j 运算将输入类型统一提升为 c16
            if res.itemsize == 4:
                res = res.astype('c8')
            else:
                res = res.astype('c16')
            # 将 res 的虚部设为 res_j
            res.imag = res_j
        else:
            # 否则，只读取实数部分
            res = self.read_numeric()
        # 将读取的数据重塑为指定的形状，并转置返回
        return res.reshape(header.dims[::-1]).T

    # 使用 VarHeader5 类型的 header 参数读取稀疏矩阵
    cdef object read_sparse(self, VarHeader5 header):
        ''' 从流中读取稀疏矩阵 '''
        # 定义 rowind、indptr、data 和 data_j 四个 cnp.ndarray 类型的变量
        cdef cnp.ndarray rowind, indptr, data, data_j
        # 定义 M、N、nnz 三个 size_t 类型的变量
        cdef size_t M, N, nnz
        # 读取 rowind 和 indptr 数组
        rowind = self.read_numeric()
        indptr = self.read_numeric()
        # 从 header 中获取矩阵的行数 M 和列数 N
        M, N = header.dims[0], header.dims[1]
        # 将 indptr 数组截取到长度为 N+1
        indptr = indptr[:N+1]
        # 获取 nnz（非零元素的个数），即 indptr 的最后一个元素
        nnz = indptr[-1]
        # 如果 header.is_complex 为真
        if header.is_complex:
            # 避免复制数组以节省内存，读取实数部分和虚数部分
            data   = self.read_numeric(False)
            data_j = self.read_numeric(False)
            # 将 data 加上 data_j * 1j，构成复数数据
            data = data + (data_j * 1j)
        elif header.is_logical:
            # 如果 header.is_logical 为真，按照逻辑类型读取数据
            data = self.read_numeric(True, nnz)
        else:
            # 否则，按照普通数值类型读取数据
            data = self.read_numeric()

        # 根据 MATLAB API 文档，构建稀疏矩阵 csc_matrix
        return csc_matrix(
            (data[:nnz], rowind[:nnz], indptr),
            shape=(M, N))
    # 从流中读取单元数组
    cpdef cnp.ndarray read_cells(self, VarHeader5 header):
        ''' Read cell array from stream '''
        cdef:
            size_t i  # 定义变量 i，用于循环索引
            cnp.ndarray[object, ndim=1] result  # 创建一个一维数组对象 result，元素类型为 object
        # 考虑单元格的 Fortran 索引方式
        tupdims = tuple(header.dims[::-1])  # 反转 header 的维度元组
        cdef size_t length = self.size_from_header(header)  # 调用 size_from_header 方法获取数组长度
        result = np.empty(length, dtype=object)  # 创建一个长度为 length 的空数组，元素类型为 object
        for i in range(length):
            result[i] = self.read_mi_matrix()  # 循环读取每个单元格的数据并存入 result 数组
        return result.reshape(tupdims).T  # 将 result 数组 reshape 成 tupdims 指定的形状并转置返回

    # 读取结构矩阵的字段名
    def read_fieldnames(self):
        '''Read fieldnames for struct-like matrix.'''
        cdef int n_names  # 定义整型变量 n_names
        return self.cread_fieldnames(&n_names)  # 调用 cread_fieldnames 方法并返回结果

    # 内联方法，读取字段名
    cdef inline object cread_fieldnames(self, int *n_names_ptr):
        cdef:
            cnp.int32_t namelength  # 定义 namelength 变量，类型为 cnp.int32_t
            int i, n_names  # 定义整型变量 i 和 n_names
            list field_names  # 定义列表 field_names，用于存储字段名
            object name  # 定义对象变量 name，用于存储字段名
        # 读取字段名并存入列表
        cdef int res = self.read_into_int32s(&namelength, 4)  # 调用 read_into_int32s 方法读取 namelength
        if res != 1:
            raise ValueError('Only one value for namelength')  # 若读取结果不为 1，则抛出 ValueError 异常
        cdef object names = self.read_int8_string()  # 调用 read_int8_string 方法读取字段名字符串
        field_names = []  # 初始化字段名列表为空列表
        n_names = PyBytes_Size(names) // namelength  # 计算字段名个数
        # 创建重复计数和指针数组
        cdef:
            int *n_duplicates  # 定义整型指针 n_duplicates
        n_duplicates = <int *>calloc(n_names, sizeof(int))  # 分配 n_names 长度的整型内存空间
        cdef:
            char *names_ptr = names  # 定义字符指针 names_ptr 指向 names
            char *n_ptr = names  # 定义字符指针 n_ptr 指向 names
            int j, dup_no  # 定义整型变量 j 和 dup_no
        for i in range(n_names):
            name = PyUnicode_FromString(n_ptr)  # 根据 n_ptr 创建 Python Unicode 对象 name
            # 检查是否是重复字段名，如是则重命名
            dup_no = 0
            for j in range(i):
                if strcmp(n_ptr, names_ptr + j * namelength) == 0:  # 如果找到重复的字段名
                    n_duplicates[j] += 1  # 对应位置的重复计数加 1
                    dup_no = n_duplicates[j]  # 记录重复序号
                    break
            if dup_no != 0:
                name = '_%d_%s' % (dup_no, name)  # 如果有重复，则在名字前加上序号
            field_names.append(name)  # 将处理后的字段名添加到 field_names 列表中
            n_ptr += namelength  # 指针 n_ptr 向后移动 namelength 长度
        free(n_duplicates)  # 释放 n_duplicates 的内存空间
        n_names_ptr[0] = n_names  # 将字段名个数写入 n_names_ptr 指向的位置
        return field_names  # 返回包含字段名的列表
    # 从流中读取结构体或对象数组
    cpdef cnp.ndarray read_struct(self, VarHeader5 header):
        ''' Read struct or object array from stream

        Objects are just structs with an extra field *classname*,
        defined before (this here) struct format structure
        '''
        cdef:
            int i, n_names
            cnp.ndarray[object, ndim=1] result
            object dt, tupdims
        
        # 读取字段名列表
        cdef object field_names = self.cread_fieldnames(&n_names)
        
        # 准备结构体数组的维度
        tupdims = tuple(header.dims[::-1])
        
        # 根据头部信息计算数组长度
        cdef size_t length = self.size_from_header(header)
        
        # 如果将结构体视为记录数组
        if self.struct_as_record:
            if not n_names:
                # 如果没有字段名，无法创建有效的 dtype 表示，返回空对象数组
                return np.empty(tupdims, dtype=object).T
            
            # 创建记录数组的 dtype
            dt = [(field_name, object) for field_name in field_names]
            rec_res = np.empty(length, dtype=dt)
            
            # 逐个字段读取数据到记录数组
            for i in range(length):
                for field_name in field_names:
                    rec_res[i][field_name] = self.read_mi_matrix()
            
            return rec_res.reshape(tupdims).T
        
        # 向后兼容之前的格式
        obj_template = mio5p.mat_struct()
        obj_template._fieldnames = field_names
        
        # 创建对象数组
        result = np.empty(length, dtype=object)
        
        # 逐个对象填充数据
        for i in range(length):
            item = pycopy(obj_template)
            for name in field_names:
                item.__dict__[name] = self.read_mi_matrix()
            
            with cython.boundscheck(False):
                result[i] = item
        
        return result.reshape(tupdims).T

    # 读取不透明类型数据（函数工作区）
    cpdef object read_opaque(self, VarHeader5 hdr):
        ''' Read opaque (function workspace) type

        Looking at some mat files, the structure of this type seems to
        be:

        * array flags as usual (already read into `hdr`)
        * 3 int8 strings
        * a matrix

        Then there's a matrix at the end of the mat file that seems have
        the anonymous function workspaces - we load it as
        ``__function_workspace__``

        See the comments at the beginning of ``mio5.py``
        '''
        
        # res 和函数返回值都没有被声明为 cnp.ndarray，因为当前的 Cython (0.23.4) 中这会增加无用的检查。
        res = np.empty((1,), dtype=OPAQUE_DTYPE)
        res0 = res[0]
        
        # 读取三个 int8 字符串和一个矩阵到不透明数据结构
        res0['s0'] = self.read_int8_string()
        res0['s1'] = self.read_int8_string()
        res0['s2'] = self.read_int8_string()
        res0['arr'] = self.read_mi_matrix()
        
        return res
```