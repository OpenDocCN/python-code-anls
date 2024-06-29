# `D:\src\scipysrc\numpy\numpy\__init__.pxd`

```
# 导入必要的 Cython 模块和 NumPy 静态声明
#
# 如果调用任何 PyArray_* 函数，必须首先调用 import_array。
#
# 作者：Dag Sverre Seljebotn
#

# 定义缓冲区格式字符串的最大长度
DEF _buffer_format_string_len = 255

# 导入 Cython 中的 Python 缓冲区模块
cimport cpython.buffer as pybuf
# 导入 Cython 中的 Python 引用模块
from cpython.ref cimport Py_INCREF
# 导入 Cython 中的 Python 内存模块
from cpython.mem cimport PyObject_Malloc, PyObject_Free
# 导入 Cython 中的 Python 对象模块
from cpython.object cimport PyObject, PyTypeObject
# 导入 Cython 中的 Python 缓冲区模块
from cpython.buffer cimport PyObject_GetBuffer
# 导入 Cython 中的 Python 类型模块
from cpython.type cimport type
# 导入 libc 中的标准输入输出模块
cimport libc.stdio as stdio

# 从外部导入声明
cdef extern from *:
    # 留下一个标记，表明 NumPy 声明来自 NumPy 本身而不是 Cython。
    # 参考 https://github.com/cython/cython/issues/3573
    """
    /* Using NumPy API declarations from "numpy/__init__.pxd" */
    """

# 从 Python.h 头文件中外部导入声明
cdef extern from "Python.h":
    ctypedef int Py_intptr_t
    bint PyObject_TypeCheck(object obj, PyTypeObject* type)

# 从 numpy/arrayobject.h 头文件中外部导入声明
cdef extern from "numpy/arrayobject.h":
    # 定义 NumPy 中的整型和无符号整型类型
    ctypedef signed long npy_intp
    ctypedef unsigned long npy_uintp

    # 定义 NumPy 中的布尔类型
    ctypedef unsigned char      npy_bool

    # 定义 NumPy 中的有符号整型类型
    ctypedef signed char      npy_byte
    ctypedef signed short     npy_short
    ctypedef signed int       npy_int
    ctypedef signed long      npy_long
    ctypedef signed long long npy_longlong

    # 定义 NumPy 中的无符号整型类型
    ctypedef unsigned char      npy_ubyte
    ctypedef unsigned short     npy_ushort
    ctypedef unsigned int       npy_uint
    ctypedef unsigned long      npy_ulong
    ctypedef unsigned long long npy_ulonglong

    # 定义 NumPy 中的浮点类型
    ctypedef float        npy_float
    ctypedef double       npy_double
    ctypedef long double  npy_longdouble

    # 定义 NumPy 中的特定位数整型类型
    ctypedef signed char        npy_int8
    ctypedef signed short       npy_int16
    ctypedef signed int         npy_int32
    ctypedef signed long long   npy_int64
    ctypedef signed long long   npy_int96
    ctypedef signed long long   npy_int128

    # 定义 NumPy 中的特定位数无符号整型类型
    ctypedef unsigned char      npy_uint8
    ctypedef unsigned short     npy_uint16
    ctypedef unsigned int       npy_uint32
    ctypedef unsigned long long npy_uint64
    ctypedef unsigned long long npy_uint96
    ctypedef unsigned long long npy_uint128

    # 定义 NumPy 中的特定位数浮点类型
    ctypedef float        npy_float32
    ctypedef double       npy_float64
    ctypedef long double  npy_float80
    ctypedef long double  npy_float96
    ctypedef long double  npy_float128

    # 定义 NumPy 中的复数类型
    ctypedef struct npy_cfloat:
        pass

    ctypedef struct npy_cdouble:
        pass

    ctypedef struct npy_clongdouble:
        pass

    ctypedef struct npy_complex64:
        pass

    ctypedef struct npy_complex128:
        pass

    ctypedef struct npy_complex160:
        pass

    ctypedef struct npy_complex192:
        pass

    ctypedef struct npy_complex256:
        pass
    # 定义结构体 PyArray_Dims，包含一个指向 npy_intp 类型的指针和一个整型 len
    ctypedef struct PyArray_Dims:
        npy_intp *ptr
        int len
    
    # 定义枚举类型 NPY_TYPES，列举了多种 NumPy 数值类型常量
    cdef enum NPY_TYPES:
        NPY_BOOL         # 布尔类型
        NPY_BYTE         # 字节（8位有符号整数）
        NPY_UBYTE        # 无符号字节（8位无符号整数）
        NPY_SHORT        # 短整型（16位有符号整数）
        NPY_USHORT       # 无符号短整型（16位无符号整数）
        NPY_INT          # 整型（通常为32位有符号整数）
        NPY_UINT         # 无符号整型（32位无符号整数）
        NPY_LONG         # 长整型（通常为64位有符号整数）
        NPY_ULONG        # 无符号长整型（64位无符号整数）
        NPY_LONGLONG     # 长长整型（通常为128位有符号整数）
        NPY_ULONGLONG    # 无符号长长整型（128位无符号整数）
        NPY_FLOAT        # 单精度浮点型
        NPY_DOUBLE       # 双精度浮点型
        NPY_LONGDOUBLE   # 长双精度浮点型
        NPY_CFLOAT       # 复数，32位浮点实部和虚部
        NPY_CDOUBLE      # 复数，64位浮点实部和虚部
        NPY_CLONGDOUBLE  # 复数，长双精度浮点实部和虚部
        NPY_OBJECT       # Python 对象
        NPY_STRING       # 字符串（固定长度）
        NPY_UNICODE      # Unicode 字符串
        NPY_VOID         # 未知类型
        NPY_DATETIME     # 日期时间
        NPY_TIMEDELTA    # 时间间隔
        NPY_NTYPES_LEGACY  # NumPy 1.14 之前的类型数
        NPY_NOTYPE       # 无类型
    
        # 更详细的整数类型定义
        NPY_INT8          # 8位有符号整数
        NPY_INT16         # 16位有符号整数
        NPY_INT32         # 32位有符号整数
        NPY_INT64         # 64位有符号整数
        NPY_INT128        # 128位有符号整数
        NPY_INT256        # 256位有符号整数
        NPY_UINT8         # 8位无符号整数
        NPY_UINT16        # 16位无符号整数
        NPY_UINT32        # 32位无符号整数
        NPY_UINT64        # 64位无符号整数
        NPY_UINT128       # 128位无符号整数
        NPY_UINT256       # 256位无符号整数
        # 浮点数类型定义
        NPY_FLOAT16       # 半精度浮点数
        NPY_FLOAT32       # 单精度浮点数
        NPY_FLOAT64       # 双精度浮点数
        NPY_FLOAT80       # 扩展精度浮点数
        NPY_FLOAT96       # 96位浮点数
        NPY_FLOAT128      # 128位浮点数
        NPY_FLOAT256      # 256位浮点数
        # 复数类型定义
        NPY_COMPLEX32     # 32位复数
        NPY_COMPLEX64     # 64位复数
        NPY_COMPLEX128    # 128位复数
        NPY_COMPLEX160    # 160位复数
        NPY_COMPLEX192    # 192位复数
        NPY_COMPLEX256    # 256位复数
        NPY_COMPLEX512    # 512位复数
    
        NPY_INTP          # 平台指针大小的整数
        NPY_DEFAULT_INT   # 默认整数类型（通常与平台的 native int 相同）
    
    # 定义枚举类型 NPY_ORDER，指定数组在内存中的存储顺序
    ctypedef enum NPY_ORDER:
        NPY_ANYORDER      # 任意顺序
        NPY_CORDER        # C 顺序（行优先）
        NPY_FORTRANORDER  # Fortran 顺序（列优先）
        NPY_KEEPORDER     # 保持当前顺序
    
    # 定义枚举类型 NPY_CASTING，指定数据类型转换的规则
    ctypedef enum NPY_CASTING:
        NPY_NO_CASTING         # 禁止任何类型转换
        NPY_EQUIV_CASTING      # 仅允许等价类型转换
        NPY_SAFE_CASTING       # 安全类型转换（不损失精度）
        NPY_SAME_KIND_CASTING  # 同类型转换
        NPY_UNSAFE_CASTING     # 不安全类型转换
    
    # 定义枚举类型 NPY_CLIPMODE，指定数据裁剪（超出范围处理）的模式
    ctypedef enum NPY_CLIPMODE:
        NPY_CLIP   # 裁剪
        NPY_WRAP   # 包裹
        NPY_RAISE  # 引发异常
    
    # 定义枚举类型 NPY_SCALARKIND，指定标量的类型
    ctypedef enum NPY_SCALARKIND:
        NPY_NOSCALAR      # 非标量
        NPY_BOOL_SCALAR   # 布尔标量
        NPY_INTPOS_SCALAR # 正整数标量
        NPY_INTNEG_SCALAR # 负整数标量
        NPY_FLOAT_SCALAR  # 浮点数标量
        NPY_COMPLEX_SCALAR  # 复数标量
        NPY_OBJECT_SCALAR    # Python 对象标量
    
    # 定义枚举类型 NPY_SORTKIND，指定数组排序算法的类型
    ctypedef enum NPY_SORTKIND:
        NPY_QUICKSORT  # 快速排序
        NPY_HEAPSORT   # 堆排序
        NPY_MERGESORT  # 归并排序
    
    # 定义枚举类型 NPY_SEARCHSIDE，指定搜索算法的方向
    ctypedef enum NPY_SEARCHSIDE:
        NPY_SEARCHLEFT   # 从左边搜索
        NPY_SEARCHRIGHT  # 从右边搜索
    
    # 未命名的枚举变量，定义了一些已经废弃的 NumPy 标记常量，请勿在新代码中使用
    enum:
        NPY_C_CONTIGUOUS
        NPY_F_CONTIGUOUS
        NPY_CONTIGUOUS
        NPY_FORTRAN
        NPY_OWNDATA
        NPY_FORCECAST
        NPY_ENSURECOPY
        NPY_ENSUREARRAY
        NPY_ELEMENTSTRIDES
        NPY_ALIGNED
        NPY_NOTSWAPPED
        NPY_WRITEABLE
        NPY_ARR_HAS_DESCR
    
        NPY_BEHAVED
        NPY_BEHAVED_NS
        NPY_CARRAY
        NPY_CARRAY_RO
        NPY_FARRAY
        NPY_FARRAY_RO
        NPY_DEFAULT
    
        NPY_IN_ARRAY
        NPY_OUT_ARRAY
        NPY_INOUT_ARRAY
        NPY_IN_FARRAY
        NPY_OUT_FARRAY
        NPY_INOUT_FARRAY
    
        NPY_UPDATE_ALL
    enum:
        # NumPy 1.7 引入的枚举，用以替代上述已弃用的枚举。
        NPY_ARRAY_C_CONTIGUOUS
        NPY_ARRAY_F_CONTIGUOUS
        NPY_ARRAY_OWNDATA
        NPY_ARRAY_FORCECAST
        NPY_ARRAY_ENSURECOPY
        NPY_ARRAY_ENSUREARRAY
        NPY_ARRAY_ELEMENTSTRIDES
        NPY_ARRAY_ALIGNED
        NPY_ARRAY_NOTSWAPPED
        NPY_ARRAY_WRITEABLE
        NPY_ARRAY_WRITEBACKIFCOPY

        NPY_ARRAY_BEHAVED
        NPY_ARRAY_BEHAVED_NS
        NPY_ARRAY_CARRAY
        NPY_ARRAY_CARRAY_RO
        NPY_ARRAY_FARRAY
        NPY_ARRAY_FARRAY_RO
        NPY_ARRAY_DEFAULT

        NPY_ARRAY_IN_ARRAY
        NPY_ARRAY_OUT_ARRAY
        NPY_ARRAY_INOUT_ARRAY
        NPY_ARRAY_IN_FARRAY
        NPY_ARRAY_OUT_FARRAY
        NPY_ARRAY_INOUT_FARRAY

        NPY_ARRAY_UPDATE_ALL

    cdef enum:
        # NumPy 2.x 中为 64，在 NumPy 1.x 中为 32。
        NPY_MAXDIMS  # Used for dimensions in NumPy arrays
        NPY_RAVEL_AXIS  # 在类似 PyArray_Mean 的函数中使用的轴

    ctypedef void (*PyArray_VectorUnaryFunc)(void *, void *, npy_intp, void *,  void *)

    ctypedef struct PyArray_ArrayDescr:
        # shape 是一个元组，但是在非 PyObject* 声明中，Cython 不支持 "tuple shape"，因此我们将其声明为 PyObject*。
        PyObject* shape

    ctypedef struct PyArray_Descr:
        pass

    ctypedef class numpy.dtype [object PyArray_Descr, check_size ignore]:
        # 当可能时，使用 PyDataType_* 宏，但是对于一些字段，没有宏来访问，因此一些字段被定义了。
        cdef PyTypeObject* typeobj
        cdef char kind
        cdef char type
        # Numpy 有时会在不通知的情况下更改此字段（例如，在小端机器上，它有时会将 "|" 更改为 "<"），如果这对你很重要，请使用 PyArray_IsNativeByteOrder(dtype.byteorder) 而不是直接访问此字段。
        cdef char byteorder
        # Flags 在 Cython <3 中无法直接访问。使用 PyDataType_FLAGS。
        # cdef char flags
        cdef int type_num
        # itemsize/elsize、alignment、fields、names 和 subarray 必须使用 `PyDataType_*` 访问器宏。在 Cython 3 中，你仍然可以使用 getter 属性 `dtype.itemsize`。

    ctypedef class numpy.flatiter [object PyArrayIterObject, check_size ignore]:
        # 通过宏使用

    ctypedef class numpy.broadcast [object PyArrayMultiIterObject, check_size ignore]:
        cdef int numiter
        cdef npy_intp size, index
        cdef int nd
        cdef npy_intp *dimensions
        cdef void **iters

    ctypedef struct PyArrayObject:
        # 用于在无法替换 PyArrayObject*（如 PyArrayObject**）的情况下使用。
        pass
    ctypedef class numpy.ndarray [object PyArrayObject, check_size ignore]:
        # 定义一个 Cython 类，模拟 NumPy 的 ndarray 对象，这里忽略了一些不稳定的字段。
        cdef __cythonbufferdefaults__ = {"mode": "strided"}
    
        cdef:
            # data 指向数组数据的指针
            char *data
            # ndim 表示数组的维度数
            int ndim "nd"
            # shape 指向数组维度的指针
            npy_intp *shape "dimensions"
            # strides 指向数组步长的指针
            npy_intp *strides
            # descr 表示数组的数据类型，自 NumPy 1.7 起已不推荐使用
            dtype descr  # deprecated since NumPy 1.7 !
            # base 指向 ndarray 的基对象，不是公共接口，不建议使用
            PyObject* base #  NOT PUBLIC, DO NOT USE !
    
    
    int _import_array() except -1
    # 导入 NumPy C API，返回 -1 表示失败，0 表示成功
    # 这里使用 __pyx_import_array 来避免 _import_array 在使用时被标记为已使用
    
    int __pyx_import_array "_import_array"() except -1
    # 同名函数，用于防止 _import_array 在这里被标记为已使用
    
    #
    # 从 ndarrayobject.h 中提取的宏定义
    #
    
    bint PyArray_CHKFLAGS(ndarray m, int flags) nogil
    # 检查 ndarray 对象 m 的 flags 标志位是否满足条件，不涉及 GIL
    
    bint PyArray_IS_C_CONTIGUOUS(ndarray arr) nogil
    # 检查数组 arr 是否是 C 连续存储的，不涉及 GIL
    
    bint PyArray_IS_F_CONTIGUOUS(ndarray arr) nogil
    # 检查数组 arr 是否是 Fortran 连续存储的，不涉及 GIL
    
    bint PyArray_ISCONTIGUOUS(ndarray m) nogil
    # 检查数组 m 是否是连续存储的，不涉及 GIL
    
    bint PyArray_ISWRITEABLE(ndarray m) nogil
    # 检查数组 m 是否可写，不涉及 GIL
    
    bint PyArray_ISALIGNED(ndarray m) nogil
    # 检查数组 m 是否按照其数据类型的要求对齐，不涉及 GIL
    
    int PyArray_NDIM(ndarray) nogil
    # 返回数组的维度数，不涉及 GIL
    
    bint PyArray_ISONESEGMENT(ndarray) nogil
    # 检查数组是否是一段连续内存，不涉及 GIL
    
    bint PyArray_ISFORTRAN(ndarray) nogil
    # 检查数组是否是 Fortran 风格的存储，不涉及 GIL
    
    int PyArray_FORTRANIF(ndarray) nogil
    # 如果数组是 Fortran 风格的存储，返回 1，否则返回 0，不涉及 GIL
    
    void* PyArray_DATA(ndarray) nogil
    # 返回数组的数据指针，不涉及 GIL
    
    char* PyArray_BYTES(ndarray) nogil
    # 返回数组的数据字节流指针，不涉及 GIL
    
    npy_intp* PyArray_DIMS(ndarray) nogil
    # 返回数组的维度指针，不涉及 GIL
    
    npy_intp* PyArray_STRIDES(ndarray) nogil
    # 返回数组的步长指针，不涉及 GIL
    
    npy_intp PyArray_DIM(ndarray, size_t) nogil
    # 返回数组在指定维度上的大小，不涉及 GIL
    
    npy_intp PyArray_STRIDE(ndarray, size_t) nogil
    # 返回数组在指定维度上的步长，不涉及 GIL
    
    PyObject *PyArray_BASE(ndarray) nogil  # returns borrowed reference!
    # 返回数组的基对象，注意这是借用引用，不涉及 GIL
    
    PyArray_Descr *PyArray_DESCR(ndarray) nogil  # returns borrowed reference to dtype!
    # 返回数组的数据类型描述符，注意这是借用引用，不涉及 GIL
    
    PyArray_Descr *PyArray_DTYPE(ndarray) nogil  # returns borrowed reference to dtype! NP 1.7+ alias for descr.
    # 返回数组的数据类型描述符，这是从 NumPy 1.7+ 开始的别名，不涉及 GIL
    
    int PyArray_FLAGS(ndarray) nogil
    # 返回数组的标志位，不涉及 GIL
    
    void PyArray_CLEARFLAGS(ndarray, int flags) nogil  # Added in NumPy 1.7
    # 清除数组的指定标志位，不涉及 GIL
    
    void PyArray_ENABLEFLAGS(ndarray, int flags) nogil  # Added in NumPy 1.7
    # 启用数组的指定标志位，不涉及 GIL
    
    npy_intp PyArray_ITEMSIZE(ndarray) nogil
    # 返回数组中每个元素的字节大小，不涉及 GIL
    
    int PyArray_TYPE(ndarray arr) nogil
    # 返回数组的数据类型，不涉及 GIL
    
    object PyArray_GETITEM(ndarray arr, void *itemptr)
    # 从数组中获取指定位置的元素，不涉及 GIL
    
    int PyArray_SETITEM(ndarray arr, void *itemptr, object obj) except -1
    # 将对象 obj 设置到数组的指定位置，返回 -1 表示失败，不涉及 GIL
    
    bint PyTypeNum_ISBOOL(int) nogil
    # 检查给定的类型编号是否为布尔类型，不涉及 GIL
    
    bint PyTypeNum_ISUNSIGNED(int) nogil
    # 检查给定的类型编号是否为无符号整数类型，不涉及 GIL
    
    bint PyTypeNum_ISSIGNED(int) nogil
    # 检查给定的类型编号是否为有符号整数类型，不涉及 GIL
    
    bint PyTypeNum_ISINTEGER(int) nogil
    # 检查给定的类型编号是否为整数类型，不涉及 GIL
    
    bint PyTypeNum_ISFLOAT(int) nogil
    # 检查给定的类型编号是否为浮点数类型，不涉及 GIL
    
    bint PyTypeNum_ISNUMBER(int) nogil
    # 检查给定的类型编号是否为数字类型，不涉及 GIL
    
    bint PyTypeNum_ISSTRING(int) nogil
    # 检查给定的类型编号是否为字符串类型，不涉及 GIL
    
    bint PyTypeNum_ISCOMPLEX(int) nogil
    # 检查给定的类型编号是否为复数类型，不涉及 GIL
    
    bint PyTypeNum_ISFLEXIBLE(int) nogil
    # 检查给定的类型编号是否为灵活类型，不涉及 GIL
    
    bint PyTypeNum_ISUSERDEF(int) nogil
    # 检查给定的类型编号是否为用户定义类型，不涉及 GIL
    
    bint PyTypeNum_ISEXTENDED(int) nogil
    # 检查给定的类型编号是否为扩展类型，不涉及 GIL
    
    bint PyTypeNum_ISOBJECT(int) nogil
    # 检查给定的类型编号是否为对象类型，不涉及 GIL
    
    npy_intp PyDataType_ELSIZE(dtype) nogil
    # 返回数据类型的每个元素的字节大小，不涉及 GIL
    
    npy_intp PyDataType_ALIGNMENT(dtype) nogil
    # 返回数据类型的对齐方式，不涉及 GIL
    
    PyObject* PyDataType_METADATA(dtype) nogil
    # 返回数据类型的元数据，不涉及 GIL
    
    PyArray_ArrayDescr* PyDataType_SUBARRAY(dtype) nogil
    # 返回数据类型的子数组描述符，不涉及 GIL
    
    PyObject* PyDataType_NAMES(dtype) nogil
    # 返回数据类型的字段名，不涉及 GIL
    
    PyObject* PyDataType_FIELDS(dtype) nogil
    # 返回数据类型的字段信息，不涉及 GIL
    # 检查数据类型是否为布尔类型
    bint PyDataType_ISBOOL(dtype) nogil
    # 检查数据类型是否为无符号整数类型
    bint PyDataType_ISUNSIGNED(dtype) nogil
    # 检查数据类型是否为有符号整数类型
    bint PyDataType_ISSIGNED(dtype) nogil
    # 检查数据类型是否为整数类型（包括有符号和无符号）
    bint PyDataType_ISINTEGER(dtype) nogil
    # 检查数据类型是否为浮点数类型
    bint PyDataType_ISFLOAT(dtype) nogil
    # 检查数据类型是否为数字类型（包括整数和浮点数）
    bint PyDataType_ISNUMBER(dtype) nogil
    # 检查数据类型是否为字符串类型
    bint PyDataType_ISSTRING(dtype) nogil
    # 检查数据类型是否为复数类型
    bint PyDataType_ISCOMPLEX(dtype) nogil
    # 检查数据类型是否为灵活类型
    bint PyDataType_ISFLEXIBLE(dtype) nogil
    # 检查数据类型是否为用户定义类型
    bint PyDataType_ISUSERDEF(dtype) nogil
    # 检查数据类型是否为扩展类型
    bint PyDataType_ISEXTENDED(dtype) nogil
    # 检查数据类型是否为对象类型
    bint PyDataType_ISOBJECT(dtype) nogil
    # 检查数据类型是否具有字段
    bint PyDataType_HASFIELDS(dtype) nogil
    # 检查数据类型是否具有子数组
    bint PyDataType_HASSUBARRAY(dtype) nogil
    # 获取数据类型的标志位
    npy_uint64 PyDataType_FLAGS(dtype) nogil
    
    # 检查 ndarray 是否为布尔类型数组
    bint PyArray_ISBOOL(ndarray) nogil
    # 检查 ndarray 是否为无符号整数类型数组
    bint PyArray_ISUNSIGNED(ndarray) nogil
    # 检查 ndarray 是否为有符号整数类型数组
    bint PyArray_ISSIGNED(ndarray) nogil
    # 检查 ndarray 是否为整数类型数组（包括有符号和无符号）
    bint PyArray_ISINTEGER(ndarray) nogil
    # 检查 ndarray 是否为浮点数类型数组
    bint PyArray_ISFLOAT(ndarray) nogil
    # 检查 ndarray 是否为数字类型数组（包括整数和浮点数）
    bint PyArray_ISNUMBER(ndarray) nogil
    # 检查 ndarray 是否为字符串类型数组
    bint PyArray_ISSTRING(ndarray) nogil
    # 检查 ndarray 是否为复数类型数组
    bint PyArray_ISCOMPLEX(ndarray) nogil
    # 检查 ndarray 是否为灵活类型数组
    bint PyArray_ISFLEXIBLE(ndarray) nogil
    # 检查 ndarray 是否为用户定义类型数组
    bint PyArray_ISUSERDEF(ndarray) nogil
    # 检查 ndarray 是否为扩展类型数组
    bint PyArray_ISEXTENDED(ndarray) nogil
    # 检查 ndarray 是否为对象类型数组
    bint PyArray_ISOBJECT(ndarray) nogil
    # 检查 ndarray 是否具有字段
    bint PyArray_HASFIELDS(ndarray) nogil
    
    # 检查 ndarray 是否为可变数组（维度不固定）
    bint PyArray_ISVARIABLE(ndarray) nogil
    # 安全地复制内存对齐的 ndarray
    bint PyArray_SAFEALIGNEDCOPY(ndarray) nogil
    # 检查字符是否符合 ndarray 的字节顺序
    bint PyArray_ISNBO(char) nogil              # 适用于 ndarray.byteorder
    # 检查字符是否为本机字节顺序
    bint PyArray_IsNativeByteOrder(char) nogil # 适用于 ndarray.byteorder
    # 检查 ndarray 是否未交换字节顺序
    bint PyArray_ISNOTSWAPPED(ndarray) nogil
    # 检查 ndarray 是否已交换字节顺序
    bint PyArray_ISBYTESWAPPED(ndarray) nogil
    
    # 在给定标志位下交换 ndarray 的字节顺序
    bint PyArray_FLAGSWAP(ndarray, int) nogil
    
    # 检查 ndarray 是否为 C 连续存储
    bint PyArray_ISCARRAY(ndarray) nogil
    # 检查 ndarray 是否为只读的 C 连续存储
    bint PyArray_ISCARRAY_RO(ndarray) nogil
    # 检查 ndarray 是否为 Fortran 连续存储
    bint PyArray_ISFARRAY(ndarray) nogil
    # 检查 ndarray 是否为只读的 Fortran 连续存储
    bint PyArray_ISFARRAY_RO(ndarray) nogil
    # 检查 ndarray 是否表现良好（C 或 Fortran 连续存储）
    bint PyArray_ISBEHAVED(ndarray) nogil
    # 检查 ndarray 是否为只读的表现良好数组（C 或 Fortran 连续存储）
    bint PyArray_ISBEHAVED_RO(ndarray) nogil
    
    # 检查数据类型是否未交换字节顺序
    bint PyDataType_ISNOTSWAPPED(dtype) nogil
    # 检查数据类型是否已交换字节顺序
    bint PyDataType_ISBYTESWAPPED(dtype) nogil
    
    # 检查对象是否符合数组描述符
    bint PyArray_DescrCheck(object)
    
    # 检查对象是否为 ndarray
    bint PyArray_Check(object)
    # 检查对象是否为精确的 ndarray
    bint PyArray_CheckExact(object)
    
    # 检查对象是否为零维数组
    bint PyArray_IsZeroDim(object)
    # 检查对象是否为标量数组
    bint PyArray_CheckScalar(object)
    # 检查对象是否为 Python 数字
    bint PyArray_IsPythonNumber(object)
    # 检查对象是否为 Python 标量
    bint PyArray_IsPythonScalar(object)
    # 检查对象是否为任意标量
    bint PyArray_IsAnyScalar(object)
    # 检查对象是否为任意标量数组
    bint PyArray_CheckAnyScalar(object)
    
    # 获取连续存储的 ndarray
    ndarray PyArray_GETCONTIGUOUS(ndarray)
    # 检查两个 ndarray 是否具有相同的形状
    bint PyArray_SAMESHAPE(ndarray, ndarray) nogil
    # 获取 ndarray 的元素个数
    npy_intp PyArray_SIZE(ndarray) nogil
    # 获取 ndarray 的总字节数
    npy_intp PyArray_NBYTES(ndarray) nogil
    
    # 从对象创建 ndarray
    object PyArray_FROM_O(object)
    # 根据指定标志位从对象创建 ndarray
    object PyArray_FROM_OF(object m, int flags)
    # 根据指定数据类型从对象创建 ndarray
    object PyArray_FROM_OT(object m, int type)
    # 根据指定数据类型和标志位从对象创建 ndarray
    object PyArray_FROM_OTF(object m, int type, int flags)
    # 根据指定数据类型范围和标志位从对象创建 ndarray
    object PyArray_FROMANY(object m, int type, int min, int max, int flags)
    # 创建一个具有指定维度、类型和存储顺序的全零 NumPy 数组
    object PyArray_ZEROS(int nd, npy_intp* dims, int type, int fortran)
    
    # 创建一个具有指定维度、类型和存储顺序的空 NumPy 数组
    object PyArray_EMPTY(int nd, npy_intp* dims, int type, int fortran)
    
    # 用指定的值填充一个 NumPy 数组
    void PyArray_FILLWBYTE(ndarray, int val)
    
    # 从任意对象创建一个连续存储的 NumPy 数组
    object PyArray_ContiguousFromAny(op, int, int min_depth, int max_depth)
    
    # 检查两个数组的类型是否等价
    unsigned char PyArray_EquivArrTypes(ndarray a1, ndarray a2)
    
    # 检查两个字节顺序是否等价
    bint PyArray_EquivByteorders(int b1, int b2) nogil
    
    # 创建一个简单的 NumPy 数组，不初始化其数据
    object PyArray_SimpleNew(int nd, npy_intp* dims, int typenum)
    
    # 使用给定的数据创建一个简单的 NumPy 数组
    object PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data)
    
    # 将数组的数据转换为标量对象
    object PyArray_ToScalar(void* data, ndarray arr)
    
    # 获取一维数组的指定索引处的数据指针
    void* PyArray_GETPTR1(ndarray m, npy_intp i) nogil
    
    # 获取二维数组的指定索引处的数据指针
    void* PyArray_GETPTR2(ndarray m, npy_intp i, npy_intp j) nogil
    
    # 获取三维数组的指定索引处的数据指针
    void* PyArray_GETPTR3(ndarray m, npy_intp i, npy_intp j, npy_intp k) nogil
    
    # 获取四维数组的指定索引处的数据指针
    void* PyArray_GETPTR4(ndarray m, npy_intp i, npy_intp j, npy_intp k, npy_intp l) nogil
    
    # 复制一个 NumPy 数组
    object PyArray_Copy(ndarray)
    
    # 从 Python 对象创建一个 NumPy 数组
    object PyArray_FromObject(object op, int type, int min_depth, int max_depth)
    
    # 从 Python 对象创建一个连续存储的 NumPy 数组
    object PyArray_ContiguousFromObject(object op, int type, int min_depth, int max_depth)
    
    # 从 Python 对象复制数据到 NumPy 数组
    object PyArray_CopyFromObject(object op, int type, int min_depth, int max_depth)
    
    # 将数组转换为指定类型的 NumPy 数组
    object PyArray_Cast(ndarray mp, int type_num)
    
    # 在指定轴上按照给定的索引数组获取数组的子集
    object PyArray_Take(ndarray ap, object items, int axis)
    
    # 在指定位置上按照给定的索引数组放置值到数组中
    object PyArray_Put(ndarray ap, object items, object values)
    
    # 将迭代器重置到起始位置
    void PyArray_ITER_RESET(flatiter it) nogil
    
    # 将迭代器移动到下一个位置
    void PyArray_ITER_NEXT(flatiter it) nogil
    
    # 将迭代器移动到指定位置
    void PyArray_ITER_GOTO(flatiter it, npy_intp* destination) nogil
    
    # 将迭代器移动到一维数组中的指定索引处
    void PyArray_ITER_GOTO1D(flatiter it, npy_intp ind) nogil
    
    # 获取迭代器当前位置的数据指针
    void* PyArray_ITER_DATA(flatiter it) nogil
    
    # 检查迭代器是否还未遍历完所有元素
    bint PyArray_ITER_NOTDONE(flatiter it) nogil
    
    # 将多重迭代器重置到起始位置
    void PyArray_MultiIter_RESET(broadcast multi) nogil
    
    # 将多重迭代器移动到下一个位置
    void PyArray_MultiIter_NEXT(broadcast multi) nogil
    
    # 将多重迭代器移动到指定位置
    void PyArray_MultiIter_GOTO(broadcast multi, npy_intp dest) nogil
    
    # 将多重迭代器移动到一维数组中的指定索引处
    void PyArray_MultiIter_GOTO1D(broadcast multi, npy_intp ind) nogil
    
    # 获取多重迭代器当前位置的数据指针
    void* PyArray_MultiIter_DATA(broadcast multi, npy_intp i) nogil
    
    # 将多重迭代器中的指定迭代器移动到下一个位置
    void PyArray_MultiIter_NEXTi(broadcast multi, npy_intp i) nogil
    
    # 检查多重迭代器是否还未遍历完所有元素
    bint PyArray_MultiIter_NOTDONE(broadcast multi) nogil
    
    # 获取多重迭代器中的数组大小
    npy_intp PyArray_MultiIter_SIZE(broadcast multi) nogil
    
    # 获取多重迭代器中的数组维度数目
    int PyArray_MultiIter_NDIM(broadcast multi) nogil
    
    # 获取多重迭代器当前索引
    npy_intp PyArray_MultiIter_INDEX(broadcast multi) nogil
    
    # 获取多重迭代器中的各维度大小
    npy_intp* PyArray_MultiIter_DIMS(broadcast multi) nogil
    
    # 获取多重迭代器中的所有迭代器
    void** PyArray_MultiIter_ITERS(broadcast multi) nogil
    
    # 增加数组对象的引用计数
    int PyArray_INCREF(ndarray) except *  # 使用 PyArray_Item_INCREF... 增加引用计数
    # 减少 ndarray 对象的引用计数，可能会释放其占用的内存空间
    int PyArray_XDECREF(ndarray) except *
    
    # 根据给定的整数类型创建一个描述符对象
    dtype PyArray_DescrFromType(int)
    
    # 根据给定的整数类型创建一个数组类型对象
    object PyArray_TypeObjectFromType(int)
    
    # 将给定的 ndarray 对象的所有元素置为零
    char * PyArray_Zero(ndarray)
    
    # 将给定的 ndarray 对象的所有元素置为一
    char * PyArray_One(ndarray)
    
    # 判断是否可以安全地将一个整数类型转换为另一个整数类型，可能会写入错误信息
    int PyArray_CanCastSafely(int, int)
    
    # 判断是否可以将一个数据类型安全地转换为另一个数据类型，可能会写入错误信息
    npy_bool PyArray_CanCastTo(dtype, dtype)
    
    # 返回对象的数据类型编号，如果失败则返回 0
    int PyArray_ObjectType(object, int) except 0
    
    # 根据对象推断其对应的描述符对象
    dtype PyArray_DescrFromObject(object, dtype)
    
    # 返回标量对象的描述符对象
    dtype PyArray_DescrFromScalar(object)
    
    # 根据数据类型对象返回其对应的描述符对象
    dtype PyArray_DescrFromTypeObject(object)
    
    # 返回对象的总大小
    npy_intp PyArray_Size(object)
    
    # 将标量对象转换为 C 类型
    void PyArray_ScalarAsCtype(object, void *)
    
    # 返回对象的内存布局优先级
    double PyArray_GetPriority(object, double)  # 清除错误，版本 1.25 后
    
    # 创建一个迭代器对象，用于遍历数组对象
    object PyArray_IterNew(object)
    
    # 创建一个多迭代器对象，用于多数组并行遍历
    object PyArray_MultiIterNew(int, ...)
    
    # 将 Python 中的整数对象转换为 C 语言中的 int 类型，可能会出错返回 -1
    int PyArray_PyIntAsInt(object) except? -1
    
    # 将 Python 中的整数对象转换为 C 语言中的 npy_intp 类型
    npy_intp PyArray_PyIntAsIntp(object)
    
    # 广播数组以匹配指定形状
    int PyArray_Broadcast(broadcast) except -1
    
    # 使用标量对象填充数组的所有元素
    int PyArray_FillWithScalar(ndarray, object) except -1
    
    # 检查数组是否具有正确的步幅
    npy_bool PyArray_CheckStrides(int, int, npy_intp, npy_intp, npy_intp *, npy_intp *)
    # 创建一个新的描述符对象，并设置其字节顺序
    dtype PyArray_DescrNewByteorder (dtype, char)

    # 返回一个迭代器对象，该对象迭代除了指定轴以外的所有元素
    object PyArray_IterAllButAxis (object, int *)

    # 根据给定参数创建一个数组对象，可以接受各种数据类型和格式
    #object PyArray_CheckFromAny (object, dtype, int, int, int, object)

    # 根据给定的 ndarray 对象创建一个新的数组对象，可以指定数据类型
    #object PyArray_FromArray (ndarray, dtype, int)

    # 根据给定的接口对象创建一个数组对象
    object PyArray_FromInterface (object)

    # 根据给定的结构化接口对象创建一个数组对象
    object PyArray_FromStructInterface (object)

    # 根据给定的参数判断是否可以将一个标量转换为指定的数据类型
    #object PyArray_FromArrayAttr (object, dtype, object)

    # 返回给定数据类型和标量类型对应的标量类型种类
    #NPY_SCALARKIND PyArray_ScalarKind (int, ndarray*)

    # 检查是否可以将一个数据类型转换为另一个数据类型
    int PyArray_CanCoerceScalar (int, int, NPY_SCALARKIND)

    # 检查是否可以将一个标量值从一种数据类型转换为另一种数据类型
    npy_bool PyArray_CanCastScalar (type, type)

    # 移除最小的广播形状
    int PyArray_RemoveSmallest (broadcast) except -1

    # 计算给定对象的元素步长
    int PyArray_ElementStrides (object)

    # 增加数组元素的引用计数
    void PyArray_Item_INCREF (char *, dtype) except *

    # 减少数组元素的引用计数
    void PyArray_Item_XDECREF (char *, dtype) except *

    # 返回数组的转置
    object PyArray_Transpose (ndarray, PyArray_Dims *)

    # 根据给定的索引数组从数组中取值
    object PyArray_TakeFrom (ndarray, object, int, ndarray, NPY_CLIPMODE)

    # 将数组中的值放置到给定的索引位置
    object PyArray_PutTo (ndarray, object, object, NPY_CLIPMODE)

    # 根据掩码数组将值放置到数组中
    object PyArray_PutMask (ndarray, object, object)

    # 返回重复给定数组后的结果数组
    object PyArray_Repeat (ndarray, object, int)

    # 根据给定数组和索引数组选择值
    object PyArray_Choose (ndarray, object, ndarray, NPY_CLIPMODE)

    # 对数组进行排序
    int PyArray_Sort (ndarray, int, NPY_SORTKIND) except -1

    # 返回数组排序的索引
    object PyArray_ArgSort (ndarray, int, NPY_SORTKIND)

    # 在数组中搜索指定值，并返回索引
    object PyArray_SearchSorted (ndarray, object, NPY_SEARCHSIDE, PyObject *)

    # 返回数组的最大值的索引
    object PyArray_ArgMax (ndarray, int, ndarray)

    # 返回数组的最小值的索引
    object PyArray_ArgMin (ndarray, int, ndarray)

    # 返回给定形状的数组视图
    object PyArray_Reshape (ndarray, object)

    # 返回给定形状和顺序的新数组对象
    object PyArray_Newshape (ndarray, PyArray_Dims *, NPY_ORDER)

    # 返回去除长度为 1 的轴后的数组视图
    object PyArray_Squeeze (ndarray)

    # 返回数组的视图，不改变数组数据的形状
    #object PyArray_View (ndarray, dtype, type)

    # 交换数组的两个轴
    object PyArray_SwapAxes (ndarray, int, int)

    # 返回数组在指定轴上的最大值
    object PyArray_Max (ndarray, int, ndarray)

    # 返回数组在指定轴上的最小值
    object PyArray_Min (ndarray, int, ndarray)

    # 返回数组在指定轴上的最大值与最小值之差
    object PyArray_Ptp (ndarray, int, ndarray)

    # 返回数组在指定轴上的均值
    object PyArray_Mean (ndarray, int, int, ndarray)

    # 返回数组的对角线元素
    object PyArray_Trace (ndarray, int, int, int, int, ndarray)

    # 返回数组的对角线
    object PyArray_Diagonal (ndarray, int, int, int)

    # 返回数组在指定范围内的数值剪裁结果
    object PyArray_Clip (ndarray, object, object, ndarray)

    # 返回数组的共轭视图
    object PyArray_Conjugate (ndarray, ndarray)

    # 返回数组中非零元素的索引
    object PyArray_Nonzero (ndarray)

    # 返回数组在指定轴上的标准差
    object PyArray_Std (ndarray, int, int, ndarray, int)

    # 返回数组在指定轴上的总和
    object PyArray_Sum (ndarray, int, int, ndarray)

    # 返回数组在指定轴上的累积和
    object PyArray_CumSum (ndarray, int, int, ndarray)

    # 返回数组在指定轴上的积
    object PyArray_Prod (ndarray, int, int, ndarray)

    # 返回数组在指定轴上的累积积
    object PyArray_CumProd (ndarray, int, int, ndarray)

    # 判断数组中所有元素是否为真
    object PyArray_All (ndarray, int, ndarray)

    # 判断数组中是否有任意一个元素为真
    object PyArray_Any (ndarray, int, ndarray)

    # 返回根据条件数组对数组进行压缩后的结果数组
    object PyArray_Compress (ndarray, object, int, ndarray)

    # 返回按照指定顺序展平数组的结果数组
    object PyArray_Flatten (ndarray, NPY_ORDER)

    # 返回按照指定顺序展平数组的结果视图
    object PyArray_Ravel (ndarray, NPY_ORDER)

    # 返回给定整数列表中所有元素的乘积
    npy_intp PyArray_MultiplyList (npy_intp *, int)

    # 返回给定整数列表中所有整数的乘积
    int PyArray_MultiplyIntList (int *, int)

    # 返回数组中指定索引处元素的指针
    void * PyArray_GetPtr (ndarray, npy_intp*)

    # 比较两个整数列表是否相等
    int PyArray_CompareLists (npy_intp *, npy_intp *, int)

    # 将一个对象转换为 C 风格数组
    #int PyArray_AsCArray (object*, void *, npy_intp *, int, dtype)

    # 释放使用 PyArray_AsCArray() 分配的内存
    int PyArray_Free (object, void *)

    # 转换一个对象为数组
    #int PyArray_Converter (object, object*)
    int PyArray_IntpFromSequence (object, npy_intp *, int) except -1
        # 从 Python 序列中提取整数数组，存储在 npy_intp 类型的数组中
        # 参数：
        #   object: 输入的 Python 对象，应当是一个序列
        #   npy_intp *: 用于存储整数的数组指针
        #   int: 数组的长度
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    object PyArray_Concatenate (object, int)
        # 拼接多个数组成一个数组
        # 参数：
        #   object: 输入的 Python 对象，应当是包含待拼接数组的元组
        #   int: 要拼接的数组的轴（维度）
        # 返回值：
        #   拼接后的数组对象
    
    object PyArray_InnerProduct (object, object)
        # 计算两个数组的内积
        # 参数：
        #   object: 第一个输入的数组对象
        #   object: 第二个输入的数组对象
        # 返回值：
        #   内积的结果数组对象
    
    object PyArray_MatrixProduct (object, object)
        # 计算两个数组的矩阵乘积
        # 参数：
        #   object: 第一个输入的数组对象
        #   object: 第二个输入的数组对象
        # 返回值：
        #   矩阵乘积的结果数组对象
    
    object PyArray_Correlate (object, object, int)
        # 计算两个数组的相关性
        # 参数：
        #   object: 第一个输入的数组对象
        #   object: 第二个输入的数组对象
        #   int: 相关计算的模式
        # 返回值：
        #   相关性计算结果的数组对象
    
    #int PyArray_DescrConverter (object, dtype*) except 0
    #int PyArray_DescrConverter2 (object, dtype*) except 0
        # 用于转换数组描述符的函数，已废弃
    
    int PyArray_IntpConverter (object, PyArray_Dims *) except 0
        # 将 Python 对象转换为 PyArray_Dims 结构，该结构用于表示数组的维度信息
        # 参数：
        #   object: 输入的 Python 对象
        #   PyArray_Dims *: 用于存储维度信息的指针
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    #int PyArray_BufferConverter (object, chunk) except 0
        # 用于转换缓冲区对象的函数，已废弃
    
    int PyArray_AxisConverter (object, int *) except 0
        # 将 Python 对象转换为整数，表示数组的轴
        # 参数：
        #   object: 输入的 Python 对象
        #   int *: 用于存储轴值的指针
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    int PyArray_BoolConverter (object, npy_bool *) except 0
        # 将 Python 对象转换为布尔值，表示数组的布尔值
        # 参数：
        #   object: 输入的 Python 对象
        #   npy_bool *: 用于存储布尔值的指针
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    int PyArray_ByteorderConverter (object, char *) except 0
        # 将 Python 对象转换为字符，表示数组的字节顺序
        # 参数：
        #   object: 输入的 Python 对象
        #   char *: 用于存储字节顺序的指针
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    int PyArray_OrderConverter (object, NPY_ORDER *) except 0
        # 将 Python 对象转换为 NPY_ORDER 枚举值，表示数组的存储顺序
        # 参数：
        #   object: 输入的 Python 对象
        #   NPY_ORDER *: 用于存储存储顺序的指针
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    unsigned char PyArray_EquivTypes (dtype, dtype)
        # 检查两个数据类型是否等效
        # 参数：
        #   dtype: 第一个数据类型
        #   dtype: 第二个数据类型
        # 返回值：
        #   若数据类型等效则返回非零值，否则返回零
    
    #object PyArray_Zeros (int, npy_intp *, dtype, int)
    #object PyArray_Empty (int, npy_intp *, dtype, int)
        # 创建数组对象的函数，已废弃
    
    object PyArray_Where (object, object, object)
        # 根据条件返回数组中满足条件的元素
        # 参数：
        #   object: 条件数组对象
        #   object: 真值数组对象
        #   object: 假值数组对象
        # 返回值：
        #   满足条件的元素组成的数组对象
    
    object PyArray_Arange (double, double, double, int)
        # 创建等差数列的数组对象
        # 参数：
        #   double: 开始值
        #   double: 结束值
        #   double: 步长
        #   int: 数据类型
        # 返回值：
        #   创建的等差数列数组对象
    
    int PyArray_SortkindConverter (object, NPY_SORTKIND *) except 0
        # 将 Python 对象转换为 NPY_SORTKIND 枚举值，表示排序方式
        # 参数：
        #   object: 输入的 Python 对象
        #   NPY_SORTKIND *: 用于存储排序方式的指针
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    object PyArray_LexSort (object, int)
        # 对多个数组进行词典排序
        # 参数：
        #   object: 输入的元组，包含待排序的多个数组对象
        #   int: 用于指定排序的轴
        # 返回值：
        #   排序结果的数组对象
    
    object PyArray_Round (ndarray, int, ndarray)
        # 对数组进行四舍五入
        # 参数：
        #   ndarray: 输入的数组对象
        #   int: 四舍五入的小数位数
        #   ndarray: 用于存储结果的数组对象
        # 返回值：
        #   四舍五入后的数组对象
    
    unsigned char PyArray_EquivTypenums (int, int)
        # 检查两个数据类型编号是否等效
        # 参数：
        #   int: 第一个数据类型编号
        #   int: 第二个数据类型编号
        # 返回值：
        #   若数据类型编号等效则返回非零值，否则返回零
    
    int PyArray_RegisterDataType (dtype) except -1
        # 注册新的数据类型
        # 参数：
        #   dtype: 待注册的数据类型
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    int PyArray_RegisterCastFunc (dtype, int, PyArray_VectorUnaryFunc *) except -1
        # 注册新的类型转换函数
        # 参数：
        #   dtype: 待注册的数据类型
        #   int: 源数据类型编号
        #   PyArray_VectorUnaryFunc *: 类型转换函数指针
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    int PyArray_RegisterCanCast (dtype, int, NPY_SCALARKIND) except -1
        # 注册新的数据类型转换能力
        # 参数：
        #   dtype: 目标数据类型
        #   int: 源数据类型编号
        #   NPY_SCALARKIND: 标量类型
        # 返回值：
        #   成功时返回 0，失败时返回 -1
    
    #void PyArray_InitArrFuncs (PyArray_ArrFuncs *)
        # 初始化数组函数，已废弃
    
    object PyArray_IntTupleFromIntp (int, npy_intp *)
        # 将整数数组转换为整数元组
        # 参数：
        #   int: 数组的长度
        #   npy_intp *: 输入的整数数组
        # 返回值：
# Typedefs that matches the runtime dtype objects in
# the numpy module.

# The ones that are commented out needs an IFDEF function
# in Cython to enable them only on the right systems.

# 定义与 numpy 模块中运行时数据类型对象相匹配的类型别名

ctypedef npy_int8       int8_t
ctypedef npy_int16      int16_t
ctypedef npy_int32      int32_t
ctypedef npy_int64      int64_t
#ctypedef npy_int96      int96_t
#ctypedef npy_int128     int128_t

ctypedef npy_uint8      uint8_t
ctypedef npy_uint16     uint16_t
ctypedef npy_uint32     uint32_t
ctypedef npy_uint64     uint64_t
#ctypedef npy_uint96     uint96_t
#ctypedef npy_uint128    uint128_t

ctypedef npy_float32    float32_t
ctypedef npy_float64    float64_t
#ctypedef npy_float80    float80_t
#ctypedef npy_float128   float128_t

ctypedef float complex  complex64_t
ctypedef double complex complex128_t

ctypedef npy_longlong   longlong_t
ctypedef npy_ulonglong  ulonglong_t

ctypedef npy_intp       intp_t
ctypedef npy_uintp      uintp_t

ctypedef npy_double     float_t
ctypedef npy_double     double_t
ctypedef npy_longdouble longdouble_t

ctypedef float complex       cfloat_t
ctypedef double complex      cdouble_t
ctypedef double complex      complex_t
ctypedef long double complex clongdouble_t

# 定义各种数值类型的类型别名，以匹配 numpy 模块中的相应数据类型

cdef inline object PyArray_MultiIterNew1(a):
    return PyArray_MultiIterNew(1, <void*>a)

cdef inline object PyArray_MultiIterNew2(a, b):
    return PyArray_MultiIterNew(2, <void*>a, <void*>b)

cdef inline object PyArray_MultiIterNew3(a, b, c):
    return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)

cdef inline object PyArray_MultiIterNew4(a, b, c, d):
    return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)

cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
    return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)

# 定义 Cython 中的内联函数，用于创建多迭代器对象，参数数量从1到5不等

cdef inline tuple PyDataType_SHAPE(dtype d):
    if PyDataType_HASSUBARRAY(d):
        return <tuple>d.subarray.shape
    else:
        return ()

# 定义 Cython 中的内联函数，用于获取给定数据类型的形状信息，若是子数组则返回其形状，否则返回空元组

cdef extern from "numpy/ndarrayobject.h":
    PyTypeObject PyTimedeltaArrType_Type
    PyTypeObject PyDatetimeArrType_Type
    ctypedef int64_t npy_timedelta
    ctypedef int64_t npy_datetime

# 从 numpy/ndarrayobject.h 头文件导入内容：定义了时间增量和日期时间类型的 PyTypeObject 结构体，以及对应的时间增量和日期时间的类型别名

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base
        int64_t num

    ctypedef struct npy_datetimestruct:
        int64_t year
        int32_t month, day, hour, min, sec, us, ps, as

# 从 numpy/ndarraytypes.h 头文件导入内容：定义了日期时间元数据结构 PyArray_DatetimeMetaData 和日期时间结构 npy_datetimestruct 的结构体定义

cdef extern from "numpy/arrayscalars.h":

    # abstract types
    ctypedef class numpy.generic [object PyObject]:
        pass
    ctypedef class numpy.number [object PyObject]:
        pass
    ctypedef class numpy.integer [object PyObject]:
        pass
    ctypedef class numpy.signedinteger [object PyObject]:
        pass
    ctypedef class numpy.unsignedinteger [object PyObject]:
        pass
    ctypedef class numpy.inexact [object PyObject]:
        pass
    ctypedef class numpy.floating [object PyObject]:
        pass
    ctypedef class numpy.complexfloating [object PyObject]:
        pass

# 从 numpy/arrayscalars.h 头文件导入内容：定义了一系列 numpy 抽象类型的类定义
    # 定义一个 Cython 类型的声明，表示 numpy 的 flexible 类型，继承自 object PyObject
    ctypedef class numpy.flexible [object PyObject]:
        pass
    
    # 定义一个 Cython 类型的声明，表示 numpy 的 character 类型，继承自 object PyObject
    ctypedef class numpy.character [object PyObject]:
        pass
    
    # 定义一个 Cython 结构体，表示 Python 中的 datetime 标量对象
    ctypedef struct PyDatetimeScalarObject:
        # PyObject_HEAD
        # 日期时间值
        npy_datetime obval
        # 日期时间的元数据
        PyArray_DatetimeMetaData obmeta
    
    # 定义一个 Cython 结构体，表示 Python 中的 timedelta 标量对象
    ctypedef struct PyTimedeltaScalarObject:
        # PyObject_HEAD
        # 时间增量值
        npy_timedelta obval
        # 时间增量的元数据
        PyArray_DatetimeMetaData obmeta
    
    # 定义一个 Cython 枚举类型，表示 numpy 的日期时间单位
    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_Y   # 年
        NPY_FR_M   # 月
        NPY_FR_W   # 周
        NPY_FR_D   # 日
        NPY_FR_B   # 工作日
        NPY_FR_h   # 小时
        NPY_FR_m   # 分钟
        NPY_FR_s   # 秒
        NPY_FR_ms  # 毫秒
        NPY_FR_us  # 微秒
        NPY_FR_ns  # 纳秒
        NPY_FR_ps  # 皮秒
        NPY_FR_fs  # 飞秒
        NPY_FR_as  # 阿秒
        NPY_FR_GENERIC  # 通用时间单位
cdef extern from "numpy/arrayobject.h":
    # 导入 NumPy 的 C-API 头文件 "numpy/arrayobject.h"

    # 定义在 datetime_strings.c 中的 NumPy 内部函数：
    # 返回 ISO 8601 格式日期时间字符串的长度
    int get_datetime_iso_8601_strlen "NpyDatetime_GetDatetimeISO8601StrLen" (
            int local, NPY_DATETIMEUNIT base)
    
    # 创建 ISO 8601 格式日期时间字符串
    int make_iso_8601_datetime "NpyDatetime_MakeISO8601Datetime" (
            npy_datetimestruct *dts, char *outstr, npy_intp outlen,
            int local, int utc, NPY_DATETIMEUNIT base, int tzoffset,
            NPY_CASTING casting) except -1

    # 定义在 datetime.c 中的 NumPy 内部函数：
    # 将 Python 的 datetime 对象转换为 npy_datetimestruct 结构
    int convert_pydatetime_to_datetimestruct "NpyDatetime_ConvertPyDateTimeToDatetimeStruct" (
            PyObject *obj, npy_datetimestruct *out,
            NPY_DATETIMEUNIT *out_bestunit, int apply_tzinfo) except -1
    
    # 将 datetime64 转换为 npy_datetimestruct 结构
    int convert_datetime64_to_datetimestruct "NpyDatetime_ConvertDatetime64ToDatetimeStruct" (
            PyArray_DatetimeMetaData *meta, npy_datetime dt,
            npy_datetimestruct *out) except -1
    
    # 将 npy_datetimestruct 结构转换为 datetime64
    int convert_datetimestruct_to_datetime64 "NpyDatetime_ConvertDatetimeStructToDatetime64"(
            PyArray_DatetimeMetaData *meta, const npy_datetimestruct *dts,
            npy_datetime *out) except -1


#
# ufunc API
#

cdef extern from "numpy/ufuncobject.h":
    # 导入 NumPy 的 C-API 头文件 "numpy/ufuncobject.h"

    ctypedef void (*PyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *)

    ctypedef class numpy.ufunc [object PyUFuncObject, check_size ignore]:
        cdef:
            int nin, nout, nargs
            int identity
            PyUFuncGenericFunction *functions
            void **data
            int ntypes
            int check_return
            char *name
            char *types
            char *doc
            void *ptr
            PyObject *obj
            PyObject *userloops

    cdef enum:
        PyUFunc_Zero
        PyUFunc_One
        PyUFunc_None
        UFUNC_FPE_DIVIDEBYZERO
        UFUNC_FPE_OVERFLOW
        UFUNC_FPE_UNDERFLOW
        UFUNC_FPE_INVALID

    # 创建一个 ufunc 对象并注册其函数和数据
    object PyUFunc_FromFuncAndData(PyUFuncGenericFunction *,
          void **, char *, int, int, int, int, char *, char *, int)
    
    # 为特定数据类型注册一个循环函数
    int PyUFunc_RegisterLoopForType(ufunc, int,
                                    PyUFuncGenericFunction, int *, void *) except -1
    
    # 定义一系列 ufunc 函数，每个函数对应不同的数据类型转换和操作
    void PyUFunc_f_f_As_d_d \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_d_d \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_f_f \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_g_g \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_F_F_As_D_D \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_F_F \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_D_D \
         (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_G_G \
         (char **, npy_intp *, npy_intp *, void *)
    # 定义接受一个字符指针、两个整型指针和一个空指针参数的 PyUFunc_O_O 函数
    void PyUFunc_O_O \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受六个字符指针、三个整型指针和一个空指针参数的 PyUFunc_ff_f_As_dd_d 函数
    void PyUFunc_ff_f_As_dd_d \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受三个字符指针、两个整型指针和一个空指针参数的 PyUFunc_ff_f 函数
    void PyUFunc_ff_f \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受三个字符指针、两个整型指针和一个空指针参数的 PyUFunc_dd_d 函数
    void PyUFunc_dd_d \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受三个字符指针、两个整型指针和一个空指针参数的 PyUFunc_gg_g 函数
    void PyUFunc_gg_g \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受六个字符指针、三个整型指针和一个空指针参数的 PyUFunc_FF_F_As_DD_D 函数
    void PyUFunc_FF_F_As_DD_D \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受三个字符指针、两个整型指针和一个空指针参数的 PyUFunc_DD_D 函数
    void PyUFunc_DD_D \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受三个字符指针、两个整型指针和一个空指针参数的 PyUFunc_FF_F 函数
    void PyUFunc_FF_F \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受三个字符指针、两个整型指针和一个空指针参数的 PyUFunc_GG_G 函数
    void PyUFunc_GG_G \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受一个字符指针、两个整型指针和一个空指针参数的 PyUFunc_OO_O 函数
    void PyUFunc_OO_O \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受一个字符指针、两个整型指针和一个空指针参数的 PyUFunc_O_O_method 函数
    void PyUFunc_O_O_method \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受两个字符指针、两个整型指针和一个空指针参数的 PyUFunc_OO_O_method 函数
    void PyUFunc_OO_O_method \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义接受一个字符指针、两个整型指针和一个空指针参数的 PyUFunc_On_Om 函数
    void PyUFunc_On_Om \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 声明清除浮点错误状态的 PyUFunc_clearfperr 函数
    void PyUFunc_clearfperr()
    
    # 声明获取当前浮点错误状态的 PyUFunc_getfperr 函数
    int PyUFunc_getfperr()
    
    # 定义接受一个 ufunc 对象、一个 PyUFuncGenericFunction 指针、一个整型指针和一个 PyUFuncGenericFunction 指针参数的 PyUFunc_ReplaceLoopBySignature 函数
    int PyUFunc_ReplaceLoopBySignature \
        (ufunc, PyUFuncGenericFunction, int *, PyUFuncGenericFunction *)
    
    # 定义接受一个 PyUFuncGenericFunction 指针数组、一个空指针数组、一个字符串、五个整型参数和三个字符参数的 PyUFunc_FromFuncAndDataAndSignature 函数
    object PyUFunc_FromFuncAndDataAndSignature \
             (PyUFuncGenericFunction *, void **, char *, int, int, int,
              int, char *, char *, int, char *)
    
    # 声明调用 _import_umath 函数并捕获异常，返回值为 -1
    int _import_umath() except -1
# 声明一个内联函数，用于设置数组的基对象
cdef inline void set_array_base(ndarray arr, object base):
    # 增加基对象的引用计数，确保在下面窃取引用之前执行此操作！
    Py_INCREF(base)
    # 设置数组的基对象为指定的 base 对象
    PyArray_SetBaseObject(arr, base)

# 声明一个内联函数，用于获取数组的基对象
cdef inline object get_array_base(ndarray arr):
    # 获取数组的基对象
    base = PyArray_BASE(arr)
    # 如果基对象为空，则返回 None
    if base is NULL:
        return None
    # 否则，返回基对象转换为 Python 对象的结果
    return <object>base

# 下面是更适合于 Cython 代码的 import_* 函数版本。

# 声明一个内联函数，用于导入 numpy 库
cdef inline int import_array() except -1:
    try:
        # 调用 Cython 专用的 __pyx_import_array() 函数来导入 numpy 库
        __pyx_import_array()
    except Exception:
        # 如果导入失败，则抛出 ImportError 异常
        raise ImportError("numpy._core.multiarray failed to import")

# 声明一个内联函数，用于导入 umath 模块
cdef inline int import_umath() except -1:
    try:
        # 调用 _import_umath() 函数来导入 umath 模块
        _import_umath()
    except Exception:
        # 如果导入失败，则抛出 ImportError 异常
        raise ImportError("numpy._core.umath failed to import")

# 声明一个内联函数，用于导入 ufunc 模块
cdef inline int import_ufunc() except -1:
    try:
        # 调用 _import_umath() 函数来导入 umath 模块
        _import_umath()
    except Exception:
        # 如果导入失败，则抛出 ImportError 异常
        raise ImportError("numpy._core.umath failed to import")

# 声明一个内联函数，用于判断对象是否为 timedelta64 类型
cdef inline bint is_timedelta64_object(object obj):
    """
    Cython 的等价函数，用于检查 obj 是否为 np.timedelta64 类型的实例

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return PyObject_TypeCheck(obj, &PyTimedeltaArrType_Type)

# 声明一个内联函数，用于判断对象是否为 datetime64 类型
cdef inline bint is_datetime64_object(object obj):
    """
    Cython 的等价函数，用于检查 obj 是否为 np.datetime64 类型的实例

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return PyObject_TypeCheck(obj, &PyDatetimeArrType_Type)

# 声明一个内联函数，用于获取 numpy datetime64 对象的 int64 值
cdef inline npy_datetime get_datetime64_value(object obj) nogil:
    """
    返回标量 numpy datetime64 对象下层的 int64 值

    注意，要解释为 datetime，还需要对应的单位，可以使用 get_datetime64_unit 获取单位信息。
    """
    return (<PyDatetimeScalarObject*>obj).obval

# 声明一个内联函数，用于获取 numpy timedelta64 对象的 int64 值
cdef inline npy_timedelta get_timedelta64_value(object obj) nogil:
    """
    返回标量 numpy timedelta64 对象下层的 int64 值
    """
    return (<PyTimedeltaScalarObject*>obj).obval

# 声明一个内联函数，用于获取 numpy datetime64 对象的单位信息
cdef inline NPY_DATETIMEUNIT get_datetime64_unit(object obj) nogil:
    """
    返回 numpy datetime64 对象的 dtype 中的单位部分信息
    """
    return <NPY_DATETIMEUNIT>(<PyDatetimeScalarObject*>obj).obmeta.base

# 在 numpy 1.6 中新增的迭代器 API
ctypedef int (*NpyIter_IterNextFunc)(NpyIter* it) noexcept nogil
ctypedef void (*NpyIter_GetMultiIndexFunc)(NpyIter* it, npy_intp* outcoords) noexcept nogil

# 从 "numpy/arrayobject.h" 头文件中导入相关定义
cdef extern from "numpy/arrayobject.h":

    # 定义 NpyIter 结构体
    ctypedef struct NpyIter:
        pass

    # 定义 NPY_FAIL 和 NPY_SUCCEED 常量
    cdef enum:
        NPY_FAIL
        NPY_SUCCEED
    # 定义枚举类型，用于迭代器追踪不同的标志位

    # 表示 C 顺序的索引
    NPY_ITER_C_INDEX
    # 表示 Fortran 顺序的索引
    NPY_ITER_F_INDEX
    # 表示多维索引
    NPY_ITER_MULTI_INDEX
    # 外部用户代码执行一维最内层循环
    NPY_ITER_EXTERNAL_LOOP
    # 将所有操作数转换为共同的数据类型
    NPY_ITER_COMMON_DTYPE
    # 操作数可能包含引用，在迭代期间需要 API 访问
    NPY_ITER_REFS_OK
    # 允许零大小的操作数，迭代时检查 IterSize 是否为 0
    NPY_ITER_ZEROSIZE_OK
    # 允许进行约简操作（大小为 0 的步幅，但维度大小 > 1）
    NPY_ITER_REDUCE_OK
    # 启用子范围迭代
    NPY_ITER_RANGED
    # 启用缓冲区
    NPY_ITER_BUFFERED
    # 当启用缓冲区时，尽可能增长内部循环
    NPY_ITER_GROWINNER
    # 延迟分配缓冲区，直到第一次 Reset* 调用
    NPY_ITER_DELAY_BUFALLOC
    # 当指定 NPY_KEEPORDER 时，禁用反转负步幅轴
    NPY_ITER_DONT_NEGATE_STRIDES
    NPY_ITER_COPY_IF_OVERLAP
    # 操作数将被读取和写入
    NPY_ITER_READWRITE
    # 操作数只能被读取
    NPY_ITER_READONLY
    # 操作数只能被写入
    NPY_ITER_WRITEONLY
    # 操作数的数据必须是本机字节顺序
    NPY_ITER_NBO
    # 操作数的数据必须是对齐的
    NPY_ITER_ALIGNED
    # 操作数的数据必须是连续的（在内部循环内）
    NPY_ITER_CONTIG
    # 可能复制操作数以满足要求
    NPY_ITER_COPY
    # 可能使用 WRITEBACKIFCOPY 复制操作数以满足要求
    NPY_ITER_UPDATEIFCOPY
    # 如果操作数为 NULL，则分配它
    NPY_ITER_ALLOCATE
    # 如果分配了操作数，则不使用任何子类型
    NPY_ITER_NO_SUBTYPE
    # 这是一个虚拟数组插槽，操作数为 NULL，但临时数据在那里
    NPY_ITER_VIRTUAL
    # 要求维度与迭代器维度完全匹配
    NPY_ITER_NO_BROADCAST
    # 此数组正在使用掩码，影响缓冲区到数组的复制
    NPY_ITER_WRITEMASKED
    # 此数组是所有 WRITEMASKED 操作数的掩码
    NPY_ITER_ARRAYMASK
    # 假定对 COPY_IF_OVERLAP 进行元素级访问的迭代器顺序数据访问
    NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE

    # 构造和销毁函数

    # 创建新的迭代器对象
    NpyIter* NpyIter_New(ndarray arr, npy_uint32 flags, NPY_ORDER order,
                         NPY_CASTING casting, dtype datatype) except NULL
    # 创建新的多重迭代器对象
    NpyIter* NpyIter_MultiNew(npy_intp nop, PyArrayObject** op, npy_uint32 flags,
                              NPY_ORDER order, NPY_CASTING casting, npy_uint32*
                              op_flags, PyArray_Descr** op_dtypes) except NULL
    // 创建一个新的 NpyIter 对象，具有高级配置选项
    NpyIter* NpyIter_AdvancedNew(npy_intp nop,           // 操作数的数量
                                 PyArrayObject** op,     // 操作数对象的指针数组
                                 npy_uint32 flags,       // 迭代器的标志
                                 NPY_ORDER order,        // 迭代的顺序
                                 NPY_CASTING casting,    // 类型转换方式
                                 npy_uint32* op_flags,   // 操作数的标志数组
                                 PyArray_Descr** op_dtypes, // 操作数的数据类型数组
                                 int oa_ndim,            // 操作数轴的数量
                                 int** op_axes,          // 操作数轴的数组的指针
                                 const npy_intp* itershape, // 迭代器的形状
                                 npy_intp buffersize)    // 缓冲区的大小
                                 except NULL
    
    // 复制一个现有的 NpyIter 对象，返回新的对象
    NpyIter* NpyIter_Copy(NpyIter* it) except NULL
    
    // 移除指定轴上的迭代
    int NpyIter_RemoveAxis(NpyIter* it, int axis) except NPY_FAIL
    
    // 移除多索引迭代
    int NpyIter_RemoveMultiIndex(NpyIter* it) except NPY_FAIL
    
    // 启用外部循环模式
    int NpyIter_EnableExternalLoop(NpyIter* it) except NPY_FAIL
    
    // 释放 NpyIter 对象的内存
    int NpyIter_Deallocate(NpyIter* it) except NPY_FAIL
    
    // 重置迭代器状态
    int NpyIter_Reset(NpyIter* it, char** errmsg) except NPY_FAIL
    
    // 重置到指定迭代索引范围
    int NpyIter_ResetToIterIndexRange(NpyIter* it, npy_intp istart, npy_intp iend, char** errmsg) except NPY_FAIL
    
    // 重置基础指针数组
    int NpyIter_ResetBasePointers(NpyIter* it, char** baseptrs, char** errmsg) except NPY_FAIL
    
    // 跳转到指定多索引位置
    int NpyIter_GotoMultiIndex(NpyIter* it, const npy_intp* multi_index) except NPY_FAIL
    
    // 跳转到指定索引位置
    int NpyIter_GotoIndex(NpyIter* it, npy_intp index) except NPY_FAIL
    
    // 获取迭代器的总大小
    npy_intp NpyIter_GetIterSize(NpyIter* it) nogil
    
    // 获取当前迭代的索引
    npy_intp NpyIter_GetIterIndex(NpyIter* it) nogil
    
    // 获取迭代索引范围
    void NpyIter_GetIterIndexRange(NpyIter* it, npy_intp* istart, npy_intp* iend) nogil
    
    // 跳转到指定迭代索引
    int NpyIter_GotoIterIndex(NpyIter* it, npy_intp iterindex) except NPY_FAIL
    
    // 判断是否需要延迟缓冲区分配
    npy_bool NpyIter_HasDelayedBufAlloc(NpyIter* it) nogil
    
    // 判断是否启用外部循环
    npy_bool NpyIter_HasExternalLoop(NpyIter* it) nogil
    
    // 判断是否具有多索引
    npy_bool NpyIter_HasMultiIndex(NpyIter* it) nogil
    
    // 判断是否具有索引
    npy_bool NpyIter_HasIndex(NpyIter* it) nogil
    
    // 判断是否需要缓冲区
    npy_bool NpyIter_RequiresBuffering(NpyIter* it) nogil
    
    // 判断是否已缓冲
    npy_bool NpyIter_IsBuffered(NpyIter* it) nogil
    
    // 判断是否为增长内部迭代
    npy_bool NpyIter_IsGrowInner(NpyIter* it) nogil
    
    // 获取缓冲区大小
    npy_intp NpyIter_GetBufferSize(NpyIter* it) nogil
    
    // 获取迭代器的维度数
    int NpyIter_GetNDim(NpyIter* it) nogil
    
    // 获取操作数的数量
    int NpyIter_GetNOp(NpyIter* it) nogil
    
    // 获取指定轴的步幅数组
    npy_intp* NpyIter_GetAxisStrideArray(NpyIter* it, int axis) except NULL
    
    // 获取迭代器的形状
    int NpyIter_GetShape(NpyIter* it, npy_intp* outshape) nogil
    
    // 获取操作数的数据类型数组
    PyArray_Descr** NpyIter_GetDescrArray(NpyIter* it)
    
    // 获取操作数对象的指针数组
    PyArrayObject** NpyIter_GetOperandArray(NpyIter* it)
    
    // 获取指定迭代索引的迭代视图
    ndarray NpyIter_GetIterView(NpyIter* it, npy_intp i)
    
    // 获取读取标志
    void NpyIter_GetReadFlags(NpyIter* it, char* outreadflags)
    
    // 获取写入标志
    void NpyIter_GetWriteFlags(NpyIter* it, char* outwriteflags)
    
    // 创建兼容步幅数组
    int NpyIter_CreateCompatibleStrides(NpyIter* it, npy_intp itemsize, npy_intp* outstrides) except NPY_FAIL
    
    // 判断是否是第一次访问指定操作数
    npy_bool NpyIter_IsFirstVisit(NpyIter* it, int iop) nogil
    
    // 获取迭代器的下一个迭代函数
    NpyIter_IterNextFunc* NpyIter_GetIterNext(NpyIter* it, char** errmsg) except NULL
    
    // 获取获取多索引的函数
    NpyIter_GetMultiIndexFunc* NpyIter_GetGetMultiIndex(NpyIter* it, char** errmsg) except NULL
    # 获取当前迭代器 `it` 的数据指针数组，在不使用全局解释器锁（GIL）的情况下
    char** NpyIter_GetDataPtrArray(NpyIter* it) nogil
    
    # 获取迭代器 `it` 的初始数据指针数组，在不使用全局解释器锁（GIL）的情况下
    char** NpyIter_GetInitialDataPtrArray(NpyIter* it) nogil
    
    # 获取迭代器 `it` 的索引指针数组
    npy_intp* NpyIter_GetIndexPtr(NpyIter* it)
    
    # 获取迭代器 `it` 内部循环的步幅数组，在不使用全局解释器锁（GIL）的情况下
    npy_intp* NpyIter_GetInnerStrideArray(NpyIter* it) nogil
    
    # 获取迭代器 `it` 内部循环的大小指针，在不使用全局解释器锁（GIL）的情况下
    npy_intp* NpyIter_GetInnerLoopSizePtr(NpyIter* it) nogil
    
    # 将迭代器 `it` 内部固定步幅的数组复制到 `outstrides` 数组中，在不使用全局解释器锁（GIL）的情况下
    void NpyIter_GetInnerFixedStrideArray(NpyIter* it, npy_intp* outstrides) nogil
    
    # 检查迭代器 `it` 的当前迭代是否需要使用 Python/C API，返回布尔值，在不使用全局解释器锁（GIL）的情况下
    npy_bool NpyIter_IterationNeedsAPI(NpyIter* it) nogil
    
    # 调试函数：打印迭代器 `it` 的调试信息
    void NpyIter_DebugPrint(NpyIter* it)
```