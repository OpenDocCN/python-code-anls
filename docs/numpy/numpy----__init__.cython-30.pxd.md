# `D:\src\scipysrc\numpy\numpy\__init__.cython-30.pxd`

```py
# NumPy static imports for Cython >= 3.0
#
# If any of the PyArray_* functions are called, import_array must be
# called first.  This is done automatically by Cython 3.0+ if a call
# is not detected inside of the module.
#
# Author: Dag Sverre Seljebotn
#

# 从cpython.ref中导入Py_INCREF函数
from cpython.ref cimport Py_INCREF
# 从cpython.object中导入PyObject, PyTypeObject, PyObject_TypeCheck对象
from cpython.object cimport PyObject, PyTypeObject, PyObject_TypeCheck
# 使用libc.stdio标准库中的所有函数
cimport libc.stdio as stdio


# 从外部导入声明，这里是为了说明NumPy API声明来自于NumPy自身而不是Cython
cdef extern from *:
    """
    /* Using NumPy API declarations from "numpy/__init__.cython-30.pxd" */
    """


# 从"numpy/arrayobject.h"中导入声明，定义了多种NumPy中的数据类型
cdef extern from "numpy/arrayobject.h":
    # 定义整型指针和无符号整型指针类型
    ctypedef signed long npy_intp
    ctypedef unsigned long npy_uintp

    # 定义布尔类型
    ctypedef unsigned char npy_bool

    # 定义有符号和无符号各种整型数据类型，包括char、short、int、long、long long等
    ctypedef signed char npy_byte
    ctypedef signed short npy_short
    ctypedef signed int npy_int
    ctypedef signed long npy_long
    ctypedef signed long long npy_longlong

    ctypedef unsigned char npy_ubyte
    ctypedef unsigned short npy_ushort
    ctypedef unsigned int npy_uint
    ctypedef unsigned long npy_ulong
    ctypedef unsigned long long npy_ulonglong

    # 定义浮点数数据类型，包括float、double、long double
    ctypedef float npy_float
    ctypedef double npy_double
    ctypedef long double npy_longdouble

    # 定义各种精度的整型数据类型，如int8、int16、int32、int64等
    ctypedef signed char npy_int8
    ctypedef signed short npy_int16
    ctypedef signed int npy_int32
    ctypedef signed long long npy_int64
    ctypedef signed long long npy_int96
    ctypedef signed long long npy_int128

    ctypedef unsigned char npy_uint8
    ctypedef unsigned short npy_uint16
    ctypedef unsigned int npy_uint32
    ctypedef unsigned long long npy_uint64
    ctypedef unsigned long long npy_uint96
    ctypedef unsigned long long npy_uint128

    # 定义各种精度的浮点数数据类型，如float32、float64等
    ctypedef float npy_float32
    ctypedef double npy_float64
    ctypedef long double npy_float80
    ctypedef long double npy_float96
    ctypedef long double npy_float128

    # 定义复数结构体类型
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

    # 定义PyArray_Dims结构体，包含指向整型指针的指针和长度
    ctypedef struct PyArray_Dims:
        npy_intp *ptr
        int len
    # 定义 NumPy 类型的枚举
    cdef enum NPY_TYPES:
        NPY_BOOL            # 布尔类型
        NPY_BYTE            # 有符号字节
        NPY_UBYTE           # 无符号字节
        NPY_SHORT           # 有符号短整型
        NPY_USHORT          # 无符号短整型
        NPY_INT             # 有符号整型
        NPY_UINT            # 无符号整型
        NPY_LONG            # 有符号长整型
        NPY_ULONG           # 无符号长整型
        NPY_LONGLONG        # 长长整型（有符号）
        NPY_ULONGLONG       # 长长整型（无符号）
        NPY_FLOAT           # 单精度浮点数
        NPY_DOUBLE          # 双精度浮点数
        NPY_LONGDOUBLE      # 长双精度浮点数
        NPY_CFLOAT          # 单精度复数
        NPY_CDOUBLE         # 双精度复数
        NPY_CLONGDOUBLE     # 长双精度复数
        NPY_OBJECT          # Python 对象
        NPY_STRING          # 字符串
        NPY_UNICODE         # Unicode 字符串
        NPY_VOID            # 空类型
        NPY_DATETIME        # 时间日期类型
        NPY_TIMEDELTA       # 时间间隔类型
        NPY_NTYPES_LEGACY   # 遗留的类型数量
        NPY_NOTYPE          # 未指定类型
    
        NPY_INT8            # 8 位整型
        NPY_INT16           # 16 位整型
        NPY_INT32           # 32 位整型
        NPY_INT64           # 64 位整型
        NPY_INT128          # 128 位整型
        NPY_INT256          # 256 位整型
        NPY_UINT8           # 8 位无符号整型
        NPY_UINT16          # 16 位无符号整型
        NPY_UINT32          # 32 位无符号整型
        NPY_UINT64          # 64 位无符号整型
        NPY_UINT128         # 128 位无符号整型
        NPY_UINT256         # 256 位无符号整型
        NPY_FLOAT16         # 16 位浮点数
        NPY_FLOAT32         # 32 位浮点数
        NPY_FLOAT64         # 64 位浮点数
        NPY_FLOAT80         # 80 位浮点数
        NPY_FLOAT96         # 96 位浮点数
        NPY_FLOAT128        # 128 位浮点数
        NPY_FLOAT256        # 256 位浮点数
        NPY_COMPLEX32       # 32 位复数
        NPY_COMPLEX64       # 64 位复数
        NPY_COMPLEX128      # 128 位复数
        NPY_COMPLEX160      # 160 位复数
        NPY_COMPLEX192      # 192 位复数
        NPY_COMPLEX256      # 256 位复数
        NPY_COMPLEX512      # 512 位复数
    
        NPY_INTP            # 平台相关的整数类型
        NPY_DEFAULT_INT     # 非编译时常数（通常不是）
    
    # 定义 NumPy 数组的顺序枚举
    ctypedef enum NPY_ORDER:
        NPY_ANYORDER        # 任意顺序
        NPY_CORDER          # C 顺序
        NPY_FORTRANORDER    # Fortran 顺序
        NPY_KEEPORDER       # 保持原顺序
    
    # 定义 NumPy 类型转换枚举
    ctypedef enum NPY_CASTING:
        NPY_NO_CASTING      # 无转换
        NPY_EQUIV_CASTING   # 等效转换
        NPY_SAFE_CASTING    # 安全转换
        NPY_SAME_KIND_CASTING  # 相同种类的转换
        NPY_UNSAFE_CASTING  # 不安全的转换
    
    # 定义 NumPy 数组边界处理方式枚举
    ctypedef enum NPY_CLIPMODE:
        NPY_CLIP            # 截断处理
        NPY_WRAP            # 环绕处理
        NPY_RAISE           # 报错处理
    
    # 定义标量类型枚举
    ctypedef enum NPY_SCALARKIND:
        NPY_NOSCALAR        # 非标量
        NPY_BOOL_SCALAR     # 布尔标量
        NPY_INTPOS_SCALAR   # 正整数标量
        NPY_INTNEG_SCALAR   # 负整数标量
        NPY_FLOAT_SCALAR    # 浮点数标量
        NPY_COMPLEX_SCALAR  # 复数标量
        NPY_OBJECT_SCALAR   # Python 对象标量
    
    # 定义排序算法枚举
    ctypedef enum NPY_SORTKIND:
        NPY_QUICKSORT       # 快速排序
        NPY_HEAPSORT        # 堆排序
        NPY_MERGESORT       # 归并排序
    
    # 定义搜索边界枚举
    ctypedef enum NPY_SEARCHSIDE:
        NPY_SEARCHLEFT      # 左侧搜索
        NPY_SEARCHRIGHT     # 右侧搜索
    
    # 遗留的 NumPy 标记，自 NumPy 1.7 起废弃！不要在新代码中使用！
    enum:
        NPY_C_CONTIGUOUS    # C 连续数组
        NPY_F_CONTIGUOUS    # Fortran 连续数组
        NPY_CONTIGUOUS      # 连续数组
        NPY_FORTRAN         # Fortran 数组
        NPY_OWNDATA         # 拥有数据
        NPY_FORCECAST       # 强制类型转换
        NPY_ENSURECOPY      # 确保拷贝
        NPY_ENSUREARRAY     # 确保数组
        NPY_ELEMENTSTRIDES  # 元素步长
        NPY_ALIGNED         # 对齐的
        NPY_NOTSWAPPED      # 未交换的
        NPY_WRITEABLE       # 可写的
        NPY_ARR_HAS_DESCR   # 数组有描述
    
        NPY_BEHAVED         # 表现正常
        NPY_BEHAVED_NS      # 表现正常（无标量）
        NPY_CARRAY          # C 数组
        NPY_CARRAY_RO       # 只读 C 数组
        NPY_FARRAY          # Fortran 数组
        NPY_FARRAY_RO       # 只读 Fortran 数组
        NPY_DEFAULT         # 默认
    
        NPY_IN_ARRAY        # 输入数组
        NPY_OUT_ARRAY       # 输出数组
        NPY_INOUT_ARRAY     # 输入输出数组
        NPY_IN_FARRAY       # 输入 Fortran 数组
        NPY_OUT_FARRAY      # 输出 Fortran 数组
        NPY_INOUT_FARRAY    # 输入输出 Fortran 数组
    
        NPY_UPDATE_ALL      # 更新全部
    # 定义 NumPy 数组的属性和行为标志，用于描述数组的内存布局和操作方式

    enum:
        # 指示数组以 C 风格连续存储（行优先）的标志
        NPY_ARRAY_C_CONTIGUOUS
        # 指示数组以 Fortran 风格连续存储（列优先）的标志
        NPY_ARRAY_F_CONTIGUOUS
        # 指示数组拥有自己的数据副本的标志
        NPY_ARRAY_OWNDATA
        # 强制数据类型转换的标志
        NPY_ARRAY_FORCECAST
        # 确保返回数组的副本而不是视图的标志
        NPY_ARRAY_ENSURECOPY
        # 确保返回一个 ndarray 而不是一个子类的标志
        NPY_ARRAY_ENSUREARRAY
        # 允许传递元素步长参数的标志
        NPY_ARRAY_ELEMENTSTRIDES
        # 数组数据是否按照特定对齐要求对齐的标志
        NPY_ARRAY_ALIGNED
        # 数组数据未被交换的标志
        NPY_ARRAY_NOTSWAPPED
        # 数组可写的标志
        NPY_ARRAY_WRITEABLE
        # 如果数组通过复制后可以写回原始数组，则写回的标志
        NPY_ARRAY_WRITEBACKIFCOPY

        # 默认行为的数组标志
        NPY_ARRAY_BEHAVED
        # 不带隐式复制的默认行为数组标志
        NPY_ARRAY_BEHAVED_NS
        # C 风格连续数组的标志
        NPY_ARRAY_CARRAY
        # 只读的 C 风格连续数组的标志
        NPY_ARRAY_CARRAY_RO
        # Fortran 风格连续数组的标志
        NPY_ARRAY_FARRAY
        # 只读的 Fortran 风格连续数组的标志
        NPY_ARRAY_FARRAY_RO
        # 默认数组标志
        NPY_ARRAY_DEFAULT

        # 输入数组的标志
        NPY_ARRAY_IN_ARRAY
        # 输出数组的标志
        NPY_ARRAY_OUT_ARRAY
        # 输入输出数组的标志
        NPY_ARRAY_INOUT_ARRAY
        # 输入 Fortran 风格连续数组的标志
        NPY_ARRAY_IN_FARRAY
        # 输出 Fortran 风格连续数组的标志
        NPY_ARRAY_OUT_FARRAY
        # 输入输出 Fortran 风格连续数组的标志
        NPY_ARRAY_INOUT_FARRAY

        # 更新所有数组的标志
        NPY_ARRAY_UPDATE_ALL

    cdef enum:
        # NumPy 数组可以拥有的最大维度数，在 NumPy 2.x 为 64，在 NumPy 1.x 为 32
        NPY_MAXDIMS
        # 用于拉平操作（如求平均值）的轴标志
        NPY_RAVEL_AXIS

    ctypedef void (*PyArray_VectorUnaryFunc)(void *, void *, npy_intp, void *, void *)

    ctypedef struct PyArray_ArrayDescr:
        # PyArray_ArrayDescr 结构体，包含一个形状元组，但 Cython 不支持在非 PyObject 声明内使用 "tuple shape"，所以这里声明为 PyObject*
        PyObject* shape

    ctypedef struct PyArray_Descr:
        # PyArray_Descr 结构体，目前只是一个占位符
        pass
    # 定义一个 Cython 类型描述符类 numpy.dtype，这个类对应于 NumPy 中的数据类型描述符。
    ctypedef class numpy.dtype [object PyArray_Descr, check_size ignore]:
        # typeobj 是指向 PyTypeObject 结构体的指针，用于描述数据类型的 Python 类型对象
        cdef PyTypeObject* typeobj
        # kind 表示数据类型的种类，如 'i' 表示整数类型
        cdef char kind
        # type 表示数据类型的具体类型，例如 'i' 表示整数，'f' 表示浮点数
        cdef char type
        # byteorder 表示数据的字节顺序，常见值有 '|', '<', '>', '=' 等
        # 注意：Numpy 有时会在共享的数据类型对象上突然改变此字段的值，在小端机器上可能会由 '|' 改为 '<'。如果这对你很重要，使用 PyArray_IsNativeByteOrder(dtype.byteorder) 来检查而不是直接访问此字段。
        cdef char byteorder
        # type_num 是一个整数，用于标识数据类型的编号
        cdef int type_num
    
        # 定义 itemsize 属性，返回数据类型的元素大小（以字节为单位）
        @property
        cdef inline npy_intp itemsize(self) noexcept nogil:
            return PyDataType_ELSIZE(self)
    
        # 定义 alignment 属性，返回数据类型的对齐方式（以字节为单位）
        @property
        cdef inline npy_intp alignment(self) noexcept nogil:
            return PyDataType_ALIGNMENT(self)
    
        # fields 属性返回一个对象，表示数据类型的字段。注意，它可能为 NULL，使用前需检查 PyDataType_HASFIELDS。
        @property
        cdef inline object fields(self):
            return <object>PyDataType_FIELDS(self)
    
        # names 属性返回一个元组，表示数据类型的字段名。注意，它可能为 NULL，使用前需检查 PyDataType_HASFIELDS。
        @property
        cdef inline tuple names(self):
            return <tuple>PyDataType_NAMES(self)
    
        # subarray 属性返回一个 PyArray_ArrayDescr 指针，表示数据类型的子数组描述符。
        # 使用 PyDataType_HASSUBARRAY 来检查此字段是否有效，指针可能为 NULL。
        @property
        cdef inline PyArray_ArrayDescr* subarray(self) noexcept nogil:
            return PyDataType_SUBARRAY(self)
    
        # flags 属性返回一个 npy_uint64 类型的整数，表示数据类型的标志位。
        # 这个属性文档字符串指示它返回数据类型的标志位。
        @property
        cdef inline npy_uint64 flags(self) noexcept nogil:
            """The data types flags."""
            return PyDataType_FLAGS(self)
    
    
    # 定义一个 Cython 类型描述符类 numpy.flatiter，这个类用于处理扁平化的迭代器。
    ctypedef class numpy.flatiter [object PyArrayIterObject, check_size ignore]:
        # 通过宏使用这个类
        pass
    # 定义了一个 Cython 类 `numpy.broadcast`，这个类的内部实现使用了 C 语言风格的类型定义。
    ctypedef class numpy.broadcast [object PyArrayMultiIterObject, check_size ignore]:
    
        @property
        cdef inline int numiter(self) noexcept nogil:
            """返回需要广播到相同形状的数组的数量。"""
            return PyArray_MultiIter_NUMITER(self)
    
        @property
        cdef inline npy_intp size(self) noexcept nogil:
            """返回广播后的总大小。"""
            return PyArray_MultiIter_SIZE(self)
    
        @property
        cdef inline npy_intp index(self) noexcept nogil:
            """返回当前广播结果的一维索引。"""
            return PyArray_MultiIter_INDEX(self)
    
        @property
        cdef inline int nd(self) noexcept nogil:
            """返回广播结果的维数。"""
            return PyArray_MultiIter_NDIM(self)
    
        @property
        cdef inline npy_intp* dimensions(self) noexcept nogil:
            """返回广播结果的形状。"""
            return PyArray_MultiIter_DIMS(self)
    
        @property
        cdef inline void** iters(self) noexcept nogil:
            """返回一个包含迭代器对象的数组，这些迭代器用于一起广播的数组。
            返回后，这些迭代器将被调整以进行广播。"""
            return PyArray_MultiIter_ITERS(self)
    
    
    ctypedef struct PyArrayObject:
        # 用于在无法使用 ndarray 替代 PyArrayObject* 的情况下，如 PyArrayObject**。
        pass
    ctypedef class numpy.ndarray [object PyArrayObject, check_size ignore]:
        cdef __cythonbufferdefaults__ = {"mode": "strided"}

        # NOTE: no field declarations since direct access is deprecated since NumPy 1.7
        # Instead, we use properties that map to the corresponding C-API functions.

        @property
        cdef inline PyObject* base(self) noexcept nogil:
            """Returns a borrowed reference to the object owning the data/memory.
            """
            return PyArray_BASE(self)

        @property
        cdef inline dtype descr(self):
            """Returns an owned reference to the dtype of the array.
            """
            return <dtype>PyArray_DESCR(self)

        @property
        cdef inline int ndim(self) noexcept nogil:
            """Returns the number of dimensions in the array.
            """
            return PyArray_NDIM(self)

        @property
        cdef inline npy_intp *shape(self) noexcept nogil:
            """Returns a pointer to the dimensions/shape of the array.
            The number of elements matches the number of dimensions of the array (ndim).
            Can return NULL for 0-dimensional arrays.
            """
            return PyArray_DIMS(self)

        @property
        cdef inline npy_intp *strides(self) noexcept nogil:
            """Returns a pointer to the strides of the array.
            The number of elements matches the number of dimensions of the array (ndim).
            """
            return PyArray_STRIDES(self)

        @property
        cdef inline npy_intp size(self) noexcept nogil:
            """Returns the total size (in number of elements) of the array.
            """
            return PyArray_SIZE(self)

        @property
        cdef inline char* data(self) noexcept nogil:
            """The pointer to the data buffer as a char*.
            This is provided for legacy reasons to avoid direct struct field access.
            For new code that needs this access, you probably want to cast the result
            of `PyArray_DATA()` instead, which returns a 'void*'.
            """
            return PyArray_BYTES(self)


    int _import_array() except -1
    # A second definition so _import_array isn't marked as used when we use it here.
    # Do not use - subject to change any time.
    int __pyx_import_array "_import_array"() except -1

    #
    # Macros from ndarrayobject.h
    #
    bint PyArray_CHKFLAGS(ndarray m, int flags) nogil
    bint PyArray_IS_C_CONTIGUOUS(ndarray arr) nogil
    bint PyArray_IS_F_CONTIGUOUS(ndarray arr) nogil
    bint PyArray_ISCONTIGUOUS(ndarray m) nogil
    bint PyArray_ISWRITEABLE(ndarray m) nogil
    bint PyArray_ISALIGNED(ndarray m) nogil

    int PyArray_NDIM(ndarray) nogil
    bint PyArray_ISONESEGMENT(ndarray) nogil
    bint PyArray_ISFORTRAN(ndarray) nogil
    int PyArray_FORTRANIF(ndarray) nogil

    void* PyArray_DATA(ndarray) nogil
    char* PyArray_BYTES(ndarray) nogil


注释：

# 定义一个 Cython 类，模拟 NumPy 的 ndarray 对象
ctypedef class numpy.ndarray [object PyArrayObject, check_size ignore]:
    # 定义 Cython 类的属性，默认使用 strided 模式
    cdef __cythonbufferdefaults__ = {"mode": "strided"}

    # 注意：不声明字段，因为自 NumPy 1.7 起直接访问已被弃用
    # 使用属性来映射对应的 C-API 函数

    @property
    cdef inline PyObject* base(self) noexcept nogil:
        """返回一个借用引用，指向拥有数据/内存的对象。
        """
        return PyArray_BASE(self)

    @property
    cdef inline dtype descr(self):
        """返回数组的 dtype 的拥有引用。
        """
        return <dtype>PyArray_DESCR(self)

    @property
    cdef inline int ndim(self) noexcept nogil:
        """返回数组的维度数目。
        """
        return PyArray_NDIM(self)

    @property
    cdef inline npy_intp *shape(self) noexcept nogil:
        """返回指向数组维度/形状的指针。
        元素数目与数组的维度数目 (ndim) 相匹配。
        对于0维数组可能返回NULL。
        """
        return PyArray_DIMS(self)

    @property
    cdef inline npy_intp *strides(self) noexcept nogil:
        """返回指向数组步长的指针。
        元素数目与数组的维度数目 (ndim) 相匹配。
        """
        return PyArray_STRIDES(self)

    @property
    cdef inline npy_intp size(self) noexcept nogil:
        """返回数组的总大小（元素数目）。
        """
        return PyArray_SIZE(self)

    @property
    cdef inline char* data(self) noexcept nogil:
        """指向数据缓冲区的 char* 指针。
        出于遗留原因提供此方法，以避免直接访问结构字段。
        对于需要此访问的新代码，建议使用 `PyArray_DATA()` 的结果进行转换，它返回 'void*'。
        """
        return PyArray_BYTES(self)


int _import_array() except -1
# 第二个定义，以避免在此处使用时标记 _import_array 为已使用。
# 不要使用 - 随时可能更改。

int __pyx_import_array "_import_array"() except -1

#
# 从 ndarrayobject.h 中的宏定义
#
bint PyArray_CHKFLAGS(ndarray m, int flags) nogil
bint PyArray_IS_C_CONTIGUOUS(ndarray arr) nogil
bint PyArray_IS_F_CONTIGUOUS(ndarray arr) nogil
bint PyArray_ISCONTIGUOUS(ndarray m) nogil
bint PyArray_ISWRITEABLE(ndarray m) nogil
bint PyArray_ISALIGNED(ndarray m) nogil

int PyArray_NDIM(ndarray) nogil
bint PyArray_ISONESEGMENT(ndarray) nogil
bint PyArray_ISFORTRAN(ndarray) nogil
int PyArray_FORTRANIF(ndarray) nogil

void* PyArray_DATA(ndarray) nogil
char* PyArray_BYTES(ndarray) nogil
    npy_intp* PyArray_DIMS(ndarray) nogil
    // 返回指向数组维度的指针，不会引发GIL

    npy_intp* PyArray_STRIDES(ndarray) nogil
    // 返回指向数组步幅的指针，不会引发GIL

    npy_intp PyArray_DIM(ndarray, size_t) nogil
    // 返回指定索引的数组维度大小，不会引发GIL

    npy_intp PyArray_STRIDE(ndarray, size_t) nogil
    // 返回指定索引的数组步幅大小，不会引发GIL

    PyObject *PyArray_BASE(ndarray) nogil  // returns borrowed reference!
    // 返回数组的基础对象，借用引用，不会引发GIL

    PyArray_Descr *PyArray_DESCR(ndarray) nogil  // returns borrowed reference to dtype!
    // 返回数组的描述符，借用引用，不会引发GIL

    PyArray_Descr *PyArray_DTYPE(ndarray) nogil  // returns borrowed reference to dtype! NP 1.7+ alias for descr.
    // 返回数组的数据类型描述符，借用引用，不会引发GIL，NP 1.7+中是descr的别名

    int PyArray_FLAGS(ndarray) nogil
    // 返回数组的标志位，不会引发GIL

    void PyArray_CLEARFLAGS(ndarray, int flags) nogil  // Added in NumPy 1.7
    // 清除数组的指定标志位，不会引发GIL，NumPy 1.7中新增

    void PyArray_ENABLEFLAGS(ndarray, int flags) nogil  // Added in NumPy 1.7
    // 启用数组的指定标志位，不会引发GIL，NumPy 1.7中新增

    npy_intp PyArray_ITEMSIZE(ndarray) nogil
    // 返回数组元素的大小（字节数），不会引发GIL

    int PyArray_TYPE(ndarray arr) nogil
    // 返回数组的数据类型编号，不会引发GIL

    object PyArray_GETITEM(ndarray arr, void *itemptr)
    // 从数组中获取指定位置的元素，不会引发GIL

    int PyArray_SETITEM(ndarray arr, void *itemptr, object obj) except -1
    // 将对象设置到数组的指定位置，不会引发GIL，异常时返回-1

    bint PyTypeNum_ISBOOL(int) nogil
    // 检查指定的数据类型编号是否是布尔类型，不会引发GIL

    bint PyTypeNum_ISUNSIGNED(int) nogil
    // 检查指定的数据类型编号是否是无符号整数类型，不会引发GIL

    bint PyTypeNum_ISSIGNED(int) nogil
    // 检查指定的数据类型编号是否是有符号整数类型，不会引发GIL

    bint PyTypeNum_ISINTEGER(int) nogil
    // 检查指定的数据类型编号是否是整数类型，不会引发GIL

    bint PyTypeNum_ISFLOAT(int) nogil
    // 检查指定的数据类型编号是否是浮点数类型，不会引发GIL

    bint PyTypeNum_ISNUMBER(int) nogil
    // 检查指定的数据类型编号是否是数值类型，不会引发GIL

    bint PyTypeNum_ISSTRING(int) nogil
    // 检查指定的数据类型编号是否是字符串类型，不会引发GIL

    bint PyTypeNum_ISCOMPLEX(int) nogil
    // 检查指定的数据类型编号是否是复数类型，不会引发GIL

    bint PyTypeNum_ISFLEXIBLE(int) nogil
    // 检查指定的数据类型编号是否是灵活类型，不会引发GIL

    bint PyTypeNum_ISUSERDEF(int) nogil
    // 检查指定的数据类型编号是否是用户定义类型，不会引发GIL

    bint PyTypeNum_ISEXTENDED(int) nogil
    // 检查指定的数据类型编号是否是扩展类型，不会引发GIL

    bint PyTypeNum_ISOBJECT(int) nogil
    // 检查指定的数据类型编号是否是对象类型，不会引发GIL

    npy_intp PyDataType_ELSIZE(dtype) nogil
    // 返回数据类型描述符的元素大小（字节数），不会引发GIL

    npy_intp PyDataType_ALIGNMENT(dtype) nogil
    // 返回数据类型描述符的对齐方式，不会引发GIL

    PyObject* PyDataType_METADATA(dtype) nogil
    // 返回数据类型描述符的元数据对象，不会引发GIL

    PyArray_ArrayDescr* PyDataType_SUBARRAY(dtype) nogil
    // 返回数据类型描述符的子数组描述符，不会引发GIL

    PyObject* PyDataType_NAMES(dtype) nogil
    // 返回数据类型描述符的字段名称，不会引发GIL

    PyObject* PyDataType_FIELDS(dtype) nogil
    // 返回数据类型描述符的字段描述符，不会引发GIL

    bint PyDataType_ISBOOL(dtype) nogil
    // 检查指定的数据类型描述符是否是布尔类型，不会引发GIL

    bint PyDataType_ISUNSIGNED(dtype) nogil
    // 检查指定的数据类型描述符是否是无符号整数类型，不会引发GIL

    bint PyDataType_ISSIGNED(dtype) nogil
    // 检查指定的数据类型描述符是否是有符号整数类型，不会引发GIL

    bint PyDataType_ISINTEGER(dtype) nogil
    // 检查指定的数据类型描述符是否是整数类型，不会引发GIL

    bint PyDataType_ISFLOAT(dtype) nogil
    // 检查指定的数据类型描述符是否是浮点数类型，不会引发GIL

    bint PyDataType_ISNUMBER(dtype) nogil
    // 检查指定的数据类型描述符是否是数值类型，不会引发GIL

    bint PyDataType_ISSTRING(dtype) nogil
    // 检查指定的数据类型描述符是否是字符串类型，不会引发GIL

    bint PyDataType_ISCOMPLEX(dtype) nogil
    // 检查指定的数据类型描述符是否是复数类型，不会引发GIL

    bint PyDataType_ISFLEXIBLE(dtype) nogil
    // 检查指定的数据类型描述符是否是灵活类型，不会引发GIL

    bint PyDataType_ISUSERDEF(dtype) nogil
    // 检查指定的数据类型描述符是否是用户定义类型，不会引发GIL

    bint PyDataType_ISEXTENDED(dtype) nogil
    // 检查指定的数据类型描述符是否是扩展类型，不会引发GIL

    bint PyDataType_ISOBJECT(dtype) nogil
    // 检查指定的数据类型描述符是否是对象类型，不会引发GIL

    bint PyDataType_HASFIELDS(dtype) nogil
    // 检查指定的数据类型描述符是否有字段，不会引发GIL

    bint PyDataType_HASSUBARRAY(dtype) nogil
    // 检查指定的数据类型描述符是否有子数组，不会引发GIL

    npy_uint64 PyDataType_FLAGS(dtype) nogil
    // 返回数据类型描述符的标志位，不会引发GIL

    bint PyArray_ISBOOL(ndarray) nogil
    // 检查数组是否是布尔类型，不会引发GIL

    bint PyArray_ISUNSIGNED(ndarray) nogil
    // 检查数组是否是无符号整数类型，不会引发GIL

    bint PyArray_ISSIGNED(ndarray) nogil
    // 检查数组是否是有符号整数类型，不会引发GIL

    bint PyArray_ISINTEGER(ndarray) nogil
    // 检查数组是否是整数类型，不会引发GIL

    bint PyArray_ISFLOAT(ndarray) nogil
    // 检查数组是否是浮点数类型，不会引发GIL

    bint PyArray_ISNUMBER(ndarray) nogil
    // 检查数组是否是数值类型，不会引发GIL

    bint PyArray
    # 检查数组的字节顺序是否与本地字节顺序相同，适用于ndarray.byteorder
    bint PyArray_IsNativeByteOrder(char) nogil

    # 检查ndarray是否未交换字节顺序
    bint PyArray_ISNOTSWAPPED(ndarray) nogil

    # 检查ndarray是否已交换字节顺序
    bint PyArray_ISBYTESWAPPED(ndarray) nogil

    # 根据给定的整数值对ndarray进行字节交换
    bint PyArray_FLAGSWAP(ndarray, int) nogil

    # 检查ndarray是否是C连续数组
    bint PyArray_ISCARRAY(ndarray) nogil

    # 检查ndarray是否是只读的C连续数组
    bint PyArray_ISCARRAY_RO(ndarray) nogil

    # 检查ndarray是否是Fortran连续数组
    bint PyArray_ISFARRAY(ndarray) nogil

    # 检查ndarray是否是只读的Fortran连续数组
    bint PyArray_ISFARRAY_RO(ndarray) nogil

    # 检查ndarray是否具有行为符合规范
    bint PyArray_ISBEHAVED(ndarray) nogil

    # 检查ndarray是否具有只读的行为符合规范
    bint PyArray_ISBEHAVED_RO(ndarray) nogil

    # 检查dtype是否未交换字节顺序
    bint PyDataType_ISNOTSWAPPED(dtype) nogil

    # 检查dtype是否已交换字节顺序
    bint PyDataType_ISBYTESWAPPED(dtype) nogil

    # 检查对象是否是有效的PyArray_Descr对象
    bint PyArray_DescrCheck(object)

    # 检查对象是否是PyArray类型或其子类型
    bint PyArray_Check(object)

    # 检查对象是否是PyArray类型且精确匹配
    bint PyArray_CheckExact(object)

    # 检查对象是否是零维数组
    bint PyArray_IsZeroDim(object)

    # 检查对象是否是标量数组
    bint PyArray_CheckScalar(object)

    # 检查对象是否是Python数值类型
    bint PyArray_IsPythonNumber(object)

    # 检查对象是否是Python标量数组
    bint PyArray_IsPythonScalar(object)

    # 检查对象是否是任何标量数组
    bint PyArray_IsAnyScalar(object)

    # 检查对象是否是任何标量数组的类型
    bint PyArray_CheckAnyScalar(object)

    # 获取ndarray的连续副本
    ndarray PyArray_GETCONTIGUOUS(ndarray)

    # 检查两个ndarray是否具有相同的形状
    bint PyArray_SAMESHAPE(ndarray, ndarray) nogil

    # 返回ndarray的元素数
    npy_intp PyArray_SIZE(ndarray) nogil

    # 返回ndarray占用的字节数
    npy_intp PyArray_NBYTES(ndarray) nogil

    # 从任何对象创建一个PyArray对象
    object PyArray_FROM_O(object)

    # 从文件对象创建PyArray对象，支持指定标志
    object PyArray_FROM_OF(object m, int flags)

    # 从对象创建PyArray对象，支持指定类型
    object PyArray_FROM_OT(object m, int type)

    # 从对象创建PyArray对象，支持指定类型和标志
    object PyArray_FROM_OTF(object m, int type, int flags)

    # 从任何对象创建PyArray对象，支持指定类型和范围
    object PyArray_FROMANY(object m, int type, int min, int max, int flags)

    # 创建一个指定形状、类型和存储顺序的全零数组
    object PyArray_ZEROS(int nd, npy_intp* dims, int type, int fortran)

    # 创建一个指定形状、类型和存储顺序的空数组
    object PyArray_EMPTY(int nd, npy_intp* dims, int type, int fortran)

    # 用指定的字节填充ndarray
    void PyArray_FILLWBYTE(ndarray, int val)

    # 从任何对象创建一个连续的PyArray对象
    object PyArray_ContiguousFromAny(op, int, int min_depth, int max_depth)

    # 检查两个ndarray的等效数组类型
    unsigned char PyArray_EquivArrTypes(ndarray a1, ndarray a2)

    # 检查两个字节顺序是否等效
    bint PyArray_EquivByteorders(int b1, int b2) nogil

    # 创建一个简单的PyArray对象，指定形状和类型
    object PyArray_SimpleNew(int nd, npy_intp* dims, int typenum)

    # 从给定数据创建一个简单的PyArray对象，指定形状、类型和数据
    object PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data)

    # 将ndarray转换为Python标量
    object PyArray_ToScalar(void* data, ndarray arr)

    # 获取ndarray的指针，支持1维索引
    void* PyArray_GETPTR1(ndarray m, npy_intp i) nogil

    # 获取ndarray的指针，支持2维索引
    void* PyArray_GETPTR2(ndarray m, npy_intp i, npy_intp j) nogil

    # 获取ndarray的指针，支持3维索引
    void* PyArray_GETPTR3(ndarray m, npy_intp i, npy_intp j, npy_intp k) nogil

    # 获取ndarray的指针，支持4维索引
    void* PyArray_GETPTR4(ndarray m, npy_intp i, npy_intp j, npy_intp k, npy_intp l) nogil

    # 创建ndarray的副本
    object PyArray_Copy(ndarray)

    # 从对象创建PyArray对象，支持指定类型和深度范围
    object PyArray_FromObject(object op, int type, int min_depth, int max_depth)

    # 从对象创建连续的PyArray对象，支持指定类型和深度范围
    object PyArray_ContiguousFromObject(object op, int type, int min_depth, int max_depth)

    # 从对象创建PyArray对象的副本，支持指定类型和深度范围
    object PyArray_CopyFromObject(object op, int type, int min_depth, int max_depth)

    # 将ndarray转换为指定类型的PyArray对象
    object PyArray_Cast(ndarray mp, int type_num)
    # 定义 PyArray_Take 函数，用于从数组中按照给定轴取出指定的元素
    object PyArray_Take(ndarray ap, object items, int axis)
    
    # 定义 PyArray_Put 函数，用于将指定的值放置到数组中的指定位置
    object PyArray_Put(ndarray ap, object items, object values)
    
    # 重置 flatiter 对象，使其指向迭代器的起始位置
    void PyArray_ITER_RESET(flatiter it) nogil
    
    # 将 flatiter 对象向前移动到下一个元素位置
    void PyArray_ITER_NEXT(flatiter it) nogil
    
    # 将 flatiter 对象移动到指定的多维索引位置
    void PyArray_ITER_GOTO(flatiter it, npy_intp* destination) nogil
    
    # 将 flatiter 对象移动到一维索引位置
    void PyArray_ITER_GOTO1D(flatiter it, npy_intp ind) nogil
    
    # 返回 flatiter 对象当前位置的数据指针
    void* PyArray_ITER_DATA(flatiter it) nogil
    
    # 检查 flatiter 对象是否迭代完毕，返回 bint 类型
    bint PyArray_ITER_NOTDONE(flatiter it) nogil
    
    # 重置 broadcast multi 对象，使其指向多迭代器的起始位置
    void PyArray_MultiIter_RESET(broadcast multi) nogil
    
    # 将 broadcast multi 对象向前移动到下一个元素位置
    void PyArray_MultiIter_NEXT(broadcast multi) nogil
    
    # 将 broadcast multi 对象移动到指定的一维索引位置
    void PyArray_MultiIter_GOTO(broadcast multi, npy_intp dest) nogil
    
    # 将 broadcast multi 对象移动到一维索引位置
    void PyArray_MultiIter_GOTO1D(broadcast multi, npy_intp ind) nogil
    
    # 返回 broadcast multi 对象当前索引 i 处的数据指针
    void* PyArray_MultiIter_DATA(broadcast multi, npy_intp i) nogil
    
    # 将 broadcast multi 对象的第 i 个迭代器向前移动到下一个元素位置
    void PyArray_MultiIter_NEXTi(broadcast multi, npy_intp i) nogil
    
    # 检查 broadcast multi 对象是否所有迭代器都未迭代完，返回 bint 类型
    bint PyArray_MultiIter_NOTDONE(broadcast multi) nogil
    
    # 返回 broadcast multi 对象的总大小
    npy_intp PyArray_MultiIter_SIZE(broadcast multi) nogil
    
    # 返回 broadcast multi 对象的维度数
    int PyArray_MultiIter_NDIM(broadcast multi) nogil
    
    # 返回 broadcast multi 对象当前索引
    npy_intp PyArray_MultiIter_INDEX(broadcast multi) nogil
    
    # 返回 broadcast multi 对象包含的迭代器数量
    int PyArray_MultiIter_NUMITER(broadcast multi) nogil
    
    # 返回 broadcast multi 对象的维度数组
    npy_intp* PyArray_MultiIter_DIMS(broadcast multi) nogil
    
    # 返回 broadcast multi 对象的迭代器数组
    void** PyArray_MultiIter_ITERS(broadcast multi) nogil
    
    # 递增 ndarray 对象的引用计数
    int PyArray_INCREF (ndarray) except *  # uses PyArray_Item_INCREF...
    
    # 递减 ndarray 对象的引用计数
    int PyArray_XDECREF (ndarray) except *  # uses PyArray_Item_DECREF...
    
    # 根据给定的整数类型创建并返回 dtype 对象
    dtype PyArray_DescrFromType (int)
    
    # 根据给定的整数类型创建并返回对应的 PyArray_TypeObject 对象
    object PyArray_TypeObjectFromType (int)
    
    # 创建并返回指定大小的以零填充的数组
    char * PyArray_Zero (ndarray)
    
    # 创建并返回指定大小的以一填充的数组
    char * PyArray_One (ndarray)
    
    # 检查从一个整数类型到另一个整数类型是否可以安全转换，写入错误信息
    int PyArray_CanCastSafely (int, int)  # writes errors
    
    # 检查从一个 dtype 对象到另一个 dtype 对象是否可以安全转换，写入错误信息
    npy_bool PyArray_CanCastTo (dtype, dtype)  # writes errors
    
    # 返回 object 对象的类型，如果不符合则返回 0
    int PyArray_ObjectType (object, int) except 0
    
    # 根据给定的 object 对象和 dtype 对象返回描述符 dtype
    dtype PyArray_DescrFromObject (object, dtype)
    
    # 返回标量对象的描述符 dtype
    dtype PyArray_DescrFromScalar (object)
    
    # 根据给定的对象返回其对应的描述符 dtype
    dtype PyArray_DescrFromTypeObject (object)
    
    # 返回对象的大小（元素数量）
    npy_intp PyArray_Size (object)
    
    # 确保返回的对象是 ndarray 类型
    object PyArray_EnsureArray (object)
    
    # 确保返回的对象是任意数组类型
    object PyArray_EnsureAnyArray (object)
    object PyArray_Return (ndarray)
    # 返回一个 ndarray 对象，用于函数返回值

    #object PyArray_GetField (ndarray, dtype, int)
    # 从结构化数组中获取字段值，根据给定的 dtype 和索引 int

    #int PyArray_SetField (ndarray, dtype, int, object) except -1
    # 在结构化数组中设置字段值，根据给定的 dtype、索引 int 和对象 object，若失败返回 -1

    object PyArray_Byteswap (ndarray, npy_bool)
    # 对数组进行字节交换，根据给定的 npy_bool 参数决定是否原地交换

    object PyArray_Resize (ndarray, PyArray_Dims *, int, NPY_ORDER)
    # 调整数组大小，使用 PyArray_Dims * 指定新的维度，int 指定新的维度数目，NPY_ORDER 指定数组的存储顺序

    int PyArray_CopyInto (ndarray, ndarray) except -1
    # 将一个数组的内容复制到另一个数组中，若失败返回 -1

    int PyArray_CopyAnyInto (ndarray, ndarray) except -1
    # 将任意类型的数组的内容复制到另一个数组中，若失败返回 -1

    int PyArray_CopyObject (ndarray, object) except -1
    # 将对象的内容复制到数组中，若失败返回 -1

    object PyArray_NewCopy (ndarray, NPY_ORDER)
    # 创建一个数组的深拷贝，指定数组的存储顺序为 NPY_ORDER

    object PyArray_ToList (ndarray)
    # 将数组转换为 Python 列表对象

    object PyArray_ToString (ndarray, NPY_ORDER)
    # 将数组转换为字符串，指定数组的存储顺序为 NPY_ORDER

    int PyArray_ToFile (ndarray, stdio.FILE *, char *, char *) except -1
    # 将数组保存到文件中，使用指定的文件指针 stdio.FILE *，若失败返回 -1

    int PyArray_Dump (object, object, int) except -1
    # 将对象的数据以二进制形式保存到文件中，指定格式 int，若失败返回 -1

    object PyArray_Dumps (object, int)
    # 将对象的数据以字符串形式返回，指定格式 int

    int PyArray_ValidType (int)
    # 检查给定的整数是否为有效的数组数据类型，无法报错

    void PyArray_UpdateFlags (ndarray, int)
    # 更新数组的标志位，根据给定的整数 int

    object PyArray_New (type, int, npy_intp *, int, npy_intp *, void *, int, int, object)
    # 创建一个新的数组对象，根据给定的 type 和各种参数配置

    #object PyArray_NewFromDescr (type, dtype, int, npy_intp *, npy_intp *, void *, int, object)
    # 根据数组描述符创建一个新的数组对象，具体参数见文档

    #dtype PyArray_DescrNew (dtype)
    # 根据给定的数据类型描述符创建一个新的描述符对象 dtype

    dtype PyArray_DescrNewFromType (int)
    # 根据给定的数据类型创建一个新的数据类型描述符对象 dtype

    double PyArray_GetPriority (object, double)
    # 获取对象的优先级，清除错误（自 1.25 版本开始支持）

    object PyArray_IterNew (object)
    # 创建一个数组的迭代器对象，根据给定的数组对象

    object PyArray_MultiIterNew (int, ...)
    # 创建一个多数组的迭代器对象，根据给定的数组数量和每个数组的参数

    int PyArray_PyIntAsInt (object) except? -1
    # 将 Python 整数对象转换为 C 语言整数，若失败返回 -1

    npy_intp PyArray_PyIntAsIntp (object)
    # 将 Python 整数对象转换为 C 语言整数（通常是数组索引类型）

    int PyArray_Broadcast (broadcast) except -1
    # 广播数组，使得所有输入数组都具有相同的形状和维度，若失败返回 -1

    int PyArray_FillWithScalar (ndarray, object) except -1
    # 使用标量值填充数组，若失败返回 -1

    npy_bool PyArray_CheckStrides (int, int, npy_intp, npy_intp, npy_intp *, npy_intp *)
    # 检查数组的步幅是否符合指定的条件，返回布尔值 npy_bool

    dtype PyArray_DescrNewByteorder (dtype, char)
    # 根据给定的数据类型描述符和字节顺序创建一个新的数据类型描述符对象 dtype

    object PyArray_IterAllButAxis (object, int *)
    # 创建一个数组的迭代器对象，除了指定的轴以外，根据给定的数组对象和轴索引 int *

    #object PyArray_CheckFromAny (object, dtype, int, int, int, object)
    # 检查是否可以从任何对象创建数组，根据给定的参数和类型

    #object PyArray_FromArray (ndarray, dtype, int)
    # 根据现有数组创建一个新的数组对象，根据给定的数据类型和参数

    object PyArray_FromInterface (object)
    # 根据给定的接口对象创建一个新的数组对象

    object PyArray_FromStructInterface (object)
    # 根据给定的结构化接口对象创建一个新的数组对象

    #object PyArray_FromArrayAttr (object, dtype, object)
    # 根据对象的属性创建一个新的数组对象，根据给定的数据类型和属性对象

    #NPY_SCALARKIND PyArray_ScalarKind (int, ndarray*)
    # 获取标量的类型种类，根据给定的整数和数组对象指针

    int PyArray_CanCoerceScalar (int, int, NPY_SCALARKIND)
    # 检查是否可以将一个标量从一种类型转换为另一种类型，根据给定的参数和类型种类

    npy_bool PyArray_CanCastScalar (type, type)
    # 检查是否可以将一个标量从一种类型强制转换为另一种类型，根据给定的数据类型

    int PyArray_RemoveSmallest (broadcast) except -1
    # 移除广播的最小形状，若失败返回 -1

    int PyArray_ElementStrides (object)
    # 计算数组元素的步幅，根据给定的数组对象

    void PyArray_Item_INCREF (char *, dtype) except *
    # 增加数组元素的引用计数，根据给定的元素地址和数据类型

    void PyArray_Item_XDECREF (char *, dtype) except *
    # 减少数组元素的引用计数，根据给定的元素地址和数据类型

    object PyArray_Transpose (ndarray, PyArray_Dims *)
    # 对数组进行转置操作，根据给定的数组对象和维度

    object PyArray_TakeFrom (ndarray, object, int, ndarray, NPY_CLIPMODE)
    # 从数组中取出指定元素构成新的数组，根据给定的参数和取值模式

    object PyArray_PutTo (ndarray, object, object, NPY_CLIPMODE)
    # 将一组值放置到数组中指定的位置，根据给定的参数和放置模式

    object PyArray_PutMask (ndarray, object, object)
    # 根据掩码数组将值放置到目标数组中，根据给定的参数

    object PyArray_Repeat (ndarray, object, int)
    # 对数组进行重复操作，根据给定的参数

    object PyArray_Choose (ndarray, object, ndarray, NPY_CLIPMODE)
    # 根据索引数组从给定的选择数组中选择值，根据给定的参数和选择模式

    int PyArray_Sort (ndarray, int, NPY_SORTKIND) except -1
    # 对数组进行排序操作，根据给定的
    # 创建一个新的数组，将其形状重塑为指定维度和顺序
    object PyArray_Newshape (ndarray, PyArray_Dims *, NPY_ORDER)
    
    # 去除数组中维度为1的轴
    object PyArray_Squeeze (ndarray)
    
    # 交换数组中指定的两个轴的位置
    object PyArray_SwapAxes (ndarray, int, int)
    
    # 返回数组中指定轴上的最大值
    object PyArray_Max (ndarray, int, ndarray)
    
    # 返回数组中指定轴上的最小值
    object PyArray_Min (ndarray, int, ndarray)
    
    # 返回数组中指定轴上的最大值和最小值之差
    object PyArray_Ptp (ndarray, int, ndarray)
    
    # 返回数组中指定轴上的平均值
    object PyArray_Mean (ndarray, int, int, ndarray)
    
    # 返回数组对角线元素的和
    object PyArray_Trace (ndarray, int, int, int, int, ndarray)
    
    # 返回数组对角线的视图
    object PyArray_Diagonal (ndarray, int, int, int)
    
    # 将数组元素限制在给定范围内
    object PyArray_Clip (ndarray, object, object, ndarray)
    
    # 返回数组中非零元素的索引数组
    object PyArray_Nonzero (ndarray)
    
    # 返回数组指定轴上的标准差
    object PyArray_Std (ndarray, int, int, ndarray, int)
    
    # 返回数组指定轴上的元素和
    object PyArray_Sum (ndarray, int, int, ndarray)
    
    # 返回数组指定轴上的累积和
    object PyArray_CumSum (ndarray, int, int, ndarray)
    
    # 返回数组指定轴上的元素乘积
    object PyArray_Prod (ndarray, int, int, ndarray)
    
    # 返回数组指定轴上的累积乘积
    object PyArray_CumProd (ndarray, int, int, ndarray)
    
    # 检查数组指定轴上的所有元素是否为真
    object PyArray_All (ndarray, int, ndarray)
    
    # 检查数组指定轴上是否有任何元素为真
    object PyArray_Any (ndarray, int, ndarray)
    
    # 压缩数组，根据条件筛选元素
    object PyArray_Compress (ndarray, object, int, ndarray)
    
    # 返回一个展平的数组
    object PyArray_Flatten (ndarray, NPY_ORDER)
    
    # 返回一个展平的数组，与 Flatten 功能相似
    object PyArray_Ravel (ndarray, NPY_ORDER)
    
    # 计算列表中整数的乘积
    npy_intp PyArray_MultiplyList (npy_intp *, int)
    
    # 计算整数列表中整数的乘积
    int PyArray_MultiplyIntList (int *, int)
    
    # 返回数组中指定索引的元素指针
    void * PyArray_GetPtr (ndarray, npy_intp*)
    
    # 比较两个整数数组是否相等
    int PyArray_CompareLists (npy_intp *, npy_intp *, int)
    
    # 从 Python 序列中提取整数并转换为数组的索引
    int PyArray_IntpFromSequence (object, npy_intp *, int) except -1
    
    # 连接数组的序列，沿指定轴连接
    object PyArray_Concatenate (object, int)
    
    # 计算数组的内积
    object PyArray_InnerProduct (object, object)
    
    # 计算两个数组的矩阵乘积
    object PyArray_MatrixProduct (object, object)
    
    # 计算数组的相关性
    object PyArray_Correlate (object, object, int)
    
    # 检查两个 dtype 是否等效
    unsigned char PyArray_EquivTypes (dtype, dtype)  # 清除错误
    
    # 根据条件返回数组的索引
    object PyArray_Where (object, object, object)
    
    # 返回指定范围内的均匀间隔的值作为数组
    object PyArray_Arange (double, double, double, int)
    
    # 将排序方式的 Python 对象转换为排序枚举类型
    int PyArray_SortkindConverter (object, NPY_SORTKIND *) except 0
    
    # 返回数组元素四舍五入到指定精度
    object PyArray_Round (ndarray, int, ndarray)
    
    # 检查两个 dtype 编号是否等效
    unsigned char PyArray_EquivTypenums (int, int)
    
    # 注册自定义数据类型到 NumPy 中
    int PyArray_RegisterDataType (dtype) except -1
    
    # 注册自定义类型之间的转换函数到 NumPy 中
    int PyArray_RegisterCastFunc (dtype, int, PyArray_VectorUnaryFunc *) except -1
    # 注册可以转换的数据类型，返回是否成功注册的状态值，若失败返回 -1
    int PyArray_RegisterCanCast (dtype, int, NPY_SCALARKIND) except -1

    # 初始化数组函数操作，传入一个指向 PyArray_ArrFuncs 结构体的指针
    void PyArray_InitArrFuncs (PyArray_ArrFuncs *)

    # 将一个整数转换为长度为 int 的整数元组对象，并返回该对象
    object PyArray_IntTupleFromIntp (int, npy_intp *)

    # 将一个对象转换为 NPY_CLIPMODE 枚举类型，返回是否成功转换的状态值，若失败返回 0
    int PyArray_ClipmodeConverter (object, NPY_CLIPMODE *) except 0

    # 将一个对象转换为 ndarray 对象，返回是否成功转换的状态值，若失败返回 0
    int PyArray_OutputConverter (object, ndarray*) except 0

    # 将一个对象广播到指定形状，返回广播后的对象
    object PyArray_BroadcastToShape (object, npy_intp *, int)

    # 将一个对象转换为 dtype 类型的描述符对象，返回是否成功转换的状态值，若失败返回 0
    int PyArray_DescrAlignConverter (object, dtype*) except 0

    # 将一个对象转换为 dtype 类型的描述符对象，返回是否成功转换的状态值，若失败返回 0
    int PyArray_DescrAlignConverter2 (object, dtype*) except 0

    # 将一个对象转换为搜索方向相关的值，返回是否成功转换的状态值，若失败返回 0
    int PyArray_SearchsideConverter (object, void *) except 0

    # 检查给定的轴是否在合法范围内，返回检查后的轴值，可能会修改传入的 int 值
    object PyArray_CheckAxis (ndarray, int *, int)

    # 计算整数数组中各元素的乘积，返回结果
    npy_intp PyArray_OverflowMultiplyList (npy_intp *, int)

    # 设置 ndarray 对象的基础对象，会“偷取” base 的引用计数，返回是否成功设置的状态值，若失败返回 -1
    int PyArray_SetBaseObject(ndarray, base) except -1 # NOTE: steals a reference to base! Use "set_array_base()" instead.
# Typedefs that matches the runtime dtype objects in
# the numpy module.

# The ones that are commented out needs an IFDEF function
# in Cython to enable them only on the right systems.

# 定义各数据类型的C语言类型别名，以匹配numpy模块中的运行时数据类型对象。

ctypedef npy_int8       int8_t
ctypedef npy_int16      int16_t
ctypedef npy_int32      int32_t
ctypedef npy_int64      int64_t
#ctypedef npy_int96      int96_t  # 需要在Cython中使用IFDEF函数根据系统情况启用

ctypedef npy_uint8      uint8_t
ctypedef npy_uint16     uint16_t
ctypedef npy_uint32     uint32_t
ctypedef npy_uint64     uint64_t
#ctypedef npy_uint96     uint96_t  # 需要在Cython中使用IFDEF函数根据系统情况启用

ctypedef npy_float32    float32_t
ctypedef npy_float64    float64_t
#ctypedef npy_float80    float80_t  # 需要在Cython中使用IFDEF函数根据系统情况启用
#ctypedef npy_float128   float128_t  # 需要在Cython中使用IFDEF函数根据系统情况启用

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

# 定义Cython中的内联函数，用于创建指定数量输入的PyArray_MultiIter对象

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

# 定义Cython中的内联函数，用于根据数据类型返回其形状信息的元组

cdef inline tuple PyDataType_SHAPE(dtype d):
    if PyDataType_HASSUBARRAY(d):
        return <tuple>d.subarray.shape
    else:
        return ()

# 从numpy/ndarrayobject.h外部引入PyTypeObject类型的对象
# 用于numpy中的时间差和日期时间数组类型的处理

cdef extern from "numpy/ndarrayobject.h":
    PyTypeObject PyTimedeltaArrType_Type
    PyTypeObject PyDatetimeArrType_Type
    ctypedef int64_t npy_timedelta
    ctypedef int64_t npy_datetime

# 从numpy/ndarraytypes.h外部引入结构体和类型定义
# 用于numpy中日期时间相关的元数据和结构体的处理

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base
        int64_t num

    ctypedef struct npy_datetimestruct:
        int64_t year
        int32_t month, day, hour, min, sec, us, ps, as

# 从numpy/arrayscalars.h外部引入的抽象类型定义
# 用于numpy中的各种数值类型的处理

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
    # 定义一个 Cython 类，表示 NumPy 中的 flexible 类型
    ctypedef class numpy.flexible [object PyObject]:
        pass
    
    # 定义一个 Cython 类，表示 NumPy 中的 character 类型
    ctypedef class numpy.character [object PyObject]:
        pass
    
    # 定义一个 Cython 结构体，用于存储 Python 中的日期时间标量对象
    ctypedef struct PyDatetimeScalarObject:
        # PyObject_HEAD
        npy_datetime obval  # 存储日期时间值
        PyArray_DatetimeMetaData obmeta  # 存储日期时间元数据
    
    # 定义一个 Cython 结构体，用于存储 Python 中的时间增量标量对象
    ctypedef struct PyTimedeltaScalarObject:
        # PyObject_HEAD
        npy_timedelta obval  # 存储时间增量值
        PyArray_DatetimeMetaData obmeta  # 存储时间增量元数据
    
    # 定义一个 Cython 枚举，表示 NumPy 中的日期时间单位
    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_Y  # 年
        NPY_FR_M  # 月
        NPY_FR_W  # 周
        NPY_FR_D  # 天
        NPY_FR_B  # 工作日
        NPY_FR_h  # 小时
        NPY_FR_m  # 分钟
        NPY_FR_s  # 秒
        NPY_FR_ms  # 毫秒
        NPY_FR_us  # 微秒
        NPY_FR_ns  # 纳秒
        NPY_FR_ps  # 皮秒
        NPY_FR_fs  # 飞秒
        NPY_FR_as  # 阿秒
        NPY_FR_GENERIC  # 通用日期时间单位
cdef extern from "numpy/arrayobject.h":
    # 定义了以下函数作为 NumPy 的 C-API 的一部分

    # 在 datetime_strings.c 中定义的函数，返回 ISO 8601 格式日期时间字符串长度
    int get_datetime_iso_8601_strlen "NpyDatetime_GetDatetimeISO8601StrLen" (
            int local, NPY_DATETIMEUNIT base)
    # 在 datetime_strings.c 中定义的函数，将日期时间结构体转换为 ISO 8601 格式字符串
    int make_iso_8601_datetime "NpyDatetime_MakeISO8601Datetime" (
            npy_datetimestruct *dts, char *outstr, npy_intp outlen,
            int local, int utc, NPY_DATETIMEUNIT base, int tzoffset,
            NPY_CASTING casting) except -1

    # 在 datetime.c 中定义的函数，将 Python datetime 对象转换为日期时间结构体
    # 如果对象不是日期时间类型，可能返回 1 （成功返回 0）
    int convert_pydatetime_to_datetimestruct "NpyDatetime_ConvertPyDateTimeToDatetimeStruct" (
            PyObject *obj, npy_datetimestruct *out,
            NPY_DATETIMEUNIT *out_bestunit, int apply_tzinfo) except -1
    # 在 datetime.c 中定义的函数，将 datetime64 数据转换为日期时间结构体
    int convert_datetime64_to_datetimestruct "NpyDatetime_ConvertDatetime64ToDatetimeStruct" (
            PyArray_DatetimeMetaData *meta, npy_datetime dt,
            npy_datetimestruct *out) except -1
    # 在 datetime.c 中定义的函数，将日期时间结构体转换为 datetime64 数据
    int convert_datetimestruct_to_datetime64 "NpyDatetime_ConvertDatetimeStructToDatetime64"(
            PyArray_DatetimeMetaData *meta, const npy_datetimestruct *dts,
            npy_datetime *out) except -1


#
# ufunc API
#

cdef extern from "numpy/ufuncobject.h":
    # 定义了 ufunc 相关的 API 和类型

    # 定义了 PyUFuncGenericFunction 类型，表示 ufunc 的通用函数指针类型
    ctypedef void (*PyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *)

    # 定义了 numpy.ufunc 类型的结构体
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

    # 定义了一些 ufunc 相关的枚举常量
    cdef enum:
        PyUFunc_Zero
        PyUFunc_One
        PyUFunc_None
        UFUNC_FPE_DIVIDEBYZERO
        UFUNC_FPE_OVERFLOW
        UFUNC_FPE_UNDERFLOW
        UFUNC_FPE_INVALID

    # 声明了 PyUFunc_FromFuncAndData 函数，用于创建 ufunc 对象并初始化其函数指针和数据指针
    object PyUFunc_FromFuncAndData(PyUFuncGenericFunction *,
          void **, char *, int, int, int, int, char *, char *, int)
    # 声明了 PyUFunc_RegisterLoopForType 函数，注册指定类型的循环函数到 ufunc 中
    int PyUFunc_RegisterLoopForType(ufunc, int,
                                    PyUFuncGenericFunction, int *, void *) except -1
    # 声明了一系列特定类型的 ufunc 通用函数
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
    # 定义一个不返回值的函数 PyUFunc_O_O，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_O_O \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_ff_f_As_dd_d，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_ff_f_As_dd_d \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_ff_f，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_ff_f \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_dd_d，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_dd_d \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_gg_g，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_gg_g \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_FF_F_As_DD_D，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_FF_F_As_DD_D \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_DD_D，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_DD_D \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_FF_F，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_FF_F \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_GG_G，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_GG_G \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_OO_O，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_OO_O \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的方法 PyUFunc_O_O_method，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_O_O_method \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的方法 PyUFunc_OO_O_method，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_OO_O_method \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的方法 PyUFunc_On_Om，接受 char**、npy_intp*、npy_intp* 和 void* 作为参数
    void PyUFunc_On_Om \
         (char **, npy_intp *, npy_intp *, void *)
    
    # 定义一个不返回值的函数 PyUFunc_clearfperr，清除浮点错误状态
    void PyUFunc_clearfperr()
    
    # 返回当前浮点错误状态的函数 PyUFunc_getfperr
    int PyUFunc_getfperr()
    
    # 用新的函数指针替换给定签名的循环函数
    int PyUFunc_ReplaceLoopBySignature \
        (ufunc, PyUFuncGenericFunction, int *, PyUFuncGenericFunction *)
    
    # 根据给定的函数指针和签名创建一个 ufunc 对象
    object PyUFunc_FromFuncAndDataAndSignature \
             (PyUFuncGenericFunction *, void **, char *, int, int, int,
              int, char *, char *, int, char *)
    
    # 导入 umath 模块，返回 0 表示成功，-1 表示失败
    int _import_umath() except -1
# 增加数组对象的基础对象引用计数，确保在下面的引用转移之前执行此操作！
cdef inline void set_array_base(ndarray arr, object base) except *:
    Py_INCREF(base)  # 重要的是在下面的引用转移之前执行此操作！

cdef inline object get_array_base(ndarray arr):
    base = PyArray_BASE(arr)
    if base is NULL:
        return None
    return <object>base

# 适用于 Cython 代码的 import_* 函数的版本。
cdef inline int import_array() except -1:
    try:
        __pyx_import_array()
    except Exception:
        raise ImportError("numpy._core.multiarray failed to import")

cdef inline int import_umath() except -1:
    try:
        _import_umath()
    except Exception:
        raise ImportError("numpy._core.umath failed to import")

cdef inline int import_ufunc() except -1:
    try:
        _import_umath()
    except Exception:
        raise ImportError("numpy._core.umath failed to import")

# 检查对象是否为 np.timedelta64 类型的 Cython 等效实现。
cdef inline bint is_timedelta64_object(object obj) noexcept:
    """
    Cython equivalent of `isinstance(obj, np.timedelta64)`

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return PyObject_TypeCheck(obj, &PyTimedeltaArrType_Type)

# 检查对象是否为 np.datetime64 类型的 Cython 等效实现。
cdef inline bint is_datetime64_object(object obj) noexcept:
    """
    Cython equivalent of `isinstance(obj, np.datetime64)`

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return PyObject_TypeCheck(obj, &PyDatetimeArrType_Type)

# 获取 numpy datetime64 标量对象底层的 int64 值，不使用 GIL。
cdef inline npy_datetime get_datetime64_value(object obj) noexcept nogil:
    """
    returns the int64 value underlying scalar numpy datetime64 object

    Note that to interpret this as a datetime, the corresponding unit is
    also needed.  That can be found using `get_datetime64_unit`.
    """
    return (<PyDatetimeScalarObject*>obj).obval

# 获取 numpy timedelta64 标量对象底层的 int64 值，不使用 GIL。
cdef inline npy_timedelta get_timedelta64_value(object obj) noexcept nogil:
    """
    returns the int64 value underlying scalar numpy timedelta64 object
    """
    return (<PyTimedeltaScalarObject*>obj).obval

# 获取 numpy datetime64 对象的单位部分。
cdef inline NPY_DATETIMEUNIT get_datetime64_unit(object obj) noexcept nogil:
    """
    returns the unit part of the dtype for a numpy datetime64 object.
    """
    return <NPY_DATETIMEUNIT>(<PyDatetimeScalarObject*>obj).obmeta.base

# 在 v1.6 中添加的迭代器 API。
ctypedef int (*NpyIter_IterNextFunc)(NpyIter* it) noexcept nogil
ctypedef void (*NpyIter_GetMultiIndexFunc)(NpyIter* it, npy_intp* outcoords) noexcept nogil

cdef extern from "numpy/arrayobject.h":
    # 定义 NpyIter 结构体，用于迭代器 API。
    ctypedef struct NpyIter:
        pass

    # 定义返回值枚举，表示操作成功或失败。
    cdef enum:
        NPY_FAIL
        NPY_SUCCEED
    # 枚举常量，用于迭代器的不同设置
    cdef enum:
        # 表示 C 顺序的索引
        NPY_ITER_C_INDEX
        # 表示 Fortran 顺序的索引
        NPY_ITER_F_INDEX
        # 多维索引
        NPY_ITER_MULTI_INDEX
        # 外部代码处理一维最内层循环
        NPY_ITER_EXTERNAL_LOOP
        # 将所有操作数转换为共同的数据类型
        NPY_ITER_COMMON_DTYPE
        # 操作数可能包含引用，迭代过程中需要 API 访问
        NPY_ITER_REFS_OK
        # 允许零大小的操作数，迭代检查 IterSize 是否为 0
        NPY_ITER_ZEROSIZE_OK
        # 允许缩减操作（大小为 0 的步幅，但维度大小大于 1）
        NPY_ITER_REDUCE_OK
        # 启用子范围迭代
        NPY_ITER_RANGED
        # 启用缓冲
        NPY_ITER_BUFFERED
        # 当启用缓冲时，尽可能增长内部循环
        NPY_ITER_GROWINNER
        # 延迟分配缓冲区，直到第一次 Reset* 调用
        NPY_ITER_DELAY_BUFALLOC
        # 当指定 NPY_KEEPORDER 时，禁止反转负步幅轴
        NPY_ITER_DONT_NEGATE_STRIDES
        # 如果有重叠，则复制操作数
        NPY_ITER_COPY_IF_OVERLAP
        # 操作数将被读取和写入
        NPY_ITER_READWRITE
        # 操作数只能被读取
        NPY_ITER_READONLY
        # 操作数只能被写入
        NPY_ITER_WRITEONLY
        # 操作数的数据必须是本机字节顺序
        NPY_ITER_NBO
        # 操作数的数据必须对齐
        NPY_ITER_ALIGNED
        # 操作数的数据必须在内部循环中是连续的
        NPY_ITER_CONTIG
        # 可以复制操作数以满足要求
        NPY_ITER_COPY
        # 可以使用 WRITEBACKIFCOPY 复制操作数以满足要求
        NPY_ITER_UPDATEIFCOPY
        # 如果操作数为 NULL，则分配它
        NPY_ITER_ALLOCATE
        # 如果操作数被分配，则不使用任何子类型
        NPY_ITER_NO_SUBTYPE
        # 这是虚拟数组槽，操作数为 NULL，但存在临时数据
        NPY_ITER_VIRTUAL
        # 要求维度与迭代器维度完全匹配
        NPY_ITER_NO_BROADCAST
        # 此数组正在使用掩码，影响缓冲区 -> 数组复制
        NPY_ITER_WRITEMASKED
        # 此数组是所有 WRITEMASKED 操作数的掩码
        NPY_ITER_ARRAYMASK
        # 假定迭代器顺序数据访问以处理 COPY_IF_OVERLAP
        NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE

    # 创建和销毁函数声明
    # 创建新的迭代器对象，返回 NpyIter 指针
    NpyIter* NpyIter_New(ndarray arr, npy_uint32 flags, NPY_ORDER order,
                         NPY_CASTING casting, dtype datatype) except NULL
    # 创建多个迭代器对象，返回 NpyIter 指针
    NpyIter* NpyIter_MultiNew(npy_intp nop, PyArrayObject** op, npy_uint32 flags,
                              NPY_ORDER order, NPY_CASTING casting, npy_uint32*
                              op_flags, PyArray_Descr** op_dtypes) except NULL
    // 创建一个新的 NpyIter 对象，并返回其指针
    NpyIter* NpyIter_AdvancedNew(npy_intp nop, PyArrayObject** op,
                                 npy_uint32 flags, NPY_ORDER order,
                                 NPY_CASTING casting, npy_uint32* op_flags,
                                 PyArray_Descr** op_dtypes, int oa_ndim,
                                 int** op_axes, const npy_intp* itershape,
                                 npy_intp buffersize) except NULL

    // 复制给定的 NpyIter 对象，并返回其副本的指针
    NpyIter* NpyIter_Copy(NpyIter* it) except NULL

    // 从 NpyIter 对象中移除指定的轴
    int NpyIter_RemoveAxis(NpyIter* it, int axis) except NPY_FAIL

    // 从 NpyIter 对象中移除多重索引
    int NpyIter_RemoveMultiIndex(NpyIter* it) except NPY_FAIL

    // 启用 NpyIter 对象的外部循环模式
    int NpyIter_EnableExternalLoop(NpyIter* it) except NPY_FAIL

    // 释放 NpyIter 对象的内存资源
    int NpyIter_Deallocate(NpyIter* it) except NPY_FAIL

    // 重置 NpyIter 对象的迭代状态
    int NpyIter_Reset(NpyIter* it, char** errmsg) except NPY_FAIL

    // 重置 NpyIter 对象的迭代状态，限定于指定的迭代索引范围
    int NpyIter_ResetToIterIndexRange(NpyIter* it, npy_intp istart,
                                      npy_intp iend, char** errmsg) except NPY_FAIL

    // 设置 NpyIter 对象的基础指针，用于重置迭代状态
    int NpyIter_ResetBasePointers(NpyIter* it, char** baseptrs, char** errmsg) except NPY_FAIL

    // 将 NpyIter 对象的迭代位置移动到指定的多重索引位置
    int NpyIter_GotoMultiIndex(NpyIter* it, const npy_intp* multi_index) except NPY_FAIL

    // 将 NpyIter 对象的迭代位置移动到指定的索引位置
    int NpyIter_GotoIndex(NpyIter* it, npy_intp index) except NPY_FAIL

    // 返回 NpyIter 对象的迭代尺寸
    npy_intp NpyIter_GetIterSize(NpyIter* it) nogil

    // 返回 NpyIter 对象的当前迭代索引
    npy_intp NpyIter_GetIterIndex(NpyIter* it) nogil

    // 返回 NpyIter 对象的迭代索引范围
    void NpyIter_GetIterIndexRange(NpyIter* it, npy_intp* istart,
                                   npy_intp* iend) nogil

    // 将 NpyIter 对象的迭代位置移动到指定的迭代索引
    int NpyIter_GotoIterIndex(NpyIter* it, npy_intp iterindex) except NPY_FAIL

    // 检查 NpyIter 对象是否有延迟缓冲区分配
    npy_bool NpyIter_HasDelayedBufAlloc(NpyIter* it) nogil

    // 检查 NpyIter 对象是否使用外部循环模式
    npy_bool NpyIter_HasExternalLoop(NpyIter* it) nogil

    // 检查 NpyIter 对象是否有多重索引
    npy_bool NpyIter_HasMultiIndex(NpyIter* it) nogil

    // 检查 NpyIter 对象是否有索引
    npy_bool NpyIter_HasIndex(NpyIter* it) nogil

    // 检查 NpyIter 对象是否需要缓冲区
    npy_bool NpyIter_RequiresBuffering(NpyIter* it) nogil

    // 检查 NpyIter 对象是否已经缓冲
    npy_bool NpyIter_IsBuffered(NpyIter* it) nogil

    // 检查 NpyIter 对象是否在增长内部
    npy_bool NpyIter_IsGrowInner(NpyIter* it) nogil

    // 返回 NpyIter 对象的缓冲区大小
    npy_intp NpyIter_GetBufferSize(NpyIter* it) nogil

    // 返回 NpyIter 对象的维度数
    int NpyIter_GetNDim(NpyIter* it) nogil

    // 返回 NpyIter 对象的操作数数量
    int NpyIter_GetNOp(NpyIter* it) nogil

    // 返回 NpyIter 对象指定轴的步长数组
    npy_intp* NpyIter_GetAxisStrideArray(NpyIter* it, int axis) except NULL

    // 返回 NpyIter 对象的形状数组
    int NpyIter_GetShape(NpyIter* it, npy_intp* outshape) nogil

    // 返回 NpyIter 对象的数据类型数组
    PyArray_Descr** NpyIter_GetDescrArray(NpyIter* it)

    // 返回 NpyIter 对象的操作数数组
    PyArrayObject** NpyIter_GetOperandArray(NpyIter* it)

    // 返回 NpyIter 对象指定索引的迭代视图
    ndarray NpyIter_GetIterView(NpyIter* it, npy_intp i)

    // 将 NpyIter 对象的读取标志复制到输出数组
    void NpyIter_GetReadFlags(NpyIter* it, char* outreadflags)

    // 将 NpyIter 对象的写入标志复制到输出数组
    void NpyIter_GetWriteFlags(NpyIter* it, char* outwriteflags)

    // 创建与 NpyIter 对象兼容的步长数组
    int NpyIter_CreateCompatibleStrides(NpyIter* it, npy_intp itemsize,
                                        npy_intp* outstrides) except NPY_FAIL

    // 检查指定操作数是否首次访问
    npy_bool NpyIter_IsFirstVisit(NpyIter* it, int iop) nogil

    // 返回用于迭代 NpyIter 对象的下一个函数指针
    NpyIter_IterNextFunc* NpyIter_GetIterNext(NpyIter* it, char** errmsg) except NULL

    // 返回用于获取 NpyIter 对象的多重索引的函数指针
    NpyIter_GetMultiIndexFunc* NpyIter_GetGetMultiIndex(NpyIter* it,
                                                        char** errmsg) except NULL
    # 获取迭代器中数据指针数组的地址，用于访问数据，不使用全局解释器锁（GIL）
    char** NpyIter_GetDataPtrArray(NpyIter* it) nogil
    
    # 获取迭代器初始数据指针数组的地址，不使用全局解释器锁（GIL）
    char** NpyIter_GetInitialDataPtrArray(NpyIter* it) nogil
    
    # 获取迭代器的索引指针数组的地址
    npy_intp* NpyIter_GetIndexPtr(NpyIter* it)
    
    # 获取迭代器内部步幅数组的地址，不使用全局解释器锁（GIL）
    npy_intp* NpyIter_GetInnerStrideArray(NpyIter* it) nogil
    
    # 获取迭代器内部循环大小指针的地址，不使用全局解释器锁（GIL）
    npy_intp* NpyIter_GetInnerLoopSizePtr(NpyIter* it) nogil
    
    # 将迭代器的内部固定步幅数组复制到给定的输出数组中，不使用全局解释器锁（GIL）
    void NpyIter_GetInnerFixedStrideArray(NpyIter* it, npy_intp* outstrides) nogil
    
    # 检查迭代器是否需要 Python C API 支持，不使用全局解释器锁（GIL）
    npy_bool NpyIter_IterationNeedsAPI(NpyIter* it) nogil
    
    # 调试用：打印迭代器的调试信息
    void NpyIter_DebugPrint(NpyIter* it)
```