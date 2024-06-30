# `D:\src\scipysrc\scipy\scipy\ndimage\src\_cytest.pyx`

```
# 导入需要的 CPython 内存操作函数和 PyCapsule 相关函数
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.pycapsule cimport (
    PyCapsule_New, PyCapsule_SetContext, PyCapsule_GetContext, PyCapsule_GetPointer
)

# 导入 NumPy 并且指定 cimport 类型为 npy_intp
cimport numpy as np
from numpy cimport npy_intp as intp

# 调用 NumPy 的 import_array() 函数
np.import_array()

# 定义私有函数 _destructor，用于释放回调数据
cdef void _destructor(obj) noexcept:
    cdef void *callback_data = PyCapsule_GetContext(obj)
    PyMem_Free(callback_data)

# 定义私有函数 _destructor_data，用于释放回调数据
cdef void _destructor_data(obj) noexcept:
    cdef void *callback_data = PyCapsule_GetPointer(obj, NULL)
    PyMem_Free(callback_data)

# 定义私有函数 _filter1d，实现一维滤波操作
cdef int _filter1d(double *input_line, intp input_length, double *output_line,
               intp output_length, void *callback_data) noexcept:
    cdef intp i, j
    cdef intp filter_size = (<intp *>callback_data)[0]

    for i in range(output_length):
        output_line[i] = 0
        for j in range(filter_size):
            output_line[i] += input_line[i+j]
        output_line[i] /= filter_size
    return 1

# 定义 Python 可调用函数 filter1d，创建并返回一维滤波器的 PyCapsule
def filter1d(intp filter_size, with_signature=False):
    cdef intp *callback_data = <intp *>PyMem_Malloc(sizeof(intp))
    cdef char *signature = NULL
    if not callback_data:
        raise MemoryError()
    callback_data[0] = filter_size

    if with_signature:
        signature = "int (double *, npy_intp, double *, npy_intp, void *)"

    try:
        capsule = PyCapsule_New(<void *>_filter1d, signature, _destructor)
        PyCapsule_SetContext(capsule, callback_data)
    except:  # noqa: E722
        PyMem_Free(callback_data)
        raise
    return capsule

# 定义 Python 可调用函数 filter1d_capsule，创建并返回一维滤波器的 PyCapsule
def filter1d_capsule(intp filter_size):
    cdef intp *callback_data = <intp *>PyMem_Malloc(sizeof(intp))
    if not callback_data:
        raise MemoryError()
    callback_data[0] = filter_size

    try:
        capsule = PyCapsule_New(<void *>callback_data, NULL, _destructor_data)
    except:  # noqa: E722
        PyMem_Free(callback_data)
        raise
    return capsule

# 定义私有函数 _filter2d，实现二维滤波操作
cdef int _filter2d(double *buffer, intp filter_size, double *res,
               void *callback_data) noexcept:
    cdef intp i
    cdef double *weights = <double *>callback_data

    res[0] = 0
    for i in range(filter_size):
        res[0] += weights[i]*buffer[i]
    return 1

# 定义 Python 可调用函数 filter2d，创建并返回二维滤波器的 PyCapsule
def filter2d(seq, with_signature=False):
    cdef double *callback_data = <double *>PyMem_Malloc(len(seq)*sizeof(double))
    cdef char *signature = NULL
    if not callback_data:
        raise MemoryError()
    for i, item in enumerate(seq):
        callback_data[i] = float(item)

    if with_signature:
        signature = "int (double *, npy_intp, double *, void *)"

    try:
        capsule = PyCapsule_New(<void *>_filter2d, signature, _destructor)
        PyCapsule_SetContext(capsule, callback_data)
    except:  # noqa: E722
        PyMem_Free(callback_data)
        raise
    return capsule

# 定义 Python 可调用函数 filter2d_capsule，创建并返回二维滤波器的 PyCapsule
def filter2d_capsule(seq):
    cdef double *callback_data = <double *>PyMem_Malloc(len(seq)*sizeof(double))
    if not callback_data:
        raise MemoryError()
    # 遍历序列 seq，使用 enumerate 获取索引 i 和对应的元素 item
    for i, item in enumerate(seq):
        # 将每个元素 item 转换为浮点数，并存储在 callback_data 中的对应索引 i 处
        callback_data[i] = float(item)

    try:
        # 尝试创建一个 PyCapsule 对象，用于封装 callback_data 的指针
        capsule = PyCapsule_New(<void *>callback_data, NULL, _destructor_data)
    except:  # 捕获所有异常，不发出 E722 警告
        # 在异常情况下，释放 callback_data 所占用的内存
        PyMem_Free(callback_data)
        # 重新抛出当前异常
        raise
    # 返回创建的 PyCapsule 对象
    return capsule
# 定义一个Cython函数，用于将输出坐标转换为输入坐标，并返回整数值1
cdef int _transform(intp *output_coordinates, double *input_coordinates,
                int output_rank, int input_rank, void *callback_data) noexcept:
    # 声明一个整数变量i
    cdef intp i
    # 从回调数据中获取偏移量，转换为双精度浮点数
    cdef double shift = (<double *>callback_data)[0]

    # 遍历输入坐标的维度
    for i in range(input_rank):
        # 根据输出坐标和偏移量，计算输入坐标
        input_coordinates[i] = output_coordinates[i] - shift
    # 返回整数值1，表示转换成功
    return 1


# 定义一个Python函数，返回一个PyCapsule对象，其中包含_transform函数的指针和相关数据
def transform(double shift, with_signature=False):
    # 分配双精度浮点数大小的内存，存储回调数据
    cdef double *callback_data = <double *>PyMem_Malloc(sizeof(double))
    cdef char *signature = NULL
    # 如果内存分配失败，则抛出内存错误异常
    if not callback_data:
        raise MemoryError()
    # 将偏移量存储到回调数据中
    callback_data[0] = shift

    # 如果需要函数签名
    if with_signature:
        # 设置函数签名为指定格式
        signature = "int (npy_intp *, double *, int, int, void *)"

    try:
        # 创建一个PyCapsule对象，包含_transform函数的指针，指定签名和析构函数_destructor
        capsule = PyCapsule_New(<void *>_transform, signature, _destructor)
        # 将回调数据设置为PyCapsule对象的上下文数据
        PyCapsule_SetContext(capsule, callback_data)
    except:  # 捕获所有异常
        # 如果出现异常，释放回调数据内存并重新抛出异常
        PyMem_Free(callback_data)
        raise
    # 返回创建的PyCapsule对象
    return capsule


# 定义一个Python函数，返回一个PyCapsule对象，其中包含指向callback_data的指针和析构函数_destructor_data
def transform_capsule(double shift):
    # 分配双精度浮点数大小的内存，存储回调数据
    cdef double *callback_data = <double *>PyMem_Malloc(sizeof(double))
    # 如果内存分配失败，则抛出内存错误异常
    if not callback_data:
        raise MemoryError()
    # 将偏移量存储到回调数据中
    callback_data[0] = shift

    try:
        # 创建一个PyCapsule对象，包含callback_data的指针，没有指定签名，但指定了析构函数_destructor_data
        capsule = PyCapsule_New(<void *>callback_data, NULL, _destructor_data)
    except:  # 捕获所有异常
        # 如果出现异常，释放回调数据内存并重新抛出异常
        PyMem_Free(callback_data)
        raise
    # 返回创建的PyCapsule对象
    return capsule
```