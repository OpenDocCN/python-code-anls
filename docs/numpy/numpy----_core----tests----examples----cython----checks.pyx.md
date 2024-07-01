# `.\numpy\numpy\_core\tests\examples\cython\checks.pyx`

```py
"""
Functions in this module give python-space wrappers for cython functions
exposed in numpy/__init__.pxd, so they can be tested in test_cython.py
"""
# 导入Cython声明
cimport numpy as cnp
# 导入numpy数组支持
cnp.import_array()

# 检查对象是否为timedelta64类型
def is_td64(obj):
    return cnp.is_timedelta64_object(obj)

# 检查对象是否为datetime64类型
def is_dt64(obj):
    return cnp.is_datetime64_object(obj)

# 获取datetime64对象的值
def get_dt64_value(obj):
    return cnp.get_datetime64_value(obj)

# 获取timedelta64对象的值
def get_td64_value(obj):
    return cnp.get_timedelta64_value(obj)

# 获取datetime64对象的单位
def get_dt64_unit(obj):
    return cnp.get_datetime64_unit(obj)

# 检查对象是否为整数（包括numpy.int和Python的int）
def is_integer(obj):
    return isinstance(obj, (cnp.integer, int))

# 获取ISO 8601日期时间字符串的长度
def get_datetime_iso_8601_strlen():
    return cnp.get_datetime_iso_8601_strlen(0, cnp.NPY_FR_ns)

# 将datetime64对象转换为datetimestruct结构
def convert_datetime64_to_datetimestruct():
    cdef:
        cnp.npy_datetimestruct dts  # datetimestruct结构体
        cnp.PyArray_DatetimeMetaData meta  # datetime元数据
        cnp.int64_t value = 1647374515260292  # 示例时间戳

    meta.base = cnp.NPY_FR_us  # 设置时间基准为微秒
    meta.num = 1  # 设置数量为1
    cnp.convert_datetime64_to_datetimestruct(&meta, value, &dts)  # 执行转换
    return dts  # 返回datetimestruct结构体

# 生成ISO 8601格式的日期时间字符串
def make_iso_8601_datetime(dt: "datetime"):
    cdef:
        cnp.npy_datetimestruct dts  # datetimestruct结构体
        char result[36]  # 结果字符串数组，长度为36对应NPY_FR_s

    # 将datetime对象的日期时间信息填充到datetimestruct结构体中
    dts.year = dt.year
    dts.month = dt.month
    dts.day = dt.day
    dts.hour = dt.hour
    dts.min = dt.minute
    dts.sec = dt.second
    dts.us = dt.microsecond
    dts.ps = dts.as = 0  # 置ps和as字段为0

    # 生成ISO 8601格式的日期时间字符串
    cnp.make_iso_8601_datetime(
        &dts,
        result,
        sizeof(result),
        0,  # 本地时区标志
        0,  # UTC标志
        cnp.NPY_FR_s,  # 时间基准为秒
        0,  # 时区偏移量
        cnp.NPY_NO_CASTING,
    )
    return result  # 返回生成的日期时间字符串

# 从广播对象中获取multiiter结构体
cdef cnp.broadcast multiiter_from_broadcast_obj(object bcast):
    cdef dict iter_map = {
        1: cnp.PyArray_MultiIterNew1,
        2: cnp.PyArray_MultiIterNew2,
        3: cnp.PyArray_MultiIterNew3,
        4: cnp.PyArray_MultiIterNew4,
        5: cnp.PyArray_MultiIterNew5,
    }
    # 获取bcast对象中的基础数组
    arrays = [x.base for x in bcast.iters]
    # 根据数组数量选择对应的multiiter构造函数，并返回结果
    cdef cnp.broadcast result = iter_map[len(arrays)](*arrays)
    return result  # 返回广播对象的multiiter

# 获取广播对象的大小（元素数量）
def get_multiiter_size(bcast: "broadcast"):
    cdef cnp.broadcast multi = multiiter_from_broadcast_obj(bcast)
    return multi.size  # 返回广播对象的大小

# 获取广播对象的维度数
def get_multiiter_number_of_dims(bcast: "broadcast"):
    cdef cnp.broadcast multi = multiiter_from_broadcast_obj(bcast)
    return multi.nd  # 返回广播对象的维度数

# 获取广播对象的当前索引
def get_multiiter_current_index(bcast: "broadcast"):
    cdef cnp.broadcast multi = multiiter_from_broadcast_obj(bcast)
    return multi.index  # 返回广播对象当前的索引值

# 获取广播对象的迭代器数量
def get_multiiter_num_of_iterators(bcast: "broadcast"):
    cdef cnp.broadcast multi = multiiter_from_broadcast_obj(bcast)
    return multi.numiter  # 返回广播对象的迭代器数量

# 获取广播对象的形状
def get_multiiter_shape(bcast: "broadcast"):
    cdef cnp.broadcast multi = multiiter_from_broadcast_obj(bcast)
    # 返回广播对象每个维度的大小构成的元组
    return tuple([multi.dimensions[i] for i in range(bcast.nd)])

# 获取广播对象的迭代器
def get_multiiter_iters(bcast: "broadcast"):
    cdef cnp.broadcast multi = multiiter_from_broadcast_obj(bcast)
    # 创建一个元组，包含了一个生成器表达式的结果，生成器表达式用于获取<cnp.flatiter>multi.iters[i]的值，其中i的取值范围是从0到bcast.numiter-1。
    return tuple([<cnp.flatiter>multi.iters[i] for i in range(bcast.numiter)])
def get_default_integer():
    # 如果默认整数类型是 long，返回 long 类型的数据类型对象
    if cnp.NPY_DEFAULT_INT == cnp.NPY_LONG:
        return cnp.dtype("long")
    # 如果默认整数类型是 intp，返回 intp 类型的数据类型对象
    if cnp.NPY_DEFAULT_INT == cnp.NPY_INTP:
        return cnp.dtype("intp")
    # 默认情况下返回 None
    return None

def conv_intp(cnp.intp_t val):
    # 简单地返回输入的整数值
    return val

def get_dtype_flags(cnp.dtype dtype):
    # 返回给定数据类型的标志属性
    return dtype.flags

cdef cnp.NpyIter* npyiter_from_nditer_obj(object it):
    """A function to create a NpyIter struct from a nditer object.

    This function is only meant for testing purposes and only extracts the
    necessary info from nditer to test the functionality of NpyIter methods
    """
    cdef:
        cnp.NpyIter* cit
        cnp.PyArray_Descr* op_dtypes[3]
        cnp.npy_uint32 op_flags[3]
        cnp.PyArrayObject* ops[3]
        cnp.npy_uint32 flags = 0

    # 根据 nditer 对象的属性设置迭代器的标志位
    if it.has_index:
        flags |= cnp.NPY_ITER_C_INDEX
    if it.has_delayed_bufalloc:
        flags |= cnp.NPY_ITER_BUFFERED | cnp.NPY_ITER_DELAY_BUFALLOC
    if it.has_multi_index:
        flags |= cnp.NPY_ITER_MULTI_INDEX

    # 将所有操作数设置为只读模式
    for i in range(it.nop):
        op_flags[i] = cnp.NPY_ITER_READONLY

    # 获取每个操作数的数据类型和对象
    for i in range(it.nop):
        op_dtypes[i] = cnp.PyArray_DESCR(it.operands[i])
        ops[i] = <cnp.PyArrayObject*>it.operands[i]

    # 创建 NpyIter 对象，用于迭代操作
    cit = cnp.NpyIter_MultiNew(it.nop, &ops[0], flags, cnp.NPY_KEEPORDER,
                               cnp.NPY_NO_CASTING, &op_flags[0],
                               <cnp.PyArray_Descr**>NULL)
    return cit

def get_npyiter_size(it: "nditer"):
    # 创建 NpyIter 对象并获取其迭代的大小
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    result = cnp.NpyIter_GetIterSize(cit)
    cnp.NpyIter_Deallocate(cit)
    return result

def get_npyiter_ndim(it: "nditer"):
    # 创建 NpyIter 对象并获取其维度数
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    result = cnp.NpyIter_GetNDim(cit)
    cnp.NpyIter_Deallocate(cit)
    return result

def get_npyiter_nop(it: "nditer"):
    # 创建 NpyIter 对象并获取其操作数数量
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    result = cnp.NpyIter_GetNOp(cit)
    cnp.NpyIter_Deallocate(cit)
    return result

def get_npyiter_operands(it: "nditer"):
    # 创建 NpyIter 对象并获取其操作数的 ndarray 对象
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    try:
        arr = cnp.NpyIter_GetOperandArray(cit)
        return tuple([<cnp.ndarray>arr[i] for i in range(it.nop)])
    finally:
        cnp.NpyIter_Deallocate(cit)

def get_npyiter_itviews(it: "nditer"):
    # 创建 NpyIter 对象并获取其操作数的迭代视图
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    result = tuple([cnp.NpyIter_GetIterView(cit, i) for i in range(it.nop)])
    cnp.NpyIter_Deallocate(cit)
    return result

def get_npyiter_dtypes(it: "nditer"):
    # 创建 NpyIter 对象并获取其操作数的数据类型
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    try:
        arr = cnp.NpyIter_GetDescrArray(cit)
        return tuple([<cnp.dtype>arr[i] for i in range(it.nop)])
    finally:
        cnp.NpyIter_Deallocate(cit)

def npyiter_has_delayed_bufalloc(it: "nditer"):
    # 创建 NpyIter 对象并检查其是否具有延迟缓冲分配
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    result = cnp.NpyIter_HasDelayedBufAlloc(cit)
    # 调用 cnp 模块中的 NpyIter_Deallocate 函数，用于释放 NpyIter 对象占用的资源
    cnp.NpyIter_Deallocate(cit)
    # 返回 result 变量作为函数的返回结果
    return result
# 检查给定的 nditer 对象是否有索引
def npyiter_has_index(it: "nditer"):
    # 从 nditer 对象获取相应的 NpyIter 指针
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    # 调用 C 函数检查 NpyIter 对象是否有索引
    result = cnp.NpyIter_HasIndex(cit)
    # 释放 NpyIter 对象占用的内存
    cnp.NpyIter_Deallocate(cit)
    return result


# 检查给定的 nditer 对象是否有多重索引
def npyiter_has_multi_index(it: "nditer"):
    # 从 nditer 对象获取相应的 NpyIter 指针
    cdef cnp.NpyIter* cit = npyiter_from_nditer_obj(it)
    # 调用 C 函数检查 NpyIter 对象是否有多重索引
    result = cnp.NpyIter_HasMultiIndex(cit)
    # 释放 NpyIter 对象占用的内存
    cnp.NpyIter_Deallocate(cit)
    return result


# 检查给定的 nditer 对象是否已经迭代完成
def npyiter_has_finished(it: "nditer"):
    cdef cnp.NpyIter* cit
    try:
        # 尝试从 nditer 对象获取相应的 NpyIter 指针
        cit = npyiter_from_nditer_obj(it)
        # 将 NpyIter 定位到指定的迭代索引
        cnp.NpyIter_GotoIterIndex(cit, it.index)
        # 检查迭代索引是否小于迭代的总大小，并返回结果
        return not (cnp.NpyIter_GetIterIndex(cit) < cnp.NpyIter_GetIterSize(cit))
    finally:
        # 无论如何都要释放 NpyIter 对象占用的内存
        cnp.NpyIter_Deallocate(cit)


# 编译并测试填充字节的功能，用于回归测试 gh-25878
def compile_fillwithbyte():
    # 定义一个长度为 2 的 npy_intp 类型的数组 dims
    cdef cnp.npy_intp dims[2]
    dims = (1, 2)
    # 创建一个形状为 dims，类型为 NPY_UINT8 的全零数组 pos
    pos = cnp.PyArray_ZEROS(2, dims, cnp.NPY_UINT8, 0)
    # 调用 C 函数填充数组 pos 的每个元素为指定的字节值
    cnp.PyArray_FILLWBYTE(pos, 1)
    return pos


# 在 C 编译模式下增加复数结构数组的实部和虚部
def inc2_cfloat_struct(cnp.ndarray[cnp.cfloat_t] arr):
    # 只在 C 编译模式下有效，增加数组 arr 中索引为 1 的元素的实部和虚部
    arr[1].real += 1
    arr[1].imag += 1
    # 在所有编译模式下都有效，增加数组 arr 中索引为 1 的元素的实部和虚部
    arr[1].real = arr[1].real + 1
    arr[1].imag = arr[1].imag + 1
```