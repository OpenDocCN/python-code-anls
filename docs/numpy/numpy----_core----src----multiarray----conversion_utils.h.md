# `.\numpy\numpy\_core\src\multiarray\conversion_utils.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_CONVERSION_UTILS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CONVERSION_UTILS_H_

#include "numpy/ndarraytypes.h"

// 定义整数转换函数，将 Python 对象转换为 PyArray_Dims 结构
NPY_NO_EXPORT int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq);

// 定义从 Python 整数对象转换为 npy_intp 类型的函数
NPY_NO_EXPORT int
PyArray_IntpFromPyIntConverter(PyObject *o, npy_intp *val);

// 定义可选整数转换函数，将 Python 对象转换为 PyArray_Dims 结构
NPY_NO_EXPORT int
PyArray_OptionalIntpConverter(PyObject *obj, PyArray_Dims *seq);

// 定义复制模式的枚举类型
typedef enum {
    NPY_COPY_NEVER = 0,
    NPY_COPY_ALWAYS = 1,
    NPY_COPY_IF_NEEDED = 2,
} NPY_COPYMODE;

// 定义转换函数，将 Python 对象转换为 NPY_COPYMODE 枚举类型
NPY_NO_EXPORT int
PyArray_CopyConverter(PyObject *obj, NPY_COPYMODE *copyflag);

// 定义类型转换和复制模式的枚举类型
typedef enum {
    NPY_AS_TYPE_COPY_IF_NEEDED = 0,
    NPY_AS_TYPE_COPY_ALWAYS = 1,
} NPY_ASTYPECOPYMODE;

// 定义类型转换和复制模式的转换函数
NPY_NO_EXPORT int
PyArray_AsTypeCopyConverter(PyObject *obj, NPY_ASTYPECOPYMODE *copyflag);

// 定义缓冲区转换函数，将 Python 对象转换为 PyArray_Chunk 结构
NPY_NO_EXPORT int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf);

// 定义布尔值转换函数，将 Python 对象转换为 npy_bool 类型
NPY_NO_EXPORT int
PyArray_BoolConverter(PyObject *object, npy_bool *val);

// 定义可选布尔值转换函数，将 Python 对象转换为 int 类型
NPY_NO_EXPORT int
PyArray_OptionalBoolConverter(PyObject *object, int *val);

// 定义字节顺序转换函数，将 Python 对象转换为 char 类型
NPY_NO_EXPORT int
PyArray_ByteorderConverter(PyObject *obj, char *endian);

// 定义排序方式转换函数，将 Python 对象转换为 NPY_SORTKIND 枚举类型
NPY_NO_EXPORT int
PyArray_SortkindConverter(PyObject *obj, NPY_SORTKIND *sortkind);

// 定义搜索方向转换函数，将 Python 对象转换为 void 指针类型
NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr);

// 定义将 Python 整数对象转换为 int 类型的函数
NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o);

// 定义将 Python 整数对象转换为 npy_intp 类型的函数
NPY_NO_EXPORT npy_intp
PyArray_PyIntAsIntp(PyObject *o);

// 定义从索引序列转换为 npy_intp 数组的函数
NPY_NO_EXPORT npy_intp
PyArray_IntpFromIndexSequence(PyObject *seq, npy_intp *vals, npy_intp maxvals);

// 定义从序列转换为 npy_intp 数组的函数
NPY_NO_EXPORT int
PyArray_IntpFromSequence(PyObject *seq, npy_intp *vals, int maxvals);

// 定义类型字符串转换函数，接受 itemsize 和 gentype 两个整数参数
NPY_NO_EXPORT int
PyArray_TypestrConvert(int itemsize, int gentype);

// 定义将 npy_intp 类型值转换为 Python 整数对象的内联函数
static inline PyObject *
PyArray_PyIntFromIntp(npy_intp const value)
{
    // 根据平台字长选择不同的 PyLong_From* 函数转换
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyLong_FromLong((long)value);
#else
    return PyLong_FromLongLong((npy_longlong)value);
#endif
}

// 定义将 npy_intp 数组转换为 Python 元组对象的函数
NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, npy_intp const *vals);

// 定义相关模式转换函数，将 Python 对象转换为 NPY_CORRELATEMODE 枚举类型
NPY_NO_EXPORT int
PyArray_CorrelatemodeConverter(PyObject *object, NPY_CORRELATEMODE *val);

// 定义选择类型转换函数，将 Python 对象转换为 NPY_SELECTKIND 枚举类型
NPY_NO_EXPORT int
PyArray_SelectkindConverter(PyObject *obj, NPY_SELECTKIND *selectkind);

/*
 * 将轴参数转换为长度为 ndim 的布尔标志数组，
 * 每个指定的轴对应的标志为 True。
 *
 * 如果 obj 是 None，则所有标志设置为 True。
 * 如果 obj 是元组，则元组中的每个轴设置为 True。
 * 如果 obj 是整数，则只有该轴设置为 True。
 */
NPY_NO_EXPORT int
PyArray_ConvertMultiAxis(PyObject *axis_in, int ndim, npy_bool *out_axis_flags);

// 设备字符串转换函数，将 Python 对象转换为 NPY_DEVICE 枚举类型
NPY_NO_EXPORT int
PyArray_DeviceConverterOptional(PyObject *object, NPY_DEVICE *device);

#endif  // NUMPY_CORE_SRC_MULTIARRAY_CONVERSION_UTILS_H_
/**
 * WARNING: This flag is a bad idea, but was the only way to both
 *   1) Support unpickling legacy pickles with object types.
 *   2) Deprecate (and later disable) usage of O4 and O8
 *
 * The key problem is that the pickled representation unpickles by
 * directly calling the dtype constructor, which has no way of knowing
 * that it is in an unpickle context instead of a normal context without
 * evil global state like we create here.
 */
extern NPY_NO_EXPORT NPY_TLS int evil_global_disable_warn_O4O8_flag;

/*
 * Convert function which replaces np._NoValue with NULL.
 * As a converter returns 0 on error and 1 on success.
 */
NPY_NO_EXPORT int
_not_NoValue(PyObject *obj, PyObject **out);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CONVERSION_UTILS_H_ */


注释：
```