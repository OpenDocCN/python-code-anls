# `.\numpy\numpy\_core\src\multiarray\common.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_COMMON_H_
#define NUMPY_CORE_SRC_MULTIARRAY_COMMON_H_

#include <structmember.h>  // 导入 structmember.h 头文件
#include "numpy/npy_common.h"  // 导入 numpy 公共头文件
#include "numpy/ndarraytypes.h"  // 导入 ndarraytypes.h 头文件
#include "npy_cpu_features.h"  // 导入 npy_cpu_features.h 头文件
#include "npy_cpu_dispatch.h"  // 导入 npy_cpu_dispatch.h 头文件
#include "numpy/npy_cpu.h"  // 导入 numpy CPU 头文件

#include "npy_static_data.h"  // 导入 npy_static_data.h 头文件
#include "npy_import.h"  // 导入 npy_import.h 头文件
#include <limits.h>  // 导入 limits.h 头文件

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())  // 定义宏，用于检查是否发生了错误转换并设置了异常

#ifdef NPY_ALLOW_THREADS
#define NPY_BEGIN_THREADS_NDITER(iter) \  // 如果允许多线程，定义宏以开始线程
        do { \
            if (!NpyIter_IterationNeedsAPI(iter)) { \  // 检查迭代器是否需要 API
                NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter)); \  // 根据迭代器大小设置线程阈值
            } \
        } while(0)
#else
#define NPY_BEGIN_THREADS_NDITER(iter)  // 如果不允许多线程，则为空宏
#endif

NPY_NO_EXPORT PyArray_Descr *  // 声明不导出的函数返回 PyArray_Descr 指针
PyArray_DTypeFromObjectStringDiscovery(
        PyObject *obj, PyArray_Descr *last_dtype, int string_type);  // 函数原型，从对象字符串中发现数据类型

/*
 * Recursively examines the object to determine an appropriate dtype
 * to use for converting to an ndarray.
 *
 * 'obj' is the object to be converted to an ndarray.
 *
 * 'maxdims' is the maximum recursion depth.
 *
 * 'out_dtype' should be either NULL or a minimal starting dtype when
 * the function is called. It is updated with the results of type
 * promotion. This dtype does not get updated when processing NA objects.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int  // 声明不导出的函数返回整型
PyArray_DTypeFromObject(PyObject *obj, int maxdims,
                        PyArray_Descr **out_dtype);  // 函数原型，从对象中确定适当的数据类型

/*
 * Returns NULL without setting an exception if no scalar is matched, a
 * new dtype reference otherwise.
 */
NPY_NO_EXPORT PyArray_Descr *  // 声明不导出的函数返回 PyArray_Descr 指针
_array_find_python_scalar_type(PyObject *op);  // 函数原型，查找 Python 标量类型

NPY_NO_EXPORT npy_bool  // 声明不导出的函数返回 npy_bool 类型
_IsWriteable(PyArrayObject *ap);  // 函数原型，检查数组是否可写

NPY_NO_EXPORT PyObject *  // 声明不导出的函数返回 PyObject 指针
convert_shape_to_string(npy_intp n, npy_intp const *vals, char *ending);  // 函数原型，将形状转换为字符串

/*
 * Sets ValueError with "matrices not aligned" message for np.dot and friends
 * when a.shape[i] should match b.shape[j], but doesn't.
 */
NPY_NO_EXPORT void  // 声明不导出的函数返回空
dot_alignment_error(PyArrayObject *a, int i, PyArrayObject *b, int j);  // 函数原型，设置 "matrices not aligned" 错误消息

/**
 * unpack tuple of PyDataType_FIELDS(dtype) (descr, offset, title[not-needed])
 *
 * @param "value" should be the tuple.
 *
 * @return "descr" will be set to the field's dtype
 * @return "offset" will be set to the field's offset
 *
 * returns -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int  // 声明不导出的函数返回整型
_unpack_field(PyObject *value, PyArray_Descr **descr, npy_intp *offset);  // 函数原型，解压 PyDataType_FIELDS 元组

/*
 * check whether arrays with datatype dtype might have object fields. This will
 * only happen for structured dtypes (which may have hidden objects even if the
 * HASOBJECT flag is false), object dtypes, or subarray dtypes whose base type
 * is either of these.
 */
NPY_NO_EXPORT int  // 声明不导出的函数返回整型
_may_have_objects(PyArray_Descr *dtype);  // 函数原型，检查数据类型是否可能包含对象字段

#endif  // NUMPY_CORE_SRC_MULTIARRAY_COMMON_H_
/*
 * Returns -1 and sets an exception if *index is an invalid index for
 * an array of size max_item, otherwise adjusts it in place to be
 * 0 <= *index < max_item, and returns 0.
 * 'axis' should be the array axis that is being indexed over, if known. If
 * unknown, use -1.
 * If _save is NULL it is assumed the GIL is taken
 * If _save is not NULL it is assumed the GIL is not taken and it
 * is acquired in the case of an error
 */
static inline int
check_and_adjust_index(npy_intp *index, npy_intp max_item, int axis,
                       PyThreadState * _save)
{
    /* Check that index is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*index < -max_item) || (*index >= max_item))) {
        NPY_END_THREADS;  // Release the GIL before raising an exception
        /* Try to be as clear as possible about what went wrong. */
        if (axis >= 0) {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for axis %d with size %"NPY_INTP_FMT,
                         *index, axis, max_item);  // Format error message for axis index
        } else {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for size %"NPY_INTP_FMT, *index, max_item);  // Format error message for size index
        }
        return -1;  // Return -1 indicating error
    }
    /* adjust negative indices */
    if (*index < 0) {
        *index += max_item;  // Adjust negative index to positive
    }
    return 0;  // Return 0 indicating success
}

/*
 * Returns -1 and sets an exception if *axis is an invalid axis for
 * an array of dimension ndim, otherwise adjusts it in place to be
 * 0 <= *axis < ndim, and returns 0.
 *
 * msg_prefix: borrowed reference, a string to prepend to the message
 */
static inline int
check_and_adjust_axis_msg(int *axis, int ndim, PyObject *msg_prefix)
{
    /* Check that axis is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*axis < -ndim) || (*axis >= ndim))) {
        /* Invoke the AxisError constructor */
        PyObject *exc = PyObject_CallFunction(
                npy_static_pydata.AxisError, "iiO", *axis, ndim,
                msg_prefix);  // Create AxisError exception with axis information
        if (exc == NULL) {
            return -1;  // Return -1 indicating error
        }
        PyErr_SetObject(npy_static_pydata.AxisError, exc);  // Set the created exception object
        Py_DECREF(exc);  // Decrement reference count of the exception object

        return -1;  // Return -1 indicating error
    }
    /* adjust negative indices */
    if (*axis < 0) {
        *axis += ndim;  // Adjust negative axis index to positive
    }
    return 0;  // Return 0 indicating success
}

static inline int
check_and_adjust_axis(int *axis, int ndim)
{
    return check_and_adjust_axis_msg(axis, ndim, Py_None);  // Call check_and_adjust_axis_msg with default message prefix
}

/* used for some alignment checks */
/*
 * GCC releases before GCC 4.9 had a bug in _Alignof.  See GCC bug 52023
 * <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52023>.
 * clang versions < 8.0.0 have the same bug.
 */
#if (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112 \
     || (defined __GNUC__ && __GNUC__ < 4 + (__GNUC_MINOR__ < 9) \
  && !defined __clang__) \
     || (defined __clang__ && __clang_major__ < 8))
# define NPY_ALIGNOF(type) offsetof(struct {char c; type v;}, v)  // Macro definition for alignment check
#else
# define NPY_ALIGNOF(type) _Alignof(type)  // Macro definition for alignment check
#endif
#endif
#define  NPY_ALIGNOF_UINT(type) npy_uint_alignment(sizeof(type))
/*
 * 禁用无害的编译器警告 "4116: unnamed type definition in
 * parentheses"，这是由 _ALIGN 宏引起的。
 */
#if defined(_MSC_VER)
#pragma warning(disable:4116)
#endif

/*
 * 如果指针对齐到 'alignment'，则返回 true
 */
static inline int
npy_is_aligned(const void * p, const npy_uintp alignment)
{
    /*
     * 假设 alignment 是 2 的幂，符合 C 标准要求。
     * 假设从指针转换为 uintp 后可以进行比特位 & 运算（不是 C 标准要求，但 glibc 中使用）。
     * 这个测试比直接取模更快。
     * 注意 alignment 值为 0 是允许的，并返回 False。
     */
    return ((npy_uintp)(p) & ((alignment) - 1)) == 0;
}

/* 获取相应的 "uint" 对齐，根据 itemsize，在复制代码中使用 */
static inline npy_uintp
npy_uint_alignment(int itemsize)
{
    npy_uintp alignment = 0; /* 返回值为 0 表示不对齐 */

    switch(itemsize){
        case 1:
            return 1;
        case 2:
            alignment = NPY_ALIGNOF(npy_uint16);
            break;
        case 4:
            alignment = NPY_ALIGNOF(npy_uint32);
            break;
        case 8:
            alignment = NPY_ALIGNOF(npy_uint64);
            break;
        case 16:
            /*
             * 16 字节类型使用 2 个 uint64 赋值进行复制。
             * 参见 lowlevel_strided_loops.c 中的跨步复制函数。
             */
            alignment = NPY_ALIGNOF(npy_uint64);
            break;
        default:
            break;
    }

    return alignment;
}

/*
 * 带有步长和反转参数的 memchr
 * 适用于小搜索，其中调用 libc 的 memchr 代价高昂。
 * 步长必须是大小的倍数。
 * 与 memchr 不同，如果未找到 needle，则返回末尾的一个步长。
 */
#ifdef __clang__
    /*
     * 下面的代码当前使用 !NPY_ALIGNMENT_REQUIRED，这应该是可以的，
     * 但会导致 clang sanitizer 发出警告。可以修改代码以避免这种“非对齐”访问，
     * 但应仔细检查性能变化。
     */
    __attribute__((no_sanitize("alignment")))
#endif
static inline char *
npy_memchr(char * haystack, char needle,
           npy_intp stride, npy_intp size, npy_intp * psubloopsize, int invert)
{
    char * p = haystack;
    npy_intp subloopsize = 0;

    if (!invert) {
        /*
         * 这通常是确定要处理的元素的路径，
         * 这里性能不是很重要。
         * 如果 0 字节靠近开始，memchr 的设置成本很高。
         */
        while (subloopsize < size && *p != needle) {
            subloopsize++;
            p += stride;
        }
    }
    else {
        /* 否则情况下，处理通常是跳过路径元素 */
        
        if (!NPY_ALIGNMENT_REQUIRED && needle == 0 && stride == 1) {
            /* 如果不需要对齐且查找元素为0且步长为1时 */
            
            /* 迭代直到最后一个4的倍数 */
            char * block_end = haystack + size - (size % sizeof(unsigned int));
            while (p < block_end) {
                unsigned int  v = *(unsigned int*)p;
                if (v != 0) {
                    break;
                }
                p += sizeof(unsigned int);
            }
            
            /* 处理剩余部分 */
            subloopsize = (p - haystack);
        }
        
        while (subloopsize < size && *p == needle) {
            subloopsize++;
            p += stride;
        }
    }

    *psubloopsize = subloopsize;

    return p;
/*
 * Simple helper to create a tuple from an array of items. The `make_null_none`
 * flag means that NULL entries are replaced with None, which is occasionally
 * useful.
 */
static inline PyObject *
PyArray_TupleFromItems(int n, PyObject *const *items, int make_null_none)
{
    // 创建一个包含 n 个元素的元组
    PyObject *tuple = PyTuple_New(n);
    if (tuple == NULL) {
        return NULL;
    }
    // 遍历 items 数组，将每个元素添加到 tuple 中
    for (int i = 0; i < n; i ++) {
        PyObject *tmp;
        // 如果 make_null_none 为真且 items[i] 为 NULL，则使用 Py_None 替代
        if (!make_null_none || items[i] != NULL) {
            tmp = items[i];
        }
        else {
            tmp = Py_None;
        }
        // 增加 tmp 的引用计数，并将其设置为 tuple 的第 i 个元素
        Py_INCREF(tmp);
        PyTuple_SET_ITEM(tuple, i, tmp);
    }
    return tuple;
}

/*
 * Returns 0 if the array has rank 0, -1 otherwise. Prints a deprecation
 * warning for arrays of _size_ 1.
 */
NPY_NO_EXPORT int
check_is_convertible_to_scalar(PyArrayObject *v);


#include "ucsnarrow.h"

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 *
 * If `out` is non-NULL, memory overlap is checked with ap1 and ap2, and an
 * updateifcopy temporary array may be returned. If `result` is non-NULL, the
 * output array to be returned (`out` if non-NULL and the newly allocated array
 * otherwise) is incref'd and put to *result.
 */
NPY_NO_EXPORT PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum, PyArrayObject **result);


/*
 * Used to indicate a broadcast axis, see also `npyiter_get_op_axis` in
 * `nditer_constr.c`.  This may be the preferred API for reduction axes
 * probably. So we should consider making this public either as a macro or
 * function (so that the way we flag the axis can be changed).
 */
#define NPY_ITER_REDUCTION_AXIS(axis) (axis + (1 << (NPY_BITSOF_INT - 2)))

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_COMMON_H_ */
```