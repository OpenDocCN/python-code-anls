# `.\numpy\numpy\_core\src\multiarray\ctors.c`

```py
/*
 * 定义常量，指定使用的 NumPy API 版本
 * 禁用已弃用的 NumPy API
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义标识符，指示这是多维数组模块
 * 定义标识符，指示这是通用数学函数模块
 */
#define _MULTIARRAYMODULE
#define _UMATHMODULE

/*
 * 清理 PY_SSIZE_T 类型的宏定义，确保只使用官方 API
 * 包含 Python 标准库头文件
 * 包含 NumPy 的数组对象头文件
 * 包含数组对象的核心头文件
 * 包含数组标量头文件
 * 包含 NumPy 的数学函数头文件
 * 包含 NumPy 的配置文件
 * 包含 NumPy 的 Python 兼容性头文件
 * 包含 NumPy 的 Ctypes 头文件
 * 包含 NumPy 的静态数据头文件
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_ctypes.h"

#include "npy_static_data.h"

#include "common.h"
#include "ctors.h"
#include "convert_datatype.h"
#include "descriptor.h"
#include "dtypemeta.h"
#include "refcount.h"  /* for PyArray_SetObjectsToNone */
#include "shape.h"
#include "npy_buffer.h"
#include "lowlevel_strided_loops.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "array_assign.h"
#include "mapping.h" /* for array_item_asarray */
#include "templ_common.h" /* for npy_mul_sizes_with_overflow */
#include "alloc.h"
#include <assert.h>

#include "get_attr_string.h"
#include "array_coercion.h"

#include "umathmodule.h"

/*
 * 定义全局静态变量，存储未能避免复制的错误消息
 */
NPY_NO_EXPORT const char *npy_no_copy_err_msg = (
        "Unable to avoid copy while creating an array as requested.\n"
        "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` "
        "to allow a copy when needed (no behavior change in NumPy 1.x).\n"
        "For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.");

/*
 * 从文件或字符串中读取数据的功能模块
 *
 * 尽可能使用相同的代码来处理文件和字符串，
 * 因此 fromstring 和 fromfile 的语义是相同的，特别是对于文本表示的处理。
 */

/*
 * 下一个元素解析和分隔符跳过的扫描功能
 * 这些函数返回：
 *   - 0 表示还有数据需要读取
 *   - -1 表示在字符串/文件末尾停止读取
 *   - -2 表示在未到达末尾前停止读取
 *
 * dtype 特定的解析函数可能会设置 Python 错误状态
 * （它们必须首先获取全局解释器锁 GIL）。
 */
typedef int (*next_element)(void **, void *, PyArray_Descr *, void *);
typedef int (*skip_separator)(void **, const char *, void *);

/*
 * 检查字符串是否完全读取
 * 如果 end 为 NULL，则检查是否以 '\0' 结尾
 * 否则，检查是否达到指定的结束位置
 */
static npy_bool
string_is_fully_read(char const* start, char const* end) {
    if (end == NULL) {
        return *start == '\0';  /* 使用空字符结尾 */
    }
    else {
        return start >= end;  /* 使用固定长度 */
    }
}

/*
 * 从字符串中读取下一个元素并解析
 * s 是要解析的字符串的起始地址
 * dptr 是目标数据的指针
 * dtype 是数组元素的描述符
 * end 是字符串的结束位置（可选）
 */
static int
fromstr_next_element(char **s, void *dptr, PyArray_Descr *dtype,
                     const char *end)
{
    char *e = *s;
    /*
     * 使用 PyDataType_GetArrFuncs 函数获取解析函数，
     * 并调用 fromstr 方法解析 s 指向的字符串到 dptr 指向的目标数据
     * e 指向解析后的字符串结尾位置
     */
    int r = PyDataType_GetArrFuncs(dtype)->fromstr(*s, dptr, &e, dtype);
    /*
     * fromstr 对基本数据类型返回 0；
     * 如果 s 没有改变，说明发生了错误或者已经达到末尾
     */
    if (*s == e || r < 0) {
        /* 没有读取到任何内容，可能是字符串末尾或者出现了错误 */
        if (string_is_fully_read(*s, end)) {
            return -1;  /* 字符串已经全部读取完毕 */
        }
        return -2;  /* 未到达字符串末尾就停止读取 */
    }
    *s = e;  /* 更新 s 的位置到解析后的字符串结尾 */
    # 检查条件：如果 `end` 不为 NULL 且指针 `s` 指向的值大于 `end`
    if (end != NULL && *s > end) {
        # 如果满足条件，停止迭代
        return -1;
    }
    # 如果条件不满足，则返回 0
    return 0;
/*
 * Static function to read the next element from a file pointer based on the given data type.
 * Uses the dtype's scanning function to read data into dptr.
 * Returns 0 if successful (1 item read), -1 if EOF reached, -2 if unable to read more due to an error.
 */
static int
fromfile_next_element(FILE **fp, void *dptr, PyArray_Descr *dtype,
                      void *NPY_UNUSED(stream_data))
{
    /* the NULL argument is for backwards-compatibility */
    // 使用 NULL 参数是为了向后兼容
    int r = PyDataType_GetArrFuncs(dtype)->scanfunc(*fp, dptr, NULL, dtype);
    // r 可能是 EOF 或者读取的项目数（0 或 1）
    if (r == 1) {
        return 0;
    }
    else if (r == EOF) {
        return -1;
    }
    else {
        /* unable to read more, but EOF not reached indicating an error. */
        // 无法继续读取更多，但未达到 EOF，表明存在错误
        return -2;
    }
}

/*
 * Remove multiple whitespace from the separator, and add a space to the
 * beginning and end. This simplifies the separator-skipping code below.
 */
static char *
swab_separator(const char *sep)
{
    int skip_space = 0;
    char *s, *start;

    s = start = malloc(strlen(sep)+3);
    if (s == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    /* add space to front if there isn't one */
    // 如果前面没有空格，向开头添加一个空格
    if (*sep != '\0' && !isspace(*sep)) {
        *s = ' '; s++;
    }
    while (*sep != '\0') {
        if (isspace(*sep)) {
            if (skip_space) {
                sep++;
            }
            else {
                *s = ' ';
                s++;
                sep++;
                skip_space = 1;
            }
        }
        else {
            *s = *sep;
            s++;
            sep++;
            skip_space = 0;
        }
    }
    /* add space to end if there isn't one */
    // 如果末尾没有空格，向末尾添加一个空格
    if (s != start && s[-1] == ' ') {
        *s = ' ';
        s++;
    }
    *s = '\0';
    return start;
}

/*
 * Assuming that the separator is the next bit in the string (file), skip it.
 *
 * Single spaces in the separator are matched to arbitrary-long sequences
 * of whitespace in the input. If the separator consists only of spaces,
 * it matches one or more whitespace characters.
 *
 * If we can't match the separator, return -2.
 * If we hit the end of the string (file), return -1.
 * Otherwise, return 0.
 */
static int
fromstr_skip_separator(char **s, const char *sep, const char *end)
{
    char *string = *s;
    int result = 0;

    while (1) {
        char c = *string;
        if (string_is_fully_read(string, end)) {
            result = -1;
            break;
        }
        else if (*sep == '\0') {
            if (string != *s) {
                /* matched separator */
                result = 0;
                break;
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;
                break;
            }
        }
        else if (*sep == ' ') {
            /* whitespace wildcard */
            if (!isspace(c)) {
                sep++;
                continue;
            }
        }
        else if (*sep != c) {
            result = -2;
            break;
        }
        else {
            sep++;
        }
        string++;
    }
    *s = string;
    return result;
}

static int
/*
 * Skip over a separator in a file stream pointed to by 'fp'. The separator can
 * be specified as a C string 'sep'. If the separator is found in the stream, the
 * function stops and returns 0. If the end of the file (EOF) is reached before
 * finding the separator, it returns -1. If the separator is a whitespace wildcard
 * that doesn't match, it returns -2.
 */
fromfile_skip_separator(FILE **fp, const char *sep, void *NPY_UNUSED(stream_data))
{
    int result = 0;             // Initialize the result indicator
    const char *sep_start = sep; // Pointer to the start of the separator string

    while (1) {
        int c = fgetc(*fp);     // Read a character from the file stream

        if (c == EOF) {          // If end of file is reached
            result = -1;         // Set result to indicate EOF
            break;               // Exit the loop
        }
        else if (*sep == '\0') { // If the separator pointer points to the null character
            ungetc(c, *fp);      // Push back the character read
            if (sep != sep_start) {
                /* matched separator */
                result = 0;      // Set result to indicate successful match
                break;           // Exit the loop
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;     // Set result to indicate mismatch
                break;           // Exit the loop
            }
        }
        else if (*sep == ' ') {  // If the separator is a whitespace wildcard
            /* whitespace wildcard */
            if (!isspace(c)) {   // If the character is not a whitespace
                sep++;           // Move to the next character in the separator
                sep_start++;     // Move the start pointer accordingly
                ungetc(c, *fp);  // Push back the character read
            }
            else if (sep == sep_start) {
                sep_start--;     // Adjust start pointer if separator starts with whitespace
            }
        }
        else if (*sep != c) {    // If the character doesn't match the separator
            ungetc(c, *fp);      // Push back the character read
            result = -2;         // Set result to indicate mismatch
            break;               // Exit the loop
        }
        else {
            sep++;               // Move to the next character in the separator
        }
    }
    return result;              // Return the final result
}

/*
 * Change a sub-array field to the base descriptor
 * and update the dimensions and strides
 * appropriately.  Dimensions and strides are added
 * to the end.
 *
 * Strides are only added if given (because data is given).
 */
static int
_update_descr_and_dimensions(PyArray_Descr **des, npy_intp *newdims,
                             npy_intp *newstrides, int oldnd)
{
    _PyArray_LegacyDescr *old;
    int newnd;
    int numnew;
    npy_intp *mydim;
    int i;
    int tuple;

    old = (_PyArray_LegacyDescr *)*des;  // Cast *des to legacy descriptor type
    *des = old->subarray->base;          // Set *des to the base descriptor of the sub-array

    mydim = newdims + oldnd;             // Initialize mydim pointer to newdims offset by oldnd
    tuple = PyTuple_Check(old->subarray->shape);  // Check if shape is a tuple
    if (tuple) {
        numnew = PyTuple_GET_SIZE(old->subarray->shape);  // Get size of tuple
    }
    else {
        numnew = 1;                      // Otherwise set numnew to 1
    }

    newnd = oldnd + numnew;              // Calculate new number of dimensions
    if (newnd > NPY_MAXDIMS) {           // Check if newnd exceeds maximum dimensions
        goto finish;                     // Jump to finish label if true
    }
    if (tuple) {
        for (i = 0; i < numnew; i++) {
            mydim[i] = (npy_intp) PyLong_AsLong(
                    PyTuple_GET_ITEM(old->subarray->shape, i));  // Convert tuple items to long integers
        }
    }
    else {
        mydim[0] = (npy_intp) PyLong_AsLong(old->subarray->shape);  // Convert shape to long integer
    }

    if (newstrides) {
        npy_intp tempsize;
        npy_intp *mystrides;

        mystrides = newstrides + oldnd;  // Initialize mystrides pointer to newstrides offset by oldnd
        /* Make new strides -- always C-contiguous */
        tempsize = (*des)->elsize;       // Get element size of descriptor
        for (i = numnew - 1; i >= 0; i--) {
            mystrides[i] = tempsize;     // Set stride for each dimension
            tempsize *= mydim[i] ? mydim[i] : 1;  // Update element size
        }
    }

 finish:
    Py_INCREF(*des);                    // Increment reference count of *des
    Py_DECREF(old);                     // Decrement reference count of old descriptor
    return newnd;                       // Return new number of dimensions
}

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize)
{
    npy_intp i;
    char *tout = dst;
    char *tin = src;

#define _COPY_N_SIZE(size) \
    for(i=0; i<N; i++) { \
        memcpy(tout, tin, size); \
        tin += instrides; \
        tout += outstrides; \
    } \
    return


# 循环复制数据块，N 次迭代
for(i=0; i<N; i++) { \
    # 使用 memcpy 函数复制大小为 size 的内存块，从 tin 到 tout
    memcpy(tout, tin, size); \
    # 更新 tin 指针，使其指向下一个输入数据块
    tin += instrides; \
    # 更新 tout 指针，使其指向下一个输出数据块
    tout += outstrides; \
} \
# 函数返回
return



    switch(elsize) {
    case 8:
        _COPY_N_SIZE(8);
    case 4:
        _COPY_N_SIZE(4);
    case 1:
        _COPY_N_SIZE(1);
    case 2:
        _COPY_N_SIZE(2);
    case 16:
        _COPY_N_SIZE(16);
    default:
        _COPY_N_SIZE(elsize);
    }


# 根据 elsize 变量的值选择不同大小的数据块复制操作
switch(elsize) {
case 8:
    # 调用 _COPY_N_SIZE 宏，复制大小为 8 字节的数据块
    _COPY_N_SIZE(8);
case 4:
    # 调用 _COPY_N_SIZE 宏，复制大小为 4 字节的数据块
    _COPY_N_SIZE(4);
case 1:
    # 调用 _COPY_N_SIZE 宏，复制大小为 1 字节的数据块
    _COPY_N_SIZE(1);
case 2:
    # 调用 _COPY_N_SIZE 宏，复制大小为 2 字节的数据块
    _COPY_N_SIZE(2);
case 16:
    # 调用 _COPY_N_SIZE 宏，复制大小为 16 字节的数据块
    _COPY_N_SIZE(16);
default:
    # 对于 elsize 的其它值，调用 _COPY_N_SIZE 宏，复制指定大小的数据块
    _COPY_N_SIZE(elsize);
}
#undef _COPY_N_SIZE
/*
 * Recursive helper to assign using a coercion cache. This function
 * must consume the cache depth first, just as the cache was originally
 * produced.
 */
NPY_NO_EXPORT int
PyArray_AssignFromCache_Recursive(
        PyArrayObject *self, const int ndim, coercion_cache_obj **cache)
{
    /* Consume first cache element by extracting information and freeing it */
    // 从缓存中取出第一个元素，增加其引用计数
    PyObject *obj = (*cache)->arr_or_sequence;
    Py_INCREF(obj);
    npy_bool sequence = (*cache)->sequence;
    int depth = (*cache)->depth;
    // 从缓存中移除当前元素并释放内存
    *cache = npy_unlink_coercion_cache(*cache);

    /* The element is either a sequence, or an array */
    // 判断当前元素是序列还是数组
    if (!sequence) {
        /* Straight forward array assignment */
        // 如果是数组，进行直接的数组赋值操作
        assert(PyArray_Check(obj));
        if (PyArray_CopyInto(self, (PyArrayObject *)obj) < 0) {
            goto fail;
        }
    }
    else {
        assert(depth != ndim);
        npy_intp length = PySequence_Length(obj);
        // 检查序列长度是否与数组的第一维度匹配
        if (length != PyArray_DIMS(self)[0]) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Inconsistent object during array creation? "
                    "Content of sequences changed (length inconsistent).");
            goto fail;
        }

        for (npy_intp i = 0; i < length; i++) {
            PyObject *value = PySequence_Fast_GET_ITEM(obj, i);

            if (ndim == depth + 1) {
                /*
                 * Straight forward assignment of elements.  Note that it is
                 * possible for such an element to be a 0-D array or array-like.
                 * `PyArray_Pack` supports arrays as well as we want: We
                 * support exact NumPy arrays, but at this point ignore others.
                 * (Please see the `PyArray_Pack` function comment if this
                 * rightly confuses you.)
                 */
                // 直接对元素进行赋值操作。注意，元素可能是0维数组或类似数组。
                char *item;
                item = (PyArray_BYTES(self) + i * PyArray_STRIDES(self)[0]);
                if (PyArray_Pack(PyArray_DESCR(self), item, value) < 0) {
                    goto fail;
                }
                /* If this was an array(-like) we still need to unlike int: */
                // 如果当前元素是数组（或类似数组），需要移除缓存中对应的转换对象
                if (*cache != NULL && (*cache)->converted_obj == value) {
                    *cache = npy_unlink_coercion_cache(*cache);
                }
            }
            else {
                // 否则，递归处理下一维度的数组赋值操作
                PyArrayObject *view;
                view = (PyArrayObject *)array_item_asarray(self, i);
                if (view == NULL) {
                    goto fail;
                }
                if (PyArray_AssignFromCache_Recursive(view, ndim, cache) < 0) {
                    Py_DECREF(view);
                    goto fail;
                }
                Py_DECREF(view);
            }
        }
    }
    // 释放当前元素的引用
    Py_DECREF(obj);
    return 0;

  fail:
    // 出现错误时，释放当前元素的引用并返回错误码
    Py_DECREF(obj);
    return -1;
}
/**
 * Fills an item based on a coercion cache object. It consumes the cache
 * object while doing so.
 *
 * @param self Array to fill.
 * @param cache coercion_cache_object, will be consumed. The cache must not
 *        contain a single array (must start with a sequence). The array case
 *        should be handled by `PyArray_FromArray()` before.
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignFromCache(PyArrayObject *self, coercion_cache_obj *cache) {
    int ndim = PyArray_NDIM(self);
    /*
     * Do not support ndim == 0 now with an array in the cache.
     * The ndim == 0 is special because np.array(np.array(0), dtype=object)
     * should unpack the inner array.
     * Since the single-array case is special, it is handled previously
     * in either case.
     */
    assert(cache->sequence);
    assert(ndim != 0);  /* guaranteed if cache contains a sequence */

    if (PyArray_AssignFromCache_Recursive(self, ndim, &cache) < 0) {
        /* free the remaining cache. */
        npy_free_coercion_cache(cache);
        return -1;
    }

    /*
     * Sanity check, this is the initial call, and when it returns, the
     * cache has to be fully consumed, otherwise something is wrong.
     * NOTE: May be nicer to put into a recursion helper.
     */
    if (cache != NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Inconsistent object during array creation? "
                "Content of sequences changed (cache not consumed).");
        npy_free_coercion_cache(cache);
        return -1;
    }
    return 0;
}



static void
raise_memory_error(int nd, npy_intp const *dims, PyArray_Descr *descr)
{
    PyObject *shape = PyArray_IntTupleFromIntp(nd, dims);
    if (shape == NULL) {
        goto fail;
    }

    /* produce an error object */
    PyObject *exc_value = PyTuple_Pack(2, shape, (PyObject *)descr);
    Py_DECREF(shape);
    if (exc_value == NULL){
        goto fail;
    }
    PyErr_SetObject(npy_static_pydata._ArrayMemoryError, exc_value);
    Py_DECREF(exc_value);
    return;

fail:
    /* we couldn't raise the formatted exception for some reason */
    PyErr_WriteUnraisable(NULL);
    PyErr_NoMemory();
}



/*
 * Generic new array creation routine.
 * Internal variant with calloc argument for PyArray_Zeros.
 *
 * steals a reference to descr. On failure or PyDataType_SUBARRAY(descr), descr will
 * be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr_int(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base, _NPY_CREATION_FLAGS cflags)
{
    PyArrayObject_fields *fa;
    npy_intp nbytes;

    if (descr == NULL) {
        return NULL;
    }
    if (nd > NPY_MAXDIMS || nd < 0) {
        PyErr_Format(PyExc_ValueError,
                "number of dimensions must be within [0, %d]", NPY_MAXDIMS);
        Py_DECREF(descr);
        return NULL;
    }
    /* 如果数据类型定义了一个 finalization 函数，则执行最终描述符的操作 */
    PyArrayDTypeMeta_FinalizeDescriptor *finalize =
            NPY_DT_SLOTS(NPY_DTYPE(descr))->finalize_descr;
    if (finalize != NULL && data == NULL) {
        Py_SETREF(descr, finalize(descr));
        if (descr == NULL) {
            return NULL;
        }
    }

    /* 计算描述符元素的字节大小 */
    nbytes = descr->elsize;
    
    /*
     * 除非显式禁止，否则在某些情况下我们会替换数据类型。
     * 主要情况是我们不会创建带有子数组数据类型的数组
     * （除非是在请求时的内部使用）。大多数情况下也不会创建 S0/U0 数组
     * （除非 data == NULL，这可能是一个视图，在这种情况下增加数据类型可能是错误的）。
     */
    if (!(cflags & _NPY_ARRAY_ENSURE_DTYPE_IDENTITY)) {
        if (PyDataType_SUBARRAY(descr)) {
            PyObject *ret;
            npy_intp newdims[2*NPY_MAXDIMS];
            npy_intp *newstrides = NULL;
            memcpy(newdims, dims, nd*sizeof(npy_intp));
            if (strides) {
                newstrides = newdims + NPY_MAXDIMS;
                memcpy(newstrides, strides, nd*sizeof(npy_intp));
            }
            nd = _update_descr_and_dimensions(&descr, newdims,
                                              newstrides, nd);
            ret = PyArray_NewFromDescr_int(
                    subtype, descr,
                    nd, newdims, newstrides, data,
                    flags, obj, base, cflags);
            return ret;
        }

        /* 检查数据类型的元素大小 */
        if (PyDataType_ISUNSIZED(descr)) {
            if (!PyDataType_ISFLEXIBLE(descr) &&
                NPY_DT_is_legacy(NPY_DTYPE(descr))) {
                PyErr_SetString(PyExc_TypeError, "Empty data-type");
                Py_DECREF(descr);
                return NULL;
            }
            else if (PyDataType_ISSTRING(descr)
                        && !(cflags & _NPY_ARRAY_ALLOW_EMPTY_STRING)
                        && data == NULL) {
                PyArray_DESCR_REPLACE(descr);
                if (descr == NULL) {
                    return NULL;
                }
                if (descr->type_num == NPY_STRING) {
                    nbytes = descr->elsize = 1;
                }
                else {
                    nbytes = descr->elsize = sizeof(npy_ucs4);
                }
            }
        }
    }

    /* 分配 subtype 对象的内存空间 */
    fa = (PyArrayObject_fields *) subtype->tp_alloc(subtype, 0);
    if (fa == NULL) {
        Py_DECREF(descr);
        return NULL;
    }
    fa->_buffer_info = NULL;
    fa->nd = nd;
    fa->dimensions = NULL;
    fa->data = NULL;
    fa->mem_handler = NULL;

    /* 如果 data 为空，则设置默认标志 */
    if (data == NULL) {
        fa->flags = NPY_ARRAY_DEFAULT;
        if (flags) {
            fa->flags |= NPY_ARRAY_F_CONTIGUOUS;
            if (nd > 1) {
                fa->flags &= ~NPY_ARRAY_C_CONTIGUOUS;
            }
            flags = NPY_ARRAY_F_CONTIGUOUS;
        }
    }
    else {
        // 如果条件不成立，清除写回复制标志位
        fa->flags = (flags & ~NPY_ARRAY_WRITEBACKIFCOPY);
    }
    // 将描述符赋值给数组结构体的描述符字段
    fa->descr = descr;
    // 将数组的基础对象指针设为 NULL
    fa->base = (PyObject *)NULL;
    // 将弱引用列表指针设为 NULL
    fa->weakreflist = (PyObject *)NULL;

    /* 以下为零填充逻辑所需，这里定义和初始化，
       以便清理逻辑可以放在失败块中 */
    NPY_traverse_info fill_zero_info;
    NPY_traverse_info_init(&fill_zero_info);

    if (nd > 0) {
        // 分配缓存维度数组的内存
        fa->dimensions = npy_alloc_cache_dim(2 * nd);
        if (fa->dimensions == NULL) {
            // 分配内存失败，抛出内存错误异常
            PyErr_NoMemory();
            goto fail;
        }
        // 计算步幅数组的地址
        fa->strides = fa->dimensions + nd;

        /*
         * 复制维度，检查它们，并计算总数组大小 `nbytes`
         */
        int is_zero = 0;
        for (int i = 0; i < nd; i++) {
            // 将维度值赋给数组结构体中的维度数组
            fa->dimensions[i] = dims[i];

            if (fa->dimensions[i] == 0) {
                /*
                 * 继续计算最大大小，"仿佛"这个维度是1，
                 * 以便正确地触发溢出错误
                 */
                is_zero = 1;
                continue;
            }

            if (fa->dimensions[i] < 0) {
                // 如果维度值为负数，设置值错误异常并跳转到失败块
                PyErr_SetString(PyExc_ValueError,
                        "negative dimensions are not allowed");
                goto fail;
            }

            /*
             * 在将维度值相乘以获取数组的总大小时，需要注意避免整数溢出问题。
             */
            if (npy_mul_sizes_with_overflow(&nbytes, nbytes, fa->dimensions[i])) {
                PyErr_SetString(PyExc_ValueError,
                        "array is too big; `arr.size * arr.dtype.itemsize` "
                        "is larger than the maximum possible size.");
                goto fail;
            }
        }
        if (is_zero) {
            // 如果存在维度为零的情况，总大小设为0
            nbytes = 0;
        }

        /* 填充步幅（或者如果已经传入则复制它们） */
        if (strides == NULL) {
            /* 填充步幅并设置连续性标志位 */
            _array_fill_strides(fa->strides, dims, nd, descr->elsize,
                                flags, &(fa->flags));
        }
        else {
            /* 使用者提供了步幅（使用者需保证正确性） */
            for (int i = 0; i < nd; i++) {
                // 将传入的步幅复制到数组结构体中的步幅数组
                fa->strides[i] = strides[i];
            }
            /* 由于步幅已传入，需要更新连续性标志 */
            PyArray_UpdateFlags((PyArrayObject *)fa,
                    NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
        }
    }
    else {
        // 如果没有维度，将维度和步幅设为 NULL，并设置连续性标志
        fa->dimensions = NULL;
        fa->strides = NULL;
        fa->flags |= NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS;
    }
    if (data == NULL) {
        /* 如果 data 为空指针，则执行以下操作 */

        /* 定义用于存储数组方法的标志 */
        NPY_ARRAYMETHOD_FLAGS zero_flags;

        /* 获取填充零值循环的函数指针 */
        PyArrayMethod_GetTraverseLoop *get_fill_zero_loop =
            NPY_DT_SLOTS(NPY_DTYPE(descr))->get_fill_zero_loop;

        /* 如果存在填充零值循环的函数 */
        if (get_fill_zero_loop != NULL) {
            /* 调用填充零值循环函数，填充 fill_zero_info 结构体 */
            if (get_fill_zero_loop(
                    NULL, descr, 1, descr->elsize, &(fill_zero_info.func),
                    &(fill_zero_info.auxdata), &zero_flags) < 0) {
                goto fail;  /* 如果出错，则跳转到 fail 标签处 */
            }
        }

        /*
         * 如果 dtype 的 NPY_NEEDS_INIT 标志被设置，或者 cflags 的 _NPY_ARRAY_ZEROED
         * 标志被设置且填充零值函数为空，使用 calloc 安全地分配一个零值填充的数组。
         * 否则，根据条件使用 malloc 分配数组。
         */
        int use_calloc = (
                PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT) ||
                ((cflags & _NPY_ARRAY_ZEROED) && (fill_zero_info.func == NULL)));

        /* 存储当前的内存处理器到 fa->mem_handler */
        fa->mem_handler = PyDataMem_GetHandler();

        /* 如果内存处理器获取失败，则跳转到 fail 标签处 */
        if (fa->mem_handler == NULL) {
            goto fail;
        }

        /*
         * 如果 nbytes 为 0，则设置为 1，同时确保所有步幅都为 0。
         * 这样即使是零空间数组（例如 shape=(0,)），也能正确暴露缓冲区（a.data）。
         */
        if (nbytes == 0) {
            nbytes = 1;
            for (int i = 0; i < nd; i++) {
                fa->strides[i] = 0;
            }
        }

        /* 根据 use_calloc 的值选择使用 calloc 还是 malloc 分配内存 */
        if (use_calloc) {
            data = PyDataMem_UserNEW_ZEROED(nbytes, 1, fa->mem_handler);
        }
        else {
            data = PyDataMem_UserNEW(nbytes, fa->mem_handler);
        }

        /* 如果内存分配失败，则引发内存错误并跳转到 fail 标签处 */
        if (data == NULL) {
            raise_memory_error(fa->nd, fa->dimensions, descr);
            goto fail;
        }

        /*
         * 如果数组需要特定 dtype 的零值填充逻辑，则执行该逻辑。
         */
        if (NPY_UNLIKELY((cflags & _NPY_ARRAY_ZEROED)
                         && (fill_zero_info.func != NULL))) {
            npy_intp size = PyArray_MultiplyList(fa->dimensions, fa->nd);
            if (fill_zero_info.func(
                    NULL, descr, data, size, descr->elsize,
                    fill_zero_info.auxdata) < 0) {
                goto fail;  /* 如果填充逻辑执行失败，则跳转到 fail 标签处 */
            }
        }

        /* 设置数组对象的 NPY_ARRAY_OWNDATA 标志，表示该数组拥有数据 */
        fa->flags |= NPY_ARRAY_OWNDATA;
    }
    else {
        /* 如果 data 不为空指针，则执行以下操作 */

        /* 在这种情况下，不应该调用内存处理器 */
        fa->mem_handler = NULL;

        /* 如果数据是由外部传入的，对象不会拥有该数据 */
        fa->flags &= ~NPY_ARRAY_OWNDATA;
    }

    /* 将 data 指针赋值给数组对象的数据成员 */
    fa->data = data;
    /*
     * 总是更新对齐标志。对于非所有者数据或输入步长可能未对齐的情况，
     * 这一步很重要。在某些平台（如debian sparc）上，malloc函数不提供
     * 足够的对齐来支持long double类型。
     */
    PyArray_UpdateFlags((PyArrayObject *)fa, NPY_ARRAY_ALIGNED);

    /* 
     * 设置基础对象。在这里设置非常重要，这样下面的__array_finalize__方法可以接收到它
     */
    if (base != NULL) {
        Py_INCREF(base);
        if (PyArray_SetBaseObject((PyArrayObject *)fa, base) < 0) {
            goto fail;
        }
    }

    /*
     * 调用__array_finalize__方法（如果请求了子类型）。如果obj为NULL，
     * 则使用Py_None作为Python回调函数。
     * 为了提高速度，我们跳过如果__array_finalize__是从ndarray继承而来
     * （因为该函数什么也不做），或者出于向后兼容性考虑，如果它是None。
     */
    if (subtype != &PyArray_Type) {
        PyObject *res, *func;
        func = PyObject_GetAttr((PyObject *)subtype, npy_interned_str.array_finalize);
        if (func == NULL) {
            goto fail;
        }
        else if (func == npy_static_pydata.ndarray_array_finalize) {
            Py_DECREF(func);
        }
        else if (func == Py_None) {
            Py_DECREF(func);
            /*
             * 2022-01-08, NumPy 1.23; 当停止使用期结束后，移除这段代码，
             * 以便产生"NoneType object is not callable" TypeError。
             */
            if (DEPRECATE(
                    "Setting __array_finalize__ = None to indicate no finalization"
                    "should be done is deprecated.  Instead, just inherit from "
                    "ndarray or, if that is not possible, explicitly set to "
                    "ndarray.__array_function__; this will raise a TypeError "
                    "in the future. (Deprecated since NumPy 1.23)") < 0) {
                goto fail;
            }
        }
        else {
            if (PyCapsule_CheckExact(func)) {
                /* 这里存储了一个C函数 */
                PyArray_FinalizeFunc *cfunc;
                cfunc = PyCapsule_GetPointer(func, NULL);
                Py_DECREF(func);
                if (cfunc == NULL) {
                    goto fail;
                }
                if (cfunc((PyArrayObject *)fa, obj) < 0) {
                    goto fail;
                }
            }
            else {
                if (obj == NULL) {
                    obj = Py_None;
                }
                res = PyObject_CallFunctionObjArgs(func, (PyObject *)fa, obj, NULL);
                Py_DECREF(func);
                if (res == NULL) {
                    goto fail;
                }
                else {
                    Py_DECREF(res);
                }
            }
        }
    }
    NPY_traverse_info_xfree(&fill_zero_info);
    return (PyObject *)fa;

 fail:
    NPY_traverse_info_xfree(&fill_zero_info);
    Py_XDECREF(fa->mem_handler);
    Py_DECREF(fa);
    # 返回空值(NULL)，通常表示没有有效的返回结果
    return NULL;
/*NUMPY_API
 * Generic new array creation routine.
 *
 * steals a reference to descr. On failure or when PyDataType_SUBARRAY(dtype) is
 * true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(
        PyTypeObject *subtype, PyArray_Descr *descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj)
{
    // 检查 subtype 是否为空，若为空则设置错误信息并返回空
    if (subtype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "subtype is NULL in PyArray_NewFromDescr");
        return NULL;
    }

    // 检查 descr 是否为空，若为空则设置错误信息并返回空
    if (descr == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "descr is NULL in PyArray_NewFromDescr");
        return NULL;
    }

    // 调用 PyArray_NewFromDescrAndBase 函数创建新的数组对象
    return PyArray_NewFromDescrAndBase(
            subtype, descr,
            nd, dims, strides, data,
            flags, obj, NULL);
}

/*
 * Sets the base object using PyArray_SetBaseObject
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescrAndBase(
        PyTypeObject *subtype, PyArray_Descr *descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base)
{
    // 调用 PyArray_NewFromDescr_int 函数创建新的数组对象
    return PyArray_NewFromDescr_int(subtype, descr, nd,
                                    dims, strides, data,
                                    flags, obj, base, 0);
}

/*
 * Creates a new array with the same shape as the provided one,
 * with possible memory layout order, data type and shape changes.
 *
 * prototype - The array the new one should be like.
 * order     - NPY_CORDER - C-contiguous result.
 *             NPY_FORTRANORDER - Fortran-contiguous result.
 *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
 *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
 * descr     - If not NULL, overrides the data type of the result.
 * dtype     - If not NULL and if descr is NULL, overrides the data type
               of the result, so long as dtype is non-parameteric
 * ndim      - If not -1, overrides the shape of the result.
 * dims      - If ndim is not -1, overrides the shape of the result.
 * subok     - If 1, use the prototype's array subtype, otherwise
 *             always create a base-class array.
 *
 * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
 * PyDataType_SUBARRAY(dtype) is true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArrayWithShape(PyArrayObject *prototype, NPY_ORDER order,
                              PyArray_Descr *descr, PyArray_DTypeMeta *dtype, int ndim,
                              npy_intp const *dims, int subok)
{
    PyObject *ret = NULL;

    // 如果 ndim 为 -1，则从原型数组获取维度和形状
    if (ndim == -1) {
        ndim = PyArray_NDIM(prototype);
        dims = PyArray_DIMS(prototype);
    }
    // 如果 order 是 NPY_KEEPORDER 且 ndim 与原型数组的维度不同，则将 order 设置为 NPY_CORDER
    else if (order == NPY_KEEPORDER && (ndim != PyArray_NDIM(prototype))) {
        order = NPY_CORDER;
    }
    if (descr == NULL && dtype == NULL) {
        /* 如果没有指定数据类型，使用原型对象的数据类型 */
        descr = PyArray_DESCR(prototype);
        // 增加数据类型的引用计数，确保其不被销毁
        Py_INCREF(descr);
    }
    else if (descr == NULL) {
        // 从指定的数据类型推断描述符
        descr = _infer_descr_from_dtype(dtype);
        // 如果推断失败，返回空指针
        if (descr == NULL) {
            return NULL;
        }
    }

    /* 处理 ANYORDER 和简单的 KEEPORDER 情况 */
    switch (order) {
        case NPY_ANYORDER:
            // 如果原型对象是 Fortran 连续的，使用 Fortran 排序；否则使用 C 排序
            order = PyArray_ISFORTRAN(prototype) ?
                                    NPY_FORTRANORDER : NPY_CORDER;
            break;
        case NPY_KEEPORDER:
            // 如果原型对象是 C 连续的或者维度小于等于 1，使用 C 排序
            if (PyArray_IS_C_CONTIGUOUS(prototype) || ndim <= 1) {
                order = NPY_CORDER;
                break;
            }
            // 如果原型对象是 Fortran 连续的，使用 Fortran 排序
            else if (PyArray_IS_F_CONTIGUOUS(prototype)) {
                order = NPY_FORTRANORDER;
                break;
            }
            break;
        default:
            break;
    }

    /* 如果不是 KEEPORDER，这很简单 */
    if (order != NPY_KEEPORDER) {
        // 根据描述符创建新的数组对象
        ret = PyArray_NewFromDescr(subok ? Py_TYPE(prototype) : &PyArray_Type,
                                        descr,
                                        ndim,
                                        dims,
                                        NULL,
                                        NULL,
                                        order,
                                        subok ? (PyObject *)prototype : NULL);
    }
    /* KEEPORDER 需要对步长进行分析 */
    else {
        npy_intp strides[NPY_MAXDIMS], stride;
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        int idim;

        // 创建已排序的步长排列
        PyArray_CreateSortedStridePerm(ndim,
                                        PyArray_STRIDES(prototype),
                                        strideperm);

        /* 构建新的步长 */
        stride = descr->elsize;
        // 如果步长为 0 并且描述符为字符串类型（如 dtype=str 或 dtype=bytes）
        if (stride == 0 && PyDataType_ISSTRING(descr)) {
            /* 对于 dtype=str 或 dtype=bytes 的特殊情况 */
            if (descr->type_num == NPY_STRING) {
                /* dtype 是 bytes */
                stride = 1;
            }
            else {
                /* dtype 是 str（type_num 是 NPY_UNICODE）*/
                stride = 4;
            }
        }
        for (idim = ndim-1; idim >= 0; --idim) {
            npy_intp i_perm = strideperm[idim].perm;
            strides[i_perm] = stride;
            stride *= dims[i_perm];
        }

        /* 最后，分配数组对象 */
        ret = PyArray_NewFromDescr(subok ? Py_TYPE(prototype) : &PyArray_Type,
                                        descr,
                                        ndim,
                                        dims,
                                        strides,
                                        NULL,
                                        0,
                                        subok ? (PyObject *)prototype : NULL);
    }
    if (ret == NULL) {
        return NULL;
    }
    /* 
       检查返回的数组对象的数据类型描述符是否需要引用计数检查，
       如果需要，将数组对象中的所有元素设置为None（即清空数组对象的内容）。
       这段逻辑被`empty`、`empty_like`和`ndarray.__new__`共享使用。
    */
    if (PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)ret))) {
        // 如果设置元素为None失败，则释放数组对象并返回NULL
        if (PyArray_SetObjectsToNone((PyArrayObject *)ret) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    // 返回处理后的数组对象
    return ret;
/*NUMPY_API
 * Creates a new array with the same shape as the provided one,
 * with possible memory layout order and data type changes.
 *
 * prototype - The array the new one should be like.
 * order     - NPY_CORDER - C-contiguous result.
 *             NPY_FORTRANORDER - Fortran-contiguous result.
 *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
 *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
 * dtype     - If not NULL, overrides the data type of the result.
 * subok     - If 1, use the prototype's array subtype, otherwise
 *             always create a base-class array.
 *
 * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
 * PyDataType_SUBARRAY(dtype) is true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArray(PyArrayObject *prototype, NPY_ORDER order,
                     PyArray_Descr *dtype, int subok)
{
    // 检查传入的 prototype 是否为 NULL
    if (prototype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "prototype is NULL in PyArray_NewLikeArray");
        return NULL;
    }
    // 调用 PyArray_NewLikeArrayWithShape 函数创建一个新数组，传递给它参数
    return PyArray_NewLikeArrayWithShape(prototype, order, dtype, NULL, -1, NULL, subok);
}

/*NUMPY_API
 * Generic new array creation routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_New(
        PyTypeObject *subtype, int nd, npy_intp const *dims, int type_num,
        npy_intp const *strides, void *data, int itemsize, int flags,
        PyObject *obj)
{
    PyArray_Descr *descr;
    PyObject *new;

    // 检查 subtype 是否为 NULL
    if (subtype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "subtype is NULL in PyArray_New");
        return NULL;
    }

    // 根据 type_num 获取数组描述符
    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    // 如果描述符为未定义大小的数据类型，需要确保 itemsize 大于 0
    if (PyDataType_ISUNSIZED(descr)) {
        if (itemsize < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "data type must provide an itemsize");
            Py_DECREF(descr);
            return NULL;
        }
        // 替换描述符的大小
        PyArray_DESCR_REPLACE(descr);
        if (descr == NULL) {
            return NULL;
        }
        descr->elsize = itemsize;
    }
    // 使用描述符创建新的数组对象
    new = PyArray_NewFromDescr(subtype, descr, nd, dims, strides,
                               data, flags, obj);
    return new;
}


NPY_NO_EXPORT PyArray_Descr *
_dtype_from_buffer_3118(PyObject *memoryview)
{
    PyArray_Descr *descr;
    Py_buffer *view = PyMemoryView_GET_BUFFER(memoryview);
    // 如果视图的格式不为空，根据格式获取描述符
    if (view->format != NULL) {
        descr = _descriptor_from_pep3118_format(view->format);
        if (descr == NULL) {
            return NULL;
        }
    }
    else {
        /* 如果没有指定格式，假定为字节数组
         * TODO: 这里更合理的是使用 void 类型，因为它不需要空字符结尾。
         */
        // 创建一个新的描述符，类型为 NPY_STRING
        descr = PyArray_DescrNewFromType(NPY_STRING);
        if (descr == NULL) {
            return NULL;
        }
        // 设置描述符的元素大小
        descr->elsize = view->itemsize;
    }
    return descr;
}


NPY_NO_EXPORT PyObject *
/* 定义函数 _array_from_buffer_3118，接受一个 Python 内存视图对象作为参数 */
_array_from_buffer_3118(PyObject *memoryview)
{
    /* 声明 Py_buffer 结构体指针 view，用于存储内存视图信息 */
    Py_buffer *view;
    /* 声明 PyArray_Descr 结构体指针 descr，并初始化为 NULL */
    PyArray_Descr *descr = NULL;
    /* 声明 PyObject 指针 r，并初始化为 NULL */
    PyObject *r = NULL;
    /* 声明整型变量 nd 和 flags */
    int nd, flags;
    /* 声明 Py_ssize_t 类型的变量 d */
    Py_ssize_t d;
    /* 声明大小为 NPY_MAXDIMS 的整型数组 shape 和 strides */
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];

    /* 从 memoryview 中获取 Py_buffer 结构体视图，并赋值给 view */
    view = PyMemoryView_GET_BUFFER(memoryview);

    /* 如果视图的 suboffsets 不为 NULL，抛出异常并返回 NULL */
    if (view->suboffsets != NULL) {
        PyErr_SetString(PyExc_BufferError,
                "NumPy currently does not support importing buffers which "
                "include suboffsets as they are not compatible with the NumPy"
                "memory layout without a copy.  Consider copying the original "
                "before trying to convert it to a NumPy array.");
        return NULL;
    }

    /* 获取视图的维度赋值给 nd */
    nd = view->ndim;
    /* 调用 _dtype_from_buffer_3118 函数获取数据类型描述符，并赋值给 descr */
    descr = _dtype_from_buffer_3118(memoryview);

    /* 如果获取的描述符为 NULL，则返回 NULL */
    if (descr == NULL) {
        return NULL;
    }

    /* 执行完函数功能的初步检查 */
    if (descr->elsize != view->itemsize) {
        /* 如果视图对象的每个元素大小与描述符的每个元素大小不匹配，则执行以下操作 */

        /* Ctypes has bugs in its PEP3118 implementation, which we need to
         * work around.
         *
         * ctypes 在其 PEP3118 实现中存在缺陷，我们需要进行以下的处理方式。
         *
         * bpo-10746
         * bpo-32780
         * bpo-32782
         *
         * 注意，即使上述问题在主要版本中得到修复，我们也必须放弃早期的修补版本才能真正利用这些修复。
         */
        if (!npy_ctypes_check(Py_TYPE(view->obj))) {
            /* 如果视图对象的类型不符合 PEP3118 的规范 */

            PyErr_Format(
                    PyExc_RuntimeError,
                   "Item size %zd for PEP 3118 buffer format "
                    "string %s does not match the dtype %c item size %d.",
                    view->itemsize, view->format, descr->type,
                    descr->elsize);
            Py_DECREF(descr);
            return NULL;
        }

        if (PyErr_Warn(
                    PyExc_RuntimeWarning,
                    "A builtin ctypes object gave a PEP3118 format "
                    "string that does not match its itemsize, so a "
                    "best-guess will be made of the data type. "
                    "Newer versions of python may behave correctly.") < 0) {
            Py_DECREF(descr);
            return NULL;
        }

        /* Thankfully, np.dtype(ctypes_type) works in most cases.
         * For an array input, this produces a dtype containing all the
         * dimensions, so the array is now 0d.
         */
        /* 幸运的是，在大多数情况下，np.dtype(ctypes_type)是有效的。
         * 对于数组输入，这将产生一个包含所有维度的 dtype，因此数组现在是 0 维。
         */
        nd = 0;
        Py_DECREF(descr);
        descr = (PyArray_Descr *)PyObject_CallFunctionObjArgs(
                (PyObject *)&PyArrayDescr_Type, Py_TYPE(view->obj), NULL);
        if (descr == NULL) {
            return NULL;
        }
        if (descr->elsize != view->len) {
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "For the given ctypes object, neither the item size "
                    "computed from the PEP 3118 buffer format nor from "
                    "converting the type to a np.dtype matched the actual "
                    "size. This is a bug both in python and numpy");
            Py_DECREF(descr);
            return NULL;
        }
    }

    if (view->shape != NULL) {
        /* 如果视图对象具有形状信息 */

        int k;
        if (nd > NPY_MAXDIMS || nd < 0) {
            PyErr_Format(PyExc_RuntimeError,
                "PEP3118 dimensions do not satisfy 0 <= ndim <= NPY_MAXDIMS");
            goto fail;
        }
        for (k = 0; k < nd; ++k) {
            shape[k] = view->shape[k];
        }
        if (view->strides != NULL) {
            for (k = 0; k < nd; ++k) {
                strides[k] = view->strides[k];
            }
        }
        else {
            d = view->len;
            for (k = 0; k < nd; ++k) {
                if (view->shape[k] != 0) {
                    d /= view->shape[k];
                }
                strides[k] = d;
            }
        }
    }
    else {
        // 如果视图的维度 nd 大于 1
        if (nd == 1) {
            // 计算视图的第一个维度的大小
            shape[0] = view->len / view->itemsize;
            // 设置第一个维度的步长为每个元素的大小
            strides[0] = view->itemsize;
        }
        // 如果视图的维度 nd 大于 1
        else if (nd > 1) {
            // 抛出运行时错误，指示从 PEP 3118 缓冲区格式计算的维度大于 1，但形状为空
            PyErr_SetString(PyExc_RuntimeError,
                           "ndim computed from the PEP 3118 buffer format "
                           "is greater than 1, but shape is NULL.");
            // 转到错误处理标签
            goto fail;
        }
    }

    // 根据视图的只读属性设置数组标志
    flags = NPY_ARRAY_BEHAVED & (view->readonly ? ~NPY_ARRAY_WRITEABLE : ~0);
    // 创建一个新的 NumPy 数组对象
    r = PyArray_NewFromDescrAndBase(
            &PyArray_Type, descr,    // 使用指定的类型和描述符创建数组
            nd, shape, strides, view->buf,  // 数组的维度、形状、步长和数据缓冲区
            flags, NULL, memoryview);  // 数组的标志、基础对象和内存视图
    // 返回创建的数组对象
    return r;
fail:
    Py_XDECREF(r);
    Py_XDECREF(descr);
    // 返回 NULL，表示函数执行失败
    return NULL;
}

/**
 * Attempts to extract an array from an array-like object.
 *
 * array-like is defined as either
 *
 * * an object implementing the PEP 3118 buffer interface;
 * * an object with __array_struct__ or __array_interface__ attributes;
 * * an object with an __array__ function.
 *
 * @param op The object to convert to an array
 * @param requested_type a requested dtype instance, may be NULL; The result
 *                       DType may be used, but is not enforced.
 * @param writeable whether the result must be writeable.
 * @param context Unused parameter, must be NULL (should be removed later).
 * @param copy Specifies the copy behavior.
 * @param was_copied_by__array__ Set to 1 if it can be assumed that a copy
 *        was made by implementor.
 *
 * @returns The array object, Py_NotImplemented if op is not array-like,
 *          or NULL with an error set. (A new reference to Py_NotImplemented
 *          is returned.)
 */
NPY_NO_EXPORT PyObject *
_array_from_array_like(PyObject *op,
        PyArray_Descr *requested_dtype, npy_bool writeable, PyObject *context,
        int copy, int *was_copied_by__array__) {
    PyObject* tmp;

    /*
     * If op supports the PEP 3118 buffer interface.
     * We skip bytes and unicode since they are considered scalars. Unicode
     * would fail but bytes would be incorrectly converted to a uint8 array.
     */
    // 检查对象是否支持 PEP 3118 缓冲区接口
    if (PyObject_CheckBuffer(op) && !PyBytes_Check(op) && !PyUnicode_Check(op)) {
        // 将对象转换为内存视图
        PyObject *memoryview = PyMemoryView_FromObject(op);
        if (memoryview == NULL) {
            /* TODO: Should probably not blanket ignore errors. */
            // 如果转换失败，清除错误状态
            PyErr_Clear();
        }
        else {
            // 从内存视图中创建数组
            tmp = _array_from_buffer_3118(memoryview);
            Py_DECREF(memoryview);
            if (tmp == NULL) {
                return NULL;
            }

            // 如果需要可写性，并且数组对象不可写，则返回 NULL
            if (writeable
                && PyArray_FailUnlessWriteable(
                        (PyArrayObject *)tmp, "PEP 3118 buffer") < 0) {
                Py_DECREF(tmp);
                return NULL;
            }

            return tmp;
        }
    }

    /*
     * If op supports the __array_struct__ or __array_interface__ interface.
     */
    // 尝试使用 __array_struct__ 或 __array_interface__ 接口创建数组
    tmp = PyArray_FromStructInterface(op);
    if (tmp == NULL) {
        return NULL;
    }
    if (tmp == Py_NotImplemented) {
        /* Until the return, NotImplemented is always a borrowed reference*/
        // 如果返回 Py_NotImplemented，尝试使用 __array__ 接口创建数组
        tmp = PyArray_FromInterface(op);
        if (tmp == NULL) {
            return NULL;
        }
    }

    // 如果仍然返回 Py_NotImplemented，尝试从 op 的数组属性创建数组
    if (tmp == Py_NotImplemented) {
        tmp = PyArray_FromArrayAttr_int(
                op, requested_dtype, copy, was_copied_by__array__);
        if (tmp == NULL) {
            return NULL;
        }
    }
    # 检查 tmp 是否不等于 Py_NotImplemented
    if (tmp != Py_NotImplemented) {
        # 如果 writeable 为真，并且对于数组接口对象 tmp，确保其可写性
        if (writeable &&
                PyArray_FailUnlessWriteable((PyArrayObject *)tmp,
                        "array interface object") < 0) {
            # 如果检查可写性失败，释放 tmp 并返回 NULL
            Py_DECREF(tmp);
            return NULL;
        }
        # 返回 tmp
        return tmp;
    }

    /* 直到这里，Py_NotImplemented 被借用 */
    # 增加 Py_NotImplemented 的引用计数
    Py_INCREF(Py_NotImplemented);
    # 返回 Py_NotImplemented
    return Py_NotImplemented;
/*NUMPY_API
 * Does not check for NPY_ARRAY_ENSURECOPY and NPY_ARRAY_NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 */
// 定义一个不导出的 NumPy API 函数，从任意 Python 对象创建 NumPy 数组
NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context)
{
    // 定义一个结构体变量，用于存储 dtype 的信息，初始化为 NULL
    npy_dtype_info dt_info = {NULL, NULL};

    // 提取给定的 newtype 中的 dtype 和 descr，并存储在 dt_info 中
    int res = PyArray_ExtractDTypeAndDescriptor(
        newtype, &dt_info.descr, &dt_info.dtype);

    // 释放 newtype 的引用
    Py_XDECREF(newtype);

    // 如果提取过程失败，释放资源并返回 NULL
    if (res < 0) {
        Py_XDECREF(dt_info.descr);
        Py_XDECREF(dt_info.dtype);
        return NULL;
    }

    // 定义一个整型变量，用于存储是否为标量对象的信息
    int was_scalar;
    // 调用内部函数 PyArray_FromAny_int 进行实际的数组创建操作
    PyObject* ret =  PyArray_FromAny_int(
            op, dt_info.descr, dt_info.dtype,
            min_depth, max_depth, flags, context, &was_scalar);

    // 释放 dtype 和 descr 的引用
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return ret;
}

/*
 * Internal version of PyArray_FromAny that accepts a dtypemeta. Borrows
 * references to the descriptor and dtype.
 *
 * The `was_scalar` output returns 1 when the object was a "scalar".
 * This means it was:
 * - Recognized as a scalar by a/the dtype.  This can be DType specific,
 *   for example a tuple may be a scalar, but only for structured dtypes.
 * - Anything not recognized as an instance of a DType's scalar type but also not
 *   convertible to an array.  (no __array__ protocol, etc.)
 *   these must map to `dtype=object` (if a dtype wasn't specified).
 */
// 定义一个不导出的内部函数，接受 dtypemeta 并创建 NumPy 数组
NPY_NO_EXPORT PyObject *
PyArray_FromAny_int(PyObject *op, PyArray_Descr *in_descr,
                    PyArray_DTypeMeta *in_DType, int min_depth, int max_depth,
                    int flags, PyObject *context, int *was_scalar)
{
    /*
     * This is the main code to make a NumPy array from a Python
     * Object.  It is called from many different places.
     */
    // 定义指向 PyArrayObject 结构体的指针，并初始化为 NULL
    PyArrayObject *arr = NULL, *ret;
    // 定义一个指向 PyArray_Descr 结构体的指针，并初始化为 NULL
    PyArray_Descr *dtype = NULL;
    // 定义一个 coercion_cache_obj 结构体的指针，并初始化为 NULL
    coercion_cache_obj *cache = NULL;
    // 定义一个整型变量，用于存储数组的维度数，初始化为 0
    int ndim = 0;
    // 定义一个长度为 NPY_MAXDIMS 的整型数组，用于存储数组的形状
    npy_intp dims[NPY_MAXDIMS];

    // 如果 context 不为空，抛出运行时错误并返回 NULL
    if (context != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "'context' must be NULL");
        return NULL;
    }

    // 默认情况下，copy 设置为 -1
    int copy = -1;
    // 根据 flags 的值设置 copy 的具体行为
    int was_copied_by__array__ = 0;

    if (flags & NPY_ARRAY_ENSURENOCOPY) {
        copy = 0;
    } else if (flags & NPY_ARRAY_ENSURECOPY) {
        copy = 1;
    }

    // 调用 PyArray_DiscoverDTypeAndShape 函数，获取对象的 dtype 和形状
    ndim = PyArray_DiscoverDTypeAndShape(
            op, NPY_MAXDIMS, dims, &cache, in_DType, in_descr, &dtype,
            copy, &was_copied_by__array__);

    // 如果获取过程中出现错误，返回 NULL
    if (ndim < 0) {
        return NULL;
    }

    // 如果 cache 为 NULL，表示对象被视为标量
    *was_scalar = (cache == NULL);

    // 如果 dtype 为 NULL，则使用默认的 dtype
    if (dtype == NULL) {
        dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    // 如果 min_depth 不为 0 且 ndim 小于 min_depth，抛出值错误并返回 NULL
    if (min_depth != 0 && ndim < min_depth) {
        PyErr_SetString(PyExc_ValueError,
                "object of too small depth for desired array");
        Py_DECREF(dtype);
        npy_free_coercion_cache(cache);
        return NULL;
    }
    // 如果最大深度不为零且数组维度超过最大深度，则抛出值错误异常
    if (max_depth != 0 && ndim > max_depth) {
        PyErr_SetString(PyExc_ValueError,
                "object too deep for desired array");
        // 释放 dtype 引用
        Py_DECREF(dtype);
        // 释放类型强制缓存
        npy_free_coercion_cache(cache);
        // 返回空指针
        return NULL;
    }

    /* 已获取正确的参数，但缓存可能已经保存了结果 */
    if (cache != NULL && !(cache->sequence)) {
        /*
         * 只有一个类似数组，并且已经被转换，可能仍然具有错误的类型，
         * 但下面会处理这种情况。
         */
        assert(cache->converted_obj == op);
        // 将 cache 中的数组对象转换为 PyArrayObject 类型
        arr = (PyArrayObject *)(cache->arr_or_sequence);
        /* 可能需要强制类型转换或标记（例如复制） */
        if (was_copied_by__array__ == 1) {
            flags = flags & ~NPY_ARRAY_ENSURECOPY;
        }
        // 从给定数组对象 arr 创建一个新的 PyArrayObject 对象
        PyObject *res = PyArray_FromArray(arr, dtype, flags);
        // 解除类型强制缓存
        npy_unlink_coercion_cache(cache);
        // 返回创建的数组对象
        return res;
    }
    else if (cache == NULL && PyArray_IsScalar(op, Void) &&
            !(((PyVoidScalarObject *)op)->flags & NPY_ARRAY_OWNDATA) &&
             ((in_descr == NULL) && (in_DType == NULL))) {
        /*
         * 特殊情况，返回一个对空标量的视图，主要是为了允许类似于“reversed”赋值的操作:
         *    arr[indx]["field"] = val  # 而不是 arr["field"][indx] = val
         *
         * 不清楚在这条特定的代码路径中是否有必要。
         * 注意，只有在用户没有提供 dtype（newtype 为 NULL）时才会激活这条路径。
         */
        assert(ndim == 0);

        // 根据描述符和基础数据创建一个新的 PyArrayObject 对象
        return PyArray_NewFromDescrAndBase(
                &PyArray_Type, dtype,
                0, NULL, NULL,
                ((PyVoidScalarObject *)op)->obval,
                ((PyVoidScalarObject *)op)->flags,
                NULL, op);
    }
    /*
     * 如果执行到这里，肯定需要创建一个副本，因为我们要么从标量（cache == NULL）转换，
     * 要么从（嵌套的）序列转换。
     */
    if (flags & NPY_ARRAY_ENSURENOCOPY) {
        // 设置值错误异常，指示不允许复制操作
        PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
        // 释放 dtype 引用
        Py_DECREF(dtype);
        // 释放类型强制缓存
        npy_free_coercion_cache(cache);
        // 返回空指针
        return NULL;
    }
    /*
     * 如果缓存为空并且输入描述不为空，并且输入的数据类型是有符号的并且
     * op 是一个 Generic 类型的标量数组，那么执行以下操作。
     */
    if (cache == NULL && in_descr != NULL &&
            PyDataType_ISSIGNED(dtype) &&
            PyArray_IsScalar(op, Generic)) {
        assert(ndim == 0);
        /*
         * 这里存在一个可能的不一致性，例如：
         *
         *     np.array(np.float64(np.nan), dtype=np.int64)
         *
         * 和以下方式表现不同：
         *
         *     np.array([np.float64(np.nan)], dtype=np.int64)
         *     arr1d_int64[0] = np.float64(np.nan)
         *     np.array(np.array(np.nan), dtype=np.int64)
         *
         * 后者不会抛出错误，而是使用了典型的类型转换。这种错误是期望的，
         * 但是总是抛出错误似乎是一个更大的变化，应该在其他时间考虑，
         * 并且不希望零维数组表现不同于标量。
         * 这保留了这种行为，主要是由于 pandas 中的问题，该问题依赖于
         * try/except（尽管希望在某个时候有更好的解决方案）：
         * https://github.com/pandas-dev/pandas/issues/35481
         */
        return PyArray_FromScalar(op, dtype);
    }

    /* 没有直接传入数组（或类似数组的对象）。 */
    if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
        PyErr_SetString(PyExc_TypeError,
                        "WRITEBACKIFCOPY used for non-array input.");
        Py_DECREF(dtype);
        npy_free_coercion_cache(cache);
        return NULL;
    }

    /* 创建一个新的数组并复制数据 */
    Py_INCREF(dtype);  /* 在可能被替换的子数组中保持引用 */
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, dtype, ndim, dims, NULL, NULL,
            flags&NPY_ARRAY_F_CONTIGUOUS, NULL);
    if (ret == NULL) {
        npy_free_coercion_cache(cache);
        Py_DECREF(dtype);
        return NULL;
    }
    if (ndim == PyArray_NDIM(ret)) {
        /*
         * 没有发生尺寸的追加，因此使用实际的 dtype。对于 S0 或 U0，
         * 可以被替换为 S1 或 U1，虽然这可能会发生变化。
         */
        Py_SETREF(dtype, PyArray_DESCR(ret));
        Py_INCREF(dtype);
    }

    if (cache == NULL) {
        /* 这是一个单项。直接设置它。 */
        assert(ndim == 0);

        if (PyArray_Pack(dtype, PyArray_BYTES(ret), op) < 0) {
            Py_DECREF(dtype);
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(dtype);
        return (PyObject *)ret;
    }
    assert(ndim != 0);
    assert(op == cache->converted_obj);

    /* 将维度数减少到检测到的数量 */
    int out_ndim = PyArray_NDIM(ret);
    PyArray_Descr *out_descr = PyArray_DESCR(ret);
    if (out_ndim != ndim) {
        ((PyArrayObject_fields *)ret)->nd = ndim;
        ((PyArrayObject_fields *)ret)->descr = dtype;
    }

    int success = PyArray_AssignFromCache(ret, cache);

    ((PyArrayObject_fields *)ret)->nd = out_ndim;
    # 将 ret 强制转换为 PyArrayObject_fields 类型，设置其 descr 字段为 out_descr
    ((PyArrayObject_fields *)ret)->descr = out_descr;
    # 减少 dtype 的引用计数，释放其占用的内存
    Py_DECREF(dtype);
    # 如果 success 小于 0，说明出现错误，需要释放 ret 对象并返回 NULL
    if (success < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    # 返回 ret 对象的 PyObject 指针类型
    return (PyObject *)ret;
/*NUMPY_API
 * steals a reference to descr -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context)
{
    // 定义一个结构体变量 dt_info，包含两个指针成员，初始化为 NULL
    npy_dtype_info dt_info = {NULL, NULL};

    // 调用 PyArray_ExtractDTypeAndDescriptor 函数，提取描述符和数据类型
    int res = PyArray_ExtractDTypeAndDescriptor(
        descr, &dt_info.descr, &dt_info.dtype);

    // 释放输入参数 descr 的引用
    Py_XDECREF(descr);

    // 如果提取过程出错，则释放描述符和数据类型，返回 NULL
    if (res < 0) {
        Py_XDECREF(dt_info.descr);
        Py_XDECREF(dt_info.dtype);
        return NULL;
    }

    // 调用 PyArray_CheckFromAny_int 函数，传入提取的描述符和数据类型
    // 以及其他参数，获取返回值保存在 ret 变量中
    PyObject* ret =  PyArray_CheckFromAny_int(
        op, dt_info.descr, dt_info.dtype, min_depth, max_depth, requires,
        context);

    // 释放描述符和数据类型的引用
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);

    // 返回 PyArray_CheckFromAny_int 函数的返回值
    return ret;
}

/*
 * Internal version of PyArray_CheckFromAny that accepts a dtypemeta. Borrows
 * references to the descriptor and dtype.
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny_int(PyObject *op, PyArray_Descr *in_descr,
                         PyArray_DTypeMeta *in_DType, int min_depth,
                         int max_depth, int requires, PyObject *context)
{
    // 声明一个 PyObject 类型的指针变量 obj
    PyObject *obj;

    // 下面是 PyArray_CheckFromAny_int 函数的具体实现，未完待续...


这里只提供了部分代码的注释，如果需要完整的注释，请继续扩展注释内容以涵盖整个函数的实现细节。
    # 检查是否需要设置 NPY_ARRAY_NOTSWAPPED 标志位
    if (requires & NPY_ARRAY_NOTSWAPPED) {
        # 如果未提供输入描述符并且操作对象是 NumPy 数组且已经字节交换过
        if (!in_descr && PyArray_Check(op) &&
                PyArray_ISBYTESWAPPED((PyArrayObject* )op)) {
            # 根据操作对象创建一个新的描述符
            in_descr = PyArray_DescrNew(PyArray_DESCR((PyArrayObject *)op));
            # 如果描述符创建失败则返回空
            if (in_descr == NULL) {
                return NULL;
            }
        }
        # 如果已经存在输入描述符并且其字节顺序不是本机字节顺序
        else if (in_descr && !PyArray_ISNBO(in_descr->byteorder)) {
            # 替换输入描述符为本机字节顺序
            PyArray_DESCR_REPLACE(in_descr);
        }
        # 如果输入描述符存在并且其字节顺序不是 NPY_IGNORE
        if (in_descr && in_descr->byteorder != NPY_IGNORE) {
            # 将输入描述符的字节顺序设置为本机字节顺序
            in_descr->byteorder = NPY_NATIVE;
        }
    }

    # 定义一个整型变量 was_scalar
    int was_scalar;
    # 将操作对象 op 转换为 NumPy 数组，返回结果存入 obj
    obj = PyArray_FromAny_int(op, in_descr, in_DType, min_depth,
                              max_depth, requires, context, &was_scalar);
    # 如果转换失败则返回空
    if (obj == NULL) {
        return NULL;
    }

    # 如果需要设置 NPY_ARRAY_ELEMENTSTRIDES 标志位且 obj 没有元素步长
    if ((requires & NPY_ARRAY_ELEMENTSTRIDES)
            && !PyArray_ElementStrides(obj)) {
        PyObject *ret;
        # 如果需要确保不复制且无法满足条件，则设置错误并返回空
        if (requires & NPY_ARRAY_ENSURENOCOPY) {
            PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
            return NULL;
        }
        # 否则创建 obj 的副本并存入 ret
        ret = PyArray_NewCopy((PyArrayObject *)obj, NPY_ANYORDER);
        # 减少 obj 的引用计数
        Py_DECREF(obj);
        # 将 obj 设置为 ret
        obj = ret;
    }
    # 返回 obj
    return obj;
    /* 结束之前的代码块并开启一个新的代码块
     * 此处定义了一个 NumPy API，用于从给定的 PyArrayObject 中创建一个新的 PyArrayObject
     * 参数说明：
     *   arr: 要转换的 PyArrayObject
     *   newtype: 新的数据类型描述符，如果为 NULL，则使用 arr 的数据类型描述符
     *   flags: 控制转换过程的标志
     */
    NPY_NO_EXPORT PyObject *
    PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags)
    {

        PyArrayObject *ret = NULL;
        int copy = 0;
        int arrflags;
        PyArray_Descr *oldtype;
        NPY_CASTING casting = NPY_SAFE_CASTING;

        oldtype = PyArray_DESCR(arr);

        /* 如果 newtype 为 NULL，则检查 arr 是否是带有 Null newtype 的数组对象
         * 如果是，则直接返回 arr，增加其引用计数
         * 否则继续进行类型转换的检查
         */
        if (newtype == NULL) {
            if (flags == 0) {
                Py_INCREF(arr);
                return (PyObject *)arr;
            }
            newtype = oldtype;
            Py_INCREF(oldtype);
        }
        else if (PyDataType_ISUNSIZED(newtype)) {
            /* 如果 newtype 是无大小的数据类型描述符，
             * 尝试用 arr 的数据类型描述符替换 newtype 的描述符
             * 如果替换失败，则返回 NULL
             */
            PyArray_DESCR_REPLACE(newtype);
            if (newtype == NULL) {
                return NULL;
            }
            newtype->elsize = oldtype->elsize;
        }

        /* 如果 flags 中包含 NPY_ARRAY_SAME_KIND_CASTING，
         * 设置 casting 规则为 NPY_SAME_KIND_CASTING
         */
        if (flags & NPY_ARRAY_SAME_KIND_CASTING) {
            casting = NPY_SAME_KIND_CASTING;
        }

        /* 如果 flags 中包含 NPY_ARRAY_FORCECAST，
         * 强制使用不安全的 casting 规则
         */
        if (flags & NPY_ARRAY_FORCECAST) {
            casting = NPY_UNSAFE_CASTING;
        }

        /* 如果不能将 arr 安全地转换为 newtype 所指定的数据类型，
         * 则抛出错误并返回 NULL
         */
        if (!PyArray_CanCastArrayTo(arr, newtype, casting)) {
            PyErr_Clear();
            npy_set_invalid_cast_error(
                    PyArray_DESCR(arr), newtype, casting, PyArray_NDIM(arr) == 0);
            Py_DECREF(newtype);
            return NULL;
        }

        arrflags = PyArray_FLAGS(arr);

        /* 检查是否需要进行复制操作：
         *   如果 flags 包含 NPY_ARRAY_ENSURECOPY，则必须复制
         *   如果 flags 包含 NPY_ARRAY_C_CONTIGUOUS 并且 arr 不是 C 连续的，则必须复制
         *   如果 flags 包含 NPY_ARRAY_ALIGNED 并且 arr 不是对齐的，则必须复制
         *   如果 flags 包含 NPY_ARRAY_F_CONTIGUOUS 并且 arr 不是 Fortran 连续的，则必须复制
         *   如果 flags 包含 NPY_ARRAY_WRITEABLE 并且 arr 不可写，则必须复制
         */
        copy = (flags & NPY_ARRAY_ENSURECOPY) ||
               ((flags & NPY_ARRAY_C_CONTIGUOUS) &&
                       (!(arrflags & NPY_ARRAY_C_CONTIGUOUS))) ||
               ((flags & NPY_ARRAY_ALIGNED) &&
                       (!(arrflags & NPY_ARRAY_ALIGNED))) ||
               ((flags & NPY_ARRAY_F_CONTIGUOUS) &&
                       (!(arrflags & NPY_ARRAY_F_CONTIGUOUS))) ||
               ((flags & NPY_ARRAY_WRITEABLE) &&
                       (!(arrflags & NPY_ARRAY_WRITEABLE)));

        /* 如果不需要复制，则尝试使用视图的方式进行转换，
         * 如果不安全或者无法进行视图转换，则需要进行复制
         */
        if (!copy) {
            npy_intp view_offset;
            npy_intp is_safe = PyArray_SafeCast(oldtype, newtype, &view_offset, NPY_NO_CASTING, 1);
            copy = !(is_safe && (view_offset != NPY_MIN_INTP));
        }
        
    // 如果需要复制数组
    if (copy) {
        // 如果设置了 NPY_ARRAY_ENSURENOCOPY 标志，设置错误信息并返回空
        if (flags & NPY_ARRAY_ENSURENOCOPY) {
            PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
            Py_DECREF(newtype);
            return NULL;
        }

        // 默认的数组复制顺序为 NPY_KEEPORDER
        NPY_ORDER order = NPY_KEEPORDER;
        // 默认允许生成子类数组
        int subok = 1;

        /* 根据 flags 设置复制的顺序 */
        // 如果设置了 NPY_ARRAY_F_CONTIGUOUS 标志，则使用 Fortran 排序顺序
        if (flags & NPY_ARRAY_F_CONTIGUOUS) {
            order = NPY_FORTRANORDER;
        }
        // 如果设置了 NPY_ARRAY_C_CONTIGUOUS 标志，则使用 C 排序顺序
        else if (flags & NPY_ARRAY_C_CONTIGUOUS) {
            order = NPY_CORDER;
        }

        // 如果设置了 NPY_ARRAY_ENSUREARRAY 标志，则不允许生成子类数组
        if ((flags & NPY_ARRAY_ENSUREARRAY)) {
            subok = 0;
        }

        // 增加新类型的引用计数
        Py_INCREF(newtype);
        // 根据给定的 order、newtype 和 subok 创建一个新的数组对象 ret
        ret = (PyArrayObject *)PyArray_NewLikeArray(arr, order,
                                                    newtype, subok);
        // 如果创建失败，则释放 newtype 并返回空
        if (ret == NULL) {
            Py_DECREF(newtype);
            return NULL;
        }

        // 获取实际的维度和数据类型
        int actual_ndim = PyArray_NDIM(ret);
        PyArray_Descr *actual_dtype = PyArray_DESCR(ret);
        // 如果实际维度与 arr 不同，更新 ret 的维度和描述符
        if (actual_ndim != PyArray_NDIM(arr)) {
            ((PyArrayObject_fields *)ret)->nd = PyArray_NDIM(arr);
            ((PyArrayObject_fields *)ret)->descr = newtype;
        }

        // 将 arr 的数据复制到 ret 中，返回值为成功复制的元素数
        int success = PyArray_CopyInto(ret, arr);

        // 释放 newtype，并将 ret 的维度和描述符还原为原始值
        Py_DECREF(newtype);
        ((PyArrayObject_fields *)ret)->nd = actual_ndim;
        ((PyArrayObject_fields *)ret)->descr = actual_dtype;

        // 如果复制失败，释放 ret 并返回空
        if (success < 0) {
            Py_DECREF(ret);
            return NULL;
        }

        // 如果设置了 NPY_ARRAY_WRITEBACKIFCOPY 标志，设置 writeback 基数组
        if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            Py_INCREF(arr);
            // 将 arr 设置为 ret 的 writeback 基数组，如果失败则释放 ret 并返回空
            if (PyArray_SetWritebackIfCopyBase(ret, arr) < 0) {
                Py_DECREF(ret);
                return NULL;
            }
        }
    }
    /*
     * 如果不需要复制，则根据需要获取适当的视图或直接返回 ret 的引用。
     */
    else {
        // 如果需要确保生成数组，并且 arr 不是 PyArray_Type 的确切实例，则需要视图
        int needview = ((flags & NPY_ARRAY_ENSUREARRAY) &&
                        !PyArray_CheckExact(arr));

        // 释放 newtype
        Py_DECREF(newtype);

        // 如果需要视图，则创建 arr 的视图
        if (needview) {
            PyTypeObject *subtype = NULL;

            // 如果需要确保生成数组，则 subtype 设置为 PyArray_Type
            if (flags & NPY_ARRAY_ENSUREARRAY) {
                subtype = &PyArray_Type;
            }
            // 创建 arr 的视图，如果失败则返回空
            ret = (PyArrayObject *)PyArray_View(arr, NULL, subtype);
            if (ret == NULL) {
                return NULL;
            }
        }
        // 否则，增加 arr 的引用计数并将其赋给 ret
        else {
            Py_INCREF(arr);
            ret = arr;
        }
    }

    // 返回 ret 对象的 PyObject 类型指针
    return (PyObject *)ret;
/*NUMPY_API */
/* 定义一个不导出的函数 PyArray_FromStructInterface，接收一个 PyObject 类型的输入 */
NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input)
{
    /* 定义 PyArray_Descr 类型的指针变量 thetype，并初始化为 NULL */
    PyArray_Descr *thetype = NULL;
    /* 定义 PyArrayInterface 类型的指针变量 inter */
    PyArrayInterface *inter;
    /* 定义 PyObject 类型的变量 attr */
    PyObject *attr;
    /* 定义字符类型的变量 endian，并初始化为 NPY_NATBYTE */
    char endian = NPY_NATBYTE;

    /* 调用 PyArray_LookupSpecial_OnInstance 函数查找对象 input 的 npy_interned_str.array_struct 属性 */
    attr = PyArray_LookupSpecial_OnInstance(input, npy_interned_str.array_struct);
    /* 如果未找到属性 */
    if (attr == NULL) {
        /* 如果出现了错误，则返回 NULL */
        if (PyErr_Occurred()) {
            return NULL;
        } else {
            /* 否则返回 Py_NotImplemented */
            return Py_NotImplemented;
        }
    }
    /* 如果 attr 不是一个精确的 PyCapsule 对象 */
    if (!PyCapsule_CheckExact(attr)) {
        /* 如果 input 是一个类型对象，并且 attr 具有 "__get__" 属性 */
        if (PyType_Check(input) && PyObject_HasAttrString(attr, "__get__")) {
            /*
             * 如果输入是一个类，那么 attr 应该是类似属性的对象。
             * 这不能被解释为一个数组，但是是有效的。
             * （由于查找是在实例上而不是类型上，所以需要这个）
             */
            /* 释放 attr 的引用并返回 Py_NotImplemented */
            Py_DECREF(attr);
            return Py_NotImplemented;
        }
        /* 转到失败处理 */
        goto fail;
    }
    /* 从 PyCapsule 中获取指针 inter */
    inter = PyCapsule_GetPointer(attr, NULL);
    /* 如果 inter 为空指针，则转到失败处理 */
    if (inter == NULL) {
        goto fail;
    }
    /* 如果 inter 的版本号不等于 2，则转到失败处理 */
    if (inter->two != 2) {
        goto fail;
    }
    /* 如果 inter 的 flags 中不包含 NPY_ARRAY_NOTSWAPPED 标志 */
    if ((inter->flags & NPY_ARRAY_NOTSWAPPED) != NPY_ARRAY_NOTSWAPPED) {
        /* 将 endian 设为 NPY_OPPBYTE，并清除 inter 的 NPY_ARRAY_NOTSWAPPED 标志位 */
        endian = NPY_OPPBYTE;
        inter->flags &= ~NPY_ARRAY_NOTSWAPPED;
    }

    /* 如果 inter 的 flags 中包含 NPY_ARR_HAS_DESCR 标志 */
    if (inter->flags & NPY_ARR_HAS_DESCR) {
        /* 尝试从 inter 的 descr 字段中转换为 PyArray_Descr 类型的对象，并将结果存入 thetype */
        if (PyArray_DescrConverter(inter->descr, &thetype) == NPY_FAIL) {
            /* 如果转换失败，则将 thetype 置为 NULL 并清除错误状态 */
            thetype = NULL;
            PyErr_Clear();
        }
    }

    /* 如果 thetype 仍为 NULL */
    if (thetype == NULL) {
        /* 从格式化字符串 "%c%c%d" 创建 PyUnicode 对象 type_str */
        PyObject *type_str = PyUnicode_FromFormat(
            "%c%c%d", endian, inter->typekind, inter->itemsize);
        /* 如果创建 type_str 失败，则释放 attr 的引用并返回 NULL */
        if (type_str == NULL) {
            Py_DECREF(attr);
            return NULL;
        }
        /* 尝试从 type_str 转换为 PyArray_Descr 类型的对象，并将结果存入 thetype */
        int ok = PyArray_DescrConverter(type_str, &thetype);
        /* 释放 type_str 的引用 */
        Py_DECREF(type_str);
        /* 如果转换不成功，则释放 attr 的引用并返回 NULL */
        if (ok != NPY_SUCCEED) {
            Py_DECREF(attr);
            return NULL;
        }
    }

    /* 创建一个长度为 2 的 PyTuple 对象 refs */
    PyObject *refs = PyTuple_New(2);
    /* 如果创建 refs 失败，则释放 attr 的引用并返回 NULL */
    if (!refs) {
        Py_DECREF(attr);
        return NULL;
    }

    /* 增加输入对象 input 的引用计数 */
    Py_INCREF(input);
    /* 将 input 存入 refs 的第一个元素位置 */
    PyTuple_SET_ITEM(refs, 0, input);

    /* 增加 PyCapsule 对象 attr 的引用计数，并将其存入 refs 的第二个元素位置 */
    PyTuple_SET_ITEM(refs, 1,  attr);

    /* 创建 numpy 数组，此调用会增加 refs 的引用计数 */
    PyObject *ret = PyArray_NewFromDescrAndBase(
            &PyArray_Type, thetype,
            inter->nd, inter->shape, inter->strides, inter->data,
            inter->flags, NULL, refs);

    /* 释放 refs 的引用 */
    Py_DECREF(refs);

    /* 返回创建的 numpy 数组对象 */
    return ret;

 fail:
    /* 设置 ValueError 异常，并返回 NULL */
    PyErr_SetString(PyExc_ValueError, "invalid __array_struct__");
    /* 释放 attr 的引用并返回 NULL */
    Py_DECREF(attr);
    return NULL;
}
/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *origin)
{
    PyObject *iface = NULL;  // 定义接口对象指针，初始化为 NULL
    PyObject *attr = NULL;   // 定义属性对象指针，初始化为 NULL
    PyObject *base = NULL;   // 定义基本对象指针，初始化为 NULL
    PyArrayObject *ret;      // 定义返回的数组对象指针
    PyArray_Descr *dtype = NULL;  // 定义数组描述符指针，初始化为 NULL
    char *data = NULL;       // 数据指针，初始化为 NULL
    Py_buffer view;          // 缓冲区视图对象
    int i, n;                // 整型变量 i, n，用于循环计数或索引
    npy_intp dims[NPY_MAXDIMS], strides[NPY_MAXDIMS];  // 维度和步幅数组
    int dataflags = NPY_ARRAY_BEHAVED;  // 数据标志，初始为 NPY_ARRAY_BEHAVED

    iface = PyArray_LookupSpecial_OnInstance(origin, npy_interned_str.array_interface);

    if (iface == NULL) {  // 如果接口对象为空
        if (PyErr_Occurred()) {
            return NULL;  // 如果发生错误，返回空指针
        }
        return Py_NotImplemented;  // 否则返回未实现
    }

    if (!PyDict_Check(iface)) {  // 如果接口对象不是字典类型
        if (PyType_Check(origin) && PyObject_HasAttrString(iface, "__get__")) {
            /*
             * 如果输入是一个类，iface 应该是类似属性的对象。
             * 这不能被解释为数组，但是是有效的。
             * (由于查找是在实例而不是类型上进行的，因此需要)
             */
            Py_DECREF(iface);  // 减少接口对象的引用计数
            return Py_NotImplemented;  // 返回未实现
        }

        Py_DECREF(iface);  // 减少接口对象的引用计数
        PyErr_SetString(PyExc_ValueError,
                "Invalid __array_interface__ value, must be a dict");  // 设置错误信息
        return NULL;  // 返回空指针
    }

    /* 从接口规范中获取类型字符串 */
    int result = PyDict_GetItemStringRef(iface, "typestr", &attr);
    if (result <= 0) {  // 如果获取失败
        Py_DECREF(iface);  // 减少接口对象的引用计数
        if (result == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Missing __array_interface__ typestr");  // 设置错误信息
        }
        return NULL;  // 返回空指针
    }

    /* 允许字节以确保向后兼容 */
    if (!PyBytes_Check(attr) && !PyUnicode_Check(attr)) {  // 如果属性既不是字节类型也不是 Unicode 类型
        PyErr_SetString(PyExc_TypeError,
                    "__array_interface__ typestr must be a string");  // 设置类型错误信息
        goto fail;  // 跳转到失败标签
    }

    /* 从类型字符串获取数据类型描述符 */
    if (PyArray_DescrConverter(attr, &dtype) != NPY_SUCCEED) {  // 如果类型字符串转换为描述符失败
        goto fail;  // 跳转到失败标签
    }

    /*
     * 如果 dtype 是 NPY_VOID 类型，则查看 'descr' 属性中是否有额外信息。
     */
    // 检查 dtype 是否为 NPY_VOID 类型
    if (dtype->type_num == NPY_VOID) {
        // 定义一个空描述符对象
        PyObject *descr = NULL;
        // 从 iface 字典中获取 "descr" 键对应的值
        result = PyDict_GetItemStringRef(iface, "descr", &descr);
        // 若获取失败，则跳转至 fail 标签处
        if (result == -1) {
            goto fail;
        }
        // 定义一个新的数组描述符指针
        PyArray_Descr *new_dtype = NULL;
        // 如果成功获取了描述符对象
        if (result == 1) {
            // 检查描述符是否为默认值
            int is_default = _is_default_descr(descr, attr);
            // 若检查失败，则释放描述符对象并跳转至 fail 标签处
            if (is_default < 0) {
                Py_DECREF(descr);
                goto fail;
            }
            // 如果描述符不是默认值
            if (!is_default) {
                // 尝试将描述符转换为新的数组描述符对象
                if (PyArray_DescrConverter2(descr, &new_dtype) != NPY_SUCCEED) {
                    Py_DECREF(descr);
                    goto fail;
                }
                // 如果成功转换，则设置 dtype 指针指向新的描述符对象
                if (new_dtype != NULL) {
                    Py_SETREF(dtype, new_dtype);
                }
            }
        }
        // 释放描述符对象的引用计数
        Py_DECREF(descr);
    }
    // 清空 attr 指针的值
    Py_CLEAR(attr);

    /* 从接口规范中获取形状元组 */
    result = PyDict_GetItemStringRef(iface, "shape", &attr);
    // 若获取失败，则返回空值
    if (result < 0) {
        return NULL;
    }
    // 如果 "shape" 不存在于 iface 字典中
    if (result == 0) {
        /* 当指定了 'data' 时，必须指定形状 */
        // 检查 iface 字典中是否包含 "data" 键
        int result = PyDict_ContainsString(iface, "data");
        // 如果检查失败，则释放 attr 并返回空值
        if (result < 0) {
            Py_DECREF(attr);
            return NULL;
        }
        // 如果 "data" 存在于 iface 字典中
        else if (result == 1) {
            // 释放 iface 和 attr，设置异常并返回空值
            Py_DECREF(iface);
            Py_DECREF(attr);
            PyErr_SetString(PyExc_ValueError,
                    "Missing __array_interface__ shape");
            return NULL;
        }
        // 否则（假设 shape 为标量）
        else {
            /* 注意：data 和 base 指针应为 NULL */
            // 将 dims[0] 设置为 0，n 也设置为 0
            n = dims[0] = 0;
        }
    }
    // 如果 attr 不是元组类型
    else if (!PyTuple_Check(attr)) {
        // 设置类型错误异常并跳转至 fail 标签处
        PyErr_SetString(PyExc_TypeError,
                "shape must be a tuple");
        goto fail;
    }
    // 否则，假设 attr 是元组类型
    else {
        // 获取元组的大小，并将各个元素转换为 PyInt，存储在 dims 数组中
        n = PyTuple_GET_SIZE(attr);
        for (i = 0; i < n; i++) {
            PyObject *tmp = PyTuple_GET_ITEM(attr, i);
            dims[i] = PyArray_PyIntAsIntp(tmp);
            // 如果转换出错，则跳转至 fail 标签处
            if (error_converting(dims[i])) {
                goto fail;
            }
        }
    }
    // 清空 attr 指针的值
    Py_CLEAR(attr);

    /* 从接口规范中获取数据缓冲区 */
    result = PyDict_GetItemStringRef(iface, "data", &attr);
    // 若获取失败，则返回空值
    if (result == -1){
        return NULL;
    }

    /* 处理通过指针访问数据的情况 */
    // 检查 attr 是否存在且为 PyTuple 类型
    if (attr && PyTuple_Check(attr)) {
        PyObject *dataptr;
        // 如果 PyTuple 的大小不为 2，则抛出类型错误
        if (PyTuple_GET_SIZE(attr) != 2) {
            PyErr_SetString(PyExc_TypeError,
                    "__array_interface__ data must be a 2-tuple with "
                    "(data pointer integer, read-only flag)");
            goto fail;
        }
        // 获取 PyTuple 中的第一个元素作为数据指针
        dataptr = PyTuple_GET_ITEM(attr, 0);
        // 检查数据指针是否为 PyLong 类型
        if (PyLong_Check(dataptr)) {
            // 将 PyLong 数据指针转换为 void* 类型
            data = PyLong_AsVoidPtr(dataptr);
            // 如果转换失败，则跳转到 fail 标签处
            if (data == NULL && PyErr_Occurred()) {
                goto fail;
            }
        }
        else {
            // 若第一个元素不是整数，则抛出类型错误
            PyErr_SetString(PyExc_TypeError,
                    "first element of __array_interface__ data tuple "
                    "must be an integer.");
            goto fail;
        }
        // 获取 PyTuple 中第二个元素的真值
        int istrue = PyObject_IsTrue(PyTuple_GET_ITEM(attr,1));
        // 若获取真值失败，则跳转到 fail 标签处
        if (istrue == -1) {
            goto fail;
        }
        // 若第二个元素为真，则将 dataflags 中的可写标志位清除
        if (istrue) {
            dataflags &= ~NPY_ARRAY_WRITEABLE;
        }
        // 将 base 设置为 origin
        base = origin;
    }

    /* Case for data access through buffer */
    else if (attr) {
        // 如果 attr 不为空且不为 Py_None，则将 base 设置为 attr
        if (attr != Py_None) {
            base = attr;
        }
        else {
            // 否则将 base 设置为 origin
            base = origin;
        }
        // 尝试获取 buffer 视图并检查是否可写
        if (PyObject_GetBuffer(base, &view,
                    PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
            PyErr_Clear();
            // 若获取失败，则尝试简单获取 buffer 视图
            if (PyObject_GetBuffer(base, &view,
                        PyBUF_SIMPLE) < 0) {
                goto fail;
            }
            // 若视图不可写，则清除 dataflags 中的可写标志位
            dataflags &= ~NPY_ARRAY_WRITEABLE;
        }
        // 将 data 设置为 buffer 视图的指针
        data = (char *)view.buf;
        /*
         * In Python 3 both of the deprecated functions PyObject_AsWriteBuffer and
         * PyObject_AsReadBuffer that this code replaces release the buffer. It is
         * up to the object that supplies the buffer to guarantee that the buffer
         * sticks around after the release.
         */
        // 释放 Python 3 中已弃用的函数 PyObject_AsWriteBuffer 和 PyObject_AsReadBuffer 所释放的 buffer
        PyBuffer_Release(&view);

        /* Get offset number from interface specification */
        // 从接口规范中获取偏移量数值
        PyObject *offset = NULL;
        // 尝试从 iface 字典中获取 "offset" 对应的值
        result = PyDict_GetItemStringRef(iface, "offset", &offset);
        // 若出错则跳转到 fail 标签处
        if (result == -1) {
            goto fail;
        }
        else if (result == 1) {
            // 若成功获取到 offset，则将其转换为长长整型
            npy_longlong num = PyLong_AsLongLong(offset);
            // 若转换出错，则抛出类型错误
            if (error_converting(num)) {
                PyErr_SetString(PyExc_TypeError,
                        "__array_interface__ offset must be an integer");
                Py_DECREF(offset);
                goto fail;
            }
            // 将 data 增加偏移量 num
            data += num;
            Py_DECREF(offset);
        }
    }
    // 清除 attr 引用
    Py_CLEAR(attr);

    // 使用 PyArray_NewFromDescrAndBase 创建 PyArrayObject 对象
    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, dtype,
            n, dims, NULL, data,
            dataflags, NULL, base);
    /*
     * Ref to dtype was stolen by PyArray_NewFromDescrAndBase
     * Prevent DECREFing dtype in fail codepath by setting to NULL
     */
    // 防止在失败的代码路径中释放 dtype，因为它已被 PyArray_NewFromDescrAndBase 拿走了引用
    dtype = NULL;
    // 若创建失败，则跳转到 fail 标签处
    if (ret == NULL) {
        goto fail;
    }
    # 检查数据是否为 NULL
    if (data == NULL) {
        # 如果返回数组的大小大于 1，则抛出值错误异常
        if (PyArray_SIZE(ret) > 1) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot coerce scalar to array with size > 1");
            Py_DECREF(ret);  # 减少返回数组的引用计数
            goto fail;  # 转到错误处理代码块
        }
        # 将 origin 转换为标量并设置为返回数组的第一个元素
        if (PyArray_SETITEM(ret, PyArray_DATA(ret), origin) < 0) {
            Py_DECREF(ret);  # 减少返回数组的引用计数
            goto fail;  # 转到错误处理代码块
        }
    }
    # 获取 iface 字典中 "strides" 键对应的值和属性是否为 Py_None
    result = PyDict_GetItemStringRef(iface, "strides", &attr);
    # 如果获取失败，返回 NULL
    if (result == -1){
        return NULL;
    }
    # 如果 result 为 1 并且 attr 不为 Py_None
    if (result == 1 && attr != Py_None) {
        # 如果 attr 不是元组，则抛出类型错误异常
        if (!PyTuple_Check(attr)) {
            PyErr_SetString(PyExc_TypeError,
                    "strides must be a tuple");
            Py_DECREF(ret);  # 减少返回数组的引用计数
            goto fail;  # 转到错误处理代码块
        }
        # 如果 strides 数量与 shape 数量不匹配，则抛出值错误异常
        if (n != PyTuple_GET_SIZE(attr)) {
            PyErr_SetString(PyExc_ValueError,
                    "mismatch in length of strides and shape");
            Py_DECREF(ret);  # 减少返回数组的引用计数
            goto fail;  # 转到错误处理代码块
        }
        # 遍历 strides 元组，将每个元素转换为整数并存储在 strides 数组中
        for (i = 0; i < n; i++) {
            PyObject *tmp = PyTuple_GET_ITEM(attr, i);
            strides[i] = PyArray_PyIntAsIntp(tmp);
            # 如果转换出错，减少返回数组的引用计数并转到错误处理代码块
            if (error_converting(strides[i])) {
                Py_DECREF(ret);  # 减少返回数组的引用计数
                goto fail;  # 转到错误处理代码块
            }
        }
        # 如果 strides 数量大于 0，则复制 strides 数组到返回数组的步幅数组
        if (n) {
            memcpy(PyArray_STRIDES(ret), strides, n*sizeof(npy_intp));
        }
        Py_DECREF(attr);  # 减少 attr 的引用计数
    }
    # 更新返回数组的标志位
    PyArray_UpdateFlags(ret, NPY_ARRAY_UPDATE_ALL);
    Py_DECREF(iface);  # 减少 iface 的引用计数
    return (PyObject *)ret;  # 返回返回数组的 Python 对象

 fail:
    Py_XDECREF(attr);  # 安全地减少 attr 的引用计数
    Py_XDECREF(dtype);  # 安全地减少 dtype 的引用计数
    Py_XDECREF(iface);  # 安全地减少 iface 的引用计数
    return NULL;  # 返回 NULL 表示发生错误
/*
 * Returns -1 and an error set or 0 with the original error cleared, must
 * be called with an error set.
 */
static inline int
check_or_clear_and_warn_error_if_due_to_copy_kwarg(PyObject *kwnames)
{
    if (kwnames == NULL) {
        return -1;  /* 如果未传入 kwnames，则不可能是复制参数引起的错误 */
    }
    if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
        return -1;  /* 如果错误不是 TypeError 类型，则不可能是复制参数引起的错误 */
    }

    /*
     * In most cases, if we fail, we assume the error was unrelated to the
     * copy kwarg and simply restore the original one.
     */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    if (value == NULL) {
        goto restore_error;  /* 如果没有捕获到错误值，恢复原始错误 */
    }

    PyObject *str_value = PyObject_Str(value);
    if (str_value == NULL) {
        goto restore_error;  /* 如果无法将错误值转换为字符串，恢复原始错误 */
    }
    int copy_kwarg_unsupported = PyUnicode_Contains(
            str_value, npy_interned_str.array_err_msg_substr);
    Py_DECREF(str_value);
    if (copy_kwarg_unsupported == -1) {
        goto restore_error;  /* 如果在错误消息中查找复制参数不支持的字符串时出错，恢复原始错误 */
    }
    if (copy_kwarg_unsupported) {
        /*
         * TODO: As of now NumPy 2.0, the this warning is only triggered with
         *       `copy=False` allowing downstream to not notice it.
         */
        Py_DECREF(type);
        Py_DECREF(value);
        Py_XDECREF(traceback);
        if (DEPRECATE("__array__ implementation doesn't accept a copy keyword, "
                      "so passing copy=False failed. __array__ must implement "
                      "'dtype' and 'copy' keyword arguments.") < 0) {
            return -1;  /* 如果向下兼容提示失败，则返回错误 */
        }
        return 0;  /* 向下兼容提示成功 */
    }

  restore_error:
    PyErr_Restore(type, value, traceback);
    return -1;  /* 恢复原始错误并返回错误状态 */
}


/**
 * Check for an __array__ attribute and call it when it exists.
 *
 *  .. warning:
 *      If returned, `NotImplemented` is borrowed and must not be Decref'd
 *
 * @param op The Python object to convert to an array.
 * @param descr The desired `arr.dtype`, passed into the `__array__` call,
 *        as information but is not checked/enforced!
 * @param copy Specifies the copy behavior
 *        NOTE: For copy == -1 it passes `op.__array__(copy=None)`,
 *              for copy == 0, `op.__array__(copy=False)`, and
 *              for copy == 1, `op.__array__(copy=True).
 * @param was_copied_by__array__ Set to 1 if it can be assumed that a copy
 *        was made by implementor.
 * @returns NotImplemented if `__array__` is not defined or a NumPy array
 *          (or subclass).  On error, return NULL.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr_int(PyObject *op, PyArray_Descr *descr, int copy,
                          int *was_copied_by__array__)
{
    PyObject *new;
    PyObject *array_meth;

    array_meth = PyArray_LookupSpecial_OnInstance(op, npy_interned_str.array);
    if (array_meth == NULL) {
        if (PyErr_Occurred()) {
            return NULL;  /* 如果查找 __array__ 方法失败，返回 NULL */
        }
        return Py_NotImplemented;  /* 如果没有错误但没有 __array__ 方法，则返回 NotImplemented */
    }
    if (PyType_Check(op) && PyObject_HasAttrString(array_meth, "__get__")) {
        /*
         * 如果输入是一个类，`array_meth` 可能是类似属性的对象。
         * 这不能被解释为一个数组（调用），但是是有效的。
         * 在这种情况下，尝试对 `array_meth.__call__()` 的调用是无用的。
         * （因为查找是在实例上而不是类型上）
         */
        Py_DECREF(array_meth);
        return Py_NotImplemented;
    }

    Py_ssize_t nargs = 0;
    PyObject *arguments[2];
    PyObject *kwnames = NULL;

    if (descr != NULL) {
        arguments[0] = (PyObject *)descr;
        nargs++;
    }

    /*
     * 只有当 `copy` 的值不是默认值时，我们才尝试传递它；
     * 为了向后兼容性，如果因为调用的 `__array__` 方法的签名不包含 `copy` 而失败，则重试。
     */
    if (copy != -1) {
        kwnames = npy_static_pydata.kwnames_is_copy;
        arguments[nargs] = copy == 1 ? Py_True : Py_False;
    }

    int must_copy_but_copy_kwarg_unimplemented = 0;
    new = PyObject_Vectorcall(array_meth, arguments, nargs, kwnames);
    if (new == NULL) {
        if (check_or_clear_and_warn_error_if_due_to_copy_kwarg(kwnames) < 0) {
            /* 错误未清除（或设置了新错误） */
            Py_DECREF(array_meth);
            return NULL;
        }
        if (copy == 0) {
            /* 无法避免复制，因此报错。 */
            PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
            Py_DECREF(array_meth);
            return NULL;
        }
        /*
         * 错误似乎是由于传递了 `copy` 导致的。我们尝试更详细地了解消息并可能重试。
         */
        must_copy_but_copy_kwarg_unimplemented = 1;
        new = PyObject_Vectorcall(array_meth, arguments, nargs, NULL);
        if (new == NULL) {
            Py_DECREF(array_meth);
            return NULL;
        }
    }

    Py_DECREF(array_meth);

    if (!PyArray_Check(new)) {
        PyErr_SetString(PyExc_ValueError,
                        "object __array__ method not "  \
                        "producing an array");
        Py_DECREF(new);
        return NULL;
    }
    /* TODO: Remove was_copied_by__array__ argument */
    if (was_copied_by__array__ != NULL && copy == 1 &&
        must_copy_but_copy_kwarg_unimplemented == 0) {
        /* 我们可以假设进行了复制 */
        *was_copied_by__array__ = 1;
    }

    return new;
/*NUMPY_API
 */
/*NUMPY_API接口*/

NPY_NO_EXPORT PyObject *
/*不导出到外部的函数，返回一个PyObject对象*/

PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode, PyObject *context)
/*从数组属性创建一个PyArray对象的函数*/

{
    if (context != NULL) {
        /*如果context不为空，则抛出一个运行时错误*/
        PyErr_SetString(PyExc_RuntimeError, "'context' must be NULL");
        return NULL;
    }

    return PyArray_FromArrayAttr_int(op, typecode, 0, NULL);
    /*调用PyArray_FromArrayAttr_int函数，传入参数op、typecode、0和NULL，并返回其结果*/
}


/*NUMPY_API
* new reference -- accepts NULL for mintype
*/
/*NUMPY_API接口，新引用，可以接受NULL作为mintype*/

NPY_NO_EXPORT PyArray_Descr *
/*不导出到外部的函数，返回一个PyArray_Descr指针*/

PyArray_DescrFromObject(PyObject *op, PyArray_Descr *mintype)
/*从Python对象创建一个PyArray_Descr对象*/

{
    PyArray_Descr *dtype;

    dtype = mintype;
    /*将mintype赋值给dtype*/
    Py_XINCREF(dtype);
    /*增加dtype的引用计数*/

    if (PyArray_DTypeFromObject(op, NPY_MAXDIMS, &dtype) < 0) {
        /*如果PyArray_DTypeFromObject函数调用失败，则返回NULL*/
        return NULL;
    }

    if (dtype == NULL) {
        /*如果dtype为空，则返回默认类型的PyArray_Descr对象*/
        return PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    else {
        /*否则返回dtype*/
        return dtype;
    }
}


/*NUMPY_API
 * This is a quick wrapper around
 * PyArray_FromAny(op, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL)
 * that special cases Arrays and PyArray_Scalars up front
 * It *steals a reference* to the object
 * It also guarantees that the result is PyArray_Type
 * Because it decrefs op if any conversion needs to take place
 * so it can be used like PyArray_EnsureArray(some_function(...))
 */
/*NUMPY_API接口
 * 这是一个快速包装函数，围绕着
 * PyArray_FromAny(op, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL)
 * 特殊处理数组和PyArray_Scalar对象
 * 它“窃取引用”对象
 * 它还保证结果是PyArray_Type类型
 * 因为如果需要进行任何转换，它会解除op的引用
 * 所以可以像PyArray_EnsureArray(some_function(...))一样使用
 */
NPY_NO_EXPORT PyObject *
/*不导出到外部的函数，返回一个PyObject对象*/

PyArray_EnsureArray(PyObject *op)
/*确保返回一个PyArray对象的函数*/

{
    PyObject *new;

    if ((op == NULL) || (PyArray_CheckExact(op))) {
        /*如果op为空或者op是PyArray对象，则直接将op赋值给new，并增加其引用计数*/
        new = op;
        Py_XINCREF(new);
    }
    else if (PyArray_Check(op)) {
        /*如果op是PyArray对象，则调用PyArray_View函数创建一个视图对象赋给new*/
        new = PyArray_View((PyArrayObject *)op, NULL, &PyArray_Type);
    }
    else if (PyArray_IsScalar(op, Generic)) {
        /*如果op是标量对象，则调用PyArray_FromScalar函数创建一个数组对象赋给new*/
        new = PyArray_FromScalar(op, NULL);
    }
    else {
        /*否则调用PyArray_FROM_OF函数创建一个PyArray对象赋给new*/
        new = PyArray_FROM_OF(op, NPY_ARRAY_ENSUREARRAY);
    }
    Py_XDECREF(op);
    /*解除op的引用*/
    return new;
}


/*NUMPY_API*/
/*NUMPY_API接口*/

NPY_NO_EXPORT PyObject *
/*不导出到外部的函数，返回一个PyObject对象*/

PyArray_EnsureAnyArray(PyObject *op)
/*确保返回任何PyArray对象的函数*/

{
    if (op && PyArray_Check(op)) {
        /*如果op不为空且是PyArray对象，则直接返回op*/
        return op;
    }
    return PyArray_EnsureArray(op);
    /*否则调用PyArray_EnsureArray函数处理op并返回结果*/
}


/*
 * Private implementation of PyArray_CopyAnyInto with an additional order
 * parameter.
 */
/*
 * PyArray_CopyAnyInto的私有实现，带有额外的order参数
 */
NPY_NO_EXPORT int
/*不导出到外部的函数，返回一个整数*/

PyArray_CopyAsFlat(PyArrayObject *dst, PyArrayObject *src, NPY_ORDER order)
/*以扁平方式复制数组的函数*/

{
    NpyIter *dst_iter, *src_iter;

    NpyIter_IterNextFunc *dst_iternext, *src_iternext;
    char **dst_dataptr, **src_dataptr;
    npy_intp dst_stride, src_stride;
    npy_intp *dst_countptr, *src_countptr;
    npy_uint32 baseflags;

    npy_intp dst_count, src_count, count;
    npy_intp dst_size, src_size;
    int needs_api;

    NPY_BEGIN_THREADS_DEF;

    if (PyArray_FailUnlessWriteable(dst, "destination array") < 0) {
        /*如果目标数组dst不可写，则返回-1*/
        return -1;
    }

    /*
     * If the shapes match and a particular order is forced
     * for both, use the more efficient CopyInto
     */
    /*
     * 如果形状匹配，并且强制使用特定的顺序，
     * 则使用更有效的CopyInto函数
     */
    if (order != NPY_ANYORDER && order != NPY_KEEPORDER &&
            PyArray_NDIM(dst) == PyArray_NDIM(src) &&
            PyArray_CompareLists(PyArray_DIMS(dst), PyArray_DIMS(src),
                                PyArray_NDIM(dst))) {
        /*如果order不是NPY_ANYORDER和NPY_KEEPORDER，并且dst和src的维度和形状相同，则调用PyArray_CopyInto函数*/
        return PyArray_CopyInto(dst, src);
    }

    dst_size = PyArray_SIZE(dst);
    /*计算目标数组dst的总元素个数*/
    src_size = PyArray_SIZE(src);
    /*计算源数组src的总元素个数*/
    # 如果目标数组大小与源数组大小不相等，则抛出值错误异常并返回-1
    if (dst_size != src_size) {
        PyErr_Format(PyExc_ValueError,
                "cannot copy from array of size %" NPY_INTP_FMT " into an array "
                "of size %" NPY_INTP_FMT, src_size, dst_size);
        return -1;
    }

    # 对于大小为零的数组，不需要执行任何操作，直接返回0
    /* Zero-sized arrays require nothing be done */
    if (dst_size == 0) {
        return 0;
    }

    # 定义基础迭代器标志，用于配置迭代器的行为
    baseflags = NPY_ITER_EXTERNAL_LOOP |
                NPY_ITER_DONT_NEGATE_STRIDES |
                NPY_ITER_REFS_OK;

    """
     * 基于匹配 src 和 dst 的 C 顺序遍历进行复制。
     * 使用两个迭代器，可以找到能够一次处理的最大子块。
     """
    # 创建写迭代器以便写入目标数组，基础标志为写入模式和基础标志
    dst_iter = NpyIter_New(dst, NPY_ITER_WRITEONLY | baseflags,
                                order,
                                NPY_NO_CASTING,
                                NULL);
    if (dst_iter == NULL) {
        return -1;
    }
    # 创建只读迭代器以便读取源数组，基础标志为只读模式和基础标志
    src_iter = NpyIter_New(src, NPY_ITER_READONLY | baseflags,
                                order,
                                NPY_NO_CASTING,
                                NULL);
    if (src_iter == NULL) {
        NpyIter_Deallocate(dst_iter);
        return -1;
    }

    # 获取内部循环的迭代函数和数据指针数组，缓冲区被禁用，可以缓存步幅
    dst_iternext = NpyIter_GetIterNext(dst_iter, NULL);
    dst_dataptr = NpyIter_GetDataPtrArray(dst_iter);
    dst_stride = NpyIter_GetInnerStrideArray(dst_iter)[0];
    dst_countptr = NpyIter_GetInnerLoopSizePtr(dst_iter);

    src_iternext = NpyIter_GetIterNext(src_iter, NULL);
    src_dataptr = NpyIter_GetDataPtrArray(src_iter);
    src_stride = NpyIter_GetInnerStrideArray(src_iter)[0];
    src_countptr = NpyIter_GetInnerLoopSizePtr(src_iter);

    # 如果获取迭代函数或数据指针数组失败，则释放迭代器并返回-1
    if (dst_iternext == NULL || src_iternext == NULL) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }

    # 检查是否需要使用 Python C API
    needs_api = NpyIter_IterationNeedsAPI(dst_iter) ||
                NpyIter_IterationNeedsAPI(src_iter);

    """
     * 因为迭代器中禁用了缓冲区，所以在整个迭代过程中内部循环的步幅将保持不变。
     * 因此，可以将它们传递给此函数，以利用连续的步幅等优势。
     """
    # 获取数据类型转换的信息和数组方法标志
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(
                    IsUintAligned(src) && IsAligned(src) &&
                    IsUintAligned(dst) && IsAligned(dst),
                    src_stride, dst_stride,
                    PyArray_DESCR(src), PyArray_DESCR(dst),
                    0,
                    &cast_info, &flags) != NPY_SUCCEED) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }
    # 检查是否需要 Python C API
    needs_api |= (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    // 如果 flags 中未设置 NPY_METH_NO_FLOATINGPOINT_ERRORS 标志位，则清除浮点错误状态栏
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)src_iter);
    }
    // 如果不需要使用 API，则启动线程
    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    // 获取目标和源的元素个数
    dst_count = *dst_countptr;
    src_count = *src_countptr;
    // 设置参数数组和步长数组
    char *args[2] = {src_dataptr[0], dst_dataptr[0]};
    npy_intp strides[2] = {src_stride, dst_stride};

    int res = 0;
    for (;;) {
        /* Transfer the biggest amount that fits both */
        // 计算当前循环中可以处理的最大元素个数
        count = (src_count < dst_count) ? src_count : dst_count;
        // 调用类型转换函数进行数据转换
        if (cast_info.func(&cast_info.context,
                args, &count, strides, cast_info.auxdata) < 0) {
            res = -1;
            break;
        }

        /* If we exhausted the dst block, refresh it */
        // 如果目标块已经处理完毕，则刷新目标迭代器
        if (dst_count == count) {
            res = dst_iternext(dst_iter);
            if (res == 0) {
                break;
            }
            dst_count = *dst_countptr;
            args[1] = dst_dataptr[0];
        }
        else {
            // 更新目标块中剩余元素个数及指针位置
            dst_count -= count;
            args[1] += count * dst_stride;
        }

        /* If we exhausted the src block, refresh it */
        // 如果源块已经处理完毕，则刷新源迭代器
        if (src_count == count) {
            res = src_iternext(src_iter);
            if (res == 0) {
                break;
            }
            src_count = *src_countptr;
            args[0] = src_dataptr[0];
        }
        else {
            // 更新源块中剩余元素个数及指针位置
            src_count -= count;
            args[0] += count * src_stride;
        }
    }

    // 结束多线程操作
    NPY_END_THREADS;

    // 释放类型转换信息结构体
    NPY_cast_info_xfree(&cast_info);
    // 释放目标迭代器，如果失败则设置 res 为 -1
    if (!NpyIter_Deallocate(dst_iter)) {
        res = -1;
    }
    // 释放源迭代器，如果失败则设置 res 为 -1
    if (!NpyIter_Deallocate(src_iter)) {
        res = -1;
    }

    // 如果处理成功且 flags 中未设置 NPY_METH_NO_FLOATINGPOINT_ERRORS 标志位，则检查浮点错误状态栏
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        // 获取浮点错误状态栏
        int fpes = npy_get_floatstatus_barrier((char *)&src_iter);
        // 如果存在浮点错误且处理失败，则返回 -1
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    // 返回处理结果
    return res;
/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap
 * Does not require src and dest to have "broadcastable" shapes
 * (only the same number of elements).
 *
 * TODO: For NumPy 2.0, this could accept an order parameter which
 *       only allows NPY_CORDER and NPY_FORDER.  Could also rename
 *       this to CopyAsFlat to make the name more intuitive.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dst, PyArrayObject *src)
{
    // 调用 PyArray_CopyAsFlat 函数，将 src 数组的内容复制到 dst 数组中，使用 C 记录顺序
    return PyArray_CopyAsFlat(dst, src, NPY_CORDER);
}

/*NUMPY_API
 * Copy an Array into another array.
 * Broadcast to the destination shape if necessary.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_CopyInto(PyArrayObject *dst, PyArrayObject *src)
{
    // 调用 PyArray_AssignArray 函数，将 src 数组的内容广播到 dst 数组中，使用不安全的类型转换
    return PyArray_AssignArray(dst, src, NULL, NPY_UNSAFE_CASTING);
}

/*NUMPY_API
 * PyArray_CheckAxis
 *
 * check that axis is valid
 * convert 0-d arrays to 1-d arrays
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags)
{
    PyObject *temp1, *temp2;
    int n = PyArray_NDIM(arr);

    if (*axis == NPY_RAVEL_AXIS || n == 0) {
        // 如果 axis 是 NPY_RAVEL_AXIS 或者数组是零维的，则进行处理
        if (n != 1) {
            // 将数组展平成一维数组
            temp1 = PyArray_Ravel(arr, 0);
            if (temp1 == NULL) {
                // 如果展平失败，则将 axis 设置为 0 并返回 NULL
                *axis = 0;
                return NULL;
            }
            if (*axis == NPY_RAVEL_AXIS) {
                // 如果原始 axis 是 NPY_RAVEL_AXIS，则将 axis 更新为展平后数组的维度减一
                *axis = PyArray_NDIM((PyArrayObject *)temp1) - 1;
            }
        } else {
            // 如果数组已经是一维的，则直接使用 arr
            temp1 = (PyObject *)arr;
            Py_INCREF(temp1);
            *axis = 0;
        }
        // 如果 flags 为假且 axis 为 0，则返回 temp1
        if (!flags && *axis == 0) {
            return temp1;
        }
    } else {
        // 如果 axis 不是 NPY_RAVEL_AXIS 或者数组不是零维的，则直接使用 arr
        temp1 = (PyObject *)arr;
        Py_INCREF(temp1);
    }
    // 如果 flags 为真，则检查并从任意数组创建一个新的数组
    if (flags) {
        temp2 = PyArray_CheckFromAny((PyObject *)temp1, NULL,
                                     0, 0, flags, NULL);
        Py_DECREF(temp1);
        if (temp2 == NULL) {
            return NULL;
        }
    } else {
        temp2 = (PyObject *)temp1;
    }
    // 检查并调整 axis 的有效性，并返回 temp2
    n = PyArray_NDIM((PyArrayObject *)temp2);
    if (check_and_adjust_axis(axis, n) < 0) {
        Py_DECREF(temp2);
        return NULL;
    }
    return temp2;
}


/*NUMPY_API
 * Zeros
 *
 * steals a reference to type. On failure or when PyDataType_SUBARRAY(dtype) is
 * true, dtype will be decrefed.
 * accepts NULL type
 */
NPY_NO_EXPORT PyObject *
PyArray_Zeros(int nd, npy_intp const *dims, PyArray_Descr *type, int is_f_order)
{
    npy_dtype_info dt_info = {NULL, NULL};

    // 从 type 中提取 dtype 和描述符，并保存到 dt_info 中
    int res = PyArray_ExtractDTypeAndDescriptor(
        type, &dt_info.descr, &dt_info.dtype);

    // steal reference
    // 释放 type 的引用
    Py_XDECREF(type);

    if (res < 0) {
        // 如果提取失败，则释放相关资源并返回 NULL
        Py_XDECREF(dt_info.descr);
        Py_XDECREF(dt_info.dtype);
        return NULL;
    }

    // 调用 PyArray_Zeros_int 函数创建一个全零数组，并返回结果
    PyObject *ret = PyArray_Zeros_int(nd, dims, dt_info.descr, dt_info.dtype,
                                      is_f_order);

    // 释放描述符和 dtype 的引用
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);

    return ret;
}
/*
 *  接受 dtypemeta 的 PyArray_Zeros 内部版本。
 *  借用描述符和 dtype 的引用。
 */
NPY_NO_EXPORT PyObject *
PyArray_Zeros_int(int nd, npy_intp const *dims, PyArray_Descr *descr,
                  PyArray_DTypeMeta *dtype, int is_f_order)
{
    PyObject *ret = NULL;

    // 如果描述符为空，根据 dtype 推断描述符
    if (descr == NULL) {
        descr = _infer_descr_from_dtype(dtype);
        // 推断失败则返回空
        if (descr == NULL) {
            return NULL;
        }
    }

    /*
     * PyArray_NewFromDescr_int 偷取对 descr 的引用，
     * 增加引用计数以便此函数的调用者可以清理 descr
     */
    Py_INCREF(descr);
    ret = PyArray_NewFromDescr_int(
            &PyArray_Type, descr,
            nd, dims, NULL, NULL,
            is_f_order, NULL, NULL,
            _NPY_ARRAY_ZEROED);

    return ret;
}


/*NUMPY_API
 * Empty
 *
 * 接受 NULL 类型
 * 偷取对类型的引用
 */
NPY_NO_EXPORT PyObject *
PyArray_Empty(int nd, npy_intp const *dims, PyArray_Descr *type, int is_f_order)
{
    npy_dtype_info dt_info = {NULL, NULL};

    // 提取数据类型和描述符，偷取对类型的引用
    int res = PyArray_ExtractDTypeAndDescriptor(
        type, &dt_info.descr, &dt_info.dtype);

    // 偷取引用
    Py_XDECREF(type);

    if (res < 0) {
        return NULL;
    }

    PyObject *ret = PyArray_Empty_int(
        nd, dims, dt_info.descr, dt_info.dtype, is_f_order);

    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return ret;
}

/*
 *  接受 dtypemeta 的 PyArray_Empty 内部版本。
 *  借用描述符和 dtype 的引用。
 */
NPY_NO_EXPORT PyObject *
PyArray_Empty_int(int nd, npy_intp const *dims, PyArray_Descr *descr,
                  PyArray_DTypeMeta *dtype, int is_f_order)
{
    PyArrayObject *ret;

    // 如果描述符为空，根据 dtype 推断描述符
    if (descr == NULL) {
        descr = _infer_descr_from_dtype(dtype);
        // 推断失败则返回空
        if (descr == NULL) {
            return NULL;
        }
    }

    /*
     * PyArray_NewFromDescr 偷取对 descr 的引用，
     * 增加引用计数以便此函数的调用者可以清理 descr
     */
    Py_INCREF(descr);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                descr, nd, dims,
                                                NULL, NULL,
                                                is_f_order, NULL);
    if (ret == NULL) {
        return NULL;
    }

    /* empty、empty_like 和 ndarray.__new__ 共享的逻辑 */
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
        if (PyArray_SetObjectsToNone(ret) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    return (PyObject *)ret;
}

/*
 *  类似于 ceil(value)，但检查溢出。
 *
 *  成功返回 0，失败返回 -1。失败时设置 PyExc_Overflow 异常。
 */
static npy_intp
_arange_safe_ceil_to_intp(double value)
{
    double ivalue;

    ivalue = npy_ceil(value);
    /* 条件反转以处理 NaN */
    # 检查 ivalue 是否为 NaN（Not a Number）
    if (npy_isnan(ivalue)) {
        # 如果是 NaN，则设置一个 ValueError 异常，并返回 -1 表示错误
        PyErr_SetString(PyExc_ValueError,
            "arange: cannot compute length");
        return -1;
    }
    # 检查 ivalue 是否在 NPY_MIN_INTP 和 NPY_MAX_INTP 之间的双精度浮点数范围内
    if (!((double)NPY_MIN_INTP <= ivalue && ivalue <= (double)NPY_MAX_INTP)) {
        # 如果超出范围，则设置一个 OverflowError 异常，并返回 -1 表示错误
        PyErr_SetString(PyExc_OverflowError,
                "arange: overflow while computing length");
        return -1;
    }

    # 将 ivalue 强制转换为 npy_intp 类型，并返回该值
    return (npy_intp)ivalue;
/*NUMPY_API
  Arange,
*/
NPY_NO_EXPORT PyObject *
PyArray_Arange(double start, double stop, double step, int type_num)
{
    npy_intp length;  /* 定义一个整数型变量，用于存储数组的长度 */
    PyArrayObject *range;  /* 定义一个PyArrayObject类型的指针，用于表示生成的数组对象 */
    PyArray_ArrFuncs *funcs;  /* 定义一个PyArray_ArrFuncs类型的指针，用于表示数组的函数集 */
    PyObject *obj;  /* 定义一个PyObject类型的指针，用于临时存储对象 */
    int ret;  /* 定义一个整数型变量，用于存储函数调用的返回值 */
    double delta, tmp_len;  /* 定义两个双精度浮点型变量，用于存储计算中间结果 */
    NPY_BEGIN_THREADS_DEF;  /* 定义一个宏，用于开启线程 */

    delta = stop - start;  /* 计算起始点与结束点的差值 */
    tmp_len = delta / step;  /* 计算步长对应的长度 */

    /* 检查长度是否为零，避免浮点数计算误差 */
    if (tmp_len == 0.0 && delta != 0.0) {
        if (npy_signbit(tmp_len)) {  /* 如果长度为零且为负数 */
            length = 0;  /* 将长度设为零 */
        } else {
            length = 1;  /* 否则将长度设为一 */
        }
    } else {
        length = _arange_safe_ceil_to_intp(tmp_len);  /* 否则根据安全的ceil函数将长度转换为整数 */
        if (error_converting(length)) {  /* 如果在转换过程中出现错误 */
            return NULL;  /* 返回空指针 */
        }
    }

    if (length <= 0) {  /* 如果长度小于等于零 */
        length = 0;  /* 将长度设为零 */
        return PyArray_New(&PyArray_Type, 1, &length, type_num,
                           NULL, NULL, 0, 0, NULL);  /* 创建一个空的PyArray对象并返回 */
    }
    range = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &length, type_num,
                                         NULL, NULL, 0, 0, NULL);  /* 创建指定长度的PyArray对象 */
    if (range == NULL) {  /* 如果创建失败 */
        return NULL;  /* 返回空指针 */
    }
    funcs = PyDataType_GetArrFuncs(PyArray_DESCR(range));  /* 获取数组描述符对应的函数集 */

    /*
     * 将起始值放入缓冲区，并将下一个值放入第二个位置
     * 如果长度大于2，则调用内部循环，否则停止
     */
    obj = PyFloat_FromDouble(start);  /* 创建起始值的Python浮点对象 */
    ret = funcs->setitem(obj, PyArray_DATA(range), range);  /* 将起始值放入数组对象的数据区 */
    Py_DECREF(obj);  /* 释放Python对象的引用计数 */
    if (ret < 0) {  /* 如果设置操作失败 */
        goto fail;  /* 跳转到错误处理的标签 */
    }
    if (length == 1) {  /* 如果长度为1 */
        return (PyObject *)range;  /* 直接返回数组对象 */
    }
    obj = PyFloat_FromDouble(start + step);  /* 创建下一个值的Python浮点对象 */
    ret = funcs->setitem(obj, PyArray_BYTES(range) + PyArray_ITEMSIZE(range),
                         range);  /* 将下一个值放入数组对象的数据区的第二个位置 */
    Py_DECREF(obj);  /* 释放Python对象的引用计数 */
    if (ret < 0) {  /* 如果设置操作失败 */
        goto fail;  /* 跳转到错误处理的标签 */
    }
    if (length == 2) {  /* 如果长度为2 */
        return (PyObject *)range;  /* 直接返回数组对象 */
    }
    if (!funcs->fill) {  /* 如果填充函数不存在 */
        PyErr_SetString(PyExc_ValueError,
                        "no fill-function for data-type.");  /* 设置错误消息 */
        Py_DECREF(range);  /* 释放数组对象的引用计数 */
        return NULL;  /* 返回空指针 */
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(range));  /* 开始使用指定描述符的线程 */
    funcs->fill(PyArray_DATA(range), length, range);  /* 填充数组对象的数据区 */
    NPY_END_THREADS;  /* 结束线程 */
    if (PyErr_Occurred()) {  /* 如果发生了Python错误 */
        goto fail;  /* 跳转到错误处理的标签 */
    }
    return (PyObject *)range;  /* 返回填充后的数组对象 */

 fail:
    Py_DECREF(range);  /* 释放数组对象的引用计数 */
    return NULL;  /* 返回空指针，表示失败 */
}

/*
 * the formula is len = (intp) ceil((stop - start) / step);
 */
static npy_intp
_calc_length(PyObject *start, PyObject *stop, PyObject *step, PyObject **next, int cmplx)
{
    npy_intp len, tmp;  /* 定义两个整数型变量，用于存储计算结果 */
    PyObject *zero, *val;  /* 定义两个Python对象指针，用于存储常量和临时变量 */
    int next_is_nonzero, val_is_zero;  /* 定义两个整数型变量，用于表示逻辑判断结果 */
    double value;  /* 定义一个双精度浮点型变量，用于存储计算中间结果 */

    *next = PyNumber_Subtract(stop, start);  /* 计算停止值与起始值的差，并存储在指定的Python对象中 */
    if (!(*next)) {  /* 如果计算结果为空 */
        if (PyTuple_Check(stop)) {  /* 如果停止值是元组 */
            PyErr_Clear();  /* 清除之前的错误信息 */
            PyErr_SetString(PyExc_TypeError,
                            "arange: scalar arguments expected "\
                            "instead of a tuple.");  /* 设置类型错误的错误消息 */
        }
        return -1;  /* 返回-1，表示计算失败 */
    }

    zero = PyLong_FromLong(0);  /* 创建一个值为0的Python长整型对象 */
    if (!zero) {  /* 如果创建失败 */
        Py_DECREF(*next);  /* 释放next指向的对象的引用计数 */
        *next = NULL;  /* 将next指针设为NULL */
        return -1;  /* 返回-1，表示失败 */
    }

    next_is_nonzero = PyObject_RichCompareBool(*next, zero, Py_NE);  /* 检查next指向的对象是否不等于0 */


    if (!(*next)) {  /* 如果计算结果为空 */
        if (PyTuple_Check(stop)) {  /* 如果停止值是元组 */
            PyErr_Clear();  /* 清除之前的错误信息 */
            PyErr_SetString(PyExc_TypeError,
                            "arange: scalar arguments expected "\
                            "instead of a tuple.");  /* 设置类型错误的错误消息 */
        }
        return -1;  /* 返回-1，表示计算失败 */
    }

    zero = PyLong_FromLong(0);  /* 创建一个值为0的Python长整型对象 */
    if (!zero) {  /* 如果创建失败 */
        Py_DECREF(*next);  /* 释放next指向的对象的引用计数 */
        *next = NULL;  /* 将next指针设为NULL */
        return -1;  /* 返回-1，表示失败 */
    }

    next_is_nonzero = PyObject_RichCompareBool(*next, zero, Py_NE);  /* 检查next指向的对象是否不等于0 */
    val_is_zero = PyObject_RichCompareBool(step, zero, Py_EQ);  /* 检查步长是否等于0 */

    if (next_is_nonzero && val_is_zero) {  /* 如果计算结果不为0且步长为0 */
        PyErr_SetString(PyExc_ValueError, "step cannot be zero");  /* 抛出步长不能为0的错误 */
        Py_DECREF(zero);  /* 释放zero指向的对象的引用计数 */
        Py_DECREF(*next);  /* 释放next指向的对象的引用计数 */
        return -1;  /* 返回-1，表示失败 */
    }

    if (cmplx) {  /* 如果cmplx为真 */
        /* Integer division will give correct results here. */
        *next = PyNumber_FloorDivide(*next, step);  /* 计算next指向的对象与步长的整数除法 */
        if (!(*next)) {  /* 如果计算结果为空 */
            PyErr_SetString(PyExc_ValueError,
                            "arange: overflow while computing length");  /* 抛出计算长度时溢出的错误 */
            Py_DECREF(zero);  /* 释放zero指向的对象的引用计数 */
            return -1;  /* 返回-1，表示失败 */
        }
        tmp = PyLong_AsLongLong(*next);  /* 将计算结果转换为长长整型 */
        if (error_converting(tmp)) {  /* 如果转换失败 */
            PyErr_SetString(PyExc_ValueError,
                            "arange: overflow while computing length");  /* 抛出计算长度时溢出的错误 */
            Py_DECREF(zero);  /* 释放zero指向的对象的引用计数 */
            return -1;  /* 返回-1，表示失败 */
        }
        len = (npy_intp)tmp;  /* 将长长整型转换为整数型 */
    } else {  /* 否则 */
        value = PyFloat_AsDouble(*next);  /* 将next指向的对象转换为双精度浮点型 */
        if (value < 0) {  /* 如果值小于0 */
            len = 0;  /* 将长度设为0 */
        } else {  /* 否则 */
            len = (npy_intp)ceil(value);  /* 将值向上取整并转换为整数型 */
        }
    }

    Py_DECREF(zero);  /* 释放zero指向的对象的引用计数 */
    return len;  /* 返回计算得到的长度 */
}
    # 如果 next_is_nonzero 等于 -1，表示出现错误：
    if (next_is_nonzero == -1) {
        # 减少对 zero 的引用计数，因为不再需要
        Py_DECREF(zero);
        # 减少对 *next 的引用计数，因为不再需要
        Py_DECREF(*next);
        # 将 *next 置为 NULL，指示无效状态
        *next = NULL;
        # 返回 -1 表示错误状态
        return -1;
    }
    
    # 对 *next 进行真实除法，得到结果 val
    val = PyNumber_TrueDivide(*next, step);
    # 减少对 *next 的引用计数，因为不再需要
    Py_DECREF(*next);
    # 将 *next 置为 NULL，指示无效状态
    *next = NULL;

    # 如果 val 为假值（NULL），表示出现错误：
    if (!val) {
        # 减少对 zero 的引用计数，因为不再需要
        Py_DECREF(zero);
        # 返回 -1 表示错误状态
        return -1;
    }

    # 检查 val 是否等于 zero
    val_is_zero = PyObject_RichCompareBool(val, zero, Py_EQ);
    # 减少对 zero 的引用计数，因为不再需要
    Py_DECREF(zero);
    # 如果比较出错，返回 -1 表示错误状态
    if (val_is_zero == -1) {
        # 减少对 val 的引用计数，因为不再需要
        Py_DECREF(val);
        # 返回 -1 表示错误状态
        return -1;
    }

    # 如果需要复数计算，并且 val 是复数类型：
    if (cmplx && PyComplex_Check(val)) {
        # 获取复数 val 的实部值
        value = PyComplex_RealAsDouble(val);
        # 如果转换出错，减少对 val 的引用计数，返回 -1 表示错误状态
        if (error_converting(value)) {
            Py_DECREF(val);
            return -1;
        }
        # 将实部值转换为安全整数值 len
        len = _arange_safe_ceil_to_intp(value);
        # 如果转换出错，返回 -1 表示错误状态
        if (error_converting(len)) {
            Py_DECREF(val);
            return -1;
        }
        # 获取复数 val 的虚部值
        value = PyComplex_ImagAsDouble(val);
        # 减少对 val 的引用计数，因为不再需要
        Py_DECREF(val);
        # 如果转换出错，返回 -1 表示错误状态
        if (error_converting(value)) {
            return -1;
        }
        # 将虚部值转换为安全整数值 tmp
        tmp = _arange_safe_ceil_to_intp(value);
        # 如果转换出错，返回 -1 表示错误状态
        if (error_converting(tmp)) {
            return -1;
        }
        # 取 len 和 tmp 中的较小值作为最终的 len
        len = PyArray_MIN(len, tmp);
    }
    else {
        # 获取浮点数 val 的双精度值
        value = PyFloat_AsDouble(val);
        # 减少对 val 的引用计数，因为不再需要
        Py_DECREF(val);
        # 如果转换出错，返回 -1 表示错误状态
        if (error_converting(value)) {
            return -1;
        }

        /* 下溢和除以无穷大检查 */
        # 如果 val 是零，并且 next_is_nonzero 不为零：
        if (val_is_zero && next_is_nonzero) {
            # 如果 value 是负数，则 len 为 0，否则为 1
            if (npy_signbit(value)) {
                len = 0;
            }
            else {
                len = 1;
            }
        }
        else {
            # 将浮点数值 value 转换为安全整数值 len
            len = _arange_safe_ceil_to_intp(value);
            # 如果转换出错，返回 -1 表示错误状态
            if (error_converting(len)) {
                return -1;
            }
        }
    }

    # 如果 len 大于 0：
    if (len > 0) {
        # 计算 start + step 的值作为 *next
        *next = PyNumber_Add(start, step);
        # 如果 *next 为空，返回 -1 表示错误状态
        if (!*next) {
            return -1;
        }
    }
    # 返回 len 作为成功计算的长度值
    return len;
}

/*NUMPY_API
 *
 * ArangeObj,
 *
 * this doesn't change the references
 */
NPY_NO_EXPORT PyObject *
PyArray_ArangeObj(PyObject *start, PyObject *stop, PyObject *step, PyArray_Descr *dtype)
{
    PyArrayObject *range = NULL;
    PyArray_ArrFuncs *funcs;
    PyObject *next = NULL;
    PyArray_Descr *native = NULL;
    npy_intp length;
    int swap;
    NPY_BEGIN_THREADS_DEF;

    /* Datetime arange is handled specially */
    // 如果 dtype 是日期时间或时间增量类型，或者 start/stop/step 中有任何一个是日期时间或时间增量类型，则调用特殊处理函数 datetime_arange
    if ((dtype != NULL && (dtype->type_num == NPY_DATETIME ||
                           dtype->type_num == NPY_TIMEDELTA)) ||
            (dtype == NULL && (is_any_numpy_datetime_or_timedelta(start) ||
                              is_any_numpy_datetime_or_timedelta(stop) ||
                              is_any_numpy_datetime_or_timedelta(step)))) {
        return (PyObject *)datetime_arange(start, stop, step, dtype);
    }

    /* We need to replace many of these, so hold on for easier cleanup */
    // 增加对象的引用计数，以便后续容易进行清理
    Py_XINCREF(start);
    Py_XINCREF(stop);
    Py_XINCREF(step);
    Py_XINCREF(dtype);

    if (!dtype) {
        /* intentionally made to be at least NPY_LONG */
        // 如果没有指定 dtype，则默认为 NPY_INTP 类型，并根据 start/stop/step 中的对象类型进行调整
        dtype = PyArray_DescrFromType(NPY_INTP);
        Py_SETREF(dtype, PyArray_DescrFromObject(start, dtype));
        if (dtype == NULL) {
            goto fail;
        }
        if (stop && stop != Py_None) {
            Py_SETREF(dtype, PyArray_DescrFromObject(stop, dtype));
            if (dtype == NULL) {
                goto fail;
            }
        }
        if (step && step != Py_None) {
            Py_SETREF(dtype, PyArray_DescrFromObject(step, dtype));
            if (dtype == NULL) {
                goto fail;
            }
        }
    }

    /*
     * If dtype is not in native byte-order then get native-byte
     * order version.  And then swap on the way out.
     */
    // 如果 dtype 不是本机字节顺序，则获取本机字节顺序的版本，并在输出时进行字节交换
    if (!PyArray_ISNBO(dtype->byteorder)) {
        native = PyArray_DescrNewByteorder(dtype, NPY_NATBYTE);
        if (native == NULL) {
            goto fail;
        }
        swap = 1;
    }
    else {
        Py_INCREF(dtype);
        native = dtype;
        swap = 0;
    }

    funcs = PyDataType_GetArrFuncs(native);
    if (!funcs->fill) {
        /* This effectively forbids subarray types as well... */
        // 如果函数指针 funcs->fill 为空，则报错，不支持 arange() 对此类型的输入
        PyErr_Format(PyExc_TypeError,
                "arange() not supported for inputs with DType %S.",
                Py_TYPE(dtype));
        goto fail;
    }

    if (!step || step == Py_None) {
        // 如果 step 未指定或者为 None，则将 step 设置为 1
        Py_XSETREF(step, PyLong_FromLong(1));
        if (step == NULL) {
            goto fail;
        }
    }
    if (!stop || stop == Py_None) {
        // 如果 stop 未指定或者为 None，则将 stop 设置为 start，同时将 start 设置为 0
        Py_XSETREF(stop, start);
        start = PyLong_FromLong(0);
        if (start == NULL) {
            goto fail;
        }
    }

    /* calculate the length and next = start + step*/
    // 计算数组的长度，并计算 next = start + step
    length = _calc_length(start, stop, step, &next,
                          PyTypeNum_ISCOMPLEX(dtype->type_num));
    PyObject *err = PyErr_Occurred();
    /*
     * 如果发生错误，检查错误类型是否为溢出错误，
     * 如果是，则设置一个新的 ValueError 异常，并返回失败
     */
    if (err) {
        if (PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed size exceeded");
        }
        goto fail;
    }
    
    /*
     * 如果 length 小于等于 0，则将其设置为 0
     */
    if (length <= 0) {
        length = 0;
    }

    // 增加对 native 对象的引用计数
    Py_INCREF(native);
    
    // 使用给定的 native 数据类型创建一个新的一维数组对象
    range = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, &length, native);
    if (range == NULL) {
        goto fail;
    }

    // 如果 length 为 0，则直接跳转到完成标签
    if (length == 0) {
        goto finish;
    }

    /*
     * 将 start 的值放入缓冲区的第一个位置，
     * 将 next 的值放入缓冲区的第二个位置，
     * 如果 length 大于 2，则调用内部循环，否则停止
     */
    if (funcs->setitem(start, PyArray_DATA(range), range) < 0) {
        goto fail;
    }
    if (length == 1) {
        goto finish;
    }
    if (funcs->setitem(next, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                       range) < 0) {
        goto fail;
    }
    if (length == 2) {
        goto finish;
    }

    // 在多线程环境中使用描述符的线程安全开始宏
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(range));
    // 使用 funcs 的 fill 函数填充缓冲区的数据
    funcs->fill(PyArray_DATA(range), length, range);
    // 结束多线程环境下的操作
    NPY_END_THREADS;

    // 检查是否发生 Python 异常
    if (PyErr_Occurred()) {
        goto fail;
    }

finish:
    /*
     * TODO: 这种交换可以通过 nditer 在运行时处理
     * 如果 swap 为真，则进行数组的字节顺序交换操作
     */
    if (swap) {
        PyObject *new;
        // 对数组进行字节顺序交换，并返回新的对象
        new = PyArray_Byteswap(range, 1);
        if (new == NULL) {
            goto fail;
        }
        Py_DECREF(new);
        // 在原地交换字节顺序后，替换数据类型描述符
        Py_DECREF(PyArray_DESCR(range));
        Py_INCREF(dtype);
        ((PyArrayObject_fields *)range)->descr = dtype;
    }

    // 释放 dtype 相关的引用计数
    Py_DECREF(dtype);
    Py_DECREF(native);
    Py_DECREF(start);
    Py_DECREF(stop);
    Py_DECREF(step);
    Py_XDECREF(next);
    // 返回成功创建的数组对象
    return (PyObject *)range;

fail:
    // 如果发生错误，释放所有相关的 Python 对象引用计数，并返回 NULL
    Py_XDECREF(dtype);
    Py_XDECREF(native);
    Py_XDECREF(start);
    Py_XDECREF(stop);
    Py_XDECREF(step);
    Py_XDECREF(next);
    Py_XDECREF(range);
    return NULL;
/* This function creates a NumPy array by reading binary data from a file stream.
   It does not take ownership of the dtype reference. */
static PyArrayObject *
array_fromfile_binary(FILE *fp, PyArray_Descr *dtype, npy_intp num, size_t *nread)
{
    PyArrayObject *r;       /* PyArrayObject pointer for the resulting array */
    npy_off_t start, numbytes;  /* Variables for file position and number of bytes */
    int elsize;              /* Size of each element in bytes */

    /* If num is negative, calculate the number of elements from file size */
    if (num < 0) {
        int fail = 0;
        start = npy_ftell(fp);  /* Get current file position */
        if (start < 0) {
            fail = 1;
        }
        if (npy_fseek(fp, 0, SEEK_END) < 0) {  /* Seek to end of file */
            fail = 1;
        }
        numbytes = npy_ftell(fp);  /* Get total size of the file */
        if (numbytes < 0) {
            fail = 1;
        }
        numbytes -= start;  /* Calculate bytes from start to end */
        if (npy_fseek(fp, start, SEEK_SET) < 0) {  /* Return to original position */
            fail = 1;
        }
        if (fail) {
            PyErr_SetString(PyExc_OSError,
                            "could not seek in file");  /* Raise error if seeking fails */
            return NULL;
        }
        num = numbytes / dtype->elsize;  /* Calculate number of elements */
    }

    /*
     * Array creation may move sub-array dimensions from the dtype to array
     * dimensions, so we need to use the original element size when reading.
     */
    elsize = dtype->elsize;  /* Retrieve element size */

    Py_INCREF(dtype);  /* Increase reference count to dtype (no ownership transfer) */

    /* Create a new NumPy array object */
    r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &num,
                                              NULL, NULL, 0, NULL);
    if (r == NULL) {
        return NULL;
    }

    /* Begin thread-safe section for file reading */
    NPY_BEGIN_ALLOW_THREADS;
    *nread = fread(PyArray_DATA(r), elsize, num, fp);  /* Read data into the array */
    NPY_END_ALLOW_THREADS;
    
    return r;  /* Return the created NumPy array */
}
    for (i = 0; num < 0 || i < num; i++) {
        // 循环读取数据，直到达到指定数量(num < 0表示无限读取)，或者读取完毕
        stop_reading_flag = next(&stream, dptr, dtype, stream_data);
        // 调用next函数读取下一个数据块，并返回状态标志
        if (stop_reading_flag < 0) {
            // 如果读取操作返回负值，表示出错或结束
            break;
        }
        // 增加已读取数据块的计数
        *nread += 1;
        // 更新当前缓冲区位置
        thisbuf += 1;
        // 移动数据指针到下一个数据块的位置
        dptr += dtype->elsize;
        // 如果未指定读取数量(num < 0)且当前缓冲区已满，则进行内存重分配
        if (num < 0 && thisbuf == size) {
            // 累加总字节数
            totalbytes += bytes;
            /* The handler is always valid */
            // 根据新的总字节数重新分配内存，更新数据指针
            tmp = PyDataMem_UserRENEW(PyArray_DATA(r), totalbytes,
                                  PyArray_HANDLER(r));
            // 如果内存重新分配失败
            if (tmp == NULL) {
                err = 1;
                break;
            }
            // 更新数据指针到新分配的内存位置
            ((PyArrayObject_fields *)r)->data = tmp;
            dptr = tmp + (totalbytes - bytes);
            // 重置当前缓冲区位置
            thisbuf = 0;
        }
        // 调用skip_sep函数跳过分隔符，并返回状态标志
        stop_reading_flag = skip_sep(&stream, clean_sep, stream_data);
        // 如果skip_sep函数返回负值，表示出错或结束
        if (stop_reading_flag < 0) {
            // 如果当前是按照请求读取数量(num == i + 1)，则可选择性地停止读取
            if (num == i + 1) {
                /* if we read as much as requested sep is optional */
                stop_reading_flag = -1;
            }
            break;
        }
    }
    // 如果读取数量为负数(num < 0)
    if (num < 0) {
        // 计算需要重新分配的内存大小
        const size_t nsize = PyArray_MAX(*nread,1)*dtype->elsize;

        // 如果需要重新分配的内存大小不为零
        if (nsize != 0) {
            /* The handler is always valid */
            // 根据新的内存大小重新分配内存，更新数据指针
            tmp = PyDataMem_UserRENEW(PyArray_DATA(r), nsize,
                                  PyArray_HANDLER(r));
            // 如果内存重新分配失败
            if (tmp == NULL) {
                err = 1;
            }
            else {
                // 更新数组对象的第一个维度大小
                PyArray_DIMS(r)[0] = *nread;
                // 更新数据指针到新分配的内存位置
                ((PyArrayObject_fields *)r)->data = tmp;
            }
        }
    }
    // 结束线程锁保护区域
    NPY_END_ALLOW_THREADS;

    // 释放动态分配的内存
    free(clean_sep);

    // 如果停止标志是-2
    if (stop_reading_flag == -2) {
        // 如果已经发生了错误异常，则直接释放数组对象并返回空
        if (PyErr_Occurred()) {
            /* If an error is already set (unlikely), do not create new one */
            Py_DECREF(r);
            return NULL;
        }
        // 输出警告信息，表明字符串或文件读取到结尾时有不匹配的数据
        /* 2019-09-12, NumPy 1.18 */
        if (DEPRECATE(
                "string or file could not be read to its end due to unmatched "
                "data; this will raise a ValueError in the future.") < 0) {
            goto fail;
        }
    }
fail:
    if (err == 1) {
        PyErr_NoMemory();
    }
    // 检查是否有 Python 异常发生
    if (PyErr_Occurred()) {
        // 出现异常时，减少返回对象的引用计数，并返回空指针
        Py_DECREF(r);
        return NULL;
    }
    // 返回处理后的对象
    return r;
}
#undef FROM_BUFFER_SIZE

/*NUMPY_API
 *
 * 给定一个 `FILE *` 指针 `fp` 和一个 `PyArray_Descr`，返回一个对应于文件中编码数据的数组。
 *
 * `dtype` 的引用被窃取（传入的 dtype 可能不会被保留）。
 *
 * 要读取的元素数为 `num`；如果 `num` < 0，则尽可能多地读取。
 *
 * 如果 `sep` 为 NULL 或空，则假定为二进制数据；否则为文本数据，其中 `sep` 是元素之间的分隔符。
 * 分隔符中的空白匹配文本中的任意长度的空白，并添加分隔符周围的空白的匹配。
 *
 * 对于内存映射文件，请使用缓冲区接口。此例程不会读取比必要更多的数据。
 */
NPY_NO_EXPORT PyObject *
PyArray_FromFile(FILE *fp, PyArray_Descr *dtype, npy_intp num, char *sep)
{
    PyArrayObject *ret;
    size_t nread = 0;

    if (dtype == NULL) {
        // 如果 dtype 为空，则返回空指针
        return NULL;
    }

    if (PyDataType_REFCHK(dtype)) {
        // 如果 dtype 是对象数组，则设置 ValueError，并释放 dtype 的引用
        PyErr_SetString(PyExc_ValueError,
                "Cannot read into object array");
        Py_DECREF(dtype);
        return NULL;
    }
    if (dtype->elsize == 0) {
        /* Nothing to read, just create an empty array of the requested type */
        // 没有需要读取的内容，只需创建请求类型的空数组
        return PyArray_NewFromDescr_int(
                &PyArray_Type, dtype,
                1, &num, NULL, NULL,
                0, NULL, NULL,
                _NPY_ARRAY_ALLOW_EMPTY_STRING);
    }
    if ((sep == NULL) || (strlen(sep) == 0)) {
        // 如果 sep 为 NULL 或空，则使用二进制数据读取数组
        ret = array_fromfile_binary(fp, dtype, num, &nread);
    }
    else {
        if (PyDataType_GetArrFuncs(dtype)->scanfunc == NULL) {
            // 如果 dtype 的扫描函数为空，则设置 ValueError，并释放 dtype 的引用
            PyErr_SetString(PyExc_ValueError,
                    "Unable to read character files of that array type");
            Py_DECREF(dtype);
            return NULL;
        }
        // 否则，使用文本数据读取数组
        ret = array_from_text(dtype, num, sep, &nread, fp,
                (next_element) fromfile_next_element,
                (skip_separator) fromfile_skip_separator, NULL);
    }
    if (ret == NULL) {
        // 如果返回的数组对象为空，则释放 dtype 的引用，并返回空指针
        Py_DECREF(dtype);
        return NULL;
    }
    if (((npy_intp) nread) < num) {
        /*
         * Realloc memory for smaller number of elements, use original dtype
         * which may have include a subarray (and is used for `nread`).
         */
        // 重新分配内存以适应较少的元素数，使用可能包含子数组的原始 dtype（用于 `nread`）。
        const size_t nsize = PyArray_MAX(nread,1) * dtype->elsize;
        char *tmp;

        /* The handler is always valid */
        // 处理程序始终有效
        if((tmp = PyDataMem_UserRENEW(PyArray_DATA(ret), nsize,
                                     PyArray_HANDLER(ret))) == NULL) {
            // 如果重新分配失败，则释放 dtype 和 ret 的引用，并返回内存错误
            Py_DECREF(dtype);
            Py_DECREF(ret);
            return PyErr_NoMemory();
        }
        ((PyArrayObject_fields *)ret)->data = tmp;
        PyArray_DIMS(ret)[0] = nread;
    }
    // 释放 dtype 的引用，并返回结果数组对象
    Py_DECREF(dtype);
    return (PyObject *)ret;
}

/*NUMPY_API*/
/*
 * 创建一个 NumPy 数组对象，从给定的缓冲区 `buf` 中构造，具有指定的数据类型 `type`、元素个数 `count` 和偏移量 `offset`。
 * 这个函数返回一个 PyArrayObject 指针。
 */
NPY_NO_EXPORT PyObject *
PyArray_FromBuffer(PyObject *buf, PyArray_Descr *type,
                   npy_intp count, npy_intp offset)
{
    PyArrayObject *ret;         // 返回的 NumPy 数组对象
    char *data;                 // 缓冲区数据的指针
    Py_buffer view;             // Python 缓冲区视图结构体
    Py_ssize_t ts;              // 缓冲区总大小
    npy_intp s, n;              // 剩余数据大小 `s` 和数组元素个数 `n`
    int itemsize;               // 元素大小
    int writeable = 1;          // 缓冲区是否可写，默认可写

    if (type == NULL) {
        return NULL;            // 如果数据类型为空，返回空指针
    }

    if (PyDataType_REFCHK(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create an OBJECT array from memory"\
                        " buffer");
        Py_DECREF(type);
        return NULL;            // 如果数据类型要求引用检查，则返回空指针
    }
    if (PyDataType_ISUNSIZED(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "itemsize cannot be zero in type");
        Py_DECREF(type);
        return NULL;            // 如果数据类型要求元素大小非零，则返回空指针
    }

    /*
     * 如果对象支持 `releasebuffer`，新的缓冲区协议允许将内存生命周期绑定到 `Py_buffer view`。
     * NumPy 不能直接持有 `view` 本身（因为它不是一个对象），所以必须将原始对象包装在一个 Python `memoryview` 中，它负责管理生命周期。
     * 为了向后兼容 `arr.base`，在可能的情况下尽量避免这样做。（例如，NumPy 数组永远不会在此处被包装！）
     */
    if (Py_TYPE(buf)->tp_as_buffer
            && Py_TYPE(buf)->tp_as_buffer->bf_releasebuffer) {
        buf = PyMemoryView_FromObject(buf);  // 将对象 `buf` 转换为 `memoryview`
        if (buf == NULL) {
            return NULL;    // 转换失败则返回空指针
        }
    }
    else {
        Py_INCREF(buf);     // 增加 `buf` 的引用计数
    }

    // 尝试获取缓冲区 `buf` 的视图 `view`，支持写入和简单模式
    if (PyObject_GetBuffer(buf, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        writeable = 0;      // 获取失败时，标记为不可写
        PyErr_Clear();      // 清除错误信息
        if (PyObject_GetBuffer(buf, &view, PyBUF_SIMPLE) < 0) {
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;    // 再次获取失败则返回空指针
        }
    }
    data = (char *)view.buf;    // 缓冲区数据的起始地址
    ts = view.len;              // 缓冲区的总长度
    /* `buf` 是一个数组或者内存视图；因此我们知道 `view` 不拥有数据 */
    PyBuffer_Release(&view);    // 释放 `view`

    // 如果偏移量 `offset` 小于 0 或大于缓冲区长度 `ts`，则抛出错误
    if ((offset < 0) || (offset > ts)) {
        PyErr_Format(PyExc_ValueError,
                     "offset must be non-negative and no greater than buffer "\
                     "length (%" NPY_INTP_FMT ")", (npy_intp)ts);
        Py_DECREF(buf);
        Py_DECREF(type);
        return NULL;    // 返回空指针
    }

    data += offset;             // 根据偏移量调整数据指针
    s = (npy_intp)ts - offset;  // 剩余数据大小
    n = (npy_intp)count;        // 数组元素个数
    itemsize = type->elsize;    // 元素大小

    // 如果 `n` 小于 0，则尝试根据缓冲区大小和元素大小计算元素个数
    if (n < 0) {
        if (itemsize == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot determine count if itemsize is 0");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;    // 返回空指针
        }
        if (s % itemsize != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer size must be a multiple"\
                            " of element size");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;    // 返回空指针
        }
        n = s/itemsize;     // 计算元素个数
    }
    else {
        // 如果条件不满足，则执行以下代码块
        if (s < n*itemsize) {
            // 如果缓冲区大小小于请求的大小，则设置异常并返回 NULL
            PyErr_SetString(PyExc_ValueError,
                            "buffer is smaller than requested"\
                            " size");
            // 减少 buf 对象的引用计数
            Py_DECREF(buf);
            // 减少 type 对象的引用计数
            Py_DECREF(type);
            // 返回 NULL
            return NULL;
        }
    }

    // 创建新的 PyArrayObject 对象，基于给定的描述符和数据
    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, type,
            1, &n, NULL, data,
            NPY_ARRAY_DEFAULT, NULL, buf);
    // 减少 buf 对象的引用计数
    Py_DECREF(buf);
    // 如果创建失败，则返回 NULL
    if (ret == NULL) {
        return NULL;
    }

    // 如果不可写，则清除 ret 对象的写入标志
    if (!writeable) {
        PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
    // 返回 PyArrayObject 对象的 PyObject* 指针
    return (PyObject *)ret;
/*NUMPY_API
 *
 * 给定一个指向字符串数据的指针 ``data``，字符串长度 ``slen``，和一个 ``PyArray_Descr``，
 * 返回一个对应于该字符串数据编码的数组。
 *
 * 如果 dtype 为 NULL，则使用默认的数组类型（double）。
 * 如果非空，则引用被窃取。
 *
 * 如果 ``slen`` 小于 0，则使用字符串的结尾作为文本数据。
 * 对于二进制数据，``slen`` 小于 0 是一个错误（因为嵌入的空值是正常的）。
 *
 * 要读取的元素数为 ``num``；如果它小于 0，则尽可能多地读取。
 *
 * 如果 ``sep`` 为 NULL 或空，则假定为二进制数据；否则假定为文本数据，并且 ``sep`` 作为分隔符。
 * 分隔符中的空白匹配文本中任意长度的空白，并在分隔符周围的空白处进行匹配。
 */
NPY_NO_EXPORT PyObject *
PyArray_FromString(char *data, npy_intp slen, PyArray_Descr *dtype,
                   npy_intp num, char *sep)
{
    int itemsize;               /* 元素大小 */
    PyArrayObject *ret;         /* 返回的数组对象 */
    npy_bool binary;            /* 是否为二进制数据 */

    if (dtype == NULL) {
        dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        if (dtype == NULL) {
            return NULL;        /* 返回空，表示类型转换失败 */
        }
    }
    if (PyDataType_FLAGCHK(dtype, NPY_ITEM_IS_POINTER) ||
                    PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot create an object array from"    \
                        " a string");
        Py_DECREF(dtype);
        return NULL;            /* 返回空，表示无法从字符串创建对象数组 */
    }
    itemsize = dtype->elsize;   /* 获取数组元素大小 */
    if (itemsize == 0) {
        PyErr_SetString(PyExc_ValueError, "zero-valued itemsize");
        Py_DECREF(dtype);
        return NULL;            /* 返回空，表示元素大小为零 */
    }

    binary = ((sep == NULL) || (strlen(sep) == 0));   /* 是否为二进制数据 */
    if (binary) {
        if (num < 0 ) {
            if (slen % itemsize != 0) {
                PyErr_SetString(PyExc_ValueError,
                                "string size must be a "\
                                "multiple of element size");
                Py_DECREF(dtype);
                return NULL;    /* 返回空，表示字符串大小不是元素大小的整数倍 */
            }
            num = slen / itemsize;   /* 计算需要读取的元素个数 */
        }
        else {
            if (slen < num * itemsize) {
                PyErr_SetString(PyExc_ValueError,
                                "string is smaller than " \
                                "requested size");
                Py_DECREF(dtype);
                return NULL;    /* 返回空，表示字符串长度小于请求的大小 */
            }
        }
        /*
         * NewFromDescr 可能会替换 dtype 以吸收子数组形状到数组中，
         * 因此事先获取大小。
         */
        npy_intp size_to_copy = num * dtype->elsize;   /* 待复制的数据大小 */
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type, dtype,
                                 1, &num, NULL, NULL,
                                 0, NULL);    /* 从描述符创建新数组对象 */
        if (ret == NULL) {
            return NULL;    /* 返回空，表示从描述符创建数组失败 */
        }
        memcpy(PyArray_DATA(ret), data, size_to_copy);   /* 复制数据到数组对象中 */
    }
    else {
        /* 从字符型字符串中读取数据 */

        /* 初始化已读取字符数为0 */
        size_t nread = 0;
        
        /* 指向字符串结尾的指针 */
        char *end;

        /* 如果数据类型的fromstr函数为空，抛出数值错误异常 */
        if (PyDataType_GetArrFuncs(dtype)->fromstr == NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "don't know how to read "       \
                            "character strings with that "  \
                            "array type");
            Py_DECREF(dtype);
            return NULL;
        }

        /* 如果字符串长度小于0，则结尾指针置空 */
        if (slen < 0) {
            end = NULL;
        }
        else {
            end = data + slen;  /* 否则，指向数据加上长度的位置 */
        }

        /* 调用array_from_text函数，将dtype、num、sep、nread等参数传递进去 */
        ret = array_from_text(dtype, num, sep, &nread,
                              data,
                              (next_element) fromstr_next_element,
                              (skip_separator) fromstr_skip_separator,
                              end);

        /* 释放Python对象的引用 */
        Py_DECREF(dtype);
    }

    /* 返回PyObject类型的ret指针 */
    return (PyObject *)ret;
}

/*NUMPY_API
 *
 * steals a reference to dtype (which cannot be NULL)
 */
NPY_NO_EXPORT PyObject *
PyArray_FromIter(PyObject *obj, PyArray_Descr *dtype, npy_intp count)
{
    PyObject *iter = NULL;  // 定义迭代器对象
    PyArrayObject *ret = NULL;  // 定义返回的 NumPy 数组对象
    npy_intp i, elsize, elcount;  // 定义整数变量 i、元素大小和元素个数

    if (dtype == NULL) {  // 检查数据类型是否为空，如果是则返回空
        return NULL;
    }

    iter = PyObject_GetIter(obj);  // 获取对象的迭代器
    if (iter == NULL) {  // 如果迭代器获取失败则跳转至结束标签
        goto done;
    }

    if (PyDataType_ISUNSIZED(dtype)) {
        /* If this error is removed, the `ret` allocation may need fixing */
        PyErr_SetString(PyExc_ValueError,
                "Must specify length when using variable-size data-type.");
        goto done;  // 如果数据类型是可变大小的，则设置错误信息并跳转至结束标签
    }
    if (count < 0) {
        elcount = PyObject_LengthHint(obj, 0);  // 获取对象的长度提示
        if (elcount < 0) {  // 如果长度提示小于零则跳转至结束标签
            goto done;
        }
    }
    else {
        elcount = count;  // 否则使用给定的元素计数
    }

    elsize = dtype->elsize;  // 获取数据类型的元素大小

    /*
     * Note that PyArray_DESCR(ret) may not match dtype.  There are exactly
     * two cases where this can happen: empty strings/bytes/void (rejected
     * above) and subarray dtypes (supported by sticking with `dtype`).
     */
    Py_INCREF(dtype);  // 增加数据类型的引用计数
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1,
                                                &elcount, NULL,NULL, 0, NULL);  // 从数据类型描述创建新的 NumPy 数组对象
    if (ret == NULL) {  // 如果创建失败则跳转至结束标签
        goto done;
    }

    char *item = PyArray_BYTES(ret);  // 获取数组对象的数据指针
    for (i = 0; i < count || count == -1; i++, item += elsize) {
        // 从迭代器中获取下一个对象
        PyObject *value = PyIter_Next(iter);
        if (value == NULL) {
            // 如果获取失败，可能是因为迭代器已经耗尽
            if (PyErr_Occurred()) {
                /* 获取下一个项目失败，可能是由于迭代器耗尽 */
                goto done;
            }
            break;
        }

        // 如果元素计数超过了预分配的元素数并且元素大小不为0，则进行动态扩展
        if (NPY_UNLIKELY(i >= elcount) && elsize != 0) {
            char *new_data = NULL;
            npy_intp nbytes;
            /*
              扩展 PyArray_DATA(ret) 的大小：
              类似于 PyListObject 的策略，但我们使用50%的过分分配 => 0, 4, 8, 14, 23, 36, 56, 86 ...
              TODO: loadtxt 代码现在使用了一个 `growth` 辅助函数，在这里可能适合重用。
            */
            elcount = (i >> 1) + (i < 4 ? 4 : 2) + i;
            // 计算所需的字节数，并尝试分配内存
            if (!npy_mul_sizes_with_overflow(&nbytes, elcount, elsize)) {
                /* 处理程序始终有效 */
                new_data = PyDataMem_UserRENEW(
                        PyArray_BYTES(ret), nbytes, PyArray_HANDLER(ret));
            }
            if (new_data == NULL) {
                // 内存分配失败，设置 MemoryError 并跳转到完成标签
                PyErr_SetString(PyExc_MemoryError,
                        "cannot allocate array memory");
                Py_DECREF(value);
                goto done;
            }
            // 更新数组的数据指针为重新分配的内存块
            ((PyArrayObject_fields *)ret)->data = new_data;
            /* 调整数组的大小以进行清理： */
            PyArray_DIMS(ret)[0] = elcount;
            // 重置 `item` 指针以指向重新分配的块中的位置
            item = new_data + i * elsize;
            if (PyDataType_FLAGCHK(dtype, NPY_NEEDS_INIT)) {
                /* 初始化新块： */
                memset(item, 0, nbytes - i * elsize);
            }
        }

        // 将值打包到数组中
        if (PyArray_Pack(dtype, item, value) < 0) {
            Py_DECREF(value);
            goto done;
        }
        Py_DECREF(value);
    }

    // 如果实际迭代次数少于预期的元素数，则抛出错误并跳转到完成标签
    if (i < count) {
        PyErr_Format(PyExc_ValueError,
                "iterator too short: Expected %zd but iterator had only %zd "
                "items.", (Py_ssize_t)count, (Py_ssize_t)i);
        goto done;
    }

    /*
     * 重新分配数据，以便不保留多余的内存，并修复数组的第一个维度（可能有多个维度）。
     */
    if (i == 0 || elsize == 0) {
        /* 重新分配数据时大小不能为零。 */
    }
    else {
        /* 调整数组大小到实际的最终大小（可能过大） */
        /* 处理器始终有效 */

        // 使用 PyDataMem_UserRENEW 重新分配内存，以适应新的大小
        char *new_data = PyDataMem_UserRENEW(
                PyArray_DATA(ret), i * elsize, PyArray_HANDLER(ret));

        // 检查内存分配是否成功
        if (new_data == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate array memory");
            goto done;  // 内存分配失败，跳转到完成操作
        }

        // 更新数组对象的数据指针
        ((PyArrayObject_fields *)ret)->data = new_data;

        // 如果 count 小于 0，修正数组的步幅以确保 C 连续性
        if (count < 0) {
            /*
             * 如果 count 小于零，步幅可能全为 0
             * （即使在后续维度中对于 `count < 0` 也是如此！）
             * 因此，再次修正所有步幅以保证 C 连续性。
             */
            int oflags;
            _array_fill_strides(
                    PyArray_STRIDES(ret), PyArray_DIMS(ret), PyArray_NDIM(ret),
                    PyArray_ITEMSIZE(ret), NPY_ARRAY_C_CONTIGUOUS, &oflags);
            // 设置第一个维度的步幅为元素大小
            PyArray_STRIDES(ret)[0] = elsize;
            assert(oflags & NPY_ARRAY_C_CONTIGUOUS);  // 断言数组现在为 C 连续
        }
    }

    // 设置数组的第一个维度大小为 i
    PyArray_DIMS(ret)[0] = i;

done:
    // 释放迭代器对象
    Py_XDECREF(iter);
    // 释放数据类型对象
    Py_XDECREF(dtype);

    // 如果出现错误，清理并返回空指针
    if (PyErr_Occurred()) {
        Py_XDECREF(ret);  // 释放数组对象
        return NULL;  // 返回空指针
    }

    // 返回成功构建的数组对象
    return (PyObject *)ret;
/*
 * This is the main array creation routine.
 *
 * Flags argument has multiple related meanings
 * depending on data and strides:
 *
 * If data is given, then flags is flags associated with data.
 * If strides is not given, then a contiguous strides array will be created
 * and the NPY_ARRAY_C_CONTIGUOUS bit will be set.  If the flags argument
 * has the NPY_ARRAY_F_CONTIGUOUS bit set, then a FORTRAN-style strides array will be
 * created (and of course the NPY_ARRAY_F_CONTIGUOUS flag bit will be set).
 *
 * If data is not given but created here, then flags will be NPY_ARRAY_DEFAULT
 * and a non-zero flags argument can be used to indicate a FORTRAN style
 * array is desired.
 *
 * Dimensions and itemsize must have been checked for validity.
 */

NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp const *dims, int nd, size_t itemsize,
                    int inflag, int *objflags)
{
    int i;
    npy_bool not_cf_contig = 0;
    npy_bool nod = 0; /* A dim != 1 was found */

    /* Check if new array is both F- and C-contiguous */
    for (i = 0; i < nd; i++) {
        if (dims[i] != 1) {
            if (nod) {
                not_cf_contig = 1;
                break;
            }
            nod = 1;
        }
    }

    /* Only make Fortran strides if not contiguous as well */
    if ((inflag & (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS)) ==
                                            NPY_ARRAY_F_CONTIGUOUS) {
        for (i = 0; i < nd; i++) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
            else {
                not_cf_contig = 0;
            }
        }
        if (not_cf_contig) {
            *objflags = ((*objflags)|NPY_ARRAY_F_CONTIGUOUS) &
                                            ~NPY_ARRAY_C_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS);
        }
    }
    else {
        for (i = nd - 1; i >= 0; i--) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
            else {
                not_cf_contig = 0;
            }
        }
        if (not_cf_contig) {
            *objflags = ((*objflags)|NPY_ARRAY_C_CONTIGUOUS) &
                                            ~NPY_ARRAY_F_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
        }
    }
    return;
}
```