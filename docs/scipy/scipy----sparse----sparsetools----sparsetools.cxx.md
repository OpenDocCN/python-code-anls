# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\sparsetools.cxx`

```
/*
 * sparsetools.cxx
 *
 * Python module wrapping the sparsetools C++ routines.
 *
 * Each C++ routine is templated vs. an integer (I) and a data (T) parameter.
 * The `generate_sparsetools.py` script generates `*_impl.h` headers
 * that contain thunk functions with a datatype-based switch statement calling
 * each templated instantiation.
 *
 * `generate_sparsetools.py` also generates a PyMethodDef list of Python
 * routines and the corresponding functions call the thunk functions via
 * `call_thunk`.
 *
 * The `call_thunk` function below determines the templated I and T data types
 * based on the Python arguments. It then allocates arrays with pointers to
 * the raw data, with appropriate types, and calls the thunk function after
 * that.
 *
 * The types of arguments are specified by a "spec". This is given in a format
 * where one character represents one argument. The one-character values are
 * listed below in the call_spec function.
 */

// Define a unique symbol for the PyArray_API, specific to scipy's sparse tools
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_sparsetools_ARRAY_API

// Include Python.h for Python C API functions
#include <Python.h>

// Standard C++ library headers
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdlib>

// NumPy header for ndarray objects
#include "numpy/ndarrayobject.h"

// Custom header files for sparsetools and utility functions
#include "sparsetools.h"
#include "util.h"

// Maximum number of arguments supported
#define MAX_ARGS 16

// List of supported data types for the integer (I) parameter
static const int supported_I_typenums[] = {NPY_INT32, NPY_INT64};
static const int n_supported_I_typenums = sizeof(supported_I_typenums) / sizeof(int);

// List of supported data types for the data (T) parameter
static const int supported_T_typenums[] = {NPY_BOOL,
                                           NPY_BYTE, NPY_UBYTE,
                                           NPY_SHORT, NPY_USHORT,
                                           NPY_INT, NPY_UINT,
                                           NPY_LONG, NPY_ULONG,
                                           NPY_LONGLONG, NPY_ULONGLONG,
                                           NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                                           NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE};
static const int n_supported_T_typenums = sizeof(supported_T_typenums) / sizeof(int);

// Function prototypes for utility functions
static PyObject *array_from_std_vector_and_free(int typenum, void *p);
static void *allocate_std_vector_typenum(int typenum);
static void free_std_vector_typenum(int typenum, void *p);
static PyObject *c_array_from_object(PyObject *obj, int typenum, int is_output);
/*
 * Call a thunk function, dealing with input and output arrays.
 *
 * Resolves the templated <integer> and <data> dtypes from the `args` argument
 * list.
 *
 * Parameters
 * ----------
 * ret_spec : {'i', 'v'}
 *     Return value spec. 'i' for integer, 'v' for void.
 * spec
 *     String whose each character specifies a types of an
 *     argument:
 *
 *     'i': <integer> scalar
 *     'I': <integer> array
 *     'T': <data> array
 *     'V': std::vector<integer>
 *     'W': std::vector<data>
 *     'B': npy_bool array
 *     '*': indicates that the next argument is an output argument
 * thunk : PY_LONG_LONG thunk(int I_typenum, int T_typenum, void **)
 *     Thunk function to call. It is passed a void** array of pointers to
 *     arguments, constructed according to `spec`. The types of data pointed
 *     to by each element agree with I_typenum and T_typenum, or are bools.
 * args
 *     Python tuple containing unprocessed arguments.
 *
 * Returns
 * -------
 * return_value
 *     The Python return value
 *
 */
PyObject *
call_thunk(char ret_spec, const char *spec, thunk_t *thunk, PyObject *args)
{
    void *arg_list[MAX_ARGS];
    PyObject *arg_arrays[MAX_ARGS];
    int is_output[MAX_ARGS];
    PyObject *return_value = NULL;
    int I_typenum = NPY_INT32;
    int T_typenum = -1;
    int VW_count = 0;
    int I_in_arglist = 0;
    int T_in_arglist = 0;
    int next_is_output = 0;
    int j, k, arg_j;
    const char *p;
    PY_LONG_LONG ret;
    Py_ssize_t max_array_size = 0;
    NPY_BEGIN_THREADS_DEF;

    // Check if args is a tuple; if not, raise a ValueError
    if (!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_ValueError, "args is not a tuple");
        return NULL;
    }

    // Initialize arrays and flags
    for (j = 0; j < MAX_ARGS; ++j) {
        arg_list[j] = NULL;
        arg_arrays[j] = NULL;
        is_output[j] = 0;
    }

    /*
     * Detect data types in the signature
     */
    arg_j = 0;
    j = 0;
    // Iterate over the spec string to determine argument types
    }

    // Check if the number of processed arguments matches the tuple size
    if (arg_j != PyTuple_Size(args)) {
        PyErr_SetString(PyExc_ValueError, "too many arguments");
        goto fail;
    }

    // Check for unsupported data types in input arguments
    if ((I_in_arglist && I_typenum == -1) ||
        (T_in_arglist && T_typenum == -1)) {
        PyErr_SetString(PyExc_ValueError,
                        "unsupported data types in input");
        goto fail;
    }

    /*
     * Cast and extract argument arrays
     */
    j = 0;
    for (p = spec; *p != '\0'; ++p, ++j) {
        PyObject *arg;
        int cur_typenum;

        if (*p == '*') {
            --j;
            continue;
        }
        else if (*p == 'i' || *p == 'l') {
            /* Integer scalars */
            PY_LONG_LONG value;

            // 将 Python 对象转换为 PY_LONG_LONG 类型的整数
            value = PyLong_AsLongLong(arg_arrays[j]);
            if (PyErr_Occurred()) {
                goto fail;  // 如果出现错误，跳转到失败处理部分
            }

            // 根据 *p 和 typenum 决定如何分配内存和保存值
            if ((*p == 'l' || PyArray_EquivTypenums(I_typenum, NPY_INT64))
                    && value == (npy_int64)value) {
                arg_list[j] = std::malloc(sizeof(npy_int64));
                *(npy_int64*)arg_list[j] = (npy_int64)value;
            }
            else if (*p == 'i' && PyArray_EquivTypenums(I_typenum, NPY_INT32)
                     && value == (npy_int32)value) {
                arg_list[j] = std::malloc(sizeof(npy_int32));
                *(npy_int32*)arg_list[j] = (npy_int32)value;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                                "could not convert integer scalar");  // 设置错误信息
                goto fail;  // 转换失败，跳转到失败处理部分
            }
            continue;  // 继续下一个循环迭代
        }
        else if (*p == 'B') {
            /* Boolean arrays already cast */
        }
        else if (*p == 'V') {
            // 分配一个 std::vector，并将其保存到 arg_list[j] 中
            arg_list[j] = allocate_std_vector_typenum(I_typenum);
            if (arg_list[j] == NULL) {
                goto fail;  // 如果分配失败，跳转到失败处理部分
            }
            continue;  // 继续下一个循环迭代
        }
        else if (*p == 'W') {
            // 分配一个 std::vector，并将其保存到 arg_list[j] 中
            arg_list[j] = allocate_std_vector_typenum(T_typenum);
            if (arg_list[j] == NULL) {
                goto fail;  // 如果分配失败，跳转到失败处理部分
            }
            continue;  // 继续下一个循环迭代
        }
        else {
            cur_typenum = (*p == 'I' || *p == 'i') ? I_typenum : T_typenum;

            /* Cast if necessary */
            arg = arg_arrays[j];

            // 检查是否需要类型转换
            if (PyArray_EquivTypenums(PyArray_TYPE((PyArrayObject *)arg), cur_typenum)) {
                /* No cast needed. */
            }
            else if (!is_output[j] || PyArray_CanCastSafely(cur_typenum, PyArray_TYPE((PyArrayObject *)arg))) {
                // 如果不是输出数组或者可以安全转换类型，则执行转换
                arg_arrays[j] = c_array_from_object(arg, cur_typenum, is_output[j]);
                Py_DECREF(arg);
                if (arg_arrays[j] == NULL) {
                    goto fail;  // 如果转换失败，跳转到失败处理部分
                }
            }
            else {
                // 输出数组类型不兼容，设置错误信息并跳转到失败处理部分
                PyErr_SetString(PyExc_ValueError,
                                "Output dtype not compatible with inputs.");
                goto fail;
            }
        }

        // 将 PyArrayObject 的数据指针保存到 arg_list[j] 中
        arg_list[j] = PyArray_DATA((PyArrayObject *)arg_arrays[j]);

        // 更新最大数组大小
        if (PyArray_SIZE((PyArrayObject *)arg_arrays[j]) > max_array_size) {
            max_array_size = PyArray_SIZE((PyArrayObject *)arg_arrays[j]);
        }
    }


    /*
     * Call thunk
     */
    // 如果最大数组大小超过100，释放全局解释器锁（GIL）的阈值：这不是一个无代价的操作
    if (max_array_size > 100) {
        NPY_BEGIN_THREADS;  // 开始线程，释放全局解释器锁
    }
    try {
        // 调用 thunk 函数执行计算，使用给定的类型编号和参数列表
        ret = thunk(I_typenum, T_typenum, arg_list);
        NPY_END_THREADS;  // 结束线程，重新获取全局解释器锁
    } catch (const std::bad_alloc &e) {
        NPY_END_THREADS;  // 如果捕获到内存分配错误，结束线程
        PyErr_SetString(PyExc_MemoryError, e.what());  // 设置内存错误异常
        goto fail;  // 跳转到错误处理部分
    } catch (const std::exception &e) {
        NPY_END_THREADS;  // 如果捕获到其他异常，结束线程
        PyErr_SetString(PyExc_RuntimeError, e.what());  // 设置运行时错误异常
        goto fail;  // 跳转到错误处理部分
    }

    /*
     * 生成返回值
     */

    // 根据返回值规范处理返回值
    switch (ret_spec) {
    case 'i':
    case 'l':
        // 如果返回值规范为 'i' 或 'l'，创建一个 PyLong 对象返回
        return_value = PyLong_FromLongLong(ret);
        break;
    case 'v':
        // 如果返回值规范为 'v'，返回 None 对象，并增加其引用计数
        Py_INCREF(Py_None);
        return_value = Py_None;
        break;
    default:
        // 如果返回值规范不合法，设置值错误异常
        PyErr_SetString(PyExc_ValueError,
                        "internal error: invalid return value spec");
    }

    /*
     * 转换任何 std::vector 输出数组为 Python 元组
     */
    if (VW_count > 0) {
        PyObject *new_ret;  // 新的返回值元组
        PyObject *old_ret = return_value;  // 保存原始返回值对象的引用
        int pos;

        return_value = NULL;  // 将返回值对象置空

        // 创建一个新的 Python 元组，长度为 VW_count 加上可能的原始返回值 Py_None
        new_ret = PyTuple_New(VW_count + (old_ret == Py_None ? 0 : 1));
        if (new_ret == NULL) {
            goto fail;  // 如果创建失败，跳转到错误处理部分
        }
        if (old_ret != Py_None) {
            // 如果原始返回值不是 Py_None，则将其放入元组的第一个位置
            PyTuple_SET_ITEM(new_ret, 0, old_ret);
            pos = 1;
        }
        else {
            // 如果原始返回值是 Py_None，减少其引用计数并调整位置索引
            Py_DECREF(old_ret);
            pos = 0;
        }

        j = 0;
        // 遍历返回值规范字符串 spec
        for (p = spec; *p != '\0'; ++p, ++j) {
            if (*p == '*') {
                --j;
                continue;  // 跳过 '*' 符号
            }
            else if (*p == 'V' || *p == 'W') {
                PyObject *arg;
                // 根据 spec 中的 'V' 或 'W' 创建相应类型的 Python 数组
                if (*p == 'V') {
                    arg = array_from_std_vector_and_free(I_typenum, arg_list[j]);
                } else {
                    arg = array_from_std_vector_and_free(T_typenum, arg_list[j]);
                }
                arg_list[j] = NULL;  // 将 arg_list 中的对应位置置空
                if (arg == NULL) {
                    Py_XDECREF(new_ret);  // 如果创建数组失败，减少新元组的引用计数
                    goto fail;  // 跳转到错误处理部分
                }
                PyTuple_SET_ITEM(new_ret, pos, arg);  // 将创建的数组放入新元组的指定位置
                ++pos;  // 更新位置索引
            }
        }

        return_value = new_ret;  // 更新返回值为新创建的元组
    }
fail:
    /*
     * Cleanup
     */
    // 用于迭代处理参数规范字符串
    for (j = 0, p = spec; *p != '\0'; ++p, ++j) {
        // 如果当前字符是 '*'，跳过当前参数处理
        if (*p == '*') {
            --j;
            continue;
        }
        // 如果当前参数是输出参数，且参数数组不为空且是 NumPy 数组类型，解决写回问题
        if (is_output[j] && arg_arrays[j] != NULL && PyArray_Check(arg_arrays[j])) {
            PyArray_ResolveWritebackIfCopy((PyArrayObject *)arg_arrays[j]);
        }
        // 释放参数数组的 Python 引用
        Py_XDECREF(arg_arrays[j]);
        // 如果当前参数是 'i' 或 'l' 类型，并且参数列表不为空，释放其内存
        if ((*p == 'i' || *p == 'l') && arg_list[j] != NULL) {
            std::free(arg_list[j]);
        }
        // 如果当前参数是 'V' 类型，并且参数列表不为空，释放其内存
        else if (*p == 'V' && arg_list[j] != NULL) {
            free_std_vector_typenum(I_typenum, arg_list[j]);
        }
        // 如果当前参数是 'W' 类型，并且参数列表不为空，释放其内存
        else if (*p == 'W' && arg_list[j] != NULL) {
            free_std_vector_typenum(T_typenum, arg_list[j]);
        }
    }
    // 返回函数的返回值
    return return_value;
}


/*
 * Helper functions for dealing with std::vector templated instantiation.
 */

// 根据类型号分配对应类型的 std::vector 内存空间
static void *allocate_std_vector_typenum(int typenum)
{
#define PROCESS(ntype, ctype)                                   \
    if (PyArray_EquivTypenums(typenum, ntype)) {                \
        return (void*)(new std::vector<ctype>());               \
    }

    try {
        SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)
    } catch (std::exception &e) {
        /* failed */
    }

#undef PROCESS

    // 若分配失败，设置运行时错误并返回 NULL
    PyErr_SetString(PyExc_RuntimeError,
                    "failed to allocate std::vector");
    return NULL;
}

// 释放对应类型的 std::vector 内存空间
static void free_std_vector_typenum(int typenum, void *p)
{
#define PROCESS(ntype, ctype)                                   \
    if (PyArray_EquivTypenums(typenum, ntype)) {                \
        delete ((std::vector<ctype>*)p);                        \
        return;                                                 \
    }

    SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)

#undef PROCESS
}

// 根据 std::vector 转换为 NumPy 数组并释放内存空间
static PyObject *array_from_std_vector_and_free(int typenum, void *p)
{
#define PROCESS(ntype, ctype)                                   \
    if (PyArray_EquivTypenums(typenum, ntype)) {                \
        std::vector<ctype> *v = (std::vector<ctype>*)p;         \
        npy_intp length = v->size();                            \
        // 根据 std::vector 的长度创建 NumPy 数组对象
        PyObject *obj = PyArray_SimpleNew(1, &length, typenum); \
        // 如果长度大于 0，复制 std::vector 数据到 NumPy 数组中
        if (length > 0) {                                       \
            memcpy(PyArray_DATA((PyArrayObject *)obj), &((*v)[0]), \
                   sizeof(ctype)*length);                       \
        }                                                       \
        // 释放 std::vector 内存空间
        delete v;                                               \
        return obj;                                             \
    }

    SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)

#undef PROCESS

    // 若转换失败，设置运行时错误并返回 NULL
    PyErr_SetString(PyExc_RuntimeError,
                    "failed to convert std::vector output array");
    return NULL;
}

static PyObject *c_array_from_object(PyObject *obj, int typenum, int is_output)
{
    # 如果 is_output 不为真，则执行以下代码块
    if (!is_output) {
        # 如果 typenum 等于 -1，则调用 PyArray_FROM_OF 函数，返回一个 C 连续且未交换的 NumPy 数组对象
        if (typenum == -1) {
            return PyArray_FROM_OF(obj, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED);
        }
        # 否则，调用 PyArray_FROM_OTF 函数，返回一个指定类型的 NumPy 数组对象，保证是 C 连续且未交换的
        else {
            return PyArray_FROM_OTF(obj, typenum, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED);
        }
    }
    # 如果 is_output 为真，则执行以下代码块
    else {
        # 定义 flags 变量，包括 C 连续、可写、写回如果复制、未交换的标志
        int flags = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_WRITEABLE|NPY_ARRAY_WRITEBACKIFCOPY|NPY_ARRAY_NOTSWAPPED;
        # 如果 typenum 等于 -1，则调用 PyArray_FROM_OF 函数，返回一个符合 flags 标志的 NumPy 数组对象
        if (typenum == -1) {
            return PyArray_FROM_OF(obj, flags);
        }
        # 否则，调用 PyArray_FROM_OTF 函数，返回一个指定类型的 NumPy 数组对象，同时符合 flags 标志
        else {
            return PyArray_FROM_OTF(obj, typenum, flags);
        }
    }
}

/*
 * Python module initialization
 */

extern "C" {

#include "sparsetools_impl.h"

// 定义模块的结构体，指定模块名和方法列表
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,   // 模块定义头部初始化
    "_sparsetools",          // 模块名
    NULL,                    // 模块文档
    -1,                      // 不持有状态信息
    sparsetools_methods,     // 模块方法列表
    NULL,
    NULL,
    NULL,
    NULL
};

// Python 模块初始化函数
PyMODINIT_FUNC
PyInit__sparsetools(void)
{
    import_array();          // 导入 NumPy 数组支持
    return PyModule_Create(&moduledef);  // 创建并返回 Python 模块对象
}

} /* extern "C" */
```