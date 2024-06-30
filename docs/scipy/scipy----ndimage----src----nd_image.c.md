# `D:\src\scipysrc\scipy\scipy\ndimage\src\nd_image.c`

```
/*
 * 版权所有（C）2003-2005年Peter J. Verveer
 *
 * 源代码和二进制形式的重新分发及其修改都是允许的，不论是否包含以下条件：
 *
 * 1. 源代码的重新分发必须保留上述版权声明、此条件列表和以下免责声明。
 *
 * 2. 在二进制形式的重新分发中，必须在提供的文档和/或其他材料中重复上述版权声明、
 *    此条件列表和以下免责声明。
 *
 * 3. 未经特定的书面许可，不得使用作者的名字来认可或推广基于这个软件的产品。
 *
 * 本软件由作者按原样提供，任何明示或暗示的保证，包括但不限于适销性和特定用途的适用性
 * 暗示保证都是被拒绝的。无论如何，作者都不对任何直接、间接、偶然、特殊、惩罚性或后果性
 * 损害（包括但不限于替代商品或服务的采购、使用数据或利润的损失或业务中断）负责，
 * 即使事先已被告知此类损害的可能性。
 */

/*
 * 这两个导入语句的顺序不应更改，请参阅 ni_support.h 中的注释说明。
 */
#include "nd_image.h"
#include "ni_support.h"

#include "ni_filters.h"
#include "ni_fourier.h"
#include "ni_morphology.h"
#include "ni_interpolation.h"
#include "ni_measure.h"

#include "ccallback.h"

/*
 * 定义一个结构体，用于存储Python回调函数的额外参数和关键字参数。
 */
typedef struct {
    PyObject *extra_arguments;  // 额外的参数
    PyObject *extra_keywords;   // 额外的关键字参数
} NI_PythonCallbackData;

/* Numarray 辅助函数 */

/*
 * 创建一个新的NumPy数组，指定类型和形状，并根据缓冲区的内容复制到数组中，
 * 或者如果缓冲区为NULL，则设置为全零。
 */
static PyArrayObject *
NA_NewArray(void *buffer, enum NPY_TYPES type, int ndim, npy_intp *shape)
{
    PyArrayObject *result;

    if (type == NPY_NOTYPE) {
        type = NPY_DOUBLE;
    }

    result = (PyArrayObject *)PyArray_SimpleNew(ndim, shape, type);
    if (result == NULL) {
        return NULL;
    }

    if (buffer == NULL) {
        memset(PyArray_DATA(result), 0, PyArray_NBYTES(result));
    }
    else {
        memcpy(PyArray_DATA(result), buffer, PyArray_NBYTES(result));
    }

    return result;
}

/* 将Python数组类对象转换为适合作为输入数组的数组对象。 */
static int
NI_ObjectToInputArray(PyObject *object, PyArrayObject **array)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *array = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags,
                                                   NULL);
    return *array != NULL;
}
/* Like NI_ObjectToInputArray, but with special handling for Py_None. */
static int
NI_ObjectToOptionalInputArray(PyObject *object, PyArrayObject **array)
{
    // 如果对象是 Py_None，则将输出数组指针设置为 NULL，并返回成功
    if (object == Py_None) {
        *array = NULL;
        return 1;
    }
    // 否则，调用通用的输入数组转换函数 NI_ObjectToInputArray
    return NI_ObjectToInputArray(object, array);
}

/* Converts a Python array-like object into a behaved output array. */
static int
NI_ObjectToOutputArray(PyObject *object, PyArrayObject **array)
{
    // 设置 NumPy 数组的标志，确保其表现正常并可写回
    int flags = NPY_ARRAY_BEHAVED_NS | NPY_ARRAY_WRITEBACKIFCOPY;
    
    /*
     * 这个条件也会被 PyArray_CheckFromAny 调用捕捉到，
     * 但是在这里我们显式检查以提供更合理的错误消息。
     */
    if (PyArray_Check(object) &&
            !PyArray_ISWRITEABLE((PyArrayObject *)object)) {
        PyErr_SetString(PyExc_ValueError, "output array is read-only.");
        return 0;
    }
    
    /*
     * 如果输入数组未对齐或者是字节交换的，这个调用将创建一个新的对齐的本地字节顺序数组，
     * 并将 object 的内容复制到其中。对于输出数组，这个复制是不必要的，所以这可以被优化。
     * 但是很容易不正确地执行 NPY_ARRAY_UPDATEIFCOPY，所以我们让 NumPy 为我们执行
     * 并为此付出性能代价。
     */
    *array = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags,
                                                   NULL);
    return *array != NULL;
}

/* Like NI_ObjectToOutputArray, but with special handling for Py_None. */
static int
NI_ObjectToOptionalOutputArray(PyObject *object, PyArrayObject **array)
{
    // 如果对象是 Py_None，则将输出数组指针设置为 NULL，并返回成功
    if (object == Py_None) {
        *array = NULL;
        return 1;
    }
    // 否则，调用通用的输出数组转换函数 NI_ObjectToOutputArray
    return NI_ObjectToOutputArray(object, array);
}

/* Converts a Python array-like object into a behaved input/output array. */
static int
NI_ObjectToInputOutputArray(PyObject *object, PyArrayObject **array)
{
    /*
     * 这个检查也在 NI_ObjectToOutputArray 中进行，
     * 这里再次检查是为了提供更具体的错误消息。
     */
    if (PyArray_Check(object) &&
            !PyArray_ISWRITEABLE((PyArrayObject *)object)) {
        PyErr_SetString(PyExc_ValueError, "input/output array is read-only.");
        return 0;
    }
    // 调用通用的输出数组转换函数 NI_ObjectToOutputArray
    return NI_ObjectToOutputArray(object, array);
}

/* Checks that an origin value was received for each array dimension. */
static int
_validate_origin(PyArrayObject *array, PyArray_Dims origin)
{
    // 检查 'origin' 数组的长度是否与输入数组的维度数相同
    if (origin.len != PyArray_NDIM(array)) {
        PyErr_Format(PyExc_ValueError,
                     "Invalid %d element 'origin' sequence for "
                     "%d-dimensional input array.",
                     origin.len, PyArray_NDIM(array));
        return 0;
    }
    return 1;
}

/*********************************************************************/
/* wrapper functions: */
/*********************************************************************/

static PyObject *Py_Correlate1D(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL, *weights = NULL;
    # 声明整型变量 axis 和 mode，以及双精度浮点型变量 cval，以及 npy_intp 类型变量 origin
    int axis, mode;
    double cval;
    npy_intp origin;

    # 使用 PyArg_ParseTuple 解析传入的参数元组，参数格式为 "O&O&iO&idn"
    # 第一个参数为 NI_ObjectToInputArray 函数解析 input 到 input 变量
    # 第二个参数为 NI_ObjectToInputArray 函数解析 weights 到 weights 变量，同时读取 axis 作为整型
    # 第三个参数为 NI_ObjectToOutputArray 函数解析 output 到 output 变量，同时读取 mode 作为整型，cval 作为双精度浮点数，origin 作为 npy_intp 类型
    if (!PyArg_ParseTuple(args, "O&O&iO&idn" ,
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToInputArray, &weights, &axis,
                          NI_ObjectToOutputArray, &output, &mode, &cval,
                          &origin))
        # 如果解析失败，则跳转到 exit 标签处
        goto exit;

    # 调用 NI_Correlate1D 函数，传入 input, weights, axis, output, mode, cval 和 origin 参数
    NI_Correlate1D(input, weights, axis, output, (NI_ExtendMode)mode, cval,
                   origin);
    # 解析 output 数组，如果有副本则进行写回操作
    PyArray_ResolveWritebackIfCopy(output);
exit:
    // 释放输入数组对象的引用
    Py_XDECREF(input);
    // 释放权重数组对象的引用
    Py_XDECREF(weights);
    // 释放输出数组对象的引用
    Py_XDECREF(output);
    // 如果出现异常，则返回 NULL；否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_Correlate(PyObject *obj, PyObject *args)
{
    // 声明输入、输出和权重的 NumPy 数组对象指针
    PyArrayObject *input = NULL, *output = NULL, *weights = NULL;
    // 声明用于表示数组维度的结构体
    PyArray_Dims origin = {NULL, 0};
    // 声明模式和常量值
    int mode;
    double cval;

    // 解析传入的 Python 参数元组，将参数解析为对应的 NumPy 数组对象
    if (!PyArg_ParseTuple(args, "O&O&O&idO&", NI_ObjectToInputArray, &input,
                          NI_ObjectToInputArray, &weights,
                          NI_ObjectToOutputArray, &output,
                          &mode, &cval,
                          PyArray_IntpConverter, &origin)) {
        // 如果解析失败，则跳转到 exit 标签处
        goto exit;
    }
    // 验证输入数组和起始位置数组是否匹配
    if (!_validate_origin(input, origin)) {
        // 如果验证失败，则跳转到 exit 标签处
        goto exit;
    }

    // 调用 C 库中的函数进行相关操作
    NI_Correlate(input, weights, output, (NI_ExtendMode)mode, cval,
                 origin.ptr);
    // 解决输出数组的写回问题
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入、权重和输出数组对象的引用
    Py_XDECREF(input);
    Py_XDECREF(weights);
    Py_XDECREF(output);
    // 释放起始位置数组的内存
    PyDimMem_FREE(origin.ptr);
    // 如果出现异常，则返回 NULL；否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_UniformFilter1D(PyObject *obj, PyObject *args)
{
    // 声明输入和输出的 NumPy 数组对象指针
    PyArrayObject *input = NULL, *output = NULL;
    // 声明过滤器大小、轴、模式和常量值
    int axis, mode;
    npy_intp filter_size, origin;
    double cval;

    // 解析传入的 Python 参数元组，将参数解析为对应的 NumPy 数组对象
    if (!PyArg_ParseTuple(args, "O&niO&idn",
                          NI_ObjectToInputArray, &input,
                          &filter_size, &axis,
                          NI_ObjectToOutputArray, &output,
                          &mode, &cval, &origin))
        // 如果解析失败，则跳转到 exit 标签处
        goto exit;

    // 调用 C 库中的函数进行一维均匀过滤操作
    NI_UniformFilter1D(input, filter_size, axis, output, (NI_ExtendMode)mode,
                       cval, origin);
    // 解决输出数组的写回问题
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入和输出数组对象的引用
    Py_XDECREF(input);
    Py_XDECREF(output);
    // 如果出现异常，则返回 NULL；否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_MinOrMaxFilter1D(PyObject *obj, PyObject *args)
{
    // 声明输入和输出的 NumPy 数组对象指针
    PyArrayObject *input = NULL, *output = NULL;
    // 声明过滤器大小、轴、模式、最小值标志和起始位置
    int axis, mode, minimum;
    npy_intp filter_size, origin;
    double cval;

    // 解析传入的 Python 参数元组，将参数解析为对应的 NumPy 数组对象
    if (!PyArg_ParseTuple(args, "O&niO&idni",
                          NI_ObjectToInputArray, &input,
                          &filter_size, &axis,
                          NI_ObjectToOutputArray, &output,
                          &mode, &cval, &origin, &minimum))
        // 如果解析失败，则跳转到 exit 标签处
        goto exit;

    // 调用 C 库中的函数进行一维最小或最大值过滤操作
    NI_MinOrMaxFilter1D(input, filter_size, axis, output, (NI_ExtendMode)mode,
                        cval, origin, minimum);
    // 解决输出数组的写回问题
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入和输出数组对象的引用
    Py_XDECREF(input);
    Py_XDECREF(output);
    // 如果出现异常，则返回 NULL；否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_MinOrMaxFilter(PyObject *obj, PyObject *args)
{
    // 声明输入、输出、足印和结构的 NumPy 数组对象指针
    PyArrayObject *input = NULL, *output = NULL, *footprint = NULL;
    PyArrayObject *structure = NULL;
    // 声明用于表示数组维度的结构体
    PyArray_Dims origin = {NULL, 0};
    // 声明模式和最小值标志
    int mode, minimum;
    // 声明常量值
    double cval;
    # 如果无法解析参数元组 `args`，则跳转到 `exit` 标签
    if (!PyArg_ParseTuple(args, "O&O&O&O&idO&i",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToInputArray, &footprint,
                          NI_ObjectToOptionalInputArray, &structure,
                          NI_ObjectToOutputArray, &output,
                          &mode, &cval,
                          PyArray_IntpConverter, &origin,
                          &minimum)) {
        goto exit;
    }
    
    # 验证 `origin` 是否在 `input` 数组的有效范围内，如果不在则跳转到 `exit` 标签
    if (!_validate_origin(input, origin)) {
        goto exit;
    }

    # 使用输入数组 `input` 和相关参数进行最小值或最大值滤波操作，结果存入输出数组 `output`
    NI_MinOrMaxFilter(input, footprint, structure, output, (NI_ExtendMode)mode,
                      cval, origin.ptr, minimum);
    
    # 解析输出数组 `output`，确保写回操作完成（如果有拷贝）
    PyArray_ResolveWritebackIfCopy(output);
exit:
    Py_XDECREF(input);
    Py_XDECREF(footprint);
    Py_XDECREF(output);
    PyDimMem_FREE(origin.ptr);
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_RankFilter(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL, *footprint = NULL;
    PyArray_Dims origin = {NULL, 0};
    int mode, rank;
    double cval;

    if (!PyArg_ParseTuple(args, "O&iO&O&idO&",
                          NI_ObjectToInputArray, &input, &rank,
                          NI_ObjectToInputArray, &footprint,
                          NI_ObjectToOutputArray, &output,
                          &mode, &cval,
                          PyArray_IntpConverter, &origin)) {
        // 解析传入的参数元组，如果解析失败则跳转到 exit 标签处
        goto exit;
    }
    if (!_validate_origin(input, origin)) {
        // 验证 origin 参数是否有效，无效则跳转到 exit 标签处
        goto exit;
    }

    // 调用 C 函数 NI_RankFilter 进行排名滤波操作
    NI_RankFilter(input, rank, footprint, output, (NI_ExtendMode)mode, cval,
                  origin.ptr);
    // 解决输出数组的写回问题
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 清理变量，避免内存泄漏，并根据错误状态决定返回值
    Py_XDECREF(input);
    Py_XDECREF(footprint);
    Py_XDECREF(output);
    PyDimMem_FREE(origin.ptr);
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static int Py_Filter1DFunc(double *iline, npy_intp ilen,
                           double *oline, npy_intp olen, void *data)
{
    PyArrayObject *py_ibuffer = NULL, *py_obuffer = NULL;
    PyObject *rv = NULL, *args = NULL, *tmp = NULL;
    npy_intp ii;
    double *po = NULL;
    ccallback_t *callback = (ccallback_t *)data;
    NI_PythonCallbackData *cbdata = (NI_PythonCallbackData*)callback->info_p;

    // 创建输入和输出的 NumPy 数组对象
    py_ibuffer = NA_NewArray(iline, NPY_DOUBLE, 1, &ilen);
    py_obuffer = NA_NewArray(NULL, NPY_DOUBLE, 1, &olen);
    if (!py_ibuffer || !py_obuffer)
        // 如果创建失败则跳转到 exit 标签处
        goto exit;
    // 构建函数参数元组
    tmp = Py_BuildValue("(OO)", py_ibuffer, py_obuffer);
    if (!tmp)
        // 如果构建失败则跳转到 exit 标签处
        goto exit;
    // 将额外的参数附加到参数元组中
    args = PySequence_Concat(tmp, cbdata->extra_arguments);
    if (!args)
        // 如果构建失败则跳转到 exit 标签处
        goto exit;
    // 调用 Python 回调函数
    rv = PyObject_Call(callback->py_function, args, cbdata->extra_keywords);
    if (!rv)
        // 如果调用失败则跳转到 exit 标签处
        goto exit;
    // 获取输出数组的数据指针并复制到输出数组中
    po = (double*)PyArray_DATA(py_obuffer);
    for(ii = 0; ii < olen; ii++)
        oline[ii] = po[ii];

exit:
    // 清理变量，避免内存泄漏
    Py_XDECREF(py_ibuffer);
    Py_XDECREF(py_obuffer);
    Py_XDECREF(rv);
    Py_XDECREF(args);
    Py_XDECREF(tmp);
    // 根据错误状态决定返回值
    return PyErr_Occurred() ? 0 : 1;
}

static PyObject *Py_GenericFilter1D(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL;
    PyObject *fnc = NULL, *extra_arguments = NULL, *extra_keywords = NULL;
    void *func = NULL, *data = NULL;
    NI_PythonCallbackData cbdata;
    int axis, mode;
    npy_intp origin, filter_size;
    double cval;
    ccallback_t callback;
    static ccallback_signature_t callback_signatures[] = {
        {"int (double *, intptr_t, double *, intptr_t, void *)"},
        {"int (double *, npy_intp, double *, npy_intp, void *)"},
#if NPY_SIZEOF_INTP == NPY_SIZEOF_SHORT
        {"int (double *, short, double *, short, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_INT
        {"int (double *, int, double *, int, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONG
        {"int (double *, long, double *, long, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONGLONG
        {"int (double *, long long, double *, long long, void *)"},
#endif
        {NULL}
    };


// 根据 NPY_SIZEOF_INTP 的大小定义不同的函数签名字符串，用于回调函数的注册
// 如果 NPY_SIZEOF_INTP 等于 NPY_SIZEOF_INT，则使用 int 类型的参数
// 如果 NPY_SIZEOF_INTP 等于 NPY_SIZEOF_LONG，则使用 long 类型的参数
// 如果 NPY_SIZEOF_INTP 等于 NPY_SIZEOF_LONGLONG，则使用 long long 类型的参数
// 最后一个元素是 NULL，用于标志列表的结束


    callback.py_function = NULL;
    callback.c_function = NULL;

    if (!PyArg_ParseTuple(args, "O&OniO&idnOO",
                          NI_ObjectToInputArray, &input,
                          &fnc, &filter_size, &axis,
                          NI_ObjectToOutputArray, &output,
                          &mode, &cval, &origin,
                          &extra_arguments, &extra_keywords))
        goto exit;


// 尝试解析 Python 的函数参数，并进行错误处理
// NI_ObjectToInputArray 和 NI_ObjectToOutputArray 是对象转换函数
// 参数分别为 input, fnc, filter_size, axis, output, mode, cval, origin,
// extra_arguments, extra_keywords 分别对应解析后的 Python 对象
// 如果解析失败，则跳转到 exit 标签


    if (!PyTuple_Check(extra_arguments)) {
        PyErr_SetString(PyExc_RuntimeError, "extra_arguments must be a tuple");
        goto exit;
    }
    if (!PyDict_Check(extra_keywords)) {
        PyErr_SetString(PyExc_RuntimeError,
                                        "extra_keywords must be a dictionary");
        goto exit;
    }


// 检查 extra_arguments 是否为元组，extra_keywords 是否为字典
// 如果不是，则设置相应的异常并跳转到 exit 标签


    if (PyCapsule_CheckExact(fnc) && PyCapsule_GetName(fnc) == NULL) {
        /* 'Legacy' low-level callable */
        func = PyCapsule_GetPointer(fnc, NULL);
        data = PyCapsule_GetContext(fnc);
    } else {
        int ret;

        ret = ccallback_prepare(&callback, callback_signatures, fnc, CCALLBACK_DEFAULTS);
        if (ret == -1) {
            goto exit;
        }

        if (callback.py_function != NULL) {
            cbdata.extra_arguments = extra_arguments;
            cbdata.extra_keywords = extra_keywords;
            callback.info_p = (void*)&cbdata;
            func = Py_Filter1DFunc;
            data = (void*)&callback;
        }
        else {
            func = callback.c_function;
            data = callback.user_data;
        }
    }


// 根据传入的 fnc 对象类型进行分支处理
// 如果 fnc 是 PyCapsule 类型且没有名称，则将其作为低级别的回调函数（Legacy）
// 否则，通过 ccallback_prepare 函数准备 callback 结构体和签名列表
// 如果准备过程失败（返回值为 -1），则跳转到 exit 标签
// 如果 callback.py_function 不为空，则设置回调函数的额外参数和关键字
// 否则，使用 callback.c_function 和 callback.user_data


    NI_GenericFilter1D(input, func, data, filter_size, axis, output,
                       (NI_ExtendMode)mode, cval, origin);
    PyArray_ResolveWritebackIfCopy(output);


// 调用 NI_GenericFilter1D 函数进行一维泛化滤波操作
// 参数依次为 input, func, data, filter_size, axis, output, mode, cval, origin
// 如果 output 是副本，则解析写回


exit:
    if (callback.py_function != NULL || callback.c_function != NULL) {
        ccallback_release(&callback);
    }
    Py_XDECREF(input);
    Py_XDECREF(output);
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}


// 退出标签，处理清理工作及异常情况
// 如果 callback.py_function 或 callback.c_function 不为空，则释放回调资源
// 释放 input 和 output 对象的引用
// 如果有异常发生，则返回 NULL，否则返回一个空的 Python 对象


static int Py_FilterFunc(double *buffer, npy_intp filter_size,
                                                 double *output, void *data)
{
    PyArrayObject *py_buffer = NULL;
    PyObject *rv = NULL, *args = NULL, *tmp = NULL;
    ccallback_t *callback = (ccallback_t *)data;
    NI_PythonCallbackData *cbdata = (NI_PythonCallbackData*)callback->info_p;

    py_buffer = NA_NewArray(buffer, NPY_DOUBLE, 1, &filter_size);
    if (!py_buffer)
        goto exit;
    tmp = Py_BuildValue("(O)", py_buffer);
    if (!tmp)
        goto exit;
    args = PySequence_Concat(tmp, cbdata->extra_arguments);
    if (!args)
        goto exit;
    rv = PyObject_Call(callback->py_function, args, cbdata->extra_keywords);


// 定义一个名为 Py_FilterFunc 的静态函数，用于 Python 回调
// 创建 py_buffer 对象，表示输入数据的 NumPy 数组
// 如果创建失败，则跳转到 exit 标签
// 使用 Py_BuildValue 创建一个包含 py_buffer 的元组 tmp
// 如果创建失败，则跳转到 exit 标签
// 将 tmp 与额外参数 extra_arguments 连接成 args
// 如果连接失败，则跳转到 exit 标签
// 调用 callback.py_function 执行 Python 回调函数，并传入 args 和 extra_keywords
    # 检查 rv 是否为假值（即 NULL 或 0），如果是，则跳转到 exit 标签处
    if (!rv)
        goto exit;
    # 将 PyFloat_AsDouble 函数返回值赋给 output 指针所指向的位置，即取出 Python 浮点数对象的双精度浮点数值
    *output = PyFloat_AsDouble(rv);
    // 释放申请的 Python 对象的内存
    Py_XDECREF(py_buffer);
    Py_XDECREF(rv);
    Py_XDECREF(args);
    Py_XDECREF(tmp);
    // 如果发生了异常，返回 0；否则返回 1
    return PyErr_Occurred() ? 0 : 1;
}

// Python C 扩展模块中的一个函数，实现通用的滤波操作
static PyObject *Py_GenericFilter(PyObject *obj, PyObject *args)
{
    // 定义输入和输出的数组对象，以及用于脚印和其他参数的 Python 对象
    PyArrayObject *input = NULL, *output = NULL, *footprint = NULL;
    PyObject *fnc = NULL, *extra_arguments = NULL, *extra_keywords = NULL;
    // 函数指针和数据指针
    void *func = NULL, *data = NULL;
    // 回调函数数据结构
    NI_PythonCallbackData cbdata;
    // 滤波模式和原点
    int mode;
    PyArray_Dims origin = {NULL, 0};
    // 常数值
    double cval;
    // 回调函数结构体
    ccallback_t callback;
    // 支持的回调函数签名列表
    static ccallback_signature_t callback_signatures[] = {
        {"int (double *, intptr_t, double *, void *)"},
        {"int (double *, npy_intp, double *, void *)"},
#if NPY_SIZEOF_INTP == NPY_SIZEOF_SHORT
        {"int (double *, short, double *, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_INT
        {"int (double *, int, double *, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONG
        {"int (double *, long, double *, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONGLONG
        {"int (double *, long long, double *, void *)"},
#endif
        {NULL}
    };

    // 初始化回调结构体
    callback.py_function = NULL;
    callback.c_function = NULL;

    // 解析传入的 Python 参数元组，将各个参数赋值给对应的变量
    if (!PyArg_ParseTuple(args, "O&OO&O&idO&OO",
                          NI_ObjectToInputArray, &input,
                          &fnc,
                          NI_ObjectToInputArray, &footprint,
                          NI_ObjectToOutputArray, &output,
                          &mode, &cval,
                          PyArray_IntpConverter, &origin,
                          &extra_arguments, &extra_keywords)) {
        // 解析失败则跳转到 exit 标签
        goto exit;
    }

    // 验证 origin 参数是否有效
    if (!_validate_origin(input, origin)) {
        // 验证失败则跳转到 exit 标签
        goto exit;
    }

    // 检查 extra_arguments 是否为元组类型
    if (!PyTuple_Check(extra_arguments)) {
        PyErr_SetString(PyExc_RuntimeError, "extra_arguments must be a tuple");
        // 设置错误并跳转到 exit 标签
        goto exit;
    }

    // 检查 extra_keywords 是否为字典类型
    if (!PyDict_Check(extra_keywords)) {
        PyErr_SetString(PyExc_RuntimeError,
                                        "extra_keywords must be a dictionary");
        // 设置错误并跳转到 exit 标签
        goto exit;
    }

    // 如果 fnc 是一个 PyCapsule 对象且其名称为空
    if (PyCapsule_CheckExact(fnc) && PyCapsule_GetName(fnc) == NULL) {
        // 获取 PyCapsule 对象的指针和上下文
        func = PyCapsule_GetPointer(fnc, NULL);
        data = PyCapsule_GetContext(fnc);
    } else {
        int ret;

        // 使用 ccallback_prepare 函数准备回调函数
        ret = ccallback_prepare(&callback, callback_signatures, fnc, CCALLBACK_DEFAULTS);
        if (ret == -1) {
            // 准备失败则跳转到 exit 标签
            goto exit;
        }

        // 如果有 Python 函数
        if (callback.py_function != NULL) {
            // 设置额外的参数和关键字参数
            cbdata.extra_arguments = extra_arguments;
            cbdata.extra_keywords = extra_keywords;
            callback.info_p = (void*)&cbdata;
            // 设置函数指针和数据指针
            func = Py_FilterFunc;
            data = (void*)&callback;
        }
        else {
            // 设置 C 函数指针和用户数据指针
            func = callback.c_function;
            data = callback.user_data;
        }
    }

    // 调用通用滤波函数 NI_GenericFilter 进行实际操作
    NI_GenericFilter(input, func, data, footprint, output, (NI_ExtendMode)mode,
                     cval, origin.ptr);
    // 解析输出数组的写回操作
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 函数退出点，无论如何都会执行此处释放资源
    // 检查 callback 结构体中的 py_function 或 c_function 是否不为 NULL，如果有其中之一不为 NULL，则释放 callback 结构体资源
    if (callback.py_function != NULL || callback.c_function != NULL) {
        ccallback_release(&callback);
    }
    // 释放 input 对象的引用计数，避免内存泄漏
    Py_XDECREF(input);
    // 释放 output 对象的引用计数，避免内存泄漏
    Py_XDECREF(output);
    // 释放 footprint 对象的引用计数，避免内存泄漏
    Py_XDECREF(footprint);
    // 释放 origin.ptr 指针所指向的内存块，使用 PyDimMem_FREE 函数，避免内存泄漏
    PyDimMem_FREE(origin.ptr);
    // 检查是否有异常发生（由 PyErr_Occurred() 判断），若有异常则返回 NULL，否则返回一个空的 Py_BuildValue("")
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
static PyObject *Py_FourierFilter(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL, *parameters = NULL;
    int axis, filter_type;
    npy_intp n;

    // 解析传入的参数元组，指定输入和输出数组以及其它参数
    if (!PyArg_ParseTuple(args, "O&O&niO&i",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToInputArray, &parameters,
                          &n, &axis,
                          NI_ObjectToOutputArray, &output,
                          &filter_type))
        goto exit;

    // 调用 C 库函数进行傅里叶滤波
    NI_FourierFilter(input, parameters, n, axis, output, filter_type);
    // 确保输出数组写回到 Python 对象中
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入和输出数组的引用
    Py_XDECREF(input);
    Py_XDECREF(parameters);
    Py_XDECREF(output);
    // 如果发生异常，则返回 NULL；否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_FourierShift(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL, *shifts = NULL;
    int axis;
    npy_intp n;

    // 解析传入的参数元组，指定输入数组、位移数组、轴信息以及输出数组
    if (!PyArg_ParseTuple(args, "O&O&niO&",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToInputArray, &shifts,
                          &n, &axis,
                          NI_ObjectToOutputArray, &output))
        goto exit;

    // 调用 C 库函数进行傅里叶位移
    NI_FourierShift(input, shifts, n, axis, output);
    // 确保输出数组写回到 Python 对象中
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入和输出数组的引用
    Py_XDECREF(input);
    Py_XDECREF(shifts);
    Py_XDECREF(output);
    // 如果发生异常，则返回 NULL；否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_SplineFilter1D(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL;
    int axis, order, mode;

    // 解析传入的参数元组，指定输入数组、阶数、轴信息、输出数组和模式
    if (!PyArg_ParseTuple(args, "O&iiO&i",
                          NI_ObjectToInputArray, &input, &order, &axis,
                          NI_ObjectToOutputArray, &output, &mode))
        goto exit;

    // 调用 C 库函数进行一维样条滤波
    NI_SplineFilter1D(input, order, axis, mode, output);
    // 确保输出数组写回到 Python 对象中
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入和输出数组的引用
    Py_XDECREF(input);
    Py_XDECREF(output);
    // 如果发生异常，则返回 NULL；否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static int Py_Map(npy_intp *ocoor, double* icoor, int orank, int irank,
                                    void *data)
{
    PyObject *coors = NULL, *rets = NULL, *args = NULL, *tmp = NULL;
    npy_intp ii;
    ccallback_t *callback = (ccallback_t *)data;
    NI_PythonCallbackData *cbdata = (NI_PythonCallbackData*)callback->info_p;

    // 创建输入坐标的 Python 元组对象
    coors = PyTuple_New(orank);
    if (!coors)
        goto exit;
    // 将 C 数组中的坐标数据转换为 Python 的长整型并设置到元组中
    for(ii = 0; ii < orank; ii++) {
        PyTuple_SetItem(coors, ii, PyLong_FromSsize_t(ocoor[ii]));
        if (PyErr_Occurred())
            goto exit;
    }
    // 构建调用 Python 回调函数的参数元组
    tmp = Py_BuildValue("(O)", coors);
    if (!tmp)
        goto exit;
    args = PySequence_Concat(tmp, cbdata->extra_arguments);
    if (!args)
        goto exit;
    // 调用注册的 Python 回调函数并获取返回值
    rets = PyObject_Call(callback->py_function, args, cbdata->extra_keywords);
    if (!rets)
        goto exit;
    // 将 Python 返回的浮点数结果复制回 C 数组中
    for(ii = 0; ii < irank; ii++) {
        icoor[ii] = PyFloat_AsDouble(PyTuple_GetItem(rets, ii));
        if (PyErr_Occurred())
            goto exit;
    }

exit:
    // 释放 Python 对象的引用
    Py_XDECREF(coors);
    Py_XDECREF(tmp);
    Py_XDECREF(args);
    Py_XDECREF(rets);
    // 返回操作是否成功的状态
    return PyErr_Occurred() ? -1 : 0;
}
exit:
    // 释放Python对象coors占用的内存
    Py_XDECREF(coors);
    // 释放Python对象tmp占用的内存
    Py_XDECREF(tmp);
    // 释放Python对象rets占用的内存
    Py_XDECREF(rets);
    // 释放Python对象args占用的内存
    Py_XDECREF(args);
    // 如果发生了异常，返回0；否则返回1
    return PyErr_Occurred() ? 0 : 1;
}


static PyObject *Py_GeometricTransform(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL;
    PyArrayObject *coordinates = NULL, *matrix = NULL, *shift = NULL;
    PyObject *fnc = NULL, *extra_arguments = NULL, *extra_keywords = NULL;
    int mode, order, nprepad;
    double cval;
    void *func = NULL, *data = NULL;
    NI_PythonCallbackData cbdata;
    ccallback_t callback;
    static ccallback_signature_t callback_signatures[] = {
        // 定义不同类型的回调函数签名
        {"int (intptr_t *, double *, int, int, void *)"},
        {"int (npy_intp *, double *, int, int, void *)"},
#if NPY_SIZEOF_INTP == NPY_SIZEOF_SHORT
        {"int (short *, double *, int, int, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_INT
        {"int (int *, double *, int, int, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONG
        {"int (long *, double *, int, int, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONGLONG
        {"int (long long *, double *, int, int, void *)"},
#endif
        {NULL}
    };

    // 初始化回调结构体中的Python函数和C函数为NULL
    callback.py_function = NULL;
    callback.c_function = NULL;

    // 解析Python传入的参数元组
    if (!PyArg_ParseTuple(args, "O&OO&O&O&O&iidiOO",
                          NI_ObjectToInputArray, &input,
                          &fnc,
                          NI_ObjectToOptionalInputArray, &coordinates,
                          NI_ObjectToOptionalInputArray, &matrix,
                          NI_ObjectToOptionalInputArray, &shift,
                          NI_ObjectToOutputArray, &output,
                          &order, &mode, &cval, &nprepad,
                          &extra_arguments, &extra_keywords))
        // 如果解析失败，跳转到exit标签处
        goto exit;
    # 如果给定的回调函数不是 None
    if (fnc != Py_None) {
        # 如果额外参数不是一个元组，抛出运行时错误并跳转到退出标签
        if (!PyTuple_Check(extra_arguments)) {
            PyErr_SetString(PyExc_RuntimeError,
                                            "extra_arguments must be a tuple");
            goto exit;
        }
        # 如果额外关键字不是一个字典，抛出运行时错误并跳转到退出标签
        if (!PyDict_Check(extra_keywords)) {
            PyErr_SetString(PyExc_RuntimeError,
                                            "extra_keywords must be a dictionary");
            goto exit;
        }
        # 如果回调函数是一个确切的 Capsule 并且其名称为 NULL
        if (PyCapsule_CheckExact(fnc) && PyCapsule_GetName(fnc) == NULL) {
            # 从 Capsule 中获取函数指针和数据指针
            func = PyCapsule_GetPointer(fnc, NULL);
            data = PyCapsule_GetContext(fnc);
        } else {
            int ret;

            # 准备回调函数以供调用
            ret = ccallback_prepare(&callback, callback_signatures, fnc, CCALLBACK_DEFAULTS);
            # 如果准备失败，跳转到退出标签
            if (ret == -1) {
                goto exit;
            }

            # 如果回调函数是 Python 函数对象
            if (callback.py_function != NULL) {
                # 设置回调数据结构的额外参数和额外关键字
                cbdata.extra_arguments = extra_arguments;
                cbdata.extra_keywords = extra_keywords;
                callback.info_p = (void*)&cbdata;
                # 使用 Python 的 Map 函数作为回调函数
                func = Py_Map;
                data = (void*)&callback;
            }
            else {
                # 否则使用 C 函数作为回调函数
                func = callback.c_function;
                data = callback.user_data;
            }
        }
    }

    # 执行几何变换操作，调用函数 func 处理输入和输出数组
    NI_GeometricTransform(input, func, data, matrix, shift, coordinates,
                          output, order, (NI_ExtendMode)mode, cval, nprepad);
    # 如果输出数组是副本，则解决写回（writeback）问题
    PyArray_ResolveWritebackIfCopy(output);
exit:
    // 如果回调函数不为 NULL，则释放回调结构体资源
    if (callback.py_function != NULL || callback.c_function != NULL) {
        ccallback_release(&callback);
    }
    // 递减输入数组对象的引用计数
    Py_XDECREF(input);
    // 递减输出数组对象的引用计数
    Py_XDECREF(output);
    // 递减坐标数组对象的引用计数
    Py_XDECREF(coordinates);
    // 递减矩阵数组对象的引用计数
    Py_XDECREF(matrix);
    // 递减位移数组对象的引用计数
    Py_XDECREF(shift);
    // 如果出现异常，则返回 NULL；否则返回一个空的 Python 字符串对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_ZoomShift(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL, *shift = NULL;
    PyArrayObject *zoom = NULL;
    int mode, order, nprepad, grid_mode;
    double cval;

    // 解析 Python 函数参数，并将其转换为相应的 NumPy 数组对象
    if (!PyArg_ParseTuple(args, "O&O&O&O&iidii",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToOptionalInputArray, &zoom,
                          NI_ObjectToOptionalInputArray, &shift,
                          NI_ObjectToOutputArray, &output,
                          &order, &mode, &cval, &nprepad, &grid_mode))
        goto exit;

    // 调用 C 函数 NI_ZoomShift 进行图像变换操作
    NI_ZoomShift(input, zoom, shift, output, order, (NI_ExtendMode)mode, cval,
                 nprepad, grid_mode);
    // 如果输出数组是副本，则在此解除写回操作
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 递减输入数组对象的引用计数
    Py_XDECREF(input);
    // 递减位移数组对象的引用计数
    Py_XDECREF(shift);
    // 递减缩放数组对象的引用计数
    Py_XDECREF(zoom);
    // 递减输出数组对象的引用计数
    Py_XDECREF(output);
    // 如果出现异常，则返回 NULL；否则返回一个空的 Python 字符串对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_FindObjects(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL;
    PyObject *result = NULL, *tuple = NULL, *start = NULL, *end = NULL;
    PyObject *slc = NULL;
    int jj;
    npy_intp max_label;
    npy_intp ii, *regions = NULL;

    // 解析 Python 函数参数，并将其转换为相应的 NumPy 数组对象
    if (!PyArg_ParseTuple(args, "O&n",
                          NI_ObjectToInputArray, &input, &max_label))
        goto exit;

    // 如果 max_label 小于 0，则将其设为 0
    if (max_label < 0)
        max_label = 0;
    // 如果 max_label 大于 0，则为 regions 分配内存空间
    if (max_label > 0) {
        if (PyArray_NDIM(input) > 0) {
            regions = (npy_intp*)malloc(2 * max_label * PyArray_NDIM(input) *
                                        sizeof(npy_intp));
        } else {
            regions = (npy_intp*)malloc(max_label * sizeof(npy_intp));
        }
        // 如果分配内存失败，则抛出内存错误异常，并跳转到 exit 标签处
        if (!regions) {
            PyErr_NoMemory();
            goto exit;
        }
    }

    // 调用 C 函数 NI_FindObjects 进行对象查找操作
    if (!NI_FindObjects(input, max_label, regions))
        goto exit;

    // 创建一个空的 Python 列表对象，用于存储查找结果
    result = PyList_New(max_label);
    // 如果创建列表对象失败，则抛出内存错误异常，并跳转到 exit 标签处
    if (!result) {
        PyErr_NoMemory();
        goto exit;
    }
    for(ii = 0; ii < max_label; ii++) {
        // 计算当前标签的起始索引
        npy_intp idx =
                PyArray_NDIM(input) > 0 ? 2 * PyArray_NDIM(input) * ii : ii;
        // 检查当前区域是否有效
        if (regions[idx] >= 0) {
            // 创建一个新的元组对象，元组长度与输入数组的维度相同
            tuple = PyTuple_New(PyArray_NDIM(input));
            // 检查是否分配内存成功
            if (!tuple) {
                PyErr_NoMemory();
                // 跳转到退出标签，释放资源
                goto exit;
            }
            // 遍历输入数组的每一个维度
            for(jj = 0; jj < PyArray_NDIM(input); jj++) {
                // 计算起始和结束索引
                start = PyLong_FromSsize_t(regions[idx + jj]);
                end = PyLong_FromSsize_t(regions[idx + jj +
                                             PyArray_NDIM(input)]);
                // 检查是否分配内存成功
                if (!start || !end) {
                    PyErr_NoMemory();
                    // 跳转到退出标签，释放资源
                    goto exit;
                }
                // 创建切片对象
                slc = PySlice_New(start, end, NULL);
                // 检查是否分配内存成功
                if (!slc) {
                    PyErr_NoMemory();
                    // 跳转到退出标签，释放资源
                    goto exit;
                }
                // 释放起始和结束对象的引用
                Py_DECREF(start);
                Py_DECREF(end);
                start = end = NULL;
                // 将切片对象添加到元组中
                PyTuple_SetItem(tuple, jj, slc);
                slc = NULL;
            }
            // 将元组对象添加到结果列表中
            PyList_SetItem(result, ii, tuple);
            tuple = NULL;
        } else {
            // 如果当前区域无效，将 None 添加到结果列表中
            Py_INCREF(Py_None);
            PyList_SetItem(result, ii, Py_None);
        }
    }

    // 增加结果列表的引用计数
    Py_INCREF(result);

 exit:
    // 减少输入数组、结果列表、元组、起始、结束、切片对象的引用计数
    Py_XDECREF(input);
    Py_XDECREF(result);
    Py_XDECREF(tuple);
    Py_XDECREF(start);
    Py_XDECREF(end);
    Py_XDECREF(slc);
    // 释放动态分配的 regions 数组
    free(regions);
    // 如果发生异常，返回 NULL；否则返回结果列表
    if (PyErr_Occurred()) {
        return NULL;
    } else {
        return result;
    }
/*
   实现 ndimage.value_indices() 函数。
   通过数组数据进行三次遍历。我们使用 ndimage 的 NI_Iterator 来遍历输入数组的所有元素。

   为了支持所有的 numpy 数据类型，我们定义了几个宏，这些宏带有一个用于特定数组数据类型的参数。
   通过使用这些宏，所有的比较都使用与输入数组相同的数据类型进行。

   宏 VALUEINDICES_MINVAL(valType)：
   获取最小值的宏，参数为数据类型的指针。

   宏 VALUEINDICES_MAXVAL(valType)：
   获取最大值的宏，参数为数据类型的指针。

   宏 VALUEINDICES_IGNOREVAL(valType)：
   获取忽略值的宏，参数为数据类型的指针。

   宏 CASE_VALUEINDICES_SET_MINMAX(valType)：
   设置最小值和最大值的情况宏。
   - 获取当前值。
   - 如果忽略值为 None 或当前值不等于忽略值，则：
     - 如果最小最大值未设置，则设置为当前值，并标记已设置。
     - 否则，更新最小值和最大值。

   宏 CASE_VALUEINDICES_MAKEHISTOGRAM(valType)：
   创建直方图的情况宏。
   - 计算可能值的数量。
   - 分配内存以存储直方图。
   - 如果分配成功，则：
     - 初始化点迭代器。
     - 获取数组数据。
     - 遍历数组，更新直方图。

   宏 CASE_VALUEINDICES_GET_VALUEOFFSET(valType)：
   获取值偏移的情况宏。
   - 获取当前值。
   - 计算值的偏移。
   - 检查当前值是否为忽略值。

   宏 CASE_VALUEINDICES_MAKE_VALUEOBJ_FROMOFFSET(valType, ii)：
   从偏移创建值对象的情况宏。
   - 根据偏移和最小值计算值。
   - 将值转换为标量对象。

   NI_ValueIndices 函数实现：
   - 参数包括输入数组 arr、是否忽略值 ignoreValIsNone、忽略值数组 ignorevalArr。
   - 定义了几个局部变量和指针。
   - 获取传入的参数。

*/
static PyObject *NI_ValueIndices(PyObject *self, PyObject *args)
{
    PyArrayObject *arr, *ndxArr, *minMaxArr, *ignorevalArr;
    PyObject *t=NULL, *valObj=NULL, **ndxPtr=NULL, *ndxTuple, *valDict;
    int ignoreValIsNone, valueIsIgnore=0, ndim, j, arrType, minMaxUnset=1;
    NI_Iterator ndiIter;
    char *arrData;
    npy_intp *hist=NULL, *valCtr=NULL, ii, numPossibleVals=0;
    npy_intp arrSize, iterIndex, dims[1];

    /* 获取传入的参数 */
    if (!PyArg_ParseTuple(args, "O!iO!", &PyArray_Type, &arr, &ignoreValIsNone,
            &PyArray_Type, &ignorevalArr))
        return NULL;
    # 获取数组的大小（元素个数）
    arrSize = PyArray_SIZE(arr);
    # 获取数组的数据类型
    arrType = PyArray_TYPE(arr);
    # 获取数组的数据指针，转换为字符类型指针
    arrData = (char *)PyArray_DATA(arr);
    # 获取数组的维度数量
    ndim = PyArray_NDIM(arr);
    # 检查数组的数据类型是否为整数类型，如果不是则设置异常并返回空值
    if (!PyTypeNum_ISINTEGER(arrType)) {
        PyErr_SetString(PyExc_ValueError, "Parameter 'arr' must be an integer array");
        return NULL;
    }

    /* This dictionary is the final return value */
    # 创建一个空的字典作为最终返回的值
    valDict = PyDict_New();
    if (valDict == NULL) return PyErr_NoMemory();

    /* We use a small numpy array for the min and max values, as this will
       take the same datatype as the input array */
    # 创建一个小的 numpy 数组用于存储最小值和最大值，该数组将使用与输入数组相同的数据类型
    dims[0] = 2;
    minMaxArr = (PyArrayObject *)PyArray_SimpleNew(1, dims, arrType);
    if (minMaxArr == NULL) return PyErr_NoMemory();

    /* First pass. Just set the min & max */
    # 第一次遍历，仅设置最小值和最大值
    NI_InitPointIterator(arr, &ndiIter);
    for (iterIndex=0; iterIndex<arrSize; iterIndex++) {
        switch(arrType) {
        # 根据数组的数据类型进行不同的操作
        case NPY_INT8:   CASE_VALUEINDICES_SET_MINMAX(npy_int8); break;
        case NPY_UINT8:  CASE_VALUEINDICES_SET_MINMAX(npy_uint8); break;
        case NPY_INT16:  CASE_VALUEINDICES_SET_MINMAX(npy_int16); break;
        case NPY_UINT16: CASE_VALUEINDICES_SET_MINMAX(npy_uint16); break;
        case NPY_INT32:  CASE_VALUEINDICES_SET_MINMAX(npy_int32); break;
        case NPY_UINT32: CASE_VALUEINDICES_SET_MINMAX(npy_uint32); break;
        case NPY_INT64:  CASE_VALUEINDICES_SET_MINMAX(npy_int64); break;
        case NPY_UINT64: CASE_VALUEINDICES_SET_MINMAX(npy_uint64); break;
        default:
            switch(arrType) {
            case NPY_UINT: CASE_VALUEINDICES_SET_MINMAX(npy_uint); break;
            case NPY_INT:  CASE_VALUEINDICES_SET_MINMAX(npy_int); break;
            }
        }
        NI_ITERATOR_NEXT(ndiIter, arrData);
    }

    /* Second pass, creates a histogram of all the possible values between
       min and max. If min/max were not set, then the array was all ignore
       values. */
    # 第二次遍历，创建在最小值和最大值之间所有可能值的直方图。如果未设置最小值/最大值，则数组全为忽略值。
    if (!minMaxUnset) {
        switch(arrType) {
        case NPY_INT8:   CASE_VALUEINDICES_MAKEHISTOGRAM(npy_int8); break;
        case NPY_UINT8:  CASE_VALUEINDICES_MAKEHISTOGRAM(npy_uint8); break;
        case NPY_INT16:  CASE_VALUEINDICES_MAKEHISTOGRAM(npy_int16); break;
        case NPY_UINT16: CASE_VALUEINDICES_MAKEHISTOGRAM(npy_uint16); break;
        case NPY_INT32:  CASE_VALUEINDICES_MAKEHISTOGRAM(npy_int32); break;
        case NPY_UINT32: CASE_VALUEINDICES_MAKEHISTOGRAM(npy_uint32); break;
        case NPY_INT64:  CASE_VALUEINDICES_MAKEHISTOGRAM(npy_int64); break;
        case NPY_UINT64: CASE_VALUEINDICES_MAKEHISTOGRAM(npy_uint64); break;
        default:
            switch(arrType) {
            case NPY_INT:  CASE_VALUEINDICES_MAKEHISTOGRAM(npy_int); break;
            case NPY_UINT: CASE_VALUEINDICES_MAKEHISTOGRAM(npy_uint); break;
            }
        }
    }
    // 检查指针 hist 是否为非空
    if (hist != NULL) {
        /* 为跟踪分配的索引值而分配本地数据结构
           分配 valCtr 数组来记录每个可能值的计数
           分配 ndxPtr 数组来存储 PyObject 指针
        */
        valCtr = (npy_intp *)calloc(numPossibleVals, sizeof(npy_intp));
        ndxPtr = (PyObject **)calloc(numPossibleVals, sizeof(PyObject *));
        
        // 检查内存分配是否成功
        if (valCtr == NULL)
            PyErr_SetString(PyExc_MemoryError, "Couldn't allocate valCtr");
        else if (ndxPtr == NULL)
            PyErr_SetString(PyExc_MemoryError, "Couldn't allocate ndxPtr");
    }

    }

    /* 清理所有分配的内存资源 */
    // 如果 hist 指针非空，释放 hist 指向的内存
    if (hist != NULL) free(hist);
    // 释放 valCtr 数组所占用的内存
    if (valCtr != NULL) free(valCtr);
    // 释放 ndxPtr 数组所占用的内存
    if (ndxPtr != NULL) free(ndxPtr);
    // 释放对 minMaxArr 的 Python 对象的引用
    Py_DECREF(minMaxArr);

    // 检查是否有异常发生
    if (PyErr_Occurred()) {
        // 如果有异常，释放 valDict 的 Python 对象的引用并返回空指针
        Py_DECREF(valDict);
        return NULL;
    } else
        // 如果没有异常，返回 valDict 的 Python 对象的引用
        return valDict;
}

static PyObject *Py_WatershedIFT(PyObject *obj, PyObject *args)
{
    // 声明需要用到的输入和输出数组对象
    PyArrayObject *input = NULL, *output = NULL, *markers = NULL;
    PyArrayObject *strct = NULL;

    // 解析传入的参数元组，将其转换为相应的数组对象
    if (!PyArg_ParseTuple(args, "O&O&O&O&", NI_ObjectToInputArray, &input,
                    NI_ObjectToInputArray, &markers, NI_ObjectToInputArray,
                    &strct, NI_ObjectToOutputArray, &output))
        // 如果解析失败，跳转到exit标签处
        goto exit;

    // 调用内部函数进行分水岭变换
    NI_WatershedIFT(input, markers, strct, output);
    // 解决输出数组的写回问题（如果有复制的情况）
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入和输出数组对象的引用
    Py_XDECREF(input);
    Py_XDECREF(markers);
    Py_XDECREF(strct);
    Py_XDECREF(output);
    // 如果有错误发生，则返回NULL；否则返回一个空字符串
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_DistanceTransformBruteForce(PyObject *obj,
                                                PyObject *args)
{
    // 声明需要用到的输入和输出数组对象，以及一个整数metric
    PyArrayObject *input = NULL, *output = NULL, *features = NULL;
    PyArrayObject *sampling = NULL;
    int metric;

    // 解析传入的参数元组，将其转换为相应的数组对象和整数
    if (!PyArg_ParseTuple(args, "O&iO&O&O&",
                          NI_ObjectToInputArray, &input,
                          &metric,
                          NI_ObjectToOptionalInputArray, &sampling,
                          NI_ObjectToOptionalOutputArray, &output,
                          NI_ObjectToOptionalOutputArray, &features))
        // 如果解析失败，跳转到exit标签处
        goto exit;

    // 调用内部函数进行距离变换的粗糙实现
    NI_DistanceTransformBruteForce(input, metric, sampling, output, features);
    // 解决输出数组的写回问题（如果有复制的情况）
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放输入和输出数组对象的引用
    Py_XDECREF(input);
    Py_XDECREF(sampling);
    Py_XDECREF(output);
    Py_XDECREF(features);
    // 如果有错误发生，则返回NULL；否则返回一个空字符串
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_DistanceTransformOnePass(PyObject *obj, PyObject *args)
{
    // 声明需要用到的输入和输出数组对象
    PyArrayObject *strct = NULL, *distances = NULL, *features = NULL;

    // 解析传入的参数元组，将其转换为相应的数组对象
    if (!PyArg_ParseTuple(args, "O&O&O&",
                          NI_ObjectToInputArray, &strct,
                          NI_ObjectToInputOutputArray, &distances,
                          NI_ObjectToOptionalOutputArray, &features))
        // 如果解析失败，跳转到exit标签处
        goto exit;

    // 调用内部函数进行一次通道的距离变换
    NI_DistanceTransformOnePass(strct, distances, features);

exit:
    // 释放输入和输出数组对象的引用
    Py_XDECREF(strct);
    Py_XDECREF(distances);
    Py_XDECREF(features);
    // 如果有错误发生，则返回NULL；否则返回一个空字符串
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_EuclideanFeatureTransform(PyObject *obj, PyObject *args)
{
    // 声明需要用到的输入和输出数组对象
    PyArrayObject *input = NULL, *features = NULL, *sampling = NULL;

    // 解析传入的参数元组，将其转换为相应的数组对象
    if (!PyArg_ParseTuple(args, "O&O&O&",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToOptionalInputArray, &sampling,
                          NI_ObjectToOutputArray, &features))
        // 如果解析失败，跳转到exit标签处
        goto exit;

    // 调用内部函数进行欧几里得特征变换
    NI_EuclideanFeatureTransform(input, sampling, features);

exit:
    // 释放输入和输出数组对象的引用
    Py_XDECREF(input);
    Py_XDECREF(sampling);
    Py_XDECREF(features);
    // 如果有错误发生，则返回NULL；否则返回一个空字符串
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}
static void _FreeCoordinateList(PyObject *obj)
{
    // 调用NI_FreeCoordinateList函数释放传入的NI_CoordinateList对象
    NI_FreeCoordinateList((NI_CoordinateList*)PyCapsule_GetPointer(obj, NULL));
}

static PyObject *Py_BinaryErosion(PyObject *obj, PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL, *strct = NULL;
    PyArrayObject *mask = NULL;
    PyObject *cobj = NULL;
    int border_value, invert, center_is_true;
    int changed = 0, return_coordinates;
    NI_CoordinateList *coordinate_list = NULL;
    PyArray_Dims origin = {NULL, 0};

    // 解析传入的参数元组，将其转换为各种类型的数组对象和整数
    if (!PyArg_ParseTuple(args, "O&O&O&O&iO&iii",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToInputArray, &strct,
                          NI_ObjectToOptionalInputArray, &mask,
                          NI_ObjectToOutputArray, &output,
                          &border_value,
                          PyArray_IntpConverter, &origin,
                          &invert, &center_is_true, &return_coordinates)) {
        // 转换失败时跳转到exit标签
        goto exit;
    }
    // 验证origin是否有效
    if (!_validate_origin(input, origin)) {
        // 验证失败时跳转到exit标签
        goto exit;
    }
    // 调用NI_BinaryErosion函数进行二进制侵蚀操作
    if (!NI_BinaryErosion(input, strct, mask, output, border_value,
                          origin.ptr, invert, center_is_true, &changed,
                          return_coordinates ? &coordinate_list : NULL)) {
        // 操作失败时跳转到exit标签
        goto exit;
    }
    // 如果需要返回坐标列表，则创建一个PyCapsule对象
    if (return_coordinates) {
        cobj = PyCapsule_New(coordinate_list, NULL, _FreeCoordinateList);
    }
    // 解析输出数组以便写回（如果有必要）
    PyArray_ResolveWritebackIfCopy(output);

exit:
    // 释放所有输入和输出的Python对象
    Py_XDECREF(input);
    Py_XDECREF(strct);
    Py_XDECREF(mask);
    Py_XDECREF(output);
    // 释放origin.ptr指向的内存
    PyDimMem_FREE(origin.ptr);
    // 如果发生错误，释放cobj并返回NULL
    if (PyErr_Occurred()) {
        Py_XDECREF(cobj);
        return NULL;
    } else {
        // 如果需要返回坐标列表，构建一个包含changed和cobj的元组返回
        if (return_coordinates) {
            return Py_BuildValue("iN", changed, cobj);
        } else {
            // 否则，只返回一个整数changed
            return Py_BuildValue("i", changed);
        }
    }
}

static PyObject *Py_BinaryErosion2(PyObject *obj, PyObject *args)
{
    PyArrayObject *array = NULL, *strct = NULL, *mask = NULL;
    PyObject *cobj = NULL;
    int invert, niter;
    PyArray_Dims origin = {NULL, 0};

    // 解析传入的参数元组，将其转换为各种类型的数组对象、整数和PyCapsule对象
    if (!PyArg_ParseTuple(args, "O&O&O&iO&iO",
                          NI_ObjectToInputOutputArray, &array,
                          NI_ObjectToInputArray, &strct,
                          NI_ObjectToOptionalInputArray,
                          &mask, &niter,
                          PyArray_IntpConverter, &origin,
                          &invert, &cobj)) {
        // 转换失败时跳转到exit标签
        goto exit;
    }
    // 验证origin是否有效
    if (!_validate_origin(array, origin)) {
        // 验证失败时跳转到exit标签
        goto exit;
    }
    // 如果cobj是一个PyCapsule对象，则从中获取NI_CoordinateList指针并调用NI_BinaryErosion2函数
    if (PyCapsule_CheckExact(cobj)) {
        NI_CoordinateList *cobj_data = PyCapsule_GetPointer(cobj, NULL);
        if (!NI_BinaryErosion2(array, strct, mask, niter, origin.ptr, invert,
                               &cobj_data)) {
            // 操作失败时跳转到exit标签
            goto exit;
        }
    }
    else {
        // 如果cobj不是PyCapsule对象，则设置一个运行时错误
        PyErr_SetString(PyExc_RuntimeError, "cannot convert CObject");
    }

exit:
    // 释放所有输入的Python对象
    Py_XDECREF(array);
    Py_XDECREF(strct);
    Py_XDECREF(mask);
    // 释放origin.ptr指向的内存
    PyDimMem_FREE(origin.ptr);
    // 检查是否有 Python 异常发生，如果有则返回 NULL，否则返回一个空的 Python 对象
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyMethodDef methods[] = {
    // 定义模块中的多个方法及其对应的 C 函数
    {"correlate1d",           (PyCFunction)Py_Correlate1D,
     METH_VARARGS, NULL},
    {"correlate",             (PyCFunction)Py_Correlate,
     METH_VARARGS, NULL},
    {"uniform_filter1d",      (PyCFunction)Py_UniformFilter1D,
     METH_VARARGS, NULL},
    {"min_or_max_filter1d",   (PyCFunction)Py_MinOrMaxFilter1D,
        METH_VARARGS, NULL},
    {"min_or_max_filter",     (PyCFunction)Py_MinOrMaxFilter,
        METH_VARARGS, NULL},
    {"rank_filter",           (PyCFunction)Py_RankFilter,
     METH_VARARGS, NULL},
    {"generic_filter",        (PyCFunction)Py_GenericFilter,
     METH_VARARGS, NULL},
    {"generic_filter1d",      (PyCFunction)Py_GenericFilter1D,
     METH_VARARGS, NULL},
    {"fourier_filter",        (PyCFunction)Py_FourierFilter,
     METH_VARARGS, NULL},
    {"fourier_shift",         (PyCFunction)Py_FourierShift,
     METH_VARARGS, NULL},
    {"spline_filter1d",       (PyCFunction)Py_SplineFilter1D,
     METH_VARARGS, NULL},
    {"geometric_transform",   (PyCFunction)Py_GeometricTransform,
        METH_VARARGS, NULL},
    {"zoom_shift",            (PyCFunction)Py_ZoomShift,
     METH_VARARGS, NULL},
    {"find_objects",          (PyCFunction)Py_FindObjects,
     METH_VARARGS, NULL},
    {"value_indices",         (PyCFunction)NI_ValueIndices,
     METH_VARARGS, NULL},
    {"watershed_ift",         (PyCFunction)Py_WatershedIFT,
     METH_VARARGS, NULL},
    {"distance_transform_bf", (PyCFunction)Py_DistanceTransformBruteForce,
     METH_VARARGS, NULL},
    {"distance_transform_op", (PyCFunction)Py_DistanceTransformOnePass,
     METH_VARARGS, NULL},
    {"euclidean_feature_transform",
     (PyCFunction)Py_EuclideanFeatureTransform,
     METH_VARARGS, NULL},
    {"binary_erosion",        (PyCFunction)Py_BinaryErosion,
     METH_VARARGS, NULL},
    {"binary_erosion2",       (PyCFunction)Py_BinaryErosion2,
     METH_VARARGS, NULL},
    // 终止方法数组的标志
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    // 定义 Python 模块的基本信息
    PyModuleDef_HEAD_INIT,
    "_nd_image",  // 模块名称
    NULL,         // 模块文档
    -1,           // 模块状态，-1 表示不保留模块状态
    methods,      // 模块中的方法数组
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__nd_image(void)
{
    // 导入 NumPy 数组处理库
    import_array();
    // 创建并返回 Python 模块对象
    return PyModule_Create(&moduledef);
}
```