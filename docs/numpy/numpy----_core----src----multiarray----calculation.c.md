# `.\numpy\numpy\_core\src\multiarray\calculation.c`

```
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义宏，指定不使用已弃用的 NumPy API 版本

#define _MULTIARRAYMODULE
// 定义宏，指定编译器在链接时使用 multiarray 模块

#define PY_SSIZE_T_CLEAN
// 定义宏，指定 Python 的 Py_ssize_t 类型使用“clean” API

#include <Python.h>
// 包含 Python 核心头文件

#include <structmember.h>
// 包含结构成员相关的头文件

#include "numpy/arrayobject.h"
// 包含 NumPy 数组对象相关的头文件

#include "lowlevel_strided_loops.h"
// 包含低级步进循环的头文件

#include "dtypemeta.h"
// 包含数据类型元信息的头文件

#include "npy_config.h"
// 包含 NumPy 配置的头文件

#include "common.h"
// 包含通用功能的头文件

#include "number.h"
// 包含数字处理相关的头文件

#include "calculation.h"
// 包含计算相关的头文件

#include "array_assign.h"
// 包含数组赋值相关的头文件

static double
power_of_ten(int n)
{
    static const double p10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    // 静态数组，包含 10 的幂次方值

    double ret;
    // 返回值

    if (n < 9) {
        // 如果 n 小于 9，则直接返回对应 p10 数组的值
        ret = p10[n];
    }
    else {
        // 如果 n 大于等于 9，则使用循环计算 10 的 n 次方
        ret = 1e9;
        while (n-- > 9) {
            ret *= 10.;
        }
    }
    return ret;
    // 返回计算结果
}

NPY_NO_EXPORT PyObject *
_PyArray_ArgMinMaxCommon(PyArrayObject *op,
        int axis, PyArrayObject *out, int keepdims,
        npy_bool is_argmax)
{
    PyArrayObject *ap = NULL, *rp = NULL;
    // 定义 PyArrayObject 类型的指针变量 ap 和 rp

    PyArray_ArgFunc* arg_func = NULL;
    // 定义 PyArray_ArgFunc 类型的指针变量 arg_func

    char *ip, *func_name;
    // 定义字符指针变量 ip 和 func_name

    npy_intp *rptr;
    // 定义 npy_intp 类型的指针变量 rptr

    npy_intp i, n, m;
    // 定义 npy_intp 类型的变量 i, n, m

    int elsize;
    // 定义整型变量 elsize

    // 保存 axis 的副本，因为后续调用 PyArray_CheckAxis 会改变它
    int axis_copy = axis;

    npy_intp _shape_buf[NPY_MAXDIMS];
    // 定义长度为 NPY_MAXDIMS 的 npy_intp 类型数组 _shape_buf

    npy_intp *out_shape;
    // 定义 npy_intp 类型的指针变量 out_shape

    // 保存原始数组的维度数和形状，当 keepdims 为 True 时有用
    npy_intp* original_op_shape = PyArray_DIMS(op);
    int out_ndim = PyArray_NDIM(op);
    // 获取原始数组的维度数和形状

    NPY_BEGIN_THREADS_DEF;
    // 定义一个宏，用于线程控制，开始线程定义

    if ((ap = (PyArrayObject *)PyArray_CheckAxis(op, &axis, 0)) == NULL) {
        // 检查并获取经过轴处理后的数组 ap，如果失败则返回空指针
        return NULL;
    }

    /*
     * We need to permute the array so that axis is placed at the end.
     * And all other dimensions are shifted left.
     */
    // 我们需要重新排列数组，使得指定的 axis 放在最后一个位置，其他维度左移
    if (axis != PyArray_NDIM(ap)-1) {
        PyArray_Dims newaxes;
        npy_intp dims[NPY_MAXDIMS];
        int j;

        newaxes.ptr = dims;
        newaxes.len = PyArray_NDIM(ap);

        for (j = 0; j < axis; j++) {
            dims[j] = j;
        }
        for (j = axis; j < PyArray_NDIM(ap) - 1; j++) {
            dims[j] = j + 1;
        }
        dims[PyArray_NDIM(ap) - 1] = axis;

        // 对 ap 进行转置操作，得到新的数组 op
        op = (PyArrayObject *)PyArray_Transpose(ap, &newaxes);
        Py_DECREF(ap);

        if (op == NULL) {
            return NULL;
        }
    }
    else {
        op = ap;
    }

    // 获取原生字节顺序的连续副本
    PyArray_Descr *descr = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(op));
    if (descr == NULL) {
        return NULL;
    }

    // 从原始数组创建一个新的数组对象 ap
    ap = (PyArrayObject *)PyArray_FromArray(op, descr, NPY_ARRAY_DEFAULT);
    Py_DECREF(op);

    if (ap == NULL) {
        return NULL;
    }

    // 决定输出数组的形状
    if (!keepdims) {
        out_ndim = PyArray_NDIM(ap) - 1;
        out_shape = PyArray_DIMS(ap);
    }
    else {
        out_shape = _shape_buf;
        // 如果需要，将输出形状初始化为形状缓冲区的内容
        if (axis_copy == NPY_RAVEL_AXIS) {
            // 如果轴复制标志为 NPY_RAVEL_AXIS，将输出形状的所有维度设为 1
            for (int i = 0; i < out_ndim; i++) {
                out_shape[i] = 1;
            }
        }
        else {
            /*
             * 虽然 `ap` 可能已经转置，但是对于 `out` 来说我们可以忽略这一点，
             * 因为转置仅重新排列大小为 1 的 `axis`（不改变内存布局）。
             */
            // 复制原始操作形状到输出形状，除了指定的轴外，其它维度保持不变
            memcpy(out_shape, original_op_shape, out_ndim * sizeof(npy_intp));
            out_shape[axis] = 1;
        }
    }

    // 如果是求取 argmax
    if (is_argmax) {
        // 设置函数名为 "argmax"
        func_name = "argmax";
        // 获取对应数据类型的 argmax 函数指针
        arg_func = PyDataType_GetArrFuncs(PyArray_DESCR(ap))->argmax;
    }
    else {
        // 设置函数名为 "argmin"
        func_name = "argmin";
        // 获取对应数据类型的 argmin 函数指针
        arg_func = PyDataType_GetArrFuncs(PyArray_DESCR(ap))->argmin;
    }
    // 如果未找到合适的函数指针，抛出类型错误异常
    if (arg_func == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "data type not ordered");
        goto fail;
    }
    // 计算元素大小（以字节为单位）
    elsize = PyArray_ITEMSIZE(ap);
    // 获取数组的最后一个维度大小
    m = PyArray_DIMS(ap)[PyArray_NDIM(ap)-1];
    // 如果最后一个维度大小为 0，则抛出值错误异常
    if (m == 0) {
        PyErr_Format(PyExc_ValueError,
                    "attempt to get %s of an empty sequence",
                    func_name);
        goto fail;
    }

    // 如果输出对象为 NULL
    if (!out) {
        // 创建一个新的数组对象 rp，以 intp 类型的描述符和指定的输出形状
        rp = (PyArrayObject *)PyArray_NewFromDescr(
                Py_TYPE(ap), PyArray_DescrFromType(NPY_INTP),
                out_ndim, out_shape, NULL, NULL,
                0, (PyObject *)ap);
        // 如果创建失败，则跳转到 fail 标签处理异常
        if (rp == NULL) {
            goto fail;
        }
    }
    else {
        // 如果输出对象不为 NULL，检查其维度和形状是否与预期匹配
        if ((PyArray_NDIM(out) != out_ndim) ||
                !PyArray_CompareLists(PyArray_DIMS(out), out_shape,
                                        out_ndim)) {
            PyErr_Format(PyExc_ValueError,
                    "output array does not match result of np.%s.",
                    func_name);
            goto fail;
        }
        // 使用给定的数组对象 out 创建一个新的数组对象 rp，写入数据时若有副本则写回
        rp = (PyArrayObject *)PyArray_FromArray(out,
                              PyArray_DescrFromType(NPY_INTP),
                              NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        // 如果创建失败，则跳转到 fail 标签处理异常
        if (rp == NULL) {
            goto fail;
        }
    }

    // 开始线程保护，以数组对象 ap 的描述符
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap));
    // 计算 ap 的总元素个数除以 m，得到 n 的值
    n = PyArray_SIZE(ap)/m;
    // 获取 rp 对象的数据指针
    rptr = (npy_intp *)PyArray_DATA(rp);
    // 遍历数组 ap 中的元素，每次处理 elsize*m 个元素
    for (ip = PyArray_DATA(ap), i = 0; i < n; i++, ip += elsize*m) {
        // 调用 arg_func 处理 ip 指向的数据，结果存入 rptr 指向的位置
        arg_func(ip, m, rptr, ap);
        // 移动 rptr 到下一个位置
        rptr += 1;
    }
    // 结束线程保护，以数组对象 ap 的描述符
    NPY_END_THREADS_DESCR(PyArray_DESCR(ap));

    // 释放数组对象 ap 的引用
    Py_DECREF(ap);
    /* 如果需要，触发 WRITEBACKIFCOPY */
    // 如果 out 不为 NULL 且 out 不等于 rp，则解析 WRITEBACKIFCOPY
    if (out != NULL && out != rp) {
        PyArray_ResolveWritebackIfCopy(rp);
        // 释放 rp 的引用，将其设置为 out，并增加其引用计数
        Py_DECREF(rp);
        rp = out;
        Py_INCREF(rp);
    }
    // 返回 rp 对象的 PyObject 指针形式
    return (PyObject *)rp;

 fail:
    // 处理失败的情况：释放数组对象 ap 的引用，同时释放 rp 对象的引用
    Py_DECREF(ap);
    Py_XDECREF(rp);
    // 返回 NULL 指针，表示发生异常
    return NULL;
/*NUMPY_API
 * ArgMaxWithKeepdims
 */
NPY_NO_EXPORT PyObject*
_PyArray_ArgMaxWithKeepdims(PyArrayObject *op,
        int axis, PyArrayObject *out, int keepdims)
{
    // 调用共同的最大值和最小值查找函数，返回最大值的索引，保持维度信息
    return _PyArray_ArgMinMaxCommon(op, axis, out, keepdims, 1);
}

/*NUMPY_API
 * ArgMax
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMax(PyArrayObject *op, int axis, PyArrayObject *out)
{
    // 调用共同的最大值和最小值查找函数，返回最大值的索引，不保持维度信息
    return _PyArray_ArgMinMaxCommon(op, axis, out, 0, 1);
}

/*NUMPY_API
 * ArgMinWithKeepdims
 */
NPY_NO_EXPORT PyObject *
_PyArray_ArgMinWithKeepdims(PyArrayObject *op,
        int axis, PyArrayObject *out, int keepdims)
{
    // 调用共同的最大值和最小值查找函数，返回最小值的索引，保持维度信息
    return _PyArray_ArgMinMaxCommon(op, axis, out, keepdims, 0);
}

/*NUMPY_API
 * ArgMin
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMin(PyArrayObject *op, int axis, PyArrayObject *out)
{
    // 调用共同的最大值和最小值查找函数，返回最小值的索引，不保持维度信息
    return _PyArray_ArgMinMaxCommon(op, axis, out, 0, 0);
}

/*NUMPY_API
 * Max
 */
NPY_NO_EXPORT PyObject *
PyArray_Max(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;

    // 检查并调整轴，以确保它在合法范围内
    arr = (PyArrayObject *)PyArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    // 使用通用的规约函数找到数组中的最大值
    ret = PyArray_GenericReduceFunction(arr, n_ops.maximum, axis,
                                        PyArray_DESCR(arr)->type_num, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Min
 */
NPY_NO_EXPORT PyObject *
PyArray_Min(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;

    // 检查并调整轴，以确保它在合法范围内
    arr=(PyArrayObject *)PyArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    // 使用通用的规约函数找到数组中的最小值
    ret = PyArray_GenericReduceFunction(arr, n_ops.minimum, axis,
                                        PyArray_DESCR(arr)->type_num, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Ptp (peak-to-peak)
 */
NPY_NO_EXPORT PyObject *
PyArray_Ptp(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;
    PyObject *obj1 = NULL, *obj2 = NULL;

    // 检查并调整轴，以确保它在合法范围内
    arr=(PyArrayObject *)PyArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    // 获取数组沿指定轴的最大值
    obj1 = PyArray_Max(arr, axis, out);
    if (obj1 == NULL) {
        goto fail;
    }
    // 获取数组沿指定轴的最小值
    obj2 = PyArray_Min(arr, axis, NULL);
    if (obj2 == NULL) {
        goto fail;
    }
    Py_DECREF(arr);
    // 计算最大值和最小值的差值
    if (out) {
        ret = PyObject_CallFunction(n_ops.subtract, "OOO", out, obj2, out);
    }
    else {
        ret = PyNumber_Subtract(obj1, obj2);
    }
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    return ret;

 fail:
    Py_XDECREF(arr);
    Py_XDECREF(obj1);
    Py_XDECREF(obj2);
    return NULL;
}

/*NUMPY_API
 * Std (standard deviation)
 */
NPY_NO_EXPORT PyObject *
PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
            int variance)
{
    // 调用内部函数，计算数组的标准差或方差
    return __New_PyArray_Std(self, axis, rtype, out, variance, 0);
}

/* Helper function for PyArray_Std */
NPY_NO_EXPORT PyObject *
__New_PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
                  int variance, int num)
{
    PyObject *obj1 = NULL, *obj2 = NULL, *obj3 = NULL;
    // 这里是实现标准差或方差计算的具体逻辑，未在这里注释
}
    // 声明指向 NumPy 数组的指针，用于存储三个数组对象及一个返回对象
    PyArrayObject *arr1 = NULL, *arr2 = NULL, *arrnew = NULL;
    PyObject *ret = NULL, *newshape = NULL;
    int i, n;
    npy_intp val;

    // 检查并返回 self 对象中指定轴的 PyArrayObject 对象
    arrnew = (PyArrayObject *)PyArray_CheckAxis(self, &axis, 0);
    if (arrnew == NULL) {
        return NULL;
    }

    /* 计算并重塑均值 */
    // 确保 arrnew 是一个数组对象，并计算其指定轴上的均值
    arr1 = (PyArrayObject *)PyArray_EnsureAnyArray(
                    PyArray_Mean(arrnew, axis, rtype, NULL));
    if (arr1 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    // 获取 arrnew 的维度数，并创建一个新的元组 newshape
    n = PyArray_NDIM(arrnew);
    newshape = PyTuple_New(n);
    if (newshape == NULL) {
        Py_DECREF(arr1);
        Py_DECREF(arrnew);
        return NULL;
    }

    // 为 newshape 元组赋值，根据 arrnew 的维度设置新的形状
    for (i = 0; i < n; i++) {
        if (i == axis) {
            val = 1;
        }
        else {
            val = PyArray_DIM(arrnew,i);
        }
        PyTuple_SET_ITEM(newshape, i, PyLong_FromSsize_t(val));
    }

    // 根据 newshape 元组重塑 arr1 数组对象
    arr2 = (PyArrayObject *)PyArray_Reshape(arr1, newshape);
    Py_DECREF(arr1);
    Py_DECREF(newshape);
    if (arr2 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    /* 计算 x = x - mx */
    // 确保 arrnew 和 arr2 是数组对象，计算它们的减法
    arr1 = (PyArrayObject *)PyArray_EnsureAnyArray(
                PyNumber_Subtract((PyObject *)arrnew, (PyObject *)arr2));
    Py_DECREF(arr2);
    if (arr1 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    /* 计算 x * x */
    // 如果 arr1 是复数数组，取其共轭；否则直接增加其引用计数
    if (PyArray_ISCOMPLEX(arr1)) {
        obj3 = PyArray_Conjugate(arr1, NULL);
    }
    else {
        obj3 = (PyObject *)arr1;
        Py_INCREF(arr1);
    }
    if (obj3 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    // 确保 arr1 和 obj3 是数组对象，使用通用的二进制函数计算它们的乘法
    arr2 = (PyArrayObject *)PyArray_EnsureAnyArray(
                PyArray_GenericBinaryFunction((PyObject *)arr1, obj3,
                                               n_ops.multiply));
    Py_DECREF(arr1);
    Py_DECREF(obj3);
    if (arr2 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    // 如果 arr2 是复数数组，获取其实部；根据 rtype 设置新的数据类型
    if (PyArray_ISCOMPLEX(arr2)) {
        obj3 = PyObject_GetAttrString((PyObject *)arr2, "real");
        switch(rtype) {
        case NPY_CDOUBLE:
            rtype = NPY_DOUBLE;
            break;
        case NPY_CFLOAT:
            rtype = NPY_FLOAT;
            break;
        case NPY_CLONGDOUBLE:
            rtype = NPY_LONGDOUBLE;
            break;
        }
    }
    else {
        obj3 = (PyObject *)arr2;
        Py_INCREF(arr2);
    }
    if (obj3 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    /* 计算 add.reduce(x*x,axis) */
    // 使用通用的减少函数计算 arr3（即 obj3）沿指定轴的加法
    obj1 = PyArray_GenericReduceFunction((PyArrayObject *)obj3, n_ops.add,
                                         axis, rtype, NULL);
    Py_DECREF(obj3);
    Py_DECREF(arr2);
    if (obj1 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    // 获取 arrnew 在指定轴上的维度，减去 num，如果结果为 0，则设置为 1
    n = PyArray_DIM(arrnew,axis);
    Py_DECREF(arrnew);
    n = (n-num);
    if (n == 0) {
        n = 1;
    }

    // 创建一个新的 PyFloat 对象，表示计算结果的倒数
    obj2 = PyFloat_FromDouble(1.0/((double )n));
    if (obj2 == NULL) {
        Py_DECREF(obj1);
        return NULL;
    }

    // 计算最终的结果，obj1 乘以 obj2
    ret = PyNumber_Multiply(obj1, obj2);
    // 减少对象 obj1 的引用计数
    Py_DECREF(obj1);
    // 减少对象 obj2 的引用计数
    Py_DECREF(obj2);

    // 如果方差为假值（0或NULL），执行以下代码块
    if (!variance) {
        // 将返回值 ret 转换为 PyArrayObject 对象
        arr1 = (PyArrayObject *)PyArray_EnsureAnyArray(ret);
        /* 对返回值进行平方根运算 */
        ret = PyArray_GenericUnaryFunction(arr1, n_ops.sqrt);
        // 减少 arr1 对象的引用计数
        Py_DECREF(arr1);
    }

    // 如果返回值 ret 为 NULL，则返回 NULL
    if (ret == NULL) {
        return NULL;
    }

    // 如果 self 是一个确切的 PyArray 对象，跳转到 finish 标签
    if (PyArray_CheckExact(self)) {
        goto finish;
    }

    // 如果 self 是 PyArray 对象，并且 self 和 ret 的类型相同，跳转到 finish 标签
    if (PyArray_Check(self) && Py_TYPE(self) == Py_TYPE(ret)) {
        goto finish;
    }

    // 将返回值 ret 转换为 PyArrayObject 对象
    arr1 = (PyArrayObject *)PyArray_EnsureArray(ret);

    // 如果 arr1 为 NULL，则返回 NULL
    if (arr1 == NULL) {
        return NULL;
    }

    // 将 arr1 转换为一个以 self 类型为基础的视图，返回值赋给 ret
    ret = PyArray_View(arr1, NULL, Py_TYPE(self));

    // 减少 arr1 对象的引用计数
    Py_DECREF(arr1);
/*NUMPY_API
 * Round
 */
NPY_NO_EXPORT PyObject *
PyArray_Round(PyArrayObject *a, int decimals, PyArrayObject *out)
{
    PyObject *f, *ret = NULL, *tmp, *op1, *op2;
    int ret_int=0;
    PyArray_Descr *my_descr;
    // 如果提供了输出数组 out，并且其形状与输入数组 a 不匹配，则报错并返回 NULL
    if (out && (PyArray_SIZE(out) != PyArray_SIZE(a))) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid output shape");
        return NULL;
    }
    // ...以下代码省略，未提供的部分不需要添加注释
}
    # 如果输入数组是复数数组，则进行以下操作
    if (PyArray_ISCOMPLEX(a)) {
        # 定义变量用于存储部分结果和数组对象
        PyObject *part;
        PyObject *round_part;
        PyObject *arr;
        int res;

        # 如果提供了输出数组对象，则使用它，增加其引用计数
        if (out) {
            arr = (PyObject *)out;
            Py_INCREF(arr);
        }
        # 否则复制输入数组a生成新的数组arr
        else {
            arr = PyArray_Copy(a);
            if (arr == NULL) {
                return NULL;
            }
        }

        /* arr.real = a.real.round(decimals) */
        # 获取复数数组a的实部，并确保其为数组对象
        part = PyObject_GetAttrString((PyObject *)a, "real");
        if (part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        # 对实部数组进行四舍五入操作，结果存储在round_part中
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        # 将四舍五入后的实部数组赋值给arr的实部属性
        res = PyObject_SetAttrString(arr, "real", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(arr);
            return NULL;
        }

        /* arr.imag = a.imag.round(decimals) */
        # 获取复数数组a的虚部，并确保其为数组对象
        part = PyObject_GetAttrString((PyObject *)a, "imag");
        if (part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        # 对虚部数组进行四舍五入操作，结果存储在round_part中
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        # 将四舍五入后的虚部数组赋值给arr的虚部属性
        res = PyObject_SetAttrString(arr, "imag", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(arr);
            return NULL;
        }
        # 返回处理后的复数数组对象arr
        return arr;
    }
    /* do the most common case first */
    # 处理最常见的情况：decimals >= 0
    if (decimals >= 0) {
        # 如果输入数组是整数数组
        if (PyArray_ISINTEGER(a)) {
            # 如果提供了输出数组对象out，则将输入数组a的内容复制到out中
            if (out) {
                if (PyArray_AssignArray(out, a,
                            NULL, NPY_DEFAULT_ASSIGN_CASTING) < 0) {
                    return NULL;
                }
                # 增加输出数组对象的引用计数并返回
                Py_INCREF(out);
                return (PyObject *)out;
            }
            else {
                # 增加输入数组a的引用计数并返回
                Py_INCREF(a);
                return (PyObject *)a;
            }
        }
        # 如果decimals为0，则调用n_ops.rint函数对输入数组a进行舍入操作
        if (decimals == 0) {
            if (out) {
                return PyObject_CallFunction(n_ops.rint, "OO", a, out);
            }
            return PyObject_CallFunction(n_ops.rint, "O", a);
        }
        # 如果decimals大于0，则指定op1为n_ops.multiply，op2为n_ops.true_divide
        op1 = n_ops.multiply;
        op2 = n_ops.true_divide;
    }
    else {
        # 如果decimals小于0，则指定op1为n_ops.true_divide，op2为n_ops.multiply，并将decimals取反
        op1 = n_ops.true_divide;
        op2 = n_ops.multiply;
        decimals = -decimals;
    }
    // 如果输出对象 out 为空，则根据输入数组 a 的类型决定如何处理
    if (!out) {
        // 如果输入数组 a 是整数类型，则设置返回整数标志并创建双精度浮点数类型描述符
        if (PyArray_ISINTEGER(a)) {
            ret_int = 1;
            my_descr = PyArray_DescrFromType(NPY_DOUBLE);
        }
        else {
            // 否则，增加输入数组 a 的描述符的引用计数，并将其作为描述符
            Py_INCREF(PyArray_DESCR(a));
            my_descr = PyArray_DESCR(a);
        }
        // 使用数组 a 的维度和描述符创建一个空的数组对象 out
        out = (PyArrayObject *)PyArray_Empty(PyArray_NDIM(a), PyArray_DIMS(a),
                                             my_descr,
                                             PyArray_ISFORTRAN(a));
        // 如果创建出错，返回空指针
        if (out == NULL) {
            return NULL;
        }
    }
    else {
        // 如果输出对象不为空，增加其引用计数
        Py_INCREF(out);
    }
    // 根据给定的小数位数创建一个 Python 浮点数对象 f
    f = PyFloat_FromDouble(power_of_ten(decimals));
    // 如果创建出错，返回空指针
    if (f == NULL) {
        return NULL;
    }
    // 调用指定函数 op1，传递数组 a、浮点数 f 和输出对象 out 作为参数
    ret = PyObject_CallFunction(op1, "OOO", a, f, out);
    // 如果调用出错，跳转至清理代码块 finish
    if (ret == NULL) {
        goto finish;
    }
    // 调用函数 n_ops.rint，传递 ret 作为参数，执行四舍五入操作
    tmp = PyObject_CallFunction(n_ops.rint, "OO", ret, ret);
    // 如果调用出错，释放 ret 并跳转至清理代码块 finish
    if (tmp == NULL) {
        Py_DECREF(ret);
        ret = NULL;
        goto finish;
    }
    // 释放 tmp 对象
    Py_DECREF(tmp);
    // 再次调用指定函数 op2，传递 ret、f 和 ret 作为参数
    tmp = PyObject_CallFunction(op2, "OOO", ret, f, ret);
    // 如果调用出错，释放 ret 并跳转至清理代码块 finish
    if (tmp == NULL) {
        Py_DECREF(ret);
        ret = NULL;
        goto finish;
    }
    // 释放 tmp 对象
    Py_DECREF(tmp);

 finish:
    // 释放浮点数对象 f
    Py_DECREF(f);
    // 释放输出对象 out
    Py_DECREF(out);
    // 如果返回整数标志为真且 ret 不为空，则将 ret 转换为输入数组 a 的类型并返回
    if (ret_int && ret != NULL) {
        Py_INCREF(PyArray_DESCR(a));
        tmp = PyArray_CastToType((PyArrayObject *)ret,
                                 PyArray_DESCR(a), PyArray_ISFORTRAN(a));
        Py_DECREF(ret);
        return tmp;
    }
    // 返回 ret 对象
    return ret;
/*NUMPY_API
 * Mean
 */
NPY_NO_EXPORT PyObject *
PyArray_Mean(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *obj1 = NULL, *obj2 = NULL, *ret;
    PyArrayObject *arr;

    // 检查并调整轴向，确保数组符合要求
    arr = (PyArrayObject *)PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    // 对数组进行通用约简操作，使用加法操作
    obj1 = PyArray_GenericReduceFunction(arr, n_ops.add, axis,
                                         rtype, out);
    // 创建一个包含数组指定轴向维度的浮点数对象
    obj2 = PyFloat_FromDouble((double)PyArray_DIM(arr,axis));
    Py_DECREF(arr);
    // 检查对象是否创建成功，否则清理并返回空
    if (obj1 == NULL || obj2 == NULL) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    // 如果没有指定输出对象，则进行真实除法操作
    if (!out) {
        ret = PyNumber_TrueDivide(obj1, obj2);
    }
    // 否则，调用分裂函数进行除法操作
    else {
        ret = PyObject_CallFunction(n_ops.divide, "OOO", out, obj2, out);
    }
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    return ret;
}

/*NUMPY_API
 * Any
 */
NPY_NO_EXPORT PyObject *
PyArray_Any(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyObject *arr, *ret;

    // 检查并调整轴向，确保数组符合要求
    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    // 对数组进行通用约简操作，使用逻辑或操作
    ret = PyArray_GenericReduceFunction((PyArrayObject *)arr,
                                        n_ops.logical_or, axis,
                                        NPY_BOOL, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * All
 */
NPY_NO_EXPORT PyObject *
PyArray_All(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyObject *arr, *ret;

    // 检查并调整轴向，确保数组符合要求
    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    // 对数组进行通用约简操作，使用逻辑与操作
    ret = PyArray_GenericReduceFunction((PyArrayObject *)arr,
                                        n_ops.logical_and, axis,
                                        NPY_BOOL, out);
    Py_DECREF(arr);
    return ret;
}


/*NUMPY_API
 * Clip
 */
NPY_NO_EXPORT PyObject *
PyArray_Clip(PyArrayObject *self, PyObject *min, PyObject *max, PyArrayObject *out)
{
    // 将 None 视为 NULL 处理
    if (min == Py_None) {
        min = NULL;
    }
    if (max == Py_None) {
        max = NULL;
    }

    // 必须设置 max 或 min，否则报错
    if ((max == NULL) && (min == NULL)) {
        PyErr_SetString(PyExc_ValueError,
                        "array_clip: must set either max or min");
        return NULL;
    }

    // 根据条件调用对应的最小值或最大值函数，或者调用剪裁函数
    if (min == NULL) {
        return PyObject_CallFunctionObjArgs(n_ops.minimum, self, max, out, NULL);
    }
    else if (max == NULL) {
        return PyObject_CallFunctionObjArgs(n_ops.maximum, self, min, out, NULL);
    }
    else {
        return PyObject_CallFunctionObjArgs(n_ops.clip, self, min, max, out, NULL);
    }
}


/*NUMPY_API
 * Conjugate
 */
NPY_NO_EXPORT PyObject *
PyArray_Conjugate(PyArrayObject *self, PyArrayObject *out)
{
    // TO DO: Implement conjugation function
    // 尚未实现的共轭函数
    return NULL;
}
    # 检查数组是否为复数类型、对象类型或用户自定义类型
    if (PyArray_ISCOMPLEX(self) || PyArray_ISOBJECT(self) ||
            PyArray_ISUSERDEF(self)) {
        # 如果未提供输出数组（out），则调用通用的一元函数，返回其共轭
        if (out == NULL) {
            return PyArray_GenericUnaryFunction(self,
                                                n_ops.conjugate);
        }
        # 如果提供了输出数组（out），则调用通用的二元函数，将结果存储在输出数组中
        else {
            return PyArray_GenericBinaryFunction((PyObject *)self,
                                                 (PyObject *)out,
                                                 n_ops.conjugate);
        }
    }
    else {
        PyArrayObject *ret;
        # 如果数组不是数值类型，发出弃用警告
        if (!PyArray_ISNUMBER(self)) {
            /* 2017-05-04, 1.13 */
            if (DEPRECATE("attempting to conjugate non-numeric dtype; this "
                          "will error in the future to match the behavior of "
                          "np.conjugate") < 0) {
                return NULL;
            }
        }
        # 如果提供了输出数组（out），则尝试将 self 的内容复制到输出数组
        if (out) {
            # 使用默认转换方式，将 self 的内容赋值给输出数组
            if (PyArray_AssignArray(out, self,
                        NULL, NPY_DEFAULT_ASSIGN_CASTING) < 0) {
                return NULL;
            }
            # 将输出数组赋值给返回对象 ret
            ret = out;
        }
        # 如果未提供输出数组（out），则直接将 self 赋值给返回对象 ret
        else {
            ret = self;
        }
        # 增加返回对象 ret 的引用计数
        Py_INCREF(ret);
        # 返回增加引用计数后的返回对象 ret
        return (PyObject *)ret;
    }
/*NUMPY_API
 * Trace
 */
/* 定义一个不导出的函数 PyArray_Trace，接受一个 PyArrayObject 类型的参数 self，
 * 以及几个整型参数 offset, axis1, axis2, rtype 和一个 PyArrayObject 类型的参数 out。
 * 返回一个 PyObject 指针。
 */
NPY_NO_EXPORT PyObject *
PyArray_Trace(PyArrayObject *self, int offset, int axis1, int axis2,
              int rtype, PyArrayObject *out)
{
    PyObject *diag = NULL, *ret = NULL;

    /* 调用 PyArray_Diagonal 函数，传入 self, offset, axis1, axis2 参数，返回值赋给 diag */
    diag = PyArray_Diagonal(self, offset, axis1, axis2);
    /* 如果 diag 为 NULL，则直接返回 NULL */
    if (diag == NULL) {
        return NULL;
    }
    /* 调用 PyArray_GenericReduceFunction 函数，传入 diag, n_ops.add, -1, rtype, out 参数，
     * 返回值赋给 ret
     */
    ret = PyArray_GenericReduceFunction((PyArrayObject *)diag, n_ops.add, -1, rtype, out);
    /* 减少 diag 的引用计数 */
    Py_DECREF(diag);
    /* 返回 ret 指针 */
    return ret;
}
```