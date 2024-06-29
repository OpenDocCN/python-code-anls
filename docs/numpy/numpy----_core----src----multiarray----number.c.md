# `.\numpy\numpy\_core\src\multiarray\number.c`

```py
/*
 * 定义宏，确保使用的 NumPy API 版本没有废弃的部分
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义宏，标识 _MULTIARRAYMODULE
 */
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T 的旧定义，确保使用最新版本的类型定义
 */
#define PY_SSIZE_T_CLEAN

/*
 * 包含 Python 核心头文件
 */
#include <Python.h>

/*
 * 包含结构成员头文件，用于定义结构体成员
 */
#include <structmember.h>

/*
 * 包含 NumPy 的数组对象头文件
 */
#include "numpy/arrayobject.h"

/*
 * 包含 NumPy 配置文件
 */
#include "npy_config.h"

/*
 * 包含 NumPy 的 Python 兼容性功能
 */
#include "npy_pycompat.h"

/*
 * 包含 NumPy 的导入功能
 */
#include "npy_import.h"

/*
 * 包含 NumPy 的公共功能
 */
#include "common.h"

/*
 * 包含 NumPy 的数值操作
 */
#include "number.h"

/*
 * 包含 NumPy 的临时省略功能
 */
#include "temp_elide.h"

/*
 * 包含 NumPy 的二元操作重载
 */
#include "binop_override.h"

/*
 * 包含 NumPy 的通用函数操作重载
 */
#include "ufunc_override.h"

/*
 * 包含 NumPy 的抽象数据类型功能
 */
#include "abstractdtypes.h"

/*
 * 包含 NumPy 的数据类型转换功能
 */
#include "convert_datatype.h"

/*************************************************************************
 ****************   实现数字协议 ****************************
 *************************************************************************/

/*
 * 这个全局变量不在全局数据结构中，避免在 multiarraymodule.h 中需要包含 NumericOps 结构的定义
 * 在模块初始化期间以线程安全的方式填充
 */
NPY_NO_EXPORT NumericOps n_ops; /* 注意：静态对象被初始化为零 */

/*
 * 函数的前向声明，可能会重新安排函数而不是将其移动
 */
static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo));
static PyObject *
array_inplace_matrix_multiply(PyArrayObject *m1, PyObject *m2);

/*
 * 字典可以包含任何数值操作，按名称存储
 * 不包含在字典中的操作将不会被更改
 */

/* FIXME - 宏包含返回值 */
#define SET(op) \
    res = PyDict_GetItemStringRef(dict, #op, &temp); \
    if (res == -1) { \
        return -1; \
    } \
    else if (res == 1) { \
        if (!(PyCallable_Check(temp))) { \
            Py_DECREF(temp); \
            return -1; \
        } \
        Py_XSETREF(n_ops.op, temp); \
    }

/*
 * 设置数值操作函数的实现
 */
NPY_NO_EXPORT int
_PyArray_SetNumericOps(PyObject *dict)
{
    PyObject *temp = NULL;
    int res;
    
    /*
     * 设置每种数值操作函数，通过从字典中获取并检查其有效性，然后进行设置
     */
    SET(add);
    SET(subtract);
    SET(multiply);
    SET(divide);
    SET(remainder);
    SET(divmod);
    SET(power);
    SET(square);
    SET(reciprocal);
    SET(_ones_like);
    SET(sqrt);
    SET(cbrt);
    SET(negative);
    SET(positive);
    SET(absolute);
    SET(invert);
    SET(left_shift);
    SET(right_shift);
    SET(bitwise_and);
    SET(bitwise_or);
    SET(bitwise_xor);

    return 0;
}


注释结束。
    # 初始化一组函数指针，用于对应不同的操作符
    SET(less);
    SET(less_equal);
    SET(equal);
    SET(not_equal);
    SET(greater);
    SET(greater_equal);
    SET(floor_divide);
    SET(true_divide);
    SET(logical_or);
    SET(logical_and);
    SET(floor);
    SET(ceil);
    SET(maximum);
    SET(minimum);
    SET(rint);
    SET(conjugate);
    SET(matmul);
    SET(clip);
    
    # 为矩阵乘法（matmul）初始化静态全局变量
    npy_static_pydata.axes_1d_obj_kwargs = Py_BuildValue(
            "{s, [(i), (i, i), (i)]}", "axes", -1, -2, -1, -1);
    # 检查初始化结果是否为 NULL，若是则返回 -1
    if (npy_static_pydata.axes_1d_obj_kwargs == NULL) {
        return -1;
    }
    
    # 为二维矩阵乘法（matmul）初始化静态全局变量
    npy_static_pydata.axes_2d_obj_kwargs = Py_BuildValue(
            "{s, [(i, i), (i, i), (i, i)]}", "axes", -2, -1, -2, -1, -2, -1);
    # 检查初始化结果是否为 NULL，若是则返回 -1
    if (npy_static_pydata.axes_2d_obj_kwargs == NULL) {
        return -1;
    }
    
    # 初始化成功，返回 0 表示正常完成
    return 0;
# 返回包含操作关键字的字典对象，用于向操作函数传递参数
static PyObject *
_get_keywords(int rtype, PyArrayObject *out)
{
    PyObject *kwds = NULL;
    
    // 如果数据类型不是 NPY_NOTYPE 或者输出对象不为空
    if (rtype != NPY_NOTYPE || out != NULL) {
        // 创建一个空的 Python 字典对象
        kwds = PyDict_New();
        
        // 如果数据类型不是 NPY_NOTYPE
        if (rtype != NPY_NOTYPE) {
            // 根据数据类型创建一个 NumPy 数组描述符对象
            PyArray_Descr *descr;
            descr = PyArray_DescrFromType(rtype);
            
            // 如果描述符对象有效
            if (descr) {
                // 向字典中添加 "dtype" 键，值为描述符对象的 Python 对象表示
                PyDict_SetItemString(kwds, "dtype", (PyObject *)descr);
                Py_DECREF(descr);
            }
        }
        
        // 如果输出对象不为空，则向字典中添加 "out" 键，值为输出对象的 Python 对象表示
        if (out != NULL) {
            PyDict_SetItemString(kwds, "out", (PyObject *)out);
        }
    }
    
    // 返回组装好的关键字字典
    return kwds;
}

# 使用给定操作符进行数组的通用约简操作
NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;

    // 使用 Py_BuildValue 函数创建一个参数元组，包含 m1 和 axis
    args = Py_BuildValue("(Oi)", m1, axis);
    // 调用 _get_keywords 函数获取操作关键字字典
    kwds = _get_keywords(rtype, out);
    // 获取操作对象 op 的 "reduce" 方法
    meth = PyObject_GetAttrString(op, "reduce");
    
    // 如果获取成功并且 meth 是可调用的
    if (meth && PyCallable_Check(meth)) {
        // 调用 meth 方法，传递 args 和 kwds 作为参数
        ret = PyObject_Call(meth, args, kwds);
    }
    
    // 释放 args 和 meth 对象的引用计数
    Py_DECREF(args);
    Py_DECREF(meth);
    // 释放 kwds 对象的引用计数，使用 Py_XDECREF 避免空指针异常
    Py_XDECREF(kwds);
    
    // 返回操作结果
    return ret;
}

# 使用给定操作符进行数组的通用累积操作
NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;

    // 使用 Py_BuildValue 函数创建一个参数元组，包含 m1 和 axis
    args = Py_BuildValue("(Oi)", m1, axis);
    // 调用 _get_keywords 函数获取操作关键字字典
    kwds = _get_keywords(rtype, out);
    // 获取操作对象 op 的 "accumulate" 方法
    meth = PyObject_GetAttrString(op, "accumulate");
    
    // 如果获取成功并且 meth 是可调用的
    if (meth && PyCallable_Check(meth)) {
        // 调用 meth 方法，传递 args 和 kwds 作为参数
        ret = PyObject_Call(meth, args, kwds);
    }
    
    // 释放 args 和 meth 对象的引用计数
    Py_DECREF(args);
    Py_DECREF(meth);
    // 释放 kwds 对象的引用计数，使用 Py_XDECREF 避免空指针异常
    Py_XDECREF(kwds);
    
    // 返回操作结果
    return ret;
}

# 使用给定操作符进行数组的通用二元操作
NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyObject *m1, PyObject *m2, PyObject *op)
{
    // 直接调用 PyObject_CallFunctionObjArgs 函数，传递 op、m1 和 m2 作为参数
    return PyObject_CallFunctionObjArgs(op, m1, m2, NULL);
}

# 使用给定操作符进行数组的通用一元操作
NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    // 直接调用 PyObject_CallFunctionObjArgs 函数，传递 op 和 m1 作为参数
    return PyObject_CallFunctionObjArgs(op, m1, NULL);
}

# 使用给定操作符进行数组的原位二元操作
static PyObject *
PyArray_GenericInplaceBinaryFunction(PyArrayObject *m1,
                                     PyObject *m2, PyObject *op)
{
    // 直接调用 PyObject_CallFunctionObjArgs 函数，传递 op、m1 和 m2 作为参数
    return PyObject_CallFunctionObjArgs(op, m1, m2, m1, NULL);
}

# 使用给定操作符进行数组的原位一元操作
static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    // 直接调用 PyObject_CallFunctionObjArgs 函数，传递 op、m1 和 m1 作为参数
    return PyObject_CallFunctionObjArgs(op, m1, m1, NULL);
}

# 数组加法的特定实现
static PyObject *
array_add(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 使用 BINOP_GIVE_UP_IF_NEEDED 宏，处理 m1 和 m2，调用 nb_add 函数或 array_add 函数
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_add, array_add);
    
    // 尝试使用 try_binary_elide 函数进行二进制操作的优化
    if (try_binary_elide(m1, m2, &array_inplace_add, &res, 1)) {
        return res;
    }
    
    // 调用 PyArray_GenericBinaryFunction 函数进行通用的二元操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.add);
}

# 数组减法的特定实现
static PyObject *
array_subtract(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 使用 BINOP_GIVE_UP_IF_NEEDED 宏，处理 m1 和 m2，调用 nb_subtract 函数或 array_subtract 函数
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_subtract, array_subtract);
    
    // 尝试使用 try_binary_elide 函数进行二进制操作的优化
    if (try_binary_elide(m1, m2, &array_inplace_subtract, &res, 0)) {
        return res;
    }
    
    // 调用 PyArray_GenericBinaryFunction 函数进行通用的二元操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.subtract);
}

# 数组乘法的特定实现
static PyObject *
array_multiply(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 使用 BINOP_GIVE_UP_IF_NEEDED 宏，处理 m1 和 m2，调用 nb_multiply 函数或 array_multiply 函数
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_multiply, array_multiply);
    
    // 尝试使用 try_binary_elide 函数进行二进制操作的优化
    if (try_binary_elide(m1, m2, &array_inplace_multiply, &res, 0)) {
        return res;
    }
    
    // 调用 PyArray_GenericBinaryFunction 函数进行通用的二元操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.multiply);
}

# 以下代码继续完成数组的特定操作，但超出了给定的代码块范围，因此不在此注释之内
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_multiply, array_multiply);
    # 尝试在必要时放弃二元操作，使用 nb_multiply 或 array_multiply 函数处理 m1 和 m2

    if (try_binary_elide(m1, m2, &array_inplace_multiply, &res, 1)) {
        # 尝试优化二元操作，如果成功则返回结果 res
        return res;
    }

    # 如果优化失败，调用通用的数组二元操作函数 PyArray_GenericBinaryFunction 处理 m1 和 m2，
    # 使用乘法操作符 n_ops.multiply
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.multiply);
/*
 * 返回 PyArray_GenericBinaryFunction 的余数计算结果
 * 如果需要，使用宏 BINOP_GIVE_UP_IF_NEEDED 确保 m1 和 m2 是有效的操作数
 */
static PyObject *
array_remainder(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_remainder, array_remainder);
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.remainder);
}

/*
 * 返回 PyArray_GenericBinaryFunction 的商余运算结果
 * 使用宏 BINOP_GIVE_UP_IF_NEEDED 确保 m1 和 m2 是有效的操作数
 */
static PyObject *
array_divmod(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_divmod, array_divmod);
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.divmod);
}

/*
 * 返回 PyArray_GenericBinaryFunction 的矩阵乘法结果
 * 使用宏 BINOP_GIVE_UP_IF_NEEDED 确保 m1 和 m2 是有效的操作数
 */
static PyObject *
array_matrix_multiply(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_matrix_multiply, array_matrix_multiply);
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.matmul);
}

/*
 * 原地执行矩阵乘法，并返回结果
 * 使用 INPLACE_GIVE_UP_IF_NEEDED 宏确保 self 和 other 是有效的操作数
 * 根据 self 的维度数选择正确的 kwargs，用于调用 n_ops.matmul
 * 如果操作失败，根据错误类型设置相应的错误信息
 */
static PyObject *
array_inplace_matrix_multiply(PyArrayObject *self, PyObject *other)
{
    INPLACE_GIVE_UP_IF_NEEDED(self, other,
            nb_inplace_matrix_multiply, array_inplace_matrix_multiply);

    PyObject *args = PyTuple_Pack(3, self, other, self);
    if (args == NULL) {
        return NULL;
    }
    PyObject *kwargs;

    /*
     * 不像 `matmul(a, b, out=a)`，我们确保结果不会进行广播，
     * 如果没有 `out` 参数，结果维度比 `a` 少。
     * 因为 matmul 的签名是 '(n?,k),(k,m?)->(n?,m?)'，这种情况正好是第二个操作数具有核心维度的情况。
     *
     * 这里的错误可能会令人困惑，但现在我们通过传递正确的 `axes=` 来强制执行。
     */
    if (PyArray_NDIM(self) == 1) {
        kwargs = npy_static_pydata.axes_1d_obj_kwargs;
    }
    else {
        kwargs = npy_static_pydata.axes_2d_obj_kwargs;
    }
    PyObject *res = PyObject_Call(n_ops.matmul, args, kwargs);
    Py_DECREF(args);

    if (res == NULL) {
        /*
         * 如果异常是 AxisError，说明 axes 参数设置不正确，
         * 这通常是因为第二个操作数不是二维的。
         */
        if (PyErr_ExceptionMatches(npy_static_pydata.AxisError)) {
            PyErr_SetString(PyExc_ValueError,
                "inplace matrix multiplication requires the first operand to "
                "have at least one and the second at least two dimensions.");
        }
    }

    return res;
}

/*
 * 确定对象是否是标量，并在是标量时将其转换为双精度数，
 * 将结果放入 out_exponent 参数中，并返回相应的“标量种类”。
 * 如果对象不是标量（或有其他错误条件），返回 NPY_NOSCALAR，out_exponent 未定义。
 */
static NPY_SCALARKIND
is_scalar_with_conversion(PyObject *o2, double* out_exponent)
{
    PyObject *temp;
    const int optimize_fpexps = 1;

    if (PyLong_Check(o2)) {
        long tmp = PyLong_AsLong(o2);
        if (error_converting(tmp)) {
            PyErr_Clear();
            return NPY_NOSCALAR;
        }
        *out_exponent = (double)tmp;
        return NPY_INTPOS_SCALAR;
    }

    if (optimize_fpexps && PyFloat_Check(o2)) {
        *out_exponent = PyFloat_AsDouble(o2);
        return NPY_FLOAT_SCALAR;
    }
    /* 继续处理其他类型的标量情况 */
    # 检查 o2 是否为 NumPy 数组
    if (PyArray_Check(o2)) {
        # 如果 o2 是零维数组，并且是整数或者允许浮点数表达式优化且是浮点数类型
        if ((PyArray_NDIM((PyArrayObject *)o2) == 0) &&
                ((PyArray_ISINTEGER((PyArrayObject *)o2) ||
                 (optimize_fpexps && PyArray_ISFLOAT((PyArrayObject *)o2))))) {
            # 获取 o2 的浮点数表示
            temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
            # 如果转换失败，返回 NPY_NOSCALAR
            if (temp == NULL) {
                return NPY_NOSCALAR;
            }
            # 将 o2 的浮点数值赋给 out_exponent
            *out_exponent = PyFloat_AsDouble(o2);
            Py_DECREF(temp);
            # 如果 o2 是整数类型，返回 NPY_INTPOS_SCALAR
            if (PyArray_ISINTEGER((PyArrayObject *)o2)) {
                return NPY_INTPOS_SCALAR;
            }
            else { /* ISFLOAT */
                return NPY_FLOAT_SCALAR;
            }
        }
    }
    # 如果 o2 是标量整数或者允许浮点数表达式优化且是标量浮点数
    else if (PyArray_IsScalar(o2, Integer) ||
                (optimize_fpexps && PyArray_IsScalar(o2, Floating))) {
        # 获取 o2 的浮点数表示
        temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
        # 如果转换失败，返回 NPY_NOSCALAR
        if (temp == NULL) {
            return NPY_NOSCALAR;
        }
        # 将 o2 的浮点数值赋给 out_exponent
        *out_exponent = PyFloat_AsDouble(o2);
        Py_DECREF(temp);

        # 如果 o2 是整数类型，返回 NPY_INTPOS_SCALAR
        if (PyArray_IsScalar(o2, Integer)) {
                return NPY_INTPOS_SCALAR;
        }
        else { /* IsScalar(o2, Floating) */
            return NPY_FLOAT_SCALAR;
        }
    }
    # 如果 o2 是 Python 索引类型
    else if (PyIndex_Check(o2)) {
        # 获取 o2 的索引值
        PyObject* value = PyNumber_Index(o2);
        Py_ssize_t val;
        # 如果获取索引值失败，清除错误并返回 NPY_NOSCALAR
        if (value == NULL) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return NPY_NOSCALAR;
        }
        # 将索引值转换为 Py_ssize_t 类型
        val = PyLong_AsSsize_t(value);
        Py_DECREF(value);
        # 如果转换失败，清除错误并返回 NPY_NOSCALAR
        if (error_converting(val)) {
            PyErr_Clear();
            return NPY_NOSCALAR;
        }
        # 将转换后的值赋给 out_exponent，并返回 NPY_INTPOS_SCALAR
        *out_exponent = (double) val;
        return NPY_INTPOS_SCALAR;
    }
    # 如果 o2 不满足以上条件，返回 NPY_NOSCALAR
    return NPY_NOSCALAR;
/*
 * optimize float array or complex array to a scalar power
 * returns 0 on success, -1 if no optimization is possible
 * the result is in value (can be NULL if an error occurred)
 */
static int
fast_scalar_power(PyObject *o1, PyObject *o2, int inplace,
                  PyObject **value)
{
    double exponent;
    NPY_SCALARKIND kind;   /* NPY_NOSCALAR is not scalar */

    // 检查 o1 是否是数组，并且不是对象数组，同时尝试从 o2 中提取指数值
    if (PyArray_Check(o1) &&
            !PyArray_ISOBJECT((PyArrayObject *)o1) &&
            ((kind=is_scalar_with_conversion(o2, &exponent))>0)) {
        PyArrayObject *a1 = (PyArrayObject *)o1;
        PyObject *fastop = NULL;
        
        // 如果数组是浮点数或复数类型
        if (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) {
            // 根据指数值选择合适的快速操作函数
            if (exponent == 1.0) {
                fastop = n_ops.positive;
            }
            else if (exponent == -1.0) {
                fastop = n_ops.reciprocal;
            }
            else if (exponent ==  0.0) {
                fastop = n_ops._ones_like;
            }
            else if (exponent ==  0.5) {
                fastop = n_ops.sqrt;
            }
            else if (exponent ==  2.0) {
                fastop = n_ops.square;
            }
            else {
                return -1;  // 如果找不到匹配的快速操作，返回 -1
            }

            // 根据 inplace 标志选择相应的操作函数
            if (inplace || can_elide_temp_unary(a1)) {
                *value = PyArray_GenericInplaceUnaryFunction(a1, fastop);
            }
            else {
                *value = PyArray_GenericUnaryFunction(a1, fastop);
            }
            return 0;  // 返回成功标志
        }
        // 对于指数为 2.0 的情况，特殊处理
        else if (exponent == 2.0) {
            fastop = n_ops.square;
            // 如果 inplace 标志为真，则使用原地操作函数
            if (inplace) {
                *value = PyArray_GenericInplaceUnaryFunction(a1, fastop);
            }
            else {
                // 只有 FLOAT_SCALAR 和整数类型会特别处理
                if (kind == NPY_FLOAT_SCALAR && PyArray_ISINTEGER(a1)) {
                    // 将数组类型转换为双精度浮点型
                    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_DOUBLE);
                    a1 = (PyArrayObject *)PyArray_CastToType(a1, dtype,
                            PyArray_ISFORTRAN(a1));
                    if (a1 != NULL) {
                        // 执行转换后的操作
                        *value = PyArray_GenericInplaceUnaryFunction(a1, fastop);
                        Py_DECREF(a1);
                    }
                }
                else {
                    // 否则，使用一般的操作函数
                    *value = PyArray_GenericUnaryFunction(a1, fastop);
                }
            }
            return 0;  // 返回成功标志
        }
    }
    // 如果没有找到适合的快速操作，返回 -1
    return -1;
}

static PyObject *
array_power(PyObject *a1, PyObject *o2, PyObject *modulo)
{
    PyObject *value = NULL;
    if (modulo != Py_None) {
        /* 如果给定的模数不是 None，则返回 NotImplemented
           这里暂时不支持模幂运算（gh-8804）
        */
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    BINOP_GIVE_UP_IF_NEEDED(a1, o2, nb_power, array_power);
    /* 如果需要放弃，此处进行操作，包括传递参数 a1, o2, nb_power, array_power */

    if (fast_scalar_power(a1, o2, 0, &value) != 0) {
        /* 如果快速标量幂运算返回非零值，表示出错，则使用通用的数组二元函数来计算结果
           使用参数 a1, o2, n_ops.power 调用 PyArray_GenericBinaryFunction 函数
        */
        value = PyArray_GenericBinaryFunction(a1, o2, n_ops.power);
    }
    /* 返回计算得到的 value */
    return value;
static PyObject *
array_positive(PyArrayObject *m1)
{
    // 检查是否可以省略临时一元操作
    if (can_elide_temp_unary(m1)) {
        // 如果可以省略，则使用原地一元函数执行正操作
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.positive);
    }
    // 否则，使用通用一元函数执行正操作
    return PyArray_GenericUnaryFunction(m1, n_ops.positive);
}

static PyObject *
array_negative(PyArrayObject *m1)
{
    // 检查是否可以省略临时一元操作
    if (can_elide_temp_unary(m1)) {
        // 如果可以省略，则使用原地一元函数执行负操作
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.negative);
    }
    // 否则，使用通用一元函数执行负操作
    return PyArray_GenericUnaryFunction(m1, n_ops.negative);
}

static PyObject *
array_absolute(PyArrayObject *m1)
{
    // 检查是否可以省略临时一元操作并且不是复数数组
    if (can_elide_temp_unary(m1) && !PyArray_ISCOMPLEX(m1)) {
        // 如果可以省略且不是复数数组，则使用原地一元函数执行绝对值操作
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.absolute);
    }
    // 否则，使用通用一元函数执行绝对值操作
    return PyArray_GenericUnaryFunction(m1, n_ops.absolute);
}

static PyObject *
array_invert(PyArrayObject *m1)
{
    // 检查是否可以省略临时一元操作
    if (can_elide_temp_unary(m1)) {
        // 如果可以省略，则使用原地一元函数执行按位取反操作
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.invert);
    }
    // 否则，使用通用一元函数执行按位取反操作
    return PyArray_GenericUnaryFunction(m1, n_ops.invert);
}

static PyObject *
array_left_shift(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 如果必要，放弃二元操作
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_lshift, array_left_shift);
    // 尝试省略二元操作
    if (try_binary_elide(m1, m2, &array_inplace_left_shift, &res, 0)) {
        return res;
    }
    // 否则，使用通用二元函数执行左移操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_right_shift(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 如果必要，放弃二元操作
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_rshift, array_right_shift);
    // 尝试省略二元操作
    if (try_binary_elide(m1, m2, &array_inplace_right_shift, &res, 0)) {
        return res;
    }
    // 否则，使用通用二元函数执行右移操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_bitwise_and(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 如果必要，放弃二元操作
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_and, array_bitwise_and);
    // 尝试省略二元操作
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_and, &res, 1)) {
        return res;
    }
    // 否则，使用通用二元函数执行按位与操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_bitwise_or(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 如果必要，放弃二元操作
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_or, array_bitwise_or);
    // 尝试省略二元操作
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_or, &res, 1)) {
        return res;
    }
    // 否则，使用通用二元函数执行按位或操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_bitwise_xor(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 如果必要，放弃二元操作
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_xor, array_bitwise_xor);
    // 尝试省略二元操作
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_xor, &res, 1)) {
        return res;
    }
    // 否则，使用通用二元函数执行按位异或操作
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2)
{
    // 如果必要，放弃原地二元操作
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_add, array_inplace_add);
    // 使用通用原地二元函数执行加法操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2)
{
    // 如果必要，放弃原地二元操作
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_subtract, array_inplace_subtract);
    # 调用 NumPy C API 中的通用原地二元函数，执行矩阵 m1 和 m2 的减法操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.subtract);
static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2)
{
    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_multiply, array_inplace_multiply);
    // 使用通用的原地二进制函数处理乘法操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.multiply);
}

static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2)
{
    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_remainder, array_inplace_remainder);
    // 使用通用的原地二进制函数处理取余操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.remainder);
}

static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    // 忽略掉 modulo 参数
    PyObject *value = NULL;

    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            a1, o2, nb_inplace_power, array_inplace_power);
    // 尝试使用快速标量幂运算，如果失败则使用通用的原地二进制函数处理幂运算
    if (fast_scalar_power((PyObject *)a1, o2, 1, &value) != 0) {
        value = PyArray_GenericInplaceBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}

static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2)
{
    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_lshift, array_inplace_left_shift);
    // 使用通用的原地二进制函数处理左移操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2)
{
    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_rshift, array_inplace_right_shift);
    // 使用通用的原地二进制函数处理右移操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_and, array_inplace_bitwise_and);
    // 使用通用的原地二进制函数处理按位与操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_or, array_inplace_bitwise_or);
    // 使用通用的原地二进制函数处理按位或操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    // 如果需要放弃原地操作，则直接返回
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_xor, array_inplace_bitwise_xor);
    // 使用通用的原地二进制函数处理按位异或操作
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_floor_divide(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    // 如果需要放弃二进制操作，则直接返回
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_floor_divide, array_floor_divide);
    // 尝试通过二进制简化操作执行地板除法，如果成功则返回结果
    if (try_binary_elide(m1, m2, &array_inplace_floor_divide, &res, 0)) {
        return res;
    }
    // 使用通用的二进制函数处理地板除法
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.floor_divide);
}

static PyObject *
array_true_divide(PyObject *m1, PyObject *m2)
{
    PyObject *res;
    PyArrayObject *a1 = (PyArrayObject *)m1;

    // 如果需要放弃二进制操作，则直接返回
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_true_divide, array_true_divide);
    // 如果 m1 是精确的数组对象，并且是浮点型或复数型，并且可以尝试简化二进制操作，则返回结果
    if (PyArray_CheckExact(m1) &&
            (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) &&
            try_binary_elide(m1, m2, &array_inplace_true_divide, &res, 0)) {
        return res;
    }
    // 使用通用的二进制函数处理真除法
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.true_divide);
}
/*
 * Perform inplace floor division on two NumPy arrays.
 * If necessary, give up inplace operation and revert to generic operation.
 */
static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_floor_divide, array_inplace_floor_divide);
    // Call a generic inplace binary function for floor division
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.floor_divide);
}

/*
 * Perform inplace true division on two NumPy arrays.
 * If necessary, give up inplace operation and revert to generic operation.
 */
static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_true_divide, array_inplace_true_divide);
    // Call a generic inplace binary function for true division
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.true_divide);
}

/*
 * Check if a NumPy array is nonzero.
 * If the array has one element, convert to bool; handle potential recursion.
 * If the array has zero elements, issue a deprecation warning.
 * If the array has more than one element, raise a ValueError.
 */
static int
_array_nonzero(PyArrayObject *mp)
{
    npy_intp n;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        int res;
        if (Py_EnterRecursiveCall(" while converting array to bool")) {
            return -1;
        }
        // Get the nonzero function from array descriptor and apply to data
        res = PyDataType_GetArrFuncs(PyArray_DESCR(mp))->nonzero(PyArray_DATA(mp), mp);
        /* nonzero has no way to indicate an error, but one can occur */
        if (PyErr_Occurred()) {
            res = -1;
        }
        Py_LeaveRecursiveCall();
        return res;
    }
    else if (n == 0) {
        /* 2017-09-25, 1.14 */
        // Issue deprecation warning for truth value of empty array
        if (DEPRECATE("The truth value of an empty array is ambiguous. "
                      "Returning False, but in future this will result in an error. "
                      "Use `array.size > 0` to check that an array is not empty.") < 0) {
            return -1;
        }
        return 0;
    }
    else {
        // Raise ValueError for ambiguous truth value of array with more than one element
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array "
                        "with more than one element is ambiguous. "
                        "Use a.any() or a.all()");
        return -1;
    }
}

/*
 * Convert a NumPy array to a scalar if allowed, and apply the given builtin function to it.
 * Handle recursion if array holds references.
 */
NPY_NO_EXPORT PyObject *
array_scalar_forward(PyArrayObject *v,
                     PyObject *(*builtin_func)(PyObject *),
                     const char *where)
{
    // Check if the array can be converted to a scalar
    if (check_is_convertible_to_scalar(v) < 0) {
        return NULL;
    }

    PyObject *scalar;
    scalar = PyArray_GETITEM(v, PyArray_DATA(v));
    if (scalar == NULL) {
        return NULL;
    }

    /* Need to guard against recursion if our array holds references */
    if (PyDataType_REFCHK(PyArray_DESCR(v))) {
        PyObject *res;
        // Enter recursion guard with given location string
        if (Py_EnterRecursiveCall(where) != 0) {
            Py_DECREF(scalar);
            return NULL;
        }
        // Apply builtin_func to scalar and handle recursion exit
        res = builtin_func(scalar);
        Py_DECREF(scalar);
        Py_LeaveRecursiveCall();
        return res;
    }
    else {
        // Apply builtin_func to scalar without recursion handling
        PyObject *res;
        res = builtin_func(scalar);
        Py_DECREF(scalar);
        return res;
    }
}

/*
 * Convert a NumPy array to a floating-point scalar.
 */
NPY_NO_EXPORT PyObject *
array_float(PyArrayObject *v)
{
    // Forward array to scalar conversion for float
    return array_scalar_forward(v, &PyNumber_Float, " in ndarray.__float__");
}
// 定义函数 `array_int`，接受一个 PyArrayObject 类型的参数 `v`，并将其传递给 `array_scalar_forward` 函数进行处理，返回处理结果
array_int(PyArrayObject *v)
{
    // 调用 `array_scalar_forward` 函数，传递参数 `v` 和 `&PyNumber_Long`，并附加错误信息字符串 " in ndarray.__int__"
    return array_scalar_forward(v, &PyNumber_Long, " in ndarray.__int__");
}

// 定义静态函数 `array_index`，接受一个 PyArrayObject 类型的参数 `v`
static PyObject *
array_index(PyArrayObject *v)
{
    // 如果 `v` 不是整数标量数组或者不是零维数组
    if (!PyArray_ISINTEGER(v) || PyArray_NDIM(v) != 0) {
        // 设置类型错误异常，并返回 NULL
        PyErr_SetString(PyExc_TypeError,
            "only integer scalar arrays can be converted to a scalar index");
        return NULL;
    }
    // 返回 `v` 的数据部分对应的 Python 对象
    return PyArray_GETITEM(v, PyArray_DATA(v));
}

// 定义 `array_as_number` 结构体，实现 PyNumberMethods 结构体
NPY_NO_EXPORT PyNumberMethods array_as_number = {
    // 定义各种数学运算方法
    .nb_add = array_add,
    .nb_subtract = array_subtract,
    .nb_multiply = array_multiply,
    .nb_remainder = array_remainder,
    .nb_divmod = array_divmod,
    .nb_power = (ternaryfunc)array_power,
    .nb_negative = (unaryfunc)array_negative,
    .nb_positive = (unaryfunc)array_positive,
    .nb_absolute = (unaryfunc)array_absolute,
    .nb_bool = (inquiry)_array_nonzero,
    .nb_invert = (unaryfunc)array_invert,
    .nb_lshift = array_left_shift,
    .nb_rshift = array_right_shift,
    .nb_and = array_bitwise_and,
    .nb_xor = array_bitwise_xor,
    .nb_or = array_bitwise_or,

    // 类型转换方法
    .nb_int = (unaryfunc)array_int,
    .nb_float = (unaryfunc)array_float,

    // 原地运算方法
    .nb_inplace_add = (binaryfunc)array_inplace_add,
    .nb_inplace_subtract = (binaryfunc)array_inplace_subtract,
    .nb_inplace_multiply = (binaryfunc)array_inplace_multiply,
    .nb_inplace_remainder = (binaryfunc)array_inplace_remainder,
    .nb_inplace_power = (ternaryfunc)array_inplace_power,
    .nb_inplace_lshift = (binaryfunc)array_inplace_left_shift,
    .nb_inplace_rshift = (binaryfunc)array_inplace_right_shift,
    .nb_inplace_and = (binaryfunc)array_inplace_bitwise_and,
    .nb_inplace_xor = (binaryfunc)array_inplace_bitwise_xor,
    .nb_inplace_or = (binaryfunc)array_inplace_bitwise_or,

    // 浮点数运算方法
    .nb_floor_divide = array_floor_divide,
    .nb_true_divide = array_true_divide,
    .nb_inplace_floor_divide = (binaryfunc)array_inplace_floor_divide,
    .nb_inplace_true_divide = (binaryfunc)array_inplace_true_divide,

    // 返回整数索引的方法
    .nb_index = (unaryfunc)array_index,

    // 矩阵乘法方法
    .nb_matrix_multiply = array_matrix_multiply,
    .nb_inplace_matrix_multiply = (binaryfunc)array_inplace_matrix_multiply,
};
```