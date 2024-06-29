# `.\numpy\numpy\_core\src\umath\_scaled_float_dtype.c`

```py
/*
 * This file implements a basic scaled float64 DType.  The reason is to have
 * a simple parametric DType for testing.  It is not meant to be a useful
 * DType by itself, but due to the scaling factor has similar properties as
 * a Unit DType.
 *
 * The code here should be seen as a work in progress.  Some choices are made
 * to test certain code paths, but that does not mean that they must not
 * be modified.
 *
 * NOTE: The tests were initially written using private API and ABI, ideally
 *       they should be replaced/modified with versions using public API.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION    /* Ensure no deprecated API is used */
#define _MULTIARRAYMODULE                        /* Define _MULTIARRAYMODULE */
#define _UMATHMODULE                             /* Define _UMATHMODULE */

#include "numpy/ndarrayobject.h"                 /* Include necessary headers */
#include "numpy/ufuncobject.h"

#include "array_method.h"                        /* Additional headers specific to implementation */
#include "common.h"
#include "numpy/npy_math.h"
#include "convert_datatype.h"
#include "dtypemeta.h"
#include "dispatching.h"
#include "gil_utils.h"
#include "multiarraymodule.h"

typedef struct {
    PyArray_Descr base;                         /* Define structure for scaled float descriptor */
    double scaling;                             /* Scaling factor for the float */
} PyArray_SFloatDescr;

static PyArray_DTypeMeta PyArray_SFloatDType;   /* Define DTypeMeta object for scaled float */
static PyArray_SFloatDescr SFloatSingleton;     /* Singleton instance of scaled float descriptor */


static int
sfloat_is_known_scalar_type(PyArray_DTypeMeta *NPY_UNUSED(cls), PyTypeObject *type)
{
    /* Check if the given type is a float type */
    if (type == &PyFloat_Type) {
        return 1;   /* Return true if it's a float */
    }
    return 0;       /* Otherwise, return false */
}


static PyArray_Descr *
sfloat_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    Py_INCREF(&SFloatSingleton);   /* Increment reference count for the singleton descriptor */
    return (PyArray_Descr *)&SFloatSingleton;   /* Return the descriptor as PyArray_Descr* */
}


static PyArray_Descr *
sfloat_discover_from_pyobject(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return sfloat_default_descr(cls);   /* Return the default descriptor for a given Python object */
}


static PyArray_DTypeMeta *
sfloat_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    /* Common dtype function to check compatibility with another dtype */
    if (NPY_DT_is_legacy(other) && other->type_num == NPY_DOUBLE) {
        Py_INCREF(cls);
        return cls;   /* Return self if the other dtype is legacy and double */
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;   /* Otherwise, return NotImplemented */
}


static PyArray_Descr *
sfloat_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    PyArray_SFloatDescr *sf1 = (PyArray_SFloatDescr *)descr1;
    PyArray_SFloatDescr *sf2 = (PyArray_SFloatDescr *)descr2;
    /* Choose the descriptor with the larger scaling factor */
    if (sf1->scaling >= sf2->scaling) {
        Py_INCREF(descr1);
        return descr1;   /* Return the first descriptor if its scaling is larger or equal */
    }
    Py_INCREF(descr2);
    return descr2;       /* Otherwise, return the second descriptor */
}


/*
 * Implement minimal getitem and setitem to make this DType mostly(?) safe to
 * expose in Python.
 * TODO: This should not use the old-style API, but the new-style is missing!
*/

static PyObject *
sfloat_getitem(char *data, PyArrayObject *arr)
{
    PyArray_SFloatDescr *descr = (PyArray_SFloatDescr *)PyArray_DESCR(arr);
    double value;

    memcpy(&value, data, sizeof(double));   /* Copy the double value from data */
    return PyFloat_FromDouble(value * descr->scaling);   /* Return scaled float value */
}


static int
sfloat_setitem(PyObject *obj, char *data, PyArrayObject *arr)
{
    # 检查对象是否为精确的浮点数类型，如果不是则抛出异常并返回-1
    if (!PyFloat_CheckExact(obj)) {
        PyErr_SetString(PyExc_NotImplementedError,
                "Currently only accepts floats");
        return -1;
    }

    # 获取数组对象的浮点数描述符
    PyArray_SFloatDescr *descr = (PyArray_SFloatDescr *)PyArray_DESCR(arr);
    
    # 将Python浮点数对象转换为C语言的double类型值
    double value = PyFloat_AsDouble(obj);
    
    # 将值按描述符中的缩放因子进行缩放处理
    value /= descr->scaling;

    # 将处理后的double值复制到数据缓冲区中
    memcpy(data, &value, sizeof(double));
    
    # 操作成功完成，返回0表示成功
    return 0;
/* Special DType methods and the descr->f slot storage */

/* 定义结构体变量 sfloat_slots，包含一系列与特定数据类型相关的函数指针 */
NPY_DType_Slots sfloat_slots = {
    .discover_descr_from_pyobject = &sfloat_discover_from_pyobject,
    .is_known_scalar_type = &sfloat_is_known_scalar_type,
    .default_descr = &sfloat_default_descr,
    .common_dtype = &sfloat_common_dtype,
    .common_instance = &sfloat_common_instance,
    .f = {
        .getitem = (PyArray_GetItemFunc *)&sfloat_getitem,
        .setitem = (PyArray_SetItemFunc *)&sfloat_setitem,
    }
};

/* 定义静态变量 SFloatSingleton，描述浮点数类型的属性 */
static PyArray_SFloatDescr SFloatSingleton = {{
        .byteorder = '|',  /* 不需要处理字节顺序 */
        .flags = NPY_USE_GETITEM | NPY_USE_SETITEM,
        .type_num = -1,
        .elsize = sizeof(double),
        .alignment = NPY_ALIGNOF(double),
    },
    .scaling = 1,
};

/* 复制浮点数描述符并进行缩放 */
static PyArray_Descr *sfloat_scaled_copy(PyArray_SFloatDescr *self, double factor) {
    PyArray_SFloatDescr *new = PyObject_New(
            PyArray_SFloatDescr, (PyTypeObject *)&PyArray_SFloatDType);
    if (new == NULL) {
        return NULL;
    }
    /* 不复制 PyObject_HEAD 部分 */
    memcpy((char *)new + sizeof(PyObject),
            (char *)self + sizeof(PyObject),
            sizeof(PyArray_SFloatDescr) - sizeof(PyObject));

    new->scaling = new->scaling * factor;
    return (PyArray_Descr *)new;
}

/* Python 函数，用于复制浮点数类型并进行缩放 */
PyObject *python_sfloat_scaled_copy(PyArray_SFloatDescr *self, PyObject *arg)
{
    if (!PyFloat_Check(arg)) {
        PyErr_SetString(PyExc_TypeError,
                "Scaling factor must be a python float.");
        return NULL;
    }
    double factor = PyFloat_AsDouble(arg);

    return (PyObject *)sfloat_scaled_copy(self, factor);
}

/* 返回浮点数类型的 scaling 属性 */
static PyObject *sfloat_get_scaling(PyArray_SFloatDescr *self, PyObject *NPY_UNUSED(args))
{
    return PyFloat_FromDouble(self->scaling);
}

/* 返回浮点数类型的 __reduce__ 方法 */
static PyObject *sfloat___reduce__(PyArray_SFloatDescr *self)
{
    return Py_BuildValue("(O(d))", Py_TYPE(self), self->scaling);
}

/* 定义浮点数类型的方法表 */
PyMethodDef sfloat_methods[] = {
    {"scaled_by",
         (PyCFunction)python_sfloat_scaled_copy, METH_O,
        "Method to get a dtype copy with different scaling, mainly to "
        "avoid having to implement many ways to create new instances."},
    {"get_scaling",
        (PyCFunction)sfloat_get_scaling, METH_NOARGS, NULL},
    {"__reduce__",
       (PyCFunction)sfloat___reduce__, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

/* 创建新的浮点数类型对象 */
static PyObject *sfloat_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    double scaling = 1.;
    static char *kwargs_strs[] = {"scaling", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|d:_ScaledFloatTestDType", kwargs_strs, &scaling)) {
        return NULL;
    }
    if (scaling == 1.) {
        Py_INCREF(&SFloatSingleton);
        return (PyObject *)&SFloatSingleton;
    }
    return (PyObject *)sfloat_scaled_copy(&SFloatSingleton, scaling);
}

/* 返回浮点数类型对象的字符串表示 */
static PyObject *sfloat_repr(PyArray_SFloatDescr *self)
{
    PyObject *scaling = PyFloat_FromDouble(self->scaling);
    /* 返回浮点数类型的字符串表示，包含 scaling 属性 */

    return PyUnicode_FromFormat("<SFloatDescr with scaling %f>", self->scaling);
}
    # 如果 scaling 是空指针，则返回空指针
    if (scaling == NULL) {
        return NULL;
    }
    # 使用 PyUnicode_FromFormat 函数创建一个新的 Python Unicode 对象，
    # 格式化字符串为 "_ScaledFloatTestDType(scaling=%R)"，其中 %R 是格式化参数，
    # 表示将 scaling 对象转换为相应的 Python 表示形式
    PyObject *res = PyUnicode_FromFormat(
            "_ScaledFloatTestDType(scaling=%R)", scaling);
    # 减少 scaling 对象的引用计数，因为在创建 res 后不再需要 scaling
    Py_DECREF(scaling);
    # 返回创建的 Unicode 对象 res
    return res;
}
    if (((PyArray_SFloatDescr *)loop_descrs[0])->scaling
            == ((PyArray_SFloatDescr *)loop_descrs[1])->scaling) {
        /* 检查两个描述符的 scaling 属性是否相同 */
        *view_offset = 0;
        // 如果相同，则视为只是一个视图，设置视图偏移为0
        return NPY_NO_CASTING;
    }
    else if (-((PyArray_SFloatDescr *)loop_descrs[0])->scaling
             == ((PyArray_SFloatDescr *)loop_descrs[1])->scaling) {
        /* 检查两个描述符的 scaling 属性是否互为相反数 */
        // 如果是相反数，则改变符号不会丢失精度
        return NPY_EQUIV_CASTING;
    }
    // 如果两者不是相同的 scaling 或者互为相反数
    /* 技术上讲，这不是一个安全的类型转换，因为可能会发生溢出或下溢 */
    // 返回同类别的类型转换，但不保证安全性
    return NPY_SAME_KIND_CASTING;
/*
 * Casting to and from doubles.
 *
 * To keep things interesting, we ONLY define the trivial cast with a factor
 * of 1.  All other casts have to be handled by the sfloat to sfloat cast.
 *
 * The casting machinery should optimize this step away normally, since we
 * flag the this is a view.
 */
static int
cast_float_to_from_sfloat(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取数据的长度
    npy_intp N = dimensions[0];
    // 输入数据指针
    char *in = data[0];
    // 输出数据指针
    char *out = data[1];
    // 遍历数据并进行类型转换
    for (npy_intp i = 0; i < N; i++) {
        // 将输入的 double 类型数据复制给输出
        *(double *)out = *(double *)in;
        // 更新输入指针的位置
        in += strides[0];
        // 更新输出指针的位置
        out += strides[1];
    }
    // 返回成功状态
    return 0;
}


/*
 * Resolve descriptors for casting from float to sfloat and vice versa.
 */
static NPY_CASTING
float_to_from_sfloat_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *NPY_UNUSED(given_descrs[2]),
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    // 为输入和输出的数据类型获取默认描述符
    loop_descrs[0] = NPY_DT_CALL_default_descr(dtypes[0]);
    if (loop_descrs[0] == NULL) {
        // 如果获取失败，返回错误状态
        return -1;
    }
    loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
    if (loop_descrs[1] == NULL) {
        // 如果获取失败，返回错误状态
        return -1;
    }
    // 设置视图偏移量为0
    *view_offset = 0;
    // 返回无需类型转换的状态
    return NPY_NO_CASTING;
}


/*
 * Cast from sfloat to boolean (for testing the logical functions).
 */
static int
cast_sfloat_to_bool(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取数据的长度
    npy_intp N = dimensions[0];
    // 输入数据指针
    char *in = data[0];
    // 输出数据指针
    char *out = data[1];
    // 遍历数据并进行类型转换
    for (npy_intp i = 0; i < N; i++) {
        // 将输入的 double 类型数据转换为布尔类型后赋给输出
        *(npy_bool *)out = *(double *)in != 0;
        // 更新输入指针的位置
        in += strides[0];
        // 更新输出指针的位置
        out += strides[1];
    }
    // 返回成功状态
    return 0;
}

/*
 * Resolve descriptors for casting from sfloat to boolean.
 */
static NPY_CASTING
sfloat_to_bool_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 增加对给定描述符的引用
    Py_INCREF(given_descrs[0]);
    // 设置输入描述符为给定的描述符
    loop_descrs[0] = given_descrs[0];
    if (loop_descrs[0] == NULL) {
        // 如果获取失败，返回错误状态
        return -1;
    }
    // 设置输出描述符为布尔类型的描述符（不可能失败）
    loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);  /* cannot fail */
    // 返回不安全类型转换状态
    return NPY_UNSAFE_CASTING;
}


/*
 * Initialize casting functions for sfloat.
 */
static int
sfloat_init_casts(void)
{
    // 定义输入和输出的数据类型为单精度浮点型
    PyArray_DTypeMeta *dtypes[2] = {&PyArray_SFloatDType, &PyArray_SFloatDType};
    // 初始化类型槽
    PyType_Slot slots[4] = {{0, NULL}};
    // 定义方法的规范
    PyArrayMethod_Spec spec = {
        .name = "sfloat_to_sfloat_cast",
        .nin = 1,
        .nout = 1,
        /* minimal guaranteed casting */
        .casting = NPY_SAME_KIND_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = dtypes,
        .slots = slots,
    };

    // 设置第一个槽位为解析描述符的方法
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &sfloat_to_sfloat_resolve_descriptors;

    // 设置第二个槽位为处理步进循环的方法
    slots[1].slot = NPY_METH_strided_loop;
    ```
    # 将第一个槽位的函数指针设置为 cast_sfloat_to_sfloat_aligned 函数的地址
    slots[1].pfunc = &cast_sfloat_to_sfloat_aligned;

    # 设置第二个槽位的标志为 NPY_METH_unaligned_strided_loop
    slots[2].slot = NPY_METH_unaligned_strided_loop;
    # 将第二个槽位的函数指针设置为 cast_sfloat_to_sfloat_unaligned 函数的地址
    slots[2].pfunc = &cast_sfloat_to_sfloat_unaligned;

    # 向 PyArray_AddCastingImplementation_FromSpec 函数添加类型转换规范
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    # 设置 spec 的名称为 "float_to_sfloat_cast"
    spec.name = "float_to_sfloat_cast";
    # 设置 spec 的标志为 NPY_METH_NO_FLOATINGPOINT_ERRORS
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    # 获取 PyArray_DoubleDType 的指针，并将其赋值给 dtypes 数组的第一个元素
    PyArray_DTypeMeta *double_DType = &PyArray_DoubleDType;
    dtypes[0] = double_DType;

    # 设置 slots 数组的各个槽位的值和函数指针，用于处理浮点到半浮点的转换
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &float_to_from_sfloat_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_float_to_from_sfloat;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    # 再次向 PyArray_AddCastingImplementation_FromSpec 函数添加类型转换规范
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    # 设置 spec 的名称为 "sfloat_to_float_cast"
    spec.name = "sfloat_to_float_cast";
    # 设置 dtypes 数组的第一个元素为 PyArray_SFloatDType 的指针
    dtypes[0] = &PyArray_SFloatDType;
    # 设置 dtypes 数组的第二个元素为 double_DType 指针
    dtypes[1] = double_DType;

    # 再次向 PyArray_AddCastingImplementation_FromSpec 函数添加类型转换规范
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    # 设置 slots 数组的各个槽位的值和函数指针，用于处理半浮点到布尔型的转换
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &sfloat_to_bool_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_sfloat_to_bool;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    # 设置 spec 的名称为 "sfloat_to_bool_cast"
    spec.name = "sfloat_to_bool_cast";
    # 设置 dtypes 数组的第一个元素为 PyArray_SFloatDType 的指针
    dtypes[0] = &PyArray_SFloatDType;
    # 设置 dtypes 数组的第二个元素为 PyArray_BoolDType 的指针
    dtypes[1] = &PyArray_BoolDType;

    # 再次向 PyArray_AddCastingImplementation_FromSpec 函数添加类型转换规范
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    # 返回 0 表示函数执行成功
    return 0;
/*
 * We also wish to test very simple ufunc functionality.  So create two
 * ufunc loops:
 * 1. Multiplication, which can multiply the factors and work with that.
 * 2. Addition, which needs to use the common instance, and runs into
 *    cast safety subtleties since we will implement it without an additional
 *    cast.
 */
static int
multiply_sfloats(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取第一个维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数组的指针
    char *in1 = data[0];
    char *in2 = data[1];
    // 获取输出数组的指针
    char *out = data[2];
    // 遍历数组进行乘法运算
    for (npy_intp i = 0; i < N; i++) {
        // 将输入数组中的双精度浮点数相乘，结果存入输出数组中
        *(double *)out = *(double *)in1 * *(double *)in2;
        // 更新输入数组和输出数组的指针位置
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


static NPY_CASTING
multiply_sfloats_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * Multiply the scaling for the result.  If the result was passed in we
     * simply ignore it and let the casting machinery fix it up here.
     */
    // 获取第一个输入描述符的缩放因子
    double factor = ((PyArray_SFloatDescr *)given_descrs[1])->scaling;
    // 创建新的输出描述符，使用第一个输入描述符和缩放因子
    loop_descrs[2] = sfloat_scaled_copy(
            (PyArray_SFloatDescr *)given_descrs[0], factor);
    // 检查是否成功创建新的输出描述符
    if (loop_descrs[2] == 0) {
        return -1;
    }
    // 增加输入描述符的引用计数
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    // 增加第二个输入描述符的引用计数
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    // 指定无需类型转换
    return NPY_NO_CASTING;
}


/*
 * Unlike the multiplication implementation above, this loops deals with
 * scaling (casting) internally.  This allows to test some different paths.
 */
static int
add_sfloats(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取第一个输入描述符的缩放因子
    double fin1 = ((PyArray_SFloatDescr *)context->descriptors[0])->scaling;
    // 获取第二个输入描述符的缩放因子
    double fin2 = ((PyArray_SFloatDescr *)context->descriptors[1])->scaling;
    // 获取输出描述符的缩放因子
    double fout = ((PyArray_SFloatDescr *)context->descriptors[2])->scaling;

    // 计算输入输出之间的缩放因子
    double fact1 = fin1 / fout;
    double fact2 = fin2 / fout;
    // 检查第一个缩放因子是否有效
    if (check_factor(fact1) < 0) {
        return -1;
    }
    // 检查第二个缩放因子是否有效
    if (check_factor(fact2) < 0) {
        return -1;
    }

    // 获取第一个维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数组的指针
    char *in1 = data[0];
    char *in2 = data[1];
    // 获取输出数组的指针
    char *out = data[2];
    // 遍历数组进行加法运算
    for (npy_intp i = 0; i < N; i++) {
        // 将输入数组中的双精度浮点数按比例相加，结果存入输出数组中
        *(double *)out = (*(double *)in1 * fact1) + (*(double *)in2 * fact2);
        // 更新输入数组和输出数组的指针位置
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


static NPY_CASTING
/*
 * 根据给定的描述符解析添加单精度浮点数，并解决描述符。
 * 如果没有给定输出描述符，则使用常见实例。
 */
add_sfloats_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * 如果没有给定输出描述符，则使用常见实例:
     */
    if (given_descrs[2] == NULL) {
        // 使用给定的输入描述符创建一个共享的单精度浮点数描述符
        loop_descrs[2] = sfloat_common_instance(
                given_descrs[0], given_descrs[1]);
        if (loop_descrs[2] == 0) {
            return -1; // 失败返回
        }
    }
    else {
        // 增加引用计数以使用给定的输出描述符
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }
    // 增加引用计数以使用给定的输入描述符
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    /* 如果比例因子不匹配，我们在ufunc内部进行隐式转换！ */
    double fin1 = ((PyArray_SFloatDescr *)loop_descrs[0])->scaling;
    double fin2 = ((PyArray_SFloatDescr *)loop_descrs[1])->scaling;
    double fout = ((PyArray_SFloatDescr *)loop_descrs[2])->scaling;

    // 如果输入和输出的比例因子完全匹配，则不需要转换
    if (fin1 == fout && fin2 == fout) {
        return NPY_NO_CASTING;
    }
    // 如果输入的绝对值和输出的绝对值相匹配，则可以进行等价转换
    if (npy_fabs(fin1) == npy_fabs(fout) && npy_fabs(fin2) == npy_fabs(fout)) {
        return NPY_EQUIV_CASTING;
    }
    // 否则，进行相同类型的转换
    return NPY_SAME_KIND_CASTING;
}


/*
 * 使用 "PyUFunc_AddWrappingLoop" API 定义 hypot 循环。
 * 目前仅用于映射到双精度 hypot 循环。
 */
static int
translate_given_descrs_to_double(
        int nin, int nout, PyArray_DTypeMeta *const wrapped_dtypes[],
        PyArray_Descr *const given_descrs[], PyArray_Descr *new_descrs[])
{
    assert(nin == 2 && nout == 1);
    for (int i = 0; i < 3; i++) {
        if (given_descrs[i] == NULL) {
            new_descrs[i] = NULL;
        }
        else {
            // 将给定的描述符转换为双精度描述符
            new_descrs[i] = PyArray_DescrFromType(NPY_DOUBLE);
        }
    }
    return 0;
}


/*
 * 将给定的描述符转换为循环描述符。
 */
static int
translate_loop_descrs(
        int nin, int nout, PyArray_DTypeMeta *const new_dtypes[],
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *NPY_UNUSED(original_descrs[]),
        PyArray_Descr *loop_descrs[])
{
    assert(nin == 2 && nout == 1);
    // 使用给定的输入描述符创建共享的单精度浮点数描述符
    loop_descrs[0] = sfloat_common_instance(
            given_descrs[0], given_descrs[1]);
    if (loop_descrs[0] == 0) {
        return -1; // 失败返回
    }
    // 增加引用计数以使用共享的描述符
    Py_INCREF(loop_descrs[0]);
    loop_descrs[1] = loop_descrs[0];
    Py_INCREF(loop_descrs[0]);
    loop_descrs[2] = loop_descrs[0];
    return 0;
}


/*
 * 根据给定的 ufunc 名称获取 ufunc 对象。
 */
static PyObject *
sfloat_get_ufunc(const char *ufunc_name)
{
    // 导入 numpy 模块
    PyObject *mod = PyImport_ImportModule("numpy");
    if (mod == NULL) {
        return NULL; // 导入失败
    }
    // 获取指定名称的 ufunc 对象
    PyObject *ufunc = PyObject_GetAttrString(mod, ufunc_name);
    Py_DECREF(mod);
    // 检查获取的对象是否为 ufunc 类型
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        Py_DECREF(ufunc);
        // 如果不是 ufunc 类型，则抛出类型错误异常
        PyErr_Format(PyExc_TypeError,
                "numpy.%s was not a ufunc!", ufunc_name);
        return NULL;
    }
    // 返回获取到的 ufunc 对象
    return ufunc;
}
    # 返回 ufunc 变量的值
    return ufunc;
/*
 * Add a new loop to the specified ufunc for floating-point addition.
 * This function retrieves the ufunc object by name and prepares the necessary
 * information to add a new loop for the given data types.
 */
static int
sfloat_add_loop(const char *ufunc_name,
        PyArray_DTypeMeta *dtypes[3], PyObject *meth_or_promoter)
{
    // 获取指定名称的ufunc对象
    PyObject *ufunc = sfloat_get_ufunc(ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }
    
    // 创建一个包含三个数据类型的元组
    PyObject *dtype_tup = PyArray_TupleFromItems(3, (PyObject **)dtypes, 1);
    if (dtype_tup == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }
    
    // 打包类型元组和方法/推广器成为一个元组info
    PyObject *info = PyTuple_Pack(2, dtype_tup, meth_or_promoter);
    Py_DECREF(dtype_tup);
    if (info == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }
    
    // 添加循环到ufunc对象中
    int res = PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
    Py_DECREF(ufunc);
    Py_DECREF(info);
    return res;
}


/*
 * Add a wrapping loop to the specified ufunc for floating-point addition.
 * This function retrieves the ufunc object by name and adds a wrapping loop
 * using the provided data types and their corresponding wrapped versions.
 */
static int
sfloat_add_wrapping_loop(const char *ufunc_name, PyArray_DTypeMeta *dtypes[3])
{
    // 获取指定名称的ufunc对象
    PyObject *ufunc = sfloat_get_ufunc(ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }
    
    // 准备包裹的数据类型为双精度浮点类型
    PyArray_DTypeMeta *double_dt = &PyArray_DoubleDType;
    PyArray_DTypeMeta *wrapped_dtypes[3] = {double_dt, double_dt, double_dt};
    
    // 添加包裹循环到ufunc对象中
    int res = PyUFunc_AddWrappingLoop(
        ufunc, dtypes, wrapped_dtypes, &translate_given_descrs_to_double,
        &translate_loop_descrs);
    Py_DECREF(ufunc);

    return res;
}


/*
 * Add promoters to convert normal floats to scaled floats.
 * This function is used to promote data types for multiplication operations.
 */
static int
promote_to_sfloat(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_DTypeMeta *const signature[3],
        PyArray_DTypeMeta *new_dtypes[3])
{
    // 遍历每个数据类型，将其推广为单精度浮点类型
    for (int i = 0; i < 3; i++) {
        PyArray_DTypeMeta *new = &PyArray_SFloatDType;
        if (signature[i] != NULL) {
            new = signature[i];
        }
        Py_INCREF(new);
        new_dtypes[i] = new;
    }
    return 0;
}


/*
 * Initialize custom ufuncs for single-precision floating-point operations.
 * This function initializes and registers ufuncs for multiplication and addition
 * of single-precision floating-point numbers.
 */
static int
sfloat_init_ufuncs(void) {
    // 定义三个单精度浮点类型的数据类型元组
    PyArray_DTypeMeta *dtypes[3] = {
            &PyArray_SFloatDType, &PyArray_SFloatDType, &PyArray_SFloatDType};
    
    // 定义三个PyType_Slot槽
    PyType_Slot slots[3] = {{0, NULL}};
    
    // 定义一个PyArrayMethod_Spec结构体用于描述方法
    PyArrayMethod_Spec spec = {
        .nin = 2,
        .nout =1,
        .dtypes = dtypes,
        .slots = slots,
    };
    
    // 添加乘法方法的解析描述符和循环函数
    spec.name = "sfloat_multiply";
    spec.casting = NPY_NO_CASTING;
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &multiply_sfloats_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &multiply_sfloats;
    
    // 创建一个PyBoundArrayMethodObject对象
    PyBoundArrayMethodObject *bmeth = PyArrayMethod_FromSpec_int(&spec, 0);
    if (bmeth == NULL) {
        return -1;
    }
    
    // 添加乘法方法的循环到ufunc对象中
    int res = sfloat_add_loop("multiply",
            bmeth->dtypes, (PyObject *)bmeth->method);
    Py_DECREF(bmeth);
    if (res < 0) {
        return -1;
    }
    
    // 添加加法方法的解析描述符和循环函数
    spec.name = "sfloat_add";
    spec.casting = NPY_SAME_KIND_CASTING;
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &add_sfloats_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &add_sfloats;
    
    // 创建一个PyBoundArrayMethodObject对象
    bmeth = PyArrayMethod_FromSpec_int(&spec, 0);
    if (bmeth == NULL) {
        return -1;
    }
    
    // 添加加法方法的循环到ufunc对象中
    res = sfloat_add_loop("add",
            bmeth->dtypes, (PyObject *)bmeth->method);
    Py_DECREF(bmeth);
    if (res < 0) {
        return -1;
    }
    
    return 0;
}
    # 使用给定的规格创建一个整数类型的数组方法对象
    bmeth = PyArrayMethod_FromSpec_int(&spec, 0);
    if (bmeth == NULL) {
        return -1;
    }
    # 调用特定方法名称为"add"的浮点数加法循环，并传入方法对象的数据类型和方法对象本身作为参数
    res = sfloat_add_loop("add",
            bmeth->dtypes, (PyObject *)bmeth->method);
    # 减少方法对象的引用计数
    Py_DECREF(bmeth);
    if (res < 0) {
        return -1;
    }

    # 注意：如果类型的缩放因子可以为负数，则包装实际上不正确
    if (sfloat_add_wrapping_loop("hypot", dtypes) < 0) {
        return -1;
    }

    """
     * 为双精度乘法的两个方向添加提升器。
     """
    # 创建一个指向双精度数据类型的指针
    PyArray_DTypeMeta *double_DType = &PyArray_DoubleDType;

    # 创建一个包含提升到单精度浮点数、双精度和空指针的数据类型元数据数组
    PyArray_DTypeMeta *promoter_dtypes[3] = {
            &PyArray_SFloatDType, double_DType, NULL};

    # 创建一个新的 Python Capsule 对象，用于封装提升到单精度浮点数的函数
    PyObject *promoter = PyCapsule_New(
            &promote_to_sfloat, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        return -1;
    }
    # 调用特定方法名称为"multiply"的浮点数乘法循环，并传入提升器的数据类型数组和提升器本身作为参数
    res = sfloat_add_loop("multiply", promoter_dtypes, promoter);
    if (res < 0) {
        Py_DECREF(promoter);
        return -1;
    }
    # 调整提升器的数据类型数组，反向传入参数，并再次调用浮点数乘法循环
    promoter_dtypes[0] = double_DType;
    promoter_dtypes[1] = &PyArray_SFloatDType;
    res = sfloat_add_loop("multiply", promoter_dtypes, promoter);
    # 减少提升器对象的引用计数
    Py_DECREF(promoter);
    if (res < 0) {
        return -1;
    }

    # 返回操作成功的标志
    return 0;
"""
/*
 * Python entry point, exported via `umathmodule.h` and `multiarraymodule.c`.
 * TODO: Should be moved when the necessary API is not internal anymore.
 */
"""

# 检查全局变量以确保未初始化时获取单精度浮点数据类型
NPY_NO_EXPORT PyObject *
get_sfloat_dtype(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(args))
{
    if (npy_thread_unsafe_state.get_sfloat_dtype_initialized) {
        # 如果已经初始化，增加对单精度浮点数据类型的引用计数并返回它
        Py_INCREF(&PyArray_SFloatDType);
        return (PyObject *)&PyArray_SFloatDType;
    }

    # 将单精度浮点数据类型的基类设置为描述符类型
    PyArray_SFloatDType.super.ht_type.tp_base = &PyArrayDescr_Type;

    # 准备单精度浮点数据类型对象，如果失败则返回空指针
    if (PyType_Ready((PyTypeObject *)&PyArray_SFloatDType) < 0) {
        return NULL;
    }

    # 创建一个新的字典来存储类型的转换实现
    NPY_DT_SLOTS(&PyArray_SFloatDType)->castingimpls = PyDict_New();
    if (NPY_DT_SLOTS(&PyArray_SFloatDType)->castingimpls == NULL) {
        return NULL;
    }

    # 使用单精度浮点数据类型初始化一个新的 Python 对象
    PyObject *o = PyObject_Init(
            (PyObject *)&SFloatSingleton, (PyTypeObject *)&PyArray_SFloatDType);
    if (o == NULL) {
        return NULL;
    }

    # 初始化单精度浮点类型的类型转换
    if (sfloat_init_casts() < 0) {
        return NULL;
    }

    # 初始化单精度浮点类型的通用函数
    if (sfloat_init_ufuncs() < 0) {
        return NULL;
    }

    # 将状态标记为已初始化
    npy_thread_unsafe_state.get_sfloat_dtype_initialized = NPY_TRUE;
    # 返回单精度浮点数据类型对象
    return (PyObject *)&PyArray_SFloatDType;
}
```