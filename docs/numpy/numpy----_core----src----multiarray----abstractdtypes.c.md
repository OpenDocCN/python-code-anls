# `.\numpy\numpy\_core\src\multiarray\abstractdtypes.c`

```
/*
 * 定义宏以避免使用已弃用的 NumPy API 版本，并指定多维数组模块。
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 引入必要的头文件和库文件，包括 Python 标准库、结构成员定义、NumPy 的相关头文件、
 * 以及自定义的数据类型元信息、抽象数据类型、数组强制转换和通用函数。
 */
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "dtypemeta.h"
#include "abstractdtypes.h"
#include "array_coercion.h"
#include "common.h"


/*
 * 定义静态内联函数 int_default_descriptor，用于返回整数类型的数组描述符。
 */
static inline PyArray_Descr *
int_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_INTP);
}

/*
 * 根据 Python 的 long 对象发现合适的数据类型描述符。
 * 如果长整型不足以表示，会依次检查长长整型、无符号长整型，最后返回对象类型。
 */
static PyArray_Descr *
discover_descriptor_from_pylong(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyLong_Check(obj));
    
    long long value = PyLong_AsLongLong(obj);
    if (error_converting(value)) {
        PyErr_Clear();
    }
    else {
        if (NPY_MIN_INTP <= value && value <= NPY_MAX_INTP) {
            return PyArray_DescrFromType(NPY_INTP);
        }
        return PyArray_DescrFromType(NPY_LONGLONG);
    }

    unsigned long long uvalue = PyLong_AsUnsignedLongLong(obj);
    if (uvalue == (unsigned long long)-1 && PyErr_Occurred()){
        PyErr_Clear();
    }
    else {
        return PyArray_DescrFromType(NPY_ULONGLONG);
    }

    return PyArray_DescrFromType(NPY_OBJECT);
}

/*
 * 定义静态内联函数 float_default_descriptor，用于返回双精度浮点数类型的数组描述符。
 */
static inline PyArray_Descr *
float_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_DOUBLE);
}

/*
 * 根据 Python 的 float 对象发现合适的数据类型描述符，总是返回双精度浮点数类型。
 */
static PyArray_Descr*
discover_descriptor_from_pyfloat(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyFloat_CheckExact(obj));
    return PyArray_DescrFromType(NPY_DOUBLE);
}

/*
 * 定义静态内联函数 complex_default_descriptor，用于返回双精度复数类型的数组描述符。
 */
static inline PyArray_Descr *
complex_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_CDOUBLE);
}

/*
 * 根据 Python 的 complex 对象发现合适的数据类型描述符，总是返回复数128位类型。
 */
static PyArray_Descr*
discover_descriptor_from_pycomplex(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyComplex_CheckExact(obj));
    return PyArray_DescrFromType(NPY_COMPLEX128);
}

/*
 * 初始化并映射 Python 类型到 NumPy 的数据类型。
 * 如果初始化任何一个抽象数据类型失败，则返回 -1。
 * 延迟赋值以避免在 Windows 编译器中出现 "error C2099: initializer is not a constant"。
 */
NPY_NO_EXPORT int
initialize_and_map_pytypes_to_dtypes()
{
    if (PyType_Ready((PyTypeObject *)&PyArray_IntAbstractDType) < 0) {
        return -1;
    }
    if (PyType_Ready((PyTypeObject *)&PyArray_FloatAbstractDType) < 0) {
        return -1;
    }
    if (PyType_Ready((PyTypeObject *)&PyArray_ComplexAbstractDType) < 0) {
        return -1;
    }

    /*
     * 延迟赋值以避免在 Windows 编译器中出现 "error C2099: initializer is not a constant"。
     * 期望将来能够在结构体中完成这些赋值。
     */
    ((PyTypeObject *)&PyArray_PyLongDType)->tp_base =
        (PyTypeObject *)&PyArray_IntAbstractDType;
    PyArray_PyLongDType.scalar_type = &PyLong_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyLongDType) < 0) {
        return -1;
    }

    ((PyTypeObject *)&PyArray_PyFloatDType)->tp_base =
        (PyTypeObject *)&PyArray_FloatAbstractDType;
    PyArray_PyFloatDType.scalar_type = &PyFloat_Type;
    
    /* 省略部分代码，根据上下文可能还有其他内容 */

    return 0;
}
    // 如果 PyArray_PyFloatDType 类型未准备好，返回错误码 -1
    if (PyType_Ready((PyTypeObject *)&PyArray_PyFloatDType) < 0) {
        return -1;
    }
    // 将 PyArray_PyComplexDType 的基类设置为 PyArray_ComplexAbstractDType
    ((PyTypeObject *)&PyArray_PyComplexDType)->tp_base =
        (PyTypeObject *)&PyArray_ComplexAbstractDType;
    // 将 PyArray_PyComplexDType 的标量类型设置为 PyComplex_Type
    PyArray_PyComplexDType.scalar_type = &PyComplex_Type;
    // 如果 PyArray_PyComplexDType 类型未准备好，返回错误码 -1
    if (PyType_Ready((PyTypeObject *)&PyArray_PyComplexDType) < 0) {
        return -1;
    }

    /* 注册新的数据类型以便发现 */
    // 将 PyLong_Type 映射为 PyArray_PyLongDType 数据类型
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyLongDType, &PyLong_Type, NPY_FALSE) < 0) {
        return -1;
    }
    // 将 PyFloat_Type 映射为 PyArray_PyFloatDType 数据类型
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyFloatDType, &PyFloat_Type, NPY_FALSE) < 0) {
        return -1;
    }
    // 将 PyComplex_Type 映射为 PyArray_PyComplexDType 数据类型
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyComplexDType, &PyComplex_Type, NPY_FALSE) < 0) {
        return -1;
    }

    /*
     * 将 str、bytes 和 bool 映射为 NumPy 的数据类型。
     * 这里使用 `is_known_scalar_type` 函数来完成。
     * 注意：`is_known_scalar_type` 函数目前被视为初步版本，
     *       可能会通过额外的抽象数据类型来实现相同的功能。
     */
    // 将 PyUnicode_Type 映射为 Unicode 字符串类型
    PyArray_DTypeMeta *dtype;
    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_UNICODE));
    if (_PyArray_MapPyTypeToDType(dtype, &PyUnicode_Type, NPY_FALSE) < 0) {
        return -1;
    }
    // 将 PyBytes_Type 映射为字节串类型
    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_STRING));
    if (_PyArray_MapPyTypeToDType(dtype, &PyBytes_Type, NPY_FALSE) < 0) {
        return -1;
    }
    // 将 PyBool_Type 映射为布尔类型
    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_BOOL));
    if (_PyArray_MapPyTypeToDType(dtype, &PyBool_Type, NPY_FALSE) < 0) {
        return -1;
    }

    // 成功执行，返回成功码 0
    return 0;
/*
 * The following functions define the "common DType" for the abstract dtypes.
 *
 * Note that the logic with respect to the "higher" dtypes such as floats
 * could likely be more logically defined for them, but since NumPy dtypes
 * largely "know" each other, that is not necessary.
 */

static PyArray_DTypeMeta *
int_common_dtype(PyArray_DTypeMeta *NPY_UNUSED(cls), PyArray_DTypeMeta *other)
{
    // 如果传入的数据类型是遗留类型且类型编号小于遗留类型数量
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        // 如果传入的类型是布尔类型
        if (other->type_num == NPY_BOOL) {
            /* Use the default integer for bools: */
            // 返回默认整数类型(PyArray_IntpDType)的新引用
            return NPY_DT_NewRef(&PyArray_IntpDType);
        }
    }
    // 如果传入的数据类型是遗留类型
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        // 创建一个指向PyArray_UInt8DType的指针
        PyArray_DTypeMeta *uint8_dt = &PyArray_UInt8DType;
        // 调用传入类型和uint8_dt的共同数据类型函数
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, uint8_dt);
        // 释放uint8_dt的引用
        Py_DECREF(uint8_dt);
        // 如果返回为空
        if (res == NULL) {
            // 清除错误
            PyErr_Clear();
        }
        // 如果返回为Py_NotImplemented
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            // 释放res的引用
            Py_DECREF(res);
        }
        // 否则
        else {
            // 返回res
            return res;
        }
        /* Try again with `int8`, an error may have been set, though */
        // 创建一个指向PyArray_Int8DType的指针
        PyArray_DTypeMeta *int8_dt = &PyArray_Int8DType;
        // 调用传入类型和int8_dt的共同数据类型函数
        res = NPY_DT_CALL_common_dtype(other, int8_dt);
        // 如果返回为空
        if (res == NULL) {
            // 清除错误
            PyErr_Clear();
        }
        // 如果返回为Py_NotImplemented
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            // 释放res的引用
            Py_DECREF(res);
        }
        // 否则
        else {
            // 返回res
            return res;
        }
        /* And finally, we will try the default integer, just for sports... */
        // 创建一个指向PyArray_IntpDType的指针
        PyArray_DTypeMeta *default_int = &PyArray_IntpDType;
        // 调用传入类型和default_int的共同数据类型函数
        res = NPY_DT_CALL_common_dtype(other, default_int);
        // 如果返回为空
        if (res == NULL) {
            // 清除错误
            PyErr_Clear();
        }
        // 返回res
        return res;
    }
    // 增加Py_NotImplemented的引用计数
    Py_INCREF(Py_NotImplemented);
    // 返回Py_NotImplemented
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
float_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // 如果传入的数据类型是遗留类型且类型编号小于遗留类型数量
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        // 如果传入的类型是布尔类型或者整数类型
        if (other->type_num == NPY_BOOL || PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            // 返回默认双精度浮点类型(PyArray_DoubleDType)的新引用
            return NPY_DT_NewRef(&PyArray_DoubleDType);
        }
    }
    // 如果传入的类型是PyArray_PyLongDType
    else if (other == &PyArray_PyLongDType) {
        // 增加cls的引用计数并返回cls
        Py_INCREF(cls);
        return cls;
    }
    // 返回Py_NotImplemented
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}
    else if (NPY_DT_is_legacy(other)) {
        /* 如果 'other' 是遗留数据类型，执行以下操作 */
        /* 这是向后兼容的回退，通常会执行正确的操作... */
        
        // 指向半精度数据类型的指针
        PyArray_DTypeMeta *half_dt = &PyArray_HalfDType;
        
        // 查找 'other' 和半精度数据类型的公共数据类型
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, half_dt);
        
        // 如果未找到公共数据类型，清除错误状态
        if (res == NULL) {
            PyErr_Clear();
        }
        // 如果返回结果是 Py_NotImplemented，则释放 'res'
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        // 否则返回找到的公共数据类型
        else {
            return res;
        }
        
        // 使用双精度（默认的浮点数）再次尝试
        // 指向双精度数据类型的指针
        PyArray_DTypeMeta *double_dt = &PyArray_DoubleDType;
        
        // 查找 'other' 和双精度数据类型的公共数据类型
        res = NPY_DT_CALL_common_dtype(other, double_dt);
        
        // 返回找到的公共数据类型
        return res;
    }
    
    // 增加对 Py_NotImplemented 的引用计数
    Py_INCREF(Py_NotImplemented);
    
    // 返回 Py_NotImplemented，表示未找到合适的数据类型
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}



/*
 * 返回两个数据类型的共同复合数据类型。
 * 如果 `other` 是遗留类型并且类型编号小于 `NPY_NTYPES_LEGACY`，
 * 对于布尔和整数类型，使用默认的复合数据类型 PyArray_CDoubleDType。
 * 否则，如果 `other` 是遗留类型，则使用通常情况下正确的复合数据类型（fallback）。
 * 如果使用 PyArray_CFloatDType 没有得到结果，清除异常，并尝试使用 PyArray_CDoubleDType。
 * 如果 `other` 是 PyArray_PyLongDType 或 PyArray_PyFloatDType，返回 `cls`。
 * 否则，返回 Py_NotImplemented 表示未实现。
 */
static PyArray_DTypeMeta *
complex_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        if (other->type_num == NPY_BOOL ||
                PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return NPY_DT_NewRef(&PyArray_CDoubleDType);
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        PyArray_DTypeMeta *cfloat_dt = &PyArray_CFloatDType;
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, cfloat_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* Retry with cdouble (the default complex) */
        PyArray_DTypeMeta *cdouble_dt = &PyArray_CDoubleDType;
        res = NPY_DT_CALL_common_dtype(other, cdouble_dt);
        return res;
    }
    else if (other == &PyArray_PyLongDType ||
             other == &PyArray_PyFloatDType) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}



/*
 * 在 arraytypes.c.src 中定义所有常规数据类型可以继承的抽象数值数据类型。
 * 在此还定义与 Python 标量对应的数据类型。
 */
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_IntAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._IntegerAbstractDType",
        .tp_base = &PyArrayDescr_Type,
        .tp_basicsize = sizeof(PyArray_Descr),
        .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .flags = NPY_DT_ABSTRACT,
};

/*
 * 定义与 PyLong 对象对应的数据类型结构 PyArray_PyLongDType。
 * 其中，discover_descr_from_pyobject 使用 discover_descriptor_from_pylong 函数，
 * default_descr 使用 int_default_descriptor 函数，
 * common_dtype 使用 int_common_dtype 函数。
 */
NPY_DType_Slots pylongdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pylong,
    .default_descr = int_default_descriptor,
    .common_dtype = int_common_dtype,
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyLongDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._PyLongDType",
        .tp_base = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
        .tp_basicsize = sizeof(PyArray_Descr),
        .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .dt_slots = &pylongdtype_slots,
    .scalar_type = NULL,  /* set in initialize_and_map_pytypes_to_dtypes */
};

/*
 * 定义所有常规数据类型可以继承的抽象浮点数数据类型 PyArray_FloatAbstractDType。
 */
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_FloatAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._FloatAbstractDType",
        .tp_base = &PyArrayDescr_Type,
        .tp_basicsize = sizeof(PyArray_Descr),
       .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,
    .flags = NPY_DT_ABSTRACT,
};

/*
 * 定义与 PyFloat 对象对应的数据类型结构 PyArray_PyFloatDType。
 * 其中，discover_descr_from_pyobject 使用 discover_descriptor_from_pyfloat 函数。
 */
NPY_DType_Slots pyfloatdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pyfloat,
    .default_descr = float_default_descriptor,
    # 设置默认描述器为浮点数的默认描述器
    .common_dtype = float_common_dtype,
    # 设置常见数据类型为浮点数的常见数据类型
# 定义 PyArray_PyFloatDType 类型的全局变量，表示 numpy 中的浮点数数据类型
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyFloatDType = {{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._PyFloatDType",
        .tp_base = NULL,  /* 在 initialize_and_map_pytypes_to_dtypes 中设置 */
        .tp_basicsize = sizeof(PyArray_Descr),
       .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,  # 数据类型编号设为 -1
    .dt_slots = &pyfloatdtype_slots,  # 使用 pyfloatdtype_slots 描述符槽
    .scalar_type = NULL,  /* 在 initialize_and_map_pytypes_to_dtypes 中设置 */
};

# 定义 PyArray_ComplexAbstractDType 类型的全局变量，表示 numpy 中的复数抽象数据类型
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_ComplexAbstractDType = {{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._ComplexAbstractDType",
        .tp_base = &PyArrayDescr_Type,  # 基类设为 PyArrayDescr_Type
        .tp_basicsize = sizeof(PyArray_Descr),
         .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,  # 数据类型编号设为 -1
    .flags = NPY_DT_ABSTRACT,  # 设置抽象标志
};

# 定义 pycomplexdtype_slots 结构体，包含复数类型的描述符槽函数
NPY_DType_Slots pycomplexdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pycomplex,
    .default_descr = complex_default_descriptor,
    .common_dtype = complex_common_dtype,
};

# 定义 PyArray_PyComplexDType 类型的全局变量，表示 numpy 中的复数数据类型
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyComplexDType = {{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_name = "numpy.dtypes._PyComplexDType",
        .tp_base = NULL,  /* 在 initialize_and_map_pytypes_to_dtypes 中设置 */
        .tp_basicsize = sizeof(PyArray_Descr),
         .tp_flags = Py_TPFLAGS_DEFAULT,
    },},
    .type_num = -1,  # 数据类型编号设为 -1
    .dt_slots = &pycomplexdtype_slots,  # 使用 pycomplexdtype_slots 描述符槽
    .scalar_type = NULL,  /* 在 initialize_and_map_pytypes_to_dtypes 中设置 */
};
```