# `.\numpy\numpy\_core\include\numpy\arrayscalars.h`

```py
#ifndef NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_

#ifndef _MULTIARRAYMODULE
// 定义一个布尔类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;
#endif


// 定义一个有符号字符类型的标量对象结构
typedef struct {
        PyObject_HEAD
        signed char obval;
} PyByteScalarObject;


// 定义一个短整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        short obval;
} PyShortScalarObject;


// 定义一个整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        int obval;
} PyIntScalarObject;


// 定义一个长整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        long obval;
} PyLongScalarObject;


// 定义一个长长整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_longlong obval;
} PyLongLongScalarObject;


// 定义一个无符号字符类型的标量对象结构
typedef struct {
        PyObject_HEAD
        unsigned char obval;
} PyUByteScalarObject;


// 定义一个无符号短整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        unsigned short obval;
} PyUShortScalarObject;


// 定义一个无符号整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        unsigned int obval;
} PyUIntScalarObject;


// 定义一个无符号长整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        unsigned long obval;
} PyULongScalarObject;


// 定义一个无符号长长整型类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_ulonglong obval;
} PyULongLongScalarObject;


// 定义一个半精度浮点数类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_half obval;
} PyHalfScalarObject;


// 定义一个单精度浮点数类型的标量对象结构
typedef struct {
        PyObject_HEAD
        float obval;
} PyFloatScalarObject;


// 定义一个双精度浮点数类型的标量对象结构
typedef struct {
        PyObject_HEAD
        double obval;
} PyDoubleScalarObject;


// 定义一个长双精度浮点数类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_longdouble obval;
} PyLongDoubleScalarObject;


// 定义一个复数-单精度浮点数类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_cfloat obval;
} PyCFloatScalarObject;


// 定义一个复数-双精度浮点数类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_cdouble obval;
} PyCDoubleScalarObject;


// 定义一个复数-长双精度浮点数类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_clongdouble obval;
} PyCLongDoubleScalarObject;


// 定义一个通用对象类型的标量对象结构
typedef struct {
        PyObject_HEAD
        PyObject * obval;
} PyObjectScalarObject;

// 定义一个日期时间类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_datetime obval;
        PyArray_DatetimeMetaData obmeta;
} PyDatetimeScalarObject;

// 定义一个时间间隔类型的标量对象结构
typedef struct {
        PyObject_HEAD
        npy_timedelta obval;
        PyArray_DatetimeMetaData obmeta;
} PyTimedeltaScalarObject;


// 定义一个字符类型的标量对象结构
typedef struct {
        PyObject_HEAD
        char obval;
} PyScalarObject;

// 将字符串标量对象定义为字节对象（PyBytesObject）
#define PyStringScalarObject PyBytesObject

#ifndef Py_LIMITED_API
// 定义一个Unicode字符串类型的标量对象结构
typedef struct {
        /* 注意：PyObject_HEAD 宏定义位于此处 */
        PyUnicodeObject base;
        Py_UCS4 *obval;
    #if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
        char *buffer_fmt;
    #endif
} PyUnicodeScalarObject;
#endif


// 定义一个变长字符串类型的标量对象结构
typedef struct {
        PyObject_VAR_HEAD
        char *obval;
#if defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD
        /* 在内部使用子类允许访问名称/字段 */
        _PyArray_LegacyDescr *descr;
#else
        PyArray_Descr *descr;
#endif
        int flags;
        PyObject *base;
    #if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
        void *_buffer_info;  /* 私有缓冲区信息，标记以允许警告 */
    #endif
/* 定义了一些宏和结构体用于处理 NumPy 中的标量对象 */

} PyVoidScalarObject;

/* 宏定义部分 */

/* 定义了返回 NumPy 中布尔标量对象 False 的宏 */
#define PyArrayScalar_False ((PyObject *)(&(_PyArrayScalar_BoolValues[0])))

/* 定义了返回 NumPy 中布尔标量对象 True 的宏 */
#define PyArrayScalar_True ((PyObject *)(&(_PyArrayScalar_BoolValues[1])))

/* 定义了根据长整型值 i 返回对应 NumPy 中标量对象的宏 */
#define PyArrayScalar_FromLong(i) \
        ((PyObject *)(&(_PyArrayScalar_BoolValues[((i)!=0)])))

/* 定义了根据长整型值 i 返回对应 NumPy 中标量对象并增加引用计数的宏 */
#define PyArrayScalar_RETURN_BOOL_FROM_LONG(i)                  \
        return Py_INCREF(PyArrayScalar_FromLong(i)), \
                PyArrayScalar_FromLong(i)

/* 定义了返回 NumPy 中布尔标量对象 False 并增加引用计数的宏 */
#define PyArrayScalar_RETURN_FALSE              \
        return Py_INCREF(PyArrayScalar_False),  \
                PyArrayScalar_False

/* 定义了返回 NumPy 中布尔标量对象 True 并增加引用计数的宏 */
#define PyArrayScalar_RETURN_TRUE               \
        return Py_INCREF(PyArrayScalar_True),   \
                PyArrayScalar_True

/* 定义了根据类名 cls 动态生成对应的新的 NumPy 标量对象的宏 */
#define PyArrayScalar_New(cls) \
        Py##cls##ArrType_Type.tp_alloc(&Py##cls##ArrType_Type, 0)

#ifndef Py_LIMITED_API
/* 非限制 API 情况下，使用 PyArrayScalar_VAL 宏获取标量对象的值 */
#define PyArrayScalar_VAL(obj, cls)             \
        ((Py##cls##ScalarObject *)obj)->obval

/* 非限制 API 情况下，使用 PyArrayScalar_ASSIGN 宏给标量对象赋值 */
#define PyArrayScalar_ASSIGN(obj, cls, val) \
        PyArrayScalar_VAL(obj, cls) = val
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_ */
```