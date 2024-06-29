# `.\numpy\numpy\_core\src\multiarray\scalarapi.c`

```
/*
   定义 NPY_NO_DEPRECATED_API 为 NPY_API_VERSION，避免使用已废弃的 NumPy API 版本
   定义 _MULTIARRAYMODULE，用于多维数组模块

   引入必要的头文件和库
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>             // Python C API 的主头文件
#include <structmember.h>       // 结构成员相关宏定义

#include "numpy/arrayobject.h"  // NumPy 数组对象相关接口
#include "numpy/arrayscalars.h" // NumPy 数组标量对象接口

#include "numpy/npy_math.h"     // NumPy 数学库函数接口

#include "npy_config.h"         // NumPy 配置文件

// 引入其他自定义的头文件
#include "array_coercion.h"     // 数组类型转换相关
#include "ctors.h"              // 构造函数相关
#include "descriptor.h"         // 数组描述符相关
#include "dtypemeta.h"          // 数据类型元信息相关
#include "scalartypes.h"        // 标量类型相关

#include "common.h"             // 公共函数和宏定义

// 定义静态函数 _descr_from_subtype，根据子类型创建 PyArray_Descr 对象
static PyArray_Descr *
_descr_from_subtype(PyObject *type)
{
    PyObject *mro;
    mro = ((PyTypeObject *)type)->tp_mro;
    // 如果类型的方法解析顺序（MRO）长度小于 2，返回通用的对象类型描述符
    if (PyTuple_GET_SIZE(mro) < 2) {
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    // 否则返回基于 MRO 的第二个项的类型描述符
    return PyArray_DescrFromTypeObject(PyTuple_GET_ITEM(mro, 1));
}

// 定义 NPY_NO_EXPORT 的标量值提取函数，根据给定的描述符提取标量值
NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr)
{
    int type_num;
    int align;
    uintptr_t memloc;
    // 如果描述符为 NULL，根据标量对象生成描述符，并获取其类型编号
    if (descr == NULL) {
        descr = PyArray_DescrFromScalar(scalar);
        type_num = descr->type_num;
        Py_DECREF(descr);
    }
    else {
        // 否则直接获取描述符的类型编号
        type_num = descr->type_num;
    }
    // 根据类型编号进行不同的处理分支
    switch (type_num) {
#define CASE(ut,lt) case NPY_##ut: return &PyArrayScalar_VAL(scalar, lt)
        // 不同类型的标量值提取宏定义分支
        CASE(BOOL, Bool);
        CASE(BYTE, Byte);
        CASE(UBYTE, UByte);
        CASE(SHORT, Short);
        CASE(USHORT, UShort);
        CASE(INT, Int);
        CASE(UINT, UInt);
        CASE(LONG, Long);
        CASE(ULONG, ULong);
        CASE(LONGLONG, LongLong);
        CASE(ULONGLONG, ULongLong);
        CASE(HALF, Half);
        CASE(FLOAT, Float);
        CASE(DOUBLE, Double);
        CASE(LONGDOUBLE, LongDouble);
        CASE(CFLOAT, CFloat);
        CASE(CDOUBLE, CDouble);
        CASE(CLONGDOUBLE, CLongDouble);
        CASE(OBJECT, Object);
        CASE(DATETIME, Datetime);
        CASE(TIMEDELTA, Timedelta);
#undef CASE
        // 对于字符串类型，返回字符串对象的指针作为标量值
        case NPY_STRING:
            return (void *)PyBytes_AsString(scalar);
        // 对于 Unicode 类型，如果未初始化则进行延迟初始化，并返回 Unicode 数据指针
        case NPY_UNICODE:
            /* 懒初始化，以减少字符串标量占用的内存 */
            if (PyArrayScalar_VAL(scalar, Unicode) == NULL) {
                Py_UCS4 *raw_data = PyUnicode_AsUCS4Copy(scalar);
                if (raw_data == NULL) {
                    return NULL;
                }
                PyArrayScalar_VAL(scalar, Unicode) = raw_data;
                return (void *)raw_data;
            }
            return PyArrayScalar_VAL(scalar, Unicode);
        // 对于 VOID 类型，直接返回 VOID 数据的指针作为标量值
        case NPY_VOID:
            /* 注意：这里不需要使用 &，因此不能使用 CASE 宏 */
            return PyArrayScalar_VAL(scalar, Void);
    }

    /*
     * 如果标量是用户定义类型，并且关联有（注册的）dtype，
     * 那么它不能是灵活的（用户 dtype 不能是灵活的），
     * 因此我们可以（而且基本上只能）假设以下逻辑始终有效。
     * 即假设这个逻辑对我们的大多数类型也有效。
     */

    /*
     * 使用对齐标志确定在 PyObject_HEAD 之后数据从何处开始
     */
    memloc = (uintptr_t)scalar;
    memloc += sizeof(PyObject);
    /* 现在将内存地址向最接近的对齐值进行调整 */
    align = descr->alignment;  // 获取描述符中的对齐值
    if (align > 1) {  // 如果对齐值大于1
        memloc = ((memloc + align - 1) / align) * align;  // 根据对齐值调整内存地址
    }
    return (void *)memloc;  // 将调整后的内存地址转换为void指针并返回
/*NUMPY_API
 * return 1 if an object is exactly a numpy scalar
 */
NPY_NO_EXPORT int
PyArray_CheckAnyScalarExact(PyObject * obj)
{
    // 如果传入的对象为空，则设置错误并返回0
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "obj is NULL in PyArray_CheckAnyScalarExact");
        return 0;
    }

    // 调用is_anyscalar_exact函数检查对象是否为精确的任意标量
    return is_anyscalar_exact(obj);
}

/*NUMPY_API
 * Convert to c-type
 *
 * no error checking is performed -- ctypeptr must be same type as scalar
 * in case of flexible type, the data is not copied
 * into ctypeptr which is expected to be a pointer to pointer
 */
NPY_NO_EXPORT void
PyArray_ScalarAsCtype(PyObject *scalar, void *ctypeptr)
{
    PyArray_Descr *typecode;
    void *newptr;

    // 从标量对象中获取描述符
    typecode = PyArray_DescrFromScalar(scalar);
    // 通过标量值函数获取标量的内存地址
    newptr = scalar_value(scalar, typecode);

    // 如果类型码为扩展类型，ctypeptr是指针的指针，直接赋值
    if (PyTypeNum_ISEXTENDED(typecode->type_num)) {
        void **ct = (void **)ctypeptr;
        *ct = newptr;
    }
    // 否则，使用memcpy将数据拷贝到ctypeptr指向的内存中
    else {
        memcpy(ctypeptr, newptr, typecode->elsize);
    }
    Py_DECREF(typecode);
    return;
}

/*NUMPY_API
 * Cast Scalar to c-type
 *
 * The output buffer must be large-enough to receive the value, this function
 * should only be used for subclasses of `np.generic`, we can only guarantee
 * it works for NumPy builtins.
 */
NPY_NO_EXPORT int
PyArray_CastScalarToCtype(PyObject *scalar, void *ctypeptr,
                          PyArray_Descr *outcode)
{
    PyArray_Descr* descr;

    // 获取标量对象的描述符
    descr = PyArray_DescrFromScalar(scalar);
    if (descr == NULL) {
        return -1;
    }
    // 获取标量值的内存地址
    void *src = scalar_value(scalar, descr);
    if (src == NULL) {
        Py_DECREF(descr);
        return -1;
    }

    // 调用np_cast_raw_scalar_item进行类型转换
    int res = npy_cast_raw_scalar_item(descr, src, outcode, ctypeptr);
    Py_DECREF(descr);
    return res;
}

/*NUMPY_API
 * Cast Scalar to c-type
 */
NPY_NO_EXPORT int
PyArray_CastScalarDirect(PyObject *scalar, PyArray_Descr *indescr,
                         void *ctypeptr, int outtype)
{
    // 获取输出数据类型的描述符
    PyArray_Descr *out_dt = PyArray_DescrFromType(outtype);
    if (out_dt == NULL) {
        return -1;
    }
    // 获取输入标量值的内存地址
    void *src = scalar_value(scalar, indescr);
    if (src == NULL) {
        Py_DECREF(out_dt);
        return -1;
    }

    // 调用np_cast_raw_scalar_item进行直接类型转换
    int res = npy_cast_raw_scalar_item(indescr, src, out_dt, ctypeptr);
    Py_DECREF(out_dt);
    return res;
}

/*NUMPY_API
 * Get 0-dim array from scalar
 *
 * 0-dim array from array-scalar object
 * always contains a copy of the data
 * unless outcode is NULL, it is of void type and the referrer does
 * not own it either.
 *
 * steals reference to outcode
 */
NPY_NO_EXPORT PyObject *
PyArray_FromScalar(PyObject *scalar, PyArray_Descr *outcode)
{
    /* convert to 0-dim array of scalar typecode */
    // 获取标量对象的类型描述符
    PyArray_Descr *typecode = PyArray_DescrFromScalar(scalar);
    if (typecode == NULL) {
        Py_XDECREF(outcode);
        return NULL;
    }

    // ...
}
    # 检查是否为 void 类型，并且标量对象没有所有权且输出代码为空
    if ((typecode->type_num == NPY_VOID) &&
            !(((PyVoidScalarObject *)scalar)->flags & NPY_ARRAY_OWNDATA) &&
            outcode == NULL) {
        # 创建新的数组对象，使用给定的描述符和基础数据
        return PyArray_NewFromDescrAndBase(
                &PyArray_Type, typecode,
                0, NULL, NULL,
                ((PyVoidScalarObject *)scalar)->obval,
                ((PyVoidScalarObject *)scalar)->flags,
                NULL, (PyObject *)scalar);
    }

    # 使用给定的描述符创建新的数组对象
    PyArrayObject *r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
            typecode,
            0, NULL,
            NULL, NULL, 0, NULL);
    if (r == NULL) {
        # 如果创建失败，释放输出代码对象并返回空
        Py_XDECREF(outcode);
        return NULL;
    }
    /* the dtype used by the array may be different to the one requested */
    # 数组使用的数据类型可能与请求的不同

    # 检查数据类型标志是否支持使用 setitem 方法
    typecode = PyArray_DESCR(r);
    if (PyDataType_FLAGCHK(typecode, NPY_USE_SETITEM)) {
        # 如果支持，调用数组类型的 setitem 函数设置数据
        if (PyDataType_GetArrFuncs(typecode)->setitem(scalar, PyArray_DATA(r), r) < 0) {
            # 如果设置失败，释放数组对象和输出代码对象并返回空
            Py_DECREF(r);
            Py_XDECREF(outcode);
            return NULL;
        }
    }
    else {
        # 否则，获取标量对象的内存指针
        char *memptr = scalar_value(scalar, typecode);

        # 将标量对象的数据复制到数组对象中
        memcpy(PyArray_DATA(r), memptr, PyArray_ITEMSIZE(r));
        if (PyDataType_FLAGCHK(typecode, NPY_ITEM_HASOBJECT)) {
            /* Need to INCREF just the PyObject portion */
            # 如果数据类型标志表明对象需要增加引用计数，则增加对象部分的引用计数
            PyArray_Item_INCREF(memptr, typecode);
        }
    }

    # 如果输出代码为空，直接返回数组对象
    if (outcode == NULL) {
        return (PyObject *)r;
    }
    # 如果输出代码与数组数据类型等价
    if (PyArray_EquivTypes(outcode, typecode)) {
        # 如果数据类型等价且长度相同（对于扩展类型）
        if (!PyTypeNum_ISEXTENDED(typecode->type_num)
                || (outcode->elsize == typecode->elsize)) {
            '''
             * 由于类型是等价的，并且我们还没有将数组传递给任何人，
             * 让我们把数据类型固定为请求的类型，即使它与传入的类型是等价的。
             '''
            # 将数组的描述符设置为输出代码
            Py_SETREF(((PyArrayObject_fields *)r)->descr, outcode);

            return (PyObject *)r;
        }
    }

    # 如果需要，将数组转换为所需的输出数据类型
    PyObject *ret = PyArray_CastToType(r, outcode, 0);
    Py_DECREF(r);
    return ret;
/*New reference */
/*NUMPY_API
 */
/* 从 Python 类型对象创建 NumPy 数组描述符。
   如果类型是内置类型，则使用其类型编号。
   返回对应的数组描述符对象或者在出错时返回 NULL。
*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromTypeObject(PyObject *type)
{
    /* 如果是内置类型，获取其类型编号 */
    int typenum = _typenum_fromtypeobj(type, 1);
    if (typenum != NPY_NOTYPE) {
        // 根据类型编号获取对应的数组描述符对象并返回
        return PyArray_DescrFromType(typenum);
    }

    /* 检查通用的类型 */
    if ((type == (PyObject *) &PyNumberArrType_Type) ||
            (type == (PyObject *) &PyInexactArrType_Type) ||
            (type == (PyObject *) &PyFloatingArrType_Type)) {
        // 警告：将 `np.inexact` 或 `np.floating` 转换为 dtype 已不建议使用。
        // 返回默认的 `float64` 数组描述符对象，不严格正确。
        if (DEPRECATE("Converting `np.inexact` or `np.floating` to "
                      "a dtype is deprecated. The current result is `float64` "
                      "which is not strictly correct.") < 0) {
            return NULL;
        }
        typenum = NPY_DOUBLE;
    }
    else if (type == (PyObject *) &PyComplexFloatingArrType_Type) {
        // 警告：将 `np.complex` 转换为 dtype 已不建议使用。
        // 返回默认的 `complex128` 数组描述符对象，不严格正确。
        if (DEPRECATE("Converting `np.complex` to a dtype is deprecated. "
                      "The current result is `complex128` which is not "
                      "strictly correct.") < 0) {
            return NULL;
        }
        typenum = NPY_CDOUBLE;
    }
    else if ((type == (PyObject *) &PyIntegerArrType_Type) ||
            (type == (PyObject *) &PySignedIntegerArrType_Type)) {
        // 警告：将 `np.integer` 或 `np.signedinteger` 转换为 dtype 已不建议使用。
        // 返回默认的 `np.int_` 或者系统相关的整数类型数组描述符对象，不严格正确。
        if (DEPRECATE("Converting `np.integer` or `np.signedinteger` to "
                      "a dtype is deprecated. The current result is "
                      "`np.dtype(np.int_)` which is not strictly correct. "
                      "Note that the result depends on the system. To ensure "
                      "stable results use may want to use `np.int64` or "
                      "`np.int32`.") < 0) {
            return NULL;
        }
        typenum = NPY_LONG;
    }
    else if (type == (PyObject *) &PyUnsignedIntegerArrType_Type) {
        // 警告：将 `np.unsignedinteger` 转换为 dtype 已不建议使用。
        // 返回默认的 `np.uint` 或者系统相关的无符号整数类型数组描述符对象，不严格正确。
        if (DEPRECATE("Converting `np.unsignedinteger` to a dtype is "
                      "deprecated. The current result is `np.dtype(np.uint)` "
                      "which is not strictly correct. Note that the result "
                      "depends on the system. To ensure stable results you may "
                      "want to use `np.uint64` or `np.uint32`.") < 0) {
            return NULL;
        }
        typenum = NPY_ULONG;
    }
    else if (type == (PyObject *) &PyCharacterArrType_Type) {
        // 警告：将 `np.character` 转换为 dtype 已不建议使用。
        // 返回默认的 `np.str_` 或者 'S1' 类型数组描述符对象，不严格正确。
        if (DEPRECATE("Converting `np.character` to a dtype is deprecated. "
                      "The current result is `np.dtype(np.str_)` "
                      "which is not strictly correct. Note that `np.character` "
                      "is generally deprecated and 'S1' should be used.") < 0) {
            return NULL;
        }
        typenum = NPY_STRING;
    }
    /*
     * 否则 --- type 是数组标量的子类型，
     * 没有对应于注册的数据类型对象。
     */
    else if ((type == (PyObject *) &PyGenericArrType_Type) ||
            (type == (PyObject *) &PyFlexibleArrType_Type)) {
        /*
         * 如果是 `np.generic` 或 `np.flexible` 类型，
         * 转换为 dtype 已经被弃用。
         * 当前结果是 `np.dtype(np.void)`，
         * 这不是严格正确的。
         */
        if (DEPRECATE("Converting `np.generic` to a dtype is "
                      "deprecated. The current result is `np.dtype(np.void)` "
                      "which is not strictly correct.") < 0) {
            return NULL;
        }
        // 设置 typenum 为 NPY_VOID
        typenum = NPY_VOID;
    }

    // 如果 typenum 不等于 NPY_NOTYPE
    if (typenum != NPY_NOTYPE) {
        // 返回对应 typenum 的 PyArray_Descr 对象
        return PyArray_DescrFromType(typenum);
    }

    /*
     * 否则 --- type 是数组标量的子类型，
     * 没有对应于注册的数据类型对象。
     */

    /* 对于 VOID 子类型执行特殊操作 */
    if (PyType_IsSubtype((PyTypeObject *)type, &PyVoidArrType_Type)) {
        // 创建一个新的 _PyArray_LegacyDescr 对象，类型为 NPY_VOID
        _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr  *)PyArray_DescrNewFromType(NPY_VOID);
        if (new == NULL) {
            return NULL;
        }
        // 尝试从 dtype 属性转换为 _PyArray_LegacyDescr 类型
        _PyArray_LegacyDescr *conv = (_PyArray_LegacyDescr *)(
                _arraydescr_try_convert_from_dtype_attr(type));
        if (conv == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        // 如果转换成功且是遗留类型，则复制字段、名称、元素大小和子数组信息
        if ((PyObject *)conv != Py_NotImplemented && PyDataType_ISLEGACY(conv)) {
            new->fields = conv->fields;
            Py_XINCREF(new->fields);
            new->names = conv->names;
            Py_XINCREF(new->names);
            new->elsize = conv->elsize;
            new->subarray = conv->subarray;
            conv->subarray = NULL;
        }
        Py_DECREF(conv);
        Py_XDECREF(new->typeobj);
        // 设置新的 _PyArray_LegacyDescr 的 typeobj 为当前 type 的 PyTypeObject 类型
        new->typeobj = (PyTypeObject *)type;
        Py_INCREF(type);
        // 返回新的 _PyArray_LegacyDescr 类型对象
        return (PyArray_Descr *)new;
    }
    // 否则，调用 _descr_from_subtype 函数处理
    return _descr_from_subtype(type);
/*NUMPY_API
 * 从数组标量对象中获取描述符对象。
 *
 * 返回一个新的引用。
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromScalar(PyObject *sc)
{
    int type_num;  // 声明一个整数变量type_num
    PyArray_Descr *descr;  // 声明一个PyArray_Descr类型的指针descr，用于存储描述符对象的地址

    if (PyArray_IsScalar(sc, Void)) {  // 如果输入的对象是空类型的标量
        descr = (PyArray_Descr *)((PyVoidScalarObject *)sc)->descr;  // 获取空类型标量对象中的描述符对象
        Py_INCREF(descr);  // 增加描述符对象的引用计数
        return descr;  // 返回描述符对象
    }

    if (PyArray_IsScalar(sc, Datetime) || PyArray_IsScalar(sc, Timedelta)) {
        PyArray_DatetimeMetaData *dt_data;  // 声明一个PyArray_DatetimeMetaData类型的指针dt_data

        if (PyArray_IsScalar(sc, Datetime)) {
            descr = PyArray_DescrNewFromType(NPY_DATETIME);  // 如果是日期时间类型的标量，创建一个日期时间描述符对象
        }
        else {
            /* Timedelta */
            descr = PyArray_DescrNewFromType(NPY_TIMEDELTA);  // 如果是时间增量类型的标量，创建一个时间增量描述符对象
        }
        if (descr == NULL) {  // 如果描述符对象创建失败
            return NULL;  // 返回空指针
        }
        dt_data = &(((PyArray_DatetimeDTypeMetaData *)((_PyArray_LegacyDescr *)descr)->c_metadata)->meta);  // 获取日期时间数据元数据
        memcpy(dt_data, &((PyDatetimeScalarObject *)sc)->obmeta, sizeof(PyArray_DatetimeMetaData));  // 复制标量对象中的日期时间元数据到描述符对象中

        return descr;  // 返回描述符对象
    }

    descr = PyArray_DescrFromTypeObject((PyObject *)Py_TYPE(sc));  // 从类型对象中获取描述符对象
    if (descr == NULL) {  // 如果获取描述符对象失败
        return NULL;  // 返回空指针
    }
    if (PyDataType_ISLEGACY(descr) && PyDataType_ISUNSIZED(descr)) {  // 如果描述符对象是遗留类型且未定义大小
        PyArray_DESCR_REPLACE(descr);  // 替换描述符对象
        if (descr == NULL) {  // 如果描述符对象替换失败
            return NULL;  // 返回空指针
        }
        type_num = descr->type_num;  // 获取描述符对象的类型编号
        if (type_num == NPY_STRING) {  // 如果是字符串类型
            descr->elsize = PyBytes_GET_SIZE(sc);  // 获取字节大小
        }
        else if (type_num == NPY_UNICODE) {  // 如果是Unicode类型
            descr->elsize = PyUnicode_GET_LENGTH(sc) * 4;  // 获取Unicode字符的字节大小
        }
        else {
            _PyArray_LegacyDescr *ldescr = (_PyArray_LegacyDescr *)descr;  // 强制转换为遗留描述符对象类型
            PyArray_Descr *dtype;  // 声明一个PyArray_Descr类型的指针dtype
            dtype = (PyArray_Descr *)PyObject_GetAttrString(sc, "dtype");  // 从对象中获取dtype属性
            if (dtype != NULL) {  // 如果获取dtype属性成功
                descr->elsize = dtype->elsize;  // 获取dtype对象的元素大小
                ldescr->fields = PyDataType_FIELDS(dtype);  // 获取字段信息
                Py_XINCREF(ldescr->fields);  // 增加字段信息的引用计数
                ldescr->names = PyDataType_NAMES(dtype);  // 获取字段名信息
                Py_XINCREF(ldescr->names);  // 增加字段名信息的引用计数
                Py_DECREF(dtype);  // 释放dtype对象的引用
            }
            PyErr_Clear();  // 清除异常状态
        }
    }
    return descr;  // 返回描述符对象
}

/*NUMPY_API
 * 根据类型编号获取类型对象 -- 可能返回空值。
 *
 * 返回一个新的引用。
 */
NPY_NO_EXPORT PyObject *
PyArray_TypeObjectFromType(int type)
{
    PyArray_Descr *descr;  // 声明一个PyArray_Descr类型的指针descr
    PyObject *obj;  // 声明一个PyObject类型的指针obj

    descr = PyArray_DescrFromType(type);  // 根据类型编号获取描述符对象
    if (descr == NULL) {  // 如果获取描述符对象失败
        return NULL;  // 返回空指针
    }
    obj = (PyObject *)descr->typeobj;  // 获取描述符对象的类型对象
    Py_XINCREF(obj);  // 增加类型对象的引用计数
    Py_DECREF(descr);  // 释放描述符对象的引用
    return obj;  // 返回类型对象
}

/* 不对descr做任何操作（不会为NULL） */
/*NUMPY_API
  获取描述符描述的内存区域的标量等效对象。
*/
NPY_NO_EXPORT PyObject *
PyArray_Scalar(void *data, PyArray_Descr *descr, PyObject *base)
{
    PyTypeObject *type;  // 声明一个PyTypeObject类型的指针type
    PyObject *obj;  // 声明一个PyObject类型的指针obj
    void *destptr;  // 声明一个void类型的指针destptr
    PyArray_CopySwapFunc *copyswap;  // 声明一个PyArray_CopySwapFunc类型的指针copyswap
    int type_num;  // 声明一个整数变量type_num
    int itemsize;  // 声明一个整数变量itemsize
    int swap;  // 声明一个整数变量swap

    type_num = descr->type_num;  // 获取描述符对象的类型编号
    # 如果数据类型是布尔类型
    if (type_num == NPY_BOOL) {
        # 返回一个布尔值，从数据中获取
        PyArrayScalar_RETURN_BOOL_FROM_LONG(*(npy_bool*)data);
    }
    # 如果数据类型描述符包含 NPY_USE_GETITEM 标志
    else if (PyDataType_FLAGCHK(descr, NPY_USE_GETITEM)) {
        # 使用描述符对应的获取数组函数获取数据
        return PyDataType_GetArrFuncs(descr)->getitem(data, base);
    }
    # 获取元素大小
    itemsize = descr->elsize;
    # 获取数据交换函数
    copyswap = PyDataType_GetArrFuncs(descr)->copyswap;
    # 获取数据类型对象
    type = descr->typeobj;
    # 判断是否需要交换字节序
    swap = !PyArray_ISNBO(descr->byteorder);
    # 如果数据类型是字符串类型
    if (PyTypeNum_ISSTRING(type_num)) {
        /* 消除空字符 */
        char *dptr = data;

        # 将指针移动到字符串末尾
        dptr += itemsize - 1;
        # 如果末尾是空字符，向前移动指针直到非空字符为止
        while(itemsize && *dptr-- == 0) {
            itemsize--;
        }
        # 如果数据类型是 Unicode 并且字符串长度大于 0
        if (type_num == NPY_UNICODE && itemsize) {
            /*
             * 确保字符串长度是4的倍数
             * 向上舍入到最接近的倍数
             */
            itemsize = (((itemsize - 1) >> 2) + 1) << 2;
        }
    }
    # 如果数据类型是 Unicode 类型
    if (type_num == NPY_UNICODE) {
        /* 在这里需要完整的字符串长度，否则 copyswap 将写入过多的字节 */
        # 分配缓冲区
        void *buff = PyArray_malloc(descr->elsize);
        # 如果分配失败，返回内存错误
        if (buff == NULL) {
            return PyErr_NoMemory();
        }
        /* copyswap 需要一个数组对象，但实际上只关心数据类型 */
        # 如果基对象为空，创建一个虚拟数组对象作为基对象
        PyArrayObject_fields dummy_arr;
        if (base == NULL) {
            dummy_arr.descr = descr;
            base = (PyObject *)&dummy_arr;
        }
        # 执行数据交换
        copyswap(buff, data, swap, base);

        /* 截断发生在这里 */
        # 根据 buff 中的数据创建 Unicode 对象
        PyObject *u = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buff, itemsize / 4);
        # 释放缓冲区内存
        PyArray_free(buff);
        # 如果创建 Unicode 对象失败，返回空
        if (u == NULL) {
            return NULL;
        }

        # 构建一个参数元组
        PyObject *args = Py_BuildValue("(O)", u);
        # 如果构建参数元组失败，释放 Unicode 对象并返回空
        if (args == NULL) {
            Py_DECREF(u);
            return NULL;
        }
        # 使用类型对象的 tp_new 方法创建对象
        obj = type->tp_new(type, args, NULL);
        # 释放 Unicode 对象和参数元组的引用
        Py_DECREF(u);
        Py_DECREF(args);
        # 返回创建的对象
        return obj;
    }
    # 如果数据类型的 tp_itemsize 不为 0
    if (type->tp_itemsize != 0) {
        /* 字符串类型 */
        # 使用类型的 tp_alloc 方法分配对象
        obj = type->tp_alloc(type, itemsize);
    }
    else {
        # 否则，使用类型的 tp_alloc 方法分配对象
        obj = type->tp_alloc(type, 0);
    }
    # 如果分配对象失败，返回空
    if (obj == NULL) {
        return NULL;
    }
    # 如果数据类型是日期时间类型
    if (PyTypeNum_ISDATETIME(type_num)) {
        /*
         * 我们需要将分辨率信息复制到标量中
         * 从元数据字典中获取 void * 指针
         */
        # 获取日期时间元数据
        PyArray_DatetimeMetaData *dt_data;

        # 强制转换描述符类型，并获取元数据中的日期时间数据
        dt_data = &(((PyArray_DatetimeDTypeMetaData *)((_PyArray_LegacyDescr *)descr)->c_metadata)->meta);
        # 将日期时间数据复制到标量对象的元数据中
        memcpy(&(((PyDatetimeScalarObject *)obj)->obmeta), dt_data,
               sizeof(PyArray_DatetimeMetaData));
    }
    if (PyTypeNum_ISFLEXIBLE(type_num)) {
        // 检查是否是灵活类型（即字符串或对象）
        if (type_num == NPY_STRING) {
            // 如果是字符串类型，则获取其字符数据指针
            destptr = PyBytes_AS_STRING(obj);
            // 在Python版本低于3.11.0b0时，重置字符串对象的哈希值
            #if PY_VERSION_HEX < 0x030b00b0
                ((PyBytesObject *)obj)->ob_shash = -1;
            #endif
            // 将数据内容复制到字符串对象中
            memcpy(destptr, data, itemsize);
            // 返回修改后的对象
            return obj;
        }
        else {
            // 对于非字符串类型，获取其指针对象
            PyVoidScalarObject *vobj = (PyVoidScalarObject *)obj;
            // 清空对象的基类和描述符，设置为传入的描述符
            vobj->base = NULL;
            vobj->descr = (_PyArray_LegacyDescr *)descr;
            // 增加描述符的引用计数
            Py_INCREF(descr);
            // 清空对象的值，并设置其大小为传入的itemsize
            vobj->obval = NULL;
            Py_SET_SIZE(vobj, itemsize);
            // 设置对象的标志为C连续存储、F顺序存储和拥有数据标志
            vobj->flags = NPY_ARRAY_CARRAY | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA;
            // 默认不进行数据交换
            swap = 0;
            // 如果描述符包含字段信息
            if (PyDataType_HASFIELDS(descr)) {
                // 如果有基类存在，则设置基类信息，并且不拥有数据标志位清除
                if (base) {
                    // 增加基类的引用计数
                    Py_INCREF(base);
                    vobj->base = base;
                    // 获取基类的标志位
                    vobj->flags = PyArray_FLAGS((PyArrayObject *)base);
                    vobj->flags &= ~NPY_ARRAY_OWNDATA;
                    // 设置对象的值为传入的数据指针
                    vobj->obval = data;
                    // 返回修改后的对象
                    return obj;
                }
            }
            // 如果itemsize为0，则直接返回对象
            if (itemsize == 0) {
                return obj;
            }
            // 分配新的内存空间作为对象的值
            destptr = PyDataMem_NEW(itemsize);
            // 如果内存分配失败，则释放对象并返回内存错误异常
            if (destptr == NULL) {
                Py_DECREF(obj);
                return PyErr_NoMemory();
            }
            // 设置对象的值为新分配的内存空间
            vobj->obval = destptr;

            /*
             * 没有基类可用于复制交换，也不需要进行交换。
             * 直接将数据复制到目标中。
             */
            // 如果没有提供基类，则直接复制数据到目标中
            if (base == NULL) {
                memcpy(destptr, data, itemsize);
                // 返回修改后的对象
                return obj;
            }
        }
    }
    else {
        // 对于非灵活类型，使用标量值函数获取对象的值
        destptr = scalar_value(obj, descr);
    }
    /* 对象类型为OBJECT时，复制交换操作会增加引用计数 */
    // 调用复制交换函数对目标进行数据交换操作
    copyswap(destptr, data, swap, base);
    // 返回修改后的对象
    return obj;
/* 返回数组标量，如果遇到0维数组对象 */

/* NUMPY_API
 *
 * 如果数组是0维且匹配Python类型，则返回数组或适当的Python对象。
 * 对于0维数组，如果匹配Python类型，则释放对mp的引用。
 */
NPY_NO_EXPORT PyObject *
PyArray_Return(PyArrayObject *mp)
{
    // 如果传入的数组对象为空，则返回空
    if (mp == NULL) {
        return NULL;
    }
    // 如果发生了异常，释放对mp的引用并返回空
    if (PyErr_Occurred()) {
        Py_XDECREF(mp);
        return NULL;
    }
    // 如果mp不是数组对象，则直接返回mp
    if (!PyArray_Check(mp)) {
        return (PyObject *)mp;
    }
    // 如果数组是0维的
    if (PyArray_NDIM(mp) == 0) {
        // 将数组的数据转换为标量对象，并返回标量对象，同时释放对mp的引用
        PyObject *ret;
        ret = PyArray_ToScalar(PyArray_DATA(mp), mp);
        Py_DECREF(mp);
        return ret;
    }
    else {
        // 如果数组不是0维的，则直接返回mp
        return (PyObject *)mp;
    }
}
```