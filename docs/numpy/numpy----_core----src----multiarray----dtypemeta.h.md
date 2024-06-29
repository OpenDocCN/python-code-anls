# `.\numpy\numpy\_core\src\multiarray\dtypemeta.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_

// 定义 _MULTIARRAYMODULE 宏，用于标识当前为 multiarray 模块
#define _MULTIARRAYMODULE
// 包含 NumPy 的 arrayobject.h 头文件
#include "numpy/arrayobject.h"

// 包含自定义的 array_method.h 和 dtype_traversal.h 头文件
#include "array_method.h"
#include "dtype_traversal.h"

#ifdef __cplusplus
extern "C" {
#endif

// 包含 dtype_api.h 头文件，定义了部分公共的 DType 标志
#include "numpy/dtype_api.h"

/* DType flags, currently private, since we may just expose functions
   Other publicly visible flags are in _dtype_api.h                   */
// 定义 NPY_DT_LEGACY 标志，表示遗留 DType
#define NPY_DT_LEGACY 1 << 0

// 定义 DTypeMeta 结构体，包含一系列 DType 方法指针
typedef struct {
    // 从 Python 对象中发现描述符的函数指针
    PyArrayDTypeMeta_DiscoverDescrFromPyobject *discover_descr_from_pyobject;
    // 判断是否为已知的标量类型的函数指针
    PyArrayDTypeMeta_IsKnownScalarType *is_known_scalar_type;
    // 获取默认描述符的函数指针
    PyArrayDTypeMeta_DefaultDescriptor *default_descr;
    // 获取公共 DType 的函数指针
    PyArrayDTypeMeta_CommonDType *common_dtype;
    // 获取公共实例的函数指针
    PyArrayDTypeMeta_CommonInstance *common_instance;
    // 确保规范性的函数指针
    PyArrayDTypeMeta_EnsureCanonical *ensure_canonical;
    /*
     * 用于实验性用户 DType，目前仅使用。
     */
    // 设置项目的函数指针
    PyArrayDTypeMeta_SetItem *setitem;
    // 获取项目的函数指针
    PyArrayDTypeMeta_GetItem *getitem;
    /*
     * 可能为 NULL，获取一个清理函数的函数指针。清理包括释放任何引用的数据并将其置于安全状态。
     * 对于 Python 对象，使用 `Py_CLEAR` 相当于 `Py_DECREF` 并将 `PyObject *` 置为 NULL。
     * 第二次调用清理函数必须安全。
     * 如果 DType 类未实现 `get_clear_loop`，设置 NPY_ITEM_REFCOUNT 为其 dtype 实例是无效的。
     * 注意，NPY_ITEM_REFCOUNT 可以指示非 Python 对象的引用。
     */
    PyArrayMethod_GetTraverseLoop *get_clear_loop;
    /*
     * 可能为 NULL，设置一个函数指针，指向一个用于填充零值的遍历循环函数。
     * 如果未定义 get_fill_zero_loop 或其设置的函数指针为 NULL，则数组缓冲区将使用 calloc 分配。
     * 如果定义了该函数并设置了非 NULL 的函数指针，则数组缓冲区将使用 malloc 分配，并调用零填充循环函数指针以填充缓冲区。
     * 为获得最佳性能，请避免在对于该 DType 适合使用 calloc 的情况下使用该函数。
     * 注意，目前仅用于零填充新分配的数组缓冲区。尝试用于已填充缓冲区的零填充将无法正确工作，特别是对于包含引用的数组。
     * 如果需要这样做，请首先清除数组。
     */
    PyArrayMethod_GetTraverseLoop *get_fill_zero_loop;
    /*
     * 可能为 NULL，执行 DType 的最终处理函数，返回该 DType 或具有与操作数 DType 相同参数（如果有）的新创建实例。
     */
    PyArrayDTypeMeta_FinalizeDescriptor *finalize_descr;
    /*
    `
    /*
     * 存储了用于类型转换的 ArrayMethod 对象，以便快速访问：
     */
    PyArrayMethodObject *within_dtype_castingimpl;
    
    /*
     * 字典类型的 ArrayMethod 对象，表示大多数可能的类型转换（结构化和对象类型除外）。
     * 这在未来可能会变成弱映射。
     */
    PyObject *castingimpls;
    
    /*
     * 存储 `descr->f` 的空间，因为在过渡期间可能需要在这里进行一些自定义，
     * 我们需要在每个 dtype 实例上设置它以确保向后兼容性。（保持在末尾）
     */
    PyArray_ArrFuncs f;
/*
 * 结构体定义：NPY_DType_Slots
 * 该结构体定义了一系列用于描述数据类型的函数指针
 */
} NPY_DType_Slots;

/*
 * 更新常量定义：NPY_NUM_DTYPE_SLOTS
 * 如果在 within_dtype_castingimpl 之前添加了新的插槽，必须更新此常量
 */
#define NPY_NUM_DTYPE_SLOTS 11
/*
 * 更新常量定义：NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS
 * 描述 PyArray 的数组函数插槽数量
 */
#define NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS 22
/*
 * 计算最大的数组函数插槽索引
 */
#define NPY_DT_MAX_ARRFUNCS_SLOT \
  NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS + _NPY_DT_ARRFUNCS_OFFSET

/*
 * 定义宏：NPY_DT_SLOTS(dtype)
 * 获取给定数据类型的插槽结构体指针
 */
#define NPY_DT_SLOTS(dtype) ((NPY_DType_Slots *)(dtype)->dt_slots)

/*
 * 定义宏：NPY_DT_is_legacy(dtype)
 * 检查给定数据类型是否为遗留类型
 */
#define NPY_DT_is_legacy(dtype) (((dtype)->flags & NPY_DT_LEGACY) != 0)
/*
 * 定义宏：NPY_DT_is_abstract(dtype)
 * 检查给定数据类型是否为抽象类型
 */
#define NPY_DT_is_abstract(dtype) (((dtype)->flags & NPY_DT_ABSTRACT) != 0)
/*
 * 定义宏：NPY_DT_is_parametric(dtype)
 * 检查给定数据类型是否为参数化类型
 */
#define NPY_DT_is_parametric(dtype) (((dtype)->flags & NPY_DT_PARAMETRIC) != 0)
/*
 * 定义宏：NPY_DT_is_numeric(dtype)
 * 检查给定数据类型是否为数值类型
 */
#define NPY_DT_is_numeric(dtype) (((dtype)->flags & NPY_DT_NUMERIC) != 0)
/*
 * 定义宏：NPY_DT_is_user_defined(dtype)
 * 检查给定数据类型是否为用户定义类型
 */
#define NPY_DT_is_user_defined(dtype) (((dtype)->type_num == -1))

/*
 * 定义多个类方法调用的宏，因为这些宏需要数据类型作为第一个参数
 */
#define NPY_DT_CALL_discover_descr_from_pyobject(dtype, obj)  \
    NPY_DT_SLOTS(dtype)->discover_descr_from_pyobject(dtype, obj)
#define NPY_DT_CALL_is_known_scalar_type(dtype, obj)  \
    (NPY_DT_SLOTS(dtype)->is_known_scalar_type != NULL  \
        && NPY_DT_SLOTS(dtype)->is_known_scalar_type(dtype, obj))
#define NPY_DT_CALL_default_descr(dtype)  \
    NPY_DT_SLOTS(dtype)->default_descr(dtype)
#define NPY_DT_CALL_common_dtype(dtype, other)  \
    NPY_DT_SLOTS(dtype)->common_dtype(dtype, other)
#define NPY_DT_CALL_ensure_canonical(descr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->ensure_canonical(descr)
#define NPY_DT_CALL_getitem(descr, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->getitem(descr, data_ptr)
#define NPY_DT_CALL_setitem(descr, value, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->setitem(descr, value, data_ptr)

/*
 * 内联函数：PyArray_DTypeFromTypeNum
 * 根据类型编号创建一个 PyArray_DTypeMeta 对象
 * 这个函数在增量实现基于 DTypeMeta 的新数据类型时非常方便
 */
static inline PyArray_DTypeMeta *
PyArray_DTypeFromTypeNum(int typenum)
{
    // 从类型编号创建描述符
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    // 获取描述符对应的 DTypeMeta 对象
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);
    // 增加对象的引用计数
    Py_INCREF(dtype);
    // 减少描述符的引用计数
    Py_DECREF(descr);
    // 返回 DTypeMeta 对象
    return dtype;
}

/*
 * 不导出的函数：dtypemeta_discover_as_default
 * 作为默认值发现函数
 */
NPY_NO_EXPORT PyArray_Descr *
dtypemeta_discover_as_default(
        PyArray_DTypeMeta *cls, PyObject* obj);

/*
 * 不导出的函数：dtypemeta_initialize_struct_from_spec
 * 从规范初始化结构体函数
 */
NPY_NO_EXPORT int
dtypemeta_initialize_struct_from_spec(PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec, int priv);

/*
 * 不导出的函数：python_builtins_are_known_scalar_types
 * Python 内建类型是否为已知标量类型函数
 */
NPY_NO_EXPORT int
python_builtins_are_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype);

/*
 * 不导出的函数：dtypemeta_wrap_legacy_descriptor
 * 包装遗留描述符函数
 */
NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(
    _PyArray_LegacyDescr *descr, PyArray_ArrFuncs *arr_funcs,
    PyTypeObject *dtype_super_class, const char *name, const char *alias);

/*
 * 不导出的函数：initialize_legacy_dtypemeta_aliases
 * 初始化遗留 DTypeMeta 别名函数
 */
NPY_NO_EXPORT void
initialize_legacy_dtypemeta_aliases(_PyArray_LegacyDescr **_builtin_descrs);

/*
 * NumPy 内建数据类型的注释
 */

// 注意：内建的遗留数据类型没有静态的 DTypeMeta
// 声明外部全局变量，指向不同数据类型的元数据对象
extern PyArray_DTypeMeta *_Bool_dtype;
extern PyArray_DTypeMeta *_Byte_dtype;
extern PyArray_DTypeMeta *_UByte_dtype;
extern PyArray_DTypeMeta *_Short_dtype;
extern PyArray_DTypeMeta *_UShort_dtype;
extern PyArray_DTypeMeta *_Int_dtype;
extern PyArray_DTypeMeta *_UInt_dtype;
extern PyArray_DTypeMeta *_Long_dtype;
extern PyArray_DTypeMeta *_ULong_dtype;
extern PyArray_DTypeMeta *_LongLong_dtype;
extern PyArray_DTypeMeta *_ULongLong_dtype;
extern PyArray_DTypeMeta *_Int8_dtype;
extern PyArray_DTypeMeta *_UInt8_dtype;
extern PyArray_DTypeMeta *_Int16_dtype;
extern PyArray_DTypeMeta *_UInt16_dtype;
extern PyArray_DTypeMeta *_Int32_dtype;
extern PyArray_DTypeMeta *_UInt32_dtype;
extern PyArray_DTypeMeta *_Int64_dtype;
extern PyArray_DTypeMeta *_UInt64_dtype;
extern PyArray_DTypeMeta *_Intp_dtype;
extern PyArray_DTypeMeta *_UIntp_dtype;
extern PyArray_DTypeMeta *_DefaultInt_dtype;
extern PyArray_DTypeMeta *_Half_dtype;
extern PyArray_DTypeMeta *_Float_dtype;
extern PyArray_DTypeMeta *_Double_dtype;
extern PyArray_DTypeMeta *_LongDouble_dtype;
extern PyArray_DTypeMeta *_CFloat_dtype;
extern PyArray_DTypeMeta *_CDouble_dtype;
extern PyArray_DTypeMeta *_CLongDouble_dtype;
extern PyArray_DTypeMeta *_Bytes_dtype;
extern PyArray_DTypeMeta *_Unicode_dtype;
extern PyArray_DTypeMeta *_Datetime_dtype;
extern PyArray_DTypeMeta *_Timedelta_dtype;
extern PyArray_DTypeMeta *_Object_dtype;
extern PyArray_DTypeMeta *_Void_dtype;

// 定义宏，简化访问各种数据类型的元数据对象的方式
#define PyArray_BoolDType (*(_Bool_dtype))
#define PyArray_ByteDType (*(_Byte_dtype))
#define PyArray_UByteDType (*(_UByte_dtype))
#define PyArray_ShortDType (*(_Short_dtype))
#define PyArray_UShortDType (*(_UShort_dtype))
#define PyArray_IntDType (*(_Int_dtype))
#define PyArray_UIntDType (*(_UInt_dtype))
#define PyArray_LongDType (*(_Long_dtype))
#define PyArray_ULongDType (*(_ULong_dtype))
#define PyArray_LongLongDType (*(_LongLong_dtype))
#define PyArray_ULongLongDType (*(_ULongLong_dtype))
#define PyArray_Int8DType (*(_Int8_dtype))
#define PyArray_UInt8DType (*(_UInt8_dtype))
#define PyArray_Int16DType (*(_Int16_dtype))
#define PyArray_UInt16DType (*(_UInt16_dtype))
#define PyArray_Int32DType (*(_Int32_dtype))
#define PyArray_UInt32DType (*(_UInt32_dtype))
#define PyArray_Int64DType (*(_Int64_dtype))
#define PyArray_UInt64DType (*(_UInt64_dtype))
#define PyArray_IntpDType (*(_Intp_dtype))
#define PyArray_UIntpDType (*(_UIntp_dtype))
#define PyArray_DefaultIntDType (*(_DefaultInt_dtype))
#define PyArray_HalfDType (*(_Half_dtype))
#define PyArray_FloatDType (*(_Float_dtype))
#define PyArray_DoubleDType (*(_Double_dtype))
#define PyArray_LongDoubleDType (*(_LongDouble_dtype))
#define PyArray_CFloatDType (*(_CFloat_dtype))
定义了一个宏，用于获取 CFloatDType
#define PyArray_CDoubleDType (*(_CDouble_dtype))
定义了一个宏，用于获取 CDoubleDType
#define PyArray_CLongDoubleDType (*(_CLongDouble_dtype))
定义了一个宏，用于获取 CLongDoubleDType
/* String/Bytes */
#define PyArray_BytesDType (*(_Bytes_dtype))
定义了一个宏，用于获取 BytesDType
#define PyArray_UnicodeDType (*(_Unicode_dtype))
定义了一个宏，用于获取 UnicodeDType
// StringDType is not a legacy DType and has a static dtypemeta implementation
// we can refer to, so no need for the indirection we use for the built-in
// dtypes.
外部声明了一个 StringDType 变量
/* Datetime/Timedelta */
#define PyArray_DatetimeDType (*(_Datetime_dtype))
定义了一个宏，用于获取 DatetimeDType
#define PyArray_TimedeltaDType (*(_Timedelta_dtype))
定义了一个宏，用于获取 TimedeltaDType
/* Object/Void */
#define PyArray_ObjectDType (*(_Object_dtype))
定义了一个宏，用于获取 ObjectDType
#define PyArray_VoidDType (*(_Void_dtype))
定义了一个宏，用于获取 VoidDType

#ifdef __cplusplus
}
#endif


/* Internal version see dtypmeta.c for more information. */
// 获取数组函数的内部版本，详细信息请参考 dtypmeta.c
static inline PyArray_ArrFuncs *
PyDataType_GetArrFuncs(const PyArray_Descr *descr)
{
    return &NPY_DT_SLOTS(NPY_DTYPE(descr))->f;
}

/*
 * Internal versions.  Note that `PyArray_Pack` or `PyArray_Scalar` are often
 * preferred (PyArray_Pack knows how to cast and deal with arrays,
 * PyArray_Scalar will convert to the Python type).
 */
// 获取数组中的某一项的内部版本函数
static inline PyObject *
PyArray_GETITEM(const PyArrayObject *arr, const char *itemptr)
{
    return PyDataType_GetArrFuncs(((PyArrayObject_fields *)arr)->descr)->getitem(
            (void *)itemptr, (PyArrayObject *)arr);
}

// 设置数组中的某一项的内部版本函数
static inline int
PyArray_SETITEM(PyArrayObject *arr, char *itemptr, PyObject *v)
{
    return PyDataType_GetArrFuncs(((PyArrayObject_fields *)arr)->descr)->setitem(
            v, itemptr, arr);
}



#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_ */
```