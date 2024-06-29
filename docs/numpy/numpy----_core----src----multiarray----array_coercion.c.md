# `.\numpy\numpy\_core\src\multiarray\array_coercion.c`

```py
// 定义以避免使用废弃的 NumPy API 版本
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义用于标识 Multiarray 模块
#define _MULTIARRAYMODULE
// 定义用于标识 Umath 模块
#define _UMATHMODULE

// 清除 PY_SSIZE_T 的旧定义，确保只使用干净的定义
#define PY_SSIZE_T_CLEAN
// 包含 Python.h 头文件，提供 Python C API 的功能
#include <Python.h>

// 包含 NumPy 3k 兼容性工具头文件，提供与 Python 3k 兼容的功能
#include "numpy/npy_3kcompat.h"

// 包含低级分步循环头文件，定义了低级的步进循环操作
#include "lowlevel_strided_loops.h"
// 包含 NumPy 数组对象头文件，提供了数组对象的定义和操作
#include "numpy/arrayobject.h"
// 包含 NumPy 数学函数头文件，提供了数学函数的声明和定义
#include "numpy/npy_math.h"

// 包含描述符头文件，提供了数据类型描述符的定义
#include "descriptor.h"
// 包含数据类型转换头文件，提供了数据类型转换的相关函数和结构体
#include "convert_datatype.h"
// 包含数据类型元信息头文件，提供了数据类型元信息的定义
#include "dtypemeta.h"
// 包含字符串数据类型头文件，提供了字符串数据类型的定义
#include "stringdtype/dtype.h"

// 包含 NumPy 参数解析头文件，提供了解析参数的函数和结构体
#include "npy_argparse.h"
// 包含抽象数据类型头文件，提供了抽象数据类型的定义和操作
#include "abstractdtypes.h"
// 包含数组强制转换头文件，提供了数组类型之间强制转换的函数
#include "array_coercion.h"
// 包含构造函数头文件，提供了数组对象的构造函数定义
#include "ctors.h"
// 包含通用功能头文件，提供了一些通用的辅助函数和宏定义
#include "common.h"
// 包含日期时间头文件，提供了日期时间相关的函数和结构体
#include "_datetime.h"
// 包含 NumPy 导入头文件，提供了导入 NumPy 模块的函数和宏定义
#include "npy_import.h"
// 包含引用计数头文件，提供了引用计数相关的函数和宏定义
#include "refcount.h"

// 包含 Umath 模块的头文件，提供了 Umath 模块的函数和宏定义
#include "umathmodule.h"
/*
 * This file defines helpers for some of the ctors.c functions which
 * create an array from Python sequences and types.
 * When creating an array with ``np.array(...)`` we have to do two main things:
 *
 * 1. Find the exact shape of the resulting array
 * 2. Find the correct dtype of the resulting array.
 *
 * In most cases these two things are can be done in a single processing step.
 * There are in principle three different calls that should be distinguished:
 *
 * 1. The user calls ``np.array(..., dtype=np.dtype("<f8"))``
 * 2. The user calls ``np.array(..., dtype="S")``
 * 3. The user calls ``np.array(...)``
 *
 * In the first case, in principle only the shape needs to be found. In the
 * second case, the DType class (e.g. string) is already known but the DType
 * instance (e.g. length of the string) has to be found.
 * In the last case the DType class needs to be found as well. Note that
 * it is not necessary to find the DType class of the entire array, but
 * the DType class needs to be found for each element before the actual
 * dtype instance can be found.
 *
 * Further, there are a few other things to keep in mind when coercing arrays:
 *
 *   * For UFunc promotion, Python scalars need to be handled specially to
 *     allow value based casting.  This requires python complex/float to
 *     have their own DTypes.
 *   * It is necessary to decide whether or not a sequence is an element.
 *     For example tuples are considered elements for structured dtypes, but
 *     otherwise are considered sequences.
 *     This means that if a dtype is given (either as a class or instance),
 *     it can effect the dimension discovery part.
 *     For the "special" NumPy types structured void and "c" (single character)
 *     this is special cased.  For future user-types, this is currently
 *     handled by providing calling an `is_known_scalar` method.  This method
 *     currently ensures that Python numerical types are handled quickly.
 *
 * In the initial version of this implementation, it is assumed that dtype
 * discovery can be implemented sufficiently fast.  That is, it is not
 * necessary to create fast paths that only find the correct shape e.g. when
 * ``dtype=np.dtype("f8")`` is given.
 *
 * The code here avoid multiple conversion of array-like objects (including
 * sequences). These objects are cached after conversion, which will require
 * additional memory, but can drastically speed up coercion from array like
 * objects.
 */
/* 全局变量，用于存储 Python 类型到 DType 类型的映射字典 */
PyObject *_global_pytype_to_type_dict = NULL;


/* 枚举类型，用于在发现 dtype 和形状时跟踪或信号一些情况 */
enum _dtype_discovery_flags {
    FOUND_RAGGED_ARRAY = 1 << 0,                    // 发现了不规则数组
    GAVE_SUBCLASS_WARNING = 1 << 1,                 // 发出子类警告
    PROMOTION_FAILED = 1 << 2,                      // 升级失败
    DISCOVER_STRINGS_AS_SEQUENCES = 1 << 3,         // 将字符串发现为序列
    DISCOVER_TUPLES_AS_ELEMENTS = 1 << 4,           // 将元组发现为元素
    MAX_DIMS_WAS_REACHED = 1 << 5,                  // 达到最大维数
    DESCRIPTOR_WAS_SET = 1 << 6,                    // 描述符已设置
    COPY_WAS_CREATED_BY__ARRAY__ = 1 << 7,          // 复制由__array__创建
};


/**
 * 向全局类型字典中添加已知的序列类型，注意当传入 DType 时可能会忽略此查找。
 *
 * @return -1 表示错误，0 表示成功
 */
static int
_prime_global_pytype_to_type_dict(void)
{
    int res;

    /* 添加基本的 Python 序列类型 */
    res = PyDict_SetItem(_global_pytype_to_type_dict,
                         (PyObject *)&PyList_Type, Py_None);
    if (res < 0) {
        return -1;
    }
    res = PyDict_SetItem(_global_pytype_to_type_dict,
                         (PyObject *)&PyTuple_Type, Py_None);
    if (res < 0) {
        return -1;
    }
    /* NumPy 数组不作为标量处理 */
    res = PyDict_SetItem(_global_pytype_to_type_dict,
                         (PyObject *)&PyArray_Type, Py_None);
    if (res < 0) {
        return -1;
    }
    return 0;
}


/**
 * 将一个新的映射从 Python 类型到 DType 类添加。对于用户定义的传统 dtype，
 * 除非 pytype 是从 `np.generic` 继承的，否则此函数将不执行任何操作。
 *
 * 假设 DType 类确保持有 python 类型（这个假设是保证的）。
 * 此功能取代了 ``_typenum_fromtypeobj``。
 *
 * @param DType 要映射到的 DType 类
 * @param pytype 要映射的 Python 类型
 * @param userdef 是否是用户定义的标量。我们确保用户定义的标量从我们的标量继承（目前）。
 */
NPY_NO_EXPORT int
_PyArray_MapPyTypeToDType(
        PyArray_DTypeMeta *DType, PyTypeObject *pytype, npy_bool userdef)
{
    PyObject *Dtype_obj = (PyObject *)DType;

    if (userdef && !PyObject_IsSubclass(
                    (PyObject *)pytype, (PyObject *)&PyGenericArrType_Type)) {
        /*
         * 我们期望用户定义的 dtype（目前）将子类化某个 numpy 标量类，以允许自动发现。
         */
        if (NPY_DT_is_legacy(DType)) {
            /*
             * 对于传统的用户定义 dtype，发现依赖于子类化，但支持任意类型对象，因此不执行任何操作。
             */
            return 0;
        }
        /*
         * 我们当前强制用户 DType 必须从 `np.generic` 继承
         * （这应该成为 `np.generic` 基类，并可能被完全取消）。
         */
        PyErr_Format(PyExc_RuntimeError,
                "当前仅支持为从 `np.generic` 派生的标量注册 DType，得到 '%S'。",
                (PyObject *)pytype);
        return -1;
    }
    /* 如果全局字典 _global_pytype_to_type_dict 不存在，则创建 */
    if (NPY_UNLIKELY(_global_pytype_to_type_dict == NULL)) {
        _global_pytype_to_type_dict = PyDict_New();
        if (_global_pytype_to_type_dict == NULL) {
            // 创建失败，返回错误代码
            return -1;
        }
        // 初始化全局字典 _global_pytype_to_type_dict
        if (_prime_global_pytype_to_type_dict() < 0) {
            // 初始化失败，返回错误代码
            return -1;
        }
    }

    // 检查字典中是否包含给定的 pytype 对象
    int res = PyDict_Contains(_global_pytype_to_type_dict, (PyObject *)pytype);
    if (res < 0) {
        // 包含检查失败，返回错误代码
        return -1;
    }
    else if (DType == &PyArray_StringDType) {
        // 如果 DType 是 PyArray_StringDType，它的标量类型为 str，
        // 我们允许它，因为它不参与 DType 推断，因此不添加到 pytype 到 type 映射中
        return 0;
    }
    else if (res) {
        // 如果字典中已经存在相同的 pytype 映射，抛出运行时错误
        PyErr_SetString(PyExc_RuntimeError,
                "Can only map one python type to DType.");
        return -1;
    }

    // 将 Dtype_obj 添加到 _global_pytype_to_type_dict 中，关联到 pytype
    return PyDict_SetItem(_global_pytype_to_type_dict,
            (PyObject *)pytype, Dtype_obj);
/**
 * Lookup the DType for a registered known python scalar type.
 *
 * @param pytype Python Type to look up
 * @return DType, None if it is a known non-scalar, or NULL if an unknown object.
 */
static inline PyArray_DTypeMeta *
npy_discover_dtype_from_pytype(PyTypeObject *pytype)
{
    PyObject *DType;

    if (pytype == &PyArray_Type) {
        DType = Py_None;
    }
    else if (pytype == &PyFloat_Type) {
        DType = (PyObject *)&PyArray_PyFloatDType;
    }
    else if (pytype == &PyLong_Type) {
        DType = (PyObject *)&PyArray_PyLongDType;
    }
    else {
        DType = PyDict_GetItem(_global_pytype_to_type_dict,
                               (PyObject *)pytype);

        if (DType == NULL) {
            /* the python type is not known */
            return NULL;
        }
    }
    Py_INCREF(DType);
    assert(DType == Py_None || PyObject_TypeCheck(DType, (PyTypeObject *)&PyArrayDTypeMeta_Type));
    return (PyArray_DTypeMeta *)DType;
}

/*
 * Note: This function never fails, but will return `NULL` for unknown scalars
 *       and `None` for known array-likes (e.g. tuple, list, ndarray).
 */
NPY_NO_EXPORT PyObject *
PyArray_DiscoverDTypeFromScalarType(PyTypeObject *pytype)
{
    return (PyObject *)npy_discover_dtype_from_pytype(pytype);
}


/**
 * Find the correct DType class for the given python type. If flags is NULL
 * this is not used to discover a dtype, but only for conversion to an
 * existing dtype. In that case the Python (not NumPy) scalar subclass
 * checks are skipped.
 *
 * @param obj The python object, mainly type(pyobj) is used, the object
 *        is passed to reuse existing code at this time only.
 * @param flags Flags used to know if warnings were already given. If
 *        flags is NULL, this is not
 * @param fixed_DType if not NULL, will be checked first for whether or not
 *        it can/wants to handle the (possible) scalar value.
 * @return New reference to either a DType class, Py_None, or NULL on error.
 */
static inline PyArray_DTypeMeta *
discover_dtype_from_pyobject(
        PyObject *obj, enum _dtype_discovery_flags *flags,
        PyArray_DTypeMeta *fixed_DType)
{
    if (fixed_DType != NULL) {
        /*
         * Let the given DType handle the discovery.  This is when the
         * scalar-type matches exactly, or the DType signals that it can
         * handle the scalar-type.  (Even if it cannot handle here it may be
         * asked to attempt to do so later, if no other matching DType exists.)
         */
        if ((Py_TYPE(obj) == fixed_DType->scalar_type) ||
                NPY_DT_CALL_is_known_scalar_type(fixed_DType, Py_TYPE(obj))) {
            Py_INCREF(fixed_DType);
            return fixed_DType;
        }
    }

    PyArray_DTypeMeta *DType = npy_discover_dtype_from_pytype(Py_TYPE(obj));
    if (DType != NULL) {
        return DType;
    }
    /*
     * 现在我们还没有找到清晰的映射，但主要是为了向后兼容性，我们必须进一步尝试
     * 将输入解释为已知的标量类型之一。
     */
    PyArray_Descr *legacy_descr;
    // 检查对象是否为标量对象，如果是则获取其描述符
    if (PyArray_IsScalar(obj, Generic)) {
        legacy_descr = PyArray_DescrFromScalar(obj);
        // 如果获取描述符失败，则返回空指针
        if (legacy_descr == NULL) {
            return NULL;
        }
    }
    // 如果标志为空，则返回空类型对象
    else if (flags == NULL) {
        Py_INCREF(Py_None);
        return (PyArray_DTypeMeta *)Py_None;
    }
    // 如果对象为字节串，则获取字节类型描述符
    else if (PyBytes_Check(obj)) {
        legacy_descr = PyArray_DescrFromType(NPY_BYTE);
    }
    // 如果对象为 Unicode 字符串，则获取 Unicode 类型描述符
    else if (PyUnicode_Check(obj)) {
        legacy_descr = PyArray_DescrFromType(NPY_UNICODE);
    }
    // 否则调用 _array_find_python_scalar_type 函数尝试寻找 Python 标量类型
    else {
        legacy_descr = _array_find_python_scalar_type(obj);
    }

    // 如果成功获取描述符，则返回描述符并增加引用计数
    if (legacy_descr != NULL) {
        DType = NPY_DTYPE(legacy_descr);
        Py_INCREF(DType);
        Py_DECREF(legacy_descr);
        /* TODO: 启用关于子类处理的警告 */
        // 如果未曾发出过子类警告，则发出警告并设置标志
        if ((0) && !((*flags) & GAVE_SUBCLASS_WARNING)) {
            if (DEPRECATE_FUTUREWARNING(
                    "in the future NumPy will not automatically find the "
                    "dtype for subclasses of scalars known to NumPy (i.e. "
                    "python types). Use the appropriate `dtype=...` to create "
                    "this array. This will use the `object` dtype or raise "
                    "an error in the future.") < 0) {
                return NULL;
            }
            *flags |= GAVE_SUBCLASS_WARNING;
        }
        // 返回数据类型对象
        return DType;
    }
    // 如果没有匹配的描述符，则返回空类型对象
    Py_INCREF(Py_None);
    return (PyArray_DTypeMeta *)Py_None;
/**
 * Discover the correct descriptor from a known DType class and scalar.
 * If the fixed DType can discover a dtype instance/descr all is fine,
 * if it cannot and DType is used instead, a cast will have to be tried.
 *
 * @param fixed_DType A user provided fixed DType, can be NULL
 * @param DType A discovered DType (by discover_dtype_from_pyobject);
 *        this can be identical to `fixed_DType`, if it obj is a
 *        known scalar. Can be `NULL` indicating no known type.
 * @param obj The Python scalar object. At the time of calling this function
 *        it must be known that `obj` should represent a scalar.
 */
static inline PyArray_Descr *
find_scalar_descriptor(
        PyArray_DTypeMeta *fixed_DType, PyArray_DTypeMeta *DType,
        PyObject *obj)
{
    PyArray_Descr *descr;

    if (DType == NULL && fixed_DType == NULL) {
        /* No known DType and no fixed one means we go to object. */
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    else if (DType == NULL) {
        /*
         * If no DType is known/found, give the fixed give one a second
         * chance.  This allows for example string, to call `str(obj)` to
         * figure out the length for arbitrary objects.
         */
        descr = NPY_DT_CALL_discover_descr_from_pyobject(fixed_DType, obj);
    }
    else {
        descr = NPY_DT_CALL_discover_descr_from_pyobject(DType, obj);
    }
    if (descr == NULL) {
        return NULL;
    }
    if (fixed_DType == NULL) {
        return descr;
    }

    Py_SETREF(descr, PyArray_CastDescrToDType(descr, fixed_DType));
    return descr;
}


/**
 * Helper function for casting a raw value from one descriptor to another.
 * This helper uses the normal casting machinery, but e.g. does not care about
 * checking cast safety.
 *
 * @param from_descr Descriptor of the source data.
 * @param from_item Pointer to the raw data to be casted.
 * @param to_descr Descriptor of the target data.
 * @param to_item Pointer to the location where the casted data will be stored.
 * @return Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
npy_cast_raw_scalar_item(
        PyArray_Descr *from_descr, char *from_item,
        PyArray_Descr *to_descr, char *to_item)
{
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    
    // Obtain the casting information and method flags for the data transfer.
    if (PyArray_GetDTypeTransferFunction(
            0, 0, 0, from_descr, to_descr, 0, &cast_info,
            &flags) == NPY_FAIL) {
        return -1;
    }

    // Clear any floating-point errors before casting if needed.
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(from_item);
    }

    // Set up arguments and perform the cast operation.
    char *args[2] = {from_item, to_item};
    const npy_intp strides[2] = {0, 0};
    const npy_intp length = 1;
    if (cast_info.func(&cast_info.context,
            args, &length, strides, cast_info.auxdata) < 0) {
        NPY_cast_info_xfree(&cast_info);
        return -1;
    }
    NPY_cast_info_xfree(&cast_info);

    // Check and handle floating-point errors after casting if necessary.
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(to_item);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    return 0;
}
/*
 * Assign a single element in an array from a python value.
 *
 * The dtypes SETITEM should only be trusted to generally do the right
 * thing if something is known to be a scalar *and* is of a python type known
 * to the DType (which should include all basic Python math types), but in
 * general a cast may be necessary.
 * This function handles the cast, which is for example hit when assigning
 * a float128 to complex128.
 *
 * TODO: This function probably needs to be passed an "owner" for the sake of
 *       future HPy (non CPython) support
 *
 * NOTE: We do support 0-D exact NumPy arrays correctly via casting here.
 *       There be dragons, because we must NOT support generic array-likes.
 *       The problem is that some (e.g. astropy's Quantity and our masked
 *       arrays) have divergent behaviour for `__array__` as opposed to
 *       `__float__`.  And they rely on that.
 *       That is arguably bad as it limits the things that work seamlessly
 *       because `__float__`, etc. cannot even begin to cover all of casting.
 *       However, we have no choice.  We simply CANNOT support array-likes
 *       here without finding a solution for this first.
 *       And the only plausible one I see currently, is expanding protocols
 *       in some form, either to indicate that we want a scalar or to indicate
 *       that we want the unsafe version that `__array__` currently gives
 *       for both objects.
 *
 *       If we ever figure out how to expand this to other array-likes, care
 *       may need to be taken. `PyArray_FromAny`/`PyArray_AssignFromCache`
 *       uses this function but know if the input is an array, array-like,
 *       or scalar.  Relaxing things here should be OK, but looks a bit
 *       like possible recursion, so it may make sense to make a "scalars only"
 *       version of this function.
 *
 * @param descr The descriptor describing the data type of the array.
 * @param item Pointer to the location where the value should be assigned.
 * @param value Python object representing the value to be assigned.
 * @return 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_Pack(PyArray_Descr *descr, void *item, PyObject *value)
{
    /* Initialize fields for the PyArrayObject */
    PyArrayObject_fields arr_fields = {
        .flags = NPY_ARRAY_WRITEABLE,  /* assume array is not behaved. */
    };
    
    /* Set the type of the PyArrayObject to PyArray_Type */
    Py_SET_TYPE(&arr_fields, &PyArray_Type);
    
    /* Set the reference count of the PyArrayObject to 1 */
    Py_SET_REFCNT(&arr_fields, 1);
    
    /* Check if the dtype is NPY_OBJECT */
    if (NPY_UNLIKELY(descr->type_num == NPY_OBJECT)) {
        /*
         * For object dtype, store objects directly; casting may lose type information.
         * For other dtypes, discard the type information.
         * TODO: This path might be necessary for Categorical[object].
         */
        arr_fields.descr = descr;
        return PyDataType_GetArrFuncs(descr)->setitem(value, item, &arr_fields);
    }
    
    /*
     * Call discover_dtype_from_pyobject to determine the dtype meta for the value.
     * This includes a check for is_known_scalar_type.
     */
    PyArray_DTypeMeta *DType = discover_dtype_from_pyobject(
            value, NULL, NPY_DTYPE(descr));
    if (DType == NULL) {
        return -1;
    }
    if (DType == (PyArray_DTypeMeta *)Py_None && PyArray_CheckExact(value)
            && PyArray_NDIM((PyArrayObject *)value) == 0) {
        /*
         * 警告：不要放宽上述 `PyArray_CheckExact` 的条件，除非你
         *       仔细阅读了函数文档中的注意事项并理解了它。
         *
         * 注意：`ndim == 0` 的检查可能应该是一个错误，但是不幸的是。
         *       `arr.__float__()` 对于一个元素的数组有效，因此在某些情况下，
         *       我们需要像处理标量一样处理它。
         *       （如果我们成功弃用上述功能，我们可以这样做。）
         */
        Py_DECREF(DType);

        PyArrayObject *arr = (PyArrayObject *)value;
        if (PyArray_DESCR(arr) == descr && !PyDataType_REFCHK(descr)) {
            /* 当描述符显然匹配时的轻量级快速路径 */
            memcpy(item, PyArray_BYTES(arr), descr->elsize);
            return 0;  /* 成功（它是类似数组的结构） */
        }
        return npy_cast_raw_scalar_item(
                PyArray_DESCR(arr), PyArray_BYTES(arr), descr, item);

    }
    if (DType == NPY_DTYPE(descr) || DType == (PyArray_DTypeMeta *)Py_None) {
        /* 我们可以直接设置元素（或至少尝试） */
        Py_XDECREF(DType);
        arr_fields.descr = descr;
        return PyDataType_GetArrFuncs(descr)->setitem(value, item, &arr_fields);
    }
    PyArray_Descr *tmp_descr;
    tmp_descr = NPY_DT_CALL_discover_descr_from_pyobject(DType, value);
    Py_DECREF(DType);
    if (tmp_descr == NULL) {
        return -1;
    }

    char *data = PyObject_Malloc(tmp_descr->elsize);
    if (data == NULL) {
        PyErr_NoMemory();
        Py_DECREF(tmp_descr);
        return -1;
    }
    if (PyDataType_FLAGCHK(tmp_descr, NPY_NEEDS_INIT)) {
        memset(data, 0, tmp_descr->elsize);
    }
    arr_fields.descr = tmp_descr;
    if (PyDataType_GetArrFuncs(tmp_descr)->setitem(value, data, &arr_fields) < 0) {
        PyObject_Free(data);
        Py_DECREF(tmp_descr);
        return -1;
    }
    int res = npy_cast_raw_scalar_item(tmp_descr, data, descr, item);

    if (PyDataType_REFCHK(tmp_descr)) {
        if (PyArray_ClearBuffer(tmp_descr, data, 0, 1, 1) < 0) {
            res = -1;
        }
    }

    PyObject_Free(data);
    Py_DECREF(tmp_descr);
    return res;
}

/*
 * 更新数组形状信息。
 *
 * @param curr_ndim 当前数组的维度
 * @param max_ndim 允许的最大维度
 * @param out_shape 输出数组的形状
 * @param new_ndim 新增加的维度数
 * @param new_shape 新的形状
 * @param sequence 是否为序列
 * @param flags 数据类型发现标志位
 * @return 成功与否的标志，如果数组不规则返回-1
 */
static int
update_shape(int curr_ndim, int *max_ndim,
             npy_intp out_shape[], int new_ndim,
             const npy_intp new_shape[], npy_bool sequence,
             enum _dtype_discovery_flags *flags)
{
    int success = 0;  /* 如果数组不规则，则操作不成功 */
    const npy_bool max_dims_reached = *flags & MAX_DIMS_WAS_REACHED;

    if (curr_ndim + new_ndim > *max_ndim) {
        success = -1;
        /* 只更新/检查尽可能多的维度，max_ndim 不改变 */
        new_ndim = *max_ndim - curr_ndim;
    }
    else if (!sequence && (*max_ndim != curr_ndim + new_ndim)) {
        /*
         * 对于序列，不更新 max_ndim，否则缩减并检查。
         * 这是深度优先，因此如果已经设置，out_shape 将填充。
         */
        *max_ndim = curr_ndim + new_ndim;
        /* 如果已经设置了形状，则数组也不规则 */
        if (max_dims_reached) {
            success = -1;
        }
    }
    for (int i = 0; i < new_ndim; i++) {
        npy_intp curr_dim = out_shape[curr_ndim + i];
        npy_intp new_dim = new_shape[i];

        if (!max_dims_reached) {
            out_shape[curr_ndim + i] = new_dim;
        }
        else if (new_dim != curr_dim) {
            /* 数组不规则，并且此维度已经不可用 */
            success = -1;
            if (!sequence) {
                /* 移除我们不能使用的维度： */
                *max_ndim -= new_ndim - i;
            }
            else {
                assert(i == 0);
                /* 对于序列，max_ndim 通常不会更新，因此现在设置为： */
                *max_ndim = curr_ndim;
            }
            break;
        }
    }
    if (!sequence) {
        *flags |= MAX_DIMS_WAS_REACHED;
    }
    return success;
}

#ifndef Py_GIL_DISABLED
#define COERCION_CACHE_CACHE_SIZE 5
static int _coercion_cache_num = 0;
static coercion_cache_obj *_coercion_cache_cache[COERCION_CACHE_CACHE_SIZE];
#else
#define COERCION_CACHE_CACHE_SIZE 0
#endif

/*
 * 偷取对对象的引用。
 */
static inline int
npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr, int ndim)
{
    coercion_cache_obj *cache;
#if COERCION_CACHE_CACHE_SIZE > 0
    if (_coercion_cache_num > 0) {
        _coercion_cache_num--;
        cache = _coercion_cache_cache[_coercion_cache_num];
    }
    else
#endif
    {
        cache = PyMem_Malloc(sizeof(coercion_cache_obj));
    }
    if (cache == NULL) {
        Py_DECREF(arr_or_sequence);
        PyErr_NoMemory();
        return -1;
    }
    cache->converted_obj = converted_obj;
    cache->arr_or_sequence = arr_or_sequence;
    cache->sequence = sequence;
    cache->depth = ndim;
    cache->next = NULL;
    **next_ptr = cache;
    *next_ptr = &(cache->next);
    return 0;
}

/**
 * 取消链接强制转换缓存项。
 *
 * @param current 当前的强制转换缓存项
 * @return 下一个强制转换缓存对象（或 NULL）
 */
# 定义一个名为 npy_unlink_coercion_cache 的函数，用于从缓存中移除一个协同缓存对象，并返回下一个对象。
NPY_NO_EXPORT coercion_cache_obj *
npy_unlink_coercion_cache(coercion_cache_obj *current)
{
    // 获取当前对象的下一个对象引用
    coercion_cache_obj *next = current->next;
    // 减少当前对象关联的数组或序列对象的引用计数
    Py_DECREF(current->arr_or_sequence);
#if COERCION_CACHE_CACHE_SIZE > 0
    // 如果缓存大小限制大于零，则执行以下操作
    if (_coercion_cache_num < COERCION_CACHE_CACHE_SIZE) {
        // 将当前对象存储在缓存数组中，并增加缓存计数
        _coercion_cache_cache[_coercion_cache_num] = current;
        _coercion_cache_num++;
    }
    else
#endif
    {
        // 如果缓存已满，则释放当前对象的内存
        PyMem_Free(current);
    }
    // 返回下一个缓存对象
    return next;
}

# 定义一个名为 npy_free_coercion_cache 的函数，用于释放从指定对象开始的所有协同缓存对象。
NPY_NO_EXPORT void
npy_free_coercion_cache(coercion_cache_obj *next) {
    /* We only need to check from the last used cache pos */
    // 只需检查从最后一个使用的缓存位置开始的所有对象
    while (next != NULL) {
        // 调用 npy_unlink_coercion_cache 函数移除当前对象，并更新下一个对象的引用
        next = npy_unlink_coercion_cache(next);
    }
}

# 取消定义 COERCION_CACHE_CACHE_SIZE 符号
#undef COERCION_CACHE_CACHE_SIZE

/**
 * 处理升级和可能的类型转换步骤。此函数不应在需要描述符的情况下调用。在这种情况下，
 * 输出的数据类型不重要，因此我们不能冒险出现升级错误。
 *
 * @param out_descr 当前的描述符。
 * @param descr 要升级的新发现的描述符
 * @param fixed_DType 用户提供的（固定的）DType，如果为 NULL 则不提供
 * @param flags dtype 发现标志，用于信号升级失败。
 * @return -1 表示错误，0 表示成功。
 */
static inline int
handle_promotion(PyArray_Descr **out_descr, PyArray_Descr *descr,
        PyArray_DTypeMeta *fixed_DType, enum _dtype_discovery_flags *flags)
{
    // 断言描述符未设置标志
    assert(!(*flags & DESCRIPTOR_WAS_SET));

    // 如果当前描述符为空，则增加新描述符的引用计数，并将其赋值给当前描述符
    if (*out_descr == NULL) {
        Py_INCREF(descr);
        *out_descr = descr;
        return 0;
    }

    // 否则，尝试将新描述符升级为当前描述符的类型
    PyArray_Descr *new_descr = PyArray_PromoteTypes(descr, *out_descr);

    // 如果升级失败
    if (NPY_UNLIKELY(new_descr == NULL)) {
        // 如果有固定的 DType 或者捕获到 FutureWarning 异常
        if (fixed_DType != NULL || PyErr_ExceptionMatches(PyExc_FutureWarning)) {
            /*
             * 如果 DType 是固定的，升级不应该失败。不要捕获 FutureWarning
             * （用于字符串+数值升级）。这里只能捕获 TypeError 或者总是引发错误。
             */
            return -1;
        }
        PyErr_Clear();
        // 设置升级失败的标志，并继续使用对象类型，因为可能需要维度信息
        *flags |= PROMOTION_FAILED;
        new_descr = PyArray_DescrFromType(NPY_OBJECT);
    }

    // 设置输出描述符为新描述符
    Py_SETREF(*out_descr, new_descr);
    return 0;
}
/**
 * 处理标量对象，确定数组描述符并处理类型提升。
 *
 * @param obj 要处理的 Python 对象。
 * @param curr_dims 当前维度。
 * @param max_dims 最大维度。
 * @param out_descr 输出的数组描述符。
 * @param out_shape 输出的数组形状。
 * @param fixed_DType 固定的数据类型。
 * @param flags 类型发现标志。
 * @param DType 数据类型元信息。
 * @return 处理完成后的最大维度。
 */
handle_scalar(
        PyObject *obj, int curr_dims, int *max_dims,
        PyArray_Descr **out_descr, npy_intp *out_shape,
        PyArray_DTypeMeta *fixed_DType,
        enum _dtype_discovery_flags *flags, PyArray_DTypeMeta *DType)
{
    PyArray_Descr *descr;

    if (update_shape(curr_dims, max_dims, out_shape,
            0, NULL, NPY_FALSE, flags) < 0) {
        *flags |= FOUND_RAGGED_ARRAY;
        return *max_dims;
    }
    if (*flags & DESCRIPTOR_WAS_SET) {
        /* 不需要进行任何类型提升 */
        return *max_dims;
    }
    /* 这是一个标量，因此找到其描述符 */
    descr = find_scalar_descriptor(fixed_DType, DType, obj);
    if (descr == NULL) {
        return -1;
    }
    if (handle_promotion(out_descr, descr, fixed_DType, flags) < 0) {
        Py_DECREF(descr);
        return -1;
    }
    Py_DECREF(descr);
    return *max_dims;
}


/**
 * 根据数组对象和数据类型类找到正确的描述符。
 *
 * 这相当于将数组的描述符/数据类型转换为新的数据类型类。
 *
 * @param arr 数组对象。
 * @param DType 要转换到的数据类型类（为 NULL 表示方便起见）。
 * @param out_descr 将设置的输出描述符。如果数组是对象类型且没有元素，则结果可以为 NULL。
 * @return 失败返回 -1，成功返回 0。
 */
static int
find_descriptor_from_array(
        PyArrayObject *arr, PyArray_DTypeMeta *DType, PyArray_Descr **out_descr)
{
    enum _dtype_discovery_flags flags = 0;
    *out_descr = NULL;

    if (DType == NULL) {
        *out_descr = PyArray_DESCR(arr);
        Py_INCREF(*out_descr);
        return 0;
    }
    /* 其他情况的处理逻辑在这里 */
}
    if (NPY_UNLIKELY(NPY_DT_is_parametric(DType) && PyArray_ISOBJECT(arr))) {
        /*
         * We have one special case, if (and only if) the input array is of
         * object DType and the dtype is not fixed already but parametric.
         * Then, we allow inspection of all elements, treating them as
         * elements. We do this recursively, so nested 0-D arrays can work,
         * but nested higher dimensional arrays will lead to an error.
         */
        assert(DType->type_num != NPY_OBJECT);  /* not parametric */

        PyArrayIterObject *iter;
        // 创建一个新的迭代器对象，用于遍历数组元素
        iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
        if (iter == NULL) {
            return -1;
        }
        while (iter->index < iter->size) {
            PyArray_DTypeMeta *item_DType;
            /*
             * Note: If the array contains typed objects we may need to use
             *       the dtype to use casting for finding the correct instance.
             */
            // 获取当前迭代指针位置的元素对象
            PyObject *elem = PyArray_GETITEM(arr, iter->dataptr);
            if (elem == NULL) {
                Py_DECREF(iter);
                return -1;
            }
            // 探索元素对象的数据类型
            item_DType = discover_dtype_from_pyobject(elem, &flags, DType);
            if (item_DType == NULL) {
                Py_DECREF(iter);
                Py_DECREF(elem);
                return -1;
            }
            if (item_DType == (PyArray_DTypeMeta *)Py_None) {
                Py_SETREF(item_DType, NULL);
            }
            int flat_max_dims = 0;
            // 处理标量元素，进行数据类型转换和标量处理
            if (handle_scalar(elem, 0, &flat_max_dims, out_descr,
                    NULL, DType, &flags, item_DType) < 0) {
                Py_DECREF(iter);
                Py_DECREF(elem);
                Py_XDECREF(*out_descr);
                Py_XDECREF(item_DType);
                return -1;
            }
            Py_XDECREF(item_DType);
            Py_DECREF(elem);
            PyArray_ITER_NEXT(iter);  // 移动迭代器到下一个元素位置
        }
        Py_DECREF(iter);  // 释放迭代器对象
    }
    else if (NPY_UNLIKELY(DType->type_num == NPY_DATETIME) &&
                PyArray_ISSTRING(arr)) {
        /*
         * TODO: This branch should be deprecated IMO, the workaround is
         *       to cast to the object to a string array. Although a specific
         *       function (if there is even any need) would be better.
         *       This is value based casting!
         * Unless of course we actually want to support this kind of thing
         * in general (not just for object dtype)...
         */
        // 处理特殊情况：如果数组的数据类型为日期时间类型，而数组本身为字符串数组
        PyArray_DatetimeMetaData meta;
        meta.base = NPY_FR_GENERIC;
        meta.num = 1;

        // 尝试找到字符串数组的 datetime64 类型
        if (find_string_array_datetime64_type(arr, &meta) < 0) {
            return -1;
        }
        else {
            // 根据找到的 datetime64 类型创建日期时间数据类型描述符
            *out_descr = create_datetime_dtype(NPY_DATETIME, &meta);
            if (*out_descr == NULL) {
                return -1;
            }
        }
    }
    else {
        /*
         * 如果不是对象数组，则确定数据类型的转换，或者直接使用返回的 DType。
         */
        // 将数组的描述符转换为指定的 DType 类型
        *out_descr = PyArray_CastDescrToDType(PyArray_DESCR(arr), DType);
        // 如果转换失败，则返回错误码 -1
        if (*out_descr == NULL) {
            return -1;
        }
    }
    // 返回成功码 0
    return 0;
/**
 * Given a dtype or DType object, find the correct descriptor to cast the
 * array to.  In some places, this function is used with dtype=NULL which
 * means that legacy behavior is used: The dtype instances "S0", "U0", and
 * "V0" are converted to mean the DType classes instead.
 * When dtype != NULL, this path is ignored, and the function does nothing
 * unless descr == NULL. If both descr and dtype are null, it returns the
 * descriptor for the array.
 *
 * This function is identical to normal casting using only the dtype, however,
 * it supports inspecting the elements when the array has object dtype
 * (and the given datatype describes a parametric DType class).
 *
 * @param arr PyArrayObject* representing the input array
 * @param dtype NULL or a dtype class
 * @param descr A dtype instance, if the dtype is NULL the dtype class is
 *              found and e.g. "S0" is converted to denote only String.
 * @return A concrete dtype instance or NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptDescriptorToArray(
        PyArrayObject *arr, PyArray_DTypeMeta *dtype, PyArray_Descr *descr)
{
    /* If the requested dtype is flexible, adapt it */
    PyArray_Descr *new_descr;
    int res;

    // If both dtype and descr are provided, return descr directly
    if (dtype != NULL && descr != NULL) {
        /* descr was given and no special logic, return (call not necessary) */
        Py_INCREF(descr);
        return descr;
    }

    // If dtype is NULL, extract dtype and descriptor
    if (dtype == NULL) {
        // Extract dtype and descriptor
        res = PyArray_ExtractDTypeAndDescriptor(descr, &new_descr, &dtype);
        if (res < 0) {
            return NULL;
        }
        // If new_descr is found, return it and release dtype
        if (new_descr != NULL) {
            Py_DECREF(dtype);
            return new_descr;
        }
    }
    else {
        // If dtype is not NULL and descr is NULL, just return dtype
        assert(descr == NULL);  /* gueranteed above */
        Py_INCREF(dtype);
    }

    // Find descriptor from array and update new_descr
    res = find_descriptor_from_array(arr, dtype, &new_descr);
    if (res < 0) {
        Py_DECREF(dtype);
        return NULL;
    }
    // If new_descr is still NULL, handle object array case
    if (new_descr == NULL) {
        /* This is an object array but contained no elements, use default */
        new_descr = NPY_DT_CALL_default_descr(dtype);
    }
    // Release dtype and return new_descr
    Py_XDECREF(dtype);
    return new_descr;
}
/**
 * Recursion helper for `PyArray_DiscoverDTypeAndShape`.  See its
 * documentation for additional details.
 *
 * @param obj The current (possibly nested) object
 *            当前（可能是嵌套的）对象
 * @param curr_dims The current depth, i.e. initially 0 and increasing.
 *                  当前深度，初始为0，递增。
 * @param max_dims Maximum number of dimensions, modified during discovery.
 *                 最大维度数，在发现过程中被修改。
 * @param out_descr dtype instance (or NULL) to promoted and update.
 *                  用于推广和更新的 dtype 实例（或空）
 * @param out_shape The current shape (updated)
 *                  当前形状（已更新）
 * @param coercion_cache_tail_ptr The tail of the linked list of coercion
 *                                cache objects, which hold on to converted sequences and arrays.
 *                                用于强制转换缓存对象链表的尾部，保存转换的序列和数组。
 * @param fixed_DType User provided fixed DType class
 *                    用户提供的固定 DType 类
 * @param flags Discovery flags (reporting and behaviour flags, see def.)
 *              发现标志（报告和行为标志，详见定义）
 * @param copy Specifies the copy behavior. -1 is corresponds to copy=None,
 *             0 to copy=False, and 1 to copy=True in the Python API.
 *             指定复制行为。-1 对应于 Python API 中的 copy=None，
 *             0 对应于 copy=False，1 对应于 copy=True。
 * @return The updated number of maximum dimensions (i.e. scalars will set
 *         this to the current dimensions).
 *         更新后的最大维度数（即标量将其设置为当前维度）。
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape_Recursive(
        PyObject *obj, int curr_dims, int max_dims, PyArray_Descr**out_descr,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj ***coercion_cache_tail_ptr,
        PyArray_DTypeMeta *fixed_DType, enum _dtype_discovery_flags *flags,
        int copy)
{
    PyArrayObject *arr = NULL;
    PyObject *seq;

    /*
     * The first step is to find the DType class if it was not provided,
     * alternatively we have to find out that this is not a scalar at all
     * (which could fail and lead us to `object` dtype).
     * 
     * 首先是如果没有提供 DType 类，则找到对应的 DType 类，
     * 或者我们必须确认这根本不是标量（可能会失败并导致使用 `object` dtype）。
     */
    PyArray_DTypeMeta *DType = NULL;

    if (NPY_UNLIKELY(*flags & DISCOVER_STRINGS_AS_SEQUENCES)) {
        /*
         * We currently support that bytes/strings are considered sequences,
         * if the dtype is np.dtype('c'), this should be deprecated probably,
         * but requires hacks right now.
         * 
         * 我们目前支持将字节/字符串视为序列，
         * 如果 dtype 是 np.dtype('c')，这可能应该被弃用，但目前需要一些技巧。
         */
        if (PyBytes_Check(obj) && PyBytes_Size(obj) != 1) {
            goto force_sequence_due_to_char_dtype;
        }
        else if (PyUnicode_Check(obj) && PyUnicode_GetLength(obj) != 1) {
            goto force_sequence_due_to_char_dtype;
        }
    }

    /* If this is a known scalar, find the corresponding DType class */
    // 如果这是已知的标量，则找到对应的 DType 类
    DType = discover_dtype_from_pyobject(obj, flags, fixed_DType);
    if (DType == NULL) {
        return -1;
    }
    else if (DType == (PyArray_DTypeMeta *)Py_None) {
        Py_DECREF(Py_None);
    }
    else {
        max_dims = handle_scalar(
                obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                flags, DType);
        Py_DECREF(DType);
        return max_dims;
    }

force_sequence_due_to_char_dtype:
    // 强制将字符 dtype 视为序列的情况
    // (此处可能包含特定的处理逻辑，但在注释中未提及具体实现细节)
    return -1;  // 返回错误状态
}
    /*
     * 此时我们期望找到一个序列或类似数组的对象。
     * 尽管仍然可能失败并且需要使用 `object`。
     */
    if (PyArray_Check(obj)) {
        // 如果 obj 是一个 NumPy 数组，则直接使用它
        arr = (PyArrayObject *)obj;
        // 增加数组对象的引用计数，确保不会被释放
        Py_INCREF(arr);
    }
    else {
        // 如果不是 NumPy 数组，则尝试从类似数组对象创建一个数组对象
        PyArray_Descr *requested_descr = NULL;
        if (*flags & DESCRIPTOR_WAS_SET) {
            /* 如果已经设置了描述符，__array__ 可能会使用请求的描述符 */
            requested_descr = *out_descr;
        }
        int was_copied_by__array__ = 0;
        // 通过 _array_from_array_like 函数创建数组对象
        arr = (PyArrayObject *)_array_from_array_like(obj,
                requested_descr, 0, NULL, copy, &was_copied_by__array__);
        if (arr == NULL) {
            return -1;  // 创建数组对象失败，返回错误
        }
        else if (arr == (PyArrayObject *)Py_NotImplemented) {
            Py_DECREF(arr);
            arr = NULL;
        }
        if (was_copied_by__array__ == 1) {
            *flags |= COPY_WAS_CREATED_BY__ARRAY__;
        }
    }
    if (arr != NULL) {
        /*
         * 这是一个数组对象，将被添加到缓存中，保持数组的引用（拥有所有权）。
         */
        // 将数组对象添加到类型强制缓存中
        if (npy_new_coercion_cache(obj, (PyObject *)arr,
                0, coercion_cache_tail_ptr, curr_dims) < 0) {
            return -1;  // 添加缓存失败，返回错误
        }

        if (curr_dims == 0) {
            /*
             * 对于反向广播的特殊情况，如果只有一个类似数组对象，则忽略 max_dims；
             * 这对 PyArray_CopyObject 是必需的。
             */
            // 复制数组对象的形状到 out_shape，设置 max_dims
            memcpy(out_shape, PyArray_SHAPE(arr),
                   PyArray_NDIM(arr) * sizeof(npy_intp));
            max_dims = PyArray_NDIM(arr);
        }
        else if (update_shape(curr_dims, &max_dims, out_shape,
                PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FALSE, flags) < 0) {
            *flags |= FOUND_RAGGED_ARRAY;
            return max_dims;  // 更新形状失败，返回 max_dims
        }

        if (*flags & DESCRIPTOR_WAS_SET) {
            return max_dims;  // 如果已设置描述符，则返回 max_dims
        }
        /*
         * 对于数组，我们可能不仅需要将 dtype 转换为用户提供的 fixed_DType；
         * 如果这是一个对象数组，则可能需要逐个检查元素。
         * 注意，首先找到数组的描述符，然后再提升（不同的结合性）。
         */
        // 查找数组的描述符，并根据需要进行提升
        PyArray_Descr *cast_descr;
        if (find_descriptor_from_array(arr, fixed_DType, &cast_descr) < 0) {
            return -1;  // 查找描述符失败，返回错误
        }
        if (cast_descr == NULL) {
            /* 对象数组没有元素，无需提升/调整。 */
            return max_dims;  // 如果 cast_descr 为 NULL，返回 max_dims
        }
        if (handle_promotion(out_descr, cast_descr, fixed_DType, flags) < 0) {
            Py_DECREF(cast_descr);
            return -1;  // 处理提升失败，返回错误
        }
        Py_DECREF(cast_descr);
        return max_dims;  // 成功处理提升，返回 max_dims
    }
    /*
     * 首先检查对象是否被视为序列，并递归处理。除非达到了维度限制。
     */
    npy_bool is_sequence = PySequence_Check(obj);
    // 如果对象是序列
    if (is_sequence) {
        // 检查序列的大小是否大于等于0
        is_sequence = PySequence_Size(obj) >= 0;
        // 如果不是序列，处理错误
        if (NPY_UNLIKELY(!is_sequence)) {
            /* 注意：这里可能只是引发所有的错误 */
            // 如果引发了递归错误或内存错误，认为是无法恢复的错误，终止执行
            if (PyErr_ExceptionMatches(PyExc_RecursionError) ||
                    PyErr_ExceptionMatches(PyExc_MemoryError)) {
                /*
                 * 这些是无法恢复的错误，继续执行可能导致解释器崩溃。
                 */
                return -1;
            }
            // 清除异常状态
            PyErr_Clear();
        }
    }
    // 如果禁用了元组作为元素的发现，并且对象是元组，则不视为序列
    if (NPY_UNLIKELY(*flags & DISCOVER_TUPLES_AS_ELEMENTS) &&
            PyTuple_Check(obj)) {
        is_sequence = NPY_FALSE;
    }
    // 如果当前维度已达到最大维度或对象不是序列，则处理为标量
    if (curr_dims == max_dims || !is_sequence) {
        /* 清除任何可能导致后续调用出错的 PySequence_Size 错误 */
        // 处理对象为标量，更新相关参数
        max_dims = handle_scalar(
                obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                flags, NULL);
        // 如果对象是序列，则标记为不规则数组或过深的数组
        if (is_sequence) {
            *flags |= FOUND_RAGGED_ARRAY;
        }
        return max_dims;
    }
    /* 如果我们停止支持 bytes/str 的子类，这里可能需要更多处理： */
    // 确保对象不是 bytes 或 str 的子类
    assert(!PyBytes_Check(obj) && !PyUnicode_Check(obj));

  force_sequence_due_to_char_dtype:

    /*
     * 确保我们有一个序列（对于 PyPy 是必需的）
     */
    // 尝试将对象转换为快速序列，如果失败，则处理特定的错误
    seq = PySequence_Fast(obj, "Could not convert object to sequence");
    if (seq == NULL) {
        /*
         * 特别处理类似于字典的对象，视为标量处理
         */
        // 如果是 KeyError 不报错，而是视为标量处理
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            // 处理对象为标量，更新相关参数
            max_dims = handle_scalar(
                    obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                    flags, NULL);
            return max_dims;
        }
        // 其他异常情况，返回错误
        return -1;
    }
    /* 缓存在这里获取序列的所有权 */
    // 将序列对象和原始对象加入类型转换缓存
    if (npy_new_coercion_cache(obj, seq, 1, coercion_cache_tail_ptr, curr_dims) < 0) {
        return -1;
    }

    // 获取序列的大小和对象数组
    npy_intp size = PySequence_Fast_GET_SIZE(seq);
    PyObject **objects = PySequence_Fast_ITEMS(seq);

    // 更新形状，如果是不规则的情况，则进行更新
    if (update_shape(curr_dims, &max_dims,
                     out_shape, 1, &size, NPY_TRUE, flags) < 0) {
        // 如果是不规则的情况，则标记
        *flags |= FOUND_RAGGED_ARRAY;
        return max_dims;
    }
    // 如果序列为空，则表示这是最后一个维度
    if (size == 0) {
        // 如果序列为空，标记已达到最大维度
        *flags |= MAX_DIMS_WAS_REACHED;
        return curr_dims + 1;
    }

    /* 允许处理键盘中断。参见 gh issue 18117。*/
    // 检查是否有键盘中断信号
    if (PyErr_CheckSignals() < 0) {
        return -1;
    }
    /*
     * 对于一个序列，我们无论如何都需要复制最终的聚合结果。
     * 因此不需要显式传递 `copy=True`，所以我们将 `copy=None`（如果需要则复制）。
     */
    如果 (copy == 1) {
        copy = -1;  // 将 copy 设置为 -1，表示需要复制
    }

    /* 递归调用每个序列项 */
    for (Py_ssize_t i = 0; i < size; i++) {
        max_dims = PyArray_DiscoverDTypeAndShape_Recursive(
                objects[i], curr_dims + 1, max_dims,
                out_descr, out_shape, coercion_cache_tail_ptr, fixed_DType,
                flags, copy);

        if (max_dims < 0) {
            return -1;  // 如果递归调用返回负值，直接返回 -1
        }
    }
    return max_dims;  // 返回计算出的最大维度
/**
 * Finds the DType and shape of an arbitrary nested sequence. This is the
 * general purpose function to find the parameters of the array (but not
 * the array itself) as returned by `np.array()`
 *
 * Note: Before considering to make part of this public, we should consider
 *       whether things such as `out_descr != NULL` should be supported in
 *       a public API.
 *
 * @param obj Scalar or nested sequences.
 *            输入参数：标量或嵌套序列。
 * @param max_dims Maximum number of dimensions (after this scalars are forced)
 *                 最大维度数（超过这个数后标量被强制处理）。
 * @param out_shape Will be filled with the output shape (more than the actual
 *                  shape may be written).
 *                  输出参数：将被填充为输出形状（可能写入比实际形状更多的内容）。
 * @param coercion_cache NULL initialized reference to a cache pointer.
 *                       May be set to the first coercion_cache, and has to be freed using
 *                       npy_free_coercion_cache.
 *                       This should be stored in a thread-safe manner (i.e. function static)
 *                       and is designed to be consumed by `PyArray_AssignFromCache`.
 *                       If not consumed, must be freed using `npy_free_coercion_cache`.
 *                       用于缓存指针的空初始化引用。
 *                       可能被设置为第一个 coercion_cache，并且必须使用 npy_free_coercion_cache 进行释放。
 *                       应以线程安全的方式存储（即函数静态），并设计用于 `PyArray_AssignFromCache` 的消耗。
 *                       如果未被消耗，必须使用 `npy_free_coercion_cache` 进行释放。
 * @param fixed_DType A user provided fixed DType class.
 *                    用户提供的固定 DType 类。
 * @param requested_descr A user provided fixed descriptor. This is always
 *                        returned as the discovered descriptor, but currently only used
 *                        for the ``__array__`` protocol.
 *                        用户提供的固定描述符。这总是作为发现的描述符返回，但目前仅用于 ``__array__`` 协议。
 * @param out_descr Set to the discovered output descriptor. This may be
 *                  non NULL but only when fixed_DType/requested_descr are not given.
 *                  If non NULL, it is the first dtype being promoted and used if there
 *                  are no elements.
 *                  The result may be unchanged (remain NULL) when converting a
 *                  sequence with no elements. In this case it is callers responsibility
 *                  to choose a default.
 *                  设置为发现的输出描述符。这可能是非 NULL，但只有在没有给定 fixed_DType/requested_descr 时才是如此。
 *                  如果非 NULL，则它是被提升并用于没有元素时的第一个 dtype。
 *                  当转换一个没有元素的序列时，结果可能不变（保持 NULL）。在这种情况下，由调用者负责选择默认值。
 * @param copy Specifies the copy behavior. -1 is corresponds to copy=None,
 *             0 to copy=False, and 1 to copy=True in the Python API.
 *             指定复制行为。-1 对应于 Python API 中的 copy=None，0 对应于 copy=False，1 对应于 copy=True。
 * @param was_copied_by__array__ Set to 1 if it can be assumed that a copy was
 *                               made by implementor.
 *                               如果可以假定实现者已经进行了复制，则设置为 1。
 * @return dimensions of the discovered object or -1 on error.
 *         WARNING: If (and only if) the output is a single array, the ndim
 *                  returned _can_ exceed the maximum allowed number of dimensions.
 *                  It might be nice to deprecate this? But it allows things such as
 *                  `arr1d[...] = np.array([[1,2,3,4]])`
 *         发现对象的维度或错误时返回 -1。
 *         警告：如果输出是单个数组，则返回的 ndim 可以超过允许的最大维数。
 *              这可能会被弃用？但它允许诸如 `arr1d[...] = np.array([[1,2,3,4]])` 的操作。
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr, int copy, int *was_copied_by__array__)
{
    coercion_cache_obj **coercion_cache_head = coercion_cache;
    *coercion_cache = NULL;
    enum _dtype_discovery_flags flags = 0;

    /*
     * Support a passed in descriptor (but only if nothing was specified).
     */
    assert(*out_descr == NULL || fixed_DType == NULL);
    /* Validate input of requested descriptor and DType */
}
    # 如果 fixed_DType 不为空指针，则进行断言检查其类型是否为 PyArrayDTypeMeta_Type
    if (fixed_DType != NULL) {
        assert(PyObject_TypeCheck(
                (PyObject *)fixed_DType, (PyTypeObject *)&PyArrayDTypeMeta_Type));
    }

    # 如果 requested_descr 不为空指针，则执行以下操作
    if (requested_descr != NULL) {
        # 如果 fixed_DType 不为空指针，则断言 fixed_DType 是否等于 requested_descr 的 NPY_DTYPE
        if (fixed_DType != NULL) {
            assert(fixed_DType == NPY_DTYPE(requested_descr));
        }
        # 将 requested_descr 作为输出描述符，并增加其引用计数
        Py_INCREF(requested_descr);
        *out_descr = requested_descr;
        # 设置标志位，表示描述符已经设置
        flags |= DESCRIPTOR_WAS_SET;
    }

    /*
     * 调用递归函数，可能需要扩展设置以更好地处理缓存。
     */

    /* 遗留发现标志 */
    if (requested_descr != NULL) {
        # 如果 requested_descr 的类型编号为 NPY_STRING，且其类型为 'c'
        if (requested_descr->type_num == NPY_STRING &&
                requested_descr->type == 'c') {
            /* 字符类型的字符串变体（应该已被弃用...） */
            flags |= DISCOVER_STRINGS_AS_SEQUENCES;
        }
        # 如果 requested_descr 的类型编号为 NPY_VOID，并且具有命名字段或子数组
        else if (requested_descr->type_num == NPY_VOID &&
                    (((_PyArray_LegacyDescr *)requested_descr)->names
                     || ((_PyArray_LegacyDescr *)requested_descr)->subarray))  {
            /* Void 类型是一个嵌合体，可能是结构化的也可能不是... */
            flags |= DISCOVER_TUPLES_AS_ELEMENTS;
        }
    }

    # 调用 PyArray_DiscoverDTypeAndShape_Recursive 函数来发现数据类型和形状
    int ndim = PyArray_DiscoverDTypeAndShape_Recursive(
            obj, 0, max_dims, out_descr, out_shape, &coercion_cache,
            fixed_DType, &flags, copy);
    # 如果返回值小于 0，则跳转到失败处理标签
    if (ndim < 0) {
        goto fail;
    }

    # 如果 was_copied_by__array__ 不为空且标志位中包含 COPY_WAS_CREATED_BY__ARRAY__
    if (was_copied_by__array__ != NULL && flags & COPY_WAS_CREATED_BY__ARRAY__) {
        # 设置 was_copied_by__array__ 为 1，表示由 __array__ 创建了拷贝
        *was_copied_by__array__ = 1;
    }
    if (NPY_UNLIKELY(flags & FOUND_RAGGED_ARRAY)) {
        /*
         * 如果标志中包含 FOUND_RAGGED_ARRAY，说明发现了不规则数组。
         * 这可能是由于达到了最大维度并且维度被减少，这种情况被称为不规则数组。
         * 否则，我们仅仅是达到了最大维度，这种情况稍微有所不同。
         * 例如，对于 `[1, [2, 3]]` 这样的数组，最大维度是 1，但是发现了序列。
         *
         * 在这种情况下，我们需要通知用户并清除缓存，因为可能太深了。
         */

        /* 处理达到最大深度的情况： */
        int too_deep = ndim == max_dims;

        if (fixed_DType == NULL || fixed_DType->type_num != NPY_OBJECT) {
            /* 只有对象类型的数据类型支持不规则数组的情况，统一报错 */

            if (!too_deep) {
                PyObject *shape = PyArray_IntTupleFromIntp(ndim, out_shape);
                PyErr_Format(PyExc_ValueError,
                        "setting an array element with a sequence. The "
                        "requested array has an inhomogeneous shape after "
                        "%d dimensions. The detected shape was "
                        "%R + inhomogeneous part.",
                        ndim, shape);
                Py_DECREF(shape);
            }
            else {
                PyErr_Format(PyExc_ValueError,
                        "setting an array element with a sequence. The "
                        "requested array would exceed the maximum number of "
                        "dimension of %d.",
                        max_dims);
            }
            goto fail;
        }

        /*
         * 如果数组是不规则的，缓存可能太深，因此需要清理它。
         * 但是缓存的深度将与数组保持一致。
         */
        coercion_cache_obj **next_ptr = coercion_cache_head;
        coercion_cache_obj *current = *coercion_cache_head;  /* 要检查的项目 */
        while (current != NULL) {
            if (current->depth > ndim) {
                /* 删除 "next" 缓存项并将指针前进（与后续操作不同） */
                current = npy_unlink_coercion_cache(current);
                continue;
            }
            /* 将 prev 和 next 都前进，并将 prev->next 设置为新项 */
            *next_ptr = current;
            next_ptr = &(current->next);
            current = current->next;
        }
        *next_ptr = NULL;
    }
    /* 这里也可以检查是否达到了最大维度 */

    if (requested_descr != NULL) {
        /* 如果提供了描述符，则确保我们没有意外更改它 */
        assert(*out_descr == requested_descr);
    }
    else if (NPY_UNLIKELY(*out_descr == NULL)) {
        /*
         * 如果输出描述符指针指向 NULL，则执行以下代码块：
         * 当对象不包含任何元素（长度为零的序列）时，可能找不到描述符。
         * 如果请求了固定的数据类型（DType），则使用它来定义输出的数据类型。
         * 否则，out_descr 将保持为 NULL，调用者需要设置正确的默认值。
         */
        if (fixed_DType != NULL) {
            // 使用固定的数据类型（DType）调用函数获取默认描述符
            *out_descr = NPY_DT_CALL_default_descr(fixed_DType);
            // 检查描述符是否成功获取，如果为 NULL，则跳转到失败处理标签
            if (*out_descr == NULL) {
                goto fail;
            }
        }
    }
    // 返回数组的维度
    return ndim;

  fail:
    // 释放强制转换缓存
    npy_free_coercion_cache(*coercion_cache_head);
    // 将强制转换缓存头指针设置为 NULL
    *coercion_cache_head = NULL;
    // 将输出描述符设置为 NULL
    Py_XSETREF(*out_descr, NULL);
    // 返回错误码 -1
    return -1;
/*
 * Python API function to expose the dtype+shape discovery functionality
 * directly.
 */
NPY_NO_EXPORT PyObject *
_discover_array_parameters(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;  // 声明一个PyObject类型的变量obj，用于存储传入的Python对象
    npy_dtype_info dt_info = {NULL, NULL};  // 声明并初始化一个npy_dtype_info结构体变量dt_info，用于存储dtype信息
    npy_intp shape[NPY_MAXDIMS];  // 声明一个长度为NPY_MAXDIMS的npy_intp类型数组shape，用于存储数组的形状

    NPY_PREPARE_ARGPARSER;  // 准备解析参数的宏定义，这里将在后续代码中使用

    // 解析传入的参数
    if (npy_parse_arguments(
            "_discover_array_parameters", args, len_args, kwnames,
            "", NULL, &obj,
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            NULL, NULL, NULL) < 0) {
        /* fixed is last to parse, so never necessary to clean up */
        return NULL;  // 解析参数失败时返回NULL
    }

    coercion_cache_obj *coercion_cache = NULL;  // 声明一个coercion_cache_obj类型指针变量coercion_cache，并初始化为NULL
    PyObject *out_dtype = NULL;  // 声明一个PyObject类型指针变量out_dtype，并初始化为NULL
    int ndim = PyArray_DiscoverDTypeAndShape(
            obj, NPY_MAXDIMS, shape,
            &coercion_cache,
            dt_info.dtype, dt_info.descr, (PyArray_Descr **)&out_dtype, 0, NULL);  // 调用PyArray_DiscoverDTypeAndShape函数获取数组的dtype和形状信息
    Py_XDECREF(dt_info.dtype);  // 安全释放dt_info中的dtype成员
    Py_XDECREF(dt_info.descr);  // 安全释放dt_info中的descr成员
    if (ndim < 0) {
        return NULL;  // 获取dtype和形状信息失败时返回NULL
    }
    npy_free_coercion_cache(coercion_cache);  // 释放类型转换缓存对象的内存
    if (out_dtype == NULL) {
        /* Empty sequence, report this as None. */
        out_dtype = Py_None;  // 如果dtype为空，将out_dtype设置为Py_None
        Py_INCREF(Py_None);  // 增加Py_None的引用计数
    }

    PyObject *shape_tuple = PyArray_IntTupleFromIntp(ndim, shape);  // 将ndim和shape数组转换为Python元组对象
    if (shape_tuple == NULL) {
        return NULL;  // 转换为元组失败时返回NULL
    }

    PyObject *res = PyTuple_Pack(2, (PyObject *)out_dtype, shape_tuple);  // 将out_dtype和shape_tuple打包成一个元组对象res
    Py_DECREF(out_dtype);  // 减少out_dtype的引用计数
    Py_DECREF(shape_tuple);  // 减少shape_tuple的引用计数
    return res;  // 返回包含dtype和形状信息的元组对象
}
```