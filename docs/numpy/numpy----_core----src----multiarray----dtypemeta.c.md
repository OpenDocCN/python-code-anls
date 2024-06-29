# `.\numpy\numpy\_core\src\multiarray\dtypemeta.c`

```
/* Array Descr Object */
/* 定义一个宏，设置 NumPy 的 API 版本 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/* 定义一个宏，指示编译器包含多维数组的头文件 */
#define _MULTIARRAYMODULE

/* 清除 PY_SSIZE_T 类型的宏定义，确保使用标准的 ssize_t 类型 */
#define PY_SSIZE_T_CLEAN
/* 包含 Python 核心头文件 */
#include <Python.h>
/* 包含结构成员定义的头文件 */
#include <structmember.h>

/* 包含 NumPy 的数组类型定义 */
#include <numpy/ndarraytypes.h>
/* 包含 NumPy 的数组标量定义 */
#include <numpy/arrayscalars.h>
/* 包含 NumPy 的数学函数定义 */
#include <numpy/npy_math.h>

/* 包含自定义的头文件 */
#include "npy_import.h"

#include "abstractdtypes.h"
#include "arraytypes.h"
#include "common.h"
#include "dtypemeta.h"
#include "descriptor.h"
#include "_datetime.h"
#include "array_coercion.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "usertypes.h"
#include "conversion_utils.h"
#include "templ_common.h"
#include "refcount.h"
#include "dtype_traversal.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"

/* 包含断言处理的标准头文件 */
#include <assert.h>

/* 定义 dtypemeta_dealloc 函数，用于释放 PyArray_DTypeMeta 对象 */
static void
dtypemeta_dealloc(PyArray_DTypeMeta *self) {
    /* 确保不会意外删除静态定义的 DType */
    assert(((PyTypeObject *)self)->tp_flags & Py_TPFLAGS_HEAPTYPE);

    /* 释放 scalar_type 成员的引用 */
    Py_XDECREF(self->scalar_type);
    /* 释放 singleton 成员的引用 */
    Py_XDECREF(self->singleton);
    /* 释放 dt_slots 成员内存 */
    Py_XDECREF(NPY_DT_SLOTS(self)->castingimpls);
    PyMem_Free(self->dt_slots);
    /* 调用 PyType_Type.tp_dealloc 函数释放对象内存 */
    PyType_Type.tp_dealloc((PyObject *) self);
}

/* 定义 dtypemeta_alloc 函数，用于分配 PyArray_DTypeMeta 类型对象 */
static PyObject *
dtypemeta_alloc(PyTypeObject *NPY_UNUSED(type), Py_ssize_t NPY_UNUSED(items))
{
    /* 抛出类型错误，说明 DTypes 只能通过 NumPy API 创建 */
    PyErr_SetString(PyExc_TypeError,
            "DTypes can only be created using the NumPy API.");
    return NULL;
}

/* 定义 dtypemeta_new 函数，用于创建 PyArray_DTypeMeta 类型对象 */
static PyObject *
dtypemeta_new(PyTypeObject *NPY_UNUSED(type),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    /* 抛出类型错误，说明 Preliminary-API 中不能子类化 DType */
    PyErr_SetString(PyExc_TypeError,
            "Preliminary-API: Cannot subclass DType.");
    return NULL;
}

/* 定义 dtypemeta_init 函数，用于初始化 PyArray_DTypeMeta 类型对象 */
static int
dtypemeta_init(PyTypeObject *NPY_UNUSED(type),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    /* 抛出类型错误，说明 Preliminary-API 中不能使用 __init__ 初始化 DType 类 */
    PyErr_SetString(PyExc_TypeError,
            "Preliminary-API: Cannot __init__ DType class.");
    return -1;
}

/* 定义 dtype_does_not_promote 函数，用于处理 DType 不会提升的情况 */
static PyArray_DTypeMeta *
dtype_does_not_promote(
        PyArray_DTypeMeta *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(other))
{
    /* 因为 other 肯定不是 self，所以直接返回 Py_NotImplemented */
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

/* 定义 dtypemeta_discover_as_default 函数，用于发现 PyArray_DTypeMeta 类型的默认描述符 */
NPY_NO_EXPORT PyArray_Descr *
dtypemeta_discover_as_default(
        PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    /* 调用 NPY_DT_CALL_default_descr 函数获取默认描述符 */
    return NPY_DT_CALL_default_descr(cls);
}

/* 定义 use_new_as_default 函数，用于使用自身类型创建默认描述符 */
static PyArray_Descr *
use_new_as_default(PyArray_DTypeMeta *self)
{
    /* 调用 PyObject_CallObject 创建 self 对象的实例 */
    PyObject *res = PyObject_CallObject((PyObject *)self, NULL);
    if (res == NULL) {
        return NULL;
    }
    /* 检查返回值是否为有效的 dtype 实例 */
    if (!PyArray_DescrCheck(res)) {
        /* 如果不是有效的 dtype 实例，抛出运行时错误 */
        PyErr_Format(PyExc_RuntimeError,
                "Instantiating %S did not return a dtype instance, this is "
                "invalid (especially without a custom `default_descr()`).",
                self);
        Py_DECREF(res);
        return NULL;
    }
    # 将 void 指针 res 转换为 PyArray_Descr 指针类型，以便进行后续操作
    PyArray_Descr *descr = (PyArray_Descr *)res;
    
    """
     * 应该在这里对描述符进行更多的健全性检查，
     * 以确保用户没有做出不当行为。但最终，我们对此的控制力有限。
     """
    
    # 返回描述符，这是一个 PyArray_Descr 指针类型的对象
    return descr;
/*
 * legacy_setitem_using_DType:
 *   Function for setting an item using a legacy NumPy DType.
 *   Parameters:
 *     - obj: Python object representing the data to set.
 *     - data: Pointer to the data buffer.
 *     - arr: Pointer to the NumPy array object.
 *   Returns:
 *     - Returns -1 on error, 0 otherwise.
 *   Notes:
 *     - Checks if arr is NULL and raises an exception if so.
 *     - Retrieves the setitem function pointer from the NumPy DType descriptor.
 */
static int
legacy_setitem_using_DType(PyObject *obj, void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return -1;
    }
    PyArrayDTypeMeta_SetItem *setitem;
    setitem = NPY_DT_SLOTS(NPY_DTYPE(PyArray_DESCR(arr)))->setitem;
    return setitem(PyArray_DESCR(arr), obj, data);
}

/*
 * legacy_getitem_using_DType:
 *   Function for getting an item using a legacy NumPy DType.
 *   Parameters:
 *     - data: Pointer to the data buffer.
 *     - arr: Pointer to the NumPy array object.
 *   Returns:
 *     - Returns a Python object on success, NULL on error.
 *   Notes:
 *     - Checks if arr is NULL and raises an exception if so.
 *     - Retrieves the getitem function pointer from the NumPy DType descriptor.
 */
static PyObject *
legacy_getitem_using_DType(void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return NULL;
    }
    PyArrayDTypeMeta_GetItem *getitem;
    getitem = NPY_DT_SLOTS(NPY_DTYPE(PyArray_DESCR(arr)))->getitem;
    return getitem(PyArray_DESCR(arr), data);
}

/*
 * default_funcs:
 *   Default array functions structure.
 *   Notes:
 *     - Contains pointers to legacy getitem and setitem functions.
 */
PyArray_ArrFuncs default_funcs = {
        .getitem = &legacy_getitem_using_DType,
        .setitem = &legacy_setitem_using_DType,
};

/*
 * dtypemeta_initialize_struct_from_spec:
 *   Internal function to initialize a DTypeMeta structure from a specification.
 *   Parameters:
 *     - DType: Pointer to the DTypeMeta structure to initialize.
 *     - spec: Pointer to the specification structure.
 *     - priv: Integer indicating private initialization (non-zero) or public (zero).
 *   Returns:
 *     - Returns -1 on error, 0 otherwise.
 *   Notes:
 *     - Checks if the DType is already registered and raises an exception if so.
 *     - Initializes various fields of the DTypeMeta structure based on the spec.
 *     - Sets default values for DType slots and function pointers.
 */
NPY_NO_EXPORT int
dtypemeta_initialize_struct_from_spec(
        PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec, int priv)
{
    if (DType->dt_slots != NULL) {
        PyErr_Format(PyExc_RuntimeError,
                "DType %R appears already registered?", DType);
        return -1;
    }

    DType->flags = spec->flags;
    DType->dt_slots = PyMem_Calloc(1, sizeof(NPY_DType_Slots));
    if (DType->dt_slots == NULL) {
        return -1;
    }

    /* Set default values (where applicable) */
    NPY_DT_SLOTS(DType)->discover_descr_from_pyobject =
            &dtypemeta_discover_as_default;
    NPY_DT_SLOTS(DType)->is_known_scalar_type = (
            &python_builtins_are_known_scalar_types);
    NPY_DT_SLOTS(DType)->default_descr = use_new_as_default;
    NPY_DT_SLOTS(DType)->common_dtype = dtype_does_not_promote;
    NPY_DT_SLOTS(DType)->common_instance = NULL;
    NPY_DT_SLOTS(DType)->setitem = NULL;
    NPY_DT_SLOTS(DType)->getitem = NULL;
    NPY_DT_SLOTS(DType)->get_clear_loop = NULL;
    NPY_DT_SLOTS(DType)->get_fill_zero_loop = NULL;
    NPY_DT_SLOTS(DType)->finalize_descr = NULL;
    NPY_DT_SLOTS(DType)->f = default_funcs;

    PyType_Slot *spec_slot = spec->slots;
    /* invalid type num. Ideally, we get away with it! */
    DType->type_num = -1;

    /*
     * Handle the scalar type mapping.
     */
    Py_INCREF(spec->typeobj);
}
    # 将 DType 的 scalar_type 设置为 spec 的 typeobj
    DType->scalar_type = spec->typeobj;
    
    # 如果 scalar_type 是堆类型(Py_TPFLAGS_HEAPTYPE)，则设置其 "__associated_array_dtype__" 属性为 DType
    if (PyType_GetFlags(spec->typeobj) & Py_TPFLAGS_HEAPTYPE) {
        if (PyObject_SetAttrString((PyObject *)DType->scalar_type,
                "__associated_array_dtype__", (PyObject *)DType) < 0) {
            Py_DECREF(DType);
            return -1;
        }
    }
    
    # 将 DType 映射到其对应的 Python 类型，返回小于 0 则表示错误
    if (_PyArray_MapPyTypeToDType(DType, DType->scalar_type, 0) < 0) {
        Py_DECREF(DType);
        return -1;
    }

    /* 确保 castingimpls 字典已定义（不确定此处是否需要）*/
    NPY_DT_SLOTS(DType)->castingimpls = PyDict_New();
    if (NPY_DT_SLOTS(DType)->castingimpls == NULL) {
        return -1;
    }

    /*
     * 现在，注册当前定义的所有类型转换！
     */
    PyArrayMethod_Spec **next_meth_spec = spec->casts;
    while (1) {
        PyArrayMethod_Spec *meth_spec = *next_meth_spec;
        next_meth_spec++;
        if (meth_spec == NULL) {
            break;
        }
        
        /*
         * 用户尚不知道 DType 的名称，因此我们必须为他们填写它！
         */
        // 将未定义的 dtypes[i] 填充为 DType
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == NULL) {
                meth_spec->dtypes[i] = DType;
            }
        }
        
        /* 注册转换操作！ */
        // priv 指示这是否是一个内部调用
        int res = PyArray_AddCastingImplementation_FromSpec(meth_spec, priv);

        /* 再次进行清理，以防别人有不良想法... */
        // 将填充过的 dtypes[i] 重置为空
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == DType) {
                meth_spec->dtypes[i] = NULL;
            }
        }

        if (res < 0) {
            return -1;
        }
    }

    return 0;
/**
 * tp_is_gc slot of Python types. This is implemented only for documentation
 * purposes to indicate and document the subtleties involved.
 *
 * Python Type objects are either statically created (typical C-Extension type)
 * or HeapTypes (typically created in Python).
 * HeapTypes have the Py_TPFLAGS_HEAPTYPE flag and are garbage collected.
 * Our DTypeMeta instances (`np.dtype` and its subclasses) *may* be HeapTypes
 * if the Py_TPFLAGS_HEAPTYPE flag is set (they are created from Python).
 * They are not for legacy DTypes or np.dtype itself.
 *
 * @param self The Python type object (DTypeMeta instance) to check
 * @return nonzero if the object is garbage collected
 */
static inline int
dtypemeta_is_gc(PyObject *dtype_class)
{
    // Return the result of checking if the type object is garbage collected
    return PyType_Type.tp_is_gc(dtype_class);
}

/**
 * This function is currently not used, but will probably be necessary
 * in the future when we implement HeapTypes (python/dynamically
 * defined types). It should be revised at that time.
 *
 * @param type The DTypeMeta instance to traverse
 * @param visit The visit function for garbage collection traversal
 * @param arg Custom argument passed to the visit function
 * @return Always asserts and does not return a value
 */
static int
dtypemeta_traverse(PyArray_DTypeMeta *type, visitproc visit, void *arg)
{
    // Assert conditions related to the legacy status and type object
    assert(0);
    assert(!NPY_DT_is_legacy(type) && (PyTypeObject *)type != &PyArrayDescr_Type);
    // Visit specific attributes of the DTypeMeta instance for garbage collection
    Py_VISIT(type->singleton);
    Py_VISIT(type->scalar_type);
    // Use PyType_Type to traverse the DTypeMeta instance
    return PyType_Type.tp_traverse((PyObject *)type, visit, arg);
}

/**
 * Create a new instance of a legacy DTypeMeta object with default settings.
 *
 * @param self The DTypeMeta class instance (Python type object)
 * @param args Tuple of arguments passed to the constructor
 * @param kwargs Dictionary of keyword arguments passed to the constructor
 * @return New instance of the legacy DTypeMeta object or NULL on failure
 */
static PyObject *
legacy_dtype_default_new(PyArray_DTypeMeta *self,
        PyObject *args, PyObject *kwargs)
{
    /* TODO: This should allow endianness and possibly metadata */
    if (NPY_DT_is_parametric(self)) {
        // Raise an error if trying to instantiate a parametric legacy DTypeMeta
        PyErr_Format(PyExc_TypeError,
                "Preliminary-API: Flexible/Parametric legacy DType '%S' can "
                "only be instantiated using `np.dtype(...)`", self);
        return NULL;
    }

    if (PyTuple_GET_SIZE(args) != 0 ||
                (kwargs != NULL && PyDict_Size(kwargs))) {
        // Reject instantiation with arguments other than an empty tuple
        PyErr_Format(PyExc_TypeError,
                "currently only the no-argument instantiation is supported; "
                "use `np.dtype` instead.");
        return NULL;
    }
    // Increment the reference count of the singleton attribute and return it
    Py_INCREF(self->singleton);
    return (PyObject *)self->singleton;
}

/**
 * Create a new instance of a Unicode string DTypeMeta object.
 *
 * @param self The DTypeMeta class instance (Python type object)
 * @param args Tuple of arguments passed to the constructor
 * @param kwargs Dictionary of keyword arguments passed to the constructor
 * @return New instance of the Unicode string DTypeMeta object or NULL on failure
 */
static PyObject *
string_unicode_new(PyArray_DTypeMeta *self, PyObject *args, PyObject *kwargs)
{
    npy_intp size;

    static char *kwlist[] = {"", NULL};

    // Parse arguments to get the size of the string
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist,
                                     PyArray_IntpFromPyIntConverter, &size)) {
        return NULL;
    }

    // Raise an error if the size of the string is negative
    if (size < 0) {
        PyErr_Format(PyExc_ValueError,
                     "Strings cannot have a negative size but a size of "
                     "%"NPY_INTP_FMT" was given", size);
        return NULL;
    }
    # 如果数组类型为 NPY_UNICODE，每个字符占据4个字节
    if (self->type_num == NPY_UNICODE) {
        # 计算字符串所需的总字节数，并检查是否溢出
        if (npy_mul_sizes_with_overflow(&size, size, 4)) {
            # 如果溢出，则设置类型错误异常，并返回空指针
            PyErr_SetString(
                PyExc_TypeError,
                "Strings too large to store inside array.");
            return NULL;
        }
    }

    # 检查字符串所需的总字节数是否超过最大整数表示范围
    if (size > NPY_MAX_INT) {
        PyErr_SetString(PyExc_TypeError,
                        "Strings too large to store inside array.");
        return NULL;
    }

    # 根据数组的类型编号创建一个新的描述符对象
    PyArray_Descr *res = PyArray_DescrNewFromType(self->type_num);

    # 如果创建描述符对象失败，则返回空指针
    if (res == NULL) {
        return NULL;
    }

    # 设置描述符对象的元素大小为计算得到的 size
    res->elsize = (int)size;
    # 返回描述符对象作为 Python 对象
    return (PyObject *)res;
}

static PyArray_Descr *
nonparametric_discover_descr_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj)
{
    /* 如果对象是正确的标量类型，则返回单例 */
    assert(!NPY_DT_is_parametric(cls));
    // 增加引用计数，返回单例描述符
    Py_INCREF(cls->singleton);
    return cls->singleton;
}


static PyArray_Descr *
string_discover_descr_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj)
{
    // 初始化项大小为-1
    npy_intp itemsize = -1;
    // 如果对象是字节对象
    if (PyBytes_Check(obj)) {
        // 获取字节对象的大小
        itemsize = PyBytes_Size(obj);
    }
    // 如果对象是 Unicode 对象
    else if (PyUnicode_Check(obj)) {
        // 获取 Unicode 对象的长度
        itemsize = PyUnicode_GetLength(obj);
    }
    // 如果项大小不为-1
    if (itemsize != -1) {
        // 如果描述符的类型是 NPY_UNICODE
        if (cls->type_num == NPY_UNICODE) {
            // 将项大小乘以4（Unicode 的字节数）
            itemsize *= 4;
        }
        // 如果项大小超过 NPY_MAX_INT
        if (itemsize > NPY_MAX_INT) {
            // 设置类型错误异常
            PyErr_SetString(PyExc_TypeError,
                    "string too large to store inside array.");
        }
        // 从类型号创建新的描述符
        PyArray_Descr *res = PyArray_DescrNewFromType(cls->type_num);
        // 如果创建失败则返回空
        if (res == NULL) {
            return NULL;
        }
        // 设置描述符的元素大小为项大小（强制转换为int类型）
        res->elsize = (int)itemsize;
        return res;
    }
    // 否则调用 PyArray_DTypeFromObjectStringDiscovery 函数进行字符串发现
    return PyArray_DTypeFromObjectStringDiscovery(obj, NULL, cls->type_num);
}


static PyArray_Descr *
void_discover_descr_from_pyobject(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    // 如果对象是空对象标量
    if (PyArray_IsScalar(obj, Void)) {
        // 获取空对象的描述符
        PyVoidScalarObject *void_obj = (PyVoidScalarObject *)obj;
        // 增加空对象描述符的引用计数
        Py_INCREF(void_obj->descr);
        // 返回空对象的描述符
        return (PyArray_Descr *)void_obj->descr;
    }
    // 如果对象是字节对象
    if (PyBytes_Check(obj)) {
        // 从类型号创建新的描述符
        PyArray_Descr *descr = PyArray_DescrNewFromType(NPY_VOID);
        // 如果创建失败则返回空
        if (descr == NULL) {
            return NULL;
        }
        // 获取字节对象的大小
        Py_ssize_t itemsize = PyBytes_Size(obj);
        // 如果字节对象大小超过 NPY_MAX_INT
        if (itemsize > NPY_MAX_INT) {
            // 设置类型错误异常
            PyErr_SetString(PyExc_TypeError,
                    "byte-like to large to store inside array.");
            // 减少描述符的引用计数
            Py_DECREF(descr);
            return NULL;
        }
        // 设置描述符的元素大小为字节对象的大小（强制转换为int类型）
        descr->elsize = (int)itemsize;
        return descr;
    }
    // 否则格式化设置类型错误异常
    PyErr_Format(PyExc_TypeError,
            "A bytes-like object is required, not '%s'", Py_TYPE(obj)->tp_name);
    return NULL;
}


static PyArray_Descr *
discover_datetime_and_timedelta_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj) {
    // 如果对象是日期时间或时间增量的标量对象
    if (PyArray_IsScalar(obj, Datetime) ||
            PyArray_IsScalar(obj, Timedelta)) {
        // 声明日期时间元数据指针
        PyArray_DatetimeMetaData *meta;
        // 从标量对象获取描述符
        PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
        // 从描述符获取日期时间元数据
        meta = get_datetime_metadata_from_dtype(descr);
        // 如果日期时间元数据为空则返回空
        if (meta == NULL) {
            return NULL;
        }
        // 创建新的日期时间类型描述符
        PyArray_Descr *new_descr = create_datetime_dtype(cls->type_num, meta);
        // 释放原描述符的引用计数
        Py_DECREF(descr);
        return new_descr;
    }
    // 否则查找对象的日期时间类型
    else {
        return find_object_datetime_type(obj, cls->type_num);
    }
}


static PyArray_Descr *
nonparametric_default_descr(PyArray_DTypeMeta *cls)
{
    // 增加非参数化描述符的引用计数
    Py_INCREF(cls->singleton);
    // 返回非参数化描述符
    return cls->singleton;
}
/*
 * 对于大多数内置（和遗留）的数据类型，canonical 属性意味着确保本机字节顺序。
 * （这里我们不关心元数据。）
 */
static PyArray_Descr *
ensure_native_byteorder(PyArray_Descr *descr)
{
    // 如果描述符的字节顺序是本机字节顺序，则增加其引用计数并返回该描述符。
    if (PyArray_ISNBO(descr->byteorder)) {
        Py_INCREF(descr);
        return descr;
    }
    // 否则，返回一个新的具有本机字节顺序的描述符。
    else {
        return PyArray_DescrNewByteorder(descr, NPY_NATIVE);
    }
}


/* 确保单例的副本（以防我们在某处进行了适配） */
static PyArray_Descr *
datetime_and_timedelta_default_descr(PyArray_DTypeMeta *cls)
{
    // 返回一个新的描述符，其类型为 cls->singleton。
    return PyArray_DescrNew(cls->singleton);
}


static PyArray_Descr *
void_default_descr(PyArray_DTypeMeta *cls)
{
    // 创建一个新的描述符，其类型为 cls->singleton。
    PyArray_Descr *res = PyArray_DescrNew(cls->singleton);
    // 如果创建失败，返回 NULL。
    if (res == NULL) {
        return NULL;
    }
    /*
     * 对于 `np.array([], dtype="V")` 的遗留行为是使用 "V8"。
     * 这是因为 `[]` 使用 `float64` 作为数据类型，然后用它作为空 void 类型的请求大小。
     */
    // 设置描述符的元素大小为 8。
    res->elsize = 8;
    return res;
}

static PyArray_Descr *
string_and_unicode_default_descr(PyArray_DTypeMeta *cls)
{
    // 根据 cls->type_num 创建一个新的描述符。
    PyArray_Descr *res = PyArray_DescrNewFromType(cls->type_num);
    // 如果创建失败，返回 NULL。
    if (res == NULL) {
        return NULL;
    }
    // 设置描述符的元素大小为 1。
    res->elsize = 1;
    // 如果数据类型是 NPY_UNICODE，则将元素大小乘以 4。
    if (cls->type_num == NPY_UNICODE) {
        res->elsize *= 4;
    }
    return res;
}


static PyArray_Descr *
string_unicode_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    // 如果 descr1 的元素大小大于等于 descr2，则返回确保 canonical 后的 descr1。
    if (descr1->elsize >= descr2->elsize) {
        return NPY_DT_CALL_ensure_canonical(descr1);
    }
    // 否则返回确保 canonical 后的 descr2。
    else {
        return NPY_DT_CALL_ensure_canonical(descr2);
    }
}


static PyArray_Descr *
void_ensure_canonical(_PyArray_LegacyDescr *self)
{
    // 如果存在子数组，则确保其基类型是 canonical 的。
    if (self->subarray != NULL) {
        PyArray_Descr *new_base = NPY_DT_CALL_ensure_canonical(
                self->subarray->base);
        // 如果获取失败，返回 NULL。
        if (new_base == NULL) {
            return NULL;
        }
        // 如果新的基类型和旧的基类型相同，直接返回自身。
        if (new_base == self->subarray->base) {
            /* 只返回自身，无需修改 */
            Py_DECREF(new_base);
            Py_INCREF(self);
            return (PyArray_Descr *)self;
        }
        // 否则，创建一个新的描述符，并设置新的基类型。
        PyArray_Descr *new = PyArray_DescrNew((PyArray_Descr *)self);
        if (new == NULL) {
            return NULL;
        }
        Py_SETREF(((_PyArray_LegacyDescr *)new)->subarray->base, new_base);
        return new;
    }
    // 对于非结构化的 void 类型，始终是 canonical 的。
    else {
        Py_INCREF(self);
        return (PyArray_Descr *)self;
    }
}


static PyArray_Descr *
void_common_instance(_PyArray_LegacyDescr *descr1, _PyArray_LegacyDescr *descr2)
{
    # 检查描述符1和描述符2是否都没有子数组和字段名
    if (descr1->subarray == NULL && descr1->names == NULL &&
            descr2->subarray == NULL && descr2->names == NULL) {
        # 如果元素大小不相等，则返回错误并建议使用 np.bytes_ 数据类型以填充较短的值
        if (descr1->elsize != descr2->elsize) {
            PyErr_SetString(npy_static_pydata.DTypePromotionError,
                    "Invalid type promotion with void datatypes of different "
                    "lengths. Use the `np.bytes_` datatype instead to pad the "
                    "shorter value with trailing zero bytes.");
            return NULL;
        }
        # 增加描述符1的引用计数，并返回其指针类型的转型结果
        Py_INCREF(descr1);
        return (PyArray_Descr *)descr1;
    }

    # 如果描述符1和描述符2都有字段名
    if (descr1->names != NULL && descr2->names != NULL) {
        # 导入并调用 numpy._core._internal 模块中的 _promote_fields 函数
        npy_cache_import("numpy._core._internal", "_promote_fields",
                         &npy_thread_unsafe_state._promote_fields);
        # 如果 _promote_fields 函数未找到，则返回空指针
        if (npy_thread_unsafe_state._promote_fields == NULL) {
            return NULL;
        }
        # 调用 _promote_fields 函数，传入 descr1 和 descr2 作为参数
        PyObject *result = PyObject_CallFunctionObjArgs(
                npy_thread_unsafe_state._promote_fields,
                descr1, descr2, NULL);
        # 如果调用失败，则返回空指针
        if (result == NULL) {
            return NULL;
        }
        # 检查返回的结果是否为有效的描述符对象类型
        if (!PyObject_TypeCheck(result, Py_TYPE(descr1))) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Internal NumPy error: `_promote_fields` did not return "
                    "a valid descriptor object.");
            Py_DECREF(result);
            return NULL;
        }
        # 返回转换后的描述符对象
        return (PyArray_Descr *)result;
    }
    # 如果描述符1和描述符2都有子数组
    else if (descr1->subarray != NULL && descr2->subarray != NULL) {
        # 比较两个子数组的形状是否相等
        int cmp = PyObject_RichCompareBool(
                descr1->subarray->shape, descr2->subarray->shape, Py_EQ);
        # 如果比较失败，则返回空指针
        if (error_converting(cmp)) {
            return NULL;
        }
        # 如果形状不相等，则返回错误信息
        if (!cmp) {
            PyErr_SetString(npy_static_pydata.DTypePromotionError,
                    "invalid type promotion with subarray datatypes "
                    "(shape mismatch).");
            return NULL;
        }
        # 提升子数组的基本类型并返回新的描述符
        PyArray_Descr *new_base = PyArray_PromoteTypes(
                descr1->subarray->base, descr2->subarray->base);
        # 如果提升失败，则返回空指针
        if (new_base == NULL) {
            return NULL;
        }
        /*
         * 如果是相同的数据类型且容器没有变化，可以保留标识和元数据。
         * 这可能需要进一步改进。
         */
        # 如果描述符1和描述符2相同，并且新的基础类型与描述符1的子数组的基础类型相同，则保留描述符1的引用计数并返回其指针类型的转型结果
        if (descr1 == descr2 && new_base == descr1->subarray->base) {
            Py_DECREF(new_base);
            Py_INCREF(descr1);
            return (PyArray_Descr *)descr1;
        }

        # 创建一个新的描述符对象，并设置其子数组的基础类型为新的基础类型
        PyArray_Descr *new_descr = PyArray_DescrNew((PyArray_Descr *)descr1);
        # 如果创建新的描述符对象失败，则释放新的基础类型并返回空指针
        if (new_descr == NULL) {
            Py_DECREF(new_base);
            return NULL;
        }
        Py_SETREF(((_PyArray_LegacyDescr *)new_descr)->subarray->base, new_base);
        # 返回新的描述符对象
        return new_descr;
    }

    # 如果既不是没有子数组和字段名，也不是都有字段名，也不是都有子数组，则返回结构化数据类型的错误信息
    PyErr_SetString(npy_static_pydata.DTypePromotionError,
            "invalid type promotion with structured datatype(s).");
    # 返回空值(NULL)
    return NULL;
/* 
 * 下面的函数判断Python内置的标量类型是否已知。
 * 主要目的是确保不会发生pyfloat->float64->整数的转换。
 * 姑且假设子类会被拒绝发现。
 * 这个函数仅用于我们发现的作为有效DType的Python标量类。
 */
NPY_NO_EXPORT int
python_builtins_are_known_scalar_types(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyTypeObject *pytype)
{
    /*
     * 始终接受常见的Python类型，以确保我们不会进行pyfloat->float64->整数的转换。
     * 子类希望被拒绝发现。
     * 这仅适用于我们发现的作为有效DType的Python标量类。
     */
    if (pytype == &PyFloat_Type ||
        pytype == &PyLong_Type ||
        pytype == &PyBool_Type ||
        pytype == &PyComplex_Type ||
        pytype == &PyUnicode_Type ||
        pytype == &PyBytes_Type)
    {
        return 1;  // 是已知的标量类型
    }
    return 0;  // 不是已知的标量类型
}


/*
 * 如果Python内置的标量类型是已知的，返回1；否则返回0。
 */
static int
signed_integers_is_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype)
{
    if (python_builtins_are_known_scalar_types(cls, pytype)) {
        return 1;  // 是已知的标量类型
    }
    /* 转换我们的标量（在太大的无符号数和NaN等情况下引发异常）。 */
    return PyType_IsSubtype(pytype, &PyGenericArrType_Type);
}


/*
 * 如果Python内置的标量类型是已知的，返回1；否则返回0。
 */
static int
datetime_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype)
{
    if (python_builtins_are_known_scalar_types(cls, pytype)) {
        return 1;  // 是已知的标量类型
    }
    /*
     * 为了能够从任意字符串中识别描述符，datetime 必须负责。
     * 否则我们会尝试转换，这种转换并不真正支持这种情况。
     * 只有对象数组在这种方式上有特殊情况。
     */
    return (PyType_IsSubtype(pytype, &PyBytes_Type) ||
            PyType_IsSubtype(pytype, &PyUnicode_Type));
}


/*
 * 如果Python内置的标量类型是已知的，返回1；否则返回0。
 */
static int
string_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype) {
    if (python_builtins_are_known_scalar_types(cls, pytype)) {
        return 1;  // 是已知的标量类型
    }
    if (PyType_IsSubtype(pytype, &PyDatetimeArrType_Type)) {
        /*
         * TODO: 这可能应该被弃用或以其他方式解决。
         *       不幸的是，必须在 `String->setitem` 中弃用。
         *
         * 目前datetime不会转换为较短的字符串，但对于任意值的字符串强制使用`str(obj)[:len]`。
         * 这意味着`np.array(np.datetime64("2020-01-01"), "U9")`
         * 和 `np.array(np.datetime64("2020-01-01")).astype("U9")` 的行为不同。
         */
        return 1;  // 是已知的标量类型
    }
    return 0;  // 不是已知的标量类型
}


/*
 * 下面的一组函数定义了内置类型的常见dtype运算符。
 */
static PyArray_DTypeMeta *
default_builtin_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    assert(cls->type_num < NPY_NTYPES_LEGACY);
    if (NPY_UNLIKELY(!NPY_DT_is_legacy(other))) {
        /*
         * 处理我们理解的非遗留类型：Python 标量。
         * 这些类型可能比具体的不精确类型优先级低，
         * 但可以改变结果的类型（复数、浮点数、整数）。
         * 如果我们自己的 DType 不是数值类型或者优先级更低（例如整数但抽象类型是浮点数），
         * 则信号表示未实现。
         */
        if (other == &PyArray_PyComplexDType) {
            if (PyTypeNum_ISCOMPLEX(cls->type_num)) {
                Py_INCREF(cls);
                return cls;
            }
            else if (cls->type_num == NPY_HALF || cls->type_num == NPY_FLOAT) {
                return NPY_DT_NewRef(&PyArray_CFloatDType);
            }
            else if (cls->type_num == NPY_DOUBLE) {
                return NPY_DT_NewRef(&PyArray_CDoubleDType);
            }
            else if (cls->type_num == NPY_LONGDOUBLE) {
                return NPY_DT_NewRef(&PyArray_CLongDoubleDType);
            }
        }
        else if (other == &PyArray_PyFloatDType) {
            if (PyTypeNum_ISCOMPLEX(cls->type_num)
                    || PyTypeNum_ISFLOAT(cls->type_num)) {
                Py_INCREF(cls);
                return cls;
            }
        }
        else if (other == &PyArray_PyLongDType) {
            if (PyTypeNum_ISCOMPLEX(cls->type_num)
                    || PyTypeNum_ISFLOAT(cls->type_num)
                    || PyTypeNum_ISINTEGER(cls->type_num)
                    || cls->type_num == NPY_TIMEDELTA) {
                Py_INCREF(cls);
                return cls;
            }
        }
        // 返回未实现的信号
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    // 如果 other 的类型号大于 cls 的类型号
    if (other->type_num > cls->type_num) {
        /*
         * 让更通用（类型号更大）的 DType 处理此情况
         * （注意半精度类型在所有其他类型之后，这里可以正常工作）。
         */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }

    /*
     * 注意：可能应该在某个时候重新审视提升表的使用。
     * 可能最有用的是完全移除它，然后考虑添加一个快速路径/缓存 `PyArray_CommonDType()` 本身。
     */
    // 获取公共类型的类型号
    int common_num = _npy_type_promotion_table[cls->type_num][other->type_num];
    // 如果没有公共类型，返回未实现的信号
    if (common_num < 0) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    // 根据公共类型的类型号返回对应的 DType
    return PyArray_DTypeFromTypeNum(common_num);
static PyArray_DTypeMeta *
string_unicode_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // 断言确保cls的类型编号小于NPY_NTYPES_LEGACY且不等于other
    assert(cls->type_num < NPY_NTYPES_LEGACY && cls != other);
    // 如果other不是遗留类型，或者不是数字类型并且(cls是unicode而other是string)，则返回NotImplemented
    if (!NPY_DT_is_legacy(other) || (!PyTypeNum_ISNUMBER(other->type_num) &&
            !(cls->type_num == NPY_UNICODE && other->type_num == NPY_STRING))) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    /*
     * 内建类型按复杂性排序（除了object类型）。
     * 或许我们不应该把数字和字符串视为“常见”，但目前我们这样做了。
     */
    // 增加对cls的引用计数并返回它
    Py_INCREF(cls);
    return cls;
}


static PyArray_DTypeMeta *
datetime_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    /*
     * 时间增量和日期时间实际上不应该进行提升。当前这么做意味着我们在比较类型解析器中需要额外的hack。
     * 对于比较，我们必须确保优雅地拒绝它，以便返回True/False值的数组。
     */
    // 如果cls是日期时间类型且other是时间增量类型，则返回cls
    if (cls->type_num == NPY_DATETIME && other->type_num == NPY_TIMEDELTA) {
        /*
         * TODO: 实际上我们目前允许在这里进行提升。在`np.add(datetime, timedelta)`中当前依赖于此，
         *       而对于连接操作，类型转换步骤将失败。
         */
        Py_INCREF(cls);
        return cls;
    }
    // 否则调用默认的内建常见类型处理函数
    return default_builtin_common_dtype(cls, other);
}



static PyArray_DTypeMeta *
object_common_dtype(
        PyArray_DTypeMeta *cls, PyArray_DTypeMeta *NPY_UNUSED(other))
{
    /*
     * object类型DType是特殊的，它可以表示一切，包括所有潜在的用户定义DType。
     * 在此推迟（或出错）的一个原因可能是，其他的DType不支持标量，因此`arr1d[0]`返回一个0-D数组，
     * 而`arr.astype(object)`将失败。但对象类型转换是特殊的。
     */
    // 增加对cls的引用计数并返回它
    Py_INCREF(cls);
    return cls;
}
# 此函数接受一个 PyArray_Descr，并用新创建的 dtype 子类(DTypeMeta 实例)替换其基类。
# 这样做时需要记住一些微妙之处，首先是对于类对象本身，它可能是 HeapType，也可能不是。
# 由于我们是用 C 定义 DType，所以我们不会将其定义为 HeapType，从而使其与典型的*静态*类型相同(只是我们使用 malloc 来分配内存而已)。
# 我们也可以使用另一种方法，但似乎没有理由那样做。
#
# DType 实例(实际的数据类型或描述符)基于传入的原型。这些原型不应该被垃圾回收，因此不设置 Py_TPFLAGS_HAVE_GC。(我们可以允许这样做，但那将需要分配一个新的对象，因为 GC 需要在实际结构体之前获取信息)。
#
# 以上是为什么我们在这里应该像为静态类型一样工作的原因。
# 否则，我们会在 C 定义的扩展类与 Python 子类之间有所模糊。例如，`class MyInt(int): pass` 与 `class Float64(np.dtype): pass` 是非常不同的，因为后者不应该是 HeapType，并且其实例应该是确切的 PyArray_Descr 结构体。
#
# @param descr 应该包装的描述符。
# @param name DType 的名称。
# @param alias 也设置为新类的第二个名称，用于内建类型 (`np.types.LongDType` 对应 `np.types.Int64DType`)。有些可能有更多的别名，例如 `intp` 不是自己的东西，在写这篇文章时，这些还没有添加。
#
# @returns 成功返回 0，失败返回 -1。
NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(
    _PyArray_LegacyDescr *descr, PyArray_ArrFuncs *arr_funcs,
    PyTypeObject *dtype_super_class, const char *name, const char *alias)
{
    int has_type_set = Py_TYPE(descr) == &PyArrayDescr_Type;

    if (!has_type_set) {
        # 如果类型是从现有的内建 dtype 填充进来的，则接受
        for (int i = 0; i < NPY_NTYPES_LEGACY; i++) {
            PyArray_Descr *builtin = PyArray_DescrFromType(i);
            has_type_set = Py_TYPE(descr) == Py_TYPE(builtin);
            Py_DECREF(builtin);
            if (has_type_set) {
                break;
            }
        }
    }
    if (!has_type_set) {
        PyErr_Format(PyExc_RuntimeError,
                "在创建/包装旧版本 DType 时，原始类不是 PyArrayDescr_Type (它在这一步被替换)。"
                "创建自定义 DType 用于类型 %S 的扩展程序必须修改，以确保 `Py_TYPE(descr) == &PyArrayDescr_Type` 或者一个现有的 dtype (假设它只是被复制过来并且可以被替换)。",
                descr->typeobj, Py_TYPE(descr));
        return -1;
    }

    NPY_DType_Slots *dt_slots = PyMem_Malloc(sizeof(NPY_DType_Slots));
    // 检查传入的 dt_slots 是否为 NULL，如果是则返回错误码 -1
    if (dt_slots == NULL) {
        return -1;
    }
    
    // 将 dt_slots 的内存清零，大小为 NPY_DType_Slots 结构的大小
    memset(dt_slots, '\0', sizeof(NPY_DType_Slots));

    // 分配 PyArray_DTypeMeta 类的内存空间
    PyArray_DTypeMeta *dtype_class = PyMem_Malloc(sizeof(PyArray_DTypeMeta));
    // 如果分配失败，则释放先前分配的 dt_slots 的内存，并返回错误码 -1
    if (dtype_class == NULL) {
        PyMem_Free(dt_slots);
        return -1;
    }

    /*
     * 通过复制原型实例的方式初始化结构字段，除了我们自己的字段之外，
     * 在不同的 DType 之间会有变化。
     * 特别是任何对象的初始化必须严格从未更改的原型复制以避免复杂性（例如 PyPy）。
     * 任何类型的插槽在 PyType_Ready 之前都需要固定，尽管大多数将在那里自动继承。
     */
    static PyArray_DTypeMeta prototype = {
        {{
            PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
            .tp_name = NULL,  // 在下面设置
            .tp_basicsize = sizeof(_PyArray_LegacyDescr),
            .tp_flags = Py_TPFLAGS_DEFAULT,
            .tp_base = NULL,  // 在下面设置
            .tp_new = (newfunc)legacy_dtype_default_new,
            .tp_doc = (
                "DType class corresponding to the scalar type and dtype of "
                "the same name.\n\n"
                "Please see `numpy.dtype` for the typical way to create\n"
                "dtype instances and :ref:`arrays.dtypes` for additional\n"
                "information."),
        },},
        .flags = NPY_DT_LEGACY,
        /* 进一步的字段在不同的 DType 中不是共同的 */
    };
    
    // 将原型实例的内容复制到 dtype_class 中
    memcpy(dtype_class, &prototype, sizeof(PyArray_DTypeMeta));
    
    // 设置 Type 的名称和超类
    ((PyTypeObject *)dtype_class)->tp_name = name;
    ((PyTypeObject *)dtype_class)->tp_base = dtype_super_class;
    dtype_class->dt_slots = dt_slots;

    // 让 Python 完成初始化过程
    if (PyType_Ready((PyTypeObject *)dtype_class) < 0) {
        Py_DECREF(dtype_class);
        return -1;
    }
    
    // 为 dt_slots 的 castingimpls 字典分配空间
    dt_slots->castingimpls = PyDict_New();
    // 如果分配失败，则释放 dtype_class 内存，并返回错误码 -1
    if (dt_slots->castingimpls == NULL) {
        Py_DECREF(dtype_class);
        return -1;
    }

    /*
     * 填充在不同 DType 之间变化的 DTypeMeta 信息，
     * 任何变量类型信息在 PyType_Ready() 之前需要设置。
     */
    dtype_class->singleton = (PyArray_Descr *)descr;
    Py_INCREF(descr->typeobj);
    dtype_class->scalar_type = descr->typeobj;
    dtype_class->type_num = descr->type_num;
    dt_slots->f = *arr_funcs;

    // 设置默认函数（对大多数 dtype 正确），在下面可能会被覆盖
    dt_slots->default_descr = nonparametric_default_descr;
    dt_slots->discover_descr_from_pyobject = (
            nonparametric_discover_descr_from_pyobject);
    dt_slots->is_known_scalar_type = python_builtins_are_known_scalar_types;
    dt_slots->common_dtype = default_builtin_common_dtype;
    dt_slots->common_instance = NULL;
    dt_slots->ensure_canonical = ensure_native_byteorder;
    dt_slots->get_fill_zero_loop = NULL;
    // 设置 finalize_descr 字段为 NULL
    dt_slots->finalize_descr = NULL;

    // 如果 dtype_class 的类型编号是有符号整数类型
    if (PyTypeNum_ISSIGNED(dtype_class->type_num)) {
        // 将 is_known_scalar_type 设置为 signed_integers_is_known_scalar_types 函数
        dt_slots->is_known_scalar_type = signed_integers_is_known_scalar_types;
    }

    // 如果 descr 的类型编号是用户定义的类型
    if (PyTypeNum_ISUSERDEF(descr->type_num)) {
        // 将 common_dtype 设置为 legacy_userdtype_common_dtype_function 函数
        dt_slots->common_dtype = legacy_userdtype_common_dtype_function;
    }
    // 否则如果 descr 的类型编号是 NPY_OBJECT
    else if (descr->type_num == NPY_OBJECT) {
        // 将 common_dtype 设置为 object_common_dtype
        dt_slots->common_dtype = object_common_dtype;
        // 设置 get_fill_zero_loop 为 npy_object_get_fill_zero_loop 函数
        dt_slots->get_fill_zero_loop = npy_object_get_fill_zero_loop;
        // 设置 get_clear_loop 为 npy_get_clear_object_strided_loop 函数
        dt_slots->get_clear_loop = npy_get_clear_object_strided_loop;
    }
    // 否则如果 descr 的类型编号是日期时间类型
    else if (PyTypeNum_ISDATETIME(descr->type_num)) {
        // 设置 dtype_class 的 flags 添加 NPY_DT_PARAMETRIC 标志位
        dtype_class->flags |= NPY_DT_PARAMETRIC;
        // 设置 default_descr 为 datetime_and_timedelta_default_descr 函数
        dt_slots->default_descr = datetime_and_timedelta_default_descr;
        // 设置 discover_descr_from_pyobject 为 discover_datetime_and_timedelta_from_pyobject 函数
        dt_slots->discover_descr_from_pyobject = (
                discover_datetime_and_timedelta_from_pyobject);
        // 设置 common_dtype 为 datetime_common_dtype
        dt_slots->common_dtype = datetime_common_dtype;
        // 设置 common_instance 为 datetime_type_promotion
        dt_slots->common_instance = datetime_type_promotion;
        // 如果 descr 的类型编号是 NPY_DATETIME
        if (descr->type_num == NPY_DATETIME) {
            // 设置 is_known_scalar_type 为 datetime_known_scalar_types 函数
            dt_slots->is_known_scalar_type = datetime_known_scalar_types;
        }
    }
    // 否则如果 descr 的类型编号是灵活类型
    else if (PyTypeNum_ISFLEXIBLE(descr->type_num)) {
        // 设置 dtype_class 的 flags 添加 NPY_DT_PARAMETRIC 标志位
        dtype_class->flags |= NPY_DT_PARAMETRIC;
        // 如果 descr 的类型编号是 NPY_VOID
        if (descr->type_num == NPY_VOID) {
            // 设置 default_descr 为 void_default_descr 函数
            dt_slots->default_descr = void_default_descr;
            // 设置 discover_descr_from_pyobject 为 void_discover_descr_from_pyobject 函数
            dt_slots->discover_descr_from_pyobject = (
                    void_discover_descr_from_pyobject);
            // 设置 common_instance 为 void_common_instance
            dt_slots->common_instance = (PyArrayDTypeMeta_CommonInstance *)void_common_instance;
            // 设置 ensure_canonical 为 void_ensure_canonical 函数
            dt_slots->ensure_canonical = (PyArrayDTypeMeta_EnsureCanonical *)void_ensure_canonical;
            // 设置 get_fill_zero_loop 为 npy_get_zerofill_void_and_legacy_user_dtype_loop 函数
            dt_slots->get_fill_zero_loop =
                    (PyArrayMethod_GetTraverseLoop *)npy_get_zerofill_void_and_legacy_user_dtype_loop;
            // 设置 get_clear_loop 为 npy_get_clear_void_and_legacy_user_dtype_loop 函数
            dt_slots->get_clear_loop =
                    (PyArrayMethod_GetTraverseLoop *)npy_get_clear_void_and_legacy_user_dtype_loop;
        }
        // 否则
        else {
            // 设置 default_descr 为 string_and_unicode_default_descr 函数
            dt_slots->default_descr = string_and_unicode_default_descr;
            // 设置 is_known_scalar_type 为 string_known_scalar_types 函数
            dt_slots->is_known_scalar_type = string_known_scalar_types;
            // 设置 discover_descr_from_pyobject 为 string_discover_descr_from_pyobject 函数
            dt_slots->discover_descr_from_pyobject = (
                    string_discover_descr_from_pyobject);
            // 设置 common_dtype 为 string_unicode_common_dtype
            dt_slots->common_dtype = string_unicode_common_dtype;
            // 设置 common_instance 为 string_unicode_common_instance
            dt_slots->common_instance = string_unicode_common_instance;
            // 将 dtype_class 强制转换为 PyTypeObject 类型，并设置 tp_new 方法为 string_unicode_new 函数
            ((PyTypeObject*)dtype_class)->tp_new = (newfunc)string_unicode_new;
        }
    }

    // 如果 descr 的类型编号是数字类型
    if (PyTypeNum_ISNUMBER(descr->type_num)) {
        // 设置 dtype_class 的 flags 添加 NPY_DT_NUMERIC 标志位
        dtype_class->flags |= NPY_DT_NUMERIC;
    }

    // 将 descr 的类型对象映射到 dtype_class，并根据是否是用户定义的类型来决定
    if (_PyArray_MapPyTypeToDType(dtype_class, descr->typeobj,
            PyTypeNum_ISUSERDEF(dtype_class->type_num)) < 0) {
        // 出错时释放 dtype_class，并返回 -1
        Py_DECREF(dtype_class);
        return -1;
    }

    // 最后，将 descr 的类设置为 dtype_class
    Py_SET_TYPE(descr, (PyTypeObject *)dtype_class);
    /* 如果描述符的类型不是用户定义的内置数据类型，则执行以下操作 */
    if (!PyTypeNum_ISUSERDEF(descr->type_num)) {
        /* 导入 numpy.dtypes 模块中的 _add_dtype_helper 函数，并存储在全局变量中 */
        npy_cache_import("numpy.dtypes", "_add_dtype_helper",
                         &npy_thread_unsafe_state._add_dtype_helper);
        /* 如果无法导入 _add_dtype_helper 函数，则返回错误码 */
        if (npy_thread_unsafe_state._add_dtype_helper == NULL) {
            return -1;
        }

        /* 调用 _add_dtype_helper 函数，向其传递 dtype_class 对象和 alias 字符串作为参数 */
        if (PyObject_CallFunction(
                npy_thread_unsafe_state._add_dtype_helper,
                "Os", (PyObject *)dtype_class, alias) == NULL) {
            return -1;
        }
    }

    /* 操作成功完成，返回成功码 */
    return 0;
/*
 * 获取抽象类型的属性的函数
 */
static PyObject *
dtypemeta_get_abstract(PyArray_DTypeMeta *self) {
    return PyBool_FromLong(NPY_DT_is_abstract(self));
}

/*
 * 获取遗留类型的属性的函数
 */
static PyObject *
dtypemeta_get_legacy(PyArray_DTypeMeta *self) {
    return PyBool_FromLong(NPY_DT_is_legacy(self));
}

/*
 * 获取参数化类型的属性的函数
 */
static PyObject *
dtypemeta_get_parametric(PyArray_DTypeMeta *self) {
    return PyBool_FromLong(NPY_DT_is_parametric(self));
}

/*
 * 获取是否为数值类型的属性的函数
 */
static PyObject *
dtypemeta_get_is_numeric(PyArray_DTypeMeta *self) {
    return PyBool_FromLong(NPY_DT_is_numeric(self));
}

/*
 * 定义每个 DType（数据类型）类的公开信息。
 * 包含用于访问和设置的属性定义。
 */
static PyGetSetDef dtypemeta_getset[] = {
        {"_abstract", (getter)dtypemeta_get_abstract, NULL, NULL, NULL},  // 获取抽象属性的访问器
        {"_legacy", (getter)dtypemeta_get_legacy, NULL, NULL, NULL},      // 获取遗留属性的访问器
        {"_parametric", (getter)dtypemeta_get_parametric, NULL, NULL, NULL},  // 获取参数化属性的访问器
        {"_is_numeric", (getter)dtypemeta_get_is_numeric, NULL, NULL, NULL},  // 获取是否为数值类型属性的访问器
        {NULL, NULL, NULL, NULL, NULL}  // 结束符号
};

/*
 * 定义 DTypeMeta 结构体的成员变量及其描述信息。
 */
static PyMemberDef dtypemeta_members[] = {
    {"type",
        T_OBJECT, offsetof(PyArray_DTypeMeta, scalar_type), READONLY,
        "scalar type corresponding to the DType."},  // DType 对应的标量类型
    {NULL, 0, 0, 0, NULL},  // 结束符号
};

/*
 * 初始化遗留 DTypeMeta 别名的函数。
 * 将 _builtin_descrs 中的每个内置描述符与其对应的 NPY_DTYPE 宏扩展连接。
 */
NPY_NO_EXPORT void
initialize_legacy_dtypemeta_aliases(_PyArray_LegacyDescr **_builtin_descrs) {
    _Bool_dtype = NPY_DTYPE(_builtin_descrs[NPY_BOOL]);  // 布尔类型的别名
    _Byte_dtype = NPY_DTYPE(_builtin_descrs[NPY_BYTE]);  // 字节类型的别名
    _UByte_dtype = NPY_DTYPE(_builtin_descrs[NPY_UBYTE]);  // 无符号字节类型的别名
    _Short_dtype = NPY_DTYPE(_builtin_descrs[NPY_SHORT]);  // 短整型的别名
    _UShort_dtype = NPY_DTYPE(_builtin_descrs[NPY_USHORT]);  // 无符号短整型的别名
    _Int_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT]);  // 整型的别名
    _UInt_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT]);  // 无符号整型的别名
    _Long_dtype = NPY_DTYPE(_builtin_descrs[NPY_LONG]);  // 长整型的别名
    _ULong_dtype = NPY_DTYPE(_builtin_descrs[NPY_ULONG]);  // 无符号长整型的别名
    _LongLong_dtype = NPY_DTYPE(_builtin_descrs[NPY_LONGLONG]);  // 长长整型的别名
    _ULongLong_dtype = NPY_DTYPE(_builtin_descrs[NPY_ULONGLONG]);  // 无符号长长整型的别名
    _Int8_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT8]);  // 8 位整型的别名
    _UInt8_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT8]);  // 无符号 8 位整型的别名
    _Int16_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT16]);  // 16 位整型的别名
    _UInt16_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT16]);  // 无符号 16 位整型的别名
    _Int32_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT32]);  // 32 位整型的别名
    _UInt32_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT32]);  // 无符号 32 位整型的别名
    _Int64_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT64]);  // 64 位整型的别名
    _UInt64_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT64]);  // 无符号 64 位整型的别名
    _Intp_dtype = NPY_DTYPE(_builtin_descrs[NPY_INTP]);  // 平台相关整型的别名
    _UIntp_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINTP]);  // 无符号平台相关整型的别名
    _DefaultInt_dtype = NPY_DTYPE(_builtin_descrs[NPY_DEFAULT_INT]);  // 默认整型的别名
    _Half_dtype = NPY_DTYPE(_builtin_descrs[NPY_HALF]);  // 半精度浮点数的别名
    _Float_dtype = NPY_DTYPE(_builtin_descrs[NPY_FLOAT]);  // 单精度浮点数的别名
    _Double_dtype = NPY_DTYPE(_builtin_descrs[NPY_DOUBLE]);  // 双精度浮点数的别名
    _LongDouble_dtype = NPY_DTYPE(_builtin_descrs[NPY_LONGDOUBLE]);  // 长双精度浮点数的别名
    _CFloat_dtype = NPY_DTYPE(_builtin_descrs[NPY_CFLOAT]);  // 复数单精度浮点数的别名
    _CDouble_dtype = NPY_DTYPE(_builtin_descrs[NPY_CDOUBLE]);  // 复数双精度浮点数的别名
    _CLongDouble_dtype = NPY_DTYPE(_builtin_descrs[NPY_CLONGDOUBLE]);  // 复数长双精度浮点数的别名
}
    // NPY_STRING 是 Python 2 的遗留名称，用于描述字节字符串类型的数据类型
    _Bytes_dtype = NPY_DTYPE(_builtin_descrs[NPY_STRING]);
    // NPY_UNICODE 是描述 Unicode 字符串类型的数据类型
    _Unicode_dtype = NPY_DTYPE(_builtin_descrs[NPY_UNICODE]);
    // NPY_DATETIME 是描述日期时间类型的数据类型
    _Datetime_dtype = NPY_DTYPE(_builtin_descrs[NPY_DATETIME]);
    // NPY_TIMEDELTA 是描述时间间隔类型的数据类型
    _Timedelta_dtype = NPY_DTYPE(_builtin_descrs[NPY_TIMEDELTA]);
    // NPY_OBJECT 是描述 Python 对象类型的数据类型
    _Object_dtype = NPY_DTYPE(_builtin_descrs[NPY_OBJECT]);
    // NPY_VOID 是描述 void（空）数据类型的数据类型
    _Void_dtype = NPY_DTYPE(_builtin_descrs[NPY_VOID]);
}

# 定义 PyArrayDTypeMeta_Type 类型对象，用于描述 NumPy 的数据类型元类
NPY_NO_EXPORT PyTypeObject PyArrayDTypeMeta_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)  # 初始化变长对象头部，设置为 NULL，大小为 0
    .tp_name = "numpy._DTypeMeta",   # 类型对象的名称字符串
    .tp_basicsize = sizeof(PyArray_DTypeMeta),  # 类型对象的基本大小，即 PyArray_DTypeMeta 结构体的大小
    .tp_dealloc = (destructor)dtypemeta_dealloc,  # 类型对象的析构函数指针，用于释放对象内存
    /* Types are garbage collected (see dtypemeta_is_gc documentation) */
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,  # 类型对象的标志，包括默认标志和垃圾回收标志
    .tp_doc = "Preliminary NumPy API: The Type of NumPy DTypes (metaclass)",  # 类型对象的文档字符串
    .tp_traverse = (traverseproc)dtypemeta_traverse,  # 类型对象的遍历函数指针，用于支持垃圾回收
    .tp_members = dtypemeta_members,  # 类型对象的成员变量描述结构体指针
    .tp_getset = dtypemeta_getset,  # 类型对象的属性获取和设置函数结构体指针
    .tp_base = NULL,  /* set to PyType_Type at import time */  # 类型对象的基类指针，在导入时设置为 PyType_Type
    .tp_init = (initproc)dtypemeta_init,  # 类型对象的初始化函数指针，用于对象的初始化
    .tp_alloc = dtypemeta_alloc,  # 类型对象的内存分配函数指针
    .tp_new = dtypemeta_new,  # 类型对象的新建对象函数指针，用于创建新对象
    .tp_is_gc = dtypemeta_is_gc,  # 类型对象的垃圾回收检查函数指针，用于确定对象是否应当被回收
};

# 各种 NumPy 数据类型元类的全局变量声明，初始值均为 NULL
PyArray_DTypeMeta *_Bool_dtype = NULL;
PyArray_DTypeMeta *_Byte_dtype = NULL;
PyArray_DTypeMeta *_UByte_dtype = NULL;
PyArray_DTypeMeta *_Short_dtype = NULL;
PyArray_DTypeMeta *_UShort_dtype = NULL;
PyArray_DTypeMeta *_Int_dtype = NULL;
PyArray_DTypeMeta *_UInt_dtype = NULL;
PyArray_DTypeMeta *_Long_dtype = NULL;
PyArray_DTypeMeta *_ULong_dtype = NULL;
PyArray_DTypeMeta *_LongLong_dtype = NULL;
PyArray_DTypeMeta *_ULongLong_dtype = NULL;
PyArray_DTypeMeta *_Int8_dtype = NULL;
PyArray_DTypeMeta *_UInt8_dtype = NULL;
PyArray_DTypeMeta *_Int16_dtype = NULL;
PyArray_DTypeMeta *_UInt16_dtype = NULL;
PyArray_DTypeMeta *_Int32_dtype = NULL;
PyArray_DTypeMeta *_UInt32_dtype = NULL;
PyArray_DTypeMeta *_Int64_dtype = NULL;
PyArray_DTypeMeta *_UInt64_dtype = NULL;
PyArray_DTypeMeta *_Intp_dtype = NULL;
PyArray_DTypeMeta *_UIntp_dtype = NULL;
PyArray_DTypeMeta *_DefaultInt_dtype = NULL;
PyArray_DTypeMeta *_Half_dtype = NULL;
PyArray_DTypeMeta *_Float_dtype = NULL;
PyArray_DTypeMeta *_Double_dtype = NULL;
PyArray_DTypeMeta *_LongDouble_dtype = NULL;
PyArray_DTypeMeta *_CFloat_dtype = NULL;
PyArray_DTypeMeta *_CDouble_dtype = NULL;
PyArray_DTypeMeta *_CLongDouble_dtype = NULL;
PyArray_DTypeMeta *_Bytes_dtype = NULL;
PyArray_DTypeMeta *_Unicode_dtype = NULL;
PyArray_DTypeMeta *_Datetime_dtype = NULL;
PyArray_DTypeMeta *_Timedelta_dtype = NULL;
PyArray_DTypeMeta *_Object_dtype = NULL;
PyArray_DTypeMeta *_Void_dtype = NULL;
/*
 * NUMPY_API
 * 获取 ArrFuncs 结构体，该结构体现在存在于 DType 而非描述符中。
 * 应尽量避免使用此结构体，但某些功能仍然需要使用。
 *
 * 除了检查 NULL 之后的 getitem、setitem、copyswap 和 copyswapn 之外，
 * 使用任何插槽都是无效的。通常鼓励检查 NULL。
 *
 * 由于公共版本是静态内联函数，仅在 2.x 上调用函数，但在 1.x 上直接访问 `descr` 结构，
 * 所以此函数使用下划线标记为“私有”版本。一旦 1.x 向后兼容性消失，应直接导出而不带下划线。
 * 在内部，我们定义了一个私有内联函数 `PyDataType_GetArrFuncs` 以方便直接访问 `DType` 插槽。
 */
NPY_NO_EXPORT PyArray_ArrFuncs *
_PyDataType_GetArrFuncs(const PyArray_Descr *descr)
{
    // 调用公共函数 PyDataType_GetArrFuncs 获取 ArrFuncs 结构体
    return PyDataType_GetArrFuncs(descr);
}
```