# `.\numpy\numpy\_core\src\multiarray\descriptor.c`

```
/* Array Descr Object */

/* Define to use the latest NumPy API version */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/* Define to enable multiarray module */
#define _MULTIARRAYMODULE

/* Define to clean PY_SSIZE_T definition */
#define PY_SSIZE_T_CLEAN
/* Include Python.h header file */
#include <Python.h>
/* Include structmember.h header file for structure member handling */
#include <structmember.h>

/* Include errno.h header file for error number definitions */
#include <errno.h>

/* Include NumPy core arrayobject header file */
#include "numpy/arrayobject.h"
/* Include NumPy array scalars header file */
#include "numpy/arrayscalars.h"
/* Include NumPy math functions header file */
#include "numpy/npy_math.h"

/* Include NumPy configuration header file */
#include "npy_config.h"
/* Include NumPy ctypes header file for ctypes integration */
#include "npy_ctypes.h"
/* Include NumPy import header file */
#include "npy_import.h"

/* Include datetime functions from _datetime.h */
#include "_datetime.h"
/* Include common functions from common.h */
#include "common.h"
/* Include conversion utilities from conversion_utils.h */
#include "conversion_utils.h"
/* Include templ_common.h for common templated functions */
#include "templ_common.h"
/* Include descriptor.h for dtype descriptor definitions */
#include "descriptor.h"
/* Include static data definitions from npy_static_data.h */
#include "npy_static_data.h"
/* Include multiarraymodule.h for thread-unsafe state access */
#include "multiarraymodule.h"
/* Include allocation functions from alloc.h */
#include "alloc.h"
/* Include assert.h header file for assertions */
#include "assert.h"
/* Include buffer definitions from npy_buffer.h */
#include "npy_buffer.h"
/* Include dtype metadata from dtypemeta.h */
#include "dtypemeta.h"
/* Include string dtype definitions from stringdtype/dtype.h */
#include "stringdtype/dtype.h"

/* Define PyDictProxy_Check macro if not already defined */
#ifndef PyDictProxy_Check
#define PyDictProxy_Check(obj) (Py_TYPE(obj) == &PyDictProxy_Type)
#endif

/* Initialize global variable for storing type dictionary */
static PyObject *typeDict = NULL;   /* Must be explicitly loaded */

/* Declare function for trying to convert from an inherited tuple */
static PyArray_Descr *
_try_convert_from_inherit_tuple(PyArray_Descr *type, PyObject *newobj);

/* Declare function for converting from any Python object */
static PyArray_Descr *
_convert_from_any(PyObject *obj, int align);

/*
 * This function creates a dtype object when the object is a ctypes subclass.
 *
 * Returns `Py_NotImplemented` if the type is not a ctypes subclass.
 */
static PyArray_Descr *
_try_convert_from_ctypes_type(PyTypeObject *type)
{
    PyObject *_numpy_dtype_ctypes;
    PyObject *res;

    /* Check if the given type is a ctypes subclass */
    if (!npy_ctypes_check(type)) {
        /* Increment reference count and return Py_NotImplemented */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }

    /* Import the numpy._core._dtype_ctypes module */
    _numpy_dtype_ctypes = PyImport_ImportModule("numpy._core._dtype_ctypes");
    if (_numpy_dtype_ctypes == NULL) {
        return NULL;  /* Return NULL on import failure */
    }
    /* Call dtype_from_ctypes_type function from _numpy_dtype_ctypes module */
    res = PyObject_CallMethod(_numpy_dtype_ctypes, "dtype_from_ctypes_type", "O", (PyObject *)type);
    Py_DECREF(_numpy_dtype_ctypes);  /* Decrement reference count of module */

    if (res == NULL) {
        return NULL;  /* Return NULL if call fails */
    }

    /*
     * Sanity check that dtype_from_ctypes_type returned the correct type,
     * since an incorrect return type could lead to segfaults.
     */
    if (!PyObject_TypeCheck(res, &PyArrayDescr_Type)) {
        Py_DECREF(res);  /* Decrement reference count if type check fails */
        PyErr_BadInternalCall();  /* Raise internal call error */
        return NULL;  /* Return NULL on error */
    }

    return (PyArray_Descr *)res;  /* Return dtype object */
}

/*
 * This function creates a dtype object when the object has a "dtype" attribute,
 * and it can be converted to a dtype object.
 *
 * Returns `Py_NotImplemented` if this is not possible.
 * Currently the only failure mode for a NULL return is a RecursionError.
 */
static PyArray_Descr *
_try_convert_from_dtype_attr(PyObject *obj)
{
    /* Attempt to retrieve the "dtype" attribute from the object */
    PyObject *dtypedescr = PyObject_GetAttrString(obj, "dtype");
    if (dtypedescr == NULL) {
        /*
         * Handle case where fetching the attribute fails, possibly due to
         * recursion limit being hit.
         */
        goto fail;
    }
    /* Continue function implementation here */
    /* 如果给定的数据类型描述符是一个有效的数组描述符，则直接返回它 */
    if (PyArray_DescrCheck(dtypedescr)) {
        /* `dtype` 属性已经是一个有效的描述符 */
        return (PyArray_Descr *)dtypedescr;
    }

    /* 进入递归调用保护区域，用于转换给定数据类型的`.dtype`属性 */
    if (Py_EnterRecursiveCall(
            " while trying to convert the given data type from its "
            "`.dtype` attribute.") != 0) {
        Py_DECREF(dtypedescr);
        return NULL;
    }

    /* 调用_convert_from_any函数进行数据类型转换 */
    PyArray_Descr *newdescr = _convert_from_any(dtypedescr, 0);
    Py_DECREF(dtypedescr);
    Py_LeaveRecursiveCall();

    /* 如果转换失败，则跳转到标签fail处处理 */
    if (newdescr == NULL) {
        goto fail;
    }

    /* 在2021年1月5日被弃用，NumPy 1.21版本开始 */
    /* 发出弃用警告，说明`.dtype`属性必须是有效的dtype实例，可能需要通过`np.dtype(data_type.dtype)`强制转换 */
    if (DEPRECATE("in the future the `.dtype` attribute of a given data"
                  "type object must be a valid dtype instance. "
                  "`data_type.dtype` may need to be coerced using "
                  "`np.dtype(data_type.dtype)`. (Deprecated NumPy 1.20)") < 0) {
        Py_DECREF(newdescr);
        return NULL;
    }

    /* 返回新的数据类型描述符 */
    return newdescr;

  fail:
    /* 处理失败的情况，忽略除递归错误外的所有异常，以便给ctypes一个充分的尝试机会 */
    if (!PyErr_ExceptionMatches(PyExc_RecursionError)) {
        PyErr_Clear();
        /* 返回Py_NotImplemented对象，表示操作未实现 */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }
    return NULL;
/* Expose to another file with a prefixed name */
/* 将此函数暴露给带有前缀名称的另一个文件 */
NPY_NO_EXPORT PyArray_Descr *
_arraydescr_try_convert_from_dtype_attr(PyObject *obj)
{
    /* 调用内部函数_try_convert_from_dtype_attr处理对象 */
    return _try_convert_from_dtype_attr(obj);
}

/*
 * Sets the global typeDict object, which is a dictionary mapping
 * dtype names to numpy scalar types.
 */
/*
 * 设置全局的typeDict对象，这是一个将dtype名称映射到numpy标量类型的字典。
 */
NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *dict;

    /* 尝试解析参数元组，获取字典对象 */
    if (!PyArg_ParseTuple(args, "O:set_typeDict", &dict)) {
        return NULL;
    }
    /* 减少旧的引用（如果有的话） */
    Py_XDECREF(typeDict);
    /* 将typeDict设置为新的字典对象 */
    typeDict = dict;
    /* 创建对字典对象的内部引用 */
    Py_INCREF(dict);
    /* 返回None对象 */
    Py_RETURN_NONE;
}

#define _chk_byteorder(arg) (arg == '>' || arg == '<' ||        \
                             arg == '|' || arg == '=')

static int
_check_for_commastring(const char *type, Py_ssize_t len)
{
    Py_ssize_t i;
    int sqbracket;

    /* 检查字符串开头是否为整数 */
    if ((type[0] >= '0'
                && type[0] <= '9')
            || ((len > 1)
                && _chk_byteorder(type[0])
                && (type[1] >= '0'
                && type[1] <= '9'))) {
        return 1;
    }
    /* 检查空元组 */
    if (((len > 1)
                && (type[0] == '('
                && type[1] == ')'))
            || ((len > 3)
                && _chk_byteorder(type[0])
                && (type[1] == '('
                && type[2] == ')'))) {
        return 1;
    }
    /*
     * 检查方括号 [] 外的逗号存在。这允许参数化dtype中的逗号。
     */
    sqbracket = 0;
    for (i = 0; i < len; i++) {
        switch (type[i]) {
            case ',':
                if (sqbracket == 0) {
                    return 1;
                }
                break;
            case '[':
                ++sqbracket;
                break;
            case ']':
                --sqbracket;
                break;
        }
    }
    return 0;
}

#undef _chk_byteorder

static int
is_datetime_typestr(char const *type, Py_ssize_t len)
{
    /* 检查长度小于2的情况 */
    if (len < 2) {
        return 0;
    }
    /* 检查是否以'M8'或'm8'开头 */
    if (type[1] == '8' && (type[0] == 'M' || type[0] == 'm')) {
        return 1;
    }
    /* 检查长度小于10的情况 */
    if (len < 10) {
        return 0;
    }
    /* 检查是否以"datetime64"开头 */
    if (strncmp(type, "datetime64", 10) == 0) {
        return 1;
    }
    /* 检查长度小于11的情况 */
    if (len < 11) {
        return 0;
    }
    /* 检查是否以"timedelta64"开头 */
    if (strncmp(type, "timedelta64", 11) == 0) {
        return 1;
    }
    return 0;
}

static PyArray_Descr *
_convert_from_tuple(PyObject *obj, int align)
{
    /* 检查元组大小是否为2 */
    if (PyTuple_GET_SIZE(obj) != 2) {
        PyErr_Format(PyExc_TypeError,
            "Tuple must have size 2, but has size %zd",
            PyTuple_GET_SIZE(obj));
        return NULL;
    }
    /* 尝试将第一个元素解析为类型 */
    PyArray_Descr *type = _convert_from_any(PyTuple_GET_ITEM(obj, 0), align);
    if (type == NULL) {
        return NULL;
    }
    /* 获取元组的第二个元素 */
    PyObject *val = PyTuple_GET_ITEM(obj,1);
    /* 尝试将下一个元素解释为类型 */
    // 尝试将给定类型和值转换为描述符对象
    PyArray_Descr *res = _try_convert_from_inherit_tuple(type, val);
    
    // 如果转换成功，返回描述符对象，释放类型对象并结束函数
    if ((PyObject *)res != Py_NotImplemented) {
        Py_DECREF(type);
        return res;
    }
    
    // 若转换失败但未崩溃，则继续执行以下代码

    // 如果类型对象是无大小的数据类型
    if (PyDataType_ISUNSIZED(type)) {
        // 尝试解析元组中的下一个项作为类型大小
        int itemsize = PyArray_PyIntAsInt(PyTuple_GET_ITEM(obj,1));

        // 如果解析出错，设置异常并释放类型对象并返回空
        if (error_converting(itemsize)) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid itemsize in generic type tuple");
            Py_DECREF(type);
            return NULL;
        }
        
        // 替换类型对象的描述符，如果失败则返回空
        PyArray_DESCR_REPLACE(type);
        if (type == NULL) {
            return NULL;
        }
        
        // 如果类型是 NPY_UNICODE，设置元素大小为 itemsize 左移 2 位
        if (type->type_num == NPY_UNICODE) {
            type->elsize = itemsize << 2;
        }
        else {
            type->elsize = itemsize;
        }
        
        // 返回修改后的类型对象
        return type;
    }
    // 如果类型对象有元数据并且值是字典或字典代理类型
    else if (type->metadata && (PyDict_Check(val) || PyDictProxy_Check(val))) {
        // 假设值是元数据字典，尝试将其合并到类型对象的元数据中
        if (PyDict_Merge(type->metadata, val, 0) == -1) {
            Py_DECREF(type);
            return NULL;
        }
        
        // 返回更新后的类型对象
        return type;
    }
    
    // 如果执行到这里，意味着发生了错误，清理并返回空
    fail:
        Py_XDECREF(type);
        npy_free_cache_dim_obj(shape);
        return NULL;
}

/*
 * obj is a list.  Each item is a tuple with
 *
 * (field-name, data-type (either a list or a string), and an optional
 * shape parameter).
 *
 * field-name can be a string or a 2-tuple
 * data-type can now be a list, string, or 2-tuple
 *          (string, metadata dictionary)
 */
static PyArray_Descr *
_convert_from_array_descr(PyObject *obj, int align)
{
    // 获取列表 obj 的长度
    int n = PyList_GET_SIZE(obj);
    // 创建一个元组 nameslist，长度为 n
    PyObject *nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }

    /* Types with fields need the Python C API for field access */
    // 定义 dtypeflags 标志，需要使用 Python C API
    char dtypeflags = NPY_NEEDS_PYAPI;
    // 初始化 maxalign 和 totalsize
    int maxalign = 1;
    int totalsize = 0;
    // 创建一个空的字典 fields
    PyObject *fields = PyDict_New();
    if (!fields) {
        return NULL;
    }
    // 此处多余的右括号应删除
    }

    // 如果 maxalign 大于 1，则调整 totalsize 为下一个对齐偏移量
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }

    // 创建一个新的 _PyArray_LegacyDescr 结构体，类型为 NPY_VOID
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    if (new == NULL) {
        goto fail;
    }
    // 设置新结构体的字段和名称列表等属性
    new->fields = fields;
    new->names = nameslist;
    new->elsize = totalsize;
    new->flags = dtypeflags;

    /* Structured arrays get a sticky aligned bit */
    // 如果 align 为真，设置新结构体的 flags 包括 NPY_ALIGNED_STRUCT
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
        new->alignment = maxalign;
    }
    // 返回 PyArray_Descr 指针类型的 new 结构体
    return (PyArray_Descr *)new;

 fail:
    // 失败时释放 fields 和 nameslist，返回 NULL
    Py_DECREF(fields);
    Py_DECREF(nameslist);
    return NULL;

}

/*
 * a list specifying a data-type can just be
 * a list of formats.  The names for the fields
 * will default to f0, f1, f2, and so forth.
 */
static PyArray_Descr *
_convert_from_list(PyObject *obj, int align)
{
    // 获取列表 obj 的长度
    int n = PyList_GET_SIZE(obj);
    /*
     * Ignore any empty string at end which _internal._commastring
     * can produce
     */
    // 获取列表 obj 中的最后一个元素
    PyObject *last_item = PyList_GET_ITEM(obj, n-1);
    // 如果最后一个元素是字符串类型
    if (PyUnicode_Check(last_item)) {
        // 获取字符串的长度
        Py_ssize_t s = PySequence_Size(last_item);
        if (s < 0) {
            return NULL;
        }
        // 如果字符串长度为 0，将列表长度 n 减 1
        if (s == 0) {
            n = n - 1;
        }
    }
    // 如果列表长度为 0，抛出值错误异常
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Expected at least one field name");
        return NULL;
    }
    // 创建一个元组 nameslist，长度为 n
    PyObject *nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    // 创建一个空的字典 fields
    PyObject *fields = PyDict_New();
    if (!fields) {
        Py_DECREF(nameslist);
        return NULL;
    }

    /* Types with fields need the Python C API for field access */
    // 定义 dtypeflags 标志，需要使用 Python C API
    char dtypeflags = NPY_NEEDS_PYAPI;
    // 初始化 maxalign 和 totalsize
    int maxalign = 1;
    int totalsize = 0;
    for (int i = 0; i < n; i++) {
        // 从 Python 列表中获取第 i 个元素，并尝试将其转换为 PyArray_Descr 结构
        PyArray_Descr *conv = _convert_from_any(
                PyList_GET_ITEM(obj, i), align);
        // 如果转换失败（返回 NULL），跳转到错误处理标签
        if (conv == NULL) {
            goto fail;
        }
        // 将 conv 结构中的 flags 与 NPY_FROM_FIELDS 按位或，更新 dtypeflags
        dtypeflags |= (conv->flags & NPY_FROM_FIELDS);
        // 如果需要按照指定对齐方式对齐
        if (align) {
            // 获取 conv 结构的对齐值
            int _align = conv->alignment;
            // 如果对齐值大于 1，则按照该对齐值对 totalsize 进行向上对齐
            if (_align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            // 更新 maxalign 为当前计算出的对齐值和已有 maxalign 的最大值
            maxalign = PyArray_MAX(maxalign, _align);
        }
        // 根据 totalsize 创建一个 PyLong 对象作为 size_obj
        PyObject *size_obj = PyLong_FromLong((long) totalsize);
        // 如果创建失败，释放 conv 并跳转到错误处理标签
        if (!size_obj) {
            Py_DECREF(conv);
            goto fail;
        }
        // 创建一个新的元组对象 tup，包含 conv 和 size_obj
        PyObject *tup = PyTuple_New(2);
        // 如果创建失败，释放 size_obj 和 conv，并跳转到错误处理标签
        if (!tup) {
            Py_DECREF(size_obj);
            Py_DECREF(conv);
            goto fail;
        }
        // 设置 tup 的第一个元素为 conv，第二个元素为 size_obj
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        PyTuple_SET_ITEM(tup, 1, size_obj);
        // 根据索引 i 创建一个新的 key，格式为 "f%d"
        PyObject *key = PyUnicode_FromFormat("f%d", i);
        // 如果创建失败，释放 tup，并跳转到错误处理标签
        if (!key) {
            Py_DECREF(tup);
            goto fail;
        }
        // 将 key 添加到 nameslist 中，同时 key 的引用计数会增加（"steals a reference to key"）
        PyTuple_SET_ITEM(nameslist, i, key);
        // 将 tup 添加到 fields 字典中，key 为 key
        int ret = PyDict_SetItem(fields, key, tup);
        // 释放 tup，并如果 PyDict_SetItem 返回负值（失败），跳转到错误处理标签
        Py_DECREF(tup);
        if (ret < 0) {
            goto fail;
        }
        // 更新 totalsize，增加 conv 的元素大小
        totalsize += conv->elsize;
    }
    // 创建一个新的 _PyArray_LegacyDescr 结构，类型为 NPY_VOID
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    // 如果创建失败，跳转到错误处理标签
    if (new == NULL) {
        goto fail;
    }
    // 设置 new 结构的 fields、names 和 flags
    new->fields = fields;
    new->names = nameslist;
    new->flags = dtypeflags;
    // 如果最大对齐值大于 1，按照 maxalign 对 totalsize 进行向上对齐
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    // 如果需要对结构数组进行对齐，设置 NPY_ALIGNED_STRUCT 标志并更新对齐值
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
        new->alignment = maxalign;
    }
    // 设置 new 结构的元素大小为 totalsize，并返回该结构的指针类型
    new->elsize = totalsize;
    return (PyArray_Descr *)new;

 fail:
    // 错误处理标签：释放 nameslist 和 fields，并返回 NULL
    Py_DECREF(nameslist);
    Py_DECREF(fields);
    return NULL;
/*
 * helper function for _try_convert_from_inherit_tuple to disallow dtypes of the form
 * (old_dtype, new_dtype) where either of the dtypes contains python
 * objects - these dtypes are not useful and can be a source of segfaults,
 * when an attempt is made to interpret a python object as a different dtype
 * or vice versa
 * an exception is made for dtypes of the form ('O', [('name', 'O')]), which
 * people have been using to add a field to an object array without fields
 */
static int
_validate_union_object_dtype(_PyArray_LegacyDescr *new, _PyArray_LegacyDescr *conv)
{
    PyObject *name, *tup;
    PyArray_Descr *dtype;

    // 如果两个描述符都不是引用检查，说明它们不包含Python对象
    if (!PyDataType_REFCHK((PyArray_Descr *)new)
            && !PyDataType_REFCHK((PyArray_Descr *)conv)) {
        return 0;
    }
    // 如果新描述符有字段或者种类不是 'O'（Python对象），则验证失败
    if (PyDataType_HASFIELDS(new) || new->kind != 'O') {
        goto fail;
    }
    // 如果旧描述符没有字段或字段元组大小不为1，则验证失败
    if (!PyDataType_HASFIELDS(conv) || PyTuple_GET_SIZE(conv->names) != 1) {
        goto fail;
    }
    // 获取字段名
    name = PyTuple_GET_ITEM(conv->names, 0);
    if (name == NULL) {
        return -1;
    }
    // 从字段字典中获取名字对应的项
    tup = PyDict_GetItemWithError(conv->fields, name);
    # 如果元组对象 `tup` 是 NULL，表示传入的参数未正确获取，需要进行错误处理
    if (tup == NULL) {
        # 如果没有已设置的 Python 异常状态，说明字段名未正确包含在字段中
        /* fields was missing the name it claimed to contain */
        PyErr_BadInternalCall();
        # 返回错误码 -1
        return -1;
    }
    # 从元组 `tup` 中获取第一个元素，应该是一个 PyArray_Descr 对象指针
    dtype = (PyArray_Descr *)PyTuple_GET_ITEM(tup, 0);
    # 如果获取的 `dtype` 是 NULL，说明获取失败，返回错误码 -1
    if (dtype == NULL) {
        return -1;
    }
    # 如果 `dtype` 的类型不是对象（'O'），跳转到错误处理标签 `fail`
    if (dtype->kind != 'O') {
        goto fail;
    }
    # 如果以上条件都没有问题，则返回成功标志 0
    return 0;
/*
 * Try to convert a descriptor from an inherit tuple to a legacy descriptor.
 * This function handles cases where the new data-type inherits from the old
 * data-type, ensuring compatibility or raising appropriate errors if not.
 *
 * Parameters:
 * - type: The original data descriptor to convert from.
 * - newobj: The new object descriptor to convert to.
 *
 * Returns:
 * - A new PyArray_Descr object if successful, or Py_NotImplemented if conversion fails.
 */

static PyArray_Descr *
_try_convert_from_inherit_tuple(PyArray_Descr *type, PyObject *newobj)
{
    // Check if newobj is a scalar or tuple of integers
    if (PyArray_IsScalar(newobj, Integer) || _is_tuple_of_integers(newobj)) {
        /* It's a subarray or flexible type instead */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }

    // Attempt to convert newobj to a legacy descriptor
    _PyArray_LegacyDescr *conv = (_PyArray_LegacyDescr *)_convert_from_any(newobj, 0);
    if (conv == NULL) {
        /* Conversion failed, let someone else handle it */
        PyErr_Clear();
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }

    // Ensure both type and conv are legacy descriptors
    if (!PyDataType_ISLEGACY(type) || !PyDataType_ISLEGACY(conv)) {
        /* New-style DTypes not supported */
        Py_DECREF(conv);
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }

    // Create a new legacy descriptor based on type
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNew(type);
    if (new == NULL) {
        // Handle memory allocation failure
        goto fail;
    }

    // Handle unsized type or mismatched elsize
    if (PyDataType_ISUNSIZED(new)) {
        new->elsize = conv->elsize;
    } else if (new->elsize != conv->elsize) {
        // Error if elsize mismatch
        PyErr_SetString(PyExc_ValueError,
                "mismatch in size of old and new data-descriptor");
        Py_DECREF(new);
        goto fail;
    } else if (_validate_union_object_dtype(new, conv) < 0) {
        // Validate union object data type
        Py_DECREF(new);
        goto fail;
    }

    // Copy fields and metadata from conv to new
    if (PyDataType_HASFIELDS(conv)) {
        Py_XDECREF(new->fields);
        new->fields = conv->fields;
        Py_XINCREF(new->fields);

        Py_XDECREF(new->names);
        new->names = conv->names;
        Py_XINCREF(new->names);
    }
    if (conv->metadata != NULL) {
        Py_XDECREF(new->metadata);
        new->metadata = conv->metadata;
        Py_XINCREF(new->metadata);
    }

fail:
    // Handle failure by setting appropriate error message
    PyErr_SetString(PyExc_ValueError,
            "dtypes of the form (old_dtype, new_dtype) containing the object "
            "dtype are not supported");
    return -1;
}
    /*
     * Certain flags must be inherited from the fields.  This is needed
     * only for void dtypes (or subclasses of it such as a record dtype).
     * For other dtypes, the field part will only be used for direct field
     * access and thus flag inheritance should not be necessary.
     * (We only allow object fields if the dtype is object as well.)
     * This ensures copying over of the NPY_FROM_FIELDS "inherited" flags.
     */
    如果新的数据类型是 NPY_VOID 类型：
        // 将新的数据类型的标志位设置为从 conv 中获得的标志位
        new->flags = conv->flags;
    
    // 释放 conv 对象的引用计数
    Py_DECREF(conv);
    // 返回一个新的 PyArray_Descr 指针，即 new 指针强制转换
    return (PyArray_Descr *)new;

 fail:
    // 在发生错误时释放 conv 对象的引用计数
    Py_DECREF(conv);
    // 返回 NULL 表示操作失败
    return NULL;
/*
 * Validates that any field of the structured array 'dtype' which has
 * the NPY_ITEM_HASOBJECT flag set does not overlap with another field.
 *
 * This algorithm is worst case O(n^2). It could be done with a sort
 * and sweep algorithm, but the structured dtype representation is
 * rather ugly right now, so writing something better can wait until
 * that representation is made sane.
 *
 * Returns 0 on success, -1 if an exception is raised.
 */
static int
_validate_object_field_overlap(_PyArray_LegacyDescr *dtype)
{
    PyObject *names, *fields, *key, *tup, *title;
    Py_ssize_t i, j, names_size;
    PyArray_Descr *fld_dtype, *fld2_dtype;
    int fld_offset, fld2_offset;

    /* Get some properties from the dtype */
    names = dtype->names;  // 获取 dtype 中的字段名元组
    names_size = PyTuple_GET_SIZE(names);  // 获取字段名元组的大小
    fields = dtype->fields;  // 获取 dtype 中的字段字典

    // 循环遍历所有字段名
    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);  // 获取当前字段名
        if (key == NULL) {
            return -1;  // 如果字段名为空，返回异常
        }
        tup = PyDict_GetItemWithError(fields, key);  // 从字段字典中获取字段元组
        if (tup == NULL) {
            if (!PyErr_Occurred()) {
                /* fields was missing the name it claimed to contain */
                PyErr_BadInternalCall();  // 如果字段字典中缺少相应字段名，抛出异常
            }
            return -1;
        }
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &fld_offset, &title)) {
            return -1;  // 解析字段元组失败，返回异常
        }

        /* If this field has objects, check for overlaps */
        if (PyDataType_REFCHK(fld_dtype)) {  // 如果字段类型包含引用类型，进行重叠检查
            // 再次遍历所有字段名
            for (j = 0; j < names_size; ++j) {
                if (i != j) {
                    key = PyTuple_GET_ITEM(names, j);  // 获取另一个字段名
                    if (key == NULL) {
                        return -1;  // 如果字段名为空，返回异常
                    }
                    tup = PyDict_GetItemWithError(fields, key);  // 获取另一个字段元组
                    if (tup == NULL) {
                        if (!PyErr_Occurred()) {
                            /* fields was missing the name it claimed to contain */
                            PyErr_BadInternalCall();  // 如果字段字典中缺少相应字段名，抛出异常
                        }
                        return -1;
                    }
                    if (!PyArg_ParseTuple(tup, "Oi|O", &fld2_dtype,
                                                &fld2_offset, &title)) {
                        return -1;  // 解析另一个字段元组失败，返回异常
                    }
                    /* Raise an exception if it overlaps */
                    // 如果两个字段有重叠部分，抛出异常
                    if (fld_offset < fld2_offset + fld2_dtype->elsize &&
                                fld2_offset < fld_offset + fld_dtype->elsize) {
                        PyErr_SetString(PyExc_TypeError,
                                "Cannot create a NumPy dtype with overlapping "
                                "object fields");
                        return -1;
                    }
                }
            }
        }
    }

    /* It passed all the overlap tests */
    // 所有重叠检查通过，返回成功
    return 0;
}
/*
 * 创建一个空的 Python 字典对象，用于存储字段信息
 */
PyObject *fields = PyDict_New();
if (fields == NULL) {
    // 如果内存分配失败，返回内存错误异常
    return (PyArray_Descr *)PyErr_NoMemory();
}

/*
 * 使用 PyMapping_GetItemString 函数获取字典对象中的 "names" 键对应的值
 * 这支持 dictproxy 对象的访问方式
 */
PyObject *names = PyMapping_GetItemString(obj, "names");
if (names == NULL) {
    // 如果找不到 "names" 键，清理已分配的资源并调用 _convert_from_field_dict 函数处理
    Py_DECREF(fields);
    PyErr_Clear();  // 清理异常状态
    return _convert_from_field_dict(obj, align);
}

/*
 * 使用 PyMapping_GetItemString 函数获取字典对象中的 "formats" 键对应的值
 */
PyObject *descrs = PyMapping_GetItemString(obj, "formats");
if (descrs == NULL) {
    // 如果找不到 "formats" 键，清理已分配的资源并调用 _convert_from_field_dict 函数处理
    Py_DECREF(fields);
    PyErr_Clear();  // 清理异常状态
    Py_DECREF(names);  // 清理已获取的 "names" 值
    return _convert_from_field_dict(obj, align);
}

// 获取 names 列表的长度
int n = PyObject_Length(names);

/*
 * 使用 PyMapping_GetItemString 函数获取字典对象中的 "offsets" 键对应的值
 * 如果找不到该键，清理异常状态
 */
PyObject *offsets = PyMapping_GetItemString(obj, "offsets");
if (!offsets) {
    PyErr_Clear();  // 清理异常状态
}

/*
 * 使用 PyMapping_GetItemString 函数获取字典对象中的 "titles" 键对应的值
 * 如果找不到该键，清理异常状态
 */
PyObject *titles = PyMapping_GetItemString(obj, "titles");
if (!titles) {
    PyErr_Clear();  // 清理异常状态
}
    # 检查 'names', 'formats', 'offsets', 'titles' 四个字典的长度是否一致，若不一致则抛出异常并跳转到失败处理标签
    if ((n > PyObject_Length(descrs))
        || (offsets && (n > PyObject_Length(offsets)))
        || (titles && (n > PyObject_Length(titles)))) {
        PyErr_SetString(PyExc_ValueError,
                "'names', 'formats', 'offsets', and 'titles' dict "
                "entries must have the same length");
        goto fail;
    }

    """
    如果字典中有属性 'aligned'，则根据其值覆盖 align 变量的设置，如果该属性不存在则忽略。
    如果 'aligned' 的值不是 True 或 False，则抛出异常并跳转到失败处理标签。
    """
    PyObject *tmp = PyMapping_GetItemString(obj, "aligned");
    if (tmp == NULL) {
        PyErr_Clear();
    } else {
        if (tmp == Py_True) {
            align = 1;
        }
        else if (tmp != Py_False) {
            Py_DECREF(tmp);
            PyErr_SetString(PyExc_ValueError,
                    "NumPy dtype descriptor includes 'aligned' entry, "
                    "but its value is neither True nor False");
            goto fail;
        }
        Py_DECREF(tmp);
    }

    """
    对于包含字段的数据类型，需要使用 Python C API 进行字段访问。
    """
    char dtypeflags = NPY_NEEDS_PYAPI;
    int totalsize = 0;
    int maxalign = 1;
    int has_out_of_order_fields = 0;
    }

    """
    创建一个新的 NumPy 数据类型描述符对象 new，并根据情况设置对齐方式和大小。
    如果创建失败，则跳转到失败处理标签。
    """
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    if (new == NULL) {
        goto fail;
    }
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    if (!PyTuple_Check(names)) {
        Py_SETREF(names, PySequence_Tuple(names));
        if (names == NULL) {
            Py_DECREF(new);
            goto fail;
        }
    }
    new->names = names;
    new->fields = fields;
    new->flags = dtypeflags;
    """
    new 对象接管了 names 和 fields 对象的 DECREF 责任。
    """
    names = NULL;
    fields = NULL;

    """
    如果字段不是按顺序排列，并且存在 OBJECT 类型，则需要验证 OBJECT 类型是否与其他类型重叠。
    """
    if (has_out_of_order_fields && PyDataType_REFCHK((PyArray_Descr *)new)) {
        if (_validate_object_field_overlap(new) < 0) {
            Py_DECREF(new);
            goto fail;
        }
    }

    """
    对于结构化数组，设置 NPY_ALIGNED_STRUCT 标志位。
    """
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
    }

    """
    如果提供了 'itemsize' 属性，则覆盖 new 的 itemsize 值。
    """
    tmp = PyMapping_GetItemString(obj, "itemsize");
    if (tmp == NULL) {
        PyErr_Clear();
    } else {
        // 从临时对象 tmp 中获取 itemsize，并转换为整数类型
        int itemsize = (int)PyArray_PyIntAsInt(tmp);
        // 释放 tmp 对象的引用计数
        Py_DECREF(tmp);
        // 如果转换出错，则释放新创建的 PyArray_Descr 对象并跳转到失败标签
        if (error_converting(itemsize)) {
            Py_DECREF(new);
            goto fail;
        }
        /* 确保 itemsize 不会比 new->elsize 小 */
        if (itemsize < new->elsize) {
            // 抛出数值错误异常，提示无法将 itemsize 覆盖为比 new->elsize 更小的值
            PyErr_Format(PyExc_ValueError,
                    "NumPy dtype descriptor requires %d bytes, "
                    "cannot override to smaller itemsize of %d",
                    new->elsize, itemsize);
            Py_DECREF(new);
            goto fail;
        }
        /* 如果设置了 align，确保对齐方式能整除 itemsize */
        if (align && new->alignment > 0 && itemsize % new->alignment != 0) {
            // 抛出数值错误异常，提示指定的对齐方式无法整除指定的 itemsize
            PyErr_Format(PyExc_ValueError,
                    "NumPy dtype descriptor requires alignment of %d bytes, "
                    "which is not divisible into the specified itemsize %d",
                    new->alignment, itemsize);
            Py_DECREF(new);
            goto fail;
        }
        /* 设置新的 itemsize */
        new->elsize = itemsize;
    }

    /* 如果提供了 metadata，则添加到新创建的 PyArray_Descr 对象中 */
    PyObject *metadata = PyMapping_GetItemString(obj, "metadata");

    if (metadata == NULL) {
        // 清除当前异常状态
        PyErr_Clear();
    }
    else if (new->metadata == NULL) {
        // 如果当前 metadata 为空，则将其赋值给新创建的 PyArray_Descr 对象的 metadata
        new->metadata = metadata;
    }
    else {
        // 将 metadata 合并到 new->metadata 中，如果出错则释放新创建的 PyArray_Descr 对象并跳转到失败标签
        int ret = PyDict_Merge(new->metadata, metadata, 0);
        Py_DECREF(metadata);
        if (ret < 0) {
            Py_DECREF(new);
            goto fail;
        }
    }

    // 释放其他的临时对象的引用计数
    Py_XDECREF(fields);
    Py_XDECREF(names);
    Py_XDECREF(descrs);
    Py_XDECREF(offsets);
    Py_XDECREF(titles);
    // 返回新创建的 PyArray_Descr 对象的指针
    return (PyArray_Descr *)new;

 fail:
    // 如果发生失败，则释放所有的临时对象的引用计数并返回 NULL
    Py_XDECREF(fields);
    Py_XDECREF(names);
    Py_XDECREF(descrs);
    Py_XDECREF(offsets);
    Py_XDECREF(titles);
    return NULL;
/*NUMPY_API*/
// 定义一个函数 PyArray_DescrNewFromType，根据给定的类型编号创建并返回一个新的描述符对象
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewFromType(int type_num)
{
    // 声明两个指向 PyArray_Descr 结构的指针变量
    PyArray_Descr *old;
    PyArray_Descr *new;

    // 调用 PyArray_DescrFromType 函数获取给定类型编号对应的描述符对象
    old = PyArray_DescrFromType(type_num);
    // 如果获取失败，返回空指针
    if (old == NULL) {
        return NULL;
    }
    // 使用获取到的描述符对象创建一个新的描述符对象
    new = PyArray_DescrNew(old);
    // 减少旧描述符对象的引用计数
    Py_DECREF(old);
    // 返回新创建的描述符对象
    return new;
}

/*NUMPY_API
 * 从对象获取类型编号，如果是 None，则返回 NULL
 */
// 定义一个函数 PyArray_DescrConverter2，用于从对象中获取类型描述符
NPY_NO_EXPORT int
PyArray_DescrConverter2(PyObject *obj, PyArray_Descr **at)
{
    // 如果传入的对象是 None，则将输出的描述符指针设置为 NULL，并返回成功状态码
    if (obj == Py_None) {
        *at = NULL;
        return NPY_SUCCEED;
    }
    else {
        // 否则调用 PyArray_DescrConverter 函数来获取描述符
        return PyArray_DescrConverter(obj, at);
    }
}

/**
 * 检查描述符是否是传统的“灵活”DType实例，这通常是指未附加到数组的实例，比如长度为0的字符串或没有单位的日期时间。
 * 这些实例应该已经被大部分废弃，通常只表示大多数“dtype”参数的 DType 类。
 *
 * TODO: 这个函数最终应该收到弃用警告并被移除。
 *
 * @param descr 要检查的描述符
 * @return 如果不是具体的 dtype 实例则返回 1，否则返回 0
 */
// 定义一个静态函数 descr_is_legacy_parametric_instance，用于检查描述符是否是传统的“灵活”DType实例
static int
descr_is_legacy_parametric_instance(PyArray_Descr *descr,
                                    PyArray_DTypeMeta *DType)
{
    // 如果不是传统的 DType 实例，则返回 0
    if (!NPY_DT_is_legacy(DType)) {
        return 0;
    }

    // 如果是无大小的数据类型，则返回 1
    if (PyDataType_ISUNSIZED(descr)) {
        return 1;
    }
    // 如果是带有通用时间单位的灵活描述符
    if (PyDataType_ISDATETIME(descr)) {
        // 获取日期时间元数据
        PyArray_DatetimeMetaData *meta;
        meta = get_datetime_metadata_from_dtype(descr);
        // 如果基本单位是通用的，则返回 1
        if (meta->base == NPY_FR_GENERIC) {
            return 1;
        }
    }
    // 否则返回 0
    return 0;
}

/**
 * 给定一个描述符（dtype 实例），处理将传统的灵活“无大小”描述符转换为它们的 DType。返回的 DType 和描述符都可以为空（如果输入为空）。但是当设置描述符时总是设置 DType。
 *
 * @param dtype 要处理的描述符
 * @param out_descr 输出的描述符指针
 * @param out_DType 输出的 DType 指针
 * @return 成功返回 0，失败返回 -1
 */
// 定义一个函数 PyArray_ExtractDTypeAndDescriptor，用于从描述符中提取 DType 和描述符
NPY_NO_EXPORT int
PyArray_ExtractDTypeAndDescriptor(PyArray_Descr *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType)
{
    // 将输出的 DType 和描述符初始化为空
    *out_DType = NULL;
    *out_descr = NULL;

    // 如果输入的描述符不为空
    if (dtype != NULL) {
        // 获取描述符对应的 DType
        *out_DType = NPY_DTYPE(dtype);
        // 增加 DType 的引用计数
        Py_INCREF(*out_DType);
        // 如果描述符不是传统的灵活参数化实例
        if (!descr_is_legacy_parametric_instance((PyArray_Descr *)dtype,
                                                    *out_DType)) {
            // 设置输出的描述符
            *out_descr = (PyArray_Descr *)dtype;
            // 增加描述符的引用计数
            Py_INCREF(*out_descr);
        }
    }
    // 返回成功状态码
    return 0;
}
/**
 * Convert a Python object to a numpy dtype descriptor or class.
 * This function fills the npy_dtype_info structure with the result
 * on success.
 *
 * @param obj Python object representing a dtype instance (descriptor) or DType class.
 * @param[out] dt_info Pointer to npy_dtype_info struct where the dtype class and
 *                    dtype/descriptor instance are filled. If `obj` is None, both
 *                    fields will be NULL. On error, both will remain NULL.
 * @return NPY_SUCCEED (1) on success, NPY_FAIL (0) on failure.
 */
NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterRequired(PyObject *obj, npy_dtype_info *dt_info)
{
    /*
     * Allow dtype classes pass, this could also be generalized to at least
     * some scalar types (right now most of these give instances or)
     */
    
    dt_info->dtype = NULL; // Initialize dtype to NULL
    dt_info->descr = NULL; // Initialize descr to NULL

    if (PyObject_TypeCheck(obj, &PyArrayDTypeMeta_Type)) {
        if (obj == (PyObject *)&PyArrayDescr_Type) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot convert np.dtype into a dtype.");
            return NPY_FAIL; // Return failure if trying to convert np.dtype into dtype
        }
        Py_INCREF(obj); // Increment reference count to obj
        dt_info->dtype = (PyArray_DTypeMeta *)obj; // Set dtype to DTypeMeta object
        dt_info->descr = NULL; // Leave descriptor as NULL
        return NPY_SUCCEED; // Return success
    }

    PyArray_Descr *descr;
    if (PyArray_DescrConverter(obj, &descr) != NPY_SUCCEED) {
        return NPY_FAIL; // Return failure if conversion fails
    }

    /*
     * The above converts e.g. "S" or "S0" to the prototype instance, we make
     * it behave the same as the DType.  This is not fully correct, "S0" should
     * be considered an instance with actual 0 length.
     * TODO: It would be nice to fix that eventually.
     */
    
    int res = PyArray_ExtractDTypeAndDescriptor(
                descr, &dt_info->descr, &dt_info->dtype); // Extract dtype and descriptor
    Py_DECREF(descr); // Decrement reference count to descr
    if (res < 0) {
        return NPY_FAIL; // Return failure if extraction fails
    }
    return NPY_SUCCEED; // Return success
}

/**
 * Optional converter function that initializes npy_dtype_info struct to NULL
 * if obj is None, otherwise delegates to PyArray_DTypeOrDescrConverterRequired.
 *
 * @param obj Python object representing a dtype instance (descriptor) or DType class,
 *            or None.
 * @param[out] dt_info Pointer to npy_dtype_info struct where the dtype class and
 *                    dtype/descriptor instance are filled. If `obj` is None, both
 *                    fields will be NULL. On error, both will remain NULL.
 * @return NPY_SUCCEED (1) if obj is None, or the result of
 *         PyArray_DTypeOrDescrConverterRequired.
 */
NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterOptional(PyObject *obj, npy_dtype_info *dt_info)
{
    if (obj == Py_None) {
        /* caller must have initialized for the optional version */
        return NPY_SUCCEED; // Return success if obj is None
    }
    return PyArray_DTypeOrDescrConverterRequired(obj, dt_info); // Delegate to required converter otherwise
}

/**
 * Given a DType class, returns the default instance (descriptor). This
 * checks for a `singleton` first and only calls the `default_descr` function
 * if necessary.
 *
 * @param DType DType class object from which to retrieve the default instance.
 * @return PyArray_Descr * Pointer to the default instance descriptor on success,
 *         NULL on failure.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_GetDefaultDescr(PyArray_DTypeMeta *DType)
{
    if (DType->singleton != NULL) {
        Py_INCREF(DType->singleton); // Increment reference count to singleton
        return DType->singleton; // Return the singleton if available
    }
    return NPY_DT_CALL_default_descr(DType); // Call default_descr function to retrieve default descriptor
}

/**
 * Get a dtype instance from a python type.
 *
 * @param obj Python object representing a type.
 * @return PyArray_Descr * Pointer to the dtype instance on success,
 *         NULL on failure.
 */
static PyArray_Descr *
_convert_from_type(PyObject *obj) {
    PyTypeObject *typ = (PyTypeObject*)obj;

    if (PyType_IsSubtype(typ, &PyGenericArrType_Type)) {
        return PyArray_DescrFromTypeObject(obj); // Return descriptor from type object if it's a subtype of PyGenericArrType_Type
    }
    else if (typ == &PyLong_Type) {
        return PyArray_DescrFromType(NPY_INTP); // Return descriptor from NPY_INTP if the type is PyLong_Type
    }

    // More conditions can be added here for other types if needed

    /* If obj doesn't match any condition, return NULL */
    return NULL;
}
    else if (typ == &PyFloat_Type) {
        // 如果对象类型为 Python 的 float 类型，则返回对应的双精度浮点类型描述符
        return PyArray_DescrFromType(NPY_DOUBLE);
    }
    else if (typ == &PyComplex_Type) {
        // 如果对象类型为 Python 的 complex 类型，则返回对应的双精度复数类型描述符
        return PyArray_DescrFromType(NPY_CDOUBLE);
    }
    else if (typ == &PyBool_Type) {
        // 如果对象类型为 Python 的 bool 类型，则返回对应的布尔类型描述符
        return PyArray_DescrFromType(NPY_BOOL);
    }
    else if (typ == &PyBytes_Type) {
        /*
         * TODO: This should be deprecated, and have special handling for
         *       dtype=bytes/"S" in coercion: It should not rely on "S0".
         */
        // 如果对象类型为 Python 的 bytes 类型，则返回对应的字符串类型描述符
        return PyArray_DescrFromType(NPY_STRING);
    }
    else if (typ == &PyUnicode_Type) {
        /*
         * TODO: This should be deprecated, and have special handling for
         *       dtype=str/"U" in coercion: It should not rely on "U0".
         */
        // 如果对象类型为 Python 的 unicode 类型，则返回对应的 Unicode 字符串类型描述符
        return PyArray_DescrFromType(NPY_UNICODE);
    }
    else if (typ == &PyMemoryView_Type) {
        // 如果对象类型为 Python 的 memoryview 类型，则返回对应的 void 类型描述符
        return PyArray_DescrFromType(NPY_VOID);
    }
    else if (typ == &PyBaseObject_Type) {
        // 如果对象类型为 Python 的基本对象类型，则返回对应的对象类型描述符
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    else {
        // 尝试从对象的 dtype 属性转换描述符
        PyArray_Descr *ret = _try_convert_from_dtype_attr(obj);
        // 如果成功转换，则返回转换后的描述符
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        Py_DECREF(ret);

        /*
         * Note: this comes after _try_convert_from_dtype_attr because the ctypes
         * type might override the dtype if numpy does not otherwise
         * support it.
         */
        // 尝试从 ctypes 类型尝试转换描述符
        ret = _try_convert_from_ctypes_type(typ);
        // 如果成功转换，则返回转换后的描述符
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        Py_DECREF(ret);

        /*
         * All other classes are treated as object. This can be convenient
         * to convey an intention of using it for a specific python type
         * and possibly allow converting to a new type-specific dtype in the future. It may make sense to
         * only allow this only within `dtype=...` keyword argument context
         * in the future.
         */
        // 所有其他类型都视为对象类型，返回对象类型描述符
        return PyArray_DescrFromType(NPY_OBJECT);
    }
/* 
   static PyArray_Descr * _convert_from_any(PyObject *obj, int align)
   {
       /* default */
       if (obj == Py_None) {
           // 如果输入是 Py_None，则返回默认类型的数组描述符
           return PyArray_DescrFromType(NPY_DEFAULT_TYPE);
       }
       else if (PyArray_DescrCheck(obj)) {
           // 如果输入是数组描述符，则增加其引用计数并返回
           PyArray_Descr *ret = (PyArray_Descr *)obj;
           Py_INCREF(ret);
           return ret;
       }
       else if (PyType_Check(obj)) {
           // 如果输入是类型对象，则调用 _convert_from_type 处理
           return _convert_from_type(obj);
       }
       /* or a typecode string */
       else if (PyBytes_Check(obj)) {
           /* Allow bytes format strings: convert to unicode */
           // 如果输入是字节对象，则将其转换为 Unicode 对象
           PyObject *obj2 = PyUnicode_FromEncodedObject(obj, NULL, NULL);
           if (obj2 == NULL) {
               /* Convert the exception into a TypeError */
               // 如果转换过程中出现异常，将其转换为 TypeError 并返回空指针
               if (PyErr_ExceptionMatches(PyExc_UnicodeDecodeError)) {
                   PyErr_SetString(PyExc_TypeError,
                           "data type not understood");
               }
               return NULL;
           }
           // 调用 _convert_from_str 处理 Unicode 对象
           PyArray_Descr *ret = _convert_from_str(obj2, align);
           Py_DECREF(obj2);
           return ret;
       }
       else if (PyUnicode_Check(obj)) {
           // 如果输入是 Unicode 对象，则调用 _convert_from_str 处理
           return _convert_from_str(obj, align);
       }
       else if (PyTuple_Check(obj)) {
           /* or a tuple */
           // 如果输入是元组，则尝试递归地处理
           if (Py_EnterRecursiveCall(
                   " while trying to convert the given data type from"
                   " a tuple object" ) != 0) {
               return NULL;
           }
           // 调用 _convert_from_tuple 处理元组对象
           PyArray_Descr *ret = _convert_from_tuple(obj, align);
           Py_LeaveRecursiveCall();
           return ret;
       }
       else if (PyList_Check(obj)) {
           /* or a list */
           // 如果输入是列表，则尝试递归地处理
           if (Py_EnterRecursiveCall(
                   " while trying to convert the given data type from"
                   " a list object" ) != 0) {
               return NULL;
           }
           // 调用 _convert_from_array_descr 处理列表对象
           PyArray_Descr *ret = _convert_from_array_descr(obj, align);
           Py_LeaveRecursiveCall();
           return ret;
       }
       else if (PyDict_Check(obj) || PyDictProxy_Check(obj)) {
           /* or a dictionary */
           // 如果输入是字典或字典代理对象，则尝试递归地处理
           if (Py_EnterRecursiveCall(
                   " while trying to convert the given data type from"
                   " a dict object" ) != 0) {
               return NULL;
           }
           // 调用 _convert_from_dict 处理字典对象
           PyArray_Descr *ret = _convert_from_dict(obj, align);
           Py_LeaveRecursiveCall();
           return ret;
       }
       else if (PyArray_Check(obj)) {
           // 如果输入是数组对象，则设置一个类型错误并返回空指针
           PyErr_SetString(PyExc_TypeError, "Cannot construct a dtype from an array");
           return NULL;
       }
   }
*/
    else {
        // 尝试从对象获取描述符
        PyArray_Descr *ret = _try_convert_from_dtype_attr(obj);
        // 如果返回的不是 Py_NotImplemented 对象，则直接返回该描述符
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        // 减少描述符的引用计数
        Py_DECREF(ret);

        /*
         * 注意: 这里放在 _try_convert_from_dtype_attr 后面，
         * 因为 ctypes 类型可能会覆盖 dtype，如果 numpy 没有其他支持的话。
         */
        // 尝试从 ctypes 类型转换
        ret = _try_convert_from_ctypes_type(Py_TYPE(obj));
        // 如果返回的不是 Py_NotImplemented 对象，则直接返回该描述符
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        // 减少描述符的引用计数
        Py_DECREF(ret);

        // 抛出类型错误异常，指示无法解释 obj 作为数据类型
        PyErr_Format(PyExc_TypeError, "Cannot interpret '%R' as a data type", obj);
        // 返回空指针表示失败
        return NULL;
    }
/*NUMPY_API
 * Get typenum from an object -- None goes to NPY_DEFAULT_TYPE
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 *
 * Returns a new reference in *at, but the returned should not be
 * modified as it may be one of the canonical immutable objects or
 * a reference to the input obj.
 */
NPY_NO_EXPORT int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
    // 调用 _convert_from_any 函数将 Python 对象转换为 PyArray_Descr 结构体
    *at = _convert_from_any(obj, 0);
    // 返回转换结果是否成功的状态
    return (*at) ? NPY_SUCCEED : NPY_FAIL;
}

/** Convert a bytestring specification into a dtype */
static PyArray_Descr *
_convert_from_str(PyObject *obj, int align)
{
    /* Check for a string typecode. */
    // 将 Python 对象转换为 UTF-8 字符串及其长度
    Py_ssize_t len = 0;
    char const *type = PyUnicode_AsUTF8AndSize(obj, &len);
    if (type == NULL) {
        return NULL;
    }

    /* Empty string is invalid */
    // 空字符串不合法
    if (len == 0) {
        goto fail;
    }

    /* check for commas present or first (or second) element a digit */
    // 检查字符串中是否包含逗号，或者第一个（或第二个）字符是否是数字
    if (_check_for_commastring(type, len)) {
        // 如果满足条件，调用 _convert_from_commastring 函数进行转换
        return _convert_from_commastring(obj, align);
    }

    /* Process the endian character. '|' is replaced by '='*/
    // 处理字节顺序字符，将 '|' 替换为 '='
    char endian = '=';
    switch (type[0]) {
        case '>':
        case '<':
        case '=':
            endian = type[0];
            ++type;
            --len;
            break;

        case '|':
            endian = '=';
            ++type;
            --len;
            break;
    }

    /* Just an endian character is invalid */
    // 只有字节顺序字符是无效的情况
    if (len == 0) {
        goto fail;
    }

    /* Check for datetime format */
    // 检查是否为日期时间格式
    if (is_datetime_typestr(type, len)) {
        // 如果是日期时间格式，调用 parse_dtype_from_datetime_typestr 函数解析数据类型
        PyArray_Descr *ret = parse_dtype_from_datetime_typestr(type, len);
        if (ret == NULL) {
            return NULL;
        }
        /* ret has byte order '=' at this point */
        // 在此处，ret 的字节顺序为 '='
        if (!PyArray_ISNBO(endian)) {
            ret->byteorder = endian;
        }
        return ret;
    }

    int check_num = NPY_NOTYPE + 10;
    int elsize = 0;
    /* A typecode like 'd' */
    // 类似于 'd' 的类型码
    if (len == 1) {
        /* Python byte string characters are unsigned */
        // Python 字节字符串的字符是无符号的
        check_num = (unsigned char) type[0];
    }
    /* Possibly a kind + size like 'f8' but also could be 'bool' */
    else {
        // 指向类型字符串中类型字符后的结束位置
        char *typeend = NULL;
        // 类型码
        int kind;

        /* 尝试解析整数，并确保其为字符串的剩余部分 */
        errno = 0;
        // 将类型字符串的第一个字符后面的内容解析为长整型数值
        long result = strtol(type + 1, &typeend, 10);
        // 检查是否有解析发生
        npy_bool some_parsing_happened = !(type == typeend);
        // 检查是否整个字符串被消耗
        npy_bool entire_string_consumed = *typeend == '\0';
        // 检查解析是否成功
        npy_bool parsing_succeeded =
                (errno == 0) && some_parsing_happened && entire_string_consumed;
        // 确保数值不会溢出或为负数
        if (result > INT_MAX || result < 0) {
            goto fail;
        }

        // 将解析的整数值作为元素大小
        elsize = (int)result;


        // 如果解析成功并且类型字符串被完全消耗
        if (parsing_succeeded && typeend - type == len) {

            // 获取类型码的第一个字符
            kind = type[0];
            // 根据类型码进行相应处理
            switch (kind) {
                case NPY_STRINGLTR:
                    // 字符串类型处理
                    check_num = NPY_STRING;
                    break;

                case NPY_DEPRECATED_STRINGLTR2:
                    // 已弃用的字符串类型处理
                    if (DEPRECATE("Data type alias 'a' was deprecated in NumPy 2.0. "
                                  "Use the 'S' alias instead.") < 0) {
                        return NULL;
                    }
                    check_num = NPY_STRING;
                    break;

                /*
                 * 当指定 UNICODE 的长度时，
                 * 给出的是字符数以匹配 STRING 接口。
                 * 每个字符可能超过一个字节，itemsize 必须是字节数。
                 */
                case NPY_UNICODELTR:
                    // UNICODE 类型处理
                    check_num = NPY_UNICODE;
                    // 将元素大小左移两位（相当于乘以 4）
                    elsize <<= 2;
                    break;

                case NPY_VOIDLTR:
                    // VOID 类型处理
                    check_num = NPY_VOID;
                    break;

                default:
                    // 如果元素大小为 0，则默认为 NPY_NOTYPE + 10
                    if (elsize == 0) {
                        check_num = NPY_NOTYPE + 10;
                    }
                    // 支持通用处理 c8、i4、f8 等
                    else {
                        check_num = PyArray_TypestrConvert(elsize, kind);
                        // 如果转换失败，默认加 10
                        if (check_num == NPY_NOTYPE) {
                            check_num += 10;
                        }
                        // 重置元素大小为 0
                        elsize = 0;
                    }
            }
        }
        // 如果解析成功但是类型字符串未被完全消耗，则失败
        else if (parsing_succeeded) {
            goto fail;
        }
    }

    // 如果发生了 Python 异常，则失败
    if (PyErr_Occurred()) {
        goto fail;
    }

    // 返回描述符对象
    PyArray_Descr *ret;
    # 检查 `check_num` 是否为 `NPY_NOTYPE + 10` 或者根据该数字获取对应的数组描述符
    if ((check_num == NPY_NOTYPE + 10) ||
            (ret = PyArray_DescrFromType(check_num)) == NULL) {
        PyErr_Clear();
        # 清除异常状态，检查对象是否在 `typeDict` 中注册
        /* Now check to see if the object is registered in typeDict */
        if (typeDict == NULL) {
            # 如果 `typeDict` 为 NULL，则跳转到 `fail` 标签处
            goto fail;
        }
        # 从 `typeDict` 中获取键为 `obj` 的项，返回 NULL 表示未找到，可能出错时返回 NULL
        PyObject *item = PyDict_GetItemWithError(typeDict, obj);
        if (item == NULL) {
            # 如果出现了异常，则返回 NULL
            if (PyErr_Occurred()) {
                return NULL;
            }
            # 如果 `type` 等于特定字符串，则抛出特定类型的异常并返回 NULL
            if (
                strcmp(type, "int0") == 0 || strcmp(type, "uint0") == 0 ||
                strcmp(type, "void0") == 0 || strcmp(type, "object0") == 0 ||
                strcmp(type, "str0") == 0 || strcmp(type, "bytes0") == 0 ||
                strcmp(type, "bool8") == 0
            ) {
                PyErr_Format(PyExc_TypeError,
                        "Alias %R was removed in NumPy 2.0. Use a name "
                        "without a digit at the end.", obj);
                return NULL;
            }
            # 否则跳转到 `fail` 标签处
            goto fail;
        }

        # 如果 `type` 等于 "a"，则发出关于该别名被弃用的警告信息，如果发生错误则返回 NULL
        if (strcmp(type, "a") == 0) {
            if (DEPRECATE("Data type alias 'a' was deprecated in NumPy 2.0. "
                          "Use the 'S' alias instead.") < 0) {
                return NULL;
            }
        }

        /*
         * 可能仅仅调度到 `_convert_from_type`，但是不知道用户可能注入到 `np.typeDict` 中的内容。
         * Probably only ever dispatches to `_convert_from_type`, but who
         * knows what users are injecting into `np.typeDict`.
         */
        # 返回使用 `_convert_from_any` 函数将 `item` 转换为所需类型的结果，带有对齐参数 `align`
        return _convert_from_any(item, align);
    }

    # 如果返回的数组描述符 `ret` 是无大小的并且其元素大小不等于 `elsize`，则替换描述符的大小为 `elsize`
    if (PyDataType_ISUNSIZED(ret) && ret->elsize != elsize) {
        PyArray_DESCR_REPLACE(ret);
        # 如果替换后 `ret` 为 NULL，则返回 NULL
        if (ret == NULL) {
            return NULL;
        }
        # 设置 `ret` 的元素大小为 `elsize`
        ret->elsize = elsize;
    }
    # 如果 `endian` 不等于 '=' 且 `endian` 是本机字节序，则设置 `endian` 为 '='
    if (endian != '=' && PyArray_ISNBO(endian)) {
        endian = '=';
    }
    # 如果 `endian` 不等于 '=' 且 `ret` 的字节顺序不是 '|' 且不等于 `endian`，则替换描述符的字节顺序为 `endian`
    if (endian != '=' && ret->byteorder != '|' && ret->byteorder != endian) {
        PyArray_DESCR_REPLACE(ret);
        # 如果替换后 `ret` 为 NULL，则返回 NULL
        if (ret == NULL) {
            return NULL;
        }
        # 设置 `ret` 的字节顺序为 `endian`
        ret->byteorder = endian;
    }
    # 返回最终确定的数组描述符 `ret`
    return ret;
/*
 * 如果对象的数据类型不被理解，抛出类型错误并返回空指针。
 * 这里使用 PyErr_Format 来格式化错误信息，%R 代表对象 obj。
 */
fail:
    PyErr_Format(PyExc_TypeError, "data type %R not understood", obj);
    return NULL;
}

/** Array Descr Objects for dynamic types **/

/*
 * 这里定义了一些静态的 PyArray_Descr 对象，对应基本的内置类型。
 * 这些对象在适当的情况下应当进行 DECREF 和 INCREF 操作。
 * 如果在引用计数上出现错误，可能会导致尝试对这些内置对象进行解引用，从而引发问题。
 *
 * 这种方法允许我们使用引用计数来管理所有 PyArray_Descr 对象
 * （无论它们是静态分配还是动态分配的）。
 */

/*NUMPY_API
 * base 不能为空
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNew(PyArray_Descr *base_descr)
{
    if (!PyDataType_ISLEGACY(base_descr)) {
        /* 
         * 主要用途是对字符串进行变异，因此在新风格 DTypes 上可能禁止此操作是合理的。
         * 如果 base_descr 不是旧风格的数据类型，则抛出运行时错误。
         */
        PyErr_SetString(PyExc_RuntimeError,
            "cannot use `PyArray_DescrNew` on new style DTypes.");
        return NULL;
    }
    _PyArray_LegacyDescr *base = (_PyArray_LegacyDescr *)base_descr;
    _PyArray_LegacyDescr *newdescr = PyObject_New(_PyArray_LegacyDescr, Py_TYPE(base));

    if (newdescr == NULL) {
        return NULL;
    }
    /* 不复制 PyObject_HEAD 部分 */
    memcpy((char *)newdescr + sizeof(PyObject),
           (char *)base + sizeof(PyObject),
           sizeof(_PyArray_LegacyDescr) - sizeof(PyObject));

    /*
     * c_metadata 使用值拥有模型，需要克隆它（基本上是深拷贝，
     * 但 auxdata 克隆函数仍具有一定的灵活性），以便新的 PyArray_Descr 对象拥有数据的拷贝。
     * 如果 base->c_metadata 不为空，则克隆它，避免多次释放内存。
     */
    if (base->c_metadata != NULL) {
        newdescr->c_metadata = NPY_AUXDATA_CLONE(base->c_metadata);
        if (newdescr->c_metadata == NULL) {
            PyErr_NoMemory();
            /* 错误处理，释放 newdescr，避免内存泄漏 */
            Py_DECREF(newdescr);
            return NULL;
        }
    }

    if (newdescr->fields == Py_None) {
        newdescr->fields = NULL;
    }
    Py_XINCREF(newdescr->fields);
    Py_XINCREF(newdescr->names);
    if (newdescr->subarray) {
        newdescr->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (newdescr->subarray == NULL) {
            Py_DECREF(newdescr);
            return (PyArray_Descr *)PyErr_NoMemory();
        }
        memcpy(newdescr->subarray, base->subarray, sizeof(PyArray_ArrayDescr));
        Py_INCREF(newdescr->subarray->shape);
        Py_INCREF(newdescr->subarray->base);
    }
    Py_XINCREF(newdescr->typeobj);
    Py_XINCREF(newdescr->metadata);
    newdescr->hash = -1;

    return (PyArray_Descr *)newdescr;
}

/*
 * 如果没有引用计数问题，通常不应对内置类型调用此函数。
 */
static void
arraydescr_dealloc(PyArray_Descr *self)
{
    # 释放 self 对象中的 typeobj 引用，如果存在的话
    Py_XDECREF(self->typeobj);
    
    # 如果 self 不是传统的 dtype（即非传统 dtype），则必须没有 fields 等属性
    if (!PyDataType_ISLEGACY(self)) {
        # 调用 self 对象的类型的 tp_free 方法释放对象内存
        Py_TYPE(self)->tp_free((PyObject *)self);
        return;
    }
    
    # 将 self 转换为 _PyArray_LegacyDescr 类型
    _PyArray_LegacyDescr *lself = (_PyArray_LegacyDescr *)self;

    # 如果 lself 的 fields 属性为 Py_None，表示出现了引用计数错误
    if (lself->fields == Py_None) {
        # 打印错误信息到 stderr，显示出错的 dtype 编号和类型
        fprintf(stderr, "*** Reference count error detected: "
                "an attempt was made to deallocate the dtype %d (%c) ***\n",
                self->type_num, self->type);
        # 断言程序停止，以便调试
        assert(0);
        # 增加 self 的引用计数两次（此处可能是出于调试目的，确保不被释放）
        Py_INCREF(self);
        Py_INCREF(self);
        return;
    }
    
    # 释放 lself 的 names 和 fields 属性的引用计数
    Py_XDECREF(lself->names);
    Py_XDECREF(lself->fields);
    
    # 如果 lself 的 subarray 存在，则释放其 shape 和 base 属性，最后释放 subarray 本身
    if (lself->subarray) {
        Py_XDECREF(lself->subarray->shape);
        Py_DECREF(lself->subarray->base);
        PyArray_free(lself->subarray);
    }
    
    # 释放 lself 的 metadata 属性的引用计数
    Py_XDECREF(lself->metadata);
    
    # 释放 lself 的 c_metadata 指针，并将其设置为 NULL
    NPY_AUXDATA_FREE(lself->c_metadata);
    lself->c_metadata = NULL;
    
    # 最终调用 self 对象的类型的 tp_free 方法释放对象内存
    Py_TYPE(self)->tp_free((PyObject *)self);
/*
 * we need to be careful about setting attributes because these
 * objects are pointed to by arrays that depend on them for interpreting
 * data.  Currently no attributes of data-type objects can be set
 * directly except names.
 */
static PyMemberDef arraydescr_members[] = {
    {"type",
        T_OBJECT, offsetof(PyArray_Descr, typeobj), READONLY, NULL},  // 成员变量：类型对象
    {"kind",
        T_CHAR, offsetof(PyArray_Descr, kind), READONLY, NULL},  // 成员变量：数据类型的种类
    {"char",
        T_CHAR, offsetof(PyArray_Descr, type), READONLY, NULL},  // 成员变量：数据类型的字符表示
    {"num",
        T_INT, offsetof(PyArray_Descr, type_num), READONLY, NULL},  // 成员变量：数据类型的数值编码
    {"byteorder",
        T_CHAR, offsetof(PyArray_Descr, byteorder), READONLY, NULL},  // 成员变量：字节顺序
    {"itemsize",
        T_PYSSIZET, offsetof(PyArray_Descr, elsize), READONLY, NULL},  // 成员变量：数据类型的字节大小
    {"alignment",
        T_PYSSIZET, offsetof(PyArray_Descr, alignment), READONLY, NULL},  // 成员变量：数据类型的对齐方式
    {"flags",
#if NPY_ULONGLONG == NPY_UINT64
        T_ULONGLONG, offsetof(PyArray_Descr, flags), READONLY, NULL},  // 成员变量：数据类型的标志位
#else
    #error Assuming long long is 64bit, if not replace with getter function.
#endif
    {NULL, 0, 0, 0, NULL},  // 结束标记
};

static PyObject *
arraydescr_subdescr_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASSUBARRAY(self)) {
        Py_RETURN_NONE;  // 如果没有子数组，返回 None
    }
    return Py_BuildValue("OO",
            PyDataType_SUBARRAY(self)->base, PyDataType_SUBARRAY(self)->shape);  // 构建返回值元组，包含子数组的基础类型和形状
}

NPY_NO_EXPORT PyObject *
arraydescr_protocol_typestr_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    char basic_ = self->kind;  // 获取数据类型的基础类型字符
    char endian = self->byteorder;  // 获取数据类型的字节顺序
    int size = self->elsize;  // 获取数据类型的字节大小
    PyObject *ret;

    if (endian == '=') {  // 如果字节顺序为 '='
        endian = '<';  // 将字节顺序设为 '<'
        if (!PyArray_IsNativeByteOrder(endian)) {  // 如果不是本地字节顺序
            endian = '>';  // 将字节顺序设为 '>'
        }
    }
    if (self->type_num == NPY_UNICODE) {  // 如果数据类型为 Unicode
        size >>= 2;  // 右移两位，相当于除以 4
    }
    if (self->type_num == NPY_OBJECT) {  // 如果数据类型为对象
        ret = PyUnicode_FromFormat("%c%c", endian, basic_);  // 格式化生成 Unicode 对象
    }
    else {  // 否则
        ret = PyUnicode_FromFormat("%c%c%d", endian, basic_, size);  // 格式化生成 Unicode 对象，包含数据类型大小
    }
    if (ret == NULL) {  // 如果生成失败
        return NULL;  // 返回空指针
    }

    if (PyDataType_ISDATETIME(self)) {  // 如果是日期时间类型
        PyArray_DatetimeMetaData *meta;
        meta = get_datetime_metadata_from_dtype(self);  // 获取日期时间元数据
        if (meta == NULL) {  // 如果获取失败
            Py_DECREF(ret);  // 释放生成的 Unicode 对象
            return NULL;  // 返回空指针
        }
        PyObject *umeta = metastr_to_unicode(meta, 0);  // 将元数据转换为 Unicode 字符串
        if (umeta == NULL) {  // 如果转换失败
            Py_DECREF(ret);  // 释放生成的 Unicode 对象
            return NULL;  // 返回空指针
        }

        Py_SETREF(ret, PyUnicode_Concat(ret, umeta));  // 连接两个 Unicode 对象
        Py_DECREF(umeta);  // 释放 umeta 对象的引用
    }
    return ret;  // 返回 Unicode 对象
}

static PyObject *
arraydescr_name_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    /* let python handle this */  // 让 Python 处理这部分
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy._core._dtype");  // 导入 numpy._core._dtype 模块
    if (_numpy_dtype == NULL) {  // 如果导入失败
        return NULL;  // 返回空指针
    }
    res = PyObject_CallMethod(_numpy_dtype, "_name_get", "O", self);  // 调用 _numpy_dtype 对象的 _name_get 方法
    Py_DECREF(_numpy_dtype);  // 释放 _numpy_dtype 对象的引用
    return res;  // 返回调用结果
}

static PyObject *
/*
 * 返回一个数组描述符的基本数据类型，如果没有子数组，则增加引用计数并返回自身。
 */
PyObject *
arraydescr_base_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASSUBARRAY(self)) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    Py_INCREF(PyDataType_SUBARRAY(self)->base);
    return (PyObject *)(PyDataType_SUBARRAY(self)->base);
}

/*
 * 返回数组描述符的形状，如果没有子数组，则返回空元组。
 * 如果有子数组，确保形状是元组并增加其引用计数后返回。
 */
static PyObject *
arraydescr_shape_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASSUBARRAY(self)) {
        return PyTuple_New(0);
    }
    assert(PyTuple_Check(PyDataType_SUBARRAY(self)->shape));
    Py_INCREF(PyDataType_SUBARRAY(self)->shape);
    return PyDataType_SUBARRAY(self)->shape;
}

/*
 * 返回数组描述符的维度数。如果没有子数组，则返回长整型 0。
 * 否则，获取子数组的形状元组的大小并作为长整型返回。
 */
static PyObject *
arraydescr_ndim_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    Py_ssize_t ndim;

    if (!PyDataType_HASSUBARRAY(self)) {
        return PyLong_FromLong(0);
    }

    /*
     * PyTuple_Size 具有内置检查，确保参数为元组
     */
    ndim = PyTuple_Size(PyDataType_SUBARRAY(self)->shape);
    return PyLong_FromLong(ndim);
}

/*
 * 返回数组描述符的协议描述符。如果没有字段，返回默认描述符。
 * 否则，调用 _numpy_internal 模块的 _array_descr 方法获取描述符。
 */
NPY_NO_EXPORT PyObject *
arraydescr_protocol_descr_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    PyObject *dobj, *res;
    PyObject *_numpy_internal;

    if (!PyDataType_HASFIELDS(self)) {
        /* 获取默认描述符 */
        dobj = PyTuple_New(2);
        if (dobj == NULL) {
            return NULL;
        }
        PyTuple_SET_ITEM(dobj, 0, PyUnicode_FromString(""));
        PyTuple_SET_ITEM(dobj, 1, arraydescr_protocol_typestr_get(self, NULL));
        res = PyList_New(1);
        if (res == NULL) {
            Py_DECREF(dobj);
            return NULL;
        }
        PyList_SET_ITEM(res, 0, dobj);
        return res;
    }

    _numpy_internal = PyImport_ImportModule("numpy._core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_internal, "_array_descr", "O", self);
    Py_DECREF(_numpy_internal);
    return res;
}

/*
 * 返回数组描述符是否为内建类型（返回 1）或用户定义数据类型描述符（返回 2）。
 * 如果既不是，返回 0。
 */
static PyObject *
arraydescr_isbuiltin_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    long val;
    val = 0;
    if (PyDataType_FIELDS(self) == Py_None) {
        val = 1;
    }
    if (PyTypeNum_ISUSERDEF(self->type_num)) {
        val = 2;
    }
    return PyLong_FromLong(val);
}

/*
 * 检查数组描述符是否为本地字节顺序。
 * 如果没有字段，则检查字节顺序是否为本地顺序。
 * 否则，遍历字段字典，逐个检查是否都是本地顺序。
 */
static int
_arraydescr_isnative(PyArray_Descr *self)
{
    if (!PyDataType_HASFIELDS(self)) {
        return PyArray_ISNBO(self->byteorder);
    }
    else {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(PyDataType_FIELDS(self), &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return -1;
            }
            if (!_arraydescr_isnative(new)) {
                return 0;
            }
        }
    }
    return 1;
}
/*
 * 返回 Py_True 如果该数据类型描述符具有本机字节顺序且没有定义字段，
 * 或者如果所有子字段都具有本机字节顺序且定义了字段。
 */
static PyObject *
arraydescr_isnative_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;
    int retval;
    // 调用内部函数检查是否具有本机字节顺序
    retval = _arraydescr_isnative(self);
    if (retval == -1) {
        return NULL;
    }
    // 根据返回值设置 ret 为 Py_True 或 Py_False
    ret = retval ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}

/*
 * 返回 Py_True 如果 self 的标志中包含 NPY_ALIGNED_STRUCT，
 * 否则返回 Py_False。
 */
static PyObject *
arraydescr_isalignedstruct_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;
    // 检查是否标志中包含 NPY_ALIGNED_STRUCT
    ret = (self->flags & NPY_ALIGNED_STRUCT) ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}

/*
 * 如果 self 没有字段定义，则返回 Py_None；
 * 否则返回一个指向字段字典的 PyDictProxy 对象。
 */
static PyObject *
arraydescr_fields_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASFIELDS(self)) {
        // 如果没有字段定义，则返回 Py_None
        Py_RETURN_NONE;
    }
    // 返回字段字典的 PyDictProxy 对象
    return PyDictProxy_New(PyDataType_FIELDS(self));
}

/*
 * 如果 self 的 metadata 为 NULL，则返回 Py_None；
 * 否则返回一个指向 metadata 字典的 PyDictProxy 对象。
 */
static PyObject *
arraydescr_metadata_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (self->metadata == NULL) {
        // 如果 metadata 为 NULL，则返回 Py_None
        Py_RETURN_NONE;
    }
    // 返回 metadata 字典的 PyDictProxy 对象
    return PyDictProxy_New(self->metadata);
}

/*
 * 如果 PyDataType_FLAGCHK(self, NPY_ITEM_HASOBJECT) 为真，则返回 Py_True；
 * 否则返回 Py_False。
 */
static PyObject *
arraydescr_hasobject_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (PyDataType_FLAGCHK(self, NPY_ITEM_HASOBJECT)) {
        // 如果标志中包含 NPY_ITEM_HASOBJECT，则返回 Py_True
        Py_RETURN_TRUE;
    }
    else {
        // 否则返回 Py_False
        Py_RETURN_FALSE;
    }
}

/*
 * 如果 self 没有字段定义，则返回 Py_None；
 * 否则返回一个指向字段名元组的引用，并增加其引用计数。
 */
static PyObject *
arraydescr_names_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASFIELDS(self)) {
        // 如果没有字段定义，则返回 Py_None
        Py_RETURN_NONE;
    }
    // 增加字段名元组的引用计数并返回其引用
    Py_INCREF(PyDataType_NAMES(self));
    return PyDataType_NAMES(self);
}

/*
 * 设置 self 的字段名。
 * val 是新的字段名序列，应该是一个字符串序列。
 * 返回 -1 表示出错，0 表示成功。
 */
static int
arraydescr_names_set(
        _PyArray_LegacyDescr *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    int N = 0;
    int i;
    PyObject *new_names;
    PyObject *new_fields;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete dtype names attribute");
        return -1;
    }
    if (!PyDataType_HASFIELDS(self)) {
        PyErr_SetString(PyExc_ValueError,
                "there are no fields defined");
        return -1;
    }

    /*
     * FIXME
     *
     * This deprecation has been temporarily removed for the NumPy 1.7
     * release. It should be re-added after the 1.7 branch is done,
     * and a convenience API to replace the typical use-cases for
     * mutable names should be implemented.
     *
     * if (DEPRECATE("Setting NumPy dtype names is deprecated, the dtype "
     *                "will become immutable in a future version") < 0) {
     *     return -1;
     * }
     */

    // 获取当前字段名元组的大小
    N = PyTuple_GET_SIZE(self->names);
    // 检查 val 是否为序列且长度与字段名元组相等
    if (!PySequence_Check(val) || PyObject_Size((PyObject *)val) != N) {
        /* 应该是 TypeError，但是这段代码应该被弃用。 */
        PyErr_Format(PyExc_ValueError,
                "must replace all names at once with a sequence of length %d",
                N);
        return -1;
    }
    /* 确保所有条目都是字符串 */
    // 省略部分代码，处理确保所有条目是字符串的逻辑
}
    for (i = 0; i < N; i++) {
        PyObject *item;
        int valid;
        // 获取序列 `val` 中第 `i` 个元素
        item = PySequence_GetItem(val, i);
        // 检查 `item` 是否为 Unicode 字符串
        valid = PyUnicode_Check(item);
        // 如果 `item` 不是有效的 Unicode 字符串
        if (!valid) {
            // 抛出值错误异常，指明非字符串类型的名称
            PyErr_Format(PyExc_ValueError,
                    "item #%d of names is of type %s and not string",
                    i, Py_TYPE(item)->tp_name);
            Py_DECREF(item); // 释放 `item` 的引用
            return -1; // 返回错误标志
        }
        Py_DECREF(item); // 释放 `item` 的引用
    }
    /* Invalidate cached hash value */
    // 将缓存的哈希值设为无效值 `-1`
    self->hash = -1;
    /* Update dictionary keys in fields */
    // 将序列 `val` 转换为元组 `new_names`
    new_names = PySequence_Tuple(val);
    // 如果转换失败，返回错误标志
    if (new_names == NULL) {
        return -1;
    }
    // 创建新的空字典 `new_fields`
    new_fields = PyDict_New();
    // 如果创建失败，释放 `new_names` 并返回错误标志
    if (new_fields == NULL) {
        Py_DECREF(new_names);
        return -1;
    }
    for (i = 0; i < N; i++) {
        PyObject *key;
        PyObject *item;
        PyObject *new_key;
        int ret;
        // 获取 `self->names` 元组中第 `i` 个元素作为键 `key`
        key = PyTuple_GET_ITEM(self->names, i);
        // 在 `self->fields` 字典中查找键为 `key` 的项，引用计数不增加
        /* Borrowed references to item and new_key */
        item = PyDict_GetItemWithError(self->fields, key);
        // 如果未找到 `key` 对应的项且没有错误发生
        if (item == NULL) {
            if (!PyErr_Occurred()) {
                // 抛出内部调用错误异常
                PyErr_BadInternalCall();
            }
            Py_DECREF(new_names); // 释放 `new_names` 的引用
            Py_DECREF(new_fields); // 释放 `new_fields` 的引用
            return -1; // 返回错误标志
        }
        // 获取 `new_names` 元组中第 `i` 个元素作为新键 `new_key`
        new_key = PyTuple_GET_ITEM(new_names, i);
        // 检查 `new_fields` 字典中是否包含 `new_key`
        ret = PyDict_Contains(new_fields, new_key);
        // 如果检查过程中发生错误，释放 `new_names` 和 `new_fields` 的引用并返回错误标志
        if (ret < 0) {
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
        // 如果 `new_fields` 中已经包含 `new_key`，抛出值错误异常
        else if (ret != 0) {
            PyErr_SetString(PyExc_ValueError, "Duplicate field names given.");
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
        // 将 `item` 与 `new_key` 添加到 `new_fields` 字典中
        if (PyDict_SetItem(new_fields, new_key, item) < 0) {
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
    }

    /* Replace names */
    // 释放旧的 `self->names` 引用，将 `new_names` 赋值给 `self->names`
    Py_DECREF(self->names);
    self->names = new_names;

    /* Replace fields */
    // 释放旧的 `self->fields` 引用，将 `new_fields` 赋值给 `self->fields`
    Py_DECREF(self->fields);
    self->fields = new_fields;

    return 0;
# 定义一个静态的属性获取器和设置器的列表，用于描述数组描述符对象的各种属性
static PyGetSetDef arraydescr_getsets[] = {
    # 子数据类型描述符的获取器，使用arraydescr_subdescr_get函数
    {"subdtype",
        (getter)arraydescr_subdescr_get,
        NULL, NULL, NULL},
    # 描述符的获取器，使用arraydescr_protocol_descr_get函数
    {"descr",
        (getter)arraydescr_protocol_descr_get,
        NULL, NULL, NULL},
    # 字符串形式的类型描述符的获取器，使用arraydescr_protocol_typestr_get函数
    {"str",
        (getter)arraydescr_protocol_typestr_get,
        NULL, NULL, NULL},
    # 名称的获取器，使用arraydescr_name_get函数
    {"name",
        (getter)arraydescr_name_get,
        NULL, NULL, NULL},
    # 基本类型的获取器，使用arraydescr_base_get函数
    {"base",
        (getter)arraydescr_base_get,
        NULL, NULL, NULL},
    # 数组形状的获取器，使用arraydescr_shape_get函数
    {"shape",
        (getter)arraydescr_shape_get,
        NULL, NULL, NULL},
    # 数组维度的获取器，使用arraydescr_ndim_get函数
    {"ndim",
        (getter)arraydescr_ndim_get,
        NULL, NULL, NULL},
    # 是否为内建类型的获取器，使用arraydescr_isbuiltin_get函数
    {"isbuiltin",
        (getter)arraydescr_isbuiltin_get,
        NULL, NULL, NULL},
    # 是否为本地字节序的获取器，使用arraydescr_isnative_get函数
    {"isnative",
        (getter)arraydescr_isnative_get,
        NULL, NULL, NULL},
    # 是否为对齐结构的获取器，使用arraydescr_isalignedstruct_get函数
    {"isalignedstruct",
        (getter)arraydescr_isalignedstruct_get,
        NULL, NULL, NULL},
    # 字段的获取器，使用arraydescr_fields_get函数
    {"fields",
        (getter)arraydescr_fields_get,
        NULL, NULL, NULL},
    # 元数据的获取器，使用arraydescr_metadata_get函数
    {"metadata",
        (getter)arraydescr_metadata_get,
        NULL, NULL, NULL},
    # 名称列表的获取器和设置器，使用arraydescr_names_get和arraydescr_names_set函数
    {"names",
        (getter)arraydescr_names_get,
        (setter)arraydescr_names_set,
        NULL, NULL},
    # 是否含有对象的获取器，使用arraydescr_hasobject_get函数
    {"hasobject",
        (getter)arraydescr_hasobject_get,
        NULL, NULL, NULL},
    # 结束符号，没有getter、setter和doc
    {NULL, NULL, NULL, NULL, NULL},
};
    if (subtype != &PyArrayDescr_Type) {
        # 检查 subtype 是否不是 PyArrayDescr_Type 类型的对象
        if (Py_TYPE(subtype) == &PyArrayDTypeMeta_Type &&
                (NPY_DT_SLOTS((PyArray_DTypeMeta *)subtype)) != NULL &&
                !NPY_DT_is_legacy((PyArray_DTypeMeta *)subtype) &&
                subtype->tp_new != PyArrayDescr_Type.tp_new) {
            '''
            如果 subtype 是 PyArrayDTypeMeta_Type 类型的对象，并且满足以下条件：
            - NPY_DT_SLOTS((PyArray_DTypeMeta *)subtype) 不为 NULL
            - 不是遗留的 dtype
            - subtype 的 tp_new 方法不等于 PyArrayDescr_Type.tp_new 方法
            
            则认为其是一个正确初始化的用户自定义 dtype。分配内存并尽可能初始化主要部分。
            TODO: 这可能应该是一个用户函数，并强制执行诸如正确设置 `elsize` 等规则。
            TODO: 这是实验性的 API！
            '''
            # 将 subtype 转换为 PyArray_DTypeMeta 类型
            PyArray_DTypeMeta *DType = (PyArray_DTypeMeta *)subtype;
            # 分配一个 PyArray_Descr 对象
            PyArray_Descr *descr = (PyArray_Descr *)subtype->tp_alloc(subtype, 0);
            if (descr == 0) {
                # 分配内存失败，抛出内存错误异常
                PyErr_NoMemory();
                return NULL;
            }
            # 增加对 scalar_type 的引用计数
            Py_XINCREF(DType->scalar_type);
            # 设置 descr 对象的 typeobj 和 type_num 属性
            descr->typeobj = DType->scalar_type;
            descr->type_num = DType->type_num;
            # 设置 descr 对象的 flags、byteorder、elsize 和 hash 属性
            descr->flags = NPY_USE_GETITEM|NPY_USE_SETITEM;
            descr->byteorder = '|';  # 如果 DType 使用了 byteorder，则允许其覆盖
            descr->elsize = -1;  # 初始化为无效值
            descr->hash = -1;
            # 返回 PyArray_Descr 对象的 PyObject 表示
            return (PyObject *)descr;
        }
        '''
        若程序流执行到这里，表明 subtype 类型不符合预期：
        DTypeMeta 类应该防止这种情况发生。
        '''
        # 抛出系统错误异常，指示不应该从 np.dtype.__new__() 继承
        PyErr_Format(PyExc_SystemError,
                "'%S' must not inherit np.dtype.__new__(). User DTypes should "
                "currently call `PyArrayDescr_Type.tp_new` from their new.",
                subtype);
        return NULL;
    }

    PyObject *odescr, *metadata=NULL;
    PyArray_Descr *conv;
    npy_bool align = NPY_FALSE;
    npy_bool copy = NPY_FALSE;
    npy_bool copied = NPY_FALSE;

    static char *kwlist[] = {"dtype", "align", "copy", "metadata", NULL};

    # 解析输入参数，以及可选的 align、copy 和 metadata 参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O!:dtype", kwlist,
                &odescr,
                PyArray_BoolConverter, &align,
                PyArray_BoolConverter, &copy,
                &PyDict_Type, &metadata)) {
        return NULL;
    }

    # 将 odescr 转换为 PyArray_Descr 对象，使用 align 参数
    conv = _convert_from_any(odescr, align);
    if (conv == NULL) {
        return NULL;
    }

    # 如果需要复制且 conv 不是一个复制
    if (copy && PyDataType_FIELDS(conv) == Py_None) {
        # 替换 conv 的描述符
        PyArray_DESCR_REPLACE(conv);
        if (conv == NULL) {
            return NULL;
        }
        # 设置 copied 标志为 True
        copied = NPY_TRUE;
    }
    // 如果 metadata 不为空
    if ((metadata != NULL)) {
        // 检查 conv 不是旧式数据类型
        if (!PyDataType_ISLEGACY(conv)) {
            // 抛出类型错误异常，因为不能给新式数据类型附加元数据
            PyErr_SetString(PyExc_TypeError,
                    "cannot attach metadata to new style DType");
            // 释放 conv 的引用计数，并返回空指针
            Py_DECREF(conv);
            return NULL;
        }
        /*
         * We need to be sure to make a new copy of the data-type and any
         * underlying dictionary
         */
        // 如果还没有复制过数据类型描述符
        if (!copied) {
            // 替换 conv 的数据类型描述符为一个新的副本
            PyArray_DESCR_REPLACE(conv);
            // 如果替换后 conv 为空，则返回空指针
            if (conv == NULL) {
                return NULL;
            }
            // 设置复制标记为真
            copied = NPY_TRUE;
        }
        // 将 conv 强制转换为旧式数据类型描述符类型
        _PyArray_LegacyDescr *lconv = (_PyArray_LegacyDescr *)conv;
        // 如果旧式描述符已经有元数据
        if ((lconv->metadata != NULL)) {
            /*
             * Make a copy of the metadata before merging with the
             * input metadata so that this data-type descriptor has
             * its own copy
             */
            /* Save a reference */
            // 保存原元数据的引用
            odescr = lconv->metadata;
            // 创建元数据的新副本
            lconv->metadata = PyDict_Copy(odescr);
            // 减少原元数据的引用计数
            Py_DECREF(odescr);

            /*
             * Update conv->metadata with anything new in metadata
             * keyword, but do not over-write anything already there
             */
            // 将输入的 metadata 合并到 conv->metadata 中，不覆盖已有的内容
            if (PyDict_Merge(lconv->metadata, metadata, 0) != 0) {
                // 合并失败时，释放 conv 的引用计数，并返回空指针
                Py_DECREF(conv);
                return NULL;
            }
        }
        else {
            /* Make a copy of the input dictionary */
            // 创建输入字典 metadata 的新副本，并赋给 lconv->metadata
            lconv->metadata = PyDict_Copy(metadata);
        }
    }

    // 返回 conv 强制转换为 PyObject 指针类型
    return (PyObject *)conv;
/*
 * Return a tuple of
 * (cleaned metadata dictionary, tuple with (str, num))
 */
static PyObject *
_get_pickleabletype_from_datetime_metadata(PyArray_Descr *dtype)
{
    PyObject *ret, *dt_tuple;
    PyArray_DatetimeMetaData *meta;

    /* Create the 2-item tuple to return */
    ret = PyTuple_New(2);
    if (ret == NULL) {
        return NULL;
    }

    /* Store the metadata dictionary */
    if (dtype->metadata != NULL) {
        Py_INCREF(dtype->metadata);  // 增加元数据字典的引用计数
        PyTuple_SET_ITEM(ret, 0, dtype->metadata);  // 将元数据字典设置为返回元组的第一个元素
    } else {
        PyTuple_SET_ITEM(ret, 0, PyDict_New());  // 如果元数据为空，则创建一个新的空字典
    }

    /* Convert the datetime metadata into a tuple */
    meta = get_datetime_metadata_from_dtype(dtype);  // 获取日期时间的元数据
    if (meta == NULL) {
        Py_DECREF(ret);  // 如果获取失败，释放返回的元组
        return NULL;
    }
    /* Use a 4-tuple that numpy 1.6 knows how to unpickle */
    dt_tuple = PyTuple_New(4);  // 创建一个包含四个元素的元组
    if (dt_tuple == NULL) {
        Py_DECREF(ret);  // 如果创建失败，释放返回的元组
        return NULL;
    }
    PyTuple_SET_ITEM(dt_tuple, 0,
            PyBytes_FromString(_datetime_strings[meta->base]));  // 设置元组的第一个元素为日期时间基础的字节串
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyLong_FromLong(meta->num));  // 设置元组的第二个元素为日期时间的数值
    PyTuple_SET_ITEM(dt_tuple, 2,
            PyLong_FromLong(1));  // 设置元组的第三个元素为整数1
    PyTuple_SET_ITEM(dt_tuple, 3,
            PyLong_FromLong(1));  // 设置元组的第四个元素为整数1

    PyTuple_SET_ITEM(ret, 1, dt_tuple);  // 将日期时间元组设置为返回元组的第二个元素

    return ret;  // 返回包含元数据字典和日期时间元组的元组
}

/*
 * return a tuple of (callable object, args, state).
 *
 * TODO: This method needs to change so that unpickling doesn't
 *       use __setstate__. This is required for the dtype
 *       to be an immutable object.
 */
static PyObject *
arraydescr_reduce(PyArray_Descr *self, PyObject *NPY_UNUSED(args))
{
    /*
     * version number of this pickle type. Increment if we need to
     * change the format. Be sure to handle the old versions in
     * arraydescr_setstate.
    */
    const int version = 4;  // 定义当前 pickle 格式的版本号为4
    PyObject *ret, *mod, *obj;
    PyObject *state;
    char endian;
    int elsize, alignment;

    ret = PyTuple_New(3);  // 创建一个包含三个元素的元组作为返回值
    if (ret == NULL) {
        return NULL;
    }
    mod = PyImport_ImportModule("numpy._core._multiarray_umath");  // 导入 numpy 的 umath 模块
    if (mod == NULL) {
        Py_DECREF(ret);  // 如果导入模块失败，释放返回的元组
        return NULL;
    }
    obj = PyObject_GetAttr(mod, npy_interned_str.dtype);  // 获取 umath 模块中的 dtype 属性对象
    Py_DECREF(mod);  // 释放导入的 umath 模块对象
    if (obj == NULL) {
        Py_DECREF(ret);  // 如果获取 dtype 属性失败，释放返回的元组
        return NULL;
    }
    PyTuple_SET_ITEM(ret, 0, obj);  // 将获取的 dtype 属性对象设置为返回元组的第一个元素
    if (PyTypeNum_ISUSERDEF(self->type_num)
            || ((self->type_num == NPY_VOID
                    && self->typeobj != &PyVoidArrType_Type))) {
        obj = (PyObject *)self->typeobj;  // 如果类型为用户定义或者是 VOID 类型但不是 PyVoidArrType_Type 类型
        Py_INCREF(obj);  // 增加类型对象的引用计数
    }
    else if (!NPY_DT_is_legacy(NPY_DTYPE(self))) {
        PyErr_SetString(PyExc_RuntimeError,
                "Custom dtypes cannot use the default pickle implementation "
                "for NumPy dtypes. Add a custom pickle implementation to the "
                "DType to avoid this error");  // 如果不是遗留的类型且不是用户定义的类型，抛出运行时错误
        return NULL;
    }
    else {
        // 获取元素大小
        elsize = self->elsize;
        // 如果是Unicode类型，则右移两位（相当于除以4），因为Unicode字符大小为4字节
        if (self->type_num == NPY_UNICODE) {
            elsize >>= 2;
        }
        // 根据kind和elsize创建Python Unicode对象
        obj = PyUnicode_FromFormat("%c%d",self->kind, elsize);
    }
    // 将obj、Py_False和Py_True打包成一个元组，放入ret元组的第二个位置
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(NOO)", obj, Py_False, Py_True));

    /*
     * 现在返回至少包含字节顺序、子数组和字段的状态
     */
    // 获取字节顺序
    endian = self->byteorder;
    // 如果字节顺序为平台本地顺序
    if (endian == '=') {
        endian = '<';
        // 如果不是本地字节顺序，设为大端字节顺序
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    // 如果是日期时间类型
    if (PyDataType_ISDATETIME(self)) {
        PyObject *newobj;
        // 创建一个包含9个元素的元组
        state = PyTuple_New(9);
        // 设置第一个元素为版本号的长整型对象
        PyTuple_SET_ITEM(state, 0, PyLong_FromLong(version));
        /*
         * newobj是Python元数据字典和日期时间信息（str, num）的元组
         */
        // 从日期时间元数据获取可序列化的类型对象
        newobj = _get_pickleabletype_from_datetime_metadata(self);
        // 如果获取失败，释放资源并返回空
        if (newobj == NULL) {
            Py_DECREF(state);
            Py_DECREF(ret);
            return NULL;
        }
        // 设置第九个元素为newobj
        PyTuple_SET_ITEM(state, 8, newobj);
    }
    // 如果有元数据
    else if (self->metadata) {
        // 创建一个包含9个元素的元组
        state = PyTuple_New(9);
        // 设置第一个元素为版本号的长整型对象
        PyTuple_SET_ITEM(state, 0, PyLong_FromLong(version));
        // 增加元数据的引用计数并设置为第九个元素
        Py_INCREF(self->metadata);
        PyTuple_SET_ITEM(state, 8, self->metadata);
    }
    // 否则使用版本3的pickle格式
    else { /* Use version 3 pickle format */
        // 创建一个包含8个元素的元组
        state = PyTuple_New(8);
        // 设置第一个元素为版本号3的长整型对象
        PyTuple_SET_ITEM(state, 0, PyLong_FromLong(3));
    }

    // 设置元组state的第一个元素为字节顺序的Unicode对象
    PyTuple_SET_ITEM(state, 1, PyUnicode_FromFormat("%c", endian));
    // 获取数组描述子描述信息
    PyTuple_SET_ITEM(state, 2, arraydescr_subdescr_get(self, NULL));
    // 如果有字段信息
    if (PyDataType_HASFIELDS(self)) {
        // 增加字段名和字段数据的引用计数，并设置为第三和第四个元素
        Py_INCREF(PyDataType_NAMES(self));
        Py_INCREF(PyDataType_FIELDS(self));
        PyTuple_SET_ITEM(state, 3, PyDataType_NAMES(self));
        PyTuple_SET_ITEM(state, 4, PyDataType_FIELDS(self));
    }
    // 如果没有字段信息
    else {
        // 设置第三和第四个元素为Py_None，并增加引用计数
        PyTuple_SET_ITEM(state, 3, Py_None);
        PyTuple_SET_ITEM(state, 4, Py_None);
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
    }

    /* 对于扩展类型，还包括elsize和alignment */
    // 如果是扩展类型，设置elsize和alignment
    if (PyTypeNum_ISEXTENDED(self->type_num)) {
        elsize = self->elsize;
        alignment = self->alignment;
    }
    // 否则设置elsize和alignment为-1
    else {
        elsize = -1;
        alignment = -1;
    }
    // 设置第五、六、七个元素为elsize、alignment和flags的长整型对象
    PyTuple_SET_ITEM(state, 5, PyLong_FromLong(elsize));
    PyTuple_SET_ITEM(state, 6, PyLong_FromLong(alignment));
    PyTuple_SET_ITEM(state, 7, PyLong_FromUnsignedLongLong(self->flags));

    // 设置ret元组的第三个元素为state
    PyTuple_SET_ITEM(ret, 2, state);
    // 返回ret元组
    return ret;
/*
 * 返回 NPY_OBJECT_DTYPE_FLAGS，如果此数据类型在设置状态时具有使用对象部分，
 * 因为 hasobject 没有被存储。
 */
static char
_descr_find_object(PyArray_Descr *self)
{
    // 检查数据类型的标志或者类型号是否为 NPY_OBJECT，或者种类是否为 'O'
    if (self->flags
            || self->type_num == NPY_OBJECT
            || self->kind == 'O') {
        return NPY_OBJECT_DTYPE_FLAGS;
    }
    // 如果数据类型有字段
    if (PyDataType_HASFIELDS(self)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        // 遍历字段字典
        while (PyDict_Next(PyDataType_FIELDS(self), &pos, &key, &value)) {
            // 跳过特定情况下的标题处理
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            // 解析字段的信息
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                PyErr_Clear();
                return 0;
            }
            // 递归查找字段的对象部分
            if (_descr_find_object(new)) {
                new->flags = NPY_OBJECT_DTYPE_FLAGS;
                return NPY_OBJECT_DTYPE_FLAGS;
            }
        }
    }
    // 没有找到对象部分，返回 0
    return 0;
}

/*
 * state 至少包含 byteorder、subarray 和 fields，对于 EXTENDED 数组还可以包括
 * elsize 和 alignment。
 */
static PyObject *
arraydescr_setstate(_PyArray_LegacyDescr *self, PyObject *args)
{
    int elsize = -1, alignment = -1;
    int version = 4;
    char endian;
    PyObject *endian_obj;
    PyObject *subarray, *fields, *names = NULL, *metadata=NULL;
    int incref_names = 1;
    int int_dtypeflags = 0;
    npy_uint64 dtypeflags;

    // 如果不是旧版本的数据类型，抛出运行时错误
    if (!PyDataType_ISLEGACY(self)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Cannot unpickle new style DType without custom methods.");
        return NULL;
    }

    // 如果字段为空，返回 None
    if (self->fields == Py_None) {
        Py_RETURN_NONE;
    }

    // 检查参数元组的大小和类型
    if (PyTuple_GET_SIZE(args) != 1
            || !(PyTuple_Check(PyTuple_GET_ITEM(args, 0)))) {
        PyErr_BadInternalCall();
        return NULL;
    }

    // 根据参数元组的大小选择不同的解析方式
    switch (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0))) {
    case 9:
        // 解析包含 9 个元素的参数元组
        if (!PyArg_ParseTuple(args, "(iOOOOiiiO):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags, &metadata)) {
            PyErr_Clear();
            return NULL;
        }
        break;
    case 8:
        // 解析包含 8 个元素的参数元组
        if (!PyArg_ParseTuple(args, "(iOOOOiii):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags)) {
            return NULL;
        }
        break;
    case 7:
        // 解析包含 7 个元素的参数元组
        if (!PyArg_ParseTuple(args, "(iOOOOii):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    /* 处理版本号为 6 的情况 */
    case 6:
        /* 解析参数元组，期望格式为 (整数, 对象, 对象, 对象, 整数, 整数) */
        if (!PyArg_ParseTuple(args, "(iOOOii):__setstate__",
                    &version,
                    &endian_obj, &subarray, &fields,
                    &elsize, &alignment)) {
            return NULL;
        }
        break;
    /* 处理版本号为 5 的情况 */
    case 5:
        version = 0;
        /* 解析参数元组，期望格式为 (对象, 对象, 对象, 整数, 整数) */
        if (!PyArg_ParseTuple(args, "(OOOii):__setstate__",
                    &endian_obj, &subarray, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    /* 默认情况下 */
    default:
        /* 抛出错误 */
        if (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0)) > 5) {
            /* 如果参数元组的第一个元素的大小大于 5，则尝试将其转换为长整型 */
            version = PyLong_AsLong(PyTuple_GET_ITEM(args, 0));
        }
        else {
            /* 否则版本号为 -1 */
            version = -1;
        }
    }

    /*
     * 如果有新的 pickle 格式需求，增加版本号。
     * 但是应当仍然能够处理旧版本。
     */
    if (version < 0 || version > 4) {
        /* 如果版本号不在有效范围内，抛出相应错误 */
        PyErr_Format(PyExc_ValueError,
                     "can't handle version %d of numpy.dtype pickle",
                     version);
        return NULL;
    }
    /* 使缓存的哈希值无效化 */
    self->hash = -1;

    /* 如果版本号为 1 或 0 */
    if (version == 1 || version == 0) {
        /* 如果 fields 不是 None */
        if (fields != Py_None) {
            PyObject *key, *list;
            key = PyLong_FromLong(-1);
            /* 从 fields 字典中获取值，错误处理方式获取 */
            list = PyDict_GetItemWithError(fields, key);
            if (!list) {
                if (!PyErr_Occurred()) {
                    /* 如果 fields 中缺少它所声称包含的名称 */
                    PyErr_BadInternalCall();
                }
                return NULL;
            }
            /* 增加引用计数并设置 names 为获取到的列表 */
            Py_INCREF(list);
            names = list;
            /* 从 fields 字典中删除 key */
            PyDict_DelItem(fields, key);
            incref_names = 0;
        }
        else {
            /* 否则设置 names 为 None */
            names = Py_None;
        }
    }

    /* 解析 endian 字段 */
    if (PyUnicode_Check(endian_obj) || PyBytes_Check(endian_obj)) {
        PyObject *tmp = NULL;
        char *str;
        Py_ssize_t len;

        /* 如果 endian_obj 是 Unicode 对象，则转换为 ASCII 字符串 */
        if (PyUnicode_Check(endian_obj)) {
            tmp = PyUnicode_AsASCIIString(endian_obj);
            if (tmp == NULL) {
                return NULL;
            }
            endian_obj = tmp;
        }

        /* 获取 endian_obj 的 C 字符串表示和长度 */
        if (PyBytes_AsStringAndSize(endian_obj, &str, &len) < 0) {
            Py_XDECREF(tmp);
            return NULL;
        }
        /* 如果长度不为 1，则抛出错误 */
        if (len != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "endian is not 1-char string in Numpy dtype unpickling");
            Py_XDECREF(tmp);
            return NULL;
        }
        /* 设置 endian 为 str 的第一个字符 */
        endian = str[0];
        Py_XDECREF(tmp);
    }
    else {
        /* 如果 endian_obj 不是字符串，则抛出错误 */
        PyErr_SetString(PyExc_ValueError,
                        "endian is not a string in Numpy dtype unpickling");
        return NULL;
    }

    /* 如果 fields 是 None 而 names 不是，或者 names 是 None 而 fields 不是，则抛出错误 */
    if ((fields == Py_None && names != Py_None) ||
        (names == Py_None && fields != Py_None)) {
        PyErr_Format(PyExc_ValueError,
                "inconsistent fields and names in Numpy dtype unpickling");
        return NULL;
    }
    # 检查 names 是否不是 None 且不是元组类型，如果条件成立，抛出数值错误异常并返回空
    if (names != Py_None && !PyTuple_Check(names)) {
        PyErr_Format(PyExc_ValueError,
                "non-tuple names in Numpy dtype unpickling");
        return NULL;
    }

    # 检查 fields 是否不是 None 且不是字典类型，如果条件成立，抛出数值错误异常并返回空
    if (fields != Py_None && !PyDict_Check(fields)) {
        PyErr_Format(PyExc_ValueError,
                "non-dict fields in Numpy dtype unpickling");
        return NULL;
    }

    # 如果 endian 不是 '|' 并且是本地字节顺序，则将 endian 设置为 '='
    if (endian != '|' && PyArray_IsNativeByteOrder(endian)) {
        endian = '=';
    }
    # 将对象的字节顺序设置为 endian
    self->byteorder = endian;

    # 如果对象已经有子数组，则释放其之前的引用和内存
    if (self->subarray) {
        Py_XDECREF(self->subarray->base);
        Py_XDECREF(self->subarray->shape);
        PyArray_free(self->subarray);
    }
    # 将子数组引用置为空
    self->subarray = NULL;

    # 如果 subarray 不是 None
    if (subarray != Py_None) {
        PyObject *subarray_shape;

        """
         * 确保 subarray[0] 是 ArrayDescr 类型，
         * 并且从 subarray[1] 中获取的 subarray_shape 是由整数组成的元组。
         """
        # 如果 subarray 不是正确的元组结构，抛出数值错误异常并返回空
        if (!(PyTuple_Check(subarray) &&
              PyTuple_Size(subarray) == 2 &&
              PyArray_DescrCheck(PyTuple_GET_ITEM(subarray, 0)))) {
            PyErr_Format(PyExc_ValueError,
                         "incorrect subarray in __setstate__");
            return NULL;
        }
        # 获取 subarray 的第二个元素作为 subarray_shape
        subarray_shape = PyTuple_GET_ITEM(subarray, 1);
        # 如果 subarray_shape 是数字类型，将其转换为长整型
        if (PyNumber_Check(subarray_shape)) {
            PyObject *tmp;
            tmp = PyNumber_Long(subarray_shape);
            if (tmp == NULL) {
                return NULL;
            }
            # 构建一个包含 tmp 的元组，并释放 tmp 的引用
            subarray_shape = Py_BuildValue("(O)", tmp);
            Py_DECREF(tmp);
            if (subarray_shape == NULL) {
                return NULL;
            }
        }
        # 如果 subarray_shape 是由整数组成的元组，增加其引用计数
        else if (_is_tuple_of_integers(subarray_shape)) {
            Py_INCREF(subarray_shape);
        }
        # 否则，抛出数值错误异常并返回空
        else {
            PyErr_Format(PyExc_ValueError,
                         "incorrect subarray shape in __setstate__");
            return NULL;
        }

        # 为 self->subarray 分配内存
        self->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (self->subarray == NULL) {
            return PyErr_NoMemory();
        }
        # 将 subarray 的第一个元素作为 self->subarray->base，并增加其引用计数
        self->subarray->base = (PyArray_Descr *)PyTuple_GET_ITEM(subarray, 0);
        Py_INCREF(self->subarray->base);
        # 将 subarray_shape 赋值给 self->subarray->shape
        self->subarray->shape = subarray_shape;
    }
    if (fields != Py_None) {
        /*
         * 确保字段名的类型适当
         */
        Py_ssize_t i;
        int names_ok = 1;
        PyObject *name;

        // 检查字段名是否都是合适的字符串类型
        for (i = 0; i < PyTuple_GET_SIZE(names); ++i) {
            name = PyTuple_GET_ITEM(names, i);
            if (!PyUnicode_Check(name)) {
                names_ok = 0;
                break;
            }
        }

        // 如果字段名类型全部正确
        if (names_ok) {
            // 释放之前的字段对象，并将新的字段对象赋值给self->fields
            Py_XDECREF(self->fields);
            self->fields = fields;
            // 增加字段对象的引用计数
            Py_INCREF(fields);
            // 释放旧的字段名对象，并将新的字段名对象赋值给self->names
            Py_XDECREF(self->names);
            self->names = names;
            // 如果需要增加字段名对象的引用计数，则增加它
            if (incref_names) {
                Py_INCREF(names);
            }
        }
        else {
            /*
             * 为了支持在Python 3中加载由Python 2生成的pickle文件时的编码问题，
             * 我们需要更宽松地将字段名从字节字符串转换为Unicode。
             */
            PyObject *tmp, *new_name, *field;

            // 创建一个新的空字典对象tmp
            tmp = PyDict_New();
            if (tmp == NULL) {
                return NULL;
            }
            // 释放之前的字段对象，并将新的字段对象赋值给self->fields
            Py_XDECREF(self->fields);
            self->fields = tmp;

            // 创建一个新的空元组对象tmp
            tmp = PyTuple_New(PyTuple_GET_SIZE(names));
            if (tmp == NULL) {
                return NULL;
            }
            // 释放旧的字段名对象，并将新的字段名对象赋值给self->names
            Py_XDECREF(self->names);
            self->names = tmp;

            // 遍历字段名元组，处理每个字段名
            for (i = 0; i < PyTuple_GET_SIZE(names); ++i) {
                name = PyTuple_GET_ITEM(names, i);
                // 在fields字典中查找对应的字段
                field = PyDict_GetItemWithError(fields, name);
                if (!field) {
                    if (!PyErr_Occurred()) {
                        /* fields缺少它声称包含的字段名 */
                        PyErr_BadInternalCall();
                    }
                    return NULL;
                }

                // 如果字段名是Unicode类型，则直接使用它
                if (PyUnicode_Check(name)) {
                    new_name = name;
                    Py_INCREF(new_name);
                }
                else {
                    // 将字节字符串转换为ASCII编码的Unicode对象
                    new_name = PyUnicode_FromEncodedObject(name, "ASCII", "strict");
                    if (new_name == NULL) {
                        return NULL;
                    }
                }

                // 将新的字段名对象添加到self->names元组中
                PyTuple_SET_ITEM(self->names, i, new_name);
                // 将字段名和字段值添加到self->fields字典中
                if (PyDict_SetItem(self->fields, new_name, field) != 0) {
                    return NULL;
                }
            }
        }
    }

    // 如果数组类型是扩展类型，设置元素大小和对齐方式
    if (PyTypeNum_ISEXTENDED(self->type_num)) {
        self->elsize = elsize;
        self->alignment = alignment;
    }

    /*
     * 我们使用转换为字符的整数以保持与pickle数组的向后兼容性。
     * 以前的版本使用int编码标志，即使在PyArray_Descr结构中实际上是一个char。
     */
    if (int_dtypeflags < 0 && int_dtypeflags >= -128) {
        /* NumPy以前使用的是char类型。所以如果是有符号数，进行规范化。 */
        int_dtypeflags += 128;
    }
    // 将整数标志转换为dtypeflags
    dtypeflags = int_dtypeflags;
    # 如果 dtypeflags 不等于 int_dtypeflags，抛出值错误异常，并返回空值
    if (dtypeflags != int_dtypeflags) {
        PyErr_Format(PyExc_ValueError,
                     "incorrect value for flags variable (overflow)");
        return NULL;
    }
    else {
        # 将 self 对象的 flags 属性设置为 dtypeflags
        self->flags = dtypeflags;
    }

    # 如果 version 小于 3，调用 _descr_find_object 函数来设置 self 对象的 flags 属性
    if (version < 3) {
        self->flags = _descr_find_object((PyArray_Descr *)self);
    }

    """
     * 我们对 metadata 持有引用，所以在丢弃 Py_None 时不需要更改引用计数。
     """
    # 如果 metadata 是 Py_None，则将其设置为 NULL
    if (metadata == Py_None) {
        metadata = NULL;
    }

    # 如果 self 对象是日期时间类型，并且 metadata 不为 NULL
    if (PyDataType_ISDATETIME(self) && (metadata != NULL)) {
        PyObject *old_metadata;
        PyArray_DatetimeMetaData temp_dt_data;

        # 如果 metadata 不是元组或者元组大小不为 2，则抛出值错误异常，并返回空值
        if ((! PyTuple_Check(metadata)) || (PyTuple_Size(metadata) != 2)) {
            PyErr_Format(PyExc_ValueError,
                    "Invalid datetime dtype (metadata, c_metadata): %R",
                    metadata);
            return NULL;
        }

        # 将元组第二个元素转换为日期时间元数据，存储在 temp_dt_data 中
        if (convert_datetime_metadata_tuple_to_datetime_metadata(
                                    PyTuple_GET_ITEM(metadata, 1),
                                    &temp_dt_data,
                                    NPY_TRUE) < 0) {
            return NULL;
        }

        # 保存旧的 metadata 引用，并更新 self 对象的 metadata 和 c_metadata 字段
        old_metadata = self->metadata;
        self->metadata = PyTuple_GET_ITEM(metadata, 0);
        memcpy((char *) &((PyArray_DatetimeDTypeMetaData *)self->c_metadata)->meta,
               (char *) &temp_dt_data,
               sizeof(PyArray_DatetimeMetaData));
        Py_XINCREF(self->metadata);
        Py_XDECREF(old_metadata);
    }
    else {
        # 否则，保存旧的 metadata 引用，并更新 self 对象的 metadata 字段
        PyObject *old_metadata = self->metadata;
        self->metadata = metadata;
        Py_XINCREF(self->metadata);
        Py_XDECREF(old_metadata);
    }

    # 返回 Py_None
    Py_RETURN_NONE;
/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to DEFAULT type.
 *
 * any object with the .fields attribute and/or .itemsize attribute (if the
 *.fields attribute does not give the total size -- i.e. a partial record
 * naming).  If itemsize is given it must be >= size computed from fields
 *
 * The .fields attribute must return a convertible dictionary if present.
 * Result inherits from NPY_VOID.
*/
NPY_NO_EXPORT int
PyArray_DescrAlignConverter(PyObject *obj, PyArray_Descr **at)
{
    // 调用_convert_from_any函数，尝试从obj转换为PyArray_Descr，并强制对齐
    *at = _convert_from_any(obj, 1);
    // 返回转换是否成功的标志
    return (*at) ? NPY_SUCCEED : NPY_FAIL;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to NULL.
 */
NPY_NO_EXPORT int
PyArray_DescrAlignConverter2(PyObject *obj, PyArray_Descr **at)
{
    // 如果obj为Py_None，则将*at设置为NULL
    if (obj == Py_None) {
        *at = NULL;
        // 返回成功标志
        return NPY_SUCCEED;
    }
    else {
        // 否则调用PyArray_DescrAlignConverter进行转换
        return PyArray_DescrAlignConverter(obj, at);
    }
}



/*NUMPY_API
 *
 * returns a copy of the PyArray_Descr structure with the byteorder
 * altered:
 * no arguments:  The byteorder is swapped (in all subfields as well)
 * single argument:  The byteorder is forced to the given state
 * (in all subfields as well)
 *
 * Valid states:  ('big', '>') or ('little' or '<')
 * ('native', or '=')
 *
 * If a descr structure with | is encountered it's own
 * byte-order is not changed but any fields are:
 *
 *
 * Deep bytorder change of a data-type descriptor
 * *** Leaves reference count of self unchanged --- does not DECREF self ***
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewByteorder(PyArray_Descr *oself, char newendian)
{
    char endian;

    // 如果oself不是旧式的dtype，则返回TypeError
    if (!PyDataType_ISLEGACY(oself)) {
        PyErr_SetString(PyExc_TypeError,
            "Cannot use DescrNewByteOrder for this new style DTypes.");
        return NULL;
    }

    // 将oself转换为旧式的dtype结构体
    _PyArray_LegacyDescr *self = (_PyArray_LegacyDescr *)oself;
    // 创建一个oself的拷贝new
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNew(oself);
    if (new == NULL) {
        return NULL;
    }
    // 获取new的字节顺序
    endian = new->byteorder;
    // 如果endian不是NPY_IGNORE
    if (endian != NPY_IGNORE) {
        // 如果newendian为NPY_SWAP，则交换字节顺序
        if (newendian == NPY_SWAP) {
            /* swap byteorder */
            // 如果当前字节顺序是大端序，则设为反向的小端序，反之亦然
            if (PyArray_ISNBO(endian)) {
                endian = NPY_OPPBYTE;
            }
            else {
                endian = NPY_NATBYTE;
            }
            new->byteorder = endian;
        }
        // 如果newendian不是NPY_IGNORE，则设定新的字节顺序
        else if (newendian != NPY_IGNORE) {
            new->byteorder = newendian;
        }
    }
    // 检查是否有字段描述信息存在
    if (PyDataType_HASFIELDS(new)) {
        PyObject *newfields;    // 新字段的字典对象
        PyObject *key, *value;  // 键和值对象
        PyObject *newvalue;     // 新值对象
        PyObject *old;          // 旧值对象
        PyArray_Descr *newdescr; // 新的数组描述符对象
        Py_ssize_t pos = 0;     // 迭代器位置
        int len, i;             // 长度和迭代器变量

        // 创建一个新的空字典对象来存储新字段
        newfields = PyDict_New();
        if (newfields == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        /* 创建具有替换 PyArray_Descr 对象的新字典 */
        // 遍历原字段字典
        while (PyDict_Next(self->fields, &pos, &key, &value)) {
            // 跳过标题键
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            // 检查键是否为 Unicode 对象，值是否为元组对象且长度至少为2
            if (!PyUnicode_Check(key) || !PyTuple_Check(value) ||
                ((len=PyTuple_GET_SIZE(value)) < 2)) {
                continue;
            }
            // 获取原字段元组中的第一个元素，即旧的数组描述符对象
            old = PyTuple_GET_ITEM(value, 0);
            // 如果旧值不是数组描述符对象，则跳过
            if (!PyArray_DescrCheck(old)) {
                continue;
            }
            // 使用新的字节顺序创建一个新的数组描述符对象
            newdescr = PyArray_DescrNewByteorder(
                    (PyArray_Descr *)old, newendian);
            // 如果创建失败，释放资源并返回空
            if (newdescr == NULL) {
                Py_DECREF(newfields); Py_DECREF(new);
                return NULL;
            }
            // 创建一个新的元组对象，将新的数组描述符对象放在第一个位置
            newvalue = PyTuple_New(len);
            PyTuple_SET_ITEM(newvalue, 0, (PyObject *)newdescr);
            // 将原元组对象中的其余元素复制到新元组对象中
            for (i = 1; i < len; i++) {
                old = PyTuple_GET_ITEM(value, i);
                Py_INCREF(old);
                PyTuple_SET_ITEM(newvalue, i, old);
            }
            // 将新值元组对象添加到新字段字典中
            int ret = PyDict_SetItem(newfields, key, newvalue);
            Py_DECREF(newvalue);
            // 如果添加失败，释放资源并返回空
            if (ret < 0) {
                Py_DECREF(newfields);
                Py_DECREF(new);
                return NULL;
            }
        }
        // 释放原字段字典对象，并将新字段字典对象赋给新对象的字段
        Py_DECREF(new->fields);
        new->fields = newfields;
    }
    // 如果存在子数组对象
    if (new->subarray) {
        // 释放原子数组对象的基础描述符对象，并使用新的字节顺序创建一个新的描述符对象
        Py_DECREF(new->subarray->base);
        new->subarray->base = PyArray_DescrNewByteorder(
                self->subarray->base, newendian);
        // 如果创建失败，释放资源并返回空
        if (new->subarray->base == NULL) {
            Py_DECREF(new);
            return NULL;
        }
    }
    // 返回新的数组描述符对象
    return (PyArray_Descr *)new;
/*
 * 创建一个新的字节顺序的数组描述符对象。
 * 这个函数用于处理数组描述符的字节顺序。
 * 如果没有指定字节顺序，则默认为 NPY_SWAP。
 */
static PyObject *
arraydescr_newbyteorder(PyArray_Descr *self, PyObject *args)
{
    // 默认的字节顺序为 NPY_SWAP
    char endian = NPY_SWAP;

    // 尝试解析传入的参数，如果解析失败则返回 NULL
    if (!PyArg_ParseTuple(args, "|O&:newbyteorder", PyArray_ByteorderConverter,
                &endian)) {
        return NULL;
    }

    // 调用 PyArray_DescrNewByteorder 函数来创建新的字节顺序的数组描述符对象
    return (PyObject *)PyArray_DescrNewByteorder(self, endian);
}

/*
 * 获取数组描述符类中的特定项。
 * 当传入的参数个数不为 1 时，返回一个 TypeError 异常。
 * 否则，调用 Py_GenericAlias 函数处理参数并返回结果。
 */
static PyObject *
arraydescr_class_getitem(PyObject *cls, PyObject *args)
{
    // 检查传入参数的长度
    const Py_ssize_t args_len = PyTuple_Check(args) ? PyTuple_Size(args) : 1;

    // 如果参数个数不为 1，则返回一个带有错误信息的 TypeError 异常
    if (args_len != 1) {
        return PyErr_Format(PyExc_TypeError,
                            "Too %s arguments for %s",
                            args_len > 1 ? "many" : "few",
                            ((PyTypeObject *)cls)->tp_name);
    }

    // 调用 Py_GenericAlias 函数处理参数并返回结果
    return Py_GenericAlias(cls, args);
}

/*
 * 数组描述符对象的方法定义数组。
 * 包括用于 pickling 的方法、newbyteorder 方法以及用于 typing 的 __class_getitem__ 方法。
 */
static PyMethodDef arraydescr_methods[] = {
    /* for pickling */
    {"__reduce__",
        (PyCFunction)arraydescr_reduce,
        METH_VARARGS, NULL},
    {"__setstate__",
        (PyCFunction)arraydescr_setstate,
        METH_VARARGS, NULL},
    {"newbyteorder",
        (PyCFunction)arraydescr_newbyteorder,
        METH_VARARGS, NULL},
    /* for typing; requires python >= 3.9 */
    {"__class_getitem__",
        (PyCFunction)arraydescr_class_getitem,
        METH_CLASS | METH_O, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

/*
 * 检查结构化数据类型 'dtype' 是否具有简单的未对齐布局。
 * 当所有字段按顺序排列且没有对齐填充时返回 true。
 * 如果满足条件，可以通过字段名和数据类型的列表重新构建 dtype，而不需要额外的 dtype 参数。
 * 返回 1 表示具有简单的布局，返回 0 表示不是。
 */
NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype)
{
    PyObject *names, *fields, *key, *tup, *title;
    Py_ssize_t i, names_size;
    PyArray_Descr *fld_dtype;
    int fld_offset;
    npy_intp total_offset;

    /* 从 dtype 中获取一些属性 */
    names = PyDataType_NAMES(dtype);
    names_size = PyTuple_GET_SIZE(names);
    fields = PyDataType_FIELDS(dtype);

    /* 从零开始计算总偏移量 */
    total_offset = 0;

    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        if (key == NULL) {
            return 0;
        }
        tup = PyDict_GetItem(fields, key);
        if (tup == NULL) {
            return 0;
        }
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &fld_offset, &title)) {
            PyErr_Clear();
            return 0;
        }
        /* 如果这个字段不按照预期顺序排列，不是简单的布局 */
        if (total_offset != fld_offset) {
            return 0;
        }
        /* 获取下一个偏移量 */
        total_offset += fld_dtype->elsize;
    }

    /*
     * 如果 itemsize 不等于最终偏移量，不是简单的布局。
     */
    if (total_offset != dtype->elsize) {
        return 0;
    }

    /* 所有测试通过，具有简单的布局 */
    return 1;
}
    # 返回整数值 1
    return 1;
}

/*
 * The general dtype repr function.
 */
static PyObject *
arraydescr_repr(PyArray_Descr *dtype)
{
    PyObject *_numpy_dtype;
    PyObject *res;
    // 导入 numpy._core._dtype 模块
    _numpy_dtype = PyImport_ImportModule("numpy._core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    // 调用 _numpy_dtype 模块的 __repr__ 方法，返回 dtype 的字符串表示
    res = PyObject_CallMethod(_numpy_dtype, "__repr__", "O", dtype);
    Py_DECREF(_numpy_dtype);
    return res;
}
/*
 * The general dtype str function.
 */
static PyObject *
arraydescr_str(PyArray_Descr *dtype)
{
    PyObject *_numpy_dtype;
    PyObject *res;
    // 导入 numpy._core._dtype 模块
    _numpy_dtype = PyImport_ImportModule("numpy._core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    // 调用 _numpy_dtype 模块的 __str__ 方法，返回 dtype 的字符串表示
    res = PyObject_CallMethod(_numpy_dtype, "__str__", "O", dtype);
    Py_DECREF(_numpy_dtype);
    return res;
}

static PyObject *
arraydescr_richcompare(PyArray_Descr *self, PyObject *other, int cmp_op)
{
    // 将 other 转换为 PyArray_Descr 类型
    PyArray_Descr *new = _convert_from_any(other, 0);
    if (new == NULL) {
        /* Cannot convert `other` to dtype */
        PyErr_Clear();
        // 返回未实现的 NotImplemented
        Py_RETURN_NOTIMPLEMENTED;
    }

    npy_bool ret;
    switch (cmp_op) {
    case Py_LT:
        // 比较 self 和 new 是否等价，并且 self 可以转换为 new
        ret = !PyArray_EquivTypes(self, new) && PyArray_CanCastTo(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_LE:
        // 检查 self 是否可以转换为 new
        ret = PyArray_CanCastTo(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_EQ:
        // 检查 self 和 new 是否等价
        ret = PyArray_EquivTypes(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_NE:
        // 检查 self 和 new 是否不等价
        ret = !PyArray_EquivTypes(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_GT:
        // 比较 self 和 new 是否不等价，并且 new 可以转换为 self
        ret = !PyArray_EquivTypes(self, new) && PyArray_CanCastTo(new, self);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_GE:
        // 检查 new 是否可以转换为 self
        ret = PyArray_CanCastTo(new, self);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    default:
        Py_DECREF(new);
        // 返回未实现的 NotImplemented
        Py_RETURN_NOTIMPLEMENTED;
    }
}

static int
descr_nonzero(PyObject *NPY_UNUSED(self))
{
    /* `bool(np.dtype(...)) == True` for all dtypes. Needed to override default
     * nonzero implementation, which checks if `len(object) > 0`. */
    // 所有的 dtype 的 bool() 操作都返回 True，用于覆盖默认的非零判断实现
    return 1;
}

static PyNumberMethods descr_as_number = {
    .nb_bool = (inquiry)descr_nonzero,
};

/*************************************************************************
 ****************   Implement Mapping Protocol ***************************
 *************************************************************************/

static Py_ssize_t
descr_length(PyObject *self0)
{
    PyArray_Descr *self = (PyArray_Descr *)self0;

    if (PyDataType_HASFIELDS(self)) {
        // 如果 self 具有字段信息，返回字段数目
        return PyTuple_GET_SIZE(PyDataType_NAMES(self));
    }
    else {
        // 如果 self 没有字段信息，返回 0
        return 0;
    }
}

static PyObject *
descr_repeat(PyObject *self, Py_ssize_t length)
{
    PyObject *tup;
    PyArray_Descr *new;
    // 代码继续...
    # 检查长度是否小于 0
    if (length < 0) {
        # 如果小于 0，则返回一个相应的异常，并包含错误信息和长度值
        return PyErr_Format(PyExc_ValueError,
                "Array length must be >= 0, not %"NPY_INTP_FMT, (npy_intp)length);
    }
    # 使用给定的 self 和 length 构建一个 Python 元组对象
    tup = Py_BuildValue("O" NPY_SSIZE_T_PYFMT, self, length);
    # 检查元组对象是否构建成功
    if (tup == NULL) {
        # 如果构建失败，则返回 NULL
        return NULL;
    }
    # 将构建的元组对象转换为适当的数据结构 new
    new = _convert_from_any(tup, 0);
    # 释放元组对象的引用计数
    Py_DECREF(tup);
    # 返回转换后的新对象
    return (PyObject *)new;
static int
_check_has_fields(PyArray_Descr *self)
{
    // 检查给定的 NumPy 数据类型描述符是否具有字段信息
    if (!PyDataType_HASFIELDS(self)) {
        // 若没有字段信息，则抛出一个 Key 错误异常
        PyErr_Format(PyExc_KeyError, "There are no fields in dtype %S.", self);
        return -1;
    }
    else {
        // 如果有字段信息，则返回 0 表示成功
        return 0;
    }
}

static PyObject *
_subscript_by_name(_PyArray_LegacyDescr *self, PyObject *op)
{
    // 根据字段名从字段字典中获取对应的对象
    PyObject *obj = PyDict_GetItemWithError(self->fields, op);
    if (obj == NULL) {
        // 如果获取失败，且没有设置过错误状态，则抛出 Key 错误异常
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_KeyError,
                    "Field named %R not found.", op);
        }
        return NULL;
    }
    // 从获取到的对象中取出描述符对象
    PyObject *descr = PyTuple_GET_ITEM(obj, 0);
    // 增加描述符的引用计数并返回
    Py_INCREF(descr);
    return descr;
}

static PyObject *
_subscript_by_index(_PyArray_LegacyDescr *self, Py_ssize_t i)
{
    // 根据索引从字段名元组中获取字段名对象
    PyObject *name = PySequence_GetItem(self->names, i);
    PyObject *ret;
    if (name == NULL) {
        // 如果获取失败，则抛出索引错误异常
        PyErr_Format(PyExc_IndexError,
                     "Field index %zd out of range.", i);
        return NULL;
    }
    // 根据字段名对象调用 _subscript_by_name 获取对应的描述符对象
    ret = _subscript_by_name(self, name);
    // 减少字段名对象的引用计数
    Py_DECREF(name);
    return ret;
}

static npy_bool
_is_list_of_strings(PyObject *obj)
{
    int seqlen, i;
    // 检查对象是否为列表类型
    if (!PyList_CheckExact(obj)) {
        return NPY_FALSE;
    }
    // 获取列表的长度
    seqlen = PyList_GET_SIZE(obj);
    // 遍历列表中的每个元素，检查是否都是 Unicode 字符串类型
    for (i = 0; i < seqlen; i++) {
        PyObject *item = PyList_GET_ITEM(obj, i);
        if (!PyUnicode_Check(item)) {
            return NPY_FALSE;
        }
    }

    return NPY_TRUE;
}

NPY_NO_EXPORT PyArray_Descr *
arraydescr_field_subset_view(_PyArray_LegacyDescr *self, PyObject *ind)
{
    int seqlen, i;
    PyObject *fields = NULL;
    PyObject *names = NULL;

    // 获取索引对象的长度
    seqlen = PySequence_Size(ind);
    if (seqlen == -1) {
        return NULL;
    }

    // 创建一个新的空字典对象用于存储字段
    fields = PyDict_New();
    if (fields == NULL) {
        goto fail;
    }
    // 创建一个新的元组对象来存储字段名
    names = PyTuple_New(seqlen);
    if (names == NULL) {
        goto fail;
    }
    // 遍历输入的序列，处理每个元素
    for (i = 0; i < seqlen; i++) {
        PyObject *name;
        PyObject *tup;

        // 获取序列中第i个元素作为字段名
        name = PySequence_GetItem(ind, i);
        if (name == NULL) {
            // 如果获取字段名失败，跳转到错误处理标签fail
            goto fail;
        }

        /* 让字段名元组现在就获取一个引用，这样如果后面发生错误，我们就不需要
         * 在释放引用之前再次递减它。
         */
        PyTuple_SET_ITEM(names, i, name);

        // 在self->fields字典中查找字段名对应的元组tup
        tup = PyDict_GetItemWithError(self->fields, name);
        if (tup == NULL) {
            // 如果字段名在self->fields中不存在，设置KeyError异常
            if (!PyErr_Occurred()) {
                PyErr_SetObject(PyExc_KeyError, name);
            }
            // 跳转到错误处理标签fail
            goto fail;
        }

        /* 禁止使用标题作为索引 */
        if (PyTuple_Size(tup) == 3) {
            // 如果元组tup的大小为3，表示有标题，获取标题对象
            PyObject *title = PyTuple_GET_ITEM(tup, 2);
            // 比较标题和字段名是否相等
            int titlecmp = PyObject_RichCompareBool(title, name, Py_EQ);
            if (titlecmp < 0) {
                // 比较出错，跳转到错误处理标签fail
                goto fail;
            }
            if (titlecmp == 1) {
                /* 如果标题 == 字段名，说明传入的是标题而不是字段名 */
                PyErr_SetString(PyExc_KeyError,
                            "cannot use field titles in multi-field index");
                // 跳转到错误处理标签fail
                goto fail;
            }
            // 将标题和元组tup添加到fields字典中
            if (PyDict_SetItem(fields, title, tup) < 0) {
                // 添加失败，跳转到错误处理标签fail
                goto fail;
            }
        }
        /* 禁止重复的字段索引 */
        if (PyDict_Contains(fields, name)) {
            // 如果fields字典中已经包含字段名name
            PyObject *msg = NULL;
            PyObject *fmt = PyUnicode_FromString(
                                   "duplicate field of name {!r}");
            if (fmt != NULL) {
                // 格式化错误消息
                msg = PyObject_CallMethod(fmt, "format", "O", name);
                Py_DECREF(fmt);
            }
            // 设置ValueError异常，并传入错误消息
            PyErr_SetObject(PyExc_ValueError, msg);
            Py_XDECREF(msg);
            // 跳转到错误处理标签fail
            goto fail;
        }
        // 将字段名name和元组tup添加到fields字典中
        if (PyDict_SetItem(fields, name, tup) < 0) {
            // 添加失败，跳转到错误处理标签fail
            goto fail;
        }
    }

    // 创建一个新的数组描述符view_dtype，类型为NPY_VOID
    _PyArray_LegacyDescr *view_dtype = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    if (view_dtype == NULL) {
        // 如果创建失败，跳转到错误处理标签fail
        goto fail;
    }
    // 设置view_dtype的元素大小、字段名元组、字段字典和标志
    view_dtype->elsize = self->elsize;
    view_dtype->names = names;
    view_dtype->fields = fields;
    view_dtype->flags = self->flags;
    // 返回创建的数组描述符view_dtype
    return (PyArray_Descr *)view_dtype;
fail:
    Py_XDECREF(fields);  # 释放 fields 对象的引用计数
    Py_XDECREF(names);   # 释放 names 对象的引用计数
    return NULL;         # 返回 NULL 表示函数执行失败
}

static PyObject *
descr_subscript(PyArray_Descr *self, PyObject *op)
{
    _PyArray_LegacyDescr *lself = (_PyArray_LegacyDescr *)self;  # 将 self 转换为 _PyArray_LegacyDescr 类型

    if (_check_has_fields(self) < 0) {  # 检查 self 是否有字段，小于 0 表示出错
        return NULL;  # 返回 NULL 表示出错
    }

    if (PyUnicode_Check(op)) {  # 检查 op 是否为 Unicode 对象
        return _subscript_by_name(lself, op);  # 根据字段名 op 返回对应的子描述符
    }
    else if (_is_list_of_strings(op)) {  # 检查 op 是否为字符串列表
        return (PyObject *)arraydescr_field_subset_view(lself, op);  # 返回字段子集的视图
    }
    else {
        Py_ssize_t i = PyArray_PyIntAsIntp(op);  # 将 op 转换为 Py_ssize_t 类型
        if (error_converting(i)) {  # 检查是否在转换过程中出错
            /* 如果转换为整数出现类型错误，调整错误消息 */
            PyObject *err = PyErr_Occurred();  # 获取当前的错误对象
            if (PyErr_GivenExceptionMatches(err, PyExc_TypeError)) {  # 检查错误类型是否为 TypeError
                PyErr_SetString(PyExc_TypeError,
                        "Field key must be an integer field offset, "
                        "single field name, or list of field names.");  # 设置错误消息
            }
            return NULL;  # 返回 NULL 表示出错
        }
        return _subscript_by_index(lself, i);  # 根据索引 i 返回对应的子描述符
    }
}

static PySequenceMethods descr_as_sequence = {
    (lenfunc) descr_length,                  /* sq_length */  # 返回序列长度的函数指针
    (binaryfunc) NULL,                       /* sq_concat */  # 序列连接函数指针，未实现
    (ssizeargfunc) descr_repeat,             /* sq_repeat */  # 序列重复函数指针
    (ssizeargfunc) NULL,                     /* sq_item */  # 获取序列项的函数指针，未实现
    (ssizessizeargfunc) NULL,                /* sq_slice */  # 获取序列切片的函数指针，未实现
    (ssizeobjargproc) NULL,                  /* sq_ass_item */  # 设置序列项的函数指针，未实现
    (ssizessizeobjargproc) NULL,             /* sq_ass_slice */  # 设置序列切片的函数指针，未实现
    (objobjproc) NULL,                       /* sq_contains */  # 序列包含检查的函数指针，未实现
    (binaryfunc) NULL,                       /* sq_inplace_concat */  # 序列原地连接的函数指针，未实现
    (ssizeargfunc) NULL,                     /* sq_inplace_repeat */  # 序列原地重复的函数指针，未实现
};

static PyMappingMethods descr_as_mapping = {
    descr_length,                                /* mp_length*/  # 映射长度函数指针
    (binaryfunc)descr_subscript,                 /* mp_subscript*/  # 映射子脚本函数指针
    (objobjargproc)NULL,                         /* mp_ass_subscript*/  # 映射赋值子脚本函数指针，未实现
};

/****************** End of Mapping Protocol ******************************/

/*
 * NOTE: Since this is a MetaClass, the name has Full appended here, the
 *       correct name of the type is PyArrayDescr_Type.
 */
NPY_NO_EXPORT PyArray_DTypeMeta PyArrayDescr_TypeFull = {
    {{
        /* NULL represents `type`, this is set to DTypeMeta at import time */
        // 使用 NULL 表示 `type`，在导入时设置为 DTypeMeta
        PyVarObject_HEAD_INIT(NULL, 0)
        // 初始化 Python 变量对象头部，指定初始值为 NULL 和大小为 0

        .tp_name = "numpy.dtype",
        // 设置类型对象的名称为 "numpy.dtype"
        
        .tp_basicsize = sizeof(PyArray_Descr),
        // 设置类型对象的基本大小为 PyArray_Descr 结构体的大小
        
        .tp_dealloc = (destructor)arraydescr_dealloc,
        // 设置类型对象的析构函数为 arraydescr_dealloc
        
        .tp_repr = (reprfunc)arraydescr_repr,
        // 设置类型对象的表示函数为 arraydescr_repr
        
        .tp_as_number = &descr_as_number,
        // 设置类型对象的数字操作结构体指针为 descr_as_number
        
        .tp_as_sequence = &descr_as_sequence,
        // 设置类型对象的序列操作结构体指针为 descr_as_sequence
        
        .tp_as_mapping = &descr_as_mapping,
        // 设置类型对象的映射操作结构体指针为 descr_as_mapping
        
        .tp_str = (reprfunc)arraydescr_str,
        // 设置类型对象的字符串表示函数为 arraydescr_str
        
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        // 设置类型对象的标志为默认标志和基类型标志的按位或
        
        .tp_richcompare = (richcmpfunc)arraydescr_richcompare,
        // 设置类型对象的富比较函数为 arraydescr_richcompare
        
        .tp_methods = arraydescr_methods,
        // 设置类型对象的方法列表为 arraydescr_methods
        
        .tp_members = arraydescr_members,
        // 设置类型对象的成员列表为 arraydescr_members
        
        .tp_getset = arraydescr_getsets,
        // 设置类型对象的属性获取设置函数列表为 arraydescr_getsets
        
        .tp_new = arraydescr_new,
        // 设置类型对象的新建函数为 arraydescr_new
    },},
    // 结束 PyVarObject_HEAD_INIT 初始化

    .singleton = NULL,
    // 设置 singleton 属性为 NULL

    .type_num = -1,
    // 设置 type_num 属性为 -1，表示未指定具体的类型编号

    .scalar_type = NULL,
    // 设置 scalar_type 属性为 NULL，表示未指定具体的标量类型

    .flags = NPY_DT_ABSTRACT,
    // 设置 flags 属性为 NPY_DT_ABSTRACT，表示这是一个抽象数据类型
};



# 结束了一个代码块，该代码块可能是一个函数、循环、条件语句或其他代码结构的结束
```