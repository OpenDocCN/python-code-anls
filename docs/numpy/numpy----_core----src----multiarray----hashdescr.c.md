# `.\numpy\numpy\_core\src\multiarray\hashdescr.c`

```
/*
 * Define NPY_NO_DEPRECATED_API to the current NPY_API_VERSION
 * and _MULTIARRAYMODULE for compilation purposes.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * Clean PY_SSIZE_T_CLEAN to ensure Python.h defines Py_ssize_t.
 * Include necessary headers for Python and NumPy.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/*
 * Include the configuration header for NumPy and hash descriptor header.
 */
#include "npy_config.h"
#include "hashdescr.h"

/*
 * How does this work ? The hash is computed from a list which contains all the
 * information specific to a type. The hard work is to build the list
 * (_array_descr_walk). The list is built as follows:
 *      * If the dtype is builtin (no fields, no subarray), then the list
 *      contains 6 items which uniquely define one dtype (_array_descr_builtin)
 *      * If the dtype is a compound array, one walk on each field. For each
 *      field, we append title, names, offset to the final list used for
 *      hashing, and then append the list recursively built for each
 *      corresponding dtype (_array_descr_walk_fields)
 *      * If the dtype is a subarray, one adds the shape tuple to the list, and
 *      then append the list recursively built for each corresponding dtype
 *      (_array_descr_walk_subarray)
 */

/*
 * Static function declarations for internal use.
 */
static int _is_array_descr_builtin(PyArray_Descr* descr);
static int _array_descr_walk(PyArray_Descr* descr, PyObject *l);
static int _array_descr_walk_fields(PyObject *names, PyObject* fields, PyObject* l);
static int _array_descr_builtin(PyArray_Descr* descr, PyObject *l);

/*
 * Normalize endian character: always return 'I', '<' or '>'
 */
static char _normalize_byteorder(char byteorder)
{
    switch(byteorder) {
        case '=':
            if (PyArray_GetEndianness() == NPY_CPU_BIG) {
                return '>';
            }
            else {
                return '<';
            }
        default:
            return byteorder;
    }
}

/*
 * Return true if descr is a builtin type
 */
static int _is_array_descr_builtin(PyArray_Descr* descr)
{
    if (PyDataType_HASFIELDS(descr)) {
        return 0;
    }
    if (PyDataType_HASSUBARRAY(descr)) {
        return 0;
    }
    return 1;
}

/*
 * Add to l all the items which uniquely define a builtin type
 */
static int _array_descr_builtin(PyArray_Descr* descr, PyObject *l)
{
    Py_ssize_t i;
    PyObject *t, *item;
    char nbyteorder = _normalize_byteorder(descr->byteorder);

    /*
     * For builtin type, hash relies on : kind + byteorder + flags +
     * type_num + elsize + alignment
     */
    t = Py_BuildValue("(cccii)", descr->kind, nbyteorder,
            descr->flags, descr->elsize, descr->alignment);

    for(i = 0; i < PyTuple_Size(t); ++i) {
        item = PyTuple_GetItem(t, i);
        if (item == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Error while computing builtin hash");
            goto clean_t;
        }
        PyList_Append(l, item);
    }

    Py_DECREF(t);
    return 0;

clean_t:
    Py_DECREF(t);
    return -1;
}

/*
 * Walk inside the fields and add every item which will be used for hashing
 * into the list l
 *
 * Return 0 on success
 */
/*
 * 遍历数组描述字段，将字段名、描述符、偏移量添加到列表 l 中
 *
 * 如果 names 不是元组，则设置异常并返回 -1
 */
static int _array_descr_walk_fields(PyObject *names, PyObject* fields, PyObject* l)
{
    PyObject *key, *value, *foffset, *fdescr, *ftitle;
    Py_ssize_t pos = 0;
    int st;

    if (!PyTuple_Check(names)) {
        PyErr_SetString(PyExc_SystemError,
                "(Hash) names is not a tuple ???");
        return -1;
    }
    /*
     * 如果 fields 不是字典，则设置异常并返回 -1
     */
    if (!PyDict_Check(fields)) {
        PyErr_SetString(PyExc_SystemError,
                "(Hash) fields is not a dict ???");
        return -1;
    }

    for (pos = 0; pos < PyTuple_GET_SIZE(names); pos++) {
        /*
         * 对于每个字段，将键、描述符、偏移量添加到 l 中
         */
        key = PyTuple_GET_ITEM(names, pos);
        value = PyDict_GetItem(fields, key);
        /* XXX: 这些检查是否必要？ */
        /*
         * 如果值为空，则设置异常并返回 -1
         */
        if (value == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) names and fields inconsistent ???");
            return -1;
        }
        /*
         * 如果键不是 Unicode 字符串，则设置异常并返回 -1
         */
        if (!PyUnicode_Check(key)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) key of dtype dict not a string ???");
            return -1;
        }
        /*
         * 如果值不是元组，则设置异常并返回 -1
         */
        if (!PyTuple_Check(value)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) value of dtype dict not a dtype ???");
            return -1;
        }
        /*
         * 如果元组中的项少于 2 个，则设置异常并返回 -1
         */
        if (PyTuple_GET_SIZE(value) < 2) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Less than 2 items in dtype dict ???");
            return -1;
        }
        /*
         * 将键添加到列表 l 中
         */
        PyList_Append(l, key);

        /*
         * 获取元组中的描述符，检查是否为有效的描述符
         */
        fdescr = PyTuple_GET_ITEM(value, 0);
        if (!PyArray_DescrCheck(fdescr)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) First item in compound dtype tuple not a descr ???");
            return -1;
        }
        else {
            Py_INCREF(fdescr);
            /*
             * 递归调用 _array_descr_walk 处理描述符
             */
            st = _array_descr_walk((PyArray_Descr*)fdescr, l);
            Py_DECREF(fdescr);
            if (st) {
                return -1;
            }
        }

        /*
         * 获取元组中的偏移量，检查是否为整数类型
         */
        foffset = PyTuple_GET_ITEM(value, 1);
        if (!PyLong_Check(foffset)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Second item in compound dtype tuple not an int ???");
            return -1;
        }
        else {
            /*
             * 将偏移量添加到列表 l 中
             */
            PyList_Append(l, foffset);
        }

        /*
         * 如果元组中的项数大于 2，则获取并添加第三项到列表 l 中
         */
        if (PyTuple_GET_SIZE(value) > 2) {
            ftitle = PyTuple_GET_ITEM(value, 2);
            PyList_Append(l, ftitle);
        }
    }

    return 0;
}

/*
 * 遍历子数组描述，将其形状和描述符本身添加到列表 l 中
 *
 * 成功时返回 0
 */
static int _array_descr_walk_subarray(PyArray_ArrayDescr* adescr, PyObject *l)
{
    PyObject *item;
    Py_ssize_t i;
    int st;

    /*
     * 将形状和描述符本身添加到要哈希的对象列表中
     */
    // 检查 adescr 结构体中的 shape 成员是否为元组类型
    if (PyTuple_Check(adescr->shape)) {
        // 遍历元组中的每个元素
        for(i = 0; i < PyTuple_Size(adescr->shape); ++i) {
            // 获取元组中的第 i 个元素
            item = PyTuple_GetItem(adescr->shape, i);
            // 如果获取的元素为空，则设置异常并返回错误码 -1
            if (item == NULL) {
                PyErr_SetString(PyExc_SystemError,
                        "(Hash) Error while getting shape item of subarray dtype ???");
                return -1;
            }
            // 将获取的元素添加到列表 l 中
            PyList_Append(l, item);
        }
    }
    // 如果 adescr 结构体中的 shape 成员是整数类型
    else if (PyLong_Check(adescr->shape)) {
        // 将整数类型的 shape 添加到列表 l 中
        PyList_Append(l, adescr->shape);
    }
    // 如果 shape 不是元组也不是整数类型，则设置异常并返回错误码 -1
    else {
        PyErr_SetString(PyExc_SystemError,
                "(Hash) Shape of subarray dtype neither a tuple or int ???");
        return -1;
    }

    // 增加 adescr 结构体中 base 成员的引用计数
    Py_INCREF(adescr->base);
    // 递归调用 _array_descr_walk 函数，处理 adescr 结构体中 base 成员，将结果保存在列表 l 中
    st = _array_descr_walk(adescr->base, l);
    // 减少 adescr 结构体中 base 成员的引用计数
    Py_DECREF(adescr->base);

    // 返回 _array_descr_walk 函数的返回值
    return st;
/*
 * 'Root' function to walk into a dtype. May be called recursively
 */
static int _array_descr_walk(PyArray_Descr* descr, PyObject *l)
{
    int st;

    // 检查描述符是否是内置数组描述符，如果是则调用内置函数处理
    if (_is_array_descr_builtin(descr)) {
        return _array_descr_builtin(descr, l);
    }
    else {
        // 将描述符转换为旧版数组描述符对象
        _PyArray_LegacyDescr *ldescr = (_PyArray_LegacyDescr *)descr;
        // 如果字段不为空，则遍历字段进行处理
        if(ldescr->fields != NULL && ldescr->fields != Py_None) {
            st = _array_descr_walk_fields(ldescr->names, ldescr->fields, l);
            if (st) {
                return -1;
            }
        }
        // 如果存在子数组描述符，则递归处理子数组描述符
        if(ldescr->subarray != NULL) {
            st = _array_descr_walk_subarray(ldescr->subarray, l);
            if (st) {
                return -1;
            }
        }
    }

    return 0;
}

/*
 * Return 0 if successful
 */
static int _PyArray_DescrHashImp(PyArray_Descr *descr, npy_hash_t *hash)
{
    PyObject *l, *tl;
    int st;

    // 创建一个空列表对象
    l = PyList_New(0);
    if (l == NULL) {
        return -1;
    }

    // 递归遍历描述符结构，将信息存入列表对象
    st = _array_descr_walk(descr, l);
    if (st) {
        Py_DECREF(l);
        return -1;
    }

    /*
     * Convert the list to tuple and compute the tuple hash using python
     * builtin function
     */
    // 将列表对象转换为元组对象，并使用 Python 内置的哈希函数计算元组的哈希值
    tl = PyList_AsTuple(l);
    Py_DECREF(l);
    if (tl == NULL)
        return -1;

    // 将计算得到的哈希值存入指定的变量
    *hash = PyObject_Hash(tl);
    Py_DECREF(tl);
    if (*hash == -1) {
        /* XXX: does PyObject_Hash set an exception on failure ? */
#if 0
        PyErr_SetString(PyExc_SystemError,
                "(Hash) Error while hashing final tuple");
#endif
        return -1;
    }

    return 0;
}

NPY_NO_EXPORT npy_hash_t
PyArray_DescrHash(PyObject* odescr)
{
    PyArray_Descr *descr;
    int st;

    // 检查输入的对象是否为有效的数组描述符，否则报错
    if (!PyArray_DescrCheck(odescr)) {
        PyErr_SetString(PyExc_ValueError,
                "PyArray_DescrHash argument must be a type descriptor");
        return -1;
    }
    descr = (PyArray_Descr*)odescr;

    // 如果描述符的哈希值为-1，表示尚未计算哈希值，需要计算
    if (descr->hash == -1) {
        // 调用内部函数计算描述符的哈希值
        st = _PyArray_DescrHashImp(descr, &descr->hash);
        if (st) {
            return -1;
        }
    }

    // 返回计算得到的哈希值
    return descr->hash;
}
```