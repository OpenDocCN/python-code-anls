# `.\numpy\numpy\_core\src\multiarray\stringdtype\dtype.c`

```
/* StringDType 类的实现 */
#define PY_SSIZE_T_CLEAN  // 清除 Python 中使用的 ssize_t 的宏定义
#include <Python.h>       // Python C API 的主头文件
#include "structmember.h" // Python 结构成员访问的辅助宏

#define NPY_NO_DEPRECATED_API NPY_API_VERSION  // 使用最新的 NumPy API 版本
#define _MULTIARRAYMODULE  // 多维数组模块标识符

#include "numpy/arrayobject.h"    // NumPy 数组对象的头文件
#include "numpy/ndarraytypes.h"   // NumPy 数组类型的头文件
#include "numpy/npy_math.h"       // NumPy 中的数学函数
#include "static_string.h"        // 静态字符串实用工具
#include "dtypemeta.h"            // 数据类型元信息
#include "dtype.h"                // 数据类型定义
#include "casts.h"                // 类型转换函数
#include "gil_utils.h"            // 全局解释器锁 (GIL) 工具函数
#include "conversion_utils.h"     // 类型转换实用工具
#include "npy_import.h"           // NumPy 导入实用工具
#include "multiarraymodule.h"     // 多维数组模块核心

/*
 * 内部辅助函数，用于创建新实例
 */
PyObject *
new_stringdtype_instance(PyObject *na_object, int coerce)
{
    PyObject *new =
            PyArrayDescr_Type.tp_new((PyTypeObject *)&PyArray_StringDType, NULL, NULL); // 创建新的 PyArrayDescr_Type 实例

    if (new == NULL) {
        return NULL;  // 如果创建失败，返回空指针
    }

    char *default_string_buf = NULL;  // 默认字符串缓冲区
    char *na_name_buf = NULL;          // NA 名称缓冲区

    npy_string_allocator *allocator = NpyString_new_allocator(PyMem_RawMalloc, PyMem_RawFree,
                                                              PyMem_RawRealloc);  // 创建字符串分配器

    if (allocator == NULL) {
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to create string allocator");  // 如果分配器创建失败，设置内存错误并跳转到 fail 标签
        goto fail;
    }

    npy_static_string default_string = {0, NULL};  // 默认静态字符串结构体
    npy_static_string na_name = {0, NULL};         // NA 名称静态字符串结构体

    Py_XINCREF(na_object);  // 增加 NA 对象的引用计数
    ((PyArray_StringDTypeObject *)new)->na_object = na_object;  // 设置新实例的 NA 对象指针
    int has_null = na_object != NULL;  // 是否具有 NULL 值
    int has_nan_na = 0;  // 是否具有 NaN 或 NA 值
    int has_string_na = 0;  // 是否具有字符串类型的 NA 值
    if (has_null) {
        // 如果存在缺失值标记

        // 首先检查是否为字符串类型
        if (PyUnicode_Check(na_object)) {
            // 如果是 Python 字符串对象
            has_string_na = 1;
            // 获取字符串对象的 UTF-8 编码及其大小
            Py_ssize_t size = 0;
            const char *buf = PyUnicode_AsUTF8AndSize(na_object, &size);
            if (buf == NULL) {
                goto fail;
            }
            // 分配内存并拷贝字符串数据
            default_string.buf = PyMem_RawMalloc(size);
            if (default_string.buf == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
            memcpy((char *)default_string.buf, buf, size);
            default_string.size = size;
        }
        else {
            // 若非字符串对象，则视为类 NaN 的对象
            PyObject *ne_result = PyObject_RichCompare(na_object, na_object, Py_NE);
            if (ne_result == NULL) {
                goto fail;
            }
            // 检查比较结果是否真值
            int is_truthy = PyObject_IsTrue(ne_result);
            if (is_truthy == -1) {
                PyErr_Clear();
                has_nan_na = 1;
            }
            else if (is_truthy == 1) {
                has_nan_na = 1;
            }
            Py_DECREF(ne_result);
        }

        // 转换缺失值对象为字符串
        PyObject *na_pystr = PyObject_Str(na_object);
        if (na_pystr == NULL) {
            goto fail;
        }

        // 获取字符串对象的 UTF-8 编码及其大小
        Py_ssize_t size = 0;
        const char *utf8_ptr = PyUnicode_AsUTF8AndSize(na_pystr, &size);
        if (utf8_ptr == NULL) {
            Py_DECREF(na_pystr);
            goto fail;
        }
        // 分配内存并拷贝字符串数据
        na_name.buf = PyMem_RawMalloc(size);
        if (na_name.buf == NULL) {
            Py_DECREF(na_pystr);
            goto fail;
        }
        memcpy((char *)na_name.buf, utf8_ptr, size);
        na_name.size = size;
        Py_DECREF(na_pystr);
    }

    // 转换为 PyArray_StringDTypeObject 类型的指针
    PyArray_StringDTypeObject *snew = (PyArray_StringDTypeObject *)new;

    // 设置结构体成员变量值
    snew->has_nan_na = has_nan_na;
    snew->has_string_na = has_string_na;
    snew->coerce = coerce;
    snew->allocator = allocator;
    snew->array_owned = 0;
    snew->na_name = na_name;
    snew->default_string = default_string;

    // 转换为 PyArray_Descr 类型的指针
    PyArray_Descr *base = (PyArray_Descr *)new;
    // 设置描述符的大小和对齐方式
    base->elsize = SIZEOF_NPY_PACKED_STATIC_STRING;
    base->alignment = ALIGNOF_NPY_PACKED_STATIC_STRING;
    // 设置标志位
    base->flags |= NPY_NEEDS_INIT;
    base->flags |= NPY_LIST_PICKLE;
    base->flags |= NPY_ITEM_REFCOUNT;
    // 设置数据类型编号和类型标识
    base->type_num = NPY_VSTRING;
    base->kind = NPY_VSTRINGLTR;
    base->type = NPY_VSTRINGLTR;

    // 返回新创建的描述符对象
    return new;
fail:
    // 减少 new 对象的引用计数，因为返回 NULL 意味着无法返回新对象
    Py_DECREF(new);
    // 检查并释放默认字符串缓冲区
    if (default_string_buf != NULL) {
        PyMem_RawFree(default_string_buf);
    }
    // 检查并释放 na_name_buf 字符串缓冲区
    if (na_name_buf != NULL) {
        PyMem_RawFree(na_name_buf);
    }
    // 检查并释放分配器
    if (allocator != NULL) {
        NpyString_free_allocator(allocator);
    }
    // 返回 NULL 指示出错
    return NULL;
}

static int
na_eq_cmp(PyObject *a, PyObject *b) {
    // 检查对象是否完全相同，包括 None 和 Pandas.NA 这样的单例对象
    if (a == b) {
        return 1;
    }
    // 如果其中一个对象为 NULL，则返回不相等
    if (a == NULL || b == NULL) {
        return 0;
    }
    // 如果两个对象都是浮点数对象，则进行 NaN 检查
    if (PyFloat_Check(a) && PyFloat_Check(b)) {
        // 获取浮点数值并检查是否为 NaN
        double a_float = PyFloat_AsDouble(a);
        if (a_float == -1.0 && PyErr_Occurred()) {
            return -1;  // 出错时返回 -1
        }
        double b_float = PyFloat_AsDouble(b);
        if (b_float == -1.0 && PyErr_Occurred()) {
            return -1;  // 出错时返回 -1
        }
        // 如果两者均为 NaN，则视为相等
        if (npy_isnan(a_float) && npy_isnan(b_float)) {
            return 1;
        }
    }
    // 使用 PyObject_RichCompareBool 函数比较对象是否相等
    int ret = PyObject_RichCompareBool(a, b, Py_EQ);
    if (ret == -1) {
        PyErr_Clear();  // 清除异常状态
        return 0;
    }
    return ret;
}

// 设置确定 dtype 实例间相等性的逻辑规则
int
_eq_comparison(int scoerce, int ocoerce, PyObject *sna, PyObject *ona)
{
    // 如果 scoerce 与 ocoerce 不相等，则返回不相等
    if (scoerce != ocoerce) {
        return 0;
    }
    // 调用 na_eq_cmp 函数比较 sna 和 ona 对象是否相等
    return na_eq_cmp(sna, ona);
}

// 当处理不同 dtype 的混合时，用于确定返回的正确 dtype 实例
NPY_NO_EXPORT int
stringdtype_compatible_na(PyObject *na1, PyObject *na2, PyObject **out_na) {
    // 如果 na1 和 na2 都不为空，则比较它们是否相等
    if ((na1 != NULL) && (na2 != NULL)) {
        int na_eq = na_eq_cmp(na1, na2);

        // 如果比较出错，返回 -1
        if (na_eq < 0) {
            return -1;
        }
        // 如果 na1 和 na2 不相等，抛出类型错误异常
        else if (na_eq == 0) {
            PyErr_Format(PyExc_TypeError,
                         "Cannot find a compatible null string value for "
                         "null strings '%R' and '%R'", na1, na2);
            return -1;
        }
    }
    // 如果 out_na 不为空，则将 na1 或 na2 赋给 *out_na
    if (out_na != NULL) {
        *out_na = na1 ? na1 : na2;
    }
    return 0;
}

/*
 * 用于确定处理不同 dtype（例如从标量列表创建数组时）时返回的正确 dtype 实例
 */
static PyArray_StringDTypeObject *
common_instance(PyArray_StringDTypeObject *dtype1, PyArray_StringDTypeObject *dtype2)
{
    PyObject *out_na_object = NULL;

    // 检查 na1 和 na2 是否兼容，如果不兼容，返回 NULL，并抛出类型错误异常
    if (stringdtype_compatible_na(
                dtype1->na_object, dtype2->na_object, &out_na_object) == -1) {
        PyErr_Format(PyExc_TypeError,
                     "Cannot find common instance for incompatible dtypes "
                     "'%R' and '%R'", (PyObject *)dtype1, (PyObject *)dtype2);
        return NULL;
    }

    // 返回新的 string dtype 实例，基于 out_na_object 和 dtype1 是否强制转换的条件
    return (PyArray_StringDTypeObject *)new_stringdtype_instance(
            out_na_object, dtype1->coerce && dtype1->coerce);
}
/*
 *  用于确定用于数据类型提升的正确“常见”数据类型。
 *  cls 总是 PyArray_StringDType，other 是任意其他数据类型。
 */
static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // 如果 other 的类型码是 NPY_UNICODE
    if (other->type_num == NPY_UNICODE) {
        /*
         *  我们需要从 unicode 进行类型转换，因此允许 unicode 转换为 PyArray_StringDType
         */
        Py_INCREF(cls);  // 增加 cls 的引用计数
        return cls;      // 返回 cls
    }
    Py_INCREF(Py_NotImplemented);  // 增加 Py_NotImplemented 的引用计数
    return (PyArray_DTypeMeta *)Py_NotImplemented;  // 返回 Py_NotImplemented
}

/*
 * 返回一个对 `scalar` 的字符串表示的新引用。
 * 如果 scalar 不是字符串且 coerce 非零，则调用 __str__ 将其转换为字符串。
 * 如果 coerce 为零，则对非字符串或非 NA 输入引发错误。
 */
static PyObject *
as_pystring(PyObject *scalar, int coerce)
{
    PyTypeObject *scalar_type = Py_TYPE(scalar);  // 获取 scalar 的类型对象

    // 如果 scalar 的类型是 PyUnicode_Type
    if (scalar_type == &PyUnicode_Type) {
        Py_INCREF(scalar);  // 增加 scalar 的引用计数
        return scalar;      // 返回 scalar
    }

    // 如果 coerce 为零
    if (coerce == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "StringDType 只允许在禁用字符串强制转换时使用字符串数据。");
        return NULL;  // 返回 NULL，表示出错
    }
    else {
        // 尝试将 scalar 转换为字符串
        scalar = PyObject_Str(scalar);
        if (scalar == NULL) {
            // 如果 PyObject_Str 调用失败，返回 NULL
            return NULL;
        }
    }
    return scalar;  // 返回 scalar
}

/*
 * 从 Python 对象 `obj` 中发现描述符，并返回 PyArray_Descr 对象。
 */
static PyArray_Descr *
string_discover_descriptor_from_pyobject(PyTypeObject *NPY_UNUSED(cls),
                                         PyObject *obj)
{
    PyObject *val = as_pystring(obj, 1);  // 获取 obj 的字符串表示
    if (val == NULL) {
        return NULL;  // 如果获取失败，返回 NULL
    }

    Py_DECREF(val);  // 释放 val 的引用计数

    // 创建一个新的字符串数据类型描述符实例并返回
    PyArray_Descr *ret = (PyArray_Descr *)new_stringdtype_instance(NULL, 1);

    return ret;  // 返回描述符实例
}

/*
 * 将 Python 对象 `obj` 插入到数据类型为 `descr` 的数组中，在 dataptr 给定的位置。
 */
int
stringdtype_setitem(PyArray_StringDTypeObject *descr, PyObject *obj, char **dataptr)
{
    npy_packed_static_string *sdata = (npy_packed_static_string *)dataptr;

    // 借用引用
    PyObject *na_object = descr->na_object;

    // 在获取分配器后需要比较结果，但在获取分配器时不能使用需要 GIL 的函数，因此在获取分配器前进行比较。

    // 执行 na_eq_cmp 比较
    int na_cmp = na_eq_cmp(obj, na_object);
    if (na_cmp == -1) {
        return -1;  // 如果比较失败，返回 -1
    }

    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);

    // 如果 na_object 不为 NULL
    if (na_object != NULL) {
        if (na_cmp) {
            // 如果比较结果为真，尝试将空值打包到 sdata 中
            if (NpyString_pack_null(allocator, sdata) < 0) {
                PyErr_SetString(PyExc_MemoryError,
                                "Failed to pack null string during StringDType "
                                "setitem");
                goto fail;  // 打包失败，跳转到 fail 标签处处理错误
            }
            goto success;  // 成功处理
        }
    }

    // 将 obj 转换为字符串，并根据 descr->coerce 的值进行处理
    PyObject *val_obj = as_pystring(obj, descr->coerce);
    # 如果 val_obj 为 NULL，则跳转到失败处理的标签
    if (val_obj == NULL) {
        goto fail;
    }

    # 初始化长度变量为 0
    Py_ssize_t length = 0;
    # 将 val_obj 转换为 UTF-8 编码的字符串，并获取其长度
    const char *val = PyUnicode_AsUTF8AndSize(val_obj, &length);
    # 如果转换失败（val 为 NULL），释放 val_obj 并跳转到失败处理的标签
    if (val == NULL) {
        Py_DECREF(val_obj);
        goto fail;
    }

    # 使用 NpyString_pack 函数将字符串数据 val 打包到 sdata 中，
    # 如果返回值小于 0，表示打包过程出错，设置内存错误异常信息，
    # 释放 val_obj 并跳转到失败处理的标签
    if (NpyString_pack(allocator, sdata, val, length) < 0) {
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to pack string during StringDType "
                        "setitem");
        Py_DECREF(val_obj);
        goto fail;
    }
    # 打包成功后，释放 val_obj
    Py_DECREF(val_obj);
// 释放分配给 NpyString 的内存空间
NpyString_release_allocator(allocator);

// 返回成功标志
return 0;

fail:
    // 释放分配给 NpyString 的内存空间
    NpyString_release_allocator(allocator);

    // 返回失败标志
    return -1;
}

static PyObject *
stringdtype_getitem(PyArray_StringDTypeObject *descr, char **dataptr)
{
    // 初始化变量
    PyObject *val_obj = NULL;
    npy_packed_static_string *psdata = (npy_packed_static_string *)dataptr;
    npy_static_string sdata = {0, NULL};
    int has_null = descr->na_object != NULL;
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 加载字符串数据
    int is_null = NpyString_load(allocator, psdata, &sdata);

    // 处理加载过程中的错误
    if (is_null < 0) {
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to load string in StringDType getitem");
        // 转到失败处理标签
        goto fail;
    }
    else if (is_null == 1) {
        // 处理字符串为空的情况
        if (has_null) {
            // 返回NA对象
            PyObject *na_object = descr->na_object;
            Py_INCREF(na_object);
            val_obj = na_object;
        }
        else {
            // 返回空字符串对象
            val_obj = PyUnicode_FromStringAndSize("", 0);
        }
    }
    else {
        // 处理字符串不为空的情况
#ifndef PYPY_VERSION
        val_obj = PyUnicode_FromStringAndSize(sdata.buf, sdata.size);
#else
        // PyPy 版本兼容性处理
        val_obj = PyUnicode_FromStringAndSize(
                sdata.buf == NULL ? "" : sdata.buf, sdata.size);
#endif
        // 检查对象是否成功创建
        if (val_obj == NULL) {
            // 转到失败处理标签
            goto fail;
        }
    }

    // 释放分配给 NpyString 的内存空间
    NpyString_release_allocator(allocator);

    // 返回创建的对象
    return val_obj;

fail:
    // 处理失败情况下的资源释放
    NpyString_release_allocator(allocator);

    // 返回空对象
    return NULL;
}

// PyArray_NonzeroFunc
// Unicode 字符串长度非零时返回真值。
npy_bool
nonzero(void *data, void *arr)
{
    // 获取描述器
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)PyArray_DESCR(arr);
    int has_null = descr->na_object != NULL;
    int has_nan_na = descr->has_nan_na;
    int has_string_na = descr->has_string_na;

    // 检查空值情况
    if (has_null && NpyString_isnull((npy_packed_static_string *)data)) {
        // 检查字符串 NA 值的情况
        if (!has_string_na) {
            // 检查 NaN NA 值的情况
            if (has_nan_na) {
                // numpy 将 NaN 视为真值，与 Python 保持一致
                return 1;
            }
            else {
                return 0;
            }
        }
    }

    // 返回字符串长度是否非零
    return NpyString_size((npy_packed_static_string *)data) != 0;
}

// PyArray_CompareFunc 的实现
// 按照字符码点比较 Unicode 字符串
int
compare(void *a, void *b, void *arr)
{
    // 获取描述器
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)PyArray_DESCR(arr);
    // 获取分配器并锁定互斥量
    NpyString_acquire_allocator(descr);
    // 执行比较操作
    int ret = _compare(a, b, descr, descr);
    // 释放分配给 NpyString 的内存空间
    NpyString_release_allocator(descr->allocator);
    // 返回比较结果
    return ret;
}

// _compare 函数的实现
// 比较两个 PyArray_StringDTypeObject 对象的字符串
int
_compare(void *a, void *b, PyArray_StringDTypeObject *descr_a,
         PyArray_StringDTypeObject *descr_b)
{
    // 获取分配器
    npy_string_allocator *allocator_a = descr_a->allocator;
    npy_string_allocator *allocator_b = descr_b->allocator;
    // 获取描述符 descr_b 的分配器 allocator，用于字符串分配器的操作

    // descr_a and descr_b are either the same object or objects
    // that are equal, so we can safely refer only to descr_a.
    // This is enforced in the resolve_descriptors for comparisons
    // descr_a 和 descr_b 要么是同一个对象，要么是相等的对象，因此我们可以安全地只引用 descr_a。
    // 这在解析描述符用于比较时是强制执行的。

    // Note that even though the default_string isn't checked in comparisons,
    // it will still be the same for both descrs because the value of
    // default_string is always the empty string unless na_object is a string.
    // 注意，即使在比较中没有检查 default_string，它在两个描述符中仍将是相同的，
    // 因为除非 na_object 是字符串，否则 default_string 的值始终为空字符串。

    int has_null = descr_a->na_object != NULL;
    // 检查 descr_a 是否具有空值对象 na_object

    int has_string_na = descr_a->has_string_na;
    // 检查 descr_a 是否有字符串的 NA 值

    int has_nan_na = descr_a->has_nan_na;
    // 检查 descr_a 是否有 NaN 的 NA 值

    npy_static_string *default_string = &descr_a->default_string;
    // 获取描述符 descr_a 的默认静态字符串指针 default_string

    const npy_packed_static_string *ps_a = (npy_packed_static_string *)a;
    // 将 a 转换为 npy_packed_static_string 指针类型 ps_a

    npy_static_string s_a = {0, NULL};
    // 初始化静态字符串 s_a，长度为 0，内容为空

    int a_is_null = NpyString_load(allocator_a, ps_a, &s_a);
    // 使用 allocator_a 加载 ps_a 指向的字符串数据到 s_a 中，返回是否为 null 的标志

    const npy_packed_static_string *ps_b = (npy_packed_static_string *)b;
    // 将 b 转换为 npy_packed_static_string 指针类型 ps_b

    npy_static_string s_b = {0, NULL};
    // 初始化静态字符串 s_b，长度为 0，内容为空

    int b_is_null = NpyString_load(allocator_b, ps_b, &s_b);
    // 使用 allocator_b 加载 ps_b 指向的字符串数据到 s_b 中，返回是否为 null 的标志

    if (NPY_UNLIKELY(a_is_null == -1 || b_is_null == -1)) {
        // 如果加载字符串失败（返回 -1）
        char *msg = "Failed to load string in string comparison";
        // 错误消息字符串
        npy_gil_error(PyExc_MemoryError, msg);
        // 抛出内存错误异常
        return 0;
        // 返回 0 表示比较失败
    }
    else if (NPY_UNLIKELY(a_is_null || b_is_null)) {
        // 如果任一字符串为 null
        if (has_null && !has_string_na) {
            // 如果描述符允许 null 值且没有字符串 NA 值
            if (has_nan_na) {
                // 如果有 NaN 的 NA 值
                if (a_is_null) {
                    return 1;
                    // a 是 null，返回 1
                }
                else if (b_is_null) {
                    return -1;
                    // b 是 null，返回 -1
                }
            }
            else {
                // 没有 NaN 的 NA 值，报错
                npy_gil_error(
                        PyExc_ValueError,
                        "Cannot compare null that is not a nan-like value");
                // 抛出值错误异常
                return 0;
                // 返回 0 表示比较失败
            }
        }
        else {
            // 如果描述符不允许 null 值或有字符串 NA 值
            if (a_is_null) {
                s_a = *default_string;
                // 将 s_a 设置为默认字符串
            }
            if (b_is_null) {
                s_b = *default_string;
                // 将 s_b 设置为默认字符串
            }
        }
    }
    // 返回字符串 s_a 和 s_b 的比较结果
    return NpyString_cmp(&s_a, &s_b);
}

// PyArray_ArgFunc
// 返回数组中具有最高Unicode代码点的元素的索引。
int
argmax(char *data, npy_intp n, npy_intp *max_ind, void *arr)
{
    // 获取数组描述符
    PyArray_Descr *descr = PyArray_DESCR(arr);
    // 获取元素大小
    npy_intp elsize = descr->elsize;
    // 初始化最大索引为0
    *max_ind = 0;
    // 遍历数组
    for (int i = 1; i < n; i++) {
        // 比较当前元素与当前最大元素的Unicode代码点
        if (compare(data + i * elsize, data + (*max_ind) * elsize, arr) > 0) {
            // 更新最大元素的索引
            *max_ind = i;
        }
    }
    return 0;
}

// PyArray_ArgFunc
// 返回数组中具有最低Unicode代码点的元素的索引。
int
argmin(char *data, npy_intp n, npy_intp *min_ind, void *arr)
{
    // 获取数组描述符
    PyArray_Descr *descr = PyArray_DESCR(arr);
    // 获取元素大小
    npy_intp elsize = descr->elsize;
    // 初始化最小索引为0
    *min_ind = 0;
    // 遍历数组
    for (int i = 1; i < n; i++) {
        // 比较当前元素与当前最小元素的Unicode代码点
        if (compare(data + i * elsize, data + (*min_ind) * elsize, arr) < 0) {
            // 更新最小元素的索引
            *min_ind = i;
        }
    }
    return 0;
}

static PyArray_StringDTypeObject *
stringdtype_ensure_canonical(PyArray_StringDTypeObject *self)
{
    // 增加引用计数，确保字符串数据类型对象规范化
    Py_INCREF(self);
    return self;
}

static int
stringdtype_clear_loop(void *NPY_UNUSED(traverse_context),
                       const PyArray_Descr *descr, char *data, npy_intp size,
                       npy_intp stride, NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取字符串数据类型对象
    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)descr;
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);
    // 遍历数据，释放每个字符串
    while (size--) {
        npy_packed_static_string *sdata = (npy_packed_static_string *)data;
        // 如果数据不为空，释放字符串内存
        if (data != NULL && NpyString_free(sdata, allocator) < 0) {
            // 在清理循环中发生内存错误时，抛出异常
            npy_gil_error(PyExc_MemoryError,
                          "String deallocation failed in clear loop");
            goto fail;
        }
        // 移动到下一个字符串
        data += stride;
    }
    // 释放字符串分配器
    NpyString_release_allocator(allocator);
    return 0;

fail:
    // 失败时释放字符串分配器
    NpyString_release_allocator(allocator);
    return -1;
}

static int
stringdtype_get_clear_loop(void *NPY_UNUSED(traverse_context),
                           PyArray_Descr *NPY_UNUSED(descr),
                           int NPY_UNUSED(aligned),
                           npy_intp NPY_UNUSED(fixed_stride),
                           PyArrayMethod_TraverseLoop **out_loop,
                           NpyAuxData **NPY_UNUSED(out_auxdata),
                           NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 设置标志以避免浮点错误
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    // 设置清理循环函数指针
    *out_loop = &stringdtype_clear_loop;
    return 0;
}

static int
stringdtype_is_known_scalar_type(PyArray_DTypeMeta *cls,
                                 PyTypeObject *pytype)
{
    // 检查Python内建类型是否为已知的标量类型
    if (python_builtins_are_known_scalar_types(cls, pytype)) {
        return 1;
    }
    // 接受所有内建的numpy数据类型

    return 1;
}
    // 如果 pytype 是以下任意一种数组类型，则返回 1
    else if (pytype == &PyBoolArrType_Type ||
             pytype == &PyByteArrType_Type ||
             pytype == &PyShortArrType_Type ||
             pytype == &PyIntArrType_Type ||
             pytype == &PyLongArrType_Type ||
             pytype == &PyLongLongArrType_Type ||
             pytype == &PyUByteArrType_Type ||
             pytype == &PyUShortArrType_Type ||
             pytype == &PyUIntArrType_Type ||
             pytype == &PyULongArrType_Type ||
             pytype == &PyULongLongArrType_Type ||
             pytype == &PyHalfArrType_Type ||
             pytype == &PyFloatArrType_Type ||
             pytype == &PyDoubleArrType_Type ||
             pytype == &PyLongDoubleArrType_Type ||
             pytype == &PyCFloatArrType_Type ||
             pytype == &PyCDoubleArrType_Type ||
             pytype == &PyCLongDoubleArrType_Type ||
             pytype == &PyIntpArrType_Type ||
             pytype == &PyUIntpArrType_Type ||
             pytype == &PyDatetimeArrType_Type ||
             pytype == &PyTimedeltaArrType_Type)
    {
        // 返回 1 表示 pytype 是数组类型之一
        return 1;
    }
    // 否则返回 0
    return 0;
}

// 结束函数 stringdtype_finalize_descr

PyArray_Descr *
stringdtype_finalize_descr(PyArray_Descr *dtype)
{
    // 将传入的 dtype 转换为 PyArray_StringDTypeObject 类型
    PyArray_StringDTypeObject *sdtype = (PyArray_StringDTypeObject *)dtype;
    // 如果 sdtype 指向的数组未被所有者拥有
    if (sdtype->array_owned == 0) {
        // 标记数组已被所有者拥有
        sdtype->array_owned = 1;
        // 增加 dtype 的引用计数
        Py_INCREF(dtype);
        // 返回原始的 dtype
        return dtype;
    }
    // 否则创建一个新的 PyArray_StringDTypeObject 实例，使用 sdtype 的 na_object 和 coerce 属性
    PyArray_StringDTypeObject *ret = (PyArray_StringDTypeObject *)new_stringdtype_instance(
            sdtype->na_object, sdtype->coerce);
    // 标记新实例的数组已被所有者拥有
    ret->array_owned = 1;
    // 返回新实例的描述符
    return (PyArray_Descr *)ret;
}

// 静态数组 PyArray_StringDType_Slots 的定义和初始化
static PyType_Slot PyArray_StringDType_Slots[] = {
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject,
         &string_discover_descriptor_from_pyobject},
        {NPY_DT_setitem, &stringdtype_setitem},
        {NPY_DT_getitem, &stringdtype_getitem},
        {NPY_DT_ensure_canonical, &stringdtype_ensure_canonical},
        {NPY_DT_PyArray_ArrFuncs_nonzero, &nonzero},
        {NPY_DT_PyArray_ArrFuncs_compare, &compare},
        {NPY_DT_PyArray_ArrFuncs_argmax, &argmax},
        {NPY_DT_PyArray_ArrFuncs_argmin, &argmin},
        {NPY_DT_get_clear_loop, &stringdtype_get_clear_loop},
        {NPY_DT_finalize_descr, &stringdtype_finalize_descr},
        {_NPY_DT_is_known_scalar_type, &stringdtype_is_known_scalar_type},
        {0, NULL}};

// 创建新的 StringDType 实例
static PyObject *
stringdtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    // 定义关键字参数数组
    static char *kwargs_strs[] = {"coerce", "na_object", NULL};

    PyObject *na_object = NULL;
    int coerce = 1;

    // 使用 PyArg_ParseTupleAndKeywords 解析参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|$pO&:StringDType",
                                     kwargs_strs, &coerce,
                                     _not_NoValue, &na_object)) {
        return NULL;
    }

    // 返回创建的新 StringDType 实例
    return new_stringdtype_instance(na_object, coerce);
}

// 释放 StringDType 对象的内存
static void
stringdtype_dealloc(PyArray_StringDTypeObject *self)
{
    // 释放 na_object 的引用
    Py_XDECREF(self->na_object);
    // 如果 allocator 不为空，则释放其分配的资源
    if (self->allocator != NULL) {
        NpyString_free_allocator(self->allocator);
    }
    // 释放 na_name 和 default_string 的内存
    PyMem_RawFree((char *)self->na_name.buf);
    PyMem_RawFree((char *)self->default_string.buf);
    // 调用父类的 dealloc 方法释放对象内存
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

// 返回 StringDType 对象的字符串表示形式
static PyObject *
stringdtype_repr(PyArray_StringDTypeObject *self)
{
    PyObject *ret = NULL;
    // 借用 na_object 的引用
    PyObject *na_object = self->na_object;
    int coerce = self->coerce;

    // 根据 na_object 和 coerce 属性生成不同的字符串表示形式
    if (na_object != NULL && coerce == 0) {
        ret = PyUnicode_FromFormat("StringDType(na_object=%R, coerce=False)",
                                   na_object);
    }
    else if (na_object != NULL) {
        ret = PyUnicode_FromFormat("StringDType(na_object=%R)", na_object);
    }
    else if (coerce == 0) {
        ret = PyUnicode_FromFormat("StringDType(coerce=False)", coerce);
    }
    else {
        ret = PyUnicode_FromString("StringDType()");
    }

    return ret;
}

// 实现 __reduce__ 魔法方法以重建 StringDType 对象
// 导入 "numpy._core._internal" 模块中的 "_convert_to_stringdtype_kwargs" 函数，
// 并将其存储在全局变量 npy_thread_unsafe_state._convert_to_stringdtype_kwargs 中。
// 这个操作并不是性能关键，仅仅是为了方便使用 Python 中的 pickle 模块进行对象序列化。
static PyObject *
stringdtype__reduce__(PyArray_StringDTypeObject *self, PyObject *NPY_UNUSED(args))
{
    npy_cache_import("numpy._core._internal", "_convert_to_stringdtype_kwargs",
                     &npy_thread_unsafe_state._convert_to_stringdtype_kwargs);

    // 如果未能成功导入 _convert_to_stringdtype_kwargs 函数，返回 NULL 表示错误。
    if (npy_thread_unsafe_state._convert_to_stringdtype_kwargs == NULL) {
        return NULL;
    }

    // 如果 self->na_object 不为 NULL，则返回一个包含三个元素的元组：
    // (npy_thread_unsafe_state._convert_to_stringdtype_kwargs, self->coerce, self->na_object)。
    if (self->na_object != NULL) {
        return Py_BuildValue(
                "O(iO)", npy_thread_unsafe_state._convert_to_stringdtype_kwargs,
                self->coerce, self->na_object);
    }

    // 如果 self->na_object 为 NULL，则返回一个包含两个元素的元组：
    // (npy_thread_unsafe_state._convert_to_stringdtype_kwargs, self->coerce)。
    return Py_BuildValue(
            "O(i)", npy_thread_unsafe_state._convert_to_stringdtype_kwargs,
            self->coerce);
}

// 定义 PyArray_StringDType 对象的方法列表
static PyMethodDef PyArray_StringDType_methods[] = {
        {
                "__reduce__",                               // 方法名
                (PyCFunction)stringdtype__reduce__,         // 方法实现函数
                METH_NOARGS,                                // 方法接受的参数类型标志
                "Reduction method for a StringDType object", // 方法的文档字符串
        },
        {NULL, NULL, 0, NULL},                              // 方法列表结束标志
};

// 定义 PyArray_StringDType 对象的成员变量列表
static PyMemberDef PyArray_StringDType_members[] = {
        {"na_object",                                   // 成员变量名
         T_OBJECT_EX,                                   // 变量的数据类型
         offsetof(PyArray_StringDTypeObject, na_object), // 变量在结构体中的偏移量
         READONLY,                                      // 变量的标志，表示只读
         "The missing value object associated with the dtype instance"}, // 变量的文档字符串
        {"coerce",                                      // 成员变量名
         T_BOOL,                                        // 变量的数据类型
         offsetof(PyArray_StringDTypeObject, coerce),   // 变量在结构体中的偏移量
         READONLY,                                      // 变量的标志，表示只读
         "Controls whether non-string values should be coerced to string"}, // 变量的文档字符串
        {NULL, 0, 0, 0, NULL},                           // 成员变量列表结束标志
};

// 定义 PyArray_StringDType 对象的富比较方法
static PyObject *
PyArray_StringDType_richcompare(PyObject *self, PyObject *other, int op)
{
    // 如果 op 不是 Py_EQ 或 Py_NE，或者 other 不是 self 的同一类型，返回 NotImplemented。
    if (((op != Py_EQ) && (op != Py_NE)) ||
        (Py_TYPE(other) != Py_TYPE(self))) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    // 将 self 和 other 转换为 PyArray_StringDTypeObject 类型，因为我们已知它们是该类型的实例。
    PyArray_StringDTypeObject *sself = (PyArray_StringDTypeObject *)self;
    PyArray_StringDTypeObject *sother = (PyArray_StringDTypeObject *)other;

    // 调用 _eq_comparison 函数比较 self 和 other 的 coerce 和 na_object 成员变量。
    int eq = _eq_comparison(sself->coerce, sother->coerce, sself->na_object,
                            sother->na_object);

    // 如果 _eq_comparison 返回 -1，表示比较出错，返回 NULL。
    if (eq == -1) {
        return NULL;
    }

    // 根据比较结果和 op 的值返回 Py_True 或 Py_False。
    if ((op == Py_EQ && eq) || (op == Py_NE && !eq)) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    Py_INCREF(Py_False);
    return Py_False;
}

// 定义 PyArray_StringDType 对象的哈希计算方法
static Py_hash_t
PyArray_StringDType_hash(PyObject *self)
{
    // 将 self 转换为 PyArray_StringDTypeObject 类型。
    PyArray_StringDTypeObject *sself = (PyArray_StringDTypeObject *)self;
    PyObject *hash_tup = NULL;

    // 如果 sself->na_object 不为 NULL，则构建一个包含两个元素的元组 (sself->coerce, sself->na_object)。
    // 否则，构建一个包含一个元素的元组 (sself->coerce)。
    if (sself->na_object != NULL) {
        hash_tup = Py_BuildValue("(iO)", sself->coerce, sself->na_object);
    }
    else {
        hash_tup = Py_BuildValue("(i)", sself->coerce);
    }

    // 计算元组 hash_tup 的哈希值，并返回结果。
    Py_hash_t ret = PyObject_Hash(hash_tup);
    Py_DECREF(hash_tup);
    return ret;
}
"""
/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
*/

PyArray_DTypeMeta PyArray_StringDType = {
        {{
                // 设置类型名称为 "numpy.dtypes.StringDType"，基本大小为 PyArray_StringDTypeObject 的大小
                PyVarObject_HEAD_INIT(NULL, 0).tp_name =
                        "numpy.dtypes.StringDType",
                // 分配内存空间大小为 PyArray_StringDTypeObject 的大小
                .tp_basicsize = sizeof(PyArray_StringDTypeObject),
                // 设置构造函数为 stringdtype_new
                .tp_new = stringdtype_new,
                // 设置析构函数为 stringdtype_dealloc
                .tp_dealloc = (destructor)stringdtype_dealloc,
                // 设置对象的字符串表示形式为 stringdtype_repr
                .tp_repr = (reprfunc)stringdtype_repr,
                // 设置对象的字符串形式为 stringdtype_repr
                .tp_str = (reprfunc)stringdtype_repr,
                // 设置对象的方法为 PyArray_StringDType_methods
                .tp_methods = PyArray_StringDType_methods,
                // 设置对象的成员为 PyArray_StringDType_members
                .tp_members = PyArray_StringDType_members,
                // 设置对象的比较函数为 PyArray_StringDType_richcompare
                .tp_richcompare = PyArray_StringDType_richcompare,
                // 设置对象的哈希函数为 PyArray_StringDType_hash
                .tp_hash = PyArray_StringDType_hash,
        }},
        /* rest, filled in during DTypeMeta initialization */
};

NPY_NO_EXPORT int
init_string_dtype(void)
{
    // 获取字符串类型的强制转换列表
    PyArrayMethod_Spec **PyArray_StringDType_casts = get_casts();

    // 字符串类型的元数据规范
    PyArrayDTypeMeta_Spec PyArray_StringDType_DTypeSpec = {
            // 标志为 NPY_DT_PARAMETRIC
            .flags = NPY_DT_PARAMETRIC,
            // 类型对象为 PyUnicode_Type
            .typeobj = &PyUnicode_Type,
            // 槽位为 PyArray_StringDType_Slots
            .slots = PyArray_StringDType_Slots,
            // 强制转换为 PyArray_StringDType_casts
            .casts = PyArray_StringDType_casts,
    };

    /* Loaded dynamically, so needs to be set here: */
    // 将 PyArray_StringDType 的类型设置为 PyArrayDTypeMeta_Type
    ((PyObject *)&PyArray_StringDType)->ob_type = &PyArrayDTypeMeta_Type;
    // 将 PyArray_StringDType 的基类设置为 PyArrayDescr_Type
    ((PyTypeObject *)&PyArray_StringDType)->tp_base = &PyArrayDescr_Type;
    // 如果 PyType_Ready((PyTypeObject *)&PyArray_StringDType) 小于 0，则返回 -1
    if (PyType_Ready((PyTypeObject *)&PyArray_StringDType) < 0) {
        return -1;
    }

    // 如果 dtypemeta_initialize_struct_from_spec 初始化失败，则返回 -1
    if (dtypemeta_initialize_struct_from_spec(
                &PyArray_StringDType, &PyArray_StringDType_DTypeSpec, 1) < 0) {
        return -1;
    }

    // 获取默认的描述符
    PyArray_Descr *singleton =
            NPY_DT_CALL_default_descr(&PyArray_StringDType);

    // 如果 singleton 为空，则返回 -1
    if (singleton == NULL) {
        return -1;
    }

    // 设置 PyArray_StringDType 的单例为 singleton
    PyArray_StringDType.singleton = singleton;
    // 设置 PyArray_StringDType 的类型编号为 NPY_VSTRING
    PyArray_StringDType.type_num = NPY_VSTRING;

    // 释放 PyArray_StringDType_casts 的内存
    for (int i = 0; PyArray_StringDType_casts[i] != NULL; i++) {
        PyMem_Free(PyArray_StringDType_casts[i]->dtypes);
        PyMem_Free(PyArray_StringDType_casts[i]);
    }

    PyMem_Free(PyArray_StringDType_casts);

    // 返回成功
    return 0;
}

int
free_and_copy(npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator,
              const npy_packed_static_string *in,
              npy_packed_static_string *out, const char *location)
{
    // 释放输出的字符串内存
    if (NpyString_free(out, out_allocator) < 0) {
        // 如果释放失败，则报错并返回 -1
        npy_gil_error(PyExc_MemoryError, "Failed to deallocate string in %s", location);
        return -1;
    }
    // 复制输入到输出字符串
    if (NpyString_dup(in, out, in_allocator, out_allocator) < 0) {
        // 如果复制失败，则报错并返回 -1
        npy_gil_error(PyExc_MemoryError, "Failed to allocate string in %s", location);
        return -1;
    }
    // 操作成功，返回 0
    return 0;
}
/*
 * 使用一个预定义的 npy_static_string 实例，分配到栈上并初始化为 {0, NULL}，
 * 将指向这个栈上分配的未打包字符串的指针传递给该函数，用来填充新分配字符串的内容。
 */
NPY_NO_EXPORT int
load_new_string(npy_packed_static_string *out, npy_static_string *out_ss,
                size_t num_bytes, npy_string_allocator *allocator,
                const char *err_context)
{
    // 将输出参数 out 强制转换为 npy_packed_static_string 指针
    npy_packed_static_string *out_pss = (npy_packed_static_string *)out;
    // 如果调用 NpyString_free 函数释放 out_pss 失败
    if (NpyString_free(out_pss, allocator) < 0) {
        // 报告内存错误，并指明错误上下文
        npy_gil_error(PyExc_MemoryError,
                      "Failed to deallocate string in %s", err_context);
        return -1;  // 返回错误码
    }
    // 调用 NpyString_newemptysize 函数尝试分配 num_bytes 大小的新字符串到 out_pss
    if (NpyString_newemptysize(num_bytes, out_pss, allocator) < 0) {
        // 报告内存错误，并指明错误上下文
        npy_gil_error(PyExc_MemoryError,
                      "Failed to allocate string in %s", err_context);
        return -1;  // 返回错误码
    }
    // 调用 NpyString_load 函数加载字符串到 out_ss，并使用 allocator 进行分配
    if (NpyString_load(allocator, out_pss, out_ss) == -1) {
        // 报告内存错误，并指明错误上下文
        npy_gil_error(PyExc_MemoryError,
                      "Failed to load string in %s", err_context);
        return -1;  // 返回错误码
    }
    // 操作成功完成，返回成功码
    return 0;
}
```