# `.\numpy\numpy\_core\src\multiarray\flagsobject.c`

```py
/*
 * Array Flags Object
 * 定义了一些与数组标志相关的宏和函数
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T 未定义的宏
 */
#define PY_SSIZE_T_CLEAN

/*
 * 引入必要的头文件和库文件
 */
#include <Python.h>             // Python 核心头文件
#include <structmember.h>       // 结构成员访问支持

#include "numpy/arrayobject.h"  // NumPy 数组对象接口
#include "arrayobject.h"        // 数组对象接口
#include "numpy/arrayscalars.h" // NumPy 数组标量接口

#include "npy_config.h"         // NumPy 配置文件

#include "array_assign.h"       // 数组赋值相关功能
#include "common.h"             // 通用功能
#include "flagsobject.h"        // 数组标志对象

/*
 * 静态函数声明：更新连续性标志
 */
static void
_UpdateContiguousFlags(PyArrayObject *ap);

/*
 * 获取新的 ArrayFlagsObject 对象
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFlagsObject(PyObject *obj)
{
    PyObject *flagobj;
    int flags;

    // 如果传入的对象是空指针，则设置默认的标志位
    if (obj == NULL) {
        flags = NPY_ARRAY_C_CONTIGUOUS |
                NPY_ARRAY_OWNDATA |
                NPY_ARRAY_F_CONTIGUOUS |
                NPY_ARRAY_ALIGNED;
    }
    else {
        // 检查传入的对象是否为 NumPy 数组
        if (!PyArray_Check(obj)) {
            PyErr_SetString(PyExc_ValueError,
                    "Need a NumPy array to create a flags object");
            return NULL;
        }

        // 获取传入数组的标志位
        flags = PyArray_FLAGS((PyArrayObject *)obj);
    }

    // 分配并初始化新的 ArrayFlagsObject 对象
    flagobj = PyArrayFlags_Type.tp_alloc(&PyArrayFlags_Type, 0);
    if (flagobj == NULL) {
        return NULL;
    }
    // 增加传入对象的引用计数，并将其赋值给 ArrayFlagsObject 对象的 arr 成员
    Py_XINCREF(obj);
    ((PyArrayFlagsObject *)flagobj)->arr = obj;
    // 将计算得到的标志位赋值给 ArrayFlagsObject 对象的 flags 成员
    ((PyArrayFlagsObject *)flagobj)->flags = flags;

    return flagobj;
}

/*NUMPY_API
 * 同时更新多个标志位
 */
NPY_NO_EXPORT void
PyArray_UpdateFlags(PyArrayObject *ret, int flagmask)
{
    // 总是同时更新连续性标志位，因为从一个标志位推断另一个并不容易
    if (flagmask & (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS)) {
        _UpdateContiguousFlags(ret);
    }

    // 更新对齐标志位
    if (flagmask & NPY_ARRAY_ALIGNED) {
        if (IsAligned(ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
    }

    /*
     * 默认情况下，WRITEABLE 不在 UPDATE_ALL 中，所以需要额外检查
     * 更新可写标志位
     */
    if (flagmask & NPY_ARRAY_WRITEABLE) {
        if (_IsWriteable(ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
    }
    return;
}
/*
 * 更新数组对象的连续性标志位。
 * 根据数组维度和步长，确定数组是否以 C 或 F 连续存储。
 */
static void
_UpdateContiguousFlags(PyArrayObject *ap)
{
    npy_intp sd;          // 步长的值
    npy_intp dim;         // 数组的当前维度长度
    int i;                // 迭代器

    npy_bool is_c_contig = 1;  // 是否是 C 连续的标志位，默认为真

    sd = PyArray_ITEMSIZE(ap);  // 获取数组元素的大小
    for (i = PyArray_NDIM(ap) - 1; i >= 0; --i) {
        dim = PyArray_DIMS(ap)[i];  // 获取当前维度的长度
        /* contiguous by definition */
        if (dim == 0) {  // 如果当前维度长度为0，数组被定义为连续的
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);  // 启用 C 连续标志
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);  // 启用 F 连续标志
            return;  // 返回
        }
        if (dim != 1) {  // 如果当前维度长度不为1
            if (PyArray_STRIDES(ap)[i] != sd) {  // 检查当前维度的步长是否与期望的步长相同
                is_c_contig = 0;  // 如果步长不符合要求，则不是 C 连续
            }
            sd *= dim;  // 更新步长值
        }
    }
    if (is_c_contig) {
        PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);  // 启用 C 连续标志
    }
    else {
        PyArray_CLEARFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);  // 清除 C 连续标志
    }

    /* 检查是否是 Fortran 连续 */
    sd = PyArray_ITEMSIZE(ap);  // 重新获取数组元素的大小
    for (i = 0; i < PyArray_NDIM(ap); ++i) {
        dim = PyArray_DIMS(ap)[i];  // 获取当前维度的长度
        if (dim != 1) {  // 如果当前维度长度不为1
            if (PyArray_STRIDES(ap)[i] != sd) {  // 检查当前维度的步长是否与期望的步长相同
                PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);  // 清除 F 连续标志
                return;  // 返回
            }
            sd *= dim;  // 更新步长值
        }
    }
    PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);  // 启用 F 连续标志
    return;  // 返回
}

/*
 * 释放数组标志对象的资源
 */
static void
arrayflags_dealloc(PyArrayFlagsObject *self)
{
    Py_XDECREF(self->arr);  // 释放数组对象的引用
    Py_TYPE(self)->tp_free((PyObject *)self);  // 释放对象内存
}

/*
 * 定义获取标志位的宏函数，用于不同标志位的获取操作
 */
#define _define_get(UPPER, lower) \
    static PyObject * \
    arrayflags_ ## lower ## _get( \
            PyArrayFlagsObject *self, void *NPY_UNUSED(ignored)) \
    { \
        return PyBool_FromLong((self->flags & (UPPER)) == (UPPER)); \
    }

/*
 * 定义获取标志位并生成警告信息的宏函数
 */
static char *msg = "future versions will not create a writeable "
    "array from broadcast_array. Set the writable flag explicitly to "
    "avoid this warning.";

#define _define_get_warn(UPPER, lower) \
    static PyObject * \
    arrayflags_ ## lower ## _get( \
            PyArrayFlagsObject *self, void *NPY_UNUSED(ignored)) \
    { \
        // 检查标志位中是否包含 NPY_ARRAY_WARN_ON_WRITE 标志
        if (self->flags & NPY_ARRAY_WARN_ON_WRITE) { \
            // 如果包含，则发出 FutureWarning 警告并检查是否出错
            if (PyErr_Warn(PyExc_FutureWarning, msg) < 0) {\
                // 如果发生错误，返回 NULL
                return NULL; \
            } \
        }\
        // 返回一个 PyBool 对象，表示 self->flags 中的 UPPER 标志是否全部设置
        return PyBool_FromLong((self->flags & (UPPER)) == (UPPER)); \
    }
/* 定义宏以获取相应标志 */
_define_get(NPY_ARRAY_C_CONTIGUOUS, contiguous)
_define_get(NPY_ARRAY_F_CONTIGUOUS, fortran)
_define_get(NPY_ARRAY_WRITEBACKIFCOPY, writebackifcopy)
_define_get(NPY_ARRAY_OWNDATA, owndata)
_define_get(NPY_ARRAY_ALIGNED, aligned)
_define_get(NPY_ARRAY_WRITEABLE, writeable_no_warn)
_define_get_warn(NPY_ARRAY_WRITEABLE, writeable)
_define_get_warn(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE, behaved)
_define_get_warn(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE|
            NPY_ARRAY_C_CONTIGUOUS, carray)

/* 定义静态函数：获取数组标志中的 C 连续和 F 连续 */
static PyObject *
arrayflags_forc_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *item;

    // 如果数组是 F 连续或者 C 连续，返回 Py_True，否则返回 Py_False
    if (((self->flags & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS) ||
        ((self->flags & NPY_ARRAY_C_CONTIGUOUS) == NPY_ARRAY_C_CONTIGUOUS)) {
        item = Py_True;
    }
    else {
        item = Py_False;
    }
    Py_INCREF(item);
    return item;
}

/* 定义静态函数：获取数组标志中的 F 连续但非 C 连续 */
static PyObject *
arrayflags_fnc_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *item;

    // 如果数组是 F 连续且不是 C 连续，返回 Py_True，否则返回 Py_False
    if (((self->flags & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS) &&
        !((self->flags & NPY_ARRAY_C_CONTIGUOUS) == NPY_ARRAY_C_CONTIGUOUS)) {
        item = Py_True;
    }
    else {
        item = Py_False;
    }
    Py_INCREF(item);
    return item;
}

/* 定义静态函数：获取数组标志中的 A (对齐)、W (可写)、F (F 连续) 的组合 */
static PyObject *
arrayflags_farray_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *item;

    // 如果数组同时满足对齐、可写、F 连续的条件，且不是 C 连续，返回 Py_True，否则返回 Py_False
    if (((self->flags & (NPY_ARRAY_ALIGNED|
                         NPY_ARRAY_WRITEABLE|
                         NPY_ARRAY_F_CONTIGUOUS)) != 0) &&
        !((self->flags & NPY_ARRAY_C_CONTIGUOUS) != 0)) {
        item = Py_True;
    }
    else {
        item = Py_False;
    }
    Py_INCREF(item);
    return item;
}

/* 定义静态函数：获取数组标志的整数表示 */
static PyObject *
arrayflags_num_get(PyArrayFlagsObject *self, void *NPY_UNUSED(ignored))
{
    // 返回数组标志的整数表示
    return PyLong_FromLong(self->flags);
}

/* 定义静态函数：设置 writebackifcopy 标志 */
/* 假定 setflags 的顺序是 write、align、uic */
static int
arrayflags_writebackifcopy_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
{
    PyObject *res;

    // 如果传入的 obj 是 NULL，则不能删除 writebackifcopy 属性，返回错误
    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags writebackifcopy attribute");
        return -1;
    }
    // 如果 self->arr 是 NULL，则不能在数组标量上设置标志，返回错误
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set flags on array scalars.");
        return -1;
    }
    // 判断传入的 obj 是否为真值
    int istrue = PyObject_IsTrue(obj);
    if (istrue == -1) {
        return -1;
    }
    // 调用 self->arr 的 setflags 方法，传递三个参数：Py_None、Py_None 和 istrue 的真假值
    res = PyObject_CallMethod(self->arr, "setflags", "OOO", Py_None, Py_None,
                              (istrue ? Py_True : Py_False));
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

/* 定义静态函数：设置 aligned 标志 */
static int
arrayflags_aligned_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
{
    PyObject *res;

    // 如果传入的 obj 是 NULL，则不能删除 aligned 属性，返回错误
    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags aligned attribute");
        return -1;
    }
    // 检查数组指针是否为 NULL，如果是，则抛出异常并返回错误码
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set flags on array scalars.");
        return -1;
    }
    
    // 检查对象是否为真值，并返回对应的整数结果，如果出错则返回错误码
    int istrue = PyObject_IsTrue(obj);
    if (istrue == -1) {
        return -1;
    }
    
    // 调用数组对象的 setflags 方法，设置其标志位
    // 参数依次为 Py_None（空对象）、(istrue 为真时为 Py_True 否则为 Py_False)、Py_None
    res = PyObject_CallMethod(self->arr, "setflags", "OOO", Py_None,
                              (istrue ? Py_True : Py_False),
                              Py_None);
    
    // 如果调用出错（返回结果为 NULL），则返回错误码
    if (res == NULL) {
        return -1;
    }
    
    // 释放调用结果对象的引用计数，避免内存泄漏
    Py_DECREF(res);
    
    // 返回成功状态码
    return 0;
static int
arrayflags_writeable_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
{
    PyObject *res;

    // 如果传入的对象是空，则设置错误并返回
    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags writeable attribute");
        return -1;
    }
    
    // 如果数组对象为 NULL，则设置错误并返回
    if (self->arr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set flags on array scalars.");
        return -1;
    }

    // 检查传入的对象是否为真值
    int istrue = PyObject_IsTrue(obj);
    if (istrue == -1) {
        return -1;
    }
    
    // 调用数组对象的 setflags 方法来设置可写标志
    res = PyObject_CallMethod(self->arr, "setflags", "OOO",
                              (istrue ? Py_True : Py_False),
                              Py_None, Py_None);
    if (res == NULL) {
        return -1;
    }
    
    // 减少对结果的引用计数，避免内存泄漏
    Py_DECREF(res);
    
    // 返回成功
    return 0;
}

static int
arrayflags_warn_on_write_set(
        PyArrayFlagsObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
{
    /*
     * This code should go away in a future release, so do not mangle the
     * array_setflags function with an extra kwarg
     */
    int ret;
    
    // 如果传入的对象是空，则设置错误并返回
    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete flags _warn_on_write attribute");
        return -1;
    }
    
    // 检查传入的对象是否为真值
    ret = PyObject_IsTrue(obj);
    if (ret > 0) {
        // 如果数组对象不可写，则设置错误并返回
        if (!(PyArray_FLAGS((PyArrayObject*)self->arr) & NPY_ARRAY_WRITEABLE)) {
            PyErr_SetString(PyExc_ValueError,
                        "cannot set '_warn_on_write' flag when 'writable' is "
                        "False");
            return -1;
        }
        // 启用数组对象的 WARN_ON_WRITE 标志
        PyArray_ENABLEFLAGS((PyArrayObject*)self->arr, NPY_ARRAY_WARN_ON_WRITE);
    }
    else if (ret < 0) {
        return -1;
    }
    else {
        // 如果传入的对象不是真值，则设置错误并返回
        PyErr_SetString(PyExc_ValueError,
                        "cannot clear '_warn_on_write', set "
                        "writeable True to clear this private flag");
        return -1;
    }
    
    // 返回成功
    return 0;
}

static PyGetSetDef arrayflags_getsets[] = {
    {"contiguous",
        (getter)arrayflags_contiguous_get,
        NULL,
        NULL, NULL},
    {"c_contiguous",
        (getter)arrayflags_contiguous_get,
        NULL,
        NULL, NULL},
    {"f_contiguous",
        (getter)arrayflags_fortran_get,
        NULL,
        NULL, NULL},
    {"fortran",
        (getter)arrayflags_fortran_get,
        NULL,
        NULL, NULL},
    {"writebackifcopy",
        (getter)arrayflags_writebackifcopy_get,
        (setter)arrayflags_writebackifcopy_set,
        NULL, NULL},
    {"owndata",
        (getter)arrayflags_owndata_get,
        NULL,
        NULL, NULL},
    {"aligned",
        (getter)arrayflags_aligned_get,
        (setter)arrayflags_aligned_set,
        NULL, NULL},
    {"writeable",
        (getter)arrayflags_writeable_get,
        (setter)arrayflags_writeable_set,
        NULL, NULL},
    {"_writeable_no_warn",
        (getter)arrayflags_writeable_no_warn_get,
        (setter)NULL,
        NULL, NULL},
    {"_warn_on_write",
        (getter)NULL,
        (setter)arrayflags_warn_on_write_set,
        NULL, NULL},

设置一个名为 `_warn_on_write` 的属性，它具有以下特性：
- getter 为 NULL，表示没有定义 getter 函数。
- setter 函数为 `arrayflags_warn_on_write_set`，用于设置 `_warn_on_write` 属性的值。
- 没有额外的文档字符串或数据。


    {"fnc",
        (getter)arrayflags_fnc_get,
        NULL,
        NULL, NULL},

设置一个名为 `fnc` 的属性，具有以下特性：
- getter 函数为 `arrayflags_fnc_get`，用于获取 `fnc` 属性的值。
- 没有 setter 函数。
- 没有额外的文档字符串或数据。


    {"forc",
        (getter)arrayflags_forc_get,
        NULL,
        NULL, NULL},

设置一个名为 `forc` 的属性，具有以下特性：
- getter 函数为 `arrayflags_forc_get`，用于获取 `forc` 属性的值。
- 没有 setter 函数。
- 没有额外的文档字符串或数据。


    {"behaved",
        (getter)arrayflags_behaved_get,
        NULL,
        NULL, NULL},

设置一个名为 `behaved` 的属性，具有以下特性：
- getter 函数为 `arrayflags_behaved_get`，用于获取 `behaved` 属性的值。
- 没有 setter 函数。
- 没有额外的文档字符串或数据。


    {"carray",
        (getter)arrayflags_carray_get,
        NULL,
        NULL, NULL},

设置一个名为 `carray` 的属性，具有以下特性：
- getter 函数为 `arrayflags_carray_get`，用于获取 `carray` 属性的值。
- 没有 setter 函数。
- 没有额外的文档字符串或数据。


    {"farray",
        (getter)arrayflags_farray_get,
        NULL,
        NULL, NULL},

设置一个名为 `farray` 的属性，具有以下特性：
- getter 函数为 `arrayflags_farray_get`，用于获取 `farray` 属性的值。
- 没有 setter 函数。
- 没有额外的文档字符串或数据。


    {"num",
        (getter)arrayflags_num_get,
        NULL,
        NULL, NULL},

设置一个名为 `num` 的属性，具有以下特性：
- getter 函数为 `arrayflags_num_get`，用于获取 `num` 属性的值。
- 没有 setter 函数。
- 没有额外的文档字符串或数据。


    {NULL, NULL, NULL, NULL, NULL},

属性列表结束标记，用于指示没有更多的属性。
    };
    
    // 定义 arrayflags_getitem 函数，接收一个 PyArrayFlagsObject 类型的 self 参数和一个 ind 参数
    static PyObject *
    arrayflags_getitem(PyArrayFlagsObject *self, PyObject *ind)
    {
        // 声明一个指向字符的指针 key，初始化为 NULL
        char *key = NULL;
        // 声明一个字符数组 buf，用于临时存储字符串
        char buf[16];
        // 声明一个整数变量 n，用于存储字符串的长度
        int n;

        // 如果 ind 是 Unicode 字符串
        if (PyUnicode_Check(ind)) {
            // 声明一个 PyObject 指针 tmp_str
            PyObject *tmp_str;
            // 将 Unicode 字符串转换为 ASCII 字符串，并赋给 tmp_str
            tmp_str = PyUnicode_AsASCIIString(ind);
            // 如果转换失败，返回 NULL
            if (tmp_str == NULL) {
                return NULL;
            }
            // 将 tmp_str 转换为 C 风格的字符串，并赋给 key
            key = PyBytes_AS_STRING(tmp_str);
            // 获取 key 的长度，并赋给 n
            n = PyBytes_GET_SIZE(tmp_str);
            // 如果字符串长度超过 16，释放 tmp_str 并跳转到 fail 标签处
            if (n > 16) {
                Py_DECREF(tmp_str);
                goto fail;
            }
            // 将 key 复制到 buf 中
            memcpy(buf, key, n);
            // 释放 tmp_str
            Py_DECREF(tmp_str);
            // 将 buf 的地址赋给 key
            key = buf;
        }
        // 如果 ind 是字节字符串
        else if (PyBytes_Check(ind)) {
            // 将 ind 转换为 C 风格的字符串，并赋给 key
            key = PyBytes_AS_STRING(ind);
            // 获取 key 的长度，并赋给 n
            n = PyBytes_GET_SIZE(ind);
        }
        // 如果 ind 不是字符串类型，跳转到 fail 标签处
        else {
            goto fail;
        }
        
        // 根据字符串长度 n 执行不同的操作
        switch(n) {
        // 如果字符串长度为 1
        case 1:
            // 根据 key 的第一个字符执行不同的操作
            switch(key[0]) {
            case 'C':
                // 返回 arrayflags_contiguous_get 函数的结果
                return arrayflags_contiguous_get(self, NULL);
            case 'F':
                // 返回 arrayflags_fortran_get 函数的结果
                return arrayflags_fortran_get(self, NULL);
            case 'W':
                // 返回 arrayflags_writeable_get 函数的结果
                return arrayflags_writeable_get(self, NULL);
            case 'B':
                // 返回 arrayflags_behaved_get 函数的结果
                return arrayflags_behaved_get(self, NULL);
            case 'O':
                // 返回 arrayflags_owndata_get 函数的结果
                return arrayflags_owndata_get(self, NULL);
            case 'A':
                // 返回 arrayflags_aligned_get 函数的结果
                return arrayflags_aligned_get(self, NULL);
            case 'X':
                // 返回 arrayflags_writebackifcopy_get 函数的结果
                return arrayflags_writebackifcopy_get(self, NULL);
            default:
                // 如果 key 不匹配上述字符，跳转到 fail 标签处
                goto fail;
            }
            break;
        // 如果字符串长度为 2
        case 2:
            // 如果 key 是 "CA"，返回 arrayflags_carray_get 函数的结果
            if (strncmp(key, "CA", n) == 0) {
                return arrayflags_carray_get(self, NULL);
            }
            // 如果 key 是 "FA"，返回 arrayflags_farray_get 函数的结果
            if (strncmp(key, "FA", n) == 0) {
                return arrayflags_farray_get(self, NULL);
            }
            break;
        // 如果字符串长度为 3
        case 3:
            // 如果 key 是 "FNC"，返回 arrayflags_fnc_get 函数的结果
            if (strncmp(key, "FNC", n) == 0) {
                return arrayflags_fnc_get(self, NULL);
            }
            break;
        // 如果字符串长度为 4
        case 4:
            // 如果 key 是 "FORC"，返回 arrayflags_forc_get 函数的结果
            if (strncmp(key, "FORC", n) == 0) {
                return arrayflags_forc_get(self, NULL);
            }
            break;
        // 如果字符串长度为 6
        case 6:
            // 如果 key 是 "CARRAY"，返回 arrayflags_carray_get 函数的结果
            if (strncmp(key, "CARRAY", n) == 0) {
                return arrayflags_carray_get(self, NULL);
            }
            // 如果 key 是 "FARRAY"，返回 arrayflags_farray_get 函数的结果
            if (strncmp(key, "FARRAY", n) == 0) {
                return arrayflags_farray_get(self, NULL);
            }
            break;
        // 如果字符串长度为 7
        case 7:
            // 根据 key 的值返回相应函数的结果
            if (strncmp(key,"FORTRAN",n) == 0) {
                return arrayflags_fortran_get(self, NULL);
            }
            if (strncmp(key,"BEHAVED",n) == 0) {
                return arrayflags_behaved_get(self, NULL);
            }
            if (strncmp(key,"OWNDATA",n) == 0) {
                return arrayflags_owndata_get(self, NULL);
            }
            if (strncmp(key,"ALIGNED",n) == 0) {
                return arrayflags_aligned_get(self, NULL);
            }
            break;
        // 如果字符串长度为 9
        case 9:
            // 如果 key 是 "WRITEABLE"，返回 arrayflags_writeable_get 函数的结果
            if (strncmp(key,"WRITEABLE",n) == 0) {
                return arrayflags_writeable_get(self, NULL);
            }
            break;
        // 如果字符串长度为 10
        case 10:
            // 如果 key 是 "CONTIGUOUS"，返回 arrayflags_contiguous_get 函数的结果
            if (strncmp(key,"CONTIGUOUS",n) == 0) {
                return arrayflags_contiguous_get(self, NULL);
            }
            break;
        
        // 如果字符串长度不匹配上述任何情况，跳转到 fail 标签处
        fail:
            // 返回 NULL
            return NULL;
        }
    ```
    # 检查 case 值为 12 时的条件语句
    case 12:
        # 如果 key 的前 n 个字符与 "C_CONTIGUOUS" 完全匹配，则调用 arrayflags_contiguous_get 函数
        if (strncmp(key, "C_CONTIGUOUS", n) == 0) {
            return arrayflags_contiguous_get(self, NULL);
        }
        # 如果 key 的前 n 个字符与 "F_CONTIGUOUS" 完全匹配，则调用 arrayflags_fortran_get 函数
        if (strncmp(key, "F_CONTIGUOUS", n) == 0) {
            return arrayflags_fortran_get(self, NULL);
        }
        # 跳出 switch-case 结构
        break;
    
    # 检查 case 值为 15 时的条件语句
    case 15:
        # 如果 key 的前 n 个字符与 "WRITEBACKIFCOPY" 完全匹配，则调用 arrayflags_writebackifcopy_get 函数
        if (strncmp(key, "WRITEBACKIFCOPY", n) == 0) {
            return arrayflags_writebackifcopy_get(self, NULL);
        }
        # 跳出 switch-case 结构
        break;
    }
    
    # 如果未匹配到任何已知的 flag，执行到这一行，设置 KeyError 异常并返回 NULL
 fail:
    PyErr_SetString(PyExc_KeyError, "Unknown flag");
    return NULL;
}

static int
arrayflags_setitem(PyArrayFlagsObject *self, PyObject *ind, PyObject *item)
{
    char *key;  // 声明一个指向字符的指针变量 key
    char buf[16];  // 声明一个长度为 16 的字符数组 buf，用于存储字符串
    int n;  // 声明一个整型变量 n，用于存储字符串长度

    if (PyUnicode_Check(ind)) {  // 检查 ind 是否为 Unicode 对象
        PyObject *tmp_str;  // 声明一个 PyObject 类型的指针 tmp_str
        tmp_str = PyUnicode_AsASCIIString(ind);  // 将 Unicode 对象转换为 ASCII 字符串对象
        key = PyBytes_AS_STRING(tmp_str);  // 获取转换后的 ASCII 字符串的指针
        n = PyBytes_GET_SIZE(tmp_str);  // 获取转换后的 ASCII 字符串的长度
        if (n > 16) n = 16;  // 如果长度超过 16，则截断为 16
        memcpy(buf, key, n);  // 将 key 指向的内容复制到 buf 中
        Py_DECREF(tmp_str);  // 释放临时字符串对象的引用
        key = buf;  // 将 key 指向 buf，此时 key 指向 buf 的内容
    }
    else if (PyBytes_Check(ind)) {  // 检查 ind 是否为字节对象
        key = PyBytes_AS_STRING(ind);  // 获取字节对象的指针
        n = PyBytes_GET_SIZE(ind);  // 获取字节对象的长度
    }
    else {
        goto fail;  // 如果 ind 既不是 Unicode 对象也不是字节对象，则跳转到 fail 标签处
    }

    if (((n==9) && (strncmp(key, "WRITEABLE", n) == 0)) ||  // 检查是否为 "WRITEABLE" 或 "W"
        ((n==1) && (strncmp(key, "W", n) == 0))) {
        return arrayflags_writeable_set(self, item, NULL);  // 调用 arrayflags_writeable_set 处理
    }
    else if (((n==7) && (strncmp(key, "ALIGNED", n) == 0)) ||  // 检查是否为 "ALIGNED" 或 "A"
             ((n==1) && (strncmp(key, "A", n) == 0))) {
        return arrayflags_aligned_set(self, item, NULL);  // 调用 arrayflags_aligned_set 处理
    }
    else if (((n==15) && (strncmp(key, "WRITEBACKIFCOPY", n) == 0)) ||  // 检查是否为 "WRITEBACKIFCOPY" 或 "X"
             ((n==1) && (strncmp(key, "X", n) == 0))) {
        return arrayflags_writebackifcopy_set(self, item, NULL);  // 调用 arrayflags_writebackifcopy_set 处理
    }

 fail:
    PyErr_SetString(PyExc_KeyError, "Unknown flag");  // 设置 Key 错误异常
    return -1;  // 返回 -1 表示出错
}

static char *
_torf_(int flags, int val)
{
    if ((flags & val) == val) {  // 检查 flags 中是否包含 val 的位
        return "True";  // 如果包含，返回字符串 "True"
    }
    else {
        return "False";  // 如果不包含，返回字符串 "False"
    }
}

static PyObject *
arrayflags_print(PyArrayFlagsObject *self)
{
    int fl = self->flags;  // 获取 self 对象的 flags 属性值
    const char *_warn_on_write = "";  // 声明一个指向常量字符的指针 _warn_on_write，并初始化为空字符串

    if (fl & NPY_ARRAY_WARN_ON_WRITE) {  // 检查 flags 中是否包含 NPY_ARRAY_WARN_ON_WRITE 标志位
        _warn_on_write = "  (with WARN_ON_WRITE=True)";  // 如果包含，设置 _warn_on_write
    }
    return PyUnicode_FromFormat(
                        "  %s : %s\n  %s : %s\n"
                        "  %s : %s\n  %s : %s%s\n"
                        "  %s : %s\n  %s : %s\n",
                        "C_CONTIGUOUS",    _torf_(fl, NPY_ARRAY_C_CONTIGUOUS),  // 使用 _torf_ 函数获取对应标志位的值
                        "F_CONTIGUOUS",    _torf_(fl, NPY_ARRAY_F_CONTIGUOUS),
                        "OWNDATA",         _torf_(fl, NPY_ARRAY_OWNDATA),
                        "WRITEABLE",       _torf_(fl, NPY_ARRAY_WRITEABLE),
                        _warn_on_write,  // 输出 WARN_ON_WRITE 的状态信息
                        "ALIGNED",         _torf_(fl, NPY_ARRAY_ALIGNED),
                        "WRITEBACKIFCOPY", _torf_(fl, NPY_ARRAY_WRITEBACKIFCOPY)
    );
}

static PyObject*
arrayflags_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
    if (!PyObject_TypeCheck(other, &PyArrayFlags_Type)) {  // 检查 other 是否为 PyArrayFlagsObject 类型
        Py_RETURN_NOTIMPLEMENTED;  // 如果不是，返回未实现错误
    }

    npy_bool eq = ((PyArrayFlagsObject*) self)->flags ==  // 比较 self 和 other 的 flags 属性是否相等
                   ((PyArrayFlagsObject*) other)->flags;

    if (cmp_op == Py_EQ) {  // 如果比较操作是等于
        return PyBool_FromLong(eq);  // 返回布尔值表示是否相等
    }
    else if (cmp_op == Py_NE) {  // 如果比较操作是不等于
        return PyBool_FromLong(!eq);  // 返回布尔值表示是否不相等
    }
    else {
        Py_RETURN_NOTIMPLEMENTED;  // 其他比较操作返回未实现错误
    }
}

static PyMappingMethods arrayflags_as_mapping = {
    (lenfunc)NULL,                       /*mp_length*/  // 长度函数为空，表示不支持长度操作
    (binaryfunc)arrayflags_getitem,      /*mp_subscript*/  // 子script操作使用 arrayflags_getitem 函数
    (objobjargproc)arrayflags_setitem,   /*mp_ass_subscript*/


(objobjargproc)arrayflags_setitem,   # 将arrayflags_setitem函数指针赋给mp_ass_subscript，用于实现对象的子脚本赋值操作
# 定义静态函数arrayflags_new，用于创建新的PyArrayFlags_Type对象
static PyObject *
arrayflags_new(PyTypeObject *NPY_UNUSED(self), PyObject *args, PyObject *NPY_UNUSED(kwds))
{
    PyObject *arg=NULL;
    // 解包参数args，获取函数的唯一参数arg
    if (!PyArg_UnpackTuple(args, "flagsobj", 0, 1, &arg)) {
        // 解包失败，返回NULL
        return NULL;
    }
    // 如果arg非空且为PyArray对象
    if ((arg != NULL) && PyArray_Check(arg)) {
        // 返回一个新的PyArrayFlagsObject对象，其标记与给定的PyArray对象相关联
        return PyArray_NewFlagsObject(arg);
    }
    else {
        // 否则，返回一个新的PyArrayFlagsObject对象，不与任何PyArray对象相关联
        return PyArray_NewFlagsObject(NULL);
    }
}

// 定义PyArrayFlags_Type类型对象
NPY_NO_EXPORT PyTypeObject PyArrayFlags_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    // 对象类型名称
    .tp_name = "numpy._core.multiarray.flagsobj",
    // 对象基本大小
    .tp_basicsize = sizeof(PyArrayFlagsObject),
    // 对象析构函数，用于释放对象占用的内存
    .tp_dealloc = (destructor)arrayflags_dealloc,
    // 对象的字符串表示函数，用于打印对象信息
    .tp_repr = (reprfunc)arrayflags_print,
    // 对象作为映射类型的接口
    .tp_as_mapping = &arrayflags_as_mapping,
    // 对象的字符串表示函数，与tp_repr相同
    .tp_str = (reprfunc)arrayflags_print,
    // 对象的标志位，默认为Py_TPFLAGS_DEFAULT
    .tp_flags = Py_TPFLAGS_DEFAULT,
    // 对象的富比较函数
    .tp_richcompare = arrayflags_richcompare,
    // 对象的属性获取和设置函数
    .tp_getset = arrayflags_getsets,
    // 对象的构造函数，用于创建新对象实例
    .tp_new = arrayflags_new,
};
```