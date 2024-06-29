# `.\numpy\numpy\_core\src\multiarray\array_converter.c`

```
/*
 * This file defines an _array_converter object used internally in NumPy to
 * deal with `__array_wrap__` and `result_type()` for multiple arguments
 * where converting inputs to arrays would lose the necessary information.
 *
 * The helper thus replaces many asanyarray/asarray calls.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION  // Define to prevent use of deprecated API
#define _MULTIARRAYMODULE  // Define to indicate this is part of the multiarray module

#define PY_SSIZE_T_CLEAN  // Define to use the cleaned-up Py_ssize_t API
#include <Python.h>  // Include Python.h for Python/C API
#include <structmember.h>  // Include structmember.h for structure member definitions

#include "numpy/arrayobject.h"  // Include arrayobject.h for NumPy array object definitions
#include "arrayobject.h"  // Include arrayobject.h for array-related functions
#include "array_converter.h"  // Include array_converter.h for array conversion utilities
#include "arraywrap.h"  // Include arraywrap.h for array wrapping functions
#include "numpy/arrayscalars.h"  // Include arrayscalars.h for NumPy array scalars
#include "npy_argparse.h"  // Include npy_argparse.h for argument parsing utilities
#include "abstractdtypes.h"  // Include abstractdtypes.h for abstract data type definitions
#include "convert_datatype.h"  // Include convert_datatype.h for data type conversion utilities
#include "descriptor.h"  // Include descriptor.h for descriptor definitions
#include "npy_static_data.h"  // Include npy_static_data.h for static data used in NumPy
#include "ctors.h"  // Include ctors.h for constructor functions

#include "npy_config.h"  // Include npy_config.h for NumPy configuration

#include "array_assign.h"  // Include array_assign.h for array assignment utilities

#include "common.h"  // Include common.h for common utility functions
#include "get_attr_string.h"  // Include get_attr_string.h for functions dealing with attribute strings

// Define a static function to create a new array converter object
static PyObject *
array_converter_new(
        PyTypeObject *cls, PyObject *args, PyObject *kwds)
{
    // Check if keywords are provided; array creation helper doesn't support keywords
    if (kwds != NULL && PyDict_GET_SIZE(kwds) != 0) {
        PyErr_SetString(PyExc_TypeError,
                "Array creation helper doesn't support keywords.");
        return NULL;
    }

    // Determine the number of arguments passed
    Py_ssize_t narrs_ssize_t = (args == NULL) ? 0 : PyTuple_GET_SIZE(args);
    int narrs = (int)narrs_ssize_t;

    // Limit the number of arguments to NPY_MAXARGS
    /* Limit to NPY_MAXARGS for now. */
    if (narrs_ssize_t > NPY_MAXARGS) {
        PyErr_SetString(PyExc_RuntimeError,
            "too many arrays.");
        return NULL;
    }

    // Allocate memory for the PyArrayArrayConverterObject instance
    PyArrayArrayConverterObject *self = PyObject_NewVar(
            PyArrayArrayConverterObject, cls, narrs);
    if (self == NULL) {
        return NULL;
    }

    // Initialize the PyObject instance with PyArrayArrayConverter_Type and narrs
    PyObject_InitVar((PyVarObject *)self, &PyArrayArrayConverter_Type, narrs);

    // Initialize attributes of the array converter object
    self->narrs = 0;
    self->flags = 0;
    self->wrap = NULL;
    self->wrap_type = NULL;

    // If no arguments are passed, return the initialized object
    if (narrs == 0) {
        return (PyObject *)self;
    }

    // Set flags for the array converter object
    self->flags = (NPY_CH_ALL_PYSCALARS | NPY_CH_ALL_SCALARS);

    // Initialize creation_item pointer for iterating over items
    creation_item *item = self->items;

    // Increase self->narrs in loop for cleanup
    /* increase self->narrs in loop for cleanup */
    for (int i = 0; i < narrs; i++, item++) {
        // 将参数元组中的第 i 个对象赋给当前处理的 item 的 object 字段
        item->object = PyTuple_GET_ITEM(args, i);

        /* Fast path if input is an array (maybe FromAny should be faster): */
        // 如果 item->object 是一个 NumPy 数组，则执行快速路径
        if (PyArray_Check(item->object)) {
            // 增加 item->object 的引用计数
            Py_INCREF(item->object);
            // 将 item->object 转换为 PyArrayObject，并赋给 item->array
            item->array = (PyArrayObject *)item->object;
            // 表示 item->object 不是标量输入
            item->scalar_input = 0;
        }
        else {
            // 从 item->object 创建 PyArrayObject 对象，尝试转换为 NumPy 数组
            item->array = (PyArrayObject *)PyArray_FromAny_int(
                    item->object, NULL, NULL, 0, 0, 0, NULL,
                    &item->scalar_input);
            // 如果转换失败，跳转到失败标签 fail
            if (item->array == NULL) {
                goto fail;
            }
        }

        /* At this point, assume cleanup should happen for this item */
        // 假设此时应为该 item 执行清理工作
        self->narrs++;
        // 增加 item->object 的引用计数
        Py_INCREF(item->object);
        // 获取 item->array 的数据类型，并赋给 item->DType
        item->DType = NPY_DTYPE(PyArray_DESCR(item->array));
        // 增加 item->DType 的引用计数
        Py_INCREF(item->DType);

        /*
         * Check whether we were passed a an int/float/complex Python scalar.
         * If not, set `descr` and clear pyscalar/scalar flags as needed.
         */
        // 检查是否传递了 int/float/complex 类型的 Python 标量
        if (item->scalar_input && npy_mark_tmp_array_if_pyscalar(
                item->object, item->array, &item->DType)) {
            // 如果是 Python 标量，则设置 item->descr 为 NULL
            item->descr = NULL;
            // 不标记存储的数组为 Python 文字量
            ((PyArrayObject_fields *)(item->array))->flags &= (
                    ~NPY_ARRAY_WAS_PYTHON_LITERAL);
        }
        else {
            // 否则获取 item->array 的描述符，并增加引用计数
            item->descr = PyArray_DESCR(item->array);
            Py_INCREF(item->descr);

            // 如果不是标量输入
            if (item->scalar_input) {
                // 清除 self 的标量输入标志位
                self->flags &= ~NPY_CH_ALL_PYSCALARS;
            }
            else {
                // 清除 self 的标量输入和标量标志位
                self->flags &= ~(NPY_CH_ALL_PYSCALARS | NPY_CH_ALL_SCALARS);
            }
        }
    }

    // 成功处理完所有项后，返回 self 对象作为 PyObject 指针
    return (PyObject *)self;

  fail:
    // 处理失败时，减少 self 的引用计数并返回 NULL
    Py_DECREF(self);
    return NULL;
static PyObject *
array_converter_get_scalar_input(PyArrayArrayConverterObject *self)
{
    // 创建一个元组，用于存放返回结果，元组长度为 self->narrs
    PyObject *ret = PyTuple_New(self->narrs);
    if (ret == NULL) {
        return NULL;
    }

    // 遍历 self->items 数组，为每个元素创建一个布尔值对象，表示是否为标量输入
    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        if (item->scalar_input) {
            // 如果是标量输入，增加 True 的引用计数，并设置到元组的第 i 个位置
            Py_INCREF(Py_True);
            PyTuple_SET_ITEM(ret, i, Py_True);
        }
        else {
            // 如果不是标量输入，增加 False 的引用计数，并设置到元组的第 i 个位置
            Py_INCREF(Py_False);
            PyTuple_SET_ITEM(ret, i, Py_False);
        }
    }
    return ret;  // 返回填充好的元组对象
}

static int
find_wrap(PyArrayArrayConverterObject *self)
{
    // 如果 self->wrap 已经被设置，直接返回 0，不执行任何操作
    if (self->wrap != NULL) {
        return 0;  /* nothing to do */
    }

    // 分配临时空间用于存放对象指针，长度为 self->narrs
    PyObject **objects = PyMem_Malloc(self->narrs * sizeof(PyObject *));
    if (objects == NULL) {
        PyErr_NoMemory();  // 内存分配失败，设置内存错误并返回 -1
        return -1;
    }

    // 将 self->items 数组中的 object 指针复制到 objects 数组中
    for (int i = 0; i < self->narrs; i++) {
        objects[i] = self->items[i].object;
    }

    // 调用 npy_find_array_wrap 函数查找数组包装器，将结果存储在 self->wrap 和 self->wrap_type 中
    int ret = npy_find_array_wrap(
            self->narrs, objects, &self->wrap, &self->wrap_type);

    PyMem_FREE(objects);  // 释放临时对象数组的内存空间
    return ret;  // 返回 npy_find_array_wrap 函数的执行结果
}

typedef enum {
    CONVERT = 0,
    PRESERVE = 1,
    CONVERT_IF_NO_ARRAY = 2,
} scalar_policy;

static int
pyscalar_mode_conv(PyObject *obj, scalar_policy *policy)
{
    // 预先定义三个字符串对象的数组
    PyObject *strings[3] = {
            npy_interned_str.convert, npy_interned_str.preserve,
            npy_interned_str.convert_if_no_array};

    // 第一轮快速匹配，通过对象的身份进行比较
    for (int i = 0; i < 3; i++) {
        if (obj == strings[i]) {
            *policy = i;  // 匹配成功，设置 policy 的值为 i
            return 1;     // 返回匹配成功的标志
        }
    }

    // 第二轮比较，通过 PyObject_RichCompareBool 函数进行比较
    for (int i = 0; i < 3; i++) {
        int cmp = PyObject_RichCompareBool(obj, strings[i], Py_EQ);
        if (cmp < 0) {
            return 0;  // 比较失败，返回错误标志
        }
        if (cmp) {
            *policy = i;  // 匹配成功，设置 policy 的值为 i
            return 1;     // 返回匹配成功的标志
        }
    }

    // 如果没有匹配成功，则设置 ValueError 异常并返回错误标志
    PyErr_SetString(PyExc_ValueError,
            "invalid pyscalar mode, must be 'convert', 'preserve', or "
            "'convert_if_no_array' (default).");
    return 0;  // 返回匹配失败的标志
}

static PyObject *
array_converter_as_arrays(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 设置默认的 subok 和 policy 值
    npy_bool subok = NPY_TRUE;
    scalar_policy policy = CONVERT_IF_NO_ARRAY;

    // 解析函数参数，其中 "as_arrays" 是函数名
    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("as_arrays", args, len_args, kwnames,
            "$subok", &PyArray_BoolConverter, &subok,
            /* how to handle scalars (ignored if dtype is given). */
            "$pyscalars", &pyscalar_mode_conv, &policy,
            NULL, NULL, NULL) < 0) {
        return NULL;  // 解析参数失败，返回 NULL
    }

    // 如果 policy 为 CONVERT_IF_NO_ARRAY，则根据 self->flags 的设置调整 policy 的值
    if (policy == CONVERT_IF_NO_ARRAY) {
        if (self->flags & NPY_CH_ALL_PYSCALARS) {
            policy = CONVERT;
        }
        else {
            policy = PRESERVE;
        }
    }

    // 创建一个元组用于存放返回结果，长度为 self->narrs
    PyObject *res = PyTuple_New(self->narrs);
    if (res == NULL) {
        return NULL;  // 创建元组失败，返回 NULL
    }

    // 获取 self->items 数组的首地址
    creation_item *item = self->items;
    // 继续函数实现的下一步操作...
    // 循环遍历 self 对象的 narrs 个数
    for (int i = 0; i < self->narrs; i++, item++) {
        // 定义结果项指针 res_item
        PyObject *res_item;
        // 如果 item 的描述符为 NULL 并且策略为 PRESERVE
        if (item->descr == NULL && policy == PRESERVE) {
            // 将 res_item 指向 item 的对象，并增加其引用计数
            res_item = item->object;
            Py_INCREF(res_item);
        }
        // 否则
        else {
            // 将 res_item 指向 item 的数组对象，并增加其引用计数
            res_item = (PyObject *)item->array;
            Py_INCREF(res_item);
            // 如果 subok 参数为假
            if (!subok) {
                /* PyArray_EnsureArray steals the reference... */
                // 调用 PyArray_EnsureArray 确保 res_item 是数组对象
                res_item = PyArray_EnsureArray(res_item);
                // 如果返回值为 NULL，则跳转到 fail 标签
                if (res_item == NULL) {
                    goto fail;
                }
            }
        }

        // 将 res_item 添加到元组 res 的第 i 个位置
        if (PyTuple_SetItem(res, i, res_item) < 0) {
            // 如果添加失败，则跳转到 fail 标签
            goto fail;
        }
    }

    // 成功返回结果元组 res
    return res;

  fail:
    // 出错时，减少结果元组 res 的引用计数
    Py_DECREF(res);
    // 返回 NULL 指针
    return NULL;
}


static PyObject *
array_converter_wrap(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    PyObject *to_scalar = Py_None;
    npy_bool ensure_scalar;

    // 检查是否已经初始化包装器，若未初始化则返回 NULL
    if (find_wrap(self) < 0) {
        return NULL;
    }

    NPY_PREPARE_ARGPARSER;
    // 解析参数，支持额外参数 $to_scalar 作为布尔值，若为 None 则由输入对象决定
    if (npy_parse_arguments("wrap", args, len_args, kwnames,
            "", NULL, &obj,
            /* Three-way "bool", if `None` inspect input to decide. */
            "$to_scalar", NULL, &to_scalar,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    // 根据 $to_scalar 决定是否确保对象为标量
    if (to_scalar == Py_None) {
        ensure_scalar = self->flags & NPY_CH_ALL_SCALARS;
    }
    else {
        // 将 $to_scalar 转换为布尔值，若转换失败则返回 NULL
        if (!PyArray_BoolConverter(to_scalar, &ensure_scalar)) {
            return NULL;
        }
    }

    // 应用包装操作并返回结果
    return npy_apply_wrap(
        obj, NULL, self->wrap, self->wrap_type, NULL, ensure_scalar, NPY_FALSE);
}


static PyObject *
array_converter_result_type(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArray_Descr *result = NULL;
    npy_dtype_info dt_info = {NULL, NULL};
    npy_bool ensure_inexact = NPY_FALSE;

    /* 分配临时空间（可以优化掉） */
    void *DTypes_and_descrs = PyMem_Malloc(
            ((self->narrs + 1) * 2) * sizeof(PyObject *));
    if (DTypes_and_descrs == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    PyArray_DTypeMeta **DTypes = DTypes_and_descrs;
    PyArray_Descr **descrs = (PyArray_Descr **)(DTypes + self->narrs + 1);

    NPY_PREPARE_ARGPARSER;
    // 解析参数，支持可选参数 "extra_dtype" 和 "ensure_inexact"
    if (npy_parse_arguments("result_type", args, len_args, kwnames,
            "|extra_dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            "|ensure_inexact", &PyArray_BoolConverter, &ensure_inexact,
            NULL, NULL, NULL) < 0) {
        goto finish;
    }

    int ndescrs = 0;
    int nDTypes = 0;
    creation_item *item = self->items;
    // 遍历项目列表，收集数据类型和描述符
    for (int i = 0; i < self->narrs; i++, item++) {
        DTypes[nDTypes] = item->DType;
        nDTypes++;
        if (item->descr != NULL) {
            descrs[ndescrs] = item->descr;
            ndescrs++;
        }
    }

    // 若需要确保结果为非精确数，则设置默认浮点数类型
    if (ensure_inexact) {
        if (dt_info.dtype != NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "extra_dtype and ensure_inexact are mutually exclusive.");
            goto finish;
        }
        Py_INCREF(&PyArray_PyFloatDType);
        dt_info.dtype = &PyArray_PyFloatDType;
    }

    // 添加额外的数据类型和描述符
    if (dt_info.dtype != NULL) {
        DTypes[nDTypes] = dt_info.dtype;
        nDTypes++;
    }
    if (dt_info.descr != NULL) {
        descrs[ndescrs] = dt_info.descr;
        ndescrs++;
    }

    // 推断通用数据类型
    PyArray_DTypeMeta *common_dtype = PyArray_PromoteDTypeSequence(
            nDTypes, DTypes);
    if (common_dtype == NULL) {
        goto finish;
    }
    // 若无描述符，则调用默认描述符生成函数
    if (ndescrs == 0) {
        result = NPY_DT_CALL_default_descr(common_dtype);
    }
    else {
        // 否则，调用 PyArray_CastToDTypeAndPromoteDescriptors 函数进行类型转换和描述符提升
        result = PyArray_CastToDTypeAndPromoteDescriptors(
                ndescrs, descrs, common_dtype);
    }
    // 释放 common_dtype 对象的引用计数
    Py_DECREF(common_dtype);

  finish:
    // 释放 dt_info.descr 对象的引用计数
    Py_XDECREF(dt_info.descr);
    // 释放 dt_info.dtype 对象的引用计数
    Py_XDECREF(dt_info.dtype);
    // 释放 DTypes_and_descrs 所占用的内存
    PyMem_Free(DTypes_and_descrs);
    // 返回 result 对象作为 PyObject 指针类型的结果
    return (PyObject *)result;
# 定义一个静态的属性获取器和设置器数组，用于Python对象的特性获取和设置
static PyGetSetDef array_converter_getsets[] = {
    {"scalar_input",
        (getter)array_converter_get_scalar_input,  # 获取标量输入的函数指针
        NULL,  # 没有设置器，因此设为NULL
        NULL, NULL},  # 获取器和设置器的文档字符串为空
    {NULL, NULL, NULL, NULL, NULL},  # 结束符号，用于指示属性列表的结束
};

# 定义一个静态的方法数组，用于Python对象的方法定义
static PyMethodDef array_converter_methods[] = {
    {"as_arrays", 
        (PyCFunction)array_converter_as_arrays,  # 指向转换为数组的函数指针
        METH_FASTCALL | METH_KEYWORDS, NULL},  # 使用快速调用和关键字参数的方法标志，文档字符串为空
    {"result_type",
        (PyCFunction)array_converter_result_type,  # 指向结果类型函数的函数指针
        METH_FASTCALL | METH_KEYWORDS, NULL},  # 使用快速调用和关键字参数的方法标志，文档字符串为空
    {"wrap",
        (PyCFunction)array_converter_wrap,  # 指向包装函数的函数指针
        METH_FASTCALL | METH_KEYWORDS, NULL},  # 使用快速调用和关键字参数的方法标志，文档字符串为空
    {NULL, NULL, 0, NULL}  # 结束符号，用于指示方法列表的结束
};

# 定义一个静态的析构函数，用于释放内存和资源
static void
array_converter_dealloc(PyArrayArrayConverterObject *self)
{
    creation_item *item = self->items;  # 初始化创建项指针，指向self对象的items数组
    for (int i = 0; i < self->narrs; i++, item++) {
        Py_XDECREF(item->array);  # 递减引用计数并释放数组对象
        Py_XDECREF(item->object);  # 递减引用计数并释放Python对象
        Py_XDECREF(item->DType);  # 递减引用计数并释放数据类型对象
        Py_XDECREF(item->descr);  # 递减引用计数并释放描述符对象
    }

    Py_XDECREF(self->wrap);  # 递减引用计数并释放wrap对象
    Py_XDECREF(self->wrap_type);  # 递减引用计数并释放wrap_type对象
    PyObject_Del((PyObject *)self);  # 删除Python对象self并释放其内存
}

# 定义一个长度函数，返回PyArrayArrayConverterObject对象的长度
static Py_ssize_t
array_converter_length(PyArrayArrayConverterObject *self)
{
    return self->narrs;  # 返回对象self的narrs属性，即数组的数量
}

# 定义一个项目获取函数，返回PyArrayArrayConverterObject对象的第item项
static PyObject *
array_converter_item(PyArrayArrayConverterObject *self, Py_ssize_t item)
{
    /* Python ensures no negative indices (and probably the below also) */
    # Python确保没有负索引（也可能下面的操作也是如此）

    if (item < 0 || item >= self->narrs) {
        PyErr_SetString(PyExc_IndexError, "invalid index");  # 设置索引错误异常并返回NULL
        return NULL;
    }

    /* Follow the `as_arrays` default of `CONVERT_IF_NO_ARRAY`: */
    # 遵循`as_arrays`的默认设置`CONVERT_IF_NO_ARRAY`

    PyObject *res;
    if (self->items[item].descr == NULL
            && !(self->flags & NPY_CH_ALL_PYSCALARS)) {
        res = self->items[item].object;  # 如果描述符为NULL且标志中不包含所有Python标量，则结果为对象
    }
    else {
        res = (PyObject *)self->items[item].array;  # 否则结果为数组对象的PyObject指针转换
    }

    Py_INCREF(res);  # 增加结果对象的引用计数
    return res;  # 返回结果对象
}

# 定义一个序列方法结构体，包含了长度函数和项目获取函数
static PySequenceMethods array_converter_as_sequence = {
    .sq_length = (lenfunc)array_converter_length,  # 序列长度函数指针
    .sq_item = (ssizeargfunc)array_converter_item,  # 序列项目获取函数指针
};

# 定义一个PyTypeObject对象，描述PyArrayArrayConverterObject对象的类型信息
NPY_NO_EXPORT PyTypeObject PyArrayArrayConverter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)  # 初始化Python对象头部信息
    .tp_name = "numpy._core._multiarray_umath._array_converter",  # 类型对象的名称
    .tp_basicsize = sizeof(PyArrayArrayConverterObject),  # 类型对象的基本大小
    .tp_itemsize = sizeof(creation_item),  # 类型对象的每个项的大小
    .tp_new = array_converter_new,  # 类型对象的构造函数
    .tp_dealloc = (destructor)array_converter_dealloc,  # 类型对象的析构函数
    .tp_flags = Py_TPFLAGS_DEFAULT,  # 类型对象的标志
    .tp_getset = array_converter_getsets,  # 类型对象的属性获取器和设置器数组
    .tp_methods = array_converter_methods,  # 类型对象的方法数组
    .tp_as_sequence = &array_converter_as_sequence,  # 类型对象的序列方法结构体
};
```