# `.\numpy\numpy\_core\src\multiarray\getset.c`

```
/* Array Descr Object */

/* Define to prevent deprecated API usage */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Define to enable multiarray module */
#define _MULTIARRAYMODULE

/* Ensure Python.h uses modern Py_ssize_t definitions */
#define PY_SSIZE_T_CLEAN

/* Include Python core header */
#include <Python.h>

/* Include structmember.h for C struct and object member API */
#include <structmember.h>

/* Include NumPy's array object header */
#include "numpy/arrayobject.h"

/* Include NumPy configuration */
#include "npy_config.h"

/* Include NumPy import utilities */
#include "npy_import.h"

/* Include common utility functions */
#include "common.h"

/* Include conversion utilities */
#include "conversion_utils.h"

/* Include constructors for arrays */
#include "ctors.h"

/* Include dtype meta information */
#include "dtypemeta.h"

/* Include scalar types definitions */
#include "scalartypes.h"

/* Include array descriptor definitions */
#include "descriptor.h"

/* Include flags object definitions */
#include "flagsobject.h"

/* Include getter/setter definitions */
#include "getset.h"

/* Include main array object definitions */
#include "arrayobject.h"

/* Include memory overlap handling */
#include "mem_overlap.h"

/* Include memory allocation utilities */
#include "alloc.h"

/* Include buffer handling utilities */
#include "npy_buffer.h"

/* Include shape manipulation utilities */
#include "shape.h"

/* Include multiarray module utilities */
#include "multiarraymodule.h"

/*******************  array attribute get and set routines ******************/

/* Retrieve the number of dimensions of the array */
static PyObject *
array_ndim_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(PyArray_NDIM(self));
}

/* Retrieve array flags as a flags object */
static PyObject *
array_flags_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_NewFlagsObject((PyObject *)self);
}

/* Retrieve array shape as a tuple of integers */
static PyObject *
array_shape_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_DIMS(self));
}

/* Set array shape from a Python object */
static int
array_shape_set(PyArrayObject *self, PyObject *val, void* NPY_UNUSED(ignored))
{
    int nd;
    PyArrayObject *ret;

    /* Check if val is NULL (deletion not allowed) */
    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array shape");
        return -1;
    }

    /* Attempt to reshape the array using the provided shape */
    ret = (PyArrayObject *)PyArray_Reshape(self, val);
    if (ret == NULL) {
        return -1;
    }

    /* Check if the reshape operation resulted in a new data pointer */
    if (PyArray_DATA(ret) != PyArray_DATA(self)) {
        Py_DECREF(ret);
        PyErr_SetString(PyExc_AttributeError,
                        "Incompatible shape for in-place modification. Use "
                        "`.reshape()` to make a copy with the desired shape.");
        return -1;
    }

    /* Obtain the number of dimensions after reshape */
    nd = PyArray_NDIM(ret);

    /* Handle array reshaping */
    if (nd > 0) {
        /* Allocate new dimensions and strides */
        npy_intp *_dimensions = npy_alloc_cache_dim(2 * nd);
        if (_dimensions == NULL) {
            Py_DECREF(ret);
            PyErr_NoMemory();
            return -1;
        }

        /* Free old dimensions and strides */
        npy_free_cache_dim_array(self);

        /* Update array fields with new dimensions and strides */
        ((PyArrayObject_fields *)self)->nd = nd;
        ((PyArrayObject_fields *)self)->dimensions = _dimensions; 
        ((PyArrayObject_fields *)self)->strides = _dimensions + nd;

        /* Copy new dimensions and strides */
        if (nd) {
            memcpy(PyArray_DIMS(self), PyArray_DIMS(ret), nd*sizeof(npy_intp));
            memcpy(PyArray_STRIDES(self), PyArray_STRIDES(ret), nd*sizeof(npy_intp));
        }
    }
    else {
        /* Free old dimensions and strides for zero-dimensional arrays */
        npy_free_cache_dim_array(self);        
        ((PyArrayObject_fields *)self)->nd = 0;
        ((PyArrayObject_fields *)self)->dimensions = NULL;
        ((PyArrayObject_fields *)self)->strides = NULL;
    }

    /* Release temporary reshape result */
    Py_DECREF(ret);

    /* Update array flags */
    PyArray_UpdateFlags(self, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);

    /* Successful array shape update */
    return 0;
}
/*
 * 从数组对象中获取步幅信息，并返回一个包含步幅信息的元组对象
 */
array_strides_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_STRIDES(self));
}

/*
 * 设置数组对象的步幅信息
 */
static int
array_strides_set(PyArrayObject *self, PyObject *obj, void *NPY_UNUSED(ignored))
{
    PyArray_Dims newstrides = {NULL, -1};
    PyArrayObject *new;
    npy_intp numbytes = 0;
    npy_intp offset = 0;
    npy_intp lower_offset = 0;
    npy_intp upper_offset = 0;
    Py_buffer view;

    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array strides");
        return -1;
    }
    if (!PyArray_OptionalIntpConverter(obj, &newstrides) ||
        newstrides.len == -1) {
        PyErr_SetString(PyExc_TypeError, "invalid strides");
        return -1;
    }
    if (newstrides.len != PyArray_NDIM(self)) {
        PyErr_Format(PyExc_ValueError, "strides must be "       \
                     " same length as shape (%d)", PyArray_NDIM(self));
        goto fail;
    }
    new = self;
    while(PyArray_BASE(new) && PyArray_Check(PyArray_BASE(new))) {
        new = (PyArrayObject *)(PyArray_BASE(new));
    }
    /*
     * 通过缓冲区接口获取PyArray_BASE(new)的可用内存，如果失败则从当前的new获取
     */
    if (PyArray_BASE(new) &&
            PyObject_GetBuffer(PyArray_BASE(new), &view, PyBUF_SIMPLE) >= 0) {
        offset = PyArray_BYTES(self) - (char *)view.buf;
        numbytes = view.len + offset;
        PyBuffer_Release(&view);
    }
    else {
        PyErr_Clear();
        offset_bounds_from_strides(PyArray_ITEMSIZE(new), PyArray_NDIM(new),
                                   PyArray_DIMS(new), PyArray_STRIDES(new),
                                   &lower_offset, &upper_offset);

        offset = PyArray_BYTES(self) - (PyArray_BYTES(new) + lower_offset);
        numbytes = upper_offset - lower_offset;
    }

    /* numbytes == 0 is special here, but the 0-size array case always works */
    if (!PyArray_CheckStrides(PyArray_ITEMSIZE(self), PyArray_NDIM(self),
                              numbytes, offset,
                              PyArray_DIMS(self), newstrides.ptr)) {
        PyErr_SetString(PyExc_ValueError, "strides is not "\
                        "compatible with available memory");
        goto fail;
    }
    if (newstrides.len) {
        memcpy(PyArray_STRIDES(self), newstrides.ptr, sizeof(npy_intp)*newstrides.len);
    }
    PyArray_UpdateFlags(self, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS |
                              NPY_ARRAY_ALIGNED);
    npy_free_cache_dim_obj(newstrides);
    return 0;

 fail:
    npy_free_cache_dim_obj(newstrides);
    return -1;
}

/*
 * 返回数组对象的优先级作为一个Python浮点数对象
 */
static PyObject *
array_priority_get(PyArrayObject *NPY_UNUSED(self), void *NPY_UNUSED(ignored))
{
    return PyFloat_FromDouble(NPY_PRIORITY);
}

/*
 * 返回数组对象的类型描述符字符串
 */
static PyObject *
array_typestr_get(PyArrayObject *self)
{
    return arraydescr_protocol_typestr_get(PyArray_DESCR(self), NULL);
}

/*
 * 继续添加函数定义...
 */
static PyObject *
array_interface_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *dict;      // 创建一个新的 Python 字典对象，用于存储接口信息
    PyObject *obj;       // 用于临时存储从其他函数返回的 Python 对象
    int ret;             // 用于存储 PyDict_SetItemString 函数的返回值

    dict = PyDict_New(); // 创建一个新的空字典对象
    if (dict == NULL) {  // 检查字典创建是否成功
        return NULL;     // 如果创建失败，返回空指针
    }

    /* dataptr */
    obj = array_dataptr_get(self, NULL);      // 调用 array_dataptr_get 获取数据指针对象
    ret = PyDict_SetItemString(dict, "data", obj); // 将数据指针对象存入字典中
    Py_DECREF(obj);     // 减少数据指针对象的引用计数
    if (ret < 0) {       // 检查 PyDict_SetItemString 是否成功
        Py_DECREF(dict); // 失败时释放字典对象并返回空指针
        return NULL;
    }

    obj = array_protocol_strides_get(self);  // 调用 array_protocol_strides_get 获取步幅对象
    ret = PyDict_SetItemString(dict, "strides", obj); // 将步幅对象存入字典中
    Py_DECREF(obj);     // 减少步幅对象的引用计数
    if (ret < 0) {       // 检查 PyDict_SetItemString 是否成功
        Py_DECREF(dict); // 失败时释放字典对象并返回空指针
        return NULL;
    }

    obj = array_protocol_descr_get(self);    // 调用 array_protocol_descr_get 获取描述符对象
    ret = PyDict_SetItemString(dict, "descr", obj); // 将描述符对象存入字典中
    Py_DECREF(obj);     // 减少描述符对象的引用计数
    if (ret < 0) {       // 检查 PyDict_SetItemString 是否成功
        Py_DECREF(dict); // 失败时释放字典对象并返回空指针
        return NULL;
    }

    obj = arraydescr_protocol_typestr_get(PyArray_DESCR(self), NULL); // 调用 arraydescr_protocol_typestr_get 获取类型字符串对象
    ret = PyDict_SetItemString(dict, "typestr", obj); // 将类型字符串对象存入字典中
    Py_DECREF(obj);     // 减少类型字符串对象的引用计数
    if (ret < 0) {       // 检查 PyDict_SetItemString 是否成功
        Py_DECREF(dict); // 失败时释放字典对象并返回空指针
        return NULL;
    }

    obj = array_shape_get(self, NULL);       // 调用 array_shape_get 获取形状对象
    ret = PyDict_SetItemString(dict, "shape", obj); // 将形状对象存入字典中
    Py_DECREF(obj);     // 减少形状对象的引用计数

    // 返回填充完毕的字典对象，包含了"data", "strides", "descr", "typestr", "shape"等键对应的值
    return dict;
}
    # 如果 ret 小于 0，则表示在之前的操作中出现了错误，需要释放字典对象并返回空指针
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    # 创建一个整数对象，表示版本号为 3
    obj = PyLong_FromLong(3);
    # 将整数对象作为值，键为 "version"，添加到字典中
    ret = PyDict_SetItemString(dict, "version", obj);
    Py_DECREF(obj);
    # 如果 ret 小于 0，则表示在设置字典项时出现了错误，需要释放字典对象并返回空指针
    if (ret < 0) {
        Py_DECREF(dict);
        return NULL;
    }

    # 返回已经填充好的字典对象
    return dict;
}

/*
 * 返回一个内存视图对象，表示数组的数据
 */
static PyObject *
array_data_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyMemoryView_FromObject((PyObject *)self);
}

/*
 * 返回一个长整型对象，表示数组每个元素的字节大小
 */
static PyObject *
array_itemsize_get(PyArrayObject *self, void* NPY_UNUSED(ignored))
{
    return PyLong_FromLong((long) PyArray_ITEMSIZE(self));
}

/*
 * 返回一个整型对象，表示数组中元素的总数
 */
static PyObject *
array_size_get(PyArrayObject *self, void* NPY_UNUSED(ignored))
{
    return PyArray_PyIntFromIntp(PyArray_SIZE(self));
}

/*
 * 返回一个整型对象，表示数组所占用的总字节数
 */
static PyObject *
array_nbytes_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_PyIntFromIntp(PyArray_NBYTES(self));
}

/*
 * 当数组的数据类型发生改变时调用此函数。
 * 若itemsize保持不变或者数组是单段的（连续或Fortran），并且维度兼容，
 * 则形状和步长也将被相应调整。
 */
static int
array_descr_set(PyArrayObject *self, PyObject *arg, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *newtype = NULL;

    // 如果传入的arg为NULL，则不能删除数组的数据类型，抛出异常
    if (arg == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array dtype");
        return -1;
    }

    // 尝试将arg转换为PyArray_Descr类型的对象newtype
    if (!(PyArray_DescrConverter(arg, &newtype)) ||
        newtype == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "invalid data-type for array");
        return -1;
    }

    /* 检查是否涉及到包含对象的内存重解释 */
    if (_may_have_objects(PyArray_DESCR(self)) || _may_have_objects(newtype)) {
        PyObject *safe;

        // 导入numpy._core._internal模块中的_view_is_safe函数
        npy_cache_import("numpy._core._internal", "_view_is_safe",
                         &npy_thread_unsafe_state._view_is_safe);
        if (npy_thread_unsafe_state._view_is_safe == NULL) {
            goto fail;
        }

        // 调用_view_is_safe函数，检查是否安全
        safe = PyObject_CallFunction(npy_thread_unsafe_state._view_is_safe,
                                     "OO", PyArray_DESCR(self), newtype);
        if (safe == NULL) {
            goto fail;
        }
        Py_DECREF(safe);
    }

    /*
     * 若新类型是无大小的void类型，则它的大小应该与当前dtype的itemsize相匹配。
     * 若不匹配，则将newtype的大小调整为当前dtype的itemsize。
     */
    if (newtype->type_num == NPY_VOID &&
            PyDataType_ISUNSIZED(newtype) &&
            newtype->elsize != PyArray_ITEMSIZE(self)) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return -1;
        }
        newtype->elsize = PyArray_ITEMSIZE(self);
    }

    /* 更改dtype的大小会导致形状发生变化 */
    // 检查新数据类型的元素大小是否与数组元素大小相同
    if (newtype->elsize != PyArray_ITEMSIZE(self)) {
        /* 禁止的情况 */
        
        // 如果数组是0维的，只有当元素大小不变时才支持更改数据类型
        if (PyArray_NDIM(self) == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Changing the dtype of a 0d array is only supported "
                    "if the itemsize is unchanged");
            // 跳转到错误处理标签
            goto fail;
        }
        // 如果新数据类型是子数组类型，只有当总元素大小不变时才支持更改数据类型
        else if (PyDataType_HASSUBARRAY(newtype)) {
            PyErr_SetString(PyExc_ValueError,
                    "Changing the dtype to a subarray type is only supported "
                    "if the total itemsize is unchanged");
            // 跳转到错误处理标签
            goto fail;
        }

        /* 只在最后一个轴上调整大小 */
        int axis = PyArray_NDIM(self) - 1;
        // 如果最后一个轴的维度不为1，并且数组大小不为0，并且最后一个轴上的步长不等于元素大小，则报错
        if (PyArray_DIMS(self)[axis] != 1 &&
                PyArray_SIZE(self) != 0 &&
                PyArray_STRIDES(self)[axis] != PyArray_ITEMSIZE(self)) {
            PyErr_SetString(PyExc_ValueError,
                    "To change to a dtype of a different size, the last axis "
                    "must be contiguous");
            // 跳转到错误处理标签
            goto fail;
        }

        npy_intp newdim;

        // 如果新数据类型的元素大小小于数组的元素大小
        if (newtype->elsize < PyArray_ITEMSIZE(self)) {
            /* 如果兼容，增加最后一个轴的大小 */
            // 如果新数据类型的元素大小为0或者原始数据类型的大小不能整除新数据类型的大小，则报错
            if (newtype->elsize == 0 ||
                    PyArray_ITEMSIZE(self) % newtype->elsize != 0) {
                PyErr_SetString(PyExc_ValueError,
                        "When changing to a smaller dtype, its size must be a "
                        "divisor of the size of original dtype");
                // 跳转到错误处理标签
                goto fail;
            }
            // 计算新的维度大小
            newdim = PyArray_ITEMSIZE(self) / newtype->elsize;
            // 更新最后一个轴的维度
            PyArray_DIMS(self)[axis] *= newdim;
            // 更新最后一个轴的步长为新数据类型的元素大小
            PyArray_STRIDES(self)[axis] = newtype->elsize;
        }
        else /* newtype->elsize > PyArray_ITEMSIZE(self) */ {
            /* 如果兼容，减少相关轴的大小 */
            // 计算新的维度大小
            newdim = PyArray_DIMS(self)[axis] * PyArray_ITEMSIZE(self);
            // 如果不能整除新数据类型的大小，则报错
            if ((newdim % newtype->elsize) != 0) {
                PyErr_SetString(PyExc_ValueError,
                        "When changing to a larger dtype, its size must be a "
                        "divisor of the total size in bytes of the last axis "
                        "of the array.");
                // 跳转到错误处理标签
                goto fail;
            }
            // 更新最后一个轴的维度
            PyArray_DIMS(self)[axis] = newdim / newtype->elsize;
            // 更新最后一个轴的步长为新数据类型的元素大小
            PyArray_STRIDES(self)[axis] = newtype->elsize;
        }
    }

    /* 将视图作为子数组会增加维数 */
    if (PyDataType_HASSUBARRAY(newtype)) {
        /*
         * 如果新类型有子数组，
         * 创建新的数组对象，并从中更新维度、步长和描述符
         */
        PyArrayObject *temp;
        /*
         * 在这里我们会减少 newtype 的引用计数。
         * temp 将会获取它的引用
         */
        temp = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type, newtype, PyArray_NDIM(self),
                                 PyArray_DIMS(self), PyArray_STRIDES(self),
                                 PyArray_DATA(self), PyArray_FLAGS(self), NULL);
        if (temp == NULL) {
            return -1;
        }
        npy_free_cache_dim_array(self);
        ((PyArrayObject_fields *)self)->dimensions = PyArray_DIMS(temp);
        ((PyArrayObject_fields *)self)->nd = PyArray_NDIM(temp);
        ((PyArrayObject_fields *)self)->strides = PyArray_STRIDES(temp);
        newtype = PyArray_DESCR(temp);
        Py_INCREF(PyArray_DESCR(temp));
        /* 避免释放器删除这些 */
        ((PyArrayObject_fields *)temp)->nd = 0;
        ((PyArrayObject_fields *)temp)->dimensions = NULL;
        Py_DECREF(temp);
    }

    Py_DECREF(PyArray_DESCR(self));
    ((PyArrayObject_fields *)self)->descr = newtype;
    PyArray_UpdateFlags(self, NPY_ARRAY_UPDATE_ALL);
    return 0;

 fail:
    Py_DECREF(newtype);
    return -1;
static PyObject *
array_struct_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    // 分配内存以存储 PyArrayInterface 结构
    PyArrayInterface *inter;

    inter = (PyArrayInterface *)PyArray_malloc(sizeof(PyArrayInterface));
    // 检查内存分配是否成功
    if (inter==NULL) {
        // 内存分配失败，返回内存错误异常
        return PyErr_NoMemory();
    }
    // 设置 PyArrayInterface 结构的字段
    inter->two = 2;
    inter->nd = PyArray_NDIM(self);
    inter->typekind = PyArray_DESCR(self)->kind;
    inter->itemsize = PyArray_ITEMSIZE(self);
    inter->flags = PyArray_FLAGS(self);
    if (inter->flags & NPY_ARRAY_WARN_ON_WRITE) {
        /* Export a warn-on-write array as read-only */
        // 如果数组标志包含 NPY_ARRAY_WARN_ON_WRITE，将其设置为只读
        inter->flags = inter->flags & ~NPY_ARRAY_WARN_ON_WRITE;
        inter->flags = inter->flags & ~NPY_ARRAY_WRITEABLE;
    }
    /* reset unused flags */
    // 重置未使用的标志位
    inter->flags &= ~(NPY_ARRAY_WRITEBACKIFCOPY | NPY_ARRAY_OWNDATA);
    if (PyArray_ISNOTSWAPPED(self)) inter->flags |= NPY_ARRAY_NOTSWAPPED;
    /*
     * Copy shape and strides over since these can be reset
     * when the array is "reshaped".
     */
    // 复制形状和步幅，因为这些在“重塑”数组时可能会被重置
    if (PyArray_NDIM(self) > 0) {
        // 分配内存以存储形状和步幅
        inter->shape = (npy_intp *)PyArray_malloc(2*sizeof(npy_intp)*PyArray_NDIM(self));
        if (inter->shape == NULL) {
            // 内存分配失败，释放已分配的内存，并返回内存错误异常
            PyArray_free(inter);
            return PyErr_NoMemory();
        }
        // 设置步幅为形状数组的末尾
        inter->strides = inter->shape + PyArray_NDIM(self);
        if (PyArray_NDIM(self)) {
            // 复制形状和步幅数据
            memcpy(inter->shape, PyArray_DIMS(self), sizeof(npy_intp)*PyArray_NDIM(self));
            memcpy(inter->strides, PyArray_STRIDES(self), sizeof(npy_intp)*PyArray_NDIM(self));
        }
    }
    else {
        // 数组没有维度，设置形状和步幅为 NULL
        inter->shape = NULL;
        inter->strides = NULL;
    }
    // 设置数据指针
    inter->data = PyArray_DATA(self);
    if (PyDataType_HASFIELDS(PyArray_DESCR(self))) {
        // 如果数据类型有字段，获取字段描述符
        inter->descr = arraydescr_protocol_descr_get(PyArray_DESCR(self), NULL);
        if (inter->descr == NULL) {
            // 获取描述符失败，清除错误状态
            PyErr_Clear();
        }
        else {
            // 设置数组描述符标志位
            inter->flags &= NPY_ARR_HAS_DESCR;
        }
    }
    else {
        // 没有字段，描述符设置为 NULL
        inter->descr = NULL;
    }
    // 创建 PyCapsule 对象来封装 inter 结构
    PyObject *ret = PyCapsule_New(inter, NULL, gentype_struct_free);
    if (ret == NULL) {
        // 创建 PyCapsule 对象失败，返回 NULL
        return NULL;
    }
    // 增加数组对象的引用计数
    Py_INCREF(self);
    // 将数组对象设置为 PyCapsule 对象的上下文
    if (PyCapsule_SetContext(ret, self) < 0) {
        // 设置上下文失败，返回 NULL
        return NULL;
    }
    // 返回 PyCapsule 对象
    return ret;
}

static PyObject *
array_base_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    // 检查数组的基础对象是否为 NULL
    if (PyArray_BASE(self) == NULL) {
        // 基础对象为 NULL，返回 None
        Py_RETURN_NONE;
    }
    else {
        // 增加基础对象的引用计数，并返回基础对象
        Py_INCREF(PyArray_BASE(self));
        return PyArray_BASE(self);
    }
}

/*
 * Create a view of a complex array with an equivalent data-type
 * except it is real instead of complex.
 */
static PyArrayObject *
_get_part(PyArrayObject *self, int imag)
{
    // 定义浮点类型编号、数据类型、返回的数组对象、偏移量等变量
    int float_type_num;
    PyArray_Descr *type;
    PyArrayObject *ret;
    int offset;
    # 根据当前数组的描述符中的类型号进行切换
    switch (PyArray_DESCR(self)->type_num) {
        # 如果是复数浮点数，设置对应的浮点数类型号为NPY_FLOAT
        case NPY_CFLOAT:
            float_type_num = NPY_FLOAT;
            break;
        # 如果是双精度复数浮点数，设置浮点数类型号为NPY_DOUBLE
        case NPY_CDOUBLE:
            float_type_num = NPY_DOUBLE;
            break;
        # 如果是长双精度复数浮点数，设置浮点数类型号为NPY_LONGDOUBLE
        case NPY_CLONGDOUBLE:
            float_type_num = NPY_LONGDOUBLE;
            break;
        # 如果以上情况都不匹配，则抛出异常并返回NULL
        default:
            PyErr_Format(PyExc_ValueError,
                         "Cannot convert complex type number %d to float",
                         PyArray_DESCR(self)->type_num);
            return NULL;

    }
    # 根据浮点数类型号获取对应的描述符
    type = PyArray_DescrFromType(float_type_num);
    # 如果获取描述符失败，则返回NULL
    if (type == NULL) {
        return NULL;
    }

    # 如果imag为真，则偏移量为描述符元素大小，否则偏移量为0
    offset = (imag ? type->elsize : 0);

    # 如果数组的字节顺序不是本机字节顺序
    if (!PyArray_ISNBO(PyArray_DESCR(self)->byteorder)) {
        # 复制描述符并检查是否成功
        Py_SETREF(type, PyArray_DescrNew(type));
        if (type == NULL) {
            return NULL;
        }
        # 设置复制后的描述符的字节顺序与数组的字节顺序相同
        type->byteorder = PyArray_DESCR(self)->byteorder;
    }
    # 使用提供的描述符和数据创建新的数组对象
    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            Py_TYPE(self),
            type,
            PyArray_NDIM(self),
            PyArray_DIMS(self),
            PyArray_STRIDES(self),
            PyArray_BYTES(self) + offset,
            PyArray_FLAGS(self), (PyObject *)self, (PyObject *)self);
    # 如果创建数组对象失败，则返回NULL
    if (ret == NULL) {
        return NULL;
    }
    # 返回创建的数组对象
    return ret;
/* For Object arrays, we need to get and set the
   real part of each element.
 */
static PyObject *
array_real_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;

    // 如果数组是复数类型的
    if (PyArray_ISCOMPLEX(self)) {
        // 调用内部函数获取数组的实部
        ret = _get_part(self, 0);
        return (PyObject *)ret;
    }
    else {
        // 如果不是复数类型，增加引用计数并返回自身
        Py_INCREF(self);
        return (PyObject *)self;
    }
}

static int
array_real_set(PyArrayObject *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;
    PyArrayObject *new;
    int retcode;

    // 如果传入的值是空，则无法删除数组的实部
    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array real part");
        return -1;
    }
    // 如果数组是复数类型的
    if (PyArray_ISCOMPLEX(self)) {
        // 调用内部函数获取数组的实部
        ret = _get_part(self, 0);
        if (ret == NULL) {
            return -1;
        }
    }
    else {
        // 如果不是复数类型，增加引用计数并返回自身
        Py_INCREF(self);
        ret = self;
    }
    // 将传入的值转换为数组对象
    new = (PyArrayObject *)PyArray_FROM_O(val);
    if (new == NULL) {
        Py_DECREF(ret);
        return -1;
    }
    // 将新值复制到实部数组中
    retcode = PyArray_CopyInto(ret, new);
    Py_DECREF(ret);
    Py_DECREF(new);
    return retcode;
}

/* For Object arrays we need to get
   and set the imaginary part of
   each element
*/
static PyObject *
array_imag_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;

    // 如果数组是复数类型的
    if (PyArray_ISCOMPLEX(self)) {
        // 调用内部函数获取数组的虚部
        ret = _get_part(self, 1);
    }
    else {
        // 如果不是复数类型，增加描述符的引用计数并创建一个新的数组对象
        Py_INCREF(PyArray_DESCR(self));
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                Py_TYPE(self),
                PyArray_DESCR(self),
                PyArray_NDIM(self),
                PyArray_DIMS(self),
                NULL, NULL,
                PyArray_ISFORTRAN(self),
                (PyObject *)self, NULL, _NPY_ARRAY_ZEROED);
        if (ret == NULL) {
            return NULL;
        }
        // 清除可写标志，使得数组不可写
        PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
    return (PyObject *) ret;
}

static int
array_imag_set(PyArrayObject *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    // 如果传入的值是空，则无法删除数组的虚部
    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array imaginary part");
        return -1;
    }
    // 如果数组是复数类型的
    if (PyArray_ISCOMPLEX(self)) {
        PyArrayObject *ret;
        PyArrayObject *new;
        int retcode;

        // 调用内部函数获取数组的虚部
        ret = _get_part(self, 1);
        if (ret == NULL) {
            return -1;
        }
        // 将传入的值转换为数组对象
        new = (PyArrayObject *)PyArray_FROM_O(val);
        if (new == NULL) {
            Py_DECREF(ret);
            return -1;
        }
        // 将新值复制到虚部数组中
        retcode = PyArray_CopyInto(ret, new);
        Py_DECREF(ret);
        Py_DECREF(new);
        return retcode;
    }
    else {
        // 如果不是复数类型，抛出类型错误
        PyErr_SetString(PyExc_TypeError,
                "array does not have imaginary part to set");
        return -1;
    }
}

static PyObject *
array_flat_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    // 返回一个数组的迭代器对象
    return PyArray_IterNew((PyObject *)self);
}

static int
array_flat_set(PyArrayObject *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    // 省略部分代码，不在注释范围内
}
    // 声明一个指向 PyArrayObject 结构体的指针，用于存储数组对象
    PyArrayObject *arr = NULL;
    // 初始化返回值为 -1，表示函数执行失败
    int retval = -1;
    // 声明两个迭代器对象的指针，分别用于自身数组和传入数组
    PyArrayIterObject *selfit = NULL, *arrit = NULL;
    // 声明一个指向数组描述符的指针，用于存储数组的类型信息
    PyArray_Descr *typecode;
    // 用于存储是否需要交换字节顺序的标志
    int swap;
    // 声明一个函数指针，指向复制和交换数据的函数
    PyArray_CopySwapFunc *copyswap;

    // 如果传入的值为 NULL，则设置异常并返回 -1
    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array flat iterator");
        return -1;
    }
    // 检查当前数组是否可写，若不可写则返回 -1
    if (PyArray_FailUnlessWriteable(self, "array") < 0) return -1;
    // 获取当前数组的类型描述符，并增加其引用计数
    typecode = PyArray_DESCR(self);
    Py_INCREF(typecode);
    // 尝试根据传入的值创建一个新的数组对象，并强制类型转换为当前数组的类型
    arr = (PyArrayObject *)PyArray_FromAny(val, typecode,
                  0, 0, NPY_ARRAY_FORCECAST | PyArray_FORTRAN_IF(self), NULL);
    // 如果创建数组对象失败，则返回 -1
    if (arr == NULL) {
        return -1;
    }
    // 创建传入数组的迭代器对象
    arrit = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
    // 如果创建迭代器对象失败，则跳转到退出标签
    if (arrit == NULL) {
        goto exit;
    }
    // 创建自身数组的迭代器对象
    selfit = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    // 如果创建迭代器对象失败，则跳转到退出标签
    if (selfit == NULL) {
        goto exit;
    }
    // 如果传入数组的大小为 0，则直接将返回值设为 0 并跳转到退出标签
    if (arrit->size == 0) {
        retval = 0;
        goto exit;
    }
    // 判断是否需要交换字节顺序
    swap = PyArray_ISNOTSWAPPED(self) != PyArray_ISNOTSWAPPED(arr);
    // 获取当前数组类型描述符的复制和交换函数
    copyswap = PyDataType_GetArrFuncs(PyArray_DESCR(self))->copyswap;
    // 如果当前数组的数据类型需要引用计数检查，则执行以下循环
    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        // 在自身数组的迭代器上进行循环，释放每个元素的引用计数，并增加传入数组对应元素的引用计数
        while (selfit->index < selfit->size) {
            PyArray_Item_XDECREF(selfit->dataptr, PyArray_DESCR(self));
            PyArray_Item_INCREF(arrit->dataptr, PyArray_DESCR(arr));
            // 使用 memmove 函数复制传入数组的元素到自身数组中
            memmove(selfit->dataptr, arrit->dataptr, sizeof(PyObject **));
            // 如果需要交换字节顺序，则调用对应的交换函数
            if (swap) {
                copyswap(selfit->dataptr, NULL, swap, self);
            }
            // 更新迭代器的指针位置
            PyArray_ITER_NEXT(selfit);
            PyArray_ITER_NEXT(arrit);
            // 如果传入数组的迭代器达到末尾，则重置迭代器
            if (arrit->index == arrit->size) {
                PyArray_ITER_RESET(arrit);
            }
        }
        // 设置返回值为 0 并跳转到退出标签
        retval = 0;
        goto exit;
    }

    // 若当前数组的数据类型不需要引用计数检查，则执行以下循环
    while(selfit->index < selfit->size) {
        // 调用复制和可能的交换函数，将传入数组的元素复制到自身数组中
        copyswap(selfit->dataptr, arrit->dataptr, swap, self);
        // 更新迭代器的指针位置
        PyArray_ITER_NEXT(selfit);
        PyArray_ITER_NEXT(arrit);
        // 如果传入数组的迭代器达到末尾，则重置迭代器
        if (arrit->index == arrit->size) {
            PyArray_ITER_RESET(arrit);
        }
    }
    // 设置返回值为 0

 exit:
    // 释放迭代器对象的引用计数
    Py_XDECREF(selfit);
    Py_XDECREF(arrit);
    Py_XDECREF(arr);
    // 返回函数的执行结果
    return retval;
}

// 定义静态函数 `array_transpose_get`，用于获取数组的转置
static PyObject *
array_transpose_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    // 调用 NumPy 提供的函数 PyArray_Transpose 对数组进行转置操作
    return PyArray_Transpose(self, NULL);
}

// 定义静态函数 `array_matrix_transpose_get`，用于获取矩阵的转置
static PyObject *
array_matrix_transpose_get(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    // 调用 NumPy 提供的函数 PyArray_MatrixTranspose 对矩阵进行转置操作
    return PyArray_MatrixTranspose(self);
}

// 定义静态函数 `array_ptp`，处理 ptp 属性被移除的错误情况
static PyObject *
array_ptp(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    // 设置异常消息，说明 ptp 属性在 NumPy 2.0 版本中被移除，提供替代方法的建议
    PyErr_SetString(PyExc_AttributeError,
                    "`ptp` was removed from the ndarray class in NumPy 2.0. "
                    "Use np.ptp(arr, ...) instead.");
    // 返回空指针，表示操作失败
    return NULL;
}

// 定义静态函数 `array_newbyteorder`，处理 newbyteorder 属性被移除的错误情况
static PyObject *
array_newbyteorder(PyArrayObject *self, PyObject *args)
{
    // 设置异常消息，说明 newbyteorder 属性在 NumPy 2.0 版本中被移除，提供替代方法的建议
    PyErr_SetString(PyExc_AttributeError,
                    "`newbyteorder` was removed from the ndarray class "
                    "in NumPy 2.0. "
                    "Use `arr.view(arr.dtype.newbyteorder(order))` instead.");
    // 返回空指针，表示操作失败
    return NULL;
}

// 定义静态函数 `array_itemset`，处理 itemset 属性被移除的错误情况
static PyObject *
array_itemset(PyArrayObject *self, PyObject *args)
{
    // 设置异常消息，说明 itemset 属性在 NumPy 2.0 版本中被移除，提供替代方法的建议
    PyErr_SetString(PyExc_AttributeError,
                    "`itemset` was removed from the ndarray class in "
                    "NumPy 2.0. Use `arr[index] = value` instead.");
    // 返回空指针，表示操作失败
    return NULL;
}

// 定义静态函数 `array_device`，返回数组对象所在的设备类型
static PyObject *
array_device(PyArrayObject *self, void *NPY_UNUSED(ignored))
{
    // 返回一个 PyUnicode 对象，表示该数组位于 CPU 设备上
    return PyUnicode_FromString("cpu");
}
    {"itemset",
        (getter)array_itemset,
        NULL,
        NULL, NULL},

# 创建一个元组项，包含属性名称 "itemset" 和一个函数指针 (getter)array_itemset，
# 后面的三个元素都设置为 NULL，表示没有其他特定的设置。

    {"device",
        (getter)array_device,
        NULL,
        NULL, NULL},

# 创建一个元组项，包含属性名称 "device" 和一个函数指针 (getter)array_device，
# 后面的三个元素都设置为 NULL，表示没有其他特定的设置。

    {"__array_interface__",
        (getter)array_interface_get,
        NULL,
        NULL, NULL},

# 创建一个元组项，包含属性名称 "__array_interface__" 和一个函数指针 (getter)array_interface_get，
# 后面的三个元素都设置为 NULL，表示没有其他特定的设置。

    {"__array_struct__",
        (getter)array_struct_get,
        NULL,
        NULL, NULL},

# 创建一个元组项，包含属性名称 "__array_struct__" 和一个函数指针 (getter)array_struct_get，
# 后面的三个元素都设置为 NULL，表示没有其他特定的设置。

    {"__array_priority__",
        (getter)array_priority_get,
        NULL,
        NULL, NULL},

# 创建一个元组项，包含属性名称 "__array_priority__" 和一个函数指针 (getter)array_priority_get，
# 后面的三个元素都设置为 NULL，表示没有其他特定的设置。

    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */

# 创建一个 Sentinel（哨兵）项，所有的元素都为 NULL，用于标志元组的结束。
};

/****************** end of attribute get and set routines *******************/


注释：


};  
这是一个注释行，表示以下代码段是 JavaScript 或类似语言中的对象或结构的结束。
/****************** end of attribute get and set routines *******************/
这是一个注释行，标记了上面代码段的结束，说明之前的代码段包含了属性的获取和设置相关的例程。
```