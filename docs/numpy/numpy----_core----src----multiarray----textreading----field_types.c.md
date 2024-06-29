# `.\numpy\numpy\_core\src\multiarray\textreading\field_types.c`

```
/*
 * 包含必要的头文件来引入所需的类型和函数声明
 */
#include "field_types.h"
#include "conversions.h"
#include "str_to_int.h"

/*
 * 定义宏，指定使用的 NumPy API 版本，禁用已弃用的 API
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"
#include "alloc.h"

/*
 * 引入特定领域增长的自定义头文件
 */
#include "textreading/growth.h"

/*
 * 清除字段类型数组及其描述符的函数
 */
NPY_NO_EXPORT void
field_types_xclear(int num_field_types, field_type *ft) {
    assert(num_field_types >= 0);
    if (ft == NULL) {
        return;
    }
    for (int i = 0; i < num_field_types; i++) {
        Py_XDECREF(ft[i].descr);  // 释放 Python 对象的引用
        ft[i].descr = NULL;  // 将字段描述符指针设为 NULL
    }
    PyMem_Free(ft);  // 释放内存
}

/*
 * 获取从 UCS4 编码到指定 NumPy DType 的转换函数
 */
static set_from_ucs4_function *
get_from_ucs4_function(PyArray_Descr *descr)
{
    if (descr->type_num == NPY_BOOL) {
        return &npy_to_bool;  // 返回布尔型转换函数指针
    }
    else if (PyDataType_ISSIGNED(descr)) {
        switch (descr->elsize) {
            case 1:
                return &npy_to_int8;  // 返回 int8 转换函数指针
            case 2:
                return &npy_to_int16;  // 返回 int16 转换函数指针
            case 4:
                return &npy_to_int32;  // 返回 int32 转换函数指针
            case 8:
                return &npy_to_int64;  // 返回 int64 转换函数指针
            default:
                assert(0);  // 断言，如果不是预期的大小则终止程序
        }
    }
    else if (PyDataType_ISUNSIGNED(descr)) {
        switch (descr->elsize) {
            case 1:
                return &npy_to_uint8;  // 返回 uint8 转换函数指针
            case 2:
                return &npy_to_uint16;  // 返回 uint16 转换函数指针
            case 4:
                return &npy_to_uint32;  // 返回 uint32 转换函数指针
            case 8:
                return &npy_to_uint64;  // 返回 uint64 转换函数指针
            default:
                assert(0);  // 断言，如果不是预期的大小则终止程序
        }
    }
    else if (descr->type_num == NPY_FLOAT) {
        return &npy_to_float;  // 返回 float 转换函数指针
    }
    else if (descr->type_num == NPY_DOUBLE) {
        return &npy_to_double;  // 返回 double 转换函数指针
    }
    else if (descr->type_num == NPY_CFLOAT) {
        return &npy_to_cfloat;  // 返回复数浮点数转换函数指针
    }
    else if (descr->type_num == NPY_CDOUBLE) {
        return &npy_to_cdouble;  // 返回复数双精度转换函数指针
    }
    else if (descr->type_num == NPY_STRING) {
        return &npy_to_string;  // 返回字符串转换函数指针
    }
    else if (descr->type_num == NPY_UNICODE) {
        return &npy_to_unicode;  // 返回 Unicode 转换函数指针
    }
    return &npy_to_generic;  // 默认返回通用转换函数指针
}

/*
 * 递归增长字段类型数组的函数
 */
static npy_intp
field_type_grow_recursive(PyArray_Descr *descr,
        npy_intp num_field_types, field_type **ft, npy_intp *ft_size,
        npy_intp field_offset)
{
    # 检查描述符是否具有子数组
    if (PyDataType_HASSUBARRAY(descr)) {
        # 定义一个用于存储形状信息的结构体
        PyArray_Dims shape = {NULL, -1};

        # 尝试将子数组的形状转换为整数数组
        if (!(PyArray_IntpConverter(PyDataType_SUBARRAY(descr)->shape, &shape))) {
             # 如果形状转换失败，则设置错误消息并清理内存后返回-1
             PyErr_SetString(PyExc_ValueError, "invalid subarray shape");
             field_types_xclear(num_field_types, *ft);
             return -1;
        }
        
        # 计算子数组的总大小
        npy_intp size = PyArray_MultiplyList(shape.ptr, shape.len);
        # 释放形状对象占用的内存
        npy_free_cache_dim_obj(shape);
        
        # 递归地扩展字段类型数组，以处理子数组的基本类型
        for (npy_intp i = 0; i < size; i++) {
            num_field_types = field_type_grow_recursive(PyDataType_SUBARRAY(descr)->base,
                    num_field_types, ft, ft_size, field_offset);
            field_offset += PyDataType_SUBARRAY(descr)->base->elsize;
            # 如果递归过程中出现错误，则返回-1
            if (num_field_types < 0) {
                return -1;
            }
        }
        
        # 返回处理完的字段类型数量
        return num_field_types;
    }
    # 如果描述符具有字段信息
    else if (PyDataType_HASFIELDS(descr)) {
        # 获取描述符中字段的数量
        npy_int num_descr_fields = PyTuple_Size(PyDataType_NAMES(descr));
        # 如果获取字段数量失败，则清理内存并返回-1
        if (num_descr_fields < 0) {
            field_types_xclear(num_field_types, *ft);
            return -1;
        }
        
        # 遍历描述符中的每个字段
        for (npy_intp i = 0; i < num_descr_fields; i++) {
            # 获取字段的键值
            PyObject *key = PyTuple_GET_ITEM(PyDataType_NAMES(descr), i);
            # 获取字段的元组表示
            PyObject *tup = PyObject_GetItem(PyDataType_FIELDS(descr), key);
            # 如果获取元组失败，则清理内存并返回-1
            if (tup == NULL) {
                field_types_xclear(num_field_types, *ft);
                return -1;
            }
            
            PyArray_Descr *field_descr;
            PyObject *title;
            int offset;
            # 尝试解析字段的元组信息
            if (!PyArg_ParseTuple(tup, "Oi|O", &field_descr, &offset, &title)) {
                Py_DECREF(tup);
                field_types_xclear(num_field_types, *ft);
                return -1;
            }
            
            Py_DECREF(tup);
            
            # 递归地扩展字段类型数组，处理当前字段
            num_field_types = field_type_grow_recursive(
                    field_descr, num_field_types, ft, ft_size,
                    field_offset + offset);
            # 如果递归过程中出现错误，则返回-1
            if (num_field_types < 0) {
                return -1;
            }
        }
        
        # 返回处理完的字段类型数量
        return num_field_types;
    }
    
    # 如果字段类型数组的大小小于等于当前字段类型数量
    if (*ft_size <= num_field_types) {
        # 计算新的分配大小
        npy_intp alloc_size = grow_size_and_multiply(
                ft_size, 4, sizeof(field_type));
        # 如果计算分配大小出错，则清理内存并返回-1
        if (alloc_size < 0) {
            field_types_xclear(num_field_types, *ft);
            return -1;
        }
        
        # 尝试重新分配字段类型数组的内存
        field_type *new_ft = PyMem_Realloc(*ft, alloc_size);
        # 如果重新分配内存失败，则清理内存并返回-1
        if (new_ft == NULL) {
            field_types_xclear(num_field_types, *ft);
            return -1;
        }
        
        # 更新字段类型数组的指针
        *ft = new_ft;
    }
    
    # 增加描述符的引用计数
    Py_INCREF(descr);
    # 将描述符和相关函数设置添加到字段类型数组中
    (*ft)[num_field_types].descr = descr;
    (*ft)[num_field_types].set_from_ucs4 = get_from_ucs4_function(descr);
    (*ft)[num_field_types].structured_offset = field_offset;
    
    # 返回增加一个字段类型后的字段类型数量
    return num_field_types + 1;
}

/*
 * Prepare the "field_types" for the given dtypes/descriptors.  Currently,
 * we copy the itemsize, but the main thing is that we check for custom
 * converters.
 */
NPY_NO_EXPORT npy_intp
field_types_create(PyArray_Descr *descr, field_type **ft)
{
    // 如果输入的数据类型是子数组（subarray），则抛出类型错误异常
    if (PyDataType_SUBARRAY(descr) != NULL) {
        PyErr_SetString(PyExc_TypeError,
                "file reader does not support subarray dtypes.  You can"
                "put the dtype into a structured one using "
                "`np.dtype(('name', dtype))` to avoid this limitation.");
        return -1;
    }

    // 初始化初始 field_type 数组的大小为 4
    npy_intp ft_size = 4;
    // 分配内存以存储 field_type 数组
    *ft = PyMem_Malloc(ft_size * sizeof(field_type));
    // 检查内存分配是否成功
    if (*ft == NULL) {
        return -1;
    }
    // 递归地填充 field_type 数组，并返回填充的元素个数
    return field_type_grow_recursive(descr, 0, ft, &ft_size, 0);
}
```