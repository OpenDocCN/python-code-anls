# `.\numpy\numpy\_core\src\multiarray\descriptor.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_

/*
 * In some API calls we wish to allow users to pass a DType class or a
 * dtype instances with different meanings.
 * This struct is mainly used for the argument parsing in
 * `PyArray_DTypeOrDescrConverter`.
 */
typedef struct {
    PyArray_DTypeMeta *dtype;   // 指向 DType 类的指针
    PyArray_Descr *descr;       // 指向描述符实例的指针
} npy_dtype_info;

// 可选的数据类型或描述符转换函数声明
NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterOptional(PyObject *, npy_dtype_info *dt_info);

// 必需的数据类型或描述符转换函数声明
NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterRequired(PyObject *, npy_dtype_info *dt_info);

// 提取数据类型和描述符的函数声明
NPY_NO_EXPORT int
PyArray_ExtractDTypeAndDescriptor(PyArray_Descr *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType);

// 返回数组描述符协议类型字符串的函数声明
NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(
        PyArray_Descr *, void *);

// 返回数组描述符协议描述符的函数声明
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(
        PyArray_Descr *self, void *);

/*
 * offset:    A starting offset.
 * alignment: A power-of-two alignment.
 *
 * This macro returns the smallest value >= 'offset'
 * that is divisible by 'alignment'. Because 'alignment'
 * is a power of two and integers are twos-complement,
 * it is possible to use some simple bit-fiddling to do this.
 */
// 计算下一个对齐偏移量的宏定义
#define NPY_NEXT_ALIGNED_OFFSET(offset, alignment) \
                (((offset) + (alignment) - 1) & (-(alignment)))

// 设置类型字典的函数声明
NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);

// 尝试从数据类型属性转换为数组描述符的函数声明
NPY_NO_EXPORT PyArray_Descr *
_arraydescr_try_convert_from_dtype_attr(PyObject *obj);

// 检查数据类型是否采用简单的非对齐结构布局的函数声明
NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype);

/*
 * Filter the fields of a dtype to only those in the list of strings, ind.
 *
 * No type checking is performed on the input.
 *
 * Raises:
 *   ValueError - if a field is repeated
 *   KeyError - if an invalid field name (or any field title) is used
 */
// 将数据类型的字段过滤为指定列表中的字段的函数声明
NPY_NO_EXPORT PyArray_Descr *
arraydescr_field_subset_view(_PyArray_LegacyDescr *self, PyObject *ind);

extern NPY_NO_EXPORT char const *_datetime_strings[];

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_ */
```