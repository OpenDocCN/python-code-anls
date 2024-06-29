# `.\numpy\numpy\_core\src\multiarray\mapping.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_

// 外部声明，用于将数组对象视为映射对象的方法集合
extern NPY_NO_EXPORT PyMappingMethods array_as_mapping;

/*
 * 用于存储高级（也称为fancy）索引所需信息的对象。
 * 不是公共对象，因此原则上不必是Python对象。
 */
} PyArrayMapIterObject;

// 外部声明，定义了PyArrayMapIterObject的类型对象
extern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;

/*
 * 结构体，用于解析索引。
 * 即整数索引只需解析一次，后续需要验证切片和数组，
 * 对于省略号，需要确定其代表的切片数量。
 */
typedef struct {
    /*
     * 索引对象：切片、数组或NULL。拥有一个引用。
     */
    PyObject *object;
    /*
     * 整数索引的值，省略号代表的切片数量，
     * 如果输入是整数数组则为-1，布尔数组的原始大小。
     */
    npy_intp value;
    /* 索引类型，参见mapping.c中的常量 */
    int type;
} npy_index_info;


// 下面是一系列用于数组对象的非公开（NPY_NO_EXPORT）函数声明

// 返回数组对象的长度
NPY_NO_EXPORT Py_ssize_t
array_length(PyArrayObject *self);

// 返回数组对象的第i个元素作为新的数组对象
NPY_NO_EXPORT PyObject *
array_item_asarray(PyArrayObject *self, npy_intp i);

// 返回数组对象的第i个元素作为标量对象
NPY_NO_EXPORT PyObject *
array_item_asscalar(PyArrayObject *self, npy_intp i);

// 返回数组对象的第i个元素
NPY_NO_EXPORT PyObject *
array_item(PyArrayObject *self, Py_ssize_t i);

// 返回数组对象的索引操作结果作为新的数组对象
NPY_NO_EXPORT PyObject *
array_subscript_asarray(PyArrayObject *self, PyObject *op);

// 返回数组对象的索引操作结果
NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op);

// 对数组对象的第i个元素进行赋值操作
NPY_NO_EXPORT int
array_assign_item(PyArrayObject *self, Py_ssize_t i, PyObject *v);

/*
 * 映射调用的原型声明 —— 不属于C-API的一部分，
 * 因为只有作为getitem调用的一部分才有用。
 */

// 重置PyArrayMapIterObject对象，准备进行映射迭代
NPY_NO_EXPORT int
PyArray_MapIterReset(PyArrayMapIterObject *mit);

// 将PyArrayMapIterObject对象向前移动一步
NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit);

// 检查PyArrayMapIterObject对象的索引是否有效
NPY_NO_EXPORT int
PyArray_MapIterCheckIndices(PyArrayMapIterObject *mit);

// 根据PyArrayMapIterObject对象的轴交换映射
NPY_NO_EXPORT void
PyArray_MapIterSwapAxes(PyArrayMapIterObject *mit, PyArrayObject **ret, int getmap);

// 创建新的PyArrayMapIterObject对象，用于处理高级索引
NPY_NO_EXPORT PyObject*
PyArray_MapIterNew(npy_index_info *indices , int index_num, int index_type,
                   int ndim, int fancy_ndim,
                   PyArrayObject *arr, PyArrayObject *subspace,
                   npy_uint32 subspace_iter_flags, npy_uint32 subspace_flags,
                   npy_uint32 extra_op_flags, PyArrayObject *extra_op,
                   PyArray_Descr *extra_op_dtype);

// 如果重叠，则在数组a和索引之间复制部分数据
NPY_NO_EXPORT PyObject *
PyArray_MapIterArrayCopyIfOverlap(PyArrayObject * a, PyObject * index,
                                  int copy_if_overlap, PyArrayObject *extra_op);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_ */
```