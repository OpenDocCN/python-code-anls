# `.\numpy\numpy\_core\src\multiarray\item_selection.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_ITEM_SELECTION_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ITEM_SELECTION_H_

/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
// 声明一个函数 count_boolean_trues，用于计算原始布尔数组中 True 的数量
NPY_NO_EXPORT npy_intp
count_boolean_trues(int ndim, char *data, npy_intp const *ashape, npy_intp const *astrides);

/*
 * Gets a single item from the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 */
// 声明一个函数 PyArray_MultiIndexGetItem，从数组中获取单个元素，基于给定的多重索引数组
NPY_NO_EXPORT PyObject *
PyArray_MultiIndexGetItem(PyArrayObject *self, const npy_intp *multi_index);

/*
 * Sets a single item in the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 *
 * Returns 0 on success, -1 on failure.
 */
// 声明一个函数 PyArray_MultiIndexSetItem，向数组中设置单个元素，基于给定的多重索引数组
NPY_NO_EXPORT int
PyArray_MultiIndexSetItem(PyArrayObject *self, const npy_intp *multi_index,
                                                PyObject *obj);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ITEM_SELECTION_H_ */
```