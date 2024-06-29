# `.\numpy\numpy\_core\src\multiarray\shape.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_

#include "conversion_utils.h"

/*
 * 创建一个按照 NpyIter 对象的 KEEPORDER 行为进行排序的步幅排列。
 * 因为这是基于多个输入步幅进行操作，np.ndarray 结构体 npy_stride_sort_item 的 'stride' 成员无用，
 * 我们简单地对索引列表进行 argsort。
 *
 * 调用者应该已经验证每个数组在数组列表中的 'ndim' 是否匹配。
 */
NPY_NO_EXPORT void
PyArray_CreateMultiSortedStridePerm(int narrays, PyArrayObject **arrays,
                        int ndim, int *out_strideperm);

/*
 * 类似于 PyArray_Squeeze，但允许调用者选择要挤压的大小为一的维度的子集。
 */
NPY_NO_EXPORT PyObject *
PyArray_SqueezeSelected(PyArrayObject *self, npy_bool *axis_flags);

/*
 * 返回矩阵的转置（交换最后两个维度）。
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixTranspose(PyArrayObject *ap);

/*
 * 使用复制模式参数 _copy 进行数组重塑。
 */
NPY_NO_EXPORT PyObject *
_reshape_with_copy_arg(PyArrayObject *array, PyArray_Dims *newdims,
                       NPY_ORDER order, NPY_COPYMODE copy);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_ */


这些注释解释了每个函数的目的和关键参数。每个注释都遵循了代码本身的结构和逻辑顺序。
```