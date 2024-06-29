# `.\numpy\numpy\_core\src\multiarray\calculation.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_CALCULATION_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CALCULATION_H_

// 声明不导出的函数，用于计算数组中的最大值索引，支持指定轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_ArgMax(PyArrayObject* self, int axis, PyArrayObject *out);

// 声明不导出的函数，用于计算数组中的最大值索引，支持指定轴、输出对象和保持维度标志
NPY_NO_EXPORT PyObject*
_PyArray_ArgMaxWithKeepdims(PyArrayObject* self, int axis, PyArrayObject *out, int keepdims);

// 声明不导出的函数，用于计算数组中的最小值索引，支持指定轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_ArgMin(PyArrayObject* self, int axis, PyArrayObject *out);

// 声明不导出的函数，用于计算数组中的最小值索引，支持指定轴、输出对象和保持维度标志
NPY_NO_EXPORT PyObject*
_PyArray_ArgMinWithKeepdims(PyArrayObject* self, int axis, PyArrayObject *out, int keepdims);

// 声明不导出的函数，用于计算数组中的最大值，支持指定轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Max(PyArrayObject* self, int axis, PyArrayObject* out);

// 声明不导出的函数，用于计算数组中的最小值，支持指定轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Min(PyArrayObject* self, int axis, PyArrayObject* out);

// 声明不导出的函数，用于计算数组中的峰峰值（最大值与最小值之差），支持指定轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Ptp(PyArrayObject* self, int axis, PyArrayObject* out);

// 声明不导出的函数，用于计算数组中的均值，支持指定轴、返回类型和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Mean(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

// 声明不导出的函数，用于对数组进行四舍五入，支持指定小数位数和输出对象
NPY_NO_EXPORT PyObject *
PyArray_Round(PyArrayObject *a, int decimals, PyArrayObject *out);

// 声明不导出的函数，用于计算数组中的迹（对角线元素之和），支持指定偏移、轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Trace(PyArrayObject* self, int offset, int axis1, int axis2,
                int rtype, PyArrayObject* out);

// 声明不导出的函数，用于裁剪数组，将元素限制在指定范围内，支持最小值、最大值和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Clip(PyArrayObject* self, PyObject* min, PyObject* max, PyArrayObject *out);

// 声明不导出的函数，用于对数组进行共轭操作，支持输出对象
NPY_NO_EXPORT PyObject*
PyArray_Conjugate(PyArrayObject* self, PyArrayObject* out);

// 声明不导出的函数，用于对数组进行四舍五入，支持指定小数位数和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Round(PyArrayObject* self, int decimals, PyArrayObject* out);

// 声明不导出的函数，用于计算数组中的标准差，支持指定轴、返回类型、输出对象和方差标志
NPY_NO_EXPORT PyObject*
PyArray_Std(PyArrayObject* self, int axis, int rtype, PyArrayObject* out,
                int variance);

// 声明不导出的函数，用于计算数组中的标准差，支持指定轴、返回类型、输出对象、方差标志和数值
NPY_NO_EXPORT PyObject *
__New_PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
                  int variance, int num);

// 声明不导出的函数，用于计算数组中的总和，支持指定轴、返回类型和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Sum(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

// 声明不导出的函数，用于计算数组的累积和，支持指定轴、返回类型和输出对象
NPY_NO_EXPORT PyObject*
PyArray_CumSum(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

// 声明不导出的函数，用于计算数组中的乘积，支持指定轴、返回类型和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Prod(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

// 声明不导出的函数，用于计算数组的累积乘积，支持指定轴、返回类型和输出对象
NPY_NO_EXPORT PyObject*
PyArray_CumProd(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

// 声明不导出的函数，用于判断数组中所有元素是否都为真，支持指定轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_All(PyArrayObject* self, int axis, PyArrayObject* out);

// 声明不导出的函数，用于判断数组中是否有任一元素为真，支持指定轴和输出对象
NPY_NO_EXPORT PyObject*
PyArray_Any(PyArrayObject* self, int axis, PyArrayObject* out);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CALCULATION_H_ */
```