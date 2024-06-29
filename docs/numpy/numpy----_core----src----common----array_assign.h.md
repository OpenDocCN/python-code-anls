# `.\numpy\numpy\_core\src\common\array_assign.h`

```py
#ifndef NUMPY_CORE_SRC_COMMON_ARRAY_ASSIGN_H_
#define NUMPY_CORE_SRC_COMMON_ARRAY_ASSIGN_H_

/*
 * An array assignment function for copying arrays, treating the
 * arrays as flat according to their respective ordering rules.
 * This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array.
 * dst_order: The rule for how 'dst' is to be made flat.
 * src: The source array.
 * src_order: The rule for how 'src' is to be made flat.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 *
 * Returns 0 on success, -1 on failure.
 */

/* Not yet implemented
NPY_NO_EXPORT int
PyArray_AssignArrayAsFlat(PyArrayObject *dst, NPY_ORDER dst_order,
                  PyArrayObject *src, NPY_ORDER src_order,
                  NPY_CASTING casting,
                  npy_bool preservena, npy_bool *preservewhichna);
*/

/*
 * Declares a function to assign the content of 'src' array to 'dst' array,
 * handling a possible 'wheremask' array and casting rules.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignArray(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting);

/*
 * Assigns a raw scalar value to every element of 'dst' array,
 * considering casting rules and optional 'wheremask'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignRawScalar(PyArrayObject *dst,
                        PyArray_Descr *src_dtype, char *src_data,
                        PyArrayObject *wheremask,
                        NPY_CASTING casting);

/******** LOW-LEVEL SCALAR TO ARRAY ASSIGNMENT ********/

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_scalar(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data);

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp const *wheremask_strides);

/******** LOW-LEVEL ARRAY MANIPULATION HELPERS ********/

/*
 * Internal detail of how much to buffer during array assignments which
 * need it. This is for more complex NA masking operations where masks
 * need to be inverted or combined together.
 */
#define NPY_ARRAY_ASSIGN_BUFFERSIZE 8192

/*
 * Broadcasts strides to match the given dimensions. Can be used,
 * for instance, to set up a raw iteration.
 *
 * 'strides_name' is used to produce an error message if the strides
 * cannot be broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
/*
 * 计算广播操作后的数组的步幅。
 * ndim: 数组的维度
 * shape: 数组的形状
 * strides_ndim: 步幅数组的维度
 * strides_shape: 步幅数组的形状
 * strides: 步幅数组的值
 * strides_name: 步幅数组的名称
 * out_strides: 输出的步幅数组
 */
broadcast_strides(int ndim, npy_intp const *shape,
                int strides_ndim, npy_intp const *strides_shape, npy_intp const *strides,
                char const *strides_name,
                npy_intp *out_strides);

/*
 * 检查一个数据指针加上一组步幅是否指向所有元素都按照给定的对齐方式对齐的原始数组。
 * 如果数据对齐到给定的对齐方式返回 1，否则返回 0。
 * alignment 应为二的幂，或者可以是特殊值 0 表示不能对齐，此时总是返回 0（false）。
 */
NPY_NO_EXPORT int
raw_array_is_aligned(int ndim, npy_intp const *shape,
                     char *data, npy_intp const *strides, int alignment);

/*
 * 检查数组是否按照其数据类型的“真实对齐方式”对齐。
 */
NPY_NO_EXPORT int
IsAligned(PyArrayObject *ap);

/*
 * 检查数组是否按照其数据类型的“无符号整数对齐方式”对齐。
 */
NPY_NO_EXPORT int
IsUintAligned(PyArrayObject *ap);

/* 如果两个数组有重叠的数据返回 1，否则返回 0 */
NPY_NO_EXPORT int
arrays_overlap(PyArrayObject *arr1, PyArrayObject *arr2);


#endif  /* NUMPY_CORE_SRC_COMMON_ARRAY_ASSIGN_H_ */
```