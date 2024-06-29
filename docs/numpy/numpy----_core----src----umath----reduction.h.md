# `.\numpy\numpy\_core\src\umath\reduction.h`

```
#ifndef _NPY_PRIVATE__REDUCTION_H_
#define _NPY_PRIVATE__REDUCTION_H_

/************************************************************
 * Typedefs used by PyArray_ReduceWrapper, new in 1.7.
 ************************************************************/

/*
 * This typedef defines a function pointer type for assigning a reduction identity
 * to the result array before performing the reduction computation. The `data`
 * parameter is passed through from PyArray_ReduceWrapper.
 *
 * This function should return -1 on failure or 0 on success.
 */
typedef int (PyArray_AssignReduceIdentityFunc)(PyArrayObject *result,
                                               void *data);

/*
 * This typedef defines a function pointer type for the inner definition of the reduce loop.
 * It was intended for customization of the reduce loop at a lower level per ufunc.
 * This aspect of the API might be deprecated or restructured in future versions.
 */
typedef int (PyArray_ReduceLoopFunc)(PyArrayMethod_Context *context,
                                     PyArrayMethod_StridedLoop *strided_loop,
                                     NpyAuxData *auxdata,
                                     NpyIter *iter,
                                     char **dataptrs,
                                     npy_intp const *strides,
                                     npy_intp const *countptr,
                                     NpyIter_IterNextFunc *iternext,
                                     int needs_api,
                                     npy_intp skip_first_count);

#endif  // _NPY_PRIVATE__REDUCTION_H_
/*
 * This function executes all the standard NumPy reduction function
 * boilerplate code, just calling the appropriate inner loop function where
 * necessary.
 *
 * operand     : The array to be reduced.
 * out         : NULL, or the array into which to place the result.
 * wheremask   : NOT YET SUPPORTED, but this parameter is placed here
 *               so that support can be added in the future without breaking
 *               API compatibility. Pass in NULL.
 * operand_dtype : The dtype the inner loop expects for the operand.
 * result_dtype : The dtype the inner loop expects for the result.
 * casting     : The casting rule to apply to the operands.
 * axis_flags  : Flags indicating the reduction axes of 'operand'.
 * reorderable : If True, the reduction being done is reorderable, which
 *               means specifying multiple axes of reduction at once is ok,
 *               and the reduction code may calculate the reduction in an
 *               arbitrary order. The calculation may be reordered because
 *               of cache behavior or multithreading requirements.
 * keepdims    : If true, leaves the reduction dimensions in the result
 *               with size one.
 * identity    : If Py_None, PyArray_CopyInitialReduceValues is used, otherwise
 *               this value is used to initialize the result to
 *               the reduction's unit.
 * loop        : The loop which does the reduction.
 * data        : Data which is passed to the inner loop.
 * buffersize  : Buffer size for the iterator. For the default, pass in 0.
 * funcname    : The name of the reduction function, for error messages.
 * errormask   : forwarded from _get_bufsize_errmask
 */
NPY_NO_EXPORT PyArrayObject *
PyUFunc_ReduceWrapper(PyArrayMethod_Context *context,
        PyArrayObject *operand, PyArrayObject *out, PyArrayObject *wheremask,
        npy_bool *axis_flags, int keepdims,
        PyObject *initial, PyArray_ReduceLoopFunc *loop,
        npy_intp buffersize, const char *funcname, int errormask)
{
    // 实现所有标准 NumPy 函数的归约逻辑，根据需要调用适当的内部循环函数

    // 返回归约结果的数组对象
    NPY_NO_EXPORT PyArrayObject * 
    // 执行归约操作的上下文
    context,

    // 要归约的数组
    * operand, 
    
    // 存放结果的数组，如果为 NULL 则在函数内部创建
    * out, 
    
    // 状态掩码，目前不支持，为了未来兼容性保留
    * wheremask, 
    
    // 操作数的数据类型，内部循环期望的类型
    * operand_dtype, 
    
    // 结果的数据类型，内部循环期望的类型
    * result_dtype, 
    
    // 应用于操作数的类型转换规则
    * casting, 
    
    // 标志指示 'operand' 的归约轴
    * axis_flags, 
    
    // 如果为 True，保留结果中尺寸为一的归约维度
    keepdims, 
    
    // 如果为 Py_None，使用 PyArray_CopyInitialReduceValues 初始化结果
    * identity, 
    
    // 执行归约的循环
    * loop, 
    
    // 传递给内部循环的数据
    * data, 
    
    // 迭代器的缓冲区大小，默认为 0
    buffersize, 
    
    // 归约函数的名称，用于错误消息
    * funcname, 
    
    // 从 _get_bufsize_errmask 转发的错误掩码
    * errormask);
}
```