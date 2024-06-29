# `.\numpy\numpy\_core\src\multiarray\arraywrap.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_

// 定义结构体 NpyUFuncContext，用于保存 ufunc 对象及其输入输出参数
typedef struct {
    PyObject *ufunc;    // ufunc 对象
    PyObject *in;       // 输入对象
    PyObject *out;      // 输出对象
    int out_i;          // 输出索引
} NpyUFuncContext;

// 不导出的函数声明：将 obj 对象应用于 wrap 包装并返回结果
NPY_NO_EXPORT PyObject *
npy_apply_wrap(
        PyObject *obj, PyObject *original_out,
        PyObject *wrap, PyObject *wrap_type,
        NpyUFuncContext *context, npy_bool return_scalar, npy_bool force_wrap);

// 不导出的函数声明：将 arr_of_subclass 数组对象应用于 towrap 包装并返回结果
NPY_NO_EXPORT PyObject *
npy_apply_wrap_simple(PyArrayObject *arr_of_subclass, PyArrayObject *towrap);

// 不导出的函数声明：寻找数组的包装器并设置输出包装器和包装类型
NPY_NO_EXPORT int
npy_find_array_wrap(
        int nin, PyObject *const *inputs,
        PyObject **out_wrap, PyObject **out_wrap_type);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_ */


注释：
- `#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_`：如果未定义 `NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_`，则执行下面的内容，避免头文件被多次包含。
- `#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_`：定义 `NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_` 宏，确保头文件只被包含一次。
- `typedef struct { ... } NpyUFuncContext;`：定义了一个结构体 `NpyUFuncContext`，用于存储 ufunc 对象及其输入输出参数。
- `NPY_NO_EXPORT PyObject * npy_apply_wrap(...)`：不导出的函数声明，将给定的对象应用于包装器并返回结果。
- `NPY_NO_EXPORT PyObject * npy_apply_wrap_simple(...)`：不导出的函数声明，对给定的数组对象应用简单的包装器并返回结果。
- `NPY_NO_EXPORT int npy_find_array_wrap(...)`：不导出的函数声明，寻找数组的包装器并设置输出包装器和包装类型。
- `#endif /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_ */`：结束条件编译指令块，确保头文件内容完整且不会重复包含。
```