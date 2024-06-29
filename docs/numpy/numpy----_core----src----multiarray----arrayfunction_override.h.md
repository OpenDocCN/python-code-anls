# `.\numpy\numpy\_core\src\multiarray\arrayfunction_override.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_
// 如果没有定义 NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_，则执行以下代码
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_

// 声明 PyArrayFunctionDispatcher_Type，作为 extern 外部变量
extern NPY_NO_EXPORT PyTypeObject PyArrayFunctionDispatcher_Type;

// 定义 array__get_implementing_args 函数，返回 PyObject 指针
NPY_NO_EXPORT PyObject *
array__get_implementing_args(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args);

// 定义 array_implement_c_array_function_creation 函数，返回 PyObject 指针
NPY_NO_EXPORT PyObject *
array_implement_c_array_function_creation(
        const char *function_name, PyObject *like,
        PyObject *args, PyObject *kwargs,
        PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames);

// 定义 array_function_method_impl 函数，返回 PyObject 指针
NPY_NO_EXPORT PyObject *
array_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs);

// 结束条件，结束 ifndef 指令
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_ */
```