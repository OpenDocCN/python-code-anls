# `.\numpy\numpy\_core\src\multiarray\datetime_strings.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_

/*
 * This is the Python-exposed datetime_as_string function.
 */
# 定义一个条件编译指令，防止重复包含同一头文件
NPY_NO_EXPORT PyObject *
# 定义函数 array_datetime_as_string，接受三个参数：self（未使用）、args 和 kwds
array_datetime_as_string(PyObject *NPY_UNUSED(self), PyObject *args,
                                PyObject *kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_ */
```