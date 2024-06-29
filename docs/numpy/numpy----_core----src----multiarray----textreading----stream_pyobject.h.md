# `.\numpy\numpy\_core\src\multiarray\textreading\stream_pyobject.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_

// 定义了一个宏，用于确保 Py_ssize_t 类型在包含 Python.h 之前被定义
#define PY_SSIZE_T_CLEAN

// 包含 Python.h 头文件，这是 Python C API 的主要头文件
#include <Python.h>

// 包含自定义的 stream 头文件，用于处理流的操作
#include "textreading/stream.h"

// 声明一个不导出的函数，该函数将 Python 对象转换为 stream 对象，使用指定的编码
NPY_NO_EXPORT stream *
stream_python_file(PyObject *obj, const char *encoding);

// 声明一个不导出的函数，该函数将 Python 可迭代对象转换为 stream 对象，使用指定的编码
NPY_NO_EXPORT stream *
stream_python_iterable(PyObject *obj, const char *encoding);

// 结束对头文件的声明
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_ */
```