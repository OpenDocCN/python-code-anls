# `.\numpy\numpy\_core\src\multiarray\textreading\readtext.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_READTEXT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_READTEXT_H_

NPY_NO_EXPORT PyObject *
_load_from_filelike(PyObject *mod,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames);
        // 声明一个不导出的函数_load_from_filelike，接受模块对象及其它参数

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_READTEXT_H_ */
        // 定义了一个头文件宏，用于防止多次包含该头文件
```