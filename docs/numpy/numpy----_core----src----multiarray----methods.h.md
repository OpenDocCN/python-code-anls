# `.\numpy\numpy\_core\src\multiarray\methods.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_METHODS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_METHODS_H_

#include "npy_static_data.h"   // 包含静态数据头文件 npy_static_data.h
#include "npy_import.h"        // 包含导入头文件 npy_import.h

extern NPY_NO_EXPORT PyMethodDef array_methods[];   // 声明外部数组方法的 PyMethodDef 结构数组

/*
 * Pathlib support, takes a borrowed reference and returns a new one.
 * The new object may be the same as the old.
 */
static inline PyObject *
NpyPath_PathlikeToFspath(PyObject *file)
{
    // 检查 file 是否是 os_PathLike 实例，如果不是，则增加其引用计数并返回
    if (!PyObject_IsInstance(file, npy_static_pydata.os_PathLike)) {
        Py_INCREF(file);
        return file;
    }
    // 如果 file 是 os_PathLike 实例，则调用 os_fspath 函数处理并返回其结果
    return PyObject_CallFunctionObjArgs(npy_static_pydata.os_fspath,
                                        file, NULL);
}

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_METHODS_H_ */
```