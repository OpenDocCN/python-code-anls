# `.\pytorch\torch\csrc\StorageMethods.h`

```py
#ifndef THP_STORAGE_METHODS_INC
#define THP_STORAGE_METHODS_INC

// 如果 THP_STORAGE_METHODS_INC 宏没有定义，则开始定义该宏，防止头文件重复包含


#include <Python.h>

// 包含 Python.h 头文件，这是 CPython 的 C API 头文件，提供了与 Python 解释器交互的函数和数据结构定义


PyMethodDef* THPStorage_getMethods();

// 声明一个函数原型 THPStorage_getMethods()，该函数返回一个 PyMethodDef* 类型的指针，用于定义 Python 扩展模块中的方法


#endif

// 结束 THP_STORAGE_METHODS_INC 宏的定义部分
```