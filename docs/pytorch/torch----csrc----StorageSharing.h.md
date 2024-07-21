# `.\pytorch\torch\csrc\StorageSharing.h`

```
#ifndef THP_STORAGE_SHARING_INC
#define THP_STORAGE_SHARING_INC



// 如果 THP_STORAGE_SHARING_INC 宏未定义，则定义 THP_STORAGE_SHARING_INC 宏
#include <Python.h>
// 包含 Python.h 头文件，以便使用 Python C API

PyMethodDef* THPStorage_getSharingMethods();
// 声明一个名为 THPStorage_getSharingMethods 的函数，返回类型为 PyMethodDef*



#endif



// 结束 THP_STORAGE_SHARING_INC 宏的定义
```