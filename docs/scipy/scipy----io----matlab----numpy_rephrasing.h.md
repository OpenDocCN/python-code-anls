# `D:\src\scipysrc\scipy\scipy\io\matlab\numpy_rephrasing.h`

```
#include <numpy/arrayobject.h>
# 包含 numpy 数组对象的头文件

#define PyArray_Set_BASE(arr, obj) PyArray_SetBaseObject(arr, obj)
# 定义宏 PyArray_Set_BASE，用于设置 numpy 数组对象的基础对象

#define PyArray_PyANewFromDescr(descr, nd, dims, data, parent)                 \
        PyArray_NewFromDescr(&PyArray_Type, descr, nd, dims,                  \
                             NULL, data, 0, parent)
# 定义宏 PyArray_PyANewFromDescr，用于创建一个新的 numpy 数组对象，使用给定的描述符、维度、数据和父对象
```