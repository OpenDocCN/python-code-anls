# `.\pytorch\torch\csrc\TypeInfo.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

struct THPDTypeInfo {
  PyObject_HEAD at::ScalarType type;
};
// 定义 THPDTypeInfo 结构体，包含 PyObject_HEAD 和 at::ScalarType 类型

struct THPFInfo : THPDTypeInfo {};
// 定义 THPFInfo 结构体作为 THPDTypeInfo 的子类

struct THPIInfo : THPDTypeInfo {};
// 定义 THPIInfo 结构体作为 THPDTypeInfo 的子类

extern PyTypeObject THPFInfoType;
extern PyTypeObject THPIInfoType;
// 外部声明 THPFInfoType 和 THPIInfoType，它们是 PyTypeObject 类型的变量

inline bool THPFInfo_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPFInfoType;
}
// 定义 THPFInfo_Check 内联函数，用于检查给定对象是否属于 THPFInfo 类型

inline bool THPIInfo_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPIInfoType;
}
// 定义 THPIInfo_Check 内联函数，用于检查给定对象是否属于 THPIInfo 类型

void THPDTypeInfo_init(PyObject* module);
// 声明 THPDTypeInfo_init 函数，该函数用于初始化 THPDTypeInfo 结构体
```