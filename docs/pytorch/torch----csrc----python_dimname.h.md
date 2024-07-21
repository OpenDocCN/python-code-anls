# `.\pytorch\torch\csrc\python_dimname.h`

```
#pragma once
// 使用 #pragma once 防止头文件被多次包含
#include <ATen/Dimname.h>
// 包含 ATen 库中的 Dimname 头文件

#include <torch/csrc/python_headers.h>
// 包含 Torch C++ 前端与 Python 交互所需的头文件

// 声明函数 THPDimname_parse，接受一个 PyObject 指针参数，返回一个 at::Dimname 对象
at::Dimname THPDimname_parse(PyObject* obj);

// 声明函数 THPUtils_checkDimname，接受一个 PyObject 指针参数，返回一个布尔值，用于检查对象是否为 Dimname 类型
bool THPUtils_checkDimname(PyObject* obj);

// 声明函数 THPUtils_checkDimnameList，接受一个 PyObject 指针参数，返回一个布尔值，用于检查对象是否为 Dimname 列表类型
bool THPUtils_checkDimnameList(PyObject* obj);
```