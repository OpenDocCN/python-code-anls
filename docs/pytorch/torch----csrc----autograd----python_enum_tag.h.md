# `.\pytorch\torch\csrc\autograd\python_enum_tag.h`

```
#pragma once


// 使用指令 "#pragma once" 确保此头文件只被编译一次，避免重复包含

#include <torch/csrc/python_headers.h>


// 包含名为 "torch/csrc/python_headers.h" 的头文件，该头文件可能定义了与 Python 相关的宏、函数或数据结构

namespace torch::autograd {
void initEnumTag(PyObject* module);
}


// 在 torch::autograd 命名空间中声明一个函数 initEnumTag，该函数接受一个名为 module 的 PyObject 指针作为参数
```