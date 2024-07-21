# `.\pytorch\torch\csrc\api\include\torch\python\init.h`

```py
#pragma once


// 使用 pragma once 指令确保头文件只被编译一次，避免多重包含问题



#include <torch/csrc/utils/python_stub.h>


// 包含 Torch C++ 前端的 Python 存根头文件，用于与 Python 的交互



namespace torch {
namespace python {


// 定义命名空间 torch::python，用于封装 Torch 的 Python 相关功能



/// Initializes Python bindings for the C++ frontend.
void init_bindings(PyObject* module);


// 声明函数 init_bindings，用于初始化 C++ 前端与 Python 的绑定
// 参数 module 是一个 PyObject 指针，表示要初始化的 Python 模块



} // namespace python
} // namespace torch


// 结束命名空间 python 和 torch 的定义
```