# `.\pytorch\torch\csrc\dynamo\init.h`

```py
#pragma once
// 使用 pragma once 指令，确保头文件只被编译一次，提高编译效率

// 引入 pybind11 库中的 complex.h 文件，提供复数类型支持
#include <pybind11/complex.h>

// 引入 torch 库中的 pybind.h 文件，用于 PyTorch Python 绑定
#include <torch/csrc/utils/pybind.h>

// 引入 Python.h 头文件，提供与 Python 解释器交互的功能
#include <Python.h>

// 声明 torch::dynamo 命名空间，用于定义动态绑定的相关函数
namespace torch::dynamo {
    
// 初始化动态绑定函数，接受一个 PyObject 指针作为参数
void initDynamoBindings(PyObject* torch);
}
```