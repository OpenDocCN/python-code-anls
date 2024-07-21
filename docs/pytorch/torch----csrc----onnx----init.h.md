# `.\pytorch\torch\csrc\onnx\init.h`

```
#pragma once


// 使用#pragma once确保头文件只被编译一次，避免多重包含的问题



#include <torch/csrc/utils/pybind.h>


// 包含torch库中用于Python绑定的头文件pybind.h，以便在C++中调用Python接口



namespace torch::onnx {


// 定义命名空间torch::onnx，用于包含所有与ONNX相关的功能和类型



void initONNXBindings(PyObject* module);


// 声明一个函数initONNXBindings，用于初始化ONNX相关的Python绑定
// 接受一个PyObject*类型的module参数，表示要初始化的Python模块对象



} // namespace torch::onnx


// 命名空间torch::onnx的结束标记
```