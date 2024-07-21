# `.\pytorch\torch\csrc\jit\python\init.h`

```
#pragma once


// 使用 `#pragma once` 预处理指令，确保头文件在编译过程中只被包含一次



#include <torch/csrc/utils/pybind.h>


// 包含 torch 库的 pybind.h 头文件，用于支持 Python 绑定的工具函数和宏定义



namespace torch::jit {


// 声明了一个命名空间 torch::jit，用于包裹所有的 JIT 相关代码



void initJITBindings(PyObject* module);


// 声明了一个名为 initJITBindings 的函数，其参数为 PyObject* 类型，用于初始化 JIT 绑定



} // namespace torch::jit


// 结束了命名空间 torch::jit 的声明
```