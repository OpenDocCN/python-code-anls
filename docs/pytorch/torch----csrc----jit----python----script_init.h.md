# `.\pytorch\torch\csrc\jit\python\script_init.h`

```py
#pragma once


// 使用 #pragma once 指令，确保当前头文件只被编译一次，防止多重包含



#include <torch/csrc/jit/python/pybind.h>


// 包含头文件 <torch/csrc/jit/python/pybind.h>，引入相关声明和定义，以便后续使用其中的功能和类型



namespace torch::jit {


// 命名空间 torch::jit 的开始，用于将一组相关的符号封装在一起，以避免与其他代码发生命名冲突



void initJitScriptBindings(PyObject* module);


// 声明函数 initJitScriptBindings，该函数接受一个 PyObject 指针参数，用于初始化 JIT 脚本的绑定



} // namespace torch::jit


// 命名空间 torch::jit 的结束，表示命名空间作用域的结束
```