# `.\pytorch\torch\csrc\autograd\python_nn_functions.h`

```py
#pragma once
// 使用预处理指令 `#pragma once`，确保此头文件只被编译一次

namespace torch::autograd {
// 命名空间 `torch::autograd` 开始

void initNNFunctions(PyObject* module);
// 声明函数 `initNNFunctions`，该函数接受一个 PyObject 指针作为参数，无返回值

}
// 命名空间 `torch::autograd` 结束
```