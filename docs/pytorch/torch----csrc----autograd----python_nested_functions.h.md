# `.\pytorch\torch\csrc\autograd\python_nested_functions.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保本文件只被编译一次

namespace torch::autograd {
// 声明命名空间torch::autograd，用于封装autograd相关的代码

PyMethodDef* get_nested_functions_manual();
// 声明一个函数get_nested_functions_manual()，返回类型为PyMethodDef*

void initNestedFunctions(PyObject* module);
// 声明一个函数initNestedFunctions，接受一个PyObject指针作为参数

} // namespace torch::autograd
// 结束命名空间torch::autograd
```