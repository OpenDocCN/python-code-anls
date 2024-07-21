# `.\pytorch\torch\csrc\monitor\python_init.h`

```
#pragma once
// 使用 pragma once 指令确保头文件只被编译一次，防止重复包含

#include <torch/csrc/utils/pybind.h>
// 包含 PyTorch 的 pybind.h 头文件，用于 PyTorch C++ 和 Python 之间的绑定

namespace torch {
namespace monitor {

void initMonitorBindings(PyObject* module);
// 声明一个函数 initMonitorBindings，该函数用于初始化监控模块的绑定，接受一个 PyObject* 类型的参数 module

}
} // namespace torch
// 命名空间 torch 下的 monitor 命名空间，用于放置与监控相关的函数和类声明
```