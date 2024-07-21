# `.\pytorch\torch\csrc\jit\python\python_custom_class.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/utils/pybind.h>
// 包含 torch 库中的 Python 绑定工具的头文件

#include <torch/custom_class.h>
// 包含 torch 库中自定义类相关的头文件

namespace torch::jit {
// 命名空间 torch::jit，包含 Torch 的 JIT（Just-In-Time）功能相关的内容

void initPythonCustomClassBindings(PyObject* module);
// 声明一个函数 initPythonCustomClassBindings，用于初始化 Python 中的自定义类绑定

struct ScriptClass {
// 定义一个结构体 ScriptClass，用于表示 Torch 的脚本类

  ScriptClass(c10::StrongTypePtr class_type)
      : class_type_(std::move(class_type)) {}
  // 结构体构造函数，接受一个 c10::StrongTypePtr 类型的参数 class_type，并将其移动赋值给成员变量 class_type_

  py::object __call__(py::args args, py::kwargs kwargs);
  // 声明一个成员函数 __call__，用于在 Python 中调用此脚本类的实例

  c10::StrongTypePtr class_type_;
  // 成员变量，表示脚本类的类型
};

} // namespace torch::jit
// 命名空间结束，结束了对 torch::jit 命名空间的定义
```