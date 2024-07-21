# `.\pytorch\torch\csrc\autograd\python_saved_variable_hooks.h`

```py
#pragma once
// 使用#pragma once指令确保头文件只被编译一次，避免重复包含

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
// 引入所需的头文件，用于定义和声明程序中所需的各种类型、函数和宏

namespace py = pybind11;
// 命名空间别名，简化pybind11命名空间的使用

namespace torch::autograd {

// 定义一个结构体PySavedVariableHooks，继承自SavedVariableHooks
struct PySavedVariableHooks : public SavedVariableHooks {
  PySavedVariableHooks(py::function& pack_hook, py::function& unpack_hook);
  // 构造函数声明，接受两个py::function类型参数作为包装和解包钩子函数

  void call_pack_hook(const at::Tensor& tensor) override;
  // 重写SavedVariableHooks的call_pack_hook方法，传入一个at::Tensor类型参数

  at::Tensor call_unpack_hook() override;
  // 重写SavedVariableHooks的call_unpack_hook方法，返回at::Tensor类型对象

  ~PySavedVariableHooks() override;
  // 虚析构函数声明，用于释放资源

 private:
  PyObject* pack_hook_;
  // 指向包装钩子函数的Python对象的指针

  PyObject* unpack_hook_;
  // 指向解包钩子函数的Python对象的指针

  PyObject* data_ = nullptr;
  // 指向数据的Python对象的指针，默认为空指针
};

// 定义一个结构体PyDefaultSavedVariableHooks
struct PyDefaultSavedVariableHooks {
  static void push_hooks(py::function& pack_hook, py::function& unpack_hook);
  // 声明静态方法push_hooks，接受两个py::function类型参数作为包装和解包钩子函数

  static void pop_hooks();
  // 声明静态方法pop_hooks，用于弹出钩子函数

  static std::unique_ptr<SavedVariableHooks> get_hooks();
  // 声明静态方法get_hooks，返回一个std::unique_ptr<SavedVariableHooks>对象
};

} // namespace torch::autograd
// 结束torch::autograd命名空间
```