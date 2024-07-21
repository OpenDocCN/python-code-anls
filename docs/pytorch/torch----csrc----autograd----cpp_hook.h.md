# `.\pytorch\torch\csrc\autograd\cpp_hook.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/autograd/function_hook.h>
// 包含 Torch 自动求导功能钩子函数的头文件

#include <functional>
// 包含 C++ 标准库中的函数对象的头文件

#include <memory>
// 包含 C++ 标准库中的智能指针的头文件

namespace torch::autograd {
// 声明 torch::autograd 命名空间

using hooks_list =
    std::vector<std::function<at::TensorBase(const at::TensorBase&)>>;
// 定义 hooks_list 类型为 std::vector，存储的元素是接受 const at::TensorBase& 参数并返回 at::TensorBase 的函数对象

struct CppFunctionTensorPreHook : public FunctionPreHook {
  // 定义结构体 CppFunctionTensorPreHook，继承自 FunctionPreHook 类

  CppFunctionTensorPreHook(std::shared_ptr<hooks_list> hooks, size_t value_idx);
  // 构造函数声明，接受一个 shared_ptr 指向 hooks_list 和一个 size_t 类型的 value_idx 参数

  variable_list operator()(const variable_list& values) override;
  // 重载函数调用运算符，接受一个 variable_list 类型的参数并返回 variable_list 类型的结果

  std::shared_ptr<hooks_list> hooks_;
  // 成员变量，存储 hooks_list 类型的 shared_ptr 智能指针

  size_t value_idx_;
  // 成员变量，存储 size_t 类型的 value_idx
};

struct CppFunctionSingleTensorPreHook : public FunctionPreHook {
  // 定义结构体 CppFunctionSingleTensorPreHook，继承自 FunctionPreHook 类

  CppFunctionSingleTensorPreHook(
      std::function<at::TensorBase(const at::TensorBase&)> hook,
      size_t value_idx);
  // 构造函数声明，接受一个接受 const at::TensorBase& 参数并返回 at::TensorBase 的函数对象 hook 和一个 size_t 类型的 value_idx 参数

  variable_list operator()(const variable_list& values) override;
  // 重载函数调用运算符，接受一个 variable_list 类型的参数并返回 variable_list 类型的结果

  std::function<at::TensorBase(const at::TensorBase&)> hook_;
  // 成员变量，存储接受 const at::TensorBase& 参数并返回 at::TensorBase 的函数对象

  size_t value_idx_;
  // 成员变量，存储 size_t 类型的 value_idx
};

} // namespace torch::autograd
// 结束 torch::autograd 命名空间声明
```