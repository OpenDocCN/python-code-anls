# `.\pytorch\torch\csrc\jit\testing\hooks_for_testing.h`

```py
#pragma once
#include <torch/csrc/Export.h>  // 包含 Torch 导出相关的头文件
#include <torch/csrc/jit/api/compilation_unit.h>  // 包含 Torch JIT 编译单元的 API 头文件
#include <functional>  // 包含 C++ 标准库中的函数对象相关头文件
#include <memory>  // 包含 C++ 标准库中的智能指针相关头文件

namespace torch {
namespace jit {

// 定义 Module 结构体
struct Module;

// 定义 ModuleHook 类型，用于表示对模块的钩子函数
using ModuleHook = std::function<void(Module module)>;

// 定义 FunctionHook 类型，用于表示对函数的钩子函数
using FunctionHook = std::function<void(StrongFunctionPtr function)>;

// 声明 didFinishEmitModule 函数，用于在模块发射完成后调用
TORCH_API void didFinishEmitModule(Module module);

// 声明 didFinishEmitFunction 函数，用于在函数发射完成后调用
TORCH_API void didFinishEmitFunction(StrongFunctionPtr defined);

// 声明 setEmitHooks 函数，用于设置模块和函数的发射钩子
TORCH_API void setEmitHooks(ModuleHook for_module, FunctionHook for_fn);

// 声明 getEmitHooks 函数，用于获取当前设置的模块和函数的发射钩子
TORCH_API std::pair<ModuleHook, FunctionHook> getEmitHooks();

} // namespace jit
} // namespace torch
```