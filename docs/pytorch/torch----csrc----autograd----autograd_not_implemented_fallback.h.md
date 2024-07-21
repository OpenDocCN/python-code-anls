# `.\pytorch\torch\csrc\autograd\autograd_not_implemented_fallback.h`

```py
#pragma once
// 预处理指令：确保本头文件在编译过程中只包含一次

#include <torch/library.h>
// 包含 Torch 库的头文件

namespace torch::autograd {

// 进入 torch::autograd 命名空间

// 默认的 DispatchKey::Autograd 回退函数，用于内置操作符
// 可以用于注册自定义操作符
TORCH_API torch::CppFunction autogradNotImplementedFallback();
// 声明 autogradNotImplementedFallback 函数原型

// 默认的 DispatchKey::AdInplaceOrView 回退函数，用于内置操作符
// 可以用于注册自定义操作符
TORCH_API torch::CppFunction autogradNotImplementedInplaceOrViewFallback();
// 声明 autogradNotImplementedInplaceOrViewFallback 函数原型

// 默认的 DispatchKey::Autograd 回退函数，用于其他所有操作符（即自定义操作符）
TORCH_API torch::CppFunction basicAutogradNotImplementedFallback();
// 声明 basicAutogradNotImplementedFallback 函数原型

enum class AutogradFallbackMode {
  Nothing, // 回退为重新调度
  Warn,    // 如果调用 backward，则发出警告
  Error,   // 如果调用 backward，则抛出错误
};

// 更改 "basicAutogradNotImplementedFallback" 的行为方式
// 在 Python 中对应为：
// - torch._C._set_autograd_fallback_mode(str) -> None
// - torch._C._get_autograd_fallback_mode() -> str
TORCH_API void setAutogradFallbackMode(AutogradFallbackMode mode);
// 声明 setAutogradFallbackMode 函数原型

TORCH_API AutogradFallbackMode getAutogradFallbackMode();
// 声明 getAutogradFallbackMode 函数原型

} // namespace torch::autograd
// 结束 torch::autograd 命名空间
```