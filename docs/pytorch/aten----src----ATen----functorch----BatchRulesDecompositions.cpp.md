# `.\pytorch\aten\src\ATen\functorch\BatchRulesDecompositions.cpp`

```
// 包含标准头文件，这些文件通常用于C++编程
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/PlumbingHelper.h>

// 定义了 at::functorch 命名空间，用于存放 FunctTorch 相关的功能和实现
namespace at::functorch {

// 定义了一个宏 OP_DECOMPOSE，用于简化操作符函数的注册
#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
// 定义了另一个宏 OP_DECOMPOSE2，用于处理带有两个参数的操作符函数的注册
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

// 实现 Torch 库的 ATen 部分的函数注册，使用了 FuncTorchVmapMode 模式
TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  // 分别注册了四个操作符函数，其实现通过 native 命名空间调用对应的 ATen 函数
  OP_DECOMPOSE(alpha_dropout_);
  OP_DECOMPOSE(dropout_);
  OP_DECOMPOSE(feature_alpha_dropout_);
  OP_DECOMPOSE(feature_dropout_);
}

// 定义了一个静态函数 unsupportedData，用于处理在 vmap 转换下直接使用 `.data` 的情况
static void unsupportedData(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // 引发异常，提示在 vmap 转换下不允许直接使用 `.data` 进行变异操作
    TORCH_CHECK(false, "mutating directly with `.data` under vmap transform is not allowed.");
}

} // namespace at::functorch
```