# `.\pytorch\aten\src\ATen\functorch\BatchRulesDynamic.cpp`

```py
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含批处理规则的辅助函数和批处理的回退策略
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
// 包含调度器的头文件，用于分发操作
#include <ATen/core/dispatch/Dispatcher.h>
// 包含 C10 库的元编程实用工具
#include <c10/util/Metaprogramming.h>

// 该文件包含返回动态形状张量的操作的批处理规则。
// 通常我们不支持这些操作的 vmap，因此我们会对它们引发错误。

namespace at::functorch {

// 定义未支持的动态形状操作时的错误处理函数
namespace {
void unsupportedDynamicOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false, "vmap: We do not support batching operators that can output dynamic shape. ",
        "Attempted to vmap over ", op.schema().operator_name(), ". ",
        "Please voice your support in https://github.com/pytorch/functorch/issues/256");
}

// 宏定义，用于注册未支持的动态操作的错误处理函数
#define UNSUPPORTED_DYNAMIC(op) \
    m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>());

// 以下是一些特定本地标量稠密操作的错误处理函数
// 当尝试对这些操作使用 vmap 时会引发错误
void unsupportedLocalScalarDense(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap: It looks like you're either (1) calling .item() on a Tensor or ",
        "(2) attempting to use a Tensor in some data-dependent control flow or ",
        "(3) encountering this error in PyTorch internals. ",
        "For (1): we don't support vmap over calling .item() on a Tensor, please try to ",
        "rewrite what you're doing with other operations. ",
        "For (2): If you're doing some ",
        "control flow instead, we don't support that yet, please shout over at ",
        "https://github.com/pytorch/functorch/issues/257 . ",
        "For (3): please file an issue.");
}

// 错误处理函数，当尝试在张量上调用 .item() 时引发错误
void unsupportedItem(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap: It looks like you're calling .item() on a Tensor. ",
        "We don't support vmap over calling .item() on a Tensor, please try to ",
        "rewrite what you're doing with other operations. If error is occurring ",
        "somewhere inside PyTorch internals, please file a bug report.");
}

// 错误处理函数，当尝试在控制流中使用张量时引发错误
void unsupportedIsNonzero(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap: It looks like you're attempting to use a Tensor in some ",
        "data-dependent control flow. ",
        "We don't support that yet, please shout over at ",
        "https://github.com/pytorch/functorch/issues/257 .");
}

// 错误处理函数，当尝试在 torch.allclose 上使用 vmap 时引发错误
void unsupportedAllclose(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap over torch.allclose isn't supported yet. Please voice your ",
        "support over at github.com/pytorch/functorch/issues/275");
}
}

// 在 ATen 库中注册批处理实现的宏
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
    // 注册未支持的动态操作的错误处理函数
    UNSUPPORTED_DYNAMIC(nonzero);
    UNSUPPORTED_DYNAMIC(where);
    // 调用 UNSUPPORTED_DYNAMIC 宏，处理 unique_dim 动态特性不支持的情况
    UNSUPPORTED_DYNAMIC(unique_dim);
    // 调用 UNSUPPORTED_DYNAMIC 宏，处理 unique_consecutive 动态特性不支持的情况
    UNSUPPORTED_DYNAMIC(unique_consecutive);
    // 调用 UNSUPPORTED_DYNAMIC 宏，处理 unique_dim_consecutive 动态特性不支持的情况
    UNSUPPORTED_DYNAMIC(unique_dim_consecutive);
    // 调用 UNSUPPORTED_DYNAMIC 宏，处理 _unique2 动态特性不支持的情况
    UNSUPPORTED_DYNAMIC(_unique2);
    // 在 C++ 模块 m 中实现 "_local_scalar_dense" 方法，使用未支持的本地标量密集函数
    m.impl("_local_scalar_dense", torch::CppFunction::makeFromBoxedFunction<&unsupportedLocalScalarDense>());
    // 在 C++ 模块 m 中实现 "item" 方法，使用未支持的 item 函数
    m.impl("item", torch::CppFunction::makeFromBoxedFunction<&unsupportedItem>());
    // 在 C++ 模块 m 中实现 "is_nonzero" 方法，使用未支持的 is_nonzero 函数
    m.impl("is_nonzero", torch::CppFunction::makeFromBoxedFunction<&unsupportedIsNonzero>());
    // 在 C++ 模块 m 中实现 "allclose" 方法，使用未支持的 allclose 函数
    m.impl("allclose", torch::CppFunction::makeFromBoxedFunction<&unsupportedAllclose>());
}

} // namespace at::functorch
```