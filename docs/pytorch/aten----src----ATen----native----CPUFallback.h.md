# `.\pytorch\aten\src\ATen\native\CPUFallback.h`

```
#pragma once
// 预处理指令，确保本头文件仅被编译一次

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 IValue 类定义

#include <ATen/core/stack.h>
// 包含 ATen 库中的 stack 类定义

#include <ATen/core/boxing/KernelFunction.h>
// 包含 ATen 库中的 KernelFunction 类定义

#include <ATen/core/dispatch/Dispatcher.h>
// 包含 ATen 库中的 Dispatcher 类定义

#include <c10/util/Metaprogramming.h>
// 包含 c10 库中的 Metaprogramming 实用工具

#include <torch/library.h>
// 包含 Torch 库中的 library 头文件

namespace at::native {

// 这个函数实现了一个针对 CPU 的包装回退策略。
// 外部后端可以在其上添加自定义日志，以定制其自己的 CPU 回退策略。
TORCH_API void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool error_on_views = false);

// 这是一个辅助函数，后端可以使用它来直接调用其封装的 CPU 回退策略。
// TODO: 在 https://github.com/pytorch/pytorch/pull/58092 完成后更新并添加用法示例。
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _call_fallback_fn final {};

// 上述模板的特化，用于定义具体的回退策略调用方法
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _call_fallback_fn<fallback_fn, Op, symint, ReturnType(ParameterTypes...)> final {
    static ReturnType call(typename c10::maybe_keep_symint<symint, ParameterTypes>::type... args) {
        auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow((const char*) Op::name, (const char*) Op::overload_name)
            .typed<ReturnType (typename c10::maybe_keep_symint<symint, ParameterTypes>::type...)>();
        return c10::impl::BoxedKernelWrapper<ReturnType (typename c10::maybe_keep_symint<symint, ParameterTypes>::type...)>::call(
            c10::BoxedKernel::makeFromFunction<fallback_fn>(),
            op,
            c10::DispatchKeySet(), // 我们知道 cpu_fallback 不使用 dispatch keyset。
            args...
            );
    }
};

// 对于需要符号整数支持的回退策略调用，使用此别名
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op>
using call_fallback_fn_symint = _call_fallback_fn<fallback_fn, Op, true, typename Op::schema>;

// 对于不需要符号整数支持的回退策略调用，使用此别名
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op>
using call_fallback_fn = _call_fallback_fn<fallback_fn, Op, false, typename Op::schema>;

} // namespace at::native
// 结束 at::native 命名空间
```