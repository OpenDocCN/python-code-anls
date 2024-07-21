# `.\pytorch\test\edge\templates\RegisterCodegenUnboxedKernels.cpp`

```py
#include <operator_registry.h>
#include <event_tracer_hooks.h>
#include "${fn_header}" // Generated Function import headers

namespace torch {
namespace executor {

using namespace internal;

namespace {
// 定义一个别名，用于表示 Kernel 对象的 ArrayRef
using KernelArrayRef = ::at::ArrayRef<::torch::executor::Kernel>;

// 静态数组，包含要注册的内核对象
static Kernel kernels_to_register[] = {
    ${unboxed_kernels} // Generated operators
};

// 显式转换为 ArrayRef，以便 API 可以接受一个空的 C 数组作为 Kernels 参数
static KernelArrayRef kernel_array_ref(
    kernels_to_register,
    kernels_to_register + sizeof(kernels_to_register) / sizeof(Kernel));

// 注册内核对象，将静态变量的赋值保留在静态初始化阶段，返回值未使用
static auto success_with_kernel_reg = register_kernels(kernel_array_ref);
} // namespace
} // namespace executor
} // namespace torch
```