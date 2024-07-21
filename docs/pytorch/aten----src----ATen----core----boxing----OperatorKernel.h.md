# `.\pytorch\aten\src\ATen\core\boxing\OperatorKernel.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <c10/util/intrusive_ptr.h>
// 包含 c10 库的 intrusive_ptr.h 文件

namespace c10 {
// 命名空间 c10 开始

/**
 * Inherit from OperatorKernel to implement a c10 kernel.
 *
 * Example:
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 *
 * The kernel class is allowed to have members but these are equivalent
 * to global variables. The kernel implementation is responsible for
 * preventing race conditions on them.
 *
 * See below for how to register this kernel with PyTorch.
 */
// 定义 OperatorKernel 结构体，继承自 c10::intrusive_ptr_target
struct TORCH_API OperatorKernel : public c10::intrusive_ptr_target {
  // 析构函数，使用默认实现，虚函数重写
  ~OperatorKernel() override = default;
};

}  // namespace c10
// 命名空间 c10 结束
```