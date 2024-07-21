# `.\pytorch\aten\src\ATen\native\batch_norm.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 类头文件
#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub 头文件

namespace at::native {

using batch_norm_fn = void (*)(Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);
// 定义 batch_norm_fn 类型为指向接受特定参数的无返回值函数指针

using batch_norm_collect_stats_fn = void (*)(Tensor&, Tensor&, const Tensor&);
// 定义 batch_norm_collect_stats_fn 类型为指向接受特定参数的无返回值函数指针

using batch_norm_backward_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&,
        const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);
// 定义 batch_norm_backward_fn 类型为指向接受特定参数的无返回值函数指针

DECLARE_DISPATCH(batch_norm_fn, batch_norm_cpu_stub);
// 声明 batch_norm_cpu_stub 函数的调度分发器，用于 CPU 实现的批归一化操作

DECLARE_DISPATCH(batch_norm_collect_stats_fn, batch_norm_cpu_collect_stats_stub);
// 声明 batch_norm_cpu_collect_stats_stub 函数的调度分发器，用于 CPU 实现的批归一化统计信息收集操作

DECLARE_DISPATCH(batch_norm_backward_fn, batch_norm_cpu_backward_stub);
// 声明 batch_norm_cpu_backward_stub 函数的调度分发器，用于 CPU 实现的批归一化反向传播操作

// 在 TensorAccessor 定义有效时使用，用于解决未定义的问题...
template <typename scalar_t>
static TensorAccessor<scalar_t, 1> conditional_accessor_1d(const Tensor& t) {
  if (! t.defined()) {
    // 如果张量未定义，则返回一个空的 TensorAccessor
    return TensorAccessor<scalar_t, 1>(nullptr, nullptr, nullptr);
  }
  // 否则返回张量 t 的一维 TensorAccessor
  return t.accessor<scalar_t, 1>();
}

// 在 Tensor 的数据指针定义有效时使用
template <typename scalar_t>
static scalar_t* conditional_data_ptr(const Tensor& t) {
  if constexpr (std::is_const_v<scalar_t>) {
    // 如果 scalar_t 是 const 类型，返回 t 的连续常量数据指针（如果张量已定义），否则返回 nullptr
    return t.defined() ? t.contiguous().const_data_ptr<scalar_t>() : nullptr;
  } else {
    // 否则返回 t 的连续数据指针（如果张量已定义），否则返回 nullptr
    return t.defined() ? t.contiguous().data_ptr<scalar_t>() : nullptr;
  }
}

} // namespace at::native
// 结束 at::native 命名空间声明
```