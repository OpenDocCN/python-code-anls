# `.\pytorch\tools\autograd\templates\variable_factories.h`

```
// 预处理指令，确保头文件只包含一次
#pragma once

// 使用 `${generated_comment}` 作为生成的注释，通常用于自动生成文档或标识文件生成工具

// 包含必要的头文件
#include <ATen/core/Tensor.h>
#include <ATen/TracerMode.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/MemoryFormat.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <torch/csrc/autograd/variable.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含标准 ATen 函数头文件，否则包含自定义的运算符头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/from_blob.h>
$ops_headers
#endif

#include <functional>  // 包含用于函数对象的标准库头文件
#include <initializer_list>  // 包含用于初始化列表的标准库头文件
#include <utility>  // 包含用于实用程序的标准库头文件

namespace torch {

/// 注意事项：当前 `torch::tensor(...)` 不支持混合数据类型
/// (例如 `torch::tensor({{bool, 2.0}})` 不起作用)。可能通过迭代所有子列表来找到
/// 可以表示所有元素的最大数据类型，或者使用可变模板，在未来支持这一功能。

/// 注意事项：使用浮点类型、`at::ArrayRef` / `std::vector` / (嵌套的) 列表初始化浮点类型
/// 总是产生与 Python `torch.tensor` 行为一致的默认数据类型的张量。

/// 注意事项：使用整数类型、`at::ArrayRef` / `std::vector` / (嵌套的) 列表初始化整数类型
/// 总是产生与 Python `torch.tensor` 行为一致的 `at::kLong` (即 int64_t) 数据类型的张量。

/// 注意事项：当前 `torch::tensor` 不支持以下数据类型：
/// - `unsigned int`
/// - `unsigned long int`
/// - `unsigned long long int`
/// - `long long int`
inline at::Tensor tensor(detail::TensorDataContainer tensor_data_container, const at::TensorOptions& options = {}) {
  return autograd::make_variable(
    // 注意：我们从 TensorOptions 中移除 requires_grad 设置，因为这个设置会被忽略
    // 我们在此处显式处理 requires_grad，而不是将其传递到内核中。
    tensor_data_container.convert_to_tensor(options.requires_grad(c10::nullopt)),
    options.requires_grad());
}

/// 通用的删除函数类型
using Deleter = std::function<void(void*)>;

/// 内存格式类型
using at::MemoryFormat;

/// 将给定的 `data` 作为 `Tensor` 暴露出来，但不接管原始数据的所有权。
/// `sizes` 应该指定张量的形状，`strides` 每个维度的步长。
/// 当张量数据通常会被释放时，将调用 `deleter` 函数（一个 `std::function<void(void*)>`）。
/// `TensorOptions` 指定返回张量的额外配置选项，例如如何解释 `data` 的类型。
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  // 创建张量，并使用 lambda 表达式初始化
  at::Tensor tensor = ([&]() {
    // 自动调度到梯度下方（暂时移除自动求导），TODO: 删除
    at::AutoDispatchBelowAutograd guard;  
    // 不使用追踪器分发模式
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    // 返回创建的张量
    // 使用给定的数据、大小、步长、释放器和梯度选项创建张量，并返回对应的 Tensor 对象
    return at::from_blob(data, sizes, strides, deleter, options.requires_grad(c10::nullopt));
  })();
  // 使用给定的 Tensor 和梯度选项创建一个 Variable 对象，并返回
  return autograd::make_variable(tensor, options.requires_grad());
}

/// 命名空间结束

/// 将给定的 `data` 作为 `Tensor` 暴露，不接管原始数据。`sizes` 应指定张量的形状，`strides` 每个维度的步幅。`TensorOptions` 指定返回张量的附加配置选项，例如如何解释 `data` 的类型。
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options = at::TensorOptions()) {
  // 进入匿名函数作用域
  at::Tensor tensor = ([&]() {
    // 在自动微分下的分发保护，待移除
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    // 跟踪器分发模式设为无跟踪器
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    // 使用给定参数创建张量对象
    return at::from_blob(data, sizes, strides, options.requires_grad(c10::nullopt));
  })();
  // 将张量包装为可变变量，并设置是否需要梯度
  return autograd::make_variable(tensor, options.requires_grad());
}

/// 将给定的 `data` 作为 `Tensor` 暴露，不接管原始数据。`sizes` 应指定张量的形状。`deleter`（一个 `std::function<void(void*)>` 函数）将在正常情况下释放张量数据时调用。`TensorOptions` 指定返回张量的附加配置选项，例如如何解释 `data` 的类型。
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  // 进入匿名函数作用域
  at::Tensor tensor = ([&]() {
    // 在自动微分下的分发保护，待移除
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    // 跟踪器分发模式设为无跟踪器
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    // 使用给定参数创建张量对象
    return at::from_blob(data, sizes, deleter, options.requires_grad(c10::nullopt));
  })();
  // 将张量包装为可变变量，并设置是否需要梯度
  return autograd::make_variable(tensor, options.requires_grad());
}

/// 将给定的 `data` 作为 `Tensor` 暴露，不接管原始数据。`sizes` 应指定张量的形状。`TensorOptions` 指定返回张量的附加配置选项，例如如何解释 `data` 的类型。
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  // 进入匿名函数作用域
  at::Tensor tensor = ([&]() {
    // 在自动微分下的分发保护，待移除
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    // 跟踪器分发模式设为无跟踪器
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    // 使用给定参数创建张量对象
    return at::from_blob(data, sizes, options.requires_grad(c10::nullopt));
  })();
  // 将张量包装为可变变量，并设置是否需要梯度
  return autograd::make_variable(tensor, options.requires_grad());
}

${function_definitions}

} // namespace torch
```