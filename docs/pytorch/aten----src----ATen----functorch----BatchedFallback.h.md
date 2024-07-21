# `.\pytorch\aten\src\ATen\functorch\BatchedFallback.h`

```
// 版权声明和许可信息
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once // 确保头文件只包含一次

#include <ATen/ATen.h> // 引入 ATen 库
#include <ATen/core/op_registration/op_registration.h> // 引入 ATen 的操作注册头文件
#include <torch/library.h> // 引入 Torch 库

namespace at::functorch {

// 以下是 vmap 回退（也称为 BatchedTensor 回退或 Batched 回退）的代码，用于当操作没有批处理规则实现时运行。

// 如果操作没有批处理规则实现，则使用此实现作为回退。回退不适用于 out= 变体或视图操作；
// 即，它适用于无位置操作和非视图的就地操作。
//
// 对于无位置操作，回退实际上取 `stack` 中的所有 BatchedTensor，切片它们，并在所有相应的切片上运行 `op`，
// 以生成输出的切片。然后，输出切片通过 `torch.stack` 来创建最终的返回结果。
//
// 由于回退会引入从堆叠切片输出中的额外复制，其性能并不是很好。因此，我们优先为操作编写批处理规则。
void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

// 用于嵌套张量的 vmap 回退
void batchedNestedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

// vmap 回退时出错的处理
void vmapErrorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

// 默认情况下，vmap 回退会发出警告，但如果用户发现它太烦人，可以禁用。
TORCH_API bool isVmapFallbackWarningEnabled();
TORCH_API void setVmapFallbackWarningEnabled(bool enabled);

// 用于测试。默认情况下启用 vmap 回退。当其被禁用时，会引发错误。
TORCH_API bool isVmapFallbackEnabled();
TORCH_API void setVmapFallbackEnabled(bool enabled);

// 将 std::vector<IValue> 转换为特定类型 A 的结果
template <typename A>
A vector_to_result(const std::vector<IValue>& buffer) {
  return buffer[0].to<A>();
}

// 将 std::vector<IValue> 转换为类型 A 和 B 的结果元组
template <typename A, typename B>
std::tuple<A, B> vector_to_result(const std::vector<IValue>& buffer) {
  return std::make_tuple(buffer[0].to<A>(), buffer[1].to<B>());
}

// 将 std::vector<IValue> 转换为类型 A、B 和 C 的结果元组
template <typename A, typename B, typename C>
std::tuple<A, B, C> vector_to_result(const std::vector<IValue>& buffer) {
  return std::make_tuple(buffer[0].to<A>(), buffer[1].to<B>(), buffer[2].to<C>());
}

// slow_fallback 是一种在某些封装内核中调用 vmap 回退的方法。可能有更好的元编程方式来实现这一点。
template <typename Ret>
Ret slow_fallback(const c10::OperatorHandle& op, ArrayRef<IValue> args) {
  // 将参数 args 转换为 std::vector<IValue>
  std::vector<IValue> stack(args.begin(), args.end());
  // 调用 batchedTensorForLoopFallback 进行 vmap 回退
  batchedTensorForLoopFallback(op, &stack);
  // 将结果转换为类型 Ret 并返回
  return vector_to_result<Ret>(stack);
}

// 模板继续定义未完成的部分
template <typename A, typename B>
// 定义一个函数模板，返回类型为 std::tuple<A, B>，接受一个操作符句柄和参数列表
std::tuple<A, B> slow_fallback(const c10::OperatorHandle& op, ArrayRef<IValue> args) {
  // 创建一个包含参数列表的 IValue 类型的堆栈
  std::vector<IValue> stack(args.begin(), args.end());
  // 调用 batchedTensorForLoopFallback 函数处理堆栈中的数据
  batchedTensorForLoopFallback(op, &stack);
  // 将堆栈转换为 std::tuple<A, B> 类型的结果并返回
  return vector_to_result<A, B>(stack);
}

// 定义一个函数模板，返回类型为 std::tuple<A, B, C>，接受一个操作符句柄和参数列表
template <typename A, typename B, typename C>
std::tuple<A, B, C> slow_fallback(const c10::OperatorHandle& op, ArrayRef<IValue> args) {
  // 创建一个包含参数列表的 IValue 类型的堆栈
  std::vector<IValue> stack(args.begin(), args.end());
  // 调用 batchedTensorForLoopFallback 函数处理堆栈中的数据
  batchedTensorForLoopFallback(op, &stack);
  // 将堆栈转换为 std::tuple<A, B, C> 类型的结果并返回
  return vector_to_result<A, B, C>(stack);
}

// 命名空间声明结束
} // namespace at::functorch
```