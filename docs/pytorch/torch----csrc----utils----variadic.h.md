# `.\pytorch\torch\csrc\utils\variadic.h`

```py
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Variadic.h>
#include <torch/csrc/autograd/variable.h>

#include <type_traits>
#include <utility>

namespace torch {

using at::IterArgs;

// 结构体 CountTensors 继承自 IterArgs<CountTensors>，用于统计张量数量
struct CountTensors : IterArgs<CountTensors> {
  size_t out = 0; // 记录张量数量的变量

  // 处理单个张量的操作符重载
  void operator()(const at::Tensor& x) {
    out += 1;
  }

  // 处理 std::optional<at::Tensor> 类型的操作符重载
  void operator()(const std::optional<at::Tensor>& x) {
    out += x.has_value(); // 如果有值则加1
  }

  // 处理 at::ArrayRef<at::Tensor> 类型的操作符重载
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out += xs.size(); // 加上数组中张量的数量
  }
};

// 模板函数 count_tensors 统计传入参数中的张量数量
template <typename... Args>
size_t count_tensors(Args&&... args) {
  return CountTensors().apply(std::forward<Args>(args)...).out; // 调用 CountTensors 的 apply 方法并返回张量数量
}

// 结构体 CountVariables 继承自 IterArgs<CountVariables>，用于统计变量数量
struct CountVariables : IterArgs<CountVariables> {
  size_t out = 0; // 记录变量数量的变量

  // 处理 autograd::Variable 类型的操作符重载
  void operator()(const autograd::Variable& x) {
    out += 1;
  }

  // 处理 at::ArrayRef<autograd::Variable> 类型的操作符重载
  void operator()(at::ArrayRef<autograd::Variable> xs) {
    out += xs.size(); // 加上数组中变量的数量
  }
};

// 模板函数 count_variables 统计传入参数中的变量数量
template <typename... Args>
inline size_t count_variables(Args&&... args) {
  return CountVariables().apply(std::forward<Args>(args)...).out; // 调用 CountVariables 的 apply 方法并返回变量数量
}

//===----------------------------------------------------------------------===//
//                std::index_sequence shim for C++11
//===----------------------------------------------------------------------===//

// 一个包含类型模板参数索引的容器
template <size_t... Is>
struct Indices {};

// 减少索引 N，将 N-1 添加到索引列表中并转发已有内容
template <size_t N, size_t... Is>
struct MakeIndices : MakeIndices<N - 1, N - 1, Is...> {};

// 当 N 为零时的部分特化，定义基本情况
template <size_t... Is>
struct MakeIndices<0, Is...> {
  using indices = Indices<Is...>; // 定义包含 0 到 N-1 的索引列表的 typedef
};

//===----------------------------------------------------------------------===//
//                                 Utilities
//===----------------------------------------------------------------------===//

// apply 函数调用给定的函数对传入参数进行处理
template <typename Function, typename... Ts>
void apply(Function function, Ts&&... ts) {
  // https://stackoverflow.com/questions/13978916/inserting-a-variadic-argument-list-into-a-vector
  // 创建一个虚拟数组，以便每个函数调用都按顺序进行评估
  // `(function(), 0)` 是因为 `function` 应该返回 `void`，所以根据逗号运算符，它被评估并丢弃其结果（`void`）。
  // 然后评估零，并将其用作数组的一个元素。第一个零确保数组不为空。
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  int _[]{0, (function(std::forward<Ts>(ts)), 0)...};
  (void)_; // 确保数组不会被使用，防止编译器发出未使用变量的警告
}

template <
    typename ReturnType,
    typename... Ts,
    typename Function,
    typename Accessor>
ReturnType unpack(Function function, Accessor accessor) {
  // 调用模板函数 unpack，传入 function、accessor，并生成模板参数索引序列
  return ReturnType(unpack<ReturnType, Ts...>(
      std::move(function),
      std::move(accessor),
      typename MakeIndices<sizeof...(Ts)>::indices()));
}

template <
    typename ReturnType,
    typename... Ts,
    typename Function,
    typename Accessor,
    size_t... Is>
ReturnType unpack(Function function, Accessor accessor, Indices<Is...>) {
  // 调用 function，使用 accessor 模板调用生成 Ts 类型的参数，传入 Is 索引序列
  return ReturnType(function(accessor.template operator()<Ts>(Is)...));
}

} // namespace torch
```