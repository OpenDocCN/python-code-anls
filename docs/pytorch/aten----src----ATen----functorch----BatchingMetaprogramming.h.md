# `.\pytorch\aten\src\ATen\functorch\BatchingMetaprogramming.h`

```
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/Tensor.h>
#include <ATen/VmapGeneratedPlumbing.h>

// This file contains template metaprogramming things that are used for our
// batching rules.
//
// See NOTE: [vmap plumbing] for more details on why this is necessary.
// The plumbing has a bunch of metaprogramming hacks for determining the signature
// of a batching rule from the signature of the operator, many of which use the
// helper functions in this file.

namespace at::functorch {

// Metaprogramming things

// Define typelist alias for variadic template arguments
template <class... Items> using typelist = c10::guts::typelist::typelist<Items...>;

// Extract the first type from a typelist
template <class TypeList> using head_t = c10::guts::typelist::head_t<TypeList>;

// Concatenate two typelists
template <class TL1, class TL2> using concat_t = c10::guts::typelist::concat_t<TL1, TL2>;

// Debugging utility for template metaprogramming
template <typename T> class debug_t;

// Define tail operation for typelists
template<class TypeList>
struct tail final {
    static_assert(c10::guts::false_t<TypeList>::value,
                  "In typelist::tail<T>, the T argument must be typelist<...>.");
};
template<class Head, class... Tail>
struct tail<typelist<Head, Tail...>> final {
  using type = typelist<Tail...>;
};

// Define conditional type transformation based on types
template <class First, class Second, class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext {
  using type = Next;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<Tensor, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<const Tensor&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<Tensor&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<optional<Tensor>, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<const optional<Tensor>&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<optional<Tensor>&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<std::vector<Tensor>, optional<int64_t>, Next, Tail> {
  using type = Tail;
};

// Define a metaprogramming struct for removing batch dimensions after tensors in a typelist
template <class TypeList> struct RemoveBatchDimAfterTensor {
  using first = head_t<TypeList>;
  using next = tail_t<TypeList>;
  using second = head_t<next>;
  using tail = tail_t<next>;

  using type = concat_t<
    typelist<first>,
    typename IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<first, second, next, tail>::type
  >;
};
    // typename 关键字用于声明类型名
    typename RemoveBatchDimAfterTensor<
      // 根据条件选择类型：如果 first 是张量且 second 是批量维度，则选择 tail 类型；否则选择 next 类型
      typename IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<first, second, next, tail>::type
    >::type
  >;
};  // 结构体或类定义结束

template <class Type> struct RemoveBatchDimAfterTensor<typelist<Type>> {
  using type = typelist<Type>;  // 如果 TypeList 中只有一个 Type，移除批处理维度后返回相同的 TypeList
};

template <> struct RemoveBatchDimAfterTensor<typelist<>> {
  using type = typelist<>;  // 如果 TypeList 为空，返回空的 TypeList
};

template<class TypeList> using remove_batch_dim_after_tensor_t = typename RemoveBatchDimAfterTensor<TypeList>::type;  // 使用 RemoveBatchDimAfterTensor 获取移除批处理维度后的 TypeList

template <typename T> struct UnpackSingleItemTuple {
  using type = T;  // 如果输入不是 std::tuple，则直接返回 T
};

template <typename T> struct UnpackSingleItemTuple<std::tuple<T>> {
  using type = T;  // 如果输入是 std::tuple<T>，则返回 T
};

template <typename T> using unpack_single_item_tuple_t = typename UnpackSingleItemTuple<T>::type;  // 使用 UnpackSingleItemTuple 获取解压后的类型

template <typename Return, typename TupleArgs> struct BuildFunctionHelper;
template <typename Return, typename... Args> struct BuildFunctionHelper<Return, std::tuple<Args...>> {
  using type = Return(Args...);  // 根据返回类型 Return 和参数 Args... 构建函数类型
};

template <typename Return, typename TL>
struct BuildFunction {
  using type = typename BuildFunctionHelper<Return, c10::guts::typelist::to_tuple_t<TL>>::type;  // 根据返回类型 Return 和 TypeList TL 构建函数类型
};

template <typename Return, typename TL> using build_function_t = typename BuildFunction<Return, TL>::type;  // 使用 BuildFunction 构建函数类型

template <typename batch_rule_t> struct ToOperatorType {
  using batch_rule_return_type = typename c10::guts::function_traits<batch_rule_t>::return_type;  // 获取 batch_rule_t 函数的返回类型
  using batch_rule_parameter_types = typename c10::guts::function_traits<batch_rule_t>::parameter_types;  // 获取 batch_rule_t 函数的参数类型列表

  using operator_parameter_types = remove_batch_dim_after_tensor_t<batch_rule_parameter_types>;  // 移除参数类型列表中的批处理维度
  using operator_return_type =
    unpack_single_item_tuple_t<
      c10::guts::typelist::to_tuple_t<
        remove_batch_dim_after_tensor_t<
          c10::guts::typelist::from_tuple_t<batch_rule_return_type>>>>;  // 移除返回类型中的批处理维度，并解压可能的单元素 tuple

  using type = build_function_t<operator_return_type, operator_parameter_types>;  // 构建最终的操作符函数类型
};

template <typename batch_rule_t> using to_operator_t = typename ToOperatorType<batch_rule_t>::type;  // 使用 ToOperatorType 获取操作符函数类型

} // namespace at::functorch
```