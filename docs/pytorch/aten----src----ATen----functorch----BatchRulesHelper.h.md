# `.\pytorch\aten\src\ATen\functorch\BatchRulesHelper.h`

```
// 版权声明及许可信息
// 版权所有，Facebook公司及其关联公司
// 此源代码根据根目录下的LICENSE文件中的BSD风格许可证授权。

// 包含必要的头文件
#pragma once

#include <c10/util/TypeList.h> // 引入C10库中的TypeList工具

#include <ATen/ATen.h> // 引入ATen库
#include <ATen/Operators.h> // 引入ATen运算符

#include <ATen/functorch/DynamicLayer.h> // 引入functorch中的动态层相关头文件
#include <ATen/functorch/TensorWrapper.h> // 引入functorch中的Tensor封装头文件
#include <ATen/functorch/BatchingMetaprogramming.h> // 引入functorch中的批处理元编程头文件
#include <ATen/functorch/LegacyVmapTransforms.h> // 引入functorch中的遗留Vmap转换头文件
#include <ATen/functorch/BatchedFallback.h> // 引入functorch中的批处理回退相关头文件
#include <ATen/functorch/PlumbingHelper.h> // 引入functorch中的Plumbing辅助头文件
#include <ATen/core/dispatch/Dispatcher.h> // 引入ATen核心的调度器
#include <ATen/VmapGeneratedPlumbing.h> // 引入Vmap生成的Plumbing工具

#include <utility> // 引入C++标准库中的utility工具

// 此文件包含用于批处理规则的辅助函数

namespace at::functorch {

// 定义了一些TORCH_API修饰的函数接口，具体实现可在相应的源文件中找到
TORCH_API Tensor reshape_dim_into(int64_t src, int64_t dst, const Tensor& x);
TORCH_API Tensor reshape_dim_outof(int64_t src, int64_t size1, const Tensor& x);

TORCH_API Tensor reshape_dim_outof_symint(int64_t src, const c10::SymInt& size1, const Tensor& x);

// 将张量的批处理维度（如果有的话）移到最前面
Tensor moveBatchDimToFront(const Tensor& tensor, optional<int64_t> maybe_batch_dim);

// 返回排除批处理维度后的张量秩
int64_t rankWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim);

// 返回排除批处理维度后的张量元素数
int64_t numelWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim);

// 如果maybe_empty非空，则返回new_val，否则返回空
optional<int64_t> valIfNonempty(optional<int64_t> maybe_empty, int64_t new_val);

// 获取张量的物理维度，考虑是否有批处理维度和逻辑维度
int64_t getPhysicalDim(const Tensor& tensor, bool has_batch_dim, int64_t logical_dim);

// 获取张量的物理维度向量，考虑是否有批处理维度和逻辑维度数组
VmapDimVector getPhysicalDims(const Tensor& tensor, bool has_batch_dim, IntArrayRef logical_dims);

// 报告Vmap不兼容的就地操作错误
void vmapIncompatibleInplaceError(const char* schema_name);

// 如果可能，将张量填充到逻辑秩
Tensor maybePadToLogicalRank(const Tensor& tensor, optional<int64_t> has_bdim, int64_t logical_rank);

// 检查随机性类型
void check_randomness(RandomnessType randomness);
void check_randomness(RandomnessType randomness, bool any_tensor_bdim);

// 确保张量具有批处理维度的辅助函数
inline Tensor ensure_has_bdim(const Tensor& tensor, bool has_bdim, c10::SymInt batch_size) {
  if (has_bdim) {
    return tensor;
  }
  const auto sizes = tensor.sym_sizes();
  SymDimVector expanded_shape;
  expanded_shape.reserve(sizes.size());
  expanded_shape.emplace_back(std::move(batch_size));
  expanded_shape.insert(expanded_shape.end(), sizes.begin(), sizes.end());
  return tensor.expand_symint(expanded_shape);
}

// 定义宏，用于将操作（op）与其批处理规则（batch_rule）绑定
#define VMAP_SUPPORT(op, batch_rule) \
  m.impl(#op, op ## _generated_plumbing<decltype(&batch_rule), &batch_rule>);

// 定义宏，用于将操作（op）的特定重载（overload）与其批处理规则（batch_rule）绑定
#define VMAP_SUPPORT2(op, overload, batch_rule) \
  m.impl(#op "." #overload, op ## _ ## overload ## _generated_plumbing<decltype(&batch_rule), &batch_rule>);

// 定义宏，用于将操作（op）分解为其基本实现
#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

// 不要直接使用此模板，使用BASIC_UNARY_BATCH_RULE以避免一些痛苦
template <typename A, A a, typename C>
struct BasicUnaryBatchRuleHelper;

// 结尾处可能有更多代码，但没有提供的信息无法确定
// 以上是该文件的注释部分，仅对提供的代码段进行注释，不对其余代码作出推测
// 定义一个结构体模板 `BasicUnaryBatchRuleHelper`，接受三个模板参数：
// - F：函数指针类型
// - Func：函数指针 `&fn`
// - c10::guts::typelist::typelist<A, T...>：模板参数列表
struct BasicUnaryBatchRuleHelper<F, Func, c10::guts::typelist::typelist<A, T...>> {
  // 定义静态成员函数 `apply`，接受以下参数：
  // - tensor：输入的张量
  // - batch_dim：批处理维度的可选值
  // - extra_args...：可变参数列表
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    // 调用 Func 函数指针，将张量 tensor 和额外参数传递给它，返回一个元组
    return std::make_tuple(Func(tensor, std::forward<T>(extra_args)...), batch_dim);
  }
};

// 定义一个宏 `BASIC_UNARY_BATCH_RULE`，用于简化 `BasicUnaryBatchRuleHelper` 的调用
// 使用示例：BASIC_UNARY_BATCH_RULE(at::sin)
// 错误使用示例：BASIC_UNARY_BATCH_RULE(&at::sin)，重要提示：不要传递函数指针给这个宏!!
#define BASIC_UNARY_BATCH_RULE(fn) SINGLE_ARG(\
    BasicUnaryBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

// 定义一个宏 `UNARY_POINTWISE`，用于支持一元操作符的批处理规则
#define UNARY_POINTWISE(op) \
  VMAP_SUPPORT(op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));

// 定义一个结构体模板 `VariadicBdimsBatchRuleHelper`，接受四个模板参数：
// - F：函数指针类型
// - Func：函数指针 `&fn`
// - A：模板参数
// - T...：剩余的模板参数列表
template <typename F, F Func, typename A, typename... T>
struct VariadicBdimsBatchRuleHelper<F, Func, c10::guts::typelist::typelist<A, T...>> {
  // 定义静态成员函数 `apply`，接受以下参数：
  // - tensor：输入的张量
  // - batch_dim：批处理维度的可选值
  // - extra_args...：可变参数列表
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    // 将批处理维度移动到张量的最前面
    auto tensor_ = moveBatchDimToFront(tensor, batch_dim);
    // 调用 Func 函数指针，将移动后的张量 tensor_ 和额外参数传递给它，返回一个元组
    return std::make_tuple(Func(tensor_, std::forward<T>(extra_args)...), 0);
  }
};

// 定义一个宏 `VARIADIC_BDIMS_BATCH_RULE`，用于简化 `VariadicBdimsBatchRuleHelper` 的调用
// 使用示例：VARIADIC_BDIMS_BATCH_RULE(at::cholesky_inverse)
// 错误使用示例：VARIADIC_BDIMS_BATCH_RULE(&at::cholesky_inverse)，重要提示：不要传递函数指针给这个宏!!
#define VARIADIC_BDIMS_BATCH_RULE(fn) SINGLE_ARG(\
    VariadicBdimsBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

// 定义一个宏 `VARIADIC_BDIMS`，用于支持多个参数的批处理规则
#define VARIADIC_BDIMS(op) \
  VMAP_SUPPORT(op, VARIADIC_BDIMS_BATCH_RULE(ATEN_FN(op)));

// 定义一个宏 `VARIADIC_BDIMS2`，用于支持多个参数和重载函数的批处理规则
#define VARIADIC_BDIMS2(op, overload) \
  VMAP_SUPPORT2(op, overload, VARIADIC_BDIMS_BATCH_RULE(ATEN_FN2(op, overload)));

// 定义一个模板函数 `boxed_tensor_inputs_batch_rule`，接受两个模板参数：
// - F：函数指针类型
// - Func：函数指针
template<class F, F Func>
void boxed_tensor_inputs_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 获取操作符的模式
  const auto& schema = op.schema();
  // 获取返回值的数量
  const auto num_returns = schema.returns().size();
  // 获取参数的数量
  const auto num_arguments = schema.arguments().size();

  // 临时排除分发键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查当前层是否逃逸，如果逃逸则抛出异常
  vmap_check_escaped(maybe_layer, "boxed_tensor_inputs_batch_rule");

  // 获取当前层的层级
  int64_t cur_level = maybe_layer->layerId();

  // 获取原始参数
  auto orig_arguments = torch::jit::last(*stack, num_arguments);
  // 如果原始参数中没有一个参与当前层级，则调用操作符的盒装函数并返回
  if (std::none_of(orig_arguments.begin(), orig_arguments.end(), ivalueParticipatesInCurrentLevel)) {
    op.callBoxed(stack);
    return;
  }

  // 弹出参数并存储到 arguments 中
  auto arguments = torch::jit::pop(*stack, num_arguments);
  // 存储张量输入的向量和张量位置的向量
  std::vector<std::pair<Tensor, optional<int64_t>>> tensor_inputs;
  std::vector<int64_t> tensor_pos;
  // 对参数索引进行循环，从 0 到 num_arguments
  for (const auto idx : c10::irange(0, num_arguments)) {
    // 获取参数的 IValue
    const auto& ivalue = arguments[idx];
  // 如果输入值是张量
  if (ivalue.isTensor()) {
    // 解包张量及其维度信息，根据当前级别进行处理
    auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(ivalue.toTensor(), cur_level);
    // 将解包后的张量及其维度信息加入张量输入列表
    tensor_inputs.emplace_back(tensor_value, tensor_bdim);
    // 记录张量在原始参数列表中的位置
    tensor_pos.push_back(static_cast<int64_t>(idx));
  }
}

// 将张量输入列表传递给函数 Func 进行处理
Func(tensor_inputs);

// 初始化张量索引
size_t tensor_idx = 0;
// 断言张量位置列表非空
TORCH_INTERNAL_ASSERT(!tensor_pos.empty());
// 遍历每个参数索引
for (const auto arg_idx : c10::irange(0, num_arguments)) {
  // 如果张量索引超出张量位置列表范围，或者当前参数索引不等于张量位置列表中的索引
  if (tensor_idx >= tensor_pos.size() || (int64_t)arg_idx != tensor_pos[tensor_idx]) {
    // 将参数压入 Torch 的堆栈中
    torch::jit::push(stack, arguments[arg_idx]);
  } else {
    // 断言张量索引小于张量输入列表的大小
    TORCH_INTERNAL_ASSERT(tensor_idx < tensor_inputs.size());
    // 将张量值压入 Torch 的堆栈中
    torch::jit::push(stack, tensor_inputs[tensor_idx].first);
    // 增加张量索引以指向下一个张量
    tensor_idx++;
  }
}

// 调用 op 的 boxed 版本，传入堆栈
op.callBoxed(stack);

// 从堆栈中弹出返回值列表，数量为 num_returns
const auto returns = torch::jit::pop(*stack, num_returns);
// 遍历每个返回值
for (const auto& ret : returns) {
  // 如果返回值是张量
  if (ret.isTensor()) {
    // 将返回的张量进行批处理，放入堆栈中
    torch::jit::push(stack, makeBatched(ret.toTensor(), 0, cur_level));
  } else {
    // 如果返回值不是张量，抛出错误，因为此 boxed 版本不支持非张量返回值
    TORCH_INTERNAL_ASSERT(false, "This boxed batching rule does not currently support ops that return non-tensor values");
  }
}
}

// 处理逐点运算的函数，接受一个存储张量和可选整数对的向量作为输入
inline void handle_pointwise_ops(std::vector<std::pair<Tensor, optional<int64_t>>> &tensor_inputs) {
  // 初始化输出逻辑秩为0
  int64_t out_logical_rank = 0;
  // 遍历输入张量对
  for (auto& tensor_input : tensor_inputs) {
    // 计算当前张量的逻辑秩（不包括批次维度）
    int64_t cur_logical_rank = rankWithoutBatchDim(tensor_input.first, tensor_input.second);
    // 更新输出的最大逻辑秩
    out_logical_rank = std::max(out_logical_rank, cur_logical_rank);
  }
  // 对每个张量对执行以下操作
  for (auto& tensor_input: tensor_inputs) {
    // 将批次维度移动到张量的最前面
    tensor_input.first = moveBatchDimToFront(tensor_input.first, tensor_input.second);
    // 如果需要，将张量填充到与输出逻辑秩相同的尺寸
    tensor_input.first = maybePadToLogicalRank(tensor_input.first, tensor_input.second, out_logical_rank);
  }
}

// 定义宏 POINTWISE_BOXED，用于注册处理逐点操作的函数
#define POINTWISE_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_tensor_inputs_batch_rule<decltype(&handle_pointwise_ops), &handle_pointwise_ops>>());

// 定义宏 POINTWISE_BOXED2，用于注册处理逐点操作的函数，带有重载标识
#define POINTWISE_BOXED2(op, overload) \
  m.impl(#op "." #overload, torch::CppFunction::makeFromBoxedFunction<boxed_tensor_inputs_batch_rule<decltype(&handle_pointwise_ops), &handle_pointwise_ops>>());

// 处理变长批次维度的函数，接受一个存储张量和可选整数对的向量作为输入
inline void handle_variadic_bdims(std::vector<std::pair<Tensor, optional<int64_t>>> &tensor_inputs) {
  // 对每个张量对执行以下操作
  for (auto & tensor_input : tensor_inputs) {
    // 将批次维度移动到张量的最前面
    tensor_input.first = moveBatchDimToFront(tensor_input.first, tensor_input.second);
  }
}

// 定义宏 VARIADIC_BDIMS_BOXED，用于注册处理变长批次维度操作的函数
#define VARIADIC_BDIMS_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_tensor_inputs_batch_rule<decltype(&handle_variadic_bdims), &handle_variadic_bdims>>());

// 定义类型 UnpackedBatchedTensor，表示解包的批次张量对
using UnpackedBatchedTensor = std::tuple<Tensor,optional<int64_t>>;

// 查找和解包张量的函数，接受一个 Torch 栈指针，参数数量，当前级别，张量列表，位置列表和批次大小作为输入
inline void find_and_unpack_tensors(
    const torch::jit::Stack* stack,
    int64_t num_args,
    int64_t cur_level,
    SmallVector<UnpackedBatchedTensor, 5>* tensors,
    SmallVector<int64_t, 5>* tensors_pos,
    int64_t* batch_size) {

  // 初始化计算出的批次大小为 -1
  int64_t computed_batch_size = -1;
  // 计算输入参数在 Torch 栈中的起始位置
  int64_t args_begin = static_cast<int64_t>(stack->size()) - num_args;

  // 遍历参数的索引范围
  for (const auto idx : c10::irange(0, num_args)) {
    // 获取 Torch 栈中的值
    const auto& ivalue = (*stack)[args_begin + idx];
    // 如果不是张量类型，则继续下一个循环
    if (!ivalue.isTensor()) {
      continue;
    }
    // 解包张量到指定级别，并获取张量及其可能的批次维度
    auto unpacked = unwrapTensorAtLevel(ivalue.toTensor(), cur_level);
    const auto& tensor_value = std::get<0>(unpacked);
    const auto tensor_bdim = std::get<1>(unpacked);
    // 如果张量有批次维度
    if (tensor_bdim.has_value()) {
      // 获取当前张量维度下的尺寸
      auto candidate_batch_size = tensor_value.size(*tensor_bdim);
      // 如果计算出的批次大小为 -1，则更新为当前尺寸
      if (computed_batch_size == -1) {
        computed_batch_size = candidate_batch_size;
      }
      // 断言当前尺寸与计算出的批次大小相同
      TORCH_INTERNAL_ASSERT(candidate_batch_size == computed_batch_size);
    }

    // 将解包的张量对和其位置添加到对应的向量列表中
    tensors->push_back(std::move(unpacked));
    tensors_pos->push_back(idx);
  }
  // 断言计算出的批次大小大于 -1
  TORCH_INTERNAL_ASSERT(computed_batch_size > -1);
  // 更新批次大小的指针值
  *batch_size = computed_batch_size;
}

// 定义处理已存在批次维度的所有批次规则的函数，接受一个存储张量和可选整数对的向量作为输入
inline void boxed_existing_bdim_all_batch_rule(
  // 获取操作符的 schema
  const auto& schema = op.schema();
  // 获取返回值数量
  const auto num_returns = schema.returns().size();
  // 获取参数数量
  const auto num_arguments = static_cast<int64_t>(schema.arguments().size());

  // 排除 DispatchKey 为 FuncTorchBatched 的调度键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查当前层是否为 "boxed_existing_bdim_all_batch_rule"
  vmap_check_escaped(maybe_layer, "boxed_existing_bdim_all_batch_rule");
  // 获取当前层的层级 ID
  int64_t cur_level = maybe_layer->layerId();

  // 从栈中获取参数
  const auto arguments = torch::jit::last(stack, num_arguments);
  // 如果所有参数都不参与当前层次，则直接调用操作符并返回
  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    op.callBoxed(stack);
    return;
  }

  // 计算参数在栈中的起始位置
  int64_t args_begin = static_cast<int64_t>(stack->size()) - num_arguments;
  // 存储解压缩的批量张量和其位置
  SmallVector<UnpackedBatchedTensor, 5> tensor_inputs;
  SmallVector<int64_t, 5> tensor_pos;
  int64_t batch_size = 0;

  // 查找和解压缩张量
  find_and_unpack_tensors(
      stack, num_arguments, cur_level,
      &tensor_inputs, &tensor_pos, &batch_size);

  // 对于每个张量，确保它有一个批量维度并进行重塑
  for (const auto tensor_idx : c10::irange(0, tensor_inputs.size())) {
    const auto& value = std::get<0>(tensor_inputs[tensor_idx]);
    auto bdim = std::get<1>(tensor_inputs[tensor_idx]);
    auto value_ = ensure_has_bdim(value, bdim.has_value(), batch_size);
    if (!bdim.has_value()) {
      bdim = 0;
    }
    // 将重塑后的张量放回到栈的对应位置
    (*stack)[args_begin + tensor_pos[tensor_idx]] = reshape_dim_into(*bdim, 0, value_);
  }

  // 调用操作符
  op.callBoxed(stack);

  // 对于每个返回值，确保其为张量类型，并进行批量化处理
  for (const auto idx : c10::irange(args_begin, args_begin + num_returns)) {
    const auto& ret = (*stack)[idx];
    TORCH_INTERNAL_ASSERT(ret.isTensor(),
        "This boxed batching rule does not currently support ops that return non-tensor values");
    (*stack)[idx] = makeBatched(reshape_dim_outof(0, batch_size, ret.toTensor()), 0, cur_level);
  }
// 结束宏定义

// 用于所有张量参数都接受一个（普通）批次维度的情况。
// 这个批处理规则会在所有张量上扩展批次维度，将其重塑为dim 0，调用操作，然后将批次维度从dim 0中移除。
// 这不是最有效的方法；如果有其他选择，请尽量使用它们。只在没有其他选择时使用这个方法。
#define EXISTING_BDIM_ALL_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_existing_bdim_all_batch_rule>());

// 声明一个行内函数，用于处理所有张量都有可选批次维度的情况
template <int64_t feature_rank, int64_t contig_tensor_index=-1>
inline void boxed_all_tensors_have_optional_bdim(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 获取操作的架构信息
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();  // 返回值数量
  const auto num_arguments = schema.arguments().size();  // 参数数量

  // 排除 DispatchKey 为 FuncTorchBatched 的调度键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前动态层次
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查动态层次是否逃逸，记录日志
  vmap_check_escaped(maybe_layer, "boxed_all_tensors_have_optional_bdim");
  int64_t cur_level = maybe_layer->layerId();  // 当前层次的 ID

  // 从栈中获取参数值
  const auto arguments = torch::jit::last(stack, num_arguments);
  // 如果没有任何参数参与到当前层次，则调用操作并返回
  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    op.callBoxed(stack);
    return;
  }

  // 获取参数起始位置及相关的张量输入和其位置
  int64_t args_begin = static_cast<int64_t>(stack->size() - num_arguments);
  SmallVector<UnpackedBatchedTensor, 5> tensor_inputs;  // 未打包的批次张量
  SmallVector<int64_t, 5> tensor_pos;  // 张量位置
  int64_t batch_size = 0;  // 批次大小

  // 查找并解包张量，确定其位置及批次大小
  find_and_unpack_tensors(
      stack, static_cast<int64_t>(num_arguments), cur_level,
      &tensor_inputs, &tensor_pos, &batch_size);

  optional<bool> is_no_batch_dim_case;  // 是否为无批次维度情况的可选布尔值

  // 遍历每个张量输入
  for (const auto tensor_idx : c10::irange(0, tensor_inputs.size())) {
    const auto& value = std::get<0>(tensor_inputs[tensor_idx]);  // 获取张量值
    auto bdim = std::get<1>(tensor_inputs[tensor_idx]);  // 获取批次维度
    const auto logical_rank = rankWithoutBatchDim(value, bdim);  // 获取逻辑维度

    // 如果尚未确定是否为无批次维度情况，则根据逻辑维度判断
    if (!is_no_batch_dim_case.has_value()) {
      is_no_batch_dim_case = (logical_rank == feature_rank);
    }
    // 确保张量具有批次维度，根据是否有批次维度决定其大小
    auto value_ = ensure_has_bdim(value, bdim.has_value(), batch_size);
    if (!bdim.has_value()) {
      bdim = 0;
    }
    // 如果是无批次维度情况，则将批次维度移至前端
    if (*is_no_batch_dim_case) {
      TORCH_INTERNAL_ASSERT(logical_rank == feature_rank);  // 断言逻辑维度与指定维度相等
      value_ = moveBatchDimToFront(value_, bdim);  // 将批次维度移至前端
      // 如果是指定的连续张量索引，则确保连续性
      if (tensor_idx == contig_tensor_index) {
        value_ = value_.contiguous();  // 确保连续性
      }
      (*stack)[args_begin + tensor_pos[tensor_idx]] = std::move(value_);  // 将处理后的张量值存回栈中
      continue;
    }
    // 否则，逻辑维度应为指定维度加一
    TORCH_INTERNAL_ASSERT(logical_rank == feature_rank + 1);
    value_ = reshape_dim_into(*bdim, 0, value_);  // 将指定维度重塑为第一维度
    // 如果是指定的连续张量索引，则确保连续性
    if (tensor_idx == contig_tensor_index) {
      value_ = value_.contiguous();  // 确保连续性
    }
    (*stack)[args_begin + tensor_pos[tensor_idx]] = std::move(value_);  // 将处理后的张量值存回栈中
  }

  op.callBoxed(stack);  // 调用带有批处理规则的操作

  // 对于每个返回值索引，检查其是否为张量，当前不支持返回非张量值
  for (const auto idx : c10::irange(args_begin, args_begin + num_returns)) {
    const auto& ret = (*stack)[idx];
    TORCH_INTERNAL_ASSERT(ret.isTensor(),
        "This boxed batching rule does not currently support ops that return non-tensor values");
    // 如果 is_no_batch_dim_case 指针所指的值为真，则执行以下代码块
    if (*is_no_batch_dim_case) {
      // 将 ret 转换为张量，并调用 makeBatched 函数创建一个批处理张量，然后将其存入 stack 的索引为 idx 的位置
      (*stack)[idx] = makeBatched(ret.toTensor(), 0, cur_level);
    } else {
      // 否则，调用 reshape_dim_outof 函数将 ret 转换为张量，并在第一个维度上重新塑形为 batch_size，然后再调用 makeBatched 函数创建批处理张量
      // 最后将批处理张量存入 stack 的索引为 idx 的位置
      (*stack)[idx] = makeBatched(reshape_dim_outof(0, batch_size, ret.toTensor()), 0, cur_level);
    }
  }
}

// 用于许多神经网络操作符
// 操作符必须满足以下条件：
// - 所有参数必须接受可选的批处理维度
// - 所有参数必须具有相同的秩（rank）
#define ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED(feature_rank, op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_all_tensors_have_optional_bdim<feature_rank>>());

// 用于特定的操作符，要求其中一个张量是连续的
#define ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(feature_rank, op, contig_tensor_index) \
  m.impl(#op, \
         torch::CppFunction::makeFromBoxedFunction<\
             boxed_all_tensors_have_optional_bdim<\
                 feature_rank, \
                 contig_tensor_index>\
             >());

// 定义模板类和结构体以帮助现有批处理规则
template <typename A, A a, typename C>
struct ExistingBdimBatchRuleHelper;

template <typename F, F Func, typename A, typename... T>
struct ExistingBdimBatchRuleHelper<F, Func, c10::guts::typelist::typelist<A, T...>> {
  // 应用现有批处理规则的静态方法
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& self,
      optional<int64_t> self_bdim,
      T... extra_args) {
    auto self_ = reshape_dim_into(*self_bdim, 0, self);  // 将张量自身的维度重塑为特定维度
    auto out = Func(self_, std::forward<T>(extra_args)...);  // 调用给定的函数对象并传递额外参数
    return std::make_tuple(reshape_dim_outof_symint(0, self.sym_sizes()[*self_bdim], out), 0);  // 将输出张量的维度重塑为特定维度
  }
};

// 宏定义，用于创建现有批处理规则
#define EXISTING_BDIM_BATCH_RULE(fn) SINGLE_ARG(\
    ExistingBdimBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

// 宏定义，支持现有批处理规则
#define EXISTING_BDIM(op) \
  VMAP_SUPPORT(op, EXISTING_BDIM_BATCH_RULE(ATEN_FN(op)));

// 宏定义，支持带有重载的现有批处理规则
#define EXISTING_BDIM2(op, overload) \
  VMAP_SUPPORT2(op, overload, EXISTING_BDIM_BATCH_RULE(ATEN_FN2(op, overload)));

// 调用类成员函数的辅助函数模板
#define INVOKE(object,ptrToMember)  ((object).*(ptrToMember))

// 定义模板函数，用于一元就地操作的批处理规则
template <typename F, F Method, typename... ExtraArgs>
Tensor& unary_inplace_batch_rule(Tensor& self, optional<int64_t>, ExtraArgs... extra_args) {
  INVOKE(self, Method)(std::forward<ExtraArgs>(extra_args)...);  // 调用成员函数并传递额外参数
  return self;
}

// 获取批处理维度的大小，处理四个张量的情况
inline int64_t get_bdim_size4(
    const Tensor& a_value, optional<int64_t> a_bdim,
    const Tensor& b_value, optional<int64_t> b_bdim,
    const Tensor& c_value, optional<int64_t> c_bdim,
    const Tensor& d_value, optional<int64_t> d_bdim) {
  if (a_bdim)
    return a_value.size(*a_bdim);  // 返回指定维度的张量大小
  if (b_bdim)
    return b_value.size(*b_bdim);
  if (c_bdim)
    return c_value.size(*c_bdim);
  if (d_bdim)
    return d_value.size(*d_bdim);
  TORCH_INTERNAL_ASSERT(false);  // 如果没有找到批处理维度，则抛出内部断言错误
}

// 获取批处理维度的大小，处理三个张量的情况
inline int64_t get_bdim_size3(
    const Tensor& a_value, optional<int64_t> a_bdim,
    const Tensor& b_value, optional<int64_t> b_bdim,
    const Tensor& c_value, optional<int64_t> c_bdim) {
  if (a_bdim)
    return a_value.size(*a_bdim);  // 返回指定维度的张量大小
  if (b_bdim)
    return b_value.size(*b_bdim);
  if (c_bdim)
    return c_value.size(*c_bdim);
  TORCH_INTERNAL_ASSERT(false);  // 如果没有找到批处理维度，则抛出内部断言错误
}
// 计算张量维度大小，根据可选的批处理维度选择具体维度
inline int64_t get_bdim_size2(
    const Tensor& a_value, optional<int64_t> a_bdim,
    const Tensor& b_value, optional<int64_t> b_bdim) {
  // 如果 a_bdim 已指定，返回 a_value 在该维度上的大小
  if (a_bdim)
    return a_value.size(*a_bdim);
  // 如果 b_bdim 已指定，返回 b_value 在该维度上的大小
  if (b_bdim)
    return b_value.size(*b_bdim);
  // 如果未指定任何维度信息，报错
  TORCH_INTERNAL_ASSERT(false);
}

// 生成从 start 到 stop-1 的整数序列
inline VmapDimVector range(int64_t start, int64_t stop) {
  // 确保 stop 大于等于 start
  TORCH_INTERNAL_ASSERT(stop >= start);
  // 创建一个向量 dims 来存放整数序列
  VmapDimVector dims;
  // 预先分配向量的容量为 stop - start
  dims.reserve(stop - start);
  // 循环从 start 到 stop-1，将每个整数加入 dims
  for (int64_t i = start; i < stop; i++) {
    dims.emplace_back(i);
  }
  // 返回生成的整数序列向量 dims
  return dims;
}
std::tuple<Tensor, Tensor> _binary_pointwise_helper(
    const Tensor& tensor, optional<int64_t> tensor_batch_dim, const Tensor& other, optional<int64_t> other_batch_dim,
    bool do_type_promotion=true);

} // namespace at::functorch
```