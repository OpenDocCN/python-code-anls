# `.\pytorch\aten\src\ATen\functorch\BatchRulesFactory.cpp`

```
// 头文件包含，引入所需的库文件
#include <ATen/functorch/BatchRulesHelper.h>
#include <c10/core/SymIntArrayRef.h>

// 命名空间 at::functorch 下的定义
namespace at::functorch {

// 模板定义：NewBlahBatchRuleHelperSymInt，带有符号整数数组参考的批处理规则助手
template <typename A, A a, typename C>
struct NewBlahBatchRuleHelperSymInt;

// 模板特化：NewBlahBatchRuleHelperSymInt，用于函数 Func，带有类型 A、B 和参数包 T
template <typename F, F Func, typename A, typename B, typename... T>
struct NewBlahBatchRuleHelperSymInt<F, Func, typelist<A, B, T...>> {
  // 应用函数：返回张量和可选的整数
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      SymIntArrayRef shape,
      T... extra_args) {
    // 计算批次维度的大小
    const auto bdim_size = tensor.sym_size(batch_dim.value());
    // 创建符号整数向量 new_shape，并预留空间
    c10::SmallVector<c10::SymInt> new_shape;
    new_shape.reserve(shape.size() + 1);
    // 将批次维度大小作为新形状的第一个元素
    new_shape.emplace_back(bdim_size);
    // 将原始形状的元素插入到 new_shape 的末尾
    new_shape.insert(new_shape.end(), shape.begin(), shape.end());
    // 调用 Func 函数，并返回结果作为元组
    return std::make_tuple(Func(tensor, new_shape, std::forward<T>(extra_args)...), 0);
  }
};

// 模板定义：NewBlahBatchRuleHelper，带有整数数组参考的批处理规则助手
template <typename A, A a, typename C>
struct NewBlahBatchRuleHelper;

// 模板特化：NewBlahBatchRuleHelper，用于函数 Func，带有类型 A、B 和参数包 T
template <typename F, F Func, typename A, typename B, typename... T>
struct NewBlahBatchRuleHelper<F, Func, typelist<A, B, T...>> {
  // 应用函数：返回张量和可选的整数
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      IntArrayRef shape,
      T... extra_args) {
    // 计算批次维度的大小
    const auto bdim_size = tensor.size(batch_dim.value());
    // 创建整数向量 new_shape，并预留空间
    VmapDimVector new_shape;
    new_shape.reserve(shape.size() + 1);
    // 将批次维度大小作为新形状的第一个元素
    new_shape.emplace_back(bdim_size);
    // 将原始形状的元素插入到 new_shape 的末尾
    new_shape.insert(new_shape.end(), shape.begin(), shape.end());
    // 调用 Func 函数，并返回结果作为元组
    return std::make_tuple(Func(tensor, new_shape, std::forward<T>(extra_args)...), 0);
  }
};

// 宏定义：NEW_BLAH_BATCH_RULE，生成调用 NewBlahBatchRuleHelper 的代码片段
#define NEW_BLAH_BATCH_RULE(fn) SINGLE_ARG(\
    NewBlahBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

// 宏定义：NEW_BLAH_BATCH_RULE_SYMINT，生成调用 NewBlahBatchRuleHelperSymInt 的代码片段
#define NEW_BLAH_BATCH_RULE_SYMINT(fn) SINGLE_ARG(\
    NewBlahBatchRuleHelperSymInt<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

// 函数定义：_new_zeros_with_same_feature_meta_batch_rule，返回张量和可选的整数
static std::tuple<Tensor,optional<int64_t>> _new_zeros_with_same_feature_meta_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim,
    int64_t self_num_batch_dims) {
  // 别名定义：base 为 other，base_bdim 为 other_bdim，tangent 为 self，tangent_bdim 为 self_bdim
  const auto& base = other;
  const auto& base_bdim = other_bdim;
  const auto& tangent = self;
  const auto& tangent_bdim = self_bdim;

  // 三种情况的注释
  // Case 1  Case 2  Case 3
  // base        [6]  [B, 6]  [B, 6]
  // tangent  [B, 5]     [5]  [B, 5]
  // result   [B, 6]  [B, 6]  [B, 6]

  // 情况 2 和 3
  if (base_bdim) {
    // 将 base 张量的批量维度移到最前面
    auto base_ = moveBatchDimToFront(base, base_bdim);
    // 创建 tangent_ 张量作为副本
    Tensor tangent_ = tangent;
    // 如果 tangent_bdim 有值
    if (tangent_bdim.has_value()) {
      // tangent  [B, K0, K1, 5]
      // base_            [B, 6]
      // 我们希望将 B 移动到 Ks 之后，这样 self_num_batch_dims（实际上是 tangent_num_batch_dims）不会受到干扰。
      // [B, K0, K1, 6] -> [K0, K1, B, 6]
      //
      // [K0, K1, B, 6], [B, 5], 2 -> [K0, K1, B, 5]
      // 将 tangent_ 张量的批量维度移动到指定位置，以保持一致性
      tangent_ = tangent.movedim(*tangent_bdim, self_num_batch_dims);
    }
    // 使用 tangent_ 和 base_ 创建新的零张量，保持相同的特征元数据
    const auto result = at::_new_zeros_with_same_feature_meta(tangent_, base_, self_num_batch_dims);
    // 返回结果和 self_num_batch_dims
    return std::make_tuple(result, self_num_batch_dims);
  }

  // Case 1:
  // 将 tangent 张量的批量维度移到最前面
  auto tangent_ = moveBatchDimToFront(tangent, tangent_bdim);
  // 使用 tangent_ 和 base 创建新的零张量，扩展 self_num_batch_dims 后的维度
  auto result = at::_new_zeros_with_same_feature_meta(tangent_, base, self_num_batch_dims + 1);
  // 返回结果和 0
  return std::make_tuple(result, 0);
// 定义一个静态函数，用于辅助计算 linspace 和 logspace 的批处理规则
static std::tuple<Tensor,optional<int64_t>> linspace_logspace_batch_rule_helper(
    // 开始张量和可能的批处理维度
    const at::Tensor& start, optional<int64_t> start_bdim,
    // 结束张量和可能的批处理维度
    const at::Tensor& end, optional<int64_t> end_bdim,
    // 步数
    int64_t steps,
    // 可选的底数（用于 logspace）
    std::optional<double> base,
    // 可选的数据类型
    std::optional<at::ScalarType> dtype,
    // 可选的布局
    std::optional<at::Layout> layout,
    // 可选的设备
    std::optional<at::Device> device,
    // 可选的内存钉住标志
    std::optional<bool> pin_memory)
{
  // 获取批处理大小
  auto batch_size = get_bdim_size2(start, start_bdim, end, end_bdim);
  // 确保张量具有批处理维度
  auto start_ = ensure_has_bdim(start, start_bdim.has_value(), batch_size);
  auto end_ = ensure_has_bdim(end, end_bdim.has_value(), batch_size);
  // 将批处理维度移动到张量的最前面
  start_ = moveBatchDimToFront(start_, start_bdim);
  end_ = moveBatchDimToFront(end_, end_bdim);

  // 设置张量选项，包括数据类型、布局、设备和内存钉住标志
  auto tensor_options = at::TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 定义结果张量
  Tensor result;
  // 根据步数选择相应的操作
  if (steps == 0){
    // 如果步数为0，返回全零张量
    result = at::full({batch_size, 0}, 0, tensor_options);
  } else if (steps == 1){
    // 如果步数为1，返回与开始张量形状相同的张量，并复制开始张量的值，并在第二维度上增加一个维度
    result = start_.new_empty({batch_size}, tensor_options).copy_(start_).unsqueeze(1);
  } else {
    // 否则，根据起始张量、步数和结束张量，生成等间距的张量序列
    result = (start_ + at::arange(0, steps, tensor_options).unsqueeze_(1) * (end_ - start_) / (steps - 1)).transpose(0, 1);
  }

  // 如果有指定底数，则将结果张量应用指数运算
  if (base){
    result = at::pow(*base, result);
  }

  // 如果指定了数据类型且结果张量的数据类型与指定的数据类型不同，则将结果张量转换为指定数据类型
  if (dtype && result.scalar_type() != *dtype){
    result = result.to(*dtype);
  }

  // 返回结果张量和可选的整数0
  return std::make_tuple(result, 0);
}

// 定义一个静态函数，用于处理两个张量之间的 linspace 批处理规则
static std::tuple<Tensor,optional<int64_t>> linspace_Tensor_Tensor_batch_rule(
    // 开始张量和可能的批处理维度
    const at::Tensor& start, optional<int64_t> start_bdim,
    // 结束张量和可能的批处理维度
    const at::Tensor& end, optional<int64_t> end_bdim,
    // 步数
    int64_t steps,
    // 可选的数据类型
    std::optional<at::ScalarType> dtype,
    // 可选的布局
    std::optional<at::Layout> layout,
    // 可选的设备
    std::optional<at::Device> device,
    // 可选的内存钉住标志
    std::optional<bool> pin_memory){
  // 调用 linspace_logspace_batch_rule_helper 函数，并返回其结果
  return linspace_logspace_batch_rule_helper(start, start_bdim, end, end_bdim, steps, c10::nullopt, dtype, layout, device, pin_memory);
}

// 定义一个静态函数，用于处理一个张量和一个标量之间的 linspace 批处理规则
static std::tuple<Tensor,optional<int64_t>> linspace_Tensor_Scalar_batch_rule(
    // 开始张量和可能的批处理维度
    const at::Tensor& start, optional<int64_t> start_bdim,
    // 结束标量
    const at::Scalar& end,
    // 步数
    int64_t steps,
    // 可选的数据类型
    std::optional<at::ScalarType> dtype,
    // 可选的布局
    std::optional<at::Layout> layout,
    // 可选的设备
    std::optional<at::Device> device,
    // 可选的内存钉住标志
    std::optional<bool> pin_memory){

  // 将结束标量转换为张量
  auto end_t = at::native::wrapped_scalar_tensor(end, start.device());
  // 调用 linspace_logspace_batch_rule_helper 函数，并返回其结果
  return linspace_logspace_batch_rule_helper(start, start_bdim, end_t, c10::nullopt, steps, c10::nullopt, dtype, layout, device, pin_memory);
}

// 定义一个静态函数，用于处理一个标量和一个张量之间的 linspace 批处理规则
static std::tuple<Tensor,optional<int64_t>> linspace_Scalar_Tensor_batch_rule(
    // 开始标量
    const at::Scalar& start,
    // 结束张量和可能的批处理维度
    const at::Tensor& end, optional<int64_t> end_bdim,
    // 步数
    int64_t steps,
    // 可选的数据类型
    std::optional<at::ScalarType> dtype,
    // 可选的布局
    std::optional<at::Layout> layout,
    // 可选的设备
    std::optional<at::Device> device,
    // 可选的内存钉住标志
    std::optional<bool> pin_memory){

  // 将开始标量转换为张量
  auto start_t = at::native::wrapped_scalar_tensor(start, end.device());
  // 调用 linspace_logspace_batch_rule_helper 函数，并返回其结果
  return linspace_logspace_batch_rule_helper(start_t, c10::nullopt, end, end_bdim, steps, c10::nullopt, dtype, layout, device, pin_memory);
}
# 定义一个静态函数，用于处理 logspace 函数中 Tensor 到 Tensor 参数的批处理规则
static std::tuple<Tensor,optional<int64_t>> logspace_Tensor_Tensor_batch_rule(
    const at::Tensor& start, optional<int64_t> start_bdim,
    const at::Tensor& end, optional<int64_t> end_bdim,
    int64_t steps,
    double base,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory){
  # 调用帮助函数，将参数传递给其处理，并返回结果
  return linspace_logspace_batch_rule_helper(start, start_bdim, end, end_bdim, steps, c10::make_optional(base), dtype, layout, device, pin_memory);
}

# 定义一个静态函数，用于处理 logspace 函数中 Tensor 到 Scalar 参数的批处理规则
static std::tuple<Tensor,optional<int64_t>> logspace_Tensor_Scalar_batch_rule(
    const at::Tensor& start, optional<int64_t> start_bdim,
    const at::Scalar& end,
    int64_t steps,
    double base,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory){

  # 将 Scalar 类型的 end 参数转换为 Tensor 类型
  auto end_t = at::native::wrapped_scalar_tensor(end, start.device());
  # 调用帮助函数，将参数传递给其处理，并返回结果
  return linspace_logspace_batch_rule_helper(start, start_bdim, end_t, c10::nullopt, steps, c10::make_optional(base), dtype, layout, device, pin_memory);
}

# 定义一个静态函数，用于处理 logspace 函数中 Scalar 到 Tensor 参数的批处理规则
static std::tuple<Tensor,optional<int64_t>> logspace_Scalar_Tensor_batch_rule(
    const at::Scalar& start,
    const at::Tensor& end, optional<int64_t> end_bdim,
    int64_t steps,
    double base,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory){

  # 将 Scalar 类型的 start 参数转换为 Tensor 类型
  auto start_t = at::native::wrapped_scalar_tensor(start, end.device());
  # 调用帮助函数，将参数传递给其处理，并返回结果
  return linspace_logspace_batch_rule_helper(start_t, c10::nullopt, end, end_bdim, steps, c10::make_optional(base), dtype, layout, device, pin_memory);
}

# 定义一个静态函数，用于判断两个 Tensor 是否具有相同的存储大小
static bool _has_same_storage_numel_batch_rule(const Tensor& a, const Tensor& b) {
  # 总是返回 true，因为此函数的作用是规则检查
  return true;
}
// 实现 Torch 库中 aten 命名空间的 FuncTorchBatched 的函数 TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 注册函数 "_has_same_storage_numel" 的批处理规则为 _has_same_storage_numel_batch_rule
  m.impl("_has_same_storage_numel", _has_same_storage_numel_batch_rule);
  // 支持函数 ones_like 的批处理，基本一元批处理规则是 ATEN_FN(ones_like)
  VMAP_SUPPORT(ones_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(ones_like)));
  // 支持函数 zeros_like 的批处理，基本一元批处理规则是 ATEN_FN(zeros_like)
  VMAP_SUPPORT(zeros_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(zeros_like)));
  // 支持函数 empty_like 的批处理，基本一元批处理规则是 ATEN_FN(empty_like)
  VMAP_SUPPORT(empty_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(empty_like)));
  // 支持函数 randn_like 的批处理，基本一元批处理规则是 ATEN_FN(randn_like)
  VMAP_SUPPORT(randn_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(randn_like)));
  // 支持函数 rand_like 的批处理，基本一元批处理规则是 ATEN_FN(rand_like)
  VMAP_SUPPORT(rand_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(rand_like)));
  // 支持函数 full_like 的批处理，基本一元批处理规则是 ATEN_FN(full_like)
  VMAP_SUPPORT(full_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(full_like)));
  // 支持函数 new_empty 的批处理，新的批处理规则是 NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_empty))
  VMAP_SUPPORT(new_empty, NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_empty)));
  // 支持函数 new_zeros 的批处理，新的批处理规则是 NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_zeros)))
  VMAP_SUPPORT(new_zeros, NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_zeros)));
  // 支持函数 new_ones 的批处理，新的批处理规则是 NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_ones)))
  VMAP_SUPPORT(new_ones, NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_ones)));
  // 支持函数 new_full 的批处理，新的批处理规则是 NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_full)))
  VMAP_SUPPORT(new_full, NEW_BLAH_BATCH_RULE_SYMINT(ATEN_FN(new_full)));
  // 支持函数 linspace 的批处理，输入类型为 Tensor_Tensor，批处理规则是 linspace_Tensor_Tensor_batch_rule
  VMAP_SUPPORT2(linspace, Tensor_Tensor, linspace_Tensor_Tensor_batch_rule);
  // 支持函数 linspace 的批处理，输入类型为 Tensor_Scalar，批处理规则是 linspace_Tensor_Scalar_batch_rule
  VMAP_SUPPORT2(linspace, Tensor_Scalar, linspace_Tensor_Scalar_batch_rule);
  // 支持函数 linspace 的批处理，输入类型为 Scalar_Tensor，批处理规则是 linspace_Scalar_Tensor_batch_rule
  VMAP_SUPPORT2(linspace, Scalar_Tensor, linspace_Scalar_Tensor_batch_rule);
  // 支持函数 logspace 的批处理，输入类型为 Tensor_Tensor，批处理规则是 logspace_Tensor_Tensor_batch_rule
  VMAP_SUPPORT2(logspace, Tensor_Tensor, logspace_Tensor_Tensor_batch_rule);
  // 支持函数 logspace 的批处理，输入类型为 Tensor_Scalar，批处理规则是 logspace_Tensor_Scalar_batch_rule
  VMAP_SUPPORT2(logspace, Tensor_Scalar, logspace_Tensor_Scalar_batch_rule);
  // 支持函数 logspace 的批处理，输入类型为 Scalar_Tensor，批处理规则是 logspace_Scalar_Tensor_batch_rule
  VMAP_SUPPORT2(logspace, Scalar_Tensor, logspace_Scalar_Tensor_batch_rule);
  // 支持函数 _new_zeros_with_same_feature_meta 的批处理，批处理规则是 _new_zeros_with_same_feature_meta_batch_rule
  VMAP_SUPPORT(_new_zeros_with_same_feature_meta, _new_zeros_with_same_feature_meta_batch_rule);
  // 由于参数不规则，如 randint 需要额外的 int 参数，因此不确定如何将其干净地添加到批处理中
  // （这里没有提供具体的批处理规则）
}
} // 结束命名空间 at::functorch
```