# `.\pytorch\aten\src\ATen\functorch\BatchRulesRandomness.cpp`

```py
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 functorch 动态层的头文件
#include <ATen/functorch/DynamicLayer.h>
// 包含 functorch 批处理规则助手的头文件
#include <ATen/functorch/BatchRulesHelper.h>

#include <utility>

// 此文件包含了针对随机操作的批处理规则。这些规则与常规的批处理规则不同：
// 常规的批处理规则注册到 FuncTorchBatched 键上，而随机操作的批处理规则注册到 FuncTorchVmapMode 上。
// 这是因为我们需要在随机操作上进行干预，即使它们不在批处理张量上。
namespace at::functorch {

// 模板函数：random_batching_rule
template <typename F, F Func, typename... ExtraArgs>
Tensor random_batching_rule(SymIntArrayRef shape, ExtraArgs... extra_args) {
  // 临时排除 DispatchKey 为 FuncTorchVmapMode 的保护区域
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前可能存在的动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 创建 SymInt 数组 shapeVec，大小为 1，初始为当前动态层的批处理大小
  c10::SmallVector<SymInt> shapeVec(1, maybe_layer->batchSize());
  shapeVec.reserve(shape.size() + 1);
  shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());
  // 获取当前动态层的随机性类型
  RandomnessType randomness = maybe_layer->randomness();
  // 检查随机性的有效性
  check_randomness(randomness);
  // 根据随机性类型决定如何处理数据
  if (randomness == RandomnessType::Different) {
    // 对于不同的随机性类型，使用 makeBatched 函数进行批处理
    return makeBatched(Func(shapeVec, std::forward<ExtraArgs>(extra_args)...), 0, maybe_layer->layerId());
  } else {
    // 对于相同的随机性类型，直接调用 Func 函数
    return Func(shape, std::forward<ExtraArgs>(extra_args)...);
  }
}

// 模板函数：random_inplace_batching_rule
template <typename F, F Func, typename... ExtraArgs>
Tensor& random_inplace_batching_rule(Tensor& self, ExtraArgs... extra_args) {
  // 临时排除 DispatchKey 为 FuncTorchVmapMode 的保护区域
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前可能存在的动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 获取当前层级的 ID
  const auto cur_level = maybe_layer->layerId();
  // 解包当前层级下的张量 self
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  // 将批处理维度移动到张量 self_value 的最前面
  self_value = moveBatchDimToFront(self_value, self_bdim);
  // 获取当前动态层的随机性类型
  RandomnessType randomness = maybe_layer->randomness();
  // 检查随机性的有效性
  check_randomness(randomness);
  // 如果随机性为不同类型且 self_bdim 为假，抛出错误信息
  TORCH_CHECK(
    !(randomness == RandomnessType::Different && !self_bdim),
    "vmap: Cannot ask for different inplace randomness on an unbatched tensor. This will appear like same randomness. ",
    "If this is necessary for your usage, please file an issue with functorch.");
  // 根据随机性类型决定如何处理数据
  if (randomness == RandomnessType::Same && self_bdim) {
    // 对于相同的随机性类型且存在批处理维度，创建一个临时张量 intermediate
    auto intermediate = empty(self.sizes(), self.options());
    // 调用 Func 函数填充 intermediate
    Func(intermediate, std::forward<ExtraArgs>(extra_args)...);
    // 将 intermediate 的值复制给 self，批处理应该使得这一过程正常工作
    self.copy_(intermediate);
    return self;
  } else {
    // 对于其他情况，直接调用 Func 函数处理 self_value
    Func(self_value, std::forward<ExtraArgs>(extra_args)...);
    return self;
  }
}
// 实现了在 Torch 中 Bernoulli 分布的原位操作规则，支持批处理
static Tensor& bernoulli_inplace_Tensor_batching_rule(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  // 排除 TorchVmapMode 调度键，确保不会与 vmap 冲突
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前动态层，以及层级标识和随机性类型
  auto maybe_layer = maybeCurrentDynamicLayer();
  auto cur_level = maybe_layer->layerId();
  RandomnessType randomness = maybe_layer->randomness();

  // 解包当前张量的值和批处理维度
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);

  // 解包概率张量的值和批处理维度
  auto [other_value, other_bdim] = unwrapTensorAtLevel(p_, cur_level);

  // 检查随机性设置是否匹配
  check_randomness(randomness, other_bdim.has_value());

  // 如果自身没有批处理维度但概率张量有，则抛出错误
  if (!self_bdim && other_bdim) {
    vmapIncompatibleInplaceError("inplace bernoulli");
  }

  // 计算最大逻辑秩
  auto self_logical_rank = rankWithoutBatchDim(self_value, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other_value, other_bdim);
  auto max_logical_rank = std::max(self_logical_rank, other_logical_rank);

  // 将批处理维度移到张量前部
  auto self_ = moveBatchDimToFront(self_value, self_bdim);
  auto other_ = moveBatchDimToFront(other_value, other_bdim);

  // 如果维度不对齐，需要进行填充操作以达到相同逻辑秩
  self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_bdim, max_logical_rank);

  // 检查是否可以进行不同随机性类型的 Bernoulli 操作，否则报错
  TORCH_CHECK(
    !(randomness == RandomnessType::Different && !self_bdim),
    "vmap: Cannot ask for different inplace randomness on an unbatched tensor. This will appear like same randomness. ",
    "If this is necessary for your usage, please file an issue with functorch.");

  // 根据随机性类型选择不同的操作方式
  if (randomness == RandomnessType::Same && self_bdim) {
    // 创建一个空张量作为中间结果，使用概率张量进行 Bernoulli 操作，然后复制到自身
    auto intermediate = empty(self.sizes(), self.options());
    intermediate.bernoulli_(other_, std::move(gen));
    self.copy_(intermediate); // batching should make this just work out...
    return self;
  } else {
    // 使用概率张量进行 Bernoulli 操作，直接在自身上修改
    self_.bernoulli_(other_, std::move(gen));
    return self;
  }
}

// 实现了在 Torch 中的 randperm 操作的批处理规则
template <typename F, F Func, typename... ExtraArgs>
Tensor randperm_batching_rule(int64_t n, ExtraArgs... extra_args) {
  // 排除 TorchVmapMode 调度键，确保不会与 vmap 冲突
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前动态层，以及批处理大小和随机性类型
  auto maybe_layer = maybeCurrentDynamicLayer();
  auto const batch_size = maybe_layer->batchSize();
  RandomnessType randomness = maybe_layer->randomness();
  check_randomness(randomness);

  // 如果需要不同随机性类型，使用循环生成多个 randperm 张量并堆叠返回
  if (randomness == RandomnessType::Different) {
    std::vector<at::Tensor> stackedList(batch_size.guard_int(__FILE__, __LINE__));
    for (int64_t idx = 0; idx < batch_size; ++idx) {
      // 在循环中生成 randperm 张量，并将其存入 stackedList
      stackedList[idx] = Func(n, extra_args...);
    }
    // 使用 makeBatched 函数将 stackedList 堆叠成批处理形式的张量
    return makeBatched(at::stack(stackedList), 0, maybe_layer->layerId());
  } else {
    // 使用单个 randperm 操作生成张量
    return Func(n, std::forward<ExtraArgs>(extra_args)...);
  }
}
// 在函数签名中声明了一个模板函数，接受一个张量和额外参数列表，返回一个张量
Tensor unary_pointwise_random_batch_rule(const Tensor& tensor, ExtraArgs... extra_args) {
  // 进入临界区，排除 FuncTorchVmapMode 分发键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 获取当前层级的标识符
  const auto cur_level = maybe_layer->layerId();

  // 解包张量和批维度信息
  auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(tensor, cur_level);
  // 将批维度移动到张量的最前面
  tensor_value = moveBatchDimToFront(tensor_value, tensor_bdim);

  // 获取随机性类型
  RandomnessType randomness = maybe_layer->randomness();
  // 检查随机性
  check_randomness(randomness, tensor_bdim.has_value());

  // 获取张量的形状
  auto shape = tensor_value.sizes();
  // 创建一个具有 maybe_layer->batchSize() 大小的 VmapSymDimVector
  VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
  shapeVec.reserve(shape.size() + 1);
  shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());

  // 如果随机性为 Different 并且没有批维度，则扩展张量
  if (randomness == RandomnessType::Different && !tensor_bdim) {
    tensor_value = tensor_value.expand_symint(shapeVec);
  }

  // 调用 Func 函数处理张量值及其它参数
  auto out = Func(tensor_value, std::forward<ExtraArgs>(extra_args)...);

  // 如果随机性为 Same 并且没有批维度，则直接返回结果
  if (randomness == RandomnessType::Same && !tensor_bdim) {
    return out;
  }

  // 否则，对输出结果进行批处理并返回
  return makeBatched(out, 0, cur_level);
}

// 定义一个模板函数，接受一个张量和一个函数指针 Func，以及额外的参数列表，返回一个张量
template<typename F, F Func, typename... ExtraArgs>
Tensor tensor_like_random_batch_rule(const Tensor& self, ExtraArgs... extra_args) {
  // 进入临界区，排除 FuncTorchVmapMode 分发键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 获取当前层级的标识符
  const auto cur_level = maybe_layer->layerId();
  // 获取随机性类型
  RandomnessType randomness = maybe_layer->randomness();
  // 检查随机性
  check_randomness(randomness);

  // 解包张量和批维度信息
  auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(self, cur_level);
  // 将批维度移动到张量的最前面
  tensor_value = moveBatchDimToFront(tensor_value, tensor_bdim);

  // 根据随机性类型进行相应操作
  if (randomness == RandomnessType::Same && tensor_bdim) {
    tensor_value = tensor_value[0];
  } else if (randomness == RandomnessType::Different && !tensor_bdim) {
    // 获取张量的形状
    auto shape = tensor_value.sizes();
    // 创建一个具有 maybe_layer->batchSize() 大小的 VmapSymDimVector
    VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
    shapeVec.reserve(shape.size() + 1);
    shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());
    // 扩展张量
    tensor_value = tensor_value.expand_symint(shapeVec);
  }

  // 调用 Func 函数处理张量值及其它参数
  auto res = Func(tensor_value, std::forward<ExtraArgs>(extra_args)...);

  // 根据随机性类型返回结果
  return (randomness == RandomnessType::Same) ? res : makeBatched(res, 0, cur_level);
}

// 定义一个静态函数，返回一个包含两个张量的元组
static std::tuple<Tensor,Tensor> native_dropout_batching_rule(const Tensor& tensor, double p, std::optional<bool> train) {
  // 进入临界区，排除 FuncTorchVmapMode 分发键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 获取当前层级的标识符
  const auto cur_level = maybe_layer->layerId();
  // 获取随机性类型
  RandomnessType randomness = maybe_layer->randomness();

  // 解包张量和批维度信息
  auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(tensor, cur_level);
  // 将批维度移动到张量的最前面
  tensor_value = moveBatchDimToFront(tensor_value, tensor_bdim);

  // 如果 train 没有值或者 train 为真，则检查随机性
  if (!train.has_value() || train) {
    check_randomness(randomness); // 如果处于评估模式，我们不关心随机性
  }

  // 根据条件执行不同的操作
  if ((train.has_value() && !train) ||
      randomness == RandomnessType::Different) {
    if (!tensor_bdim) {
      // 如果张量未经批处理，则在调用 dropout 前添加批处理维度。
      auto shape = tensor_value.sizes();
      VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
      shapeVec.reserve(shape.size() + 1);
      shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());
      // 扩展张量的符号整数形状
      tensor_value = tensor_value.expand_symint(shapeVec);
    }
    // 调用原生的 dropout 函数，返回输出张量 output 和掩码 mask
    auto [output, mask] = at::native_dropout(tensor_value, p, train);
    // 返回由 output 和 mask 组成的元组，其中 output 和 mask 都经过了批处理
    return std::make_tuple(
        makeBatched(output, 0, cur_level),
        makeBatched(mask, 0, cur_level));
  }

  // 从 CPU 内核复制的重复代码，因为 CUDA 版本没有显式调用 bernoulli_
  // 计算概率为 1-p 的值，避免除零和 NaN 结果
  double p1m = 1. - p;
  double scale = p1m == 0 ? 0. : 1. / p1m;
  // 创建与 tensor 类型和大小相同的空 mask 张量
  Tensor mask = at::empty_like(tensor, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 在 mask 上应用 Bernoulli 分布生成随机二进制掩码
  mask.bernoulli_(p1m);
  // 计算 dropout 后的输出张量，并将其缩放以保持期望值
  const auto output = tensor.mul(mask).mul_(scale);
  // 返回由 output 和 mask 组成的元组，表示 dropout 操作后的结果
  return std::make_tuple(output, mask);
}

// 定义静态函数 `multinomial_batching_rule`，用于在批处理模式下执行多项式抽样
static Tensor multinomial_batching_rule(const Tensor& self, const int64_t num_samples, const bool replacement, const std::optional<Generator> generator) {
  // 排除 DispatchKey 为 FuncTorchVmapMode 的调度键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  // 获取当前动态层，如果存在
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 获取当前层级的 ID
  const auto cur_level = maybe_layer->layerId();

  // 在当前层级解包张量 self，获取值和批处理维度
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  // 将批处理维度移动到张量值的前面
  self_value = moveBatchDimToFront(self_value, self_bdim);

  // 获取当前层级的随机性类型
  RandomnessType randomness = maybe_layer->randomness();
  // 检查随机性是否有效，并且与批处理维度相关联
  check_randomness(randomness, self_bdim.has_value());

  // 如果随机性为 Different
  if (randomness == RandomnessType::Different) {
    // 判断是否为二维情况
    const auto is_2D_case = rankWithoutBatchDim(self_value, self_bdim) == 2;
    // 如果没有批处理维度，确保将其加入到张量值中
    if (!self_bdim.has_value()) {
      self_value = ensure_has_bdim(self_value, self_bdim.has_value(), maybe_layer->batchSize());
    }
    // 如果是二维情况，重塑张量维度
    if (is_2D_case) {
      self_value = reshape_dim_into(0, 0, self_value);
    }
    // 执行多项式抽样操作，并返回结果
    auto out = multinomial(self_value, num_samples, replacement, generator);
    // 如果是二维情况，重新塑造输出维度
    if (is_2D_case) {
      out = reshape_dim_outof_symint(0, maybe_layer->batchSize(), out);
    }
    // 将输出结果进行批处理，并返回结果
    return makeBatched(out, 0, cur_level);
  }

  // 如果随机性为 Same
  TORCH_INTERNAL_ASSERT(randomness == RandomnessType::Same); // 检查随机性不会引起错误随机性
  TORCH_INTERNAL_ASSERT(!self_bdim.has_value()); // 检查随机性不会与批处理输入相关联
  // 必须是与未批处理输入具有相同随机性的情况
  // 1D 情况：直接执行多项式抽样操作，并返回结果
  // 2D 情况：直接执行多项式抽样操作，并返回结果
  return multinomial(self_value, num_samples, replacement, generator);
}

// 模板定义：RandomBatchRuleHelper
template <typename A, A a, typename C>
struct RandomBatchRuleHelper;

// 模板定义：RandomBatchRuleHelper 特化实现
template <typename F, F Func, typename T1, typename... T>
struct RandomBatchRuleHelper<F, Func, typelist<T1, T...>> {
  // 应用随机批处理规则，返回张量
  static Tensor apply(SymIntArrayRef shape, T... extra_args) {
    return random_batching_rule<F, Func, T...>(shape, std::forward<T>(extra_args)...);
  }
};

// 模板定义：rand_int_wrapper
template <typename F, F Func, typename... T>
Tensor rand_int_wrapper(SymIntArrayRef shape, c10::SymInt high, T... extra_args) {
  // 调用函数对象 Func，执行随机整数包装操作，并返回结果张量
  return Func(high, shape, std::forward<T>(extra_args)...);
}

// 模板定义：RandomInplaceBatchRuleHelper
template <typename A, A a, typename C>
struct RandomInplaceBatchRuleHelper;

// 模板定义：RandomInplaceBatchRuleHelper 特化实现
template <typename F, F Func, typename T1, typename... T>
struct RandomInplaceBatchRuleHelper<F, Func, typelist<T1, T...>> {
  // 应用随机批处理规则，返回操作后的张量的引用
  static Tensor& apply(Tensor& self, T... extra_args) {
    return random_inplace_batching_rule<F, Func, T...>(self, std::forward<T>(extra_args)...);
  }
};

// 模板定义：RandIntBatchRuleHelper
template <typename A, A a, typename C>
struct RandIntBatchRuleHelper;

// 模板定义：RandIntBatchRuleHelper 特化实现
template <typename F, F Func, typename T1, typename T2, typename... T>
struct RandIntBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  // 应用随机整数批处理规则，返回结果张量
  static Tensor apply(c10::SymInt high, SymIntArrayRef shape, T... extra_args) {
    return random_batching_rule<decltype(&rand_int_wrapper<F, Func, T...>),
                                &rand_int_wrapper<F, Func, T...>,
                                c10::SymInt, T...>(shape, std::move(high), std::forward<T>(extra_args)...);


    # 返回一个随机批处理规则的结果
    # 使用 decltype 获取 rand_int_wrapper 函数的类型，并传递给 random_batching_rule
    # &rand_int_wrapper<F, Func, T...> 是函数指针，作为模板参数传递给 random_batching_rule
    # c10::SymInt 是额外参数 T... 中的一部分，用于模板化 random_batching_rule 的实例化
    # shape 是批处理规则的形状参数
    # std::move(high) 将 high 参数作为右值传递，确保在调用过程中的有效性和性能
    # std::forward<T>(extra_args)... 将额外的模板参数作为完美转发传递给 random_batching_rule
};

// 定义模板函数 `rand_int_low_wrapper`，用于生成随机整数张量
template <typename F, F Func, typename T0, typename T1, typename... T>
Tensor rand_int_low_wrapper(SymIntArrayRef shape, T0 scalar0, T1 scalar1, T... extra_args) {
  // 调用 Func 函数生成随机整数张量，形状为 shape
  return Func(scalar0, scalar1, shape, std::forward<T>(extra_args)...);
}

// 定义模板结构体 `RandTwoLeadingScalarsBatchRuleHelper`，处理带有两个前导标量的随机批处理规则
template <typename F, F Func, typename T0, typename T1, typename T2, typename... T>
struct RandTwoLeadingScalarsBatchRuleHelper<F, Func, typelist<T0, T1, T2, T...>> {
  static Tensor apply(T0 scalar0, T1 scalar1, SymIntArrayRef shape, T... extra_args) {
    // 调用 random_batching_rule 函数处理带有两个前导标量的批处理规则
    return random_batching_rule<decltype(&rand_int_low_wrapper<F, Func, T0, T1, T...>),
                                &rand_int_low_wrapper<F, Func, T0, T1, T...>,
                                T0, T1, T...>(shape, scalar0, scalar1, std::forward<T>(extra_args)...);
  }
};

// 定义模板结构体 `RandpermBatchRuleHelper`，处理随机排列批处理规则
template <typename F, F Func, typename T1, typename... T>
struct RandpermBatchRuleHelper<F, Func, typelist<T1, T...>> {
  static Tensor apply(int64_t n, T... extra_args) {
    // 调用 randperm_batching_rule 函数处理随机排列批处理规则
    return randperm_batching_rule<F, Func, T...>(n, std::forward<T>(extra_args)...);
  }
};

// 定义模板结构体 `UnaryPointwiseRandomBatchRule`，处理一元逐点随机批处理规则
template <typename F, F Func, typename A0, typename... T>
struct UnaryPointwiseRandomBatchRule {
  static Tensor apply(const Tensor& tensor, T... extra_args) {
    // 调用 unary_pointwise_random_batch_rule 函数处理一元逐点随机批处理规则
    return unary_pointwise_random_batch_rule<F, Func, T...>(tensor, std::forward<T>(extra_args)...);
  }
};

// 定义模板结构体 `NormalPointwiseBatchRule`，处理正态分布逐点批处理规则
template <typename F, F Func, typename A0, typename... T>
struct NormalPointwiseBatchRule {
  static Tensor apply(const Tensor& tensor, T... extra_args) {
    // 调用 unary_pointwise_random_batch_rule 函数处理正态分布逐点批处理规则
    return unary_pointwise_random_batch_rule<F, Func, T...>(tensor, std::forward<T>(extra_args)...);
  }
};

// 定义模板函数 `normal_wrapper`，用于生成正态分布张量
template<typename F, F Func, typename... T>
Tensor normal_wrapper(const Tensor& tensor, double scalar, T... extra_args) {
  // 调用 Func 函数生成正态分布张量
  return Func(scalar, tensor, extra_args...);
}

// 定义模板结构体 `UnaryPointwiseRandomLeadingFloatBatchRule`，处理带有浮点数标量的一元逐点随机批处理规则
template <typename F, F Func, typename A0, typename A1, typename... T>
struct UnaryPointwiseRandomLeadingFloatBatchRule {
  static Tensor apply(double scalar, const Tensor& tensor, T... extra_args) {
    // 调用 unary_pointwise_random_batch_rule 函数处理带有浮点数标量的一元逐点随机批处理规则
    return unary_pointwise_random_batch_rule<decltype(&normal_wrapper<F, Func, T...>),
                                         &normal_wrapper<F, Func, T...>, double,
                                         T...>(tensor, scalar, std::forward<T>(extra_args)...);
  }
};

// Torch 库实现的具体批处理规则库
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 定义宏，用于简化代码，具体内容未提供
  #define RANDOM_INPLACE_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

# 在模块 `m` 中实现一个函数重载规则，函数名称由 `op` 和 `overload` 拼接而成。

  RANDOM_INPLACE_BATCH_RULE2(bernoulli_, float);

# 调用宏 `RANDOM_INPLACE_BATCH_RULE2`，传入参数 `bernoulli_` 和 `float`，用于生成相应的代码块。

  #undef RANDOM_INPLACE_BATCH_RULE2

# 取消之前定义的宏 `RANDOM_INPLACE_BATCH_RULE2`，避免其在后续代码中的影响。
TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  // 定义宏 RANDOM_BATCH_RULE(op)，为给定操作op注册批处理规则
  #define RANDOM_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandomBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  // 定义宏 RANDOM_BATCH_RULE2(op, overload)，为给定操作op和重载overload注册批处理规则
  #define RANDOM_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  // 定义宏 RANDOM_INPLACE_BATCH_RULE(op)，为给定操作op注册原地批处理规则
  #define RANDOM_INPLACE_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  // 定义宏 RANDOM_INPLACE_BATCH_RULE2(op, overload)，为给定操作op和重载overload注册原地批处理规则
  #define RANDOM_INPLACE_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  // 定义宏 RANDINT_BATCH_RULE(op)，为给定操作op注册随机整数批处理规则
  #define RANDINT_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandIntBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                             c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  // 定义宏 RANDINT_BATCH_RULE2(op, overload)，为给定操作op和重载overload注册随机整数批处理规则
  #define RANDINT_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandIntBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  // 定义宏 RAND_TWO_LEADING_SCALARS_BATCH_RULE(op, overload)，为给定操作op和重载overload注册两个前导标量的随机批处理规则
  #define RAND_TWO_LEADING_SCALARS_BATCH_RULE(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandTwoLeadingScalarsBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                                c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  // 定义宏 RANDPERM_BATCH_RULE(op)，为给定操作op注册随机排列批处理规则
  #define RANDPERM_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandpermBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  // 定义宏 RANDPERM_BATCH_RULE2(op, overload)，为给定操作op和重载overload注册随机排列批处理规则
  #define RANDPERM_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandpermBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  // 定义宏 UNARY_POINTWISE_RANDOM(op)，为给定操作op注册单目点对点随机批处理规则
  #define UNARY_POINTWISE_RANDOM(op) \
    m.impl(#op, SINGLE_ARG(\
      UnaryPointwiseRandomBatchRule<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                                    c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  // 定义宏 UNARY_POINTWISE_RANDOM2(op, overload)，为给定操作op和重载overload注册单目点对点随机批处理规则
  #define UNARY_POINTWISE_RANDOM2(op, overload) \
    // 使用 m.impl() 函数注册单参数的随机批处理规则，针对给定的操作符和重载
    m.impl(#op"."#overload, SINGLE_ARG(\
      UnaryPointwiseRandomBatchRule<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                                    c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  // 定义宏，注册单参数随机批处理规则，适用于操作符和重载为 float 的情况
  #define UNARY_POINTWISE_RANDOM_LEADING_FLOAT(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      UnaryPointwiseRandomLeadingFloatBatchRule<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                                                c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  // 注册 randn 操作的随机批处理规则
  RANDOM_BATCH_RULE(randn);
  // 注册带有 generator 参数的 randn 操作的随机批处理规则
  RANDOM_BATCH_RULE2(randn, generator);
  // 注册带有 generator_with_names 参数的 randn 操作的随机批处理规则
  RANDOM_BATCH_RULE2(randn, generator_with_names);
  // 注册带有 names 参数的 randn 操作的随机批处理规则
  RANDOM_BATCH_RULE2(randn, names);

  // 注册 rand 操作的随机批处理规则
  RANDOM_BATCH_RULE(rand);
  // 注册带有 generator 参数的 rand 操作的随机批处理规则
  RANDOM_BATCH_RULE2(rand, generator);
  // 注册带有 generator_with_names 参数的 rand 操作的随机批处理规则
  RANDOM_BATCH_RULE2(rand, generator_with_names);
  // 注册带有 names 参数的 rand 操作的随机批处理规则
  RANDOM_BATCH_RULE2(rand, names);

  // 注册 random_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE(random_);
  // 注册带有 from 参数的 random_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE2(random_, from);
  // 注册带有 to 参数的 random_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE2(random_, to);

  // 注册 cauchy_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE(cauchy_);
  // 注册 exponential_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE(exponential_);
  // 注册 geometric_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE(geometric_);
  // 注册 log_normal_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE(log_normal_);
  // 注册 normal_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE(normal_);
  // 注册 uniform_ 操作的原地随机批处理规则
  RANDOM_INPLACE_BATCH_RULE(uniform_);

  // 注册 randint 操作的随机整数批处理规则
  RANDINT_BATCH_RULE(randint);
  // 注册带有 generator 参数的 randint 操作的随机整数批处理规则
  RANDINT_BATCH_RULE2(randint, generator);
  // 注册带有 low 参数的 randint 操作的随机整数批处理规则
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(randint, low);
  // 注册带有 low_generator 参数的 randint 操作的随机整数批处理规则
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(randint, low_generator);

  // 注册 "bernoulli_.Tensor" 操作的原地 bernoulli 分布批处理规则
  m.impl("bernoulli_.Tensor", at::functorch::bernoulli_inplace_Tensor_batching_rule);
  // 注册带有 float 参数的 bernoulli_ 操作的原地 bernoulli 分布批处理规则
  RANDOM_INPLACE_BATCH_RULE2(bernoulli_, float);
  // 注册带有 p 参数的 bernoulli 操作的单参数随机批处理规则
  UNARY_POINTWISE_RANDOM2(bernoulli, p);

  // 注册 randperm 操作的随机排列批处理规则
  RANDPERM_BATCH_RULE(randperm);
  // 注册带有 generator 参数的 randperm 操作的随机排列批处理规则
  RANDPERM_BATCH_RULE2(randperm, generator);

  // 注册带有 float_float 参数的 normal 操作的双参数随机批处理规则
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(normal, float_float);
  // 注册带有 Tensor_float 参数的 normal 操作的双参数随机批处理规则
  UNARY_POINTWISE_RANDOM2(normal, Tensor_float);
  // 注册带有 float_Tensor 参数的 normal 操作的双参数随机批处理规则
  UNARY_POINTWISE_RANDOM_LEADING_FLOAT(normal, float_Tensor);

  // 注册 "native_dropout" 操作的批处理规则，需要特别处理因为 CUDA 版本不调用 bernoulli
  m.impl("native_dropout", native_dropout_batching_rule);

  // 注册单参数随机批处理规则，针对 _standard_gamma 操作
  UNARY_POINTWISE_RANDOM(_standard_gamma);
  // 注册单参数随机批处理规则，针对 _sample_dirichlet 操作
  UNARY_POINTWISE_RANDOM(_sample_dirichlet);
  // 注册 "multinomial" 操作的批处理规则
  m.impl("multinomial", multinomial_batching_rule);
  // 注册单参数随机批处理规则，针对 poisson 操作
  UNARY_POINTWISE_RANDOM(poisson);
  // 注册单参数随机批处理规则，针对 bernoulli 操作
  UNARY_POINTWISE_RANDOM(bernoulli);

  // 定义宏 TENSOR_LIKE_COMMON_ARG_TYPES，包含通用的张量参数类型
  #define TENSOR_LIKE_COMMON_ARG_TYPES optional<ScalarType>, optional<Layout>, optional<Device>, optional<bool>, optional<MemoryFormat>
  // 注册 "randint_like" 操作的随机整数批处理规则，使用与 randint_like 相关的函数
  m.impl("randint_like", tensor_like_random_batch_rule<decltype(&ATEN_FN(randint_like)), &ATEN_FN(randint_like), int64_t, TENSOR_LIKE_COMMON_ARG_TYPES>);
  // 注册带有 low_dtype 参数的 "randint_like" 操作的随机整数批处理规则
  m.impl("randint_like.low_dtype", tensor_like_random_batch_rule<\
  // 注册 `rand` 操作的实现，使用给定的批处理规则和通用参数类型
  m.impl("rand", tensor_like_random_batch_rule<decltype(&ATEN_FN(rand)), &ATEN_FN(rand), TENSOR_LIKE_COMMON_ARG_TYPES>);
  // 注册 `randint` 操作的实现，使用给定的批处理规则和通用参数类型
  m.impl("randint", tensor_like_random_batch_rule<decltype(&ATEN_FN(randint)), &ATEN_FN(randint), TENSOR_LIKE_COMMON_ARG_TYPES>);
  // 注册 `randn` 操作的实现，使用给定的批处理规则和通用参数类型
  m.impl("randn", tensor_like_random_batch_rule<decltype(&ATEN_FN(randn)), &ATEN_FN(randn), TENSOR_LIKE_COMMON_ARG_TYPES>);

  // 取消定义预处理宏，用于 `rand` 等操作的批处理规则
  #undef RANDOM_BATCH_RULE
  #undef RANDOM_BATCH_RULE2
  #undef RANDOM_INPLACE_BATCH_RULE
  #undef RANDOM_INPLACE_BATCH_RULE2
  #undef RANDINT_BATCH_RULE
  #undef RANDINT_BATCH_RULE2
  #undef RAND_TWO_LEADING_SCALARS_BATCH_RULE
  #undef RANDPERM_BATCH_RULE
  #undef RANDPERM_BATCH_RULE2
  // 取消定义用于一元点逐元素随机操作的预处理宏
  #undef UNARY_POINTWISE_RANDOM
  #undef UNARY_POINTWISE_RANDOM2
  // 取消定义用于一元点逐元素随机操作中首个参数为浮点数的预处理宏
  #undef UNARY_POINTWISE_RANDOM_LEADING_FLOAT
  // 取消定义用于通用参数类型的预处理宏
  #undef TENSOR_LIKE_COMMON_ARG_TYPES
}

// 结束命名空间 at::functorch
} // namespace at::functorch
```