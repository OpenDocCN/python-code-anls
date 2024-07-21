# `.\pytorch\aten\src\ATen\functorch\BatchedFallback.cpp`

```
// 版权声明和许可信息

#include <ATen/functorch/BatchedFallback.h>  // 引入 BatchedFallback 功能库头文件
#include <ATen/functorch/LegacyVmapTransforms.h>  // 引入 LegacyVmapTransforms 功能库头文件
#include <ATen/functorch/TensorWrapper.h>  // 引入 TensorWrapper 功能库头文件
#include <ATen/functorch/DynamicLayer.h>  // 引入 DynamicLayer 功能库头文件
#include <ATen/functorch/PlumbingHelper.h>  // 引入 PlumbingHelper 功能库头文件

#include <ATen/Context.h>  // 引入 ATen 上下文头文件
#include <ATen/MatrixRef.h>  // 引入 ATen 矩阵引用头文件
#include <ATen/core/dispatch/Dispatcher.h>  // 引入 ATen 分发器头文件
#include <c10/util/accumulate.h>  // 引入 c10 累加工具头文件
#include <c10/util/llvmMathExtras.h>  // 引入 c10 LLVM 数学扩展头文件
#include <c10/util/irange.h>  // 引入 c10 迭代范围头文件

namespace at::functorch {

bool kVmapFallbackWarningEnabled = true;  // 定义和初始化 vmap 回退警告开关，默认开启

bool isVmapFallbackWarningEnabled() {  // 查询 vmap 回退警告是否开启
  return kVmapFallbackWarningEnabled;
}

void setVmapFallbackWarningEnabled(bool enabled) {  // 设置 vmap 回退警告开关
  kVmapFallbackWarningEnabled = enabled;
}

bool kVmapFallbackEnabled = true;  // 定义和初始化 vmap 回退功能开关，默认开启

bool isVmapFallbackEnabled() {  // 查询 vmap 回退功能是否开启
  return kVmapFallbackEnabled;
}

void setVmapFallbackEnabled(bool enabled) {  // 设置 vmap 回退功能开关
  kVmapFallbackEnabled = enabled;
}

// 给定线性索引和大小数组，计算出实际索引数组
// 示例：给定 linear_idx = 3，sizes = [5, 2]，返回 [1, 0]
static at::SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize>
computeIndex(int64_t linear_idx, IntArrayRef sizes) {
  at::SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize> result;  // 结果数组
  result.reserve(sizes.size());  // 预留空间
  for (auto it = sizes.rbegin(); it != sizes.rend(); it++) {  // 从后向前遍历 sizes
    auto remainder = linear_idx % *it;  // 计算余数
    result.push_back(remainder);  // 将余数加入结果数组
    linear_idx -= remainder;  // 减去余数
    linear_idx /= *it;  // 更新 linear_idx
  }
  std::reverse(std::begin(result), std::end(result));  // 翻转结果数组顺序
  return result;  // 返回计算出的实际索引数组
}

// 检查给定函数模式的所有返回值是否都是张量类型
static bool areAllReturnsTensors(const at::FunctionSchema& schema) {
  return std::all_of(
      schema.returns().begin(),
      schema.returns().end(),
      [] (const Argument& arg) { return arg.type() == TensorType::get(); });
}

// 检查给定函数模式的任何参数是否是张量列表类型
static bool areAnyArgumentsTensorList(const at::FunctionSchema& schema) {
  return std::any_of(
      schema.arguments().begin(),
      schema.arguments().end(),
      [] (const Argument& arg) {
        return arg.type()->isSubtypeOf(ListType::ofTensors()) ||
          arg.type()->isSubtypeOf(ListType::ofOptionalTensors());
      });
}

// 发出 vmap 回退警告消息
static void warnFallback(const c10::FunctionSchema& schema, bool is_inplace, bool is_nested=false) {
  TORCH_CHECK(isVmapFallbackEnabled(),
      schema.operator_name(), " hit the vmap fallback which is currently disabled");  // 检查 vmap 回退功能是否开启
  if (!isVmapFallbackWarningEnabled()) {  // 如果 vmap 回退警告被禁用，则直接返回
    return;
  }
  // 发出警告消息，说明性能有所下降，并建议提出 GitHub 问题以优先实现相关批处理规则
  TORCH_WARN("There is a performance drop because we have not yet implemented ",
             "the ", (is_nested ? "nested " : "") , "batching rule for ",
             schema.operator_name(), ". Please file us an issue on GitHub so that ",
             "we can prioritize its implementation.");
}

// 算法的一般流程如下
// - 首先，确定哪些参数是 BatchedTensors 并保存它们
// - 首先，我们根据操作符的schema，警告即将使用fallback（备用方案）执行。
static void batchedTensorInplaceForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  warnFallback(schema, /*in_place*/true);

  // 获取操作符的参数数量，并从堆栈中获取相应数量的参数
  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  // `self` 是需要原地修改的张量
  Tensor self = arguments[0].toTensor();
  // 获取可能的批次化实现
  const auto* self_impl = maybeGetBatchedImpl(self);
  // 创建用于记录vmap级别的位集合
  std::bitset<kVmapMaxTensorDims> self_vmap_levels;
  if (self_impl) {
    self_vmap_levels = createVmapLevelsBitset(self_impl->level());
  }

  // 确定哪些参数是批次化张量，并保存到向量中
  // 对于每个批次化张量，还记录它们在`arguments`中的位置
  at::SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(0, arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.defined()) {
      continue;
    }
    const auto* batched = maybeGetBatchedImpl(tensor);
    if (!batched) {
      continue;
    }

    // NOTE: [vmap-incompatible in-place operations]
    // 在存在某些vmap级别`l`使得`self`不在该级别上进行vmap，而其他参数在该级别上进行vmap时，
    // 不支持对`self`进行原地操作。例如，假设B0是vmap中的一个批次维度，考虑以下示例：
    // vmap(Tensor.add_, in_dims=(None, 0))(torch.ones(3), torch.ones(B0, 3))
    // - `self`是torch.ones(3)，并且不参与该vmap操作
    // - `other`是BatchedTensor(torch.ones(B0, 3))
    // 无法执行self.add_(other)，因为`other`的元素数量比`self`多，这是由于其在vmap维度上扩展了。
    //
    // 在vmap的后备方案中，当检测到这种情况时，我们应该报错。
    auto other_vmap_levels = createVmapLevelsBitset(batched->level());
    if (self_vmap_levels != (self_vmap_levels | other_vmap_levels)) {
      // 检查是否存在未对齐的vmap级别，即self_vmap_levels和other_vmap_levels的并集不等于self_vmap_levels
      // 找到一个未对齐的vmap级别进行报错
      auto additional_bdims = (self_vmap_levels | other_vmap_levels) ^ self_vmap_levels;
      // 找到最后一个设置位，即找到一个未对齐的vmap级别
      auto offending_level = llvm::findLastSet(additional_bdims.to_ulong());
      // 下面的代码会打印出类似 "vmap: aten::add_(tensor, ...) is not possible" 的信息，
      // 但最好打印出 "tensor.add_(...) is not possible"。
      // 据我所知，目前没有官方方法可以获取 "add_"，也无法判断运算符是方法还是函数的变体。
      TORCH_CHECK(false,
        "vmap: ", schema.name(), "(self, *extra_args) is not possible because ",
        "there exists a Tensor `other` in extra_args that has more elements ",
        "than `self`. This happened due to `other` being vmapped over but ",
        "`self` not being vmapped over at level ", offending_level, ". ",
        "Please try to use out-of-place operators instead of ", schema.name(), ". ",
        "If said operator is being called inside the PyTorch framework, ",
        "please file a bug report instead.");
    }
    // 将当前处理的张量添加到批处理张量列表中
    batched_tensor_inputs.push_back(tensor);
    // 将当前处理的索引位置添加到批处理张量索引列表中
    batched_tensor_inputs_position.push_back(static_cast<int64_t>(idx));
  }
  // 断言批处理张量列表不能为空
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // 使用 MultiBatchVmapTransform 将 BatchedTensor 输入转换为 VmapPhysicalViews
  // 这些视图包含所有批处理维度
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);

  // 计算总批次数
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  auto first_physical_view_sizes = input_physical_views.front().tensor().sizes();
  auto batch_sizes = ArrayRef<int64_t>(
      first_physical_view_sizes.begin(), first_physical_view_sizes.begin() + num_batch_dims);
  const auto num_batches = c10::multiply_integers(batch_sizes);
  // 没有形状检查的 API，无法计算正确的输出形状，因此报错退出
  TORCH_CHECK(num_batches > 0,
      "Batching rule not implemented for ", schema.operator_name(), ". ",
      "The fallback path does not support vmap over dims of size 0.");

  // 策略：对于每个批次，我们将会将参数的切片（如果适用）推送到 `stack` 上，并调用 `op`。
  for (int64_t linear_idx = 0; linear_idx < num_batches; ++linear_idx) {
    // 计算当前线性索引对应的索引
    auto index = computeIndex(linear_idx, batch_sizes);
    // 获取批处理张量输入的位置迭代器
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    // 获取输入物理视图的迭代器
    auto input_physical_views_iter = input_physical_views.begin();
    // 遍历所有函数参数的索引
    for (const auto arg_idx : c10::irange(0, num_arguments)) {
      // 假设 torch::jit::Stack 使用 vector<IValue> 作为后端存储，为简单起见。如果不是这种情况，需要更新此代码。
      // 获取当前参数的引用
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      
      // 检查当前参数是否为 BatchedTensor，如果不是，则将其推入堆栈并继续下一个参数
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || (int64_t)arg_idx != *batched_tensor_inputs_pos_iter) {
        torch::jit::push(stack, argument);
        continue;
      }
      
      // 当前参数是 BatchedTensor
      // 断言确保输入物理视图迭代器未达到末尾
      TORCH_INTERNAL_ASSERT(input_physical_views_iter != input_physical_views.end());
      // 获取当前参数对应的物理视图
      const auto& physical_view_for_argument = *input_physical_views_iter;
      // 使用索引获取物理视图中的张量
      auto thing = physical_view_for_argument.tensor().index(index);
      // 将获取的张量推入堆栈
      torch::jit::push(stack, thing);
      // 更新批量张量输入位置迭代器和输入物理视图迭代器
      batched_tensor_inputs_pos_iter++;
      input_physical_views_iter++;
    }

    // 调用操作函数 op，参数为堆栈中当前的内容
    op.callBoxed(stack);
    // 从堆栈中弹出一个元素
    torch::jit::drop(stack, 1);
  }

  // 丢弃堆栈中指定数量的元素，这里是 num_arguments 个
  torch::jit::drop(stack, num_arguments);
  // 将 self 推入堆栈，作为最终的返回结果
  torch::jit::push(stack, self);
}

// 安全堆栈函数，用于堆栈化一组张量
static Tensor safeStack(TensorList tensors) {
  // Lambda 函数，用于检查张量是否已定义
  auto is_defined = [](const Tensor& t) { return t.defined(); };
  
  // 如果所有张量都已定义，则堆栈化它们并返回
  if (std::all_of(tensors.begin(), tensors.end(), is_defined)) {
    return at::stack(tensors);
  }
  
  // 如果没有张量被定义，则返回一个未定义的张量
  // 注意 [vmap through backward and undefined grad] 部分的说明
  if (std::none_of(tensors.begin(), tensors.end(), is_defined)) {
    return Tensor();
  }
  
  // 如果既有定义的张量又有未定义的张量，则抛出错误
  TORCH_CHECK(false,
      "vmap: slow fallback received a mix of undefined and defined tensors ",
      "as the result of an operation. This is not supported, please file us ",
      "an issue on github.");
}

// TODO: 考虑重写以下部分以便看起来像：
// https://gist.github.com/zou3519/7b7c6a4a258d580f62d1d969851be6b1<Paste>

// 算法的一般流程如下：
// - 首先，确定哪些参数是 BatchedTensors，并将它们保存到一个向量中。
//   同时，保存每个 BatchedTensor 在参数列表中的索引，这将在后续的记录中很有用。
// - 接下来，对所有 BatchedTensors 应用 MultiBatchVmapTransform。
//   这将返回一个 VmapPhysicalView 的向量，其中包含包含所有集体批次维度的张量，
//   这些维度位于张量的前部。
// - 然后，我们尝试针对输入的每个切片调用 `op` 函数。
//   为此，我们重复地对输入参数进行切片（如果它们是 BatchedTensors），
//   将切片（或未切片的版本）放入堆栈中，调用运算符，然后从堆栈中弹出结果。
// - 前面步骤获得的每个结果都是总结果的一个切片，
//   因此我们将这些张量堆栈在一起形成最终的结果。
  // 获取操作符的模式描述信息
  const auto& schema = op.schema();
  // 返回值的数量
  const auto num_returns = schema.returns().size();
  // 参数的数量
  const auto num_arguments = schema.arguments().size();
  // 获取最后 num_arguments 个参数
  const auto arguments = torch::jit::last(stack, num_arguments);

  // 检查所有返回值是否都是张量，并且没有任何参数是张量列表
  TORCH_CHECK(areAllReturnsTensors(schema) && !areAnyArgumentsTensorList(schema),
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "We could not generate a fallback.");

  // 如果没有参数参与当前级别的运算，使用 ExcludeDispatchKeyGuard 来保护不派发 FuncTorchBatched 的调度键
  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用操作符的函数
    op.callBoxed(stack);
    return;
  }

  // 如果是就地操作符，则调用就地操作的 fallback 函数
  if (isInplaceOp(schema)) {
    batchedTensorInplaceForLoopFallback(op, stack);
    return;
  }
  // 检查模式是否可变且没有任何别名信息
  TORCH_CHECK(!schema.is_mutable() && !schema.hasAnyAliasInfo(),
              "Batching rule not implemented for ", schema.operator_name(), "; ",
              "the fallback path doesn't work on out= or view ops.");
  // 检查至少有一个返回值
  TORCH_CHECK(num_returns >= 1,
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "The fallback path does not support operations with no returns.");
  // 发出警告，指示使用了回退方案，而不是就地操作
  warnFallback(schema, /*in_place*/false);

  // 计算 arguments 起始位置
  const auto arguments_begin = stack->size() - num_arguments;

  // 确定哪些参数是 BatchedTensor，将它们保存到一个向量中
  // 对于每个 BatchedTensor，还记录它们在 arguments 中的位置
  at::SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(0, arguments.size())) {
    // 获取当前参数的 IValue
    const auto& ivalue = arguments[idx];
    // 如果不是张量，则跳过
    if (!ivalue.isTensor()) {
      continue;
    }
    // 将 IValue 转换为张量
    const auto& tensor = ivalue.toTensor();
    // 如果张量未定义，则跳过
    if (!tensor.defined()) {
      continue;
    }
    // 尝试获取 Batched 实现
    const auto* batched = maybeGetBatchedImpl(tensor);
    // 如果没有 Batched 实现，则跳过
    if (!batched) {
      continue;
    }
    // 将 BatchedTensor 添加到 batched_tensor_inputs 中
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(static_cast<int64_t>(idx));
  }
  // 确保 batched_tensor_inputs 不为空，即确保有批处理张量作为输入
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // 将 BatchedTensor 参数进行 MultiBatchVmapTransform 转换，返回包含所有批次维度的 VmapPhysicalViews
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);

  // 计算总批次数
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  auto some_sizes = input_physical_views.front().tensor().sizes();
  auto batch_sizes = ArrayRef<int64_t>(some_sizes.begin(), some_sizes.begin() + num_batch_dims);
  const auto num_batches = c10::multiply_integers(batch_sizes);

  // 没有形状检查的 API，无法计算输出的正确形状，因此直接报错
  TORCH_CHECK(num_batches > 0,
      "Batching rule not implemented for ", schema.operator_name(), ". ",
      "The fallback path does not support vmap over dims of size 0.");

  // 策略：对于每个批次，将参数的切片推送到 `stack` 上，调用 `op`，并将结果存储在 `output_shards` 中。
  //
  // 注释: [输出片段布局]
  // 假设操作有三个输出：a, b, c。
  // output_shards 的布局如下:
  // [ a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3]
  // 这样可以更轻松地在下一步中调用 at::stack([a0...a3]), at::stack([b0...b3])
  std::vector<Tensor> output_shards(num_batches * num_returns);

  // 对每个批次进行循环处理
  for (int64_t linear_idx = 0; linear_idx < num_batches; ++linear_idx) {
    // 计算当前批次的索引
    auto index = computeIndex(linear_idx, batch_sizes);
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    auto input_physical_views_iter = input_physical_views.begin();

    // 遍历每个参数的索引范围
    for (const auto arg_idx : c10::irange(0, num_arguments)) {
      // 假设 torch::jit::Stack 是基于 vector<IValue> 实现的，以简化处理。
      // 如果不是这种情况，需要更新代码。
      const auto& argument = (*stack)[arguments_begin + arg_idx];

      // 如果 argument 不是 BatchedTensor，则直接推送到栈中并继续下一个参数
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || (int64_t)arg_idx != *batched_tensor_inputs_pos_iter) {
        torch::jit::push(stack, argument);
        continue;
      }

      // 如果 argument 是 BatchedTensor，则进行处理
      TORCH_INTERNAL_ASSERT(input_physical_views_iter != input_physical_views.end());
      const auto& physical_view_for_argument = *input_physical_views_iter;
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      // 将物理视图中的张量的指定索引推送到栈中
      torch::jit::push(stack, physical_view_for_argument.tensor().index(index));

      // 更新迭代器
      batched_tensor_inputs_pos_iter++;
      input_physical_views_iter++;
    }

    // 用于调试，打印当前栈顶张量
    // std::cout << "[Fallback]: ";
    // at::dump_tensor((*stack)[stack->size() - 1].toTensor());

    // 排除 FuncTorchBatched 分发键的保护
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用op对象的callBoxed方法，传入stack作为参数
    op.callBoxed(stack);

    // 将结果存储到output_shards中。参见 NOTE: [Output shards layout] 了解存储shards的细节。
    const auto returns = torch::jit::last(stack, num_returns);
    // 遍历每个返回的Tensor，使用索引return_idx
    for (const auto return_idx : c10::irange(0, returns.size())) {
      // 将returns中的每个Tensor放入output_shards数组的适当位置
      output_shards[num_batches * return_idx + linear_idx] = returns[return_idx].toTensor();
    }
    // 从stack中丢弃num_returns个元素
    torch::jit::drop(stack, num_returns);
  }

  // 对于每个输出的Tensor，堆叠tensor的shards以形成返回值
  // 从stack中丢弃num_arguments个元素
  torch::jit::drop(stack, num_arguments);
  // 将output_shards分块成Tensor的矩阵引用
  auto output_shards_chunks = MatrixRef<Tensor>(output_shards, num_batches);
  // 遍历每个返回值的索引return_idx
  for (const auto return_idx : c10::irange(0, num_returns)) {
    // 获取output_shards_chunks中的shards
    auto shards = output_shards_chunks[return_idx];
    // 保证不使用FuncTorchBatched的分发键
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 安全地堆叠shards成flat_output
    auto flat_output = safeStack(shards);
    // 查看是否flat_output未定义。参见 NOTE [vmap through backward and undefined grad]
    if (!flat_output.defined()) {
      // 将flat_output推送回stack，并继续下一轮循环
      torch::jit::push(stack, flat_output);
      continue;
    }
    // 构造输出大小output_sizes，使用batch_sizes作为起始维度
    VmapDimVector output_sizes(batch_sizes);
    output_sizes.insert(
        output_sizes.end(),
        flat_output.sizes().begin() + 1,
        flat_output.sizes().end());
    // 将flat_output视图视图为output_sizes的形状，并应用getPhysicalToLogicalMap得到的映射
    torch::jit::push(
        stack,
        input_physical_views.front().getPhysicalToLogicalMap().apply(flat_output.view(output_sizes)));
  }
// 结束函数 batchedNestedTensorForLoopFallback
void batchedNestedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 获取操作符的模式信息
  const auto& schema = op.schema();
  // 返回值数量
  const auto num_returns = schema.returns().size();
  // 参数数量
  const auto num_arguments = schema.arguments().size();
  // 获取栈中最后 num_arguments 个元素作为当前函数的参数
  const auto arguments = torch::jit::last(stack, num_arguments);

  // 检查所有返回值是否为张量，且没有任何参数是张量列表
  TORCH_CHECK(areAllReturnsTensors(schema) && !areAnyArgumentsTensorList(schema),
              "Nested batching rule not implemented for ", schema.operator_name(), ". ",
              "We could not generate a fallback.");

  // 如果所有参数均不参与当前级别的批处理，则执行下列逻辑
  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    // 禁止 FuncTorchBatched 分发键
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 禁止 BatchedNestedTensor 分发键
    c10::impl::ExcludeDispatchKeyGuard nt_guard(DispatchKey::BatchedNestedTensor);
    // 调用操作符的函数
    op.callBoxed(stack);
    return;
  }

  // 如果是原地操作，则抛出异常，不支持嵌套张量的 vmap 回退
  if (isInplaceOp(schema)) {
    TORCH_INTERNAL_ASSERT(false, "vmap fallback not supported for in-place ops on nested tensors");
    return;
  }
  // 检查模式是否不可变且没有别名信息
  TORCH_CHECK(!schema.is_mutable() && !schema.hasAnyAliasInfo(),
              "Nested batching rule not implemented for ", schema.operator_name(), "; ",
              "the fallback path doesn't work on out= or view ops.");
  // 检查返回值数量至少为1
  TORCH_CHECK(num_returns >= 1,
              "Nested batching rule not implemented for ", schema.operator_name(), ". ",
              "The fallback path does not support operations with no returns.");
  // 发出警告，表明正在使用回退
  warnFallback(schema, /*in_place*/false, /*is_nested*/true);

  // 获取栈中参数的起始位置
  const auto arguments_begin = stack->size() - num_arguments;

  // 确定哪些参数是批处理张量，并将它们保存到一个向量中
  // 同时记录它们在 arguments 中的位置
  at::SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(0, arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.defined()) {
      continue;
    }
    const auto* batched = maybeGetBatchedImpl(tensor);
    if (!batched) {
      continue;
    }
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(static_cast<int64_t>(idx));
  }
  // 断言 batched_tensor_inputs 不为空
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // 创建一个未绑定的张量向量的向量
  std::vector<std::vector<Tensor>> unbound;
  for (auto const &batched_tensor_input: batched_tensor_inputs) {
    // 获取批处理实现
    auto *batched_impl = maybeGetBatchedImpl(batched_tensor_input);
    // 断言批处理实现的值要么是嵌套的，要么 bdim=0
    TORCH_INTERNAL_ASSERT(batched_impl->value().is_nested() || batched_impl->bdim() == 0,
        "Fallback not supported for mixed nested / non-nested arguments without bdim=0");
    // 禁止 BatchedNestedTensor 分发键
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    // 解绑批处理的值，并添加到 unbound 中
    auto this_unbound = batched_impl->value().unbind();
    // 如果 unbound 非空，则进行以下操作
    if (!unbound.empty()) {
      // 断言第一个未绑定张量的大小与当前未绑定张量的大小相同，否则抛出异常
      TORCH_INTERNAL_ASSERT(unbound.front().size() == this_unbound.size(),
          "Fallback not supported for differently-sized nested arguments");
    }
    // 将当前的未绑定张量 this_unbound 添加到 unbound 的末尾
    unbound.push_back(this_unbound);
  }

  // 获取第一个未绑定张量的组件数量
  const auto num_components = unbound.front().size();
  // 准备存储输出分片的向量，总长度为组件数量乘以返回值的数量
  std::vector<Tensor> output_shards(num_components * num_returns);
  // 遍历每个组件索引
  for (const auto component_idx : c10::irange(0, num_components)) {
    auto batched_idx = 0;
    // 迭代器用于跟踪批处理张量输入的位置
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    // 遍历每个参数索引
    for (const auto arg_idx : c10::irange(0, num_arguments)) {
      // 假设 torch::jit::Stack 由 vector<IValue> 支持，获取当前参数的值
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      // 如果当前参数不是批处理张量，将其推送到堆栈中并继续下一个参数
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || (int64_t)arg_idx != *batched_tensor_inputs_pos_iter) {
        torch::jit::push(stack, argument);
        continue;
      }
      // 如果当前参数是批处理张量，将未绑定的批处理张量组件推送到堆栈中
      torch::jit::push(stack, unbound[batched_idx][component_idx]);
      ++batched_idx;
      ++batched_tensor_inputs_pos_iter;
    }

    // 临时排除 DispatchKey::BatchedNestedTensor 分发键的保护
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    // 调用操作 op 并将结果存储在堆栈中
    op.callBoxed(stack);

    // 将结果存储到 output_shards 中的指定位置，参考注释中的 [Output shards layout]
    const auto returns = torch::jit::last(stack, num_returns);
    for (const auto return_idx : c10::irange(0, returns.size())) {
      output_shards[num_components * return_idx + component_idx] = returns[return_idx].toTensor();
    }
    // 从堆栈中移除 num_returns 个返回值
    torch::jit::drop(stack, num_returns);
  }

  // 对于每个输出张量，堆叠张量分片以形成嵌套返回
  // TODO: 确定输出何时需要嵌套，何时可以不嵌套？
  // 将堆栈中 num_arguments 个参数移除
  torch::jit::drop(stack, num_arguments);
  // 创建 output_shards 的分块引用
  auto output_shards_chunks = MatrixRef<Tensor>(output_shards, num_components);
  // 遍历每个返回值索引
  for (const auto return_idx : c10::irange(0, num_returns)) {
    // 获取返回值索引处的分片
    auto shards = output_shards_chunks[return_idx];
    // 临时排除 DispatchKey::BatchedNestedTensor 分发键的保护
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    // 从张量列表创建嵌套张量，并将其推送到堆栈中
    auto out_nt = at::_nested_tensor_from_tensor_list(shards);
    // 注意事项：嵌套张量只支持在 dim 0 上进行批处理
    torch::jit::push(stack, makeBatched(out_nt, 0, maybeCurrentDynamicLayer()->layerId()));
  }
}

// 定义函数 vmapErrorFallback，处理操作符处理异常情况
void vmapErrorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // 断言检查，如果条件为 false，则输出错误信息
    TORCH_CHECK(false, "Error: ", op.operator_name(), " requires special handling, and does not yet have a batching rule. Feel free to file a github issue!");
}

// 结束命名空间 at::functorch
} // namespace at::functorch
```