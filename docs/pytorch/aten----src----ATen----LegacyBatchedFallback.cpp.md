# `.\pytorch\aten\src\ATen\LegacyBatchedFallback.cpp`

```py
// 包含 ATen 库中的各种头文件，用于张量操作和分发功能
#include <ATen/Context.h>
#include <ATen/LegacyBatchedFallback.h>
#include <ATen/MatrixRef.h>
#include <ATen/LegacyVmapTransforms.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/accumulate.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/irange.h>

// ATen 命名空间，包含所有 ATen 库中的函数和类型定义
namespace at {

// 给定一个线性索引，根据给定的张量大小计算其对应的实际索引
// 例如：给定 linear_idx = 3，sizes = [5, 2]，返回 [1, 0]
static SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize>
computeIndex(int64_t linear_idx, IntArrayRef sizes) {
  // 用于存储计算结果的向量
  SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize> result;
  result.reserve(sizes.size());
  // 从张量大小的最后一个维度开始计算实际索引
  for (auto it = sizes.rbegin(); it != sizes.rend(); it++) {
    auto remainder = linear_idx % *it; // 计算余数
    result.push_back(remainder); // 将余数添加到结果向量中
    linear_idx -= remainder; // 减去余数，用于下一轮计算
    linear_idx /= *it; // 对线性索引进行除法操作，继续计算下一个维度的索引
  }
  std::reverse(std::begin(result), std::end(result)); // 翻转结果向量，使其按正确顺序返回
  return result; // 返回计算得到的索引向量
}

// 检查函数签名中所有返回值是否都是张量类型
static bool areAllReturnsTensors(const FunctionSchema& schema) {
  return std::all_of(
      schema.returns().begin(),
      schema.returns().end(),
      [] (const Argument& arg) { return arg.type() == TensorType::get(); });
}

// 检查函数签名中是否有任何一个参数是张量列表
static bool areAnyArgumentsTensorList(const FunctionSchema& schema) {
  return std::any_of(
      schema.arguments().begin(),
      schema.arguments().end(),
      [] (const Argument& arg) { return arg.type()->isSubtypeOf(*ListType::ofTensors()); });
}

// 检查操作符是否为原地操作。一个操作符满足以下条件时被视为原地操作：
// 1. 第一个参数是张量且被写入
// 2. 第一个参数被返回
// 3. 其他参数没有别名
static bool isInplaceOp(const c10::FunctionSchema& schema) {
  if (!schema.is_mutable() || schema.returns().size() != 1) {
    return false;
  }
  // 检查第一个参数是否被写入
  const AliasInfo* first_arg_alias_info = schema.arguments().begin()->alias_info();
  if (!first_arg_alias_info || !first_arg_alias_info->isWrite()) {
    return false;
  }
  // 检查其他参数是否存在别名
  for (auto it = schema.arguments().begin() + 1; it != schema.arguments().end(); ++it) {
    const AliasInfo* alias_info = it->alias_info();
    if (alias_info) {
      return false;
    }
  }
  // 检查第一个张量是否被返回（即输出带有 (a!)）
  const AliasInfo* return_alias_info = schema.returns()[0].alias_info();
  return return_alias_info && return_alias_info->isWrite();
}

// 如果全局上下文未启用 Vmap 回退警告，产生回退警告
static void warnFallback(const c10::FunctionSchema& schema) {
  if (!globalContext().areVmapFallbackWarningsEnabled()) {
    return;
  }



// 如果前面的条件不满足，直接返回，结束函数执行
return;



  TORCH_WARN("There is a performance drop because we have not yet implemented ",
             "the batching rule for ", schema.operator_name(), ". ",
             "You are using the legacy vmap prototype (torch._vmap_internals.vmap). ",
             "If you are using torch.autograd.functional.{jacobian, hessian} ",
             "or torch._vmap_internals.vmap: please switch to using ",
             "torch.func.{jacrev, jacfwd, hessian} and/or torch.vmap instead ",
             "for better operator coverage and performance improvements .");



// 发出一个警告消息，指示性能下降，因为尚未实现特定操作符的批处理规则
// 提示用户如果在使用特定功能（如 torch.autograd.functional.jacobian, hessian 或 torch._vmap_internals.vmap），
// 应转换为使用 torch.func.{jacrev, jacfwd, hessian} 或 torch.vmap，以获得更好的操作符覆盖率和性能改进
TORCH_WARN("There is a performance drop because we have not yet implemented ",
           "the batching rule for ", schema.operator_name(), ". ",
           "You are using the legacy vmap prototype (torch._vmap_internals.vmap). ",
           "If you are using torch.autograd.functional.{jacobian, hessian} ",
           "or torch._vmap_internals.vmap: please switch to using ",
           "torch.func.{jacrev, jacfwd, hessian} and/or torch.vmap instead ",
           "for better operator coverage and performance improvements .");
}

// The general flow of the algorithm is as follows.
// - First, we figure out which arguments are BatchedTensors and save them
//   to a vector. We also store a vector of which index of the arguments list
//   each BatchedTensor appears in. This will be useful for bookkeeping later.
// - Next, we apply the MultiBatchVmapTransform to all of the BatchedTensors.
//   This returns a vector of VmapPhysicalView that hold tensors that contain
//   all of the collective batch dimensions at the front of the tensors.
// - Then, we attempt to call `op` once per slice of the inputs. To do this,
//   we repeatedly we slice the input arguments (if they are BatchedTensors),
//   put the sliced (or a not-sliced) version of the input onto the stack, invoke
//   the operator, and then pop the results off the stack.
static void batchedTensorInplaceForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  // Emit a warning indicating fallback to non-vectorized execution
  warnFallback(schema);

  // Determine the number of arguments expected by the operator schema
  const auto num_arguments = static_cast<int64_t>(schema.arguments().size());
  // Extract the arguments from the top of the stack
  const auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  // `self` is the Tensor being modified in-place
  Tensor self = arguments[0].toTensor();
  // Retrieve batch dimensions of `self`, if it's batched
  const auto* self_impl = maybeGetBatchedImpl(self);
  std::bitset<kVmapMaxTensorDims> self_vmap_levels;
  if (self_impl) {
    // Create a bitmask of vmap levels for `self` based on its batch dimensions
    self_vmap_levels = createVmapLevelsBitset(self_impl->bdims());
  }

  // Figure out which arguments are BatchedTensor. Save them to a vector.
  // For each BatchedTensor, also record what position of `arguments` they came from.
  SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(arguments.size())) {
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
    // In-place operations on `self` are not possible if there exists some vmap
    // level `l` such that `self` is not being vmapped on that level but another
    // argument is. For example, let B0 be a batch dim inside vmap and consider
    // vmap(Tensor.add_, in_dims=(None, 0))(torch.ones(3), torch.ones(B0, 3))
    // - self is torch.ones(3) and does not participate in this vmap
    // - other is BatchedTensor(torch.ones(B0, 3))
    // There's no way to do self.add_(other) because `other` has more elements
    // elements than `self` due to being vmapped over.
    //
    // In the vmap fallback, we should error out when we detect this.
    auto other_vmap_levels = createVmapLevelsBitset(batched->bdims());
    if (self_vmap_levels != (self_vmap_levels | other_vmap_levels)) {
      // 检查 self_vmap_levels 和 other_vmap_levels 是否有交集，如果有则报错
      // 找到一个 vmap 级别进行报错
      auto additional_bdims = (self_vmap_levels | other_vmap_levels) ^ self_vmap_levels;
      // 找到最后设置位，确定引起错误的 vmap 级别
      auto offending_level = llvm::findLastSet(additional_bdims.to_ulong());
      // 下面的信息打印："vmap: aten::add_(tensor, ...) is not possible"
      // 更好的方式应该打印："tensor.add_(...) is not possible"
      // 目前没有官方方法可以获取 add_，也无法区分运算符是否有方法或函数变体
      TORCH_CHECK(false,
        "vmap: ", schema.name(), "(self, *extra_args) is not possible because ",
        "there exists a Tensor `other` in extra_args that has more elements ",
        "than `self`. This happened due to `other` being vmapped over but ",
        "`self` not being vmapped over at level ", offending_level, ". ",
        "Please try to use out-of-place operators instead of ", schema.name(), ". ",
        "If said operator is being called inside the PyTorch framework, ",
        "please file a bug report instead.");
    }
    // 将 tensor 添加到批处理张量输入列表中
    batched_tensor_inputs.push_back(tensor);
    // 将索引添加到批处理张量输入位置列表中
    batched_tensor_inputs_position.push_back(idx);
  }
  // 断言批处理张量输入列表不为空
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // 使用 MultiBatchVmapTransform 将 BatchedTensor 参数转换为 VmapPhysicalViews
  // 这些视图包含所有批次维度
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);

  // 计算批次的总数
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  auto first_physical_view_sizes = input_physical_views.front().tensor().sizes();
  auto batch_sizes = ArrayRef<int64_t>(
      first_physical_view_sizes.begin(), first_physical_view_sizes.begin() + num_batch_dims);
  // 计算批次的总数
  const auto num_batches = c10::multiply_integers(batch_sizes);
  // 如果没有形状检查的 API，无法计算正确的输出形状，因此直接报错
  TORCH_CHECK(num_batches > 0,
      "Batching rule not implemented for ", schema.operator_name(), ". ",
      "The fallback path does not support vmap over dims of size 0.");

  // 策略：对于每个批次，我们将把参数的切片（如果适用）推送到 `stack`，并调用 `op`
  for (const auto linear_idx : c10::irange(num_batches)) {
    // 计算在批次中的索引
    auto index = computeIndex(linear_idx, batch_sizes);
    // 迭代器，用于批处理张量输入位置列表
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    // 迭代器，用于输入物理视图列表
    auto input_physical_views_iter = input_physical_views.begin();
    // 遍历函数参数的索引范围
    for (const auto arg_idx : c10::irange(num_arguments)) {
      // 假设 torch::jit::Stack 是由 vector<IValue> 支持的，为简单起见。如果不是这种情况，需要更新此代码。
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      // 如果 batched_tensor_inputs_pos_iter 不在有效位置或者当前参数不是批量张量输入
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || arg_idx != *batched_tensor_inputs_pos_iter) {
        // 参数不是批量张量
        torch::jit::push(stack, argument);
        continue;
      }
      // 参数是批量张量
      TORCH_INTERNAL_ASSERT(input_physical_views_iter != input_physical_views.end());
      const auto& physical_view_for_argument = *input_physical_views_iter;
      // 将批量张量的物理视图在指定索引处的张量推送到栈上
      torch::jit::push(stack, physical_view_for_argument.tensor().index(index));
      batched_tensor_inputs_pos_iter++;
      input_physical_views_iter++;
    }

    // 调用操作函数
    op.callBoxed(stack);
    // 从栈中移除一个元素
    torch::jit::drop(stack, 1);
  }

  // 返回原地写入的张量
  torch::jit::drop(stack, num_arguments);
  // 将 self 推送回栈顶
  torch::jit::push(stack, self);
}

static Tensor safeStack(TensorList tensors) {
  auto is_defined = [](const Tensor& t) { return t.defined(); };
  // 检查是否所有的张量都已定义
  if (std::all_of(tensors.begin(), tensors.end(), is_defined)) {
    // 如果所有张量都已定义，则堆叠它们并返回结果张量
    return at::stack(tensors);
  }
  // NOTE [vmap through backward and undefined grad]
  // 当通过反向函数进行 vmap（批处理梯度计算）时，可能出现某些示例的梯度未定义的情况。
  // 在这种情况下，我们返回一个未定义的梯度。
  //
  // 理论上，可能会有一些示例生成未定义的梯度（内核可以查看梯度值并确定梯度是否全为零，
  // 如果是，则返回一个未定义的张量）。我们可以处理这种情况，方法是在堆叠张量时将未定义的梯度视为正确形状的零张量。
  // 然而，我预计这种情况很少发生（我在我们的代码库中找不到例子），因此我们在这种情况下直接报错。
  if (std::none_of(tensors.begin(), tensors.end(), is_defined)) {
    // 如果所有张量都未定义，则返回一个未定义的张量
    return Tensor();
  }
  // 报告错误，说明不能处理混合了已定义和未定义张量的情况
  TORCH_CHECK(false,
      "vmap: slow fallback received a mix of undefined and defined tensors ",
      "as the result of an operation. This is not supported, please file us ",
      "an issue on github.");
}

// The general flow of the algorithm is as follows.
// - First, we figure out which arguments are BatchedTensors and save them
//   to a vector. We also store a vector of which index of the arguments list
//   each BatchedTensor appears in. This will be useful for bookkeeping later.
// - Next, we apply the MultiBatchVmapTransform to all of the BatchedTensors.
//   This returns a vector of VmapPhysicalView that hold tensors that contain
//   all of the collective batch dimensions at the front of the tensors.
// - Then, we attempt to call `op` once per slice of the inputs. To do this,
//   we repeatedly we slice the input arguments (if they are BatchedTensors),
//   put the sliced (or a not-sliced) version of the input onto the stack, invoke
//   the operator, and then pop the results off the stack.
// - Each result obtained from the previous step is a slice of the total result,
//   so we stack those tensors together to form the final result.
void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 获取操作符的模式
  const auto& schema = op.schema();
  // 获取返回值数量
  const auto num_returns = schema.returns().size();

  // 如果是原地操作，则使用原地操作的批处理张量回退方法
  if (isInplaceOp(schema)) {
    batchedTensorInplaceForLoopFallback(op, stack);
    // 结束函数的执行，直接返回，不再执行下面的代码
    return;
  }
  // 检查条件：schema 不可变且没有别名信息
  TORCH_CHECK(!schema.is_mutable() && !schema.hasAnyAliasInfo(),
              "Batching rule not implemented for ", schema.operator_name(), "; ",
              "the fallback path doesn't work on out= or view ops.");
  // 检查条件：所有返回值都是张量且没有任何参数是张量列表
  TORCH_CHECK(areAllReturnsTensors(schema) && !areAnyArgumentsTensorList(schema),
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "We could not generate a fallback.");
  // 检查条件：返回值数量至少为 1
  TORCH_CHECK(num_returns >= 1,
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "The fallback path does not support operations with no returns.");
  // 发出警告：使用了后备方案
  warnFallback(schema);

  // 获取参数的数量，并将它们保存到 arguments 中
  const auto num_arguments = static_cast<int64_t>(schema.arguments().size());
  const auto arguments = torch::jit::last(stack, num_arguments);
  // 计算 arguments 开始的索引位置
  const auto arguments_begin = stack->size() - num_arguments;

  // 确定哪些参数是批处理张量，并将它们保存到 batched_tensor_inputs 中
  // 同时记录它们在 arguments 中的位置
  SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(arguments.size())) {
    // 获取当前参数的 IValue
    const auto& ivalue = arguments[idx];
    // 如果当前值不是张量，则继续下一个循环
    if (!ivalue.isTensor()) {
      continue;
    }
    // 将当前值转换为张量类型
    const auto& tensor = ivalue.toTensor();
    // 如果张量未定义，则继续下一个循环
    if (!tensor.defined()) {
      continue;
    }
    // 尝试获取张量的批处理实现
    const auto* batched = maybeGetBatchedImpl(tensor);
    // 如果未能成功获取批处理实现，则继续下一个循环
    if (!batched) {
      continue;
    }
    // 将找到的批处理张量添加到 batched_tensor_inputs 中
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(idx);
  }
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // MultiBatchVmapTransform the BatchedTensor arguments. This returns
  // VmapPhysicalViews that contain all of the batch dimensions.
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);
  // 将批量张量参数进行多批次Vmap变换，返回包含所有批次维度的Vmap物理视图

  // Compute the total number of batches
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  auto some_sizes = input_physical_views.front().tensor().sizes();
  auto batch_sizes = ArrayRef<int64_t>(some_sizes.begin(), some_sizes.begin() + num_batch_dims);
  const auto num_batches = c10::multiply_integers(batch_sizes);
  // 计算批次总数

  // Without a shape-checking API, we're unable to compute the correct shape of
  // the output so we just error out.
  TORCH_CHECK(num_batches > 0,
      "Batching rule not implemented for ", schema.operator_name(), ". ",
      "The fallback path does not support vmap over dims of size 0.");
  // 如果没有形状检查的API，我们无法计算输出的正确形状，因此报错

  // Strategy: For each batch, we are going to push slices (where applicable)
  // of the arguments onto `stack`, call `op`, and store the result in
  // `output_shards`.
  //
  // NOTE: [Output shards layout]
  // Assume that the operator has three outputs: a, b, c.
  // The layout of output_shards is as follows:
  // [ a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3]
  // This is so that we can call at::stack([a0...a3]), at::stack([b0...b3])
  // more easily in the next step.
  std::vector<Tensor> output_shards(num_batches * num_returns);
  // 策略：对于每个批次，我们将把参数的片段（如果适用）推送到`stack`，调用`op`，并将结果存储在`output_shards`中

  for (const auto linear_idx : c10::irange(num_batches)) {
    auto index = computeIndex(linear_idx, batch_sizes);
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    auto input_physical_views_iter = input_physical_views.begin();
    for (const auto arg_idx : c10::irange(num_arguments)) {
      // We assume that torch::jit::Stack is backed by vector<IValue> for
      // simplicity. When that is not the case, this code should be updated.
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || arg_idx != *batched_tensor_inputs_pos_iter) {
        // argument isn't a BatchedTensor
        torch::jit::push(stack, argument);
        continue;
      }
      // argument is a BatchedTensor
      TORCH_INTERNAL_ASSERT(input_physical_views_iter != input_physical_views.end());
      const auto& physical_view_for_argument = *input_physical_views_iter;
      torch::jit::push(stack, physical_view_for_argument.tensor().index(index));
      batched_tensor_inputs_pos_iter++;
      input_physical_views_iter++;
    }

    op.callBoxed(stack);

    // Store the result into `output_shards`. See NOTE: [Output shards layout]
    // to learn about the details of how we store the shards.
    const auto returns = torch::jit::last(stack, num_returns);
    // 将结果存储到`output_shards`中，参见NOTE: [Output shards layout]了解存储片段的详细信息
    // 对于每一个返回的索引，使用范围迭代器遍历所有返回值
    for (const auto return_idx : c10::irange(returns.size())) {
      // 将每个返回值的张量放置到输出分片中的正确位置
      output_shards[num_batches * return_idx + linear_idx] = returns[return_idx].toTensor();
    }
    // 丢弃堆栈上的返回值，数量为 num_returns
    torch::jit::drop(stack, num_returns);
  }

  // 对于每个输出张量，将张量的分片堆叠在一起形成一个返回值
  torch::jit::drop(stack, num_arguments);
  // 将输出分片切割为块，每个块的大小为 num_batches
  auto output_shards_chunks = MatrixRef<Tensor>(output_shards, num_batches);
  // 遍历每一个返回值的索引
  for (const auto return_idx : c10::irange(num_returns)) {
    // 获取当前返回值索引对应的所有分片
    auto shards = output_shards_chunks[return_idx];
    // 安全地堆叠分片，形成扁平化的输出张量
    auto flat_output = safeStack(shards);
    // 查看是否定义了扁平化的输出张量
    // 参见注释 [vmap through backward and undefined grad]
    if (!flat_output.defined()) {
      // 如果未定义，则推送一个未定义的张量到堆栈上，并继续下一个迭代
      torch::jit::push(stack, flat_output);
      continue;
    }
    // 构建输出张量的大小向量，首先插入批次大小信息
    VmapDimVector output_sizes(batch_sizes);
    // 然后插入剩余维度的大小信息
    output_sizes.insert(
        output_sizes.end(),
        flat_output.sizes().begin() + 1,
        flat_output.sizes().end());
    // 将变换后的输出张量推送到堆栈上
    torch::jit::push(
        stack,
        input_physical_views.front().getPhysicalToLogicalMap().apply(flat_output.view(output_sizes)));
  }
}

} // namespace at


注释：


// 结束命名空间 'at' 的定义
}
// 命名空间 'at' 结束
``` 

这段代码是C++中的命名空间结束语句。在这里，代码用于结束命名空间 'at' 的定义，并注释表明这是命名空间 'at' 的结束位置。
```