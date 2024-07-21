# `.\pytorch\aten\src\ATen\LegacyBatchingRegistrations.cpp`

```
// PyTorch C++ code for defining batching rules for tensor operations.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/LegacyVmapTransforms.h>
#include <ATen/LegacyBatchedFallback.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/core/IListRef.h>
#include <c10/util/irange.h>
#include <c10/core/SymIntArrayRef.h>

#include <utility>

namespace at {

// NOTE: [What is a batching rule?]
//
// A *batching rule* implements the logic of how to call an operator on inputs
// that have zero or more additional batch dimensions. When one does a vmap, the
// dimension(s) being vmap'ed over get recorded as batch dimensions.
//
// For example, vmap(torch.add)(x, y)
// 1. wraps `x` into batched_x = BatchedTensor(x, bdims=[(lvl=1, dim=0)];
// 2. wraps `y` into batched_y = BatchedTensor(y, bdims=[(lvl=1, dim=0)];
// 3. and then runs `torch.add(batched_x, batched_y)`.

// NOTE: [When should I add a batching rule?]
// When you are adding a new operator, you'll need to add a batching rule so
// that vmap can work efficiently with said operator. If you do not, we'll attempt
// to generate a slow fallback for the batching rule.

// NOTE: [How to write batching rules?]
// The signature of a batching rule should look like exactly like the C++ signature
// of its operator.
//
// First, see NOTE: [Logical vs physical args] in VmapTransforms.h for terminology.
//
// At a high level, what a batching rule does is the following:
// 1. Converts (logical) BatchedTensors to views on physical tensors.
// 2. Converts logical arguments (e.g. dimension indexes, shapes) to physical
//    arguments that correspond to the physical tensors.
// 3. Calls at:: operations on the physical tensors and arguments to produce
//    some physical results.
// 4. Converts physical results back to BatchedTensors.
//
// Steps 1, 2, and 4 differ for operators with different batching behaviors. When
// writing a new batching rule, please select a VmapTransform that matches the
// batching behavior of your operation. The VmapTransform provides helper functions
// to do steps (1), (2), and (4).
// (see NOTE: [What is an VmapTransform?] in VmapTransforms.h)

// Note: [Future plans]
// The API for writing a batching rule isn't stable. In the future, we'd like
// to think about the problem of translating these batching rules to TorchScript.
// Ideally batching rules in eager mode vs TorchScript would look pretty similar,
// if not use the same mechanism. In order to accomplish that we might have to
// do some refactoring.

namespace{

// PyTorch allows operations to specify dim 0 and dim -1 on a scalar tensor.
static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

// Define a batching rule for sum operation.
Tensor sum_batching_rule(const Tensor& self, OptionalIntArrayRef opt_dims, bool keepdim, optional<ScalarType> dtype) {
  if (opt_dims.has_value()) {
    auto dims = opt_dims.value();
    // PyTorch has a special case where sum(scalar_tensor, dim=0) does not fail
    // This function handles the batching logic for sum operation when dimensions are specified.
    // It converts BatchedTensors to views on physical tensors, computes the sum along specified dimensions,
    // and returns the result as a BatchedTensor.
    // 如果张量是标量（scalar）且维度为0，或者维度为空或者仅包含允许在标量张量上操作的维度，
    // 则进行特殊处理：返回一个新的标量张量（对于dim=-1也是如此）
    // 例如：
    // >>> x = torch.randn(B0)  # 每个示例都是标量
    // >>> vmap(partial(torch.sum, dim=0), x)
    // 则会复制对标量张量在dim=0维度上求和的行为。
    if (/*logical*/self.dim() == 0 && (dims.empty() || (dims.size() == 1 && is_allowed_dim_on_scalar_tensor(dims[0])))) {
      // 如果满足条件，则克隆当前张量并返回
      return self.clone();
    }
  }
  // 将逻辑维度转换为物理维度，以便在物理维度上进行操作
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度的操作维度
  auto dims_physical = self_physical.getPhysicalDims(opt_dims);
  // 在物理张量上执行求和操作，得到结果
  auto result = at::sum(self_physical.tensor(), dims_physical, keepdim, dtype);
  // 将物理维度映射回逻辑维度，返回映射后的结果
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 判断逻辑张量是否为物理标量张量
bool isPhysicalScalarTensor(const Tensor& logical_tensor) {
  // 如果张量的维度大于0，返回false
  if (logical_tensor.dim() > 0) {
    return false;
  }
  // 尝试获取批处理实现，如果存在，则返回false
  auto* batched = maybeGetBatchedImpl(logical_tensor);
  if (batched) {
    return false;
  }
  // 否则返回true，表示是物理标量张量
  return true;
}

// 二进制逐点批处理规则模板函数
template <typename F, F Func, typename... ExtraArgs>
Tensor binary_pointwise_batching_rule(
    const Tensor& self, const Tensor& other, ExtraArgs... args) {
  // 如果self和other都有维度大于0，则执行以下操作
  if (self.dim() > 0 && other.dim() > 0) {
    // 将逻辑张量转换为物理张量，进行广播变换
    auto physical_args = BroadcastingVmapTransform::logicalToPhysical({self, other});
    // 调用指定的函数处理物理张量，并获取结果
    auto result = Func(physical_args[0].tensor(), physical_args[1].tensor(), args...);
    // 将处理结果映射回逻辑张量空间，并返回结果
    return physical_args[0].getPhysicalToLogicalMap().apply(result);
  }
  // 如果self是物理标量张量
  if (isPhysicalScalarTensor(self)) {
    // 将other转换为物理张量
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    // 调用指定的函数处理self和物理张量other，并获取结果
    auto result = Func(self, other_physical.tensor(), args...);
    // 将处理结果映射回逻辑张量空间，并返回结果
    return other_physical.getPhysicalToLogicalMap().apply(result);
  }
  // 如果other是物理标量张量
  if (isPhysicalScalarTensor(other)) {
    // 将self转换为物理张量
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    // 调用指定的函数处理物理张量self和other，并获取结果
    auto result = Func(self_physical.tensor(), other, args...);
    // 将处理结果映射回逻辑张量空间，并返回结果
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }

  // 至此，至少一个操作数是逻辑标量张量的情况
  // 在这里，我们必须模拟TensorIterator在标量张量上的特殊行为。
  //
  // 举例说明：
  //   x = torch.randn(3, 10)
  //   y = torch.randn(3, dtype=torch.double)
  //   vmap(torch.mul)(torch.randn(3, 10), torch.randn(3, dtype=torch.double))
  //
  // 在每个示例级别上，我们正在添加FloatTensor[10]和DoubleTensor[]；
  // 类型提升规定结果应为FloatTensor[10]。
  // 这意味着我们不能直接将物理张量（x和y）传递给TensorIterator
  // （如果这样做，它会将它们提升为DoubleTensor）。
  //
  // FIXME(rzou): 我不想沿着模拟TensorIterator的滑坡继续走下去
  // （最好重构出TensorIterator逻辑）。此代码唯一不处理的是跨设备逻辑标量张量的情况。
  //   cpu_tensor = torch.randn(3)
  //   cuda_tensor = torch.randn(3, 10, device='cuda')
  //   vmap(torch.mul)(cpu_tensor, cuda_tensor)
  //
  // 在每个示例级别上，我们正在添加CPUTensor[]和CUDATensor[10]。
  // TensorIterator允许这种跨设备操作，因为其中一个张量是标量CPU张量。
  // 然而，在这种情况下，以下代码将抛出错误。我不希望看到这种情况的使用案例，所以
  // 这应该是目前的正常情况。
  auto logical_self = self;
  auto logical_other = other;
  // 确定结果类型
  auto result_type = at::native::result_type(logical_self, logical_other);
  // 如果logical_self的标量类型与结果类型不同，将其转换为结果类型
  if (logical_self.scalar_type() != result_type) {
    logical_self = logical_self.to(result_type);
  }
  // 如果logical_other的标量类型与结果类型不同，将其转换为结果类型
  if (logical_other.scalar_type() != result_type) {
    // 将 logical_other 转换为指定的 result_type 类型
    logical_other = logical_other.to(result_type);
  }
  // 使用 BroadcastingVmapTransform::logicalToPhysical 方法将逻辑张量转换为物理张量
  auto physical_args = BroadcastingVmapTransform::logicalToPhysical(
      {std::move(logical_self), std::move(logical_other)});
  // 调用 Func 函数，传入物理张量及额外的参数 args，并存储结果
  auto result = Func(physical_args[0].tensor(), physical_args[1].tensor(), args...);
  // 将结果应用于物理到逻辑映射，并返回映射后的结果
  return physical_args[0].getPhysicalToLogicalMap().apply(result);
}

// 扩展规则：将逻辑张量扩展到指定尺寸
Tensor expand_batching_rule(const Tensor& self, IntArrayRef size, bool implicit) {
  // 将逻辑张量转换为物理张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理尺寸
  auto size_physical = self_physical.getPhysicalShape(size);
  // 获取逻辑张量的维度
  auto self_physical_dim = self_physical.tensor().dim();

  // 检查尺寸是否合适
  TORCH_CHECK(self_physical_dim <= static_cast<int64_t>(size_physical.size()),
       "expand: the number of sizes provided (", /*logical*/size.size(), ") ",
       "must be greater or equal to the number of dimensions in the tensor (",
       /*logical dim*/self.dim(), ")");

  // 如果逻辑张量的维度与物理尺寸相同，则直接扩展
  if (self_physical_dim == static_cast<int64_t>(size_physical.size())) {
    auto result = self_physical.tensor().expand(size_physical, implicit);
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }

  // 否则，处理存在批处理维度的情况下的扩展
  TORCH_INTERNAL_ASSERT(self_physical_dim < static_cast<int64_t>(size_physical.size()));
  // 这里，我们扩展一个逻辑张量到更多维度。由于存在批处理维度，不能直接调用扩展函数。
  //
  // 举例说明，假设 B0 是批处理维度，考虑 expand(Tensor[B0, 3], [2, 3])。
  // 结果应该是大小为 [B0, 2, 3] 的张量。
  // 大小为 [B0, 3] 的物理视图不能直接扩展为大小为 [B0, 2, 3]，
  // 所以这里的策略是首先将其视为大小为 [B0, 1, 3] 的张量，然后再扩展。
  auto self_physical_size = self_physical.tensor().sizes();
  auto extra_dims = size_physical.size() - self_physical_dim;
  VmapDimVector view_shape(size_physical.size(), 1);
  std::copy(self_physical_size.begin(),
            self_physical_size.begin() + self_physical.numBatchDims(),
            view_shape.begin());
  std::copy(self_physical_size.begin() + self_physical.numBatchDims(),
            self_physical_size.end(),
            view_shape.begin() + self_physical.numBatchDims() + extra_dims);
  auto result = self_physical.tensor().view(view_shape).expand(size_physical, implicit);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 分块规则：将逻辑张量按指定维度分块
std::vector<Tensor> chunk_batching_rule(const Tensor& self, int64_t chunks, int64_t dim) {
  // 将逻辑张量转换为物理张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 对逻辑张量进行分块操作
  auto result = at::chunk(self_physical.tensor(), chunks, dim_physical);
  // 将分块结果映射回逻辑张量
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

// 夹紧规则：对逻辑张量进行夹紧操作
Tensor clamp_batching_rule(const Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max) {
  // 将逻辑张量转换为物理张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 对物理张量进行夹紧操作
  auto result = at::clamp(self_physical.tensor(), min, max);
  // 将夹紧结果映射回逻辑张量
  return self_physical.getPhysicalToLogicalMap().apply(result);
}
Tensor clamp_min_batching_rule(const Tensor& self, const Scalar& min) {
  // 将逻辑表示的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 对物理表示的张量进行最小值截断操作
  auto result = at::clamp_min(self_physical.tensor(), min);
  // 将结果映射回逻辑表示
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor clamp_max_batching_rule(const Tensor& self, const Scalar& max) {
  // 将逻辑表示的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 对物理表示的张量进行最大值截断操作
  auto result = at::clamp_max(self_physical.tensor(), max);
  // 将结果映射回逻辑表示
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

std::vector<Tensor> tensor_split_sections_batching_rule(const Tensor& self, int64_t sections, int64_t dim) {
  // 将逻辑表示的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理表示中指定维度的物理维度
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 在指定维度上对物理表示的张量进行分割操作
  auto result = at::tensor_split(self_physical.tensor(), sections, dim_physical);
  // 将分割后的结果映射回逻辑表示
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> tensor_split_indices_batching_rule(const Tensor& self, IntArrayRef indices, int64_t dim) {
  // 将逻辑表示的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理表示中指定维度的物理维度
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 在指定维度上根据给定索引对物理表示的张量进行分割操作
  auto result = at::tensor_split(self_physical.tensor(), indices, dim_physical);
  // 将分割后的结果映射回逻辑表示
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

Tensor unsqueeze_batching_rule(const Tensor& self, int64_t dim) {
  // 将逻辑表示的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 注意：unsqueeze 函数对维度参数 `dim` 有特殊处理，需根据逻辑维度调整
  // 这里使用 `maybe_wrap_dim` 函数进行调整，保证操作正确性
  auto dim_physical =
      self_physical.numBatchDims() + maybe_wrap_dim(dim, /*logical_dim*/self.dim() + 1);
  // 在物理表示的张量上进行 unsqueeze 操作
  auto result = self_physical.tensor().unsqueeze(dim_physical);
  // 将操作结果映射回逻辑表示
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor& fill_inplace_scalar_batching_rule(Tensor& self, const Scalar& value) {
  // 将逻辑表示的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 在物理表示的张量上进行填充操作（原地）
  self_physical.tensor().fill_(value);
  // 返回自身引用
  return self;
}

Tensor& fill_inplace_tensor_batching_rule(Tensor& self, const Tensor& value) {
  // 检查填充值是否为批处理张量
  auto value_batched = isBatchedTensor(value);

  if (value_batched) {
    // 如果填充值是批处理张量，则将自身和填充值一起转换为物理表示
    auto physical_args =
      BroadcastingVmapTransform::logicalToPhysical({self, value});
    // 在物理表示上进行数据复制操作
    physical_args[0].tensor().copy_(physical_args[1].tensor());
  } else {
    // 否则，只将逻辑表示的自身转换为物理表示
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    // 在物理表示的张量上进行填充操作（原地）
    self_physical.tensor().fill_(value);
  }
  // 返回自身引用
  return self;
}

Tensor& zero_inplace_batching_rule(Tensor &self) {
  // 将逻辑表示的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 在物理表示的张量上进行零填充操作（原地）
  self_physical.tensor().zero_();
  // 返回自身引用
  return self;
}
// 根据输入张量自动调整批处理维度的压缩规则
Tensor squeeze_batching_rule(const Tensor& self) {
  // 将逻辑上的输入张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理表示张量的尺寸信息
  auto physical_sizes = self_physical.tensor().sizes();

  // 不对批处理维度进行压缩处理
  VmapDimVector squeezed_sizes;
  // 获取批处理维度的数量
  int64_t num_batch_dims = self_physical.numBatchDims();
  // 将批处理维度的尺寸信息加入到压缩后的尺寸列表中
  squeezed_sizes.insert(
      squeezed_sizes.end(),
      physical_sizes.begin(),
      physical_sizes.begin() + num_batch_dims);
  // 遍历剩余的维度，将不为1的维度加入到压缩后的尺寸列表中
  for (auto it = physical_sizes.begin() + num_batch_dims; it != physical_sizes.end(); ++it) {
    if (*it != 1) {
      squeezed_sizes.push_back(*it);
    }
  }

  // 根据压缩后的尺寸重新视图化物理表示的张量
  auto result = self_physical.tensor().view(squeezed_sizes);
  // 将结果张量映射回逻辑表示并返回
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 根据指定维度压缩批处理的维度规则
Tensor squeeze_dim_batching_rule(const Tensor& self, int64_t dim) {
  // 将逻辑上的输入张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取指定维度在物理表示中的索引
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 对指定维度进行压缩操作
  auto result = self_physical.tensor().squeeze(dim_physical);
  // 将结果张量映射回逻辑表示并返回
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 根据指定维度列表压缩批处理的维度规则
Tensor squeeze_dims_batching_rule(const Tensor& self, IntArrayRef dims) {
  // 将逻辑上的输入张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取指定维度列表在物理表示中的索引
  auto dims_physical = self_physical.getPhysicalDims(dims);
  // 对指定维度列表进行压缩操作
  auto result = self_physical.tensor().squeeze(dims_physical);
  // 将结果张量映射回逻辑表示并返回
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 对批处理的张量进行跟踪规则
Tensor trace_batching_rule(const Tensor& self) {
  // 将逻辑上的输入张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取批处理对角视图
  auto self_diag = at::diagonal(self_physical.tensor(), /*offset*/0, /*dim1*/-2, /*dim2*/-1);
  // 在最后一个维度上对对角视图进行求和
  auto result =  at::sum(self_diag, -1);
  // 将结果张量映射回逻辑表示并返回
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 对跟踪操作的反向传播进行批处理规则
Tensor trace_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes) {
  // 将逻辑上的梯度张量转换为物理表示
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  // 创建与输入尺寸匹配的零张量作为梯度输入
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  // 获取梯度输入的对角视图
  auto grad_input_diag = at::diagonal(grad_input, /*offset*/0, /*dim1*/-2, /*dim2*/-1);
  // 将物理表示的梯度张量复制到对角视图中
  auto grad_physical_tensor = grad_physical.tensor().unsqueeze(-1);
  grad_input_diag.copy_(grad_physical_tensor);
  // 将结果张量映射回逻辑表示并返回
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}

// 对整数维度进行转置的批处理规则
Tensor transpose_int_batching_rule(const Tensor& self, int64_t dim0, int64_t dim1) {
  // 如果输入张量是标量且维度0和维度1都是允许的标量张量维度
  if (/*logical*/self.dim() == 0 && is_allowed_dim_on_scalar_tensor(dim0) &&
      is_allowed_dim_on_scalar_tensor(dim1)) {
    return self;
  }



    // 返回自身对象，结束函数
    return self;
  }



  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);



    // 将输入张量 self 转换为物理布局
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);



  auto dim0_physical = self_physical.getPhysicalDim(dim0);



    // 获取转换后张量 self_physical 在维度 dim0 上的物理维度
    auto dim0_physical = self_physical.getPhysicalDim(dim0);



  auto dim1_physical = self_physical.getPhysicalDim(dim1);



    // 获取转换后张量 self_physical 在维度 dim1 上的物理维度
    auto dim1_physical = self_physical.getPhysicalDim(dim1);



  auto result = self_physical.tensor().transpose(dim0_physical, dim1_physical);



    // 对 self_physical 的张量进行维度 dim0_physical 和 dim1_physical 的转置操作，并赋值给 result
    auto result = self_physical.tensor().transpose(dim0_physical, dim1_physical);



  return self_physical.getPhysicalToLogicalMap().apply(result);



    // 返回 self_physical 的物理到逻辑映射所作用于 result 的结果
    return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 定义函数：重排批处理规则，对张量进行维度重排
Tensor permute_batching_rule(const Tensor& self, IntArrayRef dims) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度的排列顺序
  auto dims_physical = self_physical.getPhysicalDims(dims);

  // 创建存储所有物理维度的向量
  VmapDimVector all_dims_physical;
  all_dims_physical.reserve(self_physical.tensor().dim());
  // 遍历所有批处理维度，并添加到物理维度向量中
  for (const auto bdim : c10::irange(self_physical.numBatchDims())) {
    all_dims_physical.push_back(bdim);
  }
  // 将给定的维度插入到物理维度向量的末尾
  all_dims_physical.insert(
      all_dims_physical.end(),
      dims_physical.begin(),
      dims_physical.end());
  // 使用物理维度重新排列张量
  auto result = self_physical.tensor().permute(all_dims_physical);
  // 将物理到逻辑映射应用于结果张量，并返回
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 定义函数：选择批处理规则，从指定维度中选择指定索引的元素
Tensor select_batching_rule(const Tensor& self, int64_t dim, int64_t index) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取指定维度的物理维度索引
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 从物理张量中选择指定维度和索引的元素
  auto result = self_physical.tensor().select(dim_physical, index);
  // 将物理到逻辑映射应用于结果张量，并返回
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 定义静态函数：获取梯度输入的物理维度
static int64_t getGradInputPhysicalDim(int64_t dim, IntArrayRef input_sizes, int64_t num_batch_dims) {
  // 返回包装后的维度，用于梯度输入的物理维度计算
  return maybe_wrap_dim(dim, input_sizes.size()) + num_batch_dims;
}

// 定义函数：选择反向批处理规则，用于梯度的选择操作
Tensor select_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t index) {
  // 将逻辑上的梯度张量转换为物理上的张量表示
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  // 创建与输入大小相同的零张量作为梯度输入
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  // 获取梯度输入的物理维度索引
  auto physical_dim = getGradInputPhysicalDim(dim, input_sizes, grad_physical.numBatchDims());
  // 从梯度物理张量中选择指定维度和索引的元素，并将梯度值复制到梯度输入张量中
  grad_input.select(physical_dim, index).copy_(grad_physical.tensor());
  // 将物理到逻辑映射应用于梯度输入张量，并返回
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}

// 定义函数：切片批处理规则，用于对张量进行切片操作
Tensor slice_batching_rule(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取指定维度的物理维度索引
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 对物理张量进行切片操作，指定起始、结束和步长
  auto result = self_physical.tensor().slice(dim_physical, start, end, step);
  // 将物理到逻辑映射应用于结果张量，并返回
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 定义函数：切片反向批处理规则，用于梯度的切片操作
Tensor slice_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  // 将逻辑上的梯度张量转换为物理上的张量表示
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  // 创建与输入大小相同的零张量作为梯度输入
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  // 获取梯度输入的物理维度索引
  auto physical_dim = getGradInputPhysicalDim(dim, input_sizes, grad_physical.numBatchDims());
  // 对梯度物理张量进行切片操作，指定起始、结束和步长，并将梯度值复制到梯度输入张量中
  grad_input.slice(physical_dim, start, end, step).copy_(grad_physical.tensor());
  // 将物理到逻辑映射应用于梯度输入张量，并返回
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}
Tensor diagonal_batching_rule(const Tensor& self, int64_t offset, int64_t dim1, int64_t dim2) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度 dim1 的索引
  auto dim1_physical = self_physical.getPhysicalDim(dim1);
  // 获取物理维度 dim2 的索引
  auto dim2_physical = self_physical.getPhysicalDim(dim2);
  // 调用 ATen 函数计算张量的对角线元素
  auto result = at::diagonal(self_physical.tensor(), offset, dim1_physical, dim2_physical);
  // 将物理张量表示转换回逻辑张量表示，并应用到结果上
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor diagonal_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  // 将逻辑上的梯度张量转换为物理上的张量表示
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  // 创建与输入尺寸相匹配的零张量
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  // 获取物理维度 dim1 的索引
  auto dim1_physical = getGradInputPhysicalDim(dim1, input_sizes, grad_physical.numBatchDims());
  // 获取物理维度 dim2 的索引
  auto dim2_physical = getGradInputPhysicalDim(dim2, input_sizes, grad_physical.numBatchDims());
  // 在梯度输入的对角线上复制梯度张量
  grad_input.diagonal(offset, dim1_physical, dim2_physical).copy_(grad_physical.tensor());
  // 将物理张量表示转换回逻辑张量表示，并应用到结果上
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}

Tensor movedim_batching_rule(const Tensor& self, IntArrayRef source, IntArrayRef destination) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取源维度的物理索引
  auto source_physical = self_physical.getPhysicalDims(source);
  // 获取目标维度的物理索引
  auto destination_physical = self_physical.getPhysicalDims(destination);
  // 调用 ATen 函数移动张量的维度
  auto result = at::movedim(self_physical.tensor(), source_physical, destination_physical);
  // 将物理张量表示转换回逻辑张量表示，并应用到结果上
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor reshape_batching_rule(const Tensor& self, IntArrayRef shape) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理形状的张量表示
  auto shape_physical = self_physical.getPhysicalShape(shape);
  // 调用张量的重塑操作
  auto result = self_physical.tensor().reshape(shape_physical);
  // 将物理张量表示转换回逻辑张量表示，并应用到结果上
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

std::vector<Tensor> split_batching_rule(const Tensor& self, int64_t split_size, int64_t dim) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度 dim 的索引
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 调用 ATen 函数在指定维度上分割张量
  auto result = at::split(self_physical.tensor(), split_size, dim_physical);
  // 将物理张量表示转换回逻辑张量表示，并在原地应用到结果上
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> split_with_sizes_batching_rule(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度 dim 的索引
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 调用 ATen 函数在指定维度上使用指定大小分割张量
  auto result = at::split_with_sizes(self_physical.tensor(), split_sizes, dim_physical);
  // 将物理张量表示转换回逻辑张量表示，并在原地应用到结果上
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}
// 对输入的张量按照给定维度进行解绑定，返回解绑后的张量列表
std::vector<Tensor> unbind_batching_rule(const Tensor& self, int64_t dim) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取指定维度的物理维度表示
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 在物理张量上执行解绑操作
  auto result = at::unbind(self_physical.tensor(), dim_physical);
  // 将物理到逻辑映射应用到解绑后的张量列表上
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

// 在指定维度上对张量进行展开操作，返回展开后的张量
Tensor unfold_batching_rule(const Tensor& self, int64_t dim, int64_t size, int64_t step) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取指定维度的物理维度表示
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 在物理张量上执行展开操作
  auto result = self_physical.tensor().unfold(dim_physical, size, step);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 将张量转换为指定内存格式的连续张量，并返回
Tensor contiguous_batching_rule(const Tensor& self, MemoryFormat memory_format) {
  // 检查内存格式是否为连续格式
  TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
      "NYI: Tensor.contiguous(...) inside of vmap for memory_format other ",
      "than torch.contiguous_format");
  // 将逻辑上的张量转换为物理上的张量表示
  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
  // 转换为指定内存格式的连续张量
  auto result = physical_view.tensor().contiguous(memory_format);
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

// 将张量按照给定大小进行重塑，并返回重塑后的张量
Tensor view_batching_rule(const Tensor& self, IntArrayRef size) {
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取按照物理维度表示的重塑大小
  auto size_physical = self_physical.getPhysicalShape(size);
  // 在物理张量上执行重塑操作
  auto result = self_physical.tensor().view(size_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 将实部张量视作复数张量，并返回结果张量
Tensor view_as_complex_batching_rule(const Tensor& self) {
  // 检查输入张量的维度不为空
  TORCH_CHECK(!self.sizes().empty(), "Input tensor must have one or more dimensions");
  // 将逻辑上的张量转换为物理上的张量表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 将物理上的张量视作复数张量
  auto result = at::view_as_complex(self_physical.tensor());
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// 检查在布局中，最小的批处理步长是否大于最大的示例步长
static void checkBatchDimsAtFrontInLayout(IntArrayRef physical_strides, int64_t num_batch_dims) {
  // 找出批处理维度中最小的步长
  auto smallest_batch_stride = std::min_element(
      physical_strides.begin(), physical_strides.begin() + num_batch_dims);
  // 找出示例维度中最大的步长
  auto largest_example_stride = std::max_element(
      physical_strides.begin() + num_batch_dims, physical_strides.end());
  if (largest_example_stride == physical_strides.end()) {
    // 如果没有示例维度，直接返回
    return;
  }
  // 检查最小的批处理步长是否大于等于最大的示例步长
  TORCH_CHECK(*smallest_batch_stride >= *largest_example_stride,
    "vmap: Calling Tensor.as_strided is not supported unless the batch dims being ",
    "vmapped over are at the front of the tensor (in memory layout). When they are ",
    "not at the front of the tensor this operation can be error prone so we "
    "actively discourage it; please file us a bug report and/or try to ");
}
    "express the as_strided operation in terms of PyTorch view operations");



# 创建一个包含字符串的Python字符串对象，描述了将 as_strided 操作用 PyTorch 视图操作表达的目的
}

// 给定 sizes、strides 和 storage_offset，返回可以索引的最大位置
// 如果不存在这样的位置（例如，具有零大小维度的张量），返回 nullopt
static optional<int64_t> maximum_indexable_location(
    IntArrayRef sizes, IntArrayRef strides, int64_t storage_offset) {
  // 调用 native::storage_size_for 函数计算存储大小
  auto result = native::storage_size_for(sizes, strides);
  // 如果结果为 0，返回 nullopt
  if (result == 0) {
    return nullopt;
  }
  // 返回结果加上 storage_offset
  return result + storage_offset;
}

// 让 x 成为 physical_tensor 的“第一个切片”。
// 这个函数检查通过 x.as_strided(sizes, strides, maybe_storage_offset) 访问的内存位置范围
// 是否在 x 可访问的内存位置范围内。
static void checkBasicAsStridedValidForSlice(
    const Tensor& physical_tensor,
    int64_t num_batch_dims,
    IntArrayRef sizes,
    IntArrayRef strides,
    optional<int64_t> maybe_storage_offset) {
  // 获取 physical_tensor 切片后的 sizes 和 strides
  auto slice_sizes = physical_tensor.sizes().slice(num_batch_dims);
  auto slice_strides = physical_tensor.strides().slice(num_batch_dims);
  // 获取基础偏移量
  auto base_offset = physical_tensor.storage_offset();

  // 获取存储偏移量，如果 maybe_storage_offset 不存在，则使用 base_offset
  auto storage_offset = maybe_storage_offset.value_or(base_offset);

  // 计算 as_strided 操作的最大可索引位置
  auto max_as_strided_loc = maximum_indexable_location(sizes, strides, storage_offset);
  // 计算切片的最大可索引位置
  auto max_slice_loc = maximum_indexable_location(slice_sizes, slice_strides, base_offset);

  // 如果 as_strided 的最大位置不存在，直接返回
  if (!max_as_strided_loc.has_value()) {
    return;
  }
  // 如果切片的最大位置不存在，抛出错误
  if (!max_slice_loc.has_value()) {
    TORCH_CHECK(false,
        "result = tensor.as_strided(", sizes, ",",  strides, ",", storage_offset, ")",
        "can access memory outside of `tensor`. `tensor` has no storage but the ",
        "passed-in (size, stride, storage_offset) imply a result with some storage. ",
        "This is not supported inside of vmap, please try to rewrite the ",
        "`as_strided` call as a sequence of PyTorch view operations");
  }

  // 检查 as_strided 操作的最大位置是否在切片的最大位置内，并且基础偏移量是否在存储偏移量内
  TORCH_CHECK(
      *max_as_strided_loc <= *max_slice_loc && base_offset <= storage_offset,
      "result = tensor.as_strided(", sizes, ",",  strides, ",", storage_offset, ")",
      "can access memory outside of `tensor`. `result` can access some",
      "memory in range [", storage_offset, ", ", *max_as_strided_loc, "], but ",
      "`tensor` can only access some memory in range [", base_offset, ", ",
      *max_slice_loc, "]. This is not supported inside of vmap, please try to",
      "rewrite the `as_strided` call as a sequence of PyTorch view operations");
}

// 用于处理 _reshape_alias_batching_rule 函数的别名批处理规则
Tensor _reshape_alias_batching_rule(const Tensor& self, IntArrayRef sizes, IntArrayRef strides) {
  // 调用 reshape_batching_rule 函数，返回结果
  return reshape_batching_rule(self, sizes);
}

// 用于处理 _new_zeros_with_same_feature_meta_batching_rule 函数的别名批处理规则
Tensor _new_zeros_with_same_feature_meta_batching_rule(
    const Tensor& self,
    const Tensor& other,
    int64_t unused_num_batch_dims) {
  // 检查是否支持 PyTorch 核心中的“批次梯度”使用情况
  TORCH_CHECK(isBatchedTensor(self) && !isBatchedTensor(other),
    "Only the 'batched grad' use case is supported in PyTorch core.");

  // 内部断言，验证 unused_num_batch_dims 是否为 0
  TORCH_INTERNAL_ASSERT(unused_num_batch_dims == 0,
    "num_batch_dims should not be explicitly passed in because it will be overridden");

# 提示：不应显式传递 `num_batch_dims`，因为它将被覆盖


  auto self_physical_view = at::MultiBatchVmapTransform::logicalToPhysical(self);

# 创建 `self` 的物理视图，用于多批次 Vmap 变换


  const auto& self_physical_tensor = self_physical_view.tensor();

# 获取 `self` 的物理张量视图


  int64_t num_batch_dims = self_physical_view.numBatchDims();

# 获取 `self` 物理视图中的批次维数数量


  checkBatchDimsAtFrontInLayout(self_physical_tensor.strides(), num_batch_dims);

# 检查在布局中是否批次维数在前，使用 `self` 的张量步长和批次维数


  auto result = at::_new_zeros_with_same_feature_meta(self_physical_tensor, other, num_batch_dims);

# 创建一个与 `self` 物理张量具有相同特征元数据的新零张量，与 `other` 的形状匹配，并传入 `num_batch_dims`


  return self_physical_view.getPhysicalToLogicalMap().apply(result);

# 将结果张量 `result` 应用物理到逻辑映射，返回逻辑视图下的结果
}

// 检查是否具有相同的存储大小批处理规则
bool _has_same_storage_numel_batching_rule(const Tensor& self, const Tensor& other) {
  // 如果self是批处理张量而other不是，则抛出错误，只支持'batched grad'用例
  TORCH_CHECK(isBatchedTensor(self) && !isBatchedTensor(other),
    "Only the 'batched grad' use case is supported in PyTorch core.");

  // 如果tangent是批处理张量，则跳过_has_same_storage_numel检查，
  // 因为在vmap中不支持使用as_strided访问输入张量无法索引的存储位置
  return true;
}

// 在vmap中as_strided的语义是什么？
// y = vmap(lambda x: x.as_strided(sizes, strides, offset))(xs)
// 这返回x的视图y，使得每个y[i]具有:
// - sizes: `sizes`
// - strides: `strides`
// - storage_offset: offset + i * x.stride(batch_dim)
//
// 换句话说，就好像我们将每个x[i]视为具有存储偏移量等于xs.offset()，
// 然后调用as_strided(sizes, sizes, offset)一样。
// （这相当于对所有i，x[i].as_strided(
//    sizes, sizes, offset + x[i].storage_offset() - xs.offset())）
//
// 注意，这可能与在for循环中实际运行as_strided不同。这是因为as_strided接受`offset`作为*绝对*偏移量的原因。
// 例如，考虑：
// >>> x = torch.tensor([0., 1., 2., 3., 4.]).as_strided([4], [1], 1)
// >>> z = [x[i].as_strided([1], [1], 1) for i in range(4)]
// 每个z[i]实际上是x的相同视图（z[i] == torch.tensor([1.])）！
// 然而，我们认为上面的for循环理解是用户错误：
// 如果用户想要以每个样本的方式使用as_strided，则应该写如下内容：
// >>> z = [x[i].as_strided([1], [1], 1 + x[i].storage_offset() - 1) for i in range(4)]
Tensor as_strided_batching_rule(
    const Tensor& tensor,
    IntArrayRef sizes,
    IntArrayRef strides,
  optional<int64_t> storage_offset) {
  // 将逻辑视图转换为物理视图
  auto physical_view = at::MultiBatchVmapTransform::logicalToPhysical(tensor);
  // 获取批次维度数量
  auto num_batch_dims = physical_view.numBatchDims();
  // 获取物理尺寸
  auto physical_sizes = physical_view.getPhysicalShape(sizes);
  // 获取物理张量的常量引用
  const auto& physical_tensor = physical_view.tensor();

  // 检查大小和步幅长度是否相同
  TORCH_CHECK(sizes.size() == strides.size(),
      "Tensor.as_strided(size, stride, ...): size and stride must have the ",
      "same length! Got size ", sizes, " and stride ", strides);

  // 执行一些合法性检查：
  // 1. 所有批次维度必须在内存布局的最前面（这不影响正确性，但是为了避免用户可能进行不合理的操作）
  // 2. as_strided(sizes, strides, storage_offset + tensor[i].offset() - tensor.offset())
  //    对输入张量的一个切片是有效的。详见注释: [当 as_strided 批处理规则会失败时？]。
  checkBatchDimsAtFrontInLayout(physical_tensor.strides(), num_batch_dims);
  checkBasicAsStridedValidForSlice(
      physical_tensor, num_batch_dims, sizes, strides, storage_offset);

  // 计算物理步幅：物理张量的批次步幅 + 逻辑步幅
  auto batch_strides = physical_tensor.strides().slice(0, num_batch_dims);
  at::VmapDimVector physical_strides;
  physical_strides.reserve(num_batch_dims + strides.size());
  physical_strides.insert(
      physical_strides.end(), batch_strides.begin(), batch_strides.end());
  physical_strides.insert(
      physical_strides.end(), strides.begin(), strides.end());

  // 如果对所有 i，zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
  // 都是合法的，那么事实证明 xs.as_strided(physical_sizes, physical_strides, offset)
  // 总是成功的，并创建一个张量 y，使得每个 y[i] 引用与 zi 相同的内存位置。
  // 见注释: [当 as_strided 批处理规则会失败时？]
  auto result = physical_view.tensor().as_strided(
      physical_sizes, physical_strides, storage_offset);
  // 应用物理到逻辑的映射，返回结果
  return physical_view.getPhysicalToLogicalMap().apply(result);
}
}

// NOTE: [When will the as_strided batching rule fail?]
// If zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid for all i, then it turns out that
// xs.as_strided(physical_sizes, physical_strides, offset) always succeeds and
// creates a tensor y such that each y[i] refers to the same memory as zi.
//
// Let's say we have xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()).
// Furthermore, let's say that as a part of being "valid" this as_strided call
// does not return a result that can index memory not indexable by xs[i].
//
// WLOG, assume that there's only one batch dim and it is at the front of the
// `xs` tensor. Let B be the batch size and S be the stride of the batch dim.
// - If the batch dim isn't at the front of the tensor, then we can just move it
// to the front with movedim/permute. This is always valid because it just swaps
// some strides around.
// - This proof also works for tensors with multiple batch dims. We just have to
// do a little accounting:
//   - instead of [B], we'd have [B0, B1, ..., Bk].
//   - instead of [S], we'd have [S0, S1, ..., Sk].
//   - instead of i, we'd have a list of indices [I0, I1, ..., Ik]
//   - instead of S * I, we'd have \sum_{i=0}^k S_i * I_i
//
// [Equation 1]
// xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()) has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
//
// x.as_strided itself checks that:
// - (sizes, strides, offset) are in bounds for `x`'s storage.
// - strides are positive
// - offset is positive
//
// Claim 1: if xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid, then
// ([B] + sizes, [S] + strides, offset + xs.offset()) are in bounds for `xs`'s storage.
//
// If we have the claim, then xs.as_strided([B] + sizes, [S] + strides, offset)
// won't error out. So all we need to check is that the memory locations are
// what we expected. See [Hand-wavy proof of Claim 1] for proof (it's not very important)
//
// xs.as_strided(physical_sizes, physical_strides, offset) is equivalent to
// xs.as_strided([B] + sizes, [S] + strides, offset)
//
// xs.as_strided([B] + sizes, [S] + strides, offset) has:
// - sizes: [B] + sizes
// - strides: [S] + strides
// - offset: offset
//
// xs.as_strided([B] + sizes, [S] + strides, offset)[i] has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
// These memory locations are exactly the same as what we got for [Equation 1],
// so the xs.as_strided([B] + sizes, [S] + strides, offset) is valid.
//
// [Hand-wavy proof of Claim 1]
// Part of our definition of being valid is that xs[i].as_strided(...)
// must return a tensor that only uses memory indexable by xs[i].
// This means that (sizes, strides, offset + xs[i].offset() - xs.offset()) satisfies:
//    offset + xs[i].offset() - xs.offset() + 1 + \sum_j (sizes[j] - 1) * strides[j]
// 实现了一个模板函数，用于解封包含批处理信息的张量，并调用函数 Func 处理其物理值，并重新打包成批处理张量返回。
template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call(const Tensor& input, ExtraArgs... args) {
  // 获取输入张量的批处理实现指针
  auto* input_batched = unsafeGetBatchedImpl(input);
  // 调用 Func 处理批处理值，并获取物理输出
  auto output_physical = Func(input_batched->value(), args...);
  // 获取旧的批处理维度
  auto old_bdims = input_batched->bdims();
  // 构造包含新批处理维度的批处理张量
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

// 实现了一个模板函数，用于解封包含批处理信息的张量，并调用成员函数 Func 处理其物理值，并重新打包成批处理张量返回。
template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call_method(const Tensor& input, ExtraArgs... extra_args) {
  // 获取输入张量的批处理实现指针
  auto* input_batched = unsafeGetBatchedImpl(input);
  // 调用成员函数 Func 处理批处理值，并获取物理输出
  auto output_physical = (input_batched->value().*Func)(extra_args...);
  // 获取旧的批处理维度
  auto old_bdims = input_batched->bdims();
  // 构造包含新批处理维度的批处理张量
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

// 实现了一个函数，用于在标量和张量之间执行幂操作的批处理规则。
Tensor pow_scalar_Tensor_batching_rule(const Scalar& other, const Tensor& self) {
  // 获取输入张量的批处理实现指针
  auto* self_batched = unsafeGetBatchedImpl(self);
  // 调用 ATen 库的 pow 函数处理批处理值，并获取物理输出
  auto output_physical = at::pow(other, self_batched->value());
  // 获取旧的批处理维度
  auto old_bdims = self_batched->bdims();
  // 构造包含新批处理维度的批处理张量
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}
// 检查是否指定了内存格式，限制在 vmap 内使用的克隆操作的内存格式
Tensor clone_batching_rule(const Tensor& self, optional<MemoryFormat> memory_format) {
  // 内存格式支持有些复杂，因为 vmap 允许移动批处理维度，并且某些内存格式依赖于张量的秩。
  // 另一个奇怪的情况是：
  // - 具有 MemoryFormat::ChannelsLast 的张量必须具有 4 个维度。我们允许用户将具有 3 个逻辑维度和 1 个批处理维度的张量克隆为 ChannelsLast 张量吗？
  // - 对于具有 3 个逻辑维度和 N>1 个批处理维度的张量又如何？
  TORCH_CHECK(!memory_format.has_value() || memory_format == MemoryFormat::Preserve
      || memory_format == MemoryFormat::Contiguous,
      "NYI: Tensor.clone(memory_format) inside vmap is only supported with ",
      "memory_format torch.preserve_format or torch.contiguous_format (got ",
      *memory_format, ")");

  if (memory_format == MemoryFormat::Contiguous) {
    // 当批处理维度不在张量的前面时存在歧义。
    // >>> x = torch.randn(3, B0, 5)
    // >>> y = vmap(lambda x: x.clone(torch.contiguous_format), in_dims=1, out_dims=0)(x)
    // >>> y[0].is_contiguous()
    // ???
    // 我们应该使整个张量连续，还是应该使非批处理维度连续？我们选择了后者，因为在哲学上 vmap 隐藏了批处理维度，并在每个样本级别上操作。
    auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
    auto output_physical = at::clone(physical_view.tensor(), memory_format);
    return physical_view.getPhysicalToLogicalMap().apply(output_physical);
  }

  TORCH_INTERNAL_ASSERT(!memory_format.has_value() || memory_format == MemoryFormat::Preserve);
  auto* self_batched = unsafeGetBatchedImpl(self);
  auto output_physical = at::clone(self_batched->value(), memory_format);
  auto old_bdims = self_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

// 注意 [matmul-like 操作的批处理规则]
// at::matmul 不会对参数进行“去展开”以获得更好的性能（也许它应该这样做）。
// 在类似 matmul 的操作（dot、mv、mm）的批处理规则中，我们应谨慎不展开任何不必要的维度。
// 例如，如果两个参数中只有一个是 BatchedTensor，则我们应尽量不将批处理维度展开到另一个参数上。
Tensor mv_batching_rule(const Tensor& self, const Tensor& other) {
  auto self_batched = isBatchedTensor(self);
  auto other_batched = isBatchedTensor(other);

  // 应该有一个形状检查的 API…
  TORCH_CHECK(self.dim() == 2 && other.dim() == 1,
      "mv(self, other): Shape mismatch: expected matrix "
      "(got `self` of size ", self.sizes(), ") ",
      "and vector (got `other` of size ", other.sizes(), ")");

  // 详见注意 [matmul-like 操作的批处理规则]，解释为何有不同的情况
  if (self_batched && !other_batched) {
    // 将自身张量转换为物理表示
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    // 执行矩阵乘法操作，self_physical.tensor() 和 other 进行乘法运算
    auto result = at::matmul(self_physical.tensor(), other);
    // 将物理表示映射回逻辑表示，并应用到结果上
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }
  // 如果只有 other 是批量张量而 self 不是
  if (!self_batched && other_batched) {
    // self_physical: [L, K], other_physical: [..., K]
    // 将张量视为 [L, K] 和 [..., K, 1]，执行矩阵乘法以得到 [..., L, 1] 大小的张量，并在最后一个维度上进行unsqueeze操作
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor().unsqueeze(-1));
    // 将物理表示映射回逻辑表示，并从结果中去除最后一个维度的unsqueeze操作
    return other_physical.getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  // 如果 self 和 other 都是批量张量
  if (self_batched && other_batched) {
    // self_physical: [..., L, K], other_physical: [..., K]
    // 将张量视为 [..., L, K] 和 [..., K, 1]，执行矩阵乘法以得到 [..., L, 1] 大小的张量，并在最后一个维度上进行unsqueeze操作
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(
        physical_args[0].tensor(),
        physical_args[1].tensor().unsqueeze(-1));
    // 将第一个批量张量的物理表示映射回逻辑表示，并从结果中去除最后一个维度的unsqueeze操作
    return physical_args[0].getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  // 如果执行到此处，表明 self 和 other 都不是批量张量，应该永远不会达到这一步
  TORCH_INTERNAL_ASSERT(false, "either self or other must be a BatchedTensor");
// 定义一个函数 `_make_dual_batching_rule`，用于创建双向批处理规则的张量
Tensor _make_dual_batching_rule(
  c10::DispatchKeySet ks,                   // 输入的调度键集合
  const Tensor& primal,                     // 主张量
  const Tensor& tangent,                    // 切向量
  int64_t level                             // 批处理级别
) {
  // 创建一个包含 `Batched` 调度键的完整后调度键集合
  DispatchKeySet after_batched_keyset =
      DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::Batched);
  // 调用 `_make_dual` 函数重新调度，使用 `ks` 和 `after_batched_keyset` 的交集
  return at::redispatch::_make_dual(ks & after_batched_keyset, primal, tangent, level);
}

// 定义函数 `dot_batching_rule`，处理 `dot` 类似操作的批处理规则
Tensor dot_batching_rule(const Tensor& self, const Tensor& other) {
  auto self_batched = isBatchedTensor(self);  // 检查 `self` 是否为批处理张量
  auto other_batched = isBatchedTensor(other);  // 检查 `other` 是否为批处理张量

  // 检查张量维度是否为1，如果不是则抛出错误信息
  TORCH_CHECK(/*logical*/self.dim() == 1 && /*logical*/other.dim() == 1,
      "dot(self, other): Shape mismatch: vector "
      "(got `self` of size ", self.sizes(), ") ",
      "and vector (got `other` of size ", other.sizes(), ")");

  // 根据 "Batching rules for matmul-like operators" 笔记，处理不同情况的批处理规则
  if (self_batched && !other_batched) {
    // 如果 `self` 是批处理张量而 `other` 不是
    // 将 `self` 和 `other` 视为 `[..., 1, K]` 和 `[K]` 进行矩阵乘法，并展开结果
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    auto result = at::matmul(self_physical.tensor().unsqueeze(-2), other);
    return self_physical.getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  if (!self_batched && other_batched) {
    // 如果 `self` 不是批处理张量而 `other` 是
    // 将 `self` 和 `other` 视为 `[K]` 和 `[..., K, 1]` 进行矩阵乘法，并展开结果
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor().unsqueeze(-1));
    return other_physical.getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  if (self_batched && other_batched) {
    // 如果 `self` 和 `other` 都是批处理张量
    // 将 `self` 和 `other` 视为 `[..., 1, K]` 和 `[..., K, 1]` 进行矩阵乘法，并展开结果
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(
        physical_args[0].tensor().unsqueeze(-2),
        physical_args[1].tensor().unsqueeze(-1));
    return physical_args[0].getPhysicalToLogicalMap().apply(result.squeeze(-1).squeeze(-1));
  }
  // 如果程序执行到这里，表明 `self` 或 `other` 必须是一个批处理张量，否则抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false, "either self or other must be a BatchedTensor");
}

// 定义函数 `bmm_batching_rule`，处理 `bmm` 操作的批处理规则
Tensor bmm_batching_rule(const Tensor& self, const Tensor& other) {
  // 检查张量维度是否为3，如果不是则抛出错误信息
  TORCH_CHECK(/*logical*/self.dim() == 3 && /*logical*/other.dim() == 3,
      "bmm(self, other): Shape mismatch: expected 3D `self` "
      "(got `self` of size ", self.sizes(), ") ",
      "and 3D `other` (got `other` of size ", other.sizes(), ")");

  // 将 `self` 和 `other` 视为物理批处理张量，执行矩阵乘法并返回逻辑映射后的结果
  auto physical_args = BroadcastingVmapTransform::logicalToPhysical({self, other});
  auto result = at::matmul(physical_args[0].tensor(), physical_args[1].tensor());
  return physical_args[0].getPhysicalToLogicalMap().apply(result);
}
// 定义了一个函数 mm_batching_rule，用于处理两个张量的矩阵乘法的批处理规则
Tensor mm_batching_rule(const Tensor& self, const Tensor& other) {
  // 检查 self 和 other 是否是批处理张量
  auto self_batched = isBatchedTensor(self);
  auto other_batched = isBatchedTensor(other);

  // 检查 self 和 other 的维度是否为二，如果不是则抛出错误信息
  TORCH_CHECK(/*logical*/self.dim() == 2 && /*logical*/other.dim() == 2,
      "mm(self, other): Shape mismatch: expected matrix "
      "(got `self` of size ", self.sizes(), ") ",
      "and matrix (got `other` of size ", other.sizes(), ")");

  // 根据矩阵乘法批处理规则进行处理，详见注释 [Batching rules for matmul-like operators]
  if (self_batched && !other_batched) {
    // 将逻辑上的 self 转换为物理上的 self，进行矩阵乘法运算
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    auto result = at::matmul(self_physical.tensor(), other);
    // 将物理上的结果映射回逻辑上，并返回结果
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }
  if (!self_batched && other_batched) {
    // 将逻辑上的 other 转换为物理上的 other，进行矩阵乘法运算
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor());
    // 将物理上的结果映射回逻辑上，并返回结果
    return other_physical.getPhysicalToLogicalMap().apply(result);
  }
  if (self_batched && other_batched) {
    // 将逻辑上的 self 和 other 转换为物理上的张量列表，进行矩阵乘法运算
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(physical_args[0].tensor(), physical_args[1].tensor());
    // 对结果进行维度调整，并将物理上的结果映射回逻辑上，并返回结果
    return physical_args[0].getPhysicalToLogicalMap().apply(result.squeeze(-1).squeeze(-1));
  }
  // 如果以上情况都不满足，则抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false, "either self or other must be a BatchedTensor");
}

// 定义了一个函数 cat_batching_rule，用于处理张量列表的拼接操作的批处理规则
Tensor cat_batching_rule(const ITensorListRef& tensors, int64_t dim) {
  // 将逻辑上的张量列表转换为物理上的视图列表
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  // 从物理视图列表中提取张量列表
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  // 检查张量列表是否为空，如果为空则抛出错误信息
  TORCH_INTERNAL_ASSERT(
      !tensors.empty(), "The dispatcher should not have dispatched here otherwise.");
  // 对物理张量列表进行拼接操作，并将结果映射回逻辑上，并返回结果
  auto result = at::cat(physical_tensors, physical_views[0].getPhysicalDim(dim));
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

// 定义了一个函数 stack_batching_rule，用于处理张量列表的堆叠操作的批处理规则
Tensor stack_batching_rule(TensorList tensors, int64_t dim) {
  // 将逻辑上的张量列表转换为物理上的视图列表
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  // 从物理视图列表中提取张量列表
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  // 检查张量列表是否为空，如果为空则抛出错误信息
  TORCH_INTERNAL_ASSERT(
      !tensors.empty(), "The dispatcher should not have dispatched here otherwise.");
  // 对物理张量列表进行堆叠操作，并将结果映射回逻辑上，并返回结果
  // 注意：由于堆叠操作会在逻辑维度上加一，因此需要手动调整维度信息
  auto dim_physical =
      physical_views[0].numBatchDims() + maybe_wrap_dim(dim, /*logical*/tensors[0].dim() + 1);
  auto result = at::stack(physical_tensors, dim_physical);
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

// 注释：以下是一条注释，指出了为什么需要使用扩展的 TensorOptions 注册操作符的原因
// 虽然 native:: 实现可以使用 TensorOptions&，但我们仍需要注册具有扩展 TensorOptions 的操作符
// 这也使得元编程变得困难：例如，我们无法使用 unwrap_and_call<..., at::to>，因为 at::to 需要 TensorOptions& (!!)
Tensor to_dtype_layout_batching_rule(
    const Tensor& self,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory,
    bool non_blocking, bool copy,
    optional<MemoryFormat> memory_format) {
      // 创建一个 TensorOptions 对象并设置其属性：数据类型(dtype)、布局(layout)、设备(device)、是否使用固定内存(pin_memory)
      auto options = TensorOptions()
        .dtype(dtype)
        .layout(layout)
        .device(device)
        .pinned_memory(pin_memory);
      
      // 获取输入张量的批处理实现指针
      auto* input_batched = unsafeGetBatchedImpl(self);
      
      // 将批处理实现的值转换为指定选项(options)的物理张量
      auto output_physical = input_batched->value().to(options, non_blocking, copy, memory_format);
      
      // 获取输入批处理的旧批处理维度
      auto old_bdims = input_batched->bdims();
      
      // 利用新的物理张量和旧批处理维度创建并返回一个新的批处理对象
      return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
    }
}

Tensor new_zeros_batching_rule(
    const Tensor& self,                              // 输入参数self：当前张量的引用
    IntArrayRef size,                                 // 输入参数size：目标张量的尺寸数组
    optional<ScalarType> dtype,                       // 可选参数dtype：张量的数据类型
    optional<Layout> layout,                          // 可选参数layout：张量的布局
    optional<Device> device,                          // 可选参数device：张量的设备
    optional<bool> pin_memory) {                      // 可选参数pin_memory：指示是否使用固定内存

  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);  // 将逻辑视图转换为物理视图
  auto physical_size = physical_view.getPhysicalShape(size);              // 获取物理视图的尺寸信息
  auto options = TensorOptions()                      // 创建TensorOptions对象，配置张量选项
    .dtype(dtype)                                     // 设置数据类型
    .layout(layout)                                   // 设置布局
    .device(device)                                   // 设置设备
    .pinned_memory(pin_memory);                       // 设置固定内存选项
  auto result = physical_view.tensor().new_zeros(     // 使用物理视图上的张量对象，创建全零张量
    physical_size, options);
  return physical_view.getPhysicalToLogicalMap().apply(result);  // 将结果张量映射回逻辑视图并返回
}

Tensor new_empty_batching_rule(
    const Tensor& self,                              // 输入参数self：当前张量的引用
    IntArrayRef size,                                 // 输入参数size：目标张量的尺寸数组
    std::optional<ScalarType> dtype,                  // 可选参数dtype：张量的数据类型
    std::optional<Layout> layout,                     // 可选参数layout：张量的布局
    std::optional<Device> device,                     // 可选参数device：张量的设备
    std::optional<bool> pin_memory) {                 // 可选参数pin_memory：指示是否使用固定内存

  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);  // 将逻辑视图转换为物理视图
  auto physical_size = physical_view.getPhysicalShape(size);              // 获取物理视图的尺寸信息
  auto result = physical_view.tensor().new_empty(     // 使用物理视图上的张量对象，创建空张量
    physical_size, TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory));
  return physical_view.getPhysicalToLogicalMap().apply(result);  // 将结果张量映射回逻辑视图并返回
}

Tensor new_empty_strided_batching_rule(
    const Tensor& self,                              // 输入参数self：当前张量的引用
    IntArrayRef size,                                 // 输入参数size：目标张量的尺寸数组
    IntArrayRef stride,                               // 输入参数stride：张量的步长数组
    optional<ScalarType> dtype,                       // 可选参数dtype：张量的数据类型
    optional<Layout> layout,                          // 可选参数layout：张量的布局
    optional<Device> device,                          // 可选参数device：张量的设备
    optional<bool> pin_memory) {
  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
  auto physical_size = physical_view.getPhysicalShape(size);

  // 计算批处理维度的形状，将其转换为IntArrayRef类型
  auto batch_shape = IntArrayRef(
      physical_view.tensor().sizes().begin(), physical_view.numBatchDims());

  // 计算批处理维度的默认步长
  auto physical_strides = at::detail::defaultStrides(batch_shape);

  // 检查大小和步长的维度是否匹配
  TORCH_CHECK(size.size() == stride.size(),
        "new_empty_strided(sizes, strides): dimensionality of sizes (",
        size.size(), ") must match dimensionality of strides (",
        stride.size(), ")");

  // 计算存储大小以用于物理步长的调整
  auto storage_size = native::storage_size_for(size, stride);

  // 调整物理步长，乘以存储大小
  for (auto& physical_stride : physical_strides) {
    physical_stride *= storage_size;
  }

  // 将给定的步长添加到物理步长末尾，形成最终的物理步长
  physical_strides.insert(physical_strides.end(), stride.begin(), stride.end());

  // 使用物理形状和物理步长创建一个新的空张量
  auto result = physical_view.tensor().new_empty_strided(
      physical_size, physical_strides, dtype, layout, device, pin_memory);

  // 将物理视图映射应用到结果张量，返回逻辑视图
  return physical_view.getPhysicalToLogicalMap().apply(result);
}
}

// 定义一个模板函数，用于处理批处理规则，比较两个张量，并返回处理结果张量
template <typename F, F Func>
Tensor comparison_pointwise_batching_rule(const Tensor& self, const Tensor& other) {
  // 将逻辑张量转换为物理张量，以进行广播
  auto physical_args = BroadcastingVmapTransform::logicalToPhysical({self, other});
  // 调用模板函数 Func 处理物理张量，并获取结果张量
  auto result = Func(physical_args[0].tensor(), physical_args[1].tensor());
  // 将处理后的结果张量映射回逻辑张量的形式，并返回
  return physical_args[0].getPhysicalToLogicalMap().apply(result);
}
}

// 实现 Torch 库中的 Batched 模块
TORCH_LIBRARY_IMPL(_, Batched, m) {
  // 设置批处理后退机制，使用指向 batchedTensorForLoopFallback 函数的 CppFunction
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedTensorForLoopFallback>());
}

// 定义宏 UNARY_POINTWISE，简化注册一元点逐元素操作的过程
#define UNARY_POINTWISE(op) m.impl(#op, \
    unwrap_and_call<Tensor (*)(const Tensor&), at::op>);
  UNARY_POINTWISE(abs);     // 注册绝对值操作
  UNARY_POINTWISE(acos);    // 注册反余弦操作
  UNARY_POINTWISE(asin);    // 注册反正弦操作
  UNARY_POINTWISE(atan);    // 注册反正切操作
  UNARY_POINTWISE(ceil);    // 注册向上取整操作
  UNARY_POINTWISE(cos);     // 注册余弦操作
  UNARY_POINTWISE(cosh);    // 注册双曲余弦操作
  UNARY_POINTWISE(conj_physical); // 注册共轭物理操作
  UNARY_POINTWISE(digamma); // 注册 digamma 函数操作
  UNARY_POINTWISE(exp);     // 注册指数函数操作
  UNARY_POINTWISE(expm1);   // 注册 expm1 函数操作
  UNARY_POINTWISE(floor);   // 注册向下取整操作
  UNARY_POINTWISE(frac);    // 注册 frac 函数操作
  UNARY_POINTWISE(lgamma);  // 注册 lgamma 函数操作
  UNARY_POINTWISE(log);     // 注册自然对数操作
  UNARY_POINTWISE(log10);   // 注册以 10 为底对数操作
  UNARY_POINTWISE(log1p);   // 注册 log1p 函数操作
  UNARY_POINTWISE(log2);    // 注册以 2 为底对数操作
  UNARY_POINTWISE(neg);     // 注册取负操作
  UNARY_POINTWISE(reciprocal); // 注册倒数操作
  UNARY_POINTWISE(relu);    // 注册 ReLU 操作
  UNARY_POINTWISE(round);   // 注册四舍五入操作
  UNARY_POINTWISE(rsqrt);   // 注册 rsqrt 函数操作
  UNARY_POINTWISE(sigmoid); // 注册 sigmoid 函数操作
  UNARY_POINTWISE(sign);    // 注册符号函数操作
  UNARY_POINTWISE(sin);     // 注册正弦函数操作
  UNARY_POINTWISE(sinh);    // 注册双曲正弦函数操作
  UNARY_POINTWISE(sqrt);    // 注册平方根函数操作
  UNARY_POINTWISE(tan);     // 注册正切函数操作
  UNARY_POINTWISE(tanh);    // 注册双曲正切函数操作
  UNARY_POINTWISE(trunc);   // 注册截断函数操作
#undef UNARY_POINTWISE

// 定义宏 TO_BATCHING_RULE，简化注册张量转换规则的过程
#define TO_BATCHING_RULE(name, ...) \
  { \
    using to_type = Tensor(Tensor::*)(__VA_ARGS__) const; \
    m.impl(name, unwrap_and_call_method< \
        to_type, &Tensor::to, __VA_ARGS__>);\
  }
  TO_BATCHING_RULE("to.device", Device, ScalarType, bool, bool, optional<MemoryFormat>) // 注册设备转换规则
  TO_BATCHING_RULE("to.dtype", ScalarType, bool, bool, optional<MemoryFormat>)         // 注册数据类型转换规则
  TO_BATCHING_RULE("to.other", const Tensor&, bool, bool, optional<MemoryFormat>)      // 注册其他转换规则
  m.impl("to.dtype_layout", to_dtype_layout_batching_rule);                             // 注册数据类型和布局转换规则
#undef TO_BATCHING_RULE

// 注册 clone 操作的批处理规则
m.impl("clone", clone_batching_rule);

// 定义三种不同类型的二元点逐元素操作的函数指针类型
using TensorTensorScalarType = Tensor (*)(const Tensor&, const Tensor&, const Scalar&);
using TensorTensorType = Tensor (*)(const Tensor&, const Tensor&);
using TensorScalarType = Tensor (*)(const Tensor&, const Scalar&);

// 定义宏 BINARY_POINTWISE，简化注册二元点逐元素操作的过程
#define BINARY_POINTWISE(op) \
  m.impl(#op".Tensor", binary_pointwise_batching_rule<TensorTensorType, at::op>); \
  m.impl(#op".Scalar", unwrap_and_call<TensorScalarType, at::op, const Scalar&>);
#define BINARY_POINTWISE_VA(op, ...) \
  { \
    using Binop = Tensor (*)(const Tensor&, const Tensor&, __VA_ARGS__); \
    using Unop = Tensor (*)(const Tensor&, const Scalar&, __VA_ARGS__); \
    m.impl(#op".Tensor", binary_pointwise_batching_rule<Binop, at::op, __VA_ARGS__>); \
    m.impl(#op".Scalar", unwrap_and_call<Unop, at::op, const Scalar&, __VA_ARGS__>); \
  }

  BINARY_POINTWISE_VA(add, const Scalar&);   // 注册加法操作
  BINARY_POINTWISE_VA(sub, const Scalar&);   // 注册减法操作
  BINARY_POINTWISE_VA(rsub, const Scalar&);  // 注册右减法操作
  BINARY_POINTWISE(mul);                    // 注册乘法操作
  BINARY_POINTWISE(div);                    // 注册除法操作
  // 定义函数指针类型 Binop，表示接受两个张量和一个可选字符串视图参数的函数指针类型
  using Binop = Tensor (*)(const Tensor&, const Tensor&, std::optional<c10::string_view>);
  // 定义函数指针类型 Unop，表示接受一个张量、一个标量和一个可选字符串视图参数的函数指针类型
  using Unop = Tensor (*)(const Tensor&, const Scalar&, std::optional<c10::string_view>);
  // 将 binary_pointwise_batching_rule 函数模板应用于 at::div 操作，实现 "div.Tensor_mode" 方法
  m.impl("div.Tensor_mode", binary_pointwise_batching_rule<Binop, at::div, std::optional<c10::string_view>>);
  // 将 unwrap_and_call 函数模板应用于 at::div 操作，实现 "div.Scalar_mode" 方法
  m.impl("div.Scalar_mode", unwrap_and_call<Unop, at::div, const Scalar&, std::optional<c10::string_view>>);
}

// at::pow 有三个非原地的重载版本
m.impl("pow.Tensor_Tensor", binary_pointwise_batching_rule<TensorTensorType, at::pow>);
m.impl("pow.Tensor_Scalar", unwrap_and_call<TensorScalarType, at::pow, const Scalar&>);
m.impl("pow.Scalar", pow_scalar_Tensor_batching_rule);

// 实现 "sigmoid_backward" 方法，使用 binary_pointwise_batching_rule 应用于 at::sigmoid_backward 操作
m.impl("sigmoid_backward", binary_pointwise_batching_rule<TensorTensorType, at::sigmoid_backward>);
// 实现 "threshold_backward" 方法，使用 binary_pointwise_batching_rule 应用于 at::threshold_backward 操作
m.impl(
    "threshold_backward",
    binary_pointwise_batching_rule<
        TensorTensorScalarType,
        at::threshold_backward,
        const Scalar&>);

// 对于 at::result_type，调用 native::result_type 的实现
// 由于 native::result_type 操作基于张量的逻辑形状，因此不需要特殊处理
m.impl("result_type.Tensor", static_cast<ScalarType (*)(const Tensor&, const Tensor&)>(native::result_type));
m.impl("result_type.Scalar", static_cast<ScalarType (*)(const Tensor&, const Scalar&)>(native::result_type));
m.impl("result_type.Scalar_Tensor", static_cast<ScalarType (*)(const Scalar&, const Tensor&)>(native::result_type));
m.impl("result_type.Scalar_Scalar", static_cast<ScalarType (*)(const Scalar&, const Scalar&)>(native::result_type));
#undef BINARY_POINTWISE_VA
#undef BINARY_POINTWISE

// 定义一个宏，用于简化操作符函数的注册
#define TRIVIAL_OP(op) m.impl(#op, \
    unwrap_and_call<Tensor (*)(const Tensor&), at::op>);

  // 注册复数视图操作符
  TRIVIAL_OP(imag)
  TRIVIAL_OP(real);
  TRIVIAL_OP(view_as_real);
  TRIVIAL_OP(conj);
  TRIVIAL_OP(_conj);
  TRIVIAL_OP(resolve_conj);
  TRIVIAL_OP(resolve_neg);

  // 使用自定义函数注册视图为复数的操作符
  m.impl("view_as_complex", view_as_complex_batching_rule);
#undef TRIVIAL

  // 注册矩阵乘法类操作符
  m.impl("mv", mv_batching_rule);
  m.impl("dot", dot_batching_rule);
  m.impl("bmm", bmm_batching_rule);
  m.impl("mm", mm_batching_rule);

  // 注册cat和stack操作符
  m.impl("cat", cat_batching_rule);
  m.impl("stack", stack_batching_rule);

  // 注册反向操作符
  m.impl("select_backward", select_backward_batching_rule);
  m.impl("slice_backward", slice_backward_batching_rule);
  m.impl("trace_backward", trace_backward_batching_rule);
  m.impl("diagonal_backward", diagonal_backward_batching_rule);

  // 注册Tensor.new_*系列操作符
  m.impl("new_empty", new_empty_batching_rule);
  m.impl("new_empty_strided", new_empty_strided_batching_rule);
  m.impl("new_zeros", new_zeros_batching_rule);

  // 注册连续性操作符
  m.impl("contiguous", contiguous_batching_rule);

  // 定义宏，用于注册比较操作符的批处理规则
#define COMPARISON_POINTWISE(op) \
  m.impl(#op".Tensor", comparison_pointwise_batching_rule<TensorTensorType, at::op>); \
  m.impl(#op".Scalar", unwrap_and_call<TensorScalarType, at::op, const Scalar&>);

  // 注册比较操作符的批处理规则
  COMPARISON_POINTWISE(eq);
  COMPARISON_POINTWISE(gt);
  COMPARISON_POINTWISE(ge);
  COMPARISON_POINTWISE(le);
  COMPARISON_POINTWISE(lt);
  COMPARISON_POINTWISE(ne);

#undef COMPARISON_POINTWISE
}

} // namespace at
```