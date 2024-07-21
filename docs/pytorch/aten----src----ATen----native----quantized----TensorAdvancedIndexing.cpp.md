# `.\pytorch\aten\src\ATen\native\quantized\TensorAdvancedIndexing.cpp`

```
// 包含 ATen 库中的各种头文件
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/core/QScheme.h>
#include <ATen/native/TensorAdvancedIndexing.h>

// ATen 命名空间
namespace at {
namespace native {

// 定义量化版本的分发调度器
DEFINE_DISPATCH(masked_fill_kernel_quantized_stub);
DEFINE_DISPATCH(index_put_kernel_quantized_stub);
DEFINE_DISPATCH(index_put_with_sort_quantized_stub);

// 匿名命名空间，内部函数实现
namespace {
// 创建用于索引填充的 TensorIterator 对象
static TensorIterator make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  // 检查 value 张量的形状是否可以广播到与 info.src 张量相匹配的形状
  TORCH_CHECK(is_expandable_to(value.sizes(), info.src.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", info.src.sizes());
  
  // 配置 TensorIteratorConfig 对象
  TensorIteratorConfig config;
  config.set_check_mem_overlap(false);  // 关闭内存重叠检查
  config.resize_outputs(false);         // 不调整输出大小
  config.check_all_same_dtype(false);   // 不检查所有张量是否具有相同的数据类型
  config.add_output(info.src);          // 添加输出张量 info.src
  config.add_input(value);              // 添加输入张量 value
  for (auto& index : info.indices) {
    config.add_input(index);            // 添加索引张量
  }
  // 构建并返回 TensorIterator 对象
  return config.build();
}

// 量化版本的 CPU 实现，用于在张量中使用掩码填充
static Tensor & masked_fill_impl_quantized_cpu(Tensor & self, const Tensor & mask, const Scalar& value) {
  NoNamesGuard guard;  // 临时禁用命名
  // 检查掩码张量的数据类型是否为布尔类型
  TORCH_CHECK(mask.dtype() == ScalarType::Bool, "masked_fill only supports boolean masks, "
    "but got dtype ", mask.dtype());

  // 如果 self 张量存在内部重叠，发出警告
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  
  // 检查 self 和 mask 张量是否有部分重叠
  at::assert_no_partial_overlap(self, mask);

  // 配置 TensorIteratorConfig 对象
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 关闭内存重叠检查（已弃用，但不是硬错误）
    .check_all_same_dtype(false)   // 不检查所有张量是否具有相同的数据类型
    .resize_outputs(false)         // 不调整输出大小
    .add_output(self)              // 添加输出张量 self
    .add_input(mask)               // 添加输入张量 mask
    .build();

  // 调用量化版本的掩码填充内核函数
  masked_fill_kernel_quantized_stub(iter.device_type(), iter, value, self.q_scale(), self.q_zero_point());
  return self;  // 返回填充后的 self 张量
}

}

// 量化版本的掩码填充函数，用于在量化张量中使用掩码填充
Tensor & masked_fill__quantized_cpu(Tensor& self, const Tensor & mask, const Scalar& value) {
  // 检查 self 张量是否为每张量仿射量化方案
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "masked_fill__quantized_cpu for quantized tensors is currently only supported for per tensor quantized tensors");
  
  // 通过广播将 self 和 mask 张量的名称传播到输出张量
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  // 调用量化版本的掩码填充实现函数
  masked_fill_impl_quantized_cpu(self, mask, value);
  
  // 如果输出张量名称不为空，则传播名称
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;  // 返回填充后的 self 张量
}

}  // namespace native
}  // namespace at
// 在 CPU 上执行的量化张量版本的 masked_fill_ 函数，用于在张量中根据条件 mask 填充指定的标量值 value
Tensor & masked_fill__quantized_cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
  // 检查张量 self 是否为按张量分量量化方案，否则抛出错误
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "masked_fill__quantized_cpu for quantized tensors is currently only supported for per tensor quantized tensors");

  // 执行广播以确定输出张量的命名，并获取可能的输出名称
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  // 检查 value 张量的维度是否为 0，因为 masked_fill_ 仅支持0维值张量，否则抛出错误
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");

  // 在 CPU 上执行量化版本的 masked_fill_ 操作
  masked_fill_impl_quantized_cpu(self, mask, value.item());

  // 如果可能的输出名称非空，则传播命名
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);

  // 返回修改后的自身张量
  return self;
}

// 在 CUDA 上执行的量化张量版本的 masked_fill_ 实现函数，用于在张量中根据条件 mask 填充标量值 value
static Tensor & masked_fill_impl_quantized_cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  // 检查 self 和 mask 张量是否在相同设备上，否则抛出错误
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());

  // 检查 mask 张量是否为布尔类型，因为 masked_fill_ 仅支持布尔掩码，否则抛出错误
  TORCH_CHECK(mask.scalar_type() == kBool, "masked_fill only supports boolean masks, "
    "but got dtype ", mask.scalar_type());

  // 检查张量 self 是否为按张量分量量化方案，否则抛出错误
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "masked_fill__quantized_cpu for quantized tensors is currently only supported for per tensor quantized tensors");

  // 执行广播以确定输出张量的命名，并获取可能的输出名称
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  // 如果 self 与 mask 有内部重叠，发出警告，因为此操作在扩展张量上已过时
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  // 确保 self 和 mask 张量没有部分重叠
  at::assert_no_partial_overlap(self, mask);

  // 将 mask 张量扩展到与 self 相同的形状，并在原地扩展
  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  // 配置张量迭代器，设置检查内存重叠为 false，检查所有相同数据类型为 false，不调整输出大小，设置输出为 self 张量，输入为 self 和 b_mask 张量
  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_input(*b_mask)
      .build();

  // 调用量化版本的 masked_fill_ 内核函数，传递设备类型、迭代器、标量值、self 的量化缩放因子和零点
  masked_fill_kernel_quantized_stub(iter.device_type(), iter, value, self.q_scale(), self.q_zero_point());

  // 如果可能的输出名称非空，则传播命名
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);

  // 返回修改后的自身张量
  return self;
}

// 在 CUDA 上执行的量化张量版本的 masked_fill_ 函数，用于在张量中根据条件 mask 填充标量值 value
Tensor & masked_fill__quantized_cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  // 检查输入张量是否不在 CPU 上，否则抛出错误
  TORCH_CHECK(!self.device().is_cpu(), "masked_fill_: Expected inputs to be on same device")

  // 调用 CUDA 上的 masked_fill_impl_quantized_cuda 函数进行具体实现
  return masked_fill_impl_quantized_cuda(self, mask, value);
}

// 在 CUDA 上执行的量化张量版本的 masked_fill_ 函数，用于在张量中根据条件 mask 填充标量值 value 的张量形式
Tensor & masked_fill__quantized_cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  // 检查 value 张量的维度是否为 0，因为 masked_fill_ 仅支持0维值张量，否则抛出错误
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");

  // 检查输入张量是否不在 CPU 上，否则抛出错误
  TORCH_CHECK(!self.device().is_cpu(), "masked_fill_: Expected inputs to be on same device")

  // 调用 CUDA 上的 masked_fill_impl_quantized_cuda 函数进行具体实现
  return masked_fill_impl_quantized_cuda(self, mask, value.item());
}
// 检查索引数量是否超过张量维度，如果超过则抛出错误信息
TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
// 检查值是否为量化类型，如果是则抛出错误信息
TORCH_CHECK(!value.is_quantized(), "Value argument for quantized input_put should not be quantized");
// 检查张量的量化方案是否为每张量仿射，否则抛出错误信息
TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "index_put for quantized tensors is currently only supported for per tensor quantized tensors");
// 检查是否设置了累积标志，对于量化张量，累积操作目前不支持，如果设置了则抛出错误信息
TORCH_CHECK(!accumulate, "index_put for quantized tensors is currently only supported for accumulate=False");

// 检查是否存在内存重叠情况，如果存在则发出警告提示，建议在操作前先克隆张量
if (at::has_internal_overlap(self) == MemOverlap::Yes) {
  TORCH_WARN(
    "Use of index_put_ on expanded tensors is deprecated. "
    "Please clone() the tensor before performing this operation. "
    "This also applies to advanced indexing e.g. tensor[indices] = tensor");
}

// 判断是否可以调度到 masked_fill 操作
auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
if (std::get<0>(masked_fill_dispatch)) {
  // 如果可以调度到 masked_fill 操作，则调用该操作并返回结果
  return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
}

// 如果值的设备与张量的设备不同且值为标量，则将值转移到张量所在设备
auto value_ = value;
if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
  value_ = value.to(self.device());
}

// 检查张量与值是否存在内存重叠
at::assert_no_overlap(self, value);

// 遍历索引列表，检查每个索引是否与张量存在内存重叠
for (const std::optional<Tensor>& index: indices) {
  if (index.has_value()) {
    at::assert_no_overlap(self, *index);
  }
}

// 创建用于 index_put 操作的信息对象
auto info = make_info(self, indices);
// 创建 index_put 迭代器
auto iter = make_index_put_iterator(info, value_);
// 调用量化版本的 index_put 内核函数
index_put_kernel_quantized_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate, self.q_scale(), self.q_zero_point());

// 返回修改后的自身张量
return self;
    // 使用指定值替换当前张量中的部分值，根据掩码填充操作的分派结果中的第一个元素，并返回修改后的张量
    return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
  }

  // 创建一个新变量 value_ 并将其初始化为 value
  auto value_ = value;
  // 如果 value 的设备与当前张量的设备不同，且 value 的元素数为1且维度为0，则将 value 转移到当前张量所在的设备上
  if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
    value_ = value.to(self.device());
  }
  // 断言 value 的设备与当前张量的设备相同，否则抛出错误信息
  TORCH_CHECK(value.device() == self.device(), "expected device ", self.device(), " but got device ", value.device(), " for value tensor");

  // 断言当前张量与 value 之间没有重叠的内存区域
  at::assert_no_overlap(self, value);
  // 对于 indices 中的每个可能为空的张量 index
  // 断言当前张量与每个非空的 index 之间没有重叠的内存区域
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const std::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

  // 查看是否启用确定性操作，当当前张量在 CUDA 设备上并且全局上下文启用了确定性算法时
  if (self.device().type() == DeviceType::CUDA && globalContext().deterministicAlgorithms()) {
      // 使用排序量化的索引放置操作的 CUDA 特定实现
      index_put_with_sort_quantized_stub(self.device().type(), self, indices, value_, self.q_scale(), self.q_zero_point(), unsafe);
      // 返回修改后的张量
      return self;
  }

  // 创建描述当前张量和 indices 的信息对象 info
  auto info = make_info(self, indices);
  // 创建索引放置操作的迭代器 iter
  auto iter = make_index_put_iterator(info, value_);
  // 使用量化内核执行索引放置操作
  index_put_kernel_quantized_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate, self.q_scale(), self.q_zero_point());
  // 返回修改后的张量
  return self;
}



# 这是一个单独的右花括号，用于结束某个代码块或函数的定义或结构。



}



# 这是另一个单独的右花括号，用于结束另一个代码块或函数的定义或结构。



}



# 这是最后一个单独的右花括号，用于结束另一个代码块或函数的定义或结构。
```