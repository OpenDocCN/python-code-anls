# `.\pytorch\aten\src\ATen\native\cuda\IndexKernel.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexing.h>  // For at::native::index_out
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
#else
#include <ATen/ops/index_cuda_dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/masked_scatter_native.h>
#include <ATen/ops/masked_select_native.h>
#endif

// 定义了 at::native 命名空间
namespace at::native {

// 在 CUDA 设备上实现 masked_select_out_cuda_impl 函数，返回一个 Tensor 引用
static Tensor & masked_select_out_cuda_impl(Tensor & result, const Tensor & self, const Tensor & mask) {
  // 使用 NoNamesGuard 确保操作不涉及命名
  NoNamesGuard guard;

  // 检查 mask 张量的数据类型是否为布尔类型
  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor for mask");
  
  // 检查 self 和 result 张量的数据类型是否相同
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  // 根据 mask 张量的维度，选择是否需要扩展 mask_temp 张量
  auto mask_temp = (mask.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(mask);
  
  // 根据 self 张量的维度，选择是否需要扩展 self_temp 张量
  auto self_temp = (self.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(self);

  // 将 mask_temp 和 self_temp 张量进行扩展，得到 mask_self_expanded
  auto mask_self_expanded = expand_outplace(*mask_temp, *self_temp);

  // 在 CUDA 设备上执行索引操作，将结果存储到 result 中
  at::cuda::index_out(
      result, *std::get<1>(mask_self_expanded),
      c10::List<std::optional<at::Tensor>>({*std::move(std::get<0>(mask_self_expanded))}));

  // 返回 result 引用
  return result;
}

// 在 CUDA 设备上实现 masked_select_cuda 函数，接受两个张量 self 和 mask 作为输入
Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  // 计算广播后的输出名称
  namedinference::compute_broadcast_outnames(self, mask);
  
  // 创建一个空的张量 result，使用 self 的选项
  Tensor result = at::empty({0}, self.options());
  
  // 调用 masked_select_out_cuda_impl 函数，返回结果
  return masked_select_out_cuda_impl(result, self, mask);
}

// 在 CUDA 设备上实现 masked_select_out_cuda 函数，接受三个张量 self、mask 和 result 作为输入
Tensor & masked_select_out_cuda(const Tensor & self, const Tensor & mask, Tensor & result) {
  // 计算广播后的输出名称
  namedinference::compute_broadcast_outnames(self, mask);
  
  // 调用 masked_select_out_cuda_impl 函数，返回结果
  return masked_select_out_cuda_impl(result, self, mask);
}

// 在 CUDA 设备上实现 masked_scatter__cuda 函数，接受三个张量 self、mask 和 source 作为输入
Tensor & masked_scatter__cuda(Tensor& self, const Tensor& mask, const Tensor& source) {
  // 检查 self 张量内部没有重叠
  at::assert_no_internal_overlap(self);
  
  // 检查 self 和 source 张量的数据类型是否相同
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter_: expected self and source to have same dtypes but got ",
      self.scalar_type(),
      " and ",
      source.scalar_type());
  
  // 检查 mask 张量的数据类型是否为布尔类型
  TORCH_CHECK(mask.dtype() == ScalarType::Bool, "masked_scatter_ only supports boolean masks, "
     "but got mask with dtype ", mask.dtype());

  // 在原地扩展 mask 张量，返回 MaybeOwned 的引用
  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_scatter_");

  // 如果 self 张量元素个数为 0，则直接返回 self
  if (self.numel() == 0) {
    return self;
  }

  // 创建与 self 相同大小的 maskPrefixSum 张量，数据类型为 kLong
  auto maskPrefixSum = at::empty(self.sizes(), mask.options().dtype(kLong));
  
  // 调用 launch_masked_scatter_kernel 函数，在 CUDA 设备上执行 masked scatter 操作
  launch_masked_scatter_kernel(self, *b_mask, maskPrefixSum, source);

  // 返回 self 引用
  return self;
}

}  // namespace at::native


这些注释将每行代码解释了其作用，帮助读者理解了代码中各个函数和关键操作的用途和实现方式。
```