# `.\pytorch\aten\src\ATen\native\quantized\cpu\Sorting.cpp`

```
// 定义宏，仅在包含头文件时启用 assert 操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的必要头文件
#include <ATen/core/Tensor.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

// 根据编译器设置条件，包含不同的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/topk_native.h>
#endif

// 命名空间：at 下的 native 命名空间
namespace at {
namespace native {

// 目前仅用于内部。
//
// 此实现假定输入和输出的量化器相同。
//
// 如果要公开支持此功能，需要在内核中添加重新量化步骤。
static std::tuple<Tensor&, Tensor&> quantized_topk_out_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  // 确定维度，可选是否进行包装
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  // 检查所选的索引 k 是否在合理范围内
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  // 使用给定的 self 张量分配或调整输出张量 values 和 indices
  _allocate_or_resize_output_with_indices(values, indices, self, dim_, k);

  // 调用量化 topk 操作的 CPU 实现
  qtopk_stub(kCPU, values, indices, self, k, dim, largest, sorted);

  // 返回更新后的 values 和 indices 张量的引用
  return std::forward_as_tuple(values, indices);
}

// 在 CPU 上执行量化的 topk 操作，返回排序后的值和对应的索引
std::tuple<Tensor, Tensor> topk_quantized_cpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  // 获取张量的量化方案
  auto qscheme = self.qscheme();
  // 检查是否支持在每张量量化模式下进行 Top-K 操作
  TORCH_CHECK(
      qscheme == QScheme::PER_TENSOR_AFFINE ||
          qscheme == QScheme::PER_TENSOR_SYMMETRIC,
      "Top-K is only supported on per-tensor quantization");

  // 创建空的量化张量 values 和 indices
  Tensor values = at::_empty_affine_quantized(
    {0},
    self.options(),
    self.q_scale(),
    self.q_zero_point());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));

  // 调用 quantized_topk_out_cpu 函数执行 topk 操作，并返回结果
  return quantized_topk_out_cpu(values, indices, self, k, dim, largest, sorted);
}

// 定义 qtopk_stub 的分派实现
DEFINE_DISPATCH(qtopk_stub);

}}  // namespace at::native
```