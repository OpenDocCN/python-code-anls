# `.\pytorch\aten\src\ATen\native\cuda\TensorTopK.cpp`

```
// 定义预处理宏，用于指定只有方法操作符可见
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 CUDA 实现的 TensorTopK 头文件
#include <ATen/native/cuda/TensorTopK.h>

// 引入 ATen 核心 Tensor 类和相关实用函数头文件
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
// 引入 CUDA 实现的排序函数
#include <ATen/native/cuda/Sort.h>

// 条件编译：如果未定义 AT_PER_OPERATOR_HEADERS 则引入一般的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
// 否则引入特定的头文件，包括空张量操作、CUDA 排序调度、topk 本地实现等
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sort_cuda_dispatch.h>
#include <ATen/ops/topk_native.h>
#endif

// 命名空间 at::native 内声明以下函数和结构体
namespace at::native {

// TODO: 当 CUDA 版本小于 11.6 不再支持时移除此函数
// 实现 topk_out_with_sort 函数，使用排序的方式实现 topk 操作
void topk_out_with_sort(
  const Tensor& self,              // 输入张量
  int64_t k,                       // 取 topk 的 k 值
  int64_t dim,                     // 操作的维度
  bool largest,                    // 是否取最大的 k 个值
  const Tensor& values,            // 输出的 topk 值张量
  const Tensor& indices            // 输出的 topk 索引张量
) {
  // 调用 ATen CUDA 排序函数，获取排序后的值和索引
  auto [sorted_values, sorted_indices] = at::cuda::sort(self, /* stable= */false, dim, largest);
  // 将排序后的前 k 个值拷贝到输出张量 values 中
  values.copy_(sorted_values.narrow(dim, 0, k));
  // 将排序后的前 k 个索引拷贝到输出张量 indices 中
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

// TODO: 当 CUDA 版本小于 11.6 不再支持时移除此函数
// 实现 disable_sort_for_topk 函数
bool disable_sort_for_topk();

// 根据输入张量和维度判断是否应该使用排序实现 topk
bool should_use_sort(const Tensor& self, int64_t dim) {
  // 如果全局禁用了排序，则直接返回 false
  if (disable_sort_for_topk()) return false;
  // 根据实验得出的启发式规则，对于 0 维度的张量不使用排序
  if (self.dim() == 0) return false;
  // 布尔类型张量不支持 topk 操作
  if (self.dtype() == kBool) return false;
  // 计算在指定维度上的切片大小
  int64_t slice_size = self.size(dim);
  // 如果切片大小为 0，则不使用排序
  if (slice_size == 0) return false;
  // 计算在指定维度上的切片数量
  int64_t num_slices = self.numel() / slice_size;
  // 如果切片数量小于等于 10 且切片大小大于等于 100000，则使用排序
  return num_slices <= 10 && slice_size >= 100000;
}

// 实现 topk_out_cuda 函数
TORCH_IMPL_FUNC(topk_out_cuda)
  (const Tensor& self,            // 输入张量
   int64_t k,                     // 取 topk 的 k 值
   int64_t dim,                   // 操作的维度
   bool largest,                  // 是否取最大的 k 个值
   bool sorted,                   // 是否需要排序
   const Tensor& values,          // 输出的 topk 值张量
   const Tensor& indices) {       // 输出的 topk 索引张量
  // 创建 TensorArg 对象，用于检查所有张量是否在同一 GPU 上
  TensorArg topK_arg{values, "topK", 1}, indices_arg{indices, "indices", 2}, input_arg{self, "self", 3};
  checkAllSameGPU(__func__, {topK_arg, indices_arg, input_arg});

  // 根据 self 的维度包装 dim
  dim = at::maybe_wrap_dim(dim, self);

  // 如果应该使用排序实现 topk，则调用 topk_out_with_sort 函数
  if (should_use_sort(self, dim)) {
    topk_out_with_sort(self, k, dim, largest, values, indices);
    return;
  }

  // 如果 k 为 0，则结果是一个空张量，无需启动核函数
  if (k == 0) {
    return;
  }

  // 启动 gather_topk_kernel 核函数，实现 topk 操作
  launch_gather_topk_kernel(self, k, dim, largest, values, indices);

  // 如果用户需要排序，并且 values 张量元素数量大于 1，则进行排序
  if (sorted && values.numel() > 1) {
    // 如果应该使用小排序（避免内存分配并在切片上原地执行所有排序工作），则调用 sortKeyValueInplace 函数
    if (should_use_small_sort(values, dim)) {
      sortKeyValueInplace(values, indices, dim, largest);
    } else {
      // 根据备份排序返回的索引，利用 gather 函数来重建原始索引。
      // 这不是最高效的实现，特别是因为这里进行了内存分配。
      // 如果用户希望获得更高的性能，他们应该自己使用 torch.gather()
      // 结合报告的索引，提供预先分配的张量来接收结果。

      // 创建一个与 indices 张量相同形状和类型的空张量 sortedIndices
      Tensor sortedIndices = at::empty_like(indices);
      // 创建一个与 values 张量相同形状和类型的空张量 sortedValues
      Tensor sortedValues = at::empty_like(values);
      // 调用 CUDA 的排序函数，将排序结果存储在 sortedValues 和 sortedIndices 中
      at::cuda::sort_outf(values, /* stable= */ false, dim, largest, sortedValues, sortedIndices);
      // 使用 gather 函数根据 dim 维度和 sortedIndices 重建 indices 张量
      indices.copy_(indices.gather(dim, sortedIndices));
      // 将 sortedValues 的内容复制回 values 张量，完成排序后的值更新
      values.copy_(sortedValues);
    }
  }


这段代码是在处理某种条件下的排序逻辑，根据备份排序的结果重新构建原始的索引和对应的数值数据。
}

} // namespace at::native
```