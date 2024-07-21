# `.\pytorch\aten\src\ATen\native\cuda\TensorModeKernel.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含所需的头文件
#include <ATen/native/cuda/TensorModeKernel.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>

// 根据 ROCM 是否启用确定最大块大小
constexpr int MAX_BLOCK_SIZE = AT_ROCM_ENABLED() ? 256 : 1024;

// 计算能够处理的最大网格大小（假设计算能力 >= 2.0）
constexpr int64_t MAX_GRID_SIZE = 65535LL;

// 定义 at::native 命名空间
namespace at::native {

// 实现 mode_kernel_impl 函数
void mode_kernel_impl(
    Tensor& values,           // 输出值 Tensor
    Tensor& indices,          // 输出索引 Tensor
    const Tensor& self,       // 输入 Tensor
    int64_t dim,              // 维度
    bool keepdim) {           // 是否保持维度

  auto self_sizes = ensure_nonempty_vec(self.sizes().vec()); // 确保输入 Tensor 尺寸非空
  int64_t ndim = ensure_nonempty_dim(self.dim());            // 确保输入 Tensor 维度非空
  int64_t slice_size = ensure_nonempty_size(self, dim);      // 确保切片大小非空
  int64_t slices = self.numel() / slice_size;                // 计算切片数量

  // 调整输出值和索引 Tensor 的大小，使其与输入 Tensor 大小一致（在 dim 维度上尺寸设为 1）
  assert(0 <= dim && static_cast<size_t>(dim) < self_sizes.size()); // 断言维度有效性
  self_sizes[dim] = 1;

  // 如果不保持维度，对输出值和索引 Tensor 进行展开操作
  if (!keepdim) {
    if (values.ndimension() >= dim) {
      values.unsqueeze_(dim); // 在 dim 维度上展开 values Tensor
    }
    if (indices.ndimension() >= dim) {
      indices.unsqueeze_(dim); // 在 dim 维度上展开 indices Tensor
    }
  }

  at::native::resize_output(values, self_sizes);  // 调整 values Tensor 大小
  at::native::resize_output(indices, self_sizes); // 调整 indices Tensor 大小

  // 如果 slice_size 为 1，将输入 Tensor 复制到 values，设置 indices
  if (slice_size == 1) {
    values.copy_(self); // 复制输入 Tensor 到 values
    indices.fill_(0);   // 将 indices 填充为 0
    if (!keepdim) {
      values.squeeze_(dim);  // 如果不保持维度，在 dim 维度上挤压 values Tensor
      indices.squeeze_(dim); // 如果不保持维度，在 dim 维度上挤压 indices Tensor
    }
    return;
  }

  // 开始优化的实现。首先，沿排序维度转置输入 Tensor，并使其连续。
  auto transposed = self.transpose(dim, ndim - 1); // 在 dim 和最后一维之间进行转置
  auto contiguous = transposed.contiguous();       // 将转置后的 Tensor 变为连续的

  // 还需要将 values 和 indices Tensor 视为转置的，以正确确定放置模式和索引的偏移量
  auto values_transposed = values.transpose(dim, ndim - 1); // 在 dim 和最后一维之间进行转置
  auto indices_transposed = indices.transpose(dim, ndim - 1); // 在 dim 和最后一维之间进行转置

  // 融合核心实现的需求：
  //
  // 1. slice_size <= 2 * 最大线程块大小
  // 2. 每个切片使用一个线程块，因此切片数量必须小于最大块数的限制
  // 3. 可以使用 32 位索引数学进行索引（主要是为了实现简洁性，可以更改）
  //
  // MAX_BLOCK_SIZE 和 MAX_GRID_SIZE 来自于 ATen/native/cuda/SortingCommon.cuh
  if (slice_size <= 2 * MAX_BLOCK_SIZE &&
      slices <= MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE &&
      canUse32BitIndexMath(self)) {
    launch_fused_mode_kernel(
        values_transposed, indices_transposed, contiguous, slice_size, slices); // 启动融合模式核心实现
  } else {
    // [注意：CUDA torch.mode 克隆 self]
    //
    // 如果 transposed 张量已经是连续的，它将返回一个与其存储相同的张量。
    // 因此，由于我们不希望修改 self 张量，我们对其进行克隆操作。
    if (transposed.is_same(contiguous)) {
      contiguous = contiguous.clone();
    }

    // 调用一个特定的内核函数来处理应用模式
    launch_apply_mode_kernel(
        values_transposed, indices_transposed, contiguous, dim, ndim);
  }

  // 如果 keepdim 为 false，则需要在指定维度上对 values 和 indices 进行挤压操作
  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

// 将 mode_stub 注册为 CUDA 分发函数，并实现 mode_kernel_impl
REGISTER_CUDA_DISPATCH(mode_stub, &mode_kernel_impl);
// 结束 at::native 命名空间
} // namespace at::native
```