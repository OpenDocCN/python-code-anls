# `.\pytorch\aten\src\ATen\cuda\detail\KernelUtils.h`

```
// 防止头文件被多次包含，保证在编译时每个文件只包含一次该头文件
#pragma once

// 包含定义了 std::numeric_limits 和 TORCH_INTERNAL_ASSERT 的头文件
#include <limits>
#include <c10/util/Exception.h>

// 进入 at::cuda::detail 命名空间
namespace at::cuda::detail {

// CUDA: 网格步进循环宏定义
//
// int64_t _i_n_d_e_x 专门用于在循环增量中防止溢出。
// 如果 input.numel() < INT_MAX，则 _i_n_d_e_x < INT_MAX，除了在最后一次
// 循环迭代之后 _i_n_d_e_x += blockDim.x * gridDim.x 可能大于 INT_MAX。
// 但是在这种情况下，_i_n_d_e_x >= n，因此没有进一步的迭代，并且在 i=_i_n_d_e_x 中不使用溢出的值。
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

// CUDA: 简化版网格步进循环宏定义，使用 int 作为索引类型
#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)


// 每个块使用 1024 个线程，要求 CUDA sm_2x 或更高版本支持
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: 计算需要的块数
inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block=CUDA_NUM_THREADS) {
  // 断言 N 大于 0，CUDA 核函数启动的块数必须为正数，但是 N = {N}
  TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  // 定义 int 类型的最大值
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  // 对正数进行向上取整除法，不会导致整数溢出
  auto block_num = (N - 1) / max_threads_per_block + 1;
  // 断言块数不超过 int 类型的最大值，不能在 CUDA 设备上安排太多块
  TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

  // 返回块数的整数值
  return static_cast<int>(block_num);
}

}  // namespace at::cuda::detail
```