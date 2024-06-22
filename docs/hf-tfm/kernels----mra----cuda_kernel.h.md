# `.\transformers\kernels\mra\cuda_kernel.h`

```py
// 定义线程束大小为32
#define WARP_SIZE 32
// 定义全掩码为32位1
#define FULL_MASK 0xffffffff
// 定义最佳线程数为256
#define OPTIMAL_THREADS 256

// CUDA 核函数：计算每个 batch 中每个 block 中的最大值的索引
__global__ void index_max_cuda_kernel(
  float *index_vals,            // [batch_size, 32, num_block]，存储每个 batch 中每个 block 中的最大值的索引
  int   *indices,               // [batch_size, num_block]，存储每个 batch 中每个 block 的最大值的索引
  float *max_vals,              // [batch_size, A_num_block * 32]，存储每个 batch 中每个 block 的最大值
  float *max_vals_scatter,      // [batch_size, 32, num_block]，存储每个 batch 中每个 block 的最大值在 max_vals 中的索引
  long batch_size,              // batch 大小
  long A_num_block,             // A 的 block 数
  long B_num_block,             // B 的 block 数
  long num_block                // 总 block 数
);

// CUDA 核函数：将稠密矩阵转换为稀疏矩阵
__global__ void mm_to_sparse_cuda_kernel(
  float *dense_A,               // [batch_size, A_num_block, dim, 32]，稠密矩阵 A
  float *dense_B,               // [batch_size, B_num_block, dim, 32]，稠密矩阵 B
  int   *indices,               // [batch_size, num_block]，存储每个 batch 中每个 block 的索引
  float *sparse_C,              // [batch_size, num_block, 32, 32]，稀疏矩阵 C
  long batch_size,              // batch 大小
  long A_num_block,             // A 的 block 数
  long B_num_block,             // B 的 block 数
  long dim,                     // 矩阵维度
  long num_block                // 总 block 数
);

// CUDA 核函数：稀疏矩阵与稠密矩阵相乘
__global__ void sparse_dense_mm_cuda_kernel(
  float *sparse_A,              // [batch_size, num_block, 32, 32]，稀疏矩阵 A
  int   *indices,               // [batch_size, num_block]，存储每个 batch 中每个 block 的索引
  float *dense_B,               // [batch_size, B_num_block, dim, 32]，稠密矩阵 B
  float *dense_C,               // [batch_size, A_num_block, dim, 32]，稠密矩阵 C
  long batch_size,              // batch 大小
  long A_num_block,             // A 的 block 数
  long B_num_block,             // B 的 block 数
  long dim,                     // 矩阵维度
  long num_block                // 总 block 数
);

// CUDA 核函数：对稀疏矩阵进行按行求和
__global__ void reduce_sum_cuda_kernel(
  float *sparse_A,              // [batch_size, num_block, 32, 32]，稀疏矩阵 A
  int   *indices,               // [batch_size, num_block]，存储每个 batch 中每个 block 的索引
  float *dense_C,               // [batch_size, A_num_block, 32]，稠密矩阵 C
  long batch_size,              // batch 大小
  long A_num_block,             // A 的 block 数
  long B_num_block,             // B 的 block 数
  long num_block                // 总 block 数
);

// CUDA 核函数：将稠密矩阵的数据按照 indices 分散到稀疏矩阵中
__global__ void scatter_cuda_kernel(
  float *dense_A,               // [batch_size, A_num_block, 32]，稠密矩阵 A
  int   *indices,               // [batch_size, num_block]，存储每个 batch 中每个 block 的索引
  float *sparse_C,              // [batch_size, num_block, 32, 32]，稀疏矩阵 C
  long batch_size,              // batch 大小
  long A_num_block,             // A 的 block 数
  long B_num_block,             // B 的 block 数
  long num_block                // 总 block 数
);
```