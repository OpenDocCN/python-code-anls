# `.\kernels\mra\cuda_kernel.h`

```
// 定义线程块大小为32
#define WARP_SIZE 32
// 定义全掩码为32位全1
#define FULL_MASK 0xffffffff
// 定义优化线程数为256
#define OPTIMAL_THREADS 256

// CUDA 核函数，计算每个批次中每个块中的最大值索引和最大值
__global__ void index_max_cuda_kernel(
  float *index_vals,       // [batch_size, 32, num_block]
  int   *indices,          // [batch_size, num_block]
  float *max_vals,         // [batch_size, A_num_block * 32]
  float *max_vals_scatter, // [batch_size, 32, num_block]
  long batch_size,         // 批次大小
  long A_num_block,        // A_num_block
  long B_num_block,        // B_num_block
  long num_block           // num_block
);

// CUDA 核函数，将稠密矩阵乘法结果转换为稀疏格式
__global__ void mm_to_sparse_cuda_kernel(
  float *dense_A,   // [batch_size, A_num_block, dim, 32]
  float *dense_B,   // [batch_size, B_num_block, dim, 32]
  int   *indices,   // [batch_size, num_block]
  float *sparse_C,  // [batch_size, num_block, 32, 32]
  long batch_size,  // 批次大小
  long A_num_block, // A_num_block
  long B_num_block, // B_num_block
  long dim,         // dim
  long num_block    // num_block
);

// CUDA 核函数，稀疏矩阵与稠密矩阵的乘法
__global__ void sparse_dense_mm_cuda_kernel(
  float *sparse_A,  // [batch_size, num_block, 32, 32]
  int   *indices,   // [batch_size, num_block]
  float *dense_B,   // [batch_size, B_num_block, dim, 32]
  float *dense_C,   // [batch_size, A_num_block, dim, 32]
  long batch_size,  // 批次大小
  long A_num_block, // A_num_block
  long B_num_block, // B_num_block
  long dim,         // dim
  long num_block    // num_block
);

// CUDA 核函数，计算稀疏矩阵在指定维度上的和
__global__ void reduce_sum_cuda_kernel(
  float *sparse_A,  // [batch_size, num_block, 32, 32]
  int   *indices,   // [batch_size, num_block]
  float *dense_C,   // [batch_size, A_num_block, 32]
  long batch_size,  // 批次大小
  long A_num_block, // A_num_block
  long B_num_block, // B_num_block
  long num_block    // num_block
);

// CUDA 核函数，将稠密矩阵按索引散布到稀疏矩阵中
__global__ void scatter_cuda_kernel(
  float *dense_A,   // [batch_size, A_num_block, 32]
  int   *indices,   // [batch_size, num_block]
  float *sparse_C,  // [batch_size, num_block, 32, 32]
  long batch_size,  // 批次大小
  long A_num_block, // A_num_block
  long B_num_block, // B_num_block
  long num_block    // num_block
);
```