# `.\kernels\mra\cuda_launch.h`

```
# 包含 Torch C++ 扩展的头文件
#include <torch/extension.h>
# 包含 ATen 库的头文件，用于张量操作
#include <ATen/ATen.h>
# 包含 vector 标准库，用于定义和操作动态数组
#include <vector>

# 定义宏函数 min，返回两个数中较小的那个
#define min(a, b) ((a)<(b)?(a):(b))
# 定义宏函数 max，返回两个数中较大的那个
#define max(a, b) ((a)>(b)?(a):(b))

# 声明一个函数，该函数返回一个包含多个张量的 vector
std::vector<at::Tensor> index_max_kernel(
  at::Tensor index_vals,   # 输入参数：索引值的张量
  at::Tensor indices,      # 输入参数：索引的张量
  int A_num_block,         # 输入参数：A 矩阵的块数量
  int B_num_block          # 输入参数：B 矩阵的块数量
);

# 声明一个函数，该函数执行稠密矩阵乘法并返回一个稀疏张量
at::Tensor mm_to_sparse_kernel(
  at::Tensor dense_A,      # 输入参数：稠密矩阵 A
  at::Tensor dense_B,      # 输入参数：稠密矩阵 B
  at::Tensor indices       # 输入参数：索引张量
);

# 声明一个函数，该函数执行稀疏矩阵与稠密矩阵的乘法并返回结果张量
at::Tensor sparse_dense_mm_kernel(
  at::Tensor sparse_A,     # 输入参数：稀疏矩阵 A
  at::Tensor indices,      # 输入参数：索引张量
  at::Tensor dense_B,      # 输入参数：稠密矩阵 B
  int A_num_block          # 输入参数：A 矩阵的块数量
);

# 声明一个函数，该函数执行稀疏矩阵的元素求和操作并返回结果张量
at::Tensor reduce_sum_kernel(
  at::Tensor sparse_A,     # 输入参数：稀疏矩阵 A
  at::Tensor indices,      # 输入参数：索引张量
  int A_num_block,         # 输入参数：A 矩阵的块数量
  int B_num_block          # 输入参数：B 矩阵的块数量
);

# 声明一个函数，该函数执行稠密张量的散布（scatter）操作并返回结果张量
at::Tensor scatter_kernel(
  at::Tensor dense_A,      # 输入参数：稠密张量 A
  at::Tensor indices,      # 输入参数：索引张量
  int B_num_block          # 输入参数：B 矩阵的块数量
);
```