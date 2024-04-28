# `.\transformers\kernels\mra\cuda_launch.h`

```
// 引入 Torch C++ 扩展的头文件
#include <torch/extension.h>
// 引入 ATen 库的头文件
#include <ATen/ATen.h>
// 引入 vector 标准库
#include <vector>

// 定义宏，返回两个值中的最小值
#define min(a, b) ((a)<(b)?(a):(b))
// 定义宏，返回两个值中的最大值
#define max(a, b) ((a)>(b)?(a):(b))

// 声明一个函数，该函数接受两个张量和两个整数参数，返回一个张量数组
std::vector<at::Tensor> index_max_kernel(
  // 输入张量，包含索引值
  at::Tensor index_vals,
  // 输入张量，包含索引位置
  at::Tensor indices,
  // 整数参数，表示 A 的块数
  int A_num_block,
  // 整数参数，表示 B 的块数
  int B_num_block
);

// 声明一个函数，该函数接受两个密集张量和一个索引张量作为输入，返回一个稀疏张量
at::Tensor mm_to_sparse_kernel(
  // 输入张量，表示密集张量 A
  at::Tensor dense_A,
  // 输入张量，表示密集张量 B
  at::Tensor dense_B,
  // 输入张量，表示索引
  at::Tensor indices
);

// 声明一个函数，该函数接受一个稀疏张量、一个索引张量和一个密集张量作为输入，返回一个密集张量
at::Tensor sparse_dense_mm_kernel(
  // 输入张量，表示稀疏张量 A
  at::Tensor sparse_A,
  // 输入张量，表示索引
  at::Tensor indices,
  // 输入张量，表示密集张量 B
  at::Tensor dense_B,
  // 整数参数，表示 A 的块数
  int A_num_block
);

// 声明一个函数，该函数接受一个稀疏张量、一个索引张量和两个整数参数作为输入，返回一个张量
at::Tensor reduce_sum_kernel(
  // 输入张量，表示稀疏张量 A
  at::Tensor sparse_A,
  // 输入张量，表示索引
  at::Tensor indices,
  // 整数参数，表示 A 的块数
  int A_num_block,
  // 整数参数，表示 B 的块数
  int B_num_block
);

// 声明一个函数，该函数接受一个密集张量、一个索引张量和一个整数参数作为输入，返回一个张量
at::Tensor scatter_kernel(
  // 输入张量，表示密集张量 A
  at::Tensor dense_A,
  // 输入张量，表示索引
  at::Tensor indices,
  // 整数参数，表示 B 的块数
  int B_num_block
);
```