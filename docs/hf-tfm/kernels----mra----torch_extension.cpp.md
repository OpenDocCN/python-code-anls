# `.\transformers\kernels\mra\torch_extension.cpp`

```
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include <vector>

// 定义函数 index_max，接受 index_vals、indices、A_num_block 和 B_num_block 四个参数，返回一个包含 Tensor 的 vector
std::vector<at::Tensor> index_max(
  at::Tensor index_vals,
  at::Tensor indices,
  int A_num_block,
  int B_num_block
) {
  // 调用 index_max_kernel 函数，传入参数 index_vals、indices、A_num_block 和 B_num_block
  return index_max_kernel(
    index_vals,
    indices,
    A_num_block,
    B_num_block
  );
}

// 定义函数 mm_to_sparse，接受 dense_A、dense_B 和 indices 三个参数，返回一个 Tensor
at::Tensor mm_to_sparse(
  at::Tensor dense_A,
  at::Tensor dense_B,
  at::Tensor indices
) {
  // 调用 mm_to_sparse_kernel 函数，传入参数 dense_A、dense_B 和 indices
  return mm_to_sparse_kernel(
    dense_A,
    dense_B,
    indices
  );
}

// 定义函数 sparse_dense_mm，接受 sparse_A、indices、dense_B 和 A_num_block 四个参数，返回一个 Tensor
at::Tensor sparse_dense_mm(
  at::Tensor sparse_A,
  at::Tensor indices,
  at::Tensor dense_B,
  int A_num_block
) {
  // 调用 sparse_dense_mm_kernel 函数，传入参数 sparse_A、indices、dense_B 和 A_num_block
  return sparse_dense_mm_kernel(
    sparse_A,
    indices,
    dense_B,
    A_num_block
  );
}

// 定义函数 reduce_sum，接受 sparse_A、indices、A_num_block 和 B_num_block 四个参数，返回一个 Tensor
at::Tensor reduce_sum(
  at::Tensor sparse_A,
  at::Tensor indices,
  int A_num_block,
  int B_num_block
) {
  // 调用 reduce_sum_kernel 函数，传入参数 sparse_A、indices、A_num_block 和 B_num_block
  return reduce_sum_kernel(
    sparse_A,
    indices,
    A_num_block,
    B_num_block
  );
}

// 定义函数 scatter，接受 dense_A、indices 和 B_num_block 三个参数，返回一个 Tensor
at::Tensor scatter(
  at::Tensor dense_A,
  at::Tensor indices,
  int B_num_block
) {
  // 调用 scatter_kernel 函数，传入参数 dense_A、indices 和 B_num_block
  return scatter_kernel(
    dense_A,
    indices,
    B_num_block
  );
}

// 使用 PYBIND11_MODULE 宏定义 Python 模块，注册 index_max、mm_to_sparse、sparse_dense_mm、reduce_sum 和 scatter 函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_max", &index_max, "index_max (CUDA)");
  m.def("mm_to_sparse", &mm_to_sparse, "mm_to_sparse (CUDA)");
  m.def("sparse_dense_mm", &sparse_dense_mm, "sparse_dense_mm (CUDA)");
  m.def("reduce_sum", &reduce_sum, "reduce_sum (CUDA)");
  m.def("scatter", &scatter, "scatter (CUDA)");
}
```