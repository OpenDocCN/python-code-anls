# `.\kernels\mra\torch_extension.cpp`

```
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"  // 引入 CUDA 相关的头文件
#include <vector>  // 引入 vector 容器的头文件

std::vector<at::Tensor> index_max(  // 定义函数 index_max，返回一个 Tensor 向量
  at::Tensor index_vals,  // 输入参数 index_vals，类型为 Tensor
  at::Tensor indices,  // 输入参数 indices，类型为 Tensor
  int A_num_block,  // 输入参数 A_num_block，整型
  int B_num_block  // 输入参数 B_num_block，整型
) {
  return index_max_kernel(  // 调用 index_max_kernel 函数，返回其结果
    index_vals,  // 将 index_vals 作为参数传递给 index_max_kernel 函数
    indices,  // 将 indices 作为参数传递给 index_max_kernel 函数
    A_num_block,  // 将 A_num_block 作为参数传递给 index_max_kernel 函数
    B_num_block  // 将 B_num_block 作为参数传递给 index_max_kernel 函数
  );
}

at::Tensor mm_to_sparse(  // 定义函数 mm_to_sparse，返回一个 Tensor
  at::Tensor dense_A,  // 输入参数 dense_A，类型为 Tensor
  at::Tensor dense_B,  // 输入参数 dense_B，类型为 Tensor
  at::Tensor indices  // 输入参数 indices，类型为 Tensor
) {
  return mm_to_sparse_kernel(  // 调用 mm_to_sparse_kernel 函数，返回其结果
    dense_A,  // 将 dense_A 作为参数传递给 mm_to_sparse_kernel 函数
    dense_B,  // 将 dense_B 作为参数传递给 mm_to_sparse_kernel 函数
    indices  // 将 indices 作为参数传递给 mm_to_sparse_kernel 函数
  );
}

at::Tensor sparse_dense_mm(  // 定义函数 sparse_dense_mm，返回一个 Tensor
  at::Tensor sparse_A,  // 输入参数 sparse_A，类型为 Tensor
  at::Tensor indices,  // 输入参数 indices，类型为 Tensor
  at::Tensor dense_B,  // 输入参数 dense_B，类型为 Tensor
  int A_num_block  // 输入参数 A_num_block，整型
) {
  return sparse_dense_mm_kernel(  // 调用 sparse_dense_mm_kernel 函数，返回其结果
    sparse_A,  // 将 sparse_A 作为参数传递给 sparse_dense_mm_kernel 函数
    indices,  // 将 indices 作为参数传递给 sparse_dense_mm_kernel 函数
    dense_B,  // 将 dense_B 作为参数传递给 sparse_dense_mm_kernel 函数
    A_num_block  // 将 A_num_block 作为参数传递给 sparse_dense_mm_kernel 函数
  );
}

at::Tensor reduce_sum(  // 定义函数 reduce_sum，返回一个 Tensor
  at::Tensor sparse_A,  // 输入参数 sparse_A，类型为 Tensor
  at::Tensor indices,  // 输入参数 indices，类型为 Tensor
  int A_num_block,  // 输入参数 A_num_block，整型
  int B_num_block  // 输入参数 B_num_block，整型
) {
  return reduce_sum_kernel(  // 调用 reduce_sum_kernel 函数，返回其结果
    sparse_A,  // 将 sparse_A 作为参数传递给 reduce_sum_kernel 函数
    indices,  // 将 indices 作为参数传递给 reduce_sum_kernel 函数
    A_num_block,  // 将 A_num_block 作为参数传递给 reduce_sum_kernel 函数
    B_num_block  // 将 B_num_block 作为参数传递给 reduce_sum_kernel 函数
  );
}

at::Tensor scatter(  // 定义函数 scatter，返回一个 Tensor
  at::Tensor dense_A,  // 输入参数 dense_A，类型为 Tensor
  at::Tensor indices,  // 输入参数 indices，类型为 Tensor
  int B_num_block  // 输入参数 B_num_block，整型
) {
  return scatter_kernel(  // 调用 scatter_kernel 函数，返回其结果
    dense_A,  // 将 dense_A 作为参数传递给 scatter_kernel 函数
    indices,  // 将 indices 作为参数传递给 scatter_kernel 函数
    B_num_block  // 将 B_num_block 作为参数传递给 scatter_kernel 函数
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // 定义 Python 扩展模块
  m.def("index_max", &index_max, "index_max (CUDA)");  // 将 index_max 函数绑定到 Python 中，并指定描述
  m.def("mm_to_sparse", &mm_to_sparse, "mm_to_sparse (CUDA)");  // 将 mm_to_sparse 函数绑定到 Python 中，并指定描述
  m.def("sparse_dense_mm", &sparse_dense_mm, "sparse_dense_mm (CUDA)");  // 将 sparse_dense_mm 函数绑定到 Python 中，并指定描述
  m.def("reduce_sum", &reduce_sum, "reduce_sum (CUDA)");  // 将 reduce_sum 函数绑定到 Python 中，并指定描述
  m.def("scatter", &scatter, "scatter (CUDA)");  // 将 scatter 函数绑定到 Python 中，并指定描述
}
```