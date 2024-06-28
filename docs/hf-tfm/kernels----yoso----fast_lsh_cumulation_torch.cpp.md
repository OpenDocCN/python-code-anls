# `.\kernels\yoso\fast_lsh_cumulation_torch.cpp`

```py
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "fast_lsh_cumulation.h"  // 引入自定义的头文件，包含快速LSH累积相关的函数声明
#include "common_cuda.h"           // 引入自定义的头文件，包含通用的CUDA函数声明
#include <vector>                  // 引入标准库中的向量容器

// 快速哈希函数，调用指定版本的核函数处理哈希计算
std::vector<at::Tensor> fast_hash(
  at::Tensor query_mask,      // 查询掩码，形状为[batch_size, num_query]
  at::Tensor query_vector,    // 查询向量，形状为[batch_size, num_query, vector_dim]
  at::Tensor key_mask,        // 键掩码，形状为[batch_size, num_key]
  at::Tensor key_vector,      // 键向量，形状为[batch_size, num_key, vector_dim]
  int num_hash_f,             // 哈希函数数量
  int hash_code_len,          // 哈希码长度
  bool use_cuda,              // 是否使用CUDA加速
  int version                 // 函数版本号
) {
  return fast_hash_ver1_kernel(
    query_mask,
    query_vector,
    key_mask,
    key_vector,
    num_hash_f,
    hash_code_len,
    use_cuda
  );
}

// LSH累积函数，调用指定版本的核函数执行LSH累积操作
at::Tensor lsh_cumulation(
  at::Tensor query_mask,         // 查询掩码，形状为[batch_size, num_query]
  at::Tensor query_hash_code,    // 查询哈希码，形状为[batch_size, num_query, num_hash_f]
  at::Tensor key_mask,           // 键掩码，形状为[batch_size, num_key]
  at::Tensor key_hash_code,      // 键哈希码，形状为[batch_size, num_key, num_hash_f]
  at::Tensor value,              // 值，形状为[batch_size, num_key, value_dim]
  int hashtable_capacity,        // 哈希表容量
  bool use_cuda,                 // 是否使用CUDA加速
  int version                    // 函数版本号
) {
  return lsh_cumulation_ver1_kernel(
    query_mask,
    query_hash_code,
    key_mask,
    key_hash_code,
    value,
    hashtable_capacity,
    use_cuda
  );
}

// 加权LSH累积函数，根据版本号调用不同的核函数执行不同版本的加权LSH累积操作
at::Tensor lsh_weighted_cumulation(
  at::Tensor query_mask,         // 查询掩码，形状为[batch_size, num_query]
  at::Tensor query_hash_code,    // 查询哈希码，形状为[batch_size, num_query, num_hash_f]
  at::Tensor query_weight,       // 查询权重，形状为[batch_size, num_query, weight_dim]
  at::Tensor key_mask,           // 键掩码，形状为[batch_size, num_key]
  at::Tensor key_hash_code,      // 键哈希码，形状为[batch_size, num_key, num_hash_f]
  at::Tensor key_weight,         // 键权重，形状为[batch_size, num_key, weight_dim]
  at::Tensor value,              // 值，形状为[batch_size, num_key, value_dim]
  int hashtable_capacity,        // 哈希表容量
  bool use_cuda,                 // 是否使用CUDA加速
  int version                    // 函数版本号
) {
  if (version == 1) {
    return lsh_weighted_cumulation_ver1_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else if (version == 2) {
    return lsh_weighted_cumulation_ver2_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else if (version == 3) {
    return lsh_weighted_cumulation_ver3_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else if (version == 4) {
    return lsh_weighted_cumulation_ver4_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else {
    // 默认情况下使用第三个版本的核函数
    return lsh_weighted_cumulation_ver3_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  }
}
# 使用 PYBIND11_MODULE 宏定义一个 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  # 将 fast_hash 函数绑定到 Python 模块中，并命名为 "Fast Hash (CUDA)"
  m.def("fast_hash", &fast_hash, "Fast Hash (CUDA)");
  # 将 lsh_cumulation 函数绑定到 Python 模块中，并命名为 "LSH Cumulation (CUDA)"
  m.def("lsh_cumulation", &lsh_cumulation, "LSH Cumulation (CUDA)");
  # 将 lsh_weighted_cumulation 函数绑定到 Python 模块中，并命名为 "LSH Weighted Cumulation (CUDA)"
  m.def("lsh_weighted_cumulation", &lsh_weighted_cumulation, "LSH Weighted Cumulation (CUDA)");
}
```