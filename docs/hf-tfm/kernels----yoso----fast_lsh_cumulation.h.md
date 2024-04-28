# `.\transformers\kernels\yoso\fast_lsh_cumulation.h`

```
// 导入 Torch C++ 扩展库
#include <torch/extension.h>
// 导入 ATen 库
#include <ATen/ATen.h>
// 导入 vector 库
#include <vector>

// 快速哈希版本1的核心函数，接受查询掩码、查询向量、键掩码、键向量、哈希函数数量、哈希码长度和是否使用 CUDA
std::vector<at::Tensor> fast_hash_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_vector,
  at::Tensor key_mask,
  at::Tensor key_vector,
  int num_hash_f,
  int hash_code_len,
  bool use_cuda
);

// LSH 累积版本1的核心函数，接受查询掩码、查询哈希码、键掩码、键哈希码、值、哈希表容量和是否使用 CUDA
at::Tensor lsh_cumulation_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
);

// LSH 加权累积版本1的核心函数，接受查询掩码、查询哈希码、查询权重、键掩码、键哈希码、键权重、值、哈希表容量和是否使用 CUDA
at::Tensor lsh_weighted_cumulation_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
);

// LSH 加权累积版本2的核心函数，接受查询掩码、查询哈希码、查询权重、键掩码、键哈希码、键权重、值、哈希表容量和是否使用 CUDA
at::Tensor lsh_weighted_cumulation_ver2_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
);

// LSH 加权累积版本3的核心函数，接受查询掩码、查询哈希码、查询权重、键掩码、键哈希码、键权重、值、哈希表容量和是否使用 CUDA
at::Tensor lsh_weighted_cumulation_ver3_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
);

// LSH 加权累积版本4的核心函数，接受查询掩码、查询哈希码、查询权重、键掩码、键哈希码、键权重、值、哈希表容量和是否使用 CUDA
at::Tensor lsh_weighted_cumulation_ver4_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
);
```