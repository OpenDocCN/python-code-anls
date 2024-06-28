# `.\kernels\yoso\fast_lsh_cumulation.h`

```py
// 导入 PyTorch C++ 扩展头文件
#include <torch/extension.h>
// 导入 ATen 库的头文件
#include <ATen/ATen.h>
// 导入 STL 中的 vector 容器
#include <vector>

// 定义快速哈希（版本1）的核函数，返回多个张量作为结果
std::vector<at::Tensor> fast_hash_ver1_kernel(
  // 查询掩码张量
  at::Tensor query_mask,
  // 查询向量张量
  at::Tensor query_vector,
  // 关键字掩码张量
  at::Tensor key_mask,
  // 关键字向量张量
  at::Tensor key_vector,
  // 哈希函数数量
  int num_hash_f,
  // 哈希码长度
  int hash_code_len,
  // 是否使用 CUDA
  bool use_cuda
);

// 定义哈希累积（版本1）的核函数，返回张量作为结果
at::Tensor lsh_cumulation_ver1_kernel(
  // 查询掩码张量
  at::Tensor query_mask,
  // 查询哈希码张量
  at::Tensor query_hash_code,
  // 关键字掩码张量
  at::Tensor key_mask,
  // 关键字哈希码张量
  at::Tensor key_hash_code,
  // 值张量
  at::Tensor value,
  // 哈希表容量
  int hashtable_capacity,
  // 是否使用 CUDA
  bool use_cuda
);

// 定义加权哈希累积（版本1）的核函数，返回张量作为结果
at::Tensor lsh_weighted_cumulation_ver1_kernel(
  // 查询掩码张量
  at::Tensor query_mask,
  // 查询哈希码张量
  at::Tensor query_hash_code,
  // 查询权重张量
  at::Tensor query_weight,
  // 关键字掩码张量
  at::Tensor key_mask,
  // 关键字哈希码张量
  at::Tensor key_hash_code,
  // 关键字权重张量
  at::Tensor key_weight,
  // 值张量
  at::Tensor value,
  // 哈希表容量
  int hashtable_capacity,
  // 是否使用 CUDA
  bool use_cuda
);

// 定义加权哈希累积（版本2、3、4）的核函数，具体功能与版本1类似
// 只是版本号不同，参数及返回值的张量类型与数量相同，不再重复注释每个版本的功能
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