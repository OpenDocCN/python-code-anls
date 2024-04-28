# `.\transformers\kernels\yoso\fast_lsh_cumulation_torch.cpp`

```
#include <torch/extension.h>  // 引入 PyTorch C++ 扩展头文件
#include <ATen/ATen.h>  // 引入 ATen 头文件
#include "fast_lsh_cumulation.h"  // 引入自定义的快速局部敏感哈希累积头文件
#include "common_cuda.h"  // 引入自定义的 CUDA 公共头文件
#include <vector>  // 引入 C++ 标准库中的向量容器

// 快速哈希函数，返回哈希结果
std::vector<at::Tensor> fast_hash(
  at::Tensor query_mask,  // 查询掩码
  at::Tensor query_vector,  // 查询向量
  at::Tensor key_mask,  // 关键字掩码
  at::Tensor key_vector,  // 关键字向量
  int num_hash_f,  // 哈希函数数量
  int hash_code_len,  // 哈希码长度
  bool use_cuda,  // 是否使用 CUDA
  int version  // 版本号
) {
  // 调用快速哈希版本1的核函数
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

// 局部敏感哈希累积函数，返回累积结果
at::Tensor lsh_cumulation(
  at::Tensor query_mask,  // 查询掩码
  at::Tensor query_hash_code,  // 查询哈希码
  at::Tensor key_mask,  // 关键字掩码
  at::Tensor key_hash_code,  // 关键字哈希码
  at::Tensor value,  // 值
  int hashtable_capacity,  // 哈希表容量
  bool use_cuda,  // 是否使用 CUDA
  int version  // 版本号
) {
  // 调用局部敏感哈希累积版本1的核函数
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

// 加权局部敏感哈希累积函数，返回累积结果
at::Tensor lsh_weighted_cumulation(
  at::Tensor query_mask,  // 查询掩码
  at::Tensor query_hash_code,  // 查询哈希码
  at::Tensor query_weight,  // 查询权重
  at::Tensor key_mask,  // 关键字掩码
  at::Tensor key_hash_code,  // 关键字哈希码
  at::Tensor key_weight,  // 关键字权重
  at::Tensor value,  // 值
  int hashtable_capacity,  // 哈希表容量
  bool use_cuda,  // 是否使用 CUDA
  int version  // 版本号
) {
  // 根据版本号选择不同的加权局部敏感哈希累积核函数
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
# 使用 PYBIND11_MODULE 宏定义一个 Python 模块，模块名为 TORCH_EXTENSION_NAME，模块对象为 m
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    # 将 fast_hash 函数绑定到模块 m 中，函数名为 "fast_hash"，函数指针为 &fast_hash，描述为 "Fast Hash (CUDA)"
    m.def("fast_hash", &fast_hash, "Fast Hash (CUDA)");
    # 将 lsh_cumulation 函数绑定到模块 m 中，函数名为 "lsh_cumulation"，函数指针为 &lsh_cumulation，描述为 "LSH Cumulation (CUDA)"
    m.def("lsh_cumulation", &lsh_cumulation, "LSH Cumulation (CUDA)");
    # 将 lsh_weighted_cumulation 函数绑定到模块 m 中，函数名为 "lsh_weighted_cumulation"，函数指针为 &lsh_weighted_cumulation，描述为 "LSH Weighted Cumulation (CUDA)"
    m.def("lsh_weighted_cumulation", &lsh_weighted_cumulation, "LSH Weighted Cumulation (CUDA)");
}
```