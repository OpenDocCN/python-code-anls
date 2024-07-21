# `.\pytorch\aten\src\ATen\native\sparse\FlattenIndicesCommon.h`

```
#pragma once
// 预处理指令：确保此头文件只被编译一次

#include <ATen/Tensor.h>
// 包含 ATen 库中的 Tensor 头文件，提供张量操作支持
#include <ATen/native/TensorIterator.h>
// 包含 ATen 库中的 Tensor 迭代器相关头文件，支持张量迭代操作
#include <ATen/Dispatch.h>
// 包含 ATen 库中的 Dispatch 头文件，提供分发机制支持
#include <ATen/native/sparse/Macros.h>
// 包含 ATen 库中的稀疏张量宏定义相关头文件
#include <ATen/ExpandUtils.h>
// 包含 ATen 库中的张量扩展工具头文件
#include <ATen/native/SparseTensorUtils.h>
// 包含 ATen 库中的稀疏张量工具函数头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，包含 ATen 库中的功能函数头文件
#include <ATen/NativeFunctions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，包含 ATen 库中的原生函数头文件
#else
#include <ATen/ops/arange.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，包含 ATen 库中的 arange 操作头文件
#include <ATen/ops/tensor.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，包含 ATen 库中的 tensor 操作头文件
#endif

#ifdef GPUCC
#define NAME "flatten_indices_cuda"
// 如果定义了 GPUCC，设置宏 NAME 为 "flatten_indices_cuda"
#else
#define NAME "flatten_indices_cpu"
// 如果未定义 GPUCC，设置宏 NAME 为 "flatten_indices_cpu"
#endif

namespace at::native {

namespace {
// 匿名命名空间，用于隐藏在当前文件中声明的内部函数和变量

template <template <typename func_t> class kernel_t>
// 模板定义：KernelLauncher 模板类
struct KernelLauncher {
  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    // 静态方法 launch：通过 kernel_t 类型的模板实例，启动迭代器 iter 上的计算任务，使用函数对象 f
    kernel_t<func_t>::launch(iter, f);
  }
};

template <
  template <typename func_t> class kernel_t,
  typename index_t,
  int64_t max_static_len = 0>
Tensor _flatten_indices_impl(const Tensor& indices, IntArrayRef size) {
  // 模板函数 _flatten_indices_impl：根据 indices 张量和给定大小 size 执行扁平化索引操作
  TORCH_INTERNAL_ASSERT(indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size());
  // 内部断言：确保 indices 张量维度大于 1，并且第一维度大小与 size 相等

  // 创建 hash_coeffs_storage 匿名函数以确保在 Tensor 类中拥有存储
  const auto hash_coeffs_storage = [&]() -> auto {
    auto strides = c10::contiguous_strides(size);
    return at::sparse::TensorGeometryHolder<max_static_len>(strides, strides, indices.options());
  }();
  const auto hash_coeffs = std::get<0>(*hash_coeffs_storage);
  // 获取 hash_coeffs：稀疏张量几何保持器的第一个元素，用于哈希系数

  const auto hash_indices = [&]() -> Tensor {
    // 创建 hash_indices 匿名函数以计算哈希索引
    auto sparse_dim = indices.size(0);
    auto indices_dim_stride = indices.stride(0);
    auto indices_nnz_stride = indices.stride(1);

    auto hash = at::arange(indices.size(1), indices.options().dtype(kLong));
    // 使用 arange 函数创建与 indices.size(1) 相等的长整型张量 hash

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_output(hash)
      .add_input(hash)
      .build();
    // 创建张量迭代器 iter：输出为 hash，输入为 hash，关闭内存重叠检查

    {
      const auto* RESTRICT ptr_indices = indices.const_data_ptr<index_t>();
      // 常量指针 ptr_indices 指向 indices 的数据，类型为 index_t

      KernelLauncher<kernel_t>::launch(iter,
          // 使用 KernelLauncher 启动 iter 迭代器
          [=] FUNCAPI (int64_t nnz_idx) -> int64_t {
          // 使用匿名 Lambda 函数捕获值启动迭代器，返回 int64_t
          const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
          // 常量指针 ptr_indices_dim 指向 nnz_idx * indices_nnz_stride 的数据

          auto hash = static_cast<int64_t>(0);
          // 哈希设置为零
          for (int64_t dim = 0; dim < sparse_dim; ++dim) {
            const auto dim_hash_coeff = hash_coeffs[dim];
            // 获取维度哈希系数
            const auto dim_index = ptr_indices_dim[dim * indices_dim_stride];
            // 获取维度索引
            hash += dim_index * dim_hash_coeff;
            // 哈希增加维度索引与哈希系数的乘积
          }
          return hash;
      });
    }

    return hash;
    // 返回计算得到的哈希索引张量
  }();

  return hash_indices;
  // 返回计算得到的哈希索引张量
}

template <template <typename func_t> class kernel_t>
// 模板定义：KernelLauncher 结构体
# 将稀疏张量的索引扁平化处理，以适应给定的张量尺寸
Tensor _flatten_indices(const Tensor& indices, IntArrayRef size) {
  # 检查稀疏索引张量的维度大于1，并且其第一维度大小必须与尺寸数组的长度相匹配
  TORCH_CHECK(indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size(),
      NAME, "(): the dimensionality of sparse `indices` and the length of `size` must match. ",
            "Got `indices.size(0) == ", indices.size(0), "` != `size.size() == ", size.size(), "`.");
  
  # 定义扁平化后的索引张量
  Tensor flattened_indices;
  
  # 根据索引张量的数据类型分发执行以下操作
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), NAME, [&] () {
    # 定义最大稀疏维度
    constexpr int64_t max_sparse_dims = 8;
    
    # 如果索引张量的第一维度大小不超过最大稀疏维度，则调用带有指定最大维度的扁平化实现函数
    if (indices.size(0) <= max_sparse_dims) {
      flattened_indices = _flatten_indices_impl<kernel_t, index_t, max_sparse_dims>(indices, size);
    } else {
      # 否则调用不带指定最大维度的扁平化实现函数
      flattened_indices = _flatten_indices_impl<kernel_t, index_t>(indices, size);
    }
  });
  
  # 返回扁平化后的索引张量
  return flattened_indices;
}

}

} // at::native
```