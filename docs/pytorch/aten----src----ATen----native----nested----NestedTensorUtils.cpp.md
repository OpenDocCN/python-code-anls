# `.\pytorch\aten\src\ATen\native\nested\NestedTensorUtils.cpp`

```
/**
 * 包含 ATen 的 NestedTensorImpl.h 头文件
 * 包含 ATen 的 nested tensor 工具函数头文件
 * 包含 C++ 标准库中的 Optional 头文件
 */
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <c10/util/Optional.h>

/**
 * 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 的 NativeFunctions.h 头文件；
 * 否则包含一系列用于嵌套张量操作的特定头文件。
 */
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_tensor_size_native.h>
#include <ATen/ops/_nested_tensor_storage_offsets_native.h>
#include <ATen/ops/_nested_tensor_strides_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#endif

namespace at {
namespace native {

/**
 * _nested_tensor_size 函数的简单包装，作为原生函数注册
 *
 * @param self 嵌套张量
 * @return 嵌套张量的大小张量
 */
at::Tensor _nested_tensor_size(const at::Tensor& self) {
  return get_nested_sizes(self);
}

/**
 * 获取嵌套张量的步长张量
 *
 * @param self 嵌套张量
 * @return 嵌套张量的步长张量
 */
at::Tensor _nested_tensor_strides(const at::Tensor& self){
  return  get_nested_tensor_impl(self) -> get_nested_strides();
}

/**
 * 获取嵌套张量的存储偏移量张量
 *
 * @param self 嵌套张量
 * @return 嵌套张量的存储偏移量张量
 */
at::Tensor _nested_tensor_storage_offsets(const at::Tensor& self){
  return get_nested_tensor_impl(self) -> get_storage_offsets();
}

// Helper functions for getting information about a nested tensor's shape.

/**
 * 从大小张量中获取嵌套张量的最大大小
 *
 * @param sizes 大小张量
 * @return 嵌套张量的最大大小
 */
std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(
    const Tensor& sizes) {
  if (sizes.dim() == 0) {
    return {};
  }
  const auto sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_0 = sizes.sizes()[0];
  const auto sizes_size_1 = sizes.sizes()[1];
  TORCH_INTERNAL_ASSERT(sizes_size_1 > 0);
  std::vector<int64_t> results(sizes_size_1, 0);
  for (const auto ii : c10::irange(sizes_size_0)) {
    for (const auto jj : c10::irange(sizes_size_1)) {
      auto val = sizes_ptr[ii * sizes_size_1 + jj];
      if (results[jj] < val) {
        results[jj] = val;
      }
    }
  }
  return results;
}

/**
 * 获取嵌套张量的最大大小
 *
 * @param nt 嵌套张量实现对象
 * @return 嵌套张量的最大大小
 */
std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt) {
  return NestedTensor_get_max_size_from_size_tensor(
      nt.get_nested_sizes());
}

/**
 * 获取嵌套张量一致的最后一个维度
 *
 * @param nt 嵌套张量实现对象
 * @return 嵌套张量一致的最后一个维度
 */
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt) {
  std::optional<int64_t> last_dim = nt.opt_size(-1);
  TORCH_CHECK(
      last_dim != c10::nullopt,
      "Expected all tensors in nested tensor to have the same trailing dimension, instead last dimension equals: ",
      nt.get_nested_sizes().select(1, -1));
  return *last_dim;
}

/**
 * 对嵌套张量进行分块的辅助函数
 *
 * @param self 嵌套张量
 * @param chunks 分块数
 * @param dim 分块维度
 * @return 分块后的张量向量
 */
std::vector<Tensor> chunk_nested_tensor(const Tensor& self, int64_t chunks, int64_t dim) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "chunk() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(self.dim() - 1 == dim,
           "Chunk for nested tensors is currently only supported for the last dimension.");
  TORCH_CHECK(chunks > 0,"chunk expects `chunks` to be greater than 0, got: ", chunks);
  TORCH_CHECK(self.is_contiguous(), "chunk expects `self` to be contiguous.");
  auto self_impl = get_nested_tensor_impl(self);
  const int64_t last_dim_size = get_consistent_last_dim_of_nested_tensor(*self_impl);
    // 检查最后一个维度的大小是否能被 chunks 整除，否则抛出错误信息
    TORCH_CHECK(last_dim_size % chunks == 0,
           "Chunk for nested tensors is only supported for nested tensors with trailing dimension divisible by chunks, got: ",
           last_dim_size, " % ", chunks, " != 0");
  // 获取张量对象的第一个维度大小
  int64_t n_tensors = self.size(0);
  // 计算每个分块的大小
  int64_t split_size = last_dim_size / chunks;
  // 创建存储分块张量的向量
  std::vector<Tensor> splits(chunks);
  // 获取张量对象的嵌套大小
  const auto& sizes = self_impl->get_nested_sizes();
  // 获取张量对象的嵌套步长
  const auto& strides = self_impl->get_nested_strides();
  // 获取张量对象的存储偏移量
  const auto offsets = self_impl->get_storage_offsets();
  // 获取存储偏移量的指针
  int64_t *offsets_ptr = offsets.data_ptr<int64_t>();
  // 考虑到隐含的批处理维度，将维度减一
  --dim;
  // 获取张量维度的大小
  int64_t tensor_dim = sizes.size(1);
  // 遍历每一个分块
  for (const auto split_idx : c10::irange(chunks)) {
      // 克隆张量的大小
      auto new_sizes = sizes.clone();
      // 克隆张量的步长
      auto new_strides = strides.clone();
      // 复制张量的偏移量以确保安全移动
      auto new_offsets = offsets.clone();
      // 获取大小指针
      int64_t *size_ptr = new_sizes.data_ptr<int64_t>();
      // 获取新偏移量的指针
      int64_t *new_offsets_ptr = new_offsets.data_ptr<int64_t>();
      // 计算每个分块的起始值
      int64_t start_val = split_idx * split_size;
      // 遍历张量对象中的每一个张量
      for (int64_t i : c10::irange(n_tensors)) {
        // 计算索引
        const int64_t index = i * tensor_dim + dim;
        // 更新偏移量指针
        new_offsets_ptr[i] = offsets_ptr[i] + start_val;
        // 更新大小指针
        size_ptr[index] = split_size;
    }
    // 创建新的嵌套视图张量并将其存储在分块中
    splits[split_idx] = create_nested_view_tensor(self, new_sizes, new_strides, new_offsets);
  }
  // 返回所有分块的张量向量
  return splits;
} // 结束 namespace at

} // 结束 namespace native

// 分割嵌套张量的函数，根据指定大小和维度进行分割
std::vector<Tensor> split_with_sizes_nested(
    // 输入张量自身
    const Tensor& self,
    // 分割的大小列表
    c10::IntArrayRef split_sizes,
    // 分割的维度
    int64_t dim) {
  
  // 获取输入张量的维度数
  int64_t ndim = self.dim();
  // 如果张量维度为0，抛出错误
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "split_with_sizes() cannot be applied to a 0-dim tensor.");
  }
  
  // 确定有效的维度值，处理负数维度索引
  dim = maybe_wrap_dim(dim, ndim);
  
  // 检查是否在最后一个维度上进行分割
  TORCH_CHECK(self.dim() - 1 == dim,
           "split_with_sizes for nested tensors is currently only supported for the last dimension.");
  
  // 获取分割的数量
  auto num_splits = split_sizes.size();
  // 检查分割的数量大于0
  TORCH_CHECK(num_splits > 0,
           "split_with_sizes expects number of splits to be greater than 0, got: ", num_splits);
  
  // 检查张量是否是连续的
  TORCH_CHECK(self.is_contiguous(), "split_with_sizes expects `self` to be contiguous.");

  // 确保整个维度被完全分割
  int64_t total_size = 0;
  for (const auto split_size : split_sizes) {
      total_size += split_size;
  }
  
  // 获取嵌套张量的实现
  auto self_impl = get_nested_tensor_impl(self);
  // 获取嵌套张量中最后一个维度的一致大小
  auto self_size = get_consistent_last_dim_of_nested_tensor(*self_impl);
  // 检查分割的大小是否与张量的最后一个维度大小相匹配
  TORCH_CHECK(total_size == self_size,
          "split_with_sizes expects split_sizes to sum exactly to ", self_size,
          " (input tensor's size at dimension ", dim, "), but got split_sizes=", split_sizes);

  // 获取张量的数量
  int64_t n_tensors = self.size(0);
  // 创建存储分割结果的向量
  std::vector<Tensor> splits(num_splits);
  // 获取嵌套张量的大小和步长信息
  const auto& sizes = self_impl->get_nested_sizes();
  const auto& strides = self_impl->get_nested_strides();
  const auto offsets = self_impl->get_storage_offsets();
  int64_t *offsets_ptr = offsets.data_ptr<int64_t>();
  
  // 考虑隐式的批次维度
  --dim;
  int64_t tensor_dim = sizes.size(1);
  int64_t start_val = 0;
  
  // 对每个分割进行处理
  for (const auto split_idx : c10::irange(num_splits)) {
    auto split_size = split_sizes[split_idx];
    auto new_sizes = sizes.clone();
    auto new_strides = strides.clone();
    auto new_offsets = offsets.clone();
    int64_t *size_ptr = new_sizes.data_ptr<int64_t>();
    int64_t *new_offsets_ptr = new_offsets.data_ptr<int64_t>();
    
    // 为每个分割获取起始值
    for (int64_t i : c10::irange(n_tensors)) {
      const int64_t index = i * tensor_dim + dim;
      new_offsets_ptr[i] = offsets_ptr[i] + start_val;
      size_ptr[index] = split_size;
    }
    
    // 更新起始值，准备下一个分割
    start_val += split_size;
    
    // 创建基于新大小、步长和偏移的嵌套视图张量，并存储在结果向量中
    splits[split_idx] = create_nested_view_tensor(self, new_sizes, new_strides, new_offsets);
  }
  
  // 返回分割后的张量向量
  return splits;
}
```