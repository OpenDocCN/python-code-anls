# `.\pytorch\aten\src\ATen\native\SortingUtils.h`

```py
#pragma once
// 只允许本头文件被编译一次

#include <ATen/NumericUtils.h>
// 引入 ATen 的数值工具函数

#include <ATen/native/Resize.h>
// 引入 ATen 的本地 Resize 功能

#include <c10/util/irange.h>
// 引入 c10 库的 irange 功能

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，则引入 ATen 的函数库
#else
#include <ATen/ops/empty.h>
// 否则，引入 ATen 的空操作库
#endif

namespace at::native {

// 确保在 kthvalue 和 mode 操作中得到正确的值和索引
// 这些操作将总是在将维度降为一维后执行
inline void _reduction_with_indices_allocate_or_resize_output(
    Tensor& values,                               // 输出的值
    Tensor& indices,                              // 输出的索引
    const Tensor& self,                           // 输入的自身张量
    int64_t dim_,                                 // 操作的维度
    bool keepdim) {                                // 是否保持维度

  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  // 计算包装后的维度，确保在边界内操作

  auto result_sizes = self.sizes().vec();
  // 获取输入张量的尺寸信息

  if (!result_sizes.empty()) {
    result_sizes[dim] = 1;
    // 如果尺寸不为空，将指定维度的尺寸设为 1
  }

  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    // 检查输出值的类型必须与输入类型相同

    if (!keepdim && values.dim() == self.dim() - 1) {
      // 如果不保持维度，并且值的维度比输入的少一维
      // 在调整大小时保持传入的非连续张量
      values.unsqueeze_(dim);
      // 在指定维度上扩展张量
    }

    resize_output(values, result_sizes);
    // 调整输出值的大小为指定尺寸
  } else {
    values = at::empty(result_sizes, self.options());
    // 否则，创建与指定尺寸和选项相同的空张量作为输出值
  }

  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    // 检查输出索引的类型必须是长整型标量

    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    // 检查输出索引必须在与输入相同的设备上

    if (!keepdim && indices.dim() == self.dim() - 1) {
      // 如果不保持维度，并且索引的维度比输入的少一维
      // 在调整大小时保持传入的非连续张量
      indices.unsqueeze_(dim);
      // 在指定维度上扩展张量
    }

    resize_output(indices, result_sizes);
    // 调整输出索引的大小为指定尺寸
  } else {
    indices = at::empty(result_sizes, self.options().dtype(kLong));
    // 否则，创建与指定尺寸和选项相同的空张量作为输出索引，并指定为长整型
  }
}

// 确保在 topk 操作中得到正确的值和索引
inline void _allocate_or_resize_output_with_indices(
    Tensor& values,                               // 输出的值
    Tensor& indices,                              // 输出的索引
    const Tensor& self,                           // 输入的自身张量
    int64_t dim_,                                 // 操作的维度
    int64_t k) {                                  // topk 操作的 k 值

  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  // 计算包装后的维度，确保在边界内操作

  auto result_sizes = self.sizes().vec();
  // 获取输入张量的尺寸信息

  if (!result_sizes.empty()) {
    result_sizes[dim] = k;
    // 如果尺寸不为空，将指定维度的尺寸设为 k
  }

  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    // 检查输出值的类型必须与输入类型相同

    values.resize_(result_sizes);
    // 调整输出值的大小为指定尺寸
  } else {
    values = at::empty(result_sizes, self.options());
    // 否则，创建与指定尺寸和选项相同的空张量作为输出值
  }

  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    // 检查输出索引的类型必须是长整型标量

    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    // 检查输出索引必须在与输入相同的设备上

    indices.resize_(result_sizes);
    // 调整输出索引的大小为指定尺寸
  } else {
    indices = at::empty(result_sizes, self.options().dtype(kLong));
    // 否则，创建与指定尺寸和选项相同的空张量作为输出索引，并指定为长整型
  }
}

} // namespace at::native
// 结束命名空间 at::native
```