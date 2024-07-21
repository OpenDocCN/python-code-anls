# `.\pytorch\aten\src\ATen\native\LossMulti.h`

```py
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

namespace at::native {
  // 多标签边际损失函数的形状检查函数
  inline void multilabel_margin_loss_shape_check(
    int64_t& nframe,                         // 批次大小（帧数）
    int64_t& dim,                            // 向量或矩阵的维度
    const int64_t& ndims,                    // 输入数据的维度
    const Tensor& input,                     // 输入张量
    const Tensor& target) {                  // 目标张量
    // 检查输入张量的维度是否符合预期
    TORCH_CHECK(
        (ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0,
        "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
        input.sizes());

    if (ndims <= 1) {
      nframe = 1;
      dim = ndims == 0 ? 1 : input.size(0);
      // 如果维度小于等于1，检查目标张量的维度和大小是否符合预期
      TORCH_CHECK(
          target.dim() <= 1 && target.numel() == dim,
          "inconsistent target size: ", target.sizes(), " for input of size: ",
          input.sizes());
    } else {
      nframe = input.size(0);
      dim = input.size(1);
      // 如果维度大于1，检查目标张量的维度和大小是否符合预期
      TORCH_CHECK(
          target.dim() == 2 && target.size(0) == nframe &&
          target.size(1) == dim,
          "inconsistent target size: ", target.sizes(), " for input of size: ",
          input.sizes());
    }
  }

  // 多类边际损失函数的形状检查函数
  inline void multi_margin_loss_shape_check(
    int64_t& nframe,                         // 批次大小（帧数）
    int64_t& dim,                            // 向量或矩阵的维度
    const int64_t& ndims,                    // 输入数据的维度
    const Tensor& input,                     // 输入张量
    const Tensor& target,                    // 目标张量
    const std::optional<Tensor>& weight) {   // 权重张量（可选）
    // 检查输入张量的维度是否符合预期
    TORCH_CHECK(
        (ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0,
        "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
        input.sizes());

    if (ndims <= 1) {
      nframe = 1;
      dim = ndims == 0 ? 1 : input.size(0);
    } else {
      nframe = input.size(0);
      dim = input.size(1);
    }

    // 检查目标张量的维度和大小是否符合预期
    TORCH_CHECK(
        target.dim() <= 1 && target.numel() == nframe,
        "inconsistent target size, expected ", nframe, " but got ",
        target.sizes());
    // 如果存在权重张量且已定义，检查其维度和大小是否符合预期
    if (weight && weight->defined()) {
      TORCH_CHECK(
          weight->dim() <= 1 && weight->numel() == dim,
          "inconsistent weight size, expected ", dim, " but got ",
          weight->sizes());
    }
}

} // namespace at::native
```