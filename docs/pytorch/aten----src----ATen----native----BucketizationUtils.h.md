# `.\pytorch\aten\src\ATen\native\BucketizationUtils.h`

```py
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/result_type.h>
#endif

namespace at::native {

// 对输入张量进行修剪和可能的转换，确保其连续性和数据类型一致性，以提高性能和一致性
inline void searchsorted_maybe_trim_input_tensors(
    Tensor& trimmed_input,              // 修剪后的输入张量
    Tensor& trimmed_boundaries,         // 修剪后的边界张量
    Tensor& trimmed_sorter,             // 修剪后的排序器张量
    const Tensor& raw_input,            // 原始输入张量
    const Tensor& raw_boundaries,       // 原始边界张量
    const Tensor& raw_sorter) {         // 原始排序器张量
  bool in_is_contiguous = raw_input.is_contiguous();          // 检查原始输入张量是否连续
  bool bd_is_contiguous = raw_boundaries.is_contiguous();     // 检查原始边界张量是否连续
  bool sort_is_contiguous = raw_sorter.is_contiguous();       // 检查原始排序器张量是否连续

  if (!in_is_contiguous) {
    TORCH_WARN_ONCE("torch.searchsorted(): input value tensor is non-contiguous, this will lower the performance due "
      "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous input value "
      "tensor if possible. This message will only appear once per program.");
    trimmed_input = raw_input.contiguous();     // 如果输入张量非连续，创建其连续版本
  }
  if (!bd_is_contiguous) {
    TORCH_WARN_ONCE("torch.searchsorted(): boundary tensor is non-contiguous, this will lower the performance due "
      "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous boundary "
      "tensor if possible. This message will only appear once per program.");
    trimmed_boundaries = raw_boundaries.contiguous();   // 如果边界张量非连续，创建其连续版本
  }
  if (!sort_is_contiguous) {
    TORCH_WARN_ONCE("torch.searchsorted(): sorter tensor is non-contiguous, this will lower the performance due "
      "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous sorter "
      "tensor if possible. This message will only appear once per program.");
    trimmed_sorter = raw_sorter.contiguous();   // 如果排序器张量非连续，创建其连续版本
  }
  if (raw_input.dtype() != raw_boundaries.dtype()) {   // 检查输入张量和边界张量的数据类型是否相同
    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(raw_boundaries, state);   // 更新结果类型状态，用于推断最终的数据类型
    state = at::native::update_result_type_state(raw_input, state);
    ScalarType common_stype = at::native::result_type(state);   // 获取公共的超级类型，以便进行类型转换

    TORCH_INTERNAL_ASSERT(common_stype != ScalarType::Undefined);   // 断言确保找到了有效的公共类型
    if (common_stype != raw_input.scalar_type()) {
      trimmed_input = in_is_contiguous ? raw_input.to(common_stype) : trimmed_input.to(common_stype);
      // 如果输入张量不是公共类型，将其转换为公共类型，确保一致性
    }
    # 如果常见的标量类型与原始边界的标量类型不相同，则执行以下操作
    if (common_stype != raw_boundaries.scalar_type()) {
      # 如果边界数据是连续的，将原始边界转换为常见标量类型；否则，将已修剪的边界转换为常见标量类型。
      trimmed_boundaries = bd_is_contiguous ? raw_boundaries.to(common_stype) : trimmed_boundaries.to(common_stype);
    }
  }
/* 
   搜索排序（searchsorted）：检查和修剪输入张量以供内部不规则张量类使用
   此函数将修剪输入张量和边界张量，并返回修剪后的输入张量
*/
inline void searchsorted_maybe_trim_input_tensors(
    Tensor& trimmed_input,                          // 修剪后的输入张量
    Tensor& trimmed_boundaries,                     // 修剪后的边界张量
    const Tensor& raw_input,                        // 原始输入张量
    const Tensor& raw_boundaries) {                 // 原始边界张量
  Tensor trimmed_sorter;                            // 空的修剪排序器张量
  Tensor raw_sorter;                                // 空的原始排序器张量
  return searchsorted_maybe_trim_input_tensors(     // 调用函数自身，以修剪输入张量和边界张量
      trimmed_input,
      trimmed_boundaries,
      trimmed_sorter,
      raw_input,
      raw_boundaries,
      raw_sorter);
}

/*
   搜索排序（searchsorted）：检查维度是否匹配除了最后一个维度之外的边界张量和输入张量
   如果维度不匹配，则返回 false；否则返回 true
*/
inline bool searchsorted_dims_matched_before_last_dim(const Tensor& boundaries, const Tensor& input) {
  if (boundaries.dim() != input.dim()) {             // 如果张量维度不匹配
    return false;                                   // 返回 false
  }
  const auto& dims_bd = boundaries.sizes();          // 获取边界张量的尺寸
  const auto& dims_in = input.sizes();               // 获取输入张量的尺寸
  for (int64_t dim = 0; dim + 1 < boundaries.dim(); ++dim) {
    if (dims_bd[dim] != dims_in[dim]) {              // 如果当前维度的尺寸不匹配
      return false;                                 // 返回 false
    }
  }
  return true;                                      // 所有维度都匹配，返回 true
}

/*
   搜索排序（searchsorted）：将标量转换为张量
   根据标量和设备类型创建张量，并应用标量推广规则
*/
inline Tensor searchsorted_scalar_tensor(const Scalar& scalar, const c10::Device& device) {
  auto tensor = c10::scalar_to_tensor(scalar, device);  // 将标量转换为张量
  // 这里是为了采用在 native/TypeProperties.h 中定义的标量推广规则
  // 以便与二进制操作中的类型推广规则保持一致
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);  // 设置张量为包装数值类型
  return tensor;                                       // 返回创建的张量
}

/*
   搜索排序（searchsorted）：前置检查
   检查边界张量、输入张量、输出张量、输出整数标志、右侧标志、侧边可选参数和排序器张量的各种条件
*/
inline void searchsorted_pre_check(
    const Tensor& boundaries,                         // 边界张量
    const Tensor& input,                              // 输入张量
    const Tensor& output,                             // 输出张量
    const bool out_int32,                             // 输出整数标志
    const bool right,                                 // 右侧标志
    const std::optional<c10::string_view> side_opt,   // 侧边可选参数
    const Tensor& sorter) {                           // 排序器张量
  if (side_opt) {
    const c10::string_view side = *side_opt;          // 获取侧边参数的值
    TORCH_CHECK(side == "left" || side == "right",    // 检查侧边参数值是否有效
      "torch.searchsorted(): side can only be 'left' or 'right' but got ", side);

    // 假设用户没有显式设置（right=False, side="right"）
    TORCH_CHECK(!right || side == "right",            // 检查右侧标志和侧边参数的矛盾设置
      "torch.searchsorted(): side and right can't be set to opposites, got side of ", side, " while right was True");
  }

  TORCH_CHECK(boundaries.device() == input.device(),  // 检查边界张量和输入张量的设备类型是否一致
    "torch.searchsorted(): boundaries and input value tensors should have same device type, but got boundaries tensor device type ", boundaries.device(), " and input value tensor device type ", input.device());

  if (sorter.defined()) {
    TORCH_CHECK(sorter.device() == boundaries.device(),  // 检查排序器张量和边界张量的设备类型是否一致
      "torch.searchsorted(): sorter and boundary tensors should have same device type, but got sorter tensor device type ", sorter.device(), " and input value tensor device type ", boundaries.device());

    TORCH_CHECK(sorter.sizes() == boundaries.sizes(),    // 检查排序器张量和边界张量的尺寸是否一致
      "torch.searchsorted(): boundary and sorter must have the same size, but got boundary tensor ", boundaries.sizes(), "and got sorter tensor ", sorter.sizes());

    TORCH_CHECK(sorter.scalar_type() == ScalarType::Long,  // 检查排序器张量的数据类型是否为 long 类型
      "torch.searchsorted(): sorter must be a tensor of long dtype but got dtype ", sorter.scalar_type());
    if (sorter.numel() > 0) {
      // 检查排序器是否包含元素
      auto minmax = sorter.aminmax();  // 获取排序器中的最小和最大值
      int64_t vmin = std::get<0>(minmax).item().toLong();  // 将最小值转换为 int64_t 类型
      int64_t vmax = std::get<1>(minmax).item().toLong();  // 将最大值转换为 int64_t 类型
      // 检查最小值和最大值是否在合法范围内
      TORCH_CHECK(vmin >= 0 && vmax < sorter.sizes().back(), "torch.searchsorted(): sorter index out of range");
    }
  }

  // 检查输入张量的维度是否符合要求
  TORCH_CHECK(input.dim() > 0 || (input.dim() == 0 && input.numel() == 1 && boundaries.dim() == 1),
    "torch.searchsorted(): input value can be a scalar only when boundaries tensor dimension is 1, but we got ",
    "boundaries tensor dim(", boundaries.dim(), ") and input value's dim(", input.dim(), ") numel(",
    input.numel(), ")");

  // 检查边界张量的维度是否为正数
  TORCH_CHECK(boundaries.dim() != 0, "torch.searchsorted(): boundaries tensor should have positive dimension, but ",
    "got 0 dimension");

  // 检查边界张量的维度是否正确匹配
  TORCH_CHECK(boundaries.dim() == 1 || searchsorted_dims_matched_before_last_dim(boundaries, input),
    "torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor ",
    "and input value tensor must match, but we got boundaries tensor ", boundaries.sizes(), " and input value tensor ",
    input.sizes());

  // 获取输出张量的数据类型
  ScalarType output_dtype = output.scalar_type();
  // 检查输出张量的数据类型是否正确
  TORCH_CHECK(
      (output_dtype == ScalarType::Long && !out_int32) ||
          (output_dtype == ScalarType::Int && out_int32),
      "torch.searchsorted(): output tensor's dtype is wrong, it can only be Int(int32) or Long(int64) depending on ",
      "whether out_int32 flag is True, but we got output tensor's dtype ", output_dtype,
      " and out_int32 flag is ", (out_int32 ? "True" : "False"));

  // 如果输出类型为 int32，则检查边界张量的最后一个维度大小是否小于 INT_MAX
  if (out_int32) {
    TORCH_CHECK(boundaries.sizes().back() < INT_MAX,
      "torch.searchsorted(): the size of boundaries' last dimension should be less than ", INT_MAX, ", but we got ",
      boundaries.sizes().back());
  }
}

} // namespace at::native
```