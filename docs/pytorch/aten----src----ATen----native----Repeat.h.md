# `.\pytorch\aten\src\ATen\native\Repeat.h`

```py
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <ATen/core/Tensor.h>
// 包含 ATen 核心 Tensor 类的头文件

#include <ATen/TensorOperators.h>
// 包含 ATen 张量运算相关的头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 函数的头文件
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 空张量操作相关的头文件
#endif

namespace at::native {

template <
    typename index_t,
    void compute(const index_t*, const int64_t*, index_t*, int64_t, int64_t)>
// 定义模板函数 repeat_interleave_common，接受一个 index_t 类型的模板参数和一个函数指针 compute
static inline Tensor repeat_interleave_common(
    const Tensor& repeats,
    std::optional<int64_t> output_size) {
  TORCH_CHECK(
      repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  // 检查 repeats 张量的维度是否为 1，否则抛出错误信息
  TORCH_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeats has to be Long or Int tensor");
  // 检查 repeats 张量的数据类型是否为 Long 或 Int，否则抛出错误信息
  if (repeats.size(0) == 0) {
    return at::empty_like(repeats, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  // 如果 repeats 张量的大小为 0，则返回一个与 repeats 类型和布局相同的空张量

  Tensor repeats_ = repeats.contiguous();
  // 将 repeats 张量转换为连续存储的张量
  Tensor cumsum = repeats.cumsum(0);
  // 计算 repeats 张量的累积和，结果保存在 cumsum 张量中
  int64_t total;
  if (output_size.has_value()) {
    total = output_size.value();
  } else {
    total = cumsum[-1].item<int64_t>();
    // 如果未指定 output_size，则根据 cumsum 的最后一个元素确定 total 的大小
    TORCH_CHECK(
        (repeats >= 0).all().item<uint8_t>(), "repeats can not be negative");
    // 检查 repeats 张量中的所有元素是否都为非负数，否则抛出错误信息
  }

  Tensor result = at::empty({total}, repeats.options());
  // 创建一个大小为 total 的空张量 result，使用 repeats 张量的选项（数据类型等）

  const index_t* repeat_ptr = repeats_.const_data_ptr<index_t>();
  // 获取 repeats_ 张量中数据类型为 index_t 的常量数据指针
  const int64_t* cumsum_ptr = cumsum.const_data_ptr<int64_t>();
  // 获取 cumsum 张量中数据类型为 int64_t 的常量数据指针
  index_t* result_ptr = result.data_ptr<index_t>();
  // 获取 result 张量中数据类型为 index_t 的数据指针
  compute(repeat_ptr, cumsum_ptr, result_ptr, repeats.size(0), total);
  // 调用传入的 compute 函数指针，对 repeat_ptr、cumsum_ptr 和 result_ptr 执行计算

  return result;
  // 返回计算结果的张量 result
}

} // namespace at::native
// 结束命名空间 at::native
```