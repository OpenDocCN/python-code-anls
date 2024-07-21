# `.\pytorch\aten\src\ATen\native\Repeat.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/Repeat.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/repeat_interleave_native.h>
#endif

// 定义模板函数 compute_cpu，用于在 CPU 上计算重复插值的操作
template <typename index_t>
static void compute_cpu(
    const index_t* repeat_ptr, // 重复次数指针
    const int64_t* cumsum_ptr, // 累积和指针
    index_t* result_ptr,       // 结果指针
    int64_t size,              // 大小
    int64_t result_size) {     // 结果大小
  // 检查分配的大小是否与所需大小匹配
  TORCH_CHECK(
      (result_size == cumsum_ptr[size - 1]),
      "allocated size does not match required size");
  
  // 使用并行循环在多个线程上执行以下操作
  at::parallel_for(0, size, 1, [&](int64_t i_begin, int64_t i_end) {
    // 循环遍历每个元素的索引范围
    for (const auto i : c10::irange(i_begin, i_end)) {
      // 计算结束索引
      int64_t end = cumsum_ptr[i];
      // 获取重复次数
      index_t size = repeat_ptr[i];
      // 检查重复次数是否为非负数
      TORCH_CHECK((size >= 0), "repeats can not be negative");
      // 计算开始索引
      int64_t start = end - size;
      // 将当前索引的值复制到结果数组中
      for (const auto j : c10::irange(start, end)) {
        result_ptr[j] = i;
      }
    }
  });
}

namespace at::native {

// 实现 CPU 上的 repeat_interleave 操作
Tensor repeat_interleave_cpu(
    const Tensor& repeat,              // 重复张量
    std::optional<int64_t> output_size // 可选的输出大小
) {
  // 定义输出张量
  Tensor output;
  // 根据重复张量的类型分发操作
  AT_DISPATCH_INDEX_TYPES(repeat.scalar_type(), "repeat_interleave_cpu", [&]() {
    // 调用通用的 repeat_interleave 函数生成输出
    output = repeat_interleave_common<index_t, compute_cpu<index_t>>(
        repeat, output_size);
  });

  return output; // 返回输出张量
}

// 对称整数类型的 repeat_interleave 操作
Tensor repeat_interleave_symint(
    const Tensor& self,                    // 输入张量
    const Tensor& repeats,                 // 重复张量
    std::optional<int64_t> dim,            // 可选的维度
    std::optional<SymInt> output_size      // 可选的输出大小
) {
  // 备份输入张量的共轭和否定位
  Tensor input = self;
  const auto conj = input.is_conj();
  if (conj) {
    input = input.conj();
  }
  const auto neg = input.is_neg();
  if (neg) {
    input = input._neg_view();
  }

  // 如果未指定维度，则将输入张量展平
  if (!dim) {
    input = input.flatten();
    dim = 0;
  }

  // 调整重复张量的形状以匹配输入张量的维度大小
  Tensor repeats_ = repeats;
  if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.sym_size(0) == 1)) {
    repeats_ = repeats.reshape({1}).expand_symint({input.sym_size(dim.value())});
  } else if (repeats.dim() == 1) {
    // 检查重复张量与输入张量在指定维度上的大小是否一致
    TORCH_CHECK(
        repeats.sym_size(0) == input.sym_size(dim.value()),
        "repeats must have the same size as input along dim, but got repeats.size(0) = ",
        repeats.sym_size(0), " and input.size(", dim.value(), ") = ", input.sym_size(dim.value())
    );
  } else {
    AT_ERROR("repeats must be 0-dim or 1-dim tensor");
  }

  // 使用 repeat_interleave_symint 函数对输入张量执行索引选择操作
  auto ret = input.index_select(
      dim.value(), at::repeat_interleave_symint(repeats_, output_size));

  // 恢复原始的共轭和否定位
  if (conj) {
    ret = ret.conj();
  }
  if (neg) {
    ret = ret._neg_view();
  }

  return ret; // 返回处理后的张量
}

// 对称整数类型的 repeat_interleave 操作（重载）
Tensor repeat_interleave_symint(
    const Tensor& self,        // 输入张量
    c10::SymInt repeats,       // 对称整数类型的重复参数
    std::optional<int64_t> dim_opt,
    std::optional<SymInt> output_size) {
  // 如果未指定维度参数，则将输入张量展平
  Tensor input = dim_opt ? self : self.flatten();
  // 获取有效的维度索引，确保在张量维度范围内
  int64_t dim = c10::maybe_wrap_dim(dim_opt.value_or(0), self.dim());
  // 检查重复次数是否为非负数
  TORCH_CHECK(repeats >= 0, "Repeats must be non-negative");

  // 在指定维度上添加一个新的维度
  input = input.unsqueeze(dim + 1);
  // 获取扩展后的形状
  auto expand_shape = input.sym_sizes().vec();
  expand_shape[dim + 1] = repeats;
  // 根据扩展形状扩展输入张量
  input = input.expand_symint(expand_shape);

  // 对于标量重载，这个参数实际上没有意义，但是为了与张量重载保持一致而存在
  if (output_size) {
    // 计算期望的输出大小
    auto calculated_size = (repeats * expand_shape[dim]).guard_int(__FILE__, __LINE__);
    // 检查计算出的大小与提供的输出大小是否一致
    TORCH_CHECK(*output_size == calculated_size, "repeat_interleave: Invalid output_size, expected ",
                calculated_size, " but got ", *output_size);
  }

  // 返回内存格式为连续的输入张量的克隆，并在指定的维度范围内展平
  return input.clone(at::MemoryFormat::Contiguous).flatten(dim, dim + 1);
}
}

} // namespace at::native
```