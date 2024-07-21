# `.\pytorch\aten\src\ATen\cuda\cub.h`

```
#pragma once
// 用于确保头文件只被编译一次的预处理指令

#include <cstdint>
// 包含 C++ 标准整数类型头文件

#include <c10/core/ScalarType.h>
// 包含 PyTorch 的 ScalarType 定义头文件

#include <ATen/cuda/CUDAConfig.h>
// 包含 PyTorch CUDA 配置相关头文件

// NOTE: These templates are intentionally not defined in this header,
// which aviods re-compiling them for each translation unit. If you get
// a link error, you need to add an explicit instantiation for your
// types in cub.cu
// 注意：这些模板故意没有在这个头文件中定义，这样可以避免每个翻译单元重新编译它们。
// 如果出现链接错误，您需要在 cub.cu 中为您的类型添加显式实例化。

namespace at::cuda::cub {

inline int get_num_bits(uint64_t max_key) {
  // 计算给定最大键值的比特位数
  int num_bits = 1;
  while (max_key > 1) {
    max_key >>= 1;
    num_bits++;
  }
  return num_bits;
}

namespace detail {

// radix_sort_pairs doesn't interact with value_t other than to copy
// the data, so we can save template instantiations by reinterpreting
// it as an opaque type.
// radix_sort_pairs 除了复制数据外不会与 value_t 交互，因此我们可以通过将其重新解释为不透明类型来节省模板实例化。
template <int N> struct alignas(N) OpaqueType { char data[N]; };

template<typename key_t, int value_size>
void radix_sort_pairs_impl(
    const key_t *keys_in, key_t *keys_out,
    const OpaqueType<value_size> *values_in, OpaqueType<value_size> *values_out,
    int64_t n, bool descending, int64_t begin_bit, int64_t end_bit);

}  // namespace detail

template<typename key_t, typename value_t>
void radix_sort_pairs(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int64_t n, bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8) {
  static_assert(std::is_trivially_copyable_v<value_t> ||
                AT_ROCM_ENABLED(),  // ROCm incorrectly fails this check for vector types
                "radix_sort_pairs value type must be trivially copyable");
  // 确保 value_t 类型可以被平凡复制，或者在 ROCm 平台下启用
  // ROCm 对于向量类型错误地未通过此检查。

  // Make value type opaque, so all inputs of a certain size use the same template instantiation
  // 将 value 类型设置为不透明，以便所有特定大小的输入使用相同的模板实例化
  using opaque_t = detail::OpaqueType<sizeof(value_t)>;
  static_assert(sizeof(value_t) <= 8 && (sizeof(value_t) & (sizeof(value_t) - 1)) == 0,
                "This size of value_t is not instantiated. Please instantiate it in cub.cu"
                " and modify this check.");
  // 确保 value_t 的大小已实例化，并修改此检查。

  static_assert(sizeof(value_t) == alignof(value_t), "Expected value_t to be size-aligned");
  // 确保 value_t 类型大小对齐

  detail::radix_sort_pairs_impl(
      keys_in, keys_out,
      reinterpret_cast<const opaque_t*>(values_in),
      reinterpret_cast<opaque_t*>(values_out),
      n, descending, begin_bit, end_bit);
}

template<typename key_t>
void radix_sort_keys(
    const key_t *keys_in, key_t *keys_out,
    int64_t n, bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8);

// NOTE: Intermediate sums will be truncated to input_t precision
// 注意：中间和将被截断为 input_t 精度
template <typename input_t, typename output_t>
void inclusive_sum_truncating(const input_t *input, output_t *output, int64_t n);

template <typename scalar_t>
void inclusive_sum(const scalar_t *input, scalar_t *output, int64_t n) {
  // 调用 inclusive_sum_truncating 函数，并返回结果
  return inclusive_sum_truncating(input, output, n);
}

// NOTE: Sums are done is common_type<input_t, output_t>
// 注意：求和结果为 common_type<input_t, output_t>
template <typename input_t, typename output_t>
void exclusive_sum_in_common_type(const input_t *input, output_t *output, int64_t n);

template <typename scalar_t>
// 计算输入数组的排他性求和，并将结果存储到输出数组中
void exclusive_sum(const scalar_t *input, scalar_t *output, int64_t n) {
  // 调用通用类型的排他性求和函数
  return exclusive_sum_in_common_type(input, output, n);
}

// 对布尔类型的掩码进行排他性求和，并将结果存储到输出索引数组中
void mask_exclusive_sum(const uint8_t *mask, int64_t *output_idx, int64_t n);

// 内联函数，对布尔类型的掩码进行排他性求和，并将结果存储到输出索引数组中
inline void mask_exclusive_sum(const bool *mask, int64_t *output_idx, int64_t n) {
  // 将布尔类型的掩码转换为 uint8_t 类型，然后调用对应的函数进行排他性求和
  return mask_exclusive_sum(
      reinterpret_cast<const uint8_t*>(mask), output_idx, n);
}

}  // namespace at::cuda::cub
```