# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_bfloat16_vsx.h`

```
#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]

// 定义 CPU_CAPABILITY 内联命名空间
inline namespace CPU_CAPABILITY {

// 将 BFloat16 向量转换为两个 float 向量的元组
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
    const Vectorized<BFloat16>& a) {
  constexpr int64_t K = Vectorized<BFloat16>::size();
  __at_align__ float arr[K]; // 声明长度为 K 的浮点数数组 arr
  __at_align__ BFloat16 arr2[K]; // 声明长度为 K 的 BFloat16 数组 arr2
  a.store(arr2); // 将 BFloat16 向量 a 存储到 arr2 数组中
  convert(arr2, arr, K); // 将 arr2 中的 BFloat16 转换为 float 存储到 arr 中
  return std::make_tuple(
      Vectorized<float>::loadu(arr), // 加载未对齐的 arr 到 float 向量
      Vectorized<float>::loadu(arr + Vectorized<float>::size())); // 加载未对齐的 arr 的第二部分到 float 向量
}

// 将两个 float 向量转换为 BFloat16 向量
inline Vectorized<BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  constexpr int64_t K = Vectorized<BFloat16>::size();
  __at_align__ float arr[K]; // 声明长度为 K 的浮点数数组 arr
  __at_align__ BFloat16 arr2[K]; // 声明长度为 K 的 BFloat16 数组 arr2
  a.store(arr); // 将向量 a 中的数据存储到 arr 数组中
  b.store(arr + Vectorized<float>::size()); // 将向量 b 中的数据存储到 arr 数组的第二部分
  convert(arr, arr2, K); // 将 arr 中的 float 转换为 BFloat16 存储到 arr2 中
  return Vectorized<BFloat16>::loadu(arr2); // 加载未对齐的 arr2 到 BFloat16 向量
}

// 从 BFloat16 数据中加载到 float 向量中
inline void load_fp32_from_bf16(const c10::BFloat16* data, Vectorized<float>& out) {
  __at_align__ float values[Vectorized<float>::size()]; // 声明长度为向量大小的浮点数数组 values
  for (const auto k : c10::irange(Vectorized<float>::size())) {
    values[k] = data[k]; // 从 BFloat16 数据中逐个加载到 values 数组中
  }
  out = Vectorized<float>::loadu(values); // 加载未对齐的 values 数组到 float 向量 out 中
}

// 从 BFloat16 数据中加载到两个 float 向量中
inline void load_fp32_from_bf16(
    const c10::BFloat16* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  load_fp32_from_bf16(data, out1); // 加载 BFloat16 数据到第一个 float 向量 out1 中
  data += Vectorized<float>::size(); // 移动数据指针到下一个向量大小的位置
  load_fp32_from_bf16(data, out2); // 加载 BFloat16 数据到第二个 float 向量 out2 中
}

// 从 Half 数据中加载到 float 向量中
inline void load_fp32_from_fp16(const c10::Half* data, Vectorized<float>& out) {
  __at_align__ float values[Vectorized<float>::size()]; // 声明长度为向量大小的浮点数数组 values
  for (const auto k : c10::irange(Vectorized<float>::size())) {
    values[k] = data[k]; // 从 Half 数据中逐个加载到 values 数组中
  }
  out = Vectorized<float>::loadu(values); // 加载未对齐的 values 数组到 float 向量 out 中
}

// 从 Half 数据中加载到两个 float 向量中
inline void load_fp32_from_fp16(
    const c10::Half* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  load_fp32_from_fp16(data, out1); // 加载 Half 数据到第一个 float 向量 out1 中
  data += Vectorized<float>::size(); // 移动数据指针到下一个向量大小的位置
  load_fp32_from_fp16(data, out2); // 加载 Half 数据到第二个 float 向量 out2 中
}

} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at
```