# `.\pytorch\aten\src\ATen\cpu\vec\vec.h`

```
#pragma once

#if defined(CPU_CAPABILITY_AVX512)
#include <ATen/cpu/vec/vec512/vec512.h>
#else
#include <ATen/cpu/vec/vec256/vec256.h>
#endif

namespace at::vec {
// CPU_CAPABILITY 命名空间的内联命名空间，参见 Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// 将 Vectorized<int8_t> 向量转换为 Vectorized<bool> 向量的函数
inline Vectorized<bool> convert_to_bool(Vectorized<int8_t> x) {
  // 使用 x 向量的大小声明一个 bool 数组 buffer
  __at_align__ bool buffer[x.size()];
  // 将 x 向量中非零元素存储到 buffer 中
  x.ne(Vectorized<int8_t>(0)).store(buffer);

  // 创建并初始化返回的 Vectorized<bool> 向量 ret
  Vectorized<bool> ret;
  // 断言 x 向量和 ret 向量的大小相同
  static_assert(x.size() == ret.size(), "");
  // 使用 memcpy 将 buffer 中的数据拷贝到 ret 中
  std::memcpy(ret, buffer, ret.size() * sizeof(bool));
  return ret;
}

// 特化模板，用于加载布尔值的处理，参见 NOTE [Loading boolean values]
template <>
inline Vectorized<bool> Vectorized<bool>::loadu(const void* ptr) {
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr));
}

// 特化模板，用于加载布尔值的处理，参见 NOTE [Loading boolean values]
template <>
inline Vectorized<bool> Vectorized<bool>::loadu(const void* ptr, int64_t count) {
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr, count));
}

// 模板结构 VecHoldType，根据 VT 类型定义 hold_type 类型
template <typename VT>
struct VecHoldType { using hold_type = typename VT::value_type; };

// 特化 VecHoldType 模板，对 Vectorized<BFloat16> 类型的 hold_type 进行定义
template <>
struct VecHoldType<Vectorized<BFloat16>> { using hold_type = BFloat16; };

// 特化 VecHoldType 模板，对 Vectorized<Half> 类型的 hold_type 进行定义
template <>
struct VecHoldType<Vectorized<Half>> {using hold_type = Half; };

// vechold_type 模板别名，根据 VT 类型选择 hold_type 类型
template <typename VT>
using vechold_type = typename VecHoldType<VT>::hold_type;

}} // namespace at::vec::CPU_CAPABILITY
```