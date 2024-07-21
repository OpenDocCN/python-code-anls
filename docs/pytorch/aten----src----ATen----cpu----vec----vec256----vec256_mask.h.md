# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_mask.h`

```
#pragma once
// 预处理指令，确保本头文件在编译时只包含一次

#include <ATen/cpu/vec/intrinsics.h>
// 包含 ATen 库的向量化指令头文件

#include <ATen/cpu/vec/vec_base.h>
// 包含 ATen 库的向量基础操作头文件

#include <ATen/cpu/vec/vec_mask.h>
// 包含 ATen 库的向量掩码操作头文件

namespace at::vec {
inline namespace CPU_CAPABILITY {
// 命名空间 at::vec，内嵌在 CPU_CAPABILITY 命名空间中

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
// 如果定义了 CPU_CAPABILITY_AVX2 并且未定义 _MSC_VER

template <typename T, typename mask_t>
struct VecMaskLoad<
    T,
    1,
    mask_t,
    1,
    typename std::enable_if_t<
        std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t>,
        void>> {
  // 模板结构体 VecMaskLoad 的特化，用于加载向量掩码数据

  static inline VectorizedN<T, 1> apply(
      const T* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    // 静态成员函数 apply，接受指针和掩码向量作为参数

    auto int_mask = vec_mask.template cast<int, 1>()[0];
    // 将掩码向量转换为整数类型，并获取其第一个元素

    if constexpr (std::is_same_v<T, float>) {
      // 如果 T 类型为 float
      return Vectorized<T>(_mm256_maskload_ps(ptr, int_mask));
      // 返回使用掩码加载单精度浮点数据的向量化结果
    } else {
      // 否则（T 类型为 int32_t 或 uint32_t）
      return Vectorized<T>(_mm256_maskload_epi32(ptr, int_mask));
      // 返回使用掩码加载整数数据的向量化结果
    }
  }
};

// TODO: add specialization of VecMaskLoad for bfloat16/half and int8/uint8
// TODO: 添加用于 bfloat16/half 和 int8/uint8 的 VecMaskLoad 特化实现

template <>
struct VecMaskCast<float, 1, int, 1> {
  // 模板结构体 VecMaskCast 的特化，用于向浮点数类型的转换

  static inline VecMask<float, 1> apply(const VecMask<int, 1>& vec_mask) {
    // 静态成员函数 apply，接受整数类型的掩码向量作为参数

    return Vectorized<float>(_mm256_castsi256_ps(vec_mask[0]));
    // 返回将整数掩码向量转换为单精度浮点掩码向量的向量化结果
  }
};

template <>
struct VecMaskCast<int, 1, float, 1> {
  // 模板结构体 VecMaskCast 的特化，用于向整数类型的转换

  static inline VecMask<int, 1> apply(const VecMask<float, 1>& vec_mask) {
    // 静态成员函数 apply，接受单精度浮点类型的掩码向量作为参数

    return Vectorized<int>(_mm256_castps_si256(vec_mask[0]));
    // 返回将单精度浮点掩码向量转换为整数掩码向量的向量化结果
  }
};

template <typename dst_t>
struct VecMaskCast<dst_t, 1, int64_t, 2> {
  // 模板结构体 VecMaskCast 的特化，用于不同类型之间的转换

  static inline VecMask<dst_t, 1> apply(const VecMask<int64_t, 2>& vec_mask) {
    // 静态成员函数 apply，接受双精度整数掩码向量作为参数

    auto int_vec = convert<int, 1, int64_t, 2>(VectorizedN<int64_t, 2>(vec_mask));
    // 调用 convert 函数将双精度整数掩码向量转换为整数掩码向量

    return VecMask<int, 1>(int_vec).cast<dst_t, 1>();
    // 返回将整数掩码向量转换为目标类型掩码向量的结果
  }
};

template <>
inline bool VecMask<int, 1>::all_zero() const {
  // VecMask<int, 1> 类的成员函数 all_zero 的特化实现

  return _mm256_testz_si256(mask_[0], mask_[0]);
  // 返回掩码向量的所有位是否都为零的结果
}

template <>
inline bool VecMask<int, 1>::is_masked(int i) const {
  // VecMask<int, 1> 类的成员函数 is_masked 的特化实现，接受整数参数 i

  return _mm256_movemask_ps(_mm256_castsi256_ps(mask_[0])) & (1 << i);
  // 返回掩码向量在指定位是否被置位的结果
}

template <>
inline bool VecMask<int, 1>::all_masked() const {
  // VecMask<int, 1> 类的成员函数 all_masked 的特化实现

  int mask = _mm256_movemask_ps(_mm256_castsi256_ps(mask_[0]));
  // 获取掩码向量的移动掩码结果

  return mask == 0xff;
  // 返回掩码向量的所有位是否都被置位的结果
}

#define VEC_MASK_METHOD_WITH_CAST_TO_INT(                   \
    T, N, return_type, method, args_def, args)              \
  template <>                                               \
  inline return_type VecMask<T, N>::method args_def const { \
    return cast<int, 1>().method args;                      \
  }

VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, all_zero, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, all_zero, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, is_masked, (int i), (i))
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, is_masked, (int i), (i))
VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, all_masked, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, all_masked, (), ())

#undef VEC_MASK_DEFINE_METHOD_WITH_CAST_TO_INT

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
// 命名空间结尾
```