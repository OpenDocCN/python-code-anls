# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_mask.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/cpu/vec/intrinsics.h>
// 包含 ATen 库的矢量化指令头文件

#include <ATen/cpu/vec/vec_base.h>
// 包含 ATen 库的矢量化基础类型头文件

#include <ATen/cpu/vec/vec_mask.h>
// 包含 ATen 库的矢量化掩码头文件

namespace at::vec {
inline namespace CPU_CAPABILITY {
// 进入 ATen::vec 命名空间，并定义 CPU_CAPABILITY 内联命名空间

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
// 如果定义了 CPU_CAPABILITY_AVX512 并且不是在 Microsoft Visual Studio 编译器下

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
  static inline VectorizedN<T, 1> apply(
      const T* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    // 创建一个全零的矢量
    at::vec::Vectorized<T> zero_vec(0);
    // 创建一个所有位都是 1 的矢量
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    // 将 vec_mask 转换为整数掩码
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    // 比较掩码和所有位都是 1 的矢量，生成掩码
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
    if constexpr (std::is_same_v<T, float>) {
      // 如果 T 是 float 类型，则使用掩码加载 float 数据
      return Vectorized<T>(_mm512_mask_loadu_ps(zero_vec, mmask, ptr));
    } else {
      // 否则使用掩码加载 int32_t 或 uint32_t 数据
      return Vectorized<T>(_mm512_mask_loadu_epi32(zero_vec, mmask, ptr));
    }
  }
};

template <typename data_t, typename mask_t>
struct VecMaskLoad<
    data_t,
    1,
    mask_t,
    1,
    typename std::enable_if<
        std::is_same_v<data_t, BFloat16> ||
        std::is_same_v<data_t, Half>>::type> {
  static inline VectorizedN<data_t, 1> apply(
      const data_t* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    // 创建一个所有位都是 1 的矢量
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    // 将 vec_mask 转换为整数掩码
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    // 比较掩码和所有位都是 1 的矢量，生成掩码
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
    // 创建一个全零的矢量
    auto zero = _mm256_set1_epi16(0);
    // 使用掩码加载 BFloat16 或 Half 数据
    auto temp = _mm256_mask_loadu_epi16(zero, mmask, ptr);
    // 将加载的数据插入到 512 位寄存器中
    return Vectorized<data_t>(
        _mm512_inserti32x8(_mm512_castsi256_si512(temp), zero, 1));
  }
};

template <typename data_t, typename mask_t>
struct VecMaskLoad<
    data_t,
    1,
    mask_t,
    1,
    typename std::enable_if<
        std::is_same_v<data_t, int8_t> ||
        std::is_same_v<data_t, uint8_t>>::type> {
  static inline VectorizedN<data_t, 1> apply(
      const data_t* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    // 创建一个所有位都是 1 的矢量
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    // 将 vec_mask 转换为整数掩码
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    // 比较掩码和所有位都是 1 的矢量，生成掩码
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
    // 创建一个全零的矢量
    auto zero = _mm_set1_epi8(0);
    // 使用掩码加载 int8_t 或 uint8_t 数据
    auto temp = _mm_mask_loadu_epi8(zero, mmask, ptr);
    // 将加载的数据插入到 512 位寄存器中
    return Vectorized<data_t>(
        _mm512_inserti64x2(_mm512_set1_epi32(0), temp, 0));
  }
};

template <typename mask_t>
struct VecMaskLoad<int64_t, 2, mask_t, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const int64_t* ptr,
      const VecMask<mask_t, 1>& vec_mask) {
    // 创建一个所有位都是 1 的矢量
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    // 创建一个全零的 512 位矢量
    auto zero = _mm512_set1_epi64(0);
    // 将 vec_mask 转换为整数掩码
    auto int_mask = vec_mask.template cast<int, 1>()[0];
    // 比较掩码和所有位都是 1 的矢量，生成掩码
    auto mmask = _mm512_cmp_epi32_mask(int_mask, all_ones, _MM_CMPINT_EQ);
    // 使用掩码加载两个 int64_t 数据
    at::vec::VectorizedN<int64_t, 2> result;
    result[0] = _mm512_mask_loadu_epi64(zero, (__mmask8)mmask, ptr);
    result[1] = _mm512_mask_loadu_epi64(zero, (__mmask8)(mmask >> 8), ptr + 8);
    // 使用 AVX-512 指令集的 _mm512_mask_loadu_epi64 函数加载内存中的数据到 result 数组的第二个元素
    return result;
  }
};

// 特化模板，将 VecMask<int, 1> 转换为 VecMask<float, 1>
template <>
struct VecMaskCast<float, 1, int, 1> {
  static inline VecMask<float, 1> apply(const VecMask<int, 1>& vec_mask) {
    // 使用 _mm512_castsi512_ps 将整型向量掩码转换为单精度浮点向量掩码
    return Vectorized<float>(_mm512_castsi512_ps(vec_mask[0]));
  }
};

// 特化模板，将 VecMask<float, 1> 转换为 VecMask<int, 1>
template <>
struct VecMaskCast<int, 1, float, 1> {
  static inline VecMask<int, 1> apply(const VecMask<float, 1>& vec_mask) {
    // 使用 _mm512_castps_si512 将单精度浮点向量掩码转换为整型向量掩码
    return Vectorized<int>(_mm512_castps_si512(vec_mask[0]));
  }
};

// 特化模板，将 VecMask<int64_t, 2> 转换为 VecMask<dst_t, 1>
template <typename dst_t>
struct VecMaskCast<dst_t, 1, int64_t, 2> {
  static inline VecMask<dst_t, 1> apply(const VecMask<int64_t, 2>& vec_mask) {
    // 将 int64_t 类型的双字节向量转换为 int 类型的单字节向量
    auto int_vec = convert<int, 1, int64_t, 2>(VectorizedN<int64_t, 2>(vec_mask));
    return VecMask<int, 1>(int_vec).cast<dst_t, 1>();
  }
};

// VecMask<int, 1> 类的成员方法 all_zero() 的特化实现
template <>
inline bool VecMask<int, 1>::all_zero() const {
  // 使用 _mm512_test_epi32_mask 检测向量掩码是否全为零
  __mmask16 mask = _mm512_test_epi32_mask(mask_[0], mask_[0]);
  return mask == 0;
}

// VecMask<int, 1> 类的成员方法 is_masked(int i) 的特化实现
template <>
inline bool VecMask<int, 1>::is_masked(int i) const {
  // 使用 _mm512_movepi32_mask 获取位掩码中第 i 位的掩码值
  return _mm512_movepi32_mask(mask_[0]) & (1 << i);
}

// VecMask<int, 1> 类的成员方法 all_masked() 的特化实现
template <>
inline bool VecMask<int, 1>::all_masked() const {
  // 使用 _mm512_movepi32_mask 获取整个位掩码的掩码值，并判断是否全部为 1
  __mmask16 mask = _mm512_movepi32_mask(mask_[0]);
  return mask == 0xffff;
}

// 定义带有向 int 类型的强制转换的 VEC_MASK_METHOD_WITH_CAST_TO_INT 宏
#define VEC_MASK_METHOD_WITH_CAST_TO_INT(                   \
    T, N, return_type, method, args_def, args)              \
  template <>                                               \
  inline return_type VecMask<T, N>::method args_def const { \
    // 调用 cast<int, 1>() 将当前向量掩码转换为 int 类型后，再调用指定方法
    return cast<int, 1>().method args;                      \
  }

// 以下是具体化的宏定义，用于生成带有强制转换到 int 类型的 VecMask 方法
VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, all_zero, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, all_zero, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, is_masked, (int i), (i))
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, is_masked, (int i), (i))
VEC_MASK_METHOD_WITH_CAST_TO_INT(float, 1, bool, all_masked, (), ())
VEC_MASK_METHOD_WITH_CAST_TO_INT(int64_t, 2, bool, all_masked, (), ())

// 取消宏定义，结束宏区域
#undef VEC_MASK_DEFINE_METHOD_WITH_CAST_TO_INT

// 结束 CPU_CAPABILITY 命名空间
#endif

// 结束 at::vec 命名空间
} // namespace at::vec
```