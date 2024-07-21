# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_qint.h`

```
#pragma once
// 防止头文件被多次包含

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]
// 不要在这个头文件中定义静态数据，参见注释[不要使用 AVX 编译初始化器]

#include <ATen/cpu/vec/intrinsics.h>
// 引入 ATen 库中的 CPU 矢量化指令头文件
#include <ATen/cpu/vec/vec_base.h>
// 引入 ATen 库中的 CPU 矢量化基类头文件
#include <ATen/native/quantized/AffineQuantizerBase.h>
// 引入 ATen 库中的量化基类头文件

#include <c10/util/irange.h>
// 引入 c10 库中的整数范围头文件
#include <c10/util/qint32.h>
// 引入 c10 库中的 32 位整数量化头文件
#include <c10/util/qint8.h>
// 引入 c10 库中的 8 位有符号整数量化头文件
#include <c10/util/quint8.h>
// 引入 c10 库中的 8 位无符号整数量化头文件

#include <array>
// 引入标准库中的数组头文件
#include <cmath>
// 引入标准库中的数学函数头文件

// This file defines Vectorized<> for the quantized types.
//
// 本文件为量化类型定义了 Vectorized<>
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vectorized<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).
//
// 目前，我们简单地将这些类用作量化类型与 Vectorized<float> 之间的高效转换器，
// 通常用于带宽受限的情况，其中在全精度下进行算术运算是可接受的（例如逐元素运算）。
//
// Conversions are as follows:
//  Vectorized<qint8> -> 4x Vectorized<float>
//  Vectorized<quint8> -> 4x Vectorized<float>
//  Vectorized<qint32> -> 1x Vectorized<float>
//
// 转换如下：
//  Vectorized<qint8> -> 4 倍 Vectorized<float>
//  Vectorized<quint8> -> 4 倍 Vectorized<float>
//  Vectorized<qint32> -> 1 倍 Vectorized<float>
//
// The size of the returned float vector is specified by the special
// constexpr function float_num_vecs. The type of the value returned
// from dequantize (and expected as an argument to quantize) is
// specified by float_vec_return_type.
//
// 返回的浮点向量的大小由特殊的 constexpr 函数 float_num_vecs 指定。
// 从 dequantize 返回的值的类型（并作为 quantize 的参数期望的类型）由 float_vec_return_type 指定。
//
// When writing kernels with these vectors, it is expected that floating-
// point operations will be carried out in a loop over Vectorized<T>::float_num_vecs
// iterations.
//
// 编写使用这些向量的内核时，预期浮点操作将在 Vectorized<T>::float_num_vecs 迭代循环中执行。

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

#ifdef _MSC_VER
__declspec(align(64)) struct Vectorizedqi {
 protected:
  __m512i vals;
#else
struct Vectorizedqi {
 protected:
  __m512i vals __attribute__((aligned(64)));
#endif

 public:
  Vectorizedqi() {}
  // 默认构造函数

  Vectorizedqi(__m512i v) : vals(v) {}
  // 接受 __m512i 参数的构造函数

  operator __m512i() const {
    return vals;
  }
  // 类型转换运算符，将 Vectorizedqi 转换为 __m512i 类型
};

template <typename T>
__m512i pack_saturate_and_clamp(
    __m512i first,
    __m512i second,
    T min_val,
    T max_val);

template <>
inline __m512i pack_saturate_and_clamp<int32_t>(
    __m512i first,
    __m512i second,
    int32_t min_val,
    int32_t max_val) {
  // This function is for linkage only, will not be used
  // 该函数仅用于链接，不会被使用
  AT_ERROR("pack_saturate_and_clamp<int32_t> is not supported");
}

template <>
inline __m512i pack_saturate_and_clamp<int8_t>(
    __m512i first,
    __m512i second,
    int8_t min_val,
    int8_t max_val) {
  // Pack, saturate, and clamp the values to the range [min_val, max_val]
  // 打包、饱和并将值夹在 [min_val, max_val] 范围内
  __m512i packed_and_sat = _mm512_packs_epi16(first, second);
  return _mm512_max_epi8(
      _mm512_set1_epi8(min_val),
      _mm512_min_epi8(packed_and_sat, _mm512_set1_epi8(max_val)));
}

template <>
inline __m512i pack_saturate_and_clamp<uint8_t>(
    __m512i first,
    __m512i second,
    uint8_t min_val,
    uint8_t max_val) {
  // Pack, saturate, and clamp unsigned values to the range [min_val, max_val]
  // 打包、饱和并将无符号值夹在 [min_val, max_val] 范围内
  __m512i packed_and_sat = _mm512_packus_epi16(first, second);
  return _mm512_max_epu8(
      _mm512_set1_epi8(min_val),
      _mm512_min_epu8(packed_and_sat, _mm512_set1_epi8(max_val)));
}

template <typename T>
typename std::enable_if_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>, at::vec::Vectorized<float>>
// 根据 T 的类型选择合适的返回类型
// 定义一个内联函数，将 int8_t 或 uint8_t 类型的向量转换为 float32 向量
inline convert_int8_to_float(at::vec::Vectorized<T> src) {
  // 注意：此函数仅将输入的元素数量转换为 at::vec::Vectorized<float>.size()
  // 只处理前 16*8 位
  __m128i input_128 = _mm512_castsi512_si128(src);
  // 将 16*uint8/int8 转换为 16*int32
  __m512i input_512_extended;
  if constexpr (std::is_same_v<T, uint8_t>)
    input_512_extended = _mm512_cvtepu8_epi32(input_128);
  else
    input_512_extended = _mm512_cvtepi8_epi32(input_128);
  // 将 16*int32 转换为 16*float32
  return _mm512_cvtepi32_ps(input_512_extended);
}

// 定义一个内联函数，将 float32 向量转换为 int8_t 或 uint8_t 类型的向量
template <typename T>
typename std::enable_if_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>, at::vec::Vectorized<T>>
inline convert_float_to_int8(at::vec::Vectorized<float> src) {
  // 将 float32 转换为 int32，使用截断方式
  __m512i x_values_int32 = _mm512_cvttps_epi32(src);

  // 使用有符号饱和转换将 int32 转换为 int16
  __m512i xy_packed_v = _mm512_packs_epi32(x_values_int32, x_values_int32);

  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();

  // 使用无符号饱和转换将 int16 转换为 uint8/int8
  __m512i xyzw_clamped_v = pack_saturate_and_clamp<T>(
      xy_packed_v, xy_packed_v, min_val, max_val);
  // 创建一个掩码，用于对 xyzw_clamped_v 进行重排列
  __m512i permute_mask_v =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02,
                      0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);
  // 使用 permute_mask_v 对 xyzw_clamped_v 进行重排列
  return _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);
}

// 定义一个模板函数，使用 AVX-512 实现量化操作
template <typename T>
__FORCE_INLINE void QuantizeAvx512(
    const float* src,
    T* dst,
    int len,
    float inverse_scale,
    float zero_point,
    const float* scale_ptr,
    const float* zero_point_ptr) {
    // 省略部分代码，仅注释示例中的部分内容
    int64_t zero_point) {
  constexpr int VLEN = 16;
  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();
  const __m512i min_v = _mm512_set1_epi32(min_val);
  const __m512i max_v = _mm512_set1_epi32(max_val);
  // This is the largest int32 value < int32_max exactly representable in float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  int i = 0;
  __m512 inverse_scale_v = _mm512_set1_ps(inverse_scale);
  // clang-format off
  // 定义一个用于对 SIMD 寄存器内容重新排列的掩码
  static const __m512i shuffle_mask_v = _mm512_set_epi8(
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0x0c, 0x08, 0x04, 0x00,   // 4个字节的掩码，用于每个 32 位整数
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0x0c, 0x08, 0x04, 0x00,   // 4个字节的掩码，用于每个 32 位整数
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0x0c, 0x08, 0x04, 0x00,   // 4个字节的掩码，用于每个 32 位整数
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0xff, 0xff, 0xff, 0xff,   // 4个 0xff，未使用的位
      0x0c, 0x08, 0x04, 0x00);  // 4个字节的掩码，用于每个 32 位整数
  // clang-format on
  // 定义用于对 32 位整数进行重新排列的掩码
  __m512i permute_mask_v =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02,
                       0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);
  // 定义用于对低 8 位整数进行重新排列的掩码
  __m512i permute_mask_l8_v =
      _mm512_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                       0x00, 0x00, 0x00, 0x00, 0x0c, 0x08, 0x04, 0x00);
  // 计算对齐处理的长度
  int len_aligned = len / (VLEN * 4) * (VLEN * 4);
  for (; i < len_aligned; i += 4 * VLEN) {
    // x
    // 加载 4 个连续的 512 位 SIMD 寄存器（64 位浮点数），作为 x_vals
    __m512 x_vals = _mm512_load_ps(src + i);
    // 将 x_vals 中的每个元素乘以 inverse_scale_v 中的相应元素
    __m512 x_transformed_v = _mm512_mul_ps(x_vals, inverse_scale_v);
    // 如果浮点值大于 int32_max，将其截断为 int32_float_max_val 以避免溢出
    x_transformed_v =
        _mm512_min_ps(x_transformed_v, _mm512_set1_ps(int32_float_max_val));
    // y
    __m512 y_vals = _mm512_load_ps(src + i + VLEN);
    __m512 y_transformed_v = _mm512_mul_ps(y_vals, inverse_scale_v);
    y_transformed_v =
        _mm512_min_ps(y_transformed_v, _mm512_set1_ps(int32_float_max_val));
    // z
    __m512 z_vals = _mm512_load_ps(src + i + 2 * VLEN);
    __m512 z_transformed_v = _mm512_mul_ps(z_vals, inverse_scale_v);
    z_transformed_v =
        _mm512_min_ps(z_transformed_v, _mm512_set1_ps(int32_float_max_val));
    // w
    __m512 w_vals = _mm512_load_ps(src + i + 3 * VLEN);
    __m512 w_transformed_v = _mm512_mul_ps(w_vals, inverse_scale_v);
    w_transformed_v =
        _mm512_min_ps(w_transformed_v, _mm512_set1_ps(int32_float_max_val));

    // 将转换后的浮点数向下取整到最接近的整数
    __m512i x_rounded_v = _mm512_cvtps_epi32(x_transformed_v);
    __m512i y_rounded_v = _mm512_cvtps_epi32(y_transformed_v);
    __m512i z_rounded_v = _mm512_cvtps_epi32(z_transformed_v);
    __m512i w_rounded_v = _mm512_cvtps_epi32(w_transformed_v);

    // 将 zero_point 添加到每个向量中的元素
    x_rounded_v = _mm512_add_epi32(x_rounded_v, _mm512_set1_epi32(zero_point));
    # 将向量 y_rounded_v 中的每个元素与 zero_point 相加
    y_rounded_v = _mm512_add_epi32(y_rounded_v, _mm512_set1_epi32(zero_point));
    # 将向量 z_rounded_v 中的每个元素与 zero_point 相加
    z_rounded_v = _mm512_add_epi32(z_rounded_v, _mm512_set1_epi32(zero_point));
    # 将向量 w_rounded_v 中的每个元素与 zero_point 相加
    w_rounded_v = _mm512_add_epi32(w_rounded_v, _mm512_set1_epi32(zero_point));

    # 将 x_rounded_v 和 y_rounded_v 向量中的整数元素打包成更窄的整数类型 (__m512i 类型)
    __m512i xy_packed_v = _mm512_packs_epi32(x_rounded_v, y_rounded_v);
    # 将 z_rounded_v 和 w_rounded_v 向量中的整数元素打包成更窄的整数类型 (__m512i 类型)
    __m512i zw_packed_v = _mm512_packs_epi32(z_rounded_v, w_rounded_v);
    # 对打包后的整数进行饱和处理和范围截断，并存储在 xyzw_clamped_v 中
    __m512i xyzw_clamped_v =
        pack_saturate_and_clamp<T>(xy_packed_v, zw_packed_v, min_val, max_val);

    # 使用 permute_mask_v 对 xyzw_clamped_v 向量进行按索引排列重新排列
    xyzw_clamped_v =
        _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);
    # 将 xyzw_clamped_v 向量中的整数存储到目标地址 dst + i 处
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), xyzw_clamped_v);
  }

  // 基于 fbgemm::QuantizeAvx2 (https://github.com/pytorch/FBGEMM) 的另一种 8 轨 AVX512 版本，
  // 用于在 len 较小时利用优势
  for (; i < len / VLEN * VLEN; i += VLEN) {
    # 从 src + i 处加载 512 位的单精度浮点数向量到 x_vals
    __m512 x_vals = _mm512_load_ps(src + i);
    # 将 x_vals 中的每个元素乘以 inverse_scale_v 向量中的对应元素
    __m512 x_transformed_v = _mm512_mul_ps(x_vals, inverse_scale_v);
    # 对 x_transformed_v 中的每个元素取最小值，不能超过 int32_float_max_val
    x_transformed_v =
        _mm512_min_ps(x_transformed_v, _mm512_set1_ps(int32_float_max_val));
    # 将 x_transformed_v 中的单精度浮点数元素转换为整数并向下取整到最接近的整数
    __m512i x_rounded_v = _mm512_cvtps_epi32(x_transformed_v);
    # 将 x_rounded_v 中的每个元素与 zero_point 相加
    x_rounded_v = _mm512_add_epi32(x_rounded_v, _mm512_set1_epi32(zero_point));
    # 对 x_rounded_v 中的每个元素进行截断，使其不超过指定的最大和最小值
    __m512i x_clipped_v =
        _mm512_max_epi32(min_v, _mm512_min_epi32(max_v, x_rounded_v));

    # 使用 shuffle_mask_v 对 x_clipped_v 向量中的整数进行按位重新排列
    x_clipped_v = _mm512_shuffle_epi8(x_clipped_v, shuffle_mask_v);
    # 使用 permute_mask_l8_v 对 x_clipped_v 向量中的整数进行按索引排列重新排列
    x_clipped_v = _mm512_permutexvar_epi32(permute_mask_l8_v, x_clipped_v);
    # 将 x_clipped_v 向量中的整数存储到目标地址 dst + i 处
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + i),
        _mm512_castsi512_si128(x_clipped_v));
  }

  # 处理剩余的不足一个向量长度的元素（len % VLEN 个）
  for (; i < len; ++i) {
    # 将 src[i] 乘以 inverse_scale，并加上 zero_point 后进行最近整数舍入
    float transformed = src[i] * inverse_scale;
    
    # 不完全与矢量化代码相同的行为说明
    # 上面的矢量化代码始终在中间情况下向偶数舍入
    # （https://software.intel.com/en-us/node/523819），
    # 但 std::nearbyint 在当前舍入模式为 FE_TONEAREST 时执行相同操作。
    # 然而，在实践中，这不应该是问题，因为大多数情况下使用默认的 FE_TONEAREST 舍入模式。
    # 注意，我们无法使用 std::round 实现与矢量化代码相同的行为，
    # 因为 std::round 在中间情况下是远离零舍入。
    transformed = zero_point + std::nearbyint(transformed);
    # 将 transformed 值限制在 min_val 和 max_val 之间，并存储到 dst[i] 中
    float clipped =
        std::min(std::max(transformed, float(min_val)), float(max_val));
    dst[i] = clipped;
  }
}

// 特化模板，用于处理 c10::qint32 类型的向量化操作
template<>
struct Vectorized<c10::qint32> : public Vectorizedqi {
    // 定义 size_type 类型为 int
    using size_type = int;
    
    // 返回向量化对象的大小，固定为 16
    static constexpr size_type size() {
        return 16;
    }

    // 返回浮点数向量的数量，固定为 1
    static constexpr int float_num_vecs() {
        return 1;
    }

    // 返回整数向量的数量，固定为 1
    static constexpr int int_num_vecs() {
        return 1;
    }

    // 定义浮点数向量的返回类型为包含一个浮点数向量的 std::array
    using float_vec_return_type = std::array<Vectorized<float>, 1>;
    
    // 定义整数向量的返回类型为包含一个 c10::qint32 向量的 std::array
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 1>;
    
    // 定义 value_type 为 c10::qint32::underlying 类型
    using value_type = c10::qint32::underlying;

 public:
    // 继承父类 Vectorizedqi 的构造函数
    using Vectorizedqi::Vectorizedqi;
    
    // 默认构造函数
    Vectorized() {}

    // 使用 __m512i 类型的值构造向量化对象
    Vectorized(__m512i vals_) { vals = vals_;}

    // 广播构造函数，将 c10::qint32 值广播到向量的每个元素
    Vectorized(const c10::qint32& val) {
        value_type uw = val.val_;
        vals = _mm512_set1_epi32(uw);
    }

    // 将向量化对象的值存储到指定内存地址
    void store(void* ptr, int count = size()) const {
      if (count != size()) {
        memcpy(ptr, &vals, count * sizeof(value_type));
      } else {
        _mm512_storeu_si512((__m512i*)ptr, vals);
      }
    }

    // 从指定内存地址加载数据到向量化对象
    static Vectorized<c10::qint32> loadu(const void* ptr) {
        return Vectorized<c10::qint32>(ptr);
    }

    // 从指定内存地址加载 count 个元素到向量化对象
    static Vectorized<c10::qint32> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // 确保未初始化的内存不会改变输出值，参考 https://github.com/pytorch/pytorch/issues/32502
        // 更多细节可以查看上述链接，我们不使用 "={0}" 将数组初始化为零，因为 gcc 会将其编译为两条指令，而循环只会编译为一条指令。
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        return loadu(tmp_values);
    }

    // 对向量化对象进行反量化操作，返回浮点数向量
    float_vec_return_type dequantize(
        Vectorized<float> scale,
        Vectorized<float> zero_point,
        Vectorized<float> scale_zp_premul) const {
      __m512 float_vals = _mm512_cvtepi32_ps(vals);
      return {vec::fmadd(scale, Vectorized<float>(float_vals), scale_zp_premul)};
    }

    // 对向量化对象进行反量化操作，返回浮点数向量
    float_vec_return_type dequantize(
        Vectorized<float> scale,
        Vectorized<float> zero_point) const {
      __m512 float_vals = _mm512_cvtepi32_ps(vals);
      return {(Vectorized<float>(float_vals) - zero_point) * scale};
    }

    // 将浮点数向量量化为 c10::qint32 类型向量
    static Vectorized<c10::qint32> quantize(
        const float_vec_return_type& rhs,
        float scale,
        int32_t zero_point,
        float inverse_scale) {
      Vectorized<c10::qint32> retval;
      auto rhs_data = (__m512)rhs[0];
      at::native::quantize_vec<c10::qint32, /*precision=*/32>(
          scale, zero_point, (float*)&rhs_data, (c10::qint32*)&retval.vals, 16);
      return retval;
    }

    // 返回向量化对象和另一个向量化对象的最大值组成的向量化对象
    Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
      return _mm512_max_epi32(vals, b.vals);
    }

    // 返回向量化对象和另一个向量化对象的最小值组成的向量化对象
    Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
      return _mm512_min_epi32(vals, b.vals);
    }
    // 对输入的 zero_point 应用 ReLU 函数，返回结果
    Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const {
        return maximum(zero_point);
    }

    // 对输入的 zero_point 和 q_six 应用 ReLU6 函数，返回结果
    Vectorized<c10::qint32> relu6(
        Vectorized<c10::qint32> zero_point,
        Vectorized<c10::qint32> q_six) {
      return _mm512_min_epi32(
          _mm512_max_epi32(vals, zero_point.vals), q_six.vals);
    }

    // 执行当前对象中的值和输入向量 b 之间的宽化减法操作，返回结果
    int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
      return {_mm512_sub_epi32(vals, b)};
    }

    // 从整数向量 inp 进行重新量化操作，使用 multiplier 和 zero_point，返回结果
    static Vectorized<c10::qint32> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      // 创建一个包含 multiplier 的 AVX-512 向量
      __m512 multiplier_v = _mm512_set1_ps(multiplier);
      // 创建一个包含 zero_point 的 AVX-512 整数向量
      __m512i zero_point_v = _mm512_set1_epi32(zero_point);

      // 执行输入向量 inp[0] 的缩放乘法
      __m512 scaled = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[0]), multiplier_v);
      // 将缩放后的结果舍入为整数
      __m512i rounded = _mm512_cvtps_epi32(scaled);
      // 返回舍入整数与 zero_point_v 的加法结果
      return _mm512_add_epi32(rounded, zero_point_v);
    }

 private:
    // 从内存地址 ptr 加载数据构造函数
    Vectorized(const void* ptr) {
      // 使用 AVX-512 指令集加载未对齐的 512 位整数数据到 vals 中
      vals = _mm512_loadu_si512((const __m512i*)ptr);
    }
};

/*
 * Template specialization for finding the element-wise maximum of Vectorized<c10::qint32> vectors.
 */
template <>
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
  return a.maximum(b);
}

/*
 * Template specialization for element-wise multiplication of Vectorized<c10::qint32> vectors.
 */
template <>
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  return _mm512_mullo_epi32(a, b);
}

/*
 * Template specialization for element-wise addition of Vectorized<c10::qint32> vectors.
 */
template <>
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  return _mm512_add_epi32(a, b);
}

/*
 * Requantize AVX-512 function template for converting int32 values back to int8/uint8.
 * Takes input vectors, a multiplier, and a zero point; performs scaling, rounding,
 * adding zero point, packing to int16, saturation, and permutation of results.
 */
template <typename T>
__m512i RequantizeAvx512(
    const std::array<Vectorized<c10::qint32>, 4>& inp,
    __m512 multiplier,
    __m512i zp) {
  static_assert(
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
      "Only int8_t/uint8_t are supported");
  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();
  __m512i permute_mask_v =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02,
                       0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);
  __m512 x_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[0]), multiplier);
  __m512 y_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[1]), multiplier);
  __m512 z_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[2]), multiplier);
  __m512 w_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[3]), multiplier);

  __m512i x_rounded_v = _mm512_cvtps_epi32(x_scaled_v);
  __m512i y_rounded_v = _mm512_cvtps_epi32(y_scaled_v);
  __m512i z_rounded_v = _mm512_cvtps_epi32(z_scaled_v);
  __m512i w_rounded_v = _mm512_cvtps_epi32(w_scaled_v);

  /* Add zero point */
  __m512i x_v = _mm512_add_epi32(x_rounded_v, zp);
  __m512i y_v = _mm512_add_epi32(y_rounded_v, zp);
  __m512i z_v = _mm512_add_epi32(z_rounded_v, zp);
  __m512i w_v = _mm512_add_epi32(w_rounded_v, zp);

  /* Pack to int16_t and saturate */
  __m512i xy_packed_v = _mm512_packs_epi32(x_v, y_v);
  __m512i zw_packed_v = _mm512_packs_epi32(z_v, w_v);

  __m512i xyzw_clamped_v =
      pack_saturate_and_clamp<T>(xy_packed_v, zw_packed_v, min_val, max_val);

  /*
   * xyzw_clamped_v has results in the following layout so we need to
   * permute: x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7 x8-11 y8-11 z8-11 w8-11 x12-15 y12-15 z12-15 w12-15
   */
  xyzw_clamped_v = _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);
  return xyzw_clamped_v;
}

/*
 * Template specialization for Vectorized<c10::qint8> structure.
 * Defines constants and types related to Vectorized<c10::qint8> operations.
 */
template<>
struct Vectorized<c10::qint8> : public Vectorizedqi {
    static constexpr int size() {
        return 64;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }

    static constexpr int int_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vectorized<float>, 4>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;
    using value_type = typename c10::qint8::underlying;

 public:
    using Vectorizedqi::Vectorizedqi;

    Vectorized() {}
    Vectorized(__m512i vals_) { vals = vals_;}
    // Broadcast constructor
    // 广播构造函数，使用给定的 qint8 值初始化 simd 寄存器
    Vectorized(const c10::qint8& val) {
        // 将 qint8 值扩展到 64 字节 simd 寄存器中的每个字节
        value_type uw = val.val_;
        vals = _mm512_set1_epi8(uw);
    }

    // This is needed because the compiler emits awful code for the default
    // constructor for moving the enum
    // 为了避免编译器对移动枚举的默认构造函数生成低效的代码，需要显式定义复制构造函数
    Vectorized(const Vectorized<c10::qint8>& other) : Vectorizedqi(other.vals) { }

    // This is added to avoid error: definition of implicit copy assignment operator
    // for 'Vectorized<c10::qint8>' is deprecated because it has a user-declared
    // copy constructor [-Werror,-Wdeprecated-copy]
    // 添加此函数以避免错误：因为有用户声明的复制构造函数，为 'Vectorized<c10::qint8>' 定义隐式复制赋值运算符已过时
    Vectorized& operator=(const Vectorized<c10::qint8>&) = default;

    // 将 simd 寄存器中的值存储到内存地址中
    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            // 如果存储的数量不等于 simd 寄存器的大小，则使用 memcpy 进行复制
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            // 否则直接存储 simd 寄存器中的值到 ptr 所指向的地址
            _mm512_storeu_si512((__m512i*)ptr, vals);
        }
    }

    // 从给定的内存地址加载数据到 simd 寄存器中
    static Vectorized<c10::qint8> loadu(const void* ptr) {
        return Vectorized<c10::qint8>(ptr);
    }

    // 从给定的内存地址加载指定数量的数据到 simd 寄存器中
    static Vectorized<c10::qint8> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // 初始化临时数组，避免未初始化内存影响输出值，参见 https://github.com/pytorch/pytorch/issues/32502
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        // 使用 memcpy 将 ptr 指向的数据拷贝到临时数组中
        std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        // 返回加载后的 simd 寄存器对象
        return loadu(tmp_values);
    }

 private:
    // 将 8 位整数类型转换为 32 位整数类型的 simd 寄存器
    __m512i cvtepi8_epi32(__m128i epi8_vals) const {
        return _mm512_cvtepi8_epi32(epi8_vals);
    }

 public:
  // 反量化函数，将 simd 寄存器中的 qint8 值反量化为浮点数值
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_neg_zp_premul) const {
    #if defined(_MSC_VER) && !defined(__clang__)
    // 使用 _mm_set_epi64x 将 simd 寄存器中的值设置为 64 位整数
    __m128i int_val0 = _mm_set_epi64x(vals.m512i_u64[1], vals.m512i_u64[0]);
    __m128i int_val1 = _mm_set_epi64x(vals.m512i_u64[3], vals.m512i_u64[2]);
    __m128i int_val2 = _mm_set_epi64x(vals.m512i_u64[5], vals.m512i_u64[4]);
    __m128i int_val3 = _mm_set_epi64x(vals.m512i_u64[7], vals.m512i_u64[6]);
    #else
    // 使用 _mm_set_epi64x 将 simd 寄存器中的值设置为 64 位整数
    __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
    __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
    __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
    __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);
    #endif

    // 将 32 位整数 simd 寄存器转换为浮点数 simd 寄存器
    __m512 float_val0 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val0));
    __m512 float_val1 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val1));
    __m512 float_val2 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val2));
    __m512 float_val3 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val3));

    // 使用向量化指令进行浮点数乘加运算
    auto val0 =
        vec::fmadd(scale, Vectorized<float>(float_val0), scale_neg_zp_premul);
    auto val1 =
        vec::fmadd(scale, Vectorized<float>(float_val1), scale_neg_zp_premul);
    auto val2 =
        vec::fmadd(scale, Vectorized<float>(float_val2), scale_neg_zp_premul);
    auto val3 =
        vec::fmadd(scale, Vectorized<float>(float_val3), scale_neg_zp_premul);
    return {val0, val1, val2, val3};
  }

  // 将量化的整数向量反量化为浮点数向量
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    #if defined(_MSC_VER) && !defined(__clang__)
    // 将 AVX512 整数向量的部分元素打包成 __m128i 类型
    __m128i int_val0 = _mm_set_epi64x(vals.m512i_u64[1], vals.m512i_u64[0]);
    __m128i int_val1 = _mm_set_epi64x(vals.m512i_u64[3], vals.m512i_u64[2]);
    __m128i int_val2 = _mm_set_epi64x(vals.m512i_u64[5], vals.m512i_u64[4]);
    __m128i int_val3 = _mm_set_epi64x(vals.m512i_u64[7], vals.m512i_u64[6]);
    #else
    // 将 AVX512 整数向量的部分元素打包成 __m128i 类型
    __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
    __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
    __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
    __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);
    #endif

    // 将 __m128i 整数向量转换为 __m512 浮点数向量
    __m512 float_val0 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val0));
    __m512 float_val1 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val1));
    __m512 float_val2 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val2));
    __m512 float_val3 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val3));

    // 执行反量化操作：(整数值 - 零点值) * 缩放比例
    auto val0 = (Vectorized<float>(float_val0) - zero_point) * scale;
    auto val1 = (Vectorized<float>(float_val1) - zero_point) * scale;
    auto val2 = (Vectorized<float>(float_val2) - zero_point) * scale;
    auto val3 = (Vectorized<float>(float_val3) - zero_point) * scale;
    return {val0, val1, val2, val3};
  }

  // 对给定的浮点数向量执行量化操作，返回量化后的 AVX512 整数向量
  static Vectorized<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto* rhs_data = (float*)rhs.data();
    int8_t quantized_values[64];
    // 调用 AVX512 加速的量化函数 QuantizeAvx512，将浮点数向量量化为 int8_t 类型
    QuantizeAvx512<value_type>(
        rhs_data, quantized_values, 64, inverse_scale, zero_point);
    return Vectorized<c10::qint8>::loadu(quantized_values);
  }

  // 返回当前 AVX512 整数向量和另一个向量 b 中对应位置元素的最大值
  Vectorized<c10::qint8> maximum(Vectorized<c10::qint8> b) const {
      return _mm512_max_epi8(vals, b.vals);
    }

  // 返回当前 AVX512 整数向量和另一个向量 b 中对应位置元素的最小值
  Vectorized<c10::qint8> minimum(Vectorized<c10::qint8> b) const {
      return _mm512_min_epi8(vals, b.vals);
    }

    // 对当前 AVX512 整数向量执行 ReLU 操作，返回每个位置上与零点最大值的向量
    Vectorized<c10::qint8> relu(Vectorized<c10::qint8> zero_point) const {
        return maximum(zero_point);
    }

    // 对当前 AVX512 整数向量执行 ReLU6 操作，返回每个位置上与零点和 6 的最小值的向量
    Vectorized<c10::qint8> relu6(
        Vectorized<c10::qint8> zero_point,
        Vectorized<c10::qint8> q_six) {
      return _mm512_min_epi8(
          _mm512_max_epi8(vals, zero_point.vals), q_six.vals);
    }
    // 定义一个名为 widening_subtract 的成员函数，返回类型为 int_vec_return_type，接受一个 Vectorized<c10::qint8> 类型的参数 b
    int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
      // 如果编译器是 MSC 并且不是 Clang，则使用 _mm_set_epi64x 函数设置四个 __m128i 类型的变量，从 vals.m512i_u64 数组中获取值
      __m128i int_val0 = _mm_set_epi64x(vals.m512i_u64[1], vals.m512i_u64[0]);
      __m128i int_val1 = _mm_set_epi64x(vals.m512i_u64[3], vals.m512i_u64[2]);
      __m128i int_val2 = _mm_set_epi64x(vals.m512i_u64[5], vals.m512i_u64[4]);
      __m128i int_val3 = _mm_set_epi64x(vals.m512i_u64[7], vals.m512i_u64[6]);
      // 否则，使用 vals 数组中的值设置四个 __m128i 类型的变量
      else
      __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
      __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
      __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
      __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);

      // 将每个 __m128i 类型变量转换为 __m512i 类型，使用 cvtepi8_epi32 函数
      __m512i int32_val0 = cvtepi8_epi32(int_val0);
      __m512i int32_val1 = cvtepi8_epi32(int_val1);
      __m512i int32_val2 = cvtepi8_epi32(int_val2);
      __m512i int32_val3 = cvtepi8_epi32(int_val3);

      // 如果编译器是 MSC 并且不是 Clang，则使用 _mm_set_epi64x 函数设置四个 __m128i 类型的变量，从 b.vals.m512i_u64 数组中获取值
      __m128i int_b0 = _mm_set_epi64x(b.vals.m512i_u64[1], b.vals.m512i_u64[0]);
      __m128i int_b1 = _mm_set_epi64x(b.vals.m512i_u64[3], b.vals.m512i_u64[2]);
      __m128i int_b2 = _mm_set_epi64x(b.vals.m512i_u64[5], b.vals.m512i_u64[4]);
      __m128i int_b3 = _mm_set_epi64x(b.vals.m512i_u64[7], b.vals.m512i_u64[6]);
      // 否则，使用 b.vals 数组中的值设置四个 __m128i 类型的变量
      else
      __m128i int_b0 = _mm_set_epi64x(b.vals[1], b.vals[0]);
      __m128i int_b1 = _mm_set_epi64x(b.vals[3], b.vals[2]);
      __m128i int_b2 = _mm_set_epi64x(b.vals[5], b.vals[4]);
      __m128i int_b3 = _mm_set_epi64x(b.vals[7], b.vals[6]);

      // 将每个 __m128i 类型变量转换为 __m512i 类型，使用 cvtepi8_epi32 函数
      __m512i int32_b0 = cvtepi8_epi32(int_b0);
      __m512i int32_b1 = cvtepi8_epi32(int_b1);
      __m512i int32_b2 = cvtepi8_epi32(int_b2);
      __m512i int32_b3 = cvtepi8_epi32(int_b3);

      // 使用 _mm512_sub_epi32 函数进行四个 __m512i 类型变量的减法操作
      __m512i res_0 = _mm512_sub_epi32(int32_val0, int32_b0);
      __m512i res_1 = _mm512_sub_epi32(int32_val1, int32_b1);
      __m512i res_2 = _mm512_sub_epi32(int32_val2, int32_b2);
      __m512i res_3 = _mm512_sub_epi32(int32_val3, int32_b3);

      // 返回一个初始化列表，包含四个 Vectorized<c10::qint32> 类型的对象，每个对象使用对应的 __m512i 类型变量进行初始化
      return {Vectorized<c10::qint32>(res_0),
              Vectorized<c10::qint32>(res_1),
              Vectorized<c10::qint32>(res_2),
              Vectorized<c10::qint32>(res_3)};
    }

    // 定义一个静态成员函数 requantize_from_int，返回类型为 Vectorized<c10::qint8>
    static Vectorized<c10::qint8> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      // 使用 _mm512_set1_ps 函数根据 multiplier 设置一个 __m512 类型变量 multiplier_v
      __m512 multiplier_v = _mm512_set1_ps(multiplier);
      // 使用 _mm512_set1_epi32 函数根据 zero_point 设置一个 __m512i 类型变量 zero_point_v
      __m512i zero_point_v = _mm512_set1_epi32(zero_point);
      // 调用 RequantizeAvx512<value_type> 函数，将 inp、multiplier_v 和 zero_point_v 作为参数返回结果
      return RequantizeAvx512<value_type>(inp, multiplier_v, zero_point_v);
    }

 private:
    // 定义一个私有构造函数 Vectorized，接受一个 const void* 类型的指针作为参数
    // 该构造函数用于从内存加载数据到 vals 变量中
    Vectorized(const void* ptr) {
        vals = _mm512_loadu_si512((const __m512i*)ptr);
    }
    # 结构体 Vectorized<c10::quint8> 的结束

template <>
// 特化模板，实现 Vectorized 类型为 c10::qint8 的 maximum 函数
Vectorized<c10::qint8> inline maximum(const Vectorized<c10::qint8>& a, const Vectorized<c10::qint8>& b) {
  return a.maximum(b);
}

template<>
// Vectorized<c10::quint8> 的特化模板定义，继承自 Vectorizedqi
struct Vectorized<c10::quint8> : public Vectorizedqi {
    // 返回向量大小为 64
    static constexpr int size() {
        return 64;
    }

    // 返回浮点向量数为 4
    static constexpr int float_num_vecs() {
        return 4;
    }

    // 返回整数向量数为 4
    static constexpr int int_num_vecs() {
        return 4;
    }

    // 定义类型别名，返回浮点向量和整数向量的数组
    using float_vec_return_type = std::array<Vectorized<float>, 4>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;
    using value_type = typename c10::quint8::underlying;

 public:
    // 使用基类的构造函数
    using Vectorizedqi::Vectorizedqi;
    // 默认构造函数
    Vectorized() {}

    // 基于 __m512i 值的构造函数
    Vectorized(__m512i vals_) { vals = vals_;}

    // 广播构造函数，将 c10::quint8 类型的值广播到向量中
    Vectorized(const c10::quint8& val) {
        value_type uw = val.val_;
        vals = _mm512_set1_epi8(uw);
    }

    // 复制构造函数，使用其他 Vectorized<c10::quint8> 对象进行初始化
    Vectorized(const Vectorized<c10::quint8>& other) : Vectorizedqi(other.vals) { }

    // 显式声明赋值运算符，避免编译器生成隐式的拷贝赋值操作符
    Vectorized& operator=(const Vectorized<c10::quint8>&) = default;

    // 将向量数据存储到内存中
    void store(void* ptr, int count = size()) const {
        // 如果存储的元素个数不等于向量大小，使用 memcpy 进行存储
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            // 否则使用 _mm512_storeu_si512 进行存储
            _mm512_storeu_si512((__m512i*)ptr, vals);
        }
    }

    // 从内存中加载未对齐的数据
    static Vectorized<c10::quint8> loadu(const void* ptr) {
        return Vectorized<c10::quint8>(ptr);
    }

    // 从内存中加载未对齐的数据，指定加载的元素个数
    static Vectorized<c10::quint8> loadu(const void* ptr, int64_t count) {
        // 创建临时数组，防止未初始化的内存改变输出值
        __at_align__ value_type tmp_values[size()];
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        // 使用 memcpy 加载数据到临时数组
        std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        // 调用 loadu 函数加载临时数组中的数据
        return loadu(tmp_values);
    }

 private:
    // 将 __m128i 类型的 epu8_vals 转换为 __m512i 类型
    __m512i cvtepu8_epi32(__m128i epu8_vals) const {
        return _mm512_cvtepu8_epi32(epu8_vals);
    }

 public:
  // 解量化函数，将向量数据转换为浮点向量类型
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    #if defined(_MSC_VER) && !defined(__clang__)
    // 使用 _mm_set_epi64x 创建 __m128i 类型的整数向量
    __m128i int_val0 = _mm_set_epi64x(vals.m512i_u64[1], vals.m512i_u64[0]);
    __m128i int_val1 = _mm_set_epi64x(vals.m512i_u64[3], vals.m512i_u64[2]);
    __m128i int_val2 = _mm_set_epi64x(vals.m512i_u64[5], vals.m512i_u64[4]);
    __m128i int_val3 = _mm_set_epi64x(vals.m512i_u64[7], vals.m512i_u64[6]);
    #else
    // 使用 _mm_set_epi64x 创建 __m128i 类型的整数向量
    __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
    // 设置一个包含两个64位整数的128位整数变量 int_val1
    __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
    // 设置一个包含两个64位整数的128位整数变量 int_val2
    __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
    // 设置一个包含两个64位整数的128位整数变量 int_val3
    __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);
    #endif

    // 将 int_val0 转换为浮点数向量 float_val0
    __m512 float_val0 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val0));
    // 将 int_val1 转换为浮点数向量 float_val1
    __m512 float_val1 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val1));
    // 将 int_val2 转换为浮点数向量 float_val2
    __m512 float_val2 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val2));
    // 将 int_val3 转换为浮点数向量 float_val3
    __m512 float_val3 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val3));

    // 使用向量化函数进行乘加操作，计算 val0
    auto val0 =
        vec::fmadd(scale, Vectorized<float>(float_val0), scale_zp_premul);
    // 使用向量化函数进行乘加操作，计算 val1
    auto val1 =
        vec::fmadd(scale, Vectorized<float>(float_val1), scale_zp_premul);
    // 使用向量化函数进行乘加操作，计算 val2
    auto val2 =
        vec::fmadd(scale, Vectorized<float>(float_val2), scale_zp_premul);
    // 使用向量化函数进行乘加操作，计算 val3
    auto val3 =
        vec::fmadd(scale, Vectorized<float>(float_val3), scale_zp_premul);

    // 返回四个结果向量的元组
    return {val0, val1, val2, val3};
  }

  // 对量化后的浮点数向量进行反量化操作
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    // 根据条件设置不同的128位整数变量 int_val0
    #if defined(_MSC_VER) && !defined(__clang__)
    __m128i int_val0 = _mm_set_epi64x(vals.m512i_u64[1], vals.m512i_u64[0]);
    __m128i int_val1 = _mm_set_epi64x(vals.m512i_u64[3], vals.m512i_u64[2]);
    __m128i int_val2 = _mm_set_epi64x(vals.m512i_u64[5], vals.m512i_u64[4]);
    __m128i int_val3 = _mm_set_epi64x(vals.m512i_u64[7], vals.m512i_u64[6]);
    #else
    __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
    __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
    __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
    __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);
    #endif

    // 将 int_val0 转换为浮点数向量 float_val0
    __m512 float_val0 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val0));
    // 将 int_val1 转换为浮点数向量 float_val1
    __m512 float_val1 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val1));
    // 将 int_val2 转换为浮点数向量 float_val2
    __m512 float_val2 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val2));
    // 将 int_val3 转换为浮点数向量 float_val3
    __m512 float_val3 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val3));

    // 执行浮点数向量的反量化计算，减去零点并乘以比例因子
    auto val0 = (Vectorized<float>(float_val0) - zero_point) * scale;
    auto val1 = (Vectorized<float>(float_val1) - zero_point) * scale;
    auto val2 = (Vectorized<float>(float_val2) - zero_point) * scale;
    auto val3 = (Vectorized<float>(float_val3) - zero_point) * scale;

    // 返回四个结果向量的元组
    return {val0, val1, val2, val3};
  }

  // 对浮点数向量进行量化
  static Vectorized<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    // 获取 rhs 的数据指针
    auto* rhs_data = (float*)rhs.data();
    // 初始化用于存储量化结果的数组
    uint8_t quantized_values[64];
    // 使用 AVX512 实现的量化函数 QuantizeAvx512 进行量化
    QuantizeAvx512<value_type>(
        rhs_data, quantized_values, 64, inverse_scale, zero_point);
    // 从量化结果数组中加载成向量化整数类型数据并返回
    return Vectorized<c10::quint8>::loadu(quantized_values);
  }

  // 返回两个向量的逐元素最大值
  Vectorized<c10::quint8> maximum(Vectorized<c10::quint8> b) const {
      return _mm512_max_epu8(vals, b.vals);
    }

  // 返回两个向量的逐元素最小值
  Vectorized<c10::quint8> minimum(Vectorized<c10::quint8> b) const {
      return _mm512_min_epu8(vals, b.vals);
    }

    // 执行 ReLU 操作，返回向量中每个元素与零点元素的最大值
    Vectorized<c10::quint8> relu(Vectorized<c10::quint8> zero_point) const {
        return maximum(zero_point);
    }
    // 定义一个函数 relu6，返回处理过的 Vectorized<c10::quint8> 类型
    Vectorized<c10::quint8> relu6(
        // 输入参数：最小值和六的量化表示
        Vectorized<c10::quint8> zero_point,
        Vectorized<c10::quint8> q_six) {
      // 返回执行 _mm512_min_epu8 和 _mm512_max_epu8 后的结果
      return _mm512_min_epu8(
          _mm512_max_epu8(vals, zero_point.vals), q_six.vals);
    }

    // 定义一个函数 widening_subtract，返回 int_vec_return_type 类型
    int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
      // 根据编译器定义，设置 int_val0 到 int_val3 为 vals 的特定元素
      #if defined(_MSC_VER) && !defined(__clang__)
      __m128i int_val0 = _mm_set_epi64x(vals.m512i_u64[1], vals.m512i_u64[0]);
      __m128i int_val1 = _mm_set_epi64x(vals.m512i_u64[3], vals.m512i_u64[2]);
      __m128i int_val2 = _mm_set_epi64x(vals.m512i_u64[5], vals.m512i_u64[4]);
      __m128i int_val3 = _mm_set_epi64x(vals.m512i_u64[7], vals.m512i_u64[6]);
      #else
      // 否则，设置 int_val0 到 int_val3 为 vals 的特定元素
      __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
      __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
      __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
      __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);
      #endif

      // 将 int_val0 到 int_val3 转换为 32 位整数向量
      __m512i int32_val0 = cvtepu8_epi32(int_val0);
      __m512i int32_val1 = cvtepu8_epi32(int_val1);
      __m512i int32_val2 = cvtepu8_epi32(int_val2);
      __m512i int32_val3 = cvtepu8_epi32(int_val3);

      // 根据编译器定义，设置 int_b0 到 int_b3 为 b.vals 的特定元素
      #if defined(_MSC_VER) && !defined(__clang__)
      __m128i int_b0 = _mm_set_epi64x(b.vals.m512i_u64[1], b.vals.m512i_u64[0]);
      __m128i int_b1 = _mm_set_epi64x(b.vals.m512i_u64[3], b.vals.m512i_u64[2]);
      __m128i int_b2 = _mm_set_epi64x(b.vals.m512i_u64[5], b.vals.m512i_u64[4]);
      __m128i int_b3 = _mm_set_epi64x(b.vals.m512i_u64[7], b.vals.m512i_u64[6]);
      #else
      // 否则，设置 int_b0 到 int_b3 为 b.vals 的特定元素
      __m128i int_b0 = _mm_set_epi64x(b.vals[1], b.vals[0]);
      __m128i int_b1 = _mm_set_epi64x(b.vals[3], b.vals[2]);
      __m128i int_b2 = _mm_set_epi64x(b.vals[5], b.vals[4]);
      __m128i int_b3 = _mm_set_epi64x(b.vals[7], b.vals[6]);
      #endif

      // 将 int_b0 到 int_b3 转换为 32 位整数向量
      __m512i int32_b0 = cvtepu8_epi32(int_b0);
      __m512i int32_b1 = cvtepu8_epi32(int_b1);
      __m512i int32_b2 = cvtepu8_epi32(int_b2);
      __m512i int32_b3 = cvtepu8_epi32(int_b3);

      // 分别计算 res_0 到 res_3 为 int32_val0 到 int32_val3 减去 int32_b0 到 int32_b3 的结果
      __m512i res_0 = _mm512_sub_epi32(int32_val0, int32_b0);
      __m512i res_1 = _mm512_sub_epi32(int32_val1, int32_b1);
      __m512i res_2 = _mm512_sub_epi32(int32_val2, int32_b2);
      __m512i res_3 = _mm512_sub_epi32(int32_val3, int32_b3);

      // 返回一个包含 res_0 到 res_3 的 int_vec_return_type 类型对象
      return {Vectorized<c10::qint32>(res_0),
              Vectorized<c10::qint32>(res_1),
              Vectorized<c10::qint32>(res_2),
              Vectorized<c10::qint32>(res_3)};
    }

    // 定义一个静态函数 requantize_from_int，返回 Vectorized<c10::quint8> 类型
    static Vectorized<c10::quint8> requantize_from_int(
        // 输入参数：输入类型 inp、乘数 multiplier、零点 zero_point
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      // 将 multiplier 转换为 __m512 类型
      __m512 multiplier_v = _mm512_set1_ps(multiplier);
      // 将 zero_point 转换为 __m512i 类型
      __m512i zero_point_v = _mm512_set1_epi32(zero_point);
      // 调用 RequantizeAvx512<value_type> 并返回结果
      return RequantizeAvx512<value_type>(inp, multiplier_v, zero_point_v);
    }

    // 私有成员函数：从内存加载构造函数
    Vectorized(const void* ptr) {
        // 将 ptr 所指向的数据加载到 vals 中
        vals = _mm512_loadu_si512((const __m512i*)ptr);
    }
// NOTE: These are low-performance implementations that we fall back on.

template <
    typename T,
    typename float_vec_return_type_,
    typename int_vec_return_type_,
    int size_>
struct VectorizedQuantizedConverter {
  // 返回模板参数中指定的向量大小
  static constexpr int size() {
    return size_;
  }

  // 返回用于浮点数向量的向量数目
  static constexpr int float_num_vecs() {
    return size() / 8;
  }

  // 返回用于整数向量的向量数目
  static constexpr int int_num_vecs() {
    return size() / 8;
  }

  // 定义浮点数向量返回类型
  using float_vec_return_type = float_vec_return_type_;

  // 定义整数向量返回类型
  using int_vec_return_type = int_vec_return_type_;

  // 定义值类型为 T 的底层类型
  using value_type = typename T::underlying;

  // 值数组，大小为 size_
  std::array<value_type, size_> vals;

  // 通过给定的 T 类型对象初始化值数组
  VectorizedQuantizedConverter(T val) {
    for (const auto i : c10::irange(size())) {
      vals[i] = val.val_;
    }
  }

  // 通过指针初始化值数组
  VectorizedQuantizedConverter(const void* ptr) {
    memcpy(vals.data(), ptr, sizeof(value_type) * size());
  }

  // 将值数组的数据存储到指定的内存指针中
  void store(void* ptr, int count = size()) const {
    memcpy(ptr, vals.data(), count * sizeof(value_type));
  }

  // 解量化函数，使用给定的缩放因子、零点和缩放乘积解量化数据为浮点数向量
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    float_vec_return_type rv;
    for (const auto i : c10::irange(float_num_vecs())) {
      float tmp_vals[16];
      for (const auto j : c10::irange(16)) {
        tmp_vals[j] = at::native::dequantize_val<T>(
            scale[j], zero_point[j], T(vals[16 * i + j]));
      }
      // 将解量化后的临时值存入返回的浮点数向量中
      rv[i] = Vectorized<float>(tmp_vals[0],
          tmp_vals[1],
          tmp_vals[2],
          tmp_vals[3],
          tmp_vals[4],
          tmp_vals[5],
          tmp_vals[6],
          tmp_vals[7],
          tmp_vals[8],
          tmp_vals[9],
          tmp_vals[10],
          tmp_vals[11],
          tmp_vals[12],
          tmp_vals[13],
          tmp_vals[14],
          tmp_vals[15]);
    }
    return rv;
  }

  // 解量化函数的重载，简化调用，使用给定的缩放因子和零点解量化数据为浮点数向量
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    Vectorized<float> scale_zp_premul;
    return dequantize(scale, zero_point, scale_zp_premul);
  }

 protected:
  // 默认构造函数，保护访问
  VectorizedQuantizedConverter() {}
};
// 定义一个模板结构体 Vectorized，处理 c10::qint32 类型的向量化操作，继承自 VectorizedQuantizedConverter
// 使用了以下模板参数:
// - c10::qint32：被量化的数据类型
// - std::array<Vectorized<float>, 1>：包含一个 float 向量的数组
// - std::array<Vectorized<c10::qint32>, 1>：包含一个 c10::qint32 向量的数组
// - 16：向量的大小

struct Vectorized<c10::qint32> : public VectorizedQuantizedConverter<
                                 c10::qint32,
                                 std::array<Vectorized<float>, 1>,
                                 std::array<Vectorized<c10::qint32>, 1>,
                                 16> {
  // 默认构造函数，调用基类 VectorizedQuantizedConverter 的默认构造函数
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            16>() {}

  // 使用给定的 c10::qint32 值构造函数，调用基类的相应构造函数
  Vectorized(c10::qint32 val)
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            16>(val) {}

  // 使用指针构造函数，调用基类的相应构造函数
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            16>(ptr) {}

  // 静态方法，从给定的指针加载数据到 Vectorized<c10::qint32> 对象中
  static Vectorized<c10::qint32> loadu(const void* ptr) {
    return Vectorized<c10::qint32>(ptr);
  }

  // 静态方法，从给定的指针加载指定数量的数据到 Vectorized<c10::qint32> 对象中
  static Vectorized<c10::qint32> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];

    // 确保未初始化的内存不会改变输出值。参见 https://github.com/pytorch/pytorch/issues/32502
    // 由于 gcc 编译器在数组初始化为零时会生成两条指令，因此这里使用循环来实现初始化。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }

    // 将 ptr 指向的数据复制到临时数组 tmp_values 中
    std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));

    // 调用 loadu 方法加载临时数组数据到 Vectorized<c10::qint32> 对象中
    return loadu(tmp_values);
  }

  // 静态方法，将给定的浮点向量 rhs 进行量化，并返回 Vectorized<c10::qint32> 对象
  static Vectorized<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 16> float_vals;

    // 遍历 float_vec_return_type（std::array<Vectorized<float>, 1>）中的每个元素，将其存储到 float_vals 中
    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 16], 16);
    }

    // 调用 PyTorch 中的量化函数 quantize_vec，将 float_vals 中的浮点数量化为 c10::qint32 类型，
    // 结果存储在 qvals 中
    at::native::quantize_vec<c10::qint32, /*precision=*/32>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint32*)qvals.data(),
        16 * float_num_vecs());

    // 调用 loadu 方法加载量化结果 qvals 到 Vectorized<c10::qint32> 对象中
    return Vectorized<c10::qint32>::loadu(qvals.data());
  }

  // 成员方法，返回当前对象与另一个 Vectorized<c10::qint32> 对象 b 中的每个元素的最大值构成的新对象
  Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 成员方法，返回当前对象与另一个 Vectorized<c10::qint32> 对象 b 中的每个元素的最小值构成的新对象
  Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 成员方法，对当前对象执行 relu 操作，返回结果向量中每个元素与 zero_point 中相应元素的 relu 值
  Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const  {
  // 返回给定向量中的最大值
  return maximum(zero_point);
}


Vectorized<c10::qint32> relu6(
    Vectorized<c10::qint32> zero_point,
    Vectorized<c10::qint32> q_six) {
  Vectorized<c10::qint32> retval;
  // 遍历当前向量中的每个元素
  for (const auto i : c10::irange(size())) {
    // 计算 relu6 激活函数的结果，确保值介于 zero_point 和 q_six 之间
    retval.vals[i] = std::min<value_type>(
        std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
  }
  // 返回 relu6 函数的结果向量
  return retval;
}

int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
  int_vec_return_type retval;
  // 遍历当前向量中的每个元素
  for (const auto i : c10::irange(size())) {
    // 计算向量中每个元素与 b 向量对应元素的差
    retval[0].vals[i] = vals[i] - b.vals[i];
  }
  // 返回宽度减法的结果
  return retval;
}

static Vectorized<c10::qint32> requantize_from_int(
    const int_vec_return_type& inp,
    float multiplier,
    int32_t zero_point) {
  Vectorized<c10::qint32> retval;
  // 遍历输入向量中的每个元素
  for (const auto i : c10::irange(size())) {
    // 执行从整数重新量化为 qint32 类型的转换
    // 近似计算浮点乘法结果并加上零点偏移量
    retval.vals[i] =
        std::nearbyint(static_cast<float>(inp[0].vals[i]) * multiplier) +
        zero_point;
  }
  // 返回重新量化的结果向量
  return retval;
}
};

// 特化模板：计算两个 Vectorized<c10::qint32> 对象的最大值，并返回结果
template <>
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
  return a.maximum(b);
}

// 特化模板：计算两个 Vectorized<c10::qint32> 对象的逐元素乘积，并返回结果
template <>
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  Vectorized<c10::qint32> retval;
  // 遍历向量的每个元素，计算乘积
  for (const auto i : c10::irange(std::decay_t<decltype(a)>::size())) {
    retval.vals[i] = a.vals[i] * b.vals[i];
  }
  return retval;
}

// 特化模板：计算两个 Vectorized<c10::qint32> 对象的逐元素相加，并返回结果
template <>
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  Vectorized<c10::qint32> retval;
  // 遍历向量的每个元素，计算相加
  for (const auto i : c10::irange(std::decay_t<decltype(a)>::size())) {
    retval.vals[i] = a.vals[i] + b.vals[i];
  }
  return retval;
}

// 结构体模板特化：用于处理 c10::qint8 类型的向量化操作
template <>
struct Vectorized<c10::qint8> : public VectorizedQuantizedConverter<
                                c10::qint8,
                                std::array<Vectorized<float>, 4>,
                                std::array<Vectorized<c10::qint32>, 4>,
                                64> {
  // 默认构造函数：调用基类的默认构造函数
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>() {}
  
  // 构造函数：使用给定的 qint8 值调用基类的构造函数
  Vectorized(c10::qint8 val)
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(val) {}
  
  // 构造函数：使用指针调用基类的构造函数
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(ptr) {}

  // 静态方法：从指定地址加载数据，返回一个 Vectorized<c10::qint8> 对象
  static Vectorized<c10::qint8> loadu(const void* ptr) {
    return Vectorized<c10::qint8>(ptr);
  }

  // 静态方法：从指定地址加载指定数量的数据，返回一个 Vectorized<c10::qint8> 对象
  static Vectorized<c10::qint8> loadu(const void* ptr, int64_t count) {
    // 临时存储数组，用于复制加载数据
    __at_align__ value_type tmp_values[size()];
    // 循环初始化临时数组，避免未初始化内存影响输出值。参考链接详细描述了此问题：
    // https://github.com/pytorch/pytorch/issues/32502
    // 我们不使用“={0}”来将数组初始化为零，因为 gcc 会将其编译成两条指令，而使用循环只需一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    // 将指定地址的数据复制到临时数组
    std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
    // 调用 loadu 方法加载临时数组中的数据，并返回结果
    return loadu(tmp_values);
  }

  // 静态方法：将浮点数向量 rhs 量化为 qint8 类型的向量，使用指定的缩放比例和零点，返回结果
  static Vectorized<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 16> float_vals;

    // 遍历浮点数向量，存储到 float_vals 数组中
    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 16], 16);
    }

    // 调用 PyTorch 的量化函数 quantize_vec，将 float_vals 中的数据量化为 qint8 类型，并存储到 qvals 数组中
    at::native::quantize_vec<c10::qint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint8*)qvals.data(),
        16 * float_num_vecs());

    // 返回量化后的 Vectorized<c10::qint8> 对象
    return loadu(qvals.data());
    // 从数据数组中加载一个 Vectorized 对象，使用 qint8 类型，数据来源于 qvals 数组
    return Vectorized<c10::qint8>::loadu(qvals.data());
  }

  // 计算当前对象与 b 的每个元素的最大值，返回一个新的 Vectorized 对象
  Vectorized<c10::qint8> maximum(Vectorized<c10::qint8> b) const {
    Vectorized<c10::qint8> retval;
    // 遍历当前对象中的每个元素
    for (const auto i : c10::irange(size())) {
      // 将当前对象和 b 的对应位置的值中的较大者赋值给 retval 中对应位置的值
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 计算当前对象与 b 的每个元素的最小值，返回一个新的 Vectorized 对象
  Vectorized<c10::qint8> minimum(Vectorized<c10::qint8> b) const {
    Vectorized<c10::qint8> retval;
    // 遍历当前对象中的每个元素
    for (const auto i : c10::irange(size())) {
      // 将当前对象和 b 的对应位置的值中的较小者赋值给 retval 中对应位置的值
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 对当前对象执行 ReLU 操作，返回一个新的 Vectorized 对象
  Vectorized<c10::qint8> relu(Vectorized<c10::qint8> zero_point) const {
    // 调用 maximum 方法，以 zero_point 作为参数
    return maximum(zero_point);
  }

  // 对当前对象执行 ReLU6 操作，返回一个新的 Vectorized 对象
  Vectorized<c10::qint8> relu6(
      Vectorized<c10::qint8> zero_point,
      Vectorized<c10::qint8> q_six) {
    Vectorized<c10::qint8> retval;
    // 遍历当前对象中的每个元素
    for (const auto i : c10::irange(size())) {
      // 计算当前对象中的值、zero_point 中的值和 q_six 中的值的最大值和最小值，赋值给 retval 中对应位置的值
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  // 执行宽松减法操作，返回一个 int_vec_return_type 对象
  int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    // 遍历每个整数向量
    for (const auto i : c10::irange(int_num_vecs())) {
      // 遍历当前整数向量中的每个元素
      for (const auto j : c10::irange(elem_per_int_vec)) {
        // 计算当前对象和 b 的对应位置的值之间的宽松减法结果，并存储在 retval 中对应位置的值
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }

  // 根据输入的整数向量返回一个 Vectorized 对象，执行从整数的重量化转换
  static Vectorized<c10::qint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vectorized<c10::qint8> retval;
    // 遍历每个整数向量
    for (const auto i : c10::irange(int_num_vecs())) {
      // 遍历当前整数向量中的每个元素
      for (const auto j : c10::irange(elem_per_int_vec)) {
        // 对输入元素执行重量化转换，然后根据 multiplier 和 zero_point 进行量化调整，并存储在 retval 中对应位置的值
        int32_t rounded =
            std::nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
            zero_point;
        retval.vals[i * elem_per_int_vec + j] =
            std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
      }
    }
    return retval;
  }
};

// 特化模板函数，计算两个 c10::qint8 向量的最大值
template <>
Vectorized<c10::qint8> inline maximum(const Vectorized<c10::qint8>& a, const Vectorized<c10::qint8>& b) {
  return a.maximum(b);
}

// 定义 c10::quint8 的特化结构体，继承自 VectorizedQuantizedConverter 类模板
template <>
struct Vectorized<c10::quint8> : public VectorizedQuantizedConverter<
                                 c10::quint8,
                                 std::array<Vectorized<float>, 4>,
                                 std::array<Vectorized<c10::qint32>, 4>,
                                 64> {
  // 默认构造函数，调用基类的默认构造函数
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>() {}

  // 使用 c10::quint8 类型值构造函数，调用基类的对应值构造函数
  Vectorized(c10::quint8 val)
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(val) {}

  // 使用指针构造函数，调用基类的指针构造函数
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(ptr) {}

  // 静态方法，从给定指针加载数据到 c10::quint8 向量
  static Vectorized<c10::quint8> loadu(const void* ptr) {
    return Vectorized<c10::quint8>(ptr);
  }

  // 静态方法，从给定指针加载指定数量的数据到 c10::quint8 向量
  static Vectorized<c10::quint8> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];
    // 确保未初始化的内存不会改变输出值。参考链接中提到，不使用“={0}”初始化数组为零，因为 gcc 会将其编译为两条指令，而使用循环编译为一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    // 将从 ptr 处开始的 count 个数据拷贝到 tmp_values 数组
    std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
    // 调用 loadu 方法加载 tmp_values 数组数据到 c10::quint8 向量
    return loadu(tmp_values);
  }

  // 静态方法，将 float 向量 rhs 量化为 c10::quint8 向量
  static Vectorized<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 16> float_vals;

    // 将 float 向量 rhs 中的数据存储到 float_vals 数组中
    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 16], 16);
    }

    // 调用 PyTorch 的 quantize_vec 函数，将 float 数据量化为 c10::quint8 数据
    at::native::quantize_vec<c10::quint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::quint8*)qvals.data(),
        16 * float_num_vecs());

    // 调用 loadu 方法加载量化后的数据到 c10::quint8 向量
    return Vectorized<c10::quint8>::loadu(qvals.data());
  }

  // 计算当前向量和另一个向量 b 的最大值，返回新的 c10::quint8 向量
  Vectorized<c10::quint8> maximum(Vectorized<c10::quint8> b) const {
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(size())) {
      // 对每个元素计算最大值，并存储到 retval 中
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 计算当前向量和另一个向量 b 的最小值，返回新的 c10::quint8 向量
  Vectorized<c10::quint8> minimum(Vectorized<c10::quint8> b) const {
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(size())) {
      // 对每个元素计算最小值，并存储到 retval 中
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 对当前向量执行 ReLU 操作，返回新的 c10::quint8 向量
  Vectorized<c10::quint8> relu(Vectorized<c10::quint8> zero_point) const {
  // 返回 zero_point 的最大值
  return maximum(zero_point);
}


Vectorized<c10::quint8> relu6(
    Vectorized<c10::quint8> zero_point,
    Vectorized<c10::quint8> q_six) {
  Vectorized<c10::quint8> retval;
  // 遍历向量中的每个元素
  for (const auto i : c10::irange(size())) {
    // 对每个元素进行 relu6 操作，即取值在 [zero_point.vals[i], q_six.vals[i]] 之间
    retval.vals[i] = std::min<value_type>(
        std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
  }
  return retval;
}

// 执行宽展减法操作，返回结果类型为 int_vec_return_type
int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
  int_vec_return_type retval;
  constexpr int elem_per_int_vec = size() / int_num_vecs();
  // 遍历每个向量
  for (const auto i : c10::irange(int_num_vecs())) {
    // 遍历每个向量中的元素
    for (const auto j : c10::irange(elem_per_int_vec)) {
      // 执行宽展减法：将当前对象和向量 b 中对应位置的元素相减
      retval[i].vals[j] =
          static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
          static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
    }
  }
  return retval;
}

// 从整数返回重新量化向量
static Vectorized<c10::quint8> requantize_from_int(
    const int_vec_return_type& inp,
    float multiplier,
    int32_t zero_point) {
  constexpr int elem_per_int_vec = size() / int_num_vecs();
  constexpr auto min_val = std::numeric_limits<value_type>::min();
  constexpr auto max_val = std::numeric_limits<value_type>::max();
  Vectorized<c10::quint8> retval;
  // 遍历每个向量
  for (const auto i : c10::irange(int_num_vecs())) {
    // 遍历每个向量中的元素
    for (const auto j : c10::irange(elem_per_int_vec)) {
      // 计算重新量化的值
      int32_t rounded =
          std::nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
          zero_point;
      // 将结果限制在最小值和最大值之间
      retval.vals[i * elem_per_int_vec + j] =
          std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
    }
  }
  return retval;
}
};

// 结束匿名命名空间

template <>
// 特化模板函数，处理 c10::quint8 类型的向量化最大值操作
Vectorized<c10::quint8> inline maximum(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  // 调用 Vectorized 类的 maximum 方法，返回两个向量的逐元素最大值
  return a.maximum(b);
}

#endif // defined(CPU_CAPABILITY_AVX512) && !defined(MSVC)
// 结束特化模板函数并检查编译器环境，仅在定义了 CPU_CAPABILITY_AVX512 且未定义 MSVC 时有效

}}}
// 结束命名空间 "at::vec256"
```