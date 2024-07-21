# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_qint.h`

```py
// 使用 #pragma once 指令确保头文件只被包含一次
#pragma once

// 在这个头文件中不要定义静态数据！
// 参见 Note [Do not compile initializers with AVX]

// 包含 ATen 库的向量化指令和基础向量化类
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>

// 包含量化模块的基类 AffineQuantizerBase
#include <ATen/native/quantized/AffineQuantizerBase.h>

// 包含 C10 实用工具的头文件，包括整数范围、qint32、qint8 和 quint8
#include <c10/util/irange.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>

// 包含标准库头文件 array 和 cmath
#include <array>
#include <cmath>

// 这个文件定义了用于量化类型的 Vectorized<> 类。

// 当前，我们简单地将这些类用作量化类型与 Vectorized<float> 之间的高效转换器，
// 通常在带宽受限的情况下执行完全精度的算术操作是可接受的（例如逐元素操作）。

// 转换规则如下：
//  Vectorized<qint8> -> 4x Vectorized<float>
//  Vectorized<quint8> -> 4x Vectorized<float>
//  Vectorized<qint32> -> 1x Vectorized<float>

// 返回的浮点向量的大小由特殊 constexpr 函数 float_num_vecs 指定。
// 在 dequantize 函数返回的值类型（作为 quantize 函数的参数）由 float_vec_return_type 指定。

// 在使用这些向量编写内核时，预期浮点操作将在循环中的 Vectorized<T>::float_num_vecs 迭代中执行。

namespace at::vec {
inline namespace CPU_CAPABILITY {

// 如果定义了 CPU_CAPABILITY_AVX2 宏

#ifdef CPU_CAPABILITY_AVX2

// 对于 MSVC 编译器，使用 __declspec(align(64)) 来确保结构体 Vectorizedqi 的对齐性
#ifdef _MSC_VER
__declspec(align(64)) struct Vectorizedqi {
 protected:
  __m256i vals;
#else
// 对于其他编译器，使用 __attribute__((aligned(64))) 来确保结构体 Vectorizedqi 的对齐性
struct Vectorizedqi {
 protected:
  __m256i vals __attribute__((aligned(64)));
#endif

 public:
  Vectorizedqi() {}  // 默认构造函数，未执行任何操作
  Vectorizedqi(__m256i v) : vals(v) {}  // 使用给定 __m256i 初始化 vals 成员
  operator __m256i() const {  // 类型转换运算符，将 Vectorizedqi 转换为 __m256i
    return vals;
  }
};

// 模板函数 pack_saturate_and_clamp 的特化实现，处理 int32_t 类型
template <>
inline __m256i pack_saturate_and_clamp<int32_t>(
    __m256i /*first*/,
    __m256i /*second*/,
    int32_t /*min_val*/,
    int32_t /*max_val*/) {
  // 此函数仅用于链接，不会被使用
  AT_ERROR("pack_saturate_and_clamp<int32_t> is not supported");
}

// 模板函数 pack_saturate_and_clamp 的特化实现，处理 int8_t 类型
template <>
inline __m256i pack_saturate_and_clamp<int8_t>(
    __m256i first,
    __m256i second,
    int8_t min_val,
    int8_t max_val) {
  // 对两个 __m256i 进行饱和包装和截断操作，确保结果在指定的最小和最大值范围内
  __m256i packed_and_sat = _mm256_packs_epi16(first, second);
  return _mm256_max_epi8(
      _mm256_set1_epi8(min_val),
      _mm256_min_epi8(packed_and_sat, _mm256_set1_epi8(max_val)));
}

// 模板函数 pack_saturate_and_clamp 的特化实现，处理 uint8_t 类型
template <>
inline __m256i pack_saturate_and_clamp<uint8_t>(
    __m256i first,
    __m256i second,
    uint8_t min_val,
    uint8_t max_val) {
  // 对两个 __m256i 进行无符号饱和包装和截断操作，确保结果在指定的最小和最大值范围内
  __m256i packed_and_sat = _mm256_packus_epi16(first, second);
  return _mm256_max_epu8(
      _mm256_set1_epi8(min_val),
      _mm256_min_epu8(packed_and_sat, _mm256_set1_epi8(max_val)));
}

// 模板函数 pack_saturate_and_clamp 的通用实现，使用 std::enable_if_t 控制模板特化
template <typename T>
typename std::enable_if_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>, at::vec::Vectorized<float>>
// 定义一个模板函数，将输入向量 src 中的元素转换为 float 类型
inline convert_int8_to_float(at::vec::Vectorized<T> src) {
  // 注意：此函数仅转换 at::vec::Vectorized<float>.size() 个元素的输入
  // 只处理前 8*8 位
  __m128i input_128 = _mm256_castsi256_si128(src);
  // 将 8*uint8/int8 转换为 8*int32
  __m256i input_256_int32;
  if constexpr (std::is_same_v<T, uint8_t>)
    input_256_int32 = _mm256_cvtepu8_epi32(input_128);
  else
    input_256_int32 = _mm256_cvtepi8_epi32(input_128);
  // 将 8*int32 转换为 8*float
  return _mm256_cvtepi32_ps(input_256_int32);
}

// 定义一个模板函数，将输入向量 src 中的 float 元素转换为 int8_t 或 uint8_t 类型
template <typename T>
typename std::enable_if_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>, at::vec::Vectorized<T>>
inline convert_float_to_int8(at::vec::Vectorized<float> src) {
  // 将 float32 转换为 int32（截断方式）
  __m256i x_values_int32 = _mm256_cvttps_epi32(src);

  // 使用有符号饱和转换将 int32 转换为 int16
  __m256i xy_packed_v = _mm256_packs_epi32(x_values_int32, x_values_int32);

  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();

  // 使用无符号饱和转换将 int16 转换为 uint8/int8
  __m256i xyzw_clamped_v = pack_saturate_and_clamp<T>(
      xy_packed_v, xy_packed_v, min_val, max_val);
  __m256i permute_mask_v =
    _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  return _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
}

// 定义一个模板函数，使用 AVX2 指令集对浮点数组 src 进行量化为整数类型 T 的操作
template <typename T>
__FORCE_INLINE void QuantizeAvx2(
    const float* src,
    T* dst,
    int len,
    float inverse_scale,
    int64_t zero_point) {
  constexpr int VLEN = 8;
  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();
  const __m256i min_v = _mm256_set1_epi32(min_val);
  const __m256i max_v = _mm256_set1_epi32(max_val);
  // 这是最大的 int32 值，小于 int32_max 并且可以精确表示为 float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  int i = 0;
  __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
  // clang-format off
  static const __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00);
  // clang-format on
  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256i permute_mask_l8_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
  int len_aligned = len / (VLEN * 4) * (VLEN * 4);
  for (; i < len_aligned; i += 4 * VLEN) {
    // x
    __m256 x_vals = _mm256_load_ps(src + i);
    __m256 x_transformed_v = _mm256_mul_ps(x_vals, inverse_scale_v);
    // 如果浮点值大于 int32_max，则 _mm256_cvtps_epi32 将它们转换为负数。在 int32_float_max_val 处截断
    // Clip at int32_float_max_val to avoid overflow
    x_transformed_v =
        _mm256_min_ps(x_transformed_v, _mm256_set1_ps(int32_float_max_val));
    // Load y values from memory into AVX register
    __m256 y_vals = _mm256_load_ps(src + i + VLEN);
    // Multiply y values by inverse scaling factor
    __m256 y_transformed_v = _mm256_mul_ps(y_vals, inverse_scale_v);
    // Clip y_transformed_v to avoid exceeding int32_float_max_val
    y_transformed_v =
        _mm256_min_ps(y_transformed_v, _mm256_set1_ps(int32_float_max_val));
    // Load z values from memory into AVX register
    __m256 z_vals = _mm256_load_ps(src + i + 2 * VLEN);
    // Multiply z values by inverse scaling factor
    __m256 z_transformed_v = _mm256_mul_ps(z_vals, inverse_scale_v);
    // Clip z_transformed_v to avoid exceeding int32_float_max_val
    z_transformed_v =
        _mm256_min_ps(z_transformed_v, _mm256_set1_ps(int32_float_max_val));
    // Load w values from memory into AVX register
    __m256 w_vals = _mm256_load_ps(src + i + 3 * VLEN);
    // Multiply w values by inverse scaling factor
    __m256 w_transformed_v = _mm256_mul_ps(w_vals, inverse_scale_v);
    // Clip w_transformed_v to avoid exceeding int32_float_max_val
    w_transformed_v =
        _mm256_min_ps(w_transformed_v, _mm256_set1_ps(int32_float_max_val));

    // Convert transformed float vectors to integers
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
    __m256i y_rounded_v = _mm256_cvtps_epi32(y_transformed_v);
    __m256i z_rounded_v = _mm256_cvtps_epi32(z_transformed_v);
    __m256i w_rounded_v = _mm256_cvtps_epi32(w_transformed_v);

    // Add zero point to each of the integer vectors
    x_rounded_v = _mm256_add_epi32(x_rounded_v, _mm256_set1_epi32(zero_point));
    y_rounded_v = _mm256_add_epi32(y_rounded_v, _mm256_set1_epi32(zero_point));
    z_rounded_v = _mm256_add_epi32(z_rounded_v, _mm256_set1_epi32(zero_point));
    w_rounded_v = _mm256_add_epi32(w_rounded_v, _mm256_set1_epi32(zero_point));

    // Pack pairs of 32-bit integers into 16-bit integers
    __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
    __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
    // Saturate and clamp packed values within the specified min_val and max_val range
    __m256i xyzw_clamped_v =
        pack_saturate_and_clamp<T>(xy_packed_v, zw_packed_v, min_val, max_val);

    // Reorder the packed data elements according to permute_mask_v
    xyzw_clamped_v =
        _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
    // Store the result into destination memory starting from position i
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), xyzw_clamped_v);
  }

  // Additional 8-lane AVX2 version to process remaining elements when len is not a multiple of VLEN
  for (; i < len / VLEN * VLEN; i += VLEN) {
    // Load VLEN float values into AVX register from source memory
    __m256 x_vals = _mm256_load_ps(src + i);
    // Multiply x values by inverse scaling factor
    __m256 x_transformed_v = _mm256_mul_ps(x_vals, inverse_scale_v);
    // Clip x_transformed_v to avoid exceeding int32_float_max_val
    x_transformed_v =
        _mm256_min_ps(x_transformed_v, _mm256_set1_ps(int32_float_max_val));
    // Convert transformed float vector to integers
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
    // Add zero point to the integer vector
    x_rounded_v = _mm256_add_epi32(x_rounded_v, _mm256_set1_epi32(zero_point));
    // Clamp the integer vector between min_v and max_v
    __m256i x_clipped_v =
        _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, x_rounded_v));

    // Shuffle and rearrange the 32-bit integers within x_clipped_v according to shuffle_mask_v
    x_clipped_v = _mm256_shuffle_epi8(x_clipped_v, shuffle_mask_v);
    // Permute the 32-bit integers in x_clipped_v based on permute_mask_l8_v
    x_clipped_v = _mm256_permutevar8x32_epi32(x_clipped_v, permute_mask_l8_v);
    // Store the lower 64 bits of x_clipped_v into destination memory starting from position i
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + i),
        _mm256_castsi256_si128(x_clipped_v));
  }

  // Process remaining elements one by one using scalar operations
  for (; i < len; ++i) {
    // Scale the source element by inverse_scale
    float transformed = src[i] * inverse_scale;

    // This comment explains that the behavior here is different from the vectorized code above
    // and notes that the vectorized code always rounds to even in halfway cases
    // 将 transformed 进行四舍五入并加上 zero_point，得到最终变换后的值
    transformed = zero_point + std::nearbyint(transformed);
    // 将 transformed 的值限制在 [min_val, max_val] 的范围内，并进行截断处理
    float clipped =
        std::min(std::max(transformed, float(min_val)), float(max_val));
    // 将处理后的 clipped 值存入目标数组 dst 的第 i 个位置
    dst[i] = clipped;
    }



template<>
struct Vectorized<c10::qint32> : public Vectorizedqi {
    // 定义 size_type 为 int 类型
    using size_type = int;
    
    // 返回向量化对象的大小为 8
    static constexpr size_type size() {
        return 8;
    }

    // 返回浮点数向量的数量为 1
    static constexpr int float_num_vecs() {
        return 1;
    }

    // 返回整数向量的数量为 1
    static constexpr int int_num_vecs() {
        return 1;
    }

    // 定义浮点数向量返回类型为包含一个浮点数向量化对象的数组
    using float_vec_return_type = std::array<Vectorized<float>, 1>;
    
    // 定义整数向量返回类型为包含一个 qint32 向量化对象的数组
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 1>;
    
    // 定义值类型为 c10::qint32 的底层类型
    using value_type = c10::qint32::underlying;

 public:
    // 使用基类 Vectorizedqi 的构造函数
    using Vectorizedqi::Vectorizedqi;

    // 默认构造函数
    Vectorized() {}

    // 带有 __m256i 值参数的构造函数
    Vectorized(__m256i vals_) { vals = vals_;}

    // 广播构造函数，使用 qint32 值作为参数
    Vectorized(const c10::qint32& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi32(uw);
    }

    // 将向量化对象的数据存储到指针 ptr 所指向的内存中，可选参数 count 表示存储的元素个数
    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            // 如果 count 不等于 size()，则使用 memcpy 复制 vals 的数据到 ptr
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            // 否则使用 _mm256_storeu_si256 将 vals 存储到 ptr
            _mm256_storeu_si256((__m256i*)ptr, vals);
        }
    }

    // 从 ptr 所指向的内存中加载数据到向量化对象中
    static Vectorized<c10::qint32> loadu(const void* ptr) {
        return Vectorized<c10::qint32>(ptr);
    }

    // 从 ptr 所指向的内存中加载 count 个数据到向量化对象中
    static Vectorized<c10::qint32> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // 确保未初始化的内存不会改变输出值
        // 参考 https://github.com/pytorch/pytorch/issues/32502 获取更多详细信息
        // 我们不使用 "={0}" 来初始化数组为零，因为 gcc 会将其编译为两条指令，而循环编译为一条指令。
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        // 使用 std::memcpy 将 ptr 指向的数据复制到 tmp_values 中
        std::memcpy(
            tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        // 返回 _mm256_loadu_si256 加载 tmp_values 中的数据
        return _mm256_loadu_si256((const __m256i*)tmp_values);
    }

    // 将 qint32 向量化对象进行反量化操作，返回浮点数向量化对象
    float_vec_return_type dequantize(
        Vectorized<float> scale,
        Vectorized<float> /*zero_point*/,
        Vectorized<float> scale_zp_premul) const {
      __m256 float_vals = _mm256_cvtepi32_ps(vals);
      return {vec::fmadd(scale, Vectorized<float>(float_vals), scale_zp_premul)};
    }

    // 将 qint32 向量化对象进行反量化操作，返回浮点数向量化对象
    float_vec_return_type dequantize(
        Vectorized<float> scale,
        Vectorized<float> zero_point) const {
      __m256 float_vals = _mm256_cvtepi32_ps(vals);
      return {(Vectorized<float>(float_vals) - zero_point) * scale};
    }

    // 将浮点数向量化对象 rhs 进行量化操作，返回 qint32 向量化对象
    static Vectorized<c10::qint32> quantize(
        const float_vec_return_type& rhs,
        float scale,
        int32_t zero_point,
        float /*inverse_scale*/) {
      Vectorized<c10::qint32> retval;
      auto rhs_data = (__m256)rhs[0];
      // 使用 at::native::quantize_vec 进行向量化量化操作
      at::native::quantize_vec<c10::qint32, /*precision=*/32>(
          scale, zero_point, (float*)&rhs_data, (c10::qint32*)&retval.vals, 8);
      return retval;
    }

    // 返回当前向量化对象和 b 的最大值构成的新向量化对象
    Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
      return _mm256_max_epi32(vals, b.vals);
    }
    # 返回两个向量中每个对应元素的最小值向量
    Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
      return _mm256_min_epi32(vals, b.vals);
    }

    # 对向量中的每个元素执行 ReLU 操作
    Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const {
        return maximum(zero_point);
    }

    # 对向量中的每个元素执行 ReLU6 操作，将元素限制在 [zero_point, q_six] 范围内
    Vectorized<c10::qint32> relu6(
        Vectorized<c10::qint32> zero_point,
        Vectorized<c10::qint32> q_six) {
      return _mm256_min_epi32(
          _mm256_max_epi32(vals, zero_point.vals), q_six.vals);
    }

    # 返回两个向量中每个对应元素的宽展（widening）差向量
    int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
      return {_mm256_sub_epi32(vals, b)};
    }

    # 从整数向量转换到指定的量化后的整数向量
    static Vectorized<c10::qint32> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      # 创建包含相同浮点数的 AVX 向量
      __m256 multiplier_v = _mm256_set1_ps(multiplier);
      # 创建包含相同整数的 AVX 向量
      __m256i zero_point_v = _mm256_set1_epi32(zero_point);

      # 执行浮点数乘法，得到缩放后的 AVX 向量
      __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[0]), multiplier_v);
      # 执行浮点数转整数，得到四舍五入后的 AVX 向量
      __m256i rounded = _mm256_cvtps_epi32(scaled);
      # 返回加上零点的 AVX 向量
      return _mm256_add_epi32(rounded, zero_point_v);
    }

 private:
    # 从内存中加载数据构造函数，使用给定指针初始化 AVX 向量
    Vectorized(const void* ptr) {
      vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

// 模板特化：计算两个 Vectorized<c10::qint32> 向量的最大值
template <>
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
  return a.maximum(b);
}

// 模板特化：计算两个 Vectorized<c10::qint32> 向量的逐元素乘积
template <>
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  return _mm256_mullo_epi32(a, b);
}

// 模板特化：计算两个 Vectorized<c10::qint32> 向量的逐元素加法
template <>
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  return _mm256_add_epi32(a, b);
}

/*
 * 将 int32 转换回 int8/uint8 的函数
 */
template <typename T>
__m256i RequantizeAvx2(
    const std::array<Vectorized<c10::qint32>, 4>& inp,
    __m256 multiplier,
    __m256i zp) {
  static_assert(
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
      "Only int8_t/uint8_t are supported");
  constexpr auto min_val = std::numeric_limits<T>::min();  // 获取最小值
  constexpr auto max_val = std::numeric_limits<T>::max();  // 获取最大值
  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);  // 创建用于重排的掩码
  __m256 x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[0]), multiplier);  // 将 inp[0] 扩展为 float 后乘以 multiplier
  __m256 y_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[1]), multiplier);  // 将 inp[1] 扩展为 float 后乘以 multiplier
  __m256 z_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[2]), multiplier);  // 将 inp[2] 扩展为 float 后乘以 multiplier
  __m256 w_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[3]), multiplier);  // 将 inp[3] 扩展为 float 后乘以 multiplier

  __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);  // 将 x_scaled_v 向下取整到整数
  __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);  // 将 y_scaled_v 向下取整到整数
  __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);  // 将 z_scaled_v 向下取整到整数
  __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);  // 将 w_scaled_v 向下取整到整数

  /* 添加零点偏置 */
  __m256i x_v = _mm256_add_epi32(x_rounded_v, zp);  // 将 x_rounded_v 加上 zp
  __m256i y_v = _mm256_add_epi32(y_rounded_v, zp);  // 将 y_rounded_v 加上 zp
  __m256i z_v = _mm256_add_epi32(z_rounded_v, zp);  // 将 z_rounded_v 加上 zp
  __m256i w_v = _mm256_add_epi32(w_rounded_v, zp);  // 将 w_rounded_v 加上 zp

  /* 将结果打包为 int16_t 并进行饱和 */
  __m256i xy_packed_v = _mm256_packs_epi32(x_v, y_v);  // 将 x_v 和 y_v 打包为 int16_t
  __m256i zw_packed_v = _mm256_packs_epi32(z_v, w_v);  // 将 z_v 和 w_v 打包为 int16_t

  __m256i xyzw_clamped_v =
      pack_saturate_and_clamp<T>(xy_packed_v, zw_packed_v, min_val, max_val);  // 调用 pack_saturate_and_clamp 函数，对打包结果进行饱和和截断

  /*
   * xyzw_clamped_v 的布局如下，因此需要重排：x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7
   */
  xyzw_clamped_v = _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);  // 使用 permute_mask_v 进行重排
  return xyzw_clamped_v;  // 返回重排后的结果向量
}

// 模板特化：Vectorized<c10::qint8> 结构体，继承自 Vectorizedqi
template<>
struct Vectorized<c10::qint8> : public Vectorizedqi {
    static constexpr int size() {
        return 32;  // 返回向量大小为 32
    }

    static constexpr int float_num_vecs() {
        return 4;  // 返回 float 向量数为 4
    }

    static constexpr int int_num_vecs() {
        return 4;  // 返回 int 向量数为 4
    }

    using float_vec_return_type = std::array<Vectorized<float>, 4>;  // 定义返回 float 向量的类型
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;  // 定义返回 int 向量的类型
    using value_type = typename c10::qint8::underlying;  // 定义 value_type 类型为 c10::qint8 的 underlying 类型

 public:
    using Vectorizedqi::Vectorizedqi;  // 使用基类的构造函数

    Vectorized() {}  // 默认构造函数
    Vectorized(__m256i vals_) { vals = vals_;}  // 使用给定值构造 Vectorized 对象

    // 广播构造函数
    Vectorized(const c10::qint8& val) {
        value_type uw = val.val_;  // 获取 c10::qint8 对象的值
        vals = _mm256_set1_epi8(uw);  // 使用该值进行广播赋值给 vals
    }

    // This is needed because the compiler emits awful code for the default
    // constructor for moving the enum
    // NOLINTNEXTLINE(clang-diagnostic-deprecated-copy)
    // 暂时关闭 clang 对于 -Wdeprecated-copy 警告
    C10_CLANG_DIAGNOSTIC_PUSH()
    #if C10_CLANG_HAS_WARNING("-Wdeprecated-copy")
    // 忽略 -Wdeprecated-copy 警告
    C10_CLANG_DIAGNOSTIC_IGNORE("-Wdeprecated-copy")
    #endif
    // 从另一个 Vectorized<c10::qint8> 对象构造当前对象
    Vectorized(const Vectorized<c10::qint8>& other) : Vectorizedqi(other.vals) { }
    // 恢复之前的 clang 诊断设置
    C10_CLANG_DIAGNOSTIC_POP()

    // 将当前对象的数据存储到给定的指针位置
    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            // 如果 count 不等于当前对象的大小，则拷贝部分数据到 ptr
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            // 否则直接存储整个 __m256i 数据到 ptr
            _mm256_storeu_si256((__m256i*)ptr, vals);
        }
    }

    // 从未对齐内存中加载数据到 Vectorized<c10::qint8> 对象
    static Vectorized<c10::qint8> loadu(const void* ptr) {
        return Vectorized<c10::qint8>(ptr);
    }

    // 从未对齐内存中加载指定数量的数据到 Vectorized<c10::qint8> 对象
    static Vectorized<c10::qint8> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // 确保未初始化的内存不会改变输出值。参见 https://github.com/pytorch/pytorch/issues/32502
        // 更多细节。我们不使用 "= {0}" 初始化数组为零，因为 gcc 会将其编译成两条指令，而循环只会编译成一条指令。
        // 初始化临时数组为零
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        // 从 ptr 处拷贝 count 个 value_type 大小的数据到 tmp_values
        std::memcpy(
            tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        // 将 tmp_values 中的数据加载为 __m256i 类型返回
        return _mm256_loadu_si256((const __m256i*)tmp_values);
    }

 private:
    // 将 __m128i 类型数据转换为 __m256i 类型数据
    __m256i cvtepi8_epi32(__m128i epi8_vals) const {
        return _mm256_cvtepi8_epi32(epi8_vals);
    }

 public:
  // 反量化操作，将当前对象的数据转换为浮点数，使用给定的 scale 和 scale_neg_zp_premul
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> /*zero_point*/,
      Vectorized<float> scale_neg_zp_premul) const {
    // 提取当前对象的 vals 中的每个 64 位整数到 __m128i 中
    __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
    __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
    __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
    __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

    // 将每个 __m128i 类型数据转换为 __m256 类型浮点数
    __m256 float_val0 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val0));
    __m256 float_val1 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val1));
    __m256 float_val2 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val2));
    __m256 float_val3 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val3));

    // 使用 scale 和 scale_neg_zp_premul 计算反量化后的浮点数值
    auto val0 =
        vec::fmadd(scale, Vectorized<float>(float_val0), scale_neg_zp_premul);
    auto val1 =
        vec::fmadd(scale, Vectorized<float>(float_val1), scale_neg_zp_premul);
    auto val2 =
        vec::fmadd(scale, Vectorized<float>(float_val2), scale_neg_zp_premul);
    auto val3 =
        vec::fmadd(scale, Vectorized<float>(float_val3), scale_neg_zp_premul);
    // 返回四个计算结果作为 float_vec_return_type 类型
    return {val0, val1, val2, val3};
  }

  // 另一种反量化操作，将当前对象的数据转换为浮点数，使用给定的 scale 和 zero_point
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    // 提取当前对象的 vals 中的第一个 64 位整数到 __m128i 中
    __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
    // 使用 AVX2 指令集创建一个包含 vals 中第二个元素的 __m128i 类型整数值
    __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
    // 使用 AVX2 指令集创建一个包含 vals 中第三个元素的 __m128i 类型整数值
    __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
    // 使用 AVX2 指令集创建一个包含 vals 中第四个元素的 __m128i 类型整数值
    __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

    // 将 int_val0 转换为单精度浮点数并进行零点和缩放的处理
    __m256 float_val0 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val0));
    // 将 int_val1 转换为单精度浮点数并进行零点和缩放的处理
    __m256 float_val1 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val1));
    // 将 int_val2 转换为单精度浮点数并进行零点和缩放的处理
    __m256 float_val2 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val2));
    // 将 int_val3 转换为单精度浮点数并进行零点和缩放的处理
    __m256 float_val3 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val3));

    // 计算处理后的值，使用 Vectorized 类封装并进行零点和缩放的处理
    auto val0 = (Vectorized<float>(float_val0) - zero_point) * scale;
    auto val1 = (Vectorized<float>(float_val1) - zero_point) * scale;
    auto val2 = (Vectorized<float>(float_val2) - zero_point) * scale;
    auto val3 = (Vectorized<float>(float_val3) - zero_point) * scale;
    // 返回四个处理后的值作为结果
    return {val0, val1, val2, val3};
  }

  // 使用 AVX2 指令集将 rhs 向量量化为 qint8 类型，返回 Vectorized 对象
  static Vectorized<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float /*scale*/,
      int32_t zero_point,
      float inverse_scale) {
    // 获取 rhs 数据的指针
    auto* rhs_data = (float*)rhs.data();
    // 创建一个 int8_t 数组存储量化后的值
    int8_t quantized_values[32];
    // 使用 AVX2 指令集将 rhs 数据量化为 qint8 类型，存储在 quantized_values 中
    QuantizeAvx2<value_type>(
        rhs_data, quantized_values, 32, inverse_scale, zero_point);
    // 返回包含 quantized_values 的 Vectorized<c10::qint8> 对象
    return Vectorized<c10::qint8>::loadu(quantized_values);
  }

  // 返回当前对象和参数 b 之间的 qint8 向量中的最大值
  Vectorized<c10::qint8> maximum(Vectorized<c10::qint8> b) const {
      return _mm256_max_epi8(vals, b.vals);
    }

  // 返回当前对象和参数 b 之间的 qint8 向量中的最小值
  Vectorized<c10::qint8> minimum(Vectorized<c10::qint8> b) const {
      return _mm256_min_epi8(vals, b.vals);
    }

    // 对当前对象进行 ReLU 操作，返回结果
    Vectorized<c10::qint8> relu(Vectorized<c10::qint8> zero_point) const {
        return maximum(zero_point);
    }

    // 对当前对象进行 ReLU6 操作，返回结果
    Vectorized<c10::qint8> relu6(
        Vectorized<c10::qint8> zero_point,
        Vectorized<c10::qint8> q_six) {
      // 对当前对象与 zero_point 和 q_six 进行 ReLU6 操作，返回结果
      return _mm256_min_epi8(
          _mm256_max_epi8(vals, zero_point.vals), q_six.vals);
    }
    // 定义函数 widening_subtract，返回类型为 int_vec_return_type
    int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
      // 提取 vals 向量的每个64位元素，并分别设置成__m128i类型的变量
      __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
      __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
      __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
      __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

      // 将每个__m128i类型的变量转换为__m256i类型，即将每个64位整数扩展到32位整数
      __m256i int32_val0 = cvtepi8_epi32(int_val0);
      __m256i int32_val1 = cvtepi8_epi32(int_val1);
      __m256i int32_val2 = cvtepi8_epi32(int_val2);
      __m256i int32_val3 = cvtepi8_epi32(int_val3);

      // 从向量 b 中提取每个64位元素，并分别设置成__m128i类型的变量
      __m128i int_b0 = _mm_set1_epi64x(_mm256_extract_epi64(b, 0));
      __m128i int_b1 = _mm_set1_epi64x(_mm256_extract_epi64(b, 1));
      __m128i int_b2 = _mm_set1_epi64x(_mm256_extract_epi64(b, 2));
      __m128i int_b3 = _mm_set1_epi64x(_mm256_extract_epi64(b, 3));

      // 将每个__m128i类型的变量转换为__m256i类型，即将每个64位整数扩展到32位整数
      __m256i int32_b0 = cvtepi8_epi32(int_b0);
      __m256i int32_b1 = cvtepi8_epi32(int_b1);
      __m256i int32_b2 = cvtepi8_epi32(int_b2);
      __m256i int32_b3 = cvtepi8_epi32(int_b3);

      // 对每一组扩展后的32位整数进行相减操作，得到结果__m256i类型的向量
      __m256i res_0 = _mm256_sub_epi32(int32_val0, int32_b0);
      __m256i res_1 = _mm256_sub_epi32(int32_val1, int32_b1);
      __m256i res_2 = _mm256_sub_epi32(int32_val2, int32_b2);
      __m256i res_3 = _mm256_sub_epi32(int32_val3, int32_b3);

      // 返回结果向量，类型为 int_vec_return_type
      return {Vectorized<c10::qint32>(res_0),
              Vectorized<c10::qint32>(res_1),
              Vectorized<c10::qint32>(res_2),
              Vectorized<c10::qint32>(res_3)};
    }

    // 静态函数 requantize_from_int，返回类型为 Vectorized<c10::qint8>
    static Vectorized<c10::qint8> requantize_from_int(
        const int_vec_return_type& inp,  // 输入参数为 int_vec_return_type 类型的引用
        float multiplier,  // 浮点数乘数
        int32_t zero_point) {  // 32位整数零点
      // 创建包含相同乘数的__m256类型的向量
      __m256 multiplier_v = _mm256_set1_ps(multiplier);
      // 创建包含相同零点的__m256i类型的向量
      __m256i zero_point_v = _mm256_set1_epi32(zero_point);
      // 调用 RequantizeAvx2 函数，对输入参数 inp 进行重新量化
      return RequantizeAvx2<value_type>(inp, multiplier_v, zero_point_v);
    }

 private:
    // Load from memory 构造函数，使用指针 ptr 加载数据到 vals 中
    Vectorized(const void* ptr) {
        vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

// 特化模板，用于计算两个 Vectorized<c10::qint8> 类型对象的最大值
template <>
Vectorized<c10::qint8> inline maximum(const Vectorized<c10::qint8>& a, const Vectorized<c10::qint8>& b) {
  return a.maximum(b);
}

// 特化模板，为 Vectorized<c10::quint8> 结构体定义各种静态成员函数和类型别名
template<>
struct Vectorized<c10::quint8> : public Vectorizedqi {
    // 返回向量大小
    static constexpr int size() {
        return 32;
    }

    // 返回浮点数向量个数
    static constexpr int float_num_vecs() {
        return 4;
    }

    // 返回整数向量个数
    static constexpr int int_num_vecs() {
        return 4;
    }

    // 值类型别名
    using float_vec_return_type = std::array<Vectorized<float>, 4>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;
    using value_type = typename c10::quint8::underlying;

public:
    using Vectorizedqi::Vectorizedqi; // 使用基类的构造函数
    Vectorized() {} // 默认构造函数

    // 广播构造函数，将 c10::quint8 类型的值扩展为向量
    Vectorized(const c10::quint8& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi8(uw);
    }

    // 复制构造函数，忽略 Clang 的过时复制警告
    C10_CLANG_DIAGNOSTIC_PUSH()
    #if C10_CLANG_HAS_WARNING("-Wdeprecated-copy")
    C10_CLANG_DIAGNOSTIC_IGNORE("-Wdeprecated-copy")
    #endif
    Vectorized(const Vectorized<c10::quint8>& other) : Vectorizedqi(other.vals) { }
    C10_CLANG_DIAGNOSTIC_POP()

    // 将向量数据存储到内存中
    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            _mm256_storeu_si256((__m256i*)ptr, vals);
        }
    }

    // 从内存中加载数据到向量，可能加载指定数量的数据
    static Vectorized<c10::quint8> loadu(const void* ptr) {
        return Vectorized<c10::quint8>(ptr);
    }

    // 从内存中加载数据到向量，加载指定数量的数据并处理未初始化内存的问题
    static Vectorized<c10::quint8> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // 确保未初始化的内存不会影响输出值
        // 参见 https://github.com/pytorch/pytorch/issues/32502 获取更多详情
        // 我们不使用"={0}"初始化数组为零，因为 gcc 会编译为两条指令，而循环只会编译为一条指令。
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        std::memcpy(
            tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        return _mm256_loadu_si256((const __m256i*)tmp_values);
    }

private:
    // 将 __m128i 类型的值转换为 __m256i 类型
    __m256i cvtepu8_epi32(__m128i epu8_vals) const {
        return _mm256_cvtepu8_epi32(epu8_vals);
    }

public:
    // 对向量进行反量化操作，返回浮点数向量数组
    float_vec_return_type dequantize(
        Vectorized<float> scale,
        Vectorized<float> /*zero_point*/,
        Vectorized<float> scale_zp_premul) const {
      // 设置每个元素的整数值，然后将其转换为浮点数
      __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
      __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
      __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
      __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

      // 将整数转换为浮点数向量
      __m256 float_val0 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val0));
      __m256 float_val1 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val1));
    __m256 float_val2 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val2));
    __m256 float_val3 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val3));
    
    将整数类型的向量 `int_val2` 和 `int_val3` 转换为单精度浮点类型的向量 `float_val2` 和 `float_val3`。
    
    
    auto val0 =
        vec::fmadd(scale, Vectorized<float>(float_val0), scale_zp_premul);
    auto val1 =
        vec::fmadd(scale, Vectorized<float>(float_val1), scale_zp_premul);
    auto val2 =
        vec::fmadd(scale, Vectorized<float>(float_val2), scale_zp_premul);
    auto val3 =
        vec::fmadd(scale, Vectorized<float>(float_val3), scale_zp_premul);
    
    使用向量化指令 `fmadd` 计算四个浮点数向量 `float_val0` 到 `float_val3` 的线性组合，其中 `scale` 是缩放因子，`scale_zp_premul` 是预乘的零点缩放因子。
    
    
    float_vec_return_type dequantize(
        Vectorized<float> scale,
        Vectorized<float> zero_point) const {
      __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
      __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
      __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
      __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));
    
      __m256 float_val0 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val0));
      __m256 float_val1 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val1));
      __m256 float_val2 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val2));
      __m256 float_val3 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val3));
    
      auto val0 = (Vectorized<float>(float_val0) - zero_point) * scale;
      auto val1 = (Vectorized<float>(float_val1) - zero_point) * scale;
      auto val2 = (Vectorized<float>(float_val2) - zero_point) * scale;
      auto val3 = (Vectorized<float>(float_val3) - zero_point) * scale;
      return {val0, val1, val2, val3};
    }
    
    在给定的缩放和零点情况下，将存储在 `vals` 中的整数值转换为浮点数，并进行反量化操作，返回四个浮点数向量。
    
    
    static Vectorized<c10::quint8> quantize(
        const float_vec_return_type& rhs,
        float /*scale*/,
        int32_t zero_point,
        float inverse_scale) {
      auto* rhs_data = (float*)rhs.data();
      uint8_t quantized_values[32];
      QuantizeAvx2<value_type>(
          rhs_data, quantized_values, 32, inverse_scale, zero_point);
      return Vectorized<c10::quint8>::loadu(quantized_values);
    }
    
    将浮点数向量 `rhs` 进行量化，使用 AVX2 指令集中的 `QuantizeAvx2` 函数，将结果存储为 `c10::quint8` 类型的向量。
    
    
    Vectorized<c10::quint8> maximum(Vectorized<c10::quint8> b) const {
        return _mm256_max_epu8(vals, b.vals);
    }
    
    返回当前向量 `vals` 和参数向量 `b` 逐元素比较后的最大值向量。
    
    
    Vectorized<c10::quint8> minimum(Vectorized<c10::quint8> b) const {
        return _mm256_min_epu8(vals, b.vals);
    }
    
    返回当前向量 `vals` 和参数向量 `b` 逐元素比较后的最小值向量。
    
    
    Vectorized<c10::quint8> relu(Vectorized<c10::quint8> zero_point) const {
        return maximum(zero_point);
    }
    
    将当前向量应用于 ReLU 激活函数，返回每个元素与零点向量 `zero_point` 中对应元素的最大值向量。
    
    
    Vectorized<c10::quint8> relu6(
        Vectorized<c10::quint8> zero_point,
        Vectorized<c10::quint8> q_six) {
      return _mm256_min_epu8(
          _mm256_max_epu8(vals, zero_point.vals), q_six.vals);
    }
    
    将当前向量应用于 ReLU6 激活函数，返回每个元素经过 ReLU6 处理后的结果，即限制在 `[zero_point, q_six]` 范围内。
    // 定义一个返回值类型为 int_vec_return_type 的函数，名称为 widening_subtract，接受一个 Vectorized<c10::quint8> 类型的参数 b
    int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
      // 从 vals 的每个元素中提取 64 位整数，然后使用 _mm_set1_epi64x 创建包含这些值的 __m128i 类型变量 int_val0 到 int_val3
      __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
      __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
      __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
      __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

      // 将 int_val0 到 int_val3 转换为包含对应的四个 32 位整数的 __m256i 类型变量 int32_val0 到 int32_val3
      __m256i int32_val0 = cvtepu8_epi32(int_val0);
      __m256i int32_val1 = cvtepu8_epi32(int_val1);
      __m256i int32_val2 = cvtepu8_epi32(int_val2);
      __m256i int32_val3 = cvtepu8_epi32(int_val3);

      // 从 b 的每个元素中提取 64 位整数，然后使用 _mm_set1_epi64x 创建包含这些值的 __m128i 类型变量 int_b0 到 int_b3
      __m128i int_b0 = _mm_set1_epi64x(_mm256_extract_epi64(b, 0));
      __m128i int_b1 = _mm_set1_epi64x(_mm256_extract_epi64(b, 1));
      __m128i int_b2 = _mm_set1_epi64x(_mm256_extract_epi64(b, 2));
      __m128i int_b3 = _mm_set1_epi64x(_mm256_extract_epi64(b, 3));

      // 将 int_b0 到 int_b3 转换为包含对应的四个 32 位整数的 __m256i 类型变量 int32_b0 到 int32_b3
      __m256i int32_b0 = cvtepu8_epi32(int_b0);
      __m256i int32_b1 = cvtepu8_epi32(int_b1);
      __m256i int32_b2 = cvtepu8_epi32(int_b2);
      __m256i int32_b3 = cvtepu8_epi32(int_b3);

      // 使用 _mm256_sub_epi32 计算 int32_val0 到 int32_val3 减去对应的 int32_b0 到 int32_b3 的结果，保存在 res_0 到 res_3 中
      __m256i res_0 = _mm256_sub_epi32(int32_val0, int32_b0);
      __m256i res_1 = _mm256_sub_epi32(int32_val1, int32_b1);
      __m256i res_2 = _mm256_sub_epi32(int32_val2, int32_b2);
      __m256i res_3 = _mm256_sub_epi32(int32_val3, int32_b3);

      // 返回一个包含四个 Vectorized<c10::qint32> 类型对象的 int_vec_return_type 对象，每个对象初始化时使用 res_0 到 res_3
      return {Vectorized<c10::qint32>(res_0),
              Vectorized<c10::qint32>(res_1),
              Vectorized<c10::qint32>(res_2),
              Vectorized<c10::qint32>(res_3)};
    }

    // 静态函数，名称为 requantize_from_int，返回类型为 Vectorized<c10::quint8>
    static Vectorized<c10::quint8> requantize_from_int(
        // 接受 const int_vec_return_type& 类型的 inp 参数，一个 float 类型的 multiplier 参数，一个 int32_t 类型的 zero_point 参数
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      // 使用 _mm256_set1_ps 创建包含 multiplier 值的 __m256 类型变量 multiplier_v
      __m256 multiplier_v = _mm256_set1_ps(multiplier);
      // 使用 _mm256_set1_epi32 创建包含 zero_point 值的 __m256i 类型变量 zero_point_v
      __m256i zero_point_v = _mm256_set1_epi32(zero_point);
      // 调用 RequantizeAvx2<value_type> 函数，传递 inp、multiplier_v 和 zero_point_v 作为参数，并返回结果
      return RequantizeAvx2<value_type>(inp, multiplier_v, zero_point_v);
    }

    // 私有成员函数，名称为 Vectorized，接受 const void* 类型的 ptr 参数
    // 从内存中加载数据构造函数
    Vectorized(const void* ptr) {
        // 使用 _mm256_loadu_si256 从 ptr 加载数据到 vals 变量中
        vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
// NOTE: These are low-performance implementations that we fall back on
// if we are not building with AVX2. This may not be an issue, because
// currently for quantization we assume the user has at least AVX512
// installed, so these can simply act as a reference implementation.
//
// If in the future we relax this requirement (AVX2+), we should probably
// revisit these implementations

template <
    typename T,
    typename float_vec_return_type_,
    typename int_vec_return_type_,
    int size_>
struct VectorizedQuantizedConverter {
  // 返回当前向量化类型的大小
  static constexpr int size() {
    return size_;
  }

  // 返回浮点数向量的数量，假设每个向量包含8个元素
  static constexpr int float_num_vecs() {
    return size() / 8;
  }

  // 返回整数向量的数量，假设每个向量包含8个元素
  static constexpr int int_num_vecs() {
    return size() / 8;
  }

  // 定义浮点数向量和整数向量的返回类型
  using float_vec_return_type = float_vec_return_type_;
  using int_vec_return_type = int_vec_return_type_;

  // 定义值类型为T的底层类型
  using value_type = typename T::underlying;
  // 值数组，大小为size_
  std::array<value_type, size_> vals;

  // 根据给定的T类型的值val初始化值数组vals
  VectorizedQuantizedConverter(T val) {
    for (const auto i : c10::irange(size())) {
      vals[i] = val.val_;
    }
  }

  // 根据给定的指针ptr，从中复制size_个value_type类型的值到值数组vals
  VectorizedQuantizedConverter(const void* ptr) {
    memcpy(vals.data(), ptr, sizeof(value_type) * size());
  }

  // 将值数组vals的内容存储到给定的ptr指针中，可以指定存储的元素个数count
  void store(void* ptr, int count = size()) const {
    memcpy(ptr, vals.data(), count * sizeof(value_type));
  }

  // 对量化的值进行反量化操作，使用给定的scale、zero_point和scale_zp_premul向量
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> /*scale_zp_premul*/) const {
    float_vec_return_type rv;
    // 对每个浮点数向量进行处理
    for (const auto i : c10::irange(float_num_vecs())) {
      float tmp_vals[8];
      // 对当前向量的8个元素进行反量化操作
      for (const auto j : c10::irange(8)) {
        tmp_vals[j] = at::native::dequantize_val<T>(
            scale[j], zero_point[j], T(vals[8 * i + j]));
      }
      // 将反量化后的结果存储到返回值rv中
      rv[i] = Vectorized<float>(tmp_vals[0],
          tmp_vals[1],
          tmp_vals[2],
          tmp_vals[3],
          tmp_vals[4],
          tmp_vals[5],
          tmp_vals[6],
          tmp_vals[7]);
    }
    return rv;
  }

  // 对量化的值进行反量化操作，使用给定的scale和zero_point向量
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    Vectorized<float> scale_zp_premul;
    // 调用带三个参数的反量化函数，将结果返回
    return dequantize(scale, zero_point, scale_zp_premul);
  }

 protected:
  VectorizedQuantizedConverter() {}
};
// 定义模板结构体 Vectorized，用于处理 c10::qint32 类型的向量化操作，继承自 VectorizedQuantizedConverter
struct Vectorized<c10::qint32> : public VectorizedQuantizedConverter<
                                 c10::qint32,
                                 std::array<Vectorized<float>, 1>,
                                 std::array<Vectorized<c10::qint32>, 1>,
                                 8> {
  // 默认构造函数，调用基类 VectorizedQuantizedConverter 的默认构造函数
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            8>() {}

  // 接受 c10::qint32 参数的构造函数，调用基类带参构造函数
  Vectorized(c10::qint32 val)
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            8>(val) {}

  // 接受 const void* 指针参数的构造函数，调用基类带参构造函数
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            8>(ptr) {}

  // 静态方法，从指定地址 ptr 加载数据到 Vectorized<c10::qint32> 对象
  static Vectorized<c10::qint32> loadu(const void* ptr) {
    return Vectorized<c10::qint32>(ptr);
  }

  // 静态方法，从指定地址 ptr 加载 count 个元素到 Vectorized<c10::qint32> 对象
  static Vectorized<c10::qint32> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];

    // 保证未初始化的内存不会改变输出值。参见 https://github.com/pytorch/pytorch/issues/32502 的详细信息。
    // 我们不使用 "={0}" 初始化数组为零，因为 gcc 会将其编译成两条指令，而循环只需一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }

    // 使用 memcpy 将 ptr 指向的数据拷贝到 tmp_values 中，拷贝的字节数为 count * sizeof(value_type)
    std::memcpy(
        tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));

    return Vectorized<c10::qint32>(tmp_values);
  }

  // 静态方法，将浮点型向量 rhs 进行量化，并返回 Vectorized<c10::qint32> 对象
  static Vectorized<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float /*inverse_scale*/) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 8> float_vals;

    // 将 rhs 中的向量数据存储到 float_vals 中
    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 8], 8);
    }

    // 调用 PyTorch 的 quantize_vec 函数，将浮点型数据量化为 c10::qint32 类型
    at::native::quantize_vec<c10::qint32, /*precision=*/32>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint32*)qvals.data(),
        8 * float_num_vecs());

    // 加载量化后的数据到 Vectorized<c10::qint32> 对象并返回
    return Vectorized<c10::qint32>::loadu(qvals.data());
  }

  // 返回当前对象与 b 中对应元素最大值组成的 Vectorized<c10::qint32> 对象
  Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 返回当前对象与 b 中对应元素最小值组成的 Vectorized<c10::qint32> 对象
  Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  // 对当前对象进行 ReLU（整流线性单元）操作，返回结果
  Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const {
  // 返回 zero_point 中的最大值
  return maximum(zero_point);
}


Vectorized<c10::qint32> relu6(
    Vectorized<c10::qint32> zero_point,
    Vectorized<c10::qint32> q_six) {
  Vectorized<c10::qint32> retval;
  // 遍历对象的大小范围
  for (const auto i : c10::irange(size())) {
    // 计算 vals[i]、zero_point.vals[i] 和 q_six.vals[i] 之间的最大值和最小值
    retval.vals[i] = std::min<value_type>(
        std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
  }
  // 返回处理后的结果
  return retval;
}

// 返回类型为 int_vec_return_type 的宽幅减法操作
int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
  int_vec_return_type retval;
  // 遍历对象的大小范围
  for (const auto i : c10::irange(size())) {
    // 计算 vals[i] 与 b.vals[i] 的差，并存入 retval[0].vals[i]
    retval[0].vals[i] = vals[i] - b.vals[i];
  }
  // 返回处理后的结果
  return retval;
}

// 从整数向量返回重新量化的结果
static Vectorized<c10::qint32> requantize_from_int(
    const int_vec_return_type& inp,
    float multiplier,
    int32_t zero_point) {
  Vectorized<c10::qint32> retval;
  // 遍历对象的大小范围
  for (const auto i : c10::irange(size())) {
    // 将 inp[0].vals[i] 乘以 multiplier 后四舍五入，并加上 zero_point，存入 retval.vals[i]
    retval.vals[i] =
        std::nearbyint(static_cast<float>(inp[0].vals[i]) * multiplier) +
        zero_point;
  }
  // 返回处理后的结果
  return retval;
}
};

template <>
// 定义模板特化，用于返回两个 Vectorized<c10::qint32> 向量中的元素最大值的向量化结果
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
  return a.maximum(b);
}

template <>
// 定义模板特化，用于实现两个 Vectorized<c10::qint32> 向量的逐元素乘法操作
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  Vectorized<c10::qint32> retval;
  // 遍历向量中的每个元素，执行乘法操作并存储结果到新的向量中
  for (const auto i : c10::irange(std::decay_t<decltype(a)>::size())) {
    retval.vals[i] = a.vals[i] * b.vals[i];
  }
  return retval;
}

template <>
// 定义模板特化，用于实现两个 Vectorized<c10::qint32> 向量的逐元素加法操作
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  Vectorized<c10::qint32> retval;
  // 遍历向量中的每个元素，执行加法操作并存储结果到新的向量中
  for (const auto i : c10::irange(std::decay_t<decltype(a)>::size())) {
    retval.vals[i] = a.vals[i] + b.vals[i];
  }
  return retval;
}

template <>
// 定义模板特化，Vectorized<c10::qint8> 结构体，继承自 VectorizedQuantizedConverter，用于量化转换操作
struct Vectorized<c10::qint8> : public VectorizedQuantizedConverter<
                                c10::qint8,
                                std::array<Vectorized<float>, 4>,
                                std::array<Vectorized<c10::qint32>, 4>,
                                32> {
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            32>() {}
  Vectorized(c10::qint8 val)
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            32>(val) {}
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            32>(ptr) {}

  static Vectorized<c10::qint8> loadu(const void* ptr) {
    // 使用未对齐方式加载数据指针，并返回相应的 Vectorized<c10::qint8> 向量
    return Vectorized<c10::qint8>(ptr);
  }

  static Vectorized<c10::qint8> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];
    // 确保未初始化的内存不会改变输出值，参考 https://github.com/pytorch/pytorch/issues/32502
    // 不使用“={0}”将数组初始化为零，因为 gcc 会将其编译为两条指令，而使用循环只需一条指令
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    // 使用 memcpy 将源数据的 count 倍 value_type 字节复制到临时数组 tmp_values 中
    std::memcpy(
        tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
    // 返回基于 tmp_values 的 Vectorized<c10::qint8> 向量
    return Vectorized<c10::qint8>(tmp_values);
  }

  static Vectorized<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float /*inverse_scale*/) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 8> float_vals;

    // 遍历 float_vec_return_type 类型的向量 rhs，将其存储到 float_vals 数组中
    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 8], 8);
    }

    // 使用 ATen 的 quantize_vec 函数进行向量量化操作，将结果存储到 qvals 中
    at::native::quantize_vec<c10::qint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint8*)qvals.data(),
        8 * float_num_vecs());
  return Vectorized<c10::qint8>::loadu(qvals.data());
}

// 返回当前向量和另一个向量元素的最大值构成的新向量
Vectorized<c10::qint8> maximum(Vectorized<c10::qint8> b) const {
  Vectorized<c10::qint8> retval;
  // 遍历向量中的每个元素，取当前向量和给定向量对应位置的最大值
  for (const auto i : c10::irange(size())) {
    retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
  }
  return retval;
}

// 返回当前向量和另一个向量元素的最小值构成的新向量
Vectorized<c10::qint8> minimum(Vectorized<c10::qint8> b) const {
  Vectorized<c10::qint8> retval;
  // 遍历向量中的每个元素，取当前向量和给定向量对应位置的最小值
  for (const auto i : c10::irange(size())) {
    retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
  }
  return retval;
}

// 对当前向量进行 ReLU 操作，即将小于零的元素设为零
Vectorized<c10::qint8> relu(Vectorized<c10::qint8> zero_point) const {
  return maximum(zero_point);
}

// 对当前向量执行 ReLU6 操作，将小于零的元素设为零，大于六的元素设为六
Vectorized<c10::qint8> relu6(
    Vectorized<c10::qint8> zero_point,
    Vectorized<c10::qint8> q_six) {
  Vectorized<c10::qint8> retval;
  // 遍历向量中的每个元素，进行 ReLU6 计算
  for (const auto i : c10::irange(size())) {
    retval.vals[i] = std::min<value_type>(
        std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
  }
  return retval;
}

// 返回当前向量与另一个向量逐元素相减的结果
int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
  int_vec_return_type retval;
  constexpr int elem_per_int_vec = size() / int_num_vecs();
  // 对每个向量执行宽松减法操作
  for (const auto i : c10::irange(int_num_vecs())) {
    for (const auto j : c10::irange(elem_per_int_vec)) {
      retval[i].vals[j] =
          static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
          static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
    }
  }
  return retval;
}

// 根据输入的整数向量返回重新量化后的向量
static Vectorized<c10::qint8> requantize_from_int(
    const int_vec_return_type& inp,
    float multiplier,
    int32_t zero_point) {
  constexpr int elem_per_int_vec = size() / int_num_vecs();
  constexpr auto min_val = std::numeric_limits<value_type>::min();
  constexpr auto max_val = std::numeric_limits<value_type>::max();
  Vectorized<c10::qint8> retval;
  // 对每个向量执行重新量化操作
  for (const auto i : c10::irange(int_num_vecs())) {
    for (const auto j : c10::irange(elem_per_int_vec)) {
      // 对每个元素执行重新量化，确保在最小和最大值范围内
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

template <>
// 特化模板，用于计算两个 c10::qint8 向量的最大值
Vectorized<c10::qint8> inline maximum(const Vectorized<c10::qint8>& a, const Vectorized<c10::qint8>& b) {
  return a.maximum(b);
}

template <>
// 特化模板，定义了 Vectorized 类型为 c10::quint8 的结构体，继承自 VectorizedQuantizedConverter
struct Vectorized<c10::quint8> : public VectorizedQuantizedConverter<
                                 c10::quint8,
                                 std::array<Vectorized<float>, 4>,
                                 std::array<Vectorized<c10::qint32>, 4>,
                                 32> {
  // 默认构造函数
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            32>() {}

  // 值构造函数，初始化为指定的 c10::quint8 值
  Vectorized(c10::quint8 val)
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            32>(val) {}

  // 指针构造函数，从指定的内存地址加载数据
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            32>(ptr) {}

  // 静态方法，从未对齐的内存地址加载数据
  static Vectorized<c10::quint8> loadu(const void* ptr) {
    return Vectorized<c10::quint8>(ptr);
  }

  // 静态方法，从未对齐的内存地址加载指定数量的数据
  static Vectorized<c10::quint8> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];
    // 确保未初始化的内存不会改变输出值，请参阅 https://github.com/pytorch/pytorch/issues/32502
    // 更多细节。我们不使用 "={0}" 初始化数组为零，因为 gcc 会将其编译为两条指令，而循环编译为一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    // 使用 memcpy 将数据从给定地址复制到临时数组中
    std::memcpy(
        tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
    // 返回从临时数组构造的 Vectorized<c10::quint8> 对象
    return Vectorized<c10::quint8>(tmp_values);
  }

  // 静态方法，将 float 向量 rhs 进行量化为 c10::quint8 类型
  static Vectorized<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float /*inverse_scale*/) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 8> float_vals;

    // 将 float 向量 rhs 中的数据存储到 float_vals 数组中
    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 8], 8);
    }

    // 使用 PyTorch 的量化函数将 float_vals 数组量化为 c10::quint8，并存储到 qvals 数组中
    at::native::quantize_vec<c10::quint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::quint8*)qvals.data(),
        8 * float_num_vecs());

    // 返回从 qvals 数组加载的 Vectorized<c10::quint8> 对象
    return Vectorized<c10::quint8>::loadu(qvals.data());
  }

  // 方法，计算当前对象与向量 b 的每个元素的最大值并返回结果
  Vectorized<c10::quint8> maximum(Vectorized<c10::quint8> b) const {
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(size())) {
      // 计算当前向量 vals 中第 i 个元素与向量 b 的第 i 个元素的最大值，并存储到返回值中
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    // 返回计算后的结果向量
    return retval;
  }

  // 方法，计算当前对象与向量 b 的每个元素的最小值并返回结果
  Vectorized<c10::quint8> minimum(Vectorized<c10::quint8> b) const {
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(size())) {
      // 计算当前向量 vals 中第 i 个元素与向量 b 的第 i 个元素的最小值，并存储到返回值中
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    // 返回计算后的结果向量
    return retval;
  }


这段代码是一个 C++ 的模板特化示例，定义了操作 c10::quint8 类型向量的各种方法和构造函数。
  }

  // 执行 ReLU 操作，将每个元素与零点比较取较大值
  Vectorized<c10::quint8> relu(Vectorized<c10::quint8> zero_point) const {
    return maximum(zero_point);
  }


  // 执行 ReLU6 操作，将每个元素与零点和六点之间的最大最小值进行比较
  Vectorized<c10::quint8> relu6(
      Vectorized<c10::quint8> zero_point,
      Vectorized<c10::quint8> q_six) {
    Vectorized<c10::quint8> retval;
    // 对每个向量元素执行 ReLU6 运算
    for (const auto i : c10::irange(size())) {
      // 计算每个位置的结果，限制在 zero_point 和 q_six 之间
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  // 执行宽展减法操作，返回向量间的减法结果
  int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    // 对每个整数向量执行逐元素的减法运算
    for (const auto i : c10::irange(int_num_vecs())) {
      for (const auto j : c10::irange(elem_per_int_vec)) {
        // 执行减法操作并存储结果
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }

  // 从整数向量重新量化为 quint8 向量
  static Vectorized<c10::quint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vectorized<c10::quint8> retval;
    // 对每个整数向量中的元素执行重新量化操作
    for (const auto i : c10::irange(int_num_vecs())) {
      for (const auto j : c10::irange(elem_per_int_vec)) {
        // 计算量化后的值，并限制在 quint8 范围内
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

// 结束模板特化的定义

template <>
// 实现 Vectorized 类模板的特化，针对 c10::quint8 类型
Vectorized<c10::quint8> inline maximum(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  // 调用 Vectorized 类的 maximum 方法，返回 a 和 b 中的每个元素的最大值组成的 Vectorized 对象
  return a.maximum(b);
}

#endif // if defined(CPU_CAPABILITY_AVX2)
}} // namespace at::vec::CPU_CAPABILITY
```