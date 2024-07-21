# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_half_neon.h`

```py
#pragma once
// 在此头文件中不要定义静态数据！
// 详见注意事项 [不要使用 AVX 编译初始化器]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vec256_float_neon.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

namespace at::vec {
// 见注意事项 [CPU_CAPABILITY 命名空间]
inline namespace CPU_CAPABILITY {

// 目前仅包含 aarch64 的实现。
// 目前不支持 aarch32，原因如下：
// 1. aarch64 和 aarch32 ISA 有差异，aarch64 的内部函数在 aarch32 上不兼容。
// 2. Android NDK r21 在编译 aarch32 时存在问题，Clang 会崩溃。
//    参考链接：
//    https://github.com/android/ndk/issues/1248
//    https://bugs.llvm.org/show_bug.cgi?id=45824
// 很可能我们会使用内联汇编来支持 aarch32。
#if !defined(C10_MOBILE) && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#ifdef __BIG_ENDIAN__
#error "不支持大端序。"
#endif

// BlendHalfRegs 模板结构体，用于混合半精度寄存器
template <int index, bool mask_val>
struct BlendHalfRegs {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res);
};

// 当 mask_val 为 true 时的特化实现
template <int index>
struct BlendHalfRegs<index, true> {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res) {
    // 将 b 的第 index 个元素设为 res 的第 index 个元素
    return vsetq_lane_f16(vgetq_lane_f16(b, index), res, index);
  }
};

// 当 mask_val 为 false 时的特化实现
template <int index>
struct BlendHalfRegs<index, false> {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res) {
    // 将 a 的第 index 个元素设为 res 的第 index 个元素
    return vsetq_lane_f16(vgetq_lane_f16(a, index), res, index);
  }
};

// 在 ARM 上，Half 类型支持 float16_t -> Half 构造函数和 Half -> float16_t 转换
template <>
class Vectorized<c10::Half> {
 private:
  float16x8x2_t values; // 存储两个 float16x8_t 向量

 public:
  using value_type = c10::Half; // value_type 应为 c10::Half，以适应 vec_base.h 的接口
  using size_type = int;
  static constexpr size_type size() {
    static_assert(sizeof(float16x8x2_t) == 16 * sizeof(value_type));
    return 16; // 返回向量长度为 16
  }

 private:
  // 使用私有的 map 函数实现各种方法
  Vectorized<c10::Half> map2(
      const Vectorized<c10::Half>& second,
      c10::Half (*const f)(c10::Half, c10::Half)) const {
    __at_align__ c10::Half tmp_first[size()]; // 声明临时数组 tmp_first
    __at_align__ c10::Half tmp_second[size()]; // 声明临时数组 tmp_second
    store(tmp_first); // 将当前对象的数据存储到 tmp_first 中
    second.store(tmp_second); // 将第二个对象的数据存储到 tmp_second 中
    for (const auto i : c10::irange(size())) {
      tmp_first[i] = f(tmp_first[i], tmp_second[i]); // 对每个元素执行函数 f 操作
    }
    return loadu(tmp_first); // 返回加载 tmp_first 中的数据后的对象
  }

  // 使用 Vectorized<float> 的方法映射到当前对象
  Vectorized<c10::Half> map_with_vec_float_method(
      Vectorized<float> (Vectorized<float>::*m)() const) const {
    // 将低位的 float16x8_t 转换为两个 float32x4_t 变量，应用 m，然后再转换回来
    float32x4_t v00 = vcvt_f32_f16(vget_low_f16(values.val[0]));
    float32x4_t v01 = vcvt_f32_f16(vget_high_f16(values.val[0]));
    // 使用向量化乘法操作，将两个 float 向量 v00 和 v01 与矩阵 m 相乘，返回向量化结果
    Vectorized<float> mv0 = (Vectorized<float>(v00, v01).*m)();

    // 将得到的低位 float16x8_t 转换为两个 float32x4_t 变量 r00 和 r01，然后转换为 float16x4_t
    float16x4_t r00 = vcvt_f16_f32(mv0.get_low());
    float16x4_t r01 = vcvt_f16_f32(mv0.get_high());

    // 将高位 float16x8_t 转换为两个 float32x4_t 变量 v10 和 v11，然后转换为 float16x4_t
    float32x4_t v10 = vcvt_f32_f16(vget_low_f16(values.val[1]));
    float32x4_t v11 = vcvt_f32_f16(vget_high_f16(values.val[1]));
    // 使用向量化乘法操作，将两个 float 向量 v10 和 v11 与矩阵 m 相乘，返回向量化结果
    Vectorized<float> mv1 = (Vectorized<float>(v10, v11).*m)();
    // 将得到的低位 float16x8_t 转换为两个 float32x4_t 变量 r10 和 r11，然后转换为 float16x4_t
    float16x4_t r10 = vcvt_f16_f32(mv1.get_low());
    float16x4_t r11 = vcvt_f16_f32(mv1.get_high());

    // 将 r00 和 r01 合并为一个 float16x8_t 变量，再将 r10 和 r11 合并为另一个 float16x8_t 变量，
    // 最终封装成 Vectorized<c10::Half> 返回
    return Vectorized<c10::Half>(
        vcombine_f16(r00, r01), vcombine_f16(r10, r11));
}
    
Vectorized<c10::Half> map2_with_vec_float_method(
    const Vectorized<c10::Half>& second,
    Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
        const) const {
    // 将低位 float16x8_t 转换为两个 float32x4_t 变量 v00 和 v01，以及 second 的低位 float16x8_t 转换为两个 float32x4_t 变量 second_v00 和 second_v01
    float32x4_t v00 = vcvt_f32_f16(vget_low_f16(values.val[0]));
    float32x4_t v01 = vcvt_f32_f16(vget_high_f16(values.val[0]));
    float32x4_t second_v00 = vcvt_f32_f16(vget_low_f16(second.get_low()));
    float32x4_t second_v01 = vcvt_f32_f16(vget_high_f16(second.get_low()));
    // 使用向量化乘法操作，将 v00 和 v01 与 second_v00 和 second_v01 相乘，返回向量化结果
    Vectorized<float> mv0 = (Vectorized<float>(v00, v01).*m)(
        Vectorized<float>(second_v00, second_v01));
    // 将得到的低位 float16x8_t 转换为两个 float32x4_t 变量 r00 和 r01，然后转换为 float16x4_t
    float16x4_t r00 = vcvt_f16_f32(mv0.get_low());
    float16x4_t r01 = vcvt_f16_f32(mv0.get_high());

    // 将高位 float16x8_t 转换为两个 float32x4_t 变量 v10 和 v11，以及 second 的高位 float16x8_t 转换为两个 float32x4_t 变量 second_v10 和 second_v11
    float32x4_t v10 = vcvt_f32_f16(vget_low_f16(values.val[1]));
    float32x4_t v11 = vcvt_f32_f16(vget_high_f16(values.val[1]));
    float32x4_t second_v10 = vcvt_f32_f16(vget_low_f16(second.get_high()));
    float32x4_t second_v11 = vcvt_f32_f16(vget_high_f16(second.get_high()));
    // 使用向量化乘法操作，将 v10 和 v11 与 second_v10 和 second_v11 相乘，返回向量化结果
    Vectorized<float> mv1 = (Vectorized<float>(v10, v11).*m)(
        Vectorized<float>(second_v10, second_v11));
    // 将得到的低位 float16x8_t 转换为两个 float32x4_t 变量 r10 和 r11，然后转换为 float16x4_t
    float16x4_t r10 = vcvt_f16_f32(mv1.get_low());
    float16x4_t r11 = vcvt_f16_f32(mv1.get_high());

    // 将 r00 和 r01 合并为一个 float16x8_t 变量，再将 r10 和 r11 合并为另一个 float16x8_t 变量，
    // 最终封装成 Vectorized<c10::Half> 返回
    // 返回一个包含两个 float16x8_t 向量的 Vectorized 对象
    return Vectorized<c10::Half>(
        vcombine_f16(r00, r01), vcombine_f16(r10, r11));
    }
    
    public:
    // 默认构造函数
    Vectorized() {}
    // 接受 float16x8x2_t 参数的构造函数
    Vectorized(float16x8x2_t v) : values(v) {}
    
    // 为了符合 vec_base.h 接口而添加的接受 c10::Half 类型参数的构造函数
    // 同时也包含接受 float16_t 参数的构造函数
    Vectorized(c10::Half val)
        : values{vdupq_n_f16((float16_t)val), vdupq_n_f16((float16_t)val)} {
    }
    Vectorized(float16_t val) : values{vdupq_n_f16(val), vdupq_n_f16(val)} {}
    
    // 接受 16 个 float16_t 类型参数的构造函数
    Vectorized(
        float16_t val0,
        float16_t val1,
        float16_t val2,
        float16_t val3,
        float16_t val4,
        float16_t val5,
        float16_t val6,
        float16_t val7,
        float16_t val8,
        float16_t val9,
        float16_t val10,
        float16_t val11,
        float16_t val12,
        float16_t val13,
        float16_t val14,
        float16_t val15)
        : values{
              val0,
              val1,
              val2,
              val3,
              val4,
              val5,
              val6,
              val7,
              val8,
              val9,
              val10,
              val11,
              val12,
              val13,
              val14,
              val15} {}
    
    // 接受两个 float16x8_t 向量作为参数的构造函数
    Vectorized(float16x8_t val0, float16x8_t val1) : values{val0, val1} {}
    
    // 类型转换操作符，将 Vectorized 对象转换为 float16x8x2_t 类型
    operator float16x8x2_t() const {
        return values;
    }
    
    // 静态方法，根据掩码 mask 对两个 Vectorized<c10::Half> 对象进行混合
    template <int64_t mask>
    static Vectorized<c10::Half> blend(
        const Vectorized<c10::Half>& a,
        const Vectorized<c10::Half>& b) {
      Vectorized<c10::Half> vec;
    
      // 对第一个向量进行掩码混合操作
      vec.values.val[0] = BlendHalfRegs<0, (mask & 0x01) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
      vec.values.val[0] = BlendHalfRegs<1, (mask & 0x02) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
      vec.values.val[0] = BlendHalfRegs<2, (mask & 0x04) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
      vec.values.val[0] = BlendHalfRegs<3, (mask & 0x08) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
      vec.values.val[0] = BlendHalfRegs<4, (mask & 0x10) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
      vec.values.val[0] = BlendHalfRegs<5, (mask & 0x20) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
      vec.values.val[0] = BlendHalfRegs<6, (mask & 0x40) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
      vec.values.val[0] = BlendHalfRegs<7, (mask & 0x80) != 0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    
      // 对第二个向量进行掩码混合操作
      vec.values.val[1] = BlendHalfRegs<0, (mask & 0x10) != 0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
      vec.values.val[1] = BlendHalfRegs<1, (mask & 0x20) != 0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
      vec.values.val[1] = BlendHalfRegs<2, (mask & 0x40) != 0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<3, (mask & 0x80) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    // 使用 BlendHalfRegs 模板，根据掩码的第 7 位（mask & 0x80 != 0）选择实现 3 的混合操作，将结果存入 vec 的第 1 个值中
    vec.values.val[1] = BlendHalfRegs<4, (mask & 0x10) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    // 使用 BlendHalfRegs 模板，根据掩码的第 4 位（mask & 0x10 != 0）选择实现 4 的混合操作，将结果存入 vec 的第 1 个值中
    vec.values.val[1] = BlendHalfRegs<5, (mask & 0x20) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    // 使用 BlendHalfRegs 模板，根据掩码的第 5 位（mask & 0x20 != 0）选择实现 5 的混合操作，将结果存入 vec 的第 1 个值中
    vec.values.val[1] = BlendHalfRegs<6, (mask & 0x40) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    // 使用 BlendHalfRegs 模板，根据掩码的第 6 位（mask & 0x40 != 0）选择实现 6 的混合操作，将结果存入 vec 的第 1 个值中
    vec.values.val[1] = BlendHalfRegs<7, (mask & 0x80) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    // 使用 BlendHalfRegs 模板，根据掩码的第 7 位（mask & 0x80 != 0）选择实现 7 的混合操作，将结果存入 vec 的第 1 个值中

    return vec;
  }
  static Vectorized<c10::Half> blendv(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      const Vectorized<c10::Half>& mask) {
    // 注意：使用 blendv 非常笨拙，因为 0xFFFF 是 FP16 中的众多 NaN 之一，这是由于 mask 需要 Half 类型（来自 vec_base）

    // TODO
    // 注意：这要求每个值，即每个 uint 值的 mask，要么全是零，要么全是 1。
    // 我们可能需要某种形式的断言？
    // 但这会影响性能。

    // 初始化一个新的 Vectorized<c10::Half> 对象，使用 mask 的值进行初始化
    Vectorized<c10::Half> vec(mask.values);

    // 使用 vbslq_f16 直接进行混合，根据 mask 的低位部分选择 a 或 b 的对应值，并存入 vec 的第 0 个值中
    vec.values.val[0] = vbslq_f16(
        vreinterpretq_u16_f16(vec.values.val[0]),
        b.values.val[0],
        a.values.val[0]);

    // 使用 vbslq_f16 直接进行混合，根据 mask 的高位部分选择 a 或 b 的对应值，并存入 vec 的第 1 个值中
    vec.values.val[1] = vbslq_f16(
        vreinterpretq_u16_f16(vec.values.val[1]),
        b.values.val[1],
        a.values.val[1]);

    return vec;
  }
  template <typename step_t>
  static Vectorized<c10::Half> arange(
      c10::Half base = 0.0,
      step_t step = static_cast<step_t>(1)) {
    const Vectorized<c10::Half> base_vec(base);
    const Vectorized<c10::Half> step_vec(step);
    const Vectorized<c10::Half> step_sizes(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    // 返回基于步长和起始值计算得到的向量
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vectorized<c10::Half> set(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      int64_t count = size()) {
    // 初始化一个 uint16_t 类型的数组，用于存放掩码值，初始值全为 0
    uint16_t pre_mask[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 将数组 pre_mask 中的前 count 个元素设置为 0xFFFF
    for (int i = 0; i < count; i++) {
      pre_mask[i] = 0xFFFF;
    }

    // 使用 vld1q_u16_x2 将 pre_mask 转换为双 quadword 形式的 mask
    uint16x8x2_t mask = vld1q_u16_x2(pre_mask);

    // 直接使用 vbslq_f16 进行混合，根据 mask 的值选择 a 或 b 的对应值，并存入 vec 中
    Vectorized<c10::Half> vec(
        vbslq_f16(
            // 低位
            mask.val[0],
            b.values.val[0],
            a.values.val[0]),
        // 高位
        vbslq_f16(mask.val[1], b.values.val[1], a.values.val[1]));

    return vec;
  }
  static Vectorized<c10::Half> loadu(const void* ptr, int64_t count = size()) {
    // 如果 count 等于 size()，则使用 vld1q_f16_x2 从指针 ptr 处加载数据并返回
    if (count == size()) {
      return vld1q_f16_x2(reinterpret_cast<const float16_t*>(ptr));
    }
    // 否则，暂不处理 count 小于 size() 的情况，可以视情况补充逻辑
  }
  // 如果 count 等于 size 的一半，则执行以下代码块
  } else if (count == (size() >> 1)) {
    // 创建一个存储半精度浮点数的向量
    Vectorized<c10::Half> res;
    // 将第一个值从 ptr 转换为 float16_t 类型后存储到 res 的值中
    res.values.val[0] = vld1q_f16(reinterpret_cast<const float16_t*>(ptr));
    // 将 res 的第二个值置零
    std::memset(&res.values.val[1], 0, sizeof(res.values.val[1]));
    return res;
  }

  // 创建一个 float16_t 类型的临时数组 tmp_values，大小为 size()
  __at_align__ float16_t tmp_values[size()];
  // 将 tmp_values 数组中的每个元素初始化为 0
  for (const auto i : c10::irange(size())) {
    tmp_values[i] = 0;
  }
  // 将 ptr 指向的 count 个 float16_t 类型的数据复制到 tmp_values 数组中
  std::memcpy(
      tmp_values,
      reinterpret_cast<const float16_t*>(ptr),
      count * sizeof(float16_t));
  // 返回一个加载了 tmp_values 数据的向量
  return vld1q_f16_x2(reinterpret_cast<const float16_t*>(tmp_values));
}

// 将向量中的数据存储到指定的内存地址 ptr 中，存储的数量由 count 决定，默认为 size()
void store(void* ptr, int64_t count = size()) const {
  // 如果 count 等于 size()，直接将 values 中的数据存储到 ptr 指向的地址中
  if (count == size()) {
    vst1q_f16_x2(reinterpret_cast<float16_t*>(ptr), values);
    return;
  }
  // 如果 count 等于 size() 的一半，只存储 values 的第一个值到 ptr 指向的地址中
  else if (count == (size() >> 1)) {
    vst1q_f16(reinterpret_cast<float16_t*>(ptr), values.val[0]);
  }
  // 否则，创建一个 float16_t 类型的临时数组 tmp_values，大小为 size()
  else {
    float16_t tmp_values[size()];
    // 将 values 中的数据存储到 tmp_values 数组中
    vst1q_f16_x2(reinterpret_cast<float16_t*>(tmp_values), values);
    // 将 tmp_values 数组中的 count 个数据复制到 ptr 指向的地址中
    std::memcpy(ptr, tmp_values, count * sizeof(float16_t));
  }
}

// 返回存储在 values 中的第一个向量元素的引用（低位）
inline const float16x8_t& get_low() const {
  return values.val[0];
}

// 返回存储在 values 中的第一个向量元素的引用（低位），可修改版本
inline float16x8_t& get_low() {
  return values.val[0];
}

// 返回存储在 values 中的第二个向量元素的引用（高位）
inline const float16x8_t& get_high() const {
  return values.val[1];
}

// 返回存储在 values 中的第二个向量元素的引用（高位），可修改版本
inline float16x8_t& get_high() {
  return values.val[1];
}

// 索引运算符重载，返回存储在 tmp 数组中索引为 idx 的元素
// 注意：这是一个非常慢的实现，只在 vec256_qint 中引用了它，一旦为 ARM 专门优化，应该移除
c10::Half operator[](int idx) const {
  __at_align__ c10::Half tmp[size()];
  store(tmp);
  return tmp[idx];
}

// 索引运算符重载，返回存储在 tmp 数组中索引为 idx 的元素，可修改版本
// 注意：这是一个非常慢的实现，只在 vec256_qint 中引用了它，一旦为 ARM 专门优化，应该移除
c10::Half operator[](int idx) {
  __at_align__ c10::Half tmp[size()];
  store(tmp);
  return tmp[idx];
}

// 返回一个整数，表示 tmp 数组中每个元素是否为零的掩码
int zero_mask() const {
  __at_align__ c10::Half tmp[size()];
  store(tmp);
  int mask = 0;
  for (int i = 0; i < size(); ++i) {
    if (tmp[i] == 0) {
      mask |= (1 << i);
    }
  }
  return mask;
}

// 返回一个新的 Vectorized 对象，其中元素表示对应位置上 tmp 数组元素是否为 NaN
Vectorized<c10::Half> isnan() const {
  __at_align__ c10::Half tmp[size()];
  __at_align__ c10::Half res[size()];
  store(tmp);
  for (const auto i : c10::irange(size())) {
    if (_isnan(tmp[i])) {
      // 如果 tmp[i] 是 NaN，则将 res[i] 全部设为 0xFF（表示 true）
      std::memset(static_cast<void*>(&res[i]), 0xFF, sizeof(c10::Half));
    } else {
      // 否则，将 res[i] 全部设为 0（表示 false）
      std::memset(static_cast<void*>(&res[i]), 0, sizeof(c10::Half));
    }
  }
  // 加载 res 数组到 Vectorized 对象并返回
  return loadu(res);
};

// 返回一个布尔值，指示 tmp 数组中是否存在 NaN 或 Infinity
bool has_inf_nan() const {
  __at_align__ c10::Half tmp[size()];
  store(tmp);
  for (const auto i : c10::irange(size())) {
    if (_isnan(tmp[i]) || _isinf(tmp[i])) {
      return true;
    }
  }
  return false;
}

// 返回一个新的 Vectorized 对象，其中每个元素应用函数 f 处理后的结果
Vectorized<c10::Half> map(c10::Half (*const f)(c10::Half)) const {
  __at_align__ c10::Half tmp[size()];
  store(tmp);
  for (const auto i : c10::irange(size())) {
    tmp[i] = f(tmp[i]);
  }
  // 加载处理后的 tmp 数组到 Vectorized 对象并返回
  return loadu(tmp);
}
  // 返回一个使用绝对值的向量对象，对于值中的每个元素执行vabsq_f16操作
  return Vectorized<c10::Half>(
    vabsq_f16(values.val[0]), vabsq_f16(values.val[1]));
}

// 返回一个角度的向量对象，使用混合运算来处理小于零的值和NaN值
Vectorized<c10::Half> angle() const {
  auto zero = Vectorized<c10::Half>(0);
  auto pi = Vectorized<c10::Half>(c10::pi<c10::Half>);
  auto tmp = blendv(zero, pi, *this < zero);
  return blendv(tmp, *this, isnan());
}

// 返回实部的向量对象，即原始向量对象本身
Vectorized<c10::Half> real() const {
  return *this;
}

// 返回虚部的向量对象，即所有元素为零的向量对象
Vectorized<c10::Half> imag() const {
  return Vectorized<c10::Half>(0);
}

// 返回共轭的向量对象，即原始向量对象本身
Vectorized<c10::Half> conj() const {
  return *this;
}

// 返回acos函数应用后的向量对象，通过将向量转换为float，应用acos函数，再转回半精度
Vectorized<c10::Half> acos() const {
  return map_with_vec_float_method(&Vectorized<float>::acos);
}

// 返回acosh函数应用后的向量对象，通过将向量转换为float，应用acosh函数，再转回半精度
Vectorized<c10::Half> acosh() const {
  return map_with_vec_float_method(&Vectorized<float>::acosh);
}

// 返回asin函数应用后的向量对象，通过将向量转换为float，应用asin函数，再转回半精度
Vectorized<c10::Half> asin() const {
  return map_with_vec_float_method(&Vectorized<float>::asin);
}

// 返回atan函数应用后的向量对象，通过将向量转换为float，应用atan函数，再转回半精度
Vectorized<c10::Half> atan() const {
  return map_with_vec_float_method(&Vectorized<float>::atan);
}

// 返回atanh函数应用后的向量对象，通过将向量转换为float，应用atanh函数，再转回半精度
Vectorized<c10::Half> atanh() const {
  return map_with_vec_float_method(&Vectorized<float>::atanh);
}

// 返回atan2函数应用后的向量对象，通过将向量转换为float，应用atan2函数，再转回半精度
Vectorized<c10::Half> atan2(const Vectorized<c10::Half>& exp) const {
  return map2_with_vec_float_method(exp, &Vectorized<float>::atan2);
}

// 返回copysign函数应用后的向量对象，通过将向量转换为float，应用copysign函数，再转回半精度
Vectorized<c10::Half> copysign(const Vectorized<c10::Half>& sign) const {
  return map2_with_vec_float_method(sign, &Vectorized<float>::copysign);
}

// 返回erf函数应用后的向量对象，通过将向量转换为float，应用erf函数，再转回半精度
Vectorized<c10::Half> erf() const {
  return map_with_vec_float_method(&Vectorized<float>::erf);
}

// 返回erfc函数应用后的向量对象，通过将向量转换为float，应用erfc函数，再转回半精度
Vectorized<c10::Half> erfc() const {
  return map_with_vec_float_method(&Vectorized<float>::erfc);
}

// 返回erfinv函数应用后的向量对象，通过将向量转换为float，应用erfinv函数，再转回半精度
Vectorized<c10::Half> erfinv() const {
  return map_with_vec_float_method(&Vectorized<float>::erfinv);
}

// 返回exp函数应用后的向量对象，通过将向量转换为float，应用exp函数，再转回半精度
Vectorized<c10::Half> exp() const {
  return map_with_vec_float_method(&Vectorized<float>::exp);
}

// 返回exp2函数应用后的向量对象，通过将向量转换为float，应用exp2函数，再转回半精度
Vectorized<c10::Half> exp2() const {
  return map_with_vec_float_method(&Vectorized<float>::exp2);
}

// 返回expm1函数应用后的向量对象，通过将向量转换为float，应用expm1函数，再转回半精度
Vectorized<c10::Half> expm1() const {
  return map_with_vec_float_method(&Vectorized<float>::expm1);
}

// 返回exp_u20函数应用后的向量对象，通过将向量转换为float，应用exp_u20函数，再转回半精度
Vectorized<c10::Half> exp_u20() const {
  return map_with_vec_float_method(&Vectorized<float>::exp_u20);
}

// 返回fmod函数应用后的向量对象，通过将向量转换为float，应用fmod函数，再转回半精度
Vectorized<c10::Half> fmod(const Vectorized<c10::Half>& q) const {
  // 该函数存在精度问题，因此使用map2来处理
  return map2(q, std::fmod);
}

// 返回hypot函数应用后的向量对象，通过将向量转换为float，应用hypot函数，再转回半精度
Vectorized<c10::Half> hypot(const Vectorized<c10::Half>& b) const {
  return map2_with_vec_float_method(b, &Vectorized<float>::hypot);
}

// 返回i0函数应用后的向量对象，通过将向量转换为float，应用i0函数，再转回半精度
Vectorized<c10::Half> i0() const {
  return map_with_vec_float_method(&Vectorized<float>::i0);
}

// 返回i0e函数应用后的向量对象，通过将向量转换为float，应用i0e函数，再转回半精度
Vectorized<c10::Half> i0e() const {
  return map_with_vec_float_method(&Vectorized<float>::i0e);
}
  return map_with_vec_float_method(&Vectorized<float>::digamma);


// 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 digamma 方法进行映射操作，返回结果
Vectorized<c10::Half> digamma() const {
  return map_with_vec_float_method(&Vectorized<float>::digamma);
}



Vectorized<c10::Half> igamma(const Vectorized<c10::Half>& x) const {
  // 使用 map2_with_vec_float_method 函数，对 x 进行和 Vectorized 类的 float 版本的 igamma 方法的二元映射操作，返回结果
  return map2_with_vec_float_method(x, &Vectorized<float>::igamma);
}



Vectorized<c10::Half> igammac(const Vectorized<c10::Half>& x) const {
  // 使用 map2_with_vec_float_method 函数，对 x 进行和 Vectorized 类的 float 版本的 igammac 方法的二元映射操作，返回结果
  return map2_with_vec_float_method(x, &Vectorized<float>::igammac);
}



Vectorized<c10::Half> log() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 log 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::log);
}



Vectorized<c10::Half> log10() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 log10 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::log10);
}



Vectorized<c10::Half> log1p() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 log1p 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::log1p);
}



Vectorized<c10::Half> log2() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 log2 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::log2);
}



Vectorized<c10::Half> nextafter(const Vectorized<c10::Half>& b) const {
  // 这个函数无法与转换一起使用，因此我们使用 map2 函数
  // 使用 map2 函数，对输入参数 b 和 std::nextafter 函数进行映射操作，返回结果
  return map2(b, std::nextafter);
}



Vectorized<c10::Half> frac() const;



Vectorized<c10::Half> sin() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 sin 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::sin);
}



Vectorized<c10::Half> sinh() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 sinh 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::sinh);
}



Vectorized<c10::Half> cos() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 cos 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::cos);
}



Vectorized<c10::Half> cosh() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 cosh 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::cosh);
}



Vectorized<c10::Half> ceil() const {
  // 这个函数与转换一起使用有问题，因此我们使用 map 函数
  // 使用 map 函数，对 at::native::ceil_impl 函数进行映射操作，返回结果
  return map(at::native::ceil_impl);
}



Vectorized<c10::Half> floor() const {
  // 这个函数与转换一起使用有问题，因此我们使用 map 函数
  // 使用 map 函数，对 at::native::floor_impl 函数进行映射操作，返回结果
  return map(at::native::floor_impl);
}



Vectorized<c10::Half> neg() const {
  // 使用 vnegq_f16 函数，对 values 的两个 val 数组进行逐元素取负操作，返回结果
  return Vectorized<c10::Half>(
      vnegq_f16(values.val[0]), vnegq_f16(values.val[1]));
}



inline Vectorized<c10::Half> round() const {
  // 这个函数与转换一起使用有问题，因此我们使用 map 函数
  // 使用 map 函数，对 at::native::round_impl 函数进行映射操作，返回结果
  return map(at::native::round_impl);
}



inline Vectorized<c10::Half> tan() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 tan 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::tan);
}



inline Vectorized<c10::Half> tanh() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 tanh 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::tanh);
}



Vectorized<c10::Half> trunc() const {
  // 使用 vrndq_f16 函数，对 values 的两个 val 数组进行逐元素向零取整操作，返回结果
  float16x8_t r0 = vrndq_f16(values.val[0]);
  float16x8_t r1 = vrndq_f16(values.val[1]);
  return Vectorized<c10::Half>(r0, r1);
}



Vectorized<c10::Half> lgamma() const {
  // 调用 map_with_vec_float_method 函数，对 Vectorized 类的 float 版本的 lgamma 方法进行映射操作，返回结果
  return map_with_vec_float_method(&Vectorized<float>::lgamma);
}



Vectorized<c10::Half> sqrt() const {
  // 使用 vsqrtq_f16 函数，对 values 的两个 val 数组进行逐元素求平方根操作，返回结果
  return Vectorized<c10::Half>(
      vsqrtq_f16(values.val[0]), vsqrtq_f16(values.val[1]));
}



Vectorized<c10::Half> reciprocal() const {
  // 创建常数向量 ones，赋值为 1.0
  auto ones = vdupq_n_f16(1.0f);
  // 对 values 的两个 val 数组进行逐元素求倒数操作
  auto r0 = vdivq_f16(ones, values.val[0]);
  auto r1 = vdivq_f16(ones, values.val[1]);
  // 返回逐元素求倒数后的结果向量
  return Vectorized<c10::Half>(r0, r1);
}



Vectorized<c10::Half> rsqrt() const {
  Vectorized<c10::Half> pow(const Vectorized<c10::Half>& exp) const {
    // 使用 map2_with_vec_float_method 函数将指数应用到每个元素上
    return map2_with_vec_float_method(exp, &Vectorized<float>::pow);
  }

  Vectorized<c10::Half> operator==(const Vectorized<c10::Half>& other) const {
    // 比较两个 Vectorized<c10::Half> 对象的每个元素是否相等，返回结果为 float16x8_t 类型
    float16x8_t r0 =
        vreinterpretq_f16_u16(vceqq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vceqq_f16(values.val[1], other.values.val[1]));
    // 返回一个新的 Vectorized<c10::Half> 对象，包含比较结果的两个向量 r0 和 r1
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator!=(const Vectorized<c10::Half>& other) const {
    // 比较两个 Vectorized<c10::Half> 对象的每个元素是否不相等，返回结果为 float16x8_t 类型
    float16x8_t r0 = vreinterpretq_f16_u16(
        vmvnq_u16(vceqq_f16(values.val[0], other.values.val[0])));
    float16x8_t r1 = vreinterpretq_f16_u16(
        vmvnq_u16(vceqq_f16(values.val[1], other.values.val[1])));
    // 返回一个新的 Vectorized<c10::Half> 对象，包含比较结果的两个向量 r0 和 r1
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator<(const Vectorized<c10::Half>& other) const {
    // 比较两个 Vectorized<c10::Half> 对象的每个元素是否小于，返回结果为 float16x8_t 类型
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcltq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcltq_f16(values.val[1], other.values.val[1]));
    // 返回一个新的 Vectorized<c10::Half> 对象，包含比较结果的两个向量 r0 和 r1
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator<=(const Vectorized<c10::Half>& other) const {
    // 比较两个 Vectorized<c10::Half> 对象的每个元素是否小于或等于，返回结果为 float16x8_t 类型
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcleq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcleq_f16(values.val[1], other.values.val[1]));
    // 返回一个新的 Vectorized<c10::Half> 对象，包含比较结果的两个向量 r0 和 r1
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator>(const Vectorized<c10::Half>& other) const {
    // 比较两个 Vectorized<c10::Half> 对象的每个元素是否大于，返回结果为 float16x8_t 类型
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcgtq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcgtq_f16(values.val[1], other.values.val[1]));
    // 返回一个新的 Vectorized<c10::Half> 对象，包含比较结果的两个向量 r0 和 r1
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator>=(const Vectorized<c10::Half>& other) const {
    // 比较两个 Vectorized<c10::Half> 对象的每个元素是否大于或等于，返回结果为 float16x8_t 类型
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcgeq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcgeq_f16(values.val[1], other.values.val[1]));
    // 返回一个新的 Vectorized<c10::Half> 对象，包含比较结果的两个向量 r0 和 r1
    return Vectorized<c10::Half>(r0, r1);
  }
}; // Vectorized<Half>

// 重载 + 运算符，用于两个 Vectorized<c10::Half> 对象的加法操作
template <>
Vectorized<c10::Half> inline operator+(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 分别对低位和高位的 float16x8_t 进行加法运算
  float16x8_t r0 = vaddq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vaddq_f16(a.get_high(), b.get_high());
  // 返回新的 Vectorized<c10::Half> 对象，包含计算结果
  return Vectorized<c10::Half>(r0, r1);
}

// 重载 - 运算符，用于两个 Vectorized<c10::Half> 对象的减法操作
template <>
Vectorized<c10::Half> inline operator-(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 分别对低位和高位的 float16x8_t 进行减法运算
  float16x8_t r0 = vsubq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vsubq_f16(a.get_high(), b.get_high());
  // 返回新的 Vectorized<c10::Half> 对象，包含计算结果
  return Vectorized<c10::Half>(r0, r1);
}

// 重载 * 运算符，用于两个 Vectorized<c10::Half> 对象的乘法操作
template <>
Vectorized<c10::Half> inline operator*(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 分别对低位和高位的 float16x8_t 进行乘法运算
  float16x8_t r0 = vmulq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vmulq_f16(a.get_high(), b.get_high());
  // 返回新的 Vectorized<c10::Half> 对象，包含计算结果
  return Vectorized<c10::Half>(r0, r1);
}

// 重载 / 运算符，用于两个 Vectorized<c10::Half> 对象的除法操作
template <>
Vectorized<c10::Half> inline operator/(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 分别对低位和高位的 float16x8_t 进行除法运算
  float16x8_t r0 = vdivq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vdivq_f16(a.get_high(), b.get_high());
  // 返回新的 Vectorized<c10::Half> 对象，包含计算结果
  return Vectorized<c10::Half>(r0, r1);
}

// frac 方法的实现，返回当前对象减去截断后的结果
inline Vectorized<c10::Half> Vectorized<c10::Half>::frac() const {
  return *this - this->trunc();
}

// 重载 maximum 函数，实现 IEEE 754 201X 的 maximum 操作，处理 NaN
template <>
Vectorized<c10::Half> inline maximum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 分别对低位和高位的 float16x8_t 进行 maximum 运算
  float16x8_t r0 = vmaxq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vmaxq_f16(a.get_high(), b.get_high());
  // 返回新的 Vectorized<c10::Half> 对象，包含计算结果
  return Vectorized<c10::Half>(r0, r1);
}

// 重载 minimum 函数，实现 IEEE 754 201X 的 minimum 操作，处理 NaN
template <>
Vectorized<c10::Half> inline minimum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 分别对低位和高位的 float16x8_t 进行 minimum 运算
  float16x8_t r0 = vminq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vminq_f16(a.get_high(), b.get_high());
  // 返回新的 Vectorized<c10::Half> 对象，包含计算结果
  return Vectorized<c10::Half>(r0, r1);
}

// 重载 clamp 函数，将 a 限制在 min 和 max 之间
template <>
Vectorized<c10::Half> inline clamp(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& min,
    const Vectorized<c10::Half>& max) {
  // 先将 a 与 min 取 maximum，再与 max 取 minimum，实现限制范围内的值
  return minimum(max, maximum(min, a));
}

// 重载 clamp_max 函数，将 a 限制在不超过 max 的范围内
template <>
Vectorized<c10::Half> inline clamp_max(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& max) {
  // 将 a 与 max 取 minimum，实现限制不超过 max 的值
  return minimum(max, a);
}

// 重载 clamp_min 函数，将 a 限制在不低于 min 的范围内
template <>
Vectorized<c10::Half> inline clamp_min(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& min) {
  // 将 a 与 min 取 maximum，实现限制不低于 min 的值
  return maximum(min, a);
}

// 重载 & 运算符的部分，暂时省略
    const Vectorized<c10::Half>& b) {


// 定义一个函数，接受两个 Vectorized<c10::Half> 类型的引用参数，并返回一个 Vectorized<c10::Half> 对象
float16x8_t r0 = vreinterpretq_f16_u16(vandq_u16(
    vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));


// 创建一个 float16x8_t 类型的变量 r0，其中包含 a 的低位元素与 b 的低位元素按位与后的结果
float16x8_t r1 = vreinterpretq_f16_u16(vandq_u16(
    vreinterpretq_u16_f16(a.get_high()),
    vreinterpretq_u16_f16(b.get_high())));


// 创建一个 float16x8_t 类型的变量 r1，其中包含 a 的高位元素与 b 的高位元素按位与后的结果
return Vectorized<c10::Half>(r0, r1);


// 使用 r0 和 r1 创建一个新的 Vectorized<c10::Half> 对象，并将其作为函数的返回值
template <>
// 重载运算符 |，用于对两个半精度向量进行按位或操作
Vectorized<c10::Half> inline operator|(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 将a和b的低位和高位分别进行按位或操作，并将结果转换为半精度浮点数向量
  float16x8_t r0 = vreinterpretq_f16_u16(vorrq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(vorrq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  // 返回按位或操作后的结果向量
  return Vectorized<c10::Half>(r0, r1);
}

template <>
// 重载运算符 ^，用于对两个半精度向量进行按位异或操作
Vectorized<c10::Half> inline operator^(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  // 将a和b的低位和高位分别进行按位异或操作，并将结果转换为半精度浮点数向量
  float16x8_t r0 = vreinterpretq_f16_u16(veorq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(veorq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  // 返回按位异或操作后的结果向量
  return Vectorized<c10::Half>(r0, r1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::eq(
    const Vectorized<c10::Half>& other) const {
  // 返回当前向量与另一个向量的相等比较结果，并使用1构造新的向量
  return (*this == other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::ne(
    const Vectorized<c10::Half>& other) const {
  // 返回当前向量与另一个向量的不等比较结果，并使用1构造新的向量
  return (*this != other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::gt(
    const Vectorized<c10::Half>& other) const {
  // 返回当前向量与另一个向量的大于比较结果，并使用1构造新的向量
  return (*this > other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::ge(
    const Vectorized<c10::Half>& other) const {
  // 返回当前向量与另一个向量的大于等于比较结果，并使用1构造新的向量
  return (*this >= other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::lt(
    const Vectorized<c10::Half>& other) const {
  // 返回当前向量与另一个向量的小于比较结果，并使用1构造新的向量
  return (*this < other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::le(
    const Vectorized<c10::Half>& other) const {
  // 返回当前向量与另一个向量的小于等于比较结果，并使用1构造新的向量
  return (*this <= other) & Vectorized<c10::Half>(1);
}

template <>
// 将半精度浮点数组转换为有符号16位整型数组
inline void convert(const float16_t* src, int16_t* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 按照向量长度循环处理数据
  for (i = 0; i <= (n - Vectorized<c10::Half>::size());
       i += Vectorized<c10::Half>::size()) {
    // 将半精度浮点数组转换为有符号16位整型数组
    vst1q_s16(dst + i, vcvtq_s16_f16(vld1q_f16(src + i)));
    vst1q_s16(dst + i + 8, vcvtq_s16_f16(vld1q_f16(src + i + 8)));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 处理剩余不足一个向量长度的数据
  for (; i < n; i++) {
    dst[i] = static_cast<int16_t>(src[i]);
  }
}

template <>
// 将有符号16位整型数组转换为半精度浮点数组
inline void convert(const int16_t* src, float16_t* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 按照向量长度循环处理数据
  for (i = 0; i <= (n - Vectorized<c10::Half>::size());
       i += Vectorized<c10::Half>::size()) {
    // 将有符号16位整型数组转换为半精度浮点数组
    vst1q_f16(dst + i, vcvtq_f16_s16(vld1q_s16(src + i)));
    vst1q_f16(dst + i + 8, vcvtq_f16_s16(vld1q_s16(src + i + 8)));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 处理剩余不足一个向量长度的数据
  for (; i < n; i++) {
    dst[i] = static_cast<float16_t>(src[i]);
  }
}
    // 定义一个名为 r0 的变量，使用 vfmaq_f16 函数对两个低位元素向量进行乘累加操作
    float16x8_t r0 = vfmaq_f16(c.get_low(), a.get_low(), b.get_low());
    // 定义一个名为 r1 的变量，使用 vfmaq_f16 函数对两个高位元素向量进行乘累加操作
    float16x8_t r1 = vfmaq_f16(c.get_high(), a.get_high(), b.get_high());
    // 使用 r0 和 r1 变量创建一个新的 Vectorized<c10::Half> 对象，并返回该对象
    return Vectorized<c10::Half>(r0, r1);
}

// 结束 fmsub 的特化模板的定义

template <>
// 定义 fmsub 函数模板的特化版本，处理半精度向量
Vectorized<c10::Half> inline fmsub(
    const Vectorized<c10::Half>& a,  // 第一个半精度向量参数
    const Vectorized<c10::Half>& b,  // 第二个半精度向量参数
    const Vectorized<c10::Half>& c) {  // 第三个半精度向量参数
  // 计算低位部分的 fmsub 结果
  float16x8_t r0 = vfmsq_f16(c.get_low(), a.get_low(), b.get_low());
  // 计算高位部分的 fmsub 结果
  float16x8_t r1 = vfmsq_f16(c.get_high(), a.get_high(), b.get_high());
  // 返回合并了低位和高位结果的半精度向量
  return Vectorized<c10::Half>(r0, r1);
}

#endif /* defined(aarch64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(C10_MOBILE) */

} // namespace CPU_CAPABILITY
} // namespace at::vec


这段代码是一个特化的函数模板，用于执行半精度向量的 Fused Multiply-Subtract (fmsub) 运算。
```