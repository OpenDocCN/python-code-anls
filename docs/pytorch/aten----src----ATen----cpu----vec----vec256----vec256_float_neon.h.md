# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_float_neon.h`

```
#pragma once
// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#if defined(__aarch64__) && defined(AT_BUILD_ARM_VEC256_WITH_SLEEF)
#include <sleef.h>
#endif

// Sleef offers vectorized versions of some transcendentals
// such as sin, cos, tan etc..
// However for now opting for STL, since we are not building
// with Sleef for mobile yet.

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// Right now contains only aarch64 implementation.
// Due to follow two reasons aarch32 is not currently supported.
// 1. Due to difference in ISA been aarch32 and aarch64, intrinsics
//    that work for aarch64 dont work for aarch32.
// 2. Android NDK r21 has problems with compiling aarch32.
//    Clang seg faults.
//    https://github.com/android/ndk/issues/1248
//    https://bugs.llvm.org/show_bug.cgi?id=45824
// Most likely we will do aarch32 support with inline asm.
#if defined(__aarch64__)

#ifdef __BIG_ENDIAN__
#error "Big endian is not supported."
#endif

#if defined(AT_BUILD_ARM_VEC256_WITH_SLEEF)
// Macro to conditionally choose between Sleef and non-Sleef implementations
#define USE_SLEEF(sleef_code, non_sleef_code) sleef_code
#else
#define USE_SLEEF(sleef_code, non_sleef_code) non_sleef_code
#endif

// Template struct for blending vectors based on a mask and index
template<int index, bool mask_val>
struct BlendRegs {
  // Implementation of vector blending based on mask value
  static float32x4_t impl(
    const float32x4_t& a, const float32x4_t& b, float32x4_t& res);
};

// Specialization when mask_val is true
template<int index>
struct BlendRegs<index, true>{
  // Implementation for true mask value scenario
  static float32x4_t impl(
      const float32x4_t& a, const float32x4_t& b, float32x4_t& res) {
    return vsetq_lane_f32(vgetq_lane_f32(b, index), res, index);
  }
};

// Specialization when mask_val is false
template<int index>
struct BlendRegs<index, false>{
  // Implementation for false mask value scenario
  static float32x4_t impl(
      const float32x4_t& a, const float32x4_t& b, float32x4_t& res) {
    return vsetq_lane_f32(vgetq_lane_f32(a, index), res, index);
  }
};

// Template specialization for Vectorized class with float type
template <> class Vectorized<float> {
private:
  float32x4x2_t values; // Stores two vectors of four floats each

public:
  using value_type = float;
  using size_type = int;

  // Returns the vector size, which is 8 floats
  static constexpr size_type size() {
    return 8;
  }

  // Default constructor
  Vectorized() {}

  // Constructor initializing with a vector of two vectors of four floats each
  Vectorized(float32x4x2_t v) : values(v) {}

  // Constructor initializing with a single float value replicated across all elements
  Vectorized(float val) : values{vdupq_n_f32(val), vdupq_n_f32(val) } {}

  // Constructor initializing with individual float values for each element
  Vectorized(float val0, float val1, float val2, float val3,
         float val4, float val5, float val6, float val7) :
         values{val0, val1, val2, val3, val4, val5, val6, val7} {}

  // Constructor initializing with two vectors of four floats each
  Vectorized(float32x4_t val0, float32x4_t val1) : values{val0, val1} {}

  // Conversion operator to convert Vectorized object to float32x4x2_t type
  operator float32x4x2_t() const {
    return values;
  }

  // Template function for blending two Vectorized<float> objects based on a mask
  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    Vectorized<float> vec;

    // Blend each lane of the vectors based on the mask value
    vec.values.val[0] =
      BlendRegs<0, (mask & 0x01)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] =
      BlendRegs<1, (mask & 0x02)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] =
      BlendRegs<2, (mask & 0x04)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    // 使用 BlendRegs 模板类的实现，根据 mask 的第 3 位（0x04）的值选择性地混合 a 和 b 向量的第一个元素，并将结果存入 vec 的第一个元素

    vec.values.val[0] =
      BlendRegs<3, (mask & 0x08)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    // 使用 BlendRegs 模板类的实现，根据 mask 的第 4 位（0x08）的值选择性地混合 a 和 b 向量的第一个元素（覆盖上一个操作的结果），并将结果存入 vec 的第一个元素

    // 1.
    vec.values.val[1] =
      BlendRegs<0, (mask & 0x10)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    // 使用 BlendRegs 模板类的实现，根据 mask 的第 5 位（0x10）的值选择性地混合 a 和 b 向量的第二个元素，并将结果存入 vec 的第二个元素

    vec.values.val[1] =
      BlendRegs<1, (mask & 0x20)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    // 使用 BlendRegs 模板类的实现，根据 mask 的第 6 位（0x20）的值选择性地混合 a 和 b 向量的第二个元素（覆盖上一个操作的结果），并将结果存入 vec 的第二个元素

    vec.values.val[1] =
      BlendRegs<2, (mask & 0x40)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    // 使用 BlendRegs 模板类的实现，根据 mask 的第 7 位（0x40）的值选择性地混合 a 和 b 向量的第二个元素（覆盖上一个操作的结果），并将结果存入 vec 的第二个元素

    vec.values.val[1] =
      BlendRegs<3, (mask & 0x80)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    // 使用 BlendRegs 模板类的实现，根据 mask 的第 8 位（0x80）的值选择性地混合 a 和 b 向量的第二个元素（覆盖上一个操作的结果），并将结果存入 vec 的第二个元素

    return vec;
  }
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    // TODO
    // NB: This requires that each value, i.e., each uint value,
    // of the mask either all be zeros or all be 1s.
    // We perhaps need some kind of an assert?
    // But that will affect performance.
    // 根据掩码 mask 中的每个值（每个 uint 值）要么全为零要么全为 1，选择性地混合向量 a 和 b，并返回结果向量
    Vectorized<float> vec(mask.values);
    vec.values.val[0] = vbslq_f32(
        vreinterpretq_u32_f32(vec.values.val[0]),
        b.values.val[0],
        a.values.val[0]);
    // 根据 vec 的第一个元素作为掩码，选择性地在向量 b 和 a 之间混合，并将结果存入 vec 的第一个元素

    vec.values.val[1] = vbslq_f32(
        vreinterpretq_u32_f32(vec.values.val[1]),
        b.values.val[1],
        a.values.val[1]);
    // 根据 vec 的第二个元素作为掩码，选择性地在向量 b 和 a 之间混合，并将结果存入 vec 的第二个元素

    return vec;
  }
  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    const Vectorized<float> base_vec(base);
    const Vectorized<float> step_vec(step);
    const Vectorized<float> step_sizes(0, 1, 2, 3, 4, 5, 6, 7);
    // 返回一个向量，该向量包含基础值加上步长乘以步长大小的每个元素
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
                           int64_t count = size()) {
    switch (count) {
      case 0:
        // 如果 count 为 0，直接返回向量 a
        return a;
      case 1:
        {
          // 创建一个 Vectorized<float> 对象 vec
          Vectorized<float> vec;
          // 静态初始化一个 uint32x4_t 类型的低位掩码 mask_low
          static uint32x4_t mask_low = {0xFFFFFFFF, 0x0, 0x0, 0x0};
          // 将 mask_low 转换为 float 后赋值给 vec 的第一个值
          vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
          // 复制 a 的第二个值给 vec 的第二个值
          vec.values.val[1] = a.values.val[1];
          // 使用条件选择指令 vbslq_f32，根据 mask_low 对应位置的条件选择 b 或 a，并赋值给 vec 的第一个值
          vec.values.val[0] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[0]),
              b.values.val[0],
              a.values.val[0]);
          // 返回 vec
          return vec;
        }
      case 2:
        {
          // 创建一个 Vectorized<float> 对象 vec
          Vectorized<float> vec;
          // 静态初始化一个 uint32x4_t 类型的低位掩码 mask_low
          static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
          // 将 mask_low 转换为 float 后赋值给 vec 的第一个值
          vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
          // 复制 a 的第二个值给 vec 的第二个值
          vec.values.val[1] = a.values.val[1];
          // 使用条件选择指令 vbslq_f32，根据 mask_low 对应位置的条件选择 b 或 a，并赋值给 vec 的第一个值
          vec.values.val[0] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[0]),
              b.values.val[0],
              a.values.val[0]);
          // 返回 vec
          return vec;
        }
      case 3:
        {
          // 创建一个 Vectorized<float> 对象 vec
          Vectorized<float> vec;
          // 静态初始化一个 uint32x4_t 类型的低位掩码 mask_low
          static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
          // 将 mask_low 转换为 float 后赋值给 vec 的第一个值
          vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
          // 复制 a 的第二个值给 vec 的第二个值
          vec.values.val[1] = a.values.val[1];
          // 使用条件选择指令 vbslq_f32，根据 mask_low 对应位置的条件选择 b 或 a，并赋值给 vec 的第一个值
          vec.values.val[0] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[0]),
              b.values.val[0],
              a.values.val[0]);
          // 返回 vec
          return vec;
        }
      case 4:
        // 如果 count 为 4，返回一个 Vectorized<float> 对象，以 b 的第一个值和 a 的第二个值构造
        return Vectorized<float>(b.values.val[0], a.values.val[1]);
      case 5:
        {
          // 创建一个 Vectorized<float> 对象 vec
          Vectorized<float> vec;
          // 静态初始化一个 uint32x4_t 类型的高位掩码 mask_high
          static uint32x4_t mask_high = {0xFFFFFFFF, 0x0, 0x0, 0x0};
          // 将 b 的第一个值赋给 vec 的第一个值
          vec.values.val[0] = b.values.val[0];
          // 将 mask_high 转换为 float 后赋值给 vec 的第二个值
          vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
          // 使用条件选择指令 vbslq_f32，根据 mask_high 对应位置的条件选择 b 的第二个值或 a 的第二个值，并赋值给 vec 的第二个值
          vec.values.val[1] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[1]),
              b.values.val[1],
              a.values.val[1]);
          // 返回 vec
          return vec;
        }
      case 6:
        {
          // 创建一个 Vectorized<float> 对象 vec
          Vectorized<float> vec;
          // 静态初始化一个 uint32x4_t 类型的高位掩码 mask_high
          static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
          // 将 b 的第一个值赋给 vec 的第一个值
          vec.values.val[0] = b.values.val[0];
          // 将 mask_high 转换为 float 后赋值给 vec 的第二个值
          vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
          // 使用条件选择指令 vbslq_f32，根据 mask_high 对应位置的条件选择 b 的第二个值或 a 的第二个值，并赋值给 vec 的第二个值
          vec.values.val[1] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[1]),
              b.values.val[1],
              a.values.val[1]);
          // 返回 vec
          return vec;
        }
      case 7:
        {
          // 创建一个 Vectorized<float> 对象 vec
          Vectorized<float> vec;
          // 静态初始化一个 uint32x4_t 类型的高位掩码 mask_high
          static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
          // 将 b 的第一个值赋给 vec 的第一个值
          vec.values.val[0] = b.values.val[0];
          // 将 mask_high 转换为 float 后赋值给 vec 的第二个值
          vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
          // 使用条件选择指令 vbslq_f32，根据 mask_high 对应位置的条件选择 b 的第二个值或 a 的第二个值，并赋值给 vec 的第二个值
          vec.values.val[1] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[1]),
              b.values.val[1],
              a.values.val[1]);
          // 返回 vec
          return vec;
        }
    }
    // 默认返回向量 b
    return b;
  }
  // 静态方法 loadu，加载非对齐内存中的数据，参数 ptr 指向要加载的内存地址，count 为加载的元素数量，默认为 size()
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    // 如果 count 等于 size()
    if (count == size()) {
      // 使用 vld1q_f32_x2 加载 ptr 指向的两个 float 值，并返回作为 Vectorized<float> 对象
      return vld1q_f32_x2(reinterpret_cast<const float*>(ptr));
    }
    // 如果 count 等于 size 的一半
    else if (count == (size() >> 1)) {
      // 创建一个 Vectorized 对象 res，用于存储结果
      Vectorized<float> res;
      // 将 ptr 强制转换为 float 指针后加载到 res 的第一个值
      res.values.val[0] = vld1q_f32(reinterpret_cast<const float*>(ptr));
      // 将 res 的第二个值设为全零
      res.values.val[1] = vdupq_n_f32(0.f);
      // 返回结果 res
      return res;
    }
    // 如果 count 不等于 size 的一半
    else {
      // 创建临时数组 tmp_values，大小为 size()
      __at_align__ float tmp_values[size()];
      // 将 tmp_values 数组的所有元素初始化为 0.0
      for (const auto i : c10::irange(size())) {
        tmp_values[i] = 0.0;
      }
      // 将 ptr 解释为 float 指针后的 count 个 float 值复制到 tmp_values 中
      std::memcpy(
          tmp_values,
          reinterpret_cast<const float*>(ptr),
          count * sizeof(float));
      // 返回以 tmp_values 数组创建的 Vectorized 对象
      return vld1q_f32_x2(reinterpret_cast<const float*>(tmp_values));
    }
  }
  // 将数据存储到指定内存位置 ptr，存储数量为 count，默认为 size()
  void store(void* ptr, int64_t count = size()) const {
    // 如果 count 等于 size()
    if (count == size()) {
      // 将 values 的数据存储到 ptr 指向的内存中
      vst1q_f32_x2(reinterpret_cast<float*>(ptr), values);
    }
    // 如果 count 等于 size() 的一半
    else if (count == (size() >> 1)) {
      // 将 values 的第一个元素存储到 ptr 指向的内存中
      vst1q_f32(reinterpret_cast<float*>(ptr), values.val[0]);
    }
    // 其他情况
    else {
      // 创建临时数组 tmp_values，大小为 size()
      float tmp_values[size()];
      // 将 values 的数据存储到 tmp_values 数组中
      vst1q_f32_x2(reinterpret_cast<float*>(tmp_values), values);
      // 将 tmp_values 数组的前 count 个 float 值复制到 ptr 指向的内存中
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  // 返回 values 的第一个元素的常量引用
  inline const float32x4_t& get_low() const {
    return values.val[0];
  }
  // 返回 values 的第一个元素的引用
  inline float32x4_t& get_low() {
    return values.val[0];
  }
  // 返回 values 的第二个元素的常量引用
  inline const float32x4_t& get_high() const {
    return values.val[1];
  }
  // 返回 values 的第二个元素的引用
  inline float32x4_t& get_high() {
    return values.val[1];
  }
  // 索引运算符重载，返回指定索引处的元素
  // 注意：实现非常慢，仅因 vec256_qint 引用了此函数而存在
  // 一旦我们为 ARM 优化该实现，应删除此函数。TODO (kimishpatel)
  float operator[](int idx) const {
    // 创建临时数组 tmp，大小为 size()
    __at_align__ float tmp[size()];
    // 将 Vectorized 对象的数据存储到 tmp 中
    store(tmp);
    // 返回 tmp 数组中指定索引处的值
    return tmp[idx];
  }
  // 非常慢的索引运算符重载，返回指定索引处的元素
  float operator[](int idx) {
    // 创建临时数组 tmp，大小为 size()
    __at_align__ float tmp[size()];
    // 将 Vectorized 对象的数据存储到 tmp 中
    store(tmp);
    // 返回 tmp 数组中指定索引处的值
    return tmp[idx];
  }
  // 返回零掩码，指示哪些元素为零
  // 在布尔版本中，我们希望判断是否有任何 1 或全为零等情况时可以更快地完成
  int zero_mask() const {
    // 创建临时数组 tmp，大小为 size()
    __at_align__ float tmp[size()];
    // 将 Vectorized 对象的数据存储到 tmp 中
    store(tmp);
    // 初始化 mask 为 0
    int mask = 0;
    // 遍历 tmp 数组
    for (int i = 0; i < size(); ++ i) {
      // 如果 tmp 数组中的元素等于 0.0
      if (tmp[i] == 0.f) {
        // 将对应位置的掩码位置为 1
        mask |= (1 << i);
      }
    }
    // 返回零掩码
    return mask;
  }
  // 检查是否存在 Inf 或 NaN
  bool has_inf_nan() const {
    // 创建临时数组 tmp，大小为 size()
    __at_align__ float tmp[size()];
    // 将 Vectorized 对象的数据存储到 tmp 中
    store(tmp);
    // 遍历 tmp 数组
    for (const auto i : c10::irange(size())) {
      // 如果 tmp 数组中的元素是 NaN 或 Inf
      if(_isnan(tmp[i]) || _isinf(tmp[i])) {
        // 返回 true
        return true;
      }
    }
    // 返回 false
    return false;
  }
  // 使用指定函数 f 映射每个元素，并返回结果
  Vectorized<float> map(float (*const f)(float)) const {
    // 创建临时数组 tmp，大小为 size()
    __at_align__ float tmp[size()];
    // 将 Vectorized 对象的数据存储到 tmp 中
    store(tmp);
    // 对 tmp 数组中的每个元素应用函数 f
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    // 返回经过映射后的 Vectorized 对象
    return loadu(tmp);
  }
  // 返回每个元素的绝对值
  Vectorized<float> abs() const {
  Vectorized<float> angle() const {
    // 创建一个包含零的向量
    auto zero = Vectorized<float>(0);
    // 创建一个包含 pi 值的向量
    auto pi = Vectorized<float>(c10::pi<float>);
    // 根据当前向量的符号，选择零或 pi 的值，生成一个新的向量
    auto tmp = blendv(zero, pi, *this < zero);
    // 如果当前向量中存在 NaN 值，则选择当前向量的值，否则选择之前生成的 tmp 值
    return blendv(tmp, *this, isnan());
  }
  Vectorized<float> real() const {
    // 返回当前向量的实部，即当前向量本身
    return *this;
  }
  Vectorized<float> imag() const {
    // 返回当前向量的虚部，即一个所有元素值为 0 的向量
    return Vectorized<float>(0.f);
  }
  Vectorized<float> conj() const {
    // 返回当前向量的共轭，即当前向量本身
    return *this;
  }
  Vectorized<float> acos() const {
    // 使用 Sleef 库计算 acos 函数的结果，并返回对应的向量
    return USE_SLEEF(
      Vectorized<float>(Sleef_acosf4_u10(values.val[0]), Sleef_acosf4_u10(values.val[1])),
      map(std::acos)
    );
  }
  Vectorized<float> acosh() const {
    // 使用 Sleef 库计算 acosh 函数的结果，并返回对应的向量
    return USE_SLEEF(
      Vectorized<float>(Sleef_acoshf4_u10(values.val[0]), Sleef_acoshf4_u10(values.val[1])),
      map(std::acosh)
    );
  }
  Vectorized<float> asin() const {
    // 使用 Sleef 库计算 asin 函数的结果，并返回对应的向量
    return USE_SLEEF(
      Vectorized<float>(Sleef_asinf4_u10(values.val[0]), Sleef_asinf4_u10(values.val[1])),
      map(std::asin)
    );
  }
  Vectorized<float> atan() const {
    // 使用 Sleef 库计算 atan 函数的结果，并返回对应的向量
    return USE_SLEEF(
      Vectorized<float>(Sleef_atanf4_u10(values.val[0]), Sleef_atanf4_u10(values.val[1])),
      map(std::atan)
    );
  }
  Vectorized<float> atanh() const {
    // 使用 Sleef 库计算 atanh 函数的结果，并返回对应的向量
    return USE_SLEEF(
      Vectorized<float>(Sleef_atanhf4_u10(values.val[0]), Sleef_atanhf4_u10(values.val[1])),
      map(std::atanh)
    );
  }
  Vectorized<float> atan2(const Vectorized<float> &exp) const {
    // 使用 Sleef 库计算 atan2 函数的结果，或者使用标准库的 std::atan2 函数
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_atan2f4_u10(values.val[0], exp.values.val[0]),
                                 Sleef_atan2f4_u10(values.val[1], exp.values.val[1]));
      },
      {
        // 创建临时数组存储当前向量和参数向量的数据
        __at_align__ float tmp[size()];
        __at_align__ float tmp_exp[size()];
        store(tmp);
        exp.store(tmp_exp);
        // 遍历数组，计算每个元素的 atan2 结果并存储到临时数组中
        for (const auto i : c10::irange(size())) {
          tmp[i] = std::atan2(tmp[i], tmp_exp[i]);
        }
        // 加载临时数组数据并返回对应的向量
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    // 使用 Sleef 库计算 copysign 函数的结果，或者使用标准库的 std::copysign 函数
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_copysignf4(values.val[0], sign.values.val[0]),
                                 Sleef_copysignf4(values.val[1], sign.values.val[1]));
      },
      {
        // 创建临时数组存储当前向量和参数向量的数据
        __at_align__ float tmp[size()];
        __at_align__ float tmp_sign[size()];
        store(tmp);
        sign.store(tmp_sign);
        // 遍历数组，计算每个元素的 copysign 结果并存储到临时数组中
        for (size_type i = 0; i < size(); i++) {
          tmp[i] = std::copysign(tmp[i], tmp_sign[i]);
        }
        // 加载临时数组数据并返回对应的向量
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> erf() const;
  Vectorized<float> erfc() const {
    // 使用 Sleef 库计算 erfc 函数的结果，或者使用标准库的 std::erfc 函数
    return USE_SLEEF(
      Vectorized<float>(Sleef_erfcf4_u15(values.val[0]), Sleef_erfcf4_u15(values.val[1])),
      map(std::erfc)
    );
  }
  Vectorized<float> erfinv() const {
    // 使用 map 函数调用 calc_erfinv 来计算 erfinv 函数的结果，并返回对应的向量
    return map(calc_erfinv);
  }
  Vectorized<float> exp() const {
    // 使用 Sleef 库计算 exp 函数的结果，或者使用标准库的 std::exp 函数
    return USE_SLEEF(
      Vectorized<float>(Sleef_expf4_u10(values.val[0]), Sleef_expf4_u10(values.val[1])),
      map(std::exp)
    );
  }
  Vectorized<float> log2() const {
    // 如果启用 Sleef，则使用 Sleef 库计算两个向量的对数
    return USE_SLEEF(
      Vectorized<float>(Sleef_log2f4_u10(values.val[0]), Sleef_log2f4_u10(values.val[1])),
      // 否则，使用标准库中的 log2 函数逐元素计算向量的对数
      map(std::log2)
    );
  }
    return USE_SLEEF(
      Vectorized<float>(Sleef_log2f4_u10(values.val[0]), Sleef_log2f4_u10(values.val[1])),
      map(std::log2)
    );
  }


    // 返回使用SLEEF库计算的log2的向量化结果
    Vectorized<float> nextafter(const Vectorized<float> &b) const {
      USE_SLEEF(
        {
          // 使用SLEEF库计算两个向量之间每个元素的下一个浮点数
          return Vectorized<float>(Sleef_nextafterf4(values.val[0], b.values.val[0]),
                                   Sleef_nextafterf4(values.val[1], b.values.val[1]));
        },
        {
          // 创建临时数组存储当前向量和参数向量的值
          __at_align__ float tmp[size()];
          __at_align__ float tmp_b[size()];
          store(tmp);
          b.store(tmp_b);
          // 对每个元素进行std::nextafter操作
          for (const auto i : c10::irange(size())) {
            tmp[i] = std::nextafter(tmp[i], tmp_b[i]);
          }
          // 加载更新后的值返回向量
          return loadu(tmp);
        }
      )
    }


    Vectorized<float> frac() const;


    // 返回使用SLEEF库计算的sin的向量化结果
    Vectorized<float> sin() const {
      return USE_SLEEF(
        Vectorized<float>(Sleef_sinf4_u10(values.val[0]), Sleef_sinf4_u10(values.val[1])),
        map(std::sin)
      );
    }


    // 返回使用SLEEF库计算的sinh的向量化结果
    Vectorized<float> sinh() const {
      return USE_SLEEF(
        Vectorized<float>(Sleef_sinhf4_u10(values.val[0]), Sleef_sinhf4_u10(values.val[1])),
        map(std::sinh)
      );
    }


    // 返回使用SLEEF库计算的cos的向量化结果
    Vectorized<float> cos() const {
      return USE_SLEEF(
        Vectorized<float>(Sleef_cosf4_u10(values.val[0]), Sleef_cosf4_u10(values.val[1])),
        map(std::cos)
      );
    }


    // 返回使用SLEEF库计算的cosh的向量化结果
    Vectorized<float> cosh() const {
      return USE_SLEEF(
        Vectorized<float>(Sleef_coshf4_u10(values.val[0]), Sleef_coshf4_u10(values.val[1])),
        map(std::cosh)
      );
    }


    // 返回向量元素应用ceil_impl函数后的结果
    Vectorized<float> ceil() const {
      return map(at::native::ceil_impl);
    }


    // 返回向量元素应用floor_impl函数后的结果
    Vectorized<float> floor() const {
      return map(at::native::floor_impl);
    }


    // 返回元素取反后的向量
    Vectorized<float> neg() const {
      return Vectorized<float>(
          vnegq_f32(values.val[0]),
          vnegq_f32(values.val[1]));
    }


    // 返回向量元素应用round_impl函数后的结果
    Vectorized<float> round() const {
      // 我们不使用std::round，因为我们希望将中间值四舍五入到最近的偶数整数
      return map(at::native::round_impl);
    }


    // 返回使用SLEEF库计算的tan的向量化结果
    Vectorized<float> tan() const {
      return USE_SLEEF(
        Vectorized<float>(Sleef_tanf4_u10(values.val[0]), Sleef_tanf4_u10(values.val[1])),
        map(std::tan)
      );
    }


    // 返回使用SLEEF库计算的tanh的向量化结果
    Vectorized<float> tanh() const {
      return USE_SLEEF(
        Vectorized<float>(Sleef_tanhf4_u10(values.val[0]), Sleef_tanhf4_u10(values.val[1])),
        map(std::tanh)
      );
    }


    // 返回向量元素应用vrndq_f32函数后的结果，即四舍五入到最近的整数
    Vectorized<float> trunc() const {
      float32x4_t r0 = vrndq_f32(values.val[0]);
      float32x4_t r1 = vrndq_f32(values.val[1]);
      return Vectorized<float>(r0, r1);
    }


    // 返回使用SLEEF库计算的lgamma的向量化结果
    Vectorized<float> lgamma() const {
      return USE_SLEEF(
        Vectorized<float>(Sleef_lgammaf4_u10(values.val[0]), Sleef_lgammaf4_u10(values.val[1])),
        map(std::lgamma)
      );
    }


    // 返回向量元素应用vsqrtq_f32函数后的结果，即平方根
    Vectorized<float> sqrt() const {
      return Vectorized<float>(
          vsqrtq_f32(values.val[0]),
          vsqrtq_f32(values.val[1]));
    }


    // 返回向量元素的倒数
    Vectorized<float> reciprocal() const {
      auto r0 = vdivq_f32(vdupq_n_f32(1.0f), values.val[0]);
      auto r1 = vdivq_f32(vdupq_n_f32(1.0f), values.val[1]);
  // 返回一个新的 Vectorized<float> 对象，其中每个元素是 this 对象与 other 对象对应元素是否相等的比较结果
  Vectorized<float> eq(const Vectorized<float>& other) const;
  // 返回一个新的 Vectorized<float> 对象，其中每个元素是 this 对象与 other 对象对应元素是否不相等的比较结果
  Vectorized<float> ne(const Vectorized<float>& other) const;
  // 返回一个新的 Vectorized<float> 对象，其中每个元素是 this 对象与 other 对象对应元素是否大于的比较结果
  Vectorized<float> gt(const Vectorized<float>& other) const;
  // 返回一个新的 Vectorized<float> 对象，其中每个元素是 this 对象与 other 对象对应元素是否大于等于的比较结果
  Vectorized<float> ge(const Vectorized<float>& other) const;
  // 返回一个新的 Vectorized<float> 对象，其中每个元素是 this 对象与 other 对象对应元素是否小于的比较结果
  Vectorized<float> lt(const Vectorized<float>& other) const;
  // 返回一个新的 Vectorized<float> 对象，其中每个元素是 this 对象与 other 对象对应元素是否小于等于的比较结果
  Vectorized<float> le(const Vectorized<float>& other) const;
// 结束模板特化定义
};

// 定义模板特化：向量化浮点数加法运算符重载
template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 使用 NEON 指令实现低位和高位向量的浮点数加法
  float32x4_t r0 = vaddq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vaddq_f32(a.get_high(), b.get_high());
  // 返回结果向量
  return Vectorized<float>(r0, r1);
}

// 定义模板特化：向量化浮点数减法运算符重载
template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 使用 NEON 指令实现低位和高位向量的浮点数减法
  float32x4_t r0 = vsubq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vsubq_f32(a.get_high(), b.get_high());
  // 返回结果向量
  return Vectorized<float>(r0, r1);
}

// 定义模板特化：向量化浮点数乘法运算符重载
template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 使用 NEON 指令实现低位和高位向量的浮点数乘法
  float32x4_t r0 = vmulq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vmulq_f32(a.get_high(), b.get_high());
  // 返回结果向量
  return Vectorized<float>(r0, r1);
}

// 定义模板特化：向量化浮点数除法运算符重载
template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 使用 NEON 指令实现低位和高位向量的浮点数除法
  float32x4_t r0 = vdivq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vdivq_f32(a.get_high(), b.get_high());
  // 返回结果向量
  return Vectorized<float>(r0, r1);
}

// 实现 frac 函数，使用减法实现截断
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// 实现 IEEE 754 201X 中的最大值操作，如果有 NaN 则传播 NaN
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 使用 NEON 指令实现低位和高位向量的最大值操作
  float32x4_t r0 = vmaxq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vmaxq_f32(a.get_high(), b.get_high());
  // 返回结果向量
  return Vectorized<float>(r0, r1);
}

// 实现 IEEE 754 201X 中的最小值操作，如果有 NaN 则传播 NaN
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 使用 NEON 指令实现低位和高位向量的最小值操作
  float32x4_t r0 = vminq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vminq_f32(a.get_high(), b.get_high());
  // 返回结果向量
  return Vectorized<float>(r0, r1);
}

// 实现 clamp 函数，使用 minimum 和 maximum 函数实现范围约束
template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return minimum(max, maximum(min, a));
}

// 实现 clamp_max 函数，使用 minimum 函数实现上限约束
template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return minimum(max, a);
}

// 实现 clamp_min 函数，使用 maximum 函数实现下限约束
template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return maximum(min, a);
}

// 定义模板特化：向量化浮点数按位与运算符重载
template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 使用 NEON 指令实现低位和高位向量的浮点数按位与运算
  float32x4_t r0 = vreinterpretq_f32_u32(vandq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  float32x4_t r1 = vreinterpretq_f32_u32(vandq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  // 返回结果向量
  return Vectorized<float>(r0, r1);
}
// 定义浮点类型的向量化按位或运算符重载函数
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 对低四个浮点数进行按位或操作，并将结果重新解释为浮点数向量
  float32x4_t r0 = vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  // 对高四个浮点数进行按位或操作，并将结果重新解释为浮点数向量
  float32x4_t r1 = vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  // 返回由结果向量构造的浮点数向量对象
  return Vectorized<float>(r0, r1);
}

// 特化模板，定义浮点类型的向量化按位异或运算符重载函数
template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 对低四个浮点数进行按位异或操作，并将结果重新解释为浮点数向量
  float32x4_t r0 = vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  // 对高四个浮点数进行按位异或操作，并将结果重新解释为浮点数向量
  float32x4_t r1 = vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  // 返回由结果向量构造的浮点数向量对象
  return Vectorized<float>(r0, r1);
}

// 定义浮点类型的向量化相等比较函数
inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
  // 对当前对象与给定对象进行相等比较，并与全为1的向量进行按位与操作
  return (*this == other) & Vectorized<float>(1.0f);
}

// 定义浮点类型的向量化不等比较函数
inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
  // 对当前对象与给定对象进行不等比较，并与全为1的向量进行按位与操作
  return (*this != other) & Vectorized<float>(1.0f);
}

// 定义浮点类型的向量化大于比较函数
inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
  // 对当前对象与给定对象进行大于比较，并与全为1的向量进行按位与操作
  return (*this > other) & Vectorized<float>(1.0f);
}

// 定义浮点类型的向量化大于等于比较函数
inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
  // 对当前对象与给定对象进行大于等于比较，并与全为1的向量进行按位与操作
  return (*this >= other) & Vectorized<float>(1.0f);
}

// 定义浮点类型的向量化小于比较函数
inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
  // 对当前对象与给定对象进行小于比较，并与全为1的向量进行按位与操作
  return (*this < other) & Vectorized<float>(1.0f);
}

// 定义浮点类型的向量化小于等于比较函数
inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
  // 对当前对象与给定对象进行小于等于比较，并与全为1的向量进行按位与操作
  return (*this <= other) & Vectorized<float>(1.0f);
}

// 特化模板，定义将浮点数组转换为整数数组的函数
template <>
inline void convert(const float* src, int32_t* dst, int64_t n) {
  int64_t i;
  // 循环处理数组，每次处理 Vectorized<float> 的大小
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    // 将加载的浮点数向量转换为整数向量，并存储到目标数组中
    vst1q_s32(dst + i, vcvtq_s32_f32(vld1q_f32(src + i)));
    vst1q_s32(dst + i + 4, vcvtq_s32_f32(vld1q_f32(src + i + 4)));
  }
  // 处理剩余的元素，转换成整数后存储到目标数组中
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<int32_t>(src[i]);
  }
}

// 特化模板，定义将整数数组转换为浮点数组的函数
template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  int64_t i;
  // 循环处理数组，每次处理 Vectorized<float> 的大小
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    // 将加载的整数向量转换为浮点数向量，并存储到目标数组中
    vst1q_f32(dst + i, vcvtq_f32_s32(vld1q_s32(src + i)));
    vst1q_f32(dst + i + 4, vcvtq_f32_s32(vld1q_s32(src + i + 4)));
  }
  // 处理剩余的元素，转换成浮点数后存储到目标数组中
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

// 特化模板，定义浮点类型的向量化 Fused Multiply-Add 操作函数
template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  // 对低四个浮点数执行 Fused Multiply-Add 操作，并得到结果向量
  float32x4_t r0 = vfmaq_f32(c.get_low(), a.get_low(), b.get_low());
  // 对高四个浮点数执行 Fused Multiply-Add 操作，并得到结果向量
  float32x4_t r1 = vfmaq_f32(c.get_high(), a.get_high(), b.get_high());
  // 返回由结果向量构造的浮点数向量对象
  return Vectorized<float>(r0, r1);
}
// 定义一个模板函数 fmsub，用于执行向量化的浮点数减法和乘法
Vectorized<float> inline fmsub(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  // 对低位和高位分别执行 vfmsq_f32 操作，并将结果存储在 r0 和 r1 中
  float32x4_t r0 = vfmsq_f32(c.get_low(), a.get_low(), b.get_low());
  float32x4_t r1 = vfmsq_f32(c.get_high(), a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

// 实现 Vectorized<float> 类的 erf 方法
inline Vectorized<float> Vectorized<float>::erf() const{
    // 定义常量向量
    const Vectorized<float> neg_zero_vec(-0.f); // 负零向量
    const Vectorized<float> one_vec(1.0f);      // 值为 1 的向量
    const Vectorized<float> p(0.3275911f);      // 常数 p
    const Vectorized<float> p1(0.254829592f);   // 常数 p1
    const Vectorized<float> p2(-0.284496736f);  // 常数 p2
    const Vectorized<float> p3(1.421413741f);   // 常数 p3
    const Vectorized<float> p4(-1.453152027f);  // 常数 p4
    const Vectorized<float> p5(1.061405429f);   // 常数 p5
    
    // 计算符号函数 sign(x)，使用负零向量进行按位与运算
    auto sign_mask = neg_zero_vec & *this;
    auto abs_vec = this->abs(); // 计算绝对值向量
    
    // 计算 t = 1 / (p * abs(x) + 1)
    auto tmp0 = fmadd(p, abs_vec, one_vec);
    auto t = one_vec / tmp0;
    
    // 计算 r = p5 * t ^ 4 + p4 * t ^ 3 + p3 * t ^ 2 + p2 * t + p1
    auto tmp1 = fmadd(p5, t, p4);
    auto tmp2 = fmadd(tmp1, t, p3);
    auto tmp3 = fmadd(tmp2, t, p2);
    auto r = fmadd(tmp3, t, p1);
    
    // 计算 -exp(- x * x)，其中 x 是当前向量对象
    auto pow_2 = (*this) * (*this); // 计算平方
    auto neg_pow_2 = pow_2 ^ neg_zero_vec; // 取反操作
    auto tmp4 = neg_pow_2.map(std::exp); // 对每个元素应用指数函数，可以优化为更快的指数函数实现
    auto tmp5 = tmp4 ^ neg_zero_vec; // 取反操作
    
    // 计算 erf(x) = sign(x) * (1 - r * t * exp(- x * x))
    auto tmp6 = t * tmp5;
    auto tmp7 = fmadd(tmp6, r, one_vec);
    return tmp7 ^ sign_mask; // 返回最终结果，使用符号向量调整符号
}
#endif /* defined(aarch64) */

}} // namespace at::vec::CPU_CAPABILITY
```