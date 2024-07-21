# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_qint8_vsx.h`

```py
// 一旦这个头文件被包含，编译器会确保本文件内容只被包含一次，避免重复定义
#pragma once

// 引入向量指令集的头文件，用于优化向量化操作
#include <ATen/cpu/vec/intrinsics.h>
// 向量化操作的基类定义
#include <ATen/cpu/vec/vec_base.h>
// 引入 VSX 架构的 VSX 助手函数
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
// 引入 qint8 类型的定义
#include <c10/util/qint8.h>
// 引入数组容器的支持
#include <array>

// 此文件定义了量化类型的 Vectorized<> 模板特化版本。

// 以下为此特化版本的说明文档
//
//
// 目前，我们简单地使用这些类作为量化类型和 Vectorized<float> 之间的高效转换器，
// 通常在带宽受限情况下使用完整精度进行算术操作是可接受的（例如逐元素操作）。
//
//
// 转换规则如下：
//  Vectorized<qint8> -> 4x Vectorized<float>
//
// 返回的浮点向量的大小由特殊的 constexpr 函数 float_num_vecs 指定。
// 从反量化（dequantize）返回的值类型（以及作为量化（quantize）参数的预期类型）
// 由 float_vec_return_type 指定。
//
// 在编写使用这些向量的内核时，预期将在循环中执行浮点操作，迭代次数为 Vectorized<T>::float_num_vecs。

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

// Vectorized<> 模板的特化声明，专门用于 c10::qint8 类型
template <>
struct Vectorized<c10::qint8> {
 private:
  // 使用联合体确保向量和布尔向量可以共享内存
  union {
    struct {
      vint8 _vec0;   // 第一个 qint8 向量
      vint8 _vec1;   // 第二个 qint8 向量
    };
    struct {
      vbool8 _vecb0; // 第一个布尔向量
      vbool8 _vecb1; // 第二个布尔向量
    };

  } __attribute__((__may_alias__));  // 使用 __may_alias__ 提示编译器允许别名访问

 public:
  Vectorized() {}  // 默认构造函数

  using size_type = int;
  static constexpr size_type size() {
    return 32;  // 返回向量的大小
  }

  static constexpr size_t float_num_vecs() {
    return 4;   // 返回浮点向量的数量
  }
  static constexpr int int_num_vecs() {
    return 4;   // 返回整数向量的数量
  }
  using float_vec_return_type = std::array<Vectorized<float>, 4>;  // 浮点向量数组类型定义
  using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;  // 整数向量数组类型定义
  using value_type = typename c10::qint8::underlying;  // 使用 qint8 的底层值类型
  using vec_internal_type = vint8;  // 向量内部类型定义
  using vec_internal_mask_type = vbool8;  // 向量内部布尔掩码类型定义

  // 广播构造函数
  C10_ALWAYS_INLINE Vectorized(const c10::qint8& val)
      : _vec0{vec_splats(val.val_)}, _vec1{vec_splats(val.val_)} {}

  // 复制构造函数
  C10_ALWAYS_INLINE Vectorized(const Vectorized<c10::qint8>& other)
      : _vec0{other._vec0}, _vec1(other._vec1) {}

  // 构造函数，初始化为给定的 vint8 向量
  C10_ALWAYS_INLINE Vectorized(vint8 v) : _vec0{v}, _vec1{v} {}

  // 构造函数，初始化为给定的 vbool8 布尔向量
  C10_ALWAYS_INLINE Vectorized(vbool8 vmask) : _vecb0{vmask}, _vecb1{vmask} {}

  // 构造函数，分别初始化两个 vint8 向量
  C10_ALWAYS_INLINE Vectorized(vint8 v1, vint8 v2) : _vec0{v1}, _vec1{v2} {}

  // 构造函数，分别初始化两个 vbool8 布尔向量
  C10_ALWAYS_INLINE Vectorized(vbool8 v1, vbool8 v2) : _vecb0{v1}, _vecb1{v2} {}

  // 返回第一个 vint8 向量的常引用
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }

  // 返回第二个 vint8 向量的常引用
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  // 加载未对齐的数据到 Vectorized<c10::qint8> 中
  static C10_ALWAYS_INLINE Vectorized<c10::qint8> loadu(
      const void* ptr,
      int count = size()) {
    if (count == size()) {
      // 使用 VSX 指令加载数据到向量中
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const vint8*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const vint8*>(ptr))};
    }
    // 如果 count 小于 size，则使用标准库函数拷贝数据
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));
    // 返回加载后的 Vectorized<c10::qint8> 对象
    return {tmp_values};


这段代码片段中的注释已经完整解释了每行代码的作用和意图。
    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }


    // 返回一个包含两个向量加载结果的数组
    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }



  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }


  // 将向量存储到指定地址的函数
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    // 如果 count 等于 size()
    if (count == size()) {
      // 将 _vec0 和 _vec1 分别存储到 ptr+offset0 和 ptr+offset16 处
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      // 否则，如果 count 大于 0
      __at_align__ value_type tmp_values[size()]; // 创建临时数组 tmp_values
      vec_vsx_st(_vec0, offset0, tmp_values);     // 将 _vec0 存储到 tmp_values[offset0] 处
      vec_vsx_st(_vec1, offset16, tmp_values);    // 将 _vec1 存储到 tmp_values[offset16] 处
      // 将 tmp_values 的前 count 个元素拷贝到 ptr 指向的地址
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }



 public:
  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    vint16 vecshi0 = vec_unpackh(_vec0);
    vint16 vecshi1 = vec_unpackl(_vec0);

    vint16 vecshi2 = vec_unpackh(_vec1);
    vint16 vecshi3 = vec_unpackl(_vec1);

    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 veci1 = vec_unpackl(vecshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    vint32 veci3 = vec_unpackl(vecshi1);

    vint32 veci4 = vec_unpackh(vecshi2);
    vint32 veci5 = vec_unpackl(vecshi2);

    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 veci7 = vec_unpackl(vecshi3);

    vfloat32 vecf0_0 = vec_float(veci0);
    vfloat32 vecf1_0 = vec_float(veci1);

    vfloat32 vecf0_1 = vec_float(veci2);
    vfloat32 vecf1_1 = vec_float(veci3);

    vfloat32 vecf0_2 = vec_float(veci4);
    vfloat32 vecf1_2 = vec_float(veci5);

    vfloat32 vecf0_3 = vec_float(veci6);
    vfloat32 vecf1_3 = vec_float(veci7);
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();
    vfloat32 scale_zp_premul0 = scale_zp_premul.vec0();
    vfloat32 scale_zp_premul1 = scale_zp_premul.vec1();
    return {
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_0, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_0, scale_zp_premul1)},
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_1, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_1, scale_zp_premul1)},
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_2, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_2, scale_zp_premul1)},
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_3, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_3, scale_zp_premul1)}};
  }


 public:
  // 对象的一种方法，用于反量化操作，返回一个包含四个 Vectorized<float> 元素的数组
  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    // 将 _vec0 和 _vec1 按照规则拆分为 vint16 和 vint32 类型的变量
    vint16 vecshi0 = vec_unpackh(_vec0);
    vint16 vecshi1 = vec_unpackl(_vec0);

    vint16 vecshi2 = vec_unpackh(_vec1);
    vint16 vecshi3 = vec_unpackl(_vec1);

    // 进一步拆分为 vint32 类型的变量
    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 veci1 = vec_unpackl(vecshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    vint32 veci3 = vec_unpackl(vecshi1);

    vint32 veci4 = vec_unpackh(vecshi2);
    vint32 veci5 = vec_unpackl(vecshi2);

    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 veci7 = vec_unpackl(vecshi3);

    // 将 vint32 类型的变量转换为 vfloat32 类型的变量
    vfloat32 vecf0_0 = vec_float(veci0);
    vfloat32 vecf1_0 = vec_float(veci1);

    vfloat32 vecf0_1 = vec_float(veci2);
    vfloat32 vecf1_1 = vec_float(veci3);

    vfloat32 vecf0_2 = vec_float(veci4);
    vfloat32 vecf1_2 = vec_float(veci5);

    vfloat32 vecf0_3 = vec_float(veci6);
    vfloat32 vecf1_3 = vec_float(veci7);

    // 获取 scale 和 scale_zp_premul 各元素的值
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();
    vfloat32 scale_zp_premul0 = scale_zp_premul.vec0();
    vfloat32 scale_zp_premul1 = scale_zp_premul.vec1();

    // 返回四个 Vectorized<float> 元素的数组
    return {
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_0, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_0, scale_zp_premul1)},
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_1, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_1, scale_zp_premul1)},
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_2, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_2, scale_zp_premul1)},
        Vectorized<float>{
            vec_madd(scale_vec0, vecf0_3, scale_zp_premul0),
            vec_madd(scale_vec1, vecf1_3, scale_zp_premul1)}};
  }



  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    vint16 vecshi0 = vec_unpackh(_vec0);
    vint16 vecshi1 = vec_unpackl(_vec0);

    vint16 vecshi2 = vec_unpackh(_vec1);
    vint16 vecshi3 = vec_unpackl(_vec1);

    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 veci1 = vec_unpackl(vecshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    v
    // 将 vecshi3 向上拆分为两个 vint32 向量
    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 veci7 = vec_unpackl(vecshi3);

    // 将 vint32 向量转换为 vfloat32 向量
    vfloat32 vecf0_0 = vec_float(veci0);
    vfloat32 vecf1_0 = vec_float(veci1);

    vfloat32 vecf0_1 = vec_float(veci2);
    vfloat32 vecf1_1 = vec_float(veci3);

    vfloat32 vecf0_2 = vec_float(veci4);
    vfloat32 vecf1_2 = vec_float(veci5);

    vfloat32 vecf0_3 = vec_float(veci6);
    vfloat32 vecf1_3 = vec_float(veci7);

    // 从 scale 和 zero_point 中获取单独的 vfloat32 值
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();
    vfloat32 zero_point0 = zero_point.vec0();
    vfloat32 zero_point1 = zero_point.vec1();

    // 返回四个 Vectorized<float> 对象，每个对象进行量化操作
    return {
        Vectorized<float>{
            (vecf0_0 - zero_point0) * scale_vec0,
            (vecf1_0 - zero_point1) * scale_vec1},
        Vectorized<float>{
            (vecf0_1 - zero_point0) * scale_vec0,
            (vecf1_1 - zero_point1) * scale_vec1},
        Vectorized<float>{
            (vecf0_2 - zero_point0) * scale_vec0,
            (vecf1_2 - zero_point1) * scale_vec1},
        Vectorized<float>{
            (vecf0_3 - zero_point0) * scale_vec0,
            (vecf1_3 - zero_point1) * scale_vec1}};
}

// 静态方法：将四个 Vectorized<float> 向量进行量化操作，返回 Vectorized<c10::qint8> 向量
static Vectorized<c10::qint8> quantize(
    const float_vec_return_type& rhs,  // 输入的四个 Vectorized<float> 向量
    float scale,                       // 量化的缩放因子
    int32_t zero_point,                // 量化的零点
    float inverse_scale) {             // 量化的反向缩放因子

    // 使用 inverse_scale 创建一个 vfloat32 向量
    vfloat32 inverse_scale_v = vec_splats(inverse_scale);
    // 使用 zero_point 创建一个 vfloat32 向量
    vfloat32 vec_zero_point = vec_splats((float)zero_point);

    // 从 rhs 中提取四个 Vectorized<float> 向量
    Vectorized<float> vf0 = rhs[0];
    Vectorized<float> vf1 = rhs[1];
    Vectorized<float> vf2 = rhs[2];
    Vectorized<float> vf3 = rhs[3];

    // 将每个 Vectorized<float> 向量拆分为 vfloat32 向量
    vfloat32 vecf0 = vf0.vec0();
    vfloat32 vecf1 = vf0.vec1();
    vfloat32 vecf2 = vf1.vec0();
    vfloat32 vecf3 = vf1.vec1();
    vfloat32 vecf4 = vf2.vec0();
    vfloat32 vecf5 = vf2.vec1();
    vfloat32 vecf6 = vf3.vec0();
    vfloat32 vecf7 = vf3.vec1();

    // 每个 vfloat32 向量乘以 inverse_scale_v 进行反向量化
    vecf0 = vec_mul(vecf0, inverse_scale_v);
    vecf1 = vec_mul(vecf1, inverse_scale_v);
    vecf2 = vec_mul(vecf2, inverse_scale_v);
    vecf3 = vec_mul(vecf3, inverse_scale_v);
    vecf4 = vec_mul(vecf4, inverse_scale_v);
    vecf5 = vec_mul(vecf5, inverse_scale_v);
    vecf6 = vec_mul(vecf6, inverse_scale_v);
    vecf7 = vec_mul(vecf7, inverse_scale_v);

    // 每个 vfloat32 向量加上 vec_zero_point 进行量化
    vecf0 = vec_add(vec_rint(vecf0), vec_zero_point);
    vecf1 = vec_add(vec_rint(vecf1), vec_zero_point);
    vecf2 = vec_add(vec_rint(vecf2), vec_zero_point);
    vecf3 = vec_add(vec_rint(vecf3), vec_zero_point);
    vecf4 = vec_add(vec_rint(vecf4), vec_zero_point);
    vecf5 = vec_add(vec_rint(vecf5), vec_zero_point);
    vecf6 = vec_add(vec_rint(vecf6), vec_zero_point);
    vecf7 = vec_add(vec_rint(vecf7), vec_zero_point);

    // 将每个 vfloat32 向量转换为 vint32 向量
    vint32 veci0 = vec_signed(vecf0);
    vint32 veci1 = vec_signed(vecf1);
    vint32 veci2 = vec_signed(vecf2);
    vint32 veci3 = vec_signed(vecf3);
    // 将浮点向量转换为有符号整数向量
    vint32 veci4 = vec_signed(vecf4);
    vint32 veci5 = vec_signed(vecf5);
    vint32 veci6 = vec_signed(vecf6);
    vint32 veci7 = vec_signed(vecf7);

    // 对每个向量进行下界和上界的限制，并进行交叉组合操作
    // veci0 = vec_min(vmax, vec_max(vmin, vecf0));
    // veci1 = vec_min(vmax, vec_max(vmin, vecf1));
    // veci2 = vec_min(vmax, vec_max(vmin, vecf2));
    // veci3 = vec_min(vmax, vec_max(vmin, vecf3));

    // 对每个向量进行下界和上界的限制，并进行交叉组合操作
    // veci4 = vec_min(vmax, vec_max(vmin, vecf4));
    // veci5 = vec_min(vmax, vec_max(vmin, vecf5));
    // veci6 = vec_min(vmax, vec_max(vmin, vecf6));
    // veci7 = vec_min(vmax, vec_max(vmin, vecf7));
    // 使用 vec_packs 函数进行整数向量的打包操作，已经包含了 CLAMP 功能
    vint16 vecshi0 = vec_packs(veci0, veci1);
    vint16 vecshi1 = vec_packs(veci2, veci3);
    vint16 vecshi2 = vec_packs(veci4, veci5);
    vint16 vecshi3 = vec_packs(veci6, veci7);

    // 进一步将打包的有符号整数向量转换为 8 位有符号整数向量
    vint8 vec0 = vec_packs(vecshi0, vecshi1);
    vint8 vec1 = vec_packs(vecshi2, vecshi3);

    // 返回两个 8 位整数向量的组合结果
    return {vec0, vec1};
  }

  // 实现整数向量的 ReLU 激活函数
  Vectorized<c10::qint8> C10_ALWAYS_INLINE relu(Vectorized<c10::qint8> zero_point) const {
    // 分别对两个向量中的每个元素取最大值，实现 ReLU 操作
    return {vec_max(_vec0, zero_point._vec0), vec_max(_vec1, zero_point._vec1)};
  }

  // 实现整数向量的 ReLU6 激活函数
  Vectorized<c10::qint8> C10_ALWAYS_INLINE
  relu6(Vectorized<c10::qint8> zero_point, Vectorized<c10::qint8> q_six) const {
    // 对两个向量的每个元素分别取最大值，再与指定的阈值进行比较取最小值，实现 ReLU6 操作
    vint8 max0 = vec_max(_vec0, zero_point._vec0);
    vint8 max1 = vec_max(_vec1, zero_point._vec1);
    return {vec_min(max0, q_six._vec0), vec_min(max1, q_six._vec1)};
  }

  // 宽化子函数，计算两个整数向量的宽化差
  int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
    // 将每个向量的上半部分和下半部分进行拆分和宽化，得到更高精度的整数向量
    vint16 vecshi0 = vec_unpackh(_vec0);
    vint16 vecBshi0 = vec_unpackh(b._vec0);
    vint16 vecshi1 = vec_unpackl(_vec0);
    vint16 vecBshi1 = vec_unpackl(b._vec0);

    vint16 vecshi2 = vec_unpackh(_vec1);
    vint16 vecBshi2 = vec_unpackh(b._vec1);
    vint16 vecshi3 = vec_unpackl(_vec1);
    vint16 vecBshi3 = vec_unpackl(b._vec1);

    // 对拆分后的高精度整数向量进行逐元素相减操作
    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 vecBi0 = vec_unpackh(vecBshi0);
    vint32 veci1 = vec_unpackl(vecshi0);
    vint32 vecBi1 = vec_unpackl(vecBshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    vint32 vecBi2 = vec_unpackh(vecBshi1);
    vint32 veci3 = vec_unpackl(vecshi1);
    vint32 vecBi3 = vec_unpackl(vecBshi1);

    vint32 veci4 = vec_unpackh(vecshi2);
    vint32 vecBi4 = vec_unpackh(vecBshi2);
    vint32 veci5 = vec_unpackl(vecshi2);
    vint32 vecBi5 = vec_unpackl(vecBshi2);

    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 vecBi6 = vec_unpackh(vecBshi3);
    vint32 veci7 = vec_unpackl(vecshi3);
    vint32 vecBi7 = vec_unpackl(vecBshi3);

    // 将高精度的整数向量重新组合成原始的 8 位整数向量，并返回计算结果
    return {
        Vectorized<c10::qint32>(veci0 - vecBi0, veci1 - vecBi1),
        Vectorized<c10::qint32>(veci2 - vecBi2, veci3 - vecBi3),
        Vectorized<c10::qint32>(veci4 - vecBi4, veci5 - vecBi5),
        Vectorized<c10::qint32>(veci6 - vecBi6, veci7 - vecBi7)};
  }

  // 从整数向量返回重新量化结果
  static Vectorized<c10::qint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    // 使用给定的乘数将输入的整数向量重新量化，并返回重新量化后的结果
    vfloat32 vec_multiplier = vec_splats(multiplier);
    // 将标量 zero_point 扩展为向量 vec_zero_point
    vint32 vec_zero_point = vec_splats(zero_point);

    // 分别将输入向量 inp 中的四个元素分配给 vi0, vi1, vi2, vi3
    Vectorized<c10::qint32> vi0 = inp[0];
    Vectorized<c10::qint32> vi1 = inp[1];
    Vectorized<c10::qint32> vi2 = inp[2];
    Vectorized<c10::qint32> vi3 = inp[3];

    // 将每个输入向量中的两个部分转换为浮点向量 vecf0 到 vecf7
    vfloat32 vecf0 = vec_float(vi0.vec0());
    vfloat32 vecf1 = vec_float(vi0.vec1());
    vfloat32 vecf2 = vec_float(vi1.vec0());
    vfloat32 vecf3 = vec_float(vi1.vec1());

    vfloat32 vecf4 = vec_float(vi2.vec0());
    vfloat32 vecf5 = vec_float(vi2.vec1());
    vfloat32 vecf6 = vec_float(vi3.vec0());
    vfloat32 vecf7 = vec_float(vi3.vec1());

    // 将每个浮点向量与乘数 vec_multiplier 相乘
    vecf0 = vec_mul(vecf0, vec_multiplier);
    vecf1 = vec_mul(vecf1, vec_multiplier);
    vecf2 = vec_mul(vecf2, vec_multiplier);
    vecf3 = vec_mul(vecf3, vec_multiplier);

    vecf4 = vec_mul(vecf4, vec_multiplier);
    vecf5 = vec_mul(vecf5, vec_multiplier);
    vecf6 = vec_mul(vecf6, vec_multiplier);
    vecf7 = vec_mul(vecf7, vec_multiplier);

    // 对每个浮点向量进行舍入到最近的整数
    vecf0 = vec_rint(vecf0);
    vecf1 = vec_rint(vecf1);
    vecf2 = vec_rint(vecf2);
    vecf3 = vec_rint(vecf3);

    vecf4 = vec_rint(vecf4);
    vecf5 = vec_rint(vecf5);
    vecf6 = vec_rint(vecf6);
    vecf7 = vec_rint(vecf7);

    // 将每个浮点向量转换为有符号整数向量
    vint32 veci0 = vec_signed(vecf0);
    vint32 veci1 = vec_signed(vecf1);
    vint32 veci2 = vec_signed(vecf2);
    vint32 veci3 = vec_signed(vecf3);

    vint32 veci4 = vec_signed(vecf4);
    vint32 veci5 = vec_signed(vecf5);
    vint32 veci6 = vec_signed(vecf6);
    vint32 veci7 = vec_signed(vecf7);

    // 将每个整数向量与 vec_zero_point 相加
    veci0 = vec_add(veci0, vec_zero_point);
    veci1 = vec_add(veci1, vec_zero_point);
    veci2 = vec_add(veci2, vec_zero_point);
    veci3 = vec_add(veci3, vec_zero_point);

    veci4 = vec_add(veci4, vec_zero_point);
    veci5 = vec_add(veci5, vec_zero_point);
    veci6 = vec_add(veci6, vec_zero_point);
    veci7 = vec_add(veci7, vec_zero_point);

    // 将每两个整数向量打包为更窄的有符号整数向量
    vint16 vecshi0 = vec_packs(veci0, veci1);
    vint16 vecshi1 = vec_packs(veci2, veci3);
    vint16 vecshi2 = vec_packs(veci4, veci5);
    vint16 vecshi3 = vec_packs(veci6, veci7);

    // 将每两个更窄的整数向量打包为最终的有符号整数向量
    vint8 vec0 = vec_packs(vecshi0, vecshi1);
    vint8 vec1 = vec_packs(vecshi2, vecshi3);

    // 返回由两个最终向量组成的结果向量
    return {vec0, vec1};
  }

  // 定义成员运算符重载函数，用于比较两个 c10::qint8 类型的向量是否相等
  DEFINE_MEMBER_OP(operator==, c10::qint8, vec_cmpeq)
  // 定义成员运算符重载函数，用于比较两个 c10::qint8 类型的向量是否不等
  DEFINE_MEMBER_OP(operator!=, c10::qint8, vec_cmpne)
  // 定义成员运算符重载函数，用于比较两个 c10::qint8 类型的向量是否小于
  DEFINE_MEMBER_OP(operator<, c10::qint8, vec_cmplt)
  // 定义成员运算符重载函数，用于比较两个 c10::qint8 类型的向量是否小于等于
  DEFINE_MEMBER_OP(operator<=, c10::qint8, vec_cmple)
  // 定义成员运算符重载函数，用于比较两个 c10::qint8 类型的向量是否大于
  DEFINE_MEMBER_OP(operator>, c10::qint8, vec_cmpgt)
  // 定义成员运算符重载函数，用于比较两个 c10::qint8 类型的向量是否大于等于
  DEFINE_MEMBER_OP(operator>=, c10::qint8, vec_cmpge)
  // 定义成员运算符重载函数，用于对两个 c10::qint8 类型的向量执行加法
  DEFINE_MEMBER_OP(operator+, c10::qint8, vec_add)
  // 定义成员运算符重载函数，用于对两个 c10::qint8 类型的向量执行减法
  DEFINE_MEMBER_OP(operator-, c10::qint8, vec_sub)
  // 定义成员运算符重载函数，用于对两个 c10::qint8 类型的向量执行乘法
  DEFINE_MEMBER_OP(operator*, c10::qint8, vec_mul)
  // 定义成员函数，模拟二元操作符重载，用于对两个 c10::qint8 类型的向量执行除法
  DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, c10::qint8, /)
  // 定义成员运算符重载函数，用于找到两个 c10::qint8 类型的向量的最大值
  DEFINE_MEMBER_OP(maximum, c10::qint8, vec_max)
  // 定义成员运算符重载函数，用于找到两个 c10::qint8 类型的向量的最小值
  DEFINE_MEMBER_OP(minimum, c10::qint8, vec_min)
  // 定义成员运算符重载函数，用于对两个 c10::qint8 类型的向量执行按位与操作
  DEFINE_MEMBER_OP(operator&, c10::qint8, vec_and)
  // 定义成员运算符重载函数，用于对两个 c10::qint8 类型的向量执行按位或操作
  DEFINE_MEMBER_OP(operator|, c10::qint8, vec_or)
  // 定义成员运算符重载函数，用于对两个 c10::qint8 类型的向量执行按位异或操作
  DEFINE_MEMBER_OP(operator^, c10::qint8, vec_xor)
};

// 特化模板，实现两个 Vectorized<c10::qint8> 向量的最大值比较
template <>
Vectorized<c10::qint8> inline maximum(
    const Vectorized<c10::qint8>& a,
    const Vectorized<c10::qint8>& b) {
  // 调用 Vectorized 类的 maximum 方法，返回 a 和 b 向量中对应位置的最大值向量
  return a.maximum(b);
}

// 特化模板，实现两个 Vectorized<c10::qint8> 向量的最小值比较
template <>
Vectorized<c10::qint8> inline minimum(
    const Vectorized<c10::qint8>& a,
    const Vectorized<c10::qint8>& b) {
  // 调用 Vectorized 类的 minimum 方法，返回 a 和 b 向量中对应位置的最小值向量
  return a.minimum(b);
}
} // namespace vec
} // namespace at
```