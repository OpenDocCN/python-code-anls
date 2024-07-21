# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_quint8_vsx.h`

```
// 防止头文件重复包含
#pragma once

// 引入ATen库的向量化指令和基础向量化类
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>

// 引入C10实用工具，包括整数范围和quint8类型定义
#include <c10/util/irange.h>
#include <c10/util/quint8.h>
#include <array>

// 此文件定义了用于量化类型的Vectorized<>模板特化

// 命名空间at和vec，内联命名空间为CPU_CAPABILITY
namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

// 定义无符号掩码常量vint16，并使用VSX扩展向量库的vec_splats函数初始化为0xFF
const vint16 mask_unsigned = vec_splats((short int)0xFF);

// Vectorized<c10::quint8>模板特化
template <>
struct Vectorized<c10::quint8> {
 private:
  union {
    // 使用联合体定义两组相同类型的向量或布尔向量
    struct {
      vuint8 _vec0;
      vuint8 _vec1;
    };
    struct {
      vbool8 _vecb0;
      vbool8 _vecb1;
    };

  } __attribute__((__may_alias__)); // 标记联合体可能具有别名访问

 public:
  // 默认构造函数
  Vectorized() {}

  // 定义size_type为int类型，并返回向量大小为32
  using size_type = int;
  static constexpr size_type size() {
    return 32;
  }

  // 定义float_num_vecs()静态成员函数，返回4，表示四倍的Vectorized<float>
  static constexpr size_t float_num_vecs() {
    return 4;
  }

  // 定义int_num_vecs()静态成员函数，返回4，表示四倍的Vectorized<c10::qint32>
  static constexpr int int_num_vecs() {
    return 4;
  }

  // 定义float_vec_return_type为存储四个Vectorized<float>的数组类型
  using float_vec_return_type = std::array<Vectorized<float>, 4>;

  // 定义int_vec_return_type为存储四个Vectorized<c10::qint32>的数组类型
  using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;

  // 定义value_type为c10::quint8类型的底层类型
  using value_type = typename c10::quint8::underlying;

  // 定义vec_internal_type为vuint8类型的向量内部类型
  using vec_internal_type = vuint8;

  // 定义vec_internal_mask_type为vbool8类型的向量掩码类型
  using vec_internal_mask_type = vbool8;

  // 构造函数，使用c10::quint8类型值val构造广播向量
  C10_ALWAYS_INLINE Vectorized(const c10::quint8& val)
      : _vec0(vec_splats(val.val_)), _vec1(vec_splats(val.val_)) {}

  // 复制构造函数，从另一个Vectorized<c10::quint8>对象other构造新对象
  C10_ALWAYS_INLINE Vectorized(const Vectorized<c10::quint8>& other)
      : _vec0{other._vec0}, _vec1(other._vec1) {}

  // 向量构造函数，使用两个vuint8向量v1和v2初始化
  C10_ALWAYS_INLINE Vectorized(vuint8 v1, vuint8 v2) : _vec0{v1}, _vec1{v2} {}

  // 布尔向量构造函数，使用两个vbool8向量v1和v2初始化
  C10_ALWAYS_INLINE Vectorized(vbool8 v1, vbool8 v2) : _vecb0{v1}, _vecb1{v2} {}

  // 返回_vec0成员向量的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }

  // 返回_vec1成员向量的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  // 静态成员函数loadu，从指针ptr加载数据到向量中，count参数默认为size()
  static C10_ALWAYS_INLINE Vectorized<c10::quint8> loadu(
      const void* ptr,
      int count = size()) {
    if (count == size()) {
      // 如果count等于size()，使用VSX指令vec_vsx_ld加载数据到向量中
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
    }
    // 如果count不等于size()，创建临时的value_type类型数组tmp_values
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));
    // 使用 std::memcpy 将指针 ptr 处开始的内存内容复制到 tmp_values 数组中，
    // 复制的字节数为 std::min(count, size()) * sizeof(value_type)
    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 如果 count 等于当前对象的大小 size()，
      // 将 _vec0 和 _vec1 向量存储到 ptr 指向的内存地址，使用 vec_vsx_st 函数
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      // 否则，如果 count 大于 0，
      __at_align__ value_type tmp_values[size()];
      // 创建临时数组 tmp_values，其大小为当前对象的大小 size()
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      // 将 _vec0 和 _vec1 向量存储到 tmp_values 数组中
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
      // 将 tmp_values 数组中的部分数据（最多 count 大小）复制到 ptr 指向的内存地址
    }
  }

 public:
  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    // unpacking unsigned as signed
    // 将 _vec0 和 _vec1 向量的每个元素从 unsigned 类型解包为 signed 类型
    vint16 vecshi0 = vec_unpackh((vint8)_vec0);
    vint16 vecshi1 = vec_unpackl((vint8)_vec0);

    vint16 vecshi2 = vec_unpackh((vint8)_vec1);
    vint16 vecshi3 = vec_unpackl((vint8)_vec1);

    // signed ->  unsigned
    // 将解包后的 signed 类型向量再次转换为 unsigned 类型
    vecshi0 = vec_and(vecshi0, mask_unsigned);
    vecshi1 = vec_and(vecshi1, mask_unsigned);

    vecshi2 = vec_and(vecshi2, mask_unsigned);
    vecshi3 = vec_and(vecshi3, mask_unsigned);

    // 将每个解包并转换后的向量再次解包为 vint32 类型
    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 veci1 = vec_unpackl(vecshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    vint32 veci3 = vec_unpackl(vecshi1);

    vint32 veci4 = vec_unpackh(vecshi2);
    vint32 veci5 = vec_unpackl(vecshi2);

    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 veci7 = vec_unpackl(vecshi3);

    // 将 vint32 类型的向量转换为 vfloat32 类型的浮点向量
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
    // 返回四个 Vectorized<float> 类型的对象，每个对象包含两个浮点向量，
    // 每个浮点向量通过乘法和加法操作生成
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
    // unpacking unsigned as signed
    // 将 _vec0 向量的每个元素从 unsigned 类型解包为 signed 类型
    vint16 vecshi0 = vec_unpackh((vint8)_vec0);
    // 将_vec0向量的每个8位元素解包成16位有符号整数向量vecshi1
    vint16 vecshi1 = vec_unpackl((vint8)_vec0);

    // 将_vec1向量的每个8位元素解包成16位有符号整数向量vecshi2（高位）
    vint16 vecshi2 = vec_unpackh((vint8)_vec1);

    // 将_vec1向量的每个8位元素解包成16位有符号整数向量vecshi3（低位）
    vint16 vecshi3 = vec_unpackl((vint8)_vec1);

    // 将vecshi0的元素与无符号掩码mask_unsigned进行按位与操作
    vecshi0 = vec_and(vecshi0, mask_unsigned);

    // 将vecshi1的元素与无符号掩码mask_unsigned进行按位与操作
    vecshi1 = vec_and(vecshi1, mask_unsigned);

    // 将vecshi2的元素与无符号掩码mask_unsigned进行按位与操作
    vecshi2 = vec_and(vecshi2, mask_unsigned);

    // 将vecshi3的元素与无符号掩码mask_unsigned进行按位与操作
    vecshi3 = vec_and(vecshi3, mask_unsigned);

    // 将vecshi0的高位元素解包成32位有符号整数向量veci0
    vint32 veci0 = vec_unpackh(vecshi0);

    // 将vecshi0的低位元素解包成32位有符号整数向量veci1
    vint32 veci1 = vec_unpackl(vecshi0);

    // 将vecshi1的高位元素解包成32位有符号整数向量veci2
    vint32 veci2 = vec_unpackh(vecshi1);

    // 将vecshi1的低位元素解包成32位有符号整数向量veci3
    vint32 veci3 = vec_unpackl(vecshi1);

    // 将vecshi2的高位元素解包成32位有符号整数向量veci4
    vint32 veci4 = vec_unpackh(vecshi2);

    // 将vecshi2的低位元素解包成32位有符号整数向量veci5
    vint32 veci5 = vec_unpackl(vecshi2);

    // 将vecshi3的高位元素解包成32位有符号整数向量veci6
    vint32 veci6 = vec_unpackh(vecshi3);

    // 将vecshi3的低位元素解包成32位有符号整数向量veci7
    vint32 veci7 = vec_unpackl(vecshi3);

    // 将veci0的元素转换为单精度浮点向量vecf0_0
    vfloat32 vecf0_0 = vec_float(veci0);

    // 将veci1的元素转换为单精度浮点向量vecf1_0
    vfloat32 vecf1_0 = vec_float(veci1);

    // 将veci2的元素转换为单精度浮点向量vecf0_1
    vfloat32 vecf0_1 = vec_float(veci2);

    // 将veci3的元素转换为单精度浮点向量vecf1_1
    vfloat32 vecf1_1 = vec_float(veci3);

    // 将veci4的元素转换为单精度浮点向量vecf0_2
    vfloat32 vecf0_2 = vec_float(veci4);

    // 将veci5的元素转换为单精度浮点向量vecf1_2
    vfloat32 vecf1_2 = vec_float(veci5);

    // 将veci6的元素转换为单精度浮点向量vecf0_3
    vfloat32 vecf0_3 = vec_float(veci6);

    // 将veci7的元素转换为单精度浮点向量vecf1_3
    vfloat32 vecf1_3 = vec_float(veci7);

    // 将scale的vec0部分赋值给scale_vec0
    vfloat32 scale_vec0 = scale.vec0();

    // 将scale的vec1部分赋值给scale_vec1
    vfloat32 scale_vec1 = scale.vec1();

    // 将zero_point的vec0部分赋值给zero_point0
    vfloat32 zero_point0 = zero_point.vec0();

    // 将zero_point的vec1部分赋值给zero_point1
    vfloat32 zero_point1 = zero_point.vec1();

    // 返回四个向量化的浮点值，经过量化和反量化处理
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

// 静态方法：将输入的浮点向量化数据rhs进行量化，使用给定的scale和zero_point参数，返回量化后的无符号8位整数向量
static Vectorized<c10::quint8> quantize(
    const float_vec_return_type& rhs,
    float scale,
    int32_t zero_point,
    float inverse_scale) {
    // 创建一个单精度浮点向量，每个元素都是inverse_scale
    vfloat32 vec_inverse = vec_splats(inverse_scale);

    // 创建一个单精度浮点向量，每个元素都是zero_point
    vfloat32 vec_zero_point = vec_splats((float)zero_point);

    // 从rhs中获取四个Vectorized<float>对象
    Vectorized<float> vf0 = rhs[0];
    Vectorized<float> vf1 = rhs[1];
    Vectorized<float> vf2 = rhs[2];
    Vectorized<float> vf3 = rhs[3];

    // 从Vectorized<float>对象中提取单精度浮点向量的元素
    vfloat32 vecf0 = vf0.vec0();
    vfloat32 vecf1 = vf0.vec1();
    vfloat32 vecf2 = vf1.vec0();
    vfloat32 vecf3 = vf1.vec1();
    vfloat32 vecf4 = vf2.vec0();
    vfloat32 vecf5 = vf2.vec1();
    vfloat32 vecf6 = vf3.vec0();
    vfloat32 vecf7 = vf3.vec1();

    // 将每个单精度浮点向量元素乘以inverse_scale，实现量化操作
    vecf0 = vec_mul(vecf0, vec_inverse);
    vecf1 = vec_mul(vecf1, vec_inverse);
    vecf2 = vec_mul(vecf2, vec_inverse);
    vecf3 = vec_mul(vecf3, vec_inverse);
    vecf4 = vec_mul(vecf4, vec_inverse);
    vecf5 = vec_mul(vecf5, vec_inverse);
    vecf6 = vec_mul(vecf6, vec_inverse);
    vecf7 = vec_mul(vecf7, vec_inverse);
    // 对向量进行舍入并加上零点偏移量
    vecf0 = vec_add(vec_rint(vecf0), vec_zero_point);
    vecf1 = vec_add(vec_rint(vecf1), vec_zero_point);
    vecf2 = vec_add(vec_rint(vecf2), vec_zero_point);
    vecf3 = vec_add(vec_rint(vecf3), vec_zero_point);

    vecf4 = vec_add(vec_rint(vecf4), vec_zero_point);
    vecf5 = vec_add(vec_rint(vecf5), vec_zero_point);
    vecf6 = vec_add(vec_rint(vecf6), vec_zero_point);
    vecf7 = vec_add(vec_rint(vecf7), vec_zero_point);

    // 将浮点向量转换为有符号整数向量
    vint32 veci0 = vec_signed(vecf0);
    vint32 veci1 = vec_signed(vecf1);
    vint32 veci2 = vec_signed(vecf2);
    vint32 veci3 = vec_signed(vecf3);

    vint32 veci4 = vec_signed(vecf4);
    vint32 veci5 = vec_signed(vecf5);
    vint32 veci6 = vec_signed(vecf6);
    vint32 veci7 = vec_signed(vecf7);

    // 将有符号整数向量打包成有符号短整数向量
    vint16 vecshi0 = vec_packs(veci0, veci1);
    vint16 vecshi1 = vec_packs(veci2, veci3);
    vint16 vecshi2 = vec_packs(veci4, veci5);
    vint16 vecshi3 = vec_packs(veci6, veci7);

    // 将有符号短整数向量打包成无符号字节向量
    vuint8 vec0 = vec_packsu(vecshi0, vecshi1);
    vuint8 vec1 = vec_packsu(vecshi2, vecshi3);

    // 返回两个无符号字节向量
    return {vec0, vec1};
  }

  // 使用 ReLU 函数对每个向量执行逐元素的最大值运算
  Vectorized<c10::quint8> C10_ALWAYS_INLINE relu(Vectorized<c10::quint8> zero_point) const {
    return {vec_max(_vec0, zero_point._vec0), vec_max(_vec1, zero_point._vec1)};
  }

  // 使用 ReLU6 函数对每个向量执行逐元素的最大值和最小值运算
  Vectorized<c10::quint8> C10_ALWAYS_INLINE
  relu6(Vectorized<c10::quint8> zero_point, Vectorized<c10::quint8> q_six) const {
    // 对每个向量执行逐元素的最大值运算
    vuint8 max0 = vec_max(_vec0, zero_point._vec0);
    vuint8 max1 = vec_max(_vec1, zero_point._vec1);
    // 对每个向量执行逐元素的最小值运算
    return {vec_min(max0, q_six._vec0), vec_min(max1, q_six._vec1)};
  }

  // 执行宽化减法操作，将两个向量中每个元素的差宽化为更大的整数类型
  int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
    // 将第一个向量的每个字节解包为有符号短整数，并执行位与运算
    vint16 vecshi0 = vec_unpackh((vint8)_vec0);
    vint16 vecBshi0 = vec_unpackh((vint8)b._vec0);
    vint16 vecshi1 = vec_unpackl((vint8)_vec0);
    vint16 vecBshi1 = vec_unpackl((vint8)b._vec0);

    vint16 vecshi2 = vec_unpackh((vint8)_vec1);
    vint16 vecBshi2 = vec_unpackh((vint8)b._vec1);
    vint16 vecshi3 = vec_unpackl((vint8)_vec1);
    vint16 vecBshi3 = vec_unpackl((vint8)b._vec1);

    // 对解包后的有符号短整数向量执行位与运算
    vecshi0 = vec_and(vecshi0, mask_unsigned);
    vecBshi0 = vec_and(vecBshi0, mask_unsigned);
    vecshi1 = vec_and(vecshi1, mask_unsigned);
    vecBshi1 = vec_and(vecBshi1, mask_unsigned);

    vecshi2 = vec_and(vecshi2, mask_unsigned);
    vecBshi2 = vec_and(vecBshi2, mask_unsigned);
    vecshi3 = vec_and(vecshi3, mask_unsigned);
    vecBshi3 = vec_and(vecBshi3, mask_unsigned);

    // 将解包后的有符号短整数向量进一步解包为有符号整数向量
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
    // 继续执行宽化减法操作

    vint32 veci7 = vec_unpackl(vecBshi3);

    // 返回宽化减法的结果向量
    return {veci0 - vecBi0, veci1 - vecBi1, veci2 - vecBi2, veci3 - vecBi3,
            veci4 - vecBi4, veci5 - vecBi5, veci6 - vecBi6, veci7 - vecBi7};
  }
    // 将输入向量中的高位元素解包成低位元素组成的向量
    vint32 veci7 = vec_unpackl(vecshi3);
    // 将输入向量中的高位元素解包成低位元素组成的向量
    vint32 vecBi7 = vec_unpackl(vecBshi3);

    // 返回一个包含四个元素的向量数组，每个元素表示对应位置元素之差
    return {
        Vectorized<c10::qint32>(veci0 - vecBi0, veci1 - vecBi1),
        Vectorized<c10::qint32>(veci2 - vecBi2, veci3 - vecBi3),
        Vectorized<c10::qint32>(veci4 - vecBi4, veci5 - vecBi5),
        Vectorized<c10::qint32>(veci6 - vecBi6, veci7 - vecBi7)};
  }

  // 从整型向量返回量化的无符号八位向量
  static Vectorized<c10::quint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    // 创建浮点数向量，每个元素为输入的乘数
    vfloat32 vec_multiplier = vec_splats(multiplier);
    // 创建整型向量，每个元素为输入的零点
    vint32 vec_zero_point = vec_splats(zero_point);

    // 从输入的整型向量中获取第一个向量
    Vectorized<c10::qint32> vi0 = inp[0];
    // 从输入的整型向量中获取第二个向量
    Vectorized<c10::qint32> vi1 = inp[1];
    // 从输入的整型向量中获取第三个向量
    Vectorized<c10::qint32> vi2 = inp[2];
    // 从输入的整型向量中获取第四个向量
    Vectorized<c10::qint32> vi3 = inp[3];

    // 将第一个向量中的两个元素转换成浮点数向量
    vfloat32 vecf0 = vec_float(vi0.vec0());
    vfloat32 vecf1 = vec_float(vi0.vec1());
    // 将第二个向量中的两个元素转换成浮点数向量
    vfloat32 vecf2 = vec_float(vi1.vec0());
    vfloat32 vecf3 = vec_float(vi1.vec1());

    // 将第三个向量中的两个元素转换成浮点数向量
    vfloat32 vecf4 = vec_float(vi2.vec0());
    vfloat32 vecf5 = vec_float(vi2.vec1());
    // 将第四个向量中的两个元素转换成浮点数向量
    vfloat32 vecf6 = vec_float(vi3.vec0());
    vfloat32 vecf7 = vec_float(vi3.vec1());

    // 将每个浮点数向量元素与乘数向量中对应元素相乘
    vecf0 = vec_mul(vecf0, vec_multiplier);
    vecf1 = vec_mul(vecf1, vec_multiplier);
    vecf2 = vec_mul(vecf2, vec_multiplier);
    vecf3 = vec_mul(vecf3, vec_multiplier);

    vecf4 = vec_mul(vecf4, vec_multiplier);
    vecf5 = vec_mul(vecf5, vec_multiplier);
    vecf6 = vec_mul(vecf6, vec_multiplier);
    vecf7 = vec_mul(vecf7, vec_multiplier);

    // 将每个浮点数向量元素四舍五入到最近的整数
    vecf0 = vec_rint(vecf0);
    vecf1 = vec_rint(vecf1);
    vecf2 = vec_rint(vecf2);
    vecf3 = vec_rint(vecf3);

    vecf4 = vec_rint(vecf4);
    vecf5 = vec_rint(vecf5);
    vecf6 = vec_rint(vecf6);
    vecf7 = vec_rint(vecf7);

    // 将每个浮点数向量元素转换为带符号整数向量
    vint32 veci0 = vec_signed(vecf0);
    vint32 veci1 = vec_signed(vecf1);
    vint32 veci2 = vec_signed(vecf2);
    vint32 veci3 = vec_signed(vecf3);

    vint32 veci4 = vec_signed(vecf4);
    vint32 veci5 = vec_signed(vecf5);
    vint32 veci6 = vec_signed(vecf6);
    vint32 veci7 = vec_signed(vecf7);

    // 将每个整数向量元素与零点向量中对应元素相加
    veci0 = vec_add(veci0, vec_zero_point);
    veci1 = vec_add(veci1, vec_zero_point);
    veci2 = vec_add(veci2, vec_zero_point);
    veci3 = vec_add(veci3, vec_zero_point);

    veci4 = vec_add(veci4, vec_zero_point);
    veci5 = vec_add(veci5, vec_zero_point);
    veci6 = vec_add(veci6, vec_zero_point);
    veci7 = vec_add(veci7, vec_zero_point);

    // 将每两个整数向量打包成一个带符号短整数向量
    vint16 vecshi0 = vec_packs(veci0, veci1);
    vint16 vecshi1 = vec_packs(veci2, veci3);
    vint16 vecshi2 = vec_packs(veci4, veci5);
    vint16 vecshi3 = vec_packs(veci6, veci7);

    // 将每两个带符号短整数向量打包成一个无符号八位整数向量
    vuint8 vec0 = vec_packsu(vecshi0, vecshi1);
    vuint8 vec1 = vec_packsu(vecshi2, vecshi3);
    返回一个集合，其中包含两个向量 vec0 和 vec1
  }

  定义一个成员运算符重载函数，实现 c10::quint8 类型的相等比较，并返回结果向量 vec_cmpeq
  定义一个成员运算符重载函数，实现 c10::quint8 类型的不等比较，并返回结果向量 vec_cmpne
  定义一个成员运算符重载函数，实现 c10::quint8 类型的小于比较，并返回结果向量 vec_cmplt
  定义一个成员运算符重载函数，实现 c10::quint8 类型的小于等于比较，并返回结果向量 vec_cmple
  定义一个成员运算符重载函数，实现 c10::quint8 类型的大于比较，并返回结果向量 vec_cmpgt
  定义一个成员运算符重载函数，实现 c10::quint8 类型的大于等于比较，并返回结果向量 vec_cmpge
  定义一个成员运算符重载函数，实现 c10::quint8 类型的加法，并返回结果向量 vec_add
  定义一个成员运算符重载函数，实现 c10::quint8 类型的减法，并返回结果向量 vec_sub
  定义一个成员运算符重载函数，实现 c10::quint8 类型的乘法，并返回结果向量 vec_mul
  定义一个成员函数，实现 c10::quint8 类型的除法运算符模拟，并返回结果向量
  定义一个成员运算符重载函数，实现 c10::quint8 类型的最大值比较，并返回结果向量 vec_max
  定义一个成员运算符重载函数，实现 c10::quint8 类型的最小值比较，并返回结果向量 vec_min
  定义一个成员运算符重载函数，实现 c10::quint8 类型的按位与操作，并返回结果向量 vec_and
  定义一个成员运算符重载函数，实现 c10::quint8 类型的按位或操作，并返回结果向量 vec_or
  定义一个成员运算符重载函数，实现 c10::quint8 类型的按位异或操作，并返回结果向量 vec_xor
};

template`
};
# 结束当前命名空间作用域

template <>
# 定义模板特化，指定模板参数为 c10::quint8 类型
Vectorized<c10::quint8> inline maximum(
    const Vectorized<c10::quint8>& a,
    const Vectorized<c10::quint8>& b) {
  # 调用 Vectorized 类的 maximum 方法，返回 a 和 b 的逐元素最大值
  return a.maximum(b);
}

template <>
# 定义模板特化，指定模板参数为 c10::quint8 类型
Vectorized<c10::quint8> inline minimum(
    const Vectorized<c10::quint8>& a,
    const Vectorized<c10::quint8>& b) {
  # 调用 Vectorized 类的 minimum 方法，返回 a 和 b 的逐元素最小值
  return a.minimum(b);
}

} // namespace
# 结束 vec 命名空间作用域
} // namespace at
# 结束 at 命名空间作用域
```