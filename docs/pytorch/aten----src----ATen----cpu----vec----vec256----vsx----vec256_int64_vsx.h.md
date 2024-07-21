# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_int64_vsx.h`

```py
// 预处理指令，确保头文件仅被包含一次
#pragma once

// 包含 ATen 库的向量化指令和基本类型定义
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>

// ATen 命名空间
namespace at {
namespace vec {

// 内部嵌套命名空间 CPU_CAPABILITY 的注释说明
// 参见注释 [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// 模板特化：Vectorized 类模板的 int64_t 类型
template <>
class Vectorized<int64_t> {
 private:
  // 联合体，允许不同视图访问相同内存位置
  union {
    // 内部结构体，包含两个 vint64 向量
    struct {
      vint64 _vec0;
      vint64 _vec1;
    };
    // 内部结构体，包含两个 vbool64 布尔向量
    struct {
      vbool64 _vecb0;
      vbool64 _vecb1;
    };

  } __attribute__((__may_alias__));  // 标记允许别名优化

 public:
  // 类型定义
  using value_type = int64_t;
  using vec_internal_type = vint64;
  using vec_internal_mask_type = vbool64;
  using size_type = int;
  using ElementType = signed long long;

  // 返回向量长度
  static constexpr size_type size() {
    return 4;
  }

  // 默认构造函数
  Vectorized() {}

  // 向量化构造函数，初始化所有向量元素为 v
  C10_ALWAYS_INLINE Vectorized(vint64 v) : _vec0{v}, _vec1{v} {}

  // 向量化构造函数，使用布尔向量初始化
  C10_ALWAYS_INLINE Vectorized(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}

  // 向量化构造函数，使用两个 vint64 向量初始化
  C10_ALWAYS_INLINE Vectorized(vint64 v1, vint64 v2) : _vec0{v1}, _vec1{v2} {}

  // 向量化构造函数，使用两个 vbool64 布尔向量初始化
  C10_ALWAYS_INLINE Vectorized(vbool64 v1, vbool64 v2) : _vecb0{v1}, _vecb1{v2} {}

  // 向量化构造函数，使用标量初始化所有元素
  C10_ALWAYS_INLINE Vectorized(int64_t scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}

  // 向量化构造函数，使用四个标量分别初始化元素
  C10_ALWAYS_INLINE Vectorized(
      int64_t scalar1,
      int64_t scalar2,
      int64_t scalar3,
      int64_t scalar4)
      : _vec0{vint64{scalar1, scalar2}}, _vec1{vint64{scalar3, scalar4}} {}

  // 返回第一个向量
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }

  // 返回第二个向量
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  // 模板函数：根据掩码 mask 合并向量 a 和 b
  template <uint64_t mask>
  static std::enable_if_t<mask == 0, Vectorized<int64_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return a;
  }

  // 模板函数：根据掩码 mask 合并向量 a 和 b
  template <uint64_t mask>
  static std::enable_if_t<mask == 3, Vectorized<int64_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return {b._vec0, a._vec1};
  }

  // 模板函数：根据掩码 mask 合并向量 a 和 b
  template <uint64_t mask>
  static std::enable_if_t<(mask & 15) == 15, Vectorized<int64_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return b;
  }

  // 模板函数：根据掩码 mask 合并向量 a 和 b
  template <uint64_t mask>
  static std::enable_if_t<(mask > 0 && mask < 3), Vectorized<int64_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    // 定义掩码 g0 和 g1
    constexpr uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
    constexpr uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
    // 创建第一部分掩码
    const vbool64 mask_1st = (vbool64){g0, g1};
    // 返回合并后的向量
    return {(vint64)vec_sel(a._vec0, b._vec0, (vbool64)mask_1st), a._vec1};
  }

  // 模板函数：根据掩码 mask 合并向量 a 和 b
  template <uint64_t mask>
  static std::enable_if_t<(mask > 3) && (mask & 3) == 0, Vectorized<int64_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    // 定义掩码 g0_2 和 g1_2
    constexpr uint64_t g0_2 = ((mask & 4) >> 2) * 0xffffffffffffffff;
    constexpr uint64_t g1_2 = ((mask & 8) >> 3) * 0xffffffffffffffff;
    // 创建第二部分掩码
    const vbool64 mask_2nd = (vbool64){g0_2, g1_2};
    return {a._vec0, (vint64)vec_sel(a._vec1, b._vec1, (vbool64)mask_2nd)};
  }



    // 返回一个 Vectorized<int64_t> 对象，其中：
    // - 使用 a._vec0 作为第一个元素
    // - 使用 vec_sel(a._vec1, b._vec1, (vbool64)mask_2nd) 作为第二个元素
    // 这里的 vec_sel 函数根据 mask_2nd 的布尔值掩码选择 a._vec1 或 b._vec1 的对应元素
    template <uint64_t mask>
    static std::enable_if_t<
        (mask > 3) && (mask & 3) != 0 && (mask & 15) != 15,
        Vectorized<int64_t>>
        C10_ALWAYS_INLINE blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
      constexpr uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
      constexpr uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
      constexpr uint64_t g0_2 = ((mask & 4) >> 2) * 0xffffffffffffffff;
      constexpr uint64_t g1_2 = ((mask & 8) >> 3) * 0xffffffffffffffff;

      const vbool64 mask_1st = (vbool64){g0, g1};
      const vbool64 mask_2nd = (vbool64){g0_2, g1_2};
      return {
          (vint64)vec_sel(a._vec0, b._vec0, (vbool64)mask_1st),
          (vint64)vec_sel(a._vec1, b._vec1, (vbool64)mask_2nd)};
    }



    // 返回一个 Vectorized<int64_t> 对象，使用给定的掩码 mask 来混合向量 a 和 b 的元素：
    // - 根据掩码 mask，构造两个 vbool64 类型的掩码 mask_1st 和 mask_2nd
    // - 使用 vec_sel 函数基于 mask_1st 和 mask_2nd 选择 a 和 b 的对应元素
    static Vectorized<int64_t> C10_ALWAYS_INLINE blendv(
        const Vectorized<int64_t>& a,
        const Vectorized<int64_t>& b,
        const Vectorized<int64_t>& mask) {
      // 这里的 mask 是通过比较 vec256 而获得的掩码

      return {
          vec_sel(a._vec0, b._vec0, mask._vecb0),
          vec_sel(a._vec1, b._vec1, mask._vecb1)};
    }



    // 返回一个 Vectorized<int64_t> 对象，其中的元素是一个等差数列
    // - base 参数指定起始值，默认为 0
    // - step_t 类型的 step 参数指定步长，默认为 1
    template <typename step_t>
    static Vectorized<int64_t> arange(int64_t base = 0., step_t step = static_cast<step_t>(1)) {
      return Vectorized<int64_t>(base, base + step, base + 2 * step, base + 3 * step);
    }



    // 返回一个 Vectorized<int64_t> 对象，根据 count 的不同选择不同的操作：
    // - 当 count 为 0 时，返回向量 a
    // - 当 count 为 1 时，返回 blend<1>(a, b)，即使用掩码 1 进行混合
    // - 当 count 为 2 时，返回 blend<3>(a, b)，即使用掩码 3 进行混合
    // - 当 count 为 3 时，返回 blend<7>(a, b)，即使用掩码 7 进行混合
    // - 否则，返回向量 b
    static Vectorized<int64_t> C10_ALWAYS_INLINE
    set(const Vectorized<int64_t>& a,
        const Vectorized<int64_t>& b,
        size_t count = size()) {
      switch (count) {
        case 0:
          return a;
        case 1:
          return blend<1>(a, b);
        case 2:
          return blend<3>(a, b);
        case 3:
          return blend<7>(a, b);
      }

      return b;
    }



    // 返回一个 Vectorized<value_type> 对象，从指针 ptr 加载数据，根据 count 的值选择不同的加载方式：
    // - 当 count 等于 size() 时，将 ptr 解释为 double 类型的指针，加载两个 double 类型的值作为向量的元素
    // - 否则，从 ptr 复制 std::min(count, size()) 个 value_type 类型的值到临时数组 tmp_values 中，然后加载该数组的内容作为向量的元素
    static Vectorized<value_type> C10_ALWAYS_INLINE
    loadu(const void* ptr, int count = size()) {
      if (count == size()) {
        static_assert(sizeof(double) == sizeof(value_type));
        const double* dptr = reinterpret_cast<const double*>(ptr);
        return {// treat it as double load
                (vint64)vec_vsx_ld(offset0, dptr),
                (vint64)vec_vsx_ld(offset16, dptr)};
      }

      __at_align__ double tmp_values[size()] = {};
      std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

      return {
          (vint64)vec_vsx_ld(offset0, tmp_values),
          (vint64)vec_vsx_ld(offset16, tmp_values)};
    }



    // 将向量的内容存储到 ptr 指向的内存中，根据 count 的值选择不同的存储方式：
    // - 当 count 等于 size() 时，将向量的元素作为 double 类型存储到 ptr 指向的内存中
    // - 当 count 大于 0 但小于 size() 时，将向量的元素存储到临时数组 tmp_values 中，再将数组的内容复制到 ptr 指向的内存中
    void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
      if (count == size()) {
        double* dptr = reinterpret_cast<double*>(ptr);
        vec_vsx_st((vfloat64)_vec0, offset0, dptr);
        vec_vsx_st((vfloat64)_vec1, offset16, dptr);
      } else if (count > 0) {
        __at_align__ double tmp_values[size()];
        vec_vsx_st((vfloat64)_vec0, offset0, tmp_values);
        vec_vsx_st((vfloat64)_vec1, offset16, tmp_values);
        std::memcpy(
            ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
      }
    }
  }
  }
  const int64_t& operator[](int idx) const = delete;
  int64_t& operator[](int idx) = delete;

  // 返回角度向量，根据当前向量元素的正负情况选择 0 或 π
  Vectorized<int64_t> angle() const {
    return blendv(
      Vectorized<int64_t>(0), Vectorized<int64_t>(c10::pi<int64_t>), *this < Vectorized<int64_t>(0));
  }

  // 返回实部向量，即返回当前向量本身
  Vectorized<int64_t> real() const {
    return *this;
  }

  // 返回虚部向量，始终返回一个零向量
  Vectorized<int64_t> imag() const {
    return Vectorized<int64_t>{0};
  }

  // 返回共轭向量，即返回当前向量本身
  Vectorized<int64_t> conj() const {
    return *this;
  }

  // 返回绝对值向量，分别对每个元素进行绝对值运算
  Vectorized<int64_t> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  // 返回取反向量，分别对每个元素进行取反运算
  Vectorized<int64_t> C10_ALWAYS_INLINE neg() const {
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }

  // 定义按位取反操作符
  DEFINE_MEMBER_UNARY_OP(operator~, int64_t, vec_not)
  
  // 定义相等操作符
  DEFINE_MEMBER_OP(operator==, int64_t, vec_cmpeq)
  
  // 定义不等操作符
  DEFINE_MEMBER_OP(operator!=, int64_t, vec_cmpne)
  
  // 定义小于操作符
  DEFINE_MEMBER_OP(operator<, int64_t, vec_cmplt)
  
  // 定义小于等于操作符
  DEFINE_MEMBER_OP(operator<=, int64_t, vec_cmple)
  
  // 定义大于操作符
  DEFINE_MEMBER_OP(operator>, int64_t, vec_cmpgt)
  
  // 定义大于等于操作符
  DEFINE_MEMBER_OP(operator>=, int64_t, vec_cmpge)
  
  // 定义与 1 的比较操作符
  DEFINE_MEMBER_OP_AND_ONE(eq, int64_t, vec_cmpeq)
  
  // 定义与 1 的不等操作符
  DEFINE_MEMBER_OP_AND_ONE(ne, int64_t, vec_cmpne)
  
  // 定义与 1 的小于操作符
  DEFINE_MEMBER_OP_AND_ONE(lt, int64_t, vec_cmplt)
  
  // 定义与 1 的小于等于操作符
  DEFINE_MEMBER_OP_AND_ONE(le, int64_t, vec_cmple)
  
  // 定义与 1 的大于操作符
  DEFINE_MEMBER_OP_AND_ONE(gt, int64_t, vec_cmpgt)
  
  // 定义与 1 的大于等于操作符
  DEFINE_MEMBER_OP_AND_ONE(ge, int64_t, vec_cmpge)
  
  // 定义加法操作符
  DEFINE_MEMBER_OP(operator+, int64_t, vec_add)
  
  // 定义减法操作符
  DEFINE_MEMBER_OP(operator-, int64_t, vec_sub)
  
  // 定义乘法操作符
  DEFINE_MEMBER_OP(operator*, int64_t, vec_mul)
  
  // 定义除法操作符
  DEFINE_MEMBER_OP(operator/, int64_t, vec_div)
  
  // 定义最大值操作符
  DEFINE_MEMBER_OP(maximum, int64_t, vec_max)
  
  // 定义最小值操作符
  DEFINE_MEMBER_OP(minimum, int64_t, vec_min)
  
  // 定义按位与操作符
  DEFINE_MEMBER_OP(operator&, int64_t, vec_and)
  
  // 定义按位或操作符
  DEFINE_MEMBER_OP(operator|, int64_t, vec_or)
  
  // 定义按位异或操作符
  DEFINE_MEMBER_OP(operator^, int64_t, vec_xor)
// 定义一个特化的模板函数，实现整数向量的按位左移操作
template <>
Vectorized<int64_t> inline operator<<(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    // 将向量 b 的第一个和第二个部分转换为无符号64位整数
    vuint64 shift_vec0 = reinterpret_cast<vuint64>(b.vec0());
    vuint64 shift_vec1 = reinterpret_cast<vuint64>(b.vec1());
    // 返回按位左移操作的结果向量
    return Vectorized<int64_t>{vec_sl(a.vec0(), shift_vec0), vec_sl(a.vec1(), shift_vec1)};
}

// 定义一个特化的模板函数，实现整数向量的按位右移操作
template <>
Vectorized<int64_t> inline operator>>(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    // 将向量 b 的第一个和第二个部分转换为无符号64位整数
    vuint64 shift_vec0 = reinterpret_cast<vuint64>(b.vec0());
    vuint64 shift_vec1 = reinterpret_cast<vuint64>(b.vec1());
    // 返回按位右移操作的结果向量
    return Vectorized<int64_t>{vec_sr(a.vec0(), shift_vec0), vec_sr(a.vec1(), shift_vec1)};
}

// 定义一个特化的模板函数，实现整数向量的最大值操作
template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // 调用向量 a 的 maximum 方法，传入向量 b 作为参数
  return a.maximum(b);
}

// 定义一个特化的模板函数，实现整数向量的最小值操作
template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // 调用向量 a 的 minimum 方法，传入向量 b 作为参数
  return a.minimum(b);
}

} // namespace
} // namespace vec
} // namespace at
```