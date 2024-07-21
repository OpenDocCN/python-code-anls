# `.\pytorch\aten\src\ATen\cpu\vec\vec_n.h`

```py
#pragma once

#include <ATen/cpu/vec/vec_base.h>  // 包含向量化类型的基本定义
#include <array>  // 包含标准数组模板

namespace at::vec {
inline namespace CPU_CAPABILITY {

/**
 * @brief 表示具有 `N * Vectorized<T>::size()` 个元素的向量化类型的类模板，
 * 旨在支持任意大小的向量。一个特定的用例是表示从不同大小的数据类型转换得到的具有相同
 * 向量元素数量的向量，例如 `VectorizedN<float, 2>` 可以是从两个 `Vectorized<bfloat16>`
 * 转换得到的向量，`VectorizedN<int64_t, 2>` 可以是从两个 `Vectorized<int32_t>` 转换得到的向量。
 *
 * 它支持大多数 `Vectorized<T>` 的操作，其实现通过对 `N` 进行循环委托给 `Vectorized<T>`。
 *
 * @tparam T 向量化元素的基础类型。
 * @tparam N `Vectorized<T>` 的数量。
 */
template <typename T, int N>
class VectorizedN {
 public:
  using value_type = T;  // 元素类型
  using size_type = int;  // 大小类型

  static constexpr size_type size_T = sizeof(T);  // 元素类型的大小
  static constexpr size_type size() {  // 计算总大小
    return Vectorized<T>::size() * N;
  }

 private:
  std::array<Vectorized<T>, N> values;  // 包含 N 个 Vectorized<T> 对象的数组

 public:
  // methods not implemented yet:
  // variadic constructor, operator T*, as_bytes, zero_mask

#define VECTORIZEDN_DEFINE_UNARY_OP(op)                             \
  VectorizedN<T, N> op() const {                                    \
    return unary_op([](const Vectorized<T>& a) { return a.op(); }); \
  }

#define VECTORIZEDN_DEFINE_BINARY_OP(op)                            \
  VectorizedN<T, N> op(const VectorizedN<T, N>& other) const {      \
    return binary_op(                                               \
        other, [](const Vectorized<T>& a, const Vectorized<T>& b) { \
          return a.op(b);                                           \
        });                                                         \
  }

  /**
   * @brief 对每个值应用一元操作符。
   * @param op 一元操作符函数对象。
   * @return 执行操作后的结果向量。
   */
  template <typename Op>
  inline VectorizedN<T, N> unary_op(Op op) const {
    VectorizedN<T, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result.values[i] = op(values[i]);
    }
    return result;
  }

  /**
   * @brief 对每个值应用二元操作符。
   * @param other 另一个 VectorizedN 对象。
   * @param op 二元操作符函数对象。
   * @return 执行操作后的结果向量。
   */
  template <typename Op>
  inline VectorizedN<T, N> binary_op(const VectorizedN<T, N>& other, Op op)
      const {
    VectorizedN<T, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result.values[i] = op(values[i], other.values[i]);
    }
    return result;
  }

  /**
   * @brief 默认构造函数。
   */
  VectorizedN() = default;

  /**
   * @brief 使用指定值构造向量化对象。
   * @param val 初始值。
   */
  explicit VectorizedN(T val) {
    for (int i = 0; i < N; ++i) {
      values[i] = Vectorized<T>(val);
    }
  }

  /**
   * @brief 隐式转换为 Vectorized<T> 类型。
   */
  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  VectorizedN(const Vectorized<T>& val) : values({val}) {}

  /**
   * @brief 隐式转换为 Vectorized<T> 类型。
   */
  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  inline operator Vectorized<T>() const {
    return values[0];
  }

  /**
   * @brief 访问指定索引处的 Vectorized<T> 对象。
   * @param i 索引。
   * @return 指定索引处的对象引用。
   */
  inline const Vectorized<T>& operator[](int i) const {
    return values[i];
  }

  /**
   * @brief 访问指定索引处的 Vectorized<T> 对象。
   * @param i 索引。
   * @return 指定索引处的对象引用。
   */
  inline Vectorized<T>& operator[](int i) {
  // 返回第 i 个元素的值
  return values[i];
}

// 使用掩码 mask 进行混合操作，返回混合后的结果向量
template <int64_t mask>
static VectorizedN<T, N> blend(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& b) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    // 调用 Vectorized<T>::blend<mask> 方法，将 a 和 b 向量的第 i 个元素混合后存入结果向量
    result.values[i] = Vectorized<T>::template blend<mask>(a.values[i], b.values[i]);
  }
  return result;
}

// 使用掩码向量 mask 进行向量化混合操作，返回混合后的结果向量
static VectorizedN<T, N> blendv(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& b,
    const VectorizedN<T, N>& mask) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    // 调用 Vectorized<T>::blendv 方法，将 a、b 和 mask 向量的第 i 个元素进行混合操作并存入结果向量
    result.values[i] =
        Vectorized<T>::blendv(a.values[i], b.values[i], mask.values[i]);
  }
  return result;
}

// 生成一个以 base 为起始值、以 step 为步长、长度为 N 的等差数列向量
template <typename step_t>
static VectorizedN<T, N> arange(
    T base = static_cast<T>(0),
    step_t step = static_cast<step_t>(1)) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    // 调用 Vectorized<T>::arange 方法生成等差数列的第 i 个元素并存入结果向量
    result.values[i] = Vectorized<T>::arange(base, step);
    base += step * Vectorized<T>::size();
  }
  return result;
}

// 使用向量 a 和 b 进行设置操作，将前 count 个元素设为给定值并返回结果向量
static VectorizedN<T, N> set(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& b,
    int64_t count = size()) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    // 调用 Vectorized<T>::set 方法将 a 和 b 向量的前 count 个元素进行设置操作并存入结果向量
    result.values[i] = Vectorized<T>::set(
        a.values[i],
        b.values[i],
        std::min(count, (int64_t)Vectorized<T>::size()));
    count -= Vectorized<T>::size();
    if (count <= 0) {
      break;
    }
  }
  return result;
}

// 从内存地址 ptr 处加载 N 个未对齐的元素到向量中并返回结果向量
static VectorizedN<T, N> loadu(const void* ptr) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    // 调用 Vectorized<T>::loadu 方法加载从 ptr 开始的未对齐数据到结果向量中
    result.values[i] = Vectorized<T>::loadu(ptr);
    ptr = static_cast<const T*>(ptr) + Vectorized<T>::size();
  }
  return result;
}

// 从内存地址 ptr 处加载最多 count 个未对齐的元素到向量中并返回结果向量
static VectorizedN<T, N> loadu(const void* ptr, int64_t count) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    // 调用 Vectorized<T>::loadu 方法加载从 ptr 开始的最多 count 个未对齐数据到结果向量中
    result.values[i] = Vectorized<T>::loadu(
        ptr, std::min(count, (int64_t)Vectorized<T>::size()));
    ptr = static_cast<const T*>(ptr) + Vectorized<T>::size();
    count -= Vectorized<T>::size();
    if (count <= 0) {
      break;
    }
  }
  return result;
}

// 将向量的数据存储到内存地址 ptr 处
void store(void* ptr) const {
  for (int i = 0; i < N; ++i) {
    // 调用 values[i].store(ptr) 将第 i 个元素的数据存储到 ptr 处
    values[i].store(ptr);
    ptr = static_cast<T*>(ptr) + Vectorized<T>::size();
  }
}

// 将向量的最多 count 个数据存储到内存地址 ptr 处
void store(void* ptr, int count) const {
  for (int i = 0; i < N; ++i) {
    // 调用 values[i].store(ptr, count) 将第 i 个元素的最多 count 个数据存储到 ptr 处
    values[i].store(ptr, std::min(count, (int)Vectorized<T>::size()));
    ptr = static_cast<T*>(ptr) + Vectorized<T>::size();
    count -= Vectorized<T>::size();
    if (count <= 0) {
      break;
    }
  }
}

// 检查向量中是否有元素包含无穷大或 NaN，并返回结果
bool has_inf_nan() const {
  for (int i = 0; i < N; ++i) {
    // 调用 values[i].has_inf_nan() 检查第 i 个元素是否包含无穷大或 NaN
    if (values[i].has_inf_nan()) {
      return true;
    }
  }
  return false;
}

// 将向量中每个元素应用函数 f，并返回结果向量
VectorizedN<T, N> map(T (*const f)(T)) const {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    // 调用 values[i].map(f) 对第 i 个元素应用函数 f 并存入结果向量
    result.values[i] = values[i].map(f);
  }
  }

  // 将函数 f 映射到当前向量的每个元素上，并返回结果向量
  VectorizedN<T, N> map(T (*const f)(const T&)) const {
    // 创建一个新的 VectorizedN 对象用于存储映射后的结果
    VectorizedN<T, N> result;
    // 遍历当前向量中的每个元素
    for (int i = 0; i < N; ++i) {
      // 将函数 f 应用到当前元素的每个分量上，并将结果存入 result 向量中
      result.values[i] = values[i].map(f);
    }
    // 返回包含映射结果的新 VectorizedN 对象
    return result;
  }

  // 定义一元操作符 abs 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(abs)
  // 定义一元操作符 sgn 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(sgn)
  // 定义一元操作符 angle 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(angle)
  // 定义一元操作符 real 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(real)
  // 定义一元操作符 imag 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(imag)
  // 定义一元操作符 conj 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(conj)
  // 定义一元操作符 acos 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(acos)
  // 定义一元操作符 acosh 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(acosh)
  // 定义一元操作符 asin 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(asin)
  // 定义一元操作符 atan 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(atan)
  // 定义一元操作符 atanh 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(atanh)
  // 定义二元操作符 atan2 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(atan2)
  // 定义二元操作符 copysign 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(copysign)
  // 定义一元操作符 erf 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(erf)
  // 定义一元操作符 erfc 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(erfc)
  // 定义一元操作符 erfinv 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(erfinv)
  // 定义一元操作符 exp 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(exp)
  // 定义一元操作符 exp2 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(exp2)
  // 定义一元操作符 expm1 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(expm1)
  // 定义一元操作符 exp_u20 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(exp_u20)
  // 定义一元操作符 frac 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(frac)
  // 定义二元操作符 fmod 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(fmod)
  // 定义一元操作符 log 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(log)
  // 定义一元操作符 log10 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(log10)
  // 定义一元操作符 log1p 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(log1p)
  // 定义一元操作符 log2 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(log2)
  // 定义一元操作符 ceil 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(ceil)
  // 定义一元操作符 cos 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(cos)
  // 定义一元操作符 cosh 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(cosh)
  // 定义一元操作符 floor 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(floor)
  // 定义二元操作符 hypot 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(hypot)
  // 定义一元操作符 i0 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(i0)
  // 定义一元操作符 i0e 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(i0e)
  // 定义一元操作符 digamma 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(digamma)
  // 定义二元操作符 igamma 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(igamma)
  // 定义二元操作符 igammac 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(igammac)
  // 定义一元操作符 neg 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(neg)
  // 定义二元操作符 nextafter 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(nextafter)
  // 定义一元操作符 round 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(round)
  // 定义一元操作符 sin 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(sin)
  // 定义一元操作符 sinh 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(sinh)
  // 定义一元操作符 tan 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(tan)
  // 定义一元操作符 tanh 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(tanh)
  // 定义一元操作符 trunc 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(trunc)
  // 定义一元操作符 lgamma 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(lgamma)
  // 定义一元操作符 sqrt 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(sqrt)
  // 定义一元操作符 reciprocal 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(reciprocal)
  // 定义一元操作符 rsqrt 的向量化版本
  VECTORIZEDN_DEFINE_UNARY_OP(rsqrt)
  // 定义二元操作符 pow 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(pow)
  // 定义二元操作符 operator== 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(operator==)
  // 定义二元操作符 operator!= 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(operator!=)
  // 定义二元操作符 operator>= 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(operator>=)
  // 定义二元操作符 operator<= 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(operator<=)
  // 定义二元操作符 operator> 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(operator>)
  // 定义二元操作符 operator< 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(operator<)
  // 定义二元操作符 eq 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(eq)
  // 定义二元操作符 ne 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(ne)
  // 定义二元操作符 gt 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(gt)
  // 定义二元操作符 ge 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(ge)
  // 定义二元操作符 lt 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(lt)
  // 定义二元操作符 le 的向量化版本
  VECTORIZEDN_DEFINE_BINARY_OP(le)
// 取消宏定义 VECTORIZEDN_DEFINE_UNARY_OP
// 取消宏定义 VECTORIZEDN_DEFINE_BINARY_OP
};

// 定义全局宏 VECTORIZEDN_DEFINE_UNARY_OP_GLOBAL(op)，接受一元操作符 'op'
template <typename T, int N>
inline VectorizedN<T, N> op(const VectorizedN<T, N>& a) {
  // 调用 a 的 unary_op 方法，传入 lambda 表达式，该 lambda 表达式内部调用 op 操作符
  return a.unary_op([](const Vectorized<T>& a) { return op(a); });
}

// 定义全局宏 VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(op)，接受二元操作符 'op'
template <typename T, int N>
inline VectorizedN<T, N> op(
    const VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  // 调用 a 的 binary_op 方法，传入 lambda 表达式，该 lambda 表达式内部调用 op 操作符
  return a.binary_op(b, [](const Vectorized<T>& a, const Vectorized<T>& b) {
    return op(a, b);
  });
}

// 定义全局宏 VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(op)，接受原地二元操作符 'op'
template <typename T, int N>
inline VectorizedN<T, N>& op(
    VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  // 将 a 赋值为 a 的 binary_op 方法的结果，传入 lambda 表达式，该 lambda 表达式内部调用 op 操作符
  a = a.binary_op(b, [](const Vectorized<T>& a, const Vectorized<T>& b) {
    return op(a, b);
  });
  return a;
}

// 使用全局宏定义二元操作符 +
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator+)
// 使用全局宏定义二元操作符 -
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator-)
// 使用全局宏定义二元操作符 *
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator*)
// 使用全局宏定义二元操作符 /
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator/)
// 使用全局宏定义二元操作符 %
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator%)
// 使用全局宏定义二元操作符 ||
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator||)
// 使用全局宏定义二元操作符 <<
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator<<)
// 使用全局宏定义二元操作符 >>
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator>>)
// 使用全局宏定义二元操作符 maximum
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(maximum)
// 使用全局宏定义二元操作符 minimum
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(minimum)
// 使用全局宏定义二元操作符 fmadd
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(fmadd)
// 使用全局宏定义二元操作符 fmsub
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(fmsub)
// 使用全局宏定义二元操作符 clamp
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(clamp)
// 使用全局宏定义二元操作符 clamp_max
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(clamp_max)
// 使用全局宏定义二元操作符 clamp_min
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(clamp_min)
// 使用全局宏定义二元操作符 &
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator&)
// 使用全局宏定义二元操作符 |
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator|)
// 使用全局宏定义二元操作符 ^
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator^)
// 使用全局宏定义一元操作符 ~
VECTORIZEDN_DEFINE_UNARY_OP_GLOBAL(operator~)

// 使用全局宏定义原地二元操作符 +=
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator+=)
// 使用全局宏定义原地二元操作符 -=
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator-=)
// 使用全局宏定义原地二元操作符 *=
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator*=)
// 使用全局宏定义原地二元操作符 /=
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator/=)
// 使用全局宏定义原地二元操作符 %=
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator%=)
// 使用全局宏定义原地二元操作符 <<=
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator<<=)
// 使用全局宏定义原地二元操作符 >>=
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator>>=)

// 取消宏定义 VECTORIZEDN_DEFINE_UNARY_OP_GLOBAL
// 取消宏定义 VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL
// 取消宏定义 VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL
# 定义一个模板函数，用于对一个向量化数据结构中的元素进行归约操作
# T: 数据类型
# N: 向量中元素的数量
# OpVec: 归约操作的函数对象类型
template <typename T, int N, typename OpVec>
inline T vec_reduce_all(const OpVec& vec_fun, VectorizedN<T, N> acc_vec) {
    # 创建一个变量 vec_result，用于存储归约的结果，初始化为第一个向量化元素
    Vectorized<T> vec_result = acc_vec[0];
    
    # 循环遍历余下的 N-1 个向量化元素
    for (int i = 1; i < N; i++) {
        # 使用 vec_fun 对 vec_result 和 acc_vec[i] 进行归约操作，并更新 vec_result
        vec_result = vec_fun(vec_result, acc_vec[i]);
    }
    
    # 递归调用 vec_reduce_all 函数，将归约的结果 vec_result 作为参数继续归约
    return vec_reduce_all(vec_fun, vec_result);
}

# 结束 CPU_CAPABILITY 命名空间
} // namespace CPU_CAPABILITY

# 结束 at::vec 命名空间
} // namespace at::vec
```