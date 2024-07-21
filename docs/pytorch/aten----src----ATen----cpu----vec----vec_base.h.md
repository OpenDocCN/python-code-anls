# `.\pytorch\aten\src\ATen\cpu\vec\vec_base.h`

```py
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]
//
// Note [Do not compile initializers with AVX]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// If you define a static initializer in this file, the initialization will use
// AVX instructions because these object files are compiled with AVX enabled.
// We need to avoid non-trivial global data in these architecture specific files
// because there's no way to guard the global initializers with CPU capability
// detection.
//
// See https://github.com/pytorch/pytorch/issues/37577 for an instance
// of this bug in the past.

#include <array>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <cmath>
#include <type_traits>
#include <climits>

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/native/Math.h>
#include <ATen/NumericUtils.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/copysign.h>
#include <ATen/native/cpu/zmath.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <c10/util/Load.h>

#if defined(__GNUC__)
#define __FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define __FORCE_INLINE __forceinline
#endif

#if defined(_MSC_FULL_VER)
/*
https://learn.microsoft.com/en-us/cpp/overview/compiler-versions?view=msvc-170
Use _MSC_FULL_VER to identify current compiler is msvc,
Windows llvm will not have this defination.
*/
#define __msvc_cl__
#endif

// These macros helped us unify vec_base.h
#ifdef CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(64)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(64))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 64
#define int_vector __m512i
#else // CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(32))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 32
#define int_vector __m256i
#endif // CPU_CAPABILITY_AVX512

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {
// at::Half and at::BFloat16 should be treated as floating point

// 检查模板类型是否为浮点数类型（包括 at::Half 和 at::BFloat16）
template <typename T>
struct is_floating_point:
    std::integral_constant<bool,
      std::is_floating_point_v<T> ||
      std::is_same_v<T, at::Half> ||
      std::is_same_v<T, at::BFloat16>> {
};

// 是否为浮点数类型的模板变量
template<typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

// 检查模板类型是否为缩减的浮点数类型（at::Half 或 at::BFloat16）
template <typename T>
struct is_reduced_floating_point:
    std::integral_constant<bool,
      std::is_same_v<T, at::Half> ||
      std::is_same_v<T, at::BFloat16>> {
};

// 是否为缩减浮点数类型的模板变量
template <typename T>
constexpr bool is_reduced_floating_point_v = is_reduced_floating_point<T>::value;

// 检查模板类型是否为8位整数类型
template <typename T>
struct is_8bit_integer:
    // 定义一个继承自 std::integral_constant 的模板类，该类表示一个布尔常量
    std::integral_constant<bool,
      // 使用 std::is_same_v 判断模板参数 T 是否与 unsigned char 或 signed char 相同，并返回布尔值
      std::is_same_v<T, unsigned char> ||
      std::is_same_v<T, signed char>> {
// 结束宏定义区域
};

// 定义模板，检查类型是否为8位整数
template <typename T>
constexpr bool is_8bit_integer_v = is_8bit_integer<T>::value;

// 模板特化，定义大小为n的整数类型
template<size_t n> struct int_of_size;

// 宏定义，用于定义int_of_size模板的特化
#define DEFINE_INT_OF_SIZE(int_t) \
template<> struct int_of_size<sizeof(int_t)> { using type = int_t; }

// 特化int_of_size模板，定义不同大小整数类型
DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

// 解除宏定义
#undef DEFINE_INT_OF_SIZE

// 模板，使用int_of_size模板的特化，获取与T大小相同的整数类型
template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

// 注意：如果你对一种类型进行特化，你必须定义所有操作！

// 模拟向量化类型
#if defined(__s390x__)
template <class T, class TEMP=void>
#else
template <class T>
#endif
struct Vectorized {
private:
  // 在匿名命名空间内定义向量化数据，类型为T，长度为VECTOR_WIDTH / sizeof(T)
  __at_align__ T values[VECTOR_WIDTH / sizeof(T)];
public:
  using value_type = T;  // 值类型为T
  using size_type = int;  // 大小类型为int

  // 注意 [constexpr static function to avoid odr-usage compiler bug]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 为什么要将size定义为静态constexpr函数，而不是普通的静态constexpr int size;变量？
  // 问题在于ODR规则对静态constexpr成员与静态成员的区别。因为这个类（及其所有派生类）位于
  // 匿名命名空间中：它们被完全内联到其使用位置，因为我们需要为不同的指令集多次编译它。

  // 由于这个约束，我们不能为这个类中的任何静态成员提供单一的定义；因为我们想要多次编译这个类，
  // 实际上没有一个好的地方放置定义。现在问题是：如果我们ODR使用了一个静态constexpr成员，
  // 我们*必须*提供一个定义。没有定义，你会得到一个编译错误，比如：
  //
  //    relocation R_X86_64_PC32 against undefined symbol
  //    `_ZN2at6vec25612_GLOBAL__N_16VectorizedIdE4sizeE' can not be used when making
  //    a shared object; recompile with -fPIC
  //
  // 如果这是C++17，我们可以用内联变量替换静态constexpr变量，它不需要一个定义。但我们现在不是
  // C++17。所以下一个最好的办法是用一个静态constexpr（因此是内联的）函数替换成员，它也不需要
  // ODR。另外，根据C++标准，如果我们从未ODR使用一个constexpr变量，我们不必定义它。但是似乎
  // 一些版本的GCC/Clang在判断一个标识符是否ODR使用时存在缺陷，而且无论如何很难判断一个变量是否
  // 被ODR使用。所以最好从根本上解决问题。

  // 所以最好的办法是用一个静态constexpr函数替换这个成员，它不需要ODR定义。
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(T);  // 返回向量宽度除以T类型大小的结果
  }

  // 默认构造函数，将values数组所有元素初始化为0
  Vectorized() : values{static_cast<T>(0)} {}

  // 带参构造函数，将values数组所有元素初始化为val
  Vectorized(T val) {
    for (int i = 0; i != size(); i++) {  // 遍历values数组
      values[i] = val;  // 将val赋值给values数组中的每个元素
  }
  }
  // 模板构造函数，接受任意数量的参数并将它们初始化到 values 数组中
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) : values{vals...}{
  }
  // 转换操作符，返回 values 数组的常量指针
  // 这也暗示了 const T& operator[](int idx) const
  inline operator const T*() const {
    return values;
  }
  // 转换操作符，返回 values 数组的非常量指针
  // 这也暗示了 T& operator[](int idx)
  inline operator T*() {
    return values;
  }
  // 返回 values 数组的 char* 表示，用于类型解释
  auto as_bytes() const -> const char* {
    return reinterpret_cast<const char*>(values);
  }
  // 根据给定的掩码 mask_ 对两个 Vectorized 对象 a 和 b 进行混合
  template <int64_t mask_>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    int64_t mask = mask_;
    Vectorized vector;
    for (const auto i : c10::irange(size())) {
      if (mask & 0x01) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
      mask = mask >> 1;
    }
    return vector;
  }
  // 根据给定的掩码向量 mask 对两个 Vectorized 对象 a 和 b 进行混合
  static Vectorized<T> blendv(const Vectorized<T>& a, const Vectorized<T>& b,
                          const Vectorized<T>& mask) {
    Vectorized vector;
    int_same_size_t<T> buffer[size()];
    mask.store(buffer);
    for (const auto i : c10::irange(size())) {
      if (buffer[i] & 0x01)
       {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  // 生成从 base 开始，步长为 step 的向量
  template<typename step_t>  // step 有时需要更高精度类型（例如，T=int, step_t=double）
  static Vectorized<T> arange(T base = static_cast<T>(0), step_t step = static_cast<step_t>(1)) {
    Vectorized vector;
    for (const auto i : c10::irange(size())) {
      vector.values[i] = base + i * step;
    }
    return vector;
  }
  // 使用给定的向量 a 和 b 以及 count 设置 Vectorized 对象
  static Vectorized<T> set(const Vectorized<T>& a, const Vectorized<T>& b, int64_t count = size()) {
    Vectorized vector;
    for (const auto i : c10::irange(size())) {
      if (i < count) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  // 从 void* 指针加载数据到 Vectorized 对象中
  static Vectorized<T> loadu(const void* ptr) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, VECTOR_WIDTH);
    return vector;
  }
  // 从 void* 指针加载 count 个元素到 Vectorized 对象中
  static Vectorized<T> loadu(const void* ptr, int64_t count) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, count * sizeof(T));
    return vector;
  }
  // 加载四分之一长度的数据到 Vectorized 对象中（仅适用于字节类型）
  static Vectorized<T> loadu_one_fourth(const void* ptr) {
    static_assert(std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>, "For byte types only");
    return Vectorized::loadu(ptr, 8);
  }

  // 将 values 数组中的数据存储到指定的 void* 指针中
  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }
  // 返回一个整数掩码，其中所有零元素被转换为 1 位，其他元素被转换为 0 位
  int zero_mask() const {
    int mask = 0;
    for (int i = 0; i < size(); ++ i) {
      if (values[i] == static_cast<T>(0)) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  // 返回一个 Vectorized 对象，表示 values 数组中每个元素是否为 NaN
  Vectorized<T> isnan() const {
    Vectorized<T> vector;
    // 循环遍历当前向量中的每个元素
    for (int64_t i = 0; i != size(); i++) {
      // 检查当前元素是否为 NaN（Not a Number）
      if (_isnan(values[i])) {
        // 如果是 NaN，则将向量中第 i 个元素的内存内容设置为 0xFF
        std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
      } else {
        // 如果不是 NaN，则将向量中第 i 个元素的内存内容设置为 0
        std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
      }
    }
    // 返回修改后的向量
    return vector;
  }
  // 检查向量中是否存在无穷大或 NaN（Not a Number）
  bool has_inf_nan() const {
    for (int64_t i = 0; i != size(); i++) {
      // 如果当前元素是 NaN 或者无穷大，则返回 true
      if(_isnan(values[i]) || _isinf(values[i])) {
        return true;
      }
    }
    // 如果没有发现 NaN 或者无穷大，则返回 false
    return false;
  }
  // 对向量中的每个元素应用指定的函数并返回结果
  Vectorized<T> map(T (*const f)(T)) const {
    Vectorized<T> ret;
    // 循环遍历向量中的每个元素，并将应用函数后的结果存储在新的向量中
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    // 返回应用函数后的新向量
    return ret;
  }
  // 对向量中的每个元素应用指定的函数并返回结果（函数参数为常量引用）
  Vectorized<T> map(T (*const f)(const T &)) const {
    Vectorized<T> ret;
    // 循环遍历向量中的每个元素，并将应用函数后的结果存储在新的向量中
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    // 返回应用函数后的新向量
    return ret;
  }
  // 计算向量中每个元素的绝对值，但不适用于浮点数和复数类型（SFINAE机制）
  template <typename other_t_abs = T,
            typename std::enable_if_t<!is_floating_point_v<other_t_abs> && !c10::is_complex<other_t_abs>::value, int> = 0>
  Vectorized<T> abs() const {
    // other_t_abs 是为了 SFINAE 和代码清晰性而设，确保其不被改变
    static_assert(std::is_same_v<other_t_abs, T>, "other_t_abs must be T");
    // 使用 map 函数对每个元素应用绝对值函数
    return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
  }
  // 计算浮点数类型向量中每个元素的绝对值（SFINAE机制）
  template <typename float_t_abs = T,
            typename std::enable_if_t<is_floating_point_v<float_t_abs>, int> = 0>
  Vectorized<T> abs() const {
    // float_t_abs 是为了 SFINAE 和代码清晰性而设，确保其不被改变
    static_assert(std::is_same_v<float_t_abs, T>, "float_t_abs must be T");
    // 使用 map 函数对每个元素应用绝对值函数，特别处理浮点数的 -0.0 问题
    return map([](T x) -> T { return std::abs(x); });
  }
  // 计算复数类型向量中每个元素的绝对值（SFINAE机制）
  template <typename complex_t_abs = T,
            typename std::enable_if_t<c10::is_complex<complex_t_abs>::value, int> = 0>
  Vectorized<T> abs() const {
    // complex_t_abs 是为了 SFINAE 和代码清晰性而设，确保其不被改变
    static_assert(std::is_same_v<complex_t_abs, T>, "complex_t_abs must be T");
    // 使用 map 函数对每个元素应用复数的绝对值函数
    return map([](T x) { return static_cast<T>(std::abs(x)); });
  }

  // 对复数类型向量中每个元素应用符号函数
  template <typename other_t_sgn = T,
            typename std::enable_if_t<c10::is_complex<other_t_sgn>::value, int> = 0>
  Vectorized<T> sgn() const {
    // 使用 map 函数对每个元素应用符号函数
    return map(at::native::sgn_impl);
  }

  // 对非复数类型向量中每个元素应用角度函数
  template <typename other_t_angle = T,
            typename std::enable_if_t<!c10::is_complex<other_t_angle>::value, int> = 0>
  Vectorized<T> angle() const {
    // other_t_angle 是为了 SFINAE 和代码清晰性而设，确保其不被改变
    static_assert(std::is_same_v<other_t_angle, T>, "other_t_angle must be T");
    // 使用 map 函数对每个元素应用角度函数
    return map(at::native::angle_impl<T>);  // 编译器无法在没有 <T> 的情况下解析重载
  }
  // 对复数类型向量中每个元素应用角度函数
  template <typename complex_t_angle = T,
            typename std::enable_if_t<c10::is_complex<complex_t_angle>::value, int> = 0>
  Vectorized<T> angle() const {
  // complex_t_angle is for SFINAE and clarity. Make sure it is not changed.
  static_assert(std::is_same_v<complex_t_angle, T>, "complex_t_angle must be T");
  // 返回一个 Vectorized 对象，其中每个元素是输入向量的幅角
  return map([](T x) { return static_cast<T>(std::arg(x)); });
}
template <typename other_t_real = T,
          typename std::enable_if_t<!c10::is_complex<other_t_real>::value, int> = 0>
Vectorized<T> real() const {
  // other_t_real is for SFINAE and clarity. Make sure it is not changed.
  static_assert(std::is_same_v<other_t_real, T>, "other_t_real must be T");
  // 对于非复数类型，返回当前向量本身
  return *this;
}
template <typename complex_t_real = T,
          typename std::enable_if_t<c10::is_complex<complex_t_real>::value, int> = 0>
Vectorized<T> real() const {
  // complex_t_real is for SFINAE and clarity. Make sure it is not changed.
  static_assert(std::is_same_v<complex_t_real, T>, "complex_t_real must be T");
  // 返回一个 Vectorized 对象，其中每个元素是输入向量的实部
  return map([](T x) { return static_cast<T>(x.real()); });
}
template <typename other_t_imag = T,
          typename std::enable_if_t<!c10::is_complex<other_t_imag>::value, int> = 0>
Vectorized<T> imag() const {
  // other_t_imag is for SFINAE and clarity. Make sure it is not changed.
  static_assert(std::is_same_v<other_t_imag, T>, "other_t_imag must be T");
  // 对于非复数类型，返回一个全为零的 Vectorized 对象
  return Vectorized(0);
}
template <typename complex_t_imag = T,
          typename std::enable_if_t<c10::is_complex<complex_t_imag>::value, int> = 0>
Vectorized<T> imag() const {
  // complex_t_imag is for SFINAE and clarity. Make sure it is not changed.
  static_assert(std::is_same_v<complex_t_imag, T>, "complex_t_imag must be T");
  // 返回一个 Vectorized 对象，其中每个元素是输入向量的虚部
  return map([](T x) { return static_cast<T>(x.imag()); });
}
template <typename other_t_conj = T,
          typename std::enable_if_t<!c10::is_complex<other_t_conj>::value, int> = 0>
Vectorized<T> conj() const {
  // other_t_conj is for SFINAE and clarity. Make sure it is not changed.
  static_assert(std::is_same_v<other_t_conj, T>, "other_t_conj must be T");
  // 对于非复数类型，返回当前向量本身
  return *this;
}
template <typename complex_t_conj = T,
          typename std::enable_if_t<c10::is_complex<complex_t_conj>::value, int> = 0>
Vectorized<T> conj() const {
  // complex_t_conj is for SFINAE and clarity. Make sure it is not changed.
  static_assert(std::is_same_v<complex_t_conj, T>, "complex_t_conj must be T");
  // 返回一个 Vectorized 对象，其中每个元素是输入向量的共轭
  return map([](T x) { return static_cast<T>(std::conj(x)); });
}
// 返回一个新的 Vectorized 对象，其中每个元素是输入向量经过 acos 函数处理的结果
Vectorized<T> acos() const {
  return map(std::acos);
}
// 返回一个新的 Vectorized 对象，其中每个元素是输入向量经过 acosh 函数处理的结果
Vectorized<T> acosh() const {
  return map(std::acosh);
}
// 返回一个新的 Vectorized 对象，其中每个元素是输入向量经过 asin 函数处理的结果
Vectorized<T> asin() const {
  return map(std::asin);
}
// 返回一个新的 Vectorized 对象，其中每个元素是输入向量经过 atan 函数处理的结果
Vectorized<T> atan() const {
  return map(std::atan);
}
// 返回一个新的 Vectorized 对象，其中每个元素是输入向量经过 atanh 函数处理的结果
Vectorized<T> atanh() const {
  return map(std::atanh);
}
// 返回一个新的 Vectorized 对象，其中每个元素是当前向量与给定向量逐元素进行 atan2 计算的结果
Vectorized<T> atan2(const Vectorized<T> &exp) const {
  Vectorized<T> ret;
  for (const auto i : c10::irange(size())) {
    ret[i] = std::atan2(values[i], exp[i]);
  }
  return ret;
}
template <
  typename U = T,
  typename std::enable_if_t<is_floating_point_v<U>, int> = 0>
Vectorized<T> copysign(const Vectorized<T> &sign) const {
  Vectorized<T> ret;
  // 循环遍历向量中的每个元素，将其赋值为 values[i] 的符号与 sign[i] 相同的值
  for (size_type i = 0; i < size(); i++) {
    ret[i] = c10::copysign(values[i], sign[i]);
  }
  return ret;
}
Vectorized<T> erf() const {
  // 返回应用 std::erf 函数到当前向量的结果
  return map(std::erf);
}
Vectorized<T> erfc() const {
  // 返回应用 std::erfc 函数到当前向量的结果
  return map(std::erfc);
}
Vectorized<T> erfinv() const {
  // 返回应用 calc_erfinv 函数到当前向量的结果
  return map(calc_erfinv);
}
Vectorized<T> exp() const {
  // 返回应用 std::exp 函数到当前向量的结果
  return map(std::exp);
}
Vectorized<T> exp2() const {
  // 返回应用 exp2_impl 函数到当前向量的结果
  return map(exp2_impl);
}
Vectorized<T> expm1() const {
  // 返回应用 std::expm1 函数到当前向量的结果
  return map(std::expm1);
}
Vectorized<T> exp_u20() const {
  // 返回应用 std::exp 函数到当前向量的结果
  return map(std::exp);
}
Vectorized<T> frac() const {
  // 返回当前向量减去自身取整的结果
  return *this - this->trunc();
}
template <
  typename U = T,
  typename std::enable_if_t<is_floating_point_v<U>, int> = 0>
Vectorized<T> fmod(const Vectorized<T>& q) const {
  // U 用于 SFINAE 目的，确保其不会改变
  static_assert(std::is_same_v<U, T>, "U must be T");
  Vectorized<T> ret;
  // 循环遍历向量中的每个元素，将其赋值为 values[i] 取 q[i] 余数的结果
  for (const auto i : c10::irange(size())) {
    ret[i] = std::fmod(values[i], q[i]);
  }
  return ret;
}
Vectorized<T> log() const {
  // 返回应用 std::log 函数到当前向量的结果
  return map(std::log);
}
Vectorized<T> log10() const {
  // 返回应用 std::log10 函数到当前向量的结果
  return map(std::log10);
}
Vectorized<T> log1p() const {
  // 返回应用 std::log1p 函数到当前向量的结果
  return map(std::log1p);
}
template <typename other_t_log2 = T,
          typename std::enable_if_t<!c10::is_complex<other_t_log2>::value, int> = 0>
Vectorized<T> log2() const {
  // other_t_log2 用于 SFINAE 和清晰度。确保其不会改变
  static_assert(std::is_same_v<other_t_log2, T>, "other_t_log2 must be T");
  // 返回应用 std::log2 函数到当前向量的结果
  return map(std::log2);
}
template <typename complex_t_log2 = T,
          typename std::enable_if_t<c10::is_complex<complex_t_log2>::value, int> = 0>
Vectorized<T> log2() const {
  // complex_t_log2 用于 SFINAE 和清晰度。确保其不会改变
  static_assert(std::is_same_v<complex_t_log2, T>, "complex_t_log2 must be T");
  const T log_2 = T(std::log(2.0));
  // 返回应用 std::log 函数到当前向量的结果，再除以 log_2 得到对数以 2 为底的结果
  return Vectorized(map(std::log))/Vectorized(log_2);
}
Vectorized<T> ceil() const {
  // 返回应用 at::native::ceil_impl 函数到当前向量的结果
  return map(at::native::ceil_impl);
}
Vectorized<T> cos() const {
  // 返回应用 std::cos 函数到当前向量的结果
  return map(std::cos);
}
Vectorized<T> cosh() const {
  // 返回应用 std::cosh 函数到当前向量的结果
  return map(std::cosh);
}
Vectorized<T> floor() const {
  // 返回应用 at::native::floor_impl 函数到当前向量的结果
  return map(at::native::floor_impl);
}
Vectorized<T> hypot(const Vectorized<T> &b) const {
  Vectorized<T> ret;
  // 循环遍历向量中的每个元素，将其赋值为 values[i] 和 b[i] 的直角三角形的斜边长度
  for (const auto i : c10::irange(size())) {
    ret[i] = std::hypot(values[i], b[i]);
  }
  return ret;
}
Vectorized<T> i0() const {
  // 返回应用 calc_i0 函数到当前向量的结果
  return map(calc_i0);
}
Vectorized<T> i0e() const {
  // 返回应用 calc_i0e 函数到当前向量的结果
  return map(calc_i0e);
}
Vectorized<T> digamma() const {
  // 返回应用 calc_digamma 函数到当前向量的结果
  return map(calc_digamma);
}
Vectorized<T> igamma(const Vectorized<T> &x) const {
  Vectorized<T> ret;
  // 循环遍历向量中的每个元素，将其赋值为计算 values[i] 和 x[i] 的 gamma 函数的结果
  for (const auto i : c10::irange(size())) {
    ret[i] = calc_igamma(values[i], x[i]);
  }
  return ret;
}
Vectorized<T> igammac(const Vectorized<T> &x) const {
  Vectorized<T> ret;
  // 对于每个索引 i，在 values 中计算 igammac 函数的结果并存储在 ret 中
  for (const auto i : c10::irange(size())) {
    ret[i] = calc_igammac(values[i], x[i]);
  }
  return ret;
}
Vectorized<T> neg() const {
  // 注意：需要使用尾部返回类型，因为我们需要将返回值强制转换回 T 类型，以确保在一元运算符- 中发生提升
  return map([](T x) -> T { return -x; });
}
Vectorized<T> nextafter(const Vectorized<T> &b) const {
  Vectorized<T> ret;
  // 对于每个索引 i，在 values 中计算 std::nextafter(values[i], b[i]) 的结果并存储在 ret 中
  for (const auto i : c10::irange(size())) {
    ret[i] = std::nextafter(values[i], b[i]);
  }
  return ret;
}
Vectorized<T> round() const {
  // 我们不使用 std::round，因为我们希望将中间值四舍五入到最接近的偶数
  return map(at::native::round_impl);
}
Vectorized<T> sin() const {
  // 对于每个值调用 std::sin 函数，并将结果存储在返回的 Vectorized<T> 对象中
  return map(std::sin);
}
Vectorized<T> sinh() const {
  // 对于每个值调用 std::sinh 函数，并将结果存储在返回的 Vectorized<T> 对象中
  return map(std::sinh);
}
Vectorized<T> tan() const {
  // 对于每个值调用 std::tan 函数，并将结果存储在返回的 Vectorized<T> 对象中
  return map(std::tan);
}
Vectorized<T> tanh() const {
  // 对于每个值调用 std::tanh 函数，并将结果存储在返回的 Vectorized<T> 对象中
  return map(std::tanh);
}
Vectorized<T> trunc() const {
  // 对于每个值调用 at::native::trunc_impl 函数，并将结果存储在返回的 Vectorized<T> 对象中
  return map(at::native::trunc_impl);
}
Vectorized<T> lgamma() const {
  // 对于每个值调用 std::lgamma 函数，并将结果存储在返回的 Vectorized<T> 对象中
  return map(std::lgamma);
}
Vectorized<T> sqrt() const {
  // 对于每个值调用 std::sqrt 函数，并将结果存储在返回的 Vectorized<T> 对象中
  return map(std::sqrt);
}
Vectorized<T> reciprocal() const {
  // 对于每个值计算其倒数，并将结果存储在返回的 Vectorized<T> 对象中
  return map([](T x) { return (T)(1) / x; });
}
Vectorized<T> rsqrt() const {
  // 对于每个值计算其平方根的倒数，并将结果存储在返回的 Vectorized<T> 对象中
  return map([](T x) { return (T)1 / std::sqrt(x); });
}
Vectorized<T> pow(const Vectorized<T> &exp) const {
  Vectorized<T> ret;
  // 对于每个索引 i，在 values 中计算 std::pow(values[i], exp[i]) 的结果并存储在 ret 中
  for (const auto i : c10::irange(size())) {
    ret[i] = std::pow(values[i], exp[i]);
  }
  return ret;
}
// 定义了一个私有成员函数模板，用于执行二进制谓词操作，返回一个向量化对象
template <typename Op>
inline Vectorized<T> binary_pred(const Vectorized<T>& other, Op op) const {
    // 如果谓词为真，则所有位被设置为1，否则为0
    Vectorized<T> vector;
    // 遍历当前对象的元素
    for (int64_t i = 0; i != size(); i++) {
        // 根据谓词op对当前对象的第i个值和另一个向量的第i个值进行比较
        if (op(values[i], other.values[i])) {
            // 如果谓词为真，将vector中第i个值的内存内容设置为全1
            std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
        } else {
            // 如果谓词为假，将vector中第i个值的内存内容设置为全0
            std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
        }
    }
    // 返回生成的向量化对象
    return vector;
}

// 定义了公共成员函数，用于实现相等比较操作符的重载
Vectorized<T> operator==(const Vectorized<T>& other) const { return binary_pred(other, std::equal_to<T>()); }
Vectorized<T> operator!=(const Vectorized<T>& other) const { return binary_pred(other, std::not_equal_to<T>()); }
Vectorized<T> operator>=(const Vectorized<T>& other) const { return binary_pred(other, std::greater_equal<T>()); }
Vectorized<T> operator<=(const Vectorized<T>& other) const { return binary_pred(other, std::less_equal<T>()); }
Vectorized<T> operator>(const Vectorized<T>& other) const { return binary_pred(other, std::greater<T>()); }
Vectorized<T> operator<(const Vectorized<T>& other) const { return binary_pred(other, std::less<T>()); }

// 定义了一个私有成员函数模板，用于执行布尔类型的二进制谓词操作，返回一个向量化对象
template <typename Op>
inline Vectorized<T> binary_pred_bool(const Vectorized<T>& other, Op op) const {
    // 如果谓词为真，则设置为1，否则为0
    Vectorized<T> vector;
    // 遍历当前对象的元素
    for (int i = 0; i != size(); ++i) {
        // 根据谓词op对当前对象的第i个值和另一个向量的第i个值进行比较，并将结果转换为T类型
        vector[i] = static_cast<T>(op(values[i], other.values[i]));
    }
    // 返回生成的向量化对象
    return vector;
}

// 定义了公共成员函数，用于实现布尔类型的比较操作
Vectorized<T> eq(const Vectorized<T>& other) const { return binary_pred_bool(other, std::equal_to<T>()); }
Vectorized<T> ne(const Vectorized<T>& other) const { return binary_pred_bool(other, std::not_equal_to<T>()); }
Vectorized<T> gt(const Vectorized<T>& other) const { return binary_pred_bool(other, std::greater<T>()); }
Vectorized<T> ge(const Vectorized<T>& other) const { return binary_pred_bool(other, std::greater_equal<T>()); }
Vectorized<T> lt(const Vectorized<T>& other) const { return binary_pred_bool(other, std::less<T>()); }
Vectorized<T> le(const Vectorized<T>& other) const { return binary_pred_bool(other, std::less_equal<T>()); }

// 定义了一个全局函数模板，用于实现向量化对象的加法操作
template <class T>
Vectorized<T> inline operator+(const Vectorized<T> &a, const Vectorized<T> &b) {
    Vectorized<T> c;
    // 遍历向量化对象的元素
    for (int i = 0; i != Vectorized<T>::size(); i++) {
        // 将两个向量化对象对应位置的元素相加，并存储到新的向量化对象c中
        c[i] = a[i] + b[i];
    }
    // 返回生成的向量化对象
    return c;
}

// 定义了一个全局函数模板，用于实现向量化对象的减法操作
template <class T>
Vectorized<T> inline operator-(const Vectorized<T> &a, const Vectorized<T> &b) {
    Vectorized<T> c;
    // 遍历向量化对象的元素
    for (int i = 0; i != Vectorized<T>::size(); i++) {
        // 将两个向量化对象对应位置的元素相减，并存储到新的向量化对象c中
        c[i] = a[i] - b[i];
    }
    // 返回生成的向量化对象
    return c;
}

// 定义了一个全局函数模板，用于实现向量化对象的乘法操作
template <class T>
Vectorized<T> inline operator*(const Vectorized<T> &a, const Vectorized<T> &b) {
    Vectorized<T> c;
    // 遍历向量化对象的元素
    for (int i = 0; i != Vectorized<T>::size(); i++) {
        // 将两个向量化对象对应位置的元素相乘，并存储到新的向量化对象c中
        c[i] = a[i] * b[i];
    }
    // 返回生成的向量化对象
    return c;
}
// 定义向量化除法操作符重载函数，处理两个向量之间的除法运算
template <class T>
Vectorized<T> inline operator/(const Vectorized<T> &a, const Vectorized<T> &b) __ubsan_ignore_float_divide_by_zero__ {
  // 创建一个新的向量 c
  Vectorized<T> c;
  // 对向量中的每个元素执行除法操作
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] / b[i]; // 计算 a[i] 除以 b[i] 的结果，并存储到 c[i] 中
  }
  return c; // 返回结果向量 c
}

// 定义向量化求余操作符重载函数，处理两个向量之间的求余运算
template <class T,
          typename std::enable_if_t<!is_floating_point_v<T>, int> = 0>
Vectorized<T> inline operator%(const Vectorized<T> &a, const Vectorized<T> &b) __ubsan_ignore_float_divide_by_zero__ {
  // 返回向量 a 对向量 b 求余的结果
  return a - a / b * b;
}

// 定义向量化逻辑或操作符重载函数，处理两个向量之间的逻辑或运算
template <class T> Vectorized<T> inline operator||(const Vectorized<T> &a, const Vectorized<T> &b) {
  // 创建一个新的向量 c
  Vectorized<T> c;
  // 对向量中的每个元素执行逻辑或操作
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] || b[i]; // 计算 a[i] 或 b[i] 的结果，并存储到 c[i] 中
  }
  return c; // 返回结果向量 c
}

// 实现 IEEE 754 201X 的最大值操作，如果任一输入为 NaN 则传播 NaN
template <class T,
          typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(const Vectorized<T> &a, const Vectorized<T> &b) {
  // 创建一个新的向量 c
  Vectorized<T> c;
  // 对向量中的每个元素执行最大值比较
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i]; // 如果 a[i] 大于 b[i]，则 c[i] = a[i]，否则 c[i] = b[i]
    if (_isnan(a[i])) {
      // 如果输入中的任一值为 NaN，则传播 NaN
      // 注意：上面的简单三元操作符已正确处理 b[i] 是 NaN 的情况
      c[i] = a[i]; // 将 NaN 传播到结果中
    }
  }
  return c; // 返回结果向量 c
}

// 实现 IEEE 754 201X 的最大值操作，对复数类型的处理，如果任一输入为 NaN 则传播 NaN
template <class T,
          typename std::enable_if_t<c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(const Vectorized<T> &a, const Vectorized<T> &b) {
  // 创建一个新的向量 c
  Vectorized<T> c;
  // 对向量中的每个元素执行最大值比较（使用绝对值比较）
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (std::abs(a[i]) > std::abs(b[i])) ? a[i] : b[i]; // 如果 abs(a[i]) 大于 abs(b[i])，则 c[i] = a[i]，否则 c[i] = b[i]
    if (_isnan(a[i])) {
      // 如果输入中的任一值为 NaN，则传播 NaN
      // 注意：上面的简单三元操作符已正确处理 b[i] 是 NaN 的情况
      c[i] = a[i]; // 将 NaN 传播到结果中
    }
  }
  return c; // 返回结果向量 c
}

// 实现 IEEE 754 201X 的最小值操作，如果任一输入为 NaN 则传播 NaN
template <class T,
          typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T> &a, const Vectorized<T> &b) {
  // 创建一个新的向量 c
  Vectorized<T> c;
  // 对向量中的每个元素执行最小值比较
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i]; // 如果 a[i] 小于 b[i]，则 c[i] = a[i]，否则 c[i] = b[i]
    if (_isnan(a[i])) {
      // 如果输入中的任一值为 NaN，则传播 NaN
      // 注意：上面的简单三元操作符已正确处理 b[i] 是 NaN 的情况
      c[i] = a[i]; // 将 NaN 传播到结果中
    }
  }
  return c; // 返回结果向量 c
}

// 实现 IEEE 754 201X 的最小值操作，对复数类型的处理，如果任一输入为 NaN 则传播 NaN
template <class T,
          typename std::enable_if_t<c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T> &a, const Vectorized<T> &b) {
  // 创建一个新的向量 c
  Vectorized<T> c;
  // 对向量中的每个元素执行最小值比较（使用绝对值比较）
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (std::abs(a[i]) < std::abs(b[i])) ? a[i] : b[i]; // 如果 abs(a[i]) 小于 abs(b[i])，则 c[i] = a[i]，否则 c[i] = b[i]
    if (_isnan(a[i])) {
      // 如果输入中的任一值为 NaN，则传播 NaN
      // 注意：上面的简单三元操作符已正确处理 b[i] 是 NaN 的情况
      c[i] = a[i]; // 将 NaN 传播到结果中
    }
  }
  return c; // 返回结果向量 c
}
}

// clamp 函数模板：对向量 a 中的每个元素进行范围约束，返回约束后的向量
template <class T,
          typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp(const Vectorized<T> &a, const Vectorized<T> &min_vec, const Vectorized<T> &max_vec) {
  // 创建一个新的向量 c 来存储结果
  Vectorized<T> c;
  // 遍历向量 a 中的每个元素
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    // 使用 std::min 和 std::max 函数对每个元素进行范围约束
    c[i] = std::min(std::max(a[i], min_vec[i]), max_vec[i]);
  }
  return c; // 返回范围约束后的向量
}

// clamp_max 函数模板：对向量 a 中的每个元素进行上限约束，返回约束后的向量
template <class T,
          typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_max(const Vectorized<T> &a, const Vectorized<T> &max_vec) {
  // 创建一个新的向量 c 来存储结果
  Vectorized<T> c;
  // 遍历向量 a 中的每个元素
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    // 如果元素超过了上限，则使用上限值；否则保持原值
    c[i] = a[i] > max_vec[i] ? max_vec[i] : a[i];
  }
  return c; // 返回上限约束后的向量
}

// clamp_min 函数模板：对向量 a 中的每个元素进行下限约束，返回约束后的向量
template <class T,
          typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_min(const Vectorized<T> &a, const Vectorized<T> &min_vec) {
  // 创建一个新的向量 c 来存储结果
  Vectorized<T> c;
  // 遍历向量 a 中的每个元素
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    // 如果元素低于下限，则使用下限值；否则保持原值
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c; // 返回下限约束后的向量
}

// bitwise_binary_op 函数模板：执行位运算操作，并返回结果向量
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
template <class T, typename Op>
static inline Vectorized<T> bitwise_binary_op(const Vectorized<T> &a, const Vectorized<T> &b, Op op) {
  int_vector buffer;
  // 根据不同的 CPU 支持情况选择加载函数和数据类型
#if defined(CPU_CAPABILITY_AVX2)
  int_vector a_buffer = _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer = _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)b));
#elif defined(CPU_CAPABILITY_AVX512)
  int_vector a_buffer = _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer = _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)b));
#endif
  // 执行位运算操作并存储结果到 buffer 中
  buffer = op(a_buffer, b_buffer);
  // 定义结果向量的存储空间
  __at_align__ T results[Vectorized<T>::size()];

  // 根据不同的 CPU 支持情况选择存储函数
#if defined(CPU_CAPABILITY_AVX2)
  _mm256_store_si256(reinterpret_cast<int_vector*>(results), buffer);
#elif defined(CPU_CAPABILITY_AVX512)
  _mm512_store_si512(reinterpret_cast<int_vector*>(results), buffer);
#endif
  // 返回结果向量
  return Vectorized<T>::loadu(results);
}

// 位与运算符重载函数：对向量 a 和 b 执行位与操作
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 使用 lambda 表达式封装位与操作，以保证其始终内联
#if defined(CPU_CAPABILITY_AVX2)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm256_and_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm512_and_si512(a, b); });
#endif
}

// 位或运算符重载函数：对向量 a 和 b 执行位或操作
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 使用 lambda 表达式封装位或操作，以保证其始终内联
#if defined(CPU_CAPABILITY_AVX2)
// 如果支持 AVX2 指令集，则执行按位或操作，并返回结果
return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm256_or_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
// 如果支持 AVX512 指令集，则执行按位或操作，并返回结果
return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm512_or_si512(a, b); });
#endif
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
// 对于不继承 Vectorizedi 的 Vectorized<T> 类型，定义按位异或运算符
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 使用 lambda 包装 _mm512_xor_si512 或 _mm256_xor_si256，以确保在编译时进行内联优化
#if defined(CPU_CAPABILITY_AVX2)
// 如果支持 AVX2 指令集，则执行按位异或操作，并返回结果
return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm256_xor_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
// 如果支持 AVX512 指令集，则执行按位异或操作，并返回结果
return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm512_xor_si512(a, b); });
#endif
}

#else

template <typename T>
// 从字符数组加载数据到类型 T，然后返回加载的结果
auto load(char const* data) -> T {
  T ret;
  std::memcpy(&ret, data, sizeof(ret));
  return ret;
}

template<class T, typename Op>
// 执行按位二进制操作，使用给定的操作符 Op
static inline Vectorized<T> bitwise_binary_op(const Vectorized<T> &a, const Vectorized<T> &b, Op op) {
  static constexpr uint32_t element_no = VECTOR_WIDTH / sizeof(intmax_t);
  __at_align__ intmax_t buffer[element_no];
  static_assert(VECTOR_WIDTH % sizeof(intmax_t) == 0, "VECTOR_WIDTH not a multiple of sizeof(intmax_t)");
  static_assert(sizeof(buffer) == sizeof(Vectorized<T>), "sizeof(buffer) must match sizeof(Vectorized<T>)");
  // 为了遵守严格别名规则，应使用 memcpy
  // 参考：https://github.com/pytorch/pytorch/issues/66119
  // 根据 C11 标准第 6.5 节第 7 段（http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf），使用 char* 是合法的
  const auto* a_data = a.as_bytes();
  const auto* b_data = b.as_bytes();
  // 加载每个 intmax_t 块并处理；通过增加 sizeof(intmax_t) 来增加指针位置
  for (auto& out : buffer) {
    out = op(load<intmax_t>(a_data), load<intmax_t>(b_data));
    a_data += sizeof(intmax_t);
    b_data += sizeof(intmax_t);
  }
  assert(a_data == a.as_bytes() + sizeof(a));
  assert(b_data == b.as_bytes() + sizeof(b));
  // 加载未对齐的 buffer 到 Vectorized<T> 类型并返回
  return Vectorized<T>::loadu(buffer);
}

template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
// 对于不继承 Vectorizedi 的 Vectorized<T> 类型，定义按位与运算符
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 执行按位与操作，并返回结果
  return bitwise_binary_op(a, b, std::bit_and<intmax_t>());
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
// 对于不继承 Vectorizedi 的 Vectorized<T> 类型，定义按位或运算符
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 执行按位或操作，并返回结果
  return bitwise_binary_op(a, b, std::bit_or<intmax_t>());
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
// 对于不继承 Vectorizedi 的 Vectorized<T> 类型，定义按位异或运算符
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 执行按位异或操作，并返回结果
  return bitwise_binary_op(a, b, std::bit_xor<intmax_t>());
}

#endif // defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
// 对于非向量化类型的操作符~，返回 a 按位取反后与全1向量异或的结果
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  // 使用当前类型 T 的最大整数值创建一个全1向量
  using int_t = int_same_size_t<T>;
  Vectorized<T> ones(c10::bit_cast<T>((int_t)(~(int_t)0)));  // All bits are 1
  return a ^ ones;  // 返回 a 与全1向量异或的结果
}

// 向左移位操作符<<的重载实现
template <class T> Vectorized<T> inline operator<<(const Vectorized<T> &a, const Vectorized<T> &b) {
  // 计算最大可移位数
  constexpr T max_shift = sizeof(T) * CHAR_BIT;
  Vectorized<T> c;
  // 遍历向量中的每个元素
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    // 如果移位值为负数或超过最大位数，则将结果置为0
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
      c[i] = 0;
    } else {
      // 否则进行左移位操作
      c[i] = static_cast<std::make_unsigned_t<T>>(a[i]) << shift;
    }
  }
  return c;
}

// 向右移位操作符>>的重载实现
template <class T> Vectorized<T> inline operator>>(const Vectorized<T> &a, const Vectorized<T> &b) {
  // 计算最大可移位数，考虑是否有符号类型
  constexpr T max_shift = sizeof(T) * CHAR_BIT - std::is_signed_v<T>;
  Vectorized<T> c;
  // 遍历向量中的每个元素
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    // 如果移位值为负数或超过最大位数，则将结果调整为符号保留的右移位值
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
      c[i] = a[i] >> max_shift;
    } else {
      // 否则进行右移位操作
      c[i] = a[i] >> shift;
    }
  }
  return c;
}

// 向量化类型的加法赋值操作符重载
template <typename T>
inline Vectorized<T>& operator += (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a + b;
  return a;
}

// 向量化类型的减法赋值操作符重载
template <typename T>
inline Vectorized<T>& operator -= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a - b;
  return a;
}

// 向量化类型的除法赋值操作符重载
template <typename T>
inline Vectorized<T>& operator /= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a / b;
  return a;
}

// 向量化类型的取模赋值操作符重载
template <typename T>
inline Vectorized<T>& operator %= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a % b;
  return a;
}

// 向量化类型的乘法赋值操作符重载
template <typename T>
inline Vectorized<T>& operator *= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a * b;
  return a;
}

// 向量化类型的左移位赋值操作符重载
template <typename T>
inline Vectorized<T>& operator <<= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a << b;
  return a;
}

// 向量化类型的右移位赋值操作符重载
template <typename T>
inline Vectorized<T>& operator >>= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a >> b;
  return a;
}

// 向量化类型的 Fused Multiply-Add 操作函数
template <typename T>
inline Vectorized<T> fmadd(const Vectorized<T>& a, const Vectorized<T>& b, const Vectorized<T>& c) {
  return a * b + c;
}

// 向量化类型的 Fused Multiply-Subtract 操作函数
template <typename T>
inline Vectorized<T> fmsub(const Vectorized<T>& a, const Vectorized<T>& b, const Vectorized<T>& c) {
  return a * b - c;
}

// 根据缩放值 scale 和索引向量 vindex，从基地址 base_addr 处收集数据并返回向量化结果
template <int64_t scale = 1, typename T = void>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>>
inline gather(T const* base_addr, const Vectorized<int_same_size_t<T>>& vindex) {
  static constexpr int size = Vectorized<T>::size();
  int_same_size_t<T> index_arr[size];
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (const auto i : c10::irange(size)) {
    # 将 base_addr 中的数据按照索引数组 index_arr 中的元素顺序复制到 buffer 中
    buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
    # 返回从 buffer 加载的一个向量化对象，加载时不需要对齐
    return Vectorized<T>::loadu(static_cast<void*>(buffer));
}

// 定义模板函数 mask_gather，实现从 base_addr 中根据 vindex 中的索引获取数据
template <int64_t scale = 1, typename T = void>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>>
inline mask_gather(const Vectorized<T>& src, T const* base_addr,
                   const Vectorized<int_same_size_t<T>>& vindex, Vectorized<T>& mask) {
  static constexpr int size = Vectorized<T>::size();
  T src_arr[size];  // 创建存储 src 向量化数据的数组
  int_same_size_t<T> mask_arr[size];  // 使用 int 类型数组存储掩码数据，以进行逻辑与运算
  int_same_size_t<T> index_arr[size];  // 存储索引的 int 类型数组
  src.store(static_cast<void*>(src_arr));  // 将 src 向量化数据存储到 src_arr 数组中
  mask.store(static_cast<void*>(mask_arr));  // 将 mask 向量化数据存储到 mask_arr 数组中
  vindex.store(static_cast<void*>(index_arr));  // 将 vindex 向量化数据存储到 index_arr 数组中
  T buffer[size];  // 创建缓冲区数组用于存储结果数据
  for (const auto i : c10::irange(size)) {
    if (mask_arr[i] & 0x01) {  // 检查掩码数组的最高位
      buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];  // 根据索引从 base_addr 中获取数据到 buffer 中
    } else {
      buffer[i] = src_arr[i];  // 否则从 src_arr 中获取数据到 buffer 中
    }
  }
  mask = Vectorized<T>();  // 将 mask 向量化数据清零
  return Vectorized<T>::loadu(static_cast<void*>(buffer));  // 返回从 buffer 中加载的向量化数据
}

// 将给定向量 src 转换为目标类型 dst_t 的向量，不改变位表示
template<typename dst_t, typename src_t>
struct CastImpl {
  static inline Vectorized<dst_t> apply(const Vectorized<src_t>& src) {
    src_t src_arr[Vectorized<src_t>::size()];  // 创建存储 src_t 向量化数据的数组
    src.store(static_cast<void*>(src_arr));  // 将 src 向量化数据存储到 src_arr 数组中
    return Vectorized<dst_t>::loadu(static_cast<const void*>(src_arr));  // 返回加载自 src_arr 的 dst_t 向量化数据
  }
};

// 特化 CastImpl 结构体，当源类型与目标类型相同时直接返回源向量
template<typename scalar_t>
struct CastImpl<scalar_t, scalar_t> {
  static inline Vectorized<scalar_t> apply(const Vectorized<scalar_t>& src) {
    return src;  // 直接返回源向量
  }
};

// 实现 cast 函数模板，用于将源类型 src_t 的向量转换为目标类型 dst_t 的向量
template<typename dst_t, typename src_t>
inline Vectorized<dst_t> cast(const Vectorized<src_t>& src) {
  return CastImpl<dst_t, src_t>::apply(src);  // 调用 CastImpl 来执行类型转换并返回结果
}

// 将源类型 T 的向量 src 转换为相同大小的 IntType 类型的向量
template <typename T, typename IntType = int_same_size_t<T>>
inline Vectorized<IntType> convert_to_int_of_same_size(const Vectorized<T>& src) {
  static_assert(sizeof(T) == sizeof(IntType));  // 断言 T 和 IntType 的大小相同
  static constexpr int size = Vectorized<T>::size();  // 获取向量大小

  std::array<T, size> src_arr;  // 创建存储 src 向量化数据的数组
  src.store(static_cast<void*>(src_arr.data()));  // 将 src 向量化数据存储到 src_arr 中
  std::array<IntType, size> buffer;  // 创建缓冲区数组用于存储转换后的数据
  std::transform(src_arr.cbegin(), src_arr.cend(), buffer.begin(),
                 [](const T& x) { return static_cast<IntType>(x); });  // 执行类型转换并存储到 buffer 中
  return Vectorized<IntType>::loadu(static_cast<const void*>(buffer.data()));  // 返回加载自 buffer 的向量化数据
}

template <typename T, typename IntType = int_same_size_t<T>>
// 将输入向量的整数类型转换为相同大小的浮点类型向量
template <typename T>
inline Vectorized<T> convert_to_fp_of_same_size(const Vectorized<IntType>& src) {
  // 断言确保 T 和 IntType 的大小相同
  static_assert(sizeof(T) == sizeof(IntType));
  // 获取向量化类型 T 的大小
  static constexpr int size = Vectorized<T>::size();

  // 将输入向量存储到整数数组中
  std::array<IntType, size> src_arr;
  src.store(static_cast<void*>(src_arr.data()));

  // 创建一个浮点类型的缓冲区，并进行类型转换
  std::array<T, size> buffer;
  std::transform(src_arr.cbegin(), src_arr.cend(), buffer.begin(),
                 [](const IntType& x) { return static_cast<T>(x); });

  // 将转换后的数据加载到向量化类型 T 中并返回
  return Vectorized<T>::loadu(static_cast<const void*>(buffer.data()));
}

// 将两个输入向量的元素交错重新排列成两个向量
template <typename T>
inline std::enable_if_t<Vectorized<T>::size() % 2 == 0, std::pair<Vectorized<T>, Vectorized<T>>>
deinterleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 获取向量化类型 T 的大小和一半大小
  static constexpr int size = Vectorized<T>::size();
  static constexpr int half_size = size / 2;

  // 声明数组来存储向量 a 和 b 的元素
  T a_arr[size];
  T b_arr[size];

  // 将向量 a 和 b 的元素存储到数组中
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));

  // 声明两个缓冲区来存储交错后的元素
  T buffer1[size];
  T buffer2[size];

  // 对每对元素进行交错重新排列
  for (const auto i : c10::irange(half_size)) {
    buffer1[i] = a_arr[i * 2];
    buffer1[half_size + i] = b_arr[i * 2];
    buffer2[i] = a_arr[i * 2 + 1];
    buffer2[half_size + i] = b_arr[i * 2 + 1];
  }

  // 返回包含重新排列后数据的向量对
  return std::make_pair(Vectorized<T>::loadu(static_cast<void*>(buffer1)),
                        Vectorized<T>::loadu(static_cast<void*>(buffer2)));
}
//                           Vectorized<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
template <typename T>
// 如果 Vectorized<T> 的大小是偶数，则定义一个函数 interleave2，接受两个 Vectorized<T> 类型的参数，并返回一个 std::pair<Vectorized<T>, Vectorized<T>> 对象
inline std::enable_if_t<Vectorized<T>::size() % 2 == 0, std::pair<Vectorized<T>, Vectorized<T>>>
interleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 静态声明 size 和 half_size 常量，分别表示 Vectorized<T> 的大小和一半大小
  static constexpr int size = Vectorized<T>::size();
  static constexpr int half_size = size / 2;
  // 声明两个数组 a_arr 和 b_arr 用于存储向量 a 和 b 的元素
  T a_arr[size];
  T b_arr[size];
  // 声明两个缓冲区 buffer1 和 buffer2
  T buffer1[size];
  T buffer2[size];
  // 将向量 a 和 b 的数据存储到 a_arr 和 b_arr 数组中
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));
  // 使用 c10::irange(half_size) 迭代器遍历数组的一半大小
  for (const auto i : c10::irange(half_size)) {
    // 对 buffer1 进行交错操作，将 a_arr 和 b_arr 中的元素交错存放
    buffer1[i * 2] = a_arr[i];
    buffer1[i * 2 + 1] = b_arr[i];
    // 对 buffer2 进行交错操作，将 a_arr 和 b_arr 后半部分的元素交错存放
    buffer2[i * 2] = a_arr[half_size + i];
    buffer2[i * 2 + 1] = b_arr[half_size + i];
  }
  // 返回一个 std::pair 对象，包含两个 Vectorized<T> 向量，分别从 buffer1 和 buffer2 加载数据而来
  return std::make_pair(Vectorized<T>::loadu(static_cast<void*>(buffer1)),
                        Vectorized<T>::loadu(static_cast<void*>(buffer2)));
}

template <typename src_T, typename dst_T>
// 定义一个 convert 函数模板，用于将 src_T 类型的数组转换为 dst_T 类型的数组，长度为 n
inline void convert(const src_T *src, dst_T *dst, int64_t n) {
#ifndef _MSC_VER
# pragma unroll
#endif
  // 使用 c10::irange(n) 迭代器遍历长度为 n 的数组
  for (C10_UNUSED const auto i : c10::irange(n)) {
    // 将 src 数组的元素加载并转换为 dst_T 类型，存储到 dst 数组中
    *dst = c10::convert<dst_T>(c10::load(src));
    // 移动到 src 和 dst 数组的下一个元素
    src++;
    dst++;
  }
}

template <typename T>
// 定义一个 flip 函数模板，用于反转 Vectorized<T> 类型的向量
inline Vectorized<T> flip(const Vectorized<T> & data) {
  // 静态声明 size 常量，表示 Vectorized<T> 的大小
  static constexpr int size = Vectorized<T>::size();
  // 声明 output 和 buffer 数组
  T output[size];
  T buffer[size];
  // 将 data 向量的数据存储到 buffer 数组中
  data.store(static_cast<void*>(buffer));
  // 使用 c10::irange(size) 迭代器遍历向量的大小
  for (const auto i : c10::irange(size)) {
    // 反转向量中的元素，存储到 output 数组中
    output[i] = buffer[size - i - 1];
  }
  // 返回一个新的 Vectorized<T> 向量，加载自 output 数组的数据
  return Vectorized<T>::loadu(static_cast<void*>(output));
}

// Transpose the `src` buffer of type `T` and size (M,N) into the `dst` buffer. `ld_src` is the leading
// dimension of `src` and `ld_dst` is the leading dimension of `dst`.
// 将类型为 T、大小为 (M,N) 的 src 缓冲区转置到 dst 缓冲区中。ld_src 是 src 的主导维度，ld_dst 是 dst 的主导维度。
template <typename T, int M, int N>
inline void transpose_mxn(const T* src, int64_t ld_src, T* dst, int64_t ld_dst) {
  // 使用两层循环分别遍历 M 和 N
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      // 将 src 中的数据按照转置规则写入到 dst 中
      dst[j*ld_dst + i] = src[i*ld_src + j];
    }
  }
}

}} // namespace at::vec::CPU_CAPABILITY

// additional headers for more operations that depend on vec_base
#include <ATen/cpu/vec/vec_n.h>
#include <ATen/cpu/vec/vec_mask.h>
#include <ATen/cpu/vec/vec_convert.h>
```