# `.\pytorch\c10\util\complex.h`

```py
#pragma once
// 使用预处理指令#pragma once确保头文件只被包含一次

#include <complex>
// 包含C++标准库中的复数头文件

#include <c10/macros/Macros.h>
// 包含c10库的宏定义文件

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif
// 如果编译器是CUDA或HIP，则包含thrust库中的复数头文件

C10_CLANG_DIAGNOSTIC_PUSH()
// 将当前Clang编译器的诊断设置推入诊断栈

#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
// 如果Clang支持"-Wimplicit-float-conversion"警告，则忽略该警告

#if C10_CLANG_HAS_WARNING("-Wfloat-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wfloat-conversion")
#endif
// 如果Clang支持"-Wfloat-conversion"警告，则忽略该警告

namespace c10 {
// 命名空间c10开始

// c10::complex是一个复数的实现，旨在在PyTorch支持的所有设备上工作
//
// 大多数API与std::complex重复
// 参考链接：https://en.cppreference.com/w/cpp/numeric/complex
//
// [NOTE: 复数运算符统一]
// 运算符当前在内部使用std::complex、thrust::complex和c10::complex的混合。
// 最终状态是所有运算符将在内部使用c10::complex。在此之前，可能会有一些支持所有变体的黑科技。
//
//
// [构造函数说明]
//
// 构造函数的API大多数来自C++标准：
//   https://en.cppreference.com/w/cpp/numeric/complex/complex
//
// 自C++14以来，所有构造函数在std::complex中都是constexpr的
//
// 有三种类型的构造函数：
// - 从实部和虚部初始化：
//     `constexpr complex( const T& re = T(), const T& im = T() );`
// - 隐式声明的复制构造函数
// - 转换构造函数
//
// 转换构造函数：
// - std::complex定义了float/double/long double之间的转换构造函数，
//   而我们定义了float/double之间的转换构造函数。
// - 对于这些转换构造函数，向上转型是隐式的，向下转型是显式的。
// - 我们还定义了从std::complex/thrust::complex到c10::complex的显式转换
//   - 注意，从thrust到c10的转换不是constexpr的，因为thrust没有将它们定义为constexpr ????
//
//
// [赋值运算符 =]
//
// 赋值运算符的API大多数来自C++标准：
//   https://en.cppreference.com/w/cpp/numeric/complex/operator%3D
//
// 自C++20以来，所有operator=都是constexpr的。尽管我们没有使用C++20构建，但我们也遵循这种行为。
//
// 有三种类型的赋值运算符：
// - 从相同的标量类型分配一个实数值
//   - 在std中，这被模板化为`complex& operator=(const T& x)`
//     对于float/double/long double，使用特化的`complex& operator=(T x)`
//     由于我们只支持float和double，因此将使用`complex& operator=(T x)`
// - 复制赋值运算符和转换赋值运算符
//   - 没有转换赋值运算符的特化，可转换类型完全取决于标量类型是否可转换
//
// 除了标准赋值之外，我们还提供了与std和thrust的赋值运算符
//
//
// [类型转换运算符]
//
// std::complex没有类型转换运算符。我们定义了到std::complex和thrust::complex的类型转换运算符
//
//
// [运算符""]
//
```cpp`
// std::complex has custom literals `i`, `if` and `il` defined in namespace
// `std::literals::complex_literals`. We define our own custom literals in the
// namespace `c10::complex_literals`. Our custom literals do not follow the
// same behavior as in std::complex; instead, we define _if, _id to construct
// float/double complex literals.

// [real() and imag()]
//
// In C++20, there are two overloads of these functions: one to return
// the real/imaginary part, and another to set the real/imaginary part.
// They are both constexpr. We follow this design.

// [Operator +=, -=, *=, /=]
//
// Since C++20, these operators have become constexpr. In our implementation,
// they are also constexpr.

// There are two types of such operators: operating with a real number, or
// operating with another complex number. For operations with a real number,
// the generic template form has the argument type `const T &`, while the overload
// for float/double/long double uses `T`. We will follow the same type as
// float/double/long double in std::complex.

// [Unary operators +, -]
//
// Since C++20, these operators are constexpr. We also make them constexpr.

// [Binary operators +, -, *, /]
//
// Each operator has three versions (taking + as an example):
// - complex + complex
// - complex + real
// - real + complex

// [Operators ==, !=]
//
// Each operator has three versions (taking == as an example):
// - complex == complex
// - complex == real
// - real == complex

// Some of these operators are removed in C++20, but we decide to keep them.

// [Operators <<, >>]
//
// These are implemented by casting to std::complex.

// TODO(@zasdfgbnm): c10::complex<c10::Half> is not currently supported,
// because:
// - lots of members and functions of c10::Half are not constexpr
// - thrust::complex only supports float and double

template <typename T>
struct alignas(sizeof(T) * 2) complex {
  using value_type = T;

  T real_ = T(0);
  T imag_ = T(0);

  // Default constructor
  constexpr complex() = default;

  // Constructor with real and imaginary parts
  C10_HOST_DEVICE constexpr complex(const T& re, const T& im = T())
      : real_(re), imag_(im) {}

  // Constructor from std::complex<U>
  template <typename U>
  explicit constexpr complex(const std::complex<U>& other)
      : complex(other.real(), other.imag()) {}

  // Constructor from thrust::complex<U>
#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  explicit C10_HOST_DEVICE complex(const thrust::complex<U>& other)
      : real_(other.real()), imag_(other.imag()) {}
// NOTE: The alternative implementation using initializer list is disabled
// due to a ROCm bug:
//   explicit C10_HOST_DEVICE complex(const thrust::complex<U> &other):
//   complex(other.real(), other.imag()) {}
#endif

  // 使用 SFINAE（替代模板推断失败时的备用方案）为 c10::complex<float> 和 c10::complex<double> 分别特化转换构造函数
  template <typename U = T>
  C10_HOST_DEVICE explicit constexpr complex(
      const std::enable_if_t<std::is_same_v<U, float>, complex<double>>& other)
      : real_(other.real_), imag_(other.imag_) {}
  template <typename U = T>
  C10_HOST_DEVICE constexpr complex(
      const std::enable_if_t<std::is_same_v<U, double>, complex<float>>& other)
      : real_(other.real_), imag_(other.imag_) {}

  // 将实部设置为给定值，虚部保持为零
  constexpr complex<T>& operator=(T re) {
    real_ = re;
    imag_ = 0;
    return *this;
  }

  // 实部加上给定值，虚部不变
  constexpr complex<T>& operator+=(T re) {
    real_ += re;
    return *this;
  }

  // 实部减去给定值，虚部不变
  constexpr complex<T>& operator-=(T re) {
    real_ -= re;
    return *this;
  }

  // 实部和虚部分别乘以给定值
  constexpr complex<T>& operator*=(T re) {
    real_ *= re;
    imag_ *= re;
    return *this;
  }

  // 实部和虚部分别除以给定值
  constexpr complex<T>& operator/=(T re) {
    real_ /= re;
    imag_ /= re;
    return *this;
  }

  // 从另一个复数对象复制实部和虚部
  template <typename U>
  constexpr complex<T>& operator=(const complex<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

  // 将另一个复数对象的实部和虚部分别加到当前对象的实部和虚部上
  template <typename U>
  constexpr complex<T>& operator+=(const complex<U>& rhs) {
    real_ += rhs.real();
    imag_ += rhs.imag();
    return *this;
  }

  // 将另一个复数对象的实部和虚部分别从当前对象的实部和虚部上减去
  template <typename U>
  constexpr complex<T>& operator-=(const complex<U>& rhs) {
    real_ -= rhs.real();
    imag_ -= rhs.imag();
    return *this;
  }

  // 将另一个复数对象乘到当前对象上，按照复数乘法的规则计算结果
  template <typename U>
  constexpr complex<T>& operator*=(const complex<U>& rhs) {
    // (a + bi) * (c + di) = (a*c - b*d) + (a * d + b * c) i
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }

#ifdef __APPLE__
#define FORCE_INLINE_APPLE __attribute__((always_inline))
#else
#define FORCE_INLINE_APPLE
#endif

  // 将另一个复数对象除到当前对象上，按照复数除法的规则计算结果
  template <typename U>
  constexpr FORCE_INLINE_APPLE complex<T>& operator/=(const complex<U>& rhs)
      __ubsan_ignore_float_divide_by_zero__ {
    // (a + bi) / (c + di) = (ac + bd)/(c^2 + d^2) + (bc - ad)/(c^2 + d^2) i
    // 下面的计算遵循 numpy 的复数除法规则
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();

    // 根据分母的绝对值大小选择计算方式，以避免除零错误
#if defined(__GNUC__) && !defined(__clang__)
    // std::abs 在 gcc 中已经是 constexpr
    auto abs_c = std::abs(c);
    auto abs_d = std::abs(d);
#else
    auto abs_c = c < 0 ? -c : c;
    auto abs_d = d < 0 ? -d : d;
#endif

    if (abs_c >= abs_d) {
      if (abs_c == 0 && abs_d == 0) {
        /* 除以零应该产生复数无穷大或非数字 */
        real_ = a / abs_c;
        imag_ = b / abs_d;
      } else {
        auto rat = d / c;
        auto scl = 1.0 / (c + d * rat);
        real_ = (a + b * rat) * scl;
        imag_ = (b - a * rat) * scl;
      }
    } else {
      auto rat = c / d;
      auto scl = 1.0 / (d + c * rat);
      real_ = (a * rat + b) * scl;
      imag_ = (b * rat - a) * scl;
    }
    return *this;
  }
#undef FORCE_INLINE_APPLE

  // 赋值操作符重载，从 std::complex<U> 转换为当前复数类实例
  template <typename U>
  constexpr complex<T>& operator=(const std::complex<U>& rhs) {
    // 设置实部为 rhs 的实部
    real_ = rhs.real();
    // 设置虚部为 rhs 的虚部
    imag_ = rhs.imag();
    // 返回当前复数类实例的引用
    return *this;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 在 CUDA 或 HIP 环境下，赋值操作符重载，从 thrust::complex<U> 转换为当前复数类实例
  template <typename U>
  C10_HOST_DEVICE complex<T>& operator=(const thrust::complex<U>& rhs) {
    // 设置实部为 rhs 的实部
    real_ = rhs.real();
    // 设置虚部为 rhs 的虚部
    imag_ = rhs.imag();
    // 返回当前复数类实例的引用
    return *this;
  }
#endif

  // 显式类型转换操作符重载，将当前复数类实例转换为 std::complex<U>
  template <typename U>
  explicit constexpr operator std::complex<U>() const {
    // 返回转换后的 std::complex<U> 实例
    return std::complex<U>(std::complex<T>(real(), imag()));
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 在 CUDA 或 HIP 环境下，显式类型转换操作符重载，将当前复数类实例转换为 thrust::complex<U>
  template <typename U>
  C10_HOST_DEVICE explicit operator thrust::complex<U>() const {
    // 返回转换后的 thrust::complex<U> 实例
    return static_cast<thrust::complex<U>>(thrust::complex<T>(real(), imag()));
  }
#endif

  // 显式类型转换操作符重载，将当前复数类实例转换为 bool 类型，与 NumPy 行为一致
  // 返回实部或虚部是否非零的结果
  explicit constexpr operator bool() const {
    return real() || imag();
  }

  // 返回实部的值
  C10_HOST_DEVICE constexpr T real() const {
    return real_;
  }
  // 设置实部的值
  constexpr void real(T value) {
    real_ = value;
  }
  // 返回虚部的值
  C10_HOST_DEVICE constexpr T imag() const {
    return imag_;
  }
  // 设置虚部的值
  constexpr void imag(T value) {
    imag_ = value;
  }
};

// 复数字面量的命名空间
namespace complex_literals {

// 实现浮点数虚部复数字面量的运算符重载
constexpr complex<float> operator""_if(long double imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

// 实现双精度浮点数虚部复数字面量的运算符重载
constexpr complex<double> operator""_id(long double imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

// 实现浮点数虚部复数字面量的运算符重载
constexpr complex<float> operator""_if(unsigned long long imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

// 实现双精度浮点数虚部复数字面量的运算符重载
constexpr complex<double> operator""_id(unsigned long long imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

} // namespace complex_literals

// 复数加法运算符重载，返回复数加法的结果
template <typename T>
constexpr complex<T> operator+(const complex<T>& val) {
  return val;
}

// 复数取负运算符重载，返回复数取负的结果
template <typename T>
constexpr complex<T> operator-(const complex<T>& val) {
  return complex<T>(-val.real(), -val.imag());
}

// 复数加法运算符重载，返回复数加法的结果
template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result += rhs;
}

// 复数加法运算符重载，返回复数加法的结果
template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result += rhs;
}

// 复数加法运算符重载，返回复数加法的结果
template <typename T>
constexpr complex<T> operator+(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs + rhs.real(), rhs.imag());
}

// 复数减法运算符重载，返回复数减法的结果
template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result -= rhs;
}

// 复数减法运算符重载，返回复数减法的结果
template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result -= rhs;
}

// 复数减法运算符重载，返回复数减法的结果
template <typename T>
constexpr complex<T> operator-(const T& lhs, const complex<T>& rhs) {
  complex<T> result = -rhs;
  return result += lhs;
}

// 复数乘法运算符重载，返回复数乘法的结果
template <typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result *= rhs;
}

// 模板结束
constexpr complex<T> operator*(const complex<T>& lhs, const T& rhs) {
    // 创建一个新的复数对象，并将其初始化为左操作数的副本
    complex<T> result = lhs;
    // 使用复合赋值运算符 *= 来将右操作数乘到结果中
    return result *= rhs;
}

template <typename T>
constexpr complex<T> operator*(const T& lhs, const complex<T>& rhs) {
    // 创建一个新的复数对象，并将其初始化为右操作数的副本
    complex<T> result = rhs;
    // 使用复合赋值运算符 *= 将左操作数乘到结果中
    return result *= lhs;
}

template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
    // 创建一个新的复数对象，并将其初始化为左操作数的副本
    complex<T> result = lhs;
    // 使用复合赋值运算符 /= 将右操作数除到结果中
    return result /= rhs;
}

template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const T& rhs) {
    // 创建一个新的复数对象，并将其初始化为左操作数的副本
    complex<T> result = lhs;
    // 使用复合赋值运算符 /= 将右操作数除到结果中
    return result /= rhs;
}

template <typename T>
constexpr complex<T> operator/(const T& lhs, const complex<T>& rhs) {
    // 创建一个新的复数对象，并将其初始化为左操作数和默认构造的虚部
    complex<T> result(lhs, T());
    // 使用复合赋值运算符 /= 将右操作数除到结果中
    return result /= rhs;
}

// 定义整数标量与 c10::complex 之间的运算符。当 T 是浮点数时，std::complex 不支持这样做。
// 这样做很有用，因为在操作复数和整数时，可以减少大量的 "static_cast"。
// 这使得代码更简洁，潜在上也更高效。
#define COMPLEX_INTEGER_OP_TEMPLATE_CONDITION                 \
  typename std::enable_if_t<                                  \
      std::is_floating_point_v<fT> && std::is_integral_v<iT>, \
      int> = 0

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator+(const c10::complex<fT>& a, const iT& b) {
    // 返回复数 a 与整数 b 相加的结果，整数 b 被转换为浮点数再相加
    return a + static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator+(const iT& a, const c10::complex<fT>& b) {
    // 返回整数 a 与复数 b 相加的结果，整数 a 被转换为浮点数再相加
    return static_cast<fT>(a) + b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator-(const c10::complex<fT>& a, const iT& b) {
    // 返回复数 a 减去整数 b 的结果，整数 b 被转换为浮点数再相减
    return a - static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator-(const iT& a, const c10::complex<fT>& b) {
    // 返回整数 a 减去复数 b 的结果，整数 a 被转换为浮点数再相减
    return static_cast<fT>(a) - b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator*(const c10::complex<fT>& a, const iT& b) {
    // 返回复数 a 乘以整数 b 的结果，整数 b 被转换为浮点数再相乘
    return a * static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator*(const iT& a, const c10::complex<fT>& b) {
    // 返回整数 a 乘以复数 b 的结果，整数 a 被转换为浮点数再相乘
    return static_cast<fT>(a) * b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator/(const c10::complex<fT>& a, const iT& b) {
    // 返回复数 a 除以整数 b 的结果，整数 b 被转换为浮点数再相除
    return a / static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator/(const iT& a, const c10::complex<fT>& b) {
    // 返回整数 a 除以复数 b 的结果，整数 a 被转换为浮点数再相除
    return static_cast<fT>(a) / b;
}

#undef COMPLEX_INTEGER_OP_TEMPLATE_CONDITION

template <typename T>
constexpr bool operator==(const complex<T>& lhs, const complex<T>& rhs) {
    // 检查两个复数对象的实部和虚部是否相等
    return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}
constexpr bool operator==(const complex<T>& lhs, const T& rhs) {
    // 比较复数对象 lhs 的实部是否等于 rhs，虚部是否为零，返回比较结果
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <typename T>
constexpr bool operator==(const T& lhs, const complex<T>& rhs) {
    // 比较 lhs 是否等于复数对象 rhs 的实部，以及虚部是否为零，返回比较结果
    return (lhs == rhs.real()) && (T() == rhs.imag());
}

template <typename T>
constexpr bool operator!=(const complex<T>& lhs, const complex<T>& rhs) {
    // 使用相等运算符的相反结果来定义不等运算符
    return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const complex<T>& lhs, const T& rhs) {
    // 使用相等运算符的相反结果来定义不等运算符
    return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const T& lhs, const complex<T>& rhs) {
    // 使用相等运算符的相反结果来定义不等运算符
    return !(lhs == rhs);
}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const complex<T>& x) {
    // 将复数对象 x 转换为 std::complex<T> 类型，然后输出到流 os
    return (os << static_cast<std::complex<T>>(x));
}

template <typename T, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is,
    complex<T>& x) {
    // 从输入流 is 中读取数据到 std::complex<T> 类型的临时变量 tmp
    std::complex<T> tmp;
    is >> tmp;
    // 将临时变量 tmp 赋值给复数对象 x
    x = tmp;
    // 返回输入流 is
    return is;
}

} // namespace c10

// std functions
//
// 这些函数的实现遵循 C++20 的设计

namespace std {

template <typename T>
constexpr T real(const c10::complex<T>& z) {
    // 返回复数对象 z 的实部
    return z.real();
}

template <typename T>
constexpr T imag(const c10::complex<T>& z) {
    // 返回复数对象 z 的虚部
    return z.imag();
}

template <typename T>
C10_HOST_DEVICE T abs(const c10::complex<T>& z) {
    // 根据不同编译器环境，选择使用 thrust::abs 或 std::abs 计算复数对象 z 的绝对值
#if defined(__CUDACC__) || defined(__HIPCC__)
    return thrust::abs(static_cast<thrust::complex<T>>(z));
#else
    return std::abs(static_cast<std::complex<T>>(z));
#endif
}

#if defined(USE_ROCM)
#define ROCm_Bug(x)
#else
#define ROCm_Bug(x) x
#endif

template <typename T>
C10_HOST_DEVICE T arg(const c10::complex<T>& z) {
    // 根据不同编译器环境，选择使用 thrust::atan2 或 std::atan2 计算复数对象 z 的幅角
    return ROCm_Bug(std)::atan2(std::imag(z), std::real(z));
}

#undef ROCm_Bug

template <typename T>
constexpr T norm(const c10::complex<T>& z) {
    // 返回复数对象 z 的模的平方
    return z.real() * z.real() + z.imag() * z.imag();
}

// 对于 std::conj，还有其他版本，但它们没有在这里实现
// TODO(@zasdfgbnm): 作为 c10::conj 来实现它们

template <typename T>
constexpr c10::complex<T> conj(const c10::complex<T>& z) {
    // 返回复数对象 z 的共轭复数
    return c10::complex<T>(z.real(), -z.imag());
}

// Thrust 没有 complex 到 complex 版本的 thrust::proj 函数，因此在 c10 中未实现该函数
// TODO(@zasdfgbnm): 自行实现它

// 没有 c10 版本的 std::polar，因为 std::polar 总是返回 std::complex。使用 c10::polar 替代;

} // namespace std

namespace c10 {

template <typename T>
C10_HOST_DEVICE complex<T> polar(const T& r, const T& theta = T()) {
    // 根据不同编译器环境，选择使用 thrust::polar 或 std::polar 构造极坐标表示的复数对象
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<complex<T>>(thrust::polar(r, theta));

    return static_cast<complex<T>>(std::polar(r, theta));
#endif
}


这段代码定义了一些用于复数类 `complex` 的运算符重载和一些标准库函数。每个函数和运算符都有详细的注释解释其作用和实现细节。
#else
  // std::polar() requires r >= 0, so spell out the explicit implementation to
  // avoid a branch.
  // 如果 r >= 0，则 std::polar() 要求显式实现，以避免分支。
  return complex<T>(r * std::cos(theta), r * std::sin(theta));
#endif
}
// 返回一个由极坐标 r 和角度 theta 定义的复数

} // namespace c10
// 结束 c10 命名空间

C10_CLANG_DIAGNOSTIC_POP()
// 弹出 Clang 编译器诊断设置

#define C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// 定义 C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H 宏，用于包含复杂数剩余部分的头文件
#include <c10/util/complex_math.h> // IWYU pragma: keep
// 包含复杂数的数学函数，保持与 include-what-you-use (IWYU) 规则相符
#include <c10/util/complex_utils.h> // IWYU pragma: keep
// 包含复杂数类型的实用工具，保持与 IWYU 规则相符
#undef C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// 取消定义 C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H 宏
```