# `.\pytorch\c10\util\complex_math.cpp`

```py
#include <c10/util/complex.h>

#include <cmath>

// Note [ Complex Square root in libc++]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In libc++ complex square root is computed using polar form
// This is a reasonably fast algorithm, but can result in significant
// numerical errors when arg is close to 0, pi/2, pi, or 3pi/4
// In that case provide a more conservative implementation which is
// slower but less prone to those kinds of errors
// In libstdc++ complex square root yield invalid results
// for -x-0.0j unless C99 csqrt/csqrtf fallbacks are used

#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))

namespace {
// Function template to compute complex square root using specific rules
template <typename T>
c10::complex<T> compute_csqrt(const c10::complex<T>& z) {
  constexpr auto half = T(.5);

  // Trust standard library to correctly handle infs and NaNs
  // Return square root of the complex number 'z'
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<c10::complex<T>>(
        std::sqrt(static_cast<std::complex<T>>(z)));
  }

  // Special case for square root of pure imaginary values
  // Calculate square root of a complex number with a real part of 0
  if (z.real() == T(0)) {
    if (z.imag() == T(0)) {
      return c10::complex<T>(T(0), z.imag());
    }
    auto v = std::sqrt(half * std::abs(z.imag()));
    return c10::complex<T>(v, std::copysign(v, z.imag()));
  }

  // At this point, z is non-zero and finite
  // Compute the square root using the standard algorithm
  if (z.real() >= 0.0) {
    auto t = std::sqrt((z.real() + std::abs(z)) * half);
    return c10::complex<T>(t, half * (z.imag() / t));
  }

  auto t = std::sqrt((-z.real() + std::abs(z)) * half);
  return c10::complex<T>(
      half * std::abs(z.imag() / t), std::copysign(t, z.imag()));
}

// Compute complex arccosine using formula from W. Kahan
// "Branch Cuts for Complex Elementary Functions" 1986 paper:
// cacos(z).re = 2*atan2(sqrt(1-z).re(), sqrt(1+z).re())
// cacos(z).im = asinh((sqrt(conj(1+z))*sqrt(1-z)).im())
template <typename T>
c10::complex<T> compute_cacos(const c10::complex<T>& z) {
  auto constexpr one = T(1);
  // Trust standard library to correctly handle infs and NaNs
  // Return the arccosine of the complex number 'z'
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<c10::complex<T>>(
        std::acos(static_cast<std::complex<T>>(z)));
  }
  auto a = compute_csqrt(c10::complex<T>(one - z.real(), -z.imag()));
  auto b = compute_csqrt(c10::complex<T>(one + z.real(), z.imag()));
  auto c = compute_csqrt(c10::complex<T>(one + z.real(), -z.imag()));
  auto r = T(2) * std::atan2(a.real(), b.real());
  // Explicitly unroll (a*c).imag()
  auto i = std::asinh(a.real() * c.imag() + a.imag() * c.real());
  return c10::complex<T>(r, i);
}
} // anonymous namespace

namespace c10_complex_math {
namespace _detail {
// Specialization of sqrt function for complex numbers of type float
c10::complex<float> sqrt(const c10::complex<float>& in) {
  return compute_csqrt(in);
}

// Specialization of sqrt function for complex numbers of type double
c10::complex<double> sqrt(const c10::complex<double>& in) {
  return compute_csqrt(in);
}
# 定义一个函数，计算给定复数的反余弦值，返回结果为单精度复数类型
c10::complex<float> acos(const c10::complex<float>& in) {
    return compute_cacos(in);
}

# 定义一个函数，计算给定复数的反余弦值，返回结果为双精度复数类型
c10::complex<double> acos(const c10::complex<double>& in) {
    return compute_cacos(in);
}

# 结束命名空间 _detail
} // namespace _detail

# 结束命名空间 c10_complex_math
} // namespace c10_complex_math

# 如果没有定义过 endif，那么这是一个条件编译的结束标记
#endif
```