# `.\pytorch\c10\util\complex_math.h`

```
#if !defined(C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "c10/util/complex_math.h is not meant to be individually included. Include c10/util/complex.h instead."
#endif

#if 指令检查是否未定义宏 C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H，并且如果未定义则输出错误信息指示应包含其他头文件。


namespace c10_complex_math {

// 在 c10_complex_math 命名空间内定义以下函数。


template <typename T>
C10_HOST_DEVICE inline c10::complex<T> exp(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::exp(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::exp(static_cast<std::complex<T>>(x)));
#endif
}

// 计算复数 x 的指数函数值。如果编译环境为 CUDA 或者 HIP，使用 thrust 库的 exp 函数；否则使用标准库的 exp 函数。


template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::log(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::log(static_cast<std::complex<T>>(x)));
#endif
}

// 计算复数 x 的自然对数。如果编译环境为 CUDA 或者 HIP，使用 thrust 库的 log 函数；否则使用标准库的 log 函数。


template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log10(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::log10(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::log10(static_cast<std::complex<T>>(x)));
#endif
}

// 计算复数 x 的以 10 为底的对数。如果编译环境为 CUDA 或者 HIP，使用 thrust 库的 log10 函数；否则使用标准库的 log10 函数。


template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log2(const c10::complex<T>& x) {
  const c10::complex<T> log2 = c10::complex<T>(::log(2.0), 0.0);
  return c10_complex_math::log(x) / log2;
}

// 计算复数 x 的以 2 为底的对数。首先定义一个 log2 的复数常量，然后返回 log(x) 除以 log2 的结果。


#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))
namespace _detail {
C10_API c10::complex<float> sqrt(const c10::complex<float>& in);
C10_API c10::complex<double> sqrt(const c10::complex<double>& in);
C10_API c10::complex<float> acos(const c10::complex<float>& in);
C10_API c10::complex<double> acos(const c10::complex<double>& in);
} // namespace _detail
#endif

// 如果使用的是 libc++ 或者在使用 libstdc++ 时没有启用 C99 复数支持，则在 _detail 命名空间内声明 sqrt 和 acos 函数的特化版本。


template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sqrt(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::sqrt(static_cast<thrust::complex<T>>(x)));
#elif !(                        \
    defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX)))
  return static_cast<c10::complex<T>>(
      std::sqrt(static_cast<std::complex<T>>(x)));
#else
  return _detail::sqrt(x);
#endif
}

// 计算复数 x 的平方根。如果编译环境为 CUDA 或者 HIP，使用 thrust 库的 sqrt 函数；否则根据编译器和标准库版本选择相应的 sqrt 函数或 _detail 命名空间内的特化版本。


template <typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(
    const c10::complex<T>& x,
    const c10::complex<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(
      static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(
      static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

// 计算复数 x 的复数次幂 y。如果编译环境为 CUDA 或者 HIP，使用 thrust 库的 pow 函数；否则使用标准库的 pow 函数。

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(
    const c10::complex<T>& x,
    const T& y) {

// 计算复数 x 的实数次幂 y。
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算 x 的 y 次幂，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
// 否则，使用标准库计算 x 的 y 次幂，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(
    const T& x,
    const c10::complex<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算 x 的 y 次幂，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
// 否则，使用标准库计算 x 的 y 次幂，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(
    const c10::complex<T>& x,
    const c10::complex<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算 x 的 y 次幂，并转换为 c10::complex<decltype(T() * U())> 类型返回
return static_cast<c10::complex<decltype(T() * U())>>(thrust::pow(
    static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<U>>(y)));
#else
// 否则，使用标准库计算 x 的 y 次幂，并转换为 c10::complex<decltype(T() * U())> 类型返回
return static_cast<c10::complex<decltype(T() * U())>>(std::pow(
    static_cast<std::complex<T>>(x), static_cast<std::complex<U>>(y)));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(
    const c10::complex<T>& x,
    const U& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算 x 的 y 次幂，并转换为 c10::complex<decltype(T() * U())> 类型返回
return static_cast<c10::complex<T>>(
    thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
// 否则，使用标准库计算 x 的 y 次幂，并转换为 c10::complex<decltype(T() * U())> 类型返回
return static_cast<c10::complex<T>>(
    std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(
    const T& x,
    const c10::complex<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算 x 的 y 次幂，并转换为 c10::complex<decltype(T() * U())> 类型返回
return static_cast<c10::complex<T>>(
    thrust::pow(x, static_cast<thrust::complex<U>>(y)));
#else
// 否则，使用标准库计算 x 的 y 次幂，并转换为 c10::complex<decltype(T() * U())> 类型返回
return static_cast<c10::complex<T>>(
    std::pow(x, static_cast<std::complex<U>>(y)));
#endif
}

// 三角函数

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sin(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算复数 x 的正弦，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    thrust::sin(static_cast<thrust::complex<T>>(x)));
#else
// 否则，使用标准库计算复数 x 的正弦，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    std::sin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> cos(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算复数 x 的余弦，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    thrust::cos(static_cast<thrust::complex<T>>(x)));
#else
// 否则，使用标准库计算复数 x 的余弦，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    std::cos(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> tan(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则使用 thrust 库计算复数 x 的正切，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    thrust::tan(static_cast<thrust::complex<T>>(x)));
#else
// 否则，使用标准库计算复数 x 的正切，并转换为 c10::complex<T> 类型返回
return static_cast<c10::complex<T>>(
    std::tan(static_cast<std::complex<T>>(x)));
#endif
}
C10_HOST_DEVICE inline c10::complex<T> asin(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 asin 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::asin(static_cast<thrust::complex<T>>(x)));
#else
  // 否则，使用标准库的 asin 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::asin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> acos(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 acos 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::acos(static_cast<thrust::complex<T>>(x)));
#elif !defined(_LIBCPP_VERSION)
  // 如果不是以上环境，但未定义 _LIBCPP_VERSION，则使用标准库的 acos 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::acos(static_cast<std::complex<T>>(x)));
#else
  // 其他情况下，调用 _detail 命名空间下的 acos 函数处理复数 x
  return _detail::acos(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> atan(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 atan 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::atan(static_cast<thrust::complex<T>>(x)));
#else
  // 否则，使用标准库的 atan 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::atan(static_cast<std::complex<T>>(x)));
#endif
}

// Hyperbolic functions

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sinh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 sinh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::sinh(static_cast<thrust::complex<T>>(x)));
#else
  // 否则，使用标准库的 sinh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::sinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> cosh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 cosh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::cosh(static_cast<thrust::complex<T>>(x)));
#else
  // 否则，使用标准库的 cosh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::cosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> tanh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 tanh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::tanh(static_cast<thrust::complex<T>>(x)));
#else
  // 否则，使用标准库的 tanh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::tanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> asinh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 asinh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::asinh(static_cast<thrust::complex<T>>(x)));
#else
  // 否则，使用标准库的 asinh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::asinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> acosh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译环境为 CUDA 或 HIP，则使用 thrust 库的 acosh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      thrust::acosh(static_cast<thrust::complex<T>>(x)));
#else
  // 否则，使用标准库的 acosh 函数处理复数 x
  return static_cast<c10::complex<T>>(
      std::acosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> atanh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译环境是 CUDA 或者 HIP，则使用 thrust 库计算复数的反双曲正切
return static_cast<c10::complex<T>>(
    thrust::atanh(static_cast<thrust::complex<T>>(x)));
#else
// 在其他编译环境下，使用标准库计算复数的反双曲正切
return static_cast<c10::complex<T>>(
    std::atanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log1p(const c10::complex<T>& z) {
#if defined(__APPLE__) || defined(__MACOSX) || defined(__CUDACC__) || \
    defined(__HIPCC__)
// 对于 macOS 和 CUDA，采用这个实现方式以避免高相对误差问题
// 参考：https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
// 对于 CUDA，由于 thrust::log(thrust::complex) 编译时间过长，因此使用这个实现方式

// 计算 log(1 + z)，其中 z = x + iy
// 如果定义 1 + z = r * e ^ (i * a)，则有
// log(r * e ^ (i * a)) = log(r) + i * a
// 对于 z = x + iy，r 的表达式为
// r = ((1 + x) ^ 2 + y ^ 2) ^ 0.5
//   = (1 + x ^ 2 + 2 * x + y ^ 2) ^ 0.5
// 因此，log(r) 表达式为
// log(r) = 0.5 * log(1 + x ^ 2 + 2 * x + y ^ 2)
//        = 0.5 * log1p(x * (x + 2) + y ^ 2)
// 需要根据条件使用这个表达式以避免溢出和下溢问题
T x = z.real();
T y = z.imag();
T zabs = std::abs(z);
T theta = std::atan2(y, x + T(1));
if (zabs < 0.5) {
  T r = x * (T(2) + x) + y * y;
  if (r == 0) { // 处理下溢情况
    return {x, theta};
  }
  return {T(0.5) * std::log1p(r), theta};
} else {
  T z0 = std::hypot(x + 1, y);
  return {std::log(z0), theta};
}
#else
// CPU 路径下的实现
// 基于 https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
c10::complex<T> u = z + T(1);
if (u == T(1)) {
  return z;
} else {
  auto log_u = log(u);
  if (u - T(1) == z) {
    return log_u;
  }
  return log_u * (z / (u - T(1)));
}
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> expm1(const c10::complex<T>& z) {
// 计算 exp(z) - 1，其中 z = x + i * y
// 定义 z = x + i * y
// f = e ^ (x + i * y) - 1
//   = e ^ x * e ^ (i * y) - 1
//   = (e ^ x * cos(y) - 1) + i * (e ^ x * sin(y))
//   = (e ^ x - 1) * cos(y) - (1 - cos(y)) + i * e ^ x * sin(y)
//   = expm1(x) * cos(y) - 2 * sin(y / 2) ^ 2 + i * e ^ x * sin(y)
T x = z.real();
T y = z.imag();
T a = std::sin(y / 2);
T er = std::expm1(x) * std::cos(y) - T(2) * a * a;
T ei = std::exp(x) * std::sin(y);
return {er, ei};
}
# 引入复杂数数学运算库中的平方根函数
using c10_complex_math::sqrt;
# 引入复杂数数学运算库中的正切函数
using c10_complex_math::tan;
# 引入复杂数数学运算库中的双曲正切函数
using c10_complex_math::tanh;

# 将以下标准数学函数添加到 std 命名空间中
namespace std {

    # 引入复杂数数学运算库中的反余弦函数
    using c10_complex_math::acos;
    # 引入复杂数数学运算库中的反双曲余弦函数
    using c10_complex_math::acosh;
    # 引入复杂数数学运算库中的反正弦函数
    using c10_complex_math::asin;
    # 引入复杂数数学运算库中的反双曲正弦函数
    using c10_complex_math::asinh;
    # 引入复杂数数学运算库中的反正切函数
    using c10_complex_math::atan;
    # 引入复杂数数学运算库中的反双曲正切函数
    using c10_complex_math::atanh;
    # 引入复杂数数学运算库中的余弦函数
    using c10_complex_math::cos;
    # 引入复杂数数学运算库中的双曲余弦函数
    using c10_complex_math::cosh;
    # 引入复杂数数学运算库中的指数函数
    using c10_complex_math::exp;
    # 引入复杂数数学运算库中的 expm1 函数
    using c10_complex_math::expm1;
    # 引入复杂数数学运算库中的自然对数函数
    using c10_complex_math::log;
    # 引入复杂数数学运算库中的常用对数函数
    using c10_complex_math::log10;
    # 引入复杂数数学运算库中的 log1p 函数
    using c10_complex_math::log1p;
    # 引入复杂数数学运算库中的基2对数函数
    using c10_complex_math::log2;
    # 引入复杂数数学运算库中的指数函数
    using c10_complex_math::pow;
    # 引入复杂数数学运算库中的正弦函数
    using c10_complex_math::sin;
    # 引入复杂数数学运算库中的双曲正弦函数
    using c10_complex_math::sinh;
    # 引入复杂数数学运算库中的平方根函数
    using c10_complex_math::sqrt;
    # 引入复杂数数学运算库中的正切函数
    using c10_complex_math::tan;
    # 引入复杂数数学运算库中的双曲正切函数
    using c10_complex_math::tanh;

} // namespace std
```