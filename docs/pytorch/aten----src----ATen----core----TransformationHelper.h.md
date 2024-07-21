# `.\pytorch\aten\src\ATen\core\TransformationHelper.h`

```py
/**
 * 包含头文件，这些头文件包括了数值工具、宏定义、半精度浮点数、BFloat16 类型、数学常量等。
 */
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/MathConstants.h>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <limits>
#include <type_traits>

namespace at {

// 使用 DistAccumType 定义分布累积类型，用于分布类型的累积值选择
template <typename T>
struct DistAccumType {  };

// 如果是 CUDA 或 HIP 环境下编译，将 half 类型映射为 float 类型
#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct DistAccumType<half> { using type = float; };
#endif
// 将 BFloat16 类型映射为 float 类型
template <> struct DistAccumType<BFloat16> { using type = float; };
// 将 Half 类型映射为 float 类型
template <> struct DistAccumType<Half> { using type = float; };
// 将 float 类型映射为 float 类型（即没有变化）
template <> struct DistAccumType<float> { using type = float; };
// 将 double 类型映射为 double 类型（即没有变化）
template <> struct DistAccumType<double> { using type = double; };

// 使用 DistAccumType 中定义的类型别名，用于选择合适的累积类型
template <typename T>
using dist_acctype = typename DistAccumType<T>::type;

namespace transformation {

/**
 * `torch.Tensor.random_()` 的变换函数，当指定 `from` 和 `to` 时使用。
 * `range` 表示 `to - from`
 * `base` 表示 `from`
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_from_to(V val, uint64_t range, int64_t base) {
  return static_cast<T>(static_cast<int64_t>((val % range) + base));
}

/**
 * `torch.Tensor.random_()` 的变换函数，当 `from=min_value(int64_t)` 且 `to=None` 时使用。
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_full_range(V val) {
  return static_cast<T>(static_cast<int64_t>(val));
}

/**
 * `torch.Tensor.random_()` 的变换函数，当未指定 `from` 和 `to` 时使用。
 * 为了解决 GitHub 问题 46391 报告的编译器警告，此重载版本中 T 不能是 float 或 double 类型。
 */
template <typename T, typename V>
C10_HOST_DEVICE inline std::enable_if_t<!(std::is_floating_point_v<T>), T> uniform_int(V val) {
  if constexpr (std::is_same_v<T, bool>) {
    return static_cast<bool>(val & 1);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else if constexpr (std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16>) {
    return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else {
    assert(false);  // 如果 T 不是上述任何类型，触发断言错误
    return 0;
  }
}

/**
 * `torch.Tensor.random_()` 的重载变换函数，当未指定 `from` 和 `to` 时使用，
 * 用于解决 GitHub 问题 46391 报告的编译器警告。此版本中 T 必须是 float 或 double 类型。
 */
template<typename T, typename V>
/**
 * Generates a uniformly distributed floating-point number within the range [0, 1),
 * converted to type T. This function is enabled only for floating-point types.
 */
C10_HOST_DEVICE inline std::enable_if_t<std::is_floating_point_v<T>, T> uniform_int(V val) {
  return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
}

/**
 * Generates a uniformly distributed real number within the range [from, to),
 * converted to type `dist_acctype<T>`. The input `val` is used as a seed or value
 * to transform into the desired range.
 */
template <typename T, typename V>
C10_HOST_DEVICE inline dist_acctype<T> uniform_real(V val, T from, T to) {
  constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  dist_acctype<T> x = (val & MASK) * DIVISOR;
  return (x * (to - from) + from);
}

/**
 * Transforms a normally distributed random variable `val` with mean 0.0 and standard
 * deviation 1.0 to a normally distributed random variable with mean `mean` and
 * standard deviation `std`.
 */
template <typename T>
C10_HOST_DEVICE inline T normal(T val, T mean, T std) {
  return val * std + mean;
}

/**
 * Transforms a uniformly distributed random variable `val` between 0.0 and 1.0 to
 * a Cauchy distributed random variable with location parameter `median` and
 * scale parameter `sigma`.
 */
template <typename T>
C10_HOST_DEVICE inline T cauchy(T val, T median, T sigma) {
  // Explanation of the clipping mechanism for `val` to avoid `__tanf` overflow:
  // Values of `val` close to 0 or 1 are clipped to avoid `__tanf` returning `inf` or `-inf`.
  constexpr T eps = std::numeric_limits<T>::epsilon();
  constexpr T one_minus_eps = 1 - eps;
  constexpr T zero_plus_eps = 0 + eps;
  val = (val > one_minus_eps ? one_minus_eps : val);
  val = (val < zero_plus_eps ? zero_plus_eps : val);
  return median + sigma * at::tan(c10::pi<T> * (val - static_cast<T>(0.5)));
}

/**
 * Specialized version of `cauchy` for type `double`, with the same functionality
 * as the template version but tailored for `double` precision.
 */
template <>
C10_HOST_DEVICE inline double cauchy(double val, double median, double sigma) {
  return median + sigma * at::tan(c10::pi<double> * (val - static_cast<double>(0.5)));
}

/**
 * Transforms a uniformly distributed random variable `val` between 0.0 and 1.0 to
 * an exponentially distributed random variable with rate parameter `lambda`.
 * This function differs between CUDA and CPU implementations to handle specific
 * numerical characteristics and behavior of the exponential distribution.
 */
template <typename T>
C10_HOST_DEVICE inline T exponential(T val, T lambda) {
  // Explanation of the special handling for CUDA and CPU implementations
  // due to differences in numerical characteristics and handling of logarithms.
#if defined(__CUDACC__) || defined(__HIPCC__)
  // For CUDA and HIP, special handling of log values to avoid underflow issues.
  // Reference: https://github.com/pytorch/pytorch/issues/16706
  auto log = val >= static_cast<T>(1.) - std::numeric_limits<T>::epsilon() / 2
      ? -std::numeric_limits<T>::epsilon() / 2
      : at::log(val);
  return static_cast<T>(-1.0) / lambda * log;
#else
  // Standard CPU implementation using natural logarithm.
  return static_cast<T>(-1.0) / lambda * at::log(val);
#endif
}
/**
 * 转换均匀分布的 `val`（在0.0和1.0之间）为几何分布，成功概率为 `p`。
 * 参考：https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
 */
template <typename T>
C10_HOST_DEVICE inline T geometric(T val, T p) {
  // 使用对数函数计算几何分布的变换
  return static_cast<T>(::ceil(at::log(val) / at::log1p(-p)));
}

/**
 * 将正态分布的 `val` 转换为对数正态分布。
 * 参考：https://en.wikipedia.org/wiki/Log-normal_distribution#Mode,_median,_quantiles
 */
template <typename T>
C10_HOST_DEVICE inline T log_normal(T val) {
  // 使用指数函数计算对数正态分布的变换
  return at::exp(val);
}

/**
 * 转换均匀分布的 `val`（在0.0和1.0之间）为伯努利分布，成功概率为 `p`。
 */
template <typename T>
C10_HOST_DEVICE inline T bernoulli(T val, T p) {
  // 如果 `val` 小于成功概率 `p`，返回 true（1），否则返回 false（0）
  return val < p;
}

}} // namespace at::transformation
```