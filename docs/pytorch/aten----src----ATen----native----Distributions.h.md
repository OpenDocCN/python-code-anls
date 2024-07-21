# `.\pytorch\aten\src\ATen\native\Distributions.h`

```
#pragma once
// 使用#pragma once确保头文件只被编译一次

#include <ATen/native/Math.h>
#include <c10/macros/Macros.h>
#include <c10/util/MathConstants.h>
// 引入必要的头文件

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(__CUDA_ARCH__)
#include <c10/cuda/CUDAMathCompat.h>
#define compat_exp c10::cuda::compat::exp
#define compat_ceil c10::cuda::compat::ceil
#define compat_floor c10::cuda::compat::floor
#define compat_log c10::cuda::compat::log
#define compat_pow c10::cuda::compat::pow
#define compat_sqrt c10::cuda::compat::sqrt
#define compat_tan c10::cuda::compat::tan
#define compat_abs c10::cuda::compat::abs
#define compat_log1p c10::cuda::compat::log1p
#elif defined(__HIPCC__)
#include <c10/hip/HIPMathCompat.h>
#define compat_exp c10::hip::compat::exp
#define compat_ceil c10::hip::compat::ceil
#define compat_floor c10::hip::compat::floor
#define compat_log c10::hip::compat::log
#define compat_pow c10::hip::compat::pow
#define compat_sqrt c10::hip::compat::sqrt
#define compat_tan c10::hip::compat::tan
#define compat_abs c10::hip::compat::abs
#define compat_log1p c10::hip::compat::log1p
#else
#define compat_exp std::exp
#define compat_ceil std::ceil
#define compat_floor std::floor
#define compat_log std::log
#define compat_pow std::pow
#define compat_sqrt std::sqrt
#define compat_tan std::tan
#define compat_abs std::abs
#define compat_log1p std::log1p
#endif
// 根据不同的编译器设置兼容的数学函数宏

namespace {

#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)
// we cannot use std::isnan directly due to some incompatibility of
// gcc constexpr'ing and nvcc
// 针对GPU，不能直接使用std::isnan，因为gcc constexpr和nvcc不兼容
using std::isnan;
#endif

// Here sampler_t should be function type scalar_t(void). For gpu
// "sampler" is a device function, but since ROCM doesn't have
// equivalent to nvstd::function, we use a template type parameter to
// capture it.
// sampler_t应该是函数类型 scalar_t(void)。对于GPU，“sampler”是设备函数，但是由于ROCM没有等效的nvstd::function，
// 我们使用模板类型参数来捕获它。
template<typename scalar_t, typename sampler_t>
struct BaseSampler {
  sampler_t sampler;
  C10_DEVICE BaseSampler(const sampler_t& sampler): sampler(sampler) {}
  // BaseSampler结构体，包含一个sampler_t类型的成员变量sampler，并用构造函数初始化它
  // C10_DEVICE 表示这是一个设备函数
  C10_DEVICE scalar_t sample() {
    return sampler();
  }
  // sample函数返回sampler的调用结果
};

// The function `sample_gamma` is
// is adapted from Numpy's distributions.c implementation.
// It is MIT licensed, so here is the copyright:
// 函数`sample_gamma`修改自Numpy的distributions.c实现，遵循MIT许可证
// 定义模板函数 sample_gamma，用于从 Gamma 分布中抽样
template<typename scalar_t, typename accscalar_t, typename uniform_sampler_t, typename normal_sampler_t>
C10_DEVICE scalar_t sample_gamma(scalar_t alpha, BaseSampler<accscalar_t, uniform_sampler_t>& standard_uniform, BaseSampler<accscalar_t, normal_sampler_t>& standard_normal) {
  // 初始化缩放因子为 1.0
  accscalar_t scale = 1.0f;

  // 如果 alpha 小于 1.0，增加 alpha 以提高接受概率
  if (alpha < 1.0f) {
    // 如果 alpha 等于 0，直接返回 0
    if (alpha == 0.f) return 0.f;
    // 根据 alpha 调整 scale
    scale *= compat_pow(1 - standard_uniform.sample(), 1.0f / alpha);
    alpha += 1.0f;
  }

  // 使用 Marsaglia 和 Tsang (2000) 的接受-拒绝方法抽样 Gamma 分布
  const accscalar_t d = alpha - 1.0f / 3.0f;
  const accscalar_t c = 1.0f / compat_sqrt(9.0f * d);
  for (;;) {
    accscalar_t x, y;
    // 循环直到找到合适的样本
    do {
      // 从标准正态分布中抽样
      x = standard_normal.sample();
      y = 1.0f + c * x;
    } while (y <= 0);
    // 计算必要的值
    const accscalar_t v = y * y * y;
    const accscalar_t u = 1 - standard_uniform.sample();
    const accscalar_t xx = x * x;
    // 根据接受-拒绝条件返回样本值
    if (u < 1.0f - 0.0331f * xx * xx)
      return static_cast<scalar_t>(scale * d * v);
    if (compat_log(u) < 0.5f * xx + d * (1.0f - v + compat_log(v)))
      return static_cast<scalar_t>(scale * d * v);
  }
}

// 函数 stirling_approx_tail，用于计算 Gamma 分布的 Stirling 近似尾部
template<typename scalar_t>
C10_DEVICE scalar_t stirling_approx_tail(scalar_t k) {
  // 预定义的常数数组
  const static scalar_t kTailValues[] = {
    0.0810614667953272,
    0.0413406959554092,
    0.0276779256849983,
    0.02079067210376509,
    0.0166446911898211,
    0.0138761288230707,
    0.0118967099458917,
    0.0104112652619720,
    0.00925546218271273,
    0.00833056343336287
  };
  如果 k 小于等于 9，返回预定义的尾部数值数组中索引为 k 的值
  if (k <= 9) {
    return kTailValues[static_cast<size_t>(k)];
  }
  计算 k+1 的平方
  scalar_t kp1sq = (k + 1) * (k + 1);
  根据公式计算超过预定义范围的尾部值
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
// 如果样本数或概率小于等于零，直接返回零
if (count <= 0.0 || prob <= 0.0) {
    return 0;
} else if (prob >= 1.0) { // 如果概率大于等于1，直接返回样本数
    return count;
} else if (prob <= 0.5) { // 如果概率小于等于0.5，使用变换拒绝采样
    // 如果 count 乘以 prob 大于等于 10.0，则选择使用 btrs 方法进行采样
    if (count * prob >= 10.0) {
      // 调用 btrs 方法进行二项分布采样
      return btrs<scalar_t, accscalar_t, uniform_sampler_t>(count, prob, standard_uniform);
    } else {
      // 否则选择使用二项分布反演方法进行采样
      return binomial_inversion<scalar_t, accscalar_t, uniform_sampler_t>(count, prob, standard_uniform);
    }
  } else if (prob > 0.5) {
    // 如果 prob 大于 0.5，则计算 qprob 为 1.0 减去 prob
    scalar_t qprob = 1.0 - prob;
    // 如果 count 乘以 qprob 大于等于 10.0，则选择使用 btrs 方法进行采样
    if (count * qprob >= 10.0) {
      // 计算并返回 count 减去使用 btrs 方法的采样结果
      return count - btrs<scalar_t, accscalar_t, uniform_sampler_t>(count, qprob, standard_uniform);
    } else {
      // 否则计算并返回 count 减去使用二项分布反演方法的采样结果
      return count - binomial_inversion<scalar_t, accscalar_t, uniform_sampler_t>(count, qprob, standard_uniform);
    }
  } else {
    // 如果 prob 不满足以上条件，则返回 NaN
    // 这种情况可能是 prob 是 NaN，或者 prob 小于等于 0.5 但 count * prob < 10.0
    return static_cast<scalar_t>(NAN);
  }
/*
 * This function computes the digamma function approximation using methods adapted from
 * the Cephes Math Library. It handles both positive and negative arguments and uses
 * specific series expansions for different ranges of the argument.
 */
template<typename scalar_t, typename accscalar_t>
C10_DEVICE inline scalar_t digamma_one(scalar_t x) {
  // Define a constant for the digamma at 10
  constexpr accscalar_t PSI_10 = 2.25175258906672110764;

  // Return infinity for x = 0
  if (x == 0) {
    return INFINITY;
  }

  // Initialize additional summand for corrections
  accscalar_t additional_summand = 0;

  // Check if x is an integer
  int x_is_integer = x == compat_floor(x);

  // Handle negative x
  if (x < 0) {
    // Return infinity for negative integer x
    if (x_is_integer) {
      return INFINITY;
    }

    // Adjust x and compute additional summand
    additional_summand = -c10::pi<scalar_t> /
        compat_tan(c10::pi<scalar_t> * x);
    x = 1 - x;
  }

  // Push x to be >= 10 by adding terms to result
  accscalar_t result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }

  // Exact value for x = 10
  if (x == 10) {
    return result + PSI_10 + additional_summand;
  }

  // Compute asymptotic digamma using a series approximation
  static const accscalar_t A[] = {
     8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
     7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
     3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
     8.33333333333333333333E-2,
  };

  // Calculate y based on x
  accscalar_t y = 0;
  if (x < 1.0e17f) {
    accscalar_t z = 1.0 / (x * x);
    y = z * polevl<accscalar_t>(z, A, 6);
  }

  // Return the final digamma value after all corrections
  return static_cast<scalar_t>(
      result + compat_log(x) - (0.5f / x) - y + additional_summand);
}

/*
 * Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
 * for a random number x drawn from a standard Gamma distribution Gamma(alpha).
 * Uses different expansions based on the size of x and alpha for accuracy.
 */
template <typename scalar_t, typename accscalar_t>
C10_HOST_DEVICE scalar_t standard_gamma_grad_one(scalar_t alpha_, scalar_t x_) {
  // Cast input parameters to appropriate types
  accscalar_t x = static_cast<accscalar_t>(x_);
  accscalar_t alpha = static_cast<accscalar_t>(alpha_);

  // Use Taylor series expansion for small x
  if (x < 0.8f) {
    // Initialize numerator and denominator
    accscalar_t numer = 1;
    accscalar_t denom = alpha;
    auto series1 = numer / denom;
    auto series2 = numer / (denom * denom);

    // Iterate to compute series approximation
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / static_cast<accscalar_t>(i);
      denom += 1;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }

    // Compute necessary components for gradient calculation
    const auto pow_x_alpha = compat_pow(x, alpha);
    const auto gamma_pdf = compat_pow(x, alpha - 1) * compat_exp(-x);
    const auto gamma_cdf = pow_x_alpha * series1;
    const auto gamma_cdf_alpha =
        (compat_log(x) - digamma_one<accscalar_t, accscalar_t>(alpha)) *
            gamma_cdf -
        pow_x_alpha * series2;

    // Compute and return the result ensuring no NaN
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return isnan(result) ? static_cast<scalar_t>( 0.f ) : static_cast<scalar_t>(result);
  }

  // Use a different expansion for large alpha
  if (alpha > 8.0f) {
    // 如果 x 在 [0.9 * alpha, 1.1 * alpha] 范围内
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      // 计算第一个分子
      const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
      // 计算第二个分子
      const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
          - 65 * x * x / alpha + alpha * (107 + 3600 * x);
      // 计算分母
      const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
      // 返回结果，进行类型转换为 scalar_t 类型
      return static_cast<scalar_t>(numer_1 * numer_2 / denom);
    }
    // 计算常数 denom
    const auto denom = compat_sqrt(8 * alpha);
    // 计算 term2
    const auto term2 = denom / (alpha - x);
    // 计算 term3
    const auto term3 = compat_pow(
        x - alpha - alpha * compat_log(x / alpha),
        static_cast<accscalar_t>(-1.5));
    // 根据 x 与 alpha 的大小关系选择不同的 term23 计算方式
    const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    // 计算 term1
    const auto term1 = compat_log(x / alpha) * term23 -
        compat_sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
    // 计算斯特林数近似值 stirling
    const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
    // 计算最终的 numer
    const auto numer = x * term1;
    // 返回最终结果，进行类型转换为 scalar_t 类型
    return static_cast<scalar_t>(-stirling * numer / denom);
  }

  // 使用二元有理函数逼近重参数化梯度。
  // 计算 u 和 v
  const auto u = compat_log(x / alpha);
  const auto v = compat_log(alpha);
  // 预定义的二维系数数组 coef_uv
  static const accscalar_t coef_uv[3][8] = {
    {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
     1, 0.32668115, 0.10406089, 0.0014179084},
    {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
     0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
    {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
     0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
  };
  // 计算 coef_v 数组
  accscalar_t coef_v[8];
  for (int i = 0; i < 8; ++i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  // 计算 p 和 q
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  // 返回最终结果，进行类型转换为 scalar_t 类型
  return static_cast<scalar_t>(compat_exp(p / q));
}

// 结束函数定义，此处表示 _beta_grad_alpha_small 函数的末尾

// 计算 Beta 函数在 alpha 参数上的近似重参数化梯度
// 假设 x 接近零，并使用泰勒展开
template <typename scalar_t, typename accscalar_t>
C10_DEVICE inline scalar_t _beta_grad_alpha_small(scalar_t x, scalar_t alpha, scalar_t beta) {
  // 计算系数，使用 digamma 函数的差值和对数函数的兼容版本
  const scalar_t factor = digamma_one<scalar_t, accscalar_t>(alpha)
                        - digamma_one<scalar_t, accscalar_t>(alpha + beta) - compat_log(x);
  // 初始化数值和级数项
  scalar_t numer = 1;
  scalar_t series = numer / alpha * (factor + 1 / alpha);
  // 进行泰勒级数展开，迭代10次
  for (int i = 1; i <= 10; ++i) {
    scalar_t casted_i = static_cast<scalar_t>(i);
    numer *= (casted_i - beta) * x / casted_i;
    const scalar_t denom = alpha + casted_i;
    series += numer / denom * (factor + 1 / denom);
  }
  // 计算结果并处理可能的 NaN
  const scalar_t result = x * compat_pow(1 - x, -beta) * series;
  return isnan(result) ? static_cast<scalar_t>( 0.f ) : result;
}

// 计算 Beta 函数在 beta 参数上的近似重参数化梯度
// 假设 x 接近零，并使用泰勒展开
template <typename scalar_t, typename accscalar_t>
C10_DEVICE inline scalar_t _beta_grad_beta_small(scalar_t x, scalar_t alpha, scalar_t beta) {
  // 计算系数，使用 digamma 函数的差值
  const scalar_t factor = digamma_one<scalar_t, accscalar_t>(alpha + beta) - digamma_one<scalar_t, accscalar_t>(beta);
  // 初始化数值和级数项
  scalar_t numer = 1, betas = 1, dbetas = 0, series = factor / alpha;
  // 进行泰勒级数展开，迭代8次
  for (int i = 1; i <= 8; ++i) {
    scalar_t casted_i = static_cast<scalar_t>(i);
    numer *= -x / casted_i;
    dbetas = dbetas * (beta - casted_i) + betas;
    betas = betas * (beta - casted_i);
    series += numer / (alpha + casted_i) * (dbetas + factor * betas);
  }
  // 计算结果并处理可能的 NaN
  const scalar_t result = -compat_pow(1 - x, 1 - beta) * series;
  return isnan(result) ? static_cast<scalar_t>( 0.f ) : result;
}

// 计算 Beta 函数在 alpha 参数上的近似重参数化梯度
// 假设 alpha 和 beta 都很大，并使用 Rice 鞍点展开
// 为了确保数值稳定性，使用更高精度进行计算
template<typename scalar_t, typename accscalar_t>
C10_DEVICE inline scalar_t _beta_grad_alpha_mid(accscalar_t x, accscalar_t alpha, accscalar_t beta) {
  // 计算总和和均值、标准差
  const accscalar_t total = alpha + beta;
  const accscalar_t mean = alpha / total;
  const accscalar_t std = compat_sqrt(alpha * beta / (total + 1)) / total;
  if (mean - 0.1 * std <= x && x <= mean + 0.1 * std) {
    // 避免在 x = mean 处的奇异性
    const accscalar_t poly = 47 * x * (beta * beta) * (beta * beta) + alpha * (
                           (43 + 20 * (16 + 27 * beta) * x) * (beta * beta) * beta + alpha * (
                           3 * (59 + 180 * beta - 90 * x) * (beta * beta) + alpha * (
                           (453 + 1620 * beta * (1 - x) - 455 * x) * beta + alpha * (
                           8 * (1 - x) * (135 * beta - 11)))));
    const accscalar_t prefactor_num = (1 + 12 * alpha) * (1 + 12 * beta) / (total * total);
  // 计算分子的前缀项，用于整个表达式的计算
  const accscalar_t prefactor_den = 12960 * alpha * alpha * alpha * beta * beta * (1 + 12 * total);
  
  // 返回整个表达式的计算结果
  return prefactor_num / (1 - x) * poly / prefactor_den;
}

// 计算非递归前缀项，用于整个表达式的计算
const accscalar_t prefactor = -x / compat_sqrt(2 * alpha * beta / total);

// 计算斯特林公式的近似值，用于整个表达式的计算
const accscalar_t stirling = (1 + 1 / (12 * alpha) + 1 / (288 * alpha * alpha))
                           * (1 + 1 / (12 * beta) + 1 / (288 * beta * beta))
                           / (1 + 1 / (12 * total) + 1 / (288 * total * total));

// 计算第一个项的分子部分，用于整个表达式的计算
const accscalar_t term1_num = 2 * (alpha * alpha) * (x - 1) + alpha * beta * (x - 1) - x * (beta * beta);

// 计算αx + βx，用于整个表达式的计算
const accscalar_t axbx = alpha * (x - 1) + beta * x;

// 计算第一个项的分母部分，用于整个表达式的计算
const accscalar_t term1_den = compat_sqrt(2 * alpha / beta) * compat_pow(total, static_cast<accscalar_t>(1.5f)) * axbx * axbx;

// 计算第一个项的值，用于整个表达式的计算
const accscalar_t term1 = term1_num / term1_den;

// 计算第二项的值，用于整个表达式的计算
const accscalar_t term2 = 0.5f * compat_log(alpha / (total * x));

// 计算第三项的分子部分，用于整个表达式的计算
const accscalar_t term3_num = compat_sqrt(8 * alpha * beta / total);

// 计算第三项的分母部分，用于整个表达式的计算
const accscalar_t term3_den = beta * x + alpha * (x - 1);

// 计算第三项的值，用于整个表达式的计算
const accscalar_t term3 = term3_num / term3_den;

// 计算第四项的基础部分，用于整个表达式的计算
const accscalar_t term4_base = beta * compat_log(beta / (total * (1 - x))) +
                             alpha * compat_log(alpha / (total * x));

// 计算第四项的值，用于整个表达式的计算
const accscalar_t term4 = compat_pow(term4_base, static_cast<accscalar_t>(-1.5f));

// 计算第一、二、三、四项的和，用于整个表达式的计算
const accscalar_t term1234 = term1 + term2 * (term3 + (x < mean ? term4 : -term4));

// 返回整个表达式的计算结果，并转换为所需的类型
return static_cast<scalar_t>(stirling * prefactor * term1234);
// 计算缩放重参数化梯度
//   -(d/dalpha cdf(x;alpha,beta)) / pdf(x;alpha,beta) / (1-x)
// 对从Beta分布Beta(alpha,beta)中抽取的随机数x进行计算。
// 此函数输入total=alpha+beta，以便于在Beta分布的重参数化梯度中实现Dirichlet分布。
template<typename scalar_t, typename accscalar_t>
C10_HOST_DEVICE inline scalar_t dirichlet_grad_one(scalar_t x, scalar_t alpha, scalar_t total) {
  accscalar_t x_ = static_cast<accscalar_t>(x);
  accscalar_t alpha_ = static_cast<accscalar_t>(alpha);
  accscalar_t total_ = static_cast<accscalar_t>(total);

  const scalar_t beta = total - alpha;
  const accscalar_t beta_ = total_ - alpha_;
  const scalar_t boundary = total * x * (1 - x);

  // 对于x接近0时，使用渐近逼近。
  if (x <= 0.5f && boundary < 2.5f) {
    return _beta_grad_alpha_small<scalar_t, accscalar_t>(x, alpha, beta);
  }

  // 对于x接近1时，使用渐近逼近。
  if (x >= 0.5f && boundary < 0.75f) {
    return -_beta_grad_beta_small<scalar_t, accscalar_t>(1 - x, beta, alpha);
  }

  // 当alpha和(total - alpha)都很大时，使用渐近逼近。
  if (alpha > 6 && beta > 6) {
    return _beta_grad_alpha_mid<scalar_t, accscalar_t>(x_, alpha_, beta_);
  }

  // 使用有理数修正的解析逼近。
  // 静态常量数组c存储了用于计算的系数。
  static const accscalar_t c[2][3][3][4] = {
    {{{1.003668233, -0.01061107488, -0.0657888334, 0.01201642863},
      {0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033},
      {-0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013}},
     {{0.221950385, -0.3187676331, 0.01799915743, 0.01074823814},
      {-0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057},
      {0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-05}},
     {{-0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565},
      {0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181},
      {0.005900514878, -0.001936558688, -9.495446725e-06, 5.385558597e-05}}},
    {{{1, -0.02924021934, -0.04438342661, 0.007285809825},
      {0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521},
      {-0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149}},
     {{0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273},
      {0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956},
      {-0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-05}},
     {{0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-05},
      {0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-05},
      {-0.0003477407336, 6.959756487e-05, 1.097287507e-05, -1.650964693e-06}}},
  };

  // 计算所需的对数值
  const accscalar_t u = compat_log(x_);
  const accscalar_t a = compat_log(alpha_) - u;
  const accscalar_t b = compat_log(total_) - a;
  const accscalar_t pow_u[3] = {1, u, u * u};
  const accscalar_t pow_a[3] = {1, a, a * a};
  accscalar_t p = 0.0;
  accscalar_t q = 0.0;

  // 使用系数数组进行计算
  for (int i = 0; i < 3; ++i) {
    // 循环计算累加和 p 和 q
    for (int j = 0; j < 3; ++j) {
      // 计算临时变量 ua，为 pow_u[i] 与 pow_a[j] 的乘积
      const accscalar_t ua = pow_u[i] * pow_a[j];
      // 计算累加和 p，使用多项式展开式
      p += ua * (c[0][i][j][0] + b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
      // 计算累加和 q，使用多项式展开式
      q += ua * (c[1][i][j][0] + b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
    }
  }
  // 计算最终的近似值 approx，通过调用 digamma_one 函数计算差值，然后除以 beta_
  const accscalar_t approx = x_ * (digamma_one<scalar_t, accscalar_t>(total_) - digamma_one<scalar_t, accscalar_t>(alpha_)) / beta_;
  // 返回计算的最终结果，转换为 scalar_t 类型
  return static_cast<scalar_t>(p / q * approx);
}

} // namespace


注释：


// 结束命名空间的定义

``` 

这段代码片段表示C++中命名空间定义的结束。在C++中，`}` 符号用于结束代码块，这里用于结束命名空间的定义。`// namespace` 是单行注释，用于说明紧随其后的代码行是命名空间的结束。
```