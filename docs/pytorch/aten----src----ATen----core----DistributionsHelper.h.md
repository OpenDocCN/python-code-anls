# `.\pytorch\aten\src\ATen\core\DistributionsHelper.h`

```
#pragma once

#include <ATen/core/Array.h>
#include <ATen/core/TransformationHelper.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/MathConstants.h>
#include <c10/util/Optional.h>
#include <c10/macros/Macros.h>

#include <type_traits>
#include <limits>
#include <cmath>

/**
 * Distributions kernel adapted from THRandom.cpp
 * The kernels try to follow std::random distributions signature
 * For instance: in ATen
 *      auto gen = at::detail::createCPUGenerator();
 *      at::uniform_real_distribution<double> uniform(0, 1);
 *      auto sample = uniform(gen.get());
 *
 *      vs std::random
 *
 *      std::mt19937 gen;
 *      std::uniform_real_distribution uniform(0, 1);
 *      auto sample = uniform(gen);
 */

namespace at {
namespace {

/**
 * Samples a discrete uniform distribution in the range [base, base+range) of type T
 */
template <typename T>
struct uniform_int_from_to_distribution {
  
  C10_HOST_DEVICE inline uniform_int_from_to_distribution(uint64_t range, int64_t base) : range_(range), base_(base) {}
  
  /**
   * Generates a random value of type T using the provided RNG.
   */
  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    // Depending on the size of range_, use either 32-bit or 64-bit random generation
    if ((std::is_same<T, int64_t>::value ||
         std::is_same<T, double>::value ||
         std::is_same<T, float>::value ||
         std::is_same<T, at::BFloat16>::value) && range_ >= 1ULL << 32)
    {
      return transformation::uniform_int_from_to<T>(generator->random64(), range_, base_);
    } else {
      return transformation::uniform_int_from_to<T>(generator->random(), range_, base_);
    }
  }

  private:
    uint64_t range_;   // The range of the distribution
    int64_t base_;     // The base value of the distribution
};

/**
 * Samples a discrete uniform distribution in the range [min_value(int64_t), max_value(int64_t)]
 */
template <typename T>
struct uniform_int_full_range_distribution {
  
  /**
   * Generates a random value of type T using the provided RNG.
   */
  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    return transformation::uniform_int_full_range<T>(generator->random64());
  }

};

/**
 * Samples a discrete uniform distribution in the range [0, max_value(T)] for integral types
 * and [0, 2^mantissa] for floating-point types.
 */
template <typename T>
struct uniform_int_distribution {
  
  /**
   * Generates a random value of type T using the provided RNG.
   */
  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    // Selects appropriate random generation method based on T type
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int64_t>) {
      return transformation::uniform_int<T>(generator->random64());
    } else {
      return transformation::uniform_int<T>(generator->random());
    }
  }

};

/**
 * Samples a uniform distribution in the range [from, to) of type T
 */
template <typename T>
struct uniform_real_distribution {
  
  /**
   * Constructs a uniform distribution with the specified range [from, to).
   * Checks for valid range constraints.
   */
  C10_HOST_DEVICE inline uniform_real_distribution(T from, T to) {
    TORCH_CHECK_IF_NOT_ON_CUDA(from <= to);  // Ensures from is less than or equal to to
    TORCH_CHECK_IF_NOT_ON_CUDA(to - from <= std::numeric_limits<T>::max());  // Ensures valid range within type T
    from_ = from;  // Sets the lower bound of the distribution
    to_ = to;      // Sets the upper bound of the distribution
  }

  /**
   * Generates a random value of type T using the provided RNG.
   */
  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG generator) {
    # 如果模板类型 T 是 double 类型，则使用64位随机数生成器生成均匀分布的实数，范围是 from_ 到 to_
    if constexpr (std::is_same_v<T, double>) {
        return transformation::uniform_real<T>(generator->random64(), from_, to_);
    } else {
        # 如果模板类型 T 不是 double 类型，则使用普通随机数生成器生成均匀分布的实数，范围是 from_ 到 to_
        return transformation::uniform_real<T>(generator->random(), from_, to_);
    }
  }

  private:
    # 存储均匀分布的起始值 from_
    T from_;
    # 存储均匀分布的结束值 to_
    T to_;
// 定义宏，用于生成检查是否具有特定成员函数的结构体模板
#define DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(member)              \
template <typename T>                                                \
struct has_member_##member                                           \
{                                                                    \
    typedef char yes;                                                \
    typedef long no;                                                 \
    // 检测是否能通过 decltype 获取到成员函数指针，并使用 char 表示成功
    template <typename U> static yes test(decltype(&U::member));     \
    // 若无法获取成员函数指针，使用 long 表示失败
    template <typename U> static no test(...);                       \
    // 如果能获取到成员函数指针，则表示有该成员函数
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes); \
}

// 生成用于检查 RNG 类型是否具有指定双精度正态分布采样函数的结构体模板
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_double_normal_sample);
// 生成用于检查 RNG 类型是否具有指定设置双精度正态分布采样函数的结构体模板
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_double_normal_sample);
// 生成用于检查 RNG 类型是否具有指定单精度正态分布采样函数的结构体模板
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_float_normal_sample);
// 生成用于检查 RNG 类型是否具有指定设置单精度正态分布采样函数的结构体模板
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_float_normal_sample);

// 定义宏，生成可能的下一正态分布采样方法模板，根据 RNG 和返回类型生成函数
#define DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(TYPE)                                      \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          // 使用 SFINAE 技术进行编译时条件判断，确保 RNG 具有所需的成员函数                     \
          typename std::enable_if_t<(                                                               \
            has_member_next_##TYPE##_normal_sample<RNG>::value &&                                   \
            has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  \
          ), int> = 0>                                                                              \
// 内联函数，尝试获取下一个指定类型正态分布样本，并将结果存入 ret 中
C10_HOST_DEVICE inline bool maybe_get_next_##TYPE##_normal_sample(RNG* generator, ret_type* ret) {  \
  // 如果成功获取到下一个正态分布样本                                              \
  if (generator->next_##TYPE##_normal_sample()) {                                                   \
    // 将样本值复制到 ret 中                                                          \
    *ret = *(generator->next_##TYPE##_normal_sample());                                             \
    // 设置下一个正态分布样本为无效值                                               \
    generator->set_next_##TYPE##_normal_sample(std::optional<TYPE>());                              \
    // 返回成功获取样本的标志                                                        \
    return true;                                                                                    \
  }                                                                                                 \
  // 未能成功获取样本，返回失败标志                                                      \
  return false;                                                                                     \
}                                                                                                   \
                                                                                                    \
/**
 * 生成下一个类型为 TYPE 的正态分布样本，根据 RNG 选择不同的实现
 * 如果 RNG 不支持直接生成正态分布样本，则返回 false
 */
template <typename RNG, typename ret_type,
          typename std::enable_if_t<(
            !has_member_next_##TYPE##_normal_sample<RNG>::value ||
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value
          ), int> = 0>
C10_HOST_DEVICE inline bool maybe_get_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type* /*ret*/) {
  return false;
}

/**
 * 如果 RNG 支持设置下一个类型为 TYPE 的正态分布样本，则将缓存的值设置进去
 */
template <typename RNG, typename ret_type,
          typename std::enable_if_t<(
            has_member_set_next_##TYPE##_normal_sample<RNG>::value
          ), int> = 0>
C10_HOST_DEVICE inline void maybe_set_next_##TYPE##_normal_sample(RNG* generator, ret_type cache) {
  generator->set_next_##TYPE##_normal_sample(cache);
}

/**
 * 如果 RNG 不支持设置下一个类型为 TYPE 的正态分布样本，则不执行任何操作
 */
template <typename RNG, typename ret_type,
          typename std::enable_if_t<(
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value
          ), int> = 0>
C10_HOST_DEVICE inline void maybe_set_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type /*cache*/) {
}

/**
 * 生成下一个双精度浮点数（double）正态分布样本的方法
 */
DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(double);

/**
 * 生成下一个单精度浮点数（float）正态分布样本的方法
 */
DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(float);

/**
 * 正态分布结构体，实现了对正态分布的采样
 * 使用 Box-Muller 方法
 * 接受均值（mean）和标准差（stdv）作为输入参数
 * 注意，Box-Muller 方法每次返回两个样本，因此在 CPUGeneratorImpl 类中缓存了“下一个”样本
 */
template <typename T>
struct normal_distribution {

  /**
   * 构造函数，初始化均值和标准差，并进行检查确保标准差为非负数
   */
  C10_HOST_DEVICE inline normal_distribution(T mean_in, T stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in >= 0, "stdv_in must be positive: ", stdv_in);
    mean = mean_in;
    stdv = stdv_in;
  }

  /**
   * 操作符重载，用于生成正态分布样本
   * 使用指定的 RNG 对象作为参数
   * 返回样本的类型为 dist_acctype<T>
   */
  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG generator){
    dist_acctype<T> ret;
    // 如果可用，返回缓存的值
    if constexpr (std::is_same_v<T, double>) {
      // 如果可以获取下一个双精度正态分布样本，则返回经过转换的正态分布值
      if (maybe_get_next_double_normal_sample(generator, &ret)) {
        return transformation::normal(ret, mean, stdv);
      }
    } else {
      // 如果可以获取下一个单精度正态分布样本，则返回经过转换的正态分布值
      if (maybe_get_next_float_normal_sample(generator, &ret)) {
        return transformation::normal(ret, mean, stdv);
      }
    }
    // 否则生成新的正态分布值
    uniform_real_distribution<T> uniform(0.0, 1.0);
    const dist_acctype<T> u1 = uniform(generator);
    const dist_acctype<T> u2 = uniform(generator);
    const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log1p(-u2));
    const dist_acctype<T> theta = static_cast<T>(2.0) * c10::pi<T> * u1;
    // 根据 T 的类型设置下一个正态分布样本
    if constexpr (std::is_same_v<T, double>) {
      maybe_set_next_double_normal_sample(generator, r * ::sin(theta));
    } else {
      maybe_set_next_float_normal_sample(generator, r * ::sin(theta));
    }
    // 计算并返回正态分布值
    ret = r * ::cos(theta);
    return transformation::normal(ret, mean, stdv);
  }

  private:
    T mean;   // 正态分布的均值
    T stdv;   // 正态分布的标准差
};

template <typename T>
struct DiscreteDistributionType { using type = float; };

template <> struct DiscreteDistributionType<double> { using type = double; };

/**
 * Samples a bernoulli distribution given a probability input
 */
template <typename T>
struct bernoulli_distribution {

  C10_HOST_DEVICE inline bernoulli_distribution(T p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in >= 0 && p_in <= 1);
    p = p_in;
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::bernoulli<T>(uniform(generator), p);
  }

  private:
    T p;
};

/**
 * Samples a geometric distribution given a probability input
 */
template <typename T>
struct geometric_distribution {

  C10_HOST_DEVICE inline geometric_distribution(T p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in > 0 && p_in < 1);
    p = p_in;
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::geometric<T>(uniform(generator), p);
  }

  private:
    T p;
};

/**
 * Samples an exponential distribution given a lambda input
 */
template <typename T>
struct exponential_distribution {

  C10_HOST_DEVICE inline exponential_distribution(T lambda_in) : lambda(lambda_in) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::exponential<T>(uniform(generator), lambda);
  }

  private:
    T lambda;
};

/**
 * Samples a cauchy distribution given median and sigma as inputs
 */
template <typename T>
struct cauchy_distribution {

  C10_HOST_DEVICE inline cauchy_distribution(T median_in, T sigma_in) : median(median_in), sigma(sigma_in) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::cauchy<T>(uniform(generator), median, sigma);
  }

  private:
    T median;
    T sigma;
};

/**
 * Samples a lognormal distribution
 * Takes mean and standard deviation as inputs
 * Outputs two samples at a time
 */
template <typename T>
struct lognormal_distribution {

  C10_HOST_DEVICE inline lognormal_distribution(T mean_in, T stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in > 0);
    mean = mean_in;
    stdv = stdv_in;
  }

  template<typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator){
    normal_distribution<T> normal(mean, stdv);
    return transformation::log_normal<T>(normal(generator));
  }

  private:
    T mean;
    T stdv;
};



} // namespace at


注释：

// 这是一个命名空间结束的标志，指示所有的结构体和模板定义属于命名空间 `at`
```