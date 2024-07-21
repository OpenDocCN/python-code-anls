# `.\pytorch\aten\src\ATen\native\cpu\DistributionTemplates.h`

```py
// 仅在编译一次时加载本文件
#pragma once

// 包含 ATen 库的 CPU 应用工具头文件
#include <ATen/CPUApplyUtils.h>
// 包含 ATen 库的分发头文件
#include <ATen/Dispatch.h>
// 包含 ATen 库的第二版本分发头文件
#include <ATen/Dispatch_v2.h>
// 包含 ATen 库的扩展基础头文件
#include <ATen/ExpandBase.h>
// 包含 ATen 库的核心分布助手头文件
#include <ATen/core/DistributionsHelper.h>
// 包含 ATen 库的张量迭代器头文件
#include <ATen/native/TensorIterator.h>
// 包含 ATen 库的 CPU 特定循环头文件
#include <ATen/native/cpu/Loops.h>
// 包含 <limits> 头文件，提供数字极限值
#include <limits>
// 包含 <mutex> 头文件，提供互斥体和锁定机制
#include <mutex>

// 如果定义了 CPU_CAPABILITY_AVX2，则包含 AVX2 加速数学函数头文件
#ifdef CPU_CAPABILITY_AVX2
#include <ATen/native/cpu/avx_mathfun.h>
#include <c10/util/irange.h>
#endif

// 定义 ATen 命名空间
namespace at {
// 定义 native 命名空间
namespace native {
// 定义 templates 命名空间
namespace templates {
// 定义 CPU 命名空间
namespace cpu {
// 定义匿名命名空间，用于私有函数和结构的封装

// ==================================================== Random ========================================================

// 模板函数：生成指定范围内随机数的核心函数
template<typename RNG>
void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, RNG generator) {
  // 根据迭代器的数据类型派发函数，生成指定范围的随机整数
  AT_DISPATCH_V2(iter.dtype(), "random_from_to_kernel_cpu", AT_WRAP([&] {
    // 使用互斥锁保护随机数生成器
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 在 CPU 上串行执行核心算法
    cpu_serial_kernel(iter, [range, base, generator]() -> scalar_t {
      // 创建指定范围的均匀整数分布
      uniform_int_from_to_distribution<scalar_t> random(range, base);
      // 返回生成的随机数
      return random(generator);
    });
  }), kBool, kHalf, kBFloat16, AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

// 特殊情况处理函数：处理完整 64 位整数范围的随机数生成
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIteratorBase& iter, RNG generator) {
  // 根据迭代器的数据类型派发函数，生成完整 64 位整数范围的随机数
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cpu", [&] {
    // 对于整型、双精度、单精度和 BFloat16 类型，使用互斥锁保护随机数生成器
    if constexpr (std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) {
      std::lock_guard<std::mutex> lock(generator->mutex_);
      // 在 CPU 上串行执行核心算法
      cpu_serial_kernel(iter, [generator]() -> scalar_t {
        // 创建完整 64 位整数范围的均匀整数分布
        uniform_int_full_range_distribution<scalar_t> random;
        // 返回生成的随机数
        return random(generator);
      });
    } else {
      // 如果不是支持的类型，抛出错误信息
      TORCH_CHECK(false, "random_full_64_bits_range_kernel_cpu handles only int64, double, float and bfloat16");
    }
  });
}

// RandomFromToKernel 结构体：根据不同的随机数生成器类型调用相应的随机数生成函数
template<typename RNG>
struct RandomFromToKernel {
  // 调用函数：生成指定范围内的随机数
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen) {
    // 调用 random_from_to_kernel 函数生成指定范围内的随机数
    random_from_to_kernel(iter, range, base, check_generator<RNG>(gen));
  }
  // 调用函数：生成完整 64 位整数范围内的随机数
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    // 调用 random_full_64_bits_range_kernel 函数生成完整 64 位整数范围内的随机数
    random_full_64_bits_range_kernel(iter, check_generator<RNG>(gen));
  }
};

// 模板函数：生成随机数的核心函数
template<typename RNG>
void random_kernel(TensorIteratorBase& iter, RNG generator) {
  // 使用互斥锁保护随机数生成器
  std::lock_guard<std::mutex> lock(generator->mutex_);
  // 派发函数，根据数据类型生成随机数
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "random_kernel_cpu", [&] {
    // 在 CPU 上串行执行核心算法
    cpu_serial_kernel(iter, [generator]() -> scalar_t {
      // 创建均匀整数分布
      uniform_int_distribution<scalar_t> random;
      // 返回生成的随机数
      return random(generator);
    });
  });
}

// 结构体结束标记
struct RandomKernel {
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    // 调用 random_kernel 函数处理迭代器和随机数生成器
    random_kernel(iter, check_generator<RNG>(gen));
  }
};

// ==================================================== Normal ========================================================

#ifdef CPU_CAPABILITY_AVX2
static void normal_fill_16_AVX2(float *data,
                         const __m256* two_pi,
                         const __m256* one,
                         const __m256* minus_two,
                         const __m256* mean,
                         const __m256* std_v) {
  // 计算 AVX2 版本的 normal_fill_16 函数，填充前16个元素的正态分布数据
  const __m256 u1 = _mm256_sub_ps(*one, _mm256_loadu_ps(data));
  const __m256 u2 = _mm256_loadu_ps(data + 8);
  // 使用 avx_mathfun.h 中的 sincos256_ps 和 log256_ps 函数
  const __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(*minus_two, log256_ps(u1)));
  const __m256 theta = _mm256_mul_ps(*two_pi, u2);
  __m256 sintheta, costheta;
  // 计算 theta 的正弦和余弦值
  sincos256_ps(theta, &sintheta, &costheta);
  // 计算正态分布的两个分量
  const __m256 n1 = _mm256_mul_ps(radius, costheta);
  const __m256 n2 = _mm256_mul_ps(radius, sintheta);
  // 存储计算结果
  _mm256_storeu_ps(data, _mm256_fmadd_ps(n1, *std_v, *mean));
  _mm256_storeu_ps(data + 8, _mm256_fmadd_ps(n2, *std_v, *mean));
}

template<typename RNG>
void normal_fill_AVX2(const TensorBase &self, const float mean, const float std, RNG generator) {
  float *data = self.data_ptr<float>();
  auto size = self.numel();
  std::lock_guard<std::mutex> lock(generator->mutex_);
  // 使用 AVX2 指令集计算正态分布
  for (const auto i : c10::irange(size)) {
    at::uniform_real_distribution<float> uniform(0, 1);
    data[i] = uniform(generator);
  }
  const __m256 two_pi = _mm256_set1_ps(2.0f * c10::pi<double>);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_two = _mm256_set1_ps(-2.0f);
  const __m256 mean_v = _mm256_set1_ps(mean);
  const __m256 std_v = _mm256_set1_ps(std);

  // 以 16 个元素为一组进行 AVX2 计算
  for (int64_t i = 0; i < size - 15; i += 16) {
    normal_fill_16_AVX2(data + i, &two_pi, &one, &minus_two, &mean_v, &std_v);
  }

  // 如果剩余不足16个元素，则重新计算最后的16个元素
  if (size % 16 != 0) {
    // 调整数据指针位置到剩余的数据起始点
    data = data + size - 16;
    for (const auto i : c10::irange(16)) {
      at::uniform_real_distribution<float> uniform(0, 1);
      data[i] = uniform(generator);
    }
    // 使用 AVX2 计算最后的16个元素的正态分布
    normal_fill_16_AVX2(data, &two_pi, &one, &minus_two, &mean_v, &std_v);
  }
}
#endif

template <typename scalar_t>
static void normal_fill_16(scalar_t *data, const scalar_t mean, const scalar_t std) {
  // 使用普通的循环计算正态分布的16个元素
  for (const auto j : c10::irange(8)) {
    const scalar_t u1 = 1 - data[j]; // [0, 1) -> (0, 1] for log.
    const scalar_t u2 = data[j + 8];
    const scalar_t radius = std::sqrt(-2 * std::log(u1));
    const scalar_t theta = 2.0f * c10::pi<double> * u2;
    data[j] = radius * std::cos(theta) * std + mean;
    data[j + 8] = radius * std::sin(theta) * std + mean;
  }
}

#if defined(__VSX__)  || defined(CPU_CAPABILITY_VSX)
#ifdef VSX
// 使用 VSX 指令集填充正态分布数据到 Tensor 中，参数包括数据指针、均值、标准差和随机数生成器
template <typename scalar_t, typename RNG>
void normal_fill_VSX(const TensorBase &self, const scalar_t mean, const scalar_t std, RNG generator) {
  // 获取数据指针并转换为 float*
  float *data = self.data_ptr<float>();
  // 获取 Tensor 元素个数
  auto size = self.numel();
  // 使用互斥锁保护随机数生成器
  std::lock_guard<std::mutex> lock(generator->mutex_);
  // 遍历 Tensor 中的每个元素
  for (const auto i : c10::irange(size)) {
    // 创建指定范围的均匀分布
    at::uniform_real_distribution<scalar_t> uniform(0, 1);
    // 使用均匀分布生成随机数并填充到数据中
    data[i] = uniform(generator);
  }

  // 定义 SIMD 向量类型
  using Vec = Vectorized<float>;
  // 初始化常量向量
  const Vec two_pi = Vec(2.0f * c10::pi<double>);
  const Vec one = Vec(1.0f);
  const Vec minus_two = Vec(-2.0f);
  const Vec var_vec = Vec(std);
  const Vec mean_vec = Vec(mean);

  // 使用 SIMD 填充正态分布数据，每次处理 16 个元素
  for (int64_t i = 0; i < size - 15; i += 16) {
    // 检查 SIMD 向量大小是否为 8
    if(Vec::size()==8) {
      // 调用填充 16 个元素的 VSX 版本函数
      normal_fill_16_VSX(data + i, two_pi, one, minus_two, mean_vec, var_vec);
    }
    else{
      // 否则调用普通版本函数填充正态分布数据
      normal_fill_16<scalar_t>(data + i, mean, std);
    }
  }

  // 处理剩余不足 16 个元素的情况
  if (size % 16 != 0) {
    // 重新定位数据指针到最后的 16 个元素处
    data = data + size - 16;
    // 再次遍历最后的 16 个元素
    for (const auto i : c10::irange(16)) {
      // 使用均匀分布生成随机数并填充到数据中
      at::uniform_real_distribution<scalar_t> uniform(0, 1);
      data[i] = uniform(generator);
    }
    // 检查 SIMD 向量大小是否为 8 并调用相应的填充函数
    if(Vec::size()==8){
      normal_fill_16_VSX(data, two_pi, one, minus_two, mean_vec, var_vec);
    }
    else{
      normal_fill_16<scalar_t>(data, mean, std);
    }
  }
}
#endif // VSX

// 普通版本的正态分布填充函数，适用于不支持 SIMD 指令集的平台
template <typename scalar_t, typename RNG>
void normal_fill(const TensorBase &self, const scalar_t mean, const scalar_t std, RNG generator) {
  // 获取数据指针并转换为 scalar_t*
  scalar_t *data = self.data_ptr<scalar_t>();
  // 获取 Tensor 元素个数
  auto size = self.numel();
  // 使用互斥锁保护随机数生成器
  std::lock_guard<std::mutex> lock(generator->mutex_);
  // 遍历 Tensor 中的每个元素
  for (const auto i : c10::irange(size)) {
    // 创建指定范围的均匀分布
    at::uniform_real_distribution<scalar_t> uniform(0, 1);
    // 使用均匀分布生成随机数并填充到数据中
    data[i] = uniform(generator);
  }

  // 使用普通版本的填充函数，每次处理 16 个元素
  for (int64_t i = 0; i < size - 15; i += 16) {
    normal_fill_16<scalar_t>(data + i, mean, std);
  }

  // 处理剩余不足 16 个元素的情况
  if (size % 16 != 0) {
    // 重新定位数据指针到最后的 16 个元素处
    data = data + size - 16;
    // 再次遍历最后的 16 个元素
    for (const auto i : c10::irange(16)) {
      // 使用均匀分布生成随机数并填充到数据中
      at::uniform_real_distribution<scalar_t> uniform(0, 1);
      data[i] = uniform(generator);
    }
    // 调用普通版本的填充函数填充正态分布数据
    normal_fill_16<scalar_t>(data, mean, std);
  }
}
#elif defined(__VSX__)  || defined(CPU_CAPABILITY_VSX)
    // 如果定义了__VSX__或者CPU_CAPABILITY_VSX，则使用VSX指令集版本的normal_fill函数填充张量
    normal_fill_VSX(self, static_cast<float>(mean), static_cast<float>(std), generator);
#else
    // 否则，使用通用版本的normal_fill函数填充张量
    normal_fill(self, static_cast<float>(mean), static_cast<float>(std), generator);
#endif
  } else {
    // 对于非特定条件的情况，使用AT_DISPATCH_FLOATING_TYPES_AND2宏来处理浮点类型和可能的kHalf、kBFloat16类型
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "normal_kernel_cpu", [&] {
      // 如果张量大小大于等于16并且是连续的，则调用优化后的normal_fill函数填充张量
      if (size >= 16 && self.is_contiguous()) {
        normal_fill<scalar_t>(self, static_cast<scalar_t>(mean), static_cast<scalar_t>(std), generator);
      } else {
        // 否则，创建一个TensorIterator用于无操作数操作，并使用cpu_serial_kernel函数进行计算
        auto iter = TensorIterator::borrowing_nullary_op(self);
        // 使用互斥锁保护生成器
        std::lock_guard<std::mutex> lock(generator->mutex_);
        // 在cpu_serial_kernel中使用lambda函数生成正态分布的随机数，并填充到张量中
        cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
          at::normal_distribution<double> normal(mean, std);
          return static_cast<scalar_t>(normal(generator));
        });
      }
    });
  }
}

template<typename RNG>
struct NormalKernel {
  // NormalKernel结构体，用于封装normal_kernel函数，用于处理正态分布的随机数填充到张量中
  void operator()(Tensor& self, double mean, double std, std::optional<Generator> gen) {
    normal_kernel(self, mean, std, check_generator<RNG>(gen));
  }
};

// ==================================================== Uniform =======================================================

template<typename RNG>
void uniform_kernel(TensorIteratorBase& iter, double from_, double to_, RNG generator) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "uniform_kernel_cpu", [&]() {
    // 使用互斥锁保护生成器
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    // 创建均匀分布的实例uniform，并使用cpu_serial_kernel函数填充张量
    at::uniform_real_distribution<scalar_t> uniform(from, to);
    cpu_serial_kernel(iter, [&uniform, generator]() -> scalar_t {
      return static_cast<scalar_t>(uniform(generator));
    });
  });
}

template<typename RNG>
struct UniformKernel {
  // UniformKernel结构体，封装uniform_kernel函数，用于处理均匀分布的随机数填充到张量中
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
    uniform_kernel(iter, from, to, check_generator<RNG>(gen));
  }
};

// ==================================================== Cauchy ========================================================

template<typename RNG>
void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, RNG generator) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "cauchy_cpu", [&]() {
    // 使用互斥锁保护生成器
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 创建柯西分布的实例cauchy，并使用cpu_serial_kernel函数填充张量
    at::cauchy_distribution<double> cauchy(median, sigma);
    cpu_serial_kernel(iter, [&cauchy, generator]() -> scalar_t {
      return static_cast<scalar_t>(cauchy(generator));
    });
  });
}

template<typename RNG>
struct CauchyKernel {
  // CauchyKernel结构体，封装cauchy_kernel函数，用于处理柯西分布的随机数填充到张量中
  void operator()(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
    cauchy_kernel(iter, median, sigma, check_generator<RNG>(gen));
  }
};

// ================================================== LogNormal =======================================================
// 使用 TensorIteratorBase 迭代器和给定的均值和标准差生成服从对数正态分布的随机数，并执行 CPU 上的计算核心函数
void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, RNG generator) {
  // 派发宏，允许处理所有浮点类型、半精度浮点和 BFloat16 类型，并命名为 "log_normal_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_cpu", [&]() {
    // 获取随机数生成器的互斥锁，确保线程安全
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 创建对数正态分布对象，使用给定的均值和标准差
    at::lognormal_distribution<double> logNormal(mean, std);
    // 调用串行 CPU 核心函数，生成并返回一个标量值，类型为当前迭代器的数据类型
    cpu_serial_kernel(iter, [&logNormal, generator]() -> scalar_t {
      return static_cast<scalar_t>(logNormal(generator));
    });
  });
}

// RNG 模板结构体，用于封装对数正态分布随机数生成器的调用
template<typename RNG>
struct LogNormalKernel {
  // 运算符重载，调用 log_normal_kernel 函数来执行对数正态分布随机数的生成和计算
  void operator()(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
    // 调用 log_normal_kernel 函数，传递给定的迭代器、均值、标准差和生成器
    log_normal_kernel(iter, mean, std, check_generator<RNG>(gen));
  }
};

// =================================================== Geometric ======================================================

// 根据给定的几何分布概率参数 p 和随机数生成器生成随机数，并执行 CPU 上的计算核心函数
template<typename RNG>
void geometric_kernel(TensorIteratorBase& iter, double p, RNG generator) {
  // 派发宏，允许处理所有数据类型，并命名为 "geometric_cpu"
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "geometric_cpu", [&]() {
    // 获取随机数生成器的互斥锁，确保线程安全
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 创建几何分布对象，使用给定的概率参数 p
    at::geometric_distribution<double> geometric(p);
    // 调用串行 CPU 核心函数，生成并返回一个标量值，类型为当前迭代器的数据类型
    cpu_serial_kernel(iter, [&geometric, generator]() -> scalar_t {
      return static_cast<scalar_t>(geometric(generator));
    });
  });
}

// RNG 模板结构体，用于封装几何分布随机数生成器的调用
template<typename RNG>
struct GeometricKernel {
  // 运算符重载，调用 geometric_kernel 函数来执行几何分布随机数的生成和计算
  void operator()(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
    // 调用 geometric_kernel 函数，传递给定的迭代器、概率参数 p 和生成器
    geometric_kernel(iter, p, check_generator<RNG>(gen));
  }
};

// ================================================== Exponential =====================================================

// 根据给定的指数分布参数 lambda 和随机数生成器生成随机数，并执行 CPU 上的计算核心函数
template<typename RNG>
void exponential_kernel(TensorIteratorBase& iter, double lambda, RNG generator) {
  // 检查数据类型是否为浮点类型，指数分布要求数据类型为浮点数
  TORCH_CHECK(isFloatingType(iter.dtype()), "Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ", iter.dtype());
  // 派发宏，允许处理所有浮点类型、半精度浮点和 BFloat16 类型，并命名为 "exponential_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "exponential_cpu", [&]() {
    // 获取随机数生成器的互斥锁，确保线程安全
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 创建指数分布对象，使用给定的 lambda 参数
    at::exponential_distribution<double> exponential(lambda);
    // 调用串行 CPU 核心函数，生成并返回一个标量值，类型为当前迭代器的数据类型
    cpu_serial_kernel(iter, [&exponential, generator]() -> scalar_t {
      return static_cast<scalar_t>(exponential(generator));
    });
  });
}

// RNG 模板结构体，用于封装指数分布随机数生成器的调用
template<typename RNG>
struct ExponentialKernel {
  // 运算符重载，调用 exponential_kernel 函数来执行指数分布随机数的生成和计算
  void operator()(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
    // 调用 exponential_kernel 函数，传递给定的迭代器、lambda 参数和生成器
    exponential_kernel(iter, lambda, check_generator<RNG>(gen));
  }
};

// ================================================== Bernoulli =======================================================

// 根据给定的伯努利分布参数 p 和随机数生成器生成随机数，并执行 CPU 上的计算核心函数
template<typename RNG>
void bernoulli_kernel(const TensorBase &self, const TensorBase &p_, RNG generator) {
  // 派发宏，允许处理所有数据类型，包括布尔型、半精度浮点和 BFloat16 类型，并命名为 "bernoulli_tensor_cpu_self_"
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
  self.scalar_type(), "bernoulli_tensor_cpu_self_", [&] {
    // 获取随机数生成器的互斥锁，确保线程安全
    std::lock_guard<std::mutex> lock(generator->mutex_);
    `
        # 定义 self_t 为 scalar_t 类型的别名
        using self_t = scalar_t;
        # 将 p_ 张量转移到 CPU 上
        auto p_cpu = p_.to(kCPU);
        # 在原地扩展 self 张量
        auto p = expand_inplace(self, p_cpu);
        # 配置张量迭代器，设置输出为 self 张量，常量输入为 p，不检查所有张量是否具有相同的数据类型，构建迭代器
        auto iter = TensorIteratorConfig()
            .add_output(self)
            .add_const_input(*p)
            .check_all_same_dtype(false)
            .build();
        # 如果 p 的数据类型是双精度浮点数
        if (p->scalar_type() == kDouble) {
          # 使用 CPU 序列化内核，对每个 p_val 执行 Bernoulli 分布采样，并返回 self_t 类型的值
          cpu_serial_kernel(iter, [&](const double p_val) -> self_t {
            at::bernoulli_distribution<double> bernoulli(p_val);
            return static_cast<self_t>(bernoulli(generator));
          });
        } else {
          # 否则，使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，遍历浮点数类型（除了 BFloat16 和 Half 之外的所有浮点数类型）
          AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half,
          p->scalar_type(), "bernoulli_tensor_cpu_p_", [&] {
            # 定义 p_t 为 scalar_t 类型的别名
            using p_t = scalar_t;
            # 使用 CPU 序列化内核，对每个 p_val 执行 Bernoulli 分布采样，并返回 self_t 类型的值
            cpu_serial_kernel(iter, [&](const p_t p_val) -> self_t {
              at::bernoulli_distribution<float> bernoulli(p_val);
              return static_cast<self_t>(bernoulli(generator));
            });
          });
        }
      });
}

// 定义模板函数 `bernoulli_kernel`，接受泛型 RNG 类型、TensorBase 类型的 self 引用、double 类型的 p 和 RNG 类型的 generator
template<typename RNG>
void bernoulli_kernel(const TensorBase &self, double p, RNG generator) {
  // 使用宏 `AT_DISPATCH_ALL_TYPES_AND3` 分发所有数据类型和额外的三种数据类型，包括 Bool、BFloat16 和 Half
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
    self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
      // 查看 “Note [Acquire lock when using random generators]” 注释
      // 使用互斥锁锁定随机生成器的互斥量
      std::lock_guard<std::mutex> lock(generator->mutex_);
      // 借用 Nullary 操作创建张量迭代器
      auto iter = TensorIterator::borrowing_nullary_op(self);
      // 调用 CPU 串行内核函数，传入 lambda 函数，返回类型为 scalar_t
      cpu_serial_kernel(iter, [p, generator]() -> scalar_t {
        // 创建 Bernoulli 分布对象，概率为 p
        at::bernoulli_distribution<double> bernoulli(p);
        // 使用 generator 生成伯努利随机数，并转换为 scalar_t 类型返回
        return static_cast<scalar_t>(bernoulli(generator));
      });
    });
}

// 定义模板结构体 `BernoulliKernel`
template<typename RNG>
struct BernoulliKernel {
  // 重载操作符 ()，接受 TensorBase 类型的 self 引用、double 类型的 p 和可选的 Generator 对象 gen
  void operator()(const TensorBase &self, double p, std::optional<Generator> gen) {
    // 调用 bernoulli_kernel 函数，传入 self、p 和检查后的 RNG 类型的 generator
    bernoulli_kernel(self, p, check_generator<RNG>(gen));
  }
  // 重载操作符 ()，接受两个 TensorBase 类型的引用 self 和 p_，以及可选的 Generator 对象 gen
  void operator()(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen) {
    // 调用 bernoulli_kernel 函数，传入 self、p_ 和检查后的 RNG 类型的 generator
    bernoulli_kernel(self, p_, check_generator<RNG>(gen));
  }
};

}}}}}
```