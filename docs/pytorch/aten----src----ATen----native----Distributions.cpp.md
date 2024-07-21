# `.\pytorch\aten\src\ATen\native\Distributions.cpp`

```py
// 定义编译时选项，仅包含断言方法的运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

// 包含 ATen 库中的特定模块头文件
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/cpu/Loops.h>

// 根据编译选项选择是否包含特定的运算符头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/bernoulli_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/cauchy_native.h>
#include <ATen/ops/div.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exponential_native.h>
#include <ATen/ops/geometric_native.h>
#include <ATen/ops/log_normal_native.h>
#include <ATen/ops/multinomial_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/poisson_native.h>
#include <ATen/ops/random_native.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/uniform_native.h>
#include <ATen/ops/zeros.h>
#endif

// 包含 C++ 标准库的头文件
#include <functional>
#include <type_traits>
#include <utility>
#include <assert.h>  // NOLINTNEXTLINE(modernize-deprecated-headers)
#include <float.h>   // NOLINTNEXTLINE(modernize-deprecated-headers)

// 使用匿名命名空间定义本文件私有的部分
namespace {

/*
 * This section is a counterpart to Distributions.cu
 *
 */

// 函数 `sample_poisson` 来自于 Numpy 的 distributions.c 实现
// 这部分代码使用了 MIT 许可证，下面是版权声明的一部分
/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
int64_t sample_poisson(double lambda, at::CPUGeneratorImpl* generator) {
  // 检查泊松率是否非负
  TORCH_CHECK(lambda >= 0, "invalid Poisson rate, expected rate to be non-negative");

  // 创建一个在[0.0, 1.0]范围内的均匀分布对象
  at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);

  // 如果 lambda >= 10，使用变换拒绝法 (Hoermann, 1993)
  if (lambda >= 10) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t k;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double U, V, a, b, invalpha, vr, us;

    double slam = std::sqrt(lambda);
    double loglam = std::log(lambda);
    b = 0.931 + 2.53 * slam;
    a = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr = 0.9277 - 3.6224 / (b - 2);

    // 使用变换拒绝法计算泊松分布的样本
    while (true) {
      U = standard_uniform(generator) - 0.5;
      V = standard_uniform(generator);
      us = 0.5 - std::fabs(U);
      k = (int64_t)std::floor((2 * a / us + b) * U + lambda + 0.43);
      if ((us >= 0.07) && (V <= vr)) {
        return k;
      }
      if ((k < 0) || ((us < 0.013) && (V > us))) {
        continue;
      }
      if ((std::log(V) + std::log(invalpha) - std::log(a / (us * us) + b)) <=
          (-lambda + k * loglam - std::lgamma((double)k + 1))) {
        return k;
      }
    }
  } else if (lambda == 0) {
    // 如果 lambda == 0，直接返回0
    return 0;
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t X;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double prod, U, enlam;

    enlam = std::exp(-lambda);
    X = 0;
    prod = 1.0;

    // 使用指数分布的方法计算 lambda < 10 时的泊松分布样本
    while (true) {
      U = standard_uniform(generator);
      prod *= U;
      if (prod > enlam) {
        X += 1;
      } else {
        return X;
      }
    }
  }
}

} // namespace

namespace at::native {

// ==================================================== Bernoulli =====================================================

// BernoulliTensorStub 结构体定义
template<typename RNG>
struct BernoulliStub {
  // 运算符重载，对 Tensor 进行 Bernoulli 操作
  void operator()(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
    bernoulli_tensor_stub(self.device().type(), self, p_, gen);
  }

  // 运算符重载，对 Tensor 进行 Bernoulli 操作（标量版本）
  void operator()(Tensor& self, double p, std::optional<Generator> gen) {
    bernoulli_scalar_stub(self.device().type(), self, p, gen);
  }
};

// 生成并返回一个与输入张量相同形状的 Bernoulli 分布的张量
Tensor bernoulli(const Tensor& self, std::optional<Generator> gen) {
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result.bernoulli_(self, std::move(gen));
  return result;
}
// 从给定的张量 self 生成一个服从伯努利分布的张量，概率为 p，使用可选的随机数生成器 gen
Tensor bernoulli(const Tensor& self, double p, std::optional<Generator> gen) {
  // 根据 self 的形状和类型创建一个新的张量 result
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 在 result 上生成伯努利分布的随机数，并将结果保存在 result 中
  result.bernoulli_(p, std::move(gen));
  // 返回生成的结果张量 result
  return result;
}

// 在给定的输出张量 result 上生成服从伯努利分布的随机数，概率为 self，使用可选的随机数生成器 gen
Tensor& bernoulli_out(const Tensor& self, std::optional<Generator> gen, Tensor& result) {
  // 调用底层实现函数，生成伯努利分布的随机数，并将结果存储在 result 中
  return at::native::templates::bernoulli_out_impl<BernoulliStub, Generator>(result, self, std::move(gen));
}

// 在给定的张量 self 上就地生成服从伯努利分布的随机数，概率为 p_，使用可选的随机数生成器 gen
Tensor& bernoulli_(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  // 调用底层实现函数，就地生成伯努利分布的随机数，并将结果存储在 self 中
  return at::native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p_, std::move(gen));
}

// 在给定的张量 self 上就地生成服从伯努利分布的随机数，概率为 p，使用可选的随机数生成器 gen
Tensor& bernoulli_(Tensor& self, double p, std::optional<Generator> gen) {
  // 调用底层实现函数，就地生成伯努利分布的随机数，并将结果存储在 self 中
  return at::native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p, std::move(gen));
}

// ================================================== LogNormal =======================================================

// LogNormal 分布的随机数生成器结构体模板，使用指定的 RNG
template<typename RNG>
struct LogNormalStub {
  // 操作符重载函数，根据给定的均值 mean、标准差 std 和可选的随机数生成器 gen，在迭代器 iter 上生成 LogNormal 分布的随机数
  void operator()(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
    // 调用底层实现函数，根据设备类型和参数在 iter 上生成 LogNormal 分布的随机数
    log_normal_stub(iter.device_type(), iter, mean, std, gen);
  }
};

// 在给定的张量 self 上就地生成服从 LogNormal 分布的随机数，均值为 mean，标准差为 std，使用可选的随机数生成器 gen
Tensor& log_normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  // 调用底层实现函数，就地生成 LogNormal 分布的随机数，并将结果存储在 self 中
  return at::native::templates::log_normal_impl_<LogNormalStub, Generator>(self, mean, std, std::move(gen));
}

// ==================================================== Cauchy ========================================================

// Cauchy 分布的随机数生成器结构体模板，使用指定的 RNG
template<typename RNG>
struct CauchyStub {
  // 操作符重载函数，根据给定的中位数 median、标准差 sigma 和可选的随机数生成器 gen，在迭代器 iter 上生成 Cauchy 分布的随机数
  void operator()(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
    // 调用底层实现函数，根据设备类型和参数在 iter 上生成 Cauchy 分布的随机数
    cauchy_stub(iter.device_type(), iter, median, sigma, gen);
  }
};

// 在给定的张量 self 上就地生成服从 Cauchy 分布的随机数，中位数为 median，标准差为 sigma，使用可选的随机数生成器 gen
Tensor& cauchy_(Tensor& self, double median, double sigma, std::optional<Generator> gen) {
  // 调用底层实现函数，就地生成 Cauchy 分布的随机数，并将结果存储在 self 中
  return at::native::templates::cauchy_impl_<CauchyStub, Generator>(self, median, sigma, std::move(gen));
}

// ================================================== Exponential =====================================================

// Exponential 分布的随机数生成器结构体模板，使用指定的 RNG
template<typename RNG>
struct ExponentialStub {
  // 操作符重载函数，根据给定的 lambda 参数和可选的随机数生成器 gen，在迭代器 iter 上生成 Exponential 分布的随机数
  void operator()(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
    // 调用底层实现函数，根据设备类型和参数在 iter 上生成 Exponential 分布的随机数
    exponential_stub(iter.device_type(), iter, lambda, gen);
  }
};

// 在给定的张量 self 上就地生成服从 Exponential 分布的随机数，参数 lambda，使用可选的随机数生成器 gen
Tensor& exponential_(Tensor& self, double lambda, std::optional<Generator> gen) {
  // 调用底层实现函数，就地生成 Exponential 分布的随机数，并将结果存储在 self 中
  return at::native::templates::exponential_impl_<ExponentialStub, Generator>(self, lambda, std::move(gen));
}

// =================================================== Geometric ======================================================

// Geometric 分布的随机数生成器结构体模板，使用指定的 RNG
template<typename RNG>
struct GeometricStub {
  // 操作符重载函数，根据给定的概率 p 和可选的随机数生成器 gen，在迭代器 iter 上生成 Geometric 分布的随机数
  void operator()(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
    // 调用底层实现函数，根据设备类型和参数在 iter 上生成 Geometric 分布的随机数
    geometric_stub(iter.device_type(), iter, p, gen);
  }
};

// 在给定的张量 self 上就地生成服从 Geometric 分布的随机数，概率为 p，使用可选的随机数生成器 gen
Tensor& geometric_(Tensor& self, double p, std::optional<Generator> gen) {
  // 调用底层实现函数，就地生成 Geometric 分布的随机数，并将结果存储在 self 中
  return at::native::templates::geometric_impl_<GeometricStub, Generator>(self, p, std::move(gen));
}

// ==================================================== Uniform =======================================================

// Uniform 分布的随机数生成器结构体模板，使用指定的 RNG
// （未完整提供，可能需要根据具体实现继续补充）
// 定义模板结构体 UniformStub，用于处理均匀分布生成器
template<typename RNG>
struct UniformStub {
  // 定义函数调用操作符，接受迭代器对象 iter、范围 [from, to]、生成器 gen（可选）
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
    // 调用 uniform_stub 函数，根据迭代器的设备类型，对迭代器进行均匀分布采样操作
    uniform_stub(iter.device_type(), iter, from, to, gen);
  }
};

// 定义模板结构体 UniformMeta，仅占位用途，无实质操作
template<typename RNG>
struct UniformMeta {
  // No-op! 仅定义函数调用操作符，但未实现任何功能
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  }
};

// 定义 uniform_ 函数，使用 UniformStub 模板实现的均匀分布操作
Tensor& uniform_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  // 调用 native 命名空间下的 templates 模块，调用 uniform_impl_ 函数进行实际的均匀分布操作
  return at::native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, std::move(gen));
}

// 定义 uniform_meta_ 函数，使用 UniformMeta 模板，但并未实现具体功能
Tensor& uniform_meta_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  // 调用 native 命名空间下的 templates 模块，调用 uniform_impl_ 函数，但此处不执行任何实际操作
  return at::native::templates::uniform_impl_<UniformMeta, Generator>(self, from, to, std::move(gen));
}

// ==================================================== Normal ========================================================

// 定义模板结构体 NormalStub，用于处理正态分布生成器
template<typename RNG>
struct NormalStub {
  // 定义函数调用操作符，接受张量 self、均值 mean、标准差 std、生成器 gen（可选）
  void operator()(Tensor& self, double mean, double std, std::optional<Generator> gen) {
    // 调用 normal_stub 函数，根据张量的设备类型，对张量进行正态分布采样操作
    normal_stub(self.device().type(), self, mean, std, gen);
  }
};

// 定义模板结构体 NormalMeta，仅占位用途，无实质操作
template<typename RNG>
struct NormalMeta {
  // No-op! 仅定义函数调用操作符，但未实现任何功能
  void operator()(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  }
};

// 定义 normal_ 函数，使用 NormalStub 模板实现的正态分布操作（原地修改）
Tensor& normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_impl_ 函数进行实际的正态分布操作
  return at::native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, std::move(gen));
}

// 定义 normal_meta_ 函数，使用 NormalMeta 模板，但并未实现具体功能
Tensor& normal_meta_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_impl_ 函数，但此处不执行任何实际操作
  return at::native::templates::normal_impl_<NormalMeta, Generator>(self, mean, std, std::move(gen));
}

// 定义 normal_out 函数，将正态分布生成结果写入指定输出张量 output（mean 为张量，std 为浮点数）
Tensor& normal_out(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& output) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_out_impl 函数，使用 NormalStub 模板进行操作
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, std::move(gen));
}

// 定义 normal_out_meta 函数，将正态分布生成结果写入指定输出张量 output（mean 为张量，std 为浮点数），但并未实现具体功能
Tensor& normal_out_meta(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& output) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_out_impl 函数，使用 NormalMeta 模板，但此处不执行任何实际操作
  return at::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, std::move(gen));
}

// 定义 normal_out 函数，将正态分布生成结果写入指定输出张量 output（mean 为浮点数，std 为张量）
Tensor& normal_out(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_out_impl 函数，使用 NormalStub 模板进行操作
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, std::move(gen));
}

// 定义 normal_out_meta 函数，将正态分布生成结果写入指定输出张量 output（mean 为浮点数，std 为张量），但并未实现具体功能
Tensor& normal_out_meta(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_out_impl 函数，使用 NormalMeta 模板，但此处不执行任何实际操作
  return at::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, std::move(gen));
}

// 定义 normal_out 函数，将正态分布生成结果写入指定输出张量 output（mean、std 均为张量）
Tensor& normal_out(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_out_impl 函数，使用 NormalStub 模板进行操作
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, std::move(gen));
}

// 定义 normal_out_meta 函数，将正态分布生成结果写入指定输出张量 output（mean、std 均为张量），但并未实现具体功能
Tensor& normal_out_meta(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  // 调用 native 命名空间下的 templates 模块，调用 normal_out_impl 函数，使用 NormalMeta 模板，但此处不执行任何实际操作
  return at::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, std::move(gen));
}
// functional tensor float
Tensor normal(const Tensor& mean, double std, std::optional<Generator> gen) {
  // 使用 NormalStub 模板进行正态分布采样，返回采样结果张量
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, std::move(gen));
}

Tensor normal_meta(const Tensor& mean, double std, std::optional<Generator> gen) {
  // 使用 NormalMeta 模板进行正态分布采样，返回采样结果张量
  return at::native::templates::normal_impl<NormalMeta, Generator>(mean, std, std::move(gen));
}

// functional float tensor
Tensor normal(double mean, const Tensor& std, std::optional<Generator> gen) {
  // 使用 NormalStub 模板进行正态分布采样，返回采样结果张量
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, std::move(gen));
}

Tensor normal_meta(double mean, const Tensor& std, std::optional<Generator> gen) {
  // 使用 NormalMeta 模板进行正态分布采样，返回采样结果张量
  return at::native::templates::normal_impl<NormalMeta, Generator>(mean, std, std::move(gen));
}

// functional tensor tensor
Tensor normal(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  // 使用 NormalStub 模板进行正态分布采样，返回采样结果张量
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, std::move(gen));
}

Tensor normal_meta(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  // 使用 NormalMeta 模板进行正态分布采样，返回采样结果张量
  return at::native::templates::normal_impl<NormalMeta, Generator>(mean, std, std::move(gen));
}

// functional variant, only used by the functionalization pass.
Tensor normal_functional(const Tensor& self, double mean, double std, std::optional<at::Generator> generator) {
  // 对自身张量进行克隆操作，并在克隆的张量上应用正态分布采样
  return self.clone().normal_(mean, std, std::move(generator));
}

// ==================================================== Random ========================================================

template<typename RNG>
struct RandomStub {
  // 通过 RandomStub 结构体调用随机数生成函数，根据传入的生成器生成随机数填充迭代器
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_stub(iter.device_type(), iter, gen);
  }
};

Tensor& random_(Tensor& self, std::optional<Generator> gen) {
  // 使用 RandomStub 模板进行随机数生成操作，填充到自身张量中
  return at::native::templates::random_impl<RandomStub, Generator>(self, std::move(gen));
}

template<typename RNG>
struct RandomFromToStub {
  // 通过 RandomFromToStub 结构体调用范围随机数生成函数，根据传入的范围、起始点和生成器生成随机数填充迭代器
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t from, std::optional<Generator> gen) {
    random_from_to_stub(iter.device_type(), iter, range, from, gen);
  }
  // 通过 RandomFromToStub 结构体调用全 64 位范围随机数生成函数，根据传入的生成器生成随机数填充迭代器
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_full_64_bits_range_stub(iter.device_type(), iter, gen);
  }
};

Tensor& random_(Tensor& self, int64_t from, optional<int64_t> to, std::optional<Generator> gen) {
  // 使用 RandomFromToStub 模板进行范围随机数生成操作，填充到自身张量中
  return at::native::templates::random_from_to_impl<RandomFromToStub, Generator>(self, from, to, std::move(gen));
}

Tensor& random_(Tensor& self, int64_t to, std::optional<Generator> gen) {
  // 使用 random_ 函数的重载版本，起始点默认为 0 进行范围随机数生成操作
  return random_(self, 0, to, std::move(gen));
}

Tensor& random_meta_(Tensor& self, std::optional<Generator> gen) {
  // 对元数据张量进行随机数填充操作，没有进行错误检查
  return self;
}

Tensor& random_meta_(Tensor& self, int64_t from, optional<int64_t> to, std::optional<Generator> gen) {
  // 对元数据张量进行范围随机数填充操作，没有进行错误检查
  return self;
}

Tensor& random_meta_(Tensor& self, int64_t to, std::optional<Generator> gen) {
  // 对元数据张量进行范围随机数填充操作，没有进行错误检查
  return self;
}
// ====================================================================================================================

// 定义了计算标准伽马分布梯度的 CPU 版本函数
Tensor _standard_gamma_grad_cpu(const Tensor& self, const Tensor& output) {
  // 创建一个与输入张量相同大小和选项的空张量
  Tensor ret = at::empty(self.sizes(), self.options());
  // 配置张量迭代器，设置输出张量和输入张量
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(self)
    .add_input(output)
    .build();
  // 根据输入张量的数据类型分发到对应的浮点类型函数 "_standard_gamma_grad_cpu"
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "_standard_gamma_grad_cpu", [&] {
    // 调用 CPU 串行内核函数，计算每个元素的梯度
    cpu_serial_kernel(iter, [](scalar_t self_val, scalar_t output_val) -> scalar_t{
      return standard_gamma_grad_one<scalar_t, double>(self_val, output_val);
    });
  });
  // 返回计算后的梯度张量
  return ret;
}

// 定义了计算狄利克雷分布梯度的 CPU 版本函数
Tensor _dirichlet_grad_cpu(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  // 创建一个与输入张量 x 相同大小和选项的空张量
  Tensor ret = at::empty(x.sizes(), x.options());
  // 配置张量迭代器，设置输出张量和输入张量
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(x)
    .add_input(alpha)
    .add_input(total)
    .build();
  // 根据输入张量 x 的数据类型分发到对应的浮点类型函数 "_dirichlet_grad_cpu"
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "_dirichlet_grad_cpu", [&] {
    // 调用 CPU 串行内核函数，计算每个元素的梯度
    cpu_serial_kernel(iter, [](scalar_t x_val, scalar_t alpha_val, scalar_t total_val) -> scalar_t{
      return dirichlet_grad_one<scalar_t, double>(x_val, alpha_val, total_val);
    });
  });
  // 返回计算后的梯度张量
  return ret;
}

/*
 * This section is a counterpart to Distributions.cu
 */

// 定义了计算二项分布采样的 CPU 版本函数
Tensor _s_binomial_cpu(const Tensor& count, const Tensor& prob, std::optional<Generator> gen) {
  // 创建一个与输入张量 count 相同大小和选项的零张量
  Tensor ret = at::zeros(count.sizes(), count.options());
  // 配置张量迭代器，设置输出张量和输入张量
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(count)
    .add_input(prob)
    .build();
  // 根据输出张量 ret 的数据类型分发到对应的浮点类型函数 "binomial_cpu"
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "binomial_cpu", [&] {
    // 获取或使用默认的 CPU 随机数生成器
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // 查看 "Acquire lock when using random generators" 注释
    // 使用互斥锁保护随机数生成器访问
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 调用 CPU 串行内核函数，使用指定的随机数生成器计算每个元素的二项分布采样
    cpu_serial_kernel(iter, [generator](scalar_t count_val, scalar_t prob_val) -> scalar_t{
      // 定义 lambda 函数，生成标准均匀分布的随机数
      auto uniform_lambda = [generator] () {
        at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
        return standard_uniform(generator);
      };
      // 创建基于 lambda 函数的标准均匀分布的随机数采样器
      BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);
      // 使用二项分布采样函数生成样本，并将结果转换为目标标量类型
      auto sample = sample_binomial<scalar_t, double, decltype(uniform_lambda)>(count_val, prob_val, standard_uniform);
      return static_cast<scalar_t>(sample);
    });
  });
  // 返回计算后的二项分布采样张量
  return ret;
}

// 定义了计算泊松分布采样的 CPU 版本函数
Tensor _s_poisson_cpu(const Tensor& lambda, std::optional<Generator> gen) {
  // 创建一个与输入张量 lambda 相同大小和选项的零张量
  Tensor ret = at::zeros(lambda.sizes(), lambda.options());
  // 配置张量迭代器，设置输出张量和输入张量
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(lambda)
    .build();
  // 根据输出张量 ret 的数据类型分发到对应的浮点类型函数 "poisson_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, ret.scalar_type(), "poisson_cpu", [&] {
    // 获取或使用默认的 CPU 随机数生成器
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // 查看 "Acquire lock when using random generators" 注释
    // 使用互斥锁保护随机数生成器访问
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 调用 CPU 串行内核函数，使用指定的随机数生成器计算每个元素的泊松分布采样
    cpu_serial_kernel(iter, [](scalar_t lambda_val) -> scalar_t{
      // 调用泊松分布采样函数生成样本，并将结果转换为目标标量类型
      auto sample = sample_poisson<scalar_t, double>(lambda_val);
      return static_cast<scalar_t>(sample);
    });
  });
  // 返回计算后的泊松分布采样张量
  return ret;
}
    cpu_serial_kernel(iter, [generator](scalar_t lambda_val) -> scalar_t{
      return static_cast<scalar_t>(sample_poisson(static_cast<double>(lambda_val), generator));
    });
  });
  return ret;



    // 调用 cpu_serial_kernel 函数，在每个 iter 上执行指定的 lambda 函数
    cpu_serial_kernel(iter, [generator](scalar_t lambda_val) -> scalar_t{
      // 使用 sample_poisson 函数对 lambda_val 执行泊松分布采样，并转换结果为 scalar_t 类型后返回
      return static_cast<scalar_t>(sample_poisson(static_cast<double>(lambda_val), generator));
    });
  });
  // 返回最终结果 ret
  return ret;
}

// 定义一个函数 `_s_gamma_cpu`，计算 Gamma 分布的样本
Tensor _s_gamma_cpu(const Tensor& alpha, std::optional<Generator> gen) {
  // 创建一个与 alpha 相同形状和类型的零张量 ret
  Tensor ret = at::zeros(alpha.sizes(), alpha.options());
  // 配置张量迭代器，用于 alpha 张量的计算
  auto iter = TensorIteratorConfig()
    .add_output(ret)  // 将 ret 设置为输出张量
    .add_input(alpha)  // 将 alpha 设置为输入张量
    .build();
  // 根据 ret 的数据类型分发到相应的 CPU 函数 "gamma_cpu"
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "gamma_cpu", [&] {
    // 获取或者创建一个 CPU 随机数生成器
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // 查看注释 [使用随机生成器时要获取锁]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 调用 CPU 串行内核函数，对每个元素进行操作
    cpu_serial_kernel(iter, [generator](scalar_t alpha_val) -> scalar_t{
      // 定义均匀分布的 lambda 表达式
      auto uniform_lambda = [generator] () {
        at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
        return standard_uniform(generator);
      };
      // 创建基础的双精度随机数生成器，使用均匀分布 lambda 表达式
      BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

      // 定义正态分布的 lambda 表达式
      auto normal_lambda = [generator] () {
        at::normal_distribution<double> normal(0.0, 1.0);
        return normal(generator);
      };
      // 创建基础的双精度随机数生成器，使用正态分布 lambda 表达式
      BaseSampler<double, decltype(normal_lambda)> standard_normal(normal_lambda);

      // 使用 sample_gamma 函数生成 Gamma 分布样本
      auto sample = sample_gamma<scalar_t, double, decltype(uniform_lambda), decltype(normal_lambda)>(alpha_val, standard_uniform, standard_normal);
      // 返回样本，确保不低于 scalar_t 类型的最小值
      return std::max(std::numeric_limits<scalar_t>::min(), (scalar_t) sample);
    });
  });

  // 返回计算得到的 ret 张量
  return ret;
}

// 定义一个函数 `_s_dirichlet_cpu`，计算 Dirichlet 分布的样本
Tensor _s_dirichlet_cpu(const Tensor& alpha, std::optional<Generator> gen) {
  // 创建一个与 alpha 相同形状和类型的零张量 ret
  Tensor ret = at::zeros(alpha.sizes(), alpha.options());
  // 根据 ret 的数据类型分发到相应的 CPU 函数 "dirichlet"
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "dirichlet", [&] {
    // 创建一个与 alpha 相同形状和类型的双精度零张量 gamma
    Tensor gamma = at::zeros(alpha.sizes(), alpha.options().dtype(ScalarType::Double));
    // 获取或者创建一个 CPU 随机数生成器
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // 查看注释 [使用随机生成器时要获取锁]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    /* 通过将 alpha 张量转换为双精度以避免下溢，生成 gamma 样本。 */
    auto iter1 = TensorIteratorConfig()
      .add_output(gamma)  // 将 gamma 设置为输出张量
      .add_input(alpha)   // 将 alpha 设置为输入张量
      .check_all_same_dtype(false)  // 不检查所有张量是否具有相同的数据类型
      .build();
    // 调用 CPU 串行内核函数，对每个元素进行操作
    cpu_serial_kernel(iter1, [generator](scalar_t alpha_val) -> double{
      // 定义均匀分布的 lambda 表达式
      auto uniform_lambda = [generator] () {
        at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
        return standard_uniform(generator);
      };
      // 创建基础的双精度随机数生成器，使用均匀分布 lambda 表达式
      BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

      // 定义正态分布的 lambda 表达式
      auto normal_lambda = [generator] () {
        at::normal_distribution<double> normal(0.0, 1.0);
        return normal(generator);
      };
      // 创建基础的双精度随机数生成器，使用正态分布 lambda 表达式
      BaseSampler<double, decltype(normal_lambda)> standard_normal(normal_lambda);

      // 使用 sample_gamma 函数生成双精度 Gamma 分布样本
      auto sample = sample_gamma<double, double, decltype(uniform_lambda), decltype(normal_lambda)>
        (alpha_val, standard_uniform, standard_normal);
      // 返回样本，确保不低于双精度的最小值
      return std::max(std::numeric_limits<double>::min(), sample);
    });
    /* 标准化并将其转换回到 scalar_t 类型。 */
    // 计算 gamma 张量沿指定维度的和，扩展到与 alpha 相同的形状
    Tensor gamma_sum = gamma.sum(-1, true).expand(alpha.sizes());
    # 创建一个 Tensor 迭代器的配置对象，并进行链式调用配置
    auto iter2 = TensorIteratorConfig()
      .add_output(ret)  # 将 ret 添加为输出
      .add_input(gamma)  # 将 gamma 添加为输入
      .add_input(gamma_sum)  # 将 gamma_sum 添加为输入
      .check_all_same_dtype(false)  # 设置不要求所有输入输出的数据类型相同
      .build();  # 构建迭代器配置

    # 调用 CPU 串行内核函数 cpu_serial_kernel，传入迭代器和 Lambda 表达式
    cpu_serial_kernel(iter2, [](double gamma_val, double gamma_sum_val) -> scalar_t{
      auto ret_val = gamma_val / gamma_sum_val;  # 计算 gamma_val 与 gamma_sum_val 的比值
      auto min_val = std::numeric_limits<scalar_t>::min();  # 获取 scalar_t 类型的最小值
      auto max_val = std::nexttoward(static_cast<scalar_t>(1.0f), 0.0f);  # 获取紧接着 1.0f 的 scalar_t 类型值
      return std::min(max_val, std::max(min_val, static_cast<scalar_t>(ret_val)));  # 返回 ret_val 的修正值，确保在 min_val 和 max_val 之间
    });

  });

  # 返回 ret 变量
  return ret;
}

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);

// 多项式分布抽样函数，将结果写入预分配的结果张量中
Tensor& multinomial_out(const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    std::optional<Generator> gen,
    Tensor& result) {
  // 检查结果张量与输入张量是否在同一设备上
  TORCH_CHECK(
      result.device() == self.device(),
      "multinomial arguments must have the same device");
  // 检查输入张量维度是否在1到2之间
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  // 检查输入张量是否为浮点数类型
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  // 检查结果张量是否为长整型
  TORCH_CHECK(result.scalar_type() == ScalarType::Long,
      "multinomial expects Long tensor out, got: ", result.scalar_type());
  // 检查抽样数量是否大于0
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  // 获取概率分布张量最后一个维度的大小
  int64_t n_categories = self.size(-1);
  // 检查是否可以进行无放回抽样，或者抽样数量不超过概率分布张量最后一个维度的大小
  TORCH_CHECK(with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // 检查分类数是否小于等于 FLOAT32_MAX_CONSECUTIVE_INT
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  // 根据输入张量的维度调整结果张量的大小
  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  // 如果结果张量元素数量为0，则直接返回结果张量
  if (result.numel() == 0) {
    return result;
  }

  // 对于无放回抽样或只抽取一个样本的快速路径
  // 参考链接：https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
  if (!with_replacement || n_sample == 1) {
    // 对输入张量进行有效性检查
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(
        is_valid.to<bool>(),
        "probability tensor contains either `inf`, `nan` or element < 0");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool zero_prob_condition;
    // 如果输入张量是一维的，检查概率和是否为0
    if (self.dim() == 1){
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      // 如果输入张量是二维的，检查每行的概率和是否为0
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    // 如果概率和为0，抛出异常
    TORCH_CHECK(
        !zero_prob_condition,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    // 根据 Gumbel Softmax 算法生成指数分布 q
    // s = argmax( logp - log(-log(eps)) ) 其中 eps ~ U(0, 1)
    // 这里可以对公式应用指数函数，不会影响 argmax 或 topk 的结果。然后我们有
    // s = argmax( p / (-log(eps)) ) 其中 eps ~ U(0, 1).
    // 我们也可以通过以下简化公式
    // s = argmax( p / q ) 其中 q ~ Exp(1)
    Tensor q = at::empty_like(self).exponential_(1, std::move(gen));
    // 理论上，从指数分布生成0的概率为0。但是在 CUDA 中有保护措施以避免0，但是在 CPU 端
    // 从指数分布生成0的概率非常低
    // 使用 at::div_out 函数计算 self 与 q 的元素级别除法，结果存储在 q 中
    at::div_out(q, self, q);

    // 如果采样数 n_sample 为 1，则执行下列操作
    if (n_sample == 1) {
      // 使用 at::argmax_out 函数，在 q 的最后一个维度上计算每行的最大值索引，并保持维度不变
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      // 创建一个与 result 具有相同大小和设备类型的空 Tensor
      Tensor vals = at::empty(result.sizes(), self.options());
      // 使用 at::topk_out 函数，在 result 的基础上计算 q 的最大值及其索引，输出到 vals
      at::topk_out(vals, result, q, n_sample);
    }

    // 返回处理后的 result Tensor
    return result;
  }

  // 调用 multinomial_with_replacement_stub 函数进行多项分布采样
  // 采样结果存储在 result 中，使用 self 和指定的采样数 n_sample，使用 gen 生成器
  multinomial_with_replacement_stub(
      result.device().type(), result, self, n_sample, gen);
  
  // 返回多项分布采样的 result Tensor
  return result;
}

// 结束 at::native 命名空间的定义

Tensor multinomial(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    std::optional<Generator> gen) {
  // 创建一个空的张量 result，用于存储多项式采样的结果
  Tensor result = at::empty({0}, self.options().dtype(kLong));
  // 调用 native 命名空间中的 multinomial_out 函数，进行多项式采样，并将结果存储到 result 中
  native::multinomial_out(self, n_sample, with_replacement, std::move(gen), result);
  // 返回多项式采样的结果张量
  return result;
}

} // 结束命名空间 at::native
```