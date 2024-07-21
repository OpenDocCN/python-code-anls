# `.\pytorch\aten\src\ATen\native\cpu\DistributionKernels.cpp`

```py
// 定义宏以仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的必要头文件
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/cpu/DistributionTemplates.h>

#include <ATen/native/UnaryOps.h>

// 根据编译选项选择是否包含完整的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <cmath>  // 包含数学运算函数
#include <limits> // 包含数值极限常量
#include <type_traits> // 包含类型特性支持

// 如果支持 MKL，则包含相应的头文件
#if AT_MKL_ENABLED()
#include <mkl.h>
#include <cpuinfo.h>
#endif

// 定义 at::native 命名空间
namespace at::native {
namespace {

// 静态函数，实现 Cauchy 分布的计算
static void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  // 获取 CPU 生成器实例，如果未提供生成器则使用默认 CPU 生成器
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 实现函数计算 Cauchy 分布
  templates::cpu::cauchy_kernel(iter, median, sigma, generator);
}

// 实现 Bernoulli 分布的张量版本计算
void bernoulli_tensor_kernel(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen) {
  // 获取 CPU 生成器实例，如果未提供生成器则使用默认 CPU 生成器
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 实现函数计算 Bernoulli 分布
  templates::cpu::bernoulli_kernel(self, p_, generator);
}

// 如果不支持 MKL，则提供 Bernoulli 分布的标量版本计算实现
#if !AT_MKL_ENABLED()
void bernoulli_scalar_kernel_default(const TensorBase &self, double p, std::optional<Generator> gen) {
  // 获取 CPU 生成器实例，如果未提供生成器则使用默认 CPU 生成器
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 实现函数计算 Bernoulli 分布
  templates::cpu::bernoulli_kernel(self, p, generator);
}

// 定义 Bernoulli 分布的标量版本计算
void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  // 调用默认版本的标量计算 Bernoulli 分布
  bernoulli_scalar_kernel_default(self, p, gen);
}
// 如果支持 MKL，则提供 Bernoulli 分布的标量版本计算实现
#else
void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  // 获取 CPU 生成器实例，如果未提供生成器则使用默认 CPU 生成器
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  int64_t seed;
  {
    // 见注释 [使用随机生成器时获取锁]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    // 从生成器中获取随机数种子
    seed = generator->random();
  }
  int64_t n = self.numel();  // 获取张量的元素个数
  bool contig = self.is_contiguous();  // 检查张量是否连续存储

  // 针对所有类型及特定类型进行分派，执行 Bernoulli 分布的标量计算
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
  self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
    at::Tensor tmp_int_tensor;
    // 如果是 int 类型且张量是连续存储的，直接使用原始张量
    if (std::is_same<scalar_t, int>::value && contig) {
      tmp_int_tensor = self;
    } else {
      // 否则创建一个相同大小的空张量用于存储整数类型的数据
      tmp_int_tensor = at::empty(self.sizes(), self.options().dtype(at::kInt));
    }

    // 获取原始张量和整数类型临时张量的指针
    scalar_t *self_ptr = self.data_ptr<scalar_t>();
    int *sample_int_ptr = tmp_int_tensor.data_ptr<int>();
    // 定义了一个 lambda 函数 sample，用于对指定范围 [begin, end) 的数据进行抽样
    auto sample = [&](int64_t begin, int64_t end) {
      // 计算待抽样数据的长度
      int64_t len = end - begin;
      // 如果长度大于 0，则进行抽样操作
      if (len > 0) {
        // 创建随机数生成流 stream，并设置种子为 seed，使用 MCG31 算法
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed);
        // 将 stream 跳到指定位置 begin
        vslSkipAheadStream(stream, begin);
        // 使用 Bernoulli 分布生成随机数填充 sample_int_ptr + begin
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, len,
          sample_int_ptr + begin, p);
        // 删除生成流 stream
        vslDeleteStream(&stream);

        // 如果非标量类型并且数据是连续存储的，则进行向量化复制
        if (!std::is_same<scalar_t, int>::value && contig) {
          // self_seg 指向 self_ptr 的 begin 处
          scalar_t *self_seg = self_ptr + begin;
          // tmp_seg 指向 sample_int_ptr 的 begin 处
          int* tmp_seg = sample_int_ptr + begin;
          // 调用 vec::convert 将 tmp_seg 的数据转换为 scalar_t 类型，并存入 self_seg
          at::vec::convert<int, scalar_t>(tmp_seg, self_seg, len);
        }
      }
    };

    // 使用 parallel_for 并行执行 sample 函数，抽样范围为 [0, n)，grain_size 为 800
    parallel_for(0, n, /* grain_size= */ 800, sample);

    // 如果数据不是连续存储的，则调用 OptionalTensorRef(self) 的 copy_ 方法复制 tmp_int_tensor 的数据到 self
    if (!contig) {
      OptionalTensorRef(self)->copy_(tmp_int_tensor);
    }
#else
void exponential_kernel(TensorIteratorBase &iter, double lambda, std::optional<Generator> gen) {
  // 检查迭代器的数据类型是否为浮点型，因为指数分布是一个连续概率分布，要求数据类型必须是浮点型
  TORCH_CHECK(isFloatingType(iter.dtype()), "Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ", iter.dtype());

  // 获取迭代器中的第一个张量
  Tensor self = iter.tensor(0);

  // 检查 lambda 值的有效性，lambda 必须大于0且不是无穷大或NaN
  if (lambda > 0 && !std::isinf(lambda) && !std::isnan(lambda)) {
    // 获取随机数生成器或使用默认的 CPU 随机数生成器
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    int64_t seed;
    {
      // 见注释 [使用随机数生成器时获取锁]
      // 获取锁，确保在使用随机数生成器时线程安全
      std::lock_guard<std::mutex> lock(generator->mutex_);

      // 根据张量的数据类型选择不同精度的随机数生成方法
      if (self.scalar_type() == at::kDouble)
        seed = generator->random64();  // 双精度浮点型使用64位随机数生成器
      else
        seed = generator->random();     // 其他浮点型使用默认随机数生成器
    }

    // 获取张量中元素的数量
    int64_t n = self.numel();

    // 检查张量是否连续存储
    bool contig = self.is_contiguous();
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏来处理浮点类型和半精度类型的情况
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "exponential_cpu", [&] {
      // 声明一个临时的张量变量
      at::Tensor tmp_tensor;
      // 判断当前标量类型是否为双精度浮点数或单精度浮点数，并且是否是连续的
      constexpr bool is_df = std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value;
      if (is_df && contig) {
        // 如果是双精度浮点数且是连续的，则直接使用原始张量 self
        tmp_tensor = self;
      } else if (std::is_same<scalar_t, double>::value) {
        // 如果是双精度浮点数，则创建一个双精度空张量 tmp_tensor
        tmp_tensor = at::empty(self.sizes(), self.options().dtype(at::kDouble));
      } else {
        // 其他情况创建一个单精度浮点数空张量 tmp_tensor
        tmp_tensor = at::empty(self.sizes(), self.options().dtype(at::kFloat));
      }

      // 获取原始张量 self 和临时张量 tmp_tensor 的数据指针
      scalar_t *self_ptr = self.data_ptr<scalar_t>();
      using tmp_scalar_t = typename std::conditional_t<std::is_same<scalar_t, double>::value, double, float>;
      tmp_scalar_t *sample_ptr = tmp_tensor.data_ptr<tmp_scalar_t>();

      // 定义一个 lambda 函数 sample，用于生成随机数
      auto sample = [&](int64_t begin, int64_t end) {
        int64_t len = end - begin;
        if (len > 0) {
          // 创建一个随机数流 stream
          VSLStreamStatePtr stream;
          if constexpr (std::is_same<scalar_t, double>::value) {
            // 如果是双精度浮点数，则使用双精度指数分布函数生成随机数
            vslNewStream(&stream, VSL_BRNG_MCG31, seed);
            vslSkipAheadStream(stream, begin);
            vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, stream, len,
              (double *)(sample_ptr + begin), eps, 1./lambda);
            vslDeleteStream(&stream);
          } else {
            // 否则使用单精度指数分布函数生成随机数
            vslNewStream(&stream, VSL_BRNG_MCG31, seed);
            vslSkipAheadStream(stream, begin);
            vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, stream, len,
              (float *) (sample_ptr + begin), eps, 1./lambda);
            vslDeleteStream(&stream);
          }
          // 如果不是双精度浮点数且是连续的，则进行向量化的拷贝
          if (!is_df && contig) {
            scalar_t *self_seg = self_ptr + begin;
            tmp_scalar_t *tmp_seg = sample_ptr + begin;
            at::vec::convert<tmp_scalar_t, scalar_t>(tmp_seg, self_seg, len);
          }
        }
      };

      // 并行处理生成随机数，使用 parallel_for 函数
      parallel_for(0, n, /* grain_size= */ 800, sample);

      // 如果不是连续的，则执行原始张量到临时张量的拷贝
      if (!contig) {
        self.copy_(tmp_tensor);
      }
    });
  } else {
    // 当出现 inf 和 nan 的情况时，使用默认版本的指数核函数
    exponential_kernel_default(iter, lambda, gen);
#ifdef 条件编译指令，用于在编译时根据条件判断是否包含代码段
static void geometric_kernel(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
  // 获取或默认创建一个 CPU 生成器实例
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 几何核心函数
  templates::cpu::geometric_kernel(iter, p, generator);
}

static void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  // 获取或默认创建一个 CPU 生成器实例
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 对数正态核心函数
  templates::cpu::log_normal_kernel(iter, mean, std, generator);
}

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  // 获取或默认创建一个 CPU 生成器实例
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 均匀分布核心函数
  templates::cpu::uniform_kernel(iter, from, to, generator);
}

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  // 获取或默认创建一个 CPU 生成器实例
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 正态分布核心函数
  templates::cpu::normal_kernel(self, mean, std, generator);
}

static void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen) {
  // 获取或默认创建一个 CPU 生成器实例
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 指定范围内随机数核心函数
  templates::cpu::random_from_to_kernel(iter, range, base, generator);
}

static void random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen) {
  // 获取或默认创建一个 CPU 生成器实例
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 随机数核心函数
  templates::cpu::random_kernel(iter, generator);
}

// 这是处理特定情况的特殊核心函数：
// from（包含）= std::numeric_limits<int64_t>::lowest()
// to（不包含）= None（= std::numeric_limits<int64_t>::max() + 1）
static void random_full_64_bits_range_kernel(TensorIteratorBase& iter, std::optional<Generator> gen) {
  // 获取或默认创建一个 CPU 生成器实例
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  // 调用模板中的 CPU 64 位全范围随机数核心函数
  templates::cpu::random_full_64_bits_range_kernel(iter, generator);
}

} // namespace (anonymous)

// 注册分发函数，将伯努利张量函数与对应的内核函数绑定
REGISTER_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel);
// 注册分发函数，将伯努利标量函数与对应的内核函数绑定
REGISTER_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel);
// 注册分发函数，将柯西分布函数与对应的内核函数绑定
REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel);
// 注册分发函数，将指数分布函数与对应的内核函数绑定
REGISTER_DISPATCH(exponential_stub, &exponential_kernel);
// 注册分发函数，将几何分布函数与对应的内核函数绑定
REGISTER_DISPATCH(geometric_stub, &geometric_kernel);
// 注册分发函数，将对数正态分布函数与对应的内核函数绑定
REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel);
// 注册分发函数，将正态分布函数与对应的内核函数绑定
REGISTER_DISPATCH(normal_stub, &normal_kernel);
// 注册分发函数，将均匀分布函数与对应的内核函数绑定
REGISTER_DISPATCH(uniform_stub, &uniform_kernel);
// 注册分发函数，将指定范围内随机数函数与对应的内核函数绑定
REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel);
// 注册分发函数，将64位全范围随机数函数与对应的内核函数绑定
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel);
// 注册分发函数，将随机数函数与对应的内核函数绑定
REGISTER_DISPATCH(random_stub, &random_kernel);

} // namespace at::native
```