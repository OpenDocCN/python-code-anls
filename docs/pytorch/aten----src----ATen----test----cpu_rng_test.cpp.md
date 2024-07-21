# `.\pytorch\aten\src\ATen\test\cpu_rng_test.cpp`

```
#include <gtest/gtest.h>
#include <ATen/test/rng_test.h>
#include <ATen/Generator.h>
#include <c10/core/GeneratorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <stdexcept>

using namespace at;

#ifndef ATEN_CPU_STATIC_DISPATCH
// 定义自定义 RNG 的键值
namespace {
  
// 自定义 CPU Generator 类，继承自 c10::GeneratorImpl
constexpr auto kCustomRNG = DispatchKey::CustomRNGKeyId;
struct TestCPUGenerator : public c10::GeneratorImpl {
  // 构造函数，设置设备类型为 CPU，并指定使用自定义 RNG 的 DispatchKeySet
  TestCPUGenerator(uint64_t value) : GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(kCustomRNG)}, value_(value) { }
  // 默认析构函数
  ~TestCPUGenerator() override = default;

  // 返回预设的伪随机数值
  uint32_t random() { return value_; }
  // 返回预设的 64 位伪随机数值
  uint64_t random64() { return value_; }
  // 返回下一个单精度正态分布的样本值（可选）
  std::optional<float> next_float_normal_sample() { return next_float_normal_sample_; }
  // 返回下一个双精度正态分布的样本值（可选）
  std::optional<double> next_double_normal_sample() { return next_double_normal_sample_; }
  // 设置下一个单精度正态分布的样本值（可选）
  void set_next_float_normal_sample(std::optional<float> randn) { next_float_normal_sample_ = randn; }
  // 设置下一个双精度正态分布的样本值（可选）
  void set_next_double_normal_sample(std::optional<double> randn) { next_double_normal_sample_ = randn; }
  
  // 设置当前种子值（未实现）
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  // 设置偏移量（未实现）
  void set_offset(uint64_t offset) override { throw std::runtime_error("not implemented"); }
  // 获取偏移量（未实现）
  uint64_t get_offset() const override { throw std::runtime_error("not implemented"); }
  // 获取当前种子值（未实现）
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  // 获取种子值（未实现）
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  // 设置状态（未实现）
  void set_state(const c10::TensorImpl& new_state) override { throw std::runtime_error("not implemented"); }
  // 获取状态（未实现）
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { throw std::runtime_error("not implemented"); }
  // 克隆实现（未实现）
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  // 返回设备类型为 CPU
  static DeviceType device_type() { return DeviceType::CPU; }

  // 存储设定的伪随机数值
  uint64_t value_;
  // 下一个单精度正态分布的样本值（可选）
  std::optional<float> next_float_normal_sample_;
  // 下一个双精度正态分布的样本值（可选）
  std::optional<double> next_double_normal_sample_;
};

// ==================================================== Random ========================================================

// 实现随机数填充操作，调用具体的 CPU 随机数内核
Tensor& random_(Tensor& self, std::optional<Generator> generator) {
  return at::native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
}

// 实现从 from 到 to 的随机数填充操作，调用具体的 CPU 随机数内核
Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, std::optional<Generator> generator) {
  return at::native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

// 实现从 0 到 to 的随机数填充操作，调用具体的 CPU 随机数内核
Tensor& random_to(Tensor& self, int64_t to, std::optional<Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

// ==================================================== Normal ========================================================
# 使用正态分布填充给定张量，直接修改并返回该张量
Tensor& normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl_<native::templates::cpu::NormalKernel, TestCPUGenerator>(self, mean, std, gen);
}

# 从正态分布生成数据，存入预分配的输出张量，并返回该张量的引用
Tensor& normal_Tensor_float_out(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

# 从正态分布生成数据，存入预分配的输出张量，并返回该张量的引用
Tensor& normal_float_Tensor_out(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

# 从正态分布生成数据，存入预分配的输出张量，并返回该张量的引用
Tensor& normal_Tensor_Tensor_out(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

# 从正态分布生成数据，返回新的张量
Tensor normal_Tensor_float(const Tensor& mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

# 从正态分布生成数据，返回新的张量
Tensor normal_float_Tensor(double mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

# 从正态分布生成数据，返回新的张量
Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

# 使用均匀分布填充给定张量，直接修改并返回该张量
Tensor& uniform_(Tensor& self, double from, double to, std::optional<Generator> generator) {
  return at::native::templates::uniform_impl_<native::templates::cpu::UniformKernel, TestCPUGenerator>(self, from, to, generator);
}

# 使用柯西分布填充给定张量，直接修改并返回该张量
Tensor& cauchy_(Tensor& self, double median, double sigma, std::optional<Generator> generator) {
  return at::native::templates::cauchy_impl_<native::templates::cpu::CauchyKernel, TestCPUGenerator>(self, median, sigma, generator);
}

# 使用对数正态分布填充给定张量，直接修改并返回该张量
Tensor& log_normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::log_normal_impl_<native::templates::cpu::LogNormalKernel, TestCPUGenerator>(self, mean, std, gen);
}

# 使用几何分布填充给定张量，直接修改并返回该张量
Tensor& geometric_(Tensor& self, double p, std::optional<Generator> gen) {
  return at::native::templates::geometric_impl_<native::templates::cpu::GeometricKernel, TestCPUGenerator>(self, p, gen);
}
// ================================================== Exponential =====================================================

// 实现了对张量的指数分布随机采样，使用指定的生成器（可选）
Tensor& exponential_(Tensor& self, double lambda, std::optional<Generator> gen) {
  return at::native::templates::exponential_impl_<native::templates::cpu::ExponentialKernel, TestCPUGenerator>(self, lambda, gen);
}

// ================================================== Bernoulli =======================================================

// 根据张量的概率分布 p_ 进行伯努利分布的随机采样，使用指定的生成器（可选）
Tensor& bernoulli_Tensor(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(self, p_, gen);
}

// 根据指定的概率 p 进行伯努利分布的随机采样，使用指定的生成器（可选）
Tensor& bernoulli_float(Tensor& self, double p, std::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(self, p, gen);
}

// 根据张量的概率分布和指定生成器，在输出张量中进行伯努利分布的随机采样
Tensor& bernoulli_out(const Tensor& self, std::optional<Generator> gen, Tensor& result) {
  return at::native::templates::bernoulli_out_impl<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(result, self, gen);
}

// 注册自定义的随机数生成函数实现到 PyTorch 的 ATen 库中
TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
  // Random
  m.impl("random_.from",             random_from_to);             // 实现从指定范围内随机采样
  m.impl("random_.to",               random_to);                  // 实现到指定范围内随机采样
  m.impl("random_",                  random_);                    // 实现张量的随机采样
  // Normal
  m.impl("normal_",                  normal_);                    // 实现正态分布的随机采样
  m.impl("normal.Tensor_float_out",  normal_Tensor_float_out);    // 实现张量与标量参数的正态分布随机采样
  m.impl("normal.float_Tensor_out",  normal_float_Tensor_out);    // 实现标量与张量参数的正态分布随机采样
  m.impl("normal.Tensor_Tensor_out", normal_Tensor_Tensor_out);   // 实现两个张量参数的正态分布随机采样
  m.impl("normal.Tensor_float",      normal_Tensor_float);        // 实现张量与标量参数的正态分布随机采样
  m.impl("normal.float_Tensor",      normal_float_Tensor);        // 实现标量与张量参数的正态分布随机采样
  m.impl("normal.Tensor_Tensor",     normal_Tensor_Tensor);       // 实现两个张量参数的正态分布随机采样
  m.impl("uniform_",                 uniform_);                   // 实现张量的均匀分布随机采样
  // Cauchy
  m.impl("cauchy_",                  cauchy_);                    // 实现柯西分布的随机采样
  // LogNormal
  m.impl("log_normal_",              log_normal_);                // 实现对数正态分布的随机采样
  // Geometric
  m.impl("geometric_",               geometric_);                 // 实现几何分布的随机采样
  // Exponential
  m.impl("exponential_",             exponential_);               // 实现指数分布的随机采样
  // Bernoulli
  m.impl("bernoulli.out",            bernoulli_out);               // 实现张量的伯努利分布随机采样
  m.impl("bernoulli_.Tensor",        bernoulli_Tensor);           // 实现张量的伯努利分布随机采样
  m.impl("bernoulli_.float",         bernoulli_float);            // 实现标量参数的伯努利分布随机采样
}

// 测试类，用于测试随机数生成器的功能
class RNGTest : public ::testing::Test {
};

// 随机数生成中使用的魔数
static constexpr auto MAGIC_NUMBER = 424242424242424242ULL;

// ==================================================== Random ========================================================
// 在 RNGTest 类中定义一个测试函数 RandomFromTo，用于测试从特定数据类型的最小值到最大值之间的随机数生成器
TEST_F(RNGTest, RandomFromTo) {
  // 创建一个 CPU 设备对象
  const at::Device device("cpu");
  // 对每种数据类型进行从最小值到最大值的随机数生成器测试
  test_random_from_to<TestCPUGenerator, torch::kBool, bool>(device);
  test_random_from_to<TestCPUGenerator, torch::kUInt8, uint8_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt8, int8_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt16, int16_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt32, int32_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt64, int64_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kFloat32, float>(device);
  test_random_from_to<TestCPUGenerator, torch::kFloat64, double>(device);
}

// 在 RNGTest 类中定义一个测试函数 Random，用于测试随机数生成器能否正确生成各种数据类型的随机数
TEST_F(RNGTest, Random) {
  // 创建一个 CPU 设备对象
  const at::Device device("cpu");
  // 对每种数据类型进行随机数生成器的测试
  test_random<TestCPUGenerator, torch::kBool, bool>(device);
  test_random<TestCPUGenerator, torch::kUInt8, uint8_t>(device);
  test_random<TestCPUGenerator, torch::kInt8, int8_t>(device);
  test_random<TestCPUGenerator, torch::kInt16, int16_t>(device);
  test_random<TestCPUGenerator, torch::kInt32, int32_t>(device);
  test_random<TestCPUGenerator, torch::kInt64, int64_t>(device);
  test_random<TestCPUGenerator, torch::kFloat32, float>(device);
  test_random<TestCPUGenerator, torch::kFloat64, double>(device);
}

// 该测试用例证明 Tensor.random_() 分布能够生成无符号 64 位最大值（64 个 1）
// 参考链接：https://github.com/pytorch/pytorch/issues/33299
TEST_F(RNGTest, Random64bits) {
  // 创建一个生成器，以 uint64_t 类型的最大值作为种子
  auto gen = at::make_generator<TestCPUGenerator>(std::numeric_limits<uint64_t>::max());
  // 创建一个 int64_t 类型的 Tensor
  auto actual = torch::empty({1}, torch::kInt64);
  // 使用生成器生成随机数，并设置范围为 int64_t 类型的最小值到无限大，生成的结果应该等于 uint64_t 类型的最大值
  actual.random_(std::numeric_limits<int64_t>::min(), c10::nullopt, gen);
  // 断言生成的随机数应该等于 uint64_t 类型的最大值
  ASSERT_EQ(static_cast<uint64_t>(actual[0].item<int64_t>()), std::numeric_limits<uint64_t>::max());
}

// 在 RNGTest 类中定义一个测试函数 Normal，用于测试正态分布生成器的正确性
TEST_F(RNGTest, Normal) {
  // 设置正态分布的均值和标准差
  const auto mean = 123.45;
  const auto std = 67.89;
  // 创建一个生成器，以 MAGIC_NUMBER 作为种子
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个大小为 10 的空 Tensor
  auto actual = torch::empty({10});
  // 使用生成器生成正态分布随机数
  actual.normal_(mean, std, gen);

  // 创建一个与 actual 相同大小的空 Tensor
  auto expected = torch::empty_like(actual);
  // 调用 native 函数计算期望的正态分布结果
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 的值是否全部接近
  ASSERT_TRUE(torch::allclose(actual, expected));
}

// 在 RNGTest 类中定义一个测试函数 Normal_float_Tensor_out，用于测试正态分布生成器生成到指定 Tensor 的正确性
TEST_F(RNGTest, Normal_float_Tensor_out) {
  // 设置正态分布的均值和标准差
  const auto mean = 123.45;
  const auto std = 67.89;
  // 创建一个生成器，以 MAGIC_NUMBER 作为种子
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个大小为 10 的空 Tensor
  auto actual = torch::empty({10});
  // 使用生成器生成正态分布随机数，输出到 actual 中
  at::normal_out(actual, mean, torch::full({10}, std), gen);

  // 创建一个与 actual 相同大小的空 Tensor
  auto expected = torch::empty_like(actual);
  // 调用 native 函数计算期望的正态分布结果
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 的值是否全部接近
  ASSERT_TRUE(torch::allclose(actual, expected));
}
TEST_F(RNGTest, Normal_Tensor_float_out) {
  // 设置正态分布的均值和标准差
  const auto mean = 123.45;
  const auto std = 67.89;
  // 创建一个测试用的随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个空的张量用于存储生成的随机数
  auto actual = torch::empty({10});
  // 使用指定的均值和标准差生成正态分布的随机数并存储到 actual 张量中
  at::normal_out(actual, torch::full({10}, mean), std, gen);

  // 创建一个与 actual 相同大小的空张量 expected
  auto expected = torch::empty_like(actual);
  // 调用 CPU 上的正态分布核函数，生成期望的随机数填充到 expected 张量中
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  // 使用 torch::allclose 检查 actual 和 expected 张量是否接近
  ASSERT_TRUE(torch::allclose(actual, expected));
}

// 后续的测试用例（Normal_Tensor_Tensor_out, Normal_float_Tensor, Normal_Tensor_float, Normal_Tensor_Tensor）
// 都以类似的方式进行操作，用不同的参数配置测试正态分布生成函数

TEST_F(RNGTest, Uniform) {
  // 设置均匀分布的起始值和结束值
  const auto from = -24.24;
  const auto to = 42.42;
  // 创建一个测试用的随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个 3x3 大小的空张量 actual
  auto actual = torch::empty({3, 3});
  // 使用指定的起始值和结束值生成均匀分布的随机数填充到 actual 张量中
  actual.uniform_(from, to, gen);

  // 创建一个与 actual 相同大小的空张量 expected
  auto expected = torch::empty_like(actual);
  // 创建一个迭代器并调用 CPU 上的均匀分布核函数，生成期望的随机数填充到 expected 张量中
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::uniform_kernel(iter, from, to, check_generator<TestCPUGenerator>(gen));

  // 使用 torch::allclose 检查 actual 和 expected 张量是否接近
  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ==================================================== Cauchy ========================================================
// Cauchy 分布的测试用例未提供代码，因此无需注释
// 在 RNGTest 测试类中定义 Cauchy 分布的测试函数
TEST_F(RNGTest, Cauchy) {
  // 定义中值和标准差
  const auto median = 123.45;
  const auto sigma = 67.89;
  // 创建测试用的随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个空的张量 actual，并生成 Cauchy 分布的随机数填充到 actual 中
  auto actual = torch::empty({3, 3});
  actual.cauchy_(median, sigma, gen);

  // 创建一个与 actual 相同大小的空张量 expected
  auto expected = torch::empty_like(actual);
  // 创建一个张量迭代器 iter，并使用 CPU 上的 cauchy_kernel 填充 expected
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::cauchy_kernel(iter, median, sigma, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 和 expected 的所有元素近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== LogNormal =======================================================

// 在 RNGTest 测试类中定义 LogNormal 分布的测试函数
TEST_F(RNGTest, LogNormal) {
  // 定义均值和标准差
  const auto mean = 12.345;
  const auto std = 6.789;
  // 创建测试用的随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个大小为 10 的空张量 actual，并生成 LogNormal 分布的随机数填充到 actual 中
  auto actual = torch::empty({10});
  actual.log_normal_(mean, std, gen);

  // 创建一个与 actual 相同大小的空张量 expected
  auto expected = torch::empty_like(actual);
  // 创建一个张量迭代器 iter，并使用 CPU 上的 log_normal_kernel 填充 expected
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::log_normal_kernel(iter, mean, std, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 和 expected 的所有元素近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== Geometric =======================================================

// 在 RNGTest 测试类中定义 Geometric 分布的测试函数
TEST_F(RNGTest, Geometric) {
  // 定义几何分布的概率参数 p
  const auto p = 0.42;
  // 创建测试用的随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个大小为 3x3 的空张量 actual，并生成几何分布的随机数填充到 actual 中
  auto actual = torch::empty({3, 3});
  actual.geometric_(p, gen);

  // 创建一个与 actual 相同大小的空张量 expected
  auto expected = torch::empty_like(actual);
  // 创建一个张量迭代器 iter，并使用 CPU 上的 geometric_kernel 填充 expected
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::geometric_kernel(iter, p, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 和 expected 的所有元素近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== Exponential =====================================================

// 在 RNGTest 测试类中定义 Exponential 分布的测试函数
TEST_F(RNGTest, Exponential) {
  // 定义指数分布的参数 lambda
  const auto lambda = 42;
  // 创建测试用的随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个大小为 3x3 的空张量 actual，并生成指数分布的随机数填充到 actual 中
  auto actual = torch::empty({3, 3});
  actual.exponential_(lambda, gen);

  // 创建一个与 actual 相同大小的空张量 expected
  auto expected = torch::empty_like(actual);
  // 创建一个张量迭代器 iter，并使用 CPU 上的 exponential_kernel 填充 expected
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::exponential_kernel(iter, lambda, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 和 expected 的所有元素近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ==================================================== Bernoulli =====================================================

// 在 RNGTest 测试类中定义 Bernoulli 分布的测试函数
TEST_F(RNGTest, Bernoulli_Tensor) {
  // 定义伯努利分布的概率参数 p
  const auto p = 0.42;
  // 创建测试用的随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个大小为 3x3 的空张量 actual，并生成伯努利分布的随机数填充到 actual 中
  auto actual = torch::empty({3, 3});
  actual.bernoulli_(torch::full({3,3}, p), gen);

  // 创建一个与 actual 相同大小的空张量 expected
  auto expected = torch::empty_like(actual);
  // 使用 CPU 上的 bernoulli_kernel 生成伯努利分布的数据填充 expected
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  // 断言 actual 和 expected 的所有元素近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}
TEST_F(RNGTest, Bernoulli_scalar) {
  // 设置概率值
  const auto p = 0.42;
  // 创建随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个空的张量
  auto actual = torch::empty({3, 3});
  // 在指定概率下填充张量的元素为伯努利随机变量
  actual.bernoulli_(p, gen);

  // 创建一个与 actual 相同大小的空张量
  auto expected = torch::empty_like(actual);
  // 调用 CPU 的伯努利核函数生成期望的伯努利随机变量张量
  native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 张量近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli) {
  // 设置概率值
  const auto p = 0.42;
  // 创建随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 使用给定的概率在 gen 生成的随机数张量上生成伯努利随机变量张量
  auto actual = at::bernoulli(torch::full({3,3}, p), gen);

  // 创建一个与 actual 相同大小的空张量
  auto expected = torch::empty_like(actual);
  // 调用 CPU 的伯努利核函数生成期望的伯努利随机变量张量
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 张量近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_2) {
  // 设置概率值
  const auto p = 0.42;
  // 创建随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 使用给定的概率在生成的全为 p 的张量上生成伯努利随机变量张量
  auto actual = torch::full({3,3}, p).bernoulli(gen);

  // 创建一个与 actual 相同大小的空张量
  auto expected = torch::empty_like(actual);
  // 调用 CPU 的伯努利核函数生成期望的伯努利随机变量张量
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 张量近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_p) {
  // 设置概率值
  const auto p = 0.42;
  // 创建随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 在空的 3x3 张量上使用给定的概率生成伯努利随机变量张量
  auto actual = at::bernoulli(torch::empty({3, 3}), p, gen);

  // 创建一个与 actual 相同大小的空张量
  auto expected = torch::empty_like(actual);
  // 调用 CPU 的伯努利核函数生成期望的伯努利随机变量张量
  native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 张量近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_p_2) {
  // 设置概率值
  const auto p = 0.42;
  // 创建随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 在空的 3x3 张量上使用给定的概率生成伯努利随机变量张量
  auto actual = torch::empty({3, 3}).bernoulli(p, gen);

  // 创建一个与 actual 相同大小的空张量
  auto expected = torch::empty_like(actual);
  // 调用 CPU 的伯努利核函数生成期望的伯努利随机变量张量
  native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 张量近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_out) {
  // 设置概率值
  const auto p = 0.42;
  // 创建随机数生成器
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  // 创建一个空的 3x3 张量
  auto actual = torch::empty({3, 3});
  // 在给定的输出张量上使用给定的概率生成伯努利随机变量张量
  at::bernoulli_out(actual, torch::full({3,3}, p), gen);

  // 创建一个与 actual 相同大小的空张量
  auto expected = torch::empty_like(actual);
  // 调用 CPU 的伯努利核函数生成期望的伯努利随机变量张量
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  // 断言 actual 与 expected 张量近似相等
  ASSERT_TRUE(torch::allclose(actual, expected));
}
```