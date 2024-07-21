# `.\pytorch\test\cpp\lazy\test_lazy_ops.cpp`

```py
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>
#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/torch.h>
#include <iostream>

namespace torch {
namespace lazy {

// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
#ifndef FBCODE_CAFFE2

namespace {
// This registers the torchscript backend, without which lazy device won't work.
// FIXME: This registers the backend for the whole test binary. We should
// probably do it and undo it in the test fixture below.
// 初始化后端的静态函数，确保torchscript后端被注册
static bool inline init_backend() {
  torch::lazy::InitTorchScriptBackend();
  return true;
}
// 判断后端是否已经初始化的标志
static const bool backend_initialized = init_backend();

} // namespace

// 测试的基类，继承自gtest的测试基类Test
class LazyTsTest : public ::testing::Test {
 protected:
  void SetUp() override; // 设置测试环境
  void TearDown() override; // 清理测试环境

  static void CommonSetup() {} // 静态公共设置方法

  void ExpectCounterNotChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {} // 期望计数器未改变

  void ExpectCounterChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {} // 期望计数器已改变

  void ResetCounters() {} // 重置计数器

 private:
  void MakeEndSnapshot() {} // 创建结束快照
};

// Lazy操作的测试基类，继承自LazyTsTest
class LazyOpsTestBase : public LazyTsTest {
 protected:
  static void SetUpTestCase() {} // 设置测试用例
};

// 在LazyTsTest类中的SetUp方法中执行的操作
void LazyTsTest::SetUp() {
  (void)backend_initialized; // 避免未使用参数的警告
  at::manual_seed(42); // 设置随机种子
  torch::lazy::LazyGraphExecutor::Get()->SetRngSeed(
      torch::lazy::BackendDevice(), 42); // 设置随机数生成器种子
}

// 在LazyTsTest类中的TearDown方法中执行的操作
void LazyTsTest::TearDown() {} // 清理测试环境

namespace {
using torch::lazy::DebugUtil;

// Lazy操作的具体测试类，继承自LazyOpsTestBase
class LazyOpsTest : public LazyOpsTestBase {};

// 判断是否为CUDA环境的静态函数
static inline bool IsCuda() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType() == at::kCUDA;
}

// 获取默认设备类型的静态函数
static inline at::DeviceType DefaultDevice() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType();
}

} // namespace

// 测试用例，测试标量张量的创建和比较
TEST_F(LazyOpsTest, TestScalarTensor) {
  torch::Tensor scalar_tensor = torch::scalar_tensor(
      1., torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历每个设备，测试Lazy标量张量的创建和比较
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_scalar_tensor = torch::scalar_tensor(
        1., torch::TensorOptions(torch::kFloat).device(torch::kLazy));
    AllClose(scalar_tensor, lazy_scalar_tensor); // 判断两个张量是否近似相等
  });
}

// 测试用例，测试张量的克隆操作
TEST_F(LazyOpsTest, TestClone) {
  // 遍历每个设备，测试Lazy张量的克隆操作
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device); // 将张量a复制到指定设备
    torch::Tensor lazy_b = lazy_a.clone(); // 克隆Lazy张量
    AllClose(a, lazy_b); // 判断两个张量是否近似相等
    # 将 lazy_a 中的所有元素增加 1.0
    lazy_a.add_(1.0);
    # 检查 a 和 lazy_b 的所有元素是否近似相等
    AllClose(a, lazy_b);
  });
}

// 定义测试用例 TestTo
TEST_F(LazyOpsTest, TestTo) {
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 {2, 2} 的随机浮点数张量 a，放置在默认设备上
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 检查张量 a 和 lazy_a 是否全部接近
    AllClose(a, lazy_a);
  });
}

// 定义测试用例 TestIsFloatingPoint
TEST_F(LazyOpsTest, TestIsFloatingPoint) {
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 {2, 2} 的随机浮点数张量 a，放置在默认设备上
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 检查张量 a 和 lazy_a 是否为浮点数类型
    bool is_float = torch::is_floating_point(a);
    bool lazy_is_float = torch::is_floating_point(lazy_a);
    // 断言检查两者是否相等
    EXPECT_EQ(is_float, lazy_is_float);
  });
}

// 定义测试用例 TestIsSigned
TEST_F(LazyOpsTest, TestIsSigned) {
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 {2, 2} 的随机浮点数张量 a，放置在默认设备上
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 检查张量 a 和 lazy_a 是否为有符号类型
    bool is_signed = torch::is_signed(a);
    bool lazy_is_signed = torch::is_signed(lazy_a);
    // 断言检查两者是否相等
    EXPECT_EQ(is_signed, lazy_is_signed);
  });
}

// 定义测试用例 TestCastByte
TEST_F(LazyOpsTest, TestCastByte) {
  // 创建一个形状为 {2, 2} 的随机浮点数张量 a，乘以 100
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 将张量 a 转换为字节类型张量 b
  torch::Tensor b = torch::_cast_Byte(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 lazy_a 转换为字节类型张量 lazy_b
    torch::Tensor lazy_b = torch::_cast_Byte(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllEqual(b, lazy_b);
  });
}

// 定义测试用例 TestCastChar
TEST_F(LazyOpsTest, TestCastChar) {
  // 创建一个形状为 {2, 2} 的随机浮点数张量 a，乘以 100
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 将张量 a 转换为字符类型张量 b
  torch::Tensor b = torch::_cast_Char(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 lazy_a 转换为字符类型张量 lazy_b
    torch::Tensor lazy_b = torch::_cast_Char(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllEqual(b, lazy_b);
  });
}

// 定义测试用例 TestCastShort
TEST_F(LazyOpsTest, TestCastShort) {
  // 创建一个形状为 {2, 2} 的随机浮点数张量 a，乘以 100
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 将张量 a 转换为短整型类型张量 b
  torch::Tensor b = torch::_cast_Short(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 lazy_a 转换为短整型类型张量 lazy_b
    torch::Tensor lazy_b = torch::_cast_Short(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllEqual(b, lazy_b);
  });
}

// 定义测试用例 TestCastInt
TEST_F(LazyOpsTest, TestCastInt) {
  // 创建一个形状为 {2, 2} 的随机浮点数张量 a，乘以 100
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 将张量 a 转换为整型类型张量 b
  torch::Tensor b = torch::_cast_Int(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 lazy_a 转换为整型类型张量 lazy_b
    torch::Tensor lazy_b = torch::_cast_Int(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllEqual(b, lazy_b);
  });
}

// 定义测试用例 TestCastLong
TEST_F(LazyOpsTest, TestCastLong) {
  // 创建一个形状为 {2, 2} 的随机浮点数张量 a，乘以 100
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 将张量 a 转换为长整型类型张量 b
  torch::Tensor b = torch::_cast_Long(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定的设备上，返回复制后的张量 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 lazy_a 张量的数据类型转换为长整型，返回转换后的张量 lazy_b
    torch::Tensor lazy_b = torch::_cast_Long(lazy_a);
    // 检查张量 b 和 lazy_b 是否所有元素都相等，返回布尔值结果
    AllEqual(b, lazy_b);
  });
TEST_F(LazyOpsTest, TestCastFloat) {
  // 创建一个大小为 [2, 2] 的浮点数张量 a
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 将张量 a 转换为浮点数类型的张量 b
  torch::Tensor b = torch::_cast_Float(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，并创建懒惰张量 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将懒惰张量 lazy_a 转换为浮点数类型的懒惰张量 lazy_b
    torch::Tensor lazy_b = torch::_cast_Float(lazy_a);
    // 断言张量 b 和懒惰张量 lazy_b 在所有设备上的相等性
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRetainType) {
  // 创建一个大小为 [2, 2] 的字节类型的懒惰张量 lazy_a
  torch::Tensor lazy_a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  // 创建一个大小为 [2, 2] 的字节类型的懒惰张量 lazy_b
  torch::Tensor lazy_b = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  // 对懒惰张量 lazy_a 和 lazy_b 执行加法操作，结果为懒惰张量 lazy_c
  torch::Tensor lazy_c = lazy_a + lazy_b;
  // 断言懒惰张量 lazy_c 的标量类型为 Byte
  EXPECT_EQ(lazy_c.scalar_type(), torch::ScalarType::Byte);
}

TEST_F(LazyOpsTest, TestLogicalTypeWithInterop) {
  // 创建一个大小为 [2, 12, 20, 64] 的浮点数类型的懒惰张量 query
  torch::Tensor query = torch::rand(
      {2, 12, 20, 64},
      torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  // 创建一个大小为 [2, 12, 64, 20] 的浮点数类型的懒惰张量 key
  torch::Tensor key = torch::rand(
      {2, 12, 64, 20},
      torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  // 计算查询张量 query 和键张量 key 的矩阵乘法，然后除以标量 8，结果为 scores
  torch::Tensor scores =
      torch::matmul(query, key) /
      torch::scalar_tensor(
          8, torch::TensorOptions(torch::kDouble).device(torch::kLazy));
  // 对 scores 进行 softmax 操作，沿着最后一个维度（-1）
  torch::Tensor p_attn = torch::softmax(scores, /*dim=*/-1);
  // 断言 p_attn 的标量类型为 Float
  EXPECT_EQ(p_attn.scalar_type(), torch::ScalarType::Float);
}

TEST_F(LazyOpsTest, TestAdd) {
  // 创建一个大小为 [2, 2] 的浮点数类型的张量 a
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 [2, 2] 的浮点数类型的张量 b
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 和 b 的元素级加法，结果为张量 c
  torch::Tensor c = torch::add(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 和 b 复制到指定设备上，并创建对应的懒惰张量 lazy_a 和 lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算懒惰张量 lazy_a 和 lazy_b 的元素级加法，结果为懒惰张量 lazy_c
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    // 断言张量 c 和懒惰张量 lazy_c 在所有设备上的近似相等性
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddHalf) {
  // 创建一个大小为 [2, 2] 的半精度浮点数类型的张量 a
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  // 创建一个大小为 [2, 2] 的半精度浮点数类型的张量 b
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  // 计算张量 a 和 b 的元素级加法，结果为张量 c
  torch::Tensor c = torch::add(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 和 b 复制到指定设备上，并创建对应的懒惰张量 lazy_a 和 lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算懒惰张量 lazy_a 和 lazy_b 的元素级加法，结果为懒惰张量 lazy_c
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    // 断言张量 c 和懒惰张量 lazy_c 在所有设备上的近似相等性
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddMixedPrecision) {
  // 创建一个大小为 [2, 2] 的浮点数类型的张量 a
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 [2, 2] 的半精度浮点数类型的张量 b
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  // 计算张量 a 和 b 的元素级加法，结果为张量 c
  torch::Tensor c = torch::add(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 和 b 复制到指定设备上，并创建对应的懒惰张量 lazy_a 和 lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算懒惰张量 lazy_a 和 lazy_b 的元素级加法，结果为懒惰张量 lazy_c
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    // 断言张量 c 和懒惰张量 lazy_c 在所有设备上的近似相等性
    AllClose(c, lazy_c);
  });
}
TEST_F(LazyOpsTest, TestAddInPlace) {
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建随机张量 a，指定其设备为默认设备
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 创建随机张量 b，指定其设备为默认设备
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 b 复制到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 原地将张量 a 和 b 相加，得到张量 c
    torch::Tensor c = a.add_(b);
    // 在 lazy_a 和 lazy_b 上原地执行相加操作，得到 lazy_c
    torch::Tensor lazy_c = lazy_a.add_(lazy_b);
    // 检查张量 a 和 lazy_a 是否相似
    AllClose(a, lazy_a);
    // 检查张量 c 和 lazy_c 是否相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalar) {
  // 创建随机张量 a，指定其设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建标量 b，赋值为 1
  torch::Scalar b(1);
  // 张量 a 和标量 b 执行加法操作，得到张量 c
  torch::Tensor c = torch::add(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在 lazy_a 和标量 b 上执行加法操作，得到 lazy_c
    torch::Tensor lazy_c = torch::add(lazy_a, b);
    // 检查张量 c 和 lazy_c 是否相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalarInPlace) {
  // 创建标量 b，赋值为 1
  torch::Scalar b(1);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建随机张量 a，指定其设备为默认设备
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 原地将张量 a 和标量 b 相加，得到张量 c
    torch::Tensor c = a.add_(b);
    // 在 lazy_a 和标量 b 上原地执行相加操作，得到 lazy_c
    torch::Tensor lazy_c = lazy_a.add_(b);
    // 检查张量 a 和 lazy_a 是否相似
    AllClose(a, lazy_a);
    // 检查张量 c 和 lazy_c 是否相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddZeroSizeDim) {
  // 创建形状为 {0, 2} 的随机张量 a，指定其设备为默认设备
  torch::Tensor a = torch::rand(
      {0, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建形状为 {1, 2} 的随机张量 b，指定其设备为默认设备
  torch::Tensor b = torch::rand(
      {1, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 张量 a 和 b 执行加法操作，得到张量 c
  torch::Tensor c = torch::add(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在 lazy_a 和 lazy_b 上执行加法操作，得到 lazy_c
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    // 检查张量 c 和 lazy_c 是否相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSub) {
  // 创建随机张量 a，指定其设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机张量 b，指定其设备为默认设备
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 张量 a 和 b 执行减法操作，得到张量 c
  torch::Tensor c = torch::sub(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在 lazy_a 和 lazy_b 上执行减法操作，得到 lazy_c
    torch::Tensor lazy_c = torch::sub(lazy_a, lazy_b);
    // 检查张量 c 和 lazy_c 是否相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubInPlace) {
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建随机张量 a，指定其设备为默认设备
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 创建随机张量 b，指定其设备为默认设备
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量 b 复制到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 原地将张量 a 和 b 相减，得到张量 c
    torch::Tensor c = a.sub_(b);
    // 在 lazy_a 和 lazy_b 上原地执行相减操作，得到 lazy_c
    torch::Tensor lazy_c = lazy_a.sub_(lazy_b);
    // 检查张量 a 和 lazy_a 是否相似
    AllClose(a, lazy_a);
    // 检查张量 c 和 lazy_c 是否相似
    AllClose(c, lazy_c);
  });
}
}

TEST_F(LazyOpsTest, TestSubScalar) {
  // 创建一个大小为2x2的随机张量a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建标量b，值为1
  torch::Scalar b(1);
  // 计算张量c = a - b
  torch::Tensor c = torch::sub(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算lazy_c = lazy_a - b
    torch::Tensor lazy_c = torch::sub(lazy_a, b);
    // 检查是否所有元素在数值上接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubScalarInPlace) {
  // 创建标量b，值为1
  torch::Scalar b(1);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个大小为2x2的随机张量a，使用默认设备上的浮点数选项
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在原地计算张量a = a - b，得到张量c
    torch::Tensor c = a.sub_(b);
    // 在指定设备上原地计算lazy_a = lazy_a - b，得到lazy_c
    torch::Tensor lazy_c = lazy_a.sub_(b);
    // 检查两个张量a和lazy_a所有元素在数值上接近
    AllClose(a, lazy_a);
    // 检查两个张量c和lazy_c所有元素在数值上接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMul) {
  // 创建一个大小为2x2的随机张量a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为2x2的随机张量b，使用默认设备上的浮点数选项
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量c = a * b
  torch::Tensor c = torch::mul(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a和b分别复制到指定设备上，得到lazy_a和lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上计算lazy_c = lazy_a * lazy_b
    torch::Tensor lazy_c = torch::mul(lazy_a, lazy_b);
    // 检查张量c和lazy_c所有元素在数值上接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulInPlace) {
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个大小为2x2的随机张量a，使用默认设备上的浮点数选项
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 创建一个大小为2x2的随机张量b，使用默认设备上的浮点数选项
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量b复制到指定设备上，得到lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在原地计算张量a = a * b，得到张量c
    torch::Tensor c = a.mul_(b);
    // 在指定设备上原地计算lazy_a = lazy_a * lazy_b，得到lazy_c
    torch::Tensor lazy_c = lazy_a.mul_(lazy_b);
    // 检查两个张量a和lazy_a所有元素在数值上接近
    AllClose(a, lazy_a);
    // 检查两个张量c和lazy_c所有元素在数值上接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalar) {
  // 创建一个大小为2x2的随机张量a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建标量b，值为3
  torch::Scalar b(3);
  // 计算张量c = a * b
  torch::Tensor c = torch::mul(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算lazy_c = lazy_a * b
    torch::Tensor lazy_c = torch::mul(lazy_a, b);
    // 检查张量c和lazy_c所有元素在数值上接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalarInPlace) {
  // 创建标量b，值为3
  torch::Scalar b(3);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个大小为2x2的随机张量a，使用默认设备上的浮点数选项
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在原地计算张量a = a * b，得到张量c
    torch::Tensor c = a.mul_(b);
    // 在指定设备上原地计算lazy_a = lazy_a * b，得到lazy_c
    torch::Tensor lazy_c = lazy_a.mul_(b);
    // 检查两个张量a和lazy_a所有元素在数值上接近
    AllClose(a, lazy_a);
    // 检查两个张量c和lazy_c所有元素在数值上接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestDiv) {
  // 对于每个标量类型scalar_type1进行以下循环
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 创建一个张量 `a`，根据 `scalar_type1` 的浮点类型状态随机初始化或整数随机初始化，形状为 3x4
    torch::Tensor a = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    
    // 对于给定的一组标量类型 `scalar_type2`，依次进行以下操作
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      // 创建一个张量 `b`，根据 `scalar_type2` 的浮点类型状态随机初始化或整数随机初始化，形状为 3x4
      torch::Tensor b = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
    
      // 使用张量 `a` 和 `b` 计算元素级别的除法得到张量 `c`
      torch::Tensor c = torch::div(a, b);
    
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 `a` 和 `b` 复制到指定设备上，并计算在该设备上的元素级别除法得到张量 `lazy_c`
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = torch::div(lazy_a, lazy_b);
    
        // 检查张量 `c` 和 `lazy_c` 在指定设备上是否全部接近
        AllClose(c, lazy_c);
      });
    }
    }
}

# 定义测试用例 LazyOpsTest.TestDivWithRoundingMode，测试 torch.div 方法的不同取整模式
TEST_F(LazyOpsTest, TestDivWithRoundingMode) {
  # 可选的取整模式列表，包括 "trunc", "floor" 和无取整模式
  std::optional<c10::string_view> rounding_modes[] = {
      "trunc", "floor", c10::nullopt};
  # 遍历每种取整模式
  for (const auto& rounding_mode : rounding_modes) {
    # 遍历每种标量类型
    for (torch::ScalarType scalar_type1 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      # 根据标量类型设定下界，如果是 torch::kByte 则下界是 0，否则是 -100
      int lower_bound = (scalar_type1 == torch::kByte) ? 0 : -100;
      # 根据标量类型生成张量 a
      torch::Tensor a = isFloatingType(scalar_type1)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
          : torch::randint(
                lower_bound, 50, {3, 4}, torch::TensorOptions(scalar_type1));
      # 再次遍历每种标量类型
      for (torch::ScalarType scalar_type2 :
           {torch::kFloat,
            torch::kByte,
            torch::kChar,
            torch::kShort,
            torch::kInt,
            torch::kLong}) {
        # 根据标量类型生成张量 b
        torch::Tensor b = isFloatingType(scalar_type2)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
            : torch::randint(
                  51, 100, {3, 4}, torch::TensorOptions(scalar_type2));
        # 使用 torch.div 方法计算张量 a 和 b 的除法结果 c，带有指定的取整模式
        torch::Tensor c = torch::div(a, b, rounding_mode);
        # 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          # 将张量 a 和 b 复制到指定设备上
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          # 使用 torch.div 方法在指定设备上计算张量 lazy_a 和 lazy_b 的除法结果 lazy_c，带有指定的取整模式
          torch::Tensor lazy_c = torch::div(lazy_a, lazy_b, rounding_mode);
          # 检查 c 和 lazy_c 在设备上是否接近（值相近）
          AllClose(c, lazy_c);
        });
      }
    }
  }
}

# 定义测试用例 LazyOpsTest.TestDivInPlace，测试 torch.div_ 方法的原位操作
TEST_F(LazyOpsTest, TestDivInPlace) {
  # 遍历标量类型为 torch::kFloat 的情况
  for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
    # 根据标量类型生成张量 a
    torch::Tensor a = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    # 再次遍历标量类型为 torch::kFloat 的情况
    for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
      # 根据标量类型生成张量 b
      torch::Tensor b = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
      # 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        # 将张量 a 复制到指定设备上
        torch::Tensor lazy_a = CopyToDevice(a, device);
        # 使用 torch.div_ 方法对张量 a 原位进行除法操作，并将结果保存在张量 c 中
        torch::Tensor c = a.div_(b);
        # 将张量 b 复制到指定设备上
        torch::Tensor lazy_b = CopyToDevice(b, device);
        # 使用 torch.div_ 方法对张量 lazy_a 原位进行除法操作，并将结果保存在张量 lazy_c 中
        torch::Tensor lazy_c = lazy_a.div_(lazy_b);
        # 检查 c 和 lazy_c 在设备上是否接近（值相近）
        AllClose(c, lazy_c);
      });
    }
  }
}

# 定义测试用例 LazyOpsTest.TestDivInPlaceWithRoundingMode，测试 torch.div_ 方法的原位操作并带有取整模式
TEST_F(LazyOpsTest, TestDivInPlaceWithRoundingMode) {
  # 可选的取整模式列表，包括 "trunc", "floor" 和无取整模式
  std::optional<c10::string_view> rounding_modes[] = {
      "trunc", "floor", c10::nullopt};
  # 遍历每种取整模式
  for (const auto& rounding_mode : rounding_modes) {
    // 对于每个浮点类型，循环执行以下操作
    for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
      // 如果当前标量类型是浮点类型，创建一个形状为{3, 4}的随机张量a，类型为scalar_type1
      torch::Tensor a = isFloatingType(scalar_type1)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
          // 如果不是浮点类型，创建一个在指定范围内的随机整数张量a，类型为scalar_type1
          : torch::randint(
                -100, 100, {3, 4}, torch::TensorOptions(scalar_type1));
      // 对于每个浮点类型，循环执行以下操作
      for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
        // 如果当前标量类型是浮点类型，创建一个形状为{3, 4}的随机张量b，类型为scalar_type2
        torch::Tensor b = isFloatingType(scalar_type2)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
            // 如果不是浮点类型，创建一个在指定范围内的随机整数张量b，类型为scalar_type2
            : torch::randint(
                  1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
        // 对于每个设备，执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          // 将张量a复制到指定设备上，得到lazy_a
          torch::Tensor lazy_a = CopyToDevice(a, device);
          // 计算张量a除以张量b的结果c，并使用指定的舍入模式rounding_mode
          torch::Tensor c = a.div_(b, rounding_mode);
          // 将张量b复制到指定设备上，得到lazy_b
          torch::Tensor lazy_b = CopyToDevice(b, device);
          // 在指定设备上，计算lazy_a除以lazy_b的结果lazy_c，并使用指定的舍入模式rounding_mode
          torch::Tensor lazy_c = lazy_a.div_(lazy_b, rounding_mode);
          // 检查张量c和lazy_c是否在数值上相近
          AllClose(c, lazy_c);
        });
      }
    }
}

TEST_F(LazyOpsTest, TestDivScalar) {
  // 遍历各种标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型选择生成随机张量 a
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              1,
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 遍历布尔值 true 和 false
    for (bool is_float : {true, false}) {
      // 根据 is_float 选择标量 b
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      // 计算张量 c = a / b
      torch::Tensor c = torch::div(a, b);
      // 针对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 a 复制到指定设备上
        torch::Tensor lazy_a = CopyToDevice(a, device);
        // 在指定设备上计算 lazy_c = lazy_a / b
        torch::Tensor lazy_c = torch::div(lazy_a, b);
        // 检查两个张量是否近似相等
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivScalarInPlace) {
  // 遍历标量类型 torch::kFloat
  for (torch::ScalarType scalar_type : {torch::kFloat}) {
    // 根据标量类型选择生成随机张量 a
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              1,
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 遍历布尔值 true 和 false
    for (bool is_float : {true, false}) {
      // 根据 is_float 选择标量 b
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      // 针对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 a 复制到指定设备上
        torch::Tensor lazy_a = CopyToDevice(a, device);
        // 在原地进行除法运算 c = a / b
        torch::Tensor c = a.div_(b);
        // 在指定设备上进行原地除法运算 lazy_c = lazy_a / b
        torch::Tensor lazy_c = lazy_a.div_(b);
        // 检查两个张量是否近似相等
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivOut) {
  // 遍历标量类型 torch::kFloat 和 torch::kDouble
  for (torch::ScalarType scalar_type : {torch::kFloat, torch::kDouble}) {
    // 生成随机张量 a 和 b
    torch::Tensor a = torch::rand(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::rand(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建空张量 c
    torch::Tensor c = torch::empty(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 在 c 中进行除法运算 c = a / b
    torch::div_out(c, a, b);
    // 针对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 和 b 复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      // 创建与 lazy_b 相同选项的空张量 lazy_c
      torch::Tensor lazy_c = torch::empty({3, 4}, lazy_b.options());
      // 在指定设备上进行除法运算 lazy_c = lazy_a / lazy_b
      torch::div_out(lazy_c, lazy_a, lazy_b);
      // 检查两个张量是否近似相等
      AllClose(c, lazy_c);
    });
  }
}

TEST_F(LazyOpsTest, TestRsubScalar) {
  // 生成随机输入张量 input
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义标量 other 和 alpha
  torch::Scalar other(1.5);
  torch::Scalar alpha(2.5);
  // 计算张量 result = alpha * other - input
  torch::Tensor result = torch::rsub(input, other, alpha);
  // 针对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 input 复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上计算 lazy_result = alpha * other - lazy_input
    torch::Tensor lazy_result = torch::rsub(lazy_input, other, alpha);
    // 检查两个张量是否近似相等
    AllClose(result, lazy_result);
  });
}
TEST_F(LazyOpsTest, TestNe) {
  // 创建一个形状为 [2, 3] 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 [2, 3] 的随机张量 b，使用默认设备
  torch::Tensor b = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 和 b 逐元素不等的结果，得到张量 c
  torch::Tensor c = torch::ne(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 拷贝到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 拷贝到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算 lazy_a 和 lazy_b 逐元素不等的结果，得到 lazy_c
    torch::Tensor lazy_c = torch::ne(lazy_a, lazy_b);
    // 检查 c 和 lazy_c 是否完全相等
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestNeInplace) {
  // 创建一个形状为 [2, 3] 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 a，得到 a_copy
  torch::Tensor a_copy = a.clone();
  // 克隆张量 a，得到张量 b
  torch::Tensor b = a.clone();
  // 修改张量 b 的第一个元素，加上 1
  b[0] += 1;
  // 在张量 a 上进行原地操作，计算 a 与 b 逐元素不等的结果
  a.ne_(b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a_copy 拷贝到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    // 将张量 b 拷贝到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在 lazy_a 上进行原地操作，计算 lazy_a 和 lazy_b 逐元素不等的结果
    lazy_a.ne_(lazy_b);
    // 检查 lazy_a 和 a 是否在给定设备上全部接近
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestEq) {
  // 创建一个形状为 [2, 3] 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 a，得到张量 b
  torch::Tensor b = a.clone();
  // 计算张量 a 和 b 逐元素相等的结果，得到张量 c
  torch::Tensor c = torch::eq(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 拷贝到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 拷贝到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算 lazy_a 和 lazy_b 逐元素相等的结果，得到 lazy_c
    torch::Tensor lazy_c = torch::eq(lazy_a, lazy_b);
    // 检查 c 和 lazy_c 是否完全相等
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEqInplace) {
  // 创建一个形状为 [2, 3] 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 a，得到张量 b
  torch::Tensor b = a.clone();
  // 修改张量 b 的第一个元素，加上 1
  b[0] += 1;
  // 克隆张量 a，得到 a_copy
  torch::Tensor a_copy = a.clone();
  // 在张量 a 上进行原地操作，计算 a 与 b 逐元素相等的结果
  a.eq_(b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a_copy 拷贝到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    // 将张量 b 拷贝到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在 lazy_a 上进行原地操作，计算 lazy_a 和 lazy_b 逐元素相等的结果
    lazy_a.eq_(lazy_b);
    // 检查 lazy_a 和 a 是否在给定设备上全部接近
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGe) {
  // 创建一个形状为 [2, 3] 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 a，得到张量 b
  torch::Tensor b = a.clone();
  // 计算张量 a 和 b 逐元素大于等于的结果，得到张量 c
  torch::Tensor c = torch::ge(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 拷贝到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 拷贝到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算 lazy_a 和 lazy_b 逐元素大于等于的结果，得到 lazy_c
    torch::Tensor lazy_c = torch::ge(lazy_a, lazy_b);
    // 检查 c 和 lazy_c 是否完全相等
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGeInplace) {
  // 创建一个形状为 [2, 3] 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 a，得到张量 b
  torch::Tensor b = a.clone();
  // 修改张量 b 的第一个和第二个元素，分别加上 1 和减去 1
  b[0] += 1;
  b[1] -= 1;
  // 克隆张量 a，得到 a_copy
  torch::Tensor a_copy = a.clone();
  // 在张量 a 上进行原地操作，计算 a 与 b 逐元素大于等于的结果
  a.ge_(b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a_copy 拷贝到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    // 将张量 b 拷贝到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在 lazy_a 上进行原地操作，计算 lazy_a 和 lazy_b 逐元素大于等于的结果
    lazy_a.ge_(lazy_b);
    // 检查 lazy_a 和 a 是否在给定设备上全部接近
    AllClose(lazy_a, a);
  });
}
TEST_F(LazyOpsTest, TestLe) {
  // 生成一个大小为 2x3 的随机张量 `a`，使用默认设备（通常是 CPU）
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 `a` 到张量 `b`
  torch::Tensor b = a.clone();
  // 计算张量 `a` 和 `b` 逐元素的小于等于比较，结果存储在张量 `c` 中
  torch::Tensor c = torch::le(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 `a` 复制到指定设备，并创建 `lazy_a`
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 `b` 复制到指定设备，并创建 `lazy_b`
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上计算 `lazy_a` 和 `lazy_b` 的小于等于比较，结果存储在 `lazy_c`
    torch::Tensor lazy_c = torch::le(lazy_a, lazy_b);
    // 检查张量 `c` 和 `lazy_c` 在指定设备上的相等性
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLeInplace) {
  // 生成一个大小为 2x3 的随机张量 `a`，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 `a` 到张量 `b`
  torch::Tensor b = a.clone();
  // 修改张量 `b` 的第一个元素加 1，第二个元素减 1
  b[0] += 1;
  b[1] -= 1;
  // 克隆张量 `a` 到 `a_copy`
  torch::Tensor a_copy = a.clone();
  // 在张量 `a` 上进行逐元素的小于等于比较，并替换 `a` 的值
  a.le_(b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 `a_copy` 复制到指定设备，并创建 `lazy_a`
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    // 将张量 `b` 复制到指定设备，并创建 `lazy_b`
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上，对 `lazy_a` 进行逐元素的小于等于比较，同时修改 `lazy_a` 的值
    lazy_a.le_(lazy_b);
    // 检查 `lazy_a` 和 `a` 在指定设备上的近似相等性
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGt) {
  // 生成一个大小为 2x3 的随机张量 `a`，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 根据 `a` 生成一个大小相同的张量 `b`，每个元素加上 1
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  // 计算张量 `b` 和 `a` 逐元素的大于比较，结果存储在张量 `c` 中
  torch::Tensor c = torch::gt(b, a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 `a` 复制到指定设备，并创建 `lazy_a`
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 `b` 复制到指定设备，并创建 `lazy_b`
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上计算 `lazy_b` 和 `lazy_a` 的大于比较，结果存储在 `lazy_c`
    torch::Tensor lazy_c = torch::gt(lazy_b, lazy_a);
    // 检查张量 `c` 和 `lazy_c` 在指定设备上的相等性
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGtInplace) {
  // 生成一个大小为 2x3 的随机张量 `a`，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 `a` 到张量 `b`
  torch::Tensor b = a.clone();
  // 修改张量 `b` 的第一个元素加 1，第二个元素减 1
  b[0] += 1;
  b[1] -= 1;
  // 克隆张量 `a` 到 `a_copy`
  torch::Tensor a_copy = a.clone();
  // 在张量 `a` 上进行逐元素的大于比较，并替换 `a` 的值
  a.gt_(b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 `a_copy` 复制到指定设备，并创建 `lazy_a`
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    // 将张量 `b` 复制到指定设备，并创建 `lazy_b`
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上，对 `lazy_a` 进行逐元素的大于比较，同时修改 `lazy_a` 的值
    lazy_a.gt_(lazy_b);
    // 检查 `lazy_a` 和 `a` 在指定设备上的近似相等性
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestLt) {
  // 生成一个大小为 2x3 的随机张量 `a`，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 根据 `a` 生成一个大小相同的张量 `b`，每个元素加上 1
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  // 计算张量 `a` 和 `b` 逐元素的小于比较，结果存储在张量 `c` 中
  torch::Tensor c = torch::lt(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 `a` 复制到指定设备，并创建 `lazy_a`
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 `b` 复制到指定设备，并创建 `lazy_b`
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上计算 `lazy_a` 和 `lazy_b` 的小于比较，结果存储在 `lazy_c`
    torch::Tensor lazy_c = torch::lt(lazy_a, lazy_b);
    // 检查张量 `c` 和 `lazy_c` 在指定设备上的相等性
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLtInplace) {
  // 生成一个大小为 2x3 的随机张量 `a`，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 克隆张量 `a` 到张量 `b`
  torch::Tensor b = a.clone();
  // 修改张量 `b` 的第一个元素加 1，第二个元素减 1
  b[0] += 1;
  b[1] -= 1;
  // 克隆张量 `a` 到 `a_copy`
  torch::Tensor a_copy = a.clone();
  // 在张量 `a` 上进行逐元素的小于比较，并替换 `a` 的值
  a.lt_(b);
  // 对每个设备执行以下操作
TEST_F(LazyOpsTest, TestNeScalar) {
  // 创建一个形状为 (2, 3) 的张量，其元素全部为 1
  torch::Tensor input = torch::ones({2, 3});
  // 创建一个标量，其值为 0.0
  torch::Scalar other(float(0));
  // 对输入张量 input 中的每个元素与标量 other 进行不等于操作，返回结果张量
  torch::Tensor result = torch::ne(input, other);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 中的每个元素与标量 other 进行不等于操作，返回结果张量
    torch::Tensor lazy_result = torch::ne(lazy_input, other);
    // 检查 result 和 lazy_result 是否全等
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEqScalar) {
  // 创建一个形状为 (2, 3) 的张量，其元素全部为 1
  torch::Tensor input = torch::ones({2, 3});
  // 创建一个标量，其值为 1.0
  torch::Scalar other(float(1));
  // 对输入张量 input 中的每个元素与标量 other 进行等于操作，返回结果张量
  torch::Tensor result = torch::eq(input, other);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 中的每个元素与标量 other 进行等于操作，返回结果张量
    torch::Tensor lazy_result = torch::eq(lazy_input, other);
    // 检查 result 和 lazy_result 是否全等
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalar) {
  // 创建一个形状为 (2, 3) 的张量，其元素全部为 1
  torch::Tensor input = torch::ones({2, 3});
  // 创建一个标量，其值为 1.0
  torch::Scalar other(float(1));
  // 对输入张量 input 中的每个元素与标量 other 进行大于等于操作，返回结果张量
  torch::Tensor result = torch::ge(input, other);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 中的每个元素与标量 other 进行大于等于操作，返回结果张量
    torch::Tensor lazy_result = torch::ge(lazy_input, other);
    // 检查 result 和 lazy_result 是否全等
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalarInplace) {
  // 创建一个范围从 -1.0 到 1.0（步长为 0.5）的张量
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量，其值为 0.0
  torch::Scalar other(float(0));
  // 复制 input，并在原地将其每个元素与标量 other 进行大于等于操作
  torch::Tensor input_copy = input.clone();
  input.ge_(other);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input_copy 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    // 在 lazy_input 上原地将其每个元素与标量 other 进行大于等于操作
    lazy_input.ge_(other);
    // 检查 lazy_input 和 input 是否近似相等
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestLeScalar) {
  // 创建一个形状为 (2, 3) 的张量，其元素全部为 1
  torch::Tensor input = torch::ones({2, 3});
  // 创建一个标量，其值为 1.0
  torch::Scalar other(float(1));
  // 对输入张量 input 中的每个元素与标量 other 进行小于等于操作，返回结果张量
  torch::Tensor result = torch::le(input, other);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 中的每个元素与标量 other 进行小于等于操作，返回结果张量
    torch::Tensor lazy_result = torch::le(lazy_input, other);
    // 检查 result 和 lazy_result 是否全等
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLeScalarInplace) {
  // 创建一个范围从 -1.0 到 1.0（步长为 0.5）的张量
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量，其值为 0.0
  torch::Scalar other(float(0));
  // 复制 input，并在原地将其每个元素与标量 other 进行小于等于操作
  torch::Tensor input_copy = input.clone();
  input.le_(other);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input_copy 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    // 在 lazy_input 上原地将其每个元素与标量 other 进行小于等于操作
    lazy_input.le_(other);
    // 检查 lazy_input 和 input 是否近似相等
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestGtScalar) {
  // 创建一个形状为 (2, 3) 的张量，其元素全部为 1
  torch::Tensor input = torch::ones({2, 3});
  // 创建一个标量，其值为 0.5
  torch::Scalar other(float(0.5));
  // 对输入张量 input 中的每个元素与标量 other 进行大于操作，返回结果张量
  torch::Tensor result = torch::gt(input, other);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 中的每个元素与标量 other 进行大于操作，返回结果张量
    torch::Tensor lazy_result = torch::gt(lazy_input, other);
    // 检查 result 和 lazy_result 是否全等
    AllEqual(result, lazy_result);
  });
}
// 定义测试用例 TestGtScalarInplace，验证在输入张量上执行大于标量的原位操作
TEST_F(LazyOpsTest, TestGtScalarInplace) {
  // 创建从-1到1.5（步长为0.5）的浮点数张量，指定默认设备
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建标量值为0的张量标量 other
  torch::Scalar other(float(0));
  // 克隆输入张量 input，并命名为 input_copy
  torch::Tensor input_copy = input.clone();
  // 原位操作：将 input 中大于 other 的元素置为 true
  input.gt_(other);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input_copy 复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    // 在 lazy_input 上执行大于标量 other 的原位操作
    lazy_input.gt_(other);
    // 验证 lazy_input 和 input 在设备上的值是否全部相等
    AllClose(lazy_input, input);
  });
}

// 定义测试用例 TestLtScalar，验证在输入张量上执行小于标量的操作
TEST_F(LazyOpsTest, TestLtScalar) {
  // 创建形状为 [2, 3] 的全 1 的张量 input
  torch::Tensor input = torch::ones({2, 3});
  // 创建标量值为 1.5 的张量标量 other
  torch::Scalar other(float(1.5));
  // 在 input 和 other 之间执行小于操作，返回结果张量 result
  torch::Tensor result = torch::lt(input, other);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在 lazy_input 和标量 other 之间执行小于操作，返回 lazy_result
    torch::Tensor lazy_result = torch::lt(lazy_input, other);
    // 验证 lazy_result 是否与全局的 result 在设备上全部相等
    AllEqual(result, lazy_result);
  });
}

// 定义测试用例 TestLtScalarInplace，验证在输入张量上执行小于标量的原位操作
TEST_F(LazyOpsTest, TestLtScalarInplace) {
  // 创建从-1到1.5（步长为0.5）的浮点数张量，指定默认设备
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建标量值为0的张量标量 other
  torch::Scalar other(float(0));
  // 克隆输入张量 input，并命名为 input_copy
  torch::Tensor input_copy = input.clone();
  // 原位操作：将 input 中小于 other 的元素置为 true
  input.lt_(other);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input_copy 复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    // 在 lazy_input 上执行小于标量 other 的原位操作
    lazy_input.lt_(other);
    // 验证 lazy_input 和 input 在设备上的值是否全部相等
    AllClose(lazy_input, input);
  });
}

// 定义测试用例 TestIntegerAdd，验证整数张量的加法操作
TEST_F(LazyOpsTest, TestIntegerAdd) {
  // 定义整数类型列表
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 遍历每种整数类型
    for (auto type : types) {
      // 创建指定类型的随机整数张量 a 和 b，形状为 [2, 2]
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      // 根据类型选择标量值为整数或浮点数的 one
      torch::Scalar one =
          isIntegralType(type, false) ? torch::Scalar(1) : torch::Scalar(1.0);
      // 计算 b 加上 one 的结果张量 c
      torch::Tensor c = torch::add(b, one);

      // 将 a 和 b 复制到指定设备上，得到 lazy_a 和 lazy_b
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      // 在 lazy_b 和 one 之间执行加法操作，得到 lazy_c
      torch::Tensor lazy_c = torch::add(lazy_b, one);

      // 验证 c 和 lazy_c 在设备上是否全部相等
      AllEqual(c, lazy_c);
    }
  });
}
    // 对每个维度 n 进行循环
    for (auto n : dims) {
      // 使用 torch::rand() 生成一个 m 行 n 列的随机张量 a，数据类型为 float，放置在默认设备上
      torch::Tensor a = torch::rand(
          {m, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 对张量 a 进行奇异值分解 (SVD)，计算 U、S、V 是否需要计算
      auto b = torch::svd(a, /*some=*/true, /*compute_uv=*/true);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 a 复制到指定设备上，得到 lazy_a
        torch::Tensor lazy_a = CopyToDevice(a, device);
        // 对 lazy_a 进行奇异值分解 (SVD)，计算 U、S、V 是否需要计算
        auto lazy_b = torch::svd(lazy_a, /*some=*/true, /*compute_uv=*/true);
        // U 和 V 矩阵的列向量可能有不同的符号，因此只能通过绝对值来比较它们
        // 检查 b 的第一个返回元组的绝对值与 lazy_b 的绝对值是否全部接近
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        // 获取 b 的奇异值向量 diag 和 lazy_b 的奇异值向量 lazy_diag，确保它们的尺寸相同
        AllClose(
            std::get<1>(b),
            std::get<1>(lazy_b),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        // 检查 b 的第三个返回元组的绝对值与 lazy_b 的绝对值是否全部接近
        AllClose(
            std::get<2>(b).abs(),
            std::get<2>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
TEST_F(LazyOpsTest, TestQR) {
  // 定义静态维度数组 dims，包含两个维度值 4 和 7
  static const int dims[] = {4, 7};
  // 遍历 dims 数组中的每个维度 m
  for (auto m : dims) {
    // 再次遍历 dims 数组中的每个维度 n
    for (auto n : dims) {
      // 创建一个大小为 m x n 的随机浮点数张量 a，使用默认设备
      torch::Tensor a = torch::rand(
          {m, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 对张量 a 进行 QR 分解，并存储结果在 b 中
      auto b = torch::qr(a);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 a 复制到指定设备，得到 lazy_a
        torch::Tensor lazy_a = CopyToDevice(a, device);
        // 在指定设备上进行 lazy_a 的 QR 分解，存储结果在 lazy_b 中
        auto lazy_b = torch::qr(lazy_a);
        // 检查两个 QR 分解的结果的绝对值之差是否在给定容差范围内
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        // 检查两个 QR 分解的结果的绝对值之差是否在给定容差范围内
        AllClose(
            std::get<1>(b).abs(),
            std::get<1>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCholesky) {
  // 定义静态维度数组 dims，包含两个维度值 4 和 7
  static const int dims[] = {4, 7};
  // 遍历 dims 数组中的每个维度 m
  for (auto m : dims) {
    // 对每个 m，遍历两种上三角矩阵类型：上三角和下三角
    for (bool upper : {true, false}) {
      // 创建一个大小为 3 x m x m 的随机浮点数张量 a，使用默认设备
      torch::Tensor a = torch::rand(
          {3, m, m},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 创建一个正定的对称矩阵 pd_a，通过 a 和其转置以及单位矩阵的和得到
      torch::Tensor pd_a =
          torch::matmul(a, torch::transpose(a, 1, 2)) +
          torch::eye(
              m, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 对 pd_a 进行 Cholesky 分解，存储结果在 b 中
      auto b = torch::cholesky(pd_a, upper);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 pd_a 复制到指定设备，得到 lazy_a
        torch::Tensor lazy_a = CopyToDevice(pd_a, device);
        // 在指定设备上进行 lazy_a 的 Cholesky 分解，存储结果在 lazy_b 中
        auto lazy_b = torch::cholesky(lazy_a, upper);
        // 检查两个 Cholesky 分解的结果是否在给定容差范围内相等
        AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestLogDet) {
  // 定义静态维度数组 dims，包含两个维度值 4 和 7
  static const int dims[] = {4, 7};
  // 遍历 dims 数组中的每个维度 m
  for (auto m : dims) {
    // 创建一个大小为 3 x m x m 的随机浮点数张量 a，使用默认设备
    torch::Tensor a = torch::rand(
        {3, m, m}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 创建一个正定的对称矩阵 pd_a，通过 a 和其转置以及单位矩阵的和得到
    torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
        torch::eye(m,
                   torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 计算 pd_a 的对数行列式，存储结果在 b 中
    torch::Tensor b = torch::logdet(pd_a);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 pd_a 复制到指定设备，得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(pd_a, device);
      // 在指定设备上计算 lazy_a 的对数行列式，存储结果在 lazy_b 中
      torch::Tensor lazy_b = torch::logdet(lazy_a);
      // 检查两个对数行列式的计算结果是否在给定容差范围内相等
      AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestTriangularSolve) {
  // 定义静态维度数组 dims，包含两个维度值 4 和 7
  static const int dims[] = {4, 7};
  // 遍历两种矩阵类型：批处理（batched_a）和非批处理
  for (bool batched_a : {true, false}) {
    // 遍历 batched_b 变量的布尔值集合 {true, false}
    for (bool batched_b : {true, false}) {
      // 遍历 dims 向量中的每个维度 m
      for (auto m : dims) {
        // 遍历 dims 向量中的每个维度 n
        for (auto n : dims) {
          // 遍历 upper 变量的布尔值集合 {true, false}
          for (bool upper : {true, false}) {
            // 遍历 transpose 变量的布尔值集合 {true, false}
            for (bool transpose : {true, false}) {
              // 遍历 unitriangular 变量的布尔值集合 {true, false}
              for (bool unitriangular : {true, false}) {
                // 使用 torch 库生成一个 m x m 的随机张量 a
                torch::Tensor a = torch::randn(
                    {m, m},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice()));
                // 使用 torch 库生成一个 m x n 的随机张量 b
                torch::Tensor b = torch::randn(
                    {m, n},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice()));
                // 如果 batched_b 为 true，则将张量 a 在第一个维度上扩展为 {3, m, m} 并克隆
                a = batched_a ? a.expand({3, m, m}).clone() : a;
                // 如果 batched_b 为 true，则将张量 b 在第一个维度上扩展为 {3, m, n} 并克隆
                b = batched_b ? b.expand({3, m, n}).clone() : b;
                // 调用 torch::triangular_solve 函数进行三角解算
                auto result = torch::triangular_solve(
                    b,
                    a,
                    /*upper=*/upper,
                    /*transpose=*/transpose,
                    /*unitriangular=*/unitriangular);
                // 对每个设备执行以下操作
                ForEachDevice([&](const torch::Device& device) {
                  // 将张量 a 复制到指定设备上并命名为 lazy_a
                  torch::Tensor lazy_a = CopyToDevice(a, device);
                  // 将张量 b 复制到指定设备上并命名为 lazy_b
                  torch::Tensor lazy_b = CopyToDevice(b, device);
                  // 在指定设备上调用 torch::triangular_solve 进行三角解算
                  auto lazy_result = torch::triangular_solve(
                      lazy_b,
                      lazy_a,
                      /*upper=*/upper,
                      /*transpose=*/transpose,
                      /*unitriangular=*/unitriangular);
                  // 检查两个结果的第一个元素在指定容差下是否全部接近
                  AllClose(
                      std::get<0>(result),
                      std::get<0>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                  // 检查两个结果的第二个元素在指定容差下是否全部接近
                  AllClose(
                      std::get<1>(result),
                      std::get<1>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                });
              }
            }
          }
        }
      }
    }
}

# 测试用例：LazyOpsTest 中的 TestKthValue 函数
TEST_F(LazyOpsTest, TestKthValue) {
  # 创建一个形状为 (4, 5, 3) 的随机浮点数张量 a，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 迭代 k 从 1 到 3
  for (int k = 1; k <= 3; ++k) {
    # 获取张量的维度
    int rank = a.dim();
    # 迭代 dim 从 -rank 到 rank-1
    for (int dim = -rank; dim < rank; ++dim) {
      # 迭代 keepdim 为 false 和 true
      for (bool keepdim : {false, true}) {
        # 调用 torch::kthvalue 函数获取第 k 个值
        auto b = torch::kthvalue(a, k, dim, keepdim);
        # 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          # 将张量 a 拷贝到指定设备上，得到 lazy_a
          torch::Tensor lazy_a = CopyToDevice(a, device);
          # 在指定设备上调用 torch::kthvalue 函数，获取第 k 个值，得到 lazy_b
          auto lazy_b = torch::kthvalue(lazy_a, k, dim, keepdim);
          # 对比两个结果的第一个值是否全部接近
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          # 对比两个结果的第二个值是否全部相等
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

# 测试用例：LazyOpsTest 中的 TestTopK 函数
TEST_F(LazyOpsTest, TestTopK) {
  # 创建一个形状为 (4, 5, 3) 的随机浮点数张量 a，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 迭代 k 从 1 到 3
  for (int k = 1; k <= 3; ++k) {
    # 获取张量的维度
    int rank = a.dim();
    # 迭代 dim 从 -rank 到 rank-1
    for (int dim = -rank; dim < rank; ++dim) {
      # 迭代 largest 为 false 和 true
      for (bool largest : {false, true}) {
        # 调用 torch::topk 函数获取前 k 个元素
        auto b = torch::topk(a, k, dim, largest, /*sorted=*/true);
        # 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          # 将张量 a 拷贝到指定设备上，得到 lazy_a
          torch::Tensor lazy_a = CopyToDevice(a, device);
          # 在指定设备上调用 torch::topk 函数，获取前 k 个元素，得到 lazy_b
          auto lazy_b = torch::topk(lazy_a, k, dim, largest, /*sorted=*/true);
          # 对比两个结果的第一个值是否全部接近
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          # 对比两个结果的第二个值是否全部相等
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

# 测试用例：LazyOpsTest 中的 TestSort 函数
TEST_F(LazyOpsTest, TestSort) {
  # 创建一个形状为 (4, 5, 3) 的随机浮点数张量 a，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 迭代 k 从 1 到 3
  for (int k = 1; k <= 3; ++k) {
    # 迭代 dim 从 0 到 2
    for (int dim = 0; dim < 3; ++dim) {
      # 迭代 descending 为 false 和 true
      for (bool descending : {false, true}) {
        # 调用 torch::sort 函数对张量 a 进行排序
        auto b = torch::sort(a, dim, descending);
        # 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          # 将张量 a 拷贝到指定设备上，得到 lazy_a
          torch::Tensor lazy_a = CopyToDevice(a, device);
          # 在指定设备上调用 torch::sort 函数，对 lazy_a 进行排序，得到 lazy_b
          auto lazy_b = torch::sort(lazy_a, dim, descending);
          # 对比两个结果的第一个值是否全部接近
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          # 对比两个结果的第二个值是否全部相等
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

# 测试用例：LazyOpsTest 中的 TestSortDescWithMinValue 函数
TEST_F(LazyOpsTest, TestSortDescWithMinValue) {
  # 创建一个包含数值 {-128, 100} 的 int8_t 类型的向量 values
  std::vector<int8_t> values{-128, 100};
  # 创建 torch::Tensor 类型的输入张量 input，指定张量选项为 torch::kChar
  torch::Tensor input =
      torch::tensor(values, torch::TensorOptions(torch::kChar));
  # 调用 torch::sort 函数对 input 进行排序，descending 参数设为 true
  auto output = torch::sort(input, /*dim=*/0, /*descending=*/true);
  # 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将张量 input 拷贝到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    # 在指定设备上调用 torch::sort 函数，对 lazy_input 进行排序，descending 参数设为 true，得到 lazy_output
    auto lazy_output = torch::sort(lazy_input, /*dim=*/0, /*descending=*/true);
    # 对比两个结果的第一个值是否全部相等
    AllEqual(std::get<0>(output), std::get<0>(lazy_output));
    # 对比两个结果的第二个值是否全部相等
    AllEqual(std::get<1>(output), std::get<1>(lazy_output));
  });
}

# 测试用例：LazyOpsTest 中的 TestArgSort 函数
TEST_F(LazyOpsTest, TestArgSort) {
  # 创建一个形状为 (4, 5, 3) 的随机浮点数张量 a，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 迭代 k 从 1 到 3
  for (int k = 1; k <= 3; ++k) {
    # 迭代结束
    // 循环遍历三个维度（dim=0, 1, 2）
    for (int dim = 0; dim < 3; ++dim) {
      // 对于每个维度，分别进行升序和降序排序
      for (bool descending : {false, true}) {
        // 在给定维度上对张量 a 进行排序，返回排序后的索引张量 b
        torch::Tensor b = torch::argsort(a, dim, descending);
        // 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          // 将张量 a 复制到指定设备上，返回在该设备上的张量 lazy_a
          torch::Tensor lazy_a = CopyToDevice(a, device);
          // 在指定设备上对 lazy_a 进行排序，返回排序后的索引张量 lazy_b
          torch::Tensor lazy_b = torch::argsort(lazy_a, dim, descending);
          // 检查在不同设备上排序的结果是否相等
          AllEqual(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMin) {
  // 创建一个大小为 [2, 2] 的随机张量 a，数据类型为 float，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 [2, 2] 的随机张量 b，数据类型为 float，使用默认设备
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 和 b 的元素级最小值张量 c
  torch::Tensor c = torch::min(a, b);
  // 遍历每个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备上，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上计算 lazy_a 和 lazy_b 的元素级最小值 lazy_c
    torch::Tensor lazy_c = torch::min(lazy_a, lazy_b);
    // 检查 c 和 lazy_c 是否近似相等
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMax) {
  // 创建一个大小为 [2, 2] 的随机张量 a，数据类型为 float，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 [2, 2] 的随机张量 b，数据类型为 float，使用默认设备
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 和 b 的元素级最大值张量 c
  torch::Tensor c = torch::max(a, b);
  // 遍历每个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备上，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上计算 lazy_a 和 lazy_b 的元素级最大值 lazy_c
    torch::Tensor lazy_c = torch::max(lazy_a, lazy_b);
    // 检查 c 和 lazy_c 是否近似相等
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestUnaryMin) {
  // 创建一个大小为 [2, 2] 的随机张量 input，数据类型为 float，使用默认设备
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 input 的全局最小值张量 output
  torch::Tensor output = torch::min(input);
  // 遍历每个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 input 复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上计算 lazy_input 的全局最小值 lazy_output
    torch::Tensor lazy_output = torch::min(lazy_input);
    // 检查 output 和 lazy_output 是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestUnaryMax) {
  // 创建一个大小为 [2, 2] 的随机张量 input，数据类型为 float，使用默认设备
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 input 的全局最大值张量 output
  torch::Tensor output = torch::max(input);
  // 遍历每个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 input 复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上计算 lazy_input 的全局最大值 lazy_output
    torch::Tensor lazy_output = torch::max(lazy_input);
    // 检查 output 和 lazy_output 是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAll) {
  // 对于每种标量类型 scalar_type，依次执行以下操作
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型生成随机张量 a，使用相应的张量选项，数据类型和设备为默认设置
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 计算张量 a 是否所有元素都为 true 的张量 b
    torch::Tensor b = torch::all(a);
    // 遍历每个设备，并在每个设备上执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上，得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上计算 lazy_a 是否所有元素都为 true 的张量 lazy_b
      torch::Tensor lazy_b = torch::all(lazy_a);
      // 检查张量 b 和 lazy_b 是否相等
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAllDim) {
  // 创建一个大小为 [2, 3, 4] 的随机整数张量 a，数据类型为 byte，使用默认设备
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 获取张量 a 的维度数
  int rank = a.dim();
  // 对于张量 a 的每一个维度 dim，依次执行以下操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算在维度 dim 上是否所有元素都为 true 的张量 b
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/false);
    // 待续...
    # 对于每个设备执行以下操作，使用lambda表达式捕获当前设备
    ForEachDevice([&](const torch::Device& device) {
      # 将张量a复制到指定设备上，并返回一个延迟加载的张量lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      # 在指定维度dim上对lazy_a中的所有元素进行逻辑AND操作，并不保留维度信息，返回延迟加载的张量lazy_b
      torch::Tensor lazy_b = torch::all(lazy_a, dim, /*keepdim=*/false);
      # 检查张量b是否与lazy_b相等
      EqualValues(b, lazy_b);
    });
  }
}

# 定义测试用例 TestAllDimKeep，测试 torch::all 函数在保持维度的情况下的行为
TEST_F(LazyOpsTest, TestAllDimKeep) {
  # 创建一个形状为 (2, 3, 4)，值在 [0, 5) 范围内的随机整数张量 a
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  # 获取张量 a 的维度
  int rank = a.dim();
  # 遍历张量的每个维度
  for (int dim = -rank; dim < rank; ++dim) {
    # 对张量 a 沿指定维度 dim 执行 torch::all 操作，并保持维度
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/true);
    # 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      # 将张量 a 复制到指定设备，得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      # 对 lazy_a 沿指定维度 dim 执行 torch::all 操作，并保持维度
      torch::Tensor lazy_b = torch::all(lazy_a, dim, /*keepdim=*/true);
      # 断言张量 b 与 lazy_b 在值上的相等性
      EqualValues(b, lazy_b);
    });
  }
}

# 定义测试用例 TestAmax，测试 torch::amax 函数在计算最大值的行为
TEST_F(LazyOpsTest, TestAmax) {
  # 创建一个形状为 (4, 3, 4)，值在 [0, 1) 范围内的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 获取张量 input 的维度
  int rank = input.dim();
  # 对于每种 keepdim 的取值，分别进行测试
  for (bool keepdim : {false, true}) {
    # 遍历张量的每个维度
    for (int dim = -rank; dim < rank; ++dim) {
      # 对张量 input 沿指定维度 dim 执行 torch::amax 操作，并根据 keepdim 参数保持维度
      torch::Tensor values = torch::amax(input, {dim}, /*keepdim=*/keepdim);
      # 针对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        # 将张量 input 复制到指定设备，得到 lazy_input
        torch::Tensor lazy_input = CopyToDevice(input, device);
        # 对 lazy_input 沿指定维度 dim 执行 torch::amax 操作，并根据 keepdim 参数保持维度
        torch::Tensor lazy_values =
            torch::amax(lazy_input, {dim}, /*keepdim=*/keepdim);
        # 断言张量 values 与 lazy_values 在值上的近似相等性
        AllClose(values, lazy_values);
      });
    }
    # 遍历两两不同的维度组合
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        # 跳过相同维度或符合互斥条件的维度组合
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        # 对张量 input 沿指定维度 dim1 和 dim2 执行 torch::amax 操作，并根据 keepdim 参数保持维度
        torch::Tensor values =
            torch::amax(input, {dim1, dim2}, /*keepdim=*/keepdim);
        # 针对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          # 将张量 input 复制到指定设备，得到 lazy_input
          torch::Tensor lazy_input = CopyToDevice(input, device);
          # 对 lazy_input 沿指定维度 dim1 和 dim2 执行 torch::amax 操作，并根据 keepdim 参数保持维度
          torch::Tensor lazy_values =
              torch::amax(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          # 断言张量 values 与 lazy_values 在值上的近似相等性
          AllClose(values, lazy_values);
        });
      }
    }
  }
  # 断言预期计数器（aten::.*）没有变化
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  # 断言预期计数器（xla::amax）有变化
  ExpectCounterChanged("xla::amax", GetIgnoredCounters());
}

# 定义测试用例 TestAmin，测试 torch::amin 函数在计算最小值的行为
TEST_F(LazyOpsTest, TestAmin) {
  # 创建一个形状为 (4, 3, 4)，值在 [0, 1) 范围内的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 获取张量 input 的维度
  int rank = input.dim();
  # 对于每种 keepdim 的取值，分别进行测试
  for (bool keepdim : {false, true}) {
    # 遍历张量的每个维度
    for (int dim = -rank; dim < rank; ++dim) {
      # 对张量 input 沿指定维度 dim 执行 torch::amin 操作，并根据 keepdim 参数保持维度
      torch::Tensor values = torch::amin(input, {dim}, /*keepdim=*/keepdim);
      # 针对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        # 将张量 input 复制到指定设备，得到 lazy_input
        torch::Tensor lazy_input = CopyToDevice(input, device);
        # 对 lazy_input 沿指定维度 dim 执行 torch::amin 操作，并根据 keepdim 参数保持维度
        torch::Tensor lazy_values =
            torch::amin(lazy_input, {dim}, /*keepdim=*/keepdim);
        # 断言张量 values 与 lazy_values 在值上的近似相等性
        AllClose(values, lazy_values);
      });
    }
    // 外层循环遍历维度 dim1，从 -rank 到 rank-1
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      // 内层循环遍历维度 dim2，从 -rank 到 rank-1
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        // 跳过满足以下任一条件的情况：dim1 等于 dim2，或者 dim1 等于 rank + dim2，或者 dim2 等于 rank + dim1
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        // 在给定的维度 dim1 和 dim2 上计算输入张量 input 的最小值，并保持维度不变（如果 keepdim 为真）
        torch::Tensor values =
            torch::amin(input, {dim1, dim2}, /*keepdim=*/keepdim);
        // 对每个设备执行以下操作：
        ForEachDevice([&](const torch::Device& device) {
          // 将输入张量 input 复制到指定设备上
          torch::Tensor lazy_input = CopyToDevice(input, device);
          // 在给定的维度 dim1 和 dim2 上计算 lazy_input 的最小值，并保持维度不变（如果 keepdim 为真）
          torch::Tensor lazy_values =
              torch::amin(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          // 检查 values 和 lazy_values 是否近似相等
          AllClose(values, lazy_values);
        });
      }
    }
  }
  // 检查 "aten::.*" 的计数器是否没有发生变化
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 检查 "xla::amin" 的计数器是否发生了变化
  ExpectCounterChanged("xla::amin", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestAny) {
  // 遍历不同的数据类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据数据类型选择生成随机张量a
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 对张量a进行torch::any操作，得到张量b
    torch::Tensor b = torch::any(a);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量a复制到指定设备上得到lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上执行torch::any操作得到lazy_b
      torch::Tensor lazy_b = torch::any(lazy_a);
      // 检查lazy_b与b是否相等
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDim) {
  // 生成随机张量a
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 获取张量a的维度数
  int rank = a.dim();
  // 遍历所有维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 在指定维度上执行torch::any操作，得到张量b
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/false);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量a复制到指定设备上得到lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上执行torch::any操作得到lazy_b
      torch::Tensor lazy_b = torch::any(lazy_a, dim, /*keepdim=*/false);
      // 检查lazy_b与b是否相等
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDimKeep) {
  // 生成随机张量a
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 获取张量a的维度数
  int rank = a.dim();
  // 遍历所有维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 在指定维度上执行torch::any操作，得到张量b，并保持维度
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/true);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量a复制到指定设备上得到lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上执行torch::any操作得到lazy_b，并保持维度
      torch::Tensor lazy_b = torch::any(lazy_a, dim, /*keepdim=*/true);
      // 检查lazy_b与b是否相等
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMean) {
  // 生成随机张量a
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a的平均值得到张量b
  torch::Tensor b = torch::mean(a);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上执行torch::mean操作得到lazy_b
    torch::Tensor lazy_b = torch::mean(lazy_a);
    // 检查lazy_b的大小是否与b相等
    ASSERT_EQ(b.sizes(), lazy_b.sizes());
    // 检查lazy_b与b的近似性
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanCast) {
  // 生成随机张量a
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a的平均值并将结果转换为torch::kDouble类型得到张量b
  torch::Tensor b = torch::mean(a, torch::kDouble);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上执行torch::mean操作并转换为torch::kDouble类型得到lazy_b
    torch::Tensor lazy_b = torch::mean(lazy_a, torch::kDouble);
    // 检查lazy_b与b的近似性
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanInDim) {
  // 生成随机张量a
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量a的维度数
  int rank = a.dim();
  // 遍历所有维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 在指定维度上计算张量a的平均值得到张量b
    torch::Tensor b = torch::mean(a, {dim});
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量a复制到指定设备上得到lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上执行torch::mean操作得到lazy_b
      torch::Tensor lazy_b = torch::mean(lazy_a, {dim});
    ForEachDevice([&](const torch::Device& device) {
        // 对每个设备执行以下操作
        torch::Tensor lazy_a = CopyToDevice(a, device);
        // 将张量 a 复制到当前设备，并创建 lazy_a 引用
        torch::Tensor lazy_b = torch::mean(lazy_a, {dim});
        // 计算 lazy_a 张量沿指定维度 dim 的均值，结果存储在 lazy_b 中
        AllClose(b, lazy_b);
        // 检查张量 b 是否与 lazy_b 在数值上全部接近
    });
}
// 结束 ForEachDevice 循环体
}

TEST_F(LazyOpsTest, TestMeanInDims) {
  // 生成一个指定形状的随机张量 a，数据类型为 float，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历两种维度组合：{0, 1} 和 {-3, -2}
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 计算张量 a 在指定维度 dims 上的平均值得到张量 b
    torch::Tensor b = torch::mean(a, dims);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备，并命名为 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算 lazy_a 在指定维度 dims 上的平均值得到 lazy_b
      torch::Tensor lazy_b = torch::mean(lazy_a, dims);
      // 检查张量 b 和 lazy_b 是否接近
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimsKeepCast) {
  // 生成一个指定形状的随机张量 a，数据类型为 float，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历两种维度组合：{0, 1} 和 {-3, -2}
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 计算张量 a 在指定维度 dims 上的加权平均值，保持类型转换为 double，得到张量 b
    torch::Tensor b = torch::mean(a, dims, true, torch::kDouble);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备，并命名为 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算 lazy_a 在指定维度 dims 上的加权平均值，保持类型转换为 double，得到 lazy_b
      torch::Tensor lazy_b = torch::mean(lazy_a, dims, true, torch::kDouble);
      // 检查张量 b 和 lazy_b 是否接近
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimOut) {
  // 生成一个指定形状的随机张量 a，数据类型为 float，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量 a 的维度数
  int rank = a.dim();
  // 对每个维度进行循环处理
  for (int dim = -rank; dim < rank; ++dim) {
    // 生成一个指定形状的空张量 b，数据类型为 float，存储在默认设备上
    torch::Tensor b = torch::empty(
        {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 在指定维度 dim 上计算张量 a 的平均值，并存储到张量 b
    torch::mean_out(b, a, {dim});
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备，并命名为 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 生成一个与 lazy_a 相同形状的空张量 lazy_b，数据类型与 lazy_a 保持一致
      torch::Tensor lazy_b = torch::empty({4, 4}, lazy_a.options());
      // 在指定维度 dim 上计算 lazy_a 的平均值，并存储到 lazy_b
      torch::mean_out(lazy_b, lazy_a, {dim});
      // 检查张量 b 和 lazy_b 是否接近
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStd) {
  // 生成一个指定形状的随机张量 a，数据类型为 float，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历是否使用无偏估计的选项 true 和 false
  for (auto unbiased : {true, false}) {
    // 计算张量 a 的标准差，根据 unbiased 参数决定是否使用无偏估计，得到张量 b
    torch::Tensor b = torch::std(a, unbiased);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备，并命名为 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算 lazy_a 的标准差，根据 unbiased 参数决定是否使用无偏估计，得到 lazy_b
      torch::Tensor lazy_b = torch::std(lazy_a, unbiased);
      // 检查张量 b 和 lazy_b 是否接近
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStdInDim) {
  // 生成一个指定形状的随机张量 a，数据类型为 float，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量 a 的维度数
  int rank = a.dim();
  // 遍历是否使用无偏估计的选项 true 和 false
  for (auto unbiased : {true, false}) {
    // 遍历是否保持维度的选项 true 和 false
    for (auto keepdim : {true, false}) {
      // 对每个维度进行循环处理
      for (int dim = -rank; dim < rank; ++dim) {
        // 计算张量 a 在指定维度 dim 上的标准差，根据 unbiased 和 keepdim 参数进行计算，得到张量 b
        torch::Tensor b = torch::std(a, {dim}, unbiased, keepdim);
        // 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          // 将张量 a 复制到指定设备，并命名为 lazy_a
          torch::Tensor lazy_a = CopyToDevice(a, device);
          // 计算 lazy_a 在指定维度 dim 上的标准差，根据 unbiased 和 keepdim 参数进行计算，得到 lazy_b
          torch::Tensor lazy_b = torch::std(lazy_a, {dim}, unbiased, keepdim);
          // 检查张量 b 和 lazy_b 是否接近
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestStdWithCorrection) {
  // 生成一个指定形状的随机张量 a，数据类型为 float，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历修正项的可能值，包括 1、2 和未指定修正
  std::optional<c10::Scalar> corrections[] = {1, 2, c10::nullopt};
  for (const auto& correction : corrections) {
    // 待完成
    // 对于每个布尔值 keepdim（true 或 false），执行以下操作
    for (auto keepdim : {true, false}) {
      // 对于每个维度 dim，依次处理两个维度列表 {0, 1} 和 {-3, -2}
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        // 计算张量 a 沿指定维度 dim 的标准差，并根据 keepdim 是否保持维度
        torch::Tensor b = torch::std(a, dim, correction, keepdim);

        // 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          // 将张量 a 复制到指定设备上
          torch::Tensor lazy_a = CopyToDevice(a, device);
          // 计算复制到设备上的张量 lazy_a 沿指定维度 dim 的标准差，并根据 keepdim 是否保持维度
          torch::Tensor lazy_b = torch::std(lazy_a, dim, correction, keepdim);
          // 检查张量 b 和 lazy_b 是否在设备上全部相等
          AllClose(b, lazy_b);
        });
      }
    }
  }


这段代码的主要作用是针对张量 `a` 的标准差计算，对不同的维度组合和保留维度的设置（`keepdim`）进行多次计算和比较，确保在不同设备上的计算结果一致。
}

TEST_F(LazyOpsTest, TestStdMeanWithCorrection) {
  // 创建一个形状为{4, 3, 4}的随机张量a，使用浮点数选项，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义包含可选标量的数组，这里包括整数1、2以及空值
  std::optional<c10::Scalar> corrections[] = {1, 2, c10::nullopt};
  // 对每个correction进行迭代
  for (const auto& correction : corrections) {
    // 对于每个keepdim值为true和false进行迭代
    for (auto keepdim : {true, false}) {
      // 对于每个维度dim进行迭代，这里包括{0, 1}和{-3, -2}两组
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        // 计算张量a在给定维度dim上的标准差和均值
        auto b = torch::std_mean(a, dim, correction, keepdim);
        // 对每个设备执行操作
        ForEachDevice([&](const torch::Device& device) {
          // 将张量a复制到指定设备上
          torch::Tensor lazy_a = CopyToDevice(a, device);
          // 计算在指定设备上张量lazy_a在维度dim上的标准差和均值
          auto lazy_b = torch::std_mean(lazy_a, dim, correction, keepdim);
          // 检查两组结果是否全部接近
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSum) {
  // 创建一个形状为{4, 3, 4}的随机张量a，使用浮点数选项，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a的总和
  torch::Tensor b = torch::sum(a);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算在指定设备上张量lazy_a的总和
    torch::Tensor lazy_b = torch::sum(lazy_a);
    // 检查总和结果是否全部接近
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumCast) {
  // 创建一个形状为{4, 3, 4}的随机张量a，使用浮点数选项，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a在转换为双精度后的总和
  torch::Tensor b = torch::sum(a, torch::kDouble);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算在指定设备上张量lazy_a在转换为双精度后的总和
    torch::Tensor lazy_b = torch::sum(lazy_a, torch::kDouble);
    // 检查总和结果是否全部接近
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumU8) {
  // 创建一个形状为{256}的全一张量a，数据类型为字节类型，放置在默认设备上
  torch::Tensor a = torch::ones(
      {256}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 计算张量a的总和
  torch::Tensor b = torch::sum(a);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算在指定设备上张量lazy_a的总和
    torch::Tensor lazy_b = torch::sum(lazy_a);
    // 检查两组结果是否全部相等
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumInDim) {
  // 创建一个形状为{4, 3, 4}的随机张量a，使用浮点数选项，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量a的维度数
  int rank = a.dim();
  // 对每个维度dim进行迭代，包括从-rank到rank-1的值
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量a在给定维度dim上的总和
    torch::Tensor b = torch::sum(a, {dim});
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量a复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算在指定设备上张量lazy_a在给定维度dim上的总和
      torch::Tensor lazy_b = torch::sum(lazy_a, {dim});
      // 检查两组结果是否全部接近
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDims) {
  // 创建一个形状为{4, 3, 4}的随机张量a，使用浮点数选项，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每组维度dims进行迭代，这里包括{{0, 1}, {-3, -2}}两组
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 计算张量a在给定维度dims上的总和
    torch::Tensor b = torch::sum(a, dims);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量a复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算在指定设备上张量lazy_a在给定维度dims上的总和
      torch::Tensor lazy_b = torch::sum(lazy_a, dims);
      // 检查两组结果是否全部接近
      AllClose(b, lazy_b);
    });
  }
}
TEST_F(LazyOpsTest, TestSumInDimsKeep) {
  // 创建一个形状为 {4, 3, 4} 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度组合 {{0, 1}, {-3, -2}}
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 对张量 a 按指定维度 dims 求和，并保持维度不变，结果保存在张量 b 中
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true);
    // 对于每个设备，将张量 a 复制到设备上，然后在该设备上按指定维度 dims 求和，保持维度不变，结果保存在 lazy_b 中，并验证其与 b 是否近似相等
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::sum(lazy_a, dims, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDimsKeepCast) {
  // 创建一个形状为 {4, 3, 4} 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度组合 {{0, 1}, {-3, -2}}
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 对张量 a 按指定维度 dims 求和，并保持维度不变，数据类型转换为 double，结果保存在张量 b 中
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true, torch::kDouble);
    // 对于每个设备，将张量 a 复制到设备上，然后在该设备上按指定维度 dims 求和，保持维度不变，数据类型转换为 double，结果保存在 lazy_b 中，并验证其与 b 是否近似相等
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::sum(lazy_a, dims, /*keepdim=*/true, torch::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVar) {
  // 创建一个形状为 {4, 3, 4} 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历 unbiased 取值 true 和 false
  for (bool unbiased : {true, false}) {
    // 计算张量 a 的方差，结果保存在张量 b 中
    torch::Tensor b = torch::var(a, unbiased);
    // 对于每个设备，将张量 a 复制到设备上，然后在该设备上计算方差，结果保存在 lazy_b 中，并验证其与 b 是否近似相等
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::var(lazy_a, unbiased);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVarWithDim) {
  // 创建一个形状为 {4, 3, 4} 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度组合 {{0, 1}, {-3, -2}}
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 遍历 keepDim 取值 true 和 false
    for (bool keepDim : {true, false}) {
      // 遍历 unbiased 取值 true 和 false
      for (bool unbiased : {true, false}) {
        // 计算张量 a 按指定维度 dims 的方差，结果保存在张量 b 中
        torch::Tensor b = torch::var(a, dims, unbiased, keepDim);
        // 对于每个设备，将张量 a 复制到设备上，然后在该设备上按指定维度 dims 计算方差，结果保存在 lazy_b 中，并验证其与 b 是否近似相等
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::var(lazy_a, dims, unbiased, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestVarWithCorrection) {
  // 创建一个形状为 {4, 3, 4} 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义修正参数的可选值
  std::optional<c10::Scalar> corrections[] = {1, 2, c10::nullopt};
  // 遍历维度组合 {{0, 1}, {-3, -2}}
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 遍历 keepDim 取值 true 和 false
    for (bool keepDim : {true, false}) {
      // 遍历修正参数的可选值
      for (const auto& correction : corrections) {
        // 计算张量 a 按指定维度 dim 的方差，应用指定的修正参数，结果保存在张量 b 中
        torch::Tensor b = torch::var(a, dim, correction, keepDim);
        // 对于每个设备，将张量 a 复制到设备上，然后在该设备上按指定维度 dim 计算方差，应用指定的修正参数，结果保存在 lazy_b 中，并验证其与 b 是否近似相等
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::var(lazy_a, dim, correction, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
  // 预期操作计数器 "aten::.*" 不会更改，使用 GetIgnoredCounters() 获取忽略的计数器
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 预期操作计数器 "lazy::var" 将会更改，使用 GetIgnoredCounters() 获取忽略的计数器
  ExpectCounterChanged("lazy::var", GetIgnoredCounters());
}
// 在 LazyOpsTest 测试环境下，测试带有修正的变量均值计算
TEST_F(LazyOpsTest, TestVarMeanWithCorrection) {
  // 创建一个大小为 [4, 3, 4] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个包含可选修正值的数组
  std::optional<c10::Scalar> corrections[] = {1, 2, c10::nullopt};
  // 遍历两个维度组合
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 遍历修正值数组
    for (const auto& correction : corrections) {
      // 遍历 keepdim 的两个值：true 和 false
      for (auto keepdim : {true, false}) {
        // 计算张量 a 在指定维度 dim 上的方差均值，考虑修正值和是否保持维度
        auto b = torch::var_mean(a, dim, correction, keepdim);
        // 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          // 将张量 a 复制到指定设备上
          torch::Tensor lazy_a = CopyToDevice(a, device);
          // 在指定设备上计算张量 lazy_a 在维度 dim 上的方差均值，考虑修正值和是否保持维度
          auto lazy_b = torch::var_mean(lazy_a, dim, correction, keepdim);
          // 检查计算结果是否接近
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          // 检查索引是否相等
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

// 在 LazyOpsTest 测试环境下，测试在指定维度上的最大值
TEST_F(LazyOpsTest, TestMaxInDim) {
  // 创建一个大小为 [4, 3, 4] 的随机输入张量 input，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算输入张量的秩
  int rank = input.dim();
  // 遍历所有维度 dim 的可能性
  for (int dim = -rank; dim < rank; ++dim) {
    // 遍历是否保持维度的两种情况：false 和 true
    for (bool keepdim : {false, true}) {
      // 计算输入张量在指定维度 dim 上的最大值及其索引
      auto values_indices = torch::max(input, dim, /*keepdim=*/keepdim);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将输入张量 input 复制到指定设备上
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 计算在指定设备上输入张量 lazy_input 在维度 dim 上的最大值及其索引
        auto lazy_values_indices =
            torch::max(lazy_input, dim, /*keepdim=*/keepdim);
        // 检查计算结果是否接近
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        // 检查索引是否全部相等
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

// 在 LazyOpsTest 测试环境下，测试在指定维度上的最小值
TEST_F(LazyOpsTest, TestMinInDim) {
  // 创建一个大小为 [4, 3, 4] 的随机输入张量 input，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算输入张量的秩
  int rank = input.dim();
  // 遍历所有维度 dim 的可能性
  for (int dim = -rank; dim < rank; ++dim) {
    // 遍历是否保持维度的两种情况：false 和 true
    for (bool keepdim : {false, true}) {
      // 计算输入张量在指定维度 dim 上的最小值及其索引
      auto values_indices = torch::min(input, dim, /*keepdim=*/keepdim);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将输入张量 input 复制到指定设备上
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 计算在指定设备上输入张量 lazy_input 在维度 dim 上的最小值及其索引
        auto lazy_values_indices =
            torch::min(lazy_input, dim, /*keepdim=*/keepdim);
        // 检查计算结果是否接近
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        // 检查索引是否全部相等
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

// 在 LazyOpsTest 测试环境下，测试张量的范数计算
TEST_F(LazyOpsTest, TestNorm) {
  // 创建一个大小为 [4, 3, 4] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的范数
  torch::Tensor b = torch::norm(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算在指定设备上张量 lazy_a 的范数
    torch::Tensor lazy_b = torch::norm(lazy_a);
    // 检查计算结果是否接近
    AllClose(b, lazy_b);
  });
}

// 在 LazyOpsTest 测试环境下，测试在指定维度上的范数计算
TEST_F(LazyOpsTest, TestNormInDim) {
  // 创建一个大小为 [4, 3, 4] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历指定的维度 dim：1 和 -2
  for (int dim : {1, -2}) {
    // 计算张量 a 在指定维度 dim 上的 L2 范数，不保持维度
    torch::Tensor b = torch::norm(a, 2, {dim}, /*keepdim=*/false);
    // 省略对每个设备的迭代，因为代码截断
}
    # 对于每个设备执行以下操作，使用 lambda 表达式捕获当前设备对象
    ForEachDevice([&](const torch::Device& device) {
      # 将张量 a 复制到当前设备上，返回一个延迟执行的张量 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      # 计算 lazy_a 的 L2 范数，沿指定维度 dim，不保持维度，返回延迟执行的张量 lazy_b
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, {dim}, /*keepdim=*/false);
      # 检查张量 b 是否与 lazy_b 在数值上全部相等
      AllClose(b, lazy_b);
    });
  }
}

# 定义测试用例 LazyOpsTest 中的 TestNormInDims 方法
TEST_F(LazyOpsTest, TestNormInDims) {
  # 创建一个形状为 {4, 3, 4} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 遍历两个维度选项，分别为 {1, 2} 和 {-2, -1}
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    # 计算张量 a 沿指定维度 dims 的 L2 范数，不保持维度
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/false);
    # 对于每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      # 将张量 a 复制到指定设备，得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      # 计算 lazy_a 沿指定维度 dims 的 L2 范数，不保持维度
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, dims, /*keepdim=*/false);
      # 检查 lazy_b 是否与 b 近似相等
      AllClose(b, lazy_b);
    });
  }
}

# 定义测试用例 LazyOpsTest 中的 TestNormInDimsKeep 方法
TEST_F(LazyOpsTest, TestNormInDimsKeep) {
  # 创建一个形状为 {4, 3, 4} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 遍历两个维度选项，分别为 {1, 2} 和 {-2, -1}
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    # 计算张量 a 沿指定维度 dims 的 L2 范数，保持维度
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/true);
    # 对于每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      # 将张量 a 复制到指定设备，得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      # 计算 lazy_a 沿指定维度 dims 的 L2 范数，保持维度
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, dims, /*keepdim=*/true);
      # 检查 lazy_b 是否与 b 近似相等
      AllClose(b, lazy_b);
    });
  }
}

# 定义测试用例 LazyOpsTest 中的 TestNormalTwoTensor 方法
TEST_F(LazyOpsTest, TestNormalTwoTensor) {
  # 创建一个形状为 {10, 10, 10} 的全零张量 mean，数据类型为 kFloat
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  # 创建一个形状为 {10, 10, 10} 的全一张量 std，数据类型为 kFloat
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  # 对于每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将张量 mean 复制到指定设备，得到 lazy_mean
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    # 将张量 std 复制到指定设备，得到 lazy_std
    at::Tensor lazy_std = CopyToDevice(std, device);
    # 在指定设备上生成一个正态分布的张量 lazy_normal
    at::Tensor lazy_normal = at::normal(lazy_mean, lazy_std);
    # 计算 lazy_normal 的均值和标准差，并转换为双精度数值
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    # 断言均值落在 [-0.06, 0.06] 区间内
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    # 断言标准差落在 [0.94, 1.06] 区间内
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

# 定义测试用例 LazyOpsTest 中的 TestNormalDoubleMean 方法
TEST_F(LazyOpsTest, TestNormalDoubleMean) {
  # 创建一个形状为 {10, 10, 10} 的全一张量 std，数据类型为 kFloat
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  # 对于每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将张量 std 复制到指定设备，得到 lazy_std
    at::Tensor lazy_std = CopyToDevice(std, device);
    # 在指定设备上生成一个正态分布的张量 lazy_normal，均值为 0，标准差为 lazy_std
    at::Tensor lazy_normal = at::normal(0, lazy_std);
    # 计算 lazy_normal 的均值和标准差，并转换为双精度数值
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    # 断言均值落在 [-0.06, 0.06] 区间内
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    # 断言标准差落在 [0.94, 1.06] 区间内
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

# 定义测试用例 LazyOpsTest 中的 TestNormalDoubleStd 方法
TEST_F(LazyOpsTest, TestNormalDoubleStd) {
  # 创建一个形状为 {10, 10, 10} 的全零张量 mean，数据类型为 kFloat
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  # 对于每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将张量 mean 复制到指定设备，得到 lazy_mean
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    # 在指定设备上生成一个正态分布的张量 lazy_normal，均值为 lazy_mean，标准差为 1
    at::Tensor lazy_normal = at::normal(lazy_mean, 1);
    # 计算 lazy_normal 的均值和标准差，并转换为双精度数值
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    # 断言均值落在 [-0.06, 0.06] 区间内
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    # 断言标准差落在 [0.94, 1.06] 区间内
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

# 定义测试用例 LazyOpsTest 中的 TestNormalInPlace 方法
TEST_F(LazyOpsTest, TestNormalInPlace) {
  # 创建一个形状为 {10, 10, 10} 的全零张量 a，数据类型为 kFloat
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  # 对于每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将张量 a 复制到指定设备，得到 lazy_a
    at::Tensor lazy_a = CopyToDevice(a, device);
    # 在 lazy_a 上就地生成一个正态分布，均值为 0，标准差为 1
    lazy_a.normal_(/*mean=*/0, /*std
    // 计算 lazy_a 的均值并转换为双精度浮点数
    double res_mean = lazy_a.mean().item().toDouble();
    // 计算 lazy_a 的标准差并转换为双精度浮点数
    double res_std = lazy_a.std().item().toDouble();
    // 断言均值 res_mean 大于 -0.06
    EXPECT_GT(res_mean, -0.06);
    // 断言均值 res_mean 小于 0.06
    EXPECT_LT(res_mean, 0.06);
    // 断言标准差 res_std 大于 0.94
    EXPECT_GT(res_std, 0.94);
    // 断言标准差 res_std 小于 1.06
    EXPECT_LT(res_std, 1.06);
}

TEST_F(LazyOpsTest, TestUniformInPlace) {
  // 定义误差范围为1e-3
  const double eps = 1e-3;
  // 创建一个全零的浮点型Tensor，大小为10x10x10
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将Tensor复制到指定设备，并返回懒惰Tensor
    at::Tensor lazy_a = CopyToDevice(a, device);
    // 在懒惰Tensor上执行均匀分布的原地操作，范围从0到1
    lazy_a.uniform_(/*from=*/0, /*to=*/1);
    // 将懒惰Tensor转移到CPU上
    at::Tensor cpu_a = ToCpuTensor(lazy_a);
    // 计算CPU上Tensor的最小值，并转换为double类型
    double res_min = cpu_a.min().item().toDouble();
    // 计算CPU上Tensor的最大值，并转换为double类型
    double res_max = cpu_a.max().item().toDouble();
    // 断言：最小值应大于0.0减去误差范围eps
    EXPECT_GT(res_min, 0.0 - eps);
    // 断言：最大值应小于1.0加上误差范围eps
    EXPECT_LT(res_max, 1.0 + eps);
  });
}

TEST_F(LazyOpsTest, TestRandomInPlace) {
  // 对多个数据类型进行循环测试
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 定义误差范围为0.2
    const double eps = 0.2;
    // 创建一个全零的Tensor，大小为10x10x10，数据类型由dtype决定
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将Tensor复制到指定设备，并返回懒惰Tensor
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在懒惰Tensor上执行随机分布的原地操作，范围从0到10
      lazy_a.random_(/*from=*/0, /*to=*/10);
      // 计算懒惰Tensor的元素和的平均值，并转换为double类型
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      // 计算懒惰Tensor的最小值，并转换为double类型
      double res_min = lazy_a.min().item().toDouble();
      // 计算懒惰Tensor的最大值，并转换为double类型
      double res_max = lazy_a.max().item().toDouble();
      // 断言：平均值应大于4.5减去误差范围eps
      EXPECT_GT(res_mean, 4.5 - eps);
      // 断言：平均值应小于4.5加上误差范围eps
      EXPECT_LT(res_mean, 4.5 + eps);
      // 断言：最小值应为0.0
      EXPECT_EQ(res_min, 0.0);
      // 断言：最大值应为9.0
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefaultFrom) {
  // 对多个数据类型进行循环测试
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 定义误差范围为0.2
    const double eps = 0.2;
    // 创建一个全零的Tensor，大小为10x10x10，数据类型由dtype决定
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将Tensor复制到指定设备，并返回懒惰Tensor
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在懒惰Tensor上执行随机分布的原地操作，范围从0到10，默认起始值为0
      lazy_a.random_(/*to=*/10);
      // 计算懒惰Tensor的元素和的平均值，并转换为double类型
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      // 计算懒惰Tensor的最小值，并转换为double类型
      double res_min = lazy_a.min().item().toDouble();
      // 计算懒惰Tensor的最大值，并转换为double类型
      double res_max = lazy_a.max().item().toDouble();
      // 断言：平均值应大于4.5减去误差范围eps
      EXPECT_GT(res_mean, 4.5 - eps);
      // 断言：平均值应小于4.5加上误差范围eps
      EXPECT_LT(res_mean, 4.5 + eps);
      // 断言：最小值应为0.0
      EXPECT_EQ(res_min, 0.0);
      // 断言：最大值应为9.0
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefault) {
  // 对多个数据类型进行循环测试
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 创建一个全零的Tensor，大小为10，数据类型由dtype决定
    auto input = torch::zeros({10}, torch::TensorOptions(dtype));
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将Tensor复制到指定设备，并返回懒惰Tensor
      auto lazyInput = CopyToDevice(input, device);
      // 在懒惰Tensor上执行默认的随机分布的原地操作
      lazyInput.random_();
      // 将懒惰Tensor转移到CPU上
      auto output = ToCpuTensor(lazyInput);
      // 断言：输出Tensor中所有元素都不等于输入Tensor中的任何元素
      EXPECT_TRUE(torch::all(output.ne(input)).item<bool>());
    });
  }
}

TEST_F(LazyOpsTest, TestNormGeneral) {
  // 生成一个随机的浮点型Tensor，大小为4x3x4
  torch::Tensor a = torch::randn(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对Tensor执行3.5范数计算
  torch::Tensor b = torch::norm(a, 3.5);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将Tensor复制到指定设备，并返回懒惰Tensor
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 使用 PyTorch 的 C++ 扩展模块，计算 lazy_a 的 3.5 范数，返回结果为 Tensor 类型，存储在 lazy_b 中
    torch::Tensor lazy_b = torch::norm(lazy_a, 3.5);
    // 使用 AllClose 函数检查 Tensor b 是否与 lazy_b 在数值上相近
    AllClose(b, lazy_b);
}

// 定义测试用例 LazyOpsTest 中的 TestNormNuclear 方法
TEST_F(LazyOpsTest, TestNormNuclear) {
  // 创建大小为 4x3x4 的随机浮点数张量 a
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的 L1 范数并存储在张量 b 中
  torch::Tensor b = torch::norm(a, 1);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并存储在 lazy_a 中
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的 L1 范数并存储在 lazy_b 中
    torch::Tensor lazy_b = torch::norm(lazy_a, 1);
    // 检查张量 b 和 lazy_b 是否在所有元素上近似相等
    AllClose(b, lazy_b);
  });
}

// 定义测试用例 LazyOpsTest 中的 TestFrobeniusNormInDim 方法
TEST_F(LazyOpsTest, TestFrobeniusNormInDim) {
  // 创建大小为 4x3x4 的随机浮点数张量 a
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度 1 和 -2
  for (int dim : {1, -2}) {
    // 计算张量 a 沿指定维度的 Frobenius 范数并存储在张量 b 中
    torch::Tensor b = torch::frobenius_norm(a, {dim}, /*keepdim=*/false);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备，并存储在 lazy_a 中
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算 lazy_a 沿指定维度的 Frobenius 范数并存储在 lazy_b 中
      torch::Tensor lazy_b =
          torch::frobenius_norm(lazy_a, {dim}, /*keepdim=*/false);
      // 检查张量 b 和 lazy_b 是否在所有元素上近似相等
      AllClose(b, lazy_b);
    });
  }
}

// 定义测试用例 LazyOpsTest 中的 TestFrobeniusNormInDims 方法
TEST_F(LazyOpsTest, TestFrobeniusNormInDims) {
  // 创建大小为 4x3x4 的随机浮点数张量 a
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历不同的维度组合
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    // 计算张量 a 沿指定维度的 Frobenius 范数并存储在张量 b 中
    torch::Tensor b = torch::frobenius_norm(a, dims, /*keepdim=*/false);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备，并存储在 lazy_a 中
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算 lazy_a 沿指定维度的 Frobenius 范数并存储在 lazy_b 中
      torch::Tensor lazy_b =
          torch::frobenius_norm(lazy_a, dims, /*keepdim=*/false);
      // 检查张量 b 和 lazy_b 是否在所有元素上近似相等
      AllClose(b, lazy_b);
    });
  }
}

// 定义测试用例 LazyOpsTest 中的 TestGroupNorm 方法
TEST_F(LazyOpsTest, TestGroupNorm) {
  // 设置通道数为 6
  int num_channels = 6;
  // 创建大小为 20x6x10x10 的随机浮点数输入张量 input
  torch::Tensor input = torch::rand(
      {20, num_channels, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 num_channels 的随机浮点数权重张量 weight
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 num_channels 的随机浮点数偏置张量 bias
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置 epsilon 为 1e-05
  double eps = 1e-05;
  // 遍历不同的组数
  for (int num_groups : {3, 6, 1}) {
    // 计算输入张量 input 的 group normalization，并存储在输出张量 output 中
    torch::Tensor output = torch::group_norm(
        input,
        num_groups,
        weight,
        bias,
        eps,
        /*cudnn_enabled=*/false);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量 input 复制到指定设备，并存储在 lazy_input 中
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 将权重张量 weight 复制到指定设备，并存储在 lazy_weight 中
      torch::Tensor lazy_weight = CopyToDevice(weight, device);
      // 将偏置张量 bias 复制到指定设备，并存储在 lazy_bias 中
      torch::Tensor lazy_bias = CopyToDevice(bias, device);
      // 计算 lazy_input 的 group normalization，并存储在 lazy_output 中
      torch::Tensor lazy_output = torch::group_norm(
          lazy_input,
          num_groups,
          lazy_weight,
          lazy_bias,
          eps,
          /*cudnn_enabled=*/false);
      // 检查张量 output 和 lazy_output 是否在指定的相对误差和绝对误差下相等
      AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}
TEST_F(LazyOpsTest, TestGroupNormBackward) {
  // 定义通道数
  int num_channels = 6;
  // 创建随机张量作为输入，并指定设备和梯度跟踪
  torch::Tensor input = torch::rand(
      {2, num_channels, 5, 5},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  // 创建随机张量作为权重，并指定设备和梯度跟踪
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  // 创建随机张量作为偏置，并指定设备和梯度跟踪
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  // 定义 epsilon 值
  double eps = 1e-05;
  // 对未定义的权重进行循环测试
  for (bool undef_weight : {true, false}) {
    // 对不同的组数进行循环测试
    for (int num_groups : {3, 6, 1}) {
      // 定义测试函数，用于计算 group normalization
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::group_norm(
            /*input=*/inputs[0],
            num_groups,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      // 定义未定义的张量
      torch::Tensor undef;
      // 对每个设备执行测试反向传播
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-3,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestInstanceNorm) {
  // 定义批次大小和通道数
  int batch = 5;
  int num_channels = 20;
  // 创建随机张量作为输入，并指定设备
  torch::Tensor input = torch::rand(
      {batch, num_channels, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机张量作为权重，并指定设备
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机张量作为偏置，并指定设备
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建全零张量作为 running mean，并指定设备
  torch::Tensor running_mean = torch::zeros(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建全一张量作为 running variance，并指定设备
  torch::Tensor running_var = torch::ones(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 momentum 值
  double momentum = 0.1;
  // 定义 epsilon 值
  double eps = 1e-05;
  // 执行 instance normalization，并指定是否使用输入统计数据，关闭 cuDNN 加速
  torch::Tensor output = torch::instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      /*use_input_stats=*/true,
      momentum,
      eps,
      /*cudnn_enabled=*/false);
  // 对每个设备执行操作的懒惰版本
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    torch::Tensor lazy_bias = CopyToDevice(bias, device);
    torch::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
    torch::Tensor lazy_running_var = CopyToDevice(running_var, device);
    torch::Tensor lazy_output = torch::instance_norm(
        lazy_input,
        lazy_weight,
        lazy_bias,
        lazy_running_mean,
        lazy_running_var,
        /*use_input_stats=*/true,
        momentum,
        eps,
        /*cudnn_enabled=*/false);
    // 调用 AllClose 函数，比较 output 和 lazy_output 的近似程度
    AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

# 定义测试用例 TestLayerNorm，用于测试 layer_norm 函数
TEST_F(LazyOpsTest, TestLayerNorm) {
  # 创建一个大小为 [20, 10, 10, 10] 的随机张量 input，使用默认设备（DefaultDevice）
  torch::Tensor input = torch::rand(
      {20, 10, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  # 定义 eps（epsilon）为 1e-05
  double eps = 1e-05;
  # 定义一个未定义的张量 undef
  torch::Tensor undef;
  # 对于每个 undef_weight 取值为 true 和 false 的情况
  for (bool undef_weight : {true, false}) {
    # 对于每个 normalized_size 取值为 2 和 3 的情况
    for (int64_t normalized_size : {2, 3}) {
      # 创建一个 normalized_shape 向量，元素均为 10，长度为 normalized_size
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      # 创建一个大小为 normalized_shape 的随机权重张量 weight
      torch::Tensor weight = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      # 创建一个大小为 normalized_shape 的随机偏置张量 bias
      torch::Tensor bias = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      # 调用 torch::layer_norm 函数计算输出张量 output
      torch::Tensor output = torch::layer_norm(
          input,
          normalized_shape,
          undef_weight ? undef : weight,
          undef_weight ? undef : bias,
          eps,
          /*cudnn_enabled=*/false);
      # 对于每个设备，复制 input、weight 和 bias 到该设备，计算 lazy_output
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_bias =
            undef_weight ? undef : CopyToDevice(bias, device);
        torch::Tensor lazy_output = torch::layer_norm(
            lazy_input,
            normalized_shape,
            lazy_weight,
            lazy_bias,
            eps,
            /*cudnn_enabled=*/false);
        # 检查 output 和 lazy_output 是否在一定的误差范围内相等
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

# 定义测试用例 TestLayerNormBackward，用于测试 layer_norm 函数的反向传播
TEST_F(LazyOpsTest, TestLayerNormBackward) {
  # 创建一个大小为 [2, 3, 3, 3] 的随机张量 input，使用默认设备（DefaultDevice），并要求计算梯度
  torch::Tensor input = torch::rand(
      {2, 3, 3, 3},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  # 定义 eps（epsilon）为 1e-05
  double eps = 1e-05;
  # 对于每个 undef_weight 取值为 true 和 false 的情况
  for (bool undef_weight : {true, false}) {
    # 对于每个 normalized_size 取值为 2 和 3 的情况
    for (int64_t normalized_size : {2, 3}) {
      # 创建一个 normalized_shape 向量，元素均为 3，长度为 normalized_size
      std::vector<int64_t> normalized_shape(normalized_size, 3);
      # 定义一个 lambda 函数 testfn，用于执行 layer_norm 函数
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::layer_norm(
            /*input=*/inputs[0],
            normalized_shape,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      # 创建一个大小为 normalized_shape 的随机权重张量 weight，并要求计算梯度
      torch::Tensor weight = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      # 创建一个大小为 normalized_shape 的随机偏置张量 bias，并要求计算梯度
      torch::Tensor bias = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      # 定义一个未定义的张量 undef
      torch::Tensor undef;
      # 对于每个设备，执行反向传播测试
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-4,
            /*derivative_level=*/2);
      });
    }
  }
}
TEST_F(LazyOpsTest, TestNuclearNorm) {
  // 创建一个大小为 (4, 3) 的随机张量 a，使用默认设备（DefaultDevice）
  torch::Tensor a = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的核范数，并存储在张量 b 中
  torch::Tensor b = torch::nuclear_norm(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并存储在 lazy_a 中
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的核范数，并存储在 lazy_b 中
    torch::Tensor lazy_b = torch::nuclear_norm(lazy_a);
    // 验证张量 b 和 lazy_b 是否接近
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestPairwiseDistance) {
  // 创建大小为 (4, 3) 的随机张量 x1，使用默认设备（DefaultDevice）
  torch::Tensor x1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 (4, 3) 的随机张量 x2，使用默认设备（DefaultDevice）
  torch::Tensor x2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个很小的 epsilon 值
  double eps = 1e-6;
  // 对于 keepdim 的两种取值 false 和 true 进行迭代
  for (bool keepdim : {false, true}) {
    // 对于 p 的多个取值（1, 2, 3, 4）进行迭代
    for (double p : {1, 2, 3, 4}) {
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 计算 x1 和 x2 之间的成对距离，使用给定的 p、eps 和 keepdim 参数
        torch::Tensor output =
            torch::pairwise_distance(x1, x2, p, eps, keepdim);
        // 将张量 x1 复制到指定设备，并存储在 lazy_x1 中
        torch::Tensor lazy_x1 = CopyToDevice(x1, device);
        // 将张量 x2 复制到指定设备，并存储在 lazy_x2 中
        torch::Tensor lazy_x2 = CopyToDevice(x2, device);
        // 计算 lazy_x1 和 lazy_x2 之间的成对距离，使用给定的 p、eps 和 keepdim 参数
        torch::Tensor lazy_output =
            torch::pairwise_distance(lazy_x1, lazy_x2, p, eps, keepdim);
        // 验证 output 和 lazy_output 是否接近，使用相对容差 1e-5 和绝对容差 1e-5
        AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCosineSimilarity) {
  // 创建大小为 (4, 3) 的随机张量 x1，使用默认设备（DefaultDevice）
  torch::Tensor x1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 (4, 3) 的随机张量 x2，使用默认设备（DefaultDevice）
  torch::Tensor x2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个很小的 epsilon 值
  double eps = 1e-8;
  // 获取张量 x1 的秩
  int rank = x1.dim();
  // 对于每个维度 dim 进行迭代，从 -rank 到 rank-1
  for (int dim = -rank; dim < rank; ++dim) {
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 计算张量 x1 和 x2 之间的余弦相似度，使用给定的 dim 和 eps 参数
      torch::Tensor output = torch::cosine_similarity(x1, x2, dim, eps);
      // 将张量 x1 复制到指定设备，并存储在 lazy_x1 中
      torch::Tensor lazy_x1 = CopyToDevice(x1, device);
      // 将张量 x2 复制到指定设备，并存储在 lazy_x2 中
      torch::Tensor lazy_x2 = CopyToDevice(x2, device);
      // 计算 lazy_x1 和 lazy_x2 之间的余弦相似度，使用给定的 dim 和 eps 参数
      torch::Tensor lazy_output =
          torch::cosine_similarity(lazy_x1, lazy_x2, dim, eps);
      // 验证 output 和 lazy_output 是否接近
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestCosineEmbeddingLoss) {
  // 创建大小为 (4, 3) 的随机张量 input1，使用默认设备（DefaultDevice）
  torch::Tensor input1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 (4, 3) 的随机张量 input2，使用默认设备（DefaultDevice）
  torch::Tensor input2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 (4) 的随机张量 target，使用默认设备（DefaultDevice）
  torch::Tensor target = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于每个 reduction 类型进行迭代，包括 Mean 和 Sum
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    # 对于每个 margin 值（0 和 0.2），执行以下操作
    for (double margin : {0., 0.2}) {
      # 对于每个设备，执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        # 计算输入张量 input1 和 input2 之间的余弦嵌入损失
        torch::Tensor output = torch::cosine_embedding_loss(
            input1, input2, target, margin, reduction);
        # 将 input1 拷贝到指定设备上
        torch::Tensor lazy_input1 = CopyToDevice(input1, device);
        # 将 input2 拷贝到指定设备上
        torch::Tensor lazy_input2 = CopyToDevice(input2, device);
        # 将 target 拷贝到指定设备上
        torch::Tensor lazy_target = CopyToDevice(target, device);
        # 在指定设备上计算拷贝的输入张量之间的余弦嵌入损失
        torch::Tensor lazy_output = torch::cosine_embedding_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        # 检查 output 和 lazy_output 是否在数值上相似
        AllClose(output, lazy_output);
      });
    }
  }


这段代码是一个嵌套循环结构，用于计算两组输入张量之间的余弦嵌入损失（cosine embedding loss）。其中涉及到对不同设备（例如 GPU）上的张量进行操作，并且在不同的 margin 值（0 和 0.2）下重复执行这些计算。
}

TEST_F(LazyOpsTest, TestHingeEmbeddingLoss) {
  // 创建一个随机张量作为输入数据，设备为默认设备
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个随机张量作为目标数据，设备为默认设备
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历不同的减少方式：均值和求和
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    // 遍历不同的边缘值
    for (double margin : {0., 0.2}) {
      // 对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 计算带有给定边缘值和减少方式的 Hinge Embedding Loss
        torch::Tensor output =
            torch::hinge_embedding_loss(input, target, margin, reduction);
        // 将输入数据复制到指定设备
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 将目标数据复制到指定设备
        torch::Tensor lazy_target = CopyToDevice(target, device);
        // 计算带有给定边缘值和减少方式的 Hinge Embedding Loss（延迟计算版本）
        torch::Tensor lazy_output = torch::hinge_embedding_loss(
            lazy_input, lazy_target, margin, reduction);
        // 检查输出和延迟计算输出是否接近
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestTripletMarginLoss) {
  // 创建一个随机张量作为锚点，设备为默认设备
  torch::Tensor anchor = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个随机张量作为正样本，设备为默认设备
  torch::Tensor positive = torch::abs(torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  // 创建一个随机张量作为负样本，设备为默认设备
  torch::Tensor negative = torch::neg(torch::abs(torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()))));
  double eps = 1e-6;
  // 遍历不同的边缘值
  for (double margin : {0., 0.2}) {
    // 遍历不同的 p 值
    for (double p : {1, 2, 3, 4}) {
      // 遍历是否交换的标志位
      for (bool swap : {false, true}) {
        // 遍历不同的减少方式：均值和求和
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum}) {
          // 对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 计算带有给定参数的 Triplet Margin Loss
            torch::Tensor output = torch::triplet_margin_loss(
                anchor, positive, negative, margin, p, eps, swap, reduction);
            // 将锚点数据复制到指定设备
            torch::Tensor lazy_anchor = CopyToDevice(anchor, device);
            // 将正样本数据复制到指定设备
            torch::Tensor lazy_positive = CopyToDevice(positive, device);
            // 将负样本数据复制到指定设备
            torch::Tensor lazy_negative = CopyToDevice(negative, device);
            // 计算带有给定参数的 Triplet Margin Loss（延迟计算版本）
            torch::Tensor lazy_output = torch::triplet_margin_loss(
                lazy_anchor,
                lazy_positive,
                lazy_negative,
                margin,
                p,
                eps,
                swap,
                reduction);
            // 检查输出和延迟计算输出是否接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestBinaryCrossEntropy) {
  int batch = 10;
  int classes = 5;
  // 创建一个随机张量作为输入数据，大小为 batch x classes，设备为默认设备
  torch::Tensor input = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个随机张量作为目标数据，大小为 batch x classes，设备为默认设备
  torch::Tensor target = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个随机张量作为权重，大小为 batch x classes，设备为默认设备
  torch::Tensor weight = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor undef;
  // 遍历不同的减少方式：均值、求和、无
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean,
        torch::Reduction::Sum,
        torch::Reduction::None}) {
    // 对于每个布尔值 undef_weight，在 false 和 true 两种情况下分别执行以下操作
    for (bool undef_weight : {false, true}) {
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 计算在给定设备上的二元交叉熵损失
        torch::Tensor output = torch::binary_cross_entropy(
            input, target, undef_weight ? undef : weight, reduction);
        // 将输入数据复制到指定设备上
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 将目标数据复制到指定设备上
        torch::Tensor lazy_target = CopyToDevice(target, device);
        // 如果 undef_weight 为 true，则将权重数据复制到指定设备上；否则，使用 undef
        torch::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        // 在指定设备上计算懒惰版本的二元交叉熵损失
        torch::Tensor lazy_output = torch::binary_cross_entropy(
            lazy_input, lazy_target, lazy_weight, reduction);
        // 检查两种计算版本的输出是否在指定的数值容差范围内相等
        AllClose(output, lazy_output, /*rtol=*/1e-4, /*atol=*/1e-5);
      });
    }
}

TEST_F(LazyOpsTest, TestMarginRankingLoss) {
  // 创建输入张量 input1，形状为 {4, 3}，随机初始化，使用默认设备
  torch::Tensor input1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建输入张量 input2，形状为 {4, 3}，随机初始化，使用默认设备
  torch::Tensor input2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建目标张量 target，形状为 {4, 3}，随机初始化，使用默认设备
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历两种约简方式：Mean 和 Sum
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    // 遍历两种边界值：0. 和 0.2
    for (double margin : {0., 0.2}) {
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 计算 margin ranking loss
        torch::Tensor output = torch::margin_ranking_loss(
            input1, input2, target, margin, reduction);
        // 将 input1 复制到指定设备并命名为 lazy_input1
        torch::Tensor lazy_input1 = CopyToDevice(input1, device);
        // 将 input2 复制到指定设备并命名为 lazy_input2
        torch::Tensor lazy_input2 = CopyToDevice(input2, device);
        // 将 target 复制到指定设备并命名为 lazy_target
        torch::Tensor lazy_target = CopyToDevice(target, device);
        // 计算在指定设备上的 margin ranking loss
        torch::Tensor lazy_output = torch::margin_ranking_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        // 验证输出与 lazy_output 的近似性
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestBCEWithLogits) {
  // 定义批次大小为 10，类别数为 5
  int batch = 10;
  int classes = 5;
  // 创建输入张量 input，形状为 {batch, classes}，随机初始化，使用默认设备
  torch::Tensor input = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建目标张量 target，形状为 {batch, classes}，随机初始化，使用默认设备
  torch::Tensor target = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建权重张量 weight，形状为 {classes}，随机初始化，使用默认设备
  torch::Tensor weight = torch::rand(
      {classes}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建正样本权重张量 pos_weight，形状为 {classes}，随机初始化，使用默认设备
  torch::Tensor pos_weight = torch::rand(
      {classes}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建未定义张量 undef，用于条件选择
  torch::Tensor undef;
  // 遍历两种约简方式：Mean 和 Sum
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    // 遍历两种权重定义情况：未定义和定义
    for (bool undef_weight : {false, true}) {
      // 遍历两种正样本权重定义情况：未定义和定义
      for (bool undef_pos_weight : {false, true}) {
        // 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          // 计算 binary cross entropy with logits
          torch::Tensor output = torch::binary_cross_entropy_with_logits(
              input,
              target,
              undef_weight ? undef : weight,
              undef_pos_weight ? undef : pos_weight,
              reduction);
          // 将 input 复制到指定设备并命名为 lazy_input
          torch::Tensor lazy_input = CopyToDevice(input, device);
          // 将 target 复制到指定设备并命名为 lazy_target
          torch::Tensor lazy_target = CopyToDevice(target, device);
          // 根据条件将 weight 复制到指定设备并命名为 lazy_weight 或者使用 undef
          torch::Tensor lazy_weight =
              undef_weight ? undef : CopyToDevice(weight, device);
          // 根据条件将 pos_weight 复制到指定设备并命名为 lazy_pos_weight 或者使用 undef
          torch::Tensor lazy_pos_weight =
              undef_pos_weight ? undef : CopyToDevice(pos_weight, device);
          // 计算在指定设备上的 binary cross entropy with logits
          torch::Tensor lazy_output = torch::binary_cross_entropy_with_logits(
              lazy_input, lazy_target, lazy_weight, lazy_pos_weight, reduction);
        });
      }
    }
  }
}
TEST_F(LazyOpsTest, TestKlDiv) {
  // 生成一个大小为 (4, 3) 的随机张量 input，并指定其在默认设备上的数据类型为 float
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 生成一个与 input 相同大小的随机张量 target，并指定其在默认设备上的数据类型为 float
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历 log_target 取值为 true 和 false
  for (bool log_target : {true, false}) {
    // 遍历 reduction 取值为 Mean 和 Sum
    for (torch::Reduction::Reduction reduction :
         {torch::Reduction::Mean, torch::Reduction::Sum}) {
      // 对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 计算 input 和 target 的 KL 散度，结果为 output
        torch::Tensor output =
            torch::kl_div(input, target, reduction, log_target);
        // 将 input 复制到指定设备上得到 lazy_input
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 将 target 复制到指定设备上得到 lazy_target
        torch::Tensor lazy_target = CopyToDevice(target, device);
        // 计算 lazy_input 和 lazy_target 的 KL 散度，结果为 lazy_output
        torch::Tensor lazy_output =
            torch::kl_div(lazy_input, lazy_target, reduction, log_target);
        // 检查 output 和 lazy_output 是否近似相等
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestProd) {
  // 生成一个大小为 (4, 3, 4) 的随机张量 a，并指定其在默认设备上的数据类型为 float
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的所有元素的乘积，结果为 b
  torch::Tensor b = torch::prod(a);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的所有元素的乘积，结果为 lazy_b
    torch::Tensor lazy_b = torch::prod(lazy_a);
    // 检查 b 和 lazy_b 是否近似相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdCast) {
  // 生成一个大小为 (4, 3, 4) 的随机张量 a，并指定其在默认设备上的数据类型为 float
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的所有元素的乘积，并将结果类型转换为 double，结果为 b
  torch::Tensor b = torch::prod(a, torch::kDouble);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的所有元素的乘积，并将结果类型转换为 double，结果为 lazy_b
    torch::Tensor lazy_b = torch::prod(lazy_a, torch::kDouble);
    // 检查 b 和 lazy_b 是否近似相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdInDim) {
  // 生成一个大小为 (4, 3, 4) 的随机张量 a，并指定其在默认设备上的数据类型为 float
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量 a 的维度数
  int rank = a.dim();
  // 对张量 a 的每个维度执行操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量 a 沿指定维度 dim 的元素乘积，结果为 b
    torch::Tensor b = torch::prod(a, dim);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算 lazy_a 沿指定维度 dim 的元素乘积，结果为 lazy_b
      torch::Tensor lazy_b = torch::prod(lazy_a, dim);
      // 检查 b 和 lazy_b 是否近似相等
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeepCast) {
  // 生成一个大小为 (4, 3, 4) 的随机张量 a，并指定其在默认设备上的数据类型为 float
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量 a 的维度数
  int rank = a.dim();
  // 对张量 a 的每个维度执行操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量 a 沿指定维度 dim 的元素乘积，并保持维度，结果为 b
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true, torch::kDouble);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 计算 lazy_a 沿指定维度 dim 的元素乘积，并保持维度，结果为 lazy_b
      torch::Tensor lazy_b =
          torch::prod(lazy_a, dim, /*keepdim=*/true, torch::kDouble);
      // 检查 b 和 lazy_b 是否近似相等
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeep) {
  // 生成一个大小为 (4, 3, 4) 的随机张量 a，并指定其在默认设备上的数据类型为 float
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量 a 的维度数
  int rank = a.dim();
  // 对张量 a 的每个维度执行操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量 a 沿指定维度 dim 的元素乘积，并保持维度，结果为 b
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      // 对每个设备执行以下操作，lambda 表达式捕获 device 变量
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 将张量 a 复制到当前设备上，并创建 lazy_a
      torch::Tensor lazy_b = torch::prod(lazy_a, dim, /*keepdim=*/true);
      // 计算 lazy_a 张量沿指定维度 dim 的乘积，并保持维度不变，存储在 lazy_b 中
      AllClose(b, lazy_b);
      // 检查张量 b 是否与 lazy_b 在数值上全部接近
    });
  }
}

// 在 LazyOpsTest 测试类中定义 TestCumSum 方法
TEST_F(LazyOpsTest, TestCumSum) {
  // 生成一个形状为 {4, 3, 4} 的随机浮点数张量，并指定其设备为默认设备
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量的维度
  int rank = input.dim();
  // 遍历张量的每一个维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量在指定维度上的累积和，并赋值给 result
    torch::Tensor result = torch::cumsum(input, dim);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算复制后的张量在指定维度上的累积和
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      // 检查结果是否全部接近
      AllClose(result, lazy_result);
    });
  }
}

// 在 LazyOpsTest 测试类中定义 TestCumSumCast 方法
TEST_F(LazyOpsTest, TestCumSumCast) {
  // 生成一个形状为 {4, 3, 4} 的随机浮点数张量，并指定其设备为默认设备
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量的维度
  int rank = input.dim();
  // 遍历张量的每一个维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量在指定维度上的累积和，并指定结果类型为 double，赋值给 result
    torch::Tensor result = torch::cumsum(input, dim, torch::kDouble);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算复制后的张量在指定维度上的累积和，结果类型为 double
      torch::Tensor lazy_result =
          torch::cumsum(lazy_input, dim, torch::kDouble);
      // 检查结果是否全部接近
      AllClose(result, lazy_result);
    });
  }
}

// 在 LazyOpsTest 测试类中定义 TestCumSumLong 方法
TEST_F(LazyOpsTest, TestCumSumLong) {
  // 生成一个形状为 {4, 3, 4} 的随机长整型张量，并指定其设备为默认设备
  torch::Tensor input = torch::randint(
      1000,
      {4, 3, 4},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 获取张量的维度
  int rank = input.dim();
  // 遍历张量的每一个维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量在指定维度上的累积和，并赋值给 result
    torch::Tensor result = torch::cumsum(input, dim);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算复制后的张量在指定维度上的累积和
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      // 检查结果是否全部相等
      AllEqual(result, lazy_result);
    });
  }
}

// 在 LazyOpsTest 测试类中定义 TestCumSumCastLong 方法
TEST_F(LazyOpsTest, TestCumSumCastLong) {
  // 生成一个形状为 {4, 3, 4} 的随机浮点数张量，并指定其设备为默认设备
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量的维度
  int rank = input.dim();
  // 遍历张量的每一个维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量在指定维度上的累积和，并指定结果类型为 long，赋值给 result
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算复制后的张量在指定维度上的累积和，结果类型为 long
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim, torch::kLong);
      // 检查结果是否全部相等
      AllEqual(result, lazy_result);
    });
  }
}

// 在 LazyOpsTest 测试类中定义 TestCumProd 方法
TEST_F(LazyOpsTest, TestCumProd) {
  // 生成一个形状为 {4, 3, 4} 的随机浮点数张量，并指定其设备为默认设备
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量的维度
  int rank = input.dim();
  // 遍历张量的每一个维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量在指定维度上的累积乘积，并赋值给 result
    torch::Tensor result = torch::cumprod(input, dim);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算复制后的张量在指定维度上的累积乘积
      torch::Tensor lazy_result = torch::cumprod(lazy_input, dim);
      // 检查结果是否全部接近
      AllClose(result, lazy_result);
    });
  }
}

// 在 LazyOpsTest 测试类中定义 TestCumProdCast 方法
TEST_F(LazyOpsTest, TestCumProdCast) {
  // 生成一个形状为 {4, 3, 4} 的随机浮点数张量，并乘以 10，指定其设备为默认设备
  torch::Tensor input = torch::mul(
      torch::rand(
          {4, 3, 4},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice())),
      10);
  // 获取张量的维度
  int rank = input.dim();
  // 遍历张量的每一个维度
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算张量在指定维度上的累积乘积，并指定结果类型为 double，赋值给 result
    torch::Tensor result = torch::cumprod(input, dim, torch::kDouble);
    // 对于每个设备执行以下操作（lambda 表达式作为参数）
    ForEachDevice([&](const torch::Device& device) {
      // 将输入数据复制到指定设备上，并返回对应的惰性张量
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定维度上计算输入张量的累积乘积，结果为双精度类型
      torch::Tensor lazy_result =
          torch::cumprod(lazy_input, dim, torch::kDouble);
      // 检查计算结果与预期结果是否全部接近
      AllClose(result, lazy_result);
    });
  }
TEST_F(LazyOpsTest, TestCumProdLong) {
  // 创建一个形状为 {2, 3} 的长整型张量，元素值为 [0, 7) 的随机整数
  torch::Tensor input = torch::randint(
      7, {2, 3}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 获取张量的秩
  int rank = input.dim();
  // 在每个维度上进行累积和操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算沿指定维度的累积和
    torch::Tensor result = torch::cumsum(input, dim);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定设备上计算沿指定维度的累积和
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      // 检查结果张量是否相等
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdCastLong) {
  // 创建一个形状为 {2, 3} 的浮点型张量，元素值为 [0, 7) 的随机数乘以7
  torch::Tensor input =
      torch::rand(
          {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      7;
  // 获取张量的秩
  int rank = input.dim();
  // 在每个维度上进行累积和操作，同时将结果强制转换为长整型
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算沿指定维度的累积和，并将结果转换为长整型
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定设备上计算沿指定维度的累积和，并将结果转换为长整型
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim, torch::kLong);
      // 检查结果张量是否相等
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMin) {
  // 创建一个形状为 {4, 4, 4} 的浮点型张量，元素为随机数
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量的最小值索引，不保留维度
  torch::Tensor b = torch::argmin(a, c10::nullopt, /*keepdim=*/false);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算张量的最小值索引，不保留维度
    torch::Tensor lazy_b =
        torch::argmin(lazy_a, c10::nullopt, /*keepdim=*/false);
    // 检查结果张量是否相等
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMinDim) {
  // 创建一个形状为 {4, 4, 4} 的浮点型张量，元素为随机数
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对指定维度计算张量的最小值索引，不保留维度
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上对指定维度计算张量的最小值索引，不保留维度
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/false);
      // 检查结果张量是否相等
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinDimKeep) {
  // 创建一个形状为 {4, 4, 4} 的浮点型张量，元素为随机数
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对指定维度计算张量的最小值索引，并保留维度
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/true);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上对指定维度计算张量的最小值索引，并保留维度
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/true);
      // 检查结果张量是否相等
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinSameValue) {
  // 创建一个形状为 {4, 4, 4} 的浮点型张量，元素值均为1
  torch::Tensor a = torch::ones(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量的最小值索引
  torch::Tensor b = torch::argmin(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算张量的最小值索引
    torch::Tensor lazy_b = torch::argmin(lazy_a);
    // 检查结果张量是否相等
    AllEqual(b, lazy_b);
  });
}
TEST_F(LazyOpsTest, TestArgMinWrapper) {
  // 创建一个 4x4x4 的随机张量 a，浮点数类型，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度列表 {1, -2}
  for (int dim : {1, -2}) {
    // 对张量 a 按指定维度 dim 计算最小值索引，不保持维度
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上计算张量 lazy_a 沿 dim 维度的最小值索引，不保持维度
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/false);
      // 检查张量 b 和 lazy_b 是否相等
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMax) {
  // 创建一个 4x4x4 的随机张量 a，浮点数类型，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的全局最大值索引，不指定维度，不保持维度
  torch::Tensor b = torch::argmax(a, c10::nullopt, /*keepdim=*/false);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算张量 lazy_a 的全局最大值索引，不指定维度，不保持维度
    torch::Tensor lazy_b =
        torch::argmax(lazy_a, c10::nullopt, /*keepdim=*/false);
    // 检查张量 b 和 lazy_b 是否相等
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxDim) {
  // 创建一个 4x4x4 的随机张量 a，浮点数类型，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度列表 {1, -2}
  for (int dim : {1, -2}) {
    // 计算张量 a 沿 dim 维度的最大值索引，不保持维度
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上计算张量 lazy_a 沿 dim 维度的最大值索引，不保持维度
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/false);
      // 检查张量 b 和 lazy_b 是否相等
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxDimKeep) {
  // 创建一个 4x4x4 的随机张量 a，浮点数类型，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度列表 {1, -2}
  for (int dim : {1, -2}) {
    // 计算张量 a 沿 dim 维度的最大值索引，保持维度
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/true);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上计算张量 lazy_a 沿 dim 维度的最大值索引，保持维度
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/true);
      // 检查张量 b 和 lazy_b 是否相等
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxSameValue) {
  // 创建一个所有元素为 1 的 4x4x4 张量 a，浮点数类型，设备为默认设备
  torch::Tensor a = torch::ones(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的全局最大值索引，不指定维度，不保持维度
  torch::Tensor b = torch::argmax(a, c10::nullopt, /*keepdim=*/false);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算张量 lazy_a 的全局最大值索引，不指定维度，不保持维度
    torch::Tensor lazy_b =
        torch::argmax(lazy_a, c10::nullopt, /*keepdim=*/false);
    // 检查张量 b 和 lazy_b 是否相等
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxWrapper) {
  // 创建一个 4x4x4 的随机张量 a，浮点数类型，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历维度列表 {1, -2}
  for (int dim : {1, -2}) {
    // 计算张量 a 沿 dim 维度的最大值索引，不保持维度
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上计算张量 lazy_a 沿 dim 维度的最大值索引，不保持维度
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/false);
      // 检查张量 b 和 lazy_b 是否相等
      AllEqual(b, lazy_b);
    });
  }
}
TEST_F(LazyOpsTest, TestAsin) {
  // 创建一个大小为 {2, 2} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的反正弦值并赋给张量 b
  torch::Tensor b = torch::asin(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的反正弦值并赋给 lazy_b
    torch::Tensor lazy_b = torch::asin(lazy_a);
    // 检查张量 b 和 lazy_b 是否在给定的相对容差和绝对容差下相似
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAsinh) {
  // 创建一个大小为 {2, 2} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的反双曲正弦值并赋给张量 b
  torch::Tensor b = torch::asinh(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的反双曲正弦值并赋给 lazy_b
    torch::Tensor lazy_b = torch::asinh(lazy_a);
    // 检查张量 b 和 lazy_b 是否在给定的相对容差和绝对容差下相似
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAsinhInPlace) {
  // 创建一个大小为 {2, 2} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算张量 a 的反双曲正弦值并赋给张量 b
    torch::Tensor b = torch::asinh_(a);
    // 在 lazy_a 上原地计算反双曲正弦值并赋给 lazy_b
    torch::Tensor lazy_b = torch::asinh_(lazy_a);
    // 检查张量 a 和 lazy_a 是否在给定的相对容差和绝对容差下相似
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    // 检查张量 b 和 lazy_b 是否在给定的相对容差和绝对容差下相似
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSin) {
  // 创建一个大小为 {2, 2} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的正弦值并赋给张量 b
  torch::Tensor b = torch::sin(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的正弦值并赋给 lazy_b
    torch::Tensor lazy_b = torch::sin(lazy_a);
    // 检查张量 b 和 lazy_b 是否在给定的相对容差和绝对容差下相似
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSinh) {
  // 创建一个大小为 {2, 2} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的双曲正弦值并赋给张量 b
  torch::Tensor b = torch::sinh(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的双曲正弦值并赋给 lazy_b
    torch::Tensor lazy_b = torch::sinh(lazy_a);
    // 检查张量 b 和 lazy_b 是否在给定的相对容差和绝对容差下相似
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcos) {
  // 创建一个大小为 {2, 2} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的反余弦值并赋给张量 b
  torch::Tensor b = torch::acos(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的反余弦值并赋给 lazy_b
    torch::Tensor lazy_b = torch::acos(lazy_a);
    // 检查张量 b 和 lazy_b 是否在给定的相对容差和绝对容差下相似
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcosh) {
  // 创建一个大小为 {2, 2} 的随机张量 a，使用默认设备，并乘以 100
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100;
  // 计算张量 a 的反双曲余弦值并赋给张量 b
  torch::Tensor b = torch::acosh(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的反双曲余弦值并赋给 lazy_b
    torch::Tensor lazy_b = torch::acosh(lazy_a);
    // 检查张量 b 和 lazy_b 是否在给定的相对容差和绝对容差下相似
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}
TEST_F(LazyOpsTest, TestAcoshInPlace) {
  // 创建一个形状为 {2, 2} 的随机张量 a，数据类型为 float，位于默认设备上
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算张量 a 的反双曲余弦，并将结果保存在张量 b 中
    torch::Tensor b = torch::acosh_(a);
    // 在复制到设备上的张量 lazy_a 上执行反双曲余弦操作，并将结果保存在 lazy_b 中
    torch::Tensor lazy_b = torch::acosh_(lazy_a);
    // 检查张量 a 和 lazy_a 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    // 检查张量 b 和 lazy_b 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestCos) {
  // 创建一个形状为 {2, 2} 的随机张量 a，数据类型为 float，位于默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的余弦，并将结果保存在张量 b 中
  torch::Tensor b = torch::cos(a);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在复制到设备上的张量 lazy_a 上计算余弦，并将结果保存在 lazy_b 中
    torch::Tensor lazy_b = torch::cos(lazy_a);
    // 检查张量 b 和 lazy_b 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestCosh) {
  // 创建一个形状为 {2, 2} 的随机张量 a，数据类型为 float，位于默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的双曲余弦，并将结果保存在张量 b 中
  torch::Tensor b = torch::cosh(a);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在复制到设备上的张量 lazy_a 上计算双曲余弦，并将结果保存在 lazy_b 中
    torch::Tensor lazy_b = torch::cosh(lazy_a);
    // 检查张量 b 和 lazy_b 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtan) {
  // 创建一个形状为 {2, 2} 的随机张量 a，数据类型为 float，位于默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的反正切，并将结果保存在张量 b 中
  torch::Tensor b = torch::atan(a);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在复制到设备上的张量 lazy_a 上计算反正切，并将结果保存在 lazy_b 中
    torch::Tensor lazy_b = torch::atan(lazy_a);
    // 检查张量 b 和 lazy_b 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtanh) {
  // 创建一个形状为 {2, 2} 的随机张量 a，数据类型为 float，位于默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的双曲反正切，并将结果保存在张量 b 中
  torch::Tensor b = torch::atanh(a);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在复制到设备上的张量 lazy_a 上计算双曲反正切，并将结果保存在 lazy_b 中
    torch::Tensor lazy_b = torch::atanh(lazy_a);
    // 检查张量 b 和 lazy_b 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtanhInPlace) {
  // 创建一个形状为 {2, 2} 的随机张量 a，数据类型为 float，位于默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在原地计算张量 a 的双曲反正切，并将结果保存在张量 b 中
    torch::Tensor b = torch::atanh_(a);
    // 在原地计算复制到设备上的张量 lazy_a 的双曲反正切，并将结果保存在 lazy_b 中
    torch::Tensor lazy_b = torch::atanh_(lazy_a);
    // 检查张量 a 和 lazy_a 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    // 检查张量 b 和 lazy_b 的元素是否在相对容差 1e-3 和绝对容差 1e-5 内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtan2) {
  // 创建形状为 {2, 2} 的随机张量 a 和 b，数据类型为 float，位于默认设备上
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计
    # 计算张量 lazy_a 和 lazy_b 的反正切，并存储结果到 lazy_c 张量中
    torch::Tensor lazy_c = torch::atan2(lazy_a, lazy_b);
    # 使用 AllClose 函数检查 c 张量和 lazy_c 张量是否在指定的相对容差（rtol）和绝对容差（atol）范围内接近
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-5);
}

TEST_F(LazyOpsTest, TestTan) {
  // 生成一个大小为 [2, 2] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对张量 a 中的每个元素计算 tan 函数值，得到张量 b
  torch::Tensor b = torch::tan(a);
  // 对于每个设备，将张量 a 复制到该设备上，并计算在该设备上的 tan 函数结果 lazy_b
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::tan(lazy_a);
    // 检查张量 b 和 lazy_b 在设备上是否近似相等，指定相对误差为 1e-3，绝对误差为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestTanh) {
  // 生成一个大小为 [2, 2] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对张量 a 中的每个元素计算 tanh 函数值，得到张量 b
  torch::Tensor b = torch::tanh(a);
  // 对于每个设备，将张量 a 复制到该设备上，并计算在该设备上的 tanh 函数结果 lazy_b
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::tanh(lazy_a);
    // 检查张量 b 和 lazy_b 在设备上是否近似相等，指定相对误差为 1e-3，绝对误差为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestClampMinMax) {
  // 生成一个大小为 [2, 2] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置最小值和最大值
  torch::Scalar min_val(0.311);
  torch::Scalar max_val(0.409);
  // 对张量 a 中的每个元素进行 clamp 操作，限制在 [min_val, max_val] 范围内，得到张量 b
  torch::Tensor b = torch::clamp(a, min_val, max_val);
  // 对于每个设备，将张量 a 复制到该设备上，并计算在该设备上的 clamp 操作结果 lazy_b
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp(lazy_a, min_val, max_val);
    // 检查张量 b 和 lazy_b 在设备上是否近似相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMin) {
  // 生成一个大小为 [2, 2] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置最小值
  torch::Scalar min_val(0.311);
  // 对张量 a 中的每个元素进行 clamp_min 操作，限制下界为 min_val，得到张量 b
  torch::Tensor b = torch::clamp(a, min_val, c10::nullopt);
  // 对于每个设备，将张量 a 复制到该设备上，并计算在该设备上的 clamp_min 操作结果 lazy_b
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp(lazy_a, min_val, c10::nullopt);
    // 检查张量 b 和 lazy_b 在设备上是否近似相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMax) {
  // 生成一个大小为 [2, 2] 的随机张量 a，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置最大值
  torch::Scalar max_val(0.409);
  // 对张量 a 中的每个元素进行 clamp_max 操作，限制上界为 max_val，得到张量 b
  torch::Tensor b = torch::clamp(a, c10::nullopt, max_val);
  // 对于每个设备，将张量 a 复制到该设备上，并计算在该设备上的 clamp_max 操作结果 lazy_b
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp(lazy_a, c10::nullopt, max_val);
    // 使用 AllClose 函数比较 b 和 lazy_b 是否全部相等
    AllClose(b, lazy_b);
  });
TEST_F(LazyOpsTest, TestClampMinExplicitInPlace) {
  // 创建一个大小为 2x2 的随机张量 a，数据类型为 float，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置最小值为 0.311
  torch::Scalar min_val(0.311);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在张量 a 上执行 clamp_min_ 操作，限制最小值为 min_val，得到张量 b
    torch::Tensor b = torch::clamp_min_(a, min_val);
    // 在 lazy_a 上执行 clamp_min_ 操作，限制最小值为 min_val，得到 lazy_b
    torch::Tensor lazy_b = torch::clamp_min_(lazy_a, min_val);
    // 检查张量 a 和 lazy_a 是否全部相等
    AllClose(a, lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMaxExplicitInPlace) {
  // 创建一个大小为 2x2 的随机张量 a，数据类型为 float，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置最大值为 0.409
  torch::Scalar max_val(0.409);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在张量 a 上执行 clamp_max_ 操作，限制最大值为 max_val，得到张量 b
    torch::Tensor b = torch::clamp_max_(a, max_val);
    // 在 lazy_a 上执行 clamp_max_ 操作，限制最大值为 max_val，得到 lazy_b
    torch::Tensor lazy_b = torch::clamp_max_(lazy_a, max_val);
    // 检查张量 a 和 lazy_a 是否全部相等
    AllClose(a, lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCeil) {
  // 创建一个大小为 2x2 的随机张量 a，数据类型为 float，设备为默认设备
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 对张量 a 执行向上取整操作，得到张量 b
  torch::Tensor b = torch::ceil(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在 lazy_a 上执行向上取整操作，得到 lazy_b
    torch::Tensor lazy_b = torch::ceil(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFloor) {
  // 创建一个大小为 2x2 的随机张量 a，数据类型为 float，设备为默认设备
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 对张量 a 执行向下取整操作，得到张量 b
  torch::Tensor b = torch::floor(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在 lazy_a 上执行向下取整操作，得到 lazy_b
    torch::Tensor lazy_b = torch::floor(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRound) {
  // 创建一个大小为 10 的张量 a，包含随机数和特殊值 -0.5 和 0.5
  torch::Tensor a = torch::cat(
      {torch::randn(
           {8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
           100.0,
       torch::tensor(
           {-0.5, 0.5},
           torch::TensorOptions(torch::kFloat).device(DefaultDevice()))},
      0);
  // 对张量 a 执行四舍五入操作，得到张量 b
  torch::Tensor b = torch::round(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在 lazy_a 上执行四舍五入操作，得到 lazy_b
    torch::Tensor lazy_b = torch::round(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTrunc) {
  // 创建一个大小为 2x2 的随机张量 a，数据类型为 float，设备为默认设备
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 对张量 a 执行截断操作，得到张量 b
  torch::Tensor b = torch::trunc(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在 lazy_a 上执行截断操作，得到 lazy_b
    torch::Tensor lazy_b = torch::trunc(lazy_a);
    // 检查张量 b 和 lazy_b 是否全部相等
    AllClose(b, lazy_b);
  });
}
TEST_F(LazyOpsTest, TestFrac) {
  // 创建一个大小为2x2的随机张量a，元素取值范围在[-100, 100)，数据类型为float32，使用默认设备
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 对张量a进行分数函数操作，得到张量b
  torch::Tensor b = torch::frac(a);
  // 对每个设备执行以下操作：
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 对lazy_a进行分数函数操作，得到lazy_b
    torch::Tensor lazy_b = torch::frac(lazy_a);
    // 检查张量b与lazy_b是否接近
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNeg) {
  // 创建一个大小为2x2的随机张量a，数据类型为float32，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对张量a进行取负操作，得到张量b
  torch::Tensor b = torch::neg(a);
  // 对每个设备执行以下操作：
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 对lazy_a进行取负操作，得到lazy_b
    torch::Tensor lazy_b = torch::neg(lazy_a);
    // 检查张量b与lazy_b是否接近
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestBitwiseNot) {
  // 定义一组数据类型
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  // 对每个设备执行以下操作：
  ForEachDevice([&](const torch::Device& device) {
    // 遍历每种数据类型
    for (auto type : types) {
      // 创建一个大小为2x2的随机整数张量a，取值范围在[0, 63)，指定数据类型
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      // 对张量a进行按位取反操作，得到张量b
      torch::Tensor b = torch::bitwise_not(a);
      // 将张量a复制到指定设备，得到lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 对lazy_a进行按位取反操作，得到lazy_b
      torch::Tensor lazy_b = torch::bitwise_not(lazy_a);
      // 检查张量b与lazy_b是否完全相等
      AllEqual(b, lazy_b);
    }
  });
}

TEST_F(LazyOpsTest, TestBitwiseNotInPlace) {
  // 定义一组数据类型
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  // 对每个设备执行以下操作：
  ForEachDevice([&](const torch::Device& device) {
    // 遍历每种数据类型
    for (auto type : types) {
      // 创建一个大小为2x2的随机整数张量a，取值范围在[0, 63)，指定数据类型
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      // 将张量a复制到指定设备，得到lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 对张量a进行原地按位取反操作
      a.bitwise_not_();
      // 对lazy_a进行原地按位取反操作
      lazy_a.bitwise_not_();
      // 检查张量a与lazy_a是否完全相等
      AllEqual(a, lazy_a);
    }
  });
}

TEST_F(LazyOpsTest, TestSign) {
  // 创建一个大小为2x2的随机张量a，元素取值范围在[-100, 100)，数据类型为float32，使用默认设备
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 对张量a进行符号函数操作，得到张量b
  torch::Tensor b = torch::sign(a);
  // 对每个设备执行以下操作：
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 对lazy_a进行符号函数操作，得到lazy_b
    torch::Tensor lazy_b = torch::sign(lazy_a);
    // 检查张量b与lazy_b是否接近
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSignByte) {
  // 创建一个大小为2x2的随机整数张量a，取值范围在[0, 255]，数据类型为uint8，使用默认设备
  torch::Tensor a = torch::randint(
      256, {2, 2}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 对张量a进行符号函数操作，得到张量b
  torch::Tensor b = torch::sign(a);
  // 对每个设备执行以下操作：
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 对lazy_a进行符号函数操作，得到lazy_b
    torch::Tensor lazy_b = torch::sign(lazy_a);
    // 检查张量b与lazy_b是否完全相等
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestAbs) {
  // 创建一个大小为2x2的随机张量a，元素取值范围在[-inf, inf)，数据类型为float32，使用默认设备
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对张量a进行绝对值函数操作，得到张量b
  torch::Tensor b = torch::abs(a);
  // 对每个设备执行以下操作：
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 对lazy_a进行绝对值函数操作，得到lazy_b
    torch::Tensor lazy_b = torch::abs(lazy_a);
    // 检查张量b与lazy_b是否接近
    AllClose(b, lazy_b);
  });
}
TEST_F(LazyOpsTest, TestAbsByte) {
  // 创建一个形状为 {2, 2}、数值在 0 到 255 之间的随机整数张量，数据类型为字节（8 位无符号整数）
  torch::Tensor a = torch::randint(
      256, {2, 2}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 对张量 a 中的每个元素取绝对值，得到张量 b
  torch::Tensor b = torch::abs(a);
  // 遍历所有设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上对 lazy_a 中的每个元素取绝对值，得到 lazy_b
    torch::Tensor lazy_b = torch::abs(lazy_a);
    // 检查张量 b 和 lazy_b 在当前设备上是否相等
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestEmptyLike) {
  // 创建一个形状为 {2, 2}、随机数值的浮点张量
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与张量 a 具有相同形状和设备的空张量 b
  torch::Tensor b = torch::empty_like(a);
  // 遍历所有设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 创建一个与 lazy_a 具有相同形状和设备的空张量 lazy_b
    torch::Tensor lazy_b = torch::empty_like(lazy_a);
    // 检查 lazy_b 的形状是否与 b 相同
    EXPECT_EQ(b.sizes(), lazy_b.sizes());
  });
}

TEST_F(LazyOpsTest, TestEmptyLikeOptions) {
  // 创建一个形状为 {2, 2}、随机数值的浮点张量 a
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与张量 a 具有相同形状和设备的空张量 b，使用指定的选项
  torch::Tensor b = torch::empty_like(
      a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历所有设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 创建一个与 lazy_a 具有相同形状和设备的空张量 lazy_b，使用指定的选项
    torch::Tensor lazy_b = torch::empty_like(
        lazy_a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 检查 lazy_b 的形状是否与 b 相同
    EXPECT_EQ(b.sizes(), lazy_b.sizes());
  });
}

TEST_F(LazyOpsTest, TestEmpty) {
  // 创建一个形状为 {2, 2}、元素全为 0 的浮点张量 a
  torch::Tensor a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历所有设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 {2, 2}、元素未初始化的浮点张量 lazy_a，设备为当前设备
    torch::Tensor lazy_a = torch::empty(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    // 检查 lazy_a 的形状是否与 a 相同
    EXPECT_EQ(a.sizes(), lazy_a.sizes());
  });
}

TEST_F(LazyOpsTest, TestZeroInPlace) {
  // 创建一个形状为 {2, 2}、元素全为 1 的浮点张量 input
  torch::Tensor input = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

  // 遍历所有设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 input 复制到指定设备上，得到 lazyInput
    torch::Tensor lazyInput = CopyToDevice(input, device);
    // 在当前设备上执行零操作，修改 input，并将结果保存在 output 中
    auto& output = torch::zero_(input);
    // 在指定设备上执行零操作，修改 lazyInput，并将结果保存在 lazyOutput 中
    auto& lazyOutput = torch::zero_(lazyInput);
    // 检查 output 和 lazyOutput 是否在数值上相等
    AllClose(output, lazyOutput);
  });
}

TEST_F(LazyOpsTest, TestZerosLike) {
  // 创建一个形状为 {2, 2}、随机数值的浮点张量 a
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与张量 a 具有相同形状和设备的零张量 b
  torch::Tensor b = torch::zeros_like(a);
  // 遍历所有设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 创建一个与 lazy_a 具有相同形状和设备的零张量 lazy_b
    torch::Tensor lazy_b = torch::zeros_like(lazy_a);
    // 检查 a 和 lazy_a 在数值上是否相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestZerosLikeOptions) {
  // 创建一个形状为 {2, 2}、随机数值的浮点张量 a
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与张量 a 具有相同形状和设备的零张量 b，使用指定的选项
  torch::Tensor b = torch::zeros_like(
      a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历所有设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 创建一个与 lazy_a 具有相同形状和设备的零张量 lazy_b，使用指定的选项
    torch::Tensor lazy_b = torch::zeros_like(
        lazy_a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 检查 a 和 lazy_a 在数值上是否相等
    AllClose(a, lazy_a);
  });
}
TEST_F(LazyOpsTest, TestZeros) {
  // 创建一个 2x2 的浮点数张量，所有元素为 0，设备为默认设备
  torch::Tensor a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个 2x2 的浮点数张量，所有元素为 0，设备为当前迭代的设备
    torch::Tensor lazy_a = torch::zeros(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    // 检查张量 a 和 lazy_a 是否在数值上近似相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnes) {
  // 创建一个 2x2 的浮点数张量，所有元素为 1，设备为默认设备
  torch::Tensor a = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个 2x2 的浮点数张量，所有元素为 1，设备为当前迭代的设备
    torch::Tensor lazy_a =
        torch::ones({2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    // 检查张量 a 和 lazy_a 是否在数值上近似相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnesLike) {
  // 创建一个 2x2 的随机浮点数张量，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 根据张量 a 创建一个元素全为 1 的张量 b
  torch::Tensor b = torch::ones_like(a);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到当前迭代的设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 根据 lazy_a 创建一个元素全为 1 的张量 lazy_b
    torch::Tensor lazy_b = torch::ones_like(lazy_a);
    // 检查张量 a 和 lazy_a 是否在数值上近似相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnesLikeOptions) {
  // 创建一个 2x2 的随机浮点数张量，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 根据张量 a 创建一个元素全为 1 的张量 b，设备为默认设备
  torch::Tensor b = torch::ones_like(
      a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到当前迭代的设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 根据 lazy_a 创建一个元素全为 1 的张量 lazy_b，设备为默认设备
    torch::Tensor lazy_b = torch::ones_like(
        lazy_a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 检查张量 a 和 lazy_a 是否在数值上近似相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFull) {
  // 创建一个 2x2 的浮点数张量，所有元素为 3.1165，设备为默认设备
  torch::Tensor a = torch::full(
      {2, 2},
      3.1165,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个 2x2 的浮点数张量，所有元素为 3.1165，设备为当前迭代的设备
    torch::Tensor lazy_a = torch::full(
        {2, 2}, 3.1165, torch::TensorOptions(torch::kFloat).device(device));
    // 检查张量 a 和 lazy_a 是否在数值上近似相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFullLike) {
  // 创建一个 2x2 的随机浮点数张量，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 根据张量 a 创建一个元素全为 3.1165 的张量 b
  torch::Tensor b = torch::full_like(a, 3.1165);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到当前迭代的设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 根据 lazy_a 创建一个元素全为 3.1165 的张量 lazy_b
    torch::Tensor lazy_b = torch::full_like(lazy_a, 3.1165);
    // 检查张量 a 和 lazy_a 是否在数值上近似相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFullLikeOptions) {
  // 创建一个 2x2 的随机浮点数张量，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 根据张量 a 创建一个元素全为 3.1165 的张量 b，设备为默认设备
  torch::Tensor b = torch::full_like(
      a, 3.1165, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到当前迭代的设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 根据 lazy_a 创建一个元素全为 3.1165 的张量 lazy_b，设备为默认设备
    torch::Tensor lazy_b = torch::full_like(
        lazy_a,
        3.1165,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 检查张量 a 和 lazy_a 是否在数值上近似相等
    AllClose(a, lazy_a);
  });
}
TEST_F(LazyOpsTest, TestARange) {
  // 在两组范围上进行迭代：{0.0, 100.0, 0.5} 和 {0.0, -100.0, -0.5}
  for (auto& ranges : std::vector<std::vector<float>>{
           {0.0, 100.0, 0.5}, {0.0, -100.0, -0.5}}) {
    // 创建张量 a，包含从 ranges[0] 到 ranges[1] 的数列，步长为 ranges[2]
    torch::Tensor a = torch::arange(
        ranges[0],
        ranges[1],
        ranges[2],
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 在指定设备上创建 lazy_a 张量，同样包含从 ranges[0] 到 ranges[1] 的数列，步长为 ranges[2]
      torch::Tensor lazy_a = torch::arange(
          ranges[0],
          ranges[1],
          ranges[2],
          torch::TensorOptions(torch::kFloat).device(device));
      // 检验 a 和 lazy_a 是否近似相等
      AllClose(a, lazy_a);
    });
  }
}

TEST_F(LazyOpsTest, TestARangeOut) {
  // 创建随机张量 a，形状为 {4}，数据类型为 kFloat，放置在默认设备上
  torch::Tensor a = torch::randn(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在两组范围上进行迭代：{0.0, 100.0, 0.5} 和 {0.0, -100.0, -0.5}
  for (auto& ranges : std::vector<std::vector<float>>{
           {0.0, 100.0, 0.5}, {0.0, -100.0, -0.5}}) {
    // 创建张量 b，包含从 ranges[0] 到 ranges[1] 的数列，步长为 ranges[2]，并将结果存储在 a 中
    torch::Tensor b = torch::arange_out(a, ranges[0], ranges[1], ranges[2]);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备上
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在指定设备上创建 lazy_b 张量，包含从 ranges[0] 到 ranges[1] 的数列，步长为 ranges[2]
      torch::Tensor lazy_b =
          torch::arange_out(lazy_a, ranges[0], ranges[1], ranges[2]);
      // 检验 b 和 lazy_b 是否近似相等
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestDimARange) {
  // 创建随机张量 like，形状为 {2, 2}，数据类型为 kFloat，放置在默认设备上
  torch::Tensor like = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在第一维上创建张量 a，按照 like 张量的第一维度，步长为 1
  torch::Tensor a = torch::_dim_arange(like, 1);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 like 张量复制到指定设备上
    torch::Tensor lazy_like = CopyToDevice(like, device);
    // 在指定设备上创建 lazy_a 张量，按照 lazy_like 张量的第一维度，步长为 1
    torch::Tensor lazy_a = torch::_dim_arange(lazy_like, 1);
    // 检验 a 和 lazy_a 是否近似相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestBartlettWindow) {
  int window_length = 10;
  // 针对两种周期性进行迭代：非周期性和周期性
  for (bool periodic : {false, true}) {
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 创建 Bartlett 窗口张量 output
      torch::Tensor output = torch::bartlett_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

      // 在指定设备上创建 lazy_output 张量，对应相同的 Bartlett 窗口参数
      torch::Tensor lazy_output = torch::bartlett_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      // 检验 output 和 lazy_output 是否近似相等，设置相对误差和绝对误差的容差
      AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(LazyOpsTest, TestBlackmanWindow) {
  int window_length = 10;
  // 针对两种周期性进行迭代：非周期性和周期性
  for (bool periodic : {false, true}) {
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 创建 Blackman 窗口张量 output
      torch::Tensor output = torch::blackman_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 在指定设备上创建 lazy_output 张量，对应相同的 Blackman 窗口参数
      torch::Tensor lazy_output = torch::blackman_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      // 检验 output 和 lazy_output 是否近似相等，设置相对误差和绝对误差的容差
      AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(LazyOpsTest, TestHammingWindow) {
  double alpha = 0.54;
  double beta = 0.46;
  int window_length = 10;
  // 针对两种周期性进行迭代：非周期性和周期性
  for (bool periodic : {false, true}) {
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 创建 Hamming 窗口张量 output
      torch::Tensor output = torch::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

      // 在指定设备上创建 lazy_output 张量，对应相同的 Hamming 窗口参数
      torch::Tensor lazy_output = torch::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          torch::TensorOptions(torch::kFloat).device(device));
      // 检验 output 和 lazy_output 是否近似相等，设置相对误差和绝对误差的容差
      AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}
    // 对于每个设备执行以下操作，使用 lambda 表达式进行设备遍历
    ForEachDevice([&](const torch::Device& device) {
      // 在默认设备上创建 Hamming 窗口张量
      torch::Tensor output = torch::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 在当前设备上创建 Hamming 窗口张量，设备由 lambda 表达式中的 device 参数确定
      torch::Tensor lazy_output = torch::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          torch::TensorOptions(torch::kFloat).device(device));
      // 检查两个张量是否在数值上接近
      AllClose(output, lazy_output);
    });
  }


这段代码通过 `ForEachDevice` 函数遍历每个设备，使用 lambda 表达式对每个设备执行以下操作：

1. 创建默认设备上的 Hamming 窗口张量 `output`。
2. 使用 lambda 表达式中的当前设备参数创建对应设备上的 Hamming 窗口张量 `lazy_output`。
3. 检查在两个设备上创建的张量 `output` 和 `lazy_output` 是否在数值上接近。

注释详细解释了每个操作的目的和使用情况。
}

// 定义一个测试用例 TestHannWindow
TEST_F(LazyOpsTest, TestHannWindow) {
  // 设置窗口长度为10
  int window_length = 10;
  // 对于每种周期性，执行以下操作
  for (bool periodic : {false, true}) {
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 使用 torch::hann_window 创建输出张量 output
      torch::Tensor output = torch::hann_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 使用 torch::hann_window 在指定设备上创建 lazy_output
      torch::Tensor lazy_output = torch::hann_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      // 检验 output 和 lazy_output 是否近似相等
      AllClose(output, lazy_output);
    });
  }
}

// 定义一个测试用例 TestLogSigmoid
TEST_F(LazyOpsTest, TestLogSigmoid) {
  // 创建形状为 {2, 2} 的空张量 a，随机初始化在默认设备上
  torch::Tensor a = torch::empty(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在范围 [-1.0, 1.0] 内均匀分布填充张量 a
  a.uniform_(-1.0, 1.0);
  // 对张量 a 执行 torch::log_sigmoid 操作，结果存储在张量 b 中
  torch::Tensor b = torch::log_sigmoid(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上执行 torch::log_sigmoid 操作，结果存储在 lazy_b 中
    torch::Tensor lazy_b = torch::log_sigmoid(lazy_a);
    // 检验 b 和 lazy_b 是否近似相等，使用指定的相对和绝对误差容限
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

// 定义一个测试用例 TestLogSigmoidForward
TEST_F(LazyOpsTest, TestLogSigmoidForward) {
  // 创建形状为 {2, 2} 的空张量 a，随机初始化在默认设备上
  torch::Tensor a = torch::empty(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在范围 [-1.0, 1.0] 内均匀分布填充张量 a
  a.uniform_(-1.0, 1.0);
  // 执行 torch::log_sigmoid_forward 操作，返回结果存储在 tuple 中
  auto tuple = torch::log_sigmoid_forward(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上执行 torch::log_sigmoid_forward 操作，返回结果存储在 lazy_tuple 中
    auto lazy_tuple = torch::log_sigmoid_forward(lazy_a);
    // 检验 tuple 的第一个元素与 lazy_tuple 的第一个元素是否近似相等，使用指定的相对和绝对误差容限
    AllClose(
        std::get<0>(tuple),
        std::get<0>(lazy_tuple),
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
    // 检验 tuple 的第二个元素与 lazy_tuple 的第二个元素是否近似相等，使用指定的相对和绝对误差容限
    AllClose(
        std::get<1>(tuple),
        std::get<1>(lazy_tuple),
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
  });
}

// 定义一个测试用例 TestLogsumexp
TEST_F(LazyOpsTest, TestLogsumexp) {
  // 创建形状为 {3, 4, 3} 的随机张量 a，存储在默认设备上
  torch::Tensor a = torch::rand(
      {3, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于给定的维度列表，执行以下操作
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    // 对于每种 keepdim，执行以下操作
    for (bool keepdim : {false, true}) {
      // 执行 torch::logsumexp 操作，结果存储在张量 b 中
      torch::Tensor b = torch::logsumexp(a, dims, keepdim);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 a 复制到指定设备上得到 lazy_a
        torch::Tensor lazy_a = CopyToDevice(a, device);
        // 在指定设备上执行 torch::logsumexp 操作，结果存储在 lazy_b 中
        torch::Tensor lazy_b = torch::logsumexp(lazy_a, dims, keepdim);
        // 检验 b 和 lazy_b 是否近似相等
        AllClose(b, lazy_b);
      });
    }
  }
}

// 定义一个测试用例 TestSiLU
TEST_F(LazyOpsTest, TestSiLU) {
  // 创建形状为 {2, 2} 的随机张量 a，存储在默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 执行 torch::silu 操作，结果存储在张量 b 中
  torch::Tensor b = torch::silu(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上执行 torch::silu 操作，结果存储在 lazy_b 中
    torch::Tensor lazy_b = torch::silu(lazy_a);
    // 检验 b 和 lazy_b 是否近似相等，使用指定的相对和绝对误差容限
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  // 检验 "lazy::silu_out" 的计数器是否符合预期
  ExpectCounterChanged("lazy::silu_out", GetIgnoredCounters());
}

// 定义一个测试用例 TestSigmoid
TEST_F(LazyOpsTest, TestSigmoid) {
  // 创建形状为 {2, 2} 的随机张量 a，存储在默认设备上
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 执行 torch::sigmoid 操作，结果存储在张量 b 中
  torch::Tensor b = torch::sigmoid(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上执行 torch::sigmoid 操作，结果存储在 lazy_b 中
    torch::Tensor lazy_b = torch::sigmoid(lazy_a);
    //```
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);



    // 使用 AllClose 函数比较 b 和 lazy_b 两个变量的值是否近似相等，
    // 其中 rtol 参数表示相对容差为 1e-3，
    // atol 参数表示绝对容差为 1e-5。
    // 这是一个匿名函数或者 Lambda 函数的调用，比较函数的结果。
  });
}

TEST_F(LazyOpsTest, TestMatmul_1x1) {
  // 创建大小为4的随机张量a，设备为默认设备，数据类型为float
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为4的随机张量b，设备为默认设备，数据类型为float
  torch::Tensor b = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的矩阵乘积
  torch::Tensor c = torch::matmul(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，并命名为lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，并命名为lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算在指定设备上的lazy_a和lazy_b的矩阵乘积
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    // 检查c和lazy_c是否在指定容差范围内相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_2x1) {
  // 创建大小为(3, 4)的随机张量a，设备为默认设备，数据类型为float
  torch::Tensor a = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为4的随机张量b，设备为默认设备，数据类型为float
  torch::Tensor b = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的矩阵乘积
  torch::Tensor c = torch::matmul(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，并命名为lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，并命名为lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算在指定设备上的lazy_a和lazy_b的矩阵乘积
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    // 检查c和lazy_c是否在指定容差范围内相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_1x2) {
  // 创建大小为4的随机张量a，设备为默认设备，数据类型为float
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为(4, 3)的随机张量b，设备为默认设备，数据类型为float
  torch::Tensor b = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的矩阵乘积
  torch::Tensor c = torch::matmul(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，并命名为lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，并命名为lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算在指定设备上的lazy_a和lazy_b的矩阵乘积
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    // 检查c和lazy_c是否在指定容差范围内相似
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_2x2) {
  // 创建大小为(2, 4)的随机张量a，设备为默认设备，数据类型为float
  torch::Tensor a = torch::rand(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为(4, 3)的随机张量b，设备为默认设备，数据类型为float
  torch::Tensor b = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的矩阵乘积
  torch::Tensor c = torch::matmul(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，并命名为lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，并命名为lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算在指定设备上的lazy_a和lazy_b的矩阵乘积
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    // 检查c和lazy_c是否在指定容差范围内相似，设置相对容差为1e-3，绝对容差为1e-4
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestMatmulBcast) {
  // 创建大小为(4, 2, 3, 2, 4)的随机张量a，设备为默认设备，数据类型为float
  torch::Tensor a = torch::rand(
      {4, 2, 3, 2, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为(2, 1, 4, 3)的随机张量b，设备为默认设备，数据类型为float
  torch::Tensor b = torch::rand(
      {2, 1, 4, 3},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的矩阵乘积
  torch::Tensor c = torch::matmul(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，并命名为lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，并命名为lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 计算在指定设备上的lazy_a和lazy_b的矩阵乘积
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    // 检查c和lazy_c是否在指定容差范围内相似
    AllClose(c, lazy_c);
  });
}
TEST_F(LazyOpsTest, TestDot) {
  // 创建大小为4的随机张量a和b，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的点积，存储在张量c中
  torch::Tensor c = torch::dot(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a和b复制到指定设备上，生成lazy_a和lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在当前设备上计算lazy_a和lazy_b的点积，存储在lazy_c中
    torch::Tensor lazy_c = torch::dot(lazy_a, lazy_b);
    // 检查张量c和lazy_c是否在当前设备上全部接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestTensorDot) {
  // 创建大小为[6, 4, 8]和[4, 7, 8]的随机张量a和b，存储在默认设备上
  torch::Tensor a = torch::rand(
      {6, 4, 8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4, 7, 8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 指定维度，计算张量a和b的张量点积，存储在张量c中
  std::vector<int64_t> dims_a = {1, 2};
  std::vector<int64_t> dims_b = {0, 2};
  torch::Tensor c = torch::tensordot(a, b, dims_a, dims_b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a和b复制到指定设备上，生成lazy_a和lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在当前设备上计算lazy_a和lazy_b的张量点积，存储在lazy_c中
    torch::Tensor lazy_c = torch::tensordot(lazy_a, lazy_b, dims_a, dims_b);
    // 检查张量c和lazy_c是否在当前设备上全部接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGer) {
  // 创建大小为4和5的随机张量a和b，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的外积，存储在张量c中
  torch::Tensor c = torch::ger(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a和b复制到指定设备上，生成lazy_a和lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在当前设备上计算lazy_a和lazy_b的外积，存储在lazy_c中
    torch::Tensor lazy_c = torch::ger(lazy_a, lazy_b);
    // 检查张量c和lazy_c是否在当前设备上全部接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMv) {
  // 创建大小为[4, 3]和[3]的随机张量a和b，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a和b的矩阵-向量乘积，存储在张量c中
  torch::Tensor c = torch::mv(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a和b复制到指定设备上，生成lazy_a和lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在当前设备上计算lazy_a和lazy_b的矩阵-向量乘积，存储在lazy_c中
    torch::Tensor lazy_c = torch::mv(lazy_a, lazy_b);
    // 检查张量c和lazy_c是否在当前设备上全部接近
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMvOut) {
  // 创建大小为[4, 3]和[3]的随机张量a和b，存储在默认设备上
  torch::Tensor a = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个空张量c，存储在默认设备上
  torch::Tensor c = torch::empty(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在张量c上计算张量a和b的矩阵-向量乘积
  torch::mv_out(c, a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a和b复制到指定设备上，生成lazy_a和lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 创建一个与lazy_b相同设备和数据类型的空张量lazy_c
    torch::Tensor lazy_c = torch::empty({4}, lazy_b.options());
    // 在当前设备上计算lazy_a和lazy_b的矩阵-向量乘积，存储在lazy_c中
    torch::mv_out(lazy_c, lazy_a, lazy_b);
    // 检查张量c和lazy_c是否在当前设备上全部接近
    AllClose(c, lazy_c);
  });
}
TEST_F(LazyOpsTest, TestBatchAddBatchMatMul) {
  // 创建大小为 {3, 6, 5} 的随机浮点张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {3, 6, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 6, 4} 的随机浮点张量 b，使用默认设备
  torch::Tensor b = torch::rand(
      {3, 6, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 4, 5} 的随机浮点张量 c，使用默认设备
  torch::Tensor c = torch::rand(
      {3, 4, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置标量 alpha 为 0.5
  torch::Scalar alpha = 0.5;
  // 设置标量 beta 为 1.5
  torch::Scalar beta = 1.5;
  // 执行 baddbmm 操作，并将结果存储在张量 d 中
  torch::Tensor d = torch::baddbmm(a, b, c, beta, alpha);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并创建 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备并创建 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量 c 复制到指定设备并创建 lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在指定设备上执行 baddbmm 操作，并将结果存储在 lazy_d 中
    torch::Tensor lazy_d = torch::baddbmm(lazy_a, lazy_b, lazy_c, beta, alpha);
    // 检查 d 和 lazy_d 是否在给定的相对误差和绝对误差下全部相等
    AllClose(d, lazy_d, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestBatchAddBatchMatMulInPlace) {
  // 创建大小为 {3, 6, 5} 的随机浮点张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {3, 6, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 6, 4} 的随机浮点张量 b，使用默认设备
  torch::Tensor b = torch::rand(
      {3, 6, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 4, 5} 的随机浮点张量 c，使用默认设备
  torch::Tensor c = torch::rand(
      {3, 4, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置标量 alpha 为 0.5
  torch::Scalar alpha = 0.5;
  // 设置标量 beta 为 1.5
  torch::Scalar beta = 1.5;
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并创建 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备并创建 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量 c 复制到指定设备并创建 lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在指定设备上执行 baddbmm_ 操作，直接修改 lazy_a，并将结果存储在 d 中
    torch::Tensor d = a.baddbmm_(b, c, beta, alpha);
    // 在指定设备上执行 baddbmm_ 操作，直接修改 lazy_a，并将结果存储在 lazy_d 中
    torch::Tensor lazy_d = lazy_a.baddbmm_(lazy_b, lazy_c, beta, alpha);
    // 检查 d 和 lazy_d 是否在给定的相对误差和绝对误差下全部相等
    AllClose(d, lazy_d, /*rtol=*/1e-3, /*atol=*/1e-4);
    // 检查 a 和 lazy_a 是否在给定的相对误差和绝对误差下全部相等
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestBatchMatMul) {
  // 创建大小为 {3, 6, 4} 的随机浮点张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {3, 6, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 4, 5} 的随机浮点张量 b，使用默认设备
  torch::Tensor b = torch::rand(
      {3, 4, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 执行 bmm 操作，并将结果存储在张量 c 中
  torch::Tensor c = torch::bmm(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并创建 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备并创建 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上执行 bmm 操作，并将结果存储在 lazy_c 中
    torch::Tensor lazy_c = torch::bmm(lazy_a, lazy_b);
    // 检查 c 和 lazy_c 是否在给定的相对误差和绝对误差下全部相等
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestChainMatMul) {
  // 创建大小为 {5, 4} 的随机浮点张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {5, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {4, 6} 的随机浮点张量 b，使用默认设备
  torch::Tensor b = torch::rand(
      {4, 6}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {6, 2} 的随机浮点张量 c，使用默认设备
  torch::Tensor c = torch::rand(
      {6, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {2, 7} 的随机浮点张量 d，使用默认设备
  torch::Tensor d = torch::rand(
      {2, 7}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 执行 chain_matmul 操作，并将结果存储在张量 result 中
  torch::Tensor result = torch::chain_matmul({a, b, c, d});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备并创建 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备并创建 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上执行 chain_matmul 操作，并将结果存储在 lazy_result 中
    torch::Tensor lazy_result = torch::chain_matmul({lazy_a, lazy_b, c, d});
    // 检查 result 和 lazy_result 是否在给定的相对误差和绝对误差下全部相等
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}
    # 将张量 b 复制到指定的设备上，返回复制后的张量
    torch::Tensor lazy_b = CopyToDevice(b, device);
    
    # 将张量 c 复制到指定的设备上，返回复制后的张量
    torch::Tensor lazy_c = CopyToDevice(c, device);
    
    # 将张量 d 复制到指定的设备上，返回复制后的张量
    torch::Tensor lazy_d = CopyToDevice(d, device);
    
    # 使用链式矩阵乘法计算张量列表 {lazy_a, lazy_b, lazy_c, lazy_d} 的乘积，返回结果张量
    torch::Tensor lazy_result =
        torch::chain_matmul({lazy_a, lazy_b, lazy_c, lazy_d});
    
    # 检查 lazy_result 和 result 是否在相对误差 rtol=1e-3 和绝对误差 atol=1e-4 的范围内全部相等
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-4);
TEST_F(LazyOpsTest, TestLinear) {
  // 创建一个形状为 {2, 4} 的随机张量 input，并将其放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 {3, 4} 的随机张量 weight，并将其放置在默认设备上
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 {3} 的随机张量 bias，并将其放置在默认设备上
  torch::Tensor bias = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 使用 torch::linear 函数计算输入 input 和权重 weight 的线性变换结果
  torch::Tensor result = torch::linear(input, weight);
  // 使用 torch::linear 函数计算输入 input、权重 weight 和偏置 bias 的线性变换结果
  torch::Tensor result_with_bias = torch::linear(input, weight, bias);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 将 weight 张量复制到指定设备上，得到 lazy_weight
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    // 将 bias 张量复制到指定设备上，得到 lazy_bias
    torch::Tensor lazy_bias = CopyToDevice(bias, device);
    // 使用 torch::linear 函数计算 lazy_input 和 lazy_weight 的线性变换结果
    torch::Tensor lazy_result = torch::linear(lazy_input, lazy_weight);
    // 使用 torch::linear 函数计算 lazy_input、lazy_weight 和 lazy_bias 的线性变换结果
    torch::Tensor lazy_result_with_bias =
        torch::linear(lazy_input, lazy_weight, lazy_bias);
    // 检查 result 和 lazy_result 是否在指定的相对和绝对容差范围内全部相等
    AllClose(result, lazy_result, /*rtol=*/1e-2, /*atol=*/1e-4);
    // 检查 result_with_bias 和 lazy_result_with_bias 是否在指定的相对和绝对容差范围内全部相等
    AllClose(
        result_with_bias,
        lazy_result_with_bias,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestPinverse) {
  // 创建一个形状为 {4, 6} 的随机张量 input，并将其放置在默认设备上
  torch::Tensor input = torch::rand(
      {4, 6}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 使用 torch::pinverse 函数计算 input 的伪逆
  torch::Tensor result = torch::pinverse(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 torch::pinverse 函数计算 lazy_input 的伪逆
    torch::Tensor lazy_result = torch::pinverse(lazy_input);
    // 检查 result 和 lazy_result 是否在指定的相对容差范围内全部相等
    AllClose(result, lazy_result, /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestEinsumOuter) {
  // 创建一个形状为 {5} 的随机张量 a，并将其放置在默认设备上
  torch::Tensor a = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 {5} 的随机张量 b，并将其放置在默认设备上
  torch::Tensor b = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 指定 einsum 方程式为 "i,j->ij"，计算张量 a 和 b 的外积
  std::string equation = "i,j->ij";
  torch::Tensor c = torch::einsum(equation, {a, b});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a 张量复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 b 张量复制到指定设备上，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 使用 torch::einsum 函数根据给定的方程式计算 lazy_a 和 lazy_b 的结果 lazy_c
    torch::Tensor lazy_c = torch::einsum(equation, {lazy_a, lazy_b});
    // 检查 c 和 lazy_c 是否全部相等
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEinsumOuterBackward) {
  // 创建一个形状为 {5} 的随机张量 a，同时标记为需要梯度计算
  torch::Tensor a = torch::rand(
      {5},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  // 创建一个形状为 {5} 的随机张量 b，同时标记为需要梯度计算
  torch::Tensor b = torch::rand(
      {5},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  // 指定 einsum 方程式为 "i,j->ij"
  std::string equation = "i,j->ij";
  // 定义一个测试函数 testfn，使用 torch::einsum 计算给定输入的结果
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 在指定设备上测试反向传播函数 TestBackward，使用 testfn 作为计算函数
    TestBackward({a, b}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}
// 定义测试函数 TestEinsumBatchMatMul，用于测试批量矩阵乘法的 einsum 操作
TEST_F(LazyOpsTest, TestEinsumBatchMatMul) {
  // 创建随机初始化的张量 a，形状为 {3, 2, 5}，数据类型为 float，设备为默认设备
  torch::Tensor a = torch::rand(
      {3, 2, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机初始化的张量 b，形状为 {3, 5, 4}，数据类型为 float，设备为默认设备
  torch::Tensor b = torch::rand(
      {3, 5, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 einsum 方程字符串
  std::string equation = "bij,bjk->bik";
  // 执行 einsum 操作，计算张量 c
  torch::Tensor c = torch::einsum(equation, {a, b});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备，得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上执行 einsum 操作，计算 lazy_c
    torch::Tensor lazy_c = torch::einsum(equation, {lazy_a, lazy_b});
    // 验证 lazy_c 是否与 c 在指定设备上全部接近
    AllClose(c, lazy_c);
  });
}

// 定义测试函数 TestEinsumPyTorchLowerBilinear，用于测试下三角双线性 einsum 操作
TEST_F(LazyOpsTest, TestEinsumPyTorchLowerBilinear) {
  // 创建随机初始化的张量 a，形状为 {3, 5, 4}，数据类型为 float，设备为默认设备
  torch::Tensor a = torch::rand(
      {3, 5, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机初始化的张量 l，形状为 {2, 5}，数据类型为 float，设备为默认设备
  torch::Tensor l = torch::rand(
      {2, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机初始化的张量 r，形状为 {2, 4}，数据类型为 float，设备为默认设备
  torch::Tensor r = torch::rand(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 einsum 方程字符串
  std::string equation = "bn,anm,bm->ba";
  // 执行 einsum 操作，计算张量 c
  torch::Tensor c = torch::einsum(equation, {l, a, r});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 l 复制到指定设备，得到 lazy_l
    torch::Tensor lazy_l = CopyToDevice(l, device);
    // 将张量 a 复制到指定设备，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 r 复制到指定设备，得到 lazy_r
    torch::Tensor lazy_r = CopyToDevice(r, device);
    // 在指定设备上执行 einsum 操作，计算 lazy_c
    torch::Tensor lazy_c = torch::einsum(equation, {lazy_l, lazy_a, lazy_r});
    // 验证 lazy_c 是否与 c 在指定设备上全部接近
    AllClose(c, lazy_c);
  });
}

// 定义测试函数 TestEinsumPyTorchLowerDiagonal，用于测试下三角对角线 einsum 操作
TEST_F(LazyOpsTest, TestEinsumPyTorchLowerDiagonal) {
  // 创建随机初始化的张量 input，形状为 {3, 3}，数据类型为 float，设备为默认设备
  torch::Tensor input = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 einsum 方程字符串
  std::string equation = "ii->i";
  // 执行 einsum 操作，计算张量 result
  torch::Tensor result = torch::einsum(equation, {input});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上执行 einsum 操作，计算 lazy_result
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_input});
    // 验证 lazy_result 是否与 result 在指定设备上全部接近
    AllClose(result, lazy_result);
  });
}

// 定义测试函数 TestEinsumPyTorchLowerBatchDiagonal，用于测试下三角批量对角线 einsum 操作
TEST_F(LazyOpsTest, TestEinsumPyTorchLowerBatchDiagonal) {
  // 创建随机初始化的张量 input，形状为 {4, 3, 3}，数据类型为 float，设备为默认设备
  torch::Tensor input = torch::rand(
      {4, 3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 einsum 方程字符串
  std::string equation = "...ii->...i";
  // 执行 einsum 操作，计算张量 result
  torch::Tensor result = torch::einsum(equation, {input});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上执行 einsum 操作，计算 lazy_result
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_input});
    // 验证 lazy_result 是否与 result 在指定设备上全部接近
    AllClose(result, lazy_result);
  });
}

// 定义测试函数 TestEinsumPyTorchLowerBatchPermute，用于测试下三角批量转置 einsum 操作
TEST_F(LazyOpsTest, TestEinsumPyTorchLowerBatchPermute) {
  // 创建随机初始化的张量 input，形状为 {2, 3, 4, 5}，数据类型为 float，设备为默认设备
  torch::Tensor input = torch::rand(
      {2, 3, 4, 5},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 einsum 方程字符串
  std::string equation = "...ij->...ji";
  // 执行 einsum 操作，计算张量 result
  torch::Tensor result = torch::einsum(equation, {input});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 input 复制到指定设备，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上执行 einsum 操作，计算 lazy_result
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_input});
    // 验证 lazy_result 是否与 result 在指定设备上全部接近
    AllClose(result, lazy_result);
  });
}
TEST_F(LazyOpsTest, TestEinsumPyTorchLowerRepeatedAxis) {
  // 创建形状为 {2, 3, 3} 的随机张量 x，使用默认设备并指定数据类型为 float
  torch::Tensor x = torch::rand(
      {2, 3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建形状为 {4} 的随机张量 y，使用默认设备并指定数据类型为 float
  torch::Tensor y = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义字符串方程 "ijj,k->ik"
  std::string equation = "ijj,k->ik";
  // 使用 einsum 函数根据方程计算结果张量 result
  torch::Tensor result = torch::einsum(equation, {x, y});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 x 拷贝到指定设备上，得到 lazy_x
    torch::Tensor lazy_x = CopyToDevice(x, device);
    // 将张量 y 拷贝到指定设备上，得到 lazy_y
    torch::Tensor lazy_y = CopyToDevice(y, device);
    // 使用 einsum 函数根据方程计算结果张量 lazy_result
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_x, lazy_y});
    // 检查 result 和 lazy_result 在所有设备上是否近似相等
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBilinear) {
  // 定义批大小、输入特征数和输出特征数
  int batch_size = 16;
  int in1_features = 4;
  int in2_features = 6;
  int out_features = 8;
  // 创建形状为 {batch_size, in1_features} 的随机张量 input1
  torch::Tensor input1 = torch::rand(
      {batch_size, in1_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建形状为 {batch_size, in2_features} 的随机张量 input2
  torch::Tensor input2 = torch::rand(
      {batch_size, in2_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建形状为 {out_features, in1_features, in2_features} 的随机张量 weight
  torch::Tensor weight = torch::rand(
      {out_features, in1_features, in2_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建形状为 {out_features} 的随机张量 bias
  torch::Tensor bias = torch::rand(
      {out_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input1 拷贝到指定设备上，得到 lazy_input1
    torch::Tensor lazy_input1 = CopyToDevice(input1, device);
    // 将 input2 拷贝到指定设备上，得到 lazy_input2
    torch::Tensor lazy_input2 = CopyToDevice(input2, device);
    // 将 weight 拷贝到指定设备上，得到 lazy_weight
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    // 将 bias 拷贝到指定设备上，得到 lazy_bias
    torch::Tensor lazy_bias = CopyToDevice(bias, device);
    // 使用 bilinear 函数计算结果张量 result
    torch::Tensor result = torch::bilinear(input1, input2, weight, bias);
    // 使用 bilinear 函数计算结果张量 lazy_result
    torch::Tensor lazy_result =
        torch::bilinear(lazy_input1, lazy_input2, lazy_weight, lazy_bias);
    // 检查 result 和 lazy_result 在所有设备上是否近似相等
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2D) {
  // 定义批大小、输入高度、宽度、上采样后的高度和宽度，以及通道数
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  // 创建形状为 {batch_size, chans, h, w} 的随机张量 input
  torch::Tensor input = torch::rand(
      {batch_size, chans, h, w},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 拷贝到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 upsample_nearest2d 函数对 input 进行上采样得到结果张量 result
    torch::Tensor result = torch::upsample_nearest2d(input, {uh, uw});
    // 使用 upsample_nearest2d 函数对 lazy_input 进行上采样得到结果张量 lazy_result
    torch::Tensor lazy_result = torch::upsample_nearest2d(lazy_input, {uh, uw});
    // 检查 result 和 lazy_result 在所有设备上是否近似相等
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DBackward) {
  // 定义批大小、输入高度、宽度、上采样后的高度和宽度，以及通道数
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  // 定义测试函数 testfn，接收输入张量并进行上采样
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::upsample_nearest2d(inputs[0], {uh, uw});
  };
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 在此处将代码片段补充完整以保证完整性
    TestBackward(
        {torch::rand(  // 使用 torch 库生成一个随机张量，形状为 {batch_size, chans, h, w}
            {batch_size, chans, h, w},  // 张量的形状参数
            torch::TensorOptions(torch::kFloat)  // 指定张量数据类型为浮点数类型
                .device(DefaultDevice())  // 设置张量的设备为默认设备
                .requires_grad(true))},  // 设置张量需要计算梯度
        device,  // 将设备作为参数传递给 TestBackward 函数
        testfn);  // 将 testfn 函数作为参数传递给 TestBackward 函数
  });
TEST_F(LazyOpsTest, TestUpsampleNearest2DWithScale) {
  // 定义测试用例参数
  int batch_size = 2;  // 批大小为2
  int h = 5;           // 输入图像高度为5
  int w = 5;           // 输入图像宽度为5
  int chans = 2;       // 通道数为2
  double scale_h = 2.5;  // 垂直方向的缩放比例为2.5
  double scale_w = 3.4;  // 水平方向的缩放比例为3.4
  
  // 生成随机输入张量
  torch::Tensor input = torch::rand(
      {batch_size, chans, h, w},  // 输入张量维度
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));  // 指定张量选项和设备
  
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 执行最近邻插值上采样
    torch::Tensor result = torch::upsample_nearest2d(
        input, c10::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    // 在复制到设备上的输入上执行最近邻插值上采样
    torch::Tensor lazy_result = torch::upsample_nearest2d(
        lazy_input, c10::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    // 检查两个结果张量是否全部接近
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DBackwardWithScale) {
  // 定义测试函数
  int batch_size = 2;  // 批大小为2
  int h = 5;           // 输入图像高度为5
  int w = 5;           // 输入图像宽度为5
  int chans = 2;       // 通道数为2
  double scale_h = 2.5;  // 垂直方向的缩放比例为2.5
  double scale_w = 3.4;  // 水平方向的缩放比例为3.4
  
  // 定义测试函数
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::upsample_nearest2d(
        inputs[0], c10::nullopt, at::ArrayRef<double>{scale_h, scale_w});
  };
  
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 测试反向传播
    TestBackward(
        {torch::rand(
            {batch_size, chans, h, w},  // 输入张量维度
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},  // 需要梯度
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestUpsampleBilinear2D) {
  // 定义测试用例参数
  int batch_size = 2;  // 批大小为2
  int h = 5;           // 输入图像高度为5
  int w = 5;           // 输入图像宽度为5
  int uh = 8;          // 上采样后的高度为8
  int uw = 8;          // 上采样后的宽度为8
  int chans = 2;       // 通道数为2
  
  // 对每个对齐角落参数执行以下操作
  for (bool align_corners : {true, false}) {
    // 生成随机输入张量
    torch::Tensor input = torch::rand(
        {batch_size, chans, h, w},  // 输入张量维度
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));  // 指定张量选项和设备
    
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 执行双线性插值上采样
      torch::Tensor result =
          torch::upsample_bilinear2d(input, {uh, uw}, align_corners);
      // 在复制到设备上的输入上执行双线性插值上采样
      torch::Tensor lazy_result =
          torch::upsample_bilinear2d(lazy_input, {uh, uw}, align_corners);
      // 检查两个结果张量是否全部接近
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestUpsampleBilinear2DBackward) {
  // 定义测试用例参数
  int batch_size = 2;  // 批大小为2
  int h = 5;           // 输入图像高度为5
  int w = 5;           // 输入图像宽度为5
  int uh = 8;          // 上采样后的高度为8
  int uw = 8;          // 上采样后的宽度为8
  int chans = 2;       // 通道数为2
  
  // 对每个对齐角落参数执行以下操作
  for (bool align_corners : {true, false}) {
    // 定义测试函数
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::upsample_bilinear2d(inputs[0], {uh, uw}, align_corners);
    };
    
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 测试反向传播
      TestBackward(
          {torch::rand(
              {batch_size, chans, h, w},  // 输入张量维度
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},  // 需要梯度
          device,
          testfn);
    });
  }
}
TEST_F(LazyOpsTest, TestAddCMul) {
  // 创建大小为2x2的随机浮点数张量a，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为2x2的随机浮点数张量b，设备为默认设备
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为2x2的随机浮点数张量c，设备为默认设备
  torch::Tensor c = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 执行张量a和张量b的加权乘法加和运算，系数为3.1165，结果赋给张量d
  torch::Tensor d = torch::addcmul(a, b, c, 3.1165);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，得到lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量c复制到指定设备上，得到lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在指定设备上执行加权乘法加和运算，系数为3.1165，结果赋给lazy_d
    torch::Tensor lazy_d = torch::addcmul(lazy_a, lazy_b, lazy_c, 3.1165);
    // 检查d和lazy_d在指定设备上的近似相等性
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestAddCDiv) {
  // 创建大小为2x2的随机浮点数张量a，设备为默认设备
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为2x2的随机浮点数张量b，设备为默认设备
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为2x2的随机浮点数张量c，取绝对值后加1，设备为默认设备
  torch::Tensor c =
      torch::abs(torch::rand(
          {2, 2},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()))) +
      1.0;
  // 执行张量a和张量b的加权除法加和运算，系数为3.1165，结果赋给张量d
  torch::Tensor d = torch::addcdiv(a, b, c, 3.1165);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，得到lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量c复制到指定设备上，得到lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在指定设备上执行加权除法加和运算，系数为3.1165，结果赋给lazy_d
    torch::Tensor lazy_d = torch::addcdiv(lazy_a, lazy_b, lazy_c, 3.1165);
    // 检查d和lazy_d在指定设备上的近似相等性
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestAddCDivWithBroadcast) {
  // 创建大小为1x3的随机浮点数张量a，设备为默认设备
  torch::Tensor a = torch::rand(
      {1, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为3x1的随机浮点数张量b，设备为默认设备
  torch::Tensor b = torch::rand(
      {3, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为1x3的随机浮点数张量c，取绝对值后加1，设备为默认设备
  torch::Tensor c =
      torch::abs(torch::rand(
          {1, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()))) +
      1.0;
  // 执行张量a和张量b的加权除法加和运算，系数为3.1165，结果赋给张量d
  torch::Tensor d = torch::addcdiv(a, b, c, 3.1165);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，得到lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，得到lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量c复制到指定设备上，得到lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在指定设备上执行加权除法加和运算，系数为3.1165，结果赋给lazy_d
    torch::Tensor lazy_d = torch::addcdiv(lazy_a, lazy_b, lazy_c, 3.1165);
    // 检查d和lazy_d在指定设备上的近似相等性
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestSize) {
  // 创建大小为2x1x4x6的随机浮点数张量input，设备为默认设备
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量input的维度数，赋给rank
  int rank = input.dim();
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量input复制到指定设备上，得到lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 遍历张量input的所有维度
    for (int dim = -rank; dim < rank; ++dim) {
      // 检查input和lazy_input在指定维度上的大小是否相等
      EXPECT_EQ(torch::size(input, dim), torch::size(lazy_input, dim));
    }
  });
}

TEST_F(LazyOpsTest, TestSelect) {
  // 定义整数向量input_sizes为{14, 24, 8}
  std::vector<int64_t> input_sizes = {14, 24, 8};
  // 获取input_sizes的维度数，赋给rank
  int rank = input_sizes.size();
  // 遍历input_sizes的所有维度
  for (int dim = -rank; dim < rank; ++dim) {
    # 定义一个 lambda 函数 testfn，接受一个包含 torch::Tensor 的向量 inputs 作为参数，并返回一个 torch::Tensor
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      # 在 inputs[0] 的指定维度 dim 上选择索引为 0 的元素，返回一个新的 torch::Tensor
      return torch::select(inputs[0], dim, 0);
    };
    # 对于每一个设备，执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      # 调用 TestBackward 函数进行反向传播测试
      TestBackward(
          # 创建一个包含随机数据的张量，并标记为需要梯度计算
          {torch::rand(
              input_sizes,
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          # 指定设备
          device,
          # 使用之前定义的 testfn 函数进行测试
          testfn);
    });
  };


这段代码片段主要是一个 C++ 的函数或者方法体，包含了 lambda 函数定义以及对每个设备执行反向传播测试的逻辑。
}

TEST_F(LazyOpsTest, TestBernoulliScalarProb) {
  // 创建一个大小为1000的全零张量，数据类型为float，放置在默认设备上
  torch::Tensor input = torch::zeros(
      1000, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对概率为0.1进行Bernoulli采样
    torch::Tensor lazy_output = torch::bernoulli(lazy_input, 0.1);
    // 计算非零元素的比例
    double frac = lazy_output.sum().item().toDouble() / input.numel();
    // 断言比例大于0.06
    EXPECT_GT(frac, 0.06);
    // 断言比例小于0.14
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliTensorProb) {
  // 创建一个包含1000个0.1的张量作为概率值
  std::vector<float> prob_values(1000, 0.1);
  torch::Tensor input = torch::tensor(
      prob_values, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对张量进行Bernoulli采样
    torch::Tensor lazy_output = torch::bernoulli(lazy_input);
    // 计算非零元素的比例
    double frac = lazy_output.sum().item().toDouble() / input.numel();
    // 断言比例大于0.06
    EXPECT_GT(frac, 0.06);
    // 断言比例小于0.14
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliScalarProbInPlace) {
  // 创建一个大小为1000的全零张量，数据类型为float，放置在默认设备上
  torch::Tensor input = torch::zeros(
      1000, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地对概率为0.1进行Bernoulli采样
    lazy_input.bernoulli_(0.1);
    // 计算非零元素的比例
    double frac = lazy_input.sum().item().toDouble() / input.numel();
    // 断言比例大于0.06
    EXPECT_GT(frac, 0.06);
    // 断言比例小于0.14
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliTensorProbInPlace) {
  // 创建一个大小为1000的全零张量，数据类型为float，放置在默认设备上
  torch::Tensor input = torch::zeros(
      1000, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个概率值为0.1的标量张量，放置在默认设备上
  torch::Tensor prob = torch::scalar_tensor(
      0.1, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量和概率张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_prob = CopyToDevice(prob, device);
    // 在原地对张量进行Bernoulli采样
    lazy_input.bernoulli_(lazy_prob);
    // 计算非零元素的比例
    double frac = lazy_input.sum().item().toDouble() / input.numel();
    // 断言比例大于0.06
    EXPECT_GT(frac, 0.06);
    // 断言比例小于0.14
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestDropout) {
  // 创建一个大小为{17, 21}的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {17, 21}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 对张量应用dropout操作，且在训练模式下
    torch::Tensor lazy_b = torch::dropout(lazy_a, 0.1, /*train=*/true);
    // 计算非零元素的比例
    double prob =
        static_cast<double>(lazy_b.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    // 断言比例大于0.86
    EXPECT_GT(prob, 0.86);
    // 断言比例小于0.94
    EXPECT_LT(prob, 0.94);
  });
}

TEST_F(LazyOpsTest, TestDropoutInPlace) {
  // 创建一个大小为{17, 21}的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {17, 21}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在原地对张量应用dropout操作，且在训练模式下
    torch::dropout_(lazy_a, 0.1, /*train=*/true);
    // 计算非零元素的比例
    double prob =
        static_cast<double>(lazy_a.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    # 断言：期望 prob 大于 0.85
    EXPECT_GT(prob, 0.85);
    # 断言：期望 prob 小于 0.94
    EXPECT_LT(prob, 0.94);
}

TEST_F(LazyOpsTest, TestRandperm) {
  // 设置随机数种子为 5
  unsigned n = 5;
  // 生成一个包含 n 个元素的随机排列张量，使用 Lazy 设备
  torch::Tensor shuffle = torch::randperm(
      n, torch::TensorOptions(torch::kLong).device(torch::kLazy));
  // 将 shuffle 张量复制到 CPU 设备
  torch::Tensor shuffle_cpu = CopyToDevice(shuffle, torch::kCPU);
  // 将 shuffle_cpu 转换为整数向量
  std::vector<int64_t> shuffle_data(
      shuffle_cpu.data_ptr<int64_t>(), shuffle_cpu.data_ptr<int64_t>() + n);
  // 检查 shuffle_data 是否是长度为 n 的排列
  EXPECT_TRUE(
      shuffle_data.size() == n && torch::lazy::IsPermutation(shuffle_data));
}

TEST_F(LazyOpsTest, TestSlice) {
  // 创建一个形状为 {32, 24, 16} 的随机张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {32, 24, 16},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对张量 a 进行切片操作，从第 1 维的索引 0 开始，长度为 16，步长为 1
  torch::Tensor b = torch::slice(a, 1, 0, 16, 1);
  // 针对每个设备执行操作，将张量 a 和 b 复制到指定设备，然后检查是否相等
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::slice(lazy_a, 1, 0, 16, 1);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTake) {
  // 创建一个形状为 {4, 4} 的随机浮点张量 a，使用默认设备
  torch::Tensor a = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 生成一个形状为 {5} 的随机整数张量 b，使用默认设备
  torch::Tensor b = torch::randint(
      16, {5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 使用张量 b 中的索引值从张量 a 中提取元素形成新张量 c
  torch::Tensor c = torch::take(a, b);
  // 针对每个设备执行操作，将张量 a、b 和 c 复制到指定设备，然后检查是否相等
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::take(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestTakeBackward) {
  // 定义一个函数 testfn，使用输入张量列表进行 take 操作
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::take(inputs[0], inputs[1]);
  };
  // 针对每个设备执行反向测试，使用随机浮点张量和随机整数张量作为输入
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
             {4, 4},
             torch::TensorOptions(torch::kFloat)
                 .device(DefaultDevice())
                 .requires_grad(true)),
         torch::randint(
             16,
             {5},
             torch::TensorOptions(torch::kLong).device(DefaultDevice()))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestStack) {
  // 创建三个形状相同的随机浮点张量 a、b、c，使用默认设备
  torch::Tensor a = torch::rand(
      {2, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {2, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的维度数量 + 1
  int rank = a.dim() + 1;
  // 对每个维度执行栈操作，创建一个沿指定维度堆叠的新张量 d
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor d = torch::stack({a, b, c}, dim);
    // 针对每个设备执行操作，将张量 a、b、c 和 d 复制到指定设备，然后检查是否相等
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::stack({lazy_a, lazy_b, lazy_c}, dim);
      AllClose(d, lazy_d);
    });
  }
}
TEST_F(LazyOpsTest, TestCat) {
  // 创建一个形状为 {2, 1, 3} 的随机浮点数张量 a，并指定在默认设备上进行操作
  torch::Tensor a = torch::rand(
      {2, 1, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 {2, 2, 3} 的随机浮点数张量 b，并指定在默认设备上进行操作
  torch::Tensor b = torch::rand(
      {2, 2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 {2, 3, 3} 的随机浮点数张量 c，并指定在默认设备上进行操作
  torch::Tensor c = torch::rand(
      {2, 3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对维度 {1, -2} 进行循环迭代
  for (int dim : {1, -2}) {
    // 在指定维度上连接张量 a, b, c，并创建张量 d
    torch::Tensor d = torch::cat({a, b, c}, dim);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 拷贝到指定设备上，得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 将张量 b 拷贝到指定设备上，得到 lazy_b
      torch::Tensor lazy_b = CopyToDevice(b, device);
      // 将张量 c 拷贝到指定设备上，得到 lazy_c
      torch::Tensor lazy_c = CopyToDevice(c, device);
      // 在指定维度上连接 lazy_a, lazy_b, lazy_c，并创建张量 lazy_d
      torch::Tensor lazy_d = torch::cat({lazy_a, lazy_b, lazy_c}, dim);
      // 断言 d 的大小和数据类型与 lazy_d 的大小和数据类型相同
      EXPECT_TRUE(d.sizes() == lazy_d.sizes() && d.dtype() == lazy_d.dtype());
      // 检查 d 和 lazy_d 是否接近
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestUnbind) {
  // 创建一个形状为 {4, 3, 7} 的随机浮点数张量 input，并指定在默认设备上进行操作
  torch::Tensor input = torch::rand(
      {4, 3, 7}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取张量的维度数
  int rank = input.dim();
  // 对每个维度进行循环迭代
  for (int dim = -rank; dim < rank; ++dim) {
    // 在指定维度上解绑张量 input，并得到输出列表 output
    std::vector<torch::Tensor> output = torch::unbind(input, dim);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 input 拷贝到指定设备上，得到 lazy_input
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定维度上解绑 lazy_input，并得到输出列表 lazy_output
      std::vector<torch::Tensor> lazy_output = torch::unbind(lazy_input, dim);
      // 断言输出列表的大小相同
      ASSERT_EQ(output.size(), lazy_output.size());
      // 对每个解绑后的张量进行检查，确保其在设备上的结果接近
      for (size_t i = 0; i < output.size(); ++i) {
        AllClose(output[i], lazy_output[i]);
      }
    });
  }
}

TEST_F(LazyOpsTest, TestRepeat) {
  // 定义重复操作的参数列表和输入形状列表
  std::vector<std::vector<int64_t>> repeats_list = {{4, 2}, {4, 2, 3}};
  std::vector<std::vector<int64_t>> input_size_list = {{3}, {2, 4}};
  // 对每组重复参数和输入形状进行循环迭代
  for (const auto& repeats : repeats_list) {
    for (const auto& input_size : input_size_list) {
      // 创建指定形状的随机浮点数张量 input，并指定在默认设备上进行操作
      torch::Tensor input = torch::rand(
          input_size,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 对 input 执行重复操作，并创建输出张量 output
      torch::Tensor output = input.repeat(repeats);
      // 对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 input 拷贝到指定设备上，得到 lazy_input
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 在指定设备上执行重复操作，并创建 lazy_output
        torch::Tensor lazy_output = lazy_input.repeat(repeats);
        // 检查 lazy_output 和 output 是否接近
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestGather) {
  // 创建一个形状为 {3, 3} 的随机浮点数张量 a，并指定在默认设备上进行操作
  torch::Tensor a = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 {3, 3} 的空长整型张量 b，并指定在默认设备上进行操作
  torch::Tensor b = torch::empty(
      {3, 3}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 初始化张量 b 的值
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = (i + j) % 3;
    }
  }
  // 对是否稀疏梯度进行循环迭代
  for (bool sparse_grad : {false, true}) {
    // 在指定维度上执行 gather 操作，并创建张量 c
    torch::Tensor c = torch::gather(a, 1, b, sparse_grad);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 拷贝到指定设备上，得到 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 将张量 b 拷贝到指定设备上，得到 lazy_b
      torch::Tensor lazy_b = CopyToDevice(b, device);
      // 在指定设备上执行 gather 操作，并创建 lazy_c
      torch::Tensor lazy_c = torch::gather(lazy_a, 1, lazy_b, sparse_grad);
      // 检查 c 和 lazy_c 是否接近
      AllClose(c, lazy_c);
    });
  }
}
TEST_F(LazyOpsTest, TestScatter) {
  // 创建大小为 {3, 5} 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 5} 的随机张量 b，使用默认设备上的浮点数选项
  torch::Tensor b = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 5} 的空张量 c，使用默认设备上的长整型选项
  torch::Tensor c = torch::empty(
      {3, 5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 遍历两个维度的循环
  for (int dim = 0; dim < 2; ++dim) {
    // 循环遍历张量的第一维度（i）和第二维度（j）
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        // 计算 (i + j) % c.sizes()[dim] 并赋值给 c[i][j]
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    // 对张量 a 执行 scatter 操作，维度为 dim，索引张量为 c，值张量为 b，并赋值给 d
    torch::Tensor d = torch::scatter(a, dim, c, b);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 拷贝到指定设备上，并命名为 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 将张量 b 拷贝到指定设备上，并命名为 lazy_b
      torch::Tensor lazy_b = CopyToDevice(b, device);
      // 将张量 c 拷贝到指定设备上，并命名为 lazy_c
      torch::Tensor lazy_c = CopyToDevice(c, device);
      // 对拷贝到设备上的张量 lazy_a 执行 scatter 操作，维度为 dim，索引张量为 lazy_c，值张量为 lazy_b，并赋值给 lazy_d
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, lazy_b);
      // 检查张量 d 和 lazy_d 在指定设备上是否全部接近
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterR1) {
  // 创建大小为 {5} 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {2} 的随机张量 b，使用默认设备上的浮点数选项
  torch::Tensor b = torch::rand(
      {2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {2} 的空张量 c，使用默认设备上的长整型选项
  torch::Tensor c = torch::empty(
      {2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 给张量 c 的元素赋值
  c[0] = 1;
  c[1] = 3;
  // 对张量 a 执行 scatter 操作，维度为 0，索引张量为 c，值张量为 b，并赋值给 d
  torch::Tensor d = torch::scatter(a, 0, c, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 拷贝到指定设备上，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 拷贝到指定设备上，并命名为 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量 c 拷贝到指定设备上，并命名为 lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 对拷贝到设备上的张量 lazy_a 执行 scatter 操作，维度为 0，索引张量为 lazy_c，值张量为 lazy_b，并赋值给 lazy_d
    torch::Tensor lazy_d = torch::scatter(lazy_a, 0, lazy_c, lazy_b);
    // 检查张量 d 和 lazy_d 在指定设备上是否全部接近
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestScatterR3) {
  // 创建大小为 {3, 5, 2} 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {3, 5, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 4, 2} 的随机张量 b，使用默认设备上的浮点数选项
  torch::Tensor b = torch::rand(
      {3, 4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 4, 2} 的空张量 c，使用默认设备上的长整型选项
  torch::Tensor c = torch::empty(
      {3, 4, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 遍历三个维度的循环
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 2; k++) {
        // 计算 (i + j + k) % 4 并赋值给 c[i][j][k]
        c[i][j][k] = (i + j + k) % 4;
      }
    }
  }
  // 对张量 a 执行 scatter 操作，维度为 1，索引张量为 c，值张量为 b，并赋值给 d
  torch::Tensor d = torch::scatter(a, 1, c, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 拷贝到指定设备上，并命名为 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 拷贝到指定设备上，并命名为 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量 c 拷贝到指定设备上，并命名为 lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 对拷贝到设备上的张量 lazy_a 执行 scatter 操作，维度为 1，索引张量为 lazy_c，值张量为 lazy_b，并赋值给 lazy_d
    torch::Tensor lazy_d = torch::scatter(lazy_a, 1, lazy_c, lazy_b);
    // 检查张量 d 和 lazy_d 在指定设备上是否全部接近
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestScatterBiggerSource) {
  // 创建大小为 {4, 4} 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {8, 8} 的随机张量 b，使用默认设备上的浮点数选项
  torch::Tensor b = torch::rand(
      {8, 8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {4, 4} 的空张量 c，使用默认设备上的长整型选项
  torch::Tensor c = torch::empty(
      {4, 4}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 遍历第一个维度的循环
  for (int i = 0; i < 4; i++) {
    // 循环计算矩阵 c 的元素
    for (int j = 0; j < 4; j++) {
      // 计算 c[i][j]，使用模运算确保结果在 0 到 3 之间循环
      c[i][j] = (i + j) % 4;
    }
  }
  // 遍历维度进行张量操作
  for (int dim = 0; dim < 2; ++dim) {
    // 在维度 dim 上进行张量散点操作，生成张量 d
    torch::Tensor d = torch::scatter(a, dim, c, b);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到指定设备，生成 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 将张量 b 复制到指定设备，生成 lazy_b
      torch::Tensor lazy_b = CopyToDevice(b, device);
      // 将矩阵 c 复制到指定设备，生成 lazy_c
      torch::Tensor lazy_c = CopyToDevice(c, device);
      // 在指定设备上进行张量散点操作，生成 lazy_d
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, lazy_b);
      // 验证 lazy_d 和 d 是否全部接近
      AllClose(d, lazy_d);
    });
  }
TEST_F(LazyOpsTest, TestScatterScalar) {
  // 创建一个大小为 4x4 的浮点数张量 a，初始化为随机值，设备为默认设备
  torch::Tensor a = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量张量 b，值为浮点数 1.0，用于scatter操作
  torch::Scalar b = 1.0f;
  // 创建一个大小为 4x4 的长整型张量 c，未初始化
  torch::Tensor c = torch::empty(
      {4, 4}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 嵌套循环为张量 c 赋值，对每个元素赋值为 (i + j) % 4
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  // 对每个维度执行 scatter 操作
  for (int dim = 0; dim < 2; ++dim) {
    // 执行 scatter 操作，将结果保存在张量 d 中
    torch::Tensor d = torch::scatter(a, dim, c, b);
    // 对每个设备执行操作，复制相应的张量到设备并进行 scatter 操作
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, b);
      // 检查结果是否全部接近
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterReduceAdd) {
  // 创建一个大小为 3x5 的浮点数张量 a，初始化为随机值，设备为默认设备
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 3x5 的浮点数张量 b，初始化为随机值，设备为默认设备
  torch::Tensor b = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 3x5 的长整型张量 c，未初始化
  torch::Tensor c = torch::empty(
      {3, 5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 对每个维度执行 scatter 操作
  for (int dim = 0; dim < 2; ++dim) {
    // 嵌套循环为张量 c 赋值，对每个元素赋值为 (i + j) % c.sizes()[dim]
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    // 执行 scatter_add 操作，将结果保存在张量 d 中
    torch::Tensor d = torch::scatter(a, dim, c, b, "add");
    // 对每个设备执行操作，复制相应的张量到设备并进行 scatter_add 操作
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, lazy_b, "add");
      // 检查结果是否全部接近
      AllClose(d, lazy_d);
    });
  }

  // 预期某些计数器不会改变
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 预期 lazy::scatter_out 计数器会改变
  ExpectCounterChanged("lazy::scatter_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestScatterAdd) {
  // 创建一个大小为 3x5 的浮点数张量 a，初始化为随机值，设备为默认设备
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 3x5 的浮点数张量 b，初始化为随机值，设备为默认设备
  torch::Tensor b = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 3x5 的长整型张量 c，未初始化
  torch::Tensor c = torch::empty(
      {3, 5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 对每个维度执行 scatter_add 操作
  for (int dim = 0; dim < 2; ++dim) {
    // 嵌套循环为张量 c 赋值，对每个元素赋值为 (i + j) % c.sizes()[dim]
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    // 执行 scatter_add 操作，将结果保存在张量 d 中
    torch::Tensor d = torch::scatter_add(a, dim, c, b);
    // 对每个设备执行操作，复制相应的张量到设备并进行 scatter_add 操作
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter_add(lazy_a, dim, lazy_c, lazy_b);
      // 检查结果是否全部接近
      AllClose(d, lazy_d);
    });
  }
}
TEST_F(LazyOpsTest, TestScatterAddInPlace) {
  // 创建一个大小为 4x4 的随机浮点数张量 b
  torch::Tensor b = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 4x4 的空长整型张量 c
  torch::Tensor c = torch::empty(
      {4, 4}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 初始化张量 c，使其每个元素等于 (i + j) % 4，其中 i 和 j 是索引
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  // 对于每个维度进行操作
  for (int dim = 0; dim < 2; ++dim) {
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 创建一个大小为 4x4 的随机浮点数张量 a，并复制到指定设备
      torch::Tensor a = torch::rand(
          {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 在维度 dim 上对张量 a 进行 scatter_add_ 操作，使用张量 c 和 b
      torch::Tensor d = a.scatter_add_(dim, c, b);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      // 对 lazy_a 在指定设备上进行 scatter_add_ 操作，使用 lazy_c 和 lazy_b
      torch::Tensor lazy_d = lazy_a.scatter_add_(dim, lazy_c, lazy_b);
      // 验证 d 和 lazy_d 是否近似相等
      AllClose(d, lazy_d);
      // 验证 a 和 lazy_a 是否近似相等
      AllClose(a, lazy_a);
    });
  }
}

TEST_F(LazyOpsTest, TestIndexSelect) {
  // 针对不同标量类型进行迭代
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型创建随机张量 a
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 对于每个索引标量类型进行迭代
    for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
      // 创建大小为 2 的空张量 b，根据索引标量类型
      torch::Tensor b = torch::empty(
          {2}, torch::TensorOptions(index_scalar_type).device(DefaultDevice()));
      b[0] = 0;
      b[1] = 2;
      // 对于每个偏移量进行迭代
      for (auto offset : {-2, 0}) {
        // 在指定偏移量和维度上使用索引张量 b 对张量 a 进行索引选择
        torch::Tensor c0 = torch::index_select(a, 0 + offset, b);
        torch::Tensor c1 = torch::index_select(a, 1 + offset, b);
        // 对每个设备执行操作
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          // 在指定设备上进行索引选择操作，使用 lazy_a、偏移量和 lazy_b
          torch::Tensor lazy_c0 =
              torch::index_select(lazy_a, 0 + offset, lazy_b);
          torch::Tensor lazy_c1 =
              torch::index_select(lazy_a, 1 + offset, lazy_b);
          // 验证 c0 和 lazy_c0 是否全等
          AllEqual(c0, lazy_c0);
          // 验证 c1 和 lazy_c1 是否全等
          AllEqual(c1, lazy_c1);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestIndexSelectRank0) {
  // 针对不同标量类型进行迭代
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型创建随机张量 a
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建标量张量 b，值为 2
    torch::Tensor b = torch::scalar_tensor(
        2, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    // 使用标量张量 b 对张量 a 进行索引选择
    torch::Tensor c0 = torch::index_select(a, 0, b);
    // 使用 torch 库的函数，根据索引从张量 a 中选择列 b，结果存储在张量 c1 中
    torch::Tensor c1 = torch::index_select(a, 1, b);
    
    // 对每个设备执行以下操作：将张量 a 和 b 复制到当前设备，生成 lazy_a 和 lazy_b
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 a 复制到当前设备，生成 lazy_a
      torch::Tensor lazy_a = CopyToDevice(a, device);
      // 将张量 b 复制到当前设备，生成 lazy_b
      torch::Tensor lazy_b = CopyToDevice(b, device);
    
      // 在当前设备上，根据 lazy_b 的索引选择 lazy_a 的行，结果存储在 lazy_c0 中
      torch::Tensor lazy_c0 = torch::index_select(lazy_a, 0, lazy_b);
      // 在当前设备上，根据 lazy_b 的索引选择 lazy_a 的列，结果存储在 lazy_c1 中
      torch::Tensor lazy_c1 = torch::index_select(lazy_a, 1, lazy_b);
    
      // 检查全局张量 c0 和当前设备上的 lazy_c0 是否相等
      AllEqual(c0, lazy_c0);
      // 检查全局张量 c1 和当前设备上的 lazy_c1 是否相等
      AllEqual(c1, lazy_c1);
    });
}

TEST_F(LazyOpsTest, TestInverse) {
  // 检查是否在 CUDA 上运行，如果是，则跳过测试，并注明原因
  if (IsCuda()) {
    GTEST_SKIP();
  }
  // 创建一个大小为 5x5 的随机张量 a，使用默认设备（CPU/GPU）
  torch::Tensor a = torch::randn(
      {5, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 的逆矩阵 b
  torch::Tensor b = torch::inverse(a);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 的逆矩阵 lazy_b
    torch::Tensor lazy_b = torch::inverse(lazy_a);
    // 检查 b 和 lazy_b 是否在指定的误差范围内相等
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestIsnan) {
  // 创建一个包含 NaN 值的张量 a
  torch::Tensor a = torch::tensor(
      {1.0, 2.0, std::nan("1"), 4.0},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 a 中每个元素是否为 NaN，得到张量 b
  torch::Tensor b = torch::isnan(a);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_a 中每个元素是否为 NaN，得到 lazy_b
    torch::Tensor lazy_b = torch::isnan(lazy_a);
    // 检查张量 b 和 lazy_b 是否完全相等
    AllEqual(b, lazy_b);
  });
  // 检查特定的计数器预期没有改变
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 检查特定的计数器预期发生改变
  ExpectCounterChanged("lazy::isnan", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestExpand) {
  // 创建一个随机大小为 3x4 的张量 a
  torch::Tensor a = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 将张量 a 沿指定维度扩展为 2x3x4 的张量 b，不隐式扩展
  torch::Tensor b = a.expand({2, 3, 4}, /*implicit=*/false);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 lazy_a 沿指定维度扩展为 2x3x4 的张量 lazy_b，不隐式扩展
    torch::Tensor lazy_b = lazy_a.expand({2, 3, 4}, /*implicit=*/false);
    // 检查张量 b 和 lazy_b 是否在指定的误差范围内相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestExpandBack) {
  // 创建一个随机大小为 3x1 的张量 a
  torch::Tensor a = torch::rand(
      {3, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 将张量 a 沿指定维度扩展为 3x4 的张量 b，不隐式扩展
  torch::Tensor b = a.expand({3, 4}, /*implicit=*/false);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将 lazy_a 沿指定维度扩展为 3x4 的张量 lazy_b，不隐式扩展
    torch::Tensor lazy_b = lazy_a.expand({3, 4}, /*implicit=*/false);
    // 检查张量 b 和 lazy_b 是否在指定的误差范围内相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestExpandAs) {
  // 创建一个随机大小为 3x4 的张量 a
  torch::Tensor a = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个随机大小为 2x3x4 的张量 b
  torch::Tensor b = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 将张量 a 沿着张量 b 的形状扩展，得到张量 c
  torch::Tensor c = torch::native::expand_as(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 和 b 复制到指定设备上，得到 lazy_a 和 lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将 lazy_a 沿着 lazy_b 的形状扩展，得到 lazy_c
    torch::Tensor lazy_c = torch::native::expand_as(lazy_a, lazy_b);
    // 检查张量 c 和 lazy_c 是否在指定的误差范围内相等
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEye) {
  int n = 5;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个单位矩阵，大小为 n x n，使用默认设备
    torch::Tensor out = torch::eye(
        n, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 创建一个单位矩阵，大小为 n x n，使用指定设备
    torch::Tensor lazy_out =
        torch::eye(n, torch::TensorOptions(torch::kFloat).device(device));
    // 检查两个单位矩阵是否在指定的误差范围内相等
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestEyeWide) {
  int lines = 3;
  int cols = 5;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个单位矩阵，形状为 lines x cols，数据类型为 float，使用默认设备
    torch::Tensor out = torch::eye(
        lines,
        cols,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 创建一个单位矩阵，形状为 lines x cols，数据类型为 float，使用指定的设备
    torch::Tensor lazy_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    // 检查两个矩阵是否在数值上相等
    AllClose(out, lazy_out);
}

TEST_F(LazyOpsTest, TestEyeNarrow) {
  // 设置测试用例中的行数和列数
  int lines = 5;
  int cols = 3;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建单位矩阵，数据类型为浮点数，并指定默认设备
    torch::Tensor out = torch::eye(
        lines,
        cols,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 创建延迟初始化的单位矩阵，数据类型和设备与当前设备匹配
    torch::Tensor lazy_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    // 检验两个矩阵是否全部元素接近
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestBroadcastTensors) {
  // 创建随机张量 a 和 b，形状分别为 {2, 1, 1} 和 {2, 1}，数据类型为浮点数，并指定默认设备
  torch::Tensor a = torch::rand(
      {2, 1, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 广播张量 a 和 b，结果存储在向量 c 中
  std::vector<torch::Tensor> c = torch::broadcast_tensors({a, b});
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 和 b 复制到指定设备，得到延迟初始化的张量 lazy_a 和 lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 对延迟初始化的张量 lazy_a 和 lazy_b 进行广播，结果存储在延迟初始化的向量 lazy_c 中
    std::vector<torch::Tensor> lazy_c =
        torch::broadcast_tensors({lazy_a, lazy_b});
    // 断言 c 和 lazy_c 的大小相等
    ASSERT_EQ(c.size(), lazy_c.size());
    // 遍历每个张量，检验其元素是否全部接近
    for (size_t i = 0; i < c.size(); ++i) {
      AllClose(c[i], lazy_c[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestOneIndex) {
  // 针对每种标量类型执行以下操作
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型创建随机张量 params，形状为 {4, 3, 5, 6, 7}，指定默认设备
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建随机索引张量 indices，形状为 {2, 4, 3}，数据类型为长整型，并指定默认设备
    torch::Tensor indices = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    // 使用索引操作获取结果张量 result
    torch::Tensor result = torch::index(params, {indices});
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将张量 params 和 indices 复制到指定设备，得到延迟初始化的张量 lazy_params 和 lazy_indices
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices = CopyToDevice(indices, device);
      // 使用延迟初始化的张量进行索引操作，得到延迟初始化的结果张量 lazy_result
      torch::Tensor lazy_result = torch::index(lazy_params, {lazy_indices});
      // 断言 result 和 lazy_result 的所有元素是否相等
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestOneIndexTransfer) {
  // 针对每种标量类型执行以下操作
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型创建随机张量 params，形状为 {4, 3, 5, 6, 7}，指定默认设备
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建随机索引张量 indices，形状为 {2, 4, 3}，数据类型为长整型，并指定默认设备
    torch::Tensor indices = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    // 使用索引操作获取结果张量 result
    torch::Tensor result = torch::index(params, {indices});
    # 对每个设备执行以下操作，使用 lambda 表达式捕获当前设备
    ForEachDevice([&](const torch::Device& device) {
        # 将参数复制到指定设备上，返回一个惰性张量 lazy_params
        torch::Tensor lazy_params = CopyToDevice(params, device);
        # 使用 CPU 上的索引 indices.cpu() 对惰性张量 lazy_params 进行索引操作，返回惰性结果 lazy_result
        torch::Tensor lazy_result = torch::index(lazy_params, {indices.cpu()});
        # 检查两个张量 result 和 lazy_result 是否完全相等
        AllEqual(result, lazy_result);
    });
}


这段代码通过注释解释了对每个设备执行操作的过程，包括参数复制、索引操作和结果比较。
TEST_F(LazyOpsTest, TestNonzero) {
  // 创建一个大小为 (4, 2) 的全零张量，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::zeros(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 修改张量中的部分元素值
  a[0][1] = 1.0;
  a[1][0] = 2.0;
  a[3][1] = 3.0;
  // 对张量执行非零元素索引操作，返回非零元素的索引
  torch::Tensor b = torch::nonzero(a);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上执行非零元素索引操作
    torch::Tensor lazy_b = torch::nonzero(lazy_a);
    // 检查两个张量是否在数值上近似相等
    AllClose(b, lazy_b);

    // 如果启用了 "nonzero" 实验特性
    if (DebugUtil::ExperimentEnabled("nonzero")) {
      // 确保未出现任何 aten:: 调用
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    // 重置计数器
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMaskedSelect) {
  // 创建一个大小为 (3, 5) 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 (5,) 的随机布尔张量，放置在默认设备上
  torch::Tensor b = torch::randint(
      0, 2, {5}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  // 根据布尔掩码 b，从张量 a 中选择元素，形成新的张量 c
  torch::Tensor c = torch::masked_select(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 和 b 复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上执行 masked_select 操作
    torch::Tensor lazy_c = torch::masked_select(lazy_a, lazy_b);
    // 检查两个张量是否在数值上近似相等
    AllClose(c, lazy_c);

    // 如果启用了 "masked_select" 实验特性
    if (DebugUtil::ExperimentEnabled("masked_select")) {
      // 确保未出现任何 aten:: 调用
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    // 重置计数器
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMaskedScatter) {
  // 创建一个大小为 (3, 5) 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 (3, 5) 的随机布尔张量，放置在默认设备上
  torch::Tensor b = torch::randint(
      0, 2, {3, 5}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  // 创建一个大小为 (15,) 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor c = torch::rand(
      {15}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 根据布尔掩码 b，将张量 c 的值散布到张量 a 中，形成新的张量 d
  torch::Tensor d = torch::masked_scatter(a, b, c);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a、b 和 c 复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在指定设备上执行 masked_scatter 操作
    torch::Tensor lazy_d = torch::masked_scatter(lazy_a, lazy_b, lazy_c);
    // 检查两个张量是否在数值上近似相等
    AllClose(d, lazy_d);

    // 如果启用了 "masked_scatter" 实验特性
    if (DebugUtil::ExperimentEnabled("masked_scatter")) {
      // 确保未出现任何 aten:: 调用
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    // 重置计数器
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMultiIndexHeadNull) {
  // 遍历不同的标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型创建不同类型的随机张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建一个未初始化的 Tensor 变量 indices_null
    torch::Tensor indices_null;
    
    // 生成一个大小为 {2, 4, 3}，元素值在[-3, 3)范围内的随机整数 Tensor indices_0
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    
    // 生成一个大小为 {2, 4, 3}，元素值在[-3, 3)范围内的随机整数 Tensor indices_1
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    
    // 使用 indices_null, indices_0, indices_1 进行索引操作，得到结果 Tensor result
    torch::Tensor result =
        torch::index(params, {indices_null, indices_0, indices_1});
    
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 params 复制到指定设备上，得到 lazy_params
      torch::Tensor lazy_params = CopyToDevice(params, device);
      
      // 将 indices_0 复制到指定设备上，得到 lazy_indices_0
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      
      // 将 indices_1 复制到指定设备上，得到 lazy_indices_1
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      
      // 使用 lazy_params, indices_null, lazy_indices_0, lazy_indices_1 进行索引操作，得到 lazy_result
      torch::Tensor lazy_result = torch::index(
          lazy_params, {indices_null, lazy_indices_0, lazy_indices_1});
      
      // 检查 result 和 lazy_result 是否全部相等
      AllEqual(result, lazy_result);
    });
TEST_F(LazyOpsTest, TestMultiIndexMiddleNull) {
  // 遍历各种标量类型，包括浮点数和整数类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型生成随机张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},  // 参数形状为 4x3x5x6x7
              torch::TensorOptions(scalar_type).device(DefaultDevice()))  // 张量选项为指定设备
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},  // 参数形状为 4x3x5x6x7
              torch::TensorOptions(scalar_type).device(DefaultDevice()));  // 张量选项为指定设备
    // 生成随机整数张量 indices_0
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},  // 参数形状为 2x4x3
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));  // 张量选项为长整型，指定设备
    // 定义空张量 indices_null
    torch::Tensor indices_null;
    // 生成随机整数张量 indices_1
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 4, 3},  // 参数形状为 2x4x3
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));  // 张量选项为长整型，指定设备
    // 使用 torch::index 函数根据 indices_0, indices_null, indices_1 索引 params，得到结果 result
    torch::Tensor result =
        torch::index(params, {indices_0, indices_null, indices_1});
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 params 复制到指定设备，并命名为 lazy_params
      torch::Tensor lazy_params = CopyToDevice(params, device);
      // 将 indices_0 复制到指定设备，并命名为 lazy_indices_0
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      // 将 indices_1 复制到指定设备，并命名为 lazy_indices_1
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      // 使用 torch::index 函数根据 lazy_params, lazy_indices_0, indices_null, lazy_indices_1 索引，并得到 lazy_result
      torch::Tensor lazy_result = torch::index(
          lazy_params, {lazy_indices_0, indices_null, lazy_indices_1});
      // 检查 result 和 lazy_result 是否全部相等
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexTailNull) {
  // 遍历各种标量类型，包括浮点数和整数类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型生成随机张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},  // 参数形状为 4x3x5x6x7
              torch::TensorOptions(scalar_type).device(DefaultDevice()))  // 张量选项为指定设备
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},  // 参数形状为 4x3x5x6x7
              torch::TensorOptions(scalar_type).device(DefaultDevice()));  // 张量选项为指定设备
    // 生成随机整数张量 indices_0
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},  // 参数形状为 2x4x3
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));  // 张量选项为长整型，指定设备
    // 定义空张量 indices_null
    torch::Tensor indices_null;
    // 生成随机整数张量 indices_1
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 4, 3},  // 参数形状为 2x4x3
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));  // 张量选项为长整型，指定设备
    // 使用 torch::index 函数根据 indices_0, indices_1, indices_null 索引 params，得到结果 result
    torch::Tensor result =
        torch::index(params, {indices_0, indices_1, indices_null});
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 params 复制到指定设备，并命名为 lazy_params
      torch::Tensor lazy_params = CopyToDevice(params, device);
      // 将 indices_0 复制到指定设备，并命名为 lazy_indices_0
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      // 将 indices_1 复制到指定设备，并命名为 lazy_indices_1
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      // 使用 torch::index 函数根据 lazy_params, lazy_indices_0, lazy_indices_1, indices_null 索引，并得到 lazy_result
      torch::Tensor lazy_result = torch::index(
          lazy_params, {lazy_indices_0, lazy_indices_1, indices_null});
      // 检查 result 和 lazy_result 是否全部相等
      AllEqual(result, lazy_result);
    });
  }
}
TEST_F(LazyOpsTest, TestMultiIndexMiddleBroadcast) {
  // 遍历各种标量类型，包括浮点数、字节、字符、短整型、整型和长整型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型随机生成张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 随机生成索引张量 indices_0 和 indices_1
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 1, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    // 使用 torch::index 函数根据 indices_0 和 indices_1 对 params 进行索引操作，得到结果张量 result
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 params、indices_0 和 indices_1 复制到指定设备上
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      // 使用 torch::index 函数在指定设备上对 lazy_params 进行索引操作，得到 lazy_result
      torch::Tensor lazy_result =
          torch::index(lazy_params, {lazy_indices_0, lazy_indices_1});
      // 检查 result 和 lazy_result 是否相等
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexTailBroadcast) {
  // 类似前一个测试，这里进行尾部广播的多重索引测试
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型随机生成张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 随机生成索引张量 indices_0 和 indices_1
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 1, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 1},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    // 使用 torch::index 函数根据 indices_0 和 indices_1 对 params 进行索引操作，得到结果张量 result
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 params、indices_0 和 indices_1 复制到指定设备上
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      // 使用 torch::index 函数在指定设备上对 lazy_params 进行索引操作，得到 lazy_result
      torch::Tensor lazy_result =
          torch::index(lazy_params, {lazy_indices_0, lazy_indices_1});
      // 检查 result 和 lazy_result 是否相等
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMaskIndex) {
  // 类似前两个测试，这里进行掩码索引的测试
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 生成一个随机的张量，如果标量类型是浮点型，则使用 torch::rand() 函数生成，否则使用 torch::randint() 函数生成
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {2, 2}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {2, 2},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 生成一个随机的索引张量，元素值在 0 和 2 之间
    torch::Tensor indices = torch::randint(
        0,
        2,
        {2, 2},
        torch::TensorOptions(torch::kBool).device(DefaultDevice()));
    // 使用索引张量从 params 中获取对应位置的值，生成结果张量
    torch::Tensor result = torch::index(params, {indices});
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 params 复制到指定设备上
      torch::Tensor lazy_params = CopyToDevice(params, device);
      // 将 indices 复制到指定设备上
      torch::Tensor lazy_indices = CopyToDevice(indices, device);
      // 使用 lazy_indices 从 lazy_params 中获取对应位置的值，生成结果张量
      torch::Tensor lazy_result = torch::index(lazy_params, {lazy_indices});
      // 检查 result 和 lazy_result 是否相等
      AllEqual(result, lazy_result);
    });
}

TEST_F(LazyOpsTest, TestOneIndexPut) {
  // 循环遍历不同的数据类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据数据类型生成参数张量，根据浮点类型或整数类型选择不同的生成方式
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 生成索引张量，长整型，形状为 {2, 4, 3}
    torch::Tensor indices = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    // 根据数据类型生成值张量，根据浮点类型或整数类型选择不同的生成方式
    torch::Tensor values = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 循环遍历累积标志，false 和 true
    for (bool accumulate : {false, true}) {
      // 如果累积为 true 并且当前设备是 CUDA 设备，则跳过当前测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      // 使用索引张量对参数张量进行索引更新，根据累积标志决定是否进行累积
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将参数张量、索引张量、值张量复制到指定设备
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 使用索引张量对参数张量在指定设备上进行索引更新，根据累积标志决定是否进行累积
        torch::Tensor lazy_result = torch::index_put(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        // 检查结果张量和延迟执行的结果张量是否相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestOneIndexPutInPlace) {
  // 生成索引张量，长整型，形状为 {2, 4, 3}
  torch::Tensor indices = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 循环遍历不同的数据类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 生成值张量，全为1，根据数据类型选择生成方式
    torch::Tensor values = torch::ones(
        {3, 5, 6, 7},
        torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 对于 accumulate 取 {false, true} 中的每个值，执行循环
    for (bool accumulate : {false, true}) {
      // 如果 accumulate 为 true 并且当前环境支持 CUDA，则跳过当前测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      // 针对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 根据标量类型生成一个随机张量 params，其形状为 {4, 3, 5, 6, 7}
        torch::Tensor params = isFloatingType(scalar_type)
            ? torch::rand(
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        // 将 params 复制到指定设备，并得到 lazy_params
        torch::Tensor lazy_params = CopyToDevice(params.clone(), device);
        // 使用 index_put_ 函数根据给定的 indices 和 values 在 params 上进行操作，根据 accumulate 的值决定是否累加
        torch::Tensor result =
            torch::index_put_(params, {indices}, values, accumulate);
        // 将 indices 和 values 复制到指定设备，得到 lazy_indices 和 lazy_values
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 在 lazy_params 上使用 index_put_ 函数进行相同的操作，根据 accumulate 的值决定是否累加
        torch::Tensor lazy_result = torch::index_put_(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        // 检查 result 和 lazy_result 是否完全相等
        AllEqual(result, lazy_result);
        // 检查 params 和 lazy_params 是否完全相等
        AllEqual(params, lazy_params);
      });
    }
  }


这段代码中涉及了很多 TorchScript 或 PyTorch 的相关操作，包括张量的创建、复制、索引操作以及设备的管理。注释对每一行代码进行了详细解释，包括条件判断、函数调用和操作的目的。
}

TEST_F(LazyOpsTest, TestOneIndexPutTransfer) {
  // 生成一个随机整数张量，用作索引，形状为 [2, 4, 3]
  torch::Tensor indices = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  
  // 遍历不同的标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型生成随机参数张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 创建值为全 1 的张量 values，形状为 [3, 5, 6, 7]
    torch::Tensor values = torch::ones(
        {3, 5, 6, 7},
        torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 遍历是否累积的标志
    for (bool accumulate : {false, true}) {
      // 如果累积并且当前环境为 CUDA，则跳过测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      
      // 使用 torch::index_put 在 params 上进行索引赋值操作，结果为 result
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 params 和 values 复制到指定设备上
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        
        // 在指定设备上使用 torch::index_put 进行索引赋值操作，结果为 lazy_result
        torch::Tensor lazy_result =
            torch::index_put(lazy_params, {indices}, lazy_values, accumulate);
        
        // 检查 result 和 lazy_result 是否全部相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPut) {
  // 生成两个随机整数张量作为索引，形状均为 [2, 4, 3]
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  
  // 遍历不同的标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型生成随机参数张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 创建值为全 1 的张量 values，形状为 [5, 6, 7]
    torch::Tensor values = torch::ones(
        {5, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 遍历布尔值accumulate的两种可能取值：false和true
    for (bool accumulate : {false, true}) {
      // 如果accumulate为true并且当前环境支持CUDA，则跳过测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      // 使用torch库的index_put函数，根据给定的indices_0、indices_1和values更新params张量
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      // 对于每一个设备，执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将params、indices_0、indices_1和values复制到当前设备上
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 使用torch库在当前设备上进行index_put操作，根据lazy_indices_0、lazy_indices_1和lazy_values更新lazy_params张量
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        // 检查result和lazy_result是否相等
        AllEqual(result, lazy_result);
      });
    }
}

TEST_F(LazyOpsTest, TestMultiIndexPutHeadNull) {
  // 创建一个大小为 (2, 4, 3) 的长整型张量，元素值在 [-3, 3] 之间随机选取
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 创建一个空张量 indices_null
  torch::Tensor indices_null;
  // 创建一个大小为 (2, 4, 3) 的长整型张量，元素值在 [-3, 3] 之间随机选取
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 对于每一种标量类型进行迭代
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 如果标量类型是浮点类型，则创建一个大小为 (4, 3, 3, 6, 7) 的随机张量，设备为默认设备
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        // 否则创建一个在 [0, 100) 范围内整型张量，大小为 (4, 3, 3, 6, 7)，设备为默认设备
        : torch::randint(
              100,
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建一个大小为 (3, 6, 7) 的值全为 1 的张量，设备为默认设备
    torch::Tensor values = torch::ones(
        {3, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 对于累加标志位 false 和 true 进行迭代
    for (bool accumulate : {false, true}) {
      // 如果累加为 true 且当前设备是 CUDA 设备，则跳过当前测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      // 使用 torch::index_put 对 params 进行索引更新操作，使用 indices_null, indices_0, indices_1 作为索引
      torch::Tensor result = torch::index_put(
          params, {indices_null, indices_0, indices_1}, values, accumulate);
      // 对于每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 params, indices_0, indices_1, values 复制到指定设备上
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 使用 torch::index_put 在指定设备上对 lazy_params 进行索引更新操作
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {indices_null, lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        // 检查 result 和 lazy_result 在所有设备上的值是否相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutMiddleNull) {
  // 创建一个大小为 (2, 4, 3) 的长整型张量，元素值在 [-3, 3] 之间随机选取
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 创建一个空张量 indices_null
  torch::Tensor indices_null;
  // 创建一个大小为 (2, 4, 3) 的长整型张量，元素值在 [-3, 3] 之间随机选取
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 对于每一种标量类型进行迭代
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 如果标量类型是浮点类型，则创建一个大小为 (4, 3, 3, 6, 7) 的随机张量，设备为默认设备
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        // 否则创建一个在 [0, 100) 范围内整型张量，大小为 (4, 3, 3, 6, 7)，设备为默认设备
        : torch::randint(
              100,
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建一个大小为 (3, 6, 7) 的值全为 1 的张量，设备为默认设备
    torch::Tensor values = torch::ones(
        {3, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 对于每个布尔值累积，分别为false和true进行循环迭代
    for (bool accumulate : {false, true}) {
      // 如果当前为累积模式并且CUDA可用，则跳过当前测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      // 使用torch::index_put函数对params张量进行索引操作，并返回结果张量result
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_null, indices_1}, values, accumulate);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将params、indices_0、indices_1和values复制到当前设备上，并创建对应的张量
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 使用torch::index_put函数在当前设备上进行索引操作，并返回结果张量lazy_result
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, indices_null, lazy_indices_1},
            lazy_values,
            accumulate);
        // 检查result和lazy_result张量是否在所有设备上相等
        AllEqual(result, lazy_result);
      });
    }
// 定义一个测试函数 TestMultiIndexPutTailNull，使用 Google Test 框架
TEST_F(LazyOpsTest, TestMultiIndexPutTailNull) {
  // 生成一个大小为 [2, 4, 3] 的长整型随机张量 indices_0
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 生成一个大小为 [2, 4, 3] 的长整型随机张量 indices_1
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 创建一个空张量 indices_null
  torch::Tensor indices_null;
  // 遍历标量类型列表，包括 kFloat、kByte、kChar、kShort、kInt、kLong
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型判断是否为浮点类型，创建张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建全为 1 的张量 values
    torch::Tensor values = torch::ones(
        {3, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 遍历布尔值 accumulate，包括 false 和 true
    for (bool accumulate : {false, true}) {
      // 如果 accumulate 为 true 且在 CUDA 设备上，跳过当前测试用例
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      // 使用 torch::index_put 对 params 进行索引赋值操作，结果为 result
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_1, indices_null}, values, accumulate);
      // 对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 params、indices_0、indices_1、values 复制到指定设备上
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 使用 torch::index_put 在指定设备上进行索引赋值操作，结果为 lazy_result
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1, indices_null},
            lazy_values,
            accumulate);
        // 断言 result 与 lazy_result 在所有设备上相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

// 定义一个测试函数 TestMultiIndexPutMiddleBroadcast
TEST_F(LazyOpsTest, TestMultiIndexPutMiddleBroadcast) {
  // 生成一个大小为 [2, 4, 3] 的长整型随机张量 indices_0
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 生成一个大小为 [2, 1, 3] 的长整型随机张量 indices_1
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 1, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 遍历标量类型列表，包括 kFloat、kByte、kChar、kShort、kInt、kLong
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型判断是否为浮点类型，创建张量 params
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建全为 1 的张量 values
    torch::Tensor values = torch::ones(
        {5, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 遍历布尔值序列 {false, true}
    for (bool accumulate : {false, true}) {
        // 如果 accumulate 为 true 并且当前环境支持 CUDA，跳过当前测试
        if (accumulate && IsCuda()) {
            GTEST_SKIP();
        }
        // 使用 torch 的 index_put 函数更新张量 params 中的数据，根据给定的 indices_0、indices_1 和 values
        torch::Tensor result =
            torch::index_put(params, {indices_0, indices_1}, values, accumulate);
        
        // 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
            // 将 params、indices_0、indices_1 和 values 复制到指定设备上
            torch::Tensor lazy_params = CopyToDevice(params, device);
            torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
            torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
            torch::Tensor lazy_values = CopyToDevice(values, device);
            // 使用 torch 的 index_put 函数在指定设备上更新 lazy_params 的数据
            torch::Tensor lazy_result = torch::index_put(
                lazy_params,
                {lazy_indices_0, lazy_indices_1},
                lazy_values,
                accumulate);
            // 检查 result 和 lazy_result 是否在所有设备上相等
            AllEqual(result, lazy_result);
        });
    }
}

// 定义测试用例 `TestMultiIndexPutTailBroadcast`
TEST_F(LazyOpsTest, TestMultiIndexPutTailBroadcast) {
  // 生成随机整数张量 `indices_0`，形状为 (2, 1, 3)，值范围为 -3 到 3
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 1, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  
  // 生成随机整数张量 `indices_1`，形状为 (2, 1)，值范围为 -3 到 3
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 1},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  
  // 遍历多种标量类型进行测试
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    
    // 根据标量类型生成参数张量 `params`
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 生成值为全 1 的张量 `values`，与参数张量 `params` 的数据类型和设备保持一致
    torch::Tensor values = torch::ones(
        {5, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 遍历布尔值 `accumulate` 进行测试
    for (bool accumulate : {false, true}) {
      // 如果 accumulate 为真且当前设备为 CUDA，则跳过测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      
      // 执行索引赋值操作，得到结果张量 `result`
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      
      // 在每个设备上执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将参数张量 `params` 复制到指定设备，得到 `lazy_params`
        torch::Tensor lazy_params = CopyToDevice(params, device);
        
        // 将索引张量 `indices_0` 复制到指定设备，得到 `lazy_indices_0`
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        
        // 将索引张量 `indices_1` 复制到指定设备，得到 `lazy_indices_1`
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        
        // 将值张量 `values` 复制到指定设备，得到 `lazy_values`
        torch::Tensor lazy_values = CopyToDevice(values, device);
        
        // 在指定设备上执行索引赋值操作，得到 `lazy_result`
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        
        // 断言 `result` 和 `lazy_result` 在结果上是否全部相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

// 定义测试用例 `TestMaskIndexPut`
TEST_F(LazyOpsTest, TestMaskIndexPut) {
  // 创建索引张量 `indices`，包含元素 {0, 1}，数据类型为 kByte，并转换为 kBool 类型
  torch::Tensor indices =
      torch::tensor(
          {0, 1}, torch::TensorOptions(torch::kByte).device(DefaultDevice()))
          .to(torch::kBool);
  
  // 遍历多种标量类型进行测试
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    
    // 根据标量类型生成参数张量 `params`
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {2, 2},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {2, 2},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 生成值为全 1 的张量 `values`，与参数张量 `params` 的数据类型和设备保持一致
    torch::Tensor values = torch::ones(
        {2}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 遍历布尔值数组 {false, true}，分别设定 accumulate 参数为 false 和 true
    for (bool accumulate : {false, true}) {
      // 使用 torch 库的 index_put 函数，根据给定的 indices 和 values 修改 params 张量
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 params 张量复制到指定设备上
        torch::Tensor lazy_params = CopyToDevice(params, device);
        // 将 indices 张量复制到指定设备上
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        // 将 values 张量复制到指定设备上
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 在指定设备上使用 index_put 函数，根据 lazy_indices 和 lazy_values 修改 lazy_params 张量
        torch::Tensor lazy_result = torch::index_put(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        // 检查在不同设备上计算的结果是否相等
        AllEqual(result, lazy_result);
      });
    }
}

TEST_F(LazyOpsTest, TestIndexPutImpl) {
  // 生成一个形状为 {2, 4, 3} 的长整型随机张量 indices
  torch::Tensor indices = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 遍历不同的标量类型：浮点数、字节、字符、短整型、整型、长整型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据当前标量类型创建形状为 {3, 5, 6, 7} 的张量 values
    torch::Tensor values = torch::ones(
        {3, 5, 6, 7},
        torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 遍历是否累积操作为 false 或 true
    for (bool accumulate : {false, true}) {
      // 如果累积为 true 且处于 CUDA 设备上，则跳过测试
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      // 针对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 根据标量类型是浮点型时，创建形状为 {4, 3, 5, 6, 7} 的随机张量 params
        torch::Tensor params = isFloatingType(scalar_type)
            ? torch::rand(
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            // 如果不是浮点型，创建形状为 {4, 3, 5, 6, 7} 的长整型随机张量 params
            : torch::randint(
                  100,
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        // 将 params 克隆到当前设备上，并命名为 lazy_params
        torch::Tensor lazy_params = CopyToDevice(params.clone(), device);
        // 调用 _index_put_impl_ 函数进行索引放置操作
        torch::Tensor result = torch::_index_put_impl_(
            params, {indices}, values, accumulate, /*unsafe=*/true);
        // 将 indices 复制到当前设备上，并命名为 lazy_indices
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        // 将 values 复制到当前设备上，并命名为 lazy_values
        torch::Tensor lazy_values = CopyToDevice(values, device);
        // 在 lazy_params 上执行懒惰版本的索引放置操作
        torch::Tensor lazy_result = torch::_index_put_impl_(
            lazy_params,
            {lazy_indices},
            lazy_values,
            accumulate,
            /*unsafe=*/true);
        // 断言结果 result 和 lazy_result 相等
        AllEqual(result, lazy_result);
        // 断言 params 和 lazy_params 相等
        AllEqual(params, lazy_params);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithScalar) {
  // 创建形状为 {2} 的长整型张量 index，其值为 {0, 2}
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 创建标量值为 42
  torch::Scalar value = 42;
  // 遍历不同的标量类型：浮点数、字节、字符、短整型、整型、长整型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型是否为浮点数创建基础张量 base
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        // 如果不是浮点数，创建形状为 {3, 4, 5} 的长整型随机张量 base
        : torch::randint(
              100,
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 计算基础张量 base 的维度 rank
    int rank = base.dim();
    // 遍历 base 的每一个维度
    for (int dim = -rank; dim < rank; ++dim) {
      // 调用 torch::index_fill 函数在指定维度 dim 上以 index 和 value 进行填充
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      // 针对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 base 复制到当前设备上，并命名为 lazy_base
        torch::Tensor lazy_base = CopyToDevice(base, device);
        // 将 index 复制到当前设备上，并命名为 lazy_index
        torch::Tensor lazy_index = CopyToDevice(index, device);
        // 在 lazy_base 上执行懒惰版本的 index_fill 操作
        torch::Tensor lazy_result =
            torch::index_fill(lazy_base, dim, lazy_index, value);
        // 断言结果 result 和 lazy_result 相等
        AllEqual(result, lazy_result);
      });
    }
  }
}
#`
TEST_F(LazyOpsTest, TestIndexFillWithScalarInPlace) {
  // 创建一个索引张量，包含元素 0 和 2，数据类型为长整型，并指定设备
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 定义一个标量值为 42
  torch::Scalar value = 42;
  // 设置张量的维度为 3
  int rank = 3;
  // 遍历数据类型列表，包括浮点类型和整型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 遍历所有维度，范围从 -rank 到 rank
    for (int dim = -rank; dim < rank; ++dim) {
      // 在每个设备上执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 根据数据类型生成一个随机张量 base
        torch::Tensor base = isFloatingType(scalar_type)
            ? torch::rand(
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        // 将 base 张量复制到指定设备
        torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
        // 使用 index 和 value 执行 index_fill 操作，并返回结果张量
        torch::Tensor result = base.index_fill_(dim, index, value);
        // 将索引张量复制到指定设备
        torch::Tensor lazy_index = CopyToDevice(index, device);
        // 使用索引张量和 value 执行 index_fill 操作（延迟计算），返回结果张量
        torch::Tensor lazy_result =
            lazy_base.index_fill_(dim, lazy_index, value);
        // 比较两个张量是否相等
        AllEqual(result, lazy_result);
        // 比较原始张量和延迟计算张量是否相等
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithTensor) {
  // 创建一个索引张量，包含元素 0 和 2，数据类型为长整型，并指定设备
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 遍历数据类型列表，包括浮点类型和整型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据数据类型生成一个随机张量 base
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 创建一个张量 value，值为标量 42，数据类型与 base 相同，指定设备
    torch::Tensor value = torch::scalar_tensor(
        42, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 获取 base 张量的维度数
    int rank = base.dim();
    // 遍历所有维度，范围从 -rank 到 rank
    for (int dim = -rank; dim < rank; ++dim) {
      // 使用 index_fill 操作计算结果张量
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      // 在每个设备上执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 base 张量复制到指定设备
        torch::Tensor lazy_base = CopyToDevice(base, device);
        // 将索引张量复制到指定设备
        torch::Tensor lazy_index = CopyToDevice(index, device);
        // 将 value 张量复制到指定设备
        torch::Tensor lazy_value = CopyToDevice(value, device);
        // 使用 index_fill 操作（延迟计算），返回结果张量
        torch::Tensor lazy_result =
            torch::index_fill(lazy_base, dim, lazy_index, lazy_value);
        // 比较结果张量是否相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithTensorInPlace) {
  // 创建一个索引张量，包含元素 0 和 2，数据类型为长整型，并指定设备
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 遍历数据类型列表，包括浮点类型和整型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 创建一个标量值为42的张量，使用给定的标量类型和默认设备
    torch::Tensor value = torch::scalar_tensor(
        42, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 设置张量的维度等级为3
    int rank = 3;
    // 对每一个维度进行循环操作，从负等级到正等级
    for (int dim = -rank; dim < rank; ++dim) {
      // 针对每一个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 如果标量类型是浮点类型，则创建一个形状为{3, 4, 5}的随机张量，放置在默认设备上
        torch::Tensor base = isFloatingType(scalar_type)
            ? torch::rand(
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            // 如果标量类型是整数类型，则创建一个在[0, 100)范围内的随机整数张量，放置在默认设备上
            : torch::randint(
                  100,
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        // 将base张量克隆并复制到指定设备上，得到lazy_base张量
        torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
        // 在指定维度dim上使用value值填充base张量，得到result张量
        torch::Tensor result = base.index_fill_(dim, index, value);
        // 将索引index复制到指定设备上，得到lazy_index张量
        torch::Tensor lazy_index = CopyToDevice(index, device);
        // 将value值复制到指定设备上，得到lazy_value张量
        torch::Tensor lazy_value = CopyToDevice(value, device);
        // 在指定维度dim上使用lazy_value值填充lazy_base张量，得到lazy_result张量
        torch::Tensor lazy_result =
            lazy_base.index_fill_(dim, lazy_index, lazy_value);
        // 检查result张量与lazy_result张量是否相等
        AllEqual(result, lazy_result);
        // 检查base张量与lazy_base张量是否相等
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillRank0) {
  // 创建一个标量张量 index，其值为 2，数据类型为长整型，放置在默认设备上
  torch::Tensor index = torch::scalar_tensor(
      2, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  
  // 遍历不同的标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型判断 base 张量的创建方式：如果是浮点类型，则随机初始化；否则随机整数初始化
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 创建一个标量张量 value，其值为 42，数据类型与 base 相同，放置在默认设备上
    torch::Tensor value = torch::scalar_tensor(
        42, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 获取 base 张量的维度
    int rank = base.dim();
    
    // 遍历 base 张量的所有维度
    for (int dim = -rank; dim < rank; ++dim) {
      // 使用 index 和 value 对 base 张量进行索引填充操作，生成结果张量 result
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 base、index、value 复制到指定设备上
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        
        // 在指定设备上使用复制后的张量进行索引填充操作，生成 lazy_result
        torch::Tensor lazy_result =
            torch::index_fill(lazy_base, dim, lazy_index, lazy_value);
        
        // 验证结果张量 result 和 lazy_result 在所有设备上的数据是否相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexAdd) {
  // 索引的大小
  int index_size = 10;
  
  // 遍历不同的标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型判断 base 张量的创建方式：如果是浮点类型，则随机初始化；否则随机整数初始化
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    
    // 获取 base 张量的维度
    int rank = base.dim();
    
    // 外层循环遍历维度范围从负数到正数（不包括正数）
    for (int dim = -rank; dim < rank; ++dim) {
      // 内层循环遍历索引标量类型，包括 torch::kInt 和 torch::kLong
      for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
        // 生成一个随机整数张量 index，范围在 [0, base.size(dim))，大小为 index_size
        torch::Tensor index = torch::randint(
            0,
            base.size(dim),
            {index_size},
            torch::TensorOptions(index_scalar_type).device(DefaultDevice()));
        // 创建一个向量 value_sizes，从 base 的 sizes 中获取，作为 value 的尺寸
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        // 计算规范化的维度 canonical_dim，如果 dim 为负数则加上 rank
        int canonical_dim = dim < 0 ? dim + rank : dim;
        // 设置 value_sizes 中 canonical_dim 维度的大小为 index_size
        value_sizes[canonical_dim] = index_size;
        // 生成一个随机数或整数张量 value，类型由 scalar_type 决定
        torch::Tensor value = isFloatingType(scalar_type)
            ? torch::rand(
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        // 使用 index 和 value 对 base 在 dim 维度上进行索引加法操作，结果保存在 result 中
        torch::Tensor result = torch::index_add(base, dim, index, value);
        // 对每个设备执行以下操作
        ForEachDevice([&](const torch::Device& device) {
          // 将 base, index, value 拷贝到指定设备上
          torch::Tensor lazy_base = CopyToDevice(base, device);
          torch::Tensor lazy_index = CopyToDevice(index, device);
          torch::Tensor lazy_value = CopyToDevice(value, device);
          // 在指定设备上对 lazy_base 进行索引加法操作，结果保存在 lazy_result 中
          torch::Tensor lazy_result =
              torch::index_add(lazy_base, dim, lazy_index, lazy_value);
          // 检查结果 lazy_result 是否与之前的 result 在所有设备上保持一致
          AllClose(result, lazy_result);
        });
      }
    }
// 定义一个测试用例 LazyOpsTest.TestIndexAddInPlace，测试在指定维度上原地索引加法操作
TEST_F(LazyOpsTest, TestIndexAddInPlace) {
  // 设置索引大小和张量的维度数
  int index_size = 10;
  int rank = 3;
  // 遍历所有的标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 在每个维度上进行循环操作
    for (int dim = -rank; dim < rank; ++dim) {
      // 对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 根据标量类型生成基础张量 base
        torch::Tensor base = isFloatingType(scalar_type)
            ? torch::rand(
                  {5, 3, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {5, 3, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        // 在指定维度上生成随机索引张量 index
        torch::Tensor index = torch::randint(
            0,
            base.size(dim),
            {index_size},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        // 创建 value_sizes，包含了基础张量的尺寸
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        // 计算规范化的维度索引
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        // 根据标量类型生成值张量 value
        torch::Tensor value = isFloatingType(scalar_type)
            ? torch::rand(
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        // 将 base 克隆到指定设备上，并生成 lazy_base
        torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
        // 在 base 上进行原地索引加法操作，返回结果 result
        torch::Tensor result = base.index_add_(dim, index, value);
        // 将 index 和 value 复制到指定设备，生成 lazy_index 和 lazy_value
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        // 在 lazy_base 上进行原地索引加法操作，返回 lazy_result
        torch::Tensor lazy_result =
            lazy_base.index_add_(dim, lazy_index, lazy_value);
        // 检查 result 和 lazy_result 是否近似相等
        AllClose(result, lazy_result);
        // 检查 base 和 lazy_base 是否近似相等
        AllClose(base, lazy_base);
      });
    }
  }
}

// 定义一个测试用例 LazyOpsTest.TestIndexAddRank0，测试在零维度上索引加法操作
TEST_F(LazyOpsTest, TestIndexAddRank0) {
  // 遍历所有的标量类型
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型生成基础张量 base
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 获取 base 的维度 rank
    int rank = base.dim();
    // 对于每一个维度 dim，从 -rank 到 rank-1 进行循环
    for (int dim = -rank; dim < rank; ++dim) {
      // 在设备上生成一个随机整数张量 index，其形状与 base 在 dim 维度上的大小相同
      torch::Tensor index = torch::randint(
          0,
          base.size(dim),
          at::IntArrayRef{},
          torch::TensorOptions(torch::kLong).device(DefaultDevice()));
      // 创建一个整数向量 value_sizes，包含 base 张量的所有维度大小
      std::vector<int64_t> value_sizes(
          base.sizes().begin(), base.sizes().end());
      // 计算规范化后的维度 canonical_dim，如果 dim 小于 0 则加上 rank
      int canonical_dim = dim < 0 ? dim + rank : dim;
      // 将 value_sizes 中 canonical_dim 对应的大小设为 1
      value_sizes[canonical_dim] = 1;
      // 根据标量类型生成一个随机张量 value，如果是浮点类型则使用 torch::rand，否则使用 torch::randint
      torch::Tensor value = isFloatingType(scalar_type)
          ? torch::rand(
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()))
          : torch::randint(
                100,
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()));
      // 使用 index 和 value 对 base 张量进行索引加法操作，dim 维度上进行操作
      torch::Tensor result = torch::index_add(base, dim, index, value);
      // 对于每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 base、index、value 分别复制到指定设备上
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        // 在指定设备上执行索引加法操作，并得到 lazy_result
        torch::Tensor lazy_result =
            torch::index_add(lazy_base, dim, lazy_index, lazy_value);
        // 检查 result 和 lazy_result 在指定设备上是否完全相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

# 在 LazyOpsTest 测试类中定义 TestIndexCopy 方法
TEST_F(LazyOpsTest, TestIndexCopy) {
  # 对于每种标量类型进行循环测试
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    # 如果标量类型是浮点型，则创建随机张量 base
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        # 如果标量类型不是浮点型，则创建随机整数张量 base
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    # 获取张量 base 的维度
    int rank = base.dim();
    # 对张量 base 的每个维度进行循环
    for (int dim = -rank; dim < rank; ++dim) {
      # 创建随机排列索引张量 index
      torch::Tensor index = torch::randperm(
          base.size(dim),
          torch::TensorOptions(torch::kLong).device(DefaultDevice()));
      # 根据标量类型，创建随机值张量 value
      torch::Tensor value = isFloatingType(scalar_type)
          ? torch::rand(
                base.sizes(),
                torch::TensorOptions(scalar_type).device(DefaultDevice()))
          : torch::randint(
                100,
                base.sizes(),
                torch::TensorOptions(scalar_type).device(DefaultDevice()));
      # 使用 index_copy 函数在指定维度 dim 上替换 base 的索引值为 value，得到结果张量 result
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      # 对每个设备进行操作，将 base、index、value 复制到设备上，并进行 index_copy 操作，检查结果是否一致
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            torch::index_copy(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

# 在 LazyOpsTest 测试类中定义 TestIndexCopyInPlace 方法
TEST_F(LazyOpsTest, TestIndexCopyInPlace) {
  # 如果在 CUDA 上运行测试，则跳过该测试
  if (IsCuda()) {
    GTEST_SKIP();
  }
  # 设置索引大小和张量的秩
  int index_size = 10;
  int rank = 3;
  # 对于每种标量类型进行循环测试
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
        // 遍历维度范围从负rank到rank之间的每一个维度
        ForEachDevice([&](const torch::Device& device) {
            // 对每个设备执行以下操作
            torch::Tensor base = isFloatingType(scalar_type)
                ? torch::rand(
                      {5, 3, 7},
                      torch::TensorOptions(scalar_type).device(DefaultDevice()))
                : torch::randint(
                      100,
                      {5, 3, 7},
                      torch::TensorOptions(scalar_type).device(DefaultDevice()));
            // 根据数据类型生成随机张量 base，根据设备选项放置在默认设备上
    
            torch::Tensor index = torch::randint(
                0,
                base.size(dim),
                {index_size},
                torch::TensorOptions(torch::kLong).device(DefaultDevice()));
            // 生成一个随机索引张量 index，索引范围在 base 的 dim 维度上，大小为 index_size
    
            std::vector<int64_t> value_sizes(
                base.sizes().begin(), base.sizes().end());
            int canonical_dim = dim < 0 ? dim + rank : dim;
            value_sizes[canonical_dim] = index_size;
            // 创建 value_sizes 向量，从 base 的大小中获取，并根据 canonical_dim 调整索引大小
    
            torch::Tensor value = isFloatingType(scalar_type)
                ? torch::rand(
                      value_sizes,
                      torch::TensorOptions(scalar_type).device(DefaultDevice()))
                : torch::randint(
                      100,
                      value_sizes,
                      torch::TensorOptions(scalar_type).device(DefaultDevice()));
            // 根据数据类型生成随机张量 value，根据设备选项放置在默认设备上
    
            torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
            // 复制 base 到指定设备，产生 lazy_base
    
            torch::Tensor result = base.index_copy_(dim, index, value);
            // 在指定维度 dim 上，用 value 张量的值替换 base 的索引为 index 的值，结果保存在 result 中
    
            torch::Tensor lazy_index = CopyToDevice(index, device);
            // 复制 index 张量到指定设备，产生 lazy_index
    
            torch::Tensor lazy_value = CopyToDevice(value, device);
            // 复制 value 张量到指定设备，产生 lazy_value
    
            torch::Tensor lazy_result =
                lazy_base.index_copy_(dim, lazy_index, lazy_value);
            // 在指定维度 dim 上，用 lazy_value 张量的值替换 lazy_base 的索引为 lazy_index 的值，结果保存在 lazy_result 中
    
            AllEqual(result, lazy_result);
            // 检查 result 和 lazy_result 是否相等
    
            AllEqual(base, lazy_base);
            // 检查 base 和 lazy_base 是否相等
        });
    }
TEST_F(LazyOpsTest, TestIndexCopyRank0) {
  // 针对每种标量类型进行测试
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 如果是浮点类型，则生成随机张量 base
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        // 如果不是浮点类型，则生成随机整数张量 base
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    // 获取张量 base 的秩
    int rank = base.dim();
    // 遍历张量 base 的每一个维度
    for (int dim = -rank; dim < rank; ++dim) {
      // 生成一个随机索引张量 index
      torch::Tensor index = torch::randint(
          0,
          base.size(dim),
          at::IntArrayRef{},
          torch::TensorOptions(torch::kLong).device(DefaultDevice()));
      // 复制 base 的大小信息，并修改指定维度 canonical_dim 的大小为 1
      std::vector<int64_t> value_sizes(
          base.sizes().begin(), base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      // 生成一个与 base 类型相同的随机值张量 value
      torch::Tensor value = isFloatingType(scalar_type)
          ? torch::rand(
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()))
          : torch::randint(
                100,
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()));
      // 使用 index 和 value 对 base 进行索引复制，得到结果张量 result
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      // 对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 base、index、value 复制到指定设备，并执行 lazy 操作
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        // 在指定设备上进行索引复制操作，并得到 lazy_result
        torch::Tensor lazy_result =
            torch::index_copy(lazy_base, dim, lazy_index, lazy_value);
        // 检查 result 和 lazy_result 是否相等
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestRelu) {
  // 生成一个随机输入张量 input
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 执行 relu 操作，并得到输出张量 output
  torch::Tensor output = torch::relu(input);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备，并执行 lazy relu 操作
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::relu(lazy_input);
    // 检查输出张量 output 和 lazy_output 是否接近相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReluInPlace) {
  // 生成一个随机输入张量 input
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备，并执行 lazy in-place relu 操作
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地对 input 执行 relu 操作，并得到输出张量 output
    torch::Tensor output = torch::relu_(input);
    // 在指定设备上进行 lazy in-place relu 操作，并得到 lazy_output
    torch::Tensor lazy_output = torch::relu_(lazy_input);
    // 检查输出张量 output 和 lazy_output 是否接近相等
    AllClose(output, lazy_output);
    // 检查 input 和 lazy_input 是否接近相等
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestHardshrink) {
  // 生成一个随机输入张量 input
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 执行 hardshrink 操作，并得到输出张量 output
  torch::Tensor output = torch::hardshrink(input);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备，并执行 lazy hardshrink 操作
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行 lazy hardshrink 操作，并得到 lazy_output
    torch::Tensor lazy_output = torch::hardshrink(lazy_input);
    # 创建一个 Tensor 对象 `lazy_output`，通过对输入的 Tensor `lazy_input` 进行硬阈值收缩操作
    torch::Tensor lazy_output = torch::hardshrink(lazy_input);
    
    # 检查 `output` 和 `lazy_output` 是否在数值上接近（即误差非常小）
    AllClose(output, lazy_output);
TEST_F(LazyOpsTest, TestHardSigmoid) {
  // 生成一个形状为 {10} 的随机张量，使用默认设备，并设置为浮点类型
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 应用硬 sigmoid 函数，生成输出张量
  torch::Tensor output = torch::hardsigmoid(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，生成 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上应用硬 sigmoid 函数，生成 lazy_output 张量
    torch::Tensor lazy_output = torch::hardsigmoid(lazy_input);
    // 检查输出和 lazy 输出张量是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoidInPlace) {
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 生成形状为 {10} 的随机张量 input，并将其复制到指定设备上生成 lazy_input
    torch::Tensor input = torch::randn(
        {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地应用硬 sigmoid 函数，修改 input，并返回输出张量
    torch::Tensor output = torch::hardsigmoid_(input);
    // 在指定设备上原地应用硬 sigmoid 函数，修改 lazy_input，并返回 lazy_output
    torch::Tensor lazy_output = torch::hardsigmoid_(lazy_input);
    // 检查原始 input 和 lazy_input 是否近似相等
    AllClose(input, lazy_input);
    // 检查输出和 lazy 输出张量是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoidBackward) {
  // 定义一个测试函数，接收张量数组 inputs 并应用硬 sigmoid 函数返回张量
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardsigmoid(inputs[0]);
  };
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 执行反向传播测试，传入形状为 {10} 的随机张量，设置为需要梯度计算
    TestBackward(
        {torch::randn(
            {10},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestSoftshrink) {
  // 生成形状为 {10} 的随机张量 input，使用默认设备，并设置为浮点类型
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 应用 softshrink 函数，生成输出张量
  torch::Tensor output = torch::softshrink(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，生成 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上应用 softshrink 函数，生成 lazy_output 张量
    torch::Tensor lazy_output = torch::softshrink(lazy_input);
    // 检查输出和 lazy 输出张量是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardtanh) {
  // 生成形状为 {10} 的随机张量 input，使用默认设备，并设置为浮点类型
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 应用 hardtanh 函数，生成输出张量
  torch::Tensor output = torch::hardtanh(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，生成 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上应用 hardtanh 函数，生成 lazy_output 张量
    torch::Tensor lazy_output = torch::hardtanh(lazy_input);
    // 检查输出和 lazy 输出张量是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardtanhInPlace) {
  // 生成形状为 {10} 的随机张量 input，使用默认设备，并设置为浮点类型
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，生成 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地应用 hardtanh 函数，修改 input，并返回输出张量
    torch::Tensor output = torch::hardtanh_(input);
    // 在指定设备上原地应用 hardtanh 函数，修改 lazy_input，并返回 lazy_output
    torch::Tensor lazy_output = torch::hardtanh_(lazy_input);
    // 检查输出和 lazy 输出张量是否近似相等
    AllClose(output, lazy_output);
    // 检查原始 input 和 lazy_input 是否近似相等
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestLeakyRelu) {
  // 生成形状为 {2, 1, 4, 6} 的随机张量 input，使用默认设备，并设置为浮点类型
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义负斜率为 0.01 的 leaky relu 函数，生成输出张量
  double negative_slope = 0.01;
  torch::Tensor output = torch::leaky_relu(input, negative_slope);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，生成 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上应用 leaky relu 函数，生成 lazy_output 张量
    torch::Tensor lazy_output = torch::leaky_relu(lazy_input, negative_slope);
    // 检查输出和 lazy 输出张量是否近似相等
    AllClose(output, lazy_output);
  });
}
    // 将输入数据复制到指定设备上，并返回在设备上的张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对复制到设备上的输入数据应用带泄漏线性整流函数（Leaky ReLU），使用给定的负斜率
    torch::Tensor lazy_output = torch::leaky_relu(lazy_input, negative_slope);
    // 检查输出张量和经过泄漏线性整流后的张量是否全部接近
    AllClose(output, lazy_output);
  });
TEST_F(LazyOpsTest, TestLeakyReluInPlace) {
  // 创建一个形状为 [2, 1, 4, 6] 的随机张量 input，使用默认设备上的浮点数选项
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置 Leaky ReLU 的负斜率为 0.01
  double negative_slope = 0.01;
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备，得到 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地对 input 应用 Leaky ReLU，并将结果保存到 output
    torch::Tensor output = torch::leaky_relu_(input, negative_slope);
    // 在原地对 lazy_input 应用 Leaky ReLU，并将结果保存到 lazy_output
    torch::Tensor lazy_output = torch::leaky_relu_(lazy_input, negative_slope);
    // 检查 output 和 lazy_output 是否全部近似相等
    AllClose(output, lazy_output);
    // 检查 input 和 lazy_input 是否全部近似相等
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestExp) {
  // 创建一个形状为 [2, 2] 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 b = exp(a)
  torch::Tensor b = torch::exp(a);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a 复制到指定设备，得到 lazy_a 张量
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_b = exp(lazy_a)
    torch::Tensor lazy_b = torch::exp(lazy_a);
    // 检查 b 和 lazy_b 是否全部近似相等，相对误差限制为 1e-3，绝对误差限制为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestExpm1) {
  // 创建一个形状为 [2, 2] 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 b = expm1(a)
  torch::Tensor b = torch::expm1(a);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a 复制到指定设备，得到 lazy_a 张量
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_b = expm1(lazy_a)
    torch::Tensor lazy_b = torch::expm1(lazy_a);
    // 检查 b 和 lazy_b 是否全部近似相等，相对误差限制为 1e-3，绝对误差限制为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog) {
  // 创建一个形状为 [2, 2] 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 b = log(a)
  torch::Tensor b = torch::log(a);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a 复制到指定设备，得到 lazy_a 张量
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_b = log(lazy_a)
    torch::Tensor lazy_b = torch::log(lazy_a);
    // 检查 b 和 lazy_b 是否全部近似相等，相对误差限制为 1e-3，绝对误差限制为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog2) {
  // 创建一个形状为 [2, 2] 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 b = log2(a)
  torch::Tensor b = torch::log2(a);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a 复制到指定设备，得到 lazy_a 张量
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_b = log2(lazy_a)
    torch::Tensor lazy_b = torch::log2(lazy_a);
    // 检查 b 和 lazy_b 是否全部近似相等，相对误差限制为 1e-3，绝对误差限制为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog10) {
  // 创建一个形状为 [2, 2] 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 b = log10(a)
  torch::Tensor b = torch::log10(a);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a 复制到指定设备，得到 lazy_a 张量
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_b = log10(lazy_a)
    torch::Tensor lazy_b = torch::log10(lazy_a);
    // 检查 b 和 lazy_b 是否全部近似相等，相对误差限制为 1e-3，绝对误差限制为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog1p) {
  // 创建一个形状为 [2, 2] 的随机张量 a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量 b = log1p(a)
  torch::Tensor b = torch::log1p(a);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a 复制到指定设备，得到 lazy_a 张量
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 计算 lazy_b = log1p(lazy_a)
    torch::Tensor lazy_b = torch::log1p(lazy_a);
    // 检查 b 和 lazy_b 是否全部近似相等，相对误差限制为 1e-3，绝对误差限制为 1e-5
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}
TEST_F(LazyOpsTest, TestErf) {
  // 生成一个随机的2x2张量a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a的误差函数，并将结果保存在张量b中
  torch::Tensor b = torch::erf(a);
  // 遍历每个设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，生成lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在当前设备上计算lazy_a的误差函数，结果保存在lazy_b中
    torch::Tensor lazy_b = torch::erf(lazy_a);
    // 检查张量b和lazy_b是否在指定的相对和绝对误差范围内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErfc) {
  // 生成一个随机的2x2张量a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a的余补误差函数，并将结果保存在张量b中
  torch::Tensor b = torch::erfc(a);
  // 遍历每个设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，生成lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在当前设备上计算lazy_a的余补误差函数，结果保存在lazy_b中
    torch::Tensor lazy_b = torch::erfc(lazy_a);
    // 检查张量b和lazy_b是否在指定的相对和绝对误差范围内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErfinv) {
  // 生成一个随机的2x2张量a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a的反误差函数，并将结果保存在张量b中
  torch::Tensor b = torch::erfinv(a);
  // 遍历每个设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，生成lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在当前设备上计算lazy_a的反误差函数，结果保存在lazy_b中
    torch::Tensor lazy_b = torch::erfinv(lazy_a);
    // 检查张量b和lazy_b是否在指定的相对和绝对误差范围内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSqrt) {
  // 生成一个随机的2x2张量a，使用默认设备上的浮点数选项，并取其绝对值后开平方
  torch::Tensor a = torch::abs(torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  // 计算张量a的平方根，并将结果保存在张量b中
  torch::Tensor b = torch::sqrt(a);
  // 遍历每个设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，生成lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在当前设备上计算lazy_a的平方根，结果保存在lazy_b中
    torch::Tensor lazy_b = torch::sqrt(lazy_a);
    // 检查张量b和lazy_b是否在指定的相对和绝对误差范围内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestRsqrt) {
  // 生成一个随机的2x2张量a，使用默认设备上的浮点数选项，并取其绝对值后计算其倒数的平方根
  torch::Tensor a = torch::abs(torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  // 计算张量a的倒数的平方根，并将结果保存在张量b中
  torch::Tensor b = torch::rsqrt(a);
  // 遍历每个设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，生成lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在当前设备上计算lazy_a的倒数的平方根，结果保存在lazy_b中
    torch::Tensor lazy_b = torch::rsqrt(lazy_a);
    // 检查张量b和lazy_b是否在指定的相对和绝对误差范围内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestReciprocal) {
  // 生成一个随机的2x2张量a，使用默认设备上的浮点数选项
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量a的倒数，并将结果保存在张量b中
  torch::Tensor b = torch::reciprocal(a);
  // 遍历每个设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，生成lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在当前设备上计算lazy_a的倒数，结果保存在lazy_b中
    torch::Tensor lazy_b = torch::reciprocal(lazy_a);
    // 检查张量b和lazy_b是否在指定的相对和绝对误差范围内接近
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorScalar) {
  // 生成一个随机的2x2张量base，使用默认设备上的浮点数选项
  torch::Tensor base = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 指定一个标量指数exponent
  torch::Scalar exponent = 4.09;
  // 计算张量base的exponent次幂，并将结果保存在张量result中
  torch::Tensor result = torch::pow(base, exponent);
  // 遍历每个设备，并在每个设备上进行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量base复制到指定设备上，生成lazy_base
    torch::Tensor lazy_base = CopyToDevice(base, device);
    // 在当前设备上计算lazy_base的exponent次幂，结果保存在lazy_result中
    torch::Tensor lazy_result = torch::pow(lazy_base, exponent);
    // 检查张量result和lazy_result是否在指定的相对和绝对误差范围内接近
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}
TEST_F(LazyOpsTest, TestPowTensorScalarInPlace) {
  // 创建一个大小为2x2的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor base = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量为4.09
  torch::Scalar exponent = 4.09;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将base张量克隆到指定设备上，并转换为lazy_base
    torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
    // 对base张量执行指数幂操作，并返回结果
    torch::Tensor result = base.pow_(exponent);
    // 对lazy_base张量执行指数幂操作，并返回结果
    torch::Tensor lazy_result = lazy_base.pow_(exponent);
    // 检查result和lazy_result张量是否在相对和绝对误差容限内近似相等
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    // 检查base和lazy_base张量是否在相对和绝对误差容限内近似相等
    AllClose(base, lazy_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensor) {
  // 创建一个大小为4x2的随机张量，数据类型为float，放置在默认设备上，并对其取绝对值
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  // 创建一个与base相同大小的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor exponent = torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对base张量和exponent张量执行元素级指数幂操作，并返回结果
  torch::Tensor result = torch::pow(base, exponent);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将base张量和exponent张量分别复制到指定设备上，并转换为lazy_base和lazy_exponent
    torch::Tensor lazy_base = CopyToDevice(base, device);
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    // 对lazy_base张量和lazy_exponent张量执行元素级指数幂操作，并返回结果
    torch::Tensor lazy_result = torch::pow(lazy_base, lazy_exponent);
    // 检查result和lazy_result张量是否在相对和绝对误差容限内近似相等
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensorInPlace) {
  // 创建一个大小为4x2的随机张量，数据类型为float，放置在默认设备上，并对其取绝对值
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  // 创建一个与base相同大小的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor exponent = torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将base张量克隆到指定设备上，并转换为lazy_base
    torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
    // 对base张量执行元素级指数幂操作，并返回结果
    torch::Tensor result = base.pow_(exponent);
    // 将exponent张量复制到指定设备上，并转换为lazy_exponent
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    // 对lazy_base张量和lazy_exponent张量执行元素级指数幂操作，并返回结果
    torch::Tensor lazy_result = lazy_base.pow_(lazy_exponent);
    // 检查result和lazy_result张量是否在相对和绝对误差容限内近似相等
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    // 检查base和lazy_base张量是否在相对和绝对误差容限内近似相等
    AllClose(base, lazy_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensorBroadcast) {
  // 创建一个大小为4x2的随机张量，数据类型为float，放置在默认设备上，并对其取绝对值
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  // 创建一个大小为4x1的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor exponent = torch::rand(
      {4, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对base张量和exponent张量执行元素级指数幂操作，并返回结果
  torch::Tensor result = torch::pow(base, exponent);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将base张量和exponent张量分别复制到指定设备上，并转换为lazy_base和lazy_exponent
    torch::Tensor lazy_base = CopyToDevice(base, device);
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    // 对lazy_base张量和lazy_exponent张量执行元素级指数幂操作，并返回结果
    torch::Tensor lazy_result = torch::pow(lazy_base, lazy_exponent);
    // 检查result和lazy_result张量是否在相对和绝对误差容限内近似相等
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowScalarTensor) {
  // 创建一个标量为3.5
  torch::Scalar base = 3.5;
  // 创建一个大小为4x2的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor exponent = torch::rand({4, 2});
  // 对base标量和exponent张量执行元素级指数幂操作，并返回结果
  torch::Tensor result = torch::pow(base, exponent);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将exponent张量复制到指定设备上，并转换为lazy_exponent
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    // 对base标量和lazy_exponent张量执行元素级指数幂操作，并返回结果
    torch::Tensor lazy_result = torch::pow(base, lazy_exponent);
    // 使用 AllClose 函数对 result 和 lazy_result 进行比较
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowIntExponent) {
  // 创建一个 4x2 的张量 base，其中元素为随机生成的正数，设备为默认设备
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  // 设置指数为标量值 3
  torch::Scalar exponent = 3;
  // 计算 base 的 exponent 次幂得到结果张量 result
  torch::Tensor result = torch::pow(base, exponent);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 base 复制到指定设备上得到 lazy_base
    torch::Tensor lazy_base = CopyToDevice(base, device);
    // 在指定设备上计算 lazy_base 的 exponent 次幂得到 lazy_result
    torch::Tensor lazy_result = torch::pow(lazy_base, exponent);
    // 检查 lazy_result 是否与 result 在指定容差范围内相等
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestFmodScalar) {
  // 创建一个 2x2 的随机张量 a，元素在 [0, 100) 范围内，设备为默认设备
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 设置除数为标量值 2.0
  torch::Scalar divisor = 2.0;
  // 计算 a 对 divisor 取模得到结果张量 b
  torch::Tensor b = torch::fmod(a, divisor);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算 lazy_a 对 divisor 取模得到 lazy_b
    torch::Tensor lazy_b = torch::fmod(lazy_a, divisor);
    // 检查 lazy_b 是否与 b 在指定容差范围内相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFmodScalarInPlace) {
  // 设置除数为标量值 2.0
  torch::Scalar divisor = 2.0;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个 2x2 的随机张量 a，元素在 [0, 100) 范围内，设备为默认设备
    torch::Tensor a =
        torch::rand(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上就地计算 lazy_a 对 divisor 取模得到 b
    torch::Tensor b = a.fmod_(divisor);
    // 在指定设备上就地计算 lazy_a 对 divisor 取模得到 lazy_b
    torch::Tensor lazy_b = lazy_a.fmod_(divisor);
    // 检查 b 和 lazy_b 是否相等
    AllClose(b, lazy_b);
    // 检查 a 和 lazy_a 是否相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFmodTensor) {
  // 创建一个 2x2 的随机张量 a，元素在 [0, 100) 范围内，设备为默认设备
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 创建一个 2x2 的随机张量 b，元素在 [0, 10) 范围内，设备为默认设备
  torch::Tensor b =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  // 计算 a 对 b 每个元素取模得到结果张量 c
  torch::Tensor c = torch::fmod(a, b);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 和 b 分别复制到指定设备上得到 lazy_a 和 lazy_b
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上计算 lazy_a 对 lazy_b 每个元素取模得到 lazy_c
    torch::Tensor lazy_c = torch::fmod(lazy_a, lazy_b);
    // 检查 lazy_c 是否与 c 在指定容差范围内相等
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestFmodTensorInPlace) {
  // 创建一个 2x2 的随机张量 b，元素在 [0, 10) 范围内，设备为默认设备
  torch::Tensor b =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个 2x2 的随机张量 a，元素在 [0, 100) 范围内，设备为默认设备
    torch::Tensor a =
        torch::rand(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上就地计算 lazy_a 对 b 每个元素取模得到 c
    torch::Tensor c = a.fmod_(b);
    // 将张量 b 复制到指定设备上得到 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在指定设备上就地计算 lazy_a 对 lazy_b 每个元素取模得到 lazy_c
    torch::Tensor lazy_c = lazy_a.fmod_(lazy_b);
    // 检查 c 和 lazy_c 是否相等
    AllClose(c, lazy_c);
    // 检查 a 和 lazy_a 是否相等
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestRemainderScalar) {
  // 创建一个 2x2 的随机张量 a，元素在 [-100, 100) 范围内，设备为默认设备
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 设置除数为标量值 -2.0
  torch::Scalar divisor = -2.0;
  // 计算 a 对 divisor 每个元素取余得到结果张量 b
  torch::Tensor b = torch::remainder(a, divisor);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上得到 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在指定设备上计算 lazy_a 对 divisor 每个元素取余得到 lazy_b
    torch::Tensor lazy_b = torch::remainder(lazy_a, divisor);
    // 使用 torch::remainder 函数计算 lazy_a 除以 divisor 的余数，返回结果作为 lazy_b
    torch::Tensor lazy_b = torch::remainder(lazy_a, divisor);
    // 调用 AllClose 函数检查 b 和 lazy_b 的数值是否全部接近，通常用于测试或验证
    AllClose(b, lazy_b);
}

TEST_F(LazyOpsTest, TestRemainderScalarInPlace) {
  // 定义标量除数为 -2.0
  torch::Scalar divisor = -2.0;
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建大小为 {2, 2} 的随机张量 a，值在设备上，范围乘以 100.0
    torch::Tensor a =
        torch::randn(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    // 将张量 a 复制到指定设备上，形成 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在原张量 a 上执行取余操作，结果存储在 b 中
    torch::Tensor b = a.remainder_(divisor);
    // 在 lazy_a 上执行取余操作，结果存储在 lazy_b 中
    torch::Tensor lazy_b = lazy_a.remainder_(divisor);
    // 检查 b 和 lazy_b 是否接近
    AllClose(b, lazy_b);
    // 检查 a 和 lazy_a 是否接近
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestRemainderTensor) {
  // 创建大小为 {2, 2} 的随机张量 a，值在设备上，范围乘以 100.0
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  // 创建大小为 {2, 2} 的随机张量 b，值在设备上，范围乘以 10.0
  torch::Tensor b =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  // 使用 torch::remainder 函数计算张量 a 与 b 的余数，结果存储在 c 中
  torch::Tensor c = torch::remainder(a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，形成 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备上，形成 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在 lazy_a 和 lazy_b 上执行取余操作，结果存储在 lazy_c 中
    torch::Tensor lazy_c = torch::remainder(lazy_a, lazy_b);
    // 检查 c 和 lazy_c 是否接近，指定相对误差和绝对误差
    AllClose(c, lazy_c, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(LazyOpsTest, TestRemainderTensorInPlace) {
  // 创建大小为 {2, 2} 的随机张量 b，值在设备上，范围乘以 10.0
  torch::Tensor b =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建大小为 {2, 2} 的随机张量 a，值在设备上，范围乘以 100.0
    torch::Tensor a =
        torch::randn(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    // 将张量 a 复制到指定设备上，形成 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 在原张量 a 上执行取余操作，结果存储在 c 中
    torch::Tensor c = a.remainder_(b);
    // 将张量 b 复制到指定设备上，形成 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 在 lazy_a 和 lazy_b 上执行取余操作，结果存储在 lazy_c 中
    torch::Tensor lazy_c = lazy_a.remainder_(lazy_b);
    // 检查 c 和 lazy_c 是否接近，指定相对误差和绝对误差
    AllClose(c, lazy_c, /*rtol=*/1e-4, /*atol=*/1e-6);
    // 检查 a 和 lazy_a 是否接近，指定相对误差和绝对误差
    AllClose(a, lazy_a, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(LazyOpsTest, TestWhere) {
  // 创建大小为 {3, 3} 的随机张量 a，值在设备上
  torch::Tensor a = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 3} 的随机张量 b，值在设备上
  torch::Tensor b = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建大小为 {3, 3} 的空张量 c，数据类型为 torch::kByte，在设备上
  torch::Tensor c = torch::empty(
      {3, 3}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 循环初始化张量 c，使对角线为真
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  // 使用 torch::where 函数基于张量 c 的真值，选择 a 或 b 中的值，结果存储在 d 中
  torch::Tensor d = torch::where(c, a, b);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上，形成 lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备上，形成 lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量 c 复制到指定设备上，形成 lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在 lazy_c 的真值处，选择 lazy_a 或 lazy_b 的值，结果存储在 lazy_d 中
    torch::Tensor lazy_d = torch::where(lazy_c, lazy_a, lazy_b);
    // 检查 d 和 lazy_d 是否接近
    AllClose(d, lazy_d);
  });
}
TEST_F(LazyOpsTest, TestWhereBroadcast) {
  // 创建一个大小为3x3的随机浮点数张量a
  torch::Tensor a = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量值为0的浮点数张量b
  torch::Tensor b = torch::zeros(
      {}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为3x3的空字节类型张量c
  torch::Tensor c = torch::empty(
      {3, 3}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  // 使用循环为张量c赋值，对角线元素为1，其余为0
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  // 根据条件张量c，在a和b张量间进行广播操作，生成结果张量d
  torch::Tensor d = torch::where(c, a, b);
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量a复制到指定设备上，生成lazy_a
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量b复制到指定设备上，生成lazy_b
    torch::Tensor lazy_b = CopyToDevice(b, device);
    // 将张量c复制到指定设备上，生成lazy_c
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 在指定设备上根据条件张量lazy_c，进行广播操作，生成lazy_d
    torch::Tensor lazy_d = torch::where(lazy_c, lazy_a, lazy_b);
    // 验证结果张量d和lazy_d在当前设备上是否全部接近
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestThreshold) {
  // 创建一个大小为2x1x4x6的随机浮点数输入张量input
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置阈值和值
  float threshold = 0.4;
  float value = 20;
  // 在输入张量上进行阈值处理，生成输出张量output
  torch::Tensor output = torch::threshold(input, threshold, value);
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量input复制到指定设备上，生成lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行阈值处理，生成lazy_output
    torch::Tensor lazy_output = torch::threshold(lazy_input, threshold, value);
    // 验证结果张量output和lazy_output在当前设备上是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestThresholdBackward) {
  // 设置阈值和值
  float threshold = 0.4;
  float value = 20;
  // 定义测试函数，对输入张量进行阈值处理
  auto testFunction =
      [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::threshold(inputs[0], threshold, value);
  };
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 在指定设备上进行反向传播测试，验证梯度是否正确
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testFunction);
  });
}

TEST_F(LazyOpsTest, TestThresholdInPlace) {
  // 创建一个大小为2x1x4x6的随机浮点数输入张量input
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 将输入张量input克隆到输出张量output
  torch::Tensor output = input.clone();
  // 设置阈值和值
  float threshold = 0.4;
  float value = 20;
  // 在输出张量上进行就地阈值处理
  torch::threshold_(output, threshold, value);
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量input复制到指定设备上，生成lazy_output
    torch::Tensor lazy_output = CopyToDevice(input, device);
    // 在指定设备上进行就地阈值处理，生成lazy_output
    torch::threshold_(lazy_output, threshold, value);
    // 验证结果张量output和lazy_output在当前设备上是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestElu) {
  // 创建一个大小为2x1x4x6的随机浮点数输入张量input
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置ELU函数的参数
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  // 对输入张量input进行ELU激活函数操作，生成输出张量output
  torch::Tensor output = torch::elu(input, alpha, scale, input_scale);
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量input复制到指定设备上，生成lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上对输入张量进行ELU激活函数操作，生成lazy_output
    torch::Tensor lazy_output =
        torch::elu(lazy_input, alpha, scale, input_scale);
    // 验证结果张量output和lazy_output在当前设备上是否全部接近
    AllClose(output, lazy_output);
  });
}
TEST_F(LazyOpsTest, TestEluInPlace) {
  // 创建一个随机张量作为输入，形状为[2, 1, 4, 6]，使用默认设备上的浮点张量选项
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 alpha、scale 和 input_scale 为标量值
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地计算 ELU 激活函数，返回结果到 output
    torch::Tensor output = torch::elu_(input, alpha, scale, input_scale);
    // 在原地计算 ELU 激活函数，返回结果到 lazy_output
    torch::Tensor lazy_output =
        torch::elu_(lazy_input, alpha, scale, input_scale);
    // 检查 output 和 lazy_output 是否在数值上接近
    AllClose(output, lazy_output);
    // 检查 input 和 lazy_input 是否在数值上接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestSelu) {
  // 创建一个随机张量作为输入，形状为[2, 1, 4, 6]，使用默认设备上的浮点张量选项
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算 SELU 激活函数，并将结果保存到 output
  torch::Tensor output = torch::selu(input);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在 lazy_input 上计算 SELU 激活函数，返回结果到 lazy_output
    torch::Tensor lazy_output = torch::selu(lazy_input);
    // 检查 output 和 lazy_output 是否在数值上接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSeluInPlace) {
  // 创建一个随机张量作为输入，形状为[2, 1, 4, 6]，使用默认设备上的浮点张量选项
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地计算 SELU 激活函数，返回结果到 output
    torch::Tensor output = torch::selu_(input);
    // 在原地计算 SELU 激活函数，返回结果到 lazy_output
    torch::Tensor lazy_output = torch::selu_(lazy_input);
    // 检查 output 和 lazy_output 是否在数值上接近
    AllClose(output, lazy_output);
    // 检查 input 和 lazy_input 是否在数值上接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestCelu) {
  // 创建一个随机张量作为输入，形状为[2, 1, 4, 6]，使用默认设备上的浮点张量选项
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 alpha 为标量值
  torch::Scalar alpha = 2.5;
  // 计算 CELU 激活函数，并将结果保存到 output
  torch::Tensor output = torch::celu(input, alpha);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在 lazy_input 上计算 CELU 激活函数，返回结果到 lazy_output
    torch::Tensor lazy_output = torch::celu(lazy_input, alpha);
    // 检查 output 和 lazy_output 是否在数值上接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestCeluInPlace) {
  // 创建一个随机张量作为输入，形状为[2, 1, 4, 6]，使用默认设备上的浮点张量选项
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 alpha 为标量值
  torch::Scalar alpha = 2.5;
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地计算 CELU 激活函数，返回结果到 output
    torch::Tensor output = torch::celu_(input, alpha);
    // 在原地计算 CELU 激活函数，返回结果到 lazy_output
    torch::Tensor lazy_output = torch::celu_(lazy_input, alpha);
    // 检查 output 和 lazy_output 是否在数值上接近
    AllClose(output, lazy_output);
    // 检查 input 和 lazy_input 是否在数值上接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestGelu) {
  // 创建一个随机张量作为输入，形状为[2, 3]，使用默认设备上的浮点张量选项
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算 GELU 激活函数，并将结果保存到 output
  torch::Tensor output = torch::gelu(input);
  // 对于每个设备，执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在 lazy_input 上计算 GELU 激活函数，返回结果到 lazy_output
    torch::Tensor lazy_output = torch::gelu(lazy_input);
    // 检查 output 和 lazy_output 是否在数值上接近
    AllClose(output, lazy_output);
  });
}
TEST_F(LazyOpsTest, TestAddMatMul) {
  // 设置输入通道数和输出通道数
  int in_channels = 32;
  int out_channels = 320;
  // 设置标签数
  int labels = 50;
  // 创建随机浮点数张量作为输入，指定在默认设备上
  torch::Tensor input = torch::rand(
      {in_channels, out_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机浮点数张量作为权重，指定在默认设备上
  torch::Tensor weight = torch::rand(
      {out_channels, labels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机浮点数张量作为偏置，指定在默认设备上
  torch::Tensor bias = torch::rand(
      {labels}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 测试 beta 不等于 1，通过 CPU 互操作
  for (double beta : {1., 2.}) {
    // 使用 addmm 函数进行矩阵相加操作，包括 beta 参数
    torch::Tensor output = torch::addmm(bias, input, weight, /*beta=*/beta);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 将权重张量复制到指定设备上
      torch::Tensor lazy_weight = CopyToDevice(weight, device);
      // 将偏置张量复制到指定设备上
      torch::Tensor lazy_bias = CopyToDevice(bias, device);
      // 在指定设备上使用 addmm 函数进行懒惰计算，包括 beta 参数
      torch::Tensor lazy_output =
          torch::addmm(lazy_bias, lazy_input, lazy_weight, /*beta=*/beta);
      // 检查输出结果是否相等
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestEmbedding) {
  // 创建随机浮点数张量 a，指定在默认设备上
  torch::Tensor a = torch::rand(
      {32, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机长整型张量 i，指定在默认设备上
  torch::Tensor i = torch::randint(
      0,
      31,
      {3, 4},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 使用 embedding 函数创建嵌入张量 b，包括 padding_idx、scale_grad_by_freq 和 sparse 参数
  torch::Tensor b = torch::embedding(
      a,
      i,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备上
    torch::Tensor lazy_a = CopyToDevice(a, device);
    // 将张量 i 复制到指定设备上
    torch::Tensor lazy_i = CopyToDevice(i, device);
    // 在指定设备上使用 embedding 函数创建嵌入张量 lazy_b，包括 padding_idx、scale_grad_by_freq 和 sparse 参数
    torch::Tensor lazy_b = torch::embedding(
        lazy_a,
        lazy_i,
        /*padding_idx=*/0,
        /*scale_grad_by_freq=*/false,
        /*sparse=*/false);
    // 检查输出结果是否相等
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestOneHot) {
  // 设置类别数
  int num_classes = 5;
  // 创建随机长整型张量 input，指定在默认设备上
  torch::Tensor input = torch::randint(
      0,
      num_classes,
      {10},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  // 使用 one_hot 函数创建独热编码张量 output，指定类别数
  torch::Tensor output = torch::one_hot(input, num_classes);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上使用 one_hot 函数创建独热编码张量 lazy_output，指定类别数
    torch::Tensor lazy_output = torch::one_hot(lazy_input, num_classes);
    // 检查输出结果是否相等
    AllEqual(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTranspose) {
  // 创建随机浮点数张量 input，指定在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 使用 t 函数进行转置操作，创建转置后的张量 output
  torch::Tensor output = torch::t(input);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上使用 t 函数进行懒惰转置操作，创建转置后的张量 lazy_output
    torch::Tensor lazy_output = torch::t(lazy_input);
    // 检查输出结果是否相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeInPlace) {
  // 创建随机浮点数张量 input，指定在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);

    // 在指定设备上执行原地转置操作
    torch::Tensor lazy_input_transposed = torch::t_(lazy_input);
    // 检查原地转置后的张量是否与原始转置操作结果相等
    AllClose(input, lazy_input_transposed);
  });
}
    // 将输入张量进行转置，并赋值给输出张量
    torch::Tensor output = input.t_();
    // 将惰性输入张量进行转置，并赋值给惰性输出张量
    torch::Tensor lazy_output = lazy_input.t_();
    // 断言：确保惰性输出张量的大小与输出张量的大小相等
    EXPECT_EQ(lazy_output.sizes(), output.sizes());
    // 检查两个张量是否在误差容限内全部相等
    AllClose(output, lazy_output);
    // 检查输入张量与惰性输入张量是否在误差容限内全部相等
    AllClose(input, lazy_input);
  });
TEST_F(LazyOpsTest, TestReshape) {
  // 创建一个大小为 {32, 20, 4, 4} 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对输入张量进行形状重塑为 {-1, 320} 的张量
  torch::Tensor output = torch::reshape(input, {-1, 320});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，生成 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 进行形状重塑为 {-1, 320}
    torch::Tensor lazy_output = torch::reshape(lazy_input, {-1, 320});
    // 检查 output 和 lazy_output 是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestResize) {
  // 创建一个大小为 {2, 2, 4} 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 保存 input 的副本
  torch::Tensor saved_input = input.clone();
  // 使用 resize_() 方法将 input 的大小调整为 {3, 3}
  input.resize_({3, 3});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 saved_input 复制到指定设备上，生成 lazy_input
    torch::Tensor lazy_input = CopyToDevice(saved_input, device);
    // 使用 resize_() 方法将 lazy_input 的大小调整为 {3, 3}
    lazy_input.resize_({3, 3});
    // 检查 input 和 lazy_input 是否全部接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestViewResize) {
  // 创建一个大小为 {8, 2} 的零张量，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::zeros(
      {8, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 保存 input 的副本
  torch::Tensor saved_input = input.clone();
  // 使用 view() 方法将 input 转换为大小为 {4, 4} 的张量，output 保存结果
  torch::Tensor output = input.view({4, 4});
  // 使用 resize_() 方法将 output 的大小调整为 {3, 3}
  output.resize_({3, 3});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 saved_input 复制到指定设备上，生成 lazy_input
    torch::Tensor lazy_input = CopyToDevice(saved_input, device);
    // 使用 view() 方法将 lazy_input 转换为大小为 {4, 4} 的张量，lazy_output 保存结果
    torch::Tensor lazy_output = lazy_input.view({4, 4});
    // 使用 resize_() 方法将 lazy_output 的大小调整为 {3, 3}
    lazy_output.resize_({3, 3});
    // 检查 input 和 lazy_input 是否全部接近
    AllClose(input, lazy_input);
    // 检查 output 和 lazy_output 是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestView) {
  // 创建一个大小为 {32, 20, 4, 4} 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 使用 view() 方法将 input 转换为大小为 {-1, 320} 的张量，output 保存结果
  torch::Tensor output = input.view({-1, 320});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，生成 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 view() 方法将 lazy_input 转换为大小为 {-1, 320} 的张量，lazy_output 保存结果
    torch::Tensor lazy_output = lazy_input.view({-1, 320});
    // 检查 output 和 lazy_output 是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestViewMod) {
  // 创建一个大小为 {32, 20, 4, 4} 的零张量，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::zeros(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个值为 1.0 的张量，数据类型为 float，放置在默认设备上
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 使用 view() 方法将 input 转换为大小为 {-1, 320} 的张量，output 保存结果
  torch::Tensor output = input.view({-1, 320});
  // 对 output 应用 add_() 方法，增加值为 1.0 的 one
  output.add_(one, 1.0);
  // 对 input 应用 add_() 方法，增加值为 1.0 的 one
  input.add_(one, 1.0);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个大小为 {32, 20, 4, 4} 的零张量 xinput，数据类型为 float，放置在指定设备上
    torch::Tensor xinput = torch::zeros(
        {32, 20, 4, 4},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将 xinput 复制到指定设备上，生成 lazy_input
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    // 将 one 复制到指定设备上，生成 lazy_one
    torch::Tensor lazy_one = CopyToDevice(one, device);
    // 使用 view() 方法将 lazy_input 转换为大小为 {-1, 320} 的张量，lazy_output 保存结果
    torch::Tensor lazy_output = lazy_input.view({-1, 320});
    // 对 lazy_output 应用 add_() 方法，增加值为 1.0 的 lazy_one
    lazy_output.add_(lazy_one, 1.0);
    // 对 lazy_input 应用 add_() 方法，增加值为 1.0 的 lazy_one
    lazy_input.add_(lazy_one, 1.0);
    // 检查 output 和 lazy_output 是否全部接近
    AllClose(output, lazy_output);
    // 检查 input 和 lazy_input 是否全部接近
    AllClose(input, lazy_input);
  });
}
TEST_F(LazyOpsTest, TestViewModComplex) {
  // 创建一个形状为 {32, 20, 4, 4}，元素全为零的浮点类型张量，放置在默认设备上
  torch::Tensor input = torch::zeros(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个值为 1.0 的浮点类型张量，放置在默认设备上
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 进行形状重塑，变为 {-1, 320}，返回结果为 output1
  torch::Tensor output1 = input.view({-1, 320});
  // output1 张量中的所有元素加上 one 张量的每个元素的乘积，原地操作
  output1.add_(one, 1.0);
  // 对 input 再次进行形状重塑，变为 {-1, 160}，返回结果为 output2
  torch::Tensor output2 = input.view({-1, 160});
  // output2 张量中的所有元素加上 one 张量的每个元素的乘积，原地操作
  output2.add_(one, 1.0);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 {32, 20, 4, 4}，元素全为零的浮点类型张量，放置在当前设备上
    torch::Tensor xinput = torch::zeros(
        {32, 20, 4, 4},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将 xinput 张量复制到指定设备上，返回 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    // 将 one 张量复制到指定设备上，返回 lazy_one 张量
    torch::Tensor lazy_one = CopyToDevice(one, device);
    // 对 lazy_input 进行形状重塑，变为 {-1, 320}，返回结果为 lazy_output1
    torch::Tensor lazy_output1 = lazy_input.view({-1, 320});
    // lazy_output1 张量中的所有元素加上 lazy_one 张量的每个元素的乘积，原地操作
    lazy_output1.add_(lazy_one, 1.0);
    // 对 lazy_input 再次进行形状重塑，变为 {-1, 160}，返回结果为 lazy_output2
    torch::Tensor lazy_output2 = lazy_input.view({-1, 160});
    // lazy_output2 张量中的所有元素加上 lazy_one 张量的每个元素的乘积，原地操作
    lazy_output2.add_(lazy_one, 1.0);
    // 检查 output1 和 lazy_output1 是否在指定设备上全部近似相等
    AllClose(output1, lazy_output1);
    // 检查 output2 和 lazy_output2 是否在指定设备上全部近似相等
    AllClose(output2, lazy_output2);
  });
}

TEST_F(LazyOpsTest, TestViewOfViewMod) {
  // 创建一个形状为 {32, 20, 4, 4}，元素全为零的浮点类型张量，放置在默认设备上
  torch::Tensor input = torch::zeros(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个值为 1.0 的浮点类型张量，放置在默认设备上
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 进行形状重塑，变为 {-1, 320}，返回结果为 output1
  torch::Tensor output1 = input.view({-1, 320});
  // output1 张量中的所有元素加上 one 张量的每个元素的乘积，原地操作
  output1.add_(one, 1.0);
  // 对 output1 进行形状重塑，变为 {-1, 160}，返回结果为 output2
  torch::Tensor output2 = output1.view({-1, 160});
  // output2 张量中的所有元素加上 one 张量的每个元素的乘积，原地操作
  output2.add_(one, 1.0);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 {32, 20, 4, 4}，元素全为零的浮点类型张量，放置在当前设备上
    torch::Tensor xinput = torch::zeros(
        {32, 20, 4, 4},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将 xinput 张量复制到指定设备上，返回 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    // 将 one 张量复制到指定设备上，返回 lazy_one 张量
    torch::Tensor lazy_one = CopyToDevice(one, device);
    // 对 lazy_input 进行形状重塑，变为 {-1, 320}，返回结果为 lazy_output1
    torch::Tensor lazy_output1 = lazy_input.view({-1, 320});
    // lazy_output1 张量中的所有元素加上 lazy_one 张量的每个元素的乘积，原地操作
    lazy_output1.add_(lazy_one, 1.0);
    // 对 lazy_output1 进行形状重塑，变为 {-1, 160}，返回结果为 lazy_output2
    torch::Tensor lazy_output2 = lazy_output1.view({-1, 160});
    // lazy_output2 张量中的所有元素加上 lazy_one 张量的每个元素的乘积，原地操作
    lazy_output2.add_(lazy_one, 1.0);
    // 检查 output1 和 lazy_output1 是否在指定设备上全部近似相等
    AllClose(output1, lazy_output1);
    // 检查 output2 和 lazy_output2 是否在指定设备上全部近似相等
    AllClose(output2, lazy_output2);
  });
}

TEST_F(LazyOpsTest, TestViewSqueezeAddInPlace) {
  // 创建一个形状为 {2, 3, 1}，元素全为零的浮点类型张量，放置在默认设备上
  torch::Tensor input = torch::zeros(
      {2, 3, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个视图的形状，{2, 3, 1, 1}
  std::vector<int64_t> view_size = {2, 3, 1, 1};
  // 压缩维度的索引，这里是第 2 维
  int squeeze_dim = 2;
  // 创建一个值为 1.0 的浮点类型张量，放置在默认设备上
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 张量复制到指定设备上，返回 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 input 进行形状重塑，变为 view_size，返回结果为 output
    torch::Tensor output = input.view(view_size);
    // 在指定维度上压缩 output 张量，原地操作
    output.squeeze_(squeeze_dim);
    // output 张量中的所有元素加上 one 张量的每个元素的乘积，原地操作
    output.add_(one, 1.0);
    // 将 one 张量复制到指定设备上，返回 lazy_one 张量
    torch::Tensor lazy_one = CopyToDevice(one, device);
    // 对 lazy_input 进行形状重塑，变为 view_size，返回结果为 lazy_output
    torch::Tensor lazy_output = lazy_input.view(view
TEST_F(LazyOpsTest, TestUnsafeView) {
  // 创建一个形状为 {32, 20, 4, 4} 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 使用 _unsafe_view 函数对输入张量进行形状变换，将其视图调整为 {-1, 320}
  torch::Tensor output = torch::_unsafe_view(input, {-1, 320});
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 张量复制到指定设备，得到 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 _unsafe_view 函数在指定设备上对 lazy_input 进行形状变换，将其视图调整为 {-1, 320}
    torch::Tensor lazy_output = torch::_unsafe_view(lazy_input, {-1, 320});
    // 检查 output 和 lazy_output 是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestNarrow) {
  // 创建一个形状为 {8, 10, 4, 4} 的随机张量，数据类型为 float，放置在默认设备上
  torch::Tensor a = torch::rand(
      {8, 10, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于每个维度 dim 和起始位置 start 执行以下操作
  for (int64_t dim : {1, -3}) {
    for (int64_t start : {2, -8}) {
      // 使用 narrow 函数在指定维度 dim 上从 start 开始取长度为 6 的子张量 b
      torch::Tensor b = a.narrow(dim, start, 6);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将张量 a 复制到指定设备，得到 lazy_a 张量
        torch::Tensor lazy_a = CopyToDevice(a, device);
        // 使用 narrow 函数在指定设备上对 lazy_a 在维度 dim 上从 start 开始取长度为 6 的子张量 lazy_b
        torch::Tensor lazy_b = lazy_a.narrow(dim, start, 6);
        // 检查 b 和 lazy_b 是否近似相等
        AllClose(b, lazy_b);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdate) {
  // 对于每个维度 dim 和起始位置 start 执行以下操作
  for (int64_t dim : {1, -2}) {
    for (int64_t start : {2, -6}) {
      // 创建一个形状为 {3, 8, 3} 的随机张量 a，数据类型为 float，放置在默认设备上
      torch::Tensor a = torch::rand(
          {3, 8, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 复制 a 得到 a_copy 张量
      torch::Tensor a_copy = a.clone();
      // 创建一个形状为 {3, 4, 3} 的随机张量 b，数据类型为 float，放置在默认设备上
      torch::Tensor b = torch::rand(
          {3, 4, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 使用 narrow 函数在指定维度 dim 上从 start 开始取长度为 4 的子张量 c
      torch::Tensor c = a.narrow(dim, start, 4);
      // 将 b 加到 c 上，乘以因子 1.0
      c.add_(b, 1.0);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 a_copy 和 b 复制到指定设备，得到 lazy_a 和 lazy_b 张量
        torch::Tensor lazy_a = CopyToDevice(a_copy, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        // 使用 narrow 函数在指定设备上对 lazy_a 在维度 dim 上从 start 开始取长度为 4 的子张量 lazy_c
        torch::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        // 将 lazy_b 加到 lazy_c 上，乘以因子 1.0
        lazy_c.add_(lazy_b, 1.0);
        // 检查 c 和 lazy_c 是否近似相等
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateBaseCheck) {
  // 对于每个维度 dim 和起始位置 start 执行以下操作
  for (int64_t dim : {0, -2}) {
    for (int64_t start : {2, -6}) {
      // 创建一个形状为 {8, 3} 的零张量 a，数据类型为 float，放置在默认设备上
      torch::Tensor a = torch::zeros(
          {8, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 复制 a 得到 a_copy 张量
      torch::Tensor a_copy = a.clone();
      // 创建一个形状为 {4, 3} 的全一张量 b，数据类型为 float，放置在默认设备上
      torch::Tensor b = torch::ones(
          {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 使用 narrow 函数在指定维度 dim 上从 start 开始取长度为 4 的子张量 c
      torch::Tensor c = a.narrow(dim, start, 4);
      // 将 b 加到 c 上，乘以因子 1.0
      c.add_(b, 1.0);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将 a_copy 和 b 复制到指定设备，得到 lazy_a 和 lazy_b 张量
        torch::Tensor lazy_a = CopyToDevice(a_copy, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        // 使用 narrow 函数在指定设备上对 lazy_a 在维度 dim 上从 start 开始取长度为 4 的子张量 lazy_c
        torch::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        // 将 lazy_b 加到 lazy_c 上，乘以因子 1.0
        lazy_c.add_(lazy_b, 1.0);
        // 检查 a 和 lazy_a 是否近似相等
        AllClose(a, lazy_a);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateTwoSlices) {
  // 对于每个维度 dim 执行以下操作
  for (int64_t dim : {0, -2}) {
    for (int64_t start0 : {2, -6}) {
      // 在两个指定的起始位置循环：2 和 -6
      for (int64_t start1 : {6, -2}) {
        // 在两个指定的起始位置循环：6 和 -2
        torch::Tensor a = torch::zeros(
            {8, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        // 创建一个8行3列的全零张量 a，数据类型为 float，在默认设备上
        torch::Tensor a_copy = a.clone();
        // 克隆张量 a，赋值给 a_copy
        torch::Tensor b = torch::ones(
            {2, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        // 创建一个2行3列的全一张量 b，数据类型为 float，在默认设备上
        torch::Tensor c = b + 1;
        // 创建张量 c，其值为张量 b 的每个元素加1
        torch::Tensor d = a.narrow(dim, start0, 2);
        // 根据指定维度 dim，从张量 a 的 start0 处开始，长度为2的窄切片，赋值给 d
        torch::Tensor e = a.narrow(dim, start1, 2);
        // 根据指定维度 dim，从张量 a 的 start1 处开始，长度为2的窄切片，赋值给 e
        d.add_(b, 1.0);
        // 将张量 d 的每个元素加上张量 b 的对应元素，并加上偏移值 1.0，结果存回 d
        e.add_(c, 1.0);
        // 将张量 e 的每个元素加上张量 c 的对应元素，并加上偏移值 1.0，结果存回 e
        ForEachDevice([&](const torch::Device& device) {
          // 对每个设备执行以下操作：
          torch::Tensor lazy_a = CopyToDevice(a_copy, device);
          // 将张量 a_copy 拷贝到指定设备上，得到 lazy_a
          torch::Tensor lazy_b = CopyToDevice(b, device);
          // 将张量 b 拷贝到指定设备上，得到 lazy_b
          torch::Tensor lazy_c = CopyToDevice(c, device);
          // 将张量 c 拷贝到指定设备上，得到 lazy_c
          torch::Tensor lazy_d = lazy_a.narrow(dim, start0, 2);
          // 在指定设备上，根据指定维度 dim，从 lazy_a 的 start0 处开始，长度为2的窄切片，赋值给 lazy_d
          torch::Tensor lazy_e = lazy_a.narrow(dim, start1, 2);
          // 在指定设备上，根据指定维度 dim，从 lazy_a 的 start1 处开始，长度为2的窄切片，赋值给 lazy_e
          lazy_d.add_(lazy_b, 1.0);
          // 在指定设备上，将张量 lazy_d 的每个元素加上 lazy_b 的对应元素，并加上偏移值 1.0，结果存回 lazy_d
          lazy_e.add_(lazy_c, 1.0);
          // 在指定设备上，将张量 lazy_e 的每个元素加上 lazy_c 的对应元素，并加上偏移值 1.0，结果存回 lazy_e
          AllClose(d, lazy_d);
          // 检查张量 d 和 lazy_d 是否近似相等
          AllClose(e, lazy_e);
          // 检查张量 e 和 lazy_e 是否近似相等
          AllClose(a, lazy_a);
          // 检查张量 a 和 lazy_a 是否近似相等
        });
      }
    }
}

TEST_F(LazyOpsTest, TestNarrowUpdateView) {
  for (int64_t dim : {0, -3}) {  // 遍历维度0和倒数第3维
    for (int64_t start : {2, -6}) {  // 遍历起始索引2和倒数第6个位置
      torch::Tensor a = torch::rand(  // 创建形状为{8, 2, 3}的随机张量a
          {8, 2, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor a_copy = a.clone();  // 克隆张量a为a_copy
      torch::Tensor b = torch::rand(  // 创建形状为{4, 6}的随机张量b
          {4, 6}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor c = a.narrow(dim, start, 4);  // 在维度dim上从start开始缩窄为长度4的张量c
      torch::Tensor d = c.view({4, 6});  // 将张量c视图重塑为形状{4, 6}的张量d
      d.add_(b, 1.0);  // 张量d和张量b相加（就地加法）
      ForEachDevice([&](const torch::Device& device) {  // 对于每个设备执行以下操作
        torch::Tensor lazy_a = CopyToDevice(a_copy, device);  // 将a_copy复制到指定设备上得到lazy_a
        torch::Tensor lazy_b = CopyToDevice(b, device);  // 将b复制到指定设备上得到lazy_b
        torch::Tensor lazy_c = lazy_a.narrow(dim, start, 4);  // 在指定设备上对lazy_a进行维度dim上的缩窄操作
        torch::Tensor lazy_d = lazy_c.view({4, 6});  // 将lazy_c视图重塑为形状{4, 6}的lazy_d
        lazy_d.add_(lazy_b, 1.0);  // lazy_d和lazy_b相加（就地加法）
        AllClose(d, lazy_d);  // 检查d和lazy_d是否在设备上全部相等
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowInNarrowUpdate) {
  for (int64_t dim : {1, -2}) {  // 遍历维度1和倒数第2维
    for (int64_t start0 : {1, -7}) {  // 遍历起始索引1和倒数第7个位置
      for (int64_t start1 : {1, -5}) {  // 遍历起始索引1和倒数第5个位置
        torch::Tensor a = torch::rand(  // 创建形状为{3, 8, 3}的随机张量a
            {3, 8, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor a_copy = a.clone();  // 克隆张量a为a_copy
        torch::Tensor b = torch::rand(  // 创建形状为{3, 2, 3}的随机张量b
            {3, 2, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor c = a.narrow(dim, start0, 6);  // 在维度dim上从start0开始缩窄为长度6的张量c
        torch::Tensor d = c.narrow(dim, start1, 2);  // 在维度dim上从start1开始缩窄为长度2的张量d
        d.add_(b, 1.0);  // 张量d和张量b相加（就地加法）
        ForEachDevice([&](const torch::Device& device) {  // 对于每个设备执行以下操作
          torch::Tensor lazy_a = CopyToDevice(a_copy, device);  // 将a_copy复制到指定设备上得到lazy_a
          torch::Tensor lazy_b = CopyToDevice(b, device);  // 将b复制到指定设备上得到lazy_b
          torch::Tensor lazy_c = lazy_a.narrow(dim, start0, 6);  // 在指定设备上对lazy_a进行维度dim上的缩窄操作
          torch::Tensor lazy_d = lazy_c.narrow(dim, start1, 2);  // 在指定设备上对lazy_c进行维度dim上的缩窄操作
          lazy_d.add_(lazy_b, 1.0);  // lazy_d和lazy_b相加（就地加法）
          AllClose(a, lazy_a);  // 检查a和lazy_a是否在设备上全部相等
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowCopy) {
  for (int64_t dim : {1, -3}) {  // 遍历维度1和倒数第3维
    for (int64_t start : {2, -8}) {  // 遍历起始索引2和倒数第8个位置
      ForEachDevice([&](const torch::Device& device) {  // 对于每个设备执行以下操作
        torch::Tensor input = torch::rand(  // 创建形状为{8, 10, 4, 4}的随机张量input
            {8, 10, 4, 4},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor lazy_input = CopyToDevice(input, device);  // 将input复制到指定设备上得到lazy_input
        torch::Tensor result = input.narrow_copy(dim, start, 6);  // 在维度dim上从start开始复制长度为6的张量result
        input.add_(1);  // input每个元素加1（就地加法）
        torch::Tensor lazy_result = lazy_input.narrow_copy(dim, start, 6);  // 在指定设备上对lazy_input进行维度dim上的复制操作
        lazy_input.add_(1);  // lazy_input每个元素加1（就地加法）
        AllClose(result, lazy_result);  // 检查result和lazy_result是否在设备上全部相等
      });
    }
  }
}

TEST_F(LazyOpsTest, TestViewAs) {
  torch::Tensor input = torch::rand(  // 创建形状为{32, 20, 4, 4}的随机张量input
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor empty = torch::empty({32, 320});  // 创建形状为{32, 320}的空张量empty
  torch::Tensor output = input.view_as(empty);  // 将input视图重塑为empty的形状
  ForEachDevice([&](const torch::Device& device) {  // 对于每个设备执行以下操作
    torch::Tensor lazy_input = CopyToDevice(input, device);  // 将input复制到指定设备上得到lazy_input
    torch::Tensor lazy_empty = CopyToDevice(empty, device);  // 将empty复制到指定设备上得到lazy_empty
    // 将 lazy_input 张量重塑为与 lazy_empty 相同的形状，并存储为 lazy_output
    torch::Tensor lazy_output = lazy_input.view_as(lazy_empty);
    // 检查 output 和 lazy_output 是否全部接近（元素级比较），返回布尔值
    AllClose(output, lazy_output);
});
TEST_F(LazyOpsTest, TestLogSoftmax) {
  // 创建一个随机张量作为输入，形状为 [5, 3, 4, 2]，浮点数类型，使用默认设备
  torch::Tensor input = torch::rand(
      {5, 3, 4, 2},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 获取输入张量的维度
    int rank = input.dim();
    // 在每一个维度上执行 log_softmax 操作
    for (int dim = -rank; dim < rank; ++dim) {
      // 计算输入张量在指定维度上的 log_softmax 结果
      torch::Tensor output = torch::log_softmax(input, dim);
      // 在指定设备上计算输入张量在指定维度上的 log_softmax 结果
      torch::Tensor lazy_output = torch::log_softmax(lazy_input, dim);
      // 检查两个张量的值是否在相对容差 1e-3 内全部接近
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxCast) {
  // 创建一个随机张量作为输入，形状为 [5, 3, 4, 2]，浮点数类型，使用默认设备
  torch::Tensor input = torch::rand(
      {5, 3, 4, 2},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 获取输入张量的维度
    int rank = input.dim();
    // 在每一个维度上执行带有数据类型转换的 log_softmax 操作
    for (int dim = -rank; dim < rank; ++dim) {
      // 计算输入张量在指定维度上的带有数据类型转换的 log_softmax 结果
      torch::Tensor output = torch::log_softmax(input, dim, torch::kDouble);
      // 在指定设备上计算输入张量在指定维度上的带有数据类型转换的 log_softmax 结果
      torch::Tensor lazy_output =
          torch::log_softmax(lazy_input, dim, torch::kDouble);
      // 检查两个张量的值是否在相对容差 1e-3 内全部接近
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxWrapper) {
  // 创建一个随机张量作为输入，形状为 [10, 2, 6, 4]，浮点数类型，使用默认设备
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 获取输入张量的维度
    int rank = input.dim();
    // 在每一个维度上执行带有参数的 log_softmax 操作
    for (int dim = -rank; dim < rank; ++dim) {
      // 计算输入张量在指定维度上的带有参数的 log_softmax 结果
      torch::Tensor output =
          torch::_log_softmax(input, dim, /*half_to_float=*/false);
      // 在指定设备上计算输入张量在指定维度上的带有参数的 log_softmax 结果
      torch::Tensor lazy_output =
          torch::_log_softmax(lazy_input, dim, /*half_to_float=*/false);
      // 检查两个张量的值是否在相对容差 1e-3 内全部接近
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmax) {
  // 创建一个随机张量作为输入，形状为 [10, 2, 6, 4]，浮点数类型，使用默认设备
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 获取输入张量的维度
    int rank = input.dim();
    // 在每一个维度上执行 softmax 操作
    for (int dim = -rank; dim < rank; ++dim) {
      // 计算输入张量在指定维度上的 softmax 结果
      torch::Tensor output = torch::softmax(input, dim);
      // 在指定设备上计算输入张量在指定维度上的 softmax 结果
      torch::Tensor lazy_output = torch::softmax(lazy_input, dim);
      // 检查两个张量的值是否在相对容差 1e-3 内全部接近
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmaxCast) {
  // 创建一个随机张量作为输入，形状为 [10, 2, 6, 4]，浮点数类型，使用默认设备
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 获取输入张量的维度
    int rank = input.dim();
    // 在每一个维度上执行带有数据类型转换的 softmax 操作
    for (int dim = -rank; dim < rank; ++dim) {
      // 计算输入张量在指定维度上的带有数据类型转换的 softmax 结果
      torch::Tensor output = torch::softmax(input, dim, torch::kDouble);
      // 在指定设备上计算输入张量在指定维度上的带有数据类型转换的 softmax 结果
      torch::Tensor lazy_output =
          torch::softmax(lazy_input, dim, torch::kDouble);
      // 检查两个张量的值是否在相对容差 1e-3 内全部接近
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}
TEST_F(LazyOpsTest, TestSoftmaxWrapper) {
  // 创建一个形状为 {10, 2, 6, 4} 的随机张量 input，数据类型为 float，设备为默认设备
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 获取 input 的维度数，并赋值给 rank
    int rank = input.dim();
    // 遍历维度从 -rank 到 rank-1
    for (int dim = -rank; dim < rank; ++dim) {
      // 在指定维度 dim 上应用 softmax 操作，half_to_float=false
      torch::Tensor output =
          torch::_softmax(input, dim, /*half_to_float=*/false);
      // 在 lazy_input 上应用相同的 softmax 操作，half_to_float=false
      torch::Tensor lazy_output =
          torch::_softmax(lazy_input, dim, /*half_to_float=*/false);
      // 检查 output 和 lazy_output 是否在相对误差 rtol=1e-3 内全部接近
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftplus) {
  // 创建一个形状为 {2, 1, 4, 6} 的随机张量 input，数据类型为 float，设备为默认设备
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在 input 上应用 softplus 操作，得到 output
  torch::Tensor output = torch::softplus(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在 lazy_input 上应用 softplus 操作，得到 lazy_output
    torch::Tensor lazy_output = torch::softplus(lazy_input);
    // 检查 output 和 lazy_output 是否在相对误差 rtol=1e-4 内全部接近
    AllClose(output, lazy_output, /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestMaxPool1D) {
  // 创建一个形状为 {1, 16, 56} 的随机张量 input，数据类型为 float，设备为默认设备
  torch::Tensor input = torch::rand(
      {1, 16, 56}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 kernel_size 为 3
  int kernel_size = 3;
  // 对于 stride 从 1 到 2 的每个值
  for (int stride = 1; stride <= 2; ++stride) {
    // 对于 padding 从 0 到 1 的每个值
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 互操作测试 dilation 从 1 到 2 的每个值
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 在 input 上应用 max_pool1d 操作，得到 output
          torch::Tensor output = torch::max_pool1d(
              input,
              /*kernel_size=*/{kernel_size},
              /*stride=*/{stride},
              /*padding=*/{padding},
              /*dilation=*/{dilation},
              /*ceil_mode=*/ceil_mode);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将 input 复制到指定设备上，得到 lazy_input
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在 lazy_input 上应用相同的 max_pool1d 操作，得到 lazy_output
            torch::Tensor lazy_output = torch::max_pool1d(
                lazy_input,
                /*kernel_size=*/{kernel_size},
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*dilation=*/{dilation},
                /*ceil_mode=*/ceil_mode);
            // 检查 output 和 lazy_output 是否全部接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2D) {
  // 创建一个形状为 {1, 4, 14, 14} 的随机张量 input，数据类型为 float，设备为默认设备
  torch::Tensor input = torch::rand(
      {1, 4, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义 kernel_size 为 3
  int kernel_size = 3;
  // 对于 stride 从 1 到 2 的每个值
  for (int stride = 1; stride <= 2; ++stride) {
    // 使用 padding 变量进行循环，测试两种情况：0 和 1
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true 的情况
      for (bool ceil_mode : {false, true}) {
        // 使用 dilation 变量进行循环，测试两种情况：1 和 2
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 使用 torch::max_pool2d 函数进行最大池化操作，生成输出张量
          torch::Tensor output = torch::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},  // 池化核大小
              /*stride=*/{stride, stride},                  // 步幅
              /*padding=*/{padding, padding},              // 填充
              /*dilation=*/{dilation, dilation},           // 扩展
              /*ceil_mode=*/ceil_mode);                    // 是否启用 ceil_mode

          // 对每个设备执行操作，复制输入张量到设备并进行最大池化操作
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);  // 将输入复制到指定设备
            torch::Tensor lazy_output = torch::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},  // 池化核大小
                /*stride=*/{stride, stride},                  // 步幅
                /*padding=*/{padding, padding},              // 填充
                /*dilation=*/{dilation, dilation},           // 扩展
                /*ceil_mode=*/ceil_mode);                    // 是否启用 ceil_mode
            AllClose(output, lazy_output);  // 检查输出张量是否相似
          });
        }
      }
    }
}

TEST_F(LazyOpsTest, TestMaxPool2DWithIndices) {
  // 创建一个形状为 [1, 4, 14, 14] 的随机张量作为输入，使用默认设备（CPU 或 GPU）
  torch::Tensor input = torch::rand(
      {1, 4, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义池化核大小为 3
  int kernel_size = 3;
  // 遍历不同的步幅（stride）
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历不同的填充（padding）
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 交互测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 交互测试不同的扩展（dilation）
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 调用 max_pool2d_with_indices 函数进行池化操作，并获取输出张量和索引
          auto outputs = torch::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 针对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上调用 max_pool2d_with_indices 函数进行池化操作
            auto lazy_outputs = torch::max_pool2d_with_indices(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            // 验证两个张量是否全部接近
            AllClose(std::get<0>(outputs), std::get<0>(lazy_outputs));
            // 验证两个索引张量是否全部接近
            AllClose(std::get<1>(outputs), std::get<1>(lazy_outputs));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DNonSquare) {
  // 创建一个形状为 [1, 4, 14, 14] 的随机张量作为输入，使用默认设备（CPU 或 GPU）
  torch::Tensor input = torch::rand(
      {1, 4, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义池化核大小为 4
  int kernel_size = 4;
  // 遍历不同的步幅（stride）
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历不同的填充（padding）
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 交互测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 交互测试不同的扩展（dilation）
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 调用 max_pool2d 函数进行池化操作，并获取输出张量
          torch::Tensor output = torch::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 针对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上调用 max_pool2d 函数进行池化操作
            torch::Tensor lazy_output = torch::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            // 验证两个张量是否全部接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}
TEST_F(LazyOpsTest, TestMaxPool3D) {
  // 创建一个大小为[1, 1, 8, 8, 8]的随机张量作为输入，使用默认设备
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 最大池化操作的核大小设为3
  int kernel_size = 3;
  // 循环遍历步长为1和2的情况
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历填充为0和1的情况
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过CPU互操作测试 ceil_mode=true 的情况
      for (bool ceil_mode : {false, true}) {
        // 通过CPU互操作测试膨胀率为1和2的情况
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 执行3D最大池化操作，生成输出张量
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行3D最大池化操作，生成惰性计算的输出张量
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            // 验证计算结果是否相等
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DWithIndices) {
  // 创建一个大小为[1, 1, 8, 8, 8]的随机张量作为输入，使用默认设备
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 最大池化操作的核大小设为3
  int kernel_size = 3;
  // 循环遍历步长为1和2的情况
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历填充为0和1的情况
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过CPU互操作测试 ceil_mode=true 的情况
      for (bool ceil_mode : {false, true}) {
        // 通过CPU互操作测试膨胀率为1和2的情况
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 执行3D最大池化操作，返回输出张量和索引张量的元组
          auto outputs = torch::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行3D最大池化操作，返回惰性计算的输出张量和索引张量的元组
            auto lazy_outputs = torch::max_pool3d_with_indices(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);

            // 验证输出张量是否相等
            AllClose(std::get<0>(outputs), std::get<0>(lazy_outputs));
            // 验证索引张量是否相等
            AllClose(std::get<1>(outputs), std::get<1>(lazy_outputs));
          });
        }
      }
    }
  }
}
// 在 LazyOpsTest 测试类中的 TestMaxPool3DIncompleteAttributes 测试函数
TEST_F(LazyOpsTest, TestMaxPool3DIncompleteAttributes) {
  // 创建一个形状为 {1, 1, 8, 8, 8} 的随机浮点数张量作为输入
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置卷积核大小为 3
  int kernel_size = 3;
  // 迭代两次，stride 取值为 1 和 2
  for (int stride = 1; stride <= 2; ++stride) {
    // 迭代两次，padding 取值为 0 和 1
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 互操作测试 dilation 取值为 1 和 2
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 使用 torch::max_pool3d 函数计算最大池化结果，输出张量 output
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上计算最大池化结果，输出张量 lazy_output
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            // 检查输出结果是否接近预期结果 output
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

// 在 LazyOpsTest 测试类中的 TestMaxPool3DNonSquare 测试函数
TEST_F(LazyOpsTest, TestMaxPool3DNonSquare) {
  // 创建一个形状为 {1, 1, 8, 8, 8} 的随机浮点数张量作为输入
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置卷积核大小为 4
  int kernel_size = 4;
  // 迭代两次，stride 取值为 1 和 2
  for (int stride = 1; stride <= 2; ++stride) {
    // 迭代两次，padding 取值为 0 和 1
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 互操作测试 dilation 取值为 1 和 2
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 使用 torch::max_pool3d 函数计算最大池化结果，输出张量 output
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上计算最大池化结果，输出张量 lazy_output
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            // 检查输出结果是否接近预期结果 output
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}
TEST_F(LazyOpsTest, TestMaxPool2DNoBatch) {
  // 创建一个大小为 [4, 14, 14] 的随机张量作为输入，使用默认设备
  torch::Tensor input = torch::rand(
      {4, 14, 14}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置池化操作的核大小为 3
  int kernel_size = 3;
  // 遍历步长为 1 和 2 的情况
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历填充大小为 0 和 1 的情况
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 交互测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 交互测试扩张（dilation）
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 对输入进行二维最大池化操作
          torch::Tensor output = torch::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 对复制的输入张量执行二维最大池化操作
            torch::Tensor lazy_output = torch::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            // 检查输出张量和延迟操作的输出张量是否接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DNoBatch) {
  // 创建一个大小为 [1, 8, 8, 8] 的随机张量作为输入，使用默认设备
  torch::Tensor input = torch::rand(
      {1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 设置池化操作的核大小为 3
  int kernel_size = 3;
  // 遍历步长为 1 和 2 的情况
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历填充大小为 0 和 1 的情况
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 交互测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 交互测试扩张（dilation）
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 对输入进行三维最大池化操作
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 对复制的输入张量执行三维最大池化操作
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            // 检查输出张量和延迟操作的输出张量是否接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}
TEST_F(LazyOpsTest, TestAvgPool1D) {
  // 创建大小为 [4, 1, 28] 的随机张量作为输入，使用默认设备并指定为浮点类型
  torch::Tensor input = torch::rand(
      {4, 1, 28}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义池化核大小为 2
  int kernel_size = 2;
  // 循环遍历不同的步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历不同的填充值
    for (int padding = 0; padding <= 1; ++padding) {
      // 遍历是否包含填充的选项
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 互操作测试 ceil_mode=true
        for (bool ceil_mode : {false, true}) {
          // 执行一维平均池化操作，生成输出张量
          torch::Tensor output = torch::avg_pool1d(
              input,
              /*kernel_size=*/{kernel_size},
              /*stride=*/{stride},
              /*padding=*/{padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          // 对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行一维平均池化操作，生成懒惰计算的输出张量
            torch::Tensor lazy_output = torch::avg_pool1d(
                lazy_input,
                /*kernel_size=*/{kernel_size},
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 检查懒惰计算的输出与原始输出是否近似相等
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2D) {
  // 创建大小为 [2, 1, 14, 14] 的随机张量作为输入，使用默认设备并指定为浮点类型
  torch::Tensor input = torch::rand(
      {2, 1, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义池化核大小为 2
  int kernel_size = 2;
  // 循环遍历不同的步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历不同的填充值
    for (int padding = 0; padding <= 1; ++padding) {
      // 遍历是否包含填充的选项
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 互操作测试 ceil_mode=true
        for (bool ceil_mode : {false, true}) {
          // 执行二维平均池化操作，生成输出张量
          torch::Tensor output = torch::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          // 对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行二维平均池化操作，生成懒惰计算的输出张量
            torch::Tensor lazy_output = torch::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 将懒惰计算的输出转移到 CPU 并检查是否近似相等
            AllClose(output, lazy_output.to(torch::kCPU));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNonSquare) {
  // 创建大小为 [2, 1, 14, 14] 的随机张量作为输入，使用默认设备并指定为浮点类型
  torch::Tensor input = torch::rand(
      {2, 1, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义池化核大小为 4
  int kernel_size = 4;
  // 循环遍历不同的步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历不同的填充值
    for (int padding = 0; padding <= 1; ++padding) {
      // 与之前的测试用例相似，但没有测试 ceil_mode=true
      for (bool count_include_pad : {true, false}) {
    // 外层循环，设置 padding 为 0 和 1 两种情况
    for (int padding = 0; padding <= 1; ++padding) {
      // 内层循环，遍历 count_include_pad 为 true 和 false 两种情况
      for (bool count_include_pad : {true, false}) {
        // 在 CPU 互操作中测试 ceil_mode=true 的情况
        // 再次循环，测试 ceil_mode 为 false 和 true 两种情况
        for (bool ceil_mode : {false, true}) {
          // 使用 torch::avg_pool2d 函数进行二维平均池化操作
          torch::Tensor output = torch::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入数据复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行二维平均池化操作
            torch::Tensor lazy_output = torch::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 检查两个输出张量是否在所有设备上都相等
            AllClose(output, lazy_output);
          });
        }
      }
    }
TEST_F(LazyOpsTest, TestAvgPool3D) {
  // 创建一个大小为 {1, 1, 7, 7, 7} 的随机张量作为输入，使用默认设备（默认为CPU）
  torch::Tensor input = torch::rand(
      {1, 1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 池化核大小设定为2
  int kernel_size = 2;
  // 对于步长从1到2的循环
  for (int stride = 1; stride <= 2; ++stride) {
    // 对于填充从0到1的循环
    for (int padding = 0; padding <= 1; ++padding) {
      // 对于包括填充的计数为true和false的循环
      for (bool count_include_pad : {true, false}) {
        // 通过CPU交互测试 ceil_mode=true 的情况
        for (bool ceil_mode : {false, true}) {
          // 使用 torch::avg_pool3d 函数进行三维平均池化操作
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上使用 torch::avg_pool3d 函数进行相同的池化操作
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 检查输出是否在所有设备上都接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DIncompleteAttributes) {
  // 创建一个大小为 {1, 1, 7, 7, 7} 的随机张量作为输入，使用默认设备（默认为CPU）
  torch::Tensor input = torch::rand(
      {1, 1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 池化核大小设定为2
  int kernel_size = 2;
  // 对于步长从1到2的循环
  for (int stride = 1; stride <= 2; ++stride) {
    // 对于填充从0到1的循环
    for (int padding = 0; padding <= 1; ++padding) {
      // 对于包括填充的计数为true和false的循环
      for (bool count_include_pad : {true, false}) {
        // 通过CPU交互测试 ceil_mode=true 的情况
        for (bool ceil_mode : {false, true}) {
          // 使用 torch::avg_pool3d 函数进行三维平均池化操作，但步长未指定（空列表）
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上使用 torch::avg_pool3d 函数进行相同的池化操作
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 检查输出是否在所有设备上都接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNonSquare) {
  // 创建一个大小为 {1, 1, 7, 7, 7} 的随机张量作为输入，使用默认设备（默认为CPU）
  torch::Tensor input = torch::rand(
      {1, 1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 池化核大小设定为4
  int kernel_size = 4;
  // 对于步长从1到2的循环
  for (int stride = 1; stride <= 2; ++stride) {
    // 这里继续进行其他循环操作，没有给出完整的代码段
    // 外层循环，设置填充值为0和1时的两次迭代
    for (int padding = 0; padding <= 1; ++padding) {
      // 内层循环，测试是否包含填充的两种情况
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 互操作测试 ceil_mode=true 的两种情况
        for (bool ceil_mode : {false, true}) {
          // 使用 torch 库进行三维平均池化操作
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入数据复制到指定设备
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行三维平均池化操作
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 检查两个输出张量是否全部相等
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }


这段代码嵌套了多层循环，对输入数据进行了不同配置的三维平均池化操作，并在不同的设备上执行了相同的操作，最后验证每次操作的输出是否一致。
TEST_F(LazyOpsTest, TestAvgPool2DNoBatch) {
  // 创建一个大小为 (1, 7, 7) 的随机张量作为输入，并指定为默认设备上的浮点张量
  torch::Tensor input = torch::rand(
      {1, 7, 7}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义池化操作的核大小
  int kernel_size = 2;
  // 遍历不同的步长值
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历不同的填充值
    for (int padding = 0; padding <= 1; ++padding) {
      // 遍历是否包含填充值的选项
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 互操作测试 ceil_mode=true 的情况
        for (bool ceil_mode : {false, true}) {
          // 执行平均池化操作，根据给定的参数
          torch::Tensor output = torch::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          // 对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行延迟评估的平均池化操作
            torch::Tensor lazy_output = torch::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 验证输出结果是否全部接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNoBatch) {
  // 创建一个大小为 (1, 7, 7, 7) 的随机张量作为输入，并指定为默认设备上的浮点张量
  torch::Tensor input = torch::rand(
      {1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义池化操作的核大小
  int kernel_size = 2;
  // 遍历不同的步长值
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历不同的填充值
    for (int padding = 0; padding <= 1; ++padding) {
      // 遍历是否包含填充值的选项
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 互操作测试 ceil_mode=true 的情况
        for (bool ceil_mode : {false, true}) {
          // 执行 3D 平均池化操作，根据给定的参数
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          // 对每个设备执行操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输入张量复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行延迟评估的 3D 平均池化操作
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            // 验证输出结果是否全部接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2D) {
  // 创建一个大小为 (4, 1, 28, 28) 的随机张量作为输入，并指定为默认设备上的浮点张量
  torch::Tensor input = torch::rand(
      {4, 1, 28, 28},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历不同的输出大小
  for (int64_t output_size : {7, 4}) {
    // 执行自适应平均池化操作，根据给定的输出大小
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    # 对于每个设备，执行以下操作
    ForEachDevice([&](const torch::Device& device) {
        # 将输入数据复制到指定设备上，返回一个惰性加载的张量
        torch::Tensor lazy_input = CopyToDevice(input, device);
        # 对复制到设备上的输入张量进行自适应平均池化，输出的大小为 (output_size, output_size)
        torch::Tensor lazy_output =
            torch::adaptive_avg_pool2d(lazy_input, {output_size, output_size});
        # 检查输出张量和预期输出是否全部接近
        AllClose(output, lazy_output);
    });
  }


这段代码的主要作用是针对每个设备执行一系列操作，包括将输入数据复制到设备上，然后对复制的数据进行自适应平均池化操作，并检查操作后的输出是否和预期输出相近。
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3D) {
  // 创建一个随机的输入张量，形状为 [9, 4, 56, 28, 28]
  torch::Tensor input = torch::rand(
      {9, 4, 56, 28, 28},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于每个输出尺寸，执行自适应三维平均池化操作
  for (int64_t output_size : {7, 4}) {
    // 执行自适应三维平均池化，生成输出张量
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    // 针对每个设备，将输入张量复制到设备上，并执行相同的池化操作，检验结果是否一致
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::adaptive_avg_pool3d(
          lazy_input, {output_size, output_size, output_size});
      AllClose(output, lazy_output);  // 检验两个输出张量是否近似相等
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DNoBatch) {
  // 创建一个随机的输入张量，形状为 [3, 56, 28, 28]
  torch::Tensor input = torch::rand(
      {3, 56, 28, 28},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于每个输出尺寸，执行自适应三维平均池化操作
  for (int64_t output_size : {7, 4}) {
    // 执行自适应三维平均池化，生成输出张量
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    // 针对每个设备，将输入张量复制到设备上，并执行相同的池化操作，检验结果是否一致
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::adaptive_avg_pool3d(
          lazy_input, {output_size, output_size, output_size});
      AllClose(output, lazy_output);  // 检验两个输出张量是否近似相等
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DNoBatch) {
  // 创建一个随机的输入张量，形状为 [1, 56, 56]
  torch::Tensor input = torch::rand(
      {1, 56, 56}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于每个输出尺寸，执行自适应二维平均池化操作
  for (int64_t output_size : {7, 8}) {
    // 执行自适应二维平均池化，生成输出张量
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    // 针对每个设备，将输入张量复制到设备上，并执行相同的池化操作，检验结果是否一致
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output =
          torch::adaptive_avg_pool2d(lazy_input, {output_size, output_size});
      AllClose(output, lazy_output);  // 检验两个输出张量是否近似相等
    });
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool2D) {
  int kernel_size = 2;
  // 创建一个随机的输入张量，形状为 [2, 2, 8, 8]
  torch::Tensor input = torch::rand(
      {2, 2, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对于每个步幅大小，执行最大反池化二维操作
  for (int stride = 1; stride <= 2; ++stride) {
    // 在每次迭代中，执行最大反池化二维操作

    torch::Tensor output = torch::max_unpool2d(input, {kernel_size, kernel_size}, {stride, stride});
    // 循环遍历填充值 padding，测试 ceil_mode=true 的 CPU 互操作性能
    for (int padding = 0; padding <= 1; ++padding) {
        // 通过 CPU 互操作测试 ceil_mode=false 和 ceil_mode=true 两种情况
        for (bool ceil_mode : {false, true}) {
            // 循环遍历膨胀值 dilation，测试其对应的 CPU 互操作性能
            for (int dilation = 1; dilation <= 2; ++dilation) {
                // 声明张量 output 和 indices
                torch::Tensor output;
                torch::Tensor indices;
                // 调用 max_pool2d_with_indices 函数执行最大池化操作，并返回 output 和 indices
                std::tie(output, indices) = torch::max_pool2d_with_indices(
                    input,
                    /*kernel_size=*/{kernel_size, kernel_size},
                    /*stride=*/{stride, stride},
                    /*padding=*/{padding, padding},
                    /*dilation=*/{dilation, dilation},
                    /*ceil_mode=*/ceil_mode);

                // 创建输出张量大小的向量 output_size
                std::vector<int64_t> output_size({input.size(2), input.size(3)});
                // 调用 max_unpool2d 函数，返回未池化的张量 utensor
                at::Tensor utensor =
                    torch::max_unpool2d(output, indices, output_size);

                // 针对每个设备执行以下操作
                ForEachDevice([&](const torch::Device& device) {
                    // 将 output 复制到指定设备上，并创建 lazy_output 张量
                    torch::Tensor lazy_output = CopyToDevice(output, device);
                    // 将 indices 复制到指定设备上，并创建 lazy_indices 张量
                    torch::Tensor lazy_indices = CopyToDevice(indices, device);
                    // 在指定设备上调用 max_unpool2d 函数，返回 lazy_utensor
                    at::Tensor lazy_utensor =
                        torch::max_unpool2d(lazy_output, lazy_indices, output_size);
                    // 检查 utensor 和 lazy_utensor 是否近似相等
                    AllClose(utensor, lazy_utensor);
                });
            }
        }
    }
}

TEST_F(LazyOpsTest, TestMaxUnpool3D) {
  // 定义最大池化的核大小为2
  int kernel_size = 2;
  // 生成一个随机的5维张量作为输入
  torch::Tensor input = torch::rand(
      {1, 1, 4, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 循环遍历不同的步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历不同的填充方式
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 交互测试 ceil_mode=true 的情况
      for (bool ceil_mode : {false, true}) {
        // 通过 CPU 交互测试不同的膨胀（dilation）值
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 声明输出张量和索引张量
          torch::Tensor output;
          torch::Tensor indices;
          // 执行带索引的3维最大池化操作
          std::tie(output, indices) = torch::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          // 构建输出张量的尺寸
          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          // 执行3维最大反池化操作
          at::Tensor utensor = torch::max_unpool3d(
              output,
              indices,
              output_size,
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding});

          // 对每个设备执行特定操作
          ForEachDevice([&](const torch::Device& device) {
            // 将输出和索引张量复制到指定设备
            torch::Tensor lazy_output = CopyToDevice(output, device);
            torch::Tensor lazy_indices = CopyToDevice(indices, device);
            // 在指定设备上执行最大反池化操作
            at::Tensor lazy_utensor = torch::max_unpool3d(
                lazy_output,
                lazy_indices,
                output_size,
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding});
            // 检查两个张量是否全部接近
            AllClose(utensor, lazy_utensor);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss) {
  // TODO(whc) debug divide-by-zero failure under ASAN
  // 跳过当前测试，因为存在 ASAN 下的除零错误
  GTEST_SKIP();

  // 定义批大小为6，类别数为2
  int batch = 6;
  int classes = 2;
  // 循环遍历数据类型，目前只处理 torch::kFloat 类型
  for (auto dtype : {torch::kFloat}) {
    // 遍历忽略索引集合 {-1, 0, 1, 5}
    for (int ignore_index : {-1, 0, 1, 5}) {
      // 遍历是否使用默认权重的布尔值 {false, true}
      for (bool def_weight : {false, true}) {
        // 创建随机数填充的张量作为输入数据
        torch::Tensor input = torch::rand(
            {batch, classes},
            torch::TensorOptions(dtype).device(DefaultDevice()));
        // 创建随机整数填充的张量作为目标数据
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0),
            classes,
            {batch},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        // 定义权重张量
        torch::Tensor weight;
        // 如果使用默认权重，则创建随机数填充的权重张量
        if (def_weight) {
          weight = torch::rand(
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));
        }
        // 遍历损失函数的减少方式 {平均，总和，无}
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean,
              torch::Reduction::Sum,
              torch::Reduction::None}) {
          // 计算负对数似然损失函数
          torch::Tensor output = torch::nll_loss(
              /*self=*/input,
              /*target=*/target,
              /*weight=*/weight,
              /*reduction=*/reduction,
              /*ignore_index=*/ignore_index);

          // 针对每个设备，将输入数据、目标数据和权重数据复制到指定设备上，并计算相应的损失
          ForEachDevice([&](const torch::Device& device) {
            // 将输入数据复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 将目标数据复制到指定设备上
            torch::Tensor lazy_target = CopyToDevice(target, device);
            // 如果使用默认权重，则将权重数据复制到指定设备上；否则创建一个空张量
            torch::Tensor lazy_weight =
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();
            // 在指定设备上计算负对数似然损失函数
            torch::Tensor lazy_output = torch::nll_loss(
                /*self=*/lazy_input,
                /*target=*/lazy_target,
                /*weight=*/lazy_weight,
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
            // 验证两种计算得到的损失是否接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss2d) {
  int batch = 6;  // 设置批次大小为6
  int classes = 2;  // 设置类别数为2
  int height = 3;  // 设置高度为3
  int width = 3;  // 设置宽度为3
  // TODO(asuhan): Fix the torch::kDouble case.  // 待办事项：修复 torch::kDouble 情况
  for (auto dtype : {torch::kFloat}) {  // 迭代浮点数类型
    for (int ignore_index : {-1, 0, 1, 5}) {  // 迭代忽略索引 -1, 0, 1, 5
      for (bool def_weight : {false, true}) {  // 迭代默认权重为假或真
        torch::Tensor input = torch::rand(  // 生成随机张量作为输入
            {batch, classes, height, width},  // 维度为 [批次大小, 类别数, 高度, 宽度]
            torch::TensorOptions(dtype).device(DefaultDevice()));  // 设置张量选项和默认设备
        torch::Tensor target = torch::randint(  // 生成随机整数张量作为目标
            std::min(ignore_index, 0),  // 最小值为忽略索引和0的最小值
            classes,  // 类别数
            {batch, height, width},  // 维度为 [批次大小, 高度, 宽度]
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));  // 设置张量选项和默认设备为长整型
        torch::Tensor weight;  // 权重张量
        if (def_weight) {  // 如果定义了权重
          weight = torch::rand(  // 生成随机张量作为权重
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));  // 维度为 [类别数], 设置张量选项和默认设备
        }
        for (torch::Reduction::Reduction reduction :  // 迭代减少操作方式
             {torch::Reduction::Mean,  // 均值
              torch::Reduction::Sum,  // 和
              torch::Reduction::None}) {  // 无
          torch::Tensor output = torch::nll_loss2d(  // 计算二维负对数似然损失
              /*self=*/input,  // 输入张量
              /*target=*/target,  // 目标张量
              /*weight=*/weight,  // 权重张量
              /*reduction=*/reduction,  // 减少操作方式
              /*ignore_index=*/ignore_index);  // 忽略索引

          ForEachDevice([&](const torch::Device& device) {  // 对于每个设备执行操作
            torch::Tensor lazy_input = CopyToDevice(input, device);  // 将输入复制到指定设备
            torch::Tensor lazy_target = CopyToDevice(target, device);  // 将目标复制到指定设备
            torch::Tensor lazy_weight =  // 惰性权重张量
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();  // 如果定义了权重，则复制到指定设备；否则为空张量
            torch::Tensor lazy_output = torch::nll_loss2d(  // 计算二维负对数似然损失（惰性）
                /*self=*/lazy_input,  // 惰性输入张量
                /*target=*/lazy_target,  // 惰性目标张量
                /*weight=*/lazy_weight,  // 惰性权重张量
                /*reduction=*/reduction,  // 减少操作方式
                /*ignore_index=*/ignore_index);  // 忽略索引
            AllClose(output, lazy_output);  // 检查输出是否接近
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSmoothL1Loss) {
  torch::Tensor input = torch::randn(  // 生成随机正态分布张量作为输入
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));  // 维度为 [2, 4], 设置张量选项和默认设备为浮点数
  torch::Tensor target = torch::randn(  // 生成随机正态分布张量作为目标
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));  // 维度为 [2, 4], 设置张量选项和默认设备为浮点数
  for (torch::Reduction::Reduction reduction :  // 迭代减少操作方式
       {torch::Reduction::None,  // 无
        torch::Reduction::Mean,  // 均值
        torch::Reduction::Sum}) {  // 和
    for (double beta : {0.25, 1.}) {  // 迭代 beta 值
      torch::Tensor output =  // 平滑 L1 损失
          torch::smooth_l1_loss(input, target, reduction, beta);

      ForEachDevice([&](const torch::Device& device) {  // 对于每个设备执行操作
        torch::Tensor lazy_input = CopyToDevice(input, device);  // 将输入复制到指定设备
        torch::Tensor lazy_target = CopyToDevice(target, device);  // 将目标复制到指定设备
        torch::Tensor lazy_output =  // 惰性输出
            torch::smooth_l1_loss(lazy_input, lazy_target, reduction, beta);  // 计算平滑 L1 损失（惰性）
        AllClose(output, lazy_output);  // 检查输出是否接近
      });
    }
  }
}
TEST_F(LazyOpsTest, TestL1Loss) {
  // 创建一个大小为 [2, 4] 的随机张量 input，设备为默认设备
  torch::Tensor input = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 [2, 4] 的随机张量 target，设备为默认设备
  torch::Tensor target = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对不同的减少方式（None、Mean、Sum），计算 L1 损失
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    // 计算 L1 损失
    torch::Tensor output = torch::l1_loss(input, target, reduction);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 将 target 复制到指定设备上
      torch::Tensor lazy_target = CopyToDevice(target, device);
      // 在指定设备上计算 L1 损失
      torch::Tensor lazy_output =
          torch::l1_loss(lazy_input, lazy_target, reduction);
      // 检查输出是否全部相等
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestL1LossBackward) {
  // 针对不同的减少方式（None、Mean、Sum），执行测试函数
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    // 定义测试函数，计算 L1 损失
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::l1_loss(inputs[0], inputs[1], reduction);
    };
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 执行反向传播测试
      TestBackward(
          {torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat).device(DefaultDevice()))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestMseLoss) {
  // 创建一个大小为 [2, 4] 的随机张量 input，设备为默认设备
  torch::Tensor input = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个大小为 [2, 4] 的随机张量 target，设备为默认设备
  torch::Tensor target = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对不同的减少方式（None、Mean、Sum），计算 MSE 损失
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    // 计算 MSE 损失
    torch::Tensor output = torch::mse_loss(input, target, reduction);
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 将 target 复制到指定设备上
      torch::Tensor lazy_target = CopyToDevice(target, device);
      // 在指定设备上计算 MSE 损失
      torch::Tensor lazy_output =
          torch::mse_loss(lazy_input, lazy_target, reduction);
      // 检查输出是否全部相等
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestMseLossBackward) {
  // 针对不同的减少方式（None、Mean、Sum），执行测试函数
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    // 定义测试函数，计算 MSE 损失
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::mse_loss(inputs[0], inputs[1], reduction);
    };
    // 对每个设备执行操作
    ForEachDevice([&](const torch::Device& device) {
      // 执行反向传播测试
      TestBackward(
          {torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat).device(DefaultDevice()))},
          device,
          testfn);
    });
  }
}
    # 对每个设备执行操作，使用 lambda 表达式捕获设备变量
    ForEachDevice([&](const torch::Device& device) {
      # 调用 TestBackward 函数，测试反向传播功能
      TestBackward(
          # 提供两个随机生成的张量作为输入
          {torch::rand(
               {2, 4},  # 张量形状为 2x4
               torch::TensorOptions(torch::kFloat)  # 使用单精度浮点数选项
                   .device(DefaultDevice())  # 指定默认设备
                   .requires_grad(true)),  # 允许计算梯度
           torch::rand(
               {2, 4},  # 另一个张量形状为 2x4
               torch::TensorOptions(torch::kFloat)  # 使用单精度浮点数选项
                   .device(DefaultDevice()))},  # 使用默认设备
          device,  # 传递当前设备作为参数
          testfn);  # 传递测试函数作为参数
    });
  }
TEST_F(LazyOpsTest, TestBatchNorm1D) {
  // 定义批量归一化操作的测试函数，使用 Google Test 的测试框架

  int num_features = 3;
  // 定义特征数为3

  torch::Tensor input = torch::rand(
      {2, num_features, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 生成一个形状为[2, num_features, 4]的随机浮点数张量作为输入，使用默认设备

  torch::Tensor weight = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 生成一个长度为num_features的随机浮点数张量作为权重，使用默认设备

  torch::Tensor bias = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 生成一个长度为num_features的随机浮点数张量作为偏置，使用默认设备

  torch::Tensor running_mean = torch::zeros(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 生成一个长度为num_features的零张量作为运行时均值，使用默认设备

  torch::Tensor running_var = torch::ones(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 生成一个长度为num_features的全1张量作为运行时方差，使用默认设备

  double momentum = 0.1;
  // 定义动量为0.1

  double eps = 0.5;
  // 定义ε为0.5

  torch::Tensor undef;
  // 定义一个未定义的张量

  for (bool training : {true, false}) {
    // 遍历训练标志true和false
    for (bool undef_weight_bias : {false, true}) {
      // 遍历未定义权重和偏置标志false和true

      torch::Tensor output = torch::batch_norm(
          /*input=*/input,
          /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean,
          /*running_var=*/running_var,
          /*training=*/training,
          /*momentum=*/momentum,
          /*eps=*/eps,
          /*cudnn_enabled=*/false);
      // 执行批量归一化操作，根据未定义权重和偏置标志选择使用未定义张量或指定的权重和偏置

      ForEachDevice([&](const torch::Device& device) {
        // 对于每个设备执行以下操作
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 将输入复制到指定设备上形成延迟加载的输入张量
        torch::Tensor lazy_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        // 根据未定义权重和偏置标志选择复制未定义张量或指定的权重到指定设备上形成延迟加载的权重张量
        torch::Tensor lazy_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        // 根据未定义权重和偏置标志选择复制未定义张量或指定的偏置到指定设备上形成延迟加载的偏置张量
        torch::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
        // 将运行时均值复制到指定设备上形成延迟加载的运行时均值张量
        torch::Tensor lazy_running_var = CopyToDevice(running_var, device);
        // 将运行时方差复制到指定设备上形成延迟加载的运行时方差张量
        torch::Tensor lazy_output = torch::batch_norm(
            /*input=*/lazy_input,
            /*weight=*/lazy_weight,
            /*bias=*/lazy_bias,
            /*running_mean=*/lazy_running_mean,
            /*running_var=*/lazy_running_var,
            /*training=*/training,
            /*momentum=*/momentum,
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
        // 在指定设备上执行批量归一化操作，形成延迟加载的输出张量
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
        // 检查批量归一化的输出与延迟加载的输出张量是否在给定的相对容差和绝对容差范围内相等
      });
    }
  }
}
TEST_F(LazyOpsTest, TestBatchNorm2D) {
  // 定义特征数量
  int num_features = 3;
  // 创建随机输入张量，形状为 [2, num_features, 4, 4]
  torch::Tensor input = torch::rand(
      {2, num_features, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机权重张量，形状为 [num_features]
  torch::Tensor weight = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机偏置张量，形状为 [num_features]
  torch::Tensor bias = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建全零的运行均值张量，形状为 [num_features]
  torch::Tensor running_mean = torch::zeros(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建全一的运行方差张量，形状为 [num_features]
  torch::Tensor running_var = torch::ones(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义动量和 epsilon 值
  double momentum = 0.1;
  double eps = 0.5;
  // 未定义的张量
  torch::Tensor undef;
  // 对训练和非训练模式分别进行迭代
  for (bool training : {true, false}) {
    // 对权重和偏置是否未定义分别进行迭代
    for (bool undef_weight_bias : {false, true}) {
      // 执行批量归一化操作，生成输出张量
      torch::Tensor output = torch::batch_norm(
          /*input=*/input,
          /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean,
          /*running_var=*/running_var,
          /*training=*/training,
          /*momentum=*/momentum,
          /*eps=*/eps,
          /*cudnn_enabled=*/false);
      // 针对每个设备执行操作
      ForEachDevice([&](const torch::Device& device) {
        // 将输入张量复制到指定设备，创建惰性输入张量
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 如果未定义权重和偏置，则复制未定义张量到设备，否则复制权重和偏置到设备
        torch::Tensor lazy_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        // 复制运行均值和方差到指定设备
        torch::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
        torch::Tensor lazy_running_var = CopyToDevice(running_var, device);
        // 执行批量归一化操作，生成惰性输出张量
        torch::Tensor lazy_output = torch::batch_norm(
            /*input=*/lazy_input,
            /*weight=*/lazy_weight,
            /*bias=*/lazy_bias,
            /*running_mean=*/lazy_running_mean,
            /*running_var=*/lazy_running_var,
            /*training=*/training,
            /*momentum=*/momentum,
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
        // 检查输出张量和惰性输出张量是否相近
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDim) {
  // 创建随机输入张量，形状为 [2, 3]
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备，创建惰性输入张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 检查输入张量和惰性输入张量的维度是否相等
    EXPECT_EQ(input.dim(), lazy_input.dim());
  });
}

TEST_F(LazyOpsTest, TestContiguous) {
  // 创建随机输入张量，形状为 [2, 3]
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 执行 contiguous 操作，生成输出张量
  torch::Tensor output = torch::native::contiguous(input);
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备，创建惰性输入张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 执行 contiguous 操作，生成惰性输出张量
    torch::Tensor lazy_output = torch::native::contiguous(lazy_input);
    // 检查输出张量和惰性输出张量是否相近
    AllClose(output, lazy_output);
  });
}
TEST_F(LazyOpsTest, TestSqueezeAll) {
  // 创建一个形状为 {2, 1, 3, 1} 的随机张量 input，使用默认设备的浮点选项
  torch::Tensor input = torch::rand(
      {2, 1, 3, 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对 input 进行 squeeze 操作，去除所有维度为 1 的维度
  torch::Tensor output = torch::squeeze(input);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 进行 squeeze 操作
    torch::Tensor lazy_output = torch::squeeze(lazy_input);
    // 检查输出和 lazy 输出是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSqueezeAllInPlace) {
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 {2, 1, 3, 1} 的随机张量 input，使用默认设备的浮点选项
    torch::Tensor input = torch::rand(
        {2, 1, 3, 1},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将 input 复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在原地对 input 进行 squeeze 操作
    torch::Tensor output = input.squeeze_();
    // 在原地对 lazy_input 进行 squeeze 操作
    torch::Tensor lazy_output = lazy_input.squeeze_();
    // 检查输出和 lazy 输出是否近似相等
    AllClose(output, lazy_output);
    // 检查 input 和 lazy_input 是否近似相等
    AllClose(input, lazy_input);
    // 断言 input 和 lazy_input 的维度相同
    ASSERT_EQ(input.dim(), lazy_input.dim());
    // 遍历每个维度索引，断言 input 和 lazy_input 的每个维度大小相同
    for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
      ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
    }
  });
}

TEST_F(LazyOpsTest, TestSqueezeOne) {
  // 创建一个形状为 {2, 1, 3, 1} 的随机张量 input，使用默认设备的浮点选项
  torch::Tensor input = torch::rand(
      {2, 1, 3, 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取输入张量的秩（维度数）
  int rank = input.dim();
  // 遍历所有可能的维度进行 squeeze 操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 在指定维度 dim 上对 input 进行 squeeze 操作
    torch::Tensor output = torch::squeeze(input, dim);
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定维度 dim 上对 lazy_input 进行 squeeze 操作
      torch::Tensor lazy_output = torch::squeeze(lazy_input, dim);
      // 检查输出和 lazy 输出是否近似相等
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestSqueezeOneInPlace) {
  // 定义张量的秩（维度数）
  int rank = 4;
  // 遍历所有可能的维度进行 squeeze 操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 创建一个形状为 {2, 1, 3, 1} 的随机张量 input，使用默认设备的浮点选项
      torch::Tensor input = torch::rand(
          {2, 1, 3, 1},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在原地在指定维度 dim 上对 input 进行 squeeze 操作
      torch::Tensor output = input.squeeze_(dim);
      // 在原地在指定维度 dim 上对 lazy_input 进行 squeeze 操作
      torch::Tensor lazy_output = lazy_input.squeeze_(dim);
      // 检查输出和 lazy 输出是否近似相等
      AllClose(output, lazy_output);
      // 检查 input 和 lazy_input 是否近似相等
      AllClose(input, lazy_input);
      // 断言 input 和 lazy_input 的维度相同
      ASSERT_EQ(input.dim(), lazy_input.dim());
      // 遍历每个维度索引，断言 input 和 lazy_input 的每个维度大小相同
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
      }
    });
  }
}

TEST_F(LazyOpsTest, TestUnsqueeze) {
  // 创建一个形状为 {2, 3} 的随机张量 input，使用默认设备的浮点选项
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算输入张量的秩（维度数）加一
  int rank = input.dim() + 1;
  // 遍历所有可能的维度进行 unsqueeze 操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 在指定维度 dim 上对 input 进行 unsqueeze 操作
    torch::Tensor output = torch::unsqueeze(input, dim);
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定维度 dim 上对 lazy_input 进行 unsqueeze 操作
      torch::Tensor lazy_output = torch::unsqueeze(lazy_input, dim);
      // 检查输出和 lazy 输出是否近似相等
      AllClose(output, lazy_output);
    });
  }
}
TEST_F(LazyOpsTest, TestUnsqueezeInPlace) {
  // 创建一个形状为 [2, 3] 的随机浮点数张量，并放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算张量的维度并加一，得到 rank
  int rank = input.dim() + 1;
  // 对每一个可能的维度进行循环操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上得到 lazy_input
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定维度 dim 上对 input 执行 unsqueeze 操作得到 output
      torch::Tensor output = input.unsqueeze_(dim);
      // 在指定维度 dim 上对 lazy_input 执行 unsqueeze 操作得到 lazy_output
      torch::Tensor lazy_output = lazy_input.unsqueeze_(dim);
      // 验证 output 和 lazy_output 是否接近
      AllClose(output, lazy_output);
      // 验证 input 和 lazy_input 是否接近
      AllClose(input, lazy_input);
      // 断言 input 和 lazy_input 的维度相同
      ASSERT_EQ(input.dim(), lazy_input.dim());
      // 遍历 input 和 lazy_input 的每一个维度，断言它们的尺寸相同
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
      }
    });
  }
}

TEST_F(LazyOpsTest, TestMaskedFill) {
  // 创建一个形状为 [2, 3] 的随机浮点数张量，并放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 [2, 3] 的随机布尔掩码张量，并放置在默认设备上
  torch::Tensor mask = torch::randint(
      0, 2, {2, 3}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  // 创建一个标量值为 42
  torch::Scalar value(42);
  // 使用掩码 mask 在 input 上进行 masked_fill 操作，得到结果张量 result
  torch::Tensor result = torch::masked_fill(input, mask, value);
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 和 mask 复制到指定设备上得到 lazy_input 和 lazy_mask
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_mask = CopyToDevice(mask, device);
    // 在指定设备上使用 lazy_mask 在 lazy_input 上进行 masked_fill 操作，得到 lazy_result
    torch::Tensor lazy_result =
        torch::masked_fill(lazy_input, lazy_mask, value);
    // 验证 result 和 lazy_result 是否接近
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestMaskedFillInPlace) {
  // 创建一个标量值为 42
  torch::Scalar value(42);
  // 创建一个形状为 [2, 3] 的随机布尔掩码张量，并放置在默认设备上
  torch::Tensor mask = torch::randint(
      0, 2, {2, 3}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为 [2, 3] 的随机浮点数张量，并放置在默认设备上
    torch::Tensor input = torch::rand(
        {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将 input 复制到指定设备上得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 将 mask 复制到指定设备上得到 lazy_mask
    torch::Tensor lazy_mask = CopyToDevice(mask, device);
    // 在指定设备上使用 mask 在 input 上进行 masked_fill_ 操作，得到结果张量 result
    torch::Tensor result = input.masked_fill_(mask, value);
    // 在指定设备上使用 lazy_mask 在 lazy_input 上进行 masked_fill_ 操作，得到 lazy_result
    torch::Tensor lazy_result = lazy_input.masked_fill_(lazy_mask, value);
    // 验证 result 和 lazy_result 是否接近
    AllClose(result, lazy_result);
    // 验证 input 和 lazy_input 是否接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestMaskedFillBroadcast) {
  // 创建一个形状为 [2, 5, 4, 3] 的随机浮点数张量，并放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 5, 4, 3},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个形状为 [4, 1] 的随机布尔掩码张量，并放置在默认设备上
  torch::Tensor mask = torch::randint(
      0, 2, {4, 1}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  // 创建一个标量值为 42
  torch::Scalar value(42);
  // 使用掩码 mask 在 input 上进行 masked_fill 操作，得到结果张量 result
  torch::Tensor result = torch::masked_fill(input, mask, value);
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 和 mask 复制到指定设备上得到 lazy_input 和 lazy_mask
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_mask = CopyToDevice(mask, device);
    // 在指定设备上使用 lazy_mask 在 lazy_input 上进行 masked_fill 操作，得到 lazy_result
    torch::Tensor lazy_result =
        torch::masked_fill(lazy_input, lazy_mask, value);
    // 验证 result 和 lazy_result 是否接近
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestFill) {
  // 创建一个标量值为 42
  torch::Scalar value(42);
  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 创建一个形状为 [2, 3] 的空张量 input，数据类型为 float，设备为默认设备
    torch::Tensor input = torch::empty(
        {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    
    # 将 input 复制到指定设备上，生成 lazy_input 张量
    torch::Tensor lazy_input = CopyToDevice(input, device);
    
    # 使用指定的 value 值填充 input 张量，生成 result 张量
    torch::Tensor result = torch::fill_(input, value);
    
    # 使用指定的 value 值填充 lazy_input 张量，生成 lazy_result 张量
    torch::Tensor lazy_result = torch::fill_(lazy_input, value);
    
    # 检查 result 和 lazy_result 张量的值是否在误差范围内相等
    AllClose(result, lazy_result);
    
    # 检查 input 和 lazy_input 张量的值是否在误差范围内相等
    AllClose(input, lazy_input);
}

TEST_F(LazyOpsTest, TestFillWithRank0) {
  // 创建一个标量张量，其值为42
  torch::Tensor value = torch::scalar_tensor(42);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个形状为{2, 3}的空张量，数据类型为float，放置在默认设备上
    torch::Tensor input = torch::empty(
        {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 用给定的值填充输入张量，并返回结果张量
    torch::Tensor result = torch::fill_(input, value);
    // 将标量值复制到指定设备上
    torch::Tensor lazy_value = CopyToDevice(value, device);
    // 用给定的值填充延迟加载的输入张量，并返回结果张量
    torch::Tensor lazy_result = torch::fill_(lazy_input, value);
    // 检查两个结果张量是否全部接近
    AllClose(result, lazy_result);
    // 检查两个输入张量是否全部接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestPermute) {
  // 创建一个形状为{2, 3, 4}的随机张量，数据类型为float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义维度置换的所有可能
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  // 获取输入张量的秩
  int rank = input.dim();
  // 遍历每个维度置换
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    // 遍历负维度标志
    for (bool negative_dims : {false, true}) {
      // 如果负维度标志为真，则减去输入张量的秩
      if (negative_dims) {
        std::for_each(
            dims_permutation.begin(),
            dims_permutation.end(),
            [rank](int64_t& dim) { dim -= rank; });
      }
      // 对输入张量执行维度置换，并返回结果张量
      torch::Tensor output = input.permute(dims_permutation);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将输入张量复制到指定设备上
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 对延迟加载的输入张量执行维度置换，并返回结果张量
        torch::Tensor lazy_output = lazy_input.permute(dims_permutation);
        // 检查两个输出张量是否全部接近
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestPermuteMod) {
  // 定义维度置换的所有可能
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  // 定义输入张量的大小
  std::vector<int64_t> input_sizes = {2, 3, 4};
  // 获取输入张量的秩
  int rank = input_sizes.size();
  // 遍历每个维度置换
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    // 遍历负维度标志
    for (bool negative_dims : {false, true}) {
      // 如果负维度标志为真，则减去输入张量的秩
      if (negative_dims) {
        std::for_each(
            dims_permutation.begin(),
            dims_permutation.end(),
            [rank](int64_t& dim) { dim -= rank; });
      }
      // 创建一个形状为{2, 3, 4}的零张量，数据类型为float，放置在默认设备上
      torch::Tensor input = torch::zeros(
          input_sizes,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 创建一个值为1.0的张量，数据类型为float，放置在默认设备上
      torch::Tensor one = torch::tensor(
          1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 对输入张量执行维度置换，并返回结果张量
      torch::Tensor output = input.permute(dims_permutation);
      // 在结果张量上执行原地加法操作
      output.add_(one, 1.0);
      // 在输入张量上执行原地加法操作
      input.add_(one, 1.0);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 创建一个形状为{2, 3, 4}的零张量，数据类型为float，放置在指定设备上
        torch::Tensor xinput = torch::zeros(
            input_sizes,
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        // 将零张量复制到指定设备上
        torch::Tensor lazy_input = CopyToDevice(xinput, device);
        // 将值为1.0的张量复制到指定设备上
        torch::Tensor lazy_one = CopyToDevice(one, device);
        // 对延迟加载的输入张量执行维度置换，并返回结果张量
        torch::Tensor lazy_output = lazy_input.permute(dims_permutation);
        // 在延迟加载的输出张量上执行原地加法操作
        lazy_output.add_(lazy_one, 1.0);
        // 在延迟加载的输入张量上执行原地加法操作
        lazy_input.add_(lazy_one, 1.0);
        // 检查两个输出张量是否全部接近
        AllClose(output, lazy_output);
        // 检查两个输入张量是否全部接近
        AllClose(input, lazy_input);
      });
    }
  }


注释：


    # 这里的代码段结束了两个嵌套的代码块。
    # 最内层的代码块是第一个 "}"，结束了内部的代码逻辑或循环。
    # 外层的代码块是第二个 "}"，结束了更外层的逻辑结构，如函数、类或条件语句块。
    # 注意：此处缺少上下文，无法准确确定具体结束了哪些代码逻辑。
  }


这段代码看起来是一个嵌套结构的代码块，结束了两个不同层级的代码逻辑。
}

TEST_F(LazyOpsTest, TestFlip) {
  // 创建一个随机张量作为输入，形状为 [2, 3, 4]，浮点类型，放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义所有可能的维度翻转组合
  std::vector<std::vector<int64_t>> dim_powerset = {
      {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}};
  // 遍历所有维度翻转组合
  for (std::vector<int64_t> flip_dims : dim_powerset) {
    // 对于是否翻转负维度的情况进行处理
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        // 如果翻转负维度，将维度值减去3
        std::for_each(
            flip_dims.begin(), flip_dims.end(), [](int64_t& dim) { dim -= 3; });
      }
      // 使用给定的维度翻转张量，并保存输出结果
      torch::Tensor output = torch::flip(input, flip_dims);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将输入张量复制到指定设备上
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 在指定设备上进行懒惰翻转操作
        torch::Tensor lazy_output = torch::flip(lazy_input, flip_dims);
        // 检查输出结果是否全部接近
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestPixelShuffle) {
  // 创建一个随机张量作为输入，形状为 [5, 18, 4, 4]，浮点类型，放置在默认设备上
  torch::Tensor input = torch::rand(
      {5, 18, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义像素混洗的放大因子
  int upscale_factor = 3;
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行懒惰像素混洗操作
    torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
    // 在指定设备上进行懒惰像素混洗操作
    torch::Tensor lazy_output =
        torch::pixel_shuffle(lazy_input, upscale_factor);
    // 检查输出结果是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSumToSize) {
  // 创建一个随机张量作为输入，形状为 [4, 6, 3, 7]，浮点类型，放置在默认设备上
  torch::Tensor input = torch::rand(
      {4, 6, 3, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 指定输出的目标大小
  std::vector<int64_t> out_size = {4, 1, 1, 7};
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行懒惰大小调整求和操作
    torch::Tensor output = input.sum_to_size(out_size);
    // 在指定设备上进行懒惰大小调整求和操作
    torch::Tensor lazy_output = lazy_input.sum_to_size(out_size);
    // 检查输出结果是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeDims) {
  // 创建一个随机张量作为输入，形状为 [2, 3, 4]，浮点类型，放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 指定要转置的两个维度
  int dim0 = 0;
  int dim1 = 2;
  // 对输入张量进行指定维度的转置，并保存输出结果
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行懒惰维度转置操作
    torch::Tensor lazy_output = torch::transpose(lazy_input, dim0, dim1);
    // 检查输出结果是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeDimsMod) {
  // 指定输入张量的大小
  std::vector<int64_t> input_sizes = {2, 3, 4};
  // 指定要转置的两个维度
  int dim0 = 0;
  int dim1 = 2;
  // 创建一个大小为 input_sizes 的零张量，浮点类型，放置在默认设备上
  torch::Tensor input = torch::zeros(
      input_sizes, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量张量值为1.0，浮点类型，放置在默认设备上
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 对输入张量进行指定维度的转置，并在转置后的结果上添加一个标量值
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个大小为 input_sizes 的零张量，浮点类型，放置在默认设备上
    torch::Tensor xinput = torch::zeros(
        input_sizes,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    // 将张量 `one` 复制到指定设备上，返回结果保存在 `lazy_one` 中
    torch::Tensor lazy_one = CopyToDevice(one, device);

    // 对输入张量 `lazy_input` 进行维度 `dim0` 和 `dim1` 的转置操作，结果保存在 `lazy_output` 中
    torch::Tensor lazy_output = torch::transpose(lazy_input, dim0, dim1);

    // 将 `lazy_one` 张量加到 `lazy_output` 张量上，结果保存在 `lazy_output` 中
    lazy_output.add_(lazy_one, 1.0);

    // 将 `lazy_one` 张量加到 `lazy_input` 张量上，结果保存在 `lazy_input` 中
    lazy_input.add_(lazy_one, 1.0);

    // 检查 `output` 张量和 `lazy_output` 张量是否在数值上非常接近
    AllClose(output, lazy_output);

    // 检查 `input` 张量和 `lazy_input` 张量是否在数值上非常接近
    AllClose(input, lazy_input);
}

TEST_F(LazyOpsTest, TestTransposeDimsInPlace) {
  // 创建一个形状为 {2, 3, 4} 的随机张量，数据类型为浮点数，设备为默认设备
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义两个维度进行转置操作
  int dim0 = 0;
  int dim1 = 2;
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 原地转置输入张量，并获得输出张量
    torch::Tensor output = input.transpose_(dim0, dim1);
    // 在惰性复制的输入张量上进行原地转置，并获得惰性输出张量
    torch::Tensor lazy_output = lazy_input.transpose_(dim0, dim1);
    // 检查两个输出张量是否全部近似相等
    AllClose(output, lazy_output);
    // 检查输入张量和惰性复制的输入张量是否全部近似相等
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestSplit) {
  // 创建一个形状为 {7, 8, 9} 的随机张量，数据类型为浮点数，设备为默认设备
  torch::Tensor input = torch::rand(
      {7, 8, 9}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取输入张量的维度数
  int rank = input.dim();
  // 针对每个分割大小执行以下操作
  for (int split_size : {2, 3}) {
    // 针对每个维度执行以下操作
    for (int dim = -rank; dim < rank; ++dim) {
      // 在给定维度上按指定大小分割输入张量，并获得输出张量向量
      std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
      // 对每个设备执行以下操作
      ForEachDevice([&](const torch::Device& device) {
        // 将输入张量复制到指定设备上
        torch::Tensor lazy_input = CopyToDevice(input, device);
        // 在给定维度上按指定大小分割惰性复制的输入张量，并获得惰性输出张量向量
        std::vector<torch::Tensor> lazy_outputs =
            torch::split(lazy_input, split_size, dim);
        // 断言输出张量向量和惰性输出张量向量的大小相等
        ASSERT_EQ(outputs.size(), lazy_outputs.size());
        // 对比每对输出张量和惰性输出张量，检查它们是否全部近似相等
        for (size_t i = 0; i < outputs.size(); ++i) {
          AllClose(outputs[i], lazy_outputs[i]);
        }
      });
    }
  }
}

TEST_F(LazyOpsTest, TestSplitEmpty) {
  // 创建一个形状为 {0} 的随机张量，数据类型为浮点数，设备为默认设备
  torch::Tensor input = torch::rand(
      {0}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义分割大小和维度
  int split_size = 0;
  int dim = 0;
  // 在指定维度上按指定大小分割输入张量，并获得输出张量向量
  std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定维度上按指定大小分割惰性复制的输入张量，并获得惰性输出张量向量
    std::vector<torch::Tensor> lazy_outputs =
        torch::split(lazy_input, split_size, dim);
    // 断言输出张量向量和惰性输出张量向量的大小相等
    ASSERT_EQ(outputs.size(), lazy_outputs.size());
    // 对比每对输出张量和惰性输出张量，检查它们是否全部近似相等
    for (size_t i = 0; i < outputs.size(); ++i) {
      AllClose(outputs[i], lazy_outputs[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestSplitWithSizes) {
  // 创建一个形状为 {15, 15, 15} 的随机张量，数据类型为浮点数，设备为默认设备
  torch::Tensor input = torch::rand(
      {15, 15, 15},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取输入张量的维度数
  int rank = input.dim();
  // 针对每个维度执行以下操作
  for (int dim = -rank; dim < rank; ++dim) {
    // 在指定维度上按指定大小分割输入张量，并获得输出张量向量
    std::vector<torch::Tensor> outputs =
        torch::split_with_sizes(input, {4, 5, 6}, dim);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将输入张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定维度上按指定大小分割惰性复制的输入张量，并获得惰性输出张量向量
      std::vector<torch::Tensor> lazy_outputs =
          torch::split_with_sizes(lazy_input, {4, 5, 6}, dim);
      // 断言输出张量向量和惰性输出张量向量的大小相等
      ASSERT_EQ(outputs.size(), lazy_outputs.size());
      // 对比每对输出张量和惰性输出张量，检查它们是否全部近似相等
      for (size_t i = 0; i < outputs.size(); ++i) {
        AllClose(outputs[i], lazy_outputs[i]);
      }
    });
  }
}

TEST_F(LazyOpsTest, TestCrossImplicitDim) {
  // 定义不同维度大小的列表
  std::vector<std::vector<int64_t>> dim_sizes = {
      {4, 5, 3}, {4, 3, 5}, {3, 4, 5}};
  // 针对每个维度大小列表执行以下操作
  for (auto dim_size : dim_sizes) {
    // 创建一个形状为 dim_size 的随机张量，数据类型为浮点数，设备为默认设备
    torch::Tensor input = torch::rand(
        dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 在此处继续添加
    // 使用 torch 库生成一个指定维度和浮点数类型的随机张量 other
    torch::Tensor other = torch::rand(
        dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    // 计算 input 和 other 的叉乘，生成结果张量 result
    torch::Tensor result = torch::cross(input, other);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上，并生成 lazy_input 张量
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 将 other 复制到指定设备上，并生成 lazy_other 张量
      torch::Tensor lazy_other = CopyToDevice(other, device);
      // 在指定设备上计算 lazy_input 和 lazy_other 的叉乘，生成 lazy_result 张量
      torch::Tensor lazy_result = torch::cross(lazy_input, lazy_other);
      // 检查结果 lazy_result 是否与主设备上的 result 在所有设备上都相似
      AllClose(result, lazy_result);
    });
TEST_F(LazyOpsTest, TestCrossExplicitDim) {
  // 定义一个大小为3x3的维度向量
  std::vector<int64_t> dim_size = {3, 3};
  // 创建一个随机填充的浮点数张量 input
  torch::Tensor input = torch::rand(
      dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个随机填充的浮点数张量 other，与 input 具有相同的维度
  torch::Tensor other = torch::rand(
      dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 获取维度的秩（rank）
  int rank = dim_size.size();
  // 对于每个维度 dim，从 -rank 到 rank 进行遍历
  for (int dim = -rank; dim < rank; ++dim) {
    // 计算 input 和 other 张量在维度 dim 上的叉乘结果
    torch::Tensor result = torch::cross(input, other, dim);
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 和 other 张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_other = CopyToDevice(other, device);
      // 在指定设备上计算 input 和 other 张量在维度 dim 上的叉乘结果
      torch::Tensor lazy_result = torch::cross(lazy_input, lazy_other, dim);
      // 检查计算结果是否在所有设备上都一致
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCrossZeroDim) {
  // 创建一个形状为 {0, 1, 3, 0} 的零张量 input
  torch::Tensor input = torch::rand(
      {0, 1, 3, 0},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算 input 张量与自身的叉乘结果
  torch::Tensor result = torch::cross(input, input);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上计算 input 张量与自身的叉乘结果
    torch::Tensor lazy_result = torch::cross(lazy_input, lazy_input);
    // 检查计算结果是否在所有设备上都一致
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestTriu) {
  // 定义矩阵的大小为 5x5
  int size = 5;
  // 创建一个形状为 {5, 5} 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 测试所有可能的对角线和越界情况（应该不执行任何操作）
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    // 计算对 input 张量应用上三角矩阵截取操作的结果
    torch::Tensor output = torch::triu(input, diagonal);
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定设备上计算对 input 张量应用上三角矩阵截取操作的结果
      torch::Tensor lazy_output = torch::triu(lazy_input, diagonal);
      // 检查计算结果是否在所有设备上都一致
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuNonSquare) {
  // 定义矩阵的大小为 5x6
  int size = 5;
  // 创建一个形状为 {5, 6} 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {size, size + 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 测试所有可能的对角线和越界情况（应该不执行任何操作）
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    // 计算对 input 张量应用上三角矩阵截取操作的结果
    torch::Tensor output = torch::triu(input, diagonal);
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定设备上计算对 input 张量应用上三角矩阵截取操作的结果
      torch::Tensor lazy_output = torch::triu(lazy_input, diagonal);
      // 检查计算结果是否在所有设备上都一致
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuBatch) {
  // 定义矩阵的大小为 5x5，批次大小为 3
  int size = 5;
  int batch_size = 3;
  // 创建一个形状为 {3, 5, 5} 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {batch_size, size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 测试所有可能的对角线和越界情况（应该不执行任何操作）
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    // 计算对 input 张量应用上三角矩阵截取操作的结果
    torch::Tensor output = torch::triu(input, diagonal);
    // 针对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 张量复制到指定设备上
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在指定设备上计算对 input 张量应用上三角矩阵截取操作的结果
      torch::Tensor lazy_output = torch::triu(lazy_input, diagonal);
      // 检查计算结果是否在所有设备上都一致
      AllClose(output, lazy_output);
    });
  }



// 这里的代码片段似乎是 JavaScript 的一部分，但是缺少上下文无法完全理解其含义和作用。
// 在一个函数或事件处理程序中，可能是一个闭包的结尾或某个函数的结束。
// 结尾的 '});' 可能是一个匹配的闭合，但具体的功能和逻辑需要更多上下文才能准确描述其作用。
TEST_F(LazyOpsTest, TestTrilInPlace) {
  int size = 5;
  // 对每一个可能的对角线值进行测试，从负大小到正大小
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    // 对于每一个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 创建一个随机张量作为输入，并复制到指定设备上
      torch::Tensor input = torch::rand(
          {size, size},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算输入张量在给定对角线下三角部分的操作，同时更新原始张量
      torch::Tensor output = input.tril_(diagonal);
      torch::Tensor lazy_output = lazy_input.tril_(diagonal);
      // 检验两个张量是否在数值上相近
      AllClose(output, lazy_output);
      // 检验原始输入张量是否保持不变
      AllClose(input, lazy_input);
    });
  }
}


这段代码是用于测试PyTorch中`tril_`函数的不同用例。具体注释如下：

1. **TEST_F(LazyOpsTest, TestTrilInPlace)**: 定义了一个测试用例，用于测试`tril_`函数的原位操作。
2. **int size = 5;**: 定义了测试中使用的矩阵大小为5。
3. **// Test all diagonals and out of bounds (must be no-op).**: 注释指出要测试所有可能的对角线值，包括超出边界的情况（应该是无操作）。
4. **for (int diagonal = -size; diagonal <= size; ++diagonal)**: 循环遍历所有可能的对角线值。
5. **ForEachDevice([&](const torch::Device& device) { ... });**: 对于每个设备执行以下操作的lambda函数。
6. **torch::Tensor input = torch::rand(...)**: 创建一个随机的浮点数张量作为输入。
7. **torch::Tensor lazy_input = CopyToDevice(input, device);**: 将输入张量复制到指定设备上。
8. **torch::Tensor output = input.tril_(diagonal);**: 在原地修改输入张量，将其变为下三角矩阵，并返回修改后的张量。
9. **torch::Tensor lazy_output = lazy_input.tril_(diagonal);**: 在指定设备上执行类似的下三角变换操作，并返回修改后的张量。
10. **AllClose(output, lazy_output);**: 检查两个张量的值是否在数值上相近。
11. **AllClose(input, lazy_input);**: 检查原始输入张量是否保持不变，即是否不受`tril_`操作的影响。
    // 对每个设备执行以下操作，使用lambda表达式进行设备迭代
    ForEachDevice([&](const torch::Device& device) {
      // 创建一个指定大小的随机张量 input，使用浮点数选项，并将其设备设置为默认设备
      torch::Tensor input = torch::rand(
          {size, size},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      // 将 input 复制到指定设备上，生成 lazy_input 张量
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 在原地将 input 张量下三角部分进行操作，保留指定对角线以上的元素
      torch::Tensor output = input.tril_(diagonal);
      // 在 lazy_input 张量上进行类似操作，生成 lazy_output 张量
      torch::Tensor lazy_output = lazy_input.tril_(diagonal);
      // 检查 output 和 lazy_output 张量是否全部近似相等
      AllClose(output, lazy_output);
      // 检查 input 和 lazy_input 张量是否全部近似相等
      AllClose(input, lazy_input);
    });
  }
}

// 定义一个测试用例 LazyOpsTest 中的 TestTrace 测试函数
TEST_F(LazyOpsTest, TestTrace) {
  // 定义变量 n 为 5
  int n = 5;
  // 创建一个大小为 n x n 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {n, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算 input 的迹（对角线元素之和）并赋值给 output
  torch::Tensor output = torch::trace(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并赋值给 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 计算 lazy_input 的迹，并赋值给 lazy_output
    torch::Tensor lazy_output = torch::trace(lazy_input);
    // 检查 output 和 lazy_output 是否在所有设备上都近似相等
    AllClose(output, lazy_output);
  });
}

// 定义一个测试用例 LazyOpsTest 中的 TestTraceWide 测试函数
TEST_F(LazyOpsTest, TestTraceWide) {
  // 定义变量 lines 为 3，cols 为 5
  int lines = 3;
  int cols = 5;
  // 创建一个大小为 lines x cols 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {lines, cols},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算 input 的迹并赋值给 output
  torch::Tensor output = torch::trace(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并赋值给 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 计算 lazy_input 的迹，并赋值给 lazy_output
    torch::Tensor lazy_output = torch::trace(lazy_input);
    // 检查 output 和 lazy_output 是否在所有设备上都近似相等
    AllClose(output, lazy_output);
  });
}

// 定义一个测试用例 LazyOpsTest 中的 TestTraceNarrow 测试函数
TEST_F(LazyOpsTest, TestTraceNarrow) {
  // 定义变量 lines 为 5，cols 为 3
  int lines = 5;
  int cols = 3;
  // 创建一个大小为 lines x cols 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {lines, cols},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 计算 input 的迹并赋值给 output
  torch::Tensor output = torch::trace(input);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并赋值给 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 计算 lazy_input 的迹，并赋值给 lazy_output
    torch::Tensor lazy_output = torch::trace(lazy_input);
    // 检查 output 和 lazy_output 是否在所有设备上都近似相等
    AllClose(output, lazy_output);
  });
}

// 定义一个测试用例 LazyOpsTest 中的 TestDiagRank1 测试函数
TEST_F(LazyOpsTest, TestDiagRank1) {
  // 定义变量 size 为 7
  int size = 7;
  // 创建一个大小为 size 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {size}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 测试所有可能的对角线和超出边界的情况（必须不执行任何操作）
  for (int diagonal = -2 * size; diagonal <= 2 * size; ++diagonal) {
    // 计算 input 的对角线张量（以指定对角线偏移 diagonal）并赋值给 output
    torch::Tensor output = torch::diag(input, diagonal);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上，并赋值给 lazy_input
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算 lazy_input 的对角线张量（以指定对角线偏移 diagonal）并赋值给 lazy_output
      torch::Tensor lazy_output = torch::diag(lazy_input, diagonal);
      // 检查 output 和 lazy_output 是否在所有设备上都近似相等
      AllClose(output, lazy_output);
    });
  }
}

// 定义一个测试用例 LazyOpsTest 中的 TestDiagRank2 测试函数
TEST_F(LazyOpsTest, TestDiagRank2) {
  // 定义变量 size 为 7
  int size = 7;
  // 创建一个大小为 size x size 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 测试所有可能的对角线和超出边界的情况（必须不执行任何操作）
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    // 计算 input 的对角线张量（以指定对角线偏移 diagonal）并赋值给 output
    torch::Tensor output = torch::diag(input, diagonal);
    // 对每个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 将 input 复制到指定设备上，并赋值给 lazy_input
      torch::Tensor lazy_input = CopyToDevice(input, device);
      // 计算 lazy_input 的对角线张量（以指定对角线偏移 diagonal）并赋值给 lazy_output
      torch::Tensor lazy_output = torch::diag(lazy_input, diagonal);
      // 检查 output 和 lazy_output 是否在所有设备上都近似相等
      AllClose(output, lazy_output);
    });
  }
}

// 定义一个测试用例 LazyOpsTest 中的 TestDiagFlat 测试函数
TEST_F(LazyOpsTest, TestDiagFlat) {
  // 创建一个大小为 4 x 3 x 6 x 7 的随机浮点数张量 input
  torch::Tensor input = torch::rand(
      {4, 3, 6, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 测试所有可能的对角线和超出边界的情况
  for (int diagonal = -10; diagonal < 10; ++diagonal) {
    // 计算 input 的扁平化对角线张量（以指定对角线偏移 diagonal）并赋值给 output
    torch::Tensor output = torch::diagflat(input, diagonal);
    # 对于每个设备执行以下操作（Lambda 表达式作为参数传入）
    ForEachDevice([&](const torch::Device& device) {
        # 将输入数据复制到指定设备上，返回惰性张量 lazy_input
        torch::Tensor lazy_input = CopyToDevice(input, device);
        # 使用 lazy_input 创建一个以 diagonal 为对角线元素的对角矩阵 lazy_output
        torch::Tensor lazy_output = torch::diagflat(lazy_input, diagonal);
        # 检查输出 lazy_output 是否与预期输出 output 在数值上全部相等
        AllClose(output, lazy_output);
    });
}


这段代码的注释解释了在每个设备上执行操作的过程，包括将输入数据复制到设备上，生成对角矩阵并检查其与预期输出的数值是否完全相等。
}

TEST_F(LazyOpsTest, TestDiagonal) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::diagonal(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalUpdate) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    auto input = torch::rand(
        {size, size},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    auto input_clone = input.clone();
    auto output = torch::diagonal(input, diagonal);
    output.add_(1);

    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input_clone, device);
      torch::Tensor lazy_output = torch::diagonal(lazy_input, diagonal);
      lazy_output.add_(1);

      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalNonSquare) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size + 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::diagonal(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalBatch) {
  int size = 5;
  int batch_size = 3;
  int dim1 = 1;
  int dim2 = 2;
  torch::Tensor input = torch::rand(
      {batch_size, size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output =
        torch::diagonal(input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output =
          torch::diagonal(lazy_input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestFlatten) {
  torch::Tensor input = torch::rand({4, 7, 5, 3});
  int rank = input.dim();
  for (int pos_start_dim = 0; pos_start_dim < rank; ++pos_start_dim) {
    // Iterate over all dimensions starting from pos_start_dim and flatten them
    // Perform the flattening operation on the input tensor
    torch::Tensor flat_input = torch::flatten(input, pos_start_dim);
    // Ensure the flattened tensor retains the same elements
    AllEqual(flat_input.numel(), input.numel());
  }
}
    // 循环开始，从 pos_start_dim 到 rank 的维度
    for (int pos_end_dim = pos_start_dim; pos_end_dim < rank; ++pos_end_dim) {
      // 内部循环，处理 negative_start_dim 和 negative_end_dim 的组合
      for (bool negative_start_dim : {false, true}) {
        // 内部循环，处理 negative_end_dim 的值
        for (bool negative_end_dim : {false, true}) {
          // 根据 negative_start_dim 来确定 start_dim 的值
          int start_dim =
              negative_start_dim ? pos_start_dim - rank : pos_start_dim;
          // 根据 negative_end_dim 来确定 end_dim 的值
          int end_dim = negative_end_dim ? pos_end_dim - rank : pos_end_dim;
          // 执行 torch 库的 flatten 操作，将 input 张量展平
          torch::Tensor output = torch::flatten(input, start_dim, end_dim);
          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 将 input 复制到指定设备上
            torch::Tensor lazy_input = CopyToDevice(input, device);
            // 在指定设备上执行 torch 库的 flatten 操作，将 lazy_input 张量展平
            torch::Tensor lazy_output =
                torch::flatten(lazy_input, start_dim, end_dim);
            // 检查 output 和 lazy_output 是否在数值上接近
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
TEST_F(LazyOpsTest, TestLogicalAnd) {
  // 遍历不同的标量类型进行测试
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    // 根据标量类型，生成左操作数张量
    torch::Tensor lhs = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    // 再次遍历不同的标量类型进行测试
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      // 根据标量类型，生成右操作数张量
      torch::Tensor rhs = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
      // 计算逻辑与操作，并在每个设备上执行
      torch::Tensor result = torch::logical_and(lhs, rhs);
      ForEachDevice([&](const torch::Device& device) {
        // 将左右操作数复制到指定设备上
        torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
        torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
        // 在指定设备上执行逻辑与操作
        torch::Tensor lazy_result = torch::logical_and(lazy_lhs, lazy_rhs);
        // 检查结果是否一致
        AllEqual(result, lazy_result);
      });
    }
  }

  // 检查预期的计数器未改变
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 检查预期的计数器改变（逻辑与操作）
  ExpectCounterChanged("xla::logical_and_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestBitwiseAnd) {
  // 生成随机整数张量作为左操作数和右操作数
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 计算按位与操作
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    // 将左右操作数复制到指定设备上
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    // 在指定设备上执行按位与操作
    torch::Tensor lazy_result = lazy_lhs.__and__(lazy_rhs);
    // 检查结果是否一致
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndInPlace) {
  // 生成随机整数张量作为左操作数和右操作数
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    // 将左操作数复制到指定设备上
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    // 执行原地按位与操作，并获取结果
    torch::Tensor result = lhs.__iand__(rhs);
    // 将右操作数复制到指定设备上
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    // 在指定设备上执行原地按位与操作，并获取结果
    torch::Tensor lazy_result = lazy_lhs.__iand__(lazy_rhs);
    // 检查结果是否一致
    AllEqual(result, lazy_result);
    // 检查左操作数是否一致
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndScalar) {
  // 生成随机整数张量作为左操作数
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 定义标量作为右操作数
  torch::Scalar rhs(123456789);
  // 计算张量与标量的按位与操作
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    // 将左操作数复制到指定设备上
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    // 在指定设备上执行张量与标量的按位与操作
    torch::Tensor lazy_result = lazy_lhs.__and__(rhs);
    // 检查结果是否一致
    AllEqual(result, lazy_result);
  });
}
    # 将左手边的张量复制到指定的设备上，返回复制后的张量对象
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    
    # 创建一个惰性操作结果张量，对左手边的张量执行按位与操作，并将结果赋给 lazy_result
    torch::Tensor lazy_result = lazy_lhs.__and__(rhs);
    
    # 检查两个张量的所有元素是否相等，并返回比较结果
    AllEqual(result, lazy_result);
TEST_F(LazyOpsTest, TestBitwiseAndScalarInPlace) {
  // 创建一个形状为 {4, 2} 的整型张量 lhs，其值在 [0, INT_MAX] 之间随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 创建一个标量 rhs，值为 123456789
  torch::Scalar rhs(123456789);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 lhs 复制到指定设备并创建 lazy_lhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    // 在 lhs 上进行按位与操作，并将结果存储在 result 中
    torch::Tensor result = lhs.__iand__(rhs);
    // 在 lazy_lhs 上进行按位与操作，并将结果存储在 lazy_result 中
    torch::Tensor lazy_result = lazy_lhs.__iand__(rhs);
    // 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
    // 检查 lhs 和 lazy_lhs 是否相等
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndPromotion) {
  // 创建一个形状为 {4, 2} 的浮点型张量 input，其值在 [0, 1] 之间随机生成，并放置在默认设备上
  torch::Tensor input = torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 将 input 重塑为一维张量 view
  torch::Tensor view = input.reshape(-1);
  // 对 view 中的每个元素进行按位与和不等于零的判断，并将结果存储在 result 中
  torch::Tensor result = torch::__and__(view.gt(0), view.ne(0));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备并创建 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 将 lazy_input 重塑为一维张量 lazy_view
    torch::Tensor lazy_view = lazy_input.reshape(-1);
    // 在 lazy_view 中的每个元素进行按位与和不等于零的判断，并将结果存储在 lazy_result 中
    torch::Tensor lazy_result =
        torch::__and__(lazy_view.gt(0), lazy_view.ne(0));
    // 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOr) {
  // 创建一个形状为 {4, 2} 的整型张量 lhs，其值在 [0, INT_MAX] 之间随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 创建一个形状与 lhs 相同的整型张量 rhs，其值也在 [0, INT_MAX] 之间随机生成
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 在 lhs 和 rhs 上进行按位或操作，并将结果存储在 result 中
  torch::Tensor result = lhs.__or__(rhs);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 lhs 和 rhs 复制到指定设备并创建 lazy_lhs 和 lazy_rhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    // 在 lazy_lhs 和 lazy_rhs 上进行按位或操作，并将结果存储在 lazy_result 中
    torch::Tensor lazy_result = lazy_lhs.__or__(lazy_rhs);
    // 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrInPlace) {
  // 创建一个形状为 {4, 2} 的整型张量 lhs，其值在 [0, INT_MAX] 之间随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 创建一个形状与 lhs 相同的整型张量 rhs，其值也在 [0, INT_MAX] 之间随机生成
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 lhs 复制到指定设备并创建 lazy_lhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    // 在 lhs 上进行按位或操作，并将结果存储在 result 中
    torch::Tensor result = lhs.__ior__(rhs);
    // 将 rhs 复制到指定设备并创建 lazy_rhs
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    // 在 lazy_lhs 和 lazy_rhs 上进行按位或操作，并将结果存储在 lazy_result 中
    torch::Tensor lazy_result = lazy_lhs.__ior__(lazy_rhs);
    // 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
    // 检查 lhs 和 lazy_lhs 是否相等
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrScalar) {
  // 创建一个形状为 {4, 2} 的整型张量 lhs，其值在 [0, INT_MAX] 之间随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  // 创建一个标量 rhs，值为 123456789
  torch::Scalar rhs(123456789);
  // 在 lhs 上进行按位或操作，并将结果存储在 result 中
  torch::Tensor result = lhs.__or__(rhs);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 lhs 复制到指定设备并创建 lazy_lhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    // 在 lazy_lhs 上进行按位或操作，并将结果存储在 lazy_result 中
    torch::Tensor lazy_result = lazy_lhs.__or__(rhs);
    // 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
  });
}
# 在 LazyOpsTest 测试类中定义一个名为 TestBitwiseOrScalarInPlace 的测试方法
TEST_F(LazyOpsTest, TestBitwiseOrScalarInPlace) {
  # 创建一个形状为 (4, 2) 的整数张量 lhs，其值在 [0, int32 最大值] 范围内随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  # 创建一个标量 rhs，其值为 123456789
  torch::Scalar rhs(123456789);
  # 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将 lhs 复制到指定设备上，并赋值给 lazy_lhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    # 在 lhs 上执行按位或操作，并将结果赋给 result
    torch::Tensor result = lhs.__ior__(rhs);
    # 在 lazy_lhs 上执行按位或操作，并将结果赋给 lazy_result
    torch::Tensor lazy_result = lazy_lhs.__ior__(rhs);
    # 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
    # 检查 lhs 和 lazy_lhs 是否相等
    AllEqual(lhs, lazy_lhs);
  });
}

# 在 LazyOpsTest 测试类中定义一个名为 TestBitwiseXor 的测试方法
TEST_F(LazyOpsTest, TestBitwiseXor) {
  # 创建一个形状为 (4, 2) 的整数张量 lhs，其值在 [0, int32 最大值] 范围内随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  # 创建一个形状为 (4, 2) 的整数张量 rhs，其值在 [0, int32 最大值] 范围内随机生成
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  # 在 lhs 和 rhs 上执行按位异或操作，并将结果赋给 result
  torch::Tensor result = lhs.__xor__(rhs);
  # 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将 lhs 和 rhs 复制到指定设备上，并赋值给 lazy_lhs 和 lazy_rhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    # 在 lazy_lhs 和 lazy_rhs 上执行按位异或操作，并将结果赋给 lazy_result
    torch::Tensor lazy_result = lazy_lhs.__xor__(lazy_rhs);
    # 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
  });
}

# 在 LazyOpsTest 测试类中定义一个名为 TestBitwiseXorInPlace 的测试方法
TEST_F(LazyOpsTest, TestBitwiseXorInPlace) {
  # 创建一个形状为 (4, 2) 的整数张量 lhs，其值在 [0, int32 最大值] 范围内随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  # 创建一个形状为 (4, 2) 的整数张量 rhs，其值在 [0, int32 最大值] 范围内随机生成
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  # 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将 lhs 复制到指定设备上，并赋值给 lazy_lhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    # 在 lhs 上执行按位异或赋值操作，并将结果赋给 result
    torch::Tensor result = lhs.__ixor__(rhs);
    # 将 rhs 复制到指定设备上，并赋值给 lazy_rhs
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    # 在 lazy_lhs 上执行按位异或赋值操作，并将结果赋给 lazy_result
    torch::Tensor lazy_result = lazy_lhs.__ixor__(lazy_rhs);
    # 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
    # 检查 lhs 和 lazy_lhs 是否相等
    AllEqual(lhs, lazy_lhs);
  });
}

# 在 LazyOpsTest 测试类中定义一个名为 TestBitwiseXorScalar 的测试方法
TEST_F(LazyOpsTest, TestBitwiseXorScalar) {
  # 创建一个形状为 (4, 2) 的整数张量 lhs，其值在 [0, int32 最大值] 范围内随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  # 创建一个标量 rhs，其值为 123456789
  torch::Scalar rhs(123456789);
  # 在 lhs 上执行按位异或操作，并将结果赋给 result
  torch::Tensor result = lhs.__xor__(rhs);
  # 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将 lhs 复制到指定设备上，并赋值给 lazy_lhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    # 在 lazy_lhs 上执行按位异或操作，并将结果赋给 lazy_result
    torch::Tensor lazy_result = lazy_lhs.__xor__(rhs);
    # 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
  });
}

# 在 LazyOpsTest 测试类中定义一个名为 TestBitwiseXorScalarInPlace 的测试方法
TEST_F(LazyOpsTest, TestBitwiseXorScalarInPlace) {
  # 创建一个形状为 (4, 2) 的整数张量 lhs，其值在 [0, int32 最大值] 范围内随机生成
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  # 创建一个标量 rhs，其值为 123456789
  torch::Scalar rhs(123456789);
  # 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    # 将 lhs 复制到指定设备上，并赋值给 lazy_lhs
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    # 在 lhs 上执行按位异或赋值操作，并将结果赋给 result
    torch::Tensor result = lhs.__ixor__(rhs);
    # 在 lazy_lhs 上执行按位异或赋值操作，并将结果赋给 lazy_result
    torch::Tensor lazy_result = lazy_lhs.__ixor__(rhs);
    # 检查 result 和 lazy_result 是否相等
    AllEqual(result, lazy_result);
    # 检查 lhs 和 lazy_lhs 是否相等
    AllEqual(lhs, lazy_lhs);
  });
}
TEST_F(LazyOpsTest, TestLshift) {
  // 创建一个大小为 (4, 2) 的张量，元素值为1，数据类型为int32，设备为默认设备
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 创建一个与 input 相同大小的随机整数张量，数值范围为 [0, 16)，数据类型为int32，设备为默认设备
  torch::Tensor shift_amount = torch::randint(
      16,
      input.sizes(),
      torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 对 input 和 shift_amount 进行左移操作，并返回结果张量
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 将 shift_amount 复制到指定设备上，并命名为 lazy_shift_amount
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    // 在指定设备上进行左移操作，并将结果保存在 lazy_result 中
    torch::Tensor lazy_result =
        torch::__lshift__(lazy_input, lazy_shift_amount);
    // 检查结果 lazy_result 是否与全局结果 result 在指定设备上的数据是否全部接近
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLshiftInPlace) {
  // 创建一个大小为 (4, 2) 的张量，元素值为1，数据类型为int32，设备为默认设备
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 创建一个与 input 相同大小的随机整数张量，数值范围为 [0, 16)，数据类型为int32，设备为默认设备
    torch::Tensor shift_amount = torch::randint(
        16,
        input.sizes(),
        torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
    // 对 lazy_input 在指定设备上进行原地左移操作，并将结果保存在 result 中
    torch::Tensor result = input.__ilshift__(shift_amount);
    // 将 shift_amount 复制到指定设备上，并命名为 lazy_shift_amount
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    // 在指定设备上进行原地左移操作，并将结果保存在 lazy_result 中
    torch::Tensor lazy_result = lazy_input.__ilshift__(lazy_shift_amount);
    // 检查结果 lazy_result 是否与全局结果 result 在指定设备上的数据是否全部接近
    AllClose(result, lazy_result);
    // 检查 input 和 lazy_input 是否在指定设备上的数据是否全部接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestLshiftScalar) {
  // 创建一个大小为 (4, 2) 的张量，元素值为1，数据类型为int32，设备为默认设备
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 定义一个标量 shift_amount 值为3
  torch::Scalar shift_amount = 3;
  // 对 input 进行左移操作，并返回结果张量
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行左移操作，并将结果保存在 lazy_result 中
    torch::Tensor lazy_result = torch::__lshift__(lazy_input, shift_amount);
    // 检查结果 lazy_result 是否与全局结果 result 在指定设备上的数据是否全部接近
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLshiftScalarInPlace) {
  // 创建一个大小为 (4, 2) 的张量，元素值为1，数据类型为int32，设备为默认设备
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 定义一个标量 shift_amount 值为3
  torch::Scalar shift_amount = 3;
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行原地左移操作，并将结果保存在 result 中
    torch::Tensor result = input.__ilshift__(shift_amount);
    // 在指定设备上进行原地左移操作，并将结果保存在 lazy_result 中
    torch::Tensor lazy_result = lazy_input.__ilshift__(shift_amount);
    // 检查结果 lazy_result 是否与全局结果 result 在指定设备上的数据是否全部接近
    AllClose(result, lazy_result);
    // 检查 input 和 lazy_input 是否在指定设备上的数据是否全部接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestRshift) {
  // 创建一个大小为 (4, 2) 的张量，元素值为1，数据类型为int32，设备为默认设备
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 创建一个与 input 相同大小的随机整数张量，数值范围为 [0, 16)，数据类型为int32，设备为默认设备
  torch::Tensor shift_amount = torch::randint(
      16,
      input.sizes(),
      torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 对 input 进行右移操作，并返回结果张量
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  // 针对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 将 shift_amount 复制到指定设备上，并命名为 lazy_shift_amount
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    // 使用torch命名空间下的Tensor类创建lazy_result变量，其值为将lazy_input向右位移lazy_shift_amount得到的结果
    torch::Tensor lazy_result =
        torch::__rshift__(lazy_input, lazy_shift_amount);
    // 调用AllClose函数，比较result和lazy_result是否接近，用于测试两个Tensor是否在数值上相等
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestRshiftInPlace) {
  // 创建一个形状为 (4, 2) 的全一张量，数据类型为 int32，在默认设备上创建
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 遍历每一个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并创建 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 创建一个与 input 相同形状的张量，包含随机整数，范围为 [0, 16)
    torch::Tensor shift_amount = torch::randint(
        16,
        input.sizes(),
        torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
    // 对 input 执行就地右移操作，结果保存在 result 中
    torch::Tensor result = input.__irshift__(shift_amount);
    // 将 shift_amount 复制到指定设备上，并创建 lazy_shift_amount
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    // 对 lazy_input 执行就地右移操作，结果保存在 lazy_result 中
    torch::Tensor lazy_result = lazy_input.__irshift__(lazy_shift_amount);
    // 检查 result 和 lazy_result 是否接近
    AllClose(result, lazy_result);
    // 检查 input 和 lazy_input 是否接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestRshiftScalar) {
  // 创建一个形状为 (4, 2) 的全一张量，数据类型为 int32，在默认设备上创建
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 创建一个标量 shift_amount，值为 3
  torch::Scalar shift_amount = 3;
  // 对 input 执行标量右移操作，结果保存在 result 中
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  // 遍历每一个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并创建 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 lazy_input 执行标量右移操作，结果保存在 lazy_result 中
    torch::Tensor lazy_result = torch::__rshift__(lazy_input, shift_amount);
    // 检查 result 和 lazy_result 是否接近
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestRshiftScalarInPlace) {
  // 创建一个形状为 (4, 2) 的全一张量，数据类型为 int32，在默认设备上创建
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 创建一个标量 shift_amount，值为 3
  torch::Scalar shift_amount = 3;
  // 遍历每一个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并创建 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 对 input 执行就地右移操作，结果保存在 result 中
    torch::Tensor result = input.__irshift__(shift_amount);
    // 对 lazy_input 执行就地右移操作，结果保存在 lazy_result 中
    torch::Tensor lazy_result = lazy_input.__irshift__(shift_amount);
    // 检查 result 和 lazy_result 是否接近
    AllClose(result, lazy_result);
    // 检查 input 和 lazy_input 是否接近
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestMeshgrid) {
  // 创建形状为 (3) 的随机张量 a，数据类型为 float，在默认设备上创建
  torch::Tensor a = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建形状为 (2) 的随机张量 b，数据类型为 float，在默认设备上创建
  torch::Tensor b = torch::rand(
      {2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建形状为 (4) 的随机张量 c，数据类型为 float，在默认设备上创建
  torch::Tensor c = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 使用 a, b, c 创建 meshgrid，结果保存在 d 中
  auto d = torch::meshgrid({a, b, c});
  // 遍历每一个设备，并在每个设备上执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 a, b, c 复制到指定设备上，并创建对应的 lazy_a, lazy_b, lazy_c
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    // 使用 lazy_a, lazy_b, lazy_c 创建 meshgrid，结果保存在 lazy_d 中
    auto lazy_d = torch::meshgrid({lazy_a, lazy_b, lazy_c});
    // 检查 d 和 lazy_d 的尺寸是否相同
    EXPECT_EQ(d.size(), lazy_d.size());
    // 遍
    // 调用 `AllClose` 函数，比较 `output` 和 `lazy_output` 的近似相等性
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestConstantPadIncomplete) {
  // 创建一个大小为 {4, 2, 5} 的随机张量 input，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {4, 2, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个填充向量 pad，内容为 {1, 2}
  std::vector<int64_t> pad{1, 2};
  // 定义填充值 pad_value 为 5
  float pad_value = 5;
  // 使用 torch::constant_pad_nd 函数对 input 进行常量填充，得到 output 张量
  torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 torch::constant_pad_nd 函数对 lazy_input 进行常量填充，得到 lazy_output 张量
    torch::Tensor lazy_output =
        torch::constant_pad_nd(lazy_input, pad, pad_value);
    // 检查 output 和 lazy_output 是否在设备上全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dRank3) {
  // 创建一个大小为 {2, 3, 4} 的随机张量 input，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个填充向量 pad，内容为 {2, 2, 2, 2}
  std::vector<int64_t> pad{2, 2, 2, 2};
  // 使用 torch::reflection_pad2d 函数对 input 进行反射填充，得到 output 张量
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 torch::reflection_pad2d 函数对 lazy_input 进行反射填充，得到 lazy_output 张量
    torch::Tensor lazy_output = torch::reflection_pad2d(lazy_input, pad);
    // 检查 output 和 lazy_output 是否在设备上全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dRank4) {
  // 创建一个大小为 {2, 2, 3, 4} 的随机张量 input，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {2, 2, 3, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个填充向量 pad，内容为 {2, 2, 2, 2}
  std::vector<int64_t> pad{2, 2, 2, 2};
  // 使用 torch::reflection_pad2d 函数对 input 进行反射填充，得到 output 张量
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 torch::reflection_pad2d 函数对 lazy_input 进行反射填充，得到 lazy_output 张量
    torch::Tensor lazy_output = torch::reflection_pad2d(lazy_input, pad);
    // 检查 output 和 lazy_output 是否在设备上全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dBackward) {
  // 定义一个填充向量 pad，内容为 {2, 3, 1, 2}
  std::vector<int64_t> pad{2, 3, 1, 2};
  // 定义一个函数 testfn，接受输入张量数组 inputs，并返回通过 torch::reflection_pad2d 函数处理的张量
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::reflection_pad2d(inputs[0], pad);
  };
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 创建一个大小为 {1, 2, 4, 4} 的随机张量，数据类型为 float，需要梯度，放置在默认设备上，并命名为 inputs
    TestBackward(
        {torch::rand(
            {1, 2, 4, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1d) {
  // 创建一个大小为 {1, 4} 的随机张量 input，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {1, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个填充向量 pad，内容为 {1, 2}
  std::vector<int64_t> pad{1, 2};
  // 使用 torch::replication_pad1d 函数对 input 进行复制填充，得到 output 张量
  torch::Tensor output = torch::replication_pad1d(input, pad);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 torch::replication_pad1d 函数对 lazy_input 进行复制填充，得到 lazy_output 张量
    torch::Tensor lazy_output = torch::replication_pad1d(lazy_input, pad);
    // 检查 output 和 lazy_output 是否在设备上全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1dZeroPad) {
  // 创建一个大小为 {1, 4} 的随机张量 input，数据类型为 float，放置在默认设备上
  torch::Tensor input = torch::rand(
      {1, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个填充向量 pad，内容为 {1, 0}
  std::vector<int64_t> pad{1, 0};
  // 使用 torch::replication_pad1d 函数对 input 进行复制填充，得到 output 张量
  torch::Tensor output = torch::replication_pad1d(input, pad);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上，并命名为 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 torch::replication_pad1d 函数对 lazy_input 进行复制填充，得到 lazy_output 张量
    torch::Tensor lazy_output = torch::replication_pad1d(lazy_input, pad);
    // 使用函数 AllClose 检查 output 和 lazy_output 是否全部接近
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1dBackward) {
  // 定义一个整型向量 pad，表示在1维张量周围进行复制填充的尺寸
  std::vector<int64_t> pad{2, 3};
  // 定义一个 lambda 函数 testfn，接收输入张量并返回进行复制填充后的张量
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad1d(inputs[0], pad);
  };
  // 对每一个设备进行测试
  ForEachDevice([&](const torch::Device& device) {
    // 测试反向传播
    TestBackward(
        // 创建一个随机张量作为输入，设备为默认设备，需要梯度
        {torch::rand(
            {2, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        // 使用定义的 testfn 函数进行测试
        testfn);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2d) {
  // 创建一个随机输入张量
  torch::Tensor input = torch::rand(
      {1, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个整型向量 pad，表示在2维张量周围进行复制填充的尺寸
  std::vector<int64_t> pad{1, 2, 2, 1};
  // 对输入张量进行复制填充
  torch::Tensor output = torch::replication_pad2d(input, pad);
  // 对每一个设备进行测试
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行复制填充
    torch::Tensor lazy_output = torch::replication_pad2d(lazy_input, pad);
    // 检查输出结果是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2dZeroPad) {
  // 创建一个随机输入张量
  torch::Tensor input = torch::rand(
      {1, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个整型向量 pad，表示在2维张量周围进行复制填充的尺寸（特殊情况，部分维度填充为0）
  std::vector<int64_t> pad{1, 0, 0, 1};
  // 对输入张量进行复制填充
  torch::Tensor output = torch::replication_pad2d(input, pad);
  // 对每一个设备进行测试
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行复制填充
    torch::Tensor lazy_output = torch::replication_pad2d(lazy_input, pad);
    // 检查输出结果是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2dBackward) {
  // 定义一个整型向量 pad，表示在2维张量周围进行复制填充的尺寸
  std::vector<int64_t> pad{2, 3, 1, 1};
  // 定义一个 lambda 函数 testfn，接收输入张量并返回进行复制填充后的张量
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad2d(inputs[0], pad);
  };
  // 对每一个设备进行测试
  ForEachDevice([&](const torch::Device& device) {
    // 测试反向传播
    TestBackward(
        // 创建一个随机张量作为输入，设备为默认设备，需要梯度
        {torch::rand(
            {2, 3, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        // 使用定义的 testfn 函数进行测试
        testfn);
  });
}

TEST_F(LazyOpsTest, TestAsStrided) {
  // 创建一个随机输入张量
  torch::Tensor input = torch::rand(
      {128, 320}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个整型向量 size，表示目标张量的大小
  std::vector<int64_t> size = {128, 20, 4, 4};
  // 定义一个整型向量 stride，表示目标张量的步幅
  std::vector<int64_t> stride = {320, 16, 4, 1};
  // 对输入张量进行 as_strided 操作，生成目标张量
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  // 对每一个设备进行测试
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行 as_strided 操作，生成目标张量
    torch::Tensor lazy_output =
        torch::as_strided(lazy_input, /*size=*/size, /*stride=*/stride);
    // 检查输出结果是否近似相等
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAsStridedInPlace) {
  // 创建一个随机输入张量
  torch::Tensor input = torch::rand(
      {128, 320}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义一个整型向量 size，表示目标张量的大小
  std::vector<int64_t> size = {128, 20, 4, 4};
  // 定义一个整型向量 stride，表示目标张量的步幅
  std::vector<int64_t> stride = {320, 16, 4, 1};
  // 对每一个设备进行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 在指定设备上进行 in-place 的 as_strided 操作
    torch::Tensor lazy_output =
        torch::as_strided(lazy_input, /*size=*/size, /*stride=*/stride);
    // 使用 torch 库中的函数 torch::as_strided_ 对输入的张量进行视图操作，创建输出张量
    torch::Tensor output =
        torch::as_strided_(input, /*size=*/size, /*stride=*/stride);
    // 使用 torch::as_strided_ 对延迟加载的输入张量进行视图操作，创建延迟加载的输出张量
    torch::Tensor lazy_output =
        torch::as_strided_(lazy_input, /*size=*/size, /*stride=*/stride);
    // 检查两个输出张量是否在数值上相等
    AllClose(output, lazy_output);
    // 检查原始输入张量和延迟加载的输入张量是否在数值上相等
    AllClose(input, lazy_input);
    // Lambda 函数的结束标志
    });
}

// 定义测试用例 LazyOpsTest 中的一个测试函数 TestAsStridedWithOffset
TEST_F(LazyOpsTest, TestAsStridedWithOffset) {
  // 创建一个形状为 {4, 8, 2} 的随机张量 input，数据类型为 float，使用默认设备
  torch::Tensor input = torch::rand(
      {4, 8, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义张量的大小 size 和步长 stride
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  // 定义存储偏移 storage_offset
  int64_t storage_offset = 4;
  // 调用 torch::as_strided 函数创建新的张量 output
  torch::Tensor output = torch::as_strided(
      input,
      /*size=*/size,
      /*stride=*/stride,
      /*storage_offset=*/storage_offset);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 input 复制到指定设备上得到 lazy_input
    torch::Tensor lazy_input = CopyToDevice(input, device);
    // 使用 torch::as_strided 在指定设备上创建 lazy_output
    torch::Tensor lazy_output = torch::as_strided(
        lazy_input,
        /*size=*/size,
        /*stride=*/stride,
        /*storage_offset=*/storage_offset);
    // 检查 output 和 lazy_output 是否相似
    AllClose(output, lazy_output);
  });
}

// 定义测试用例 LazyOpsTest 中的一个测试函数 TestAsStridedWithInplaceCopy
TEST_F(LazyOpsTest, TestAsStridedWithInplaceCopy) {
  // 创建全为 1 的张量 grad，形状为 {4}，数据类型为 float，使用默认设备
  torch::Tensor grad = torch::ones(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 定义张量的大小 size 和步长 stride
  std::vector<int64_t> size = {4};
  std::vector<int64_t> stride = {1};
  // 创建全为 0 的张量 output，形状为 {4}，与 grad 的选项相同
  torch::Tensor output = torch::zeros({4}, grad.options());
  // 在 output 上执行 as_strided 操作，并将 grad 复制到结果中
  output.as_strided(size, stride).copy_(grad);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 grad 复制到指定设备上得到 lazy_grad
    torch::Tensor lazy_grad = CopyToDevice(grad, device);
    // 创建全为 0 的张量 lazy_output，形状为 {4}，与 lazy_grad 的选项相同
    torch::Tensor lazy_output = torch::zeros({4}, lazy_grad.options());
    // 在 lazy_output 上执行 as_strided 操作，并将 lazy_grad 复制到结果中
    lazy_output.as_strided(size, stride).copy_(lazy_grad);
    // 检查 output 和 lazy_output 是否相似
    AllClose(output, lazy_output);
  });
}

// 定义测试用例 LazyOpsTest 中的一个测试函数 TestEmptyStrided
TEST_F(LazyOpsTest, TestEmptyStrided) {
  // 定义张量的大小 size 和步长 stride
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  // 创建空的张量 output，形状为 {4, 4, 2}，按照给定的 size 和 stride
  torch::Tensor output = torch::empty_strided(/*size=*/size, /*stride=*/stride);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 在指定设备上创建空的 lazy_output，形状为 {4, 4, 2}，按照给定的 size 和 stride
    torch::Tensor lazy_output =
        torch::empty_strided(/*size=*/size, /*stride=*/stride);
    // 检查 output 和 lazy_output 的大小和步长是否相同
    EXPECT_EQ(output.sizes(), lazy_output.sizes());
    EXPECT_EQ(output.strides(), lazy_output.strides());
  });
}

// 定义测试用例 LazyOpsTest 中的一个测试函数 TestAvgPool2DBackward
TEST_F(LazyOpsTest, TestAvgPool2DBackward) {
  // 定义池化核大小 kernel_size
  int kernel_size = 2;
  // 对于不同的步长 stride
  for (int stride = 1; stride <= 2; ++stride) {
    // 对于不同的填充 padding
    for (int padding = 0; padding <= 1; ++padding) {
      // 对于包括填充 count_include_pad 的不同设置
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 互操作测试 ceil_mode=true
        for (bool ceil_mode : {false, true}) {
          // 定义测试函数 testfn，接受输入张量列表，并返回平均池化结果
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool2d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          // 对每个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 在设备上测试反向传播
            TestBackward(
                {torch::rand(
                    {1, 1, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}
TEST_F(LazyOpsTest, TestAvgPool3DBackward) {
  // 定义池化核大小
  int kernel_size = 2;
  // 循环遍历步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历填充大小
    for (int padding = 0; padding <= 1; ++padding) {
      // 遍历是否包含填充的选项
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 交互测试 ceil_mode=true
        for (bool ceil_mode : {false, true}) {
          // 定义测试函数，执行 3D 平均池化操作
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            // 调用 PyTorch 的 avg_pool3d 函数进行池化操作
            return torch::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          // 对每个设备执行反向传播测试
          ForEachDevice([&](const torch::Device& device) {
            // 生成随机张量作为输入
            TestBackward(
                {torch::rand(
                    {1, 1, 7, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNoBatchBackward) {
  // 定义池化核大小
  int kernel_size = 2;
  // 循环遍历步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历填充大小
    for (int padding = 0; padding <= 1; ++padding) {
      // 遍历是否包含填充的选项
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 交互测试 ceil_mode=true
        for (bool ceil_mode : {false, true}) {
          // 定义测试函数，执行 2D 平均池化操作
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            // 调用 PyTorch 的 avg_pool2d 函数进行池化操作
            return torch::avg_pool2d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          // 对每个设备执行反向传播测试
          ForEachDevice([&](const torch::Device& device) {
            // 生成随机张量作为输入
            TestBackward(
                {torch::rand(
                    {1, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNoBatchBackward) {
  // 定义池化核大小
  int kernel_size = 2;
  // 循环遍历步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 循环遍历填充大小
    for (int padding = 0; padding <= 1; ++padding) {
      // 遍历是否包含填充的选项
      for (bool count_include_pad : {true, false}) {
        // 通过 CPU 交互测试 ceil_mode=true
        for (bool ceil_mode : {false, true}) {
          // 定义测试函数，执行 3D 平均池化操作（不带批处理）
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            // 调用 PyTorch 的 avg_pool3d 函数进行池化操作
            return torch::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          // 对每个设备执行反向传播测试
          ForEachDevice([&](const torch::Device& device) {
            // 生成随机张量作为输入
            TestBackward(
                {torch::rand(
                    {1, 1, 7, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}
    // 循环变量 padding 从 0 到 1，用于控制池化操作的填充大小
    for (int padding = 0; padding <= 1; ++padding) {
      // 循环变量 count_include_pad 遍历 {true, false}，确定是否包含填充在内
      for (bool count_include_pad : {true, false}) {
        // 在 CPU 互操作中测试 ceil_mode=true 的情况
        // 循环变量 ceil_mode 遍历 {false, true}，决定是否启用 ceil 模式
        for (bool ceil_mode : {false, true}) {
          // 定义一个 Lambda 函数 testfn，接受输入张量并进行三维平均池化操作
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            // 调用 torch::avg_pool3d 函数进行平均池化操作
            return torch::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          // 对每一个设备执行以下操作
          ForEachDevice([&](const torch::Device& device) {
            // 测试反向传播
            TestBackward(
                {torch::rand(
                    {1, 7, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DNoBatchBackward) {
  // 如果当前环境是 CUDA，跳过测试
  if (IsCuda()) {
    GTEST_SKIP();
  }
  // 对于每个输出大小进行迭代测试
  for (int64_t output_size : {7, 4}) {
    // 定义测试函数，输入为一个张量向量，输出为经过自适应三维平均池化后的张量
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    // 针对每个设备进行测试
    ForEachDevice([&](const torch::Device& device) {
      // 调用测试反向传播函数，输入一个张量，其形状为 [1, 56, 28, 28]，在默认设备上进行计算，要求梯度
      TestBackward(
          {torch::rand(
              {1, 56, 28, 28},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DBackward) {
  // 如果当前环境是 CUDA，跳过测试
  if (IsCuda()) {
    GTEST_SKIP();
  }
  // 对于每个输出大小进行迭代测试
  for (int64_t output_size : {7, 4}) {
    // 定义测试函数，输入为一个张量向量，输出为经过自适应三维平均池化后的张量
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    // 针对每个设备进行测试
    ForEachDevice([&](const torch::Device& device) {
      // 调用测试反向传播函数，输入一个张量，其形状为 [4, 1, 56, 28, 28]，在指定设备上进行计算，要求梯度
      TestBackward(
          {torch::rand(
              {4, 1, 56, 28, 28},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DBackward) {
  // 对于每个输出大小进行迭代测试
  for (int64_t output_size : {7, 8}) {
    // 定义测试函数，输入为一个张量向量，输出为经过自适应二维平均池化后的张量
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    // 针对每个设备进行测试
    ForEachDevice([&](const torch::Device& device) {
      // 调用测试反向传播函数，输入一个张量，其形状为 [4, 1, 56, 56]，在指定设备上进行计算，要求梯度
      TestBackward(
          {torch::rand(
              {4, 1, 56, 56},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DNoBatchBackward) {
  // 对于每个输出大小进行迭代测试
  for (int64_t output_size : {7, 8}) {
    // 定义测试函数，输入为一个张量向量，输出为经过自适应二维平均池化后的张量
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    // 针对每个设备进行测试
    ForEachDevice([&](const torch::Device& device) {
      // 调用测试反向传播函数，输入一个张量，其形状为 [1, 56, 56]，在指定设备上进行计算，要求梯度
      TestBackward(
          {torch::rand(
              {1, 56, 56},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestConv2D) {
  // 输入通道数和输出通道数都为 4，卷积核大小为 3
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  // 对于步长为 1 到 3 的情况进行迭代测试
  for (int stride = 1; stride <= 3; ++stride) {
    // 循环遍历不同的填充值，包括 0、1、2
    for (int padding = 0; padding <= 2; ++padding) {
      // 遍历是否包含偏置项的布尔值，true 和 false
      for (bool with_bias : {true, false}) {
        // 循环遍历不同的扩张值，包括 1、2、3
        for (int dilation = 1; dilation <= 3; ++dilation) {
          // 循环遍历不同的组数，包括 1、2、4，覆盖正常、分组、深度可分离卷积
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            // 对每个设备执行以下操作
            ForEachDevice([&](const torch::Device& device) {
              // 创建随机输入张量，形状为 [1, in_channels, 7, 7]
              torch::Tensor input = torch::rand(
                  {1, in_channels, 7, 7},
                  torch::TensorOptions(torch::kDouble).device(DefaultDevice()));
              // 创建随机权重张量，形状为 [out_channels, in_channels / groups, kernel_size, kernel_size]
              torch::Tensor weight = torch::rand(
                  {out_channels,
                   in_channels / groups,
                   kernel_size,
                   kernel_size},
                  torch::TensorOptions(torch::kDouble).device(DefaultDevice()));
              // 如果包含偏置项，则创建随机偏置张量，形状为 [out_channels]，否则为空张量
              torch::Tensor bias = with_bias
                  ? torch::rand(
                        {out_channels},
                        torch::TensorOptions(torch::kDouble)
                            .device(DefaultDevice()))
                  : torch::Tensor();

              // 将输入张量复制到指定设备上
              torch::Tensor lazy_input = CopyToDevice(input, device);
              // 将权重张量复制到指定设备上
              torch::Tensor lazy_weight = CopyToDevice(weight, device);
              // 如果包含偏置项，则将偏置张量复制到指定设备上，否则创建空张量
              torch::Tensor lazy_bias =
                  with_bias ? CopyToDevice(bias, device) : torch::Tensor();

              // 执行卷积操作，计算输出张量
              torch::Tensor output = torch::conv2d(
                  input,
                  weight,
                  bias,
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
              // 在延迟复制的输入、权重、偏置上执行卷积操作，计算输出张量
              torch::Tensor lazy_output = torch::conv2d(
                  lazy_input,
                  lazy_weight,
                  lazy_bias,
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
              // 检查两种方法计算的输出张量是否近似相等
              AllClose(output, lazy_output);
            });
          }
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestConv2DBackward) {
  // 定义输入通道数和输出通道数
  int in_channels = 4;
  int out_channels = 4;
  // 定义卷积核大小
  int kernel_size = 3;
  // 循环遍历步长（stride）从1到3
  for (int stride = 1; stride <= 3; ++stride) {
    // 循环遍历填充（padding）从0到2
    for (int padding = 0; padding <= 2; ++padding) {
      // 遍历是否使用偏置（with_bias）的布尔值 true 和 false
      for (bool with_bias : {true, false}) {
        // 循环遍历扩张（dilation）从1到3
        for (int dilation = 1; dilation <= 3; ++dilation) {
          // 遍历分组（groups）值为1, 2, 4，分别表示普通、分组和深度可分离卷积
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            // 定义测试函数，接收输入张量的向量并返回卷积结果张量
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              // 执行二维卷积操作
              return torch::conv2d(
                  inputs[0],
                  inputs[1],
                  inputs[2],
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
            };

            // 对每个设备进行迭代执行测试
            ForEachDevice([&](const torch::Device& device) {
              // 根据是否使用偏置生成偏置张量
              torch::Tensor bias = with_bias
                  ? torch::rand(
                        {out_channels},
                        torch::TensorOptions(torch::kDouble)
                            .device(DefaultDevice()))
                  : torch::Tensor();
              // 执行反向传播测试
              TestBackward(
                  {torch::rand(
                       {1, in_channels, 7, 7},
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   torch::rand(
                       {out_channels,
                        in_channels / groups,
                        kernel_size,
                        kernel_size},
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   bias},
                  device,
                  testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTransposedConv2DBackward) {
  // 设置输入通道数和输出通道数
  int in_channels = 4;
  int out_channels = 4;
  // 设置卷积核大小
  int kernel_size = 3;
  // 循环遍历步长（stride）从1到2
  for (int stride = 1; stride <= 2; ++stride) {
    // 外层循环：对填充(padding)进行迭代，从0到1
    for (int padding = 0; padding <= 1; ++padding) {
      // 第二层循环：对扩张(dilation)进行迭代，从1到2
      for (int dilation = 1; dilation <= 2; ++dilation) {
        // 第三层循环：对输出填充(output_padding)进行迭代，从0开始直到 stride 和 dilation 中的最大值
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation);
             ++output_padding) {
          // 第四层循环：对是否包含偏置(with_bias)进行迭代，分别为 true 和 false
          for (bool with_bias : {true, false}) {
            // 第五层循环：对分组数(groups)进行迭代，包括1、2、4，覆盖普通、分组、深度卷积
            for (int groups :
                 {1, 2, 4}) { // covers normal, grouped, depthwise conv.
              // 定义测试函数 testfn，接受输入张量列表，返回转置卷积的张量
              auto testfn = [&](const std::vector<torch::Tensor>& inputs)
                  -> torch::Tensor {
                // 调用 PyTorch 的二维转置卷积函数 torch::conv_transpose2d
                return torch::conv_transpose2d(
                    inputs[0],  // 输入张量
                    inputs[1],  // 权重张量
                    inputs[2],  // 偏置张量
                    /*stride=*/{stride, stride + 1},  // 步幅
                    /*padding=*/{padding, padding + 1},  // 填充
                    /*output_padding=*/output_padding,  // 输出填充
                    /*groups=*/groups,  // 分组数
                    /*dilation=*/{dilation, dilation + 1});  // 扩张
              };

              // 对每个设备执行操作的 Lambda 函数，其中包含使用 torch::rand 创建张量的操作
              ForEachDevice([&](const torch::Device& device) {
                // 创建随机输入张量 input，形状为 {4, out_channels, 7, 7}
                torch::Tensor input = torch::rand(
                    {4, out_channels, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true));
                // 创建随机权重张量 weight，形状为 {out_channels, in_channels/groups, kernel_size, kernel_size}
                torch::Tensor weight = torch::rand(
                    {out_channels,
                     in_channels / groups,
                     kernel_size,
                     kernel_size},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true));
                // 如果 with_bias 为 true，则创建随机偏置张量 bias，形状为 {in_channels}
                torch::Tensor bias = with_bias
                    ? torch::rand(
                          {in_channels},
                          torch::TensorOptions(torch::kFloat)
                              .device(DefaultDevice())
                              .requires_grad(true))
                    : torch::Tensor();  // 否则不创建偏置

                // 调用测试反向传播的函数 TestBackward，传入输入张量、设备、测试函数 testfn，以及误差容差参数
                TestBackward(
                    {input, weight, bias},
                    device,
                    testfn,
                    /*rtol=*/1e-5,
                    /*atol=*/1e-5);
              });
            }
          };
        }
      }
    }
  }
}

// 定义测试用例 LazyOpsTest 的成员函数 TestConv3DBackward
TEST_F(LazyOpsTest, TestConv3DBackward) {
  int in_channels = 4; // 输入通道数
  int out_channels = 4; // 输出通道数
  int kernel_size = 3; // 卷积核大小
  // 遍历步长
  for (int stride = 1; stride <= 3; ++stride) {
    // 遍历填充
    for (int padding = 1; padding <= 2; ++padding) {
      // 遍历是否使用偏置
      for (bool with_bias : {true, false}) {
        // 遍历扩张率
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 遍历分组数，涵盖普通、分组、深度卷积
          for (int groups :
               {1, 2, 4}) {
            // 定义测试函数 testfn，接受输入张量列表并返回张量
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              // 调用 PyTorch 的三维卷积函数 conv3d
              return torch::conv3d(
                  inputs[0],
                  inputs[1],
                  inputs[2],
                  /*stride=*/{stride, stride, stride}, // 设置步长
                  /*padding=*/{padding, padding, padding}, // 设置填充
                  /*dilation=*/{dilation, dilation, dilation}, // 设置扩张率
                  groups); // 设置分组数
            };

            // 对每个设备执行测试
            ForEachDevice([&](const torch::Device& device) {
              // 如果使用偏置，则生成随机偏置张量，否则为空张量
              torch::Tensor bias = with_bias
                  ? torch::rand(
                        {out_channels},
                        torch::TensorOptions(torch::kDouble)
                            .device(DefaultDevice()))
                  : torch::Tensor();
              // 调用 TestBackward 函数进行反向传播测试
              TestBackward(
                  {torch::rand(
                       {4, in_channels, 7, 7, 7}, // 输入张量大小
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   torch::rand(
                       {out_channels,
                        in_channels / groups,
                        kernel_size,
                        kernel_size,
                        kernel_size}, // 卷积核张量大小
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   bias}, // 偏置张量
                  device,
                  testfn); // 测试函数
            });
          }
        };
      }
    }
  }
}

// 定义测试用例 LazyOpsTest 的成员函数 TestTransposedConv3DBackward
TEST_F(LazyOpsTest, TestTransposedConv3DBackward) {
  int in_channels = 4; // 输入通道数
  int out_channels = 4; // 输出通道数
  int kernel_size = 3; // 卷积核大小
  // 遍历步长
  for (int stride = 1; stride <= 2; ++stride) {
    // 外层循环：遍历填充(padding)参数的可能取值，0和1
    for (int padding = 0; padding <= 1; ++padding) {
      // 中层循环：遍历膨胀(dilation)参数的可能取值，1和2
      for (int dilation = 1; dilation <= 2; ++dilation) {
        // 内层循环：根据步幅和膨胀参数的最大值遍历输出填充(output_padding)参数的可能取值
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation);
             ++output_padding) {
          // 循环：遍历是否带偏置(with_bias)的布尔取值，true和false
          for (bool with_bias : {true, false}) {
            // 循环：遍历组数(groups)的可能取值，1、2、4，涵盖正常、分组、深度卷积
            for (int groups :
                 {1, 2, 4}) { // covers normal, grouped, depthwise conv.
              // 定义测试函数(testfn)，采用 lambda 表达式，输入为一个包含三个张量的向量，返回一个张量
              auto testfn = [&](const std::vector<torch::Tensor>& inputs)
                  -> torch::Tensor {
                // 调用 torch 库的三维转置卷积函数 conv_transpose3d，传入参数如下：
                // - inputs[0]: 输入张量
                // - inputs[1]: 权重张量
                // - inputs[2]: 可选的偏置张量
                // - stride: 步幅参数，数组形式
                // - padding: 填充参数，数组形式
                // - output_padding: 输出填充参数
                // - groups: 组数参数
                // - dilation: 膨胀参数，数组形式
                return torch::conv_transpose3d(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    /*stride=*/{stride, stride + 1, stride},
                    /*padding=*/{padding, padding + 1, stride},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1, dilation});
              };
              // 对每一种设备执行操作，此处使用 ForEachDevice 函数迭代不同设备
              ForEachDevice([&](const torch::Device& device) {
                // 生成随机张量 input，形状为 {4, out_channels, 7, 7, 7}，双精度类型
                torch::Tensor input = torch::rand(
                    {4, out_channels, 7, 7, 7},
                    torch::TensorOptions(torch::kDouble)
                        .device(DefaultDevice())
                        .requires_grad(true));
                // 生成随机张量 weight，形状根据参数设置，双精度类型
                torch::Tensor weight = torch::rand(
                    {out_channels,
                     in_channels / groups,
                     kernel_size,
                     kernel_size,
                     kernel_size},
                    torch::TensorOptions(torch::kDouble)
                        .device(DefaultDevice())
                        .requires_grad(true));
                // 如果 with_bias 为 true，生成随机张量 bias，形状为 {in_channels}，双精度类型
                // 否则设置 bias 为空张量
                torch::Tensor bias = with_bias
                    ? torch::rand(
                          {in_channels},
                          torch::TensorOptions(torch::kDouble)
                              .device(DefaultDevice())
                              .requires_grad(true))
                    : torch::Tensor();
                // 调用 TestBackward 函数，测试反向传播，传入参数为 {input, weight, bias}，设备为当前设备
                TestBackward({input, weight, bias}, device, testfn);
              });
            }
          };
        }
      }
    }
  }


这段代码包含了多层嵌套循环，用于测试三维转置卷积的各种参数组合，在不同的设备上进行反向传播测试。
// 定义一个测试用例函数 TestMaxPool2DBackward，属于 LazyOpsTest 测试集合，用于测试最大池化层的反向传播
TEST_F(LazyOpsTest, TestMaxPool2DBackward) {
  int kernel_size = 3; // 设置卷积核大小为 3
  for (int stride = 1; stride <= 2; ++stride) { // 迭代不同的步长值（1 和 2）
    for (int padding = 0; padding <= 1; ++padding) { // 迭代不同的填充值（0 和 1）
      // 通过 CPU 互操作测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) { // 遍历 ceil_mode 的两种取值（false 和 true）
        // 定义一个 lambda 函数 testfn，接受输入向量并返回最大池化操作的结果张量
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          // 执行二维最大池化操作，使用输入的第一个张量作为输入
          return torch::max_pool2d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size}, // 设置卷积核大小
              /*stride=*/{stride, stride}, // 设置步长
              /*padding=*/{padding, padding}, // 设置填充
              /*dilation=*/{1, 1}, // 设置扩张
              /*ceil_mode=*/ceil_mode); // 设置是否使用 ceil_mode
        };

        // 对每个设备执行测试
        ForEachDevice([&](const torch::Device& device) {
          // 调用 TestBackward 函数，传入随机生成的张量作为输入，设备类型为 device，测试函数为 testfn
          TestBackward(
              {torch::rand(
                  {1, 2, 8, 8}, // 生成大小为 [1, 2, 8, 8] 的随机浮点数张量
                  torch::TensorOptions(torch::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}



// 定义一个测试用例函数 TestMaxPool3DBackward，属于 LazyOpsTest 测试集合，用于测试三维最大池化层的反向传播
TEST_F(LazyOpsTest, TestMaxPool3DBackward) {
  int kernel_size = 3; // 设置卷积核大小为 3
  for (int stride = 1; stride <= 2; ++stride) { // 迭代不同的步长值（1 和 2）
    for (int padding = 0; padding <= 1; ++padding) { // 迭代不同的填充值（0 和 1）
      // 通过 CPU 互操作测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) { // 遍历 ceil_mode 的两种取值（false 和 true）
        // 定义一个 lambda 函数 testfn，接受输入向量并返回三维最大池化操作的结果张量
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          // 执行三维最大池化操作，使用输入的第一个张量作为输入
          return torch::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size}, // 设置卷积核大小
              /*stride=*/{stride, stride, stride}, // 设置步长
              /*padding=*/{padding, padding, padding}, // 设置填充
              /*dilation=*/{1, 1, 1}, // 设置扩张
              /*ceil_mode=*/ceil_mode); // 设置是否使用 ceil_mode
        };

        // 对每个设备执行测试
        ForEachDevice([&](const torch::Device& device) {
          // 调用 TestBackward 函数，传入随机生成的张量作为输入，设备类型为 device，测试函数为 testfn
          TestBackward(
              {torch::rand(
                  {1, 2, 4, 4, 4}, // 生成大小为 [1, 2, 4, 4, 4] 的随机浮点数张量
                  torch::TensorOptions(torch::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}



// 定义一个测试用例函数 TestMaxPool2DNoBatchBackward，属于 LazyOpsTest 测试集合，用于测试无批处理的二维最大池化层的反向传播
TEST_F(LazyOpsTest, TestMaxPool2DNoBatchBackward) {
  int kernel_size = 3; // 设置卷积核大小为 3
  for (int stride = 1; stride <= 2; ++stride) { // 迭代不同的步长值（1 和 2）
    // 循环测试两种填充（0 和 1）
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true
      // 使用 lambda 表达式定义测试函数 testfn，接受一个输入张量向量并返回最大池化后的张量
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        // 调用 torch::max_pool2d 函数进行最大池化操作
        return torch::max_pool2d(
            inputs[0],
            /*kernel_size=*/{kernel_size, kernel_size},  // 池化核大小
            /*stride=*/{stride, stride},  // 步幅大小
            /*padding=*/{padding, padding},  // 填充大小，根据外部循环变量 padding
            /*dilation=*/{1, 1},  // 膨胀大小
            /*ceil_mode=*/ceil_mode);  // 是否启用 ceil_mode
      };

      // 对每一个设备执行测试
      ForEachDevice([&](const torch::Device& device) {
        // 调用 TestBackward 函数进行反向传播测试
        TestBackward(
            {torch::rand(
                {2, 8, 8},  // 随机生成大小为 [2, 8, 8] 的浮点张量
                torch::TensorOptions(torch::kFloat)
                    .device(DefaultDevice())  // 设置默认设备
                    .requires_grad(true))},  // 需要计算梯度
            device,
            testfn);  // 使用前面定义的 testfn 进行测试
      });
    }
}

TEST_F(LazyOpsTest, TestMaxPool3DNoBatchBackward) {
  // 定义最大池化的核大小
  int kernel_size = 3;
  // 遍历不同的步幅
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历不同的填充大小
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 定义测试函数，对输入进行 3D 最大池化操作
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        // 对每个设备进行测试
        ForEachDevice([&](const torch::Device& device) {
          // 执行反向传播测试，传入随机生成的张量作为输入
          TestBackward(
              {torch::rand(
                  {2, 4, 4, 4},
                  torch::TensorOptions(torch::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool2DBackward) {
  // 定义最大反池化的核大小
  int kernel_size = 2;
  // 随机生成一个输入张量
  torch::Tensor input = torch::rand(
      {2, 2, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历不同的步幅
  for (int stride = 1; stride <= 2; ++stride) {
    // 遍历不同的填充大小
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true
      for (bool ceil_mode : {false, true}) {
        // 遍历不同的扩张大小
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 定义输出张量和索引张量
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          // 定义测试函数，对输入进行 2D 最大反池化操作
          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool2d(inputs[0], inputs[1], output_size);
          };

          // 对每个设备进行测试
          ForEachDevice([&](const torch::Device& device) {
            // 执行反向传播测试，传入输出张量和索引张量作为输入
            TestBackward(
                {output.requires_grad_(true), indices}, device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool3DBackward) {
  // 定义最大反池化的核大小
  int kernel_size = 2;
  // 随机生成一个输入张量
  torch::Tensor input = torch::rand(
      {1, 1, 4, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 遍历不同的步幅
  for (int stride = 1; stride <= 2; ++stride) {
    // 使用 padding 变量进行循环，值从 0 到 1
    for (int padding = 0; padding <= 1; ++padding) {
      // 通过 CPU 互操作测试 ceil_mode=true 的情况
      for (bool ceil_mode : {false, true}) {
        // 使用 dilation 变量进行循环，值从 1 到 2
        for (int dilation = 1; dilation <= 2; ++dilation) {
          // 声明 torch::Tensor 类型的 output 和 indices 变量
          torch::Tensor output;
          torch::Tensor indices;
          // 调用 torch::max_pool3d_with_indices 函数进行最大池化操作
          std::tie(output, indices) = torch::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          // 创建包含三个元素的 output_size 向量
          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          // 声明并定义 lambda 函数 testfn，接受输入并进行 max_unpool3d 操作
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool3d(
                inputs[0],
                inputs[1],
                output_size,
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding});
          };

          // 对每个设备执行 TestBackward 函数，测试反向传播
          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {output.requires_grad_(true), indices}, device, testfn);
          });
        }
      }
    }
  }
TEST_F(LazyOpsTest, TestTanhBackward) {
  // 定义一个测试函数，计算输入张量的双曲正切函数
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::tanh(inputs[0]);
  };
  // 针对每个设备执行测试反向传播
  ForEachDevice([&](const torch::Device& device) {
    // 调用测试函数，测试反向传播
    TestBackward(
        {torch::rand(
            {2, 2},  // 创建一个大小为2x2的随机张量
            torch::TensorOptions(torch::kFloat)  // 使用浮点数选项
                .device(DefaultDevice())  // 设置默认设备
                .requires_grad(true))},  // 设置需要计算梯度
        device,
        testfn,
        /*rtol=*/1e-3,  // 相对误差容忍度
        /*atol=*/1e-5);  // 绝对误差容忍度
  });
}

TEST_F(LazyOpsTest, TestSigmoidBackward) {
  // 定义一个测试函数，计算输入张量的sigmoid函数
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::sigmoid(inputs[0]);
  };
  // 针对每个设备执行测试反向传播
  ForEachDevice([&](const torch::Device& device) {
    // 调用测试函数，测试反向传播
    TestBackward(
        {torch::rand(
            {2, 2},  // 创建一个大小为2x2的随机张量
            torch::TensorOptions(torch::kFloat)  // 使用浮点数选项
                .device(DefaultDevice())  // 设置默认设备
                .requires_grad(true))},  // 设置需要计算梯度
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestLogSigmoidBackward) {
  // 定义一个测试函数，计算输入张量的log sigmoid函数
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::log_sigmoid(inputs[0]);
  };
  // 针对每个设备执行测试反向传播
  ForEachDevice([&](const torch::Device& device) {
    // 调用测试函数，测试反向传播
    TestBackward(
        {torch::rand(
            {2, 2},  // 创建一个大小为2x2的随机张量
            torch::TensorOptions(torch::kFloat)  // 使用浮点数选项
                .device(DefaultDevice())  // 设置默认设备
                .requires_grad(true))},  // 设置需要计算梯度
        device,
        testfn,
        /*rtol=*/1e-3,  // 相对误差容忍度
        /*atol=*/1e-5);  // 绝对误差容忍度
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    // 定义一个测试函数，计算输入张量在指定维度上的log softmax函数
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::log_softmax(inputs[0], dim);
    };

    // 针对每个设备执行测试反向传播
    ForEachDevice([&](const torch::Device& device) {
      // 调用测试函数，测试反向传播
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},  // 创建一个大小为5x3x4x2的随机张量
              torch::TensorOptions(torch::kFloat)  // 使用浮点数选项
                  .device(DefaultDevice())  // 设置默认设备
                  .requires_grad(true))},  // 设置需要计算梯度
          device,
          testfn,
          /*rtol=*/1e-3,  // 相对误差容忍度
          /*atol=*/1e-4);  // 绝对误差容忍度
    });
  }
}

TEST_F(LazyOpsTest, TestSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    // 定义一个测试函数，计算输入张量在指定维度上的softmax函数
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::softmax(inputs[0], dim);
    };

    // 针对每个设备执行测试反向传播
    ForEachDevice([&](const torch::Device& device) {
      // 调用测试函数，测试反向传播
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},  // 创建一个大小为5x3x4x2的随机张量
              torch::TensorOptions(torch::kFloat)  // 使用浮点数选项
                  .device(DefaultDevice())  // 设置默认设备
                  .requires_grad(true))},  // 设置需要计算梯度
          device,
          testfn,
          /*rtol=*/1e-3,  // 相对误差容忍度
          /*atol=*/1e-4);  // 绝对误差容忍度
    });
  }
}

TEST_F(LazyOpsTest, TestSoftplusBackward) {
  // 定义一个测试函数，计算输入张量的softplus函数
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softplus(inputs[0]);
  };
  // 针对每个设备执行测试反向传播
  ForEachDevice([&](const torch::Device& device) {
    // 调用测试函数，测试反向传播
    TestBackward(
        {torch::rand(  // 使用 torch 库生成一个随机张量，形状为 {2, 1, 4, 6}
            {2, 1, 4, 6},  // 张量的形状参数
            torch::TensorOptions(torch::kFloat)  // 设置张量的选项，使用浮点数类型
                .device(DefaultDevice())  // 将张量放在默认设备上
                .requires_grad(true))},  // 设置张量需要计算梯度
        device,  // 传入设备参数
        testfn,  // 传入测试函数
        /*rtol=*/1e-4);  // 设置相对容差为 1e-4，用于测试
  });
TEST_F(LazyOpsTest, TestGeluBackward) {
  // 定义一个 lambda 函数 testfn，接受一个输入向量的数组并返回 GELU 激活函数的输出张量
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    // 调用 PyTorch 的 GELU 激活函数并返回结果张量
    return torch::gelu(inputs[0]);
  };
  // 对每个设备执行测试
  ForEachDevice([&](const torch::Device& device) {
    // 调用 TestBackward 函数进行反向传播测试
    TestBackward(
        // 提供一个随机生成的张量作为输入
        {torch::randn(
            {100},  // 张量的形状是 100
            torch::TensorOptions(torch::kFloat)  // 数据类型为浮点型
                .device(DefaultDevice())  // 使用默认设备
                .requires_grad(true))},  // 设置需要梯度计算
        device,  // 指定设备
        testfn);  // 传入定义的测试函数
  });
}
    return torch::gelu(inputs[0]);
  };


  # 返回输入张量的 GELU（Gaussian Error Linear Unit）函数的计算结果
  return torch::gelu(inputs[0]);
  # 这里使用了 Torch 深度学习框架中的 GELU 函数，对输入张量的第一个元素进行计算并返回结果
  # GELU 函数是一种非线性激活函数，常用于神经网络的隐藏层

  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 3},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });


  # 对每个设备执行以下操作：使用随机张量进行反向传播测试
  ForEachDevice([&](const torch::Device& device) {
    # 定义测试函数，用于反向传播测试
    TestBackward(
        # 使用随机生成的张量作为输入，形状为 [2, 3]，数据类型为浮点数
        {torch::rand(
            {2, 3},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        # 指定当前设备
        device,
        # 使用之前定义的测试函数进行测试
        testfn);
  });
  # 预期计数器 "lazy::gelu_backward" 的变化情况与忽略的计数器（GetIgnoredCounters()）进行比较


  # 检查 "lazy::gelu_backward" 计数器是否有变化，并与忽略的计数器进行比较
  ExpectCounterChanged("lazy::gelu_backward", GetIgnoredCounters());
  # 该断言用于检查 GELU 反向传播函数是否按预期被调用
  # ExpectCounterChanged 是一个函数或宏，用于检查特定计数器的变化情况
  # 它会比较实际计数器的值与预期值，GetIgnoredCounters() 可能用于获取被忽略的计数器的当前值
}

TEST_F(LazyOpsTest, TestLeakyReluBackward) {
  // 设置负斜率值
  double negative_slope = 0.01;
  // 定义测试函数，使用 lambda 表达式
  auto testfn = [=](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    // 调用 PyTorch 的 leaky_relu 函数
    return torch::leaky_relu(inputs[0], negative_slope);
  };
  // 对每个设备执行测试
  ForEachDevice([&](const torch::Device& device) {
    // 调用 TestBackward 函数进行反向传播测试
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestTransposeBackward) {
  // 定义测试函数，使用 lambda 表达式
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    // 调用 PyTorch 的转置操作
    return torch::t(inputs[0]);
  };
  // 对每个设备执行测试
  ForEachDevice([&](const torch::Device& device) {
    // 调用 TestBackward 函数进行反向传播测试
    TestBackward(
        {torch::rand(
            {2, 3},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestAddMatMulBackward) {
  // 设置输入通道数、输出通道数和标签数
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  // 测试 beta 不等于 1.0 的情况，通过 CPU 互操作进行
  for (double beta : {1., 2.}) {
    // 定义测试函数，使用 lambda 表达式
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      // 调用 PyTorch 的 addmm 函数，指定 beta 值
      return torch::addmm(inputs[0], inputs[1], inputs[2], /*beta=*/beta);
    };
    // 对每个设备执行测试
    ForEachDevice([&](const torch::Device& device) {
      // 调用 TestBackward 函数进行反向传播测试
      TestBackward(
          {torch::rand(
               {labels},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {in_channels, out_channels},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {out_channels, labels},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestBinaryCrossEntropyBackward) {
  // 设置批量大小和类别数
  int batch = 6;
  int classes = 2;
  // 循环处理每种数据类型
  for (auto dtype : {torch::kFloat}) {
    // 针对不同的 def_weight 值（false 和 true），循环执行以下操作
    for (bool def_weight : {false, true}) {
      // 创建一个随机初始化的张量 input，形状为 {batch, classes}，数据类型为 dtype，允许梯度计算
      torch::Tensor input = torch::rand(
          {batch, classes}, torch::TensorOptions(dtype).requires_grad(true));
      // 创建一个随机初始化的张量 target，形状与 input 相同，数据类型为 dtype
      torch::Tensor target =
          torch::rand({batch, classes}, torch::TensorOptions(dtype));
      // 声明一个张量 weight
      torch::Tensor weight;
      // 如果 def_weight 为 true，则创建一个随机初始化的张量 weight，形状与 input 相同，数据类型为 dtype
      if (def_weight) {
        weight = torch::rand({batch, classes}, torch::TensorOptions(dtype));
      }
      // 针对三种不同的 reduction 类型（Mean、Sum、None），循环执行以下操作
      for (torch::Reduction::Reduction reduction :
           {torch::Reduction::Mean,
            torch::Reduction::Sum,
            torch::Reduction::None}) {
        // 定义一个 lambda 函数 testfn，接受一个输入张量列表，返回一个经过二元交叉熵计算的张量
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::binary_cross_entropy(
              /*self=*/inputs[0],
              /*target=*/inputs[1],
              /*weight=*/inputs[2],
              /*reduction=*/reduction);
        };
        // 对每个设备执行 ForEachDevice 函数，传入参数包括 input、target、weight，以及当前设备 device
        ForEachDevice([&](const torch::Device& device) {
          // 调用 TestBackward 函数，传入参数为 {input, target, weight}，当前设备 device，lambda 函数 testfn，以及指定的相对误差容限 rtol 和绝对误差容限 atol
          TestBackward(
              {input, target, weight},
              device,
              testfn,
              /*rtol=*/1e-4,
              /*atol=*/1e-7);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLossBackward) {
  // TODO(whc) debug divide-by-zero failure under ASAN
  GTEST_SKIP();  // 跳过当前测试用例，因为在 ASAN 下存在除零错误

  int batch = 6;  // 定义批次大小为 6
  int classes = 2;  // 定义类别数为 2
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {  // 迭代处理数据类型为 torch::kFloat 的情况
    for (int ignore_index : {-1, 0, 1, 5}) {  // 遍历不同的 ignore_index 值
      for (bool def_weight : {false, true}) {  // 遍历是否定义权重的情况
        torch::Tensor input = torch::rand(
            {batch, classes},  // 生成指定形状的随机张量作为输入
            torch::TensorOptions(dtype)
                .device(DefaultDevice())
                .requires_grad(true));  // 指定数据类型、设备，并启用梯度追踪
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0),  // 生成随机整数张量作为目标值
            classes,
            {batch},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));  // 指定数据类型、设备
        torch::Tensor weight;
        if (def_weight) {  // 如果定义了权重
          weight = torch::rand(
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));  // 生成指定形状的随机权重张量
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean,
              torch::Reduction::Sum,
              torch::Reduction::None}) {  // 遍历不同的损失函数降维方式
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {  // 定义测试函数，接受输入张量并返回计算结果
            return torch::nll_loss(
                /*self=*/inputs[0],  // 输入张量
                /*target=*/inputs[1],  // 目标张量
                /*weight=*/inputs[2],  // 权重张量
                /*reduction=*/reduction,  // 损失函数的降维方式
                /*ignore_index=*/ignore_index);  // 忽略的索引值
          };
          ForEachDevice([&](const torch::Device& device) {  // 遍历不同的设备
            TestBackward(
                {input, target, weight},  // 输入的张量列表
                device,
                testfn,  // 测试函数
                /*rtol=*/1e-5,  // 相对误差容差
                /*atol=*/1e-8);  // 绝对误差容差
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss2dBackward) {
  int batch = 6;  // 定义批次大小为 6
  int classes = 2;  // 定义类别数为 2
  int height = 3;  // 定义高度为 3
  int width = 3;  // 定义宽度为 3
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {  // 迭代处理数据类型为 torch::kFloat 的情况
    // 遍历四个特定的 ignore_index 值：-1, 0, 1, 5
    for (int ignore_index : {-1, 0, 1, 5}) {
      // 遍历两种 def_weight 值：false 和 true
      for (bool def_weight : {false, true}) {
        // 创建一个随机张量作为输入，形状为 {batch, classes, height, width}
        torch::Tensor input = torch::rand(
            {batch, classes, height, width},
            torch::TensorOptions(dtype)
                .device(DefaultDevice())
                .requires_grad(true));
        // 创建一个随机整数张量作为目标，形状为 {batch, height, width}，数值在 [std::min(ignore_index, 0), classes) 范围内
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0),
            classes,
            {batch, height, width},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        // 定义权重张量
        torch::Tensor weight;
        // 如果 def_weight 为 true，则创建一个形状为 {classes} 的随机张量作为权重
        if (def_weight) {
          weight = torch::rand(
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));
        }
        // 遍历三种 reduction 模式：Mean, Sum, None
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean,
              torch::Reduction::Sum,
              torch::Reduction::None}) {
          // 定义一个测试函数，计算 NLL loss
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::nll_loss2d(
                /*self=*/inputs[0],        // 输入数据
                /*target=*/inputs[1],      // 目标数据
                /*weight=*/inputs[2],      // 权重数据
                /*reduction=*/reduction,   // 损失函数的缩减模式
                /*ignore_index=*/ignore_index);  // 忽略索引值
          };
          // 对每个设备执行测试
          ForEachDevice([&](const torch::Device& device) {
            // 调用反向传播测试函数 TestBackward
            TestBackward(
                {input, target, weight},    // 输入参数
                device,                     // 设备
                testfn,
                /*rtol=*/1e-5,              // 相对误差容限
                /*atol=*/1e-8);             // 绝对误差容限
          });
        }
      }
    }
}

// 定义一个测试用例，测试 smooth_l1_loss 函数的反向传播
TEST_F(LazyOpsTest, TestSmoothL1LossBackward) {
  // 创建一个随机张量作为输入，设定其为浮点型数据类型，开启梯度计算
  torch::Tensor input = torch::randn(
      {2, 4},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  // 创建一个随机张量作为目标，设定其为浮点型数据类型，放置在默认设备上
  torch::Tensor target = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 针对不同的减少方式（None、Mean、Sum），执行测试
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    // 针对不同的 beta 值（0.25、1.0），定义测试函数 testfn
    for (double beta : {0.25, 1.}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        // 调用 smooth_l1_loss 函数计算 Smooth L1 损失
        return torch::smooth_l1_loss(
            /*input=*/inputs[0],
            /*target=*/inputs[1],
            /*reduction=*/reduction,
            /*beta=*/beta);
      };
      // 对每个设备执行测试，验证反向传播
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, target},
            device,
            testfn,
            /*rtol=*/1e-5,
            /*atol=*/1e-8);
      });
    }
  }
}

// 定义一个测试用例，测试 view 函数的反向传播
TEST_F(LazyOpsTest, TestViewBackward) {
  // 定义测试函数 testfn，用于执行 view 操作
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return inputs[0].view({-1, 320});
  };
  // 对每个设备执行测试，验证反向传播
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {32, 20, 4, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

// 定义一个测试用例，测试 batch_norm 函数的反向传播
TEST_F(LazyOpsTest, TestBatchNorm2DBackward) {
  // 定义 batch_norm 函数所需的参数：momentum 和 eps
  double momentum = 0.1;
  double eps = 0.5;
  // 定义测试函数 testfn，用于执行 batch_norm 操作
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0],
        /*weight=*/inputs[1],
        /*bias=*/inputs[2],
        /*running_mean=*/inputs[3],
        /*running_var=*/inputs[4],
        /*training=*/true,
        /*momentum=*/momentum,
        /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  // 定义 num_features 变量，表示特征的数量
  int num_features = 3;
  // 定义未定义张量
  torch::Tensor undef;
  // 针对未定义权重和偏置的情况，执行测试
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      // 使用 ForEachDevice 函数遍历每个设备，device 是当前设备的引用
      
      torch::Tensor input = torch::rand(
          {2, num_features, 4, 4},  // 创建一个形状为 [2, num_features, 4, 4] 的随机张量
          torch::TensorOptions(torch::kFloat)  // 指定张量数据类型为浮点数
              .device(DefaultDevice())  // 设置张量的设备为默认设备
              .requires_grad(true));  // 声明张量需要计算梯度
      
      torch::Tensor weight = undef_weight_bias
          ? undef  // 如果 undef_weight_bias 为 true，则 weight 为未定义状态
          : torch::rand(
                {num_features},  // 创建一个形状为 [num_features] 的随机张量
                torch::TensorOptions(torch::kFloat)  // 指定张量数据类型为浮点数
                    .device(DefaultDevice())  // 设置张量的设备为默认设备
                    .requires_grad(true));  // 声明张量需要计算梯度
      
      torch::Tensor bias = undef_weight_bias
          ? undef  // 如果 undef_weight_bias 为 true，则 bias 为未定义状态
          : torch::rand(
                {num_features},  // 创建一个形状为 [num_features] 的随机张量
                torch::TensorOptions(torch::kFloat)  // 指定张量数据类型为浮点数
                    .device(DefaultDevice())  // 设置张量的设备为默认设备
                    .requires_grad(true));  // 声明张量需要计算梯度
      
      torch::Tensor running_mean = torch::zeros(
          {num_features},  // 创建一个形状为 [num_features] 的零张量
          torch::TensorOptions(torch::kFloat)  // 指定张量数据类型为浮点数
              .device(DefaultDevice()));  // 设置张量的设备为默认设备
      
      torch::Tensor running_var = torch::ones(
          {num_features},  // 创建一个形状为 [num_features] 的全一张量
          torch::TensorOptions(torch::kFloat)  // 指定张量数据类型为浮点数
              .device(DefaultDevice()));  // 设置张量的设备为默认设备
      
      TestBackward(
          {input, weight, bias, running_mean, running_var},  // 将创建的张量作为参数传递给 TestBackward 函数
          device,  // 当前设备
          testfn,  // 测试函数
          /*rtol=*/1e-3,  // 相对误差容限
          /*atol=*/1e-4);  // 绝对误差容限
    });
  }
}

// 定义一个测试用例，测试三维批归一化的反向传播
TEST_F(LazyOpsTest, TestBatchNorm3DBackward) {
  // 设置动量和 epsilon 参数
  double momentum = 0.1;
  double eps = 0.5;
  // 定义一个 Lambda 函数，接受输入向量并返回批归一化后的结果张量
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0],    // 输入张量
        /*weight=*/inputs[1],   // 权重张量
        /*bias=*/inputs[2],     // 偏置张量
        /*running_mean=*/inputs[3],   // 运行时均值张量
        /*running_var=*/inputs[4],    // 运行时方差张量
        /*training=*/true,      // 是否处于训练模式
        /*momentum=*/momentum,  // 动量值
        /*eps=*/eps,            // epsilon 值
        /*cudnn_enabled=*/false);   // 是否启用 cuDNN 加速
  };
  // 定义特征数量
  int num_features = 3;
  // 定义未定义的张量
  torch::Tensor undef;
  // 遍历 undef_weight_bias 变量的两种布尔值
  for (bool undef_weight_bias : {false, true}) {
    // 对每一个设备执行以下操作
    ForEachDevice([&](const torch::Device& device) {
      // 生成随机输入张量
      torch::Tensor input = torch::rand(
          {2, num_features, 4, 4, 2},   // 输入张量的形状
          torch::TensorOptions(torch::kFloat)   // 张量选项，指定为浮点数张量
              .device(DefaultDevice())   // 指定设备为默认设备
              .requires_grad(true));     // 指定需要计算梯度
      // 如果 undef_weight_bias 为假，则生成随机权重张量，否则为未定义张量
      torch::Tensor weight = undef_weight_bias
          ? undef
          : torch::rand(
                {num_features},   // 权重张量的形状
                torch::TensorOptions(torch::kFloat)   // 张量选项，指定为浮点数张量
                    .device(DefaultDevice())   // 指定设备为默认设备
                    .requires_grad(true));   // 指定需要计算梯度
      // 如果 undef_weight_bias 为假，则生成随机偏置张量，否则为未定义张量
      torch::Tensor bias = undef_weight_bias
          ? undef
          : torch::rand(
                {num_features},   // 偏置张量的形状
                torch::TensorOptions(torch::kFloat)   // 张量选项，指定为浮点数张量
                    .device(DefaultDevice())   // 指定设备为默认设备
                    .requires_grad(true));   // 指定需要计算梯度
      // 创建全零的运行时均值张量
      torch::Tensor running_mean = torch::zeros(
          {num_features},   // 张量形状
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));   // 张量选项，指定为浮点数张量，并设定设备为默认设备
      // 创建全一的运行时方差张量
      torch::Tensor running_var = torch::ones(
          {num_features},   // 张量形状
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));   // 张量选项，指定为浮点数张量，并设定设备为默认设备
      // 执行反向传播测试
      TestBackward(
          {input, weight, bias, running_mean, running_var},   // 输入参数列表
          device,   // 设备
          testfn,   // 测试函数
          /*rtol=*/1e-3,   // 相对误差容差
          /*atol=*/1e-3);   // 绝对误差容差
    });
  }
}

// 定义一个测试用例，测试带 logits 的二元交叉熵的反向传播
TEST_F(LazyOpsTest, TestBCEWithLogitsBackward) {
  // 定义批次大小和类别数量
  int batch = 10;
  int classes = 5;
  // 定义未定义的张量
  torch::Tensor undef;
  // 遍历不同的减少(reduction)模式
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    // 定义测试函数，接受输入向量并返回带 logits 的二元交叉熵损失张量
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::binary_cross_entropy_with_logits(
          /*input=*/inputs[0],    // 输入 logits 张量
          /*target=*/inputs[1],   // 目标张量
          /*weight=*/inputs[2],   // 权重张量
          /*pos_weight=*/inputs[3],   // 正例权重张量
          /*reduction=*/reduction);   // 减少(reduction)模式
    };
    // 遍历两个布尔变量的组合，表示是否存在未定义的权重和正权重
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        // 创建一个随机张量作为输入，形状为(batch, classes)，浮点类型，需要计算梯度
        torch::Tensor input = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        // 创建一个随机张量作为目标，形状同输入，浮点类型，需要计算梯度
        torch::Tensor target = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        // 根据undef_weight确定是否使用未定义的权重张量或者创建随机的权重张量
        torch::Tensor weight = undef_weight
            ? undef  // 未定义权重
            : torch::rand(
                  {classes},
                  torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        // 根据undef_pos_weight确定是否使用未定义的正权重张量或者创建随机的正权重张量
        torch::Tensor pos_weight = undef_pos_weight
            ? undef  // 未定义正权重
            : torch::rand(
                  {classes},
                  torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        // 对每一个设备执行以下操作，lambda函数通过捕获列表引用外部变量
        ForEachDevice([&](const torch::Device& device) {
          // 调用测试反向传播函数，传入输入、目标、权重和正权重张量以及设备
          TestBackward(
              {input, target, weight, pos_weight},
              device,
              testfn,
              /*rtol=*/1e-3,  // 相对误差容限
              /*atol=*/1e-5);  // 绝对误差容限
        });
      }
    }
}

TEST_F(LazyOpsTest, TestKlDivBackward) {
  // 创建一个随机张量作为输入，形状为[4, 3]，数据类型为浮点型，需要梯度计算
  torch::Tensor input = torch::rand(
      {4, 3},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())  // 指定默认设备
          .requires_grad(true));     // 设置需要计算梯度

  // 创建一个随机张量作为目标，形状同样为[4, 3]，数据类型为浮点型，需要梯度计算
  torch::Tensor target = torch::rand(
      {4, 3},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())  // 指定默认设备
          .requires_grad(true));     // 设置需要计算梯度

  // 针对三种不同的约简方式（Mean、Sum、None），分别进行测试
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean,
        torch::Reduction::Sum,
        torch::Reduction::None}) {
    
    // 定义一个测试函数，接受一个输入张量列表，返回 KL 散度计算结果
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      // 调用 PyTorch 提供的 KL 散度计算函数
      return torch::kl_div(/*self=*/inputs[0], /*target=*/inputs[1], reduction);
    };

    // 针对每个设备，执行测试函数
    ForEachDevice([&](const torch::Device& device) {
      // 调用测试函数，测试输入和目标张量在指定设备上的梯度计算
      TestBackward(
          {input, target},     // 输入参数为 input 和 target 张量
          device,              // 当前设备
          testfn,              // 测试函数 testfn
          /*rtol=*/1e-4,       // 相对误差容忍度
          /*atol=*/1e-5);      // 绝对误差容忍度
    });
  }
}

TEST_F(LazyOpsTest, TestEmbeddingBackward) {
  int num_weights = 32;

  // 对于每个 padding_idx，从 -1 到 num_weights-1 进行迭代
  for (int padding_idx = -1; padding_idx < num_weights; ++padding_idx) {

    // 对于每种 scale_grad_by_freq 的取值（false 和 true），分别进行测试
    for (bool scale_grad_by_freq : {false, true}) {

      // 定义一个测试函数，接受一个输入张量列表，返回 embedding 的结果
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        // 调用 PyTorch 提供的 embedding 函数进行嵌入操作
        return torch::embedding(
            inputs[0],
            inputs[1],
            /*padding_idx=*/padding_idx,          // 设置 padding_idx 参数
            /*scale_grad_by_freq=*/scale_grad_by_freq,  // 设置 scale_grad_by_freq 参数
            /*sparse=*/false);                    // 设置 sparse 参数为 false
      };

      // 针对每个设备，执行测试函数
      ForEachDevice([&](const torch::Device& device) {
        // 创建一个随机权重张量，形状为 [num_weights, 7]，数据类型为浮点型，需要梯度计算
        torch::Tensor weight = torch::rand(
            {num_weights, 7},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));

        // 创建一个随机索引张量，形状为 [3, 9, 4]，数据类型为长整型，位于默认设备上
        torch::Tensor indices = torch::randint(
            num_weights,
            {3, 9, 4},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));

        // 调用测试函数，测试权重和索引张量在指定设备上的梯度计算
        TestBackward(
            {weight, indices},  // 输入参数为 weight 和 indices 张量
            device,             // 当前设备
            testfn,             // 测试函数 testfn
            /*rtol=*/1e-5,      // 相对误差容忍度
            /*atol=*/1e-8);     // 绝对误差容忍度
      });
    }
  }
}

TEST_F(LazyOpsTest, TestAmpForeachNonFiniteCheckAndUnscale) {
  if (IsCuda()) {
    // TODO(whc) debug failure on cuda
    GTEST_SKIP();
  }

  // 创建包含四个元素的张量 grads0，元素为 {1, 2, 3, 4}，数据类型为 float，设备为默认设备
  torch::Tensor grads0 = torch::tensor(
      {1, 2, 3, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 创建包含四个元素的张量 grads1，元素为 {1.0, 2.0, std::nan("1"), 4.0}，数据类型为 float，设备为默认设备
  torch::Tensor grads1 = torch::tensor(
      {1.0, 2.0, std::nan("1"), 4.0},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 创建标量张量 inv_scale，值为 0.2，数据类型为 float，设备为默认设备
  torch::Tensor inv_scale = torch::scalar_tensor(
      0.2, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 创建标量张量 found_inf，值为 0，数据类型为 float，设备为默认设备
  torch::Tensor found_inf = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 计算 grads0 乘以 inv_scale 的张量
  torch::Tensor grads_output0 = grads0 * inv_scale;
  
  // 创建标量张量 found_inf_output0，值为 0，数据类型为 float，设备为默认设备
  torch::Tensor found_inf_output0 = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 创建标量张量 found_inf_output1，值为 1，数据类型为 float，设备为默认设备
  torch::Tensor found_inf_output1 = torch::scalar_tensor(
      1, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 如果 grads0 的设备是 CPU，则跳过当前测试
    if (grads0.device() == at::kCPU) {
      GTEST_SKIP();
    }
    // 将 grads0 复制到指定设备上得到 lazy_grads0
    torch::Tensor lazy_grads0 = CopyToDevice(grads0, device);
    
    // 将 inv_scale 复制到指定设备上得到 lazy_inv_scale
    torch::Tensor lazy_inv_scale = CopyToDevice(inv_scale, device);
    
    // 将 found_inf 复制到指定设备上得到 lazy_found_inf
    torch::Tensor lazy_found_inf = CopyToDevice(found_inf, device);
    
    // 在指定设备上执行非有限检查并按比例缩放操作
    torch::_amp_foreach_non_finite_check_and_unscale_(
        lazy_grads0, lazy_found_inf, lazy_inv_scale);
    
    // 检查 lazy_grads0 是否与 grads_output0 在指定误差范围内相等
    AllClose(grads_output0, lazy_grads0, /*rtol=*/1e-2, /*atol=*/1e-4);
    
    // 检查 lazy_found_inf 是否与 found_inf_output0 相等
    AllEqual(found_inf_output0, lazy_found_inf);

    // 将 grads1 复制到指定设备上得到 lazy_grads1
    torch::Tensor lazy_grads1 = CopyToDevice(grads1, device);
    
    // 在指定设备上执行非有限检查并按比例缩放操作
    torch::_amp_foreach_non_finite_check_and_unscale_(
        lazy_grads1, lazy_found_inf, lazy_inv_scale);
    
    // 检查 lazy_found_inf 是否与 found_inf_output1 相等
    AllEqual(found_inf_output1, lazy_found_inf);
  });
}

TEST_F(LazyOpsTest, TestAmpUpdateScale) {
  // 初始化一个整数张量，用于跟踪增长次数，设备为默认设备
  torch::Tensor growth_tracker = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  // 初始化一个浮点数张量，表示当前的缩放比例，设备为默认设备
  torch::Tensor current_scale = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 初始化一个浮点数张量，表示找到无穷大值的标志，设备为默认设备
  torch::Tensor found_inf = torch::scalar_tensor(
      1, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 初始化一个浮点数张量，表示未找到无穷大值的标志，设备为默认设备
  torch::Tensor not_found_inf = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 缩放增长因子，设为2.0
  float scale_growth_factor = 2.0;
  // 缩放回退因子，设为0.5
  float scale_backoff_factor = 0.5;
  // 增长间隔，设为3
  int growth_interval = 3;

  // 初始化预期的增长跟踪结果和当前缩放结果张量
  torch::Tensor growth_tracker_result0 = torch::scalar_tensor(
      1, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result0 = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor growth_tracker_result1 = torch::scalar_tensor(
      2, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result1 = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor growth_tracker_result2 = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result2 = torch::scalar_tensor(
      8, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor growth_tracker_result3 = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result3 = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

  // 对每一个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 如果增长跟踪张量在CPU设备上，跳过当前测试
    if (growth_tracker.device() == at::kCPU) {
      GTEST_SKIP();
    }
    // 将各张量复制到指定设备上
    torch::Tensor lazy_growth_tracker = CopyToDevice(growth_tracker, device);
    torch::Tensor lazy_current_scale = CopyToDevice(current_scale, device);
    torch::Tensor lazy_found_inf = CopyToDevice(found_inf, device);
    torch::Tensor lazy_not_found_inf = CopyToDevice(not_found_inf, device);

    // 调用 _amp_update_scale_ 函数进行缩放更新操作
    torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    // 检查当前缩放结果是否与预期结果0接近
    AllClose(
        current_scale_result0,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    // 检查增长跟踪结果是否与预期结果0相等
    AllEqual(growth_tracker_result0, lazy_growth_tracker);

    // 再次调用 _amp_update_scale_ 函数进行缩放更新操作
    torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    // 检查当前缩放结果是否与预期结果1接近
    AllClose(
        current_scale_result1,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    // 检查增长跟踪结果是否与预期结果1相等

    // torch::_amp_update_scale_ 返回当前缩放的引用
    # 使用 torch::_amp_update_scale_ 函数更新懒惰（lazy）版本的当前缩放因子，考虑增长跟踪器、未找到的无穷值、缩放增长因子、缩放回退因子和增长间隔参数
    lazy_current_scale = torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    
    # 检查 current_scale_result2 与 lazy_current_scale 的近似相等性，相对容差为 1e-2，绝对容差为 1e-4
    AllClose(
        current_scale_result2,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    
    # 检查 growth_tracker_result2 与 lazy_growth_tracker 的完全相等性
    AllEqual(growth_tracker_result2, lazy_growth_tracker);
    
    # 使用 torch::_amp_update_scale_ 函数更新懒惰（lazy）版本的当前缩放因子，考虑增长跟踪器、找到的无穷值、缩放增长因子、缩放回退因子和增长间隔参数
    lazy_current_scale = torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    
    # 检查 current_scale_result3 与 lazy_current_scale 的近似相等性，相对容差为 1e-2，绝对容差为 1e-4
    AllClose(
        current_scale_result3,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    
    # 检查 growth_tracker_result3 与 lazy_growth_tracker 的完全相等性
    AllEqual(growth_tracker_result3, lazy_growth_tracker);
    });
    # 验证在匿名函数（lambda）中没有发生 "aten::.*" 计数器的变化
    ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    
    # 验证 "lazy::_amp_update_scale_" 计数器在匿名函数（lambda）中发生了变化
    ExpectCounterChanged("lazy::_amp_update_scale_", GetIgnoredCounters());
}

// 定义测试用例 TestEarlySyncLiveTensors，测试懒惰操作中同步实时张量
TEST_F(LazyOpsTest, TestEarlySyncLiveTensors) {
  // 创建一个标量张量，并指定其浮点数值和设备选项
  torch::Tensor scalar_tensor = torch::scalar_tensor(
      1., torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 提取标量张量的值作为标量
  torch::Scalar scalar1 = scalar_tensor.item();
  
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将标量张量复制到指定设备上
    torch::Tensor lazy_scalar_tensor = CopyToDevice(scalar_tensor, device);
    // 提取懒惰标量张量的值作为标量
    torch::Scalar scalar2 = lazy_scalar_tensor.item();
    // 断言两个标量的浮点值相等
    ASSERT_EQ(scalar1.to<float>(), scalar2.to<float>());
  });
  
  // 如果实验开启了 "early_sync"
  if (DebugUtil::ExperimentEnabled("early_sync")) {
    // 期望 "EarlySyncLiveTensorsCount" 计数器已改变
    ExpectCounterChanged("EarlySyncLiveTensorsCount", GetIgnoredCounters());
  } else {
    // 期望 "EarlySyncLiveTensorsCount" 计数器未改变
    ExpectCounterNotChanged("EarlySyncLiveTensorsCount", GetIgnoredCounters());
  }
  // 期望 "aten::_local_scalar_dense" 计数器已改变
  ExpectCounterChanged("aten::_local_scalar_dense", GetIgnoredCounters());
}

// 定义测试用例 TestLerp，测试线性插值操作
TEST_F(LazyOpsTest, TestLerp) {
  // 创建随机起始张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机结束张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机权重张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 执行线性插值操作，生成结果张量
  torch::Tensor res = torch::lerp(start, end, weight);
  
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将起始张量复制到指定设备上
    torch::Tensor lazy_start = CopyToDevice(start, device);
    // 将结束张量复制到指定设备上
    torch::Tensor lazy_end = CopyToDevice(end, device);
    // 将权重张量复制到指定设备上
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    // 执行懒惰版本的线性插值操作，生成懒惰结果张量
    torch::Tensor lazy_res = torch::lerp(lazy_start, lazy_end, lazy_weight);
    // 断言结果张量与懒惰结果张量在给定设备上近似相等
    AllClose(res, lazy_res);
  });
  
  // 期望 "aten::.*" 计数器未改变
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 期望 "lazy::lerp" 计数器已改变
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

// 定义测试用例 TestLerpScalar，测试标量线性插值操作
TEST_F(LazyOpsTest, TestLerpScalar) {
  // 创建随机起始张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机结束张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建标量权重，指定其值为 3.0
  torch::Scalar weight = torch::Scalar(3.0);
  // 执行标量线性插值操作，生成结果张量
  torch::Tensor res = torch::lerp(start, end, weight);
  
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将起始张量复制到指定设备上
    torch::Tensor lazy_start = CopyToDevice(start, device);
    // 将结束张量复制到指定设备上
    torch::Tensor lazy_end = CopyToDevice(end, device);
    // 执行懒惰版本的线性插值操作，使用相同的标量权重
    torch::Tensor lazy_res = torch::lerp(lazy_start, lazy_end, weight);
    // 断言结果张量与懒惰结果张量在给定设备上近似相等
    AllClose(res, lazy_res);
  });
  
  // 期望 "aten::.*" 计数器未改变
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 期望 "lazy::lerp" 计数器已改变
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

// 定义测试用例 TestLerpInplace，测试原地线性插值操作
TEST_F(LazyOpsTest, TestLerpInplace) {
  // 创建随机输入张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor input = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机结束张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建随机权重张量，指定形状、浮点数类型和默认设备选项
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 复制输入张量以备份
  torch::Tensor input_copy = input.clone();
  // 在原地执行线性插值操作
  input.lerp_(end, weight);
  
  // 对每个设备执行操作
  ForEachDevice([&](const torch::Device& device) {
    // 将备份的输入张量复制到指定设备上
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
  
    // 将 `end` 张量复制到指定设备上，并返回复制后的张量
    torch::Tensor lazy_end = CopyToDevice(end, device);
    
    // 将 `weight` 张量复制到指定设备上，并返回复制后的张量
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    
    // 使用线性插值函数 `lerp_` 将 `lazy_input` 张量从 `lazy_end` 到 `lazy_weight` 进行插值操作
    lazy_input.lerp_(lazy_end, lazy_weight);
    
    // 检查 `lazy_input` 是否与 `input` 张量在数值上近似相等
    AllClose(lazy_input, input);
  });
  
  // 预期未改变的计数器 `aten::*` 是否未发生变化，获取当前忽略的计数器列表
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  
  // 预期 `lazy::lerp` 计数器是否发生变化，获取当前忽略的计数器列表
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
TEST_F(LazyOpsTest, TestLerpScalarInplace) {
  // 创建一个大小为 [3, 4] 的随机张量 `input`，使用默认设备上的浮点数选项
  torch::Tensor input = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与 `input` 相同大小的随机张量 `end`
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量 `weight`，值为 3.0
  torch::Scalar weight = torch::Scalar(3.0);
  // 克隆 `input`，得到 `input_copy`
  torch::Tensor input_copy = input.clone();
  // 将 `input` 在原地进行线性插值操作，结果存回 `input`
  input.lerp_(end, weight);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 `input_copy` 复制到指定设备上，得到 `lazy_input`
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    // 将 `end` 复制到指定设备上，得到 `lazy_end`
    torch::Tensor lazy_end = CopyToDevice(end, device);
    // 在 `lazy_input` 上进行线性插值操作，结果存回 `lazy_input`
    lazy_input.lerp_(lazy_end, weight);
    // 验证 `lazy_input` 是否与 `input` 在指定设备上的数据一致
    AllClose(lazy_input, input);
  });
  // 验证在计数器中是否没有改变与 'aten::*' 相关的操作计数
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 验证在计数器中是否改变了 'lazy::lerp' 操作的计数
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpOut) {
  // 创建一个大小为 [3, 4] 的随机张量 `start`，使用默认设备上的浮点数选项
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与 `start` 相同大小的随机张量 `end`
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与 `start` 相同大小的随机张量 `weight`
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与 `start` 相同大小的空张量 `res`
  torch::Tensor res = torch::empty(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在 `res` 上进行线性插值操作，将 `start` 和 `end` 按 `weight` 混合后的结果存入 `res`
  torch::lerp_out(res, start, end, weight);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 `start` 复制到指定设备上，得到 `lazy_start`
    torch::Tensor lazy_start = CopyToDevice(start, device);
    // 将 `end` 复制到指定设备上，得到 `lazy_end`
    torch::Tensor lazy_end = CopyToDevice(end, device);
    // 将 `weight` 复制到指定设备上，得到 `lazy_weight`
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    // 创建一个与 `lazy_start` 相同大小的空张量 `lazy_res`
    torch::Tensor lazy_res = torch::empty({3, 4}, lazy_start.options());
    // 在 `lazy_res` 上进行线性插值操作，将 `lazy_start` 和 `lazy_end` 按 `lazy_weight` 混合后的结果存入 `lazy_res`
    torch::lerp_out(lazy_res, lazy_start, lazy_end, lazy_weight);
    // 验证 `res` 和 `lazy_res` 在指定设备上的数据是否一致
    AllClose(res, lazy_res);
  });
  // 验证在计数器中是否没有改变与 'aten::*' 相关的操作计数
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 验证在计数器中是否改变了 'lazy::lerp' 操作的计数
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpScalarOut) {
  // 创建一个大小为 [3, 4] 的随机张量 `start`，使用默认设备上的浮点数选项
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个与 `start` 相同大小的随机张量 `end`
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个标量 `weight`，值为 3.0
  torch::Scalar weight = torch::Scalar(3.0);
  // 创建一个与 `start` 相同大小的空张量 `res`
  torch::Tensor res = torch::empty(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 在 `res` 上进行线性插值操作，将 `start` 和 `end` 按 `weight` 混合后的结果存入 `res`
  torch::lerp_out(res, start, end, weight);
  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将 `start` 复制到指定设备上，得到 `lazy_start`
    torch::Tensor lazy_start = CopyToDevice(start, device);
    // 将 `end` 复制到指定设备上，得到 `lazy_end`
    torch::Tensor lazy_end = CopyToDevice(end, device);
    // 创建一个与 `lazy_start` 相同大小的空张量 `lazy_res`
    torch::Tensor lazy_res = torch::empty({3, 4}, lazy_start.options());
    // 在 `lazy_res` 上进行线性插值操作，将 `lazy_start` 和 `lazy_end` 按 `weight` 混合后的结果存入 `lazy_res`
    torch::lerp_out(lazy_res, lazy_start, lazy_end, weight);
    // 验证 `res` 和 `lazy_res` 在指定设备上的数据是否一致
    AllClose(res, lazy_res);
  });
  // 验证在计数器中是否没有改变与 'aten::*' 相关的操作计数
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  // 验证在计数器中是否改变了 'lazy::lerp' 操作的计数
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}
TEST_F(LazyOpsTest, IsAliasOf) {
  // 创建一个张量 a，形状为 [4]，使用默认设备的浮点张量选项
  auto a = torch::empty(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // 创建一个张量 b，形状为 [4]，使用默认设备的浮点张量选项
  auto b = torch::empty(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

  // 对每个设备执行以下操作
  ForEachDevice([&](const torch::Device& device) {
    // 将张量 a 复制到指定设备，返回一个延迟张量 lazy_a
    auto lazy_a = CopyToDevice(a, device);
    // 将张量 b 复制到指定设备，返回一个延迟张量 lazy_b
    auto lazy_b = CopyToDevice(b, device);
    // 断言：如果 a 不是 b 的别名，则 lazy_a 不是 lazy_b 的别名
    EXPECT_EQ(!a.is_alias_of(b), !lazy_a.is_alias_of(lazy_b));

    // 创建张量 c 作为 a 的视图，形状为 [2, 2]
    auto c = a.view({2, 2});
    // 创建延迟张量 lazy_c 作为 lazy_a 的视图，形状为 [2, 2]
    auto lazy_c = lazy_a.view({2, 2});
    // 断言：a 是否是 c 的别名，则 lazy_a 是否是 lazy_c 的别名
    EXPECT_EQ(a.is_alias_of(c), lazy_a.is_alias_of(lazy_c));

    // 创建张量 d 作为 c 的视图，形状为 [1, 4]
    auto d = c.view({1, 4});
    // 创建延迟张量 lazy_d 作为 lazy_c 的视图，形状为 [1, 4]
    auto lazy_d = lazy_c.view({1, 4});
    // 断言：d 是否是 c 的别名，则 lazy_d 是否是 lazy_c 的别名
    EXPECT_EQ(d.is_alias_of(c), lazy_d.is_alias_of(lazy_c));
    // 断言：d 是否是 a 的别名，则 lazy_d 是否是 lazy_a 的别名
    EXPECT_EQ(d.is_alias_of(a), lazy_d.is_alias_of(lazy_a));
  });
}

#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace torch
```