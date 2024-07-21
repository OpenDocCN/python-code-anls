# `.\pytorch\test\cpp\api\misc.cpp`

```py
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件

#include <torch/torch.h> // 包含 LibTorch 的头文件

#include <test/cpp/api/support.h> // 包含测试支持文件的头文件

#include <functional> // 包含功能库的头文件，用于支持函数式编程

using namespace torch::test; // 使用 torch::test 命名空间

void torch_warn_once_A() {
  TORCH_WARN_ONCE("warn once"); // 发出一次性警告信息 "warn once"
}

void torch_warn_once_B() {
  TORCH_WARN_ONCE("warn something else once"); // 发出一次性警告信息 "warn something else once"
}

void torch_warn() {
  TORCH_WARN("warn multiple times"); // 发出多次性警告信息 "warn multiple times"
}

TEST(UtilsTest, WarnOnce) {
  {
    WarningCapture warnings; // 捕获警告信息

    torch_warn_once_A(); // 调用函数发出一次性警告 "warn once"
    torch_warn_once_A(); // 再次调用函数发出一次性警告 "warn once"
    torch_warn_once_B(); // 调用函数发出一次性警告 "warn something else once"
    torch_warn_once_B(); // 再次调用函数发出一次性警告 "warn something else once"

    ASSERT_EQ(count_substr_occurrences(warnings.str(), "warn once"), 1); // 断言捕获的警告信息中包含 "warn once" 的次数为1
    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn something else once"),
        1); // 断言捕获的警告信息中包含 "warn something else once" 的次数为1
  }
  {
    WarningCapture warnings; // 捕获警告信息

    torch_warn(); // 多次调用函数发出警告 "warn multiple times"
    torch_warn(); // 多次调用函数发出警告 "warn multiple times"
    torch_warn(); // 多次调用函数发出警告 "warn multiple times"

    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn multiple times"), 3); // 断言捕获的警告信息中包含 "warn multiple times" 的次数为3
  }
}

TEST(NoGradTest, SetsGradModeCorrectly) {
  torch::manual_seed(0); // 设置随机种子为0
  torch::NoGradGuard guard; // 创建一个禁用梯度计算的守卫
  torch::nn::Linear model(5, 2); // 创建一个线性模型
  auto x = torch::randn({10, 5}, torch::requires_grad()); // 生成一个形状为 (10, 5) 的张量，并要求梯度
  auto y = model->forward(x); // 模型前向传播
  torch::Tensor s = y.sum(); // 对模型输出进行求和

  // 模拟 Python API 的行为：
  ASSERT_THROWS_WITH(
      s.backward(),
      "element 0 of tensors does not require grad and does not have a grad_fn"); // 断言在调用 backward() 时抛出特定异常信息
}

struct AutogradTest : torch::test::SeedingFixture {
  AutogradTest() {
    x = torch::randn({3, 3}, torch::requires_grad()); // 初始化一个形状为 (3, 3) 的张量，并要求梯度
    y = torch::randn({3, 3}); // 初始化一个形状为 (3, 3) 的张量
    z = x * y; // 对张量进行逐元素乘法
  }
  torch::Tensor x, y, z; // 定义三个张量 x, y, z
};

TEST_F(AutogradTest, CanTakeDerivatives) {
  z.backward(torch::ones_like(z)); // 对 z 进行反向传播，梯度为形状和 z 相同的全1张量
  ASSERT_TRUE(x.grad().allclose(y)); // 断言 x 的梯度接近于 y
}

TEST_F(AutogradTest, CanTakeDerivativesOfZeroDimTensors) {
  z.sum().backward(); // 对 z 的所有元素求和后进行反向传播
  ASSERT_TRUE(x.grad().allclose(y)); // 断言 x 的梯度接近于 y
}

TEST_F(AutogradTest, CanPassCustomGradientInputs) {
  z.sum().backward(torch::ones({}) * 2); // 对 z 的所有元素求和后进行反向传播，使用自定义的梯度输入（全1张量乘以2）
  ASSERT_TRUE(x.grad().allclose(y * 2)); // 断言 x 的梯度接近于 y 的两倍
}

TEST(UtilsTest, AmbiguousOperatorDefaults) {
  auto tmp = at::empty({}, at::kCPU); // 创建一个空的标量张量在 CPU 上
  at::_test_ambiguous_defaults(tmp); // 调用测试函数，传入空张量 tmp
  at::_test_ambiguous_defaults(tmp, 1); // 再次调用测试函数，传入两个参数
  at::_test_ambiguous_defaults(tmp, 1, 1); // 再次调用测试函数，传入三个参数
  at::_test_ambiguous_defaults(tmp, 2, "2"); // 再次调用测试函数，传入两个参数，其中一个为字符串
}

int64_t get_first_element(c10::OptionalIntArrayRef arr) {
  return arr.value()[0]; // 返回 OptionalIntArrayRef 对象的第一个元素
}

TEST(OptionalArrayRefTest, DanglingPointerFix) {
  // 确保 `OptionalArrayRef` 的转换构造函数在给定单个值时不会创建悬空指针
  ASSERT_TRUE(get_first_element(300) == 300); // 断言从单个值创建的 OptionalIntArrayRef 对象的第一个元素为 300
  ASSERT_TRUE(get_first_element({400}) == 400); // 断言从初始化列表创建的 OptionalIntArrayRef 对象的第一个元素为 400
}
```