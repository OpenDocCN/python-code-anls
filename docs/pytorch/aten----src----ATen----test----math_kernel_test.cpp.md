# `.\pytorch\aten\src\ATen\test\math_kernel_test.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <c10/util/irange.h>

using namespace at;

// 检查两个张量是否在指定的相对容差和绝对容差下全部相等
bool allClose(const at::Tensor& t1, const at::Tensor& t2, double rtol=1e-5, double atol=1e-8) {
  // 检查张量形状是否相同，如果不同则打印形状差异信息并返回 false
  if (!t1.is_same_size(t2)) {
    std::cerr << "Difference in tensor shapes: "
      << t1.sizes() << " v.s. " << t2.sizes() << std::endl;
    return false;
  }
  // 使用 allclose 方法检查张量的值是否在指定容差下全部相等
  bool equal = t1.allclose(t2, rtol, atol);
  // 如果不相等，则打印张量值的差异信息
  if (!equal) {
    std::cerr << "Difference in tensor value: \nFirst tensor:\n"
        << t1 << "\nSecond tensor:\n" << t2 << std::endl;
  }
  // 返回张量是否全部相等的结果
  return equal;
}

// 定义一个宏，用于断言两个张量在指定容差下全部相等
#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, rtol, atol) \
  ASSERT_TRUE(allClose(t1, t2, rtol, atol));

// 测试用例，测试 native_group_norm 函数的前向计算
TEST(MathKernelTest, NativeGroupNorm) {
  int num_channels = 6;
  int N = 2;
  int H = 2, W = 2;
  int HxW = H * W;

  // 生成随机输入张量
  const auto input = randn({N, num_channels, H, W});
  // 生成随机权重张量和偏置张量
  const auto weight = randn({num_channels});
  const auto bias = randn({num_channels});
  double eps = 1e-05;
  
  // 循环测试是否计算正确，包括测试未定义权重的情况
  for (bool undef_weight: {true, false}) {
    // 循环测试不同的分组数
    for (int num_groups: {3, 6, 1}) {
      Tensor undef;
      // 调用 native_group_norm 进行计算
      auto out = at::native::native_group_norm(
            input, undef_weight ? undef : weight, undef_weight ? undef : bias,
            N, num_channels, HxW, num_groups, eps);
      // 调用 math_group_norm 进行计算
      auto math_out = at::native::math_group_norm(
            input, undef_weight ? undef : weight, undef_weight ? undef : bias,
            N, num_channels, HxW, num_groups, eps);
      // 断言计算结果是否在指定容差内全部相等
      ASSERT_ALLCLOSE_TOLERANCES(std::get<0>(out), std::get<0>(math_out), 1e-4, 1e-6);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<1>(out), std::get<1>(math_out), 1e-4, 1e-6);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<2>(out), std::get<2>(math_out), 1e-4, 1e-6);
    }
  }
}

// 测试用例，测试 native_layer_norm 函数的前向计算
TEST(MathKernelTest, NativeLayerNorm) {
  // 生成随机输入张量
  const auto input = rand({20, 10, 10, 10});

  double eps = 1e-05;
  // 循环测试是否计算正确，包括测试未定义权重的情况
  for (bool undef_weight: {true, false}) {
    // 循环测试不同的归一化尺寸
    for (int normalized_size: {2, 3}) {
      Tensor undef;
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      // 生成随机权重张量和偏置张量
      const auto weight = rand(normalized_shape);
      const auto bias = rand(normalized_shape);

      // 调用 native_layer_norm 进行计算
      auto out = at::native_layer_norm(
            input, normalized_shape, undef_weight ? undef : weight, undef_weight ? undef : bias,
            eps);
      // 调用 math_native_layer_norm 进行计算
      auto math_out = at::native::math_native_layer_norm(
            input, normalized_shape, undef_weight ? undef : weight, undef_weight ? undef : bias,
            eps);
      // 断言计算结果是否在指定容差内全部相等
      ASSERT_ALLCLOSE_TOLERANCES(std::get<0>(out), std::get<0>(math_out), 1e-3, 1e-5);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<1>(out), std::get<1>(math_out), 1e-3, 1e-5);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<2>(out), std::get<2>(math_out), 1e-3, 1e-5);
    }
  }
}
TEST(MathKernelTest, Addr) {
  // 创建一个包含元素为1到3的向量
  const auto vec1 = arange(1., 4.);
  // 创建一个包含元素为1到2的向量
  const auto vec2 = arange(1., 3.);
  // 创建一个3行2列的零矩阵
  const auto M = zeros({3, 2});

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 遍历beta取值为1.0, 1.2, 0.0
  for (float beta: {1., 1.2, 0.}) {
    // 当beta为0时，将M矩阵中指定位置设置为正无穷和NaN
    if (beta == 0) {
      M[0][0] = std::numeric_limits<float>::infinity();
      M[2][0] = std::numeric_limits<float>::quiet_NaN();
    }
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    // 遍历alpha取值为1.0, 2.0, 0.0
    for (float alpha: {1., 2., 0.}) {
      // 调用at::native::addr函数计算结果
      auto out = at::native::addr(M, vec1, vec2, beta, alpha);
      // 调用at::native::math_addr函数计算结果
      auto math_out = at::native::math_addr(M, vec1, vec2, beta, alpha);
      // 断言两个结果在给定精度下相等
      ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);
    }
  }
}

TEST(MathKernelTest, SiluBackward) {
  // 创建一个形状为[20, 10]的随机张量作为输入
  const auto input = rand({20, 10});
  // 创建一个形状为[20, 10]的随机张量作为梯度输出
  const auto grad_output = rand({20, 10});
  // 调用at::cpu::silu_backward函数计算结果
  auto out = at::cpu::silu_backward(grad_output, input);
  // 调用at::native::math_silu_backward函数计算结果
  auto math_out = at::native::math_silu_backward(grad_output, input);
  // 断言两个结果在给定精度下相等
  ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);
}

TEST(MathKernelTest, MishBackward) {
  // 创建一个形状为[20, 10]的随机张量作为输入
  const auto input = rand({20, 10});
  // 创建一个形状为[20, 10]的随机张量作为梯度输出
  const auto grad_output = rand({20, 10});
  // 调用at::native::mish_backward函数计算结果
  auto out = at::native::mish_backward(grad_output, input);
  // 调用at::native::math_mish_backward函数计算结果
  auto math_out = at::native::math_mish_backward(grad_output, input);
  // 断言两个结果在给定精度下相等
  ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);
}

TEST(MathKernelTest, Bmm)  {
  // 定义一个测试函数test_bmm，参数为最后一个维度的大小
  auto test_bmm = [](int64_t last_dim) {
    // 创建一个形状为[1, 4, 4]的单精度浮点数随机张量x
    auto x = rand({1, 4, 4}, at::kFloat);
    // 创建一个形状为[1, 4, last_dim]的双精度浮点数随机张量y
    auto y = rand({1, 4, last_dim}, at::kDouble);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    // 断言调用at::bmm函数会抛出异常
    EXPECT_THROW(auto z = at::bmm(x, y), std::exception);
  };

  // 测试不同的最后维度大小：5和1000
  test_bmm(5);
  test_bmm(1000);
}
```