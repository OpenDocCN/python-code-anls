# `.\pytorch\test\cpp\api\fft.cpp`

```
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>

// 引入 Torch 相关头文件
#include <c10/util/irange.h>
#include <test/cpp/api/support.h>
#include <torch/torch.h>

// 定义一个简单的一维离散傅里叶变换函数
torch::Tensor naive_dft(torch::Tensor x, bool forward = true) {
  // 内部断言，确保输入张量是一维的
  TORCH_INTERNAL_ASSERT(x.dim() == 1);
  // 使输入张量变为连续存储
  x = x.contiguous();
  // 创建一个与输入张量相同大小的零张量
  auto out_tensor = torch::zeros_like(x);
  const int64_t len = x.size(0);

  // 计算单位复根 exp(-2*pi*j*n/N) 或其逆变换
  std::vector<c10::complex<double>> roots(len);
  const auto angle_base = (forward ? -2.0 : 2.0) * M_PI / len;
  for (const auto i : c10::irange(len)) {
    auto angle = i * angle_base;
    roots[i] = c10::complex<double>(std::cos(angle), std::sin(angle));
  }

  // 获取输入和输出张量的数据指针
  const auto in = x.data_ptr<c10::complex<double>>();
  const auto out = out_tensor.data_ptr<c10::complex<double>>();
  // 执行离散傅里叶变换的计算
  for (const auto i : c10::irange(len)) {
    for (const auto j : c10::irange(len)) {
      out[i] += roots[(j * i) % len] * in[j];
    }
  }
  // 返回计算结果张量
  return out_tensor;
}

// 测试用例：验证 FFT 正确性
TEST(FFTTest, fft) {
  // 创建一个大小为 128 的复数双精度随机张量
  auto t = torch::randn(128, torch::kComplexDouble);
  // 计算 Torch 中的 FFT
  auto actual = torch::fft::fft(t);
  // 计算期望值，使用自定义的傅里叶变换函数
  auto expect = naive_dft(t);
  // 断言两个张量在数值上的接近性
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// 测试用例：验证实数输入的 FFT 正确性
TEST(FFTTest, fft_real) {
  // 创建一个大小为 128 的双精度实数随机张量
  auto t = torch::randn(128, torch::kDouble);
  // 计算 Torch 中的 FFT
  auto actual = torch::fft::fft(t);
  // 计算期望值，先将实数张量转换为复数再进行 FFT
  auto expect = torch::fft::fft(t.to(torch::kComplexDouble));
  // 断言两个张量在数值上的接近性
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// 测试用例：验证 FFT 进行零填充时的正确性
TEST(FFTTest, fft_pad) {
  // 创建一个大小为 128 的复数双精度随机张量
  auto t = torch::randn(128, torch::kComplexDouble);
  // 计算 Torch 中的 FFT，使用不同的零填充尺寸
  auto actual = torch::fft::fft(t, 200);
  // 计算期望值，使用常量填充函数并计算 FFT
  auto expect = torch::fft::fft(torch::constant_pad_nd(t, {0, 72}));
  // 断言两个张量在数值上的接近性
  ASSERT_TRUE(torch::allclose(actual, expect));

  // 重新计算 FFT，使用不同的零填充尺寸
  actual = torch::fft::fft(t, 64);
  // 重新计算期望值，使用常量填充函数并计算 FFT
  expect = torch::fft::fft(torch::constant_pad_nd(t, {0, -64}));
  // 断言两个张量在数值上的接近性
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// 测试用例：验证 FFT 的归一化参数的影响
TEST(FFTTest, fft_norm) {
  // 创建一个大小为 128 的复数双精度随机张量
  auto t = torch::randn(128, torch::kComplexDouble);
  // 计算非归一化 FFT，使用默认参数
  auto unnorm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/{});
  // 计算归一化 FFT，使用 "forward" 形式的归一化
  auto norm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"forward");
  // 断言两个张量在数值上的接近性，通过除以长度来归一化
  ASSERT_TRUE(torch::allclose(unnorm / 128, norm));

  // 计算正交归一化 FFT
  auto ortho_norm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"ortho");
  // 断言两个张量在数值上的接近性，通过除以 sqrt(长度) 来正交归一化
  ASSERT_TRUE(torch::allclose(unnorm / std::sqrt(128), ortho_norm));
}

// 测试用例：验证逆 FFT 的正确性
TEST(FFTTest, ifft) {
  // 创建一个大小为 128 的复数双精度随机张量
  auto T = torch::randn(128, torch::kComplexDouble);
  // 计算 Torch 中的逆 FFT
  auto actual = torch::fft::ifft(T);
  // 计算期望值，使用自定义的傅里叶变换函数进行逆变换并归一化
  auto expect = naive_dft(T, /*forward=*/false) / 128;
  // 断言两个张量在数值上的接近性
  ASSERT_TRUE(torch::allclose(actual, expect));
}


这段代码是一组用于测试 Torch 中 FFT 函数的 Google Test 测试用例，每个测试用例都包含了对 FFT 函数不同使用情况的验证。
TEST(FFTTest, fft_ifft) {
  // 生成一个大小为 77x1 的随机复数张量
  auto t = torch::randn(77, torch::kComplexDouble);
  // 对张量进行 FFT 变换
  auto T = torch::fft::fft(t);
  // 断言变换后张量的第一个维度为 77
  ASSERT_EQ(T.size(0), 77);
  // 断言变换后张量的数据类型为 torch::kComplexDouble

  // 对 FFT 变换结果进行逆变换
  auto t_round_trip = torch::fft::ifft(T);
  // 断言逆变换后张量的第一个维度为 77
  ASSERT_EQ(t_round_trip.size(0), 77);
  // 断言逆变换后张量的数据类型为 torch::kComplexDouble
  // 断言逆变换后张量与原始张量在数值上的接近程度
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}

TEST(FFTTest, rfft) {
  // 生成一个大小为 129x1 的随机实数张量
  auto t = torch::randn(129, torch::kDouble);
  // 对实数张量进行实部 FFT 变换
  auto actual = torch::fft::rfft(t);
  // 期望值为对实数张量进行复数 FFT 变换后取部分结果
  auto expect = torch::fft::fft(t.to(torch::kComplexDouble)).slice(0, 0, 65);
  // 断言实际结果与期望结果的接近程度
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, rfft_irfft) {
  // 生成一个大小为 128x1 的随机实数张量
  auto t = torch::randn(128, torch::kDouble);
  // 对实数张量进行实部 FFT 变换
  auto T = torch::fft::rfft(t);
  // 断言变换后张量的第一个维度为 65
  ASSERT_EQ(T.size(0), 65);
  // 断言变换后张量的数据类型为 torch::kComplexDouble

  // 对 FFT 变换结果进行逆变换
  auto t_round_trip = torch::fft::irfft(T);
  // 断言逆变换后张量的第一个维度为 128
  ASSERT_EQ(t_round_trip.size(0), 128);
  // 断言逆变换后张量的数据类型为 torch::kDouble
  // 断言逆变换后张量与原始张量在数值上的接近程度
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}

TEST(FFTTest, ihfft) {
  // 生成一个大小为 129x1 的随机实数张量
  auto T = torch::randn(129, torch::kDouble);
  // 对实数张量进行 Hermite FFT 变换
  auto actual = torch::fft::ihfft(T);
  // 期望值为对实数张量进行复数逆 FFT 变换后取部分结果
  auto expect = torch::fft::ifft(T.to(torch::kComplexDouble)).slice(0, 0, 65);
  // 断言实际结果与期望结果的接近程度
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, hfft_ihfft) {
  // 生成一个大小为 64x1 的随机复数张量
  auto t = torch::randn(64, torch::kComplexDouble);
  // 将复数张量的第一个元素设为 0.5，以满足 Hermite 对称性
  t[0] = .5;
  // 对复数张量进行 Hermite FFT 变换
  auto T = torch::fft::hfft(t, 127);
  // 断言变换后张量的第一个维度为 127
  ASSERT_EQ(T.size(0), 127);
  // 断言变换后张量的数据类型为 torch::kDouble

  // 对 Hermite FFT 变换结果进行逆变换
  auto t_round_trip = torch::fft::ihfft(T);
  // 断言逆变换后张量的第一个维度为 64
  ASSERT_EQ(t_round_trip.size(0), 64);
  // 断言逆变换后张量的数据类型为 torch::kComplexDouble
  // 断言逆变换后张量与原始张量在数值上的接近程度
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}
```