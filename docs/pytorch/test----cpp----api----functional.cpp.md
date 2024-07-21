# `.\pytorch\test\cpp\api\functional.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/util/irange.h>  // 引入用于范围迭代的头文件
#include <torch/torch.h>  // 引入 PyTorch 的头文件

#include <test/cpp/api/support.h>  // 引入测试支持函数的头文件

namespace F = torch::nn::functional;  // 命名空间别名，简化对 torch::nn::functional 的引用

using namespace torch::nn;  // 使用 torch::nn 命名空间

struct FunctionalTest : torch::test::SeedingFixture {};  // 定义一个测试结构体，继承自 SeedingFixture 类

TEST_F(FunctionalTest, Conv1d) {  // 定义 Conv1d 测试用例
  auto x = torch::arange(30, torch::dtype(torch::kFloat).requires_grad(true))  // 创建张量 x，从 0 到 29，数据类型为 kFloat，需要梯度
               .reshape({2, 3, 5});  // 重塑张量 x 的形状为 2x3x5
  auto weight =
      torch::arange(18, torch::dtype(torch::kFloat).requires_grad(true))  // 创建张量 weight，从 0 到 17，数据类型为 kFloat，需要梯度
          .reshape({2, 3, 3});  // 重塑张量 weight 的形状为 2x3x3
  auto y = F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));  // 使用 Conv1d 进行卷积操作，设置步长为 1，得到输出张量 y
  auto expected = torch::tensor(
      {{{312., 348., 384.}, {798., 915., 1032.}}},  // 预期的输出张量 expected
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));  // 断言 y 与 expected 张量近似相等

  auto y_no_options = F::conv1d(x, weight);  // 使用默认选项进行 Conv1d 操作，得到输出张量 y_no_options
  ASSERT_TRUE(torch::allclose(y_no_options, expected));  // 断言 y_no_options 与 expected 张量近似相等
}

TEST_F(FunctionalTest, Conv2dEven) {  // 定义 Conv2dEven 测试用例
  auto x = torch::arange(75, torch::dtype(torch::kFloat).requires_grad(true))  // 创建张量 x，从 0 到 74，数据类型为 kFloat，需要梯度
               .reshape({1, 3, 5, 5});  // 重塑张量 x 的形状为 1x3x5x5
  auto weight =
      torch::arange(54, torch::dtype(torch::kFloat).requires_grad(true))  // 创建张量 weight，从 0 到 53，数据类型为 kFloat，需要梯度
          .reshape({2, 3, 3, 3});  // 重塑张量 weight 的形状为 2x3x3x3
  auto y = F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));  // 使用 Conv2d 进行卷积操作，设置步长为 1，得到输出张量 y
  auto expected = torch::tensor(
      {{{{15219., 15570., 15921.},
         {16974., 17325., 17676.},
         {18729., 19080., 19431.}},

        {{37818., 38898., 39978.},
         {43218., 44298., 45378.},
         {48618., 49698., 50778.}}}},  // 预期的输出张量 expected
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));  // 断言 y 与 expected 张量近似相等

  auto y_no_options = F::conv2d(x, weight);  // 使用默认选项进行 Conv2d 操作，得到输出张量 y_no_options
  ASSERT_TRUE(torch::allclose(y_no_options, expected));  // 断言 y_no_options 与 expected 张量近似相等
}

TEST_F(FunctionalTest, Conv2dUneven) {  // 定义 Conv2dUneven 测试用例
  auto x = torch::arange(60, torch::dtype(torch::kFloat).requires_grad(true))  // 创建张量 x，从 0 到 59，数据类型为 kFloat，需要梯度
               .reshape({1, 3, 5, 4});  // 重塑张量 x 的形状为 1x3x5x4
  auto weight =
      torch::arange(36, torch::dtype(torch::kFloat).requires_grad(true))  // 创建张量 weight，从 0 到 35，数据类型为 kFloat，需要梯度
          .reshape({2, 3, 3, 2});  // 重塑张量 weight 的形状为 2x3x3x2
  auto y = F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));  // 使用 Conv2d 进行卷积操作，设置步长为 1，得到输出张量 y
  auto expected = torch::tensor(
      {{{{5289., 5442., 5595.}, {5901., 6054., 6207.}, {6513., 6666., 6819.}},

        {{13227., 13704., 14181.},
         {15135., 15612., 16089.},
         {17043., 17520., 17997.}}}},  // 预期的输出张量 expected
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));  // 断言 y 与 expected 张量近似相等

  auto y_no_options = F::conv2d(x, weight);  // 使用默认选项进行 Conv2d 操作，得到输出张量 y_no_options
  ASSERT_TRUE(torch::allclose(y_no_options, expected));  // 断言 y_no_options 与 expected 张量近似相等
}
TEST_F(FunctionalTest, Conv3d) {
  // 创建一个五维张量 x，包含 375 个元素，每个元素为 float 类型，需要计算梯度
  auto x = torch::arange(375, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5, 5});
  // 创建一个五维张量 weight，包含 162 个元素，每个元素为 float 类型，需要计算梯度
  auto weight =
      torch::arange(162, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({2, 3, 3, 3, 3});
  // 对输入张量 x 和权重张量 weight 进行三维卷积操作，步长为 1
  auto y = F::conv3d(x, weight, F::Conv3dFuncOptions().stride(1));
  // 创建一个预期结果的张量 expected，包含预期的卷积结果
  auto expected = torch::tensor(
      {{{{{700704., 703944., 707184.},
          {716904., 720144., 723384.},
          {733104., 736344., 739584.}},

         {{781704., 784944., 788184.},
          {797904., 801144., 804384.},
          {814104., 817344., 820584.}},

         {{862704., 865944., 869184.},
          {878904., 882144., 885384.},
          {895104., 898344., 901584.}}},

        {{{1724220., 1734021., 1743822.},
          {1773225., 1783026., 1792827.},
          {1822230., 1832031., 1841832.}},

         {{1969245., 1979046., 1988847.},
          {2018250., 2028051., 2037852.},
          {2067255., 2077056., 2086857.}},

         {{2214270., 2224071., 2233872.},
          {2263275., 2273076., 2282877.},
          {2312280., 2322081., 2331882.}}}}},
      torch::kFloat);
  // 断言卷积结果 y 与预期结果 expected 在容差范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));

  // 对输入张量 x 和权重张量 weight 进行三维卷积操作，使用默认选项
  auto y_no_options = F::conv3d(x, weight);
  // 断言卷积结果 y_no_options 与预期结果 expected 在容差范围内相等
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

TEST_F(FunctionalTest, MaxPool1d) {
  // 创建一个尺寸为 [1, 1, 5] 的全一张量 x
  auto x = torch::ones({1, 1, 5});
  // 对输入张量 x 进行一维最大池化操作，窗口大小为 3，步长为 2
  auto y = F::max_pool1d(x, F::MaxPool1dFuncOptions(3).stride(2));

  // 断言池化后的张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言池化后的张量 y 与全一张量在容差范围内相等
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  // 断言池化后的张量 y 的尺寸为 [1, 1, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(FunctionalTest, MaxPool2d) {
  // 创建一个尺寸为 [2, 5, 5] 的全一张量 x
  auto x = torch::ones({2, 5, 5});
  // 对输入张量 x 进行二维最大池化操作，窗口大小为 3，步长为 2
  auto y = F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));

  // 断言池化后的张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言池化后的张量 y 与全一张量在容差范围内相等
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言池化后的张量 y 的尺寸为 [2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(FunctionalTest, MaxPool2dBackward) {
  // 创建一个尺寸为 [1, 2, 4, 4] 的随机张量 input，需要计算梯度
  auto input = torch::rand(
      {1, 2, 4, 4}, torch::dtype(torch::kFloat).requires_grad(true));
  // 对输入张量 input 进行二维最大池化操作，窗口大小为 2
  auto output = F::max_pool2d(input, F::MaxPool2dFuncOptions(2));
  // 计算池化结果 output 的所有元素的和
  auto s = output.sum();
  // 对池化结果 output 的所有元素的和进行反向传播
  s.backward();
  // 断言输入张量 input 的尺寸与梯度的尺寸相同
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, MaxPool3d) {
  // 创建一个尺寸为 [2, 5, 5, 5] 的全一张量 x
  auto x = torch::ones({2, 5, 5, 5});
  // 对输入张量 x 进行三维最大池化操作，窗口大小为 3，步长为 2
  auto y = F::max_pool3d(x, F::MaxPool3dFuncOptions(3).stride(2));

  // 断言池化后的张量 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言池化后的张量 y 与全一张量在容差范围内相等
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  // 断言池化后的张量 y 的尺寸为 [2, 2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool1d) {
  // 创建一个尺寸为 [1, 1, 5] 的全一张量 x
  auto x = torch::ones({1, 1, 5});
  // 对输入张量 x 进行一维平均池化操作，窗口大小为 3，步长为 2
  auto y = F::avg_pool1d(x, F::AvgPool1dFuncOptions(3).stride(2));

  // 断言池化后的张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言池化后的张量 y 与全一张量在容差范围内相等
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  // 断言池化后的张量 y 的尺寸为 [1, 1, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}
TEST_F(FunctionalTest, AvgPool2d) {
  // 创建一个大小为 (2, 5, 5) 的张量 x，所有元素均为1
  auto x = torch::ones({2, 5, 5});
  // 对张量 x 进行 2D 平均池化操作，池化窗口大小为 3，步幅为 2
  auto y = F::avg_pool2d(x, F::AvgPool2dFuncOptions(3).stride(2));

  // 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言 y 的值与一个大小为 (2, 2, 2) 的张量（所有元素为1）非常接近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言 y 的尺寸为 [2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool3d) {
  // 创建一个大小为 (2, 5, 5, 5) 的张量 x，所有元素均为1
  auto x = torch::ones({2, 5, 5, 5});
  // 对张量 x 进行 3D 平均池化操作，池化窗口大小为 3，步幅为 2
  auto y = F::avg_pool3d(x, F::AvgPool3dFuncOptions(3).stride(2));

  // 断言 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言 y 的值与一个大小为 (2, 2, 2, 2) 的张量（所有元素为1）非常接近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  // 断言 y 的尺寸为 [2, 2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, FractionalMaxPool2d) {
  // 创建一个大小为 (2, 5, 5) 的张量 x，所有元素均为1
  auto x = torch::ones({2, 5, 5});
  // 对张量 x 进行 2D 分数最大池化操作，输出尺寸为 (2, 2)，池化窗口大小为 3
  auto y = F::fractional_max_pool2d(
      x, F::FractionalMaxPool2dFuncOptions(3).output_size(2));

  // 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言 y 的值与一个大小为 (2, 2, 2) 的张量（所有元素为1）非常接近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言 y 的尺寸为 [2, 2, 2]

  // 对张量 x 进行 2D 分数最大池化操作，输出尺寸为 (2, 2)，同时返回池化索引
  auto y_with_indices = F::fractional_max_pool2d_with_indices(
      x, F::FractionalMaxPool2dFuncOptions(3).output_size(2));
  // 断言 y 与返回结果的第一个元素相等
  ASSERT_TRUE(torch::equal(y, std::get<0>(y_with_indices)));
  // 断言返回结果的第二个元素（池化索引）与预期的张量非常接近
  ASSERT_TRUE(torch::allclose(
      std::get<1>(y_with_indices),
      torch::tensor({{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}})));
  // 断言返回结果的第二个元素（池化索引）的尺寸为 [2, 2, 2]
  ASSERT_EQ(
      std::get<1>(y_with_indices).sizes(), std::vector<int64_t>({2, 2, 2}));

  // 创建一个大小为 (2, 2, 5, 5) 的张量 x1，所有元素均为1
  auto x1 = torch::ones({2, 2, 5, 5});
  // 对张量 x1 进行 2D 分数最大池化操作，输出尺寸为 (2, 2)，池化窗口大小为 3
  auto y1 = F::fractional_max_pool2d(
      x1, F::FractionalMaxPool2dFuncOptions(3).output_size(2));

  // 断言 y1 的维度为 4
  ASSERT_EQ(y1.ndimension(), 4);
  // 断言 y1 的值与一个大小为 (2, 2, 2, 2) 的张量（所有元素为1）非常接近
  ASSERT_TRUE(torch::allclose(y1, torch::ones({2, 2, 2, 2})));
  // 断言 y1 的尺寸为 [2, 2, 2, 2]

  // 对张量 x1 进行 2D 分数最大池化操作，输出尺寸为 (2, 2)，同时返回池化索引
  auto y1_with_indices = F::fractional_max_pool2d_with_indices(
      x1, F::FractionalMaxPool2dFuncOptions(3).output_size(2));
  // 断言 y1 与返回结果的第一个元素相等
  ASSERT_TRUE(torch::equal(y1, std::get<0>(y1_with_indices)));
  // 断言返回结果的第二个元素（池化索引）与预期的张量非常接近
  ASSERT_TRUE(torch::allclose(
      std::get<1>(y1_with_indices),
      torch::tensor(
          {{{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}},
           {{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}}})));
  // 断言返回结果的第二个元素（池化索引）的尺寸为 [2, 2, 2, 2]
  ASSERT_EQ(
      std::get<1>(y1_with_indices).sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, FractionalMaxPool3d) {
  // 创建一个大小为 (2, 5, 5, 5) 的张量 x，所有元素均为1
  auto x = torch::ones({2, 5, 5, 5});
  // 对张量 x 进行 3D 分数最大池化操作，输出尺寸为 (2, 2)，池化窗口大小为 3
  auto y = F::fractional_max_pool3d(
      x, F::FractionalMaxPool3dFuncOptions(3).output_size(2));

  // 断言 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言 y 的值与一个大小为 (2, 2, 2, 2) 的张量（所有元素为1）非常接近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  // 断言 y 的尺寸为 [2, 2, 2, 2]

  // 对张量 x 进行 3D 分数最大池化操作，输出尺寸为 (2, 2)，同时返回池化索引
  auto y_with_indices = F::fractional_max_pool3d_with_indices(
      x, F::FractionalMaxPool3dFuncOptions(3).output_size(2));
  // 断言 y 与返回结果的第一个元素相等
  ASSERT_TRUE(torch::equal(y, std::get<0>(y_with_indices)));
  // 断言返回结果的第二个元素（池化索引）与预期的张量非常接近
  ASSERT_TRUE(torch::allclose(
      std::get<1>(y_with_indices),
      torch::tensor(
          {{{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}},
           {{{0, 2}, {10, 12}}, {{50, 52}, {60, 62
TEST_F(FunctionalTest, LPPool1d) {
  // 定义 LPPool1d 的参数
  int norm_type = 2;
  int stride = 2;
  int kernel_size = 3;

  // 创建一个大小为 {1, 1, 5} 的全1张量 x
  auto x = torch::ones({1, 1, 5});
  // 对 x 进行 LPPool1d 操作，返回结果 y
  auto y = F::lp_pool1d(
      x, F::LPPool1dFuncOptions(norm_type, kernel_size).stride(stride));
  // 计算期望结果 expected
  auto expected =
      (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) *
       kernel_size)
          .pow(1. / norm_type);

  // 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言 y 与 expected 在误差范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));
  // 断言 y 的大小为 {1, 1, 2}
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, LPPool2d) {
  // 定义 LPPool2d 的参数
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({2, 3});

  // 创建一个大小为 {1, 1, 2, 5} 的全1张量 x
  auto x = torch::ones({1, 1, 2, 5});
  // 对 x 进行 LPPool2d 操作，返回结果 y
  auto y = F::lp_pool2d(
      x, F::LPPool2dFuncOptions(norm_type, kernel_size).stride(stride));
  // 计算期望结果 expected
  auto expected =
      (torch::pow(torch::tensor({{{{1, 1}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1]))
          .pow(1. / norm_type);

  // 断言 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言 y 与 expected 在误差范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));
  // 断言 y 的大小为 {1, 1, 1, 2}
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 2}));
}

TEST_F(FunctionalTest, LPPool3d) {
  // 定义 LPPool3d 的参数
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({1, 2, 3});

  // 创建一个大小为 {1, 1, 1, 2, 5} 的全1张量 x
  auto x = torch::ones({1, 1, 1, 2, 5});
  // 对 x 进行 LPPool3d 操作，返回结果 y
  auto y = F::lp_pool3d(
      x, F::LPPool3dFuncOptions(norm_type, kernel_size).stride(stride));
  // 计算期望结果 expected
  auto expected =
      (torch::pow(torch::tensor({{{{{1, 1}}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1] * kernel_size[2]))
          .pow(1. / norm_type);

  // 断言 y 的维度为 5
  ASSERT_EQ(y.ndimension(), 5);
  // 断言 y 与 expected 在误差范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));
  // 断言 y 的大小为 {1, 1, 1, 1, 2}
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 1, 2}));
}

TEST_F(FunctionalTest, CosineSimilarity) {
  // 创建两个输入张量 input1 和 input2
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  // 计算 input1 和 input2 的余弦相似度，返回结果 output
  auto output = F::cosine_similarity(
      input1, input2, F::CosineSimilarityFuncOptions().dim(1));
  // 计算期望结果 expected
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  // 断言 output 与 expected 在误差范围内相等
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, SmoothL1LossDefaultOptions) {
  // 创建输入张量 input 和目标张量 target
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算 Smooth L1 Loss，返回结果 output
  auto output = F::smooth_l1_loss(input, target);
  // 计算期望结果 expected
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  // 对 output 求和并进行反向传播
  auto s = output.sum();
  s.backward();
  // 断言 output 与 expected 相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言 input 的大小与梯度的大小相等
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}
TEST_F(FunctionalTest, SmoothL1LossBeta) {
  // 创建输入张量，包括数值和梯度信息
  auto input = torch::tensor(
      {0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算 Smooth L1 Loss，指定了平均值作为缩减方式，并设置 beta 为 0.5
  auto output =
      F::smooth_l1_loss(
          input, target, /*reduction=*/torch::kMean, /*beta=*/0.5);
  // 创建期望的输出张量
  auto expected = torch::tensor(1.67, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 执行反向传播
  s.backward();
  // 断言输出张量与期望张量在数值上近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和梯度张量的形状相同
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, SmoothL1LossBetaOptions) {
  // 创建输入张量，包括数值和梯度信息
  auto input = torch::tensor(
      {0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 创建 Smooth L1 Loss 的选项对象，指定了平均值作为缩减方式，并设置 beta 为 0.5
  auto output =
      F::smooth_l1_loss(
          input,
          target,
          F::SmoothL1LossFuncOptions().reduction(torch::kMean).beta(0.5));
  // 创建期望的输出张量
  auto expected = torch::tensor(1.67, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 执行反向传播
  s.backward();
  // 断言输出张量与期望张量在数值上近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和梯度张量的形状相同
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, SmoothL1LossNoReduction) {
  // 创建输入张量，包括数值和梯度信息
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算 Smooth L1 Loss，指定了无缩减，并注释了 reduction 参数为 torch::kNone
  auto output =
      F::smooth_l1_loss(input, target, /*reduction=*/torch::kNone);
  // 创建期望的输出张量
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 执行反向传播
  s.backward();
  // 断言输出张量与期望张量在数值上近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和梯度张量的形状相同
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, HuberLossDefaultOptions) {
  // 创建输入张量，包括数值和梯度信息
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算 Huber Loss，默认使用选项
  auto output = F::huber_loss(input, target);
  // 创建期望的输出张量
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 执行反向传播
  s.backward();
  // 断言输出张量与期望张量在数值上近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和梯度张量的形状相同
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, HuberLossDelta) {
  // 创建输入张量，包括数值和梯度信息
  auto input = torch::tensor(
      {0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 创建 Huber Loss 的选项对象，指定了平均值作为缩减方式，并设置 delta 为 0.5
  auto options = F::HuberLossFuncOptions().reduction(torch::kMean).delta(0.5);
  // 计算 Huber Loss，使用指定的选项
  auto output = F::huber_loss(input, target, options);
  // 创建期望的输出张量
  auto expected = torch::tensor(1.67 * 0.5, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 执行反向传播
  s.backward();
  // 断言输出张量与期望张量在数值上近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和梯度张量的形状相同
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}
TEST_F(FunctionalTest, HuberLossNoReduction) {
  // 创建一个张量作为输入，设置为可求导的浮点数类型
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 创建 Huber 损失函数的选项，指定不进行减少操作
  auto options = F::HuberLossFuncOptions().reduction(torch::kNone);
  // 计算 Huber 损失
  auto output = F::huber_loss(input, target, options);
  // 期望的输出结果
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 对总和进行反向传播
  s.backward();
  // 断言输出张量是否接近期望值
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的尺寸与梯度张量的尺寸相同
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, SoftMarginLossDefaultOptions) {
  // 创建一个张量作为输入，设置为可求导的浮点数类型
  auto input = torch::tensor(
      {2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  // 计算 Soft Margin 损失
  auto output = F::soft_margin_loss(input, target);
  // 期望的输出结果
  auto expected = torch::tensor({1.3767317}, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 对总和进行反向传播
  s.backward();

  // 断言输出张量是否接近期望值
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的尺寸与梯度张量的尺寸相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, MultiLabelSoftMarginLossDefaultOptions) {
  // 创建一个张量作为输入，设置为可求导的浮点数类型
  auto input = torch::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target =
      torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  // 计算多标签 Soft Margin 损失
  auto output = F::multilabel_soft_margin_loss(input, target);
  // 期望的输出结果
  auto expected = torch::tensor({0.7608436}, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 对总和进行反向传播
  s.backward();

  // 断言输出张量是否接近期望值
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的尺寸与梯度张量的尺寸相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, SoftMarginLossNoReduction) {
  // 创建一个张量作为输入，设置为可求导的浮点数类型
  auto input = torch::tensor(
      {2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  // 创建 Soft Margin 损失函数的选项，指定不进行减少操作
  auto output = F::soft_margin_loss(input, target, torch::kNone);
  // 期望的输出结果
  auto expected = torch::tensor(
      {2.1269281, 0.01814993, 0.3132617, 3.0485873}, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 对总和进行反向传播
  s.backward();

  // 断言输出张量是否接近期望值
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的尺寸与梯度张量的尺寸相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, MultiLabelSoftMarginLossWeightedNoReduction) {
  // 创建一个张量作为输入，设置为可求导的浮点数类型
  auto input = torch::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target =
      torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  // 创建权重张量
  auto weight = torch::tensor({0.1, 0.6, 0.4, 0.8}, torch::kFloat);
  // 创建多标签 Soft Margin 损失函数的选项，指定不进行减少操作，并设置权重
  auto options = F::MultilabelSoftMarginLossFuncOptions()
                     .reduction(torch::kNone)
                     .weight(weight);
  // 计算多标签 Soft Margin 损失
  auto output = F::multilabel_soft_margin_loss(input, target, options);
  // 期望的输出结果
  auto expected = torch::tensor({0.4876902, 0.3321295}, torch::kFloat);
  // 计算输出张量的总和
  auto s = output.sum();
  // 对总和进行反向传播
  s.backward();

  // 断言输出张量是否接近期望值
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的尺寸与梯度张量的尺寸相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
// 在 FunctionalTest 测试套件中，测试 torch 库中的 pairwise_distance 函数
TEST_F(FunctionalTest, PairwiseDistance) {
  // 创建第一个输入张量 input1，包含两个行向量
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  // 创建第二个输入张量 input2，包含两个行向量
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  // 计算 input1 和 input2 之间的 L1 距离，使用默认参数 p=1
  auto output = F::pairwise_distance(
      input1, input2, F::PairwiseDistanceFuncOptions().p(1));
  // 创建期望的输出张量 expected，包含两个距离值
  auto expected = torch::tensor({6, 6}, torch::kFloat);
  // 断言实际输出 output 是否与期望输出 expected 全部接近
  ASSERT_TRUE(output.allclose(expected));
}

// 在 FunctionalTest 测试套件中，测试 torch 库中的 pdist 函数
TEST_F(FunctionalTest, PDist) {
  {
    // 创建输入张量 input1，包含两个点的坐标
    auto input = torch::tensor({{-1.0, -5.0, -1.0}, {2.0, 4.0, 6.0}});
    // 计算 input 中所有点之间的欧氏距离
    auto output = F::pdist(input);
    // 创建期望的输出张量 expected，包含计算得到的距离值
    auto expected = torch::tensor({11.7898});
    // 断言实际输出 output 是否与期望输出 expected 全部接近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建输入张量 input2，包含三个点的坐标
    auto input = torch::tensor({{1.0, -1.0}, {1.0, 3.0}, {3.0, 3.0}});
    // 计算 input 中所有点之间的带权 L2 距离，使用参数 p=1.5
    auto output = F::pdist(input, 1.5);
    // 创建期望的输出张量 expected，包含计算得到的距离值
    auto expected = torch::tensor({4.0, 4.8945, 2.0});
    // 断言实际输出 output 是否与期望输出 expected 全部接近
    ASSERT_TRUE(output.allclose(expected));
  }
}

// 在 FunctionalTest 测试套件中，测试 torch 库中的 adaptive_max_pool1d 函数
TEST_F(FunctionalTest, AdaptiveMaxPool1d) {
  // 创建输入张量 x，包含一个长度为 5 的序列
  auto x = torch::ones({1, 1, 5});
  // 对输入张量 x 进行自适应最大池化，目标输出大小为 3
  auto y = F::adaptive_max_pool1d(x, F::AdaptiveMaxPool1dFuncOptions(3));

  // 断言输出张量 y 的维度是否为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 是否与全为 1 的目标张量形状一致
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  // 断言输出张量 y 的大小是否与期望的大小向量相匹配
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

// 在 FunctionalTest 测试套件中，测试 torch 库中的 adaptive_max_pool2d 函数
TEST_F(FunctionalTest, AdaptiveMaxPool2d) {
  // 创建输入张量 x，包含两个尺寸为 5x5 的图像
  auto x = torch::ones({2, 5, 5});
  // 对输入张量 x 进行自适应最大池化，目标输出大小为 3x3
  auto y = F::adaptive_max_pool2d(x, F::AdaptiveMaxPool2dFuncOptions(3));

  // 断言输出张量 y 的维度是否为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 是否与全为 1 的目标张量形状一致
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  // 断言输出张量 y 的大小是否与期望的大小向量相匹配
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

// 在 FunctionalTest 测试套件中，测试 torch 库中的 adaptive_max_pool3d 函数
TEST_F(FunctionalTest, AdaptiveMaxPool3d) {
  // 创建输入张量 x，包含两个尺寸为 5x5x5 的三维数据
  auto x = torch::ones({2, 5, 5, 5});
  // 对输入张量 x 进行自适应最大池化，目标输出大小为 3x3x3
  auto y = F::adaptive_max_pool3d(x, F::AdaptiveMaxPool3dFuncOptions(3));

  // 断言输出张量 y 的维度是否为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言输出张量 y 是否与全为 1 的目标张量形状一致
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  // 断言输出张量 y 的大小是否与期望的大小向量相匹配
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3, 3}));
}

// 在 FunctionalTest 测试套件中，测试 torch 库中的 adaptive_avg_pool1d 函数
TEST_F(FunctionalTest, AdaptiveAvgPool1d) {
  // 创建输入张量 x，包含一个长度为 5 的序列
  auto x = torch::ones({1, 1, 5});
  // 对输入张量 x 进行自适应平均池化，目标输出大小为 3
  auto y = F::adaptive_avg_pool1d(x, F::AdaptiveAvgPool1dFuncOptions(3));

  // 断言输出张量 y 的维度是否为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 是否与全为 1 的目标张量形状一致
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  // 断言输出张量 y 的大小是否与期望的大小向量相匹配
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

// 在 FunctionalTest 测试套件中，测试 torch 库中的 adaptive_avg_pool2d 函数
TEST_F(FunctionalTest, AdaptiveAvgPool2d) {
  // 创建输入张量 x，包含两个尺寸为 5x5 的图像
  auto x = torch::ones({2, 5, 5});
  // 对输入张量 x 进行自适应平均池化，目标输出大小为 3x3
  auto y = F::adaptive_avg_pool2d(x, F::AdaptiveAvgPool2dFuncOptions(3));

  // 断言输出张量 y 的维度是否为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 是否与全为 1 的目标张量形状一致
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  // 断言输出张量 y 的大小是否与期望的大小向量相匹配
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

// 在 FunctionalTest 测试套件中，测试 torch 库中的 adaptive_avg_pool3d 函数
TEST_F(FunctionalTest, AdaptiveAvgPool3d) {
  // 创建输入张量 x，包含两个尺寸为 5x5x5 的三维数据
  auto x = torch::ones({2, 5, 5, 5});
  // 对输入张量 x 进行自
// 在 FunctionalTest 测试类中定义 L1 Loss 测试函数
TEST_F(FunctionalTest, L1Loss) {
  // 生成一个大小为 [5, 6] 的张量，要求梯度计算
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为 [5, 6] 的空张量，并随机初始化为 0 或 1
  auto target = torch::empty({5, 6}).random_(2);
  // 计算输入张量经过 sigmoid 函数后与目标张量之间的 L1 Loss
  auto output = F::l1_loss(torch::sigmoid(input), target);
  // 计算输出张量的所有元素之和
  auto s = output.sum();
  // 执行反向传播
  s.backward();

  // 断言输出张量的大小为空数组
  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  // 断言输入张量的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// 在 FunctionalTest 测试类中定义 MSE Loss 测试函数
TEST_F(FunctionalTest, MSELoss) {
  // 生成一个大小为 [5, 6] 的张量，要求梯度计算
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为 [5, 6] 的空张量，并随机初始化为 0 或 1
  auto target = torch::empty({5, 6}).random_(2);
  // 计算输入张量经过 sigmoid 函数后与目标张量之间的 MSE Loss
  auto output = F::mse_loss(torch::sigmoid(input), target);
  // 计算输出张量的所有元素之和
  auto s = output.sum();
  // 执行反向传播
  s.backward();

  // 断言输出张量的大小为空数组
  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  // 断言输入张量的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// 在 FunctionalTest 测试类中定义 BCE Loss 测试函数
TEST_F(FunctionalTest, BCELoss) {
  // 生成一个大小为 [5, 6] 的张量，要求梯度计算
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为 [5, 6] 的空张量，并随机初始化为 0 或 1
  auto target = torch::empty({5, 6}).random_(2);
  // 计算输入张量经过 sigmoid 函数后与目标张量之间的 Binary Cross Entropy Loss
  auto output = F::binary_cross_entropy(torch::sigmoid(input), target);
  // 计算输出张量的所有元素之和
  auto s = output.sum();
  // 执行反向传播
  s.backward();

  // 断言输出张量的大小为空数组
  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  // 断言输入张量的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// 在 FunctionalTest 测试类中定义 KL Divergence Loss 测试函数
TEST_F(FunctionalTest, KLDivLoss) {
  // 创建 KL Divergence Loss 对象
  KLDivLoss loss;
  // 生成一个大小为 [5, 6] 的张量，要求梯度计算
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为 [5, 6] 的空张量，并随机初始化为 0 或 1
  auto target = torch::empty({5, 6}).random_(2);
  // 计算输入张量经过 sigmoid 函数后与目标张量之间的 KL Divergence Loss
  auto output = F::kl_div(torch::sigmoid(input), target);
  // 计算输出张量的所有元素之和
  auto s = output.sum();
  // 执行反向传播
  s.backward();

  // 断言输出张量的大小为空数组
  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  // 断言输入张量的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// 在 FunctionalTest 测试类中定义 Hinge Embedding Loss 测试函数
TEST_F(FunctionalTest, HingeEmbeddingLoss) {
  // 生成一个大小为 [2, 3] 的浮点型张量
  auto input = torch::tensor({{2, 22, 4}, {20, 10, 0}}, torch::kFloat);
  // 生成一个大小为 [2, 3] 的浮点型张量
  auto target = torch::tensor({{2, 6, 4}, {1, 10, 0}}, torch::kFloat);
  // 计算输入张量与目标张量之间的 Hinge Embedding Loss，设置 margin 参数为 2
  auto output = F::hinge_embedding_loss(
      input, target, F::HingeEmbeddingLossFuncOptions().margin(2));
  // 生成一个预期的大小为 [1] 的浮点型张量
  auto expected = torch::tensor({10}, torch::kFloat);

  // 断言输出张量的所有元素近似等于预期张量的所有元素
  ASSERT_TRUE(output.allclose(expected));
}
TEST_F(FunctionalTest, GridSample) {
  // 创建一个1x1x3x3的浮点张量，数值为0到8，用来作为输入
  auto input =
      torch::arange(9, torch::kFloat).view(std::vector<int64_t>({1, 1, 3, 3}));
  // 创建一个3x2x3x2的浮点张量，表示采样点的位置偏移
  auto grid = torch::tensor(
      {{{{-2., -1.}, {-1., -1.}, {0., -1.}},
        {{-1., 0.}, {0., 0.}, {1., 0.}},
        {{0., 1.}, {1., 1.}, {2., 1.}}}},
      torch::kFloat);

  // 设置GridSample函数的选项：双线性插值模式，边界填充模式为零，角点对齐方式为真
  auto options = F::GridSampleFuncOptions()
                     .mode(torch::kBilinear)
                     .padding_mode(torch::kZeros)
                     .align_corners(true);
  // 使用GridSample函数对输入进行采样，得到输出
  auto output = F::grid_sample(input, grid, options);
  // 预期的输出结果，用于与实际输出进行比较
  auto expected = torch::tensor(
      {{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 0.}}}}, torch::kFloat);

  // 断言实际输出与预期输出接近
  ASSERT_TRUE(output.allclose(expected));

  // 设置GridSample函数的选项：双线性插值模式，边界填充模式为零，角点对齐方式为假
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false);
  // 使用更新后的选项再次进行采样
  output = F::grid_sample(input, grid, options);
  // 更新预期的输出结果
  expected = torch::tensor(
      {{{{0., 0., 0.5}, {1.5, 4., 2.5}, {3.5, 2., 0.}}}}, torch::kFloat);

  // 断言实际输出与更新后的预期输出接近
  ASSERT_TRUE(output.allclose(expected));

  // 使用默认选项进行采样（双线性插值模式，边界填充模式为零，角点对齐方式为假），预期结果与上面相同
  output = F::grid_sample(input, grid);

  // 断言实际输出与默认选项下的预期输出接近
  ASSERT_TRUE(output.allclose(expected));

  // 设置GridSample函数的选项：最近邻插值模式，边界填充模式为零，角点对齐方式为真
  options = F::GridSampleFuncOptions()
                .mode(torch::kNearest)
                .padding_mode(torch::kZeros)
                .align_corners(true);
  // 使用最近邻插值模式进行采样
  output = F::grid_sample(input, grid, options);
  // 更新预期的输出结果
  expected = torch::tensor(
      {{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 0.}}}}, torch::kFloat);

  // 断言实际输出与更新后的预期输出接近
  ASSERT_TRUE(output.allclose(expected));

  // 设置GridSample函数的选项：双线性插值模式，边界填充模式为边界值复制，角点对齐方式为真
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true);
  // 使用边界值复制模式进行采样
  output = F::grid_sample(input, grid, options);
  // 更新预期的输出结果
  expected = torch::tensor(
      {{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 8.}}}}, torch::kFloat);

  // 断言实际输出与更新后的预期输出接近
  ASSERT_TRUE(output.allclose(expected));

  // 设置GridSample函数的选项：双线性插值模式，边界填充模式为反射，角点对齐方式为真
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kReflection)
                .align_corners(true);
  // 使用反射模式进行采样
  output = F::grid_sample(input, grid, options);
  // 更新预期的输出结果
  expected = torch::tensor(
      {{{{1., 0., 1.}, {3., 4., 5.}, {7., 8., 7.}}}}, torch::kFloat);

  // 断言实际输出与更新后的预期输出接近
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, AffineGrid) {
  {
    // 创建一个2x2x3的浮点张量，用来表示2D仿射变换参数
    auto theta = torch::arange(1., 13).view(std::vector<int64_t>({2, 2, 3}));
    // 定义输出张量的大小
    auto size = std::vector<int64_t>({2, 3, 2, 2});
    // 定义角点对齐方式为真
    auto align_corners = true;
    // 使用仿射网格函数生成仿射变换后的网格
    auto output = F::affine_grid(theta, size, !align_corners);
    // 预期的输出结果
    auto expected = torch::tensor(
        {{{{1.50, 1.50}, {2.50, 5.50}}, {{3.50, 6.50}, {4.50, 10.50}}},
         {{{1.50, 1.50}, {8.50, 11.50}}, {{9.50, 12.50}, {16.50, 22.50}}}});
    // 使用角点对齐方式为真的仿射网格函数生成仿射变换后的网格
    auto output_aligned = F::affine_grid(theta, size, align_corners);
    {
        // 定义期望的对齐结果张量
        auto expected_aligned = torch::tensor(
            {{{{0.0, -3.0}, {2.0, 5.0}}, {{4.0, 7.0}, {6.0, 15.0}}},
             {{{-6.0, -9.0}, {8.0, 11.0}}, {{10.0, 13.0}, {24.0, 33.0}}}});
    
        // 断言输出张量与期望张量在接近范围内相等
        ASSERT_TRUE(output.allclose(expected));
    
        // 断言对齐后的输出张量与期望的对齐结果张量在接近范围内相等
        ASSERT_TRUE(output_aligned.allclose(expected_aligned));
      }
      {
        // 3维仿射变换。
        // 创建一个1x3x4的张量theta
        auto theta = torch::arange(1., 13).view(std::vector<int64_t>({1, 3, 4}));
        
        // 定义输出大小为1x1x3x2x2
        auto size = std::vector<int64_t>({1, 1, 3, 2, 2});
        
        // 是否在角点对齐的标志
        auto align_corners = true;
        
        // 执行仿射网格生成
        auto output = F::affine_grid(theta, size, !align_corners);
        
        // 定义期望的输出张量
        auto expected = torch::tensor(
            {{{{{0.5000, -2.1667, -4.8333}, {1.5000, 2.8333, 4.1667}},
               {{2.5000, 3.8333, 5.1667}, {3.5000, 8.8333, 14.1667}}},
              {{{2.5000, 2.5000, 2.5000}, {3.5000, 7.5000, 11.5000}},
               {{4.5000, 8.5000, 12.5000}, {5.5000, 13.5000, 21.5000}}},
              {{{4.5000, 7.1667, 9.8333}, {5.5000, 12.1667, 18.8333}},
               {{6.5000, 13.1667, 19.8333}, {7.5000, 18.1667, 28.8333}}}}});
        
        // 执行对齐后的仿射网格生成
        auto output_aligned = F::affine_grid(theta, size, align_corners);
        
        // 定义期望的对齐结果张量
        auto expected_aligned = torch::tensor(
            {{{{{-2.0, -10.0, -18.0}, {0.0, 0.0, 0.0}},
               {{2.0, 2.0, 2.0}, {4.0, 12.0, 20.0}}},
              {{{1.0, -3.0, -7.0}, {3.0, 7.0, 11.0}},
               {{5.0, 9.0, 13.0}, {7.0, 19.0, 31.0}}},
              {{{4.0, 4.0, 4.0}, {6.0, 14.0, 22.0}},
               {{8.0, 16.0, 24.0}, {10.0, 26.0, 42.0}}}}});
    
        // 断言输出张量与期望张量在较小的误差范围内相等
        ASSERT_TRUE(output.allclose(expected, 1e-2));
        
        // 断言对齐后的输出张量与期望的对齐结果张量在默认的误差范围内相等
        ASSERT_TRUE(output_aligned.allclose(expected_aligned));
      }
      {
        // 创建一个1x2x3的空双精度张量theta
        auto theta = torch::empty({1, 2, 3}, torch::kDouble);
        
        // 定义输出大小为1x1x2x2
        auto size = std::vector<int64_t>({1, 1, 2, 2});
        
        // 使用异常断言检查以下情况：
        // - 期望输出大小为非零正数，但提供了[-1, 1, 2, 2]
        ASSERT_THROWS_WITH(
            F::affine_grid(torch::empty({2, 2, 3}), {-1, 1, 2, 2}),
            "Expected non-zero, positive output size. Got [-1, 1, 2, 2]");
        
        // - 期望theta具有浮点类型，但提供了整数类型
        ASSERT_THROWS_WITH(
            F::affine_grid(torch::empty({2, 2, 3}, torch::kInt), size),
            "Expected theta to have floating point type, but got int");
        
        // - 期望theta的形状为Nx2x3，但提供了[2, 3]
        ASSERT_THROWS_WITH(
            F::affine_grid(theta[0], size),
            "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
            "[1, 1, 2, 2]. Got [2, 3].");
        
        // - 期望theta的形状为Nx2x3，但提供了[1, 1, 2, 3]
        ASSERT_THROWS_WITH(
            F::affine_grid(theta.unsqueeze(0), size),
            "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
            "[1, 1, 2, 2]. Got [1, 1, 2, 3].");
        
        // - 期望theta的形状为Nx2x3，但提供了[1, 4, 3]
        ASSERT_THROWS_WITH(
            F::affine_grid(theta.repeat({1, 2, 1}), size),
            "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
            "[1, 1, 2, 2]. Got [1, 4, 3].");
        
        // - 期望theta的形状为Nx2x3，但提供了[1, 2, 6]
        ASSERT_THROWS_WITH(
            F::affine_grid(theta.repeat({1, 1, 2}), size),
            "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
            "[1, 1, 2, 2]. Got [1, 2, 6].");
      }
      {
        // 创建一个1x3x4的空双精度张量theta
        auto theta = torch::empty({1, 3, 4}, torch::kDouble);
        
        // 定义输出大小为1x1x2x2x3
        auto size = std::vector<int64_t>({1, 1, 2, 2, 3});
        
        // 继续检查theta的形状是否与预期不符的情况
        // - 期望theta的形状为Nx2x3，但提供了[1, 3, 4]
    // 断言：调用 F::affine_grid 函数，并期望抛出异常，异常信息指出期望接收形状为 Nx3x4 的批量 3D 仿射矩阵，对应尺寸为 [1, 1, 2, 2, 3]，但实际接收到的尺寸是 [3, 4]。
    ASSERT_THROWS_WITH(
        F::affine_grid(theta[0], size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [3, 4].");

    // 断言：调用 F::affine_grid 函数，并期望抛出异常，异常信息指出期望接收形状为 Nx3x4 的批量 3D 仿射矩阵，对应尺寸为 [1, 1, 2, 2, 3]，但实际接收到的尺寸是 [1, 1, 3, 4]。
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.unsqueeze(0), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 1, 3, 4].");

    // 断言：调用 F::affine_grid 函数，并期望抛出异常，异常信息指出期望接收形状为 Nx3x4 的批量 3D 仿射矩阵，对应尺寸为 [1, 1, 2, 2, 3]，但实际接收到的尺寸是 [1, 6, 4]。
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 2, 1}), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 6, 4].");

    // 断言：调用 F::affine_grid 函数，并期望抛出异常，异常信息指出期望接收形状为 Nx3x4 的批量 3D 仿射矩阵，对应尺寸为 [1, 1, 2, 2, 3]，但实际接收到的尺寸是 [1, 3, 8]。
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 1, 2}), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 3, 8].");

    // 断言：调用 F::affine_grid 函数，并期望抛出异常，异常信息指出 affine_grid 函数仅支持 4D 和 5D 的尺寸，分别用于 2D 和 3D 仿射变换，但实际接收到的尺寸是 [1, 1, 1, 2, 2, 3]。
    ASSERT_THROWS_WITH(
        F::affine_grid(theta, {1, 1, 1, 2, 2, 3}),
        "affine_grid only supports 4D and 5D sizes, for 2D and 3D affine "
        "transforms, respectively. Got size [1, 1, 1, 2, 2, 3]");

    // 断言：调用 F::affine_grid 函数，并期望抛出异常，异常信息指出 affine_grid 函数仅支持 4D 和 5D 的尺寸，分别用于 2D 和 3D 仿射变换，但实际接收到的尺寸是 [1, 1]。
    ASSERT_THROWS_WITH(
        F::affine_grid(theta, {1, 1}),
        "affine_grid only supports 4D and 5D sizes, for 2D and 3D affine "
        "transforms, respectively. Got size [1, 1]");
}
}

TEST_F(FunctionalTest, MultiMarginLoss) {
  // 定义权重张量
  auto weight = torch::tensor({0.3, 0.3, 0.4}, torch::kFloat);
  // 定义输入张量，设置 requires_grad 为 true
  auto input = torch::tensor(
      {{0.2, 0.2, 0.6}, {0.1, 0.8, 0.1}, {0.9, 0.09, 0.01}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量
  auto target = torch::tensor({2, 1, 0}, torch::kLong);
  // 计算多类别边界损失
  auto output = F::multi_margin_loss(
      input, target, F::MultiMarginLossFuncOptions().margin(2).weight(weight));
  // 定义预期输出
  auto expected = torch::tensor({0.305556}, torch::kFloat);

  // 使用 allclose 方法进行输出和预期输出的比较，允许误差范围为 1e-04
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, CosineEmbeddingLoss) {
  // 定义输入张量1
  auto input1 = torch::tensor({{2, 3, 4}, {6, 2, 4}});
  // 定义输入张量2
  auto input2 = torch::tensor({{2, 3, 5}, {9, 12, 0}});
  // 定义目标张量
  auto target = torch::tensor({1, -1});
  // 计算余弦嵌入损失
  auto output = F::cosine_embedding_loss(
      input1, input2, target, F::CosineEmbeddingLossFuncOptions().margin(0.5));
  // 定义预期输出
  auto expected = torch::tensor({0.1004}, torch::kFloat);

  // 使用 allclose 方法进行输出和预期输出的比较，允许误差范围为 1e-4
  ASSERT_TRUE(output.allclose(expected, 1e-4));
}

TEST_F(FunctionalTest, MultiLabelMarginLossDefaultOptions) {
  // 定义输入张量，设置 requires_grad 为 true
  auto input = torch::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  // 计算多标签边界损失，默认选项
  auto output = F::multilabel_margin_loss(input, target);
  // 定义预期输出
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  // 计算输出张量的和
  auto s = output.sum();
  // 反向传播
  s.backward();

  // 使用 allclose 方法进行输出和预期输出的比较
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和其梯度的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, MultiLabelMarginLossNoReduction) {
  // 定义输入张量，设置 requires_grad 为 true
  auto input = torch::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  // 计算多标签边界损失，无缩减
  auto output = F::multilabel_margin_loss(input, target, torch::kNone);
  // 定义预期输出
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  // 计算输出张量的和
  auto s = output.sum();
  // 反向传播
  s.backward();

  // 使用 allclose 方法进行输出和预期输出的比较
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和其梯度的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, TripletMarginLoss) {
  // 定义锚点张量
  auto anchor = torch::tensor({{3., 3.}}, torch::kFloat);
  // 定义正样本张量
  auto positive = torch::tensor({{2., 2.}}, torch::kFloat);
  // 定义负样本张量
  auto negative = torch::tensor({{0., 0.}}, torch::kFloat);
  // 计算三元边界损失
  auto output = F::triplet_margin_loss(
      anchor,
      positive,
      negative,
      F::TripletMarginLossFuncOptions().margin(1.0));
  // 定义预期输出
  auto expected = torch::tensor({0.}, torch::kFloat);

  // 使用 allclose 方法进行输出和预期输出的比较，允许误差范围为 1e-04
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, TripletMarginWithDistanceLossDefaultParity) {
  // 检查如果使用默认的 TripletMarginLoss 选项，使用 torch::pairwise_distance 作为距离函数，输出是否相等
  std::vector<TripletMarginWithDistanceLossOptions::reduction_t> reductions = {
      torch::kSum, torch::kMean, torch::kNone};
  std::vector<float> margins = {0.5, 1.0, 1.5};
  std::vector<bool> swaps = {true, false};

  for (auto& reduction : reductions) {
    for (auto& margin : margins) {
      // 遍历每个 margin 值，用于 Triplet Margin Loss 函数
      for (const auto& swap : swaps) {
        // 遍历每个 swap 值，用于 Triplet Margin Loss 函数的 swap 参数
        auto anchor = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        // 创建一个大小为 [100, 128] 的随机张量 anchor，用于计算损失并支持梯度
        auto positive = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        // 创建一个大小为 [100, 128] 的随机张量 positive，用于计算损失并支持梯度
        auto negative = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        // 创建一个大小为 [100, 128] 的随机张量 negative，用于计算损失并支持梯度

        auto basicOptions = F::TripletMarginLossFuncOptions()
                                .reduction(reduction)
                                .margin(margin)
                                .swap(swap);
        // 创建 Triplet Margin Loss 的选项对象 basicOptions，设定减少方式、margin 和 swap 参数
        auto distanceOptions = F::TripletMarginWithDistanceLossFuncOptions()
                                   .reduction(reduction)
                                   .margin(margin)
                                   .swap(swap);
        // 创建 Triplet Margin Loss With Distance 的选项对象 distanceOptions，设定减少方式、margin 和 swap 参数

        TripletMarginLoss basicLoss(basicOptions);
        // 使用 basicOptions 创建 TripletMarginLoss 对象 basicLoss
        TripletMarginWithDistanceLoss distanceLoss(distanceOptions);
        // 使用 distanceOptions 创建 TripletMarginWithDistanceLoss 对象 distanceLoss

        auto basicOutput =
            F::triplet_margin_loss(anchor, positive, negative, basicOptions);
        // 计算基本 Triplet Margin Loss，传入 anchor、positive、negative 和 basicOptions
        auto distanceOutput = F::triplet_margin_with_distance_loss(
            anchor, positive, negative, distanceOptions);
        // 计算带距离的 Triplet Margin Loss，传入 anchor、positive、negative 和 distanceOptions

        ASSERT_TRUE(distanceOutput.allclose(basicOutput, 1e-6, 1e-6));
        // 断言带距离的输出与基本输出在一定误差范围内相等

        // 处理 torch::kNone 减少方式
        auto sum = distanceOutput.sum();
        // 计算 distanceOutput 的总和
        sum.backward();
        // 对 sum 进行反向传播
        ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
        // 断言 anchor 和其梯度的尺寸相同
        ASSERT_EQ(positive.sizes(), positive.grad().sizes());
        // 断言 positive 和其梯度的尺寸相同
        ASSERT_EQ(negative.sizes(), negative.grad().sizes());
        // 断言 negative 和其梯度的尺寸相同
      }
    }
}

# 在 FunctionalTest 测试类中定义 NLLLoss 测试用例
TEST_F(FunctionalTest, NLLLoss) {
    # 创建输入张量，包含浮点数值
    auto input = torch::tensor(
        {{-0.1315, -3.1315, -2.5315},
         {-3.7038, -0.1038, -2.6038},
         {-2.3422, -1.3422, -0.4422}},
        torch::kFloat);
    # 创建目标张量，包含长整型数值
    auto target = torch::tensor({1, 0, 2}, torch::kLong);
    # 计算 NLLLoss，指定忽略索引和减少方式为均值
    auto output = F::nll_loss(
        input,
        target,
        F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kMean));
    # 创建期望的输出张量，包含浮点数值
    auto expected = torch::tensor(2.4258, torch::kFloat);
    # 断言输出张量与期望的输出张量非常接近
    ASSERT_TRUE(output.allclose(expected, 1e-04));
    # 再次计算 NLLLoss，并断言输出张量与期望的输出张量非常接近
    ASSERT_TRUE(F::nll_loss(input, target).allclose(expected, 1e-04));
}

# 在 FunctionalTest 测试类中定义 CrossEntropy 测试用例
TEST_F(FunctionalTest, CrossEntropy) {
    # 创建输入张量，包含浮点数值
    auto input = torch::tensor({{3., 3.}, {2., 2.}}, torch::kFloat);
    # 创建目标张量，包含长整型数值
    auto target = torch::tensor({0, 1}, torch::kLong);
    # 计算 CrossEntropy，指定忽略索引和减少方式为均值
    auto output = F::cross_entropy(
        input,
        target,
        F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
    # 创建期望的输出张量，包含浮点数值
    auto expected = torch::tensor(0.6931, torch::kFloat);

    # 断言输出张量与期望的输出张量非常接近
    ASSERT_TRUE(output.allclose(expected, 1e-04));
    # 再次计算 CrossEntropy，并断言输出张量与期望的输出张量非常接近
    ASSERT_TRUE(F::cross_entropy(input, target).allclose(expected, 1e-04));

    # 使用类索引进行标签平滑处理
    input = torch::tensor({{3., 1.}, {1., 2.}}, torch::kFloat);
    output = F::cross_entropy(
        input,
        target,
        F::CrossEntropyFuncOptions().label_smoothing(0.15).reduction(
            torch::kMean));
    expected = torch::tensor(0.3326, torch::kFloat);
    # 断言输出张量与期望的输出张量非常接近
    ASSERT_TRUE(output.allclose(expected, 1e-04));

    # 使用目标概率进行标签平滑处理
    target = torch::tensor({{0.8, 0.2}, {0.1, 0.9}}, torch::kFloat);
    output = F::cross_entropy(
        input,
        target,
        F::CrossEntropyFuncOptions().label_smoothing(0.2).reduction(
            torch::kMean));
    expected = torch::tensor(0.5701, torch::kFloat);
    # 断言输出张量与期望的输出张量非常接近
    ASSERT_TRUE(output.allclose(expected, 1e-04));
}
# 在 FunctionalTest 测试套件中定义了 MaxUnpool1d 测试用例
TEST_F(FunctionalTest, MaxUnpool1d) {
  # 创建一个 Float 类型的张量 x，值为 [[[2, 4, 5]]]，并设置需要计算梯度
  auto x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  # 创建一个 Long 类型的张量 indices，值为 [[[1, 3, 4]]]
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  # 使用 F::max_unpool1d 函数对 x 和 indices 进行最大解池操作，输出大小为 3
  auto y = F::max_unpool1d(x, indices, F::MaxUnpool1dFuncOptions(3));

  # 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  # 断言 y 的值与给定的张量接近
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  # 断言 y 的大小为 [1, 1, 9]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  # 修改 x 的值为 [[2, 4, 5]]，并重新设定计算梯度
  x = torch::tensor(
      {{2, 4, 5}}, torch::dtype(torch::kFloat).requires_grad(true));
  # 修改 indices 的值为 [1, 3, 4]
  indices = torch::tensor({{1, 3, 4}}, torch::kLong);
  # 再次使用 F::max_unpool1d 函数对 x 和 indices 进行最大解池操作，输出大小为 3
  y = F::max_unpool1d(x, indices, F::MaxUnpool1dFuncOptions(3));

  # 断言 y 的维度为 2
  ASSERT_EQ(y.ndimension(), 2);
  # 断言 y 的值与给定的张量接近
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{0, 2, 0, 4, 5, 0, 0, 0, 0}}, torch::kFloat)));
  # 断言 y 的大小为 [1, 9]

  # 修改 x 的值为 [[[2, 4, 5]]]，并重新设定计算梯度
  x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  # 修改 indices 的值为 [[[1, 3, 4]]]
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  # 再次使用 F::max_unpool1d 函数对 x 和 indices 进行最大解池操作，输出大小为 [1, 1, 9]
  y = F::max_unpool1d(
      x,
      indices,
      F::MaxUnpool1dFuncOptions(3).output_size(
          std::vector<int64_t>({1, 1, 9})));

  # 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  # 断言 y 的值与给定的张量接近
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  # 断言 y 的大小为 [1, 1, 9]

  # 修改 x 的值为 [[[2, 4, 5]]]，并重新设定计算梯度
  x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  # 修改 indices 的值为 [[[1, 3, 4]]]
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  # 再次使用 F::max_unpool1d 函数对 x 和 indices 进行最大解池操作，stride 设置为 2，padding 设置为 1
  y = F::max_unpool1d(
      x, indices, F::MaxUnpool1dFuncOptions(3).stride(2).padding(1));

  # 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  # 断言 y 的值与给定的张量接近
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  # 断言 y 的大小为 [1, 1, 5]
}
TEST_F(FunctionalTest, MaxUnpool2d) {
  // 创建一个张量 indices，指定了最大池化操作的索引
  auto indices = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}},
      torch::kLong);
  // 创建一个张量 x，包含了输入数据，类型为浮点数，并允许计算梯度
  auto x = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 对 x 进行最大池化的反池化操作，使用指定的索引和选项
  auto y = F::max_unpool2d(
      x, indices, F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));

  // 断言输出 y 的维度为 4
  ASSERT_EQ(y.dim(), 4);
  // 断言 y 和预期的张量在所有元素上是否接近
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{0, 0, 0, 0, 0},
             {0, 6, 0, 8, 9},
             {0, 0, 0, 0, 0},
             {0, 16, 0, 18, 19},
             {0, 21, 0, 23, 24}}},
           {{{0, 0, 0, 0, 0},
             {0, 31, 0, 33, 34},
             {0, 0, 0, 0, 0},
             {0, 41, 0, 43, 44},
             {0, 46, 0, 48, 49}}}},
          torch::kFloat)));
  // 断言 y 的尺寸是否符合预期
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 1, 5, 5}));

  // 更新 indices 张量的值
  indices = torch::tensor(
      {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
       {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
      torch::kLong);
  // 更新 x 张量的值
  x = torch::tensor(
      {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
       {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 再次执行最大池化的反池化操作，使用更新后的索引和选项
  y = F::max_unpool2d(
      x, indices, F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));

  // 断言输出 y 的维度为 3
  ASSERT_EQ(y.dim(), 3);
  // 断言 y 和预期的张量在所有元素上是否接近
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{0, 0, 0, 0, 0},
            {0, 6, 0, 8, 9},
            {0, 0, 0, 0, 0},
            {0, 16, 0, 18, 19},
            {0, 21, 0, 23, 24}},
           {{0, 0, 0, 0, 0},
            {0, 31, 0, 33, 34},
            {0, 0, 0, 0, 0},
            {0, 41, 0, 43, 44},
            {0, 46, 0, 48, 49}}},
          torch::kFloat)));
  // 断言 y 的尺寸是否符合预期
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 5, 5}));
}

TEST_F(FunctionalTest, MaxUnpool3d) {
  // 创建一个张量 indices，指定了最大池化操作的索引
  auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
  // 创建一个张量 x，包含了输入数据，类型为浮点数，并允许计算梯度
  auto x = torch::tensor(
      {{{{{26}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 对 x 进行最大池化的反池化操作，使用指定的索引和选项
  auto y = F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));

  // 断言输出 y 的维度为 5
  ASSERT_EQ(y.dim(), 5);
  // 断言 y 和预期的张量在所有元素上是否接近
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}}},
          torch::kFloat)));
  // 断言 y 的尺寸是否符合预期
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3, 3, 3}));

  // 更新 indices 张量的值
  indices = torch::tensor({{{{26}}}}, torch::kLong);
  // 更新 x 张量的值
  x = torch::tensor(
      {{{{26}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 再次执行最大池化的反池化操作，使用更新后的索引和选项
  y = F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));

  // 断言输出 y 的维度为 4
  ASSERT_EQ(y.dim(), 4);
  // 断言 y 和预期的张量在所有元素上是否接近
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}},
          torch::kFloat)));
  // 断言 y 的尺寸是否符合预期
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}
TEST_F(FunctionalTest, ELU) {
  // 定义测试用例中的张量大小
  const auto size = 3;
  // 遍历 inplace 参数，分别进行测试
  for (const auto inplace : {false, true}) {
    // 遍历 alpha 参数，分别进行测试
    for (const auto alpha : {0.0, 0.42, 1.0, 4.2, 42.42}) {
      // 创建一个线性空间的张量，并重新调整形状为 size*size*size
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      // 创建一个线性空间的张量，并转换为 BF16 数据类型，然后重新调整形状
      auto x_bf16 =
          torch::linspace(-10.0, 10.0, size * size * size).to(torch::kBFloat16);
      x_bf16.resize_({size, size, size});

      // ELU 激活函数的预期输出计算
      auto y_exp = torch::max(torch::zeros_like(x), x) +
          torch::min(torch::zeros_like(x), alpha * (torch::exp(x) - 1.0));
      // 使用 ELU 激活函数处理张量 x，并根据 inplace 和 alpha 参数配置
      auto y = F::elu(x, F::ELUFuncOptions().alpha(alpha).inplace(inplace));
      // 使用 ELU 激活函数处理 BF16 类型的张量 x_bf16，并根据 inplace 和 alpha 参数配置
      auto y_bf16 =
          F::elu(x_bf16, F::ELUFuncOptions().alpha(alpha).inplace(inplace));

      // 断言输出张量 y 的维度为 3
      ASSERT_EQ(y.ndimension(), 3);
      // 断言输出张量 y 的形状为 {size, size, size}
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      // 断言张量 y 与预期输出 y_exp 在数值上全部接近
      ASSERT_TRUE(torch::allclose(y, y_exp));
      // 断言 BF16 类型的输出张量 y_bf16 转换为 Float 类型后与 y 在数值上全部接近，容忍度为 1e-2
      ASSERT_TRUE(torch::allclose(y_bf16.to(torch::kFloat), y, 1e-2, 1e-2));
      // 如果 inplace 为 true，断言原始张量 x 与预期输出 y_exp 在数值上全部接近
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
        // 断言 BF16 类型的原始张量 x_bf16 转换为 Float 类型后与 y 在数值上全部接近，容忍度为 1e-2
        ASSERT_TRUE(torch::allclose(x_bf16.to(torch::kFloat), y, 1e-2, 1e-2));
      }
    }
  }
  // 断言对于输入张量为标量 1.0 时，ELU 激活函数的输出被定义
  ASSERT_TRUE(F::elu(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, SELU) {
  {
    // 定义 SELU 激活函数的缩放因子和 alpha 参数
    const double scale = 1.0507009873554804934193349852946;
    const double alpha = 1.6732632423543772848170429916717;
    // 遍历 inplace 参数，分别进行测试
    for (const auto inplace : {false, true}) {
      // 创建一个形状为 {5, 5} 的随机正态分布张量作为输入
      auto input = torch::randn({5, 5});
      // 将输入张量复制并转换为 BF16 数据类型
      auto input_bf16 = input.clone().to(torch::kBFloat16);
      // 预期的 SELU 激活函数输出计算
      auto expected = scale *
          (torch::max(torch::zeros_like(input), input) +
           torch::min(
               torch::zeros_like(input), alpha * (torch::exp(input) - 1)));
      // 使用 SELU 激活函数处理输入张量 input，并根据 inplace 参数配置
      auto output = F::selu(input, inplace);
      // 使用 SELU 激活函数处理 BF16 类型的输入张量 input_bf16，并根据 inplace 参数配置
      auto output_bf16 = F::selu(input_bf16, inplace);

      // 断言输出张量 output 与预期输出 expected 在数值上全部接近
      ASSERT_TRUE(output.allclose(expected));
      // 断言 BF16 类型的输出张量 output_bf16 转换为 Float 类型后与 output 在数值上全部接近，容忍度为 1e-2
      ASSERT_TRUE(output_bf16.to(torch::kFloat).allclose(output, 1e-2, 1e-2));
      // 如果 inplace 为 true，断言原始输入张量 input 与预期输出 expected 在数值上全部接近
      if (inplace) {
        ASSERT_TRUE(input.allclose(expected));
        // 断言 BF16 类型的原始输入张量 input_bf16 转换为 Float 类型后与 output 在数值上全部接近，容忍度为 1e-2
        ASSERT_TRUE(input_bf16.to(torch::kFloat).allclose(output, 1e-2, 1e-2));
      }
    }
  }
  {
    // 创建一个形状为 {3, 3} 的张量，从 0 到 8，数据类型为 Double
    auto input = torch::arange(0, 9, torch::kDouble).view({3, 3});
    // 使用 SELU 激活函数处理输入张量 input，期望 inplace 为 false
    auto output = F::selu(input);
    // 使用 SELU 激活函数处理输入张量 input，期望 inplace 为 false
    auto expected = F::selu(input, false);
    // 断言输出张量 output 与预期输出 expected 在数值上全部接近
    ASSERT_TRUE(output.allclose(expected));
  }
  // 断言对于输入张量为标量 1.0 时，SELU 激活函数的输出被定义
  ASSERT_TRUE(F::selu(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, GLU) {
  // 定义 GLU 操作的维度为 1
  int64_t dim = 1;
  // 创建一个形状为 {4, 2} 的正态分布张量，并要求梯度计算
  auto input = torch::randn({4, 2}, torch::requires_grad());
  // 使用 GLU 操作处理输入张量 input，指定操作的维度为 dim
  auto output = F::glu(input, dim);
  // 获取输入张量 input 在指定维度 dim 上的大小
  auto input_size = input.sizes()[dim] / 2;
  // 从 input 中选择 dim 维度上前半部分的张量
  auto first_half = input.narrow(dim, 0, input_size);
  // 从 input 中选择 dim 维度上后半部分的张量
  auto second_half = input.narrow(dim, input_size, input_size);
  // 预期的 GLU 操作输出计算
  auto expected = first_half * torch::sigmoid(second_half);

  // 断言输出张量 output 与预期输出 expected 在数值上全部接近
  ASSERT_TRUE(output.allclose(expected));
  // 断言不指定维度时，GLU 操作处理输入张量 input 的输出与预期输出 expected 在数值上全部接近
  ASSERT_TRUE(F::glu(input).allclose(expected));
}
TEST_F(FunctionalTest, GELU) {
  // 创建一个包含100个元素的张量，范围从-3.0到3.0
  const auto x = torch::linspace(-3.0, 3.0, 100);
  // 计算 GELU 函数的期望输出
  const auto y_exp = x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
  // 使用 GELU 函数计算张量 x 的输出
  const auto y = F::gelu(x, F::GELUFuncOptions().approximate("none"));
  // 断言两个张量是否在给定的误差范围内相等
  ASSERT_TRUE(torch::allclose(y, y_exp, 1.4e-06, 1e-05));
}

TEST_F(FunctionalTest, TanhGELU) {
  // 创建一个包含100个元素的张量，范围从-3.0到3.0
  const auto x = torch::linspace(-3.0, 3.0, 100);
  // 计算 TanhGELU 函数的内部值
  const auto inner = std::sqrt(2 / M_PI) * (x + 0.044715 * x.pow(3.0));
  // 计算 TanhGELU 函数的期望输出
  const auto y_exp = 0.5 * x * (1.0 + inner.tanh());
  // 使用 TanhGELU 函数计算张量 x 的输出
  const auto y = F::gelu(x, F::GELUFuncOptions().approximate("tanh"));
  // 断言两个张量是否在给定的误差范围内相等
  ASSERT_TRUE(torch::allclose(y, y_exp, 1.4e-06, 1e-05));
}

TEST_F(FunctionalTest, Hardshrink) {
  // 定义张量的大小
  const auto size = 3;
  // 遍历不同的 lambda 值
  for (const auto lambda : {-4.2, -1.0, -0.42, 0.0, 0.42, 1.0, 4.2, 42.42}) {
    // 创建一个包含 size*size*size 个元素的张量
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    // 重新调整张量的形状为 size*size*size，并设置 requires_grad 为 true
    x.resize_({size, size, size}).set_requires_grad(true);
    // 使用 Hardshrink 函数计算张量 x 的输出
    auto y = F::hardshrink(x, F::HardshrinkFuncOptions().lambda(lambda));
    // 对输出张量进行求和
    torch::Tensor s = y.sum();

    // 对输出张量进行反向传播
    s.backward();
    // 断言张量的维度为0
    ASSERT_EQ(s.ndimension(), 0);

    // 断言输出张量的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言输出张量的大小符合预期
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 计算期望输出张量
    auto y_exp = (x.abs() > lambda) * x;
    // 断言两个张量是否相等
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(FunctionalTest, OneHot) {
  { // Test #1
    // 创建一个从0到4的长整型张量
    auto x = torch::arange(0, 5, torch::kLong);
    // 使用 OneHot 函数对张量进行编码
    auto y = F::one_hot(x % 3);
    // 创建预期的编码结果张量
    auto expected = torch::tensor(
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, torch::kLong);

    // 断言输出张量的维度为2
    ASSERT_EQ(y.ndimension(), 2);
    // 断言两个张量是否相等
    ASSERT_TRUE(torch::allclose(y, expected));
    // 断言输出张量的大小符合预期
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({5, 3}));
  }

  { // Test #2
    // 创建一个从0到4的长整型张量
    auto x = torch::arange(0, 5, torch::kLong);
    // 使用指定编码长度的 OneHot 函数对张量进行编码
    auto y = F::one_hot(x % 3, 5);
    // 创建预期的编码结果张量
    auto expected = torch::tensor(
        {{1, 0, 0, 0, 0},
         {0, 1, 0, 0, 0},
         {0, 0, 1, 0, 0},
         {1, 0, 0, 0, 0},
         {0, 1, 0, 0, 0}},
        torch::kLong);

    // 断言输出张量的维度为2
    ASSERT_EQ(y.ndimension(), 2);
    // 断言两个张量是否相等
    ASSERT_TRUE(torch::allclose(y, expected));
    // 断言输出张量的大小符合预期
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({5, 5}));
  }

  { // Test #3
    // 创建一个从0到5的长整型张量
    auto x = torch::arange(0, 6, torch::kLong);
    // 使用指定形状的 OneHot 函数对张量进行编码
    auto y = F::one_hot(x.view(std::vector<int64_t>({3, 2})) % 3);
    // 创建预期的编码结果张量
    auto expected = torch::tensor(
        {{{1, 0, 0}, {0, 1, 0}},
         {{0, 0, 1}, {1, 0, 0}},
         {{0, 1, 0}, {0, 0, 1}}},
        torch::kLong);

    // 断言输出张量的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言两个张量是否相等
    ASSERT_TRUE(torch::allclose(y, expected));
    // 断言输出张量的大小符合预期
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({3, 2, 3}));
  }
}

TEST_F(FunctionalTest, Hardtanh) {
  // 定义张量的大小
  const auto size = 3;
  // 遍历不同的 min_val 值
  for (const auto min_val : {-4.2, -1.0, -0.42, 0.0}) {
    // 遍历不同的最大值，用于 Hardtanh 函数的参数测试
    for (const auto max_val : {0.0, 0.42, 1.0, 4.2}) {
      // 遍历是否原地操作的选项
      for (const auto inplace : {false, true}) {
        // 生成一个从-10到10的等间隔张量，并重新调整形状为 size*size*size
        auto x = torch::linspace(-10.0, 10.0, size * size * size);
        x.resize_({size, size, size});
        // 根据不同的最小和最大值，对张量 x 进行 Hardtanh 运算，得到 y_exp
        auto y_exp = (x < min_val) * min_val +
            ((x >= min_val) * (x <= max_val)) * x + (x > max_val) * max_val;
        // 调用 PyTorch 中的 Hardtanh 函数 F::hardtanh，设置最小值、最大值和是否原地操作
        auto y = F::hardtanh(
            x,
            F::HardtanhFuncOptions().min_val(min_val).max_val(max_val).inplace(
                inplace));

        // 断言 y 的维度为3
        ASSERT_EQ(y.ndimension(), 3);
        // 断言 y 的尺寸为 {size, size, size}
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        // 断言 y 与 y_exp 在数值上的近似性
        ASSERT_TRUE(torch::allclose(y, y_exp));
        // 如果进行了原地操作，则断言 x 和 y_exp 在数值上的近似性
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
  // 断言 Hardtanh 函数应用于标量值 1 后仍然定义
  ASSERT_TRUE(F::hardtanh(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, LeakyReLU) {
  // 定义测试用例的尺寸大小
  const auto size = 3;
  // 遍历不同的负斜率参数
  for (const auto negative_slope : {0.0, 0.42, 1.0}) {
    // 遍历是否原地操作的选项
    for (const auto inplace : {false, true}) {
      // 遍历张量类型（float或BFloat16）
      for (const auto type : {torch::kFloat, torch::kBFloat16}) {
        // 创建一个线性空间的张量，并转换为指定类型
        auto x = torch::linspace(-10.0, 10.0, size * size * size).to(type);
        // 重设张量的形状为(size, size, size)
        x.resize_({size, size, size});
        // 计算预期的输出值，根据负斜率应用Leaky ReLU函数
        auto y_exp = (x < 0) * x * negative_slope + (x >= 0) * x;
        // 调用Leaky ReLU函数计算输出y
        auto y = F::leaky_relu(
            x,
            F::LeakyReLUFuncOptions()
                .negative_slope(negative_slope)
                .inplace(inplace));

        // 断言输出y的维度为3
        ASSERT_EQ(y.ndimension(), 3);
        // 断言输出y的形状为(size, size, size)
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        // 断言y与预期输出y_exp的近似程度
        ASSERT_TRUE(torch::allclose(y, y_exp));
        // 如果是原地操作，断言输入x与预期输出y_exp的近似程度
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
  // 断言对输入为1的张量应用logsigmoid函数后的定义
  ASSERT_TRUE(F::leaky_relu(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, LogSigmoid) {
  // 定义测试用例的尺寸大小
  const auto size = 3;
  // 创建LogSigmoid模型实例
  LogSigmoid model;
  // 创建一个线性空间的张量
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  // 重设张量的形状为(size, size, size)
  x.resize_({size, size, size});
  // 调用LogSigmoid函数计算输出y
  auto y = F::logsigmoid(x);

  // 断言输出y的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出y的形状为(size, size, size)
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  // 计算预期的输出y_exp，根据LogSigmoid函数的定义
  auto y_exp = torch::log(
      torch::ones_like(x) / (torch::ones_like(x) + torch::exp(torch::neg(x))));
  // 断言y与预期输出y_exp的近似程度
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

TEST_F(FunctionalTest, GumbelSoftmax) {
  // Test 1: No-options
  {
    // 创建服从标准正态分布的随机张量logits
    auto logits = torch::randn({5});
    int expected_count = 1;
    // 调用Gumbel Softmax函数计算输出y_draw
    auto y_draw = F::gumbel_softmax(logits);

    // 断言y_draw中所有值都为非负数
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // 断言y_draw的形状与logits相同
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // 断言每次抽样只有一个选择
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  // Test 2: 1D shape, 0 and -1 dim
  for (const auto dim : {0, -1}) {
    // 创建服从标准正态分布的随机张量logits
    auto logits = torch::randn({5});
    int expected_count = 1;
    // 调用Gumbel Softmax函数计算输出y_draw
    auto y_draw = F::gumbel_softmax(
        logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(dim));

    // 断言y_draw中所有值都为非负数
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // 断言y_draw的形状与logits相同
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // 断言每次抽样只有一个选择
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  { // Test 3: 2D shape, 1 dim
    // 创建服从标准正态分布的随机张量logits
    auto logits = torch::randn({5, 4});
    int expected_count = 5;
    // 调用Gumbel Softmax函数计算输出y_draw
    auto y_draw = F::gumbel_softmax(
        logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(1));

    // 断言y_draw中所有值都为非负数
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // 断言y_draw的形状与logits相同
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // 断言每次抽样只有一个选择
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }
}
  // 断言：验证 y_draw 的总和是否与预期的数量相等
  ASSERT_TRUE(torch::allclose(
      y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
}

{ // Test 4: 3D shape, 1 and -1 dim
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  // 定义维度数组，包括 1 和 -1
  int dims[] = {1, -1};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
  // 定义期望的元素数量数组
  int expected[] = {5 * 3, 5 * 4};
  for (const auto i : c10::irange(2)) {
    // 创建随机数张量，形状为 {5, 4, 3}
    auto logits = torch::randn({5, 4, 3});
    // 获取当前期望的元素数量
    int expected_count = expected[i];
    // 使用 Gumbel Softmax 函数生成 y_draw 张量
    auto y_draw = F::gumbel_softmax(
        logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(dims[i]));

    // 断言：所有值均为正数
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // 断言：形状不变
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // 断言：每次抽取的和应该等于期望的数量
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }
}

{ // Test 5: Straight through
  // 定义抽取次数
  int num_draws = 100;
  // 定义 logits 张量，形状为 {1, 3}，并对其进行 softmax 操作
  auto logits = torch::tensor({{0.2, 0.8, 0.1}});
  logits = logits.reshape({1, 3});
  logits.requires_grad();
  auto probs = logits.softmax(-1);

  // 初始化 counts 张量，并定义 y_draw 张量
  auto counts = torch::zeros_like(logits);
  torch::Tensor y_draw;
  // 循环进行 num_draws 次抽取
  for (const auto i : c10::irange(num_draws)) {
    (void)i; // 抑制未使用变量警告
    // 生成使用 Gumbel Softmax 函数生成的 y_draw 张量
    y_draw =
        F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true));
    // 更新 counts 张量
    counts += y_draw;
  }

  // 断言：所有值均为正数
  ASSERT_GE(y_draw.min().item<int>(), 0);
  // 断言：每次实验应该抽取出 1 个值
  ASSERT_EQ(counts.sum().item<int>(), num_draws);

  // 检查结果是否接近预期
  auto expected = probs * num_draws;
  // 计算 z 分数用于检查是否接近标准正态分布
  auto z = (counts - expected) / (expected * (1 - probs)).sqrt();
  // 进行大约 99% 的双侧检验
  ASSERT_LT(z.abs().max().item<float>(), 2.58);
}
TEST_F(FunctionalTest, Softmax) {
  // 创建一个 2x5 的浮点数张量 input，其值为从0到9
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // 对 input 张量进行 softmax 操作，沿着 dim=1 的维度
  auto output = F::softmax(input, /*dim=*/1);
  // 计算每行上指数的和
  auto sum = torch::sum(torch::exp(input), 1);

  // 对每一行进行验证
  for (const auto i : c10::irange(2)) {
    // 计算预期的 softmax 结果
    auto expected = torch::exp(input[i]) / sum[i];
    // 断言 output[i] 与 expected 在数值上的接近性
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(FunctionalTest, Softmin) {
  // 创建一个 2x5 的浮点数张量 input，其值为从0到9
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // 对 input 张量进行 softmin 操作，沿着 dim=1 的维度
  auto output = F::softmin(input, /*dim=*/1);
  // 计算每行上指数的和
  auto sum = torch::sum(torch::exp(-input), 1);

  // 对每一行进行验证
  for (const auto i : c10::irange(2)) {
    // 计算预期的 softmin 结果
    auto expected = torch::exp(-input[i]) / sum[i];
    // 断言 output[i] 与 expected 在数值上的接近性
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(FunctionalTest, LogSoftmax) {
  // 创建一个 2x5 的浮点数张量 input，其值为从0到9
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // 对 input 张量进行 log_softmax 操作，沿着 dim=1 的维度
  auto output = F::log_softmax(input, /*dim=*/1);
  // 计算每行上指数的和
  auto sum = torch::sum(torch::exp(input), 1);

  // 对每一行进行验证
  for (const auto i : c10::irange(2)) {
    // 计算预期的 log_softmax 结果
    auto expected = torch::log(torch::exp(input[i]) / sum[i]);
    // 断言 output[i] 与 expected 在数值上的接近性
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(FunctionalTest, PReLU) {
  // 创建一个大小为 42x24 的随机浮点数张量 x
  const auto x = torch::rand({42, 24}) * 200 - 100;
  // 创建一个大小为 24 的随机浮点数张量 w
  const auto w = torch::rand(24) * 200 - 100;
  // 对 x 张量应用 PReLU 激活函数
  const auto y = F::prelu(x, w);
  // 断言输出张量 y 的维度符合预期
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({42, 24}));
  // 计算预期的 PReLU 输出
  const auto y_exp = (x < 0) * w * x + (x >= 0) * x;
  // 断言 y 与预期输出 y_exp 在数值上的接近性
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(FunctionalTest, LayerNorm) {
  // 创建一个大小为 2x2 的随机正态分布张量 input
  const auto input = torch::randn({2, 2});
  // 对 input 张量应用 LayerNorm 操作，设置 epsilon 为 2e-5
  auto y = F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
  // 计算预期的 LayerNorm 输出 y_exp
  auto y_exp =
      torch::layer_norm(input, {2, 2}, torch::Tensor(), torch::Tensor(), 2e-5);
  // 断言 y 与预期输出 y_exp 在数值上的接近性
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(FunctionalTest, GroupNorm) {
  // 创建一个大小为 2x2 的随机正态分布张量 input
  const auto input = torch::randn({2, 2});
  // 对 input 张量应用 GroupNorm 操作，设置 epsilon 为 2e-5
  auto y = F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
  // 计算预期的 GroupNorm 输出 y_exp
  auto y_exp =
      torch::group_norm(input, 2, torch::Tensor(), torch::Tensor(), 2e-5);
  // 断言 y 与预期输出 y_exp 在数值上的接近性
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(FunctionalTest, LocalResponseNorm) {
  // 创建一个大小为 3x3x2 的张量 x，其值为从 100 到 117
  const auto x = torch::arange(100, 118).resize_({3, 3, 2});
  // 对 x 张量应用 LocalResponseNorm 操作，设置参数 size 为 2
  const auto y = F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
  // 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言 y 的大小符合预期
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 2}));
  // 创建预期的 LocalResponseNorm 输出 y_exp
  const auto y_exp = torch::tensor(
      {{{73.7788, 74.1462}, {60.1942, 60.3302}, {60.4609, 60.5865}},
       {{75.8729, 76.2011}, {60.9331, 61.0390}, {61.1403, 61.2370}},
       {{77.7387, 78.0303}, {61.5011, 61.5807}, {61.6563, 61.7279}}},
      torch::kFloat);
  // 断言 y 与预期输出 y_exp 在数值上的接近性，设定容差为 1e-4 和 1e-7
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

TEST_F(FunctionalTest, Linear) {
  {
    // 创建一个大小为 3x3x2 的张量 x，其值为从 100 到 117
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});
    // 创建一个大小为 3x2 的张量 w，其值为从 200 到 205
    const auto w = torch::arange(200., 206).resize_({3, 2});
    // 创建一个大小为 3 的张量 b，其值为从 300 到 302
    const auto b = torch::arange(300., 303);
    // 对 x 张量应用 Linear 操作，使用 w 和 b
    const auto y = F::linear(x, w, b);
    // 断言 y 的维度为 3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言确保张量 y 的尺寸为 {3, 3, 3}
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 3}));
    // 定义期望的张量 y_exp，包含预期的浮点数值
    const auto y_exp = torch::tensor(
        {{{40601, 41004, 41407}, {41403, 41814, 42225}, {42205, 42624, 43043}},
         {{43007, 43434, 43861}, {43809, 44244, 44679}, {44611, 45054, 45497}},
         {{45413, 45864, 46315}, {46215, 46674, 47133}, {47017, 47484, 47951}}},
        torch::kFloat);
    // 断言确保张量 y 和预期的 y_exp 在数值上接近
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
  {
    // 创建张量 x，值为从100到117的连续整数，reshape成 {3, 3, 2} 的形状
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});
    // 创建张量 w，值为从200到205的连续整数，reshape成 {3, 2} 的形状
    const auto w = torch::arange(200., 206).resize_({3, 2});
    // 对张量 x 和 w 进行线性变换，计算得到张量 y
    const auto y = F::linear(x, w);
    // 断言确保张量 y 的维度为 3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言确保张量 y 的尺寸为 {3, 3, 3}
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 3}));
    // 定义期望的张量 y_exp，包含预期的整数值
    const auto y_exp = torch::tensor(
        {{{40301, 40703, 41105}, {41103, 41513, 41923}, {41905, 42323, 42741}},
         {{42707, 43133, 43559}, {43509, 43943, 44377}, {44311, 44753, 45195}},
         {{45113, 45563, 46013}, {45915, 46373, 46831}, {46717, 47183, 47649}}},
        torch::kFloat);
    // 断言确保张量 y 和预期的 y_exp 在数值上接近
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
TEST_F(FunctionalTest, Normalize) {
  const auto expected = torch::tensor(
      {{{0.00000000, 0.10000000, 0.2000, 0.30000000, 0.40000000},
        {0.14285715, 0.17142858, 0.2000, 0.22857143, 0.25714287}}},
      torch::requires_grad().dtype(torch::kFloat));
  { // Test #1

    // 创建一个输入张量，包含两个 1x5 的浮点数子张量，并启用梯度计算
    auto input = torch::tensor(
        {{{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}},
        torch::dtype(torch::kFloat).requires_grad(true));
    // 使用 F 命名空间中的 normalize 函数对输入进行标准化，采用 L1 范数进行标准化，并沿着最后一个维度进行操作
    auto norm = F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));

    // 将标准化后的张量求和，得到一个标量，以便调用 .backward() 进行反向传播
    torch::Tensor s = norm.sum();
    // 对标量 s 进行反向传播
    s.backward();

    // 断言：确保 s 是一个标量（即无维度），用于确认正确调用了 .backward()
    ASSERT_EQ(s.ndimension(), 0);
    // 断言：确保 input 的梯度张量的元素数量为 10
    ASSERT_EQ(input.grad().numel(), 10);
    // 断言：确保标准化后的结果 norm 与预期的结果 expected 很接近
    ASSERT_TRUE(torch::allclose(norm, expected));
  }

  { // Test #2 检查可选参数的变化情况
    // 创建一个输入张量 input
    auto input = torch::tensor(
        {{{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}}, torch::dtype(torch::kFloat));
    // 创建一个形状为 {1, 2, 5} 的随机张量 output
    auto output = torch::randn({1, 2, 5}, torch::dtype(torch::kFloat));
    // 调用 normalize 函数，使用指定的可选参数 p=1 和 dim=-1 进行标准化，并将结果输出到指定的输出张量 output 中
    F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1).out(output));
    // 使用默认选项调用 normalize 函数
    F::normalize(input);

    // 断言：确保输出张量 output 的内容与预期的结果 expected 很接近
    ASSERT_TRUE(torch::allclose(output, expected));
  }

  { // Test #3 标量张量的基本情况
    // 创建一个标量张量 input，并要求其梯度
    auto input = torch::randn({}, torch::requires_grad());
    // 对标量张量 input 进行标准化，采用 L1 范数进行标准化，并沿着最后一个维度进行操作
    torch::Tensor norm =
        F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
    // 对标准化后的结果进行反向传播
    norm.backward();

    // 断言：确保 input 的梯度张量的元素数量为 1
    ASSERT_EQ(input.grad().numel(), 1);
  }
}

TEST_F(FunctionalTest, ReLU) {
  // 设置张量的维度大小
  const auto size = 3;
  // 针对是否原地操作的两种情况进行循环
  for (const auto inplace : {false, true}) {
    // 创建线性空间张量，并重新调整为三维
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    // 计算期望的ReLU结果
    auto y_exp = (x < 0) * 0 + (x >= 0) * x;
    // 调用ReLU函数，可以选择是否原地操作
    auto y = F::relu(x, F::ReLUFuncOptions().inplace(inplace));

    // 断言张量的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言张量的大小为指定的三维大小
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 断言ReLU后的结果与期望的结果相近
    ASSERT_TRUE(torch::allclose(y, y_exp));
    // 如果选择了原地操作，则断言原始张量也与期望结果相近
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }

    // NOLINTNEXTLINE(bugprone-argument-comment)
    // 使用NOLINTNEXTLINE来禁止Lint警告并添加注释
    y = F::relu(x, /*inplace=*/inplace);

    // 再次断言张量的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 再次断言张量的大小为指定的三维大小
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 再次断言ReLU后的结果与期望的结果相近
    ASSERT_TRUE(torch::allclose(y, y_exp));
    // 如果选择了原地操作，则再次断言原始张量也与期望结果相近
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }
  }
  // 断言对于单个输入的ReLU函数，输出结果被定义
  ASSERT_TRUE(F::relu(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, ReLUDefaultOptions) {
  // 设置张量的维度大小
  const auto size = 3;
  // 创建线性空间张量，并重新调整为三维
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  // 计算期望的ReLU结果
  auto y_exp = (x < 0) * 0 + (x >= 0) * x;
  // 调用ReLU函数，使用默认选项（非原地操作）
  auto y = F::relu(x);

  // 断言张量的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言张量的大小为指定的三维大小
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  // 断言ReLU后的结果与期望的结果相近
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(FunctionalTest, ReLU6) {
  // 设置张量的维度大小
  const auto size = 3;
  // 针对是否原地操作的两种情况进行循环
  for (const auto inplace : {false, true}) {
    // 创建线性空间张量，并重新调整为三维
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    // 计算期望的ReLU6结果
    auto y_exp = (x < 0) * 0 + ((x >= 0) * (x <= 6)) * x + (x > 6) * 6;
    // 调用ReLU6函数，可以选择是否原地操作
    auto y = F::relu6(x, F::ReLU6FuncOptions().inplace(inplace));

    // 断言张量的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言张量的大小为指定的三维大小
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 断言ReLU6后的结果与期望的结果相近
    ASSERT_TRUE(torch::allclose(y, y_exp));
    // 如果选择了原地操作，则断言原始张量也与期望结果相近
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }

    // NOLINTNEXTLINE(bugprone-argument-comment)
    // 使用NOLINTNEXTLINE来禁止Lint警告并添加注释
    y = F::relu6(x, /*inplace=*/inplace);

    // 再次断言张量的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 再次断言张量的大小为指定的三维大小
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 再次断言ReLU6后的结果与期望的结果相近
    ASSERT_TRUE(torch::allclose(y, y_exp));
    // 如果选择了原地操作，则再次断言原始张量也与期望结果相近
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }
  }
  // 断言对于单个输入的ReLU6函数，输出结果被定义
  ASSERT_TRUE(F::relu6(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, ReLU6DefaultOptions) {
  // 设置张量的维度大小
  const auto size = 3;
  // 创建线性空间张量，并重新调整为三维
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  // 计算期望的ReLU6结果
  auto y_exp = (x < 0) * 0 + ((x >= 0) * (x <= 6)) * x + (x > 6) * 6;
  // 调用ReLU6函数，使用默认选项（非原地操作）
  auto y = F::relu6(x);

  // 断言张量的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言张量的大小为指定的三维大小
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  // 断言ReLU6后的结果与期望的结果相近
  ASSERT_TRUE(torch::allclose(y, y_exp));
}
    // 对于不同的上界值，执行三次循环
    for (const auto upper : {0.3, 0.4, 0.5}) {
      // 对于是否原地操作，执行两次循环
      for (const auto inplace : {false, true}) {
        // 对于数据类型（float 或者 bfloat16），执行两次循环
        for (const auto type : {torch::kFloat, torch::kBFloat16}) {
          // 创建一个从-10到10均匀分布的张量，并转换为指定类型
          auto x = torch::linspace(-10.0, 10.0, size * size * size).to(type);
          // 将张量形状改变为三维大小为 size*size*size
          x.resize_({size, size, size});
          // 克隆原始张量 x
          auto x_copy = x.clone();
          // 对 x 应用随机化的ReLU（RReLU）函数，根据给定的参数设置
          auto y = F::rrelu(
              x,
              F::RReLUFuncOptions().lower(lower).upper(upper).inplace(inplace));
          // 计算二值张量 z，判断 RReLU 函数是否正确应用
          auto z =
              ((x_copy >= 0) * (x_copy == y) +
               (x_copy < 0) * (y >= x_copy * upper) * (y <= lower * x_copy)) *
              1.0;

          // 断言 y 是一个三维张量
          ASSERT_EQ(y.ndimension(), 3);
          // 断言 y 的形状为 [size, size, size]
          ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
          // 断言 z 与全为1的张量形状相同，即 RReLU 函数应用正确
          ASSERT_TRUE(torch::allclose(z, torch::ones_like(z)));
          // 如果 inplace 为 true，则断言 x 等于 y，即原地操作成功
          if (inplace) {
            ASSERT_TRUE(torch::allclose(x, y));
          }
        }
      }
    }
  }
  // 断言 F::rrelu(torch::tensor(1.)) 已定义
  ASSERT_TRUE(F::rrelu(torch::tensor(1.)).defined());
}

# 定义一个测试用例 FunctionalTest.RReLUDefaultOptions，测试 torch 中的 RReLU 函数的默认选项
TEST_F(FunctionalTest, RReLUDefaultOptions) {
  # 定义测试中使用的尺寸大小
  const auto size = 3;
  # 定义 RReLU 函数的下界和上界
  const auto lower = 1.0 / 8.0;
  const auto upper = 1.0 / 3.0;
  # 对于每种数据类型进行循环测试：torch::kFloat 和 torch::kBFloat16
  for (const auto type : {torch::kFloat, torch::kBFloat16}) {
    # 创建一个从 -10 到 10 的线性空间，并转换为指定类型
    auto x = torch::linspace(-10.0, 10.0, size * size * size).to(type);
    # 重新调整 x 的形状为 size x size x size
    x.resize_({size, size, size});
    # 创建 x 的一个克隆版本 x_copy
    auto x_copy = x.clone();
    # 对 x 应用 RReLU 函数，得到 y
    auto y = F::rrelu(x);
    # 根据 RReLU 的公式计算 z
    auto z = ((x_copy >= 0) * (x_copy == y) +
              (x_copy < 0) * (y >= x_copy * upper) * (y <= lower * x_copy)) *
        1.0;

    # 断言 y 的维度为 3
    ASSERT_EQ(y.ndimension(), 3);
    # 断言 y 的大小为 [size, size, size]
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    # 断言 z 与 torch::ones_like(z) 在数值上全等
    ASSERT_TRUE(torch::allclose(z, torch::ones_like(z)));
  }
}

# 定义一个测试用例 FunctionalTest.CELU，测试 torch 中的 CELU 函数
TEST_F(FunctionalTest, CELU) {
  # 定义测试中使用的尺寸大小
  const auto size = 3;
  # 对于 inplace 参数和 alpha 参数进行循环测试
  for (const auto inplace : {false, true}) {
    for (const auto alpha : {0.42, 1.0, 4.2, 42.42}) {
      # 创建一个从 -10 到 10 的线性空间，并重新调整形状为 size x size x size
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      # 创建 x 的 kBFloat16 类型的克隆版本 x_bf16
      auto x_bf16 = x.clone().to(torch::kBFloat16);
      # 根据 CELU 的公式计算期望的 y_exp
      auto y_exp = torch::max(torch::zeros_like(x), x) +
          torch::min(torch::zeros_like(x),
                     alpha * (torch::exp(x / alpha) - 1.0));
      # 调用 CELU 函数计算 y，使用指定的 alpha 和 inplace 参数
      auto y = F::celu(x, F::CELUFuncOptions().alpha(alpha).inplace(inplace));
      # 在 kBFloat16 数据类型上调用 CELU 函数计算 y_bf16
      auto y_bf16 =
          F::celu(x_bf16, F::CELUFuncOptions().alpha(alpha).inplace(inplace));

      # 断言 y 的维度为 3
      ASSERT_EQ(y.ndimension(), 3);
      # 断言 y 的大小为 [size, size, size]
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      # 断言 y 与 y_exp 在数值上全等
      ASSERT_TRUE(torch::allclose(y, y_exp));
      # 断言 y_bf16 转换为 kFloat 后与 y 在数值上接近，精度为 1e-2
      ASSERT_TRUE(torch::allclose(y_bf16.to(torch::kFloat), y, 1e-2, 1e-2));
      # 如果 inplace 为 true，则断言 x 与 y_exp 在数值上全等
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
        # 断言 x_bf16 转换为 kFloat 后与 y 在数值上接近，精度为 1e-2
        ASSERT_TRUE(torch::allclose(x_bf16.to(torch::kFloat), y, 1e-2, 1e-2));
      }
    }
  }
  # 断言 F::celu(torch::tensor(1.)) 是定义的
  ASSERT_TRUE(F::celu(torch::tensor(1.)).defined());
}

# 定义一个测试用例 FunctionalTest.CELUDefaultOptions，测试 torch 中的 CELU 函数的默认选项
TEST_F(FunctionalTest, CELUDefaultOptions) {
  # 定义测试中使用的尺寸大小
  const auto size = 3;
  # 定义 alpha 参数为 1.0
  const auto alpha = 1.0;
  # 创建一个从 -10 到 10 的线性空间，并重新调整形状为 size x size x size
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  # 创建 x 的 kBFloat16 类型的克隆版本 x_bf16
  auto x_bf16 = x.clone().to(torch::kBFloat16);
  # 根据 CELU 的公式计算期望的 y_exp
  auto y_exp = torch::max(torch::zeros_like(x), x) +
      torch::min(torch::zeros_like(x), alpha * (torch::exp(x / alpha) - 1.0));
  # 调用 CELU 函数计算 y
  auto y = F::celu(x);
  # 在 kBFloat16 数据类型上调用 CELU 函数计算 y_bf16
  auto y_bf16 = F::celu(x_bf16);

  # 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  # 断言 y 的大小为 [size, size, size]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  # 断言 y 与 y_exp 在数值上全等
  ASSERT_TRUE(torch::allclose(y, y_exp));
  # 断言 y_bf16 转换为 kFloat 后与 y 在数值上接近，精度为 1e-2
  ASSERT_TRUE(torch::allclose(y_bf16.to(torch::kFloat), y, 1e-2, 1e-2));
}

# 定义一个测试用例 FunctionalTest.PixelShuffle，测试 torch 中的 PixelShuffle 函数
TEST_F(FunctionalTest, PixelShuffle) {
  # 创建一个输入张量 x
  auto x = torch::tensor(
      {{{{-17, 19}, {-1, 2}},
        {{7, 14}, {-3, 1}},
        {{0, -2}, {-12, 14}},
        {{-15, 0}, {-3, 9}}}},
      torch::kFloat);
  # 创建期望的输出张量 y_exp
  auto y_exp = torch::tensor(
      {{{{-17, 7, 19, 14}, {0, -15, -2, 0}, {-1, -3, 2, 1}, {-12, -3, 14, 9}}}},
      torch::kFloat);
  # 调用 PixelShuffle 函数计算输出张量 y
  auto y = F::pixel_shuffle(x, 2);

  # 断言 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  # 断言 y 的大小为 [1, 1, 4, 4]
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 4, 4}));
  # 断言 y 与 y_exp 在数值上接近
  ASSERT_TRUE(y.allclose(y_exp));
}
TEST_F(FunctionalTest, PixelUnshuffle) {
  auto x = torch::tensor(
      {{{{-17, 7, 19, 14}, {0, -15, -2, 0}, {-1, -3, 2, 1}, {-12, -3, 14, 9}}}},
      torch::kFloat);
  auto y_exp = torch::tensor(
      {{{{-17, 19}, {-1, 2}},
        {{7, 14}, {-3, 1}},
        {{0, -2}, {-12, 14}},
        {{-15, 0}, {-3, 9}}}},
      torch::kFloat);
  // 调用像素重排函数，将输入张量 x 按照指定的块大小进行重排
  auto y = F::pixel_unshuffle(x, 2);

  // 断言输出张量 y 的维度为4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言输出张量 y 的尺寸与预期相符
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 4, 2, 2}));
  // 断言输出张量 y 与预期输出张量 y_exp 在误差范围内相等
  ASSERT_TRUE(y.allclose(y_exp));
}

TEST_F(FunctionalTest, Softplus) {
  const auto size = 3;
  // 遍历不同的 beta 和 threshold 参数组合
  for (const auto beta : {0.5, 1.0, 2.0}) {
    for (const auto threshold : {1.0, 3.0, 5.0}) {
      // 创建输入张量 x，并按照指定尺寸重排
      auto x = torch::linspace(-3.0, 3.0, 61);
      x.resize_({size, size, size});
      // 计算预期的输出张量 y_exp，根据条件分段计算 softplus 函数
      auto y_exp =
          (x <= threshold) * torch::log(1 + torch::exp(x * beta)) / beta +
          (x > threshold) * x;
      // 调用 softplus 函数，计算实际输出张量 y
      auto y = F::softplus(
          x, F::SoftplusFuncOptions().beta(beta).threshold(threshold));

      // 断言输出张量 y 的维度为3
      ASSERT_EQ(y.ndimension(), 3);
      // 断言输出张量 y 的尺寸与预期相符
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      // 断言输出张量 y 与预期输出张量 y_exp 在误差范围内相等
      ASSERT_TRUE(torch::allclose(y, y_exp));
    }
  }
}

TEST_F(FunctionalTest, SoftplusDefaultOptions) {
  const auto size = 3;
  const auto beta = 1.0;
  const auto threshold = 20.0;
  // 创建输入张量 x，并按照指定尺寸重排
  auto x = torch::linspace(-3.0, 3.0, 61);
  x.resize_({size, size, size});
  // 计算预期的输出张量 y_exp，根据条件分段计算 softplus 函数
  auto y_exp = (x <= threshold) * torch::log(1 + torch::exp(x * beta)) / beta +
      (x > threshold) * x;
  // 调用 softplus 函数，计算实际输出张量 y
  auto y = F::softplus(x);

  // 断言输出张量 y 的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 的尺寸与预期相符
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  // 断言输出张量 y 与预期输出张量 y_exp 在误差范围内相等
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(FunctionalTest, Fold) {
  // 创建输入张量 input，全为1
  auto input = torch::ones({1, 3 * 2 * 2, 2}, torch::kDouble);
  // 调用 fold 函数，将输入张量按照指定参数进行折叠操作，得到输出张量 output
  auto output = F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
  // 创建预期的输出张量 expected
  auto expected = torch::tensor(
      {{{{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
        {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
        {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}}}},
      torch::kDouble);

  // 断言输出张量 output 的尺寸与预期相符
  ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 3, 2}));
  // 断言输出张量 output 与预期输出张量 expected 在误差范围内相等
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, Unfold) {
  // 创建输入张量 input，从0到11的序列
  auto input = torch::arange(0, 12, torch::kDouble).view({1, 2, 2, 3});
  // 调用 unfold 函数，根据指定参数对输入张量进行展开操作，得到输出张量 output
  auto output =
      F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
  // 创建预期的输出张量 expected
  auto expected = torch::tensor(
      {{{0.0, 0.0, 0.0, 4.0},
        {0.0, 0.0, 3.0, 5.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 10.0},
        {0.0, 0.0, 9.0, 11.0},
        {0.0, 7.0, 0.0, 0.0},
        {6.0, 8.0, 0.0, 0.0}}},
      torch::kDouble);

  // 断言输出张量 output 的尺寸与预期相符
  ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 8, 4}));
  // 断言输出张量 output 与预期输出张量 expected 在误差范围内相等
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, Softshrink) {
  const auto size = 3;
  // 遍历不同的 lambda 参数
  for (const auto lambda : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    // 创建输入张量 x，并按照指定尺寸重排，同时设置梯度计算为真
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    // NOLINTNEXTLINE(bugprone-argument-comment)
    // 使用 F::softshrink 函数对张量 x 进行软阈值处理，使用 lambda 参数作为阈值
    auto y = F::softshrink(x, /*lambda=*/lambda);
    
    // 计算张量 y 的所有元素的和
    torch::Tensor s = y.sum();
    
    // 反向传播计算张量 s 的梯度
    s.backward();
    
    // 断言张量 s 是零维的
    ASSERT_EQ(s.ndimension(), 0);
    
    // 断言张量 y 是三维的
    ASSERT_EQ(y.ndimension(), 3);
    
    // 断言张量 y 的尺寸是 [size, size, size]
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    
    // 计算预期的软阈值处理后的张量 y_exp
    auto y_exp = (x < -lambda) * (x + lambda) + (x > lambda) * (x - lambda);
    
    // 断言张量 y 与预期的 y_exp 在数值上相近
    ASSERT_TRUE(torch::allclose(y, y_exp));
}

# 定义一个测试函数 FunctionalTest，测试 softshrink 函数的默认选项
TEST_F(FunctionalTest, SoftshrinkDefaultOptions) {
  # 定义张量的大小和 lambda 值
  const auto size = 3;
  const auto lambda = 0.5;
  # 生成从 -10 到 10 的等间隔张量，并设定其大小和梯度要求为真
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size}).set_requires_grad(true);
  # 对 x 应用 softshrink 激活函数
  auto y = F::softshrink(x);
  # 计算 y 的和
  torch::Tensor s = y.sum();

  # 对 s 进行反向传播
  s.backward();
  # 断言 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);

  # 断言 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  # 断言 y 的大小与给定的大小向量相同
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  # 计算期望的 y_exp
  auto y_exp = (x < -lambda) * (x + lambda) + (x > lambda) * (x - lambda);
}

# 定义一个测试函数 FunctionalTest，测试 softsign 函数
TEST_F(FunctionalTest, Softsign) {
  # 生成服从标准正态分布的张量 x
  auto x = torch::randn(100) * 10;
  # 计算 softsign 函数的期望输出 y_exp
  auto y_exp = x / (1 + x.abs());
  # 应用 softsign 函数计算 y
  auto y = F::softsign(x);

  # 断言 y 与 y_exp 全部接近
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

# 定义一个测试函数 FunctionalTest，测试 mish 函数
TEST_F(FunctionalTest, Mish) {
  # 生成服从标准正态分布的张量 x
  auto x = torch::randn(100) * 10;
  # 计算 mish 函数的期望输出 y_exp
  auto y_exp = x * x.exp().log1p().tanh();
  # 应用 mish 函数计算 y
  auto y = F::mish(x);

  # 断言 y 与 y_exp 全部接近
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

# 定义一个测试函数 FunctionalTest，测试 tanhshrink 函数
TEST_F(FunctionalTest, Tanhshrink) {
  # 生成服从标准正态分布的张量 x
  auto x = torch::randn(100) * 10;
  # 计算 tanhshrink 函数的期望输出 y_exp
  auto y_exp = x - x.tanh();
  # 应用 tanhshrink 函数计算 y
  auto y = F::tanhshrink(x);

  # 断言 y 与 y_exp 全部接近
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

# 定义一个测试函数 FunctionalTest，测试 threshold 函数
TEST_F(FunctionalTest, Threshold) {
  # 定义大小为 3 的循环，测试不同的 threshold、value 和 inplace 参数组合
  const auto size = 3;
  for (const auto threshold : {0.5, 1.0, 2.0}) {
    for (const auto value : {0.5, 1.0, 2.0}) {
      for (const auto inplace : {false, true}) {
        # 生成从 -3 到 3 的等间隔张量 x
        auto x = torch::linspace(-3.0, 3.0, 61);
        x.resize_({size, size, size});
        # 计算 threshold 函数的期望输出 y_exp
        auto y_exp = (x <= threshold) * value + (x > threshold) * x;
        # 应用 threshold 函数计算 y
        auto y = F::threshold(
            x, F::ThresholdFuncOptions(threshold, value).inplace(inplace));

        # 断言 y 的维度为 3
        ASSERT_EQ(y.ndimension(), 3);
        # 断言 y 的大小与给定的大小向量相同
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        # 断言 y 与 y_exp 全部接近
        ASSERT_TRUE(torch::allclose(y, y_exp));
        # 如果 inplace 为真，断言 x 等于 y_exp
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
  # 断言 threshold 函数应用于标量值 1 的输出定义
  ASSERT_TRUE(F::threshold(torch::tensor(1.), F::ThresholdFuncOptions(0.5, 0.5))
                  .defined());
}

# 定义一个测试函数 FunctionalTest，测试 batch_norm 函数在 1D 情况下的行为
TEST_F(FunctionalTest, BatchNorm1d) {
  # 定义特征数量、epsilon 和动量值
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  # 生成大小为 [2, 5] 的服从标准正态分布的输入张量
  auto input = torch::randn({2, 5});
  # 生成服从标准正态分布的均值张量
  auto mean = torch::randn(5);
  # 生成均匀分布的方差张量
  auto variance = torch::rand(5);
  # 生成全 1 的权重张量和全 0 的偏置张量
  auto weight = torch::ones({num_features});
  auto bias = torch::zeros({num_features});
  # 应用 batch_norm 函数计算输出
  auto output = F::batch_norm(
      input,
      mean,
      variance,
      F::BatchNormFuncOptions()
          .weight(weight)
          .bias(bias)
          .momentum(momentum)
          .eps(eps)
          .training(false));
  # 计算期望的输出张量 expected
  auto expected = (input - mean) / torch::sqrt(variance + eps);
  # 断言输出与期望输出全部接近
  ASSERT_TRUE(output.allclose(expected));
}

# 定义一个测试函数 FunctionalTest，测试 batch_norm 函数在 1D 情况下的默认选项行为
TEST_F(FunctionalTest, BatchNorm1dDefaultOptions) {
  # 生成大小为 [2, 5] 的服从标准正态分布的输入张量
  auto input = torch::randn({2, 5});
  # 生成服从标准正态分布的均值张量
  auto mean = torch::randn(5);
  # 生成均匀分布的方差张量
  auto variance = torch::rand(5);
  # 应用 batch_norm 函数计算输出
  auto output = F::batch_norm(input, mean, variance);
  # 计算期望的输出张量 expected
  auto expected = (input - mean) / torch::sqrt(variance + 1e-5);
  # 断言输出与期望输出全部接近
  ASSERT_TRUE(output.allclose(expected));
}
# 在 FunctionalTest 测试类中，测试 BatchNorm2d 函数
TEST_F(FunctionalTest, BatchNorm2d) {
  # 定义特征数量为5
  int num_features = 5;
  # 设置 epsilon 参数为 1e-05
  double eps = 1e-05;
  # 设置动量参数为 0.1
  double momentum = 0.1;

  # 生成一个随机张量作为输入，形状为 [2, num_features, 4, 4]
  auto input = torch::randn({2, num_features, 4, 4});
  # 生成一个随机张量作为均值，形状为 [num_features]
  auto mean = torch::randn(num_features);
  # 生成一个随机张量作为方差，形状为 [num_features]
  auto variance = torch::rand(num_features);
  # 生成一个全为1的张量作为权重，形状为 [num_features]
  auto weight = torch::ones({num_features});
  # 生成一个全为0的张量作为偏置，形状为 [num_features]
  auto bias = torch::zeros({num_features});
  # 调用 F::batch_norm 函数，传入输入、均值、方差以及一系列参数设定
  auto output = F::batch_norm(
      input,
      mean,
      variance,
      F::BatchNormFuncOptions()
          .weight(weight)
          .bias(bias)
          .momentum(momentum)
          .eps(eps)
          .training(false));
  # 计算预期输出，对输入进行转置操作后计算归一化值
  auto expected = torch::transpose(
      (torch::transpose(input, 1, 3) - mean) / torch::sqrt(variance + eps),
      1,
      3);
  # 断言输出张量与预期结果张量近似相等
  ASSERT_TRUE(output.allclose(expected));
}

# 在 FunctionalTest 测试类中，测试 BatchNorm2d 函数默认选项
TEST_F(FunctionalTest, BatchNorm2dDefaultOptions) {
  # 定义特征数量为5
  int num_features = 5;
  # 设置 epsilon 参数为 1e-05
  double eps = 1e-05;

  # 生成一个随机张量作为输入，形状为 [2, num_features, 4, 4]
  auto input = torch::randn({2, num_features, 4, 4});
  # 生成一个随机张量作为均值，形状为 [num_features]
  auto mean = torch::randn(num_features);
  # 生成一个随机张量作为方差，形状为 [num_features]
  auto variance = torch::rand(num_features);
  # 调用 F::batch_norm 函数，传入输入、均值、方差
  auto output = F::batch_norm(input, mean, variance);
  # 计算预期输出，对输入进行转置操作后计算归一化值
  auto expected = torch::transpose(
      (torch::transpose(input, 1, 3) - mean) / torch::sqrt(variance + eps),
      1,
      3);
  # 断言输出张量与预期结果张量近似相等
  ASSERT_TRUE(output.allclose(expected));
}

# 在 FunctionalTest 测试类中，测试 BatchNorm3d 函数
TEST_F(FunctionalTest, BatchNorm3d) {
  # 定义特征数量为5
  int num_features = 5;
  # 设置 epsilon 参数为 1e-05
  double eps = 1e-05;
  # 设置动量参数为 0.1
  double momentum = 0.1;

  # 生成一个随机张量作为输入，形状为 [2, num_features, 2, 2, 2]
  auto input = torch::randn({2, num_features, 2, 2, 2});
  # 生成一个随机张量作为均值，形状为 [num_features]
  auto mean = torch::randn(num_features);
  # 生成一个随机张量作为方差，形状为 [num_features]
  auto variance = torch::rand(num_features);
  # 生成一个全为1的张量作为权重，形状为 [num_features]
  auto weight = torch::ones({num_features});
  # 生成一个全为0的张量作为偏置，形状为 [num_features]
  auto bias = torch::zeros({num_features});
  # 调用 F::batch_norm 函数，传入输入、均值、方差以及一系列参数设定
  auto output = F::batch_norm(
      input,
      mean,
      variance,
      F::BatchNormFuncOptions()
          .weight(weight)
          .bias(bias)
          .momentum(momentum)
          .eps(eps)
          .training(false));
  # 计算预期输出，对输入进行转置操作后计算归一化值
  auto expected = torch::transpose(
      (torch::transpose(input, 1, 4) - mean) / torch::sqrt(variance + eps),
      1,
      4);
  # 断言输出张量与预期结果张量近似相等
  ASSERT_TRUE(output.allclose(expected));
}

# 在 FunctionalTest 测试类中，测试 BatchNorm3d 函数默认选项
TEST_F(FunctionalTest, BatchNorm3dDefaultOptions) {
  # 定义特征数量为5
  int num_features = 5;
  # 设置 epsilon 参数为 1e-05
  double eps = 1e-05;

  # 生成一个随机张量作为输入，形状为 [2, num_features, 2, 2, 2]
  auto input = torch::randn({2, num_features, 2, 2, 2});
  # 生成一个随机张量作为均值，形状为 [num_features]
  auto mean = torch::randn(num_features);
  # 生成一个随机张量作为方差，形状为 [num_features]
  auto variance = torch::rand(num_features);
  # 调用 F::batch_norm 函数，传入输入、均值、方差
  auto output = F::batch_norm(input, mean, variance);
  # 计算预期输出，对输入进行转置操作后计算归一化值
  auto expected = torch::transpose(
      (torch::transpose(input, 1, 4) - mean) / torch::sqrt(variance + eps),
      1,
      4);
  # 断言输出张量与预期结果张量近似相等
  ASSERT_TRUE(output.allclose(expected));
}
TEST_F(FunctionalTest, InstanceNorm1d) {
  // 定义变量：特征数为5，ε为1e-05，动量为0.1
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  // 创建一个2x5x4的张量作为输入
  auto input = torch::arange(40.).view({2, 5, 4});
  // 创建一个长度为5的张量作为均值
  auto mean = torch::arange(5.);
  // 创建一个长度为5的张量作为方差
  auto variance = torch::arange(5.);
  // 创建一个长度为5的张量作为权重
  auto weight = torch::arange((double)num_features);
  // 创建一个长度为5的张量作为偏置
  auto bias = torch::arange((double)num_features);
  
  // 调用实例归一化函数，设置相关选项
  auto output = F::instance_norm(
      input,
      F::InstanceNormFuncOptions()
          .running_mean(mean)
          .running_var(variance)
          .weight(weight)
          .bias(bias)
          .momentum(momentum)
          .eps(eps));
  
  // 创建期望的张量作为预期输出
  auto expected = torch::tensor(
      {{{0.0000, 0.0000, 0.0000, 0.0000},
        {-0.3416, 0.5528, 1.4472, 2.3416},
        {-0.6833, 1.1056, 2.8944, 4.6833},
        {-1.0249, 1.6584, 4.3416, 7.0249},
        {-1.3665, 2.2112, 5.7888, 9.3665}},
       {{0.0000, 0.0000, 0.0000, 0.0000},
        {-0.3416, 0.5528, 1.4472, 2.3416},
        {-0.6833, 1.1056, 2.8944, 4.6833},
        {-1.0249, 1.6584, 4.3416, 7.0249},
        {-1.3665, 2.2112, 5.7888, 9.3665}}});
  
  // 断言实际输出与期望输出在误差允许范围内相等
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

TEST_F(FunctionalTest, InstanceNorm1dDefaultOptions) {
  // 创建一个2x5x4的张量作为输入
  auto input = torch::arange(40.).view({2, 5, 4});
  
  // 调用实例归一化函数，使用默认选项
  auto output = F::instance_norm(input);
  
  // 创建期望的张量作为预期输出
  auto expected = torch::tensor(
      {{{-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416}},
       {{-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416},
        {-1.3416, -0.4472, 0.4472, 1.3416}}});
  
  // 断言实际输出与期望输出在误差允许范围内相等
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

TEST_F(FunctionalTest, InstanceNorm2d) {
  // 定义变量：特征数为5，ε为1e-05，动量为0.1
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  // 创建一个2x5x2x2的张量作为输入
  auto input =
      torch::arange(2. * num_features * 2 * 2).view({2, num_features, 2, 2});
  // 创建一个长度为5的张量作为均值
  auto mean = torch::arange((double)num_features);
  // 创建一个长度为5的张量作为方差
  auto variance = torch::arange((double)num_features);
  // 创建一个长度为5的张量作为权重
  auto weight = torch::arange((double)num_features);
  // 创建一个长度为5的张量作为偏置
  auto bias = torch::arange((double)num_features);
  
  // 调用实例归一化函数，设置相关选项
  auto output = F::instance_norm(
      input,
      F::InstanceNormFuncOptions()
          .running_mean(mean)
          .running_var(variance)
          .weight(weight)
          .bias(bias)
          .momentum(momentum)
          .eps(eps));
  
  // 创建期望的张量作为预期输出
  auto expected = torch::tensor(
      {{{{0.0000, 0.0000}, {0.0000, 0.0000}},
        {{-0.3416, 0.5528}, {1.4472, 2.3416}},
        {{-0.6833, 1.1056}, {2.8944, 4.6833}},
        {{-1.0249, 1.6584}, {4.3416, 7.0249}},
        {{-1.3665, 2.2112}, {5.7888, 9.3665}}},
       {{{0.0000, 0.0000}, {0.0000, 0.0000}},
        {{-0.3416, 0.5528}, {1.4472, 2.3416}},
        {{-0.6833, 1.1056}, {2.8944, 4.6833}},
        {{-1.0249, 1.6584}, {4.3416, 7.0249}},
        {{-1.3665, 2.2112}, {5.7888, 9.3665}}}});
  
  // 断言实际输出与期望输出在误差允许范围内相等
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}
# 测试用例：InstanceNorm2dDefaultOptions
TEST_F(FunctionalTest, InstanceNorm2dDefaultOptions) {
  # 设置特征数量为5
  int num_features = 5;

  # 创建输入张量，包含固定数量的特征、形状为[2, num_features, 2, 2]
  auto input =
      torch::arange(2. * num_features * 2 * 2).view({2, num_features, 2, 2});

  # 对输入张量进行 instance normalization
  auto output = F::instance_norm(input);

  # 创建期望输出张量，这里使用了预先计算的数值
  auto expected = torch::tensor(
      {{{{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}}},
       {{{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}}}});
  
  # 断言输出张量是否与期望张量在指定的误差范围内相似
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

# 测试用例：InstanceNorm3d
TEST_F(FunctionalTest, InstanceNorm3d) {
  # 设置特征数量为5
  int num_features = 5;
  # 设置 epsilon
  double eps = 1e-05;
  # 设置 momentum
  double momentum = 0.1;

  # 创建输入张量，包含固定数量的特征、形状为[2, num_features, 2, 2, 2]
  auto input = torch::arange(2. * num_features * 2 * 2 * 2)
                   .view({2, num_features, 2, 2, 2});

  # 创建 running mean 张量
  auto mean = torch::arange((double)num_features);
  # 创建 running variance 张量
  auto variance = torch::arange((double)num_features);
  # 创建 weight 张量
  auto weight = torch::arange((double)num_features);
  # 创建 bias 张量
  auto bias = torch::arange((double)num_features);

  # 对输入张量进行 instance normalization，同时传递运行时参数和额外选项
  auto output = F::instance_norm(
      input,
      F::InstanceNormFuncOptions()
          .running_mean(mean)
          .running_var(variance)
          .weight(weight)
          .bias(bias)
          .momentum(momentum)
          .eps(eps));

  # 创建期望输出张量，这里使用了预先计算的数值
  auto expected = torch::tensor(
      {{{{{0.0000, 0.0000}, {0.0000, 0.0000}},
         {{0.0000, 0.0000}, {0.0000, 0.0000}}},
        {{{-0.5275, -0.0911}, {0.3453, 0.7818}},
         {{1.2182, 1.6547}, {2.0911, 2.5275}}},
        {{{-1.0550, -0.1822}, {0.6907, 1.5636}},
         {{2.4364, 3.3093}, {4.1822, 5.0550}}},
        {{{-1.5826, -0.2733}, {1.0360, 2.3453}},
         {{3.6547, 4.9640}, {6.2733, 7.5826}}},
        {{{-2.1101, -0.3644}, {1.3814, 3.1271}},
         {{4.8729, 6.6186}, {8.3644, 10.1101}}}},
       {{{{0.0000, 0.0000}, {0.0000, 0.0000}},
         {{0.0000, 0.0000}, {0.0000, 0.0000}}},
        {{{-0.5275, -0.0911}, {0.3453, 0.7818}},
         {{1.2182, 1.6547}, {2.0911, 2.5275}}},
        {{{-1.0550, -0.1822}, {0.6907, 1.5636}},
         {{2.4364, 3.3093}, {4.1822, 5.0550}}},
        {{{-1.5826, -0.2733}, {1.0360, 2.3453}},
         {{3.6547, 4.9640}, {6.2733, 7.5826}}},
        {{{-2.1101, -0.3644}, {1.3814, 3.1271}},
         {{4.8729, 6.6186}, {8.3644, 10.1101}}}}});
  
  # 断言输出张量是否与期望张量在指定的误差范围内相似
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}
TEST_F(FunctionalTest, InstanceNorm3dDefaultOptions) {
  // 设置特征数量为5
  int num_features = 5;

  // 创建一个5维张量作为输入，形状为[2, num_features, 2, 2, 2]
  auto input = torch::arange(2. * num_features * 2 * 2 * 2)
                   .view({2, num_features, 2, 2, 2});

  // 对输入张量进行 instance normalization
  auto output = F::instance_norm(input);

  // 创建一个期望输出张量，包含预期的标准化结果
  auto expected = torch::tensor(
      {{{{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}}},
       {{{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}}}});

  // 使用指定的容差进行断言，检验输出张量是否与期望张量在误差范围内相等
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

TEST_F(FunctionalTest, Interpolate) {
  {
    // 1D 插值
    auto input = torch::ones({1, 1, 2});

    // 创建插值函数的选项，设置目标大小为[4]，插值模式为最近邻插值
    auto options = F::InterpolateFuncOptions()
                       .size(std::vector<int64_t>({4}))
                       .mode(torch::kNearest);

    // 对输入张量进行插值操作
    auto output = F::interpolate(input, options);

    // 创建一个期望输出张量，形状为[1, 1, 4]，值全为1
    auto expected = torch::ones({1, 1, 4});

    // 使用断言检验输出张量是否与期望张量相等
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 2D 插值
    for (const auto align_corners : {true, false}) {
      // 测试浮点缩放因子的上采样和下采样
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        auto input = torch::ones({1, 1, 2, 2});

        // 创建插值函数的选项，设置缩放因子为[scale_factor, scale_factor]，插值模式为双线性插值，是否对齐角点由align_corners决定
        auto options =
            F::InterpolateFuncOptions()
                .scale_factor(std::vector<double>({scale_factor, scale_factor}))
                .mode(torch::kBilinear)
                .align_corners(align_corners);

        // 对输入张量进行插值操作
        auto output = F::interpolate(input, options);

        // 计算期望输出张量的大小，根据缩放因子确定
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));

        // 创建一个期望输出张量，形状为[1, 1, expected_size, expected_size]，值全为1
        auto expected = torch::ones({1, 1, expected_size, expected_size});

        // 使用断言检验输出张量是否与期望张量相等
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    // 3D 插值
    // 对于每个 align_corners 的布尔值进行迭代：true 和 false
    for (const auto align_corners : {true, false}) {
      // 对于每个 scale_factor 的浮点数进行迭代：0.5, 1.5, 2.0
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        // 创建一个形状为 [1, 1, 2, 2, 2] 的全一张量作为输入
        auto input = torch::ones({1, 1, 2, 2, 2});
        // 设置插值函数的选项：设置 scale_factor，插值模式为 trilinear，align_corners 根据当前循环的布尔值设定
        auto options = F::InterpolateFuncOptions()
                           .scale_factor(std::vector<double>(
                               {scale_factor, scale_factor, scale_factor}))
                           .mode(torch::kTrilinear)
                           .align_corners(align_corners);
        // 执行插值操作，生成输出张量
        auto output = F::interpolate(input, options);
        // 计算期望的输出尺寸，使用输入张量最后一个维度的大小乘以 scale_factor 后向下取整
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        // 创建一个形状为 [1, 1, expected_size, expected_size, expected_size] 的全一张量作为期望输出
        auto expected =
            torch::ones({1, 1, expected_size, expected_size, expected_size});

        // 使用断言验证输出张量是否与期望的输出张量在数值上接近
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    // 使用 ASSERT_THROWS_WITH 断言捕获插值函数的异常情况：
    // 当输入张量为 1 维时，期望抛出 "Input Error: Only 3D, 4D and 5D input Tensors supported (got 1D) "
    ASSERT_THROWS_WITH(
        F::interpolate(
            torch::randn({1}),
            F::InterpolateFuncOptions().size(std::vector<int64_t>({1}))),
        "Input Error: Only 3D, 4D and 5D input Tensors supported (got 1D) ");
  }
  {
    // 创建一个形状为 [3, 2, 2] 的随机张量作为输入
    auto input = torch::randn({3, 2, 2});
    // 使用 ASSERT_THROWS_WITH 断言捕获多种异常情况：
    // 1. 当从二维输入张量中选择一个元素进行插值时，期望抛出 "Input Error: Only 3D, 4D and 5D input Tensors supported (got 2D) for the modes: nearest | linear | bilinear | bicubic | trilinear (got kNearest)"
    ASSERT_THROWS_WITH(
        F::interpolate(
            input[0],
            F::InterpolateFuncOptions().size(std::vector<int64_t>({4, 4}))),
        "Input Error: Only 3D, 4D and 5D input Tensors supported (got 2D) "
        "for the modes: nearest | linear | bilinear | bicubic | trilinear (got kNearest)");
    // 2. 当输入张量为重新形状后的六维张量时，期望抛出 "Input Error: Only 3D, 4D and 5D input Tensors supported (got 6D) for the modes: nearest | linear | bilinear | bicubic | trilinear (got kNearest)"
    ASSERT_THROWS_WITH(
        F::interpolate(
            torch::reshape(input, {1, 1, 1, 3, 2, 2}),
            F::InterpolateFuncOptions().size(
                std::vector<int64_t>({1, 1, 1, 3, 4, 4}))),
        "Input Error: Only 3D, 4D and 5D input Tensors supported (got 6D) "
        "for the modes: nearest | linear | bilinear | bicubic | trilinear (got kNearest)");
    // 3. 当插值选项既未设置 size 也未设置 scale_factor 时，期望抛出 "either size or scale_factor should be defined"
    ASSERT_THROWS_WITH(
        F::interpolate(input, F::InterpolateFuncOptions()),
        "either size or scale_factor should be defined");
    // 4. 当同时设置了 size 和 scale_factor 时，期望抛出 "only one of size or scale_factor should be defined"
    ASSERT_THROWS_WITH(
        F::interpolate(
            input,
            F::InterpolateFuncOptions()
                .size(std::vector<int64_t>({3, 4, 4}))
                .scale_factor(std::vector<double>({0.5}))),
        "only one of size or scale_factor should be defined");
    // 5. 当 scale_factor 的形状与输入张量的形状不匹配时，期望抛出 "scale_factor shape must match input shape. Input is 1D, scale_factor size is [3, 2]"
    ASSERT_THROWS_WITH(
        F::interpolate(
            input,
            F::InterpolateFuncOptions().scale_factor(
                std::vector<double>({3, 2}))),
        "scale_factor shape must match input shape. "
        "Input is 1D, scale_factor size is [3, 2]");
    // 6. 当选择了 kNearest 插值模式同时设置了 align_corners 时，期望抛出 "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
    ASSERT_THROWS_WITH(
        F::interpolate(
            input,
            F::InterpolateFuncOptions()
                .mode(torch::kNearest)
                .align_corners(true)),
        "align_corners option can only be set with the "
        "interpolating modes: linear | bilinear | bicubic | trilinear");
  }
  {
    // 创建一个形状为 [2, 3, 32, 32] 的随机张量作为输入
    auto tensor = torch::rand({2, 3, 32, 32});
    // 创建一个形状为 [8, 10] 的目标输出尺寸
    std::vector<int64_t> osize = {8, 10};
    // 使用 _upsample_nearest_exact2d 函数生成期望的输出张量
    auto expected =
        at::native::_upsample_nearest_exact2d(tensor, osize, torch::nullopt);
    // 创建插值函数的选项对象，设置输出尺寸为 osize，插值模式为最近邻插值，不对齐角点
    auto options = F::InterpolateFuncOptions()
                       .size(osize)
                       .mode(torch::kNearestExact)
                       .align_corners(false);
    // 对给定的张量进行插值操作，使用指定的选项
    auto output = F::interpolate(tensor, options);

    // 断言输出张量与期望值张量的所有元素是否近似相等
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建随机张量，形状为 [2, 3, 32, 32]
    auto tensor = torch::rand({2, 3, 32, 32});
    // 设置期望的输出尺寸为 osize = [8, 10]
    std::vector<int64_t> osize = {8, 10};
    // 使用双线性插值方法计算期望值张量
    auto expected = at::native::_upsample_bilinear2d_aa(
        tensor, osize, false, torch::nullopt);

    // 创建插值函数的选项对象，设置输出尺寸为 osize，插值模式为双线性插值，不对齐角点，开启抗锯齿
    auto options = F::InterpolateFuncOptions()
                       .size(osize)
                       .mode(torch::kBilinear)
                       .align_corners(false)
                       .antialias(true);
    // 对给定的张量进行插值操作，使用指定的选项
    auto output = F::interpolate(tensor, options);
    // 断言输出张量与期望值张量的所有元素是否近似相等
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建随机张量，形状为 [2, 3, 32, 32]
    auto tensor = torch::rand({2, 3, 32, 32});
    // 设置期望的输出尺寸为 osize = [8, 10]
    std::vector<int64_t> osize = {8, 10};
    // 使用双三次插值方法计算期望值张量
    auto expected = at::native::_upsample_bicubic2d_aa(
        tensor, osize, false, torch::nullopt);

    // 创建插值函数的选项对象，设置输出尺寸为 osize，插值模式为双三次插值，不对齐角点，开启抗锯齿
    auto options = F::InterpolateFuncOptions()
                       .size(osize)
                       .mode(torch::kBicubic)
                       .align_corners(false)
                       .antialias(true);
    // 对给定的张量进行插值操作，使用指定的选项
    auto output = F::interpolate(tensor, options);
    // 断言输出张量与期望值张量的所有元素是否近似相等
    ASSERT_TRUE(output.allclose(expected));
  }
TEST_F(FunctionalTest, Pad3) {
  {
    // 创建一个5维张量，从0到11，使用双精度浮点数类型，形状为{1, 1, 2, 2, 3}
    auto input = torch::arange(12, torch::kDouble).reshape({1, 1, 2, 2, 3});
    // 对输入张量进行填充操作，选项为{3, 3, 2, 1, 2, 2}，使用环绕（Circular）填充模式
    auto output = F::pad(
        input, F::PadFuncOptions({3, 3, 2, 1, 2, 2}).mode(torch::kCircular));
    // 预期的输出张量
    auto expected = torch::tensor(
        {{{{{0., 1., 2., 0., 1., 2., 0., 1., 2.},
            {3., 4., 5., 3., 4., 5., 3., 4., 5.},
            {0., 1., 2., 0., 1., 2., 0., 1., 2.},
            {3., 4., 5., 3., 4., 5., 3., 4., 5.},
            {0., 1., 2., 0., 1., 2., 0., 1., 2.}},

           {{6., 7., 8., 6., 7., 8., 6., 7., 8.},
            {9., 10., 11., 9., 10., 11., 9., 10., 11.},
            {6., 7., 8., 6., 7., 8., 6., 7., 8.},
            {9., 10., 11., 9., 10., 11., 9., 10., 11.},
            {6., 7., 8., 6., 7., 8., 6., 7., 8.}},

           {{0., 1., 2., 0., 1., 2., 0., 1., 2.},
            {3., 4., 5., 3., 4., 5., 3., 4., 5.},
            {0., 1., 2., 0., 1., 2., 0., 1., 2.},
            {3., 4., 5., 3., 4., 5., 3., 4., 5.},
            {0., 1., 2., 0., 1., 2., 0., 1., 2.}},

           {{6., 7., 8., 6., 7., 8., 6., 7., 8.},
            {9., 10., 11., 9., 10., 11., 9., 10., 11.},
            {6., 7., 8., 6., 7., 8., 6., 7., 8.},
            {9., 10., 11., 9., 10., 11., 9., 10., 11.},
            {6., 7., 8., 6., 7., 8., 6., 7., 8.}},

           {{0., 1., 2., 0., 1., 2., 0., 1., 2.},
            {3., 4., 5., 3., 4., 5., 3., 4., 5.},
            {0., 1., 2., 0., 1., 2., 0., 1., 2.},
            {3., 4., 5., 3., 4., 5., 3., 4., 5.},
            {0., 1., 2., 0., 1., 2., 0., 1., 2.}},

           {{6., 7., 8., 6., 7., 8., 6., 7., 8.},
            {9., 10., 11., 9., 10., 11., 9., 10., 11.},
            {6., 7., 8., 6., 7., 8., 6., 7., 8.},
            {9., 10., 11., 9., 10., 11., 9., 10., 11.},
            {6., 7., 8., 6., 7., 8., 6., 7., 8.}}}}},
        torch::kDouble);
    // 断言输出张量的形状是否为{1, 1, 7, 9}
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 7, 9}));
    // 断言输出张量与预期张量在1e-04精度范围内是否相似
    ASSERT_TRUE(output.allclose(expected, 1e-04));
  }
}
    // 断言 output 的尺寸是否与给定的向量 {1, 1, 6, 5, 9} 相等
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 6, 5, 9}));
    // 断言 output 是否与期望值 expected 在给定的误差范围内全部相似
    ASSERT_TRUE(output.allclose(expected, 1e-04));
  }
}
TEST_F(FunctionalTest, Pad6) {
  {
    // 创建一个5维张量，值为0到17，数据类型为双精度
    auto input = torch::arange(18, torch::kDouble).reshape({1, 1, 3, 2, 3});
    // 对输入张量进行填充，使用反射模式，填充参数为{0, 2, 1, 0, 1, 2}
    auto output = F::pad(
        input, F::PadFuncOptions({0, 2, 1, 0, 1, 2}).mode(torch::kReflect));


这段代码的作用是使用 PyTorch 的函数库进行张量操作，具体是对一个5维张量进行填充操作，填充模式为反射模式，填充参数为{0, 2, 1, 0, 1, 2}，最终得到填充后的输出张量 `output`。

注意：这里的注释只针对代码的解释，不涉及测试框架的具体用途或断言的含义。
    // 创建一个张量 `expected`，其内容为一个多维数组，使用了 torch 库的 tensor 函数
    auto expected = torch::tensor(
        {{{{{9., 10., 11., 10., 9.},    // 第一组子数组
            {6., 7., 8., 7., 6.},       // 第二组子数组
            {9., 10., 11., 10., 9.}}},  // 第三组子数组

           {{3., 4., 5., 4., 3.},       // 第四组子数组
            {0., 1., 2., 1., 0.},       // 第五组子数组
            {3., 4., 5., 4., 3.}}},     // 第六组子数组

           {{9., 10., 11., 10., 9.},    // 第七组子数组
            {6., 7., 8., 7., 6.},       // 第八组子数组
            {9., 10., 11., 10., 9.}}},  // 第九组子数组

           {{15., 16., 17., 16., 15.},  // 第十组子数组
            {12., 13., 14., 13., 12.},  // 第十一组子数组
            {15., 16., 17., 16., 15.}}},// 第十二组子数组

           {{9., 10., 11., 10., 9.},    // 第十三组子数组
            {6., 7., 8., 7., 6.},       // 第十四组子数组
            {9., 10., 11., 10., 9.}}},  // 第十五组子数组

           {{3., 4., 5., 4., 3.},       // 第十六组子数组
            {0., 1., 2., 1., 0.},       // 第十七组子数组
            {3., 4., 5., 4., 3.}}}}},   // 第十八组子数组

        torch::kDouble);               // 使用双精度浮点数类型的张量
    // 使用 ASSERT_EQ 断言检查 output 张量的维度是否与指定的一维向量 {1, 1, 6, 3, 5} 相同
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 6, 3, 5}));
    // 使用 ASSERT_TRUE 断言检查 output 张量与 expected 张量在指定精度（1e-04）下是否全部相近
    ASSERT_TRUE(output.allclose(expected, 1e-04));
}
TEST_F(FunctionalTest, CTCLoss) {
  { // test CTCLoss typechecks

    // 定义目标序列的长度
    const auto target_lengths = torch::tensor({30, 25, 20});
    // 定义输入序列的长度
    const auto input_lengths = torch::tensor({50, 50, 50});
    // 生成随机整数目标张量
    const auto targets = torch::randint(1, 15, {target_lengths.sum().item<int>()}, torch::kInt);
    // 生成随机对数概率张量并进行对数softmax
    const auto log_probs = torch::randn({50, 3, 15}, torch::kFloat).log_softmax(2);

    // 将输入长度转换为浮点数类型
    const auto _input_lengths = input_lengths.to(torch::kFloat);
    // 断言CTCLoss函数调用时抛出异常，检查输入长度必须为整数
    ASSERT_THROWS_WITH(
        F::ctc_loss(log_probs, targets, _input_lengths, target_lengths),
        "input_lengths must be integral");

    // 将目标长度转换为浮点数类型
    const auto target_lengths_ = target_lengths.to(torch::kFloat);
    // 断言CTCLoss函数调用时抛出异常，检查目标长度必须为整数
    ASSERT_THROWS_WITH(
        F::ctc_loss(log_probs, targets, input_lengths, target_lengths_),
        "target_lengths must be integral");
  }
  { // test CTCLoss length checks

    // 定义目标序列的长度
    const auto target_lengths = torch::tensor({30, 25, 20});
    // 定义输入序列的长度
    const auto input_lengths = torch::tensor({50, 50, 50});
    // 生成随机整数目标张量，但形状不正确（应为3行29列）
    const auto targets = torch::randint(1, 15, {3, 29}, torch::kInt);
    // 生成随机对数概率张量并进行对数softmax
    const auto log_probs = torch::randn({50, 3, 15}, torch::kFloat).log_softmax(2);

    // 断言CTCLoss函数调用时抛出异常，检查目标张量维度是否满足要求
    ASSERT_THROWS_WITH(
        F::ctc_loss(log_probs, targets, input_lengths, target_lengths),
        "Expected tensor to have size at least 30 at dimension 1");
  }
  { // test CTCLoss empty target

    // 定义空的目标序列长度
    const auto target_lengths = torch::tensor({0, 0, 0});
    // 定义输入序列的长度
    const auto input_lengths = torch::tensor({50, 50, 50});
    // 生成空的目标张量（长度为0）
    const auto targets = torch::randint(1, 15, at::IntArrayRef({0}), torch::kLong);
    // 生成随机对数概率张量并进行对数softmax
    const auto log_probs = torch::randn({50, 3, 15}, torch::kDouble).log_softmax(2);

    // 计算CTC损失，设置选项为不进行缩减
    const auto loss = F::ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        F::CTCLossFuncOptions().reduction(torch::kNone));
    // 断言损失张量中所有值都大于等于0
    ASSERT_TRUE(loss.ge(0).all().item<bool>());
    // 断言损失张量与-log_probs在第一维度上的部分切片相等
    ASSERT_TRUE(torch::allclose(
        -log_probs.sum(0).slice(1, 0, 1).view_as(loss), loss));
  }
}
    {
      // 定义目标序列的长度，这里包括三个值：0, 9, 0
      const auto target_lengths = torch::tensor({0, 9, 0});
      // 定义输入序列的长度，每个输入序列长度均为50
      const auto input_lengths = torch::tensor({50, 50, 50});
      // 生成一个包含9个随机整数（范围1到15）的张量，作为目标序列
      const auto targets = torch::randint(1, 15, {9}, torch::kLong);
      // 生成一个形状为{50, 3, 15}的随机张量，然后对最后一个维度应用log_softmax
      const auto log_probs =
          torch::randn({50, 3, 15}, torch::kDouble).log_softmax(2);
      // 计算 CTC 损失函数，传入对数概率、目标序列、输入序列长度、目标序列长度和损失函数选项
      const auto loss = F::ctc_loss(
          log_probs,
          targets,
          input_lengths,
          target_lengths,
          F::CTCLossFuncOptions().reduction(torch::kNone));
      // 断言损失值大于等于0的所有元素为真
      ASSERT_TRUE(loss.ge(0).all().item<bool>());
      // 断言损失张量的部分值与-log_probs在特定维度上的和相似
      ASSERT_TRUE(torch::allclose(
          -log_probs.sum(0)
               .index_select(0, torch::tensor({0, 2}, torch::kLong))
               .slice(1, 0, 1)
               .view({2}),
          loss.index_select(0, torch::tensor({0, 2}, torch::kLong))));
    }
}

# 定义一个测试用例 FunctionalTest，用于测试 PoissonNLLLoss 函数
TEST_F(FunctionalTest, PoissonNLLLoss) {
  # 创建输入张量 input，包含数据 0.5, 1.5, 2.5
  const auto input = torch::tensor({0.5, 1.5, 2.5});
  # 创建目标张量 target，包含数据 1., 2., 3.
  const auto target = torch::tensor({1., 2., 3.});
  # 计算逐元素损失 component_wise_loss
  const auto component_wise_loss = torch::exp(input) - target * input;
  # 断言：component_wise_loss 的均值与 PoissonNLLLoss 函数的返回结果相等
  ASSERT_TRUE(torch::allclose(
      torch::mean(component_wise_loss), F::poisson_nll_loss(input, target)));
  # 断言：component_wise_loss 与使用选项 reduction=torch::kNone 的 PoissonNLLLoss 函数结果相等
  ASSERT_TRUE(torch::allclose(
      component_wise_loss,
      F::poisson_nll_loss(
          input,
          target,
          F::PoissonNLLLossFuncOptions().reduction(torch::kNone))));
  # 断言：component_wise_loss 的总和与使用选项 reduction=torch::kSum 的 PoissonNLLLoss 函数结果相等
  ASSERT_TRUE(torch::allclose(
      torch::sum(component_wise_loss),
      F::poisson_nll_loss(
          input,
          target,
          F::PoissonNLLLossFuncOptions().reduction(torch::kSum))));
  # 断言：component_wise_loss 的均值与使用选项 reduction=torch::kMean 的 PoissonNLLLoss 函数结果相等
  ASSERT_TRUE(torch::allclose(
      torch::mean(component_wise_loss),
      F::poisson_nll_loss(
          input,
          target,
          F::PoissonNLLLossFuncOptions().reduction(torch::kMean))));
}

# 定义一个测试用例 FunctionalTest，用于测试 MarginRankingLoss 函数
TEST_F(FunctionalTest, MarginRankingLoss) {
  {
    # 创建随机输入张量 input1 和 input2，大小为 15，乘以 10 的标准正态分布
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    # 创建目标张量 target，大小为 15，符号函数的结果
    const auto target = torch::randn(15).sign();
    # 断言：MarginRankingLoss 函数的结果与 (-target * (input1 - input2)).clamp(0).mean() 相等
    ASSERT_TRUE(torch::allclose(
        F::margin_ranking_loss(input1, input2, target),
        (-target * (input1 - input2)).clamp(0).mean()));
  }
  {
    # 创建随机输入张量 input1 和 input2，大小为 15，乘以 10 的标准正态分布
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    # 创建目标张量 target，大小为 15，符号函数的结果
    const auto target = torch::randn(15).sign();
    const auto margin = 0.5;
    # 断言：MarginRankingLoss 函数的结果与 (-target * (input1 - input2) + margin).clamp(0).sum() 相等
    ASSERT_TRUE(torch::allclose(
        F::margin_ranking_loss(
            input1,
            input2,
            target,
            F::MarginRankingLossFuncOptions().margin(0.5).reduction(
                torch::kSum)),
        (-target * (input1 - input2) + margin).clamp(0).sum()));
  }
  {
    # 创建随机输入张量 input1 和 input2，大小为 15，乘以 10 的标准正态分布
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    # 创建目标张量 target，大小为 15，符号函数的结果
    const auto target = torch::randn(15).sign();
    const auto margin = 0.5;
    # 断言：MarginRankingLoss 函数的结果与 (-target * (input1 - input2) + margin).clamp(0).mean() 相等
    ASSERT_TRUE(torch::allclose(
        F::margin_ranking_loss(
            input1,
            input2,
            target,
            F::MarginRankingLossFuncOptions().margin(0.5).reduction(
                torch::kMean)),
        (-target * (input1 - input2) + margin).clamp(0).mean()));
  }
}

# 定义一个测试用例 FunctionalTest，用于测试 ConvTranspose1d 函数
TEST_F(FunctionalTest, ConvTranspose1d) {
  # 创建输入张量 x，形状为 {2, 2, 5}，数据为 0 到 19
  auto x = torch::arange(20.).view({2, 2, 5});
  # 创建卷积核张量 weight，形状为 {2, 3, 3}，数据为 0 到 17
  auto weight = torch::arange(18.).view({2, 3, 3});
  # 使用选项 stride=1 对 x 和 weight 进行一维转置卷积
  auto y =
      F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));
  # 创建预期结果张量 expected，与 y 的结果对比
  auto expected = torch::tensor(
      {{{45., 104., 179., 212., 245., 188., 107.},
        {60., 140., 242., 293., 344., 260., 146.},
        {75., 176., 305., 374., 443., 332., 185.}},
       {{135., 304., 509., 542., 575., 428., 237.},
        {210., 460., 752., 803., 854., 620., 336.},
        {285., 616., 995., 1064., 1133., 812., 435.}}});
  # 断言：y 与 expected 相等
  ASSERT_TRUE(torch::allclose(y, expected));

  # 不使用任何选项进行一维转置卷积
  auto y_no_options = F::conv_transpose1d(x, weight);
  # 断言：y_no_options 与 expected 相等
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}
// 定义测试函数 FunctionalTest.ConvTranspose2dEven，用于测试 ConvTranspose2d 的功能
TEST_F(FunctionalTest, ConvTranspose2dEven) {
  // 创建一个张量 x，包含值从 0 到 49，形状为 {1, 2, 5, 5}
  auto x = torch::arange(50.).view({1, 2, 5, 5});
  // 创建一个张量 weight，包含值从 0 到 53，形状为 {2, 3, 3, 3}
  auto weight = torch::arange(54.).view({2, 3, 3, 3});
  // 使用 ConvTranspose2d 函数计算 x 和 weight 的转置卷积，指定步幅为 1
  auto y =
      F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
  // 创建一个期望的张量 expected，包含预计的输出值
  auto expected = torch::tensor(
      {{{{675., 1402., 2183., 2270., 2357., 1634., 849.},
         {1560., 3240., 5044., 5236., 5428., 3760., 1952.},
         {2685., 5574., 8673., 8988., 9303., 6438., 3339.},
         {3180., 6594., 10248., 10563., 10878., 7518., 3894.},
         {3675., 7614., 11823., 12138., 12453., 8598., 4449.},
         {2820., 5832., 9040., 9268., 9496., 6544., 3380.},
         {1605., 3314., 5129., 5252., 5375., 3698., 1907.}},
        {{900., 1870., 2912., 3053., 3194., 2210., 1146.},
         {2100., 4356., 6772., 7072., 7372., 5092., 2636.},
         {3630., 7518., 11670., 12147., 12624., 8706., 4500.},
         {4395., 9078., 14055., 14532., 15009., 10326., 5325.},
         {5160., 10638., 16440., 16917., 17394., 11946., 6150.},
         {3900., 8028., 12388., 12724., 13060., 8956., 4604.},
         {2190., 4502., 6938., 7115., 7292., 4994., 2564.}},
        {{1125., 2338., 3641., 3836., 4031., 2786., 1443.},
         {2640., 5472., 8500., 8908., 9316., 6424., 3320.},
         {4575., 9462., 14667., 15306., 15945., 10974., 5661.},
         {5610., 11562., 17862., 18501., 19140., 13134., 6756.},
         {6645., 13662., 21057., 21696., 22335., 15294., 7851.},
         {4980., 10224., 15736., 16180., 16624., 11368., 5828.},
         {2775., 5690., 8747., 8978., 9209., 6290., 3221.}}}});
  // 断言 y 与 expected 的张量值在误差范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));

  // 使用 ConvTranspose2d 函数计算 x 和 weight 的转置卷积，未指定任何选项
  auto y_no_options = F::conv_transpose2d(x, weight);
  // 断言 y_no_options 与 expected 的张量值在误差范围内相等
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}
TEST_F(FunctionalTest, ConvTranspose2dUneven) {
  // 创建一个形状为 (1, 2, 5, 4) 的张量 x，包含从 0 到 39 的连续数字
  auto x = torch::arange(40.).view({1, 2, 5, 4});
  // 创建一个形状为 (2, 3, 3, 2) 的张量 weight，包含从 0 到 35 的连续数字
  auto weight = torch::arange(36.).view({2, 3, 3, 2});
  // 使用 stride 为 1 创建一个二维转置卷积，应用于张量 x 和 weight
  auto y =
      F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
  // 创建一个预期的张量 expected，用于断言 y 的正确性
  auto expected = torch::tensor(
      {{{{360., 758., 796., 834., 440.},
         {832., 1752., 1836., 1920., 1012.},
         {1432., 3014., 3152., 3290., 1732.},
         {1696., 3566., 3704., 3842., 2020.},
         {1960., 4118., 4256., 4394., 2308.},
         {1504., 3152., 3252., 3352., 1756.},
         {856., 1790., 1844., 1898., 992.}},
        {{480., 1010., 1072., 1134., 596.},
         {1120., 2352., 2484., 2616., 1372.},
         {1936., 4058., 4268., 4478., 2344.},
         {2344., 4898., 5108., 5318., 2776.},
         {2752., 5738., 5948., 6158., 3208.},
         {2080., 4328., 4476., 4624., 2404.},
         {1168., 2426., 2504., 2582., 1340.}},
        {{600., 1262., 1348., 1434., 752.},
         {1408., 2952., 3132., 3312., 1732.},
         {2440., 5102., 5384., 5666., 2956.},
         {2992., 6230., 6512., 6794., 3532.},
         {3544., 7358., 7640., 7922., 4108.},
         {2656., 5504., 5700., 5896., 3052.},
         {1480., 3062., 3164., 3266., 1688.}}}});
  // 断言 y 与 expected 的值在一个小范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));

  // 创建一个未指定选项的二维转置卷积，应用于张量 x 和 weight
  auto y_no_options = F::conv_transpose2d(x, weight);
  // 断言未指定选项的结果与 expected 的值在一个小范围内相等
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

TEST_F(FunctionalTest, ConvTranspose3d) {
  // 创建一个形状为 (1, 2, 2, 2, 2) 的五维张量 x，包含从 0 到 15 的连续数字
  auto x = torch::arange(16.).view({1, 2, 2, 2, 2});
  // 创建一个形状为 (2, 2, 2, 2, 2) 的五维张量 weight，包含从 0 到 31 的连续数字
  auto weight = torch::arange(32.).view({2, 2, 2, 2, 2});
  // 使用 stride 为 1 创建一个三维转置卷积，应用于张量 x 和 weight
  auto y =
      F::conv_transpose3d(x, weight, F::ConvTranspose3dFuncOptions().stride(1));
  // 创建一个预期的张量 expected，用于断言 y 的正确性
  auto expected = torch::tensor(
      {{{{{128., 280., 154.}, {304., 664., 364.}, {184., 400., 218.}},
         {{352., 768., 420.}, {832., 1808., 984.}, {496., 1072., 580.}},
         {{256., 552., 298.}, {592., 1272., 684.}, {344., 736., 394.}}},
        {{{192., 424., 234.}, {464., 1016., 556.}, {280., 608., 330.}},
         {{544., 1184., 644.}, {1280., 2768., 1496.}, {752., 1616., 868.}},
         {{384., 824., 442.}, {880., 1880., 1004.}, {504., 1072., 570.}}}}});
  // 断言 y 与 expected 的值在一个小范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));

  // 创建一个未指定选项的三维转置卷积，应用于张量 x 和 weight
  auto y_no_options = F::conv_transpose3d(x, weight);
  // 断言未指定选项的结果与 expected 的值在一个小范围内相等
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

TEST_F(FunctionalTest, AlphaDropout) {
  // 创建一个包含 5000 个随机数的张量 input
  auto input = torch::randn(5000);
  // 计算 input 的平均值和标准差
  auto input_mean = input.mean();
  auto input_std = input.std();

  // 对于不同的丢弃率 rate 和是否原地操作 inplace 进行循环测试
  for (const auto rate : {0.2, 0.5, 0.8}) {
    for (const auto inplace : {false, true}) {
      // 克隆 input 得到 input_
      auto input_ = input.clone();
      // 应用 Alpha Dropout 到 input_，根据给定的丢弃率和选项
      auto output = F::alpha_dropout(
          input_,
          F::AlphaDropoutFuncOptions().p(rate).training(false).inplace(
              inplace));
      // 断言 output 的平均值和标准差与 input 的几乎相等
      ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
      ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
      // 如果 inplace 为 true，则断言 input_ 和 output 是相等的
      if (inplace) {
        ASSERT_TRUE(torch::allclose(input_, output));
      }
    }
  }
}
    }
  }
  # 使用 alpha_dropout 函数对输入进行 Alpha Dropout 处理，丢弃率为 0.5，不进行训练模式和推理模式的切换
  auto output = F::detail::alpha_dropout(input, 0.5, false, false);
  # 断言：验证处理后的输出均值与输入均值的接近程度在 0.1 的误差范围内
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
  # 断言：验证处理后的输出标准差与输入标准差的接近程度在 0.1 的误差范围内
  ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
}

TEST_F(FunctionalTest, FeatureAlphaDropout) {
  // 生成一个包含5000个随机数的张量作为输入
  auto input = torch::randn(5000);
  // 计算输入张量的均值和标准差
  auto input_mean = input.mean();
  auto input_std = input.std();

  // 针对不同的dropout率和inplace选项进行循环测试
  for (const auto rate : {0.2, 0.5, 0.8}) {
    for (const auto inplace : {false, true}) {
      // 克隆输入张量，以便每次迭代都从原始输入开始
      auto input_ = input.clone();
      // 应用FeatureAlphaDropout函数，设置dropout率、training模式为false、inplace选项
      auto output = F::feature_alpha_dropout(
          input_,
          F::FeatureAlphaDropoutFuncOptions().p(rate).training(false).inplace(
              inplace));
      // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.1
      ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
      // 断言输出张量的标准差接近输入张量的标准差，允许误差范围为0.1
      ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
      // 如果使用了inplace选项，则断言输入张量和输出张量相等
      if (inplace) {
        ASSERT_TRUE(torch::allclose(input_, output));
      }
    }
  }
  // 再次应用FeatureAlphaDropout函数，但没有额外选项
  auto output = F::feature_alpha_dropout(input);
  // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.1
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
  // 断言输出张量的标准差接近输入张量的标准差，允许误差范围为0.1
  ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
}

TEST_F(FunctionalTest, Dropout) {
  // 生成一个包含5000个随机数的张量作为输入
  auto input = torch::randn(5000);
  // 计算输入张量的均值和标准差
  auto input_mean = input.mean();
  auto input_std = input.std();

  // 针对不同的dropout率进行循环测试
  for (const auto rate : {0.2, 0.5, 0.8}) {
    // 应用Dropout函数，设置dropout率
    auto output = F::dropout(input, F::DropoutFuncOptions().p(rate));
    // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.01到0.05
    ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
    // 断言输出张量的标准差不小于输入张量的标准差
    ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  }
  // 再次应用Dropout函数，但没有额外选项
  auto output = F::dropout(input);
  // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.01到0.05
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  // 断言输出张量的标准差不小于输入张量的标准差
  ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  // 断言对标量输入应用Dropout函数后得到的输出张量是定义好的
  ASSERT_TRUE(F::dropout(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, Dropout2d) {
  // 生成一个形状为[2, 2, 50, 100]的随机张量作为输入
  auto input = torch::randn({2, 2, 50, 100});
  // 计算输入张量的均值和标准差
  auto input_mean = input.mean();
  auto input_std = input.std();

  // 针对不同的dropout率进行循环测试
  for (const auto rate : {0.2, 0.5, 0.8}) {
    // 应用Dropout2d函数，设置dropout率
    auto output = F::dropout2d(input, F::Dropout2dFuncOptions().p(rate));
    // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.01到0.05
    ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  }
  // 再次应用Dropout2d函数，但没有额外选项
  auto output = F::dropout2d(input);
  // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.01到0.05
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  // 断言对形状为[2, 50, 100]的随机张量应用Dropout2d函数后得到的输出张量是定义好的
  ASSERT_TRUE(F::dropout2d(torch::randn({2, 50, 100})).defined());
}

TEST_F(FunctionalTest, Dropout3d) {
  // 生成一个形状为[2, 2, 50, 10, 10]的随机张量作为输入
  auto input = torch::randn({2, 2, 50, 10, 10});
  // 计算输入张量的均值和标准差
  auto input_mean = input.mean();
  auto input_std = input.std();

  // 针对不同的dropout率进行循环测试
  for (const auto rate : {0.2, 0.5, 0.8}) {
    // 应用Dropout3d函数，设置dropout率
    auto output = F::dropout3d(input, F::Dropout3dFuncOptions().p(rate));
    // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.01到0.05
    ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  }
  // 再次应用Dropout3d函数，但没有额外选项
  auto output = F::dropout3d(input);
  // 断言输出张量的均值接近输入张量的均值，允许误差范围为0.01到0.05
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  // 断言对形状为[2, 50, 10, 10]的随机张量应用Dropout3d函数后得到的输出张量是定义好的
  ASSERT_TRUE(F::dropout3d(torch::randn({2, 50, 10, 10})).defined());
}

template <c10::ScalarType S, typename T>
void test_isfinite(const at::Device& device) {
  // 定义一组测试值，包括最小、最大、零、常数等，用于检查张量的有限性
  const std::vector<T> values = {
      std::numeric_limits<T>::lowest(),
      0,
      1,
      42,
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max()};
  // 遍历每个测试值
  for (const auto value : values) {
    // 创建一个形状为[3, 3]、值为当前测试值的张量，指定数据类型和设备
    const auto x = torch::full(
        {3, 3}, value, torch::TensorOptions().dtype(S).device(device));
    // ...
    # 检查张量 x 中的所有元素是否有非有限值，即 NaN 或 ±Inf
    ASSERT_TRUE(torch::isfinite(x).all().template item<bool>());
  }
  # 如果数据类型 T 支持无穷大（infinity）
  if (std::numeric_limits<T>::has_infinity) {
    # 获取类型 T 的无穷大值
    const auto inf = std::numeric_limits<T>::infinity();
    # 创建张量 x，包含 -inf, lowest, 0, 1, 42, min, max, inf
    const auto x = torch::tensor(
        {-inf,
         std::numeric_limits<T>::lowest(),
         static_cast<T>(0),
         static_cast<T>(1),
         static_cast<T>(42),
         std::numeric_limits<T>::min(),
         std::numeric_limits<T>::max(),
         inf},
        torch::TensorOptions().dtype(S).device(device));
    # 使用 torch::allclose 检查张量 x 的有限值情况是否与预期一致
    ASSERT_TRUE(torch::allclose(
        # 将 torch::isfinite(x) 转换为 torch::kInt 类型进行比较
        torch::isfinite(x).toType(torch::kInt),
        # 创建预期结果张量，表示 x 中每个元素是否有限
        torch::tensor(
            {false, true, true, true, true, true, true, false},
            torch::TensorOptions().device(device))
            .toType(torch::kInt)));
  }
  # 如果数据类型 T 支持静默 NaN
  if (std::numeric_limits<T>::has_quiet_NaN) {
    # 创建张量 x，包含 quiet NaN
    const auto x = torch::tensor(
        {std::numeric_limits<T>::quiet_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    # 断言张量 x 中不是所有元素都是有限的
    ASSERT_FALSE(torch::isfinite(x).all().template item<bool>());
  }
  # 如果数据类型 T 支持信号 NaN
  if (std::numeric_limits<T>::has_signaling_NaN) {
    # 创建张量 x，包含 signaling NaN
    const auto x = torch::tensor(
        {std::numeric_limits<T>::signaling_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    # 断言张量 x 中不是所有元素都是有限的
    ASSERT_FALSE(torch::isfinite(x).all().template item<bool>());
  }
}

TEST_F(FunctionalTest, isfinite) {
  // 定义使用的设备为 CPU
  const at::Device device("cpu");
  // 调用模板函数 test_isfinite，使用 torch::kUInt8 类型和 uint8_t 类型参数，传入设备对象
  test_isfinite<torch::kUInt8, uint8_t>(device);
  // 同上，使用 torch::kInt8 类型和 int8_t 类型参数
  test_isfinite<torch::kInt8, int8_t>(device);
  // 同上，使用 torch::kInt16 类型和 int16_t 类型参数
  test_isfinite<torch::kInt16, int16_t>(device);
  // 同上，使用 torch::kInt32 类型和 int32_t 类型参数
  test_isfinite<torch::kInt32, int32_t>(device);
  // 同上，使用 torch::kInt64 类型和 int64_t 类型参数
  test_isfinite<torch::kInt64, int64_t>(device);
  // 同上，使用 torch::kFloat32 类型和 float 类型参数
  test_isfinite<torch::kFloat32, float>(device);
  // 同上，使用 torch::kFloat64 类型和 double 类型参数
  test_isfinite<torch::kFloat64, double>(device);
}

TEST_F(FunctionalTest, isfinite_CUDA) {
  // 定义使用的设备为 CUDA
  const at::Device device("cuda");
  // 同 isfinite 测试函数，使用不同的数据类型参数和设备对象进行测试
  test_isfinite<torch::kUInt8, uint8_t>(device);
  test_isfinite<torch::kInt8, int8_t>(device);
  test_isfinite<torch::kInt16, int16_t>(device);
  test_isfinite<torch::kInt32, int32_t>(device);
  test_isfinite<torch::kInt64, int64_t>(device);
  test_isfinite<torch::kFloat32, float>(device);
  test_isfinite<torch::kFloat64, double>(device);
  // 对于 CUDA 设备，额外测试 torch::kFloat16 类型和 c10::Half 类型参数
  test_isfinite<torch::kFloat16, c10::Half>(device);
}

template <c10::ScalarType S, typename T>
void test_isinf(const at::Device& device) {
  // 定义一组测试数据，包括最小值、零、一般值等，作为输入
  const std::vector<T> values = {
      std::numeric_limits<T>::lowest(),
      0,
      1,
      42,
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max()};
  // 遍历测试数据，每个数据创建一个指定类型和设备的张量，并检查张量中是否存在无穷大
  for (const auto value : values) {
    const auto x = torch::full(
        {3, 3}, value, torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isinf(x).all().template item<bool>());
  }
  // 如果数据类型支持正无穷大
  if (std::numeric_limits<T>::has_infinity) {
    // 创建一个包含正负无穷大、最小值、零、一般值等数据的张量，并与预期结果比较
    const auto inf = std::numeric_limits<T>::infinity();
    const auto x = torch::tensor(
        {-inf,
         std::numeric_limits<T>::lowest(),
         static_cast<T>(0),
         static_cast<T>(1),
         static_cast<T>(42),
         std::numeric_limits<T>::min(),
         std::numeric_limits<T>::max(),
         inf},
        torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::allclose(
        // 使用 torch::allclose 比较张量中是否存在无穷大
        torch::isinf(x).toType(torch::kInt),
        torch::tensor(
            {true, false, false, false, false, false, false, true},
            torch::TensorOptions().device(device))
            .toType(torch::kInt)));
  }
  // 如果数据类型支持静默 NaN
  if (std::numeric_limits<T>::has_quiet_NaN) {
    // 创建包含静默 NaN 的张量，并检查其中是否存在无穷大
    const auto x = torch::tensor(
        {std::numeric_limits<T>::quiet_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isinf(x).all().template item<bool>());
  }
  // 如果数据类型支持信号 NaN
  if (std::numeric_limits<T>::has_signaling_NaN) {
    // 创建包含信号 NaN 的张量，并检查其中是否存在无穷大
    const auto x = torch::tensor(
        {std::numeric_limits<T>::signaling_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isinf(x).all().template item<bool>());
  }
}
TEST_F(FunctionalTest, isinf) {
  // 定义使用 CPU 设备
  const at::Device device("cpu");
  // 调用模板函数 test_isinf，测试数据类型为 torch::kUInt8 和 uint8_t，使用 CPU 设备
  test_isinf<torch::kUInt8, uint8_t>(device);
  // 同上，测试数据类型为 torch::kInt8 和 int8_t
  test_isinf<torch::kInt8, int8_t>(device);
  // 同上，测试数据类型为 torch::kInt16 和 int16_t
  test_isinf<torch::kInt16, int16_t>(device);
  // 同上，测试数据类型为 torch::kInt32 和 int32_t
  test_isinf<torch::kInt32, int32_t>(device);
  // 同上，测试数据类型为 torch::kInt64 和 int64_t
  test_isinf<torch::kInt64, int64_t>(device);
  // 同上，测试数据类型为 torch::kFloat32 和 float
  test_isinf<torch::kFloat32, float>(device);
  // 同上，测试数据类型为 torch::kFloat64 和 double
  test_isinf<torch::kFloat64, double>(device);
}

TEST_F(FunctionalTest, isinf_CUDA) {
  // 定义使用 CUDA 设备
  const at::Device device("cuda");
  // 调用模板函数 test_isinf，测试数据类型为 torch::kUInt8 和 uint8_t，使用 CUDA 设备
  test_isinf<torch::kUInt8, uint8_t>(device);
  // 同上，测试数据类型为 torch::kInt8 和 int8_t
  test_isinf<torch::kInt8, int8_t>(device);
  // 同上，测试数据类型为 torch::kInt16 和 int16_t
  test_isinf<torch::kInt16, int16_t>(device);
  // 同上，测试数据类型为 torch::kInt32 和 int32_t
  test_isinf<torch::kInt32, int32_t>(device);
  // 同上，测试数据类型为 torch::kInt64 和 int64_t
  test_isinf<torch::kInt64, int64_t>(device);
  // 同上，测试数据类型为 torch::kFloat32 和 float
  test_isinf<torch::kFloat32, float>(device);
  // 同上，测试数据类型为 torch::kFloat64 和 double
  test_isinf<torch::kFloat64, double>(device);
  // 同上，测试数据类型为 torch::kFloat16 和 c10::Half
  test_isinf<torch::kFloat16, c10::Half>(device);
}

template <c10::ScalarType S, typename T>
void test_allclose(const at::Device& device) {
  // 定义一组测试值，包括 T 类型的最小值、零、一、42、最小负值、最大值
  const std::vector<T> values = {
      std::numeric_limits<T>::lowest(),
      0,
      1,
      42,
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max()};
  // 遍历 values 中的每个值进行测试
  for (const auto value : values) {
    // 创建大小为 1 的张量 x，值为 value，数据类型为 S，设备为 device
    const auto x =
        torch::full({1}, value, torch::TensorOptions().dtype(S).device(device));
    // 创建大小为 1 的张量 y，值为 value，数据类型为 S，设备为 device
    const auto y =
        torch::full({1}, value, torch::TensorOptions().dtype(S).device(device));
    // 断言 x 和自身的所有值都近似相等
    ASSERT_TRUE(torch::allclose(x, x));
    // 断言 x 和 y 的所有值都近似相等
    ASSERT_TRUE(torch::allclose(x, y));
    // 断言 y 和 x 的所有值都近似相等
    ASSERT_TRUE(torch::allclose(y, x));
    // 断言 1.1 * x + 0.1 和 1.0 * x 的所有值都不近似相等
    ASSERT_FALSE(torch::allclose(1.1 * x + 0.1, 1.0 * x));
    // 断言 0.99 * x + 0.1 和 1.0 * x 的所有值都近似相等，相对误差为 1.1%，绝对误差为 0.1
    ASSERT_TRUE(torch::allclose(0.99 * x + 0.1, 1.0 * x, 1.1, 0.1));
  }
  // 如果 T 类型支持正无穷大
  if (std::numeric_limits<T>::has_infinity) {
    // 创建包含 -inf 和 inf 的张量 x，数据类型为 S，设备为 device
    const auto x = torch::tensor(
        {-std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()},
        torch::TensorOptions().dtype(S).device(device));
    // 创建与 x 结构相同的张量 y，值也是 -inf 和 inf
    const auto y = torch::tensor(
        {-std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()},
        torch::TensorOptions().dtype(S).device(device));
    // 断言 x 和自身的所有值都近似相等
    ASSERT_TRUE(torch::allclose(x, x));
    // 断言 x 和 y 的所有值都近似相等
    ASSERT_TRUE(torch::allclose(x, y));
    // 断言 y 和 x 的所有值都近似相等
    ASSERT_TRUE(torch::allclose(y, x));
  }
  // 如果 T 类型支持 quiet NaN
  if (std::numeric_limits<T>::has_quiet_NaN) {
    // 创建包含 quiet NaN 的张量 x，数据类型为 S，设备为 device
    const auto x = torch::tensor(
        {std::numeric_limits<T>::quiet_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    // 创建与 x 结构相同的张量 y，值也是 quiet NaN
    const auto y = torch::tensor(
        {std::numeric_limits<T>::quiet_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    // 断言 x 和自身的所有值都近似相等，允许 NaN 值相等
    ASSERT_TRUE(torch::allclose(x, x, 1.0, 0.0, /*equal_nan=*/true));
    // 断言 x 和 y 的所有值都近似相等，允许 NaN 值相等
    ASSERT_TRUE(torch::allclose(x, y, 1.0, 0.0, /*equal_nan=*/true));
    // 断言 y 和 x 的所有值都近似相等，允许 NaN 值相等
    ASSERT_TRUE(torch::allclose(y, x, 1.0, 0.0, /*equal_nan=*/true));
  }
  // 如果 T 类型支持 signaling NaN
  if (std::numeric_limits<T>::has_signaling_NaN) {
    // 创建包含 signaling NaN 的张量 x，数据类型为 S，设备为 device
    const auto x = torch::tensor(
        {std::numeric_limits<T>::signaling_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    // 创建与 x 结构相同的张量 y，值也是 signaling NaN
    const auto y = torch::tensor(
        {std::numeric_limits<T>::signaling_NaN()},
        torch::TensorOptions().dtype(S).device(device));
    // 断言 x 和自身的所有值都近似相等，允许 NaN 值相等
    ASSERT_TRUE(torch::allclose(x, x, 1.0, 0.0, /*equal_nan=*/true));
    # 使用断言确保张量 x 和 y 在允许误差范围内相似，且允许 NaN 值的存在
    ASSERT_TRUE(torch::allclose(x, y, 1.0, 0.0, /*equal_nan=*/true));
    
    # 使用断言确保张量 y 和 x 在允许误差范围内相似，且允许 NaN 值的存在
    ASSERT_TRUE(torch::allclose(y, x, 1.0, 0.0, /*equal_nan=*/true));
}

TEST_F(FunctionalTest, AllClose) {
  // 使用cpu设备进行test_allclose测试
  const at::Device device("cpu");
  test_allclose<torch::kUInt8, uint8_t>(device); // 测试torch::kUInt8类型
  test_allclose<torch::kInt8, int8_t>(device);   // 测试torch::kInt8类型
  test_allclose<torch::kInt16, int16_t>(device); // 测试torch::kInt16类型
  test_allclose<torch::kInt32, int32_t>(device); // 测试torch::kInt32类型
  test_allclose<torch::kInt64, int64_t>(device); // 测试torch::kInt64类型
  test_allclose<torch::kFloat32, float>(device); // 测试torch::kFloat32类型
  test_allclose<torch::kFloat64, double>(device); // 测试torch::kFloat64类型
}

TEST_F(FunctionalTest, AllClose_CUDA) {
  // 使用cuda设备进行test_allclose测试
  const at::Device device("cuda");
  test_allclose<torch::kUInt8, uint8_t>(device); // 测试torch::kUInt8类型
  test_allclose<torch::kInt8, int8_t>(device);   // 测试torch::kInt8类型
  test_allclose<torch::kInt16, int16_t>(device); // 测试torch::kInt16类型
  test_allclose<torch::kInt32, int32_t>(device); // 测试torch::kInt32类型
  test_allclose<torch::kInt64, int64_t>(device); // 测试torch::kInt64类型
  test_allclose<torch::kFloat32, float>(device); // 测试torch::kFloat32类型
  test_allclose<torch::kFloat64, double>(device); // 测试torch::kFloat64类型
  test_allclose<torch::kFloat16, c10::Half>(device); // 测试torch::kFloat16类型
}

TEST_F(FunctionalTest, BCEWithLogitsLoss) {
  { // 测试BCE with logits：当target和input大小不同时会抛出异常
    {
      const auto target = torch::rand(5);
      const auto input = torch::rand({5, 1});
      ASSERT_THROWS_WITH(
          F::binary_cross_entropy_with_logits(input, target),
          "must be the same as input size");
    }
    {
      const auto target = torch::rand({5, 1});
      const auto input = torch::rand(5);
      ASSERT_THROWS_WITH(
          F::binary_cross_entropy_with_logits(input, target),
          "must be the same as input size");
    }
  }
  { // 测试BCE with logits：与sigmoid和bce loss结果相同
    auto sigmoid = Sigmoid();

    auto target = torch::rand({64, 4});
    auto output = torch::rand({64, 4}) - 0.5;

    ASSERT_TRUE(torch::allclose(
        F::binary_cross_entropy_with_logits(output, target),
        F::binary_cross_entropy(sigmoid(output), target)));

    auto weight = torch::rand(4);
    ASSERT_TRUE(torch::allclose(
        F::binary_cross_entropy_with_logits(
            output,
            target,
            F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight)),
        F::binary_cross_entropy(
            sigmoid(output),
            target,
            F::BinaryCrossEntropyFuncOptions().weight(weight))));

    target = torch::zeros({4, 1}, torch::kFloat);
    output = torch::empty({4, 1}, torch::kFloat).fill_(-100);

    ASSERT_TRUE(torch::allclose(
        F::binary_cross_entropy_with_logits(output, target),
        F::binary_cross_entropy(sigmoid(output), target)));

    ASSERT_TRUE(torch::allclose(
        F::binary_cross_entropy_with_logits(
            output,
            target,
            F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(
                torch::kNone)),
        F::binary_cross_entropy(
            sigmoid(output),
            target,
            F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))));

    weight = torch::rand({1}, torch::kFloat);
  { // 测试二进制交叉熵与Logits的一致性
    // 创建输出张量和目标张量，均为3行1列的零张量
    const auto output = torch::zeros({3, 1}, torch::requires_grad());
    const auto target = torch::zeros({3, 1});
    // 计算二进制交叉熵与Logits的损失，设置为对每个样本求和的方式，并进行反向传播
    F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kSum))
        .backward();
    // 期望的梯度为全为0.5的张量
    const auto expected_grad = torch::empty({3, 1}).fill_(0.5);
    // 断言输出的梯度与期望的梯度在数值上的近似性
    ASSERT_TRUE(torch::allclose(output.grad(), expected_grad));
  }
  { // 测试二进制交叉熵与Logits对权重的广播
    // 创建目标张量和输出张量，形状为16行4列的随机张量
    const auto target = torch::rand({16, 4});
    const auto output = torch::rand({16, 4}) - 0.5;

    auto weight = torch::rand(4);
    // 使用权重对应的选项计算二进制交叉熵与Logits的损失
    auto out1 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight));

    // 将权重扩展到与输出张量相同的形状，并保持连续性
    weight = weight.expand({16, 4}).contiguous();
    // 使用扩展后的权重计算二进制交叉熵与Logits的损失
    auto out2 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight));

    // 断言两种计算方式得到的损失在数值上的近似性
    ASSERT_TRUE(torch::allclose(out1, out2));

    // 创建形状为16行1列的随机权重张量
    weight = torch::rand({16, 1});
    // 使用权重对应的选项计算二进制交叉熵与Logits的损失
    out1 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight));

    // 将权重扩展到与输出张量相同的形状，并保持连续性
    weight = weight.expand({16, 4}).contiguous();
    // 使用扩展后的权重计算二进制交叉熵与Logits的损失
    out2 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight));

    // 断言两种计算方式得到的损失在数值上的近似性
    ASSERT_TRUE(torch::allclose(out1, out2));
  }
  { // 测试二进制交叉熵与Logits中正样本权重的一致性
    // 创建目标张量和输出张量，形状为64行4列的随机张量
    const auto target = torch::rand({64, 4});
    const auto output = torch::rand({64, 4}) - 0.5;
    // 创建形状为64行4列的全1正样本权重张量
    const auto pos_weight = torch::ones({64, 4});

    // 断言不使用正样本权重和使用全1正样本权重两种方式计算得到的损失在数值上的近似性
    ASSERT_TRUE(torch::allclose(
        F::binary_cross_entropy_with_logits(output, target),
        F::binary_cross_entropy_with_logits(
            output,
            target,
            F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(
                pos_weight))));
  }
  { // 测试二进制交叉熵与Logits中正样本权重的广播
    // 创建目标张量和输出张量，形状为64行4列的随机张量
    const auto target = torch::rand({64, 4});
    const auto output = torch::rand({64, 4}) - 0.5;
    // 创建形状为4列的随机正样本权重张量
    const auto pos_weight = torch::rand(4);
    // 使用指定的正样本权重计算二进制交叉熵与Logits的损失
    const auto out1 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight));

    // 将正样本权重扩展到与目标张量相同的形状，并保持连续性
    const auto pos_weight1 = pos_weight.expand({1, 4});
    // 使用扩展后的正样本权重计算二进制交叉熵与Logits的损失
    const auto out2 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight));

    // 断言两种计算方式得到的损失在数值上的近似性
    const auto pos_weight2 = pos_weight.expand({64, 4});
    // 计算不带权重的二进制交叉熵损失
    const auto out3 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight));

    // 断言：验证 out1 和 out2 是否近似相等
    ASSERT_TRUE(torch::allclose(out1, out2));

    // 断言：验证 out1 和 out3 是否近似相等
    ASSERT_TRUE(torch::allclose(out1, out3));
  }
  { // 测试带正权重的二进制交叉熵损失在梯度为零时的正确性
    const auto output = torch::zeros({3, 1}, torch::requires_grad());
    const auto target = torch::zeros({3, 1});
    const auto pos_weight = torch::ones({3, 1});

    // 计算带正权重的二进制交叉熵损失，并进行反向传播
    F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions()
            .pos_weight(pos_weight)
            .reduction(torch::kSum))
        .backward();

    // 预期的梯度，每个元素为 0.5
    const auto expected_grad = torch::empty({3, 1}).fill_(0.5);

    // 获取计算得到的梯度
    const auto grad = output.grad();

    // 断言：验证计算得到的梯度与预期的梯度是否近似相等
    ASSERT_TRUE(torch::allclose(grad, expected_grad));
  }
  { // 测试带 logits 的二进制交叉熵在稳定性方面的特性
    const auto output = torch::tensor({0., -120.});
    const auto target = torch::tensor({0., 1.});
    const auto pos_weight = torch::tensor({1., 1.});

    // 计算不带权重的二进制交叉熵损失
    const auto out1 = F::binary_cross_entropy_with_logits(output, target);

    // 断言：验证 out1 的所有元素是否均为有限数值
    ASSERT_TRUE(torch::isfinite(out1).all().item<bool>());

    // 计算带正权重的二进制交叉熵损失
    const auto out2 = F::binary_cross_entropy_with_logits(
        output,
        target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight));

    // 断言：验证 out2 的所有元素是否均为有限数值
    ASSERT_TRUE(torch::isfinite(out2).all().item<bool>());
  }
}



# 这是一个单独的右大括号 '}'，用于结束一个代码块或数据结构（如字典、集合等）。在程序中，右大括号与对应的左大括号 '{' 一起用来定义代码块的范围，确保程序在执行时可以正确地识别和处理各个代码段。
```