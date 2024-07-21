# `.\pytorch\test\cpp\api\modules.cpp`

```
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <c10/util/irange.h> // 引入 Caffe2 的 c10 库中的 irange.h 头文件
#include <torch/torch.h> // 引入 PyTorch 核心头文件

#include <test/cpp/api/support.h> // 引入测试支持函数的头文件

#include <torch/expanding_array.h> // 引入 PyTorch 中的扩展数组头文件
#include <torch/nn/functional/activation.h> // 引入 PyTorch 中神经网络模块的激活函数头文件
#include <torch/nn/options/activation.h> // 引入 PyTorch 中神经网络模块的激活函数选项头文件
#include <limits> // 引入 C++ 标准库中的 limits 头文件，用于提供数值极限值
#include <random> // 引入 C++ 标准库中的随机数生成器头文件

using namespace torch::nn; // 使用 PyTorch 中神经网络模块的命名空间
using namespace torch::test; // 使用 PyTorch 测试命名空间

class TestModel : public torch::nn::Module { // 定义 TestModel 类，继承自 PyTorch 中的 Module 类
 public:
  TestModel()
      : l1(register_module("l1", Linear(10, 3))), // 初始化 l1 线性层，输入维度 10，输出维度 3
        l2(register_module("l2", Linear(3, 5))), // 初始化 l2 线性层，输入维度 3，输出维度 5
        l3(register_module("l3", Linear(5, 100))) {} // 初始化 l3 线性层，输入维度 5，输出维度 100

  Linear l1, l2, l3; // 定义 l1, l2, l3 三个线性层对象
};

class NestedModel : public torch::nn::Module { // 定义 NestedModel 类，继承自 PyTorch 中的 Module 类
 public:
  NestedModel()
      : param_(register_parameter("param", torch::empty({3, 2, 21}))), // 初始化 param_ 参数张量，大小为 [3, 2, 21]
        l1(register_module("l1", Linear(5, 20))), // 初始化 l1 线性层，输入维度 5，输出维度 20
        t(register_module("test", std::make_shared<TestModel>())) {} // 初始化 t 测试模型，共享指针指向 TestModel 实例

  torch::Tensor param_; // 定义 param_ 参数张量
  Linear l1; // 定义 l1 线性层对象
  std::shared_ptr<TestModel> t; // 定义 t 测试模型的共享指针
};

struct ModulesTest : torch::test::SeedingFixture {}; // 定义 ModulesTest 结构体，继承自 SeedingFixture 类，用于模块测试

TEST_F(ModulesTest, Conv1d) { // 定义 Conv1d 测试用例，继承自 ModulesTest 结构体
  Conv1d model(Conv1dOptions(3, 2, 3).stride(1).bias(false)); // 创建 Conv1d 模型，设置输入通道 3，输出通道 2，卷积核大小 3，步长 1，无偏置
  model->weight.set_data( // 设置模型权重数据
      torch::arange(18, torch::dtype(torch::kFloat)).reshape({2, 3, 3})); // 使用 torch::arange 生成的浮点数张量，重塑为 [2, 3, 3]
  auto x = torch::arange(30, torch::dtype(torch::kFloat).requires_grad(true)) // 创建张量 x，从 0 到 29，数据类型为浮点数，要求梯度
               .reshape({2, 3, 5}); // 重塑张量 x 的形状为 [2, 3, 5]
  auto y = model(x); // 对输入张量 x 进行模型前向传播，得到输出张量 y
  auto expected = torch::tensor( // 创建期望结果张量 expected
      {{{312., 348., 384.}, {798., 915., 1032.}}, // 第一个样本的期望输出
       {{852., 888., 924.}, {2553., 2670., 2787.}}}, // 第二个样本的期望输出
      torch::kFloat); // 指定数据类型为浮点数
  ASSERT_TRUE(torch::allclose(y, expected)); // 使用 allclose 函数检查模型输出 y 是否与期望值 expected 接近

  torch::Tensor s = y.sum(); // 计算张量 y 的所有元素的和，得到张量 s
  s.backward(); // 执行张量 s 的反向传播
  ASSERT_EQ(s.ndimension(), 0); // 断言张量 s 的维度为 0
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3); // 断言模型权重的梯度张量元素个数为 3 * 2 * 3
}

TEST_F(ModulesTest, Conv1dSameStrided) { // 定义 Conv1dSameStrided 测试用例，继承自 ModulesTest 结构体
  auto options = Conv1dOptions(3, 2, 3); // 创建 Conv1dOptions 选项，设置输入通道 3，输出通道 2，卷积核大小 3
  options.stride(1).padding(torch::kSame); // 设置选项的步长为 1，填充方式为 "same"
  Conv1d model_valid(options); // 创建有效的 Conv1d 模型，根据给定选项
  ASSERT_THROWS_WITH( // 使用 ASSERT_THROWS_WITH 断言检查异常信息
      [&] { Conv1d model_invalid(options.stride(2)); }(), // 尝试创建无效的 Conv1d 模型，设置步长为 2
      "padding='same' is not supported for strided convolutions"); // 断言异常信息为 "padding='same' is not supported for strided convolutions"
}

TEST_F(ModulesTest, Conv1dIvalidArg) { // 定义 Conv1dIvalidArg 测试用例，继承自 ModulesTest 结构体
  auto options = Conv1dOptions(3, 2, 3).groups(-1); // 创建 Conv1dOptions 选项，设置输入通道 3，输出通道 2，卷积核大小 3，设置组数为 -1
  ASSERT_THROWS_WITH( // 使用 ASSERT_THROWS_WITH 断言检查异常信息
      Conv1d(options), // 尝试创建 Conv1d 模型，根据给定选项
      "in_channels, groups and out_channels must"); // 断言异常信息为 "in_channels, groups and out_channels must"
}

TEST_F(ModulesTest, Conv2dEven) { // 定义 Conv2dEven 测试用例，继承自 ModulesTest 结构体
  Conv2d model(Conv2dOptions(3, 2, 3).stride(1).bias(false)); // 创建 Conv2d 模型，设置输入通道 3，输出通道 2，卷积核大小 3，步长 1，无偏置
  model->weight.set_data( // 设置模型权重数据
      torch::arange(54, torch::dtype(torch::kFloat)).reshape({2, 3, 3, 3})); // 使用 torch::arange 生成的浮点数张量，重塑为 [2, 3, 3, 3]
  auto x = torch::arange(75, torch::dtype(torch::kFloat).requires_grad(true)) // 创建张量 x，从 0 到 74，数据类型为浮点数，要求梯度
               .reshape({1, 3, 5, 5}); // 重塑张量 x 的形状为 [1, 3, 5, 5]
  auto y = model(x); // 对输入张量 x 进行模型前向传播，得到输出张量 y
  auto expected = torch::tensor( // 创建期望结果张量 expected
      {{{{15219., 15570., 15921.}, // 第一个
TEST_F(ModulesTest, Conv2dUneven) {
  // 创建一个 Conv2d 模型，设置输入通道数为 3，输出通道数为 2，卷积核大小为 {3, 2}，步长为 {1, 1}，无偏置
  Conv2d model(Conv2dOptions(3, 2, {3, 2}).stride({1, 1}).bias(false));
  // 设置模型的权重，使用 torch::arange 生成的张量，reshape 成 {2, 3, 3, 2} 的形状
  model->weight.set_data(
      torch::arange(36, torch::dtype(torch::kFloat)).reshape({2, 3, 3, 2}));
  // 创建输入张量 x，使用 torch::arange 生成的张量，形状为 {1, 3, 5, 4}，要求梯度计算
  auto x = torch::arange(60, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 4});
  // 将输入张量 x 输入模型，计算输出张量 y
  auto y = model(x);
  // 创建期望输出张量 expected，形状为 {{{{...}}}}，数据类型为 torch::kFloat
  auto expected = torch::tensor(
      {{{{5289., 5442., 5595.}, {5901., 6054., 6207.}, {6513., 6666., 6819.}},
        {{13227., 13704., 14181.},
         {15135., 15612., 16089.},
         {17043., 17520., 17997.}}}},
      torch::kFloat);
  // 断言模型输出 y 与期望输出 expected 接近
  ASSERT_TRUE(torch::allclose(y, expected));

  // 计算张量 y 的所有元素之和，并赋值给张量 s
  torch::Tensor s = y.sum();
  // 对张量 s 进行反向传播
  s.backward();
  // 断言张量 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言模型权重的梯度张量的元素数量正确，应为 3 * 2 * 3 * 2
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 2);
}

TEST_F(ModulesTest, Conv2dSameStrided) {
  // 创建 Conv2d 模型选项，设置输入通道数为 3，输出通道数为 2，卷积核大小为 {3, 4}
  auto options = Conv2dOptions(3, 2, {3, 4});
  // 设置模型选项的步长为 1，并设置填充方式为 torch::kSame
  options.stride(1).padding(torch::kSame);
  // 创建支持 "same" 填充方式的 Conv2d 模型
  Conv2d model_valid(options);
  // 断言尝试使用不支持 "same" 填充方式的 Conv2d 模型会抛出异常
  ASSERT_THROWS_WITH(
      [&] { Conv2d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
  // 断言尝试使用不支持 "same" 填充方式的 Conv2d 模型会抛出异常
  ASSERT_THROWS_WITH(
      [&] {
        Conv2d model_invalid(options.stride({1, 2}));
      }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, Conv3d) {
  // 创建一个 Conv3d 模型，设置输入通道数为 3，输出通道数为 2，卷积核大小为 {3, 3, 3}，步长为 1，无偏置
  Conv3d model(Conv3dOptions(3, 2, 3).stride(1).bias(false));
  // 设置模型的权重，使用 torch::arange 生成的张量，reshape 成 {2, 3, 3, 3, 3} 的形状
  model->weight.set_data(
      torch::arange(162, torch::dtype(torch::kFloat)).reshape({2, 3, 3, 3, 3}));
  // 创建输入张量 x，使用 torch::arange 生成的张量，形状为 {1, 3, 5, 5, 5}，要求梯度计算
  auto x = torch::arange(375, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5, 5});
  // 将输入张量 x 输入模型，计算输出张量 y
  auto y = model(x);
  // 创建期望输出张量 expected，形状为 {{{{{...}}}}}，数据类型为 torch::kFloat
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
  // 断言模型输出 y 与期望输出 expected 接近
  ASSERT_TRUE(torch::allclose(y, expected));

  // 计算张量 y 的所有元素之和，并赋值给张量 s
  torch::Tensor s = y.sum();
  // 对张量 s 进行反向传播
  s.backward();
  // 断言张量 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言模型权重的梯度张量的元素数量正确，应为 3 * 2 * 3 * 3 * 3
  ASSERT_TRUE(model->weight.grad().numel() == 3 * 2 * 3 * 3 * 3);
}
TEST_F(ModulesTest, Conv3dSameStrided) {
  // 创建 Conv3d 的选项，设置卷积核大小为 3x3x3，输出通道数为 2
  auto options = Conv3dOptions(3, 2, {3, 4, 5});
  // 设置卷积的步长为 1，并且使用 'same' 填充方式
  options.stride(1).padding(torch::kSame);
  // 创建 Conv3d 模型对象，使用上述选项
  Conv3d model_valid(options);
  // 使用不支持 'same' 填充和步长为 2 的选项创建 Conv3d 模型，预期抛出异常
  ASSERT_THROWS_WITH(
      [&] { Conv3d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
  // 使用不支持 'same' 填充和不同步长的选项创建 Conv3d 模型，预期抛出异常
  ASSERT_THROWS_WITH(
      [&] {
        Conv3d model_invalid(options.stride({1, 2, 1}));
      }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, ConvTranspose1d) {
  // 创建 ConvTranspose1d 模型对象，设置输入通道为 3，输出通道为 2，卷积核大小为 3
  ConvTranspose1d model(ConvTranspose1dOptions(3, 2, 3).stride(1).bias(false));
  // 设置模型的权重数据，使用 arange 生成的张量视图
  model->weight.set_data(torch::arange(18.).view({2, 3, 3}));
  // 创建输入张量 x，使用 arange 生成，reshape 成 {2, 2, 5}
  auto x = torch::arange(20.).reshape({2, 2, 5});
  // 对模型应用输入张量 x，得到输出张量 y
  auto y = model(x);
  // 创建期望输出张量 expected
  auto expected = torch::tensor(
      {{{45., 104., 179., 212., 245., 188., 107.},
        {60., 140., 242., 293., 344., 260., 146.},
        {75., 176., 305., 374., 443., 332., 185.}},
       {{135., 304., 509., 542., 575., 428., 237.},
        {210., 460., 752., 803., 854., 620., 336.},
        {285., 616., 995., 1064., 1133., 812., 435.}}});
  // 断言模型输出 y 与期望输出 expected 接近
  ASSERT_TRUE(torch::allclose(y, expected));

  // 计算输出张量 y 的所有元素的和，得到张量 s
  torch::Tensor s = y.sum();
  // 对张量 s 进行反向传播
  s.backward();
  // 断言张量 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言模型权重的梯度张量的元素数量为 3 * 2 * 3
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3);
}
// 定义一个测试用例，测试 ConvTranspose2d 模型的反卷积操作（偶数情况）
TEST_F(ModulesTest, ConvTranspose2dEven) {
  // 创建一个 ConvTranspose2d 模型，设置输入通道为3，输出通道为2，卷积核大小为3x3，步长为1，不使用偏置
  ConvTranspose2d model(ConvTranspose2dOptions(3, 2, 3).stride(1).bias(false));
  // 设置模型的权重为一个序列化的张量，包含54个元素，形状为{2, 3, 3, 3}
  model->weight.set_data(torch::arange(54.).view({2, 3, 3, 3}));
  // 创建一个输入张量 x，包含50个元素，形状为{1, 2, 5, 5}
  auto x = torch::arange(50.).view({1, 2, 5, 5});
  // 对输入张量 x 进行反卷积操作，得到输出张量 y
  auto y = model(x);
  // 预期的输出张量 expected，形状为{1, 2, 7, 7}，包含预先计算好的数值
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
  // 断言输出张量 y 与预期的 expected 张量在误差允许范围内相等
  ASSERT_TRUE(torch::allclose(y, expected));

  // 计算输出张量 y 的所有元素的和，得到张量 s
  torch::Tensor s = y.sum();
  // 对张量 s 进行反向传播
  s.backward();
  // 断言张量 s 的维度为0，即为标量
  ASSERT_EQ(s.ndimension(), 0);
  // 断言模型的权重梯度张量的元素数量符合预期的计算公式
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 3);
}
TEST_F(ModulesTest, ConvTranspose2dUneven) {
  // 创建 ConvTranspose2d 模型，指定输入通道数为3，输出通道数为2，卷积核大小为{3, 2}，步长为{1, 1}，不使用偏置
  ConvTranspose2d model(
      ConvTranspose2dOptions(3, 2, {3, 2}).stride({1, 1}).bias(false));
  // 设置模型的权重数据，使用torch::arange(36.)生成的张量，形状为{2, 3, 3, 2}
  model->weight.set_data(torch::arange(36.).view({2, 3, 3, 2}));
  // 创建输入张量 x，使用torch::arange(40.)生成的张量，形状为{1, 2, 5, 4}
  auto x = torch::arange(40.).view({1, 2, 5, 4});
  // 将输入张量 x 传入模型进行前向计算，得到输出张量 y
  auto y = model(x);
  // 创建期望输出张量 expected，包含预期的输出结果
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
  // 使用断言验证模型输出 y 与期望输出 expected 的接近程度
  ASSERT_TRUE(torch::allclose(y, expected));

  // 对输出张量 y 求和，得到标量张量 s
  torch::Tensor s = y.sum();
  // 对标量张量 s 进行反向传播
  s.backward();
  // 使用断言验证 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 使用断言验证模型权重的梯度张量的元素数目为 3 * 2 * 3 * 2
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 2);
}

TEST_F(ModulesTest, ConvTranspose3d) {
  // 创建 ConvTranspose3d 模型，指定输入通道数为2，输出通道数为2，卷积核大小为{2, 2, 2}，步长为1，不使用偏置
  ConvTranspose3d model(ConvTranspose3dOptions(2, 2, 2).stride(1).bias(false));
  // 设置模型的权重数据，使用torch::arange(32.)生成的张量，形状为{2, 2, 2, 2, 2}
  model->weight.set_data(torch::arange(32.).reshape({2, 2, 2, 2, 2}));
  // 创建输入张量 x，使用torch::arange(16.)生成的张量，形状为{1, 2, 2, 2, 2}
  auto x = torch::arange(16.).reshape({1, 2, 2, 2, 2});
  // 将输入张量 x 传入模型进行前向计算，得到输出张量 y
  auto y = model(x);
  // 创建期望输出张量 expected，包含预期的输出结果
  auto expected = torch::tensor(
      {{{{{128., 280., 154.}, {304., 664., 364.}, {184., 400., 218.}},
         {{352., 768., 420.}, {832., 1808., 984.}, {496., 1072., 580.}},
         {{256., 552., 298.}, {592., 1272., 684.}, {344., 736., 394.}}},
        {{{192., 424., 234.}, {464., 1016., 556.}, {280., 608., 330.}},
         {{544., 1184., 644.}, {1280., 2768., 1496.}, {752., 1616., 868.}},
         {{384., 824., 442.}, {880., 1880., 1004.}, {504., 1072., 570.}}}}});
  // 使用断言验证模型输出 y 与期望输出 expected 的接近程度
  ASSERT_TRUE(torch::allclose(y, expected));

  // 对输出张量 y 求和，得到标量张量 s
  torch::Tensor s = y.sum();
  // 对标量张量 s 进行反向传播
  s.backward();
  // 使用断言验证 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 使用断言验证模型权重的梯度张量的元素数目为 2 * 2 * 2 * 2 * 2
  ASSERT_TRUE(model->weight.grad().numel() == 2 * 2 * 2 * 2 * 2);
}

TEST_F(ModulesTest, MaxPool1d) {
  // 创建 MaxPool1d 模型，指定池化窗口大小为3，步长为2
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  // 创建输入张量 x，使用torch::ones({1, 1, 5}, torch::requires_grad())生成的张量，形状为{1, 1, 5}
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  // 将输入张量 x 传入模型进行池化操作，得到输出张量 y
  auto y = model(x);
  // 对输出张量 y 求和，得到标量张量 s
  torch::Tensor s = y.sum();

  // 对标量张量 s 进行反向传播
  s.backward();
  // 使用断言验证 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 使用断言验证 y 的值接近于形状为{1, 1, 2}的张量全为1
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  // 使用断言验证 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 使用断言验证 y 的形状为{1, 1, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}
TEST_F(ModulesTest, MaxPool1dReturnIndices) {
  // 创建一个 MaxPool1d 模型，设置窗口大小为3，步幅为2
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  // 创建一个形状为[1, 1, 5]的张量x，所有元素为1，并要求梯度计算
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  // 声明张量y和indices，并使用模型计算前向传播结果
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  // 断言y的维度为3
  ASSERT_EQ(y.dim(), 3);
  // 断言y与形状为[1, 1, 2]的张量torch::ones({1, 1, 2})在数值上相近
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  // 断言y的形状为[1, 1, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));

  // 断言indices与形状为[1, 1, 2]的长整型张量{{0, 2}}在数值上相近
  ASSERT_TRUE(
      torch::allclose(indices, torch::tensor({{{0, 2}}}, torch::kLong)));
  // 断言indices的形状为[1, 1, 2]
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool2dEven) {
  // 创建一个 MaxPool2d 模型，设置窗口大小为3，步幅为2
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  // 创建一个形状为[2, 5, 5]的张量x，所有元素为1，并要求梯度计算
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  // 使用模型计算张量x的前向传播结果
  auto y = model(x);
  // 计算y的所有元素之和
  torch::Tensor s = y.sum();

  // 对s进行反向传播
  s.backward();
  // 断言y的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言y与形状为[2, 2, 2]的张量torch::ones({2, 2, 2})在数值上相近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言s的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言y的形状为[2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dUneven) {
  // 创建一个 MaxPool2d 模型，设置窗口大小为[3, 2]，步幅为[2, 2]
  MaxPool2d model(MaxPool2dOptions({3, 2}).stride({2, 2}));
  // 创建一个形状为[2, 5, 4]的张量x，所有元素为1，并要求梯度计算
  auto x = torch::ones({2, 5, 4}, torch::requires_grad());
  // 使用模型计算张量x的前向传播结果
  auto y = model(x);
  // 计算y的所有元素之和
  torch::Tensor s = y.sum();

  // 对s进行反向传播
  s.backward();
  // 断言y的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言y与形状为[2, 2, 2]的张量torch::ones({2, 2, 2})在数值上相近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言s的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言y的形状为[2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dReturnIndices) {
  // 创建一个 MaxPool2d 模型，设置窗口大小为3，步幅为2
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  // 创建一个形状为[2, 5, 5]的张量x，所有元素为1，并要求梯度计算
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  // 声明张量y和indices，并使用模型计算前向传播结果
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  // 断言y的维度为3
  ASSERT_EQ(y.dim(), 3);
  // 断言y与形状为[2, 2, 2]的张量torch::ones({2, 2, 2})在数值上相近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言y的形状为[2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
  // 断言indices与形状为[2, 2, 2]的长整型张量{{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}}在数值上相近
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor({{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}}, torch::kLong)));
  // 断言indices的形状为[2, 2, 2]
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3d) {
  // 创建一个 MaxPool3d 模型，设置窗口大小为3，步幅为2
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  // 创建一个形状为[2, 5, 5, 5]的张量x，所有元素为1，并要求梯度计算
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  // 使用模型计算张量x的前向传播结果
  auto y = model(x);
  // 计算y的所有元素之和
  torch::Tensor s = y.sum();

  // 对s进行反向传播
  s.backward();
  // 断言y的维度为4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言y与形状为[2, 2, 2, 2]的张量torch::ones({2, 2, 2, 2})在数值上相近
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  // 断言s的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言y的形状为[2, 2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3dReturnIndices) {
  // 创建一个 MaxPool3d 模型，设置窗口大小为3，步幅为2
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  // 创建一个形状为[2, 5, 5, 5]的张量x，所有元素为1，并要求梯度计算
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  // 声明张量y和indices，并使用模型计算前向传播结果
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  // 断言y的维度为4
  ASSERT_EQ(y.dim(),
// 在 ModulesTest 测试套件中定义 AvgPool1d 测试用例
TEST_F(ModulesTest, AvgPool1d) {
  // 创建 AvgPool1d 模型，设置窗口大小为3，步长为2
  AvgPool1d model(AvgPool1dOptions(3).stride(2));
  // 创建输入张量 x，形状为 {1, 1, 5}，所有元素为1，需要梯度
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  // 将输入张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和，得到标量张量 s
  torch::Tensor s = y.sum();

  // 对标量张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 的所有元素接近于全为1
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  // 断言标量张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的形状为 {1, 1, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

// 在 ModulesTest 测试套件中定义 AvgPool2dEven 测试用例
TEST_F(ModulesTest, AvgPool2dEven) {
  // 创建 AvgPool2d 模型，设置窗口大小为3，步长为2
  AvgPool2d model(AvgPool2dOptions(3).stride(2));
  // 创建输入张量 x，形状为 {2, 5, 5}，所有元素为1，需要梯度
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  // 将输入张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和，得到标量张量 s
  torch::Tensor s = y.sum();

  // 对标量张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 的所有元素接近于全为1
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言标量张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的形状为 {2, 2, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

// 在 ModulesTest 测试套件中定义 AvgPool2dUneven 测试用例
TEST_F(ModulesTest, AvgPool2dUneven) {
  // 创建 AvgPool2d 模型，设置窗口大小为 {3, 2}，步长为 {2, 2}
  AvgPool2d model(AvgPool2dOptions({3, 2}).stride({2, 2}));
  // 创建输入张量 x，形状为 {2, 5, 4}，所有元素为1，需要梯度
  auto x = torch::ones({2, 5, 4}, torch::requires_grad());
  // 将输入张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和，得到标量张量 s
  torch::Tensor s = y.sum();

  // 对标量张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 的所有元素接近于全为1
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言标量张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的形状为 {2, 2, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

// 在 ModulesTest 测试套件中定义 AvgPool3d 测试用例
TEST_F(ModulesTest, AvgPool3d) {
  // 创建 AvgPool3d 模型，设置窗口大小为3，步长为2
  AvgPool3d model(AvgPool3dOptions(3).stride(2));
  // 创建输入张量 x，形状为 {2, 5, 5, 5}，所有元素为1，需要梯度
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  // 将输入张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和，得到标量张量 s
  torch::Tensor s = y.sum();

  // 对标量张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言输出张量 y 的所有元素接近于全为1
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  // 断言标量张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的形状为 {2, 2, 2, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

// 在 ModulesTest 测试套件中定义 FractionalMaxPool2d 测试用例
TEST_F(ModulesTest, FractionalMaxPool2d) {
  // 创建 FractionalMaxPool2d 模型，设置窗口大小为3，输出尺寸为2
  FractionalMaxPool2d model(FractionalMaxPool2dOptions(3).output_size(2));
  // 创建输入张量 x，形状为 {2, 5, 5}，所有元素为1，需要梯度
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  // 将输入张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和，得到标量张量 s
  torch::Tensor s = y.sum();

  // 对标量张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 的所有元素接近于全为1
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  // 断言标量张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的形状为 {2, 2, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

// 在 ModulesTest 测试套件中定义 FractionalMaxPool2dReturnIndices 测试用例
TEST_F(ModulesTest, FractionalMaxPool2dReturnIndices) {
  // 创建 FractionalMaxPool2d 模型，设置窗口大小为3，输出尺寸为2
  FractionalMaxPool2d model(FractionalMaxPool2dOptions(3).output_size(2));
  // 创建输入张量 x，形状为 {2, 5, 5}，所有元素为1，需要梯度
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  // 声明输出张量 y 和索引张量 indices
  torch::Tensor y, indices;
  // 将输入张量 x 输入模型，同时得到输出张量 y 和索引张量 indices
  std::tie(y, indices) = model->forward_with_indices(x);

  // 断言输出张量 y 的维度为3
  ASSERT_EQ(y.dim(), 3);
  // 断
// 定义测试用例，测试 FractionalMaxPool3d 模型
TEST_F(ModulesTest, FractionalMaxPool3d) {
  // 创建 FractionalMaxPool3d 模型对象，设置输出尺寸为 2
  FractionalMaxPool3d model(FractionalMaxPool3dOptions(3).output_size(2));
  // 创建一个大小为 [2, 5, 5, 5] 的张量 x，要求计算梯度
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  // 将张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 计算张量 y 的所有元素的和，存入张量 s
  torch::Tensor s = y.sum();

  // 对张量 s 进行反向传播
  s.backward();
  // 断言张量 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言张量 y 的所有元素与大小为 [2, 2, 2, 2] 的张量 torch::ones 相等
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  // 断言张量 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言张量 y 的大小为 [2, 2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

// 定义测试用例，测试 FractionalMaxPool3d 返回索引的情况
TEST_F(ModulesTest, FractionalMaxPool3dReturnIndices) {
  // 创建 FractionalMaxPool3d 模型对象，设置输出尺寸为 2
  FractionalMaxPool3d model(FractionalMaxPool3dOptions(3).output_size(2));
  // 创建一个大小为 [2, 5, 5, 5] 的张量 x，要求计算梯度
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  // 声明张量 y 和 indices，并将模型处理张量 x 的结果分别存入 y 和 indices
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  // 断言张量 y 的维度为 4
  ASSERT_EQ(y.dim(), 4);
  // 断言张量 y 的所有元素与大小为 [2, 2, 2, 2] 的张量 torch::ones 相等
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  // 断言张量 y 的大小为 [2, 2, 2, 2]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  // 断言张量 indices 的所有元素与给定的张量 torch::tensor 相等
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {{{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}},
           {{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}}})));
  // 断言张量 indices 的大小为 [2, 2, 2, 2]
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

// 定义测试用例，测试 LPPool1d 模型
TEST_F(ModulesTest, LPPool1d) {
  // 定义规范化类型为 2，步长为 2，核大小为 3 的 LPPool1d 模型对象
  int norm_type = 2;
  int stride = 2;
  int kernel_size = 3;
  LPPool1d model(LPPool1dOptions(norm_type, kernel_size).stride(stride));
  // 创建一个大小为 [1, 1, 5] 的张量 x
  auto x = torch::ones({1, 1, 5});
  // 将张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 根据预期计算张量 y 的期望值 expected
  auto expected =
      (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) *
       kernel_size)
          .pow(1. / norm_type);

  // 断言张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言张量 y 的所有元素与预期值 expected 相等
  ASSERT_TRUE(torch::allclose(y, expected));
  // 断言张量 y 的大小为 [1, 1, 2]
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

// 定义测试用例，测试 LPPool2d 模型
TEST_F(ModulesTest, LPPool2d) {
  // 定义规范化类型为 2，步长为 2，核大小为 [2, 3] 的 LPPool2d 模型对象
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({2, 3});
  LPPool2d model(LPPool2dOptions(norm_type, kernel_size).stride(stride));
  // 创建一个大小为 [1, 1, 2, 5] 的张量 x
  auto x = torch::ones({1, 1, 2, 5});
  // 将张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 根据预期计算张量 y 的期望值 expected
  auto expected =
      (torch::pow(torch::tensor({{{{1, 1}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1]))
          .pow(1. / norm_type);

  // 断言张量 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言张量 y 的所有元素与预期值 expected 相等
  ASSERT_TRUE(torch::allclose(y, expected));
  // 断言张量 y 的大小为 [1, 1, 1, 2]
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 2}));
}

// 定义测试用例，测试 LPPool3d 模型
TEST_F(ModulesTest, LPPool3d) {
  // 定义规范化类型为 2，步长为 2，核大小为 [1, 2, 3] 的 LPPool3d 模型对象
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({1, 2, 3});
  LPPool3d model(LPPool3dOptions(norm_type, kernel_size).stride(stride));
  // 创建一个大小为 [1, 1, 1, 2, 5] 的张量 x
  auto x = torch::ones({1, 1, 1, 2, 5});
  // 将张量 x 输入模型，得到输出张量 y
  auto y = model(x);
  // 根据预期计算张量 y 的期望值 expected
  auto expected =
      (torch::pow(torch::tensor({{{{{1, 1}}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1] * kernel_size[2]))
          .pow(1. / norm_type);

  // 断言张量 y 的维度为 5
  ASSERT_EQ(y.ndimension(), 5);
  // 断言张量 y 的所有元素与预期值 expected 相等
  ASSERT_TRUE(torch::allclose(y, expected));
  // 断言张量 y 的大小为 [1, 1, 1, 1, 2]
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 1, 2}));
}
TEST_F(ModulesTest, Identity) {
  // 创建 Identity 模块的实例
  Identity identity;
  // 创建输入张量，并设置 requires_grad 为 true，表示需要计算梯度
  auto input = torch::tensor(
      {{1, 3, 4}, {2, 3, 4}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 使用 Identity 模块进行前向传播计算
  auto output = identity->forward(input);
  // 创建预期输出张量
  auto expected = torch::tensor({{1, 3, 4}, {2, 3, 4}}, torch::kFloat);
  // 计算输出张量的所有元素之和
  auto s = output.sum();
  // 对计算得到的标量 s 进行反向传播
  s.backward();

  // 断言输出张量与预期张量相等
  ASSERT_TRUE(torch::equal(output, expected));
  // 断言输入张量的梯度与全1张量相等
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));
}

TEST_F(ModulesTest, Flatten) {
  // 创建 Flatten 模块的实例
  Flatten flatten;
  // 创建输入张量，并设置 requires_grad 为 true，表示需要计算梯度
  auto input = torch::tensor(
      {{1, 3, 4}, {2, 5, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 使用 Flatten 模块进行前向传播计算
  auto output = flatten->forward(input);
  // 创建预期输出张量
  auto expected = torch::tensor({{1, 3, 4}, {2, 5, 6}}, torch::kFloat);
  // 计算输出张量的所有元素之和
  auto s = output.sum();

  // 对计算得到的标量 s 进行反向传播
  s.backward();
  // 断言输出张量与预期张量相等
  ASSERT_TRUE(torch::equal(output, expected));
  // 断言输入张量的梯度与全1张量相等
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));

  // 使用指定的可选参数 start_dim 和 end_dim 进行测试
  Flatten flatten_optional_dims(FlattenOptions().start_dim(2).end_dim(3));
  // 创建新的输入张量，并设置 requires_grad 为 true，表示需要计算梯度
  input = torch::tensor(
      {{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
       {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}},
      torch::dtype(torch::kFloat)
          .requires_grad(true)); // Tensor with sizes (2, 2, 2, 2)

  // 使用带有可选参数的 Flatten 模块进行前向传播计算
  output = flatten_optional_dims->forward(input);
  // 创建预期输出张量
  expected = torch::tensor(
      {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}},
      torch::kFloat); // Tensor with sizes (2, 2, 4)

  // 计算输出张量的所有元素之和
  s = output.sum();
  // 对计算得到的标量 s 进行反向传播
  s.backward();
  // 断言输出张量与预期张量相等
  ASSERT_TRUE(torch::equal(output, expected));
  // 断言输入张量的梯度与全1张量相等
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));
}

TEST_F(ModulesTest, Unflatten) {
  // 创建 Unflatten 模块的实例，指定非命名张量的选项
  Unflatten unflatten(UnflattenOptions(0, {2, 2}));
  // 使用 Unflatten 模块进行前向传播计算
  auto output = unflatten->forward(torch::tensor({1, 2, 3, 4}));
  // 创建预期输出张量
  auto expected = torch::tensor({{1, 2}, {3, 4}});
  // 断言输出张量与预期张量相等
  ASSERT_TRUE(torch::equal(output, expected));

  // 创建一个函数，用于生成维度名
  auto make_dimnames = [](std::vector<std::string> names) {
    std::vector<torch::Dimname> dimnames;
    // 遍历每个名称并创建对应的维度名对象
    for (auto name : names) {
      dimnames.push_back(
          torch::Dimname::fromSymbol(torch::Symbol::dimname(name)));
    }
    return dimnames;
  };

  // 创建 Unflatten 模块的实例，指定命名张量的选项
  unflatten = Unflatten(UnflattenOptions(
      "B",
      {std::pair<std::string, int64_t>{"B1", 2},
       std::pair<std::string, int64_t>{"B2", 2}}));
  // 使用 Unflatten 模块进行前向传播计算，并为输入张量设置命名维度
  output = unflatten->forward(
      torch::tensor({{1, 2, 3, 4}}).refine_names(make_dimnames({"A", "B"})));
  // 创建预期输出张量，并设置命名维度
  expected = torch::tensor({{{1, 2}, {3, 4}}})
                 .refine_names(make_dimnames({"A", "B1", "B2"}));
  // 断言输出张量与预期张量相等
  ASSERT_TRUE(torch::equal(output, expected));
}
TEST_F(ModulesTest, AdaptiveMaxPool1d) {
  // 创建一个 AdaptiveMaxPool1d 模型对象，指定池化窗口大小为 3
  AdaptiveMaxPool1d model(3);
  // 创建一个输入张量 x，包含单个序列的浮点数值，要求计算梯度
  auto x = torch::tensor(
      {{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 将输入张量 x 传递给模型，生成输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和
  torch::Tensor s = y.sum();

  // 对张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 与指定的浮点数张量在误差允许范围内相等
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  // 断言张量 s 的维度为 0，即标量
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的尺寸符合指定的向量
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool1dReturnIndices) {
  // 创建一个 AdaptiveMaxPool1d 模型对象，指定池化窗口大小为 3
  AdaptiveMaxPool1d model(3);
  // 创建一个输入张量 x，包含单个序列的浮点数值，要求计算梯度
  auto x = torch::tensor(
      {{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 声明输出张量 y 和索引张量 indices，将模型返回的结果分别赋值给它们
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  // 断言输出张量 y 的维度为 3
  ASSERT_EQ(y.dim(), 3);
  // 断言输出张量 y 与指定的浮点数张量在误差允许范围内相等
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  // 断言输出张量 y 的尺寸符合指定的向量
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
  // 断言索引张量 indices 与指定的长整型张量在误差允许范围内相等
  ASSERT_TRUE(
      torch::allclose(indices, torch::tensor({{{1, 3, 4}}}, torch::kLong)));
  // 断言索引张量 indices 的尺寸符合指定的向量
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dEven) {
  // 创建一个 AdaptiveMaxPool2d 模型对象，指定池化窗口大小为 3
  AdaptiveMaxPool2d model(3);
  // 创建一个张量 x，包含从 0 到 49 的连续数值，并将其重塑为指定尺寸，要求计算梯度
  auto x = torch::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  // 将输入张量 x 传递给模型，生成输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和
  torch::Tensor s = y.sum();

  // 对张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 与指定的浮点数张量在误差允许范围内相等
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}},
          },
          torch::kFloat)));
  // 断言张量 s 的维度为 0，即标量
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的尺寸符合指定的向量
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dUneven) {
  // 创建一个 AdaptiveMaxPool2d 模型对象，指定不对称的池化窗口大小
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  // 创建一个张量 x，包含从 0 到 39 的连续数值，并将其重塑为指定尺寸，要求计算梯度
  auto x = torch::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  // 将输入张量 x 传递给模型，生成输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和
  torch::Tensor s = y.sum();

  // 对张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 与指定的浮点数张量在误差允许范围内相等
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{25, 27}, {33, 35}, {37, 39}},
          },
          torch::kFloat)));
  // 断言张量 s 的维度为 0，即标量
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的尺寸符合指定的向量
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}
TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesEven) {
  // 创建 AdaptiveMaxPool2d 模型对象，池化尺寸为 3
  AdaptiveMaxPool2d model(3);
  // 创建一个张量 x，包含从 0 到 49 的数字
  auto x = torch::arange(0., 50);
  // 将张量 x 重新调整为大小为 {2, 5, 5} 的三维张量，并设置需要梯度计算
  x.resize_({2, 5, 5}).set_requires_grad(true);
  // 定义张量 y 和 indices，通过 forward_with_indices 方法获取模型的输出
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);
  // 计算张量 y 的元素和
  torch::Tensor s = y.sum();

  // 对元素和张量 s 执行反向传播
  s.backward();
  // 断言 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言张量 y 是否与给定的浮点数张量近似相等
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}},
          },
          torch::kFloat)));
  // 断言张量 y 的尺寸为 {2, 3, 3}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));

  // 断言张量 indices 的维度为 3
  ASSERT_EQ(indices.ndimension(), 3);
  // 断言张量 indices 是否与给定的长整型张量近似相等
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
          },
          torch::kLong)));
  // 断言张量 indices 的尺寸为 {2, 3, 3}
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesUneven) {
  // 创建 AdaptiveMaxPool2d 模型对象，池化尺寸为 {3, 2}
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  // 创建一个张量 x，包含从 0 到 39 的数字
  auto x = torch::arange(0., 40);
  // 将张量 x 重新调整为大小为 {2, 5, 4} 的三维张量，并设置需要梯度计算
  x.resize_({2, 5, 4}).set_requires_grad(true);
  // 定义张量 y 和 indices，通过 forward_with_indices 方法获取模型的输出
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);
  // 计算张量 y 的元素和
  torch::Tensor s = y.sum();

  // 对元素和张量 s 执行反向传播
  s.backward();
  // 断言 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言张量 y 是否与给定的浮点数张量近似相等
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{25, 27}, {33, 35}, {37, 39}},
          },
          torch::kFloat)));
  // 断言张量 y 的尺寸为 {2, 3, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));

  // 断言张量 indices 的维度为 3
  ASSERT_EQ(indices.ndimension(), 3);
  // 断言张量 indices 是否与给定的长整型张量近似相等
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{5, 7}, {13, 15}, {17, 19}},
          },
          torch::kLong)));
  // 断言张量 indices 的尺寸为 {2, 3, 2}
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveMaxPool3d) {
  // 创建 AdaptiveMaxPool3d 模型对象，池化尺寸为 3
  AdaptiveMaxPool3d model(3);
  // 创建一个张量 x，包含从 0 到 63 的数字
  auto x = torch::arange(0., 64);
  // 将张量 x 重新调整为大小为 {1, 4, 4, 4} 的四维张量，并设置需要梯度计算
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  // 使用模型对张量 x 进行前向传播
  auto y = model(x);
  // 计算张量 y 的元素和
  torch::Tensor s = y.sum();

  // 对元素和张量 s 执行反向传播
  s.backward();
  // 断言 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言张量 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言张量 y 是否与给定的浮点数张量近似相等
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          torch::kFloat)));
  // 断言张量 y 的尺寸为 {1, 3, 3, 3}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}
// 定义测试用例 `AdaptiveMaxPool3dReturnIndices`，测试三维自适应最大池化
TEST_F(ModulesTest, AdaptiveMaxPool3dReturnIndices) {
  // 创建一个三维自适应最大池化模型，指定池化尺寸为3
  AdaptiveMaxPool3d model(3);
  // 创建一个张量 x，包含从0到63的数字，将其形状调整为 {1, 4, 4, 4}，并设置为需要梯度计算
  auto x = torch::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  // 声明张量 y 和 indices，通过模型的 forward_with_indices 方法计算前向结果
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);
  // 计算张量 y 的所有元素之和
  torch::Tensor s = y.sum();

  // 执行反向传播
  s.backward();
  // 断言张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言张量 y 的维度为4，并验证其值接近给定的浮点数张量
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          torch::kFloat)));
  // 断言张量 y 的尺寸为 {1, 3, 3, 3}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));

  // 断言张量 indices 的维度为4，并验证其值接近给定的长整型张量
  ASSERT_EQ(indices.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          torch::kLong)));
  // 断言张量 indices 的尺寸为 {1, 3, 3, 3}
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

// 定义测试用例 `AdaptiveAvgPool1d`，测试一维自适应平均池化
TEST_F(ModulesTest, AdaptiveAvgPool1d) {
  // 创建一个一维自适应平均池化模型，指定池化尺寸为3
  AdaptiveAvgPool1d model(3);
  // 创建一个形状为 {1, 1, 5} 的浮点型张量 x，设置为需要梯度计算
  auto x = torch::tensor(
      {{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 使用模型对张量 x 进行池化操作得到张量 y
  auto y = model(x);
  // 计算张量 y 的所有元素之和
  torch::Tensor s = y.sum();

  // 执行反向传播
  s.backward();
  // 断言张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言张量 y 的维度为3，并验证其值接近给定的浮点数张量
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{1.5, 3.0, 4.5}}}, torch::kFloat)));
  // 断言张量 y 的尺寸为 {1, 1, 3}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

// 定义测试用例 `AdaptiveAvgPool2dEven`，测试二维自适应平均池化（偶数尺寸）
TEST_F(ModulesTest, AdaptiveAvgPool2dEven) {
  // 创建一个二维自适应平均池化模型，指定池化尺寸为3
  AdaptiveAvgPool2d model(3);
  // 创建一个形状为 {2, 5, 5} 的张量 x，包含从0到49的数字，设置为需要梯度计算
  auto x = torch::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  // 使用模型对张量 x 进行池化操作得到张量 y
  auto y = model(x);
  // 计算张量 y 的所有元素之和
  torch::Tensor s = y.sum();

  // 执行反向传播
  s.backward();
  // 断言张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言张量 y 的维度为3，并验证其值接近给定的浮点数张量
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{3.0, 4.5, 6.0}, {10.5, 12.0, 13.5}, {18.0, 19.5, 21.0}},
              {{28.0, 29.5, 31.0}, {35.5, 37.0, 38.5}, {43.0, 44.5, 46.0}},
          },
          torch::kFloat)));
  // 断言张量 y 的尺寸为 {2, 3, 3}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

// 定义测试用例 `AdaptiveAvgPool2dUneven`，测试二维自适应平均池化（不等尺寸）
TEST_F(ModulesTest, AdaptiveAvgPool2dUneven) {
  // 创建一个二维自适应平均池化模型，指定池化尺寸为 {3, 2}
  AdaptiveAvgPool2d model(AdaptiveAvgPool2dOptions({3, 2}));
  // 创建一个形状为 {2, 5, 4} 的张量 x，包含从0到39的数字，设置为需要梯度计算
  auto x = torch::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  // 使用模型对张量 x 进行池化操作得到张量 y
  auto y = model(x);
  // 计算张量 y 的所有元素之和
  torch::Tensor s = y.sum();

  // 执行反向传播
  s.backward();
  // 断言张量 s 的维度为0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言张量 y 的维度为3，并验证其值接近给定的浮点数张量
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{2.5, 4.5}, {8.5, 10.5}, {14.5, 16.5}},
              {{22.5, 24.5}, {28.5, 30.5}, {34.5, 36.5}},
          },
          torch::kFloat)));
  // 断言张量 y 的尺寸为 {2, 3, 2}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}
TEST_F(ModulesTest, AdaptiveAvgPool3d) {
  // 创建一个 AdaptiveAvgPool3d 模型实例，指定输出大小为 3
  AdaptiveAvgPool3d model(3);
  
  // 创建一个张量 x，包含从 0 到 63 的数，并将其形状调整为 {1, 4, 4, 4}，并设置为需要梯度计算
  auto x = torch::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  
  // 将输入张量 x 输入到模型中，得到输出张量 y
  auto y = model(x);
  
  // 计算输出张量 y 的所有元素的和
  torch::Tensor s = y.sum();
  
  // 对和 s 进行反向传播
  s.backward();
  
  // 断言 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言输出张量 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  
  // 断言输出张量 y 与给定的张量在指定的精度下近似相等
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{10.5, 11.5, 12.5}, {14.5, 15.5, 16.5}, {18.5, 19.5, 20.5}},
              {{26.5, 27.5, 28.5}, {30.5, 31.5, 32.5}, {34.5, 35.5, 36.5}},
              {{42.5, 43.5, 44.5}, {46.5, 47.5, 48.5}, {50.5, 51.5, 52.5}},
          },
          torch::kFloat)));
  
  // 断言输出张量 y 的大小与给定的向量大小相同
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, MaxUnpool1d) {
  // 创建一个包含索引的张量 indices
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  
  // 创建一个包含数值的张量 x，并设置其需要梯度计算
  auto x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  
  // 创建一个 MaxUnpool1d 模型实例
  auto model = MaxUnpool1d{3};
  
  // 将输入张量 x 和索引张量 indices 输入到模型中，得到输出张量 y
  auto y = model->forward(x, indices);

  // 断言输出张量 y 的维度为 3
  ASSERT_EQ(y.dim(), 3);
  
  // 断言输出张量 y 与给定的张量在指定的精度下近似相等
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  
  // 断言输出张量 y 的大小与给定的向量大小相同
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  // 更新索引张量和输入张量 x 的值
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  
  // 创建一个带有额外选项的 MaxUnpool1d 模型实例
  model = MaxUnpool1d{MaxUnpool1dOptions(3).stride(2).padding(1)};
  
  // 将更新后的输入张量 x、索引张量 indices 和大小向量输入到模型中，得到输出张量 y
  y = model->forward(x, indices, std::vector<int64_t>({1, 1, 5}));

  // 断言输出张量 y 的维度为 3
  ASSERT_EQ(y.dim(), 3);
  
  // 断言输出张量 y 与给定的张量在指定的精度下近似相等
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  
  // 断言输出张量 y 的大小与给定的向量大小相同
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 5}));
}

TEST_F(ModulesTest, MaxPool1d_MaxUnpool1d) {
  // 创建一个最大池化层 MaxPool1d 实例，指定选项为步长为 2
  MaxPool1d pool{MaxPool1dOptions(2).stride(2)};
  
  // 创建一个最大解池化层 MaxUnpool1d 实例，指定选项为步长为 2
  MaxUnpool1d unpool{MaxUnpool1dOptions(2).stride(2)};
  
  // 创建一个输入张量 input
  auto input = torch::tensor({{{1, 2, 3, 4, 5, 6, 7, 8}}}, torch::kFloat);
  
  // 声明输出张量和索引张量
  torch::Tensor output, indices;
  
  // 对输入张量 input 应用最大池化层，同时获取索引张量 indices
  std::tie(output, indices) = pool->forward_with_indices(input);
  
  // 断言最大解池化层对输出张量 output 和索引张量 indices 的处理结果与给定的张量在指定的精度下近似相等
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices),
      torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}}, torch::kFloat)));

  // 创建一个输入张量 input，展示 output_size 的使用示例
  input = torch::tensor({{{1, 2, 3, 4, 5, 6, 7, 8, 9}}}, torch::kFloat);
  
  // 再次对输入张量 input 应用最大池化层，同时获取索引张量 indices
  std::tie(output, indices) = pool->forward_with_indices(input);
  
  // 断言最大解池化层对输出张量 output、索引张量 indices 和大小向量的处理结果与给定的张量在指定的精度下近似相等
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices, input.sizes().vec()),
      torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8, 0}}}, torch::kFloat)));
  
  // 断言最大解池化层对输出张量 output 和索引张量 indices 的处理结果与给定的张量在指定的精度下近似相等
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices),
      torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}}, torch::kFloat)));
}
# 在 ModulesTest 测试框架中定义 MaxUnpool2d 测试用例
TEST_F(ModulesTest, MaxUnpool2d) {
  # 创建索引张量，指定每个池化区域内最大值的索引
  auto indices = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}},
      torch::kLong);
  # 创建输入张量 x，指定池化前的张量数据，需要梯度计算
  auto x = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  # 创建 MaxUnpool2d 模块对象，指定池化窗口大小为 3，步幅为 2，填充为 1
  auto model = MaxUnpool2d{MaxUnpool2dOptions(3).stride(2).padding(1)};
  # 对模型进行前向传播，得到输出张量 y
  auto y = model->forward(x, indices);

  # 断言输出张量 y 的维度为 4
  ASSERT_EQ(y.dim(), 4);
  # 断言输出张量 y 与预期张量在数值上相近
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
  # 断言输出张量 y 的尺寸为 [2, 1, 5, 5]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 1, 5, 5}));
}

# 在 ModulesTest 测试框架中定义 MaxPool2d_MaxUnpool2d 测试用例
TEST_F(ModulesTest, MaxPool2d_MaxUnpool2d) {
  # 创建 MaxPool2d 模块对象，指定池化窗口大小为 2，步幅为 2
  MaxPool2d pool{MaxPool2dOptions(2).stride(2)};
  # 创建 MaxUnpool2d 模块对象，指定池化窗口大小为 2，步幅为 2
  MaxUnpool2d unpool{MaxUnpool2dOptions(2).stride(2)};
  # 创建输入张量 input，包含池化操作的数据
  auto input = torch::tensor(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}},
      torch::kFloat);
  # 声明输出张量 output 和索引张量 indices
  torch::Tensor output, indices;
  # 对输入张量进行池化操作，并获取池化后的输出张量和索引
  std::tie(output, indices) = pool->forward_with_indices(input);
  # 断言使用未指定输出尺寸的情况下，反池化后的张量与预期张量在数值上相近
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices),
      torch::tensor(
          {{{{0, 0, 0, 0}, {0, 6, 0, 8}, {0, 0, 0, 0}, {0, 14, 0, 16}}}},
          torch::kFloat)));

  # 断言使用指定输出尺寸的情况下，反池化后的张量与预期张量在数值上相近
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices, std::vector<int64_t>{1, 1, 5, 5}),
      torch::tensor(
          {{{{0, 0, 0, 0, 0},
             {6, 0, 8, 0, 0},
             {0, 0, 0, 14, 0},
             {16, 0, 0, 0, 0},
             {0, 0, 0, 0, 0}}}},
          torch::kFloat)));
}

# 在 ModulesTest 测试框架中定义 MaxUnpool3d 测试用例
TEST_F(ModulesTest, MaxUnpool3d) {
  # 创建索引张量，指定每个池化区域内最大值的索引
  auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
  # 创建输入张量 x，指定池化前的张量数据，需要梯度计算
  auto x = torch::tensor(
      {{{{{26}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  # 创建 MaxUnpool3d 模块对象，指定池化窗口大小为 3
  auto model = MaxUnpool3d{3};
  # 对模型进行前向传播，得到输出张量 y
  auto y = model->forward(x, indices);

  # 断言输出张量 y 的维度为 5
  ASSERT_EQ(y.dim(), 5);
  # 断言输出张量 y 与预期张量在数值上相近
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}}},
          torch::kFloat)));
  # 断言输出张量 y 的尺寸为 [1, 1, 3, 3, 3]
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3, 3, 3}));
}
TEST_F(ModulesTest, MaxUnpool3dOutputSize) {
  // 创建索引张量，用于最大化反池化操作
  auto indices = torch::tensor(
      {{{{{21, 23}, {29, 31}}, {{53, 55}, {61, 63}}}}}, torch::kLong);
  // 创建输入张量 x，指定数据类型为浮点数，并启用梯度跟踪
  auto x = torch::tensor(
      {{{{{21, 23}, {29, 31}}, {{53, 55}, {61, 63}}}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 创建 MaxUnpool3d 模型对象，设置池化窗口大小、步长和填充
  auto model = MaxUnpool3d{MaxUnpool3dOptions(3).stride(2).padding(1)};
  // 对模型进行前向传播，传入输入张量 x、索引张量 indices 和输出尺寸
  auto y = model->forward(x, indices, std::vector<int64_t>({1, 1, 4, 4, 4}));

  // 断言输出张量 y 的维度为 5
  ASSERT_EQ(y.dim(), 5);
  // 断言 y 是否与预期张量在数值上相似
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
             {{0, 0, 0, 0}, {0, 21, 0, 23}, {0, 0, 0, 0}, {0, 29, 0, 31}},
             {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
             {{0, 0, 0, 0}, {0, 53, 0, 55}, {0, 0, 0, 0}, {0, 61, 0, 63}}}}},
          torch::kFloat)));
  // 断言 y 的尺寸是否为 {1, 1, 4, 4, 4}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 4, 4, 4}));
}

TEST_F(ModulesTest, MaxPool3d_MaxUnpool3d) {
  // 创建 MaxPool3d 模型对象，设置池化窗口大小和步长
  MaxPool3d pool{MaxPool3dOptions(3).stride(2)};
  // 创建 MaxUnpool3d 模型对象，设置池化窗口大小和步长
  MaxUnpool3d unpool{MaxUnpool3dOptions(3).stride(2)};
  // 创建随机输入张量 input
  auto input = torch::randn({20, 16, 51, 33, 15});
  // 声明输出张量 output 和索引张量 indices
  torch::Tensor output, indices;
  // 对输入张量进行池化操作，同时获取池化后的输出张量和索引张量
  std::tie(output, indices) = pool->forward_with_indices(input);
  // 对池化后的输出张量进行最大化反池化操作
  auto unpooled_output = unpool(output, indices);
  // 断言反池化后的输出张量尺寸是否符合预期
  ASSERT_EQ(
      unpooled_output.sizes(), std::vector<int64_t>({20, 16, 51, 33, 15}));
}

TEST_F(ModulesTest, Linear) {
  {
    // 创建线性层模型，输入维度为 5，输出维度为 2
    Linear model(5, 2);
    // 创建随机输入张量 x，启用梯度跟踪
    auto x = torch::randn({10, 5}, torch::requires_grad());
    // 对输入张量进行模型前向传播，计算输出张量 y
    auto y = model(x);
    // 计算张量 y 的元素和
    torch::Tensor s = y.sum();

    // 对元素和张量 s 进行反向传播
    s.backward();
    // 断言输出张量 y 的维度为 2
    ASSERT_EQ(y.ndimension(), 2);
    // 断言张量 s 的维度为 0
    ASSERT_EQ(s.ndimension(), 0);
    // 断言输出张量 y 的第一维尺寸为 10
    ASSERT_EQ(y.size(0), 10);
    // 断言输出张量 y 的第二维尺寸为 2
    ASSERT_EQ(y.size(1), 2);

    // 断言模型权重的梯度张量尺寸是否正确
    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    // 计算预期的输出张量 y_exp，并与 y 进行数值比较
    auto y_exp = torch::addmm(model->bias, x, model->weight.t());
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
  {
    // 创建线性层模型，输入维度为 5，输出维度为 2，且不使用偏置
    Linear model(LinearOptions(5, 2).bias(false));
    // 创建随机输入张量 x，启用梯度跟踪
    auto x = torch::randn({10, 5}, torch::requires_grad());
    // 对输入张量进行模型前向传播，计算输出张量 y
    auto y = model(x);
    // 计算张量 y 的元素和
    torch::Tensor s = y.sum();

    // 对元素和张量 s 进行反向传播
    s.backward();
    // 断言输出张量 y 的维度为 2
    ASSERT_EQ(y.ndimension(), 2);
    // 断言张量 s 的维度为 0
    ASSERT_EQ(s.ndimension(), 0);
    // 断言输出张量 y 的第一维尺寸为 10
    ASSERT_EQ(y.size(0), 10);
    // 断言输出张量 y 的第二维尺寸为 2
    ASSERT_EQ(y.size(1), 2);

    // 断言模型权重的梯度张量尺寸是否正确
    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    // 计算预期的输出张量 y_exp，并与 y 进行数值比较
    auto y_exp = torch::mm(x, model->weight.t());
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, LocalResponseNorm) {
  {
    // 创建局部响应归一化模型对象，设置窗口大小为 2
    LocalResponseNorm model(LocalResponseNormOptions(2));
    // 创建固定范围的输入张量 x
    const auto x =
        torch::arange(100., 136, torch::requires_grad()).reshape({2, 3, 3, 2});
    // 对输入张量进行模型前向传播，计算输出张量 y
    auto y = model(x);
    // 开始测试局部响应归一化模块
    // 创建一个四维张量 y_exp，存储预期的张量数值
    const auto y_exp = torch::tensor(
        {{{{73.7788, 74.1462}, {74.5031, 74.8572}, {75.2010, 75.5420}},
          {{61.6057, 61.7227}, {61.8347, 61.9418}, {62.0441, 62.1418}},
          {{62.2349, 62.3235}, {62.4077, 62.4877}, {62.5635, 62.6353}}},
         {{{79.3915, 79.6491}, {79.8978, 80.1446}, {80.3827, 80.6190}},
          {{63.0317, 63.0742}, {63.1135, 63.1496}, {63.1826, 63.2126}},
          {{63.2396, 63.2637}, {63.2850, 63.3036}, {63.3195, 63.3328}}}},
        torch::kFloat);
    // 计算张量 y 的所有元素的和
    torch::Tensor s = y.sum();
    // 计算张量 s 的梯度
    s.backward();
    // 断言张量 y 的维度为 4
    ASSERT_EQ(y.ndimension(), 4);
    // 断言张量 s 的维度为 0，即标量
    ASSERT_EQ(s.ndimension(), 0);
    // 断言张量 y 的形状与另一张量 x 的形状相同
    ASSERT_EQ(y.sizes(), x.sizes());
    // 使用数值容差 1e-4 和绝对容差 1e-7 检查张量 y 是否与 y_exp 接近
    ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}
TEST_F(ModulesTest, Unfold) {
  {
    // 创建 Unfold 模块对象，配置为使用 {2, 2} 的窗口大小，1 的填充和 2 的步长
    Unfold model(UnfoldOptions({2, 2}).padding(1).stride(2));
    // 创建一个输入张量，包含从 2 到 13 的数字，并按照指定维度进行视图重塑
    auto input =
        torch::arange(2., 14, torch::requires_grad()).view({1, 2, 2, 3});
    // 使用 Unfold 模块处理输入张量
    auto output = model(input);
    // 期望输出的张量，包含预定义的数值和数据类型
    auto expected = torch::tensor(
        {{{0.0, 0.0, 0.0, 6.0},
          {0.0, 0.0, 5.0, 7.0},
          {0.0, 3.0, 0.0, 0.0},
          {2.0, 4.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 12.0},
          {0.0, 0.0, 11.0, 13.0},
          {0.0, 9.0, 0.0, 0.0},
          {8.0, 10.0, 0.0, 0.0}}},
        torch::kFloat);
    // 计算输出张量的所有元素之和
    auto s = output.sum();
    // 执行反向传播
    s.backward();

    // 断言输出张量的维度为 0
    ASSERT_EQ(s.ndimension(), 0);
    // 断言输出张量的尺寸符合给定的预期向量
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 8, 4}));
    // 断言输出张量与预期张量在数值上全部接近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 输入张量维度错误
    // 创建 Fold 模块对象，配置为使用 {8, 8} 的输出大小和 {3, 3} 的窗口大小
    Fold model(FoldOptions({8, 8}, {3, 3}));
    // 断言使用 4 维输入张量时会抛出异常
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 3, 16, 16})),
        "Input Error: Only unbatched (2D) or batched (3D) input Tensors are supported (got 4D)");
  }
}
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // input wrong dimension
    // 创建 Unfold 模型对象，设置输入选项为 {2, 4}
    Unfold model(UnfoldOptions({2, 4}));
    // 断言模型调用时，使用了 3D 的输入张量，应抛出异常信息
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 5, 2})),
        "Input Error: Only 4D input Tensors are supported (got 3D)");
  }
  {
    // calculated output shape is too small
    // 创建 Unfold 模型对象，设置输入选项为 {2, 3}
    Unfold model(UnfoldOptions({2, 3}));
    // 断言模型调用时，使用了空间尺寸为 (2, 2)，核大小为 (2, 3)，膨胀为 (1, 1)，填充为 (0, 0) 的输入张量，
    // 预期抛出异常，指出计算出的滑动块数组形状 (1, 0) 太小，其每个组件至少必须为一
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 2, 2, 2})),
        "Given input with spatial size (2, 2), kernel_size=(2, 3), "
        "dilation=(1, 1), padding=(0, 0), calculated shape of the array of "
        "sliding blocks as (1, 0), but its components must be at least one.");
  }
}

// 在 ModulesTest 测试套件中，测试 SimpleContainer 类
TEST_F(ModulesTest, SimpleContainer) {
  // 创建 SimpleContainer 类的共享指针
  auto model = std::make_shared<SimpleContainer>();
  // 向模型添加线性层 l1，输入大小为 10，输出大小为 3
  auto l1 = model->add(Linear(10, 3), "l1");
  // 向模型添加线性层 l2，输入大小为 3，输出大小为 5
  auto l2 = model->add(Linear(3, 5), "l2");
  // 向模型添加线性层 l3，输入大小为 5，输出大小为 100
  auto l3 = model->add(Linear(5, 100), "l3");

  // 创建大小为 [1000, 10] 的随机张量 x，并设置需要梯度
  auto x = torch::randn({1000, 10}, torch::requires_grad());
  // 使用 l1 层进行前向传播，并进行非负数截断
  x = l1(x).clamp_min(0);
  // 使用 l2 层进行前向传播，并进行非负数截断
  x = l2(x).clamp_min(0);
  // 使用 l3 层进行前向传播，并进行非负数截断
  x = l3(x).clamp_min(0);

  // 对 x 进行反向传播，梯度值设置为全部为 1 的张量
  x.backward(torch::ones_like(x));
  // 断言 x 的维度为 2
  ASSERT_EQ(x.ndimension(), 2);
  // 断言 x 的第一个维度大小为 1000
  ASSERT_EQ(x.size(0), 1000);
  // 断言 x 的第二个维度大小为 100
  ASSERT_EQ(x.size(1), 100);
  // 断言 x 的最小值为 0
  ASSERT_EQ(x.min().item<float>(), 0);
}

// 在 ModulesTest 测试套件中，测试 Embedding 类的基本用法
TEST_F(ModulesTest, EmbeddingBasic) {
  // 定义字典大小
  const int64_t dict_size = 10;
  // 创建 Embedding 类实例，字典大小为 dict_size，向量维度为 2
  Embedding model(dict_size, 2);
  // 断言模型中包含名称为 "weight" 的参数
  ASSERT_TRUE(model->named_parameters().contains("weight"));
  // 断言模型权重的维度为 2
  ASSERT_EQ(model->weight.ndimension(), 2);
  // 断言模型权重的第一个维度大小为 dict_size
  ASSERT_EQ(model->weight.size(0), dict_size);
  // 断言模型权重的第二个维度大小为 2
  ASSERT_EQ(model->weight.size(1), 2);

  // 创建大小为 [10] 的张量 x，填充为 dict_size - 1，数据类型为 int64
  auto x = torch::full({10}, dict_size - 1, torch::kInt64);
  // 对模型进行前向传播，输入张量 x
  auto y = model(x);
  // 计算输出张量 y 的总和
  torch::Tensor s = y.sum();

  // 对总和张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为 2
  ASSERT_EQ(y.ndimension(), 2);
  // 断言总和张量 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的第一个维度大小为 10
  ASSERT_EQ(y.size(0), 10);
  // 断言输出张量 y 的第二个维度大小为 2
  ASSERT_EQ(y.size(1), 2);

  // 断言模型权重梯度的元素数量为 2 * dict_size
  ASSERT_EQ(model->weight.grad().numel(), 2 * dict_size);
}

// 在 ModulesTest 测试套件中，测试 Embedding 类的多维输入
TEST_F(ModulesTest, EmbeddingList) {
  // 创建 Embedding 类实例，字典大小为 6，向量维度为 4
  Embedding model(6, 4);
  // 创建大小为 [2, 3] 的张量 x，填充为 5，数据类型为 int64
  auto x = torch::full({2, 3}, 5, torch::kInt64);
  // 对模型进行前向传播，输入张量 x
  auto y = model(x);
  // 计算输出张量 y 的总和
  torch::Tensor s = y.sum();

  // 对总和张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言输出张量 y 的第一个维度大小为 2
  ASSERT_EQ(y.size(0), 2);
  // 断言输出张量 y 的第二个维度大小为 3
  ASSERT_EQ(y.size(1), 3);
  // 断言输出张量 y 的第三个维度大小为 4
  ASSERT_EQ(y.size(2), 4);
}

// 在 ModulesTest 测试套件中，测试 Embedding 类从预训练权重加载
TEST_F(ModulesTest, EmbeddingFromPretrained) {
  // 定义预训练权重
  auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  // 从预训练权重创建 Embedding 类实例
  Embedding embedding = torch::nn::Embedding::from_pretrained(weight);
  // 创建大小为 [1] 的输入张量 input，数据类型为 int64
  auto input = torch::tensor({1}, torch::kLong);
  // 断言 embedding 对输入 input 的输出近似于给定的张量
  ASSERT_TRUE(torch::allclose(
      embedding(input), torch::tensor({4.0000, 5.1000, 6.3000})));
}

// 在 ModulesTest 测试套件中，测试 EmbeddingBag 类从预训练权重加载
TEST_F(ModulesTest, EmbeddingBagFromPretrained) {
  // 定义预训练权重
  auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  // 从预训练权重创建 EmbeddingBag 类实例
  EmbeddingBag embeddingbag = torch::nn::EmbeddingBag::from_pretrained(weight);
  // 创建大小为 [1, 2] 的零填充输入张量 input，数据类型为 int64
  auto input = torch::zeros({{1, 2}}, torch::kLong);
  // 修改输入张量 input 的第一行内容为 {1, 0}
  input[0] = torch::tensor({1, 0});
  // 断言 embeddingbag 对输入 input 的输出近似于给定的张量
  ASSERT_TRUE(torch::allclose(
      embeddingbag(input), torch::tensor({2.5000, 3.7000, 4.6500})));
}

// 在 ModulesTest 测试套件中，测试 AlphaDropout 类
TEST_F(ModulesTest, AlphaDropout) {
  // 创建 AlphaDropout 类实例，dropout 概率为 0.5
  AlphaDropout alpha_dropout(0.5);
  // 创建全为 1 的大小为 100 的张量 x，需要梯度
  torch::Tensor x = torch::ones(100, torch::requires_grad());
  // 对张量 x 进行 alpha dropout
  torch::Tensor y = alpha_dropout(x);

  // 对 y 进行反向传播，梯度设置为全部为 1 的张量
  y.backward(torch::ones_like(y));

  // 断言 y 的维度为 1
  ASSERT_EQ(y.ndimension(), 1);
  // 断言 y 的大小为 100
  ASSERT_EQ(y.size(0), 100);
  // 断言 y 的总和小于 130
  ASSERT_LT(y.sum().item<float>(), 130); // 可能的情况
  // 断言 y 的总和大于 40
  ASSERT_GT(y.sum().item<float>(), 40); // 可能的情况

  // 将 alpha_dropout 设置为评估模式
TEST_F(ModulesTest, FeatureAlphaDropout) {
  // 创建一个 FeatureAlphaDropout 对象，设置概率为 0.5
  FeatureAlphaDropout feature_alpha_dropout(0.5);
  // 创建一个大小为 10x10 的张量 x，所有元素初始化为 1，并且需要计算梯度
  torch::Tensor x = torch::ones({10, 10}, torch::requires_grad());
  // 对张量 x 应用 Alpha Dropout，得到输出张量 y
  torch::Tensor y = feature_alpha_dropout(x);

  // 对 y 进行反向传播，使用全为 1 的张量作为梯度
  y.backward(torch::ones_like(y));

  // 断言：y 的维度应为 2
  ASSERT_EQ(y.ndimension(), 2);
  // 断言：y 的第一个维度大小为 10
  ASSERT_EQ(y.size(0), 10);
  // 断言：y 的第二个维度大小为 10
  ASSERT_EQ(y.size(1), 10);
  // 断言：y 所有元素之和应小于 130，可能性存在
  ASSERT_LT(y.sum().item<float>(), 130); // Probably
  // 断言：y 所有元素之和应大于 40，可能性存在
  ASSERT_GT(y.sum().item<float>(), 40); // Probably

  // 将 feature_alpha_dropout 设置为评估模式
  feature_alpha_dropout->eval();
  // 再次对 x 应用 Alpha Dropout，得到输出张量 y
  y = feature_alpha_dropout(x);

  // 断言：y 所有元素之和应为 100
  ASSERT_EQ(y.sum().item<float>(), 100);
}

TEST_F(ModulesTest, Dropout) {
  // 遍历 inplace 参数为 false 和 true 的情况
  for (const auto inplace : {false, true}) {
    // 创建 Dropout 对象，设置概率为 0.5，并根据 inplace 参数设置选项
    Dropout dropout(DropoutOptions(0.5).inplace(inplace));
    // 创建大小为 100 的张量 x，所有元素初始化为 1
    torch::Tensor x = torch::ones(100);
    // 如果 inplace 为 false，需要设置 x 为需要计算梯度
    if (!inplace) {
      x.requires_grad_(true);
    }
    // 对张量 x 应用 Dropout，得到输出张量 y
    torch::Tensor y = dropout(x);

    // 断言：y 的维度应为 1
    ASSERT_EQ(y.ndimension(), 1);
    // 断言：y 的大小应为 100
    ASSERT_EQ(y.size(0), 100);
    // 断言：y 所有元素之和应小于 130，可能性存在
    ASSERT_LT(y.sum().item<float>(), 130); // Probably
    // 断言：y 所有元素之和应大于 70，可能性存在
    ASSERT_GT(y.sum().item<float>(), 70); // Probably
    // 如果 inplace 为 true，断言 y 应与 x 全部元素相等
    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      // 否则，对 y 进行反向传播，使用全为 1 的张量作为梯度
      y.backward(torch::ones_like(y));
    }

    // 将 dropout 设置为评估模式
    dropout->eval();
    // 再次对大小为 100 的张量应用 Dropout，得到输出张量 y
    y = dropout(torch::ones(100));
    // 断言：y 所有元素之和应为 100
    ASSERT_EQ(y.sum().item<float>(), 100);
  }
}

TEST_F(ModulesTest, Dropout2d) {
  // 设置概率 p 为 0.5
  auto p = 0.5;
  // 遍历 inplace 参数为 false 和 true 的情况
  for (const auto inplace : {false, true}) {
    // 创建 Dropout2d 对象，设置概率为 p，并根据 inplace 参数设置选项
    Dropout2d dropout(Dropout2dOptions(p).inplace(inplace));
    // 创建大小为 50x50x2x2 的空张量 x，并填充元素为 1-p
    torch::Tensor x = torch::empty({50, 50, 2, 2}).fill_(1 - p);
    // 如果 inplace 为 false，需要设置 x 为需要计算梯度
    if (!inplace) {
      x.requires_grad_(true);
    }
    // 对张量 x 应用 Dropout2d，得到输出张量 y
    torch::Tensor y = dropout(x);

    // 断言：y 的维度应为 4
    ASSERT_EQ(y.ndimension(), 4);
    // 断言：y 的第一个维度大小应为 50
    ASSERT_EQ(y.size(0), 50);
    // 断言：y 的第二个维度大小应为 50
    ASSERT_EQ(y.size(1), 50);
    // 断言：y 的第三个维度大小应为 2
    ASSERT_EQ(y.size(2), 2);
    // 断言：y 的第四个维度大小应为 2
    ASSERT_EQ(y.size(3), 2);
    // 断言：y 的平均值与 (1 - p) 的差的绝对值应小于 0.05
    ASSERT_LT((y.mean() - (1 - p)).abs().item<float>(), 0.05);

    // 如果 inplace 为 true，断言 y 应与 x 全部元素相等
    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      // 否则，对 y 进行反向传播，使用全为 1 的张量作为梯度
      y.backward(torch::ones_like(y));
    }

    // 将 dropout 设置为评估模式
    dropout->eval();
    // 再次对大小为 2x2x10x10 的张量应用 Dropout2d，得到输出张量 y
    y = dropout(torch::ones({2, 2, 10, 10}));
    // 断言：y 所有元素之和应为 400
    ASSERT_EQ(y.sum().item<float>(), 400);
  }
}

TEST_F(ModulesTest, Dropout3d) {
  // 遍历 inplace 参数为 false 和 true 的情况
  for (const auto inplace : {false, true}) {
    // 设置概率 p 为 0.5
    auto p = 0.5;
    // 创建 Dropout3d 对象，设置概率为 p，并根据 inplace 参数设置选项
    Dropout3d dropout(Dropout3dOptions(p).inplace(inplace));
    // 创建大小为 50x50x2x2x2 的空张量 x，并填充元素为 1-p
    torch::Tensor x = torch::empty({50, 50, 2, 2, 2}).fill_(1 - p);
    // 如果 inplace 为 false，需要设置 x 为需要计算梯度
    if (!inplace) {
      x.requires_grad_(true);
    }
    // 对张量 x 应用 Dropout3d，得到输出张量 y
    torch::Tensor y = dropout(x);

    // 断言：y 的维度应为 5
    ASSERT_EQ(y.ndimension(), 5);
    // 断言：y 的第一个维度大小应为 50
    ASSERT_EQ(y.size(0), 50);
    // 断言：y 的第二个维度大小应为 50
    ASSERT_EQ(y.size(1), 50);
    // 断言：y 的第三个维度大小应为 2
    ASSERT_EQ(y.size(2), 2);
    // 断言：y 的第四个维度大小应为 2
    ASSERT_EQ(y.size(3), 2);
    // 断言：y 的第五个维度大小应为 2
    ASSERT_EQ(y.size(4), 2);
    // 断言：y 的平均值与 (1 - p) 的差的绝对值应小于 0.05
    ASSERT_LT((y.mean() - (1 - p)).abs().item<float>(), 0.05);

    // 如果 inplace 为 true，断言 y 应与 x 全部元素相等
    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      // 否则，对 y 进行反向传播，使用全为 1 的张量作
TEST_F(ModulesTest, Parameters) {
  // 创建一个 NestedModel 的 shared_ptr 对象
  auto model = std::make_shared<NestedModel>();
  // 获取模型的命名参数
  auto parameters = model->named_parameters();
  // 断言参数 "param" 的维度大小
  ASSERT_EQ(parameters["param"].size(0), 3);
  ASSERT_EQ(parameters["param"].size(1), 2);
  ASSERT_EQ(parameters["param"].size(2), 21);
  // 断言参数 "l1.bias" 的维度大小
  ASSERT_EQ(parameters["l1.bias"].size(0), 20);
  // 断言参数 "l1.weight" 的维度大小
  ASSERT_EQ(parameters["l1.weight"].size(0), 20);
  ASSERT_EQ(parameters["l1.weight"].size(1), 5);
  // 断言参数 "test.l1.bias" 的维度大小
  ASSERT_EQ(parameters["test.l1.bias"].size(0), 3);
  // 断言参数 "test.l1.weight" 的维度大小
  ASSERT_EQ(parameters["test.l1.weight"].size(0), 3);
  ASSERT_EQ(parameters["test.l1.weight"].size(1), 10);
  // 断言参数 "test.l2.bias" 的维度大小
  ASSERT_EQ(parameters["test.l2.bias"].size(0), 5);
  // 断言参数 "test.l2.weight" 的维度大小
  ASSERT_EQ(parameters["test.l2.weight"].size(0), 5);
  ASSERT_EQ(parameters["test.l2.weight"].size(1), 3);
  // 断言参数 "test.l3.bias" 的维度大小
  ASSERT_EQ(parameters["test.l3.bias"].size(0), 100);
  // 断言参数 "test.l3.weight" 的维度大小
  ASSERT_EQ(parameters["test.l3.weight"].size(0), 100);
  ASSERT_EQ(parameters["test.l3.weight"].size(1), 5);
}

TEST_F(ModulesTest, FunctionalCallsSuppliedFunction) {
  // 初始化一个标志位用于记录函数是否被调用
  bool was_called = false;
  // 创建一个 Functional 对象，其操作是将输入直接返回，并设置标志位
  auto functional = Functional([&was_called](torch::Tensor input) {
    was_called = true;
    return input;
  });
  // 调用 Functional 对象，期望输出与输入相同
  auto output = functional(torch::ones(5, torch::requires_grad()));
  // 断言函数确实被调用
  ASSERT_TRUE(was_called);
  // 断言输出与预期的全为1的张量相等
  ASSERT_TRUE(output.equal(torch::ones(5, torch::requires_grad())));

  // 重置标志位
  was_called = false;
  // 再次调用 Functional 对象，期望再次记录函数被调用
  output = functional(torch::ones(5, torch::requires_grad()));
  // 断言函数确实被再次调用
  ASSERT_TRUE(was_called);
  // 断言输出与预期的全为1的张量相等
  ASSERT_TRUE(output.equal(torch::ones(5, torch::requires_grad())));
}

TEST_F(ModulesTest, FunctionalWithTorchFunction) {
  // 创建一个 Functional 对象，使用 torch::relu 函数
  auto functional = Functional(torch::relu);
  // 断言对单元素1的输入应用 relu 后的输出为1
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 1);
  // 断言对单元素1的输入应用 relu 后的输出为1
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 1);
  // 断言对单元素-1的输入应用 relu 后的输出为0
  ASSERT_EQ(functional(torch::ones({}) * -1).item<float>(), 0);
}

TEST_F(ModulesTest, FunctionalArgumentBinding) {
  // 创建一个 Functional 对象，使用 torch::elu 函数，并指定额外的参数
  auto functional =
      Functional(torch::elu, /*alpha=*/1, /*scale=*/0, /*input_scale=*/1);
  // 断言对单元素1的输入应用 elu 后的输出为0
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 0);
}

TEST_F(ModulesTest, BatchNorm1dStateful) {
  // 创建一个 BatchNorm1d 对象，设置通道数为5
  BatchNorm1d bn(5);

  // 断言该 BatchNorm1d 对象正在追踪运行时统计信息
  ASSERT_TRUE(bn->options.track_running_stats());

  // 断言运行时均值张量已定义，并且维度为1，大小为5
  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  // 断言运行时方差张量已定义，并且维度为1，大小为5
  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  // 断言已跟踪的批次数张量已定义，并且维度为0
  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  // 断言 BatchNorm1d 对象支持仿射变换
  ASSERT_TRUE(bn->options.affine());

  // 断言权重张量已定义，并且维度为1，大小为5
  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  // 断言偏置张量已定义，并且维度为1，大小为5
  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}
TEST_F(ModulesTest, BatchNorm1dStateless) {
  // 创建一个 BatchNorm1d 对象，设置不跟踪 running statistics 和不进行仿射变换
  BatchNorm1d bn(
      BatchNorm1dOptions(5).track_running_stats(false).affine(false));

  // 断言 running_mean 属性未定义
  ASSERT_FALSE(bn->running_mean.defined());
  // 断言 running_var 属性未定义
  ASSERT_FALSE(bn->running_var.defined());
  // 断言 num_batches_tracked 属性未定义
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  // 断言 weight 属性未定义
  ASSERT_FALSE(bn->weight.defined());
  // 断言 bias 属性未定义
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm1d) {
  // 创建一个 BatchNorm1d 对象，设置输入维度为 5
  BatchNorm1d bn(5);
  // 将 BatchNorm1d 设置为评估模式
  bn->eval();

  // 创建一个需要梯度的张量作为输入
  auto input = torch::arange(2. * 5 * 2).view({2, 5, 2}).requires_grad_();
  // 将输入传递给 BatchNorm1d 的前向传播函数，得到输出
  auto output = bn->forward(input);
  // 创建预期输出的张量
  auto expected = torch::tensor(
      {{{0.0000, 1.0000},
        {2.0000, 3.0000},
        {4.0000, 5.0000},
        {6.0000, 7.0000},
        {8.0000, 9.0000}},
       {{10.0000, 10.9999},
        {11.9999, 12.9999},
        {13.9999, 14.9999},
        {15.9999, 16.9999},
        {17.9999, 18.9999}}});
  // 断言输出张量与预期张量在接近数值上相等
  ASSERT_TRUE(output.allclose(expected));
  // 对输出张量进行求和操作
  auto s = output.sum();
  // 反向传播求梯度
  s.backward();

  // 断言输入张量的大小与其梯度的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, BatchNorm2dStateful) {
  // 创建一个 BatchNorm2d 对象，设置输入通道数为 5
  BatchNorm2d bn(5);

  // 断言 BatchNorm2d 对象设置了跟踪 running statistics
  ASSERT_TRUE(bn->options.track_running_stats());

  // 断言 running_mean 属性已定义，并且是一维张量
  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  // 断言 running_var 属性已定义，并且是一维张量
  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  // 断言 num_batches_tracked 属性已定义，并且是标量
  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  // 断言 BatchNorm2d 对象设置了仿射变换
  ASSERT_TRUE(bn->options.affine());

  // 断言 weight 属性已定义，并且是一维张量
  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  // 断言 bias 属性已定义，并且是一维张量
  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm2dStateless) {
  // 创建一个 BatchNorm2d 对象，设置不跟踪 running statistics 和不进行仿射变换
  BatchNorm2d bn(
      BatchNorm2dOptions(5).track_running_stats(false).affine(false));

  // 断言 running_mean 属性未定义
  ASSERT_FALSE(bn->running_mean.defined());
  // 断言 running_var 属性未定义
  ASSERT_FALSE(bn->running_var.defined());
  // 断言 num_batches_tracked 属性未定义
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  // 断言 weight 属性未定义
  ASSERT_FALSE(bn->weight.defined());
  // 断言 bias 属性未定义
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm2d) {
  // 创建一个 BatchNorm2d 对象，设置输入通道数为 5
  BatchNorm2d bn(5);
  // 将 BatchNorm2d 设置为评估模式
  bn->eval();

  // 创建一个需要梯度的四维张量作为输入
  auto input =
      torch::arange(2. * 5 * 2 * 2).view({2, 5, 2, 2}).requires_grad_();
  // 将输入传递给 BatchNorm2d 的前向传播函数，得到输出
  auto output = bn->forward(input);
  // 创建预期输出的张量
  auto expected = torch::tensor(
      {{{{0.0000, 1.0000}, {2.0000, 3.0000}},
        {{4.0000, 5.0000}, {6.0000, 7.0000}},
        {{8.0000, 9.0000}, {10.0000, 10.9999}},
        {{11.9999, 12.9999}, {13.9999, 14.9999}},
        {{15.9999, 16.9999}, {17.9999, 18.9999}}},
       {{{19.9999, 20.9999}, {21.9999, 22.9999}},
        {{23.9999, 24.9999}, {25.9999, 26.9999}},
        {{27.9999, 28.9999}, {29.9998, 30.9998}},
        {{31.9998, 32.9998}, {33.9998, 34.9998}},
        {{35.9998, 36.9998}, {37.9998, 38.9998}}}});
  // 断言输出张量与预期张量在接近数值上相等
  ASSERT_TRUE(output.allclose(expected));
  // 对输出张量进行求和操作
  auto s = output.sum();
  // 反向传播求梯度
  s.backward();

  // 断言输入张量的大小与其梯度的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
TEST_F(ModulesTest, BatchNorm3dStateful) {
  // 创建一个 BatchNorm3d 实例，参数为 5
  BatchNorm3d bn(5);

  // 断言跟踪运行统计信息被启用
  ASSERT_TRUE(bn->options.track_running_stats());

  // 断言 running_mean 已定义且维度为 1，大小为 5
  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  // 断言 running_var 已定义且维度为 1，大小为 5
  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  // 断言 num_batches_tracked 已定义且维度为 0
  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  // 断言仿射变换被启用
  ASSERT_TRUE(bn->options.affine());

  // 断言权重 weight 已定义且维度为 1，大小为 5
  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  // 断言偏置 bias 已定义且维度为 1，大小为 5
  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm3dStateless) {
  // 创建一个不跟踪运行统计信息且不进行仿射变换的 BatchNorm3d 实例，参数为 5
  BatchNorm3d bn(
      BatchNorm3dOptions(5).track_running_stats(false).affine(false));

  // 断言 running_mean 未定义
  ASSERT_FALSE(bn->running_mean.defined());
  // 断言 running_var 未定义
  ASSERT_FALSE(bn->running_var.defined());
  // 断言 num_batches_tracked 未定义
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  // 断言权重 weight 未定义
  ASSERT_FALSE(bn->weight.defined());
  // 断言偏置 bias 未定义
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm3d) {
  // 创建一个 BatchNorm3d 实例，参数为 5
  BatchNorm3d bn(5);
  // 将实例设为评估模式
  bn->eval();

  // 创建输入张量，形状为 {2, 5, 2, 2, 2}，并要求计算梯度
  auto input =
      torch::arange(2. * 5 * 2 * 2 * 2).view({2, 5, 2, 2, 2}).requires_grad_();
  // 对输入进行前向传播
  auto output = bn->forward(input);
  // 创建期望输出张量
  auto expected = torch::tensor(
      {{{{{0.0000, 1.0000}, {2.0000, 3.0000}},
         {{4.0000, 5.0000}, {6.0000, 7.0000}}},
        {{{8.0000, 9.0000}, {10.0000, 10.9999}},
         {{11.9999, 12.9999}, {13.9999, 14.9999}}},
        {{{15.9999, 16.9999}, {17.9999, 18.9999}},
         {{19.9999, 20.9999}, {21.9999, 22.9999}}},
        {{{23.9999, 24.9999}, {25.9999, 26.9999}},
         {{27.9999, 28.9999}, {29.9998, 30.9998}}},
        {{{31.9998, 32.9998}, {33.9998, 34.9998}},
         {{35.9998, 36.9998}, {37.9998, 38.9998}}}},
       {{{{39.9998, 40.9998}, {41.9998, 42.9998}},
         {{43.9998, 44.9998}, {45.9998, 46.9998}}},
        {{{47.9998, 48.9998}, {49.9997, 50.9997}},
         {{51.9997, 52.9997}, {53.9997, 54.9997}}},
        {{{55.9997, 56.9997}, {57.9997, 58.9997}},
         {{59.9997, 60.9997}, {61.9997, 62.9997}}},
        {{{63.9997, 64.9997}, {65.9997, 66.9997}},
         {{67.9997, 68.9997}, {69.9996, 70.9996}}},
        {{{71.9996, 72.9996}, {73.9996, 74.9996}},
         {{75.9996, 76.9996}, {77.9996, 78.9996}}}}});
  // 断言输出张量与期望张量在数值上相近
  ASSERT_TRUE(output.allclose(expected));
  // 对输出张量求和
  auto s = output.sum();
  // 对求和结果进行反向传播
  s.backward();

  // 断言输入张量的形状与其梯度的形状相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
// 测试模块的 InstanceNorm1dStateful 函数
TEST_F(ModulesTest, InstanceNorm1dStateful) {
  // 创建 InstanceNorm1d 类对象，使用具有跟踪运行状态和可仿射操作的选项
  InstanceNorm1d instance_norm(
      InstanceNorm1dOptions(5).track_running_stats(true).affine(true));

  // 断言是否跟踪运行状态的选项被设置为 true
  ASSERT_TRUE(instance_norm->options.track_running_stats());

  // 断言是否定义了 running_mean，并且其维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->running_mean.defined());
  ASSERT_EQ(instance_norm->running_mean.dim(), 1);
  ASSERT_EQ(instance_norm->running_mean.size(0), 5);

  // 断言是否定义了 running_var，并且其维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->running_var.defined());
  ASSERT_EQ(instance_norm->running_var.dim(), 1);
  ASSERT_EQ(instance_norm->running_var.size(0), 5);

  // 断言是否定义了 num_batches_tracked，并且其维度为 0
  ASSERT_TRUE(instance_norm->num_batches_tracked.defined());
  ASSERT_EQ(instance_norm->num_batches_tracked.dim(), 0);

  // 断言是否设置了仿射操作的选项为 true
  ASSERT_TRUE(instance_norm->options.affine());

  // 断言是否定义了 weight，并且其维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->weight.defined());
  ASSERT_EQ(instance_norm->weight.dim(), 1);
  ASSERT_EQ(instance_norm->weight.size(0), 5);

  // 断言是否定义了 bias，并且其维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->bias.defined());
  ASSERT_EQ(instance_norm->bias.dim(), 1);
  ASSERT_EQ(instance_norm->bias.size(0), 5);
}

// 测试模块的 InstanceNorm1dStateless 函数
TEST_F(ModulesTest, InstanceNorm1dStateless) {
  // 创建 InstanceNorm1d 类对象，使用不跟踪运行状态和不可仿射操作的选项
  InstanceNorm1d instance_norm(
      InstanceNorm1dOptions(5).track_running_stats(false).affine(false));

  // 断言是否未定义 running_mean
  ASSERT_FALSE(instance_norm->running_mean.defined());
  // 断言是否未定义 running_var
  ASSERT_FALSE(instance_norm->running_var.defined());
  // 断言是否未定义 num_batches_tracked
  ASSERT_FALSE(instance_norm->num_batches_tracked.defined());
  // 断言是否未定义 weight
  ASSERT_FALSE(instance_norm->weight.defined());
  // 断言是否未定义 bias
  ASSERT_FALSE(instance_norm->bias.defined());
}

// 测试模块的 InstanceNorm1d 函数
TEST_F(ModulesTest, InstanceNorm1d) {
  // 创建 InstanceNorm1d 类对象，指定维度为 5
  InstanceNorm1d instance_norm(5);
  // 设置模块为评估模式
  instance_norm->eval();

  // 创建输入张量，形状为 {2, 5, 2}，并要求计算梯度
  auto input = torch::arange(2. * 5 * 2).view({2, 5, 2}).requires_grad_();
  // 使用 instance_norm 模块对输入进行前向传播计算
  auto output = instance_norm->forward(input);
  // 创建预期输出张量，形状为 {2, 5, 2}
  auto expected = torch::tensor(
      {{{-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000}},
       {{-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000}}});
  // 断言输出是否与预期输出在指定精度范围内相近
  ASSERT_TRUE(output.allclose(expected, 1e-3));
  // 对输出张量的所有元素进行求和
  auto s = output.sum();
  // 对求和结果进行反向传播
  s.backward();

  // 断言输入张量的形状与其梯度张量的形状相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
# 测试实例归一化模块的状态（带状态）
TEST_F(ModulesTest, InstanceNorm2dStateful) {
  # 创建一个实例归一化对象，使用选项设置为：跟踪运行统计信息和支持仿射变换
  InstanceNorm2d instance_norm(
      InstanceNorm2dOptions(5).track_running_stats(true).affine(true));

  # 断言实例归一化对象是否设置为跟踪运行统计信息
  ASSERT_TRUE(instance_norm->options.track_running_stats());

  # 断言运行均值张量已定义，并且其维度为1
  ASSERT_TRUE(instance_norm->running_mean.defined());
  ASSERT_EQ(instance_norm->running_mean.dim(), 1);
  ASSERT_EQ(instance_norm->running_mean.size(0), 5);

  # 断言运行方差张量已定义，并且其维度为1
  ASSERT_TRUE(instance_norm->running_var.defined());
  ASSERT_EQ(instance_norm->running_var.dim(), 1);
  ASSERT_EQ(instance_norm->running_var.size(0), 5);

  # 断言批次跟踪计数张量已定义，并且其维度为0（标量）
  ASSERT_TRUE(instance_norm->num_batches_tracked.defined());
  ASSERT_EQ(instance_norm->num_batches_tracked.dim(), 0);

  # 断言实例归一化对象是否支持仿射变换
  ASSERT_TRUE(instance_norm->options.affine());

  # 断言权重张量已定义，并且其维度为1
  ASSERT_TRUE(instance_norm->weight.defined());
  ASSERT_EQ(instance_norm->weight.dim(), 1);
  ASSERT_EQ(instance_norm->weight.size(0), 5);

  # 断言偏置张量已定义，并且其维度为1
  ASSERT_TRUE(instance_norm->bias.defined());
  ASSERT_EQ(instance_norm->bias.dim(), 1);
  ASSERT_EQ(instance_norm->bias.size(0), 5);
}

# 测试实例归一化模块的状态（无状态）
TEST_F(ModulesTest, InstanceNorm2dStateless) {
  # 创建一个实例归一化对象，使用选项设置为：不跟踪运行统计信息和不支持仿射变换
  InstanceNorm2d instance_norm(
      InstanceNorm2dOptions(5).track_running_stats(false).affine(false));

  # 断言运行均值张量未定义
  ASSERT_FALSE(instance_norm->running_mean.defined());

  # 断言运行方差张量未定义
  ASSERT_FALSE(instance_norm->running_var.defined());

  # 断言批次跟踪计数张量未定义
  ASSERT_FALSE(instance_norm->num_batches_tracked.defined());

  # 断言权重张量未定义
  ASSERT_FALSE(instance_norm->weight.defined());

  # 断言偏置张量未定义
  ASSERT_FALSE(instance_norm->bias.defined());
}

# 测试实例归一化模块（一般情况）
TEST_F(ModulesTest, InstanceNorm2d) {
  # 创建一个实例归一化对象，设置通道数为5，并将其设置为评估模式
  InstanceNorm2d instance_norm(5);
  instance_norm->eval();

  # 创建一个输入张量，形状为[2, 5, 2, 2]，并要求梯度计算
  auto input =
      torch::arange(2. * 5 * 2 * 2).view({2, 5, 2, 2}).requires_grad_();

  # 将输入张量输入到实例归一化对象中，计算输出
  auto output = instance_norm->forward(input);

  # 创建预期的输出张量
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
  
  # 断言实际输出张量与预期输出张量的接近程度在给定的误差范围内
  ASSERT_TRUE(output.allclose(expected, 1e-3));

  # 计算输出张量所有元素的和，并对输入张量进行反向传播
  auto s = output.sum();
  s.backward();

  # 断言输入张量的形状与其梯度张量的形状相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
TEST_F(ModulesTest, InstanceNorm3dStateful) {
  // 创建一个 InstanceNorm3d 对象，设置 track_running_stats 和 affine 为 true
  InstanceNorm3d instance_norm(
      InstanceNorm3dOptions(5).track_running_stats(true).affine(true));

  // 断言 instance_norm 对象的 track_running_stats 为 true
  ASSERT_TRUE(instance_norm->options.track_running_stats());

  // 断言 instance_norm 对象的 running_mean 已定义且维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->running_mean.defined());
  ASSERT_EQ(instance_norm->running_mean.dim(), 1);
  ASSERT_EQ(instance_norm->running_mean.size(0), 5);

  // 断言 instance_norm 对象的 running_var 已定义且维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->running_var.defined());
  ASSERT_EQ(instance_norm->running_var.dim(), 1);
  ASSERT_EQ(instance_norm->running_var.size(0), 5);

  // 断言 instance_norm 对象的 num_batches_tracked 已定义且维度为 0
  ASSERT_TRUE(instance_norm->num_batches_tracked.defined());
  ASSERT_EQ(instance_norm->num_batches_tracked.dim(), 0);

  // 断言 instance_norm 对象的 affine 为 true
  ASSERT_TRUE(instance_norm->options.affine());

  // 断言 instance_norm 对象的 weight 已定义且维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->weight.defined());
  ASSERT_EQ(instance_norm->weight.dim(), 1);
  ASSERT_EQ(instance_norm->weight.size(0), 5);

  // 断言 instance_norm 对象的 bias 已定义且维度为 1，大小为 5
  ASSERT_TRUE(instance_norm->bias.defined());
  ASSERT_EQ(instance_norm->bias.dim(), 1);
  ASSERT_EQ(instance_norm->bias.size(0), 5);
}

TEST_F(ModulesTest, InstanceNorm3dStateless) {
  // 创建一个 InstanceNorm3d 对象，设置 track_running_stats 和 affine 为 false
  InstanceNorm3d instance_norm(
      InstanceNorm3dOptions(5).track_running_stats(false).affine(false));

  // 断言 instance_norm 对象的 running_mean 未定义
  ASSERT_FALSE(instance_norm->running_mean.defined());

  // 断言 instance_norm 对象的 running_var 未定义
  ASSERT_FALSE(instance_norm->running_var.defined());

  // 断言 instance_norm 对象的 num_batches_tracked 未定义
  ASSERT_FALSE(instance_norm->num_batches_tracked.defined());

  // 断言 instance_norm 对象的 weight 未定义
  ASSERT_FALSE(instance_norm->weight.defined());

  // 断言 instance_norm 对象的 bias 未定义
  ASSERT_FALSE(instance_norm->bias.defined());
}

TEST_F(ModulesTest, InstanceNorm3d) {
  // 创建一个 InstanceNorm3d 对象，指定输入通道数为 5
  InstanceNorm3d instance_norm(5);
  // 将 instance_norm 设置为评估模式
  instance_norm->eval();

  // 创建一个输入张量 input，形状为 [2, 5, 2, 2, 2]，并要求梯度计算
  auto input =
      torch::arange(2. * 5 * 2 * 2 * 2).view({2, 5, 2, 2, 2}).requires_grad_();
  
  // 调用 instance_norm 的 forward 方法进行前向计算
  auto output = instance_norm->forward(input);

  // 创建一个期望的输出张量 expected
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
  
  // 断言 output 和 expected 在误差范围内相等
  ASSERT_TRUE(output.allclose(expected, 1e-3));
  
  // 计算 output 张量的所有元素的和
  auto s = output.sum();
  
  // 对 s 进行反向传播
  s.backward();
  
  // 断言输入张量 input 和其梯度的形状相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
TEST_F(ModulesTest, Linear_CUDA) {
  // 创建一个包含输入和输出大小的线性模型对象，输入维度为5，输出维度为2
  Linear model(5, 2);
  // 将模型移动到 CUDA 设备上进行计算
  model->to(torch::kCUDA);
  // 生成一个大小为[10, 5]的随机张量，并将其移动到 CUDA 设备上，同时开启梯度追踪
  auto x =
      torch::randn({10, 5}, torch::device(torch::kCUDA).requires_grad(true));
  // 对模型进行前向计算，得到输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和
  torch::Tensor s = y.sum();

  // 对损失函数的输出张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为2
  ASSERT_EQ(y.ndimension(), 2);
  // 断言输出张量 s 的维度为0（标量）
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的第一维大小为10
  ASSERT_EQ(y.size(0), 10);
  // 断言输出张量 y 的第二维大小为2
  ASSERT_EQ(y.size(1), 2);

  // 断言模型权重的梯度张量大小为 2 * 5
  ASSERT_EQ(model->weight.grad().numel(), 2 * 5);
}

TEST_F(ModulesTest, Linear2_CUDA) {
  // 创建一个包含输入和输出大小的线性模型对象，输入维度为5，输出维度为2
  Linear model(5, 2);
  // 将模型移动到 CUDA 设备上进行计算
  model->to(torch::kCUDA);
  // 将模型从 CUDA 设备移动到 CPU 上
  model->to(torch::kCPU);
  // 生成一个大小为[10, 5]的随机张量，并开启梯度追踪
  auto x = torch::randn({10, 5}, torch::requires_grad());
  // 对模型进行前向计算，得到输出张量 y
  auto y = model(x);
  // 计算输出张量 y 的所有元素的和
  torch::Tensor s = y.sum();

  // 对损失函数的输出张量 s 进行反向传播
  s.backward();
  // 断言输出张量 y 的维度为2
  ASSERT_EQ(y.ndimension(), 2);
  // 断言输出张量 s 的维度为0（标量）
  ASSERT_EQ(s.ndimension(), 0);
  // 断言输出张量 y 的第一维大小为10
  ASSERT_EQ(y.size(0), 10);
  // 断言输出张量 y 的第二维大小为2
  ASSERT_EQ(y.size(1), 2);

  // 断言模型权重的梯度张量大小为 2 * 5
  ASSERT_EQ(model->weight.grad().numel(), 2 * 5);
}

TEST_F(ModulesTest, L1Loss) {
  // 创建 L1 损失函数对象
  L1Loss loss;
  // 生成一个大小为[5, 6]的随机张量，并开启梯度追踪
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为[5, 6]的空张量，并填充随机值
  auto target = torch::empty({5, 6}).random_(2);
  // 对输入张量经过 sigmoid 函数计算后，输入损失函数，得到输出张量
  auto output = loss->forward(torch::sigmoid(input), target);
  // 计算输出张量 output 的所有元素的和
  auto s = output.sum();
  // 对损失函数的输出张量 s 进行反向传播
  s.backward();

  // 断言输出张量 output 的大小为空向量
  ASSERT_EQ(output.sizes(), std::vector<int64_t>());
  // 断言输入张量 input 的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MSELoss) {
  // 创建均方误差损失函数对象
  MSELoss loss;
  // 生成一个大小为[5, 6]的随机张量，并开启梯度追踪
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为[5, 6]的空张量，并填充随机值
  auto target = torch::empty({5, 6}).random_(2);
  // 对输入张量经过 sigmoid 函数计算后，输入损失函数，得到输出张量
  auto output = loss->forward(torch::sigmoid(input), target);
  // 计算输出张量 output 的所有元素的和
  auto s = output.sum();
  // 对损失函数的输出张量 s 进行反向传播
  s.backward();

  // 断言输出张量 output 的大小为空向量
  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  // 断言输入张量 input 的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, BCELoss) {
  // 创建二元交叉熵损失函数对象
  BCELoss loss;
  // 生成一个大小为[5, 6]的随机张量，并开启梯度追踪
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为[5, 6]的空张量，并填充随机值
  auto target = torch::empty({5, 6}).random_(2);
  // 对输入张量经过 sigmoid 函数计算后，输入损失函数，得到输出张量
  auto output = loss->forward(torch::sigmoid(input), target);
  // 计算输出张量 output 的所有元素的和
  auto s = output.sum();
  // 对损失函数的输出张量 s 进行反向传播
  s.backward();

  // 断言输出张量 output 的大小为空向量
  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  // 断言输入张量 input 的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, KLDivLoss) {
  // 创建 KL 散度损失函数对象
  KLDivLoss loss;
  // 生成一个大小为[5, 6]的随机张量，并开启梯度追踪
  auto input = torch::randn({5, 6}, torch::requires_grad());
  // 生成一个大小为[5, 6]的空张量，并填充随机值
  auto target = torch::empty({5, 6}).random_(2);
  // 对输入张量经过 sigmoid 函数计算后，输入损失函数，得到输出张量
  auto output = loss->forward(torch::sigmoid(input), target);
  // 计算输出张量 output 的所有元素的和
  auto s = output.sum();
  // 对损失函数的输出张量 s 进行反向传播
  s.backward();

  // 断言输出张量 output 的大小为空向量
  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  // 断言输入张量 input 的大小与其梯度张量的大小相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HingeEmbeddingLoss) {
  // 创建带边界的铰链损失函数对象，边界为2
  HingeEmbeddingLoss loss(HingeEmbeddingLossOptions().margin(2));
  // 生成一个大小为[2, 3]的浮点型张量，并开启梯度追踪
  auto input = torch::tensor(
      {{2, 22, 4}, {20, 10, 0}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 生成一个大小为[2, 3]的浮点型张量，并填充随机值
  auto target = torch::tensor({{2, 6, 4}, {1,
TEST_F(ModulesTest, MultiMarginLoss) {
  // 定义权重张量
  auto weight = torch::tensor({0.3, 0.3, 0.4}, torch::kFloat);
  // 创建多类别边界损失函数，设置边界为2，使用权重张量
  MultiMarginLoss loss(MultiMarginLossOptions().margin(2).weight(weight));
  // 定义输入张量，包含三个样本，每个样本三个特征
  auto input = torch::tensor(
      {{0.2, 0.2, 0.6}, {0.1, 0.8, 0.1}, {0.9, 0.09, 0.01}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量，指定每个样本的类别
  auto target = torch::tensor({2, 1, 0}, torch::kLong);
  // 计算损失值
  auto output = loss->forward(input, target);
  // 定义预期输出
  auto expected = torch::tensor({0.305556}, torch::kFloat);
  // 计算输出张量元素的总和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言输出张量与预期值在误差范围内接近
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  // 断言输入张量和其梯度的维度相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, CosineEmbeddingLoss) {
  // 创建余弦嵌入损失函数，设置边界为0.5
  CosineEmbeddingLoss cos(CosineEmbeddingLossOptions().margin(0.5));
  // 定义输入张量1，包含两个样本，每个样本三个特征，需要计算梯度
  auto input1 = torch::tensor(
      {{2, 3, 4}, {6, 2, 4}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 定义输入张量2，包含两个样本，每个样本三个特征，需要计算梯度
  auto input2 = torch::tensor(
      {{2, 3, 5}, {9, 12, 0}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量，指定每对样本的相似性标签
  auto target = torch::tensor({1, -1});
  // 计算损失值
  auto output = cos(input1, input2, target);
  // 定义预期输出
  auto expected = torch::tensor({0.1004}, torch::kFloat);
  // 计算输出张量元素的总和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言输出张量与预期值在误差范围内接近
  ASSERT_TRUE(output.allclose(expected, 1e-4));
  // 断言输入张量和其梯度的维度相同
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
  ASSERT_EQ(input2.sizes(), input2.grad().sizes());
}

TEST_F(ModulesTest, SmoothL1LossDefaultOptions) {
  // 创建平滑L1损失函数，默认参数
  SmoothL1Loss loss;
  // 定义输入张量，包含三个样本，每个样本一个特征，需要计算梯度
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量，指定每个样本的目标值
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算损失值
  auto output = loss(input, target);
  // 定义预期输出
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  // 计算输出张量元素的总和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言输出张量与预期值在误差范围内接近
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和其梯度的维度相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HuberLossDefaultOptions) {
  // 创建Huber损失函数，默认参数
  HuberLoss loss;
  // 定义输入张量，包含三个样本，每个样本一个特征，需要计算梯度
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量，指定每个样本的目标值
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算损失值
  auto output = loss(input, target);
  // 定义预期输出
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  // 计算输出张量元素的总和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言输出张量与预期值在误差范围内接近
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和其梯度的维度相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelMarginLossDefaultOptions) {
  // 创建多标签边界损失函数，默认参数
  MultiLabelMarginLoss loss;
  // 定义输入张量，包含一个样本，具有四个标签，需要计算梯度
  auto input = torch::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 定义目标张量，指定每个标签的目标值
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  // 计算损失值
  auto output = loss->forward(input, target);
  // 定义预期输出
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  // 计算输出张量元素的总和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言输出张量与预期值在误差范围内接近
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量和其梯度的维度相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
TEST_F(ModulesTest, SmoothL1LossNoReduction) {
  // 创建 Smooth L1 Loss 对象，设置 reduction 参数为 None
  SmoothL1Loss loss(/*reduction=*/torch::kNone);
  // 创建输入张量，指定数据类型为 float32，并且需要计算梯度
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量，数据类型为 float32
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算损失值
  auto output = loss(input, target);
  // 创建期望的输出张量
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  // 对输出张量求和
  auto s = output.sum();
  // 反向传播计算梯度
  s.backward();

  // 断言输出张量是否与期望的输出张量近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的大小与梯度张量的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HuberLossNoReduction) {
  // 创建 Huber Loss 对象，设置 reduction 参数为 None
  HuberLoss loss(/*reduction=*/torch::kNone);
  // 创建输入张量，指定数据类型为 float32，并且需要计算梯度
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量，数据类型为 float32
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算损失值
  auto output = loss(input, target);
  // 创建期望的输出张量
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  // 对输出张量求和
  auto s = output.sum();
  // 反向传播计算梯度
  s.backward();

  // 断言输出张量是否与期望的输出张量近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的大小与梯度张量的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelMarginLossNoReduction) {
  // 创建 Multi-Label Margin Loss 对象，设置 reduction 参数为 None
  MultiLabelMarginLoss loss(torch::kNone);
  // 创建输入张量，包含一个维度为 4 的向量，指定数据类型为 float32，并且需要计算梯度
  auto input = torch::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量，包含一个维度为 4 的向量，数据类型为 int64
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  // 调用 forward 方法计算损失值
  auto output = loss->forward(input, target);
  // 创建期望的输出张量
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  // 对输出张量求和
  auto s = output.sum();
  // 反向传播计算梯度
  s.backward();

  // 断言输出张量是否与期望的输出张量近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的大小与梯度张量的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, SmoothL1LossBeta) {
  // 创建 Smooth L1 Loss 对象，设置 beta 参数为 0.2
  auto options = SmoothL1LossOptions().beta(0.2);
  SmoothL1Loss loss(options);
  // 创建输入张量，指定数据类型为 float32，并且需要计算梯度
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量，数据类型为 float32
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算损失值
  auto output = loss(input, target);
  // 创建期望的输出张量
  auto expected = torch::tensor(0.108333, torch::kFloat);
  // 对输出张量求和
  auto s = output.sum();
  // 反向传播计算梯度
  s.backward();

  // 断言输出张量是否与期望的输出张量近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的大小与梯度张量的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HuberLossDelta) {
  // 创建 Huber Loss 对象，设置 delta 参数为 0.2
  auto options = HuberLossOptions().delta(0.2);
  HuberLoss loss(options);
  // 创建输入张量，指定数据类型为 float32，并且需要计算梯度
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量，数据类型为 float32
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  // 计算损失值
  auto output = loss(input, target);
  // 创建期望的输出张量
  auto expected = torch::tensor(0.0216666, torch::kFloat);
  // 对输出张量求和
  auto s = output.sum();
  // 反向传播计算梯度
  s.backward();

  // 断言输出张量是否与期望的输出张量近似相等
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量的大小与梯度张量的大小相等
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}
TEST_F(ModulesTest, TripletMarginLoss) {
  // 创建 TripletMarginLoss 对象并设置 margin 为 1.0
  TripletMarginLoss loss(TripletMarginLossOptions().margin(1.0));
  
  // 创建张量 anchor，形状为 {{3., 3.}}，数据类型为浮点数，并启用梯度计算
  auto anchor = torch::tensor(
      {{3., 3.}}, torch::dtype(torch::kFloat).requires_grad(true));
  
  // 创建张量 positive，形状为 {{2., 2.}}，数据类型为浮点数，并启用梯度计算
  auto positive = torch::tensor(
      {{2., 2.}}, torch::dtype(torch::kFloat).requires_grad(true));
  
  // 创建张量 negative，形状为 {{0., 0.}}，数据类型为浮点数，并启用梯度计算
  auto negative = torch::tensor(
      {{0., 0.}}, torch::dtype(torch::kFloat).requires_grad(true));
  
  // 调用 TripletMarginLoss 对象的 forward 方法计算损失
  auto output = loss->forward(anchor, positive, negative);
  
  // 创建期望输出张量 expected，值为 {0.}，数据类型为浮点数
  auto expected = torch::tensor({0.}, torch::kFloat);
  
  // 计算 output 张量的总和
  auto s = output.sum();
  
  // 反向传播，计算梯度
  s.backward();
  
  // 断言 output 和 expected 在给定误差范围内相等
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  
  // 断言 anchor 张量的大小与梯度的大小相等
  ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
}

TEST_F(ModulesTest, TripletMarginWithDistanceLossDefaultParity) {
  // 检查使用默认的 TripletMarginLoss 选项作为距离函数时，
  // 如果使用 torch::pairwise_distance，输出是否相等（在默认情况下相等）。
  
  // 定义减少类型的向量
  std::vector<TripletMarginWithDistanceLossOptions::reduction_t> reductions = {
      torch::kSum, torch::kMean, torch::kNone};
  
  // 定义 margin 的值的向量
  std::vector<float> margins = {0.5, 1.0, 1.5};
  
  // 定义 swaps 的布尔值向量
  std::vector<bool> swaps = {true, false};

  // 遍历 reductions 向量
  for (auto& reduction : reductions) {
    // 遍历 margins 向量
    for (auto& margin : margins) {
      // 遍历 swaps 向量
      for (const auto swap : swaps) {
        // 创建 anchor 张量，形状为 {100, 128}，数据类型为浮点数，并启用梯度计算
        auto anchor = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        
        // 创建 positive 张量，形状为 {100, 128}，数据类型为浮点数，并启用梯度计算
        auto positive = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        
        // 创建 negative 张量，形状为 {100, 128}，数据类型为浮点数，并启用梯度计算
        auto negative = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));

        // 创建基本选项对象，设置 reduction、margin 和 swap
        auto basicOptions =
            TripletMarginLossOptions().reduction(reduction).margin(margin).swap(
                swap);
        
        // 创建距离选项对象，设置 reduction、margin 和 swap
        auto distanceOptions = TripletMarginWithDistanceLossOptions()
                                   .reduction(reduction)
                                   .margin(margin)
                                   .swap(swap);
        
        // 创建基本 TripletMarginLoss 对象，使用基本选项
        TripletMarginLoss basicLoss(basicOptions);
        
        // 创建带距离的 TripletMarginLoss 对象，使用距离选项
        TripletMarginWithDistanceLoss distanceLoss(distanceOptions);

        // 计算基本 TripletMarginLoss 对象的输出
        auto basicOutput = basicLoss->forward(anchor, positive, negative);
        
        // 计算带距离 TripletMarginLoss 对象的输出
        auto distanceOutput = distanceLoss->forward(anchor, positive, negative);
        
        // 使用操作符计算基本 TripletMarginLoss 对象的输出
        auto basicOperatorOutput = basicLoss(anchor, positive, negative);
        
        // 使用操作符计算带距离 TripletMarginLoss 对象的输出
        auto distanceOperatorOutput = distanceLoss(anchor, positive, negative);

        // 断言带距离 TripletMarginLoss 对象的输出与基本 TripletMarginLoss 对象的输出在给定误差范围内相等
        ASSERT_TRUE(distanceOutput.allclose(basicOutput, 1e-6, 1e-6));
        
        // 断言带距离操作符的输出与带距离 TripletMarginLoss 对象的输出在给定误差范围内相等
        ASSERT_TRUE(
            distanceOperatorOutput.allclose(distanceOutput, 1e-6, 1e-6));
        
        // 断言带距离操作符的输出与基本操作符的输出在给定误差范围内相等
        ASSERT_TRUE(
            distanceOperatorOutput.allclose(basicOperatorOutput, 1e-6, 1e-6));

        // 处理 torch::kNone reduction 的情况
        
        // 计算 distanceOutput 张量的总和
        auto sum = distanceOutput.sum();
        
        // 反向传播，计算梯度
        sum.backward();
        
        // 断言 anchor、positive 和 negative 张量的大小与梯度的大小相等
        ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
        ASSERT_EQ(positive.sizes(), positive.grad().sizes());
        ASSERT_EQ(negative.sizes(), negative.grad().sizes());
      }
    }
  }
}
// 定义测试用例 ModulesTest.TripletMarginWithDistanceLossFunctionalParity
TEST_F(ModulesTest, TripletMarginWithDistanceLossFunctionalParity) {
  // 检查 F::triplet_margin_with_distance_loss 和 TripletMarginWithDistanceLoss 之间的功能对比

  // 定义计算 pairwise distance 的 Lambda 函数
  auto pairwise_distance = [&](const torch::Tensor& x, const torch::Tensor& y) {
    return torch::pairwise_distance(x, y);
  };

  // 定义计算 cosine distance 的 Lambda 函数
  auto cosine_distance = [&](const torch::Tensor& x, const torch::Tensor& y) {
    return 1.0 - torch::cosine_similarity(x, y);
  };

  // 定义距离函数选项的向量，包括 pairwise_distance 和 cosine_distance
  std::vector<TripletMarginWithDistanceLossOptions::distance_function_t>
      distance_functions = {pairwise_distance, cosine_distance};

  // 定义减少（reduction）选项的向量，包括 torch::kSum, torch::kMean, torch::kNone
  std::vector<TripletMarginWithDistanceLossOptions::reduction_t> reductions = {
      torch::kSum, torch::kMean, torch::kNone};

  // 定义 margin 值的向量，包括 0.5, 1.0, 1.5
  std::vector<float> margins = {0.5, 1.0, 1.5};

  // 定义 swap 布尔值的向量，包括 true 和 false
  std::vector<bool> swaps = {true, false};

  // 嵌套循环，测试所有的距离函数、减少选项、margin 值和 swap 布尔值组合
  for (auto& function : distance_functions) {
    for (auto& reduction : reductions) {
      for (auto& margin : margins) {
        for (const auto swap : swaps) {
          // 创建 TripletMarginWithDistanceLoss 的选项对象
          auto moduleOptions = TripletMarginWithDistanceLossOptions()
                                   .distance_function(function)
                                   .reduction(reduction)
                                   .margin(margin)
                                   .swap(swap);

          // 创建 TripletMarginWithDistanceLossFuncOptions 的选项对象
          auto functionOptions =
              torch::nn::functional::TripletMarginWithDistanceLossFuncOptions()
                  .distance_function(function)
                  .reduction(reduction)
                  .margin(margin)
                  .swap(swap);

          // 创建随机张量 anchor, positive 和 negative
          auto anchor = torch::randn(
              {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
          auto positive = torch::randn(
              {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
          auto negative = torch::randn(
              {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));

          // 创建 TripletMarginWithDistanceLoss 模块
          TripletMarginWithDistanceLoss distanceLoss(moduleOptions);

          // 使用模块计算前向传播结果
          auto moduleOutput = distanceLoss->forward(anchor, positive, negative);

          // 直接调用模块的运算符重载进行前向传播计算
          auto moduleOperatorOutput = distanceLoss(anchor, positive, negative);

          // 使用 functional 中的函数进行前向传播计算
          auto functionOutput =
              torch::nn::functional::triplet_margin_with_distance_loss(
                  anchor, positive, negative, functionOptions);

          // 断言模块输出与 functional 函数输出在给定的误差范围内相等
          ASSERT_TRUE(moduleOutput.allclose(functionOutput, 1e-6, 1e-6));
          ASSERT_TRUE(
              moduleOperatorOutput.allclose(functionOutput, 1e-6, 1e-6));
        }
      }
    }
  }
}
TEST_F(ModulesTest, NLLLoss) {
  // 创建 NLLLoss 对象
  NLLLoss loss;
  // 创建输入张量，包含浮点数值和梯度信息
  auto input = torch::tensor(
      {{-0.1315, -3.1315, -2.5315},
       {-3.7038, -0.1038, -2.6038},
       {-2.3422, -1.3422, -0.4422}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量，包含长整型索引
  auto target = torch::tensor({1, 0, 2}, torch::kLong);
  // 前向传播，计算损失
  auto output = loss->forward(input, target);
  // 创建期望输出张量，包含浮点数值
  auto expected = torch::tensor(2.4258, torch::kFloat);
  // 对输出张量进行求和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言：输出张量的值接近期望值，允许的误差为 1e-04
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  // 断言：输入张量的尺寸与其梯度的尺寸相同
  ASSERT_TRUE(
      NLLLoss(NLLLossOptions().ignore_index(-100).reduction(torch::kMean))
          ->forward(input, target)
          .allclose(expected, 1e-04));
}

TEST_F(ModulesTest, CrossEntropyLoss) {
  // 创建 CrossEntropyLoss 对象
  CrossEntropyLoss loss;
  // 创建输入张量，包含浮点数值和梯度信息
  auto input = torch::tensor(
      {{3., 3.}, {2., 2.}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量，包含长整型索引
  auto target = torch::tensor({0, 1}, torch::kLong);
  // 前向传播，计算损失
  auto output = loss->forward(input, target);
  // 创建期望输出张量，包含浮点数值
  auto expected = torch::tensor(0.6931, torch::kFloat);
  // 对输出张量进行求和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言：输出张量的值接近期望值，允许的误差为 1e-04
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  // 断言：输入张量的尺寸与其梯度的尺寸相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());

  // 标签平滑（使用类索引）
  loss = CrossEntropyLoss(
      CrossEntropyLossOptions().label_smoothing(0.15).reduction(torch::kMean));
  // 更新输入张量，包含浮点数值和梯度信息
  input = torch::tensor(
      {{3., 1.}, {1., 2.}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 更新目标张量，包含长整型索引
  target = torch::tensor({0, 1}, torch::kLong);
  // 前向传播，计算损失
  output = loss->forward(input, target);
  // 创建期望输出张量，包含浮点数值
  expected = torch::tensor(0.3326, torch::kFloat);
  // 对输出张量进行求和
  s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言：输出张量的值接近期望值，允许的误差为 1e-04
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  // 断言：输入张量的尺寸与其梯度的尺寸相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());

  // 标签平滑（使用目标概率）
  loss = CrossEntropyLoss(
      CrossEntropyLossOptions().label_smoothing(0.2).reduction(torch::kMean));
  // 更新输入张量，包含浮点数值和梯度信息
  input = torch::tensor(
      {{3., 1.}, {1., 2.}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 更新目标张量，包含浮点数值
  target = torch::tensor({{0.8, 0.2}, {0.1, 0.9}}, torch::kFloat);
  // 前向传播，计算损失
  output = loss->forward(input, target);
  // 创建期望输出张量，包含浮点数值
  expected = torch::tensor(0.5701, torch::kFloat);
  // 对输出张量进行求和
  s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言：输出张量的值接近期望值，允许的误差为 1e-04
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  // 断言：输入张量的尺寸与其梯度的尺寸相同
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, CosineSimilarity) {
  // 创建 CosineSimilarity 对象，设置维度为1
  CosineSimilarity cos(CosineSimilarityOptions().dim(1));
  // 创建输入张量1，包含浮点数值和梯度信息
  auto input1 = torch::tensor(
      {{1, 2, 3}, {4, 5, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建输入张量2，包含浮点数值和梯度信息
  auto input2 = torch::tensor(
      {{1, 8, 3}, {2, 1, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 前向传播，计算余弦相似度
  auto output = cos->forward(input1, input2);
  // 创建期望输出张量，包含浮点数值
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  // 对输出张量进行求和
  auto s = output.sum();
  // 反向传播，计算梯度
  s.backward();

  // 断言：输出张量的值接近期望值，允许的误差为 1e-04
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  // 断言：输入张量1的尺寸与其梯度的尺寸相同
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}
TEST_F(ModulesTest, SoftMarginLossDefaultOptions) {
  // 创建 SoftMarginLoss 对象
  SoftMarginLoss loss;
  // 创建输入张量，并指定需要计算梯度
  auto input = torch::tensor(
      {2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  // 计算损失函数的输出
  auto output = loss->forward(input, target);
  // 期望的输出结果
  auto expected = torch::tensor({1.3767317}, torch::kFloat);
  // 对输出结果进行求和
  auto s = output.sum();
  // 反向传播
  s.backward();

  // 断言输出结果与期望结果的近似程度
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量与梯度张量的大小匹配
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelSoftMarginLossDefaultOptions) {
  // 创建 MultiLabelSoftMarginLoss 对象
  MultiLabelSoftMarginLoss loss;
  // 创建输入张量，并指定需要计算梯度
  auto input = torch::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target =
      torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  // 计算损失函数的输出
  auto output = loss->forward(input, target);
  // 期望的输出结果
  auto expected = torch::tensor({0.7608436}, torch::kFloat);
  // 对输出结果进行求和
  auto s = output.sum();
  // 反向传播
  s.backward();

  // 断言输出结果与期望结果的近似程度
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量与梯度张量的大小匹配
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, SoftMarginLossNoReduction) {
  // 创建不进行 reduction 的 SoftMarginLoss 对象
  SoftMarginLoss loss(torch::kNone);
  // 创建输入张量，并指定需要计算梯度
  auto input = torch::tensor(
      {2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  // 计算损失函数的输出
  auto output = loss->forward(input, target);
  // 期望的输出结果
  auto expected = torch::tensor(
      {2.1269281, 0.01814993, 0.3132617, 3.0485873}, torch::kFloat);
  // 对输出结果进行求和
  auto s = output.sum();
  // 反向传播
  s.backward();

  // 断言输出结果与期望结果的近似程度
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量与梯度张量的大小匹配
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelSoftMarginLossWeightedNoReduction) {
  // 创建带有权重且不进行 reduction 的 MultiLabelSoftMarginLoss 对象
  auto input = torch::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      torch::dtype(torch::kFloat).requires_grad(true));
  // 创建目标张量
  auto target =
      torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  // 创建权重张量
  auto weight = torch::tensor({0.1, 0.6, 0.4, 0.8}, torch::kFloat);
  // 设置 MultiLabelSoftMarginLoss 的选项，指定不进行 reduction 和设置权重
  auto options =
      MultiLabelSoftMarginLossOptions().reduction(torch::kNone).weight(weight);
  // 创建 MultiLabelSoftMarginLoss 对象
  MultiLabelSoftMarginLoss loss = MultiLabelSoftMarginLoss(options);
  // 计算损失函数的输出
  auto output = loss->forward(input, target);
  // 期望的输出结果
  auto expected = torch::tensor({0.4876902, 0.3321295}, torch::kFloat);
  // 对输出结果进行求和
  auto s = output.sum();
  // 反向传播
  s.backward();

  // 断言输出结果与期望结果的近似程度
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量与梯度张量的大小匹配
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, PairwiseDistance) {
  // 创建 PairwiseDistance 对象，并设置 p=1
  PairwiseDistance dist(PairwiseDistanceOptions().p(1));
  // 创建两个输入张量，并指定需要计算梯度
  auto input1 = torch::tensor(
      {{1, 2, 3}, {4, 5, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto input2 = torch::tensor(
      {{1, 8, 3}, {2, 1, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  // 计算 PairwiseDistance 的输出
  auto output = dist->forward(input1, input2);
  // 期望的输出结果
  auto expected = torch::tensor({6, 6}, torch::kFloat);
  // 对输出结果进行求和
  auto s = output.sum();
  // 反向传播
  s.backward();

  // 断言输出结果与期望结果的近似程度
  ASSERT_TRUE(output.allclose(expected));
  // 断言输入张量与梯度张量的大小匹配
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}
TEST_F(ModulesTest, ELU) {
  // 定义测试用例的尺寸大小
  const auto size = 3;
  // 遍历不同的 alpha 值
  for (const auto alpha : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    // 遍历是否原地操作的标志位
    for (const auto inplace : {false, true}) {
      // 创建 ELU 模型对象，根据 alpha 和 inplace 参数进行配置
      ELU model{ELUOptions().alpha(alpha).inplace(inplace)};
      // 生成一个指定范围内的等差序列作为输入数据
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      // 调整输入数据的形状为三维
      x.resize_({size, size, size});
      // 如果不是原地操作，则需要设置梯度追踪
      if (!inplace) {
        x.requires_grad_(true);
      }
      // 备份原始的输入数据
      auto x_orig = x.clone();
      // 使用 ELU 模型处理输入数据得到输出
      auto y = model(x);
      // 对输出进行求和操作
      torch::Tensor s = y.sum();

      // 断言输出的张量维度为零维
      ASSERT_EQ(s.ndimension(), 0);

      // 断言输出的张量维度为三维，并且尺寸符合预期
      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      
      // 计算预期的输出张量
      auto y_exp = torch::max(torch::zeros_like(x_orig), x_orig) +
          torch::min(torch::zeros_like(x_orig),
                     alpha * (torch::exp(x_orig) - 1.0));
      // 断言模型输出与预期结果的近似程度
      ASSERT_TRUE(torch::allclose(y, y_exp));
      
      // 如果是原地操作，则进一步断言输入数据与预期结果的近似程度；否则执行反向传播
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
      } else {
        s.backward();
      }
    }
  }
}

TEST_F(ModulesTest, SELU) {
  // 遍历是否原地操作的标志位
  for (const auto inplace : {false, true}) {
    // 创建 SELU 模型对象，根据 inplace 参数进行配置
    SELU model(inplace);
    // 生成一个随机正态分布的输入数据
    auto input = torch::randn({5, 5});
    // 如果不是原地操作，则需要设置梯度追踪
    if (!inplace) {
      input.requires_grad_(true);
    }
    // 备份原始的输入数据
    auto input_orig = input.clone();
    // 使用 SELU 模型处理输入数据得到输出
    auto output = model->forward(input);
    // 定义 SELU 激活函数的参数值
    const double scale = 1.0507009873554804934193349852946;
    const double alpha = 1.6732632423543772848170429916717;
    // 创建一个与输入数据形状相同的全零张量
    auto zero = torch::zeros_like(input);
    // 计算预期的输出张量
    auto expected = scale *
        (torch::max(zero, input_orig) +
         torch::min(zero, alpha * (torch::exp(input_orig) - 1)));
    // 对输出张量进行求和操作
    auto s = output.sum();

    // 断言输出的张量维度为零维
    ASSERT_EQ(s.ndimension(), 0);
    // 断言模型输出与预期结果的近似程度
    ASSERT_TRUE(output.allclose(expected));
    
    // 如果是原地操作，则进一步断言输入数据与预期结果的近似程度；否则执行反向传播
    if (inplace) {
      ASSERT_TRUE(input.allclose(expected));
    } else {
      s.backward();
    }
  }
}

TEST_F(ModulesTest, Hardshrink) {
  // 定义测试用例的尺寸大小
  const auto size = 3;
  // 遍历不同的 lambda 值
  for (const auto lambda : {-4.2, -1.0, -0.42, 0.0, 0.42, 1.0, 4.2, 42.42}) {
    // 创建 Hardshrink 模型对象，根据 lambda 参数进行配置
    Hardshrink model{HardshrinkOptions().lambda(lambda)};
    // 生成一个指定范围内的等差序列作为输入数据
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    // 调整输入数据的形状为三维，并设置其需要梯度追踪
    x.resize_({size, size, size}).set_requires_grad(true);
    // 使用 Hardshrink 模型处理输入数据得到输出
    auto y = model(x);
    // 对输出张量进行求和操作
    torch::Tensor s = y.sum();

    // 执行反向传播
    s.backward();
    // 断言输出的张量维度为零维
    ASSERT_EQ(s.ndimension(), 0);
    // 断言输出的张量维度为三维，并且尺寸符合预期
    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 计算预期的输出张量
    auto y_exp = (x.abs() > lambda) * x;
    // 断言模型输出与预期结果的近似程度
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, Hardtanh) {
  // 定义测试用例的尺寸大小
  const auto size = 3;
  // 遍历不同的 min_val 值
  for (const auto min_val : {-4.2, -1.0, -0.42, 0.0}) {
    // 遍历三个不同的最大值（0.42, 1.0, 4.2）
    for (const auto max_val : {0.42, 1.0, 4.2}) {
      // 遍历两种 inplace 模式（false 和 true）
      for (const auto inplace : {false, true}) {
        // 根据给定的参数创建 Hardtanh 模型对象
        Hardtanh model{
            HardtanhOptions().min_val(min_val).max_val(max_val).inplace(
                inplace)};
        // 创建一个包含从-10.0到10.0的等差序列，并重塑为 size x size x size 的张量
        auto x = torch::linspace(-10.0, 10.0, size * size * size);
        x.resize_({size, size, size});
        // 如果不是 inplace 模式，则设置张量 x 需要梯度追踪
        if (!inplace) {
          x.requires_grad_(true);
        }
        // 克隆原始输入张量 x
        auto x_orig = x.clone();
        // 将张量 x 输入到 Hardtanh 模型中，计算输出 y
        auto y = model(x);
        // 对 y 的所有元素进行求和，返回一个标量张量 s
        torch::Tensor s = y.sum();

        // 使用断言检查条件
        ASSERT_EQ(s.ndimension(), 0); // s 的维度应为 0
        ASSERT_EQ(y.ndimension(), 3); // y 的维度应为 3
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size})); // y 的尺寸应为 {size, size, size}

        // 计算预期的输出 y_exp，根据 Hardtanh 激活函数的定义
        auto y_exp = (x_orig < min_val) * min_val +
            ((x_orig >= min_val) * (x_orig <= max_val)) * x_orig +
            (x_orig > max_val) * max_val;

        // 使用断言检查模型输出 y 是否与预期输出 y_exp 接近
        ASSERT_TRUE(torch::allclose(y, y_exp));

        // 如果是 inplace 模式，则断言原始输入 x 是否与预期输出 y_exp 接近
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        } else {
          // 否则，在不是 inplace 模式下进行反向传播
          s.backward();
        }
      }
    }
}

TEST_F(ModulesTest, HardtanhMinValGEMaxVal) {
  // 断言，测试 Hardtanh 类的行为，当最小值和最大值相同时是否抛出异常
  ASSERT_THROWS_WITH(
      Hardtanh{HardtanhOptions().min_val(0.42).max_val(0.42)},
      "max_val must be greater than min_val");
  // 断言，测试 Hardtanh 类的行为，当最大值小于最小值时是否抛出异常
  ASSERT_THROWS_WITH(
      Hardtanh{HardtanhOptions().min_val(0.42).max_val(-0.42)},
      "max_val must be greater than min_val");

  // 初始化 Hardtanh 模块，设定最小值和最大值
  Hardtanh ht{HardtanhOptions().min_val(-0.42).max_val(0.42)};
  // 修改最小值为 0.42，预期会抛出异常
  ht->options.min_val(0.42);
  ASSERT_THROWS_WITH(ht->reset(), "max_val must be greater than min_val");
  // 修改最大值为 -0.42，预期会抛出异常
  ht->options.max_val(-0.42);
  ASSERT_THROWS_WITH(ht->reset(), "max_val must be greater than min_val");
}

TEST_F(ModulesTest, LeakyReLU) {
  const auto size = 3;
  // 遍历 inplace 和 negative_slope 参数组合的情况
  for (const auto inplace : {false, true}) {
    // 遍历 negative_slope 的不同取值
    for (const auto negative_slope : {0.0, 0.42, 1.0}) {
      // 遍历 type 类型的不同取值（torch::kFloat 和 torch::kBFloat16）
      for (const auto type : {torch::kFloat, torch::kBFloat16}) {
        // 初始化 LeakyReLU 模块，设定 negative_slope 和 inplace 参数
        LeakyReLU model{
            LeakyReLUOptions().negative_slope(negative_slope).inplace(inplace)};
        // 生成一个 size*size*size 大小的序列，并转换为指定类型
        auto x = torch::linspace(-10.0, 10.0, size * size * size).to(type);
        x.resize_({size, size, size});
        // 如果 inplace 为 false，则设置 x 为需要梯度
        if (!inplace) {
          x.requires_grad_(true);
        }
        // 克隆原始输入 x
        auto x_orig = x.clone();
        // 使用模块处理输入 x，得到输出 y
        auto y = model(x);
        // 对 y 求和
        torch::Tensor s = y.sum();

        // 断言：s 的维度应为 0
        ASSERT_EQ(s.ndimension(), 0);
        // 断言：y 的维度应为 3
        ASSERT_EQ(y.ndimension(), 3);
        // 断言：y 的尺寸应为 {size, size, size}
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        // 计算预期的 y_exp 值
        auto y_exp =
            (x_orig < 0) * x_orig * negative_slope + (x_orig >= 0) * x_orig;
        // 断言：y 与 y_exp 应接近
        ASSERT_TRUE(torch::allclose(y, y_exp));
        // 如果 inplace 为 true，则断言：x 与 y_exp 应接近
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        } else {
          // 否则对 s 进行反向传播
          s.backward();
        }
      }
    }
  }
}

TEST_F(ModulesTest, LogSigmoid) {
  const auto size = 3;
  // 初始化 LogSigmoid 模块
  LogSigmoid model;
  // 生成一个 size*size*size 大小的序列，并设置为需要梯度
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size}).set_requires_grad(true);
  // 使用模块处理输入 x，得到输出 y
  auto y = model(x);
  // 对 y 求和
  torch::Tensor s = y.sum();

  // 对 s 进行反向传播
  s.backward();
  // 断言：s 的维度应为 0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言：y 的维度应为 3
  ASSERT_EQ(y.ndimension(), 3);
  // 断言：y 的尺寸应为 {size, size, size}
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  // 计算预期的 y_exp 值
  auto y_exp = torch::log(
      torch::ones_like(x) / (torch::ones_like(x) + torch::exp(torch::neg(x))));
  // 断言：y 与 y_exp 应接近，使用指定的容差
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

TEST_F(ModulesTest, Softmax) {
  // 初始化 Softmax 模块，指定维度为 1
  Softmax m(/*dim=*/1);
  // 生成一个大小为 {2, 5} 的输入张量
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // 使用模块处理输入，得到输出
  auto output = m(input);
  // 计算输入每行的指数和
  auto sum = torch::sum(torch::exp(input), 1);

  // 对每行进行断言
  for (const auto i : c10::irange(2)) {
    // 计算期望的输出
    auto expected = torch::exp(input[i]) / sum[i];
    // 断言：output[i] 应接近 expected
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, Softmin) {
  // 初始化 Softmin 模块，指定维度为 1
  Softmin m(/*dim=*/1);
  // 生成一个大小为 {2, 5} 的输入张量
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // 使用模块处理输入，得到输出
  auto output = m(input);
  // 计算输入每行的指数和
  auto sum = torch::sum(torch::exp(-input), 1);

  // 对每行进行断言
  for (const auto i : c10::irange(2)) {
    // 计算期望的输出
    auto expected = torch::exp(-input[i]) / sum[i];
    // 断言：output[i] 应接近 expected
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}
TEST_F(ModulesTest, LogSoftmax) {
  // 创建 LogSoftmax 模块，指定对第二维进行操作
  LogSoftmax m(/*dim=*/1);
  // 创建一个大小为 2x5 的浮点数张量，并按顺序填充为 0 到 9
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // 对输入张量应用 LogSoftmax 操作，得到输出张量
  auto output = m(input);
  // 计算每行的指数值总和
  auto sum = torch::sum(torch::exp(input), 1);

  // 对于每个 i 在 [0, 2) 范围内
  for (const auto i : c10::irange(2)) {
    // 计算预期输出，即对第 i 行进行 log(softmax) 操作
    auto expected = torch::log(torch::exp(input[i]) / sum[i]);
    // 断言输出的第 i 行与预期的输出相似
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, AdaptiveLogSoftmaxWithLoss) {
  {
    // 创建 AdaptiveLogSoftmaxWithLoss 模块，设置参数如下：
    // num_embeddings=8, cutoffs=[4], div_value=2.0
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 4, {2}).div_value(2.));
    // 创建一个大小为 4x8 的随机张量 x
    auto x = torch::randn({4, 8});
    // 调用 log_prob 方法，计算对输入 x 的对数概率输出
    auto logprob_out = asfm->log_prob(x);
    // 断言每行指数函数的和等于 1
    ASSERT_TRUE(
        torch::allclose(torch::exp(logprob_out).data().sum(1), torch::ones(4)));
  }
  {
    // 创建 AdaptiveLogSoftmaxWithLoss 模块，设置参数如下：
    // num_embeddings=8, cutoffs=[4, 8], div_value=2.0, head_bias=true
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 10, {4, 8})
            .div_value(2.)
            .head_bias(true));
    // 创建一个大小为 64x8 的随机张量 x
    auto x = torch::randn({64, 8});
    // 调用 log_prob 方法，计算对输入 x 的对数概率输出
    auto logprob_out = asfm->log_prob(x);
    // 调用 predict 方法，得到预测的索引值输出
    auto predict_out = asfm->predict(x);
    // 断言预测的输出与对数概率输出的最大值索引相似
    ASSERT_TRUE(torch::allclose(predict_out, logprob_out.argmax(1)));
  }
  {
    // 创建 AdaptiveLogSoftmaxWithLoss 模块，设置参数如下：
    // num_embeddings=16, cutoffs=[4, 10, 15], div_value=2.0
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(2.));
    // 创建一个大小为 2x16 的浮点数张量 x，以及一个大小为 2 的长整型张量 y
    auto x = torch::arange(100, 132, torch::kFloat).reshape({2, 16});
    auto y = torch::tensor({0, 17}, torch::kLong);
    // 对输入 x 和 y 应用 AdaptiveLogSoftmaxWithLoss 模块，得到输出结果
    auto asm_out = asfm(x, y);
    // 断言输出的大小为 [2]
    ASSERT_EQ(asm_out.output.sizes(), std::vector<int64_t>({2}));
  }
  {
    // 创建 AdaptiveLogSoftmaxWithLoss 模块，设置参数如下：
    // num_embeddings=8, cutoffs=[4], div_value=2.0
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 4, {2}).div_value(2.));
    // 创建一个大小为 4x8 的随机张量 x
    auto x = torch::randn({4, 8});
    // 调用 log_prob 方法，计算对输入 x 的对数概率输出
    auto logprob_out = asfm->log_prob(x);
    // 创建一个 NLLLoss 对象
    NLLLoss nll_loss;

    // 对于每个 v 在 [0, 4) 范围内
    for (const auto v : c10::irange(4)) {
      // 创建一个大小为 4 的长整型张量 y，其所有元素均为 v
      auto y = torch::full({4}, v, torch::kLong);
      // 对输入 x 和 y 应用 AdaptiveLogSoftmaxWithLoss 模块，得到输出结果
      auto asm_out = asfm(x, y);
      auto out = asm_out.output;
      // 创建一个浮点数张量 loss，包含 asm_out 的损失值
      auto loss = torch::tensor(asm_out.loss, torch::kFloat);
      // 计算预期的损失值
      auto expected = nll_loss->forward(logprob_out, y);

      // 断言损失值与预期的损失值相似
      ASSERT_TRUE(torch::allclose(loss, expected));
      // 断言输出值与对数概率输出中对 y 的索引值进行 gather 操作后的结果相似
      ASSERT_TRUE(torch::allclose(
          out, logprob_out.gather(1, y.unsqueeze(1)).squeeze()));
    }
  }
  {
    // 创建 AdaptiveLogSoftmaxWithLoss 模块，设置参数如下：
    // num_embeddings=16, cutoffs=[4, 10, 15], div_value=2.0
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(2.));
    // 创建一个大小为 1x16 的随机张量 x，以及一个大小为 1 的长整型张量 y
    auto x = torch::randn({1, 16});
    auto y = torch::tensor({17});
    // 对输入 x 和 y 应用 AdaptiveLogSoftmaxWithLoss 模块，得到输出结果
    auto x2 = x.squeeze(0);
    auto y2 = y.squeeze(0);
    // 断言去除批次维度后的输出与有批次维度的输出相似
    ASSERT_TRUE(
        torch::allclose(asfm(x, y).output.squeeze(0), asfm(x2, y2).output));
  }
  {
    // 创建 AdaptiveLogSoftmaxWithLossOptions，设置参数如下：
    // num_embeddings=16, cutoffs=[4, 10, 15], div_value=0.0
    auto options =
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(0.);
    // 断言创建 AdaptiveLogSoftmaxWithLoss 对象时抛出异常，异常信息包含指定的字符串
    ASSERT_THROWS_WITH(
        AdaptiveLogSoftmaxWithLoss(options),
        "div_value should not be equal to 0");

    // 创建 AdaptiveLogSoftmaxWithLossOptions，设置参数如下：
    // num_embeddings=16, cutoffs=[4, 10, 15], div_value=0.25
    options =
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(0.25);
    // 断言创建 AdaptiveLogSoftmaxWithLoss 对象成功
    ASSERT_TRUE(AdaptiveLogSoftmaxWithLoss(options));
  }
}
TEST_F(ModulesTest, Softmax2d) {
  // 创建 Softmax2d 模块实例
  Softmax2d m;
  // 创建输入张量，形状为 [1, 2, 3, 4]，包含从0到23的浮点数
  auto input = torch::arange(24, torch::kFloat).reshape({1, 2, 3, 4});
  // 将输入张量传递给 Softmax2d 模块，计算输出
  auto output = m(input);
  // 计算输入张量每个位置的指数函数，然后沿第1维求和
  auto sum = torch::sum(torch::exp(input), 1);

  // 四重循环遍历输入张量的每个元素
  for (const auto i : c10::irange(1)) {
    for (const auto j : c10::irange(2)) {
      for (const auto k : c10::irange(3)) {
        for (const auto l : c10::irange(4)) {
          // 计算当前位置的预期值，即 exp(input[i][j][k][l]) / sum[i][k][l]
          auto expected = torch::exp(input[i][j][k][l]) / sum[i][k][l];
          // 断言当前输出与预期值的接近程度
          ASSERT_TRUE(torch::allclose(output[i][j][k][l], expected));
        }
      }
    }
  }
}

TEST_F(ModulesTest, PReLU) {
  // 定义 PReLU 模块的参数数量和初始值
  const auto num_parameters = 42;
  const auto init = 0.42;

  // 创建 PReLU 模块实例，指定参数数量和初始值
  PReLU model{PReLUOptions().num_parameters(num_parameters).init(init)};

  // 断言模块权重张量的大小符合预期
  ASSERT_EQ(model->weight.sizes(), std::vector<int64_t>({num_parameters}));
  // 断言模块权重张量与全初始化值的接近程度
  ASSERT_TRUE(torch::allclose(model->weight, torch::full(num_parameters, init)));

  // 创建随机输入张量 x，形状为 [100, num_parameters]
  const auto x = torch::rand({100, num_parameters}) * 200 - 100;
  // 将输入张量传递给 PReLU 模块，计算输出张量 y
  const auto y = model(x);
  // 计算输出张量 y 的总和
  const auto s = y.sum();

  // 对总和张量进行反向传播
  s.backward();
  // 断言总和张量的维度为0
  ASSERT_EQ(s.ndimension(), 0);

  // 断言输出张量 y 的维度与输入张量 x 的维度相同
  ASSERT_EQ(y.ndimension(), x.ndimension());
  // 断言输出张量 y 的形状与输入张量 x 的形状相同
  ASSERT_EQ(y.sizes(), x.sizes());
  // 计算预期输出张量 y_exp
  const auto y_exp = (x < 0) * model->weight * x + (x >= 0) * x;
  // 断言当前输出张量 y 与预期输出张量 y_exp 的接近程度
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, ReLU) {
  // 遍历是否使用原地操作的布尔值
  for (const auto inplace : {false, true}) {
    // 定义尺寸大小
    const auto size = 3;
    // 创建 ReLU 模块实例
    ReLU model(inplace);
    // 创建等间隔张量 x，范围为 [-10.0, 10.0)，形状为 [size, size, size]
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    // 如果不使用原地操作，则需要设置张量 x 的梯度追踪
    if (!inplace) {
      x.requires_grad_(true);
    }
    // 克隆原始张量 x_orig
    auto x_orig = x.clone();
    // 将输入张量传递给 ReLU 模块，计算输出张量 y
    auto y = model(x);
    // 计算输出张量 y 的总和
    torch::Tensor s = y.sum();

    // 断言总和张量的维度为0
    ASSERT_EQ(s.ndimension(), 0);
    // 断言输出张量 y 的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言输出张量 y 的形状与预期形状 [size, size, size] 相同
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 计算预期输出张量 y_exp
    auto y_exp = (x_orig < 0) * 0 + (x_orig >= 0) * x_orig;
    // 断言当前输出张量 y 与预期输出张量 y_exp 的接近程度
    ASSERT_TRUE(torch::allclose(y, y_exp));
    // 如果使用原地操作，则断言输入张量 x 与预期输出张量 y_exp 的接近程度
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    } else {
      // 否则进行反向传播
      s.backward();
    }
  }
}

TEST_F(ModulesTest, ReLU6) {
  // 遍历是否使用原地操作的布尔值
  for (const auto inplace : {false, true}) {
    // 定义尺寸大小
    const auto size = 3;
    // 创建 ReLU6 模块实例
    ReLU6 model(inplace);
    // 创建等间隔张量 x，范围为 [-10.0, 10.0)，形状为 [size, size, size]
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    // 如果不使用原地操作，则需要设置张量 x 的梯度追踪
    if (!inplace) {
      x.requires_grad_(true);
    }
    // 克隆原始张量 x_orig
    auto x_orig = x.clone();
    // 将输入张量传递给 ReLU6 模块，计算输出张量 y
    auto y = model(x);
    // 计算输出张量 y 的总和
    torch::Tensor s = y.sum();

    // 断言总和张量的维度为0
    ASSERT_EQ(s.ndimension(), 0);
    // 断言输出张量 y 的维度为3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言输出张量 y 的形状与预期形状 [size, size, size] 相同
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 计算预期输出张量 y_exp
    auto y_exp = (x_orig < 0) * 0 + ((x_orig >= 0) * (x_orig <= 6)) * x_orig +
        (x_orig > 6) * 6;
    // 断言当前输出张量 y 与预期输出张量 y_exp 的接近程度
    ASSERT_TRUE(torch::allclose(y, y_exp));
    // 如果使用原地操作，则断言输入张量 x 与预期输出张量 y_exp 的接近程度
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    } else {
      // 否则进行反向传播
      s.backward();
    }
  }
}

TEST_F(ModulesTest, RReLU) {
  // 定义尺寸大小
  const auto size = 3;
  // 遍历 RReLU 模块的下界值
  for (const auto lower : {0.01, 0.1, 0.2}) {
    # 对于不同的上界值进行迭代，如0.3、0.4、0.5
    for (const auto upper : {0.3, 0.4, 0.5}) {
      # 对于是否原地操作进行迭代，即false和true两种情况
      for (const auto inplace : {false, true}) {
        # 对于张量类型进行迭代，包括torch::kFloat和torch::kBFloat16
        for (const auto type : {torch::kFloat, torch::kBFloat16}) {
          # 根据给定的参数创建一个RReLU模型
          RReLU model{
              RReLUOptions().lower(lower).upper(upper).inplace(inplace)};
          # 创建一个大小为size*size*size的线性空间张量，并转换为指定的类型
          auto x = torch::linspace(-10.0, 10.0, size * size * size).to(type);
          # 调整张量的形状为size x size x size
          x.resize_({size, size, size});
          # 如果不是原地操作，则设置张量需要梯度追踪
          if (!inplace) {
            x.requires_grad_(true);
          }
          # 克隆原始的输入张量x
          auto x_orig = x.clone();
          # 对输入张量x应用RReLU模型，得到输出张量y
          auto y = model(x);
          # 计算输出张量y的总和
          torch::Tensor s = y.sum();

          # 使用断言确保张量s的维度为0
          ASSERT_EQ(s.ndimension(), 0);
          # 使用断言确保输出张量y的维度为3
          ASSERT_EQ(y.ndimension(), 3);
          # 使用断言确保输出张量y的尺寸为[size, size, size]
          ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
          # 计算张量z，根据RReLU函数的输出计算条件张量
          auto z =
              ((x_orig >= 0) * (x_orig == y) +
               (x_orig < 0) * (y >= x_orig * upper) * (y <= lower * x_orig)) *
              1.0;
          # 使用断言确保张量z与全1张量的每个元素都接近
          ASSERT_TRUE(torch::allclose(z, torch::ones_like(z)));
          # 如果是原地操作，使用断言确保张量x与输出张量y的每个元素都接近
          if (inplace) {
            ASSERT_TRUE(torch::allclose(x, y));
          } else {
            # 否则对总和张量s进行反向传播
            s.backward();
          }
        }
      }
    }
// 在 ModulesTest 测试套件中，测试 CELU 激活函数
TEST_F(ModulesTest, CELU) {
  // 定义输入张量的大小
  const auto size = 3;
  // 针对是否原地操作和不同的 alpha 值进行循环测试
  for (const auto inplace : {false, true}) {
    for (const auto alpha : {0.42, 1.0, 4.2, 42.42}) {
      // 创建 CELU 模型对象，根据给定的 alpha 和 inplace 参数
      CELU model{CELUOptions().alpha(alpha).inplace(inplace)};
      // 生成一个均匀分布的输入张量 x
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      // 重新调整张量 x 的形状为 size x size x size
      x.resize_({size, size, size});
      // 如果不是原地操作，则需要设置 x 的梯度跟踪
      if (!inplace) {
        x.requires_grad_(true);
      }
      // 克隆原始的输入张量 x
      auto x_orig = x.clone();
      // 对模型进行前向传播得到输出 y
      auto y = model(x);
      // 对输出 y 进行求和操作
      torch::Tensor s = y.sum();

      // 断言输出 y 的维度为 0
      ASSERT_EQ(s.ndimension(), 0);
      // 断言输出 y 的维度为 3
      ASSERT_EQ(y.ndimension(), 3);
      // 断言输出 y 的尺寸为 [size, size, size]
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      // 计算预期的输出 y_exp，根据 CELU 的定义计算
      auto y_exp = torch::max(torch::zeros_like(x_orig), x_orig) +
          torch::min(torch::zeros_like(x_orig),
                     alpha * (torch::exp(x_orig / alpha) - 1.0));
      // 断言模型的输出 y 与预期的 y_exp 接近
      ASSERT_TRUE(torch::allclose(y, y_exp));
      // 如果是原地操作，则断言输入 x 与预期的 y_exp 接近
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
      } else {
        // 否则执行反向传播
        s.backward();
      }
    }
  }
}

// 在 ModulesTest 测试套件中，测试 GLU 激活函数
TEST_F(ModulesTest, GLU) {
  // 定义 GLU 模型的维度
  int64_t dim = 1;
  // 创建 GLU 模型对象
  GLU model(dim);
  // 生成一个随机输入张量 input，并要求跟踪其梯度
  auto input = torch::randn({4, 2}, torch::requires_grad());
  // 对输入张量进行前向传播得到输出 output
  auto output = model->forward(input);
  // 获取输入张量 input 在指定维度 dim 上的尺寸
  auto input_size = input.sizes()[dim] / 2;
  // 分别获取输入张量的前半部分和后半部分
  auto first_half = input.narrow(dim, 0, input_size);
  auto second_half = input.narrow(dim, input_size, input_size);
  // 计算预期的输出 expected，根据 GLU 的定义计算
  auto expected = first_half * torch::sigmoid(second_half);
  // 对输出 output 进行求和操作
  auto s = output.sum();
  // 执行反向传播
  s.backward();

  // 断言输出 s 的维度为 0
  ASSERT_EQ(s.ndimension(), 0);
  // 断言模型的输出 output 与预期的 expected 接近
  ASSERT_TRUE(output.allclose(expected));

  // 创建具有默认选项的 GLU 模型对象
  GLU model_default_options;
  // 断言使用默认选项的 GLU 模型对输入 input 的前向传播结果与预期的 expected 接近
  ASSERT_TRUE(model_default_options->forward(input).allclose(expected));
}

// 在 ModulesTest 测试套件中，测试 GELU 激活函数
TEST_F(ModulesTest, GELU) {
  // 创建 GELU 模型对象，使用指定的近似方法
  GELU model(GELUOptions().approximate("none"));
  // 生成一个线性空间的输入张量 x
  const auto x = torch::linspace(-3.0, 3.0, 100);
  // 计算预期的输出 y_exp，根据 GELU 的定义计算
  const auto y_exp = x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
  // 对输入张量 x 进行前向传播得到输出 y
  const auto y = model(x);
  // 断言模型的输出 y 与预期的 y_exp 接近，使用指定的容差
  ASSERT_TRUE(torch::allclose(y, y_exp, 1.4e-06, 1e-05));
}

// 在 ModulesTest 测试套件中，测试使用 Tanh 近似的 GELU 激活函数
TEST_F(ModulesTest, TanhGELU) {
  // 创建 GELU 模型对象，使用 Tanh 近似方法
  GELU model(GELUOptions().approximate("tanh"));
  // 生成一个线性空间的输入张量 x
  const auto x = torch::linspace(-3.0, 3.0, 100);
  // 计算内部表达式 inner，根据使用 Tanh 近似的 GELU 定义计算
  const auto inner = std::sqrt(2 / M_PI) * (x + 0.044715 * x.pow(3.0));
  // 计算预期的输出 y_exp，根据使用 Tanh 近似的 GELU 定义计算
  const auto y_exp = 0.5 * x * (1.0 + inner.tanh());
  // 对输入张量 x 进行前向传播得到输出 y
  const auto y = model(x);
  // 断言模型的输出 y 与预期的 y_exp 接近，使用指定的容差
  ASSERT_TRUE(torch::allclose(y, y_exp, 1.4e-06, 1e-05));
}

// 禁止使用 NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) 注释，测试 Mish 激活函数
TEST_F(ModulesTest, Mish) {
  // 创建 Mish 模型对象
  Mish model;
  // 生成一个服从标准正态分布的随机输入张量 x
  auto x = torch::randn(100) * 10;
  // 计算预期的输出 y_exp，根据 Mish 的定义计算
  auto y_exp = x * x.exp().log1p().tanh();
  // 对输入张量 x 进行前向传播得到输出 y
  auto y = model(x);

  // 断言模型的输出 y 与预期的 y_exp 接近
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// 在 ModulesTest 测试套件中，测试 Sigmoid 激活函数
TEST_F(ModulesTest, Sigmoid) {
  // 创建 Sigmoid 模型对象
  Sigmoid model;
  // 生成一个服从标准正态分布的随机输入张量 x
  auto x = torch::randn(100) * 10;
  // 计算预期的输出 y_exp，根据 Sigmoid 的定义计算
  auto y_exp = 1 / (1 + torch::exp(-x));
  // 对输入张量 x 进行前向传播得到输出 y
  auto y = model(x);

  // 断言模型的输出 y 与预期的 y_exp 接近
  ASSERT_TRUE(torch::allclose(y, y_exp));
}
TEST_F(ModulesTest, PixelShuffle) {
  // 创建 PixelShuffle 模块，指定放大倍数为 2
  PixelShuffle module(/*upscale_factor=*/2);
  // 创建输入张量 x，包含特定的浮点数值
  auto x = torch::tensor(
      {{{{-17, 19}, {-1, 2}},
        {{7, 14}, {-3, 1}},
        {{0, -2}, {-12, 14}},
        {{-15, 0}, {-3, 9}}}},
      torch::kFloat);
  // 创建期望的输出张量 y_exp
  auto y_exp = torch::tensor(
      {{{{-17, 7, 19, 14}, {0, -15, -2, 0}, {-1, -3, 2, 1}, {-12, -3, 14, 9}}}},
      torch::kFloat);
  // 对输入张量应用 PixelShuffle 模块，获得输出张量 y
  auto y = module(x);

  // 断言输出张量 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言输出张量 y 的尺寸为 {1, 1, 4, 4}
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 4, 4}));
  // 断言输出张量 y 与期望输出张量 y_exp 在数值上的接近程度
  ASSERT_TRUE(y.allclose(y_exp));
}

TEST_F(ModulesTest, PixelUnshuffle) {
  // 创建 PixelUnshuffle 模块，指定缩小倍数为 2
  PixelUnshuffle module(/*downscale_factor=*/2);
  // 创建输入张量 x，包含特定的浮点数值
  auto x = torch::tensor(
      {{{{-17, 7, 19, 14}, {0, -15, -2, 0}, {-1, -3, 2, 1}, {-12, -3, 14, 9}}}},
      torch::kFloat);
  // 创建期望的输出张量 y_exp
  auto y_exp = torch::tensor(
      {{{{-17, 19}, {-1, 2}},
        {{7, 14}, {-3, 1}},
        {{0, -2}, {-12, 14}},
        {{-15, 0}, {-3, 9}}}},
      torch::kFloat);
  // 对输入张量应用 PixelUnshuffle 模块，获得输出张量 y
  auto y = module(x);

  // 断言输出张量 y 的维度为 4
  ASSERT_EQ(y.ndimension(), 4);
  // 断言输出张量 y 的尺寸为 {1, 4, 2, 2}
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 4, 2, 2}));
  // 断言输出张量 y 与期望输出张量 y_exp 在数值上的接近程度
  ASSERT_TRUE(y.allclose(y_exp));
}

TEST_F(ModulesTest, Softplus) {
  // 定义 Softplus 模型，设置 beta 和 threshold 参数
  const auto size = 3;
  for (const auto beta : {0.5, 1.0, 2.0}) {
    for (const auto threshold : {1.0, 3.0, 5.0}) {
      Softplus model{SoftplusOptions().beta(beta).threshold(threshold)};
      // 创建输入张量 x，以特定步长和尺寸生成
      auto x = torch::linspace(-3.0, 3.0, 61);
      x.resize_({size, size, size});
      // 计算期望的输出张量 y_exp
      auto y_exp =
          (x <= threshold) * torch::log(1 + torch::exp(x * beta)) / beta +
          (x > threshold) * x;
      // 对输入张量应用 Softplus 模块，获得输出张量 y
      auto y = model(x);

      // 断言输出张量 y 的维度为 3
      ASSERT_EQ(y.ndimension(), 3);
      // 断言输出张量 y 的尺寸为 {size, size, size}
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      // 断言输出张量 y 与期望输出张量 y_exp 在数值上的接近程度
      ASSERT_TRUE(torch::allclose(y, y_exp));
    }
  }
}

TEST_F(ModulesTest, Softshrink) {
  // 定义 Softshrink 模型，设置 lambda 参数
  const auto size = 3;
  for (const auto lambda : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    Softshrink model{/*lambda=*/lambda};
    // 创建输入张量 x，以特定步长和尺寸生成，并设置梯度跟踪
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    // 对输入张量应用 Softshrink 模块，获得输出张量 y
    auto y = model(x);
    // 计算输出张量 y 的总和并执行反向传播
    torch::Tensor s = y.sum();

    s.backward();
    // 断言输出张量 s 的维度为 0
    ASSERT_EQ(s.ndimension(), 0);

    // 断言输出张量 y 的维度为 3
    ASSERT_EQ(y.ndimension(), 3);
    // 断言输出张量 y 的尺寸为 {size, size, size}
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    // 计算期望的输出张量 y_exp
    auto y_exp = (x < -lambda) * (x + lambda) + (x > lambda) * (x - lambda);
    // 断言输出张量 y 与期望输出张量 y_exp 在数值上的接近程度
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, Softsign) {
  // 创建 Softsign 模型
  Softsign model;
  // 创建输入张量 x，包含从标准正态分布中随机生成的值
  auto x = torch::randn(100) * 10;
  // 计算期望的输出张量 y_exp
  auto y_exp = x / (1 + x.abs());
  // 对输入张量应用 Softsign 模块，获得输出张量 y
  auto y = model(x);

  // 断言输出张量 y 与期望输出张量 y_exp 在数值上的接近程度
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Tanh) {
  // 创建 Tanh 模型
  Tanh model;
  // 创建输入张量 x，包含从标准正态分布中随机生成的值
  auto x = torch::randn(100) * 10;
  // 计算期望的输出张量 y_exp
  auto y_exp = (x.exp() - (-x).exp()) / (x.exp() + (-x).exp());
  // 对输入张量应用 Tanh 模块，获得输出张量 y
  auto y = model(x);

  // 断言输出张量 y 与期望输出张量 y_exp 在数值上的接近程度
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Tanhshrink) {
  // 创建 Tanhshrink 模型
  Tanhshrink model;
  // 创建输入张量 x，包含从标准正态分布中随机生成的值
  auto x = torch::randn(100) * 10;
  // 计算期望的输出张量 y_exp
  auto y_exp = x - x.tanh();
  // 对输入张量应用 Tanhshrink 模块，获得输出张量 y
  auto y = model(x);

  // 断言输出张量 y 与期望输出张量 y_exp 在数值上的接近程度
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Threshold) {
  // 定义阈值模块测试用例，设置尺寸为 3
  const auto size = 3;
  for (const auto threshold : {0.5, 1.0, 2.0}) {
    // 遍历不同的阈值设定和是否原地操作的组合
    for (const auto value : {0.5, 1.0, 2.0}) {
      for (const auto inplace : {false, true}) {
        // 使用给定的阈值和参数创建阈值模型对象
        Threshold model{ThresholdOptions(threshold, value).inplace(inplace)};
        // 生成一个从-3.0到3.0的等间距张量，包含61个元素
        auto x = torch::linspace(-3.0, 3.0, 61);
        // 将张量x调整为大小为size*size*size的三维张量
        x.resize_({size, size, size});
        // 创建原始输入数据的副本
        auto x_orig = x.clone();
        // 根据阈值threshold计算期望的输出结果
        auto y_exp =
            (x_orig <= threshold) * value + (x_orig > threshold) * x_orig;
        // 使用阈值模型处理输入张量x，得到模型输出y
        auto y = model(x);

        // 断言模型输出y的维度为3
        ASSERT_EQ(y.ndimension(), 3);
        // 断言模型输出y的大小为[size, size, size]
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        // 断言模型输出y与期望的输出y_exp在数值上的接近程度
        ASSERT_TRUE(torch::allclose(y, y_exp));
        // 如果采用了原地操作，断言输入张量x与期望的输出y_exp在数值上的接近程度
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
{
  // 定义一个单元测试用例，测试1维上采样
  Upsample model(UpsampleOptions()
                     .size(std::vector<int64_t>({4}))  // 设置上采样尺寸为 [4]
                     .mode(torch::kNearest));  // 使用最近邻插值方式
  auto input = torch::ones({1, 1, 2}, torch::requires_grad());  // 创建一个全为1的输入张量 [1, 1, 2]
  auto output = model->forward(input);  // 执行模型前向传播
  auto expected = torch::ones({1, 1, 4});  // 创建预期输出张量 [1, 1, 4]
  auto s = output.sum();  // 计算输出张量的所有元素之和
  s.backward();  // 对和张量进行反向传播

  ASSERT_EQ(s.ndimension(), 0);  // 断言和张量是0维的（标量）
  ASSERT_TRUE(output.allclose(expected));  // 断言输出张量与预期张量非常接近
}
{
  for (const auto align_corners : {true, false}) {
    // 测试浮点型尺度因子的上采样和下采样
    for (const auto scale_factor : {0.5, 1.5, 2.0}) {
      Upsample model(UpsampleOptions()
                         .scale_factor(std::vector<double>({scale_factor}))  // 设置尺度因子
                         .mode(torch::kLinear)  // 使用线性插值方式
                         .align_corners(align_corners));  // 设置是否对齐角点

      auto input = torch::ones({1, 1, 2}, torch::requires_grad());  // 创建一个全为1的输入张量 [1, 1, 2]
      auto output = model->forward(input);  // 执行模型前向传播
      auto expected_size =
          static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));  // 计算预期输出尺寸
      auto expected = torch::ones({1, 1, expected_size});  // 创建预期输出张量
      auto s = output.sum();  // 计算输出张量的所有元素之和
      s.backward();  // 对和张量进行反向传播

      ASSERT_EQ(s.ndimension(), 0);  // 断言和张量是0维的（标量）
      ASSERT_TRUE(output.allclose(expected));  // 断言输出张量与预期张量非常接近
    }
  }
}
{
  // 线性（1D）上采样的空间不变性测试
  Upsample model(UpsampleOptions()
                     .scale_factor(std::vector<double>({3}))  // 设置尺度因子为 [3]
                     .mode(torch::kLinear)  // 使用线性插值方式
                     .align_corners(false));  // 不对齐角点

  auto input = torch::zeros({1, 1, 9});  // 创建一个全为0的输入张量 [1, 1, 9]
  input.narrow(2, 0, 4).normal_();  // 在输入张量的第2维上的前4个元素上施加正态分布随机数
  auto output = model->forward(input);  // 执行模型前向传播
  auto expected = model->forward(input.narrow(2, 0, 5));  // 对输入的一部分应用模型的前向传播

  ASSERT_TRUE(torch::allclose(output.narrow(2, 0, 15), expected));  // 断言输出张量的一部分与预期张量非常接近
}
    // 对于两种插值方式（双线性和双三次），以及两种是否对齐角点的方式（是和否）进行测试
    for (const auto align_corners : {true, false}) {
      // 测试浮点缩放因子的上采样和下采样
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        // 创建一个 Upsample 模型，配置插值方式、缩放因子和是否对齐角点
        Upsample model(
            UpsampleOptions()
                .scale_factor(std::vector<double>({scale_factor, scale_factor}))
                .mode(torch::kBilinear)  // 使用双线性插值模式
                .align_corners(align_corners));  // 设置是否对齐角点

        // 创建一个输入张量，全为1，大小为 [1, 1, 2, 2]，并设置需要梯度计算
        auto input = torch::ones({1, 1, 2, 2}, torch::requires_grad());
        
        // 进行前向传播得到输出
        auto output = model->forward(input);

        // 根据输入大小和缩放因子计算期望的输出大小
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));

        // 创建一个期望的输出张量，全为1，大小为 [1, 1, expected_size, expected_size]
        auto expected = torch::ones({1, 1, expected_size, expected_size});

        // 对输出张量进行求和
        auto s = output.sum();

        // 执行反向传播
        s.backward();

        // 使用断言验证条件：输出张量的维度为0（即为标量）
        ASSERT_EQ(s.ndimension(), 0);

        // 使用断言验证条件：输出张量与期望输出张量在误差允许范围内相等
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    // 对于两种插值方式（双线性和双三次），以及两种是否对齐角点的方式（是和否）进行测试
    for (const auto align_corners : {true, false}) {
      // 测试浮点缩放因子的上采样和下采样
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        // 创建一个 Upsample 模型，配置插值方式、缩放因子和是否对齐角点
        Upsample model(
            UpsampleOptions()
                .scale_factor(std::vector<double>({scale_factor, scale_factor}))
                .mode(torch::kBicubic)  // 使用双三次插值模式
                .align_corners(align_corners));  // 设置是否对齐角点

        // 创建一个输入张量，全为1，大小为 [1, 1, 2, 2]，并设置需要梯度计算
        auto input = torch::ones({1, 1, 2, 2}, torch::requires_grad());

        // 进行前向传播得到输出
        auto output = model->forward(input);

        // 根据输入大小和缩放因子计算期望的输出大小
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));

        // 创建一个期望的输出张量，全为1，大小为 [1, 1, expected_size, expected_size]
        auto expected = torch::ones({1, 1, expected_size, expected_size});

        // 对输出张量进行求和
        auto s = output.sum();

        // 执行反向传播
        s.backward();

        // 使用断言验证条件：输出张量的维度为0（即为标量）
        ASSERT_EQ(s.ndimension(), 0);

        // 使用断言验证条件：输出张量与期望输出张量在误差允许范围内相等
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
TEST_F(ModulesTest, Upsampling3D) {
  {
    // 创建一个 Upsample 模型，设置大小为 4x4x4，使用最近邻插值模式
    Upsample model(UpsampleOptions()
                       .size(std::vector<int64_t>({4, 4, 4}))
                       .mode(torch::kNearest));
    // 创建一个输入张量，全为1，形状为 [1, 1, 2, 2, 2]
    auto input = torch::ones({1, 1, 2, 2, 2}, torch::requires_grad());
    // 将输入张量传递给模型进行前向计算
    auto output = model->forward(input);
    // 创建一个预期输出张量，全为1，形状为 [1, 1, 4, 4, 4]
    auto expected = torch::ones({1, 1, 4, 4, 4});
    // 计算输出张量元素的和
    auto s = output.sum();
    // 反向传播求梯度
    s.backward();

    // 断言输出张量的维度为0
    ASSERT_EQ(s.ndimension(), 0);
    // 断言输出张量与预期张量的所有元素是否近似相等
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 对于每个 align_corners 值（true 和 false）
    for (const auto align_corners : {true, false}) {
      // 测试浮点数比例因子的上采样和下采样
      // 对于每个 scale_factor 值（0.5, 1.5, 2.0）
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        // 创建一个 Upsample 模型，设置 scale_factor 为 [scale_factor, scale_factor, scale_factor]，使用三线性插值模式
        // align_corners 根据当前循环值设置是否开启
        Upsample model(UpsampleOptions()
                           .scale_factor(std::vector<double>(
                               {scale_factor, scale_factor, scale_factor}))
                           .mode(torch::kTrilinear)
                           .align_corners(align_corners));
        // 创建一个输入张量，全为1，形状为 [1, 1, 2, 2, 2]
        auto input = torch::ones({1, 1, 2, 2, 2}, torch::requires_grad());
        // 将输入张量传递给模型进行前向计算
        auto output = model->forward(input);
        // 计算预期输出张量的大小，根据当前 scale_factor 计算
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        // 创建一个预期输出张量，全为1，形状根据当前 scale_factor 计算得出
        auto expected =
            torch::ones({1, 1, expected_size, expected_size, expected_size});
        // 计算输出张量元素的和
        auto s = output.sum();
        // 反向传播求梯度
        s.backward();

        // 断言输出张量的维度为0
        ASSERT_EQ(s.ndimension(), 0);
        // 断言输出张量与预期张量的所有元素是否近似相等
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
}

TEST_F(ModulesTest, CTCLoss) {
  // 创建一个 CTCLoss 对象，设置 reduction 方式为 none
  CTCLoss loss{CTCLossOptions().reduction(torch::kNone)};
  // 创建目标长度张量，元素全为0
  const auto target_lengths = torch::tensor({0, 0, 0});
  // 创建输入长度张量，元素全为50
  const auto input_lengths = torch::tensor({50, 50, 50});
  // 创建目标张量，随机整数值在 [1, 15] 之间
  const auto targets =
      torch::randint(1, 15, at::IntArrayRef({0}), torch::kLong);
  // 创建对数概率张量，形状为 [50, 3, 15]
  const auto log_probs =
      torch::randn({50, 3, 15}, torch::kDouble).log_softmax(2);
  // 计算模型的输出，传入对数概率、目标、输入长度和目标长度
  const auto output =
      loss->forward(log_probs, targets, input_lengths, target_lengths);
  // 断言输出张量所有元素是否非负
  ASSERT_TRUE(output.ge(0).all().item<bool>());
  // 断言对数概率张量按第0维度切片后的和，形状与输出张量相同，是否近似等于输出张量
  ASSERT_TRUE(torch::allclose(
      -log_probs.sum(0).slice(1, 0, 1).view_as(output), output));
}

TEST_F(ModulesTest, PoissonNLLLoss) {
  // 创建输入张量，值为 [0.5, 1.5, 2.5]
  const auto input = torch::tensor({0.5, 1.5, 2.5});
  // 创建目标张量，值为 [1., 2., 3.]
  const auto target = torch::tensor({1., 2., 3.});
  // 计算每个分量的损失值
  const auto component_wise_loss = torch::exp(input) - target * input;
  {
    // 创建一个 PoissonNLLLoss 对象，设置 reduction 方式为 none
    PoissonNLLLoss loss{PoissonNLLLossOptions().reduction(torch::kNone)};
    // 断言每个分量的损失值是否近似等于 PoissonNLLLoss 的前向计算结果
    ASSERT_TRUE(
        torch::allclose(component_wise_loss, loss->forward(input, target)));
  }
  {
    // 创建一个 PoissonNLLLoss 对象，设置 reduction 方式为 sum
    PoissonNLLLoss loss{PoissonNLLLossOptions().reduction(torch::kSum)};
    // 断言损失值的总和是否近似等于 PoissonNLLLoss 的前向计算结果
    ASSERT_TRUE(torch::allclose(
        torch::sum(component_wise_loss), loss->forward(input, target)));
  }
  {
    // 创建一个 PoissonNLLLoss 对象，设置 reduction 方式为 mean
    PoissonNLLLoss loss{PoissonNLLLossOptions().reduction(torch::kMean)};
    // 断言损失值的平均值是否近似等于 PoissonNLLLoss 的前向计算结果
    ASSERT_TRUE(torch::allclose(
        torch::mean(component_wise_loss), loss->forward(input, target)));
  }
}

TEST_F(ModulesTest, MarginRankingLoss) {
  {
    // 创建一个 MarginRankingLoss 对象
    MarginRankingLoss loss;
    // 创建两个输入张量，元素为服从标准正态分布的随机数乘以10
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    // 其余部分需要根据具体代码补充注释，省略部分...
    {
        // 创建一个包含15个随机数并取其符号的张量作为目标值
        const auto target = torch::randn(15).sign();
        // 断言：检查计算得到的损失与预期损失之间的接近程度
        ASSERT_TRUE(torch::allclose(
            // 调用损失函数对象 loss 的 forward 方法计算损失值
            loss->forward(input1, input2, target),
            // 计算预期的损失值：-target * (input1 - input2) 的非负部分的均值
            (-target * (input1 - input2)).clamp(0).mean()));
    }
    {
        // 创建一个 MarginRankingLoss 对象，设置 margin 为 0.5，设置损失的归约方式为求和
        MarginRankingLoss loss{
            MarginRankingLossOptions().margin(0.5).reduction(torch::kSum)};
        // 创建两个包含15个随机数的张量作为输入
        const auto input1 = torch::randn(15) * 10;
        const auto input2 = torch::randn(15) * 10;
        // 创建一个包含15个随机数并取其符号的张量作为目标值
        const auto target = torch::randn(15).sign();
        // 定义 margin 的值为 0.5
        const auto margin = 0.5;
        // 断言：检查计算得到的损失与预期损失之间的接近程度
        ASSERT_TRUE(torch::allclose(
            // 调用损失函数对象 loss 的 forward 方法计算损失值
            loss->forward(input1, input2, target),
            // 计算预期的损失值：-target * (input1 - input2) + margin 的非负部分的和
            (-target * (input1 - input2) + margin).clamp(0).sum()));
    }
    {
        // 创建一个 MarginRankingLoss 对象，设置 margin 为 0.5，设置损失的归约方式为求均值
        MarginRankingLoss loss{
            MarginRankingLossOptions().margin(0.5).reduction(torch::kMean)};
        // 创建两个包含15个随机数的张量作为输入
        const auto input1 = torch::randn(15) * 10;
        const auto input2 = torch::randn(15) * 10;
        // 创建一个包含15个随机数并取其符号的张量作为目标值
        const auto target = torch::randn(15).sign();
        // 定义 margin 的值为 0.5
        const auto margin = 0.5;
        // 断言：检查计算得到的损失与预期损失之间的接近程度
        ASSERT_TRUE(torch::allclose(
            // 调用损失函数对象 loss 的 forward 方法计算损失值
            loss->forward(input1, input2, target),
            // 计算预期的损失值：-target * (input1 - input2) + margin 的非负部分的均值
            (-target * (input1 - input2) + margin).clamp(0).mean()));
    }
}

TEST_F(ModulesTest, BCEWithLogitsLoss) {
  { // test BCE with logits raises if target and input are different size
    {
      // 生成一个大小为5的随机张量作为目标值
      const auto target = torch::rand(5);
      // 生成一个大小为[5, 1]的随机张量作为输入值
      const auto input = torch::rand({5, 1});
      // 断言损失函数在输入和目标值大小不同时抛出异常
      ASSERT_THROWS_WITH(
          BCEWithLogitsLoss()(input, target), "must be the same as input size");
    }
    {
      // 生成一个大小为[5, 1]的随机张量作为目标值
      const auto target = torch::rand({5, 1});
      // 生成一个大小为5的随机张量作为输入值
      const auto input = torch::rand(5);
      // 断言损失函数在输入和目标值大小不同时抛出异常
      ASSERT_THROWS_WITH(
          BCEWithLogitsLoss()(input, target), "must be the same as input size");
    }
  }
  { // test BCE with logits gives same result as sigmoid and bce loss
    auto sigmoid = Sigmoid();

    // 生成一个大小为[64, 4]的随机目标值张量
    auto target = torch::rand({64, 4});
    // 生成一个大小为[64, 4]的随机输出值张量，并减去0.5
    auto output = torch::rand({64, 4}) - 0.5;

    // 断言 BCEWithLogitsLoss 的计算结果与 sigmoid 和 BCELoss 结果的计算结果相似
    ASSERT_TRUE(torch::allclose(
        BCEWithLogitsLoss()(output, target),
        BCELoss()(sigmoid(output), target)));

    // 生成一个大小为4的随机权重张量
    auto weight = torch::rand(4);
    // 断言 BCEWithLogitsLoss 在使用权重时与 BCELoss 使用相同设置时结果相似
    ASSERT_TRUE(torch::allclose(
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
            output, target),
        BCELoss(BCELossOptions().weight(weight))(sigmoid(output), target)));

    // 将目标值设置为全零的大小为[4, 1]的张量，输出值设置为填充为-100的大小为[4, 1]的张量
    target = torch::zeros({4, 1}, torch::kFloat);
    output = torch::empty({4, 1}, torch::kFloat).fill_(-100);

    // 断言 BCEWithLogitsLoss 的计算结果与 sigmoid 和 BCELoss 结果的计算结果相似
    ASSERT_TRUE(torch::allclose(
        BCEWithLogitsLoss()(output, target),
        BCELoss()(sigmoid(output), target)));

    // 断言 BCEWithLogitsLoss 在设置为无约简时与 BCELoss 在相同设置下的结果相似
    ASSERT_TRUE(torch::allclose(
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().reduction(torch::kNone))(
            output, target),
        BCELoss(BCELossOptions().reduction(torch::kNone))(
            sigmoid(output), target)));

    // 生成一个大小为[1]的随机权重张量
    weight = torch::rand({1}, torch::kFloat);
    // 断言 BCEWithLogitsLoss 在使用权重时与 BCELoss 使用相同设置时结果相似
    ASSERT_TRUE(torch::allclose(
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
            output, target),
        BCELoss(BCELossOptions().weight(weight))(sigmoid(output), target)));
  }
  { // test BCE with logits has correct grad at zero
    // 生成一个大小为[3, 1]的全零张量，并声明为需要梯度计算
    const auto output = torch::zeros({3, 1}, torch::requires_grad());
    // 生成一个大小为[3, 1]的全零张量作为目标值
    const auto target = torch::zeros({3, 1});
    // 断言在输出值为全零时，BCEWithLogitsLoss 的梯度计算结果与期望的梯度值相似
    BCEWithLogitsLoss(BCEWithLogitsLossOptions().reduction(torch::kSum))(
        output, target)
        .backward();
    const auto expected_grad = torch::empty({3, 1}).fill_(0.5);
    ASSERT_TRUE(torch::allclose(output.grad(), expected_grad));
  }
  { // test BCE with logits broadcasts weights
    // 生成一个大小为[16, 4]的随机目标值张量
    const auto target = torch::rand({16, 4});
    // 生成一个大小为[16, 4]的随机输出值张量，并减去0.5
    const auto output = torch::rand({16, 4}) - 0.5;

    // 生成一个大小为4的随机权重张量
    auto weight = torch::rand(4);
    // 计算使用权重的 BCEWithLogitsLoss 结果
    auto out1 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);

    // 将权重扩展为大小为[16, 4]并确保连续性
    weight = weight.expand({16, 4}).contiguous();
    // 计算扩展权重后的 BCEWithLogitsLoss 结果
    auto out2 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);

    // 断言两种计算方式得到的结果相似
    ASSERT_TRUE(torch::allclose(out1, out2));

    // 生成一个大小为[16, 1]的随机权重张量
    weight = torch::rand({16, 1});
    // 计算使用扩展权重的 BCEWithLogitsLoss 结果
    out1 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);

    // 将权重扩展为大小为[16, 4]并确保连续性
    weight = weight.expand({16, 4}).contiguous();
  { // test BCEWithLogitsLoss with default options gives the same result
    // 创建一个64x4大小的随机目标张量
    const auto target = torch::rand({64, 4});
    // 创建一个64x4大小的随机输出张量，减去0.5以调整范围
    const auto output = torch::rand({64, 4}) - 0.5;
    // 使用默认权重参数创建BCEWithLogitsLoss对象，并计算损失
    const auto out1 = BCEWithLogitsLoss()(output, target);
    // 使用设定了权重参数的BCEWithLogitsLoss对象，再次计算损失
    const auto out2 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);
    
    // 断言两次计算的损失结果近似相等
    ASSERT_TRUE(torch::allclose(out1, out2));
  }
  { // test BCE with logits ones in pos weights are the same as none
    // 创建一个64x4大小的随机目标张量
    const auto target = torch::rand({64, 4});
    // 创建一个64x4大小的随机输出张量，减去0.5以调整范围
    const auto output = torch::rand({64, 4}) - 0.5;
    // 创建一个64x4大小的全1张量作为正样本权重
    const auto pos_weight = torch::ones({64, 4});

    // 断言使用默认正样本权重和使用指定正样本权重计算的损失结果近似相等
    ASSERT_TRUE(torch::allclose(
        BCEWithLogitsLoss()(output, target),
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().pos_weight(pos_weight))(
            output, target)));
  }
  { // test BCE with logits broadcasts pos weights
    // 创建一个64x4大小的随机目标张量
    const auto target = torch::rand({64, 4});
    // 创建一个64x4大小的随机输出张量，减去0.5以调整范围
    const auto output = torch::rand({64, 4}) - 0.5;
    // 创建一个大小为4的随机张量作为正样本权重
    const auto pos_weight = torch::rand(4);

    // 使用指定正样本权重创建BCEWithLogitsLoss对象，并计算损失
    const auto out1 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);

    // 扩展正样本权重为1x4大小，并重新计算损失
    const auto pos_weight1 = pos_weight.expand({1, 4});
    const auto out2 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);

    // 扩展正样本权重为64x4大小，并重新计算损失
    const auto pos_weight2 = pos_weight.expand({64, 4});
    const auto out3 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);

    // 断言三种方式计算的损失结果近似相等
    ASSERT_TRUE(torch::allclose(out1, out2));
    ASSERT_TRUE(torch::allclose(out1, out3));
  }
  { // test BCE with logits with pos weight has correct grad at zero
    // 创建一个大小为3x1的零张量，并设置为需要梯度计算
    const auto output = torch::zeros({3, 1}, torch::requires_grad());
    // 创建一个大小为3x1的零目标张量
    const auto target = torch::zeros({3, 1});
    // 创建一个大小为3x1的全1张量作为正样本权重
    const auto pos_weight = torch::ones({3, 1});
    
    // 使用指定正样本权重、设置损失计算方式为求和的BCEWithLogitsLoss对象，并计算损失
    BCEWithLogitsLoss(BCEWithLogitsLossOptions()
                          .pos_weight(pos_weight)
                          .reduction(torch::kSum))(output, target)
        .backward();
    
    // 创建一个大小为3x1的全0.5张量作为期望的梯度值
    const auto expected_grad = torch::empty({3, 1}).fill_(0.5);
    // 获取output张量的梯度值
    const auto grad = output.grad();
    
    // 断言计算得到的梯度值与期望的梯度值近似相等
    ASSERT_TRUE(torch::allclose(grad, expected_grad));
  }
  { // test BCE with logits stability
    // 创建一个包含[0., -120.]的张量作为输出
    const auto output = torch::tensor({0., -120.});
    // 创建一个包含[0., 1.]的张量作为目标
    const auto target = torch::tensor({0., 1.});
    // 创建一个大小为2的全1张量作为正样本权重
    const auto pos_weight = torch::tensor({1., 1.});

    // 计算使用默认正样本权重的BCEWithLogitsLoss的损失，并断言结果中所有元素都是有限的
    const auto out1 = BCEWithLogitsLoss()(output, target);
    ASSERT_TRUE(torch::isfinite(out1).all().item<bool>());

    // 使用指定正样本权重的BCEWithLogitsLoss对象计算损失，并断言结果中所有元素都是有限的
    const auto out2 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);
    ASSERT_TRUE(torch::isfinite(out2).all().item<bool>());
  }
}

namespace detail {

namespace F = torch::nn::functional;

// 执行批量矩阵乘法，输入张量 a 和 b，返回结果张量
torch::Tensor _batchmatmul(const torch::Tensor& a, const torch::Tensor& b) {
  // 断言输入张量 a 和 b 的第一维和第二维相同
  TORCH_INTERNAL_ASSERT(a.size(0) == b.size(0));
  TORCH_INTERNAL_ASSERT(a.size(1) == b.size(1));
  // 创建全零张量作为返回值，并指定数据类型为 float32
  auto retval = torch::zeros(
      {a.size(0), a.size(1), a.size(2), b.size(3)}, torch::kFloat32);
  // 遍历张量 a 的第一维和第二维
  for (const auto i : c10::irange(a.size(0))) {
    for (const auto j : c10::irange(a.size(1))) {
      // 计算矩阵乘法并存储结果到返回张量的对应位置
      retval[i][j] = torch::matmul(a[i][j], b[i][j]);
    }
  }
  return retval;
}

// 执行 softmax 操作，输入张量 x，返回归一化后的输出张量
torch::Tensor _softmax(const torch::Tensor& x) {
  // 创建与输入张量 x 相同形状的全零张量
  auto output = torch::zeros(x.sizes());
  // 遍历张量 x 的三个维度
  for (const auto i : c10::irange(x.size(0))) {
    for (const auto j : c10::irange(x.size(1))) {
      for (const auto k : c10::irange(x.size(2))) {
        // 获取当前位置的元素和对应的指数值
        const auto& x_curr = x[i][j][k];
        const auto e_x = torch::exp(x_curr - torch::max(x_curr));
        // 计算 softmax 值并存储到输出张量的对应位置
        output[i][j][k] = e_x / torch::sum(e_x);
      }
    }
  }
  return output;
}

// 执行缩放点注意力计算，输入 Q, K, V 张量及其他参数，返回注意力输出和注意力权重
std::tuple<torch::Tensor, torch::Tensor> _scaled_dot_attn_ref(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    at::IntArrayRef dims,
    const torch::Tensor& unseen_mask = {},
    const torch::Tensor& key_padding_mask = {},
    bool average_attn_weights = true) {
  // 计算 Q 和 K 的乘积并缩放，转置 K 后进行操作
  auto QKT = _batchmatmul(Q, K.permute({0, 1, 3, 2}) / std::sqrt(dims[3]));
  // 提取 QKT 的各维度大小
  const auto b1 = QKT.size(0);
  const auto b2 = QKT.size(1);
  const auto s1 = QKT.size(2);
  const auto s2 = QKT.size(3);
  // 如果存在未见数据掩码或关键填充掩码
  if (unseen_mask.defined() || key_padding_mask.defined()) {
    // 遍历 QKT 张量的各维度
    for (const auto i : c10::irange(b1)) {
      for (const auto j : c10::irange(b2)) {
        for (const auto m : c10::irange(s1)) {
          for (const auto n : c10::irange(s2)) {
            // 根据掩码值将对应位置设为负无穷大
            if (unseen_mask.defined() &&
                unseen_mask[m][n].item<double>() == 0) {
              QKT[i][j][m][n] = -std::numeric_limits<double>::infinity();
            }
            if (key_padding_mask.defined() &&
                key_padding_mask[i][n].item<double>() != 0) {
              QKT[i][j][m][n] = -std::numeric_limits<double>::infinity();
            }
          }
        }
      }
    }
  }
  // 计算 QKT 张量的 softmax 值
  auto reference = _softmax(QKT);
  auto ref_attn_weight = reference;
  // 如果需要计算平均注意力权重
  if (average_attn_weights) {
    // 对第二维求和并除以 b2 得到平均值
    ref_attn_weight = torch::sum(ref_attn_weight, /*axis=*/1) / b2;
  }
  // 计算最终的注意力输出，并返回结果及注意力权重
  reference = _batchmatmul(reference, V);
  return std::tie(reference, ref_attn_weight);
}

// 执行头部分割操作，将输入张量 X 按头部分割，返回重组后的张量
torch::Tensor _split_heads_ref(
    const torch::Tensor& X,
    at::IntArrayRef dims,
    int nheads,
    int d_head) {
  // 将输入张量 X 重塑为四维张量，形状为 {dims[0], dims[1], nheads, d_head}
  auto X_split = X.reshape({dims[0], dims[1], nheads, d_head});
  // 对重塑后的张量进行维度置换，形状变为 {dims[0], nheads, dims[1], d_head}
  auto X_split_transposed = X_split.permute({0, 2, 1, 3});
  // 最终将张量重新重塑为 {dims[0], nheads, dims[1], d_head} 形状并返回
  return X_split_transposed.reshape({dims[0], nheads, dims[1], d_head});
}

// 执行头部合并操作，将输入张量 X 按头部合并，返回重组后的张量
torch::Tensor _combine_heads_ref(
    const torch::Tensor& X,
    at::IntArrayRef dims,
    int nheads,
    int d_head) {
    # 定义一个函数，接受四个参数：X 是一个张量，dims 是一个整数列表，nheads 是头数，d_head 是头的维度
  auto X_transposed = X.permute({0, 2, 1, 3});
    # 对张量 X 进行维度置换，将索引顺序改为 {0, 2, 1, 3}
  auto reference = X_transposed.reshape({dims[0], dims[1], nheads * d_head});
    # 将经过置换的张量 X_transposed 重新整形为指定形状，其中 dims[0] 是批量大小，dims[1] 是长度，nheads * d_head 是新的特征维度
  return reference;
    # 返回整形后的张量 reference，该张量作为函数的输出结果
}

// 定义一个函数 `_fc`，接受三个参数 X, X_weight, X_bias，并返回一个 Tensor
torch::Tensor _fc(
    torch::Tensor X,
    torch::Tensor X_weight,
    torch::Tensor X_bias) {
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto X_fc_b = X_bias;  // 将 X_bias 复制给 X_fc_b
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto X_fc_w = X_weight;  // 将 X_weight 复制给 X_fc_w
  // 返回 X 与 X_fc_w 的转置矩阵相乘后加上 X_fc_b 的结果
  return torch::matmul(X, torch::t(X_fc_w)) + X_fc_b;
}

// 定义一个测试辅助函数 `_multihead_attn_test_helper`
void _multihead_attn_test_helper(
    bool add_key_padding_mask = false,  // 是否添加关键填充掩码
    bool add_bias_kv = false,           // 是否添加偏置键值对
    bool add_zero_attn = false,         // 是否添加零注意力
    bool saved_kv = false,              // 是否保存键值对
    bool same_embed_dim = false,        // 是否相同嵌入维度
    bool average_attn_weights = true) { // 是否计算平均注意力权重
  std::random_device device;  // 随机设备
  std::mt19937 generator(device());  // 以随机设备为种子创建 Mersenne Twister 伪随机数生成器
  std::uniform_int_distribution<int> d_2_10(2, 10);  // 创建均匀整数分布，范围为 [2, 10]
  std::uniform_int_distribution<int> d_3_10(3, 10);  // 创建均匀整数分布，范围为 [3, 10]
  bool registration_checked = false;  // 注册检查标志，初始为假
  // 循环执行 100 次，使用 c10::irange(100) 来生成迭代器
  for (const auto i : c10::irange(100)) {
    (void)i; // 抑制未使用变量警告
    const auto batch_sz = d_2_10(generator);  // 生成批量大小在 [2, 10] 内的随机数
    const auto seq_len = d_2_10(generator);   // 生成序列长度在 [2, 10] 内的随机数
    const auto d_head = d_3_10(generator);    // 生成头维度在 [3, 10] 内的随机数
    const auto nheads = d_3_10(generator);    // 生成头数在 [3, 10] 内的随机数
    const auto d_model = d_head * nheads;     // 计算模型维度
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int kv_dim;  // 键值维度
    if (same_embed_dim) {
      kv_dim = d_model;  // 如果要求相同的嵌入维度，则设为模型维度
    } else {
      std::uniform_int_distribution<int> d(5, 20);  // 创建均匀整数分布，范围为 [5, 20]
      kv_dim = d(generator);  // 生成键值维度
      while (kv_dim == d_model) {  // 如果键值维度等于模型维度，重新生成
        kv_dim = d(generator);
      }
    }
    std::vector<int64_t> dims{batch_sz, seq_len, kv_dim};  // 创建包含三个元素的维度向量
    torch::Tensor saved_k;  // 保存 K 的张量
    torch::Tensor saved_k_tensor;  // 保存 K 的张量副本
    torch::Tensor saved_v;  // 保存 V 的张量
    torch::Tensor saved_v_tensor;  // 保存 V 的张量副本
    if (saved_kv) {
      saved_k = torch::rand({batch_sz * nheads, seq_len, d_head});  // 生成随机张量保存为 saved_k
      saved_k_tensor = saved_k;  // 将 saved_k 复制给 saved_k_tensor
      saved_v = torch::rand({batch_sz * nheads, seq_len, d_head});  // 生成随机张量保存为 saved_v
      saved_v_tensor = saved_v;  // 将 saved_v 复制给 saved_v_tensor
    }
    torch::Tensor key_padding_mask;  // 关键填充掩码
    torch::Tensor key_padding_mask_tensor;  // 关键填充掩码张量
    if (add_key_padding_mask) {
      const auto seq_mask = torch::randint(0, 2, {1, seq_len});  // 生成随机整数张量 seq_mask
      key_padding_mask = seq_mask.repeat({batch_sz, 1}) == 1;  // 创建关键填充掩码
      key_padding_mask_tensor = key_padding_mask;  // 将关键填充掩码复制给关键填充掩码张量
    }
    const auto decoder_state = torch::rand({batch_sz, d_model});  // 生成随机张量作为解码器状态
    const torch::Tensor K = torch::rand(dims);  // 生成随机张量 K
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const torch::Tensor V = K;  // 将 K 复制给 V
    const torch::Tensor Q =
        decoder_state.clone().resize_({batch_sz, 1, d_model});  // 复制解码器状态，调整大小为 [batch_sz, 1, d_model]，作为 Q
    auto attn_mask = torch::randint(0, 2, {1, seq_len}, torch::kFloat);  // 生成随机整数张量 attn_mask
    const torch::Tensor attn_mask_tensor = attn_mask.clone();  // 克隆 attn_mask 生成 attn_mask_tensor
    attn_mask_tensor.masked_fill_(  // 根据条件填充 attn_mask_tensor
        attn_mask_tensor == 0, -std::numeric_limits<double>::infinity());  // 将为 0 的位置填充为负无穷大
    attn_mask_tensor.masked_fill_(  // 继续填充 attn_mask_tensor
        attn_mask_tensor > 0, double(0.0));  // 将大于 0 的位置填充为 0.0

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const torch::Tensor decoder_state_tensor = decoder_state;  // 将解码器状态复制给解码器状态张量
    const torch::Tensor source_hid_tensor = K.transpose(0, 1);  // 将 K 转置后的结果复制给 source_hid_tensor
    const auto options = MultiheadAttentionOptions(d_model, nheads)
                             .add_bias_kv(add_bias_kv)
                             .add_zero_attn(add_zero_attn)
                             .kdim(kv_dim)
                             .vdim(kv_dim);
    // 创建多头注意力机制的选项对象，配置模型的参数和特性

    const auto multihead_attn_module = MultiheadAttention(options);
    // 根据选项创建多头注意力机制模块

    if (!registration_checked) {
      // 确保所有参数都已正确注册
      auto named_parameters = multihead_attn_module->named_parameters();
      if (same_embed_dim) {
        ASSERT_TRUE(named_parameters.contains("in_proj_weight"));
      } else {
        ASSERT_TRUE(named_parameters.contains("q_proj_weight"));
        ASSERT_TRUE(named_parameters.contains("k_proj_weight"));
        ASSERT_TRUE(named_parameters.contains("v_proj_weight"));
      }
      if (add_bias_kv) {
        ASSERT_TRUE(named_parameters.contains("bias_k"));
        ASSERT_TRUE(named_parameters.contains("bias_v"));
      }
      // 确保所有子模块都已正确注册
      auto submodules = multihead_attn_module->named_children();
      ASSERT_TRUE(submodules.contains("out_proj"));
      registration_checked = true;
    }

    torch::Tensor bias_k;
    torch::Tensor bias_v;
    if (add_bias_kv) {
      bias_k = multihead_attn_module->bias_k.detach();
      bias_v = multihead_attn_module->bias_v.detach();
    } else {
      bias_k.reset();
      bias_v.reset();
    }
    // 根据是否添加偏置项，获取相应的偏置张量

    torch::Tensor _Q = decoder_state_tensor.unsqueeze(1).transpose(0, 1);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    torch::Tensor _V = source_hid_tensor;
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    torch::Tensor _K = source_hid_tensor;
    // 初始化查询、键、值张量

    torch::Tensor result;
    torch::Tensor result_weight;
    if (multihead_attn_module->_qkv_same_embed_dim) {
      // 如果查询、键、值使用相同的嵌入维度
      std::tie(result, result_weight) = F::multi_head_attention_forward(
          _Q,
          _K,
          _V,
          F::MultiheadAttentionForwardFuncOptions(
              /*embed_dim_to_check=*/d_model,
              /*num_heads=*/nheads,
              /*in_proj_weight=*/multihead_attn_module->in_proj_weight,
              /*in_proj_bias=*/multihead_attn_module->in_proj_bias,
              /*bias_k=*/multihead_attn_module->bias_k,
              /*bias_v=*/multihead_attn_module->bias_v,
              /*add_zero_attn=*/multihead_attn_module->options.add_zero_attn(),
              /*dropout_p=*/multihead_attn_module->options.dropout(),
              /*out_proj_weight=*/multihead_attn_module->out_proj->weight,
              /*out_proj_bias=*/multihead_attn_module->out_proj->bias)
              .training(multihead_attn_module->is_training())
              .key_padding_mask(key_padding_mask_tensor)
              .need_weights(true)
              .attn_mask(attn_mask_tensor)
              .static_k(saved_k_tensor)
              .static_v(saved_v_tensor)
              .average_attn_weights(average_attn_weights));
      // 执行多头注意力的前向传播，并返回结果张量和注意力权重
    } else {
      // 调用多头注意力前向传播函数，返回结果和结果权重
      std::tie(result, result_weight) = F::multi_head_attention_forward(
          _Q,
          _K,
          _V,
          // 设置多头注意力前向传播函数的选项
          F::MultiheadAttentionForwardFuncOptions(
              /*embed_dim_to_check=*/d_model,  // 检查嵌入维度是否一致
              /*num_heads=*/nheads,            // 指定注意力头的数量
              /*in_proj_weight=*/{},           // 输入投影权重为空
              /*in_proj_bias=*/multihead_attn_module->in_proj_bias,  // 输入投影偏置
              /*bias_k=*/multihead_attn_module->bias_k,  // K的偏置
              /*bias_v=*/multihead_attn_module->bias_v,  // V的偏置
              /*add_zero_attn=*/multihead_attn_module->options.add_zero_attn(),  // 是否添加零注意力
              /*dropout_p=*/multihead_attn_module->options.dropout(),  // 丢弃率
              /*out_proj_weight=*/multihead_attn_module->out_proj->weight,  // 输出投影权重
              /*out_proj_bias=*/multihead_attn_module->out_proj->bias)
              .training(multihead_attn_module->is_training())  // 是否训练模式
              .key_padding_mask(key_padding_mask_tensor)  // 关键填充掩码
              .need_weights(true)  // 是否需要注意力权重
              .attn_mask(attn_mask_tensor)  // 注意力掩码
              .use_separate_proj_weight(true)  // 是否使用单独的投影权重
              .q_proj_weight(multihead_attn_module->q_proj_weight)  // 查询投影权重
              .k_proj_weight(multihead_attn_module->k_proj_weight)  // 键投影权重
              .v_proj_weight(multihead_attn_module->v_proj_weight)  // 值投影权重
              .static_k(saved_k_tensor)  // 静态K
              .static_v(saved_v_tensor)  // 静态V
              .average_attn_weights(average_attn_weights));  // 平均注意力权重
    }
    // 结果展平并分离，然后分离计算图
    result = result.squeeze(0).detach();
    torch::Tensor q_proj_weight;
    torch::Tensor k_proj_weight;
    torch::Tensor v_proj_weight;
    if (multihead_attn_module->_qkv_same_embed_dim) {
      // 如果查询、键、值共享相同的嵌入维度
      q_proj_weight =
          multihead_attn_module->in_proj_weight.slice(/*dim=*/0, 0, d_model);  // 切片查询投影权重
      k_proj_weight = multihead_attn_module->in_proj_weight.slice(
          /*dim=*/0, d_model, (d_model * 2));  // 切片键投影权重
      v_proj_weight =
          multihead_attn_module->in_proj_weight.slice(/*dim=*/0, (d_model * 2));  // 切片值投影权重
    } else {
      // 否则，使用单独的查询、键、值投影权重
      q_proj_weight = multihead_attn_module->q_proj_weight;
      k_proj_weight = multihead_attn_module->k_proj_weight;
      v_proj_weight = multihead_attn_module->v_proj_weight;
    }
    // 应用全连接层到查询向量Q
    auto Q_fc =
        _fc(Q,
            q_proj_weight,
            multihead_attn_module->in_proj_bias.slice(/*dim=*/0, 0, d_model));  // 切片查询的偏置
    // 应用全连接层到键向量K
    auto K_fc =
        _fc(K,
            k_proj_weight,
            multihead_attn_module->in_proj_bias.slice(
                /*dim=*/0, d_model, (d_model * 2)));  // 切片键的偏置
    // 应用全连接层到值向量V
    auto V_fc = _fc(
        V,
        v_proj_weight,
        multihead_attn_module->in_proj_bias.slice(/*dim=*/0, (d_model * 2)));  // 切片值的偏置
    // 如果需要添加偏置项 (add_bias_kv 为 true)，则执行以下操作
    if (add_bias_kv) {
      // 将 bias_k 沿第 1 维度复制到 K_fc 的最后，以扩展其维度
      K_fc = torch::cat(
          {K_fc,
           bias_k.repeat({K_fc.size(0) / bias_k.size(0), 1, 1} /*, axis=0*/)},
          /*dim=*/1);
      // 将 bias_v 沿第 1 维度复制到 V_fc 的最后，以扩展其维度
      V_fc = torch::cat(
          {V_fc,
           bias_v.repeat({V_fc.size(0) / bias_v.size(0), 1, 1} /*, axis=0*/)},
          /*dim=*/1);
      // 如果存在注意力遮罩 attn_mask，则在其右侧添加一列全为 1 的张量
      if (attn_mask.defined()) {
        attn_mask = torch::cat({attn_mask, torch::ones({1, 1})}, /*dim=*/1);
      }
      // 如果存在键值填充遮罩 key_padding_mask，则在其右侧添加一列全为 false 的布尔张量
      if (key_padding_mask.defined()) {
        key_padding_mask = torch::cat(
            {key_padding_mask, torch::full({batch_sz, 1}, false, torch::kBool)},
            /*dim=*/1);
      }
      // 增加 dims[1] 的值，以反映添加偏置项后的维度变化
      dims[1] += 1;
    }
    // 将 Q_fc 按照给定参数拆分成多个头，存储在 Q_split 中
    const auto Q_split =
        _split_heads_ref(Q_fc, {batch_sz, 1, d_model}, nheads, d_head);
    // 初始化 K_split
    torch::Tensor K_split;
    // 如果已保存的 K_split 存在，则重新形状化 saved_k
    if (saved_k.defined()) {
      K_split = saved_k.reshape({dims[0], nheads, dims[1], d_head});
    } else {
      // 否则，将 K_fc 按照给定参数拆分成多个头，存储在 K_split 中
      K_split = _split_heads_ref(K_fc, dims, nheads, d_head);
    }
    // 初始化 V_split
    torch::Tensor V_split;
    // 如果已保存的 V_split 存在，则重新形状化 saved_v
    if (saved_v.defined()) {
      V_split = saved_v.reshape({dims[0], nheads, dims[1], d_head});
    } else {
      // 否则，将 V_fc 按照给定参数拆分成多个头，存储在 V_split 中
      V_split = _split_heads_ref(V_fc, dims, nheads, d_head);
    }
    // 如果需要添加零注意力 (add_zero_attn 为 true)
    if (add_zero_attn) {
      // 增加 dims[1] 的值，以反映添加零注意力后的维度变化
      dims[1] += 1;
      // 在 K_split 的第 2 维度上添加形状为 (batch_sz, nheads, 1, d_head) 的零张量
      K_split = torch::cat(
          {K_split,
           torch::zeros(
               {K_split.size(0), K_split.size(1), 1, K_split.size(3)})},
          /*dim=*/2);
      // 在 V_split 的第 2 维度上添加形状为 (batch_sz, nheads, 1, d_head) 的零张量
      V_split = torch::cat(
          {V_split,
           torch::zeros(
               {V_split.size(0), V_split.size(1), 1, V_split.size(3)})},
          /*dim=*/2);
      // 如果存在注意力遮罩 attn_mask，则在其右侧添加一列全为 1 的张量
      if (attn_mask.defined()) {
        attn_mask = torch::cat({attn_mask, torch::ones({1, 1})}, /*dim=*/1);
      }
      // 如果存在键值填充遮罩 key_padding_mask，则在其右侧添加一列全为 false 的布尔张量
      if (key_padding_mask.defined()) {
        key_padding_mask = torch::cat(
            {key_padding_mask, torch::full({batch_sz, 1}, false, torch::kBool)},
            /*dim=*/1);
      }
    }
    // 执行缩放点注意力机制，得到注意力头和参考的注意力权重
    torch::Tensor attn_heads;
    torch::Tensor ref_attn_weight;
    std::tie(attn_heads, ref_attn_weight) = _scaled_dot_attn_ref(
        Q_split,
        K_split,
        V_split,
        Q_split.sizes(),
        attn_mask,
        key_padding_mask,
        average_attn_weights);
    // 将注意力头重新组合成原始维度的张量
    const auto combined_attn_heads =
        _combine_heads_ref(attn_heads, {batch_sz, 1}, nheads, d_head);
    // 使用全连接层 _fc 对注意力头进行加权和线性变换，得到 reference
    auto reference =
        _fc(combined_attn_heads,
            multihead_attn_module->out_proj->weight,
            multihead_attn_module->out_proj->bias);
    // 对 reference 进行挤压操作，去除维度为 1 的轴
    // NOLINTNEXTLINE(bugprone-argument-comment)
    reference = torch::squeeze(reference, /*axis=*/1);

    // 确认 result 与 reference 的维度相同
    ASSERT_EQ(result.sizes(), std::vector<int64_t>({batch_sz, d_model}));
    // 确认 result 与 reference 在给定的容差内相等
    ASSERT_TRUE(
        torch::allclose(result, reference, 1e-5, 1e-5, /*equal_nan=*/true));

    // 确认 result_weight 与 ref_attn_weight 的维度相同
    result_weight = result_weight.detach();
    ASSERT_EQ(result_weight.sizes(), ref_attn_weight.sizes());
    // 确认 result_weight 与 ref_attn_weight 在给定的容差内相等
    ASSERT_TRUE(torch::allclose(
        result_weight, ref_attn_weight, 1e-5, 1e-5, /*equal_nan=*/true));
  }
TEST_F(ModulesTest, MultiheadAttention) {
  // 在 ModulesTest 测试套件中的 MultiheadAttention 测试用例

  using namespace ::detail;
  // 引入命名空间 detail

  for (auto average_attn_weights : {false, true}) {
    // 遍历 average_attn_weights 变量，其取值为 false 和 true

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_add_zero_attn
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/false,    // 不添加 key padding mask
        /*add_bias_kv=*/false,            // 不添加 bias kv
        /*add_zero_attn=*/true,           // 添加 zero attention
        /*saved_kv=*/false,               // 不保存 kv
        /*same_embed_dim=*/false,         // 不同嵌入维度
        /*average_attn_weights=*/average_attn_weights);  // 平均 attention 权重

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_add_bias_kv
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/false,    // 不添加 key padding mask
        /*add_bias_kv=*/true,             // 添加 bias kv
        /*add_zero_attn=*/false,          // 不添加 zero attention
        /*saved_kv=*/false,               // 不保存 kv
        /*same_embed_dim=*/false,         // 不同嵌入维度
        /*average_attn_weights=*/average_attn_weights);  // 平均 attention 权重

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_no_masking
    _multihead_attn_test_helper();

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_key_padding_mask
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,     // 添加 key padding mask
        /*add_bias_kv=*/false,            // 不添加 bias kv
        /*add_zero_attn=*/false,          // 不添加 zero attention
        /*saved_kv=*/false,               // 不保存 kv
        /*same_embed_dim=*/false,         // 不同嵌入维度
        /*average_attn_weights=*/average_attn_weights);  // 平均 attention 权重

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_saved_kv
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/false,    // 不添加 key padding mask
        /*add_bias_kv=*/false,            // 不添加 bias kv
        /*add_zero_attn=*/false,          // 不添加 zero attention
        /*saved_kv=*/true,                // 保存 kv
        /*same_embed_dim=*/false,         // 不同嵌入维度
        /*average_attn_weights=*/average_attn_weights);  // 平均 attention 权重

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_add_bias_kv_zero_attn
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,     // 添加 key padding mask
        /*add_bias_kv=*/true,             // 添加 bias kv
        /*add_zero_attn=*/true,           // 添加 zero attention
        /*saved_kv=*/false,               // 不保存 kv
        /*same_embed_dim=*/false,         // 不同嵌入维度
        /*average_attn_weights=*/average_attn_weights);  // 平均 attention 权重

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_all_arguments1
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,     // 添加 key padding mask
        /*add_bias_kv=*/false,            // 不添加 bias kv
        /*add_zero_attn=*/true,           // 添加 zero attention
        /*saved_kv=*/true,                // 保存 kv
        /*same_embed_dim=*/false,         // 不同嵌入维度
        /*average_attn_weights=*/average_attn_weights);  // 平均 attention 权重

    // 断言测试抛出异常
    ASSERT_THROWS_WITH(
        // 测试场景: test_multihead_attn_all_arguments2
        _multihead_attn_test_helper(
            /*add_key_padding_mask=*/true,     // 添加 key padding mask
            /*add_bias_kv=*/true,             // 添加 bias kv
            /*add_zero_attn=*/true,           // 添加 zero attention
            /*saved_kv=*/true,                // 保存 kv
            /*same_embed_dim=*/false,         // 不同嵌入维度
            /*average_attn_weights=*/average_attn_weights),
        "bias cannot be added to static key");

    // 调用 _multihead_attn_test_helper 进行测试
    // 测试场景: test_multihead_attn_all_arguments3
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,     // 添加 key padding mask
        /*add_bias_kv=*/false,            // 不添加 bias kv
        /*add_zero_attn=*/true,           // 添加 zero attention
        /*saved_kv=*/true,                // 保存 kv
        /*same_embed_dim=*/true,          // 相同嵌入维度
        /*average_attn_weights=*/average_attn_weights);  // 平均 attention 权重
  }
}

TEST_F(ModulesTest, PrettyPrintIdentity) {
  // 在 ModulesTest 测试套件中的 PrettyPrintIdentity 测试用例

  ASSERT_EQ(c10::str(Identity()), "torch::nn::Identity()");
  // 断言 Identity 对象转换为字符串应该为 "torch::nn::Identity()"
}


注释：以上是对 C++ 测试代码的详细注释，描述了每个测试用例的名称和参数设置，以及相关的断言。
TEST_F(ModulesTest, PrettyPrintFlatten) {
  // 测试 Flatten 函数，验证其返回的字符串表示是否符合预期
  ASSERT_EQ(c10::str(Flatten()), "torch::nn::Flatten(start_dim=1, end_dim=-1)");
  // 测试带有自定义参数的 Flatten 函数，验证其返回的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(Flatten(FlattenOptions().start_dim(2).end_dim(4))),
      "torch::nn::Flatten(start_dim=2, end_dim=4)");
}

TEST_F(ModulesTest, PrettyPrintUnflatten) {
  // 测试 Unflatten 函数，验证其返回的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(Unflatten(UnflattenOptions(0, {2, 2}))),
      "torch::nn::Unflatten(dim=0, unflattened_size={2, 2})");
  // 测试带有字符串维度和自定义大小的 Unflatten 函数，验证其返回的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(Unflatten(UnflattenOptions(
          "B",
          {std::pair<std::string, int64_t>{"B1", 2},
           std::pair<std::string, int64_t>{"B2", 2}}))),
      "torch::nn::Unflatten(dim=\"B\", unflattened_size={{\"B1\", 2}, {\"B2\", 2}})");
}

TEST_F(ModulesTest, ReflectionPad1d) {
  {
    // 创建 ReflectionPad1d 模块，设置填充宽度为2
    ReflectionPad1d m(ReflectionPad1dOptions(2));
    // 创建输入张量，进行填充操作，并验证输出张量是否与预期相近
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = torch::tensor(
        {{{2., 1., 0., 1., 2., 3., 2., 1.}, {6., 5., 4., 5., 6., 7., 6., 5.}}},
        torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建 ReflectionPad1d 模块，设置每侧填充宽度为{3, 1}
    ReflectionPad1d m(ReflectionPad1dOptions({3, 1}));
    // 创建输入张量，进行填充操作，并验证输出张量是否与预期相近
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = torch::tensor(
        {{{3., 2., 1., 0., 1., 2., 3., 2.}, {7., 6., 5., 4., 5., 6., 7., 6.}}},
        torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReflectionPad2d) {
  {
    // 创建 ReflectionPad2d 模块，设置填充宽度为2
    ReflectionPad2d m(ReflectionPad2dOptions(2));
    // 创建输入张量，进行填充操作，并验证输出张量是否与预期相近
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor(
        {{{{8., 7., 6., 7., 8., 7., 6.},
           {5., 4., 3., 4., 5., 4., 3.},
           {2., 1., 0., 1., 2., 1., 0.},
           {5., 4., 3., 4., 5., 4., 3.},
           {8., 7., 6., 7., 8., 7., 6.},
           {5., 4., 3., 4., 5., 4., 3.},
           {2., 1., 0., 1., 2., 1., 0.}}}},
        torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建 ReflectionPad2d 模块，设置每侧填充宽度为{1, 1, 2, 0}
    ReflectionPad2d m(ReflectionPad2dOptions({1, 1, 2, 0}));
    // 创建输入张量，进行填充操作，并验证输出张量是否与预期相近
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor(
        {{{{7., 6., 7., 8., 7.},
           {4., 3., 4., 5., 4.},
           {1., 0., 1., 2., 1.},
           {4., 3., 4., 5., 4.},
           {7., 6., 7., 8., 7.}}}},
        torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReflectionPad3d) {
  {
    // 创建 ReflectionPad3d 模块，设置填充宽度为1
    ReflectionPad3d m(ReflectionPad3dOptions(1));
    // 创建输入张量，进行填充操作，并验证输出张量是否与预期相近
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    // 创建一个预期的张量，表示期望的输出结果，使用特定的浮点数值初始化
    auto expected = torch::tensor(
        {{{{{7., 6., 7., 6.},
            {5., 4., 5., 4.},
            {7., 6., 7., 6.},
            {5., 4., 5., 4.}},
           {{3., 2., 3., 2.},
            {1., 0., 1., 0.},
            {3., 2., 3., 2.},
            {1., 0., 1., 0.}},
           {{7., 6., 7., 6.},
            {5., 4., 5., 4.},
            {7., 6., 7., 6.},
            {5., 4., 5., 4.}},
           {{3., 2., 3., 2.},
            {1., 0., 1., 0.},
            {3., 2., 3., 2.},
            {1., 0., 1., 0.}}}}},
        torch::kFloat);
    // 使用 ASSERT_TRUE 进行断言，验证输出张量 output 是否与预期的张量 expected 接近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建一个 ReflectionPad3d 层的实例 m，使用指定的选项进行初始化
    ReflectionPad3d m(ReflectionPad3dOptions({0, 1, 1, 0, 1, 2}));
    // 创建一个输入张量 input，使用 arange 函数生成，然后重塑为指定形状
    auto input = torch::arange(16, torch::kFloat).reshape({1, 1, 4, 2, 2});
    // 将输入张量 input 通过 ReflectionPad3d 层 m 进行处理，得到输出张量 output
    auto output = m(input);
    // 创建一个预期的输出张量 expected，使用特定的浮点数值初始化
    auto expected = torch::tensor(
        {{{{{6., 7., 6.}, {4., 5., 4.}, {6., 7., 6.}},
           {{2., 3., 2.}, {0., 1., 0.}, {2., 3., 2.}},
           {{6., 7., 6.}, {4., 5., 4.}, {6., 7., 6.}},
           {{10., 11., 10.}, {8., 9., 8.}, {10., 11., 10.}},
           {{14., 15., 14.}, {12., 13., 12.}, {14., 15., 14.}},
           {{10., 11., 10.}, {8., 9., 8.}, {10., 11., 10.}},
           {{6., 7., 6.}, {4., 5., 4.}, {6., 7., 6.}}}}},
        torch::kFloat);
    // 使用 ASSERT_EQ 进行断言，验证输出张量 output 的尺寸是否与给定的向量相等
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 7, 3, 3}));
    // 使用 ASSERT_TRUE 进行断言，验证输出张量 output 是否与预期的张量 expected 接近
    ASSERT_TRUE(output.allclose(expected));
  }
TEST_F(ModulesTest, ReplicationPad1d) {
  {
    // 创建 ReplicationPad1d 模块，设置 padding 为 2
    ReplicationPad1d m(ReplicationPad1dOptions(2));
    // 创建输入张量
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    // 对输入张量进行 ReplicationPad1d 操作
    auto output = m(input);
    // 创建期望输出张量
    auto expected = torch::tensor(
        {{{0., 0., 0., 1., 2., 3., 3., 3.}, {4., 4., 4., 5., 6., 7., 7., 7.}}},
        torch::kFloat);
    // 断言输出张量与期望输出张量相近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建 ReplicationPad1d 模块，设置 padding 为 {3, 1}
    ReplicationPad1d m(ReplicationPad1dOptions({3, 1}));
    // 创建输入张量
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    // 对输入张量进行 ReplicationPad1d 操作
    auto output = m(input);
    // 创建期望输出张量
    auto expected = torch::tensor(
        {{{0., 0., 0., 0., 1., 2., 3., 3.}, {4., 4., 4., 4., 5., 6., 7., 7.}}},
        torch::kFloat);
    // 断言输出张量与期望输出张量相近
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReplicationPad2d) {
  {
    // 创建 ReplicationPad2d 模块，设置 padding 为 2
    ReplicationPad2d m(ReplicationPad2dOptions(2));
    // 创建输入张量
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    // 对输入张量进行 ReplicationPad2d 操作
    auto output = m(input);
    // 创建期望输出张量
    auto expected = torch::tensor(
        {{{{0., 0., 0., 1., 2., 2., 2.},
           {0., 0., 0., 1., 2., 2., 2.},
           {0., 0., 0., 1., 2., 2., 2.},
           {3., 3., 3., 4., 5., 5., 5.},
           {6., 6., 6., 7., 8., 8., 8.},
           {6., 6., 6., 7., 8., 8., 8.},
           {6., 6., 6., 7., 8., 8., 8.}}}},
        torch::kFloat);
    // 断言输出张量与期望输出张量相近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建 ReplicationPad2d 模块，设置 padding 为 {1, 1, 2, 0}
    ReplicationPad2d m(ReplicationPad2dOptions({1, 1, 2, 0}));
    // 创建输入张量
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    // 对输入张量进行 ReplicationPad2d 操作
    auto output = m(input);
    // 创建期望输出张量
    auto expected = torch::tensor(
        {{{{0., 0., 1., 2., 2.},
           {0., 0., 1., 2., 2.},
           {0., 0., 1., 2., 2.},
           {3., 3., 4., 5., 5.},
           {6., 6., 7., 8., 8.}}}},
        torch::kFloat);
    // 断言输出张量与期望输出张量相近
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReplicationPad3d) {
  {
    // 创建 ReplicationPad3d 模块，设置 padding 为 1
    ReplicationPad3d m(ReplicationPad3dOptions(1));
    // 创建输入张量
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    // 对输入张量进行 ReplicationPad3d 操作
    auto output = m(input);
    // 创建期望输出张量
    auto expected = torch::tensor(
        {{{{{0., 0., 1., 1.},
            {0., 0., 1., 1.},
            {2., 2., 3., 3.},
            {2., 2., 3., 3.}},
           {{0., 0., 1., 1.},
            {0., 0., 1., 1.},
            {2., 2., 3., 3.},
            {2., 2., 3., 3.}},
           {{4., 4., 5., 5.},
            {4., 4., 5., 5.},
            {6., 6., 7., 7.},
            {6., 6., 7., 7.}},
           {{4., 4., 5., 5.},
            {4., 4., 5., 5.},
            {6., 6., 7., 7.},
            {6., 6., 7., 7.}}}}},
        torch::kFloat);
    // 断言输出张量与期望输出张量相近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建 ReplicationPad3d 模块，设置 padding 为 {1, 2, 1, 2, 1, 2}
    ReplicationPad3d m(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}));
    // 创建输入张量
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    // 对输入张量进行 ReplicationPad3d 操作
    auto output = m(input);
    // 创建一个预期的张量，包含五维的数据，每个维度都有固定的值
    auto expected = torch::tensor(
        {{{{{0., 0., 1., 1., 1.},    // 第一个子数组
            {0., 0., 1., 1., 1.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.}},
           {{0., 0., 1., 1., 1.},    // 第二个子数组
            {0., 0., 1., 1., 1.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.}},
           {{4., 4., 5., 5., 5.},    // 第三个子数组
            {4., 4., 5., 5., 5.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.}},
           {{4., 4., 5., 5., 5.},    // 第四个子数组
            {4., 4., 5., 5., 5.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.}},
           {{4., 4., 5., 5., 5.},    // 第五个子数组
            {4., 4., 5., 5., 5.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.}}}}},
        torch::kFloat);
    
    // 使用断言验证输出张量是否与预期张量在所有元素上接近
    ASSERT_TRUE(output.allclose(expected));
}
{
  // 创建一个 ZeroPad3d 模块对象，使用默认选项 {1}
  ZeroPad3d m(ZeroPad3dOptions(1));
  // 创建一个 5 维的张量作为输入，形状为 {1, 1, 2, 2, 2}
  auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
  // 对输入张量进行零填充操作
  auto output = m(input);
  // 预期的输出张量，进行零填充后的结果
  auto expected = torch::tensor(
      {{{{{0., 0., 0., 0.},
          {0., 0., 0., 0.},
          {0., 0., 0., 0.},
          {0., 0., 0., 0.}},
         {{0., 0., 0., 0.},
          {0., 0., 1., 0.},
          {0., 2., 3., 0.},
          {0., 0., 0., 0.}},
         {{0., 0., 0., 0.},
          {0., 4., 5., 0.},
          {0., 6., 7., 0.},
          {0., 0., 0., 0.}},
         {{0., 0., 0., 0.},
          {0., 0., 0., 0.},
          {0., 0., 0., 0.},
          {0., 0., 0., 0.}}}}},
      torch::kFloat);
  // 断言输出张量是否与预期结果相近
  ASSERT_TRUE(output.allclose(expected));
}
{
  // 创建一个 ZeroPad3d 模块对象，使用选项 {1, 2, 1, 2, 1, 2}
  ZeroPad3d m(ZeroPad3dOptions({1, 2, 1, 2, 1, 2}));
  // 创建一个 5 维的张量作为输入，形状为 {1, 1, 2, 2, 2}
  auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
  // 对输入张量进行零填充操作
  auto output = m(input);
    // 定义预期的张量，包含特定的数值。这个张量是一个五维张量，每个维度包含特定的数值。
    auto expected = torch::tensor(
        {{{{{0., 0., 0., 0., 0.},      // 第一层
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},      // 第二层
            {0., 0., 1., 0., 0.},
            {0., 2., 3., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},      // 第三层
            {0., 4., 5., 0., 0.},
            {0., 6., 7., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},      // 第四层
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},      // 第五层
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}}}}},
        torch::kFloat);
    // 使用 ASSERT_TRUE 来验证输出张量是否与预期的张量非常接近
    ASSERT_TRUE(output.allclose(expected));
}
}

TEST_F(ModulesTest, ConstantPad1d) {
  {
    // 创建 ConstantPad1d 模块，填充宽度为2，填充值为3.5
    ConstantPad1d m(ConstantPad1dOptions(2, 3.5));
    // 创建一个形状为[1, 2, 4]的浮点张量 input
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    // 对 input 应用 ConstantPad1d 操作，生成输出 output
    auto output = m(input);
    // 创建预期结果的张量 expected
    auto expected = torch::tensor(
        {{{3.5000, 3.5000, 0.0000, 1.0000, 2.0000, 3.0000, 3.5000, 3.5000},
          {3.5000, 3.5000, 4.0000, 5.0000, 6.0000, 7.0000, 3.5000, 3.5000}}},
        torch::kFloat);
    // 使用 allclose 函数验证 output 是否与 expected 接近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建 ConstantPad1d 模块，填充宽度为{3, 1}，填充值为3.5
    ConstantPad1d m(ConstantPad1dOptions({3, 1}, 3.5));
    // 创建一个形状为[1, 2, 3]的浮点张量 input
    auto input = torch::arange(6, torch::kFloat).reshape({1, 2, 3});
    // 对 input 应用 ConstantPad1d 操作，生成输出 output
    auto output = m(input);
    // 创建预期结果的张量 expected
    auto expected = torch::tensor(
        {{{3.5000, 3.5000, 3.5000, 0.0000, 1.0000, 2.0000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.0000, 4.0000, 5.0000, 3.5000}}},
        torch::kFloat);
    // 使用 allclose 函数验证 output 是否与 expected 接近
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad2d) {
  {
    // 创建 ConstantPad2d 模块，填充宽度为2，填充值为3.5
    ConstantPad2d m(ConstantPad2dOptions(2, 3.5));
    // 创建一个形状为[1, 2, 2]的浮点张量 input
    auto input = torch::arange(4, torch::kFloat).reshape({1, 2, 2});
    // 对 input 应用 ConstantPad2d 操作，生成输出 output
    auto output = m(input);
    // 创建预期结果的张量 expected
    auto expected = torch::tensor(
        {{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 0.0000, 1.0000, 3.5000, 3.5000},
          {3.5000, 3.5000, 2.0000, 3.0000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}},
        torch::kFloat);
    // 使用 allclose 函数验证 output 是否与 expected 接近
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 创建 ConstantPad2d 模块，填充宽度为{3, 0, 2, 1}，填充值为3.5
    ConstantPad2d m(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
    // 创建一个形状为[1, 2, 2]的浮点张量 input
    auto input = torch::arange(4, torch::kFloat).reshape({1, 2, 2});
    // 对 input 应用 ConstantPad2d 操作，生成输出 output
    auto output = m(input);
    // 创建预期结果的张量 expected
    auto expected = torch::tensor(
        {{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 0.0000, 1.0000},
          {3.5000, 3.5000, 3.5000, 2.0000, 3.0000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}},
        torch::kFloat);
    // 使用 allclose 函数验证 output 是否与 expected 接近
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad3d) {
  {
    // 创建 ConstantPad3d 模块，填充宽度为1，填充值为3.5
    ConstantPad3d m(ConstantPad3dOptions(1, 3.5));
    // 创建一个形状为[1, 1, 2, 2, 2]的浮点张量 input
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    // 对 input 应用 ConstantPad3d 操作，生成输出 output
    auto output = m(input);
    {
        // 创建一个 ConstantPad3d 对象 m，用于进行三维张量的常数填充，填充值为 3.5
        ConstantPad3d m(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
    
        // 创建一个一维浮点张量 input，包含从 0 到 7 的数值，形状为 [1, 1, 2, 2, 2]
        auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    
        // 将 input 张量通过 ConstantPad3d 对象 m 进行填充，得到填充后的输出张量 output
        auto output = m(input);
    
        // 创建一个预期的张量 expected，用于验证 ConstantPad3d 的填充操作是否正确
        auto expected = torch::tensor(
            {{{{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
               {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 0.0000, 1.0000, 3.5000, 3.5000},
                {3.5000, 2.0000, 3.0000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
               {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 4.0000, 5.0000, 3.5000, 3.5000},
                {3.5000, 6.0000, 7.0000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
               {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
               {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}}}},
            torch::kFloat);
    
        // 使用 ASSERT_TRUE 来验证 output 张量是否与预期的 expected 张量在误差允许范围内相等
        ASSERT_TRUE(output.allclose(expected));
    }
}

// 在 ModulesTest 测试套件中定义一个名为 CrossMapLRN2d 的测试用例
TEST_F(ModulesTest, CrossMapLRN2d) {
  /// size 3, default options
  // 创建一个形状为 [1, 1, 3, 3] 的张量 input，值从 0 到 8，数据类型为 float32，并设置需要梯度
  auto input =
      torch::arange(9, torch::kFloat32).view({1, 1, 3, 3}).requires_grad_(true);
  // 创建一个期望的张量 expected，表示预期的输出结果
  auto expected = torch::tensor(
      {{{{0.00000000, 0.99997497, 1.99980010},
         {2.99932500, 3.99840070, 4.99687700},
         {5.99460600, 6.99143740, 7.98722360}}}},
      torch::kFloat32);
  // 创建一个期望的梯度张量 grad_expected，表示预期的输入梯度
  auto grad_expected = torch::tensor(
      {{{{1.00000000, 0.99992496, 0.99970007},
         {0.99932520, 0.99880093, 0.99812720},
         {0.99730474, 0.99633380, 0.99521490}}}},
      torch::kFloat32);
  // 创建 CrossMapLRN2d 类的实例 crossmaplrn2d，参数为 3
  auto crossmaplrn2d = CrossMapLRN2d(3);
  // 对输入 input 进行 CrossMapLRN2d 操作，得到输出 output
  auto output = crossmaplrn2d(input);
  // 对输出结果进行求和并进行反向传播
  output.sum().backward();

  // 断言输入的梯度是否与预期的梯度 grad_expected 接近
  ASSERT_TRUE(input.grad().allclose(grad_expected));
  // 断言输出结果是否与预期的结果 expected 接近

  ASSERT_TRUE(output.allclose(expected));

  /// size change
  // 重新分配 crossmaplrn2d 实例，使用大小为 4 的 CrossMapLRN2dOptions，设置 alpha 为 1e-4，beta 为 0.75，k 为 1
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(4).alpha(1e-4).beta(0.75).k(1));
  // 再次对 input 执行 CrossMapLRN2d 操作，得到输出 output
  output = crossmaplrn2d(input);
  // 更新期望的结果为新的预期输出
  expected = torch::tensor(
      {{{{0.00000000, 0.99998120, 1.99985000},
         {2.99949400, 3.99880050, 4.99765800},
         {5.99595300, 6.99357600, 7.99041300}}}},
      torch::kFloat32);
  // 断言输出结果是否与新的预期结果 expected 接近
  ASSERT_TRUE(output.allclose(expected));

  /// alpha change
  // 重新分配 crossmaplrn2d 实例，使用大小为 3 的 CrossMapLRN2dOptions，设置 alpha 为 1e-3，beta 为 0.75，k 为 1
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-3).beta(0.75).k(1));
  // 再次对 input 执行 CrossMapLRN2d 操作，得到输出 output
  output = crossmaplrn2d(input);
  // 更新期望的结果为新的预期输出
  expected = torch::tensor(
      {{{{0.00000000, 0.99975010, 1.99800230},
         {2.99326750, 3.98407440, 4.96897600},
         {5.94656100, 6.91545720, 7.87434340}}}},
      torch::kFloat32);
  // 断言输出结果是否与新的预期结果 expected 接近
  ASSERT_TRUE(output.allclose(expected));

  /// beta change
  // 重新分配 crossmaplrn2d 实例，使用大小为 3 的 CrossMapLRN2dOptions，设置 alpha 为 1e-4，beta 为 0.95，k 为 1
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-4).beta(0.95).k(1));
  // 再次对 input 执行 CrossMapLRN2d 操作，得到输出 output
  output = crossmaplrn2d(input);
  // 更新期望的结果为新的预期输出
  expected = torch::tensor(
      {{{{0.00000000, 0.99996830, 1.99974680},
         {2.99914500, 3.99797440, 4.99604460},
         {5.99316840, 6.98915600, 7.98382000}}}},
      torch::kFloat32);
  // 断言输出结果是否与新的预期结果 expected 接近
  ASSERT_TRUE(output.allclose(expected));

  /// k change
  // 重新分配 crossmaplrn2d 实例，使用大小为 3 的 CrossMapLRN2dOptions，设置 alpha 为 1e-4，beta 为 0.75，k 为 2
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-4).beta(0.75).k(2));
  // 再次对 input 执行 CrossMapLRN2d 操作，得到输出 output
  output = crossmaplrn2d(input);
  // 更新期望的结果为新的预期输出
  expected = torch::tensor(
      {{{{0.00000000, 0.59459610, 1.18914770},
         {1.78361000, 2.37793870, 2.97208900},
         {3.56601700, 4.15967700, 4.75302650}}}},
      torch::kFloat32);
  // 断言输出结果是否与新的预期结果 expected 接近
  ASSERT_TRUE(output.allclose(expected));
}
// 在 ModulesTest 测试套件中，测试 RNNCell 类的功能
TEST_F(ModulesTest, RNNCell) {
  // 设定随机种子为 0
  torch::manual_seed(0);
  // 创建一个输入大小为 1，隐藏状态大小为 2 的 RNNCell 对象
  auto rnn = RNNCell(1, 2);

  // 创建一个大小为 {3, 1} 的随机张量作为输入
  auto input = torch::randn({3, 1});
  // 创建一个大小为 {3, 2} 的随机张量作为初始隐藏状态
  auto hx = torch::randn({3, 2});
  // 对 RNNCell 进行前向传播计算，得到输出
  auto output = rnn(input, hx);
  // 预期的输出张量
  auto expected =
      torch::tensor({{-0.5078, 0.4380}, {-0.7215, 0.2969}, {-0.1304, 0.0653}});
  // 断言输出张量与预期张量的近似程度在给定的容差范围内
  ASSERT_TRUE(torch::allclose(output, expected, 1e-05, 2e-04));

  // 只提供输入张量进行前向传播计算，更新隐藏状态
  output = rnn(input);
  // 更新预期的输出张量
  expected =
      torch::tensor({{-0.0775, 0.6688}, {-0.0734, 0.4759}, {-0.0725, 0.4225}});
  // 断言输出张量与预期张量的近似程度在给定的容差范围内
  ASSERT_TRUE(torch::allclose(output, expected, 1e-05, 2e-04));

  // 创建一个大小为 {1} 的随机张量作为输入
  input = torch::randn({1});
  // 创建一个大小为 {2} 的随机张量作为初始隐藏状态
  hx = torch::randn({2});
  // 对 RNNCell 进行前向传播计算，得到输出
  output = rnn(input, hx);
  // 更新预期的输出张量
  expected = torch::tensor({0.2808, 0.6505});
  // 断言输出张量与预期张量的近似程度在给定的容差范围内
  ASSERT_TRUE(torch::allclose(output, expected, 1e-05, 2e-04));

  {
    // 在新的作用域中创建一个大小为 {3, 2} 的随机张量作为输入
    auto input = torch::randn({3, 2});
    // 在新的作用域中创建一个大小为 {3, 2} 的随机张量作为初始隐藏状态
    auto hx = torch::randn({3, 2});
    // 断言对 RNNCell 进行前向传播计算时出现异常，异常信息应包含指定的字符串
    ASSERT_THROWS_WITH(
        rnn(input, hx), "input has inconsistent input_size: got 2 expected 1");
  }

  {
    // 在新的作用域中创建一个大小为 {3, 1} 的随机张量作为输入
    auto input = torch::randn({3, 1});
    // 在新的作用域中创建一个大小为 {3, 1} 的随机张量作为初始隐藏状态
    auto hx = torch::randn({3, 1});
    // 断言对 RNNCell 进行前向传播计算时出现异常，异常信息应包含指定的字符串
    ASSERT_THROWS_WITH(
        rnn(input, hx),
        "hidden0 has inconsistent hidden_size: got 1, expected 2");
  }

  {
    // 在新的作用域中创建一个大小为 {3, 1, 1, 1, 1} 的随机张量作为输入
    auto input = torch::randn({3, 1, 1, 1, 1});
    // 在新的作用域中创建一个大小为 {3, 2} 的随机张量作为初始隐藏状态
    auto hx = torch::randn({3, 2});
    // 断言对 RNNCell 进行前向传播计算时出现异常，异常信息应包含指定的字符串
    ASSERT_THROWS_WITH(
        rnn(input, hx), "Expected input to be 1D or 2D, got 5D instead");
  }

  {
    // 在新的作用域中创建一个大小为 {3, 1} 的随机张量作为输入
    auto input = torch::randn({3, 1});
    // 在新的作用域中创建一个大小为 {3, 1, 1, 1, 2} 的随机张量作为初始隐藏状态
    auto hx = torch::randn({3, 1, 1, 1, 2});
    // 断言对 RNNCell 进行前向传播计算时出现异常，异常信息应包含指定的字符串
    ASSERT_THROWS_WITH(
        rnn(input, hx), "Expected hidden to be 1D or 2D, got 5D instead");
  }
}
TEST_F(ModulesTest, LSTMCell) {
  // 设置随机种子为0，以确保结果可复现
  torch::manual_seed(0);
  // 创建一个输入维度为1，隐藏状态维度为2的LSTM单元
  auto lstm = LSTMCell(1, 2);

  // 创建输入数据，形状为{3, 1}，即3个样本，每个样本1维特征
  auto input = torch::randn({3, 1});
  // 初始化隐藏状态hx和细胞状态cx，形状均为{3, 2}
  auto hx = torch::randn({3, 2});
  auto cx = torch::randn({3, 2});
  // 对LSTM单元进行前向计算，得到输出output，包含更新后的hx和cx
  auto output = lstm(input, std::make_tuple(hx, cx));
  // 从输出中分离出更新后的隐藏状态hx和细胞状态cx
  auto output_hx = std::get<0>(output);
  auto output_cx = std::get<1>(output);
  // 期望的隐藏状态和细胞状态的预期值
  auto expected_hx =
      torch::tensor({{-0.2462, 0.0810}, {-0.2206, 0.1867}, {-0.0146, 0.0429}});
  auto expected_cx =
      torch::tensor({{-0.4480, 0.1071}, {-0.6245, 0.2687}, {-0.0322, 0.0518}});
  // 使用torch::allclose函数验证输出与期望值的接近程度
  ASSERT_TRUE(torch::allclose(output_hx, expected_hx, 1e-05, 2e-04));
  ASSERT_TRUE(torch::allclose(output_cx, expected_cx, 1e-05, 2e-04));

  // 第二次前向计算，只提供输入数据input，隐藏状态和细胞状态使用默认值
  output = lstm(input);
  output_hx = std::get<0>(output);
  output_cx = std::get<1>(output);
  // 更新预期的隐藏状态和细胞状态的预期值
  expected_hx =
      torch::tensor({{-0.1331, 0.1634}, {-0.1494, 0.2869}, {-0.1428, 0.2263}});
  expected_cx =
      torch::tensor({{-0.2679, 0.2180}, {-0.3049, 0.3493}, {-0.2896, 0.2853}});
  // 再次验证输出与新的预期值的接近程度
  ASSERT_TRUE(torch::allclose(output_hx, expected_hx, 1e-05, 2e-04));
  ASSERT_TRUE(torch::allclose(output_cx, expected_cx, 1e-05, 2e-04));

  // 对于单个样本输入的测试
  input = torch::randn({1});
  hx = torch::randn({2});
  cx = torch::randn({2});
  // 进行前向计算，仅提供一个样本的输入input，和自定义的hx和cx
  output = lstm(input, std::make_tuple(hx, cx));
  output_hx = std::get<0>(output);
  output_cx = std::get<1>(output);
  // 更新预期的隐藏状态和细胞状态的预期值
  expected_hx = torch::tensor({-0.0443, 0.1537});
  expected_cx = torch::tensor({-0.1195, 0.2144});
  // 验证输出与预期值的接近程度
  ASSERT_TRUE(torch::allclose(output_hx, expected_hx, 1e-05, 2e-04));
  ASSERT_TRUE(torch::allclose(output_cx, expected_cx, 1e-05, 2e-04));

  // 输入数据和隐藏状态维度不一致的异常情况测试
  {
    auto input = torch::randn({3, 2});
    auto hx = torch::randn({3, 2});
    auto cx = torch::randn({3, 2});
    // 使用ASSERT_THROWS_WITH检测是否抛出预期的异常信息
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "input has inconsistent input_size: got 2 expected 1");
  }

  // 隐藏状态维度不一致的异常情况测试
  {
    auto input = torch::randn({3, 1});
    auto hx = torch::randn({3, 1});
    auto cx = torch::randn({3, 2});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "hidden0 has inconsistent hidden_size: got 1, expected 2");
  }

  // 细胞状态维度不一致的异常情况测试
  {
    auto input = torch::randn({3, 1});
    auto hx = torch::randn({3, 2});
    auto cx = torch::randn({3, 1});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "hidden1 has inconsistent hidden_size: got 1, expected 2");
  }

  // 输入数据维度错误的异常情况测试
  {
    auto input = torch::randn({3, 1, 1, 1, 1});
    auto hx = torch::randn({3, 1});
    auto cx = torch::randn({3, 1});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "Expected input to be 1D or 2D, got 5D instead");
  }

  // 隐藏状态维度错误的异常情况测试
  {
    auto input = torch::randn({3, 1});
    auto hx = torch::randn({3, 1, 1, 1, 2});
    auto cx = torch::randn({3, 2});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "Expected hx[0] to be 1D or 2D, got 5D instead");
  }

  // 细胞状态维度错误的异常情况测试
  {
    auto input = torch::randn({3, 1});
    auto hx = torch::randn({3, 2});
    auto cx = torch::randn({3, 1, 1, 1, 2});
    # 断言异常处理器，用于验证以下代码是否抛出特定异常信息
    ASSERT_THROWS_WITH(
        # 调用 LSTM 函数，并传入输入数据和包含 hx、cx 的元组作为参数
        lstm(input, std::make_tuple(hx, cx)),
        # 预期的异常信息字符串，验证 hx[1] 应为 1D 或 2D，实际为 5D
        "Expected hx[1] to be 1D or 2D, got 5D instead");
    }
TEST_F(ModulesTest, GRUCell) {
  // 设置随机种子为0，确保结果可重复
  torch::manual_seed(0);
  // 创建一个GRU单元，输入维度为1，隐藏状态维度为2
  auto gru = GRUCell(1, 2);

  // 创建输入张量，形状为[3, 1]，并生成随机数填充
  auto input = torch::randn({3, 1});
  // 创建初始隐藏状态张量，形状为[3, 2]，并生成随机数填充
  auto hx = torch::randn({3, 2});
  // 使用GRU单元处理输入和初始隐藏状态，得到输出
  auto output = gru(input, hx);
  // 预期输出张量
  auto expected =
      torch::tensor({{1.0243, 0.3227}, {-0.5659, 0.0330}, {-0.4030, -0.2800}});
  // 使用指定的误差范围检查输出是否接近预期值
  ASSERT_TRUE(torch::allclose(output, expected, 1e-05, 2e-04));

  // 不提供初始隐藏状态，重新计算输出
  output = gru(input);
  // 更新预期输出
  expected =
      torch::tensor({{-0.0085, 0.1095}, {-0.1291, 0.2675}, {-0.1339, 0.2725}});
  // 检查输出是否接近更新后的预期值
  ASSERT_TRUE(torch::allclose(output, expected, 1e-05, 2e-04));

  // 改变输入张量的形状为[1]，初始隐藏状态的形状为[2]
  input = torch::randn({1});
  hx = torch::randn({2});
  // 使用GRU单元处理更新后的输入和初始隐藏状态，得到输出
  output = gru(input, hx);
  // 更新预期输出
  expected = torch::tensor({-1.0058, -0.3025});
  // 检查输出是否接近更新后的预期值
  ASSERT_TRUE(torch::allclose(output, expected, 1e-05, 2e-04));

  {
    // 创建输入张量，形状为[3, 2]，并生成随机数填充
    auto input = torch::randn({3, 2});
    // 创建初始隐藏状态张量，形状为[3, 2]，并生成随机数填充
    auto hx = torch::randn({3, 2});
    // 断言处理输入和初始隐藏状态时会抛出异常，并检查异常消息
    ASSERT_THROWS_WITH(
        gru(input, hx), "input has inconsistent input_size: got 2 expected 1");
  }

  {
    // 创建输入张量，形状为[3, 1]，并生成随机数填充
    auto input = torch::randn({3, 1});
    // 创建初始隐藏状态张量，形状为[3, 1]，并生成随机数填充
    auto hx = torch::randn({3, 1});
    // 断言处理输入和初始隐藏状态时会抛出异常，并检查异常消息
    ASSERT_THROWS_WITH(
        gru(input, hx),
        "hidden0 has inconsistent hidden_size: got 1, expected 2");
  }

  {
    // 创建输入张量，形状为[3, 1, 1, 1, 1]，并生成随机数填充
    auto input = torch::randn({3, 1, 1, 1, 1});
    // 创建初始隐藏状态张量，形状为[3, 2]，并生成随机数填充
    auto hx = torch::randn({3, 2});
    // 断言处理输入和初始隐藏状态时会抛出异常，并检查异常消息
    ASSERT_THROWS_WITH(
        gru(input, hx), "Expected input to be 1D or 2D, got 5D instead");
  }

  {
    // 创建输入张量，形状为[3, 1]，并生成随机数填充
    auto input = torch::randn({3, 1});
    // 创建初始隐藏状态张量，形状为[3, 1, 1, 1, 2]，并生成随机数填充
    auto hx = torch::randn({3, 1, 1, 1, 2});
    // 断言处理输入和初始隐藏状态时会抛出异常，并检查异常消息
    ASSERT_THROWS_WITH(
        gru(input, hx), "Expected hidden to be 1D or 2D, got 5D instead");
  }
}
    # 创建一个 Conv3dOptions 对象，指定卷积操作的参数：输入和输出通道数为 4，
    # 卷积核大小为 [5, 6, 7]，步长为 [1, 2, 3]，填充为 1，膨胀为 0，分组数为 2，
    # 不使用偏置，填充模式为圆形（Circular）。
    const auto options = Conv3dOptions(4, 4, std::vector<int64_t>{5, 6, 7})
                             .stride({1, 2, 3})
                             .padding(1)
                             .dilation(0)
                             .groups(2)
                             .bias(false)
                             .padding_mode(torch::kCircular);
    # 断言 Conv3d(options) 返回的字符串等于指定的格式化字符串，验证卷积参数设置正确。
    ASSERT_EQ(
        c10::str(Conv3d(options)),
        "torch::nn::Conv3d("
        "4, "
        "4, "
        "kernel_size=[5, 6, 7], "
        "stride=[1, 2, 3], "
        "padding=[1, 1, 1], "
        "dilation=[0, 0, 0], "
        "groups=2, "
        "bias=false, "
        "padding_mode=kCircular)");
TEST_F(ModulesTest, PrettyPrintConvTranspose) {
  // 测试 ConvTranspose1d 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(ConvTranspose1d(3, 4, 5)),
      "torch::nn::ConvTranspose1d(3, 4, kernel_size=5, stride=1)");

  // 测试 ConvTranspose2d 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(ConvTranspose2d(3, 4, 5)),
      "torch::nn::ConvTranspose2d(3, 4, kernel_size=[5, 5], stride=[1, 1])");
  
  // 使用选项设置 ConvTranspose2d 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(ConvTranspose2d(ConvTranspose2dOptions(3, 4, 5).stride(2))),
      "torch::nn::ConvTranspose2d(3, 4, kernel_size=[5, 5], stride=[2, 2])");

  {
    // 使用详细选项设置 ConvTranspose2d 类型，验证字符串表示是否正确
    const auto options =
        ConvTranspose2dOptions(3, 4, std::vector<int64_t>{5, 6}).stride({1, 2});
    ASSERT_EQ(
        c10::str(ConvTranspose2d(options)),
        "torch::nn::ConvTranspose2d(3, 4, kernel_size=[5, 6], stride=[1, 2])");
  }

  // 测试 ConvTranspose3d 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(ConvTranspose3d(4, 4, std::vector<int64_t>{5, 6, 7})),
      "torch::nn::ConvTranspose3d(4, 4, kernel_size=[5, 6, 7], stride=[1, 1, 1])");

  {
    // 使用详细选项设置 ConvTranspose3d 类型，验证字符串表示是否正确
    const auto options =
        ConvTranspose3dOptions(4, 4, std::vector<int64_t>{5, 6, 7})
            .stride({1, 2, 3})
            .padding(1)
            .dilation(0)
            .groups(2)
            .bias(false)
            .padding_mode(torch::kCircular);
    ASSERT_EQ(
        c10::str(ConvTranspose3d(options)),
        "torch::nn::ConvTranspose3d("
        "4, "
        "4, "
        "kernel_size=[5, 6, 7], "
        "stride=[1, 2, 3], "
        "padding=[1, 1, 1], "
        "dilation=[0, 0, 0], "
        "groups=2, "
        "bias=false, "
        "padding_mode=kCircular)");
  }
}

TEST_F(ModulesTest, PrettyPrintUpsample) {
  // 测试 Upsample 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(
          Upsample(UpsampleOptions().size(std::vector<int64_t>({2, 4, 4})))),
      "torch::nn::Upsample(size=[2, 4, 4], mode=kNearest)");

  // 使用选项设置 Upsample 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(Upsample(UpsampleOptions()
                            .scale_factor(std::vector<double>({0.5, 1.5}))
                            .mode(torch::kBilinear))),
      "torch::nn::Upsample(scale_factor=[0.5, 1.5], mode=kBilinear)");
}

TEST_F(ModulesTest, PrettyPrintFold) {
  // 测试 Fold 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(Fold(FoldOptions({2, 2}, {5, 5}))),
      "torch::nn::Fold(output_size=[2, 2], kernel_size=[5, 5], dilation=[1, 1], padding=[0, 0], stride=[1, 1])");

  // 使用选项设置 Fold 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(Fold(
          FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2, 1}).stride(2))),
      "torch::nn::Fold(output_size=[8, 8], kernel_size=[3, 3], dilation=[2, 2], padding=[2, 1], stride=[2, 2])");
}

TEST_F(ModulesTest, PrettyPrintUnfold) {
  // 测试 Unfold 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(Unfold(torch::IntArrayRef({2, 4}))),
      "torch::nn::Unfold(kernel_size=[2, 4], dilation=[1, 1], padding=[0, 0], stride=[1, 1])");

  // 使用选项设置 Unfold 类型，验证字符串表示是否正确
  ASSERT_EQ(
      c10::str(
          Unfold(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2))),
      "torch::nn::Unfold(kernel_size=[2, 4], dilation=[2, 2], padding=[2, 1], stride=[2, 2])");
}
TEST_F(ModulesTest, PrettyPrintMaxPool) {
  // 测试 MaxPool1d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(MaxPool1d(5)),
      "torch::nn::MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=false)");
  // 测试 MaxPool2d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(MaxPool2d(5)),
      "torch::nn::MaxPool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
  // 测试 MaxPool2d 使用选项参数的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(MaxPool2d(MaxPool2dOptions(5).stride(2))),
      "torch::nn::MaxPool2d(kernel_size=[5, 5], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
  // 测试 MaxPool3d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(MaxPool3d(5)),
      "torch::nn::MaxPool3d(kernel_size=[5, 5, 5], stride=[5, 5, 5], padding=[0, 0, 0], dilation=[1, 1, 1], ceil_mode=false)");
  // 测试 MaxPool3d 使用选项参数的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(MaxPool3d(MaxPool3dOptions(5).stride(2))),
      "torch::nn::MaxPool3d(kernel_size=[5, 5, 5], stride=[2, 2, 2], padding=[0, 0, 0], dilation=[1, 1, 1], ceil_mode=false)");

  // 创建包含自定义选项的 MaxPool2dOptions 对象
  const auto options =
      MaxPool2dOptions(std::vector<int64_t>{5, 6}).stride({1, 2});
  // 测试使用自定义选项参数的 MaxPool2d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(MaxPool2d(options)),
      "torch::nn::MaxPool2d(kernel_size=[5, 6], stride=[1, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
}

TEST_F(ModulesTest, PrettyPrintAvgPool) {
  // 测试 AvgPool1d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(AvgPool1d(5)),
      "torch::nn::AvgPool1d(kernel_size=5, stride=5, padding=0)");
  // 测试 AvgPool2d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(AvgPool2d(5)),
      "torch::nn::AvgPool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0])");
  // 测试 AvgPool2d 使用选项参数的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(AvgPool2d(AvgPool2dOptions(5).stride(2))),
      "torch::nn::AvgPool2d(kernel_size=[5, 5], stride=[2, 2], padding=[0, 0])");
  // 测试 AvgPool3d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(AvgPool3d(5)),
      "torch::nn::AvgPool3d(kernel_size=[5, 5, 5], stride=[5, 5, 5], padding=[0, 0, 0])");
  // 测试 AvgPool3d 使用选项参数的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(AvgPool3d(AvgPool3dOptions(5).stride(2))),
      "torch::nn::AvgPool3d(kernel_size=[5, 5, 5], stride=[2, 2, 2], padding=[0, 0, 0])");

  // 创建包含自定义选项的 AvgPool2dOptions 对象
  const auto options =
      AvgPool2dOptions(std::vector<int64_t>{5, 6}).stride({1, 2});
  // 测试使用自定义选项参数的 AvgPool2d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(AvgPool2d(options)),
      "torch::nn::AvgPool2d(kernel_size=[5, 6], stride=[1, 2], padding=[0, 0])");
}

TEST_F(ModulesTest, PrettyPrinFractionalMaxPool) {
  // 测试 FractionalMaxPool2d 使用选项参数的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(
          FractionalMaxPool2d(FractionalMaxPool2dOptions(5).output_size(1))),
      "torch::nn::FractionalMaxPool2d()");
  // 测试 FractionalMaxPool3d 使用选项参数的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(
          FractionalMaxPool3d(FractionalMaxPool3dOptions(5).output_size(1))),
      "torch::nn::FractionalMaxPool3d()");
}
TEST_F(ModulesTest, PrettyPrintLPPool) {
  // 测试 LPPool1d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(LPPool1d(2, 5)),
      "torch::nn::LPPool1d(norm_type=2, kernel_size=5, stride=5, ceil_mode=false)");
  // 测试使用 LPPool1dOptions 构造函数创建 LPPool1d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(LPPool1d(LPPool1dOptions(1, 2).stride(5).ceil_mode(true))),
      "torch::nn::LPPool1d(norm_type=1, kernel_size=2, stride=5, ceil_mode=true)");
  // 测试 LPPool2d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(LPPool2d(2, std::vector<int64_t>({1, 2}))),
      "torch::nn::LPPool2d(norm_type=2, kernel_size=[1, 2], stride=[1, 2], ceil_mode=false)");
  // 测试使用 LPPool2dOptions 构造函数创建 LPPool2d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(LPPool2d(LPPool2dOptions(1, std::vector<int64_t>({3, 4}))
                            .stride({5, 6})
                            .ceil_mode(true))),
      "torch::nn::LPPool2d(norm_type=1, kernel_size=[3, 4], stride=[5, 6], ceil_mode=true)");
  // 测试 LPPool3d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(LPPool3d(2, std::vector<int64_t>({1, 2, 3}))),
      "torch::nn::LPPool3d(norm_type=2, kernel_size=[1, 2, 3], stride=[1, 2, 3], ceil_mode=false)");
  // 测试使用 LPPool3dOptions 构造函数创建 LPPool3d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(LPPool3d(LPPool3dOptions(1, std::vector<int64_t>({3, 4, 5}))
                            .stride({5, 6, 7})
                            .ceil_mode(true))),
      "torch::nn::LPPool3d(norm_type=1, kernel_size=[3, 4, 5], stride=[5, 6, 7], ceil_mode=true)");
}

TEST_F(ModulesTest, PrettyPrintAdaptiveMaxPool) {
  // 测试 AdaptiveMaxPool1d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool1d(5)),
      "torch::nn::AdaptiveMaxPool1d(output_size=5)");

  const auto options = AdaptiveMaxPool1dOptions(3);
  // 测试使用 AdaptiveMaxPool1dOptions 构造函数创建 AdaptiveMaxPool1d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool1d(options)),
      "torch::nn::AdaptiveMaxPool1d(output_size=3)");

  // 测试 AdaptiveMaxPool2d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(5)),
      "torch::nn::AdaptiveMaxPool2d(output_size=[5, 5])");
  // 测试使用 AdaptiveMaxPool2dOptions 构造函数创建 AdaptiveMaxPool2d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(AdaptiveMaxPool2dOptions({5, 6}))),
      "torch::nn::AdaptiveMaxPool2d(output_size=[5, 6])");
  // 测试使用 AdaptiveMaxPool2dOptions 构造函数创建 AdaptiveMaxPool2d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(AdaptiveMaxPool2dOptions({5, c10::nullopt}))),
      "torch::nn::AdaptiveMaxPool2d(output_size=[5, None])");
  // 测试使用 AdaptiveMaxPool2dOptions 构造函数创建 AdaptiveMaxPool2d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(
          AdaptiveMaxPool2dOptions({c10::nullopt, c10::nullopt}))),
      "torch::nn::AdaptiveMaxPool2d(output_size=[None, None])");

  // 测试 AdaptiveMaxPool3d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(5)),
      "torch::nn::AdaptiveMaxPool3d(output_size=[5, 5, 5])");
  // 测试使用 AdaptiveMaxPool3dOptions 构造函数创建 AdaptiveMaxPool3d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(AdaptiveMaxPool3dOptions({5, 6, 7}))),
      "torch::nn::AdaptiveMaxPool3d(output_size=[5, 6, 7])");
  // 测试使用 AdaptiveMaxPool3dOptions 构造函数创建 AdaptiveMaxPool3d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(
          AdaptiveMaxPool3d(AdaptiveMaxPool3dOptions({5, c10::nullopt, 7}))),
      "torch::nn::AdaptiveMaxPool3d(output_size=[5, None, 7])");
  // 测试使用 AdaptiveMaxPool3dOptions 构造函数创建 AdaptiveMaxPool3d 类的输出字符串是否符合预期
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(AdaptiveMaxPool3dOptions(
          {c10::nullopt, c10::nullopt, c10::nullopt}))),
      "torch::nn::AdaptiveMaxPool3d(output_size=[None, None, None])");
}
TEST_F(ModulesTest, PrettyPrintAdaptiveAvgPool) {
  // 测试 AdaptiveAvgPool1d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool1d(5)),
      "torch::nn::AdaptiveAvgPool1d(output_size=5)");

  // 测试 AdaptiveAvgPool2d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(5)),
      "torch::nn::AdaptiveAvgPool2d(output_size=[5, 5])");

  // 测试带有选项参数的 AdaptiveAvgPool2d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({5, 6}))),
      "torch::nn::AdaptiveAvgPool2d(output_size=[5, 6])");

  // 测试带有部分选项参数的 AdaptiveAvgPool2d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({5, c10::nullopt}))),
      "torch::nn::AdaptiveAvgPool2d(output_size=[5, None])");

  // 测试全为选项参数为 None 的 AdaptiveAvgPool2d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(
          AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({c10::nullopt, c10::nullopt}))),
      "torch::nn::AdaptiveAvgPool2d(output_size=[None, None])");

  // 测试 AdaptiveAvgPool3d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(5)),
      "torch::nn::AdaptiveAvgPool3d(output_size=[5, 5, 5])");

  // 测试带有选项参数的 AdaptiveAvgPool3d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions({5, 6, 7}))),
      "torch::nn::AdaptiveAvgPool3d(output_size=[5, 6, 7])");

  // 测试带有部分选项参数的 AdaptiveAvgPool3d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(
          AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions({5, c10::nullopt, 7}))),
      "torch::nn::AdaptiveAvgPool3d(output_size=[5, None, 7])");

  // 测试全为选项参数为 None 的 AdaptiveAvgPool3d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions(
          {c10::nullopt, c10::nullopt, c10::nullopt}))),
      "torch::nn::AdaptiveAvgPool3d(output_size=[None, None, None])");
}

TEST_F(ModulesTest, PrettyPrintMaxUnpool) {
  // 测试 MaxUnpool1d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(MaxUnpool1d(5)),
      "torch::nn::MaxUnpool1d(kernel_size=5, stride=5, padding=0)");

  // 测试带有选项参数的 MaxUnpool1d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(MaxUnpool1d(MaxUnpool1dOptions(5).stride(3).padding(1))),
      "torch::nn::MaxUnpool1d(kernel_size=5, stride=3, padding=1)");

  // 测试 MaxUnpool2d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(MaxUnpool2d(5)),
      "torch::nn::MaxUnpool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0])");

  // 测试带有向量选项参数的 MaxUnpool2d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(MaxUnpool2d(std::vector<int64_t>{5, 6})),
      "torch::nn::MaxUnpool2d(kernel_size=[5, 6], stride=[5, 6], padding=[0, 0])");

  // 测试带有部分选项参数的 MaxUnpool2d 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(MaxUnpool2d(MaxUnpool2dOptions(std::vector<int64_t>{5, 6})
                               .stride({3, 4})
                               .padding({1, 2}))),
      "torch::nn::MaxUnpool2d(kernel_size=[5, 6], stride=[3, 4], padding=[1, 2])");
}

TEST_F(ModulesTest, PrettyPrintDropout) {
  // 测试 Dropout 的字符串表示是否正确，未指定 inplace 参数时默认为 false
  ASSERT_EQ(c10::str(Dropout()), "torch::nn::Dropout(p=0.5, inplace=false)");

  // 测试带有概率参数的 Dropout 的字符串表示是否正确，未指定 inplace 参数时默认为 false
  ASSERT_EQ(
      c10::str(Dropout(0.42)), "torch::nn::Dropout(p=0.42, inplace=false)");

  // 测试带有选项参数的 Dropout 的字符串表示是否正确，指定了 inplace 参数为 true
  ASSERT_EQ(
      c10::str(Dropout(DropoutOptions().p(0.42).inplace(true))),
      "torch::nn::Dropout(p=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintDropout2d) {
  // 测试 Dropout2d 的字符串表示是否正确，未指定 inplace 参数时默认为 false
  ASSERT_EQ(
      c10::str(Dropout2d()), "torch::nn::Dropout2d(p=0.5, inplace=false)");

  // 测试带有概率参数的 Dropout2d 的字符串表示是否正确，未指定 inplace 参数时默认为 false
  ASSERT_EQ(
      c10::str(Dropout2d(0.42)), "torch::nn::Dropout2d(p=0.42, inplace=false)");

  // 测试带有选项参数的 Dropout2d 的字符串表示是否正确，指定了 inplace 参数为 true
  ASSERT_EQ(
      c10::str(Dropout2d(Dropout2dOptions().p(0.42).inplace(true))),
      "torch::nn::Dropout2d(p=0.42, inplace=true)");
}
// 测试模块 ModulesTest 下的 PrettyPrintDropout3d 测试用例
TEST_F(ModulesTest, PrettyPrintDropout3d) {
  // 断言 Dropout3d() 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(Dropout3d()), "torch::nn::Dropout3d(p=0.5, inplace=false)");
  // 断言 Dropout3d(0.42) 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(Dropout3d(0.42)), "torch::nn::Dropout3d(p=0.42, inplace=false)");
  // 使用 Dropout3dOptions 配置 Dropout3d(p=0.42, inplace=true)，并断言其字符串表示符合预期
  ASSERT_EQ(
      c10::str(Dropout3d(Dropout3dOptions().p(0.42).inplace(true))),
      "torch::nn::Dropout3d(p=0.42, inplace=true)");
}

// 测试模块 ModulesTest 下的 PrettyPrintFunctional 测试用例
TEST_F(ModulesTest, PrettyPrintFunctional) {
  // 断言 Functional(torch::relu) 的字符串表示符合预期
  ASSERT_EQ(c10::str(Functional(torch::relu)), "torch::nn::Functional()");
}

// 测试模块 ModulesTest 下的 PrettyPrintBatchNorm1d 测试用例
TEST_F(ModulesTest, PrettyPrintBatchNorm1d) {
  // 断言 BatchNorm1d 配置符合预期字符串表示
  ASSERT_EQ(
      c10::str(BatchNorm1d(BatchNorm1dOptions(4)
                               .eps(0.5)
                               .momentum(0.1)
                               .affine(false)
                               .track_running_stats(true))),
      "torch::nn::BatchNorm1d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

// 测试模块 ModulesTest 下的 PrettyPrintBatchNorm2d 测试用例
TEST_F(ModulesTest, PrettyPrintBatchNorm2d) {
  // 断言 BatchNorm2d 配置符合预期字符串表示
  ASSERT_EQ(
      c10::str(BatchNorm2d(BatchNorm2dOptions(4)
                               .eps(0.5)
                               .momentum(0.1)
                               .affine(false)
                               .track_running_stats(true))),
      "torch::nn::BatchNorm2d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

// 测试模块 ModulesTest 下的 PrettyPrintBatchNorm3d 测试用例
TEST_F(ModulesTest, PrettyPrintBatchNorm3d) {
  // 断言 BatchNorm3d 配置符合预期字符串表示
  ASSERT_EQ(
      c10::str(BatchNorm3d(BatchNorm3dOptions(4)
                               .eps(0.5)
                               .momentum(0.1)
                               .affine(false)
                               .track_running_stats(true))),
      "torch::nn::BatchNorm3d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

// 测试模块 ModulesTest 下的 PrettyPrintInstanceNorm1d 测试用例
TEST_F(ModulesTest, PrettyPrintInstanceNorm1d) {
  // 断言 InstanceNorm1d 配置符合预期字符串表示
  ASSERT_EQ(
      c10::str(InstanceNorm1d(InstanceNorm1dOptions(4)
                                  .eps(0.5)
                                  .momentum(0.1)
                                  .affine(false)
                                  .track_running_stats(true))),
      "torch::nn::InstanceNorm1d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

// 测试模块 ModulesTest 下的 PrettyPrintInstanceNorm2d 测试用例
TEST_F(ModulesTest, PrettyPrintInstanceNorm2d) {
  // 断言 InstanceNorm2d 配置符合预期字符串表示
  ASSERT_EQ(
      c10::str(InstanceNorm2d(InstanceNorm2dOptions(4)
                                  .eps(0.5)
                                  .momentum(0.1)
                                  .affine(false)
                                  .track_running_stats(true))),
      "torch::nn::InstanceNorm2d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

// 测试模块 ModulesTest 下的 PrettyPrintInstanceNorm3d 测试用例
TEST_F(ModulesTest, PrettyPrintInstanceNorm3d) {
  // 断言 InstanceNorm3d 配置符合预期字符串表示
  ASSERT_EQ(
      c10::str(InstanceNorm3d(InstanceNorm3dOptions(4)
                                  .eps(0.5)
                                  .momentum(0.1)
                                  .affine(false)
                                  .track_running_stats(true))),
      "torch::nn::InstanceNorm3d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}
// 测试用例：PrettyPrintLayerNorm
TEST_F(ModulesTest, PrettyPrintLayerNorm) {
  // 断言：验证 LayerNorm 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(LayerNorm(LayerNormOptions({2, 2}))),
      "torch::nn::LayerNorm([2, 2], eps=1e-05, elementwise_affine=true)");
  // 断言：验证具有指定参数的 LayerNorm 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(LayerNorm(
          LayerNormOptions({2, 2}).elementwise_affine(false).eps(2e-5))),
      "torch::nn::LayerNorm([2, 2], eps=2e-05, elementwise_affine=false)");
}

// 测试用例：PrettyPrintGroupNorm
TEST_F(ModulesTest, PrettyPrintGroupNorm) {
  // 断言：验证 GroupNorm 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(GroupNorm(GroupNormOptions(2, 2))),
      "torch::nn::GroupNorm(2, 2, eps=1e-05, affine=true)");
  // 断言：验证具有指定参数的 GroupNorm 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(GroupNorm(GroupNormOptions(2, 2).eps(2e-5).affine(false))),
      "torch::nn::GroupNorm(2, 2, eps=2e-05, affine=false)");
}

// 测试用例：PrettyPrintLocalResponseNorm
TEST_F(ModulesTest, PrettyPrintLocalResponseNorm) {
  // 断言：验证 LocalResponseNorm 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(LocalResponseNorm(LocalResponseNormOptions(2))),
      "torch::nn::LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=1)");
  // 断言：验证具有指定参数的 LocalResponseNorm 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(LocalResponseNorm(
          LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.))),
      "torch::nn::LocalResponseNorm(2, alpha=0.0002, beta=0.85, k=2)");
}

// 测试用例：PrettyPrintEmbedding
TEST_F(ModulesTest, PrettyPrintEmbedding) {
  // 断言：验证 Embedding 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2))),
      "torch::nn::Embedding(num_embeddings=10, embedding_dim=2)");
  // 断言：验证具有指定参数的 Embedding 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2).padding_idx(3).max_norm(2))),
      "torch::nn::Embedding(num_embeddings=10, embedding_dim=2, padding_idx=3, max_norm=2)");
  // 断言：验证具有多个参数的 Embedding 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2)
                             .padding_idx(3)
                             .max_norm(2)
                             .norm_type(2.5)
                             .scale_grad_by_freq(true)
                             .sparse(true))),
      "torch::nn::Embedding(num_embeddings=10, embedding_dim=2, padding_idx=3, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true)");
}
TEST_F(ModulesTest, PrettyPrintEmbeddingBag) {
  // 测试 EmbeddingBag 的字符串表示，包含默认参数
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2)");
  // 测试 EmbeddingBag 的字符串表示，包含 max_norm 参数
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2).max_norm(2))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2)");
  // 测试 EmbeddingBag 的字符串表示，包含 max_norm, norm_type, scale_grad_by_freq, sparse 参数
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2)
                                .max_norm(2)
                                .norm_type(2.5)
                                .scale_grad_by_freq(true)
                                .sparse(true))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true)");
  // 测试 EmbeddingBag 的字符串表示，包含 mode 参数
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2)
                                .max_norm(2)
                                .norm_type(2.5)
                                .scale_grad_by_freq(true)
                                .sparse(true)
                                .mode(torch::kSum))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true, mode=kSum)");
  // 测试 EmbeddingBag 的字符串表示，包含 padding_idx 参数
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2)
                                .max_norm(2)
                                .norm_type(2.5)
                                .scale_grad_by_freq(true)
                                .sparse(true)
                                .mode(torch::kSum)
                                .padding_idx(5))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true, mode=kSum, padding_idx=5)");
}

TEST_F(ModulesTest, PrettyPrintL1Loss) {
  // 测试 L1Loss 的字符串表示
  ASSERT_EQ(c10::str(L1Loss()), "torch::nn::L1Loss()");
}

TEST_F(ModulesTest, PrettyPrintKLDivLoss) {
  // 测试 KLDivLoss 的字符串表示
  ASSERT_EQ(c10::str(KLDivLoss()), "torch::nn::KLDivLoss()");
}

TEST_F(ModulesTest, PrettyPrintMSELoss) {
  // 测试 MSELoss 的字符串表示
  ASSERT_EQ(c10::str(MSELoss()), "torch::nn::MSELoss()");
}

TEST_F(ModulesTest, PrettyPrintBCELoss) {
  // 测试 BCELoss 的字符串表示
  ASSERT_EQ(c10::str(BCELoss()), "torch::nn::BCELoss()");
}

TEST_F(ModulesTest, PrettyPrintHingeEmbeddingLoss) {
  // 测试 HingeEmbeddingLoss 的字符串表示，包含 margin 参数
  ASSERT_EQ(
      c10::str(HingeEmbeddingLoss(HingeEmbeddingLossOptions().margin(4))),
      "torch::nn::HingeEmbeddingLoss(margin=4)");
}

TEST_F(ModulesTest, PrettyPrintCosineEmbeddingLoss) {
  // 测试 CosineEmbeddingLoss 的字符串表示，包含 margin 参数
  ASSERT_EQ(
      c10::str(CosineEmbeddingLoss(CosineEmbeddingLossOptions().margin(0.25))),
      "torch::nn::CosineEmbeddingLoss(margin=0.25)");
}

TEST_F(ModulesTest, PrettyPrintTripletMarginLoss) {
  // 测试 TripletMarginLoss 的字符串表示，包含 margin, p, eps, swap 参数
  ASSERT_EQ(
      c10::str(TripletMarginLoss(
          TripletMarginLossOptions().margin(3).p(2).eps(1e-06).swap(false))),
      "torch::nn::TripletMarginLoss(margin=3, p=2, eps=1e-06, swap=false)");
}
// 在 ModulesTest 测试环境中，测试 PrettyPrintTripletMarginWithDistanceLoss 函数
TEST_F(ModulesTest, PrettyPrintTripletMarginWithDistanceLoss) {
  // 创建 TripletMarginWithDistanceLossOptions 对象并配置距离函数
  auto distanceOptions = TripletMarginWithDistanceLossOptions()
                             .distance_function([&](const torch::Tensor& x,
                                                    const torch::Tensor& y) {
                               return torch::pairwise_distance(x, y, 2.0, 1e-6);
                             })
                             // 设置 margin 为 1.5
                             .margin(1.5)
                             // 允许交换样本对
                             .swap(true)
                             // 设置损失函数的减少方式为均值
                             .reduction(torch::kMean);
  // 断言 TripletMarginWithDistanceLoss 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(TripletMarginWithDistanceLoss(distanceOptions)),
      "torch::nn::TripletMarginWithDistanceLoss(margin=1.5, swap=true)");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintNLLLoss 函数
TEST_F(ModulesTest, PrettyPrintNLLLoss) {
  // 断言 NLLLoss 的字符串表示符合预期
  ASSERT_EQ(c10::str(NLLLoss()), "torch::nn::NLLLoss()");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintCrossEntropyLoss 函数
TEST_F(ModulesTest, PrettyPrinCrossEntropyLoss) {
  // 断言 CrossEntropyLoss 的字符串表示符合预期
  ASSERT_EQ(c10::str(CrossEntropyLoss()), "torch::nn::CrossEntropyLoss()");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintMultiLabelMarginLoss 函数
TEST_F(ModulesTest, PrettyPrintMultiLabelMarginLoss) {
  // 断言 MultiLabelMarginLoss 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(MultiLabelMarginLoss()), "torch::nn::MultiLabelMarginLoss()");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintMultiLabelSoftMarginLoss 函数
TEST_F(ModulesTest, PrettyPrintMultiLabelSoftMarginLoss) {
  // 断言 MultiLabelSoftMarginLoss 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(MultiLabelSoftMarginLoss()),
      "torch::nn::MultiLabelSoftMarginLoss()");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintSoftMarginLoss 函数
TEST_F(ModulesTest, PrettyPrintSoftMarginLoss) {
  // 断言 SoftMarginLoss 的字符串表示符合预期
  ASSERT_EQ(c10::str(SoftMarginLoss()), "torch::nn::SoftMarginLoss()");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintCosineSimilarity 函数
TEST_F(ModulesTest, PrettyPrintCosineSimilarity) {
  // 断言默认参数下的 CosineSimilarity 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(CosineSimilarity()),
      "torch::nn::CosineSimilarity(dim=1, eps=1e-08)");
  // 断言自定义参数下的 CosineSimilarity 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(CosineSimilarity(CosineSimilarityOptions().dim(0).eps(0.5))),
      "torch::nn::CosineSimilarity(dim=0, eps=0.5)");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintPairwiseDistance 函数
TEST_F(ModulesTest, PrettyPrintPairwiseDistance) {
  // 断言默认参数下的 PairwiseDistance 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(PairwiseDistance()),
      "torch::nn::PairwiseDistance(p=2, eps=1e-06, keepdim=false)");
  // 断言自定义参数下的 PairwiseDistance 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(PairwiseDistance(
          PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true))),
      "torch::nn::PairwiseDistance(p=3, eps=0.5, keepdim=true)");
}

// 在 ModulesTest 测试环境中，测试 PrettyPrintReflectionPad 函数
TEST_F(ModulesTest, PrettyPrintReflectionPad) {
  // 断言创建 ReflectionPad1d 对象并指定 padding 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(ReflectionPad1d(ReflectionPad1dOptions(2))),
      "torch::nn::ReflectionPad1d(padding=[2, 2])");
  // 断言创建 ReflectionPad1d 对象并指定不同 padding 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(ReflectionPad1d(ReflectionPad1dOptions({3, 1}))),
      "torch::nn::ReflectionPad1d(padding=[3, 1])");
  // 断言创建 ReflectionPad2d 对象并指定 padding 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(ReflectionPad2d(ReflectionPad2dOptions(2))),
      "torch::nn::ReflectionPad2d(padding=[2, 2, 2, 2])");
  // 断言创建 ReflectionPad2d 对象并指定不同 padding 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(ReflectionPad2d(ReflectionPad2dOptions({1, 1, 2, 0}))),
      "torch::nn::ReflectionPad2d(padding=[1, 1, 2, 0])");
}
TEST_F(ModulesTest, PrettyPrintReplicationPad) {
  // 断言检查 ReplicationPad1d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ReplicationPad1d(ReplicationPad1dOptions(2))),
      "torch::nn::ReplicationPad1d(padding=[2, 2])");
  // 断言检查 ReplicationPad1d 的字符串表示是否符合预期（传入列表形式的 padding）
  ASSERT_EQ(
      c10::str(ReplicationPad1d(ReplicationPad1dOptions({3, 1}))),
      "torch::nn::ReplicationPad1d(padding=[3, 1])");
  // 断言检查 ReplicationPad2d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ReplicationPad2d(ReplicationPad2dOptions(2))),
      "torch::nn::ReplicationPad2d(padding=[2, 2, 2, 2])");
  // 断言检查 ReplicationPad2d 的字符串表示是否符合预期（传入列表形式的 padding）
  ASSERT_EQ(
      c10::str(ReplicationPad2d(ReplicationPad2dOptions({1, 1, 2, 0}))),
      "torch::nn::ReplicationPad2d(padding=[1, 1, 2, 0])");
  // 断言检查 ReplicationPad3d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ReplicationPad3d(ReplicationPad3dOptions(1))),
      "torch::nn::ReplicationPad3d(padding=[1, 1, 1, 1, 1, 1])");
  // 断言检查 ReplicationPad3d 的字符串表示是否符合预期（传入列表形式的 padding）
  ASSERT_EQ(
      c10::str(ReplicationPad3d(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}))),
      "torch::nn::ReplicationPad3d(padding=[1, 2, 1, 2, 1, 2])");
}

TEST_F(ModulesTest, PrettyPrintZeroPad) {
  // 断言检查 ZeroPad1d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ZeroPad1d(ZeroPad1dOptions(2))),
      "torch::nn::ZeroPad1d(padding=[2, 2])");
  // 断言检查 ZeroPad1d 的字符串表示是否符合预期（传入列表形式的 padding）
  ASSERT_EQ(
      c10::str(ZeroPad1d(ZeroPad1dOptions({3, 1}))),
      "torch::nn::ZeroPad1d(padding=[3, 1])");
  // 断言检查 ZeroPad2d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ZeroPad2d(ZeroPad2dOptions(2))),
      "torch::nn::ZeroPad2d(padding=[2, 2, 2, 2])");
  // 断言检查 ZeroPad2d 的字符串表示是否符合预期（传入列表形式的 padding）
  ASSERT_EQ(
      c10::str(ZeroPad2d(ZeroPad2dOptions({1, 1, 2, 0}))),
      "torch::nn::ZeroPad2d(padding=[1, 1, 2, 0])");
  // 断言检查 ZeroPad3d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ZeroPad3d(ZeroPad3dOptions(1))),
      "torch::nn::ZeroPad3d(padding=[1, 1, 1, 1, 1, 1])");
  // 断言检查 ZeroPad3d 的字符串表示是否符合预期（传入列表形式的 padding）
  ASSERT_EQ(
      c10::str(ZeroPad3d(ZeroPad3dOptions({1, 2, 1, 2, 1, 2}))),
      "torch::nn::ZeroPad3d(padding=[1, 2, 1, 2, 1, 2])");
}

TEST_F(ModulesTest, PrettyPrintConstantPad) {
  // 断言检查 ConstantPad1d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ConstantPad1d(ConstantPad1dOptions(2, 3.5))),
      "torch::nn::ConstantPad1d(padding=[2, 2], value=3.5)");
  // 断言检查 ConstantPad1d 的字符串表示是否符合预期（传入列表形式的 padding 和值）
  ASSERT_EQ(
      c10::str(ConstantPad1d(ConstantPad1dOptions({3, 1}, 3.5))),
      "torch::nn::ConstantPad1d(padding=[3, 1], value=3.5)");
  // 断言检查 ConstantPad2d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ConstantPad2d(ConstantPad2dOptions(2, 3.5))),
      "torch::nn::ConstantPad2d(padding=[2, 2, 2, 2], value=3.5)");
  // 断言检查 ConstantPad2d 的字符串表示是否符合预期（传入列表形式的 padding 和值）
  ASSERT_EQ(
      c10::str(ConstantPad2d(ConstantPad2dOptions({3, 0, 2, 1}, 3.5))),
      "torch::nn::ConstantPad2d(padding=[3, 0, 2, 1], value=3.5)");
  // 断言检查 ConstantPad3d 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(ConstantPad3d(ConstantPad3dOptions(1, 3.5))),
      "torch::nn::ConstantPad3d(padding=[1, 1, 1, 1, 1, 1], value=3.5)");
  // 断言检查 ConstantPad3d 的字符串表示是否符合预期（传入列表形式的 padding 和值）
  ASSERT_EQ(
      c10::str(ConstantPad3d(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5))),
      "torch::nn::ConstantPad3d(padding=[1, 2, 1, 2, 1, 2], value=3.5)");
}

TEST_F(ModulesTest, PrettyPrintNestedModel) {
  struct InnerTestModule : torch::nn::Module {
    InnerTestModule()
        : torch::nn::Module("InnerTestModule"),
          fc(register_module("fc", torch::nn::Linear(3, 4))),
          table(register_module("table", torch::nn::Embedding(10, 2))) {}

    torch::nn::Linear fc;
    torch::nn::Embedding table;
  };

  struct TestModule : torch::nn::Module {
    TestModule()
        : torch::nn::Module("TestModule"),  // 构造函数，继承自 torch::nn::Module，并设置模块名称为 "TestModule"
          fc(register_module("fc", torch::nn::Linear(4, 5))),  // 注册名为 "fc" 的线性层，输入维度为 4，输出维度为 5
          table(register_module(
              "table",
              torch::nn::Embedding(EmbeddingOptions(10, 2)))),  // 注册名为 "table" 的嵌入层，设置嵌入矩阵大小为 (10, 2)
          inner(register_module("inner", std::make_shared<InnerTestModule>())) {  // 注册名为 "inner" 的子模块，类型为 InnerTestModule 的共享指针
    }

    torch::nn::Linear fc;  // 定义一个名为 fc 的成员变量，表示线性层
    torch::nn::Embedding table;  // 定义一个名为 table 的成员变量，表示嵌入层
    std::shared_ptr<InnerTestModule> inner;  // 定义一个名为 inner 的成员变量，表示 InnerTestModule 的共享指针
  };

  ASSERT_EQ(
      c10::str(TestModule{}),
      "TestModule(\n"
      "  (fc): torch::nn::Linear(in_features=4, out_features=5, bias=true)\n"
      "  (table): torch::nn::Embedding(num_embeddings=10, embedding_dim=2)\n"
      "  (inner): InnerTestModule(\n"
      "    (fc): torch::nn::Linear(in_features=3, out_features=4, bias=true)\n"
      "    (table): torch::nn::Embedding(num_embeddings=10, embedding_dim=2)\n"
      "  )\n"
      ")");
}

// 测试模块中的单元测试，验证 ELU 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintELU) {
  // 断言 ELU() 的字符串表示是否为 "torch::nn::ELU(alpha=1)"
  ASSERT_EQ(c10::str(ELU()), "torch::nn::ELU(alpha=1)");
  // 断言 ELU(alpha=42.42, inplace=true) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(ELU(ELUOptions().alpha(42.42).inplace(true))),
      "torch::nn::ELU(alpha=42.42, inplace=true)");
}

// 测试模块中的单元测试，验证 SELU 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintSELU) {
  // 断言 SELU() 的字符串表示是否为 "torch::nn::SELU()"
  ASSERT_EQ(c10::str(SELU()), "torch::nn::SELU()");
  // 断言 SELU(inplace=true) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(SELU(SELUOptions().inplace(true))),
      "torch::nn::SELU(inplace=true)");
}

// 测试模块中的单元测试，验证 GLU 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintGLU) {
  // 断言 GLU() 的字符串表示是否为 "torch::nn::GLU(dim=-1)"
  ASSERT_EQ(c10::str(GLU()), "torch::nn::GLU(dim=-1)");
  // 断言 GLU(dim=1) 的字符串表示是否正确
  ASSERT_EQ(c10::str(GLU(1)), "torch::nn::GLU(dim=1)");
}

// 测试模块中的单元测试，验证 Hardshrink 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintHardshrink) {
  // 断言 Hardshrink() 的字符串表示是否为 "torch::nn::Hardshrink(0.5)"
  ASSERT_EQ(c10::str(Hardshrink()), "torch::nn::Hardshrink(0.5)");
  // 断言 Hardshrink(lambda=42.42) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(Hardshrink(HardshrinkOptions().lambda(42.42))),
      "torch::nn::Hardshrink(42.42)");
}

// 测试模块中的单元测试，验证 Hardtanh 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintHardtanh) {
  // 断言 Hardtanh() 的字符串表示是否为 "torch::nn::Hardtanh(min_val=-1, max_val=1)"
  ASSERT_EQ(c10::str(Hardtanh()), "torch::nn::Hardtanh(min_val=-1, max_val=1)");
  // 断言 Hardtanh(min_val=-42.42, max_val=0.42, inplace=true) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(Hardtanh(
          HardtanhOptions().min_val(-42.42).max_val(0.42).inplace(true))),
      "torch::nn::Hardtanh(min_val=-42.42, max_val=0.42, inplace=true)");
}

// 测试模块中的单元测试，验证 LeakyReLU 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintLeakyReLU) {
  // 断言 LeakyReLU() 的字符串表示是否为 "torch::nn::LeakyReLU(negative_slope=0.01)"
  ASSERT_EQ(c10::str(LeakyReLU()), "torch::nn::LeakyReLU(negative_slope=0.01)");
  // 断言 LeakyReLU(negative_slope=0.42, inplace=true) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(
          LeakyReLU(LeakyReLUOptions().negative_slope(0.42).inplace(true))),
      "torch::nn::LeakyReLU(negative_slope=0.42, inplace=true)");
}

// 测试模块中的单元测试，验证 LogSigmoid 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintLogSigmoid) {
  // 断言 LogSigmoid() 的字符串表示是否为 "torch::nn::LogSigmoid()"
  ASSERT_EQ(c10::str(LogSigmoid()), "torch::nn::LogSigmoid()");
}

// 测试模块中的单元测试，验证 Softmax 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintSoftmax) {
  // 断言 Softmax(dim=1) 的字符串表示是否正确
  ASSERT_EQ(c10::str(Softmax(SoftmaxOptions(1))), "torch::nn::Softmax(dim=1)");
}

// 测试模块中的单元测试，验证 Softmin 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintSoftmin) {
  // 断言 Softmin(dim=1) 的字符串表示是否正确
  ASSERT_EQ(c10::str(Softmin(SoftminOptions(1))), "torch::nn::Softmin(dim=1)");
}

// 测试模块中的单元测试，验证 LogSoftmax 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintLogSoftmax) {
  // 断言 LogSoftmax(dim=1) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(LogSoftmax(LogSoftmaxOptions(1))),
      "torch::nn::LogSoftmax(dim=1)");
}

// 测试模块中的单元测试，验证 Softmax2d 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintSoftmax2d) {
  // 断言 Softmax2d() 的字符串表示是否为 "torch::nn::Softmax2d()"
  ASSERT_EQ(c10::str(Softmax2d()), "torch::nn::Softmax2d()");
}

// 测试模块中的单元测试，验证 PReLU 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintPReLU) {
  // 断言 PReLU(num_parameters=1) 的字符串表示是否正确
  ASSERT_EQ(c10::str(PReLU()), "torch::nn::PReLU(num_parameters=1)");
  // 断言 PReLU(num_parameters=42) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(PReLU(PReLUOptions().num_parameters(42))),
      "torch::nn::PReLU(num_parameters=42)");
}

// 测试模块中的单元测试，验证 ReLU 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintReLU) {
  // 断言 ReLU() 的字符串表示是否为 "torch::nn::ReLU()"
  ASSERT_EQ(c10::str(ReLU()), "torch::nn::ReLU()");
  // 断言 ReLU(inplace=true) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(ReLU(ReLUOptions().inplace(true))),
      "torch::nn::ReLU(inplace=true)");
  // 断言 ReLU(inplace=true) 的字符串表示是否正确（使用/*inplace=*/true注释）
  ASSERT_EQ(c10::str(ReLU(/*inplace=*/true)), "torch::nn::ReLU(inplace=true)");
}

// 测试模块中的单元测试，验证 ReLU6 函数的字符串表示是否正确
TEST_F(ModulesTest, PrettyPrintReLU6) {
  // 断言 ReLU6() 的字符串表示是否为 "torch::nn::ReLU6()"
  ASSERT_EQ(c10::str(ReLU6()), "torch::nn::ReLU6()");
  // 断言 ReLU6(inplace=true) 的字符串表示是否正确
  ASSERT_EQ(
      c10::str(ReLU6(ReLU6Options().inplace(true))),
      "torch::nn::ReLU6(inplace=true)");
  // 断言 ReLU6(inplace=true) 的字符串表示是否正确（使用/*inplace=*/true注释）
  ASSERT_EQ(
      c10::str(ReLU6(/*inplace=*/true)), "torch::nn::ReLU6(inplace=true)");
}
TEST_F(ModulesTest, PrettyPrintRReLU) {
  // 断言 RReLU 的字符串表示符合预期
  ASSERT_EQ(c10::str(RReLU()), "torch::nn::RReLU(lower=0.125, upper=0.333333)");
  // 断言具有自定义选项的 RReLU 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(RReLU(RReLUOptions().lower(0.24).upper(0.42).inplace(true))),
      "torch::nn::RReLU(lower=0.24, upper=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintCELU) {
  // 断言 CELU 的字符串表示符合预期
  ASSERT_EQ(c10::str(CELU()), "torch::nn::CELU(alpha=1)");
  // 断言具有自定义选项的 CELU 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(CELU(CELUOptions().alpha(42.42).inplace(true))),
      "torch::nn::CELU(alpha=42.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintSigmoid) {
  // 断言 Sigmoid 的字符串表示符合预期
  ASSERT_EQ(c10::str(Sigmoid()), "torch::nn::Sigmoid()");
}

TEST_F(ModulesTest, PrettyPrintPixelShuffle) {
  // 断言 PixelShuffle 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(PixelShuffle(PixelShuffleOptions(5))),
      "torch::nn::PixelShuffle(upscale_factor=5)");
}

TEST_F(ModulesTest, PrettyPrintPixelUnshuffle) {
  // 断言 PixelUnshuffle 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(PixelUnshuffle(PixelUnshuffleOptions(5))),
      "torch::nn::PixelUnshuffle(downscale_factor=5)");
}

TEST_F(ModulesTest, PrettyPrintSoftplus) {
  // 断言 Softplus 的字符串表示符合预期
  ASSERT_EQ(c10::str(Softplus()), "torch::nn::Softplus(beta=1, threshold=20)");
  // 断言具有自定义选项的 Softplus 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(Softplus(SoftplusOptions().beta(0.24).threshold(42.42))),
      "torch::nn::Softplus(beta=0.24, threshold=42.42)");
}

TEST_F(ModulesTest, PrettyPrintSoftshrink) {
  // 断言 Softshrink 的字符串表示符合预期
  ASSERT_EQ(c10::str(Softshrink()), "torch::nn::Softshrink(0.5)");
  // 断言具有自定义选项的 Softshrink 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(Softshrink(SoftshrinkOptions(42.42))),
      "torch::nn::Softshrink(42.42)");
}

TEST_F(ModulesTest, PrettyPrintSoftsign) {
  // 断言 Softsign 的字符串表示符合预期
  ASSERT_EQ(c10::str(Softsign()), "torch::nn::Softsign()");
}

TEST_F(ModulesTest, PrettyPrintTanh) {
  // 断言 Tanh 的字符串表示符合预期
  ASSERT_EQ(c10::str(Tanh()), "torch::nn::Tanh()");
}

TEST_F(ModulesTest, PrettyPrintTanhshrink) {
  // 断言 Tanhshrink 的字符串表示符合预期
  ASSERT_EQ(c10::str(Tanhshrink()), "torch::nn::Tanhshrink()");
}

TEST_F(ModulesTest, PrettyPrintThreshold) {
  // 断言具有给定阈值和值的 Threshold 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(Threshold(24.24, 42.42)),
      "torch::nn::Threshold(threshold=24.24, value=42.42)");
  // 断言具有自定义选项的 Threshold 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(Threshold(ThresholdOptions(42.42, 24.24).inplace(true))),
      "torch::nn::Threshold(threshold=42.42, value=24.24, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintCTCLoss) {
  // 断言 CTCLoss 的字符串表示符合预期
  ASSERT_EQ(c10::str(CTCLoss()), "torch::nn::CTCLoss()");
  // 断言具有自定义选项的 CTCLoss 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(
          CTCLoss(CTCLossOptions().blank(42).zero_infinity(false).reduction(
              torch::kSum))),
      "torch::nn::CTCLoss()");
}

TEST_F(ModulesTest, PrettyPrintPoissonNLLLoss) {
  // 断言 PoissonNLLLoss 的字符串表示符合预期
  ASSERT_EQ(c10::str(PoissonNLLLoss()), "torch::nn::PoissonNLLLoss()");
  // 断言具有自定义选项的 PoissonNLLLoss 的字符串表示符合预期
  ASSERT_EQ(
      c10::str(PoissonNLLLoss(PoissonNLLLossOptions()
                                  .log_input(false)
                                  .full(true)
                                  .eps(0.42)
                                  .reduction(torch::kSum))),
      "torch::nn::PoissonNLLLoss()");
}
TEST_F(ModulesTest, PrettyPrintMarginRankingLoss) {
  // 确保 MarginRankingLoss 的字符串表示正确
  ASSERT_EQ(c10::str(MarginRankingLoss()), "torch::nn::MarginRankingLoss()");
  // 确保 MarginRankingLoss 的字符串表示正确，设置 margin 和 reduction 选项
  ASSERT_EQ(
      c10::str(MarginRankingLoss(
          MarginRankingLossOptions().margin(0.5).reduction(torch::kSum))),
      "torch::nn::MarginRankingLoss()");
}

TEST_F(ModulesTest, PrettyPrintCrossMapLRN2d) {
  // 确保 CrossMapLRN2d 的字符串表示正确，指定 size 参数
  ASSERT_EQ(
      c10::str(CrossMapLRN2d(4)),
      "torch::nn::CrossMapLRN2d(4, alpha=0.0001, beta=0.75, k=1)");
  // 确保 CrossMapLRN2d 的字符串表示正确，设置 CrossMapLRN2dOptions 中的参数
  ASSERT_EQ(
      c10::str(
          CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10))),
      "torch::nn::CrossMapLRN2d(3, alpha=1e-05, beta=0.1, k=10)");
}

TEST_F(ModulesTest, PrettyPrintAlphaDropout) {
  // 确保 AlphaDropout 的字符串表示正确，使用默认参数
  ASSERT_EQ(
      c10::str(AlphaDropout()),
      "torch::nn::AlphaDropout(p=0.5, inplace=false)");
  // 确保 AlphaDropout 的字符串表示正确，设置 p 参数为 0.2
  ASSERT_EQ(
      c10::str(AlphaDropout(AlphaDropoutOptions(0.2))),
      "torch::nn::AlphaDropout(p=0.2, inplace=false)");
  // 确保 AlphaDropout 的字符串表示正确，设置 p 参数为 0.2，并启用 inplace
  ASSERT_EQ(
      c10::str(AlphaDropout(AlphaDropoutOptions(0.2).inplace(true))),
      "torch::nn::AlphaDropout(p=0.2, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintFeatureAlphaDropout) {
  // 确保 FeatureAlphaDropout 的字符串表示正确，使用默认参数
  ASSERT_EQ(
      c10::str(FeatureAlphaDropout()),
      "torch::nn::FeatureAlphaDropout(p=0.5, inplace=false)");
  // 确保 FeatureAlphaDropout 的字符串表示正确，设置 p 参数为 0.2
  ASSERT_EQ(
      c10::str(FeatureAlphaDropout(FeatureAlphaDropoutOptions(0.2))),
      "torch::nn::FeatureAlphaDropout(p=0.2, inplace=false)");
  // 确保 FeatureAlphaDropout 的字符串表示正确，设置 p 参数为 0.2，并启用 inplace
  ASSERT_EQ(
      c10::str(
          FeatureAlphaDropout(FeatureAlphaDropoutOptions(0.2).inplace(true))),
      "torch::nn::FeatureAlphaDropout(p=0.2, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintBCEWithLogitsLoss) {
  // 确保 BCEWithLogitsLoss 的字符串表示正确
  ASSERT_EQ(c10::str(BCEWithLogitsLoss()), "torch::nn::BCEWithLogitsLoss()");
  // 确保 BCEWithLogitsLoss 的字符串表示正确，设置 weight 和 pos_weight 参数，并指定 reduction
  ASSERT_EQ(
      c10::str(BCEWithLogitsLoss(BCEWithLogitsLossOptions()
                                     .weight(torch::ones({3, 3}))
                                     .pos_weight(torch::ones({3, 3}))
                                     .reduction(torch::kSum))),
      "torch::nn::BCEWithLogitsLoss()");
}

TEST_F(ModulesTest, PrettyPrintMultiheadAttention) {
  // 确保 MultiheadAttention 的字符串表示正确，指定 in_features 和 head_num 参数
  ASSERT_EQ(
      c10::str(MultiheadAttention(20, 10)),
      "torch::nn::MultiheadAttention(\n  (out_proj): torch::nn::Linear(in_features=20, out_features=20, bias=true)\n)");
  // 确保 MultiheadAttention 的字符串表示正确，设置 MultiheadAttentionOptions 中的参数
  ASSERT_EQ(
      c10::str(
          MultiheadAttention(MultiheadAttentionOptions(20, 10).bias(false))),
      "torch::nn::MultiheadAttention(\n  (out_proj): torch::nn::Linear(in_features=20, out_features=20, bias=false)\n)");
}

TEST_F(ModulesTest, PrettyPrintRNNCell) {
  // 确保 RNNCell 的字符串表示正确，指定 input_size 和 hidden_size 参数
  ASSERT_EQ(c10::str(RNNCell(20, 10)), "torch::nn::RNNCell(20, 10)");
  // 确保 RNNCell 的字符串表示正确，设置 RNNCellOptions 中的参数，并关闭 bias
  ASSERT_EQ(
      c10::str(RNNCell(
          RNNCellOptions(20, 10).bias(false).nonlinearity(torch::kTanh))),
      "torch::nn::RNNCell(20, 10, bias=false)");
  // 确保 RNNCell 的字符串表示正确，设置 RNNCellOptions 中的参数，并指定 nonlinearity 为 kReLU
  ASSERT_EQ(
      c10::str(RNNCell(
          RNNCellOptions(20, 10).bias(false).nonlinearity(torch::kReLU))),
      "torch::nn::RNNCell(20, 10, bias=false, nonlinearity=kReLU)");
}
# 在 ModulesTest 测试类中，测试 PrettyPrintLSTMCell 函数
TEST_F(ModulesTest, PrettyPrintLSTMCell) {
  # 断言输出 LSTMCell(20, 10) 的字符串表示是否与预期相符
  ASSERT_EQ(c10::str(LSTMCell(20, 10)), "torch::nn::LSTMCell(20, 10)");
  # 断言输出设置了 bias=false 的 LSTMCell(20, 10) 的字符串表示是否与预期相符
  ASSERT_EQ(
      c10::str(LSTMCell(LSTMCellOptions(20, 10).bias(false))),
      "torch::nn::LSTMCell(20, 10, bias=false)");
}

# 在 ModulesTest 测试类中，测试 PrettyPrintGRUCell 函数
TEST_F(ModulesTest, PrettyPrintGRUCell) {
  # 断言输出 GRUCell(20, 10) 的字符串表示是否与预期相符
  ASSERT_EQ(c10::str(GRUCell(20, 10)), "torch::nn::GRUCell(20, 10)");
  # 断言输出设置了 bias=false 的 GRUCell(20, 10) 的字符串表示是否与预期相符
  ASSERT_EQ(
      c10::str(GRUCell(GRUCellOptions(20, 10).bias(false))),
      "torch::nn::GRUCell(20, 10, bias=false)");
}

# 在 ModulesTest 测试类中，测试 PrettyPrintAdaptiveLogSoftmaxWithLoss 函数
TEST_F(ModulesTest, PrettyPrintAdaptiveLogSoftmaxWithLoss) {
  {
    # 创建一个 AdaptiveLogSoftmaxWithLoss 对象，并设置参数
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 4, {2}).div_value(2.));
    # 断言输出 asfm 对象的字符串表示是否与预期相符
    ASSERT_EQ(
        c10::str(asfm),
        "torch::nn::AdaptiveLogSoftmaxWithLoss(\n"
        "  (head): torch::nn::Linear(in_features=8, out_features=3, bias=false)\n"
        "  (tail): torch::nn::ModuleList(\n"
        "    (0): torch::nn::Sequential(\n"
        "      (0): torch::nn::Linear(in_features=8, out_features=4, bias=false)\n"
        "      (1): torch::nn::Linear(in_features=4, out_features=2, bias=false)\n"
        "    )\n"
        "  )\n"
        ")");
  }
  {
    # 创建另一个 AdaptiveLogSoftmaxWithLoss 对象，并设置不同的参数
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 10, {4, 8})
            .div_value(2.)
            .head_bias(true));
    # 断言输出 asfm 对象的字符串表示是否与预期相符
    ASSERT_EQ(
        c10::str(asfm),
        "torch::nn::AdaptiveLogSoftmaxWithLoss(\n"
        "  (head): torch::nn::Linear(in_features=8, out_features=6, bias=true)\n"
        "  (tail): torch::nn::ModuleList(\n"
        "    (0): torch::nn::Sequential(\n"
        "      (0): torch::nn::Linear(in_features=8, out_features=4, bias=false)\n"
        "      (1): torch::nn::Linear(in_features=4, out_features=4, bias=false)\n"
        "    )\n"
        "    (1): torch::nn::Sequential(\n"
        "      (0): torch::nn::Linear(in_features=8, out_features=2, bias=false)\n"
        "      (1): torch::nn::Linear(in_features=2, out_features=2, bias=false)\n"
        "    )\n"
        "  )\n"
        ")");
  }
}
```