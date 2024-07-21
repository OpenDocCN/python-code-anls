# `.\pytorch\test\cpp\api\init.cpp`

```
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <test/cpp/api/init_baseline.h>
#include <test/cpp/api/support.h>

#include <functional>
#include <vector>

// 检查参数张量是否与期望值张量匹配
void check_exact_values(
    const std::vector<torch::Tensor>& parameters,
    const std::vector<std::vector<torch::Tensor>>& expected_parameters) {
  
  // 断言参数数量与期望值数量相等
  ASSERT_EQ(parameters.size(), expected_parameters.size());

  // 遍历每个参数张量
  for (const auto i : c10::irange(parameters.size())) {
    auto layerParameters = parameters[i];
    auto expectedLayerParameters = expected_parameters[i];

    // 检查当前层参数张量的大小是否与期望值列表中的大小相等
    if (static_cast<size_t>(layerParameters.size(0)) !=
        expectedLayerParameters.size()) {
      std::cout << "layer #" << i
                << " layerParameters size: " << layerParameters.size(0)
                << " != "
                << " expectedLayerParameters size: "
                << expectedLayerParameters.size() << std::endl;
      // 断言失败时输出错误信息
      ASSERT_TRUE(false);
    }

    // 遍历当前层的参数张量
    for (const auto p : c10::irange(layerParameters.size(0))) {
      // 始终使用双精度浮点数类型进行比较，无论张量的原始类型是什么
      auto tensor = layerParameters[p].to(torch::kFloat64);
      auto expectedTensor = expectedLayerParameters[p].to(torch::kFloat64);

      // 如果张量不接近于期望张量，则输出错误信息
      if (!tensor.allclose(expectedTensor, /*rtol=*/1e-3, /*atol=*/5e-4)) {
        std::cout << "layer " << i << ": " << tensor << " != " << expectedTensor
                  << " (parameter " << p << ")" << std::endl;
        // 断言失败时输出错误信息
        ASSERT_TRUE(false);
      }
    }
  }
}

// 检查初始化函数是否产生与基准值匹配的参数张量
void check_initializer_against_baseline(
    std::function<void(torch::Tensor)> initializer,
    std::vector<std::vector<torch::Tensor>> expected) {
  
  // 设置随机种子
  torch::manual_seed(0);

  // 创建三个线性层并应用初始化函数
  auto layer1 = torch::nn::Linear(7, 15);
  initializer(layer1->weight);
  layer1->to(torch::kFloat64);

  auto layer2 = torch::nn::Linear(15, 15);
  initializer(layer2->weight);
  layer2->to(torch::kFloat64);

  auto layer3 = torch::nn::Linear(15, 2);
  initializer(layer3->weight);
  layer3->to(torch::kFloat64);

  // 收集所有参数张量
  auto parameters = std::vector<torch::Tensor>{
      layer1->weight,
      layer2->weight,
      layer3->weight,
  };

  // 调用检查精确值函数来比较实际参数与期望值
  check_exact_values(parameters, expected);
}

// 测试用例：检查使用 Xavier 均匀初始化的参数张量是否产生 PyTorch 中的预期值
TEST(InitTest, ProducesPyTorchValues_XavierUniform) {
  auto expected = expected_parameters::Xavier_Uniform();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::xavier_uniform_(tensor);
  };
  // 调用参数张量检查函数
  check_initializer_against_baseline(initializer, expected);
}

// 测试用例：检查使用 Xavier 正态分布初始化的参数张量是否产生 PyTorch 中的预期值
TEST(InitTest, ProducesPyTorchValues_XavierNormal) {
  auto expected = expected_parameters::Xavier_Normal();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::xavier_normal_(tensor);
  };
  // 调用参数张量检查函数
  check_initializer_against_baseline(initializer, expected);
}

// 测试用例：检查使用 Kaiming 正态分布初始化的参数张量是否产生 PyTorch 中的预期值
TEST(InitTest, ProducesPyTorchValues_KaimingNormal) {
  auto expected = expected_parameters::Kaiming_Normal();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::kaiming_normal_(tensor);
  };
  // 调用参数张量检查函数
  check_initializer_against_baseline(initializer, expected);
}
TEST(InitTest, ProducesPyTorchValues_KaimingUniform) {
  // 获取预期的参数初始化结果
  auto expected = expected_parameters::Kaiming_Uniform();
  // 定义一个初始化函数，使用 kaiming_uniform_ 初始化张量
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::kaiming_uniform_(tensor);
  };
  // 检查初始化函数与基准值的对比
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, CanInitializeTensorThatRequiresGrad) {
  // 创建一个需要梯度的空张量
  auto tensor = torch::empty({3, 4}, torch::requires_grad());
  // 确保在使用需要梯度的叶子变量进行原地操作时会抛出异常
  ASSERT_THROWS_WITH(
      tensor.fill_(1),
      "a leaf Variable that requires grad "
      "is being used in an in-place operation");
  // 初始化张量为全1，并验证其元素和
  ASSERT_EQ(torch::nn::init::ones_(tensor).sum().item<int32_t>(), 12);
}

TEST(InitTest, CalculateGainWithTanh) {
  // 计算使用 Tanh 激活函数时的增益
  double gain = torch::nn::init::calculate_gain(torch::kTanh);
  ASSERT_DOUBLE_EQ(gain, 5.0 / 3.0);
}

TEST(InitTest, CalculateGainWithRelu) {
  // 计算使用 ReLU 激活函数时的增益
  double gain = torch::nn::init::calculate_gain(torch::kReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0));
}

TEST(InitTest, CalculateGainWithLeakyRelu) {
  // 计算使用 Leaky ReLU 激活函数时的增益
  double gain = torch::nn::init::calculate_gain(torch::kLeakyReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0 / (1 + pow(0.01, 2))));
}

TEST(InitTest, CanInitializeCnnWithOrthogonal) {
  // 创建一个 2D 卷积层，并使用 orthogonal_ 初始化权重参数
  torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(3, 2, 3).stride(2));
  torch::nn::init::orthogonal_(conv_layer->named_parameters()["weight"]);
}
```