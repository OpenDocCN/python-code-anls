# `.\pytorch\test\cpp\api\optim.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架头文件

#include <c10/util/irange.h>  // 引入 Caffe2 的 irange 实用工具
#include <torch/torch.h>      // 引入 PyTorch 核心头文件

#include <test/cpp/api/optim_baseline.h>  // 引入优化器基线测试的头文件
#include <test/cpp/api/support.h>         // 引入测试支持函数的头文件

#include <cmath>       // 引入数学库头文件
#include <cstdlib>     // 引入标准库头文件
#include <functional>  // 引入函数式编程相关头文件
#include <iostream>    // 引入输入输出流库头文件
#include <memory>      // 引入内存管理相关头文件
#include <random>      // 引入随机数生成库头文件
#include <vector>      // 引入向量容器头文件

using namespace torch::nn;     // 使用 PyTorch 的神经网络命名空间
using namespace torch::optim;  // 使用 PyTorch 的优化器命名空间

// 定义通用模板函数，用于测试 XOR 问题的优化器
template <typename OptimizerClass, typename Options>
bool test_optimizer_xor(Options options) {
  torch::manual_seed(0);  // 设置随机数种子为0，确保结果可复现

  // 创建一个顺序模型，包含两个线性层和两个 Sigmoid 函数
  Sequential model(
      Linear(2, 8),
      Functional(torch::sigmoid),
      Linear(8, 1),
      Functional(torch::sigmoid));

  const int64_t kBatchSize = 200;               // 定义批处理大小
  const int64_t kMaximumNumberOfEpochs = 3000;  // 定义最大迭代次数

  OptimizerClass optimizer(model->parameters(), options);  // 创建优化器对象

  float running_loss = 1;  // 初始化运行损失为1
  int epoch = 0;           // 初始化迭代次数为0
  while (running_loss > 0.1) {  // 当运行损失大于0.1时循环
    auto inputs = torch::empty({kBatchSize, 2});  // 创建空的输入张量
    auto labels = torch::empty({kBatchSize});     // 创建空的标签张量
    for (const auto i : c10::irange(kBatchSize)) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);  // 在[0, 2)内随机生成两个整数，作为输入的两个特征
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();  // 计算对应的标签值（XOR逻辑运算）
    }

    inputs.set_requires_grad(true);  // 设置输入张量需要梯度计算

    // 定义优化步骤的闭包函数
    auto step = [&](OptimizerClass& optimizer,
                    Sequential model,
                    torch::Tensor inputs,
                    torch::Tensor labels) {
      auto closure = [&]() {
        optimizer.zero_grad();         // 清除优化器的梯度
        auto x = model->forward(inputs);  // 前向传播计算模型输出
        auto loss = torch::binary_cross_entropy(x, labels);  // 计算二元交叉熵损失
        loss.backward();               // 反向传播计算梯度
        return loss;                   // 返回损失值
      };
      return optimizer.step(closure);  // 执行优化器的一步优化过程
    };

    torch::Tensor loss = step(optimizer, model, inputs, labels);  // 执行优化步骤，获取损失值

    // 更新运行损失为当前损失的指数加权移动平均值
    running_loss = running_loss * 0.99 + loss.item<float>() * 0.01;

    // 如果超过最大迭代次数，则打印错误信息并返回false
    if (epoch > kMaximumNumberOfEpochs) {
      std::cout << "Loss is too high after epoch " << epoch << ": "
                << running_loss << std::endl;
      return false;
    }
    epoch++;  // 增加迭代次数计数器
  }
  return true;  // 如果成功收敛，则返回true
}

// 定义模板函数，用于将新张量赋值给给定名称的模型参数
template <typename Parameters>
void assign_parameter(
    const Parameters& parameters,
    const char* name,
    torch::Tensor new_tensor) {
  auto parameter = parameters[name];         // 获取指定名称的参数
  parameter.set_requires_grad(false);        // 关闭参数的梯度计算
  parameter.flatten().copy_(new_tensor);     // 将新张量扁平化后复制给参数
  parameter.set_requires_grad(true);         // 开启参数的梯度计算
}

// 定义模板函数，用于检查优化器的精确数值
template <typename OptimizerClass, typename Options>
void check_exact_values(
    Options options,
    // 定义函数，接受一个模型和预期参数列表作为输入
    void train_model(
        Sequential model,
        std::vector<std::vector<torch::Tensor>> expected_parameters) {
      // 定义训练迭代次数和每隔多少次取样
      const size_t kIterations = 1001;
      const size_t kSampleEvery = 100;
    
      // 设定随机种子为0，确保实验可重复性
      torch::manual_seed(0);
    
      // 定义模型结构：输入层(2个节点) -> 隐藏层(3个节点，使用sigmoid激活函数) -> 输出层(1个节点，使用sigmoid激活函数)
      Sequential model(
          Linear(2, 3),
          Functional(torch::sigmoid),
          Linear(3, 1),
          Functional(torch::sigmoid));
    
      // 将模型参数类型转换为Float64
      model->to(torch::kFloat64);
    
      // 获取模型的命名参数列表
      auto parameters = model->named_parameters();
    
      // 分配具体的参数数值给模型的各个命名参数
      assign_parameter(
          parameters,
          "0.weight",
          torch::tensor(
              {-0.2109, -0.4976, -0.1413, -0.3420, -0.2524, 0.6976},
              torch::kFloat64));
      assign_parameter(
          parameters,
          "0.bias",
          torch::tensor({-0.1085, -0.2979, 0.6892}, torch::kFloat64));
      assign_parameter(
          parameters,
          "2.weight",
          torch::tensor({-0.0508, -0.3941, -0.2843}, torch::kFloat64));
      assign_parameter(
          parameters, "2.bias", torch::tensor({-0.0711}, torch::kFloat64));
    
      // 创建优化器对象，传入模型参数和选项
      auto optimizer = OptimizerClass(parameters.values(), options);
    
      // 定义模型的输入数据，类型为Float64，形状为(3, 2)
      torch::Tensor input =
          torch::tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, torch::kFloat64)
              .reshape({3, 2});
    
      // 开始迭代训练
      for (const auto i : c10::irange(kIterations)) {
        // 清除优化器的梯度
        optimizer.zero_grad();
        // 前向传播
        auto output = model->forward(input);
        // 计算损失函数（这里是简单的输出求和）
        auto loss = output.sum();
        // 反向传播
        loss.backward();
    
        // 定义闭包函数，返回固定值10的张量
        auto closure = []() { return torch::tensor({10}); };
        // 执行优化步骤，传入闭包函数
        optimizer.step(closure);
    
        // 每隔kSampleEvery次迭代进行一次参数比较
        if (i % kSampleEvery == 0) {
          // 断言预期参数的维度与当前参数维度相同
          ASSERT_TRUE(
              expected_parameters.at(i / kSampleEvery).size() == parameters.size());
          // 遍历模型的参数
          for (const auto p : c10::irange(parameters.size())) {
            // 断言当前参数已经被定义
            ASSERT_TRUE(parameters[p]->defined());
            // 将计算得到的参数展平并转换为Float64类型，以便比较
            auto computed = parameters[p]->flatten().to(torch::kFloat64);
            // 获取预期的参数，并转换为Float64类型
            auto expected =
                expected_parameters.at(i / kSampleEvery).at(p).to(torch::kFloat64);
            // 如果计算得到的参数与预期参数不相等，则输出错误信息并断言失败
            if (!computed.allclose(expected, /*rtol=*/1e-3, /*atol=*/5e-4)) {
              std::cout << "Iteration " << i << ": " << computed
                        << " != " << expected << " (parameter " << p << ")"
                        << std::endl;
              ASSERT_TRUE(false);
            }
          }
        }
      }
    }
TEST(OptimTest,`
TEST(OptimTest, OptimizerAccessors) {
  // 创建 Adagrad 优化器选项，学习率为 1.0
  auto options = AdagradOptions(1.0);
  // 定义一个 torch 张量参数的向量
  std::vector<torch::Tensor> params;
  // 遍历 3 次，生成随机张量，并添加到 params 中
  for (const auto i : c10::irange(3)) {
    (void)i; // 忽略未使用变量的警告
    params.push_back(torch::randn(10)); // 生成 10 元素的随机张量
  }
  // 创建 Adagrad 优化器，传入参数和选项
  auto optimizer = Adagrad(params, options);
  // 测试 defaults() 方法，使用非 const 引用
  auto& options_ = static_cast<AdagradOptions&>(optimizer.defaults());
  ASSERT_TRUE(options == options_); // 确认选项相等
  // 测试 param_groups() 方法，使用非 const 引用返回
  auto& params_groups = optimizer.param_groups();
  // NOLINTNEXTLINE(modernize-use-emplace)
  params_groups.push_back(OptimizerParamGroup(params)); // 添加参数组
  auto& params_1 = params_groups[1].params(); // 获取第二个参数组的参数
  // 遍历参数组，验证参数是否相等
  for (const auto i : c10::irange(params_1.size())) {
    torch::equal(params[i], params_1[i]);
  }

  // 测试 add_param_group() 方法，当新参数组中包含已存在的参数时，应该抛出异常
  ASSERT_THROWS_WITH(
      optimizer.add_param_group(OptimizerParamGroup(params)),
      "some parameters appear in more than one parameter group");

  // 测试 state() 方法，使用非 const 引用返回
  auto& state_ = static_cast<AdagradParamState&>(
      *(optimizer.state()[params_1[0].unsafeGetTensorImpl()]));
  state_.step(state_.step() + 1); // 更新状态步长

  // 创建新的 Adagrad 优化器实例，验证默认选项
  const auto& optimizer_ = Adagrad(params, options);
  optimizer_.defaults();
  // 测试 param_groups() 方法，使用 const 引用返回
  (void)optimizer_.param_groups();
  // 测试 state() 方法，使用 const 引用返回
  optimizer_.state();
}

#define OLD_INTERFACE_WARNING_CHECK(func)       \
  {                                             \
    torch::test::WarningCapture warnings;       \
    func;                                       \
    ASSERT_EQ(                                  \
        torch::test::count_substr_occurrences(  \
            warnings.str(), "will be removed"), \
        1);                                     \
  }

struct MyOptimizerOptions
    : public OptimizerCloneableOptions<MyOptimizerOptions> {
  // 构造函数，默认学习率为 1.0
  MyOptimizerOptions(double lr = 1.0) : lr_(lr){};
  TORCH_ARG(double, lr) = 1.0; // 定义学习率参数
};

TEST(OptimTest, OldInterface) {
  struct MyOptimizer : Optimizer {
    using Optimizer::Optimizer;
    // 重载 step 方法，接收一个 LossClosure 闭包，返回一个张量
    torch::Tensor step(LossClosure closure = nullptr) override {
      return {};
    }
    // 构造函数，接受参数和选项
    explicit MyOptimizer(
        std::vector<at::Tensor> params,
        MyOptimizerOptions defaults = {})
        : // NOLINTNEXTLINE(performance-move-const-arg)
          Optimizer(
              {std::move(OptimizerParamGroup(params))},
              std::make_unique<MyOptimizerOptions>(defaults)) {}
  };
  // 定义一个参数张量的向量
  std::vector<torch::Tensor> parameters = {
      torch::ones({2, 3}), torch::zeros({2, 3}), torch::rand({2, 3})};
  {
    // 创建 MyOptimizer 优化器实例
    MyOptimizer optimizer(parameters);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t size;
    // 检查旧接口的警告信息
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, parameters.size()); // 确认参数数量一致
  }
  {
    // 创建一个空的张量向量，用于存储模型参数
    std::vector<at::Tensor> params;
    // 使用 params 初始化一个自定义优化器对象
    MyOptimizer optimizer(params);

    // 定义一个变量 size，用于存储优化器的大小
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t size;
    // 警告检查：旧接口警告，设置 size 为 optimizer 的大小
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    // 断言：确保优化器的大小为 0
    ASSERT_EQ(size, 0);

    // 警告检查：旧接口警告，添加参数到优化器中
    OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

    // 警告检查：旧接口警告，设置 size 为 optimizer 的大小
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    // 断言：确保优化器的大小与参数列表的大小相等
    ASSERT_EQ(size, parameters.size());

    // 创建一个新的张量向量 params_
    std::vector<torch::Tensor> params_;
    // 警告检查：旧接口警告，获取优化器的参数列表
    OLD_INTERFACE_WARNING_CHECK(params_ = optimizer.parameters());
    // 遍历参数列表中的每个参数
    for (const auto p : c10::irange(size)) {
      // 断言：确保每个参数在数值上都接近其对应的输入参数
      ASSERT_TRUE(params_[p].allclose(parameters[p]));
    }
  }
  {
    // 创建一个线性层对象 linear，输入维度为 3，输出维度为 4
    Linear linear(3, 4);
    // 使用 linear 的参数初始化一个新的自定义优化器对象
    MyOptimizer optimizer(linear->parameters());

    // 定义一个变量 size，用于存储优化器的大小
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t size;
    // 警告检查：旧接口警告，设置 size 为 optimizer 的大小
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    // 断言：确保优化器的大小与 linear 的参数列表大小相等
    ASSERT_EQ(size, linear->parameters().size());
  }
TEST(OptimTest, XORConvergence_SGD) {
  // 对 SGD 优化器进行 XOR 收敛性测试，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<SGD>(
      SGDOptions(0.1).momentum(0.9).nesterov(true).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_LBFGS) {
  // 对 LBFGS 优化器进行 XOR 收敛性测试，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(LBFGSOptions(1.0)));
  // 对 LBFGS 优化器进行 XOR 收敛性测试，使用强 Wolfe 线搜索函数，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(
      LBFGSOptions(1.0).line_search_fn("strong_wolfe")));
}

TEST(OptimTest, XORConvergence_Adagrad) {
  // 对 Adagrad 优化器进行 XOR 收敛性测试，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3)));
}

TEST(OptimTest, XORConvergence_RMSprop) {
  // 对 RMSprop 优化器进行 XOR 收敛性测试，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(RMSpropOptions(0.1).centered(true)));
}

TEST(OptimTest, XORConvergence_RMSpropWithMomentum) {
  // 对 RMSprop 优化器进行 XOR 收敛性测试，使用动量，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(
      RMSpropOptions(0.1).momentum(0.9).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_Adam) {
  // 对 Adam 优化器进行 XOR 收敛性测试，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<Adam>(AdamOptions(0.1).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_AdamWithAmsgrad) {
  // 对 Adam 优化器进行 XOR 收敛性测试，启用 AMSGrad，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<Adam>(
      AdamOptions(0.1).weight_decay(1e-6).amsgrad(true)));
}

TEST(OptimTest, ProducesPyTorchValues_Adam) {
  // 验证 Adam 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<Adam>(AdamOptions(1.0), expected_parameters::Adam());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWithWeightDecay) {
  // 验证带权重衰减的 Adam 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adam_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWithWeightDecayAndAMSGrad) {
  // 验证带权重衰减和启用 AMSGrad 的 Adam 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-6).amsgrad(true),
      expected_parameters::Adam_with_weight_decay_and_amsgrad());
}

TEST(OptimTest, XORConvergence_AdamW) {
  // 对 AdamW 优化器进行 XOR 收敛性测试，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<AdamW>(AdamWOptions(0.1)));
}

TEST(OptimTest, XORConvergence_AdamWWithAmsgrad) {
  // 对 AdamW 优化器进行 XOR 收敛性测试，启用 AMSGrad，并设置相关选项
  ASSERT_TRUE(test_optimizer_xor<AdamW>(AdamWOptions(0.1).amsgrad(true)));
}

TEST(OptimTest, ProducesPyTorchValues_AdamW) {
  // 验证 AdamW 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<AdamW>(AdamWOptions(1.0), expected_parameters::AdamW());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWWithoutWeightDecay) {
  // 验证不带权重衰减的 AdamW 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<AdamW>(
      AdamWOptions(1.0).weight_decay(0),
      expected_parameters::AdamW_without_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWWithAMSGrad) {
  // 验证启用 AMSGrad 的 AdamW 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<AdamW>(
      AdamWOptions(1.0).amsgrad(true),
      expected_parameters::AdamW_with_amsgrad());
}

TEST(OptimTest, ProducesPyTorchValues_Adagrad) {
  // 验证 Adagrad 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<Adagrad>(
      AdagradOptions(1.0), expected_parameters::Adagrad());
}

TEST(OptimTest, ProducesPyTorchValues_AdagradWithWeightDecay) {
  // 验证带权重衰减的 Adagrad 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adagrad_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdagradWithWeightDecayAndLRDecay) {
  // 验证带权重衰减和学习率衰减的 Adagrad 优化器生成的参数值与预期的 PyTorch 值匹配
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3),
      expected_parameters::Adagrad_with_weight_decay_and_lr_decay());
}
TEST(OptimTest, ProducesPyTorchValues_RMSprop) {
  // 调用模板函数，检查 RMSprop 优化器生成的参数是否与预期值匹配
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1), expected_parameters::RMSprop());
}

TEST(OptimTest, ProducesPyTorchValues_RMSpropWithWeightDecay) {
  // 调用模板函数，检查带权重衰减的 RMSprop 优化器生成的参数是否与预期值匹配
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-2),
      expected_parameters::RMSprop_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_RMSpropWithWeightDecayAndCentered) {
  // 调用模板函数，检查带权重衰减和中心化的 RMSprop 优化器生成的参数是否与预期值匹配
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true),
      expected_parameters::RMSprop_with_weight_decay_and_centered());
}

TEST(
    OptimTest,
    ProducesPyTorchValues_RMSpropWithWeightDecayAndCenteredAndMomentum) {
  // 调用模板函数，检查带权重衰减、中心化和动量的 RMSprop 优化器生成的参数是否与预期值匹配
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true).momentum(0.9),
      expected_parameters::
          RMSprop_with_weight_decay_and_centered_and_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_SGD) {
  // 调用模板函数，检查 SGD 优化器生成的参数是否与预期值匹配
  check_exact_values<SGD>(SGDOptions(0.1), expected_parameters::SGD());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecay) {
  // 调用模板函数，检查带权重衰减的 SGD 优化器生成的参数是否与预期值匹配
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2),
      expected_parameters::SGD_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecayAndMomentum) {
  // 调用模板函数，检查带权重衰减和动量的 SGD 优化器生成的参数是否与预期值匹配
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2).momentum(0.9),
      expected_parameters::SGD_with_weight_decay_and_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecayAndNesterovMomentum) {
  // 调用模板函数，检查带权重衰减、动量和 Nesterov 动量的 SGD 优化器生成的参数是否与预期值匹配
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-6).momentum(0.9).nesterov(true),
      expected_parameters::SGD_with_weight_decay_and_nesterov_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_LBFGS) {
  // 调用模板函数，检查 LBFGS 优化器生成的参数是否与预期值匹配
  check_exact_values<LBFGS>(LBFGSOptions(1.0), expected_parameters::LBFGS());
}

TEST(OptimTest, ProducesPyTorchValues_LBFGS_with_line_search) {
  // 调用模板函数，检查带线搜索的 LBFGS 优化器生成的参数是否与预期值匹配
  check_exact_values<LBFGS>(
      LBFGSOptions(1.0).line_search_fn("strong_wolfe"),
      expected_parameters::LBFGS_with_line_search());
}

TEST(OptimTest, ZeroGrad) {
  // 设置随机种子为0
  torch::manual_seed(0);

  // 创建一个线性模型，输入大小为2，输出大小为8
  Linear model(2, 8);
  // 使用 SGD 优化器，学习率为0.1，优化模型的参数
  SGD optimizer(model->parameters(), 0.1);

  // 验证模型的所有参数的梯度是否未定义
  for (const auto& parameter : model->parameters()) {
    ASSERT_FALSE(parameter.grad().defined());
  }

  // 对模型进行前向传播，生成输出
  auto output = model->forward(torch::ones({5, 2}));
  // 计算损失函数的和
  auto loss = output.sum();
  // 反向传播计算梯度
  loss.backward();

  // 验证模型的所有参数的梯度是否已定义且梯度和大于0
  for (const auto& parameter : model->parameters()) {
    ASSERT_TRUE(parameter.grad().defined());
    ASSERT_GT(parameter.grad().sum().item<float>(), 0);
  }

  // 清零优化器的梯度
  optimizer.zero_grad();

  // 验证模型的所有参数的梯度是否未定义
  for (const auto& parameter : model->parameters()) {
    ASSERT_FALSE(parameter.grad().defined());
  }
}
TEST(OptimTest, ExternalVectorOfParameters) {
  // 设置随机种子为0，确保可复现性
  torch::manual_seed(0);

  // 创建包含三个不同形状张量的参数向量
  std::vector<torch::Tensor> parameters = {
      torch::randn({2, 2}), torch::randn({3, 3}), torch::randn({4, 4})};
  // 创建参数向量的副本以备后用
  std::vector<torch::Tensor> original_parameters = {
      parameters[0].clone(), parameters[1].clone(), parameters[2].clone()};

  // 将所有参数的梯度设置为1
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = torch::ones_like(parameter);
  }

  // 使用学习率1.0创建SGD优化器对象
  SGD optimizer(parameters, 1.0);

  // 执行一步优化
  optimizer.step();

  // 断言每个参数是否符合预期（原始参数减去1.0）
  ASSERT_TRUE(parameters[0].allclose(original_parameters[0] - 1.0));
  ASSERT_TRUE(parameters[1].allclose(original_parameters[1] - 1.0));
  ASSERT_TRUE(parameters[2].allclose(original_parameters[2] - 1.0));
}

TEST(OptimTest, AddParameter_LBFGS) {
  // 设置随机种子为0，确保可复现性
  torch::manual_seed(0);

  // 创建包含一个5x5形状张量的参数向量
  std::vector<torch::Tensor> parameters = {torch::randn({5, 5})};
  // 创建参数向量的副本以备后用
  std::vector<torch::Tensor> original_parameters = {parameters[0].clone()};

  // 将所有参数的梯度设置为1
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = torch::ones_like(parameter);
  }

  // 创建LBFGS优化器对象，不添加任何参数
  LBFGS optimizer(std::vector<torch::Tensor>{}, 1.0);
  // 警告：使用旧接口，向优化器添加参数
  OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

  // 执行一步优化，传递一个lambda函数，返回张量1
  optimizer.step([]() { return torch::tensor(1); });

  // 确保此处不会抛出异常
}

void check_lr_change(
    Optimizer& optimizer,
    LRScheduler& lr_scheduler,
    std::map<unsigned, double> expected_epoch_lrs) {
  // 查找预期学习率映射中的最大迭代次数
  unsigned kIterations = std::max_element(
                             expected_epoch_lrs.begin(),
                             expected_epoch_lrs.end(),
                             [](const std::pair<unsigned, double>& a,
                                const std::pair<unsigned, double>& b) -> bool {
                               return a.second > b.second;
                             })
                             ->first;

  // 遍历迭代次数范围内的每个迭代
  for (unsigned i = 0; i <= kIterations; i++) {
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if (epoch_iter != expected_epoch_lrs.end()) {
      // 比较优化器参数组的学习率与预期学习率的相似性
      ASSERT_TRUE(
          fabs(
              epoch_iter->second -
              optimizer.param_groups()[0].options().get_lr()) <
          std::numeric_limits<double>::epsilon());
    }
    // 执行一步优化
    optimizer.step();
    // 更新学习率调度器状态
    lr_scheduler.step();
  }
}

void check_lr_change_for_reduce_on_plateau(
    Optimizer& optimizer,
    ReduceLROnPlateauScheduler& lr_scheduler,
    std::vector<double> metrics,
    std::map<unsigned, double> expected_epoch_lrs) {
  // 与check_lr_change类似，但针对ReduceLROnPlateauScheduler
  unsigned kIterations = std::max_element(
                             expected_epoch_lrs.begin(),
                             expected_epoch_lrs.end(),
                             [](const std::pair<unsigned, double>& a,
                                const std::pair<unsigned, double>& b) -> bool {
                               return a.second > b.second;
                             })
                             ->first;

  for (unsigned i = 0; i <= kIterations; i++) {
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if (epoch_iter != expected_epoch_lrs.end()) {
      // 比较优化器参数组的学习率与预期学习率的相似性
      ASSERT_TRUE(
          fabs(
              epoch_iter->second -
              optimizer.param_groups()[0].options().get_lr()) <
          std::numeric_limits<double>::epsilon());
    }
    // 执行一步优化，使用指定的度量数据
    optimizer.step(metrics);
    // 更新学习率调度器状态
    lr_scheduler.step();
  }
}


这些注释解释了每个测试函数和函数中的关键步骤，帮助理解每行代码的作用和意图。
    // 根据期望的 epoch 学习率映射表，找到最大的 epoch 值
    std::map<unsigned, double> expected_epoch_lrs) {
  
  // 使用 lambda 表达式找到期望 epoch 学习率映射表中值最大的项的键
  unsigned kIterations = std::max_element(
                             expected_epoch_lrs.begin(),
                             expected_epoch_lrs.end(),
                             [](const std::pair<unsigned, double>& a,
                                const std::pair<unsigned, double>& b) -> bool {
                               return a.second > b.second;
                             })
                             ->first;

  // 循环迭代直到最大 epoch 值
  for (unsigned i = 0; i <= kIterations; i++) {
    // 在期望的 epoch 学习率映射表中查找当前 epoch 的条目
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if (epoch_iter != expected_epoch_lrs.end()) {
      // 比较两个浮点数学习率的相似性
      ASSERT_TRUE(
          fabs(
              epoch_iter->second - 
              optimizer.param_groups()[0].options().get_lr()) <
          std::numeric_limits<double>::epsilon());
    }
    // 执行优化器的步骤
    optimizer.step();
    // 调整学习率调度器的步数
    lr_scheduler.step(5.0);
  }
}

// 定义一个测试用例，验证 StepLR 调度器在 Adam 优化器上的工作是否正确
TEST(OptimTest, CheckLRChange_StepLR_Adam) {
  // 创建一个包含单个参数的张量，初始化为零
  torch::Tensor parameters = torch::zeros({1});
  // 使用 Adam 优化器初始化一个优化器对象，设置学习率为 1e-3
  auto optimizer = Adam({parameters}, AdamOptions().lr(1e-3));

  // 设置步长和衰减系数
  const unsigned step_size = 20;
  const double gamma = 0.5;
  // 创建 StepLR 调度器对象
  StepLR step_lr_scheduler(optimizer, step_size, gamma);

  // 预期的学习率变化映射，第一个参数表示 epoch，第二个参数表示对应的学习率
  const std::map<unsigned, double> expected_epoch_lrs = {{1, 1e-3}, {25, 5e-4}};

  // 调用函数检查学习率变化是否符合预期
  check_lr_change(optimizer, step_lr_scheduler, expected_epoch_lrs);
}

// 定义一个测试用例，验证 ReduceLROnPlateau 调度器在 Adam 优化器上的工作是否正确
TEST(OptimTest, CheckLRChange_ReduceLROnPlateau_Adam) {
  // 创建一个包含单个参数的张量，初始化为零
  torch::Tensor parameters = torch::zeros({1});
  // 使用 Adam 优化器初始化一个优化器对象，设置学习率为 1e-3
  auto optimizer = Adam({parameters}, AdamOptions().lr(1e-3));
  
  // 设置降低学习率的因子和等待周期
  const float factor = 0.5;
  const int patience = 20;
  // 创建 ReduceLROnPlateauScheduler 对象
  ReduceLROnPlateauScheduler reduce_lr_on_plateau_scheduler(
      optimizer,
      ReduceLROnPlateauScheduler::SchedulerMode::min,
      factor,
      patience);

  // 预期的学习率变化映射，第一个参数表示 epoch，第二个参数表示对应的学习率
  const std::map<unsigned, double> expected_epoch_lrs = {{1, 1e-3}, {25, 5e-4}};

  // 调用函数检查学习率变化是否符合预期
  check_lr_change_for_reduce_on_plateau(
      optimizer, reduce_lr_on_plateau_scheduler, expected_epoch_lrs);
}
```