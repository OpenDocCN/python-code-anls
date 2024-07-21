# `.\pytorch\test\cpp\api\parallel.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/util/irange.h>  // 引入 C10 库中的 irange.h 头文件
#include <torch/csrc/autograd/functions/comm.h>  // 引入 Torch 自动微分模块中的 comm.h 头文件
#include <torch/nn/module.h>  // 引入 Torch 神经网络模块的 module.h 头文件
#include <torch/nn/modules/conv.h>  // 引入 Torch 神经网络模块中的卷积层 conv.h 头文件
#include <torch/nn/modules/linear.h>  // 引入 Torch 神经网络模块中的线性层 linear.h 头文件
#include <torch/nn/parallel/data_parallel.h>  // 引入 Torch 神经网络模块中的数据并行 data_parallel.h 头文件
#include <torch/nn/pimpl.h>  // 引入 Torch 神经网络模块的 pimpl.h 头文件
#include <torch/optim/sgd.h>  // 引入 Torch 优化器中的 SGD 头文件
#include <torch/types.h>  // 引入 Torch 的数据类型定义头文件
#include <torch/utils.h>  // 引入 Torch 的实用工具头文件

#include <test/cpp/api/support.h>  // 引入测试支持相关的头文件

#include <iostream>  // 引入标准输入输出流库
#include <memory>  // 引入内存管理相关的头文件
#include <utility>  // 引入常用的实用程序组件头文件
#include <vector>  // 引入向量容器头文件

using namespace torch::autograd;  // 使用 Torch 自动微分命名空间
using namespace torch::nn;  // 使用 Torch 神经网络模块命名空间

struct ParallelTest : torch::test::SeedingFixture {};  // 定义并声明 ParallelTest 结构体，继承自 SeedingFixture

TEST_F(ParallelTest, DifferentiableScatter_MultiCUDA) {  // 定义测试用例 DifferentiableScatter_MultiCUDA
  Scatter scatter(  // 创建 Scatter 对象，用于数据分散
      {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});  // 设置 Scatter 对象使用的 CUDA 设备列表

  auto input = torch::ones(10, torch::requires_grad(true));  // 创建需要求梯度的全一张量 input
  auto output = scatter.apply({input});  // 对 input 进行 Scatter 操作，得到 output

  ASSERT_EQ(output.size(), 2);  // 断言 output 的大小为 2
  ASSERT_EQ(output[0].size(0), 5);  // 断言 output[0] 的第一个维度大小为 5
  ASSERT_EQ(output[1].size(0), 5);  // 断言 output[1] 的第一个维度大小为 5

  ASSERT_TRUE(torch::cat({output[0].to(torch::kCPU), output[1].to(torch::kCPU)})  // 对 output[0] 和 output[1] 进行 CPU 转换后拼接，并与 input 进行比较
                  .allclose(input));

  torch::Tensor sum = output[0].to({torch::kCUDA, 1}) + output[1];  // 计算 output[0] 在 CUDA 设备 1 上的和与 output[1] 的和
  sum.backward(torch::ones_like(sum));  // 对 sum 进行反向传播

  ASSERT_TRUE(input.grad().defined());  // 断言 input 的梯度已经定义
  ASSERT_TRUE(input.grad().device().is_cpu());  // 断言 input 的梯度在 CPU 设备上
  ASSERT_EQ(input.grad().sum().item<int32_t>(), 10);  // 断言 input 梯度总和为 10
}

TEST_F(ParallelTest, DifferentiableGather_MultiCUDA) {  // 定义测试用例 DifferentiableGather_MultiCUDA
  Gather gather(torch::Device(torch::kCUDA, 1));  // 创建 Gather 对象，用于数据聚集

  auto a = torch::ones(5, torch::requires_grad(true).device(torch::kCUDA, 0));  // 创建需要求梯度的全一张量 a，并指定在 CUDA 设备 0 上
  auto b = torch::ones(5, torch::requires_grad(true).device(torch::kCUDA, 1));  // 创建需要求梯度的全一张量 b，并指定在 CUDA 设备 1 上

  auto outputs = gather.apply({a, b});  // 对 a 和 b 进行 Gather 操作，得到 outputs
  ASSERT_EQ(outputs.size(), 1);  // 断言 outputs 的大小为 1
  torch::Tensor output = outputs.front();  // 获取 outputs 中的第一个张量

  ASSERT_EQ(output.size(0), 10);  // 断言 output 的第一个维度大小为 10
  ASSERT_EQ(output.device(), torch::Device(torch::kCUDA, 1));  // 断言 output 的设备为 CUDA 设备 1

  auto chunks = output.chunk(2);  // 将 output 分割为两个块
  ASSERT_TRUE(chunks[0].to({torch::kCUDA, 0}).allclose(a));  // 断言 chunks[0] 在 CUDA 设备 0 上的拷贝与 a 的近似相等
  ASSERT_TRUE(chunks[1].allclose(b));  // 断言 chunks[1] 与 b 的近似相等

  output.backward(torch::ones_like(output));  // 对 output 进行反向传播

  ASSERT_TRUE(a.grad().defined());  // 断言 a 的梯度已经定义
  ASSERT_EQ(a.grad().device(), torch::Device(torch::kCUDA, 0));  // 断言 a 的梯度设备为 CUDA 设备 0
  ASSERT_EQ(a.grad().sum().item<int32_t>(), 5);  // 断言 a 的梯度总和为 5

  ASSERT_TRUE(b.grad().defined());  // 断言 b 的梯度已经定义
  ASSERT_EQ(b.grad().device(), torch::Device(torch::kCUDA, 1));  // 断言 b 的梯度设备为 CUDA 设备 1
  ASSERT_EQ(b.grad().sum().item<int32_t>(), 5);  // 断言 b 的梯度总和为 5
}

TEST_F(ParallelTest, Replicate_MultiCUDA) {  // 定义测试用例 Replicate_MultiCUDA
  Linear linear(3, 4);  // 创建线性层对象 linear，输入大小为 3，输出大小为 4
  auto replicas = parallel::replicate(  // 复制线性层对象 linear 到多个 CUDA 设备上
      linear, {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});
  ASSERT_EQ(replicas.size(), 2);  // 断言 replicas 的大小为 2

  auto original_parameters = linear->parameters();  // 获取原始线性层的参数

  auto replica1_parameters = replicas[0]->parameters();  // 获取 replicas 中第一个副本的参数
  for (auto& parameter : replica1_parameters) {  // 遍历第一个副本的参数
    ASSERT_EQ(parameter.device(), torch::Device(torch::kCUDA, 0));  // 断言参数的设备为 CUDA 设备 0
  }
  replicas[0]->to(torch::kCPU);  // 将 replicas 中第一个副本移动到 CPU 设备

  ASSERT_EQ(replica1_parameters.size(), original_parameters.size());  // 断言第一个副本的参数数量与原始参数数量相同
  for (const auto i : c10::irange(original_parameters.size())) {  // 遍历原始参数的数量范围
    ASSERT_TRUE(replica1_parameters[i].allclose(original_parameters[i]));  // 断言第一个副本的每个参数与原始参数的近似相等
    // 对比第一个副本的参数是否与原始参数的内存地址不同
    ASSERT_TRUE(
        replica1_parameters[i].data_ptr<float>() !=
        original_parameters[i].data_ptr<float>());
  }

  // 获取第二个副本的参数列表
  auto replica2_parameters = replicas[1]->parameters();
  // 确保第二个副本的所有参数位于 CUDA 设备 1 上
  for (auto& parameter : replica2_parameters) {
    ASSERT_EQ(parameter.device(), torch::Device(torch::kCUDA, 1));
  }
  // 将第二个副本的参数转移到 CPU 上
  replicas[1]->to(torch::kCPU);
  // 确保第二个副本的参数数量与原始参数的数量相同
  ASSERT_EQ(replica2_parameters.size(), original_parameters.size());
  // 对比第二个副本的每个参数是否与原始参数的值近似相等
  for (const auto i : c10::irange(original_parameters.size())) {
    ASSERT_TRUE(replica2_parameters[i].allclose(original_parameters[i]));
    // 确保第二个副本的每个参数的内存地址与原始参数的内存地址不同
    ASSERT_TRUE(
        replica2_parameters[i].data_ptr<float>() !=
        original_parameters[i].data_ptr<float>());
  }
}

# 在测试类 ParallelTest 中定义一个测试用例 ParallelApply_MultiCUDA
TEST_F(ParallelTest, ParallelApply_MultiCUDA) {
  
  # 创建一个具有输入大小 3x4 的线性层对象 a
  Linear a(3, 4);

  # 克隆线性层 a 并转移到 CUDA 设备 0 上，得到线性层 b
  Linear b(std::dynamic_pointer_cast<LinearImpl>(a->clone()));
  b->to({torch::kCUDA, 0});

  # 克隆线性层 a 并转移到 CUDA 设备 1 上，得到线性层 c
  Linear c(std::dynamic_pointer_cast<LinearImpl>(a->clone()));
  c->to({torch::kCUDA, 1});

  # 创建线性层对象的向量 modules 包括 a, b, c
  std::vector<Linear> modules = {a, b, c};

  # 创建输入张量的向量 inputs，分别是在 CPU 和两个不同 CUDA 设备上的张量
  std::vector<torch::Tensor> inputs = {
      torch::ones({2, 3}),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 0})),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 1}))
  };

  # 并行应用 modules 到 inputs 上，得到 outputs
  auto outputs = parallel::parallel_apply(modules, inputs);

  # 断言输出的数量为 3
  ASSERT_EQ(outputs.size(), 3);
  
  # 断言输出的第一个张量在 CPU 上
  ASSERT_TRUE(outputs[0].device().is_cpu());

  # 断言输出的第二个张量在 CUDA 设备 0 上，并且与第一个输出张量相似
  ASSERT_EQ(outputs[1].device(), torch::Device(torch::kCUDA, 0));
  ASSERT_TRUE(outputs[1].to(torch::kCPU).allclose(outputs[0]));

  # 断言输出的第三个张量在 CUDA 设备 1 上，并且与第一个输出张量相似
  ASSERT_EQ(outputs[2].device(), torch::Device(torch::kCUDA, 1));
  ASSERT_TRUE(outputs[2].to(torch::kCPU).allclose(outputs[0]));
}

# 定义测试用例 ParallelApplyWithDifferentOutputDevice_MultiCUDA
TEST_F(ParallelTest, ParallelApplyWithDifferentOutputDevice_MultiCUDA) {
  
  # 定义一个简单的模块 M，用于返回形状为 (5,) 的整数张量
  struct M : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return torch::ones(5, torch::kInt32);
    }
  };

  # 创建三个共享指针指向 M 的模块
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(), std::make_shared<M>(), std::make_shared<M>()};

  # 创建空张量的向量作为输入
  std::vector<torch::Tensor> inputs = {
      torch::empty({}), torch::empty({}), torch::empty({})};

  # 创建设备的向量，分别是 CUDA 设备 1、CUDA 设备 0 和 CPU
  std::vector<torch::Device> devices = {
      {torch::kCUDA, 1}, {torch::kCUDA, 0}, {torch::kCPU}};

  # 并行应用 modules 到 inputs 上，并指定每个模块的设备，得到 outputs
  auto outputs = parallel::parallel_apply(modules, inputs, devices);

  # 断言输出的数量为 3
  ASSERT_EQ(outputs.size(), 3);

  # 断言输出的第一个张量在 CUDA 设备 1 上
  ASSERT_TRUE(outputs[0].device().is_cuda());
  ASSERT_EQ(outputs[0].device(), torch::Device(torch::kCUDA, 1));

  # 断言输出的第二个张量在 CUDA 设备 0 上
  ASSERT_TRUE(outputs[1].device().is_cuda());
  ASSERT_EQ(outputs[1].device(), torch::Device(torch::kCUDA, 0));

  # 断言输出的第三个张量在 CPU 上
  ASSERT_TRUE(outputs[2].device().is_cpu());
}

# 定义测试用例 ParallelApplyRethrowsException_MultiCUDA
TEST_F(ParallelTest, ParallelApplyRethrowsException_MultiCUDA) {
  
  # 定义一个简单的模块 M，重写 reset 方法并在 forward 方法中抛出异常
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      throw std::runtime_error("Badness!");
    }
  };

  # 创建共享指针指向 M 的模块 m
  auto m = std::make_shared<M>();

  # 创建形状为 (10, 3) 的全一张量作为输入
  auto input = torch::ones({10, 3});

  # 断言调用 parallel::data_parallel 会抛出指定异常 "Badness!"
  ASSERT_THROWS_WITH(parallel::data_parallel(m, input), "Badness!");
}

# 定义测试用例 DataParallelPlacesTheOutputOnTheRequestedDevice_MultiCUDA
TEST_F(
    ParallelTest,
    DataParallelPlacesTheOutputOnTheRequestedDevice_MultiCUDA) {
  
  # 定义一个简单的模块 M，重写 reset 方法并在 forward 方法中返回全一张量
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      // 返回的张量应位于指定的输出设备上
      return torch::ones(3);
    }
  };

  # 创建共享指针指向 M 的模块 m
  auto m = std::make_shared<M>();

  # 创建形状为 (10, 3) 的全一张量作为输入
  auto input = torch::ones({10, 3});

  {
    # 调用 parallel::data_parallel 函数，指定输出设备为 CUDA 设备 1
    auto output = parallel::data_parallel(
        m,
        input,
        /*devices=*/torch::nullopt,
        /*output_device=*/torch::Device(torch::kCUDA, 1));

    # 断言输出已定义，并位于 CUDA 设备 1 上
    ASSERT_TRUE(output.defined());
    ASSERT_TRUE(output.device().is_cuda());
    ASSERT_EQ(output.device().index(), 1);
  }

  {
    // 验证单设备情况下的输出设备（无需散射/聚集操作）  
    # 使用并行计算来执行数据并行操作
    auto output = parallel::data_parallel(
        m,
        input,
        /*devices=*/std::vector<torch::Device>{torch::Device(torch::kCUDA, 0)},  # 指定使用的CUDA设备列表，这里选择索引为0的设备
        /*output_device=*/torch::Device(torch::kCUDA, 1));  # 指定输出结果存储的CUDA设备为索引为1的设备
    # 确保输出张量已定义
    ASSERT_TRUE(output.defined());
    # 确保输出张量存储在CUDA设备上
    ASSERT_TRUE(output.device().is_cuda());
    # 确保输出张量存储在索引为1的CUDA设备上
    ASSERT_EQ(output.device().index(), 1);
}

// 定义一个测试案例，验证数据并行处理是否使用了所有可用的 CUDA 设备
TEST_F(ParallelTest, DataParallelUsesAllAvailableCUDADevices_CUDA) {
  // 定义一个简单的神经网络模型 M，继承自 Cloneable<M>
  struct M : torch::nn::Cloneable<M> {
    // 重置模型状态
    void reset() override {}
    
    // 前向传播函数，返回输入张量所在设备的索引
    torch::Tensor forward(torch::Tensor input) {
      return torch::tensor({input.device().index()});
    }
  };

  // 创建模型 M 的共享指针
  auto m = std::make_shared<M>();
  // 获取 CUDA 设备数量
  const auto device_count = torch::cuda::device_count();
  // 创建输入张量，确保至少有 10 个元素或者 2 倍于 CUDA 设备数量的元素个数，每个元素值为 1
  auto input = torch::ones({std::max(10, int(2 * device_count)), 3});
  // 使用 data_parallel 函数对模型 m 进行并行处理，输入为 input
  auto output = parallel::data_parallel(m, input);

  // 断言输出张量的元素个数与 CUDA 设备数量相等
  ASSERT_EQ(output.numel(), device_count);
  // 遍历输出张量的每个元素，验证其值与索引是否相等
  for (const auto i : c10::irange(device_count)) {
    ASSERT_EQ(output[i].item<int32_t>(), i);
  }
}

// 定义一个测试案例，验证多个 CUDA 设备上数据并行处理的数值等效性
TEST_F(ParallelTest, DataParallelNumericalEquivalence_MultiCUDA) {
  // 定义一个复杂的神经网络模型 M，继承自 Cloneable<M>
  struct M : torch::nn::Cloneable<M> {
    M() {
      reset();
    }

    // 重置模型，包含卷积层和全连接层的初始化
    void reset() override {
      conv = register_module(
          "conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 2, /*kernel_size=*/2)));
      fc = register_module("fc", torch::nn::Linear(8, 2));
    }

    // 前向传播函数，依次执行卷积、ReLU激活、展平、全连接和对数softmax操作
    torch::Tensor forward(torch::Tensor x) {
      x = conv->forward(x);
      x = torch::relu(x);
      x = x.view({-1, 8});
      x = fc->forward(x);
      return torch::log_softmax(x, /*dim=*/1);
    }

    torch::nn::Conv2d conv{nullptr};  // 卷积层指针
    torch::nn::Linear fc{nullptr};    // 全连接层指针
  };

  // 准备输入张量，每个元素为 1，形状为 [16, 2, 3, 3]
  auto input = torch::ones({16, 2, 3, 3});
  auto input_dp = torch::ones({16, 2, 3, 3});
  // 创建模型 M 的共享指针和数据并行模型的动态指针
  auto model = std::make_shared<M>();
  auto model_dp = std::dynamic_pointer_cast<M>(model->clone());

  // 运行 3 次训练迭代
  for (const auto i : c10::irange(3)) {
    input += i;
    input_dp += i;

    // 非并行训练
    torch::optim::SGD optim(model->parameters(), torch::optim::SGDOptions(0.1));
    auto output = model->forward(input);
    auto loss = torch::mse_loss(output, torch::zeros_like(output));
    loss.backward();
    optim.step();

    // 数据并行训练
    torch::optim::SGD optim_dp(
        model_dp->parameters(), torch::optim::SGDOptions(0.1));
    auto output_dp = parallel::data_parallel(model_dp, input_dp);
    auto loss_dp = torch::mse_loss(output_dp, torch::zeros_like(output_dp));
    loss_dp.backward();
    optim_dp.step();

    // 确保模型参数在 CPU 上的权重一致性
    model->to(torch::kCPU);
    model_dp->to(torch::kCPU);
    auto params = model->parameters();
    auto params_dp = model_dp->parameters();
    ASSERT_EQ(params.size(), params_dp.size());
    // 遍历并断言每个参数是否接近
    for (auto it = params.begin(), it_dp = params_dp.begin();
         it != params.end() && it_dp != params.end();
         ++it, ++it_dp) {
      ASSERT_TRUE(torch::allclose(*it, *it_dp));
    }
  }
}
```