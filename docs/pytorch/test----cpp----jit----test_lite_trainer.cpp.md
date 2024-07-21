# `.\pytorch\test\cpp\jit\test_lite_trainer.cpp`

```py
// 包含用于测试的头文件
#include <test/cpp/jit/test_utils.h>

// 包含 Google 测试框架的头文件
#include <gtest/gtest.h>

// 包含 PyTorch 核心 TensorOptions 相关的头文件
#include <c10/core/TensorOptions.h>

// 包含自动求导生成的变量工厂的头文件
#include <torch/csrc/autograd/generated/variable_factories.h>

// 包含 PyTorch JIT 模块相关的头文件
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/import_data.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/train/export_data.h>
#include <torch/csrc/jit/mobile/train/optim/sgd.h>
#include <torch/csrc/jit/mobile/train/random.h>
#include <torch/csrc/jit/mobile/train/sequential.h>
#include <torch/csrc/jit/serialization/import.h>

// 包含 PyTorch 数据加载器相关的头文件
#include <torch/data/dataloader.h>

// 包含 PyTorch 主头文件
#include <torch/torch.h>

// 定义测试在 torch::jit 命名空间中进行
namespace torch {
namespace jit {

// 定义 LiteTrainerTest 的单元测试用例 Params
TEST(LiteTrainerTest, Params) {
  // 创建名为 "m" 的 Module 对象
  Module m("m");
  
  // 注册一个名为 "foo" 的参数，初始化为全为1的张量，不需要梯度
  m.register_parameter("foo", torch::ones({1}, at::requires_grad()), false);
  
  // 定义 Module 的前向传播方法
  m.define(R"(
    def forward(self, x):
      b = 1.0
      return self.foo * x + b
  )");
  
  // 设置学习率和动量
  double learning_rate = 0.1, momentum = 0.1;
  // 设置迭代次数
  int n_epoc = 10;
  
  // 初始化训练数据：y = x + 1;
  // 目标数据：y = 2 x + 1
  std::vector<std::pair<Tensor, Tensor>> trainData{
      {1 * torch::ones({1}), 3 * torch::ones({1})},
  };
  
  // 创建一个 stringstream 用于保存模型
  std::stringstream ms;
  m.save(ms);
  
  // 从 stringstream 中加载模型
  auto mm = load(ms);
  
  // 获取模型的所有参数
  std::vector<::at::Tensor> parameters;
  for (auto parameter : mm.parameters()) {
    parameters.emplace_back(parameter);
  }
  
  // 使用 SGD 优化器进行模型参数的优化
  ::torch::optim::SGD optimizer(
      parameters, ::torch::optim::SGDOptions(learning_rate).momentum(momentum));
  
  // 开始训练过程
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    for (auto& data : trainData) {
      auto source = data.first, targets = data.second;
      
      // 梯度清零
      optimizer.zero_grad();
      
      // 准备训练输入数据
      std::vector<IValue> train_inputs{source};
      
      // 前向传播得到模型输出
      auto output = mm.forward(train_inputs).toTensor();
      
      // 计算损失
      auto loss = ::torch::l1_loss(output, targets);
      
      // 反向传播计算梯度
      loss.backward();
      
      // 使用优化器更新参数
      optimizer.step();
    }
  }
  
  // 创建一个 stringstream 用于保存移动端模型
  std::stringstream ss;
  m._save_for_mobile(ss);
  
  // 从 stringstream 中加载移动端模型
  mobile::Module bc = _load_for_mobile(ss);
  
  // 获取移动端模型的所有参数
  std::vector<::at::Tensor> bc_parameters = bc.parameters();
  
  // 使用 SGD 优化器进行移动端模型参数的优化
  ::torch::optim::SGD bc_optimizer(
      bc_parameters,
      ::torch::optim::SGDOptions(learning_rate).momentum(momentum));
  
  // 开始训练移动端模型的过程
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    for (auto& data : trainData) {
      auto source = data.first, targets = data.second;
      
      // 梯度清零
      bc_optimizer.zero_grad();
      
      // 准备训练输入数据
      std::vector<IValue> train_inputs{source};
      
      // 前向传播得到移动端模型输出
      auto output = bc.forward(train_inputs).toTensor();
      
      // 计算损失
      auto loss = ::torch::l1_loss(output, targets);
      
      // 反向传播计算梯度
      loss.backward();
      
      // 使用优化器更新移动端模型参数
      bc_optimizer.step();
    }
  }
  
  // 断言参数的数值是否相等
  AT_ASSERT(parameters[0].item<float>() == bc_parameters[0].item<float>());
}

// TODO 在正确加载移动端模型参数后，重新启用这些测试
/*
TEST(MobileTest, NamedParameters) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    # 定义一个方法 `add_it`，接受参数 `self` 和 `x`
    def add_it(self, x):
      # 在方法中定义变量 `b`，并赋值为 4
      b = 4
      # 返回 `self.foo`、`x` 和 `b` 三者的和
      return self.foo + x + b
  )");

  # 创建名为 `child` 的 Module 对象，命名空间为 "m2"
  Module child("m2");
  # 在 `child` 模块中注册一个名为 "foo" 的参数，值为 4 的 Torch 张量，不允许梯度更新
  child.register_parameter("foo", 4 * torch::ones({}), false);
  # 在 `child` 模块中注册一个名为 "bar" 的参数，值为 4 的 Torch 张量，不允许梯度更新
  child.register_parameter("bar", 4 * torch::ones({}), false);

  # 在主 Module `m` 中注册名为 "child1" 的子模块 `child`
  m.register_module("child1", child);
  # 克隆 `child` 模块，并在主 Module `m` 中注册名为 "child2" 的克隆子模块
  m.register_module("child2", child.clone());

  # 创建一个字符串流对象 `ss`
  std::stringstream ss;
  # 将主 Module `m` 保存为移动端可用的格式，存入字符串流 `ss`
  m._save_for_mobile(ss);
  # 从字符串流 `ss` 中加载移动端可用的 Module，并存入 `bc` 变量
  mobile::Module bc = _load_for_mobile(ss);

  # 获取主 Module `m` 的所有命名参数，并存入 `full_params`
  auto full_params = m.named_parameters();
  # 获取移动端 Module `bc` 的所有命名参数，并存入 `mobile_params`
  auto mobile_params = bc.named_parameters();
  # 断言主 Module `m` 和移动端 Module `bc` 的参数数量相等
  AT_ASSERT(full_params.size() == mobile_params.size());
  # 遍历主 Module `m` 的所有参数，对比其数值是否与移动端 Module `bc` 对应参数的数值相等
  for (const auto& e : full_params) {
    AT_ASSERT(e.value.item().toInt() ==
    mobile_params[e.name].item().toInt());
  }
}

// 测试保存和加载模块参数
TEST(MobileTest, SaveLoadParameters) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 向模块注册一个名为 "foo" 的参数，初始值为全 1 的张量
  m.register_parameter("foo", torch::ones({}), false);
  // 定义一个简单的方法 "add_it"，接受参数 x，返回 self.foo + x + b 的结果
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 创建一个名为 "m2" 的子模块
  Module child("m2");
  // 向子模块注册一个名为 "foo" 的参数，初始值为全 4 倍的张量
  child.register_parameter("foo", 4 * torch::ones({}), false);
  // 向子模块注册一个名为 "bar" 的参数，初始值为全 3 倍的张量
  child.register_parameter("bar", 3 * torch::ones({}), false);

  // 向模块 "m" 注册两个名为 "child1" 和 "child2" 的子模块
  m.register_module("child1", child);
  m.register_module("child2", child.clone());

  // 获取模块 "m" 的所有参数，并保存到 full_params
  auto full_params = m.named_parameters();

  // 创建两个字符串流对象 ss 和 ss_data
  std::stringstream ss;
  std::stringstream ss_data;

  // 将模块 "m" 保存为移动端模块到字符串流 ss
  m._save_for_mobile(ss);

  // 加载移动端模块 bc，并将其命名参数保存到字符串流 ss_data
  mobile::Module bc = _load_for_mobile(ss);
  _save_parameters(bc.named_parameters(), ss_data);

  // 加载字符串流 ss_data 中的参数，并与 full_params 进行比较
  auto mobile_params = _load_parameters(ss_data);
  AT_ASSERT(full_params.size() == mobile_params.size());
  for (const auto& e : full_params) {
    AT_ASSERT(e.value.item<int>() == mobile_params[e.name].item<int>());
  }
}

// 测试保存和加载空参数集合
TEST(MobileTest, SaveLoadParametersEmpty) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 定义一个简单的方法 "add_it"，接受参数 x，返回 x + b 的结果
  m.define(R"(
    def add_it(self, x):
      b = 4
      return x + b
  )");

  // 创建一个名为 "m2" 的子模块
  Module child("m2");
  // 向模块 "m" 注册两个名为 "child1" 和 "child2" 的子模块
  m.register_module("child1", child);
  m.register_module("child2", child.clone());

  // 创建两个字符串流对象 ss 和 ss_data
  std::stringstream ss;
  std::stringstream ss_data;

  // 将模块 "m" 保存为移动端模块到字符串流 ss
  m._save_for_mobile(ss);

  // 加载移动端模块 bc，并将其命名参数保存到字符串流 ss_data
  mobile::Module bc = _load_for_mobile(ss);
  _save_parameters(bc.named_parameters(), ss_data);

  // 加载字符串流 ss_data 中的参数，并验证其是否为空
  auto mobile_params = _load_parameters(ss_data);
  AT_ASSERT(mobile_params.size() == 0);
}

// 测试使用 ZIP 容器保存参数
TEST(MobileTest, SaveParametersDefaultsToZip) {
  // 创建一个空的参数映射 empty_parameters
  std::map<std::string, at::Tensor> empty_parameters;
  // 创建一个字符串流对象 ss_data
  std::stringstream ss_data;

  // 将空的参数映射 empty_parameters 保存到字符串流 ss_data
  _save_parameters(empty_parameters, ss_data);

  // 验证参数是否被序列化到 ZIP 容器中
  EXPECT_GE(ss_data.str().size(), 4);
  EXPECT_EQ(ss_data.str()[0], 'P');
  EXPECT_EQ(ss_data.str()[1], 'K');
  EXPECT_EQ(ss_data.str()[2], '\x03');
  EXPECT_EQ(ss_data.str()[3], '\x04');
}

// 测试使用 Flatbuffer 保存参数
TEST(MobileTest, SaveParametersCanUseFlatbuffer) {
  // 创建一个空的参数映射 empty_parameters
  std::map<std::string, at::Tensor> empty_parameters;
  // 创建一个字符串流对象 ss_data
  std::stringstream ss_data;

  // 使用 Flatbuffer 将空的参数映射 empty_parameters 保存到字符串流 ss_data
  _save_parameters(empty_parameters, ss_data, /*use_flatbuffer=*/true);

  // 验证参数是否被序列化到 Flatbuffer 中，Flatbuffer 的魔数应在偏移量 4..7 的位置
  EXPECT_GE(ss_data.str().size(), 8);
  EXPECT_EQ(ss_data.str()[4], 'P');
  EXPECT_EQ(ss_data.str()[5], 'T');
  EXPECT_EQ(ss_data.str()[6], 'M');
  EXPECT_EQ(ss_data.str()[7], 'F');
}
TEST(MobileTest, SaveLoadParametersUsingFlatbuffers) {
  // 创建一些简单的参数以便保存
  std::map<std::string, at::Tensor> input_params;
  input_params["four_by_ones"] = 4 * torch::ones({});
  input_params["three_by_ones"] = 3 * torch::ones({});

  // 使用 flatbuffers 序列化这些参数
  std::stringstream data;
  _save_parameters(input_params, data, /*use_flatbuffer=*/true);

  // Flatbuffer 的魔术字节应该在偏移量 4 到 7 处
  EXPECT_EQ(data.str()[4], 'P');
  EXPECT_EQ(data.str()[5], 'T');
  EXPECT_EQ(data.str()[6], 'M');
  EXPECT_EQ(data.str()[7], 'F');

  // 读取数据并验证其是否正确恢复
  auto output_params = _load_parameters(data);
  EXPECT_EQ(output_params.size(), 2);
  {
    auto four_by_ones = 4 * torch::ones({});
    EXPECT_EQ(
        output_params["four_by_ones"].item<int>(), four_by_ones.item<int>());
  }
  {
    auto three_by_ones = 3 * torch::ones({});
    EXPECT_EQ(
        output_params["three_by_ones"].item<int>(), three_by_ones.item<int>());
  }
}

TEST(MobileTest, LoadParametersUnexpectedFormatShouldThrow) {
  // 手动创建一些看起来不像 ZIP 或 Flatbuffer 文件的数据
  // 确保其长度超过 8 字节，因为 getFileFormat() 需要这么多数据来检测类型
  std::stringstream bad_data;
  bad_data << "abcd"
           << "efgh"
           << "ijkl";

  // 从中加载参数应该会抛出异常
  EXPECT_ANY_THROW(_load_parameters(bad_data));
}

TEST(MobileTest, LoadParametersEmptyDataShouldThrow) {
  // 从空数据流加载参数应该会抛出异常
  std::stringstream empty;
  EXPECT_ANY_THROW(_load_parameters(empty));
}

TEST(MobileTest, LoadParametersMalformedFlatbuffer) {
  // 手动创建一个带有 Flatbuffer 头的数据
  std::stringstream bad_data;
  bad_data << "PK\x03\x04PTMF\x00\x00"
           << "*}NV\xb3\xfa\xdf\x00pa";

  // 从中加载参数应该会抛出异常，并且异常消息应该包含 "Malformed Flatbuffer module"
  ASSERT_THROWS_WITH_MESSAGE(
      _load_parameters(bad_data), "Malformed Flatbuffer module");
}

TEST(LiteTrainerTest, SGD) {
  Module m("m");
  m.register_parameter("foo", torch::ones({1}, at::requires_grad()), false);
  m.define(R"(
    def forward(self, x):
      b = 1.0
      return self.foo * x + b
  )");
  double learning_rate = 0.1, momentum = 0.1;
  int n_epoc = 10;
  // 初始化: y = x + 1;
  // 目标: y = 2 x + 1
  std::vector<std::pair<Tensor, Tensor>> trainData{
      {1 * torch::ones({1}), 3 * torch::ones({1})},
  };
  // 参考：完整的 JIT 和 torch::optim::SGD
  std::stringstream ms;
  m.save(ms);
  auto mm = load(ms);
  std::vector<::at::Tensor> parameters;
  // 收集模型参数
  for (auto parameter : mm.parameters()) {
    parameters.emplace_back(parameter);
  }
  // 设置优化器为 SGD
  ::torch::optim::SGD optimizer(
      parameters, ::torch::optim::SGDOptions(learning_rate).momentum(momentum));
  // 训练循环
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    // 对训练数据集中的每个数据进行迭代
    for (auto& data : trainData) {
      // 解构每个数据元组，source 为输入，targets 为目标
      auto source = data.first, targets = data.second;
      // 清零优化器的梯度
      optimizer.zero_grad();
      // 将输入数据封装成 IValue 向量
      std::vector<IValue> train_inputs{source};
      // 使用模型 mm 进行前向传播得到输出
      auto output = mm.forward(train_inputs).toTensor();
      // 计算输出和目标之间的 L1 损失
      auto loss = ::torch::l1_loss(output, targets);
      // 反向传播计算梯度
      loss.backward();
      // 执行优化步骤
      optimizer.step();
    }
  }
  // 测试：使用轻量级解释器和 torch::jit::mobile::SGD 进行操作
  // 创建一个字符串流对象 ss
  std::stringstream ss;
  // 将模型 m 保存为移动端可用格式并写入 ss
  m._save_for_mobile(ss);
  // 从 ss 中加载为移动端模块 bc
  mobile::Module bc = _load_for_mobile(ss);
  // 获取 bc 模块的参数
  std::vector<::at::Tensor> bc_parameters = bc.parameters();
  // 使用移动端 SGD 进行优化，设置学习率和动量
  ::torch::jit::mobile::SGD bc_optimizer(
      bc_parameters,
      ::torch::jit::mobile::SGDOptions(learning_rate).momentum(momentum));
  // 对每个 epoch 进行迭代
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    // 对训练数据集中的每个数据进行迭代
    for (auto& data : trainData) {
      // 解构每个数据元组，source 为输入，targets 为目标
      auto source = data.first, targets = data.second;
      // 清零优化器的梯度
      bc_optimizer.zero_grad();
      // 将输入数据封装成 IValue 向量
      std::vector<IValue> train_inputs{source};
      // 使用 bc 模块进行前向传播得到输出
      auto output = bc.forward(train_inputs).toTensor();
      // 计算输出和目标之间的 L1 损失
      auto loss = ::torch::l1_loss(output, targets);
      // 反向传播计算梯度
      loss.backward();
      // 执行优化步骤
      bc_optimizer.step();
    }
  }
  // 断言：验证两个模型的第一个参数的浮点数值是否相等
  AT_ASSERT(parameters[0].item<float>() == bc_parameters[0].item<float>());
TEST(LiteTrainerTest, SequentialSampler) {
  // 测试顺序采样器能否与数据加载器一起使用

  // 定义批大小为10
  const int kBatchSize = 10;
  // 创建数据加载器，使用顺序采样器，加载 DummyDataset(25) 数据集，批大小为 kBatchSize
  auto data_loader = torch::data::make_data_loader<mobile::SequentialSampler>(
      DummyDataset(25), kBatchSize);
  // 初始化计数器 i 为 1
  int i = 1;
  // 遍历数据加载器中的每个批次 batch
  for (const auto& batch : *data_loader) {
    // 遍历每个批次中的每个示例 example
    for (const auto& example : batch) {
      // 断言当前示例 example 的值等于 i
      AT_ASSERT(i == example);
      // 计数器 i 自增
      i++;
    }
  }
}

TEST(LiteTrainerTest, RandomSamplerReturnsIndicesInCorrectRange) {
  // 创建具有指定大小的随机采样器
  mobile::RandomSampler sampler(10);

  // 获取下一个批次大小为 3 的样本索引
  std::vector<size_t> indices = sampler.next(3).value();
  // 遍历索引，断言每个索引小于 10
  for (auto i : indices) {
    AT_ASSERT(i < 10);
  }

  // 获取下一个批次大小为 5 的样本索引
  indices = sampler.next(5).value();
  // 遍历索引，断言每个索引小于 10
  for (auto i : indices) {
    AT_ASSERT(i < 10);
  }

  // 获取下一个批次大小为 2 的样本索引
  indices = sampler.next(2).value();
  // 遍历索引，断言每个索引小于 10
  for (auto i : indices) {
    AT_ASSERT(i < 10);
  }

  // 断言采样器不能获取大小为 10 的下一个批次
  AT_ASSERT(sampler.next(10).has_value() == false);
}

TEST(LiteTrainerTest, RandomSamplerReturnsLessValuesForLastBatch) {
  // 创建具有指定大小的随机采样器
  mobile::RandomSampler sampler(5);
  // 断言下一个批次大小为 3 的值数量为 3
  AT_ASSERT(sampler.next(3).value().size() == 3);
  // 断言下一个批次大小为 100 的值数量为 2
  AT_ASSERT(sampler.next(100).value().size() == 2);
  // 断言采样器不能获取大小为 2 的下一个批次
  AT_ASSERT(sampler.next(2).has_value() == false);
}

TEST(LiteTrainerTest, RandomSamplerResetsWell) {
  // 创建具有指定大小的随机采样器
  mobile::RandomSampler sampler(5);
  // 断言下一个批次大小为 5 的值数量为 5
  AT_ASSERT(sampler.next(5).value().size() == 5);
  // 断言采样器不能获取大小为 2 的下一个批次
  AT_ASSERT(sampler.next(2).has_value() == false);
  // 重置采样器状态
  sampler.reset();
  // 再次断言下一个批次大小为 5 的值数量为 5
  AT_ASSERT(sampler.next(5).value().size() == 5);
  // 断言采样器不能获取大小为 2 的下一个批次
  AT_ASSERT(sampler.next(2).has_value() == false);
}

TEST(LiteTrainerTest, RandomSamplerResetsWithNewSizeWell) {
  // 创建具有指定大小的随机采样器
  mobile::RandomSampler sampler(5);
  // 断言下一个批次大小为 5 的值数量为 5
  AT_ASSERT(sampler.next(5).value().size() == 5);
  // 断言采样器不能获取大小为 2 的下一个批次
  AT_ASSERT(sampler.next(2).has_value() == false);
  // 使用新的大小重置采样器
  sampler.reset(7);
  // 断言下一个批次大小为 7 的值数量为 7
  AT_ASSERT(sampler.next(7).value().size() == 7);
  // 断言采样器不能获取大小为 2 的下一个批次
  AT_ASSERT(sampler.next(2).has_value() == false);
  // 再次使用新的大小重置采样器
  sampler.reset(3);
  // 断言下一个批次大小为 3 的值数量为 3
  AT_ASSERT(sampler.next(3).value().size() == 3);
  // 断言采样器不能获取大小为 2 的下一个批次
  AT_ASSERT(sampler.next(2).has_value() == false);
}
```