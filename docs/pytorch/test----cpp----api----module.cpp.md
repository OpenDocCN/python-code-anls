# `.\pytorch\test\cpp\api\module.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架头文件

#include <c10/util/irange.h>  // 引入 C10 库中的 irange 头文件
#include <torch/torch.h>  // 引入 PyTorch C++ API 的头文件

#include <test/cpp/api/support.h>  // 引入测试支持函数的头文件

using namespace torch::nn;  // 使用 torch::nn 命名空间
using namespace torch::test;  // 使用 torch::test 命名空间

struct AGIUnit : torch::nn::Module {};  // 定义一个名为 AGIUnit 的 PyTorch 模块

namespace test {
struct AGIUnit : torch::nn::Module {};  // 定义 test 命名空间下的 AGIUnit 模块
struct AGIUnit2 : torch::nn::Module {  // 定义 test 命名空间下的 AGIUnit2 模块，构造函数设置模块名为 "Foo"
  AGIUnit2() : torch::nn::Module("Foo") {}
};
} // namespace test

struct ModuleTest : torch::test::SeedingFixture {};  // 定义一个测试 fixture 类 ModuleTest，继承自 SeedingFixture

TEST_F(ModuleTest, CanEnableAndDisableTrainingMode) {  // 定义测试用例 CanEnableAndDisableTrainingMode
  Linear module(3, 4);  // 创建一个线性层模块，输入维度为3，输出维度为4
  ASSERT_TRUE(module->is_training());  // 断言模块当前处于训练模式

  module->eval();  // 将模块切换到评估模式
  ASSERT_FALSE(module->is_training());  // 断言模块当前处于评估模式

  module->train();  // 将模块切换回训练模式
  ASSERT_TRUE(module->is_training());  // 断言模块当前处于训练模式
}

TEST_F(ModuleTest, ZeroGrad) {  // 定义测试用例 ZeroGrad
  Linear module(3, 4);  // 创建一个线性层模块，输入维度为3，输出维度为4
  auto weight = torch::ones({8, 3}, torch::requires_grad());  // 创建一个需要梯度的张量 weight，维度为 [8, 3]，初始值为全1
  auto loss = module(weight).sum();  // 对权重 weight 执行前向传播并计算输出的和作为损失值
  loss.backward();  // 对损失值进行反向传播

  for (auto& parameter : module->parameters()) {  // 遍历模块的所有参数
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto grad = parameter.grad();  // 获取参数的梯度
    ASSERT_TRUE(grad.defined());  // 断言梯度已定义
    ASSERT_NE(grad.sum().item<float>(), 0);  // 断言梯度的和不为0
  }

  module->zero_grad();  // 将模块的所有参数梯度清零

  for (auto& parameter : module->parameters()) {  // 再次遍历模块的所有参数
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto grad = parameter.grad();  // 获取参数的梯度
    ASSERT_FALSE(grad.defined());  // 断言梯度未定义
  }
}

TEST_F(ModuleTest, ZeroGradWithUndefined) {  // 定义测试用例 ZeroGradWithUndefined
  struct TestModule : torch::nn::Module {  // 定义一个名为 TestModule 的 PyTorch 模块
    TestModule() {  // 构造函数
      x = register_parameter("x", torch::ones(5, torch::requires_grad()));  // 注册名为 "x" 的参数，维度为 [5]，初始值为全1且需要梯度
      y = register_parameter("y", torch::ones(5, torch::requires_grad()));  // 注册名为 "y" 的参数，维度为 [5]，初始值为全1且需要梯度
    }
    torch::Tensor x, y;  // 定义张量 x 和 y
  };

  TestModule module;  // 创建 TestModule 的实例 module
  auto z = module.x * 2;  // 计算张量 x 的两倍
  z.sum().backward();  // 对 z 的和进行反向传播

  ASSERT_TRUE(module.x.grad().defined());  // 断言张量 x 的梯度已定义
  ASSERT_FALSE(module.y.grad().defined());  // 断言张量 y 的梯度未定义

  module.zero_grad(false);  // 调用模块的 zero_grad 函数，设置参数 set_to_none 为 false

  ASSERT_TRUE(module.x.grad().defined());  // 断言张量 x 的梯度已定义
  ASSERT_FALSE(module.y.grad().defined());  // 断言张量 y 的梯度未定义

  ASSERT_EQ(module.x.grad().sum().item<float>(), 0);  // 断言张量 x 的梯度和为0

  module.zero_grad();  // 调用模块的 zero_grad 函数，不传递参数，将所有参数的梯度清零

  ASSERT_FALSE(module.x.grad().defined());  // 断言张量 x 的梯度未定义
  ASSERT_FALSE(module.y.grad().defined());  // 断言张量 y 的梯度未定义
}

TEST_F(ModuleTest, RegisterModuleThrowsForEmptyOrDottedName) {  // 定义测试用例 RegisterModuleThrowsForEmptyOrDottedName
  struct TestModel : public torch::nn::Module {};  // 定义一个名为 TestModel 的 PyTorch 模块

  ASSERT_THROWS_WITH(  // 断言以下代码块抛出异常，并验证异常消息
      TestModel{}.register_module("name.with.dot", torch::nn::Linear(3, 4)),
      "Submodule name must not contain a dot (got 'name.with.dot')");  // 断言异常消息包含特定的文本信息

  ASSERT_THROWS_WITH(  // 断言以下代码块抛出异常，并验证异常消息
      TestModel{}.register_module("", torch::nn::Linear(3, 4)),
      "Submodule name must not be empty");  // 断言异常消息包含特定的文本信息
}

TEST_F(ModuleTest, RegisterModuleThrowsForDuplicateModuleName) {  // 定义测试用例 RegisterModuleThrowsForDuplicateModuleName
  struct TestModel : public torch::nn::Module {};  // 定义一个名为 TestModel 的 PyTorch 模块
  TestModel model;  // 创建 TestModel 的实例 model
  model.register_module("linear", torch::nn::Linear(3, 4));  // 向模块注册名为 "linear" 的子模块，类型为线性层，输入维度为3，输出维度为4

  ASSERT_THROWS_WITH(  // 断言以下代码块抛出异常，并验证异常消息
      model.register_module("linear", torch::nn::Linear(3, 4)),
      "Submodule 'linear' already defined");  // 断言异常消息包含特定的文本信息
}

TEST_F(ModuleTest, ReplaceModuleThrowsForUnknownModuleName) {  // 定义测试用例 ReplaceModuleThrowsForUnknownModuleName
  torch::nn::Module model;  // 创建一个空的 PyTorch 模块 model

  ASSERT_THROWS_WITH(  // 断言以下代码块抛出异常，并验证异常消息
      model.replace_module("linear", torch::nn::Linear(3, 4)),
      "Submodule 'linear' is not defined");  // 断言异常消息包含特定的文本信息
}
TEST_F(ModuleTest, ReplaceModule) {
  // 定义一个内部测试模型，继承自 torch::nn::Module
  struct TestModel : public torch::nn::Module {
    // 定义一个线性层 l1
    torch::nn::Linear l1{nullptr};
    // 构造函数，在模型初始化时注册一个名为 "l1" 的线性层
    TestModel() {
      l1 = register_module("l1", torch::nn::Linear(3, 4));
    }
  };

  // 创建一个 TestModel 的 shared_ptr 智能指针对象
  auto model = std::make_shared<TestModel>();
  // 替换模型中名为 "l1" 的模块为新的 Linear(5, 6) 模块
  model->l1 = model->replace_module("l1", torch::nn::Linear(5, 6));
  // 断言检查替换后的 l1.weight 的大小是否为 6
  ASSERT_EQ(model->named_parameters()["l1.weight"].size(0), 6);
  // 断言检查 model->l1 是否等于 model 中名为 "l1" 的模块，并转换为 Linear 类型
  ASSERT_EQ(model->l1.get(), model->named_modules()["l1"]->as<Linear>());
}

TEST_F(ModuleTest, UnregisterModule) {
  // 定义一个内部测试模型，继承自 torch::nn::Module
  struct TestModel : public torch::nn::Module {};
  // 创建 TestModel 的实例
  TestModel model;
  // 断言检查尝试取消注册名为 "linear" 的模块时是否抛出异常
  ASSERT_THROWS_WITH(
      model.unregister_module("linear"),
      "No Module with name `linear` is registered");
  // 注册一个名为 "linear" 的 Linear(3, 4) 模块
  model.register_module("linear", torch::nn::Linear(3, 4));
  // 取消注册名为 "linear" 的模块
  model.unregister_module("linear");
  // 断言检查此时模型的 children 是否为空
  ASSERT_TRUE(model.children().empty());
}

TEST_F(ModuleTest, RegisterParameterThrowsForEmptyOrDottedName) {
  // 定义一个内部测试模型，继承自 torch::nn::Module
  struct TestModel : public torch::nn::Module {};
  // 断言检查尝试注册带点号的参数名时是否抛出异常
  ASSERT_THROWS_WITH(
      TestModel{}.register_parameter("name.with.dot", torch::ones(5)),
      "Parameter name must not contain a dot (got 'name.with.dot')");
  // 断言检查尝试注册空参数名时是否抛出异常
  ASSERT_THROWS_WITH(
      TestModel{}.register_parameter("", torch::ones(5)),
      "Parameter name must not be empty");
}

TEST_F(ModuleTest, RegisterParameterThrowsForDuplicateModuleName) {
  // 定义一个内部测试模型，继承自 torch::nn::Module
  struct TestModel : public torch::nn::Module {};
  // 创建 TestModel 的实例
  TestModel model;
  // 注册一个名为 "p" 的参数
  model.register_parameter("p", torch::ones(5));
  // 断言检查尝试再次注册名为 "p" 的参数时是否抛出异常
  ASSERT_THROWS_WITH(
      model.register_parameter("p", torch::ones(5)),
      "Parameter 'p' already defined");
}

TEST_F(ModuleTest, RegisterParameterUndefinedTensor) {
  // 定义一个内部测试模型，继承自 torch::nn::Module
  struct TestModel : public torch::nn::Module {};
  {
    // 在作用域内创建 TestModel 的实例
    TestModel model;
    // 注册一个名为 "undefined_tensor" 的未定义张量参数，不需要梯度
    model.register_parameter(
        "undefined_tensor", torch::Tensor(), /*requires_grad=*/false);
    // 断言检查此时模型的参数数量是否为 0
    ASSERT_EQ(model.parameters().size(), 0);
  }
  {
    // 在作用域内创建一个警告捕获对象
    WarningCapture warnings;

    // 在作用域内创建 TestModel 的实例
    TestModel model;
    // 注册一个名为 "undefined_tensor" 的未定义张量参数，未指定梯度
    model.register_parameter("undefined_tensor", torch::Tensor());
    // 断言检查此时模型的参数数量是否为 0
    ASSERT_EQ(model.parameters().size(), 0);

    // 断言检查警告字符串中关于 "requires_grad=true" 参数的出现次数是否为 1
    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "Ignoring the `requires_grad=true` function parameter"),
        1);
  }
}

TEST_F(ModuleTest, RegisterBufferThrowsForEmptyOrDottedName) {
  // 定义一个内部测试模型，继承自 torch::nn::Module
  struct TestModel : public torch::nn::Module {};
  // 断言检查尝试注册带点号的缓冲区名时是否抛出异常
  ASSERT_THROWS_WITH(
      TestModel{}.register_buffer("name.with.dot", torch::ones(5)),
      "Buffer name must not contain a dot (got 'name.with.dot')");
  // 断言检查尝试注册空缓冲区名时是否抛出异常
  ASSERT_THROWS_WITH(
      TestModel{}.register_buffer("", torch::ones(5)),
      "Buffer name must not be empty");
}

TEST_F(ModuleTest, RegisterBufferThrowsForDuplicateModuleName) {
  // 定义一个内部测试模型，继承自 torch::nn::Module
  struct TestModel : public torch::nn::Module {};
  // 创建 TestModel 的实例
  TestModel model;
  // 注册一个名为 "p" 的缓冲区
  model.register_buffer("p", torch::ones(5));
  // 断言检查尝试再次注册名为 "p" 的缓冲区时是否抛出异常
  ASSERT_THROWS_WITH(
      model.register_buffer("p", torch::ones(5)), "Buffer 'p' already defined");
}
TEST_F(ModuleTest, CanGetName) {
  // AGIUnit对象的测试实例
  AGIUnit agi;
  // 调用两次以确保懒惰初始化语义没有错误
  // 检查AGIUnit对象的名称是否为"AGIUnit"
  EXPECT_EQ(agi.name(), "AGIUnit");
  // 再次检查名称是否为"AGIUnit"
  EXPECT_EQ(agi.name(), "AGIUnit");
  // 使用test命名空间中的AGIUnit类的实例，检查名称是否为"test::AGIUnit"
  EXPECT_EQ(test::AGIUnit().name(), "test::AGIUnit");
  // 使用test命名空间中的AGIUnit2类的实例，检查名称是否为"Foo"
  EXPECT_EQ(test::AGIUnit2().name(), "Foo");
}

TEST_F(ModuleTest, AsCastsModulesCorrectly) {
  // 创建一个Linear模块对象，维度为3x4
  Linear module(3, 4);
  // 断言module对象的类型转换结果是否与module本身相同
  ASSERT_EQ(module->as<Linear>(), module.get());
  // 断言module对象的类型转换结果是否与LinearImpl类型相同
  ASSERT_EQ(module->as<LinearImpl>(), module.get());
  // 断言module对象的类型转换结果是否与Module类型相同
  ASSERT_EQ(module->as<Module>(), module.get());
  // 断言module对象转换为AGIUnit类型的结果是否为nullptr
  ASSERT_EQ(module->as<AGIUnit>(), nullptr);

  // 使用shared_ptr获取module对象的原始指针，进行类型转换测试
  std::shared_ptr<Module> raw = module.ptr();
  ASSERT_EQ(raw->as<Linear>(), module.get());
  ASSERT_EQ(raw->as<LinearImpl>(), module.get());
  ASSERT_EQ(raw->as<Module>(), module.get());
  ASSERT_EQ(raw->as<AGIUnit>(), nullptr);

  // 使用Module的引用进行类型转换测试
  Module& raw_ref = *raw.get();
  ASSERT_EQ(raw_ref.as<Linear>(), module.get());
  ASSERT_EQ(raw_ref.as<LinearImpl>(), module.get());
  ASSERT_EQ(raw_ref.as<Module>(), module.get());
  ASSERT_EQ(raw_ref.as<AGIUnit>(), nullptr);
  // 如果raw_ref对象能转换为Linear类型，则验证其权重的维度是否为2
  if (auto* linear = raw_ref.as<Linear>()) {
    ASSERT_EQ(linear->weight.ndimension(), 2);
  }

  // 创建AGIUnit的实例对象进行类型转换测试
  AGIUnit unit;
  ASSERT_EQ(unit.as<Linear>(), nullptr);
  ASSERT_EQ(unit.as<LinearImpl>(), nullptr);
  // 断言unit对象转换为AGIUnit类型的结果是否为unit自身的地址
  ASSERT_EQ(unit.as<AGIUnit>(), &unit);
}

void test_DeviceOrDtypeConversionSkipsUndefinedTensor(
    torch::Device to_device,
    torch::Dtype to_dtype) {
  {
    // Case 1: Undefined tensors as parameters
    // 创建一个Linear模块对象，初始化参数为10和20，不包含偏置项
    Linear module(LinearOptions(10, 20).bias(false));
    ASSERT_TRUE(module->weight.defined());
    ASSERT_FALSE(module->bias.defined());

    // 将module对象转移到指定的设备
    module->to(to_device);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.device().type(), to_device.type());
    ASSERT_FALSE(module->bias.defined());

    // 将module对象转换为指定的数据类型
    module->to(to_dtype);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.dtype(), to_dtype);
    ASSERT_FALSE(module->bias.defined());
  }
  {
    // Case 2: Undefined tensors as buffers
    // 创建一个BatchNorm1d模块对象，初始化参数为5，不跟踪运行统计数据，支持仿射变换
    BatchNorm1d module(
        BatchNorm1dOptions(5).track_running_stats(false).affine(true));
    ASSERT_TRUE(module->weight.defined());
    ASSERT_FALSE(module->running_mean.defined());

    // 将module对象转移到指定的设备
    module->to(to_device);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.device().type(), to_device.type());
    ASSERT_FALSE(module->running_mean.defined());

    // 将module对象转换为指定的数据类型
    module->to(to_dtype);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.dtype(), to_dtype);
    ASSERT_FALSE(module->running_mean.defined());
  }
}

TEST_F(ModuleTest, DeviceOrDtypeConversionSkipsUndefinedTensor) {
  // 测试设备或数据类型转换跳过未定义张量的情况（CPU和Double类型）
  test_DeviceOrDtypeConversionSkipsUndefinedTensor(torch::kCPU, torch::kDouble);
}

TEST_F(ModuleTest, DeviceOrDtypeConversionSkipsUndefinedTensor_CUDA) {
  // 测试设备或数据类型转换跳过未定义张量的情况（CUDA和Double类型）
  test_DeviceOrDtypeConversionSkipsUndefinedTensor(
      torch::kCUDA, torch::kDouble);
}

TEST_F(ModuleTest, ParametersAndBuffersAccessorSkipsUndefinedTensor) {
  {
    // ...
    // 创建一个具有指定输入和输出大小的线性模块，并设置偏置为 false
    Linear module(LinearOptions(10, 20).bias(false));
    
    // 获取模块的参数并进行断言检查，确保参数数量为1
    auto params = module->parameters();
    ASSERT_EQ(params.size(), 1);
    
    // 获取模块的命名参数并进行断言检查，确保命名参数数量为1
    auto named_params = module->named_parameters();
    ASSERT_EQ(named_params.size(), 1);
    
    // 断言检查第一个参数指针是否与命名参数中的 "weight" 参数指针相等
    ASSERT_TRUE(pointer_equal(params[0], named_params["weight"]));
    
    // 断言检查命名参数中的 "weight" 参数指针是否与模块自身的 weight 属性指针相等
    ASSERT_TRUE(pointer_equal(named_params["weight"], module->weight));
    
    {
        // 创建一个 BatchNorm1d 模块，设置特定的选项，关闭跟踪运行统计信息和仿射变换
        BatchNorm1d module(
            BatchNorm1dOptions(5).track_running_stats(false).affine(false));
    
        // 获取模块的缓冲区并进行断言检查，确保缓冲区大小为0
        auto buffers = module->buffers();
        ASSERT_EQ(buffers.size(), 0);
    
        // 获取模块的命名缓冲区并进行断言检查，确保命名缓冲区大小为0
        auto named_buffers = module->named_buffers();
        ASSERT_EQ(named_buffers.size(), 0);
    }
    
    {
        // 创建一个 BatchNorm1d 模块，设置特定的选项，启用跟踪运行统计信息但关闭仿射变换
        BatchNorm1d module(
            BatchNorm1dOptions(5).track_running_stats(true).affine(false));
    
        // 获取模块的缓冲区并进行断言检查，确保缓冲区大小为3
        auto buffers = module->buffers();
        ASSERT_EQ(buffers.size(), 3);
    
        // 获取模块的命名缓冲区并进行断言检查，确保命名缓冲区大小为3
        auto named_buffers = module->named_buffers();
        ASSERT_EQ(named_buffers.size(), 3);
    
        // 断言检查第一个缓冲区指针是否与命名缓冲区中的 "running_mean" 参数指针相等
        ASSERT_TRUE(pointer_equal(buffers[0], named_buffers["running_mean"]));
        
        // 断言检查命名缓冲区中的 "running_mean" 参数指针是否与模块自身的 running_mean 属性指针相等
        ASSERT_TRUE(pointer_equal(named_buffers["running_mean"], module->running_mean));
    
        // 断言检查第二个缓冲区指针是否与命名缓冲区中的 "running_var" 参数指针相等
        ASSERT_TRUE(pointer_equal(buffers[1], named_buffers["running_var"]));
        
        // 断言检查命名缓冲区中的 "running_var" 参数指针是否与模块自身的 running_var 属性指针相等
        ASSERT_TRUE(pointer_equal(named_buffers["running_var"], module->running_var));
    
        // 断言检查第三个缓冲区指针是否与命名缓冲区中的 "num_batches_tracked" 参数指针相等
        ASSERT_TRUE(pointer_equal(buffers[2], named_buffers["num_batches_tracked"]));
        
        // 断言检查命名缓冲区中的 "num_batches_tracked" 参数指针是否与模块自身的 num_batches_tracked 属性指针相等
        ASSERT_TRUE(pointer_equal(named_buffers["num_batches_tracked"], module->num_batches_tracked));
    }
}

// 定义一个名为 `Conversion_MultiCUDA` 的测试用例
TEST_F(ModuleTest, Conversion_MultiCUDA) {
  // 创建一个线性模块，输入大小为128，输出大小为64
  Linear module(128, 64);
  // 遍历模块的参数
  for (auto& parameter : module->parameters()) {
    // 断言参数的设备为 CPU
    ASSERT_EQ(parameter.device(), torch::Device(torch::kCPU));
    // 断言参数的数据类型为 float32
    ASSERT_EQ(parameter.dtype(), torch::kFloat32);
  }
  {
    // 将模块转移到 CUDA 设备 0
    module->to({torch::kCUDA, 0});
    // 再次遍历模块的参数
    for (auto& parameter : module->parameters()) {
      // 断言参数的设备类型为 CUDA
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CUDA);
      // 断言参数的 CUDA 设备索引为 0
      ASSERT_EQ(parameter.device().index(), 0);
    }
    // 将模块转移到 CUDA 设备 1
    module->to({torch::kCUDA, 1});
    // 再次遍历模块的参数
    for (auto& parameter : module->parameters()) {
      // 断言参数的设备类型为 CUDA
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CUDA);
      // 断言参数的 CUDA 设备索引为 1
      ASSERT_EQ(parameter.device().index(), 1);
    }
  }
  {
    // 将模块转移到 CPU 设备
    module->to(torch::Device(torch::kCPU));
    // 再次遍历模块的参数
    for (auto& parameter : module->parameters()) {
      // 断言参数的设备类型为 CPU
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CPU);
    }
  }
  {
    // 将模块的数据类型转换为 float64
    module->to(torch::kFloat64);
    // 再次遍历模块的参数
    for (auto& parameter : module->parameters()) {
      // 断言参数的数据类型为 float64
      ASSERT_EQ(parameter.dtype(), torch::kFloat64);
    }
  }
}

// 定义一个名为 `Conversion_NoGrad_MultiCUDA` 的测试用例
TEST_F(ModuleTest, Conversion_NoGrad_MultiCUDA) {
  // 创建一个线性模块，输入大小为128，输出大小为64
  Linear module(128, 64);
  // 遍历模块的参数
  for (auto& parameter : module->parameters()) {
    // 设置参数不需要梯度计算
    parameter.requires_grad_(false);
  }
  {
    // 将模块的数据类型转换为 int32
    module->to(torch::kInt32);
    // 再次遍历模块的参数
    for (auto& parameter : module->parameters()) {
      // 断言参数的数据类型为 int32
      ASSERT_EQ(parameter.dtype(), torch::kInt32);
    }
  }
  {
    // 将模块转移到 CUDA 设备 1，并指定数据类型为 uint8
    module->to(torch::Device(torch::kCUDA, 1), torch::kUInt8);
    // 再次遍历模块的参数
    for (auto& parameter : module->parameters()) {
      // 断言参数的设备类型为 CUDA
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CUDA);
      // 断言参数的 CUDA 设备索引为 1
      ASSERT_EQ(parameter.device().index(), 1);
    }
    // 再次遍历模块的参数
    for (auto& parameter : module->parameters()) {
      // 断言参数的数据类型为 uint8
      ASSERT_EQ(parameter.dtype(), torch::kUInt8);
    }
  }
}

// 定义一个名为 `CallingCloneOnModuleThatDoesNotOverrideCloneThrows` 的测试用例
TEST_F(ModuleTest, CallingCloneOnModuleThatDoesNotOverrideCloneThrows) {
  // 定义一个不可克隆的模块结构体
  struct UnCloneable : Module {};
  // 创建一个不可克隆的模块实例
  UnCloneable module;
  // 断言调用模块的克隆方法会抛出异常，提示未实现克隆方法
  ASSERT_THROWS_WITH(module.clone(), "clone() has not been implemented");
}

// 定义一个名为 `CallingCloneOnModuleThatDoesOverrideCloneDoesNotThrow` 的测试用例
TEST_F(ModuleTest, CallingCloneOnModuleThatDoesOverrideCloneDoesNotThrow) {
  // 定义一个可克隆的模块结构体
  struct Cloneable : Module {
    // 重写克隆方法，返回空指针
    std::shared_ptr<Module> clone(
        const torch::optional<torch::Device>& device =
            torch::nullopt) const override {
      return nullptr;
    }
  };
  // 创建一个可克隆的模块实例
  Cloneable module;
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言调用模块的克隆方法不会抛出异常
  ASSERT_NO_THROW({ module.clone(); });
}

// NOLINTNEXTLINE(bugprone-exception-escape)
// 定义一个名为 `TestDistinctParametersModule` 的结构体，继承自 `Cloneable<TestDistinctParametersModule>`
struct TestDistinctParametersModule
    : public Cloneable<TestDistinctParametersModule> {
  TestDistinctParametersModule() {
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
    // 调用重置方法
    reset();
  }
  void reset() override {
    // 注册模块 `l1`，线性层，输入大小为10，输出大小为3
    l1 = register_module("l1", Linear(10, 3));
    // 注册模块 `l2`，线性层，输入大小为3，输出大小为5
    l2 = register_module("l2", Linear(3, 5));
    // 注册模块 `l3`，线性层，输入大小为5，输出大小为100
    l3 = register_module("l3", Linear(5, 100));
    // 注册缓冲区 `buffer`，大小为2x2的全1张量
    buffer = register_buffer("buf", torch::ones({2, 2}));
  }

  // 线性模块 `l1`，`l2`，`l3`
  Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
  // 缓冲区 `buffer`
  torch::Tensor buffer;
};

// 定义一个测试函数 `testDistinctParameters`，接受一个 `Module` 实例 `m1`
void testDistinctParameters(
    std::shared_ptr<Module> m1,
    // 获取第一个模块 m1 的命名参数集合
    auto params1 = m1->named_parameters();
    // 获取第二个模块 m2 的命名参数集合
    auto params2 = m2->named_parameters();
    // 断言第一个模块的参数数量为 6
    ASSERT_EQ(params1.size(), 6);
    // 断言第二个模块的参数数量为 6
    ASSERT_EQ(params2.size(), 6);
    // 遍历第一个模块的参数集合
    for (auto& param : params1) {
        // 断言当前参数在两个模块中不是相同的指针
        ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
        // 断言当前参数在两个模块中的值接近
        ASSERT_TRUE(param->allclose(params2[param.key()]));
        // 对当前参数的值增加 2
        param->add_(2);
    }
    // 再次遍历第一个模块的参数集合
    for (auto& param : params1) {
        // 断言当前参数在两个模块中的值不再接近
        ASSERT_FALSE(param->allclose(params2[param.key()]));
    }
    
    // 获取第一个模块 m1 的命名缓冲区集合
    auto buffers1 = m1->named_buffers();
    // 获取第二个模块 m2 的命名缓冲区集合
    auto buffers2 = m2->named_buffers();
    // 断言第一个模块的缓冲区数量为 1
    ASSERT_EQ(buffers1.size(), 1);
    // 断言第二个模块的缓冲区数量为 1
    ASSERT_EQ(buffers2.size(), 1);
    // 遍历第一个模块的缓冲区集合
    for (auto& buffer : buffers1) {
        // 断言当前缓冲区在两个模块中不是相同的指针
        ASSERT_FALSE(pointer_equal(buffer.value(), buffers2[buffer.key()]));
        // 断言当前缓冲区在两个模块中的值接近
        ASSERT_TRUE(buffer->allclose(buffers2[buffer.key()]));
        // 对当前缓冲区的值增加 2
        buffer->add_(2);
    }
    // 再次遍历第一个模块的缓冲区集合
    for (auto& buffer : buffers1) {
        // 断言当前缓冲区在两个模块中的值不再接近
        ASSERT_FALSE(buffer->allclose(buffers2[buffer.key()]));
    }
TEST_F(ModuleTest, CloneCreatesDistinctParameters) {
  // 创建一个测试模块的共享指针对象
  auto module = std::make_shared<TestDistinctParametersModule>();
  // 进入无梯度计算的上下文
  torch::NoGradGuard no_grad;
  // 克隆模块，生成另一个模块对象
  auto module2 = module->clone();
  // 测试克隆后的两个模块的参数是否不同
  testDistinctParameters(module, module2);
}

TEST_F(ModuleTest, CloneCreatesDistinctParametersExplicitDevice_CUDA) {
  // 创建一个测试模块的共享指针对象
  auto module = std::make_shared<TestDistinctParametersModule>();
  // 进入无梯度计算的上下文
  torch::NoGradGuard no_grad;
  // 指定设备为 CUDA 设备 0
  torch::Device device(torch::kCUDA, 0);
  // 将模块移动到指定设备
  module->to(device);
  // 在指定设备上克隆模块，生成另一个模块对象
  auto module2 = module->clone(device);
  // 测试克隆后的两个模块的参数是否不同
  testDistinctParameters(module, module2);
}

TEST_F(ModuleTest, CloneCreatesDistinctParametersExplicitDevice_MultiCUDA) {
  // 创建一个测试模块的共享指针对象
  auto module = std::make_shared<TestDistinctParametersModule>();
  // 进入无梯度计算的上下文
  torch::NoGradGuard no_grad;
  // 指定两个 CUDA 设备
  torch::Device d0(torch::kCUDA, 0);
  torch::Device d1(torch::kCUDA, 1);
  // 将模块移动到第一个 CUDA 设备
  module->to(d0);
  // 在第二个 CUDA 设备上克隆模块，生成另一个模块对象
  auto module2 = module->clone(d1);

  // 检查原始模块参数是否在第一个 CUDA 设备上
  for (auto& param : module->parameters()) {
    ASSERT_EQ(param.device(), d0);
  }

  // 检查克隆模块参数是否在第二个 CUDA 设备上
  for (auto& param : module2->parameters()) {
    ASSERT_EQ(param.device(), d1);
  }

  // 将克隆模块移回第一个 CUDA 设备，因为 allclose 需要在相同设备上比较
  module2->to(d0);
  // 测试克隆后的两个模块的参数是否不同
  testDistinctParameters(module, module2);
}

TEST_F(ModuleTest, ClonePreservesExternalReferences) {
  // 定义一个测试模块结构体，继承自 Cloneable<TestModule>
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      // 重置函数，初始化权重参数为全 1 的 4x4 张量
      reset();
    }
    void reset() override {
      weight = register_parameter("weight", torch::ones({4, 4}));
    }
    torch::Tensor weight;
  };
  // 创建一个 TestModule 的共享指针对象
  auto module = std::make_shared<TestModule>();
  {
    // 进入无梯度计算的上下文
    torch::NoGradGuard no_grad;
    // 修改权重参数，加 1
    module->weight += 1;
  }
  // 断言：权重参数和命名参数中的 weight 是相同对象
  ASSERT_TRUE(pointer_equal(module->weight, module->named_parameters()["weight"]));
  // 断言：权重参数和命名参数中的 weight 在数值上是相似的
  ASSERT_TRUE(module->weight.allclose(module->named_parameters()["weight"]));

  // 克隆模块，得到另一个 TestModule 的共享指针对象
  auto module2 = std::dynamic_pointer_cast<TestModule>(
      std::shared_ptr<Module>(module->clone()));
  // 断言：克隆后的权重参数和原始模块的权重参数不是相同对象
  ASSERT_FALSE(pointer_equal(module2->weight, module->weight));
  // 断言：克隆后的权重参数和克隆模块的命名参数中的 weight 是相同对象
  ASSERT_TRUE(pointer_equal(module2->weight, module2->named_parameters()["weight"]));
  // 断言：克隆后的权重参数在数值上和克隆模块的命名参数中的 weight 是相似的
  ASSERT_TRUE(module2->weight.allclose(module2->named_parameters()["weight"]));
  // 断言：克隆后的权重参数在数值上和原始模块的权重参数是相似的
  ASSERT_TRUE(module2->weight.allclose(module->weight));
  // 断言：克隆后的权重参数和克隆模块的命名参数中的 weight 不是相同对象
  ASSERT_FALSE(pointer_equal(module2->weight, module->named_parameters()["weight"]));
}

TEST_F(ModuleTest, CloneCopiesTheValuesOfVariablesOfSubmodules) {
  // 定义一个测试模块结构体，继承自 Cloneable<TestModule>
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      // 重置函数，初始化权重参数为全 1 的 4x4 张量
      reset();
    }
    void reset() override {
      weight = register_parameter("weight", torch::ones({4, 4}));
    }

    torch::Tensor weight;
    int value = 0;
  };
  // 定义一个嵌套的测试模块结构体，继承自 Cloneable<NestedModule>
  struct NestedModule : public Cloneable<NestedModule> {
    NestedModule() {
      // 重置函数，初始化权重参数为全 1 的 4x4 张量
      reset();
    }
    void reset() override {
      weight = register_parameter("weight", torch::ones({4, 4}));
    }

    torch::Tensor weight;
  };
}
    // 重置函数，覆盖基类中的同名函数
    void reset() override {
      // 使用指定的名称注册一个新模块，并初始化为 TestModule 的实例
      module = register_module("module", std::make_shared<TestModule>());
    }
    // 使用 std::shared_ptr<TestModule> 类型的智能指针来持有 TestModule 的实例
    std::shared_ptr<TestModule> module;
  };

  // 创建一个指向 NestedModule 实例的智能指针 a
  auto a = std::make_shared<NestedModule>();
  {
    // 进入无梯度计算环境
    torch::NoGradGuard no_grad;
    // 递增 a 模块中的权重值
    a->module->weight += 1;
    // 设置 a 模块中的值为 123
    a->module->value = 123;
  }

  // 克隆指针 a 指向的 NestedModule 实例，并将其转换为 SharedPtr 类型的指针 b
  auto b = std::dynamic_pointer_cast<NestedModule>(a->clone());

  // 断言 b 模块的权重与 a 模块的权重不相等
  ASSERT_FALSE(pointer_equal(b->module->weight, a->module->weight));
  // 断言 b 模块的权重与 b 模块的名为 "weight" 的参数相等
  ASSERT_TRUE(pointer_equal(
      b->module->weight, b->module->named_parameters()["weight"]));
  // 断言 b 模块的名为 "weight" 的参数与 a 模块的权重在数值上相似
  ASSERT_TRUE(
      b->module->named_parameters()["weight"].allclose(a->module->weight));
  // 断言 b 模块的权重与 a 模块的权重在数值上相似
  ASSERT_TRUE(b->module->weight.allclose(a->module->weight));
  // 断言 b 模块的值与 a 模块的值相等
  ASSERT_EQ(b->module->value, a->module->value);
}

// 定义一个测试用例，验证模块克隆到指定 CUDA 设备后，所有参数都保持在该设备上
TEST_F(ModuleTest, CloneToDevicePreservesTheDeviceOfParameters_CUDA) {
  // NOLINTNEXTLINE(bugprone-exception-escape)
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
      reset();
    }
    // 重置模块，注册线性层和缓冲区
    void reset() override {
      l1 = register_module("l1", Linear(10, 3));
      l2 = register_module("l2", Linear(3, 5));
      l3 = register_module("l3", Linear(5, 100));
      buffer = register_buffer("buf", torch::ones({2, 2}));
    }

    Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    torch::Tensor buffer;
  };

  TestModule m;
  torch::Device device(torch::kCUDA, 0);

  // 将模块 m 移动到指定的 CUDA 设备上
  m.to(device);

  // 克隆模块 m
  auto clone = m.clone();

  // 验证克隆后所有参数都位于指定的 CUDA 设备上
  for (const auto& parameter : clone->parameters()) {
    ASSERT_EQ(parameter.device().type(), device.type());
    ASSERT_EQ(parameter.device().index(), device.index());
  }
  for (const auto& buffer : clone->buffers()) {
    ASSERT_EQ(buffer.device().type(), device.type());
    ASSERT_EQ(buffer.device().index(), device.index());
  }
}

// 定义一个测试用例，验证模块克隆到指定的 CUDA 设备上后，所有参数都位于该设备上（多 GPU 场景）
TEST_F(
    ModuleTest,
    CloningToAParticularDevicePlacesAllParametersThere_MultiCUDA) {
  // NOLINTNEXTLINE(bugprone-exception-escape)
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
      reset();
    }
    // 重置模块，注册线性层和缓冲区
    void reset() override {
      l1 = register_module("l1", Linear(10, 3));
      l2 = register_module("l2", Linear(3, 5));
      l3 = register_module("l3", Linear(5, 100));
      buffer = register_buffer("buf", torch::ones({2, 2}));
    }

    Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    torch::Tensor buffer;
  };

  TestModule m;
  torch::Device device(torch::kCUDA, 1);

  // 克隆模块 m 到指定的 CUDA 设备上
  auto clone = m.clone(device);

  // 验证克隆后所有参数都位于指定的 CUDA 设备上
  for (const auto& parameter : clone->parameters()) {
    ASSERT_EQ(parameter.device().type(), device.type());
    ASSERT_EQ(parameter.device().index(), device.index());
  }
  for (const auto& buffer : clone->buffers()) {
    ASSERT_EQ(buffer.device().type(), device.type());
    ASSERT_EQ(buffer.device().index(), device.index());
  }
}

// 定义参数测试模块，用于验证参数的数量和名称
struct ParameterTestModule : Module {
  ParameterTestModule() {
    // 注册参数 a、b、c，并初始化它们的值
    a = register_parameter("a", torch::zeros({2, 2}));
    b = register_parameter("b", torch::ones({2, 2}));
    c = register_parameter("c", torch::ones({2, 2}) * 2);
  }

  torch::Tensor a, b, c;
};

// 测试用例：验证模块的参数数量是否正确
TEST_F(ModuleTest, HasCorrectNumberOfParameters) {
  ParameterTestModule module;
  ASSERT_EQ(module.parameters().size(), 3); // 断言模块参数的数量为 3
  ASSERT_EQ(module.named_parameters().size(), 3); // 断言模块命名参数的数量为 3
}

// 测试用例：验证模块包含具有正确名称的参数
TEST_F(ModuleTest, ContainsParametersWithTheCorrectName) {
  ParameterTestModule module;
  auto parameters = module.named_parameters();
  ASSERT_TRUE(parameters.contains("a")); // 断言参数中包含名称为 "a" 的参数
  ASSERT_TRUE(parameters.contains("b")); // 断言参数中包含名称为 "b" 的参数
  ASSERT_TRUE(parameters.contains("c")); // 断言参数中包含名称为 "c" 的参数
}

// 定义缓冲区测试模块，用于验证缓冲区的注册
struct BufferTestModule : Module {
  BufferTestModule() {
    // 注册缓冲区 a、b，并初始化它们的值
    a = register_buffer("a", torch::zeros({2, 2}));
    b = register_buffer("b", torch::ones({2, 2}));
    c = register_buffer("c", torch::ones({2, 2}) * 2);

# 使用 `register_buffer` 函数注册一个名为 "c" 的缓冲区，其内容为一个 2x2 的张量，所有元素的值为2。

  torch::Tensor a, b, c;

# 声明了三个 Torch 张量变量 a, b, c，其中 c 是之前注册的缓冲区，但此时并未赋值。
};

// 定义一个测试类 ModuleTest，用于测试模块的缓冲区相关功能
TEST_F(ModuleTest, HasCorrectNumberOfBuffers) {
  // 创建一个 BufferTestModule 对象
  BufferTestModule module;
  // 断言缓冲区的数量为 3
  ASSERT_EQ(module.buffers().size(), 3);
  // 断言命名缓冲区的数量也为 3
  ASSERT_EQ(module.named_buffers().size(), 3);
}

// 测试模块是否包含了具有正确名称的缓冲区
TEST_F(ModuleTest, ContainsBuffersWithTheCorrectName) {
  // 创建一个 BufferTestModule 对象
  BufferTestModule module;
  // 获取命名缓冲区的映射
  auto buffers = module.named_buffers();
  // 断言是否包含名称为 "a" 的缓冲区
  ASSERT_TRUE(buffers.contains("a"));
  // 断言是否包含名称为 "b" 的缓冲区
  ASSERT_TRUE(buffers.contains("b"));
  // 断言是否包含名称为 "c" 的缓冲区
  ASSERT_TRUE(buffers.contains("c"));
}

// 定义一个结构体 AImpl，继承自 torch::nn::Module
struct AImpl : torch::nn::Module {
  // 默认构造函数，初始化 x_ 为 123
  AImpl() : x_(123) {}
  // 带参数的构造函数，根据参数设置 x_
  AImpl(int x) : x_(x) {}
  // 整数类型的成员变量 x_
  int x_;
};
// 使用 TORCH_MODULE 宏将 AImpl 定义为 Torch 模块 A
TORCH_MODULE(A);

// 测试模块的默认构造函数是否调用了实现类的默认构造函数
TEST_F(
    ModuleTest,
    DefaultConstructorOfModuleHolderCallsDefaultConstructorOfImpl) {
  // 创建一个 A 类型的对象 a
  A a;
  // 断言 a 不为空
  ASSERT_TRUE(a);
  // 断言 a 不是空模块
  ASSERT_FALSE(a.is_empty());
  // 断言 a 的 x_ 值为 123
  ASSERT_EQ(a->x_, 123);
}

// 测试模块的带值构造函数是否调用了实现类的相应构造函数
TEST_F(
    ModuleTest,
    ValueConstructorOfModuleHolderCallsCorrectConstructorInImpl) {
  // 创建一个带参数的 A 类型对象 a，参数为 5
  A a(5);
  // 断言 a 不为空
  ASSERT_TRUE(a);
  // 断言 a 不是空模块
  ASSERT_FALSE(a.is_empty());
  // 断言 a 的 x_ 值为 5
  ASSERT_EQ(a->x_, 5);
}

// 测试模块的空指针构造函数是否使 ModuleHolder 保持空状态
TEST_F(ModuleTest, NullptrConstructorLeavesTheModuleHolderInEmptyState) {
  // 使用 nullptr 初始化 A 类型对象 a
  A a = nullptr;
  // 断言 a 为空
  ASSERT_FALSE(a);
  // 断言 a 是空模块
  ASSERT_TRUE(a.is_empty());
  // 尝试访问 a 的 x_ 成员变量，预期抛出异常 "Accessing empty ModuleHolder"
  ASSERT_THROWS_WITH(a->x_, "Accessing empty ModuleHolder");
}

// 定义一个测试模块 TestModule，继承自 torch::nn::Module
struct TestModule : public torch::nn::Module {
  // 构造函数，根据给定大小注册参数和缓冲区
  TestModule(int64_t size) {
    p1 = register_parameter("p1", torch::randn({size}));
    p2 = register_parameter("p2", torch::randn({size}));
    b1 = register_buffer("b1", torch::randn({size}));
    b2 = register_buffer("b2", torch::randn({size}));
  }

  // 前向传播函数，简单地返回输入
  torch::Tensor forward(torch::Tensor input) {
    return input;
  }

  // 参数和缓冲区
  torch::Tensor p1, p2, b1, b2;
};

// 测试模块的 modules() 方法是否返回预期的子模块（扁平模型情况）
TEST_F(ModuleTest, ModulesReturnsExpectedSubmodulesForFlatModel) {
  // 创建一个包含三个 TestModule 的序列模型
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  // 获取模型的所有模块
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->modules();
  // 预期的模块顺序
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model.ptr(), model[0], model[1], model[2]};
  // 断言实际返回的模块数量与预期相等
  ASSERT_EQ(modules.size(), expected.size());
  // 逐一断言每个返回的模块与预期的模块指针相等
  for (const auto i : c10::irange(expected.size())) {
    // 断言指针相等
    ASSERT_EQ(modules[i].get(), expected[i].get());
  }
}

// 测试模块的 modules() 方法在 include_self 设为 false 时是否排除自身
TEST_F(ModuleTest, ModulesExcludesSelfWhenIncludeSelfSetToFalse) {
  // 创建一个包含三个 TestModule 的序列模型
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  // 获取模型的所有模块，包括自身设为 false
  std::vector<std::shared_ptr<torch::nn::Module>> modules =
      model->modules(/*include_self=*/false);
  // 预期的模块顺序
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  // 断言实际返回的模块数量与预期相等
  ASSERT_EQ(modules.size(), expected.size());
  // 逐一断言每个返回的模块与预期的模块指针相等
  for (const auto i : c10::irange(expected.size())) {
    // 断言指针相等
    ASSERT_EQ(modules[i].get(), expected[i].get());
  }
}
TEST_F(ModuleTest, NamedModulesReturnsExpectedNamedSubmodulesForFlatModel) {
  // 创建一个包含三个测试模块的顺序神经网络模型
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  // 获取模型中所有命名模块的有序字典
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_modules();
  // 期望的模块列表，包括模型自身及其三个子模块
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model.ptr(), model[0], model[1], model[2]};
  // 断言实际得到的命名模块数与期望数相同
  ASSERT_EQ(modules.size(), expected.size());
  // 遍历期望列表，逐个断言命名模块的键和值等于期望的模块
  for (const auto i : c10::irange(expected.size())) {
    // 断言指针相等
    ASSERT_EQ(modules[i].key(), i ? std::to_string(i - 1) : std::string());
    ASSERT_EQ(modules[i].value().get(), expected[i].get());
  }
}

TEST_F(ModuleTest, NamedModulesExcludesSelfWhenIncludeSelfSetToFalse) {
  // 创建一个包含三个测试模块的顺序神经网络模型
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  // 获取模型中除自身外的所有命名模块的有序字典
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_modules(
          /*name_prefix=*/std::string(), /*include_self=*/false);
  // 期望的模块列表，仅包括三个子模块，不包括模型自身
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  // 断言实际得到的命名模块数与期望数相同
  ASSERT_EQ(modules.size(), expected.size());
  // 遍历期望列表，逐个断言命名模块的键和值等于期望的模块
  for (const auto i : c10::irange(expected.size())) {
    // 断言键是预期的数字字符串
    ASSERT_EQ(modules[i].key(), std::to_string(i));
    ASSERT_EQ(modules[i].value().get(), expected[i].get());
  }
}

TEST_F(ModuleTest, ChildrenReturnsExpectedSubmodulesForFlatModel) {
  // 创建一个包含三个测试模块的顺序神经网络模型
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  // 获取模型中所有子模块的列表
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->children();
  // 期望的子模块列表，包括模型的三个子模块
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  // 断言实际得到的子模块数与期望数相同
  ASSERT_EQ(modules.size(), expected.size());
  // 遍历期望列表，逐个断言子模块的指针相等
  for (const auto i : c10::irange(expected.size())) {
    // 断言指针相等
    ASSERT_EQ(modules[i].get(), expected[i].get());
  }

  // 对于这个平铺模型，这个断言应该成立
  ASSERT_EQ(modules, model->modules(/*include_self=*/false));
}

TEST_F(ModuleTest, NamedChildrenReturnsExpectedNamedSubmodulesForFlatModel) {
  // 创建一个包含三个测试模块的顺序神经网络模型
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  // 获取模型中所有命名子模块的有序字典
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_children();
  // 期望的子模块列表，包括模型的三个子模块
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  // 断言实际得到的命名子模块数与期望数相同
  ASSERT_EQ(modules.size(), expected.size());
  // 遍历期望列表，逐个断言命名子模块的键和值等于期望的子模块
  for (const auto i : c10::irange(expected.size())) {
    // 断言键是预期的数字字符串
    ASSERT_EQ(modules[i].key(), std::to_string(i));
    ASSERT_EQ(modules[i].value().get(), expected[i].get());
  }
}

TEST_F(ModuleTest, ParametersReturnsExpectedTensorsForFlatModel) {
  // 创建一个具有参数的测试模块
  TestModule module(1);
  // 获取模块的所有参数张量列表
  std::vector<torch::Tensor> parameters = module.parameters();
  // 断言参数张量的数量为2
  ASSERT_EQ(parameters.size(), 2);
  // 断言第一个参数张量的数据指针与模块的第一个参数的数据指针相等
  ASSERT_EQ(parameters[0].data_ptr<float>(), module.p1.data_ptr<float>());
  // 断言第二个参数张量的数据指针与模块的第二个参数的数据指针相等
  ASSERT_EQ(parameters[1].data_ptr<float>(), module.p2.data_ptr<float>());
}
// 使用 Google Test 的 TEST_F 宏定义一个测试用例，验证在扁平模型下，named_parameters 方法返回预期的张量
TEST_F(ModuleTest, NamedParametersReturnsExpectedTensorsForFlatModel) {
  // 创建一个 TestModule 实例，参数为 1
  TestModule module(1);
  // 调用 named_parameters 方法获取模型的参数字典
  torch::OrderedDict<std::string, torch::Tensor> parameters =
      module.named_parameters();
  // 断言参数字典的大小为 2
  ASSERT_EQ(parameters.size(), 2);
  // 断言第一个参数的键为 "p1"
  ASSERT_EQ(parameters[0].key(), "p1");
  // 断言第一个参数的数据指针与 module.p1 的数据指针相等
  ASSERT_EQ(parameters[0]->data_ptr<float>(), module.p1.data_ptr<float>());
  // 断言第二个参数的键为 "p2"
  ASSERT_EQ(parameters[1].key(), "p2");
  // 断言第二个参数的数据指针与 module.p2 的数据指针相等
  ASSERT_EQ(parameters[1]->data_ptr<float>(), module.p2.data_ptr<float>());
}

// 使用 Google Test 的 TEST_F 宏定义一个测试用例，验证在扁平模型下，buffers 方法返回预期的张量
TEST_F(ModuleTest, BuffersReturnsExpectedTensorsForFlatModel) {
  // 创建一个 TestModule 实例，参数为 1
  TestModule module(1);
  // 调用 buffers 方法获取模型的缓冲区张量列表
  std::vector<torch::Tensor> buffers = module.buffers();
  // 断言缓冲区张量列表的大小为 2
  ASSERT_EQ(buffers.size(), 2);
  // 断言第一个缓冲区张量的数据指针与 module.b1 的数据指针相等
  ASSERT_EQ(buffers[0].data_ptr<float>(), module.b1.data_ptr<float>());
  // 断言第二个缓冲区张量的数据指针与 module.b2 的数据指针相等
  ASSERT_EQ(buffers[1].data_ptr<float>(), module.b2.data_ptr<float>());
}

// 使用 Google Test 的 TEST_F 宏定义一个测试用例，验证在扁平模型下，named_buffers 方法返回预期的张量
TEST_F(ModuleTest, NamedBuffersReturnsExpectedTensorsForFlatModel) {
  // 创建一个 TestModule 实例，参数为 1
  TestModule module(1);
  // 调用 named_buffers 方法获取模型的命名缓冲区张量字典
  torch::OrderedDict<std::string, torch::Tensor> buffers =
      module.named_buffers();
  // 断言命名缓冲区张量字典的大小为 2
  ASSERT_EQ(buffers.size(), 2);
  // 断言第一个命名缓冲区张量的键为 "b1"
  ASSERT_EQ(buffers[0].key(), "b1");
  // 断言第一个命名缓冲区张量的数据指针与 module.b1 的数据指针相等
  ASSERT_EQ(buffers[0]->data_ptr<float>(), module.b1.data_ptr<float>());
  // 断言第二个命名缓冲区张量的键为 "b2"
  ASSERT_EQ(buffers[1].key(), "b2");
  // 断言第二个命名缓冲区张量的数据指针与 module.b2 的数据指针相等
  ASSERT_EQ(buffers[1]->data_ptr<float>(), module.b2.data_ptr<float>());
}

// 定义一个继承自 torch::nn::Module 的 TestContainer 结构体，用于测试容器模型的嵌套
struct TestContainer : torch::nn::Module {
  TestContainer(int64_t number, std::vector<TestContainer> modules = {})
      : tensor(torch::tensor(number)) {
    // 遍历 modules，逐个注册子模块
    for (const auto i : c10::irange(modules.size())) {
      register_module(
          std::to_string(i),
          std::make_shared<TestContainer>(std::move(modules[i])));
    }
  }
  torch::Tensor tensor;  // 定义一个张量成员变量
};

// 定义一个函数，返回 TestContainer 模块的 tensor 成员的整数值
int64_t get_test_container_item(std::shared_ptr<torch::nn::Module> module) {
  // 将传入的模块指针转换为 TestContainer 类型，并获取其 tensor 成员的整数值
  return std::dynamic_pointer_cast<TestContainer>(module)
      ->tensor.item<int64_t>();
}

// 创建一个深度嵌套的 TestContainer 模块树，并返回其根节点的指针
std::shared_ptr<TestContainer> make_deeply_nested_test_container() {
  // 使用 TestContainer 构造函数创建一个深度嵌套的模块树
  return std::make_shared<TestContainer>(TestContainer(
      0,
      {TestContainer(1, {TestContainer(2), TestContainer(3)}),
       TestContainer(4),
       TestContainer(
           5,
           {TestContainer(6),
            TestContainer(7, {TestContainer(8), TestContainer(9)})})}));
}

// 创建一个键值对数组，描述深度嵌套的 TestContainer 模块树的结构
std::vector<std::pair<std::string, int64_t>>
make_key_value_pairs_for_deeply_nested_container() {
  // 返回一个预定义的键值对数组，描述深度嵌套的 TestContainer 模块树的结构
  return {
      {"test_prefix", 0},
      {"test_prefix.0", 1},
      {"test_prefix.0.0", 2},
      {"test_prefix.0.1", 3},
      {"test_prefix.1", 4},
      {"test_prefix.2", 5},
      {"test_prefix.2.0", 6},
      {"test_prefix.2.1", 7},
      {"test_prefix.2.1.0", 8},
      {"test_prefix.2.1.1", 9}};
}

// 使用 Google Test 的 TEST_F 宏定义一个测试用例，验证对于深度嵌套的模型，modules 方法返回预期的子模块列表
TEST_F(ModuleTest, ModulesReturnsExpectedSubmodulesForDeepModel) {
  // 创建一个深度嵌套的 TestContainer 模块树
  auto model = make_deeply_nested_test_container();
  // 调用 modules 方法获取模型的所有子模块列表
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->modules();

  // 断言模块列表的大小为 10
  ASSERT_EQ(modules.size(), 10);
  // 使用循环遍历每个模块，并断言其 tensor 成员的整数值与其在模块列表中的索引相等
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(get_test_container_item(modules[i]), i);
  }
}
TEST_F(ModuleTest, NamedModulesReturnsExpectedNamedSubmodulesForDeepModel) {
  // 创建一个深度嵌套的测试容器模型
  auto model = make_deeply_nested_test_container();
  // 获取模型中命名子模块的有序字典，以指定的前缀命名
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_modules(/*name_prefix=*/"test_prefix");
  // 准备预期的深度嵌套容器的键值对
  auto expected = make_key_value_pairs_for_deeply_nested_container();

  // 断言命名子模块的数量与预期的数量相等
  ASSERT_EQ(modules.size(), expected.size());

  // 遍历预期结果列表
  for (const auto i : c10::irange(expected.size())) {
    // 断言每个模块的键与预期的键相等
    ASSERT_EQ(modules[i].key(), expected[i].first);
    // 断言每个模块的值与预期的值相等
    ASSERT_EQ(get_test_container_item(modules[i].value()), expected[i].second);
  }
}

TEST_F(ModuleTest, ChildrensReturnsExpectedSubmodulesForDeepModel) {
  // 创建一个深度嵌套的测试容器模型
  auto model = make_deeply_nested_test_container();
  // 获取模型的子模块列表
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->children();

  // 断言子模块的数量为3
  ASSERT_EQ(modules.size(), 3);
  // 断言第一个子模块的特定测试容器项
  ASSERT_EQ(get_test_container_item(modules[0]), 1);
  // 断言第二个子模块的特定测试容器项
  ASSERT_EQ(get_test_container_item(modules[1]), 4);
  // 断言第三个子模块的特定测试容器项
  ASSERT_EQ(get_test_container_item(modules[2]), 5);
}

TEST_F(ModuleTest, NamedChildrensReturnsExpectedNamedSubmodulesForDeepModel) {
  // 创建一个深度嵌套的测试容器模型
  auto model = make_deeply_nested_test_container();
  // 获取模型中命名子模块的有序字典
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_children();

  // 断言命名子模块的数量为3
  ASSERT_EQ(modules.size(), 3);

  // 断言第一个命名子模块的值等于预期的第一个值
  ASSERT_EQ(get_test_container_item(modules[0].value()), 1);
  // 断言第一个命名子模块的键为 "0"
  ASSERT_EQ(modules[0].key(), "0");

  // 断言第二个命名子模块的值等于预期的第二个值
  ASSERT_EQ(get_test_container_item(modules[1].value()), 4);
  // 断言第二个命名子模块的键为 "1"
  ASSERT_EQ(modules[1].key(), "1");

  // 断言第三个命名子模块的值等于预期的第三个值
  ASSERT_EQ(get_test_container_item(modules[2].value()), 5);
  // 断言第三个命名子模块的键为 "2"
  ASSERT_EQ(modules[2].key(), "2");
}

TEST_F(ModuleTest, ModuleApplyIteratesCorreclty) {
  // 创建一个深度嵌套的测试容器模型
  auto model = make_deeply_nested_test_container();
  // 初始化索引为0
  int64_t index = 0;
  // 对模型中的每个模块应用函数，断言每个模块的 tensor 值与索引相等
  model->apply([&index](torch::nn::Module& module) {
    ASSERT_EQ(module.as<TestContainer>()->tensor.item<int64_t>(), index++);
  });
  // 断言索引的最终值为10
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, ConstModuleApplyIteratesCorreclty) {
  // 创建一个深度嵌套的测试容器模型，并声明为常量模型指针
  std::shared_ptr<const TestContainer> model =
      make_deeply_nested_test_container();
  // 初始化索引为0
  int64_t index = 0;
  // 对模型中的每个模块应用函数，断言每个模块的 tensor 值与索引相等
  model->apply([&index](const torch::nn::Module& module) {
    ASSERT_EQ(module.as<TestContainer>()->tensor.item<int64_t>(), index++);
  });
  // 断言索引的最终值为10
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, NamedModuleApplyIteratesCorreclty) {
  // 创建一个深度嵌套的测试容器模型
  auto model = make_deeply_nested_test_container();
  // 准备预期的深度嵌套容器的键值对
  auto expected = make_key_value_pairs_for_deeply_nested_container();
  // 初始化索引为0
  int64_t index = 0;
  // 对模型中的每个命名模块应用函数，断言每个模块的名称与预期名称相等，同时断言每个模块的 tensor 值与预期值相等
  model->apply(
      [&index, expected](const std::string& name, torch::nn::Module& module) {
        ASSERT_EQ(name, expected[index].first);
        ASSERT_EQ(
            module.as<TestContainer>()->tensor.item<int64_t>(),
            expected[index++].second);
      },
      /*name_prefix=*/"test_prefix");
  // 断言索引的最终值为10
  ASSERT_EQ(index, 10);
}
TEST_F(ModuleTest, ConstNamedModuleApplyIteratesCorreclty) {
  // 创建一个指向深度嵌套测试容器的常量共享指针模型
  std::shared_ptr<const TestContainer> model =
      make_deeply_nested_test_container();
  // 创建预期输出的键值对列表
  auto expected = make_key_value_pairs_for_deeply_nested_container();
  // 初始化索引为0
  int64_t index = 0;
  // 对模型应用函数，迭代每个命名模块
  model->apply(
      [&index, &expected](
          const std::string& name, const torch::nn::Module& module) {
        // 断言当前模块名称与预期名称相等
        ASSERT_EQ(name, expected[index].first);
        // 断言当前模块的整数张量值与预期值相等
        ASSERT_EQ(
            module.as<const TestContainer>()->tensor.item<int64_t>(),
            expected[index++].second);
      },
      /*name_prefix=*/"test_prefix");
  // 最终索引应为10
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, ModulePointerApplyIteratesCorreclty) {
  // 创建一个指向深度嵌套测试容器的共享指针模型
  auto model = make_deeply_nested_test_container();
  // 初始化索引为0
  int64_t index = 0;
  // 对模型应用函数，迭代每个模块的指针
  model->apply([&index](const std::shared_ptr<torch::nn::Module>& module) {
    // 断言获取的测试容器项与当前索引相等
    ASSERT_EQ(get_test_container_item(module), index++);
  });
  // 最终索引应为10
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, NamedModulePointerApplyIteratesCorreclty) {
  // 创建一个指向深度嵌套测试容器的共享指针模型
  auto model = make_deeply_nested_test_container();
  // 创建预期输出的键值对列表
  auto expected = make_key_value_pairs_for_deeply_nested_container();
  // 初始化索引为0
  int64_t index = 0;
  // 对模型应用函数，迭代每个命名模块的指针
  model->apply(
      [&index, &expected](
          const std::string& name,
          const std::shared_ptr<torch::nn::Module>& module) {
        // 断言当前模块名称与预期名称相等
        ASSERT_EQ(name, expected[index].first);
        // 断言获取的测试容器项与当前索引相等
        ASSERT_EQ(get_test_container_item(module), expected[index++].second);
      },
      /*name_prefix=*/"test_prefix");
  // 最终索引应为10
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, ThrowsWhenAttemptingtoGetTopLevelModuleAsSharedPtr) {
  {
    // 创建测试模块，其值为1
    TestModule module(1);
    // 断言抛出异常信息包含指定的文本
    ASSERT_THROWS_WITH(
        module.modules(),
        "It looks like you attempted to retrieve "
        "your top-level module as a shared_ptr")
  }
  {
    // 创建测试模块，其值为1
    TestModule module(1);
    // 不应抛出异常，获取模块的子模块，不包括自身
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_NO_THROW(module.modules(/*include_self=*/false));
  }
  {
    // 创建测试模块的共享指针，其值为1
    auto module = std::make_shared<TestModule>(1);
    // 不应抛出异常，获取模块的子模块
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_NO_THROW(module->modules());
  }
}

struct EmptyModule : torch::nn::Module {};

TEST_F(ModuleTest, PrettyPrint) {
  // 定义测试模块，继承自 torch::nn::Module
  struct TestModule : torch::nn::Module {
    TestModule(int x, float y) : x_(x), y_(y) {}

    // 重写输出流的美观打印函数
    void pretty_print(std::ostream& stream) const override {
      // 将模块的参数 x 和 y 输出到流中
      stream << "TestModule(x=" << x_ << ", y=" << y_ << ")";
    }

    int x_;
    float y_;
  };

  // 断言空模块的字符串表示为 "EmptyModule"
  ASSERT_EQ(c10::str(EmptyModule{}), "EmptyModule");
  // 断言 TestModule 的字符串表示正确显示参数值
  ASSERT_EQ(c10::str(TestModule(1, 3.14)), "TestModule(x=1, y=3.14)");
}

struct ModuleWithNonTensorForwardImpl : torch::nn::Module {
  // 实现一个非张量输入的 forward 函数
  int64_t forward(torch::Tensor x) {
    return x.numel();
  }
};
// 通过 TORCH_MODULE 定义 ModuleWithNonTensorForward 类
TORCH_MODULE(ModuleWithNonTensorForward);

TEST_F(ModuleTest, CanCallForwardOnNonTensorForwardThroughPimpl) {
  // 创建 ModuleWithNonTensorForward 类型的对象 m
  ModuleWithNonTensorForward m;
  // 断言调用 m 的 forward 函数能正确返回输入张量的元素数
  ASSERT_EQ(m(torch::ones(123)), 123);
}
```