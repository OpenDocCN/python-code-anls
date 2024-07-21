# `.\pytorch\test\cpp\dist_autograd\test_dist_autograd.cpp`

```
// 包含必要的头文件：内存管理、单元测试框架、ATen库、分布式自动求导引擎相关头文件等
#include <memory>
#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/torch.h>

// 命名空间：torch -> distributed -> autograd
namespace torch {
namespace distributed {
namespace autograd {

// DistAutogradTest 类，继承自 ::testing::Test
class DistAutogradTest : public ::testing::Test {
 protected:
  // 设置测试用例的静态方法
  static void SetUpTestCase() {
    // 初始化自动求导容器 autogradContainer_，使用 ID 0
    autogradContainer_ = &DistAutogradContainer::init(0);
  }

  // 在每个测试用例结束时执行的方法
  void TearDown() override {
    // 释放当前上下文的资源
    autogradContainer_->releaseContext(
        autogradContainer_->currentContext()->contextId());
  }

  // 静态成员变量：分布式自动求导容器指针 autogradContainer_
  static DistAutogradContainer* autogradContainer_;
};

// 初始化静态成员变量 autogradContainer_
DistAutogradContainer* DistAutogradTest::autogradContainer_ = nullptr;

// 测试用例：TestSendFunctionInvalidInputs
TEST_F(DistAutogradTest, TestSendFunctionInvalidInputs) {
  // 创建张量选项，允许梯度计算
  auto options = at::TensorOptions().requires_grad(true);
  // 创建两个形状为 [3, 3] 的全一张量
  auto in1 = torch::ones({3, 3}, options);
  auto in2 = torch::ones({3, 3}, options);

  // 创建新的自动求导上下文
  autogradContainer_->newContext();
  // 获取当前自动求导上下文
  auto autogradContext = autogradContainer_->currentContext();
  // 将发送自动求导函数附加到张量上
  std::vector<torch::Tensor> tensors = {in1, in2};
  rpc::worker_id_t worker_id = 1;
  addSendRpcBackward(autogradContext, AutogradMetadata(1, 1), tensors);
  // 添加已知的工作节点 ID
  autogradContext->addKnownWorkerId(worker_id);
  // 获取发送函数
  auto send_function = autogradContext->sendFunctions()[1];

  // 确保记录了工作节点 ID
  auto knownWorkerIds = autogradContext->getKnownWorkerIds();
  ASSERT_TRUE(knownWorkerIds.find(worker_id) != knownWorkerIds.end());
  ASSERT_EQ(knownWorkerIds.size(), 1);

  // 预期这会失败，因为 SendRpcBackward 函数不应接收任何梯度作为输入
  EXPECT_THROW(send_function->apply({in1, in2}), c10::Error);

  // 预期这会失败，因为 SendRpcBackward 函数遇到未定义的梯度
  send_function->setGrads({in1, torch::autograd::Variable()});
  EXPECT_THROW(send_function->apply({}), c10::Error);
}

// 测试用例：TestInitializedContextCleanup
TEST_F(DistAutogradTest, TestInitializedContextCleanup) {
  // 创建新的自动求导上下文
  autogradContainer_->newContext();
  // 获取当前上下文的 ID
  auto contextId = autogradContainer_->currentContext()->contextId();
  // 获取分布式引擎实例
  auto& engine = DistEngine::getInstance();
  ASSERT_EQ(0, engine.numBackwardPasses());

  // 构建自动求导图
  auto x = torch::randn({2, 2}, torch::requires_grad());
  auto y = torch::randn({2, 2}, torch::requires_grad());
  auto z = (x * x + y * y).sum();
  ASSERT_NE(nullptr, z.grad_fn());

  // 执行引擎
  engine.execute(contextId, {z}, /* retainGraph */ false);

  // 验证清理是否正确
  ASSERT_EQ(0, engine.numBackwardPasses());
}
TEST_F(DistAutogradTest, TestInitializedContextCleanupSendFunction) {
  // 创建新的自动求导上下文
  autogradContainer_->newContext();
  // 获取当前自动求导上下文
  auto context = autogradContainer_->currentContext();
  // 获取分布式引擎单例
  auto& engine = DistEngine::getInstance();
  // 断言当前回传次数为0
  ASSERT_EQ(0, engine.numBackwardPasses());

  // 添加发送函数
  auto options = at::TensorOptions().requires_grad(true);  // 设置张量选项，启用梯度计算
  auto t = torch::ones({1}, options);  // 创建一个全为1的张量
  auto tensors = std::vector<torch::Tensor>{t};  // 将张量放入向量中
  addSendRpcBackward(
      context, AutogradMetadata(context->contextId(), 0), tensors);  // 向上下文中添加发送 RPC 的反向传播

  auto sendFunction = context->retrieveSendFunction(0);  // 获取上下文中索引为0的发送函数
  sendFunction->setGrads({t});  // 设置发送函数的梯度为张量 t 的梯度

  // 执行发送函数的异步操作
  engine
      .executeSendFunctionAsync(context, sendFunction, /*retrainGraph*/ false)
      ->wait();  // 等待发送函数异步执行完毕

  // 验证是否适当清理
  ASSERT_EQ(0, engine.numBackwardPasses());  // 断言当前回传次数为0
}

} // namespace autograd
} // namespace distributed
} // namespace torch
```