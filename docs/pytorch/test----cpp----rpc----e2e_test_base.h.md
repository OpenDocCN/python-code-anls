# `.\pytorch\test\cpp\rpc\e2e_test_base.h`

```
#include <gtest/gtest.h>  // 包含 Google Test 测试框架头文件

#include <torch/csrc/distributed/autograd/context/container.h>  // 包含分布式自动求导上下文容器头文件
#include <torch/csrc/distributed/autograd/context/context.h>    // 包含分布式自动求导上下文头文件
#include <torch/csrc/distributed/autograd/engine/dist_engine.h> // 包含分布式自动求导引擎头文件
#include <torch/csrc/distributed/autograd/utils.h>              // 包含分布式自动求导工具头文件
#include <torch/csrc/distributed/c10d/TCPStore.hpp>             // 包含 TCPStore 头文件
#include <torch/csrc/distributed/rpc/rref_context.h>            // 包含远程引用上下文头文件
#include <torch/csrc/distributed/rpc/script_call.h>             // 包含脚本调用头文件
#include <torch/csrc/distributed/rpc/script_remote_call.h>      // 包含脚本远程调用头文件
#include <torch/csrc/distributed/rpc/script_resp.h>             // 包含脚本响应头文件
#include <torch/csrc/distributed/rpc/utils.h>                   // 包含RPC工具头文件
#include <torch/csrc/jit/runtime/operator.h>                    // 包含 JIT 运行时操作符头文件

namespace torch {
namespace distributed {
namespace rpc {

using torch::distributed::autograd::DistAutogradContainer;  // 使用分布式自动求导容器命名空间
using torch::distributed::autograd::DistAutogradContext;    // 使用分布式自动求导上下文命名空间

DistAutogradContainer* getDistAutogradContainer();  // 声明获取分布式自动求导容器的函数

class TestE2EBase : public ::testing::Test {
 protected:
  void SetUp() override {
    // 设置分布式自动求导。
    autogradContainer = getDistAutogradContainer();

    // 设置服务器存储。
    c10d::TCPStoreOptions opts{
        /* port */ 0,
        /* isServer */ true,
        numWorkers,
        /* waitWorkers */ true,
        /* timeout */ std::chrono::seconds(10)};
    store = c10::make_intrusive<c10d::TCPStore>(serverAddress, opts);  // 创建 TCPStore 对象

    buildRpcAgent();  // 构建 RPC 代理

    rpcAgentPostProcessing();  // RPC 代理后处理
  }

  void rpcAgentPostProcessing() {
    RpcAgent::setCurrentRpcAgent(rpcAgent);  // 设置当前 RPC 代理
    std::shared_ptr<TypeResolver> typeResolver =
        std::make_shared<TypeResolver>([&](const c10::QualifiedName& qn) {
          // 用于设备映射的字典类型处理。
          auto pos = qn.name().find("Dict");
          if (pos != std::string::npos) {
            return c10::StrongTypePtr(
                nullptr,
                c10::DictType::create(
                    c10::StringType::get(), c10::StringType::get()));
          }
          return c10::StrongTypePtr(
              nullptr, c10::TensorType::create(at::Tensor()));
        });
    rpcAgent->setTypeResolver(typeResolver);  // 设置 RPC 代理的类型解析器
    rpcAgent->start();  // 启动 RPC 代理
  }

  void TearDown() override {
    rpcAgent->join();  // RPC 代理加入
    rpcAgent->shutdown();  // 关闭 RPC 代理
    RpcAgent::setCurrentRpcAgent(nullptr);  // 设置当前 RPC 代理为 nullptr
  }

  c10::intrusive_ptr<OwnerRRef> createRemoteRRef(
      at::Tensor t1,
      at::Tensor t2,
      std::shared_ptr<torch::jit::Operator> op) {
    auto& ctx = RRefContext::getInstance();  // 获取远程引用上下文实例
    auto ownerRRef = ctx.createOwnerRRef(c10::TensorType::create(t1));  // 创建所有者远程引用

    ctx.addSelfAsFork(ownerRRef);  // 防止此所有者远程引用因其他分支而被删除

    ScriptRemoteCall scriptRemoteCall(
        op, {t1, t2, 1}, ownerRRef->rrefId(), ownerRRef->rrefId());  // 创建脚本远程调用

    auto jitFuture = autograd::sendMessageWithAutograd(
        *rpcAgent,
        rpcAgent->getWorkerInfo("worker"),
        std::move(scriptRemoteCall).toMessage(),
        false);  // 发送带自动求导信息的消息

    ownerRRef->registerOwnerCreationFuture(jitFuture);  // 注册所有者远程引用的创建 future

    // 内置操作符不返回 py::object，因此不需要
    // 给 jitFuture 添加回调函数，处理可能已删除的 ownerRRef
    jitFuture->addCallback(
        // Lambda 表达式捕获 ownerRRefId，完成创建 ownerRRef 的回调操作
        [ownerRRefId = ownerRRef->rrefId()](JitFuture& jitFuture) {
          callback::finishCreatingOwnerRRef(jitFuture, ownerRRefId);
        });
    // 返回 ownerRRef
    return ownerRRef;
  }

  // 执行远程的 tensor 相加操作
  at::Tensor remoteAdd(
      // 输入的两个 tensor
      at::Tensor t1,
      at::Tensor t2,
      // 操作符的共享指针
      std::shared_ptr<torch::jit::Operator> op) {
    // 使用指定的操作符创建脚本调用对象，设置 alpha 为 1
    ScriptCall scriptCall(op, {t1, t2, /* alpha */ 1});

    // 发送 RPC 并返回结果
    auto response = autograd::sendMessageWithAutograd(
        // 发送消息的 RPC 代理
        *rpcAgent,
        // 接收消息的 worker 信息
        rpcAgent->getWorkerInfo("worker"),
        // 将脚本调用转换为消息并发送
        std::move(scriptCall).toMessage());
    // 等待 RPC 响应并抛出任何错误
    response->waitAndThrow();

    // 响应消息类型为 FORWARD_AUTOGRAD_RESP
    MessageType messageType = MessageType::FORWARD_AUTOGRAD_RESP;
    // 反序列化响应消息，并转换为 ScriptResp 对象
    auto wrappedResponse = deserializeResponse(
        // 移动响应消息并解包为自定义类 Message
        std::move(*response->value().toCustomClass<Message>()), messageType);
    // 返回 ScriptResp 对象中的 tensor 值
    return static_cast<ScriptResp&>(*wrappedResponse).value().toTensor();
  }

  // 纯虚函数，由派生类实现，用于构建 RPC 代理
  virtual void buildRpcAgent() = 0;

  // 自动梯度上下文的管理类，确保正确释放上下文
  class AutogradContextGuard {
   public:
    explicit AutogradContextGuard()
        : context(DistAutogradContainer::getInstance().newContext()) {}

    // 析构函数，释放 DistAutogradContainer 中的上下文
    ~AutogradContextGuard() {
      DistAutogradContainer::getInstance().releaseContext(context->contextId());
    }

   private:
    std::shared_ptr<DistAutogradContext> context;
  };

  // 运行训练循环的函数
  void runTrainingLoop() {
    // 创建 tensor 的选项，设置 requires_grad 为 true
    auto options = at::TensorOptions().requires_grad(true);
    // 创建两个尺寸为 3x3 的 tensor，并使用 options
    auto t1 = torch::ones({3, 3}, options);
    auto t2 = torch::ones({3, 3}, options);

    // 定义操作符的完整名称
    c10::OperatorName full_name("aten::add", "Tensor");
    // 查找匹配的操作符
    auto matchedOp = torch::jit::findOperatorFor(full_name);
    // 断言确保找到匹配的操作符
    ASSERT_TRUE(matchedOp);

    // 循环执行训练迭代 numIters 次
    for (size_t i = 0; i < numIters; i++) {
      // 创建自动梯度上下文的 guard 对象
      AutogradContextGuard guard;

      // 在一个自动梯度上下文内执行多个 RPC 的前向传播
      auto result = remoteAdd(t1, t2, matchedOp);
      for (size_t j = 0; j < 5; j++) {
        result = remoteAdd(t1, result, matchedOp);
      }

      // 创建远程的 RRef，并获取其值的 tensor
      auto rref = createRemoteRRef(t1, result, matchedOp);
      result = rref->getValue().toTensor();

      // 现在执行反向传播
      autograd::DistEngine::getInstance().execute(
          // 当前上下文的 ID
          DistAutogradContainer::currentContextId(),
          // 计算结果的和作为反向传播的输入
          {torch::sum(result)},
          /* retainGraph */ false);
    }
  }

  // 自动梯度容器的指针
  DistAutogradContainer* autogradContainer;
  // RPC 代理的共享指针
  std::shared_ptr<RpcAgent> rpcAgent;
  // 迭代的次数
  static const size_t numIters;
  // 工作节点的数量
  static const size_t numWorkers;
  // 分布式存储的指针
  c10::intrusive_ptr<c10d::Store> store;
  // 服务器地址的常量指针
  static const char* serverAddress;
};

// 结束命名空间 rpc
} // namespace rpc

// 结束命名空间 distributed
} // namespace distributed

// 结束命名空间 torch
} // namespace torch
```