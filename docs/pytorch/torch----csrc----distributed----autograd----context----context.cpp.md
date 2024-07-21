# `.\pytorch\torch\csrc\distributed\autograd\context\context.cpp`

```py
// 包含分布式自动求导上下文的头文件
#include <torch/csrc/distributed/autograd/context/context.h>

// 包含一些必要的头文件
#include <functional>

// 包含 PyTorch 的 CUDA 和 CPU 设备类型定义
#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>

// 包含自动求导的梯度累积函数定义
#include <torch/csrc/autograd/functions/accumulate_grad.h>

// 命名空间 torch 中的分布式自动求导
namespace torch {
namespace distributed {
namespace autograd {

// 使用 torch::autograd::AccumulateGrad 作为本地类型的别名
using torch::autograd::AccumulateGrad;

// DistAutogradContext 类的构造函数，初始化上下文 ID 和实现
DistAutogradContext::DistAutogradContext(int64_t contextId)
    : contextId_(contextId),
      impl_(c10::impl::VirtualGuardImpl{
          at::hasCUDA() ? c10::DeviceType::CUDA : c10::DeviceType::CPU}) {}

// 返回当前上下文的 ID
int64_t DistAutogradContext::contextId() const {
  return contextId_;
}

// 获取已知的工作节点 ID 集合
std::unordered_set<rpc::worker_id_t> DistAutogradContext::getKnownWorkerIds()
    const {
  std::lock_guard<std::mutex> guard(lock_);
  return knownWorkerIds_;
};

// 添加已知的工作节点 ID 到集合中
void DistAutogradContext::addKnownWorkerId(const rpc::worker_id_t workerId) {
  std::lock_guard<std::mutex> guard(lock_);
  knownWorkerIds_.insert(workerId);
}

// 添加发送反向 RPC 的函数及其自动求导消息 ID
void DistAutogradContext::addSendFunction(
    const std::shared_ptr<SendRpcBackward>& func,
    int64_t autograd_message_id) {
  TORCH_INTERNAL_ASSERT(func != nullptr);

  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      sendAutogradFunctions_.find(autograd_message_id) ==
      sendAutogradFunctions_.end());
  sendAutogradFunctions_.emplace(autograd_message_id, func);
}

// 添加接收反向 RPC 的函数及其自动求导消息 ID
void DistAutogradContext::addRecvFunction(
    std::shared_ptr<RecvRpcBackward>& func,
    int64_t autograd_message_id) {
  TORCH_INTERNAL_ASSERT(func != nullptr);

  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      recvAutogradFunctions_.find(autograd_message_id) ==
      recvAutogradFunctions_.end());
  recvAutogradFunctions_.emplace(autograd_message_id, func);
}

// 返回发送反向 RPC 函数及其自动求导消息 ID 的映射表
std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>>
DistAutogradContext::sendFunctions() const {
  std::lock_guard<std::mutex> guard(lock_);
  return sendAutogradFunctions_;
}

// 返回接收反向 RPC 函数及其自动求导消息 ID 的映射表
std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>>
DistAutogradContext::recvFunctions() const {
  std::lock_guard<std::mutex> guard(lock_);
  return recvAutogradFunctions_;
}

// 累积梯度到变量的旧梯度中，确保变量定义了梯度且需要梯度计算
void DistAutogradContext::accumulateGrad(
    const torch::autograd::Variable& variable,
    const torch::Tensor& grad,
    size_t num_expected_refs) {
  TORCH_INTERNAL_ASSERT(grad.defined());
  TORCH_INTERNAL_ASSERT(variable.requires_grad());

  std::lock_guard<std::mutex> guard(lock_);
  auto it = accumulatedGrads_.find(variable);
  at::Tensor old_grad;
  if (it != accumulatedGrads_.end()) {
    // 在同一变量上累积多个梯度
    old_grad = it->second;
  }
    // 保存旧的梯度值，用于后续的梯度累积
    old_grad = it->value();
  }

  // 使用前向流计算梯度。本地自动求导引擎使用 AccumulateGrad 函数来检索并应用
  // 前向流，在反向计算过程中。在分布式自动求导中，我们直接调用
  // AccumulateGrad::accumulateGrad，并跳过从自动求导函数中恢复 CUDA 流。
  // 因此，在这里手动调用它来确保流是正确的。
  auto forward_stream =
      torch::autograd::impl::grad_accumulator(variable)->stream();
  // 使用 OptionalStreamGuard 确保前向流的正确性
  c10::OptionalStreamGuard stream_guard(forward_stream);

  // 分布式自动求导不支持更高阶梯度。
  AutoGradMode grad_mode(false);

  // TODO: 当我们支持分布式自动求导中的 post_hooks 作为
  // https://github.com/pytorch/pytorch/issues/33482 的一部分时，
  // 在这里需要增加 'num_expected_refs' 的计数。
  AccumulateGrad::accumulateGrad(
      variable,
      old_grad,
      grad,
      num_expected_refs,
      // lambda 表达式用于处理梯度更新
      [this, &variable](at::Tensor&& grad_update) {
        auto device = grad_update.device();
        // 将梯度更新插入到累积梯度中
        accumulatedGrads_.insert(variable, std::move(grad_update));
        // 记录梯度事件
        recordGradEvent(device);
      });
}

std::shared_ptr<torch::autograd::GraphTask> DistAutogradContext::
    retrieveGraphTask() {
  // 加锁，确保线程安全地访问graphTask_
  std::lock_guard<std::mutex> guard(lock_);
  // 断言graphTask_不为空
  TORCH_INTERNAL_ASSERT(graphTask_);
  // 返回当前保存的graphTask_
  return graphTask_;
}

void DistAutogradContext::setGraphTask(
    std::shared_ptr<torch::autograd::GraphTask> graphTask) {
  // 加锁，确保线程安全地设置graphTask_
  std::lock_guard<std::mutex> guard(lock_);
  // 断言graphTask_尚未设置，否则会抛出异常
  TORCH_INTERNAL_ASSERT(
      !graphTask_,
      "Cannot set GraphTask multiple times for the same autograd context");
  // 移动传入的graphTask到成员变量graphTask_
  graphTask_ = std::move(graphTask);
}

void DistAutogradContext::resetGraphTask() {
  // 加锁，确保线程安全地重置graphTask_
  std::lock_guard<std::mutex> guard(lock_);
  // 将graphTask_置为空指针
  graphTask_ = nullptr;
}

void DistAutogradContext::addOutstandingRpc(
    const c10::intrusive_ptr<rpc::JitFuture>& jitFuture) {
  // 给jitFuture添加回调函数，处理异步操作结果
  jitFuture->addCallback([this](rpc::JitFuture& future) {
    if (future.hasError()) {
      // 如果出现错误，通知本地自动求导引擎
      std::unique_lock<std::mutex> lock(lock_);
      if (graphTask_) {
        // 清除异常信号
        graphTask_->set_exception_without_signal(nullptr);
        lock.unlock();
        // 标记异步操作完成，并设置错误信息
        if (!graphTask_->future_completed_.exchange(true)) {
          graphTask_->future_result_->setErrorIfNeeded(future.exception_ptr());
        }
      } else {
        // 如果GraphTask不再有效，记录警告信息
        LOG(WARNING) << "Ignoring error since GraphTask is no longer valid: "
                     << future.tryRetrieveErrorMessage();
      }
    }
  });
  // 加锁，确保线程安全地添加到outStandingRpcs_中
  std::lock_guard<std::mutex> guard(lock_);
  outStandingRpcs_.push_back(jitFuture);
}

void DistAutogradContext::clearOutstandingRpcs() {
  // 加锁，确保线程安全地清空outStandingRpcs_
  std::unique_lock<std::mutex> lock(lock_);
  outStandingRpcs_.clear();
}

void DistAutogradContext::recordGradEvent(c10::Device device) {
  if (device.is_cuda()) {
    // 如果设备是CUDA设备
    auto iter = gradReadyEvents_.find(device);
    if (iter == gradReadyEvents_.end()) {
      // 创建并记录CUDA事件
      c10::Event event(device.type());
      event.record(impl_.getStream(event.device()));
      gradReadyEvents_.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(device),
          std::forward_as_tuple(std::move(event)));
    } else {
      // 更新已存在的CUDA事件
      iter->second.record(impl_.getStream(device));
    }
  }
}

c10::intrusive_ptr<c10::ivalue::Future> DistAutogradContext::
    clearAndWaitForOutstandingRpcsAsync() {
  std::unique_lock<std::mutex> lock(lock_);
  // 移动outStandingRpcs_到局部变量
  auto outStandingRpcs = std::move(outStandingRpcs_);
  lock.unlock();

  // 异步清理和等待未完成的RPC
  struct State {
    explicit State(int32_t count)
        : future(
              c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get())),
          remaining(count) {}
    c10::intrusive_ptr<c10::ivalue::Future> future;
    std::atomic<int32_t> remaining;
    std::atomic<bool> alreadySentError{false};
  };
  auto state = std::make_shared<State>(outStandingRpcs.size());
  if (outStandingRpcs.empty()) {
    // 如果没有未完成的RPC，直接完成Future
    state->future->markCompleted(c10::IValue());
  } else {
    // 遍历所有未完成的远程过程调用（RPC）
    for (auto& rpc : outStandingRpcs) {
      // 为每个RPC添加回调函数，处理异步结果
      rpc->addCallback([state](rpc::JitFuture& future) {
        // 如果异步操作出现错误
        if (future.hasError()) {
          // 如果尚未发送过错误，使用比较并交换（CAS）来保护设置错误状态
          bool expectedAlreadySent = false;
          if (state->alreadySentError.compare_exchange_strong(
                  expectedAlreadySent, true)) {
            // 将异常信息设置到future对象中
            state->future->setError(future.exception_ptr());
          }
          return;
        }

        // 如果没有错误，减少剩余未完成RPC的数量
        if (--state->remaining == 0) {
          // 如果剩余RPC数量为0，标记future对象为已完成状态
          state->future->markCompleted(c10::IValue());
        }
      });
    }
  }
  // 返回状态对象中的future，以便调用者可以等待异步操作完成
  return state->future;
}

// 返回给定 autograd_message_id 对应的发送函数的共享指针
std::shared_ptr<SendRpcBackward> DistAutogradContext::retrieveSendFunction(
    int64_t autograd_message_id) {
  // 加锁以确保线程安全访问 sendAutogradFunctions_
  std::lock_guard<std::mutex> guard(lock_);
  // 查找 autograd_message_id 对应的发送函数
  auto it = sendAutogradFunctions_.find(autograd_message_id);
  // 检查是否找到对应的发送函数，否则抛出异常
  TORCH_CHECK(
      it != sendAutogradFunctions_.end(),
      "Could not find send function for autograd message id: ",
      autograd_message_id);
  // 返回找到的发送函数
  return it->second;
}

// 返回累积梯度 accumulatedGrads_
const c10::Dict<torch::Tensor, torch::Tensor> DistAutogradContext::
    getGradients() const {
  // 加锁以确保线程安全访问 accumulatedGrads_ 和 gradReadyEvents_
  std::lock_guard<std::mutex> guard(lock_);
  // 阻塞当前流，确保在使用梯度之前梯度计算已完成
  for (auto& entry : gradReadyEvents_) {
    auto& event = entry.second;
    event.block(impl_.getStream(event.device()));
  }
  // 返回累积的梯度 accumulatedGrads_
  return accumulatedGrads_;
}

// 为给定变量运行梯度回调函数
void DistAutogradContext::runGradCallbackForVariable(
    const torch::autograd::Variable& variable,
    GradCallback&& cb) {
  torch::Tensor grad;
  {
    std::lock_guard<std::mutex> guard(lock_);
    // 查找变量对应的梯度，确保梯度存在于 accumulatedGrads_ 中
    auto it = accumulatedGrads_.find(variable);
    TORCH_INTERNAL_ASSERT(
        it != accumulatedGrads_.end(),
        "The grad for the variable should exist in dist_autograd context.");
    // 获取变量的梯度
    grad = it->value();
  }
  // 执行梯度回调函数，并根据返回值更新梯度和记录梯度事件
  if (cb(grad)) {
    std::lock_guard<std::mutex> guard(lock_);
    auto device = grad.device();
    // 更新 accumulatedGrads_ 中的梯度
    accumulatedGrads_.insert_or_assign(variable, std::move(grad));
    // 记录梯度事件
    recordGradEvent(device);
  }
}

// 匿名命名空间，用于保存线程本地的上下文指针 tl_context_ptr
namespace {
thread_local ContextPtr tl_context_ptr;
} // namespace

// 构造函数，设置线程本地的 DistAutogradContext 上下文指针
ThreadLocalDistAutogradContext::ThreadLocalDistAutogradContext(
    ContextPtr&& new_context)
    : prev_context_ptr_(std::move(tl_context_ptr)) {
  // 将当前的 tl_context_ptr 移动到 prev_context_ptr_，然后设置新的 tl_context_ptr
  tl_context_ptr = std::move(new_context);
}

// 析构函数，恢复先前的线程本地 DistAutogradContext 上下文指针
ThreadLocalDistAutogradContext::~ThreadLocalDistAutogradContext() {
  tl_context_ptr = std::move(prev_context_ptr_);
}

// 静态方法，获取当前线程的 DistAutogradContext 上下文指针 tl_context_ptr
ContextPtr ThreadLocalDistAutogradContext::getContextPtr() {
  return tl_context_ptr;
}
```