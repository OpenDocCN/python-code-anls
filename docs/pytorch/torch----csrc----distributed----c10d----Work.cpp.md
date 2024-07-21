# `.\pytorch\torch\csrc\distributed\c10d\Work.cpp`

```py
// 引入 ATen 库中的 ThreadLocalState 头文件
#include <ATen/ThreadLocalState.h>

// 引入 torch 分布式库中的 Work.hpp 头文件
#include <torch/csrc/distributed/c10d/Work.hpp>
// 引入标准库中的实用工具
#include <utility>

// 使用 c10d 命名空间
namespace c10d {

// Work 类的构造函数实现
Work::Work(
    int rank,  // 接收工作相关的排名信息
    OpType opType,  // 工作的操作类型
    const char* profilingTitle,  // 可选的性能分析标题
    const std::optional<std::vector<at::Tensor>>& inputTensors)  // 可选的输入张量
    : rank_(rank), opType_(opType) {  // 初始化成员变量 rank_ 和 opType_

  // 如果性能分析标题不为空
  if (profilingTitle != nullptr) {
    // 创建共享的 RecordFunction 对象，用于记录函数执行
    auto recordingFunction =
        std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);

    // 如果记录功能处于活动状态
    if (recordingFunction->isActive()) {
      // 将记录函数标记为异步事件
      recordingFunction->_setAsync();

      // 如果有输入张量，将它们转换为 IValue 并传递给记录函数，以便性能分析输出包含形状信息
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto& tensor : *inputTensors) {
          inputs.emplace_back(tensor);
        }
      }

      // 在执行之前调用记录函数的 before 方法，开始记录指定的性能分析标题和输入数据
      recordingFunction->before(
          profilingTitle,
          c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));

      // 设置记录函数结束时的回调函数
      std::function<void()> end_handler = [recordingFunction]() {
        recordingFunction->end();
      };
      
      // 使用线程本地状态包装回调函数，确保回调函数执行时能继承线程的状态
      recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

// 返回工作的操作类型
OpType Work::retrieveOpType() const {
  return opType_;
}

// Work 类的析构函数默认实现
Work::~Work() = default;

// 判断工作是否已完成
bool Work::isCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

// 判断工作是否成功完成，即没有异常
bool Work::isSuccess() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !exception_;
}

// 获取工作中可能抛出的异常指针
std::exception_ptr Work::exception() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return exception_;
}

// 获取工作的源排名信息，但在此抛出错误，因为不支持该操作
int Work::sourceRank() const {
  TORCH_CHECK(
      false,
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

// 返回工作的结果张量，但在此抛出错误，因为该方法未实现
std::vector<at::Tensor> Work::result() {
  TORCH_CHECK(false, "result() not implemented.");
}

// 同步方法，但在此为空实现
void Work::synchronize() {}

// 等待工作完成，可以设置超时时间
bool Work::wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);

  // 如果超时时间为 kNoTimeout，则无限等待
  if (timeout == kNoTimeout) {
    // 等待工作完成，没有超时限制
    cv_.wait(lock, [&] { return completed_; });
  } else {
    // 等待工作完成，带有用户指定的超时时间
    cv_.wait_for(lock, timeout, [&] { return completed_; });

    // 如果超时且工作未完成，抛出异常
    if (!completed_) {
      TORCH_CHECK(false, "Operation timed out!");
    }
  }

  // 如果有异常，重新抛出异常指针
  if (exception_) {
    std::rethrow_exception(exception_);
  }

  // 同步方法，用于实现特定的同步操作
  synchronize();

  // 由于中止 API 未实现，总是返回 true
  return true;
}

// 中止工作的方法，但在此抛出错误，因为该方法未实现
void Work::abort() {
  TORCH_CHECK(false, "Work::abort not implemented.");
}

// 获取工作的 Future 对象，但在此抛出错误，因为该方法未实现
c10::intrusive_ptr<c10::ivalue::Future> Work::getFuture() {
  TORCH_CHECK(false, "Work::getFuture not implemented.");
}

} // namespace c10d
void Work::finish(std::exception_ptr exception) {
  // 获取互斥锁，确保线程安全操作
  std::unique_lock<std::mutex> lock(mutex_);
  // 标记任务完成
  completed_ = true;
  // 存储异常指针
  exception_ = std::move(exception);
  // 如果有记录函数结束的回调函数，则调用并清空回调函数
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  // 解锁互斥锁
  lock.unlock();
  // 唤醒所有等待此条件变量的线程
  cv_.notify_all();
}

void Work::finishAndThrow(std::exception_ptr exception) {
  // 获取互斥锁，确保线程安全操作
  std::unique_lock<std::mutex> lock(mutex_);
  // 标记任务完成
  completed_ = true;
  // 存储异常指针
  exception_ = std::move(exception);
  // 如果有记录函数结束的回调函数，则调用并清空回调函数
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  // 如果有异常指针，则重新抛出异常
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

float Work::getDuration() const {
  // 抛出错误，说明此后端不支持获取持续时间
  TORCH_CHECK(false, "This Backend doesn't support getDuration.");
}

uint64_t Work::getSequencenumber() const {
  // 抛出错误，说明此后端不支持获取序列号
  TORCH_CHECK(false, "This Backend doesn't support getSequencenumber.");
}

class FutureWrappingWork : public Work {
 public:
  FutureWrappingWork(c10::intrusive_ptr<c10::ivalue::Future> fut)
      : Work(), _fut(std::move(fut)) {}

  ~FutureWrappingWork() override = default;

  bool isCompleted() override {
    // 返回Future对象是否已完成
    return _fut->completed();
  }

  bool isSuccess() const override {
    // 返回Future对象是否有值
    return _fut->hasValue();
  }

  std::exception_ptr exception() const override {
    // 返回Future对象的异常指针
    return _fut->exception_ptr();
  }

  int sourceRank() const override {
    // 抛出错误，说明此函数未实现
    TORCH_CHECK(false, "FutureWrappingWork::sourceRank() not implemented");
  }

  std::vector<at::Tensor> result() override {
    // 从Future对象中提取张量数据
    return _fut->value().toPyObjectHolder()->extractTensors();
  }

  bool wait(std::chrono::milliseconds timeout) override {
    // FIXME
    // 检查是否超时为kNoTimeout，否则抛出错误
    TORCH_CHECK(
        timeout == kNoTimeout,
        "FutureWrappingWork::wait() with finite timeout not implemented");
    // 等待Future对象完成
    _fut->wait();
    return true;
  }

  void abort() override {
    // 抛出错误，说明此函数未实现
    TORCH_CHECK(false, "FutureWrappingWork::abort() not implemented");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    // 返回Future对象
    return _fut;
  }

 private:
  c10::intrusive_ptr<c10::ivalue::Future> _fut;
};

c10::intrusive_ptr<Work> Work::create_from_future(
    const c10::intrusive_ptr<c10::ivalue::Future>& future) {
  // 创建并返回FutureWrappingWork对象
  return c10::make_intrusive<FutureWrappingWork>(future);
}

} // namespace c10d
```