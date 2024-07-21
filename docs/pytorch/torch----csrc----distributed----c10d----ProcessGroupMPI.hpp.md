# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupMPI.hpp`

```py
#pragma once

#ifdef USE_C10D_MPI

#include <condition_variable> // 条件变量，用于多线程同步
#include <deque>              // 双端队列，用于存储任务队列
#include <exception>          // 异常处理
#include <memory>             // 内存管理
#include <mutex>              // 互斥锁，用于保护共享资源
#include <thread>             // 线程管理
#include <vector>             // 向量，用于存储 Tensor 等数据

#include <ATen/core/ivalue.h>      // PyTorch 中的 IValue 类
#include <ATen/core/ivalue_inl.h>  // PyTorch 中 IValue 类的内联实现

#include <torch/csrc/distributed/c10d/Backend.hpp>  // c10d 后端接口
#include <torch/csrc/distributed/c10d/Types.hpp>    // c10d 相关类型定义
#include <torch/csrc/distributed/c10d/Utils.hpp>    // c10d 工具函数

#include <c10/util/CallOnce.h>  // 单次调用的工具函数

#include <mpi.h>  // MPI 头文件

namespace c10d {

constexpr const char* MPI_BACKEND_NAME = "mpi";  // MPI 后端名称常量

// WorkEntry 是与单个 MPI 运行实例相关联的状态。
// 它包括源张量列表、目标张量列表以及操作于源或目标或两者的实际运行函数。
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* srcPtr,
      std::vector<at::Tensor>* dstPtr,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // 不可复制
  WorkEntry(const WorkEntry&) = delete;
  // 不可复制赋值
  WorkEntry& operator=(const WorkEntry&) = delete;

  // 对于输入和输出张量（就地操作），我们总是使用 src
  std::vector<at::Tensor> src;

  // 用户提供输出的副本
  const std::vector<at::Tensor> dst;

  // 仅对于接收操作，返回的源排名
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry>&)> run;  // 运行函数的回调
};

// ProcessGroupMPI 实现了 c10d 的 MPI 绑定。
//
// 期望在组内的所有进程中以相同的顺序调用此类的所有函数。
// 这是保证跨进程匹配相同调用的唯一方法。
//
// 此类提供的所有 MPI 函数都在工作线程上异步调度。
// 因此，ProcessGroupMPI 需要使用具有最低线程支持值 MPI_THREAD_SERIALIZED 的 MPI 实现。
// 这意味着进程可以是多线程的，并且多个线程可以进行 MPI 调用，但一次只能有一个：
// MPI 调用不会同时从两个不同的线程进行（所有 MPI 调用是串行的）。
// 然而，使用 MPI_THREAD_SERIALIZED，ProcessGroupMPI 只支持单个进程组。
// 换句话说，全局不能创建多于 1 个进程组。
//
// 如果希望使用多个 ProcessGroupMPI，则需要您的 MPI 实现具有 MPI_THREAD_MULTIPLE 的线程支持值。
// 这意味着多个线程可以调用 MPI，没有限制。
//
// 注意，ProcessGroupMPI 仅支持单个张量操作。
// 换句话说，输入张量向量的大小应始终为 1。
//
// 如果 MPI 使用了 CUDA-aware MPI，则可以支持 CUDA 张量，并且 ProcessGroupMPI 将自动检测此支持。
class TORCH_API ProcessGroupMPI : public Backend {
 public:
  class WorkMPI : public Work {
   public:
    // 显式构造函数，初始化 WorkMPI 对象
    explicit WorkMPI(
        std::vector<at::Tensor> outputTensors,  // 输出张量的向量
        const char* profilingTitle = nullptr,   // 可选的性能分析标题，默认为空指针
        const std::optional<std::vector<at::Tensor>>& inputTensors =  // 可选的输入张量向量，默认为无值
            c10::nullopt)
        : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),  // 调用基类 Work 的构造函数
          outputTensors_(std::move(outputTensors)),  // 移动赋值输出张量向量
          future_(c10::make_intrusive<at::ivalue::Future>(  // 创建输出结果的 Future 对象
              c10::ListType::create(c10::TensorType::get()))) {}  // 使用 Tensor 类型创建 ListType

    // 返回结果的虚函数重写声明
    std::vector<at::Tensor> result() override;

    // 返回 Future 对象的虚函数重写声明
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    friend class ProcessGroupMPI;  // 声明 ProcessGroupMPI 类为友元类

   private:
    // 完成 MPI 工作的私有方法声明
    void finishWorkMPI();
    // 处理 MPI 工作异常的私有方法声明
    void finishWorkMPIError(const std::exception_ptr& eptr);

    std::vector<at::Tensor> outputTensors_;  // 输出张量向量
    c10::intrusive_ptr<at::ivalue::Future> future_;  // 输出结果的 Future 对象指针
  };

  // AsyncWork 类，继承自 Work 类
  class AsyncWork : public Work {
   public:
    // AsyncWork 构造函数声明
    AsyncWork(
        MPI_Request request,  // MPI 请求对象
        std::vector<at::Tensor> outputTensors,  // 输出张量的向量
        const char* profilingTitle = nullptr,  // 可选的性能分析标题，默认为空指针
        const std::optional<std::vector<at::Tensor>>& inputTensors =  // 可选的输入张量向量，默认为无值
            c10::nullopt);

    // 虚析构函数声明
    ~AsyncWork() override;

    // 检查工作是否已完成的方法声明
    bool isCompleted() override;

    // 检查工作是否成功的常量方法声明
    bool isSuccess() const override;

    // 返回源排名的方法声明
    int sourceRank() const override;

    // 等待工作完成的方法声明
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    // 中止工作的方法声明
    void abort() override;

    // 返回结果的虚函数重写声明
    std::vector<at::Tensor> result() override;

   protected:
    // 填充异常信息的保护方法声明
    void populateException();

   private:
    const std::vector<at::Tensor> outputTensors_;  // 输出张量向量
    MPI_Request request_;  // MPI 请求对象
    MPI_Status status_{};  // MPI 状态对象
  };

  // 显式构造函数声明，初始化 ProcessGroupMPI 对象
  explicit ProcessGroupMPI(int rank, int size, MPI_Comm pgComm);

  // 虚析构函数声明
  ~ProcessGroupMPI() override;

  // 中止 MPI 程序的方法声明，需要在检测到异常时调用
  void abort();

  // 返回后端名称的常量方法声明
  const std::string getBackendName() const override {
};

// 结束 c10d 命名空间的定义

} // namespace c10d

// 结束对 USE_C10D_MPI 宏的条件编译

#endif // USE_C10D_MPI
```