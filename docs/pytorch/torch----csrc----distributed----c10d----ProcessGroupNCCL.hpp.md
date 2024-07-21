# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupNCCL.hpp`

```
#pragma once

#ifdef USE_C10D_NCCL

#if defined(__linux__)
#include <fcntl.h>  // 文件控制操作
#include <sys/stat.h>  // 文件状态
#include <sys/types.h>  // 系统数据类型定义
#include <unistd.h>  // POSIX 操作系统 API
#endif

#include <atomic>  // 原子操作
#include <chrono>  // 时间处理
#include <future>  // 异步执行
#include <iostream>  // 标准输入输出流
#include <list>  // 双向链表
#include <mutex>  // 互斥量
#include <thread>  // 线程支持
#include <unordered_map>  // 哈希表

#include <torch/csrc/distributed/c10d/Backend.hpp>  // 分布式后端接口
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>  // NCCL 工具函数
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>  // 带前缀的存储接口
#include <torch/csrc/distributed/c10d/Store.hpp>  // 分布式存储接口
#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>  // 节点内通信

#include <ATen/DynamicLibrary.h>  // 动态库加载
#include <ATen/cuda/CUDAContext.h>  // CUDA 上下文管理
#include <ATen/cuda/CUDAEvent.h>  // CUDA 事件管理
#include <c10/core/Stream.h>  // 计算流管理
#include <c10/core/StreamGuard.h>  // 流的上下文管理
#include <c10/cuda/CUDACachingAllocator.h>  // CUDA 内存分配器
#include <c10/cuda/CUDAGuard.h>  // CUDA 设备管理
#include <c10/cuda/CUDAStream.h>  // CUDA 流管理

#include <torch/custom_class.h>  // 自定义类支持

namespace c10d {

// 控制是否始终使用高优先级流
static std::vector<std::string> TORCH_NCCL_HIGH_PRIORITY = {
    "TORCH_NCCL_HIGH_PRIORITY"};

// 控制是否 wait() 是阻塞还是非阻塞
static std::vector<std::string> TORCH_NCCL_BLOCKING_WAIT = {
    "TORCH_NCCL_BLOCKING_WAIT",
    "NCCL_BLOCKING_WAIT"};

// 控制是否使用 NCCL 进行异步错误处理
static std::vector<std::string> TORCH_NCCL_ASYNC_ERROR_HANDLING = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING"};

// 控制在看门狗超时时是否启用调试信息转储
// 此变量必须与 TORCH_NCCL_ENABLE_MONITORING=1 和 TORCH_NCCL_TRACE_BUFFER_SIZE > 0 一起设置
static std::vector<std::string> TORCH_NCCL_DUMP_ON_TIMEOUT = {
    "TORCH_NCCL_DUMP_ON_TIMEOUT"};

// 控制是否启用 Desync 调试
// 此变量必须与 TORCH_NCCL_ASYNC_ERROR_HANDLING 一起设置
static std::vector<std::string> TORCH_NCCL_DESYNC_DEBUG = {
    "TORCH_NCCL_DESYNC_DEBUG",
    "NCCL_DESYNC_DEBUG"};

// 启用对所有 ProcessGroupNCCL 集合操作的起始事件记录，并计算每个集合操作的精确时间
// 注意：默认情况下会记录结束事件。打开此标志可能增加由于执行 CUDA 事件查询而导致的看门狗挂起的可能性，最终调用 cudaEventElapsedTime() API。
static std::vector<std::string> TORCH_NCCL_ENABLE_TIMING = {
    "TORCH_NCCL_ENABLE_TIMING",
    "NCCL_ENABLE_TIMING"};

// 启用监视线程，当 ProcessGroupNCCL 看门狗线程在 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 后未检测到心跳时，终止进程
// 可以防止作业因长时间占用集群资源而被卡住
static std::vector<std::string> TORCH_NCCL_ENABLE_MONITORING = {
    "TORCH_NCCL_ENABLE_MONITORING"};

// 控制看门狗心跳超时时间，在此超时后监控线程将终止进程
// 定义一个静态向量，包含名为 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 的字符串
static std::vector<std::string> TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"};

// 定义一个静态向量，包含名为 TORCH_NCCL_RETHROW_CUDA_ERRORS 的字符串
static std::vector<std::string> TORCH_NCCL_RETHROW_CUDA_ERRORS = {
    "TORCH_NCCL_RETHROW_CUDA_ERRORS"};

// 定义一个静态向量，包含名为 TORCH_NCCL_TRACE_BUFFER_SIZE 的字符串
// 控制飞行记录器环形缓冲区中存储的事件最大数量
static std::vector<std::string> TORCH_NCCL_TRACE_BUFFER_SIZE = {
    "TORCH_NCCL_TRACE_BUFFER_SIZE"};

// 定义一个静态向量，包含名为 TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC 的字符串
// 控制在退出之前和抛出超时异常之前，等待转储调试信息的额外时间
static std::vector<std::string> TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// 定义一个静态向量，包含名为 TORCH_NCCL_COORD_CHECK_MILSEC 的字符串
// 控制 watchdog 线程内部检查来自其他排名的协调信号的间隔时间，例如用于转储调试信息
static std::vector<std::string> TORCH_NCCL_COORD_CHECK_MILSEC = {
    "TORCH_NCCL_COORD_CHECK_MILSEC"};

// 定义一个静态向量，包含名为 TORCH_NCCL_NAN_CHECK 的字符串
static std::vector<std::string> TORCH_NCCL_NAN_CHECK = {"TORCH_NCCL_NAN_CHECK"};

// 定义一个常量指针，指向字符串 "nccl"
constexpr const char* NCCL_BACKEND_NAME = "nccl";

// 定义一个常量指针，指向字符串 "exception_dump"
constexpr const char* EXCEPTION_DUMP = "exception_dump";

// 定义一个整数常量，表示工作状态更新周期为 30 秒
constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds

// 定义一个常量，表示默认的 ProcessGroupNCCL 超时时间为 10 分钟
constexpr auto kProcessGroupNCCLDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

// 枚举类型，定义了如何处理异步 NCCL 错误的选项
// NoHandling: 不处理异步 NCCL 错误
// TearDown: 出现错误时终止进程
// CleanUpOnly: 只清理集合而不终止进程
// SkipCleanUp: 仅终止进程，不清理 NCCL 通信器
enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

// 定义一个宏，用于判断是否应该进行清理操作，不处理和跳过清理的情况下返回 false
#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

// 定义一个宏，用于判断是否应该进行终止操作，不处理和仅清理的情况下返回 false
#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

// 定义一个宏，用于打印集合哈希签名的日志信息
#define PRINT_COLLECTIVE_HASH_SIGNATURE(phase, opType, numel, hashValue)      \
  LOG(WARNING) << logPrefix() << "Hash of " << phase << " to NCCL " << opType \
               << " with size " << numel << " is " << hashValue;

// 定义一个静态向量，包含名为 TORCH_NCCL_AVOID_RECORD_STREAMS 的字符串
// 如果设置，ProcessGroupNCCL 不使用 recordStream 调用来确保在用户界面和内部通信流中使用的张量的缓存分配器安全性
static std::vector<std::string> TORCH_NCCL_AVOID_RECORD_STREAMS = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS"};

// 如果设置，ProcessGroupNCCL 在 cuda 缓存分配器上注册 postAlloc 和 preFree 钩子
// 以便每当分配或释放张量时，ProcessGroupNCCL 可以在所有可用的 NCCL 通信器上注册/取消注册张量
// 定义一个静态字符串向量，包含了用于查找环境变量的两个键名
static std::vector<std::string> TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK =
    {"TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK",
     "NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"};

// 如果编译环境为 Linux
#if defined(__linux__)
// 定义一个结构体 DumpPipe，用于创建命名管道以便于调试
struct DumpPipe {
  // 构造函数，根据给定的排名创建命名管道
  DumpPipe(int rank) {
    // 获取环境变量 TORCH_NCCL_DEBUG_INFO_PIPE_FILE 的值作为文件名的前缀
    std::string fileStem =
        getCvarString({"TORCH_NCCL_DEBUG_INFO_PIPE_FILE"}, "");
    // 如果文件名前缀为空或者环境变量 TORCH_NCCL_TRACE_BUFFER_SIZE 的值小于等于 0，则直接返回
    if (fileStem.empty() ||
        getCvarInt({"TORCH_NCCL_TRACE_BUFFER_SIZE"}, 0) <= 0) {
      return;
    }
    // 检查文件名前缀是否为空，如果为空则抛出异常
    TORCH_CHECK(!fileStem.empty(), "TORCH_NCCL_DEBUG_INFO_TEMP_FILE is empty");
    // 拼接出完整的命名管道文件名
    std::string filename = c10::str(fileStem, rank, ".pipe");
    // 检查是否能够成功删除已存在的同名命名管道文件，如果失败且错误码不是 ENOENT，则抛出异常
    TORCH_CHECK(
        unlink(filename.c_str()) != -1 || errno == ENOENT,
        "Error removing existing named pipe ",
        filename);
    // 尝试创建命名管道文件，权限为 0666，如果失败则抛出异常
    TORCH_CHECK(
        mkfifo(filename.c_str(), 0666) != -1,
        "Error creating named pipe ",
        filename);
    // 打开命名管道文件，以只读非阻塞方式打开，如果失败则抛出异常
    fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
    // 输出日志信息，标记命名管道文件已打开，写入数据到此文件可以触发 NCCL 调试信息的转储
    LOG(INFO) << "Pipe file " << filename
              << " has been opened, write to it to trigger NCCL Debug Dump.";
    // 如果打开文件失败则抛出异常
    TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
  }
  // 判断是否应该进行转储调试信息，如果命名管道文件描述符为 -1 则返回 false
  bool shouldDump() {
    if (fd_ == -1) {
      return false;
    }
    char buf[128];
    // 非阻塞读取命名管道文件中的数据，忽略 EINTR 错误，因为稍后会重新轮询
    ssize_t bytesRead = read(fd_, &buf, 128);
    // 如果读取到数据则返回 true，否则返回 false
    return bytesRead > 0;
  }
  // 析构函数，关闭命名管道文件描述符
  ~DumpPipe() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

 private:
  int fd_ = -1; // 命名管道文件描述符初始化为 -1
};
// 如果不是在 Linux 编译环境下，则定义一个空的 DumpPipe 结构体
#else
struct DumpPipe {
  DumpPipe(int rank) {} // 空的构造函数
  // 返回 false，表示不需要进行调试信息的转储
  bool shouldDump() {
    return false;
  }
};
#endif

// ProcessGroupNCCL 类实现了 c10d 的 NCCL 绑定。
//
// 预期该类的所有函数在进程组的所有进程中以相同的顺序调用。
// 这是为了确保在所有进程中匹配相同的调用。
//
// 该类提供的所有 NCCL 函数都是异步函数。具体来说，每个 NCCL 调用都会
// 在一个与当前 CUDA 流不同的单独 CUDA 流上调度。这是为了实现潜在的并发
// 和更好的性能。因此，调用者有责任确保他们的 CUDA 流等待来自该类的
// NCCL 操作。
//
// 可以通过调用以下方法之一来实现这一点：
//
// WorkNCCL::wait() 或 WorkNCCL::synchronize()，两者功能相同且为同义词。
//
// 还请注意，WorkNCCL::finishedGPUExecution() 是一个辅助函数，
// 仅由 ProcessGroupNCCL 提供，用于检查 WorkNCCL 的 NCCL 操作是否
// 在 GPU 上执行完成（不仅仅是调度）。
//
// 使用 NCCL 进程组的示例
//
//   ProcessGroupNCCL pg(store, rank, size);
//   std::shared_ptr<WorkNCCL> work = pg.allreduce(tensors);
//
//   // 到这一点，NCCL 内核已成功排队
//   // 现在，让当前流等待 NCCL 完成，这个函数是
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_API ProcessGroupNCCL : public Backend {
 public:
  class WorkNCCL : public Work, public std::enable_shared_from_this<WorkNCCL> {
   public:
    friend struct WorkInfo;

    // Constructor takes a list of CUDA devices
    // 构造函数，接受 CUDA 设备列表
    WorkNCCL(
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = c10::nullopt,
        bool desyncDebug = false,
        bool enableTiming = false,
        DebugLevel distDebugLevel = DebugLevel::Off);

    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    // 拷贝构造函数，部分复制不包括 outputs_。清理线程监控并移除已完成的任务。
    // 但是在销毁视图张量（outputs_ tensors）时可能会发生死锁，这些张量在自动求导图中。
    WorkNCCL(const WorkNCCL& w);

    ~WorkNCCL() override;

    // Checks if the NCCL kernel has started to execute.
    // 检查 NCCL 核心是否开始执行。
    bool isStarted();

    // Checks if request has completed. In this specific case of NCCL, it checks
    // if the NCCL operation has completed on the GPU in its own NCCL stream.
    // Non-blocking operation.
    // 检查请求是否完成。在这个特定的 NCCL 情况下，检查 NCCL 操作是否在其自己的 NCCL 流中完成了 GPU 上的操作。
    // 非阻塞操作。
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for NCCL work.
    // 等同于为 NCCL 工作调用 synchronize()。
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    // Let current stream wait on the completing of the NCCL work
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    // 让当前流等待 NCCL 工作完成。
    // 在异常情况下抛出。阻塞操作，将等待工作完成。
    void synchronize() override;

    // Synchronize streams by blocking each on the NCCL stream
    // 通过在 NCCL 流上阻塞每个流来同步流。
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    // 辅助函数处理异常（如果需要则抛出异常）。
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the NCCL kernels have finished
    // execution on the GPUs
    // 辅助函数，检查 NCCL 核心是否已经在 GPU 上执行完成。
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    // 获取一个将在内部标记为完成的 Future 对象。
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkNCCL object.
    // 设置异常指针（exception_ptr）在 WorkNCCL 对象上的辅助函数。
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkNCCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkNCCL object.
    // 辅助函数，如果 WorkNCCL 对象超时则返回 True，否则返回 False。
    // 在超时情况下，在 WorkNCCL 对象上设置异常。
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = c10::nullopt);

    std::vector<at::Tensor> result() override;

   protected:
    // The cached list of CUDA devices to operate on
    // 缓存的 CUDA 设备列表用于操作
    at::Device device_;

    // The start CUDA event of NCCL operator tracking this work item. These
    // start CUDA events are needed by desync debugging if enabled.
    // 跟踪此工作项的 NCCL 操作符的开始 CUDA 事件。
    // 如果启用了 desync 调试，则这些开始 CUDA 事件是必需的。
    // 用于跟踪此工作项的 NCCL 操作的开始 CUDA 事件。
    std::shared_ptr<at::cuda::CUDAEvent> ncclStartEvent_;

    // 用于跟踪此工作项的 NCCL 操作的结束 CUDA 事件。
    std::shared_ptr<at::cuda::CUDAEvent> ncclEndEvent_;

    // 此工作项使用的 NCCL 通信器。
    std::shared_ptr<NCCLComm> ncclComm_;

    // 用于 barrier 操作的张量。
    at::Tensor barrierTensor_;

    // 来自 ProcessGroupNCCL 的 blockingWait_ 的克隆。
    bool blockingWait_ = false;

    // 来自 ProcessGroupNCCL 的 avoidRecordStreams_ 的克隆。
    bool avoidRecordStreams_ = false;

    // 来自 ProcessGroupNCCL 的 opTimeout_ 的克隆。
    std::chrono::milliseconds opTimeout_;

    // 表示工作开始的时间点。
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // 记录集体操作的顺序号。
    uint64_t seq_;

    // 表示 nccl 开始事件是否已更新到存储跟踪中。用于 desync 调试。
    bool startTraceUpdated_{false};

    // 用于调试目的记录集体操作的大小。仅记录第一个设备上的大小，因为每进程多设备已不推荐使用。
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

    // 用于静态检查 NCCLErrors 的包装方法，可以为测试覆盖。
    virtual std::exception_ptr checkForNCCLErrors();

    // 友元函数，用于输出 WorkNCCL 的信息。
    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkNCCL& workNCCL);

   private:
    // synchronize 的辅助函数。
    void synchronizeInternal(std::chrono::milliseconds timeout);

    // 检查 NCCL 错误并设置适当的 exception_ptr。
    void checkAndSetException();

    // 仅检查 GPU 执行是否已开始，不修改 exception_ptr。
    bool startedGPUExecutionInternal() const;

    // 仅检查 GPU 执行是否已完成，不修改 exception_ptr。
    bool finishedGPUExecutionInternal() const;

    // 存储的引用，用于将中止的通信器写入存储。
    c10::intrusive_ptr<Store> store_;

    // 存储对 NCCL 集体输出的引用，用于结果和在字符串表示中提供更详细的消息。
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_NCCL_AVOID_RECORD_STREAMS 的实现辅助器。存储参与的非输出张量的引用（即输入、扁平化中间结果）。
    // 在 synchronizeStream 后，我们将清除此列表，即在用户可见的流与 nccl 工作流同步后。
    // 通过保持这些引用（以及 outputs_）直到集体工作重新加入用户可见流之后，我们实现了缓存分配器的安全性，而无需任何 recordStream 调用。
    // 对于原地集体操作，一些存储在此处的引用可能与 outputs_ 别名，但这不会造成任何损害。
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;
    // 由 getFuture 返回的未来对象
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // 是否启用计时
    bool timingEnabled_;

    // 唯一标识用于告知跟踪缓冲区这项工作已完成
    std::optional<uint64_t> trace_id_;

    // 分布式调试级别
    DebugLevel distDebugLevel_;

    // 友元类 ProcessGroupNCCL
    friend class ProcessGroupNCCL;
  };

  struct Options : Backend::Options {
    // 注意: ProcessGroupNCCL::Options 中的超时表示操作的超时时间。
    // 仅当 blockingWait_ 启用时使用。
    explicit Options(bool is_high_priority_stream = false);

    // 返回对象的 intrusive_ptr
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // 在高优先级 CUDA 流上安排 NCCL 操作
    bool is_high_priority_stream;
#ifdef NCCL_HAS_COMM_NONBLOCKING
    // 如果编译时开启了 NCCL 的非阻塞通信支持，则配置通信参数
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#endif

    // 可选的“父”后端和颜色，用于通过 `ncclCommSplit` 创建通信器
    std::shared_ptr<ProcessGroupNCCL> split_from;
    int64_t split_color{0};
    std::vector<uint64_t> global_ranks_in_group;
    std::string group_name;
  };

  // 用于保存进程组最新状态的结构体
  struct ProcessGroupStatus {
    // 最后一个入队的集体操作的序列号，用于标识尚未参与集体操作的进程
    // 初始化为 -1 表示尚未入队任何集体操作
    int64_t lastEnqueuedSeq{-1};
    // 最后一个作为内核启动的集体操作的序列号
    int64_t lastStartedSeq{-1};
    // 最后一个由看门狗线程标记为完成的集体操作的序列号
    // 初始化为 -1 表示尚未完成任何集体操作
    int64_t lastCompletedSeq{-1};

    // 最后一个入队的集体操作的名称
    std::string lastEnqueuedWorkName;
    // 最后一个作为内核启动的集体操作的名称
    std::string lastStartedWorkName;
    // 最后一个完成的集体操作的名称
    std::string lastCompletedWorkName;

    // 最后一个入队操作的输入大小
    size_t lastEnqueuedNumelIn;
    size_t lastEnqueuedNumelOut;
    // 最后一个完成操作的输入大小
    size_t lastCompletedNumelIn;
  // 用于存储上一个已完成的数据尺寸
  size_t lastCompletedNumelOut;
};

// 如果您希望创建多个进程组，每个组可能具有不同的rank和size，可以为每个进程组传递一个新的store实例。
// 如果只有一个store对象，则可以使用`c10d::PrefixStore`来派生作用域实例。
// 这也是torch.distributed中Python API的做法。
//
// 进程组实例保持对store的引用，因为可能会在构造函数运行之后长时间使用。
// 实际上，构造函数不会创建任何NCCL通信器。单个NCCL通信器只能用于特定设备集，并且在运行集体操作时按需创建。
// 如果稍后执行另一个集体操作，针对不同的设备集，进程组将创建另一个NCCL通信器。
// 这些NCCL通信器会被缓存并在可能的情况下重用。
//
ProcessGroupNCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options = Options::create());

// 此构造函数包括已弃用的`groupName`参数。
// 如果您的现有代码使用`groupName`，可以为store指定一个`c10d::PrefixStore(groupName, store)`来替换它。
C10_DEPRECATED ProcessGroupNCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::string& groupName,
    c10::intrusive_ptr<Options> options = Options::create())
    : ProcessGroupNCCL(store, rank, size, options) {}

// 析构函数，用于清理资源
~ProcessGroupNCCL() override;

// 返回进程组的唯一ID
uint64_t getUid() {
  return static_cast<uint64_t>(uid_);
}

// 返回进程组的选项
c10::intrusive_ptr<Options> getOptions() {
  return options_;
}

// 返回后端的名称，此处为NCCL
const std::string getBackendName() const override {
  return std::string(NCCL_BACKEND_NAME);
}

// 指示该进程组是否支持拆分操作
bool supportsSplitting() const override {
};

// 结束命名空间 c10d

// TORCH_API 用于导出该函数至 Torch 库的 API 中，生成 NCCL 通信的追踪信息
// 参数 includeCollectives: 是否包含集体操作的信息
// 参数 includeStackTraces: 是否包含堆栈跟踪信息
// 参数 onlyActive: 是否仅包含活跃的追踪信息
TORCH_API std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive);

// 获取全局的可选函数的可变引用。心跳监视器将使用此函数来转储追踪信息（如果可用）。
// 在 fbcode 中，我们在此处存储一个函数，该函数使用内部工具进行进程跟踪。
TORCH_API std::optional<
    std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper();

// 类似于 get_cpp_trace_dumper，这存储在 torch-python 层定义的函数，
// 使我们能够检查是否可以获取全局解释器锁（GIL），这对于观察 hang 的情况下有所帮助。
typedef bool (*gil_checker_t)();

TORCH_API gil_checker_t& get_gil_checker();
} // namespace c10d

#endif // USE_C10D_NCCL


这段代码是 C++ 的头文件声明部分，包含了一些函数和类型的声明，主要是为了与 Torch 深度学习库的集成和功能扩展。
```