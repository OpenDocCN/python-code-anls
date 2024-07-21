# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupMPI.cpp`

```py
// 包含 Torch 的 MPI 进程组通信头文件
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>

#ifdef USE_C10D_MPI

// 包含标准输入输出流和映射容器的头文件
#include <iostream>
#include <map>

// 包含设备守卫和整数范围工具的头文件
#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>

// 如果使用 Open MPI，需要包含用于 CUDA-aware 检查的头文件
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h> // Needed for CUDA-aware check
#endif

namespace c10d {

// 定义一个宏，用于检查 MPI 调用的返回状态并在失败时抛出错误
#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      TORCH_CHECK(false, err);                                           \
    }                                                                    \
  } while (0)

namespace {

// MPI 运算类型到 MPI_Op 对象的映射
std::map<ReduceOp::RedOpType, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};

// Torch 标量类型到 MPI_Datatype 的映射
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

// 检查 CUDA-aware MPI 支持情况的函数
bool cudaAwareMpiCheck() {
  // 运行时检查是否支持 CUDA-aware MPI
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return false;
  }
#else // !defined(MPIX_CUDA_AWARE_SUPPORT)
  return false;
#endif // MPIX_CUDA_AWARE_SUPPORT
}

// 检查输入张量的有效性
void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
  if (tensor.is_cuda() && !cudaAwareMpiCheck()) {
    TORCH_CHECK(
        false,
        "CUDA tensor detected and the MPI used doesn't "
        "have CUDA-aware MPI support");
  }
}

// 检查单个张量的函数，对传入的张量向量进行检查
void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(
        false, "MPI process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

// 检查输入张量与首张张量的大小和类型是否相同
void checkSameSizeAndType(
    const at::Tensor& t_in,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.scalar_type() != t_in.scalar_type())) {
      TORCH_CHECK(false, "Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

} // namespace

// 返回 MPI 进程组 MPI 工作对象的结果张量向量
std::vector<at::Tensor> ProcessGroupMPI::WorkMPI::result() {
  return outputTensors_;
}
}

// 返回与此工作相关联的 Future 对象
c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupMPI::WorkMPI::getFuture() {
  return future_;
}

// 将异常指针设置到 Future 对象中，并完成工作
void ProcessGroupMPI::WorkMPI::finishWorkMPIError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

// 将输出张量标记为已完成，并完成工作
void ProcessGroupMPI::WorkMPI::finishWorkMPI() {
  future_->markCompleted(at::IValue(outputTensors_));
  finish();
}

// 异步工作的构造函数，初始化成员变量和状态
ProcessGroupMPI::AsyncWork::AsyncWork(
    MPI_Request request,
    std::vector<at::Tensor> outputTensors,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
      outputTensors_(std::move(outputTensors)),
      request_(request) {
  memset(&status_, 0, sizeof(status_));
}

// 异步工作的析构函数，用于检查工作是否已完成并且请求是否为 MPI_REQUEST_NULL
ProcessGroupMPI::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << '\n';
    std::terminate();
  }
}

// 检查异步工作是否已完成
bool ProcessGroupMPI::AsyncWork::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  // request_ == MPI_REQUEST_NULL; 工作已完成
  // 如果请求不成功，填充异常
  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }

  return true;
}

// 检查异步工作是否成功完成
bool ProcessGroupMPI::AsyncWork::isSuccess() const {
  if (request_ != MPI_REQUEST_NULL) {
    TORCH_CHECK(
        false,
        "Invalid call to AsyncWork::isSuccess before work has completed");
  }

  return status_.MPI_ERROR == MPI_SUCCESS;
}

// 返回源进程的排名
int ProcessGroupMPI::AsyncWork::sourceRank() const {
  return status_.MPI_SOURCE;
}

// 等待异步工作完成，处理 MPI 请求状态和异常
bool ProcessGroupMPI::AsyncWork::wait(std::chrono::milliseconds /* unused */) {
  if (request_ == MPI_REQUEST_NULL) {
    // 如果 AsyncWork 没有调用 ProcessGroup::finish()，需要手动调用性能分析结束回调函数
    if (Work::recordFunctionEndCallback_) {
      Work::recordFunctionEndCallback_();
      Work::recordFunctionEndCallback_ = nullptr;
    }
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);

  // 如果 AsyncWork 没有调用 ProcessGroup::finish()，需要手动调用性能分析结束回调函数
  if (Work::recordFunctionEndCallback_) {
    Work::recordFunctionEndCallback_();
    Work::recordFunctionEndCallback_ = nullptr;
  }

  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }
  // 由于中止 API 未实现，总是返回 true
  return true;
}

// 中止异步工作，抛出错误表明未实现该功能
void ProcessGroupMPI::AsyncWork::abort(){
    TORCH_CHECK(false, "ProcessGroupMPI::AsyncWork::abort not implemented.")}
// 返回异步工作结果的输出张量向量
std::vector<at::Tensor> ProcessGroupMPI::AsyncWork::result() {
  return outputTensors_;
}

// 根据 MPI 错误状态填充异常信息
void ProcessGroupMPI::AsyncWork::populateException() {
  // 用于存储 MPI 错误信息的缓冲区
  std::array<char, MPI_MAX_ERROR_STRING> buf{};
  int len = buf.size();
  // 将 MPI 错误码转换为字符串描述并存储在 buf 中
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  // 创建指向运行时错误的异常指针，使用 MPI 错误字符串初始化
  exception_ = std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

// 静态全局状态
int ProcessGroupMPI::mpiThreadSupport_ = 0;
std::mutex ProcessGroupMPI::pgGlobalMutex_;
// 仅希望初始化一次
c10::once_flag ProcessGroupMPI::onceFlagInitMPI;

// MPI 环境退出函数
void ProcessGroupMPI::mpiExit() {
  // 获取全局互斥锁
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  // 调用 MPI_Finalize 完成 MPI 释放资源
  MPI_CHECK(MPI_Finalize());
}

// 初始化 MPI 环境，仅执行一次
void ProcessGroupMPI::initMPIOnce() {
  // 调用 c10::call_once 以确保只初始化一次
  c10::call_once(onceFlagInitMPI, []() {
    int mpi_was_initialized = 0;
    // 检查 MPI 是否已经初始化
    MPI_CHECK(MPI_Initialized(&mpi_was_initialized));
    if (mpi_was_initialized == 0) {
      // 如果 MPI 尚未初始化，则执行 MPI_Init_thread 初始化 MPI 环境
      MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpiThreadSupport_));
      // 检查 MPI 支持的线程级别是否符合要求
      if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
        TORCH_CHECK(
            false,
            "Used MPI implementation doesn't have the "
            "minimum level of threading support: "
            "MPI_THREAD_SERIALIZED. This is required by "
            "c10d package");
      }
      // 注册 MPI 退出处理函数
      if (std::atexit(ProcessGroupMPI::mpiExit)) {
        TORCH_CHECK(false, "Fail to register the MPI exit handler");
      }
    } else {
      // 如果 MPI 已经初始化过，发出警告
      TORCH_WARN_ONCE("MPI was previously initialized.");
    }
  });
}

// 创建 ProcessGroupMPI 实例，传入要通信的进程的排名列表
c10::intrusive_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI(
    std::vector<int> ranks) {
  // 执行 MPI 初始化
  initMPIOnce();

  // 创建 MPI 通信组
  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;

  {
    // 获取全局互斥锁
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);

    // 如果未指定任何排名，假定创建根组
    if (!ranks.empty()) {
      MPI_Group worldGroup{};
      MPI_Group ranksGroup{};
      // 获取 MPI_COMM_WORLD 的通信组
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      // 包含指定排名的新通信组
      MPI_CHECK(MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));
      // 尝试多次创建 MPI 通信组以解决可能的问题
      constexpr int kMaxNumRetries = 3;
      bool groupComm_updated = false;
      // 同步所有 MPI 进程以确保顺利创建通信组
      MPI_Barrier(MPI_COMM_WORLD);
      for (const auto i : c10::irange(kMaxNumRetries)) {
        (void)i;  // 防止编译器未使用警告
        // 尝试创建 MPI 通信组
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
          groupComm_updated = true;
          break;
        }
      }
      MPI_CHECK(groupComm_updated);
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    // 获取当前进程在指定通信组中的排名和组大小
    // 如果没有创建新组，则默认使用 MPI_COMM_WORLD
    // 检查是否存在有效的通信组
    if (groupComm != MPI_COMM_NULL) {
      // 获取当前进程在通信组中的排名
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      // 获取通信组中的进程总数
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      // 如果排名或者进程总数小于零，抛出错误信息
      if (rank < 0 || size < 0) {
        TORCH_CHECK(false, "Failed to get the world_size / rank");
      }
    }
  }

  // 如果当前进程不属于任何通信组，返回一个空的 MPI 通信组实例指针
  // 这与其他进程组类型的语义一致。
  if (groupComm == MPI_COMM_NULL) {
    return c10::intrusive_ptr<ProcessGroupMPI>();
  }

  // 创建一个新的 ProcessGroupMPI 实例，包含当前进程的排名、进程总数和通信组
  return c10::make_intrusive<ProcessGroupMPI>(rank, size, groupComm);
// ProcessGroupMPI 类的构造函数，继承自 Backend 类，初始化 stop_ 为 false，pgComm_ 为传入的 MPI 通信对象
ProcessGroupMPI::ProcessGroupMPI(int rank, int size, MPI_Comm pgComm)
    : Backend(rank, size), stop_(false), pgComm_(pgComm) {
  // 如果 pgComm_ 是 MPI_COMM_NULL，抛出错误信息
  if (pgComm_ == MPI_COMM_NULL) {
    TORCH_CHECK(false, "pgComm_ must not be MPI_COMM_NULL");
  }

  // 启动工作线程，该线程用于处理 MPI 调用
  workerThread_ = std::thread(&ProcessGroupMPI::runLoop, this);

  // 调用初始化函数
  init();
}

// ProcessGroupMPI 类的析构函数，调用 destroy 函数
ProcessGroupMPI::~ProcessGroupMPI() {
  destroy();
}

// 销毁函数，等待队列为空后停止工作线程
void ProcessGroupMPI::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  // 等待队列消费条件满足，即队列为空
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  // 队列为空时，设置 stop_ 为 true
  stop_ = true;

  // 解锁以允许线程终止
  lock.unlock();
  // 通知所有等待的生产条件变量，以使队列生产者条件得以发生变化
  queueProduceCV_.notify_all();

  // 等待工作线程结束
  workerThread_.join();
}

// 中止函数，调用 destroy 函数后调用 MPI_Abort 终止 MPI 进程组
void ProcessGroupMPI::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

// 工作循环函数，处理队列中的工作条目
void ProcessGroupMPI::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);

  // 当 stop_ 为 false 时循环执行
  while (!stop_) {
    // 如果队列为空，等待生产条件变量通知
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    // 取出队列中的工作条目
    auto workTuple = std::move(queue_.front());
    queue_.pop_front();

    // 获取工作条目和工作对象的引用
    auto& workEntry = std::get<0>(workTuple);
    auto& work = std::get<1>(workTuple);

    // 解锁以允许其他线程继续生产
    lock.unlock();
    // 通知消费条件变量，工作条目已准备就绪
    queueConsumeCV_.notify_one();

    try {
      // 执行工作条目的运行函数和 MPI 相关操作
      workEntry->run(workEntry);
      work->finishWorkMPI();
    } catch (...) {
      // 如果出现异常，调用工作对象的异常处理函数
      work->finishWorkMPIError(std::current_exception());
    }

    // 再次加锁以处理下一个工作条目
    lock.lock();
  }
}

// 将工作条目加入队列并返回工作对象的智能指针
c10::intrusive_ptr<Work> ProcessGroupMPI::enqueue(
    std::unique_ptr<WorkEntry> entry,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  // 创建 WorkMPI 对象
  auto work =
      c10::make_intrusive<WorkMPI>(entry->dst, profilingTitle, inputTensors);
  std::unique_lock<std::mutex> lock(pgMutex_);
  // 将工作条目和工作对象组成的元组加入队列
  queue_.emplace_back(std::move(entry), work);
  lock.unlock();
  // 通知生产条件变量，队列中有新的工作条目
  queueProduceCV_.notify_one();
  return work;
}

// 广播操作，将张量广播给所有进程
c10::intrusive_ptr<Work> ProcessGroupMPI::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  // 检查输入张量的数量
  checkSingleTensor(tensors);
  // 创建运行函数，执行 MPI_Bcast 函数进行广播
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };
  // 创建工作条目并加入队列
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:broadcast",
      std::optional<std::vector<at::Tensor>>(tensors));
}
    const AllreduceOptions& opts) {
  // 检查是否只包含单个张量，若不是则抛出异常
  checkSingleTensor(tensors);

  // 定义一个函数对象 runFunc，用于执行 MPI Allreduce 操作
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        // 获取 WorkEntry 中的第一个源张量
        auto data = (entry->src)[0];
        // 将数据的设备设置为当前设备
        c10::DeviceGuard guard(data.device());
        // 获取全局互斥锁，保护 MPI 操作
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        // 执行 MPI Allreduce 操作，将数据从所有进程中的每个数据块合并到 data 中
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,                           // 使用原地操作
            data.data_ptr(),                        // 数据指针
            data.numel(),                           // 数据元素数量
            mpiDatatype.at(data.scalar_type()),     // MPI 数据类型
            mpiOp.at(opts.reduceOp),                // MPI 操作类型
            pgComm_));                              // 进程组通信对象
      };

  // 创建一个指向 WorkEntry 对象的独占指针 entry，包含了 tensors 和 runFunc
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));

  // 将 WorkEntry 对象提交到队列中执行，并返回提交操作的结果
  return enqueue(
      std::move(entry),
      "mpi:all_reduce",                           // 提交操作标识
      std::optional<std::vector<at::Tensor>>(tensors));  // 可选的张量向量参数
}
}

`
# 异常情况：MPI 不支持 coalesced allreduce 操作时抛出异常
c10::intrusive_ptr<Work> ProcessGroupMPI::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(false, "allreduce_coalesced is currently not supported with MPI");
}

# 执行 reduce 操作，确保仅有一个张量被处理
c10::intrusive_ptr<Work> ProcessGroupMPI::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  checkSingleTensor(tensors);

  # 定义运行函数，用于执行 MPI 的 reduce 操作
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        auto dataPtr = (entry->src)[0].data_ptr();
        void* sendbuf = (rank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (rank_ == opts.rootRank) ? dataPtr : nullptr;

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        
        # 执行 MPI_Reduce 操作，将数据从各个进程归约到根进程
        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            mpiOp.at(opts.reduceOp),
            opts.rootRank,
            pgComm_));
      };
  
  # 创建 WorkEntry，并将运行函数与之关联
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  
  # 将任务加入队列，并返回结果
  return enqueue(
      std::move(entry),
      "mpi:reduce",
      std::optional<std::vector<at::Tensor>>(tensors));
}

# 执行 allgather 操作，将所有进程的数据聚集到所有进程的输出张量中
c10::intrusive_ptr<Work> ProcessGroupMPI::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  checkSingleTensor(inputTensors);

  # 确保输出张量的数量为1，即只支持单个张量操作
  if (outputTensors.size() != 1) {
    TORCH_CHECK(
        false,
        "MPI process group only supports a single "
        "tensor op");
  }

  # 确保输出张量的数量与进程数相同
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    TORCH_CHECK(
        false,
        "All gather: number of output tensors should equal "
        "to the world size");
  }

  # 检查输入张量和输出张量的大小和类型是否相同
  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  # 定义运行函数，用于执行 MPI 的 allgather 操作
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::vector<at::Tensor> outputDataVec = entry->dst;
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

        # 执行 MPI_Allgather 操作，将数据从每个进程的输入张量复制到所有进程的输出张量
        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            pgComm_));

        # 将平坦的输出张量数据复制回输出张量向量中的各个张量
        for (const auto i : c10::irange(outputDataVec.size())) {
          outputDataVec[i].copy_(flatOutputTensor[static_cast<int64_t>(i)]);
        }
      };
  
  # 创建 WorkEntry，并将运行函数与之关联
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors[0], std::move(runFunc));
  
  # 将任务加入队列，并返回结果
  return enqueue(
      std::move(entry),
      "mpi:all_gather",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}
c10::intrusive_ptr<Work> ProcessGroupMPI::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  TORCH_CHECK(false, "ProcessGroupMPI does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupMPI::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  // 检查输入的张量是否只有一个
  checkSingleTensor(inputTensors);

  if (rank_ != opts.rootRank) {
    // 如果当前进程不是根进程，输出张量应为空，否则报错
    if (!outputTensors.empty()) {
      TORCH_CHECK(
          false,
          "Gather: number of output tensors should be 0 "
          "for non-root");
    }
  } else {
    // 如果当前进程是根进程
    // 检查输出张量是否只有一个，否则报错
    if (outputTensors.size() != 1) {
      TORCH_CHECK(false, "Gather: multi-GPU collective is not supported");
    }
    // 检查输出张量的数量是否等于进程数，否则报错
    if (static_cast<size_t>(size_) != outputTensors[0].size()) {
      TORCH_CHECK(
          false,
          "Gather: number of output tensors should equal "
          "to the world size");
    }
    // 检查输入张量和输出张量的大小和数据类型是否匹配
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  // 定义运行函数，用于执行 MPI 的 Gather 操作
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        std::vector<at::Tensor> dstdata = entry->dst;
        // 如果当前进程是根进程，创建一个与输出张量相同形状的新张量
        if (rank_ == opts.rootRank) {
          flatOutputTensor = newLikeFlat(dstdata);
          recvbuf = flatOutputTensor.data_ptr();
        }

        // 切换当前数据张量的设备环境为数据所在的设备
        c10::DeviceGuard guard(data.device());
        // 获取全局锁，确保 MPI 操作的原子性
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        // 执行 MPI 的 Gather 操作，将数据收集到根进程的 recvbuf 中
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));

        // 如果当前进程是根进程
        if (rank_ == opts.rootRank) {
          const std::vector<at::Tensor>& outputDataVec = entry->dst;
          // 将扁平化的输出张量复制回输出张量的各个部分
          for (const auto i : c10::irange(outputDataVec.size())) {
            outputDataVec.at(i).copy_(
                flatOutputTensor[static_cast<int64_t>(i)]);
          }
        }
      };

  if (rank_ == opts.rootRank) {
    // 如果当前进程是根进程，创建一个 WorkEntry 对象并将其排队
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors[0], std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:gather",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    // 如果当前进程不是根进程，创建一个 WorkEntry 对象并将其排队
    auto entry =
        std::make_unique<WorkEntry>(&inputTensors, nullptr, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:gather",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

// 下面的 scatter 函数以及后续部分没有提供，暂无需进一步注释。
    const ScatterOptions& opts) {

# 接受一个常量引用参数 `opts`，用于配置散射操作的选项。

  checkSingleTensor(outputTensors);

# 调用函数 `checkSingleTensor`，验证输出张量 `outputTensors` 中只包含单个张量。
  
  if (rank_ != opts.rootRank) {

# 如果当前进程的排名 `rank_` 不等于 `opts` 中指定的根排名 `rootRank`，执行以下条件块。

    if (!inputTensors.empty()) {

# 如果输入张量列表 `inputTensors` 不为空，执行以下条件块。

      TORCH_CHECK(
          false,
          "Scatter: number of input tensors should be 0 "
          "for non-root");

# 抛出 Torch 错误，指出非根进程时输入张量数应为0。

    }
  } else {

# 如果当前进程的排名 `rank_` 等于 `opts` 中指定的根排名 `rootRank`，执行以下条件块。

    if (inputTensors.size() != 1) {

# 如果输入张量列表 `inputTensors` 的大小不为1，抛出 Torch 错误，指出不支持多GPU集合操作。
    
      TORCH_CHECK(false, "Scatter: multi-GPU collective is not supported");
    }
    
    if (static_cast<size_t>(size_) != inputTensors[0].size()) {

# 如果当前进程的排名 `rank_` 等于根排名 `rootRank` 且 `inputTensors` 中的第一个张量的大小不等于 `size_`，抛出 Torch 错误，指出输入张量数应与世界大小相等。

      TORCH_CHECK(
          false,
          "Scatter: number of input tensors should equal "
          "to the world size");
    }
    
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);

# 调用函数 `checkSameSizeAndType`，验证 `outputTensors[0]` 和 `inputTensors[0]` 的大小和类型相同。

  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {

# 创建一个函数对象 `runFunc`，接受一个 `std::unique_ptr<WorkEntry>&` 类型参数 `entry`，捕获 `opts` 和当前对象的引用。

        auto data = (entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;

# 声明变量 `data` 表示 `entry->dst` 中的第一个元素，初始化为 `nullptr` 的 `sendbuf`，和未初始化的 `flatInputTensor`。

        if (rank_ == opts.rootRank) {

# 如果当前进程的排名 `rank_` 等于 `opts` 中指定的根排名 `rootRank`，执行以下条件块。

          std::vector<at::Tensor>& inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

# 声明 `inputDataVec` 为 `entry->src` 的引用，`flatInputTensor` 为 `newLikeFlat(inputDataVec)` 返回的张量，将 `sendbuf` 设置为 `flatInputTensor` 的数据指针。

          // copy the input tensors to the flatten large send buffer
          for (const auto i : c10::irange(inputDataVec.size())) {
            flatInputTensor[static_cast<int64_t>(i)].copy_(inputDataVec.at(i));
          }

# 将输入张量复制到扁平化的大发送缓冲区 `flatInputTensor` 中。

        }

        c10::DeviceGuard guard(data.device());

# 在 `data.device()` 上设置设备守卫 `c10::DeviceGuard guard`。

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

# 创建全局互斥量的独占锁 `std::unique_lock<std::mutex> globalLock`。

        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));

# 使用 MPI 执行分散操作，从 `sendbuf` 分散数据到 `data.data_ptr()`，并在 `opts.rootRank` 进程中使用通信器 `pgComm_`。

      };

  if (rank_ == opts.rootRank) {

# 如果当前进程的排名 `rank_` 等于 `opts` 中指定的根排名 `rootRank`，执行以下条件块。

    auto entry = std::make_unique<WorkEntry>(
        &inputTensors[0], &outputTensors, std::move(runFunc));

# 创建一个指向 `WorkEntry` 对象的唯一指针 `entry`，初始化为 `inputTensors[0]` 和 `outputTensors` 的地址，并移动 `runFunc` 到 `entry` 中。

    return enqueue(
        std::move(entry),
        "mpi:scatter",
        !inputTensors.empty()
            ? std::optional<std::vector<at::Tensor>>(inputTensors[0])
            : c10::nullopt);

# 调用 `enqueue` 函数，将 `entry`、字符串 `"mpi:scatter"` 和 `inputTensors` 的第一个元素（如果非空）或空的 `std::optional` 作为参数。

  } else {

# 如果当前进程的排名 `rank_` 不等于 `opts` 中指定的根排名 `rootRank`，执行以下条件块。

    auto entry = std::make_unique<WorkEntry>(
        nullptr, &outputTensors, std::move(runFunc));

# 创建一个指向 `WorkEntry` 对象的唯一指针 `entry`，初始化为 `nullptr` 和 `outputTensors` 的地址，并移动 `runFunc` 到 `entry` 中。

    return enqueue(
        std::move(entry),
        "mpi:scatter",
        !inputTensors.empty()
            ? std::optional<std::vector<at::Tensor>>(inputTensors[0])
            : c10::nullopt);

# 调用 `enqueue` 函数，将 `entry`、字符串 `"mpi:scatter"` 和 `inputTensors` 的第一个元素（如果非空）或空的 `std::optional` 作为参数。

  }
}
}

c10::intrusive_ptr<Work> ProcessGroupMPI::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  // 抛出错误，因为 ProcessGroupMPI 不支持 reduce_scatter 操作
  TORCH_CHECK(false, "ProcessGroupMPI does not support reduce_scatter");
}

c10::intrusive_ptr<Work> ProcessGroupMPI::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  // 检查输入和输出张量是否是单个张量
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);

  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    // 可以使用 MPI_Alltoall 进行通信
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.type() == inputTensor.type(),
        "Tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "Tensor's dim 0 does not divide equally across group size");

    // 定义运行函数，用于执行 MPI_Alltoall 操作
    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [this](std::unique_ptr<WorkEntry>& entry) {
          auto srcdata = (entry->src)[0];  // 获取源数据张量
          auto dstdata = (entry->dst)[0];  // 获取目标数据张量
          c10::DeviceGuard guard(srcdata.device());  // 设备保护，确保在正确的设备上执行操作
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);  // 全局锁，保护全局资源
          // 执行 MPI_Alltoall 操作，将数据从每个进程的发送缓冲区发送到每个进程的接收缓冲区
          MPI_CHECK(MPI_Alltoall(
              srcdata.data_ptr(),
              srcdata.numel() / size_,
              mpiDatatype.at(srcdata.scalar_type()),
              dstdata.data_ptr(),
              dstdata.numel() / size_,
              mpiDatatype.at(dstdata.scalar_type()),
              pgComm_));
        };
    
    // 将输入和输出张量打包成 vector
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    // 创建 WorkEntry 对象，封装运行函数和相关的输入输出张量
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));

    // 将 WorkEntry 加入到队列中，返回对应的 Work 指针
    return enqueue(
        std::move(entry),
        "mpi:all_to_all",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    // 需要使用 MPI_Alltoallv 进行通信

    // 检查分割大小是否合法
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    // 定义一个函数对象 runFunc，接收一个指向 WorkEntry 对象的独占指针作为参数
    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [this, inputSplitSizes, outputSplitSizes](
            std::unique_ptr<WorkEntry>& entry) {
          // 从 WorkEntry 中获取源数据的第一个张量
          auto srcdata = (entry->src)[0];
          // 从 WorkEntry 中获取目标数据的第一个张量
          auto dstdata = (entry->dst)[0];
          // 创建存储发送和接收长度的向量，大小为通信组的大小
          std::vector<int> send_lengths(size_);
          std::vector<int> recv_lengths(size_);
          // 创建存储发送和接收偏移量的向量，大小为通信组的大小
          std::vector<int> send_offsets(size_);
          std::vector<int> recv_offsets(size_);
          // 调用 c10d::computeLengthsAndOffsets 计算源数据的长度和偏移量
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, srcdata, &send_lengths, &send_offsets);
          // 调用 c10d::computeLengthsAndOffsets 计算目标数据的长度和偏移量
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, dstdata, &recv_lengths, &recv_offsets);
          // 切换到 srcdata 所在设备的设备守护器
          c10::DeviceGuard guard(srcdata.device());
          // 获取全局互斥锁 pgGlobalMutex_ 的独占锁
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          // 使用 MPI_Alltoallv 函数进行所有节点间的数据交换
          MPI_CHECK(MPI_Alltoallv(
              srcdata.data_ptr(),                    // 发送数据的起始地址
              send_lengths.data(),                   // 每个进程发送的数据长度
              send_offsets.data(),                   // 发送数据的偏移量
              mpiDatatype.at(srcdata.scalar_type()), // 发送数据的 MPI 数据类型
              dstdata.data_ptr(),                    // 接收数据的起始地址
              recv_lengths.data(),                   // 每个进程接收的数据长度
              recv_offsets.data(),                   // 接收数据的偏移量
              mpiDatatype.at(dstdata.scalar_type()), // 接收数据的 MPI 数据类型
              pgComm_));                            // 通信组 MPI 通信器
        };
    // 创建包含 inputTensor 的向量 inputTensors
    std::vector<at::Tensor> inputTensors = {inputTensor};
    // 创建包含 outputTensor 的向量 outputTensors
    std::vector<at::Tensor> outputTensors = {outputTensor};
    // 创建一个指向 WorkEntry 对象的独占指针 entry，初始化它使用 inputTensors、outputTensors 和 runFunc
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    // 调用 enqueue 函数，将 entry 入队，传递额外的参数 "mpi:all_to_all" 和 inputTensors 的可选向量
    return enqueue(
        std::move(entry),
        "mpi:all_to_all",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

c10::intrusive_ptr<Work> ProcessGroupMPI::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  // 检查输入和输出张量的数量是否等于进程组的大小
  TORCH_CHECK(
      inputTensors.size() == static_cast<size_t>(size_),
      "Number of input tensors are not equal to group size");
  TORCH_CHECK(
      outputTensors.size() == static_cast<size_t>(size_),
      "Number of output tensors are not equal to group size");
  // 定义一个 lambda 函数，该函数实现 alltoall 操作
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        // 初始化发送和接收长度、偏移量的向量
        std::vector<int> send_lengths(size_);
        std::vector<int> recv_lengths(size_);
        std::vector<int> send_offsets(size_);
        std::vector<int> recv_offsets(size_);
        // 获取输入和输出数据的引用
        auto srcdata = entry->src;
        auto dstdata = entry->dst;
        // 计算发送数据和接收数据的长度及偏移量
        auto src_len = c10d::computeLengthsAndOffsets(
            srcdata, &send_lengths, &send_offsets);
        auto dst_len = c10d::computeLengthsAndOffsets(
            dstdata, &recv_lengths, &recv_offsets);
        // 将发送长度和接收长度转换为 int64_t 类型的向量
        std::vector<int64_t> send_lengthsL(
            send_lengths.begin(), send_lengths.end());
        std::vector<int64_t> recv_lengthsL(
            recv_lengths.begin(), recv_lengths.end());
        // 创建用于扁平化发送和接收数据的张量
        at::Tensor srcFlatData =
            at::empty({static_cast<int64_t>(src_len)}, srcdata[0].options());
        at::Tensor dstFlatData =
            at::empty({static_cast<int64_t>(dst_len)}, dstdata[0].options());
        // 将源数据扁平化并拆分成多个子张量
        auto srcFlatDataSplits =
            srcFlatData.split_with_sizes(c10::IntArrayRef(send_lengthsL), 0);
        for (const auto i : c10::irange(size_)) {
          srcFlatDataSplits[i].copy_(srcdata[i].view({-1}));
        }
        // 切换到源数据张量的设备，并获取全局锁
        c10::DeviceGuard guard1(srcdata[0].device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        // 执行 MPI 的 Alltoallv 操作，进行数据交换
        MPI_CHECK(MPI_Alltoallv(
            srcFlatData.data_ptr(),
            send_lengths.data(),
            send_offsets.data(),
            mpiDatatype.at(srcdata[0].scalar_type()),
            dstFlatData.data_ptr(),
            recv_lengths.data(),
            recv_offsets.data(),
            mpiDatatype.at(dstdata[0].scalar_type()),
            pgComm_));
        // 将接收到的扁平化数据拆分并复制到目标数据张量
        auto dstFlatDataSplits =
            dstFlatData.split_with_sizes(c10::IntArrayRef(recv_lengthsL), 0);
        for (const auto i : c10::irange(size_)) {
          dstdata[i].view({-1}).copy_(dstFlatDataSplits[i]);
        }
      };
  // 创建一个 WorkEntry 对象，用于保存运行函数和相关参数
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  // 将任务入队，并返回指向任务的指针
  return enqueue(
      std::move(entry),
      "mpi:all_to_all",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<Work> ProcessGroupMPI::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  // 检查发送张量的数量是否为单个
  checkSingleTensor(tensors);

  // 获取第一个张量的引用
  auto& tensor = tensors[0];
  // 初始化 MPI 请求对象
  MPI_Request request = MPI_REQUEST_NULL;

  {
    // 切换到张量的设备，并获取全局锁
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    // 使用 MPI_Isend 发送张量数据到指定的目标进程
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),                              // 获取张量数据的指针
        tensor.numel(),                                 // 获取张量中元素的数量
        mpiDatatype.at(tensor.scalar_type()),           // 获取张量元素类型对应的 MPI 数据类型
        dstRank,                                        // 目标进程的排名
        tag,                                            // 消息标签，用于标识消息类型
        pgComm_,                                        // 进程组通信器
        &request));                                     // 发送操作的请求对象

  }

  // 创建一个异步工作对象，用于跟踪 MPI 发送操作
  return c10::make_intrusive<AsyncWork>(
      request,                                          // MPI 发送操作的请求对象
      std::vector<at::Tensor>(),                        // 空张量向量，无关联的张量
      "mpi:send",                                       // 异步工作的描述字符串，指示 MPI 发送操作
      std::optional<std::vector<at::Tensor>>(tensors)); // 可选的包含相关张量的向量，这里是发送的张量
}

c10::intrusive_ptr<Work> ProcessGroupMPI::recv(
    std::vector<at::Tensor>& tensors,  // 接收函数，参数为张量向量
    int srcRank,                       // 源排名
    int tag) {                         // 标签
  checkSingleTensor(tensors);          // 检查是否只有一个张量

  auto& tensor = tensors[0];           // 获取第一个张量的引用
  MPI_Request request = MPI_REQUEST_NULL;  // MPI 请求对象初始化为空

  {
    c10::DeviceGuard guard(tensor.device());  // 确保当前设备是张量的设备
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);  // 获取全局互斥锁
    MPI_CHECK(MPI_Irecv(                // 非阻塞接收 MPI 消息
        tensor.data_ptr(),             // 接收数据的内存地址
        tensor.numel(),                // 接收数据的元素数量
        mpiDatatype.at(tensor.scalar_type()),  // MPI 数据类型
        srcRank,                       // 源排名
        tag,                           // 标签
        pgComm_,                       // MPI 通信器
        &request));                    // MPI 请求对象
  }

  return c10::make_intrusive<AsyncWork>(  // 返回一个异步工作对象
      request,                         // MPI 请求对象
      tensors,                         // 接收到的张量
      "mpi:recv",                      // 操作类型描述
      std::optional<std::vector<at::Tensor>>(tensors));  // 可选的张量向量
}

c10::intrusive_ptr<Work> ProcessGroupMPI::recvAnysource(
    std::vector<at::Tensor>& tensors,  // 接收函数，参数为张量向量
    int tag) {                         // 标签
  checkSingleTensor(tensors);          // 检查是否只有一个张量

  auto& tensor = tensors[0];           // 获取第一个张量的引用
  MPI_Request request = MPI_REQUEST_NULL;  // MPI 请求对象初始化为空

  {
    c10::DeviceGuard guard(tensor.device());  // 确保当前设备是张量的设备
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);  // 获取全局互斥锁
    MPI_CHECK(MPI_Irecv(                // 非阻塞接收 MPI 消息
        tensor.data_ptr(),             // 接收数据的内存地址
        tensor.numel(),                // 接收数据的元素数量
        mpiDatatype.at(tensor.scalar_type()),  // MPI 数据类型
        MPI_ANY_SOURCE,                // 从任意源接收
        tag,                           // 标签
        pgComm_,                       // MPI 通信器
        &request));                    // MPI 请求对象
  }

  return c10::make_intrusive<AsyncWork>(  // 返回一个异步工作对象
      request,                         // MPI 请求对象
      tensors,                         // 接收到的张量
      "mpi:recvAnySource",             // 操作类型描述
      std::optional<std::vector<at::Tensor>>(tensors));  // 可选的张量向量
}

c10::intrusive_ptr<Work> ProcessGroupMPI::barrier(const BarrierOptions& opts) {
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {  // 定义运行函数
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);  // 获取全局互斥锁
        MPI_CHECK(MPI_Barrier(pgComm_));  // MPI 同步屏障
      };
  auto entry =
      std::make_unique<WorkEntry>(nullptr, nullptr, std::move(runFunc));  // 创建工作条目
  return enqueue(std::move(entry), "mpi:barrier", c10::nullopt);  // 将工作条目加入队列并返回
}

c10::intrusive_ptr<Work> ProcessGroupMPI::_allgather_base(
    at::Tensor& /*unused */,
    at::Tensor& /*unused */,
    const AllgatherOptions& /*unused */) {
  TORCH_CHECK(false, "no support for _allgather_base in MPI process group");  // 抛出错误，MPI 过程组不支持 _allgather_base 操作
}

} // namespace c10d

#endif // USE_C10D_MPI
```