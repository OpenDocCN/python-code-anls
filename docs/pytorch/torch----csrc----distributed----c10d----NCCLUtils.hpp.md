# `.\pytorch\torch\csrc\distributed\c10d\NCCLUtils.hpp`

```py
#pragma once
// 如果定义了 USE_C10D_NCCL，则包含下面的头文件和代码
#ifdef USE_C10D_NCCL

// 包含标准 C 库的头文件
#include <stdio.h>
#include <stdlib.h>

// 包含 C++ 标准库的头文件
#include <memory>
#include <mutex>
#include <thread>

// 包含 PyTorch 的头文件
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

// 包含 NCCL 库的头文件
#include <nccl.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>

// 根据 NCCL 主版本和次版本号定义宏
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 14)
#define NCCL_HAS_COMM_NONBLOCKING
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 18)
#define NCCL_HAS_COMM_SPLIT
#endif

// 仅在 NCCL 版本大于等于 2.13 时启用 ncclGetLastError() 和 ncclRemoteError()
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 13)
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
// 在 NCCL 主版本大于等于 3 时同样启用这些功能
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
#endif

// 仅在 NCCL 版本大于等于 2.4 时启用错误检查功能
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 4)
#define ENABLE_NCCL_ERROR_CHECKING
// 在 NCCL 主版本大于等于 3 时同样启用错误检查
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_ERROR_CHECKING
#endif

// 仅在 NCCL 版本大于等于 2.7 时启用 P2P 支持
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 7)
#define ENABLE_NCCL_P2P_SUPPORT
// 在 NCCL 主版本大于等于 3 时同样启用 P2P 支持
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_P2P_SUPPORT
#endif

// 仅在 NCCL 版本大于等于 2.11 时启用预乘和求和支持
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 11)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
// 在 NCCL 主版本大于等于 3 时同样启用预乘和求和支持
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#endif

// 仅在 NCCL 版本大于等于 2.17 时启用 Comm CTA CGA 支持
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 17)
#define NCCL_HAS_COMM_CTA_CGA
// 在 NCCL 主版本大于等于 3 时同样启用 Comm CTA CGA 支持
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define NCCL_HAS_COMM_CTA_CGA
#endif

// 判断是否支持 NCCL 注册功能，对应 NCCL_REGISTRATION_SUPPORTED 宏或 NCCL 2.19+
#if defined(NCCL_REGISTRATION_SUPPORTED) ||                              \
    ((defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
      (NCCL_MINOR >= 19)))
#define NCCL_HAS_COMM_REGISTER
// 在 NCCL 主版本大于等于 3 时同样支持 NCCL 注册
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define NCCL_HAS_COMM_REGISTER
#endif

// 定义一个宏来检查 NCCL 命令的执行结果，以及失败原因
// 使用 do { ... } while(0) 结构来确保宏定义不会引起副作用
#define C10D_NCCL_CHECK(cmd, failureReason)                                   \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    // 如果 result 不等于 ncclSuccess，则进入条件判断
    if (result != ncclSuccess) {                                              \
      // 构造错误信息字符串，包含当前文件名、行号以及 NCCL 错误信息和详细描述
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      // 使用 TORCH_CHECK_WITH 宏抛出 DistBackendError 异常，携带错误信息 err
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)
// 宏定义：在非阻塞调用中，检查 NCCL 返回值是否成功，若不成功则抛出异常
#define C10D_NCCL_CHECK_NONBLOCKING(cmd, failureReason)                       \
  do {                                                                        \
    ncclResult_t result = cmd;                                                // 执行 NCCL 命令并获取返回结果
    if (result != ncclSuccess && result != ncclInProgress) {                  // 检查返回结果是否不成功或在进行中
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     // 构造错误信息字符串，包含文件名和行号
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                // 获取详细的 NCCL 错误信息
      TORCH_CHECK_WITH(DistBackendError, false, err);                         // 抛出异常
    }                                                                         \
  } while (0)

// 宏定义：在非阻塞调用中，检查 NCCL 返回值是否超时，若超时则抛出异常
#define C10D_NCCL_CHECK_TIMEOUT(cmd, comm, failureReason)                     \
  ncclResult_t result = cmd;                                                  // 执行 NCCL 命令并获取返回结果
  auto startTimepoint = std::chrono::steady_clock::now();                     // 记录开始时间点
  while (result == ncclInProgress) {                                          // 当 NCCL 操作仍在进行中时循环
    if (nccl_nonblocking_timeout() > 0) {                                     // 检查非阻塞超时时间是否大于零
      auto currentTimepoint = std::chrono::steady_clock::now();               // 获取当前时间点
      auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    // 计算经过的时间（秒）
                             currentTimepoint - startTimepoint)               // 从开始到当前时间的时间差
                             .count();                                       
      if (timeElapsed > nccl_nonblocking_timeout()) {                         // 如果超过了非阻塞超时时间
        std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" + // 构造超时错误信息字符串，包含文件名和行号
            std::to_string(__LINE__) + ", " +                                 
            ncclGetErrorWithVersion(result) + "\n" +                          // 获取 NCCL 错误信息
            getNcclErrorDetailStr(result, failureReason);                     // 获取详细的 NCCL 错误信息
        TORCH_CHECK_WITH(DistBackendError, false, err);                       // 抛出异常
      }                                                                       \
    }                                                                         \
    ncclCommGetAsyncError(comm, &result);                                     // 获取异步错误状态
  }                                                                           \
  if (result != ncclSuccess) {                                                // 如果最终操作不成功
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +       // 构造错误信息字符串，包含文件名和行号
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) +   // 获取 NCCL 错误信息
        "\n" + getNcclErrorDetailStr(result, failureReason);                  // 获取详细的 NCCL 错误信息
    TORCH_CHECK_WITH(DistBackendError, false, err);                           // 抛出异常
  }
// 定义一个带超时检查的宏，用于执行一个 NCCL 命令，并在超时或错误时终止程序
#define C10D_NCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)           \
  ncclResult_t state = cmd;                                                  \
  auto startTimepoint = std::chrono::steady_clock::now();                    \
  // 如果命令状态为进行中，进入循环
  if (state == ncclInProgress) {                                             \
    do {                                                                     \
      // 如果设置了非阻塞超时
      if (nccl_nonblocking_timeout() > 0) {                                  \
        auto currentTimepoint = std::chrono::steady_clock::now();            \
        // 计算从开始执行命令到现在经过的秒数
        auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>( \
                               currentTimepoint - startTimepoint)            \
                               .count();                                     \
        // 如果超过设定的超时时间，生成超时错误信息并抛出异常
        if (timeElapsed > nccl_nonblocking_timeout()) {                      \
          std::string err = "NCCL timeout in: " + std::string(__FILE__) +    \
              ":" + std::to_string(__LINE__) + ", " +                        \
              ncclGetErrorWithVersion(state) + "\n" +                        \
              getNcclErrorDetailStr(state, failureReason);                   \
          TORCH_CHECK_WITH(DistBackendError, false, err);                    \
        }                                                                    \
      }                                                                      \
      // 检查 NCCL 通信是否有异步错误
      ncclCommGetAsyncError(comm->getNcclComm(), &state);                    \
    } while (state == ncclInProgress);                                       \
  }                                                                          \
  // 如果命令执行状态不是成功，则生成错误信息并抛出异常
  if (state != ncclSuccess) {                                                \
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +      \
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) +   \
        "\n" + getNcclErrorDetailStr(state, failureReason);                  \
    TORCH_CHECK_WITH(DistBackendError, false, err);                          \
  }

// 定义一个宏，用于打印 NCCL 命令的错误信息并终止程序
#define C10D_NCCL_ASSERT(cmd)                            \
  do {                                                   \
    ncclResult_t result = cmd;                           \
    // 如果命令执行结果不是成功，则生成错误信息并打印到标准错误输出，然后终止程序
    if (result != ncclSuccess) {                         \
      std::string err = ncclGetErrorWithVersion(result); \
      fprintf(                                           \
          stderr,                                        \
          "NCCL error in: %s:%d, %s\n",                  \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
      abort();                                           \
    }                                                    \
  } while (0)

// 命名空间 c10d 下的静态变量，用于表示键名为 "entries" 的 IValue 对象
namespace c10d {
static c10::IValue entries_key = "entries";
// 定义静态常量，表示 NCCL 通信状态的键名
static c10::IValue nccl_comm_key = "nccl_comm_state";
// 定义静态常量，表示版本号的键名
static c10::IValue version_key = "version";
// 当更改 dump 的内容或格式时更新，如添加字段为次要更新，更改现有字段为主要更新
static c10::IValue version_val = "2.2";
// 定义静态常量，表示进程组配置的键名
static c10::IValue pg_config_key = "pg_config";
// 定义静态常量，表示记录 ID 的键名
static c10::IValue record_id_key = "record_id";
// 定义静态常量，表示进程组 ID 的键名
static c10::IValue pg_id_key = "pg_id";
// 定义静态常量，表示进程组名称的键名
static c10::IValue pg_name_key = "process_group";
// 定义静态常量，表示集体通信序列 ID 的键名
static c10::IValue collective_seq_id_key = "collective_seq_id";
// 定义静态常量，表示点对点通信序列 ID 的键名
static c10::IValue p2p_seq_id_key = "p2p_seq_id";
// 定义静态常量，表示是否点对点通信的键名
static c10::IValue is_p2p_key = "is_p2p";
// 定义静态常量，表示操作 ID 的键名
static c10::IValue op_id_key = "op_id";
// 定义静态常量，表示性能分析名称的键名
static c10::IValue profiling_name_key = "profiling_name";
// 定义静态常量，表示输入张量大小的键名
static c10::IValue input_sizes_key = "input_sizes";
// 定义静态常量，表示输入张量数据类型的键名
static c10::IValue input_dtypes_key = "input_dtypes";
// 定义静态常量，表示输出张量大小的键名
static c10::IValue output_sizes_key = "output_sizes";
// 定义静态常量，表示输出张量数据类型的键名
static c10::IValue output_dtypes_key = "output_dtypes";
// 定义静态常量，表示创建时间的键名，单位为纳秒
static c10::IValue time_created_key = "time_created_ns";
// 定义静态常量，表示持续时间的键名，单位为毫秒
static c10::IValue duration_key = "duration_ms";
// 定义静态常量，表示超时时间的键名，单位为毫秒
static c10::IValue timeout_key = "timeout_ms";

// 定义静态常量，表示帧信息的键名
static c10::IValue frames_key = "frames";
// 定义静态常量，表示状态信息的键名
static c10::IValue state_key = "state";
// 定义静态常量，表示行号的键名
static c10::IValue line_key = "line";
// 定义静态常量，表示名称的键名
static c10::IValue name_key = "name";
// 定义静态常量，表示文件名的键名
static c10::IValue filename_key = "filename";
// 定义静态常量，表示是否已停用的键名
static c10::IValue retired_key = "retired";
// 定义静态常量，表示开始发现时间的键名，单位为纳秒
static c10::IValue time_discovered_started_key = "time_discovered_started_ns";
// 定义静态常量，表示完成发现时间的键名，单位为纳秒
static c10::IValue time_discovered_completed_key =
    "time_discovered_completed_ns";

// 声明一个用于计算张量哈希值的函数
TORCH_API size_t hashTensors(const std::vector<at::Tensor>& tensors);
// 声明一个获取 NCCL 版本信息的函数
TORCH_API std::string getNcclVersion();
// 声明一个根据 NCCL 错误码获取错误信息的函数
TORCH_API std::string ncclGetErrorWithVersion(ncclResult_t error);
// 返回是否使用 NCCL 非阻塞模式的布尔值
bool nccl_use_nonblocking();
// 返回 NCCL 非阻塞模式的超时时间
int nccl_nonblocking_timeout();
// 返回是否应该广播 NCCL 唯一标识符的布尔值，参数指定是否发送接收自身
bool shouldBroadcastNCCLUniqueID(bool isSendRecvSelf);

// 根据抛出时机提供关于 NCCL 错误码的详细信息，可选参数为进程组失败原因
// 该函数属于 TORCH_API，提供给外部使用
TORCH_API std::string getNcclErrorDetailStr(
    ncclResult_t error,
    std::optional<std::string> processGroupFailureReason = c10::nullopt);

// 将 NCCL 调试信息写入本地磁盘或用户定义的存储
// 对调试信息写入器设置以下约束：
// 1. 写入器只能注册一次。
// 2. 一旦注册，用户无法更改（包括取消注册）。
// 3. 建议在训练设置中注册自定义写入器，
//    如果用户在调用 launchAsyncDebugDump 之前未注册，则失去注册的机会（并且默认写入器将自动注册）。
class TORCH_API DebugInfoWriter {
 public:
  // 默认虚析构函数
  virtual ~DebugInfoWriter() = default;
  // 纯虚函数，子类需实现将 NCCL 跟踪信息写入目标的功能
  virtual void write(const std::string& ncclTrace);
  // 静态函数，获取特定排名的写入器对象的引用
  static DebugInfoWriter& getWriter(int rank);
  // 静态函数，注册唯一指针类型的调试信息写入器
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);
  // 虚函数，返回写入器的目标文件名
  virtual std::string getWriterTarget() {
    return filename_;
  }

 protected:
  // 受保护构造函数，用于子类初始化
  DebugInfoWriter(std::string namePrefix, int rank) {
    filename_ = c10::str(namePrefix, rank);


    // 使用 c10 命名空间下的 str 函数将 namePrefix 和 rank 转换为字符串，并赋值给 filename_
    filename_ = c10::str(namePrefix, rank);
  }


  std::string filename_;


  // 声明一个名为 filename_ 的私有成员变量，类型为 std::string，用于存储文件名
  std::string filename_;


 private:


  // 下面的成员变量和函数为私有，只能在类的内部访问
 private:


  static std::unique_ptr<DebugInfoWriter> writer_;


  // 声明一个静态的 unique_ptr 智能指针 writer_，指向 DebugInfoWriter 类型的对象，用于调试信息写入
  static std::unique_ptr<DebugInfoWriter> writer_;


  static std::atomic<bool> hasWriterRegistered_;


  // 声明一个静态的 atomic<bool> 类型变量 hasWriterRegistered_，用于表示是否已经注册了写入器
  static std::atomic<bool> hasWriterRegistered_;
};

// RAII wrapper for NCCL communicator
class NCCLComm {
 public:
  // 构造函数，接受一个 ncclComm_t 对象作为参数
  explicit NCCLComm(ncclComm_t ncclComm)
      : ncclComm_(ncclComm),
        aborted_(false),  // 初始化 aborted_ 为 false
        ncclAsyncErr_(ncclSuccess),  // 初始化 ncclAsyncErr_ 为 ncclSuccess
        commFailureReason_(c10::nullopt),  // 初始化 commFailureReason_ 为空
        initialized_(false) {}  // 初始化 initialized_ 为 false

  // 默认构造函数，调用上述构造函数并传入 nullptr
  NCCLComm() : NCCLComm(nullptr) {}

  // 析构函数，在对象生命周期结束时调用
  ~NCCLComm() noexcept {
    // 使用互斥锁确保在读取 aborted_ 后执行内存屏障
    std::unique_lock<std::mutex> lock(mutex_);
    // 如果 ncclComm_ 存在且未被中止
    if (ncclComm_ && !aborted_) {
#ifdef ENABLE_NCCL_ERROR_CHECKING
      // 如果启用了 NCCL 错误检查，则调用 ncclCommAbort 中止通信
      C10D_NCCL_ASSERT(::ncclCommAbort(ncclComm_));
#else
      // 否则调用 ncclCommDestroy 销毁通信对象
      C10D_NCCL_ASSERT(::ncclCommDestroy(ncclComm_));
#endif
    }
  }

  // 静态工厂方法，创建 NCCLComm 对象
  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId) {
    auto comm = std::make_shared<NCCLComm>();
    // 初始化 NCCL 通信对象，返回的 comm 是共享指针
    C10D_NCCL_CHECK(
        ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank),
        c10::nullopt);
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->initialized_ = true;
    return comm;
  }

#ifdef NCCL_HAS_COMM_NONBLOCKING
  // 可选的非阻塞模式下创建 NCCLComm 对象的静态方法
  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      ncclConfig_t& config) {
    auto comm = std::make_shared<NCCLComm>();
    bool isInitialized = false;
    // 如果允许使用非阻塞模式，则配置为非阻塞
    if (nccl_use_nonblocking()) {
      config.blocking = 0;
      LOG(INFO) << "Rank " << rank
                << ": creating NCCL communicator in nonblocking mode";
      // 使用非阻塞模式初始化 NCCL 通信对象
      C10D_NCCL_CHECK_NONBLOCKING(
          ncclCommInitRankConfig(
              &(comm->ncclComm_), numRanks, commId, rank, &config),
          c10::nullopt);
    } else {
      // 否则使用阻塞模式初始化 NCCL 通信对象
      C10D_NCCL_CHECK(
          ncclCommInitRankConfig(
              &(comm->ncclComm_), numRanks, commId, rank, &config),
          c10::nullopt);
      // 在阻塞模式下，NCCL CHECK 后通信对象已初始化
      isInitialized = true;
    }
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->initialized_ = isInitialized;
    return comm;
  }
#endif

  // 分裂 NCCLComm 对象的静态方法
  static std::shared_ptr<NCCLComm> split(
      NCCLComm* source,
      int color_id,
      int rank,
      ncclConfig_t& config,
      std::vector<uint64_t>& ranks_ull);

#if defined(IS_NCCLX) && defined(NCCL_COMM_DUMP)
  // 如果定义了 NCCLX 和 NCCL_COMM_DUMP，则导出 NCCL 通信对象状态的方法
  std::unordered_map<std::string, std::string> ncclCommDump() {
    std::unordered_map<std::string, std::string> dump;
    // 如果通信对象已中止，则无法导出其状态
    if (isAborted()) {
      LOG(INFO) << "Communicator was aborted before trying to dump its state.";
      return dump;
    }
    // 导出 NCCL 通信对象的状态到 dump 中
    C10D_NCCL_CHECK(::ncclCommDump(ncclComm_, dump), c10::nullopt);
    return dump;
  }
#endif

  // 获取 NCCL 唯一标识符的方法
  ncclUniqueId getNcclId() {
  // 返回 ncclId_，即 NCCL 通信对象的 ID
  return ncclId_;
}

// 不可复制
NCCLComm(const NCCLComm&) = delete;
// 禁止赋值运算符复制
NCCLComm& operator=(const NCCLComm&) = delete;

// 不支持移动赋值，因为没有有效的使用场景
NCCLComm& operator=(NCCLComm&& other) = delete;

// 移动构造函数
NCCLComm(NCCLComm&& other) {
  // 使用 other 的锁，因为它读取 other 的状态
  // 不能使用 this.mutex_，因为当前对象正在构造中
  std::unique_lock<std::mutex> lock(other.mutex_);
  std::swap(ncclComm_, other.ncclComm_);
  std::swap(aborted_, other.aborted_);
  std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
  std::swap(initialized_, other.initialized_);
}

// 获取 ncclComm_t 对象的方法声明
ncclComm_t getNcclComm();

// 获取 NCCL 通信失败原因的可选项
std::optional<std::string> getNcclCommFailureReason() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return commFailureReason_;
}

// 终止 NCCL 通信
void ncclCommAbort(
    std::optional<std::string> commFailureReason = c10::nullopt) {
  std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
    // 如果已经中止，不应再次中止。
    if (aborted_) {
      // Should not abort twice.
      return;
    }

#ifdef NCCL_HAS_COMM_REGISTER
    // 在中止之前取消注册所有已注册的段。
    for (auto& it : registeredSegmentHandles_) {
      void* handle = it.second;
      C10D_NCCL_CHECK(
          ::ncclCommDeregister(ncclComm_, handle),
          c10::str(
              "Failed to deregister segment handle ",
              handle,
              " on ncclComm_ ",
              ncclComm_));
    }
    registeredSegmentHandles_.clear();
#endif

    // 如果由 ProcessGroupNCCL 提供了真实的失败原因（例如工作超时）
    // 设置真实的失败原因
    commFailureReason_ = commFailureReason;
    LOG(INFO) << "Aborting ncclComm_ " << ncclComm_ << " with reason: "
              << (commFailureReason ? *commFailureReason
                                    : "No abort reason provided.");
#ifndef NCCL_HAS_COMM_NONBLOCKING
    // 如果没有非阻塞通信，执行中止通信操作
    C10D_NCCL_CHECK(::ncclCommAbort(ncclComm_), commFailureReason_);
#else
    // 否则，执行带超时的中止通信操作
    C10D_NCCL_CHECK_TIMEOUT(
        ::ncclCommAbort(ncclComm_), ncclComm_, commFailureReason_);
#endif
    // 标记已中止
    aborted_ = true;
    // 将 ncclComm_ 置为空指针，避免后续使用通信器
    ncclComm_ = nullptr;

    // 如果尚未设置异步错误，将其设置为系统错误，避免使用通信器
    if (ncclAsyncErr_ == ncclSuccess) {
      ncclAsyncErr_ = ncclSystemError;
    }
#else
    // 如果禁用错误检查，则不执行任何操作
    // This is a NOOP, if error checks are disabled.
    return;
#endif
  }

  // 检查通信器是否已中止
  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  // 获取通信拆分计数器的当前值
  uint64_t getCommSplitCounter() const {
    return ncclCommSplitCounter_;
  }

  // 检查是否存在 NCCL 错误
  ncclResult_t checkForNcclError() {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
    // 如果存在异步错误，则返回错误码
    if (ncclAsyncErr_ != ncclSuccess) {
      return ncclAsyncErr_;
    }
    // 否则，检查通信器的异步错误状态
    C10D_NCCL_CHECK(
        ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr_), commFailureReason_);
    return ncclAsyncErr_;
#else
    // 如果禁用错误检查，始终返回成功状态
    // Always return success, if error checks are disabled.
    return ncclSuccess;
#endif
  }

  // 注册内存段到通信器中
  ncclResult_t registerSegment(void* ptr, size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
    // 只注册缓存分配器中的内存段，这些段保证地址范围不重叠
    // 因此，ptr 指向的内存段总是映射到唯一的句柄，当前 ptr 注册之前不应注册和释放
    TORCH_CHECK(
        registeredSegmentHandles_.count(ptr) == 0,
        "Segment with ptr ",
        ptr,
        " has already been registered on ncclComm_ ",
        ncclComm_);

    void* handle;
    // 在通信器中注册指定的内存段
    C10D_NCCL_CHECK(
        ncclCommRegister(ncclComm_, ptr, size, &handle),
        c10::str(
            "Failed to register segment with ptr ",
            ptr,
            ", size ",
            size,
            " on ncclComm_ ",
            ncclComm_));
    // 将注册的句柄和对应的指针存储到映射表中
    registeredSegmentHandles_[ptr] = handle;
    return ncclSuccess;
#else
    // 如果未启用通信注册，返回无效使用错误码
    return ncclInvalidUsage;
#endif
#endif
  }

  // 从注册表中注销给定指针对应的段落
  ncclResult_t deregisterSegment(void* ptr) {
    // 锁定互斥量，以确保线程安全
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
    // 检查给定指针是否已在注册表中
    TORCH_CHECK(
        registeredSegmentHandles_.count(ptr) == 1,
        "Segment with ptr ",
        ptr,
        " is not registered on ncclComm_ ",
        ncclComm_);

    // 获取注册的句柄
    void* handle = registeredSegmentHandles_[ptr];

    // 执行通信操作库的段落注销
    C10D_NCCL_CHECK(
        ncclCommDeregister(ncclComm_, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            ", with ptr ",
            ptr,
            " on ncclComm_ ",
            ncclComm_));

    // 从注册表中移除已注销的指针
    registeredSegmentHandles_.erase(ptr);

    // 返回成功的结果码
    return ncclSuccess;
#else
    // 如果没有段落注册功能，返回无效使用错误码
    return ncclInvalidUsage;
#endif
  }

  friend class ProcessGroupNCCL;

 protected:
  // 等待通信器初始化完成的辅助函数
  void waitUntilInitialized(int timeoutSecs);
  ncclComm_t ncclComm_; // 用于 NCCL 通信的句柄
  // 用于此通信器的唯一 nccl_id
  ncclUniqueId ncclId_;
  bool aborted_; // 标志指示通信是否中止
  uint64_t ncclCommSplitCounter_{0}; // NCCL 通信拆分计数器
  ncclResult_t ncclAsyncErr_; // NCCL 异步错误结果
  mutable std::mutex mutex_; // 可变互斥量，用于保护共享资源
  // 此通信器对应的排名
  int rank_;
  // 通信器失败原因的可选描述，由 ProcessGroupNCCL 提供更好的错误消息
  std::optional<std::string> commFailureReason_;
  bool initialized_{false}; // 指示通信器是否已初始化的标志
#ifdef NCCL_HAS_COMM_REGISTER
  // 存储由 NCCL 注册的张量句柄
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif
};

// 自动清理 premul sums 的辅助类
struct ncclRedOpRAII {
  ncclRedOpRAII() = default;
  ncclRedOpRAII(ncclRedOp_t op) : op_(op) {}
  ncclRedOpRAII(ncclRedOp_t op, ncclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  ncclRedOpRAII(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII& operator=(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII(ncclRedOpRAII&& tmp) : ncclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
#if defined(ENABLE_NCCL_PREMUL_SUM_SUPPORT)
  // 在析构函数中销毁 premul sums
  ~ncclRedOpRAII() {
    if (premul_sum_) {
      ncclRedOpDestroy(op_, comm_);
    }
  }
#endif
  operator ncclRedOp_t() const {
    return op_;
  }
  ncclRedOp_t op_; // NCCL 归约操作类型
  ncclComm_t comm_; // 相关联的通信句柄
  bool premul_sum_ = false; // 是否启用 premul sums
};

/* Helper used by work::getDuration() and nccl flight recorder */
// 从 CUDA 事件中获取持续时间的辅助函数
float getDurationFromEvent(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent);

struct NCCLTraceBuffer {
  // 获取 NCCLTraceBuffer 的静态实例
  static NCCLTraceBuffer* get() {
    // 故意在退出时泄漏，因为它将保持可能被析构的 Python 状态
    static NCCLTraceBuffer* instance = new NCCLTraceBuffer();
    return instance;
  }
  NCCLTraceBuffer() {
    // 从环境变量获取最大条目数
    max_entries_ = getCvarInt({"TORCH_NCCL_TRACE_BUFFER_SIZE"}, 0);
    // 从环境变量获取是否捕获 C++ 堆栈信息的标志
    capture_cpp_stack_ = getCvarBool({"TORCH_NCCL_TRACE_CPP_STACK"}, false);
    // 根据最大条目数是否大于零，判断是否启用跟踪
    enabled_ = max_entries_ > 0;
  }
  using Event = at::cuda::CUDAEvent; // 使用 CUDA 事件类型别名
  struct Entry {
    // incremented id in the trace buffer
    // used to figure out where in the circular entries
    // buffer this entry will be located to
    // update state information
    size_t id_;

    size_t pg_id_;

    // <group_name, group_desc>
    std::tuple<std::string, std::string> pg_name_;

    // collective_seq_id and p2p_seq_id refer to actual kernel launches (e.g. 1
    // per coalesced group).
    // collective_seq_id only increments for true collective operations (over
    // all ranks in the group). p2p_seq_id only increments over non-collective
    // operations in the group. op_id refers to logical operations (e.g. one per
    // op inside coalesced group)
    size_t collective_seq_id_;
    size_t p2p_seq_id_;
    size_t op_id_;
    std::string profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;

    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    Event *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_;

    // configured timeout for this entry
    c10::time_t timeout_ms_;

    // Is this a P2P event?
    bool isP2P_;

    std::optional<float> duration_;

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on CUDA APIs.
    std::optional<c10::time_t> time_discovered_started_;

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually complated, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on CUDA
    // APIs
    std::optional<c10::time_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int, 4> input_dims_;
    std::vector<c10::ScalarType> input_dtypes_;
    c10::SmallVector<int, 4> output_dims_;
    std::vector<c10::ScalarType> output_dtypes_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // 是否该工作条目已不再在 workMetaList_ 中？
                           // 已经退役但尚未完成的事件已超时

  };

  bool enabled_ = false; // 是否启用了记录功能
  bool capture_cpp_stack_ = false; // 是否捕获 C++ 堆栈信息
  std::mutex mutex_; // 互斥锁，用于保护共享资源
  std::vector<Entry> entries_; // 存储记录条目的容器
  size_t max_entries_ = 0; // 最大记录条目数量
  size_t next_ = 0; // 下一个可用记录条目的索引
  size_t id_ = 0; // 当前记录条目的 ID
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>>
      pg_name_to_ranks_ = {}; // 映射，将 (pg_name, pg_name) 对映射到排名列表

  std::optional<size_t> record(
      size_t pg_id,
      const std::tuple<std::string, std::string>& pg_name,
      size_t collective_seq_id,
      size_t p2p_seq_id,
      size_t op_id,
      std::string profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      Event* start,
      Event* end,
      std::chrono::milliseconds timeout_ms,
      bool isP2P) {
    if (!enabled_) {
      return c10::nullopt; // 如果未启用记录功能，则返回空值
    }
    auto traceback =
        torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
    std::lock_guard<std::mutex> guard(mutex_); // 使用互斥锁保护共享资源

    auto te = Entry{
        id_,
        pg_id,
        pg_name,
        collective_seq_id,
        p2p_seq_id,
        op_id,
        std::move(profiling_name),
        std::move(traceback),
        std::move(start),
        std::move(end),
        c10::getTime(),
        timeout_ms.count(),
        isP2P,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        {},
        {},
        false};

    for (const auto& input : inputs) {
      c10::IntArrayRef sizes = input.sizes();
      te.input_dtypes_.push_back(input.dtype().toScalarType());
      te.input_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    for (const auto& output : outputs) {
      c10::IntArrayRef sizes = output.sizes();
      te.output_dtypes_.push_back(output.dtype().toScalarType());
      te.output_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    if (entries_.size() < max_entries_) {
      entries_.emplace_back(std::move(te)); // 将记录条目 te 添加到 entries_ 中
    } else {
      entries_[next_++] = std::move(te); // 将记录条目 te 添加到 entries_ 中，并处理循环队列
      if (next_ == max_entries_) {
        next_ = 0; // 如果达到最大记录条目数量，则重置 next_ 索引
      }
    }
    return id_++; // 返回当前记录条目的 ID，并递增
  }

  void record_pg_ranks(
      const std::tuple<std::string, std::string>& pg_name,
      std::vector<uint64_t> ranks) {
    if (!enabled_) {
      return; // 如果未启用记录功能，则直接返回
    }
    std::lock_guard<std::mutex> guard(mutex_); // 使用互斥锁保护共享资源
    pg_name_to_ranks_[pg_name] = ranks; // 将 pg_name 映射到对应的排名列表 ranks
  }

  void update_state(Entry& r) {
    if (r.start_ != nullptr) {
      bool started = r.start_->query(); // 查询事件是否已开始
      if (started && !r.time_discovered_started_) {
        r.time_discovered_started_ = c10::getTime(); // 记录事件开始的时间戳
      }
    }
    if (r.end_ != nullptr) {
      bool completed = r.end_->query(); // 查询事件是否已完成
      if (completed && !r.time_discovered_completed_) {
        r.time_discovered_completed_ = c10::getTime(); // 记录事件完成的时间戳
      }
      // 注意：缺少了更新时间戳的完整代码，可能需要补充
    }
  }
  // 释放锁之前，将已完成的事件标记为 retired 并清空其 start_ 和 end_ 指针
  }

  std::vector<Entry> dump_entries() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<Entry> result;
    result.reserve(entries_.size());
    // 将从 next_ 开始到末尾的所有条目复制到结果中
    result.insert(result.end(), entries_.begin() + next_, entries_.end());
    // 将从开头到 next_ 的所有条目复制到结果中
    result.insert(result.end(), entries_.begin(), entries_.begin() + next_);
    // 查询剩余事件的状态并更新，同时清空其 start_ 和 end_ 指针
    for (auto& r : result) {
      update_state(r);
      r.start_ = r.end_ = nullptr;
    }
    return result;
  }

  /*
  将一个事件标记为已完成并释放其事件。
  此函数由看门狗线程调用，与主线程的执行是异步的。
  默认情况下 compute_duration 为 true，因为 retire_id 只在看门狗线程中调用，
  而看门狗线程可能调用 cuda API，这可能会挂起，但必须小心，避免在任何必须
  保证不会挂起的函数中计算持续时间。（同时需要启用计时 - 参见 TORCH_NCCL_ENABLE_TIMING）
  */
  void retire_id(std::optional<size_t> id, bool compute_duration = true) {
    if (!enabled_ || !id) {
      return;
    }

    bool can_compute_duration = false;
    Event* startEvent = nullptr;
    Event* endEvent = nullptr;
    std::optional<float> duration = c10::nullopt;

    std::unique_lock<std::mutex> guard(mutex_);

    // 计算 entry 在 entries_ 中的位置
    Entry* entry = &entries_.at(*id % max_entries_);
    // 如果 entry 的 id 匹配给定的 id，则更新其状态
    if (entry->id_ == *id) {
      update_state(*entry);

      // 如果需要计算持续时间，则检查是否有必要的数据，并获取 start_ 和 end_ 指针
      if (compute_duration) {
        can_compute_duration = entry->time_discovered_completed_.has_value() &&
            entry->start_ && entry->end_;
        startEvent = entry->start_;
        endEvent = entry->end_;
      }
      // 将 entry 标记为已退役，并清空其 start_ 和 end_ 指针
      entry->retired_ = true;
      entry->start_ = entry->end_ = nullptr;
    }

    // 如果可以计算持续时间，则在释放锁之后计算，因为 cudaEventDuration() 可能会挂起
    if (can_compute_duration) {
      guard.unlock();
      // 计算事件的持续时间
      duration = getDurationFromEvent(*startEvent, *endEvent);
      guard.lock();

      // 再次获取 entry 指针，检查 entry 是否已被覆盖
      entry = &entries_.at(*id % max_entries_);
      if (entry->id_ != *id) {
        LOG(INFO)
            << "retire_id abandoned for id " << *id
            << ", event was overwritten while waiting to compute duration.";
        return;
      }
      // 如果成功计算了持续时间，则将其存储在 entry 的 duration_ 中
      if (duration.has_value()) {
        entry->duration_ = duration.value();
      }
    }
  }

  const c10::List<c10::IValue> getCollectiveTrace(
      bool includeStacktraces,
      bool onlyActive) {
    // 创建一个新的列表用于存储条目
    auto entries = new_list();
    // 获取当前的所有条目
    auto result = dump_entries();
    // 初始化用于存储堆栈跟踪的数据结构
    std::vector<torch::CapturedTraceback*> tracebacks;
    torch::SymbolizedTracebacks stracebacks;
    // 初始化用于存储所有帧的列表
    std::vector<c10::IValue> all_frames;
    if (includeStacktraces) {
      // 如果需要包含堆栈跟踪信息，则执行以下操作
      for (auto& e : result) {
        // 遍历结果集中的每一个元素
        tracebacks.push_back(e.traceback_.get());
        // 将每个元素的 traceback 指针添加到 tracebacks 向量中
      }
      // 使用 torch::symbolize 对 tracebacks 进行符号化处理
      stracebacks = torch::symbolize(tracebacks);
      // 遍历处理后的符号化堆栈帧
      for (const auto& f : stracebacks.all_frames) {
        // 对于 stracebacks 中的每一个堆栈帧 f
        auto d = new_dict();
        // 创建一个新的字典 d
        d.insert(name_key, f.funcname);
        // 将堆栈帧的函数名 f.funcname 插入到字典 d 中
        d.insert(filename_key, f.filename);
        // 将堆栈帧的文件名 f.filename 插入到字典 d 中
        d.insert(line_key, int64_t(f.lineno));
        // 将堆栈帧的行号 f.lineno 转换为 int64_t 后插入到字典 d 中
        all_frames.emplace_back(std::move(d));
        // 将字典 d 移动到 all_frames 向量的末尾
      }
    }
    // 结束包含堆栈跟踪信息的条件判断和处理
    // 遍历结果集中的每个事件
    for (auto i : c10::irange(result.size())) {
      // 创建一个新的字典用于存储事件信息
      auto dict = new_dict();
      // 获取当前索引处的事件引用
      auto& e = result.at(i);
      // 如果仅包含活动事件且事件已完成，则跳过当前事件
      if (onlyActive && e.time_discovered_completed_.has_value()) {
        continue;
      }

      // 如果需要包含堆栈跟踪信息
      if (includeStacktraces) {
        // 获取当前事件的堆栈跟踪信息
        auto& tb = stracebacks.tracebacks.at(i);
        auto frames = new_list();
        // 遍历堆栈帧列表
        for (int64_t frame : tb) {
          frames.push_back(all_frames.at(frame));
        }
        // 将堆栈帧列表插入到字典中
        dict.insert(frames_key, frames);
      }

      // 将事件的各个字段插入到字典中
      dict.insert(record_id_key, int64_t(e.id_));
      dict.insert(pg_id_key, int64_t(e.pg_id_));
      dict.insert(pg_name_key, e.pg_name_);
      dict.insert(collective_seq_id_key, int64_t(e.collective_seq_id_));
      dict.insert(p2p_seq_id_key, int64_t(e.p2p_seq_id_));
      dict.insert(op_id_key, int64_t(e.op_id_));
      dict.insert(profiling_name_key, e.profiling_name_);
      dict.insert(time_created_key, int64_t(e.time_created_));
      // 如果事件具有持续时间，则插入持续时间字段
      if (e.duration_) {
        dict.insert(duration_key, *e.duration_);
      }

      // 定义函数用于读取事件输入维度的大小信息
      auto it = e.sizes_.begin();
      auto read_sizes = [&](const c10::SmallVector<int, 4>& dims) {
        auto sizes = new_list();
        // 遍历维度列表，读取每个维度的大小信息
        for (auto dim : dims) {
          auto arg_sizes = new_list();
          for (auto i : c10::irange(dim)) {
            (void)i;
            arg_sizes.push_back(*it++);
          }
          sizes.push_back(arg_sizes);
        }
        return sizes;
      };

      // 将事件的输入维度大小信息插入字典中
      dict.insert(input_sizes_key, read_sizes(e.input_dims_));
      
      // 将事件的输入数据类型转换为字符串并插入字典中
      std::vector<std::string> input_dtypes_strs;
      input_dtypes_strs.reserve(e.input_dtypes_.size());
      for (const auto& input_dtype : e.input_dtypes_) {
        input_dtypes_strs.push_back(c10::toString(input_dtype));
      }
      dict.insert(input_dtypes_key, input_dtypes_strs);
      
      // 将事件的输出维度大小信息插入字典中
      dict.insert(output_sizes_key, read_sizes(e.output_dims_));
      
      // 将事件的输出数据类型转换为字符串并插入字典中
      std::vector<std::string> output_dtypes_strs;
      output_dtypes_strs.reserve(e.output_dtypes_.size());
      for (const auto& output_dtype : e.output_dtypes_) {
        output_dtypes_strs.push_back(c10::toString(output_dtype));
      }
      dict.insert(output_dtypes_key, output_dtypes_strs);
      
      // 根据事件的状态信息插入相应的状态字段到字典中
      if (e.time_discovered_completed_.has_value()) {
        dict.insert(state_key, "completed");
      } else if (e.time_discovered_started_.has_value()) {
        dict.insert(state_key, "started");
      } else {
        dict.insert(state_key, "scheduled");
      }

      // 插入事件开始时间到字典中，如果不存在则插入空值
      dict.insert(
          time_discovered_started_key,
          e.time_discovered_started_.has_value()
              ? int64_t(*e.time_discovered_started_)
              : c10::IValue());
      
      // 插入事件完成时间到字典中，如果不存在则插入空值
      dict.insert(
          time_discovered_completed_key,
          e.time_discovered_completed_.has_value()
              ? int64_t(*e.time_discovered_completed_)
              : c10::IValue());
      
      // 插入事件是否已退役标志到字典中
      dict.insert(retired_key, e.retired_);
      
      // 插入事件超时时间到字典中
      dict.insert(timeout_key, e.timeout_ms_);
      
      // 插入事件是否为点对点通信标志到字典中
      dict.insert(is_p2p_key, e.isP2P_);

      // 将当前事件的字典插入到结果列表中
      entries.push_back(dict);
    }
  // 返回 entries 变量
  return entries;
}

// 获取 PG 配置信息并返回一个 c10 字典
const c10::Dict<c10::IValue, c10::IValue> getPgConfig() {
  // 创建一个新的 PG 配置字典
  auto pg_config = new_dict();
  // 遍历 pg_name_to_ranks_ 中的每一个键值对
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    // 创建一个新的 PG 信息字典
    auto pg_info = new_dict();
    // 向 PG 信息字典中插入名称、描述和 ranks 字段
    pg_info.insert("name", std::get<0>(pg_name));
    pg_info.insert("desc", std::get<1>(pg_name));
    pg_info.insert("ranks", ranks_str(ranks));
    // 将 PG 信息字典插入到 PG 配置字典中，键为 PG 名称
    pg_config.insert(std::get<0>(pg_name), pg_info);
  }
  // 返回 PG 配置字典
  return pg_config;
}

// 导出所有 collective 和 ncclDumpMap 的信息
std::string dump(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& ncclDumpMap,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  // 创建一个新的字典对象 result
  auto result = new_dict();
  // 向 result 中插入版本信息键值对
  result.insert(version_key, version_val);
  // 向 result 中插入 PG 配置信息键值对
  result.insert(pg_config_key, getPgConfig());

  // 如果需要包含 collective 跟踪信息
  if (includeCollectives) {
    // 向 result 中插入 collective 跟踪信息键值对
    result.insert(
        entries_key, getCollectiveTrace(includeStackTraces, onlyActive));
  }

  // 将 ncclDumpMap 转换为字典形式
  auto per_comm_dict = new_dict();
  // 如果 ncclDumpMap 存在值
  if (ncclDumpMap.has_value()) {
    // 遍历 ncclDumpMap 中的每一个键值对
    for (const auto& [ncclId, ncclDump] : ncclDumpMap.value()) {
      // 创建一个新的内部字典对象
      auto inner_dict = new_dict();
      // 遍历 ncclDump 中的每一个键值对，并插入到内部字典中
      for (const auto& [key, value] : ncclDump) {
        inner_dict.insert(key, value);
      }
      // 将内部字典插入到 per_comm_dict 中，键为 ncclId
      per_comm_dict.insert(ncclId, inner_dict);
    }
  }
  // 如果 per_comm_dict 的大小大于 0，则向 result 中插入 nccl_comm_key 键值对
  if (per_comm_dict.size() > 0) {
    result.insert(nccl_comm_key, per_comm_dict);
  }
  // 将 result 序列化为字符串并返回
  return pickle_str(result);
}
};

} // namespace c10d

#endif // USE_C10D_NCCL
```