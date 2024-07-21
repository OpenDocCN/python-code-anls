# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupNCCL.cpp`

```
#ifdef USE_C10D_NCCL
// 如果定义了 USE_C10D_NCCL 宏，则编译以下代码块

#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/monitor/instrumentation.h>
#include <torch/torch.h>

namespace c10d {

constexpr const char* const kNCCLAbortedCommStoreKey = "NCCLABORTEDCOMM";

namespace {

// 如果 NCCL 主版本号大于 2 或者主版本号等于 2 且次版本号大于等于 10，则定义 NCCL_HAS_AVG 为 1
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#define NCCL_HAS_AVG 1
#endif

// NCCL 操作类型映射表，将 ReduceOp 的枚举值映射为 ncclRedOp_t 类型
const std::map<ReduceOp::RedOpType, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
#ifdef NCCL_HAS_AVG
    {ReduceOp::AVG, ncclAvg},
#endif
};

// NCCL 数据类型映射表，将 ATen 的数据类型映射为 ncclDataType_t 类型
std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
    {at::kBool, ncclUint8},
    {at::kFloat8_e5m2, ncclUint8},
    {at::kFloat8_e4m3fn, ncclUint8},
    {at::kFloat8_e4m3fnuz, ncclUint8},
    {at::kFloat8_e5m2fnuz, ncclUint8},
#if HAS_NCCL_BF16_DATATYPE
    {at::kBFloat16, ncclBfloat16},
#endif
};

// 获取 NCCL 数据类型的辅助函数，如果类型不支持则报错
ncclDataType_t getNcclDataType(at::ScalarType type) {
  auto it = ncclDataType.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != ncclDataType.end(),
      "Input tensor data type is not supported for NCCL process group: ",
      type);
  return it->second;
}

// 检查是否允许将复杂视图看作实数的辅助函数，根据不同的 ReduceOp 类型返回布尔值
bool complexViewAsRealAllowed(const ReduceOp reduceOp) {
  switch (reduceOp) {
    case ReduceOp::SUM:
      return true;
    case ReduceOp::AVG:
      return true;
    case ReduceOp::PREMUL_SUM:
      return true;
    case ReduceOp::UNUSED:
      return true;
    default:
      return false;
  }
  return false;
}

// 如果启用了 NCCL_PREMUL_SUM 支持，则定义一个模板函数 unpackPreMulSum
#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
template <typename T, ncclDataType_t dataType>
ncclRedOpRAII unpackPreMulSum(
    const ReduceOp& reduceOp,
    const ncclComm_t& comm) {
```  
函数签名，接受一个常量引用 `ncclComm_t` 类型的参数 `comm`。


  const auto* preMulSupplement =
      reinterpret_cast<NCCLPreMulSumSupplement*>(reduceOp.supplement_.get());
```  
将 `reduceOp.supplement_.get()` 返回的指针强制转换为 `NCCLPreMulSumSupplement*` 类型，存储在 `preMulSupplement` 中。


  ncclRedOp_t preMulSum;
```  
声明 `preMulSum` 变量，用于存储 `ncclRedOp_t` 类型的对象。


  bool has_tensor = preMulSupplement->tensor_factor.defined();
```  
检查 `preMulSupplement` 的 `tensor_factor` 是否已定义，结果存储在 `has_tensor` 中。


  auto residence = has_tensor ? ncclScalarDevice : ncclScalarHostImmediate;
```  
根据 `has_tensor` 的值选择 `ncclScalarDevice` 或 `ncclScalarHostImmediate`，将结果存储在 `residence` 变量中。


  const T* ptr_factor = has_tensor
      ? preMulSupplement->tensor_factor.const_data_ptr<T>()
      : nullptr;
```  
根据 `has_tensor` 的值，如果为真，则将 `preMulSupplement->tensor_factor` 的 `const_data_ptr<T>()` 赋给 `ptr_factor`，否则将 `nullptr` 赋给 `ptr_factor`。


  T scalar_factor = T(preMulSupplement->double_factor);
```  
将 `preMulSupplement->double_factor` 转换为类型 `T`，并存储在 `scalar_factor` 中。


  ncclRedOpCreatePreMulSum(
      &preMulSum,
      // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum
      // tells us that the scalar input is strictly a multiplier.
      /*scalar=*/has_tensor ? const_cast<T*>(ptr_factor) : &scalar_factor,
      dataType,
      residence,
      comm);
```  
调用 `ncclRedOpCreatePreMulSum` 函数，创建 `preMulSum` 对象，传递给定的参数：指向 `preMulSum` 的指针、标量值（根据 `has_tensor` 的值决定是 `ptr_factor` 或 `scalar_factor`）、数据类型 `dataType`、`residence` 和 `comm`。


  return ncclRedOpRAII(preMulSum, comm);
```  
返回通过 `ncclRedOpRAII` 构造的对象，传递 `preMulSum` 和 `comm` 作为参数。
}
#endif

// 根据指定的 reduceOp、input、dataType 和 comm 获取对应的 ncclRedOpRAII 对象
ncclRedOpRAII getNcclReduceOp(
    const ReduceOp& reduceOp,          // 函数参数：指定的 Reduce 操作类型
    at::Tensor& input,                 // 函数参数：输入的 Tensor 对象的引用
    const ncclDataType_t& dataType,    // 函数参数：数据类型的引用
    const ncclComm_t& comm) {          // 函数参数：通信句柄的引用
  try {
    if (input.scalar_type() == at::kBool) {  // 检查输入 Tensor 是否为布尔类型
      if (reduceOp == ReduceOp::SUM) {
        // 对于布尔 Tensor，将 SUM 映射为 MAX，因为二者都表示按位或操作。
        // 这是为了避免使用 SUM 时的溢出问题，因为我们使用 uint8 表示布尔值（参见 ncclDataType 映射）。
        return ncclMax;
      }
#ifdef NCCL_HAS_AVG
      if (reduceOp == ReduceOp::AVG) {
        // 如果 reduceOp 是 AVG，并且输入为布尔类型，则抛出类型错误异常
        C10_THROW_ERROR(
            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
      }
#endif
    }
    if (reduceOp == ReduceOp::PREMUL_SUM) {
#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
      // 根据 dataType 的类型选择相应的 unpackPreMulSum 函数进行处理
      switch (dataType) {
        case ncclHalf:
          return unpackPreMulSum<at::Half, ncclHalf>(reduceOp, comm);
        case ncclFloat:
          return unpackPreMulSum<float, ncclFloat>(reduceOp, comm);
        case ncclDouble:
          return unpackPreMulSum<double, ncclDouble>(reduceOp, comm);
        default:
          // 如果不是支持的数据类型，则抛出类型错误异常
          C10_THROW_ERROR(
              TypeError, "PreMulSum Data type must be half, float, or double");
          ncclRedOp_t unused;  // 未使用的变量，仅为了语法完整性
          return unused;
      }
#else
      // 如果不支持 PREMUL_SUM，则抛出值错误异常
      C10_THROW_ERROR(ValueError, "PreMulSum requires NCCL>=2.11.1");
#endif
    }
    // 返回对应于 reduceOp 的 ncclOp 对象
    return ncclOp.at(reduceOp);
  } catch (const std::out_of_range&) {
    // 捕获 std::out_of_range 异常
    switch (reduceOp) {
      case ReduceOp::AVG:
        // 如果 reduceOp 是 AVG，则抛出值错误异常，说明需要 NCCL 2.10+ 的版本
        C10_THROW_ERROR(
            ValueError,
            c10::str(
                "AVG requires NCCL 2.10+. The current version is ",
                NCCL_MAJOR,
                ".",
                NCCL_MINOR));
        break;
      case ReduceOp::BAND:
        // 如果 reduceOp 是 BAND，则抛出值错误异常，说明不能与 NCCL 一起使用
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with NCCL");
        break;
      case ReduceOp::BOR:
        // 如果 reduceOp 是 BOR，则抛出值错误异常，说明不能与 NCCL 一起使用
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with NCCL");
        break;
      case ReduceOp::BXOR:
        // 如果 reduceOp 是 BXOR，则抛出值错误异常，说明不能与 NCCL 一起使用
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with NCCL");
        break;
      default:
        // 如果 reduceOp 是其他未处理的类型，则抛出值错误异常，说明是未处理的 ReduceOp
        C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
        break;
    }
  }
}

// 从设备获取一个键字符串
inline std::string getKeyFromDevice(at::Device& device) {
  return std::to_string(device.index());  // 返回设备的索引转换成字符串
}

// 根据设备键获取设备索引
inline at::DeviceIndex getIndexFromDeviceKey(const std::string& deviceKey) {
  // 初始化设备索引为 -1，表示无效值
  int index = -1;
  try {
    index = std::stoi(deviceKey);  // 尝试将设备键转换为整数索引
  } catch (const std::invalid_argument& e) {
    // 捕获无效参数异常，并记录日志
    LOG(WARNING) << c10::str(
        "Invalid deviceKey: ", deviceKey, ",", e.what(), ".");
  } catch (const std::out_of_range& e) {
    // 捕获越界异常，并记录错误日志
    LOG(ERROR) << "Out of range: " << e.what();
  }
  return static_cast<at::DeviceIndex>(index);  // 返回转换后的设备索引
}
// 根据给定的两个整数 myRank 和 peer，确定较小的和较大的整数
int lowRank = myRank < peer ? myRank : peer;
int highRank = myRank < peer ? peer : myRank;

// 将确定的较小和较大整数转换为字符串，并用 ":" 连接起来，形成发送接收对
std::string sendRecvPair =
    std::to_string(lowRank) + ":" + std::to_string(highRank);
return sendRecvPair;



// 从给定的张量中获取设备信息
inline at::Device getDevice(at::Tensor& tensor) {
  return tensor.device();
}



// [同步流] 辅助函数，让输入的 ncclStreams 等待当前流。
// NCCL 通信在 ncclStreams 上运行，但输入张量分配在不同的流上（即当前流）。
// 在当前流上挂起的张量操作完成之前，不能开始在 ncclStreams 上的通信。
// 否则，两个流上的操作可能会同时读取/写入相同的张量。
//
// 单独的同步仅仅是不够的。我们还需要确保在 ncclStreams 上使用的输入张量在其使用完成之前不会被释放。
// 这可以通过调用 c10::cuda::CUDACachingAllocator::recordStream 来实现，
// 它会记住使用流（ncclStream），在垃圾收集尝试释放输入张量时在使用流上创建一个事件，并延迟垃圾收集直到该事件完成。
void syncStream(
    at::Device& device,
    at::cuda::CUDAEvent& ncclEvent,
    at::cuda::CUDAStream& ncclStream) {
  ncclEvent.record(at::cuda::getCurrentCUDAStream(device.index()));
  ncclEvent.block(ncclStream);
}



// 将给定的 ncclUniqueId 转换为可以放入存储中的字符串表示形式
std::string buildNcclUniqueIdStr(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (const auto i : c10::irange(NCCL_UNIQUE_ID_BYTES)) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}



// 根据给定的 ncclIdStr 构建用于存储的键
std::string getNcclAbortedCommStoreKey(const std::string ncclIdStr) {
  return std::string(kNCCLAbortedCommStoreKey) + ":" + ncclIdStr;
}



// 根据异常指针获取异常的 what() 描述信息
std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
  TORCH_CHECK(exceptionPtr != nullptr);
  try {
    std::rethrow_exception(exceptionPtr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}



// 检查是否捕获了不可捕获的 NCCL 错误
inline void errorIfCapturingNonCapturableNCCL(c10::cuda::CaptureStatus status) {
  // 使用括号避免某些编译器警告
  static const uint64_t min_version =
      (((uint64_t)2) << 32) + (((uint64_t)9) << 16) + ((uint64_t)6);
  static const uint64_t cur_version = torch::cuda::nccl::version();
  if (cur_version < min_version) {
    TORCH_CHECK_WITH(
        NotImplementedError,
        status == c10::cuda::CaptureStatus::None,
        "Capturing NCCL collectives is only allowed with NCCL >= 2.9.6");
  }
}



} // namespace



// 从每个通信器映射到其设备索引的映射表。
// 此映射在从缓存分配器注册/注销缓存段时使用。参见以下设计注释：
// - 每个段应仅注册到同一设备上的通信器。
// - 在点对点场景中，不能在每个进程组中重用 devNCCLCommMap_，因为键可能是 ranks 而不是设备。
// - 由于注册钩子在任何进程组的作用域之外被调用，因此必须将此映射维护为全局变量，因此需要在所有进程组中遍历通信器。
static std::unordered_map<std::shared_ptr<NCCLComm>, int> ncclCommDevIdxMap;
static std::mutex ncclCommDevIdxMapMutex;
static bool allocatorHooksAttached = false;

// 原子布尔值，用于指示是否应转储 NCCL 追踪信息
std::atomic<bool> ProcessGroupNCCL::shouldDump_(false);

// 注册缓存分配器的钩子函数，根据追踪条目执行注册操作
void cacheAllocatorRegisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // 在 SEGMENT_ALLOC 后注册
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommDevIdxMapMutex);
  for (auto& it : ncclCommDevIdxMap) {
    auto& ncclComm = it.first;
    auto& devIdx = it.second;
    // 如果设备索引匹配，则注册分配的段
    if (te.device_ == devIdx) {
      ncclComm->registerSegment(reinterpret_cast<void*>(te.addr_), te.size_);
    }
  }
}

// 注销缓存分配器的钩子函数，根据追踪条目执行注销操作
void cacheAllocatorDeregisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // 在 SEGMENT_FREE 前注销
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommDevIdxMapMutex);
  for (auto& it : ncclCommDevIdxMap) {
    auto& ncclComm = it.first;
    auto& devIdx = it.second;
    // 如果设备索引匹配，则注销释放的段
    if (te.device_ == devIdx) {
      ncclComm->deregisterSegment(reinterpret_cast<void*>(te.addr_));
    }
  }
}

#if defined(IS_NCCLX) && defined(NCCL_COMM_DUMP)
// 从所有通信器中获取 NCCL 跟踪信息并转储
std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  // 包含 NCCL 跟踪信息的映射，按照 ncclUniqueID 进行组织
  std::unordered_map<
      std::string /* ncclUniqueID */,
      std::unordered_map<std::string, std::string> /* dump from this comm */>
      ncclDumpMap;
  // dump_nccl_trace 仅从默认 PG（uid_=0）调用，但我们希望从所有通信器中转储，因此需要迭代 ncclCommDevIdxMap
  std::vector<std::shared_ptr<NCCLComm>> allNCCLComms;
  // 在临界区内部，不希望在持有锁时进行转储，因为转储可能会挂起
  ncclCommDevIdxMapMutex.lock();
  for (auto& [ncclComm, _] : ncclCommDevIdxMap) {
    allNCCLComms.push_back(ncclComm);
  }
  ncclCommDevIdxMapMutex.unlock();
  // 对所有通信器进行迭代，获取其 NCCL 轨迹转储信息
  for (auto& ncclComm : allNCCLComms) {
    std::string ncclUniqueIDStr = buildNcclUniqueIdStr(ncclComm->getNcclId());
    ncclDumpMap[ncclUniqueIDStr] = ncclComm->ncclCommDump();
  }
  // 返回 NCCL 跟踪缓冲区中的转储信息
  return NCCLTraceBuffer::get()->dump(
      ncclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

#else
// 如果未定义 IS_NCCLX 或者 NCCL_COMM_DUMP，则提供空的 dump_nccl_trace 实现
std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  return "";  // 或者其他适当的返回值，根据具体实现确定
}
#endif
    // 返回 NCCLTraceBuffer 单例对象的 dump 方法的执行结果
    // 参数 c10::nullopt 表示不传递任何值作为参数，includeCollectives 表示是否包含集合操作的跟踪信息，
    // includeStackTraces 表示是否包含堆栈跟踪信息，onlyActive 表示是否只包含活跃的跟踪信息
    return NCCLTraceBuffer::get()->dump(
        c10::nullopt, includeCollectives, includeStackTraces, onlyActive);
}
#endif

// 返回一个指向可选的函数对象的引用，该函数接受一个函数对象作为参数，并返回空值或者一个 void 类型的回调函数
std::optional<std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper() {
  // 静态局部变量，保存一个可选的函数对象，初始状态为 c10::nullopt
  static std::optional<
      std::function<void(std::function<void(const std::string&)>)>>
      dumper(c10::nullopt);
  return dumper;  // 返回保存的函数对象
}

// 返回一个静态的 gil_checker_t 类型的引用
gil_checker_t& get_gil_checker() {
  // 静态局部变量，保存一个 gil_checker_t 类型的空指针
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;  // 返回保存的空指针
}

// 启动一个异步的 GIL 检查任务，并返回一个表示任务结果的 future 对象
std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;  // 创建一个 promise 对象，用于设置任务结果
  std::future<bool> resultFuture = resultPromise.get_future();  // 获取与 promise 对象关联的 future 对象
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");  // 检查 gil_checker 是否为空

  // 创建一个新的线程来执行 GIL 检查任务
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    c10::setThreadName("pt_nccl_gil_chk");  // 设置线程名称

    try {
      auto& gil_checker = get_gil_checker();  // 获取 GIL 检查器对象的引用
      promise.set_value((*gil_checker)());  // 调用 GIL 检查器并设置 promise 的值
    } catch (...) {
      promise.set_exception(std::current_exception());  // 捕获异常并设置 promise 的异常
    }
  });

  // 分离线程以允许其独立运行
  workerThread.detach();

  return resultFuture;  // 返回 future 对象，用于获取异步任务的结果
}

// 返回推测的 CUDA 设备，根据输入的排名确定。如果绑定到特定设备，则返回该设备；否则根据排名确定设备索引
at::Device ProcessGroupNCCL::guessDeviceForRank() const {
  TORCH_CHECK_WITH(ValueError, rank_ >= 0, "Invalid rank ", rank_);  // 检查排名是否有效

  if (getBoundDeviceId()) {  // 如果已绑定设备 ID，则返回该设备
    return *getBoundDeviceId();
  } else {  // 否则根据排名确定设备索引并返回对应的 CUDA 设备
    int16_t deviceIdx = static_cast<int16_t>(rank_ % localDeviceCount_);
    return at::Device(at::DeviceType::CUDA, deviceIdx);
  }
}

// 静态成员变量初始化：监视线程睡眠时间
const int64_t ProcessGroupNCCL::kWatchdogThreadSleepMillis = 100;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
// 线程局部存储变量初始化：NCCL 活动组计数器
thread_local uint64_t ProcessGroupNCCL::ncclActiveGroupCounter_ = 0;

// 自定义输出运算符重载：用于将 ProcessGroupNCCL::WorkNCCL 对象输出到流中
std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupNCCL::WorkNCCL& workNCCL) {
  // 构造工作信息字符串
  std::string workInfo;
  workInfo = c10::str(
      "WorkNCCL(",
      "SeqNum=",
      workNCCL.seq_,
      ", OpType=",
      opTypeToString(workNCCL.opType_),
      ", NumelIn=",
      workNCCL.numelIn_,
      ", NumelOut=",
      workNCCL.numelOut_,
      ", Timeout(ms)=",
      workNCCL.opTimeout_.count(),
      ")");
  return output << workInfo;  // 将工作信息字符串输出到流中
}

// ProcessGroupNCCL::WorkNCCL 类的构造函数定义
ProcessGroupNCCL::WorkNCCL::WorkNCCL(
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool desyncDebug,
    bool enableTiming,
    DebugLevel distDebugLevel) {
    // 构造函数体，具体内容根据实际情况补充
}
    : Work(rank, opType, profilingTitle, inputs),  // 调用基类 Work 的构造函数，传递 rank、opType、profilingTitle 和 inputs 参数
      device_(device),  // 初始化成员变量 device_，使用传入的 device 参数
      workStartTime_(std::chrono::steady_clock::now()),  // 记录当前时间作为工作开始时间，使用 steady_clock 获取稳定的时钟时间点
      seq_(seq),  // 初始化成员变量 seq_，使用传入的 seq 参数
      timingEnabled_(enableTiming),  // 初始化成员变量 timingEnabled_，使用传入的 enableTiming 参数
      distDebugLevel_(distDebugLevel) {  // 初始化成员变量 distDebugLevel_，使用传入的 distDebugLevel 参数
  
  // 创建 CUDA 事件包装器
  // 注意：实际的事件在首次记录时才会被延迟创建，并且使用 DEFAULT_FLAGS = cudaEventDisableTiming。
  if (enableTiming) {  // 如果启用了时间跟踪
    ncclStartEvent_ = std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault);  // 创建一个 CUDAEvent 对象，共享指向 cudaEventDefault 的事件句柄
  }
  ncclEndEvent_ = std::make_shared<at::cuda::CUDAEvent>(  // 创建一个 CUDAEvent 对象，根据 enableTiming 决定使用 cudaEventDefault 或 cudaEventDisableTiming
      enableTiming ? cudaEventDefault : cudaEventDisableTiming);
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const WorkNCCL& w)
    : Work(w.rank_, w.opType_),  // 调用基类 Work 的拷贝构造函数，初始化 rank_ 和 opType_
      std::enable_shared_from_this<WorkNCCL>(w),  // 初始化 std::enable_shared_from_this，允许从该对象获取 shared_ptr
      device_(w.device_),  // 初始化 device_ 成员变量
      ncclStartEvent_(w.ncclStartEvent_),  // 初始化 ncclStartEvent_ 成员变量
      ncclEndEvent_(w.ncclEndEvent_),  // 初始化 ncclEndEvent_ 成员变量
      ncclComm_(w.ncclComm_),  // 初始化 ncclComm_ 成员变量
      blockingWait_(w.blockingWait_),  // 初始化 blockingWait_ 成员变量
      opTimeout_(w.opTimeout_),  // 初始化 opTimeout_ 成员变量
      workStartTime_(w.workStartTime_),  // 初始化 workStartTime_ 成员变量
      seq_(w.seq_),  // 初始化 seq_ 成员变量
      startTraceUpdated_(w.startTraceUpdated_),  // 初始化 startTraceUpdated_ 成员变量
      numelIn_(w.numelIn_),  // 初始化 numelIn_ 成员变量
      numelOut_(w.numelOut_),  // 初始化 numelOut_ 成员变量
      store_(w.store_),  // 初始化 store_ 成员变量
      timingEnabled_(w.timingEnabled_),  // 初始化 timingEnabled_ 成员变量
      trace_id_(w.trace_id_),  // 初始化 trace_id_ 成员变量
      distDebugLevel_(w.distDebugLevel_) {  // 初始化 distDebugLevel_ 成员变量
  exception_ = w.exception_;  // 拷贝异常状态
}

ProcessGroupNCCL::WorkNCCL::~WorkNCCL() = default;  // 默认析构函数实现

bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  if (!ncclComm_->isAborted()) {  // 如果 NCCL 通信未中止
    checkAndSetException();  // 检查并设置异常状态
  }
  return exception() || finishedGPUExecutionInternal();  // 返回是否有异常或 GPU 执行是否完成的状态
}

bool ProcessGroupNCCL::WorkNCCL::isStarted() {
  if (!ncclComm_->isAborted()) {  // 如果 NCCL 通信未中止
    checkAndSetException();  // 检查并设置异常状态
  }
  return exception() || startedGPUExecutionInternal();  // 返回是否有异常或 GPU 是否已开始执行的状态
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "WorkNCCL::isSuccess() is deprecated");  // 抛出未实现错误，标识该方法已过时
}

void ProcessGroupNCCL::WorkNCCL::checkAndSetException() {
  if (exception()) {
    // 已经有异常存在
    return;
  }

  auto exception_ptr = checkForNCCLErrors();  // 检查是否有 NCCL 错误并返回异常指针
  std::unique_lock<std::mutex> lock(mutex_);  // 使用互斥锁保护异常状态
  exception_ = exception_ptr;  // 设置异常状态
  if (exception_) {
    LOG(INFO) << logPrefix()  // 记录日志前缀
              << "found async exception when checking for NCCL errors: "
              << getExceptionMsgFromExceptionPtr(exception_);  // 记录异常消息
  }
}

const std::string& ProcessGroupNCCL::WorkNCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");  // 静态变量，返回日志前缀
  return prefix;
}

void ProcessGroupNCCL::WorkNCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);  // 使用互斥锁保护异常状态
  exception_ = exception_ptr;  // 设置异常状态
}

// 检查 NCCL 核心是否在 GPU 上执行完成的辅助函数
bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() {
  checkAndSetException();  // 检查并设置异常状态
  return finishedGPUExecutionInternal();  // 返回内部 GPU 执行是否完成的状态
}

bool ProcessGroupNCCL::WorkNCCL::startedGPUExecutionInternal() const {
  // 如果禁用了时间跟踪，我们将不会分配起始事件
  if (!timingEnabled_) {
    return false;
  }
  // 检查工作对应的 CUDA 事件的状态
  if (!ncclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const {
  // 检查工作对应的 CUDA 事件的状态
  if (!ncclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::checkTimeout(
    // 如果传入超时时间，则使用它；否则使用默认操作超时时间
    std::optional<std::chrono::milliseconds> timeout) {
      // 在作用域中创建一个静态计数器，用于跟踪调用此函数的次数
      STATIC_SCOPED_WAIT_COUNTER(
          pytorch.wait_counter.ProcessGroupNCCL__checkTimeout);
      // 获取当前时间点
      auto currentTimepoint = std::chrono::steady_clock::now();
      // 计算从工作开始到现在的时间间隔，并转换为毫秒
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - workStartTime_);
      // 确定工作超时时间，优先使用传入的超时时间，否则使用默认操作超时时间
      auto workTimeout = timeout ? *timeout : opTimeout_;
    
      // 如果工作未超时，则返回 false
      if (timeElapsed < workTimeout)
        return false;
    
      // 已超时
    
      // 如果已经有异常存在，则不覆盖它，直接返回 true
      if (exception())
        return true;
    
      // 构建超时异常消息字符串
      std::string exceptionMsg = c10::str(
          logPrefix(),
          "Watchdog caught collective operation timeout: ",
          *this,
          " ran for ",
          timeElapsed.count(),
          " milliseconds before timing out.");
    
      // 记录错误日志
      LOG(ERROR) << exceptionMsg;
      // 创建异常指针并设置异常
      std::exception_ptr exception_ptr =
          std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
      setException(exception_ptr);
      // 返回 true 表示出现超时
      return true;
    }
// 定义一个成员函数，用于处理异常情况
void ProcessGroupNCCL::WorkNCCL::handleException(
    ErrorHandlingMode errorHandling) {
  // 如果存在异常
  if (exception_) {
    // 构建异常信息字符串，说明可能存在 NCCL 操作失败或超时的情况，并且提醒后续的 GPU 操作可能会在损坏或不完整的数据上运行，这是由于 CUDA 核心的异步特性导致的
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    // 记录异常信息到日志
    LOG(ERROR) << logPrefix() << exceptionMsg;
    // 记录 API 使用情况，确保只记录一次
    C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.WorkNCCL.handleException");

    // 根据错误处理模式判断是否需要终止整个进程
    if (SHOULD_TEAR_DOWN(errorHandling)) {
      // 构建终止消息字符串，说明由于数据不一致性的风险，正在终止整个进程
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      // 记录终止消息到日志
      LOG(ERROR) << logPrefix() << tearDownMsg;
      // 重新抛出之前捕获的异常，这里从新线程抛出异常到主线程
      std::rethrow_exception(exception_);
    }
  }
}

// 定义一个成员函数，用于同步操作
void ProcessGroupNCCL::WorkNCCL::synchronize() {
  // 调用内部的同步方法，没有设置超时参数，避免向公共同步 API 添加超时参数
  synchronizeInternal(kNoTimeout);
}

// 定义一个成员函数，用于同步 CUDA 流
void ProcessGroupNCCL::WorkNCCL::synchronizeStream() {
  // 获取当前 CUDA 流
  auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
  // 阻塞当前流在 NCCL 流上
  ncclEndEvent_->block(currentStream);

  // 如果需要避免记录流状态
  if (avoidRecordStreams_) {
    // 清空用于分配器安全性的预留数据
    stashed_for_allocator_safety_->clear();
  }
}

// 定义一个成员函数，用于内部同步操作
// 参数 timeout 表示超时时间
void ProcessGroupNCCL::WorkNCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  // 同步 CUDA 流
  synchronizeStream();

  // 如果是阻塞等待模式
  if (blockingWait_) {
    // 当操作未完成时循环等待
    while (!isCompleted()) {
      // 检查是否超时，如果设置了超时时间，则进行超时检查
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? c10::nullopt : c10::make_optional(timeout));
      // 如果超时
      if (timedOut) {
        // 构建超时异常信息字符串，包含工作对象信息和阻塞等待的标志信息
        std::string exceptionMsg = c10::str(
            logPrefix(),
            "Work ",
            (*this),
            " timed out in blocking wait (TORCH_NCCL_BLOCKING_WAIT=1).");
        // 记录超时异常信息到日志
        LOG(ERROR) << exceptionMsg;
        // 跳出循环
        break;
      }
      // 休眠一段时间，避免忙等待，时间为 kSynchronizeBusyWaitMillis 毫秒
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    // 如果存在异常
    if (exception()) {
      // 中止 NCCL 通信器
      abort();
      // 处理异常，可能会终止整个进程
      handleException(TearDown);
    }
  }

  // 在完成超时检查后，设备同步操作
  if (barrierTensor_.defined()) {
    // 如果使用工作对象执行屏障操作，需要在此处阻塞
    // `dist.barrier()` 只要求所有 CPU 进程进入此函数，因此我们只需确保虚拟的全局归约操作已经完成。因此，我们只需要同步**当前流**返回
    // 到
    // 获取当前设备上的 CUDA 流
    auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
    // 使用 cudaStreamSynchronize 而不是 cudaDeviceSynchronize 可以：
    // - 降低挂起的可能性；
    // - currentStream 通常是下一个操作的上下文，因此阻塞当前流可能已经阻塞了下一个计算核心；
    // - 实现更好的屏障性能。
    AT_CUDA_CHECK(cudaStreamSynchronize(currentStream));
}

// WorkNCCL 类的 wait 方法，等待操作完成，并且返回是否成功
bool ProcessGroupNCCL::WorkNCCL::wait(std::chrono::milliseconds timeout) {
  // 记录通信参数，用于性能分析和调试
  RECORD_PARAM_COMMS(
      static_cast<int>(this->seq_), // seq，操作序列号
      std::make_tuple("", ""), // PG name tuple，进程组名称元组
      rank_, // rank，进程在组中的排名
      "wait", // collective name，集合操作的名称
      0, // inNelems，输入元素数
      0, // outNelems，输出元素数
      at::kByte, // dType，数据类型
      std::vector<int64_t>(), // inSplitSizes，输入分割大小
      std::vector<int64_t>(), // outSplitSizes，输出分割大小
      -1,
      -1,
      static_cast<int>(1)); // number of device?，设备数量？
  // 调用内部的同步方法，等待操作完成
  synchronizeInternal(timeout);
  // TODO(kwen2501): this should be moved to c10d tests, to qualify a NCCL
  // upgrade. Once a NCCL version is qualified, this code should not be needed
  // at runtime.
#ifdef PGNCCL_ENABLE_HASH
  // 如果启用了哈希调试模式且调试级别为详细，则打印输出的哈希签名
  if (distDebugLevel_ >= DebugLevel::Detail) {
    auto numel = getTensorsNumel(*outputs_);
    auto hashValue = hashTensors(*outputs_);
    PRINT_COLLECTIVE_HASH_SIGNATURE(
        "output", opTypeToString(opType_), numel, hashValue);
  }
#endif
  // 总是返回 true，因为中止 API 未实现
  return true;
}

// WorkNCCL 类的 abort 方法，中止当前工作的所有通信器
void ProcessGroupNCCL::WorkNCCL::abort() {
  // 中止当前工作的所有通信器
  ncclComm_->ncclCommAbort();

  // 获取和更新通信器到设备索引映射的互斥锁
  ncclCommDevIdxMapMutex.lock();
  ncclCommDevIdxMap.erase(ncclComm_);
  ncclCommDevIdxMapMutex.unlock();
}

// 静态变量，用于分配进程组的唯一标识符
static std::atomic<size_t> process_group_id = 0;

// 多设备错误消息的常量字符串
constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupNCCL continues supporting multi-process and multi-thread modes.";

// ProcessGroupNCCL 类的构造函数，初始化一个 NCCL 进程组
ProcessGroupNCCL::ProcessGroupNCCL(
    const c10::intrusive_ptr<Store>& store, // 存储器
    int rank, // 进程的排名
    int size, // 进程组的大小
    c10::intrusive_ptr<Options> options) // 选项
    // 调用 Backend 构造函数初始化基类
    : Backend(rank, size),
      // 初始化存储对象
      store_(store),
      // 初始化选项对象
      options_(options),
      // 初始化 NCCL 通信计数器
      ncclCommCounter_(0),
      // 根据当前进程的排名获取追踪开始的键值
      traceKeyStart_(getTraceStartKey("NCCL", rank)),
      // 根据当前进程的排名获取追踪结束的键值
      traceKeyEnd_(getTraceEndKey("NCCL", rank)),
      // 终止进程组的标志，默认为 false
      terminateProcessGroup_(false),
      // 终止心跳监视线程的标志，默认为 false
      terminateHeartbeatMonitorThread_(false),
      // 集合调试信息模式，默认为 false
      collectiveDebugInfoMode_(false),
      // 设置当前进程组的唯一标识符，递增
      uid_(process_group_id++),
      // 初始化节点内部通信对象
      intraNodeComm_(initIntraNodeComm()) {
  // 检查是否存在 CUDA GPU，否则抛出错误
  TORCH_CHECK_WITH(
      ValueError,
      at::cuda::getNumGPUs() != 0,
      "ProcessGroupNCCL is only supported with GPUs, no GPUs found!");
  // 设置进程组的名称
  this->setGroupName(options_->group_name);
  // 获取本地 CUDA 设备的数量
  this->localDeviceCount_ = at::cuda::getNumGPUs();
  // 创建日志前缀
  logPrefix_ = createLogPrefix();
  // 获取是否使用阻塞等待的标志位
  blockingWait_ = getCvarBool(TORCH_NCCL_BLOCKING_WAIT, false);
  // 获取异步错误处理模式
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      getCvarInt(TORCH_NCCL_ASYNC_ERROR_HANDLING, 3 /*SkipCleanUp*/));
  // 获取是否启用解同步调试
  desyncDebug_ = getCvarBool(TORCH_NCCL_DESYNC_DEBUG, false) ||
      (dist_debug_level_ >= DebugLevel::Detail);
  // 获取是否重新抛出 CUDA 错误
  rethrowCUDAErrors_ = getCvarBool(TORCH_NCCL_RETHROW_CUDA_ERRORS, true);
  // 获取是否在超时时转储异常信息
  // TODO，应该考虑废弃 TORCH_NCCL_DUMP_ON_TIMEOUT 或者修改其名称以反映在超时和其他错误时都进行转储的情况
  dumpOnException_ = getCvarBool(TORCH_NCCL_DUMP_ON_TIMEOUT, false) ||
      (dist_debug_level_ >= DebugLevel::Detail);
  // 获取是否启用 NaN 检查
  enableNanCheck_ = getCvarBool(TORCH_NCCL_NAN_CHECK, false);
  // 设置心跳值
  heartbeat_ = 1ULL;
  // 设置监视线程是否启用的原子标志位
  monitorThreadEnabled_.store(getCvarBool(TORCH_NCCL_ENABLE_MONITORING, true));
  // 设置心跳超时时间（秒）
  heartbeatTimeoutInSec_ =
      getCvarInt(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC, 60 * 10 /*10 Mins*/);
  // 设置等待超时时的转储时间（毫秒）
  waitTimeoutDumpInMilSec_ =
      getCvarInt(TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC, 60 * 1000 /*60 Sec*/);
  // 设置协调检查间隔时间（毫秒）
  coordCheckIntervalMilSec_ = getCvarInt(TORCH_NCCL_COORD_CHECK_MILSEC, 1000);
  // 设置 NCCL 追踪缓冲区大小
  ncclTraceBufferSize_ = getCvarInt(TORCH_NCCL_TRACE_BUFFER_SIZE, 0);
  // 是否启用集合哈希调试模式
  enableCollecticeHashDebug_ = (dist_debug_level_ >= DebugLevel::Detail);
  // 通常 store_ 被包装在 PrefixStore 中，并且不同 ProcessGroupNCCL 实例的前缀不同
  // 我们需要获取基础的非前缀存储来共享跨不同的进程组实例的全局信息
  PrefixStore* prefixStore = dynamic_cast<PrefixStore*>(store_.get());
  globalStore_ =
      prefixStore ? prefixStore->getUnderlyingNonPrefixStore() : store_;
#ifdef ENABLE_NCCL_ERROR_CHECKING
  // 如果启用了 NCCL 错误检查，则根据 TORCH_NCCL_ENABLE_TIMING 和 desyncDebug_ 的值来设置 enableTiming_
  enableTiming_.store(
      getCvarBool(TORCH_NCCL_ENABLE_TIMING, false) || desyncDebug_);
#endif
  // 根据 TORCH_NCCL_AVOID_RECORD_STREAMS 的值来设置 avoidRecordStreams_
  avoidRecordStreams_ = getCvarBool(TORCH_NCCL_AVOID_RECORD_STREAMS, false);
#ifdef NCCL_HAS_COMM_REGISTER
  // 如果支持 NCCL 通信注册，则根据 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK 的值来设置 useTensorRegisterAllocatorHook_
  useTensorRegisterAllocatorHook_ =
      getCvarBool(TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK, false);
  // 如果 CUDA 分配器支持可扩展段，则禁用 useTensorRegisterAllocatorHook_
  if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
          expandable_segments()) {
    useTensorRegisterAllocatorHook_ = false;
    LOG(INFO)
        << logPrefix()
        << "disables TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK because it is not compatible with CUDA allocator expandable segments mode.";
  }
#endif

  // 如果启用了 blockingWait_
  if (blockingWait_) {
    // 如果 asyncErrorHandling_ 不是 NoHandling 或 desyncDebug_ 为真，则输出警告信息
    if (asyncErrorHandling_ != NoHandling || desyncDebug_) {
      LOG(INFO)
          << logPrefix() << "TORCH_NCCL_BLOCKING_WAIT and "
          << "TORCH_NCCL_ASYNC_ERROR_HANDLING|TORCH_NCCL_DESYNC_DEBUG"
          << "should not both be enabled. "
          << "Only TORCH_NCCL_BLOCKING_WAIT is being used in this process.";
      asyncErrorHandling_ = NoHandling;
      desyncDebug_ = false;
    }
  } else {
    // 如果 desyncDebug_ 为真且 asyncErrorHandling_ 为 NoHandling，则输出警告信息
    if (desyncDebug_ && asyncErrorHandling_ == NoHandling) {
      LOG(INFO)
          << logPrefix()
          << "TORCH_NCCL_DESYNC_DEBUG and TORCH_NCCL_ASYNC_ERROR_HANDLING "
          << "must both be enabled. "
          << "Enabling TORCH_NCCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = SkipCleanUp;
    }
  }

#ifdef ENABLE_NCCL_ERROR_CHECKING
  // 创建一个线程用于监视 ncclCommWatchdog
  ncclCommWatchdogThread_ =
      std::thread(&ProcessGroupNCCL::ncclCommWatchdog, this);
#endif

  // 初始化
  init();
  const std::string OFF = "OFF";
  // 获取 TORCH_DISTRIBUTED_DEBUG 的值
  std::string torch_distributed_debug =
      getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, OFF.c_str());
  // 输出初始化选项信息
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL initialization options: "
            << "size: " << size << ", global rank: " << globalRank()
            << ", TIMEOUT(ms): " << options_->timeout.count()
            << ", USE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << ", SPLIT_FROM: " << options_->split_from
            << ", SPLIT_COLOR: " << options_->split_color
            << ", PG Name: " << options_->group_name;

  // 输出环境信息
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL environments: "
            << "NCCL version: " << getNcclVersion()
            << ", TORCH_NCCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << ", TORCH_NCCL_DUMP_ON_TIMEOUT: " << dumpOnException_
            << ", TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: "
            << waitTimeoutDumpInMilSec_
            << ", TORCH_NCCL_DESYNC_DEBUG: " << desyncDebug_
            << ", TORCH_NCCL_ENABLE_TIMING: " << enableTiming_.load()
            << ", TORCH_NCCL_BLOCKING_WAIT: " << blockingWait_
            << ", TORCH_DISTRIBUTED_DEBUG: " << torch_distributed_debug
#ifdef NCCL_HAS_COMM_REGISTER
            << ", TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: "
            << useTensorRegisterAllocatorHook_
#endif
            << ", TORCH_NCCL_ENABLE_MONITORING: "
            << monitorThreadEnabled_.load()
            << ", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: " << heartbeatTimeoutInSec_
            << ", TORCH_NCCL_TRACE_BUFFER_SIZE: " << ncclTraceBufferSize_
            << ", TORCH_NCCL_COORD_CHECK_MILSEC: " << coordCheckIntervalMilSec_
            << ", TORCH_NCCL_NAN_CHECK: " << enableNanCheck_;

  // 将一系列的配置信息连接成一个日志记录，用于输出调试信息
  if (options_->global_ranks_in_group.empty()) {
    // 如果全局组中的排名列表为空，则起始全局排名为0
    this->globalRankStart = 0;
  } else {
    // 否则，起始全局排名为全局组中的第一个排名
    this->globalRankStart = options_->global_ranks_in_group[0];
  }

  if (options_->global_ranks_in_group.empty()) {
    // 如果全局组中的排名列表为空，则全局排名步长为1
    this->globalRankStride = 1;
  } else if (options_->global_ranks_in_group.size() == 1) {
    // 如果全局组中只有一个排名，则全局排名步长为0
    this->globalRankStride = 0;
  } else {
    // 否则，计算全局排名的步长
    bool ranksAreStrided = true;
    int startRank = options_->global_ranks_in_group[0];
    int stride =
        options_->global_ranks_in_group[1] - options_->global_ranks_in_group[0];
    for (std::vector<uint64_t>::size_type i = 0;
         i < options_->global_ranks_in_group.size();
         i++) {
      // 检查全局组中的排名是否按照步长递增
      if (options_->global_ranks_in_group[i] != startRank + i * stride) {
        ranksAreStrided = false;
        break;
      }
    }

    if (ranksAreStrided) {
      // 如果全局组中的排名按照步长递增，则设置全局排名步长为计算出的步长值
      this->globalRankStride = options_->global_ranks_in_group[1] -
          options_->global_ranks_in_group[0];
    } else {
      // 否则，设置全局排名步长为-1，表示不是按步长递增
      this->globalRankStride = -1;
    }
  }

  // 将缓存分配器的钩子附加到跟踪器，以在发生特定动作时触发钩子操作
  // 只在第一次创建进程组时附加钩子，确保 CUDA 缓存分配器已初始化
  if (useTensorRegisterAllocatorHook_ && !allocatorHooksAttached) {
    at::globalContext().lazyInitCUDA();
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorRegisterHook);
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorDeregisterHook);
    allocatorHooksAttached = true;
  }
}

void ProcessGroupNCCL::eagerConnectSingleDevice(at::Device device) {
  // 获取设备对应的键，并记录连接日志
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Eagerly connecting nccl backend with device "
            << device;
  // 获取与设备相关的 NCCL 通信器，用于 ALLREDUCE 操作
  getNCCLComm(key, device, OpType::ALLREDUCE);
}

void ProcessGroupNCCL::performNocolorSplit(at::Device device) {
  // 如果后端不支持分组，对于不在新子组中的排名，此操作无效
  // 在应该在其中的排名将使用新的通信器而不是分组
#ifdef NCCL_HAS_COMM_SPLIT
  // 检查是否定义了 NCCL_HAS_COMM_SPLIT 宏
  const auto key = getKeyFromDevice(device);
  // 从设备获取键值
  LOG(INFO) << logPrefix() << "Performing nocolor split on backend device "
            << device << ", key " << key << ", i am " << this;
  // 记录日志，指示在后端设备上执行无颜色拆分操作
  auto comm = getNCCLComm(key, device, OpType::ALLREDUCE);
  // 获取 NCCL 通信句柄
  NCCLComm::split(
      comm.get(),
      NCCL_SPLIT_NOCOLOR,
      rank_,
      options_->config,
      options_->global_ranks_in_group);
  // 使用 NCCLComm 类进行无颜色拆分操作
#endif
}

c10::intrusive_ptr<intra_node_comm::IntraNodeComm> ProcessGroupNCCL::
    initIntraNodeComm() {
  // 初始化进程组内部通信模块
  using IntraNodeComm = intra_node_comm::IntraNodeComm;
  if (!IntraNodeComm::isEnabled()) {
    // 如果进程组内部通信模块未启用，返回空指针
    return nullptr;
  }
  auto prefixStore = c10::make_intrusive<PrefixStore>("IntraNodeComm", store_);
  // 创建前缀存储对象
  auto comm = c10::make_intrusive<IntraNodeComm>(prefixStore, rank_, size_);
  // 创建进程组内部通信对象
  if (comm->rendezvous()) {
    // 如果成功创建并进行会合，返回通信对象
    return comm;
  } else {
    // 否则返回空指针
    return nullptr;
  }
}

void ProcessGroupNCCL::setSequenceNumberForGroup() {
  // 设置进程组的序列号，NCCL 从 0 开始
} // NCCL just starts sequence numbers at 0.

uint64_t ProcessGroupNCCL::getSequenceNumberForGroup() {
  // 获取进程组的序列号
  return seqCollective_;
}

void ProcessGroupNCCL::registerOnCompletionHook(
    std::function<void(std::shared_ptr<WorkInfo>)>&& hook) {
  // 注册完成时的钩子函数
  TORCH_CHECK_WITH(
      DistBackendError,
      onCompletionHook_ == nullptr,
      "ProcessGroupNCCL OnCompletion hook already registered");
  // 检查是否已经注册了完成时的钩子函数

  TORCH_CHECK_WITH(
      ValueError,
      enableTiming_.load(),
      "ProcessGroupNCCL OnCompletion hook requires recording start and end "
      "events which require setting TORCH_NCCL_ENABLE_TIMING environment variable. "
      "This is only available for NCCL version >= 2.4.");
  // 检查是否启用了计时功能

  onCompletionHook_ = std::move(hook);
  // 移动钩子函数到成员变量中

  onCompletionHookThread_ = std::thread(&ProcessGroupNCCL::runHookLoop, this);
  // 启动运行钩子函数的线程
}

// must release GIL when calling this method
void ProcessGroupNCCL::waitForPendingWorks() {
  // 等待所有未完成的工作项

  // Reasoning about hook completion:
  // 1. waitForPendingWorks should be called after user code has finished
  // calling
  //    all collectives. This means, when we got here, all of the collectives
  //    are either in workMetaList_ or has been erased from workMetaList_.
  // 2. The watchdog thread grabs both locks to move Work object from the
  //    workMetaList_ to the completedWorkList_, and the hook thread only erases
  //    a Work object after the hook is returned. Therefore, after user code
  //    calls a collective, its Work object is either in workMetaList_ or in
  //    completedWorkList_ before it finishes.
  // 3. We have three threads and two locks.
  //      a. main thread (this function) grabs two locks atomically
  //      b. watchdog thread (watchdogHandler function) always grabs
  //      workMetaListMutex_
  //         first and then grabs completedWorkListMutex_.
  //      c. hook thread (runHookLoop function) only grabs
  //      completedWorkListMutex_. Therefore, locks are always acquired in the
  //      same order and hence no deadlocks.
  // 等待所有未完成的工作项的理由和线程锁的说明
  while (true) {
    {
      // 同时锁定 workMetaListMutex_ 和 completedWorkListMutex_ 两个互斥量，确保线程安全
      std::lock(workMetaListMutex_, completedWorkListMutex_);
      // 使用 lock_guard 自动管理互斥量的锁，避免忘记解锁或异常情况下的资源泄露
      std::lock_guard<std::mutex> lockWork(workMetaListMutex_, std::adopt_lock);
      // 使用 lock_guard 自动管理互斥量的锁，避免忘记解锁或异常情况下的资源泄露
      std::lock_guard<std::mutex> lockHook(
          completedWorkListMutex_, std::adopt_lock);
    
      // 如果 workMetaList_ 和 completedWorkList_ 都为空，则直接返回，避免不必要的操作
      if (workMetaList_.empty() && completedWorkList_.empty()) {
        return;
      }
    }
    
    // 当前线程休眠指定的时间间隔，以毫秒为单位
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kWatchdogThreadSleepMillis));
    }
}

void ProcessGroupNCCL::enableCollectivesTiming() {
  // 启用集合操作的定时功能
  enableTiming_.store(true);
}

void ProcessGroupNCCL::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    bool throwException) {
  // 准备存储错误消息
  std::string errorMsg;
  // 检查未来值是否有效
  TORCH_CHECK(fut.valid(), "Expected a valid future");
  // 等待未来值或超时
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // 如果未来值准备好，尝试获取其结果
    try {
      bool result = fut.get();
      if (result) {
        // 如果操作成功完成，记录日志
        LOG(INFO) << logPrefix()
                  << "future is successfully executed for: " << futDescription;
      }
    } catch (const std::exception& e) {
      // 捕获异常，记录错误消息
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waitng for future ",
          futDescription,
          ": ",
          e.what());
      LOG(ERROR) << errorMsg;
    } catch (...) {
      // 捕获未知异常，记录错误消息
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waitng for future ",
          futDescription);
      LOG(ERROR) << errorMsg;
    }
  } else {
    // 如果超时，记录超时消息
    errorMsg = c10::str(
        logPrefix(),
        "Future for ",
        futDescription,
        " timed out after ",
        timeOutMilSec.count(),
        " ms");
    LOG(ERROR) << errorMsg;
  }
  // 如果设置了抛出异常标志且存在错误消息，则抛出异常
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
}

void ProcessGroupNCCL::abortCommsFromMap(
    std::unordered_map<std::string, std::shared_ptr<NCCLComm>>& ncclCommsMap,
    std::optional<std::string> abortReason) {
  // 可能控制多个设备，循环处理每个设备上的通信器
  for (auto& it : ncclCommsMap) {
    auto& devName = it.first;
    auto& ncclComm = it.second;
    at::cuda::OptionalCUDAGuard gpuGuard;
    // 获取设备索引并设置 CUDA 上下文
    at::DeviceIndex deviceIndex = getIndexFromDeviceKey(devName);
    if (deviceIndex >= 0) {
      gpuGuard.set_index(deviceIndex);
    }
    // 记录销毁通信器的日志信息
    LOG(INFO) << logPrefix() << "ProcessGroupNCCL destroying ncclComm_ "
              << ncclComm->ncclComm_ << " on CUDA device: " << devName;
    // 中止通信器的操作
    ncclComm->ncclCommAbort(abortReason);
    // 注意：不移除中止的通信器缓存，以便应用程序能够处理错误和恢复
    // 这是因为在某些情况下，移除通信器可能导致不一致的状态
    c10::StreamId streamId = -1;
    // 检查是否存在与设备关联的流
    if (ncclStreams_.find(devName) != ncclStreams_.end()) {
      auto stream = ncclStreams_.at(devName);
      streamId = stream.id();
    }
    // 记录信息级别日志，输出处理组 NCCL 被销毁的消息，包括 CUDA 设备名和流 ID
    LOG(INFO) << logPrefix() << "ProcessGroupNCCL destroyed "
              << " communicator on CUDA device: " << devName
              << " with stream: " << streamId;
}

// 中止当前进程组的所有通信器
bool ProcessGroupNCCL::abort(std::optional<std::string> abortReason) {
  // 在中止之前，从全局的 ncclCommDevIdxMapMutex 中移除记录，
  // 以防止新的缓存段尝试注册到已中止的通信器上。
  // 注意，ncclCommDevIdxMap 是一个全局容器，可能包含其他进程组的通信器，
  // 因此我们只需要删除当前进程组的通信器。
  ncclCommDevIdxMapMutex.lock();
  for (auto& it : devNCCLCommMap_) {
    auto& ncclComm = it.second;
    ncclCommDevIdxMap.erase(ncclComm);
  }
  ncclCommDevIdxMapMutex.unlock();

  std::lock_guard<std::mutex> lock(mutex_);
  // 调用私有函数，从 devNCCLCommMap_ 和 inInitializationCommMap_ 中
  // 中止所有的通信器。
  abortCommsFromMap(devNCCLCommMap_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, abortReason);
  return true;
}

void ProcessGroupNCCL::shutdown(std::optional<std::string> reason) {
  // 不要在这里加入线程，因为该方法的目的是中止所有通信器并信号线程退出。
  // 在这个方法中加入线程可能会导致阻塞，因此需要避免。
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();

  // 异步地启动中止操作，并等待其完成或超时
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupNCCL abort asynchrounously.";
  std::future<bool> fut = std::async(
      std::launch::async, [this, &reason]() { return this->abort(reason); });

  waitForFutureOrTimeout(fut, options_->timeout, "ProcessGroup abort", true);
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL aborts successfully.";

  // 在安全关闭心跳监控线程之前，需要等待中止操作完成。
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL destructor entered.";

  if (!terminateProcessGroup_.load()) {
    if (rank_ % localDeviceCount_ == 0) {
      TORCH_WARN_ONCE(
          "WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. ",
          "On normal program exit, the application should call destroy_process_group to ",
          "ensure that any pending NCCL operations have finished in this process. "
          "In rare cases this process can exit before this point and block the progress of "
          "another member of the process group. This constraint has always been present, "
          " but this warning has only been added since PyTorch 2.4");
    }
    // 如果用户没有显式销毁或关闭进程组，析构函数需要执行关闭操作。
    shutdown();
  }

  // 在返回之前等待所有线程完成
#ifdef ENABLE_NCCL_ERROR_CHECKING
  if (ncclCommWatchdogThread_.joinable()) {
    ncclCommWatchdogThread_.join();
    LOG(INFO) << logPrefix() << "ProcessGroupNCCL watchdog thread joined.";
  }
  if (ncclHeartbeatMonitorThread_.joinable()) {
    ncclHeartbeatMonitorThread_.join();
    LOG(INFO) << logPrefix()
              << "ProcessGroupNCCL heart beat monitor thread joined.";



# 记录信息级别日志，使用 << 运算符连接以下内容：
# 调用 logPrefix() 函数获取日志前缀，
# 字符串常量 "ProcessGroupNCCL heart beat monitor thread joined."
# 整体记录为日志消息。
#endif
  // 如果onCompletionHookThread_是可加入的，则等待其完成，并记录日志
  if (onCompletionHookThread_.joinable()) {
    onCompletionHookThread_.join();
    LOG(INFO) << logPrefix()
              << "ProcessGroupNCCL onCompletionHookThread thread joined.";
  }
}

bool ProcessGroupNCCL::dumpDebuggingInfo() {
  // 对调用此函数的所有调用进行串行化，以避免数据损坏，但允许在一个运行时中多次调用。
  // 用户需确保在后续调用覆盖之前保留了早期调用的输出文件。
  static std::mutex writeDebugInfoMutex;
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  LOG(ERROR) << logPrefix() << "ProcessGroupNCCL preparing to dump debug info.";
  if (ncclTraceBufferSize_ > 0) {
    // 默认将nccl跟踪信息转储到本地磁盘，用户可以通过继承`DebugInfoWriter`并通过`registerDebugInfoWriter`注册自定义写入器。
    auto ncclTrace = dump_nccl_trace(true, true, false);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    LOG(INFO) << logPrefix() << "ProcessGroupNCCL dumping nccl trace to "
              << writer.getWriterTarget();
    writer.write(ncclTrace);
    return true;
  }
  return false;
}

void ProcessGroupNCCL::terminateProcess(std::string errMsg) {
  // 使用`FATAL`级别记录日志，打印错误消息后，调用`std::abort()`终止程序执行。
  LOG(FATAL) << logPrefix() << errMsg;
}

int computeDeltaMS(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end) {
  // 计算两个时间点之间的毫秒数差距
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void ProcessGroupNCCL::heartbeatMonitor() {
  c10::setThreadName("pt_nccl_heartbt");

  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitMsg;
  // 根据条件选择监控间隔
  bool checkDumpSignal = (dumpOnException_ && uid_ == 0);
  int monitorPollInterval = checkDumpSignal ? coordCheckIntervalMilSec_
                                            : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  std::optional<DumpPipe> dumpPipe = c10::nullopt;
  if (uid_ == 0) {
    // 每个trainer进程有一个DumpPipe，按全局rank命名，假设processgroup(uid)==0为全局PG，并在训练器间具有全局唯一的rank id。
    dumpPipe.emplace(rank_);
  }
  while (true) {
    // 由于此锁仅在此处使用，因此此处不会有任何锁。请注意，`monitorMutex_`互斥体不应在其他地方使用，以避免死锁。
    std::unique_lock<std::mutex> lock(monitorMutex_);
    // 等待心跳监控线程的条件变量，超时时间为 monitorPollInterval 毫秒，或者收到终止信号
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // 对于正常完成或用户中断，monitorWakeUpCV_ 会被通知，此处早期返回并退出 heartbeatMonitor
      return;
    }
    // 获取当前时间点
    auto currentTime = std::chrono::steady_clock::now();

    // 为默认的 PG（即 uid_=0）线程添加额外功能，因为信号在不同 PG 中相同。
    // 我们只需要每个进程运行一次，以避免在太多独立线程中执行重复的操作。
    // 例如，我们定期检查 TCPStore 上的全局标志，以查看任何 PG 是否观察到超时并向对等体发信号来转储调试信息，
    // 并且我们避免从同一秩的所有 PG 中过多地访问 TCPStore。
    
    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000) {
      // 检查看门狗线程的心跳。
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = heartbeat_.load();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        // 如果检测到没有心跳增加且超时
        if (!shouldDump_.load()) {
          LOG(ERROR)
              << logPrefix()
              << "First PG on this rank that detected no heartbeat of its watchdog.";
        }
        shouldDump_.store(true);
        // 心跳监控超时，将在转储调试信息后终止进程
        errorMsg = c10::str(
            logPrefix(),
            "Heartbeat monitor timed out! Process will be terminated after dumping debug info.",
            " workMetaList_.size()=",
            workMetaList_.size());
        exitMsg = c10::str(
            "ProcessGroupNCCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued collectives. ",
            "This typically indicates a NCCL/CUDA API hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "CUDA api, or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer timeout would help, ",
            "you can either increase the timeout (TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_NCCL_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the hang. "
            "workMetaList_.size() = ",
            workMetaList_.size(),
            "");
        break;
      }
    }
    // 处理转储跟踪请求。只有 PG uid 为 0 的进程会响应转储请求，
    // 但这没问题，因为所有 PG 都将数据馈送到同一个飞行记录器并进行转储。转储后，训练应该继续进行。
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // 检查是否存在要转储的管道，并且应该进行转储
      // 最佳尝试转储，此处不等待转储完成
      std::future<bool> fut = std::async(
          std::launch::async, [this]() { return this->dumpDebuggingInfo(); });
    }
  }
  // 记录错误消息到日志
  LOG(ERROR) << errorMsg;

  auto& cpp_dumper = get_cpp_trace_dumper();
  if (cpp_dumper.has_value()) {
    // 获取 C++ 堆栈跟踪转储器，如果存在则进行转储
    LOG(INFO) << "Dumping c++ stacktraces:";
    cpp_dumper.value()([](const std::string& line) { LOG(INFO) << line; });
  }

  if (checkDumpSignal && shouldDump_.load()) {
    // 如果需要检查转储信号并且应该进行转储
    // 将调试信息异步存储到指定位置（默认是本地磁盘）
    std::future<bool> asyncDebugDump = std::async(
        std::launch::async, [this]() { return this->dumpDebuggingInfo(); });

    // 等待转储完成或超时
    waitForFutureOrTimeout(
        asyncDebugDump,
        std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
        "Flight recorder dump in heartbeatMonitor");
  }

  if (get_gil_checker() != nullptr) {
    // 如果存在 GIL 检查器，启动异步 GIL 检查
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      // 如果 GIL 检查超时，记录错误日志
      TORCH_CHECK(
          futStatus != std::future_status::deferred,
          "Expected the future to have been launched eagerly.");
      LOG(ERROR)
          << "Could not acquire GIL within 300 ms on exit, possible GIL induced hang";
    }
    // 记录信息日志，说明成功获取 GIL
    LOG(INFO) << "Could acquire GIL on exit";
  } else {
    // 如果不存在 GIL 检查器，记录信息日志
    LOG(INFO)
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // 有两种可能的守护线程退出情况：
  // 情况一：异步报告运行快速，并遵循以下步骤：
  // 集体超时 -> 不同步 -> 异常处理 -> 析构函数
  // -> 设置 terminateHeartbeatMonitorThread_ -> 通知 monitorWakeUpCV_.
  // 因此，以上代码要么提前返回，要么跳过下面的休眠。
  // 情况二：不同步可能较慢或者卡住。或者我们在析构函数中卡住，我们会在调用 std::abort() 杀死整个进程之前休眠一段时间。
  if ((terminateProcessGroup_.load() || collectiveDebugInfoMode_.load() ||
       shouldDump_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    // 给不同步报告生成或进程组销毁留出另外两分钟时间
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
    // 记录信息级别的日志，包括日志前缀和等待的心跳超时时间
    LOG(INFO) << logPrefix() << "slept for " << heartbeatTimeoutInSec_
              << " waiting for desync report or process group destroy.";
  }

  // 到达这里时，要么已经再次休眠了 `heartbeatTimeoutInSec_` 的时长，
  // 要么线程已经结束。因为我们不希望阻塞监控线程，所以标记线程为分离状态，
  // 并且调试信息的转储变得“尽力而为”。如果进程正常退出，标记为分离也是有意义的，
  // 因为我们并不真正关心转储调试信息。

  // 我们已经在线程内部记录了完成信息，因此在这里检查返回值可能并不必要。
  // 我们主要使用一个 future，这样如果完成了就可以提前退出。

  if (!terminateHeartbeatMonitorThread_.load()) {
    // 创建一个由 MonitorThread 报告的错误消息，因此我们抛出异常并终止整个进程。
    // TODO(fduwjj): 在有挂起调试 wiki 后，我们需要更新这里的 wiki 链接。
    const auto finalExitMsg = c10::str(logPrefix(), exitMsg);
    if (monitorThreadEnabled_.load()) {
      // 如果监控线程启用，则终止进程并显示最终退出消息
      terminateProcess(finalExitMsg);
    } else {
      // 如果 PGNCCL 监控线程被禁用，但本来会杀死此作业，则记录错误日志
      LOG(ERROR)
          << "PGNCCL Monitor Thread is disabled, but would have killed this job:\n"
          << finalExitMsg;
    }
  }
}

void ProcessGroupNCCL::ncclCommWatchdog() {
  // 设置线程名为 "pt_nccl_watchdg"
  c10::setThreadName("pt_nccl_watchdg");

  try {
    // 记录日志，标记进程组看门狗线程已启动
    VLOG(2) << logPrefix() << "Process group watchdog thread started!";
    // 启动心跳监控线程
    ncclHeartbeatMonitorThread_ =
        std::thread(&ProcessGroupNCCL::heartbeatMonitor, this);
    // 调用看门狗处理方法
    watchdogHandler();
    // 记录日志，标记进程组看门狗线程正常终止
    VLOG(2) << logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    // 处理异常情况
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      // 如果异常信息表明 CUDA 驱动关闭，记录相应日志
      LOG(INFO)
          << logPrefix()
          << "main process destroyed cuda before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();

    } else {
      // 添加来自 watchdogHandler 报告的错误消息
      const auto exitMsg = c10::str(
          logPrefix(),
          "Process group watchdog thread terminated with exception: ",
          e.what());
      // 记录错误日志
      LOG(ERROR) << exitMsg;
      // 如果重新抛出 CUDA 错误或标志指示需重新抛出，则存储异常并重新抛出
      if (C10_LIKELY(rethrowCUDAErrors_) ||
          !(std::string(e.what()).find("CUDA Error"))) {
        // TODO(whc) clean up the rethrow - why is it stored in a class var and
        // rethrown?
        watchDogException_ =
            std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
        std::rethrow_exception(watchDogException_);
      }
    }
  } catch (...) {
    // 捕获所有其他异常
    const auto exitMsg = c10::str(
        logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    // 记录错误日志
    LOG(ERROR) << exitMsg;
    // 存储异常并重新抛出
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

void ProcessGroupNCCL::logWorkStart(WorkNCCL& work) {
  // 如果已更新工作的起始跟踪，则返回
  if (work.startTraceUpdated_)
    return;

  // 如果标志表明终止进程组或存在存储错误，则返回
  if (terminateProcessGroup_.load() || storeError_)
    return;

  // 标记工作的起始跟踪已更新
  work.startTraceUpdated_ = true;
  // 更新跟踪信息，记录操作类型的起始信息
  storeError_ = !c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

void ProcessGroupNCCL::logWorkEnd(WorkNCCL& work) {
  // 如果标志表明终止进程组或存在存储错误，则返回
  if (terminateProcessGroup_.load() || storeError_)
    return;

  // 如果工作的起始跟踪尚未记录，则调用记录工作起始的方法
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  // 更新跟踪信息，记录操作类型的结束信息
  storeError_ = !c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

// 获取 NCCL 看门狗的调试信息
std::string ProcessGroupNCCL::getNCCLWatchdogDebugInfo() {
  return retrieveDesyncReport(store_, "NCCL", rank_, size_);
}

// 创建日志前缀信息
std::string ProcessGroupNCCL::createLogPrefix() const {
  // 如果进程组描述不为空且不是 "undefined"，则创建带描述的日志前缀
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str("[PG ", pg_name_, " (", pg_desc_, ") Rank ", rank_, "] ");
  }
  // 否则，创建不带描述的日志前缀
  return c10::str("[PG ", pg_name_, " Rank ", rank_, "] ");
}

// 返回日志前缀
const std::string& ProcessGroupNCCL::logPrefix() const {
  return logPrefix_;
}

// 返回全局排名
const int& ProcessGroupNCCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}
const std::vector<uint64_t>& ProcessGroupNCCL::groupRanks() const {
  // 如果全局组内排名为空且当前 uid 为 0，则创建静态全局排名向量并返回
  if (options_->global_ranks_in_group.empty() && uid_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    // 填充全局排名向量从 0 到 size_-1
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  // 否则返回选项中存储的全局组内排名向量的引用
  return options_->global_ranks_in_group;
}

void ProcessGroupNCCL::watchdogHandler() {
  bool done = false;
  // 记录最后一次更新工作列表的时间
  lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  auto lastStatusUpdateTime = std::chrono::steady_clock::now();
  // 创建完成的工作列表
  std::list<ProcessGroupNCCL::WorkNCCL> completedWorkList;

  // 在未完成或未终止进程组时循环
  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    // 只要原子变量为真，每隔 kWatchdogThreadSleepMillis 毫秒忙等待工作向量
    workMetaListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });
    // 心跳计数增加一次
    heartbeat_++;

    // 某些版本的 GLOG 支持 LOG_EVERY_MS 的 less-spammy 版本，
    // 在这种情况下不记录过多的日志。
#ifdef LOG_EVERY_MS
    // 定期记录此进程组的进度
    C10_LOG_EVERY_MS(INFO, kWorkStatusUpdatePeriodMs) << c10::str(
        logPrefix(),
        "NCCL Work update periodically: ",
        "last enqueued NCCL work: ",
        pgStatus_.lastEnqueuedSeq,
        ", last completed NCCL work: ",
        pgStatus_.lastCompletedSeq,
        ".");
#endif
    // 获取 C10dLogger 实例
    auto logger = ::c10d::C10dLogger::getLogger();
    // 如果存在 logger 并且距离上次状态更新超过 kWorkStatusUpdatePeriodMs 毫秒
    if (logger &&
        computeDeltaMS(
            lastStatusUpdateTime, std::chrono::steady_clock::now()) >=
            kWorkStatusUpdatePeriodMs) {
      ::c10d::C10dLoggingData data;
      // 记录整数值
      data.integers["pg_id"] = uid_;
      data.integers["rank"] = rank_;
      data.integers["global_rank"] = globalRank();
      data.integers["last_enqueued_work"] = pgStatus_.lastEnqueuedSeq;
      data.integers["last_started_work"] = pgStatus_.lastStartedSeq;
      data.integers["last_completed_work"] = pgStatus_.lastCompletedSeq;
      data.integers["last_enqueued_numel_in"] = pgStatus_.lastEnqueuedNumelIn;
      data.integers["last_enqueued_numel_out"] = pgStatus_.lastEnqueuedNumelOut;
      data.integers["last_completed_numel_in"] = pgStatus_.lastCompletedNumelIn;
      data.integers["last_completed_numel_out"] =
          pgStatus_.lastCompletedNumelOut;
      // 记录字符串值
      data.strings["last_enqueued_work_name"] = pgStatus_.lastEnqueuedWorkName;
      data.strings["last_started_work_name"] = pgStatus_.lastStartedWorkName;
      data.strings["last_completed_work_name"] =
          pgStatus_.lastCompletedWorkName;
      data.strings["pg_name"] = pg_name_;
      data.strings["pg_desc"] = pg_desc_;
      // 使用 logger 记录日志数据
      logger->log(data);
      // 更新最后状态更新时间
      lastStatusUpdateTime = std::chrono::steady_clock::now();
    }

    // 检查是否所有工作都已完成
    done = workMetaList_.empty();
  }
}
void ProcessGroupNCCL::runHookLoop() {
  c10::setThreadName("pt_nccl_runhook");

  bool done = false;
  // 循环直到所有工作完成或者进程组终止标志位为真
  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(completedWorkListMutex_);
    // 每隔 kWatchdogThreadSleepMillis 毫秒检查一次工作列表，只要终止标志位为真就忙等
    completedWorkListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool {
          return !completedWorkList_.empty() || terminateProcessGroup_.load();
        });

    try {
      for (auto it = completedWorkList_.begin(); it != completedWorkList_.end();
           /* no increment */) {
        const WorkNCCL& work = *it;
        // 钩子可能会获取 GIL（全局解释器锁），为了避免死锁先解锁
        lock.unlock();

        auto timeStarted =
            std::chrono::system_clock::now() +
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                work.workStartTime_ - std::chrono::steady_clock::now());
        // 调用完成钩子函数，传入工作信息的共享指针
        onCompletionHook_(std::make_shared<WorkInfo>(
            work.retrieveOpType(), // OpType 操作类型
            work.getSequencenumber(), // seq 序列号
            timeStarted, // timeStarted 开始时间
            std::chrono::system_clock::now(), // timeFinished 完成时间
            std::chrono::duration<float, std::milli>(
                work.getDuration()) // activeDuration 活跃时长
            ));

        lock.lock();
        it = completedWorkList_.erase(it);
      }
    } catch (std::exception& e) {
      if (std::string(e.what()).find("driver shutting down") !=
          std::string::npos) {
        LOG(INFO)
            << logPrefix()
            << "main process destroyed cuda before runHookLoop exited, terminating runHookLoop."
            << " (runHookLoop caught exception: " << e.what();

      } else {
        // PythonOnCompletionHook 已经提取了 Python 异常消息并用 C++ 封装了，所以这里不再需要获取 GIL
        const auto errorStr = c10::str(
            "Caught exception on rank ",
            rank_,
            " while running onCompletion hook for ProcessGroupNCCL: ",
            e.what(),
            ". Aborting all communicators.");
        
        // 在此时不需要在 WorkNCCL 上调用 abort()，因为此时 collective 已经成功完成，只需终止进程组内的所有 NCCL Communicators
        abort(errorStr);
      }
    }

    // 到这一步锁仍然被获取
    done = completedWorkList_.empty();
  }
}

std::exception_ptr ProcessGroupNCCL::WorkNCCL::checkForNCCLErrors() {
  return checkForNCCLErrorsInternal(ncclComm_);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrors(
    std::shared_ptr<NCCLComm>& ncclComm) {
  return checkForNCCLErrorsInternal(ncclComm);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrorsInternal(
  // 如果 commFailureReason 被设置，则优先使用它，而不是 checkForNcclError() 的结果。
  auto commFailureReason = ncclComm->getNcclCommFailureReason();
  if (commFailureReason != c10::nullopt) {
    // 如果 commFailureReason 已设置，抛出异常，说明 NCCL 通信器由 ProcessGroupNCCL 设置了错误。
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        c10::str(
            "NCCL communicator encountered error set by ProcessGroupNCCL: ",
            *commFailureReason)));
  }
  // 检查是否有异步的 NCCL 错误
  ncclResult_t ncclAsyncErr = ncclComm->checkForNcclError();
  // 当 TORCH_NCCL_USE_COMM_NONBLOCKING 启用非阻塞模式时，
  // 如果有待处理的 NCCL 调用，可能会返回 ncclInProgress，此时不应抛出异常。
#ifdef NCCL_HAS_COMM_NONBLOCKING
  // 如果定义了 NCCL_HAS_COMM_NONBLOCKING，则仅当 ncclAsyncErr 不是 ncclSuccess 且不是 ncclInProgress 时执行以下代码块
  if (ncclAsyncErr != ncclSuccess && ncclAsyncErr != ncclInProgress) {
#else
  // 如果未定义 NCCL_HAS_COMM_NONBLOCKING，则当 ncclAsyncErr 不是 ncclSuccess 时执行以下代码块
  if (ncclAsyncErr != ncclSuccess) {
#endif
    // 返回一个异常指针，其中包含 NCCL 错误信息和细节字符串
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        "NCCL error: " + ncclGetErrorWithVersion(ncclAsyncErr) + "\n" +
            getNcclErrorDetailStr(ncclAsyncErr)));
  }

  // 返回空指针，表示没有异常
  return nullptr;
}

void ProcessGroupNCCL::broadcastUniqueNCCLID(
    ncclUniqueId* ncclID,
    bool isSingleP2POp,
    const std::string& p2pKey,
    int p2pRank) {
  // 对于集体操作：
  // 每次创建 NCCL 通信器时，需要从 rank 0 向所有其他 rank 广播一个唯一的 ID。
  // 广播通过 rank 0 设置存储中的一个键，然后所有其他 rank 检索该键的内容。
  // 单个进程组可能创建多个 NCCL 通信器，因此使用序列号来区分它们。
  // 对于单点到点操作：
  // 序列号仅在进程组的两个进程中递增。因此，后续的集体操作将看到不同的序列号，这将导致运行时错误。
  // 为了避免这种情况，对于 p2p 通信，请使用 src:target 对代替序列号。

  std::string storeKey;
  if (!isSingleP2POp) {
    // 如果不是单点到点操作，使用 ncclCommCounter_ 的值作为存储的键
    storeKey = std::to_string(ncclCommCounter_++);
  } else {
    // 如果是单点到点操作，使用提供的 p2pKey 作为存储的键
    storeKey = p2pKey;
  }
  if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    // 如果当前进程是 rank 0 或者是单点到点操作且 p2pRank 是 0
    // 将 ncclID 转换为字节数组，然后将其存储在键为 storeKey 的存储中
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ncclID),
        reinterpret_cast<uint8_t*>(ncclID) + NCCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, vec);
  } else {
    try {
      // 如果当前进程不是 rank 0，尝试从存储中获取存储在 storeKey 中的 ncclUniqueId
      auto vec = store_->get(storeKey);
      TORCH_CHECK_WITH(
          DistBackendError,
          vec.size() == NCCL_UNIQUE_ID_BYTES,
          "Invalid size for ncclUniqueId");
      // 将存储中的 ncclUniqueId 数据拷贝回 ncclID
      std::memcpy(ncclID, vec.data(), vec.size());
    } catch (const std::exception& e) {
      // 捕获异常并生成详细的错误消息，然后抛出 DistBackendError 异常
      std::string exceptionMsg = c10::str(
          "[",
          rank_,
          "] is setting up NCCL communicator and "
          "retrieving ncclUniqueId from [0] via c10d key-value store by key '",
          storeKey,
          "', but store->get('",
          storeKey,
          "') got error: ");
      C10_THROW_ERROR(
          DistBackendError,
          exceptionMsg + e.what() +
              ". This may indicate a possible application crash on rank 0 or a network set up issue.");
    } catch (...) {
      // 捕获未知异常并生成详细的错误消息，然后抛出 DistBackendError 异常
      C10_THROW_ERROR(
          DistBackendError,
          c10::str(
              "Unknown exception while [",
              rank_,
              "] is setting up NCCL communicator and "
              "retrieving ncclUniqueId from [0] via c10d key-value store by key '",
              storeKey,
              "'",
              ". This may indicate a possible application crash on rank 0 or a network set up issue."));
    }
  }
}
void ProcessGroupNCCL::destroyNCCLComms(const std::string& devNCCLCommMapKey) {
  // 使用互斥锁确保线程安全
  std::lock_guard<std::mutex> lock(mutex_);
  // 检查指定的键是否存在于通信器映射中
  if (devNCCLCommMap_.find(devNCCLCommMapKey) == devNCCLCommMap_.end()) {
    // 如果键不存在，则抛出断言错误
    TORCH_INTERNAL_ASSERT(
        false,
        "Expected to find key ",
        devNCCLCommMapKey,
        " in NCCL communicator map.");
  }
  // 获取指定键对应的共享指针，即NCCL通信器
  std::shared_ptr<NCCLComm>& ncclComm = devNCCLCommMap_[devNCCLCommMapKey];
  // 使用ncclCommAbort()而不是ncclCommDestroy()，以避免在销毁过程组时导致段错误
  ncclComm->ncclCommAbort();
  // 从映射中移除通信器
  devNCCLCommMap_.erase(devNCCLCommMapKey);
  // 清空已使用的设备索引集合
  usedDeviceIdxs_.clear();

  // 获取互斥锁，以确保在修改ncclCommDevIdxMap时线程安全
  ncclCommDevIdxMapMutex.lock();
  // 从通信器到设备索引映射中移除相应的条目
  ncclCommDevIdxMap.erase(ncclComm);
  ncclCommDevIdxMapMutex.unlock();
}

std::shared_ptr<NCCLComm> ProcessGroupNCCL::getNCCLComm(
    const std::string& deviceKey,
    at::Device& device,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  // 对deviceKey进行空值检查，确保设备键不为空
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the NCCL Communicator since "
        "the GPU devices are not known");
  }
  // 如果设定了绑定的设备ID，则验证传入的设备是否与绑定设备相符
  if (bound_device_id_) {
    if (*bound_device_id_ != device) {
      // 如果设备不匹配，则记录错误并抛出异常
      LOG(ERROR) << logPrefix() << "Tensor found on device " << device
                 << " but backend constrained to " << *bound_device_id_;
      C10_THROW_ERROR(
          DistBackendError,
          "Attempt to perform collective on tensor not on device passed to init_process_group");
    }
  }

  // 将使用的设备索引添加到集合中
  usedDeviceIdxs_.insert(device.index());

  {
    // 使用互斥锁确保在访问devNCCLCommMap_时线程安全
    std::lock_guard<std::mutex> lock(mutex_);
    // 检查是否已经存在指定设备键的NCCL通信器
    if (devNCCLCommMap_.find(deviceKey) != devNCCLCommMap_.end()) {
      // 如果存在则直接返回缓存的通信器
      return devNCCLCommMap_[deviceKey];
    }
  }

  // 如果未缓存NCCL通信器，则创建一个新的条目
  std::shared_ptr<NCCLComm> ncclComm;

  // 创建唯一的NCCL ID并进行广播
  ncclUniqueId ncclID;

  // 重置日志前缀以包括组描述
  logPrefix_ = createLogPrefix();

#ifdef NCCL_COMM_DESCRIPTION
  // 将进程组名称和描述传递给NCCL通信器
  std::string commDesc = pg_desc_ + ':' + pg_name_;
  options_->config.commDesc = strdup(commDesc.c_str());
#endif

  // 对于batch_isend_irecv，需要提前调用ncclGroupStart()
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);
  // 对于点对点通信，较低rank的进程会获得唯一的ID
  if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
    C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID), c10::nullopt);
  }

  // 如果需要广播NCCL唯一ID，则进行广播以确保每个进程都有唯一的NCCL ID
  if (shouldBroadcastNCCLUniqueID(isSendRecvSelf)) {
    // 开始广播唯一的NCCL ID
    auto timeStarted = std::chrono::steady_clock::now();
    broadcastUniqueNCCLID(&ncclID, singleP2POp, deviceKey, p2pRank);
    // 计算时间差并转换为毫秒
    auto timerDeltaMs =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - timeStarted)
            .count() *
        1000;
    // 记录信息日志，包括时间差
    LOG(INFO) << logPrefix()
              << "ProcessGroupNCCL broadcast unique ID through store took "
              << timerDeltaMs << " ms";
  }

  at::cuda::OptionalCUDAGuard gpuGuard;

  // [Group Start/End Note] 用于确保在调用通信原语之前创建 nccl 通信器。
  // 示例情况：使用 batch_isend_irecv 向目标进程发送张量。
  // 发送方的 NCCL 调用如下：
  //   ncclGroupStart() // 在 batch_isend_irecv 中
  //   ncclCommInitRank() // 在 NCCLComm::create 内部
  //   ncclSend()
  //   ncclGroupEnd() // 在 batch_isend_irecv 中
  // 使用这种模式，nccl 通信器将在最后一个 ncclGroupEnd 中创建，
  // 这意味着当处理 ncclSend 时，传递的通信器参数为 NULL，这将导致运行时错误。
  // 因此，我们需要“关闭”所有活动的 nccl 组，以确保在遇到任何通信调用之前实际创建 nccl 通信器。
  // 这就是为什么需要下面的 for 循环的原因。
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    // 由于通信尚未启动，因此只能以阻塞方式检查
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  }

  // 获取 GPU 的全局大小和当前进程的 GPU 排名
  int numRanks, rank;

  if (!singleP2POp) {
    // 集体操作、全对全或批量点对点操作
    numRanks = getSize();
    rank = getRank();
  } else if (isSendRecvSelf) {
    // 同一进程内发送和接收
    numRanks = 1;
    rank = 0;
  } else {
    // 对于单一点对点操作，只涉及两个进程，因此 GPU 排名为 0 或 1
    numRanks = 2;
    rank = p2pRank;
  }

  // 获取设备索引
  auto deviceIndex = device.index();
  gpuGuard.set_index(deviceIndex);
#ifdef NCCL_HAS_COMM_SPLIT
  // 如果支持分割通信，并且指定了分割源头
  if (options_->split_from) {
    // 检查分割颜色是否非零，因为必须指定一个非零颜色进行分割
    TORCH_CHECK(
        options_->split_color != 0,
        "Must specify a non-zero color when splitting");
    // 在可能的情况下，查找一个有效且健康的通信器来进行分割
    std::lock_guard<std::mutex> lock(options_->split_from->mutex_);
    // 获取分割源头的设备到 NCCL 通信器的映射
    auto& other_comms = options_->split_from->devNCCLCommMap_;
    auto dit = other_comms.find(getKeyFromDevice(device));
    // 如果找到了对应设备的通信器
    if (dit != other_comms.end()) {
      auto& parentComm = dit->second;
      // 如果父通信器存在且未中止
      if (parentComm != nullptr && !parentComm->isAborted()) {
        // 执行通信器的分割操作，生成新的 NCCL 通信器
        ncclComm = NCCLComm::split(
            parentComm.get(),
            options_->split_color,
            rank,
            options_->config,
            options_->global_ranks_in_group);
      }
    }
  }
#endif

  // 简化条件嵌套，如果 ncclComm 尚未创建，则创建之
  if (!ncclComm) {
#ifdef NCCL_HAS_COMM_NONBLOCKING
    // 使用指定的配置创建 NCCL 通信器
    ncclComm = NCCLComm::create(numRanks, rank, ncclID, options_->config);
#else
    // 使用默认配置创建 NCCL 通信器
    ncclComm = NCCLComm::create(numRanks, rank, ncclID);
#endif
  }

  // 创建 NCCL 流
  bool force_high = getCvarBool(TORCH_NCCL_HIGH_PRIORITY, false);
  auto streamVal = at::cuda::getStreamFromPool(
      options_->is_high_priority_stream || force_high);

  {
    // 加锁，将设备键和 NCCL 通信器添加到初始化映射中
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, ncclComm);
  }

  // 记录进程组相关信息到跟踪缓冲区
  NCCLTraceBuffer::get()->record_pg_ranks(
      std::make_tuple(pg_name_, pg_desc_), groupRanks());

  // 记录参数化的通信操作
  RECORD_PARAM_COMMS(
      0, // seq
      std::make_tuple(pg_name_, pg_desc_), // PG name tuple
      rank, // rank
      "init", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      size_); // worldSize

  // 记录日志，标明已创建的 NCCL 通信器
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL created ncclComm_ "
            << ncclComm->ncclComm_ << " on CUDA device: " << deviceIndex;

  // 在此时点，NCCL 应该已经初始化完成，因此可以准确获取环境值，
  // 即使 NCCL 通过读取 nccl.conf 文件来设置它
  LOG(INFO) << logPrefix()
            << "NCCL_DEBUG: " << getCvarString({"NCCL_DEBUG"}, "N/A");

  // 查看 [Group Start/End Note]
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    // 调用 ncclGroupStart() 启动一个新的通信组
    C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);
  }

  // 将新创建的流对象移动到 ncclStreams_ 中保存
  ncclStreams_.emplace(deviceKey, std::move(streamVal));

  // 创建一个禁用计时的 CUDA 事件，用于后续的流同步操作
  // 这样设置提供了最佳的性能，尤其是在使用 cudaStreamWaitEvent() 和 cudaEventQuery() 时
  // 因为我们这里不需要测量 cudaEvent 的性能，所以应该使用该标志
  // TODO(kwen2501): ncclEvents_ 是否在其他地方被使用？
  ncclEvents_.emplace(deviceKey, at::cuda::CUDAEvent(cudaEventDisableTiming));

  // 将建立好的 ncclComm 与其对应的 ncclUniqueId 字符串映射保存起来
  ncclIdToCommMap_.emplace(buildNcclUniqueIdStr(ncclID), ncclComm);

  // 将设备关键字为 deviceKey 的通信资源从 inInitializationCommMap_ 移动到 devNCCLCommMap_
  auto it = inInitializationCommMap_.find(deviceKey);
  // 可能之前的线程已经将 devicesKey 从 inInitializationCommMap_ 中移除并添加到 devNCCLCommMap_ 中
  if (it != inInitializationCommMap_.end()) {
    devNCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    // 现在 ncclComms 已经完全初始化
    // 将所有活动的 CUDA 内存段注册到新的 NCCL 通信器的缓存分配器中
    if (useTensorRegisterAllocatorHook_) {
      auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
      // 如果在相同的设备上，则将内存段注册到新的 NCCL 通信器中
      for (const auto& segmentInfo : snapshot.segments) {
        TORCH_INTERNAL_ASSERT(
            segmentInfo.device == device.index(),
            "CUDA 内存段设备与当前设备不匹配");
        ncclComm->registerSegment(
            reinterpret_cast<void*>(segmentInfo.address),
            segmentInfo.total_size);
      }
    }
    // 记录 ncclComm 和设备索引之间的映射关系，以便后续的注册钩子可以将新分配的段注册到相同设备上的通信器中
    // 注意：在通信器被销毁时，需要从该映射中删除通信器，否则可能会向无效的通信器注册
    ncclCommDevIdxMapMutex.lock();
    ncclCommDevIdxMap.emplace(ncclComm, device.index());
    ncclCommDevIdxMapMutex.unlock();
  }

  // 再次查找 deviceKey 对应的通信资源
  it = devNCCLCommMap_.find(deviceKey);
  // 断言确保通信资源在缓存中已经被填充
  TORCH_INTERNAL_ASSERT(
      it != devNCCLCommMap_.end(), "Communicators not populated in cache!");

  // 返回找到的通信资源
  return it->second;
}

// 返回通信分组的计数器值
uint64_t ProcessGroupNCCL::getCommSplitCounter() const {
  // 初始化计数器为 0
  uint64_t ret = 0;
  // 遍历 ncclIdToCommMap_ 中的每一个条目
  for (const auto& i : ncclIdToCommMap_) {
    // 获取 ncclComm 对象的引用
    auto& ncclComm = i.second;
    // 累加每个 ncclComm 对象的通信分组计数器值
    ret += ncclComm->getCommSplitCounter();
  }
  // 返回累加结果
  return ret;
}

namespace {

// 检查 GPU 上单个张量的有效性
void check_gpu_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false // 操作是否是点对点操作
) {
  // 检查张量是否在 CUDA 设备上且为稠密张量
  if (!tensor.is_cuda() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
  }
  // 如果不是建议的内存格式连续，则跳过以下要求
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    if (p2p) {
      // 在 P2P 操作中检测到非连续张量，需要用户保证源和目标张量具有相同的连续性格式
      TORCH_WARN_ONCE(
          "Detected non-contiguous tensor in P2P operations. It is user "
          "responsibility to guarantee that source and destination tensors have "
          "the same contiguity format.");
    } else {
      // 非 P2P 操作中要求张量必须是连续的
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
  }
}

// 检查所有张量是否具有相同的类型、形状和所在 GPU
// TODO: test_c10d_nccl.py 应该考虑添加对错误条件的测试，例如故意传递无效张量并检查是否抛出正确的异常。
// “预期的张量列表位于同一设备上”条件可能是一个挑战，因为测试需要在同一进程中传递不同设备上的张量。
int64_t check_gpu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  // 检查张量列表不为空
  if (tensors.size() == 0) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  // 获取第一个张量作为参考
  const auto& first = tensors.front();

  int64_t total_numel = 0;
  // 遍历所有张量
  for (const auto& t : tensors) {
    // 检查张量是否在 CUDA 设备上且为稠密张量
    if (!t.is_cuda() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
    }
    // 检查张量是否具有相同的标量类型
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    // 检查张量是否是非重叠且稠密的
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    // 由于在该函数中，用户调用了一个 _coalesced 集合操作，可能有不同大小和步幅的张量。
    // 因此，我们不检查大小和步幅匹配，但确保张量位于同一设备上。
    TORCH_CHECK_WITH(
        ValueError,
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    // 累加所有张量的元素数量
    total_numel += t.numel();
  }

  // 返回所有张量的总元素数量
  return total_numel;
}

// 检查所有输入张量是否具有相同的大小
bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  // 遍历所有输入张量
  for (const auto& input_tensor : input_tensors) {
    // 如果当前张量与第一个张量的大小不同，返回 false
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  // 所有张量大小相同，返回 true
  return true;
}

} // namespace

// 初始化工作对象，并返回指向工作对象的指针
c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> ProcessGroupNCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    // 接收一组输出张量的引用，可能用于记录目的（需确认）
    const std::vector<at::Tensor>& outputs, // TODO(kwen2501): necessary?
    // 是否记录这个工作
    bool record) {
  // 创建一个 ProcessGroupNCCL::WorkNCCL 对象，并初始化它
  auto r = c10::make_intrusive<ProcessGroupNCCL::WorkNCCL>(
      device,
      rank,
      opType,
      seqCollective_,
      profilingTitle,
      // 如果 profilingTitle 不为空，则将 inputs 包装成可选的向量传递给构造函数
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : c10::nullopt,
      desyncDebug_,
      enableTiming_.load(),
      dist_debug_level_);
  // 如果需要记录
  if (record) {
    // 检查操作类型是否是点对点操作
    bool isP2P = isP2POp(opType);
    // 理想情况下，记录每个入队的工作，而不是每个创建的工作
    // 在此 PR 时，我们当前并未将每个创建的工作都入队
    // 但从已初始化的工作中偷窃引用到起始/结束 CUDA 事件是不安全的，
    // 因此我们必须确保通过 initWork 初始化的任何工作都将被入队
    // 最初，将 record() 移入 workEnqueue()，但发现这样做难以访问 profilingTitle、inputs 和 outputs 用于元数据记录，
    // 并且我们不想将这些对象附加到 Work，因为这会增加在线程间复制 Work 对象时保持这些张量存活的开销
    // 记录此工作的跟踪 ID 到 NCCLTraceBuffer
    r->trace_id_ = NCCLTraceBuffer::get()->record(
        uid_,
        std::make_tuple(pg_name_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle ? profilingTitle : "",
        inputs,
        outputs,
        r->ncclStartEvent_.get(),
        r->ncclEndEvent_.get(),
        options_->timeout,
        isP2P);
  }
  // 返回创建的 WorkNCCL 对象
  return r;
// 结束 ProcessGroupNCCL 类定义

// 返回 WorkNCCL 对象中保存的输出张量的 vector
std::vector<at::Tensor> ProcessGroupNCCL::WorkNCCL::result() {
  return *outputs_;
}

// 返回 WorkNCCL 对象中保存的 future 对象
c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupNCCL::WorkNCCL::getFuture() {
  return future_;
}

// 获取 WorkNCCL 对象的执行持续时间，前提是已启用 timingEnabled_
float ProcessGroupNCCL::WorkNCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(ncclStartEvent_, "getDuration only works if ncclStartEvents_ is populated, true if timing enabled");
  TORCH_CHECK(ncclEndEvent_, "getDuration only works if ncclEndEvents_ is populated, which should always be true");
  return ncclStartEvent_->elapsed_time(*ncclEndEvent_);
}

// 获取 WorkNCCL 对象的序列号
uint64_t ProcessGroupNCCL::WorkNCCL::getSequencenumber() const {
  return seq_;
}

// 将工作对象加入到工作元数据列表中，如果终止标志未设置
void ProcessGroupNCCL::workEnqueue(c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> work) {
  if (!terminateProcessGroup_.load()) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // 避免在清理线程中处理视图张量
    // 视图张量的销毁会调用 autograd_meta，必须在用户线程中销毁，否则会死锁。
    // 在这里我们将工作对象加入队列，但不包含输出张量 outputs_
    workMetaList_.emplace_back(*work);
    // 更新与最后一个入队工作相关的进程组状态
    pgStatus_.lastEnqueuedSeq = work->seq_;
    pgStatus_.lastEnqueuedWorkName = opTypeToString(work->opType_);
    pgStatus_.lastEnqueuedNumelIn = work->numelIn_;
    pgStatus_.lastEnqueuedNumelOut = work->numelOut_;
    lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  }
}

// ProcessGroupNCCL::Options 构造函数，初始化选项，包括是否是高优先级流
ProcessGroupNCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(NCCL_BACKEND_NAME, kProcessGroupNCCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

// 定义静态常量 CoalActive, CoalColl, CoalP2P，表示协同操作的状态
static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

// 开始协同操作的函数
void ProcessGroupNCCL::startCoalescing() {
  // 其他集体操作在创建工作之前会增加 seq_。因此，如果合并操作在初始化工作后才增加 seq_，
  // 它们将与上一个非合并集体操作的 seq_ 发生冲突（重用）。
  // 以前，seq_ 是在 endCoalescing 内部增加的，但在 initWork 之前增加。
  // 由于我们现在将一个合并组的各个操作记录到飞行记录器中，我们希望这些操作和其 'endCoalescing' 操作具有相同的 seq_。
  // 因此我们在开始时增加，这样做有一个小缺点——如果有人在没有中间操作的情况下执行 'start' 和 'end' 合并区域，我们会浪费一个 seq_。

  // 不要在这里增加 op_id_，因为 startCoalescing 不是一个逻辑操作。
  // 对于合并组中的每个逻辑操作逐个增加。
  if (coalescing_state_ & CoalP2P) {
    seqP2P_++;
  } else {
    seqCollective_++;
  }

  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescing_state_ |= CoalActive;
  groupStart();
}

// optype 用于指定复合操作类型，如 ALLGATHER 和 REDUCE_SCATTER
// 结束当前的合并操作，并返回一个指向 Work 对象的智能指针
c10::intrusive_ptr<Work> ProcessGroupNCCL::endCoalescing(OpType optype) {
  // 如果 coalescedComm_ 为 nullptr，表示没有正在合并的工作，直接返回
  if (coalescedComm_ == nullptr) {
    groupEnd();  // 执行组结束操作
    coalescing_state_ = 0;  // 重置合并状态
    return nullptr;  // 返回空指针
  }
  // 检查 coalescedDevice_ 的索引是否有效，如果不是则抛出错误信息
  TORCH_CHECK(
      coalescedDevice_.index() >= 0,
      "Somthing went wrong. Did you call end_coalescing before start_coalescing?");

  // 将 coalescedComm_ 和 coalescedDevice_ 分别赋值给局部变量 comm 和 device
  auto comm = coalescedComm_;  // 保存当前的通信对象
  auto device = coalescedDevice_;  // 保存当前的设备对象

  // 根据设备获取一个唯一的键值
  const auto key = getKeyFromDevice(device);
  // 从 ncclStreams_ 中获取与 key 相关联的 ncclStream
  auto ncclStream = ncclStreams_.at(key);

  // 创建一个 Work 对象
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  // 决定是否将工作任务排入队列，需要满足合并状态为真且没有 CUDA 流捕获任务
  bool enqueue =
      (coalescing_state_) && capture_status == c10::cuda::CaptureStatus::None;
  auto work =
      initWork(device, rank_, optype, "nccl:coalesced", {}, {}, enqueue);
  work->ncclComm_ = comm;  // 设置 Work 对象的通信对象
  work->blockingWait_ = blockingWait_;  // 设置是否阻塞等待
  work->avoidRecordStreams_ = avoidRecordStreams_;  // 设置是否避免记录流
  work->opTimeout_ = options_->timeout;  // 设置操作超时时间
  work->store_ = store_;  // 设置存储对象

  // 如果启用了时间跟踪，则在 ncclGroupEnd 前记录开始事件
  if (work->timingEnabled_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  // 根据 nccl_use_nonblocking 的返回值选择性执行非阻塞组结束操作或阻塞组结束操作
  if (nccl_use_nonblocking()) {
    groupEndNonblocking(comm);  // 非阻塞组结束操作
  } else {
    groupEnd();  // 阻塞组结束操作
  }

  // 在 ncclGroupEnd 后记录结束事件
  // TODO(eqy): 如果设置了 avoidRecordStreams_，是否仍然需要记录结束事件？
  work->ncclEndEvent_->record(ncclStream);

  // 如果设置了 avoidRecordStreams_，则为工作对象分配一个用于安全分配的共享指针
  if (avoidRecordStreams_) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }

  // 在预先检查流捕获状态之前通知图形对象
  at::cuda::CUDAGraph::inc_pending_event_queries();

  // 根据 enqueue 决定是否将工作对象排入队列或者减少待处理事件查询计数
  if (enqueue) {
    workEnqueue(work);  // 将工作对象排入队列
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();  // 减少待处理事件查询计数
  }

  coalescing_state_ = 0;  // 重置合并状态
  coalescedComm_ = nullptr;  // 将 coalescedComm_ 置空
  return work;  // 返回创建的工作对象
}

// 默认调用以 COALESCED 作为 OpType 的结束合并操作
c10::intrusive_ptr<Work> ProcessGroupNCCL::endCoalescing() {
  return endCoalescing(OpType::COALESCED);  // 调用具体 OpType 版本的结束合并操作
}

// 实现集体操作的模板方法，处理输入输出张量及相关操作
template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams) {
  // 如果启用 NaN 检查，则执行额外的检查和处理
  if (enableNanCheck_) {
    // 检查输入中是否包含 NaN（Not a Number）值
    checkForNan(input);
  }
  // 用户设置的环境可能会添加到集体调用的选项中
  avoidRecordStreams |= avoidRecordStreams_;
  // 获取当前 CUDA 流的捕获状态
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  // 如果捕获状态表明正在捕获不可捕获的 NCCL 操作，产生错误
  errorIfCapturingNonCapturableNCCL(capture_status);

  // 增加集体操作计数器和操作 ID
  seqCollective_++;
  op_id_++;

  // 获取输入张量的设备
  auto device = getDevice(input);
  // 根据设备获取关键字
  const auto key = getKeyFromDevice(device);
  // 获取 NCCL 通信对象
  auto ncclComm = getNCCLComm(key, device, opType);

  // 如果启用了合并状态，并且当前没有合并的设备
  coalescing_state_ |= CoalColl;
  if (coalescedDevice_.index() < 0) {
    coalescedDevice_ = device;
  } else {
    // 否则，确保所有设备相同，否则抛出错误
    TORCH_CHECK(
        coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
  }
  // 如果当前没有合并的通信对象，将当前的 NCCL 通信对象赋值给它
  if (coalescedComm_ == nullptr) {
    coalescedComm_ = ncclComm;
  } else {
    // 否则，确保所有通信对象相同，否则抛出错误
    TORCH_CHECK(coalescedComm_ == ncclComm, MULTI_DEVICE_ERROR_MSG);
  }

  // 由于下面多次使用，因此将 unordered_map 查找结果缓存
  auto ncclStream = ncclStreams_.at(key);

  // 首先让 NCCL 流等待输入张量分配的流事件
  syncStream(device, ncclEvents_[key], ncclStream);

  // 创建输入和输出张量的向量
  std::vector<at::Tensor> inputs{input};
  std::vector<at::Tensor> outputs{output};

  // 决定是否排队执行任务，条件是不处于合并状态且没有 CUDA 流捕获
  bool enqueue =
      !coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None;
  // 初始化任务对象
  auto work =
      initWork(device, rank_, opType, profilingTitle, inputs, outputs, enqueue);

  // 存储输出张量的引用，以便后续使用
  work->outputs_ =
      std::make_shared<std::vector<at::Tensor>>(std::move(outputs));

  // 如果避免记录流，则为了分配器安全性而存储输入张量的引用
  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
    work->stashed_for_allocator_safety_->push_back(input);
  }

  // 可选的 CUDA 设备保护
  at::cuda::OptionalCUDAGuard gpuGuard;

  // 只有在启用时间记录时才记录开始事件
  if (work->timingEnabled_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  // 执行预处理
  pre(ncclStream, work);

  // 获取 NCCL 通信对象的底层 NCCL 通信句柄
  ncclComm_t comm = ncclComm->getNcclComm();

  // `inputs` 和 `outputs` 在工作流程中被创建，同时在不同的 ncclStream 中使用。
  // 因此，两者必须都记录 ncclStream，以防在集体操作完成之前被释放。
  //
  // 这里仅记录 `inputs`，而将 `outputs` 的记录留给操作函数 `fn` 处理，以应对输入和输出不相同的情况。
  //
  // 参见 [Sync Streams]。
  if (!avoidRecordStreams) {
    if (!input.is_sparse()) {
      // 对于非稀疏输入，记录输入张量的流
      c10::cuda::CUDACachingAllocator::recordStream(
          input.storage().data_ptr(), ncclStream);
    } else {
      // 对于稀疏输入，记录索引和值张量的流
      c10::cuda::CUDACachingAllocator::recordStream(
          input.values().storage().data_ptr(), ncclStream);
      c10::cuda::CUDACachingAllocator::recordStream(
          input.indices().storage().data_ptr(), ncclStream);
    }
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果 NCCL 没有非阻塞通信特性，则使用 C10D_NCCL_CHECK 执行函数 fn
  C10D_NCCL_CHECK(
      fn(input, output, comm, ncclStream),
      ncclComm->getNcclCommFailureReason());
#else
  // 如果 NCCL 支持非阻塞通信特性，则使用 C10D_NCCL_CHECK_TIMEOUT 执行函数 fn
  C10D_NCCL_CHECK_TIMEOUT(
      fn(input, output, comm, ncclStream),
      comm,
      ncclComm->getNcclCommFailureReason());
#endif

// 将 work 提交到 ncclStream
  post(ncclStream, work);

  // 如果不处于 coalescing_state_，记录 ncclEndEvent_ 的结束事件
  if (!coalescing_state_) {
    work->ncclEndEvent_->record(ncclStream);
  }
  // 将 ncclComm_ 设置为 ncclComm
  work->ncclComm_ = ncclComm;

  {
    // 使用 streamGuard 管理 ncclStream 的 CUDA 多流
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
    std::vector<at::Device> devices{device};
    // 为 work 创建一个 intrusve_ptr 的 CUDA Future
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // 添加一个回调函数，运行 profiling end 回调函数，确保适当的同步
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false 允许我们在 ivalue::Future 中跳过同步，但前提是 lambda 不使用 "Future" 参数
          /*uses_future=*/false);
    }
    // 标记 work->future_ 完成，设置其值为 work->outputs_
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // 设置适当的工作参数
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  // 记录调试信息的大小信息，只记录第一个设备上的大小，因为多设备进程已经被弃用
  work->numelIn_ = input.numel();
  work->numelOut_ = output.numel();

  // 通知图形，增加挂起事件查询计数
  at::cuda::CUDAGraph::inc_pending_event_queries();
  // 如果需要入队，则将 work 加入工作队列，否则减少挂起事件查询计数
  if (enqueue) {
    workEnqueue(work);
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  // 返回处理好的 work 对象
  return work;
}

// ProcessGroupNCCL 类的 collectiveCoalesced 模板函数定义
template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collectiveCoalesced(
    std::vector<at::Tensor>& inputs,          // 输入张量列表的引用
    std::vector<at::Tensor>& outputs,         // 输出张量列表的引用
    Fn fn,                                   // 函数对象 fn
    OpType opType,                            // 操作类型
    const char* profilingTitle,
    // 用户设置的环境可能会在集体调用选项上添加
    avoidRecordStreams |= avoidRecordStreams_;
    
    // 获取当前 CUDA 流的捕获状态，可能会初始化上下文
    c10::cuda::CaptureStatus capture_status =
        c10::cuda::currentStreamCaptureStatusMayInitCtx();
    errorIfCapturingNonCapturableNCCL(capture_status);
    
    // 增加集体调用计数器
    seqCollective_++;
    
    // 对于 coalescingManager 集体，每个集体没有单独的 C++ 调用，因此没有飞行记录，
    // 我们一起增加 seq* 和 op_id_。与 startCoalesing/endCoalescing 流程相比，
    // 在组内每组增加一次 seq_，每个操作增加一次 op_id_
    op_id_++;
    
    // 当前 API 允许的一个场景是 inputs.size() 和 outputs.size() 都大于 0。
    // 1. 如果调用是 _coalesced 调用，则所有输入必须在同一设备上。
    //    NCCL 调用组将对每个输入分别应用集体，但整个组应该是高效的，
    //    甚至可能执行为单个融合的内核。
    auto device = getDevice(inputs[0]);
    const auto key = getKeyFromDevice(device);
    auto ncclComm = getNCCLComm(key, device, opType);
    
    // 如果处于 coalescing 状态
    if (coalescing_state_ & CoalActive) {
      coalescing_state_ |= CoalColl;
      // 如果 coalescedDevice_ 小于 0
      if (coalescedDevice_.index() < 0) {
        coalescedDevice_ = device;
      } else {
        // 否则验证 coalescedDevice_ 的索引与设备的索引是否一致
        TORCH_CHECK(
            coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
      }
      // 如果 coalescedComm_ 为空
      if (coalescedComm_ == nullptr) {
        coalescedComm_ = ncclComm;
      } else {
        // 否则验证 coalescedComm_ 是否与 ncclComm 相同
        TORCH_CHECK(coalescedComm_ == ncclComm, MULTI_DEVICE_ERROR_MSG);
      }
    }
    
    // 多次使用，因此我们保存 unordered_map 查找结果
    auto ncclStream = ncclStreams_.at(key);
    
    // 让 NCCL 流首先等待输入张量分配流
    syncStream(device, ncclEvents_[key], ncclStream);
    
    // 初始化工作对象，用于输入、输出张量的操作
    auto work = initWork(
        device, rank_, opType, profilingTitle, inputs, outputs, /*record=*/true);
    
    // 存储输出的引用，供 WorkNCCL::result 和 operator<< 使用
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);
    
    // 如果避免记录流
    if (avoidRecordStreams) {
      work->stashed_for_allocator_safety_ =
          std::make_shared<std::vector<at::Tensor>>(inputs);
    }
    
    // CUDA 可选的设备保护器
    at::cuda::OptionalCUDAGuard gpuGuard;
    
    // 在 ncclGroupStart() 内部之前应只记录开始事件
    if (work->timingEnabled_) {
      work->ncclStartEvent_->record(ncclStream);
    }
    
    // 获取 ncclComm_t 对象
    ncclComm_t comm = ncclComm->getNcclComm();
// TODO(kwen2501): this should be moved to c10d tests, to qualify a NCCL
// upgrade. Once a NCCL version is qualified, this code should not be needed at
// runtime.
#ifdef PGNCCL_ENABLE_HASH
  // 检查是否启用了 Collective Hash Debug，并且如果是，打印输入数据的哈希签名信息
  if (enableCollecticeHashDebug_.load()) {
    // 获取输入张量的元素数量
    auto numel = getTensorsNumel(inputs);
    // 计算输入张量的哈希值
    auto hashValue = hashTensors(inputs);
    // 打印收集哈希签名信息，包括操作类型、元素数量和哈希值
    PRINT_COLLECTIVE_HASH_SIGNATURE(
        "input", opTypeToString(opType), numel, hashValue);
  }
#endif

  {
    // 自动管理 NCCL 通信组，确保正确使用非阻塞模式
    torch::cuda::nccl::AutoNcclGroup nccl_group_guard(
        comm, nccl_use_nonblocking());
    // 遍历所有输入张量
    for (const auto i : c10::irange(inputs.size())) {
      // 如果不避免记录流信息
      if (!avoidRecordStreams) {
        // 如果输入张量不是稀疏张量，记录其数据指针和当前的 ncclStream
        if (!inputs[i].is_sparse()) {
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].storage().data_ptr(), ncclStream);
        } else {
          // 对于稀疏输入张量，记录索引和值张量的数据指针和当前的 ncclStream
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].values().storage().data_ptr(), ncclStream);
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].indices().storage().data_ptr(), ncclStream);
        }
      }
      // 如果 NCCL 不支持非阻塞通信，则调用 fn 函数执行操作，并检查结果
#ifndef NCCL_HAS_COMM_NONBLOCKING
      C10D_NCCL_CHECK(
          fn(inputs[i], outputs[i], comm, ncclStream),
          ncclComm->getNcclCommFailureReason());
#else
      // 否则，调用带超时的 fn 函数执行操作，并检查结果
      C10D_NCCL_CHECK_TIMEOUT(
          fn(inputs[i], outputs[i], comm, ncclStream),
          comm,
          ncclComm->getNcclCommFailureReason());
#endif
    }
  }

  // 记录 ncclEndEvent 事件完成于 ncclStream
  work->ncclEndEvent_->record(ncclStream);
  // 将 ncclComm 设置到 work 对象中
  work->ncclComm_ = ncclComm;

  {
    // 使用 ncclStream 创建多流守护器，确保后续操作在正确的流上执行
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
    // 指定 devices 为当前设备，创建一个 future 对象
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // 如果需要记录函数结束回调，则添加回调函数到 future 对象中
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false 允许我们跳过 ivalue::Future 中的同步操作，
          // 但仅在 lambda 函数不使用 "Future" 参数时有效。
          /*uses_future=*/false);
    }
  }
  work->future_->markCompleted(at::IValue(*work->outputs_));
  // 标记工作完成，并将输出值存储在对应的 future 中

  // 设置工作的各项参数
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;

  // 记录调试信息的大小。由于每个进程的多设备操作已经不推荐使用，因此仅记录第一个设备上的大小信息。
  work->numelIn_ = inputs[0].numel();
  work->numelOut_ = outputs[0].numel();

  /* Note [cuda graph capture and workEnqueue]
  
  当 CUDA 图形记录活动时，C10D 看门狗的正常行为是定期查询工作对象上的 CUDA 事件。
  但是，当 CUDA 图形记录活动时，这些事件查询可能会导致崩溃或损坏记录。

  为了确保我们不会在 CUDA 图形捕获活动时将工作对象排队到看门狗中，我们使用一种单向同步方式。
  我们预先增加一个标志，指示我们意图将工作对象排队。然后我们检查 capture_status 来查看：
  (a) 捕获是否已经在进行中（在这种情况下，我们不能排队），
  (b) 捕获尚未开始，因此我们可以信任不会开始捕获（因为开始捕获的先决条件是检查事件查询计数为 0）。

  如果由于捕获正在进行而无法排队工作，则最终会递减计数器。

  因此，我们不能轻易地将增量移动到 workEnqueue 内部，除非我们还更改 workEnqueue 的语义为 'maybeWorkEnqueue'。

  TODO:
   - 在这种情况下，我们的飞行记录器设计是否安全？我们在 CUDA 图形捕获期间记录了任何飞行记录事件吗？如果是这样，它们将不安全地用于轮询完成状态。
  */
  at::cuda::CUDAGraph::inc_pending_event_queries();

  // 如果 capture_status 为 None，则将工作对象排队
  if (capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
  } else {
    // 否则，递减事件查询计数器
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  // TODO（whc）如果工作对象未排队，我对返回它并不满意，因为用户代码与之交互时不会正常工作——例如，他们不会观察到工作完成。这会在捕获过程中导致潜在问题吗？
  // 返回工作对象
  return work;
  // 如果启用 NaN 检查，则对张量进行 NaN 检查
  if (enableNanCheck_) {
    checkForNan(tensor);
  }

  // 如果 avoidRecordStreams_ 为 true，则输出警告信息，说明避免记录流在点对点集体操作中无效
  if (avoidRecordStreams_) {
    TORCH_WARN_ONCE(
        "TORCH_NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point "
        "collectives.");
  }

  // 获取张量所在设备
  auto device = getDevice(tensor);
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;

  // 检查是否处于批量 P2P 操作中
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  if (batchP2P) {
    // 对于批量 P2P，需要像集体操作一样选择通信器，因为除了本地等级和对等点外，其他等级可能会调用此批处理
    key = getKeyFromDevice(device);
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    // 对于单一 P2P，保留旧的两个等级行为（以避免性能差异）
    key = getKeySendRecv(rank_, peer);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;

    // 如果未处于聚合状态，则增加 P2P 序列号
    if (!coalescing_state_) {
      seqP2P_++;
    }
  }

  // 增加逻辑操作计数器
  op_id_++;

  // 获取或创建 NCCL 通信器
  auto ncclComm = getNCCLComm(key, device, opType, p2pRank, isSendRecvSelf);

  // 如果处于聚合状态，设置相应的状态并检查设备一致性
  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = ncclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == ncclComm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  // 获取当前设备的 NCCL 流
  auto ncclStream = ncclStreams_.at(key);

  // 将 NCCL 流与输入张量的分配流进行同步
  syncStream(device, ncclEvents_[key], ncclStream);

  // 创建 NCCL 工作对象
  c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> work;
  if (coalescing_state_) {
    // 当处于聚合状态时，在缺乏时序/状态的每个操作上记录事件
    // 在 startCoalescing 中将会增加聚合操作计数器
    // 记录跟踪信息，以便后续的状态更新和时间监控。对于收集操作，记录进程组名称、描述、序列信息、操作ID、性能分析标题和相关张量。
    auto trace_id = NCCLTraceBuffer::get()->record(
        uid_,
        std::make_tuple(pg_name_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        nullptr,
        nullptr,
        options_->timeout,
        /*isP2P=*/true);
    // 如果希望通过代理更新合并组的工作对象来更新每个P2P操作的FlightRecorder条目的时间和状态，可以累积这些trace_id，并请求FlightRecorder从一个工作对象获取更新，并将其应用于多个条目。
    (void)trace_id;
  } else {
    // 存储输出的引用，以便WorkNCCL::result和operator<<使用。注意，这些输出仅对recv()有效，因为send()不修改输入，但我们仍然为诸如性能分析等用例创建这些输出。

    // 初始化工作对象，传入设备、排名、操作类型、性能分析标题、输入张量和空的输出张量，且不记录此操作。
    work = initWork(
        device, rank_, opType, profilingTitle, {tensor}, {}, /*record=*/false);
    // 绕过一些在Work()中可能因为将{tensor}作为输出而导致崩溃的内容，具体原因不明。
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
    work->outputs_->push_back(tensor);
    // 因为我们没有将输出{tensor}传递给initWork，所以告诉initWork不要记录，并手动调用record传递所有所需信息。
    work->trace_id_ = NCCLTraceBuffer::get()->record(
        uid_,
        std::make_tuple(pg_name_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        work->ncclStartEvent_.get(),
        work->ncclEndEvent_.get(),
        options_->timeout,
        /*isP2P=*/true);
  }

  // 是否需要gpuGuard用于下面的if块，或者可以交换它们？
  at::cuda::OptionalCUDAGuard gpuGuard;

  if (!coalescing_state_) {
    // 只有在启用时间监控时才记录开始事件。
    if (work->timingEnabled_) {
      work->ncclStartEvent_->record(ncclStream);
    }

    // 在ncclStream上进行预处理。
    pre(ncclStream, work);
  }

  // 发送张量和接收张量都在工作流中创建，并在不同的ncclStream中使用。因此，必须记录ncclStream，以防在集合完成之前被释放。
  //
  // 参见[Sync Streams]。
  c10::cuda::CUDACachingAllocator::recordStream(
      tensor.storage().data_ptr(), ncclStream);

  // 这部分似乎对P2P和合并P2P使用都是共同的？
  ncclComm_t comm_ = ncclComm->getNcclComm();
#ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果编译选项不支持非阻塞通信，则调用fn函数执行操作，检查通信失败原因
  C10D_NCCL_CHECK(
      fn(tensor, comm_, ncclStream, p2pTargetRank),
      ncclComm->getNcclCommFailureReason());
#else
  // 如果编译选项支持非阻塞通信，则调用fn函数执行操作，带有超时检查，检查通信失败原因
  C10D_NCCL_CHECK_TIMEOUT(
      fn(tensor, comm_, ncclStream, p2pTargetRank),
      ncclComm->getNcclComm(),
      ncclComm->getNcclCommFailureReason());
#endif

  // 如果不进行数据合并状态检查
  if (!coalescing_state_) {
    // 在ncclStream上提交后处理操作
    post(ncclStream);

    // 在调用ncclGroupEnd()之后记录结束事件
    work->ncclEndEvent_->record(ncclStream);
    // 设置工作对象的通信对象、阻塞等待标志、操作超时时间、存储等信息
    work->ncclComm_ = ncclComm;
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
    // 记录调试信息的大小，仅在第一个设备上记录大小，多设备每个进程已弃用
    work->numelIn_ = work->numelOut_ = tensor.numel();

    // 创建Future对象，并标记为已完成，使用输出作为标记
    {
      c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
      std::vector<at::Device> devices{device};
      work->future_ = c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()), devices);
      work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    // 添加一个回调函数，运行结束函数回调
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false 允许我们跳过ivalue::Future中的同步，只要lambda不使用"Future"参数即可
          /*uses_future=*/false);
    }
  }

  // 将P2P操作加入队列，以便可以通过NCCL看门狗取消
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();

  // 在预先检查捕获状态之前通知图形
  at::cuda::CUDAGraph::inc_pending_event_queries();

  // 如果不进行数据合并状态检查，并且捕获状态为None
  if (!coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None) {
    // 将工作对象加入工作队列，并返回工作对象
    workEnqueue(work);
    return work;
  } else {
    // 减少待处理事件查询计数，并返回空指针
    at::cuda::CUDAGraph::dec_pending_event_queries();
    return nullptr;
  }
}
// 对象方法，执行点对点通信操作
c10::intrusive_ptr<Work> ProcessGroupNCCL::pointToPoint(
    at::Tensor& tensor, // 输入张量
    Fn fn, // 函数对象
    int peer, // 对端节点
    OpType opType, // 操作类型
    const char* profilingTitle) { // 用于性能分析的标题
  // 调用另一重载的pointToPoint方法，传递参数并返回结果
  return pointToPoint(
      tensor, // 输入张量
      fn, // 函数对象
      peer, // 对端节点
      opType, // 操作类型
      [](at::cuda::CUDAStream&, // 空的CUDA流处理器函数
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {}, // 空的处理工作函数
      [](at::cuda::CUDAStream&) {}, // 空的CUDA流处理函数
      profilingTitle); // 用于性能分析的标题
}

// 对象方法，执行稀疏张量的全局归约操作
c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors, // 输入张量列表
    const AllreduceOptions& opts) { // 全局归约选项
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG); // 检查张量列表长度是否为1
  auto tensor = tensors.back(); // 获取最后一个张量
  TORCH_CHECK(
      !isFloat8Type(tensor.scalar_type()), // 检查张量的数据类型是否为Float8
      "Float8 dtypes are not currenlty supported for NCCL reductions"); // 如果是Float8类型，抛出错误

  // 如果定义了IS_NCCLX，进行张量的整合
  tensor = tensor.coalesce();
  at::Tensor outputTensor =
      torch::zeros(tensor.sizes(), tensor.options().layout(torch::kStrided)); // 创建输出张量
  // 调用collective方法执行归约操作，返回工作对象
  auto work = collective(
      tensor, // 输入张量
      outputTensor, // 输出张量
      [&](at::Tensor& input, // 归约操作的lambda函数
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type()); // 获取NCCL数据类型
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm); // 获取NCCL归约操作类型

        size_t num_elements = output.numel(); // 计算输出张量的元素数量
        auto indices = input.indices(); // 获取稀疏张量的索引
        auto sizes = input.sizes(); // 获取稀疏张量的大小
        int colSize = sizes[1]; // 获取稀疏张量的列大小
        auto rows = indices[0]; // 获取稀疏张量的行索引
        size_t blockCount = rows.sizes()[0]; // 计算块数量
        auto recvIndices = indices[0] * colSize; // 计算接收索引

        // 防止输出张量和接收索引在流外释放
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), stream); // 记录输出张量在流中的使用
        c10::cuda::CUDACachingAllocator::recordStream(
            recvIndices.storage().data_ptr(), stream); // 记录接收索引在流中的使用
        auto result = ncclAllReduceSparseBlock(
            input._values().data_ptr(), // 发送缓冲区
            recvIndices.data_ptr<int64_t>(), // 接收索引
            blockCount, // 块数量
            colSize, // 块长度
            output.data_ptr(), // 接收缓冲区
            output.numel(), // 接收数量
            ncclDataType, // NCCL数据类型
            ncclReduceOp, // NCCL归约操作类型
            comm, // NCCL通信器
            stream.stream()); // CUDA流
        return result; // 返回操作结果
      },
      [](at::cuda::CUDAStream& ncclStream, // 空的NCCL流处理器函数
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {}, // 空的工作处理函数
      [&](at::cuda::CUDAStream& ncclStream,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        // 将输出张量转换为稀疏张量并重新转换为张量
        at::cuda::CUDAStreamGuard guard(ncclStream); // CUDA流保护
        if (opts.sparseIndices.has_value()) {
          tensor = at::sparse_coo_tensor(
              opts.sparseIndices.value(), outputTensor, tensor.sizes()); // 创建COO格式的稀疏张量
        } else {
          tensor = outputTensor.to_sparse(); // 将输出张量转换为稀疏张量
        }
      },
      OpType::_ALLREDUCE_SPARSE, // 操作类型为稀疏张量全局归约
      "nccl:all_reduce_sparse"); // 用于性能分析的标题
  return work; // 返回工作对象
}
#else
  // 如果 nccl 分支不是 "exp"，则抛出错误
  C10_THROW_ERROR(
      Error,
      "NCCL does not support all_reduce with sparse tensors. Please use dense tensors instead.");
#endif
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_impl(
    at::Tensor& tensor,
    const AllreduceOptions& opts) {
  // 调用 collective() 函数，对 tensor 执行 allreduce 操作
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        // 获取输入 tensor 的数据类型对应的 nccl 数据类型
        auto ncclDataType = getNcclDataType(input.scalar_type());
        // 根据选项获取 nccl 的 reduce 操作类型
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        // 调用 ncclAllReduce 执行 NCCL 的 AllReduce 操作
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE, // 操作类型为 ALLREDUCE
      "nccl:all_reduce"); // 操作的名称为 nccl:all_reduce
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // 检查 tensors 数组的大小是否为 1，否则抛出错误 MULTI_DEVICE_ERROR_MSG
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // 获取最后一个 tensor
  auto tensor = tensors.back();
  // 如果 tensor 是复数类型，检查是否允许 opts.reduceOp 的操作，并将 tensor 转换为实部视图
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "all_reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  // 检查 tensor 是否只存在于单个 GPU 上
  check_gpu_single_tensor(tensor);

  // 如果 intraNodeComm_ 不为空且 opts.reduceOp 为 ReduceOp::SUM，则选择算法进行内节点通信的全reduce
  if (intraNodeComm_ != nullptr && opts.reduceOp == ReduceOp::SUM) {
    using namespace intra_node_comm;
    // 选择算法进行全reduce操作
    auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);
    if (algo != intra_node_comm::AllReduceAlgo::NONE) {
      // 执行内节点通信的全reduce操作
      intraNodeComm_->allReduce(tensor, algo);
      return c10::make_intrusive<IntraNodeCommWork>();
    }
  }
  // 检查 tensor 的数据类型是否为 Float8，不支持 Float8 数据类型进行 NCCL reductions
  TORCH_CHECK(
      !isFloat8Type(tensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");
  // @lint-ignore CLANGTIDY
  // 记录参数通信数据，包括序列号、进程组名称描述、输入输出 tensors、rank、collective 名称、数据量等
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_name_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  // 调用 allreduce_impl 进行实际的 allreduce 操作并返回结果
  return allreduce_impl(tensor, opts);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    // 根据传入的选项对多个张量进行联合全局归约操作
    return collectiveCoalesced(
        tensors,  // 输入张量列表
        tensors,  // 输出张量列表（与输入相同）
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // 根据输入张量类型获取相应的 NCCL 数据类型
          auto ncclDataType = getNcclDataType(input.scalar_type());
          // 根据选项中的 reduceOp 和输入张量确定 NCCL 归约操作类型
          auto ncclReduceOp =
              getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
          // 执行 NCCL 全局归约操作
          return ncclAllReduce(
              input.data_ptr(),
              output.data_ptr(),
              input.numel(),
              ncclDataType,
              ncclReduceOp,
              comm,
              stream.stream());
        },
        OpType::COALESCED,  // 操作类型为 COALESCED
        "nccl:allreduce_coalesced");  // 日志记录标识符为 nccl:allreduce_coalesced
    
    // 避免记录流的注释：collective() 函数将会存储张量数据。
}

// 函数结束标志，表示 ProcessGroupNCCL 类的 broadcast 方法结束

c10::intrusive_ptr<Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  // 检查输入张量数量是否为1，否则抛出异常
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // 取出最后一个张量作为操作对象
  auto tensor = tensors.back();
  // 如果张量是复数类型，将其视作实部张量
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  // 检查 GPU 上是否只有一个张量，否则抛出异常
  check_gpu_single_tensor(tensor);

  // @lint-ignore CLANGTIDY
  // 记录参数与通信数据，用于性能分析和调试
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // 为了匹配集合操作，使用序列号加1
      std::make_tuple(pg_name_, pg_desc_), // PG 名称元组
      tensors, // 输入张量列表
      tensors, // 输出张量列表
      opts.rootRank, // 根节点的排名
      "broadcast", // 集合操作名称
      tensor.numel(), // 输入元素数量
      tensor.numel(), // 输出元素数量
      tensor.scalar_type(), // 数据类型
      std::vector<int64_t>(), // 输入分割尺寸
      std::vector<int64_t>(), // 输出分割尺寸
      globalRankStart, // 全局排名起始点
      globalRankStride, // 全局排名步长
      this->getSize()); // 世界大小

  // 如果避免记录流或者 opts 不是异步操作，则避免记录流为真
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  // 调用 collective 方法执行集合操作，返回结果
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        // 计算根节点在 opts.rootRank 上的位置
        const auto root = opts.rootRank + opts.rootTensor;
        // 使用 NCCL 执行广播操作
        return ncclBcast(
            input.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST, // 操作类型为广播
      "nccl:broadcast", // NCCL 广播名称
      avoidRecordStreams); // 是否避免记录流的标志
}

// _broadcast_oop 在 PGNCCL 中增加了一个 out-of-place 广播操作
// 自定义集合操作可以通过合并广播操作来实现
// 一个用例是实现向量的全收集（all_gather_v）
// 其中不均匀大小的输入在参与的排名中收集
// 由于 all_gather 提供了一个 out-of-place 的 API，因此在 pg_nccl.all_gather 中也需要支持 out-of-place，
// 因此需要添加一个 out-of-place 的广播操作
c10::intrusive_ptr<Work> ProcessGroupNCCL::_broadcast_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const BroadcastOptions& opts) {
  // 如果输出张量元素数量不等于输入张量元素数量，则执行以下操作
  if (outputTensor.numel() != inputTensor.numel()) {
    # 抛出值错误异常，指示输入和输出的张量在_broadcast_oop中必须具有相同数量的元素
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }

  # 返回集合操作的结果
  return collective(
      # 将输入张量和输出张量传递给lambda函数进行处理
      inputTensor,
      outputTensor,
      # lambda函数，接受输入、输出张量、通信对象和CUDA流
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        # 计算根的位置，加上选项中的根张量偏移量
        const auto root = opts.rootRank + opts.rootTensor;
        # 调用NCCL库的广播函数，广播输入张量数据到输出张量
        return ncclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST,  # 操作类型为广播
      "nccl:_broadcast_oop");  # 操作的标识符为nccl:_broadcast_oop
}

// 定义 ProcessGroupNCCL 类中的 reduce 方法，用于执行归约操作
c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  // 检查输入张量列表是否只有一个张量
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  // 取出张量列表中的最后一个张量作为操作对象
  auto tensor = tensors.back();
  // 如果张量是复数类型
  if (tensor.is_complex()) {
    // 检查是否支持将复数张量视为实数进行操作
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    // 将复数张量视为实数张量
    tensor = at::view_as_real(tensor);
  }
  // 检查操作的张量是否在 GPU 上，并且是单一张量
  check_gpu_single_tensor(tensor);
  // 记录通信数据的参数信息，用于性能分析和调试
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_name_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "reduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  // 调用 collective 方法执行集体操作，将结果返回
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        // 计算根节点的索引
        const auto root = opts.rootRank + opts.rootTensor;
        // 获取输入张量的 NCCL 数据类型
        auto ncclDataType = getNcclDataType(input.scalar_type());
        // 获取 NCCL 归约操作类型
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        // 调用 NCCL 执行归约操作
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      "nccl:reduce");
}

// _reduce_oop 方法用于实现 ProcessGroupNCCL 类中的 out-of-place 归约操作
// 自定义的集体操作可以通过合并归约操作来实现
// 一个常见的用例是实现向量的 reduce_scatter_v 操作
// reduce_scatter 提供了 out-of-place 的 API，因此需要添加 out-of-place 归约操作
c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceOptions& opts) {
  // 如果输出张量和输入张量的元素数量不相等
  if (outputTensor.numel() != inputTensor.numel()) {
    # 抛出值错误，指示输入和输出张量在 _reduce_oop 中必须具有相同数量的元素
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }

  # 返回 collective 函数的结果，该函数接受输入张量、输出张量和一个 lambda 函数作为参数
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        # 计算根节点的全局索引，包括 opts.rootRank 和 opts.rootTensor
        const auto root = opts.rootRank + opts.rootTensor;
        # 获取输入张量的数据类型对应的 NCCL 数据类型
        const auto ncclDataType = getNcclDataType(input.scalar_type());
        # 获取 NCCL 的 reduce 操作类型，根据输入张量的数据类型、NCCL 数据类型和通信方式
        const auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        # 调用 NCCL 进行 reduce 操作，将输入张量的数据减少到输出张量中
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            (int)root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,  # 操作类型为 REDUCE，用于标识 collective 函数中的操作类型
      "nccl:_reduce_oop");  # 提供一个描述，用于标识调用的位置或类型
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,  // 以引用方式传递输出张量的向量的向量
    std::vector<at::Tensor>& inputTensors,  // 以引用方式传递输入张量的向量
    const AllgatherOptions& opts) {  // allgather 方法的选项参数

  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);  // 检查输入张量向量的大小是否为 1，否则抛出异常

  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();  // 获取输入张量向量中的最后一个张量

  check_gpu_single_tensor(inputTensor);  // 检查输入张量是否在 GPU 上

  // @lint-ignore CLANGTIDY
  auto outputTensors_ = outputTensors.back();  // 获取输出张量向量的向量中的最后一个向量

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1),  // 获取组的序列号并加 1，以匹配集体操作的序列号
      std::make_tuple(pg_name_, pg_desc_),  // 创建 PG 名称元组
      inputTensors,  // 输入张量向量
      outputTensors,  // 输出张量向量的向量
      rank_,  // 进程的排名
      "all_gather",  // 集体操作的名称
      inputTensor.numel(),  // 输入张量的元素数
      inputTensor.numel() *  // 输出张量的元素数
          this->getSize(),
      inputTensor.scalar_type(),  // 输入张量的数据类型
      std::vector<int64_t>(),  // 输入分割大小的向量
      std::vector<int64_t>(),  // 输出分割大小的向量
      globalRankStart,  // 全局排名的起始位置
      globalRankStride,  // 全局排名的步幅
      this->getSize());  // 进程组的大小

  bool same_size = check_same_size(outputTensors_);  // 检查输出张量向量是否具有相同的大小

  if (same_size) {
    // 将张量的向量展平为单个堆叠的张量。
    at::Tensor outputFlattened = newLikeFlat(outputTensors_);
    // 返回 collective 函数的结果，该函数执行集合操作，接收多个参数和函数回调
    return collective(
        // 输入张量，即需要在集合操作中收集的数据
        inputTensor,
        // 输出张量的扁平化版本，用于存储集合操作的结果
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // 如果不避免记录流，记录输出张量的流以便后续操作使用
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          // 执行 NCCL 的全局收集操作，将输入数据收集到输出数据中
          return ncclAllGather(
              input.data_ptr(),
              output.data_ptr(),
              input.numel(),
              getNcclDataType(input.scalar_type()),
              comm,
              stream.stream());
        },
        [](at::cuda::CUDAStream& ncclStream,
           c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // 避免记录流的注释：实际上在这里我们不需要存储任何东西
          // - inputTensors 已经在 collective() 中存储在 work->stashed_for_allocator_safety_
          // - outputFlattened 已经在 collective() 中存储在 work->outputs_
          // - 用户可见的 outputTensors 应该由用户持有，直到等待 work_ 结束后
          // 因此，所有参与的张量都已经被考虑到，并且在等待 work_ 结束之前不会释放回其分配的流中。
        },
        [&](at::cuda::CUDAStream& ncclStream,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // 将扁平化的输出张量复制到实际输出张量中
          at::cuda::CUDAStreamGuard guard(ncclStream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            // 参见 [Sync Streams]
            if (!avoidRecordStreams_) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  outputTensors_[j].storage().data_ptr(), ncclStream);
            }
            // 使用拷贝操作将数据从扁平化的输出张量复制到实际输出张量中
            outputTensors_[j].copy_(outputFlattened[j], true);
          }
        },
        // 操作类型为全局收集
        OpType::ALLGATHER,
        // 操作的名称为 nccl:all_gather
        "nccl:all_gather");
  } else {
    // 如果不满足上述条件，执行以下逻辑
    const auto num_reduces = outputTensors_.size();
    // 开始合并操作
    startCoalescing();
    // 遍历需要减少的张量数量
    for (const int i : c10::irange(num_reduces)) {
      auto& output = outputTensors_[i];
      // 根据当前的排名选择输入或输出张量
      auto& input = (i == rank_) ? inputTensor : output;
      // 设置广播的选项
      auto broadcastOpts = BroadcastOptions{
          static_cast<int64_t>(i), static_cast<int64_t>(0), opts.timeout};
      // 执行广播操作
      _broadcast_oop(output, input, broadcastOpts);
    }
    // 结束合并操作，并返回相应的工作对象
    auto work = endCoalescing(OpType::ALLGATHER);
    return work;
  }
}

// 不支持 allgather_coalesced 操作，抛出 NotImplementedError
c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL does not support allgather_coalesced");
}

// 执行 allgather 操作，将结果存储在输出张量中
c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      "nccl:all_gather_into_tensor_coalesced");
}

// 执行 reduce_scatter 操作，将结果存储在输出张量中
c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto outputTensor = outputTensors.back();
  check_gpu_single_tensor(outputTensor);
  // @lint-ignore CLANGTIDY
  auto inputTensors_ = inputTensors.back();
  TORCH_CHECK(
      !isFloat8Type(outputTensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");

  // 记录参数和通信数据
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_name_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "reduce_scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // 检查输入张量是否具有相同大小
  bool same_size = check_same_size(inputTensors_);
  if (same_size) {
    // 将一组张量展平为单个张量
    at::Tensor inputFlattened = newLikeFlat(inputTensors_);
    // 返回 collective 函数的结果，其中包括三个回调函数参数
    return collective(
        // 输入参数 inputFlattened，输出参数 outputTensor
        inputFlattened,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // 如果不避免记录流，则记录输出张量的流
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          // 获取输入张量的 NCCL 数据类型
          const auto ncclDataType = getNcclDataType(input.scalar_type());
          // 获取 NCCL Reduce 操作
          const auto ncclReduceOp =
              getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
          // 执行 NCCL Reduce Scatter 操作
          return ncclReduceScatter(
              input.data_ptr(),
              output.data_ptr(),
              output.numel(),
              ncclDataType,
              ncclReduceOp,
              comm,
              stream.stream());
        },
        [&](at::cuda::CUDAStream& ncclStream,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // 如果避免记录流
          if (avoidRecordStreams_) {
            // 只需要存储 inputTensors 到 work->stashed_for_allocator_safety_
            auto& v = work->stashed_for_allocator_safety_;
            v->insert(v->end(), inputTensors_.begin(), inputTensors_.end());
          }

          // 使用 ncclStream 保护 CUDA 流
          at::cuda::CUDAStreamGuard guard(ncclStream);
          // 遍历 inputTensors 数组
          for (const auto j : c10::irange(inputTensors_.size())) {
            // 如果不避免记录流，则记录 inputTensors[j] 的流
            if (!avoidRecordStreams_) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  inputTensors_[j].storage().data_ptr(), ncclStream);
            }
            // 将 inputTensors[j] 复制到 inputFlattened[j]
            inputFlattened[j].copy_(inputTensors_[j], true);
          }
        },
        [&](at::cuda::CUDAStream&,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        // 操作类型为 REDUCE_SCATTER
        OpType::REDUCE_SCATTER,
        // 标识为 "nccl:reduce_scatter"
        "nccl:reduce_scatter");
  } else {
    // 如果不满足条件，执行以下逻辑
    const auto num_reduces = inputTensors_.size();
    // 开始合并操作
    startCoalescing();
    // 遍历 inputTensors_ 数组
    for (const int i : c10::irange(num_reduces)) {
      // 获取当前输入和输出张量
      auto& input = inputTensors_[i];
      auto& output = (i == rank_) ? outputTensor : input;
      // 设置 reduceOpts 参数
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i),
          static_cast<int64_t>(0),
          opts.timeout};
      // 执行 reduce 操作
      _reduce_oop(output, input, reduceOpts);
    }
    // 结束合并操作，并返回工作对象
    auto work = endCoalescing(OpType::REDUCE_SCATTER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,  // 输出张量的引用
    at::Tensor& inputTensor,   // 输入张量的引用
    const ReduceScatterOptions& opts) {  // ReduceScatterOptions 的常量引用参数

  // 检查输入张量和输出张量的数据类型是否相同，否则抛出类型错误异常
  if (inputTensor.dtype() != outputTensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "input tensor must be the same type as the output tensor.");
  }

  // 检查输入张量的元素数量是否等于输出张量元素数量乘以进程组的大小，否则抛出数值错误异常
  if (inputTensor.numel() != outputTensor.numel() * size_) {
    C10_THROW_ERROR(
        ValueError,
        "input tensor must be the same size as output size times world size");
  }

  // @lint-ignore CLANGTIDY
  const auto& tensor = outputTensor;
  
  // 检查是否支持 Float8 数据类型，如果是则抛出错误，因为 NCCL 不支持 Float8 类型的归约操作
  TORCH_CHECK(
      !isFloat8Type(tensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");

  // 记录参数和通信数据，用于性能分析和调试
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1),  // 为了匹配集体操作，序列号加一
      std::make_tuple(pg_name_, pg_desc_),  // 进程组名称元组
      inputTensor,  // 输入张量
      outputTensor,  // 输出张量
      rank_,  // 进程的排名
      "_reduce_scatter_base",  // 集体操作的名称
      inputTensor.numel(),  // 输入张量元素数量
      tensor.numel(),  // 输出张量元素数量
      tensor.scalar_type(),  // 数据类型
      std::vector<int64_t>(),  // 输入分割大小的空向量
      std::vector<int64_t>(),  // 输出分割大小的空向量
      globalRankStart,  // 全局排名的起始值
      globalRankStride,  // 全局排名的步长
      this->getSize());  // 进程组大小

  // 是否避免记录流的标志，避免记录的情况包括避免记录标志已经设置或者 opts.asyncOp 为 false
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  // 执行集体归约操作
  return collective(
      inputTensor,  // 输入张量
      outputTensor,  // 输出张量
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams) {
          // 如果不避免记录流，则记录输出张量的流
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        // 获取输入张量的 NCCL 数据类型
        auto ncclDataType = getNcclDataType(input.scalar_type());
        // 获取 NCCL 归约操作类型
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        // 执行 NCCL 归约分散操作
        return ncclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::_REDUCE_SCATTER_BASE,  // 操作类型
      "nccl:_reduce_scatter_base",  // NCCL 的集体操作名称
      avoidRecordStreams);  // 是否避免记录流的标志
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter_tensor_coalesced(
    // 接收三个参数：输出张量的向量、输入张量的向量、ReduceScatterOptions 类型的引用
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  // 检查最后一个输入张量是否不是 Float8 类型，如果是则抛出错误
  TORCH_CHECK(
      !isFloat8Type(inputs.back().scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");
  // 调用 collectiveCoalesced 函数进行集合通信，返回结果
  return collectiveCoalesced(
      // 将输入向量、输出向量作为参数传递给 collectiveCoalesced 函数
      inputs,
      outputs,
      // 使用 lambda 表达式定义通信操作
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        // 如果不避免记录流，则使用 CUDACachingAllocator 记录输出张量的数据指针和流
        if (!avoidRecordStreams_) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        // 获取输入张量的 NCCL 数据类型
        auto ncclDataType = getNcclDataType(input.scalar_type());
        // 获取 NCCL 的 reduce 操作类型
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        // 调用 ncclReduceScatter 函数执行 reduce scatter 操作
        return ncclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      // 指定操作类型为 COALESCED
      OpType::COALESCED,
      // 设置操作名称为 "nccl:reduce_scatter_tensor_coalesced"
      "nccl:reduce_scatter_tensor_coalesced");
}

// 实现 ProcessGroupNCCL 类的 barrier 方法，用于执行集合操作屏障
c10::intrusive_ptr<Work> ProcessGroupNCCL::barrier(const BarrierOptions& opts) {
  // 记录通信参数，包括序列号加一以匹配集合操作，PG 名称和描述，rank，集合名称等
  RECORD_PARAM_COMMS(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_name_, pg_desc_), // PG name tuple
      rank_, // rank
      "barrier", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // 存储设备列表
  std::vector<at::Device> devices;

  // 如果 opts 中提供了自定义的 GPU 设备 id，则使用这些设备
  if (!opts.device_ids.empty()) {
    for (auto device : opts.device_ids) {
      devices.emplace_back(at::DeviceType::CUDA, device);
    }
  } else if (usedDeviceIdxs_.empty()) {
    // 如果没有正在调用的 NCCL 集合操作，则使用推测的设备
    // 如果多个进程位于同一节点，使用 rank 确保每个进程在不同的 GPU 上
    auto numGPUs = at::cuda::getNumGPUs();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGPUs);
    LOG(INFO)
        << logPrefix()
        << c10::str(
               " using GPU ",
               deviceIdx,
               " to perform barrier as devices used by this process are currently unknown. ",
               "This can potentially cause a hang if this rank to GPU mapping is incorrect.",
               "Specify device_ids in barrier() to force use of a particular device.");
    devices.emplace_back(guessDeviceForRank());
  } else {
    // 否则，使用已经使用过的 GPU 设备索引
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.emplace_back(at::DeviceType::CUDA, usedDeviceIdx);
    }
  }

  // 只使用最后一个设备
  auto device = devices.back();
  // 创建一个空的 Tensor，设备为指定设备，数据类型为 Byte
  at::Tensor barrierTensor =
      at::empty({1}, at::TensorOptions().device(device).dtype(at::kByte));
  // 执行全局归约以实现屏障操作
  auto work = allreduce_impl(barrierTensor);

  // 将 Work 对象转换为 ProcessGroupNCCL::WorkNCCL 类型，并设置 barrierTensor_
  auto ncclWork = dynamic_cast<ProcessGroupNCCL::WorkNCCL*>(work.get());
  TORCH_CHECK(ncclWork);
  ncclWork->barrierTensor_ = std::move(barrierTensor);
  return work;
}

// 实现 ProcessGroupNCCL 类的 alltoall_base 方法，执行基础的 AllToAll 操作
c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  // 检查输出和输入张量是否都是单个 GPU 张量
  check_gpu_single_tensor(outputTensor, true);
  check_gpu_single_tensor(inputTensor, true);

  // 如果输出和输入的分割大小都为零
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(pg_name_, pg_desc_), // PG name tuple
        inputTensor, // 输入张量，用于收集操作
        outputTensor, // 输出张量，用于收集操作
        rank_, // 当前进程的排名
        "all_to_all", // 收集操作的名称
        inputTensor.numel(), // 输入张量的元素数量
        outputTensor.numel(), // 输出张量的元素数量
        inputTensor.scalar_type(), // 输入张量的数据类型
        std::vector<int64_t>(), // 输入张量分割大小的向量
        std::vector<int64_t>(), // 输出张量分割大小的向量
        globalRankStart, // 全局排名起始值
        globalRankStride, // 全局排名步长
        this->getSize()); // 当前通信组的进程数

    // avoidRecordStreams_ 注意：collective() 函数会存储输入张量和输出张量。
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          torch::cuda::nccl::all2all_single_equal_split(
              input, output, this->getSize(), comm, stream);
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(pg_name_, pg_desc_), // PG name tuple
        inputTensor, // 输入张量，用于收集操作
        outputTensor, // 输出张量，用于收集操作
        rank_, // 当前进程的排名
        "all_to_allv", // 收集操作的名称
        inputTensor.numel(), // 输入张量的元素数量
        outputTensor.numel(), // 输出张量的元素数量
        inputTensor.scalar_type(), // 输入张量的数据类型
        inputSplitSizes, // 输入张量分割大小的向量
        outputSplitSizes, // 输出张量分割大小的向量
        globalRankStart, // 全局排名起始值
        globalRankStride, // 全局排名步长
        this->getSize()); // 当前通信组的进程数

    // avoidRecordStreams_ 注意：collective() 函数会存储输入张量和输出张量。
    // 使用 collective 函数，执行一组输入和输出张量的集体操作
    return collective(
        inputTensor,  // 输入张量
        outputTensor,  // 输出张量
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {  // Lambda 函数，接受输入张量、输出张量、通信对象和 CUDA 流作为参数
          
          // 创建用于存储发送和接收长度以及偏移量的向量
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          
          // 计算输入张量的长度和偏移量
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          
          // 计算输出张量的长度和偏移量
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          
          // 如果不避免记录流，则使用 CUDA 缓存分配器记录流
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          
          // 执行不等分割的单个 NCCL 全对全通信
          torch::cuda::nccl::all2all_single_unequal_split(
              input.data_ptr(),  // 输入数据指针
              send_lengths.data(),  // 发送长度数组的数据指针
              send_offsets.data(),  // 发送偏移量数组的数据指针
              output.data_ptr(),  // 输出数据指针
              recv_lengths.data(),  // 接收长度数组的数据指针
              recv_offsets.data(),  // 接收偏移量数组的数据指针
              input.element_size(),  // 输入元素大小
              input.scalar_type(),  // 输入张量的标量类型
              comm,  // NCCL 通信句柄
              stream);  // CUDA 流

          // 返回 NCCL 操作成功状态
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,  // 操作类型为全对全基础操作
        "nccl:all_to_all");  // 操作的名称为 nccl:all_to_all
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,       // 输出张量向量
    std::vector<at::Tensor>& inputTensors,        // 输入张量向量
    const AllToAllOptions& /* unused */) {        // AllToAllOptions 参数（未使用）

  std::vector<int64_t> inSplitSizes;              // 输入张量分割大小向量
  std::vector<int64_t> outSplitSizes;             // 输出张量分割大小向量
  int64_t total_numel = 0;                        // 张量总元素数量

  auto device = outputTensors[0].device();        // 获取输出张量的设备
  for (const auto r : c10::irange(outputTensors.size())) {  // 循环遍历输出张量向量的索引范围
    check_gpu_single_tensor(outputTensors[r], true);   // 检查单个 GPU 张量是否符合要求
    check_gpu_single_tensor(inputTensors[r], true);    // 检查单个 GPU 张量是否符合要求
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")   // 检查张量是否在同一设备上
    inSplitSizes.push_back(inputTensors[r].numel());    // 将输入张量的元素数量添加到分割大小向量中
    outSplitSizes.push_back(outputTensors[r].numel());  // 将输出张量的元素数量添加到分割大小向量中
    total_numel += inputTensors[r].numel();     // 计算所有输入张量的总元素数量
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1),   // 为匹配集体操作，对组序列号进行加1处理
      std::make_tuple(pg_name_, pg_desc_),   // PG 名称元组
      inputTensors,   // 输入张量
      outputTensors,  // 输出张量
      rank_,  // 当前进程在组中的排名
      "all_to_all",   // 集体操作的名称
      total_numel,   // 输入张量总元素数量
      total_numel,   // 输出张量总元素数量
      inputTensors.front().scalar_type(),   // 数据类型
      inSplitSizes,   // 输入张量分割大小向量
      outSplitSizes,  // 输出张量分割大小向量
      globalRankStart,   // 全局排名起始点
      globalRankStride,  // 全局排名步长
      this->getSize());  // 组大小

  return collective(
      inputTensors[0],    // 第一个输入张量
      outputTensors[0],   // 第一个输出张量
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        torch::cuda::nccl::all2all(outputTensors, inputTensors, comm, stream);  // 执行 NCCL 库的 all2all 操作
        return ncclSuccess;   // 返回 NCCL 操作成功状态
      },
      [&](at::cuda::CUDAStream&,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        if (avoidRecordStreams_) {
          // inputTensor0 and outputTensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), inputTensors.begin(), inputTensors.end());
          v->insert(v->end(), outputTensors.begin(), outputTensors.end());
        }
      },
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},  // 空 Lambda 函数
      OpType::ALLTOALL,   // 操作类型为 ALLTOALL
      "nccl:all_to_all");   // NCCL 的 all_to_all 操作字符串标识
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& tensors,   // 张量向量
    int dstRank,   // 目标排名
    int /* unused */) {  // 函数开始，参数为未使用的整数
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);  // 检查张量列表大小是否为1，否则抛出错误信息

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();  // 获取张量列表中的最后一个张量

  check_gpu_single_tensor(tensor, true);  // 检查并确保张量位于GPU上

  // 记录参数通信数据，包括序列号、进程组名称和描述、输入和输出张量列表、目标排名、集合操作名称、输入和输出元素数量、数据类型、输入和输出分割大小、全局排名起始和步长、世界大小
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1),  // 使用序列号加1以匹配集合操作的序列号
      std::make_tuple(pg_name_, pg_desc_),  // 进程组名称元组
      tensors,  // 输入张量列表
      tensors,  // 输出张量列表
      dstRank,  // 目标排名
      "send",  // 集合操作名称为“send”
      tensor.numel(),  // 输入张量元素数量
      tensor.numel(),  // 输出张量元素数量
      tensor.scalar_type(),  // 数据类型
      std::vector<int64_t>(),  // 空的输入分割大小向量
      std::vector<int64_t>(),  // 空的输出分割大小向量
      globalRankStart,  // 全局排名起始
      globalRankStride,  // 全局排名步长
      this->getSize());  // 获取世界大小

  // 执行点对点通信操作，发送张量到指定目标排名
  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int dst) {
        torch::cuda::nccl::send(input, comm, stream, dst);  // 使用NCCL库发送张量
        return ncclSuccess;  // 返回成功状态
      },
      dstRank,  // 目标排名
      OpType::SEND,  // 操作类型为发送
      c10::str("nccl:send ", rank_, "->", dstRank).c_str());  // NCCL发送的描述字符串

  return ret;  // 返回操作结果
}
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor, true);

  // 记录通信数据的参数
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 以匹配集合操作的顺序
      std::make_tuple(pg_name_, pg_desc_), // PG 名称元组
      tensors, // 输入张量列表
      tensors, // 输出张量列表
      srcRank, // 源排名
      "recv", // 集合操作名称
      tensor.numel(), // 输入张量元素数
      tensor.numel(), // 输出张量元素数
      tensor.scalar_type(), // 数据类型
      std::vector<int64_t>(), // 输入分割大小
      std::vector<int64_t>(), // 输出分割大小
      globalRankStart, // 全局排名起始值
      globalRankStride, // 全局排名步幅
      this->getSize()); // 总排名数目

  // 执行点对点接收操作
  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int src) {
        torch::cuda::nccl::recv(output, comm, stream, src);
        return ncclSuccess;
      },
      srcRank,
      OpType::RECV,
      c10::str("nccl:recv ", rank_, "<-", srcRank).c_str());
  return ret;
}

void ProcessGroupNCCL::groupStart() {
  // 开始 NCCL 通信组
  C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);
  ++ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEnd() {
  // 结束 NCCL 通信组
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  --ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEndNonblocking(std::shared_ptr<NCCLComm> comm) {
  #ifndef NCCL_HAS_COMM_NONBLOCKING
  // 如果不支持非阻塞通信，则直接结束 NCCL 通信组
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  #else
  // 如果支持非阻塞通信
  if (!nccl_use_nonblocking()) {
    // 如果没有启用非阻塞通信，则直接结束 NCCL 通信组
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  } else {
    // 否则，使用超时处理方式结束 NCCL 通信组
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), comm, c10::nullopt);
  }
  #endif
  --ncclActiveGroupCounter_;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  // 用于报告无效参数错误的 lambda 函数
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::gather: " + msg);
  };

  // 断言根排名的有效性
  assertRootRank(invalidArgument, opts.rootRank, size_);

  // 检查输入张量的数量是否为1
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();

  // 准备输出张量的列表
  std::vector<at::Tensor> outputs;

  // 如果当前进程是根进程
  if (getRank() == opts.rootRank) {
    // 如果输出张量列表不止一个张量
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      // 报告无效参数错误
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      // 检查输出张量列表的大小是否与预期的大小相同，如果不同则抛出异常
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    // 获取输入张量的选项和大小
    const auto& options = inputTensor.options();
    const auto& sizes = inputTensor.sizes();
    // 断言输入和输出张量的类型和大小匹配
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    // 将输出设置为第一个输出张量
    outputs = outputTensors[0];
  } else {
    // 如果不在根进程，初始化输出为空列表
    if (outputTensors.size() != 0) {
      // 非根进程不应该有输出张量，否则抛出异常
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // 向列表中添加一个空张量，虽然不使用它，但 `collective` 模板函数需要它来调用其函数
    outputs.emplace_back();
  }

  // 记录通信数据参数，用于性能分析和调试
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 以匹配 collective 的行为
      std::make_tuple(pg_name_, pg_desc_), // 进程组名称元组
      inputTensors, // 输入张量
      outputTensors, // 输出张量
      opts.rootRank, // 根进程的排名
      "gather", // 收集操作的名称
      inputTensor.numel(), // 输入张量的元素个数
      inputTensor.numel() * this->getSize(), // 输出张量的元素个数
      inputTensor.scalar_type(), // 数据类型
      std::vector<int64_t>(), // 输入分割大小
      std::vector<int64_t>(), // 输出分割大小
      globalRankStart, // 全局排名起始值
      globalRankStride, // 全局排名步长
      this->getSize()); // 进程组的大小

  // avoidRecordStreams_ 注意：collective() 将记录 inputTensors 和 outputs，在根进程上 outputs[0] 是关键
  // 执行收集操作
  return collective(
      inputTensor,
      outputs[0], // 仅仅为了符合 collective 接口而存在
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank;
        // 如果当前进程是根进程，记录流
        if (getRank() == root) {
          if (!avoidRecordStreams_) {
            // 对于所有输出张量，记录 CUDA 流
            for (auto output : outputs) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  output.storage().data_ptr(), stream);
            }
          }
        }
        // 执行 GPU 上的 gather 操作
        torch::cuda::nccl::gather(inputTensor, outputs, comm, stream, root);
        return ncclSuccess;
      },
      OpType::GATHER, // 操作类型为 GATHER
      "nccl:gather"); // 使用的通信库和操作名称
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  // 定义一个静态的lambda函数，用于抛出值错误异常
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::scatter: " + msg);
  };

  // 断言根节点的排名在有效范围内
  assertRootRank(invalidArgument, opts.rootRank, size_);

  // 检查输出张量的大小是否为1，否则抛出多设备错误信息
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // 获取输出张量的最后一个张量
  auto outputTensor = outputTensors.back();

  // 定义输入张量的向量
  std::vector<at::Tensor> inputs;

  // 如果当前进程在根节点
  if (getRank() == opts.rootRank) {
    // 检查输入张量向量是否只包含一个元素
    if (inputTensors.size() != 1) {
      // 构建错误信息，要求输入列表应该包含一个包含getSize()个张量的列表
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      // 抛出值错误异常
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      // 构建错误信息，要求输入列表的大小应该与进程组的大小相同
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      // 抛出值错误异常
      invalidArgument(ss.str());
    }

    // 获取输出张量的选项和尺寸
    const auto& options = outputTensor.options();
    const auto& sizes = outputTensor.sizes();
    // 断言输入张量与指定选项和尺寸匹配
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    // 将输入张量列表设置为输入张量的第一个元素
    inputs = inputTensors[0];
  } else {
    // 如果不在根节点，初始化输入张量为空占位符，使用空列表
    if (inputTensors.size() != 0) {
      // 抛出值错误异常，要求非根节点的输入应该为空
      invalidArgument("requires empty input on non-root");
    }
    // 将输入张量列表设置为空列表
    inputs = {};
    // 添加一个空张量到列表，虽然不会使用它，但是`collective`模板函数需要它来调用其函数
    inputs.emplace_back();

将一个空元素添加到名为 `inputs` 的向量中。


  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_name_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

调用 `RECORD_PARAM_COMMS_DATA` 宏，记录通信参数和元数据，包括序列号、PG名称与描述、输入与输出张量、根排名、集体操作类型、元素数等。


  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

计算是否应避免记录流 (`avoidRecordStreams`)，这取决于 `avoidRecordStreams_` 或者 `opts.asyncOp` 是否为假。


  return collective(
      outputTensor,
      inputs[0], // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams) {
            for (auto input : inputs) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  input.storage().data_ptr(), stream);
            }
          }
        }
        torch::cuda::nccl::scatter(inputs, outputTensor, comm, stream, root);
        return ncclSuccess;
      },
      OpType::SCATTER,
      "nccl:scatter",
      avoidRecordStreams);

执行集体通信操作 (`collective`)，使用 `outputTensor` 和 `inputs[0]`，在根节点时根据条件记录流。具体操作为将数据从根节点散布到其他节点。

每个注释都紧跟在对应的代码行后面，解释了该行代码的具体作用和意图。
} // namespace c10d



#endif // USE_C10D_NCCL



c10::intrusive_ptr<Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError, "ProcessGroupNCCL does not support recvAnysource");
}



// 实现接收来自任意源的操作，但在这里抛出未实现的错误，因为ProcessGroupNCCL不支持recvAnysource
c10::intrusive_ptr<Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError, "ProcessGroupNCCL does not support recvAnysource");
}



c10::intrusive_ptr<Work> ProcessGroupNCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  check_gpu_single_tensor(input_tensor);
  check_gpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "output tensor size must be equal to world_size times input tensor size");
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_name_, pg_desc_), // PG name tuple
      input_tensor, // inputTensors
      output_tensor, // outputTensors
      rank_, // rank
      "_allgather_base", // collective name
      input_tensor.numel(), // inNelems
      output_tensor.numel(), // outNelems
      output_tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the NCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `all_gather_into_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  // 调用collective函数执行全聚合操作
  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams) {
          // 如果不避免记录流，则记录输出张量的流
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        // 执行NCCL的全聚合操作
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::_ALLGATHER_BASE,
      "nccl:_all_gather_base",
      avoidRecordStreams);
}
```