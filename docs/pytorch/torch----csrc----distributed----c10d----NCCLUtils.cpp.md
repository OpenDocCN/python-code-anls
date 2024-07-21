# `.\pytorch\torch\csrc\distributed\c10d\NCCLUtils.cpp`

```
// 包含 Torch 分布式的 NCCL 相关头文件
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

// 包含 C10 库的相关头文件
#include <c10/util/CallOnce.h>
#include <c10/util/env.h>
#include <algorithm>

// 如果启用了 USE_C10D_NCCL 宏，则编译以下内容
#ifdef USE_C10D_NCCL
#include <vector>

#include <cuda_runtime.h>
#include <mutex>

// 定义一个私有命名空间，用于实现内部细节隐藏和封装
namespace {
// NCCL 初始化繁忙等待的时间间隔（毫秒）
constexpr int64_t kCommInitBusyWaitMillis = 10;
} // namespace

// 定义 c10d 命名空间
namespace c10d {

// 获取 NCCL 通信器
ncclComm_t NCCLComm::getNcclComm() {
  // 使用互斥锁确保线程安全
  std::unique_lock<std::mutex> lock(mutex_);
  // 如果通信器已中止，则抛出异常
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != c10::nullopt
        ? c10::str(" Original reason for failure was: ", *commFailureReason_)
        : "";
    TORCH_CHECK_WITH(
        DistBackendError,
        false,
        c10::str(
            "NCCL communicator was aborted on rank ",
            rank_,
            ". ",
            commFailureMsg));
  }
  // 如果尚未初始化且启用了非阻塞模式，则等待初始化完成
  if (!initialized_ && nccl_use_nonblocking()) {
    waitUntilInitialized(nccl_nonblocking_timeout());
  }

  // 返回 NCCL 通信器
  return ncclComm_;
}

// 等待 NCCL 初始化完成
void NCCLComm::waitUntilInitialized(int timeoutSecs) {
  auto startTimepoint = std::chrono::steady_clock::now();
  while (!initialized_) {
    if (ncclComm_) {
      ncclResult_t result;
      ncclCommGetAsyncError(ncclComm_, &result);
      // 如果初始化成功，则记录日志并设置初始化标志
      if (result == ncclSuccess) {
        LOG(INFO) << "Rank " << rank_ << ": NCCL communicator is initialized.";
        initialized_ = true;
        break;
      }
    }
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           currentTimepoint - startTimepoint)
                           .count();
    // 如果超时，则抛出异常
    if (timeElapsed > timeoutSecs) {
      std::string err = "NCCL timeout in communicator initialization.";
      TORCH_CHECK_WITH(DistBackendError, false, err);
    }
    // 等待一段时间后重试
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kCommInitBusyWaitMillis));
  }
}

// 如果支持 NCCL_HAS_COMM_SPLIT 并且不定义 FBCODE_CAFFE2，则执行以下内容
std::shared_ptr<NCCLComm> NCCLComm::split(
    NCCLComm* source,
    int color_id,
    int rank,
    ncclConfig_t& config,
    std::vector<uint64_t>& ranks_ull) {
  auto comm = std::make_shared<NCCLComm>();
  // 使用 ncclCommSplit 分割通信器
  C10D_NCCL_CHECK(
      ncclCommSplit(
          source->ncclComm_, color_id, rank, &(comm->ncclComm_), &config),
      c10::nullopt);
  // 增加原通信器的分割计数
  ++source->ncclCommSplitCounter_;
  // 设置分割后的通信器的排名
  comm->rank_ = rank;
  return comm;
}
#endif

// 如果未定义 FBCODE_CAFFE2，则执行以下内容
#ifndef FBCODE_CAFFE2
// 判断是否应广播 NCCL 唯一标识
bool shouldBroadcastNCCLUniqueID(bool isSendRecvSelf) {
  // 对于同一进程内的点对点通信，不需要广播
  return !isSendRecvSelf;
}
#endif

// 获取 NCCL 的版本信息
std::string getNcclVersion() {
  // 使用 c10::call_once 确保获取版本信息的线程安全性
  static c10::once_flag ncclGetVersionFlag;
  static std::string versionString;

  c10::call_once(ncclGetVersionFlag, []() {
    int version;
    // 调用 ncclGetVersion 获取 NCCL 的版本号
    ncclResult_t status = ncclGetVersion(&version);
    // 如果调用未成功返回或者版本号小于100（对应于0.1.0），则无法计算版本号
    if (status != ncclSuccess || version < 100) {
      // 设置版本字符串为 "Unknown NCCL version"
      versionString = "Unknown NCCL version";
    } else {
      // NCCL 从版本2.9开始更改版本编码方式
      // 定义主版本号基数，如果版本小于2900，则为1000，否则为10000
      const int majorBase = version < 2900 ? 1000 : 10000;
      // 定义次版本号基数为100
      const int minorBase = 100;
      // 计算NCCL主版本号
      auto ncclMajor = version / majorBase;
      // 计算NCCL次版本号
      auto ncclMinor = (version % majorBase) / minorBase;
      // 计算NCCL修订版本号
      auto ncclPatch =
          version % (ncclMajor * majorBase + ncclMinor * minorBase);
      // 组合成版本字符串，格式为 "主版本号.次版本号.修订版本号"
      versionString = std::to_string(ncclMajor) + "." +
          std::to_string(ncclMinor) + "." + std::to_string(ncclPatch);
    }
#ifdef NCCL_SUFFIX
      // 如果定义了 NCCL_SUFFIX 宏
      const auto ncclSuffix = std::string(NCCL_SUFFIX);
      // 如果 ncclSuffix 非空
      if (ncclSuffix.length()) {
        // 将 ncclSuffix 添加到版本字符串中
        versionString += "." + ncclSuffix;
      }
#endif
    }
  });

  // 返回最终的版本字符串
  return versionString;
}

#ifdef USE_C10D_NCCL
// 计算一组张量的哈希值
size_t hashTensors(const std::vector<at::Tensor>& tensors) {
  // 初始化哈希值
  size_t hash = 0;
  // 遍历每个张量
  for (auto& tensor : tensors) {
    // 如果张量有元素且有存储
    if (tensor.numel() > 0 && tensor.storage()) {
      // 计算数据大小
      size_t data_size = tensor.storage().nbytes();
      // 如果数据大小大于0且有数据指针
      if (data_size > 0 && tensor.storage().data_ptr()) {
        // 获取源数据指针和目标数据指针
        auto src = static_cast<const char*>(tensor.storage().data_ptr().get());
        char* dst = (char*)std::calloc(data_size, sizeof(char));
        // 在 GPU 上进行设备同步，以便对输出进行哈希
        cudaMemcpy(dst, src, data_size, cudaMemcpyDeviceToHost);
        // 遍历目标数据，更新哈希值
        for (size_t i = 0; i < data_size; ++i) {
          // 使用哈希组合函数更新哈希值
          hash = c10::hash_combine(
              hash, c10::get_hash(((char*)dst)[i], data_size));
        }
        // 释放目标数据内存
        free(dst);
      }
    }
  }
  // 返回计算出的哈希值
  return hash;
}
#endif

// 检查是否使用了非阻塞的 NCCL 通信器
bool nccl_use_nonblocking() {
  // 静态变量，检查环境变量是否设置了 TORCH_NCCL_USE_COMM_NONBLOCKING
  static bool nccl_use_nonblocking_ =
      c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING") == true;
  // 如果启用了非阻塞通信器，发出警告
  if (nccl_use_nonblocking_) {
    TORCH_WARN_ONCE("Using experimental non-blocking NCCL communicator.");
  }
  // 返回是否使用非阻塞通信器的布尔值
  return nccl_use_nonblocking_;
}

// 解析 TORCH_NCCL_NONBLOCKING_TIMEOUT 环境变量的值
int _parse_nccl_nonblocking_timeout() {
  const char* val = getenv("TORCH_NCCL_NONBLOCKING_TIMEOUT");
  int timeout = -1;
  // 如果环境变量存在
  if (val) {
    // 转换环境变量值为整数
    const std::string config(val);
    timeout = std::stoi(config);
    // 如果未使用非阻塞通信器但设置了超时时间，发出警告并置超时为-1
    if (!nccl_use_nonblocking() && timeout > 0) {
      TORCH_WARN(
          "TORCH_NCCL_NONBLOCKING_TIMEOUT has no effect when TORCH_NCCL_USE_COMM_NONBLOCKING is false.");
      timeout = -1;
    }
  }
  // 返回解析出的超时时间
  return timeout;
}

// 获取非阻塞通信的超时时间
int nccl_nonblocking_timeout() {
  // 静态变量，调用解析非阻塞超时函数获取超时时间
  static int timeout = _parse_nccl_nonblocking_timeout();
  // 返回非阻塞通信的超时时间
  return timeout;
}

// 获取带有 NCCL 版本信息的错误描述字符串
std::string ncclGetErrorWithVersion(ncclResult_t error) {
  // 返回 NCCL 错误字符串及其版本信息
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}

// 根据 NCCL 错误码提供更详细的错误描述信息
std::string getNcclErrorDetailStr(
    ncclResult_t error,
    std::optional<std::string> processGroupFailureReason /* = c10::nullopt */
) {
  // 如果提供了进程组失败原因，优先返回该原因
  if (processGroupFailureReason != c10::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  // 获取最后的 NCCL 错误信息
  auto ret = ncclGetLastError(NULL);
  // 如果存在最后的错误信息
  if (ret) {
    err = "\nLast error:\n" + std::string(ret);
  } else {
    err = "\nLast error: Unknown NCCL Error\n";
  }
#endif
  // 根据不同的 NCCL 错误码返回对应的解释
  switch (error) {
    case ncclUnhandledCudaError:
      interpret = "ncclUnhandledCudaError: Call to CUDA function failed.";
      break;
    case ncclSystemError:
      # 当前情况表明发生了 ncclSystemError 错误
      interpret =
          "ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. ";
#ifndef NCCL_REMOTE_ERROR
      // 如果 NCCL_REMOTE_ERROR 宏未定义，则说明 ncclRemoteError 错误处理前可能会被视为 ncclSystemError
      interpret += "It can be also caused by unexpected exit of a remote peer.";
#endif
      // 退出 switch 语句
      break;
    case ncclInternalError:
      // 设置错误解释为内部错误
      interpret = "ncclInternalError: Internal check failed.";
      break;
    case ncclInvalidArgument:
      // 设置错误解释为参数无效
      interpret = "ncclInvalidArgument: Invalid value for an argument.";
      break;
    case ncclInvalidUsage:
      // 设置错误解释为库使用无效
      interpret =
          "ncclInvalidUsage: This usually reflects invalid usage of NCCL library.";
      break;
#ifdef NCCL_REMOTE_ERROR
    case ncclRemoteError:
      // 设置错误解释为远程调用失败，可能是网络错误或远程进程意外退出引起
      interpret =
          "ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.";
      break;
#endif
    default:
      // 默认情况下设置错误解释为未知的 NCCL 错误
      interpret = "Unknown NCCL error!";
  }
  // 返回错误解释和错误代码
  return interpret + err;
}

control_plane::RegisterHandler dumpHandler{
    // 注册名为 "dump_nccl_trace_pickle" 的处理器
    "dump_nccl_trace_pickle",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto params = req.params();
      size_t validParamCount = 0;

      // 合法的参数名称
      const std::string includeCollectivesStr = "includecollectives";
      const std::string includeStackTracesStr = "includestacktraces";
      const std::string onlyActiveStr = "onlyactive";

      // 期望的参数映射，初始状态为期望全部为 true/false
      std::unordered_map<std::string, bool> expectedParams = {
          {includeCollectivesStr, true},
          {includeStackTracesStr, true},
          {onlyActiveStr, false}};

      // 遍历请求中的参数
      for (const auto& [paramName, paramValue] : params) {
        auto it = expectedParams.find(paramName);
        if (it != expectedParams.end()) {
          validParamCount++;
          // 根据参数值设置期望的参数状态
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            // 如果参数值既不是 "true" 也不是 "false"，返回错误响应
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      // 如果有效参数数量小于请求参数数量，返回错误响应
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      // 设置响应内容为 dump_nccl_trace 函数的输出结果
      res.setContent(
          dump_nccl_trace(
              expectedParams[includeCollectivesStr],
              expectedParams[includeStackTracesStr],
              expectedParams[onlyActiveStr]),
          "application/octet-stream");
    }};

void DebugInfoWriter::write(const std::string& ncclTrace) {
  // 打开文件进行写入，使用二进制模式写入数据
  std::ofstream file(filename_, std::ios::binary);

  // 检查文件是否成功打开
  if (!file.is_open()) {
    // 如果无法打开文件，记录错误信息
    LOG(ERROR) << "Error opening file for writing NCCLPG debug info: "
               << filename_;
    return;
  }


# 如果程序执行到这里，直接返回，不执行后续的代码
return;



  file.write(ncclTrace.data(), ncclTrace.size());


# 将 ncclTrace 中的数据写入到文件流 file 中
file.write(ncclTrace.data(), ncclTrace.size());



  LOG(INFO) << "Finished writing NCCLPG debug info to " << filename_;


# 记录信息日志，指示已完成将 NCCLPG 调试信息写入到指定的 filename_ 文件中
LOG(INFO) << "Finished writing NCCLPG debug info to " << filename_;
}

// 获取指定排名的调试信息写入器对象
DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
    // 如果 writer_ 指针为空，则创建新的 DebugInfoWriter 对象
    if (writer_ == nullptr) {
        // 获取文件名前缀，默认为 "/tmp/nccl_trace_rank_"
        std::string fileNamePrefix = getCvarString(
            {"TORCH_NCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/nccl_trace_rank_");
        
        // 使用 std::unique_ptr 在创建后自动删除 writer 对象
        std::unique_ptr<DebugInfoWriter> writerPtr(
            new DebugInfoWriter(fileNamePrefix, rank));
        
        // 注册新创建的调试信息写入器对象
        DebugInfoWriter::registerWriter(std::move(writerPtr));
    }
    // 返回引用的调试信息写入器对象
    return *writer_;
}

// 注册调试信息写入器对象的静态方法
void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
    // 检查是否已经注册了调试信息写入器对象
    TORCH_CHECK_WITH(
        DistBackendError,
        hasWriterRegistered_.load() == false,
        "debugInfoWriter already registered");
    
    // 将注册状态设置为已注册
    hasWriterRegistered_.store(true);
    
    // 移动给定的 writer 对象到静态成员 writer_
    writer_ = std::move(writer);
}

// 初始化静态成员 writer_
std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;

// 初始化静态原子布尔变量 hasWriterRegistered_
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

// 计算两个 CUDA 事件之间的时间间隔
float getDurationFromEvent(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent) {
    // 检查 ncclEndEvent 是否已经查询完成，只有在工作成功完成后才能调用该函数
    TORCH_CHECK(
        ncclEndEvent.query(),
        "getDuration can only be called after work is succeeded.")
    
    // 返回两个 CUDA 事件之间的时间间隔
    return ncclStartEvent.elapsed_time(ncclEndEvent);
}

// c10d 命名空间结束
} // namespace c10d

// 结束条件编译指令，检查是否使用了 C10D_NCCL
#endif // USE_C10D_NCCL
```