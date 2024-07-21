# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupGloo.cpp`

```
#include <c10/util/Exception.h> // 包含异常处理相关头文件
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp> // 包含Gloo后端的进程组头文件

#ifdef USE_C10D_GLOO

#include <torch/csrc/distributed/c10d/GlooDeviceFactory.hpp> // 包含Gloo设备工厂头文件
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // 包含前缀存储头文件
#include <chrono> // 包含时间库
#include <exception> // 包含异常类相关头文件

#ifdef _WIN32
#include <gloo/common/win.h> // Windows特定的Gloo头文件
#include <winsock2.h> // Windows套接字API头文件
#include <ws2tcpip.h> // Windows套接字API头文件
#else
#include <netdb.h> // 网络数据库操作头文件（用于主机名解析）
#include <sys/socket.h> // 套接字操作头文件
#include <unistd.h> // UNIX系统标准头文件
#endif
#include <sys/types.h> // 系统调用相关数据类型定义

#include <type_traits> // 提供了模板元编程支持
#include <utility> // 包含一些通用的实用函数

#include <gloo/allgather.h> // Gloo库中的allgather算法头文件
#include <gloo/allgatherv.h> // Gloo库中的allgatherv算法头文件
#include <gloo/allreduce.h> // Gloo库中的allreduce算法头文件
#include <gloo/alltoall.h> // Gloo库中的alltoall算法头文件
#include <gloo/alltoallv.h> // Gloo库中的alltoallv算法头文件
#include <gloo/barrier.h> // Gloo库中的barrier算法头文件
#include <gloo/broadcast.h> // Gloo库中的broadcast算法头文件
#include <gloo/gather.h> // Gloo库中的gather算法头文件
#include <gloo/reduce.h> // Gloo库中的reduce算法头文件
#include <gloo/scatter.h> // Gloo库中的scatter算法头文件

#include <ATen/ThreadLocalState.h> // ATen库中的线程局部状态头文件
#include <ATen/native/SparseTensorUtils.h> // ATen库中的稀疏张量工具头文件

#include <c10/util/StringUtil.h> // C10库中的字符串工具头文件
#include <c10/util/intrusive_ptr.h> // C10库中的内部指针头文件
#include <c10/util/irange.h> // C10库中的迭代范围头文件
#include <gloo/config.h> // Gloo库的配置头文件
#include <gloo/rendezvous/context.h> // Gloo库中的会合上下文头文件
#include <gloo/rendezvous/prefix_store.h> // Gloo库中的前缀存储头文件

#ifdef _WIN32
#define GENERATE_ALL_TYPES(type, func, ...)      \ // 定义根据类型生成模板函数的宏
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(__VA_ARGS__);                  \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<gloo::float16>(__VA_ARGS__);          \
      break;                                     \
    case ::at::ScalarType::BFloat16:             \
      func<c10::BFloat16>(__VA_ARGS__);          \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
    case ::at::ScalarType::Bool:                 \
      func<uint8_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(__VA_ARGS__);                \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \ // 检查无效的标量类型，如果出现则抛出异常
  }

#define HOST_NAME_MAX 256 // 定义主机名的最大长度
#else
#define GENERATE_ALL_TYPES(type, func, args...)  \ // 定义根据类型生成模板函数的宏
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(args);                         \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(args);                        \
      break;                                     \
    # 根据不同的标量类型选择相应的模板函数并调用，每个 case 分支对应一种标量类型
    case ::at::ScalarType::Half:                 \
      func<gloo::float16>(args);                 \
      break;                                     \
    # 对应标量类型为 BFloat16 时调用相应的模板函数
    case ::at::ScalarType::BFloat16:             \
      func<c10::BFloat16>(args);                 \
      break;                                     \
    # 对应标量类型为 Char 时调用相应的模板函数
    case ::at::ScalarType::Char:                 \
      func<int8_t>(args);                        \
      break;                                     \
    # 对应标量类型为 Byte 或 Bool 时调用相应的模板函数
    case ::at::ScalarType::Byte:                 \
    case ::at::ScalarType::Bool:                 \
      func<uint8_t>(args);                       \
      break;                                     \
    # 对应标量类型为 Int 时调用相应的模板函数
    case ::at::ScalarType::Int:                  \
      func<int32_t>(args);                       \
      break;                                     \
    # 对应标量类型为 Long 时调用相应的模板函数
    case ::at::ScalarType::Long:                 \
      func<int64_t>(args);                       \
      break;                                     \
    # 默认情况下抛出异常，提示标量类型无效
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }
#endif

namespace c10d {

namespace {

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

// 计算剩余时间的毫秒数，基于给定的开始时间、超时时间和等待所有进程标志
std::chrono::milliseconds getRemainingTime(
    steady_clock_time_point startTime,
    const std::chrono::milliseconds& timeout,
    bool waitAllRanks) {
  if (waitAllRanks) {
    // 如果等待所有进程，则直接返回指定的超时时间
    return timeout;
  }
  auto elapsedTime = std::chrono::steady_clock::now() - startTime;
  auto remainingMillis = timeout -
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime);

  // 如果没有剩余时间，返回-1表示给调用者超时
  if (remainingMillis.count() <= 0) {
    return std::chrono::milliseconds(-1);
  }

  return remainingMillis;
}

// 记录错误并使用给定的消息抛出异常，使用LOG(ERROR)
void logAndThrow(
    const std::string& logMessage,
    const std::string& errorMessage) {
  LOG(ERROR) << logMessage;
  TORCH_CHECK(false, errorMessage);
}

// 对于monitoredBarrier，检查剩余时间以完成处理所有进程，并在超时时抛出错误
void checkRemainingTime(
    const std::chrono::milliseconds& monitoredBarrierTimeout,
    const std::chrono::milliseconds& remainingTime,
    const std::vector<int>& processedRanks,
    int currentRank) {
  const std::string kNoRemainingTimeError = c10::str(
      "Rank ",
      currentRank,
      " timed out in monitoredBarrier after ",
      monitoredBarrierTimeout.count(),
      " ms.");
  if (remainingTime.count() < 0) {
    std::string rankInfo;
    if (!processedRanks.empty()) {
      rankInfo = c10::str(
          "Successfully processed ranks: ", c10::Join(", ", processedRanks));
    } else {
      rankInfo = "No ranks successfully processed in monitoredBarrier.";
    }
    auto error = c10::str(kNoRemainingTimeError, "\n", rankInfo);
    // 记录并抛出超时错误
    logAndThrow(error, error);
  }
}

// 根据ReduceOp返回对应的函数指针，用于特定类型的数据T
template <typename T, std::enable_if_t<!std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::BAND:
      // 对于非整数类型的数据，不支持ReduceOp.BAND，抛出异常
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with non-integral dtype");
      break;
    case ReduceOp::BOR:
      // 对于非整数类型的数据，不支持ReduceOp.BOR，抛出异常
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with non-integral dtype");
      break;
    case ReduceOp::BXOR:
      // 对于非整数类型的数据，不支持ReduceOp.BXOR，抛出异常
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with non-integral dtype");
      break;
    case ReduceOp::AVG:
      // Gloo不支持ReduceOp.AVG，抛出异常
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Gloo");
      break;
    case ReduceOp::PREMUL_SUM:
      // Gloo不支持ReduceOp.PREMUL_SUM，抛出异常
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");
      break;
    case ReduceOp::UNUSED:
      break;
  }

  // 处理未预料到的ReduceOp，抛出异常
  TORCH_CHECK(false, "Unhandled ReduceOp");
}
// Bitwise AND function template with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void band(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);  // 将目标指针 c 转换为类型 T*，用于存储结果
  auto ta = static_cast<const T*>(a);  // 将源指针 a 转换为 const T*，用于读取数据
  auto tb = static_cast<const T*>(b);  // 将源指针 b 转换为 const T*，用于读取数据
  for (const auto i : c10::irange(n)) {  // 遍历范围 [0, n) 内的所有索引 i
    tc[i] = ta[i] & tb[i];  // 执行按位与操作，并将结果存储在 tc[i] 中
  }
}

// Bitwise OR function template with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);  // 将目标指针 c 转换为类型 T*，用于存储结果
  auto ta = static_cast<const T*>(a);  // 将源指针 a 转换为 const T*，用于读取数据
  auto tb = static_cast<const T*>(b);  // 将源指针 b 转换为 const T*，用于读取数据
  for (const auto i : c10::irange(n)) {  // 遍历范围 [0, n) 内的所有索引 i
    tc[i] = ta[i] | tb[i];  // 执行按位或操作，并将结果存储在 tc[i] 中
  }
}

// Bitwise XOR function template with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bxor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);  // 将目标指针 c 转换为类型 T*，用于存储结果
  auto ta = static_cast<const T*>(a);  // 将源指针 a 转换为 const T*，用于读取数据
  auto tb = static_cast<const T*>(b);  // 将源指针 b 转换为 const T*，用于读取数据
  for (const auto i : c10::irange(n)) {  // 遍历范围 [0, n) 内的所有索引 i
    tc[i] = ta[i] ^ tb[i];  // 执行按位异或操作，并将结果存储在 tc[i] 中
  }
}

// 将 ReduceOp 转换为对应的函数指针，针对不同的 ReduceOp 类型返回不同的 ReduceFunc
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ReduceFunc(&::gloo::sum<T>);  // 返回 sum 函数的函数指针
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);  // 返回 product 函数的函数指针
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);  // 返回 min 函数的函数指针
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);  // 返回 max 函数的函数指针
    case ReduceOp::BAND:
      return ReduceFunc(&band<T>);  // 返回 band 函数的函数指针
    case ReduceOp::BOR:
      return ReduceFunc(&bor<T>);  // 返回 bor 函数的函数指针
    case ReduceOp::BXOR:
      return ReduceFunc(&bxor<T>);  // 返回 bxor 函数的函数指针
    case ReduceOp::AVG:
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Gloo");  // 不支持 AVG 操作，抛出错误
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");  // 不支持 PREMUL_SUM 操作，抛出错误
      break;
    case ReduceOp::UNUSED:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");  // 如果遇到未处理的 ReduceOp，抛出错误
}

// 设置数据输入到选项对象中，使用模板 T 类型
template <typename T, typename O>
void setInputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setInputs(getDataPointers<T>(tensors), tensors[0].numel());  // 调用 opts 对象的 setInputs 方法，设置输入数据指针和元素数量
}

// 设置单个数据输入到选项对象中，使用模板 T 类型
template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());  // 调用 opts 对象的 setInput 方法，设置单个输入数据指针和元素数量
}

// 设置数据输入到选项对象中，使用模板 T 类型和指定的 counts 数组
template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);  // 调用 opts 对象的 setInput 方法，设置输入数据指针和 counts 数组
}

// 设置数据输入到选项对象中，使用模板 T 类型和指定的 counts 数组
template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);  // 调用 opts 对象的 setInput 方法，设置输入数据指针和 counts 数组
}

// 设置数据输出到选项对象中，使用模板 T 类型
template <typename T, typename O>
void setOutputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setOutputs(getDataPointers<T>(tensors), tensors[0].numel());  // 调用 opts 对象的 setOutputs 方法，设置输出数据指针和元素数量
}

// 设置单个数据输出到选项对象中，使用模板 T 类型
template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());  // 调用 opts 对象的 setOutput 方法，设置单个输出数据指针和元素数量
}
// 将数据输出到指定选项的输出对象中
void setOutput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

// 将数据输出到指定选项的输出对象中（重载版本，处理不同类型的计数）
template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

// 创建一个与给定张量具有相同尺寸和步长的固定内存的张量，存储在CPU上
at::Tensor pinnedLike(at::Tensor& tensor) {
  auto* allocator = at::detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      static_cast<int64_t>(at::detail::computeStorageNbytes(
          tensor.sizes(), tensor.strides(), tensor.dtype().itemsize())),
      allocator,
      /*resizable=*/false);
  return at::empty({0}, tensor.options().device(at::kCPU))
      .set_(storage, 0, tensor.sizes(), tensor.strides());
}

// 初始化一个CUDA流和事件的向量，每个张量对应一个流和事件
// 确保这些流与当前默认流同步，以便新的工作可以与张量的所有操作串行化
void initializeStreamsEvents(
    const std::vector<at::Tensor>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events) {
  streams.reserve(tensors.size());
  events.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    c10::Device device = tensors[i].device();
    c10::impl::VirtualGuardImpl impl(device.type());

    // 在当前流上记录事件
    events.emplace_back(device.type());
    events[i].record(impl.getStream(device));

    // 获取一个非默认流来执行此设备上的异步CUDA操作
    // 确保调用者使用的默认流不会被c10d相关操作占用
    streams.push_back(
        impl.getStreamFromGlobalPool(device, /*isHighPriority=*/true));

    // 确保新流与当前流同步
    events[i].block(streams[i]);

    // 如果张量是稀疏的，需要记录数据指针到新流上
    if (tensors[i].is_sparse()) {
      if (tensors[i].is_coalesced()) {
        impl.recordDataPtrOnStream(
            tensors[i].indices().storage().data_ptr(), streams[i]);
        impl.recordDataPtrOnStream(
            tensors[i].values().storage().data_ptr(), streams[i]);
      } else {
        // 需要先合并，新张量将在刚分配的流上分配，无需单独记录
      }
    } else {
      impl.recordDataPtrOnStream(tensors[i].storage().data_ptr(), streams[i]);
    }
  }
}

// 初始化一个CUDA流的向量，每个设备一个流，并确保这些流与当前默认流同步
// 假设嵌套的张量向量中的张量都在同一设备上
void initializeStreamsEvents(
    std::vector<std::vector<at::Tensor>>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events) {
  // 确保嵌套的张量向量中的张量位于相同的设备上
  for (const auto& tensorgroup : tensors) {
    // 获取第一个张量的设备索引作为基准
    const auto device_id = tensorgroup[0].device().index();
    for (const auto& tensor : tensorgroup) {
      // 检查所有张量的设备索引是否与基准相同，若不同则报错
      if (tensor.device().index() != device_id) {
        TORCH_CHECK(
            false,
            "tensors in the nested tensor vectors need to "
            "be on the same device");
      }
    }
  }

  // 预留空间以容纳流和事件
  streams.reserve(tensors.size());
  events.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    // 获取第一个张量的设备作为当前设备
    c10::Device device = tensors[i][0].device();
    c10::impl::VirtualGuardImpl impl(device.type());
    // 在当前流上记录事件
    events.emplace_back(device.type());
    events[i].record(impl.getStream(device));
    // 获取一个非默认流来执行此输出的异步 CUDA 操作
    // 确保调用者使用的默认流不被 c10d 相关操作占用
    streams.push_back(
        impl.getStreamFromGlobalPool(device, /*isHighPriority=*/true));
    // 确保新流与当前流同步
    events[i].block(streams[i]);

    for (at::Tensor& tensor : tensors[i]) {
      // 张量在不同流上创建，因此必须在此工作中记录新流，以防止在工作完成前释放
      impl.recordDataPtrOnStream(tensor.storage().data_ptr(), streams[i]);
    }
  }
}

const auto kLoopbackAddress = "127.0.0.1";

} // namespace

// 静态方法 execute 实现
void ProcessGroupGloo::AsyncWork::execute(
    const c10::intrusive_ptr<AsyncWork>& work) {
  if (work->recordFunctionBeforeCallback_) {
    work->recordFunctionBeforeCallback_();
  }
  try {
    work->run();
  } catch (...) {
    // 处理异常，完成 Gloo 工作时抛出的异常
    work->finishWorkGlooError(std::current_exception());
    return;
  }

  // FIXME: 在此处调用是因为 Future 完成需要所有工作都同步到 CUDA
  work->synchronize();
  work->finishWorkGloo();
}

// 返回异步工作的结果张量向量
std::vector<at::Tensor> ProcessGroupGloo::AsyncWork::result() {
  TORCH_CHECK(
      isCompleted(),
      "Work needs to be completed before calling result(). "
      "Should call wait() before result().");
  TORCH_CHECK(
      outputTensors_.size() <= 1,
      "work result does not support list of lists, use .getFuture() and value()");
  return outputTensors_.empty() ? std::vector<at::Tensor>()
                                : outputTensors_.at(0);
}

// 获取与异步工作相关联的 Future 对象
c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupGloo::AsyncWork::
    getFuture() {
  return future_;
}

namespace {
// 创建与输出张量向量相对应的 Future 对象
c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() > 1) {
    # 创建一个 c10::ivalue::Future 对象，其中包含一个列表，列表的元素是另一个列表，内部元素是 Tensor 类型
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
    
    
    
    # 创建一个 c10::ivalue::Future 对象，其中包含一个列表，列表的元素是 Tensor 类型
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));
} // namespace

void returnFutureWithOutput(
    c10::intrusive_ptr<c10::ivalue::Future>& future,
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  // 如果输出张量为空，将 future 标记为完成，并返回空的 IValue
  if (outputTensors.empty()) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    return;
  }
  // 如果输出张量的数量大于1，将 future 标记为完成，并返回输出张量的 IValue
  if (outputTensors.size() > 1) {
    future->markCompleted(c10::IValue(outputTensors));
    return;
  }
  // 否则，将 future 标记为完成，并返回第一个输出张量的 IValue
  future->markCompleted(c10::IValue(outputTensors[0]));
}

inline void ProcessGroupGloo::AsyncWork::recordAsyncWorkProfilingInfo(
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  auto recordingFunction =
      std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
  // 如果记录函数是活动状态
  if (recordingFunction->isActive()) {
    // 定义在开始处理前执行的回调函数
    std::function<void()> before_handler =
        [inputTensors, profilingTitle, recordingFunction]() {
          // 工作将由不同的线程开始和完成
          recordingFunction->_setAsync();
          std::vector<c10::IValue> inputs;
          if (inputTensors) {
            inputs.reserve(inputTensors->size());
            // 将输入张量转换为 IValue 存储在 inputs 中
            for (const auto& tensor : *inputTensors) {
              inputs.emplace_back(tensor);
            }
          }
          // 在记录函数中调用 before 方法，传递性能分析标题和输入 IValue 的数组引用
          recordingFunction->before(
              profilingTitle,
              c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
        };
    // 包装并传播线程局部状态，以确保线程安全
    recordFunctionBeforeCallback_ = at::wrapPropagateTLSState(before_handler);
    // 定义在处理结束时执行的回调函数
    std::function<void()> end_handler = [recordingFunction]() {
      // 在记录函数中调用 end 方法，表示处理结束
      recordingFunction->end();
    };
    // 包装并传播线程局部状态，以确保线程安全
    recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
  }
}

ProcessGroupGloo::AsyncWork::AsyncWork(
    std::vector<std::vector<at::Tensor>> outputTensors,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    // 使用异步版本的默认性能分析器实现，替换默认性能分析器
    : Work(-1, opType, nullptr, inputTensors),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors_)),
      seq_(seq) {
  // 如果传入性能分析标题不为空，记录异步工作的性能分析信息
  if (profilingTitle != nullptr) {
    recordAsyncWorkProfilingInfo(profilingTitle, inputTensors);
  }
}

// 返回异步工作的序列号
uint64_t ProcessGroupGloo::AsyncWork::getSequencenumber() const {
  return seq_;
}

// 在发生 Gloo 错误时完成异步工作，将 future 设置为错误状态
void ProcessGroupGloo::AsyncWork::finishWorkGlooError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

// 完成 Gloo 工作时，将输出张量设置到 future 中并完成工作
void ProcessGroupGloo::AsyncWork::finishWorkGloo() {
  returnFutureWithOutput(future_, outputTensors_);
  finish();
}

ProcessGroupGloo::SendWork::SendWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    uint64_t seq)
    : Work(
          -1,                           // 设置 Work 对象的标识为 -1
          OpType::SEND,                 // 设置操作类型为 SEND
          "gloo:send",                  // 设置操作名称为 "gloo:send"
          std::optional<std::vector<at::Tensor>>({tensor})),  // 用给定的 tensor 初始化一个 optional 的 Tensor 向量

      tensor_(tensor),                 // 将传入的 tensor 赋值给成员变量 tensor_
      buffer_(std::move(buffer)),      // 使用移动语义将传入的 buffer 赋值给成员变量 buffer_
      seq_(seq) {}                     // 使用传入的 seq 值初始化成员变量 seq
// 返回当前 SendWork 对象的序列号
uint64_t ProcessGroupGloo::SendWork::getSequencenumber() const {
  return seq_;
}

// 等待发送操作完成，如果超时则抛出异常
bool ProcessGroupGloo::SendWork::wait(std::chrono::milliseconds timeout) {
  bool sendCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      sendCompleted = buffer_->waitSend();
    } else {
      sendCompleted = buffer_->waitSend(timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // 完成工作对象，并抛出异常
  finishAndThrow(exception);
  return sendCompleted;
}

// 中止发送操作等待
void ProcessGroupGloo::SendWork::abort() {
  buffer_->abortWaitSend();
}

// 构造接收工作对象，初始化各成员变量
ProcessGroupGloo::RecvWork::RecvWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle)
    : Work(
          -1,
          opType,
          profilingTitle,
          std::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      buffer_(std::move(buffer)),
      srcRank_(-1),
      seq_(seq) {}

// 返回当前 RecvWork 对象的序列号
uint64_t ProcessGroupGloo::RecvWork::getSequencenumber() const {
  return seq_;
}

// 获取源排名，使用互斥锁保护
int ProcessGroupGloo::RecvWork::sourceRank() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return srcRank_;
}

// 等待接收操作完成，如果超时则抛出异常
bool ProcessGroupGloo::RecvWork::wait(std::chrono::milliseconds timeout) {
  bool recvCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      recvCompleted = buffer_->waitRecv(&srcRank_);
    } else {
      recvCompleted = buffer_->waitRecv(&srcRank_, timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // 完成工作对象，并抛出异常
  finishAndThrow(exception);
  return recvCompleted;
}

// 中止接收操作等待
void ProcessGroupGloo::RecvWork::abort() {
  buffer_->abortWaitRecv();
}

// 构造选项对象，设置后端名称和默认线程数
ProcessGroupGloo::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(GLOO_BACKEND_NAME, timeout), threads(2) {}

namespace {

// 在 Windows 下初始化 WinSock
void socketInitialize() {
#ifdef _WIN32
  ::gloo::init_winsock();
#endif
}

// 检查主机名是否可以解析为可用地址
// 如果无法解析，返回 false；否则返回 true
bool doesHostnameResolveToUsableAddress(const std::string& hostname) {
  socketInitialize();
  struct addrinfo hints {};
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* result = nullptr;
  auto rv = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
  if (rv < 0) {
    return false;
  }
  struct addrinfo* rp = nullptr;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }
    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
#ifdef _WIN32
    // 如果是在 Windows 平台上，使用 closesocket 关闭套接字
    closesocket(fd);
#else
    // 如果不是在 Windows 平台上，使用 close 关闭文件描述符
    close(fd);
#endif
    // 如果 rv 等于 -1，表示出现了错误，继续循环处理下一个地址
    if (rv == -1) {
      continue;
    }
    // 如果 rv 不等于 -1，表示成功执行了操作，跳出循环
    break;
  }
  // 释放地址信息的内存
  freeaddrinfo(result);
  // 返回 rp 是否非空的结果
  return rp != nullptr;
}

} // namespace

std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDeviceForInterface(const std::string& interface_name) {
  // 使用给定的接口名创建网络设备对象
  return ::c10d::GlooDeviceFactory::makeDeviceForInterface(interface_name);
}

std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDeviceForHostname(const std::string& hostname) {
  // 检查指定主机名是否可以解析为可用地址，否则抛出异常
  TORCH_CHECK(
      doesHostnameResolveToUsableAddress(hostname),
      "Cannot resolve ",
      hostname,
      " to a (local) address");
  // 使用指定的主机名创建网络设备对象
  return ::c10d::GlooDeviceFactory::makeDeviceForHostname(hostname);
}

#if defined(__linux__) || defined(_WIN32)
std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDefaultDevice() {
  // 使用主机名解析出网络地址以使用
  // 注意：如果主机名无法解析为地址（例如由于配置错误的 /etc/hosts 文件），则此方法将无法工作。
  socketInitialize();
  std::array<char, HOST_NAME_MAX> hostname{};
  auto rv = gethostname(hostname.data(), HOST_NAME_MAX);
  if (rv != 0) {
    // 如果获取主机名失败，则抛出异常
    C10_THROW_ERROR(DistBackendError, std::strerror(errno));
  }

  // 如果主机名能够解析为可用地址，则使用该地址创建网络设备对象
  if (doesHostnameResolveToUsableAddress(hostname.data())) {
    return ::c10d::GlooDeviceFactory::makeDeviceForHostname(hostname.data());
  }

  // 否则，使用回环地址作为后备方案
  TORCH_WARN_ONCE(
      "Unable to resolve hostname to a (local) address. ",
      "Using the loopback address as fallback. ",
      "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress);
}
#endif

#ifdef __APPLE__
std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDefaultDevice() {
  // 使用主机名解析出网络地址以使用
  // 注意：如果主机名无法解析为地址（例如由于配置错误的 /etc/hosts 文件），则此方法将无法工作。
  const auto hostNameMax = sysconf(_SC_HOST_NAME_MAX);
  auto hostname = std::unique_ptr<char[]>(new char[hostNameMax]);
  auto rv = gethostname(hostname.get(), hostNameMax);
  if (rv != 0) {
    // 如果获取主机名失败，则抛出异常
    C10_THROW_ERROR(DistBackendError, std::strerror(errno));
  }

  // 如果主机名能够解析为可用地址，则使用该地址创建网络设备对象
  if (doesHostnameResolveToUsableAddress(hostname.get())) {
    return ::c10d::GlooDeviceFactory::makeDeviceForHostname(hostname.get());
  }

  // 否则，使用回环地址作为后备方案
  TORCH_WARN_ONCE(
      "Unable to resolve hostname to a (local) address. ",
      "Using the loopback address as fallback. ",
      "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress);
}
#endif

ProcessGroupGloo::ProcessGroupGloo(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),  // 调用基类 Backend 的构造函数，初始化 rank 和 size
      store_(new GlooStore(store)),  // 使用给定的 store 参数创建一个新的 GlooStore 对象，并用 store_ 指针管理它
      options_(std::move(options)),  // 使用 std::move 将 options 移动到 options_ 成员变量中
      stop_(false),  // 初始化 stop_ 成员变量为 false
      collectiveCounter_(0) {  // 初始化 collectiveCounter_ 成员变量为 0

  auto& devices = options_->devices;
  if (devices.empty()) {
    TORCH_CHECK(false, "No device(s) specified");  // 如果 devices 列表为空，抛出错误信息
  }

  // 为每个设备创建并连接一个上下文。
  //
  // 注意，可以多次指定相同的设备，无论是相同的对象还是不同的对象作为同一逻辑设备。
  // 这两种模式都是允许的，只是在性能上会有所不同。
  //
  // 使用相同对象多次意味着所有上下文共享一个 I/O 线程。
  // 如果使用不同对象作为同一逻辑设备，则它们将有独立的 I/O 线程。
  // 如果有一个快速的 NIC 无法被单个 I/O 线程饱和，后一种选项是必需的。
  //
  contexts_.reserve(options_->devices.size());
  for (const auto i : c10::irange(options_->devices.size())) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank_, size_);
    auto store = ::gloo::rendezvous::PrefixStore(std::to_string(i), *store_);
    context->setTimeout(options_->timeout);
    try {
      context->connectFullMesh(store, options_->devices[i]);  // 使用指定的 store 和设备连接上下文
    } catch (const std::runtime_error& e) {
      auto err = e.what();
      // TORCH_CHECK 用于打印 C++ 的堆栈跟踪信息。
      auto msg = c10::str("Gloo connectFullMesh failed with ", err);
      logAndThrow(msg, msg);  // 记录错误信息并抛出异常
    }
    contexts_.push_back(std::move(context));  // 将创建的上下文对象移动到 contexts_ 容器中
  }

  // 每个工作线程将它当前正在处理的 AsyncWork 对象存储在 workInProgress_ 向量中。
  // 它的大小必须等于工作线程的数量，以便它们可以直接使用它们启动时的工作线程索引进行索引。
  workInProgress_.resize(options_->threads);

  threads_.resize(options_->threads);  // 调整线程向量的大小为 options_->threads
  for (const auto i : c10::irange(threads_.size())) {
    threads_[i] = std::thread(&ProcessGroupGloo::runLoop, this, i);  // 创建并启动线程，每个线程运行 ProcessGroupGloo 的 runLoop 方法
  }

  init();  // 调用 init 方法进行初始化
}
}

ProcessGroupGloo::~ProcessGroupGloo() {
  // 获取工作队列锁，确保没有未完成的工作
  std::unique_lock<std::mutex> lock(workMutex_);
  // 等待直到工作队列为空
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });

  // 队列为空，设置停止标志
  stop_ = true;

  // 释放锁，允许线程终止
  lock.unlock();

  // 通知所有等待在生产条件变量上的线程
  workProduceCV_.notify_all();

  // 等待工作线程终止
  for (auto& thread : threads_) {
    thread.join();
  }
}

uint32_t ProcessGroupGloo::nextTag() {
  // 返回下一个集体操作的标签，使用原子操作确保线程安全
  return collectiveCounter_++;
}

std::shared_ptr<::gloo::Context> ProcessGroupGloo::getContext(uint32_t tag) {
  // 根据标签选择并返回上下文对象
  return contexts_[tag % contexts_.size()];
}

void ProcessGroupGloo::runLoop(int workerIndex) {
  // 获取工作队列锁
  std::unique_lock<std::mutex> lock(workMutex_);

  // 在未收到停止信号前持续运行
  while (!stop_) {
    // 如果工作队列为空，则等待生产条件变量
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }

    // 获取队列中的工作任务
    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    // 记录当前正在执行的工作任务
    workInProgress_[workerIndex] = work;
    lock.unlock();

    // 释放锁后通知消费条件变量，以免等待者立即阻塞
    workConsumeCV_.notify_one();

    // 执行异步工作任务
    AsyncWork::execute(work);
    lock.lock();
    // 清空当前正在执行的工作任务
    workInProgress_[workerIndex].reset();
  }
}

void ProcessGroupGloo::enqueue(c10::intrusive_ptr<AsyncWork> work) {
  // 获取工作队列锁
  std::unique_lock<std::mutex> lock(workMutex_);
  // 将工作任务加入队列
  workQueue_.push_back(std::move(work));
  lock.unlock();

  // 释放锁后通知生产条件变量，以免等待者立即阻塞
  workProduceCV_.notify_one();
}

namespace {

class AsyncBroadcastWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncBroadcastWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {inputs},
            OpType::BROADCAST,
            seq,
            "gloo:broadcast",
            inputs),
        context(context),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs{};
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcast(at::Tensor& tensor) {
    // 获取张量的标量类型
    const auto& scalarType = tensor.scalar_type();
    // 设置广播选项
    gloo::BroadcastOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    // 根据标量类型生成所有类型的广播函数
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    // 执行广播操作
    gloo::broadcast(opts);
  }

  void run() override {
    // 对根张量执行广播操作
    broadcast(inputs[rootTensor]);

    // 将根张量的内容复制到非根张量中
    for (const auto i : c10::irange(inputs.size())) {
      if (i == static_cast<size_t>(rootTensor)) {
        continue;
      }
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
};
// 定义 AsyncBroadcastCUDAWork 类，继承自 AsyncBroadcastWork 类
class AsyncBroadcastCUDAWork : public AsyncBroadcastWork {
 public:
  // 构造函数，初始化 AsyncBroadcastCUDAWork 对象
  AsyncBroadcastCUDAWork(
      const std::shared_ptr<gloo::Context>& context,  // 共享指针指向 GLOO 上下文
      std::vector<at::Tensor>& inputs,  // 输入张量的向量
      int rootRank,  // 根节点的排名
      int rootTensor,  // 根张量的索引
      uint32_t tag,  // 标签
      uint64_t seq)  // 序列号
      : AsyncBroadcastWork(context, inputs, rootRank, rootTensor, tag, seq) {  // 调用基类构造函数初始化
    initializeStreamsEvents(inputs, streams, events);  // 初始化流和事件

    // 创建固定在主机端的张量
    tmp = pinnedLike(inputs[rootTensor]);
    c10::OptionalStreamGuard guard;
    if (context->rank == rootRank) {
      guard.reset_stream(streams[rootTensor]);  // 重置流
      tmp.copy_(inputs[rootTensor], /* non_blocking */ true);  // 异步复制根张量到 tmp
    }
  }

  // 覆盖基类的 run 方法
  void run() override {
    // 如果当前进程是根节点
    if (context->rank == rootRank) {
      streams[rootTensor].synchronize();  // 同步流
    }

    // 在主机端张量上执行广播操作
    broadcast(tmp);

    // 启动复制回 CUDA 张量的操作
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);  // 重置流
      inputs[i].copy_(tmp, /* non_blocking */ true);  // 异步复制 tmp 到输入张量
      events[i].record(streams[i]);  // 记录事件到流
    }
  }

  // 覆盖基类的 synchronize 方法
  void synchronize() override {
    // 同步复制回 CUDA 张量的操作
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));  // 阻塞直到事件完成
    }
  }

  at::Tensor tmp;  // 临时张量
  std::vector<c10::Stream> streams{};  // 流向量
  std::vector<c10::Event> events{};  // 事件向量
};
// 定义一个继承自ProcessGroupGloo::AsyncWork的AsyncAllreduceWork类，用于执行异步的allreduce操作
class AsyncAllreduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  // 构造函数，初始化AsyncAllreduceWork对象
  AsyncAllreduceWork(
      const std::shared_ptr<gloo::Context>& context,  // GLOO通信上下文的共享指针
      std::vector<at::Tensor>& inputs,  // 输入张量的向量
      ReduceOp reduceOp,  // 减少操作的类型
      uint32_t tag,  // 操作标签
      uint64_t seq)  // 序列号
      : ProcessGroupGloo::AsyncWork(
            {inputs},  // 调用基类的构造函数，传入输入张量的向量作为参数
            OpType::ALLREDUCE,  // 操作类型为ALLREDUCE
            seq,  // 传入的序列号
            "gloo:all_reduce",  // 操作的名称
            inputs),  // 输入张量的向量
        context(context),  // 初始化成员变量context，指向传入的上下文对象
        inputs(inputs),  // 初始化成员变量inputs，传入的输入张量的向量
        reduceOp(std::move(reduceOp)),  // 初始化成员变量reduceOp，移动传入的减少操作对象
        tag(tag) {}  // 初始化成员变量tag，传入的操作标签

  std::shared_ptr<gloo::Context> context;  // GLOO通信上下文的共享指针
  std::vector<at::Tensor> inputs{};  // 输入张量的向量
  const ReduceOp reduceOp;  // 减少操作的类型
  const uint32_t tag;  // 操作标签

  // 执行allreduce操作，合并并减少输入张量的数据
  void allreduce(std::vector<at::Tensor>& tensors) {
    const auto& scalarType = tensors[0].scalar_type();  // 获取张量的标量类型
    gloo::AllreduceOptions opts(context);  // 创建AllreduceOptions对象，传入通信上下文
    opts.setReduceFunction(getFunction(scalarType, reduceOp));  // 设置减少函数
    opts.setTag(tag);  // 设置操作标签
    GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors);  // 为所有类型生成输出设置
    gloo::allreduce(opts);  // 执行allreduce操作
  }

  // 重写基类的虚函数run，执行allreduce操作
  void run() override {
    allreduce(inputs);  // 调用allreduce函数，执行allreduce操作
  }

  // 模板函数，根据模板类型T获取减少函数
  template <typename T>
  void getFunction(gloo::AllreduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);  // 调用toFunction函数，获取对应的减少函数
  }

  // 获取减少函数，根据数据类型dtype和减少操作op
  gloo::AllreduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp& op) {
    gloo::AllreduceOptions::Func fn;  // 减少函数对象
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);  // 为所有类型生成获取函数设置
    return fn;  // 返回获取的减少函数对象
  }
};

// 定义一个继承自AsyncAllreduceWork的AsyncAllreduceCoalescedWork类，用于执行异步的coalesced allreduce操作
class AsyncAllreduceCoalescedWork : public AsyncAllreduceWork {
 public:
  // 构造函数，初始化AsyncAllreduceCoalescedWork对象
  AsyncAllreduceCoalescedWork(
      const std::shared_ptr<gloo::Context>& context,  // GLOO通信上下文的共享指针
      std::vector<at::Tensor>& inputs,  // 输入张量的向量
      ReduceOp reduceOp,  // 减少操作的类型
      uint32_t tag,  // 操作标签
      uint64_t seq)  // 序列号
      : AsyncAllreduceWork(context, inputs, std::move(reduceOp), tag, seq) {}  // 调用基类的构造函数，初始化成员变量

  // 重写基类的虚函数run，执行coalesced allreduce操作
  void run() override {
    allreduceCoalesced(inputs);  // 调用allreduceCoalesced函数，执行coalesced allreduce操作
  }

 private:
  // 执行coalesced allreduce操作，合并并减少输入张量的数据
  void allreduceCoalesced(std::vector<at::Tensor>& tensors) {
    // 将密集张量压平成一个coalescedTensor
    at::Tensor coalescedTensor = flattenDenseTensors(tensors);
    std::vector<at::Tensor> allreduceInput = {coalescedTensor};  // 创建包含coalescedTensor的向量
    allreduce(allreduceInput);  // 调用基类的allreduce函数，执行allreduce操作

    // 分离并重塑张量
    size_t offset = 0;  // 偏移量初始化为0
    for (at::Tensor& tensor : tensors) {
      const int64_t tensorNumel = tensor.numel();  // 获取张量中的元素数量
      const c10::IntArrayRef tensorShape = tensor.sizes();  // 获取张量的形状
      tensor.copy_(coalescedTensor.slice(0, offset, offset + tensorNumel)  // 复制coalescedTensor的切片到张量
                       .view(tensorShape));  // 将切片视图重塑为张量的形状
      offset += tensorNumel;  // 更新偏移量
    }
  }
};
// AsyncSparseAllreduceWork 类继承自 ProcessGroupGloo::AsyncWork 类，用于执行异步的稀疏全局归约操作。
class AsyncSparseAllreduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  // 构造函数，初始化 AsyncSparseAllreduceWork 对象。
  AsyncSparseAllreduceWork(
      const std::shared_ptr<gloo::Context>& context,  // 共享指针，指向 GLOO 环境上下文
      std::vector<at::Tensor>& inputs,                 // 输入的张量向量
      uint32_t tag,                                    // 操作的标签
      uint64_t seq)                                    // 序列号
      : ProcessGroupGloo::AsyncWork(
            {inputs},                                  // 调用基类构造函数，传入输入张量作为输入数据
            OpType::_ALLREDUCE_SPARSE,                 // 稀疏全局归约操作类型
            seq,
            "gloo:sparse_all_reduce",                  // 操作的名称
            inputs),                                   // 输入的张量向量
        context(context),                              // 初始化成员变量 context
        inputs(inputs),                                // 初始化成员变量 inputs
        tag(tag) {}                                    // 初始化成员变量 tag

  std::shared_ptr<gloo::Context> context;              // GLOO 环境上下文的共享指针
  std::vector<at::Tensor> inputs{};                    // 输入的张量向量
  const uint32_t tag;                                  // 操作的标签

  // SparseTensorMetadata 类，用于管理稀疏张量的元数据
  class SparseTensorMetadata {
   public:
    static constexpr auto dim = 9;                     // 元数据的维度，固定为9

    // 构造函数，从现有的元数据张量构造，以便在收集后从对等方获取结构化的元数据访问
    explicit SparseTensorMetadata(at::Tensor metadata)
        : metadata_(std::move(metadata)),              // 移动构造传入的元数据张量
          data_(metadata_.mutable_data_ptr<int64_t>()) {  // 获取元数据张量的可变数据指针
      AT_ASSERT(metadata_.scalar_type() == at::kLong);  // 断言元数据张量的数据类型为 long
      AT_ASSERT(metadata_.dim() == 1);                  // 断言元数据张量的维度为1
      AT_ASSERT(metadata_.size(0) == dim);               // 断言元数据张量的大小为 dim
    }

    // 从稀疏张量中填充元数据
    void populate_from_sparse_tensor(const at::Tensor& tensor) {
      const auto sparse_dim = tensor.sparse_dim();      // 获取稀疏张量的稀疏维度
      AT_ASSERT(sparse_dim <= 4);                       // 断言稀疏维度不超过4
      for (const auto i : c10::irange(4)) {             // 遍历稀疏维度的前4个维度
        if (i < sparse_dim) {                           // 如果 i 小于稀疏维度
          data_[i] = tensor.size(i);                    // 设置 data_ 数组中对应位置的大小
        }
      }
      const auto dense_dim = tensor.dense_dim();        // 获取稠密维度
      AT_ASSERT(dense_dim <= 4);                        // 断言稠密维度不超过4
      for (const auto i : c10::irange(4)) {             // 遍历稠密维度的前4个维度
        if (i < dense_dim) {                            // 如果 i 小于稠密维度
          data_[i + 4] = tensor.size(sparse_dim + i);   // 设置 data_ 数组中对应位置的大小
        }
      }
      data_[8] = tensor._nnz();                         // 设置 data_ 数组的第8个位置为非零元素数量
    }

    // 获取稀疏张量的大小信息
    std::vector<int64_t> sizes() const {
      std::vector<int64_t> sizes;
      // 稀疏大小
      for (const auto i : c10::irange(4)) {             // 遍历前4个维度
        if (data_[i] <= 0) {                            // 如果大小小于等于0，跳出循环
          break;
        }
        sizes.push_back(data_[i]);                      // 将大小添加到 sizes 中
      }
      // 稠密大小
      for (const auto i : c10::irange(4, 8)) {           // 遍历第5到第8个维度
        if (data_[i] <= 0) {                            // 如果大小小于等于0，跳出循环
          break;
        }
        sizes.push_back(data_[i]);                      // 将大小添加到 sizes 中
      }
      return sizes;                                     // 返回大小向量
    }

    // 获取稀疏张量的非零元素数量
    int64_t nnz() const {
      return data_[8];                                  // 返回 data_ 数组的第8个位置的值
    }

   protected:
    at::Tensor metadata_;                               // 元数据张量
    int64_t* data_;                                     // 数据数组指针
  };

  // 稀疏全局归约通过索引和值的全局收集来实现。
  // 每个进程然后在本地对生成的稀疏张量进行求和。
  // 稀疏张量的 nnz 可能在各个进程中不同，因此首先我们对 nnz 运行全收集，
  // 然后再对 max(nnz) 运行全收集。
  at::Tensor allreduce(std::vector<at::Tensor>& tensors) {
    // TODO: This is a massive hack!  There is some confusion about
    // Variable/Tensor inside the body of this function.  Turning off
    // grad smooths over the confusion for now.  This fixes
    // test/test_c10d_gloo.py ProcessGroupGlooTest.test_sparse_allreduce_basics
    //
    // The correct fix is to stop allocating tensors that are not variables,
    // but to conveniently do this c10d must depend on torch not ATen
    // 在这个函数体内部有关于 Variable/Tensor 的一些混乱。关闭梯度可以暂时解决这个混乱。
    // 这样做修复了 test/test_c10d_gloo.py ProcessGroupGlooTest.test_sparse_allreduce_basics 的问题。
    //
    // 正确的修复方法是停止分配不是变量的张量，但为了方便起见，c10d 必须依赖于 torch 而不是 ATen。
    at::AutoDispatchBelowAutograd guard;
    auto input = tensors[0];
    
    // Perform local reduction if we have multiple inputs.
    // 如果有多个输入，则执行本地归约。
    for (const auto i : c10::irange(1, tensors.size())) {
      input += tensors[i];
    }
    
    // Need to coalesce before we can access indices and values.
    // 在访问索引和值之前，需要合并稀疏张量。
    input = input.coalesce();
    
    // Gather metadata information from all ranks.
    // 从所有秩（rank）中收集元数据信息。
    auto metadata = allgather_metadata(input);
    
    // Sanity check dimensionality across ranks.
    // 检查各秩之间的维度是否一致。
    {
      const auto expected = metadata[context->rank].sizes();
      for (const auto i : c10::irange(context->size)) {
        if (i == context->rank) {
          continue;
        }
        const auto actual = metadata[i].sizes();
        TORCH_CHECK(actual == expected, "Sparse dimensions do not match");
      }
    }
    
    // Gather all indices and all values.
    // 收集所有索引和所有值。
    auto indices = allgather_indices(input, metadata);
    auto values = allgather_values(input, metadata);
    
    // Perform global reduction.
    // 执行全局归约。
    AT_ASSERT(static_cast<int>(indices.size()) == context->size);
    AT_ASSERT(static_cast<int>(values.size()) == context->size);
    auto output = at::sparse_coo_tensor(
        indices[0], values[0], input.sizes(), input.options());
    for (const auto i : c10::irange(1, context->size)) {
      output += at::sparse_coo_tensor(
          indices[i], values[i], input.sizes(), input.options());
    }
    
    // Coalesce for good measure.
    // 最后再进行一次合并。
    return output.coalesce();
    }
    
    void run() override {
    auto output = allreduce(inputs);
    
    // This copy is needed when we run a multi-gpu version of reduce (multiple
    // inputs per rank).
    // 在运行多 GPU 版本的 reduce（每个秩有多个输入）时，需要进行这个复制。
    for (const auto i : c10::irange(inputs.size())) {
      inputs[i].copy_(output);
    }
    }
    
    private:
    std::vector<SparseTensorMetadata> allgather_metadata(
      const at::Tensor& tensor) {
    auto buffer =
        at::zeros({context->size, SparseTensorMetadata::dim}, at::kLong);
    
    // Prepare metadata vector (1 entry per rank)
    // 准备元数据向量（每个秩一个条目）
    std::vector<SparseTensorMetadata> metadata;
    metadata.reserve(context->size);
    for (const auto i : c10::irange(context->size)) {
      metadata.emplace_back(buffer.select(0, i));
    }
    
    // Populate data for this rank
    // 填充此秩的数据
    metadata[context->rank].populate_from_sparse_tensor(tensor);
    
    // Allgather metadata
    // Allgather 元数据
    gloo::AllgatherOptions opts(context);
    opts.setOutput(buffer.mutable_data_ptr<int64_t>(), buffer.numel());
    opts.setTag(tag);
    gloo::allgather(opts);
    return metadata;
  }

  // 所有节点收集索引数据
  std::vector<at::Tensor> allgather_indices(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata) {
    const auto sparseDim = tensor.sparse_dim();

    // 计算每个节点需要收集的数据量
    std::vector<size_t> counts(context->size);
    size_t totalSize = 0;
    for (const auto i : c10::irange(metadata.size())) {
      counts[i] = metadata[i].nnz() * sparseDim;
      totalSize += counts[i];
    }

    // 创建输出张量以保存收集到的索引数据
    auto output = at::empty({static_cast<int64_t>(totalSize)}, at::kLong);

    // tensors copied from cuda may not be contiguous, get a contiguous
    // tensor before use its data_ptr
    // 获取连续的索引张量
    auto input = tensor.indices().contiguous();

    // 收集所有节点的索引数据
    gloo::AllgathervOptions opts(context);
    opts.setInput(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<int64_t*>(input.const_data_ptr<int64_t>()),
        input.numel());
    opts.setOutput(output.mutable_data_ptr<int64_t>(), counts);
    opts.setTag(tag);
    gloo::allgatherv(opts);

    // 按节点编译索引张量
    std::vector<at::Tensor> indices;
    indices.reserve(metadata.size());
    int64_t offset = 0;
    for (const auto& i : metadata) {
      const auto nnz = i.nnz();
      const auto numel = sparseDim * nnz;
      // 切片输出张量以获取每个节点的索引张量
      indices.push_back(
          output.narrow(0, offset, numel).reshape({sparseDim, nnz}));
      offset += numel;
    }

    return indices;
  }

  // 所有节点收集值数据
  std::vector<at::Tensor> allgather_values(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata) {
    // 每个节点有 nnz 个 dense_dim()-维张量
    const auto valueShape = tensor.sizes().slice(tensor.sparse_dim());
    int64_t denseNumel = 1;
    for (auto dim : valueShape) {
      denseNumel *= dim;
    }

    // 计算每个节点需要收集的值数据量
    std::vector<size_t> counts(context->size);
    int64_t totalSize = 0;
    for (const auto i : c10::irange(metadata.size())) {
      counts[i] = metadata[i].nnz() * denseNumel;
      totalSize += static_cast<int64_t>(counts[i]);
    }

    // 创建输出张量以保存收集到的值数据
    auto output = at::empty({totalSize}, tensor.scalar_type());

    // 收集所有节点的值数据
    gloo::AllgathervOptions opts(context);
    // tensors copied from cuda may not be contiguous, get a contiguous
    // tensor before use its data_ptr
    // 获取连续的值张量
    at::Tensor valueTensor = tensor.values().contiguous();
    GENERATE_ALL_TYPES(valueTensor.scalar_type(), setInput, opts, valueTensor);
    GENERATE_ALL_TYPES(
        valueTensor.scalar_type(), setOutput, opts, output, counts);
    opts.setTag(tag);
    gloo::allgatherv(opts);

    // 按节点编译值张量
    std::vector<at::Tensor> values;
    values.reserve(metadata.size());
    int64_t offset = 0;
    // 切片输出张量以获取每个节点的值张量
    for (const auto& i : metadata) {
      const auto nnz = i.nnz();
      const auto numel = denseNumel * nnz;
      values.push_back(
          output.narrow(0, offset, numel).reshape(valueShape));
      offset += numel;
    }
    for (const auto& i : metadata) {
      // 遍历元数据中的每个元素
      const auto nnz = i.nnz();
      // 获取当前元素的非零元素个数
      const auto numel = denseNumel * nnz;
      // 计算当前元素的总元素个数
      auto tensorShape = std::vector<int64_t>({(int64_t)nnz});
      // 创建张量形状向量，初始值包含 nnz (非零元素个数)
      std::copy(
          valueShape.begin(),
          valueShape.end(),
          std::back_inserter(tensorShape));
      // 将 valueShape 的元素复制到 tensorShape 的末尾
      values.push_back(output.narrow(0, offset, numel).reshape(tensorShape));
      // 将 output 张量在第 0 维上的部分视图加入 values 中，并重塑为指定形状 tensorShape
      offset += numel;
      // 更新偏移量，准备处理下一个元素
    }

    return values;
  }
};

// 异步 CUDA 全局归约工作类，继承自异步全局归约工作类
class AsyncAllreduceCUDAWork : public AsyncAllreduceWork {
 public:
  // 构造函数，初始化异步 CUDA 全局归约工作对象
  AsyncAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : AsyncAllreduceWork(context, inputs, std::move(reduceOp), tag, seq) {
    // 初始化流和事件
    initializeStreamsEvents(inputs, streams, events);

    // 启动从 CUDA 张量到固定内存 CPU 张量的复制
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  // 运行函数，执行异步全局归约操作
  void run() override {
    // 同步复制操作
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // 在主机端张量上执行全局归约
    allreduce(tmp);

    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      // 将结果复制回 CUDA 张量，非阻塞操作
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  // 同步函数，同步复制回 CUDA 张量
  void synchronize() override {
    // 同步复制回 CUDA 张量
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp; // 临时存储张量的向量
  std::vector<c10::Stream> streams{}; // 流的向量
  std::vector<c10::Event> events{}; // 事件的向量
};

// 异步稀疏 CUDA 全局归约工作类，继承自异步稀疏全局归约工作类
class AsyncSparseAllreduceCUDAWork : public AsyncSparseAllreduceWork {
 public:
  // 构造函数，初始化异步稀疏 CUDA 全局归约工作对象
  AsyncSparseAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq)
      : AsyncSparseAllreduceWork(context, inputs, tag, seq) {
    // 初始化流和事件
    initializeStreamsEvents(inputs, streams, events);

    // 启动从 CUDA 张量到 CPU 张量的复制
    // 注意：必须异步执行稀疏张量的整合和复制到 CPU 内存，否则会阻塞调用者
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(
          inputs[i].coalesce().to(at::DeviceType::CPU, /*non_blocking=*/true));
    }
  }

  // 运行函数，执行异步全局归约操作
  void run() override {
    // 同步复制操作
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // 在主机端张量上执行全局归约
    auto output = allreduce(tmp);

    // 启动复制回 CUDA 张量
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      // 将结果复制回 CUDA 张量，非阻塞操作
      inputs[i].copy_(output, /*non_blocking=*/true);
      events[i].record(streams[i]);
    }
  }

  // 同步函数，同步复制回 CUDA 张量
  void synchronize() override {
    // 同步复制回 CUDA 张量
    // （未完待续，下文）
    // 遍历输入张量列表的索引范围
    for (const auto i : c10::irange(inputs.size())) {
      // 获取第 i 个输入张量的设备信息
      c10::Device device = inputs[i].device();
      // 根据设备类型创建虚拟保护实现对象，并获取对应的流
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  // 创建临时张量向量
  std::vector<at::Tensor> tmp{};
  // 创建流向量
  std::vector<c10::Stream> streams{};
  // 创建事件向量
  std::vector<c10::Event> events{};
}; // 结束类定义

} // namespace 结束命名空间

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  // 定义一个静态 Lambda 函数，用于处理无效参数异常
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce: " + msg);
  };

  // 断言输入张量向量非空
  assertNonEmpty(invalidArgument, inputs);
  // 断言输入张量的布局匹配
  assertLayoutMatch(invalidArgument, inputs);
  // 断言输入张量的类型和尺寸匹配
  assertTypeAndSizesMatch(invalidArgument, inputs);

  // 获取第一个张量的设备
  const auto& device = inputs[0].device();
  // 根据设备类型进行不同的处理
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // 如果用户给出了 CUDA 张量，则必须加载 CUDA
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      // 不支持的设备类型异常
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  // 获取第一个张量的布局
  const auto& layout = inputs[0].layout();
  // 如果布局为稀疏且选项中的 reduceOp 不是 ReduceOp::SUM，则抛出异常
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  // 创建一个异步工作指针
  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  // 根据设备类型和布局类型选择创建不同类型的异步工作对象
  if (device.type() == at::kCPU) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceWork>(
          std::move(context), inputs, opts.reduceOp, tag, seq_);
    } else if (layout == c10::kSparse) {
      work = c10::make_intrusive<AsyncSparseAllreduceWork>(
          std::move(context), inputs, tag, seq_);
    } else {
      // 不支持的布局异常
      invalidArgument("unsupported layout");
    }
  } else if (device.type() == at::kCUDA) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceCUDAWork>(
          std::move(context), inputs, opts.reduceOp, tag, seq_);
    } else if (layout == c10::kSparse) {
      work = c10::make_intrusive<AsyncSparseAllreduceCUDAWork>(
          std::move(context), inputs, tag, seq_);
    } else {
      // 不支持的布局异常
      invalidArgument("unsupported layout");
    }
  } else {
    // 无效的后端类型异常
    TORCH_CHECK(false, "Invalid backend");
  }

  // 将工作对象加入到队列中
  enqueue(work);
  // 返回工作对象指针
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce_sparse(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  // allreduce_sparse 调用默认的 allreduce 实现，用于稀疏张量的全局归约操作
  return allreduce(inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  // 定义一个静态 Lambda 函数，用于处理无效参数异常
  static auto invalidArgument = [](const std::string& msg) {
      // ProcessGroupGloo::allreduce_coalesced: + msg 用于错误消息的构建

static auto invalidArgument = [](const std::string& msg) {
  // 使用 TORCH_CHECK 来抛出 ProcessGroupGloo::allreduce_coalesced 的无效参数异常
  TORCH_CHECK(false, "ProcessGroupGloo::allreduce_coalesced: " + msg);
};
    // 使用 TORCH_CHECK 断言确保条件为 false，否则抛出错误信息
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce_coalesced: " + msg);
  };
  // 使用 assertNonEmpty 函数确保 tensors 容器不为空
  assertNonEmpty(invalidArgument, tensors);

  // tensors 将被展平并连接（coalesced）。这意味着输入的张量必须具有相同的设备、布局和类型。
  // 使用 assertLayoutMatch 函数确保 tensors 的布局匹配
  assertLayoutMatch(invalidArgument, tensors);
  // 如果不是所有张量都具有与 tensors[0] 相同的类型，则抛出错误信息
  if (!std::all_of(tensors.begin(), tensors.end(), [&](at::Tensor& t) {
        return t.options().type_equal(tensors[0].options());
      })) {
    invalidArgument("tensors must all have the same type");
  }
  // 如果不是所有张量都在相同的设备上，则抛出错误信息
  if (!std::all_of(tensors.begin(), tensors.end(), [&](at::Tensor& t) {
        return t.device() == tensors[0].device();
      })) {
    invalidArgument("tensors must all be on the same device");
  }

  // 获取第一个张量的设备和布局
  const c10::Device& device = tensors[0].device();
  const c10::Layout& layout = tensors[0].layout();

  // 在任何调用 nextTag() 增加 collectiveCounter_ 的操作之前，提前检测无效参数
  switch (device.type()) {
    // 如果设备类型是 CPU，则通过，否则抛出不支持的设备类型错误信息
    case c10::kCPU:
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  // 检查张量的布局是否为 Strided，否则抛出不支持的布局错误信息
  switch (layout) {
    case c10::kStrided:
      break;
    default:
      invalidArgument("unsupported layout");
  }

  // 创建一个 AsyncWork 对象指针，并获取下一个标签
  c10::intrusive_ptr<AsyncWork> work;
  const uint32_t tag = nextTag();
  // 获取与给定标签相关的 GLOO 环境上下文
  std::shared_ptr<gloo::Context> context = getContext(tag);
  // 递增序列号
  ++seq_;
  // 如果设备类型是 CPU 且布局是 Strided，则创建 AsyncAllreduceCoalescedWork 对象
  if (device.type() == c10::kCPU) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceCoalescedWork>(
          std::move(context), tensors, opts.reduceOp, tag, seq_);
    } else {
      // 如果布局不是 Strided，则抛出不支持的布局错误信息
      invalidArgument("unsupported layout");
    }
  } else {
    // 如果设备类型不是 CPU，则抛出无效后端错误信息
    TORCH_CHECK(false, "Invalid backend");
  }
  // 将工作任务加入队列
  enqueue(work);
  // 返回工作任务指针
  return work;
}

namespace {

class AsyncReduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncReduceWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {inputs},
            OpType::REDUCE,
            seq,
            "gloo:reduce",
            inputs),
        context(context),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        reduceOp(std::move(reduceOp)),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;  // 上下文对象的共享指针
  std::vector<at::Tensor> inputs{};  // 输入的张量向量
  const int rootRank;  // 根节点的排名
  const int rootTensor;  // 根节点的张量索引
  const ReduceOp reduceOp;  // 减少操作类型
  const uint32_t tag;  // 标记值

  void reduce(std::vector<at::Tensor>& tensors) {
    const auto& scalarType = tensors[0].scalar_type();  // 获取第一个张量的标量类型
    gloo::ReduceOptions opts(context);  // 创建Gloo的Reduce选项对象
    opts.setRoot(rootRank);  // 设置根节点的排名
    opts.setTag(tag);  // 设置标记值
    opts.setReduceFunction(getFunction(scalarType, reduceOp));  // 设置减少函数
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensors[0]);  // 生成所有类型的输出
    gloo::reduce(opts);  // 执行reduce操作
  }

  void run() override {
    reduce(inputs);  // 执行reduce操作
  }

 protected:
  template <typename T>
  void getFunction(gloo::ReduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);  // 获取指定类型和操作的函数
  }

  gloo::ReduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp& op) {
    gloo::ReduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);  // 生成所有数据类型的函数
    return fn;  // 返回函数
  }
};

class AsyncReduceCUDAWork : public AsyncReduceWork {
 public:
  AsyncReduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : AsyncReduceWork(
            context,
            inputs,
            rootRank,
            rootTensor,
            std::move(reduceOp),
            tag,
            seq) {
    initializeStreamsEvents(inputs, streams, events);  // 初始化流和事件

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());  // 为临时张量向量分配空间
    c10::OptionalStreamGuard guard;  // 可选流守卫
    for (const auto i : c10::irange(inputs.size())) {  // 遍历输入张量
      guard.reset_stream(streams[i]);  // 重置流守卫
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));  // 将CUDA张量复制到固定CPU张量并存储在tmp中
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {  // 遍历输入张量
      streams[i].synchronize();  // 同步流操作
    }

    // Run reduce on host side tensors.
    reduce(tmp);  // 在主机端执行reduce操作

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;  // 可选流守卫
    for (const auto i : c10::irange(inputs.size())) {  // 遍历输入张量
      guard.reset_stream(streams[i]);  // 重置流守卫
      inputs[i].copy_(tmp[i], /* non_blocking */ true);  // 将固定CPU张量复制回CUDA张量，非阻塞方式
      events[i].record(streams[i]);  // 记录事件到流中
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    // 对输入张量列表进行迭代，i 为索引
    for (const auto i : c10::irange(inputs.size())) {
      // 获取第 i 个输入张量所在的设备
      c10::Device device = inputs[i].device();
      // 创建虚拟保护器实现，并获取该设备对应的流
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  // 初始化临时张量列表
  std::vector<at::Tensor> tmp{};
  // 初始化流列表
  std::vector<c10::Stream> streams{};
  // 初始化事件列表
  std::vector<c10::Event> events{};
    // 定义一个类 AsyncAllgatherWork，继承自 ProcessGroupGloo::AsyncWork
    // 用于实现 Allgather 操作的异步工作
class AsyncAllgatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  // 构造函数，初始化 AsyncAllgatherWork 对象
  AsyncAllgatherWork(
      const std::shared_ptr<gloo::Context>& context,  // 共享的 GLOO 上下文指针
      std::vector<std::vector<at::Tensor>>& outputs,  // 输出张量的向量的向量
      std::vector<at::Tensor>& inputs,                // 输入张量的向量
      uint32_t tag,                                   // 操作标签
      uint64_t seq)                                   // 序列号
      : ProcessGroupGloo::AsyncWork(                  // 调用基类构造函数
            outputs,                                  // 将输出传递给基类
            OpType::ALLGATHER,                        // 操作类型为 Allgather
            seq,                                      // 传递序列号给基类
            "gloo:all_gather",                        // 操作名称
            inputs),                                  // 将输入传递给基类
        context(context),                             // 初始化上下文成员变量
        outputs(outputs),                             // 初始化输出成员变量
        inputs(inputs),                               // 初始化输入成员变量
        tag(tag) {}                                  // 初始化标签成员变量

  std::shared_ptr<gloo::Context> context;             // 共享的 GLOO 上下文指针
  std::vector<std::vector<at::Tensor>> outputs{};     // 输出张量的向量的向量
  std::vector<at::Tensor> inputs{};                   // 输入张量的向量
  const uint32_t tag;                                 // 常量标签

  // 执行 Allgather 操作
  void allgather(
      std::vector<std::vector<at::Tensor>>& outputs,   // 输出张量的向量的向量
      std::vector<at::Tensor>& inputs) {               // 输入张量的向量
    const auto& scalarType = inputs[0].scalar_type();  // 获取输入张量的标量类型
    gloo::AllgatherOptions opts(context);              // 创建 Allgather 的选项对象，使用上下文

    opts.setTag(tag);                                 // 设置选项对象的标签

    // 使用扁平化的输入张量
    at::Tensor flatInputTensor = flattenDenseTensors(inputs);  // 将输入张量扁平化
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);  // 生成所有类型的输入设置

    // 使用扁平化的输出张量
    // 第一个维度对应于索引，用于输出向量的索引，因此稍后将数据复制到实际的输出中会更容易
    at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);  // 创建与输出相同类型的新张量
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);  // 生成所有类型的输出设置
    gloo::allgather(opts);                          // 执行 Allgather 操作

    // 将数据展开到输出张量中
    // 对于输出组中的每个输出组件，使用引用方式迭代
    for (auto& outputgroup : outputs) {
      // 遍历当前输出组件中的每个元素，使用常量方式迭代
      for (const auto j : c10::irange(outputgroup.size())) {
        // 将 flatOutputTensor 中相应位置的张量数据复制到当前输出组件中的第 j 个元素
        outputgroup[j].copy_(flatOutputTensor[static_cast<int64_t>(j)]);
      }
    }
  }

  // 覆盖基类的 run 方法
  void run() override {
    // 执行 allgather 操作，将输入数据（inputs）收集到输出数据（outputs）中
    allgather(outputs, inputs);
  }
};

// CUDA 实现假设所有嵌套的输出张量向量中的张量都在同一个设备上。
class AsyncAllgatherCUDAWork : public AsyncAllgatherWork {
 public:
  AsyncAllgatherCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq)
      : AsyncAllgatherWork(context, outputs, inputs, tag, seq) {
    // 初始化输入流、事件
    initializeStreamsEvents(inputs, inputStreams, inputEvents);
    // 初始化输出流、事件
    initializeStreamsEvents(outputs, outputStreams, outputEvents);

    // 启动从 CUDA 张量到固定 CPU 张量的拷贝
    tmpInputs.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(inputStreams[i]);
      // 复制输入张量到固定内存中
      tmpInputs.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }

    // 为每个输出张量准备固定内存
    tmpOutputs.resize(outputs.size());
    for (const auto i : c10::irange(outputs.size())) {
      tmpOutputs[i].reserve(outputs[i].size());
      for (const auto j : c10::irange(outputs[i].size())) {
        tmpOutputs[i].push_back(pinnedLike(outputs[i][j]));
      }
    }
  }

  void run() override {
    // 同步拷贝操作
    for (const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();
    }

    for (const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();
    }

    // 在主机端张量上执行 allgather 操作
    allgather(tmpOutputs, tmpInputs);

    // 启动拷贝回 CUDA 张量
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      for (const auto j : c10::irange(outputs[i].size())) {
        // 非阻塞地将固定内存中的数据拷贝回 CUDA 张量
        outputs[i][j].copy_(tmpOutputs[i][j], /* non_blocking */ true);
      }
      // 记录输出事件
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // 同步拷贝回 CUDA 张量的操作
    for (const auto i : c10::irange(outputs.size())) {
      c10::Device device = outputs[i][0].device();
      // 阻塞直到输出事件完成
      outputEvents[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpInputs{};  // 临时存储的输入张量
  std::vector<c10::Stream> inputStreams{};  // 输入流
  std::vector<c10::Event> inputEvents{};  // 输入事件

  std::vector<std::vector<at::Tensor>> tmpOutputs{};  // 临时存储的输出张量
  std::vector<c10::Stream> outputStreams{};  // 输出流
  std::vector<c10::Event> outputEvents{};  // 输出事件
};

// 一个工作类，接受一个 lambda 表达式，在等待时调用该 lambda
// 用于向另一个工作添加继续操作，或将多个工作组合在一起时很有用。
class LambdaWork : public Work {
 public:
  LambdaWork(std::function<void(void)> fn) : fn_(std::move(fn)) {}

  bool wait(std::chrono::milliseconds /* unused */) override {
    // 调用 lambda 表达式
    fn_();
    return true;
  }

 private:
  std::function<void(void)> fn_;  // 存储的 lambda 表达式
};

} // namespace
// 对于给定的输出张量和输入张量执行归约散播操作，返回一个指向工作对象的智能指针
c10::intrusive_ptr<Work> ProcessGroupGloo::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  // 将输出张量放入输出张量列表
  std::vector<at::Tensor> outputTensors = {outputTensor};
  // 将输入张量放入输入张量列表
  std::vector<at::Tensor> inputTensors = {inputTensor};
  // 调用集合化的归约散播张量函数
  return reduce_scatter_tensor_coalesced(outputTensors, inputTensors, opts);
}

// 执行集合化的归约散播张量操作，返回一个指向工作对象的智能指针
c10::intrusive_ptr<Work> ProcessGroupGloo::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& opts) {
  // 检查输入张量列表和输出张量列表的长度是否相等
  if (outputTensors.size() != inputTensors.size()) {
    TORCH_CHECK(
        false, "requires input/output tensor lists to have the same length");
  }
  // 获取当前进程组的排名
  const auto rank = getRank();
  // 获取当前进程组的总大小
  const auto worldSize = getSize();
  // 创建一个缓冲区列表
  std::vector<at::Tensor> buffers;
  // 遍历输入张量列表
  for (const auto i : c10::irange(inputTensors.size())) {
    // 获取输入张量的形状
    auto inputShape = inputTensors[i].sizes().vec();
    // 获取输出张量的形状
    auto outputShape = outputTensors[i].sizes().vec();
    // 检查输出张量和输入张量的数据类型是否相同
    TORCH_CHECK_EQ(outputTensors[i].dtype(), inputTensors[i].dtype());
    // 检查输出张量的第一个维度是否符合规定
    TORCH_CHECK_EQ(outputShape[0] * worldSize, inputShape[0]);
    // 遍历除了第一个维度外的其他维度，检查形状是否匹配
    for (size_t i = 1; i < outputShape.size(); ++i) {
      TORCH_CHECK_EQ(outputShape[i], inputShape[i]);
    }
    // 将输入张量的克隆添加到缓冲区列表中
    buffers.push_back(inputTensors[i].clone());
  }
  // 创建一个工作对象列表
  std::vector<c10::intrusive_ptr<Work>> works;
  // 遍历缓冲区列表
  for (const auto i : c10::irange(buffers.size())) {
    // 创建输入张量列表，其中包含当前缓冲区
    std::vector<at::Tensor> inp = {buffers[i]};
    // 创建所有归约选项
    AllreduceOptions arOpts;
    arOpts.reduceOp = opts.reduceOp;
    // 调用全局归约操作并将结果添加到工作列表中
    works.push_back(allreduce(inp));
  }
  // 返回一个lambda工作对象，用于复制结果到输出张量
  return c10::make_intrusive<LambdaWork>(
      [rank, worldSize, buffers, outputTensors, works = std::move(works)]() {
        // 遍历输出张量列表
        for (const auto i : c10::irange(outputTensors.size())) {
          // 等待工作对象完成
          works[i]->wait();
          // 将缓冲区张量的数据拷贝到输出张量的特定部分
          outputTensors[i].copy_(buffers[i].chunk(worldSize)[rank]);
        }
      });
}

// 对给定的输出张量和输入张量执行全收集操作，返回一个指向工作对象的智能指针
c10::intrusive_ptr<Work> ProcessGroupGloo::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  // 将输出张量分块成与进程组大小相同的张量列表
  auto tensor_list = at::chunk(output_tensor, this->getSize(), 0);
  // 将张量列表放入输出列表
  std::vector<std::vector<at::Tensor>> outputs = {tensor_list};
  // 将输入张量放入输入列表
  std::vector<at::Tensor> inputs = {input_tensor};
  // 调用全收集操作并返回结果
  return this->allgather(outputs, inputs, opts);
}

// 执行全收集操作，将输入张量收集到输出张量列表中，返回一个指向工作对象的智能指针
// 注意：当前的CUDA实现假设嵌套的输出张量向量中的张量位于同一设备上。
c10::intrusive_ptr<Work> ProcessGroupGloo::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  // 匿名函数，用于抛出无效参数异常
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allgather: " + msg);
  };

  // 检查输入张量列表是否为空
  if (inputs.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  // 检查输入张量列表和输出张量列表的长度是否相等
  if (inputs.size() != outputs.size()) {
    invalidArgument(
        "requires input/output tensor lists to have the same length");
  }

  // 遍历输出张量列表
  for (const auto i : c10::irange(outputs.size())) {
    // 计算预期输出的长度，确保与实际输出长度相匹配
    const auto expected = inputs.size() * getSize();
    const auto actual = outputs[i].size();
    if (actual != expected) {
      // 抛出参数错误异常，显示期望长度和实际长度
      invalidArgument(
          "invalid output tensor list at index " + std::to_string(i) +
          " (expected length " + std::to_string(expected) + ", got " +
          std::to_string(actual) + ")");
    }
  }

  // 确保所有输入张量和输出张量具有相同的类型和大小
  assertDense(invalidArgument, inputs);

  // 获取第一个输入张量的选项和大小信息
  const auto& options = inputs[0].options();
  const auto& sizes = inputs[0].sizes();
  // 检查所有输入和输出张量的类型和大小是否匹配
  assertTypeAndSizesMatch(invalidArgument, inputs, options, sizes);
  for (const auto& output : outputs) {
    assertTypeAndSizesMatch(invalidArgument, output, options, sizes);
  }

  // 获取第一个输入张量的设备信息
  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      // 如果设备类型为 CPU，无需额外操作
      break;
    case at::kCUDA:
      // 如果用户给出了 CUDA 张量，则需要确保 CUDA 已加载
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      // 如果设备类型不受支持，抛出参数错误异常
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  // 创建异步全收集工作的指针
  c10::intrusive_ptr<AsyncAllgatherWork> work;
  // 获取下一个标签和上下文
  auto tag = nextTag();
  auto context = getContext(tag);
  // 递增序列号
  ++seq_;
  if (device.type() == at::kCPU) {
    // 如果设备类型为 CPU，创建标准的异步全收集工作
    work = c10::make_intrusive<AsyncAllgatherWork>(
        std::move(context), outputs, inputs, tag, seq_);
  } else if (device.type() == at::kCUDA) {
    // 如果设备类型为 CUDA，创建 CUDA 特定的异步全收集工作
    work = c10::make_intrusive<AsyncAllgatherCUDAWork>(
        std::move(context), outputs, inputs, tag, seq_);
  } else {
    // 如果设备类型不受支持，断言失败并显示错误信息
    TORCH_CHECK(false, "Invalid backend");
  }
  // 将工作任务加入队列
  enqueue(work);
  // 返回工作指针
  return work;
}

namespace {

class AsyncAllgatherCoalescedWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllgatherCoalescedWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& output_lists,
      std::vector<at::Tensor>& input_list,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            output_lists,
            OpType::ALLGATHER_COALESCED,
            seq,
            "gloo:all_gather",
            input_list),
        context(context),
        output_lists(output_lists),
        input_list(input_list),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;  // 上下文对象的共享指针
  std::vector<std::vector<at::Tensor>> output_lists{};  // 输出张量列表的二维向量
  std::vector<at::Tensor> input_list{};  // 输入张量列表
  const uint32_t tag;  // 操作的标签

  void allgather_coalesced() {
    assert(!output_lists.empty());  // 断言输出张量列表非空
    assert(!output_lists[0].empty());  // 断言第一个输出张量列表的元素非空
    assert(!input_list.empty());  // 断言输入张量列表非空

    const auto& scalarType = input_list[0].scalar_type();  // 获取输入张量的标量类型
    gloo::AllgatherOptions opts(context);  // 创建 Allgather 操作的选项对象
    opts.setTag(tag);  // 设置选项对象的标签

    // 使用单个扁平化的输入张量
    at::Tensor flatInputTensor = flattenDenseTensors(input_list);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // 计算所有请求张量需要分配的总元素数
    int64_t output_numel = 0;
    for (const auto& t : output_lists[0]) {
      output_numel += t.numel();
    }
    output_numel *= static_cast<int64_t>(output_lists.size());
    // 使用单个扁平化的输出张量
    at::Tensor flatOutputTensor =
        at::empty({output_numel}, output_lists[0][0].options());
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    int64_t current_element = 0;
    for (auto& output_list : output_lists) {
      for (auto& output_tensor : output_list) {
        output_tensor.copy_(
            flatOutputTensor.narrow(0, current_element, output_tensor.numel())
                .reshape(output_tensor.sizes()),
            true);
        current_element += output_tensor.numel();
      }
    }
  }

  void run() override {
    allgather_coalesced();  // 执行数据聚合操作
  }
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& output_lists,
    std::vector<at::Tensor>& input_list,
    const AllgatherOptions& /* unused */) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allgather_coalesced: " + msg);  // 检查参数有效性的静态 Lambda 函数
  };

  if (input_list.empty()) {
    invalidArgument("requires non-empty input tensor list");  // 输入张量列表为空时，抛出异常
  }

  if (output_lists.size() != static_cast<size_t>(getSize())) {
    invalidArgument("output lists should be equal to world size");  // 输出张量列表与全局大小不匹配时，抛出异常
  }

  assertSameDevice(invalidArgument, input_list);  // 断言输入张量列表设备相同

  // 期望每个输出列表中的第 i 个张量与输入列表中的第 i 个张量在类型和大小上匹配。
  for (const auto& output_list : output_lists) {
    // 检查输出列表的大小是否与输入列表相同，如果不同则抛出异常
    if (output_list.size() != input_list.size()) {
      invalidArgument(
          "invalid output size: (expected length " +
          std::to_string(input_list.size()) + ", got " +
          std::to_string(output_list.size()) + ")");
    }
    
    // 遍历输出列表，逐个检查每个输出张量的尺寸和类型是否符合预期
    for (const auto i : c10::irange(output_list.size())) {
      // 获取第 i 个输入张量的期望尺寸和实际尺寸
      const auto expected = input_list[i].sizes();
      const auto actual = output_list[i].sizes();
      
      // 如果实际尺寸与期望尺寸不符，则抛出异常
      if (actual != expected) {
        invalidArgument(
            "invalid size of output tensor at index " + std::to_string(i) +
            " (expected length " + toString(expected) + ", got " +
            toString(actual) + ")");
      }
      
      // 检查第 i 个输入张量和输出张量的类型是否相同，若不同则抛出异常
      if (!input_list[i].options().type_equal(output_list[i].options())) {
        invalidArgument(
            "invalid tensor type at index " + std::to_string(i) +
            " (expected " + input_list[i].toString() + ", got " +
            output_list[i].toString() + ")");
      }
    }
  }

  // 断言输入列表中张量的密度，确保都是密集张量
  assertDense(invalidArgument, input_list);

  // 获取下一个标签，用于标识当前操作的上下文
  auto tag = nextTag();
  
  // 根据标签获取上下文信息
  auto context = getContext(tag);
  
  // 增加序列号，用于跟踪操作的顺序
  ++seq_;
  
  // 创建一个异步的集合通信操作任务，用于在异步环境下进行数据的聚合和传输
  auto work = c10::make_intrusive<AsyncAllgatherCoalescedWork>(
      std::move(context), output_lists, input_list, tag, seq_);
  
  // 将工作任务加入任务队列中等待执行
  enqueue(work);
  
  // 返回创建的工作任务对象
  return work;
// 定义 ProcessGroupGloo 类的方法，用于执行 allgather 操作，并将结果收集到协调的张量中
c10::intrusive_ptr<Work> ProcessGroupGloo::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,  // 输出张量列表
    std::vector<at::Tensor>& inputs,   // 输入张量列表
    const AllgatherOptions& opts) {    // AllgatherOptions 参数

  TORCH_CHECK_EQ(outputs.size(), inputs.size());  // 断言输出张量数量与输入张量数量相同

  std::vector<std::vector<at::Tensor>> output_lists(getSize());  // 创建二维向量用于存储输出张量的列表
  for (auto& output : outputs) {  // 遍历输出张量列表
    auto chunks = output.chunk(getSize());  // 将每个输出张量分割成多个块
    for (const auto i : c10::irange(output_lists.size())) {  // 遍历输出列表的每个元素
      output_lists[i].push_back(std::move(chunks[i]));  // 将分割后的块移动到相应的输出列表中
    }
  }

  return allgather_coalesced(output_lists, inputs, opts);  // 调用 allgather_coalesced 方法执行所有收集操作
}

// 匿名命名空间中定义 AsyncGatherWork 类，继承自 ProcessGroupGloo::AsyncWork
class AsyncGatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncGatherWork(
      const std::shared_ptr<gloo::Context>& context,  // 共享指针指向 GLOO 上下文
      std::vector<std::vector<at::Tensor>>& outputs,  // 输出张量的二维向量
      std::vector<at::Tensor>& inputs,  // 输入张量列表
      int root,  // 根节点索引
      uint32_t tag,  // 标签
      uint64_t seq)  // 序列号
      : ProcessGroupGloo::AsyncWork(
            outputs,  // 输出张量列表
            OpType::GATHER,  // 操作类型为 Gather
            seq,  // 序列号
            "gloo:gather",  // 操作名称
            inputs),  // 输入张量列表
        context(context),  // 初始化 GLOO 上下文
        outputs(outputs),  // 初始化输出张量列表
        inputs(inputs),  // 初始化输入张量列表
        root(root),  // 初始化根节点索引
        tag(tag) {}  // 初始化标签

  std::shared_ptr<gloo::Context> context;  // 共享指针指向 GLOO 上下文
  std::vector<std::vector<at::Tensor>> outputs{};  // 输出张量的二维向量
  std::vector<at::Tensor> inputs{};  // 输入张量列表
  const int root;  // 根节点索引
  const uint32_t tag;  // 标签

  // 执行 gather 操作，将结果收集到指定的输出张量列表中
  void gather(
      std::vector<std::vector<at::Tensor>>& outputs,  // 输出张量的二维向量
      std::vector<at::Tensor>& inputs) {  // 输入张量列表
    const auto scalarType = inputs[0].scalar_type();  // 获取输入张量的标量类型
    gloo::GatherOptions opts(context);  // 创建 GLOO GatherOptions 对象，使用给定的上下文

    opts.setRoot(root);  // 设置 gather 操作的根节点
    opts.setTag(tag);  // 设置 gather 操作的标签

    // 在根节点进程上设置单个临时张量
    // 该张量稍后会分散到不同的输出张量中
    at::Tensor flatOutputTensor;
    if (context->rank == root) {  // 如果当前进程是根节点
      flatOutputTensor = newLikeFlat(outputs[0]);  // 创建一个与 outputs[0] 类似的张量
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);  // 根据输入标量类型设置输出张量
    }

    // 在所有进程上设置单个输入张量
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);  // 根据输入标量类型设置输入张量
    gloo::gather(opts);  // 执行 GLOO gather 操作

    // 在根节点进程上将结果展开到输出张量中
    if (context->rank == root) {  // 如果当前进程是根节点
      for (const auto i : c10::irange(outputs[0].size())) {  // 遍历 outputs[0] 的大小
        outputs[0][i].copy_(flatOutputTensor[static_cast<int64_t>(i)]);  // 将 flatOutputTensor 的值复制到 outputs[0][i]
      }
    }
  }

  void run() override {  // 覆盖父类的虚函数 run
    gather(outputs, inputs);  // 调用 gather 方法执行收集操作
  }
};

// 注意：当前的 CUDA 实现假设：
//     - inputs.size() 为 1
//     - outputs.size() 为 1
//     - 嵌套输出张量的大小为全局大小，即 outputs[0].size，是全局大小
// 匿名命名空间中定义 AsyncGatherCUDAWork 类，继承自 AsyncGatherWork
class AsyncGatherCUDAWork : public AsyncGatherWork {
 public:
  AsyncGatherCUDAWork(
      const std::shared_ptr<gloo::Context>& context,  // 共享指针指向 GLOO 上下文
      std::vector<std::vector<at::Tensor>>& outputs,  // 输出张量的二维向量
      std::vector<at::Tensor>& inputs,  // 输入张量列表
      int root,  // 根节点索引
      uint32_t tag,  // 标签
      uint64_t seq)  // 序列号
      : AsyncGatherWork(context, outputs, inputs, root, tag, seq) {  // 调用基类构造函数
    initializeStreamsEvents(inputs, inputStreams, inputEvents);  // 初始化流和事件
    initializeStreamsEvents(outputs, outputStreams, outputEvents);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    // 初始化输入的临时向量，并设置流的保护，以确保操作在正确的流上进行
    tmpInputs.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(inputStreams[i]);
      // 在固定内存的CPU张量上进行复制操作，以将数据从CUDA张量复制到CPU张量
      tmpInputs.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }

    tmpOutputs.resize(outputs.size());
    for (const auto i : c10::irange(outputs.size())) {
      tmpOutputs[i].reserve(outputs[i].size());
      for (const auto j : c10::irange(outputs[i].size())) {
        // 初始化输出的临时向量，确保足够的空间来存储数据
        tmpOutputs[i].push_back(pinnedLike(outputs[i][j]));
      }
    }
  }

  void run() override {
    // Synchronize with copy operations.
    // 与复制操作同步，确保所有输入流中的操作完成
    for (const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();
    }

    for (const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();
    }

    // Run gather on host side tensors.
    // 在主机端张量上执行收集操作，将数据聚集起来
    gather(tmpOutputs, tmpInputs);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      for (const auto j : c10::irange(outputs[i].size())) {
        // 将临时输出数据复制回CUDA张量，使用非阻塞方式进行复制
        outputs[i][j].copy_(tmpOutputs[i][j], /* non_blocking */ true);
      }
      // 记录输出事件，表示输出操作完成
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    // 与复制回CUDA张量的操作同步，确保所有输出事件在相应流上被阻塞
    for (const auto i : c10::irange(outputs.size())) {
      c10::Device device = outputs[i][0].device();
      outputEvents[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpInputs{};
  std::vector<c10::Stream> inputStreams{};
  std::vector<c10::Event> inputEvents{};

  std::vector<std::vector<at::Tensor>> tmpOutputs{};
  std::vector<c10::Stream> outputStreams{};
  std::vector<c10::Event> outputEvents{};
};

} // namespace

// 在 ProcessGroupGloo 类中实现 gather 方法，用于执行数据的收集操作
c10::intrusive_ptr<Work> ProcessGroupGloo::gather(
    // 输出参数，保存收集后的数据
    std::vector<std::vector<at::Tensor>>& outputs,
    // 输入参数，包含本地数据的张量
    std::vector<at::Tensor>& inputs,
    // 收集操作的选项
    const GatherOptions& opts) {
  // 匿名函数，用于抛出无效参数异常
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::gather: " + msg);
  };

  // 校验根进程的排名是否有效
  assertRootRank(invalidArgument, opts.rootRank, size_);
  // 校验输入张量是否只包含一个元素
  assertSingleElementInput(invalidArgument, inputs);
  // 校验输入张量是否为稠密张量
  assertDense(invalidArgument, inputs);

  // 如果当前进程是根进程
  if (getRank() == opts.rootRank) {
    // 校验输出列表长度是否为1
    if (outputs.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    // 校验输出列表中第一个元素的长度是否与进程组大小相同
    } else if (outputs[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputs[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    // 获取输入张量的选项和大小
    const auto& options = inputs[0].options();
    const auto& sizes = inputs[0].sizes();
    // 校验输出列表的类型和大小是否与输入张量匹配
    assertTypeAndSizesMatch(invalidArgument, outputs[0], options, sizes);
  } else {
    // 非根进程要求输出列表为空
    if (!outputs.empty()) {
      invalidArgument("requires empty output on non-root");
    }
  }

  // 获取输入张量的设备类型
  const auto& device = inputs[0].device();
  // 根据设备类型执行不同的处理逻辑
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // 如果输入张量是 CUDA 张量，则必须加载 CUDA
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      // 不支持的设备类型
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  // 创建异步收集工作的指针
  c10::intrusive_ptr<AsyncGatherWork> work;
  // 获取下一个标签
  auto tag = nextTag();
  // 获取上下文
  auto context = getContext(tag);
  // 递增序列号
  ++seq_;
  // 根据设备类型创建相应的异步收集工作
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncGatherWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncGatherCUDAWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else {
    // 不支持的后端类型
    TORCH_CHECK(false, "Invalid backend");
  }
  // 将工作任务加入队列
  enqueue(work);
  // 返回工作任务指针
  return work;
}

// 匿名命名空间结束
namespace {
class AsyncScatterWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncScatterWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(  // 调用父类构造函数，初始化异步工作
            {outputs},               // 将输出张量传递给父类的构造函数
            OpType::SCATTER,         // 设置操作类型为SCATTER
            seq,                     // 序列号
            "gloo:scatter",          // 标识该操作的名称
            !inputs.empty() ? std::optional<std::vector<at::Tensor>>(inputs[0])  // 根据输入是否为空设置可选的输入张量
                            : c10::nullopt),
        context(context),           // 初始化上下文
        outputs(outputs),           // 初始化输出张量
        inputs(inputs),             // 初始化输入张量
        root(root),                 // 初始化根节点
        tag(tag) {}                 // 初始化标签

  std::shared_ptr<gloo::Context> context;  // Gloo通信的上下文
  std::vector<at::Tensor> outputs{};        // 输出张量的向量
  std::vector<std::vector<at::Tensor>> inputs{};  // 输入张量的向量的向量
  const int root;                          // 根节点索引
  const uint32_t tag;                      // 操作标签

  void scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs) {
    const auto scalarType = outputs[0].scalar_type();  // 获取输出张量的数据类型
    gloo::ScatterOptions opts(context);               // 创建Gloo分散选项对象
    opts.setRoot(root);                               // 设置根节点索引
    opts.setTag(tag);                                 // 设置操作标签

    // 在根进程上设置输入张量列表
    if (context->rank == root) {
      GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);  // 生成所有数据类型的输入设置操作
    }

    // 在所有进程上设置单个输出张量
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);   // 生成所有数据类型的输出设置操作
    gloo::scatter(opts);                          // 执行分散操作
  }

  void run() override {
    scatter(outputs, inputs);                    // 运行分散操作
  }
};

class AsyncScatterCUDAWork : public AsyncScatterWork {
 public:
  AsyncScatterCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq)
      : AsyncScatterWork(context, outputs, inputs, root, tag, seq) {
    initializeStreamsEvents(inputs, inputStreams, inputEvents);   // 初始化输入流和事件
    initializeStreamsEvents(outputs, outputStreams, outputEvents); // 初始化输出流和事件

    // 启动从CUDA张量到固定CPU张量的复制
    tmpInputs.resize(inputs.size());                             // 调整临时输入向量的大小
    c10::OptionalStreamGuard guard;                               // 可选流守卫
    for (const auto i : c10::irange(inputs.size())) {             // 对每个输入张量向量进行循环
      guard.reset_stream(inputStreams[i]);                        // 重置输入流守卫
      tmpInputs[i].reserve(inputs[i].size());                     // 预留临时输入的空间大小
      for (const auto j : c10::irange(inputs[i].size())) {        // 对每个张量进行循环
        tmpInputs[i].push_back(                                   // 将张量拷贝到固定的CPU张量
            pinnedLike(inputs[i][j]).copy_(inputs[i][j], true));  
      }
    }

    tmpOutputs.reserve(outputs.size());                           // 预留输出的空间大小
    for (auto& output : outputs) {                                // 对每个输出进行循环
      tmpOutputs.push_back(pinnedLike(output));                   // 将每个输出与固定CPU张量进行拷贝
    }
  }

  void run() override {
    // 与复制操作同步
    for (const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();                             // 同步输入流
    }
    for (const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();                            // 同步输出流
    }

    // 在主机端张量上运行分散
    scatter(tmpOutputs, tmpInputs);                               // 运行分散操作

    // 启动从固定CPU张量到CUDA张量的复制
    c10::OptionalStreamGuard guard;
    // 对于每个输出张量，复位相关的流以确保异步复制
    for (const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      // 将临时输出张量的数据复制到输出张量中，使用非阻塞方式
      outputs[i].copy_(tmpOutputs[i], /* non_blocking */ true);
      // 记录输出事件，表示输出流已准备好
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // 与复制回CUDA张量的同步操作
    for (const auto i : c10::irange(outputs.size())) {
      // 获取当前输出张量的设备信息
      c10::Device device = outputs[i].device();
      // 阻塞当前输出事件，直到相关的CUDA流完成所有操作
      outputEvents[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpOutputs{};  // 存储临时输出张量的向量
  std::vector<c10::Stream> outputStreams{};  // 存储输出流的向量
  std::vector<c10::Event> outputEvents{};  // 存储输出事件的向量

  std::vector<std::vector<at::Tensor>> tmpInputs{};  // 存储临时输入张量的向量的向量
  std::vector<c10::Stream> inputStreams{};  // 存储输入流的向量
  std::vector<c10::Event> inputEvents{};  // 存储输入事件的向量
};

} // namespace

// 函数定义：执行scatter操作，将输入数据分发到各个进程中
c10::intrusive_ptr<Work> ProcessGroupGloo::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  
  // 匿名函数，用于报告无效参数错误，并抛出异常
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::scatter: " + msg);
  };

  // 断言：根节点的排名必须有效
  assertRootRank(invalidArgument, opts.rootRank, size_);
  
  // 断言：输出列表必须只有一个元素
  assertSingleElementOutput(invalidArgument, outputs);
  
  // 断言：输出必须是稠密的
  assertDense(invalidArgument, outputs);

  // 如果当前进程是根进程
  if (getRank() == opts.rootRank) {
    // 检查输入列表是否只有一个元素
    if (inputs.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors";
      invalidArgument(ss.str());
    } else if (inputs[0].size() != static_cast<size_t>(getSize())) {
      // 检查输入列表的大小是否与进程组的大小相匹配
      std::stringstream ss;
      ss << "Incorrect input list size " << inputs[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }
    // 检查输出的类型和大小是否与输入匹配
    const auto& options = outputs[0].options();
    const auto& sizes = outputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputs[0], options, sizes);
  } else {
    // 非根进程不应该有输入数据
    if (!inputs.empty()) {
      invalidArgument("requires empty input on non-root");
    }
  }

  // 获取输出数据的设备类型
  const auto& device = outputs[0].device();
  
  // 根据设备类型执行不同的操作
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // 如果用户给了我们一个CUDA tensor，则必须已加载CUDA
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      // 不支持的设备类型错误
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  // 创建异步scatter操作的工作对象
  c10::intrusive_ptr<AsyncScatterWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  
  // 根据设备类型选择不同的工作对象
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncScatterWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncScatterCUDAWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else {
    // 不支持的设备类型错误
    TORCH_CHECK(false, "Invalid backend");
  }
  
  // 将工作对象加入队列等待执行
  enqueue(work);
  
  // 返回工作对象
  return work;
}

// 函数定义：执行reduce_scatter操作，但ProcessGroupGloo不支持此操作
c10::intrusive_ptr<Work> ProcessGroupGloo::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  // 抛出异常，因为ProcessGroupGloo不支持reduce_scatter操作
  TORCH_CHECK(false, "ProcessGroupGloo does not support reduce_scatter");
}

// 匿名命名空间，用于限定作用域，避免全局变量污染
namespace {
class AsyncAlltoallWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAlltoallWork(
      const std::shared_ptr<gloo::Context>& context,
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(  // 调用基类构造函数，初始化异步工作
            {{outputTensor}},  // 将输出张量作为异步工作的输出
            OpType::ALLTOALL,  // 设置操作类型为 ALLTOALL
            seq,  // 设置序列号
            "gloo:all_to_all",  // 设置名称
            std::optional<std::vector<at::Tensor>>({inputTensor})),  // 可选输入张量作为参数

        context(context),  // 初始化上下文对象
        outputTensor(outputTensor),  // 初始化输出张量
        inputTensor(inputTensor),  // 初始化输入张量
        outputCounts(std::move(outputCounts)),  // 移动赋值输出计数向量
        inputCounts(std::move(inputCounts)),  // 移动赋值输入计数向量
        tag(tag) {}  // 初始化标签

  std::shared_ptr<gloo::Context> context;  // 上下文对象的共享指针
  at::Tensor outputTensor;  // 输出张量
  at::Tensor inputTensor;  // 输入张量
  std::vector<int64_t> outputCounts{};  // 输出计数向量
  std::vector<int64_t> inputCounts{};  // 输入计数向量
  const uint32_t tag;  // 标签

  void alltoall(at::Tensor& outputTensor, at::Tensor& inputTensor) {
    const auto scalarType = outputTensor.scalar_type();  // 获取输出张量的标量类型
    if (outputCounts.empty() && inputCounts.empty()) {
      // 使用 Gloo 进行 alltoall 操作
      gloo::AlltoallOptions opts(context);  // 创建 alltoall 选项对象，使用给定的上下文
      opts.setTag(tag);  // 设置标签
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor);  // 根据标量类型生成 setInput 操作
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor);  // 根据标量类型生成 setOutput 操作
      gloo::alltoall(opts);  // 执行 alltoall 操作
    } else {
      // 使用 Gloo 进行 alltoallv 操作
      c10d::checkSplitSizes(inputCounts, inputTensor, context->size);  // 检查输入张量的切分大小是否符合要求
      c10d::checkSplitSizes(outputCounts, outputTensor, context->size);  // 检查输出张量的切分大小是否符合要求
      std::vector<int64_t> sendCounts(context->size);  // 发送计数向量
      std::vector<int64_t> recvCounts(context->size);  // 接收计数向量
      std::vector<int64_t> sendOffsets(context->size);  // 发送偏移向量
      std::vector<int64_t> recvOffsets(context->size);  // 接收偏移向量
      c10d::computeLengthsAndOffsets(
          inputCounts, inputTensor, &sendCounts, &sendOffsets);  // 计算输入张量的长度和偏移量
      c10d::computeLengthsAndOffsets(
          outputCounts, outputTensor, &recvCounts, &recvOffsets);  // 计算输出张量的长度和偏移量
      gloo::AlltoallvOptions opts(context);  // 创建 alltoallv 选项对象，使用给定的上下文
      opts.setTag(tag);  // 设置标签
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor, sendCounts);  // 根据标量类型生成 setInput 操作
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor, recvCounts);  // 根据标量类型生成 setOutput 操作
      gloo::alltoallv(opts);  // 执行 alltoallv 操作
    }
  }

  void run() override {
    alltoall(outputTensor, inputTensor);  // 运行 alltoall 或 alltoallv 操作
  }
};

class AsyncAlltoallCUDAWork : public AsyncAlltoallWork {
 public:
  AsyncAlltoallCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      uint32_t tag,
      uint64_t seq)
      : AsyncAlltoallWork(  // 调用基类构造函数，初始化异步 CUDA 工作
            context,
            outputTensor,
            inputTensor,
            outputCounts,
            inputCounts,
            tag,
            seq) {
    initializeStreamsEvents({inputTensor}, inputStreams, inputEvents);  // 初始化输入流和事件
    initializeStreamsEvents({outputTensor}, outputStreams, outputEvents);  // 初始化输出流和事件
    // 开始从 CUDA 张量复制到固定的 CPU 张量。
    c10::OptionalStreamGuard guard;
    guard.reset_stream(inputStreams.front());  // 重设流为输入流的第一个
    cpuInput = pinnedLike(inputTensor).copy_(inputTensor, true);  // 在固定内存中创建与输入张量相同的张量，并复制数据到其中

    guard.reset_stream(outputStreams.front());  // 重设流为输出流的第一个
    cpuOutput = pinnedLike(outputTensor);  // 在固定内存中创建与输出张量相同的张量
  }

  void run() override {
    // 与复制操作同步。
    inputStreams.front().synchronize();  // 同步输入流的第一个
    outputStreams.front().synchronize();  // 同步输出流的第一个

    // 在主机端张量上执行 alltoall 操作。
    alltoall(cpuOutput, cpuInput);  // 执行 alltoall 操作，将固定内存中的输出张量复制到输入张量中

    // 开始复制回 CUDA 张量。
    c10::OptionalStreamGuard guard;
    guard.reset_stream(outputStreams.front());  // 重设流为输出流的第一个
    outputTensor.copy_(cpuOutput, /* non_blocking */ true);  // 将固定内存中的 cpuOutput 数据复制回 outputTensor，并使用非阻塞方式

    outputEvents.front().record(outputStreams.front());  // 记录输出事件在输出流的第一个上
  }

  void synchronize() override {
    // 与复制回 CUDA 张量同步。
    c10::Device device = outputTensor.device();
    outputEvents.front().block(
        c10::impl::VirtualGuardImpl(device.type()).getStream(device));  // 阻塞等待输出事件在与 outputTensor 设备相关的流上完成
  }

  at::Tensor cpuOutput;  // CPU 端输出张量
  std::vector<c10::Stream> outputStreams{};  // 输出流的向量
  std::vector<c10::Event> outputEvents{};  // 输出事件的向量

  at::Tensor cpuInput;  // CPU 端输入张量
  std::vector<c10::Stream> inputStreams{};  // 输入流的向量
  std::vector<c10::Event> inputEvents{};  // 输入事件的向量
};

} // namespace

// 实现 ProcessGroupGloo 类的 alltoall_base 方法，执行所有到所有的通信操作
c10::intrusive_ptr<Work> ProcessGroupGloo::alltoall_base(
    at::Tensor& outputTensor,  // 输出张量
    at::Tensor& inputTensor,   // 输入张量
    std::vector<int64_t>& outputCounts,  // 输出计数
    std::vector<int64_t>& inputCounts,   // 输入计数
    const AllToAllOptions& /* unused */) {  // 不使用的选项参数

  // lambda 函数，用于生成无效参数异常
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::alltoall_base: " + msg);
  };

  // 检查输出张量和输入张量是否在相同设备上
  TORCH_CHECK(
      outputTensor.device() == inputTensor.device(),
      "output tensor and input tensor must be on the same type of device");

  // 调用 assertDense 函数，确保输出张量和输入张量是密集的
  assertDense(invalidArgument, {outputTensor});
  assertDense(invalidArgument, {inputTensor});

  // 获取输出张量的设备类型
  const auto& device = outputTensor.device();

  // 创建 AsyncAlltoallWork 或 AsyncAlltoallCUDAWork 对象，具体根据设备类型选择
  c10::intrusive_ptr<AsyncAlltoallWork> work;
  auto tag = nextTag();  // 获取下一个标签
  auto context = getContext(tag);  // 获取上下文信息
  ++seq_;  // 序列号自增

  // 根据设备类型选择不同的工作类型
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncAlltoallWork>(
        std::move(context),
        outputTensor,
        inputTensor,
        outputCounts,
        inputCounts,
        tag,
        seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncAlltoallCUDAWork>(
        std::move(context),
        outputTensor,
        inputTensor,
        outputCounts,
        inputCounts,
        tag,
        seq_);
  } else {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  // 将工作对象加入到队列中
  enqueue(work);

  // 返回工作对象
  return work;
}

// 静态函数，检查是否仅有一个张量，并返回该张量的引用
static at::Tensor& checkSingleTensor(std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(false, "ProcessGroupGloo::send takes a single tensor");
  }
  auto& tensor = tensors[0];
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
  return tensor;
}

// 静态函数，检查标签是否为非负数，返回无符号整数标签
static uint32_t checkTag(int32_t tag) {
  TORCH_CHECK(tag >= 0, "Tag must be nonnegative");
  return (uint32_t)tag;
}

// 实现 ProcessGroupGloo 类的 send 方法，发送张量数据到指定进程
c10::intrusive_ptr<Work> ProcessGroupGloo::send(
    std::vector<at::Tensor>& tensors,  // 发送的张量数组
    int dstRank,                       // 目标进程的排名
    int tag) {                         // 标签

  auto& tensor = checkSingleTensor(tensors);  // 检查并获取单个张量的引用
  auto utag = checkTag(tag);  // 检查并获取有效的标签

  auto ptr = tensor.const_data_ptr();  // 获取张量数据指针
  auto size = tensor.numel() * tensor.element_size();  // 计算张量数据大小

  // 构造未绑定的缓冲区
  auto context = getContext(tag);  // 获取上下文信息
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto buf = context->createUnboundBuffer(const_cast<void*>(ptr), size);

  buf->send(dstRank, utag);  // 发送数据到目标进程
  ++seq_;  // 序列号自增

  // 创建并返回发送工作对象，工作对象捕获张量以防止其被释放，并且捕获未绑定的缓冲区以在发送完成时同步
  return c10::make_intrusive<SendWork>(tensor, std::move(buf), seq_);
}

// 实现 ProcessGroupGloo 类的 recv 方法，接收来自指定进程的张量数据
c10::intrusive_ptr<Work> ProcessGroupGloo::recv(
    std::vector<at::Tensor>& tensors,  // 接收的张量数组
    int srcRank,                       // 源进程的排名
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  // 检查并获取唯一的张量引用

  auto utag = checkTag(tag);
  // 检查并获取有效的标签

  auto ptr = tensor.mutable_data_ptr();
  // 获取张量的可变数据指针

  auto size = tensor.numel() * tensor.element_size();
  // 计算张量的总元素数乘以每个元素的大小，得到数据大小

  // 构造未绑定的缓冲区。
  auto context = getContext(tag);
  // 根据标签获取上下文对象

  auto buf = context->createUnboundBuffer(ptr, size);
  // 使用上下文对象创建未绑定的缓冲区，传入数据指针和大小

  buf->recv(srcRank, utag);
  // 在缓冲区上执行接收操作，从srcRank接收数据，使用utag作为标签

  ++seq_;
  // 序列号自增

  // 任务捕获了张量，防止其被释放，同时捕获了未绑定的缓冲区以在接收完成时同步。
  return c10::make_intrusive<RecvWork>(
      tensor, std::move(buf), OpType::RECV, seq_, "gloo:recv");
  // 创建并返回一个接收任务对象，捕获了张量、移动了未绑定的缓冲区，
  // 设置操作类型为RECV，使用当前序列号和指定的任务类型标识"gloo:recv"
}

// 定义 recvAnysource 函数，接收任意来源的数据并处理
c10::intrusive_ptr<Work> ProcessGroupGloo::recvAnysource(
    std::vector<at::Tensor>& tensors, // 接收包含张量的向量
    int tag) { // 接收标签参数
  auto& tensor = checkSingleTensor(tensors); // 检查并获取唯一的张量引用
  auto utag = checkTag(tag); // 检查并获取有效的标签值
  auto ptr = tensor.mutable_data_ptr(); // 获取张量的可变数据指针
  auto size = tensor.numel() * tensor.element_size(); // 计算张量的总字节数

  // 构建未绑定的缓冲区
  auto context = getContext(tag); // 获取上下文对象
  auto buf = context->createUnboundBuffer(ptr, size); // 使用数据指针和大小创建未绑定的缓冲区

  // 构建可以从中接收数据的进程排名列表。在这些绑定中，不区分排名，可以从组中的任何其他进程接收。
  std::vector<int> srcRanks;
  srcRanks.resize(size_); // 调整大小以容纳与组大小相同数量的元素
  for (const auto i : c10::irange(size_)) { // 遍历组大小范围
    srcRanks.push_back(i); // 添加每个索引作为可接收的进程排名
  }

  buf->recv(srcRanks, utag); // 接收来自指定进程排名的数据
  ++seq_; // 增加序列号

  // Work 对象捕获张量以防止其被释放，捕获未绑定的缓冲区以在接收完成时进行同步。
  return c10::make_intrusive<RecvWork>(
      tensor,
      std::move(buf),
      OpType::RECVANYSOURCE,
      seq_,
      "gloo:recvAnySource"); // 创建并返回接收工作对象
}

namespace {

// 异步屏障工作类，继承自 ProcessGroupGloo::AsyncWork
class AsyncBarrierWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncBarrierWork(
      const std::shared_ptr<gloo::Context>& context, // 上下文对象的共享指针
      std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork, // 先前工作的弱指针向量
      uint32_t tag, // 标签值
      uint64_t seq) // 序列号
      : ProcessGroupGloo::AsyncWork(
            {}, // 空显式标识符列表
            OpType::BARRIER, // 操作类型为屏障
            seq, // 操作的序列号
            "gloo:barrier", // 操作的标识符
            c10::nullopt), // 空的可选参数
        context(context), // 初始化上下文对象
        priorWork(std::move(priorWork)), // 初始化先前工作
        tag(tag) {} // 初始化标签值

  std::shared_ptr<gloo::Context> context; // 上下文对象的共享指针
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork{}; // 先前工作的弱指针向量
  const uint32_t tag; // 标签值

  void run() override { // 实现运行方法
    // 等待先前的工作完成
    for (auto& weakWork : priorWork) { // 遍历先前工作的弱指针向量
      auto work = weakWork.lock(); // 锁定弱指针以获取有效的工作指针
      if (work) { // 如果工作指针有效
        work->wait(); // 等待工作完成
      }
    }

    gloo::BarrierOptions opts(context); // 使用上下文对象创建 BarrierOptions
    opts.setTag(tag); // 设置选项的标签值
    gloo::barrier(opts); // 执行屏障操作
  }
};

} // namespace

// 实现屏障操作函数，使用给定的选项执行屏障操作
c10::intrusive_ptr<Work> ProcessGroupGloo::barrier(const BarrierOptions& opts) {
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork; // 先前工作的弱指针向量

  // 快照所有正在进行和待处理的工作为弱指针
  // 在执行屏障时，需要确保所有先前的工作都已完成才能完成屏障本身。
  {
    std::unique_lock<std::mutex> lock(workMutex_); // 锁定工作互斥量
    priorWork.insert( // 插入元素到先前工作向量
        priorWork.end(), workInProgress_.begin(), workInProgress_.end()); // 插入正在进行的工作
    priorWork.insert(priorWork.end(), workQueue_.begin(), workQueue_.end()); // 插入待处理的工作
  }

  auto tag = nextTag(); // 获取下一个标签值
  auto context = getContext(tag); // 获取上下文对象
  ++seq_; // 增加序列号
  auto work = c10::make_intrusive<AsyncBarrierWork>( // 创建异步屏障工作对象
      std::move(context), std::move(priorWork), tag, seq_); // 使用上下文对象、先前工作、标签和序列号初始化
  enqueue(work); // 将工作对象加入队列
  return work; // 返回工作对象
}

void ProcessGroupGloo::monitoredBarrier(
    const BarrierOptions& opts,
    bool waitAllRanks) {
  // 记录 API 使用情况，仅记录一次
  C10_LOG_API_USAGE_ONCE("torch.distributed.monitored_barrier");
  // 如果未指定超时时间，则使用默认超时时间
  auto monitoredBarrierTimeout =
      (opts.timeout == kUnsetTimeout) ? this->options_->timeout : opts.timeout;
  // 获取当前进程的排名
  auto rank = this->getRank();
  // 获取下一个标记
  auto t1 = nextTag();
  // 获取下一个标记
  auto t2 = nextTag();
  // 创建包含当前进程排名的张量的向量，用于通信
  std::vector<at::Tensor> commTensor = {at::tensor({rank})};
  // 只在排名非 0 的进程上强制执行超时，以防其他进程超时而导致整个作业失败而无法报告哪个进程超时。
  if (rank != 0) {
    // 发送当前进程排名给排名为 0 的进程
    auto sendWork = send(commTensor, 0, static_cast<int>(t1));
    // 从排名为 0 的进程接收数据
    auto recvWork = recv(commTensor, 0, static_cast<int>(t2));
    try {
      // 等待发送操作完成
      sendWork->wait();
      // 等待接收操作完成
      recvWork->wait();
    } catch (const std::exception& e) {
      // 如果在等待期间出现异常，记录错误信息并抛出异常
      const std::string error = c10::str(
          "Rank ",
          rank,
          " successfully reached monitoredBarrier, but received errors while waiting",
          " for send/recv from rank 0. Please check rank 0 logs for faulty rank.");
      logAndThrow(
          error, c10::str(error, "\n Original exception: \n", e.what()));
    }
    // 排名非 0 的进程处理完发送和接收后直接返回
    return;
  }
  // 获取当前时间作为开始时间
  auto startTime = std::chrono::steady_clock::now();
  // 获取整个分布式环境的进程总数
  auto worldSize = this->getSize();
  // 映射排名到接收工作和发送工作
  std::map<int, c10::intrusive_ptr<Work>> recvWorkMap;
  std::map<int, c10::intrusive_ptr<Work>> sendWorkMap;
  // 启动接收工作并等待，以解除来自非零排名进程的 sendWork->wait() 阻塞
  // 失败/挂起的进程将不会响应此调用，让排名为 0 的进程知道失败情况。
  for (const auto dstRank : c10::irange(1, worldSize)) {
    recvWorkMap.emplace(
        dstRank, recv(commTensor, dstRank, static_cast<int>(t1)));
  }

  // 定义等待循环函数，用于处理工作列表
  auto waitLoop = [&](const std::map<int, c10::intrusive_ptr<Work>>& works) {
    // 用于记录已处理的进程排名
    std::vector<int> processedRanks;
    for (auto& work : works) {
      // 遍历作业列表中的每个作业
      bool rankResponded = false;
      try {
        // 尝试执行作业
        // 注意：如果 waitAllRanks=false，则重新计算障碍物中剩余的时间，并在 wait() 中使用重新计算后的时间。
        // 如果 waitAllRanks=true，则使用原始的超时时间，因为如果等待响应的排名 n 时耗尽整个超时时间，
        // 那么在开始查询 n + 1 开始的排名时，将没有任何超时时间。
        auto remainingTime =
            getRemainingTime(startTime, monitoredBarrierTimeout, waitAllRanks);
        if (!waitAllRanks) {
          // 检查剩余时间是否足够
          checkRemainingTime(
              monitoredBarrierTimeout, remainingTime, processedRanks, work.first);
        }
        // 等待作业完成，使用剩余时间作为超时
        work.second->wait(remainingTime);
        rankResponded = true;
      } catch (const std::exception& e) {
        // 捕获异常情况
        const std::string error = c10::str(
            "[Rank 0]: Rank ",
            work.first,
            " failed to pass monitoredBarrier in ",
            monitoredBarrierTimeout.count(),
            " ms");
        if (waitAllRanks) {
          // 如果等待所有排名的响应，记录错误信息
          LOG(ERROR) << error;
        } else {
          // 否则，记录并抛出异常
          logAndThrow(
              error, c10::str(error, "\n Original exception: \n", e.what()));
        }
      }
      if (rankResponded) {
        // 如果成功响应，则将排名添加到已处理排名列表中
        processedRanks.push_back(work.first);
      }
    }

    // 如果需要收集所有失败的排名，则检查是否需要抛出异常，以表示有些排名未响应
    // 确保从排名 1 到 WORLD_SIZE - 1 的所有排名都已成功处理
    auto rankFailure =
        (processedRanks.size() != static_cast<size_t>(size_ - 1));
    if (waitAllRanks && rankFailure) {
      // 收集未成功处理的排名，并记录错误信息
      std::vector<int> failedRanks;
      for (const auto i : c10::irange(1, size_)) {
        if (std::find(processedRanks.begin(), processedRanks.end(), i) ==
            processedRanks.end()) {
          failedRanks.push_back(i);
        }
      }

      // 断言确保失败排名列表不为空
      TORCH_INTERNAL_ASSERT(!failedRanks.empty());
      // 构造错误信息字符串
      const std::string ranksStr = c10::Join(", ", failedRanks);
      const std::string error = c10::str(
          "[Rank 0]: Ranks ",
          ranksStr,
          " failed to pass monitoredBarrier in ",
          monitoredBarrierTimeout.count(),
          " ms");
      // 记录并抛出异常
      logAndThrow(error, error);
    }
  };

  // 等待接收工作映射完成
  waitLoop(recvWorkMap);

  // 如果程序顺利执行到这里，表示所有排名在 monitoredBarrier 中已经响应。
  // 现在响应所有排名的接收操作，以确保所有排名都成功退出此阻塞。
  for (const auto dstRank : c10::irange(1, worldSize)) {
    // 将发送操作添加到发送工作映射中
    sendWorkMap.emplace(
        dstRank, send(commTensor, dstRank, static_cast<int>(t2)));
  }

  // 等待发送工作映射完成
  waitLoop(sendWorkMap);
}

// 开始设置组的序列号，对于Gloo，默认从0开始
void ProcessGroupGloo::setSequenceNumberForGroup() {
} // Gloo just starts sequence numbers at 0.

// 获取组的序列号
uint64_t ProcessGroupGloo::getSequenceNumberForGroup() {
  return seq_;
}

// 启用集体操作的定时功能
void ProcessGroupGloo::enableCollectivesTiming() {
  // 没有需要执行的操作来启用定时功能
}

// c10d命名空间的结束标记
} // namespace c10d

// 如果定义了USE_C10D_GLOO，则结束
#endif // USE_C10D_GLOO
```