# `.\pytorch\c10\cuda\CUDAStream.cpp`

```py
// 包含 CUDA C++ API 相关头文件
#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <array>  // 包含数组容器的标准库头文件
#include <atomic> // 包含原子操作相关的标准库头文件
#include <cstdint> // 包含固定大小整数类型的标准库头文件

namespace c10::cuda {

namespace {

// 全局流状态和常量定义
static c10::once_flag init_flag;  // 初始化标志，用于保证初始化操作只执行一次
static DeviceIndex num_gpus = -1; // GPU 设备数量，初始为-1表示未知
static constexpr int kStreamsPerPoolBits = 5;  // 每个池子中流的数量的位数
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;  // 每个池子中流的数量
static constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;  // 默认流标志，非阻塞
static constexpr int kStreamTypeBits = 4;  // 流类型的位数

static int max_stream_priorities;  // 最大流优先级

// 非默认流
// 注意：CUDA 设备数量在运行时确定，低优先级池和高优先级池在首次为设备请求流时延迟初始化。
// 设备标志跟踪每个设备的初始化情况，低优先级和高优先级计数器跟踪每个设备中请求流时要返回的下一个流。
// 这些流被“泄漏”：它们被创建但从未被销毁，因为全局变量的销毁可能发生在 CUDA 运行时已被销毁之后，调用 cudaStreamDestroy 可能导致崩溃。
// 这可能是 CUDA 中的一个问题，为了安全起见，我们只是“忘记”销毁它们。
#if !defined(USE_ROCM)
// 仅 CUDA：用于初始化流池（一次性）
static std::array<c10::once_flag, C10_COMPILE_TIME_MAX_GPUS> device_flags;  // 设备标志数组，用于一次性初始化
#endif

// 优先级计数器和流数组
static std::array<
    std::array<std::atomic<uint32_t>, C10_COMPILE_TIME_MAX_GPUS>,
    c10::cuda::max_compile_time_stream_priorities>
    priority_counters;  // 优先级计数器数组，用于跟踪流的优先级

static std::array<
    std::array<
        std::array<cudaStream_t, kStreamsPerPool>,
        C10_COMPILE_TIME_MAX_GPUS>,
    c10::cuda::max_compile_time_stream_priorities>
    streams;  // 流数组，包含了多个设备上不同优先级的多个流

#ifdef USE_ROCM
static c10::once_flag
    stream_flags[c10::cuda::max_compile_time_stream_priorities]
                [C10_COMPILE_TIME_MAX_GPUS][kStreamsPerPool];  // ROCm 下的流标志数组
#endif

// 注意 [HIP Lazy Streams]
// ROCm/HIP 中，每个流在首次请求时延迟初始化，而不是在请求第一个流时创建所有流。HIP 流不像 CUDA 流那样轻量，
// 池化策略可以影响性能。为了避免更改池化实现，ROCm/HIP 将在每次首次请求时惰性初始化每个流。

// 注意 [StreamId assignment]
// 如何分配流 ID？
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// 零             流 ID 索引位      StreamIdType     扩展/本地流标志
// 对于外部流，StreamID 是一个 cudaStream_t 指针，这意味着最后一位将始终为 0
// 当为本地流构造 StreamId 时，将最后一位设置为 1，以区分本地流和外部流

// 我们有义务将流 ID 0 视为默认流，根据 c10::Stream 指定的不变性，因此这是对“最后一位 = 1 用于本地流”的一个例外。
// 然而，所有其他数字完全是内部实现细节，我们保留重新编号流的权利。

// 注意，MSB（Most Significant Bit，最高有效位）为零非常重要；StreamId 是*有符号*整数，超出有符号整数表示范围的无符号到有符号的转换是未定义行为。
// 您可以通过类似 https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior 的方法解决这个问题，但这似乎对此有些过度。

// 此外，外部管理的流指针（cudaStream_t）可以直接存储在 Id 字段中，因此在这种情况下，我们需要检查流的对齐方式。

class StreamIdType {
  // StreamIdType 编码了此流是默认流、外部流还是所有其他本地流的流优先级（数值越高优先级越高）
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

 public:
  // 构造函数，初始化 stream_type
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  // 判断是否为外部流
  bool isExt() const {
    return EXT == stream_type;
  }

  // 判断是否为默认流
  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  // 获取流类型
  uint8_t getStreamType() const {
    return stream_type;
  }
};

// 自定义输出流操作符，根据 StreamIdType 打印相应信息
std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << int(s.getStreamType());
  }
  return stream;
}

// 获取流的类型信息
static inline StreamIdType streamIdType(StreamId s) {
  // 外部分配的流具有 cudaStream_ptr 作为其 id，因此最后一位将为 0
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  // 最后一位表示外部/内部流，掩码应从倒数第二位开始
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_INTERNAL_ASSERT(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

// 获取流的索引信息
static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

// 构造 StreamId
StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<StreamId>(0);
  }
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      static_cast<StreamId>(st.getStreamType() << 1) | 1;
}

// 线程本地当前流
// NOLINTNEXTLINE(*-arrays)
// 定义静态线程局部变量，存储指向 StreamId 数组的独占指针，初始为空
static thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// 初始化全局流状态
// 警告：此函数只能调用一次！
static void initGlobalStreamState() {
  // 获取设备数量作为全局变量
  num_gpus = device_count();
  // 检查 GPU 数量是否符合预期的编译时最大 GPU 数量
  TORCH_CHECK(
      num_gpus <= C10_COMPILE_TIME_MAX_GPUS,
      "Number of CUDA devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile.");
  int leastPriority = -1, greatestPriority = -1;
  // 获取 CUDA 设备流优先级范围
  C10_CUDA_CHECK(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  // 注释 [HIP stream priorities]
  // HIP 流优先级为 1=低，0=默认，-1=高，这与 CUDA 不同，CUDA 为 0=默认，-1=高，-2=更高等
  // 对于 HIP，将最低优先级调整为 0
#ifdef USE_ROCM
  leastPriority = 0;
#endif
  // 计算优先级范围
  auto range = leastPriority - greatestPriority + 1;
  // 设置最大流优先级为编译时限制或计算出的范围中较小的值
  max_stream_priorities = range >= c10::cuda::max_compile_time_stream_priorities
      ? c10::cuda::max_compile_time_stream_priorities
      : range;
}

// 初始化单个 CUDA 或 HIP 流
// 参见注释 [HIP Lazy Streams]
static void initSingleStream(int p, DeviceIndex device_index, int i) {
  // 获取指定位置的流引用
  auto& stream = streams[p][device_index][i];
  // 计算优先级，越低的数字代表越高的优先级
  auto pri = -p;
  // 创建具有指定优先级的 CUDA 流
  C10_CUDA_CHECK(cudaStreamCreateWithPriority(&stream, kDefaultFlags, pri));
  // 获取 GPU 跟踪器的解释器指针
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    // 如果存在 GPU 跟踪器，跟踪 GPU 流的创建
    (*interp)->trace_gpu_stream_creation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(stream));
    // 将指定优先级对应的计数器重置为 0
    priority_counters[p][device_index] = 0;
  }
}

// 为指定设备初始化低优先级和高优先级流池
// 警告：每个设备仅调用一次！
static void initDeviceStreamState(DeviceIndex device_index) {
  // 切换到指定设备，以便流能够正确关联
  CUDAGuard device_guard{device_index};
  // 遍历每个流池中的每个流并初始化
  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_stream_priorities)) {
      initSingleStream(p, device_index, i);
    }
  }
}

// 保证初始化 CUDA 流仅发生一次的前端函数
static void initCUDAStreamsOnce() {
  // 初始化默认流（全局只调用一次）
  c10::call_once(init_flag, initGlobalStreamState);

  // 如果当前流已经初始化，直接返回
  if (current_streams) {
    return;
  }

  // 初始化当前流（线程局部）为默认流
  // NOLINTNEXTLINE(*-arrays)
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    // 将当前流初始化为默认流
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

// 辅助函数，用于验证 GPU 索引是否有效
static inline void check_gpu(DeviceIndex device_index) {
  // 断言 GPU 索引在有效范围内
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < num_gpus);
}
// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in CUDAStream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  // Increase the counter atomically and retrieve the raw index
  auto raw_idx = counter++;
  // Perform modulo operation to ensure index wraps around within kStreamsPerPool
  return raw_idx % kStreamsPerPool;
}

CUDAStream CUDAStreamForId(DeviceIndex device_index, StreamId stream_id) {
  // Create a CUDAStream object using the provided device index and stream id
  return CUDAStream(
      CUDAStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::CUDA, device_index),
          stream_id));
}

} // anonymous namespace

// See Note [StreamId assignment]
// Retrieves the CUDA stream associated with the current CUDAStream instance
cudaStream_t CUDAStream::stream() const {
  // Extract device index and stream id from the stored stream object
  c10::DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  if (st.isDefault()) {
    // Check if the stream type is default; should have si as 0
    TORCH_INTERNAL_ASSERT(
        si == 0,
        "Unrecognized stream ",
        stream_,
        " (I think this should be the default stream, but I got a non-zero index ",
        si,
        ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like c10::cuda::getStreamFromPool() to get a new stream.");
    return nullptr; // Return nullptr if the stream type is default
  } else if (st.isExt()) {
    // Return the CUDA stream directly for external stream types
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<cudaStream_t>(stream_id);
  } else {
    // For recognized stream types, retrieve and return the corresponding CUDA stream
    auto streamType = st.getStreamType();
    TORCH_INTERNAL_ASSERT(
        streamType >= 1 && streamType <= max_stream_priorities,
        "Unrecognized stream ",
        stream_,
        " (I didn't recognize the stream type, ",
        st,
        " with the value ",
        streamType,
        ")");
#ifdef USE_ROCM
    // Initialize the stream if using ROCm and it's not initialized yet
    // See Note [HIP Lazy Streams]
    c10::call_once(
        stream_flags[st.getStreamType() - 1][device_index][si],
        initSingleStream,
        st.getStreamType() - 1,
        device_index,
        si);
#endif
    // Return the CUDA stream from the streams array for the recognized stream type
    return streams[st.getStreamType() - 1][device_index][si];
  }
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
CUDAStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  // Initialize CUDA stream pools if not initialized yet
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  TORCH_CHECK(
      priority <= 0,
      "Expected cuda stream priority to be less than or equal to 0, got ",
      priority);
  check_gpu(device_index);
#if !defined(USE_ROCM)
  // Initialize device stream state if not initialized yet (CUDA-only)
  // See Note [HIP Lazy Streams]
  c10::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);
#endif
  auto pri_idx = -priority;
  pri_idx =
      std::min(pri_idx, max_stream_priorities - 1); // pri_idx is zero-based
  // Retrieve index for the stream from the corresponding priority counter
  const auto idx = get_idx(priority_counters[pri_idx][device_index]);
  // Create and return a CUDAStream object using the retrieved stream id
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return CUDAStreamForId(device_index, makeStreamId(id_type, idx));
}
// 从 CUDA 流池中获取流对象，根据是否高优先级和设备索引初始化流
CUDAStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  // 确保 CUDA 流池初始化完成
  initCUDAStreamsOnce();
  // 根据是否高优先级确定流的优先级
  int priority = isHighPriority ? -max_stream_priorities + 1 : 0;
  // 从流池中获取流对象并返回
  return getStreamFromPool(priority, device);
}

// 从外部传入的 CUDA 流获取 CUDAStream 对象，使用设备索引和流指针初始化
CUDAStream getStreamFromExternal(
    cudaStream_t ext_stream,
    DeviceIndex device_index) {
  // 将外部流指针作为实际 id 进行处理并返回相应的 CUDAStream 对象
  return CUDAStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}

// 获取默认的 CUDA 流对象，如果设备索引为 -1，则使用当前设备，设置默认设备并检查 GPU
CUDAStream getDefaultCUDAStream(DeviceIndex device_index) {
  // 确保 CUDA 流池初始化完成
  initCUDAStreamsOnce();
  // 如果设备索引为 -1，则使用当前设备索引
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  // 检查并确保设备索引有效
  check_gpu(device_index);
  // 返回指定设备上的默认 CUDAStream 对象
  return CUDAStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

// 获取当前设备上的当前 CUDA 流对象，如果设备索引为 -1，则使用当前设备，设置默认设备并检查 GPU
CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  // 确保 CUDA 流池初始化完成
  initCUDAStreamsOnce();
  // 如果设备索引为 -1，则使用当前设备索引
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  // 检查并确保设备索引有效
  check_gpu(device_index);
  // 返回指定设备上当前的 CUDAStream 对象
  return CUDAStreamForId(device_index, current_streams[device_index]);
}

// 设置当前设备的当前 CUDA 流对象
void setCurrentCUDAStream(CUDAStream stream) {
  // 确保 CUDA 流池初始化完成
  initCUDAStreamsOnce();
  // 设置给定流对象所在设备的当前流
  current_streams[stream.device_index()] = stream.id();
}

// 重载运算符 << ，使其能够输出 CUDAStream 对象到流中
std::ostream& operator<<(std::ostream& stream, const CUDAStream& s) {
  return stream << s.unwrap();
}

} // namespace c10::cuda
```