# `.\pytorch\c10\cuda\CUDAStream.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <cuda_runtime_api.h>
// 包含 CUDA 运行时 API 的头文件

#include <c10/core/DeviceGuard.h>
// 包含 DeviceGuard 类，用于管理设备选择

#include <c10/core/Stream.h>
// 包含 Stream 类，抽象了 CUDA 流

#include <c10/cuda/CUDAFunctions.h>
// 包含 CUDA 相关函数的头文件

#include <c10/util/Exception.h>
// 包含异常处理相关的头文件

/*
 * Stream pool note.
 *
 * A CUDAStream is an abstraction of an actual cuStream on the GPU. CUDAStreams
 * are backed by cuStreams, but they use several pools to minimize the costs
 * associated with creating, retaining, and destroying cuStreams.
 *
 * There are three pools per device, and a device's pools are lazily created.
 *
 * The first pool contains only the default stream. When the default stream
 * is requested it's returned.
 *
 * The second pool is the "low priority" or "default priority" streams. In
 * HIP builds there is no distinction between streams in this pool and streams
 * in the third pool (below). There are 32 of these streams per device, and
 * when a stream is requested one of these streams is returned round-robin.
 * That is, the first stream requested is at index 0, the second at index 1...
 * to index 31, then index 0 again.
 *
 * This means that if 33 low priority streams are requested, the first and
 * last streams requested are actually the same stream (under the covers)
 * and kernels enqueued on them cannot run concurrently.
 *
 * The third pool is the "high priority" streams. The third pool acts like
 * the second pool except the streams are created with a higher priority.
 *
 * These pools suggest that stream users should prefer many short-lived streams,
 * as the cost of acquiring and releasing streams is effectively zero. If
 * many longer-lived streams are required in performance critical scenarios
 * then the functionality here may need to be extended to allow, for example,
 * "reserving" a subset of the pool so that other streams do not accidentally
 * overlap the performance critical streams.
 *
 * Note: although the notion of "current stream for device" is thread local
 * (every OS thread has a separate current stream, as one might expect),
 * the stream pool is global across all threads; stream 0 is always stream 0
 * no matter which thread you use it on.  Multiple threads can synchronize
 * on the same stream.  Although the CUDA documentation is not very clear
 * on the matter, streams are thread safe; e.g., it is safe to enqueue
 * a kernel on the same stream from two different threads.
 */

namespace c10::cuda {

static constexpr int max_compile_time_stream_priorities = 4;
// 编译时支持的 CUDA 流优先级数量

// Value object representing a CUDA stream.  This is just a wrapper
// around c10::Stream, but it comes with a little extra CUDA-specific
// functionality (conversion to cudaStream_t), and a guarantee that
// the wrapped c10::Stream really is a CUDA stream.
// 表示 CUDA 流的值对象。这只是 c10::Stream 的包装，但它具有一些额外的 CUDA 特定功能
// （如转换为 cudaStream_t），并保证包装的 c10::Stream 确实是 CUDA 流。
class C10_CUDA_API CUDAStream {
 public:
  enum Unchecked { UNCHECKED };
  // 未检查的标志枚举值，用于特定的构造函数

  /// Construct a CUDAStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a CUDA stream.
  // 从 Stream 构造一个 CUDAStream。此构造是经过检查的，
  // 如果 Stream 实际上不是 CUDA 流，则会引发错误。
  explicit CUDAStream(Stream stream) : stream_(stream) {
    // 将传入的 Stream 对象初始化为 CUDAStream 的成员变量
    // 这里是构造函数的实现
    // 检查 CUDAStream 对象关联的流是否为 CUDA 设备类型
    TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
  }

  /// 从另一个 Stream 构造一个 CUDAStream，无错误检查。
  /// 此构造函数使用了“命名构造函数”习语，可通过 CUDAStream(CUDAStream::UNCHECKED, stream) 调用。
  explicit CUDAStream(Unchecked, Stream stream) : stream_(stream) {}

  // 比较运算符重载：判断两个 CUDAStream 对象是否相等
  bool operator==(const CUDAStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  // 比较运算符重载：判断两个 CUDAStream 对象是否不相等
  bool operator!=(const CUDAStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// 隐式转换为 cudaStream_t 类型
  operator cudaStream_t() const {
    return stream();
  }

  /// 隐式转换为 Stream 类型（即忘记该流是 CUDA 流）
  operator Stream() const {
    return unwrap();
  }

  /// 获取流关联的设备类型，避免在 Python API 明确指定设备类型
  DeviceType device_type() const {
    return DeviceType::CUDA;
  }

  /// 获取与此流关联的 CUDA 设备索引
  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// 获取此流关联的完整设备信息，保证是 CUDA 设备
  Device device() const {
    return Device(DeviceType::CUDA, device_index());
  }

  /// 返回此流的流 ID
  StreamId id() const {
    return stream_.id();
  }

  /// 查询流是否完成所有操作
  bool query() const {
    // 使用 DeviceGuard 确保在正确的 CUDA 设备上执行操作
    DeviceGuard guard{stream_.device()};
    // 查询 CUDA 流状态并处理可能的错误
    cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaStreamQuery(stream()));

    if (err == cudaSuccess) {
      return true;
    } else if (err != cudaErrorNotReady) {
      // 如果出现错误且不是因为流还未准备好，则检查 CUDA 错误并抛出异常
      C10_CUDA_CHECK(err);
    } else {
      // 如果流还未准备好，则忽略错误并清除错误状态
      (void)cudaGetLastError();
    }

    return false;
  }

  /// 同步 CUDA 流，等待所有操作完成
  void synchronize() const {
    // 使用 DeviceGuard 确保在正确的 CUDA 设备上执行操作
    DeviceGuard guard{stream_.device()};
    // 调用 CUDA 库函数同步 CUDA 流
    c10::cuda::stream_synchronize(stream());
  }

  /// 获取 CUDA 流的优先级
  int priority() const {
    // 使用 DeviceGuard 确保在正确的 CUDA 设备上执行操作
    DeviceGuard guard{stream_.device()};
    int priority = 0;
    // 调用 CUDA 库函数获取 CUDA 流的优先级
    C10_CUDA_CHECK(cudaStreamGetPriority(stream(), &priority));
    return priority;
  }

  /// 显式转换为 cudaStream_t 类型
  cudaStream_t stream() const;

  /// 显式转换为 Stream 类型
  Stream unwrap() const {
    return stream_;
  }

  /// 将 CUDAStream 可逆地打包为结构表示
  /// 先前流的数据被打包到一个 int64_t 中，假设字段总存储空间不超过 64 位。
  /// 参见 https://github.com/pytorch/pytorch/issues/75854 获取更多关于新平台的信息。
  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  // 从 pack() 生成的 3 个字段中解包出 CUDAStream
  static CUDAStream unpack3(
      StreamId stream_id,
      DeviceIndex device_index,
      DeviceType device_type) {
    # 调用 CUDAStream 类的静态方法 unpack3，解析 stream_id、device_index、device_type，并返回 CUDAStream 对象
    return CUDAStream(Stream::unpack3(stream_id, device_index, device_type));
  }

  static std::tuple<int, int> priority_range() {
    # 返回 PyTorch 支持的优先级范围，而不是 CUDA 支持的优先级范围。
    # PyTorch 支持的优先级是 CUDA 支持优先级的一个子集。
    int least_priority = 0, greatest_priority = 0;
    # 调用 CUDA 函数 cudaDeviceGetStreamPriorityRange，获取当前设备支持的流优先级范围，并将结果存储在 least_priority 和 greatest_priority 中
    C10_CUDA_CHECK(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
#ifdef USE_ROCM
    // 如果使用 ROCm，检查最小优先级是否为1，否则抛出异常
    TORCH_INTERNAL_ASSERT(
        least_priority == 1, "Unexpected HIP stream priority range");
    // 将最小优先级设为0，适配 ROCm 的优先级范围
    least_priority = 0;
#else
    // 如果未使用 ROCm，检查最小优先级是否为0，否则抛出异常
    TORCH_INTERNAL_ASSERT(
        least_priority == 0, "Unexpected CUDA stream priority range");
#endif
    // 检查最大优先级是否小于等于-1，否则抛出异常
    TORCH_INTERNAL_ASSERT(
        greatest_priority <= -1, "Unexpected CUDA stream priority range");
    // 调整最大优先级，确保不超过编译时最大支持的流优先级范围
    greatest_priority = std::max(
        -c10::cuda::max_compile_time_stream_priorities + 1, greatest_priority);
    // 返回最小优先级和调整后的最大优先级的元组
    return std::make_tuple(least_priority, greatest_priority);
  }

  // 现在已删除；请使用 CUDAEvent::block
  // void synchronize_with(const CUDAEvent& event) const;

 private:
  Stream stream_;
};

/**
 * 从 CUDA 流池中获取一个新的流。你可以将其视为“创建”新流，但实际上并没有真正创建；
 * 而是从池中预分配流，并以循环方式返回。
 *
 * 通过将 isHighPriority 设置为 true，可以从高优先级池中请求流；
 * 或者通过设置 device（默认为当前 CUDA 流的设备）来请求特定设备的流。
 */
C10_API CUDAStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);
// 没有默认优先级，以消除重载歧义
C10_API CUDAStream
getStreamFromPool(const int priority, DeviceIndex device = -1);

/**
 * 从外部分配的 CUDA 流获取一个 CUDAStream。
 *
 * 主要用于与不同库的互操作，用于在数据交换或类似目的上操作非 Torch 分配的流。
 */
C10_API CUDAStream
getStreamFromExternal(cudaStream_t ext_stream, DeviceIndex device_index);

/**
 * 获取默认的 CUDA 流，用于传递的 CUDA 设备，如果未传递设备索引，则用于当前设备。
 * 默认流是在没有显式使用流时大部分计算发生的地方。
 */
C10_API CUDAStream getDefaultCUDAStream(DeviceIndex device_index = -1);

/**
 * 获取当前的 CUDA 流，用于传递的 CUDA 设备，如果未传递设备索引，则用于当前设备。
 * 当前 CUDA 流通常是设备的默认 CUDA 流，但如果有人调用 'setCurrentCUDAStream' 或使用 'StreamGuard' 或 'CUDAStreamGuard'，它可能不同。
 */
C10_API CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);

/**
 * 将传递流的设备上的当前流设置为传递流。
 * 是的，你没看错：这个函数与当前设备无关：它切换传递流的设备的当前流。
 *
 * 感到困惑？避免使用此函数；更喜欢使用 'CUDAStreamGuard' 来代替
 * （它将按预期方式切换当前设备和当前流，并在之后将其重置回原始状态）。
 */
C10_API void setCurrentCUDAStream(CUDAStream stream);
# 定义重载操作符 <<，用于将 CUDAStream 对象输出到流中
C10_API std::ostream& operator<<(std::ostream& stream, const CUDAStream& s);

# 结束命名空间 c10::cuda
} // namespace c10::cuda

# 进入标准命名空间 std
namespace std {

# 特化 std::hash 模板，用于计算 CUDAStream 对象的哈希值
template <>
struct hash<c10::cuda::CUDAStream> {
  # 重载操作符 ()，计算给定 CUDAStream 对象的哈希值
  size_t operator()(c10::cuda::CUDAStream s) const noexcept {
    # 使用 std::hash<c10::Stream> 的哈希函数计算 CUDAStream 对象内部包装的 Stream 的哈希值
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
# 结束命名空间 std
} // namespace std
```