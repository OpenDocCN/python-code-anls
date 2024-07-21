# `.\pytorch\c10\core\Stream.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/Device.h>
// 包含 c10 核心库中的设备定义头文件
#include <c10/core/DeviceType.h>
// 包含 c10 核心库中的设备类型定义头文件
#include <c10/macros/Export.h>
// 包含 c10 宏定义头文件，用于导出符号
#include <c10/util/Exception.h>
// 包含 c10 实用工具中的异常处理头文件
#include <cstddef>
// 包含标准库中的 cstddef 头文件，定义了 size_t 类型
#include <cstdint>
// 包含标准库中的 cstdint 头文件，定义了整数类型
#include <functional>
// 包含标准库中的 functional 头文件，提供了函数对象的操作
#include <ostream>
// 包含标准库中的 ostream 头文件，定义了输出流对象

namespace c10 {

/// An index representing a specific stream.  A StreamId is not independently
/// meaningful without knowing the Device it is associated with; try to
/// use Stream rather than StreamId directly.
///
/// StreamIds are opaque; they are assigned by some DeviceType-specific
/// numbering system which is not visible to the user.  HOWEVER, we
/// guarantee that StreamId 0 is always a valid stream, and corresponds
/// to some sort of "default" stream.
using StreamId = int64_t;
// 定义 StreamId 类型为 int64_t，用于表示特定流的索引

struct C10_API StreamData3 {
  StreamId stream_id;
  // 流的 ID，用于标识特定流
  DeviceIndex device_index;
  // 设备索引，表示流关联的设备
  DeviceType device_type;
  // 设备类型，表示流关联的设备类型
};

// NB: I decided not to call the above StreamIndex to avoid confusion with
// DeviceIndex.  This way, you access device index with index(), and stream id
// with id()

/**
 * A stream is a software mechanism used to synchronize launched kernels
 * without requiring explicit synchronizations between kernels.  The basic
 * model is that every kernel launch is associated with a stream: every
 * kernel on the same stream is implicitly synchronized so that if I launch
 * kernels A and B on the same stream, A is guaranteed to finish before B
 * launches.  If I want B to run concurrently with A, I must schedule
 * it on a different stream.
 *
 * The Stream class is a backend agnostic value class representing a stream
 * which I may schedule a kernel on.  Every stream is associated with a device,
 * which is recorded in stream, which is used to avoid confusion about which
 * device a stream refers to.
 *
 * Streams are explicitly thread-safe, in the sense that it is OK to pass
 * a Stream from one thread to another, and kernels queued from two different
 * threads will still get serialized appropriately.  (Of course, the
 * time when the kernels get queued is undetermined unless you synchronize
 * host side ;)
 *
 * Stream does NOT have a default constructor.  Streams are for expert
 * users; if you want to use Streams, we're going to assume you know
 * how to deal with C++ template error messages if you try to
 * resize() a vector of Streams.
 *
 * Known instances of streams in backends:
 *
 *  - cudaStream_t (CUDA)
 *  - hipStream_t (HIP)
 *  - cl_command_queue (OpenCL)  (NB: Caffe2's existing OpenCL integration
 *    does NOT support command queues.)
 *
 * Because this class is device agnostic, it cannot provide backend-specific
 * functionality (e.g., get the cudaStream_t of a CUDA stream.)  There are
 * wrapper classes which provide this functionality, e.g., CUDAStream.
 */
/// Class representing a stream in an API, with specific attributes related to device and ID.
class C10_API Stream final {
 private:
  Device device_;  ///< Instance of the Device class associated with this stream.
  StreamId id_;    ///< Identifier for this stream.

 public:
  enum Unsafe { UNSAFE };    ///< Enumeration indicating unsafe construction mode.
  enum Default { DEFAULT };  ///< Enumeration indicating default construction mode.

  /// Construct a stream unsafely from given Device and StreamId.
  /// Only specific backend implementations should use this constructor directly.
  explicit Stream(Unsafe, Device device, StreamId id)
      : device_(device), id_(id) {}

  /// Construct the default stream of a Device.
  /// The default stream is fixed and different from the current stream,
  /// which may be changed by StreamGuard.
  explicit Stream(Default, Device device) : device_(device), id_(0) {}

  /// Equality comparison operator for streams.
  bool operator==(const Stream& other) const noexcept {
    return this->device_ == other.device_ && this->id_ == other.id_;
  }

  /// Inequality comparison operator for streams.
  bool operator!=(const Stream& other) const noexcept {
    return !(*this == other);
  }

  /// Retrieve the associated Device object of this stream.
  Device device() const noexcept {
    return device_;
  }

  /// Retrieve the type of the associated device.
  DeviceType device_type() const noexcept {
    return device_.type();
  }

  /// Retrieve the index of the associated device.
  DeviceIndex device_index() const noexcept {
    return device_.index();
  }

  /// Retrieve the ID of this stream.
  StreamId id() const noexcept {
    return id_;
  }

  /// Enqueue a wait instruction in the stream's work queue.
  /// This operation blocks the stream until the event is recorded,
  /// if the event is marked for recording.
  template <typename T>
  void wait(const T& event) const {
    event.block(*this);
  }

  /// Check if all asynchronous work previously enqueued on this stream
  /// has completed running on the device.
  bool query() const;

  /// Block the calling thread until all asynchronous work enqueued on this
  /// stream has completed running on the device.
  void synchronize() const;

  /// Generate a hash value representing this stream.
  /// The hash is based on device type, device index, and stream ID.
  uint64_t hash() const noexcept {
    uint64_t bits = static_cast<uint64_t>(device_type()) << 56 |
        static_cast<uint64_t>(device_index()) << 48 |
        (static_cast<uint64_t>(id()) & ((1ull << 48) - 1));
    return bits;
  }

  /// Structure packing function, allowing Stream to be packed into a StreamData3 structure.
  struct StreamData3 pack3() const {
    # 返回一个包含 id()、device_index()、device_type() 的字典
    return {id(), device_index(), device_type()};
  }

  # 解包函数，根据流ID、设备索引和设备类型创建流对象
  static Stream unpack3(
      StreamId stream_id,           # 流的唯一标识符
      DeviceIndex device_index,     # 设备索引
      DeviceType device_type) {     # 设备类型
    # 检查设备类型是否有效
    TORCH_CHECK(isValidDeviceType(device_type));
    # 使用 UNSAFE 标志和给定的设备类型、设备索引、流ID创建流对象
    return Stream(UNSAFE, Device(device_type, device_index), stream_id);
  }

  # 关于不提供设置器（setters）的注释，因为在流对象中，为什么要更改设备呢？最好一开始就正确构造它。
};

C10_API std::ostream& operator<<(std::ostream& stream, const Stream& s);


// 定义流输出运算符重载函数，用于将 c10::Stream 对象输出到流中
C10_API std::ostream& operator<<(std::ostream& stream, const Stream& s);



} // namespace c10


// 结束 c10 命名空间的定义
} // namespace c10



namespace std {


// 定义标准库命名空间 std
namespace std {



template <>
struct hash<c10::Stream> {
  size_t operator()(c10::Stream s) const noexcept {
    return std::hash<uint64_t>{}(s.hash());
  }
};


// 为 c10::Stream 特化 std::hash 结构体模板
template <>
struct hash<c10::Stream> {
  // 重载 () 运算符，计算给定 c10::Stream 对象的哈希值
  size_t operator()(c10::Stream s) const noexcept {
    return std::hash<uint64_t>{}(s.hash());
  }
};



} // namespace std


// 结束 std 命名空间的定义
} // namespace std
```