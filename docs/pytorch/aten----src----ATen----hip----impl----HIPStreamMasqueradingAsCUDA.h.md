# `.\pytorch\aten\src\ATen\hip\impl\HIPStreamMasqueradingAsCUDA.h`

```py
#pragma once
// 定义了一个头文件的预处理指令，确保此文件只被编译一次

#include <c10/hip/HIPStream.h>
// 包含了 c10 库中的 HIPStream 头文件，用于操作 HIP 流

// 在 c10::hip 命名空间中定义 HIPStreamMasqueradingAsCUDA 类
namespace c10 { namespace hip {

// Motivation: 使用 c10::hip 命名空间可以更轻松地进行 hipification，因为不必额外修复命名空间。抱歉！

// 定义一个 masquerading 类型，用于模拟 CUDA 的行为
class HIPStreamMasqueradingAsCUDA {
public:
  
  enum Unchecked { UNCHECKED };
  // 枚举类型 Unchecked，用于表示未检查的情况

  explicit HIPStreamMasqueradingAsCUDA(Stream stream)
    : HIPStreamMasqueradingAsCUDA(UNCHECKED, stream) {
    // 使用未检查的方式将 stream 强制转换成 HIPStreamMasqueradingAsCUDA
    // 检查强制转换是否正确
    TORCH_CHECK(stream.device().is_cuda() /* !!! */);
  }

  explicit HIPStreamMasqueradingAsCUDA(Unchecked, Stream stream)
    // 使用不安全的方式将 "CUDA" 流强制转换成 HIP 流
    : stream_(
        HIPStream(
          Stream(
            Stream::UNSAFE,
            Device(c10::DeviceType::HIP, stream.device_index()),
            stream.id())
        )
      ) {}

  // 新的构造函数，仅用于此目的。不进行强制转换。
  explicit HIPStreamMasqueradingAsCUDA(HIPStream stream) : stream_(stream) {}

  bool operator==(const HIPStreamMasqueradingAsCUDA& other) const noexcept {
    return stream_ == other.stream_;
  }

  bool operator!=(const HIPStreamMasqueradingAsCUDA& other) const noexcept {
    return stream_ != other.stream_;
  }

  operator hipStream_t() const { return stream_.stream(); }
  // 转换为 HIP 流类型的 hipStream_t

  operator Stream() const {
    // 不安全地将 HIP 流强制转换成 "CUDA" 流
    return Stream(Stream::UNSAFE, device(), id());
  }

  DeviceIndex device_index() const { return stream_.device_index(); }
  // 获取设备索引

  // 不安全地将 HIP 设备强制转换成 CUDA 设备
  c10::DeviceType device_type() const { return c10::DeviceType::CUDA; }

  Device device() const {
    // 不安全地将 HIP 设备强制转换成 CUDA 设备
    return Device(c10::DeviceType::CUDA, stream_.device_index());
  }

  StreamId id() const        { return stream_.id(); }
  // 获取流的 ID
  bool query() const         { return stream_.query(); }
  // 查询流的状态
  void synchronize() const   { stream_.synchronize(); }
  // 同步流
  int priority() const       { return stream_.priority(); }
  // 获取流的优先级
  hipStream_t stream() const { return stream_.stream(); }
  // 获取 HIP 流类型的流对象

  Stream unwrap() const {
    // 不安全地将 HIP 流强制转换成 "CUDA" 流
    return Stream(Stream::UNSAFE, device(), id());
  }

  c10::StreamData3 pack3() const noexcept {
    // 在封装之前，不安全地将 HIP 流强制转换成 "CUDA" 流
    return unwrap().pack3();
  }

  static HIPStreamMasqueradingAsCUDA unpack3(StreamId stream_id,
                                             DeviceIndex device_index,
                                             c10::DeviceType device_type) {
    // 注意事项: 构造函数为我们管理 CUDA 到 HIP 的翻译
    return HIPStreamMasqueradingAsCUDA(Stream::unpack3(
        stream_id, device_index, device_type));
  }

  static std::tuple<int, int> priority_range() { return HIPStream::priority_range(); }
  // 获取优先级范围的元组

  // 新方法，获取底层的 HIPStream
  HIPStream hip_stream() const { return stream_; }

private:
  HIPStream stream_;
  // 私有成员变量，存储 HIPStream 类型的对象
};

// 结束 HIPStreamMasqueradingAsCUDA 类的定义

}} // 结束 c10::hip 命名空间
// 返回一个 masquerading as CUDA 的 HIP 流，使用默认的优先级和设备索引
inline getStreamFromPoolMasqueradingAsCUDA(const bool isHighPriority = false, DeviceIndex device = -1) {
  return HIPStreamMasqueradingAsCUDA(getStreamFromPool(isHighPriority, device));
}

// 返回一个 masquerading as CUDA 的 HIP 流，使用给定的优先级和设备索引
HIPStreamMasqueradingAsCUDA
inline getStreamFromPoolMasqueradingAsCUDA(const int priority, DeviceIndex device = -1) {
  return HIPStreamMasqueradingAsCUDA(getStreamFromPool(priority, device));
}

// 返回一个 masquerading as CUDA 的 HIP 流，使用外部 HIP 流和设备索引
inline getStreamFromExternalMasqueradingAsCUDA(hipStream_t ext_stream, DeviceIndex device) {
  return HIPStreamMasqueradingAsCUDA(getStreamFromExternal(ext_stream, device));
}

// 返回默认设备上 masquerading as CUDA 的默认 HIP 流
inline HIPStreamMasqueradingAsCUDA getDefaultHIPStreamMasqueradingAsCUDA(DeviceIndex device_index = -1) {
  return HIPStreamMasqueradingAsCUDA(getDefaultHIPStream(device_index));
}

// 返回当前设备上 masquerading as CUDA 的当前 HIP 流
inline HIPStreamMasqueradingAsCUDA getCurrentHIPStreamMasqueradingAsCUDA(DeviceIndex device_index = -1) {
  return HIPStreamMasqueradingAsCUDA(getCurrentHIPStream(device_index));
}

// 设置当前 HIP 流为给定的 masquerading as CUDA 的 HIP 流
inline void setCurrentHIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA stream) {
  setCurrentHIPStream(stream.hip_stream());
}

// 自定义输出流操作符，输出 masquerading as CUDA 的 HIP 流
inline std::ostream& operator<<(std::ostream& stream, const HIPStreamMasqueradingAsCUDA& s) {
  stream << s.hip_stream() << " (masquerading as CUDA)";
  return stream;
}

}} // namespace c10::hip

// 定义 c10::hip::HIPStreamMasqueradingAsCUDA 的哈希函数
namespace std {
  template <>
  struct hash<c10::hip::HIPStreamMasqueradingAsCUDA> {
    size_t operator()(c10::hip::HIPStreamMasqueradingAsCUDA s) const noexcept {
      return std::hash<c10::Stream>{}(s.unwrap());
    }
  };
} // namespace std
```