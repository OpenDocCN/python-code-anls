# `.\pytorch\c10\core\impl\InlineStreamGuard.h`

```py
#pragma once

#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>

namespace c10::impl {

/**
 * A StreamGuard is an RAII class that changes the current device
 * to the device corresponding to some stream, and changes the
 * default stream on that device to be this stream.
 *
 * InlineStreamGuard is a helper class for implementing StreamGuards.
 * See InlineDeviceGuard for guidance on how to use this class.
 */
template <typename T>
class InlineStreamGuard : private InlineDeviceGuard<T> {
 public:
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit InlineStreamGuard() = delete;

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  explicit InlineStreamGuard(Stream stream)
      : InlineDeviceGuard<T>(stream.device()),
        original_stream_of_original_device_(
            this->impl_.getStream(original_device())),
        original_stream_of_current_device_(this->impl_.exchangeStream(stream)),
        current_stream_(stream) {}

  /// This constructor exists purely for testing
  template <
      typename U = T,
      typename = typename std::enable_if_t<std::is_same_v<U, VirtualGuardImpl>>>
  explicit InlineStreamGuard(
      Stream stream,
      const DeviceGuardImplInterface* impl)
      : InlineDeviceGuard<T>(
            stream.device(),
            impl ? impl : getDeviceGuardImpl(stream.device_type())),
        original_stream_of_original_device_(
            this->impl_.getStream(original_device())),
        original_stream_of_current_device_(this->impl_.exchangeStream(stream)),
        current_stream_(stream) {}

  /// Copy is disallowed
  InlineStreamGuard(const InlineStreamGuard<T>&) = delete;
  InlineStreamGuard<T>& operator=(const InlineStreamGuard<T>&) = delete;

  /// Move is disallowed, as StreamGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  InlineStreamGuard(InlineStreamGuard<T>&& other) = delete;
  InlineStreamGuard& operator=(InlineStreamGuard<T>&& other) = delete;

  /// Destructor that restores the original stream setting on destruction of the guard
  ~InlineStreamGuard() {
    this->impl_.exchangeStream(original_stream_of_current_device_);
  }

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// use MultiStreamGuard instead.
  void reset_stream(Stream stream) {
    // TODO: make a version that takes an impl argument.  Unfortunately,
    // the current code (as of now) does not support this.
    // Exchange the current stream with the original stream of the current device
    this->impl_.exchangeStream(original_stream_of_current_device_);
    // Set the current device to the device associated with the passed stream
    // and update the current stream to the passed stream
    original_stream_of_current_device_ = this->impl_.exchangeStream(stream);
    current_stream_ = stream;
  }

 private:
  T original_stream_of_original_device_;  ///< Original stream of the original device
  T original_stream_of_current_device_;   ///< Original stream of the current device
  Stream current_stream_;                 ///< Current stream being managed
};

} // namespace c10::impl
    // 如果流的设备与当前设备相同，则直接交换流并更新当前流变量
    if (stream.device() == this->current_device()) {
      this->impl_.exchangeStream(stream);
      current_stream_ = stream;
    } else {
      // 否则，需要先销毁原设备的流，然后重建一个流对象
      this->impl_.exchangeStream(original_stream_of_current_device_);
      // 重置当前设备为新流的设备
      this->reset_device(stream.device());
      // 将新流设为当前流，并记录为原设备的流
      original_stream_of_current_device_ = this->impl_.exchangeStream(stream);
      current_stream_ = stream;
    }
  }

  // 不清楚是否在设置设备时也应该重置当前流
  // 如果设备未更改，则可能不应重置当前流；因此，我们不提供这个功能。
  // 在 reset_device 情况下更加明确，但这仍然是一个相当奇怪的操作，因此也没有添加。

  /// 返回此 guard 前原始设备的流。
  /// 这里返回的流是*原始*设备的原始流；即，如果没有此流 guard 干扰，计算将放在这个流上。
  /// 这通常是您想要的。
  Stream original_stream() const {
    return original_stream_of_original_device_;
  }

  /// 返回最近使用此设备 guard 设置的流，无论是通过构造还是通过 set_stream。
  Stream current_stream() const {
    return current_stream_;
  }

  /// 返回最近使用此设备 guard 设置的设备，无论是通过构造还是通过 set_device/reset_device/set_index。
  Device current_device() const {
    return InlineDeviceGuard<T>::current_device();
  }

  /// 返回最近在 reset_stream() 时设置的设备，或者构造时的设备。
  Device original_device() const {
    return InlineDeviceGuard<T>::original_device();
  }

 private:
  Stream
      original_stream_of_original_device_; // 用户可能关心的原始流
  Stream original_stream_of_current_device_; // 需要恢复的当前设备的流
  Stream current_stream_;
};

/**
 * An OptionalStreamGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 * See InlineOptionalDeviceGuard for more guidance on how to use this class.
 */
template <typename T>
class InlineOptionalStreamGuard {
 public:
  /// Creates an uninitialized stream guard.
  explicit InlineOptionalStreamGuard()
      : guard_() // See Note [Explicit initialization of optional fields]
  {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit InlineOptionalStreamGuard(optional<Stream> stream_opt) : guard_() {
    if (stream_opt.has_value()) {
      guard_.emplace(stream_opt.value());
    }
  }

  /// All constructors of StreamGuard are valid for OptionalStreamGuard
  template <typename... Args>
  explicit InlineOptionalStreamGuard(Args&&... args)
      : guard_(std::in_place, std::forward<Args>(args)...) {}

  // See Note [Move construction for RAII guards is tricky]
  InlineOptionalStreamGuard(InlineOptionalStreamGuard<T>&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  InlineOptionalStreamGuard& operator=(InlineOptionalStreamGuard&& other) =
      delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the OptionalStreamGuard if it was not previously initialized.
  void reset_stream(Stream stream) {
    if (guard_.has_value()) {
      guard_->reset_stream(stream);
    } else {
      guard_.emplace(stream);
    }
  }

  /// Returns the stream that was set at the time the guard was most recently
  /// initialized, or nullopt if the guard is uninitialized.
  optional<Stream> original_stream() const {
    return guard_.has_value() ? make_optional(guard_->original_stream())
                              : nullopt;
  }

  /// Returns the most recent stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<Stream> current_stream() const {
    return guard_.has_value() ? make_optional(guard_->current_stream())
                              : nullopt;
  }

  /// Restore the original device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  optional<InlineStreamGuard<T>> guard_;
};

/**
 * InlineMultiStreamGuard is a class that manages multiple stream guards.
 * It initializes each guard with the corresponding stream from the given list.
 * This can be useful for managing streams across multiple devices.
 */
template <typename T>
class InlineMultiStreamGuard {
 public:
  /// Calls `set_stream` on each of the streams in the list.
  /// This may be useful if you need to set different streams
  /// for different devices.
  explicit InlineMultiStreamGuard(ArrayRef<Stream> streams) {
    // 如果流列表不为空
    if (!streams.empty()) {
      // 在实现对象中插入设备类型
      impl_.emplace(getDeviceTypeOfStreams(streams));
      // 预留空间以容纳原始流列表的大小
      original_streams_.reserve(streams.size());
      // 遍历流列表中的每一个流对象
      for (const Stream& s : streams) {
        // 将每个流对象交换到实现对象中，并将原始流对象保存到原始流列表中
        original_streams_.emplace_back(this->impl_->exchangeStream(s));
      }
    }
  }

  /// 禁用拷贝构造函数
  InlineMultiStreamGuard(const InlineMultiStreamGuard&) = delete;
  /// 禁用拷贝赋值运算符
  InlineMultiStreamGuard<T>& operator=(const InlineMultiStreamGuard&) = delete;

  /// 禁用移动构造函数，因为StreamGuard没有未初始化状态，对于具有非平凡析构函数的类型是必需的。
  InlineMultiStreamGuard(InlineMultiStreamGuard&& other) = delete;
  /// 禁用移动赋值运算符，同上
  InlineMultiStreamGuard& operator=(InlineMultiStreamGuard&& other) = delete;

  ~InlineMultiStreamGuard() noexcept {
    // 如果实现对象有值
    if (this->impl_.has_value()) {
      // 将保存在原始流列表中的每个流对象重新交换回实现对象中
      for (const Stream& s : original_streams_) {
        this->impl_->exchangeStream(s);
      }
    }
  }

 protected:
  // 可选类型的实现对象
  optional<T> impl_;

 private:
  /// 存储所有设备上活跃的原始流对象的向量
  std::vector<Stream> original_streams_;

  // 获取流列表中流对象的设备类型
  static DeviceType getDeviceTypeOfStreams(ArrayRef<Stream> streams) {
    // 断言流列表不为空
    TORCH_INTERNAL_ASSERT(!streams.empty());
    // 初始设备类型为第一个流对象的设备类型
    DeviceType type = streams[0].device_type();
    // 遍历流列表中的每个流对象
    for (const auto idx : c10::irange(1, streams.size())) {
      // 检查各流对象的设备类型是否一致
      TORCH_CHECK_VALUE(
          streams[idx].device_type() == type,
          "Streams have a mix of device types: stream 0 is on ",
          streams[0].device(),
          " while stream ",
          idx,
          " is on device ",
          streams[idx].device());
    }
    // 返回流列表中流对象的设备类型
    return type;
  }
};

// 结束命名空间 c10::impl
} // namespace c10::impl
```