# `.\pytorch\c10\core\StreamGuard.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <c10/core/Device.h>
// 包含 C10 库中的 Device 类头文件
#include <c10/core/Stream.h>
// 包含 C10 库中的 Stream 类头文件
#include <c10/core/impl/InlineStreamGuard.h>
// 包含 C10 库中的 InlineStreamGuard 类头文件
#include <c10/core/impl/VirtualGuardImpl.h>
// 包含 C10 库中的 VirtualGuardImpl 类头文件
#include <c10/util/ArrayRef.h>
// 包含 C10 库中的 ArrayRef 类头文件
#include <c10/util/Optional.h>
// 包含 C10 库中的 Optional 类头文件

namespace c10 {

/**
 * A StreamGuard is an RAII class that changes the current device
 * to the device corresponding to some stream, and changes the
 * default stream on that device to be this stream.
 *
 * Use of StreamGuard is HIGHLY discouraged in operator definitions.  In
 * a single operator, you probably don't know enough about the global
 * state of the world to profitably decide how to set streams.  Let
 * the caller handle this appropriately, and just use the current stream
 * in your operator code.
 *
 * This StreamGuard does NOT have an uninitialized state; it is guaranteed
 * to reset the stream and device on exit.  If you are in a situation
 * where you *might* want to setup a stream guard, see OptionalStreamGuard.
 */
struct StreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit StreamGuard() = delete;
  // 显式删除默认构造函数，确保对象总是有有效状态

  /// Set the current device to the device associated with the passed stream,
  /// and set the current  stream on that device to the passed stream.
  explicit StreamGuard(Stream stream) : guard_(stream) {}
  // 构造函数，根据传入的流对象设置当前设备，并将该设备上的默认流设置为传入的流对象

  /// Copy is disallowed
  StreamGuard(const StreamGuard&) = delete;
  // 禁止拷贝构造函数

  StreamGuard& operator=(const StreamGuard&) = delete;
  // 禁止赋值运算符的拷贝

  /// Move is disallowed, as StreamGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  StreamGuard(StreamGuard&& other) = delete;
  // 禁止移动构造函数，因为 StreamGuard 没有未初始化状态，移动构造需要非平凡析构函数支持

  StreamGuard& operator=(StreamGuard&& other) = delete;
  // 禁止移动赋值运算符

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
  /// on , use MultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }
  // 重置当前设置的流为原始流，设备为原始设备，然后设置当前设备为传入流所关联的设备，并将该设备上的流设置为传入的流

  /// Returns the stream that was set at the time the guard was constructed.
  Stream original_stream() const {
    return guard_.original_stream();
  }
  // 返回在创建 StreamGuard 对象时设置的流对象

  /// Returns the most recent stream that was set using this device guard,
  /// either from construction, or via set_stream.
  Stream current_stream() const {
    return guard_.current_stream();
  }
  // 返回使用此设备保护器设置的最近流对象，可以是在构造函数中设置的，也可以是通过 set_stream 设置的

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    // 返回当前设备对象，通过 guard_ 对象调用 current_device 方法
    return guard_.current_device();
  }

  /// 返回最近一次 reset_stream() 设置的设备，或者在构造时设置的设备。
  /// 使用 guard_ 对象的 original_device 方法获取原始设备。
  Device original_device() const {
    // 调用 guard_ 对象的 original_device 方法，返回最初的设备
    return guard_.original_device();
  }

 private:
  // 内部成员变量，用于管理虚拟设备切换的 InlineStreamGuard 对象
  c10::impl::InlineStreamGuard<impl::VirtualGuardImpl> guard_;
};

/**
 * An OptionalStreamGuard is an RAII class that manages the state of a stream,
 * setting it to a specified value upon initialization and restoring the original
 * state upon destruction.
 * See OptionalDeviceGuard for related usage guidance.
 */
struct OptionalStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalStreamGuard() = default;

  /**
   * Initialize the guard with a specific stream.
   * Sets the current device to the device associated with the passed stream
   * and sets the current stream on that device to the passed stream.
   */
  explicit OptionalStreamGuard(Stream stream) : guard_(stream) {}

  /**
   * Initialize the guard with an optional stream.
   * Sets the current device to the device associated with the passed stream,
   * and sets the current stream on that device to the passed stream,
   * if the passed stream is not nullopt.
   */
  explicit OptionalStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy construction is disallowed.
  OptionalStreamGuard(const OptionalStreamGuard&) = delete;
  /// Copy assignment is disallowed.
  OptionalStreamGuard& operator=(const OptionalStreamGuard&) = delete;

  // Move construction is disallowed due to complexities in RAII guards.
  OptionalStreamGuard(OptionalStreamGuard&& other) = delete;

  // Move assignment is disallowed due to complexities in RAII guards.
  OptionalStreamGuard& operator=(OptionalStreamGuard&& other) = delete;

  /**
   * Resets the currently set stream to the original stream and
   * the currently set device to the original device.
   * Then sets the current device to the device associated with the passed stream
   * and sets the current stream on that device to the passed stream.
   * Initializes the guard if it was not previously initialized.
   */
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /**
   * Returns the stream that was set at the time the guard was most recently
   * initialized, or nullopt if the guard is uninitialized.
   */
  optional<Stream> original_stream() const {
    return guard_.original_stream();
  }

  /**
   * Returns the most recently set stream using this guard,
   * either from construction or via reset_stream, if the guard is initialized.
   * Returns nullopt if the guard is uninitialized.
   */
  optional<Stream> current_stream() const {
    return guard_.current_stream();
  }

  /**
   * Restores the original device and stream, resetting this guard to
   * an uninitialized state.
   */
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::VirtualGuardImpl> guard_{};
};

/**
 * A MultiStreamGuard is an RAII class that sets the current streams of a set of
 * devices all at once, and resets them to their original values on destruction.
 */
// 定义一个名为 MultiStreamGuard 的结构体，用于管理多个流的状态
struct MultiStreamGuard {
  /// 在各自的设备上将当前流设置为传递的流
  explicit MultiStreamGuard(ArrayRef<Stream> streams) : guard_(streams) {}

  /// 复制构造函数被禁用
  MultiStreamGuard(const MultiStreamGuard&) = delete;
  /// 赋值运算符重载被禁用
  MultiStreamGuard& operator=(const MultiStreamGuard&) = delete;

  // 查看注释 [Move construction for RAII guards is tricky]
  // 移动构造函数被禁用
  MultiStreamGuard(MultiStreamGuard&& other) = delete;

  // 查看注释 [Move assignment for RAII guards is tricky]
  // 移动赋值运算符重载被禁用
  MultiStreamGuard& operator=(MultiStreamGuard&& other) = delete;

 private:
  // 使用 c10 命名空间中的实现类 InlineMultiStreamGuard，
  // 基于 VirtualGuardImpl 实现多流的内联管理
  c10::impl::InlineMultiStreamGuard<impl::VirtualGuardImpl> guard_;
};

} // namespace c10
```