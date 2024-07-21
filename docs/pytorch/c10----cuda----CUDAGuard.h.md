# `.\pytorch\c10\cuda\CUDAGuard.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <c10/core/DeviceType.h>
// 包含设备类型相关的头文件
#include <c10/core/impl/InlineDeviceGuard.h>
// 包含内联设备保护实现的头文件
#include <c10/core/impl/InlineStreamGuard.h>
// 包含内联流保护实现的头文件
#include <c10/cuda/CUDAMacros.h>
// 包含 CUDA 宏定义的头文件
#include <c10/cuda/impl/CUDAGuardImpl.h>
// 包含 CUDA 设备保护实现的头文件

namespace c10::cuda {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]
// 这段代码有点模板化。参见注释 [Whither the DeviceGuard boilerplate]

/// A variant of DeviceGuard that is specialized for CUDA.  It accepts
/// integer indices (interpreting them as CUDA devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// cudaSetDevice/cudaGetDevice calls); however, it can only be used
/// from code that links against CUDA directly.
/// CUDA 专用的 DeviceGuard 变体。它接受整数索引（将其解释为 CUDA 设备），比普通的 DeviceGuard 稍微高效
/// （编译为直接的 cudaSetDevice/cudaGetDevice 调用）；但是，它只能在直接链接到 CUDA 的代码中使用。
struct CUDAGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  /// 没有默认构造函数；参见注释 [Omitted default constructor from RAII]
  explicit CUDAGuard() = delete;

  /// Set the current CUDA device to the passed device index.
  /// 将当前 CUDA 设备设置为传入的设备索引。
  explicit CUDAGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current CUDA device to the passed device.  Errors if the passed
  /// device is not a CUDA device.
  /// 将当前 CUDA 设备设置为传入的设备。如果传入的设备不是 CUDA 设备，则出错。
  explicit CUDAGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  // 禁止复制构造函数和赋值运算符的使用
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  // 禁止移动构造函数和移动赋值运算符的使用（没有未初始化的状态）
  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;

  /// Sets the CUDA device to the given device.  Errors if the given device
  /// is not a CUDA device.
  /// 将 CUDA 设备设置为给定的设备。如果给定的设备不是 CUDA 设备，则出错。
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the CUDA device to the given device.  Errors if the given device
  /// is not a CUDA device.  (This method is provided for uniformity with
  /// DeviceGuard).
  /// 将 CUDA 设备设置为给定的设备。如果给定的设备不是 CUDA 设备，则出错。
  /// （为了与 DeviceGuard 保持一致，提供此方法。）
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the CUDA device to the given device index.
  /// 将 CUDA 设备设置为给定的设备索引。
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  /// 返回在构造 guard 时设置的设备
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  /// 返回最后通过 `set_device` 设置的设备，如果没有，则返回构造时传入的设备。
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  /// 当前设备的保护对象。
  c10::impl::InlineDeviceGuard<impl::CUDAGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for CUDA.  See
/// CUDAGuard for when you can use this.
/// CUDA 专用的 OptionalDeviceGuard 变体。查看 CUDAGuard 的使用条件。
// 表示一个可选的 CUDA 设备保护对象，用于管理 CUDA 设备的 RAII 封装

struct OptionalCUDAGuard {
  /// 创建一个未初始化的 OptionalCUDAGuard 对象。
  explicit OptionalCUDAGuard() : guard_() {}

  /// 如果 device_opt 不为 nullopt，则将当前 CUDA 设备设置为传入的 Device。
  explicit OptionalCUDAGuard(optional<Device> device_opt)
      : guard_(device_opt) {}

  /// 如果 device_index_opt 不为 nullopt，则将当前 CUDA 设备设置为传入的设备索引。
  explicit OptionalCUDAGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // 禁止复制构造函数
  OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
  OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;

  // 注意 [移动构造对于 RAII 保护对象很棘手]
  OptionalCUDAGuard(OptionalCUDAGuard&& other) = delete;

  // 注意 [移动赋值对于 RAII 保护对象很棘手]
  OptionalCUDAGuard& operator=(OptionalCUDAGuard&& other) = delete;

  /// 设置 CUDA 设备为给定的 device，如果未初始化则进行初始化。
  /// 如果给定的 device 不是 CUDA 设备，则会报错。
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// 设置 CUDA 设备为给定的 device，如果未初始化则进行初始化。
  /// 如果给定的 device 不是 CUDA 设备，则会报错。
  /// （此方法用于与 OptionalDeviceGuard 保持一致性。）
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// 设置 CUDA 设备为给定的设备索引，如果未初始化则进行初始化。
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// 返回在初始化此 guard 之前设置的设备，如果 guard 未初始化则返回 nullopt。
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// 返回使用此设备 guard 设置的最近设备，如果 guard 初始化则返回 nullopt。
  optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// 恢复原始的 CUDA 设备，重置此 guard 为未初始化状态。
  void reset() {
    guard_.reset();
  }

 private:
  // 内部使用的实际 CUDA 设备保护对象
  c10::impl::InlineOptionalDeviceGuard<impl::CUDAGuardImpl> guard_;
};

/// 专为 CUDA 特化的 StreamGuard 变体。详见 CUDAGuard 的使用场景。
struct CUDAStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit CUDAStreamGuard() = delete;

  /// Set the current CUDA device to the device associated with the passed
  /// stream, and set the current CUDA stream on that device to the passed
  /// stream. Errors if the Stream is not a CUDA stream.
  explicit CUDAStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  CUDAStreamGuard(const CUDAStreamGuard&) = delete;
  CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;

  /// Move is disallowed, as CUDAStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  CUDAStreamGuard(CUDAStreamGuard&& other) = delete;
  CUDAStreamGuard& operator=(CUDAStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a CUDA stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on CUDA, use CUDAMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the CUDA stream that was set at the time the guard was
  /// constructed.
  CUDAStream original_stream() const {
    return CUDAStream(CUDAStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent CUDA stream that was set using this device guard,
  /// either from construction, or via set_stream.
  CUDAStream current_stream() const {
    return CUDAStream(CUDAStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent CUDA device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the CUDA device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::CUDAGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for CUDA.  See
/// CUDAGuard for when you can use this.


注释：


/// No default constructor, see Note [Omitted default constructor from RAII]
explicit CUDAStreamGuard() = delete;
/// 明确禁用默认构造函数，详见说明文档中关于 RAII 的注释

/// Set the current CUDA device to the device associated with the passed
/// stream, and set the current CUDA stream on that device to the passed
/// stream. Errors if the Stream is not a CUDA stream.
explicit CUDAStreamGuard(Stream stream) : guard_(stream) {}
/// 使用传入的流对象，将当前 CUDA 设备切换至该流对象关联的设备，
/// 并将该设备上的当前 CUDA 流设置为传入的流对象。如果传入的流不是 CUDA 流，则报错。

/// Copy is disallowed
CUDAStreamGuard(const CUDAStreamGuard&) = delete;
CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
/// 禁用拷贝构造函数和拷贝赋值运算符

/// Move is disallowed, as CUDAStreamGuard does not have an uninitialized
/// state, which is required for moves on types with nontrivial destructors.
CUDAStreamGuard(CUDAStreamGuard&& other) = delete;
CUDAStreamGuard& operator=(CUDAStreamGuard&& other) = delete;
/// 禁用移动构造函数和移动赋值运算符，因为 CUDAStreamGuard 类没有未初始化状态，
/// 而移动构造和移动赋值通常要求对象有非平凡析构函数的未初始化状态。

/// Resets the currently set stream to the original stream and
/// the currently set device to the original device.  Then,
/// set the current device to the device associated with the passed stream,
/// and set the current stream on that device to the passed stream.
/// Errors if the stream passed is not a CUDA stream.
///
/// NOTE: this implementation may skip some stream/device setting if
/// it can prove that it is unnecessary.
///
/// WARNING: reset_stream does NOT preserve previously set streams on
/// different devices.  If you need to set streams on multiple devices
/// on CUDA, use CUDAMultiStreamGuard instead.
void reset_stream(Stream stream) {
  guard_.reset_stream(stream);
}
/// 重置当前设置的流为原始流，并将当前设置的设备重置为原始设备。
/// 然后，将当前设备切换至与传入流对象相关联的设备，并将该设备上的当前流设置为传入的流对象。
/// 如果传入的流不是 CUDA 流，则报错。

/// Returns the CUDA stream that was set at the time the guard was
/// constructed.
CUDAStream original_stream() const {
  return CUDAStream(CUDAStream::UNCHECKED, guard_.original_stream());
}
/// 返回在创建该 guard 时设置的 CUDA 流对象。

/// Returns the most recent CUDA stream that was set using this device guard,
/// either from construction, or via set_stream.
CUDAStream current_stream() const {
  return CUDAStream(CUDAStream::UNCHECKED, guard_.current_stream());
}
/// 返回使用该设备 guard 设置的最近的 CUDA 流对象，
/// 可能是通过构造函数设置，也可能是通过 set_stream 设置。

/// Returns the most recent CUDA device that was set using this device guard,
/// either from construction, or via set_device/reset_device/set_index.
Device current_device() const {
  return guard_.current_device();
}
/// 返回使用该设备 guard 设置的最近的 CUDA 设备，
/// 可能是通过构造函数设置，也可能是通过 set_device/reset_device/set_index 设置。

/// Returns the CUDA device that was set at the most recent reset_stream(),
/// or otherwise the device at construction time.
Device original_device() const {
  return guard_.original_device();
}
/// 返回在最近一次 reset_stream() 被调用时设置的 CUDA 设备，
/// 或者是在构造函数调用时设置的设备。

private:
c10::impl::InlineStreamGuard<impl::CUDAGuardImpl> guard_;
};
/// CUDAStreamGuard 的实现，用于管理 CUDA 设备和流对象之间的关联性。

/// A variant of OptionalStreamGuard that is specialized for CUDA.  See
/// CUDAGuard for when you can use this.
/// 用于管理可选的 CUDA 流的 RAII（资源获取即初始化）守卫。
struct OptionalCUDAStreamGuard {
  /// 创建一个未初始化的守卫。
  explicit OptionalCUDAStreamGuard() : guard_() {}

  /// 根据传入的流设置当前 CUDA 设备，并将该设备上的当前 CUDA 流设置为传入的流。
  /// 如果流不是 CUDA 流，则会报错。
  explicit OptionalCUDAStreamGuard(Stream stream) : guard_(stream) {}

  /// 如果传入的流不是空，设置当前设备为与传入流关联的设备，
  /// 并将该设备上的当前流设置为传入的流。
  explicit OptionalCUDAStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// 复制构造函数被禁用。
  OptionalCUDAStreamGuard(const OptionalCUDAStreamGuard&) = delete;
  /// 复制赋值操作符被禁用。
  OptionalCUDAStreamGuard& operator=(const OptionalCUDAStreamGuard&) = delete;

  // 注意 [RAII 守卫的移动构造是棘手的]
  /// 移动构造函数被禁用。
  OptionalCUDAStreamGuard(OptionalCUDAStreamGuard&& other) = delete;

  // 注意 [RAII 守卫的移动赋值是棘手的]
  /// 移动赋值操作符被禁用。
  OptionalCUDAStreamGuard& operator=(OptionalCUDAStreamGuard&& other) = delete;

  /// 将当前设置的 CUDA 流重置为原始流，同时将当前设备重置为原始设备。
  /// 然后，如果之前未初始化，将当前设备设置为与传入流关联的设备，
  /// 并将该设备上的当前流设置为传入的流。
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// 返回初始化守卫时设置的 CUDA 流，如果守卫未初始化则返回 nullopt。
  optional<CUDAStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(CUDAStream(CUDAStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// 返回最近通过此流守卫设置的当前 CUDA 流，
  /// 无论是通过构造函数还是通过 reset_stream 设置的，
  /// 如果守卫未初始化则返回 nullopt。
  optional<CUDAStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(CUDAStream(CUDAStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// 恢复原始的 CUDA 设备和流设置，将此守卫重置为未初始化状态。
  void reset() {
    guard_.reset();
  }

 private:
  /// 内部使用的 CUDA 守卫实现，负责实际的流管理。
  c10::impl::InlineOptionalStreamGuard<impl::CUDAGuardImpl> guard_;
};

/// 专门用于 CUDA 的 MultiStreamGuard 的变体。
struct CUDAMultiStreamGuard {
  // 构造函数，接受一个 CUDAStream 数组引用，初始化成员变量 guard_
  explicit CUDAMultiStreamGuard(ArrayRef<CUDAStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  // 复制构造函数被禁用
  CUDAMultiStreamGuard(const CUDAMultiStreamGuard&) = delete;
  // 复制赋值运算符被禁用
  CUDAMultiStreamGuard& operator=(const CUDAMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  // 移动构造函数被禁用，参见注释 [移动构造对于 RAII 保护是棘手的]
  CUDAMultiStreamGuard(CUDAMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  // 移动赋值运算符被禁用，参见注释 [移动赋值对于 RAII 保护是棘手的]
  CUDAMultiStreamGuard& operator=(CUDAMultiStreamGuard&& other) = delete;

 private:
  // 内部成员变量，用于实际管理 CUDA 流的 RAII 保护
  c10::impl::InlineMultiStreamGuard<impl::CUDAGuardImpl> guard_;

  // 静态成员函数，将 CUDAStream 数组转换为 Stream 类型的向量
  static std::vector<Stream> unwrapStreams(ArrayRef<CUDAStream> cudaStreams) {
    std::vector<Stream> streams;
    streams.reserve(cudaStreams.size());
    // 遍历 CUDAStream 数组，逐个将 CUDAStream 转换为 Stream 并添加到向量中
    for (const CUDAStream& cudaStream : cudaStreams) {
      streams.push_back(cudaStream);
    }
    // 返回包含所有转换后 Stream 对象的向量
    return streams;
  }
};

} // namespace c10::cuda
```