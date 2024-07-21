# `.\pytorch\c10\core\impl\InlineDeviceGuard.h`

```py
#pragma once

// This file provides implementations of InlineDeviceGuard and
// InlineOptionalDeviceGuard.

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <type_traits>
#include <utility>

namespace c10::impl {

/**
 * A DeviceGuard is an RAII class that sets a device to some value
 * on construction, and resets the device to its original value on
 * destruction.
 */
template <typename T>
class InlineDeviceGuard {
    // InlineDeviceGuard is a helper class for implementing DeviceGuards.
    // It is templated over a DeviceGuardImpl (anything that implements
    // DeviceGuardImplInterface).  There are two primary ways to instantiate
    // InlineDeviceGuard:

    // With a concrete implementation of DeviceGuardImpl, e.g., CUDAGuardImpl.
    // This is the best way to use InlineDeviceGuard, as all calls are
    // devirtualized, giving you code as efficient as straight line
    // calls to cudaGetDevice/cudaSetDevice.
public:
    // Constructor that initializes the device guard with a given device
    // implementation.
    InlineDeviceGuard() = default;
    ~InlineDeviceGuard() = default;

    // With VirtualGuardImpl, which does a virtual dispatch to a DeviceGuardImpl
    // retrieved from a DeviceType registry.  We have explicitly instantiated
    // InlineDeviceGuard this way as c10::DeviceGuard.

    // If you are in a hurry, you can use InlineDeviceGuard directly:

    // using CUDAGuard = impl::InlineDeviceGuard<CUDAGuardImpl>;

    // However, you can provide a better user experience if you explicitly write a
    // wrapper class that itself contains the template instantiation:

    // class CUDAGuard {
    // public:
    //   // ... the API ...
    // private:
    //   impl::InlineDeviceGuard<CUDAGuardImpl> guard_;
    // }

    // The wrapper class provides a good place to write documentation, and helps
    // avoid weird template instantiation errors when a user incorrectly uses the
    // class.

    // If you need to test this class, consider instantiating it with FakeGuardImpl.
};
// 声明类 InlineDeviceGuard，用于管理设备的RAII机制
class InlineDeviceGuard {
 public:
  // 注意 [忽略 RAII 的默认构造函数]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 原则上，我们可以为 DeviceGuard 添加一个默认构造函数，
  // 它读取当前设备，并承诺在退出时恢复到该设备。
  // 但是，大多数情况下，你可能实际上想要使用 OptionalDeviceGuard
  // （因为如果你从不设置设备，则实际上不需要恢复操作）。
  // 我们在这里删除构造函数，以鼓励您思考实际想要发生的事情。
  explicit InlineDeviceGuard() = delete;

  /// 将当前设备设置为传递的 Device。
  explicit InlineDeviceGuard(Device device)
      : impl_(device.type()), // 使用设备类型创建 impl_
        original_device_(     // 记录当前设备，如果索引为 -1，则使用默认设备
            device.index() == -1 ? impl_.getDevice()
                                 : impl_.exchangeDevice(device)),
        current_device_(      // 记录当前设备，如果索引为 -1，则使用原始设备，否则使用传入的设备
            device.index() == -1 ? original_device_ : device) {}

  /// 将当前设备索引设置为传递的 DeviceIndex。 (设备类型从模板参数 T 推断)
  template <
      typename U = T,
      typename =
          typename std::enable_if_t<!std::is_same_v<U, VirtualGuardImpl>>>
  explicit InlineDeviceGuard(DeviceIndex device_index)
      : InlineDeviceGuard(Device(U::static_type, device_index)) {}

  /// 使用 VirtualGuardImpl 和显式的 DeviceGuardImplInterface 指针构造 InlineDeviceGuard。
  template <
      typename U = T,
      typename = typename std::enable_if_t<std::is_same_v<U, VirtualGuardImpl>>>
  explicit InlineDeviceGuard(
      Device device,
      const DeviceGuardImplInterface* impl)
      : impl_(
            VirtualGuardImpl(impl ? impl : getDeviceGuardImpl(device.type()))),
        original_device_(
            device.index() == -1 ? impl_.getDevice()
                                 : impl_.exchangeDevice(device)),
        current_device_(device.index() == -1 ? original_device_ : device) {}

  /// 禁止复制构造函数
  InlineDeviceGuard(const InlineDeviceGuard<T>&) = delete;
  InlineDeviceGuard<T>& operator=(const InlineDeviceGuard<T>&) = delete;

  /// 禁止移动构造函数，因为 DeviceGuard 没有未初始化状态，这在具有非平凡析构函数的类型上需要
  InlineDeviceGuard(InlineDeviceGuard<T>&& other) = delete;
  InlineDeviceGuard& operator=(InlineDeviceGuard<T>&& other) = delete;

  /// 析构函数，用于在退出时恢复原始设备
  ~InlineDeviceGuard() {
    impl_.uncheckedSetDevice(original_device_);
  }

  /// 设置设备为给定的设备。
  template <
      typename U = T,
      typename std::enable_if_t<!std::is_same_v<U, VirtualGuardImpl>, int> = 0>
  void set_device(at::Device device) {
    AT_ASSERT(
        (U::static_type == DeviceType::HIP && device.is_cuda()) ||
        device.type() == U::static_type);
    auto index = device.index();
    if (index == -1)
      return;
    impl_.setDevice(device);
  /// 将当前设备设置为传入的设备。
  ///
  /// 这个方法等效于在守卫仅支持单个设备类型时调用 set_device。
  template <typename U = T>
  typename std::enable_if_t<!std::is_same_v<U, VirtualGuardImpl>> reset_device(
      at::Device device) {
    set_device(device);
  }

  /// 重置当前设备为原始设备，然后设置为传入的设备。
  ///
  /// 这个方法用于可能不同设备类型的情况下，重置当前设备。
  ///
  /// 注意：这里的实现会跳过一些设备设置，如果可以证明是不必要的。
  ///
  /// 用于测试的可选参数 impl。
  template <typename U = T>
  typename std::enable_if_t<std::is_same_v<U, VirtualGuardImpl>> reset_device(
      at::Device device,
      const impl::DeviceGuardImplInterface* impl = nullptr) {
    auto index = device.index();
    if (index == -1)
      return;
    if (device.type() == original_device_.type()) {
      AT_ASSERT(impl == nullptr || impl->type() == device.type());
      impl_.setDevice(device);
      current_device_ = device;
    } else {
      // 在原地销毁并重建 DeviceGuard
      impl_.setDevice(original_device_);
      impl_ = !impl ? VirtualGuardImpl(device.type()) : VirtualGuardImpl(impl);
      original_device_ = impl_.exchangeDevice(device);
      current_device_ = device;
    }
  }

  /// 设置设备索引为给定索引值，设备类型从原始设备类型推断。
  void set_index(DeviceIndex index) {
    reset_device(Device(original_device_.type(), index));
  }

  /// 返回最近使用 reset_device() 设置的设备，或者构造时的设备。
  Device original_device() const {
    return original_device_;
  }

  /// 返回最近通过该设备守卫设置的设备，可以是构造时设置的，也可以是通过
  /// set_device/reset_device/set_index 设置的。
  Device current_device() const {
    return current_device_;
  }
};

/**
 * A OptionalDeviceGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 *
 * InlineOptionalDeviceGuard is a helper class for implementing
 * OptionalDeviceGuards.  See guidance in InlineDeviceGuard on how to
 * use this.  See OptionalDeviceGuard for user-oriented usage notes.
 */
template <typename T>
class InlineOptionalDeviceGuard {
 public:
  // Note [Explicit initialization of optional fields]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Explicit initialization of optional fields
  // required to workaround an nvcc bug; see
  // https://github.com/pytorch/pytorch/issues/12117

  /// Creates an uninitialized OptionalDeviceGuard.
  explicit InlineOptionalDeviceGuard()
      : guard_() // See Note [Explicit initialization of optional fields]
  {}

  /// Set the current device to the passed Device, if it is not nullopt.
  explicit InlineOptionalDeviceGuard(optional<Device> device_opt)
      : guard_() { // See Note [Explicit initialization of optional fields]
    if (device_opt.has_value()) {
      guard_.emplace(device_opt.value());
    }
  }

  /// Set the current device to the passed DeviceIndex, if it is not nullopt.
  template <
      typename U = T,
      typename =
          typename std::enable_if_t<!std::is_same_v<U, VirtualGuardImpl>>>
  explicit InlineOptionalDeviceGuard(optional<DeviceIndex> device_index_opt)
      : guard_() { // See Note [Explicit initialization of optional fields]
    if (device_index_opt.has_value()) {
      guard_.emplace(device_index_opt.value());
    }
  }

  /// Resets the currently set device to its original device, and then sets the
  /// current device to the passed device (for a possibly different device
  /// type).  Initializes OptionalDeviceGuard if it is not already initialized.
  ///
  /// See notes on why this is called reset_device on InlineDeviceGuard.
  ///
  /// Optional argument is for testing only.
  template <
      typename U = T,
      typename = typename std::enable_if_t<std::is_same_v<U, VirtualGuardImpl>>>
  void reset_device(
      at::Device device,
      const DeviceGuardImplInterface* impl = nullptr) {
    if (!guard_.has_value()) {
      guard_.emplace(device, impl);
    } else {
      guard_->reset_device(device, impl);
    }
  }

  /// Resets the currently set device to its original device, and then sets the
  /// current device to the passed device.  Initializes the guard if it is
  /// not already initialized.  This is effectively equivalent to set_device
  /// when a guard supports only a single device type.
  template <
      typename U = T,
      typename =
          typename std::enable_if_t<!std::is_same_v<U, VirtualGuardImpl>>>
  void reset_device(at::Device device) {
    if (!guard_.has_value()) {
      guard_.emplace(device);
    } else {
      guard_->set_device(device);
    }
  }
  } else {
    // 如果 guard_ 已初始化，则重置设备到给定的 device
    guard_->reset_device(device);
  }
}

/// 设置设备索引为给定值。设备类型在静态时期已知。
template <
    typename U = T,
    typename = typename std::enable_if_t<!std::is_same_v<U, VirtualGuardImpl>>
>
void set_index(DeviceIndex index) {
  if (!guard_.has_value()) {
    // 如果 guard_ 尚未初始化，则用给定的 index 创建并初始化 guard_
    guard_.emplace(index);
  } else {
    // 如果 guard_ 已初始化，则设置其索引为给定的 index
    guard_->set_index(index);
  }
}

/// 返回在初始化 guard 之前立即设置的设备，如果 guard 未初始化则返回 nullopt。
optional<Device> original_device() const {
  return guard_.has_value() ? make_optional(guard_->original_device()) : nullopt;
}

/// 返回使用此设备保护器设置的最近设备，如果 guard 已初始化，则从构造或通过 set_device 设置的设备，否则返回 nullopt。
optional<Device> current_device() const {
  return guard_.has_value() ? make_optional(guard_->current_device()) : nullopt;
}

/// 恢复原始设备，将此 guard 重置为未初始化状态。
void reset() {
  // 重置 guard_，使其变为未初始化状态
  guard_.reset();
}

private:
optional<InlineDeviceGuard<T>> guard_;
};

} // namespace c10::impl
```