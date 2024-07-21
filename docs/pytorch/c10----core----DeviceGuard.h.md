# `.\pytorch\c10\core\DeviceGuard.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <c10/core/Device.h>
// 包含 c10 核心库中的 Device 头文件

#include <c10/core/impl/DeviceGuardImplInterface.h>
// 包含 c10 核心库中的 DeviceGuardImplInterface 头文件

#include <c10/core/impl/InlineDeviceGuard.h>
// 包含 c10 核心库中的 InlineDeviceGuard 头文件

#include <c10/core/impl/VirtualGuardImpl.h>
// 包含 c10 核心库中的 VirtualGuardImpl 头文件

#include <c10/util/Optional.h>
// 包含 c10 工具库中的 Optional 头文件

namespace c10 {

/// RAII guard that sets a certain default device in its constructor, and
/// changes it back to the device that was originally active upon destruction.
///
/// The device is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_device` after construction, the
/// destructor will still reset the device to the one that was active at
/// construction time.
///
/// This device guard does NOT have an uninitialized state; it is guaranteed
/// to reset a device on exit.  If you are in a situation where you *might*
/// want to setup a guard (i.e., are looking for the moral equivalent
/// of optional<DeviceGuard>), see OptionalDeviceGuard.
class DeviceGuard {
 public:
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit DeviceGuard() = delete;
  // 显式删除默认构造函数，避免无意义的默认初始化

  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device) : guard_(device) {}
  // 构造函数，设置当前设备为传入的 Device

  /// This constructor is for testing only.
  explicit DeviceGuard(
      Device device,
      const impl::DeviceGuardImplInterface* impl)
      : guard_(device, impl) {}
  // 用于测试的构造函数，传入 Device 和 DeviceGuardImplInterface 指针

  /// Copy is disallowed
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  // 禁止拷贝构造和拷贝赋值，确保唯一性和安全性

  /// Move is disallowed, as DeviceGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  DeviceGuard(DeviceGuard&& other) = delete;
  DeviceGuard& operator=(DeviceGuard&& other) = delete;
  // 禁止移动构造和移动赋值，因为 DeviceGuard 没有未初始化的状态

  /// Sets the device to the given one.  The specified device must be consistent
  /// with the device type originally specified during guard construction.
  ///
  /// TODO: The consistency check here is inconsistent with StreamGuard's
  /// behavior with set_stream, where a stream on a different device than
  /// the original one isn't an error; we just reset the stream and then
  /// switch devices.
  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }
  // 设置当前设备为指定的设备，要求指定设备与构造 guard 时的设备类型一致

  /// This method is for testing only.
  void reset_device(
      at::Device device,
      const impl::DeviceGuardImplInterface* impl) {
    guard_.reset_device(device, impl);
  }
  // 仅用于测试的方法，设置当前设备为指定的设备，并传入 DeviceGuardImplInterface 指针

  /// Sets the device index to the given one.  The device type is inferred
  /// from the original device type the guard was constructed with.
  void set_index(DeviceIndex index) {
    guard_.set_index(index);
  }
  // 设置设备索引为指定值，设备类型由构造 guard 时的原始设备类型推断而来

  /// Returns the device that was set at the time the guard was constructed.
  Device original_device() const {
    return guard_.original_device();
  }
  // 返回在构造 guard 时设置的设备

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device.
  Device current_device() const {
    return guard_.current_device();
  }



// 返回当前设备的函数调用结果
    return guard_.current_device();
  }



 private:
  impl::InlineDeviceGuard<impl::VirtualGuardImpl> guard_;



// 私有成员变量，使用 InlineDeviceGuard 模板类创建 guard_ 对象，
// 该对象的实现基于 VirtualGuardImpl 类
 private:
  impl::InlineDeviceGuard<impl::VirtualGuardImpl> guard_;
/**
 * A OptionalDeviceGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 * Morally, a OptionalDeviceGuard is equivalent to optional<DeviceGuard>, but
 * with extra constructors and methods as appropriate.
 */
class OptionalDeviceGuard {
public:
    /**
     * Construct an uninitialized OptionalDeviceGuard.
     */
    OptionalDeviceGuard();

    /**
     * Construct an OptionalDeviceGuard that sets the device to `device` on initialization.
     * @param device The device to set.
     */
    OptionalDeviceGuard(Device device);

    /**
     * Construct an OptionalDeviceGuard from a std::nullopt, leaving it uninitialized.
     * @param nullopt Represents an uninitialized state.
     */
    OptionalDeviceGuard(std::nullopt_t nullopt);

    /**
     * Set the device to a new value, resetting the previously set device if any.
     * @param device The device to set.
     */
    void set_device(Device device);

    /**
     * Reset the device to its original value.
     */
    void reset_device();

    /**
     * Return the original device value wrapped in optional.
     * @return Optional containing the original device.
     */
    std::optional<Device> original_device() const;

    /**
     * Return the current device value wrapped in optional.
     * @return Optional containing the current device.
     */
    std::optional<Device> current_device() const;

    /**
     * Destructor that resets the device to its original value if initialized.
     */
    ~OptionalDeviceGuard();
};
class OptionalDeviceGuard {
 public:
  /// Create an uninitialized guard.  Set the guard later using reset_device.
  explicit OptionalDeviceGuard() = default;

  /// Initialize the guard, setting the current device to the passed Device.
  explicit OptionalDeviceGuard(Device device) : guard_(device) {}

  /// Initialize the guard if a Device is passed; otherwise leave the
  /// guard uninitialized.
  explicit OptionalDeviceGuard(optional<Device> device) : guard_(device) {}

  /// Constructor for testing only.
  explicit OptionalDeviceGuard(
      Device device,
      const impl::DeviceGuardImplInterface* impl)
      : guard_(device, impl) {}

  /// Copy is disallowed
  OptionalDeviceGuard(const OptionalDeviceGuard&) = delete;
  OptionalDeviceGuard& operator=(const OptionalDeviceGuard&) = delete;

  /// Move is disallowed
  /// See Note [Explicit initialization of optional fields]
  /// and // Note [Move construction for RAII guards is tricky]
  /// for rationale.
  OptionalDeviceGuard(OptionalDeviceGuard&& other) = delete;
  OptionalDeviceGuard& operator=(OptionalDeviceGuard&& other) = delete;

  /// Sets the device to the given one.  The specified device must be consistent
  /// with the device type originally specified during guard construction.
  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  /// For testing only
  void reset_device(
      at::Device device,
      const impl::DeviceGuardImplInterface* impl) {
    guard_.reset_device(device, impl);
  }

  /// Returns the device that was set at the time the guard was constructed.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via reset_device.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

 private:
  impl::InlineOptionalDeviceGuard<impl::VirtualGuardImpl> guard_{};
};

// Note [Whither the DeviceGuard boilerplate]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Design note: in principle, we could avoid these wrappers using:
//
// using DeviceGuard = impl::InlineDeviceGuard<impl::VirtualGuardImpl>;
// using OptionalDeviceGuard =
// impl::InlineOptionalDeviceGuard<impl::VirtualGuardImpl>;
//
// But the error messages are worse, and our users can't just look at the
// header file to find out what's going on.  Furthermore, for specializations
// like CUDAStreamGuard, it can be profitable to replace some interfaces with
// refined types (e.g., return CUDAStream instead of Stream).  So, we eat
// the boilerplate and write out the API explicitly.

} // namespace c10
```