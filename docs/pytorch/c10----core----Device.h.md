# `.\pytorch\c10\core\Device.h`

```
#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>

namespace c10 {

/// An index representing a specific device; e.g., the 1 in GPU 1.
/// A DeviceIndex is not independently meaningful without knowing
/// the DeviceType it is associated; try to use Device rather than
/// DeviceIndex directly.
using DeviceIndex = int8_t;

/// Represents a compute device on which a tensor is located. A device is
/// uniquely identified by a type, which specifies the type of machine it is
/// (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the
/// specific compute device when there is more than one of a certain type. The
/// device index is optional, and in its defaulted state represents (abstractly)
/// "the current device". Further, there are two constraints on the value of the
/// device index, if one is explicitly stored:
/// 1. A negative index represents the current device, a non-negative index
/// represents a specific, concrete device,
/// 2. When the device type is CPU, the device index must be zero.
struct C10_API Device final {
  using Type = DeviceType;

  /// Constructs a new `Device` from a `DeviceType` and an optional device
  /// index.
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {
    validate();  // 调用 validate() 函数确保设备类型和索引的有效性
  }

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  /// Returns true if the type and index of this `Device` matches that of
  /// `other`.
  bool operator==(const Device& other) const noexcept {
    return this->type_ == other.type_ && this->index_ == other.index_;
  }

  /// Returns true if the type or index of this `Device` differs from that of
  /// `other`.
  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  /// Sets the device index.
  void set_index(DeviceIndex index) {
    index_ = index;
  }

  /// Returns the type of device this is.
  DeviceType type() const noexcept {
    return type_;
  }

  /// Returns the optional index.
  DeviceIndex index() const noexcept {
    return index_;
  }

  /// Returns true if the device has a non-default index.
  bool has_index() const noexcept {
    return index_ != -1;
  }

  /// Return true if the device is of CUDA type.
  bool is_cuda() const noexcept {
    return type_ == DeviceType::CUDA;
  }

  /// Return true if the device is of PrivateUse1 type.
  bool is_privateuseone() const noexcept {
    return type_ == DeviceType::PrivateUse1;
  }

  /// Return true if the device is of MPS type.
  bool is_mps() const noexcept {
  /// Return true if the device is of MPS type.
  bool is_mps() const noexcept {
    return type_ == DeviceType::MPS;
  }

  /// Return true if the device is of HIP type.
  bool is_hip() const noexcept {
    return type_ == DeviceType::HIP;
  }

  /// Return true if the device is of VE type.
  bool is_ve() const noexcept {
    return type_ == DeviceType::VE;
  }

  /// Return true if the device is of XPU type.
  bool is_xpu() const noexcept {
    return type_ == DeviceType::XPU;
  }

  /// Return true if the device is of IPU type.
  bool is_ipu() const noexcept {
    return type_ == DeviceType::IPU;
  }

  /// Return true if the device is of XLA type.
  bool is_xla() const noexcept {
    return type_ == DeviceType::XLA;
  }

  /// Return true if the device is of MTIA type.
  bool is_mtia() const noexcept {
    return type_ == DeviceType::MTIA;
  }

  /// Return true if the device is of HPU type.
  bool is_hpu() const noexcept {
    return type_ == DeviceType::HPU;
  }

  /// Return true if the device is of Lazy type.
  bool is_lazy() const noexcept {
    return type_ == DeviceType::Lazy;
  }

  /// Return true if the device is of Vulkan type.
  bool is_vulkan() const noexcept {
    return type_ == DeviceType::Vulkan;
  }

  /// Return true if the device is of Metal type.
  bool is_metal() const noexcept {
    return type_ == DeviceType::Metal;
  }

  /// Return true if the device is of MAIA type.
  bool is_maia() const noexcept {
    return type_ == DeviceType::MAIA;
  }

  /// Return true if the device is of META type.
  bool is_meta() const noexcept {
    return type_ == DeviceType::Meta;
  }

  /// Return true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

  /// Return true if the device supports arbitrary strides.
  bool supports_as_strided() const noexcept {
    // Check if the device type is not one of IPU, XLA, Lazy, or MTIA
    // to determine if arbitrary strides are supported.
    return type_ != DeviceType::IPU && type_ != DeviceType::XLA &&
        type_ != DeviceType::Lazy && type_ != DeviceType::MTIA;
  }

  /// Same string as returned from operator<<.
  std::string str() const;

 private:
  DeviceType type_; // 设备类型
  DeviceIndex index_ = -1; // 设备索引，默认为-1
  void validate() {
    // 在发布版本中去除这些检查会显著提高微基准测试的性能。
    // 这样做是安全的，因为在尝试切换到设备时，使用设备索引的后端会进行后续检查。
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        index_ >= -1,
        "Device index must be -1 or non-negative, got ",
        static_cast<int>(index_));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !is_cpu() || index_ <= 0,
        "CPU device index must be -1 or zero, got ",
        static_cast<int>(index_));
  }
};

// 声明重载流插入运算符，用于将 Device 对象输出到流中
C10_API std::ostream& operator<<(std::ostream& stream, const Device& device);

} // namespace c10

// 声明 std 命名空间下的模板特化：hash<c10::Device>
namespace std {
template <>
struct hash<c10::Device> {
  // 重载函数调用操作符，计算并返回 Device 对象的哈希值
  size_t operator()(c10::Device d) const noexcept {
    // 如果此处出现编译错误，请确保按照新的位掩码代码更新!
    // 静态断言，检查 DeviceType 的大小是否为 1 字节
    static_assert(sizeof(c10::DeviceType) == 1, "DeviceType is not 8-bit");
    // 静态断言，检查 DeviceIndex 的大小是否为 1 字节
    static_assert(sizeof(c10::DeviceIndex) == 1, "DeviceIndex is not 8-bit");

    // 注意 [拼接有符号整数时的风险]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 我们必须先转换为相同大小的无符号类型，然后再提升到结果类型，
    // 以防止当任何值为 -1 时发生符号扩展。
    // 如果发生符号扩展，将覆盖结果整数的 MSB 部分的所有值。
    //
    // 从 C/C++ 整数提升规则来看，实际上只需要一个 uint32_t 强制转换到结果类型，
    // 但出于明确性考虑，我们两个都使用了。
    uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type())) << 16 |
                    static_cast<uint32_t>(static_cast<uint8_t>(d.index()));

    // 返回 bits 的哈希值
    return std::hash<uint32_t>{}(bits);
  }
};
} // namespace std
```