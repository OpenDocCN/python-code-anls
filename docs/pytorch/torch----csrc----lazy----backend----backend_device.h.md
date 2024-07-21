# `.\pytorch\torch\csrc\lazy\backend\backend_device.h`

```
#pragma once
// 预处理指令，确保此头文件只被编译一次

#include <memory>
// 包含内存管理相关的标准库头文件

#include <ostream>
// 包含输出流操作相关的标准库头文件

#include <string>
// 包含字符串处理相关的标准库头文件

#include <ATen/Tensor.h>
// 包含 PyTorch ATen 库中的 Tensor 类定义头文件

#include <c10/macros/Export.h>
// 包含 ATen 库中的导出宏定义头文件

#include <c10/util/Deprecated.h>
// 包含 ATen 库中的废弃功能宏定义头文件

#include <c10/util/Optional.h>
// 包含 ATen 库中的可选类型头文件

namespace c10 {
struct Device;
}
// 声明 c10 命名空间下的 Device 结构体

namespace torch {
namespace lazy {
// 声明 torch 命名空间下的 lazy 子命名空间

// 后端设备类型基类，其它后端应该继承并定义自己支持的硬件类型
struct TORCH_API BackendDeviceType {
  int8_t type{(int8_t)at::kCPU};
  // 设备类型，默认为 CPU

  // 构造函数，默认设备类型为 at::kCPU
  BackendDeviceType() : type((int8_t)at::kCPU) {}

  // 构造函数，根据给定类型初始化设备类型
  BackendDeviceType(int8_t type) : type(type) {}

  virtual ~BackendDeviceType() = default;
  // 虚析构函数，用于多态

  virtual std::string toString() const {
    return "Unknown";
    // 返回未知类型的字符串描述
  }
};

class TORCH_API BackendDevice {
 public:
  // 默认构造函数，设置设备类型和序数为后端特定的默认值
  BackendDevice();

  // 构造函数，根据给定的类型指针和序数初始化后端设备
  BackendDevice(std::shared_ptr<BackendDeviceType>&& type, int64_t ordinal);

  // 返回设备类型
  int8_t type() const;

  // 返回设备序数
  int64_t ordinal() const {
    return ordinal_;
    // 返回序数成员变量
  }

  // 比较运算符重载，比较设备是否相等
  bool operator==(const BackendDevice& other) const {
    return compare(other) == 0;
  }
  
  // 比较运算符重载，比较设备是否不相等
  bool operator!=(const BackendDevice& other) const {
    return compare(other) != 0;
  }

  // 比较运算符重载，比较设备的序数大小
  bool operator<(const BackendDevice& rhs) const {
    return compare(rhs) < 0;
  }

  // 返回设备的字符串描述
  std::string toString() const;

 private:
  // 比较函数，用于比较两个后端设备对象
  int compare(const BackendDevice& rhs) const;

  // 使用 shared_ptr 而不是 unique_ptr，以便 BackendDevice 可以被复制
  std::shared_ptr<BackendDeviceType> type_;
  // 后端设备类型指针

  int64_t ordinal_;
  // 设备序数
};

TORCH_API std::ostream& operator<<(
    std::ostream& os,
    const BackendDevice& device);
// 输出流操作符重载，用于输出后端设备对象

// 辅助函数，将 c10::Device 转换为 BackendDevice，反之亦然
TORCH_API BackendDevice atenDeviceToBackendDevice(const c10::Device& device);
TORCH_API c10::Device backendDeviceToAtenDevice(const BackendDevice& device);

// 尝试从 lazy tensor 中提取后端设备，如果输入不是 lazy tensor 则返回 nullopt
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const at::ITensorListRef tensors);
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const at::TensorList tensors);
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const at::Tensor& tensor);
TORCH_API std::optional<BackendDevice> GetBackendDevice(
    const std::optional<c10::Device>& device);

// 变长模板，用于获取多个 tensor 的后端设备
TORCH_API std::optional<BackendDevice> GetBackendDevice();

template <typename T, typename... Args>
std::optional<BackendDevice> GetBackendDevice(
    const T& tensor,
    const Args&... forward_tensors) {
  auto optional_device = GetBackendDevice(tensor);
  // 尝试获取当前 tensor 的后端设备

  if (optional_device) {
    return optional_device;
    // 如果成功获取到后端设备，则返回结果
  }
  return GetBackendDevice(forward_tensors...);
  // 递归调用，继续尝试获取下一个 tensor 的后端设备
}

} // namespace lazy
} // namespace torch
// 命名空间结束
```