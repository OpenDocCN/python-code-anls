# `.\pytorch\torch\csrc\lazy\backend\backend_device.cpp`

```
// 引入 Torch Lazy 模块的设备后端相关头文件
#include <torch/csrc/lazy/backend/backend_device.h>

// 引入 C10 核心库中的设备、异常、可选项和字符串工具相关头文件
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/StringUtil.h>

// 引入 Torch Lazy 模块的设备后端接口和张量相关头文件
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/tensor.h>

// Torch Lazy 命名空间
namespace torch {
namespace lazy {

// BackendDevice 类的默认构造函数，初始化设备类型和序数
BackendDevice::BackendDevice()
    : type_(getBackend()->GetDefaultDeviceType()),  // 获取默认设备类型
      ordinal_(getBackend()->GetDefaultDeviceOrdinal()) {}  // 获取默认设备序数

// BackendDevice 类的带参数构造函数，接受设备类型和序数
BackendDevice::BackendDevice(
    std::shared_ptr<BackendDeviceType>&& type,
    int64_t ordinal)
    : type_(std::move(type)), ordinal_(ordinal) {}

// 返回设备类型的整数表示
int8_t BackendDevice::type() const {
  TORCH_INTERNAL_ASSERT(type_);  // 断言设备类型非空
  return type_->type;  // 返回设备类型的整数表示
}

// 返回设备的字符串表示，包括类型和序数
std::string BackendDevice::toString() const {
  TORCH_INTERNAL_ASSERT(type_);  // 断言设备类型非空
  return c10::str(type_->toString(), ordinal_);  // 返回设备类型和序数的字符串表示
}

// 比较两个 BackendDevice 对象的大小
int BackendDevice::compare(const BackendDevice& rhs) const {
  if (type() != rhs.type()) {  // 比较设备类型
    return type() < rhs.type() ? -1 : +1;  // 返回比较结果
  }
  return ordinal_ < rhs.ordinal_ ? -1 : (ordinal_ > rhs.ordinal_ ? +1 : 0);  // 比较设备序数
}

// BackendDevice 对象的输出流操作符重载，输出设备的字符串表示
std::ostream& operator<<(std::ostream& os, const BackendDevice& device) {
  os << device.toString();  // 输出设备的字符串表示
  return os;
}

// 将 ATen 的 Device 转换为 Torch Lazy 的 BackendDevice
BackendDevice atenDeviceToBackendDevice(const c10::Device& device) {
  TORCH_CHECK(device.type() == at::kLazy, device);  // 检查设备类型是否为 Lazy
  int64_t ordinal = device.has_index()
      ? device.index()
      : getBackend()->GetDefaultDeviceOrdinal();  // 获取设备序数或默认序数
  return BackendDevice(getBackend()->GetDefaultDeviceType(), ordinal);  // 返回对应的 BackendDevice
}

// 将 Torch Lazy 的 BackendDevice 转换为 ATen 的 Device
c10::Device backendDeviceToAtenDevice(const BackendDevice& device) {
  return c10::Device(at::kLazy, device.ordinal());  // 构造并返回对应的 ATen 设备对象
}

// 获取一组张量的后端设备，返回第一个能获取到的设备或空的可选项
std::optional<BackendDevice> GetBackendDevice(at::ITensorListRef tensors) {
  for (auto& tensor : tensors) {  // 遍历张量列表
    if (auto lt = TryGetLtcTensor(tensor)) {  // 尝试获取 Lazy 张量
      return lt->GetDevice();  // 返回 Lazy 张量的设备
    }
  }
  return c10::nullopt;  // 返回空的可选项
}

// 获取一组张量的后端设备，返回第一个能获取到的设备或空的可选项
std::optional<BackendDevice> GetBackendDevice(at::TensorList tensors) {
  return GetBackendDevice(at::ITensorListRef(tensors));  // 调用上述函数获取后端设备
}

// 获取单个张量的后端设备，返回能获取到的设备或空的可选项
std::optional<BackendDevice> GetBackendDevice(const at::Tensor& tensor) {
  if (auto lt = TryGetLtcTensor(tensor)) {  // 尝试获取 Lazy 张量
    return lt->GetDevice();  // 返回 Lazy 张量的设备
  }
  return c10::nullopt;  // 返回空的可选项
}

// 将 ATen 的 Device 可选项转换为 Torch Lazy 的 BackendDevice 可选项
std::optional<BackendDevice> GetBackendDevice(
    const std::optional<c10::Device>& device) {
  if (device) {  // 如果设备可选项有值
    return c10::make_optional(atenDeviceToBackendDevice(*device));  // 转换并返回对应的 BackendDevice
  }
  return c10::nullopt;  // 返回空的可选项
}

// 返回空的 Torch Lazy 的 BackendDevice 可选项
std::optional<BackendDevice> GetBackendDevice() {
  return c10::nullopt;  // 返回空的可选项
}

} // namespace lazy
} // namespace torch
```