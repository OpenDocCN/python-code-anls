# `.\pytorch\aten\src\ATen\core\DeprecatedTypeProperties.h`

```
#pragma once
// 防止头文件被多次包含

#include <c10/core/Backend.h>
// 引入 Backend 相关定义

#include <c10/core/ScalarType.h>
// 引入 ScalarType 相关定义

#include <c10/core/Layout.h>
// 引入 Layout 相关定义

#include <c10/core/TensorOptions.h>
// 引入 TensorOptions 相关定义

#include <c10/core/Storage.h>
// 引入 Storage 相关定义

#include <ATen/core/DeprecatedTypePropertiesRegistry.h>
// 引入 DeprecatedTypePropertiesRegistry 相关定义

#include <ATen/core/Generator.h>
// 引入 Generator 相关定义

namespace at {

class Tensor;
// 前置声明 Tensor 类，表示在此处引入 Tensor 类的定义

// This class specifies a Backend and a ScalarType. Currently, it primarily
// serves as a replacement return value for Tensor::type(). Previously,
// Tensor::type() returned Type&, but we are changing Type to not be
// dtype-specific.
class TORCH_API DeprecatedTypeProperties {
 public:
  DeprecatedTypeProperties(Backend backend, ScalarType scalar_type)
    : backend_(backend), scalar_type_(scalar_type) {}
  // 构造函数，接受 Backend 和 ScalarType 作为参数，初始化成员变量

  Backend backend() const {
    return backend_;
  }
  // 返回当前实例的 Backend 成员变量

  Layout layout() const {
    return layout_from_backend(backend_);
  }
  // 根据 Backend 获取相应的 Layout

  bool is_sparse() const {
    return layout_from_backend(backend()) == kSparse;
  }
  // 检查当前实例是否为稀疏张量

  bool is_sparse_csr() const {
    return layout_from_backend(backend()) == kSparseCsr;
  }
  // 检查当前实例是否为 CSR 格式的稀疏张量

  c10::DeviceType device_type() const {
    return backendToDeviceType(backend_);
  }
  // 返回当前实例的 DeviceType

  bool is_cuda() const {
    return backendToDeviceType(backend_) == kCUDA;
  }
  // 检查当前实例是否在 CUDA 设备上

  ScalarType scalarType() const {
    return scalar_type_;
  }
  // 返回当前实例的 ScalarType

  caffe2::TypeMeta typeMeta() const {
    return scalarTypeToTypeMeta(scalar_type_);
  }
  // 返回当前实例的 TypeMeta

  bool operator==(const DeprecatedTypeProperties& other) const {
    return backend_ == other.backend() && scalar_type_ == other.scalarType();
  }
  // 比较两个 DeprecatedTypeProperties 实例是否相等

  bool operator!=(const DeprecatedTypeProperties& other) const {
    return !(*this == other);
  }
  // 比较两个 DeprecatedTypeProperties 实例是否不相等

  std::string toString() const {
    std::string base_str;
    if (backend_ == Backend::Undefined || scalar_type_ == ScalarType::Undefined) {
      base_str = "UndefinedType";
    } else {
      base_str = std::string(at::toString(backend_)) + at::toString(scalar_type_) + "Type";
    }
    return base_str;
  }
  // 返回描述当前实例的字符串表示形式

  DeprecatedTypeProperties & toBackend(Backend b) const {
    return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
        b, scalar_type_);
  }
  // 返回指定 Backend 的 DeprecatedTypeProperties 实例

  DeprecatedTypeProperties & toScalarType(ScalarType s) const {
    return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
        backend_, s);
  }
  // 返回指定 ScalarType 的 DeprecatedTypeProperties 实例

  DeprecatedTypeProperties & cpu() const {
    return toBackend(Backend::CPU);
  }
  // 返回 CPU Backend 的 DeprecatedTypeProperties 实例

  DeprecatedTypeProperties & cuda() const {
    return toBackend(Backend::CUDA);
  }
  // 返回 CUDA Backend 的 DeprecatedTypeProperties 实例

  DeprecatedTypeProperties & hip() const {
    return toBackend(Backend::HIP);
  }
  // 返回 HIP Backend 的 DeprecatedTypeProperties 实例

  DeprecatedTypeProperties & privateUser1() const {
    return toBackend(Backend::PrivateUse1);
  }
  // 返回 PrivateUse1 Backend 的 DeprecatedTypeProperties 实例

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  TensorOptions options(int16_t device_index = -1) const {
  // 构造 TensorOptions 对象，根据类型和设备索引
  /// 返回一个 `TensorOptions` 对象，配置包括默认的数据类型、设备类型和布局。
  return TensorOptions().dtype(typeMeta())
                        .device(device_type(), static_cast<c10::DeviceIndex>(device_index))
                        .layout(layout());
}

/// 根据可选的设备参数构造 `TensorOptions` 对象。如果设备参数未提供，则返回一个默认配置的对象。
/// 如果提供了设备参数，则根据该设备类型和索引构造对象，并断言设备类型与当前对象的设备类型匹配。
TensorOptions options(std::optional<Device> device_opt) const {
  if (!device_opt.has_value()) {
    // 如果未提供设备参数，返回默认配置的 `TensorOptions` 对象
    return options(-1);
  } else {
    Device device = device_opt.value();
    // 断言提供的设备类型与当前对象的设备类型匹配
    AT_ASSERT(device.type() == device_type());
    // 根据提供的设备类型索引构造 `TensorOptions` 对象
    return options(device.index());
  }
}

/// 将当前对象转换为 `TensorOptions` 类型的对象，调用 `options()` 方法获取配置。
operator TensorOptions() const {
  return options();
}

/// 返回当前对象的唯一标识符 `id`，计算方法为将后端类型乘以标量类型选项的数量再加上标量类型的值。
int64_t id() const {
  return static_cast<int64_t>(backend()) *
      static_cast<int64_t>(ScalarType::NumOptions) +
      static_cast<int64_t>(scalarType());
}

/// 从指针创建一个不安全的 `Tensor` 对象，可选择保留指针所指的数据。
Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const;

/// 从指针创建一个不安全的 `Storage` 对象，可选择保留指针所指的数据。
Storage unsafeStorageFromTH(void * th_pointer, bool retain) const;

/// 复制源张量到当前设备的张量，并可选择是否非阻塞地执行复制操作。
Tensor copy(const Tensor & src, bool non_blocking=false, std::optional<Device> to_device={}) const;

private:
Backend backend_;           // 后端类型
ScalarType scalar_type_;    // 标量类型
};

}  // namespace at
```