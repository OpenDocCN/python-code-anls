# `.\pytorch\aten\src\ATen\detail\PrivateUse1HooksInterface.h`

```py
#pragma once

#include <ATen/core/Generator.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>

namespace at {

// 定义了一个名为 PrivateUse1HooksInterface 的结构体，继承自 AcceleratorHooksInterface
struct TORCH_API PrivateUse1HooksInterface : AcceleratorHooksInterface {
  // 析构函数，用于释放资源
  ~PrivateUse1HooksInterface() override = default;

  // 获取指定设备索引下的默认生成器对象的虚拟函数
  virtual const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
    // 抛出异常，表示函数未实现
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDefaultGenerator`.");
  }

  // 根据指针获取存储设备的虚拟函数
  virtual at::Device getDeviceFromPtr(void* data) const {
    // 抛出异常，表示函数未实现
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDeviceFromPtr`.");
  }

  // 获取固定内存分配器的虚拟函数
  virtual Allocator* getPinnedMemoryAllocator() const {
    // 抛出异常，表示函数未实现
    TORCH_CHECK(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getPinnedMemoryAllocator`.");
  }

  // 检查指定设备索引是否具有主要上下文的虚拟函数
  bool hasPrimaryContext(DeviceIndex device_index) const override {
    // 抛出异常，表示函数未实现
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `hasPrimaryContext`.");
  }

  // 初始化 PrivateUse1 的虚拟函数
  virtual void initPrivateUse1() const {}

  // 调整 PrivateUse1 存储的字节大小的虚拟函数
  virtual void resizePrivateUse1Bytes(const c10::Storage &storage, size_t newsize) const {
    // 抛出异常，表示函数未实现
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `resizePrivateUse1Bytes`.");
  }
};

// PrivateUse1HooksArgs 结构体的定义
struct TORCH_API PrivateUse1HooksArgs {};

// 注册 PrivateUse1HooksInterface 的全局函数声明
TORCH_API void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_);

// 获取 PrivateUse1HooksInterface 对象的全局函数声明
TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface();

// 检查 PrivateUse1HooksInterface 是否已注册的全局函数声明
TORCH_API bool isPrivateUse1HooksRegistered();

namespace detail {

// 获取 PrivateUse1HooksInterface 的详细信息的命名空间声明
TORCH_API const at::PrivateUse1HooksInterface& getPrivateUse1Hooks();

} // namespace detail

} // namespace at
```