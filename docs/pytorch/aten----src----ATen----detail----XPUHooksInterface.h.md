# `.\pytorch\aten\src\ATen\detail\XPUHooksInterface.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <ATen/core/Generator.h>
#include <c10/util/Registry.h>
// 引入所需的头文件

namespace at {
// 命名空间开始

constexpr const char* XPU_HELP =
    "The XPU backend requires Intel Extension for Pytorch;"
    "this error has occurred because you are trying "
    "to use some XPU's functionality, but the Intel Extension for Pytorch has not been "
    "loaded for some reason. The Intel Extension for Pytorch MUST "
    "be loaded, EVEN IF you don't directly use any symbols from that!";
// XPU_HELP 字符串常量，提供关于 XPU 后端的帮助信息

struct TORCH_API XPUHooksInterface {
  virtual ~XPUHooksInterface() = default;
  // XPUHooksInterface 结构，定义 XPU 钩子接口

  virtual void initXPU() const {
    // 虚函数，初始化 XPU，如果未加载 Intel Extension for Pytorch 则抛出异常
    TORCH_CHECK(
        false,
        "Cannot initialize XPU without Intel Extension for Pytorch.",
        XPU_HELP);
  }

  virtual bool hasXPU() const {
    // 虚函数，判断是否存在 XPU，始终返回 false
    return false;
  }

  virtual std::string showConfig() const {
    // 虚函数，展示 XPU 配置信息，如果未加载 Intel Extension for Pytorch 则抛出异常
    TORCH_CHECK(
        false,
        "Cannot query detailed XPU version without Intel Extension for Pytorch. ",
        XPU_HELP);
  }

  virtual int32_t getGlobalIdxFromDevice(const Device& device) const {
    // 虚函数，从设备获取 XPU 全局索引，如果未加载 ATen_xpu 库则抛出异常
    TORCH_CHECK(false, "Cannot get XPU global device index without ATen_xpu library.");
  }

  virtual Generator getXPUGenerator(C10_UNUSED DeviceIndex device_index = -1) const {
    // 虚函数，获取 XPU 生成器，如果未加载 Intel Extension for Pytorch 则抛出异常
    TORCH_CHECK(false, "Cannot get XPU generator without Intel Extension for Pytorch. ", XPU_HELP);
  }

  virtual const Generator& getDefaultXPUGenerator(C10_UNUSED DeviceIndex device_index = -1) const {
    // 虚函数，获取默认 XPU 生成器，如果未加载 Intel Extension for Pytorch 则抛出异常
    TORCH_CHECK(false, "Cannot get default XPU generator without Intel Extension for Pytorch. ", XPU_HELP);
  }

  virtual DeviceIndex getNumGPUs() const {
    // 虚函数，获取 GPU 数量，始终返回 0
    return 0;
  }

  virtual DeviceIndex current_device() const {
    // 虚函数，获取当前 XPU 设备的索引，如果未加载 ATen_xpu 库则抛出异常
    TORCH_CHECK(false, "Cannot get current device on XPU without ATen_xpu library.");
  }

  virtual Device getDeviceFromPtr(void* /*data*/) const {
    // 虚函数，根据指针获取设备，如果未加载 ATen_xpu 库则抛出异常
    TORCH_CHECK(false, "Cannot get device of pointer on XPU without ATen_xpu library.");
  }

  virtual void deviceSynchronize(DeviceIndex /*device_index*/) const {
    // 虚函数，同步 XPU 设备，如果未加载 ATen_xpu 库则抛出异常
    TORCH_CHECK(false, "Cannot synchronize XPU device without ATen_xpu library.");
  }

  virtual Allocator* getPinnedMemoryAllocator() const  {
    // 虚函数，获取 XPU 固定内存分配器，如果未加载 ATen_xpu 库则抛出异常
    TORCH_CHECK(false, "Cannot get XPU pinned memory allocator without ATen_xpu library.");
  }

  virtual bool isPinnedPtr(const void* /*data*/) const {
    // 虚函数，检查指针是否为固定内存指针，始终返回 false
    return false;
  }
};

struct TORCH_API XPUHooksArgs {};
// XPUHooksArgs 结构，定义 XPU 钩子参数

C10_DECLARE_REGISTRY(XPUHooksRegistry, XPUHooksInterface, XPUHooksArgs);
// 定义 XPUHooksRegistry 注册表

#define REGISTER_XPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(XPUHooksRegistry, clsname, clsname)
// 定义注册 XPU 钩子的宏

namespace detail {
TORCH_API const XPUHooksInterface& getXPUHooks();
} // namespace detail

} // namespace at
// 命名空间结束
```