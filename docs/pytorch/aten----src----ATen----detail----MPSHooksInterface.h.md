# `.\pytorch\aten\src\ATen\detail\MPSHooksInterface.h`

```
//  Copyright © 2022 Apple Inc.

#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <cstddef>

namespace at {

// 定义 MPSHooksInterface 结构体，继承自 AcceleratorHooksInterface
struct TORCH_API MPSHooksInterface : AcceleratorHooksInterface {
  // 定义宏 FAIL_MPSHOOKS_FUNC，用于抛出错误消息，指示调用 MPSHooksInterface 的函数需要 MPS 后端支持
  #define FAIL_MPSHOOKS_FUNC(func) \
    TORCH_CHECK(false, "Cannot execute ", func, "() without MPS backend.");

  // 析构函数，默认实现
  ~MPSHooksInterface() override = default;

  // 初始化 MPS 库状态，如果调用则抛出错误
  virtual void initMPS() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 检查是否存在 MPS 后端支持，始终返回 false
  virtual bool hasMPS() const {
    return false;
  }

  // 检查是否在 macOS 或更新版本上运行，始终抛出错误
  virtual bool isOnMacOSorNewer(unsigned major = 13, unsigned minor = 0) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取默认的 MPS 生成器，始终抛出错误
  virtual const Generator& getDefaultMPSGenerator() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取 MPS 设备分配器，始终抛出错误
  virtual Allocator* getMPSDeviceAllocator() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 同步设备，始终抛出错误
  virtual void deviceSynchronize() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 提交流，始终抛出错误
  virtual void commitStream() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取命令缓冲区，始终抛出错误
  virtual void* getCommandBuffer() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取调度队列，始终抛出错误
  virtual void* getDispatchQueue() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 清空缓存，始终抛出错误
  virtual void emptyCache() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取当前已分配的内存大小，始终抛出错误
  virtual size_t getCurrentAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取驱动程序已分配的内存大小，始终抛出错误
  virtual size_t getDriverAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取推荐的最大内存大小，始终抛出错误
  virtual size_t getRecommendedMaxMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 设置内存分数，始终抛出错误
  virtual void setMemoryFraction(double /*ratio*/) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 启动分析器追踪，始终抛出错误
  virtual void profilerStartTrace(const std::string& mode, bool waitUntilCompleted) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 停止分析器追踪，始终抛出错误
  virtual void profilerStopTrace() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 获取事件，始终抛出错误
  virtual uint32_t acquireEvent(bool enable_timing) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 释放事件，始终抛出错误
  virtual void releaseEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 记录事件，始终抛出错误
  virtual void recordEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 等待事件，始终抛出错误
  virtual void waitForEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 同步事件，始终抛出错误
  virtual void synchronizeEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 查询事件，始终抛出错误
  virtual bool queryEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 计算两个事件之间的时间，始终抛出错误
  virtual double elapsedTimeOfEvents(uint32_t start_event_id, uint32_t end_event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 检查是否有主要上下文，始终抛出错误
  bool hasPrimaryContext(DeviceIndex device_index) const override {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  // 取消定义宏 FAIL_MPSHOOKS_FUNC
  #undef FAIL_MPSHOOKS_FUNC
};

// 定义 MPSHooksArgs 结构体
struct TORCH_API MPSHooksArgs {};

// 声明 MPSHooksRegistry，用于注册 MPSHooksInterface 和 MPSHooksArgs
TORCH_DECLARE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs);

} // namespace at
#define REGISTER_MPS_HOOKS(clsname) \  // 定义一个宏，用于注册 MPS 钩子的类
  C10_REGISTER_CLASS(MPSHooksRegistry, clsname, clsname)

namespace detail {  // 进入命名空间 detail
TORCH_API const MPSHooksInterface& getMPSHooks();  // 声明一个函数 getMPSHooks，返回一个常引用到 MPSHooksInterface 类的对象

} // namespace detail  // 退出命名空间 detail
} // namespace at  // 退出命名空间 at
```