# `.\pytorch\aten\src\ATen\mps\MPSHooks.h`

```
#pragma once

// 预处理指令，确保头文件只被包含一次，以防止重复定义错误


#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/Generator.h>
#include <ATen/mps/MPSEvent.h>
#include <c10/util/Optional.h>

// 包含所需的头文件，用于定义和声明本文件中的类和函数


namespace at::mps {

// 命名空间的开始，定义了命名空间 `at::mps`


struct MPSHooks : public at::MPSHooksInterface {

// 定义了 `MPSHooks` 结构体，继承自 `at::MPSHooksInterface`


MPSHooks(at::MPSHooksArgs) {}

// `MPSHooks` 结构体的构造函数，接受 `at::MPSHooksArgs` 类型的参数，未提供具体实现


void initMPS() const override;

// 实现了 `MPSHooksInterface` 的 `initMPS` 函数，用于初始化 MPS（Metal Performance Shaders）


bool hasMPS() const override;

// 实现了 `MPSHooksInterface` 的 `hasMPS` 函数，用于检查当前环境是否支持 MPS


bool isOnMacOSorNewer(unsigned major, unsigned minor) const override;

// 实现了 `MPSHooksInterface` 的 `isOnMacOSorNewer` 函数，用于检查当前操作系统是否为指定版本的 macOS 或更新版本


const Generator& getDefaultMPSGenerator() const override;

// 实现了 `MPSHooksInterface` 的 `getDefaultMPSGenerator` 函数，返回默认的 MPS 生成器对象的引用


void deviceSynchronize() const override;

// 实现了 `MPSHooksInterface` 的 `deviceSynchronize` 函数，用于设备同步操作


void commitStream() const override;

// 实现了 `MPSHooksInterface` 的 `commitStream` 函数，提交流操作


void* getCommandBuffer() const override;

// 实现了 `MPSHooksInterface` 的 `getCommandBuffer` 函数，返回命令缓冲区的指针


void* getDispatchQueue() const override;

// 实现了 `MPSHooksInterface` 的 `getDispatchQueue` 函数，返回分发队列的指针


Allocator* getMPSDeviceAllocator() const override;

// 实现了 `MPSHooksInterface` 的 `getMPSDeviceAllocator` 函数，返回 MPS 设备分配器的指针


void emptyCache() const override;

// 实现了 `MPSHooksInterface` 的 `emptyCache` 函数，清空缓存操作


size_t getCurrentAllocatedMemory() const override;

// 实现了 `MPSHooksInterface` 的 `getCurrentAllocatedMemory` 函数，返回当前已分配内存的大小


size_t getDriverAllocatedMemory() const override;

// 实现了 `MPSHooksInterface` 的 `getDriverAllocatedMemory` 函数，返回驱动程序已分配内存的大小


size_t getRecommendedMaxMemory() const override;

// 实现了 `MPSHooksInterface` 的 `getRecommendedMaxMemory` 函数，返回推荐的最大内存大小


void setMemoryFraction(double ratio) const override;

// 实现了 `MPSHooksInterface` 的 `setMemoryFraction` 函数，设置内存分配比例


void profilerStartTrace(const std::string& mode, bool waitUntilCompleted) const override;

// 实现了 `MPSHooksInterface` 的 `profilerStartTrace` 函数，启动性能分析追踪


void profilerStopTrace() const override;

// 实现了 `MPSHooksInterface` 的 `profilerStopTrace` 函数，停止性能分析追踪


uint32_t acquireEvent(bool enable_timing) const override;

// 实现了 `MPSHooksInterface` 的 `acquireEvent` 函数，获取事件对象


void releaseEvent(uint32_t event_id) const override;

// 实现了 `MPSHooksInterface` 的 `releaseEvent` 函数，释放事件对象


void recordEvent(uint32_t event_id) const override;

// 实现了 `MPSHooksInterface` 的 `recordEvent` 函数，记录事件操作


void waitForEvent(uint32_t event_id) const override;

// 实现了 `MPSHooksInterface` 的 `waitForEvent` 函数，等待事件完成


void synchronizeEvent(uint32_t event_id) const override;

// 实现了 `MPSHooksInterface` 的 `synchronizeEvent` 函数，同步事件操作


bool queryEvent(uint32_t event_id) const override;

// 实现了 `MPSHooksInterface` 的 `queryEvent` 函数，查询事件状态


double elapsedTimeOfEvents(uint32_t start_event_id, uint32_t end_event_id) const override;

// 实现了 `MPSHooksInterface` 的 `elapsedTimeOfEvents` 函数，计算两个事件之间的时间间隔


bool hasPrimaryContext(DeviceIndex device_index) const override {

// 实现了 `MPSHooksInterface` 的 `hasPrimaryContext` 函数，检查主要上下文是否存在于指定设备上


// When MPS is available, it is always in use for the one device.
return true;

// 当 MPS 可用时，它总是在使用中的唯一设备上。


} // namespace at::mps

// 命名空间结束，结束了命名空间 `at::mps`
```