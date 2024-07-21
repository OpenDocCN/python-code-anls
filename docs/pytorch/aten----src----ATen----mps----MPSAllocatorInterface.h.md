# `.\pytorch\aten\src\ATen\mps\MPSAllocatorInterface.h`

```py
//  Copyright © 2023 Apple Inc.  // 版权声明，指明此文件的版权信息

#pragma once  // 预处理器指令，确保本文件只被编译一次

#include <c10/core/Allocator.h>  // 引入 c10 核心 Allocator 头文件
#include <c10/util/Registry.h>  // 引入 c10 工具 Registry 头文件
#include <ATen/core/ATen_fwd.h>  // 引入 ATen 前向声明头文件

#define MB(x) (x * 1048576UL)  // 定义宏，将 x 换算为以字节为单位的兆字节数

namespace at::mps {

// 这是访问 MPSAllocator 的公共接口。
// 不要声明依赖于 MPS 或 Metal 框架的方法。
class IMPSAllocator : public c10::Allocator {
public:
  // 查看 MPSAllocator.h 中方法的描述注释。
  // 清空缓存
  virtual void emptyCache() const = 0;
  // 释放不活跃的缓冲区
  virtual void freeInactiveBuffers() const = 0;
  // 获取非对齐缓冲区的大小
  virtual ssize_t getUnalignedBufferSize(const void* ptr) const = 0;
  // 获取缓冲区形状
  virtual IntArrayRef getBufferShape(const void* ptr) const = 0;
  // 获取缓冲区 ID
  virtual id_t getBufferId(const void* ptr) const = 0;
  // 设置缓冲区形状
  virtual void setBufferShape(const void* ptr, const IntArrayRef& shape) const = 0;
  // 检查是否为共享缓冲区
  virtual bool isSharedBuffer(const void* ptr) const = 0;
  // 检查是否支持共享存储
  virtual bool isSharedStorageSupported() const = 0;
  // 分配一个值为 value 的标量缓冲区
  virtual c10::DataPtr allocScalarBufferWithValue(void* value, size_t size) const = 0;
  // 格式化大小
  virtual std::string formatSize(size_t size) const = 0;
  // 设置低水位比率
  virtual void setLowWatermarkRatio(double ratio) const = 0;
  // 设置高水位比率
  virtual void setHighWatermarkRatio(double ratio) const = 0;
  // 获取低水位值
  virtual ssize_t getLowWatermarkValue() const = 0;
  // 获取低水位限制
  virtual size_t getLowWatermarkLimit() const = 0;
  // 获取高水位限制
  virtual size_t getHighWatermarkLimit() const = 0;
  // 获取总分配内存
  virtual size_t getTotalAllocatedMemory() const = 0;
  // 获取当前分配内存
  virtual size_t getCurrentAllocatedMemory() const = 0;
  // 获取驱动程序分配的内存
  virtual size_t getDriverAllocatedMemory() const = 0;
  // 获取推荐的最大内存
  virtual size_t getRecommendedMaxMemory() const = 0;
  // 获取共享缓冲区指针和标记
  virtual std::pair<const void*, uint32_t> getSharedBufferPtr(const void* ptr) const = 0;
  // 记录事件
  virtual bool recordEvents(c10::ArrayRef<const void*> buffers) const = 0;
  // 等待事件
  virtual bool waitForEvents(c10::ArrayRef<const void*> buffers) const = 0;
};

class IMpsAllocatorCallback {
 public:
  // 事件类型枚举
  enum class EventType {
    ALLOCATED,  // 缓冲区被分配以立即使用
    RECYCLED,   // 从空闲列表中提取缓冲区以重用
    FREED,      // 缓冲区放入空闲列表以供将来回收
    RELEASED,   // 缓冲区内存被释放
    ALLOCATION_FAILED  // 缓冲区分配失败
  };
  virtual ~IMpsAllocatorCallback() = default;
  // 执行 MPSAllocator 回调函数
  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// MPS 分配器将在释放内存块时执行每个注册的回调。
C10_DECLARE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);
// 注册 MPS 分配器回调
#define REGISTER_MPS_ALLOCATOR_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(MPSAllocatorCallbacksRegistry, name, __VA_ARGS__);

// 获取 IMPSAllocator 实例
IMPSAllocator* getIMPSAllocator(bool sharedAllocator = false);

} // namespace at::mps
```