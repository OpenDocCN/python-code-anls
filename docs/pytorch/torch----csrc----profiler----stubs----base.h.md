# `.\pytorch\torch\csrc\profiler\stubs\base.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <functional>
// 包含函数对象的标准库头文件

#include <memory>
// 包含智能指针相关的标准库头文件

#include <c10/core/Device.h>
// 包含 C10 库中设备相关定义的头文件

#include <c10/util/strong_type.h>
// 包含 C10 库中强类型相关定义的头文件

#include <torch/csrc/Export.h>
// 包含 Torch 导出相关的头文件

struct CUevent_st;
// 声明 CUDA 事件结构体 CUevent_st

namespace torch {
namespace profiler {
namespace impl {

// ----------------------------------------------------------------------------
// -- Annotation --------------------------------------------------------------
// ----------------------------------------------------------------------------
// 命名空间 torch::profiler::impl 下的注释部分

using ProfilerEventStub = std::shared_ptr<CUevent_st>;
// 定义 ProfilerEventStub 类型为指向 CUevent_st 的 shared_ptr

using ProfilerVoidEventStub = std::shared_ptr<void>;
// 定义 ProfilerVoidEventStub 类型为指向 void 的 shared_ptr

struct TORCH_API ProfilerStubs {
  virtual void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const = 0;
  // 纯虚函数，用于记录性能数据

  virtual float elapsed(
      const ProfilerVoidEventStub* event,
      const ProfilerVoidEventStub* event2) const = 0;
  // 纯虚函数，计算两个事件之间的时间差

  virtual void mark(const char* name) const = 0;
  // 纯虚函数，标记一个事件点，使用名称描述

  virtual void rangePush(const char* name) const = 0;
  // 纯虚函数，开始一个命名的范围

  virtual void rangePop() const = 0;
  // 纯虚函数，结束当前范围

  virtual bool enabled() const {
    return false;
  }
  // 虚函数，默认返回 false，用于检查性能分析器是否启用

  virtual void onEachDevice(std::function<void(int)> op) const = 0;
  // 纯虚函数，对每个设备执行操作

  virtual void synchronize() const = 0;
  // 纯虚函数，同步所有设备上的性能数据

  virtual ~ProfilerStubs();
  // 虚析构函数，用于派生类的释放资源

};

TORCH_API void registerCUDAMethods(ProfilerStubs* stubs);
// 注册 CUDA 相关的性能分析方法

TORCH_API const ProfilerStubs* cudaStubs();
// 获取 CUDA 相关的性能分析接口

TORCH_API void registerITTMethods(ProfilerStubs* stubs);
// 注册 Intel VTune 相关的性能分析方法

TORCH_API const ProfilerStubs* ittStubs();
// 获取 Intel VTune 相关的性能分析接口

TORCH_API void registerPrivateUse1Methods(ProfilerStubs* stubs);
// 注册私有使用1的性能分析方法

TORCH_API const ProfilerStubs* privateuse1Stubs();
// 获取私有使用1的性能分析接口

using vulkan_id_t = strong::type<
    int64_t,
    struct _VulkanID,
    strong::regular,
    strong::convertible_to<int64_t>,
    strong::hashable>;
// 定义强类型 vulkan_id_t，用于 Vulkan 相关标识符

} // namespace impl
} // namespace profiler
} // namespace torch
```