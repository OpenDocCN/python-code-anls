# `.\pytorch\aten\src\ATen\detail\HIPHooksInterface.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <c10/core/Allocator.h>
// 引入C10库中的Allocator模块

#include <c10/core/GeneratorImpl.h>
// 引入C10库中的GeneratorImpl模块

#include <c10/util/Exception.h>
// 引入C10库中的Exception模块

#include <c10/util/Registry.h>
// 引入C10库中的Registry模块

#include <memory>
// 引入标准库中的memory模块

namespace at {
// 命名空间at，包含了PyTorch的核心功能

class Context;
// 前置声明，表示类Context在此处被引用

}

// NB: Class must live in `at` due to limitations of Registry.h.
// 由于Registry.h的限制，类必须存在于at命名空间中。

namespace at {

// The HIPHooksInterface is an omnibus interface for any HIP functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of HIP code).  See
// CUDAHooksInterface for more detailed motivation.

// HIPHooksInterface是一个多功能接口，用于从CPU代码中调用任何HIP功能
// （因此必须动态分派，以允许将HIP代码单独编译）。详细信息请参见CUDAHooksInterface。

struct TORCH_API HIPHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  // 实际上不应该实现这个函数，它用于抑制-Werror=non-virtual-dtor警告

  virtual ~HIPHooksInterface() = default;
  // 虚析构函数，确保派生类的析构可以被正确调用

  // Initialize the HIP library state
  // 初始化HIP库的状态
  virtual void initHIP() const {
    AT_ERROR("Cannot initialize HIP without ATen_hip library.");
  }

  virtual std::unique_ptr<c10::GeneratorImpl> initHIPGenerator(Context*) const {
    AT_ERROR("Cannot initialize HIP generator without ATen_hip library.");
  }
  // 初始化HIP生成器，如果缺少ATen_hip库则报错

  virtual bool hasHIP() const {
    return false;
  }
  // 检查是否支持HIP，此处返回false

  virtual c10::DeviceIndex current_device() const {
    return -1;
  }
  // 返回当前设备索引，这里返回-1表示无设备

  virtual Allocator* getPinnedMemoryAllocator() const {
    AT_ERROR("Pinned memory requires HIP.");
  }
  // 获取固定内存分配器，如果没有HIP则报错

  virtual void registerHIPTypes(Context*) const {
    AT_ERROR("Cannot registerHIPTypes() without ATen_hip library.");
  }
  // 注册HIP类型，如果没有ATen_hip库则报错

  virtual int getNumGPUs() const {
    return 0;
  }
  // 获取GPU数量，此处返回0表示没有GPU
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
// 虚拟参数，用于抑制“ISO C++11要求宏中的'...'至少有一个参数”

struct TORCH_API HIPHooksArgs {};
// 定义空结构体HIPHooksArgs，用于注册HIPTHooks

TORCH_DECLARE_REGISTRY(HIPHooksRegistry, HIPHooksInterface, HIPHooksArgs);
// 使用宏TORCH_DECLARE_REGISTRY声明HIPHooksRegistry，用于注册HIPTHooksInterface

#define REGISTER_HIP_HOOKS(clsname) \
  C10_REGISTER_CLASS(HIPHooksRegistry, clsname, clsname)
// 宏定义REGISTER_HIP_HOOKS，用于注册HIPTHooks的类

namespace detail {
TORCH_API const HIPHooksInterface& getHIPHooks();
// 声明函数getHIPHooks，用于获取HIPHooksInterface的引用

} // namespace detail
} // namespace at
// 命名空间at结束
```