# `.\pytorch\aten\src\ATen\native\mkldnn\IDeepRegistration.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <c10/core/Allocator.h>

#if AT_MKLDNN_ENABLED()

// 需要在库中仅包含一次。
// 包含ideep_pin_singletons.hpp以保证单例机制。
#include <ideep_pin_singletons.hpp>

// 使用ideep命名空间
using namespace ideep;

// 定义cpu_alloc变量，并注册CPU引擎的分配器
RegisterEngineAllocator cpu_alloc(
  engine::cpu_engine(),                        // 使用CPU引擎
  [](size_t size) {                            // 分配器lambda表达式
    return c10::GetAllocator(c10::DeviceType::CPU)->raw_allocate(size);  // 调用ATen获取CPU设备的分配器并分配内存
  },
  [](void* p) {                                // 释放器lambda表达式
    c10::GetAllocator(c10::DeviceType::CPU)->raw_deallocate(p);          // 调用ATen获取CPU设备的分配器并释放内存
  }
);

// 定义命名空间at::native::mkldnn，包含清除计算缓存的函数
namespace at::native::mkldnn {

// 声明清除计算缓存的函数clear_computation_cache()
void clear_computation_cache();

// 实现清除计算缓存的函数clear_computation_cache()
void clear_computation_cache() {
  // 重置前向卷积的计算缓存
  // 该缓存也包含了最大OpenMP工作线程数
  ideep::convolution_forward::t_store().clear();  // 清空前向卷积的计算缓存
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
```