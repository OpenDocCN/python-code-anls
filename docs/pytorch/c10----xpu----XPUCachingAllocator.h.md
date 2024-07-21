# `.\pytorch\c10\xpu\XPUCachingAllocator.h`

```
#pragma once

# 预处理指令，确保头文件只被包含一次，防止重复定义


#include <c10/core/Allocator.h>
#include <c10/xpu/XPUStream.h>

# 包含必要的头文件 `<c10/core/Allocator.h>` 和 `<c10/xpu/XPUStream.h>`，用于声明所需的类和函数


namespace c10::xpu::XPUCachingAllocator {

# 命名空间开始：定义命名空间 `c10::xpu::XPUCachingAllocator`


C10_XPU_API Allocator* get();

# 声明函数 `get()`，返回类型为 `Allocator*`，用 `C10_XPU_API` 指示其在外部可见


C10_XPU_API void init(DeviceIndex device_count);

# 声明函数 `init()`，参数为 `DeviceIndex device_count`，用 `C10_XPU_API` 指示其在外部可见


C10_XPU_API void emptyCache();

# 声明函数 `emptyCache()`，无参数，用 `C10_XPU_API` 指示其在外部可见


C10_XPU_API void* raw_alloc(size_t size);

# 声明函数 `raw_alloc()`，返回 `void*`，参数 `size_t size`，用 `C10_XPU_API` 指示其在外部可见


C10_XPU_API void raw_delete(void* ptr);

# 声明函数 `raw_delete()`，无返回值，参数 `void* ptr`，用 `C10_XPU_API` 指示其在外部可见


C10_XPU_API void recordStream(const DataPtr& dataPtr, XPUStream stream);

# 声明函数 `recordStream()`，无返回值，参数 `const DataPtr& dataPtr` 和 `XPUStream stream`，用 `C10_XPU_API` 指示其在外部可见


} // namespace c10::xpu::XPUCachingAllocator

# 命名空间结束：结束命名空间 `c10::xpu::XPUCachingAllocator`
```