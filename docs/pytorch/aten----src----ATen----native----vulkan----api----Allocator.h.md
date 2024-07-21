# `.\pytorch\aten\src\ATen\native\vulkan\api\Allocator.h`

```
#pragma once
// 防止头文件重复包含

//
// Do NOT include vk_mem_alloc.h directly.
// Always include this file (Allocator.h) instead.
//
// 不要直接包含 vk_mem_alloc.h，而是始终包含 Allocator.h

#include <ATen/native/vulkan/api/vk_api.h>
// 包含 Vulkan API 头文件

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API

#define VMA_VULKAN_VERSION 1000000
// 定义 Vulkan 版本号

#ifdef USE_VULKAN_WRAPPER
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#else
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif /* USE_VULKAN_WRAPPER */
// 根据 USE_VULKAN_WRAPPER 定义选择静态或动态 Vulkan 函数载入方式

#define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (32ull * 1024 * 1024)
#define VMA_SMALL_HEAP_MAX_SIZE (256ull * 1024 * 1024)
// 定义默认的大堆块大小和小堆最大大小

#define VMA_STATS_STRING_ENABLED 0
// 禁用统计字符串功能

#ifdef VULKAN_DEBUG
#define VMA_DEBUG_ALIGNMENT 4096
#define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY 0
#define VMA_DEBUG_DETECT_CORRUPTION 1
#define VMA_DEBUG_GLOBAL_MUTEX 1
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#define VMA_DEBUG_MARGIN 64
#define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY 256
#define VMA_RECORDING_ENABLED 1
// 如果定义了 VULKAN_DEBUG，则设置调试选项

#define VMA_DEBUG_LOG(format, ...)
/*
#define VMA_DEBUG_LOG(format, ...) do { \
    printf(format, __VA_ARGS__); \
    printf("\n"); \
} while(false)
*/
// 定义调试日志宏，但此处是注释掉的实现方式

#endif /* VULKAN_DEBUG */

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif /* __clang__ */
// 在 Clang 编译器下，忽略特定的警告

#include <include/vk_mem_alloc.h>
// 包含 Vulkan 内存分配器头文件

#ifdef __clang__
#pragma clang diagnostic pop
#endif /* __clang__ */
// 恢复 Clang 编译器的警告设置

#endif /* USE_VULKAN_API */
// 结束 USE_VULKAN_API 的条件编译段落
```