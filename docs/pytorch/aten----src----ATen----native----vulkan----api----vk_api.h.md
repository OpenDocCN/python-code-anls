# `.\pytorch\aten\src\ATen\native\vulkan\api\vk_api.h`

```py
#pragma once

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下内容

#ifdef USE_VULKAN_WRAPPER
// 如果定义了 USE_VULKAN_WRAPPER 宏

#ifdef USE_VULKAN_VOLK
// 如果同时定义了 USE_VULKAN_VOLK 宏，则包含 volk.h 头文件
#include <volk.h>
#else
// 如果未定义 USE_VULKAN_VOLK 宏，则包含 vulkan_wrapper.h 头文件
#include <vulkan_wrapper.h>
#endif /* USE_VULKAN_VOLK */

#else
// 如果未定义 USE_VULKAN_WRAPPER 宏，则直接包含 Vulkan 标准头文件
#include <vulkan/vulkan.h>
#endif /* USE_VULKAN_WRAPPER */

#endif /* USE_VULKAN_API */
// 结束 USE_VULKAN_API 宏的条件编译块
```