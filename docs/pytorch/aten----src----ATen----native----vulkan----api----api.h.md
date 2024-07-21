# `.\pytorch\aten\src\ATen\native\vulkan\api\api.h`

```py
#pragma once
// 如果定义了 USE_VULKAN_API 宏，则包含 Vulkan API 相关的头文件
#ifdef USE_VULKAN_API
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <ATen/native/vulkan/api/ShaderRegistry.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Utils.h>
#endif /* USE_VULKAN_API */
```