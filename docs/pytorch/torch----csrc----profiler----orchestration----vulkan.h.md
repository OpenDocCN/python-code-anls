# `.\pytorch\torch\csrc\profiler\orchestration\vulkan.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <torch/csrc/profiler/stubs/base.h>
// 包含 Torch 的基础头文件 base.h

#include <torch/csrc/profiler/util.h>
// 包含 Torch 的工具头文件 util.h

#include <cstdint>
// 包含 C++ 标准库中的 cstdint 头文件，定义了标准整数类型

namespace torch {
namespace profiler {
namespace impl {
namespace vulkan {

// 定义一个函数指针类型 GetShaderNameAndDurationNsFn
// 这里使用 std::function 是因为需要在 lambda 上下文中捕获 QueryPool
// 参考：https://stackoverflow.com/a/28746827
using GetShaderNameAndDurationNsFn =
    std::function<std::tuple<std::string, uint64_t>(int64_t)>;
// 声明一个函数，用于注册 GetShaderNameAndDurationNsFn 类型的函数指针
TORCH_API void registerGetShaderNameAndDurationNs(
    GetShaderNameAndDurationNsFn get_shader_name_and_duration_ns);

// 声明一个函数，用于取消注册 GetShaderNameAndDurationNsFn 类型的函数指针
TORCH_API void deregisterGetShaderNameAndDurationNs();

// 声明一个函数，返回一个包含着着色器名称和持续时间的元组
std::tuple<std::string, uint64_t> getShaderNameAndDurationNs(
    const vulkan_id_t& vulkan_id);

} // namespace vulkan
} // namespace impl
} // namespace profiler
} // namespace torch
```