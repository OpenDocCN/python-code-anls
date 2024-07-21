# `.\pytorch\c10\xpu\XPUDeviceProp.h`

```py
#pragma once

#include <c10/xpu/XPUMacros.h>
#include <sycl/sycl.hpp>

namespace c10::xpu {

#define AT_FORALL_XPU_EXT_DEVICE_PROPERTIES(_)           \
  /* 定义 Intel GPU 相关的设备属性 */                     \
  /* 与 Intel GPU 相关联的 EUs 的数量为 512 */            \
  _(gpu_eu_count, 512)                                   \
                                                         \
  /* 每个子片上的 EU 数量为 8 */                          \
  _(gpu_eu_count_per_subslice, 8)                        \
                                                         \
  /* GPU 的 EU 的 SIMD 宽度为 8 */                        \
  _(gpu_eu_simd_width, 8)                                \
                                                         \
  /* 每个 GPU EU 的硬件线程数量为 8 */                     \
  _(gpu_hw_threads_per_eu, 8)

#define AT_FORALL_XPU_DEVICE_ASPECT(_)                  \
  /* 定义设备的特性 */                                    \
  /* 设备支持 sycl::half 类型 */                          \
  _(fp16)                                               \
                                                        \
  /* 设备支持 double 类型 */                             \
  _(fp64)                                               \
                                                        \
  /* 设备支持 64 位原子操作 */                            \
  _(atomic64)

#define _DEFINE_SYCL_PROP(ns, property, member) \
  ns::property::return_type member;

#define DEFINE_DEVICE_PROP(property) \
  _DEFINE_SYCL_PROP(sycl::info::device, property, property)

#define DEFINE_PLATFORM_PROP(property, member) \
  _DEFINE_SYCL_PROP(sycl::info::platform, property, member)

#define DEFINE_EXT_DEVICE_PROP(property, ...) \
  _DEFINE_SYCL_PROP(sycl::ext::intel::info::device, property, property)

#define DEFINE_DEVICE_ASPECT(member) bool has_##member;

// 定义设备属性结构体
struct C10_XPU_API DeviceProp {
  // 定义所有 XPU 设备属性
  AT_FORALL_XPU_DEVICE_PROPERTIES(DEFINE_DEVICE_PROP);

  // 平台名称
  DEFINE_PLATFORM_PROP(name, platform_name);

  // 定义所有 Intel GPU 设备属性
  AT_FORALL_XPU_EXT_DEVICE_PROPERTIES(DEFINE_EXT_DEVICE_PROP);

  // 定义所有设备特性
  AT_FORALL_XPU_DEVICE_ASPECT(DEFINE_DEVICE_ASPECT);
};

#undef _DEFINE_SYCL_PROP
#undef DEFINE_DEVICE_PROP
#undef DEFINE_PLATFORM_PROP
#undef DEFINE_EXT_DEVICE_PROP
#undef DEFINE_DEVICE_ASPECT

} // namespace c10::xpu
```