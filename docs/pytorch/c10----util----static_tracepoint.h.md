# `.\pytorch\c10\util\static_tracepoint.h`

```
#pragma once

#if defined(__ELF__) && (defined(__x86_64__) || defined(__i386__)) && \
    !(defined(TORCH_DISABLE_SDT) && TORCH_DISABLE_SDT)
// 如果目标系统是 ELF 格式，并且是 x86_64 或者 i386 架构，并且未定义 TORCH_DISABLE_SDT 宏，则定义 TORCH_HAVE_SDT 为 1
#define TORCH_HAVE_SDT 1

#include <c10/util/static_tracepoint_elfx86.h>

// 定义 TORCH_SDT 宏，用于生成静态追踪点
#define TORCH_SDT(name, ...) \
  TORCH_SDT_PROBE_N(         \
      pytorch, name, 0, TORCH_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)

// 使用 TORCH_SDT_WITH_SEMAPHORE 宏定义信号量作为全局变量，然后使用 TORCH_SDT_PROBE_N 宏生成带信号量的静态追踪点
#define TORCH_SDT_WITH_SEMAPHORE(name, ...) \
  TORCH_SDT_PROBE_N(                        \
      pytorch, name, 1, TORCH_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)

// 检查指定的静态追踪点是否启用
#define TORCH_SDT_IS_ENABLED(name) (TORCH_SDT_SEMAPHORE(pytorch, name) > 0)

#else
// 如果不满足上述条件，则定义 TORCH_HAVE_SDT 为 0，表示不支持静态追踪点
#define TORCH_HAVE_SDT 0

// 定义 TORCH_SDT 宏为空操作
#define TORCH_SDT(name, ...) \
  do {                       \
  } while (0)

// 定义 TORCH_SDT_WITH_SEMAPHORE 宏为空操作
#define TORCH_SDT_WITH_SEMAPHORE(name, ...) \
  do {                                      \
  } while (0)

// 定义 TORCH_SDT_IS_ENABLED 宏始终返回 false
#define TORCH_SDT_IS_ENABLED(name) (false)
// 定义 TORCH_SDT_DEFINE_SEMAPHORE 和 TORCH_SDT_DECLARE_SEMAPHORE 宏为空操作
#define TORCH_SDT_DEFINE_SEMAPHORE(name)
#define TORCH_SDT_DECLARE_SEMAPHORE(name)

#endif
```