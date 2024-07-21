# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\isa-checks.h`

```py
/*
 * 版权所有 (c) Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录的 LICENSE 文件中使用 BSD 风格许可证授权。
 */

#pragma once

#include <cpuinfo.h>

// 如果系统不支持 x86 SSE2 指令集，则返回
#define TEST_REQUIRES_X86_SSE2                              \
  do {                                                      \
    // 初始化 CPU 信息并检查是否支持 x86 SSE2 指令集
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_sse2()) { \
      return;                                               \
    }                                                       \
  } while (0)

// 如果系统不支持 ARM NEON 指令集，则返回
#define TEST_REQUIRES_ARM_NEON                              \
  do {                                                      \
    // 初始化 CPU 信息并检查是否支持 ARM NEON 指令集
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon()) { \
      return;                                               \
    }                                                       \
  } while (0)

// 如果系统不支持 ARM NEON FP16 ARITH 指令集，则返回
#define TEST_REQUIRES_ARM_NEON_FP16_ARITH                              \
  do {                                                                 \
    // 初始化 CPU 信息并检查是否支持 ARM NEON FP16 ARITH 指令集
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) { \
      return;                                                          \
    }                                                                  \
  } while (0)
```