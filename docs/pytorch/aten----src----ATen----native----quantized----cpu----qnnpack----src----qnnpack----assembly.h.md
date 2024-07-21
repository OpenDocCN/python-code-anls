# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\assembly.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#ifdef __ELF__
    // 定义宏 BEGIN_FUNCTION，用于开始一个函数的汇编定义
    .macro BEGIN_FUNCTION name
        // 设置段为 .text，2字节对齐
        .text
        .align 2
        // 声明函数名为全局可见，并定义其类型为函数
        .global \name
        .type \name, %function
        // 定义函数入口点
        \name:
    // 宏定义结束
    .endm

    // 定义宏 END_FUNCTION，用于结束一个函数的汇编定义
    .macro END_FUNCTION name
        // 设置函数大小
        .size \name, .-\name
    .endm
#elif defined(__MACH__)
    // 如果在 macOS 平台上
    // 定义宏 BEGIN_FUNCTION，用于开始一个函数的汇编定义
    .macro BEGIN_FUNCTION name
        // 设置段为 .text，2字节对齐
        .text
        .align 2
        // 声明函数名为全局可见，并定义其类型为函数，加前缀 "_"
        .global _\name
        .private_extern _\name
        // 定义函数入口点，加前缀 "_"
        _\name:
    // 宏定义结束
    .endm

    // 定义宏 END_FUNCTION，用于结束一个函数的汇编定义
    .macro END_FUNCTION name
    .endm
#endif
```