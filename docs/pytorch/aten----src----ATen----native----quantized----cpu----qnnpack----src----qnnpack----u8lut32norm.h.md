# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\u8lut32norm.h`

```py
/*
 * 版权声明：
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码受 BSD 风格许可证保护，详见源代码根目录下的 LICENSE 文件。
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

// 定义一个宏，声明一个名为 fn_name 的函数原型，用于 PyTorch 的 x8 LUT 32 标准化单元内核函数
#define DECLARE_PYTORCH_X8LUT32NORM_UKERNEL_FUNCTION(fn_name) \
  // PyTorch QNNP 内部函数声明，无返回值
  PYTORCH_QNNP_INTERNAL void fn_name(                 \
      // 输入大小为 n，输入数据为 x（uint8_t 类型指针），查找表为 t（uint32_t 类型指针），输出数据为 y（uint8_t 类型指针）
      size_t n, const uint8_t* x, const uint32_t* t, uint8_t* y);

// 声明一个具体的 PyTorch x8 LUT 32 标准化单元内核函数
DECLARE_PYTORCH_X8LUT32NORM_UKERNEL_FUNCTION(pytorch_u8lut32norm_ukernel__scalar)

#ifdef __cplusplus
} /* extern "C" */
#endif
```