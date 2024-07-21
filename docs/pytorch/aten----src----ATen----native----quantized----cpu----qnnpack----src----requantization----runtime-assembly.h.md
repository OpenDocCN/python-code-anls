# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\runtime-assembly.h`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __aarch64__
/* 
 * 定义名为 SUB_ZERO_POINT 的宏，用于执行零点偏移计算
 * 当启用 PYTORCH_QNNPACK_RUNTIME_QUANTIZATION 时，使用 USUBL 指令计算结果
 * 否则，使用 UXTL 指令将 vin1 扩展为 64 位并将结果写入 vout
 */
.macro SUB_ZERO_POINT vout, vin1, vin2
    USUBL \vout, \vin1, \vin2
#else /* aarch32 */
/*
 * 定义名为 SUB_ZERO_POINT 的宏，用于执行零点偏移计算
 * 当启用 PYTORCH_QNNPACK_RUNTIME_QUANTIZATION 时，使用 VSUBL.U8 指令计算结果
 * 否则，使用 VMOVL.U8 指令将 din1 扩展为 64 位并将结果写入 qout
 */
.macro SUB_ZERO_POINT qout, din1, din2
    VSUBL.U8 \qout, \din1, \din2
#endif
.endm
```