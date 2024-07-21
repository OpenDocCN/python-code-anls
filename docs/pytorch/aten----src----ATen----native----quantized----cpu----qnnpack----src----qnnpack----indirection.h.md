# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\indirection.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

// 引入 PyTorch QNNPACK 头文件
#include <pytorch_qnnpack.h>

// 引入 QNNPACK 公共头文件
#include <qnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

// 初始化 Conv3D 运算符的间接索引
PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_conv3d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size);

// 初始化 Depthwise Convolution 运算符的间接索引
PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_dwconv(
    pytorch_qnnp_operator_t op,
    size_t batch_start);

// 初始化 Deconvolution 运算符的间接索引
PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_deconv2d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size);

// 初始化 Max Pooling 运算符的间接索引
PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_maxpool2d(
    pytorch_qnnp_operator_t op,
    size_t batch_start);

// 设置 QNNPACK 运算符的步骤维度
PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_set_step_dimensions(
    pytorch_qnnp_operator_t op);

#ifdef __cplusplus
} /* extern "C" */
#endif
```