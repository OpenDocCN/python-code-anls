# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\gemmlowp-scalar.h`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits.h>
#include <stdint.h>

/*
 * The code below is adapted from Google's gemmlowp library.
 * It is only used in QNNPACK unit tests and comparative benchmarks,
 * but not the library itself.
 */

// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// 声明 gemmlowp_scalar_vqrdmulh_s32 函数，计算有符号整数 a 和 b 的饱和乘以 2 的四舍五入除法结果的高 32 位
inline static int32_t gemmlowp_scalar_vqrdmulh_s32(int32_t a, int32_t b) {
  // 判断是否会发生溢出
  const bool overflow = a == b && a == INT32_MIN;
  // 计算 a 和 b 的乘积，并将结果保存在 64 位整数 ab_64 中
  const int64_t ab_64 = (int64_t)a * (int64_t)b;
  // 根据 a 和 b 的符号异或来确定 nudge 值，用于四舍五入
  const int32_t nudge =
      (a ^ b) >= 0 ? INT32_C(0x40000000) : -INT32_C(0x3FFFFFFF);
  // 计算乘积的四舍五入除以 2 的高 32 位结果
  const int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / INT64_C(0x80000000));
  // 如果溢出，则返回 INT32_MAX，否则返回计算得到的结果
  return overflow ? INT32_MAX : ab_x2_high32;
}

// 声明 gemmlowp_scalar_rdivbypo2_s32 函数，对有符号整数 x 进行右移并进行四舍五入分为 2 的指数幂
inline static int32_t gemmlowp_scalar_rdivbypo2_s32(int32_t x, int exponent) {
  // 创建掩码，用于获取 x 的低 exponent 位
  const int32_t mask = ((1 << exponent) - 1);
  // 计算 x 除以 2 的 exponent 次幂后的余数
  const int32_t remainder = x & mask;
  // 计算阈值，用于判断是否需要进行四舍五入
  const int32_t threshold = (mask >> 1) + (int32_t)(x < 0);
  // 返回 x 右移 exponent 位并加上四舍五入的余数
  return asr_s32(x, exponent) + (int32_t)(remainder > threshold);
}
```