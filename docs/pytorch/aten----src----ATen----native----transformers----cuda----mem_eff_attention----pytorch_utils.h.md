# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\pytorch_utils.h`

```py
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// 包含 C10 库的标量类型定义
#include <c10/core/ScalarType.h>

// 包含 Cutlass 库的 bfloat16 类型定义
#include <cutlass/bfloat16.h>
// 包含 Cutlass 库的 half 类型定义
#include <cutlass/half.h>

// 定义模板，将 Cutlass 类型映射到 ATen 类型
template <typename scalar_t>
struct CutlassToAtenDtype;

// 特化模板，将 cutlass::half_t 映射到 ATen 的 ScalarType::Half 类型
template <>
struct CutlassToAtenDtype<cutlass::half_t> {
  using scalar_t = cutlass::half_t;

  // 返回 ATen 中的半精度浮点数标量类型
  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Half;
  }
};

// 特化模板，将 cutlass::bfloat16_t 映射到 ATen 的 ScalarType::BFloat16 类型
template <>
struct CutlassToAtenDtype<cutlass::bfloat16_t> {
  using scalar_t = cutlass::bfloat16_t;

  // 返回 ATen 中的 BF16 浮点数标量类型
  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::BFloat16;
  }
};

// 特化模板，将 float 映射到 ATen 的 ScalarType::Float 类型
template <>
struct CutlassToAtenDtype<float> {
  using scalar_t = float;

  // 返回 ATen 中的单精度浮点数标量类型
  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Float;
  }
};
```