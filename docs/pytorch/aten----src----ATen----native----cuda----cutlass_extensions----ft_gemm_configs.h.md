# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\ft_gemm_configs.h`

```py
/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace fastertransformer {
// 定义枚举类型 CutlassTileConfig，用于描述矩阵乘法的瓦片配置
enum class CutlassTileConfig {
    // 未定义的配置，需要通过启发式算法选择合适的配置
    Undefined,

    // 通过启发式算法选择合适的配置
    ChooseWithHeuristic,

    // SiMT 配置，描述 CTA 形状和 Warp 形状
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore 配置，CTA_N = 128, CTA_K = 64，适用于 M=32 的 Warp 形状
    CtaShape32x128x64_WarpShape32x32x64,

    // TensorCore 配置，CTA_N = 128, CTA_K = 64，适用于 M=64 的两种 Warp 形状
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // TensorCore 配置，CTA_N = 128, CTA_K = 64，适用于 M=128 的两种 Warp 形状
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape128x32x64
};

// 定义枚举类型 SplitKStyle，描述 K 方向分割的风格
enum class SplitKStyle {
    NO_SPLIT_K,      // 不进行 K 方向的分割
    SPLIT_K_SERIAL   // 串行进行 K 方向的分割
    // SPLIT_K_PARALLEL // 目前尚不支持并行进行 K 方向的分割
};

// 定义结构体 CutlassGemmConfig，描述 Cutlass GEMM 算法的配置
struct CutlassGemmConfig {
    CutlassTileConfig tile_config    = CutlassTileConfig::ChooseWithHeuristic; // 瓦片配置，默认为通过启发式算法选择
    SplitKStyle       split_k_style  = SplitKStyle::NO_SPLIT_K;               // K 方向分割的风格，默认为不分割
    int               split_k_factor = -1;                                    // K 方向分割的因子，默认为 -1
    int               stages         = -1;                                    // 阶段数，默认为 -1
};

}  // namespace fastertransformer
```