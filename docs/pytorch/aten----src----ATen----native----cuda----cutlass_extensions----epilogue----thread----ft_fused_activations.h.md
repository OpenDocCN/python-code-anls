# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\epilogue\thread\ft_fused_activations.h`

```
/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/functional.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

// 函数声明：将一个浮点数的符号位替换为另一个浮点数的符号位后返回
__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

// 函数声明：对输入的浮点数进行快速双曲正切计算
__forceinline__ __device__ float tanh_opt(float x)
{
    // 根据 CUDA 编译器版本和架构选择使用快速双曲正切函数或手动计算
#if (__CUDACC_VER_MAJOR__ < 11) || (__CUDA_ARCH__ < 750)
    // 当 CUDA 编译器版本小于 11 或 CUDA 架构小于 750 时，使用手动计算的双曲正切函数
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#else
    // 当满足条件时，使用 CUDA 提供的快速双曲正切函数
    return fast_tanh(x);
#endif
}
/////////////////////////////////////////////////////////////////////////////////////////////////
// 特化模板 GELU_taylor<float> 的实现，用于计算 GELU 激活函数在 float 类型上的值
template<>
struct GELU_taylor<float> {
    // 声明静态常量 kIsHeavy 为 true，用于指示此模板特化较为复杂
    static const bool kIsHeavy = true;

    // CUTLASS_DEVICE 表示此函数在设备上执行（可能为 GPU），计算 GELU 激活函数在 z 上的值
    CUTLASS_DEVICE
    float operator()(float const& z) const
    {
        // 定义常数 k0 和 k1，用于 GELU 激活函数计算中的参数
        float k0 = float(0.7978845608028654);
        float k1 = float(0.044715);

        // 计算 GELU 激活函数的近似值
        return float(
            // 使用 cutlass 提供的常数 half<float>() 和 one<float>() 辅助计算
            cutlass::constants::half<float>() * z
            * (cutlass::constants::one<float>() + tanh_opt(k0 * z * (cutlass::constants::one<float>() + k1 * z * z))));
    }

    // 定义 Params 类型别名为 LinearCombinationGenericParams<float>
    using Params = LinearCombinationGenericParams<float>;

    // CUTLASS_DEVICE 表示此函数在设备上执行（可能为 GPU），对 Params 参数类型进行重载
    CUTLASS_DEVICE
    float operator()(float const& scalar, Params const& params_) const
    {
        // 调用单参数的 operator() 函数，对 scalar 执行 GELU 激活函数的计算
        return this->operator()(scalar);
    }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
/////////////////////////////////////////////////////////////////////////////////////////////////
```