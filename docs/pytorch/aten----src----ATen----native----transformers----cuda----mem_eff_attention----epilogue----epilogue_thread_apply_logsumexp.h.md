# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\epilogue\epilogue_thread_apply_logsumexp.h`

```py
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include <cuda_fp16.h>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// 结构体模板：ArrayExponential，用于计算数组元素的指数
template <typename Element, int ElementsPerAccess>
struct ArrayExponential {
  // 函数调用运算符重载，计算输入数组每个元素的指数，并返回结果数组
  CUTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
      Array<Element, ElementsPerAccess> const& input) const {
    // 定义结果数组
    Array<Element, ElementsPerAccess> result;

    // 循环遍历输入数组中的每个元素
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      // 计算当前元素的指数并存储到结果数组中
      result[i] = expf(input[i]);
    }

    // 返回计算后的结果数组
    return result;
  }
};

// 结构体模板结束

// 结构体模板实例化的声明，指定每次访问的元素数量
template <int ElementsPerAccess>
/// Templated struct defining an exponential operation on an array of half-precision elements.
template <typename half_t, int ElementsPerAccess>
struct ArrayExponential {
  
  /// CUDA thread function to apply exponential operation on input array
  CUTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
      Array<half_t, ElementsPerAccess> const& input) const {
    
    // Result array initialization
    Array<half_t, ElementsPerAccess> result;

    // Determine the number of half2 vectors in the input array
    int const kVectorCount = ElementsPerAccess / 2;

    // Cast input array to __half2 for optimized memory access
    __half2 const* input_ptr =
        reinterpret_cast<__half2 const*>(input.raw_data());
    
    // Cast result array to __half2 for optimized memory access
    __half2* res_ptr = reinterpret_cast<__half2*>(result.raw_data());

    // Unroll loop to compute exponential for each pair of half2 elements
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = h2exp(input_ptr[i]);
    }

    // Return the computed result array
    return result;
  }
};
} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Templated class to apply logarithm of sum of exponentials operation.
template <
    typename ElementOutput_,   // Output element type
    typename ElementLSE_,      // Log sum exponential element type
    typename ElementAccumulator_,  // Accumulator element type from matrix multiplication
    typename ElementCompute_,  // Intermediate computation and exponential calculation element type
    int ElementsPerAccess>    // Number of elements to process per access
class ApplyLogSumExp {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementLSE = ElementLSE_;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;
  static const ScaleType::Kind kScale =
      cutlass::epilogue::thread::ScaleType::NoBetaScaling;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentLSE = Array<ElementLSE, kElementsPerAccess>;
  using FragmentScaleBias = FragmentLSE; // Used by epilogue_smem_accumulator.h

 public:
  //
  // Methods
  //

  /// Default constructor
  CUTLASS_HOST_DEVICE
  ApplyLogSumExp() {}

  /// Determines if source data is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  /// Sets the partition index and count for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  /// Main function to apply the logarithm of sum of exponentials
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& AB,
      FragmentLSE const& scale_unused,
      FragmentLSE const& bias) const {
    
    // Convert Accumulator to Compute type
    FragmentCompute frag_AB = NumericArrayConverter<
        ElementCompute,
        ElementAccumulator,
        kElementsPerAccess>()(AB);
    
    // Convert bias to Compute type for calculation
    FragmentCompute frag_lse_compute =
        NumericArrayConverter<ElementCompute, ElementLSE, kElementsPerAccess>()(
            bias);
    
    // Compute the difference between AB and bias (lse)
    FragmentCompute frag_compute;
    minus<FragmentCompute> minus_lse;
    frag_compute = minus_lse(frag_AB, frag_lse_compute);
    
    // Apply exponential function to the computed result
    detail::ArrayExponential<ElementCompute, kElementsPerAccess> apply_exp;
    frag_compute = apply_exp(frag_compute);

    // Convert Compute type to Output type
    return NumericArrayConverter<
        ElementOutput,
        ElementCompute,
        kElementsPerAccess>()(frag_compute);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


注释：


// 这段代码定义了三个嵌套的命名空间：cutlass -> epilogue -> thread
} // namespace thread
} // namespace epilogue
} // namespace cutlass
// 这些命名空间的作用是将相关的代码组织到逻辑上的分组中，以避免命名冲突和提高代码的可维护性
/////////////////////////////////////////////////////////////////////////////////////////////////


这段C++代码定义了三个命名空间（namespace），分别是`cutlass`，`epilogue`和`thread`。命名空间用于组织代码，防止不同模块之间的命名冲突，提高代码的可读性和可维护性。
```