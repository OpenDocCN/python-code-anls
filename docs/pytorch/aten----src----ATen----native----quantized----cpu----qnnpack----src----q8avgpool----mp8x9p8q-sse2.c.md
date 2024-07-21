# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8avgpool\mp8x9p8q-sse2.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>  // SSE2 intrinsic functions
#include <qnnpack/q8avgpool.h>  // Header file for QNNPACK Q8 average pooling

void pytorch_q8avgpool_ukernel_mp8x9p8q__sse2(
    size_t n,  // Number of elements in the batch
    size_t ks,  // Kernel size
    size_t kc,  // Number of channels
    const uint8_t** input,  // Pointer to input data (array of pointers)
    const uint8_t* zero,  // Pointer to zero data
    int32_t* buffer,  // Buffer for intermediate computations
    uint8_t* output,  // Pointer to output data
    size_t input_increment,  // Increment for input pointers
    size_t output_increment,  // Increment for output pointer
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {  // Quantization parameters for average pooling
  assert(n != 0);  // Ensure batch size is not zero
  assert(ks > 9);  // Ensure kernel size is greater than 9
  assert(kc >= 8);  // Ensure number of channels is at least 8

  // Load bias vector into SSE2 register
  const __m128i vbias =
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);

  // Initialize a zero-filled SSE2 register
  const __m128i vzero = _mm_setzero_si128();

  // Load scale vector from quantization parameters
  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  do {
    // Loop over each batch element
    size_t m = ks;  // Initialize m with kernel size

    // Process kernel size loop
    // (Implementation details are omitted in the provided snippet)

    // Update output pointer to point to the next batch element
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);  // Repeat until all batch elements are processed
}
```