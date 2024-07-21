# `.\pytorch\caffe2\perfkernels\embedding_lookup_idx.h`

```py
#pragma once

#include <cstdint> // 包含标准整数类型定义的头文件

namespace caffe2 {

// clang-format off
/**
 * Embedding lookup with reduction.
 *
 * `input` of size data_size * block_size
 * `indices` of size index_size
 * `offsets` of size output_size
 * `weights` nullptr or array of size index_size
 * `out` of size output_size * block_size
 *
 * Behavior is roughly equivalent to pseudocode:
 *
 * pos = 0
 * for (i = 0..output_size-1)
 *   for (k = 0..block_size-1)
 *     out[i*block_size + k] = 0
 *   start_offset = offsets[i]
 *   end_offset = offsets[i+1]
 *   length = end_offset - start_offset
 *   for (j = start_offset..end_offset-1)
 *     for (k = 0..block_size-1)
 *       out[i*block_size + k] += input[indices[pos]*block_size + k] *
 *           (weights ? weights[IS_WEIGHT_POSITIONAL ? j - start_offset : pos] : 1.0)
 *     pos += 1
 *   if (normalize_weights && length > 0)
 *     for (k = 0..block_size-1)
 *       out[i*block_size + k] /= length
 *
 * TODO: make this API also take "offsets" rather than "lengths" to match the
 *       API for PyTorch's EmbeddingBag
 */
// clang-format on

/**
 * Template function for performing embedding lookup with reduction.
 *
 * @tparam IndexType Type for indices array
 * @tparam InType Type for input data array
 * @tparam OutType Type for output data array
 * @tparam IS_WEIGHT_POSITIONAL Flag indicating whether weights are positional
 * @param block_size Size of each block in input and output
 * @param output_size Number of output entries
 * @param index_size Number of indices
 * @param data_size Number of entries in input data
 * @param input Pointer to the input data array
 * @param indices Pointer to the indices array
 * @param offsets Pointer to the offsets array
 * @param weights Optional pointer to weights array (can be nullptr)
 * @param scale_bias Optional pointer to scale and bias parameters for uint8 input
 * @param normalize_by_lengths Flag indicating whether to normalize output by lengths
 * @param out Pointer to the output data array
 */
template <
    typename IndexType,
    typename InType,
    typename OutType,
    bool IS_WEIGHT_POSITIONAL = false>
void EmbeddingLookupIdx(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const IndexType* offsets,
    const float* weights,
    const float* scale_bias,
    bool normalize_by_lengths,
    OutType* out);

} // namespace caffe2
```