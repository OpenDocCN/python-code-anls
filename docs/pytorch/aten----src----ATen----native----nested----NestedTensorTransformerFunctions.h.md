# `.\pytorch\aten\src\ATen\native\nested\NestedTensorTransformerFunctions.h`

```py
/**
 * Transformer-specific NestedTensor utility functions.
 *
 * Not co-located with NestedTensor core code yet because they only
 * support specific cases needed in transformers.
 */
#pragma once

#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

namespace c10 {
class Scalar;
} // namespace c10

namespace at {
class Tensor;
namespace native {
struct NestedTensorImpl;

// Performs matrix multiplication between a contiguous NestedTensor and a non-NestedTensor,
// assuming self.dim() == 3 and other.dim() == 2. Requires consistent dimensions and matching sizes.
Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other);

// Computes mat1 * alpha + mat2 * beta, where mat1 is a contiguous NestedTensor,
// mat2 is a Tensor, and self is not a NestedTensor. Dimensions must match requirements.
Tensor NestedTensor_times_Tensor_plus_Tensor_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    std::optional<bool> use_gelu = c10::nullopt);

// Adds a non-NestedTensor `other` to a contiguous NestedTensor `self` in place.
Tensor NestedTensor_add_NestedTensor_in_place(
    const Tensor& self,
    const Tensor& other);

// Computes batch offsets from a sizes tensor for a NestedTensor.
TORCH_API Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements);

// Converts a padded tensor into a NestedTensor using CPU operations.
Tensor NestedTensor_from_padded_tensor_cpu(
    const Tensor& padded,
    const NestedTensorImpl& nt);

// Converts a NestedTensor to a mask tensor with optional dimensions and lengths.
Tensor NestedTensor_to_mask(const Tensor& nt, std::optional<int64_t> mask_dim, std::optional<int64_t> mask_dim_length);

// Launches a kernel to remove padding from input to output tensors, with specified offsets and sizes.
template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

// Launches a kernel to remove padding from input to output tensors with a specific transformation order.
template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

// Launches a kernel to add padding to input tensors, based on offsets and sizes, with a specified padding value.
template <typename T>
void add_padding_kernelLauncher(
    T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

// Helper function for attention mechanism in transformers, supporting dropout and optional attention weights.
TORCH_API Tensor flash_attention_helper(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool need_attn_weights,
    bool is_causal);

// Memory-efficient helper for nested unpacking in transformers, supporting dropout and attention weights.
TORCH_API std::tuple<Tensor, Tensor> mem_efficient_helper_nested_unpacked(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool need_attn_weights,
    bool is_causal);
} // namespace native
} // namespace at
```