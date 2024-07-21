# `.\pytorch\aten\src\ATen\native\nested\NestedTensorTransformerUtils.h`

```
/**
 * 包含 ATen 库，提供张量运算支持
 */
#include <ATen/ATen.h>

/**
 * at::native::preprocessing 命名空间，包含预处理函数
 */
namespace at {
namespace native {
namespace preprocessing {

/**
 * 该函数处理嵌套的查询（query）、键（key）和值（value），以便与 flash-attention 或 efficient-attention 内核一起运行。
 * 返回一个包含运行融合内核所需所有数据的元组。
 * @return 元组，包含用于运行融合内核的所有必要数据
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, Tensor>
sdpa_nested_preprocessing(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value);

/**
 * 该函数处理嵌套的查询（query）、键（key）、值（value）、grad_out 和 out，以便在反向时与 flash-attention 或 efficient-attention 内核一起运行。
 * 使用这两个函数避免为 cumulative_sequence_length_q 和 cumulative_sequence_length_kv 做相同的预处理。
 * 返回一个包含运行融合内核所需所有数据的元组。
 * @return 元组，包含用于运行融合内核的所有必要数据
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
sdpa_nested_preprocessing_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_kv,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_kv);

} // namespace preprocessing
} // namespace native
} // namespace at
```