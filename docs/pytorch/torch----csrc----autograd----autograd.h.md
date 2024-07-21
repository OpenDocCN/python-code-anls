# `.\pytorch\torch\csrc\autograd\autograd.h`

```py
#pragma once

#include <torch/csrc/autograd/variable.h>

namespace torch::autograd {

/// Computes the sum of gradients of given tensors with respect to graph leaves.
///
/// The graph is differentiated using the chain rule. If any of ``tensors``
/// are non-scalar (i.e. their data has more than one element) and require
/// gradient, then the Jacobian-vector product would be computed, in this case
/// the function additionally requires specifying `grad_tensors`. It should be a
/// sequence of matching length, that contains the "vector" in the
/// Jacobian-vector product, usually the gradient of the differentiated function
/// w.r.t. corresponding tensors
/// (`torch::Tensor()` is an acceptable value for all tensors that don't need
/// gradient tensors).
///
/// This function accumulates gradients in the leaves - you might need to zero
/// them before calling it.
///
/// \param tensors Tensors of which the derivative will be computed.
/// \param grad_tensors The "vector" in the Jacobian-vector product, usually
/// gradients
///     w.r.t. each element of corresponding tensors. `torch::Tensor()` values
///     can be specified for scalar Tensors or ones that don't require grad. If
///     a `torch::Tensor()` value would be acceptable for all grad_tensors, then
///     this argument is optional.
/// \param retain_graph If `false`, the graph used to compute the grad will be
/// freed.
///     Note that in nearly all cases setting this option to `true` is not
///     needed and often can be worked around in a much more efficient way.
///     Defaults to the value of `create_graph`.
/// \param create_graph If `true`, graph of the derivative will be constructed,
/// allowing
///     to compute higher order derivative products. Defaults to `false`.
/// \param inputs Inputs w.r.t. which the gradient will be accumulated into
///     `at::Tensor::grad`. All other Tensors will be ignored. If not provided,
///     the gradient is accumulated into all the leaf Tensors that were used to
///     compute param `tensors`.
//      When inputs are provided and a given input is not a leaf,
//      the current implementation will call its grad_fn (even though it is not
//      strictly needed to get this gradients). It is an implementation detail
//      on which the user should not rely. See
//      https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for
//      more details.
TORCH_API void backward(
    const variable_list& tensors,                 // 输入的张量列表，对它们的梯度将被计算
    const variable_list& grad_tensors = {},      // 对应的梯度向量，用于 Jacobian 向量积
    std::optional<bool> retain_graph = c10::nullopt,  // 是否保留用于计算梯度的计算图
    bool create_graph = false,                   // 是否构建导数的计算图，允许计算高阶导数产品
    const variable_list& inputs = {});           // 梯度将被累积到其中的输入张量的grad属性中

/// Computes and returns the sum of gradients of outputs with respect to the
/// inputs.
///
/// ``grad_outputs`` should be a sequence of length matching ``output``
/// containing the "vector" in Jacobian-vector product, usually the pre-computed
/// gradients w.r.t. each of the outputs. If an output doesn't require_grad,
/// 定义了名为 `grad` 的函数，用于计算反向传播中的梯度。
///
/// \param outputs 差分函数的输出。
/// \param inputs 相对于其计算梯度的输入（不累积到 `at::Tensor::grad` 中）。
/// \param grad_outputs 雅可比向量积中的“向量”。通常是相对于每个输出的梯度。
///                    可以指定 `torch::Tensor()` 值用于标量张量或不需要梯度的张量。
///                    如果对所有 `grad_tensors` 都接受 `torch::Tensor()` 值，那么此参数是可选的。
///                    默认值为 `{}`。
/// \param retain_graph 如果为 `false`，则用于计算梯度的图将被释放。
///                    注意，在几乎所有情况下，设置此选项为 `true` 都是不必要的，通常可以通过更高效的方式解决。
///                    默认为 `create_graph` 的值。
/// \param create_graph 如果为 `true`，将构造导数的图，允许计算更高阶导数的乘积。
///                     默认为 `false`。
/// \param allow_unused 如果为 `false`，指定在计算输出时未使用的输入（因此它们的梯度始终为零）将会报错。
///                     默认为 `false`。
TORCH_API variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs = {},
    std::optional<bool> retain_graph = c10::nullopt,
    bool create_graph = false,
    bool allow_unused = false);

namespace forward_ad {

/// 创建一个新的双重级别并返回其索引。然后应使用此级别索引调用以下其他函数。
/// 此 API 支持在退出之前进入新级别。我们称它们为嵌套的前向自动求导级别。
/// 可用于计算高阶导数。
TORCH_API uint64_t enter_dual_level();

/// 退出给定的级别。这将清除该级别的所有梯度，并且所有具有该级别梯度的双重张量将重新变为常规张量。
/// 此函数仅可用于退出最内部的嵌套级别，因此退出顺序必须与使用上述函数进入的顺序相反。
TORCH_API void exit_dual_level(uint64_t level);

} // namespace forward_ad
} // namespace torch::autograd
```