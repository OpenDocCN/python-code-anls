# `.\pytorch\torch\csrc\distributed\autograd\autograd.h`

```py
#pragma once

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::variable_list;

/// C++ API of Distributed Autograd that kicks off the distributed backward pass
/// using the provided roots. This currently implements the
/// :ref:`fast-mode-algorithm` which assumes all RPC messages sent in the same
/// distributed autograd context across workers would be part of the autograd
/// graph during the backward pass.
///
/// We use the provided roots to discover the autograd graph and compute
/// appropriate dependencies. This method blocks until the entire
/// autograd computation is done.
///
/// This function accumulates gradients in the leaves - you might need to zero
/// them before calling it.
///
/// \param context_id The autograd context id for which we should retrieve the
///                   gradients.
/// \param roots Tensors which represent the roots of the autograd computation.
///              All the tensors should be scalars.
/// \param retain_graph If `false`, the graph used to compute the grad will be
///                     freed. Note that in nearly all cases setting this
///                     option to `true` is not needed and often can be worked
///                     around in a much more efficient way. Usually, you need
///                     to set this to `true` to run backward multiple times.
TORCH_API void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph = false);

} // namespace autograd
} // namespace distributed
} // namespace torch


注释：


#pragma once

// 包含分布式自动求导的上下文容器头文件
#include <torch/csrc/distributed/autograd/context/container.h>
// 包含分布式自动求导的引擎头文件
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

// 定义 torch 命名空间下的 distributed 命名空间和 autograd 命名空间
namespace torch {
namespace distributed {
namespace autograd {

// 使用 torch::autograd::variable_list 作为变量列表类型的别名
using torch::autograd::variable_list;

/// 分布式自动求导的 C++ API，启动使用提供的根节点进行分布式反向传播。
/// 当前实现了“快速模式算法”，假设在同一分布式自动求导上下文中发送的所有 RPC 消息
/// 在反向传播过程中都将成为自动求导图的一部分。
///
/// 我们使用提供的根节点来发现自动求导图并计算适当的依赖关系。此方法会阻塞，直到整个
/// 自动求导计算完成。
///
/// 此函数会累积叶子节点的梯度 - 在调用之前可能需要将它们清零。
///
/// \param context_id 我们应该检索梯度的自动求导上下文 ID。
/// \param roots 表示自动求导计算根节点的张量。所有张量应该是标量。
/// \param retain_graph 如果为 `false`，用于计算梯度的图将被释放。
///                     注意，在几乎所有情况下，设置此选项为 `true` 都是不必要的，
///                     而且通常可以以更高效的方式解决。通常，你需要将其设置为 `true`
///                     来多次运行反向传播。
TORCH_API void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph = false);

} // namespace autograd
} // namespace distributed
} // namespace torch
```