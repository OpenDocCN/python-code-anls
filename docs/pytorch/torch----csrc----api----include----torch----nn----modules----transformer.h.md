# `.\pytorch\torch\csrc\api\include\torch\nn\modules\transformer.h`

```py
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/options/transformer.h>
#include <torch/nn/pimpl.h>

#include <torch/types.h>

#include <ostream>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transformer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Transformer 模型。用户可以根据需要修改属性。该架构基于论文 "Attention Is All You Need"。
/// Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
/// Gomez, Lukasz Kaiser, 和 Illia Polosukhin. 2017. Attention is all you need.
/// 在 Advances in Neural Information Processing Systems 中，页面 6000-6010。
///
/// 参考 https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html 了解该 Transformer 模型的确切行为。
///
/// 参考 `torch::nn::Transformer` 类的文档了解该编码器层模型支持的构造函数参数。
///
/// 示例：
/// ```
/// Transformer trans(TransformerOptions(512, 8));
/// ```py
};

/// `TransformerImpl` 的 `ModuleHolder` 子类。
/// 参考 `TransformerImpl` 类的文档了解其提供的方法，以及如何使用 `Transformer` 和 `torch::nn::TransformerOptions` 的示例。
/// 参考 `ModuleHolder` 的文档了解 PyTorch 的模块存储语义。
TORCH_MODULE(Transformer);

} // namespace nn
} // namespace torch
```