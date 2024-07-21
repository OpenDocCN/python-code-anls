# `.\pytorch\torch\csrc\jit\passes\quantization\dedup_module_uses.h`

```py
#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

/** 
 * 递归地去重使用同一个模块的多个实例。
 * 对于每个使用该模块的地方创建一个实例克隆，这意味着类型保持不变，
 * 所有属性被复制，然后我们将原始模块的使用改为克隆模块在图中的使用。
 * 
 * 这样做是为了确保模块可以在经过破坏性变换后仍然保持模型行为不变。
 * 例如，在以下代码中：
 * 
 *   x = self.conv1(x)
 *   x = self.relu(x)
 *   x = self.conv2(x)
 *   x = self.relu(x)
 * 
 * self.relu 需要去重，以便未来的破坏性变换能够正确工作。
 */
TORCH_API void DedupModuleUses(Module& module);

} // namespace jit
} // namespace torch
```