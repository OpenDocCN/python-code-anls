# `.\pytorch\torch\csrc\jit\passes\utils\optimization_utils.cpp`

```
#include <torch/csrc/jit/passes/utils/optimization_utils.h>

namespace torch {
namespace jit {

// 检查节点 n 的参数是否都是非常量
bool nonConstantParameters(Node* n) {
  // 从第二个参数开始，逐个检查是否为常量节点
  for (size_t i = 1; i < n->inputs().size(); i++) {
    // 如果某个参数不是常量节点，则返回 true
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  // 如果所有参数都是常量节点，则返回 false
  return false;
}

} // namespace jit
} // namespace torch
```