# `.\pytorch\torch\csrc\jit\passes\graph_fuser.h`

```
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// 定义一个公共的 API 函数，用于检查是否可以在 CPU 上进行传统的融合操作
TORCH_API bool canFuseOnCPULegacy();

// 定义一个公共的 API 函数，用于覆盖是否可以在 CPU 上进行传统的融合操作的设置
TORCH_API void overrideCanFuseOnCPULegacy(bool value);

// 注意：在进行融合之前，请务必运行死代码消除（DCE），因为死代码可能会阻止融合的机会被利用。
// 在 Windows 平台上此操作不会执行，尚未实现
TORCH_API void FuseGraph(
    std::shared_ptr<Graph>& graph,
    bool strict_fuser_check = false);

// \brief 使用节点级回调自定义融合过程，确定子图中节点的包含情况。
//
// 此辅助功能会排除有别名的输入以及控制流边界之间的融合。
//
// \arg graph 需要就地修改的图
// \arg is_fusable 对图中每个可融合节点运行的回调函数
// \arg kind 生成的融合子图的标签
// \arg arg_limit 生成的融合子图应具有的最大参数数目。注意：这可能会成为融合子图的一般后置条件。
TORCH_API void CustomFuseGraph(
    std::shared_ptr<Graph>& graph,
    const std::function<bool(Node*)>& is_fusable,
    Symbol kind,
    size_t arg_limit = std::numeric_limits<size_t>::max());

} // namespace jit
} // namespace torch
```