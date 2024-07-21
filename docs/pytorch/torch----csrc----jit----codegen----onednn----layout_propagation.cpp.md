# `.\pytorch\torch\csrc\jit\codegen\onednn\layout_propagation.cpp`

```py
// 包含 Torch 的头文件：onednn/graph_helper.h, onednn/layout_propagation.h, jit_log.h
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/codegen/onednn/layout_propagation.h>
#include <torch/csrc/jit/jit_log.h>

// 定义 Torch 的命名空间
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 定义静态函数 LayoutPropagation，接受一个 Node* 参数
static void LayoutPropagation(Node* n) {
  // 如果节点不是 LLGA 子图，则直接返回
  if (!LlgaGraphHelper::isLlgaSubgraph(n))
    return;

  // 如果节点没有 output_layouts 属性，则进行初始化
  if (!n->hasAttribute(attr::output_layouts)) {
    // 获取节点输出的数量
    const auto num_output = n->outputs().size();
    // 打印调试信息，显示初始化 output_layouts 的大小
    GRAPH_DEBUG("Initial output_layouts of size ", num_output);
    // 创建一个包含 STRIDED_LAYOUT 的 layouts 向量
    std::vector<int64_t> layouts(num_output, STRIDED_LAYOUT);
    // 将 layouts 向量赋值给节点的 output_layouts 属性
    n->is_(attr::output_layouts, layouts);
  }

  // 遍历节点的输入
  for (auto input : n->inputs()) {
    // 获取输入节点
    auto prev = input->node();
    // 获取输入的偏移量
    auto offset = input->offset();
    // 如果输入节点是 LLGA 子图
    if (LlgaGraphHelper::isLlgaSubgraph(prev)) {
      // 默认使用不透明布局
      bool useOpaqueLayout = true;
      // 检查输入的每个使用情况
      for (auto& use : input->uses()) {
        // 如果使用节点不是 LLGA 子图，则取消使用不透明布局
        if (!LlgaGraphHelper::isLlgaSubgraph(use.user)) {
          useOpaqueLayout = false;
          break;
        }
      }
      // 如果所有使用情况都使用不透明布局，则设置输入节点的不透明布局
      if (useOpaqueLayout) {
        LlgaNodeWrapper(prev).setOpaqueLayout(offset);
      }
    }
  }
}

// 定义函数 LayoutPropagation，接受 Block* 的 ArrayRef 参数
static void LayoutPropagation(at::ArrayRef<Block*> blocks) {
  // 遍历每个 Block
  for (Block* block : blocks)
    // 遍历每个 Block 中的节点，对每个节点进行布局传播
    for (Node* node : block->nodes())
      LayoutPropagation(node);
}

// 定义函数 PropagateLayout，接受 std::shared_ptr<Graph> 的参数
void PropagateLayout(const std::shared_ptr<Graph>& graph) {
  // 对图的顶级块进行布局传播
  LayoutPropagation(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```