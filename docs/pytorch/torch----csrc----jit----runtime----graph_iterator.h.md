# `.\pytorch\torch\csrc\jit\runtime\graph_iterator.h`

```py
    // 包含 Torch IR 的头文件
    #include <torch/csrc/jit/ir/ir.h>

    // 命名空间 torch::jit
    namespace torch::jit {

    // 这个类用于在图中进行深度优先遍历所有节点。
    class DepthFirstGraphNodeIterator {
      Node* current_;  // 当前节点指针

     public:
      // 构造函数，初始化迭代器为图中第一个节点
      explicit DepthFirstGraphNodeIterator(std::shared_ptr<Graph>& graph)
          : current_(*(graph->block()->nodes().begin())) {}

      // 向上移动到下一个节点（可能会递归向上）
      void move_up() {
        if (current_ == nullptr) {
          return;
        }
        // 基本上我们从子块（即 current_）开始，
        // 尝试找到拥有它的块。现在我们需要检查
        // 如果该块是图的根块，或者它是一个 If/Loop/etc
        // 块。
        //
        // 如果它是图的根块，我们可以停止，因为没有“向上”的概念，
        // 但如果它是一个节点（例如 If/Loop/etc），我们需要根据我们来自哪里
        // 的逻辑移动到下一个块。
        // 这可能意味着我们需要再次向上递归遍历（例如，如果我们已经
        // 到达 if 块中 else 子句的末尾，我们需要向上到包含 if 的父块。
        //
        // 同样，如果我们已经到达包含 else 子句的父块的末尾，
        // 我们可能需要再次向上，因此这是一个递归函数。
        //
        //              BlockNode (if/loop/with)
        //                       |
        //            [Block1]  ... [Block2]
        //                |
        //   [ Node1, Node2, Node3, FromNode]
        //
        auto parent_block = current_->owningBlock();
        TORCH_INTERNAL_ASSERT(parent_block, "Every node must be owned by a block");

        // 获取拥有父块的节点。此节点必须是 if、loop 或 with。
        auto parent_node = parent_block->owningNode();
        if (parent_node == nullptr) {
          // 如果没有拥有当前块的节点，则表示我们位于图的顶部，
          // 由于我们试图向上移动，因此已经到达遍历的末尾。
          current_ = nullptr;
          return;
        }

        // 检查此根节点的类型。
    if (parent_node->kind() == prim::If) {
      // 如果父节点是 If 类型
      auto* then_block = parent_node->blocks().at(0);  // 获取 If 节点的第一个块
      auto* else_block = parent_node->blocks().at(1);  // 获取 If 节点的第二个块

      if (parent_block == else_block) {
        // 如果当前块是 else 块，则移动到父块中的下一个节点
        current_ = parent_node->next();
        if (current_->kind() == prim::Return) {
          move_up();  // 如果下一个节点是 Return，则向上移动
        }
      } else {
        // 如果当前块是 then 块，则移动到 else 块（如果不为空）
        TORCH_INTERNAL_ASSERT(parent_block == then_block);
        bool else_block_empty =
            else_block->nodes().begin() == else_block->nodes().end();

        if (!else_block_empty) {
          current_ = *(else_block->nodes().begin());  // 移动到 else 块的第一个节点
        } else {
          // 如果 else 块为空，则移动到父节点的下一个节点
          current_ = parent_node->next();
          if (current_->kind() == prim::Return) {
            move_up();  // 如果下一个节点是 Return，则向上移动
          }
        }
      }
    } else if (
        parent_node->kind() == prim::Loop ||
        parent_node->kind() == prim::With) {
      // 如果父节点是 Loop 或 With 类型，则移动到父节点的下一个节点
      current_ = parent_node->next();
      if (current_->kind() == prim::Return) {
        move_up();  // 如果下一个节点是 Return，则向上移动
      }
    } else {
      // 如果父节点类型不是 If、Loop、或 With，则抛出错误
      TORCH_INTERNAL_ASSERT(
          false, "Only if/loop/with nodes should have child blocks");
    }
  }

  // 移动到相邻的下一个节点或者向上移动到父节点
  void move_next() {
    if (current_ == nullptr) {
      return;
    }

    // 移动到当前块中的下一个节点
    current_ = current_->next();

    // 检查是否到达块的末尾，如果是，则向上移动
    if (current_->kind() == prim::Return) {
      move_up();  // 如果下一个节点是 Return，则向上移动
    }
  }

  // 移动到节点的子节点（如果存在）
  void move_into() {
    if (current_ == nullptr) {
      return;
    }

    // 检查当前节点是否包含子节点
    if (current_->kind() == prim::If || current_->kind() == prim::Loop ||
        current_->kind() == prim::With) {
      auto* first_block = current_->blocks().at(0);
      current_ = first_block->param_node();  // 移动到第一个块的参数节点
      // move_next 将根据块是否为空来处理 If、Loop 和 With 块之间的差异
      move_next();  // 移动到下一个节点
    } else {
      move_next();  // 如果没有子节点，则移动到下一个节点
    }
  }

  // 获取图中的下一个节点。如果没有节点可返回则返回 nullptr。
  Node* next() {
    auto result = current_;

    // 尝试移动到现有节点的子节点，设置要返回的下一个节点
    // 如果不可能，则移动到下一个节点，或者向上并移动到下一个节点
    move_into();

    return result;  // 返回当前节点
  }
};

} // namespace torch::jit


// 结束了在命名空间 torch::jit 中的定义
};
// 闭合命名空间声明，用于限定作用域
} // namespace torch::jit
```