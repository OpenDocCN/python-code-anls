# `.\pytorch\torch\csrc\jit\passes\insert_guards.cpp`

```
// 包含 Torch 的 JIT 模块中的头文件：插入守卫的 passes 和 profiling_record
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

// 包含需要使用的标准库头文件
#include <memory>           // C++ 标准库中的智能指针和内存管理工具
#include <unordered_set>    // C++ 标准库中的无序集合

// Torch 的命名空间：jit
namespace torch {
namespace jit {

// 定义 GuardInserter 结构体
struct GuardInserter {
  // 构造函数：接收一个 shared_ptr 指向图的对象，并初始化 graph_ 成员
  GuardInserter(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  // 主要执行函数 run()，用于插入守卫节点和移除分析节点
  void run() {
    // 在图的块中插入守卫节点
    insertGuards(graph_->block());
    // 移除图的块中的分析节点
    ProfilingRecord::removeProfilingNodes(graph_->block());
  }

 private:
  // 插入守卫节点的私有方法，递归地处理节点和它们的块
  void insertGuards(Block* b) {
    // 遍历块中的每个节点
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      // 如果节点类型为 prim::profile
      if (n->kind() == prim::profile) {
        // 获取 profiled_type 属性，并尝试转换为 TensorType 类型
        auto pttp = n->ty(attr::profiled_type)->cast<TensorType>();
        // 如果成功转换为 TensorType
        if (pttp) {
          // 创建一个 Guard 节点，保护当前节点的输入
          auto guard = graph_->create(prim::Guard, {n->input()}, 1);
          auto go = guard->output();
          // 设置 Guard 节点的输出类型为 profiled_type
          go->setType(pttp);
          // 将 Guard 节点插入到当前节点之前
          guard->insertBefore(n);
          // 替换当前节点的所有使用为 Guard 节点的输出
          n->output()->replaceAllUsesWith(go);
        } else {
          // 如果没有找到 profiled_type，保留原始输入，替换当前节点的所有使用
          n->output()->replaceAllUsesWith(n->input());
        }
        // 销毁当前节点
        it.destroyCurrent();
      } else {
        // 如果节点不是 profile 类型，则递归处理其所有块
        for (Block* ib : n->blocks()) {
          insertGuards(ib);
        }
      }
    }
  }

  // 存储图的 shared_ptr
  std::shared_ptr<Graph> graph_;
};

// 插入守卫的公共函数，创建 GuardInserter 对象并运行插入守卫的操作
void InsertGuards(std::shared_ptr<Graph> graph) {
  GuardInserter gi(std::move(graph));
  gi.run();
}

} // namespace jit
} // namespace torch
```