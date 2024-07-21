# `.\pytorch\torch\csrc\lazy\core\ir_util.cpp`

```
// 包含 Torch 库中 IR 工具的头文件
#include <torch/csrc/lazy/core/ir_util.h>

// 包含 C10 日志记录的头文件
#include <c10/util/Logging.h>

// Torch 惰性计算命名空间
namespace torch {
namespace lazy {

// 计算给定节点的后序遍历顺序，同时更新节点的发射状态
std::vector<const Node*> Util::ComputePostOrder(
    const Node* node,
    EmissionMap* emap) {
  // 后序遍历结果列表
  std::vector<const Node*> post_order;
  // 待处理节点队列
  std::vector<const Node*> queue;
  queue.push_back(node);
  while (!queue.empty()) {
    node = queue.back();
    // 查找节点在发射映射中的状态
    auto it = emap->find(node);
    if (it == emap->end()) {
      (*emap)[node] = kEmitting;  // 标记节点为发射中状态
      // 处理节点的每一个操作数
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          queue.push_back(output.node);  // 将操作数加入队列
        } else {
          // 检测到图中的循环，输出错误信息
          TORCH_CHECK(
              oit->second != kEmitting,
              "Graph loop found at ",
              output.node->ToString());
        }
      }
    } else if (it->second == kEmitting) {
      // 处理节点已处于发射中状态的情况
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        // 检查操作数节点是否已经发射完成
        TORCH_CHECK(
            oit != emap->end() && oit->second == kEmitted,
            "Graph loop found at ",
            output.node->ToString());
      }
      (*emap)[node] = kEmitted;  // 标记节点为已发射状态
      post_order.push_back(node);  // 将节点加入后序遍历结果列表
      queue.pop_back();  // 移除队列中的节点
    } else {
      TORCH_CHECK(it->second == kEmitted);
      queue.pop_back();  // 移除队列中的节点
    }
  }
  return post_order;  // 返回后序遍历结果列表
}

// 计算给定节点集合的后序遍历顺序，同时更新节点的发射状态
std::vector<const Node*> Util::ComputePostOrder(
    c10::ArrayRef<const Node*> nodes,
    EmissionMap* emap) {
  std::vector<const Node*> post_order;
  // 对每个节点进行后序遍历计算
  for (auto node : nodes) {
    auto node_post_order = ComputePostOrder(node, emap);
    post_order.insert(
        post_order.end(), node_post_order.begin(), node_post_order.end());
  }
  return post_order;  // 返回总体后序遍历结果列表
}

// 计算给定节点集合的后序遍历顺序，使用默认的发射映射
std::vector<const Node*> Util::ComputePostOrder(
    c10::ArrayRef<const Node*> nodes) {
  EmissionMap emap;
  return ComputePostOrder(nodes, &emap);  // 调用带映射参数的 ComputePostOrder
}

// 获取给定节点集合的图大小，即后序遍历顺序列表的长度
size_t Util::GetGraphSize(c10::ArrayRef<const Node*> nodes) {
  return ComputePostOrder(nodes).size();  // 返回后序遍历顺序列表的长度
}

} // namespace lazy
} // namespace torch
```