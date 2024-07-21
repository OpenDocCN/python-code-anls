# `.\pytorch\torch\csrc\lazy\core\trie.h`

```
#pragma once
// 预处理指令：确保此头文件仅被编译一次

#include <atomic>
// 引入原子操作库，用于多线程安全的操作

#include <list>
// 引入列表容器，用于存储 TrieNode 的后继节点列表

#include <c10/core/ScalarType.h>
// 引入 PyTorch 的标量类型定义

#include <torch/csrc/lazy/core/ir.h>
// 引入懒执行模块的 IR 定义

#include <torch/csrc/lazy/core/metrics.h>
// 引入懒执行模块的度量工具

namespace torch {
namespace lazy {

struct TORCH_API TrieNode {
  // Trie 树节点结构体定义

  static size_t GetNextUniqueId() {
    // 静态方法：获取下一个唯一 ID
    static thread_local size_t id_generator = 0;
    return id_generator++;
  }

  size_t unique_id;
  // 当前节点的唯一 ID

  size_t hit_counter;
  // 命中计数器，记录节点被访问次数

  NodePtr ir_node;
  // 指向 IR 节点的智能指针

  std::list<std::shared_ptr<TrieNode>> successors;
  // 子节点列表，存储 TrieNode 的智能指针

  TrieNode() : unique_id(GetNextUniqueId()), hit_counter(0), ir_node(nullptr) {}
  // 默认构造函数：初始化唯一 ID、命中计数器和 IR 节点为空

  explicit TrieNode(NodePtr node)
      : unique_id(GetNextUniqueId()),
        hit_counter(0),
        ir_node(std::move(node)) {}
  // 带参构造函数：初始化唯一 ID、命中计数器，并移动构造 IR 节点

};

class TORCH_API TrieCache {
  // Trie 缓存类定义

 public:
  static TrieCache* Get();
  // 静态方法：获取 TrieCache 实例的指针

  TrieNode* Current() const;
  // 返回当前节点的指针

  void SetCurrent(std::list<std::shared_ptr<TrieNode>>::iterator& iter);
  // 设置当前节点，通过迭代器在后继节点列表中移动以实现 LRU 缓存效果

  void ResetCurrent();
  // 重置当前节点，用于标记步骤结束时的操作

  void Insert(NodePtr ir_node);
  // 创建一个新的 TrieNode 并插入到 TrieCache 中

  void Clear();
  // 清空所有 TrieCache 节点
  // TODO: 因为不希望用户通过 Python API 明确调用此函数，可能需要引入缓存大小的阈值来避免持有张量时间过长。

  void DumpToDotFile(const std::string& file_name);
  // 将 TrieCache 转储到指定的 DOT 文件中

 private:
  TrieCache();
  // 私有构造函数：用于初始化 TrieCache

  std::shared_ptr<TrieNode> root_;
  // 根节点的智能指针

  TrieNode* current_;
  // 当前节点的指针
};

template <typename T, typename... Args>
NodePtr LookupNodeFromTrieCache(Args&&... args) {
  // 模板函数：从 TrieCache 中查找节点

  auto& successors = TrieCache::Get()->Current()->successors;
  // 获取当前 TrieCache 的后继节点列表的引用

  for (auto it = successors.begin(); it != successors.end(); it++) {
    // 遍历后继节点列表

    NodePtr ir_node = (*it)->ir_node;
    // 获取当前后继节点的 IR 节点指针

    const T* concrete_node = NodeCast<T>(ir_node.get());
    // 尝试将 IR 节点转换为指定类型 T 的指针

    if (concrete_node &&
        concrete_node->CanBeReused(std::forward<Args>(args)...)) {
      // 如果转换成功且 IR 节点可以重用给定参数...

      TORCH_LAZY_COUNTER(
          "IrNodeReused_" + c10::demangle((typeid(T).name())), 1);
      // 记录 IR 节点重用的计数器

      (*it)->hit_counter++;
      // 增加当前后继节点的命中计数

      TrieCache::Get()->SetCurrent(it);
      // 设置当前节点为此后继节点

      return ir_node;
      // 返回找到的 IR 节点指针
    }
  }
  return nullptr;
  // 若未找到可重用的 IR 节点，则返回空指针
}

} // namespace lazy
} // namespace torch
```