# `.\pytorch\torch\csrc\autograd\function.cpp`

```py
// 引入 Torch 自动求导功能模块的头文件
#include <torch/csrc/autograd/function.h>

// 引入 C10 工具中的线程本地存储功能
#include <c10/util/ThreadLocal.h>
// 引入 Torch 自动求导引擎、变量相关的头文件
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/variable.h>

// 引入 ATen 张量库的主头文件
#include <ATen/ATen.h>

// 引入标准库的相关头文件
#include <memory>
#include <string>
#include <utility>
#include <vector>

// 根据条件引入 ATen 库的不同函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

// Torch 自动求导命名空间
namespace torch {
namespace autograd {

// 当前正在评估的节点，用于在异常模式下将当前节点指定为新创建节点的父节点
C10_DEFINE_TLS_static(std::shared_ptr<Node>, tls_current_evaluating_node);
#define current_evaluating_node (tls_current_evaluating_node.get())

// 节点保护器，用于在当前评估节点更改时恢复上一个评估节点
NodeGuard::NodeGuard(std::shared_ptr<Node> node)
    : last_evaluating_node_(std::move(current_evaluating_node)) {
  current_evaluating_node = std::move(node);
}
NodeGuard::~NodeGuard() {
  // 恢复之前的评估节点
  current_evaluating_node = std::move(last_evaluating_node_);
}

// 获取当前评估节点的函数
std::shared_ptr<Node> get_current_node() {
  return current_evaluating_node;
}

// 为节点分配父节点
void Node::assign_parent() {
  metadata()->assign_parent(current_evaluating_node);
}

// 获取节点名称的函数
auto Node::name() const -> std::string {
  return c10::demangle(typeid(*this).name());
}

// 获取异常元数据的函数，如果尚未创建，则创建
AnomalyMetadata* Node::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

// 静态函数，用于收集节点的函数信息到堆栈中
static void gatherFunctions(
    Node* func,
    std::vector<std::shared_ptr<Node>>& stack) {
  func->release_variables();

  // 遍历节点的下一个边，将其函数信息加入堆栈
  for (auto& edge : func->next_edges()) {
    if (edge.function.use_count() == 1) {
      stack.emplace_back(std::move(edge.function));
    } else {
      edge.function.reset();
    }
  }
}

/*
 * 修复问题 #5534：防止深度计算图删除时的堆栈溢出
 *
 * 有时候会出现非常大的计算图节点和边。每个 std::shared_ptr<Node> 包含一系列 Edge，
 * 每个 Edge 包含一个 std::shared_ptr<Node>。删除一个 std::shared_ptr<Node> 可能会触发
 * 递归删除其他 std::shared_ptr<Node>：如果图足够深，这可能会导致堆栈溢出。这里是这种图的一个例子：
 *
 * shared_ptr<Node> -> Edge -> shared_ptr<Node> -> Edge -> ... ->
 * shared_ptr<Node>
 *
 * 解决方案是检测当我们正在递减一个 Node 的最后引用时，并且在这样做时缓冲将递归递减的 Node。
 * 然后我们可以递减（和释放）原始 Node 而不会引发递归级联，然后在逐个清空缓冲时应用相同的行为。
 * 实质上，这是将递归转换为循环，使用堆缓冲区代替递归调用堆栈。
 */
void deleteNode(Node* function) {
  // To avoid stack overflow on large computational graphs,
  // we need to track reference decrementing and freeing
  // on the heap.
  function->release_variables(); // 调用Node对象的release_variables方法，释放变量资源
  std::vector<std::shared_ptr<Node>> stack; // 创建一个存放Node对象的共享指针的vector，用于追踪引用关系
  gatherFunctions(function, stack); // 调用gatherFunctions函数，将function及其相关函数添加到stack中
  delete function; // 释放function指向的Node对象

  while (!stack.empty()) {
    auto func = std::move(stack.back()); // 从stack中取出最后一个Node对象的共享指针
    stack.pop_back(); // 移除stack中的最后一个元素
    gatherFunctions(func.get(), stack); // 调用gatherFunctions函数，将func指向的Node对象及其相关函数添加到stack中
    // Reference count is decremented on the loop backedge.
    // 在循环的回边，引用计数被递减
  }
}

at::Tensor TypeAndSize::zeros() {
  return at::zeros_symint(sym_sizes, options); // 调用at命名空间下的zeros_symint函数创建一个大小为sym_sizes的零张量，并返回
}

} // namespace autograd
} // namespace torch
```