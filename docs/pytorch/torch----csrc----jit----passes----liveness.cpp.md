# `.\pytorch\torch\csrc\jit\passes\liveness.cpp`

```
// 包含 Torch 的 JIT 环境中的头文件：生存期分析
#include <torch/csrc/jit/passes/liveness.h>

// 包含 Torch 的 JIT 环境中的头文件：别名分析
#include <torch/csrc/jit/ir/alias_analysis.h>

// 包含 Torch 的 JIT 环境中的头文件：IR 视图
#include <torch/csrc/jit/ir/ir_views.h>

// 包含 Torch 的 JIT 环境中的头文件：常量池优化
#include <torch/csrc/jit/passes/constant_pooling.h>

// 标准输入输出流操作
#include <iostream>

// C++ 标准库中的内存管理
#include <memory>

// Torch 命名空间
namespace torch {
namespace jit {

// LivenessAnalyzer 类定义，用于计算 "bailout" 生存期
struct LivenessAnalyzer {
  // 构造函数，接受共享指针指向的图对象作为参数
  explicit LivenessAnalyzer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), changed_(false) {}

  // 运行生存期分析，返回节点到值的向量映射
  std::unordered_map<Node*, std::vector<Value*>> run() {
    // 计数器节点的集合
    std::vector<Node*> counters;
    // 向基本块中插入显式使用循环计数器的操作
    insertExplicitUsesOfLoopCounters(graph_->block(), counters);

    // 实现经典的固定点生存期分析
    // 分析运行直到不再有节点生存期集合变化
    do {
      changed_ = false;
      processBlock(graph_->block(), SparseBitVector{});
    } while (changed_);

    // 移除计数器节点
    removeCounterNodes(counters);

    // 构造并返回节点到值的向量映射结果
    std::unordered_map<Node*, std::vector<Value*>> result;
    for (const auto& e : liveness_sets_) {
      result.insert({e.first, toValueVector(e.second)});
    }
    return result;
  }

  // 向基本块中插入循环计数器的显式使用
  void insertExplicitUsesOfLoopCounters(
      Block* b,
      std::vector<Node*>& counters) {
    for (auto it : b->nodes()) {
      if (it->kind() == prim::Loop) {
        // 获取循环视图
        LoopView lv(it);
        // 在循环体块中设置插入点
        WithInsertPoint guard(lv.bodyBlock());
        // 创建并插入当前循环次数存储节点
        auto ctc = graph_->create(prim::Store, {lv.currentTripCount()}, 0);
        graph_->insertNode(ctc);
        counters.push_back(ctc);
        // 创建并插入最大循环次数存储节点
        auto mtc = graph_->create(prim::Store, {lv.maxTripCount()}, 0);
        graph_->insertNode(mtc);
        counters.push_back(mtc);
      }

      // 递归处理基本块中的子块
      for (auto ib : it->blocks()) {
        insertExplicitUsesOfLoopCounters(ib, counters);
      }
    }
  }

  // 移除计数器节点
  void removeCounterNodes(std::vector<Node*>& counters) {
    for (auto n : counters) {
      n->destroy();
    }
  }

  // 打印生存期信息和图结构
  void dump(
      const std::unordered_map<Node*, std::vector<Value*>>& liveness_sets) {
    std::cout << "Liveness info:\n";
    for (auto e : liveness_sets) {
      if (!e.first->outputs().empty()) {
        std::cout << e.first->outputs()[0]->debugName();
      }

      std::cout << " " << e.first->kind().toQualString();
      std::cout << " = ";
      dump(e.second);
      std::cout << std::endl;
    }
    std::cout << "graph :\n";
    graph_->dump();
  }

  // 打印值的集合
  void dump(const std::vector<Value*>& set) {
    bool first = true;
    std::cout << "[";
    for (auto el : set) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << el->debugName() << "(" << el->unique() << ")";
    }
    std::cout << "]";
  }

 private:
  // 将值的数组转换为稀疏位向量
  SparseBitVector toSparseBitVector(at::ArrayRef<Value*> values) {
    SparseBitVector sbv;
    // 实现将值的数组转换为稀疏位向量的功能
    // (此处省略了具体的实现细节)
    return sbv;
  }

  // 图对象的共享指针
  std::shared_ptr<Graph> graph_;
  // 表示分析是否发生变化的标志
  bool changed_;
  // 节点到生存期值集合的映射
  std::unordered_map<Node*, std::vector<Value*>> liveness_sets_;
};

} // namespace jit
} // namespace torch
  // 遍历输入的值的集合，并为每个值的唯一标识设置映射关系，同时更新稀疏位向量
  for (auto v : values) {
    ids_to_values_[v->unique()] = v;
    sbv.set(v->unique());
  }
  // 返回更新后的稀疏位向量
  return sbv;
}

// 将稀疏位向量转换为值的指针向量
std::vector<Value*> toValueVector(const SparseBitVector& sbv) {
  std::vector<Value*> vec;
  // 根据稀疏位向量中的标识符获取对应的值指针，并存入向量中
  for (auto id : sbv) {
    vec.push_back(ids_to_values_[id]);
  }
  // 返回值指针向量
  return vec;
}

// 处理代码块，计算活跃变量信息
SparseBitVector processBlock(Block* b, SparseBitVector liveness) {
  // 获取代码块的输出作为使用（uses）
  auto block_outputs = toSparseBitVector(b->outputs());
  // 更新活跃变量信息，使用位或运算
  liveness |= block_outputs;

  SparseBitVector defs;
  // 逆序遍历代码块中的每个节点
  for (Node* it : b->nodes().reverse()) {
    // 将当前节点的输出标记为失效（kill outputs）
    liveness -= toSparseBitVector(it->outputs());
    // 如果节点为循环节点
    if (it->kind() == prim::Loop) {
      LoopView lv(it);
      // 注意：合并来自循环头部的变化
      auto loop_header = *lv.bodyBlock()->nodes().begin();
      auto loop_block = liveness | liveness_sets_[loop_header];
      // 递归处理循环体代码块，更新循环体中的活跃变量信息
      loop_block = processBlock(lv.bodyBlock(), loop_block);
      // 将循环体的输入变量标记为失效
      loop_block -= toSparseBitVector(lv.bodyBlock()->inputs());
      liveness |= loop_block;
    } else if (it->kind() == prim::If) {
      IfView iv(it);
      // 处理条件语句的真分支和假分支
      auto true_liveness = processBlock(iv.thenBlock(), liveness);
      auto false_liveness = processBlock(iv.elseBlock(), liveness);
      // 将真分支和假分支的活跃变量信息合并到当前活跃变量信息中
      liveness |= true_liveness;
      liveness |= false_liveness;
    }
    // 将节点的输入变量标记为活跃
    liveness |= toSparseBitVector(it->inputs());
    // 记录是否有新的活跃变量被设置
    auto changed = liveness_sets_[it] |= liveness;
    // 更新代码块的改变标志
    changed_ = changed_ | changed;
  }
  // 返回更新后的活跃变量信息
  return liveness;
}

std::shared_ptr<Graph> graph_;
bool changed_;
std::map<Node*, SparseBitVector> liveness_sets_;
std::map<size_t, Value*> ids_to_values_;
};

// 结束命名空间 torch
} // namespace torch
// 结束命名空间 jit
} // namespace jit
```