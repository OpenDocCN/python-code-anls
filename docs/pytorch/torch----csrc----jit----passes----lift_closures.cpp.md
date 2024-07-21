# `.\pytorch\torch\csrc\jit\passes\lift_closures.cpp`

```py
// 包含 Torch 的头文件 lift_closures.h，用于闭包的提升操作
#include <torch/csrc/jit/passes/lift_closures.h>

// 包含 Torch 的 IR emitter 头文件，用于生成 IR
#include <torch/csrc/jit/frontend/ir_emitter.h>

// 包含 Torch 的 IR 头文件，定义了 IR 的结构
#include <torch/csrc/jit/ir/ir.h>

// 包含 C++ 标准库中的实用工具
#include <utility>

// Torch 的命名空间开始
namespace torch {
namespace jit {

// 提升闭包函数的实现
static void liftClosure(Node* closure) {
  // 获取闭包节点的第一个块
  auto block = closure->blocks().at(0);
  // 创建一个新的子图用于存放闭包
  auto subgraph = std::make_shared<Graph>();
  // 获取闭包节点所属的图
  auto g = closure->owningGraph();
  // 在闭包后插入一个创建元组的节点，用于打包闭包的上下文
  Node* pack_context =
      g->create(prim::TupleConstruct, {}, 1)->insertAfter(closure);
  // 在子图中添加一个输入，代表闭包的上下文
  Value* context = subgraph->addInput("context");
  // 插入一个元组解包节点用于解包闭包的上下文
  Node* unpack_context =
      subgraph->insertNode(subgraph->create(prim::TupleUnpack, {context}, 0));

  // 用于存放捕获变量的映射表
  std::unordered_map<Value*, Value*> captures;
  // 用于获取变量的闭包环境
  auto env = [&](Value* v) -> Value* {
    auto it = captures.find(v);
    if (it != captures.end()) {
      return it->second;
    }
    // 将变量作为输入添加到打包上下文节点中
    pack_context->addInput(v);
    // 复制变量的元数据，并将其作为解包上下文节点的输出
    Value* r = unpack_context->addOutput()->copyMetadata(v);
    captures[v] = r;
    return r;
  };
  // 从块中克隆节点到子图中，并应用闭包环境函数
  subgraph->block()->cloneFrom(block, env);

  // 创建一个元组类型，表示闭包的上下文类型
  auto context_type = TupleType::create(
      fmap(pack_context->inputs(), [](Value* v) { return v->type(); }));
  context->setType(context_type);
  pack_context->output()->setType(context_type);

  // 创建一个新的元组节点，用于替换原始闭包节点的输出
  auto closure_tuple =
      g->create(prim::TupleConstruct, {}, 1)->insertAfter(pack_context);
  closure->output()->replaceAllUsesWith(closure_tuple->output());

  // 将闭包节点的输出和打包上下文节点的输出作为元组节点的输入
  closure_tuple->addInput(closure->output());
  closure_tuple->addInput(pack_context->output());
  // 设置元组节点的输出类型
  closure_tuple->output()->setType(
      TupleType::create({closure->output()->type(), std::move(context_type)}));

  // 删除闭包节点的原始块
  closure->eraseBlock(0);
  // 将子图设置为闭包节点的子图属性
  closure->g_(attr::Subgraph, std::move(subgraph));

  // 运行清理操作以确保子图的一致性
  runCleanupPasses(closure->g(attr::Subgraph));
}

// 递归提升闭包函数，处理给定块中的每个节点
static void liftClosures(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    // 根据节点类型进行不同的处理
    switch (n->kind()) {
      // 如果节点是闭包节点，则调用提升闭包函数
      case prim::Closure: {
        liftClosure(n);
      } break;
      // 对于其他节点类型，递归调用 liftClosures 处理其子块
      default: {
        for (Block* b : n->blocks()) {
          liftClosures(b);
        }
      }
    }
  }
}

// 对外接口，提升给定图中的所有闭包
void liftClosures(const std::shared_ptr<Graph>& to_clean) {
  liftClosures(to_clean->block());
}

} // namespace jit
} // namespace torch
// Torch 的命名空间结束
```