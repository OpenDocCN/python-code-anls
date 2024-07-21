# `.\pytorch\torch\csrc\jit\runtime\interpreter\preprocess_graph.cpp`

```py
// 引入 Torch 库中的预处理图模块
#include <torch/csrc/jit/runtime/interpreter/preprocess_graph.h>

// 引入 Torch 前端模块中的模式匹配库
#include <torch/csrc/jit/frontend/schema_matching.h>

// 引入 Torch 解释器模块中的内联发射判断功能
#include <torch/csrc/jit/runtime/interpreter/can_emit_inline.h>

// 定义 torch::jit::interpreter 命名空间
namespace torch::jit::interpreter {

// 匿名命名空间，用于包裹下面的 insertEnterMethodCalls 函数
namespace {

// 在 prim::Enter 节点之后插入显式的 prim::MethodCall 节点，
// 以实际调用对象的 __enter__ 方法。prim::Enter 节点仅将对象推入当前已进入对象的堆栈中。
// 这是必要的，因为对 prim::Enter 节点发出两条指令（一条 ENTER 将其推入已进入对象的堆栈，一条 CALL 调用 __enter__）不起作用；
// 决定何时将值从寄存器中移出的计算是基于 IR 中它的使用次数。
void insertEnterMethodCalls(Graph& g) {
  // 用于存储要处理的块的队列
  std::vector<Block*> block_queue;
  // 存储所有 prim::Enter 节点的队列
  std::vector<Node*> enter_nodes;
  // 将当前图的根块放入队列中
  block_queue.emplace_back(g.block());

  // 遍历图，深入节点所属的块，并将遇到的所有 prim::Enter 节点添加到 enter_nodes 中
  while (!block_queue.empty()) {
    Block* block = block_queue.back();
    block_queue.pop_back();

    // 遍历块中的每个节点
    for (auto node : block->nodes()) {
      // 如果节点的类型为 prim::Enter，则将其添加到 enter_nodes 中
      if (node->kind() == prim::Enter) {
        enter_nodes.emplace_back(node);
        continue;
      }

      // 将节点的所有子块加入到处理队列中
      for (auto& node_block : node->blocks()) {
        block_queue.emplace_back(node_block);
      }
    }
  }

  // 对每个 prim::Enter 节点，生成一个在其后调用实际的 prim::MethodCall 节点来调用 __enter__ 方法。
  for (auto& enter : enter_nodes) {
    // 获取输入节点的类类型
    auto cls = enter->input(0)->type()->expect<ClassType>();

    // 匹配方法模式
    MatchedSchema enter_matched_schema = matchSchema(
        cls->findMethod("__enter__")->getSchema(),  // 获取 __enter__ 方法的模式
        enter->input(0)->node()->sourceRange(),     // 获取输入节点的源范围
        g,                                         // 当前图
        {enter->input(0)},                         // 输入值列表
        {});                                       // 额外参数列表

    // 插入一个方法调用节点 "__enter__"
    Node* call = g.insertMethodCall("__enter__", enter_matched_schema)->node();
    // 将新生成的调用节点移到 prim::Enter 节点的后面
    call->moveAfter(enter);
    // 替换 prim::Enter 节点的所有使用点为新生成的调用节点
    enter->replaceAllUsesWith(call);
  }
}

// 向块中插入 Drop 节点，以清除未使用的引用：
// 这可能发生在一些情况下，例如当节点返回多个值但只有一个被使用时。
// 例如：a, b = foo()
//      return a
void dropUnused(Block* b) {
  auto createDropIfUnused = [&](ArrayRef<Value*> values) -> Node* {
    std::vector<Value*> to_drop;
    for (auto v : values) {
      // 如果值没有使用并且不是常量节点，则将其加入待删除列表
      if (v->uses().empty() && v->node()->kind() != prim::Constant) {
        to_drop.push_back(v);
      }
    }
    // 如果待删除列表为空，则返回 nullptr
    if (to_drop.empty()) {
      return nullptr;
    }
    // 创建一个 Drop 节点，并将其作为块的第一个节点插入
    return b->owningGraph()->create(prim::Drop, to_drop, 0);
  };

  // 如果存在未使用的输入值，则在块的开头插入一个 Drop 节点
  if (auto d = createDropIfUnused(b->inputs())) {
    b->prependNode(d);
  }

  // 遍历块中的每个节点
  for (auto n : b->nodes()) {
    // 对每个节点的输出值调用 createDropIfUnused，如果存在未使用的输出值，则在相应节点后插入 Drop 节点
    if (auto d = createDropIfUnused(n->outputs())) {
      d->insertAfter(n);
    }

    // 递归调用 dropUnused 函数，处理当前节点的所有子块
    for (auto block : n->blocks()) {
      dropUnused(block);
    }
  }
}

// 确保每个值在其定义所在的块中有最终使用。
// 对于大多数节点，这已经是真实的。但有以下几种例外情况：
// 1. 一个未使用的值。
// 2. A value whose last use is nested in some control flow.
// For (1) we simply add a prim::Drop node that uses the value right after
// it is defined. For (2), we insert a prim::Drop right after the control
// flow node where the last use occurs
void insertLastUses(Graph& g) {
  // struct to share common data structures
  struct InsertLastUses {
    Graph& graph;
    // have we seen this value, yet, if not, it is the last use of the value
    std::unordered_set<Value*> seen;

    // A map from an If or Loop node to the optional Drop block that
    // occurs directly after it to release any tensors that go out of scope
    // when the If/Loop exits. These are created and inserted on demand.
    std::unordered_map<Node*, Node*> drop_for_node;

    // Constructor initializing with the graph
    explicit InsertLastUses(Graph& g) : graph(g) {
      scanBlock(graph.block());
    }

    // Scan a block by processing its return node and nodes in reverse order
    void scanBlock(Block* b) {
      scanNode(b->return_node());
      for (auto n : b->nodes().reverse()) {
        scanNode(n);
      }
    }

    // Scan a node by examining its blocks and then its inputs
    void scanNode(Node* n) {
      // Recursively scan nested blocks
      for (auto b : n->blocks()) {
        scanBlock(b);
      }
      
      // Scan inputs in reverse order to handle multiple uses correctly
      for (size_t i = n->inputs().size(); i > 0; --i) {
        scanUse(n, i - 1);
      }
    }

    // Handle a specific use of a node's input
    void scanUse(Node* n, size_t i) {
      auto v = n->inputs()[i];
      auto inserted = seen.insert(v).second;
      if (!inserted) {
        return; // Skip if already seen
      }

      // Find the last use node at the same depth as the definition of v
      Node* same_depth_node = findOwnerInBlock(n, v->node()->owningBlock());
      AT_ASSERT(
          same_depth_node); // Assertion failure means v is not in scope for n, use lint!

      // If v and n are in the same block, it's already the final use
      if (same_depth_node == n) {
        return;
      }

      // Add a Drop node after the block containing the last use of v
      addToDropIfNotExists(
          findOrCreateDropInstructionForNode(same_depth_node), v);
    }

    // Find the node in a block that contains a given node 'n'
    // Returns nullptr if no such node exists
    // Example: findOwnerInBlock(n2, n0.block()) == n1
    Node* findOwnerInBlock(Node* n, Block* block) {
      // Implementation details omitted for brevity
    }
  };
}
    // 在指定的块中查找节点的所有者节点
    Node* findOwnerInBlock(Node* n, Block* block) {
      // 当节点 n 不为空且不属于目标块时，持续迭代
      while (n != nullptr && block != n->owningBlock()) {
        // 更新节点 n 为其所在块的所有者节点
        n = n->owningBlock()->owningNode();
      }
      // 返回找到的所有者节点 n
      return n;
    }

    // 查找或创建节点 n 的删除指令
    Node* findOrCreateDropInstructionForNode(Node* n) {
      // 查找节点 n 是否已存在对应的删除指令
      auto it = drop_for_node.find(n);
      // 如果未找到，创建一个新的删除节点，插入在节点 n 后面，并记录到 drop_for_node
      if (it == drop_for_node.end()) {
        auto drop_node = graph.create(prim::Drop, 0);
        drop_node->insertAfter(n);
        it = drop_for_node.emplace(n, drop_node).first;
      }
      // 返回节点 n 对应的删除指令
      return it->second;
    }

    // 如果删除节点 drop 中不存在值 v 的输入，则将其添加到输入中
    void addToDropIfNotExists(Node* drop, Value* v) {
      // 如果值 v 所在节点是常量节点，则直接返回，不做处理
      if (v->node()->kind() == prim::Constant) {
        return;
      }
      // 遍历删除节点 drop 的所有输入
      for (auto i : drop->inputs()) {
        // 如果值 v 已经是删除节点的输入之一，则直接返回，不重复添加
        if (i == v) {
          return;
        }
      }
      // 将值 v 添加为删除节点 drop 的新输入
      drop->addInput(v);
    }
  };

  // 创建 InsertLastUses 类的实例 ilu，并传入图 graph 作为参数
  InsertLastUses ilu(g);
}

} // namespace



// 在此处结束了一个命名空间的定义

PreprocessGraph::PreprocessGraph(Graph& g) : graph(g.copy()) {
    // PreprocessGraph 类的构造函数，接受一个 Graph 类的引用 g，并通过 g 的 copy 方法复制了一个 graph 对象

    insertEnterMethodCalls(*graph);
    // 调用 insertEnterMethodCalls 函数，将 graph 指针所指的对象作为参数传递

    dropUnused(graph->block());
    // 调用 dropUnused 函数，传递 graph 对象的 block() 方法的结果作为参数

    // fill in move_flags by scanning blocks;
    // 通过扫描 blocks 填充 move_flags（未提供具体代码行，可能是在 insertLastUses 函数内部实现）

    insertLastUses(*graph);
    // 调用 insertLastUses 函数，将 graph 指针所指的对象作为参数传递

    can_emit_inline = std::move(CanEmitInline(*graph.get()).can_emit_inline_);
    // 创建一个 CanEmitInline 对象，将 graph 对象的指针作为参数传递，然后通过 std::move 赋值给 can_emit_inline 变量
}
} // namespace torch::jit::interpreter



// 在此处结束了 torch::jit::interpreter 命名空间的定义


这段代码主要是 C++ 的类构造函数和一些函数调用，涉及到对图数据结构进行预处理和优化的过程。
```