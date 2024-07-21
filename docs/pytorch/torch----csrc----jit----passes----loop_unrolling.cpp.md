# `.\pytorch\torch\csrc\jit\passes\loop_unrolling.cpp`

```py
// 引入 Torch JIT passes 中的循环展开头文件
#include <torch/csrc/jit/passes/loop_unrolling.h>

// 引入 ATen 核心的符号头文件
#include <ATen/core/symbol.h>
// 引入 C10 实用工具中的异常处理头文件
#include <c10/util/Exception.h>
// 引入 C10 实用工具中的范围迭代头文件
#include <c10/util/irange.h>

// 引入 Torch JIT 中的 IR 常量定义头文件
#include <torch/csrc/jit/ir/constants.h>
// 引入 Torch JIT 中的 IR 视图头文件
#include <torch/csrc/jit/ir/ir_views.h>
// 引入 Torch JIT 中的 JIT 日志头文件
#include <torch/csrc/jit/jit_log.h>
// 引入 Torch JIT passes 中的死代码消除头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// Torch JIT 命名空间开始
namespace torch {
namespace jit {

// 匿名命名空间，用于定义内部静态函数和常量
namespace {

// 循环展开的因子常量
static constexpr int64_t kUnrollFactor = 8;
// 循环主体的最大大小限制常量
static constexpr int64_t kMaxBodySize = 32;
// 循环主体重复次数的最大限制常量
static constexpr int64_t kMaxBodyRepeats = 64;

// 判断值是否为真常量
bool isTrueConstant(Value* val) {
  // 尝试将值解析为布尔型的可选值
  std::optional<bool> maybe_value = constant_as<bool>(val);
  // 返回是否成功解析并且解析结果为真
  return maybe_value && *maybe_value;
}

// 判断节点是否为 for 循环
bool isForLoop(Node* node) {
  // 如果节点类型不是 prim::Loop，则返回 false
  if (node->kind() != prim::Loop)
    return false;
  // 获取循环开始条件和继续条件的值
  Value* start_cond = node->inputs().at(1);
  Value* continue_cond = node->blocks().at(0)->outputs().at(0);
  // 返回开始条件和继续条件都为真常量的结果
  return isTrueConstant(start_cond) && isTrueConstant(continue_cond);
}

// 计算限制块的大小，达到指定指令数后停止并返回
int64_t limitedBlockSize(Block* body, int64_t limit) {
  auto it = body->nodes().begin();
  auto end = body->nodes().end();
  // 遍历块中的节点，直到达到限制大小或遍历完所有节点
  for (int64_t i = 0; i < limit; ++it) {
    // 遍历子块，递归计算其大小
    for (Block* subblock : it->blocks()) {
      i += limitedBlockSize(subblock, limit - i);
    }
    // 如果当前节点不是未执行的操作，则计数加一
    if (!it->notExecutedOp()) {
      ++i;
    }
    // 如果遍历到结尾，则返回当前计数
    if (it == end) {
      return i;
    }
  }
  // 返回限制的最大大小
  return limit;
}

// 判断块是否为小块
bool isSmallBlock(Block* body) {
  // 返回限制块大小不超过最大主体大小限制
  return limitedBlockSize(body, kMaxBodySize + 1) <= kMaxBodySize;
}

// 内联循环主体的函数实现
// 注意：此函数只能在保证循环确切执行一次的情况下调用
void inlineBody(Node* loop) {
  auto graph = loop->owningGraph();
  auto body = loop->blocks().at(0);
  // 设置插入点为循环节点
  WithInsertPoint insert_point_guard{loop};

  // 值映射表，将循环体的输入映射到当前循环节点的输入
  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value* v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };

  // 复制循环体的节点到当前图中
  for (Node* orig : body->nodes()) {
    Node* clone = graph->insertNode(graph->createClone(orig, get_value));
    // 更新值映射表，将原节点输出映射到克隆节点输出
    for (size_t i = 0; i < orig->outputs().size(); ++i) {
      value_map[orig->outputs()[i]] = clone->outputs()[i];
    }
  }
  // 将循环节点的输出替换为循环体的输出
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs().at(i)->replaceAllUsesWith(
        get_value(body->outputs().at(i + 1)));
  }
  // 销毁循环节点，注意：在此处销毁循环节点非常重要，因为死代码消除可能无法确定其安全性，因为循环可能包含副作用
  loop->destroy();
}

// 插入块的副本，将输入传递给块的输入，并返回块输出的值列表
std::vector<Value*> insertBlockCopy(
    Graph& graph,
    Block* body,
    // 对输入的值列表进行断言，确保其大小与函数体中的输入数量相同
    TORCH_INTERNAL_ASSERT(inputs.size() == body->inputs().size());
    
    // 创建一个空的值映射，用于将原始节点中的值映射到克隆节点中的对应值
    std::unordered_map<Value*, Value*> value_map;
    
    // 定义一个lambda函数get_value，用于从值映射中获取给定值的替代值
    auto get_value = [&](Value* v) {
      auto it = value_map.find(v);
      if (it != value_map.end())
        return it->second;
      return v;  // 如果找不到映射，则返回原始值
    };
    
    // 遍历函数体中的输入节点，并将输入节点与传入的实际输入值进行映射
    auto inputs_it = inputs.begin();
    for (Value* input : body->inputs()) {
      value_map[input] = *inputs_it++;
    }
    
    // 遍历函数体中的每一个节点
    for (Node* node : body->nodes()) {
      // 在新图中插入一个克隆节点，克隆自原始节点，并使用get_value函数替换其输入值
      Node* new_node = graph.insertNode(graph.createClone(node, get_value));
    
      // 遍历原始节点的输出值，并将其与新节点的输出值进行映射
      auto outputs_it = new_node->outputs().begin();
      for (Value* output : node->outputs()) {
        value_map[output] = *outputs_it++;
      }
    }
    
    // 返回映射后的函数体的输出值列表
    return fmap(body->outputs(), get_value);
}

// 将循环体重复执行多次，生成结果存储在目标 Block 中
void repeatBody(Block* body, size_t times, Block* dest) {
  // 获取循环体所属的图
  auto graph = body->owningGraph();
  // 将插入点设置在目标 Block 中
  WithInsertPoint insert_point_guard(dest);
  // 复制循环体的输入参数的元数据到目标 Block 的输入参数中
  for (Value* input : body->inputs()) {
    dest->addInput()->copyMetadata(input);
  }

  // 获取目标 Block 的输入参数的向量
  std::vector<Value*> io = dest->inputs().vec();
  // 断言循环计数器没有被使用，用于内部调试
  TORCH_INTERNAL_ASSERT(
      !body->inputs().at(0)->hasUses(), "loop counter should be unused");
  // 循环执行循环体指定次数
  for (const auto i : c10::irange(times)) {
    (void)i; // 抑制未使用变量的警告
    // 将循环体的输入参数连接到目标 Block 的输入参数中
    io[0] = body->inputs().at(0);
    // 在图中插入循环体的副本，并更新 io 向量
    io = insertBlockCopy(*graph, body, io);
  }
  // 将生成的输出值注册到目标 Block 中
  for (Value* output : io) {
    dest->registerOutput(output);
  }

  // 可能存在一些死节点，例如阻止循环中断的常量。应尽快移除这些节点，避免人为增加循环大小，从而阻碍外部循环展开。
  EliminateDeadCode(dest, false);
}

// 将内置循环计数器替换为循环外的“可变”变量
void replaceLoopCounter(Node* loop) {
  // 获取循环所属的图和循环体
  Graph* graph = loop->owningGraph();
  Block* body = loop->blocks().at(0);
  // 在循环节点处设置插入点
  WithInsertPoint guard(loop);
  // 插入初始计数器为常量 0
  Value* init_counter = graph->insertConstant(0);

  // 在循环的第二个输入位置插入计数器
  loop->insertInput(2, init_counter);
  // 插入一个输出作为循环的第一个输出，并设置类型为整型
  loop->insertOutput(0)->setType(IntType::get());

  // 获取循环体的内部计数器，并用初始计数器的类型设置它的类型
  Value* internal_counter = body->insertInput(1)->setType(init_counter->type());
  // 将循环的第一个输入参数的所有使用替换为内部计数器
  body->inputs()[0]->replaceAllUsesWith(internal_counter);

  // 在循环体的返回节点处设置插入点
  WithInsertPoint insertPointGuard{body->return_node()};
  // 插入一个加法操作，将内部计数器加 1，并将结果作为新的输出
  Value* result = graph->insert(aten::add, {internal_counter, 1});
  body->insertOutput(1, result);
}

// 展开循环节点
void unroll(Node* loop) {
  // 获取循环所属的图和循环体
  Graph* graph = loop->owningGraph();
  Block* body = loop->blocks().at(0);

  // 如果循环计数器在循环体中被使用，则替换为外部的“可变”计数器
  if (!body->inputs()[0]->uses().empty())
    replaceLoopCounter(loop);

  // 对于长度已知且较小的循环，完全展开循环体
  Value* trip_count = loop->inputs().at(0);
  std::optional<int64_t> const_len = constant_as<int64_t>(trip_count);
  if (const_len && *const_len < kMaxBodyRepeats) {
    // 添加一个新的 Block 作为目标，重复执行循环体并将结果存储在其中
    Block* dest = loop->addBlock();
    repeatBody(body, *const_len, dest);
    // 移除原始的循环 Block，并内联替换成展开后的结果
    loop->eraseBlock(0);
    inlineBody(loop);
    return;
  }

  // 在循环节点处设置插入点
  WithInsertPoint insert_point_guard{loop};

  // 在展开循环前，克隆循环体。克隆体将成为后续处理的结尾部分
  Node* loop_epilogue =
      graph->createClone(loop, [](Value* v) { return v; })->insertAfter(loop);
  // 替换循环的输出使用为克隆体的输出
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs()[i]->replaceAllUsesWith(loop_epilogue->outputs()[i]);
  loop_epilogue->replaceInput(i + 2, loop->outputs()[i]);

  # 将循环后处理的第 i+2 个输入替换为循环的第 i 个输出

Block* dest = loop->addBlock();

  # 在循环中添加一个新的基本块（block）

repeatBody(body, kUnrollFactor, dest);

  # 使用 repeatBody 函数对循环体进行展开，展开因子为 kUnrollFactor，目标基本块为 dest

loop->eraseBlock(0);

  # 删除循环中索引为 0 的基本块

// Change the iteration counts of both loops

  # 改变两个循环的迭代计数

Value* iter_count = loop->inputs().at(0);

  # 获取循环的第一个输入作为迭代计数器

Value* unrolled_iter_count = graph->insert(
    aten::__round_to_zero_floordiv, {iter_count, kUnrollFactor});

  # 使用 graph 的 insert 方法插入一个向下取整除法节点，计算迭代计数器除以 kUnrollFactor 的结果

loop->replaceInput(0, unrolled_iter_count);

  # 替换循环的第一个输入为展开后的迭代计数器

loop_epilogue->replaceInput(
    0,
    graph->insert(
        aten::sub,
        {iter_count,
         graph->insert(aten::mul, {unrolled_iter_count, kUnrollFactor})}));

  # 更新 loop_epilogue 的第一个输入，计算原始迭代计数器减去展开后的迭代计数器乘以 kUnrollFactor 的结果
} // 结束了一个函数或者类的定义

bool UnrollLoops(Block* block, bool constant_only) {
  bool changed = false; // 初始化变量，用于记录循环是否发生了改变
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    // XXX: unroll might destroy the current node, so we need to pre-increment
    // the iterator
    Node* node = *it; // 获取当前迭代器指向的节点
    ++it; // 预先增加迭代器，因为 unroll 可能会销毁当前节点

    for (Block* subblock : node->blocks()) {
      changed |= UnrollLoops(subblock, constant_only); // 递归调用 UnrollLoops 处理子块
    }

    if (!isForLoop(node)) { // 如果当前节点不是循环节点，则继续下一个节点的处理
      continue;
    }

    if (constant_only) {
      if (node->inputs().at(0)->node()->kind() != prim::Constant) {
        continue; // 如果要求只处理常量循环，且当前循环的第一个输入不是常量，则跳过
      }
    } else {
      if (!isSmallBlock(node->blocks().at(0))) {
        continue; // 如果不是常量循环，且当前节点的第一个子块不是小块，则跳过
      }
    }

    unroll(node); // 对当前循环节点进行展开
    changed = true; // 标记循环发生了改变
  }
  return changed; // 返回是否有循环发生了改变
}

} // 匿名命名空间结束

static void addCondAsOutput(Node* loop) {
  LoopView loop_view(loop); // 创建 LoopView 对象，用于查看循环信息
  loop->addInput(loop_view.inputCond()); // 将循环条件作为输入添加到循环节点
  auto block_cond_input = loop_view.bodyBlock()->addInput(); // 在循环体块中添加输入
  block_cond_input->copyMetadata(loop_view.inputCond()); // 复制元数据到循环体块的输入
  auto cond_output_index =
      loop_view.bodyBlock()->registerOutput(loop_view.nextCond()); // 在循环体块中注册输出条件
  loop_view.bodyBlock()->outputs()[cond_output_index]->copyMetadata(
      loop_view.nextCond()); // 复制元数据到循环体块的输出条件
  auto cond_output = loop->addOutput(); // 将条件输出添加到循环节点
  cond_output->copyMetadata(loop_view.nextCond()); // 复制元数据到条件输出
}

bool LoopsPeeler::run(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before LoopsPeeler", graph); // 输出调试信息，显示处理前的图结构
  collectLoops(graph->block()); // 收集图中的循环信息
  peelLoops(); // 对收集到的循环进行处理（剥离）
  GRAPH_DUMP("After LoopsPeeler", graph); // 输出调试信息，显示处理后的图结构
  return true; // 运行成功返回 true
}

void LoopsPeeler::collectLoop(Node* n) {
  if (callback_(n)) { // 如果节点符合回调函数的条件
    if (in_loop_) { // 如果当前已经在一个循环中
      GRAPH_DEBUG("Loop ", getHeader(in_loop_), " will be unrolled"); // 输出调试信息，显示将要展开的循环头部信息
      loops_to_peel_.push_back(in_loop_); // 将需要剥离的循环添加到列表中
      in_loop_ = nullptr; // 清空当前循环指针
    }
  }
}

void LoopsPeeler::collectLoops(Block* block) {
  // we do a pre-order traversal to reduce the number
  // of peeled loops.
  for (auto n : block->nodes()) { // 对块中的每个节点进行处理
    collectLoop(n); // 收集节点中的循环信息
  }
  collectLoop(block->return_node()); // 收集块的返回节点中的循环信息

  // 处理子块
  for (auto n : block->nodes()) { // 对块中的每个节点进行处理
    auto old_in_loop_ = in_loop_; // 保存旧的循环指针
    if (n->kind() == prim::Loop) {
      in_loop_ = n; // 如果当前节点是循环节点，则更新循环指针
    }
    for (auto b : n->blocks()) {
      collectLoops(b); // 递归调用，收集子块中的循环信息
    }
    in_loop_ = old_in_loop_; // 恢复旧的循环指针
  }
}

void LoopsPeeler::peelLoops() {
  for (auto loop : loops_to_peel_) { // 遍历待剥离的循环列表
    PeelLoop(loop, num_iterations_); // 对每个循环进行剥离处理
  }
}

bool PeelProfilingLoops(const std::shared_ptr<Graph>& graph) {
  auto peel_predicate = [](Node* n) { // 定义一个剥离条件的谓词函数
    for (auto i : n->inputs()) {
      if (i->type()->isSubtypeOf(*TensorType::get())) { // 如果节点的输入是张量类型
        return true; // 返回 true，表示需要剥离该循环
      }
    }
    return false; // 否则返回 false
  };

  LoopsPeeler lp(peel_predicate); // 创建 LoopsPeeler 对象，传入剥离条件谓词
  return lp.run(graph); // 运行 LoopsPeeler 处理图中的循环
}
// 函数定义，将循环结点 `n` 剥离 `times` 次
Node* PeelLoop(Node* n, size_t times) {
  // 输出调试信息，显示正在剥离的循环结点 `n` 及剥离次数 `times`
  GRAPH_DEBUG("Peeling the loop ", getHeader(n), " ", times, " times");

  // 获取循环结点 `n` 所属的计算图
  auto graph = n->owningGraph();
  // 获取循环的原始视图
  auto orig_loop = LoopView(n);

  // 将插入点设置为循环结点 `n` 处
  WithInsertPoint wip(n);
  // 插入常量节点，表示剥离的次数
  auto times_const = graph->insertConstant(static_cast<int64_t>(times));

  // 注意事项：尽管调用者可能要求剥离 `times` 次迭代，
  // 但原始循环的 `maxTripCount` 可能小于此值，
  // 因此我们需要取两者的最小值
  auto min_trip_count =
      graph->insert(prim::min, {orig_loop.maxTripCount(), times_const});

  // 创建剥离后的克隆节点
  auto peeled_copy = graph->createClone(n, [](Value* v) { return v; });
  // 将条件作为输出添加到剥离的克隆节点
  addCondAsOutput(peeled_copy);

  // 创建新的循环视图
  LoopView new_lv(peeled_copy);
  // 将剥离后的克隆节点插入计算图中
  graph->insertNode(peeled_copy);
  // 更新新循环视图的最大迭代次数为 `min_trip_count`
  new_lv.replaceMaxTripCount(min_trip_count);

  // 计算原始循环的 `maxTripCount` 减去剥离后的迭代次数
  auto new_max_trip_count =
      graph->insert(aten::sub, {orig_loop.maxTripCount(), min_trip_count});
  // 更新原始循环的最大迭代次数
  orig_loop.replaceMaxTripCount(new_max_trip_count);

  // 更新终止条件
  auto cond_index = peeled_copy->outputs().size() - 1;
  orig_loop.replaceInputCondition(peeled_copy->output(cond_index));

  // 循环依赖项的偏移量
  static const size_t LOOP_DEPS_WITH_COND_OFFSET = 2;
  // 更新原始循环的输入，排除终止条件
  for (size_t i = 0; i < peeled_copy->outputs().size() - 1; i++) {
    n->replaceInput(LOOP_DEPS_WITH_COND_OFFSET + i, peeled_copy->output(i));
  }

  // 调整归纳变量，使其按剥离后的迭代次数调整
  {
    // 将插入点设置为原始循环体的第一个节点处
    WithInsertPoint peeled_wip(*orig_loop.bodyBlock()->nodes().begin());
    // 插入加法节点，调整迭代计数器
    auto adjusted_iter_counter =
        graph->insert(aten::add, {min_trip_count, min_trip_count});
    // 替换当前迭代计数器的所有使用为调整后的计数器
    orig_loop.currentTripCount()->replaceAllUsesWith(adjusted_iter_counter);
    // 更新加法节点的输入为原始的当前迭代计数器
    adjusted_iter_counter->node()->replaceInput(
        0, orig_loop.currentTripCount());
  }

  // 返回剥离后的克隆节点
  return peeled_copy;
}

// 循环展开函数，对计算图 `graph` 中的循环进行展开
bool UnrollLoops(std::shared_ptr<Graph>& graph) {
  // 调用内部函数 `UnrollLoops`，对计算图的块进行展开，不展开常量循环
  bool changed = UnrollLoops(graph->block(), false);
  // 如果发生变化，则消除死代码
  if (changed) {
    EliminateDeadCode(graph);
  }
  // 返回是否发生了变化
  return changed;
}

// 循环常量展开函数，对计算图 `graph` 中的循环进行常量展开
bool UnrollConstantLoops(std::shared_ptr<Graph>& graph) {
  // 调用内部函数 `UnrollLoops`，对计算图的块进行展开，展开常量循环
  bool changed = UnrollLoops(graph->block(), true);
  // 如果发生变化，则消除死代码
  if (changed) {
    EliminateDeadCode(graph);
  }
  // 返回是否发生了变化
  return changed;
}

// 命名空间 jit
} // namespace jit
// 命名空间 torch
} // namespace torch
```