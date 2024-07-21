# `.\pytorch\torch\csrc\jit\runtime\jit_trace.cpp`

```
// 包含 ATen 库中的头文件
#include <ATen/ATen.h>
// 包含 ATen 库中的并行处理功能
#include <ATen/Parallel.h>
// 包含 IValue 类型的核心头文件
#include <ATen/core/ivalue.h>
// 包含符号相关的头文件
#include <ATen/core/symbol.h>
// 包含 JIT IR 视图的头文件
#include <torch/csrc/jit/ir/ir_views.h>
// 包含 JIT 日志的头文件
#include <torch/csrc/jit/jit_log.h>
// 包含死代码消除的头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 包含模块冻结的头文件
#include <torch/csrc/jit/passes/freeze_module.h>
// 包含冻结图优化的头文件
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
// 包含内联函数的头文件
#include <torch/csrc/jit/passes/inliner.h>
// 包含插入保护机制的头文件
#include <torch/csrc/jit/passes/insert_guards.h>
// 包含移除突变的头文件
#include <torch/csrc/jit/passes/remove_mutation.h>
// 包含图执行器的头文件
#include <torch/csrc/jit/runtime/graph_executor.h>
// 包含解释器的头文件
#include <torch/csrc/jit/runtime/interpreter.h>
// 包含 JIT 跟踪的头文件
#include <torch/csrc/jit/runtime/jit_trace.h>
// 包含性能记录的头文件
#include <torch/csrc/jit/runtime/profiling_record.h>
// 包含无序映射的头文件
#include <unordered_map>

namespace torch::jit {

namespace {

// 用于维护脚本图和追踪图之间值的映射关系的辅助结构
struct TracingData {
  std::unordered_map<Value*, Value*> old_to_new_; // 老到新值的映射
  std::shared_ptr<Graph> traced_graph_ = nullptr; // 追踪图

  TracingData() {
    traced_graph_ = std::make_shared<Graph>(); // 创建一个新的图
  }
};

// 创建一个在追踪图中对应于脚本图中节点 `node` 的节点
Node* traceNode(Node* node, TracingData& td, Stack& stack) {
  GRAPH_DEBUG("Tracing node ", getHeader(node)); // 调试信息，跟踪节点
  auto* block = td.traced_graph_->block(); // 获取追踪图的块
  auto env = [&td](Value* v) { return td.old_to_new_.at(v); }; // 环境映射函数

  // 在追踪图的块中追加一个克隆节点
  auto new_node = block->appendNode(td.traced_graph_->createClone(node, env));
  // 复制输出的元数据和映射老到新的输出
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    no->copyMetadata(oo); // 复制元数据
    td.old_to_new_[oo] = no; // 更新映射关系
    GRAPH_DEBUG(
        "Mapping ",
        oo->debugName(),
        " to ",
        no->debugName()); // 输出调试信息，输出的老到新映射
  }
  return new_node; // 返回新节点
}

// 清除节点的所有输出
void eraseAllOutputs(Node* opt_pn) {
  // NOLINTNEXTLINE
  for (int i = opt_pn->outputs().size() - 1; i >= 0; i--) {
    opt_pn->eraseOutput(i); // 清除输出
  }
}

// 在块中插入跟踪节点
void insertTracingNodes(Block*, ProfilingRecord*, TracingData&);

// `createPropNodeForIfBlock` 的微妙之处在于我们需要创建一个“传播”节点，
// 该节点将在脚本节点中的 then/else 块的输出与追踪图中的输出之间进行映射传播，
// 然后映射到脚本节点的 if 节点的输出上。请注意，if 节点在追踪图中将消失，
// 但在脚本图中仍然被使用。
void createPropNodeForIfBlock(
    Block* b,
    Node* n,
    ProfilingRecord* pr,
    TracingData& td) {
  // 创建一个空的值向量
  std::vector<Value*> empty_values{};
  // 使用空值向量创建一个 ProfileIValueNode 节点
  auto opt_pn = pr->createProfileIValueNode(empty_values);
  // 清空 opt_pn 的所有输出
  eraseAllOutputs(opt_pn);
  // 向基本块 b 中插入追踪节点和剖析器相关的内容
  insertTracingNodes(b, pr, td);
  // 将 opt_pn 添加到基本块 b 中
  b->appendNode(opt_pn);
  // 定义一个可选的剖析器，捕获 pr、n、b 和 td 的引用
  std::function<void(Stack&)> optional_profiler =
      [pr, n, b, &td](Stack& stack) {
        // 使用 pr 的互斥锁进行保护
        std::lock_guard<std::mutex> lock(pr->mutex_);

        // frame_id 未使用
        int64_t frame_id = 0;
        // 从栈中弹出 frame_id
        pop(stack, frame_id);

        // 遍历基本块 b 的所有输出
        for (size_t i = 0; i < b->outputs().size(); i++) {
          // 将旧的输出映射到新的输出
          auto nbo = td.old_to_new_.at(b->outputs()[i]);
          td.old_to_new_[n->outputs()[i]] = nbo;
          // 调试输出映射关系信息
          GRAPH_DEBUG(
              "Map ",
              td.old_to_new_[n->outputs()[i]]->debugName(),
              " to ",
              nbo->debugName());
        }
      };

  // 取消注释用于调试
  // opt_pn->i_(Symbol::attr("propagate"), 1);
  // 设置 opt_pn 的回调函数为 optional_profiler
  opt_pn->setCallback(optional_profiler);
}

// 这是一个函数定义，用于跟踪循环计数器，并在循环体执行后进行相关的数据记录
void traceLoopCounter(Node* n, ProfilingRecord* pr, TracingData& td) {
  // 创建 LoopView 对象，用于获取循环的信息
  LoopView lv(n);
  // 创建表示当前迭代次数的 ProfileIValueNode 对象
  auto opt_pn = pr->createProfileIValueNode(lv.currentTripCount());
  // 清除所有输出
  eraseAllOutputs(opt_pn);
  // 将 opt_pn 添加到循环体的开头
  lv.bodyBlock()->prependNode(opt_pn);
  // 定义一个函数用于可选的性能分析，捕获 pr, n, td 的引用
  std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
    // 加锁保护，确保线程安全性
    std::lock_guard<std::mutex> lock(pr->mutex_);
    // 弹出 frame_id（未使用）
    int64_t frame_id = 0;
    pop(stack, frame_id);
    // 弹出循环计数器的值
    int64_t loop_counter = 0;
    pop(stack, loop_counter);
    // 设置插入点为追踪图的块的开头
    WithInsertPoint wip(td.traced_graph_->block());
    // 将循环计数器的常量值插入追踪图
    auto lc = td.traced_graph_->insertConstant(loop_counter);
    // 将旧的计数器值映射到新的常量值
    LoopView lv(n);
    td.old_to_new_[lv.currentTripCount()] = lc;
  };

  // 解除注释以进行调试
  // opt_pn->i_(Symbol::attr("loop_counter"), 1);
  // 设置回调函数
  opt_pn->setCallback(optional_profiler);
}

// 类似于 If 节点映射的传播方式，我们需要在循环体内传播映射到块的输入值（phi 值）
static void traceLoop(Node* n, ProfilingRecord* pr, TracingData& td) {
  // 创建空的值向量
  std::vector<Value*> empty_values{};

  // 这是用于块输入（phi 值）传播的节点
  {
    // 创建一个表示空值的 ProfileIValueNode 对象
    auto opt_pn = pr->createProfileIValueNode(empty_values);
    // 清除所有输出
    eraseAllOutputs(opt_pn);
    // 将 opt_pn 插入到节点 n 的前面
    opt_pn->insertBefore(n);
    // 创建 LoopView 对象
    LoopView lv(n);
    // 定义一个函数用于可选的性能分析，捕获 pr, n, td 的引用
    std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
      // 加锁保护，确保线程安全性
      std::lock_guard<std::mutex> lock(pr->mutex_);

      // 弹出 frame_id（未使用）
      int64_t frame_id = 0;
      pop(stack, frame_id);

      // 创建 LoopView 对象
      LoopView lv(n);
      // 断言循环体携带输入的大小与总体输入的大小相同
      TORCH_INTERNAL_ASSERT(
          lv.bodyCarriedInputs().size() == lv.carriedInputs().size());
      // 遍历循环体携带输入的值
      for (size_t i = 0; i < lv.bodyCarriedInputs().size(); i++) {
        // 获取映射的旧节点
        auto bno = td.old_to_new_.at(lv.carriedInputs()[i]);
        // 将循环体携带输入的旧节点映射到新节点
        td.old_to_new_[lv.bodyCarriedInputs()[i]] = bno;
        // 调试信息，打印映射关系
        GRAPH_DEBUG(
            "Map ",
            td.old_to_new_[lv.bodyCarriedInputs()[i]]->debugName(),
            " to ",
            bno->debugName());
      }
    };

    // 解除注释以进行调试
    // opt_pn->i_(Symbol::attr("loop_entry"), 1);
    // 设置回调函数
    opt_pn->setCallback(optional_profiler);
  }

  {
    // 在循环体中插入追踪节点
    insertTracingNodes(LoopView(n).bodyBlock(), pr, td);
    // 跟踪循环计数器
    traceLoopCounter(n, pr, td);
  }

  // 这是用于循环输出传播的节点
  {
    // 创建一个表示空值的 ProfileIValueNode 对象
    auto opt_pn = pr->createProfileIValueNode(empty_values);
    // 清除所有输出
    eraseAllOutputs(opt_pn);
    // 将 opt_pn 添加到循环体的末尾
    LoopView(n).bodyBlock()->appendNode(opt_pn);

    // 解除注释以进行调试
    // opt_pn->i_(Symbol::attr("loop_propagate"), 1);
    // 定义一个名为 optional_profiler 的函数对象，类型为 std::function<void(Stack&)>
    // 该函数用于执行性能分析，接受一个 Stack 的引用参数

    std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
      // 使用互斥锁保护 profiler 对象的互斥访问
      std::lock_guard<std::mutex> lock(pr->mutex_);

      // frame_id 变量未被使用，仅作为参数 pop 的占位符
      int64_t frame_id = 0;
      pop(stack, frame_id);

      // 创建一个 LoopView 对象 lv，并传入参数 n
      LoopView lv(n);

      // 断言，验证 lv 对象中的 bodyCarriedOutputs 和 carriedOutputs 大小相等
      TORCH_INTERNAL_ASSERT(
          lv.bodyCarriedOutputs().size() == lv.carriedOutputs().size());

      // 遍历 bodyCarriedOutputs 的大小，进行一些映射操作
      for (size_t i = 0; i < lv.bodyCarriedOutputs().size(); i++) {
        // 获取 lv.bodyCarriedOutputs()[i] 对应的新值
        auto bno = td.old_to_new_.at(lv.bodyCarriedOutputs()[i]);
        // 将 lv.carriedOutputs()[i] 映射到 bno，并更新 td.old_to_new_ 中的映射关系
        td.old_to_new_[lv.carriedOutputs()[i]] = bno;
        // 输出调试信息，显示映射关系
        GRAPH_DEBUG(
            "Map ",
            td.old_to_new_[lv.bodyCarriedOutputs()[i]]->debugName(),
            " to ",
            bno->debugName());
      }
    };

    // 取消下面的注释以进行调试
    // opt_pn->i_(Symbol::attr("loop_exit"), 1);

    // 将 optional_profiler 设置为 opt_pn 的回调函数
    opt_pn->setCallback(optional_profiler);
}

// walks all the nodes in a block and adds profiled nodes to each node
// see the comment for `optional_profiler` below
void insertTracingNodes(Block* block, ProfilingRecord* pr, TracingData& td) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto n = *it;
    it++;

    GRAPH_DEBUG("Inserting trace for ", getHeader(n));

    // 如果当前节点是 prim::If 类型，则处理其分支块
    if (n->kind() == prim::If) {
      IfView ifv(n);
      // 为 If 语句的 then 分支块创建属性节点
      createPropNodeForIfBlock(ifv.thenBlock(), n, pr, td);
      // 为 If 语句的 else 分支块创建属性节点
      createPropNodeForIfBlock(ifv.elseBlock(), n, pr, td);
      continue;
    }

    // 如果当前节点是 prim::Loop 类型，则跟踪其循环体
    if (n->kind() == prim::Loop) {
      traceLoop(n, pr, td);
      continue;
    }

    // 断言当前节点没有子块
    TORCH_INTERNAL_ASSERT(n->blocks().empty());

    // 为当前节点的输出创建一个 profile 节点
    auto opt_pn = pr->createProfileIValueNode(n->outputs());
    // 删除所有输出
    eraseAllOutputs(opt_pn);
    // 将 profile 节点插入到当前节点之后
    opt_pn->insertAfter(n);

    // 定义一个可选的性能分析器，用于处理当前节点的追踪
    std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
      std::lock_guard<std::mutex> lock(pr->mutex_);

      // frame_id 未使用
      int64_t frame_id = 0;
      pop(stack, frame_id);

      GRAPH_DEBUG("Tracing ", getHeader(n));

      // 追踪当前节点，并获取其 tracer
      auto tracer = traceNode(n, td, stack);
      auto outputs_size = n->outputs().size();
      auto iivs = pop(stack, outputs_size);
      for (size_t j = 0; j < outputs_size; j++) {
        auto& iiv = iivs[j];
        if (iiv.isTensor()) {
          auto t = iiv.toTensor();
          auto type = t.defined() ? tensorTypeInCurrentExecutionContext(t)
                                  : TensorType::get();
          // 设置 tracer 的输出类型
          tracer->outputs().at(j)->setType(type);
        }
      }
    };

    // 设置 profile 节点的回调函数为 optional_profiler
    opt_pn->setCallback(optional_profiler);
  }
}
} // namespace

// To trace graph we create a profile node for every one
// in a scripted graph. When a profiled node handler runs
// we insert a new traced node in a trace graph
// If the profiled node handler is called in a loop
// we will have multiple nodes.
// We also maintain the mapping between the outputs of traced
// nodes and the outputs of the node in the scripted graph.
// There are a few subtleties with tracing Ifs and Loops
// discussed above
std::shared_ptr<Graph> TraceGraph(std::shared_ptr<Graph> graph, Stack& stack) {
  TracingData td;
  GRAPH_DUMP("Before Inline:", graph);
  Inline(*graph.get());
  EliminateDeadCode(graph);
  GRAPH_DUMP("After Inline:", graph);

  // 对图进行仪器化，返回 profiling 记录
  auto pr = ProfilingRecord::instrumentGraph(graph);

  // 遍历 profiled_graph_ 的输入
  for (auto inp : pr->profiled_graph_->inputs()) {
    // 向 traced_graph_ 添加输入，并复制元数据
    auto ni = td.traced_graph_->addInput();
    ni->copyMetadata(inp);
    ni->setType(ni->type());


注：以上是对给定代码块逐行进行了注释说明，按照要求将注释和代码块一同输出。
  // 将输入和对应的输出索引存储到映射表中
  td.old_to_new_[inp] = ni;
}

// 使用栈中的输入设置图的输入类型。
// 在运行解释器之前必须完成此步骤，因为栈中只有运行后的输出。
for (auto i : c10::irange(stack.size())) {
  // 如果栈中的元素是张量，则设置图的输入类型
  if (stack[i].isTensor()) {
    td.traced_graph_->inputs().at(i)->setType(
        tensorTypeInCurrentExecutionContext(stack[i].toTensor()));
  }
}

// 移除分析图中的性能分析计数器和性能分析节点
ProfilingRecord::removeProfileCounter(pr->profiled_graph_->block());
ProfilingRecord::removeProfilingNodes(pr->profiled_graph_->block());

// 在分析图的块中插入追踪节点
insertTracingNodes(pr->profiled_graph_->block(), pr.get(), td);

// 输出性能分析图的内容
GRAPH_DUMP("Profiling Graph:", pr->profiled_graph_);

// 为性能分析图创建代码对象
Code cd(pr->profiled_graph_, "");

// 创建解释器状态并执行
InterpreterState is{cd};
is.run(stack);

// 遍历性能分析图的输出，并在跟踪图的块中注册输出
for (auto out : pr->profiled_graph_->outputs()) {
  td.traced_graph_->block()->registerOutput(td.old_to_new_.at(out));
}

// 输出跟踪图的内容
GRAPH_DUMP("Traced graph:", td.traced_graph_);

// 返回跟踪图对象
return td.traced_graph_;
}
} // namespace torch::jit


// 结束了命名空间 torch::jit 的定义
}
// 结束了命名空间声明
} // namespace torch::jit
```