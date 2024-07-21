# `.\pytorch\torch\csrc\jit\runtime\graph_executor_impl.h`

```py
#pragma once
// 包含 Torch JIT 运行时的图执行器头文件

#include <torch/csrc/jit/runtime/graph_executor.h> // 包含图执行器的声明

// 包含 ATen 核心库中的 IValue 类定义
#include <ATen/core/ivalue.h>

// 包含 C10 实用工具中的 Exception 头文件
#include <c10/util/Exception.h>

// 包含 Torch 自动微分中的梯度模式管理
#include <torch/csrc/autograd/grad_mode.h>

// 包含 Torch JIT 前端的跟踪器声明
#include <torch/csrc/jit/frontend/tracer.h>

// 包含 Torch JIT IR 的声明
#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch JIT 中的形状分析头文件声明
#include <torch/csrc/jit/passes/shape_analysis.h>

// 包含 Torch JIT 运行时的资源守卫声明
#include <torch/csrc/jit/resource_guard.h>

// 包含 Torch JIT 运行时参数规范声明
#include <torch/csrc/jit/runtime/argument_spec.h>

// 包含 Torch JIT 自动微分相关声明
#include <torch/csrc/jit/runtime/autodiff.h>

// 包含 Torch JIT 自定义运算符声明
#include <torch/csrc/jit/runtime/custom_operator.h>

// 包含 Torch JIT 解释器声明
#include <torch/csrc/jit/runtime/interpreter.h>

// 包含 Torch JIT 运行时的性能记录声明
#include <torch/csrc/jit/runtime/profiling_record.h>

// 包含 Torch 自动微分中的边缘声明
#include <torch/csrc/autograd/edge.h>

// 包含 Torch 自动微分中的函数声明
#include <torch/csrc/autograd/function.h>

// 包含 Torch JIT 前端的 IR 发射器声明
#include <torch/csrc/jit/frontend/ir_emitter.h>

// 包含 Torch JIT 运行时的日志记录声明
#include <torch/csrc/jit/runtime/logging.h>

#include <cstdint>        // 包含 C++ 标准库中的整数类型声明
#include <iterator>       // 包含 C++ 标准库中的迭代器声明
#include <memory>         // 包含 C++ 标准库中的内存管理声明
#include <mutex>          // 包含 C++ 标准库中的互斥量声明
#include <unordered_map>  // 包含 C++ 标准库中的无序映射声明
#include <utility>        // 包含 C++ 标准库中的实用功能声明
#include <vector>         // 包含 C++ 标准库中的向量声明

// Torch JIT 命名空间
namespace torch::jit {

// 函数：将梯度打包到节点上
void packGradient(const Gradient& gradient, Node* dnode);

// 函数：判断图是否需要梯度
bool needsGradient(const std::shared_ptr<const Graph>& graph);

// 函数：运行优化器，包括展开非常量循环和常量传播用户类
void runOptimization(
    std::shared_ptr<Graph>& graph,
    bool unroll_non_constant_loops = true,
    bool const_prop_user_classes = true);

// 函数：运行非微分图优化
void runNondiffOptimization(
    std::shared_ptr<Graph>& graph,
    bool strict_fuser_check = false);

// 函数：设置自动微分子图内联调试状态
void debugSetAutodiffSubgraphInlining(bool state);

// 函数：获取自动微分子图内联状态
bool TORCH_API getAutodiffSubgraphInlining();

// 函数：设置融合组内联调试状态
void debugSetFusionGroupInlining(bool state);

// 函数：获取融合组内联状态
bool getFusionGroupInlining();

// 可调参数：决定何时创建/保留不同可微代码的子图
const size_t autodiffSubgraphNodeThreshold = 2;
const size_t autodiffSubgraphInlineThreshold = 5;

// 图可以通过跟踪或基于语言的前端创建
// GraphExecutor 在许多不同大小和 requires_grad 状态下运行相同的图，
// 并处理每种情况的特殊化。GraphExecutor 完全不知道跟踪或模块参数，
// 以保持跟踪关注点的分离。
struct GraphExecutorImplBase {
  // 静态函数：准备图的副本，擦除形状信息
  static std::shared_ptr<Graph> prepareGraph(
      const std::shared_ptr<Graph>& graph) {
    auto copy = graph->copy();
    EraseShapeInformation(copy);
    return copy;
  }

  // 构造函数：初始化 GraphExecutorImplBase 对象
  GraphExecutorImplBase(
      const std::shared_ptr<Graph>& graph,
      std::string function_name)
      : graph(prepareGraph(graph)),  // 准备图的副本
        function_name_(std::move(function_name)),
        num_inputs(this->graph->inputs().size()),   // 输入数量
        num_outputs(this->graph->outputs().size())  // 输出数量
  {}

  // 函数：执行开始的入口点
  void run(Stack& stack);

  // 函数：异步运行图执行器
  c10::intrusive_ptr<Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch);

  // 纯虚函数：获取指定堆栈的执行计划
  virtual const ExecutionPlan& getPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth = c10::nullopt) = 0;

  // 纯虚函数：获取调试状态
  virtual GraphExecutorState getDebugState() = 0;

  // 虚析构函数
  virtual ~GraphExecutorImplBase() = default;

  // 虚函数：判断是否已优化
  virtual bool isOptimized() const {
    // 返回 false
    return false;
  }

 protected:
  friend struct GraphExecutor;

  // 未经优化的起始图。这个字段实际上是 const 的，但我们不能这样做，因为 Graph::copy() 不是 const 的（在这一点上使其 const 不是那么容易的）。
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::string function_name_;

  // 如果为 false，我们将按原样运行图，不进行任何优化。用于调试。
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const size_t num_inputs;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const size_t num_outputs;

  // GraphExecutors 可以从多个线程访问，因此每次访问 fallback 或 plan_cache 时都需要持有此线程。
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex compile_mutex;
};

} // namespace torch::jit
```