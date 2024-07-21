# `.\pytorch\torch\csrc\jit\runtime\simple_graph_executor_impl.cpp`

```py
// 包含 TorchScript 运行时的头文件
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>

// 包含一些实用工具的头文件，比如 Optional 类型
#include <c10/util/Optional.h>
// 包含简单图执行器实现的头文件
#include <torch/csrc/jit/runtime/simple_graph_executor_impl.h>
// 包含互斥锁的头文件
#include <mutex>

// torch::jit 命名空间中的实现
namespace torch::jit {

// SimpleGraphExecutorImpl 构造函数的实现
SimpleGraphExecutorImpl::SimpleGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : GraphExecutorImplBase(graph, std::move(function_name)) {}

// 获取执行计划的函数，使用给定的栈和可选的回溯深度
const ExecutionPlan& SimpleGraphExecutorImpl::getPlanFor(
    Stack& stack,
    std::optional<size_t> remaining_bailout_depth) {
  // 加锁保证线程安全
  std::lock_guard<std::mutex> lock(compile_mutex);

  // IMPORTANT: This is a hot path of calling a torchscript function. Try not to
  // add any code above this.
  // 如果已经存在执行计划，则直接返回它
  if (execution_plan_) {
    return *execution_plan_;
  }
  // 复制原始图，并运行优化前的通道
  auto copy = graph->copy();
  runNooptPassPipeline(copy);
  // 创建新的执行计划
  execution_plan_ = ExecutionPlan(copy, function_name_);

  return *execution_plan_;
}

// 获取调试状态的函数实现
GraphExecutorState SimpleGraphExecutorImpl::getDebugState() {
  // 创建一个新的调试状态对象
  GraphExecutorState state;
  // 确保执行计划已经存在
  TORCH_INTERNAL_ASSERT(execution_plan_);
  // 将执行计划中的图设置为状态对象的图
  state.graph = execution_plan_->graph.get();
  // 将执行计划作为参数规范添加到执行计划集合中
  auto opt_plan = *execution_plan_;
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  // 返回状态对象
  return state;
}

} // namespace torch::jit
```