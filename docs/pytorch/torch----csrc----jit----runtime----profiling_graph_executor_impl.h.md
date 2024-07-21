# `.\pytorch\torch\csrc\jit\runtime\profiling_graph_executor_impl.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/util/Flags.h>
// 包含 c10 库中的 Flags 头文件

#include <torch/csrc/jit/api/module.h>
// 包含 Torch 的 JIT 模块 API 头文件

#include <torch/csrc/jit/runtime/graph_executor_impl.h>
// 包含 Torch 的 JIT 运行时图执行器实现头文件

C10_DECLARE_bool(torch_jit_static_then_dynamic);
// 声明一个名为 torch_jit_static_then_dynamic 的布尔类型变量

C10_DECLARE_bool(torch_jit_always_dynamic);
// 声明一个名为 torch_jit_always_dynamic 的布尔类型变量

namespace torch::jit {
// 进入 torch::jit 命名空间

TORCH_API void runNooptPassPipeline(std::shared_ptr<Graph>& graph);
// 定义一个 Torch API 函数 runNooptPassPipeline，接受一个 Graph 类型的智能指针引用参数

struct TORCH_API ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  // 定义一个结构体 ProfilingGraphExecutorImpl，继承自 GraphExecutorImplBase

  ProfilingGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);
  // 构造函数，接受一个 Graph 类型的智能指针引用参数和一个字符串参数

  const ExecutionPlan& getPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth) override;
  // 重载 getPlanFor 方法，返回类型为 ExecutionPlan 的常引用，接受 Stack 类型引用和可选的 size_t 类型参数

  GraphExecutorState getDebugState() override;
  // 重载 getDebugState 方法，返回 GraphExecutorState 对象

  ~ProfilingGraphExecutorImpl() override = default;
  // 虚析构函数，默认实现

  void debugFlushCompilationCache();
  // 定义一个无返回值的 debugFlushCompilationCache 方法

  bool isOptimized() const override {
    return optimized_plan_.has_value();
  }
  // 重载 isOptimized 方法，返回是否存在优化计划的布尔值

 private:
  const ExecutionPlan& getOptimizedPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth);
  // 声明私有方法 getOptimizedPlanFor，返回类型为 ExecutionPlan 的常引用，接受 Stack 类型引用和可选的 size_t 类型参数

  void runProfilingInsensitiveOptimizations(std::shared_ptr<Graph>& graph);
  // 声明私有方法 runProfilingInsensitiveOptimizations，无返回值，接受 Graph 类型的智能指针引用参数

  void runProfilingOptimizations(
      std::shared_ptr<Graph>& graph,
      size_t remaining_depth);
  // 声明私有方法 runProfilingOptimizations，无返回值，接受 Graph 类型的智能指针引用参数和 size_t 类型参数

  void replaceFallbackGraphWithFallbackFunction(Block* b);
  // 声明私有方法 replaceFallbackGraphWithFallbackFunction，无返回值，接受 Block 指针参数

  FusionBehavior getCurrentBehavior(size_t remaining_depth);
  // 声明私有方法 getCurrentBehavior，返回 FusionBehavior 枚举类型，接受 size_t 类型参数

  size_t getInstantiatedBailoutDepth();
  // 声明私有方法 getInstantiatedBailoutDepth，返回 size_t 类型值

  void runNoGradOptimizations(
      std::shared_ptr<Graph>& graph,
      size_t remaining_bailout_depth);
  // 声明私有方法 runNoGradOptimizations，无返回值，接受 Graph 类型的智能指针引用参数和 size_t 类型参数

  void runFinalOptimizations(std::shared_ptr<Graph>& graph);
  // 声明私有方法 runFinalOptimizations，无返回值，接受 Graph 类型的智能指针引用参数

  void clearTheGraphCompilationIntermediateGraphs();
  // 声明私有方法 clearTheGraphCompilationIntermediateGraphs，无返回值

  std::unique_ptr<ProfilingRecord> pr_;
  // 声明一个唯一指针 pr_，指向 ProfilingRecord 对象

  std::optional<ExecutionPlan> profiling_plan_;
  // 声明一个可选的 ExecutionPlan 类型变量 profiling_plan_

  std::optional<ExecutionPlan> optimized_plan_;
  // 声明一个可选的 ExecutionPlan 类型变量 optimized_plan_

  FusionStrategy fusion_strategy_;
  // 声明一个 FusionStrategy 类型变量 fusion_strategy_

  std::optional<ExecutionPlan> fallback_plan_;
  // 声明一个可选的 ExecutionPlan 类型变量 fallback_plan_

  std::vector<std::unique_ptr<Function>> fallback_functions_;
  // 声明一个存储唯一指针的 Function 向量 fallback_functions_

  std::optional<size_t> remaining_bailout_depth_;
  // 声明一个可选的 size_t 类型变量 remaining_bailout_depth_

  int32_t time_optimized_plan_created_ = 0;
  // 声明一个 int32_t 类型变量 time_optimized_plan_created_

  bool is_graph_extra_memory_released_ = false;
  // 声明一个布尔类型变量 is_graph_extra_memory_released_
};

} // namespace torch::jit
// 退出 torch::jit 命名空间
```