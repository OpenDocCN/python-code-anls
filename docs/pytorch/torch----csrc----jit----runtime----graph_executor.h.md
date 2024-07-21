# `.\pytorch\torch\csrc\jit\runtime\graph_executor.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <atomic>
#include <memory>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/variable_tensor_list.h>
// 包含所需的头文件

C10_DECLARE_bool(torch_jit_enable_new_executor);
// 声明一个布尔型全局变量 torch_jit_enable_new_executor

C10_DECLARE_bool(torch_jit_execution_plan_reuse_code_graph);
// 声明一个布尔型全局变量 torch_jit_execution_plan_reuse_code_graph

namespace torch::jit {
// 进入 torch::jit 命名空间

struct GraphExecutorState;
struct Code;

enum ExecutorExecutionMode {
  SIMPLE,      // 执行器简单模式
  PROFILING,   // 执行器性能分析模式
};

struct ExecutionPlan {
  ExecutionPlan() = default;
  // 默认构造函数

  ExecutionPlan(std::shared_ptr<Graph> graph, std::string function_name)
      : code(graph, std::move(function_name)),  // 初始化 code 成员
        graph(
            FLAGS_torch_jit_execution_plan_reuse_code_graph
                ? code.graph()                 // 根据标志决定是否复用代码图
                : std::move(graph)) {}         // 初始化 graph 成员

  operator bool() const {
    return static_cast<bool>(graph);  // 转换为 bool 类型，检查是否有效
  }

  Code code;                           // 代码对象
  std::shared_ptr<Graph> graph;        // 图对象
};

// Notice that those structs don't manage lifetime of their members.
// They are only valid only right after you call getDebugState() and should
// never be used again once another GraphExecutor function is called.

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct GraphExecutorState {
  const Graph* graph = nullptr;        // 初始化为 nullptr 的图对象指针
  ExecutionPlan fallback;              // 执行失败时的备用执行计划
  std::unordered_map<ArgumentSpec, ExecutionPlan> execution_plans;
  // 参数规范与执行计划的无序映射
};

struct TORCH_API EnableProfilingGuard {
  EnableProfilingGuard();              // 启用性能分析的构造函数
  ~EnableProfilingGuard();             // 禁用性能分析的析构函数

 private:
  bool old_executor_mode = false;      // 旧的执行器模式
  bool old_get_optimize = false;       // 旧的优化模式
};

struct GraphExecutorImplBase;
// 图执行器基础实现的结构体声明
struct TORCH_API GraphExecutor {
  // 默认构造函数
  GraphExecutor() = default;
  
  // 构造函数，接受图和函数名作为参数
  GraphExecutor(const std::shared_ptr<Graph>& graph, std::string function_name);

  // 构造函数，接受图、函数名和执行模式作为参数
  GraphExecutor(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      ExecutorExecutionMode executor_mode);

  // 同步执行函数，接受输入栈作为参数
  void run(Stack& inputs);

  // 异步执行函数，返回 Future 指针，接受输入栈和任务启动器作为参数
  c10::intrusive_ptr<Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch);

  // 根据输入栈和剩余回退深度获取执行计划
  // 如果不传递回退深度，则从当前全局融合策略设置中初始化
  const ExecutionPlan& getPlanFor(
      Stack& inputs,
      std::optional<size_t> remaining_bailout_depth = c10::nullopt);
  
  // 获取调试状态的图执行器状态
  GraphExecutorState getDebugState();

  // 清空编译缓存的调试函数
  void debugFlushCompilationCache();

  // 检查图执行器是否已优化
  bool isOptimized() const;

 private:
  std::shared_ptr<GraphExecutorImplBase> pImpl; // 实现类的指针
};

// 替换块的后备图，返回新的节点指针
TORCH_API Node* replaceBlockWithFallbackGraph(
    Block* b,
    ArrayRef<Value*> inputs);

// 在传递给解释器之前运行必需的 passes
TORCH_API void runRequiredPasses(const std::shared_ptr<Graph>& g);

// 设置融合组内联的调试函数
TORCH_API void debugSetFusionGroupInlining(bool state);

// 获取融合组内联状态的调试函数
TORCH_API bool getFusionGroupInlining();

// 设置自动微分子图内联的调试函数
TORCH_API void debugSetAutodiffSubgraphInlining(bool state);

// 获取上次执行优化图的共享指针
TORCH_API std::shared_ptr<Graph> lastExecutedOptimizedGraph();

// 获取分析模式的原子布尔引用
TORCH_API std::atomic<bool>& getProfilingMode();

// 获取执行器模式的原子布尔引用
TORCH_API std::atomic<bool>& getExecutorMode();

// 获取分析运行数的原子大小引用
TORCH_API std::atomic<size_t>& getNumProfiledRuns();

// 获取回退深度
TORCH_API size_t getBailoutDepth();

// 检查新执行器是否已启用
TORCH_API bool IsNewExecutorEnabled();

// 图优化器启用守卫结构
struct TORCH_API GraphOptimizerEnabledGuard {
  // 构造函数，设置图执行器优化状态
  GraphOptimizerEnabledGuard(bool state)
      : old_state_(getGraphExecutorOptimize()) {
    setGraphExecutorOptimize(state);
  }

  // 析构函数，恢复图执行器优化状态
  ~GraphOptimizerEnabledGuard() {
    setGraphExecutorOptimize(old_state_);
  }

  bool old_state_; // 旧的优化状态
};

namespace detail {

// 获取梯度执行器
GraphExecutor* getGradExecutor(Operation& op);

// 获取可微图操作执行器
GraphExecutor* getDifferentiableGraphOpExecutor(Operation& op);

// 获取最后实际运行图的调试信息函数
// 以前的方法允许在某些情况下查询图执行器要运行的图，但这是脆弱的，
// 因为我们有时会改变这些决策的方式。这个接口允许测试查看优化图。
// 结束 detail 命名空间的定义

} // namespace detail

// 结束 torch::jit 命名空间的定义
} // namespace torch::jit
```