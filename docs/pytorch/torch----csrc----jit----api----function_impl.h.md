# `.\pytorch\torch\csrc\jit\api\function_impl.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/function.h>
// 包含 ATen 库中的 function 头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 的 IR 头文件

#include <torch/csrc/jit/runtime/graph_executor.h>
// 包含 Torch JIT 的图执行器头文件

namespace torch::jit {

struct TORCH_API GraphFunction : public Function {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数定义，初始化 GraphFunction 对象
  GraphFunction(
      c10::QualifiedName name,
      std::shared_ptr<Graph> graph,
      std::function<void(GraphFunction&)> function_creator,
      std::optional<ExecutorExecutionMode> executor_execution_mode =
          c10::nullopt)
      : name_(std::move(name)),
        graph_(std::move(graph)),
        executor_execution_mode_(executor_execution_mode),
        function_creator_(std::move(function_creator)) {}

  bool isGraphFunction() const override {
    // 返回 true，指示此函数为图函数
    return true;
  }

  void run(Stack& stack) override;
  // 实现运行函数，执行堆栈操作

  std::function<void(GraphFunction&)> function_creator() const {
    // 返回函数创建器
    return function_creator_;
  }

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch) override;
  // 异步运行函数，返回 Future 对象

  std::shared_ptr<Graph> graph() const {
    // 返回图对象
    return graph_;
  }

  std::shared_ptr<Graph> optimized_graph() const;
  // 返回优化过的图对象

  const c10::QualifiedName& qualname() const override {
    // 返回限定名称
    return name_;
  }

  // private/unstable api. sets the initial execution mode
  // 设置初始执行模式，私有/不稳定 API
  // 如果存在为此函数创建的执行器，则不会影响执行器
  void _set_initial_executor_execution_mode(ExecutorExecutionMode mode) {
    executor_execution_mode_ = mode;
  }

  // private/unstable api. sets flag of whether or not to ignore amp.
  // 设置是否忽略 amp 标志，私有/不稳定 API
  // 如果存在为此函数创建的执行器，则不会影响执行器
  void _set_ignore_amp(bool ignore_amp) {
    force_no_amp_ = ignore_amp;
  }

  // 如果函数未定义，则调用其方法创建器函数
  void ensure_defined() override;

  size_t num_inputs() const override {
    // 返回输入参数的数量
    return graph()->inputs().size();
  }

  Function& setSchema(FunctionSchema schema) override {
    // 设置函数的模式
    schema_ = std::make_unique<FunctionSchema>(std::move(schema));
    return *this;
  }

  const FunctionSchema& getSchema() const override;

  GraphExecutorState getDebugState() {
    // 获取调试状态
    return get_executor().getDebugState();
  }

  bool is_optimized() const {
    // 弃用警告信息，始终返回 true
    TORCH_WARN(
        "GraphFunction::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
    return true;
  }

  void check_single_output() {
    // 检查是否只有一个输出
    TORCH_CHECK(
        graph()->outputs().size() == 1,
        "Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs");
  }

  GraphExecutor& get_executor() {
    // 获取执行器对象
    ensure_defined();
    std::lock_guard<std::recursive_mutex> lock(compile_mutex);
    auto& executor = executors_[currentSpecialization()];
    if (executor) {
      return *executor;
    }
    check_single_output();
    const std::string& name = name_.name();
    std::shared_ptr<Graph> opt_graph = optimized_graph();
    if (!executor_execution_mode_) {
      // 如果没有指定执行模式，创建一个不带执行模式参数的 GraphExecutor 对象
      executor = GraphExecutor(opt_graph, name);
    } else {
      // 如果指定了执行模式，创建一个带有执行模式参数的 GraphExecutor 对象
      executor = GraphExecutor(opt_graph, name, *executor_execution_mode_);
    }
    // 返回创建好的 executor 对象的引用
    return *executor;
  }

  // 使用 Function 的 call 方法，重载为根据栈、中断点和代码引用调用执行器的计划
  using Function::call;
  bool call(
      Stack& stack,
      std::optional<size_t> bailOut,
      c10::function_ref<void(const Code&)> f) override {
    // 获取当前 executor 的执行计划，并将其代码传递给函数 f 进行处理
    f(get_executor().getPlanFor(stack, bailOut).code);
    return true;
  }

  // 清空优化过的图表数组
  void clear_optimized_graphs() {
    optimized_graphs_.fill(nullptr);
  }

 private:
  // 枚举用于特化键的特化密钥
  enum SpecializationKey {
    AutocastOff,
    CpuAutocastOn,
    GpuAutocastOn,
    CpuGpuAutocastOn,

    // 提供特化数量
    // （必须是最后一个条目）
    TotalCount
  };

  // 获取当前特化类型
  SpecializationKey currentSpecialization() const;

 private:
  // 函数的限定名称
  c10::QualifiedName name_;
  // 原始、非优化的图表
  std::shared_ptr<Graph> graph_; // 用于调试和内联

  // 允许用户指定简单/性能分析执行器的可选执行模式
  mutable std::optional<ExecutorExecutionMode> executor_execution_mode_;

  // 如果在已经通过 amp 追踪过的图表上调用，不要调用 amp 传递
  mutable bool force_no_amp_ = false;
  // 优化的图表，延迟计算。用于内联。
  mutable std::array<std::shared_ptr<Graph>, SpecializationKey::TotalCount>
      optimized_graphs_;

  // GraphFunction 可从多个线程调用，因此在首次初始化图表执行器或计算优化图表时，需要保持此锁定。
  // 我们使用可重入互斥锁，以便在一个方法中调用另一个方法时不会引发死锁（例如从 get_executor() 调用 optimized_graph()）。
  mutable std::recursive_mutex compile_mutex;

  // executor_[0] - 自动类型转换关闭
  // executor_[1] - 自动类型转换 CPU 开启
  // executor_[2] - 自动类型转换 GPU 开启
  // executor_[3] - 自动类型转换 CPU 和 GPU 开启
  std::array<std::optional<GraphExecutor>, SpecializationKey::TotalCount>
      executors_;

  // 可选函数，实际上在调用 ensure_defined() 时创建方法。
  // 编译器使用它可以无序地构造方法。
  std::function<void(GraphFunction&)> function_creator_;

  // 如果不存在，则根据图表生成默认模式
  // mutable 是因为在调用 setSchema 之前，getSchema 会缓存默认模式
  mutable std::unique_ptr<FunctionSchema> schema_;
};

// namespace torch::jit 闭合声明，结束了 torch::jit 命名空间的定义

// 使用 dynamic_cast 将 Function 类型的对象转换为 GraphFunction 指针，尝试转换，如果失败则返回 nullptr
TORCH_API GraphFunction* tryToGraphFunction(Function&) noexcept;

// 将 Function 类型的对象转换为 GraphFunction 的引用，如果无法转换则抛出异常
TORCH_API GraphFunction& toGraphFunction(Function&);

// 将常量 Function 类型的对象转换为 GraphFunction 的常量引用，如果无法转换则抛出异常
TORCH_API const GraphFunction& toGraphFunction(const Function&);

// 结束了 torch::jit 命名空间的定义
} // namespace torch::jit
```