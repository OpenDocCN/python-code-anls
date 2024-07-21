# `.\pytorch\torch\csrc\jit\api\function_impl.cpp`

```
#include <c10/util/Flags.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/peephole.h>

#ifndef C10_MOBILE
#include <ATen/autocast_mode.h>
#include <torch/csrc/jit/passes/autocast.h>
#endif

// 定义一个命令行标志，用于控制是否存储优化后的图形
C10_DEFINE_bool(
    torch_jit_do_not_store_optimized_graph,
    false,
    "Do not store the optimized graph.");

namespace torch::jit {
namespace {
// 根据 GraphFunction 返回默认的函数模式
c10::FunctionSchema defaultSchemaFor(const GraphFunction& function) {
  std::vector<c10::Argument> args;
  std::vector<c10::Argument> returns;
  Graph& g = *function.graph();
  size_t num_inputs = function.num_inputs();
  // 为输入参数创建模式
  for (const auto i : c10::irange(num_inputs)) {
    const Value* v = g.inputs().at(i);
    // 如果存在调试名称，则使用调试名称，否则使用默认的参数名称
    std::string name = v->hasDebugName() ? v->debugNameBase()
                                         : ("argument_" + std::to_string(i));
    args.emplace_back(std::move(name), unshapedType(g.inputs()[i]->type()));
  }
  // 为输出参数创建模式
  for (const auto i : c10::irange(g.outputs().size())) {
    returns.emplace_back("", unshapedType(g.outputs()[i]->type()));
  }
  return {function.name(), "", std::move(args), std::move(returns)};
}

// 尝试将函数转换为 GraphFunction 实现
template <typename T, typename F>
T* tryToGraphFunctionImpl(F& function) noexcept {
  if (!function.isGraphFunction()) {
    return nullptr;
  }

  return static_cast<T*>(&function);
}

// 将函数强制转换为 GraphFunction 实现
template <typename T, typename F>
T& toGraphFunctionImpl(F& function) {
  if (auto* g = tryToGraphFunctionImpl<T>(function)) {
    return *g;
  }

  // 如果失败，断言错误信息
  TORCH_INTERNAL_ASSERT(
      false,
      "Failed to downcast a Function to a GraphFunction. "
      "This probably indicates that the JIT calling context needs a "
      "special case on tryToGraphFunction() instead.");
}

} // namespace

// 定义一个静态函数，用于创建占位符，抛出递归方法调用错误
static void placeholderCreator(GraphFunction&) {
  throw RecursiveMethodCallError();
}

// 实现 GraphFunction 类的 run 方法
void GraphFunction::run(Stack& stack) {
  get_executor().run(stack);
}

// 实现 GraphFunction 类的 runAsync 方法
c10::intrusive_ptr<c10::ivalue::Future> GraphFunction::runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
  return get_executor().runAsync(stack, std::move(taskLauncher));
}

// 确保函数已定义
void GraphFunction::ensure_defined() {
  if (function_creator_) {
    auto creator = function_creator_;
    function_creator_ = placeholderCreator;
    creator(*this);
    function_creator_ = nullptr;
  }
  // 检查函数是否只有单一输出
  check_single_output();
}

// 获取函数的模式
const c10::FunctionSchema& GraphFunction::getSchema() const {
  if (schema_ == nullptr) {
    schema_ = std::make_unique<c10::FunctionSchema>(defaultSchemaFor(*this));
  }
  return *schema_;
}
std::shared_ptr<Graph> GraphFunction::optimized_graph() const {
  // 使用递归互斥锁，确保在编译过程中对优化图的安全访问
  std::lock_guard<std::recursive_mutex> lock(compile_mutex);

  // 定义优化后的图对象的引用
  decltype(optimized_graphs_)::value_type graph;

  // 根据标志位选择存储优化图或者临时图的引用
  auto& graph_ref = !FLAGS_torch_jit_do_not_store_optimized_graph
      ? optimized_graphs_[currentSpecialization()]
      : graph;

  // 如果已经存在优化后的图，直接返回该图
  if (graph_ref) {
    return graph_ref;
  }

  // 复制原始图形成新的优化图形
  graph_ref = graph_->copy();

  // 如果开启图执行器优化，则进行预优化
  if (getGraphExecutorOptimize()) {
    preoptimizeGraph(graph_ref, force_no_amp_);
  }

  // 返回优化后的图形引用
  return graph_ref;
}

// 根据当前环境和设置返回特定的优化专用键
GraphFunction::SpecializationKey GraphFunction::currentSpecialization() const {
  // 如果强制禁用自动混合精度，则返回禁用自动混合精度的键
  if (force_no_amp_) {
    return SpecializationKey::AutocastOff;
  }

#ifdef C10_MOBILE
  // 在移动端构建时禁用自动微分，因为自动混合精度 API 不存在
  return SpecializationKey::AutocastOff;
#else
  // 检查 CPU 和 GPU 是否启用自动混合精度
  bool cpu_enabled = at::autocast::is_autocast_enabled(at::kCPU);
  bool gpu_enabled = at::autocast::is_autocast_enabled(at::kCUDA);

  // 根据 CPU 和 GPU 的自动混合精度状态返回相应的专用键
  if (cpu_enabled && gpu_enabled) {
    return SpecializationKey::CpuGpuAutocastOn;
  } else if (!cpu_enabled && !gpu_enabled) {
    return SpecializationKey::AutocastOff;
  } else {
    return gpu_enabled ? SpecializationKey::GpuAutocastOn
                       : SpecializationKey::CpuAutocastOn;
  }
#endif
}

// 对图形进行预优化，可能包括内联、视窗优化、常量传播等
void preoptimizeGraph(std::shared_ptr<Graph>& graph, bool disable_autocast) {
  // 内联函数调用
  Inline(*graph);

  // 视窗优化，清理"None"检查和创建常量传播的机会
  PeepholeOptimize(graph, true);

  // 构建别名数据库，用于清理常量条件判断和其他简单的优化
  ConstantPropagationImmutableTypes(graph);

#ifndef C10_MOBILE
  // 注入自动混合精度转换，如果未禁用自动混合精度
  if (!disable_autocast) {
    Autocast(graph);
  }
#endif

  // 常量池化，合并常量节点
  ConstantPooling(graph);
}
```