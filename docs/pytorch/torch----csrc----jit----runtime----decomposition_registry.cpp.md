# `.\pytorch\torch\csrc\jit\runtime\decomposition_registry.cpp`

```
// 包含 Torch JIT 前端的IR生成器头文件
#include <torch/csrc/jit/frontend/ir_emitter.h>
// Torch JIT 日志记录工具头文件
#include <torch/csrc/jit/jit_log.h>
// Torch JIT 常量传播优化头文件
#include <torch/csrc/jit/passes/constant_propagation.h>
// Torch JIT 窥孔优化头文件
#include <torch/csrc/jit/passes/peephole.h>
// Torch JIT 运行时分解注册表头文件
#include <torch/csrc/jit/runtime/decomposition_registry.h>
// Torch JIT 运行时分解注册表工具头文件
#include <torch/csrc/jit/runtime/decomposition_registry_util.h>
// Torch JIT 运算符头文件
#include <torch/csrc/jit/runtime/operator.h>
// Torch JIT 导入源头文件
#include <torch/csrc/jit/serialization/import_source.h>

// C10 异常处理工具头文件
#include <c10/util/Exception.h>
// Torch JIT 自动求导分解接口头文件
#include <torch/csrc/autograd/jit_decomp_interface.h>
// Torch JIT IR 层次结构头文件
#include <torch/csrc/jit/ir/ir.h>
// Torch JIT 常量传播优化头文件（重复引用）
#include <torch/csrc/jit/passes/constant_propagation.h>
// Torch JIT 内联优化头文件
#include <torch/csrc/jit/passes/inliner.h>
// Torch JIT 窥孔优化头文件（重复引用）
#include <torch/csrc/jit/passes/peephole.h>
// Torch JIT 图执行器头文件
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <memory>
#include <unordered_map>
#include <mutex>

namespace torch::jit {

// 匿名命名空间，用于内部定义和实现不需要暴露的函数和变量
namespace {
// 互斥锁，用于线程安全地访问共享资源
std::mutex lock;

// 所有函数和图形的编译单元，确保它们在内存中保持活跃状态
auto compilation_unit = std::make_shared<CompilationUnit>();

// 将函数架构映射到图的映射表
std::unordered_map<const FunctionSchema*, std::shared_ptr<Graph>>
    schema_to_decomposition;

// 用户注册的函数集合，确保它们在内存中保持活跃状态
std::unordered_map<const FunctionSchema*, std::unique_ptr<Function>>
    user_registered_funcs;

// 将函数架构映射到函数指针的映射表
std::unordered_map<const FunctionSchema*, Function*> schema_to_function;

// 加载模块中的函数和图形
void loadModule(const CompilationUnit& module) {
  // 获取所有分解映射关系
  const auto& mappings = GetDecompositionMapping().getAllKeysAndValues();
  // 遍历映射关系
  for (const auto& pair : mappings) {
    // 获取函数架构和分解函数名
    const FunctionSchema* schema = &pair.first->schema();
    const std::string& decomposition_function_name = pair.second;

    // 获取分解函数
    Function& decomposition_function =
        module.get_function(decomposition_function_name);
    
    // 将分解函数转换为图
    std::shared_ptr<Graph> graph =
        toGraphFunction(decomposition_function).graph();

    // 更新映射表
    schema_to_function[schema] = &decomposition_function;
    schema_to_decomposition[schema] = graph;
  }
}

// 加载分解函数
void loadDecompositionFunctions() {
  std::lock_guard<std::mutex> guard(lock);
  // 如果映射表不为空，则直接返回
  if (!schema_to_decomposition.empty()) {
    return;
  }

  // 创建源对象
  auto src = std::make_shared<Source>(GetSerializedDecompositions());
  std::stringstream ss;
  std::vector<at::IValue> constantTable;
  
  // 创建源导入器
  auto resolver = std::make_shared<SourceImporterImpl>(
      compilation_unit,
      &constantTable,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      1);
  
  // 定义编译单元
  compilation_unit->define(
      c10::nullopt, GetSerializedDecompositions(), resolver, nullptr);
  
  // 加载模块中的函数和图形
  loadModule(*compilation_unit);
}

} // anonymous namespace

// 静态函数，用于分解操作节点
static void DecomposeOp(Node* n) {
  // 获取节点的函数架构
  auto schema = n->maybeSchema();
  if (!schema) {
    return;
  }
  
  // 获取节点的分解函数
  auto decomposition = GetDecomposition(n->schema());
  if (!decomposition) {
    return;
  }
  
  // 设置插入点
  WithInsertPoint guard(n);
  
  // 插入分解图并获取输出
  auto outputs =
      insertGraph(*n->owningGraph(), *decomposition->get(), n->inputs());
  
  // 断言输出数量与节点输出一致
  TORCH_INTERNAL_ASSERT(outputs.size() == n->outputs().size());
  
  // 替换节点的所有使用
  for (size_t i : c10::irange(outputs.size())) {
    n->outputs().at(i)->replaceAllUsesWith(outputs[i]);
  }
  
  // 销毁节点
  n->destroy();
}
// 递归遍历给定块中的每个节点，并对每个节点执行分解操作
static void RunDecompositions(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // 因为当前节点可能会被销毁，所以需要提前移动迭代器
    for (Block* b : n->blocks()) {
      RunDecompositions(b); // 递归调用，对嵌套的块进行分解操作
    }
    DecomposeOp(n); // 对当前节点执行分解操作
  }
}

// 对给定图执行分解操作
void RunDecompositions(std::shared_ptr<Graph> g) {
  RunDecompositions(g->block()); // 调用递归函数从图的根块开始分解
  // 进行两次局部优化过程，包括使用简单执行器来禁用形状优化
  for (C10_UNUSED const auto _ : c10::irange(2)) {
    PeepholeOptimize(g, /*disable_shape_peephole*/ true);
    ConstantPropagation(g);
  }
}

// 获取给定函数模式的分解图，如果找不到则返回空
std::optional<std::shared_ptr<Graph>> GetDecomposition(
    const FunctionSchema& schema) {
  loadDecompositionFunctions(); // 加载分解函数的映射
  GRAPH_DEBUG("Trying to find schema: ", schema); // 调试信息，尝试查找给定的模式
  auto cache_it = schema_to_decomposition.find(&schema); // 在缓存中查找给定模式的分解图
  if (cache_it != schema_to_decomposition.end()) {
    return cache_it->second; // 如果找到，则返回对应的分解图
  }
  GRAPH_DEBUG("Could not find schema: ", schema); // 调试信息，未找到给定模式
  return c10::nullopt; // 返回空的optional表示未找到
}

// 获取给定函数模式的分解函数，如果找不到则返回空
std::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema) {
  loadDecompositionFunctions(); // 加载分解函数的映射
  auto cache_it = schema_to_function.find(&schema); // 在缓存中查找给定模式的分解函数
  GRAPH_DEBUG("Trying to find schema: ", schema); // 调试信息，尝试查找给定的模式
  if (cache_it == schema_to_function.end()) {
    GRAPH_DEBUG("Could not find schema: ", schema); // 调试信息，未找到给定模式
    return c10::nullopt; // 返回空的optional表示未找到
  }
  auto& func = toGraphFunction(*cache_it->second); // 将函数转换为GraphFunction类型
  // 设置初始执行器的执行模式为简单执行器，以允许在张量子类上运行分解
  func._set_initial_executor_execution_mode(ExecutorExecutionMode::SIMPLE);
  return &func; // 返回分解函数的指针
}

// 注册给定函数模式的分解图，以便初始化一个使用简单执行器运行的GraphFunction
void RegisterDecomposition(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g) {
  loadDecompositionFunctions(); // 加载分解函数的映射
  std::lock_guard<std::mutex> guard(lock); // 使用互斥锁保护并发访问
  Inline(*g); // 内联化图中的函数调用
  for (const auto i : c10::irange(2)) {
    (void)i; // 抑制未使用变量警告
    PeepholeOptimize(g); // 使用孔径优化图
    ConstantPropagationImmutableTypes(g); // 使用常量传播优化不可变类型
  }

  // 创建新的GraphFunction对象，并使用简单执行器模式
  auto new_func = std::make_unique<GraphFunction>(
      schema.name(), g, nullptr, ExecutorExecutionMode::SIMPLE);
  user_registered_funcs.emplace(&schema, std::move(new_func)); // 将新函数注册到用户注册的函数集合中
  schema_to_function[&schema] = user_registered_funcs[&schema].get(); // 更新函数模式到函数的映射
  schema_to_decomposition[&schema] = g; // 更新函数模式到分解图的映射
}

// JitDecomp结构体，实现了JitDecompInterface接口
// 注册JitDecomp实例以进行JIT分解
struct JitDecomp final : torch::autograd::impl::JitDecompInterface {
  bool has_jit_decomposition(const c10::FunctionSchema& schema) const override;
  void run_jit_decomposition(
      const c10::OperatorHandle& op,
      torch::jit::Stack* stack) const override;
};

// 创建JitDecomp实例
JitDecomp jitDecomp;
// 注册JitDecomp实例，使其能够接收JIT分解请求
torch::autograd::impl::JitDecompRegisterer registerJitDecomp(&jitDecomp);

// 实现JitDecomp接口中的run_jit_decomposition方法
void JitDecomp::run_jit_decomposition(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) const {
    // 调用 TorchScript 的运行时函数 run_jit_decomposition，执行 JIT 操作分解
    torch::jit::run_jit_decomposition(op, stack);
}

# 检查给定函数模式是否有 JIT 分解
bool JitDecomp::has_jit_decomposition(const FunctionSchema& schema) const {
    return ::torch::jit::has_jit_decomposition(schema);
}

# 运行 JIT 分解
void run_jit_decomposition(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
    
    # 获取操作符的模式
    const auto& schema = op.schema();
    // TODO: 根据操作符模板化，并保持静态跟踪执行
    auto* trace_exec = torch::jit::GetDecompositionExecutor(schema);
    # 运行跟踪执行器
    trace_exec->run((*stack));
    
    # 如果栈顶元素是元组，则展开并压入栈中
    if (stack->back().isTuple()) {
        at::IValue tup = stack->back();
        stack->pop_back();
        for (const auto& elem : tup.toTuple()->elements()) {
            stack->push_back(elem);
        }
    }
}

# 检查函数模式是否有 JIT 分解
bool has_jit_decomposition(const FunctionSchema& schema) {
    return GetDecompositionFunction(schema).has_value();
}

# 获取 JIT 分解执行器
Function* GetDecompositionExecutor(const FunctionSchema& schema) {
    auto maybe_func = GetDecompositionFunction(schema);
    # 内部断言，确保获取到了有效的函数指针
    TORCH_INTERNAL_ASSERT(maybe_func);
    return *maybe_func;
}

# 根据字面量获取 JIT 分解执行器
Function* GetDecompositionExecutor(const char* schema_literal) {
    # 获取字面量对应的操作符并获取其模式
    auto& schema = getOperatorForLiteral(schema_literal)->schema();
    return GetDecompositionExecutor(schema);
}

} // namespace torch::jit
```