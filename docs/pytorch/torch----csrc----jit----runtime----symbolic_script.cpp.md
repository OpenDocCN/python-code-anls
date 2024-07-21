# `.\pytorch\torch\csrc\jit\runtime\symbolic_script.cpp`

```py
// 引入 Torch 的符号脚本运行时头文件

#include <torch/csrc/jit/runtime/symbolic_script.h>

// 引入 Torch 前端 IR 生成器头文件
#include <torch/csrc/jit/frontend/ir_emitter.h>
// 引入 Torch 运算符头文件
#include <torch/csrc/jit/runtime/operator.h>

// Torch JIT 命名空间
namespace torch::jit {

// 匿名命名空间，用于限定作用域和静态变量
namespace {

// 互斥锁，用于多线程同步
std::mutex lock;

// 预定义的函数名称列表
const std::vector<std::string> functions = {
    ")",
    ")",
    ")",
    ")",
    ")",
    ")",
};

// 存储从函数模式到梯度图的映射的哈希表
std::unordered_map<std::string, GradientPair> schema_to_graphs;

// 缓存编译过的梯度对的哈希表
// 实际上应该将这些图编译一次并保存在 Operator 结构中，这是对 native_functions.yaml 合并的一个临时解决方案
std::unordered_map<const FunctionSchema*, GradientPair> cached_gradient_pairs;

// 持有所有函数的 CompilationUnit，以确保它们的生命周期
CompilationUnit compilation_unit;

} // 匿名命名空间结束

// 提取闭包的函数，返回包含图和上下文的共享指针对
static std::pair<std::shared_ptr<Graph>, Value*> extractClosure(
    Value* closure) {
  // 检查闭包是否是字面值元组构造
  TORCH_CHECK(
      closure->node()->kind() == prim::TupleConstruct,
      "closure must be a literal tuple construct");
  // 获取闭包中的函数和上下文
  Value* fn = closure->node()->inputs().at(0);
  Value* context = closure->node()->inputs().at(1);

  // 检查闭包中是否包含 prim::Closure
  TORCH_CHECK(
      fn->node()->kind() == prim::Closure,
      "closure tuple must contain a prim::Closure");
  // 返回函数的图和上下文
  return std::make_pair(fn->node()->g(attr::Subgraph), context);
}

// 获取原始返回类型的辅助函数
static Argument originalReturnType(const TupleTypePtr& tup) {
  // 检查元组类型的元素数量是否大于1
  TORCH_CHECK(tup->elements().size() > 1);
  if (tup->elements().size() == 2)
    return Argument("", tup->elements().at(0));
  // 移除最后一个类型并返回
  std::vector<TypePtr> types = tup->elements().vec();
  types.pop_back();
  return Argument("", TupleType::create(std::move(types)));
}

// 在 TorchScript AD 公式中，我们将 {func_0, func_1, ...} 定义为 `func` 的重载函数
// 在将 schema 字符串添加到 schema_to_graphs 映射之前，移除后缀
static std::string overloadedSchemaString(const FunctionSchema& schema) {
  // 获取函数模式的名称
  const auto& schema_name = schema.name();
  // 找到最后一个下划线的位置
  auto pos = schema_name.find_last_of('_');
  // 提取后缀
  auto schema_name_suffix = schema_name.substr(pos + 1);
  // 获取规范的 schema 字符串
  std::string schema_string = canonicalSchemaString(schema);
  // 如果后缀为数字，则从 schema 字符串中移除后缀
  if (!schema_name_suffix.empty() &&
      schema_name_suffix.find_first_not_of("0123456789") == std::string::npos) {
    schema_string.replace(
        schema_string.find(schema_name),
        schema_name.length(),
        schema_name.substr(0, pos));
  }

  // 返回处理后的 schema 字符串
  return schema_string;
}

// 检查方法名称是否为辅助函数的辅助函数
static bool isHelperFunction(const std::string& method_name) {
  // 辅助函数名称前缀
  std::string helper_prefix = "AD_";
  // 检查方法名称是否以辅助函数前缀开头
  return method_name.compare(0, helper_prefix.length(), helper_prefix) == 0;
}

// 加载 CompilationUnit 中的模块的函数
static void loadModule(const CompilationUnit& module) {
  // 遍历模块中的所有函数
  for (const auto& method : module.get_functions()) {
    // 如果是辅助函数，跳过
    if (isHelperFunction(method->name()))
      continue;

    // 创建 GradientPair 对象，设置正向图函数
    GradientPair pair;
    pair.forward = toGraphFunction(*method).graph();

    // 查找反向函数
    Node* forward_tuple = pair.forward->outputs().at(0)->node();
    // 检查 forward_tuple 是否为 prim::TupleConstruct 类型，否则抛出错误
    if (forward_tuple->kind() != prim::TupleConstruct) {
      throw ErrorReport(forward_tuple->sourceRange())
          << "gradient must return literal a tuple";
    }

    // 声明并初始化 context 变量，用于接收从 extractClosure 函数中提取的上下文信息
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* context;
    std::tie(pair.backward, context) =
        extractClosure(forward_tuple->inputs().back());

    // 检查前向计算图的输入数量是否等于反向计算图的输出数量
    TORCH_CHECK(
        pair.forward->inputs().size() ==
            unpackOutputs(pair.backward->outputs().vec()).size(),
        "The autodiff implementation of ",
        method->name(),
        " backward() returns an incorrect number of values: ",
        unpackOutputs(pair.backward->outputs().vec()).size(),
        " instead of ",
        pair.forward->inputs().size());

    // 对前向函数进行修改，将闭包元组替换为上下文变量：
    //  backward = (<lambda>, context_tuple)
    //  return original, backward
    //  -----
    //  return original, context_tuple
    std::vector<Value*> new_inputs = forward_tuple->inputs().vec();
    new_inputs.back() = context;
    // 创建新的元组节点，替换原始的输出节点
    Value* new_tuple =
        pair.forward->appendNode(pair.forward->createTuple(new_inputs))
            ->output();
    // 注销原始输出并注册新的输出
    pair.forward->eraseOutput(0);
    pair.forward->registerOutput(new_tuple);
    // 销毁原始的 forward_tuple 节点
    forward_tuple->destroy();

    // 从原始函数的 schema 推导出新的 schema：
    const FunctionSchema& loaded_schema = method->getSchema();
    FunctionSchema actual_schema(
        Symbol::aten(loaded_schema.name()),
        loaded_schema.overload_name(),
        loaded_schema.arguments(),
        {originalReturnType(new_tuple->type()->expect<TupleType>())});

    // 修改用于函数重载的规范字符串，避免修改 schema 名称
    auto schema_string = overloadedSchemaString(actual_schema);

    // 将生成的 schema 和 pair 放入 schema_to_graphs 映射中
    schema_to_graphs[schema_string] = std::move(pair);
  }
}

// 结束 torch::jit 命名空间

static void loadFunctions() {
  // 遍历 functions 容器中的每个字符串
  for (const std::string& str : functions) {
    // 在 compilation_unit 中定义一个函数，没有默认参数，使用 nativeResolver() 解析，没有额外的属性
    compilation_unit.define(c10::nullopt, str, nativeResolver(), nullptr);
  }
  // 载入模块到 compilation_unit
  loadModule(compilation_unit);
}

// 返回指定函数模式的梯度信息对
std::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema) {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> guard(lock);
  // 如果 schema_to_graphs 映射为空，则加载函数
  if (schema_to_graphs.empty()) {
    loadFunctions();
  }
  // 在 cached_gradient_pairs 中查找给定 schema 的缓存梯度信息
  auto cache_it = cached_gradient_pairs.find(&schema);
  if (cache_it != cached_gradient_pairs.end()) {
    // 如果找到缓存的梯度信息，则返回它
    return cache_it->second;
  } else {
    // 获取 schema 的规范化字符串表示
    auto schema_str = canonicalSchemaString(schema);
    // 对于调试 AD 改变：
    // 输出正在查找的 schema_str 信息
    // std::cout << "Looking for " << schema_str << std::endl;
    // 在 schema_to_graphs 中查找给定 schema_str 的符号脚本
    auto sym_script_it = schema_to_graphs.find(schema_str);

    // 如果找到对应的符号脚本，则将其作为 schema 的梯度信息缓存，并返回它
    if (sym_script_it != schema_to_graphs.end()) {
      cached_gradient_pairs.emplace_hint(
          cache_it, &schema, sym_script_it->second);
      return sym_script_it->second;
    }
  }
  // 如果未找到任何信息，则返回空的 std::optional
  return c10::nullopt;
}

// 检查是否存在给定函数模式的梯度信息
bool hasGradientInfoForSchema(const FunctionSchema& schema) {
  // 判断 gradientInfoForSchema 返回的 std::optional 是否有值
  return gradientInfoForSchema(schema).has_value();
}

// 结束 torch::jit 命名空间
} // namespace torch::jit
```