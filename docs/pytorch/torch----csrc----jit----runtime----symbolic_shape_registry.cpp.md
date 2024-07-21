# `.\pytorch\torch\csrc\jit\runtime\symbolic_shape_registry.cpp`

```
// 包含头文件：定义了在运行时需要使用的各种依赖库
#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <unordered_map>

// torch::jit 命名空间包含了 TorchScript JIT 编译器的所有实现
namespace torch::jit {

// 匿名命名空间：用于限制内部变量的作用域，避免与外部符号冲突
namespace {

// 全局互斥锁对象，用于保护并发访问的临界区域
std::mutex lock;

// 根据是否启用 XNNPACK 定义了一个字符串常量，包含 XNNPACK 的计算函数定义
const std::string _xnnpack_shape_compute_functions =
#ifdef USE_XNNPACK
    R"(def prepacked_conv2d_clamp_run(input: List[int], conv2dOpContext: Any):
    assert isinstance(conv2dOpContext, __torch__.torch.classes.xnnpack.Conv2dOpContext)
    (weight, bias, stride, padding, dilation, groups) = unchecked_cast(
        Tuple[List[int], Optional[List[int]], List[int], List[int], List[int], int],
        ops.prepacked.unpack_prepacked_sizes_conv2d(conv2dOpContext),
    )
    return conv2d(input, weight, bias, stride, padding, dilation, groups)

def prepacked_linear_clamp_run(input: List[int], linearOpContext: Any):
    assert isinstance(linearOpContext, __torch__.torch.classes.xnnpack.LinearOpContext)
    (weight, bias) = unchecked_cast(
        Tuple[List[int], Optional[List[int]]],
        ops.prepacked.unpack_prepacked_sizes_linear(linearOpContext),
    )
    return linear(input, weight, bias)
    )"
#else
    ""
#endif
    ;

// 用于将运算符架构映射到形状计算图的操作符映射表，允许多个函数共享相同的形状计算图
// 这样做既节省内存，也通过缓存运行相同图的一系列操作的结果来加速形状分析
// 我们需要一个从架构到形状计算图的映射，因为 aten 架构在形状计算图中不可恢复，
// 形状计算图使用 List[int] 替换 Tensor 输入，而像 Conv 这样的操作符本身具有 List[int] 输入
// TODO: 考虑直接在操作符上存储形状计算图，并合并到 native_functions.yaml 中
static const OperatorMap<std::string>& conditionally_defined_ops() {
  // clang-format off
  // 静态常量映射表，将运算符架构映射到函数图
  static const OperatorMap<std::string> schema_to_function_graph{
#ifdef USE_XNNPACK
      // 定义 XNNPACK 实现的卷积和线性运算函数的映射，将其加入到函数图中
      {"prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y", "prepacked_conv2d_clamp_run"},
      {"prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y", "prepacked_linear_clamp_run"},
#endif
  };
  // clang-format on
  // 返回包含函数图的函数模式映射
  return schema_to_function_graph;
}

// 缓存函数模式到图的共享指针映射
std::unordered_map<const FunctionSchema*, std::shared_ptr<Graph>>
    cached_schema_to_graph;

// 缓存有界形状函数模式到图的映射
std::unordered_map<const FunctionSchema*, BoundedShapeGraphs>
    cached_bounded_schema_to_graph;

// 持有所有函数的编译单元
auto compilation_unit = std::make_shared<CompilationUnit>();

// 获取就地变体的函数模式（如果存在）
const at::optional<const FunctionSchema*> getInplaceVariant(
    const FunctionSchema& base_schema) {
  // 获取所有与基础模式相关的运算符
  auto& inplace_variants =
      getAllOperatorsFor(c10::Symbol::fromQualString(base_schema.name() + "_"));

  // 遍历所有就地变体，查找匹配的模式
  for (const auto& variant : inplace_variants) {
    const FunctionSchema* schema = &variant->schema();
    // 检查模式是否是基础模式的子类型
    if (!schema->isSubtypeOf(base_schema, false)) {
      continue;
    }

    Argument self_arg = schema->arguments()[0];
    // 检查第一个参数是否是写入别名信息
    if (!self_arg.alias_info()->isWrite()) {
      continue;
    }

    Argument ret_arg = schema->returns()[0];
    // 检查返回值是否是写入别名信息
    if (!ret_arg.alias_info()->isWrite()) {
      continue;
    }

    // 找到匹配的就地变体模式并返回
    return schema;
  }
  // 没有找到匹配的就地变体，返回空值
  return at::nullopt;
}

// 将张量类型映射为整数列表类型
TypePtr mapTensorToListOfInts(TypePtr type) {
  if (type->cast<TensorType>()) {
    // 如果类型是张量类型，则映射为整数列表类型
    return ListType::ofInts();
  }
  at::ArrayRef<TypePtr> contained = type->containedTypes();
  // 如果类型不包含其他类型，则直接返回
  if (contained.empty()) {
    return type;
  }
  // 映射类型及其包含的所有类型
  return type->withContained(
      fmap(type->containedTypes(), mapTensorToListOfInts));
}

// 检查函数图中是否存在 While 循环，并输出警告信息
void checkForWhileLoop(
    const FunctionSchema* schema,
    std::shared_ptr<Graph> graph) {
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto* node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    // 如果节点不是循环节点，则继续下一个节点
    if (node->kind() != prim::Loop) {
      continue;
    }
    // 获取循环视图
    LoopView loop(node);
    // 如果循环不是 For 循环类型，则输出警告信息
    if (loop.loopType() != LoopView::For) {
      TORCH_WARN(
          "While loops are not yet implemented in unrolling which may make this shape function difficult to partially evaluate: ",
          *node,
          " for schema ",
          *schema);
    }
  }
}

// 检查输入是否作为输出返回的情况，并输出警告信息（精度不高）
void checkInputReturnedAsOutput(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  // 这里可以使用别名数据库，但需要输出警告信息，因为这是不精确的
  for (size_t i : c10::irange(graph->inputs().size())) {
    Value* input = graph->inputs().at(i);
    // 使用范围迭代器遍历图的输出节点数量
    for (size_t j : c10::irange(graph->outputs().size())) {
      // 获取图的第 j 个输出节点
      Value* output = graph->outputs().at(j);
      // 检查输入节点和输出节点不相等，否则抛出错误消息
      TORCH_CHECK(
          input != output,
          "For schema: ",
          *schema,
          " input index ",
          i,
          " is returned as output index ",
          j,
          ". Shape functions must return new unaliased lists");
    }
}

// 检查输入和输出类型是否符合函数模式
void checkInputAndOutputTypes(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  // 允许额外的未使用的参数以便映射多个函数到例如一元函数
  TORCH_CHECK(
      graph->inputs().size() <= schema->arguments().size(),
      "Shape function must have fewer arguments than schema. Got ",
      graph->inputs().size(),
      " graph arguments and ",
      schema->arguments().size(),
      " schema arguments of schema: ",
      *schema);

  // 遍历输入参数，检查类型是否匹配
  for (auto i : c10::irange(graph->inputs().size())) {
    auto inp_type = schema->arguments().at(i).type();
    auto mapped_type = mapTensorToListOfInts(inp_type);
    auto graph_type = graph->inputs().at(i)->type();
    TORCH_INTERNAL_ASSERT(
        mapped_type->isSubtypeOf(graph->inputs().at(i)->type()),
        "For schema type: ",
        inp_type->str(),
        " Expected supertype of ",
        mapped_type->str(),
        " but got graph_type ",
        graph_type->str(),
        " at index ",
        i,
        " of schema: ",
        *schema);
  }

  // 检查输出参数数量是否匹配
  TORCH_CHECK(
      graph->outputs().size() == schema->returns().size(),
      "Shape function equal number of outputs as schema. Got ",
      graph->outputs().size(),
      " graph outputs and ",
      schema->returns().size(),
      " schema returns of schema: ",
      *schema);

  // 遍历输出参数，检查类型是否匹配
  for (auto i : c10::irange(schema->returns().size())) {
    auto out_type = schema->returns().at(i).type();
    auto mapped_type = mapTensorToListOfInts(out_type);
    auto graph_type = graph->outputs().at(i)->type();
    TORCH_INTERNAL_ASSERT(
        mapped_type->isSubtypeOf(graph->outputs().at(i)->type()),
        "For schema type: ",
        out_type->str(),
        " Expected supertype of ",
        mapped_type->str(),
        " but got graph_type ",
        graph_type->str(),
        " at output index ",
        i,
        " of schema: ",
        *schema);
  }
}

// 转换形状函数的图形表示
void transformShapeFunction(
    const FunctionSchema* schema_string,
    std::shared_ptr<Graph> graph) {
  // 内联图形
  Inline(*graph);

  // ATEN 操作符可以返回多个未装箱的值，与在 TorchScript 或用户注册的操作符中定义的函数形成对比
  // 后者必须使用元组
  // 在此处，修改具有多个输出的 ATEN 操作符的形状图，以使它们相互对应
  if (schema_string->returns().size() > 1) {
    TORCH_INTERNAL_ASSERT(
        graph->outputs().size() == 1 &&
        graph->outputs().at(0)->type()->cast<TupleType>());
    auto tuple_node = graph->outputs().at(0)->node();
    WithInsertPoint guard(graph->return_node());
    auto tuple_unpack_values = createTupleUnpack(tuple_node->output());
    graph->eraseOutput(0);
    // 注册拆包后的元组值作为输出
    for (Value* v : tuple_unpack_values) {
      graph->registerOutput(v);
    }
    // 输出元组拆包后的图形状态
    GRAPH_DUMP("After Output Tuple Unpacking", graph);
  }
}

// 生成形状计算函数的图形表示
std::shared_ptr<Graph> genShapeComputeFn(
    const FunctionSchema* schema_string,
    const std::string& shape_compute_function_name,
    std::unordered_map<std::string, std::shared_ptr<Graph>>& reused_functions,
    const CompilationUnit& module) {
  std::shared_ptr<Graph> graph;
  // 打印调试信息，注册模式和形状计算函数名称
  GRAPH_DEBUG(
      "Registering schema: ",
      *schema_string,
      " with shape compute func: ",
      shape_compute_function_name);
  // 检查是否已经存在重用的形状计算函数
  if (reused_functions.count(shape_compute_function_name)) {
    // 若存在重用的函数，打印调试信息
    GRAPH_DEBUG("Registering reused schema");
    // 从重用函数的映射中获取已存在的图对象
    graph = reused_functions[shape_compute_function_name];
  } else {
    // 如果不存在重用的函数，从编译单元中获取对应的函数对象
    Function& shape_compute_function =
        module.get_function(shape_compute_function_name);
    // 将函数对象转换为图对象并获取其图
    graph = toGraphFunction(shape_compute_function).graph();

    // 对图进行形状函数变换
    transformShapeFunction(schema_string, graph);
    // 注意：我们在测试文件中对注册的形状函数进行了静态分析
    // LintShapeComputeGraph(schema_string, graph);

    // 将新创建的图对象添加到重用函数映射中
    reused_functions[shape_compute_function_name] = graph;
  }
  // 允许多余的未使用参数以映射多个函数，例如一元操作
  // 断言图的输入大小不超过模式字符串中参数的大小
  TORCH_INTERNAL_ASSERT(
      graph->inputs().size() <= schema_string->arguments().size());
  // 返回最终确定的图对象
  return graph;
}
// 注册给定函数的计算图形状，将其缓存到全局映射中
void registerSchema(
    const FunctionSchema* schema_string,
    const std::string& shape_compute_function_name,
    std::unordered_map<std::string, std::shared_ptr<Graph>>& reused_functions,
    const CompilationUnit& module) {
  // 生成形状计算函数的图形
  auto graph = genShapeComputeFn(
      schema_string, shape_compute_function_name, reused_functions, module);

  // 将生成的图形注册到全局缓存映射中
  cached_schema_to_graph[schema_string] = graph;
}

// 注册具有下限和上限函数的函数模式
void registerBoundedSchema(
    const FunctionSchema* schema_string,
    const std::string& lower_bound_function_name,
    const std::string& upper_bound_function_name,
    std::unordered_map<std::string, std::shared_ptr<Graph>>& reused_functions,
    const CompilationUnit& module) {
  // 生成下限形状计算函数的图形
  auto lower_graph = genShapeComputeFn(
      schema_string, lower_bound_function_name, reused_functions, module);
  // 生成上限形状计算函数的图形
  auto upper_graph = genShapeComputeFn(
      schema_string, upper_bound_function_name, reused_functions, module);
  // 将生成的下限和上限图形注册到全局缓存映射中
  cached_bounded_schema_to_graph[schema_string] = {lower_graph, upper_graph};
}

// 加载模块并注册相关函数的形状计算图形
void loadModule(const CompilationUnit& module) {
  std::unordered_map<std::string, std::shared_ptr<Graph>> reused_functions;

  // 获取所有已定义操作符及其名称
  std::vector<std::pair<std::shared_ptr<Operator>, std::string>>
      operator_pairs = conditionally_defined_ops().getAllKeysAndValues();
  // 获取所有TensorExpr中的元素操作集合
  auto te_ops = get_tensorexpr_elementwise_set().getAllKeysAndValues();
  // 将TensorExpr操作集合合并到操作符对中
  operator_pairs.insert(operator_pairs.end(), te_ops.begin(), te_ops.end());
  // 获取更多的形状函数映射
  auto more_mappings = GetShapeFunctionMappings().getAllKeysAndValues();
  // 将更多的映射合并到操作符对中
  operator_pairs.insert(
      operator_pairs.end(), more_mappings.begin(), more_mappings.end());

  // 遍历操作符对
  for (const auto& pair : operator_pairs) {
    // 获取操作符的函数模式
    const FunctionSchema* schema_string = &pair.first->schema();
    // 获取形状计算函数的名称
    const std::string& shape_compute_function_name = pair.second;

    // 注册给定函数模式的形状计算图形
    registerSchema(
        schema_string, shape_compute_function_name, reused_functions, module);

    // 如果形状计算函数名称为 "unary"，注册其原地变体（如果存在）
    if (shape_compute_function_name == "unary") {
      auto inplace_schema = getInplaceVariant(*schema_string);
      if (inplace_schema.has_value()) {
        registerSchema(
            inplace_schema.value(), "unary", reused_functions, module);
      }
    }
    // 如果形状计算函数名称为 "broadcast"，注册其原地变体（如果存在）
    if (shape_compute_function_name == "broadcast") {
      auto inplace_schema = getInplaceVariant(*schema_string);
      if (inplace_schema.has_value()) {
        registerSchema(
            inplace_schema.value(),
            "broadcast_inplace",
            reused_functions,
            module);
      }
    }
  }

  // 现在注册有界模式的函数
  for (const auto& pair : GetBoundedShapeMappings().getAllKeysAndValues()) {
    // 获取有界模式的函数模式
    const FunctionSchema* schema_string = &pair.first->schema();
    // 获取下限和上限函数的名称
    const std::string& lower_bound_function_name = pair.second.first;
    const std::string& upper_bound_function_name = pair.second.second;
    // 注册有界模式的函数
    registerBoundedSchema(
        schema_string, lower_bound_function_name, upper_bound_function_name,
        reused_functions, module);
    # 注册一个有界模式的模式
    registerBoundedSchema(
        schema_string,                # 要注册的模式的字符串表示
        lower_bound_function_name,    # 模式的下界函数的名称
        upper_bound_function_name,    # 模式的上界函数的名称
        reused_functions,             # 重用的函数列表
        module                        # 模块对象，包含模式的定义
    )
  }
} // 匿名命名空间结束

void loadFunctions() {
  try {
    // 获取序列化的形状计算函数并加上 _xnnpack_shape_compute_functions
    auto shape_compute_functions =
        GetSerializedShapeFunctions() + _xnnpack_shape_compute_functions;

    // 创建共享指针指向 Source 对象，用于存储 shape_compute_functions 的源码
    auto src = std::make_shared<Source>(shape_compute_functions);

    // 创建 stringstream 对象 ss 用于字符串操作
    std::stringstream ss;

    // 创建常量表 constantTable 用于存储常量
    std::vector<at::IValue> constantTable;

    // 创建 SourceImporterImpl 对象 resolver，用于导入源码和解析
    auto resolver = std::make_shared<SourceImporterImpl>(
        compilation_unit,
        &constantTable,
        // lambda 表达式用于根据名称返回共享指针指向 Source 对象
        [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
        1);

    // 在编译单元中定义 shape_compute_functions，使用 resolver 解析器
    compilation_unit->define(
        c10::nullopt, shape_compute_functions, resolver, nullptr);

    // 加载编译单元中的模块
    loadModule(*compilation_unit);

  } catch (...) {
    // 异常处理：重置缓存和编译单元，以避免在后续测试中出现奇怪的错误
    compilation_unit = std::make_shared<CompilationUnit>();
    cached_schema_to_graph.clear();
    throw;
  }
}

// 匿名命名空间结束

// 根据函数模式获取形状计算图
std::optional<std::shared_ptr<Graph>> shapeComputeGraphForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  
  // 如果 cached_schema_to_graph 为空，则加载函数
  if (cached_schema_to_graph.empty()) {
    loadFunctions();
  }

  // 调试信息：尝试查找特定模式的 schema
  GRAPH_DEBUG("Trying to find schema: ", schema);

  // 在 cached_schema_to_graph 中查找特定的 schema
  auto cache_it = cached_schema_to_graph.find(&schema);

  // 如果找到，则返回对应的图形
  if (cache_it != cached_schema_to_graph.end()) {
    return cache_it->second;
  }

  // 调试信息：未找到特定模式的 schema
  GRAPH_DEBUG("Could not find schema: ", schema);

  return c10::nullopt;
}

// 根据函数模式获取有界图形计算图
TORCH_API std::optional<BoundedShapeGraphs> boundedGraphsForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  
  // 如果 cached_bounded_schema_to_graph 为空，则加载函数
  if (cached_bounded_schema_to_graph.empty()) {
    loadFunctions();
  }

  // 调试信息：在有界图中尝试查找特定模式的 schema
  GRAPH_DEBUG("Trying to find schema in bounded graphs: ", schema);

  // 在 cached_bounded_schema_to_graph 中查找特定的 schema
  auto cache_it = cached_bounded_schema_to_graph.find(&schema);

  // 如果找到，则返回对应的有界图形
  if (cache_it != cached_bounded_schema_to_graph.end()) {
    return cache_it->second;
  }

  return c10::nullopt;
}

// 注册特定函数模式的形状计算图
void RegisterShapeComputeGraphForSchema(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g) {
  std::lock_guard<std::mutex> guard(lock);
  
  // 如果 cached_schema_to_graph 为空，则加载函数
  if (cached_schema_to_graph.empty()) {
    loadFunctions();
  }

  // 转换形状函数并进行形状计算图的检查
  transformShapeFunction(&schema, g);
  LintShapeComputeGraph(&schema, g);

  // 将 schema 和对应的图形 g 存入 cached_schema_to_graph
  cached_schema_to_graph[&schema] = g;
}

// 获取已注册的形状计算模式的函数模式列表
std::vector<const FunctionSchema*> RegisteredShapeComputeSchemas() {
  std::lock_guard<std::mutex> guard(lock);
  
  // 如果 cached_schema_to_graph 为空，则加载函数
  if (cached_schema_to_graph.empty()) {
    loadFunctions();
  }

  // 创建函数模式列表 schemas
  std::vector<const FunctionSchema*> schemas;
  schemas.reserve(cached_schema_to_graph.size());

  // 遍历 cached_schema_to_graph，将所有函数模式添加到 schemas 列表中
  for (const auto& pair : cached_schema_to_graph) {
    schemas.push_back(pair.first);
  }
  return schemas;
}

// 对形状计算图进行 lint 检查
void LintShapeComputeGraph(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  // 检查输入和输出类型
  checkInputAndOutputTypes(schema, graph);
  // 检查是否有 while 循环
  checkForWhileLoop(schema, graph);
  // 检查输入是否作为输出返回
  checkInputReturnedAsOutput(schema, graph);
  // TODO: 其他检查？例如不符号优化的列表操作等
}

} // namespace torch::jit
```