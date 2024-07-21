# `.\pytorch\torch\csrc\jit\passes\tensorexpr_fuser.cpp`

```
// 引入 Torch 中 TensorExpr 模块的相关头文件
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

// 引入 ATen 库的核心头文件
#include <ATen/core/interned_strings.h>
#include <ATen/core/symbol.h>
#include <ATen/record_function.h>
#include <c10/util/FunctionRef.h>
#include <c10/util/irange.h>

// 引入 Torch 中 CUDA 代码生成接口头文件
#include <torch/csrc/jit/codegen/cuda/interface.h>

// 引入 Torch 中 Fuser 接口头文件
#include <torch/csrc/jit/codegen/fuser/interface.h>

// 引入 Torch 中 IR 别名分析头文件
#include <torch/csrc/jit/ir/alias_analysis.h>

// 引入 Torch 中 JIT 日志头文件
#include <torch/csrc/jit/jit_log.h>

// 引入 Torch 中 JIT 优化限制头文件
#include <torch/csrc/jit/jit_opt_limit.h>

// 引入 Torch 中 JIT Pass 的常用子表达式消除头文件
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

// 引入 Torch 中 JIT Pass 的常量池化头文件
#include <torch/csrc/jit/passes/constant_pooling.h>

// 引入 Torch 中 JIT Pass 的死代码消除头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 引入 Torch 中 JIT Pass 管理器头文件
#include <torch/csrc/jit/passes/pass_manager.h>

// 引入 Torch 中 JIT Pass 的冗余分析移除头文件
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>

// 引入 Torch 中 JIT Pass 的符号形状运行时融合头文件
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>

// 引入 Torch 中 JIT Pass 工具函数的子图工具头文件
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 引入 Torch 中 Runtime 自定义运算符头文件
#include <torch/csrc/jit/runtime/custom_operator.h>

// 引入 Torch 中 Runtime 图执行器头文件
#include <torch/csrc/jit/runtime/graph_executor.h>

// 引入 Torch 中 Runtime 运算符选项头文件
#include <torch/csrc/jit/runtime/operator_options.h>

// 引入 Torch 中 Runtime 符号形状注册表头文件
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>

// 引入 Torch 中 Runtime 符号形状注册表工具头文件
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>

// 引入 Torch 中 TensorExpr 的核心内核头文件
#include <torch/csrc/jit/tensorexpr/kernel.h>

// 引入 C++ 标准库中的实用工具
#include <utility>

// NOLINTNEXTLINE，用于指示不检查下一行的 lint 错误
C10_DEFINE_bool(
    torch_jit_disable_cat,
    false,
    "disable aten::cat in TE fusion groups");

// NOLINTNEXTLINE，用于指示不检查下一行的 lint 错误
C10_DEFINE_bool(
    torch_jit_enable_dynamic_shape_fusion,
    false,
    "enable TE fusion using dynamic shapes");

// Torch JIT 命名空间
namespace torch {
namespace jit {

// 静态变量，表示 TensorExpr 中的缩减操作是否启用
static bool texpr_reductions_enabled = false;

// 判断节点是否在支持的块中
static bool isSupportedForBlock(Node* node) {
  switch (node->kind()) {
    // 支持的操作包括加法和乘法
    case aten::add:
    case aten::mul:
      return true;
    default:
      return false;
  }
}

// 判断值是否仅在 size 函数中使用
bool usedOnlyInSize(Value* v) {
  const auto& uses = v->uses();
  return std::all_of(uses.begin(), uses.end(), [](const Use& u) {
    // 判断使用者是否为 "aten::size(Tensor self) -> int[]"
    return u.user->matches("aten::size(Tensor self) -> int[]");
  });
}

// 对给定的大小数组进行广播，并返回结果值
Value* broadcastSizes(at::ArrayRef<Value*> sizes, AliasDb* db) {
  // 确保大小数组非空
  AT_ASSERT(!sizes.empty());
  // 获取大小数组所在的图
  Graph* graph = sizes[0]->owningGraph();
  // 插入广播大小节点，并传入大小数组
  Node* broadcast_n =
      graph->insertNode(graph->create(prim::BroadcastSizes, sizes));
  // 设置广播节点的输出类型为整数列表
  broadcast_n->output()->setType(ListType::ofInts());
  // 在别名数据库中创建节点的值
  db->createValue(broadcast_n->output());
  // 返回广播节点的输出值
  return broadcast_n->output();
}

// TensorExpr 命名空间
namespace tensorexpr {

// 返回全局自定义运算符集合对象的引用
OperatorSet& getCustomOperatorSet() {
  static OperatorSet _g_custom_operator_set{};
  return _g_custom_operator_set;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
static const OperatorSet& supported_non_eltwise_set() {
  // 定义静态常量集合，包含非元素级别操作的运算符集合
  // clang-format off
  static const OperatorSet supported_non_eltwise_set{
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
      "aten::matmul(Tensor self, Tensor other) -> Tensor",
  };
  // clang-format on
  return supported_non_eltwise_set;
};

bool isSupported(Node* node) {
  // 对于块代码生成，允许有限的操作
  if (tensorexpr::getTEGenerateBlockCode()) {
    return isSupportedForBlock(node);
  }

  // 定义静态常量集合，包含支持的归约操作的运算符集合
  static const OperatorSet supported_reduction_set{
      "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
      "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
      "aten::softmax.int(Tensor self, int dim , ScalarType? dtype=None) -> Tensor",
      "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
  };
  // clang-format on

  // 检查节点是否属于元素级别操作集、支持的非元素级别操作集、杂项操作集、自定义操作集或支持的归约操作集
  if (get_tensorexpr_elementwise_set().contains(node) ||
      node->isMemberOf(supported_non_eltwise_set()) ||
      node->isMemberOf(supported_misc_set) ||
      node->isMemberOf(getCustomOperatorSet()) ||
      (texpr_reductions_enabled && node->isMemberOf(supported_reduction_set))) {
    // 只有在节点的输入类型是 Tensor 类型时才插入保护逻辑，因为输出类型是由输入类型唯一确定的。
    // 如果任何非 Tensor 类型的输入影响输出类型且无法在静态情况下推理，则返回 false
    for (Value* v : node->inputs()) {
      if (v->type()->cast<NumberType>()) {
        return false;
      }
    }

    // 非常数的 dtype 或 device
    for (auto arg_name : {"dtype", "device"}) {
      if (auto index = node->schema().argumentIndexWithName(arg_name)) {
        if (!toIValue(node->input(*index))) {
          return false;
        }
      }
    }

    // 如果禁用了 torch_jit_disable_cat 并且节点的类型是 aten::cat，则返回 false
    if (FLAGS_torch_jit_disable_cat && node->kind() == aten::cat) {
      return false;
    }

    return true;
  }

  // 对于未编制模式的操作，如 ConstantChunk、ListConstruct 和 TensorExprGroup，直接返回 true
  switch (node->kind()) {
    case prim::ConstantChunk:
    case prim::ListConstruct:
    case prim::TensorExprGroup:
      return true;
  }

  // 其他情况下返回 false
  return false;
}
} // namespace tensorexpr

static bool texpr_fuser_enabled_ = true;
// 设置是否启用 Tensor Expression 融合器
void setTensorExprFuserEnabled(bool val) {
  texpr_fuser_enabled_ = val;
}

// 返回当前是否启用 Tensor Expression 融合器的状态
bool tensorExprFuserEnabled() {
  // 从环境变量 PYTORCH_TENSOREXPR 获取设置值
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR");
  if (!enable_c_str) {
    return texpr_fuser_enabled_;
  }
  // 如果环境变量值为 "0"，则返回 false
  if (std::string(enable_c_str) == "0") {
    return false;
  }
  // 否则返回 true
  return true;
}

// 返回是否启用 Tensor Expression 动态形状融合
bool tensorExprDynamicShapeFusionEnabled() {
  return FLAGS_torch_jit_enable_dynamic_shape_fusion;
}

// 设置是否启用 Tensor Expression 动态形状融合
void setTensorExprDynamicShapeFusionEnabled(bool val) {
  FLAGS_torch_jit_enable_dynamic_shape_fusion = val;
}

// 设置是否启用 Tensor Expression 减少操作
bool setTexprReductionsEnabled(bool value) {
  // 保存旧的 texpr_reductions_enabled 值
  bool old_value = texpr_reductions_enabled;
  // 更新 texpr_reductions_enabled 值为新值
  texpr_reductions_enabled = value;
  // 返回旧的 texpr_reductions_enabled 值
  return old_value;
}

// 返回当前是否启用 Tensor Expression 减少操作的状态
bool texprReductionsEnabled() {
  return texpr_reductions_enabled;
}

// 静态函数，用于从块中移除配置文件中的节点并特定类型的优化
static void removeProfileNodesAndSpecializeTypes(Block* b) {
  // 迭代遍历块中的每一个节点
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    // 如果当前迭代器指向的节点是 prim::profile 类型
    if (it->kind() == prim::profile) {
      // 打印调试信息，说明正在移除 prim::profile 节点，并指明输出值的调试名称
      GRAPH_DEBUG("Removing prim::profile: %", it->output()->debugName());
      // 将 prim::profile 节点的输出值替换为其输入值
      it->output()->replaceAllUsesWith(it->input());
      // 获取 profiled_type 属性的值，期望其为 TensorType 类型
      auto profiled_type = it->ty(attr::profiled_type)->expect<TensorType>();

      // 初始化输入值的张量类型指针和是否可选的标志
      TensorTypePtr input_tensor_type = nullptr;
      bool input_is_optional = false;

      // 检查输入值的类型是否为 TensorType
      if (it->input()->type()->kind() == c10::TypeKind::TensorType) {
        input_tensor_type = it->input()->type()->expect<TensorType>();
      } else {
        // 如果输入值的类型不是 TensorType，那么期望其为 OptionalType，并获取其元素类型为 TensorType
        input_tensor_type = it->input()
                                ->type()
                                ->expectRef<OptionalType>()
                                .getElementType()
                                ->expect<TensorType>();
        input_is_optional = true;
      }

      // 如果输入值是可选的，则销毁当前迭代器指向的节点并继续下一次循环
      if (input_is_optional) {
        it.destroyCurrent();
        continue;
      }

      // 处理值可能会被不同类型使用的情况
      // 这可能发生在：
      // - 有一个未执行的使用，因此类型将是 TensorType::get()
      // - 依赖张量类型的控制流：
      //   if x.size() == 2 op(x) else op(x)
      // - 在张量类型中表示字段的值的变异
      //   op(x); x.resize_([...]); op(x)

      // 当 num_profiles = 1 时，今天最常见的情况来自第一种情况。
      // 在这种情况下，我们可以忽略非 profiled 使用，并选择任何一个 profiled 使用。
      // 因为我们在运行时保护所有张量类型，即使我们设置一个值具有来自一个使用的 profiled 类型，
      // 然后执行具有不同 profiled 类型的使用，我们仍然是正确的。
      // 将来我们可以考虑统一使用的类型，或者添加一个类型细化节点，以便使用可以具有正确对应的类型。
      if (profiled_type == TensorType::get()) {
        continue;
      }

      // 如果遇到相同值的不同 profiled 类型，则将它们合并。
      // 如果在一个循环体中展开循环导致 profiled 类型的重复，而这不是逻辑上一致的情况下，会发生这种情况
      // （参见 TestTEFuser.test_unrolled_cat）。
      if (input_tensor_type == TensorType::get()) {
        it->input()->setType(profiled_type);
      } else {
        it->input()->setType(input_tensor_type->merge(*profiled_type));
      }

      // 销毁当前迭代器指向的节点
      it.destroyCurrent();
    } else {
      // 如果当前迭代器指向的节点不是 prim::profile 类型，则遍历其所有的子块
      for (Block* ib : it->blocks()) {
        // 递归调用 removeProfileNodesAndSpecializeTypes 函数处理子块
        removeProfileNodesAndSpecializeTypes(ib);
      }
    }
// 从图中移除所有的 ProfileNode，并专门化类型
void RemoveProfileNodesAndSpecializeTypes(std::shared_ptr<Graph>& graph) {
  // 输出调试信息，显示移除 ProfileNode 前的图结构
  GRAPH_DEBUG("Before removeProfileNodesAndSpecializeTypes:\n", *graph);
  // 调用函数，移除图中的 ProfileNode 并专门化类型
  removeProfileNodesAndSpecializeTypes(graph->block());
  // 输出调试信息，显示移除 ProfileNode 后的图结构
  GRAPH_DEBUG("After removeProfileNodesAndSpecializeTypes:\n", *graph);
}

// 检查值 v 是否具有张量类型的专门化
bool hasTensorTypeSpecialization(Value* v) {
  // 如果值 v 不是张量类型，则返回 false
  if (!v->type()->cast<TensorType>()) {
    return false;
  }
  // 常量和 TensorExprGroup 总是产生专门化的张量类型，
  // TypeCheck 是由该 pass 插入的，仅由插入适当守卫的融合组使用
  if (v->node()->kind() == prim::Constant ||
      v->node()->kind() == prim::TypeCheck ||
      v->node()->kind() == prim::TensorExprGroup) {
    return false;
  }
  // 如果类型是普通的 TensorType，则返回 false
  if (v->type() == TensorType::get()) {
    return false;
  }
  // 其他情况返回 true，表示有张量类型的专门化
  return true;
}

// 移除值 v 的张量类型的专门化
static void removeTensorTypeSpecialization(Value* v) {
  // 如果值 v 具有张量类型的专门化，则将其类型设置为普通的 TensorType
  if (hasTensorTypeSpecialization(v)) {
    v->setType(TensorType::get());
  }
}

// 移除块中所有值的张量类型的专门化
void removeTensorTypeSpecializations(Block* block) {
  // 处理块的输入值
  for (Value* v : block->inputs()) {
    removeTensorTypeSpecialization(v);
  }
  // 处理块中的每个节点
  for (Node* n : block->nodes()) {
    // 递归处理每个节点的子块
    for (Block* b : n->blocks()) {
      removeTensorTypeSpecializations(b);
    }
    // 处理节点的输出值
    for (Value* v : n->outputs()) {
      removeTensorTypeSpecialization(v);
    }
  }
}

// 从图中移除所有值的张量类型的专门化
void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph) {
  removeTensorTypeSpecializations(graph->block());
}

// 为 guarded_node 插入类型检查守卫
void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    Symbol kind) {
  // 输出调试信息，显示插入类型检查守卫前的节点信息
  GRAPH_DEBUG("Inserting a typecheck guard for a node", *guarded_node);
  // 获取 guarded_node 的子图
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // 修复子图输入的类型
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  for (Value* input : guarded_node->inputs()) {
    // 只检查受保护节点的输入，并期望用户推断中间和输出形状
    if (!input->type()->cast<TensorType>()) {
      continue;
    }
    // 融合输出已经有守卫
    if (input->node()->kind() == prim::Constant ||
        input->node()->kind() == prim::FusionGroup) {
      continue;
    }
    // 将需要检查的输入添加到列表中
    inputs_to_check.push_back(input);
    // 使用 type_converter 转换输入的张量类型并添加到守卫类型列表中
    guard_types.emplace_back(
        type_converter(input->type()->expect<TensorType>()));
  }
  // 如果没有需要检查的输入，则不插入守卫
  if (inputs_to_check.empty()) {
    return;
  }

  // 添加 prim::TypeCheck 节点
  //
  // TypeCheck 节点的形式如下：
  //   %out1 : Float(2, 3), %out2 : Int(10, 30), %types_match : bool =
  //   prim::TypeCheck(%inp1 : Tensor, %inp2 : Tensor)
  //
  // 它们有 N 个输入，表示要检查的类型，以及 N+1 个输出。前 N 个输出指定了期望的类型，
  // 第 N+1 个输出是检查的结果（布尔值）。
  Node* typecheck_node =
      guarded_node->owningGraph()
          ->create(kind, inputs_to_check, inputs_to_check.size() + 1)
          ->insertBefore(guarded_node);
  // 设置 TypeCheck 节点的属性 types，用于指定类型
  typecheck_node->tys_(attr::types, std::move(guard_types));
  // 获取 TypeCheck 结果的值
  Value* typecheck_result = typecheck_node->output(inputs_to_check.size());

  // 创建一个映射，将 TypeCheck 节点的输入与输出进行关联
  std::unordered_map<Value*, Value*> typechecked_inputs;
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typechecked_inputs[typecheck_node->input(i)] = typecheck_node->output(i);
  }

  // 修正 TypeCheck 节点输出的类型，这些类型将在执行中使用
  typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typecheck_node->output(i)->setType(typecheck_node->input(i)->type());
  }

  // 插入 If 节点
  auto versioning_if =
      guarded_node->owningGraph()
          ->create(prim::If, {typecheck_result}, guarded_node->outputs().size())
          ->insertAfter(typecheck_node);
  // 设置 If 节点的输出类型与原节点输出类型相同，并替换原节点的使用
  for (size_t idx = 0; idx < guarded_node->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(guarded_node->output(idx)->type());
    guarded_node->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // 填充 false 分支，包含未优化的融合子图的复制
  WithInsertPoint guard(false_block->return_node());
  // 在 guarded_node 的 Graph 中插入子图，并获取子图的输出
  const auto subgraph_outputs = insertGraph(
      *guarded_node->owningGraph(), *subgraph, guarded_node->inputs());
  // 将子图的输出注册到 false 分支中
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // 将类型复制到 fallback 图中，因此在替换前需要删除特殊化
  removeTensorTypeSpecializations(false_block);
  // 使用 fallback 图替换 false 分支的块
  replaceBlockWithFallbackGraph(false_block, guarded_node->inputs());

  // 填充 true 分支，其中所有输入已经经过类型检查，其主体应为融合组节点
  // 将 guarded_node 移动到 true 分支的返回节点之前
  guarded_node->moveBefore(true_block->return_node());
  for (size_t idx = 0; idx < guarded_node->inputs().size(); ++idx) {
    // 如果 typechecked_inputs 中存在当前输入，则用其替换
    if (typechecked_inputs.count(guarded_node->input(idx))) {
      guarded_node->replaceInput(
          idx, typechecked_inputs.at(guarded_node->input(idx)));
    }
  }
  // 将 guarded_node 的输出注册到 true 分支中
  for (Value* output : guarded_node->outputs()) {
    true_block->registerOutput(output);
  }
} // namespace

// 检查节点是否具有不支持的 pin_memory 特性
bool has_unsupported_pin_memory(const Node* node) {
  // 如果节点的模式中存在 "pin_memory" 参数
  if (auto maybe_index = node->schema().argumentIndexWithName("pin_memory")) {
    // 获取参数在模式中的索引
    int index = *maybe_index;
    // 获取参数对应的输入节点
    auto inp = node->input(index);
    // 如果输入节点类型不是 NoneType 并且其常量值为真
    if (inp->type() != NoneType::get() &&
        constant_as<bool>(inp).value_or(true)) {
      // 返回 true，表示存在不支持的 pin_memory 特性
      return true;
    }
  }
  // 否则返回 false，表示不存在不支持的 pin_memory 特性
  return false;
}

class TensorExprFuser {
 public:
  // 构造函数，初始化 TensorExprFuser 对象
  TensorExprFuser(
      std::shared_ptr<Graph> graph,
      size_t min_group_size,
      bool add_composed_op,
      bool fuse_to_dynamic_shapes)
      : graph_(std::move(graph)),
        min_group_size_(min_group_size),
        add_composed_op_(add_composed_op),
        fuse_to_dynamic_shapes_(fuse_to_dynamic_shapes) {
    // 解析不融合选项
    parseTENotFuseOption();
  }

  // 构建融合组的表达式，计算所有中间值（和输出）的形状，基于输入的大小
  std::unordered_map<Value*, Value*> buildShapeExpressions(Node* fusion_group) {
    GRAPH_DUMP("buildShapeExpressions for ", fusion_group->g(attr::Subgraph));
    // 设置插入点到融合组的下一个节点
    WithInsertPoint insert_guard{fusion_group->next()};
    // 形状映射表，将输入/输出与其形状值对应起来
    std::unordered_map<Value*, Value*> shape_of;

    Graph* graph = fusion_group->owningGraph();
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto inputs = fusion_group->inputs();
    auto sinputs = subgraph->inputs();
    AT_ASSERT(inputs.size() == sinputs.size());
    // 遍历所有输入
    for (const auto i : c10::irange(inputs.size())) {
      // 如果输入是张量类型
      if (inputs[i]->type()->isSubtypeOf(*TensorType::get())) {
        // 插入一个 size 操作节点到计算图中
        Value* soutput = graph->insert(aten::size, {inputs[i]});
        aliasDb_->createValue(soutput);
        GRAPH_DEBUG(
            "Adding a mapping for %",
            sinputs[i]->debugName(),
            " ",
            getHeader(soutput->node()));
        // 将输入与对应的形状值映射存储到 shape_of 中
        shape_of[sinputs[i]] = soutput;
      }
    }

    // 当输出不会被移除时，可以使用其大小来代替长链的广播计算，从内核的开始处开始
    auto outputs = fusion_group->outputs();
    auto soutputs = subgraph->outputs();
    AT_ASSERT(outputs.size() == soutputs.size());
    // 遍历所有输出
    for (const auto i : c10::irange(outputs.size())) {
      // 如果输出只用于大小检查中，则跳过
      if (usedOnlyInSize(outputs[i]))
        continue;
      // 插入一个 size 操作节点到计算图中
      Value* soutput = graph->insert(aten::size, {outputs[i]});
      aliasDb_->createValue(soutput);
      // 将输出与对应的形状值映射存储到 shape_of 中
      shape_of[soutputs[i]] = soutput;
    }
    for (Node* n : subgraph->nodes()) {
      // 遍历融合子图中的每个节点
      auto tensor_inputs = filter(n->inputs(), [](Value* v) {
        // 筛选出输入是张量类型的值
        return v->type()->isSubtypeOf(*TensorType::get());
      });
      // 打印调试信息，显示当前节点正在构建尺寸信息
      GRAPH_DEBUG("Building sizes for ", getHeader(n));
      // 检查所有的张量输入是否都有尺寸信息
      bool all_inputs_have_sizes = true;
      auto shapes = fmap(tensor_inputs, [&](Value* v) {
        // 打印调试信息，显示正在获取张量的大小信息
        GRAPH_DEBUG("Getting aten::size for %", v->debugName());
        // 更新是否所有张量输入都有尺寸信息的状态
        all_inputs_have_sizes &= shape_of.count(v);
        // 返回张量的尺寸信息，如果没有则返回空指针
        return shape_of.count(v) != 0 ? shape_of.at(v) : nullptr;
      });
      // 如果有输入没有尺寸信息，则跳过当前节点的处理
      if (!all_inputs_have_sizes) {
        // 打印调试信息，说明无法计算广播尺寸因为不是所有的张量参数都有可用的尺寸信息
        GRAPH_DEBUG(
            "Not all tensor arguments have sizes available to compute the broadcasted size",
            getHeader(n));
        continue;
      }

      // 如果节点是 prim::ConstantChunk 类型
      if (n->kind() == prim::ConstantChunk) {
        // 创建一个新节点用于处理尺寸信息
        Node* sizes_node = graph->insertNode(
            graph->create(prim::ChunkSizes, shape_of.at(n->input()), 2));
        // 设置节点的属性
        sizes_node->i_(attr::dim, n->i(attr::dim));
        sizes_node->i_(attr::chunks, n->i(attr::chunks));
        // 为新节点的输出创建别名
        for (Value* output : sizes_node->outputs()) {
          aliasDb_->createValue(output);
        }
        // 获取新节点的两个输出值
        Value* regular_size = sizes_node->outputs().at(0);
        Value* last_size = sizes_node->outputs().at(1);
        // 设置输出值的类型为整数列表
        regular_size->setType(ListType::ofInts());
        last_size->setType(ListType::ofInts());
        // 获取节点的所有输出，除了最后一个输出
        auto outputs = n->outputs();
        for (Value* o : outputs.slice(0, outputs.size() - 1)) {
          // 将正常尺寸信息与节点输出值关联起来
          shape_of.emplace(o, regular_size);
        }
        // 将最后一个输出与最后一个尺寸信息关联起来
        shape_of.emplace(outputs.at(outputs.size() - 1), last_size);
        continue;
      }

      // 只支持对逐元素操作的形状计算，还有一些特定的非逐元素操作
      if (!(get_tensorexpr_elementwise_set().contains(n)) &&
          !n->isMemberOf(tensorexpr::supported_non_eltwise_set())) {
        // 如果不是逐元素操作也不是支持的非逐元素操作，则跳过当前节点的处理
        continue;
      }

      // 将当前节点的输出与计算后的形状关联起来
      shape_of.emplace(
          n->output(),
          shapes.size() == 1 ? shapes[0]
                             : broadcastSizes(shapes, aliasDb_.get()));
    }
    // 返回构建好的形状表达式
    return shape_of;
  }

  void removeOutputsUsedOnlyInSize(Node* fusion_group) {
    // 如果融合组的类型不是 prim::TensorExprGroup，则直接返回
    if (fusion_group->kind() != prim::TensorExprGroup)
      return;
    // 获取融合组的子图
    auto subgraph = fusion_group->g(attr::Subgraph);

    // 构建融合组的形状表达式
    auto shape_of = buildShapeExpressions(fusion_group);
    // 获取融合组的所有输出
    auto outputs = fusion_group->outputs().vec();
    // 获取子图的所有输出
    auto soutputs = subgraph->outputs().vec();
    // 打印警告信息，说明按照这个顺序迭代不仅仅是为了性能原因，也是为了正确性
    // i 必须反映当前输出索引的真实情况
    GRAPH_DEBUG(
        "XXX: Iterating in this order is not only good for performance reasons! "
        "It is also crucial for correctness (i has to reflect the current true "
        "index of outputs[i])!");
    // 从最后一个输出开始逆序遍历所有输出
    for (int64_t i = static_cast<int64_t>(outputs.size()) - 1; i >= 0; --i) {
      // 获取当前输出节点和其对应的字符串形式的输出
      auto output = outputs[i];
      auto soutput = soutputs[i];
      // 检查当前输出是否仅在大小计算中使用，并且其形状在 shape_of 映射中存在
      if (usedOnlyInSize(output) && shape_of.count(soutput) > 0) {
        // 获取当前输出节点的使用情况
        auto uses = output->uses();
        // 遍历当前输出节点的使用情况
        for (Use u : uses) {
          // 断言使用情况的用户节点匹配 "aten::size(Tensor self) -> int[]"
          AT_ASSERT(u.user->matches("aten::size(Tensor self) -> int[]"));
          // 将使用节点的输出替换为 shape_of 中对应的形状数据
          u.user->output()->replaceAllUsesWith(shape_of.at(soutput));
          // 销毁使用节点
          u.user->destroy();
        }
        // 从融合组中删除当前输出节点
        fusion_group->eraseOutput(i);
        // 从子图中删除当前输出节点
        subgraph->eraseOutput(i);
      }
    }
  }

  void run() {
    // 创建别名数据库并关联到图形
    aliasDb_ = std::make_unique<AliasDb>(graph_);
    // 移除冗余的性能配置节点
    RemoveRedundantProfiles(graph_);
    // 输出优化后的图形，移除冗余性能配置节点后
    GRAPH_DUMP("After removing redundant profile nodes: ", graph_);
    // 在图形的块中创建融合组
    createFusionGroups(graph_->block());
    // 输出创建融合组后的图形
    GRAPH_DUMP("After creating fusion groups: ", graph_);
    // 在初始融合期间维护别名数据库的正确性，但在内联后维护正确性比较困难，因此仅在融合完成后内联
    inlineSmallFusionGroups(graph_->block());
    // 输出内联小融合组后的图形
    GRAPH_DUMP("After inlining small fusion groups: ", graph_);
    // 如果启用了动态形状的 TensorExpr 融合
    if (fuse_to_dynamic_shapes_) {
      VLOG(1) << "TensorExpr fusion with dynamic shapes is enabled"
              << std::endl;
      // 泛化融合组
      generalizeFusionGroups(graph_->block());
      // 输出泛化融合组后的图形
      GRAPH_DUMP("After generalizing fusion groups: ", graph_);
    } else {
      // 准备融合组并保护其输出
      prepareFusionGroupAndGuardOutputs(graph_->block());
      // 输出保护融合组后的图形
      GRAPH_DUMP("After guarding fusion groups: ", graph_);
    }
  }

 private:
  // 获取或创建 TensorExpr 子图节点
  Node* getOrCreateTensorExprSubgraph(Node* n) {
    if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::TensorExprGroup) {
      return n;
    }
    // 更新图形，创建一个 tensorexpr::Group 节点
    GRAPH_UPDATE("Creating a tensorexpr::Group node from: ", *n);
    return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
        n, prim::TensorExprGroup, *aliasDb_);
  }

  // 对输入进行逆拓扑排序
  value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* b) {
    value_list result;
    // 遍历输入，筛选属于同一块的节点
    for (auto i : inputs) {
      if (i->node()->owningBlock() == b) {
        result.push_back(i);
      }
    }
    // 按照逆拓扑顺序排序
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  // 创建从节点 N 开始的融合组
  std::pair<graph_node_list::iterator, bool> createFusionGroup(
      Node* fusion_node) {
    // 允许包含 conv2d 的单节点组，因为我们只会选择在 tensorexpr 实现更快的情况下使用它们
    if (min_group_size_ == 1 || fusion_node->kind() == aten::conv2d) {
      fusion_node = getOrCreateTensorExprSubgraph(fusion_node);
    }

    // 迭代地将输入节点合并到融合组中
    GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
    auto inputs = sortReverseTopological(
        fusion_node->inputs(), fusion_node->owningBlock());
    for (auto input : inputs) {
      // 输出当前融合组的调试信息和融合节点的调试信息
      debugDumpFusionGroup("Current fusion group: ", fusion_node);
      // 输出尝试合并的节点信息
      GRAPH_DEBUG("Trying to merge: ", *input->node());
      // 尝试将当前融合节点（fusion_node）与输入节点（input->node()）合并
      if (auto maybe_fusion_group = tryMerge(fusion_node, input->node())) {
        // 如果成功合并，则新的融合组的 `inputs` 可能已更改，因此重新扫描以寻找更多合并机会
        return std::make_pair(
            maybe_fusion_group.value()->reverseIterator(), true);
      }
    }

    // 返回新的迭代器位置和指示是否成功的布尔值
    return std::make_pair(++fusion_node->reverseIterator(), false);
  }

  static void debugDumpFusionGroup(const std::string& msg, Node* n) {
    // 输出调试信息，包括消息和节点内容
    GRAPH_DEBUG(msg, *n);
    // 如果节点的类型是 prim::TensorExprGroup，则输出其子图的调试信息
    if (n->kind() == prim::TensorExprGroup) {
      GRAPH_DEBUG(*n->g(attr::Subgraph));
    }
  }

  // 在 eager 模式下，不能有未执行的操作成为 Fusion Groups 的输出，因为这会降低性能并改变别名关系
  static bool unexecutedEagerOp(Node* n) {
    // 如果节点的类型不是 aten::to、aten::_autocast_to_reduced_precision 或 aten::_autocast_to_full_precision，则返回 false
    if (n->kind() != aten::to &&
        n->kind() != aten::_autocast_to_reduced_precision &&
        n->kind() != aten::_autocast_to_full_precision) {
      return false;
    }

    // 检查节点的输入和输出张量类型是否相同
    return *n->input(0)->type()->expect<TensorType>() ==
        *n->output()->type()->expect<TensorType>();
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* n) {
    // 输出考虑的节点信息
    GRAPH_DEBUG("Considering node:", *n)

    // 如果节点无法处理，则返回下一个节点的迭代器和 false
    if (!canHandle(n)) {
      return std::make_pair(++n->reverseIterator(), false);
    }
    // 有些节点我们可以支持，但不希望从其开始一个 Fusion Group - 跳过它们
    if (n->kind() == prim::ListConstruct || n->kind() == aten::slice ||
        n->kind() == aten::unsqueeze || n->kind() == prim::ConstantChunk ||
        n->kind() == prim::Constant || unexecutedEagerOp(n)) {
      // 返回下一个节点的迭代器和 false
      return std::make_pair(++n->reverseIterator(), false);
    }
    // 创建 Fusion Group 并返回新的迭代器位置和 true
    return createFusionGroup(n);
  }

  // 将可融合节点合并到 prim::TensorExprGroup 节点的子图中
  void createFusionGroups(Block* block) {
    // 初始化任何更改标志为 true
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        bool changed;
        // 扫描节点并尝试创建 Fusion Group
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    // 递归处理每个节点的子块
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        createFusionGroups(b);
      }
    }

    // 尝试合并相邻的 Fusion Group。因为我们仅通过查看图输入来合并，没有这一步，我们将不会尝试合并不依赖彼此的相邻 Fusion Group
    std::vector<Node*> initial_fusion_groups;
    // 收集所有 prim::TensorExprGroup 类型的节点
    for (Node* n : block->nodes()) {
      if (n->kind() == prim::TensorExprGroup) {
        initial_fusion_groups.push_back(n);
      }
    }
    // 初始化前一个融合组为初始融合组的第一个元素，如果初始融合组非空，否则为 nullptr。
    Node* prev_fusion_group =
        !initial_fusion_groups.empty() ? initial_fusion_groups[0] : nullptr;

    // 遍历除第一个之外的所有初始融合组
    for (const auto i : c10::irange(1, initial_fusion_groups.size())) {
      // 尝试将刚创建的融合组合并到前一个融合组中。
      // 如果合并失败，将前一个融合组放入 fusion_groups 向量中，并在本次循环中不再处理它。
      // 如果合并成功，将合并后的组保存为“前一个”融合组，以便尝试将下一个融合组合并到它中。
      Node* fusion_group = initial_fusion_groups[i];
      debugDumpFusionGroup(
          "Trying to merge into the previous fusion group: ",
          prev_fusion_group);
      if (auto merged_fusion_group =
              tryMerge(prev_fusion_group, fusion_group)) {
        prev_fusion_group = *merged_fusion_group;
        debugDumpFusionGroup(
            "Successfully merged into the previous fusion group: ",
            prev_fusion_group);
      } else {
        GRAPH_DEBUG("Cannot merge into the previous fusion group");
        prev_fusion_group = fusion_group;
      }
    }
  }

  // 计算给定块的节点数，不包括 prim::Constants 和 prim::ListConstructs 节点
  size_t blockSize(Block* block) {
    size_t num = 0;
    for (Node* n : block->nodes()) {
      if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
        continue;
      }
      // 递归计算每个节点及其子块的节点数
      for (Block* b : n->blocks()) {
        num += blockSize(b);
      }
      num++;
    }
    return num;
  }

  // 检查给定块中是否包含 conv2d 节点
  bool hasConv(Block* block) {
    for (Node* n : block->nodes()) {
      if (n->kind() == aten::conv2d) {
        return true;
      }
    }
    return false;
  }

  // 如果融合组过小，将其拆解
  bool inlineIfTooSmall(Node* n) {
    if (n->kind() != prim::TensorExprGroup) {
      return false;
    }
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t num_nodes = blockSize(subgraph->block());
    // 如果融合组节点数小于最小组大小且不包含 conv2d 节点，则拆解该融合组
    if (num_nodes < min_group_size_ && !hasConv(subgraph->block())) {
      GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    // 清理子图中的重复常量
    ConstantPooling(subgraph);

    // 如果启用了图形调试，导出子图信息
    if (GRAPH_DEBUG_ENABLED) {
      GRAPH_EXPORT("", subgraph);
    }
    return false;
  }

  // 递归地处理块中的所有节点，如果节点过小则拆解融合组
  void inlineSmallFusionGroups(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++;

      // 递归处理节点的子块
      for (Block* b : n->blocks()) {
        inlineSmallFusionGroups(b);
      }
      // 检查并拆解过小的融合组
      inlineIfTooSmall(n);
  }
}

std::optional<Node*> tryMerge(Node* fusion_group, Node* to_merge) {
  // 如果无法合并 fusion_group 和 to_merge，则返回空的 std::optional
  if (!canMerge(fusion_group, to_merge)) {
    return c10::nullopt;
  }

  std::vector<Node*> nodes_to_merge = {to_merge};

  // 如果 to_merge 是 aten::cat 操作，则获取其输入节点（即 listconstruct），并加入 nodes_to_merge
  if (to_merge->kind() == aten::cat) {
    Node* listconstruct = to_merge->input(0)->node();
    nodes_to_merge.push_back(listconstruct);
  }

  // 首先尝试将所有待合并的节点移动到 fusion_group 旁边
  Node* move_point = fusion_group;
  for (auto n : nodes_to_merge) {
    GRAPH_UPDATE("Trying to move node next to fusion group: ", getHeader(n));
    // 如果移动失败（由于别名数据库的检查），则返回空的 std::optional
    if (!aliasDb_->moveBeforeTopologicallyValid(n, move_point)) {
      GRAPH_UPDATE("Failed to move because of AliasDB checks!");
      return c10::nullopt;
    }
    move_point = n;
  }

  // 现在所有待合并的节点已经移动到 fusion_group 旁边，可以安全地将它们合并到 fusion_group 子图中
  fusion_group = getOrCreateTensorExprSubgraph(fusion_group);

  // 将每个待合并的节点合并到 fusion_group 中，并更新别名信息
  for (auto n : nodes_to_merge) {
    GRAPH_UPDATE("Merging ", getHeader(n));
    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        n, fusion_group, *aliasDb_);
  }
  return fusion_group;
}

bool shapeIsKnown(Value* v) {
  if (v->type()->cast<TensorType>()) {
    if (!v->isCompleteTensor()) {
      return false;
    }
  }
  return true;
}

bool allShapesAreKnown(Node* node) {
  // 遍历节点的所有输入，检查它们的形状是否已知
  // TODO: 放宽检查以支持动态形状
  for (Value* input : node->inputs()) {
    if (!shapeIsKnown(input)) {
      return false;
    }
    // 如果输入是列表构造（prim::ListConstruct），则递归检查其所有元素的形状
    if (input->node()->kind() == prim::ListConstruct) {
      if (!allShapesAreKnown(input->node())) {
        return false;
      }
    }
  }
  // 检查节点的所有输出，确保它们的形状都已知
  for (Value* output : node->outputs()) {
    if (!shapeIsKnown(output)) {
      return false;
    }
  }
  return true;
}

bool canFuseOnDevice(Value* v) {
  auto type = v->type()->cast<TensorType>();
  if (!type) {
    return true;
  }
  auto device = type->device();
  if (!device) {
    return false;
  }
  // 根据设备类型判断是否可以在特定设备上融合
  if (device->is_cpu()) {
    return canFuseOnCPU();
  } else if (device->is_cuda()) {
    return canFuseOnGPU();
  } else if (device->is_xpu()) {
    return false;
  }
  return false;
}

bool isFusableOnDevice(Node* node) {
  // 检查节点的所有输入，确保它们在设备上可以融合
  for (const auto& input : node->inputs()) {
    // 如果输入是列表构造，递归检查其所有元素是否可以在设备上融合
    if (input->node()->kind() == prim::ListConstruct) {
      if (!isFusableOnDevice(input->node())) {
        return false;
      }
    }
    if (!canFuseOnDevice(input)) {
      return false;
    }
  }
  return true;
}

bool typesAreSupported(Node* node) {
  // clang-format off
  // 禁用 clang 格式化，使得模式字符串不容易通过查找定位到
    // 定义仅包含浮点操作的操作符集合
    static const OperatorSet float_only_operator_set{
      "aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor",
    };
    // 定义仅包含整数操作的操作符集合
    static const OperatorSet int_only_operator_set{
      "aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor",
    };
    // 定义仅在 CPU 上进行计算密集型操作的操作符集合
    static const OperatorSet cpu_compute_heavy_set{
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
      "aten::matmul(Tensor self, Tensor other) -> Tensor",
    };
    // 定义仅包含 GPU 上运行的操作符集合
    static const OperatorSet gpu_only_operator_set{
      // 在 CPU 上，这些操作比 ATen 内核更慢且不够精确，因为 ATen 能使用 MKL-VML，而融合器目前无法做到。
      // 融合器使用 sleef，因为 sleef 提供了能操作向量而不是大缓冲区的函数。
      "aten::erf(Tensor self) -> Tensor",
      "aten::erfc(Tensor self) -> Tensor",
    };
    // 定义幂运算操作符集合
    static const OperatorSet pow{
      "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
    };
    // clang-format on

    // 检查输入值的类型。
    for (const Value* v : node->inputs()) {
      // 如果值是 Tensor 类型
      if (auto const& tt = v->type()->cast<TensorType>()) {
        auto const& st = tt->scalarType(); // 获取标量类型
        auto const& device = tt->device(); // 获取设备信息

        // 所有的张量必须是有类型的。
        if (!st || !device) { // 如果标量类型或设备信息为空
          return false; // 返回 false
        }

        // 字节张量引入了类型提升中的太多特例情况。
        // 最好不要尝试处理它们。
        if (*st == c10::ScalarType::Byte) { // 如果是 Byte 类型
          return false; // 返回 false
        }

        // Float16 支持存在一些问题，因此暂时禁用。
        // 在 CPU 上，额外禁用它，直到我们移动到更稳定的版本或找到解决方法为止。
        if (*st == c10::ScalarType::Half && *device == c10::kCPU) { // 如果是 Half 类型且在 CPU 上
          return false; // 返回 false
        }

        // BFloat16 类型在 CPU 上也有一些问题，因此禁用它。
        if (*st == c10::ScalarType::BFloat16 && *device == c10::kCPU) {

          return false; // 返回 false
        }
      }
    }
#ifndef TORCH_ENABLE_LLVM
        // 如果未启用 LLVM，直接返回 false
        return false;
#endif
      }

      // 这些运算符仅支持浮点数，因为整数除法需要引发 ZeroDivisionError 异常。
      if (node->isMemberOf(float_only_operator_set) && !isFloatingType(*st)) {
        // 如果节点属于仅支持浮点数的运算符集合，但类型不是浮点数，则返回 false
        return false;
      }

      // 这些运算符对浮点数有复杂的类型转换规则。
      if (node->isMemberOf(int_only_operator_set) && isFloatingType(*st)) {
        // 如果节点属于仅支持整数的运算符集合，但类型是浮点数，则返回 false
        return false;
      }
    } else if (node->isMemberOf(float_only_operator_set)) {
      // 检查仅支持浮点数操作的标量操作数。
      if (!v->type()->cast<FloatType>()) {
        // 如果操作数类型不能转换为浮点数类型，则返回 false
        return false;
      }
    } else if (node->isMemberOf(int_only_operator_set)) {
      // 检查仅支持整数操作的操作符。
      if (!v->type()->cast<IntType>()) {
        // 如果操作数类型不能转换为整数类型，则返回 false
        return false;
      }
    }

    // aten::pow 操作具有特殊规则，以避免复杂的整数情况。
    // 预期第一个参数是浮点数张量，如果不是，则返回 false。
    if (node->isMemberOf(pow)) {
      auto const& tt = node->input(0)->type()->cast<TensorType>();
      if (!tt) {
        // 如果第一个输入不是张量类型，则返回 false
        return false;
      }
      auto const& st = tt->scalarType();
      if (!st || !isFloatingType(*st)) {
        // 如果标量类型不存在或者不是浮点数类型，则返回 false
        return false;
      }
    }

    // 运算符仅在 CPU 上支持。
    if (node->isMemberOf(cpu_compute_heavy_set)) {
      if (fuse_to_dynamic_shapes_) {
        // 如果启用了动态形状融合，则返回 false
        return false;
      }

      auto device = tensorexpr::pickDeviceType(node->inputs());
      if (!device) {
        device = tensorexpr::pickDeviceType(node->outputs());
      }
      if (!device || !device->is_cpu()) {
        // 如果设备类型不存在或者不是 CPU，则返回 false
        return false;
      }
    }

    // 运算符仅在 GPU 上支持。
    if (node->isMemberOf(gpu_only_operator_set)) {
      auto device = tensorexpr::pickDeviceType(node->inputs());
      if (!device) {
        device = tensorexpr::pickDeviceType(node->outputs());
      }
      if (!device || !device->is_cuda()) {
        // 如果设备类型不存在或者不是 CUDA，则返回 false
        return false;
      }
    }

    if (node->kind() == aten::to) {
      // 仅支持同一设备的转换
      auto device = tensorexpr::pickDeviceType(node->inputs());
      auto output_device = tensorexpr::pickDeviceType(node->outputs());
      if (!device || !output_device || *device != *output_device) {
        // 如果输入或输出设备类型不一致，则返回 false
        return false;
      }
      // 非阻塞仅适用于跨设备转换，我们在这里不处理复制参数，也不从内存格式开始融合组。
      
      // 所有非张量参数必须是常量
      for (size_t i = 1; i < node->inputs().size(); i++) {
        if (node->inputs().at(i)->node()->kind() != prim::Constant) {
          // 如果非张量参数不是常量，则返回 false
          return false;
        }
      }

      if (has_unsupported_pin_memory(node)) {
        // 如果有不支持的 pin_memory 操作，则返回 false
        return false;
      }
    }
    // 检查节点是否为自动类型转换至降低精度或完整精度
    if (node->kind() == aten::_autocast_to_reduced_precision ||
        node->kind() == aten::_autocast_to_full_precision) {
      
      // 遍历节点的输入，从第二个输入开始检查是否为常量，如果不是则返回false
      for (auto i : c10::irange(1, node->inputs().size())) {
        if (node->inputs().at(i)->node()->kind() != prim::Constant) {
          return false;
        }
      }

      // 根据节点的种类确定是否为降低精度或完整精度
      bool is_reduced_precision =
          node->kind() == aten::_autocast_to_reduced_precision;
      bool is_full_precision =
          node->kind() == aten::_autocast_to_full_precision;
      auto self_tensor = node->inputs()[0]; // 输入张量

      // 检查输入张量的类型是否为TensorType，并获取其标量类型和设备信息
      if (auto const& tt = self_tensor->type()->cast<TensorType>()) {
        auto st = tt->scalarType();
        if (!st.has_value()) {
          return false;
        }

        auto device = tt->device();
        if (!device.has_value()) {
          return false;
        }

        bool is_cpu = device->is_cpu();

        // 如果数据类型不是float且为降低精度并且在CPU上，则返回false
        if (*st != at::kFloat && is_reduced_precision && is_cpu) {
          // 对于CPU，如果数据类型为float，则ATen不会执行任何操作。此时ATen的性能优于NNC。
          // 因此NNC不会将其合并到其融合组中。
          return false;
        }

        // 如果数据类型不是BFloat16且为完整精度并且在CPU上，则返回false
        if (*st != at::kBFloat16 && is_full_precision && is_cpu) {
          // 对于CPU，如果数据类型为BFloat16，则ATen不会执行任何操作。此时ATen的性能优于NNC。
          // 因此NNC不会将其合并到其融合组中。
          return false;
        }
      }

      // 检查节点是否有不支持的pin_memory属性
      if (has_unsupported_pin_memory(node)) {
        return false;
      }
    }

    // 检查节点是否为unsqueeze，并确保其dim参数是常量
    if (node->kind() == aten::unsqueeze) {
      if (node->input(1)->node()->kind() != prim::Constant) {
        return false;
      }
    }

    // 对于非支持的2D卷积节点，返回false
    if (node->kind() == aten::_convolution && !tensorexpr::isConv2d(node)) {
      GRAPH_DEBUG("This aten::_convolution node is not a 2D conv");
      return false;
    }

    // 对于convolution或conv2d节点，检查其是否被TensorExpr支持或MKLDNN预打包卷积支持
    if (node->kind() == aten::_convolution || node->kind() == aten::conv2d) {
      if (!tensorexpr::conv2dIsSupportedJit(node) &&
          !tensorexpr::mkldnnPrepackedConvIsSupportedJit(node)) {
        GRAPH_DEBUG("Params of conv2d are not supported");
        return false;
      }
    }

    // 对于matmul节点，检查其输入形状是否被TensorExpr支持
    if (node->kind() == aten::matmul) {
      if (!tensorexpr::matmulIsSupported(node)) {
        GRAPH_DEBUG("Shapes of matmul inputs are not supported");
        return false;
      }
    }

    // 如果以上条件都未触发返回false，则返回true
    return true;
  }
// 定义一个宏，用于检查条件是否成立，如果条件不成立，则打印调试信息并返回 false
#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

// 判断节点是否可以进行融合操作
bool canHandle(Node* node) {
  // 检查所有输入节点的形状是否已知
  REQ(allShapesAreKnown(node));
  // 检查节点是否可以在当前设备上进行融合
  REQ(isFusableOnDevice(node));
  // 检查节点是否在不可融合的操作集合中
  REQ(operators_not_to_fuse.find(node->kind()) == operators_not_to_fuse.end());

  // 遍历节点的输入值
  for (Value* input : node->inputs()) {
    if (auto const& tt = input->type()->cast<TensorType>()) {
      auto st = tt->scalarType();
      if (!st) {
        // 所有张量类型都应该是已知的
        return false;
      }
      if (c10::isComplexType(*st) || c10::isQIntType(*st)) {
        return false;
      }
    }
  }

  // 对于特定的节点类型进行额外的检查
  if (node->kind() == aten::cat) {
    REQ(node->input(0)->node()->kind() == prim::ListConstruct);
    REQ(node->input(0)->uses().size() == 1);
    REQ(node->input(1)->node()->kind() == prim::Constant);
    auto const& listconstruct = node->input(0)->node();
    REQ(tensorexpr::pickDeviceType(listconstruct->inputs()));
  } else {
    REQ(tensorexpr::pickDeviceType(node->inputs()));
  }

  // 仅当节点类型为 aten::batch_norm 且参数 'training' 为 false 时才进行融合
  if (node->kind() == aten::batch_norm) {
    REQ(node->input(5)->node()->kind() == prim::Constant);
    REQ(!toIValue(node->input(5)).value().toBool());
  }

  // 检查节点是否受支持
  REQ(tensorexpr::isSupported(node));
  // 检查节点的类型是否受支持
  REQ(typesAreSupported(node));

  // 允许优化限制的钩子，以允许通过二分法执行通道
  REQ(JIT_OPT_ALLOWED);

  // 如果允许动态形状融合，则进一步检查节点的特定条件
  if (fuse_to_dynamic_shapes_) {
    // 仅当节点类型为 prim::ListConstruct 或 prim::TensorExprGroup，
    // 或者节点属于自定义操作集合，或者节点具有定义的形状计算图时才允许融合
    REQ(node->kind() == prim::ListConstruct ||
        node->kind() == prim::TensorExprGroup ||
        node->isMemberOf(tensorexpr::getCustomOperatorSet()) ||
        (node->maybeSchema() && shapeComputeGraphForSchema(node->schema())));
  }

  // 能够处理当前节点的所有条件均通过，返回 true
  return true;
}

// 检查是否可以合并两个节点
bool canMerge(Node* consumer, Node* producer) {
  // 仅在同一块中的节点之间进行融合
  REQ(consumer->owningBlock() == producer->owningBlock());

  // 符号检查
  REQ(canHandle(producer) || producer->kind() == prim::TensorExprGroup);
  TORCH_INTERNAL_ASSERT(
      consumer->kind() == prim::TensorExprGroup || canHandle(consumer));

  // nvrtc 对 CUDA 内核中允许的参数数量有限制
  // 这里选择一个安全的参数限制值
  constexpr size_t subgraphArgLimit = 128;
  auto const nInputs = consumer->inputs().size() +
      consumer->outputs().size() + producer->inputs().size() +
      producer->outputs().size();
  REQ(nInputs <= subgraphArgLimit);

  // 设备检查
    // 如果消费者和生产者都不是 aten::cat，执行以下条件语句块
    if (consumer->kind() != aten::cat && producer->kind() != aten::cat) {
      // aten::cat 需要特殊处理，因为它接受一个 Tensor[] 作为输入，这部分在下面的代码中处理
      auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
      REQ(consumer_device);  // 确保消费者设备类型非空
      auto producer_device = tensorexpr::pickDeviceType(producer->inputs());
      REQ(producer_device);  // 确保生产者设备类型非空
      REQ(*consumer_device == *producer_device);  // 确保消费者和生产者设备类型一致
    }

    // 别名检查
    REQ(aliasDb_->couldMoveBeforeTopologically(producer, consumer));

    // 返回别名的操作只有在只有一个使用时才能折叠
    if (producer->kind() == aten::slice ||
        producer->kind() == aten::unsqueeze ||
        producer->kind() == prim::ConstantChunk) {
      for (auto& use : producer->output(0)->uses()) {
        REQ(use.user == consumer);  // 确保只有消费者在使用这些操作的输出
      }
    }

    // 如果消费者不包含 Subgraph 属性且不是 prim::TensorExprGroup 类型
    if (!consumer->hasAttribute(attr::Subgraph) &&
        consumer->kind() != prim::TensorExprGroup) {
      // 不要为 prim::ListConstruct 初始化融合组
      REQ(consumer->kind() != prim::ListConstruct);
      REQ(consumer->kind() != aten::slice);
      REQ(consumer->kind() != aten::unsqueeze);
      REQ(consumer->kind() != prim::ConstantChunk);

      // 不要为常量操作数初始化融合组
      REQ(producer->kind() != prim::Constant);
    }

    // 如果生产者是 aten::cat 操作
    if (producer->kind() == aten::cat) {
      REQ(producer->input(0)->node()->kind() == prim::ListConstruct);  // 确保输入0是 prim::ListConstruct 类型
      REQ(producer->input(0)->uses().size() == 1);  // 确保输入0只有一个使用者
      REQ(producer->input(1)->node()->kind() == prim::Constant);  // 确保输入1是 prim::Constant 类型
      auto const& listConstruct = producer->input(0)->node();
      // 我们正在合并 listconstruct->cat->consumer。这里 cat 是生产者，
      // 我们无法确定其设备类型 - 应该使用 listconstruct 的设备类型
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
      REQ(listconstruct_device);  // 确保 listconstruct 设备类型非空
      REQ(consumer_device);  // 确保消费者设备类型非空
      REQ(*listconstruct_device == *consumer_device);  // 确保 listconstruct 和消费者设备类型一致
      for (auto const& input : listConstruct->inputs()) {
        REQ(isFusableOnDevice(input->node()));  // 确保每个输入都可以在设备上融合
      }
      REQ((nInputs + listConstruct->inputs().size()) <= subgraphArgLimit);  // 确保输入总数不超过子图参数限制
    }
    } else if (consumer->kind() == aten::cat) {
      // 检查消费节点是否为 aten::cat
      REQ(consumer->input(0)->node()->kind() == prim::ListConstruct);
      // 要求消费节点的第一个输入节点为 prim::ListConstruct
      REQ(consumer->input(0)->uses().size() == 1);
      // 要求消费节点的第一个输入节点仅被一个节点使用
      REQ(consumer->input(1)->node()->kind() == prim::Constant);
      // 要求消费节点的第二个输入节点为 prim::Constant
      auto const& listConstruct = consumer->input(0)->node();
      // 获取 listConstruct 节点的引用，即消费节点的第一个输入节点
      // 我们正在合并 listconstruct->cat。cat 是消费节点，listconstruct 是生产节点。
      // cat 没有自己的设备类型，因此我们唯一需要检查的是 listconstruct 是否有定义良好的设备
      // （例如，其所有输入具有相同的设备）。
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      // 选择 listConstruct 节点输入的设备类型
      REQ(listconstruct_device);
      // 要求 listconstruct_device 不为空，即设备类型已定义
      REQ((nInputs + listConstruct->inputs().size()) <= subgraphArgLimit);
      // 要求总输入数和 listConstruct 节点的输入数之和不超过 subgraphArgLimit
    } else {
      // 如果消费节点不是 aten::cat，要求检查生产节点是否可以在设备上融合
      REQ(isFusableOnDevice(producer));
    }

    return true;
  }
// 解除预处理指令 '#undef REQ'

void prepareFusionGroupAndGuardOutputs(Block* block) {
  // 创建存储融合组节点的向量
  std::vector<Node*> fusion_groups;
  // 遍历块中的每个节点
  for (Node* n : block->nodes()) {
    // 递归处理节点内的子块
    for (Block* b : n->blocks()) {
      prepareFusionGroupAndGuardOutputs(b);
    }
    // 如果节点是张量表达式组节点，则加入融合组向量
    if (n->kind() == prim::TensorExprGroup) {
      fusion_groups.push_back(n);
    }
  }
  // 遍历所有融合组节点
  for (Node* fusion_group : fusion_groups) {
    // 移除仅用于大小的输出
    removeOutputsUsedOnlyInSize(fusion_group);
    // 插入类型保护，以确保类型匹配
    insertTypeGuard(
        fusion_group,
        [](const TensorTypePtr& t) { return t; },
        prim::TypeCheck);
  }
}

void generalizeFusionGroups(Block* block) {
  // 创建存储融合组节点的向量
  std::vector<Node*> fusion_groups;
  // 遍历块中的每个节点
  for (Node* n : block->nodes()) {
    // 递归处理节点内的子块
    for (Block* b : n->blocks()) {
      generalizeFusionGroups(b);
    }
    // 如果节点是张量表达式组节点，则加入融合组向量
    if (n->kind() == prim::TensorExprGroup) {
      fusion_groups.push_back(n);
    }
  }
  // 遍历所有融合组节点
  for (Node* fusion_group : fusion_groups) {
    // 移除仅用于大小的输出
    removeOutputsUsedOnlyInSize(fusion_group);
    // 输出日志：生成融合组的守卫
    VLOG(1) << "GenerateGuard for fusion group: " << *fusion_group;
    // 如果生成守卫失败，则取消融合组并输出日志
    if (!GenerateGuard(fusion_group, add_composed_op_)) {
      VLOG(1) << "  Unfusing the fusion group because GenerateGuard failed"
              << std::endl;
      // 分离子图以取消融合
      SubgraphUtils::unmergeSubgraph(fusion_group);
    }
  }
}

// 解析环境变量 "PYTORCH_TENSOREXPR_DONT_FUSE" 提供的选项
void parseTENotFuseOption() {
  const char* option = std::getenv("PYTORCH_TENSOREXPR_DONT_FUSE");
  // 使用 stringstream 将选项转为流
  std::stringstream in_ss;
  if (option) {
    in_ss << option;
  }

  std::string line;
  // 按 ':' 分隔处理环境变量中的运算符列表
  while (std::getline(in_ss, line, ':')) {
    if (line.empty()) {
      continue;
    }
    // 将运算符插入不融合的集合中
    operators_not_to_fuse.insert(c10::Symbol::aten(line));
  }
}

std::shared_ptr<Graph> graph_;
std::unique_ptr<AliasDb> aliasDb_ = nullptr;

// 不融合的运算符集合
std::set<NodeKind> operators_not_to_fuse;
// 融合组的最小大小
size_t min_group_size_;
// 是否添加组合操作和内核
bool add_composed_op_;
// 是否将静态形状融合为动态形状
bool fuse_to_dynamic_shapes_;

void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size,
    bool add_composed_op,
    bool fuse_to_dynamic_shapes) {
  // 输出图形的转储信息
  GRAPH_DUMP("Before TExprFuser: ", graph);

  // 临时更改块代码生成的最小组大小
  if (tensorexpr::getTEGenerateBlockCode()) {
    min_group_size = 1;
  }

  // 如果需要添加组合操作
  if (add_composed_op) {
    // 使用TORCH_INTERNAL_ASSERT宏来确保fuse_to_dynamic_shapes为真，否则抛出错误信息"Fusing static shapes with composed op NYI"
    TORCH_INTERNAL_ASSERT(
        fuse_to_dynamic_shapes, "Fusing static shapes with composed op NYI");
  }

  // 调用函数EliminateDeadCode，用于从图中移除死代码，以避免浪费融合的努力
  EliminateDeadCode(graph);

  // 创建TensorExprFuser对象fuser，用于张量表达式的融合
  TensorExprFuser fuser(
      graph, min_group_size, add_composed_op, fuse_to_dynamic_shapes);
  // 运行张量表达式的融合过程
  fuser.run();

  // 再次调用函数EliminateDeadCode，用于进一步从图中移除死代码
  EliminateDeadCode(graph);

  // 输出当前图的状态，带有"After TExprFuser: "前缀
  GRAPH_DUMP("After TExprFuser: ", graph);
}

// 创建一个操作，将节点转换为张量表达式操作
static Operation createTensorExprOp(const Node* node) {
  // 检查节点是否具有动态形状融合属性
  bool dynamic_shape_fusion_node =
      node->hasAttribute(attr::striding_inputs_desc);
  // 如果不是动态形状融合节点，创建一个新的张量表达式内核
  if (!dynamic_shape_fusion_node) {
    auto kernel =
        std::make_shared<tensorexpr::TensorExprKernel>(node->g(attr::Subgraph));
    return [kernel](Stack& stack) {
      // 记录函数调用，使用内核名称
      RECORD_FUNCTION(kernel->getKernelName(), std::vector<c10::IValue>());
      // 运行张量表达式内核
      kernel->run(stack);
      return 0;
    };
  }

  // 处理动态形状融合启用的情况
  VLOG(1) << "Compiling a new kernel for " << *node;
  // 获取符号形状输入
  std::vector<int64_t> sym_shapes;
  if (node->hasAttribute(attr::symbolic_shape_inputs)) {
    sym_shapes = node->is(attr::symbolic_shape_inputs);
  }
  // 检查是否允许堆栈输出
  bool allow_stack_outputs = false;
  if (node->hasAttribute(attr::allow_stack_outputs)) {
    allow_stack_outputs = node->i(attr::allow_stack_outputs) == 1;
  }

  // 自定义降维函数映射表
  std::unordered_map<c10::Symbol, tensorexpr::NNCLoweringFunction>
      custom_lowerings;
  auto subgraph = node->g(attr::Subgraph);
  // 获取步幅输入描述
  IValue sym_strides = node->ival(attr::striding_inputs_desc);

  // 将步幅描述符反序列化为步幅输入枚举
  std::vector<std::vector<std::string>> sym_strides_strs =
      sym_strides.to<std::vector<std::vector<std::string>>>();
  std::vector<std::vector<StrideInput>> striding_inputs;
  for (const auto& vec : sym_strides_strs) {
    std::vector<StrideInput> input_desc;
    input_desc.reserve(vec.size());
    for (const std::string& str : vec) {
      input_desc.push_back(strideInputFromString(str));
    }
    striding_inputs.push_back(input_desc);
  }

  // 创建值到步幅输入的映射
  std::unordered_map<const Value*, std::vector<StrideInput>> stride_map;
  size_t index = 0;
  for (Value* v : subgraph->inputs()) {
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    stride_map[v] = striding_inputs[index];
    index++;
  }

  // 设置输出步幅描述
  std::vector<std::string> output_desc =
      node->ival(attr::striding_outputs_desc).to<std::vector<std::string>>();
  for (size_t i = 0; i < subgraph->outputs().size(); ++i) {
    stride_map[subgraph->outputs().at(i)] = {
        strideInputFromString(output_desc.at(i))};
  }

  // 创建张量表达式内核对象
  std::shared_ptr<tensorexpr::TensorExprKernel> kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(
          subgraph,
          custom_lowerings,
          sym_shapes,
          /*pre_alloc*/ false,
          stride_map);

  // 子图输入数量
  auto num_subgraph_inputs = subgraph->inputs().size();
  return [kernel, num_subgraph_inputs, allow_stack_outputs](Stack& stack) {
    RECORD_FUNCTION(kernel->getKernelName(), std::vector<c10::IValue>());

    // 堆栈内容：
    //   [<outputs>] <inputs>
    //
    // 如果图的输入数量与堆栈大小相同，则没有输出被传递。
    // 否则，输出张量在堆栈底部传入。因此，我们调用适当的运行函数在TensorExprKernel中。

    // 运行张量表达式内核
    kernel->run(stack);
    return 0;
  };
}
    # 如果子图输入数量等于栈的大小，或者不允许使用栈输出
    if (num_subgraph_inputs == stack.size() || !allow_stack_outputs) {
      # 调用 kernel 对象的 run 方法来执行操作，传入当前栈的状态
      kernel->run(stack);
    } else {
      # 否则，调用 kernel 对象的 runWithAllocatedOutputs 方法执行操作，传入当前栈的状态
      kernel->runWithAllocatedOutputs(stack);
    }
    # 返回成功标志 0，表示函数执行完毕
    return 0;
  };
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        prim::TensorExprGroup,
        createTensorExprOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

} // namespace jit
} // namespace torch


注释：


} // 结束了 namespace jit

// 注册 TensorExprOps 操作符到 Torch 的运算符集中
RegisterOperators TensorExprOps({
    // 定义一个名为 prim::TensorExprGroup 的运算符
    torch::jit::Operator(
        prim::TensorExprGroup,
        createTensorExprOp,  // 使用 createTensorExprOp 函数来实现这个运算符
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),  // 指定别名分析的特殊情况
});

} // 结束了 namespace torch
```