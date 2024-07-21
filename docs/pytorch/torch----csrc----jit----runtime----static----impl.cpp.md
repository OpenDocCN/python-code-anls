# `.\pytorch\torch\csrc\jit\runtime\static\impl.cpp`

```py
// 引入头文件实现静态运行时的相关功能
#include <torch/csrc/jit/runtime/static/impl.h>

// 引入 ATen 库中的其他功能和定义
#include <ATen/MemoryOverlap.h>
#include <ATen/core/symbol.h>
#include <ATen/record_function.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/macros/Macros.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/irange.h>
#include <caffe2/core/timer.h>

// 引入 JIT 编译器的相关功能和优化 pass
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/add_if_then_else.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/eliminate_no_ops.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

// 引入标准库函数和类型定义
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

// 仅在 FBCODE_CAFFE2 宏定义下引入的额外库和功能
#ifdef FBCODE_CAFFE2
#include <common/logging/logging.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#endif

// 用于测试目的的标志定义，控制静态运行时是否禁用内存重叠检查
C10_DEFINE_bool(
    static_runtime_disable_debug_memory_overlap_check,
    false,
    "If true, disable the memory overlap check in debug mode in ProcessedNode::run()");

// 命名空间 torch::jit 下的匿名命名空间，用于封装实现细节
namespace torch::jit {

// 匿名命名空间中的函数，将 c10::IValue 转换为字符串表示
namespace {

std::string iValueToString(const c10::IValue& val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

// 检查节点的所有输入是否都是张量类型
bool allArgsAreTensors(const Node* node) {
  const auto& inputs = node->inputs();
  return std::all_of(inputs.begin(), inputs.end(), [](const Value* value) {
    return value->type()->kind() == TypeKind::TensorType;
  });
}

} // namespace

// 判断节点是否是静态运行时不支持的操作
// 这些操作通常很少使用，禁止它们有助于消除图优化中的边缘情况，从而实现更激进的优化和更好的性能
static bool isUnsupportedOp(const Node* node) {
  auto kind = node->kind();
  if (kind != aten::__is__ && kind != aten::__isnot__) {
    return false;
  }

  // 不能支持带有张量参数的 aten::__is__ 和 aten::__isnot__
  // 例如在推断中移除无效的 detach 节点会影响结果
  return allArgsAreTensors(node);
}

} // namespace
// 检查是否可以启用静态运行时的实现，返回布尔值
bool canEnableStaticRuntimeImpl(const Block* block) {
  // 如果 block 为空指针，直接返回 false
  if (block == nullptr) {
    return false;
  }

  // 初始化可以支持的标志为 true
  bool can_support = true;
  // 遍历 block 中的每个节点
  for (auto* node : block->nodes()) {
    // 对每个节点的子块进行递归调用 canEnableStaticRuntimeImpl
    for (auto* subblock : node->blocks()) {
      // 通过 && 运算符来确保所有不支持的操作都能被记录下来
      can_support = canEnableStaticRuntimeImpl(subblock) && can_support;
    }

    // 获取当前节点的类型
    const auto kind = node->kind();
    // 如果节点类型是 prim::Constant，则继续下一个节点
    if (kind == prim::Constant) {
      continue;
    }
    // 获取节点的操作符指针
    const Operator* op = node->maybeOperator();
    // 如果是不支持的操作或者操作符为空且未注册的原生操作，则将 can_support 设置为 false
    if (isUnsupportedOp(node) || (!op && !nativeOpIsRegistered(kind))) {
      can_support = false;
      // 记录警告日志，显示找到的不支持的操作类型
      LOG(WARNING) << "Found unsupported op: " << kind.toQualString();
    }
  }
  // 返回是否可以支持静态运行时的结果
  return can_support;
}

} // namespace

// 图必须是冻结状态。如果图中还有 prim::CallMethod 操作，canEnableStaticRuntime 将返回 false。
bool canEnableStaticRuntime(const std::shared_ptr<torch::jit::Graph>& graph) {
  // 调用 canEnableStaticRuntimeImpl 函数来检查图中的所有块
  return canEnableStaticRuntimeImpl(graph->block());
}

namespace {

// 静态全局变量 sr_metadata_registerer，注册了 StaticRuntimeMetadata 类
auto sr_metadata_registerer = torch::class_<StaticRuntimeMetadata>(
    "StaticRuntime",
    "StaticRuntimeMetadata");

} // namespace

// 打印值集合的字符串表示，用于调试目的
std::string dumpValueSet(
    const c10::FastSet<const Value*>& value_set,
    const char* set_name) {
  // 使用流来构建输出字符串
  std::ostringstream oss;
  oss << set_name << ": {";
  // 遍历值集合，将每个值的调试名称添加到输出流中
  for (const auto* val : value_set) {
    oss << "%" << val->debugName() << ", ";
  }
  oss << "}";
  // 返回流的字符串表示
  return oss.str();
}

namespace {

// 对图进行优化，修改图的内容
void OptimizeGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  // 在优化之前打印图的状态
  GRAPH_DUMP("Before optimizations: ", graph);
  // 如果启用了 TensorExpr 融合
  if (opts.enable_tensorexpr_fusion) {
    // 如果样本输入为空，输出警告信息
    if (sample_inputs.empty()) {
      VLOG(1) << "Cannot perform TensorExpr fusion - sample_inputs is empty";
    } else {
      // 执行 TensorExpr 融合
      VLOG(1) << "Performing TensorExpr fusion";
      performTensorExprFusion(graph, std::move(sample_inputs));
    }
  }
  // 内联函数调用
  Inline(*graph);
  // 常量传播
  ConstantPropagation(graph);
  // 规范化操作
  Canonicalize(graph);
  // 再次进行常量传播
  ConstantPropagation(graph);
  // 移除张量变异操作
  RemoveTensorMutation(graph);
  // 再次进行常量传播
  ConstantPropagation(graph);
  // 消除无操作的切片
  EliminateNoOpSlice(graph);
  // 消除死代码
  EliminateDeadCode(graph);
  // 融合稀疏神经网络推断操作
  FuseInferenceOpsForSparseNN(graph);
  // 使用变参的 cat 操作
  UseVariadicCat(graph);
  // 使用变参的 stack 操作
  UseVariadicStack(graph);
  // 消除平均分割的无意义排列操作
  EliminateTrivialEquallySplit(graph);
  // 消除额外的排列操作
  EliminateExtraPermuteOps(graph);

  // 如果启用了输出变体选项
  if (opts.enable_out_variant) {
    // 使用变参的操作，将指定的操作符从一个类型转换为另一个类型
    UseVariadicOp(
        graph,
        fromQualString("fb::sigrid_transforms_torch_bind"),
        fromQualString("fb::variadic_sigrid_transforms_torch_bind"));
    UseVariadicOp(
        graph,
        fromQualString("torcharrow::inference_wrapper_run_flat"),
        fromQualString("torcharrow::variadic_inference_wrapper_run_flat"));
    // 这些融合的操作只有输出变体 - 当禁用输出变体时无法进行融合
    FuseSignLog1P(graph);
    FuseClampNaNToNum(graph);

#ifdef FBCODE_CAFFE2
    # 如果选项中启用了使用复制变体，并且未启用 TensorExpr 融合，则执行以下操作
    if (opts.use_copy_variants && !opts.enable_tensorexpr_fusion) {
      # 替换图中的某些操作以使用复制
      ReplaceWithCopy(graph);
    } else {
      # 否则，替换图中的某些操作以使用复制和排列
      ReplacePermuteWithCopy(graph);
    }
    # 如果选项中启用了使用可能复制的变体，并且未启用 TensorExpr 融合，则执行以下操作
    if (opts.use_maybe_copy_variants && !opts.enable_tensorexpr_fusion) {
      # 替换图中的某些操作以使用可能的复制
      ReplaceWithMaybeCopy(graph);
    }
    # 对图中的列表展开操作进行融合优化
    FuseListUnpack(graph);
    # 移除图中不必要的输出
    RemoveUnnecessaryOutputs(graph);
    # 对图中的权重进行预打包优化
    PrepackWeights(graph);
#endif
  }



ConstantPropagation(graph);



RemoveImmutableInputDictLookups(graph);



UseVariadicTupleUnpack(graph);



UseVariadicGroupedAccessor(graph);



EliminateNoOps(
    graph, /* custom_ops */ {fromQualString("fb::scale_gradient")});



AddIfThenElseOp(graph);



UseSplitAndSqueeze(graph);



UseInPlaceGetRealInputsFromOptionalInputsV2(graph);



GRAPH_DUMP("Final graph after optimizations: ", graph);



}

bool IsSelfInGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  return !graph->inputs().empty() && graph->inputs().at(0)->type()->is_module();
}



// remove unused input 0 from graph
bool removeSelfFromGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  if (graph->inputs().at(0)->type()->is_module()) {
    if (graph->inputs().at(0)->hasUses()) {
      return false;
    }
    graph->eraseInput(0);
  }
  return true;
}



std::vector<Value*> valueVecFromFastSet(const c10::FastSet<const Value*>& s) {
  std::vector<Value*> result;
  result.reserve(s.size());
  for (auto* v : s) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    result.emplace_back(const_cast<Value*>(v));
  }
  return result;
}



bool mayContainAlias(const AliasDb& db, const Value* v1, const Value* v2) {
  // AliasDb is not const-correct here, so we have to const_cast
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(v1), const_cast<Value*>(v2));
}



bool mayContainAlias(
    const AliasDb& db,
    const Value* a,
    const c10::FastSet<const Value*>& b) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(a), valueVecFromFastSet(b));
}



bool escapesScope(const AliasDb& db, const Value* a) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.escapesScope({const_cast<Value*>(a)});
}



void PrepareGraphForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  TORCH_CHECK(canEnableStaticRuntime(graph));
  OptimizeGraph(graph, opts, std::move(sample_inputs));

  // Static runtime moves its outputs out of the runtime
  // by default. In some rare cases, this is not actually safe to
  // do - for example, if the value is a constant, static runtime
  // needs to hold onto a copy. Rather than adding special logic
  // to handle this rare case, we use this pass to detect it and
  // create an owned reference that can be safely moved out of the
  // runtime.
  CreateOwnedRefsForSpecialValues(*graph);

  // We assume that each sub-block has at least one output. If we
  // detect any that have 0, force the sub-block to return None.
  ForceNonEmptyOutputs(*graph);
}



std::pair<std::shared_ptr<Graph>, std::optional<Module>> PrepareForStaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts,
  # 打印静态模块选项的日志信息，包括各种优化选项和变体使用情况
  LOG(INFO) << "StaticModuleOptions: enable_out_variant "
            << opts.enable_out_variant << ", optimize_memory "
            << opts.optimize_memory << ", manage_output_tensors "
            << opts.manage_output_tensors << ", use_copy_variants "
            << opts.use_copy_variants << ", use_maybe_copy_variants "
            << opts.use_maybe_copy_variants << ", enable_tensorexpr_fusion "
            << opts.enable_tensorexpr_fusion;

  # 复制模块 `m` 并命名为 `module`
  Module module = m.copy();
  
  # 如果模块未被冻结，则进行评估和冻结处理
  if (!is_frozen) {
    module.eval();  // 将模块设为评估模式
    module = freeze_module(module);  // 冻结模块，返回冻结后的模块
  }

  # 获取模块中名为 "forward" 的方法，并将其作为 `method`
  Method method = module.get_method("forward");
  
  # 获取名为 "forward" 的方法的计算图
  auto graph = module.get_method("forward").graph();

  # 如果输入样本不为空且图中包含自身输入，则将模块的输入值插入样本输入的开头
  if (!sample_inputs.empty() && IsSelfInGraphInput(graph)) {
    sample_inputs.insert(sample_inputs.begin(), m._ivalue());
  }

  # 准备图形以用于静态模块，应用静态模块选项和样本输入
  PrepareGraphForStaticModule(graph, opts, std::move(sample_inputs));

  # 返回包含计算图和模块的 `std::pair` 对象
  return std::make_pair(graph, module);
} // 结束函数 PrepareForStaticModule

std::pair<std::shared_ptr<Graph>, std::optional<Module>> PrepareForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  // 调用函数 PrepareGraphForStaticModule 准备静态模块的图形
  PrepareGraphForStaticModule(graph, opts, std::move(sample_inputs));
  // 返回图形和空的模块
  return std::make_pair(graph, c10::nullopt);
}

} // namespace 结束匿名命名空间

void ValueGroup::init(const Block& block, const AliasDb& db) {
  // 清空外部别名和输出别名集合
  external_aliases_.clear();
  output_aliases_.clear();

  // 从图形的输入中构建 external_aliases 集合，
  // 并且添加这些输入节点的别名
  external_aliases_.insert(block.inputs().begin(), block.inputs().end());

  // 遍历块中的每个节点
  for (const auto* node : block.nodes()) {
    // 如果节点是常量节点
    if (node->kind() == prim::Constant) {
      // 将节点的输出别名添加到 external_aliases
      for (const auto* output : node->outputs()) {
        external_aliases_.insert(output);
      }
    }
  }

  // 再次遍历块中的每个节点
  for (const auto* node : block.nodes()) {
    // 如果节点是常量节点，则跳过
    if (node->kind() == prim::Constant) {
      continue; // 常量已经在 external_aliases 中
    }
    // 对于节点的每个输出值
    for (const auto* v : node->outputs()) {
      // 如果 v 逃逸于作用域或者可能包含别名，则将其添加到 external_aliases
      if (escapesScope(db, v) || mayContainAlias(db, v, external_aliases_)) {
        external_aliases_.insert(v);
      }
    }
  }

  // 构建 output_aliases 集合，
  // 从输出值开始，反向遍历节点以跟踪数据流
  output_aliases_.insert(block.outputs().begin(), block.outputs().end());

  // 反向遍历块中的每个节点
  for (const auto* node : block.nodes().reverse()) {
    // 如果节点是常量节点，则跳过
    if (node->kind() == prim::Constant) {
      continue; // 常量不会创建任何别名
    }
    // 对于节点的每个输出值
    for (const auto* v : node->outputs()) {
      // 如果可能包含别名，则将其添加到 output_aliases
      if (mayContainAlias(db, v, output_aliases_)) {
        output_aliases_.insert(v);
      }
    }
  }
}

namespace {

// 判断值是否为张量列表
bool isTensorList(const Value* value) {
  auto* type = value->type()->castRaw<ListType>();
  if (!type) {
    return false;
  }
  return type->getElementType()->kind() == c10::TypeKind::TensorType;
}

// 判断是否只包含张量
bool containTensorsOnly(at::ArrayRef<Value*> values) {
  // 只有当所有输出值都是张量时返回 true
  return std::all_of(values.begin(), values.end(), [](const Value* value) {
    return value->type()->kind() == c10::TypeKind::TensorType ||
        isTensorList(value);
  });
}

// 判断节点是否为纯函数
bool isPureFunction(const Node* node) {
  auto* schema = node->maybeSchema();
  return schema &&
      schema->aliasAnalysis() == c10::AliasAnalysisKind::PURE_FUNCTION;
}

} // namespace 结束匿名命名空间

ManagedTensorRanges::ManagedTensorRanges(
    Block& block,
    const AliasDb& alias_db,
    const c10::FastSet<const Value*>& managed_tensor_values) {
  // 将块中的节点转换为节点向量
  const std::vector<Node*> nodes(block.nodes().begin(), block.nodes().end());

  // 将块中的输入值转换为集合
  const c10::FastSet<const Value*> graph_inputs(
      block.inputs().begin(), block.inputs().end());

  // 节点数目
  const auto num_nodes = static_cast<uint32_t>(nodes.size());

  // 遍历所有节点
  for (const auto i : c10::irange(num_nodes)) {
    auto* node = nodes[i];
    // 遍历节点的输入值列表
    for (auto* input : node->inputs()) {
      // 获取输入值的生命周期信息
      auto* lifetime = getLifetime(input);
      // 如果找不到生命周期信息，跳过当前循环
      if (!lifetime) {
        continue;
      }
      // 断言当前节点结束位置小于或等于 i
      DCHECK(lifetime->end <= i);
      // 更新输入值的生命周期结束位置为 i
      lifetime->end = i;
    }
    // 遍历节点的输出值列表
    for (auto* output : node->outputs()) {
      // 如果输出值不是可变类型，则跳过当前循环
      if (!alias_db.isMutableType(output)) {
        continue;
      }
      // 将输出值及其生命周期信息加入到 value_lifetimes_ 中
      value_lifetimes_.emplace(output, Lifetime(i, i));
    }
  }
  // 遍历基本块的输出值列表
  for (auto* graph_output : block.outputs()) {
    // 获取输出值的生命周期信息
    auto* lifetime = getLifetime(graph_output);
    // 如果找不到生命周期信息，跳过当前循环
    if (!lifetime) {
      continue;
    }
    // 更新输出值的生命周期结束位置为 num_nodes
    lifetime->end = num_nodes;
  }

  // 处理别名。别名可能会延长值的生命周期。如果一个节点
  // 的输入和输出可能有别名关系，将输入的生命周期结束位置设为
  // input.lifetime_end 和 output.lifetime_end 的最大值。倒序迭代以处理别名链。
  for (const auto* node : block.nodes().reverse()) {
    // 如果节点是纯函数，则跳过处理
    if (isPureFunction(node)) {
      continue;
    }

    // 收集节点输入和输出的具有跟踪生命周期的值
    auto inputs = collectValuesWithTrackedLifetimes(node->inputs());
    auto outputs = collectValuesWithTrackedLifetimes(node->outputs());
    for (auto* input : inputs) {
      // 获取输入值的生命周期信息
      auto* input_lifetime = getLifetime(input);
      DCHECK(input_lifetime != nullptr);
      for (auto* output : outputs) {
        // 如果输入和输出可能存在别名关系
        if (mayContainAlias(alias_db, input, output)) {
          // 获取输出值的生命周期信息
          auto* output_lifetime = getLifetime(output);
          DCHECK(output_lifetime != nullptr);
          // 更新输入值的生命周期结束位置为输入和输出生命周期结束位置的最大值
          input_lifetime->end =
              std::max(output_lifetime->end, input_lifetime->end);
        }
      }
    }
  }
  // 遍历 managed_tensor_values 列表中的托管张量
  for (auto* managed_tensor : managed_tensor_values) {
    // 获取托管张量的生命周期信息
    auto* lifetime = getLifetime(managed_tensor);
    DCHECK(lifetime && lifetime->end <= num_nodes);
    // 定义释放节点
    Node* freeing_node;
    // 如果生命周期结束位置为 num_nodes，则释放节点为基本块的返回节点
    if (lifetime->end == num_nodes) {
      freeing_node = block.return_node();
    } else {
      // 否则，释放节点为 nodes[lifetime->end]
      freeing_node = nodes[lifetime->end];
    }
    // 将托管张量加入到释放节点对应的新释放张量列表中
    node_to_newly_free_tensors_[freeing_node].emplace_back(managed_tensor);
  }
}

// 检查节点是否释放了托管张量
bool ManagedTensorRanges::nodeFreesManagedTensors(Node* node) const {
  // 查找节点是否在新释放张量的映射中
  auto it = node_to_newly_free_tensors_.find(node);
  // 返回节点是否在映射中，并且映射值不为空
  return it != node_to_newly_free_tensors_.end() && !it->second.empty();
}

// 获取节点之后可用的张量数值列表
const std::vector<const Value*>& ManagedTensorRanges::
    availableTensorValuesAfterNode(Node* node) const {
  // 返回节点之后可用的张量值列表
  return node_to_newly_free_tensors_.at(node);
}

// 检查两个值的生命周期是否重叠
bool ManagedTensorRanges::lifetimesOverlap(const Value* v1, const Value* v2)
    const {
  // 获取 v1 和 v2 的生命周期
  const auto* v1_lifetime = getLifetime(v1);
  const auto* v2_lifetime = getLifetime(v2);
  // 如果任一值的生命周期不存在，则认为不重叠
  if (!v1_lifetime || !v2_lifetime) {
    return false;
  }
  // 检查生命周期区间是否重叠
  if (v1_lifetime->start < v2_lifetime->start) {
    return v1_lifetime->end >= v2_lifetime->start;
  }
  return v2_lifetime->end >= v1_lifetime->start;
}

// 获取值的生命周期
const ManagedTensorRanges::Lifetime* ManagedTensorRanges::getLifetime(
    const Value* value) const {
  // 查找值对应的生命周期
  auto it = value_lifetimes_.find(value);
  // 如果找到则返回生命周期，否则返回空指针
  if (it != value_lifetimes_.end()) {
    return &it->second;
  }
  return nullptr;
}

// 获取值的生命周期（非常量版本）
ManagedTensorRanges::Lifetime* ManagedTensorRanges::getLifetime(
    const Value* value) {
  // 将 this 指针转换为 const 指针，以调用常量版本的 getLifetime
  const auto* const_this = const_cast<const ManagedTensorRanges*>(this);
  // 调用常量版本的 getLifetime，并转换结果为非常量指针
  return const_cast<ManagedTensorRanges::Lifetime*>(
      const_this->getLifetime(value));
}

// 收集具有跟踪生命周期的值列表
std::vector<const Value*> ManagedTensorRanges::
    collectValuesWithTrackedLifetimes(at::ArrayRef<const Value*> values) {
  // 存储可变值列表
  std::vector<const Value*> mutable_values;
  mutable_values.reserve(values.size());
  // 将具有跟踪生命周期的值复制到 mutable_values 中
  std::copy_if(
      values.begin(),
      values.end(),
      std::back_inserter(mutable_values),
      [this](const Value* value) { return getLifetime(value) != nullptr; });
  // 返回收集到的值列表
  return mutable_values;
}

// 构造函数：使用静态模块选项和样本输入准备静态模块
StaticModule::StaticModule(
    std::shared_ptr<torch::jit::Graph> g,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs)
    : StaticModule(
          PrepareForStaticModule(g->copy(), opts, std::move(sample_inputs)),
          opts) {}

// 构造函数：使用静态模块选项和模块对象准备静态模块
StaticModule::StaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs)
    : StaticModule(
          PrepareForStaticModule(m, is_frozen, opts, std::move(sample_inputs)),
          opts) {}

// 构造函数：使用图和可选模块对象准备静态模块
StaticModule::StaticModule(
    std::pair<std::shared_ptr<torch::jit::Graph>, std::optional<Module>>
        graph_and_module,
    const StaticModuleOptions& opts)
    // 使用成员初始化列表初始化 opts_, graph_, module_, num_inputs_
    // opts_ 是构造函数的一个成员变量，包含配置选项
    // graph_ 和 module_ 是通过 std::move() 移动的参数 graph_and_module 的值
    // num_inputs_ 初始化为图的输入数量
    : opts_(opts),
      graph_(std::move(graph_and_module.first)),
      module_(std::move(graph_and_module.second)),
      num_inputs_(graph_->inputs().size()) {
    // 使用 opts_ 创建一个 jit::StaticRuntimeMetadata 对象，作为 sr_metadata_
    sr_metadata_ = c10::make_intrusive<jit::StaticRuntimeMetadata>(opts_);
    // 递归地将元数据附加到 prim::fork 节点
    attachNodeMetadata(graph_->block());
    
    // 检查选项标志
    if (opts.manage_output_tensors) {
      // 如果 manage_output_tensors 设置为 true，则检查 enable_out_variant 也必须为 true
      TORCH_CHECK(
          opts_.enable_out_variant,
          "When manage_output_tensors is true, enable_out_variant must be set to true");
    }
    if (opts_.optimize_memory) {
      // 如果 optimize_memory 设置为 true，则检查 enable_out_variant 也必须为 true
      TORCH_CHECK(
          opts_.enable_out_variant,
          "When optimize_memory is true, enable_out_variant must be set to true");
    }
    
    // 处理模块的 schema
    if (module_.has_value()) {
      // 获取模块的 forward 方法
      Method method = module_->get_method("forward");
      // 获取 forward 方法的 schema
      schema_ = method.function().getSchema();
      // 获取 schema 参数的数量
      const auto num_schema_args = schema_->arguments().size();
      DCHECK(num_schema_args > 0);
      // 如果 removeSelfFromGraphInput 返回 true，则移除模块并调整输入数量
      if (removeSelfFromGraphInput(graph_)) {
        module_ = c10::nullopt;
        num_inputs_ = num_schema_args - 1;
      }
    }
    
    // 计算图中节点的数量和常量的数量
    {
      size_t nodes_size = 0, constants_size = 0;
      for (Node* node : graph_->nodes()) {
        // 统计节点和常量的数量
        ++(node->kind() == prim::Constant ? constants_size : nodes_size);
      }
    
      // 预留存储空间给 constants_ 和 functions_
      constants_.reserve(constants_size);
      functions_.reserve(nodes_size);
    }
    
    // 创建 AliasDb 对象，用于分析图中的别名信息，isFrozen 参数为 false
    AliasDb alias_db(graph_, /*isFrozen=*/false);
    // 打印 AliasDb 的信息
    GRAPH_DEBUG("AliasDb: ", alias_db.toString());
    
    // 准备函数和常量，以及它们在 StaticRuntime 中的索引
    c10::FastMap<const Value*, uint32_t> value_to_index;
    prepareFunctionsAndConstants(graph_->block(), alias_db, value_to_index);
    
    // 设置常量索引的偏移量
    const auto constants_index_offset = 0;
    // 设置值索引的偏移量，包括常量的大小
    const auto values_index_offset = constants_index_offset + constants().size();
    // 设置值缓冲区的大小
    value_buffer_size_ = values_index_offset;
    
    // 准备块信息，计算值缓冲区的大小
    value_buffer_size_ +=
        prepareBlockInfo(graph_->block(), values_index_offset, value_to_index);
    
    // 准备静态节点信息
    prepareStaticNodeInfos(graph_->block(), value_to_index, alias_db);
    
    // 针对每个块信息，为内存规划器准备块信息
    for (auto& block_and_info : block_infos_) {
      auto& block_info = block_and_info.second;
      block_info.prepare_for_memory_planner(alias_db, opts);
    }
}



size_t StaticModule::prepareBlockInfo(
    Block* block,
    const size_t start_idx,
    c10::FastMap<const Value*, uint32_t>& value_to_index) {
  // 将当前块的信息存储到 block_infos_ 中
  block_infos_.emplace(block, BlockInfo(start_idx, *block));

  // 获取当前块的输入数量
  const auto num_inputs = static_cast<uint32_t>(block->inputs().size());
  // 将当前块的输入值映射到索引
  for (const auto i : c10::irange(num_inputs)) {
    value_to_index.emplace(block->inputs()[i], start_idx + i);
  }
  auto cur_idx = start_idx + num_inputs;

  // 遍历当前块的节点
  for (auto* node : block->nodes()) {
    // 对当前节点中的每个子块递归调用 prepareBlockInfo，并更新当前索引
    for (auto* sub_block : node->blocks()) {
      cur_idx += prepareBlockInfo(sub_block, cur_idx, value_to_index);
    }

    // 如果当前节点是常量节点，跳过后续操作
    if (node->kind() == prim::Constant) {
      continue;
    }

    // 检查当前索引是否超过 2 字节的存储范围
    TORCH_CHECK(
        cur_idx < (1 << 16),
        "outputs offset in values table",
        cur_idx,
        " would overflow 2-byte index storage");

    // 获取当前节点的输出数量，并将输出值映射到索引
    const auto num_outputs = static_cast<uint32_t>(node->outputs().size());
    for (const auto i : c10::irange(num_outputs)) {
      value_to_index.emplace(node->outputs()[i], cur_idx + i);
    }
    cur_idx += num_outputs;
  }

  // 准备当前块的输出索引，并存储到 block_infos_ 中
  std::vector<uint16_t> output_indices;
  output_indices.reserve(block->outputs().size());
  for (auto* output : block->outputs()) {
    const auto output_idx = value_to_index.at(output);
    // 检查输出索引是否超过 2 字节的存储范围
    TORCH_CHECK(
        output_idx < (1 << 16),
        "outputs offset in values table",
        output_idx,
        " would overflow 2-byte index storage");
    output_indices.push_back(output_idx);
  }

  block_infos_.at(block).set_output_indices(std::move(output_indices));
  return cur_idx - start_idx;
}



void StaticModule::attachNodeMetadata(Block* block) {
  // 遍历当前块的节点，附加静态运行时元数据到 fork 节点
  for (auto* node : block->nodes()) {
    if (node->kind() == prim::fork) {
      node->ival_(getStaticRuntimeMetadataSymbol(), IValue(sr_metadata_));
    }
    // 递归处理当前节点中的每个子块
    for (auto* sub_block : node->blocks()) {
      attachNodeMetadata(sub_block);
    }
  }
}



void StaticModule::prepareFunctionsAndConstants(
    Block* block,
    const AliasDb& alias_db,
    c10::FastMap<const Value*, uint32_t>& value_to_index) {
  // 遍历当前块的节点
  for (auto* node : block->nodes()) {
    // 递归处理当前节点中的每个子块
    for (auto* sub_block : node->blocks()) {
      prepareFunctionsAndConstants(sub_block, alias_db, value_to_index);
    }

    // 如果当前节点是常量节点，处理常量和相关索引
    if (node->kind() == prim::Constant) {
      auto* v = node->output();
      // 检查节点输出类型是否符合预期
      TORCH_CHECK(
          v->type()->kind() != FunctionType::Kind,
          "got ",
          typeKindToString(v->type()->kind()),
          " instead of ",
          typeKindToString(FunctionType::Kind));
      // 将常量节点的值映射到索引，并存储到 constants_ 中
      value_to_index.emplace(v, constants_.size());
      constants_.emplace_back(toIValue(v).value());
      continue;
    }

    // 检查和修正运行时别名信息中的错误模式
    bool check_outputs_for_overlap =
        !alias_db.mayContainAlias(node->inputs(), node->outputs()) &&
        containTensorsOnly(node->outputs());
    // 创建并存储新的 ProcessedFunction 对象
    functions_.emplace_back(
        node, opts_.enable_out_variant, check_outputs_for_overlap);
  }
}



size_t StaticModule::prepareStaticNodeInfos(
    Block* block,
    // 获取指定块的起始节点索引
    const c10::FastMap<const Value*, uint32_t>& value_to_index,
        // 别名数据库的引用
        const AliasDb& alias_db,
        // 节点索引的起始位置
        size_t node_idx) {
      // 保存当前节点索引作为起始位置
      const auto node_start = node_idx;
    
      // 获取当前块的信息引用
      auto& block_info = block_infos_.at(block);
      // 创建静态节点信息的空向量
      std::vector<StaticNodeInfo> nodes;
      // 节点是否具有输出变体的快速映射
      c10::FastMap<Node*, bool> node_has_out_variant;
    
      // 遍历当前块中的每个节点
      for (auto* node : block->nodes()) {
        // 如果节点是常量节点，则跳过
        if (node->kind() == prim::Constant) {
          continue;
        }
    
        // 遍历节点的子块并准备静态节点信息，累加节点索引
        for (auto* sub_block : node->blocks()) {
          node_idx +=
              prepareStaticNodeInfos(sub_block, value_to_index, alias_db, node_idx);
        }
        // 获取节点输入的数量
        const auto num_outputs = static_cast<uint32_t>(node->inputs().size());
        // 处理节点输入的索引
        ProcessedNodeInputs input_indices(num_outputs);
        for (const auto input_idx : c10::irange<uint32_t>(num_outputs)) {
          auto* input = node->inputs()[input_idx];
          // 获取输入值在索引映射中的位置
          auto input_ivalue_idx = value_to_index.at(input);
          // 检查输入索引是否小于 2 字节索引存储的最大值
          TORCH_CHECK(
              input_ivalue_idx < (1 << 16),
              "input index in values table ",
              input_ivalue_idx,
              " would overflow 2-byte index storage");
          input_indices[input_idx] = input_ivalue_idx;
        }
    
        // 获取当前节点的处理函数指针
        ProcessedFunction* fn = &functions_[node_idx];
    
        // 创建一个新的静态节点
        const auto node_output_idx = node->outputs().empty()
            // 如果没有输出，则索引未使用，创建占位符值
            ? std::numeric_limits<uint16_t>::max()
            // 否则获取节点输出的索引
            : value_to_index.at(node->output(0));
        nodes.emplace_back(node, fn, std::move(input_indices), node_output_idx);
    
        // 记录节点是否有输出变体
        node_has_out_variant.emplace(node, nodes.back().has_out_variant());
        // 增加节点索引
        ++node_idx;
      }
    
      // 设置块信息的节点，传递节点是否有输出变体的信息
      block_info.set_nodes(std::move(nodes), node_has_out_variant);
      // 初始化块信息的值组别
      block_info.init_value_group(alias_db);
    
      // 返回节点处理完毕后的节点数量
      return node_idx - node_start;
    }
}

#ifdef FBCODE_CAFFE2
// 定义线程局部变量 tlsOpObserver，用于跟踪当前线程的操作观察器
thread_local SROperatorObserver* tlsOpObserver = nullptr;

// 设置当前线程的操作观察器
void SROperatorObserver::setCurrentThreadObserver(
    SROperatorObserver* observer) {
  tlsOpObserver = observer;
}

// 获取当前线程的操作观察器
SROperatorObserver* SROperatorObserver::getCurrentThreadObserver() {
  return tlsOpObserver;
}

// 在节点开始运行时调用的回调函数
void SROperatorObserver::onStart(const Node* node) {
  // 如果当前线程的操作观察器存在且具有开始回调函数，则调用之
  if (tlsOpObserver != nullptr && tlsOpObserver->startCb != nullptr) {
    tlsOpObserver->startCb(node);
  }
}

// 在节点结束运行时调用的回调函数
void SROperatorObserver::onEnd(const Node* node) {
  // 如果当前线程的操作观察器存在且具有结束回调函数，则调用之
  if (tlsOpObserver != nullptr && tlsOpObserver->endCb != nullptr) {
    tlsOpObserver->endCb(node);
  }
}
#endif // FBCODE_CAFFE2

// BlockInfo 类的构造函数，初始化输入索引和块对象的引用
BlockInfo::BlockInfo(uint32_t input_idx, Block& block)
    : input_idx_(input_idx), block_(block) {}

// 设置节点信息及其输出变体的函数
void BlockInfo::set_nodes(
    std::vector<StaticNodeInfo> nodes,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  nodes_ = std::move(nodes);

  // 遍历节点列表，标记可优化的容器类型节点
  for (auto& node : nodes_) {
    if (node.num_outputs() == 1 &&
        isOptimizableContainerType(node.node(), node_has_out_variant)) {
      node_is_optimizable_container_type_.emplace(node.node());
    }
  }
}

// 为内存规划器准备数据的函数
void BlockInfo::prepare_for_memory_planner(
    const AliasDb& alias_db,
    const StaticModuleOptions& opts) {
  // 如果未启用输出变体，则直接返回
  if (!opts.enable_out_variant) {
    return;
  }

  // 初始化图的输出值集合，用于管理输出张量
  c10::FastSet<const Value*> graph_output_values(
      block_.outputs().begin(), block_.outputs().end());

  // 收集具有输出变体的操作的输出寄存器索引
  for (StaticNodeInfo& pnode : nodes_) {
    if (!pnode.has_out_variant()) {
      continue;
    }
    auto outputs = pnode.node()->outputs();
    const auto num_outputs = static_cast<uint32_t>(outputs.size());
    for (const auto i : c10::irange(num_outputs)) {
      const Value* out_v = outputs[i];
      // 检查是否为张量类型
      bool is_tensor_type = out_v->type()->castRaw<TensorType>();
      // 如果允许管理输出张量，并且是张量类型且不属于图的输出，则加入管理集合
      if (opts.manage_output_tensors && is_tensor_type &&
          graph_output_values.find(out_v) == graph_output_values.end() &&
          value_group_.isOutputAlias(out_v)) {
        managed_output_tensor_values_.insert(out_v);
        continue;
      }
      // 如果值总是存活，则跳过
      if (value_group_.isAlwaysAlive(out_v)) {
        continue;
      }
      // 如果是张量类型，则加入管理张量集合；否则，如果是可优化容器类型，则视为泄漏
      if (is_tensor_type) {
        managed_tensor_values_.insert(out_v);
      } else if (node_is_optimizable_container_type(pnode.node())) {
        // 某些容器类型的分配时间较长，因此将其视为泄漏
        leaked_values_.insert(out_v);
      }
    }
  }

  // 处理块的输出值
  for (const Value* output : block_.outputs()) {
    // 从 managed_tensor_values_ 中移除指定的 output 元素
    managed_tensor_values_.erase(output);
  }
  // 打印 managed_tensor_values_ 集合的调试信息
  GRAPH_DEBUG("managed_tensor_values: ", dumpValueSet(managed_tensor_values_));
  // 打印 managed_output_tensor_values_ 集合的调试信息
  GRAPH_DEBUG(
      "managed_output_tensor_values_: ",
      dumpValueSet(managed_output_tensor_values_));

  // 使用 block_, alias_db 和 managed_tensor_values_ 创建 ManagedTensorRanges 对象
  managed_tensor_ranges_ =
      ManagedTensorRanges(block_, alias_db, managed_tensor_values_);
}

// 返回静态模块的选项引用
const StaticModuleOptions& StaticModule::opts() const {
  return opts_;
}

// 返回静态模块输出的数量
size_t StaticModule::num_outputs() const {
  return graph_->outputs().size();
}

// 返回静态模块输入的数量
size_t StaticModule::num_inputs() const {
  return num_inputs_;
}

// 返回静态模块的运行时实例，如果尚未缓存则创建一个新实例
StaticRuntime& StaticModule::runtime() {
  if (!cached_runtime_) {
    cached_runtime_ = std::make_unique<StaticRuntime>(*this);
  }
  return *cached_runtime_;
}

// 在测试中查找具有给定类型的节点，并返回第一个匹配的节点
Node* StaticModule::findNodeWithKindForTesting(const std::string& kind) const {
  for (auto& block_and_info : block_infos_) {
    auto& block_info = block_and_info.second;
    for (auto& pnode : block_info.nodes()) {
      if (pnode.node()->kind().toQualString() == kind) {
        return pnode.node();
      }
    }
  }
  return nullptr;
}

// 调用运行时以执行静态模块，并返回结果
c10::IValue StaticModule::operator()(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
  return runtime()(args, kwargs);
}

// 移动语义版本的调用运行时以执行静态模块，并返回结果
c10::IValue StaticModule::operator()(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
  return runtime()(std::move(args), kwargs);
}

// BlockRunner 构造函数，初始化执行静态模块的块运行器
BlockRunner::BlockRunner(
    const StaticModule& sm,
    IValue* values,
    Block* block,
    torch::jit::TaskLauncher* launcher,
    bool is_root_block)
    : static_module_(sm),
      block_info_(static_module_.block_info(block)),
      is_root_block_(is_root_block),
      first_input_is_self_(
          is_root_block_ && static_module_.first_input_is_self()),
      inputs_begin_(block_info_.block_inputs_idx()),
      // TODO(T108633124): Turn on manage output tensors for sub-blocks.
      manage_output_tensors_enabled_(
          is_root_block_ && sm.opts().manage_output_tensors),
      values_(values) {
  nodes_.reserve(block_info_.nodes().size());

  // 初始化块中的每个节点
  for (auto& pre_pnode : block_info_.nodes()) {
    nodes_.emplace_back(pre_pnode, values_);
  }

  // 初始化块的输出
  for (auto index : block_info_.block_output_indices()) {
    outputs_.emplace_back(&values_[index]);
  }

  // 遍历节点，处理包含子块的节点
  for (auto& pnode : nodes_) {
    auto* node = pnode.node();

    // 将异步任务调度器附加到处理过的节点
    pnode.set_metadata(launcher);
    auto blocks = node->blocks();
    const auto num_blocks = blocks.size();
    if (num_blocks == 0) {
      continue;
    }
    DCHECK(node->kind() == prim::If || node->kind() == prim::Loop);
    std::vector<BlockRunner> block_runners;
    block_runners.reserve(num_blocks);

    // 为每个子块创建一个块运行器
    for (auto* b : blocks) {
      block_runners.emplace_back(sm, values_, b, launcher);
    }
    pnode.set_metadata(std::move(block_runners));
  }
}

// 移动语义的 BlockRunner 构造函数，默认实现
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
BlockRunner::BlockRunner(BlockRunner&&) noexcept = default;

// BlockRunner 析构函数，默认实现
BlockRunner::~BlockRunner() = default;

// 设置指定索引处的参数为移动语义的版本
void BlockRunner::set_arg(const size_t idx, std::vector<IValue>&& args) {
  DCHECK(idx < args.size());
  Input(idx + first_input_is_self_) = std::move(args[idx]);
}

// 设置指定索引处的参数为常规版本
void BlockRunner::set_arg(const size_t idx, const std::vector<IValue>& args) {
  DCHECK(idx < args.size());
  Input(idx + first_input_is_self_) = args[idx];
}
void BlockRunner::set_arg(const size_t idx, const IValue& arg) {
  // 设置给定索引处的输入参数
  Input(idx + first_input_is_self_) = arg;
}

namespace {
void check_type(const Argument& schema_arg, const IValue& arg) {
  // 处理最常见的情况的快速路径
  if (arg.isTensor() &&
      schema_arg.type()->kind() == c10::TypeKind::TensorType) {
    return;
  }
  // 检查参数是否符合预期的类型，否则抛出异常
  TORCH_CHECK(
      arg.type()->isSubtypeOf(schema_arg.type()),
      arg.type()->annotation_str(),
      " is not a subtype of ",
      schema_arg.type()->annotation_str(),
      "; schema arg name: '",
      schema_arg.name(),
      "', ivalue: ",
      iValueToString(arg));
}
} // namespace

template <typename IValueList>
void BlockRunner::set_inputs(IValueList&& args, const KeywordArgs& kwargs) {
  const auto& schema = static_module_.schema();
  if (first_input_is_self_) {
    // 如果需要将自身作为第一个输入参数处理，设置输入为模块的 IValue
    Input(0) = static_module_.module()._ivalue();
  }

  if (!is_root_block_ || C10_UNLIKELY(!schema)) {
    // 对于非根块或者没有模式的情况，进行验证并抛出异常
    TORCH_CHECK(
        kwargs.empty(),
        "BlockRunner got kwargs; is_root_block: ",
        std::to_string(is_root_block),
        "schema: ",
        schema ? schema->name() : "(not available)");

    const auto total_num_inputs = args.size() + first_input_is_self_;
    TORCH_CHECK(
        total_num_inputs == block_info_.num_inputs(),
        "Block runner got ",
        std::to_string(total_num_inputs),
        " inputs; ",
        " first_input_is_self: ",
        std::to_string(first_input_is_self_),
        "; SR block expects ",
        std::to_string(block_info_.num_inputs()),
        " inputs for schema ",
        schema ? schema->name() : "(not available)");

    for (const auto i_arg : c10::irange(args.size())) {
      // 设置每个输入参数的值
      set_arg(i_arg, std::forward<IValueList>(args));
    }
    return;
  }

  const auto& schema_args = schema->arguments();
  size_t consumed_kwargs = 0;
  DCHECK(!schema_args.empty());
  TORCH_CHECK(
      args.size() < schema_args.size(),
      "Static runtime got ",
      std::to_string(args.size()),
      " arguments, expects ",
      std::to_string(schema_args.size() - 1),
      " for schema ",
      schema->name());

  for (const auto i_arg : c10::irange(1, schema_args.size())) {
    // 从 1 开始，因为 schema 总是包含 `self`。
    const auto& schema_arg = schema_args[i_arg];

    if (i_arg - 1 < args.size()) {
      // 如果有足够的位置参数，设置并检查其类型
      check_type(schema_arg, std::forward<IValueList>(args)[i_arg - 1]);
      set_arg(i_arg - 1, std::forward<IValueList>(args));
      continue;
    }

    auto it = kwargs.find(schema_arg.name());
    if (it != kwargs.end()) {
      // 如果找到关键字参数，设置并检查其类型
      check_type(schema_arg, it->second);
      set_arg(i_arg - 1, it->second);
      ++consumed_kwargs;
      continue;
    }

    auto maybe_default_val = schema_arg.default_value();
    if (maybe_default_val) {
      // 如果存在默认值，使用默认值设置参数
      set_arg(i_arg - 1, *maybe_default_val);
      continue;
    }
    // 使用 TORCH_CHECK 来检查条件，若条件为 false，则输出以下错误消息
    TORCH_CHECK(
        false,
        "Static runtime is missing required kwarg ",
        schema_arg.name(),
        " i_arg: ",
        std::to_string(i_arg),
        " for schema ",
        schema->name());
  }
  // 使用 TORCH_CHECK 来检查条件，确保消耗的关键字参数数量等于 kwargs 的大小
  TORCH_CHECK(
      consumed_kwargs == kwargs.size(),
      "kwargs size mismatch (consumed ",
      std::to_string(consumed_kwargs),
      ", expected ",
      std::to_string(kwargs.size()),
      " for schema ",
      schema->name());
}

/// [创建内存规划器]
/// 如果当前没有内存规划器实例，则创建一个标准内存规划器实例。
/// 内存规划器根据给定的参数进行初始化，包括块信息、是否启用输出变体、
/// 是否管理输出张量、是否优化内存。
void BlockRunner::create_memory_planner() {
  if (!planner_) {
    planner_ = std::make_unique<StandardMemoryPlanner>(
        this,
        block_info_,
        static_module_.opts().enable_out_variant,
        manage_output_tensors_enabled_,
        static_module_.opts().optimize_memory);
  }
}

namespace {

/// [销毁节点输出]
/// 根据节点的类型决定如何处理其输出，主要根据是否需要分配堆内存来判断。
/// 对于不需要堆内存分配的输出，不做处理；对于需要堆内存分配的输出，
/// 根据是否借用的情况进行销毁处理。
void destroyNodeOutputs(ProcessedNode& p_node) {
  const auto borrows_outputs = borrowsOutputs(p_node.node()->kind());
  const auto num_outputs = static_cast<uint32_t>(p_node.num_outputs());
  for (const auto i : c10::irange<uint32_t>(num_outputs)) {
    auto& output = p_node.Output(i);
    if (doesNotHeapAllocateWhenStoredInIValue(*output.type())) {
      continue;
    }

    if (borrows_outputs) {
      // 注意：这里不需要增加引用计数。这段代码只有在运行未完成时才会执行，
      // 所以不应该将任何东西返回给客户端。
      c10::MaybeOwnedTraits<IValue>::destroyBorrow(output);
    } else {
      output = IValue();
    }
  }
}

} // namespace

/// [清理中间值]
/// 倒序迭代节点列表，销毁每个节点的输出，确保所有借用的 IValue 都已清理。
void BlockRunner::clean_up_intermediate_ivalues() noexcept {
  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
    destroyNodeOutputs(*it);
  }
}

/// [重置内存]
/// 重置内存规划器，然后清理中间值和输入值。
/// 清理中间值需在输入值之前，以防某些输入是借用的，而静态运行时拥有唯一引用。
void BlockRunner::resetMemory() noexcept {
  planner_.reset();
  clean_up_intermediate_ivalues();
  clean_up_input_ivalues();
}

/// [移动输出到元组]
/// 根据输出数量，将所有输出移动到一个 IValue 元组中，并返回该元组。
c10::IValue BlockRunner::move_outputs_to_tuple(uint32_t num_outputs) {
  switch (num_outputs) {
    case 1:
      return c10::ivalue::Tuple::create(IValue(std::move(*outputs_[0])));
    case 2:
      return c10::ivalue::Tuple::create(
          IValue(std::move(*outputs_[0])), IValue(std::move(*outputs_[1])));
    case 3:
      return c10::ivalue::Tuple::create(
          IValue(std::move(*outputs_[0])),
          IValue(std::move(*outputs_[1])),
          IValue(std::move(*outputs_[2])));
    default: {
      std::vector<c10::IValue> outputs;
      outputs.reserve(num_outputs);
      for (const auto i : c10::irange(num_outputs)) {
        // 在这里使用 move。否则，需要显式清理 outputs_[i]
        outputs.emplace_back(std::move(*outputs_[i]));
      }
      return c10::ivalue::Tuple::create(std::move(outputs));
    }
  }
}

/// [检查并修正运行时的错误模式别名信息]
/// 静态运行时依赖操作符模式的别名信息来进行内存规划。由于很难强制要求别名信息正确，
/// 我们需要在运行时检测不符合模式的意外别名。仅有管理的张量别名会有问题。
/// 为避免运行时崩溃，我们可以添加运行时检测并强制操作符符合规范。
# 根据其模式，通过克隆别名来管理张量。因为所有托管张量的数据指针
# 都属于内存规划器分配的内部缓冲区的一部分，我们可以通过检查
# 内存重叠来检查别名。但是在推断期间，张量的存储可以重新调整大小，
# 因此我们需要另一种处理重新调整大小情况的方法。

# 错误的模式可能会破坏内存规划的两种方式。让我们看两个例子：

# 示例 1:
# @code
#   def forward(x):
#     a = x + x
#     b = bad_op(a)  # b最终错误地引用了a
#     return (b)
# @endcode
# bad_op: 它的模式表明它返回一个新的张量，但实际上返回一个别名。
# 在这种情况下，内存规划器会将`a`识别为托管张量，并在返回`b`之前清理其内存。
# 但实际上，`b`是`a`的别名，当`a`的数据指针被重置时，`b`的数据指针也被重置。

# 示例 2:
# @code
#   def forward(x):
#     a = x + x
#     a2 = bad_op(a) # a2错误地引用了a
#     b = a + a
#     c = b * b # c与a共享存储
#     d = c + 2 # d与b共享存储
#     e = a2 * a2
#     return (d, e)
# @endcode
# 使用内存重用算法，`c`可能会与`a`共享存储，但由于bad_op，`a2`现在别名为`a`。
# `c`覆盖了`a`和`a2`，导致错误的结果。我们通过两个步骤解决这个问题。
# 注意，当前的内存重用算法不会出现这种情况，因为它的实现方式不同。使用
# 不同的实现方式可能会改变这一点。

# 第1步，在ProcessedNodes上注释一个名为`check_memory_overlap_`的标志，
# 如果其输出不与其输入别名相符，正如AliasDb所示，并且其所有输出都是张量。
# 然后在运行时，我们检查节点的输出张量是否与内存规划器分配的内部缓冲区重叠。
# 出于延迟考虑，我们仅对后备操作运行此检查。原生操作和变体的模式通过静态
# 运行时单元测试进行验证和执行。对于第一次迭代，我们使用
# ProcessedNode::verify_and_correct_memory_overlap()进行完整的内存重叠检查，
# 因为内部缓冲区尚不存在。

# 第2步，如果在推断期间托管的张量被调整大小，则会获得一个不来自缓冲区的新数据指针。
# 我们可以通过推迟托管张量的释放时间（实质上将内部/输出缓冲区合并为一个）来处理
# 这种情况。在实施合并之前，我们向节点添加另一个标志`overlap_detected_`，以标记
# 任何在第1步检测到重叠的节点，并且如果快速检查（通过检查与内部缓冲区的内存重叠）
# 失败，则进行完整的内存重叠检查。即使添加了标志，仍然存在一个边缘情况会失败。
/// 如果在触发调整大小的同时，操作创建了一个别名，当前的检查可能无法检测到这个别名。
void BlockRunner::verify_and_correct_memory_overlap(ProcessedNode& n) {
  // 当内部/输出缓冲区合并后，可以移除这个慢速检查
  if (C10_UNLIKELY(n.check_outputs_for_memory_overlap())) {
    if (C10_UNLIKELY(!planner_)) {
      // 慢速检查，仅在第一次迭代时执行
      n.verify_and_correct_memory_overlap();
    } else {
      bool overlap_detected_with_fast_check = false;
      const auto n_outputs = static_cast<uint32_t>(n.outputs().size());
      for (auto i : c10::irange(n_outputs)) {
        auto& output = n.Output(i);
        if (output.isTensor()) {
          overlap_detected_with_fast_check |=
              fast_check_and_correct_overlap_with(n, output);
        } else if (output.isTensorList()) {
          auto tensor_list = output.toListRef();
          for (auto& ival : tensor_list) {
            overlap_detected_with_fast_check |=
                fast_check_and_correct_overlap_with(
                    n,
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                    const_cast<c10::IValue&>(ival));
          }
        }
      }
      if (n.outputs_memory_overlap_detected() &&
          !overlap_detected_with_fast_check) {
        // 慢速检查，仅在快速检查失败时运行
        n.verify_and_correct_memory_overlap();
      }
    }
  }
}

bool BlockRunner::fast_check_and_correct_overlap_with(
    ProcessedNode& n,
    c10::IValue& tensor_ival) {
  auto& tensor = tensor_ival.toTensor();
  // 如果与内部缓冲区存在重叠，则进行处理
  if (planner_->overlapWithInternalBuffer(tensor.data_ptr())) {
    DLOG(INFO) << "Detected alias for node: " << PrintNode(n.node());
    // 克隆张量以避免别名问题
    tensor_ival = at::native::clone(tensor, c10::nullopt);
    n.set_outputs_memory_overlap_detected();
    return true;
  }
  return false;
}

BlockRunner::Deallocator::~Deallocator() {
  // 假设清理过程不会抛出异常
  cleanupImpl();
#ifndef NDEBUG
  // 在调试模式下，检查是否存在内存泄漏
  block_runner_.check_for_memory_leak(/*output_returned*/ false);
#endif
}

void BlockRunner::Deallocator::cleanupImpl() {
  // 在第一次运行`run()`后创建`MemoryPlanner`，因为`MemoryPlanner`使用前一次`run()`的张量大小进行内存规划
  if (C10_LIKELY(finished_)) {
    block_runner_.create_memory_planner();
  }

  if (C10_LIKELY(block_runner_.planner_)) {
    // 使用`MemoryPlanner`释放内存
    block_runner_.planner_->deallocate();
  } else {
    // 这是第一次运行，且尚未完成，因此无法使用`MemoryPlanner`释放资源，手动重置所有内容
    block_runner_.resetMemory();
  }
  // 清理输入张量的所有拥有引用
  block_runner_.clean_up_input_ivalues();
  if (C10_UNLIKELY(!finished_)) {
    // 如果尚未完成，则释放输出张量
    block_runner_.deallocateOutputTensors();
  }
}
    const KeywordArgs& kwargs) {
  // 假设推断工作负载，因此不需要自动微分。
  // 在调度器上启用此选项显著减少了分派开销，因为它避免了至少某些函数（如 resize_ 和 resize_as_）的一轮分派。
  // 进入推断模式，禁用自动微分
  c10::InferenceMode mode;

  {
    // 在离开此作用域时，执行 Deallocator 的析构函数，用于资源清理
    auto on_exit = Deallocator(*this);

    // 如果存在 planner，则分配资源并检查输出张量的内存泄漏
    if (planner_) {
      DCHECK(!manage_output_tensors_enabled_ || checkOutputTensorMemoryLeaks());
      planner_->allocate();
    }

    // 设置输入参数
    set_inputs(std::forward<IValueList>(args), kwargs);

    // 遍历所有节点并运行
    for (auto& n : nodes_) {
      // 记录运行的节点信息（注释掉的日志语句）
      // LOG(INFO) << "Running node: " << PrintNode(n.node());
      // 运行节点 n
      n.run();
      // 检查并修正内存重叠
      verify_and_correct_memory_overlap(n);
    }
    // 在退出作用域之前标记 Deallocator 结束
    on_exit.setFinished();
  }

  // 如果输出数量大于 1，则将输出移动到元组中返回
  if (block_info_.num_outputs() > 1) {
    return move_outputs_to_tuple(block_info_.num_outputs());
  }

  // 检查是否存在内存泄漏，参数表明是否有输出返回
  DCHECK(check_for_memory_leak(/*output_returned*/ false));

  // 使用 move 操作移动输出，避免显式清理 outputs_[0]
  return std::move(*outputs_[0]);
}
template <typename IValueList>
c10::IValue BlockRunner::run_impl_record_functions(
    IValueList&& args,
    const KeywordArgs& kwargs) {
  // 获取静态运行时模型下的步骤回调函数列表
  auto step_callbacks =
      at::getStepCallbacksUnlessEmpty(at::RecordScope::STATIC_RUNTIME_MODEL);
  // 如果存在步骤回调函数
  if (C10_UNLIKELY(step_callbacks.has_value())) {
    // 使用步骤回调函数创建 RecordFunction 守卫
    at::RecordFunction guard(std::move(*step_callbacks));
    // 断言守卫当前处于活动状态
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(guard.isActive());
    // 如果守卫需要输入参数，则在执行 forward 操作前记录输入
    guard.needsInputs()
        ? guard.before(
              "forward", c10::ArrayRef<const IValue>(args.data(), args.size()))
        : guard.before("forward");

    // 调用 run_impl 函数执行实际的计算并返回结果
    return run_impl(std::forward<IValueList>(args), kwargs);
  }
  // 如果不存在步骤回调函数，则直接调用 run_impl 函数执行计算并返回结果
  return run_impl(std::forward<IValueList>(args), kwargs);
}

template <typename IValueList>
c10::intrusive_ptr<c10::ivalue::Future> BlockRunner::run_impl_async(
    IValueList&& args,
    const KeywordArgs& kwargs) {
  // 在调用线程中直接运行图形。异步操作将在附加到 ProcessedNodes 元数据的 taskLauncher 上执行
  c10::IValue output = run_impl(std::forward<IValueList>(args), kwargs);

  // 如果输出是 Future 类型，则返回它
  if (output.isFuture()) {
    return output.toFuture();
  }

  // 否则，将输出包装为 Future，标记为已完成并返回
  TypePtr return_type;
  if (block_info_.num_outputs() > 1) {
    return_type = TupleType::create(
        fmap(outputs(), [](const IValue* v) { return v->type(); }));
  } else {
    return_type = outputs().at(0)->type();
  }
  c10::intrusive_ptr<Future> future = c10::make_intrusive<Future>(return_type);
  future->markCompleted(output);
  return future;
}

template <typename IValueList>
c10::intrusive_ptr<c10::ivalue::Future> BlockRunner::
    run_impl_record_functions_async(
        IValueList&& args,
        const KeywordArgs& kwargs) {
  // 获取静态运行时模型下的步骤回调函数列表
  auto step_callbacks =
      at::getStepCallbacksUnlessEmpty(at::RecordScope::STATIC_RUNTIME_MODEL);
  // 如果存在步骤回调函数
  if (C10_UNLIKELY(step_callbacks.has_value())) {
    // 使用步骤回调函数创建 RecordFunction 守卫
    at::RecordFunction guard(std::move(*step_callbacks));
    // 断言守卫当前处于活动状态
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(guard.isActive());
    // 如果守卫需要输入参数，则在执行 forward 操作前记录输入
    guard.needsInputs()
        ? guard.before(
              "forward", c10::ArrayRef<const IValue>(args.data(), args.size()))
        : guard.before("forward");

    // 调用 run_impl_async 函数执行实际的异步计算并返回 Future 结果
    return run_impl_async(std::forward<IValueList>(args), kwargs);
  }
  // 如果不存在步骤回调函数，则直接调用 run_impl_async 函数执行异步计算并返回 Future 结果
  return run_impl_async(std::forward<IValueList>(args), kwargs);
}

c10::IValue BlockRunner::operator()(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  // 如果禁用了网络分析，则调用普通的运行函数 run_impl
  return run_impl(args, kwargs);
#else
  // 否则调用带有记录函数的运行函数 run_impl_record_functions
  return run_impl_record_functions(args, kwargs);
#endif
}

c10::IValue BlockRunner::operator()(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  // 如果禁用了网络分析，则调用普通的运行函数 run_impl
  return run_impl(std::move(args), kwargs);
#else
  // 否则调用带有记录函数的运行函数 run_impl_record_functions
  return run_impl_record_functions(std::move(args), kwargs);
#endif
}

c10::intrusive_ptr<c10::ivalue::Future> BlockRunner::runAsync(
    const std::vector<c10::IValue>& args,
    # 函数定义的参数列表开始，这里接收一个名为kwargs的关键字参数对象
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  // 如果定义了 PYTORCH_DISABLE_NET_PROFILING 宏，则调用异步执行函数 run_impl_async
  return run_impl_async(args, kwargs);
#else
  // 否则调用带函数记录的异步执行函数 run_impl_record_functions_async
  return run_impl_record_functions_async(args, kwargs);
#endif
}

c10::intrusive_ptr<c10::ivalue::Future> BlockRunner::runAsync(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  // 如果定义了 PYTORCH_DISABLE_NET_PROFILING 宏，则调用异步执行函数 run_impl_async
  return run_impl_async(std::move(args), kwargs);
#else
  // 否则调用带函数记录的异步执行函数 run_impl_record_functions_async
  return run_impl_record_functions_async(std::move(args), kwargs);
#endif
}

namespace {

std::string generate_latency_json(const std::string& label, double millis) {
#ifdef FBCODE_CAFFE2
  // 如果定义了 FBCODE_CAFFE2 宏，则生成一个包含延迟信息的 JSON 字符串
  folly::dynamic json = folly::dynamic::object();
  json["type"] = label;
  json["metric"] = "latency";
  json["unit"] = "ms";
  json["value"] = millis;
  return "PyTorchObserver " + folly::toJson(json);
#else
  // 否则返回空字符串
  (void)label;
  (void)millis;
  return "";
#endif
}

} // namespace

void BlockRunner::benchmark(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const uint32_t warmup_runs,
    const uint32_t main_runs,
    bool print_per_node_time,
    bool generate_ai_pep_output) {
  // 检查参数列表是否为空或参数列表大小与关键字参数列表大小相同
  TORCH_CHECK(kwargs_list.empty() || args_list.size() == kwargs_list.size());
  // 输出输入大小信息
  std::cout << "Input size: " << args_list.size() << std::endl;
  // 运行基准测试，并返回每次迭代的运行时间
  float time_per_iter =
      benchmark_model(args_list, kwargs_list, warmup_runs, main_runs);
  // 输出静态运行时每次迭代的毫秒数和每秒迭代次数
  std::cout << "Static runtime ms per iter: " << time_per_iter
            << ". Iters per second: " << 1000.0 / time_per_iter << std::endl;

  // 分别对各操作进行基准测试，并返回单个操作的度量结果
  IndividualMetrics results =
      benchmark_individual_ops(args_list, kwargs_list, warmup_runs, main_runs);

  // 如果需要打印每个节点的运行时间
  if (print_per_node_time) {
    // 获取节点数量
    const auto num_nodes = static_cast<uint32_t>(nodes_.size());
    // 对每个节点进行迭代
    for (const auto i : c10::irange(num_nodes)) {
      const Node* node = nodes_[i].node();
      // 输出节点编号、每次迭代的时间和节点信息
      std::cout << "Node #" << i << ": " << results.time_per_node[i]
                << " ms/iter, ";
      node->print(std::cout, 0, nullptr, false);
    }
  }

  // 将操作类型和对应的运行时间存储为向量
  std::vector<std::pair<std::string, double>> time_per_node_type_vec{
      results.time_per_node_type.begin(), results.time_per_node_type.end()};
  // 如果参数列表为空，则按节点类型实例数排序
  if (args_list.empty()) {
    std::sort(
        time_per_node_type_vec.begin(),
        time_per_node_type_vec.end(),
        [&results](auto& left, auto& right) {
          return results.instances_per_node_type[left.first] >
              results.instances_per_node_type[right.first];
        });
  } else {
    // 否则按运行时间排序
    std::sort(
        time_per_node_type_vec.begin(),
        time_per_node_type_vec.end(),
        [](auto& left, auto& right) { return left.second > right.second; });
  }
  // 输出各节点类型的运行时间
  std::cout << "Time per node type:" << std::endl;
  for (const auto& p : time_per_node_type_vec) {
    const std::string& kind = p.first;
    const double ms = p.second;
    std::cout << std::setw(15) << ms << " ms. " << std::setw(10)
              << results.percent_per_node_type[kind] << "%. " << kind << " ("
              << results.instances_per_node_type[kind] << " nodes";
    // 检查结果中是否包含指定类型的节点数量，并输出相应的信息
    if (results.out_nodes.count(kind)) {
      std::cout << ", out variant)" << std::endl;
    } else if (results.native_nodes.count(kind)) {
      std::cout << ", native)" << std::endl;
    } else {
      std::cout << ")" << std::endl;
    }

    // 如果需要生成 AI PEP 输出，则记录生成的延迟 JSON 数据
    if (generate_ai_pep_output) {
      LOG(INFO) << generate_latency_json(kind, ms);
    }
  }

  // 如果需要生成 AI PEP 输出，则记录首次迭代的运行时间
  if (generate_ai_pep_output) {
    LOG(INFO) << generate_latency_json(
        "static_runtime_first_iter", results.first_iter_time);
  }

  // 输出总运行时间
  std::cout << std::setw(15) << results.total_time << " ms. in Total"
            << std::endl;
  // 输出 BlockRunner 设置时间
  std::cout << "BlockRunner setup time: " << results.setup_time << " ms"
            << std::endl;
  // 输出内存分配时间
  std::cout << "Memory allocation time: " << results.memory_alloc_time
            << " ms\n";
  // 输出内存释放时间
  std::cout << "Memory deallocation time: " << results.memory_dealloc_time
            << " ms" << std::endl;
  // 输出输出数据释放时间
  std::cout << "Outputs deallocation time: " << results.output_dealloc_time
            << " ms" << std::endl;
  // 输出首次迭代时间
  std::cout << "First iter time: " << results.first_iter_time << " ms"
            << std::endl;
  // 输出操作符节点数量
  std::cout << "Number of operators: " << nodes_.size() << std::endl;

  // 如果存在 Planner 对象，则输出关于内存管理的统计信息
  if (planner_) {
    std::cout << "Total number of managed tensors: "
              << planner_->total_num_managed_tensors() << std::endl;
    std::cout << "Total number of managed output tensors: "
              << planner_->total_num_managed_output_tensors() << std::endl;
    std::cout << "Total number of unmanaged values: "
              << planner_->total_num_unmanaged() << std::endl;
    std::cout << "Number of unmanaged values requiring cleanup: "
              << planner_->num_unmanaged_non_scalars() << std::endl;
    std::cout << "Number of unmanaged values not requiring cleanup: "
              << planner_->num_unmanaged_scalars() << std::endl;
    std::cout << "Total memory managed: " << planner_->total_managed()
              << " bytes" << std::endl;
    // 如果静态模块启用了内存优化，则输出重用张量的统计信息
    if (static_module_.opts().optimize_memory) {
      std::cout << "Total number of reused tensors: "
                << planner_->total_reused_tensors() << std::endl;
    }
  }

  // 计算不支持的节点数量
  auto unsupported_nodes_count = results.total_nodes_count -
      results.out_nodes_count - results.native_nodes.size();
  // 输出 'out' 变体节点与总节点数的比例
  std::cout << "Total number of 'out' variant nodes/total number of nodes: "
            << results.out_nodes_count << "/" << results.total_nodes_count
            << " ("
            << 100.0 * static_cast<float>(results.out_nodes_count) /
          static_cast<float>(results.total_nodes_count)
            << "%)" << std::endl;
  // 输出未被 SR（某种资源）覆盖的节点与总节点数的比例
  std::cout << "Total number of nodes not covered by SR/total number of nodes: "
            << unsupported_nodes_count << "/" << results.total_nodes_count
            << " ("
            << 100.0 * static_cast<float>(unsupported_nodes_count) /
          static_cast<float>(results.total_nodes_count)
            << "%)" << std::endl;

  // 检查是否有内存泄漏
  check_for_memory_leak();
#ifndef NDEBUG
  // 如果处于调试模式，创建一个空的关键字参数对象
  KeywordArgs empty_kwargs;
  // 调用 display_nodes 函数，显示第一个参数列表中的节点，
  // 如果有关键字参数列表，则传递第一个关键字参数列表，否则传递空的关键字参数对象
  display_nodes(
      args_list[0], kwargs_list.size() > 0 ? kwargs_list[0] : empty_kwargs);
#endif
}

// 对模型进行基准测试，返回每个运行实例的平均时间（毫秒）
float BlockRunner::benchmark_model(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const unsigned int warmup_runs,
    const unsigned int main_runs) {
  // 确保主要运行次数至少为1
  TORCH_CHECK(main_runs >= 1);
  // 如果有关键字参数列表，则确保参数列表和关键字参数列表长度相等
  TORCH_CHECK(kwargs_list.empty() || args_list.size() == kwargs_list.size());

  // 检查关键字参数列表是否为空
  const bool is_kwargs_empty = kwargs_list.empty();
  // 创建一个空的关键字参数对象
  const KeywordArgs empty_kwargs;
  
  // 预热运行阶段，执行指定次数的预热运行
  for (const auto _n_run : c10::irange(warmup_runs)) {
    (void)_n_run; // 抑制未使用变量警告
    // 获取参数列表的长度
    const auto num_args = static_cast<uint32_t>(args_list.size());
    // 遍历参数列表
    for (const auto j : c10::irange(num_args)) {
      // 调用 operator() 方法处理参数列表中的参数，
      // 如果关键字参数列表为空，则传递空的关键字参数对象，否则传递对应的关键字参数
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      // 如果启用了管理输出张量，则释放输出张量
      if (manage_output_tensors_enabled_) {
        deallocateOutputTensors();
      }
    }
  }

  // 正式运行阶段，执行主要运行次数的运行
  caffe2::Timer timer;
  for (const auto _n_run : c10::irange(main_runs)) {
    (void)_n_run; // 抑制未使用变量警告
    // 获取参数列表的长度
    const auto num_args = static_cast<uint32_t>(args_list.size());
    // 遍历参数列表
    for (const auto j : c10::irange(num_args)) {
      // 调用 operator() 方法处理参数列表中的参数，
      // 如果关键字参数列表为空，则传递空的关键字参数对象，否则传递对应的关键字参数
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      // 如果启用了管理输出张量，则释放输出张量
      if (manage_output_tensors_enabled_) {
        deallocateOutputTensors();
      }
    }
  }

  // 计算总运行时间（毫秒）
  float millis = timer.MilliSeconds();
  // 返回每个运行实例的平均时间
  return millis /
      (static_cast<float>(main_runs) * static_cast<float>(args_list.size()));
}

// 显示 IValue 类型对象的信息
static bool display_ivalue(const IValue& iv) {
  // 如果是张量类型，显示张量的维度信息
  if (iv.isTensor()) {
    std::cout << "Tensor " << iv.toTensor().toString() << " {";
    const auto dims = iv.toTensor().sizes();
    const auto n_dims = static_cast<uint32_t>(dims.size());
    for (const auto i : c10::irange(n_dims)) {
      std::cout << iv.toTensor().sizes()[i];
      if (n_dims > i + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "}\n";
    return true;
  }
  // 如果是张量列表类型，显示张量列表的大小
  else if (iv.isTensorList()) {
    std::cout << "TensorList {" << iv.toTensorList().size() << "}\n";
    return true;
  }
  // 如果是通用字典类型，显示字典的大小
  else if (iv.isGenericDict()) {
    std::cout << "Dict {" << iv.toGenericDict().size() << "}\n";
    return true;
  }
  // 如果是元组类型，显示元组的大小
  else if (iv.isTuple()) {
    std::cout << "Tuple {" << iv.toTupleRef().elements().size() << "}\n";
    return true;
  }
  // 如果是整数类型，显示整数的值
  else if (iv.isInt()) {
    std::cout << "int {" << iv.toInt() << "}\n";
    return true;
  }
  // 如果是布尔类型，显示布尔值
  else if (iv.isBool()) {
    std::cout << "bool {" << iv.toBool() << "}\n";
    return true;
  }
  // 如果是双精度浮点数类型，显示浮点数的值
  else if (iv.isDouble()) {
    std::cout << "double {" << iv.toDouble() << "}\n";
    return true;
  }
  // 如果无法识别类型，返回 false
  return false;
}

// 显示处理节点的信息
static void display_pnode_info(const ProcessedNode& pnode) {
  // 打印节点的信息到标准输出
  pnode.node()->print(std::cout, 0, nullptr, false);
  // 获取节点的输入数量
  const auto num_inputs = static_cast<uint32_t>(pnode.num_inputs());
  // 遍历节点的每个输入
  for (const auto i : c10::irange(num_inputs)) {
    // 打印输入的索引和信息
    std::cout << "\ti" << i << ": ";
    // 遍历输入节点的所有输入索引
    if (!display_ivalue(pnode.Input(i))) {
      // 如果无法显示当前输入节点的值，打印其类型信息
      std::cout << *(pnode.node()->inputs()[i]->type()) << '\n';
    }
  }
  // 获取节点的所有输出
  const auto outputs = pnode.outputs();
  // 获取输出的数量
  const auto num_outputs = static_cast<uint32_t>(outputs.size());
  // 遍历输出节点的所有输出索引
  for (const auto i : c10::irange(num_outputs)) {
    // 打印输出节点的索引号
    std::cout << "\to" << i << ": ";
    // 如果无法显示当前输出节点的值，打印其类型信息
    if (!display_ivalue(outputs[i])) {
      std::cout << *(pnode.node()->outputs()[i]->type()) << '\n';
    }
  }
}

void BlockRunner::display_nodes(
    const std::vector<c10::IValue>& args,  // 接收参数列表作为输入
    const KeywordArgs& kwargs) {  // 接收关键字参数作为输入
  c10::InferenceMode mode;  // 进入推断模式

  auto on_exit = Deallocator(*this);  // 创建 Deallocator 对象用于资源管理

  if (planner_) {  // 如果有 planner 对象
    planner_->allocate();  // 调用其 allocate 方法进行资源分配
  }
  set_inputs(args, kwargs);  // 设置输入参数

  for (auto& node : nodes_) {  // 遍历节点列表
    node.run();  // 运行当前节点
    display_pnode_info(node);  // 显示当前节点的信息
  }
  on_exit.setFinished();  // 设置 Deallocator 完成标志
}

BlockRunner::IndividualMetrics BlockRunner::benchmark_individual_ops(
    const std::vector<std::vector<c10::IValue>>& args_list,  // 接收多个参数列表作为输入
    const std::vector<KeywordArgs>& kwargs_list,  // 接收多个关键字参数列表作为输入
    const uint32_t warmup_runs,  // 指定预热运行次数
    const uint32_t main_runs) {  // 指定主要运行次数
  TORCH_CHECK(kwargs_list.empty() || args_list.size() == kwargs_list.size());  // 检查参数有效性
  TORCH_CHECK(warmup_runs >= 1 && main_runs >= 1);  // 检查运行次数有效性

  IndividualMetrics results;  // 创建结果结构体
  results.time_per_node.resize(nodes_.size(), 0);  // 调整节点时间数组大小并初始化为零
  if (args_list.empty()) {  // 如果参数列表为空
    // 当输入为空时，从给定图形中计算操作统计数据，而不执行它。
    const auto num_nodes = static_cast<uint32_t>(nodes_.size());  // 获取节点数量
    for (const auto i : c10::irange(num_nodes)) {  // 遍历节点序号
      const Node* node = nodes_[i].node();  // 获取节点指针
      std::string kind(node->kind().toQualString());  // 获取节点类型字符串
      // TODO: 从子块收集操作统计数据
      results.time_per_node[i] = 0;  // 初始化节点时间为零
      results.time_per_node_type[kind] = 0;  // 初始化节点类型时间为零
      results.instances_per_node_type[kind]++;  // 增加节点类型实例计数
      if (nodes_[i].has_out_variant()) {  // 如果节点有输出变体
        results.out_nodes.insert(kind);  // 将节点类型添加到输出节点集合
        results.out_nodes_count++;  // 增加输出节点计数
      } else if (nodes_[i].has_native()) {  // 如果节点有本地变体
        results.native_nodes.insert(kind);  // 将节点类型添加到本地节点集合
      }
      results.total_time += results.time_per_node[i];  // 增加总时间
    }
    results.total_nodes_count = nodes_.size();  // 设置总节点数量
    results.memory_alloc_time = 0;  // 设置内存分配时间为零
    results.memory_dealloc_time = 0;  // 设置内存释放时间为零
    results.output_dealloc_time = 0;  // 设置输出释放时间为零
    for (const auto& p : results.time_per_node_type) {  // 遍历节点类型时间映射
      const std::string& kind = p.first;  // 获取节点类型
      results.percent_per_node_type[kind] = 0;  // 设置节点类型百分比为零
    }
    return results;  // 返回结果结构体
  }

  const bool is_kwargs_empty = kwargs_list.empty();  // 检查关键字参数列表是否为空
  const KeywordArgs empty_kwargs;  // 创建空关键字参数对象
  bool manage_output_tensors = static_module_.opts().manage_output_tensors;  // 管理输出张量标志位
  // 查看上面 InferenceMode 的使用说明。
  c10::InferenceMode mode;  // 进入推断模式

  // 设置时间计时器
  caffe2::Timer timer;

  set_inputs(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);  // 设置输入参数

  results.setup_time = timer.MilliSeconds();  // 记录设置时间

  // 第一次迭代对每个节点的输出张量大小进行分析，并使用分析信息初始化内存规划器。
  // 后续迭代直接使用已建立的内存规划。
  timer.Start();  // 启动计时器
  operator()(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);  // 执行操作
  if (manage_output_tensors) {  // 如果需要管理输出张量
    deallocateOutputTensors();  // 释放输出张量
  }
  results.first_iter_time = timer.MilliSeconds();  // 记录第一次迭代时间

  // 预热运行
  for (const auto _n_run : c10::irange(warmup_runs)) {  // 进行预热运行
    (void)_n_run;  // 抑制未使用变量警告
    const auto num_args = static_cast<uint32_t>(args_list.size());  // 获取参数列表数量
    // 对于每个参数 j 在 [0, num_args) 范围内循环执行以下操作
    for (const auto j : c10::irange(num_args)) {
      // 调用 operator() 处理参数 args_list[j] 和 kwargs_list[j]
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      // 如果需要管理输出张量，则释放输出张量
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
    }
  }

  // 对于每个主运行次数 i 在 [0, main_runs) 范围内循环执行以下操作
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // 抑制未使用变量警告
    // 获取参数列表的大小，并转换为 uint32_t 类型
    const auto num_args = static_cast<uint32_t>(args_list.size());
    // 对于每个参数 j 在 [0, num_args) 范围内循环执行以下操作
    for (const auto j : c10::irange(num_args)) {
      // 设置输入参数 args_list[j] 和 kwargs_list[j]
      set_inputs(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);

      // 启动计时器
      timer.Start();
      // 如果有 planner_ 对象，则进行内存分配
      if (planner_) {
        planner_->allocate();
      }
      // 计算经过的时间（毫秒）
      float millis = timer.MilliSeconds();
      // 将内存分配时间添加到结果中
      results.memory_alloc_time += millis;

      // 获取节点数，并转换为 uint32_t 类型
      const auto num_nodes = static_cast<uint32_t>(nodes_.size());
      // 对于每个节点 k 在 [0, num_nodes) 范围内循环执行以下操作
      for (const auto k : c10::irange<uint32_t>(num_nodes)) {
        // 启动计时器
        timer.Start();
        // 运行节点 nodes_[k]
        nodes_[k].run();
        // 计算节点运行时间（毫秒）并添加到结果中
        millis = timer.MilliSeconds();
        results.time_per_node[k] += millis;
        // 验证和纠正节点内存重叠
        verify_and_correct_memory_overlap(nodes_[k]);
      }

      // 启动计时器
      timer.Start();
      // 创建内存规划器
      create_memory_planner();
      // 如果有 planner_ 对象，则进行内存释放
      planner_->deallocate();
      // 清理输入张量的所有权引用
      clean_up_input_ivalues();
      // 如果需要管理输出张量，则释放输出张量
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
      // 计算内存释放时间（毫秒）并添加到结果中
      millis = timer.MilliSeconds();
      results.memory_dealloc_time += millis;

      // 启动计时器
      timer.Start();
      // 不再需要在静态运行时中保留输出引用
      c10::IValue output;
      // 如果 static_module_ 的输出数量大于 1，则将输出移动到元组中
      if (static_module_.num_outputs() > 1) {
        output = move_outputs_to_tuple(static_module_.num_outputs());
      }
      // 检查是否存在内存泄漏（输出不返回）
      DCHECK(check_for_memory_leak(/*output_returned*/ false));

      // 使用 std::move 将 outputs_[0] 的值移动到 output 中
      output = std::move(*outputs_[0]);
      // 显式释放输出以测量所需时间
      output = IValue();
      // 计算输出释放时间（毫秒）并添加到结果中
      millis = timer.MilliSeconds();
      results.output_dealloc_time += millis;
    }
  }

  // 后处理
  // 计算总迭代次数
  const float num_total_iters =
      (static_cast<float>(main_runs) * static_cast<float>(args_list.size()));
  // 获取节点数，并转换为 uint32_t 类型
  const auto num_nodes = static_cast<uint32_t>(nodes_.size());
  // 对于每个节点 i 在 [0, num_nodes) 范围内循环执行以下操作
  for (const auto i : c10::irange(num_nodes)) {
    // 获取节点的信息
    const Node* node = nodes_[i].node();
    // 获取节点类型的限定字符串
    std::string kind = std::string(node->kind().toQualString());
    // 计算每个节点类型的平均运行时间并添加到结果中
    results.time_per_node[i] /= num_total_iters;
    results.time_per_node_type[kind] += results.time_per_node[i];
    // 统计每个节点类型的实例数
    results.instances_per_node_type[kind]++;
    // 如果节点具有输出变体，则添加到输出节点集合中
    if (nodes_[i].has_out_variant()) {
      results.out_nodes.insert(kind);
      results.out_nodes_count++;
    } else if (nodes_[i].has_native()) {
      // 如果节点是本地节点，则添加到本地节点集合中
      results.native_nodes.insert(kind);
    }
    // 计算总时间并添加到结果中
    results.total_time += results.time_per_node[i];
  }
  // 计算总节点数
  results.total_nodes_count = nodes_.size();
  // 计算内存分配时间的平均值并添加到结果中
  results.memory_alloc_time /= num_total_iters;
  // 计算内存释放时间的平均值并添加到结果中
  results.memory_dealloc_time /= num_total_iters;
  // 计算输出释放时间的平均值并添加到结果中
  results.output_dealloc_time /= num_total_iters;
  // 对于每个节点类型的时间统计，添加到结果中
  for (const auto& p : results.time_per_node_type) {
    const std::string& kind = p.first;
    # 将每种节点类型的百分比存入结果对象中，计算方法为该节点类型的时间占总时间的百分比
    results.percent_per_node_type[kind] = p.second / results.total_time * 100;
  }
  # 返回计算得到的结果对象
  return results;
}
void BlockRunner::deallocateOutputTensors() {
    // 如果不管理输出张量，直接返回
    if (!static_module_.opts().manage_output_tensors) {
        // 检查是否存在未清理的输出张量缓冲区
        TORCH_CHECK(
            !planner_ || planner_->numOutputBufferBytes() == 0,
            "manage_output_tensors is disabled, but output tensor buffer is not empty.");
        return;
    }

    // 如果有规划器，释放输出张量并检查内存泄漏
    if (planner_) {
        planner_->deallocateOutputTensors();
        DCHECK(checkOutputTensorMemoryLeaks());
    }
}

bool BlockRunner::checkOutputTensorMemoryLeaks() {
    // 如果不管理输出张量或者没有规划器，则返回 true
    if (!static_module_.opts().manage_output_tensors || !planner_) {
        return true;
    }

    // 获取节点数并遍历每个节点的输出
    const auto num_nodes = static_cast<uint32_t>(nodes_.size());
    for (const auto n : c10::irange(num_nodes)) {
        auto& pnode = nodes_[n];
        const auto num_outputs = static_cast<uint32_t>(pnode.num_outputs());
        for (const auto i : c10::irange(num_outputs)) {
            const IValue* ival = &pnode.Output(i);
            const Value* val = pnode.node()->output(i);

            // 如果值不是管理的输出张量或者不是张量类型，则跳过
            if (!isManagedOutputTensorValue(val) || !ival->isTensor()) {
                // 如果值被操作（例如 to_maybe_copy_out）管理，ival 不能是张量
                // 详见 ReplaceWithMaybeCopy 的说明
                continue;
            }

            // 获取 IValue 中的张量，并检查其是否定义
            const auto& t = ival->toTensor();
            if (t.defined()) {
                auto* storage_impl = t.storage().unsafeGetStorageImpl();
                // 构造错误信息，检查存储实现是否为空
                const std::string error_msg = "Output " + std::to_string(i) + ", %" +
                    val->debugName() + " of node " + std::to_string(n) +
                    " was not cleaned up";
                TORCH_CHECK(storage_impl->data() == nullptr, error_msg);
            }
        }
    }
    // 输出日志，表示完成输出张量的内存泄漏检查
    VLOG(1) << "Finished checking for memory leak from output tensors";
    return true;
}

bool BlockRunner::isManagedOutputTensor(const IValue& ivalue) const {
    // 检查是否有规划器，并委托规划器判断是否管理输出张量
    return planner_ && planner_->isManagedOutputTensor(ivalue);
}

bool BlockRunner::isManagedOutputTensorValue(const Value* value) const {
    // 如果没有规划器或者管理输出张量的标志未启用，则返回 false
    if (!planner_ || !manage_output_tensors_enabled_) {
        return false;
    }
    // 获取块信息中管理的输出张量值集合，并检查是否包含给定的值
    const auto& managed_outputs = block_info_.managed_output_tensor_values();
    return managed_outputs.find(value) != managed_outputs.end();
}

void BlockRunner::disableManageOutputTensors() {
    // 如果管理输出张量的标志未启用，则直接返回
    if (!manage_output_tensors_enabled_) {
        return;
    }

    // 禁用管理输出张量的标志，并在有规划器的情况下重置节点的输出值
    manage_output_tensors_enabled_ = false;
    if (!planner_) {
        return;
    }

    // 重置所有节点的输出值，析构规划器以便下次运行重新构建
    for (auto& n : nodes_) {
        const auto num_outputs = static_cast<uint32_t>(n.outputs().size());
        for (const auto i : c10::irange(num_outputs)) {
            n.Output(i) = IValue();
        }
    }
    planner_.reset();
}

ProcessedFunction::ProcessedFunction(
    Node* node,
    bool enable_out_variant,
    bool check_memory_overlap)
    : check_memory_overlap_(check_memory_overlap),
      num_outputs_(node->outputs().size()) {
    // 如果启用 out 变体操作，则获取相应操作函数
    if (enable_out_variant) {
        f_ = getOutOfPlaceOperation(node);
    }
}
    if (f_) {
      # 如果 f_ 不为空，则执行以下操作
      kind_ = ProcessedFunction::Kind::kOutVariant;
      // 对于输出变体，不需要检查内存重叠
      check_memory_overlap_ = false;
      # 输出日志，切换到输出变体的处理函数，打印节点信息
      VLOG(1) << "Switch to out variant for node: " << PrintNode(node);
      # 返回，不再执行后续代码
      return;
    }
  }
  {
    # 获取本地操作函数并赋值给 f_
    f_ = getNativeOperation(node);
    # 如果 f_ 不为空，则执行以下操作
    if (f_) {
      # 设置函数类型为本地函数
      kind_ = ProcessedFunction::Kind::kNativeFunction;
#ifdef NDEBUG
      // 在优化模式下跳过此检查，因为这些操作经过了更好的验证
      check_memory_overlap_ = false;
#endif
      // 打印日志，表明正在为节点切换到本地实现
      VLOG(1) << "Switch to native impl for node: " << PrintNode(node);
      // 函数返回，结束执行
      return;
    }
  }
  {
    // 获取节点的操作符
    const Operator& op = node->getOperator();
    // 创建 lambda 表达式，捕获节点的操作和是否具有可变参数
    f_ = [node_op = op.getOperation(node),
          has_var_args = hasVarArgs(node)](ProcessedNode* pnode) mutable {
      std::vector<IValue> stack;
      // 获取节点的输入数量，并预留空间
      const auto size = static_cast<uint32_t>(pnode->num_inputs());
      stack.reserve(size + has_var_args);
      // 将节点的输入放入堆栈中
      for (const auto i : c10::irange(size)) {
        stack.emplace_back(pnode->Input(i));
      }
      // 对于可变参数的操作，需要在堆栈中存储输入的数量
      if (has_var_args) {
        stack.emplace_back(static_cast<int>(size));
      }
      // 执行节点的操作
      node_op(stack);
      // 获取节点的输出数量
      const auto num_outputs = static_cast<uint32_t>(pnode->num_outputs());
      // 断言堆栈的大小与输出数量相等
      TORCH_DCHECK_EQ(stack.size(), num_outputs);
      // 将堆栈中的输出移动到节点的输出中
      for (const auto i : c10::irange(num_outputs)) {
        pnode->Output(i) = std::move(stack[i]);
      }
    };
    // 设置函数处理类型为解释器回退
    kind_ = ProcessedFunction::Kind::kInterpreterFallback;
    // 打印日志，表明正在为节点使用解释器回退
    VLOG(1) << "Fallback interpreter for node: " << PrintNode(node);
  }
}

StaticNodeInfo::StaticNodeInfo(
    Node* node,
    ProcessedFunction* fn,
    ProcessedNodeInputs inputs,
    uint16_t outputs_offset)
    : node_(node),
      fn_(fn),
      inputs_(std::move(inputs)),
      outputs_offset_(outputs_offset) {
  // 检查节点的输出数量是否与期望的输出数量相符
  TORCH_CHECK(
      num_outputs() == node->outputs().size(),
      "Node ",
      node->kind().toQualString(),
      " has ",
      std::to_string(num_outputs()),
      " outputs, expected ",
      std::to_string(node->outputs().size()));
}

// 返回节点输入的 IValue 向量
std::vector<IValue> ProcessedNode::inputs_ivalue_vec() const {
  std::vector<IValue> result;
  // 获取节点输入的数量
  const auto num_inputs = static_cast<uint32_t>(inputs_.size());
  // 预留足够的空间
  result.reserve(num_inputs);

  // 将节点的输入转换为 IValue，并存储在结果向量中
  for (const auto idx : c10::irange(num_inputs)) {
    result.emplace_back(Input(idx));
  }
  return result;
}

// 执行节点的函数
void ProcessedNode::run() {
#ifdef FBCODE_CAFFE2
  // 在节点运行开始时触发操作观察器
  SROperatorObserver::onStart(node());
#endif
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  // 获取静态运行时的步骤回调
  auto step_callbacks =
      at::getStepCallbacksUnlessEmpty(at::RecordScope::STATIC_RUNTIME_OP);
  if (C10_UNLIKELY(step_callbacks.has_value())) {
    // 创建记录函数保护区域
    at::RecordFunction guard(std::move(*step_callbacks));
    // 断言保护区域是活跃的
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(guard.isActive());
    // 如果需要输入，准备输入数据并在保护区域前触发
    if (guard.needsInputs()) {
      const auto inputs = inputs_ivalue_vec();
      guard.before(
          get_op_name(),
          c10::ArrayRef<const IValue>(inputs.data(), inputs.size()));
    } else {
      guard.before(get_op_name());
    }
    // 如果有输出变体，设置静态运行时的输出变体
    if (has_out_variant()) {
      guard._setStaticRuntimeOutVariant();
    }

    // 运行节点处理函数
    fn_->run(this);
  } else {
    // 运行节点处理函数（无记录保护区域）
    fn_->run(this);
  }
#else
  // 运行节点处理函数（无性能分析）
  fn_->run(this);
#endif
#ifndef NDEBUG
  // 如果禁用调试模式下的内存重叠检查，则执行检查但不强制执行
  if (FLAGS_static_runtime_disable_debug_memory_overlap_check) {
    verify_no_memory_overlap();
  } else {
    // 断言调试模式下的内存重叠检查通过
    DCHECK(verify_no_memory_overlap());
  }

    verify_no_memory_overlap();
  }
#endif


注释结束。
#endif
#ifdef FBCODE_CAFFE2
  SROperatorObserver::onEnd(node());
#endif
}

// 检查两个张量是否存在内存重叠
static bool checkNoMemoryOverlap(const at::Tensor& a, const at::Tensor& b) {
  // 获取张量 a 和 b 之间的内存重叠状态
  at::MemOverlapStatus status = at::get_overlap_status(a, b);
  // 如果完全重叠或部分重叠，则返回 false
  if (status == at::MemOverlapStatus::Full ||
      status == at::MemOverlapStatus::Partial) {
    return false;
  }
  // 如果内存重叠状态为 TooHard，则记录日志
  if (status == at::MemOverlapStatus::TooHard) {
    VLOG(1) << "Detected TOO_HARD memory overlap status";
  }
  // 否则返回 true，表示不存在内存重叠
  return true;
}

// 验证当前处理节点是否存在内存重叠
bool ProcessedNode::verify_no_memory_overlap(bool force_check) const {
  // 特殊情况操作的符号集合
  const static std::array<c10::Symbol, 7> special_case_ops = {
      fromQualString("prim::TypeCheck"),
      fromQualString("prim::IfThenElse"),
      fromQualString("static_runtime::select_tensor"),
      fromQualString("static_runtime::VarTupleUnpack"),
      fromQualString("static_runtime::dict_unpack"),
      fromQualString("static_runtime::fused_split_and_squeeze"),
      fromQualString("static_runtime::create_owned_ref")};
  // 如果不是强制检查并且节点符号在特殊情况操作集合中，则直接返回 true
  if (!force_check &&
      std::find(
          begin(special_case_ops), end(special_case_ops), node()->kind()) !=
          end(special_case_ops)) {
    return true;
  }

  // 否则继续验证输出不重叠且输入不与输出重叠
  return verify_outputs_dont_overlap_each_other() &&
      verify_inputs_dont_overlap_outputs(force_check);
}

// 验证节点的输出张量之间是否不存在重叠
bool ProcessedNode::verify_outputs_dont_overlap_each_other() const {
  const auto n_outputs = static_cast<uint32_t>(num_outputs());
  for (const auto i : c10::irange(n_outputs)) {
    if (!Output(i).isTensor()) {
      continue;
    }
    const auto& out0_t = Output(i).toTensor();
    for (const auto j : c10::irange(i + 1, n_outputs)) {
      if (!Output(j).isTensor()) {
        continue;
      }
      const auto& out1_t = Output(j).toTensor();
      // 如果输出张量之间存在内存重叠，则记录日志并返回 false
      if (!checkNoMemoryOverlap(out0_t, out1_t)) {
        LOG(INFO) << "Node output " << i << " overlaps with output " << j
                  << ", " << PrintNode(node_);
        return false;
      }
    }
  }
  // 输出张量之间不存在重叠，返回 true
  return true;
}

// 验证节点的输入张量不与其输出张量重叠
bool ProcessedNode::verify_inputs_dont_overlap_outputs(bool force_check) const {
  // 获取节点的操作模式
  auto schema = node()->maybeSchema();
  // 如果操作模式为空，或者非强制检查且满足跳过检查的条件，则直接返回 true
  bool skip_check = !schema ||
      ((schema->is_mutable() || !fn_->checkMemoryOverlap()) &&
       num_outputs() == 1);
  if (!schema || (!force_check && skip_check)) {
    // 如果操作模式为空，则记录日志并返回 true
    if (!schema) {
      VLOG(2) << "Detected that op schema is null";
      return true;
    }
    // 否则记录相关信息并返回 true
    VLOG(2) << "schema->is_mutable: " << schema->is_mutable()
            << ", fn_->checkMemoryOverlap: " << fn_->checkMemoryOverlap()
            << ", num_outputs_: " << num_outputs();
    return true;
  }
  // 对于每个输入张量，检查其是否与输出张量重叠
  const auto n_inputs = static_cast<uint32_t>(inputs_.size());
  const auto n_outputs = static_cast<uint32_t>(num_outputs());
  for (const auto i : c10::irange<uint32_t>(n_inputs)) {
    const IValue* in = &Input(i);
    if (!in->isTensor()) {
      continue;
    }
    const auto& in_t = in->toTensor();
    // 对于输出节点的循环，使用范围循环遍历所有输出
    for (const auto j : c10::irange(n_outputs)) {
      // 获取当前输出节点的值
      const IValue& out = Output(j);
      // 如果输出不是张量，跳过当前循环，继续下一个输出节点
      if (!out.isTensor()) {
        continue;
      }
      // 将输出节点转换为张量类型
      const auto& out_t = out.toTensor();
      // 检查输入张量和输出张量之间是否存在内存重叠
      if (!checkNoMemoryOverlap(in_t, out_t)) {
        // 如果存在内存重叠，记录日志并返回失败
        LOG(INFO) << "Node input " << i << " overlaps with output " << j << ", "
                  << PrintNode(node_);
        LOG(INFO) << *schema;
        return false;
      }
    }
  }
  // 所有输出节点检查无内存重叠，返回成功
  return true;
}

bool ProcessedNode::check_and_correct_overlap_with(
    const at::Tensor& input,
    c10::IValue& output_ival) {
  auto& tensor = output_ival.toTensor();  // 获取输出值的张量表示
  if (!checkNoMemoryOverlap(input, tensor)) {  // 检查输入和输出张量是否存在内存重叠
    DLOG(INFO) << "Detected alias for node: " << PrintNode(node());  // 记录节点存在别名的信息
    output_ival = at::native::clone(tensor, c10::nullopt);  // 克隆输出张量以解决内存重叠问题
    set_outputs_memory_overlap_detected();  // 标记检测到了输出内存重叠
    return true;  // 返回 true 表示已检测到内存重叠并进行了修正
  }
  return false;  // 返回 false 表示未检测到内存重叠
}

void ProcessedNode::verify_and_correct_memory_overlap() {
  const auto n_inputs = static_cast<uint32_t>(inputs_.size());  // 获取输入的数量
  const auto n_outputs = static_cast<uint32_t>(num_outputs());  // 获取输出的数量
  for (const auto i : c10::irange(n_inputs)) {  // 遍历所有输入
    const IValue& in = Input(i);  // 获取当前输入
    if (!in.isTensor()) {  // 如果当前输入不是张量，则跳过
      continue;
    }
    const auto& in_t = in.toTensor();  // 获取当前输入的张量表示
    for (const auto j : c10::irange(n_outputs)) {  // 遍历所有输出
      auto& output = Output(j);  // 获取当前输出
      if (output.isTensor()) {  // 如果当前输出是张量
        check_and_correct_overlap_with(in_t, output);  // 检查并修正输入与当前输出的内存重叠
      } else if (output.isTensorList()) {  // 如果当前输出是张量列表
        auto tensors = output.toListRef();  // 获取张量列表
        for (const auto& ival : tensors) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          check_and_correct_overlap_with(in_t, const_cast<c10::IValue&>(ival));  // 检查并修正输入与当前张量列表中的张量的内存重叠
        }
#ifdef FBCODE_CAFFE2
        if (outputs_memory_overlap_detected()) {  // 如果检测到输出内存重叠
          LOG_EVERY_MS(WARNING, 60000)
              << "Detected alias for node: " << PrintNode(node());  // 记录节点存在别名的信息
        }
#endif
      }
    }
  }
}

StaticRuntime::StaticRuntime(const StaticModule& sm)
    : values_(sm.value_buffer_size()) {
  std::copy(sm.constants().begin(), sm.constants().end(), values_.data());  // 复制静态模块中的常量到运行时的值缓冲区
  // 默认的任务启动器设置为跨操作的线程池
  async_task_launcher_ = at::launch;
  block_ = std::make_unique<BlockRunner>(
      sm,
      values_.data(),
      sm.root_block(),
      &async_task_launcher_,
      true /*is_root_block*/);  // 创建并初始化 BlockRunner 对象作为根块的运行时执行单元
}

c10::IValue StaticRuntime::operator()(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
  return (*block_)(args, kwargs);  // 调用 BlockRunner 对象执行给定参数和关键字参数的运行
}

c10::IValue StaticRuntime::operator()(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
  return (*block_)(std::move(args), kwargs);  // 调用 BlockRunner 对象执行给定参数和关键字参数的运行
}

c10::intrusive_ptr<c10::ivalue::Future> StaticRuntime::runAsync(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs,
    torch::jit::TaskLauncher taskLauncher) {
  async_task_launcher_ = std::move(taskLauncher);  // 设置异步任务启动器
  return block_->runAsync(args, kwargs);  // 异步执行 BlockRunner 对象的运行
}

c10::intrusive_ptr<c10::ivalue::Future> StaticRuntime::runAsync(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs,
    torch::jit::TaskLauncher taskLauncher) {
  async_task_launcher_ = std::move(taskLauncher);  // 设置异步任务启动器
  return block_->runAsync(std::move(args), kwargs);  // 异步执行 BlockRunner 对象的运行
}

bool StaticRuntime::check_for_memory_leak(bool output_returned) {
  return block_->check_for_memory_leak(
      output_returned, /* recurse_on_sub_blocks */ true);  // 检查块运行时是否存在内存泄漏
}

bool StaticRuntime::checkOutputTensorMemoryLeaks() {
  return block_->checkOutputTensorMemoryLeaks();  // 检查块运行时输出张量是否存在内存泄漏
}
// 释放由 block_ 指向的 StaticRuntime 对象的输出张量
void StaticRuntime::deallocateOutputTensors() {
    block_->deallocateOutputTensors();
}

// 检查给定的 IValue 是否为 block_ 管理的输出张量之一，返回布尔值
bool StaticRuntime::isManagedOutputTensor(const IValue& ivalue) const {
    return block_->isManagedOutputTensor(ivalue);
}

// 禁用 block_ 对输出张量的管理功能
void StaticRuntime::disableManageOutputTensors() {
    block_->disableManageOutputTensors();
}

// 获取 block_ 使用的内存规划器的指针，并返回之
const MemoryPlanner* StaticRuntime::get_memory_planner() const {
    return block_->get_memory_planner();
}

// 结束 torch::jit 命名空间
} // namespace torch::jit
```