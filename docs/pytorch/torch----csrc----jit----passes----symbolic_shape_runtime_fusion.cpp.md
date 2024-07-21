# `.\pytorch\torch\csrc\jit\passes\symbolic_shape_runtime_fusion.cpp`

```py
// 包含头文件，引入必要的库和声明
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <sstream>
#include <utility>

// 命名空间声明
namespace torch {
namespace jit {

// 插入符号形状计算图中每个符号形状的计算，并返回从符号形状值到其运行时值的映射
static std::map<int64_t, Value*> InsertSymbolicShapesCompute(
    const ShapeComputeGraphMapping& shape_mapping,
    Node* tensorexpr_graph) {
  // 在 tensorexpr_graph 中设置插入点
  WithInsertPoint guard(tensorexpr_graph);
  // 获取父图
  auto enclosing_graph = tensorexpr_graph->owningGraph();

  // 创建形状计算图输入到父图值的映射
  std::map<Value*, Value*> shape_graph_input_to_enclosing_graph_value;
  for (const auto& pair :
       shape_mapping.enclosing_graph_value_to_shape_graph_input_) {
    shape_graph_input_to_enclosing_graph_value[pair.second] = pair.first;
  }

  // 构建形状计算图的输入列表
  std::vector<Value*> shape_compute_graph_inputs;
  for (Value* shape_graph_input :
       shape_mapping.partial_eval_shape_graph->inputs()) {
    auto enclosing_graph_input =
        shape_graph_input_to_enclosing_graph_value.find(shape_graph_input);
    // 断言确保在映射中找到了父图的对应输入
    TORCH_INTERNAL_ASSERT(
        enclosing_graph_input !=
        shape_graph_input_to_enclosing_graph_value.end());
    // 如果输入类型匹配，则直接使用父图的对应输入
    if (*enclosing_graph_input->second->type() == *shape_graph_input->type()) {
      shape_compute_graph_inputs.push_back(tensorexpr_graph->inputs().at(
          enclosing_graph_input->second->offset()));
    } else {
      // 否则，插入一个大小计算操作到父图，使用父图对应输入作为参数
      TORCH_INTERNAL_ASSERT(
          enclosing_graph_input->second->type()->cast<TensorType>() &&
          shape_graph_input->type()->isSubtypeOf(ListType::ofInts()));
      shape_compute_graph_inputs.push_back(enclosing_graph->insert(
          aten::size,
          {tensorexpr_graph->inputs().at(
              enclosing_graph_input->second->offset())}));
    }
  }

  // 插入形状计算图到父图，并获取符号形状值
  auto sym_shape_values = insertGraph(
      *enclosing_graph,
      *shape_mapping.partial_eval_shape_graph,
      shape_compute_graph_inputs);

  // 创建符号形状值到父图值的映射
  std::map<int64_t, Value*> sym_shape_to_enclosing_graph_value;
  for (size_t i = 0;
       i < shape_mapping.partial_eval_shape_graph->outputs().size();
       ++i) {
    Value* output = shape_mapping.partial_eval_shape_graph->outputs().at(i);
    auto sym_shape =
        shape_mapping.graph_output_to_symbolic_shape_dim_.find(output);
    // 断言确保找到了符号形状维度的映射
    TORCH_INTERNAL_ASSERT(
        sym_shape != shape_mapping.graph_output_to_symbolic_shape_dim_.end());
    // 将符号形状值映射到父图的输出值
    sym_shape_to_enclosing_graph_value[sym_shape->second] = sym_shape_values[i];
  }

  // 返回符号形状值到父图值的映射
  return sym_shape_to_enclosing_graph_value;
}
}

// 插入动态形状保护，用于给定节点的形状计算图映射
void insertDynamicShapesGuard(
    const ShapeComputeGraphMapping& shape_mapping,
    Node* guarded_node,
    bool add_composed_op,
    std::vector<std::vector<StrideInput>>& input_info,
    std::vector<StrideInput>& output_strides);

// 将 StrideInput 枚举类型转换为字符串表示
std::string toString(StrideInput si) {
  switch (si) {
    case StrideInput::TENSOR_CONT:
      return "TENSOR_CONT";
    case StrideInput::TENSOR_CONT_CHANNELS_LAST:
      return "TENSOR_CONT_CHANNELS_LAST";
    case StrideInput::S_ONE:
      return "S_ONE";
    case StrideInput::S_CONT:
      return "S_CONT";
    case StrideInput::S_TRAN_CONT:
      return "S_TRAN_CONT";
    case StrideInput::S_AS_ARG:
      return "S_AS_ARG";
  }
  // 如果出现未知的 StrideInput，触发内部断言错误
  TORCH_INTERNAL_ASSERT(false);
}

// 将字符串表示的 StrideInput 转换为枚举类型
StrideInput strideInputFromString(const std::string& si) {
  if (si == "TENSOR_CONT") {
    return StrideInput::TENSOR_CONT;
  } else if (si == "TENSOR_CONT_CHANNELS_LAST") {
    return StrideInput::TENSOR_CONT_CHANNELS_LAST;
  } else if (si == "S_ONE") {
    return StrideInput::S_ONE;
  } else if (si == "S_CONT") {
    return StrideInput::S_CONT;
  } else if (si == "S_TRAN_CONT") {
    return StrideInput::S_TRAN_CONT;
  } else if (si == "S_AS_ARG") {
    return StrideInput::S_AS_ARG;
  } else {
    // 如果字符串不匹配任何已知的 StrideInput 类型，触发内部断言错误
    TORCH_INTERNAL_ASSERT(false);
  }
}

// 摘要化给定维度的步幅类型
inline StrideInput summarizeStrideDim(
    const c10::IntArrayRef sizes,
    const c10::IntArrayRef strides,
    size_t dim,
    const std::vector<StrideInput>& stride_inputs,
    size_t stride_inputs_offset) {
  if (strides[dim] == 1) {
    return StrideInput::S_ONE;
  } else if (
      dim + 1 < sizes.size() &&
      strides[dim] == strides[dim + 1] * sizes[dim + 1]) {
    return StrideInput::S_CONT;
    // 转置连续依赖于前一个维度的连续，为了避免相互依赖，检查下一个维度是否为步幅连续
  } else if (
      dim > 0 && strides[dim] == strides[dim - 1] * sizes[dim - 1] &&
      (stride_inputs[dim - 1 + stride_inputs_offset] != StrideInput::S_CONT)) {
    return StrideInput::S_TRAN_CONT;
  } else {
    return StrideInput::S_AS_ARG;
  }
}

// 摘要化输入张量类型的步幅信息
static std::vector<StrideInput> summarizeInputStrides(const TensorType& tt) {
  auto strides = *tt.strides().concrete_sizes();
  auto sizes = *tt.sizes().concrete_sizes();
  if (c10::is_contiguous_strides(sizes, strides)) {
    return {StrideInput::TENSOR_CONT};
    // TODO: channels last 3d
  } else if (c10::is_channels_last_strides_2d(sizes, strides)) {
    return {StrideInput::TENSOR_CONT_CHANNELS_LAST};
  }
  // 对每个维度进行步幅摘要化
  std::vector<StrideInput> stride_inputs;
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    stride_inputs.push_back(
        summarizeStrideDim(sizes, strides, dim, stride_inputs, 0));
  }
  return stride_inputs;
};

// Todo: incorporate in codegen
// 汇总输出张量的步幅信息
static StrideInput summarizeOutputStrides(const TensorType& tt) {
  // 获取张量的步幅和大小信息
  auto strides = *tt.strides().concrete_sizes();
  auto sizes = *tt.sizes().concrete_sizes();

  // 只有在通道最后的张量中尝试维护输出步幅，否则使用连续的步幅
  // TODO: channels last 3d
  if (c10::is_channels_last_strides_2d(sizes, strides)) {
    return StrideInput::TENSOR_CONT_CHANNELS_LAST;
  }
  return StrideInput::TENSOR_CONT;
}

// 将完整形状的输入泛化为符号形状
// 值为1的维度将被保留，其他相同值的维度将被归为同一个符号形状
// 例如：Tensor(5, 3), Tensor(3, 1) -> Tensor(SS(-1), SS(-2)), Tensor(SS(-2), 1)
// 还要汇总输入的步幅行为。尺寸信息存储在类型上，步幅则被返回。参见 StrideInput 获取步幅特化的描述
static std::optional<std::vector<std::vector<StrideInput>>>
TryGeneralizeInputDimensionsToSymbolicShapes(
    std::shared_ptr<Graph> tensorexpr_graph) {
  // 映射形状到符号形状的映射
  std::map<size_t, int64_t> shape_to_sym_shape;
  // 存储输入的步幅信息
  std::vector<std::vector<StrideInput>> input_striding;

  // 遍历张量表达式图的输入值
  for (Value* v : tensorexpr_graph->inputs()) {
    // 如果值不是张量类型，则跳过
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto tt = v->type()->expectRef<TensorType>();
    
    // 如果张量的尺寸或步幅不完整，则返回空
    if (!tt.sizes().isComplete() || !tt.strides().isComplete()) {
      return c10::nullopt;
    }
    
    // 汇总该张量的输入步幅
    input_striding.push_back(summarizeInputStrides(tt));
    
    // 获取张量的符号尺寸
    std::vector<at::ShapeSymbol> shape_vec = *tt.symbolic_sizes().sizes();
    // 更新尺寸到符号形状的映射关系
    auto new_sizes = c10::fmap(shape_vec, [&](const at::ShapeSymbol& shape) {
      auto value = shape.value();
      TORCH_INTERNAL_ASSERT(value >= 0, "Expected complete tensor");
      if (value == 1) {
        return value;
      } else if (shape_to_sym_shape.count(static_cast<size_t>(value))) {
        return shape_to_sym_shape[value];
      } else {
        auto new_shape_symbol = at::ShapeSymbol::newSymbol().value();
        shape_to_sym_shape[static_cast<size_t>(value)] = new_shape_symbol;
        return new_shape_symbol;
      }
    });
    // 设置张量的类型，使用符号形状
    v->setType(tt.withSymbolicShapes(c10::SymbolicShape(new_sizes)));
  }
  // 返回汇总的输入步幅信息
  return input_striding;
}

// 将常量张量移出子图
static void moveConstantTensorsOutOfSubgraph(
    Node* tensorexpr_graph_node,
    std::shared_ptr<Graph> tensorexpr_graph) {
  // 获取父图
  auto parent = tensorexpr_graph_node->owningGraph();

  // 环境函数，用于处理值
  auto env = [&](Value* v) {
    TORCH_INTERNAL_ASSERT(
        false,
        "this should never happen since constant nodes do not have any inputs",
        v->debugName());
    return v;
  };

  // 设置插入点为指定节点的位置
  WithInsertPoint wip(tensorexpr_graph_node);
  // 存储待销毁的节点
  std::vector<Node*> to_destroy;
  // 遍历张量表达式图中的节点
  for (auto node : tensorexpr_graph->nodes()) {
    // 如果节点的类型是常量节点（prim::Constant）
    if (node->kind() == prim::Constant) {
      // 如果该常量节点的输出类型不能转换为 TensorType，则跳过处理
      if (!node->output()->type()->cast<TensorType>()) {
        continue;
      }

      // 复制常量节点，并将复制后的节点插入到父图中
      auto copy = parent->createClone(node, env);
      parent->insertNode(copy);

      // 在 te 子图中添加一个新的输入，并用这个输入替换常量节点的使用处
      auto new_const = tensorexpr_graph->addInput();
      new_const->setType(node->output()->type());
      node->output()->replaceAllUsesWith(new_const);

      // 将复制的节点作为输入添加到 te 节点中
      tensorexpr_graph_node->addInput(copy->output());

      // 将需要销毁的节点加入销毁列表
      to_destroy.push_back(node);
    }
  }

  // 遍历需要销毁的节点列表，逐个销毁这些节点
  for (auto n : to_destroy) {
    n->destroy();
  }
// 生成保护条件，确保张量表达式图节点的动态形状约束
bool GenerateGuard(Node* tensorexpr_graph_node, bool add_composed_op) {
  // 获取张量表达式子图
  auto tensorexpr_graph = SubgraphUtils::getSubgraph(tensorexpr_graph_node);

  // 将子图中的常量张量移到外部作用域
  // 这是必要的，因为符号形状分析不能很好地处理常量广播到符号形状的情况，会导致性能下降
  moveConstantTensorsOutOfSubgraph(tensorexpr_graph_node, tensorexpr_graph);

  // 泛化输入维度
  auto input_striding =
      TryGeneralizeInputDimensionsToSymbolicShapes(tensorexpr_graph);
  if (!input_striding) {
    return false;
  }

  // 获取输出的步幅信息
  std::vector<StrideInput> output_striding;
  for (Value* v : tensorexpr_graph->outputs()) {
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto tt = v->type()->expectRef<TensorType>();
    if (!tt.sizes().isComplete() || !tt.strides().isComplete()) {
      return false;
    }
    output_striding.push_back(summarizeOutputStrides(tt));
  }

  // 尝试传播形状
  auto maybe_shape_compute_mapping =
      PropagateShapesAndBuildLargeShapeComputeGraph(
          tensorexpr_graph,
          *tensorexpr_graph->nodes().begin(),
          *tensorexpr_graph->nodes().end());
  if (!maybe_shape_compute_mapping) {
    return false;
  }

  // 插入保护条件
  insertDynamicShapesGuard(
      *maybe_shape_compute_mapping,
      tensorexpr_graph_node,
      add_composed_op,
      *input_striding,
      output_striding);
  return true;
}

// 内联回退图并添加静态运行时复制输出操作
static void inlineFallbackGraphAndAddSRCopyOutOp(std::shared_ptr<Graph> graph) {
  // 深度优先遍历图节点
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::FallbackGraph) {
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(n != nullptr, "Expected to find fallback graph");

  // 获取 if 节点视图并解除子图合并
  auto if_node = n->owningBlock()->owningNode();
  IfView if_v(if_node);
  SubgraphUtils::unmergeSubgraph(n);

  // 获取 else 分支的输出
  auto false_block = if_v.elseBlock();
  std::vector<Value*> false_block_outputs(
      if_v.elseOutputs().begin(), if_v.elseOutputs().end());
  TORCH_INTERNAL_ASSERT(!false_block_outputs.empty());

  // 确保所有输出为张量类型
  for (auto out : false_block_outputs) {
    TORCH_INTERNAL_ASSERT(out->type()->cast<TensorType>());
  }

  // 创建静态运行时复制输出节点，并将其添加到 else 分支中
  auto copy_node = graph->create(
      prim::StaticRuntimeCopyOuts,
      false_block_outputs,
      false_block_outputs.size());
  false_block->appendNode(copy_node);
  for (size_t i = 0; i < false_block_outputs.size(); ++i) {
    false_block->replaceOutput(i, copy_node->outputs().at(i));
  }
}

// TODO: 与 tensorexpr_fuser 共享更多逻辑？
// 插入动态形状保护条件
void insertDynamicShapesGuard(
    const ShapeComputeGraphMapping& shape_mapping,
    Node* guarded_node,
    bool add_composed_op,
    std::vector<std::vector<StrideInput>>& input_info,
  // 在输出流中记录调试信息，表明正在为一个节点插入 prim::TensorExprDynamicGuard 保护
  GRAPH_DEBUG(
      "Inserting a prim::TensorExprDynamicGuard guard for a node",
      *guarded_node);
  // 获取被保护节点的子图
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // 修正子图输入的类型
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  // 遍历被保护节点的输入
  for (const auto i : c10::irange(guarded_node->inputs().size())) {
    Value* node_input = guarded_node->inputs().at(i);
    // 只检查被保护节点的输入是否为 Tensor 类型
    if (!node_input->type()->cast<TensorType>()) {
      continue;
    }
    inputs_to_check.push_back(node_input);
    // 创建带有不同步长的期望 Tensor 类型
    guard_types.emplace_back(
        subgraph->inputs().at(i)->type()->expect<TensorType>()->withStrides(
            c10::VaryingShape<c10::Stride>()));
  }
  // 断言检查至少存在一个需要检查的输入
  TORCH_INTERNAL_ASSERT(inputs_to_check.size());

  // 创建 prim::TensorExprDynamicGuard 节点，用于类型检查
  Node* typecheck_node =
      guarded_node->owningGraph()
          ->create(Symbol::prim("TensorExprDynamicGuard"), inputs_to_check, 1)
          ->insertBefore(guarded_node);

  // 设置 typecheck_node 的类型属性为 guard_types
  typecheck_node->tys_(attr::types, std::move(guard_types));
  // 设置 typecheck_node 的输出类型为 BoolType
  Value* typecheck_result = typecheck_node->output()->setType(BoolType::get());

  // 插入 if 节点，根据 typecheck_result 条件分支执行
  auto versioning_if =
      guarded_node->owningGraph()
          ->create(prim::If, {typecheck_result}, guarded_node->outputs().size())
          ->insertAfter(typecheck_node);

  // 将 guarded_node 的输出连接到 versioning_if 节点的输出
  for (size_t idx = 0; idx < guarded_node->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(guarded_node->output(idx)->type());
    guarded_node->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // 填充 false 分支，包含未优化的融合子图
  WithInsertPoint guard(false_block->return_node());
  // 将融合子图插入到 guarded_node 所在的图中
  const auto subgraph_outputs = insertGraph(
      *guarded_node->owningGraph(), *subgraph, guarded_node->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // 在替换之前移除特定于 Tensor 类型的特化
  removeTensorTypeSpecializations(false_block);
  // 使用后备图替换 false 分支
  replaceBlockWithFallbackGraph(false_block, guarded_node->inputs());

  // 填充 true 分支，所有输入经过类型检查，其体应为融合组节点
  guarded_node->moveBefore(true_block->return_node());

  for (Value* output : guarded_node->outputs()) {
    true_block->registerOutput(output);
  }

  // 插入符号形状计算并将其作为 TE 节点/图的输入添加
  // symbolic_shape_inputs 将是每个符号形状的列表，
  // TE 图/节点的最后 N 个输入将是 N 个符号形状的值
  auto map = InsertSymbolicShapesCompute(shape_mapping, guarded_node);
  std::vector<int64_t> symbolic_shape_inputs;
  // 遍历符号形状映射，构建符号形状输入列表，并将符号形状值作为输入添加到 guarded_node
  for (const auto& pair : map) {
    symbolic_shape_inputs.push_back(pair.first);
    guarded_node->addInput(pair.second);
    // 创建符号形状的输入名称，并将其添加到子图中
    std::stringstream ss;
    ss << "SS_" << -pair.first;
    subgraph->addInput(ss.str())->setType(IntType::get());
  }
  // 将符号形状输入列表设置为 guarded_node 的属性 symbolic_shape_inputs
  guarded_node->is_(
      attr::symbolic_shape_inputs, std::move(symbolic_shape_inputs));

  // 构建输入信息的字符串化版本，并将其存储为 IValue 类型
  std::vector<std::vector<std::string>> input_striding;
  for (auto& vec : input_info) {
    auto string_info =
        fmap(vec, [&](StrideInput inp) { return toString(inp); });
    input_striding.push_back(string_info);
  }
  auto ival = IValue(input_striding);
  // 将输入描述信息存储为 typecheck_node 和 guarded_node 的属性 striding_inputs_desc
  guarded_node->ival_(attr::striding_inputs_desc, ival);
  typecheck_node->ival_(attr::striding_inputs_desc, std::move(ival));

  // 更新子图中输入的张量类型，将其 strides 设置为空的 VaryingShape
  for (Value* v : subgraph->inputs()) {
    if (auto t = v->type()->cast<TensorType>()) {
      v->setType(t->withStrides(c10::VaryingShape<c10::Stride>()));
    }
  }
  // 更新子图中输出的张量类型，将其 strides 设置为空的 VaryingShape
  for (Value* v : subgraph->outputs()) {
    if (auto t = v->type()->cast<TensorType>()) {
      v->setType(t->withStrides(c10::VaryingShape<c10::Stride>()));
    }
  }

  // 构建输出步幅信息的字符串化版本，并将其存储为 IValue 类型
  std::vector<std::string> output_striding =
      fmap(output_strides, [&](StrideInput inp) { return toString(inp); });
  auto output_ival = IValue(output_striding);
  // 将输出描述信息存储为 guarded_node 的属性 striding_outputs_desc
  guarded_node->ival_(attr::striding_outputs_desc, std::move(output_ival));

  // 如果需要添加合成操作
  if (add_composed_op) {
    // 仅在 SR 流程中检查堆栈上的值，并将其作为张量输出传递
    // TODO: - 重构并显式地作为 TE Kernel API 的一部分
    guarded_node->i_(attr::allow_stack_outputs, 1);

    // 创建一个 TensorExprDynamicGroup 节点
    auto te_dyn_group = SubgraphUtils::createSingletonSubgraph(
        typecheck_node, prim::TensorExprDynamicGroup);
    // 将 versioning_if 节点合并到 te_dyn_group 子图中
    SubgraphUtils::mergeNodeIntoSubgraph(versioning_if, te_dyn_group);
    // 内联回退图并添加 SRCopyOut 操作
    inlineFallbackGraphAndAddSRCopyOutOp(
        SubgraphUtils::getSubgraph(te_dyn_group));
  }
// This operator defines a function `StaticRuntimeCopyOuts` that takes a Node pointer
// and returns an Operation. This function is used in the context of a fusion group
// fallback block transformation. It manages input and output tensors, ensuring that
// outputs are correctly copied to pre-allocated tensors during execution.
static Operation StaticRuntimeCopyOuts(const Node* node) {
  // Determine the number of tensor inputs to the operation from the node.
  auto num_ten_inputs = node->inputs().size();
  // Return a lambda function that operates on a Stack reference.
  return [num_ten_inputs](Stack& stack) {
    // Pop inputs from the stack based on the number of tensor inputs.
    std::vector<IValue> inputs = pop(stack, num_ten_inputs);
    // Handle the case where the stack is empty, initializing it with input elements.
    if (stack.empty()) {
      for (IValue elem : inputs) {
        push(stack, std::move(elem));
      }
    } else {
      // Retrieve outputs from the stack matching the number of tensor inputs.
      at::ArrayRef<IValue> outputs = last(stack, num_ten_inputs);
      // For each input, copy its value to the corresponding pre-allocated tensor output.
      for (size_t i = 0; i < inputs.size(); ++i) {
        IValue out = outputs[i];
        at::Tensor& out_t = out.toTensor();
        fastResizeToZero(out_t);  // Resize the output tensor.
        out_t.resize_as_(inputs[i].toTensor());  // Resize to match input tensor shape.
        out_t.copy_(inputs[i].toTensor());  // Copy input tensor data to output tensor.
      }
    }
    // Return zero to indicate successful execution.
    return 0;
  };
}

// Register the `StaticRuntimeCopyOuts` function as an operator named `prim::StaticRuntimeCopyOuts`.
// This operator is associated with the `StaticRuntimeCopyOuts` function and uses conservative alias analysis.
RegisterOperators SRCopyOuts({
    torch::jit::Operator(
        prim::StaticRuntimeCopyOuts,
        StaticRuntimeCopyOuts,
        AliasAnalysisKind::CONSERVATIVE),
});

// Placeholder comment (not related to the actual code functionality).
// On each invocation of this guard, we need to check all of the static
// information (dtype/device/requires grad/contiguity/static dims),
// and also the that the symbolic shape dimensions are observed.
// For any symbolic dimension we need to set its value on its first
// use and for all subsequent uses check that the values are equal
RegisterOperators reg_guard({
});

// Define a function `runTensorExprDynamicGroup` that executes the given `code` using an InterpreterState.
// It modifies the `stack` reference directly.
void runTensorExprDynamicGroup(const Code& code, Stack& stack) {
  // Create an InterpreterState object using the provided `code`.
  InterpreterState interpreter{code};
  // Execute the interpreter on the provided `stack`.
  interpreter.run(stack);
}
static Operation createTensorExprDynamicGroup(const Node* node) {
  // 从节点中获取子图对象
  const auto& graph = node->g(attr::Subgraph);
  // 创建一个新的 Code 对象，传入空字符串作为参数
  Code code(graph, "");
  // 此实现在每次调用 TensorExprDynamicGroup 时都会创建一个新的 Code 对象和 InterpreterState 对象，
  // 这会影响性能。理想情况下，应该在多次调用此操作时重用 Code 和 InterpreterState 对象。
  // 但目前这样做会导致 "No frames found" 错误。
  // TODO: 通过找到更好的方法来提高这部分代码的性能。
  // 注意：这段代码仅在单线程的情况下运行。
  return [code](Stack& stack) {
    // 运行 TensorExprDynamicGroup 操作，传入之前创建的 Code 对象和当前的栈对象
    runTensorExprDynamicGroup(code, stack);
    return 0;  // 返回操作执行的结果，这里始终返回0
  };
}

RegisterOperators TensorExprDynamicOp({
    torch::jit::Operator(
        prim::TensorExprDynamicGroup,
        createTensorExprDynamicGroup,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});
```