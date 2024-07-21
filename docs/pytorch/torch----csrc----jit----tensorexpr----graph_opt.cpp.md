# `.\pytorch\torch\csrc\jit\tensorexpr\graph_opt.cpp`

```py
// 包含头文件，声明使用的 Torch 的 TensorExpr 库中的相关功能
#include <torch/csrc/jit/tensorexpr/graph_opt.h>

// 包含 Torch 的 JIT 日志功能的头文件
#include <torch/csrc/jit/jit_log.h>

// 包含 Torch 的 JIT 死代码消除功能的头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 包含 Torch 的 JIT TensorExpr 融合器的头文件
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

// 包含 Torch 的 JIT 运行时符号形状注册工具的头文件
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>

// 包含 Torch 的 TensorExpr 核心功能的头文件
#include <torch/csrc/jit/tensorexpr/kernel.h>

// Torch 的 JIT 和 TensorExpr 的命名空间定义
namespace torch::jit::tensorexpr {

// 将给定的 aten::cat 操作的使用者移动到其输入参数之后的函数
static Node* moveCatAfterUse(
    Node* cat,  // aten::cat 操作的节点
    Node* user,  // 使用 aten::cat 输出的节点
    std::shared_ptr<Graph> subgraph) {  // 共享指针，表示子图

  // 以下是示例 IR：
  //   %1 = ...
  //   %2 = ...
  //   %3 = prim::ListConstruct(%1, %2)
  //   %4 = aten::cat(%3, ...)
  //   %5 = aten::relu(%4)
  //   return (%5)
  //
  // 转换后的形式应为：
  //   %1 = ...
  //   %2 = ...
  //   %5.1 = aten::relu(%1)
  //   %5.2 = aten::relu(%2)
  //   %3 = prim::ListConstruct(%5.1, %5.2)
  //   %4 = aten::cat(%3, ...)
  //   return (%4)

  // 检查 aten::cat 的输出是否被使用
  TORCH_INTERNAL_ASSERT(
      cat->output()->hasUses(),
      buildErrorMessage("aten::cat output is not used."));

  // 检查 aten::cat 的输出是否在多个地方使用
  TORCH_INTERNAL_ASSERT(
      cat->output()->uses().size() == 1,
      buildErrorMessage("aten::cat output is used in multiple places."));

  // 检查 aten::cat 的输入是否为 prim::ListConstruct 操作
  TORCH_INTERNAL_ASSERT(
      cat->input(0)->node()->kind() == prim::ListConstruct,
      buildErrorMessage("aten::cat inputs are not expected."));

  // 获取 aten::cat 的输入列表节点
  auto cat_list = cat->input(0)->node();
  auto cat_inputs = cat_list->inputs();

  // 获取用户节点输出的张量类型
  auto user_tensor_type = user->output()->type()->cast<c10::TensorType>();

  // 检查用户节点输出的张量类型是否存在
  TORCH_INTERNAL_ASSERT(
      user_tensor_type, buildErrorMessage("Unexpected user tensor type"));

  // 创建新的 aten::cat 的输入映射
  std::unordered_map<Value*, Value*> new_cat_inputs;
  for (auto inp : cat_inputs) {
    // 克隆用户节点，并替换其中的 aten::cat 输出为当前输入
    auto new_cat_input = subgraph->createClone(
        user, [&](Value* k) { return (k == cat->output()) ? inp : k; });

    // 确保新输入的张量类型与原输入类型的标量类型相同
    auto input_tensor_type = inp->type()->cast<c10::TensorType>();
    TORCH_INTERNAL_ASSERT(
        input_tensor_type, buildErrorMessage("Unexpected input tensor type"));
    auto new_input_type =
        input_tensor_type->withScalarType(user_tensor_type->scalarType());
    new_cat_input->output()->setType(new_input_type);

    // 将新节点插入到 prim::ListConstruct 之前
    new_cat_input->insertBefore(cat_list);
    new_cat_inputs[inp] = new_cat_input->output();
  }

  // 创建新的 prim::ListConstruct 节点，使用新的输入映射
  auto new_cat_list = subgraph->createClone(
      cat_list, [&](Value* k) { return new_cat_inputs[k]; });
  new_cat_list->insertBefore(cat);

  // 创建新的 aten::cat 节点，并替换其中的 prim::ListConstruct 输出
  auto new_cat = subgraph->createClone(cat, [&](Value* k) {
    return (k == cat_list->output()) ? new_cat_list->output() : k;
  });
  new_cat->output()->setType(user_tensor_type);
  new_cat->insertBefore(cat);

  // 替换用户节点的输出，并销毁原用户节点
  user->output()->replaceAllUsesWith(new_cat->output());
  user->destroy();

  // 检查 aten::cat 的输出是否再次被使用，若无，则销毁该节点
  TORCH_INTERNAL_ASSERT(
      !cat->output()->hasUses(),
      buildErrorMessage("aten::cat output is not used."));
  cat->destroy();

  // 如果 prim::ListConstruct 的输出未被使用，则销毁该节点
  if (!cat_list->output()->hasUses()) {
    cat_list->destroy();
  }

  // 返回新的 aten::cat 节点
  return new_cat;
}
// 计算给定节点的输入中张量类型的数量
static int numTensorInputs(Node* node) {
  int count = 0;  // 初始化计数器为0
  for (auto v : node->inputs()) {  // 遍历节点的输入
    if (v->type()->cast<c10::TensorType>()) {  // 如果输入是张量类型
      ++count;  // 计数器加1
    }
  }
  return count;  // 返回张量类型输入的数量
}

// 如果给定的 `cat` 节点促使类型提升，则返回 true
// 如果 `cat` 的输入具有不同的类型，则期望 `cat` 的实现将提升类型。
static bool doesCatPromoteTypes(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == aten::cat,
      buildErrorMessage("Graph node is not aten::cat."));  // 断言节点为 aten::cat 类型

  TORCH_INTERNAL_ASSERT(
      node->input(0)->node()->kind() == prim::ListConstruct,
      buildErrorMessage("aten::cat inputs are not expected."));  // 断言输入是 prim::ListConstruct 类型

  auto inputs = node->input(0)->node()->inputs();  // 获取 ListConstruct 的输入
  TORCH_INTERNAL_ASSERT(
      !inputs.empty(), buildErrorMessage("Empty inputs of ListConstruct"));  // 断言输入不为空

  auto scalar_type =
      inputs.front()->type()->cast<c10::TensorType>()->scalarType();  // 获取第一个输入的标量类型

  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inp_scalar_type =
        inputs[i]->type()->cast<c10::TensorType>()->scalarType();  // 获取其他输入的标量类型
    if (scalar_type != inp_scalar_type) {  // 如果存在不同的标量类型
      return true;  // 返回 true
    }
  }
  return false;  // 所有输入的标量类型相同，返回 false
}

// 将给定的 `aten::cat` 操作的用户移动到其输入之后。
// 需要满足以下约束条件：cat 操作及其用户。
//   * cat 操作应该只有一个使用者。
//   * 用户应该是逐元素操作。
//   * 用户应该只有一个张量输入。
//     - 如果用户有 > 1 个张量输入，则该用户操作不能应用于 cat 的输入，
//       因为其他张量输入将不会被拆分，因此这些张量的形状将与 cat 的输入不匹配。
//       例如：
//           %1 = ...
//           %2 = ...
//           %3 = prim::ListConstruct([%1, %2])
//           %4 = aten::cat(%3, ...)
//           %5 = aten::add(%4, %0)
//       在这个例子中，我们不能将 `aten::add` 移动到 `aten::cat` 的输入 `%1` 和 `%2`，
//       因为 `%0` 的形状将不同。
//    * cat 操作不提升类型。
//      - 当 cat 操作提升类型时，移动它的用户后，输入到 cat 的类型需要反映原始类型。
//        这目前没有处理。TODO
static void moveCatOpToEnd(Node* cat, std::shared_ptr<Graph> subgraph) {
  TORCH_INTERNAL_ASSERT(
      cat->kind() == aten::cat,
      buildErrorMessage("Graph node is not aten::cat."));  // 断言节点为 aten::cat 类型

  if (cat->output()->uses().size() == 1) {  // 如果 cat 操作仅有一个使用者
    auto use = cat->output()->uses().front();  // 获取使用者的信息
    if (get_tensorexpr_elementwise_set().contains(use.user) &&  // 如果使用者是逐元素操作
        numTensorInputs(use.user) == 1) {  // 如果使用者只有一个张量输入
      if (!doesCatPromoteTypes(cat)) {  // 如果 cat 操作不提升类型
        TORCH_INTERNAL_ASSERT(
            use.user->output()->owningGraph() == subgraph.get(),
            buildErrorMessage(
                "aten::cat user graph does not math the given subgraph."));  // 断言用户操作所在的图与给定的子图匹配

        auto new_cat = moveCatAfterUse(cat, use.user, subgraph);  // 在使用者后移动 cat 操作
        moveCatOpToEnd(new_cat, subgraph);  // 递归调用，处理新的 cat 操作
      }
    }
  }
}
// 将`aten::cat`操作节点移到其可能的输入位置的函数
static void moveCatOpsToEnd(std::shared_ptr<Graph> subgraph) {
  // 存储所有`aten::cat`操作节点的列表
  std::vector<Node*> cat_nodes;
  // 遍历子图中的所有节点
  for (Node* n : subgraph->nodes()) {
    // 如果节点的类型是`aten::cat`
    if (n->kind() == aten::cat) {
      // 将该节点添加到`cat_nodes`列表中
      cat_nodes.push_back(n);
    }
  }
  // 遍历所有找到的`aten::cat`节点，并将它们移动到子图的末尾
  for (auto cat : cat_nodes) {
    moveCatOpToEnd(cat, subgraph);
  }
}

// 优化`aten::cat`操作的函数，如果成功优化则返回true，否则返回false
bool OptimizeCat(const std::shared_ptr<Graph>& graph) {
  // 检查是否需要进行没有条件的`aten::cat`操作优化
  if (getCatWoConditionals()) {
    // 移动所有`aten::cat`操作到图的末尾
    moveCatOpsToEnd(graph);
    return true;
  }
  return false;
}

// 根据示例输入形状注释图的输入类型
void annotateInputShapes(
    const std::shared_ptr<Graph>& graph,
    const std::vector<std::optional<at::Tensor>>& example_inputs) {
  // 断言给定的输入与图的输入匹配
  TORCH_INTERNAL_ASSERT(
      graph->inputs().size() == example_inputs.size(),
      buildErrorMessage("Given inputs do not match the fuser graph inputs."));
  // 遍历所有示例输入
  for (size_t idx = 0; idx < example_inputs.size(); idx++) {
    // 如果示例输入存在
    if (auto t = example_inputs[idx]) {
      // 获取当前执行上下文中的具体张量类型
      auto concrete_tensor_type = tensorTypeInCurrentExecutionContext(*t);
      // 将图的输入类型设置为具体的张量类型
      graph->inputs().at(idx)->setType(concrete_tensor_type);
    }
  }
}

// 移除未使用的self参数并返回更新后的图
std::shared_ptr<Graph> removeUnusedSelfArgument(
    const std::shared_ptr<Graph>& graph) {
  // 如果图的输入为空，则直接返回图
  if (graph->inputs().empty()) {
    return graph;
  }
  // 获取图的第一个输入作为self参数
  jit::Value* self_argument = graph->inputs().at(0);
  // 如果self参数被使用或者不是模块类型，则返回图
  if (!self_argument->uses().empty() || !self_argument->type()->is_module()) {
    return graph;
  }
  // 删除图的第一个输入（即self参数）
  graph->eraseInput(0);
  return graph;
}

// 使图中的形状尺寸变为符号化形式，并返回新的符号形状
std::vector<int64_t> makeShapesSymbolic(
    std::shared_ptr<Graph>& graph,
    const std::vector<int64_t>& size_vals) {
  // 存储图中所有值的集合
  std::unordered_set<Value*> values;
  // 收集所有输入和输出值
  for (auto v : graph->inputs()) {
    values.insert(v);
  }
  for (auto v : graph->outputs()) {
    values.insert(v);
  }
  // 遍历图中的所有节点，收集它们的输入和输出值
  for (auto n : graph->nodes()) {
    for (auto v : n->inputs()) {
      values.insert(v);
    }
    for (auto v : n->outputs()) {
      values.insert(v);
    }
  }
  // 存储原始大小到符号大小的映射
  std::unordered_map<int64_t, int64_t> shape_to_sym_shape;
  // 存储新的符号化尺寸
  std::vector<int64_t> new_syms;
  // 遍历给定的尺寸值列表
  for (int64_t size_val : size_vals) {
    // 创建新的形状符号并记录映射关系
    auto new_shape_symbol = at::ShapeSymbol::newSymbol().value();
    shape_to_sym_shape[size_val] = new_shape_symbol;
    new_syms.push_back(new_shape_symbol);
    // 向图中添加一个名为"sym_shape"的输入，类型为IntType
    graph->addInput("sym_shape")->setType(IntType::get());
  }

  // 遍历图中所有值
  for (auto v : values) {
    // 如果值不是张量类型，则跳过
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    // 获取值的张量类型
    auto tt = v->type()->expect<TensorType>();
    // 如果张量类型具有符号大小
    if (!tt->symbolic_sizes().sizes()) {
      continue;
    }
    // 获取当前张量的符号大小向量
    std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();

    // 更新张量的大小为符号化形式
    auto new_sizes = c10::fmap(shape_vec, [&](const at::ShapeSymbol& shape) {
      auto value = shape.value();
      // 如果该大小在映射中有对应的符号形式，则使用符号形式替换
      if (shape_to_sym_shape.count(value)) {
        return shape_to_sym_shape.at(value);
      }
      return value;
    });
    // 设置张量类型为更新后的符号形状
    v->setType(tt->withSymbolicShapes(c10::SymbolicShape(new_sizes)));
  }

  return new_syms;
}

// 检查图是否可编译
bool isGraphCompilable(const std::shared_ptr<Graph>& graph) {
  // 遍历图的所有输入
  for (auto input : graph->inputs()) {
    auto const& t = input->type();
    // 如果输入不是张量类型，则返回false
    if (!t->cast<TensorType>()) {
      return false;
    }

    auto const& t = input->type();
    // 如果输入不是张量类型，则返回false
    if (!t->cast<TensorType>()) {
      return false;
    }
  }
  // 如果所有输入都是张量类型，则返回true
  return true;
}
    // 获取类型指针 t 的种类
    auto const& k = t->kind();
    // 如果类型不是 TensorType、FloatType、BoolType 或 IntType 中的任何一种，则输出调试信息并返回 false
    if (k != TypeKind::TensorType && k != TypeKind::FloatType &&
        k != TypeKind::BoolType && k != TypeKind::IntType) {
      GRAPH_DEBUG("Input %", input->debugName(), " has unsupported type ", *t);
      return false;
    }
  }

  // 遍历图中的每个节点
  for (auto n : graph->nodes()) {
    // 遍历节点 n 的所有输入
    for (auto v : n->inputs()) {
      // 获取输入节点的类型 t
      auto const& t = v->type();
      // 如果类型是 TensorType
      if (t->kind() == TypeKind::TensorType) {
        // 尝试将类型转换为 TensorType
        auto tt = t->cast<TensorType>();
        // 如果转换成功且不是完整的张量类型，则输出调试信息并返回 false
        if (!tt->isComplete()) {
          GRAPH_DEBUG(
              "%",
              v->debugName(),
              " is not a complete tensor! The type is: ",
              *t);
          return false;
        }
      }
    }
    // 遍历节点 n 的所有输出
    for (auto v : n->outputs()) {
      // 获取输出节点的类型 t
      auto const& t = v->type();
      // 如果类型是 TensorType
      if (t->kind() == TypeKind::TensorType) {
        // 尝试将类型转换为 TensorType
        auto tt = t->cast<TensorType>();
        // 如果转换成功且不是完整的张量类型，则输出调试信息并返回 false
        if (!tt->isComplete()) {
          GRAPH_DEBUG(
              "%", v->debugName(), " is not a complete! The type is: ", *t);
          return false;
        }
      }
    }
  }

  // TODO: 检查所有节点是否有降级实现（lowerings）
  // 若未实现所有节点的降级，返回 true
  return true;
}

static void fixupTypeInfoForValue(
    Value* v,
    std::optional<at::ScalarType> scalar_type,
    std::optional<at::Device> device) {
  Node* n = v->node();  // 获取值所属节点
  auto const& t = v->type();  // 获取值的类型
  if (t->kind() != TypeKind::TensorType) {  // 如果值不是张量类型，直接返回
    return;
  }

  if (n->kind() == prim::Constant) {  // 如果节点是常量节点
    auto const_tensor = toIValue(v)->toTensor();  // 将值转换为张量
    auto concrete_tensor_type =
        tensorTypeInCurrentExecutionContext(const_tensor);  // 获取该张量在当前执行上下文中的具体类型
    v->setType(concrete_tensor_type);  // 设置值的类型为具体类型
    return;
  }

  TensorTypePtr new_tt;
  auto tt = t->cast<TensorType>();  // 将值的类型转换为张量类型
  auto sizes = tt->sizes();  // 获取张量的大小
  if (!sizes.concrete_sizes()) {  // 如果张量的大小不是具体值
    GRAPH_DEBUG("No concrete sizes for %", v->debugName());  // 记录调试信息，张量没有具体的大小
    return;
  }
  auto strides = tt->strides();  // 获取张量的步幅
  auto dtype = tt->scalarType() ? tt->scalarType() : scalar_type;  // 获取张量的数据类型，或使用传入的数据类型
  auto concrete_sizes = *sizes.concrete_sizes();  // 获取张量的具体大小
  auto concrete_strides = strides.concrete_sizes()
      ? *strides.concrete_sizes()
      : TensorType::contiguousStridesOf(concrete_sizes);  // 获取张量的具体步幅
  new_tt = TensorType::create(
      dtype, device, concrete_sizes, concrete_strides, false);  // 创建新的张量类型

  v->setType(new_tt);  // 设置值的类型为新创建的张量类型
}

static std::optional<at::ScalarType> inferScalarType(Node* n) {
  std::optional<at::ScalarType> scalar_type;  // 推断的标量类型
  for (auto v : n->inputs()) {  // 遍历节点的输入值
    auto const& t = v->type();  // 获取值的类型
    if (t->kind() == TypeKind::TensorType) {  // 如果值的类型是张量类型
      auto tt = t->cast<TensorType>();  // 将类型转换为张量类型
      if (!scalar_type) {
        scalar_type = tt->scalarType();  // 如果还没有推断过标量类型，使用当前张量的标量类型
      }
      if (tt->scalarType() && *tt->scalarType() != scalar_type) {
        GRAPH_DEBUG(
            "Inputs of ", n, " have different scalar types, cannot fixup!");  // 记录调试信息，输入的张量类型不同，无法修复
        return c10::nullopt;  // 返回空值表示无法推断标量类型
      }
    }
  }
  return scalar_type;  // 返回推断出的标量类型
}

static std::optional<at::Device> inferDevice(Node* n) {
  std::optional<at::Device> device;  // 推断的设备类型
  for (auto v : n->inputs()) {  // 遍历节点的输入值
    auto const& t = v->type();  // 获取值的类型
    if (t->kind() == TypeKind::TensorType) {  // 如果值的类型是张量类型
      auto tt = t->cast<TensorType>();  // 将类型转换为张量类型
      if (!device) {
        device = tt->device();  // 如果还没有推断过设备类型，使用当前张量的设备类型
      }
      if (tt->device() && *tt->device() != device) {
        GRAPH_DEBUG("Inputs of ", n, " have different devices, cannot fixup!");  // 记录调试信息，输入的张量设备不同，无法修复
        return c10::nullopt;  // 返回空值表示无法推断设备类型
      }
    }
  }
  if (!device) {
    device = at::kCPU;  // 如果没有推断出设备类型，默认使用 CPU
  }
  return device;  // 返回推断出的设备类型
}

void fixupMissingShapeInfo(const std::shared_ptr<Graph>& graph) {
  for (auto input : graph->inputs()) {  // 遍历图的输入节点
    auto const& t = input->type();  // 获取输入节点的类型
    if (t->kind() == TypeKind::TensorType) {  // 如果类型是张量类型
      auto tt = t->cast<TensorType>();  // 将类型转换为张量类型
      if (!tt->scalarType()) {  // 如果张量类型没有指定数据类型
        GRAPH_DEBUG("No dtype for %", input->debugName());  // 记录调试信息，张量没有数据类型
        return;
      }
      fixupTypeInfoForValue(
          input, *tt->scalarType(), tt->device() ? *tt->device() : at::kCPU);  // 修复输入节点的类型信息
    }
  }

  for (auto n : graph->nodes()) {  // 遍历图中的所有节点
    std::optional<at::ScalarType> scalar_type = inferScalarType(n);  // 推断节点的标量类型
    std::optional<at::Device> device = inferDevice(n);  // 推断节点的设备类型

    for (auto v : n->outputs()) {  // 遍历节点的输出值
      fixupTypeInfoForValue(v, scalar_type, device);  // 修复输出值的类型信息
    }
  }
}

std::shared_ptr<Graph> removeGraphOutput(
    const std::shared_ptr<Graph>& graph,
    // 调用图形对象的eraseOutput方法，删除指定索引处的输出
    graph->eraseOutput(idx);
    // 返回更新后的图形对象指针
    return graph;
}

std::shared_ptr<Graph> replaceListOutputWithTuple(
    const std::shared_ptr<Graph>& graph) {
  // 获取图的第一个输出节点
  auto out = graph->outputs()[0];
  // 获取输出节点对应的图节点
  auto out_node = out->node();
  // 如果输出节点的类型不是 ListConstruct，则直接返回原图
  if (out_node->kind() != prim::ListConstruct) {
    return graph;
  }
  // 创建一个新的 Tuple 节点，使用 ListConstruct 节点的输入作为 Tuple 的输入
  auto tuple_node = graph->createTuple(out_node->inputs());
  // 将 Tuple 节点插入到 ListConstruct 节点之后
  tuple_node->insertAfter(out_node);
  // 替换输出节点的所有使用为 Tuple 节点的输出
  out->replaceAllUsesWith(tuple_node->output());
  // 返回修改后的图
  return graph;
}

static bool trimGraphOnce(const std::shared_ptr<Graph>& graph) {
  // 获取图的返回节点
  Node* ret = graph->return_node();
  // 将图的输入转换为无序集合
  std::unordered_set<Value*> graph_inputs(
      graph->inputs().begin(), graph->inputs().end());
  // 将图的输出转换为无序集合
  std::unordered_set<Value*> outputs(
      graph->outputs().begin(), graph->outputs().end());
  // 标记图是否发生了变化
  bool changed = false;
  // 遍历返回节点的输入
  for (size_t idx = 0; idx < ret->inputs().size(); idx++) {
    auto v = ret->inputs()[idx];
    // 如果输入是图的输入之一，则继续下一次循环
    if (graph_inputs.count(v)) {
      continue;
    }
    // 删除图的输出中的指定索引，并将产生该值的节点的所有输入添加到图的输出中
    graph->eraseOutput(idx);
    for (auto v_ins : v->node()->inputs()) {
      // 如果输入已经是图的输出之一，则继续下一次循环
      if (outputs.count(v_ins)) {
        continue;
      }
      // 如果输入节点的类型是常量，则继续下一次循环
      if (v_ins->node()->kind() == prim::Constant) {
        continue;
      }
      // 将输入节点注册为图的输出
      graph->registerOutput(v_ins);
    }
    // 标记图已经发生变化
    changed = true;
    // 跳出循环
    break;
  }
  // 返回图是否发生了变化
  return changed;
}

static std::shared_ptr<Graph> dequantizeResults(
    const std::shared_ptr<Graph>& graph) {
  // 遍历图的所有输出节点
  for (auto v : graph->outputs()) {
    // 获取输出节点的类型
    auto& t = v->type();
    // 如果类型是 TensorType
    if (t->kind() == TypeKind::TensorType) {
      auto tt = t->cast<TensorType>();
      // 如果标量类型为空或者不是量化整数类型，则继续下一次循环
      if (!tt->scalarType() || !c10::isQIntType(*tt->scalarType())) {
        continue;
      }
      // 创建一个 dequantize 节点，使用输出节点作为输入
      Node* deq = graph->create(aten::dequantize, {v});
      // 将 dequantize 节点添加到图中
      graph->appendNode(deq);
      // 设置 dequantize 节点的输出类型为浮点数
      deq->output()->setType(tt->withScalarType(c10::kFloat));
      // 将输出节点的所有使用替换为 dequantize 节点的输出
      v->replaceAllUsesAfterNodeWith(deq, deq->output());
    }
  }
  // 返回修改后的图
  return graph;
}

std::shared_ptr<Graph> trimGraph(
    const std::shared_ptr<Graph>& graph,
    int64_t iters) {
  // 标记图是否发生了变化
  bool changed = true;
  // 初始化迭代次数
  int64_t iter = 0;
  // 当图发生变化且未达到迭代次数上限时，执行循环
  while (changed && iter++ < iters) {
    // 调用 trimGraphOnce 函数，尝试精简图结构一次，并更新 changed 变量
    changed = trimGraphOnce(graph);
    // 删除图块中的死代码
    EliminateDeadCode(graph->block());
  }
  // 执行结果去量化处理
  dequantizeResults(graph);
  // 返回修改后的图
  return graph;
}

} // namespace torch::jit::tensorexpr
```