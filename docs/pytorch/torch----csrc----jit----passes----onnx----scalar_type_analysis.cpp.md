# `.\pytorch\torch\csrc\jit\passes\onnx\scalar_type_analysis.cpp`

```
// 包含C++头文件：范围操作（例如irange）
#include <c10/util/irange.h>
// 包含Torch的JIT日志相关头文件
#include <torch/csrc/jit/jit_log.h>
// 包含Torch的死代码消除相关头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 包含Torch的ONNX帮助函数相关头文件
#include <torch/csrc/jit/passes/onnx/helper.h>
// 包含Torch的ONNX标量类型分析相关头文件
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>

// Torch命名空间
namespace torch {
// Torch JIT命名空间
namespace jit {

// Torch JIT ONNX命名空间
namespace onnx {
// 使用c10::onnx命名空间中的全部内容
using namespace ::c10::onnx;
}

// Torch JIT内部匿名命名空间
namespace {

// 定义ONNX操作集版本号为14
const int ONNX_OPSET_14 = 14;

// 将Torch标量类型映射到对应的ONNX类型的无序映射表
static const std::unordered_map<c10::ScalarType, int, ScalarTypeHashFunction>
    scalarTypeToONNXTypeMap = {
        {c10::kFloat, 1},
        {c10::kByte, 2},
        {c10::kChar, 3},
        {c10::kShort, 5},
        {c10::kInt, 6},
        {c10::kLong, 7},
        {c10::kBool, 9},
        {c10::kHalf, 10},
        {c10::kDouble, 11},
        {c10::kQInt8, 12},
        {c10::kQUInt8, 13},
        {c10::kQInt32, 14},
        {c10::kBFloat16, 15},
        {c10::kFloat8_e4m3fn, 16},
        {c10::kFloat8_e5m2, 17},
        {c10::kFloat8_e4m3fnuz, 18},
        {c10::kFloat8_e5m2fnuz, 19},
};

// 根据Torch标量类型获取对应的ONNX类型
static int64_t ScalarTypeToONNXType(const c10::ScalarType& st) {
  int64_t onnx_type = -1;
  // 查找给定标量类型在映射表中的对应ONNX类型
  const auto it = scalarTypeToONNXTypeMap.find(st);
  if (it != scalarTypeToONNXTypeMap.end()) {
    // 如果找到对应映射，则返回其ONNX类型
    onnx_type = it->second;
  }
  return onnx_type;
}

// 对于这些操作符，所有输入和输出都共享相同的标量类型。
// 不需要特殊处理每个操作符的情况。
static const std::unordered_set<NodeKind> standardOps = {
    onnx::Add,
    onnx::Concat,
    onnx::Div,
    onnx::Gemm,
    onnx::Min,
    onnx::Max,
    onnx::Mod,
    onnx::Mul,
    onnx::Pow,
    onnx::Sub,
    onnx::MatMul,
    onnx::Conv,
};

// 对于这些操作符，所有输入共享相同的标量类型。
// 输出标量类型始终为Bool。
static const std::unordered_set<NodeKind> comparisonOps = {
    onnx::Greater,
    onnx::Less,
    onnx::Equal,
    onnx::GreaterOrEqual,
    onnx::LessOrEqual,
};

// 对于这些操作符，包含选择逻辑的操作。
static const std::unordered_set<NodeKind> selectorOps = {onnx::Where};

// 检查给定操作符是否为标准操作符
static bool IsStandardOp(const NodeKind& nkind) {
  // 判断操作符是否在标准操作集中
  return standardOps.find(nkind) != standardOps.end();
}

// 检查给定操作符是否为比较操作符
static bool IsComparisonOp(const NodeKind& nkind) {
  // 判断操作符是否在比较操作集中
  return comparisonOps.find(nkind) != comparisonOps.end();
}

// 检查给定操作符是否为选择操作符
static bool IsSelectorOp(const NodeKind& nkind) {
  // 判断操作符是否在选择操作集中
  return selectorOps.find(nkind) != selectorOps.end();
}

// 创建具有指定标量类型的配置文件化张量类型
static TensorTypePtr CreateProfiledTensorTypeWithScalarType(
    const TensorTypePtr& typePtr,
    const c10::ScalarType& scalar_type) {
  // 内部断言：类型指针不能为空
  TORCH_INTERNAL_ASSERT(typePtr != nullptr);
  // 返回带有指定标量类型的类型指针
  return typePtr->withScalarType({scalar_type});
}

// 检查隐式转换是否支持给定的操作符
static bool IsImplicitCastSupported(const NodeKind& nodeKind) {
  // 支持标准操作、比较操作和选择操作的隐式转换
  return IsStandardOp(nodeKind) || IsComparisonOp(nodeKind) ||
      IsSelectorOp(nodeKind);
}

// 提升标量类型之间和张量之间的类型
static std::optional<c10::ScalarType> PromoteScalarTypes(
    const std::vector<c10::ScalarType>& types) {
  if (types.empty()) {
    // 如果类型列表为空，则返回空optional
    return c10::nullopt;
  }
  auto st = types[0];
  // 对类型列表中的每一个类型进行提升操作
  for (const auto i : c10::irange(1, types.size())) {
    st = c10::promoteTypes(st, types[i]);
  }
  // 返回最终提升后的类型
  return st;
}

// 标量和张量之间的类型提升
// 根据输入的张量类型和标量类型，推断并提升标量类型，优先级参考标量类型的类别
static std::optional<c10::ScalarType> PromoteScalarTypesWithCategory(
    const std::vector<c10::ScalarType>& typesFromTensors,
    const std::vector<c10::ScalarType>& typesFromScalars) {
  // 推断张量类型的标量类型
  auto typeFromTensor = PromoteScalarTypes(typesFromTensors);
  // 推断标量类型
  auto typeFromScalar = PromoteScalarTypes(typesFromScalars);

  // 获取标量类型的类别
  auto getTypeCategory = [](c10::ScalarType t) {
    if (c10::kBool == t) {
      return 1;
    }
    if (c10::isIntegralType(t, /*includeBool=*/false)) {
      return 2;
    }
    if (c10::isFloatingType(t)) {
      return 3;
    }
    return 0;
  };

  // 如果标量类型推断失败，则返回张量类型的标量类型
  if (c10::nullopt == typeFromScalar) {
    return typeFromTensor;
  } else if (c10::nullopt == typeFromTensor) {
    return typeFromScalar;
  }

  // 获取张量类型和标量类型的类别
  auto typeCategoryFromTensor = getTypeCategory(typeFromTensor.value());
  auto typeCategoryFromScalar = getTypeCategory(typeFromScalar.value());

  // 根据标量类型的类别判断优先级，选择更高类别的标量类型
  if (typeCategoryFromScalar > typeCategoryFromTensor) {
    return typeFromScalar;
  }
  return typeFromTensor;
}

// 推断节点期望的标量类型
static std::optional<c10::ScalarType> InferExpectedScalarType(const Node* n) {
  std::vector<c10::ScalarType> typesFromTensors;
  std::vector<c10::ScalarType> typesFromScalars;

  auto get_scalar_type =
      [](const Value* input) -> std::optional<at::ScalarType> {
    if (auto* tensor_type = input->type()->castRaw<TensorType>()) {
      return tensor_type->scalarType();
    }
    // 对于比较操作，始终将标量类型提升到输入中最高的标量类型，无论该输入是张量还是标量。
    typesFromScalars.insert(
        typesFromScalars.end(),
        typesFromTensors.begin(),
        typesFromTensors.end());
    st = PromoteScalarTypes(typesFromScalars);
  } else {
    if (output_st) {
      // 如果输出标量类型可用，则使用它
      st = output_st;
    } else {
      // PyTorch 现在无论输入是张量还是标量都会进行隐式类型提升。（以前只有标量支持隐式类型转换）。
      // 根据张量和标量推断并提升标量类型，优先级参考标量类型的类别
      st = PromoteScalarTypesWithCategory(typesFromTensors, typesFromScalars);
    }
  }

  return st;
}

// 为标准操作进行低精度类型转换
static std::optional<c10::ScalarType> LowPrecisionCastForStandardOps(
    const Node* n,
    const c10::ScalarType& scalar_type) {
  // 一些标准操作在 ONNX opset 版本 < 14 不支持 uint8、int8、int16 类型。
  // 在这个 ONNX PR 中修复：https://github.com/onnx/onnx/pull/3334
  if (n->kind() != onnx::Gemm && IsStandardOp(n->kind()) &&
      (scalar_type == c10::kByte || scalar_type == c10::kChar ||
       scalar_type == c10::kShort)) {
    return c10::kLong;
  }
  return scalar_type;
}

// 更新节点输入的标量类型
static void UpdateScalarTypeForInputs(
    Node* n,
    const c10::ScalarType& scalar_type) {
  // 将标量类型转换为对应的 ONNX 类型
  const int64_t onnx_type = ScalarTypeToONNXType(scalar_type);
  // 如果无法将标量类型转换为对应的 ONNX 类型
  if (onnx_type < 0) {
    TORCH_WARN(
        "ONNX Scalar Type Analysis - Scalar type: ",
        c10::toString(scalar_type),
        " of input tensor in operator: ",
        n->kind().toDisplayString(),
        " not supported in ONNX. ");
    return;
  }


# 发出警告信息，指示在 ONNX 中不支持当前操作符中输入张量的标量类型
size_t input_idx = 0;
for (auto input : n->inputs()) {
  // 获取输入节点的张量类型，并提取其标量类型（如果有）
  auto input_tensor_type = input->type()->cast<TensorType>();
  auto input_scalar_type =
      input_tensor_type ? input_tensor_type->scalarType() : c10::nullopt;

  // 对于 onnx:Where 操作符，跳过第一个输入（即条件输入）
  if (IsSelectorOp(n->kind()) && input_idx == 0) {
    input_idx++;
    continue;
  }

  // 如果输入是常量节点或者其标量类型与期望的标量类型不符
  if ((input->node()->kind() == onnx::Constant) ||
      (input_scalar_type && (*input_scalar_type != scalar_type))) {
    if (input->node()->kind() == onnx::Constant) {
      // 直接修复标量，而不是插入一个类型转换操作符
      at::Tensor val = input->node()->t(attr::value);
      at::Tensor new_val = val.to(scalar_type);
      // 创建一个常量节点，并将修复后的标量值赋给它
      Node* const_node = n->owningGraph()->create(onnx::Constant);
      const_node->t_(attr::value, new_val);
      const_node->insertBefore(n);
      const_node->output()->setType(TensorType::create(new_val));
      const_node->copyMetadata(n);
      // 将原始节点的输入替换为修复后的常量节点的输出
      n->replaceInputWith(input, const_node->output());
    } else {
      // 创建一个类型转换节点
      Node* cast_node = n->owningGraph()->create(onnx::Cast);
      cast_node->addInput(input);
      cast_node->i_(attr::to, onnx_type);
      cast_node->insertBefore(n);
      // 设置类型转换节点的输出类型
      cast_node->output()->setType(CreateProfiledTensorTypeWithScalarType(
          input_tensor_type, scalar_type));
      cast_node->copyMetadata(n);
      // 将原始节点的输入替换为类型转换节点的输出
      n->replaceInputWith(input, cast_node->output());
    }
  }

  // 增加输入索引，以处理下一个输入节点
  input_idx++;
}
// 结束函数 UpdateScalarTypeForOutput
static void UpdateScalarTypeForOutput(
    Node* n,
    const c10::ScalarType& scalar_type) {
  // 检查输出节点是否是 TensorType 类型
  if (auto output_tensor_type = n->output()->type()->cast<TensorType>()) {
    // 如果是，设置输出节点的类型为经过 scalar_type 标量类型处理后的 ProfiledTensorType
    n->output()->setType(CreateProfiledTensorTypeWithScalarType(
        output_tensor_type, scalar_type));
  }
}

// 开始函数 RecoverScalarTypeForOutput
static void RecoverScalarTypeForOutput(
    Value* out,
    const c10::ScalarType& scalar_type) {
  // 获取值所在的节点
  Node* n = out->node();
  // 内部断言，确保节点不为空
  TORCH_INTERNAL_ASSERT(nullptr != n);
  // 将 scalar_type 转换为对应的 ONNX 类型
  const int64_t onnx_type = ScalarTypeToONNXType(scalar_type);
  // 在节点所属的图中创建一个 Cast 节点
  Node* cast_node = n->owningGraph()->create(onnx::Cast, 1);
  // 将当前值作为输入添加到 Cast 节点中
  cast_node->addInput(out);
  // 设置 Cast 节点的 to 属性为 onnx_type
  cast_node->i_(attr::to, onnx_type);
  // 将 Cast 节点插入到当前节点之后
  cast_node->insertAfter(n);
  // 复制当前节点的元数据到 Cast 节点
  cast_node->copyMetadata(n);
  // 替换所有使用当前值的节点，以使用 Cast 节点的输出
  out->replaceAllUsesAfterNodeWith(cast_node, cast_node->output());
}

// 开始函数 LowPrecisionCastNodeForStandardOps
// 此例中的错误是在使用 uint8 类型进行 add 操作时发现的
// Reference Link: https://github.com/huggingface/transformers/blob/b020a736c374460af1b34267283f957988350630/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L936
static void LowPrecisionCastNodeForStandardOps(Node* n, int opset_version) {
  // 内部断言，确保输出节点的数量为 1
  TORCH_INTERNAL_ASSERT(n->outputs().size() == 1);
  // 检查输出节点是否是 TensorType 类型，并且标量类型不为空
  if (n->output()->type()->cast<TensorType>() == nullptr ||
      n->output()->type()->cast<TensorType>()->scalarType() == c10::nullopt) {
    // 如果输出类型为 null，则跳过低精度转换
    return;
  }
  // 获取输出节点的标量类型
  auto output_scalar_type =
      n->output()->type()->cast<TensorType>()->scalarType().value();
  // 遍历所有输入节点
  for (size_t i = 0; i < n->inputs().size(); ++i) {
    // 检查输入节点是否是 TensorType 类型，并且标量类型不为空
    if (n->input(i)->type()->cast<TensorType>() == nullptr ||
        n->input(i)->type()->cast<TensorType>()->scalarType() == c10::nullopt) {
      // 如果输入类型为 null，则跳过低精度转换
      return;
    }
    // 获取输入节点的标量类型
    auto input_tensor_type =
        n->input(i)->type()->cast<TensorType>()->scalarType().value();
    // 内部断言，确保输入和输出节点的标量类型相同
    TORCH_INTERNAL_ASSERT(output_scalar_type == input_tensor_type);
  }

  // 如果 opset 版本小于 14，问题将在 ONNX opset 14 中修复
  if (opset_version < ONNX_OPSET_14) {
    // 对标准操作进行低精度转换
    auto expected_scalar_type_cast =
        LowPrecisionCastForStandardOps(n, output_scalar_type);
    // 更新输入节点的标量类型
    UpdateScalarTypeForInputs(n, *expected_scalar_type_cast);
    // 如果输出标量类型发生变化，将其恢复为原始类型
    if (output_scalar_type != *expected_scalar_type_cast) {
      RecoverScalarTypeForOutput(n->output(), output_scalar_type);
    }
  }
}

// 开始函数 ImplicitCastNodeForONNX
static void ImplicitCastNodeForONNX(Node* n) {
  // 检查是否支持隐式类型转换
  if (IsImplicitCastSupported(n->kind())) {
    # 推断节点 n 的预期标量类型
    auto expected_scalar_type = InferExpectedScalarType(n);
    # 如果成功推断出预期标量类型
    if (expected_scalar_type) {
      # 更新输入节点的标量类型为推断得到的预期标量类型
      UpdateScalarTypeForInputs(n, *expected_scalar_type);
      # 如果节点 n 不是比较操作符
      if (!IsComparisonOp(n->kind())) {
        # 更新节点 n 的输出标量类型为推断得到的预期标量类型
        UpdateScalarTypeForOutput(n, *expected_scalar_type);
      }
    }
  }
} // 结束匿名命名空间

static void ImplicitCastForONNX(Block* block) {
  // 遍历给定 block 中的所有节点
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    // 递归处理每个子块中的节点
    for (auto sub : it->blocks()) {
      ImplicitCastForONNX(sub);
    }

    // 对当前节点进行隐式类型转换处理
    ImplicitCastNodeForONNX(*it);
  }
  // 在删除带有副作用节点时允许删除死代码
  EliminateDeadCode(
      block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

static void LowPrecisionCastForStandardOpsONNX(
    Block* block,
    int opset_version) {
  // 遍历给定 block 中的所有节点
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    // 递归处理每个子块中的节点
    for (auto sub : it->blocks()) {
      LowPrecisionCastForStandardOpsONNX(sub, opset_version);
    }

    // 如果当前节点是标准操作，进行低精度类型转换处理
    if (IsStandardOp(it->kind())) {
      LowPrecisionCastNodeForStandardOps(*it, opset_version);
    }
  }
  // 在删除带有副作用节点时允许删除死代码
  EliminateDeadCode(
      block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

} // 结束匿名命名空间

void ScalarTypeAnalysisForONNX(
    const std::shared_ptr<Graph>& graph,
    bool lowprecision_cast,
    int opset_version) {
  // 打印输出 graph 在进行标量类型分析之前的状态
  GRAPH_DUMP("Before ScalarTypeAnalysisForONNX: ", graph);
  // 执行隐式类型转换处理
  ImplicitCastForONNX(graph->block());
  // 如果需要进行低精度类型转换，则执行
  if (lowprecision_cast) {
    LowPrecisionCastForStandardOpsONNX(graph->block(), opset_version);
  }
  // 打印输出 graph 在进行标量类型分析之后的状态
  GRAPH_DUMP("After ScalarTypeAnalysisForONNX: ", graph);
}

void ScalarTypeAnalysisNodeForONNX(Node* n) {
  // 对单个节点执行隐式类型转换处理
  ImplicitCastNodeForONNX(n);
}
```