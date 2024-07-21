# `.\pytorch\torch\csrc\jit\passes\onnx\helper.cpp`

```
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/onnx/back_compat.h>

#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/unsqueeze.h>
#endif

#include <onnx/onnx_pb.h>

namespace torch {
namespace jit {
namespace onnx {
using namespace ::c10::onnx;

} // namespace onnx

// 构建从输入值到参数对的映射，根据参数字典
ValueToParamPairMap buildValueToParamsMap(
    Block* b,
    const ParamMap& paramsDict) {
  ValueToParamPairMap valsToParamsMap;
  // 遍历块的输入值
  for (auto& input : b->inputs()) {
    // 根据输入值的调试名称查找在参数字典中的匹配项
    auto it = paramsDict.find(input->debugName());
    // 如果找到匹配项，将输入值和参数对插入映射中
    if (it != paramsDict.end()) {
      valsToParamsMap.emplace(input, *it);
    }
  }
  // 返回构建好的映射
  return valsToParamsMap;
}

// 删除块中未使用的输入
void eraseUnusedBlockInputs(Block* b) {
  // 逆向遍历块的输入
  for (size_t i_1 = b->inputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    // 如果输入没有被使用，则从块中删除该输入
    if (!b->inputs().at(i)->hasUses()) {
      b->eraseInput(i);
    }
  }
}

// 从映射中删除未使用的值
void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap) {
  auto it = valsToParamsMap.begin();
  // 遍历映射中的每一对
  while (it != valsToParamsMap.end()) {
    // 如果值没有被使用，则从映射中删除这对
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } else {
      ++it;
    }
  }
}

// 根据从输入值到参数对的映射构建参数字典
void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict) {
  // 清空原有的参数字典
  paramsDict.clear();
  // 将映射中每一对名称-张量参数对插入参数字典
  for (const auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}

// 将 ONNX 的数据类型转换为 ATen 的标量类型
std::optional<at::ScalarType> ONNXTypeToATenType(int32_t onnx_type) {
  // 根据 ONNX 数据类型的枚举值执行转换
  switch (onnx_type) {
    case ::ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      return at::ScalarType::Undefined;
    case ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return at::kFloat;
    case ::ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return at::kByte;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return at::kChar;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return at::kShort;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return at::kInt;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return at::kLong;
    case ::ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return at::kBool;
    case ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return at::kHalf;
    case ::ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return at::kDouble;
    case ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:
      return at::kComplexFloat;
    case ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
      return at::kComplexDouble;
    case ::ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return at::kBFloat16;
    case ::torch::onnx::TensorProto_DataType_FLOAT8E5M2:
      return at::kFloat8_e5m2;
    case ::torch::onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
      return at::kFloat8_e5m2fnuz;
    case ::torch::onnx::TensorProto_DataType_FLOAT8E4M3FN:
      return at::kFloat8_e4m3fn;
    case ::torch::onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
      return at::kFloat8_e4m3fnuz;


这些注释解释了 C++ 代码中每个函数和相应的代码行的作用和功能。
    default:
      // 如果输入的 ONNX 类型没有匹配到任何已知的类型，触发 TORCH_CHECK 错误
      TORCH_CHECK(
          false,
          "ONNX type ",
          onnx_type,
          " is an unexpected tensor scalar type");
  }
  // 返回空的 std::optional<at::ScalarType>，表示没有找到匹配的类型
  return std::optional<at::ScalarType>{};
}

// 将节点添加到块中，使用给定的符号和输入数组
Node* addNodeToBlock(Block* block, Symbol kind, ArrayRef<Value*> inputs) {
  // 在块中追加一个新节点，节点类型由符号决定
  auto new_node = block->appendNode(block->owningGraph()->create(kind));
  // 遍历输入数组，将每个输入添加到新节点中
  for (auto input : inputs) {
    new_node->addInput(input);
  }
  // 返回新创建的节点
  return new_node;
}

// 将输入添加到块中
Value* addInputToBlock(Block* block) {
  // 在块中添加一个新的输入，并返回该输入
  return block->addInput();
}

// 匿名命名空间，用于实现将 ATen 数据类型转换为 ONNX 数据类型的辅助函数
namespace {
::ONNX_NAMESPACE::TensorProto_DataType ATenTypeToOnnxType_aux(
    at::ScalarType at_type) {
  // 根据 ATen 的标量类型转换为对应的 ONNX 数据类型
  switch (at_type) {
    case at::kDouble:
      return ::ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    case at::kFloat:
      return ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    case at::kHalf:
      return ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    case at::kByte:
      return ::ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    case at::kChar:
      return ::ONNX_NAMESPACE::TensorProto_DataType_INT8;
    case at::kShort:
      return ::ONNX_NAMESPACE::TensorProto_DataType_INT16;
    case at::kInt:
      return ::ONNX_NAMESPACE::TensorProto_DataType_INT32;
    case at::kLong:
      return ::ONNX_NAMESPACE::TensorProto_DataType_INT64;
    case at::kBool:
      return ::ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    case at::kQInt8:
      return ::ONNX_NAMESPACE::TensorProto_DataType_INT8;
    case at::kQUInt8:
      return ::ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    case at::kQInt32:
      return ::ONNX_NAMESPACE::TensorProto_DataType_INT32;
    default:
      // 如果遇到未预期的标量类型，抛出错误
      TORCH_CHECK(
          false,
          "ScalarType ",
          toString(at_type),
          " is an unexpected tensor scalar type");
  }
}
} // namespace

// 将 ATen 数据类型转换为 ONNX 数据类型的函数
int ATenTypeToOnnxType(at::ScalarType at_type) {
  // 调用辅助函数，并将结果转换为整数返回
  return static_cast<int>(ATenTypeToOnnxType_aux(at_type));
}

// 创建一个 ONNX 的 Unsqueeze 节点
Node* createONNXUnsqueeze(
    Graph* graph,
    Node* n_to_insert_before,
    Value* input,
    int axis,
    int opset_version) {
  // 创建一个 Unsqueeze 节点，设置输出张量的维度
  Node* unsqueeze_node = graph->create(onnx::Unsqueeze, 1);
  unsqueeze_node->addInput(input);
  unsqueeze_node->insertBefore(n_to_insert_before);
  if (opset_version >= OPSET_VERSION_13) {
    // 对于 opset 版本大于等于 13，按照 ONNX 规范设置 axes 输入
    Node* unsqueeze_axes = graph->create(onnx::Constant, 1);
    unsqueeze_axes->insertBefore(unsqueeze_node);
    unsqueeze_axes->t_(
        attr::value, at::unsqueeze(at::scalar_to_tensor(at::Scalar(axis)), 0));
    unsqueeze_node->addInput(unsqueeze_axes->output());
  } else {
    // 对于 opset 版本小于 13，按照 ONNX 规范设置 axes 属性
    unsqueeze_node->is_(attr::axes, {0});
  }
  // 返回创建的 Unsqueeze 节点
  return unsqueeze_node;
}

// 创建一个 ONNX 的 Constant 节点
Node* createONNXConstant(
    Graph* graph,
    Node* n_to_insert_before,
    at::Tensor value) {
  // 创建一个 Constant 节点，并设置其输出张量的值
  Node* constant_node = graph->create(onnx::Constant, 1);
  constant_node->insertBefore(n_to_insert_before);
  constant_node->t_(attr::value, std::move(value));
  // 返回创建的 Constant 节点
  return constant_node;
}

// 检查节点是否可以转换为 ONNX 的 Concat 节点
bool isValidToTransformToONNXConcatNode(Node* lc_node) {
  // 检查节点的输入是否为空，以确定是否可以进行转换
  return !lc_node->inputs().empty();
}

// 将节点转换为 ONNX 的 Concat 节点
Node* transformToONNXConcatNode(
    Graph* g,
    Node* lc_node,
    bool need_new_input,
    int opset_version) {
  // ListConstruct Int[] output case, we need to transform to ONNX
  // Concat to ensure the output is a single tensor(dynamic) type in
  // order to be consumed as inputs
  // 创建一个空的 Value 指针向量 unsqueezed，用于存储处理后的输入节点
  std::vector<Value*> unsqueezed;
  // 根据需要是否添加新的输入节点，选择使用 g 的返回节点或者 lc_node
  auto new_node = need_new_input ? g->return_node() : lc_node;

  // 遍历 lc_node 的所有输入节点
  for (auto* input : lc_node->inputs()) {
    // 如果需要添加新的输入节点
    auto new_input =
        need_new_input ? g->addInput()->copyMetadata(input) : input;
    // 检查新输入节点的类型是否为 TensorType
    if (auto type = new_input->type()->cast<TensorType>()) {
      // 如果类型为 TensorType 且维度为 1
      if (type->dim() && type->dim() == 1U) {
        // 将该输入节点添加到 unsqueezed 中
        unsqueezed.emplace_back(new_input);
        continue;
      }
    }
    // 如果不是 1 维张量，则创建一个 unsqueeze 节点，确保沿 dim-0 维度都是 1
    Node* unsqueezed_node =
        createONNXUnsqueeze(g, new_node, new_input, 0, opset_version);
    // 复制 lc_node 的元数据到 unsqueezed_node
    unsqueezed_node->copyMetadata(lc_node);
    // 将 unsqueezed_node 的输出添加到 unsqueezed 中
    unsqueezed.emplace_back(unsqueezed_node->output());
  }

  // 创建一个 concat_node，用于将 unsqueezed 中的所有节点沿 axis=0 进行连接
  Node* concat_node = need_new_input
      ? g->insertNode(g->create(onnx::Concat, 1))
      : g->create(onnx::Concat, 1)->insertBefore(lc_node);
  concat_node->i_(attr::axis, 0);
  // 将 unsqueezed 中的所有节点作为 concat_node 的输入
  for (auto v : unsqueezed) {
    concat_node->addInput(v);
  }

  // 返回 concat_node，作为处理后的节点
  return concat_node;
}
} // namespace jit
} // namespace torch

void ONNXLintGraph(
    const Block* b,
    std::vector<NodeKind>& n_miss_source_range,
    std::vector<NodeKind>& n_miss_scope) {
  // 遍历当前块 b 中的每个节点 n
  for (const auto* n : b->nodes()) {
    // 遍历节点 n 中的每个子块 sub_b，递归调用 ONNXLintGraph 函数
    for (const auto* sub_b : n->blocks()) {
      ONNXLintGraph(sub_b, n_miss_source_range, n_miss_scope);
    }

    // 检查节点 n 的源范围是否为 nullptr，若是则记录节点类型到 n_miss_source_range
    if (nullptr == n->sourceRange().source()) {
      GRAPH_DEBUG("Node does not set sourceRange:", *n);
      n_miss_source_range.emplace_back(n->kind());
    }
    // 检查节点 n 的作用域名称是否为空，若是则记录节点类型到 n_miss_scope
    if (n->scopeName().empty()) {
      GRAPH_DEBUG("Node does not set scope:", *n);
      n_miss_scope.emplace_back(n->kind());
    }
  }
}

void ONNXLintGraph(const std::shared_ptr<Graph>& graph) {
  // 打印未设置作用域和源范围的节点信息
  std::vector<NodeKind> n_miss_source_range, n_miss_scope;
  ONNXLintGraph(graph->block(), n_miss_source_range, n_miss_scope);
  
  // Lambda 函数，统计特定节点类型的数量
  auto count_const = [](const std::vector<NodeKind>& vec) -> size_t {
    size_t count = 0;
    for (auto k : vec) {
      switch (k) {
        case prim::Constant:
        case prim::ListConstruct:
        case onnx::Constant:
          count++;
          break;
      }
    }
    return count;
  };
  
  // 统计未设置源范围和作用域的常量节点数量
  auto const_count_src = count_const(n_miss_source_range);
  auto const_count_scope = count_const(n_miss_scope);
  
  // 输出缺失源范围的节点信息
  GRAPH_UPDATE(
      "Missing source range.\n",
      "Total ",
      n_miss_source_range.size(),
      " nodes. Including ",
      const_count_src,
      " constants.");
  
  // 输出缺失作用域的节点信息
  GRAPH_UPDATE(
      "Missing scope.\n",
      "Total ",
      n_miss_scope.size(),
      " nodes. Including ",
      const_count_scope,
      " constants.");
}
```