# `.\pytorch\torch\csrc\jit\passes\onnx\helper.h`

```
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Utility functions for PyTorch to ONNX conversion.

// 定义一系列常量，表示支持的不同 ONNX Opset 版本
static const int OPSET_VERSION_1 = 1;
static const int OPSET_VERSION_9 = 9;
static const int OPSET_VERSION_10 = 10;
static const int OPSET_VERSION_11 = 11;
static const int OPSET_VERSION_12 = 12;
static const int OPSET_VERSION_13 = 13;
static const int OPSET_VERSION_14 = 14;
static const int OPSET_VERSION_15 = 15;
static const int OPSET_VERSION_16 = 16;

// 定义一个映射，用于将 Value 指针映射到参数名和 IValue 对的 pair
using ValueToParamPairMap = std::map<Value*, std::pair<std::string, IValue>>;

// 定义一个映射，将参数名映射到 IValue
using ParamMap = std::map<std::string, IValue>;

// 从 ValueToParamPairMap 构建 ParamMap
TORCH_API void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);

// 从 Block 和 ParamMap 构建 ValueToParamPairMap
TORCH_API ValueToParamPairMap
buildValueToParamsMap(Block* b, const ParamMap& paramsDict);

// 从 ValueToParamPairMap 中删除未使用的值
TORCH_API void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap);

// 从 Block 中删除未使用的输入
TORCH_API void eraseUnusedBlockInputs(Block* b);

// 从 ValueToParamPairMap 构建 ParamMap
TORCH_API void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);

// 向 Block 中添加一个新节点
TORCH_API Node* addNodeToBlock(
    Block* block,
    Symbol kind,
    ArrayRef<Value*> inputs);

// 向 Block 中添加一个新输入，并返回这个新输入的指针
TORCH_API Value* addInputToBlock(Block* block);

// 将 ONNX 类型转换为 ATen 类型
TORCH_API std::optional<at::ScalarType> ONNXTypeToATenType(int32_t onnx_type);

// 将 ATen 类型转换为 ONNX 类型
TORCH_API int ATenTypeToOnnxType(at::ScalarType at_type);

// 对 Graph 执行 ONNX lint 操作
TORCH_API void ONNXLintGraph(const std::shared_ptr<Graph>& graph);

// 创建一个 ONNX Unsqueeze 节点
Node* createONNXUnsqueeze(
    Graph* graph,
    Node* n_to_insert_before,
    Value* input,
    int axis,
    int opset_version);

// 创建一个 ONNX Constant 节点
Node* createONNXConstant(
    Graph* graph,
    Node* n_to_insert_before,
    at::Tensor value);

// 检查是否可以将节点转换为 ONNX Concat 节点
bool isValidToTransformToONNXConcatNode(Node* lc_node);

// 将节点转换为 ONNX Concat 节点
Node* transformToONNXConcatNode(
    Graph* graph,
    Node* lc_node,
    bool need_new_input,
    int opset_version);

// 定义一个哈希函数，用于 ScalarType 类型
class ScalarTypeHashFunction {
 public:
  size_t operator()(const c10::ScalarType& type) const {
    return static_cast<size_t>(type);
  }
};

} // namespace jit
} // namespace torch
```