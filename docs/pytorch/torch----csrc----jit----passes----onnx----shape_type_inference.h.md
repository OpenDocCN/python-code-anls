# `.\pytorch\torch\csrc\jit\passes\onnx\shape_type_inference.h`

```
#pragma once

#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 的 JIT IR 模块头文件
#include <torch/csrc/jit/passes/onnx/helper.h>  // 包含 Torch 的 ONNX 辅助函数头文件
#include <torch/csrc/jit/python/python_arg_flatten.h>  // 包含 Torch 的 Python 参数扁平化头文件

#include <utility>  // 包含标准工具类的头文件

namespace torch {
namespace jit {

// 合并现有类型和推断类型。
// 返回 {合并后的类型, 是否使用了推断类型}。
//
// 推断类型优先级较高，因为它是由 ONNX 形状推断生成的，更兼容 ONNX。
// 在 ONNX 形状推断未能生成完整推断类型或生成不完整推断类型的情况下，参考现有类型并填补缺失部分。
// 目前支持以下情况：
//  1. 现有类型: Tensor[], 推断类型: Tensor[]
//    对于张量列表，现有类型不存储内部张量的数据类型或形状，因此推断类型始终包含更多信息，并返回它。
//  2. 现有类型: Tensor, 推断类型: Tensor
//    从现有类型填补推断类型的缺失信息（形状、数据类型）。
//  3. 现有类型: Scalar[], 推断类型: Tensor
//    ONNX 用 1 维张量表示标量列表。返回推断类型，因为它与 ONNX 更兼容。
std::pair<TypePtr, bool> MergeInferredType(
    TypePtr existing_type,
    TypePtr inferred_type);

// 合并推断类型并设置映射关系到目标值。
void MergeInferredTypeAndSetMap(
    Value* dest_v,
    TypePtr existing_type,
    TypePtr inferred_type);

// 使用动态轴信息更新图输入类型。
// 标记为动态的轴将被分配为动态 ShapeSymbol。
// 如果在 dynamic_axes 中定义多个轴共享相同的 ShapeSymbol，这也是可能的。
TORCH_API void ONNXSetDynamicInputShape(
    std::shared_ptr<Graph>& graph,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    const std::vector<std::string>& input_names);

// 使用输出 Tensor 的类型更新图输出。
// 如果 onnx_shape_inference 为 true，则将输出 Tensor 的类型与推断类型比较并合并。
// 推断类型可能包含动态轴，因此优先于输出 Tensor 的类型。
TORCH_API void ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    const python::IODescriptor& desc,
    bool onnx_shape_inference,
    bool is_script,
    int opset_version);

// 如果是脚本模型，在输出中用 Optional 节点替换 None（opset > 15）。
// 当比较 PyTorch 结果和 ONNX 结果的输出格式时，这有助于对齐，因为它们对输出中的 None 有不同的处理方式。
void ReplaceGraphOutputNoneWithOptional(
    std::shared_ptr<Graph>& graph,
    size_t outputs_index);

// 为 None 创建 Optional 节点（opset > 15）。
Node* ONNXOptionalNodeForNone(std::shared_ptr<Graph>& graph);

// 使用 ONNX 形状推断工具对节点进行推断。
// 节点必须具有 ONNX 命名空间，并且必须是符合规范的有效 ONNX 节点。
// 在成功的ONNX形状推断运行后，函数使用推断的形状和类型更新节点n的输出类型。
// 否则，节点n保持不变。
TORCH_API void ONNXShapeTypeInference(
    Node* n,
    const ParamMap& params_dict,
    int opset_version);

// 使用ONNX形状推断来处理图形。
// 内部为每个节点调用ONNXShapeTypeInference，以便在遇到非法节点时仅跳过该节点，
// 而不是跳过整个图形，以提高覆盖率。
TORCH_API void ONNXShapeTypeInference(
    std::shared_ptr<Graph>& g,
    const ParamMap& params_dict,
    int opset_version);

// 检查图形g的所有输入是否都是静态的。
bool AllGraphInputsStatic(const Graph* g);

// 检查节点n的输入是否可靠或静态。
std::pair<bool, bool> AreInputsReliableOrStatic(Node* n);

// 更新torch::jit::Value类型的输出的可靠性信息。
void UpdateReliable(
    torch::jit::Value* output,
    const std::pair<bool, bool>& input_reliable,
    bool no_type_warning = false);

// 更新torch::jit::Node的可靠性信息。
void UpdateReliable(torch::jit::Node* n);

// 如果输出是可靠的，则更新torch::jit::Value的形状常量。
void UpdateShapeConstantIfReliable(torch::jit::Value* output);

} // namespace jit
} // namespace torch
```