# `.\pytorch\torch\csrc\jit\runtime\static\passes.h`

```
// 包含 Torch 的 JIT IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>

// 命名空间 torch::jit 中的函数定义
namespace torch::jit {

// 为稀疏神经网络融合推断操作
TORCH_API void FuseInferenceOpsForSparseNN(
    std::shared_ptr<torch::jit::Graph>& graph);

// 消除平凡的等分操作
TORCH_API void EliminateTrivialEquallySplit(
    std::shared_ptr<torch::jit::Graph>& graph);

// 融合列表解包操作
TORCH_API void FuseListUnpack(std::shared_ptr<torch::jit::Graph>& graph);

// 如果 outputs_are_immutable 设置为 false，则不用复制版本替换生成别名的视图操作
TORCH_API void ReplaceWithCopy(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool outputs_are_immutable = true);

// 将 Permute 操作替换为复制操作
TORCH_API void ReplacePermuteWithCopy(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool outputs_are_immutable = true);

// 可能替换为复制操作
TORCH_API void ReplaceWithMaybeCopy(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool outputs_are_immutable = true);

// 移除不可变输入字典查找
TORCH_API void RemoveImmutableInputDictLookups(
    std::shared_ptr<torch::jit::Graph>& graph);

// 检查图中是否存在指定操作
TORCH_API bool graphHasOp(std::shared_ptr<Graph>& graph, const char* op_name);

// 检查模块前向是否存在指定操作
TORCH_API bool forwardHasOp(const Module& module, const char* op_name);

// 融合 Sign 和 Log1P 操作
TORCH_API void FuseSignLog1P(std::shared_ptr<Graph>& graph);

// 使用可变元组解包
TORCH_API void UseVariadicTupleUnpack(const std::shared_ptr<Graph>& graph);

// 根据限定字符串创建符号
inline c10::Symbol fromQualString(const std::string& qual_string) {
  return c10::Symbol::fromQualString(qual_string);
}

// [为特殊值创建拥有的引用]
// StaticRuntimeBlockRunner 在 run_impl 结束时将其输出移动到返回值中。
// 然而，存在一个特殊情况可能会引起问题。如果返回的是一个常量，
// 那么 constants_ 数组中唯一的引用可能会被这次移动销毁。
// 我们可以在 run_impl 中添加特殊逻辑来处理这种情况。但由于这是一个相对罕见的角落情况，
// 更简单的方法是添加一个仅创建输入的拥有引用但不做其他操作的操作符。
// 这个拥有的引用可以安全地从 StaticRuntimeBlockRunner 中移出。
// 注意，对于标量来说，实际上这是一个复制操作。
// 如果在子块的外部范围返回值，我们也必须执行同样的操作。
TORCH_API void CreateOwnedRefsForSpecialValues(Graph& graph);

// [强制非空输出]
// 子块在某些情况下可能不返回任何值。对于 StaticRuntimeBlockRunner 来说，这是一个问题，
// 因为它假定至少返回一个输出。我们不想为这种角落情况添加特殊逻辑以减慢 SR 的执行速度，
// 因此我们简单地强制不返回任何内容的块返回 None。
TORCH_API void ForceNonEmptyOutputs(Graph& graph);

// 使用变长分组访问器
TORCH_API void UseVariadicGroupedAccessor(const std::shared_ptr<Graph>& graph);

// 消除额外的 Permute 操作
TORCH_API void EliminateExtraPermuteOps(std::shared_ptr<Graph>& graph);

// 消除无操作的 Slice 操作
TORCH_API void EliminateNoOpSlice(std::shared_ptr<Graph>& graph);

// 使用 Split 和 Squeeze 操作
TORCH_API void UseSplitAndSqueeze(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
// 移除不必要的输出
// 用于减少计算量，当后续图形中不使用时。
// 当前用于移除embedding_bag的max_indices输出，这对于计算主输出不是必需的。
TORCH_API void RemoveUnnecessaryOutputs(std::shared_ptr<Graph>& graph);

// 移除不必要的embedding_bag输出
// 用于减少计算量，从图中删除不必要的embedding_bag操作的输出。
TORCH_API void RemoveUnnecessaryEmbeddingBagOutputs(std::shared_ptr<Graph>& graph);

// 融合ClampNaNToNum操作
// 将图中的ClampNaNToNum操作进行融合优化。
TORCH_API void FuseClampNaNToNum(std::shared_ptr<Graph>& graph);

// 使用InPlaceGetRealInputsFromOptionalInputsV2
// 从可选输入中获取实际输入，进行优化。
TORCH_API void UseInPlaceGetRealInputsFromOptionalInputsV2(std::shared_ptr<Graph>& graph);

// 预打包权重
// 对图中的权重进行预打包优化处理。
TORCH_API void PrepackWeights(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
```