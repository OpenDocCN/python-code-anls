# `.\pytorch\torch\csrc\jit\passes\symbolic_shape_runtime_fusion.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <torch/csrc/Export.h>
// 引入导出宏定义，用于在编译时控制符号的导出和导入

#include <torch/csrc/jit/ir/ir.h>
// 引入 Torch JIT 的 IR 头文件，用于表示和操作计算图中的节点

#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
// 引入 Torch JIT 的符号形状分析的头文件，用于在计算图中进行符号形状的推断

#include <unordered_map>
// 引入标准库中的无序映射容器，用于存储不同类型的键值对，提供常数时间的平均访问复杂度

namespace torch {
namespace jit {

// 命名空间 torch::jit，包含 Torch 的 JIT 编译器的所有功能和实现

// Takes in a TensorExprGraph of static shapes and generalizes the input shapes
// to symbolic dimensions. Dimensions of value 1 will be preserved, otherwise
// dimensions with the same value will be bucketed to the same symbolic shape.
// E.g. Tensor(5, 3), Tensor(3, 1) -> Tensor(SS(-1), SS(-2)), Tensor(SS(-2), 1)
// From there, runs symbolic shape inference on the graph, and creates a
// versioning if in the graph with prim::TensorExprDynamicGuard checking if
// the inputs at runtime match the Generalized Symbolic Shapes that are inputs
// to the TE Kernel. The computate to calculate all symbolic dimensions is
// inlined in to the if block with the TE Kernel. All Sym Dim Value* are
// appended to the end of the TE Kernel Graph/Node inputs, and the Node is
// augmented with a integer list attr `symbolic_shape_inputs` that gives the
// mapping from Value * -> Symbolic Shape int64_t value. For more lengthy IR
// examples and walkthrough look at ShapeAnalysisTest.DynamicShapesFusion in
// `test_shape_analysis` Returns True on Success, False on Failure, can fail if
// shape propagation fails to propagate # of dims or if complete shapes on
// inputs not set

// 接受静态形状的 TensorExprGraph 并将输入形状推广为符号维度。值为 1 的维度将被保留，
// 否则具有相同值的维度将被分配到相同的符号形状中。例如，Tensor(5, 3), Tensor(3, 1)
// -> Tensor(SS(-1), SS(-2)), Tensor(SS(-2), 1)。然后，在图上运行符号形状推断，并在
// 图中创建一个版本控制，使用 prim::TensorExprDynamicGuard 检查运行时输入是否匹配
// 作为 TE Kernel 输入的广义符号形状。计算所有符号维度的过程嵌入到带有 TE Kernel 的
// if 块中。所有 Sym Dim Value* 都附加到 TE Kernel 图/节点输入的末尾，并使用整数列表
// 属性 `symbolic_shape_inputs` 扩展节点，该属性提供从 Value* -> 符号形状 int64_t 值的映射。
// 有关更详细的 IR 示例和步骤说明，请查看 `test_shape_analysis` 中的 ShapeAnalysisTest.DynamicShapesFusion。
// 成功时返回 True，失败时返回 False，如果形状传播未能传播维数或输入上未设置完整形状，可能会失败。

TORCH_API bool GenerateGuard(
    Node* tensorexpr_graph_node,
    bool add_composed_op = false);
// 生成保护条件的函数声明，接受一个 TensorExprGraph 节点和一个布尔值参数

TORCH_API void runTensorExprDynamicGroup(const Code& code, Stack& stack);
// 运行 TensorExprDynamicGroup 的函数声明，接受 Code 对象和 Stack 引用作为参数

enum class StrideInput {
  // 列出枚举类 StrideInput 的成员
  // Tensors natively store whether they are contiguous or not as a property
  // this makes it faster to query `is_contiguous` or
  // `is_contiguous(memory_format=channels_last)`
  // than looping through the sizes/strides yourself
  // For tensors with these properties, we only store one value:
  TENSOR_CONT,  // 张量具有连续性属性，通过此枚举表示
  TENSOR_CONT_CHANNELS_LAST,  // 张量具有通道为最后内存格式的属性，通过此枚举表示
  // now, we describe other cases, where there is one stride enum
  // per dimension
  S_ONE,        // STRIDE_ONE: packed，表示每个维度都具有相同步幅的情况
  S_CONT,       // STRIDE_CONTIGUOUS: stride[i + 1] * sizes[i + 1]，表示连续步幅的情况
  S_TRAN_CONT,  // STRIDE_TRANSPOSED_CONTIGUOUS: stride[i-1] * sizes[i-1]，表示转置连续步幅的情况
  S_AS_ARG,     // STRIDE_AS_ARG: stride passed in as runtime value，表示运行时传入的步幅值的情况
};

TORCH_API std::string toString(StrideInput si);
// 将 StrideInput 枚举类型转换为字符串的函数声明，接受一个 StrideInput 枚举对象作为参数

TORCH_API StrideInput strideInputFromString(const std::string& si);
// 将字符串转换为 StrideInput 枚举类型的函数声明，接受一个 const 字符串引用作为参数

} // namespace jit
} // namespace torch
// 命名空间结束，包含了 Torch JIT 编译器的所有功能和实现
```