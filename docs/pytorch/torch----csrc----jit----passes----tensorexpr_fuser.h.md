# `.\pytorch\torch\csrc\jit\passes\tensorexpr_fuser.h`

```py
#pragma once

#include <torch/csrc/Export.h>  // Torch导出宏定义
#include <torch/csrc/jit/ir/ir.h>  // Torch JIT IR模块
#include <memory>  // 内存管理模块

namespace torch {
namespace jit {

// 运行基于TensorExpressions的融合器
// 如果add_composed_op为true，创建一个单一操作，既执行类型对齐的运行时检查
// 又分派到内核/未优化图表
TORCH_API void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,  // 图形对象的共享指针
    size_t min_group_size = 2,  // 最小组大小，默认为2
    bool add_composed_op = false,  // 是否添加复合操作，默认为false
    bool fuse_to_dynamic_shapes = false);  // 是否融合到动态形状，默认为false

TORCH_API void setTensorExprFuserEnabled(bool val);  // 设置Tensor表达式融合器的启用状态
TORCH_API bool tensorExprFuserEnabled();  // 查询Tensor表达式融合器的启用状态
TORCH_API void setTensorExprDynamicShapeFusionEnabled(bool val);  // 设置动态形状融合器的启用状态
TORCH_API bool tensorExprDynamicShapeFusionEnabled();  // 查询动态形状融合器的启用状态
TORCH_API bool setTexprReductionsEnabled(bool value);  // 设置Texpr缩减的启用状态
TORCH_API bool texprReductionsEnabled();  // 查询Texpr缩减的启用状态

TORCH_API void RemoveProfileNodesAndSpecializeTypes(
    std::shared_ptr<Graph>& graph);  // 移除剖析节点并特化类型
TORCH_API bool hasTensorTypeSpecialization(Value* v);  // 检查值是否具有张量类型特化
TORCH_API void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph);  // 移除图形中的张量类型特化
TORCH_API void removeTensorTypeSpecializations(Block* block);  // 移除块中的张量类型特化

using tensor_type_converter_t =  // 张量类型转换器类型定义
    c10::function_ref<TensorTypePtr(const TensorTypePtr& t)>;

// 插入类型检查模式
//
// 在具有Subgraph属性的守护节点周围插入模式
//
//   if TypeCheck(...):
//     guarded_node
//   else:
//     FallbackGraph(...)
//
// TypeCheck包括所有Tensor输入类型，由type_converter处理，一个lambda
// TensorTypePtr(const TensorTypePtr& t)。这允许擦除类型的不相关方面。
//
// Fallback图将具有与守护节点相同的子图（预期守护节点的子图将被优化）。
TORCH_API void insertTypeGuard(
    Node* guarded_node,  // 被守护的节点
    tensor_type_converter_t type_converter,  // 类型转换器
    c10::Symbol kind);  // 符号标识

TORCH_API bool usedOnlyInSize(Value* v);  // 检查值是否仅在大小中使用
TORCH_API Value* broadcastSizes(at::ArrayRef<Value*> sizes, AliasDb* db);  // 广播大小值

namespace tensorexpr {
TORCH_API bool isSupported(Node* node);  // 检查节点是否受支持

/// 获取可修改的自定义运算符集对象。
///
/// 对于静态形状，如果自定义运算符已添加到自定义运算符集中，它将被拉入NNC融合组。
/// 但是对于动态形状，除非通过`torch::jit::RegisterShapeComputeGraphForSchema`显式注册形状函数，否则不起作用。
///
/// @return 自定义运算符集的引用
///
TORCH_API OperatorSet& getCustomOperatorSet();
} // namespace tensorexpr

} // namespace jit
} // namespace torch
```