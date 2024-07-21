# `.\pytorch\torch\csrc\lazy\ts_backend\ts_node.h`

```
#pragma once

#include <c10/util/ArrayRef.h>  // 引入ArrayRef库，用于处理数组的引用
#include <torch/csrc/jit/api/function_impl.h>  // 引入function_impl.h，提供JIT函数实现的API
#include <torch/csrc/jit/ir/ir.h>  // 引入ir.h，包含JIT的中间表示(IR)的定义
#include <torch/csrc/lazy/backend/lowering_context.h>  // 引入lowering_context.h，提供后端降低上下文的定义
#include <torch/csrc/lazy/core/ir.h>  // 引入ir.h，包含Lazy Tensor的IR定义
#include <torch/csrc/lazy/core/shape.h>  // 引入shape.h，包含Lazy Tensor的形状定义
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>  // 引入ts_lowering_context.h，提供TS后端降低上下文的定义

namespace torch {
namespace lazy {

using TSOpVector = std::vector<torch::jit::Value*>;  // 使用TSOpVector作为torch::jit::Value*的向量类型别名

// TsNode类继承自Node类，用于表示Lazy Tensor的操作节点
class TORCH_API TsNode : public lazy::Node {
 public:
  // 构造函数，初始化TsNode对象
  TsNode(
      OpKind op,
      OpList operands,
      std::vector<Shape>&& shapes,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  // 构造函数，初始化TsNode对象
  TsNode(
      OpKind op,
      OpList operands,
      const std::function<Shape()>& shape_fn,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  // 构造函数，初始化TsNode对象
  TsNode(
      OpKind op,
      OpList operands,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  // 构造函数，初始化TsNode对象
  TsNode(
      OpKind op,
      Shape shape,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  // 析构函数，默认析构TsNode对象
  ~TsNode() override = default;

  // 返回节点的哈希值
  hash_t hash() const override;

  // 返回节点形状的哈希值
  hash_t shapeHash() const override;

  // 获取Python调用栈信息的字符串表示
  const std::string getPythonStacktrace() const;

  // Lower方法是一个后端特定的方法，返回一个后端特定的类型
  virtual TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const;

 private:
  // 用于形状缓存的DAG（有尺寸信息）
  hash_t shape_hash_;
  // 用于查找编译图的DAG哈希值
  // 如果启用动态形状，则使用没有尺寸信息的DAG哈希，否则使用带有尺寸信息的DAG哈希
  hash_t dag_hash_;
};

// 注意：这里的OpKind与ltc_ops.h中的不同，否则会出现循环导入
const OpKind tensor_list_opkind = OpKind::Get("lazy_tensors::tensor_list");

// TensorList代表at::TensorList，它是一个vector[Tensor]，同时也是第一类IValue，
// 可以作为TS程序的单个输入。在Lazy Tensor代码中，如果将TensorList表示为单个节点，
// 则更容易处理TensorLists。
//
// 注意：TensorList的形状未定义。在某些地方，我们断言#shapes与#outputs匹配，
// 这源于当前所有IR节点都表示张量（此IR没有类型系统）的事实。因此，TensorList有点像一个hack。
//
// TODO(whc) 一旦Shape() API移至Node基类，也将其作为虚函数实现为NotImplemented，修复断言可能失败的问题。
struct TORCH_API TensorList : public TsNode {
  static OpKind ClassOpKind() {
  // 返回存储在 tensor_list_opkind 中的操作类型列表
  return tensor_list_opkind;
}

// 默认构造函数被删除，禁止创建 TensorList 实例没有参数的实例

// 构造函数，接受一个 OpList 类型的值作为参数，并将其转换为 TensorList
TensorList(OpList values);

// 判断当前 TensorList 对象的操作数是否可以被给定的 OpList values 重用
bool CanBeReused(OpList values) const {
  // 比较当前对象的操作数是否与 values 中的值相等，返回比较结果
  return operands() == std::vector<Output>(values.begin(), values.end());
}

// 实现 TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function, TSLoweringContext* loctx) 方法
TSOpVector Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const override;
};

// 结束 lazy 命名空间定义

} // namespace lazy
// 结束 torch 命名空间定义
} // namespace torch
```