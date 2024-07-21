# `.\pytorch\torch\csrc\jit\passes\dtype_analysis.cpp`

```py
// 包含 ATen 库的头文件：函数 schema、JIT 类型、符号
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>

// 包含 c10 库的头文件：标量类型、ArrayRef、Optional
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

// 包含 Torch JIT 库的头文件：别名分析、IR、日志、dtype 分析、操作注册
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dtype_analysis.h>
#include <torch/csrc/jit/passes/utils/op_registry.h>

// 包含 Torch 库
#include <torch/library.h>

// 根据 AT_PER_OPERATOR_HEADERS 的定义决定使用哪个 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

// 包含标准库头文件
#include <algorithm> // 包含用于算法操作的标准库
#include <memory>    // 包含智能指针 std::unique_ptr
#include <stdexcept> // 包含异常处理的标准库

// Torch JIT 命名空间
namespace torch {
namespace jit {

// 匿名命名空间，用于限制符号作用域
namespace {

// 使用 Tensor 和 ScalarType 别名
using Tensor = at::Tensor;
using ScalarType = at::ScalarType;

// ----------------------------------------------------------------------------------
// Metatensor Inference for Dtype
// ----------------------------------------------------------------------------------

// 创建用于 Metatensor 的节点参数的唯一指针栈
std::unique_ptr<Stack> MTensorArgumentCreator(Node* n) {
  auto stack = std::make_unique<std::vector<IValue>>();
  for (Value* inp : n->inputs()) {
    if (auto tp = inp->type()->cast<TensorType>()) {
      // 对于零维张量，需要特殊类型推断行为，因此需要排名
      auto rank = tp->symbolic_sizes().rank(); // 在之前已经验证过有效性
      auto tensor_size = std::vector<int64_t>(rank.value(), 1);
      stack->emplace_back(at::empty(
          tensor_size, at::TensorOptions(at::kMeta).dtype(*tp->scalarType())));
      continue;
    }
    // TODO: 填充我们已知的具体值
    if (inp->type() == FloatType::get()) {
      stack->emplace_back(1.);
    } else if (inp->type() == IntType::get()) {
      stack->emplace_back(1);
    } else if (inp->type() == BoolType::get()) {
      throw std::runtime_error(
          "Bool currently unsupported, need to verify it's safe to add for all ops");
      stack->emplace_back(false);
    } else {
      // 不处理值数组，因为简单的默认值可能是不正确的
      throw std::runtime_error("Unsupported input type for Tensor argument");
    }
  }
  return stack;
};

// 检查 Metatensor 节点参数的有效性
bool MTensorNodeArgValid(Value* value) {
  auto tensor_type = value->type()->cast<TensorType>();
  if (!tensor_type) {
    return true;
  }
  if (!tensor_type->scalarType().has_value()) {
    GRAPH_DEBUG("Argument missing Dtype");
    return false;
  }
  auto rank = tensor_type->symbolic_sizes().rank();
  return rank.has_value();
}

// 检查节点是否可以使用 Metatensor 推断
static bool canBeInferredWithMetaTensor(Node* n) {
  // 不保证 Metatensor 不会报错
  // 现在没有白名单，让执行时出错
  // 在另一个地方检查是否有 Tensor 输出
  bool args_valid =
      std::all_of(n->inputs().begin(), n->inputs().end(), MTensorNodeArgValid);

  if (!args_valid) {
    return false;
  }
  if (n->outputs().size() != 1) {
    // 目前不支持多个输出
    // ...


这段代码主要是 C++ 的头文件引入和一些函数定义，其中还包含了一些用于张量类型推断的辅助函数。
    return false;
  }
  // 获取节点 n 的可能操作符
  auto opt_op = n->maybeOperator();
  // 如果 opt_op 为空指针，则输出调试信息并返回 false
  if (!opt_op) {
    GRAPH_DEBUG("not registered with Meta");
    return false;
  }
  // 如果节点 n 注册了操作符，则返回 true
  return true;
}

// 推断节点的元数据张量
std::optional<Tensor> inferWithMetaTensor(Node* n) {
  // 调试信息：记录正在推断元数据张量的节点信息
  GRAPH_DEBUG("inferWithMetaTensor", getHeader(n));
  // 如果节点无法推断元数据张量，则返回空值
  if (!canBeInferredWithMetaTensor(n)) {
    return c10::nullopt;
  }
  // 获取节点的操作
  Operation op = n->getOperation();
  try {
    // 创建元数据张量参数栈
    auto stack = MTensorArgumentCreator(n);
    // 调试信息：记录正在执行操作的节点信息
    GRAPH_DEBUG("Running op for ", getHeader(n));
    // 执行操作
    op(*stack);
    // 调试信息：操作成功运行后的节点信息
    GRAPH_DEBUG("op run successfully", getHeader(n));
    // 调试信息：操作执行后的节点信息
    GRAPH_DEBUG("After receive!");
    // 返回操作执行结果的张量
    return stack->back().toTensor();

  } catch (...) {
    // 调试信息：捕获到使用元数据张量运行时的异常
    GRAPH_DEBUG("caught exception with Metatensor run!");
  };
  // 返回空值，表示推断失败
  return c10::nullopt;
}

// 设置张量的数据类型
bool setDtype(
    Value* value,
    ScalarType scalarType,
    bool can_overwrite_dtype = false) {
  // 获取值的张量类型
  auto tensor_type = value->type()->cast<TensorType>();
  // 内部断言：确保值是张量类型
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  // 如果值的张量类型不包含标量类型信息，则设置新的标量类型并返回 true
  if (!tensor_type->scalarType().has_value()) {
    value->setType(tensor_type->withScalarType(scalarType));
    return true;
  }
  // 如果值的标量类型与目标标量类型不一致，则根据是否允许覆盖标量类型来设置新的标量类型并返回 true
  if (tensor_type->scalarType().value() != scalarType) {
    TORCH_INTERNAL_ASSERT(
        can_overwrite_dtype,
        "Expected tensor type to be ",
        scalarType,
        " but found ",
        tensor_type->scalarType().value());
    value->setType(tensor_type->withScalarType(scalarType));
    return true;
  }
  // 如果标量类型已经一致，则返回 false
  return false;
}

// 尝试应用元数据张量的数据类型
bool tryApplyDtypeMetaTensor(Node* n) {
  // 返回是否有任何更改发生
  auto return_tensor = inferWithMetaTensor(n);
  // 如果推断失败，则返回 false
  if (!return_tensor) {
    return false;
  }
  // 调试信息：记录接收到的张量的标量类型
  GRAPH_DEBUG("Received ", toString(return_tensor->scalar_type()));
  // 将节点输出的数据类型设置为推断得到的张量的标量类型，并返回是否有更改发生
  return setDtype(n->output(), return_tensor->scalar_type());
}

// ----------------------------------------------------------------------------------
// Dtype 的自定义规则
// ----------------------------------------------------------------------------------

// Dtype 属性规则类型定义
using DtypePropRule = std::function<bool(Node*)>;

// 设置如果所有的数据类型匹配
bool setIfAllDtypeMatch(Node* n) {
  // 将所有张量输出设置为第一个输入的数据类型
  // 只有当所有输入具有相同的数据类型时才进行设置，否则不做任何操作
  TORCH_INTERNAL_ASSERT(!n->inputs().empty());
  // 获取第一个参数
  auto first_arg = n->inputs().at(0);
  // 获取第一个参数的张量类型
  auto tensor_type = first_arg->type()->cast<TensorType>();
  // 内部断言：确保第一个参数是张量类型
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  // 获取第一个参数的标量类型
  auto scalar_type = tensor_type->scalarType();
  // 如果第一个参数没有标量类型，则返回 false
  if (!scalar_type.has_value()) {
    return false;
  }
  // 遍历所有输入参数
  for (auto arg : n->inputs()) {
    // 获取参数的张量类型
    tensor_type = arg->type()->cast<TensorType>();
    // 如果参数不是张量类型，则跳过
    if (!tensor_type) {
      continue;
    }
    // 获取参数的标量类型
    auto arg_scalar_type = tensor_type->scalarType();

    // 如果参数的标量类型为空（对于可选参数），则继续下一个参数
    if (!arg_scalar_type.has_value()) {
      continue;
    }
    // 如果参数的标量类型与第一个参数的标量类型不一致，则返回 false
    if (arg_scalar_type != scalar_type) {
      return false;
    }
  }

  // 标记是否有任何更改发生
  bool changed = false;
  // 遍历所有输出
  for (auto output : n->outputs()) {
    // 如果输出是张量类型，则将其数据类型设置为第一个参数的标量类型，并更新标记
    if (output->type()->cast<TensorType>()) {
      changed |= setDtype(output, scalar_type.value());
    }
  }
  // 返回是否有任何更改发生
  return changed;
}
// DtypePropagationPass 是一个分析 pass，在图中按拓扑顺序遍历并向前传播 Dtypes（ScalarTypes），
// 从图的输入（在 input_descriptors 中表示）到所有输出张量节点。
struct DtypePropagationPass {
  explicit DtypePropagationPass(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {
    // 构建 Dtype 规则注册表
    buildDtypeRuleRegistry();
  }

  // 如果至少有一个节点的标量类型设置在张量节点上，则返回 true
  bool run() {
    return processBlocks(graph_->block());
  }

 private:
  // 处理一组块（blocks）中的块
  bool processBlocks(at::ArrayRef<Block*> blocks) {
    bool changed = false;
    for (auto block : blocks) {
      // 处理单个块（block）
      changed |= processBlock(block);
    }
    return changed;
  }

  // 处理单个块（block）
  bool processBlock(Block* block) {
    GRAPH_DEBUG("processBlock");
    bool changed = false;
    // 遍历块中的节点
    for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
      // 处理单个节点
      changed |= processNode(*it);
    }
    return changed;
  }

  // 处理单个节点
  bool processNode(Node* n) {
    GRAPH_DEBUG("processNode");
    switch (n->kind()) {
      case prim::If:
        // 处理 If 节点
        return processIf(n);
      case prim::Loop:
      case prim::CallMethod:
      case prim::CallFunction:
        // 目前不处理 Loop 和 Call
        TORCH_INTERNAL_ASSERT(false, "Loop/Call not handled now");
      default:
        break;
    }

    // 检查节点是否至少有一个张量输出
    bool has_tensor_output =
        std::any_of(n->outputs().begin(), n->outputs().end(), [](Value* v) {
          return (bool)v->type()->cast<TensorType>();
        });

    if (!has_tensor_output) {
      // 如果输出不包含张量，则无需传播
      return false;
    }

    switch (n->kind()) {
      case prim::Constant:
        // 常量节点，已经由其他内容进行了传播
        return false;
      case prim::ListConstruct:
      case prim::ListUnpack:
        // 不支持 List Construct 和 List Unpack
        TORCH_INTERNAL_ASSERT(
            false,
            "List Construct and Unpack is not supported in Dtype Propagation");
        break;
      default:
        if (n->kind().is_aten()) {
          // 处理 ATen 操作
          return processAtenOps(n);
        } else {
          // 不支持的操作类型
          TORCH_INTERNAL_ASSERT(
              false,
              n->kind().toDisplayString(),
              "Op is not supported in Dtype Propagation");
        }
    }
    return false;
  }

  // 合并张量属性（placeholder for MobileNet）
  bool mergeTensorProperties(
      const at::ArrayRef<Value*>& list1,
      const at::ArrayRef<Value*>& list2) {
    // 这是 MobileNet 的占位符实现
    TORCH_INTERNAL_ASSERT(list1.empty(), "Not implemented yet");
    return false;
  }

  // 处理 If 节点
  bool processIf(Node* node) {
    GRAPH_DEBUG("processIf");
    bool changed = false;
    auto blocks = node->blocks();
    auto true_block = blocks.at(0);
    auto false_block = blocks.at(1);

    // 处理 If 节点的两个分支块
    changed |= processBlock(true_block);
    changed |= processBlock(false_block);

    // 合并两个分支块的张量属性
    changed |=
        mergeTensorProperties(true_block->outputs(), false_block->outputs());

    return changed;
  }

  // 处理 ATen 操作节点
  bool processAtenOps(Node* n) {
    GRAPH_DEBUG("processAtenOps");
    // 打印调试信息，标记处理 Aten 操作
    GRAPH_DEBUG("case = ", n->kind(), " ", *n);
    // 打印调试信息，标记 case 类型和节点信息
    // 自定义规则匹配
    if (auto prop_fn = dtype_prop_registry_->find(n->getOperator())) {
      // 如果找到了操作符对应的属性函数
      DtypePropRule rule = *prop_fn;
      // 将找到的规则应用于当前节点 n
      return rule(n);
    }
    // 如果没有找到对应的属性函数，则尝试应用默认的 dtype meta tensor
    return tryApplyDtypeMetaTensor(n);
  }

  void buildDtypeRuleRegistry() {
    // 构建所有自定义 dtype 规则的注册表
    dtype_prop_registry_ = std::make_unique<OperatorMap<DtypePropRule>>();

    // 插入第一个输入保持不变的 nn 操作和对应的设置函数
    dtype_prop_registry_->insert(
        *nn_ops_first_input_preserving(), setIfAllDtypeMatch);
    // 插入一个张量输入形状转换的操作和对应的设置函数
    dtype_prop_registry_->insert(
        *ops_one_tensor_in_shape_transform(), setIfAllDtypeMatch);
  }
  // 唯一指针，指向 OperatorMap<DtypePropRule> 类型的 dtype 属性注册表
  std::unique_ptr<OperatorMap<DtypePropRule>> dtype_prop_registry_;
  // 共享指针，指向 Graph 类型的图对象
  std::shared_ptr<Graph> graph_;
};

} // anonymous namespace

// This analysis propagates input dtypes (if any) throughout the
// graph.
bool DtypePropagation(std::shared_ptr<Graph>& graph) {
  // 创建 DtypePropagationPass 对象，传入图形对象 graph
  DtypePropagationPass tp = DtypePropagationPass(graph);
  // 运行 DtypePropagationPass 分析，并获取是否有变化的返回值
  bool changed = tp.run();
  // 如果图形发生了变化，则输出带有日志前缀的图形信息
  if (changed) {
    GRAPH_DUMP("After TensorPropertyPropagation pass:", graph);
  }
  // 返回变化标志
  return changed;
}

} // namespace jit
} // namespace torch
```