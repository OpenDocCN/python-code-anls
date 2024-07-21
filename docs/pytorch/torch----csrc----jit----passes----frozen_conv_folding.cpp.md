# `.\pytorch\torch\csrc\jit\passes\frozen_conv_folding.cpp`

```
#include <ATen/Utils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace torch {
namespace jit {

namespace {

using Tensor = at::Tensor;

// 判断节点是否为支持的卷积操作节点
bool supportedConvNode(Node* n) {
  switch (n->kind()) {
    case aten::conv1d:
    case aten::conv2d:
    case aten::conv3d:
      return true;
    case aten::_convolution: {
      auto transposed_conv =
          constant_as<bool>(n->namedInput("transposed")).value_or(true);
      // 不处理转置卷积或者不是常量转置参数的情况
      return !transposed_conv;
    }
    default:
      return false;
  }
}

// 在给定的基本块上折叠冻结卷积和批归一化
bool FoldFrozenConvBatchnorm(Block* b) {
  bool graph_modified = false;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenConvBatchnorm(block);
    }

    // 这里应该有一些逻辑来修改图
  }
  return graph_modified;
}

// 判断节点是否为支持的加法或减法节点
bool supportedAddOrSub(Node* n) {
  static const OperatorSet add_set{
      "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
      // sub 等同于 add
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
  };
  return n->isMemberOf(add_set);
}

// 为了将加法/减法/乘法/除法与卷积融合，其常数张量的维度必须满足以下条件：
// - 调整大小以广播到权重/偏置张量的形状
// - 广播到卷积输出形状
// 它需要具有能够调整为权重/偏置张量形状的形状，因为我们需要在保持其大小不变的情况下运行操作与卷积权重/偏置。
// 它需要广播到卷积输出形状，以便我们在预先融合之前不会意外地改变操作输出的形状。
// 在权重/偏置/卷积输出张量中唯一共享的尺寸值是它们都包含一个值为 channels-out 的维度。在卷积输出张量中，这在第二维度上，
// 所以逐点操作张量可能具有第二维度的值 == channels-out，但是所有其他维度都必须为 1
bool opDoesNotBroadCastWithConv(Tensor& op_tensor, Tensor& weight_tensor) {
  if (op_tensor.ndimension() > weight_tensor.ndimension()) {
    # 如果条件不满足，返回 false
    return false;
  }
  # 从 op_tensor 的最后一个维度开始向前遍历
  for (int64_t i = op_tensor.ndimension() - 1; i >= 0; i--) {
    # 对于 channels-out 维度，检查其是否等于 weight_tensor 的第一个维度大小
    if (i == 1 && op_tensor.size(i) == weight_tensor.size(0)) {
      # 如果条件满足，跳过当前循环
      continue;
    }
    # 如果当前维度大小不为 1，则返回 false
    if (op_tensor.size(i) != 1) {
      return false;
    }
  }
  # 如果所有维度条件都满足，则返回 true
  return true;
}

bool checkConvAndBroadcastingOpPreConditions(Node* conv, Node* op) {
  // 检查卷积操作和广播操作的前置条件
  if (nonConstantParameters(conv) || nonConstantParameters(op)) {
    return false;
  }

  // 检查卷积操作的输出是否有多个使用者
  if (conv->output()->uses().size() > 1) {
    return false;
  }

  // 获取卷积操作中名为"weight"的常量输入，作为权重张量
  Tensor weight_tensor =
      constant_as<Tensor>(conv->namedInput("weight")).value();

  // 避免与可能导致类型提升的操作融合
  // 限制为浮点数可以避免标量重载时的整数/浮点数困难
  if (!weight_tensor.is_floating_point()) {
    return false;
  }

  // 如果操作的第二个输入是张量类型
  if (op->inputs().at(1)->type()->cast<TensorType>()) {
    // 尝试获取操作的第二个输入作为张量
    auto op_tensor = constant_as<Tensor>(op->inputs().at(1)).value();
    // 检查操作是否不会与卷积的权重张量进行广播
    if (!opDoesNotBroadCastWithConv(op_tensor, weight_tensor)) {
      return false;
    }

    // 如果操作的张量不是浮点数，并且类型提升后的类型与权重张量的类型不同
    if (!op_tensor.is_floating_point() &&
        c10::promoteTypes(
            op_tensor.scalar_type(), weight_tensor.scalar_type()) !=
            weight_tensor.scalar_type()) {
      return false;
    }
  }
  return true;
}

Tensor resizeConstantScalarOrTensorToShape(
    Value* v,
    const std::vector<int64_t>& shape,
    at::TensorOptions options) {
  // 根据输入值的类型，将常量标量或张量调整为指定形状的张量
  Tensor ret_tensor;
  if (v->type()->cast<TensorType>()) {
    // 如果输入值是张量类型，则将其转换为张量
    ret_tensor = constant_as<Tensor>(v).value();
  } else {
    // 如果输入值不是张量类型，则创建指定形状的零张量
    ret_tensor = at::zeros(shape, options);
    // 如果输入值是整数类型，则使用该整数填充张量
    if (v->type()->cast<IntType>()) {
      ret_tensor.fill_(constant_as<int64_t>(v).value());
    } else {
      // 否则，使用输入值作为浮点数填充张量
      ret_tensor.fill_(constant_as<double>(v).value());
    }
  }

  // 如果张量元素数量为1，则扩展为指定形状
  if (ret_tensor.numel() == 1) {
    ret_tensor = ret_tensor.reshape({1});
    ret_tensor = ret_tensor.expand(shape);
  } else {
    // 否则，验证张量元素数量与形状乘积是否相等
    TORCH_INTERNAL_ASSERT(ret_tensor.numel() == c10::multiply_integers(shape));
    ret_tensor = ret_tensor.view(shape);
  }
  return ret_tensor;
}

bool FoldFrozenConvAddOrSub(Block* b) {
  // 折叠冻结的卷积加法或减法
  bool graph_modified = false;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      // 递归处理当前节点中的所有子块
      graph_modified |= FoldFrozenConvAddOrSub(block);
    }
    // 检查节点 n 是否支持加法或减法，并且输入节点为支持的转换节点
    if (supportedAddOrSub(n) && supportedConvNode(n->inputs().at(0)->node())) {
      // 获取卷积节点和加法或减法节点
      auto conv = n->inputs().at(0)->node();
      auto add_or_sub = n;

      // 检查卷积和广播操作的前提条件
      if (!checkConvAndBroadcastingOpPreConditions(conv, add_or_sub)) {
        // 如果前提条件不满足，则继续处理下一个节点
        continue;
      }

      // 获取卷积操作的权重张量
      Tensor weight_tensor =
          constant_as<Tensor>(conv->namedInput("weight")).value();

      // 调整加法或减法节点的第二个输入为指定形状的常量标量或张量
      Tensor add_or_sub_tensor = resizeConstantScalarOrTensorToShape(
          add_or_sub->inputs().at(1),
          {weight_tensor.size(0)},
          weight_tensor.options());
      
      // 初始化偏置张量
      Tensor bias;
      if (conv->namedInput("bias")->type() == NoneType::get()) {
        // 如果卷积操作没有偏置，则创建一个与加法或减法节点相同形状的零张量
        bias = at::zeros_like(add_or_sub_tensor, weight_tensor.dtype());
      } else {
        // 否则，从卷积操作的输入中获取偏置张量
        bias = constant_as<Tensor>(conv->namedInput("bias")).value();
      }

      // 在卷积节点的插入点创建一个上下文保护
      WithInsertPoint guard(conv);

      // 用常量偏置替换加法或减法节点的输入
      add_or_sub->replaceInputWith(
          conv->output(), b->owningGraph()->insertConstant(bias));
      // 替换加法或减法节点的第二个输入
      add_or_sub->replaceInput(
          1, b->owningGraph()->insertConstant(add_or_sub_tensor));

      // 如果输入为常量，则运行节点
      auto stack_out = runNodeIfInputsAreConstant(add_or_sub);
      // 内部断言：堆栈输出不为空且大小为 1
      TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
      // 将堆栈输出的张量转换为指定类型的融合偏置张量
      Tensor fuse_bias = (*stack_out)[0].toTensor().to(bias.dtype());

      // 在图中插入一个常量节点，表示融合后的卷积偏置
      auto fused_conv_b = b->owningGraph()->insertConstant(fuse_bias);
      // 获取卷积节点的偏置输入
      auto conv_b_value = conv->namedInput("bias");

      // 设置融合后卷积偏置节点的调试名称
      fused_conv_b->setDebugName(
          conv_b_value->debugName() + "_fused_" +
          add_or_sub->kind().toUnqualString());
      
      // 用融合后的卷积偏置替换卷积节点的偏置输入
      conv->replaceInputWith(conv_b_value, fused_conv_b);
      // 替换加法或减法节点的输出使用为卷积节点的输出
      add_or_sub->output()->replaceAllUsesWith(conv->output());
      
      // 标记图已修改
      graph_modified = true;
      // 执行死代码消除以清理节点
      // DCE run after cleans up nodes
    }
  }
  // 返回图是否被修改的标志
  return graph_modified;
} // 结束当前命名空间

// 检查节点是否属于支持的乘法或除法操作集合
bool supportedMulOrDiv(Node* n) {
  // 定义包含支持的操作名称的静态集合
  static const OperatorSet add_set{
      "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
      // div 等同于 mul
      "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
  };
  // 检查节点是否属于支持的操作集合中
  return n->isMemberOf(add_set);
}

// 对于给定的基本块，尝试折叠冻结卷积层的乘法或除法操作
bool FoldFrozenConvMulOrDiv(Block* b) {
  bool graph_modified = false;
  // 遍历基本块中的每个节点
  for (Node* n : b->nodes()) {
    // 对于节点中的每个子块，递归调用 FoldFrozenConvMulOrDiv 函数
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenConvMulOrDiv(block);
    }

    // 这里有一些未完成的代码，可能需要进一步的实现
    // 暂时保留注释
    // ...

  }
  return graph_modified;
}

// 结束当前命名空间 "jit"

// 尝试对图中的冻结卷积层进行批量归一化折叠操作
bool FoldFrozenConvBatchnorm(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenConvBatchnorm(graph->block());
  // 删除图中的死代码
  EliminateDeadCode(graph);
  return graph_modified;
}

// 尝试对图中的冻结卷积层进行加法或减法折叠操作
bool FoldFrozenConvAddOrSub(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenConvAddOrSub(graph->block());
  // 删除图中的死代码
  EliminateDeadCode(graph);
  return graph_modified;
}

// 尝试对图中的冻结卷积层进行乘法或除法折叠操作
bool FoldFrozenConvMulOrDiv(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenConvMulOrDiv(graph->block());
  // 删除图中的死代码
  EliminateDeadCode(graph);
  return graph_modified;
}

// 结束命名空间 "torch"
```