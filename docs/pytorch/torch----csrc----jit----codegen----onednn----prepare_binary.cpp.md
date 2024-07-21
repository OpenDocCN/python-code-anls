# `.\pytorch\torch\csrc\jit\codegen\onednn\prepare_binary.cpp`

```
// 引入 ATen 和 Torch 库中的相关头文件
#include <aten/src/ATen/core/jit_type.h>
#include <torch/csrc/jit/codegen/onednn/prepare_binary.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

// 定义命名空间 torch::jit::fuser::onednn，用于组织代码
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 静态函数：比较节点的常量值与给定的 double 值是否相等
static bool compareConstValue(Value* v, double d) {
  auto ival = toIValue(v);  // 获取值的 IValue 表示
  return ival.has_value() &&  // 如果有有效的 IValue
      ((ival->isInt() && static_cast<int>(ival->toInt()) == d) ||  // 如果是整数且与 d 相等
       (ival->isDouble() && ival->toDouble() == d));  // 或者是双精度浮点数且与 d 相等
}

// 静态函数：处理二元操作节点的输入
static void handleBinaryOpInputs(Node* node) {
  // 如果第一个输入是张量类型
  if (node->input(0)->type()->isSubtypeOf(TensorType::get())) {
    // 获取第一个输入张量的数据类型
    auto dtypeOfFirstInput =
        node->input(0)->type()->cast<TensorType>()->scalarType().value();
    // 如果第二个输入是浮点数类型或整数类型
    if (node->input(1)->type()->isSubtypeOf(FloatType::get()) ||
        node->input(1)->type()->isSubtypeOf(IntType::get())) {
      // 推断第二个输入标量与张量的数据类型相同，以满足一DNN图的输入数据类型要求
      // 创建一个标量输入的1维张量，并"提升"其数据类型为第一个输入的数据类型
      auto promotedDtype = dtypeOfFirstInput;
      auto scalar = node->input(1);
      WithInsertPoint guard(node);  // 设置插入点为当前节点
      auto g = node->owningGraph();  // 获取当前节点所属的图

      // 将标量转换为张量，保持数据类型为提升后的数据类型
      auto t = g->insert(aten::as_tensor, {scalar}, {{"dtype", promotedDtype}});
      
      // 在 IR 中添加维度和步长信息
      std::optional<size_t> t_dim = 1;
      auto target_type = TensorTypePtr(
          TensorType::create(promotedDtype, at::kCPU, t_dim, false));
      target_type = target_type->withSizes({1});
      t->setType(target_type);

      // 将转换后的张量进行 unsqueeze 操作，使其变为一维张量
      auto unsqueezed = g->insert(aten::unsqueeze, {t, 0});
      unsqueezed->setType(target_type);
      node->replaceInput(1, unsqueezed);

      // 更新输出节点的数据类型，以反映可能的数据类型变化
      node->output()->setType(
          node->output()->type()->expect<TensorType>()->withScalarType(
              promotedDtype));


因为字符数限制的缘故，无法添加更多的内容
    } else if (node->input(1)->type()->isSubtypeOf(TensorType::get())) {
      // 如果第二个输入是张量，我们需要确保两个输入具有相同的数据类型，
      // 因为 oneDNN 图要求两个输入具有相同的数据类型。我们将遵循 PyTorch 的类型提升规则。
      auto second_input_typeptr = node->input(1)->type()->expect<TensorType>();
      // 获取第二个输入张量的数据类型
      std::optional<at::ScalarType> second_input_type =
          second_input_typeptr->scalarType();
      if (second_input_type != c10::nullopt) {
        // 第二个张量的数据类型可能在 IR 中不可用
        auto dtypeOfSecondInput = second_input_type.value();
        if (dtypeOfFirstInput != dtypeOfSecondInput) {
          // 需要进行类型提升
          auto promotedDtype =
              c10::promoteTypes(dtypeOfFirstInput, dtypeOfSecondInput);
          // 在节点的插入点设置上下文
          WithInsertPoint guard(node);
          auto g = node->owningGraph();
          if (promotedDtype == dtypeOfFirstInput) {
            // 将第二个输入张量转换为提升后的数据类型
            auto to_node_output = g->insert(
                aten::to, {node->input(1)}, {{"dtype", promotedDtype}});
            to_node_output->setType(
                node->input(1)->type()->expect<TensorType>()->withScalarType(
                    promotedDtype));
            node->replaceInput(1, to_node_output);
          } else {
            // 将第一个输入张量转换为提升后的数据类型
            auto to_node_output = g->insert(
                aten::to, {node->input(0)}, {{"dtype", promotedDtype}});
            to_node_output->setType(
                node->input(0)->type()->expect<TensorType>()->withScalarType(
                    promotedDtype));
            node->replaceInput(0, to_node_output);
          }
          // 数据类型可能已更改，因此在 IR 中也需要更新
          node->output()->setType(
              node->output()->type()->expect<TensorType>()->withScalarType(
                  promotedDtype));
        } else {
          // 两个数据类型相同
          // 在 JIT IR 中，有时会缺少数据类型的信息，
          // 我们不应默认将这些张量视为 FP32 张量。
          node->output()->setType(
              node->output()->type()->expect<TensorType>()->withScalarType(
                  dtypeOfFirstInput));
        }
      } // end inner if block
    } // end outer if block
  }
// 将标量转换为张量
static void ConvertScalarToTensor(Block* block) {
  // 遍历块中的每个节点
  for (auto node : block->nodes()) {
    // 递归处理节点中的子块
    for (auto sub : node->blocks()) {
      ConvertScalarToTensor(sub);
    }

    // 如果节点是加法、乘法或除法操作
    if (node->kind() == aten::add || node->kind() == aten::mul ||
        node->kind() == aten::div) {
      // 处理二元操作的输入
      handleBinaryOpInputs(node);
    }
  }
}

// 可能分解加法操作
static void mayDecomposeAdd(Node* node) {
  // 如果节点的输入数量小于3，则返回
  if (node->inputs().size() < 3) {
    return; // 在 BERT-mrpc 中的特殊情况，与 native_functions.yaml 不一致
  }
  // 如果 alpha 值为常量1.0
  if (toIValue(node->namedInput("alpha")).has_value()) {
    // 比较 alpha 是否等于1
    auto alphaEqualsOne = compareConstValue(node->namedInput("alpha"), 1.0);
    // 如果 alpha 不等于1
    if (!alphaEqualsOne) {
      // 在节点位置插入操作
      WithInsertPoint guard(node);
      auto g = node->owningGraph();
      // 插入乘法操作
      auto mul = g->insert(
          aten::mul, {node->namedInput("other"), node->namedInput("alpha")});
      // 如果 other 输入是张量类型
      if (node->namedInput("other")->type()->isSubtypeOf(TensorType::get())) {
        auto mulTensorTypePtr = node->namedInput("other")->type();
        mul->setType(mulTensorTypePtr);
      }
      // 替换节点的输入
      node->replaceInput(1, mul);
      auto one = g->insertConstant(1.0);
      node->replaceInput(2, one);
    }
  }
}

// 分解融合的加法操作
static void DecomposeFusedAdd(Block* block) {
  // 遍历块中的每个节点
  for (auto node : block->nodes()) {
    // 递归处理节点中的子块
    for (auto sub : node->blocks()) {
      DecomposeFusedAdd(sub);
    }

    // 如果节点是加法操作
    if (node->kind() == aten::add) {
      // 可能分解加法操作
      mayDecomposeAdd(node);
    }
  }
}

// 消除恒等乘法和加法
static void EliminateIdentityMulAdd(Block* block) {
  // 遍历块中的每个节点
  for (auto node : block->nodes()) {
    // 递归处理节点中的子块
    for (auto sub : node->blocks()) {
      EliminateIdentityMulAdd(sub);
    }

    // 如果节点是恒等于0的加法或恒等于1的乘法
    if ((node->kind() == aten::add && compareConstValue(node->input(1), 0.0)) ||
        (node->kind() == aten::mul && compareConstValue(node->input(1), 1.0))) {
      // 替换节点的输出使用情况
      node->output()->replaceAllUsesWith(node->namedInput("self"));
    }
  }
}

// 为 LLGA 准备二元操作
void PrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph) {
  // 分解融合的加法操作
  DecomposeFusedAdd(graph->block());
  // 消除恒等乘法和加法
  EliminateIdentityMulAdd(graph->block());
  // 消除死代码
  EliminateDeadCode(graph);
  // ConvertScalarToTensor 必须放在 EliminateIdentityMulAdd 之后
  ConvertScalarToTensor(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```