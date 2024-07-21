# `.\pytorch\torch\csrc\jit\passes\onnx\fixup_onnx_controlflow.cpp`

```py
#include <torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.h>
// 包含修复 ONNX 控制流的头文件

#include <ATen/InitialTensorOptions.h>
// 包含初始化张量选项的头文件

#include <c10/util/irange.h>
// 包含 C10 库中的整数范围工具头文件

#include <torch/csrc/jit/jit_log.h>
// 包含 JIT 日志功能的头文件

#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 包含死代码消除的 JIT 通行证头文件

#include <torch/csrc/jit/passes/onnx/helper.h>
// 包含 ONNX 辅助功能的 JIT 通行证头文件

#include <torch/csrc/jit/passes/onnx/peephole.h>
// 包含 ONNX 窥视孔优化的 JIT 通行证头文件

#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
// 包含 ONNX 形状类型推断的 JIT 通行证头文件

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {
const int ONNX_OPSET_13 = 13;
// 定义 ONNX 操作集版本为 13

const int ONNX_TYPE_BOOL = 9;
// 定义 ONNX 中布尔类型的表示为 9

Node* CreateCastToBoolNode(Value* val, Graph* graph) {
  // 创建一个将值转换为布尔类型的节点
  Node* cast_node = graph->create(onnx::Cast);
  cast_node->addInput(val);
  cast_node->i_(attr::to, ONNX_TYPE_BOOL);
  cast_node->output()->setType(BoolType::get());
  return cast_node;
}

Node* InsertCastForCond(
    Value* cond_val,
    Graph* graph,
    Node* consumer_node,
    int opset_version) {
  // 在条件值前插入类型转换节点
  // prev:  cond_val -> consumer_node
  // after: cond_val -> cast -> consumer_node
  // 注意：由于像 PyTorch 的 Greater/Less 运算符返回 torch.uint8 类型的张量。
  //       然而在 ONNX 的 Loop 中，条件输入必须是布尔类型，所以需要进行类型转换。
  Node* cast_node = CreateCastToBoolNode(cond_val, graph);
  cast_node->insertBefore(consumer_node);

  consumer_node->replaceInputWith(cond_val, cast_node->output());
  const ParamMap empty_params_dict = {};
  ONNXShapeTypeInference(cast_node, empty_params_dict, opset_version);
  return cast_node;
}

bool IsCondCastRequired(Value* cond_val) {
  // 判断是否需要对条件值进行类型转换
  const auto& type = cond_val->type();
  if (auto tt = type->cast<TensorType>()) {
    if (auto scalar_type = tt->scalarType()) {
      return *scalar_type != c10::kBool;
    }
  }
  return !type->isSubtypeOf(*BoolType::get());
}

bool IsErasableSequence(const Node* loop_node, size_t i) {
  // 判断循环节点中的序列是否可删除
  TORCH_INTERNAL_ASSERT(loop_node->blocks().size() == 1);
  auto* sub_block = loop_node->blocks()[0];
  auto* seq_node = sub_block->outputs()[i - 1]->node();
  auto* in_val = sub_block->inputs()[i];

  if (seq_node->kind() != ::c10::onnx::SequenceInsert) {
    return false;
  }

  if (seq_node->inputs().size() == 3) {
    // 不支持非默认的插入位置
    return false;
  }

  if (seq_node->input(0) != in_val) {
    // 只支持作用于循环传递的序列的 SequenceInsert
    return false;
  }

  const auto* init_seq_node = loop_node->inputs()[i]->node();
  const auto init_seq_node_kind = init_seq_node->kind();
  if ((init_seq_node_kind != ::c10::onnx::SequenceEmpty) &&
      (init_seq_node_kind != ::c10::prim::ListConstruct ||
       !init_seq_node->inputs().empty())) {
    // 初始序列必须为空
    return false;
  }

  if (seq_node->output()->uses().size() != 1) {
    // 该序列不支持在子块内其他地方使用
    return false;
  }

  return true;
}

// ONNX::Loop 不支持作为循环传递依赖项的 Sequence 类型。仅支持
// tensors are supported. This pass converts Sequence loop-carried dependencies
// to scan_outputs. In opset 11, only the below pattern is supported.
//
// PTIR graph:
//  ...
//  %res.1 : Tensor[] = prim::ListConstruct()
//  %res : Tensor[] = prim::Loop(%11, %22, %res.1)
//    block0(%i.1 : Tensor, %res.6 : Tensor[]):
//      ...
//      %res.3 : Tensor[] = aten::append(%res.6, %17)
//      -> (%22, %res.3)
//  return (%res.3)
//
// ONNX graph:
//  ...
//  %res : Tensor = onnx::Loop(%11, %22)
//    block0(%i.1 : Tensor):
//      ...
//      -> (%22, %17)
//  %res_seq : Tensor[] = onnx::SplitToSequence[keepdims=0](%res)
//  return (%res_seq)
std::vector<Value*> ConvertSequenceDependencies(Node* node, int opset_version) {
  if (node->kind() != ::c10::onnx::Loop) {
    // 如果节点不是 ONNX 的 Loop 节点，则直接返回其输出向量
    return node->outputs().vec();
  }

  if (opset_version >= ONNX_OPSET_13) {
    // 在 ONNX opset 13 及以上版本，应该支持作为循环传递依赖的 Sequence 类型
    return node->outputs().vec();
  }

  auto* loop_node = node;

  // 断言只有一个子块
  TORCH_INTERNAL_ASSERT(loop_node->blocks().size() == 1);
  auto* sub_block = loop_node->blocks()[0];

  std::vector<size_t> idx_to_remove;
  std::vector<Value*> new_outputs;

  // ONNX Loop 节点:
  // 子块的输入是 (iter, cond, loop-carried dependencies)
  // 子块的输出是 (cond, loop-carried dependencies, scan outputs)
  // 输入是 (iter, cond, loop-carried dependencies)
  // 输出是 (loop-carried dependencies, scan outputs)
  for (size_t i = 2; i < sub_block->inputs().size(); ++i) {
    if (IsErasableSequence(loop_node, i)) {
      // 如果是可擦除的 Sequence 类型，则执行以下操作

      // 获取 Sequence 节点
      auto* seq_node = sub_block->outputs()[i - 1]->node();
      // 用插入的元素替换序列输出
      auto inserted_value = seq_node->input(1);
      sub_block->return_node()->replaceInputWith(seq_node->output(), inserted_value);

      // 将添加的 scan_output 拆分回期望的张量序列
      auto loop_output = loop_node->output(i - 2);
      Node* split_node = loop_node->owningGraph()->create(onnx::SplitToSequence);
      loop_output->replaceAllUsesWith(split_node->output());
      split_node->i_(attr::keepdims, 0);
      split_node->addInput(loop_output);
      split_node->insertAfter(loop_node);
      split_node->output()->setType(loop_output->type());
      split_node->copyMetadata(loop_node);

      // 更新循环输出的类型
      loop_output->setType(c10::unshapedType(inserted_value->type()));

      // 应该安全移除生成序列的节点
      seq_node->destroy();

      idx_to_remove.push_back(i);
      new_outputs.push_back(split_node->output());
    } else {
      new_outputs.push_back(loop_node->output(i - 2));
    }
  }

  // 移除序列输出，并用 scan 输出替换
  for (const auto i : c10::irange(idx_to_remove.size())) {
    size_t idx = idx_to_remove[i] - i;

    sub_block->eraseInput(idx);
    loop_node->removeInput(idx);
    // 交换输出顺序。将所有扫描输出移动到最后。
    sub_block->return_node()->addInput(
        sub_block->return_node()->inputs().at(idx - 1));
    // 从返回节点中移除原先位于索引 idx - 1 处的输入。
    sub_block->return_node()->removeInput(idx - 1);

    // 在循环节点上添加一个新的输出。
    auto loop_out = loop_node->addOutput();
    // 复制循环节点第 idx - 2 个输出的元数据到新添加的输出。
    loop_out->copyMetadata(loop_node->outputs().at(idx - 2));
    // 用新添加的输出替换循环节点第 idx - 2 个输出在所有使用点上的引用。
    loop_node->outputs().at(idx - 2)->replaceAllUsesWith(loop_out);
    // 删除循环节点第 idx - 2 个输出。
    loop_node->eraseOutput(idx - 2);
  }

  // 返回更新后的新输出。
  return new_outputs;
}

// 定义一个名为 ONNXOptionalNode 的函数，返回一个 Node 指针
// 参数 opt_type: OptionalTypePtr 类型，表示可选类型的指针
// 参数 g: Graph 指针，表示图对象的指针
Node* ONNXOptionalNode(OptionalTypePtr opt_type, Graph* g) {
  // 断言 opt_type 不为空
  TORCH_INTERNAL_ASSERT(opt_type);
  // 获取 opt_type 的元素类型
  TypePtr elem_type = opt_type->getElementType();
  // 在图 g 中创建一个类型为 onnx::Optional 的 Node，有一个输入
  Node* opt_node = g->create(::c10::onnx::Optional, 1);
  // 设置节点的类型属性为 elem_type
  opt_node->ty_(Symbol::attr("type"), elem_type);
  // 设置节点的输出类型为 OptionalType，其中元素类型为 elem_type
  opt_node->output()->setType(OptionalType::create(elem_type));
  // 返回创建的节点
  return opt_node;
}

// 替换块输出 i 为一个 onnx::Optional
// 参数 opt_type: OptionalTypePtr 类型，表示可选类型的指针
// 参数 block: Block 指针，表示块对象的指针
// 参数 i: size_t 类型，表示要替换的输出索引
void ReplaceBlockOutputWithOptional(
    OptionalTypePtr opt_type,
    Block* block,
    size_t i) {
  // 调用 ONNXOptionalNode 函数创建 Optional 节点
  Node* opt_node = ONNXOptionalNode(opt_type, block->owningGraph());
  // 将 Optional 节点插入到返回节点之前
  opt_node->insertBefore(block->return_node());
  // 获取块的第 i 个输出值
  Value* block_output = block->outputs().at(i);
  // 仅替换最后一个值作为 Optional 类型只影响输出之前的值
  block_output->replaceAllUsesAfterNodeWith(opt_node, opt_node->output());
  // 如果块输出的类型不是 NoneType，则添加输入并复制元数据
  if (!block_output->type()->cast<NoneType>()) {
    opt_node->addInput(block_output);
    opt_node->copyMetadata(block_output->node());
  }
}

// 修复来自 ONNX 的限制，即块输出不能是来自块外部的值。
// 在块内部插入 Identity 节点，并将其链接到外部值作为解决方法。
void FixupONNXSubblockOutputs(Node* n) {
  // 遍历节点 n 的所有块
  for (Block* block : n->blocks()) {
    // 遍历每个块的输出值
    for (Value* output : block->outputs()) {
      // 如果输出值的节点所属的块不是当前块
      if (output->node()->owningBlock() != block) {
        Node* id_node = nullptr;
        // 如果输出值的类型是 NoneType，创建一个空的 Optional 节点
        // 否则创建一个 Identity 节点，并添加输出值作为输入
        if (output->type()->cast<NoneType>()) {
          id_node = block->owningGraph()->create(onnx::Optional);
        } else {
          id_node = block->owningGraph()->create(onnx::Identity);
          id_node->addInput(output);
        }
        // 将 Identity 或 Optional 节点插入到返回节点之前
        id_node->insertBefore(block->return_node());
        // 复制输出的元数据到节点的输出
        id_node->output()->copyMetadata(output);
        // 复制节点 n 的元数据到 id_node
        id_node->copyMetadata(n);
        // 用 id_node 的输出替换返回节点中的输出值
        block->return_node()->replaceInputWith(output, id_node->output());
      }
    }
  }
}

// 从输出中推断可选输入的类型。
void FixupONNXLoopBlockInputs(Node* n) {
  // 遍历节点 n 的所有块
  for (Block* block : n->blocks()) {
    // 遍历从1到block输入数量的范围，跳过第一个输入（通常是循环的迭代变量）
    for (const auto i : c10::irange(1, block->inputs().size())) {
      // 获取第i个输入对应的值，这个对应关系会在运行FixupONNXLoopNodeInputs后变化
      Value* input_i = block->inputs().at(i);
      // 如果输入i的类型是OptionalType，并且输出i的类型不是OptionalType
      if (input_i->type()->cast<OptionalType>() &&
          !block->outputs().at(i)->type()->cast<OptionalType>()) {
        // 合并推断类型，merged_type是合并后的类型，inferred表示是否有推断成功
        auto [merged_type, inferred] = MergeInferredType(
            input_i->type()->cast<OptionalType>()->getElementType(),
            block->outputs().at(i)->type());
        // 如果成功推断出类型
        if (inferred) {
          // 设置输入i的类型为OptionalType，其元素类型为merged_type
          input_i->setType(OptionalType::create(merged_type));
        }
      }
    }
  }
}

// Replace None in outputs with Optional.
void FixupONNXLoopBlockOutputs(Node* n) {
  // 遍历节点 n 的所有子块
  for (Block* block : n->blocks()) {
    // 输出 0 是 continue_condition，永远不会是 None
    for (const auto i : c10::irange(1, block->outputs().size())) {
      // 两个条件需要用 Optional 替换块输出
      // 1. 输出是 NoneType
      // 2. 输入是 Optional，但输出类型不是 Optional
      if ((block->outputs().at(i)->type()->cast<NoneType>()) ||
          (block->inputs().at(i + 1)->type()->cast<OptionalType>() &&
           !block->outputs().at(i)->type()->cast<OptionalType>())) {
        ReplaceBlockOutputWithOptional(
            // 输出 0 是 continue_condition。
            // 输入 (0, 1) 是 (loop_counter, cond)。因此输入 i + 1 对应输出 i。
            block->inputs().at(i + 1)->type()->cast<OptionalType>(),
            block,
            i);
      }
    }
  }
  // 修复节点 n 的子块输出
  FixupONNXSubblockOutputs(n);
}

void FixupONNXLoopNodeInputs(Node* node, int opset_version) {
  // 如果节点类型不是 ::c10::onnx::Loop，则直接返回
  if (node->kind() != ::c10::onnx::Loop) {
    return;
  }

  auto* graph = node->owningGraph();

  // 在循环外部添加对条件输入的类型转换
  Value* cond_val = node->input(1);
  if (IsCondCastRequired(cond_val)) {
    auto* cast_node = InsertCastForCond(cond_val, graph, node, opset_version);
    cast_node->copyMetadata(node);
  }

  // 设置循环的输入 cond 和 i
  TORCH_INTERNAL_ASSERT(node->blocks().size() == 1);
  auto* sub_block = node->blocks().at(0);
  Value* cond = sub_block->insertInput(1, "cond");
  cond->setType(BoolType::get());

  Value* i = sub_block->inputs().at(0);
  i->setType(TensorType::fromNumberType(*IntType::get()));

  // 在循环内部添加对条件输入的类型转换
  Value* next_cond_val = sub_block->outputs().at(0);
  if (IsCondCastRequired(next_cond_val)) {
    auto* cast_node = InsertCastForCond(
        next_cond_val, graph, sub_block->return_node(), opset_version);
    cast_node->copyMetadata(node);
  }

  // 输入 (0, 1) 是 (max_trip_count, start_condition)。跳过它们，因为它们永远不是 None 或 Optional。
  for (const auto i : c10::irange(2, node->inputs().size())) {
    Value* input = node->inputs().at(i);
    OptionalTypePtr sub_block_input_optional =
        sub_block->inputs().at(i)->type()->cast<OptionalType>();
    // 如果循环输入不是 Optional，但块输入是，用 Optional 包装循环输入。
    // 当循环接收 None 并输出非 None，或者反之时会发生这种情况。
    // 如果输入不是 OptionalType 且子块输入是可选的
    if (!input->type()->cast<OptionalType>() && sub_block_input_optional) {
      // 如果输入类型不是 NoneType
      if (!input->type()->cast<NoneType>()) {
        // 合并推断类型，返回合并后的类型及是否有推断结果
        auto [merged_type, inferred] = MergeInferredType(
            sub_block_input_optional->getElementType(), input->type());
        // 如果有推断结果
        if (inferred) {
          // 创建一个 OptionalType 对象
          sub_block_input_optional = OptionalType::create(merged_type);
          // 设置子块的输入类型为新创建的 OptionalType
          sub_block->inputs().at(i)->setType(sub_block_input_optional);
        }
      }
      // 创建一个 ONNXOptionalNode 节点，用于处理 OptionalType
      Node* opt_node = ONNXOptionalNode(sub_block_input_optional, graph);
      // 如果输入类型不是 NoneType
      if (!input->type()->cast<NoneType>()) {
        // 将输入节点添加到 ONNXOptionalNode 的输入中
        opt_node->addInput(input);
      }
      // 在当前节点之前插入 opt_node
      opt_node->insertBefore(node);
      // 用 opt_node 的输出替换当前节点对输入 input 的引用
      node->replaceInputWith(input, opt_node->output());
    }
  }
} // 匿名命名空间的结束

// 修复 ONNX 循环节点的输入和输出
std::vector<Value*> FixupONNXLoopNode(Node* node, int opset_version) {
  // 获取节点输出的数量
  auto output_size = node->outputs().size();
  // 调试信息：在 FixupONNXLoopBlockInputs 之前打印节点所属图的信息
  GRAPH_DEBUG("before FixupONNXLoopBlockInputs: ", *node->owningGraph());
  // 修复 ONNX 循环块的输入
  FixupONNXLoopBlockInputs(node);
  // 调试信息：在 FixupONNXLoopBlockInputs 之后打印节点所属图的信息
  GRAPH_DEBUG("after FixupONNXLoopBlockInputs: ", *node->owningGraph());
  // 修复 ONNX 循环节点的输入
  FixupONNXLoopNodeInputs(node, opset_version);
  // 调试信息：在 FixupONNXLoopNodeInputs 之后打印节点所属图的信息
  GRAPH_DEBUG("after FixupONNXLoopNodeInputs: ", *node->owningGraph());
  // 修复 ONNX 循环块的输出
  FixupONNXLoopBlockOutputs(node);
  // 调试信息：在 FixupONNXLoopBlockOutputs 之后打印节点所属图的信息
  GRAPH_DEBUG("after FixupONNXLoopBlockOutputs: ", *node->owningGraph());
  // 注意：输出顺序被有意改变以匹配期望的顺序，因为 ONNX 循环要求扫描输出在最后
  // 将序列依赖转换为新的输出
  auto new_outputs = ConvertSequenceDependencies(node, opset_version);
  // 复制块输出的类型到节点输出
  FixupONNXControlflowNodeOutputs(node);
  // 调试信息：在 FixupONNXControlflowNodeOutputs 之后打印节点所属图的信息
  GRAPH_DEBUG("after FixupONNXControlflowNodeOutputs: ", *node->owningGraph());
  // 断言：输出数量与新输出数量应该相等
  TORCH_INTERNAL_ASSERT(output_size == new_outputs.size());
  // 返回新的输出
  return new_outputs;
}

// 检查节点是否是 prim::Uninitialized 或 prim::Uninitialized->onnx::Identity 的输出
bool IsUninitializedNode(Node* n) {
  // 如果节点的类型是 ::c10::onnx::Identity 并且其输入的节点类型是 prim::Uninitialized，则返回 true
  if (n->kind() == ::c10::onnx::Identity &&
      n->inputs()[0]->node()->kind() == prim::Uninitialized)
    return true;
  // 如果节点的类型是 prim::Uninitialized，则返回 true
  if (n->kind() == prim::Uninitialized)
    return true;
  // 否则返回 false
  return false;
}

// 从另一个子块的输出推断未初始化输出的形状和类型
// prim::Uninitialized 节点被证明是未使用的，因此用推断的形状和类型替换该节点
void InferShapeTypeForUninitializedOutput(
    Graph* graph,
    Block* block,
    Value* uninitialized_output,
    Value* other_output,
    int opset_version) {
  Node* const_node = nullptr;
  // 如果其他输出的类型是 TensorType
  if (auto output_type = other_output->type()->cast<TensorType>()) {
    // 获取元素类型
    auto elem_type = at::initialTensorOptions().dtype(output_type->scalarType());
    // 创建一个常量节点
    const_node = graph->create(::c10::onnx::Constant, 1);
    // 如果输出类型具有确定的大小
    if (output_type->sizes().concrete_sizes().has_value()) {
      auto size = output_type->sizes().concrete_sizes().value();
      // 使用元素类型创建零填充的张量
      const_node->t_(attr::value, at::zeros(size, elem_type));
      // 设置节点输出的类型
      const_node->output()->setType(other_output->type());
    } else {
      // 如果输出类型没有确定的大小，则创建一个空的零维张量
      const_node->t_(attr::value, at::zeros({}, elem_type));
      // 设置节点输出的类型为默认的 TensorType
      const_node->output()->setType(
          TensorType::create(*(output_type->scalarType()), at::kCPU, {}, {}));
    }
  } else if (auto output_type = other_output->type()->cast<ListType>()) {
    // 如果其他输出的类型是 ListType
    TypePtr elem = output_type->getElementType();
    // 创建一个序列空节点
    const_node = graph->create(::c10::onnx::SequenceEmpty, 1);
    // 如果元素类型是 TensorType 并且具有标量类型
    if (elem->cast<TensorType>() &&
        elem->cast<TensorType>()->scalarType().has_value()) {
      auto scalar_type = elem->cast<TensorType>()->scalarType().value();
      auto onnx_type = ATenTypeToOnnxType(scalar_type);
      // 设置节点的数据类型属性
      const_node->i_(attr::dtype, onnx_type);
      // 设置节点输出的类型
      const_node->output()->setType(other_output->type());
    } else if (elem->cast<IntType>()) {
      // 如果 elem 能够转换为 IntType，则执行以下操作
      auto scalar_type = at::kLong;  // 设置标量类型为长整型
      auto onnx_type = ATenTypeToOnnxType(scalar_type);  // 将 ATen 类型转换为 ONNX 类型
      const_node->i_(attr::dtype, onnx_type);  // 在 const_node 上设置属性 dtype 为转换后的 ONNX 类型
      const_node->output()->setType(other_output->type());  // 设置 const_node 的输出类型为 other_output 的类型
    } else {
      // 如果 elem 类型无法转换为 IntType，则发出警告
      TORCH_WARN(
          "UninitializedOutput - Invalid elem Type of ListTensor found.");
      const_node->output()->setType(other_output->type());  // 设置 const_node 的输出类型为 other_output 的类型
    }
  } else if (auto output_type = other_output->type()->cast<OptionalType>()) {
    // 如果 other_output 的类型是 OptionalType
    const_node = ONNXOptionalNode(output_type, graph);  // 创建一个 ONNXOptionalNode，传入 output_type 和 graph
  }
  TORCH_CHECK(
      const_node,
      // 检查 const_node 是否存在，如果不存在则报错，指明无法从 other_output 的类型推断 prim::Uninitialized 节点的类型
      "Inferring type for prim::Uninitialized node from " +
          other_output->type()->repr_str() + " not supported.")
  const ParamMap empty_params_dict = {};  // 创建空的参数映射字典
  ONNXShapeTypeInference(const_node, empty_params_dict, opset_version);  // 使用 ONNXShapeTypeInference 推断 const_node 的形状和类型
  const_node->insertBefore(block->return_node());  // 将 const_node 插入到 return_node 前面
  const_node->copyMetadata(block->return_node());  // 复制 const_node 的元数据到 return_node
  uninitialized_output->replaceAllUsesWith(const_node->output());  // 用 const_node 的输出替换 uninitialized_output 的所有使用
  uninitialized_output->node()->destroy();  // 销毁 uninitialized_output 节点
// 如果节点类型不是 ONNX 的 If，直接返回，不进行修复操作
void ONNXFixupUninitializedOutput(Node* node, int opset_version) {
  // 输出当前图形的状态，用于调试和分析
  GRAPH_DUMP("Graph before fixing If shape type: ", node->owningGraph());
  // 获取当前的 If 节点和其所属的图形
  auto* if_node = node;
  auto* graph = if_node->owningGraph();

  // 检查输入到 ONNX If 节点是否为布尔类型，并插入必要的类型转换以确保为布尔类型
  if (!if_node->input()->type()->isSubtypeOf(*BoolType::get())) {
    // 插入类型转换节点，使输入到 If 节点的条件为布尔类型
    Node* cast_node =
        InsertCastForCond(if_node->input(), graph, if_node, opset_version);
    // 复制节点的元数据信息
    cast_node->copyMetadata(if_node);
  }

  // 获取 If 节点的两个子块：then 块和 else 块
  Block* then_block = if_node->blocks()[0];
  Block* else_block = if_node->blocks()[1];

  // 断言两个子块的输出数目相同，以便进行后续的形状和类型推断
  TORCH_INTERNAL_ASSERT(
      then_block->outputs().size() == else_block->outputs().size())

  // 遍历每个子块的输出，进行未初始化输出的形状和类型推断
  for (const auto i : c10::irange(else_block->outputs().size())) {
    Value* then_block_output = then_block->outputs()[i];
    Value* else_block_output = else_block->outputs()[i];

    // 如果两个子块的相同位置的输出都是未初始化节点，则无法推断其形状和类型
    TORCH_CHECK(
        !(IsUninitializedNode(then_block_output->node()) &&
          IsUninitializedNode(else_block_output->node())),
        "Cannot infer shape and type for ONNX If with uninitialized output in both subblocks. Please check the model graph.");

    // 如果 then 块的输出是未初始化节点，基于 else 块的输出推断其形状和类型
    if (IsUninitializedNode(then_block_output->node())) {
      InferShapeTypeForUninitializedOutput(
          graph,
          then_block,
          then_block_output,
          else_block_output,
          opset_version);
      // 设置 If 节点对应位置的输出类型为 then 块的输出类型
      if_node->outputs()[i]->setType(then_block->outputs()[i]->type());
    }
    // 如果 else 块的输出是未初始化节点，基于 then 块的输出推断其形状和类型
    else if (IsUninitializedNode(else_block_output->node())) {
      InferShapeTypeForUninitializedOutput(
          graph,
          else_block,
          else_block_output,
          then_block_output,
          opset_version);
      // 设置 If 节点对应位置的输出类型为 else 块的输出类型
      if_node->outputs()[i]->setType(else_block->outputs()[i]->type());
    }
  }
}
// 确认节点为 ONNX If 节点，如果不是则断言失败
void ONNXMergeIfBlockOutputShapes(Node* node) {
  TORCH_INTERNAL_ASSERT(node->kind() == ::c10::onnx::If);
  // 获取 If 节点的第一个分支块和第二个分支块
  Block* then_block = node->blocks().at(0);
  Block* else_block = node->blocks().at(1);

  // 断言两个分支块的输出数量相同
  TORCH_INTERNAL_ASSERT(
      then_block->outputs().size() == else_block->outputs().size())

  // 定义一个函数，查找两个符号形状的公共形状
  auto findCommonShape =
      [](const ::c10::SymbolicShape& a,
         const ::c10::SymbolicShape& b) -> ::c10::SymbolicShape {
    std::vector<::c10::ShapeSymbol> dims;
    // 如果两者的秩都存在且相等，则逐维比较形状符号
    if (a.rank() && b.rank() && a.rank() == b.rank()) {
      for (const auto j : c10::irange(a.rank().value())) {
        if (a[j] == b[j]) {
          dims.emplace_back(a[j]);
        } else {
          dims.emplace_back(::c10::ShapeSymbol::newSymbol());
        }
      }
      return ::c10::SymbolicShape(dims);
    }
    // 如果只有 a 的秩存在且大于 0，则返回 a 的符号形状
    if (a.rank() && a.rank().value() > 0) {
      return a;
    }
    // 如果只有 b 的秩存在且大于 0，则返回 b 的符号形状
    if (b.rank() && b.rank().value() > 0) {
      return b;
    }

    // 否则返回一个空的符号形状
    return ::c10::SymbolicShape();
  };

  // 定义一个函数，合并两个 TensorTypePtr 类型的对象
  auto mergeTensorType =
      [&findCommonShape](TensorTypePtr a, TensorTypePtr b) -> TensorTypePtr {
    if (a && b) {
      // 获取两个 TensorTypePtr 的符号形状
      const auto& a_shape = a->symbolic_sizes();
      const auto& b_shape = b->symbolic_sizes();
      // 查找它们的公共形状
      auto commonShape = findCommonShape(a_shape, b_shape);
      // 将公共形状应用于第一个 TensorTypePtr
      return a->withSymbolicShapes(commonShape);
    } else if (a) {
      // 如果只有第一个 TensorTypePtr 存在，则返回它
      return a;
    } else if (b) {
      // 如果只有第二个 TensorTypePtr 存在，则返回它
      return b;
    }
    // 如果两者都不存在，则返回空指针
    return nullptr;
  };

  // 定义一个函数，合并两个 ListTypePtr 类型的对象
  auto mergeListType = [&mergeTensorType](
                           ListTypePtr a, ListTypePtr b) -> ListTypePtr {
    if (a && b) {
      // 获取两个 ListTypePtr 中的元素类型，并转换为 TensorTypePtr
      auto a_tensor_type = a->getElementType()->cast<TensorType>();
      auto b_tensor_type = b->getElementType()->cast<TensorType>();
      // 合并两个 TensorTypePtr
      auto tensor_type = mergeTensorType(a_tensor_type, b_tensor_type);
      if (tensor_type) {
        // 如果成功合并，则返回包含合并后元素类型的 ListTypePtr
        return a->withContained({tensor_type})->cast<ListType>();
      }
      // 如果两个分支产生的 ListTypePtr 都没有张量形状信息，则返回其中一个 ListTypePtr
      return a;
    } else if (a) {
      // 如果只有第一个 ListTypePtr 存在，则返回它
      return a;
    } else if (b) {
      // 如果只有第二个 ListTypePtr 存在，则返回它
      return b;
    }
    // 如果两者都不存在，则返回空指针
    return nullptr;
  };

  // 定义一个函数，合并两个 OptionalTypePtr 类型的对象
  auto mergeOptionalType = [&mergeTensorType, &mergeListType](
                               OptionalTypePtr a,
                               OptionalTypePtr b) -> OptionalTypePtr {
    // 如果 a 和 b 都存在
    if (a && b) {
      // 如果 a 的元素类型可以转换为 TensorType
      if (a->getElementType()->cast<TensorType>()) {
        // 获取 a 的张量类型
        auto a_tensor_type = a->getElementType()->cast<TensorType>();
        // 获取 b 的张量类型
        auto b_tensor_type = b->getElementType()->cast<TensorType>();
        // 合并 a 和 b 的张量类型
        auto tensor_type = mergeTensorType(a_tensor_type, b_tensor_type);
        // 如果成功合并了张量类型
        if (tensor_type) {
          // 将 a 的元素类型设置为包含 tensor_type 的 OptionalType，并返回
          return a->withContained({tensor_type})->cast<OptionalType>();
        }
        // 如果无法合并张量类型，则返回 a（两个分支都产生不带张量形状的 OptionalType）
        return a;
      } else if (a->getElementType()->cast<ListType>()) {
        // 如果 a 的元素类型可以转换为 ListType
        auto a_list_type = a->getElementType()->cast<ListType>();
        // 获取 b 的 ListType
        auto b_list_type = b->getElementType()->cast<ListType>();
        // 合并 a 和 b 的 ListType
        auto list_type = mergeListType(a_list_type, b_list_type);
        // 如果成功合并了 ListType
        if (list_type) {
          // 将 a 的元素类型设置为包含 list_type 的 OptionalType，并返回
          return a->withContained({list_type})->cast<OptionalType>();
        }
        // 如果无法合并 ListType，则返回 a（两个分支都产生不带张量形状的 OptionalType）
        return a;
      }
    } else if (a) {
      // 如果只有 a 存在，则返回 a
      return a;
    } else if (b) {
      // 如果只有 b 存在，则返回 b
      return b;
    }
    // 如果 a 和 b 都不存在，则返回空指针
    return nullptr;
  };

  // 遍历 else_block 的所有输出
  for (const auto i : c10::irange(else_block->outputs().size())) {
    // 获取 node 的第 i 个输出
    Value* output_i = node->output(i);
    // 获取 then_block 的第 i 个输出的类型
    auto then_type = then_block->outputs().at(i)->type();
    // 获取 else_block 的第 i 个输出的类型
    auto else_type = else_block->outputs().at(i)->type();
    // 尝试将 then_type 转换为 TensorType
    auto then_tensor_type = then_type->cast<TensorType>();
    // 尝试将 else_type 转换为 TensorType
    auto else_tensor_type = else_type->cast<TensorType>();
    // 尝试将 then_type 转换为 ListType
    auto then_list_type = then_type->cast<ListType>();
    // 尝试将 else_type 转换为 ListType
    auto else_list_type = else_type->cast<ListType>();
    // 尝试将 then_type 转换为 OptionalType
    auto then_optional_type = then_type->cast<OptionalType>();
    // 尝试将 else_type 转换为 OptionalType
    auto else_optional_type = else_type->cast<OptionalType>();
    // 尝试将 then_type 转换为 NoneType
    auto then_none_type = then_type->cast<NoneType>();
    // 尝试将 else_type 转换为 NoneType
    auto else_none_type = else_type->cast<NoneType>();

    // 如果 then_type 或 else_type 是 TensorType
    if (then_tensor_type || else_tensor_type) {
      // 合并 then_tensor_type 和 else_tensor_type 的类型
      if (TypePtr merged_type = mergeTensorType(then_tensor_type, else_tensor_type)) {
        // 如果 then 或 else 任一为 OptionalType 或 NoneType，则创建 OptionalType
        if (else_optional_type || else_none_type || then_optional_type || then_none_type) {
          merged_type = OptionalType::create(merged_type);
        }
        // 将 output_i 的类型设置为 merged_type
        output_i->setType(merged_type);
      }
    } else if (then_list_type || else_list_type) {
      // 如果 then_type 或 else_type 是 ListType
      if (TypePtr merged_type = mergeListType(then_list_type, else_list_type)) {
        // 如果 then 或 else 任一为 OptionalType 或 NoneType，则创建 OptionalType
        if (else_optional_type || else_none_type || then_optional_type || then_none_type) {
          merged_type = OptionalType::create(merged_type);
        }
        // 将 output_i 的类型设置为 merged_type
        output_i->setType(merged_type);
      }
    }

    // 如果 then_type 或 else_type 是 OptionalType
    if (then_optional_type || else_optional_type) {
      // 合并 then_optional_type 和 else_optional_type 的类型
      if (auto optional_type = mergeOptionalType(then_optional_type, else_optional_type)) {
        // 将 output_i 的类型设置为 optional_type
        output_i->setType(optional_type);
        // 如果 then_branch 不是 OptionalType，则用 optional_type 替换 then_block 的第 i 个输出
        if (!then_optional_type) {
          ReplaceBlockOutputWithOptional(optional_type, then_block, i);
        } else if (!else_optional_type) {
          // 如果 else_branch 不是 OptionalType，则用 optional_type 替换 else_block 的第 i 个输出
          ReplaceBlockOutputWithOptional(optional_type, else_block, i);
        }
      }
    }
    # 如果条件满足：then_block 输出类型为 NoneType，且 else_block 输出类型不是 OptionalType
    if (then_none_type && !else_optional_type) {
        # 将输出 output_i 强制转换为 OptionalType，并在 then_block 中替换
        ReplaceBlockOutputWithOptional(
            output_i->type()->cast<OptionalType>(), then_block, i);
    }

    # 如果条件满足：else_block 输出类型为 NoneType，且 then_block 输出类型不是 OptionalType
    if (else_none_type && !then_optional_type) {
        # 将输出 output_i 强制转换为 OptionalType，并在 else_block 中替换
        ReplaceBlockOutputWithOptional(
            output_i->type()->cast<OptionalType>(), else_block, i);
    }
} // namespace jit
} // namespace torch
```