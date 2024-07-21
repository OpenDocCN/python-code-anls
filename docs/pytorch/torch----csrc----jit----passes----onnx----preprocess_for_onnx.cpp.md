# `.\pytorch\torch\csrc\jit\passes\onnx\preprocess_for_onnx.cpp`

```py
// 包含头文件：torch/csrc/jit/passes/onnx/preprocess_for_onnx.h
// 包含头文件：ATen/ScalarOps.h 和 c10/util/irange.h
// 包含头文件：torch/csrc/jit/jit_log.h 和 torch/csrc/jit/passes/onnx/helper.h
#include <torch/csrc/jit/passes/onnx/preprocess_for_onnx.h>
#include <ATen/ScalarOps.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

// 命名空间：torch::jit
namespace torch {
namespace jit {

// 命名空间：onnx，使用 c10::onnx 命名空间
namespace onnx {
using namespace ::c10::onnx;
}

// 匿名命名空间：用于内部实现细节
namespace {

// 查找可融合的 ListUnpack 节点
at::optional<Node*> FindFusibleListUnpack(Node* n) {
  // 条件1：输出数量必须为1
  // 条件2：该输出仅被 prim::ListUnpack 节点使用
  if (n->outputs().size() != 1) {
    return at::nullopt;
  }
  if (n->output()->uses().size() != 1) {
    return at::nullopt;
  }
  auto listUnpackNode = n->output()->uses()[0].user;
  // 输出节点必须是 prim::ListUnpack 类型
  if (listUnpackNode->kind() != prim::ListUnpack) {
    return at::nullopt;
  }
  // 返回找到的 listUnpackNode 节点
  return listUnpackNode;
}

// 融合节点和 ListUnpack
// 例如，split/unbind 节点产生大小静态的 tensor[]，后续由 ListUnpack 解包。
// 该函数将这两个节点融合，并添加额外的输入 "_outputs"，用于告知符号函数输出的数量。
//
// 示例 IR
// split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor[]
// split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
//
// graph(%input : Float(5, 4, 3, strides=[12, 3, 1])):
//   %13 : int[] = prim::Constant[value=[2, 1, 2]]()
//   %7 : int = prim::Constant[value=0]()
//   %8 : Tensor[] = aten::split_with_sizes(%input, %13, %7)
//   %9 : Float(2, 4, 3, strides=[12, 3, 1]), %10 : Float(1, 4, 3, strides=[12,
//   3, 1]), %11 : Float(2, 4, 3, strides=[12, 3, 1]) = prim::ListUnpack(%8)
//   return (%9, %10, %11)
//
// 融合后
// graph(%input : Float(5, 4, 3, strides=[12, 3, 1])):
//   %13 : int[] = prim::Constant[value=[2, 1, 2]]()
//   %7 : int = prim::Constant[value=0]()
//   %8 : int = prim::Constant[value=3]()  # 添加值为3的额外输入，表示输出的数量。
//   %14 : Float(2, 4, 3, strides=[12, 3, 1]), %15 : Float(1, 4, 3, strides=[12,
//      3, 1]), %16 : Float(2, 4, 3, strides=[12, 3, 1] =
//      aten::split_with_sizes(%input, %13, %7, %8) return (%14, %15, %16)
void FuseWithListUnpack(Node* n) {
  // 查找可融合的 ListUnpack 节点
  auto found_listUnpack = FindFusibleListUnpack(n);
  if (!found_listUnpack) {
    return;
  }

  auto listUnpack_node = found_listUnpack.value();

  // 断言：节点的输出数量必须为1
  TORCH_INTERNAL_ASSERT(n->outputs().size() == 1);
  // 步骤1：向节点添加内部输入 "_outputs"，以便后续符号函数转换时知道输出的数量。
  // 步骤2：向节点添加准确的输出数量，复制元数据并替换 listUnpack 输出的使用。
  n->i_(
      Symbol::fromQualString("attr::_outputs"),
      static_cast<int64_t>(listUnpack_node->outputs().size()));

  // 遍历 listUnpack 节点的每个输出
  for (size_t i = 0; i < listUnpack_node->outputs().size(); ++i) {
    // 向节点添加新的输出
    auto new_output = n->addOutput();
    // 将新的输出复制元数据，使用列表解包节点的第 i 个输出的元数据
    new_output->copyMetadata(listUnpack_node->output(i));
  }
  // 移除列表解包节点的所有输入
  listUnpack_node->removeAllInputs();
  // 移除原始输出，即列表解包节点的输入
  n->eraseOutput(0);
  // 用节点 n 替换列表解包节点的所有使用
  listUnpack_node->replaceAllUsesWith(n);
// 递归函数，用于在节点块中查找特定类型的节点并进行处理
static void FuseWithListUnpack(Block* b) {
  // 遍历当前节点块中的所有节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    // 对当前节点块的子块进行递归处理
    for (auto* child_block : it->blocks()) {
      FuseWithListUnpack(child_block);
    }

    // 获取当前节点的类型
    auto n_kind = it->kind();
    // 根据节点类型进行不同的处理
    switch (n_kind) {
      // 对于以下几种节点类型，继续递归处理
      case aten::split:
      case aten::split_with_sizes:
      case aten::unsafe_split:
      case aten::unsafe_split_with_sizes:
      case aten::unbind:
      case aten::unsafe_chunk:
      case aten::where:
      case aten::nonzero_numpy:
        FuseWithListUnpack(*it);
        break;
      default:
        break;
    }
  }
}

// 在图中替换 aten::add 节点为 onnx::Concat 节点的处理函数
static void ReplaceAddWithConcat(Block* b) {
  // 遍历当前节点块中的所有节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    // 对当前节点块的子块进行递归处理
    for (auto* child_block : it->blocks()) {
      ReplaceAddWithConcat(child_block);
    }
    
    // 如果当前节点是 aten::add 类型的节点，则进行处理
    if (it->kind() == aten::add) {
      // 检查输入节点是否为 ListType，如果不是则跳过
      if (!it->input(0)->type()->cast<ListType>() ||
          !it->input(1)->type()->cast<ListType>()) {
        continue;
      }

      // 获取列表元素的类型
      const auto& elem =
          it->input(0)->type()->castRaw<ListType>()->getElementType();
      // 如果列表元素类型为 IntType，则进行替换操作
      if (elem->cast<IntType>()) {
        // 创建一个 onnx::Concat 节点
        Node* concat_node = b->owningGraph()->create(onnx::Concat, 1);
        concat_node->i_(attr::axis, 0);  // 设置 Concat 节点的参数
        concat_node->insertBefore(*it);  // 将 Concat 节点插入到当前节点之前
        concat_node->addInput(it->input(0));  // 添加输入到 Concat 节点
        concat_node->addInput(it->input(1));
        // 设置输出类型为对应的 TensorType
        concat_node->outputs()[0]->setType(TensorType::fromNumberType(*elem));
        concat_node->copyMetadata(*it);  // 复制元数据信息
        it->replaceAllUsesWith(concat_node);  // 替换当前节点的所有使用者为 Concat 节点
        it->removeAllInputs();  // 移除当前节点的所有输入
        it.destroyCurrent();  // 销毁当前节点
      }
    }
  }
}



// 处理当输入到 ListUnpack 是来自于非 ListConstruct 操作的 int[] 类型的情况
// 
// 在处理之前：
// graph(%x.1 : Float(2, 3, strides=[3, 1], requires_grad=0, device=cpu)):
//   %1 : None = prim::Constant()
//   %2 : int[] = aten::size(%x.1)
//   %a.1 : int, %b.1 : int = prim::ListUnpack(%2)
//
// 在处理之后：
// graph(%x.1 : Float(2, 3, strides=[3, 1], requires_grad=0, device=cpu)):
//   %1 : None = prim::Constant()
//   %2 : int[] = aten::size(%x.1)
//   %l1.1 : int[] = aten::list(%2)
//   %a.1 : int, %b.1 : int = prim::ListUnpack(%l1.1)
// 这段代码实现了将图中的 prim::ListUnpack 节点与其父节点 prim::ListConstruct 融合的功能。
static void fuseListAndListUnpack(Block* b) {
  // 迭代当前块的每个节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    // 递归调用，处理当前节点的子块
    for (auto* child_block : it->blocks()) {
      fuseListAndListUnpack(child_block);
    }
    // 如果当前节点是 prim::ListUnpack
    if (it->kind() == prim::ListUnpack) {
      // 遍历 ListUnpack 的输出
      for (const auto i : c10::irange(it->outputs().size())) {
        auto output = it->outputs().at(i);
        // 检查 ListUnpack 的输入条件，确保可以进行融合操作
        if (it->inputs().size() == 1 &&
            it->input()->node()->kind() != prim::ListConstruct &&
            it->input()->type()->cast<ListType>() &&
            it->input()->type()->castRaw<ListType>()->getElementType()->cast<IntType>()) {
          // 创建一个 Constant 节点作为 Gather 操作的输入索引
          Node* gather_indices = b->owningGraph()->create(onnx::Constant, 1);
          gather_indices->insertBefore(*it);
          gather_indices->t_(attr::value, at::scalar_to_tensor(at::Scalar(int(i))));
          
          // 创建 Gather 节点，将 ListUnpack 的输入与 Constant 节点连接起来
          Node* gather_node = b->owningGraph()->create(onnx::Gather, 1);
          gather_node->insertBefore(*it);
          gather_node->addInput(it->input());
          gather_node->addInput(gather_indices->output());
          gather_node->copyMetadata(*it);
          
          // 用 Gather 节点的输出替换 ListUnpack 的输出
          output->replaceAllUsesWith(gather_node->output());
        }
      }
    }
  }
}

// 将图预处理为 ONNX 格式，包括执行融合操作
void PreprocessForONNX(std::shared_ptr<Graph>& graph) {
  // 执行 ListUnpack 与 ListConstruct 融合操作
  fuseListAndListUnpack(graph->block());
  GRAPH_DUMP("After fuseListAndListUnpack: ", graph);
}
```