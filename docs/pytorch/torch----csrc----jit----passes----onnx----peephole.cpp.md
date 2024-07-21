# `.\pytorch\torch\csrc\jit\passes\onnx\peephole.cpp`

```
// 包含 Torch 库中的头文件，用于 ONNX 相关的优化操作
#include <torch/csrc/jit/passes/onnx/peephole.h>

// 包含 C10 库中的异常处理和范围迭代工具
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 包含 Torch 的 JIT 日志和死代码消除优化相关的头文件
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 包含 Torch 的 ONNX 辅助函数
#include <torch/csrc/jit/passes/onnx/helper.h>

// 包含 ATen 库中的标量操作
#include <ATen/ScalarOps.h>

// 根据预编译指令选择包含不同的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full.h>
#include <ATen/ops/ones_like_native.h>
#endif

// 包含 C10 中的 Optional 类型定义
#include <c10/util/Optional.h>

// 如果是在 Visual Studio 编译环境下，定义 ssize_t 类型
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// Torch 的 JIT 命名空间开始
namespace torch {
namespace jit {

// Torch 的 ONNX 命名空间使用在这里
namespace onnx {
using namespace ::c10::onnx;
}

// 判断节点是否为 RNN 类型的函数
bool isRNN(const Node* node) {
  auto k = node->kind();
  return k == onnx::RNN || k == onnx::LSTM || k == onnx::GRU;
}

// 判断是否为无操作的转置操作的函数
bool isNopTranspose(const std::vector<int64_t>& perm) {
  for (int64_t i = 0, perm_size = perm.size(); i < perm_size; i++) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}

// 组合两个转置操作的函数，使得两个转置操作的效果等同于一个
std::vector<int64_t> composeTransposes(
    const std::vector<int64_t>& t1,
    const std::vector<int64_t>& t2) {
  // 断言两个转置操作的大小相同
  TORCH_INTERNAL_ASSERT(t1.size() == t2.size());
  std::vector<int64_t> ret;
  ret.reserve(t1.size());
  // 对 t2 中的每个元素进行处理
  for (const auto& i : t2) {
    // 断言索引 i 在 t1 的有效范围内
    TORCH_INTERNAL_ASSERT(i < int64_t(t1.size()));
    // 将 t1[t2[i]] 添加到结果向量中
    ret.push_back(t1[i]);
  }
  return ret;
}

// 获取节点中支持广播操作的位置的函数
std::vector<size_t> getBroadcastPositions(Node* node) {
  // 定义支持广播操作的节点类型及其对应的位置
  static std::unordered_map<NodeKind, std::vector<size_t>> broadcast_positions =
      {
          {onnx::Add, {0, 1}},
          {onnx::Div, {0, 1}},
          {onnx::Mul, {0, 1}},
          {onnx::Pow, {0, 1}},
          {onnx::Sub, {0, 1}},
          {onnx::Gemm, {2}},
          {onnx::Equal, {0, 1}},
          {onnx::Greater, {0, 1}},
          {onnx::Less, {0, 1}},
      };
  // 定义空的位置向量
  static std::vector<size_t> no_positions;
  std::vector<size_t> positions;

  // 查找当前节点类型是否在支持广播操作的映射中
  auto iter = broadcast_positions.find(node->kind());
  if (iter != broadcast_positions.end()) {
    // 如果存在，则添加对应位置到 positions 向量中
    for (size_t position : iter->second) {
      if (position < node->inputs().size()) {
        positions.emplace_back(position);
      }
    }
    return positions;
  }
  return no_positions;
}

// 判断一个张量是否可以广播到另一个张量的函数
// 返回一个可选的大小类型，用于确定是否可以将 `from` 扩展到 `to`，并返回扩展轴的位置。
std::optional<size_t> fusibleExpandTo(
    at::IntArrayRef from,
    at::IntArrayRef to) {
  // 如果 `from` 的维度大于 `to`，则无法进行扩展，返回空的可选类型。
  if (from.size() > to.size()) {
    return c10::nullopt;
  }

  // 逐维度比较 `from` 和 `to` 的最后几个维度，若 `from` 中对应维度为 1 或者与 `to` 不匹配，返回空的可选类型。
  for (const auto i : c10::irange(from.size())) {
    auto fdim = from[from.size() - 1 - i];
    auto tdim = to[to.size() - 1 - i];
    if (fdim != 1 && fdim != tdim) {
      return c10::nullopt;
    }
  }

  // 返回可以进行扩展的轴的位置（即 `to` 的长度减去 `from` 的长度）。
  return to.size() - from.size();
}

// 将扩展调用融合为 ONNX 操作，以便非步进后端更高效地进行广播，这是局部信息。对于 PyTorch 而言，这种优化是不必要的，因为 'expand' 操作是免费的。
void fuseBroadcast(Block* b) {
  // 遍历每个节点 `n` 在块 `b` 中的节点列表
  for (auto n : b->nodes()) {
    // 递归处理每个节点 `n` 的子块 `child_block`
    for (auto* child_block : n->blocks()) {
      fuseBroadcast(child_block);
    }

    // 获取节点 `n` 的广播位置
    auto broadcast_positions = getBroadcastPositions(n);
    // 如果广播位置不为空，确保节点 `n` 没有 `attr::axis` 属性
    if (!broadcast_positions.empty()) {
      TORCH_INTERNAL_ASSERT(!n->hasAttribute(attr::axis));
    }

    // 遍历广播位置，处理每个位置的扩展节点
    for (size_t position : broadcast_positions) {
      auto* expand_node = n->input(position)->node();

      // 确认扩展节点的类型
      if (expand_node->kind() != aten::expand ||
          expand_node->input(1)->node()->kind() != onnx::Constant ||
          expand_node->input(2)->node()->kind() != onnx::Constant) {
        continue;
      }

      auto* unexpanded_input = expand_node->input(0);

      // 需要了解扩展前的张量类型。我们通常应该拥有这个信息（因为扩展只会被跟踪，不会从符号生成），但如果某些情况下缺少此信息，则跳过。
      if (!unexpanded_input->isCompleteTensor() ||
          !n->output()->isCompleteTensor()) {
        continue;
      }

      // 获取可融合的扩展位置
      std::optional<size_t> axis = fusibleExpandTo(
          unexpanded_input->type()
              ->expectRef<TensorType>()
              .sizes()
              .concrete_sizes()
              .value(), // from
          n->output()
              ->type()
              ->expectRef<TensorType>()
              .sizes()
              .concrete_sizes()
              .value()); // to
      if (axis == c10::nullopt) {
        continue;
      }

      // 将节点 `n` 的输入位置 `position` 替换为未扩展输入，并销毁原始的扩展节点（如果没有被使用）
      n->replaceInput(position, unexpanded_input);
      if (!expand_node->hasUses()) {
        expand_node->destroy();
      }
    }
  }
}

void fuseConsecutiveTransposes(Block* b) {
  // 遍历块 `b` 中的每个节点 `n`
  for (auto n : b->nodes()) {
    // 递归处理节点 `n` 的子块 `child_block`
    for (auto* child_block : n->blocks()) {
      fuseConsecutiveTransposes(child_block);
    }
    // 继续实现该函数的其余部分
    // 检查当前节点 n 是否为转置操作，并且其输入节点也是转置操作，且二者在同一个基本块中
    if (n->kind() == onnx::Transpose &&
        n->input()->node()->kind() == onnx::Transpose &&
        n->owningBlock() == n->input()->node()->owningBlock()) {
      // 将原始输入节点保存到 origInput
      auto origInput = n->input();
      // 将当前节点 n 的置换属性设置为合成后的置换结果
      n->is_(
          attr::perm,
          composeTransposes(
              origInput->node()->is(attr::perm), n->is(attr::perm)));
      // 替换当前节点 n 的输入为原始输入节点的输入
      n->replaceInput(0, origInput->node()->input());
      // 如果原始输入节点不再被使用，则销毁原始输入节点及其所属节点
      if (origInput->uses().empty()) {
        origInput->node()->destroy();
      }
      // 继续处理下一个节点
      continue;
    }
  }
void eliminateNopTranspose(Block* b) {
  // 遍历当前块中的所有节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto n = *it;
    // 递归处理每个节点中的子块
    for (auto* child_block : n->blocks()) {
      eliminateNopTranspose(child_block);
    }
    // 如果当前节点是Transpose操作
    if (n->kind() == onnx::Transpose) {
      // 检查是否为无操作的Transpose
      if (isNopTranspose(n->is(attr::perm))) {
        // 替换输出使用为输入，从而消除Transpose操作
        n->output()->replaceAllUsesWith(n->input());
        // 销毁当前节点
        it.destroyCurrent();
        continue;
      }
    }
  }
}

void fuseTransposeIntoGemm(Block* b) {
  // 简单的转置顺序 {1, 0}
  static const std::vector<int64_t> simpleTransPerm({1, 0});

  // 遍历当前块中的所有节点
  for (auto n : b->nodes()) {
    // 递归处理每个节点中的子块
    for (auto* child_block : n->blocks()) {
      fuseTransposeIntoGemm(child_block);
    }
    // 如果当前节点是Gemm操作
    if (n->kind() == onnx::Gemm) {
      // 遍历两个输入（A和B）
      for (size_t i : {0, 1}) {
        auto inp = n->inputs()[i];
        auto trans = i == 0 ? attr::transA : attr::transB;
        // 如果输入是Transpose操作，并且转置的顺序符合简单的转置顺序
        if (inp->node()->kind() == onnx::Transpose &&
            inp->node()->is(attr::perm) == simpleTransPerm) {
          // 替换Gemm操作的输入为Transpose操作的输入
          n->replaceInput(i, inp->node()->input());
          // 更新Gemm操作的转置属性
          n->i_(trans, n->hasAttribute(trans) ? !n->i(trans) : 1);
          // 如果Transpose操作的输出没有其他使用，销毁该Transpose操作
          if (inp->uses().empty()) {
            inp->node()->destroy();
          }
        }
      }
    }
  }
}

// Why this is here:
//
//   Pytorch has a "packed" representation of sequences, as well as a
//   "padded" representation. ONNX has only one representation,
//   corresponding to pytorch's "padded". Therefore, we need to remove
//   any use of packed sequences before exporting.
//
// What this does:
//
//   This code uses the observation that
//     RNN(PackPadded(x)) == PackPadded(RNN(x))
//   and converts the first form to the second whenever possible,
//   "pushing" the packing operation past the RNN operation. Then,
//   the removeNopPacking pass removes the packing operations
//   entirely by pairing them with their inverse PadPacked. If the
//   input graph does not pair the operations, export will fail.
void pushPackingPastRnn(Block* b) {
  // 遍历当前块中的所有节点
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
    auto* n = *it;
    // 递归处理每个节点中的子块
    for (auto* child_block : n->blocks()) {
      pushPackingPastRnn(child_block);
    }

    // 如果当前节点不是PackPadded操作，继续下一个节点
    if (n->kind() != prim::PackPadded) {
      continue;
    }
    // 当PackPadded操作的输出只有一个使用者时才处理
    if (n->outputs().at(0)->uses().size() != 1) {
      continue;
    }
    Node* rnn = n->outputs()[0]->uses()[0].user;
    // 如果PackPadded操作的输出被RNN操作使用
    if (!isRNN(rnn)) {
      continue;
    }

    // 如果RNN操作和PackPadded操作在同一个块中
    if (rnn->owningBlock() != n->owningBlock()) {
      continue;
    }

    // 如果RNN操作的输出没有被其他地方使用，并且PackPadded的第二个输出有一个使用者
    if (rnn->outputs().at(0)->uses().empty() &&
        n->outputs().at(1)->uses().size() == 1) {
      // 替换PackPadded操作的输出为其输入，移除PackPadded操作
      n->outputs().at(0)->replaceAllUsesWith(n->inputs().at(0));
      n->outputs().at(1)->replaceFirstUseWith(n->inputs().at(1));
      it.destroyCurrent();
      continue;
    }

    // 如果RNN操作后面跟着Transpose和Reshape（双向RNN）或者Squeeze（单向RNN）
    // 这部分注释未提供完整信息，但是应该在代码中进行处理
  }
}
    // 获取当前节点的第一个输出的第一个使用者（user）
    Node* next = rnn->outputs().at(0)->uses().at(0).user;
    // 如果下一个节点是Transpose，则继续找下一个节点
    if (next->kind() == onnx::Transpose) {
      next = next->outputs().at(0)->uses().at(0).user;
      // 如果下一个节点不是Reshape，则继续下一轮循环
      if (next->kind() != onnx::Reshape) {
        continue;
      }
    } else if (next->kind() != onnx::Squeeze) {
      // 如果下一个节点不是Squeeze，则继续下一轮循环
      continue;
    }

    // 将当前节点的第一个输出替换为当前节点的第一个输入，移除 PackPadded 在 RNN 前面的影响
    n->outputs().at(0)->replaceAllUsesWith(n->inputs().at(0));

    // 获取当前节点的第二个输出，通常用于 batch_sizes
    Value* batch_sizes = n->outputs().at(1);
    // 遍历 batch_sizes 的所有使用者
    while (!batch_sizes->uses().empty()) {
      // 获取第一个使用者
      Use use_0 = batch_sizes->uses().at(0);
      // 获取使用者节点
      Node* user = use_0.user;
      // 如果是特定模式的 Gather 操作，执行替换操作
      if (use_0.offset == 0 && user->kind() == onnx::Gather &&
          user->i(attr::axis) == 0 &&
          user->inputs().at(1)->node()->kind() == onnx::Constant &&
          user->inputs().at(1)->node()->hasAttribute(attr::value)) {
        const at::Tensor& const_val_t =
            user->inputs().at(1)->node()->t(attr::value);
        // 如果常量值不为0，可能会产生无效的图形结构，中断循环
        if (const_val_t.item().toInt() != 0) {
          // We'll likely produce an invalid graph if this happens.
          break;
        }
        // 获取 RNN 的输入
        Value* rnn_input = rnn->inputs().at(0);
        // 创建 Shape 节点
        Node* shape = b->owningGraph()->create(onnx::Shape);
        shape->insertAfter(rnn_input->node());
        shape->addInput(rnn_input);
        shape->copyMetadata(n);
        // 替换 batch_sizes 的第一个使用者为 Shape 节点的输出
        batch_sizes->replaceFirstUseWith(shape->output());
        // 创建新的 Constant 节点，确保它不与其他节点共享
        Node* gather_indices = b->owningGraph()->create(onnx::Constant, 1);
        gather_indices->t_(attr::value, at::native::ones_like(const_val_t));
        gather_indices->copyMetadata(n);
        gather_indices->insertBefore(user);
        // 替换 Gather 操作的第二个输入为新的 Constant 节点的输出
        user->replaceInput(1, gather_indices->output());
      }
      // 如果使用者是 RNN 节点，则将其替换为当前节点的第二个输入
      else if (user == rnn) {
        batch_sizes->replaceFirstUseWith(n->inputs().at(1));
      } else {
        // 如果存在其他用途不属于 PadPacked 或死代码，则可能产生无效的图形结构，中断循环
        break;
      }
    }

    // 在 next 节点之后插入新的 PackPadded 节点
    Node* newPackPadded = b->owningGraph()->create(prim::PackPadded, 2);
    newPackPadded->copyMetadata(n);
    newPackPadded->insertAfter(next);
    newPackPadded->copyMetadata(next);

    // 使其它节点从新的 PackPadded 节点消费数据
    // 将 next 节点的第一个输出替换为 newPackPadded 节点的第一个输出
    next->outputs().at(0)->replaceAllUsesWith(newPackPadded->outputs().at(0));
    // 将 n 节点的第二个输出替换为 newPackPadded 节点的第二个输出
    n->outputs().at(1)->replaceAllUsesWith(newPackPadded->outputs().at(1));

    // 设置新的 PackPadded 节点的输入
    newPackPadded->addInput(next->outputs().at(0));
    newPackPadded->addInput(n->inputs().at(1));

    // 参考 https://github.com/pytorch/pytorch/issues/9043 查看完整描述。
    // 由于 PackPadded 目前处理不卫生，PyTorch 最终传播了不正确的类型。
    // 在长期清理出现之前，我们可以通过重新设置大小来修复这个问题。
    TensorTypePtr oldType = rnn->inputs().at(0)->type()->cast<TensorType>();
    if (oldType && oldType->isComplete()) {
      // 创建一个新的大小向量
      std::vector<int64_t> new_sizes;
      new_sizes.push_back(*oldType->sizes()[0]);
      new_sizes.push_back(*oldType->sizes()[1]);
      if (next->kind() == onnx::Reshape) {
        // 双向循环神经网络
        new_sizes.push_back(rnn->i(attr::hidden_size) * 2);
      } else {
        // 单向循环神经网络
        new_sizes.push_back(rnn->i(attr::hidden_size));
      }
      // 创建新的 TensorType，确保连续性
      TensorTypePtr newType = TensorType::createContiguous(
          *oldType->scalarType(), *oldType->device(), new_sizes);
      // 设置 next 节点的第一个输出的类型
      next->outputs().at(0)->setType(newType);
    }

    // 销毁当前迭代器指向的节点
    it.destroyCurrent();
// 从图中删除无效的填充节点，保留有效的填充节点。
// 填充节点应该是将在稍后被消除的死代码。
void removeNopPacking(Block* graph) {
  // 迭代图中的每个节点
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    auto* n = *it;
    // 递归调用以处理子块中的节点
    for (auto* child_block : n->blocks()) {
      removeNopPacking(child_block);
    }

    // 如果当前节点不是 PadPacked 类型，则跳过处理
    if (n->kind() != prim::PadPacked) {
      continue;
    }
    // 获取 PadPacked 节点的输入节点
    Node* input = n->inputs()[0]->node();
    // 如果输入节点不是 PackPadded 类型，则跳过处理
    if (input->kind() != prim::PackPadded) {
      continue;
    }
    // 如果输入节点的第一个输出不等于 PadPacked 的第一个输入，则跳过处理
    if (input->outputs()[0] != n->inputs()[0]) {
      continue;
    }
    // 如果输入节点的第二个输出不等于 PadPacked 的第二个输入，则跳过处理
    if (input->outputs()[1] != n->inputs()[1]) {
      continue;
    }
    // 将 PadPacked 的输出替换为 PackPadded 的输入
    n->outputs()[0]->replaceAllUsesWith(input->inputs()[0]);
    n->outputs()[1]->replaceAllUsesWith(input->inputs()[1]);

    // 清空 PadPacked 的所有输入
    n->removeAllInputs();
    // 移除当前处理的节点
    it.destroyCurrent();
  }
}

// 修正虚构 PadPacked 节点的形状
void hackFixupPadPackedShapes(Block* graph) {
  // 遍历图中的每个节点
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    auto* n = *it;
    // 递归调用以处理子块中的节点
    for (auto* child_block : n->blocks()) {
      removeNopPacking(child_block);
    }

    // 如果当前节点不是 PadPacked 类型，则跳过处理
    if (n->kind() != prim::PadPacked) {
      continue;
    }
    // 获取 PadPacked 节点的输入节点
    Node* input = n->inputs()[0]->node();
    // 将输入节点的第一个输出的类型设置为 PadPacked 的第一个输出的类型
    input->outputs()[0]->setType(n->outputs()[0]->type());
  }
}

// 修复默认的 RNN 状态
void fixDefaultRNNState(
    Graph* graph,
    Node* n,
    int input_index,
    int opset_version) {
  auto initial_state = n->inputs()[input_index];

  // 在 PyTorch 中，RNN 代码接受一个可选的隐藏状态。
  // 当提供输入时，一切正常。
  // 当没有提供输入时，将通过构造一个新的变量进行默认初始化，
  // 并被跟踪为具有预期形状的 ConstantOfShape。
  // 当批次大小固定时，一切正常。
  // 当指定了 h0 和 c0 但它们不是模型的输入（它们是常量）且批次大小可变时，
  // 应该使用批次大小为 1 来保存模型（否则会出错），
  // 并且以批次大小为 1 保存 h0 和 c0 的值。
  // 当以不同的批次大小调用模型时，h0 和 c0 将被广播以获取正确的形状。
  // 识别最后一种模式（4）并修复形状。
  // 注意，对于多层 RNN，Constant 和 RNN 之间会有一个 Slice 操作。
  bool needsFixing = initial_state->node()->kind() == onnx::Constant ||
      (initial_state->node()->kind() == onnx::Slice &&
       initial_state->node()->inputs()[0]->node()->kind() == onnx::Constant);

  // 如果不需要修复，则返回
  if (!needsFixing) {
    return;
  }
  
  // 创建一个形状节点，用于表示输入的形状
  Node* shape_of_input = graph->create(onnx::Shape, 1);
  // 将节点的元数据从节点 n 复制过来
  shape_of_input->copyMetadata(n);
  // 将形状节点插入到节点 n 之前
  shape_of_input->insertBefore(n);
  // 给形状节点添加输入，使用节点 n 的第一个输入
  shape_of_input->addInput(n->inputs()[0]);
  
  // 创建一个常量节点，用于指定 gather 操作的索引值为 1
  Node* gather_indices = graph->create(onnx::Constant, 1);
  // 复制节点 n 的元数据到 gather_indices 节点
  gather_indices->copyMetadata(n);
  // 将 gather_indices 节点插入到节点 n 之前
  gather_indices->insertBefore(n);
  // 设置 gather_indices 的值为标量 1，转换为张量
  gather_indices->t_(attr::value, at::scalar_to_tensor(at::Scalar(1)));
  
  // 创建一个 gather 操作节点，用于获取形状节点的第一个输出和 gather_indices 节点的输出
  Node* batch_size = graph->create(onnx::Gather, 1);
  // 复制节点 n 的元数据到 batch_size 节点
  batch_size->copyMetadata(n);
  // 将 batch_size 节点插入到节点 n 之前
  batch_size->insertBefore(n);
  // 将形状节点的第一个输出作为 batch_size 节点的输入
  batch_size->addInput(shape_of_input->outputs()[0]);
  // 将 gather_indices 节点的输出作为 batch_size 节点的输入
  batch_size->addInput(gather_indices->outputs()[0]);
  
  // 创建一个 unsqueeze 操作节点，用于在 batch_size 节点的输出上添加维度
  Node* unsqueezed_batch_size =
      createONNXUnsqueeze(graph, n, batch_size->outputs()[0], 0, opset_version);
  
  // 创建一个常量节点，用于指定隐藏状态的大小
  Node* hidden_size = graph->create(onnx::Constant, 1);
  // 复制节点 n 的元数据到 hidden_size 节点
  hidden_size->copyMetadata(n);
  // 将 hidden_size 节点插入到节点 n 之前
  hidden_size->insertBefore(n);
  // 设置 hidden_size 的值为一个长整型张量，大小为节点 n 的隐藏状态大小
  hidden_size->t_(
      attr::value,
      at::full(
          {1},
          n->i(attr::hidden_size),
          at::kLong)); // at::Scalar(n->i(attr::hidden_size)).toTensor());
  
  // 创建一个常量节点，用于指定方向的数量
  Node* num_directions = graph->create(onnx::Constant, 1);
  // 复制节点 n 的元数据到 num_directions 节点
  num_directions->copyMetadata(n);
  // 将 num_directions 节点插入到节点 n 之前
  num_directions->insertBefore(n);
  // 设置 num_directions 的值为一个标量张量，根据节点的方向属性决定是单向还是双向
  num_directions->t_(
      attr::value,
      scalar_to_tensor(at::Scalar(
          n->hasAttribute(attr::direction) &&
                  n->s(attr::direction) == "bidirectional"
              ? 2
              : 1)));
  
  // 创建一个 unsqueeze 操作节点，用于在 num_directions 节点的输出上添加维度
  Node* unsqueezed_num_directions = createONNXUnsqueeze(
      graph, n, num_directions->outputs()[0], 0, opset_version);
  
  // 创建一个 concat 操作节点，用于将各种维度连接成一个张量
  Node* concated_dims = graph->create(onnx::Concat, 1);
  // 复制节点 n 的元数据到 concated_dims 节点
  concated_dims->copyMetadata(n);
  // 将 concated_dims 节点插入到节点 n 之前
  concated_dims->insertBefore(n);
  // 设置 concat 的轴为 0，表示按照第一个维度连接
  concated_dims->i_(attr::axis, 0);
  // 将 unsqueezed_num_directions 节点的输出作为 concat 节点的输入
  concated_dims->addInput(unsqueezed_num_directions->outputs()[0]);
  // 将 unsqueezed_batch_size 节点的输出作为 concat 节点的输入
  concated_dims->addInput(unsqueezed_batch_size->outputs()[0]);
  // 将 hidden_size 节点的输出作为 concat 节点的输入
  concated_dims->addInput(hidden_size->outputs()[0]);
  
  // 创建一个 expand 操作节点，用于将初始状态扩展为指定形状
  Node* fixed_init_state = graph->create(onnx::Expand, 1);
  // 复制节点 n 的元数据到 fixed_init_state 节点
  fixed_init_state->copyMetadata(n);
  // 将 fixed_init_state 节点插入到节点 n 之前
  fixed_init_state->insertBefore(n);
  // 将初始状态作为 fixed_init_state 节点的输入
  fixed_init_state->addInput(initial_state);
  // 将 concated_dims 节点的输出作为 fixed_init_state 节点的输入
  fixed_init_state->addInput(concated_dims->outputs()[0]);
  // 将节点 n 的输入索引替换为 fixed_init_state 节点的输出
  n->replaceInput(input_index, fixed_init_state->outputs()[0]);
  
  // 如果初始状态的使用次数为零，则销毁初始状态节点
  if (initial_state->uses().empty()) {
    initial_state->node()->destroy();
  }
}

void fixDefaultRnnHiddenState(Block* b, int opset_version) {
  // 遍历当前块的所有节点
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
    auto* n = *it;
    // 对当前节点的所有子块递归调用fixDefaultRnnHiddenState函数
    for (auto* child_block : n->blocks()) {
      fixDefaultRnnHiddenState(child_block, opset_version);
    }

    // 如果当前节点不是RNN类型的节点，则继续下一个节点
    if (!isRNN(n)) {
      continue;
    }
    // 如果当前节点的输入小于6个，则继续下一个节点
    // RNN类型的节点的第六个输入是隐藏状态
    if (n->inputs().size() < 6) {
      continue;
    }
    // 调用fixDefaultRNNState函数，修复RNN节点的默认状态
    fixDefaultRNNState(b->owningGraph(), n, 5, opset_version);
  }
}

void fixDefaultLstmCellState(Block* b, int opset_version) {
  // 遍历当前块的所有节点
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
    auto* n = *it;
    // 对当前节点的所有子块递归调用fixDefaultLstmCellState函数
    for (auto* child_block : n->blocks()) {
      fixDefaultLstmCellState(child_block, opset_version);
    }

    // 如果当前节点不是LSTM类型的节点，则继续下一个节点
    if (n->kind() != onnx::LSTM) {
      continue;
    }
    // 如果当前节点的输入小于7个，则继续下一个节点
    // LSTM类型的节点的第七个输入是细胞状态
    if (n->inputs().size() < 7) {
      continue;
    }
    // 调用fixDefaultRNNState函数，修复LSTM节点的默认状态
    fixDefaultRNNState(b->owningGraph(), n, 6, opset_version);
  }
}

static bool isSafeToSpeculate(Node* n) {
  // 判断节点是否是安全的推测节点，即是否是Transpose节点
  return n->kind() == onnx::Transpose;
}

// 将操作移到控制流块外，以便始终执行，无论控制流条件的结果如何。
// 仅需要这样做是为了ONNX优化器的分割(pass)将操作放入init_net中。
// TODO: 一旦caffe2/python/onnx/backend.py中的代码不再调用optimize_onnx，删除此函数。
static void speculateOps(Block* block) {
  // 遍历块中的所有节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* n = *it;
    ++it; // 注意：先增加迭代器，以便在需要时安全移动节点

    // 对节点的所有子块递归调用speculateOps函数
    for (auto b : n->blocks()) {
      speculateOps(b);
    }
    // 如果不是安全的推测节点，则继续下一个节点
    if (!isSafeToSpeculate(n)) {
      continue;
    }
    // 只适用于只有一个输入的节点
    // 将节点移到其所嵌套的控制流之外
    auto node_input = n->input()->node();
    if (node_input->owningBlock() == n->owningBlock()) {
      continue;
    }
    // 如果此节点的输出是块输出的一部分，则跳过
    bool is_block_output = false;
    for (auto node_output : n->outputs()) {
      for (auto node_output_use : node_output->uses()) {
        if (node_output_use.user == n->owningBlock()->return_node()) {
          is_block_output = true;
          break;
        }
      }
      if (is_block_output) {
        break;
      }
    }
    if (is_block_output) {
      continue;
    }
    // 查找与node_input所在的块相同的块中包含节点n的控制流节点
    auto control_flow_node = n->owningBlock()->owningNode();
    while (control_flow_node->owningBlock() != node_input->owningBlock()) {
      control_flow_node = control_flow_node->owningBlock()->owningNode();
    }
    // 将节点放在此流节点之前
    n->moveBefore(control_flow_node);
  }
}
static void replaceInputWithList(Node* node, size_t i, ArrayRef<Value*> to) {
  // 移除节点的第 i 个输入
  node->removeInput(i);
  // 将列表中的每个值作为新输入插入节点中
  for (auto* to_val : to) {
    // 确保新输入值属于相同的计算图
    TORCH_INTERNAL_ASSERT(to_val->owningGraph() == node->owningGraph());
    node->insertInput(i++, to_val);
  }
}

static void eraseListConstruct(Block* block, int opset_version);

static void eraseListConstruct(Node* n, int opset_version) {
  // 遍历节点 n 中的每个子块，并应用 eraseListConstruct 函数
  for (auto b : n->blocks()) {
    eraseListConstruct(b, opset_version);
  }
  // 存储替换操作的元组的列表
  std::vector<std::tuple<size_t, std::vector<Value*>>> replacements;

  // 获取节点 n 所属的块
  auto block = n->owningBlock();
  size_t i = 0;
  // 遍历节点 n 的每个输入
  for (auto* input : n->inputs()) {
    // 检查当前输入节点是否为 prim::ListConstruct 类型
    if (input->node()->kind() == prim::ListConstruct) {
      auto* lc_node = input->node();
      // 获取列表元素类型
      TypePtr elem = lc_node->output()->type()->castRaw<ListType>()->getElementType();
      // 如果元素类型是 IntType，并且可以转换为 ONNX Concat 节点，则进行转换
      if (elem->cast<IntType>() &&
          isValidToTransformToONNXConcatNode(lc_node)) {
        // 将 ListConstruct 转换为 ONNX Concat 节点
        auto concat_node = transformToONNXConcatNode(
            block->owningGraph(), input->node(), false, opset_version);
        // 复制元数据信息到新的 Concat 节点
        concat_node->copyMetadata(n);
        // 将 Concat 节点的输出作为新的输入值，替换 ListConstruct 节点
        replacements.emplace_back(
            i, std::vector<Value*>({concat_node->output()}));
      } else {
        // 如果 Opset 版本大于等于 11，创建一个 SequenceConstruct 或 SequenceEmpty 节点替换 ListConstruct 节点
        if (opset_version >= OPSET_VERSION_11) {
          c10::Symbol seq_node_kind = !lc_node->inputs().empty()
              ? onnx::SequenceConstruct
              : onnx::SequenceEmpty;
          Node* seq_node = block->owningGraph()->create(
              seq_node_kind, {lc_node->inputs()}, 1);
          seq_node->copyMetadata(n);
          seq_node->insertBefore(lc_node);
          seq_node->output()->copyMetadata(lc_node->output());
          seq_node->copyMetadata(lc_node);
          // 用新创建的 Sequence 节点替换 ListConstruct 节点的所有使用
          lc_node->replaceAllUsesWith(seq_node);
        }
      }
    }
    i++;
  }

  // 反向遍历替换列表，并执行替换操作
  for (auto ritr = replacements.rbegin(); ritr != replacements.rend(); ++ritr) {
    replaceInputWithList(n, std::get<0>(*ritr), std::get<1>(*ritr));
  }
}

static void eraseListConstruct(Block* block, int opset_version) {
  // TODO: 修复此部分/可能移除这一部分。
  // 张量列表也可能用于 meshgrid 等操作。
  // 遍历块中的每个节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* n = *it;
    ++it;

    // 对每个节点应用 eraseListConstruct 函数
    eraseListConstruct(n, opset_version);
  }
  // 对块的返回节点应用 eraseListConstruct 函数
  eraseListConstruct(block->return_node(), opset_version);
}

static void eraseListUnpack(Block* block, int opset_version);

// 将 prim::ListUnpack 替换为 onnx::SequenceAt。
static void eraseListUnpack(Node* n, int opset_version) {
  // 遍历节点 n 中的每个子块，并应用 eraseListUnpack 函数
  for (auto b : n->blocks()) {
    eraseListUnpack(b, opset_version);
  }

  // 检查节点 n 是否为 prim::ListUnpack 类型
  if (n->kind() == prim::ListUnpack) {
    // 如果 Opset 版本小于 11，则抛出异常
    if (opset_version < OPSET_VERSION_11) {
      throw std::runtime_error(
          "Unsupported: ONNX export of prim::ListUnpack in opset " +
          std::to_string(opset_version) + ". Please try opset version 11.");
    }
    // 如果 Opset 版本大于等于 11，继续替换操作
    // （此处代码未完整给出）
    # 获取当前节点所属的计算图
    auto g = n->owningGraph();
    # 遍历当前节点的所有输出
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      # 创建一个常量节点，其值为当前输出的索引值 i
      auto seq_idx_n = g->create(onnx::Constant, 1);
      seq_idx_n->t_(attr::value, at::scalar_to_tensor(at::Scalar(int64_t(i))));
      # 将新创建的常量节点插入到当前节点 n 的前面

      seq_idx_n->insertBefore(n);

      # 创建一个 SequenceAt 节点，用于获取序列中特定位置的元素
      auto seq_at_n = g->create(onnx::SequenceAt, 1);
      # 设置 SequenceAt 节点的输入，即当前节点 n 的输入和刚刚创建的常量节点的输出
      seq_at_n->addInput(n->input());
      seq_at_n->addInput(seq_idx_n->output());
      # 设置 SequenceAt 节点的输出类型与当前节点输出 i 的类型相同
      seq_at_n->output()->setType(n->output(i)->type());
      # 将 SequenceAt 节点插入到当前节点 n 的前面
      seq_at_n->insertBefore(n);
      # 复制当前节点 n 的元数据到 SequenceAt 节点
      seq_at_n->copyMetadata(n);
      # 替换当前节点输出 i 的所有使用处为 SequenceAt 节点的输出
      n->output(i)->replaceAllUsesWith(seq_at_n->output());
    }
  }
// 静态方法，用于在给定的块（Block）中执行递归删除列表解包操作
static void eraseListUnpack(Block* block, int opset_version) {
  // 迭代器遍历该块中的所有节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    // 获取当前节点
    Node* n = *it;
    // 移动迭代器到下一个节点
    ++it;

    // 递归调用，删除子块中的列表解包操作
    eraseListUnpack(n, opset_version);
  }
}

// 该函数用于融合 ListConstruct 和 ListUnpack 操作，优化代码，减少死代码的存在
// 其中 ListConstruct 和 ListUnpack 可能会被优化掉
static void fuseListConstructListUnpack(Block* b) {
  // 迭代器遍历块中的所有节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    // 递归处理节点的子块
    for (auto* child_block : it->blocks()) {
      fuseListConstructListUnpack(child_block);
    }
    // 如果当前节点是 ListUnpack，并且其输入节点是 ListConstruct
    if (it->kind() == prim::ListUnpack &&
        it->input()->node()->kind() == prim::ListConstruct) {
      // 替换 ListUnpack 的输出为 ListConstruct 的输入
      for (const auto i : c10::irange(it->outputs().size())) {
        auto output = it->outputs().at(i);
        output->replaceAllUsesWith(it->input()->node()->inputs().at(i));
      }
    }
  }
}

// 从块中移除 TupleConstruct 操作，这些操作通常在量化领域中使用，并被其他量化操作符消耗
static void eraseTupleConstruct(Block* block) {
  std::vector<Value*> new_block_outputs;
  bool found_tuple_construct = false;
  // 遍历块的所有输出
  for (auto* output : block->outputs()) {
    auto output_node = output->node();
    // 如果输出节点是 TupleConstruct
    if (output_node->kind() == prim::TupleConstruct) {
      found_tuple_construct = true;
      // 将 TupleConstruct 的输入作为新的块输出
      for (auto* input : output_node->inputs()) {
        new_block_outputs.emplace_back(input);
      }
    } else {
      // 否则直接添加到新的块输出中
      new_block_outputs.emplace_back(output);
    }
  }
  // 如果找到了 TupleConstruct，则移除所有原始输出，并注册新的输出
  if (found_tuple_construct) {
    block->removeAllOutputs();
    for (auto* output : new_block_outputs) {
      block->registerOutput(output);
    }
  }
}

// 该函数移除 MaxPool 操作的未使用输出
void removeMaxPoolUnusedOutput(Block* b) {
  // 迭代器遍历块中的所有节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto n = *it;
    // 递归处理节点的子块
    for (auto* child_block : n->blocks()) {
      removeMaxPoolUnusedOutput(child_block);
    }
    // 如果当前节点是 MaxPool，并且第二个输出没有使用者，则移除第二个输出
    if (strcmp(n->kind().toQualString(), "onnx::MaxPool") == 0) {
      if (n->outputs().size() == 2 && n->outputs().at(1)->uses().empty()) {
        it->eraseOutput(1);
      }
    }
  }
}

// 该函数优化 LogSoftmax 和 NegativeLogLikelihoodLoss 操作符，融合为 SoftmaxCrossEntropyLoss 操作
static void fuseLogSoftmaxNllLoss(Block* b) {
  // 迭代器遍历块中的所有节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    // 递归处理节点的子块
    for (auto* child_block : it->blocks()) {
      fuseLogSoftmaxNllLoss(child_block);
    }
    // 这里是一个额外的大括号，可能是代码复制错误，应该删除
    }
  }
}
// 移除序列拆分和连接操作的优化，当以下条件满足时才会进行优化：
//  1. SplitToSequence 的输出没有被其他节点使用。
//  2. SplitToSequence 的属性 keepdims 和 axis 与 ConcatFromSequence 的属性 new_axis 和 axis 匹配。
// 在这种情况下，两个操作的组合是无操作的，可以安全地移除。
static void removeSequenceSplitConcat(Block* b) {
  // 遍历节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    // 递归处理子块
    for (auto* child_block : it->blocks()) {
      removeSequenceSplitConcat(child_block);
    }
    // 检查节点类型是否为 ConcatFromSequence，且输入节点类型为 SplitToSequence
    if (it->kind() == onnx::ConcatFromSequence &&
        it->input()->node()->kind() == onnx::SplitToSequence) {
      // 如果输入节点被多个节点使用，则跳过
      if (it->input()->uses().size() > 1) {
        continue;
      }

      auto split_node = it->input()->node();
      auto concat_node = *it;

      // 获取属性值
      const auto split_axis =
          split_node->hasAttribute(attr::axis) ? split_node->i(attr::axis) : 0;
      const auto split_keepdims = split_node->hasAttribute(attr::keepdims)
          ? split_node->i(attr::keepdims)
          : 1;
      const auto concat_axis = concat_node->i(attr::axis);
      const auto concat_new_axis = concat_node->hasAttribute(attr::new_axis)
          ? concat_node->i(attr::new_axis)
          : 0;
      const bool has_input_split = split_node->inputs().size() == 2;

      // 如果输入节点有两个输入，则跳过
      if (has_input_split) {
        continue;
      }

      // 如果 SplitToSequence 的 keepdims 与 ConcatFromSequence 的 new_axis 相等，则跳过
      if (split_keepdims == concat_new_axis) {
        continue;
      }

      // 如果 SplitToSequence 的 axis 与 ConcatFromSequence 的 axis 不相等，则跳过
      if (split_axis != concat_axis) {
        continue;
      }

      // 替换 ConcatFromSequence 的输出为 SplitToSequence 的输入
      concat_node->output()->replaceAllUsesWith(split_node->input());
    }
  }
}

// 为在块中用作输出的块输入插入 Identity 节点，以解决 ONNX 的限制
static void insertIdentityForInputUsedAsOutput(Block* b) {
  // 遍历块的输出
  for (auto out : b->outputs()) {
    auto n = out->node();
    // 如果节点不为空且类型为 prim::Param
    if (nullptr != n && n->kind() == prim::Param) {
      // 创建 Identity 节点并插入到块中
      Node* id_node = b->owningGraph()->create(onnx::Identity);
      id_node->insertBefore(b->return_node());
      id_node->addInput(out);
      id_node->output()->setType(out->type());
      // 替换块的返回节点输入为 Identity 节点的输出
      b->return_node()->replaceInputWith(out, id_node->output());
    }
  }

  // 递归处理子块
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      insertIdentityForInputUsedAsOutput(child_block);
    }
  }
}

// 这个优化执行 ONNX 特定的 peephole 优化
//
// 在这里编写优化之前，请问自己，“我能在 ATen 操作上执行这个优化吗”？
// 如果可以，应该认真考虑在 jit/passes/peephole.cpp 中编写优化，因为它将通用适用于 JIT。
// 这里的优化仅适用于 ONNX 导出。
void PeepholeOptimizeONNX(
    std::shared_ptr<Graph>& graph,
    int opset_version,
    // TODO: decide on fixpoint strategy
    // TODO: make it easier not to do O(k) iterations over the graph, where
    // k is the number of distinct peephole optimizations
    // 对于图中的每个块，执行一些修复以适应填充打包形状
    hackFixupPadPackedShapes(graph->block());
    // 将打包操作推送到 RNN 操作之前
    pushPackingPastRnn(graph->block());
    // 从图中删除无操作打包
    removeNopPacking(graph->block());
    // 如果批处理大小是可变的，则只需要修复隐藏状态和单元状态的默认大小
    if (!fixed_batch_size) {
      fixDefaultRnnHiddenState(graph->block(), opset_version);
      fixDefaultLstmCellState(graph->block(), opset_version);
    }
    // 融合广播操作
    fuseBroadcast(graph->block());
    // 融合连续的转置操作
    fuseConsecutiveTransposes(graph->block());
    // 消除无操作的转置
    eliminateNopTranspose(graph->block());
    // 将转置操作融合到GEMM操作中
    fuseTransposeIntoGemm(graph->block());
    // 推测运算操作
    speculateOps(graph->block());
    // 融合列表构造和列表解包操作
    fuseListConstructListUnpack(graph->block());
    // 融合LogSoftmax和NLL损失计算操作
    fuseLogSoftmaxNllLoss(graph->block());
    // 删除列表构造操作节点
    eraseListConstruct(graph->block(), opset_version);
    // 删除元组构造操作节点
    eraseTupleConstruct(graph->block());
    // 消除死代码，允许删除具有副作用的节点
    EliminateDeadCode(
        graph->block(),
        true,
        DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
    // 删除列表解包操作节点
    eraseListUnpack(graph->block(), opset_version);
    // 删除MaxPool未使用的输出
    removeMaxPoolUnusedOutput(graph->block());
    // 删除序列分割和连接操作
    removeSequenceSplitConcat(graph->block());
    // 为作为输出使用的输入插入Identity操作
    insertIdentityForInputUsedAsOutput(graph->block());
    
    // 在Peephole优化ONNX后，对图进行转储
    GRAPH_DUMP("After PeepholeOptimizeONNX", graph);
} // 关闭命名空间 jit
} // 关闭命名空间 torch
```