# `.\pytorch\torch\csrc\jit\passes\canonicalize_graph_fuser_ops.cpp`

```py
// 包含必要的头文件，用于在 Torch JIT 中进行操作和优化
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 定义 torch 命名空间和 jit 命名空间
namespace torch {
namespace jit {

// 定义结构体 ChunkOutput，表示从 Node 节点输出的值及其偏移量
struct ChunkOutput {
  ChunkOutput(Value* v, size_t o) : val(v), offset(o){};
  Value* val;    // 指向值的指针
  size_t offset; // 偏移量
};

// 静态函数，用于获取给定 Chunk Node 的输出
static std::optional<std::vector<ChunkOutput>> getChunkOutputs(Node* chunk) {
  std::vector<ChunkOutput> outputs;  // 存储 Chunk Node 的输出列表
  // 遍历 Chunk Node 的输出用途
  for (auto list_use : chunk->output()->uses()) {
    // 检查是否匹配特定的 aten::select 操作，并且输出类型为 TensorType
    if (list_use.user->matches(
            "aten::select(t[] list, int idx) -> t", attr::idx) &&
        list_use.user->output()->type()->cast<TensorType>()) {
      // 将匹配的输出添加到 outputs 中，包括值和索引
      outputs.emplace_back(
          list_use.user->output(),
          list_use.user->get<int64_t>(attr::idx).value());
    } else if (list_use.user->kind() == prim::ListUnpack) {
      // 处理 prim::ListUnpack 操作，将其输出添加到 outputs 中
      // 如果输出大小不能被 chunks 属性整除，则返回空
      if (static_cast<int64_t>(list_use.user->outputs().size()) !=
          chunk->get<int64_t>(attr::chunks).value()) {
        return c10::nullopt;
      }
      auto unpack_outputs = list_use.user->outputs();
      // 遍历 ListUnpack 的输出，并将每个输出及其索引添加到 outputs 中
      for (const auto i : c10::irange(unpack_outputs.size())) {
        outputs.emplace_back(unpack_outputs[i], i);
      }
    } else {
      return c10::nullopt;  // 如果有未匹配的输出用途，返回空
    }
  }
  return outputs;  // 返回获取到的输出列表
}

// 静态函数，用于对 Block 内部的操作进行规范化
static void CanonicalizeOps(Block* block) {
  // 遍历 Block 内的每个 Node
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    // 递归调用 CanonicalizeOps 处理子 Block
    for (auto sub : it->blocks())
      CanonicalizeOps(sub);
    
    // 检查 Node 是否匹配特定的操作类型，然后执行相应的规范化操作
    if (it->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches("aten::mul(Tensor self, Tensor other) -> Tensor") ||
        it->matches("aten::div(Tensor self, Tensor other) -> Tensor")) {
      
      // 将 rank 0 的 Tensor 常量替换为标量常量
      if (auto other = it->get<at::Tensor>(attr::other)) {
        if (other->dim() == 0) {
          WithInsertPoint insert_guard{*it};  // 在当前节点处插入新节点
          auto graph = it->owningGraph();     // 获取节点所属的图
          auto new_other = graph->insertConstant(other->item());  // 插入新的常量节点
          std::vector<Value*> inputs = it->inputs().vec();  // 获取输入节点列表
          inputs.at(1) = new_other;  // 替换第二个输入为新的常量节点
          Value* new_output =
              graph->insertNode(graph->create(it->kind(), inputs))->output();  // 插入新节点
          new_output->node()->copyMetadata(*it);  // 复制节点元数据
          new_output->copyMetadata(it->output());  // 复制输出节点的元数据
          it->output()->replaceAllUsesWith(new_output);  // 替换所有使用当前节点输出的节点
        }
      }
    }
    // 这里缺少对于 Node 其他操作类型的处理逻辑，需要根据实际情况补充
  }
}

// 结束 jit 命名空间
} // namespace jit
} // namespace torch
    } else if (it->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      // 如果当前操作符与指定的aten::chunk操作符匹配，并且其常量输入包含attr::chunks和attr::dim
      // 则执行以下替换操作
      // 将aten::chunk操作替换为ConstantChunk操作，并展开其输出

      if (auto orig_outputs = getChunkOutputs(*it)) {
        // 如果可以获取到aten::chunk操作的原始输出
        WithInsertPoint guard(*it);
        // 设置插入点为当前操作的位置
        auto* self = it->namedInput(attr::self);
        // 获取输入中名为attr::self的Tensor指针
        auto* graph = it->owningGraph();
        // 获取当前操作所属的图对象
        const auto chunks = it->get<int64_t>(attr::chunks).value();
        // 获取attr::chunks常量输入的值
        const auto dim = it->get<int64_t>(attr::dim).value();
        // 获取attr::dim常量输入的值
        auto* node =
            graph->insertNode(graph->create(prim::ConstantChunk, chunks));
        // 在图中插入一个新节点，类型为prim::ConstantChunk，并设置chunk数目为chunks
        node->addInput(self);
        // 将self作为ConstantChunk节点的输入
        node->i_(attr::chunks, chunks)->i_(attr::dim, dim);
        // 设置ConstantChunk节点的属性attr::chunks和attr::dim
        node->copyMetadata(*it);
        // 复制当前操作的元数据到ConstantChunk节点上
        for (const auto& orig_out : *orig_outputs) {
          // 遍历原始输出中的每个元素
          orig_out.val->replaceAllUsesWith(node->outputs()[orig_out.offset]);
          // 将原始输出的值替换为ConstantChunk节点相应的输出
          node->outputs()[orig_out.offset]->setType(orig_out.val->type());
          // 设置ConstantChunk节点输出的类型与原始输出的类型相同
        }
      }
    }
  }


这段代码主要是对特定操作符 `aten::chunk` 的处理逻辑。根据匹配条件替换为 `prim::ConstantChunk` 操作，并确保输出正确地连接到新节点的输出上。
}

// 结束命名空间 "jit"

// 结束命名空间 "torch"

void CanonicalizeOps(const std::shared_ptr<Graph>& graph) {
    // 调用 CanonicalizeOps 函数处理 graph 对象中的基本块
    CanonicalizeOps(graph->block());
    // 在控制台输出 graph 的信息，带有指定的前缀字符串
    GRAPH_DUMP("After CanonicalizeOps: ", graph);
    // 删除图中的死代码
    EliminateDeadCode(graph);
}
```