# `.\pytorch\torch\csrc\jit\passes\onnx\prepare_division_for_onnx.cpp`

```
// 包含头文件：torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h
#include <torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h>

// 包含头文件：torch/csrc/jit/ir/constants.h 和 torch/csrc/jit/jit_log.h
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>

// 命名空间：torch::jit，开始定义函数和类
namespace torch {
namespace jit {

// 函数：为ONNX准备除法操作，处理块内的节点
static void PrepareDivisionForONNXOnBlock(Block* block) {
  // 遍历块内的每一个节点
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    // 遍历每个节点中的子块，并对每个子块递归调用PrepareDivisionForONNXOnBlock函数
    for (auto sub : it->blocks()) {
      PrepareDivisionForONNXOnBlock(sub);
    }
    // 设置插入点为当前节点
    WithInsertPoint guard(*it);
    // 获取当前节点所属的计算图
    auto* subgraph = it->owningGraph();

    // 如果当前节点匹配 "aten::div(int a, int b) -> float"
    if (it->matches("aten::div(int a, int b) -> float")) {
      // 转换为浮点数类型进行除法操作
      std::vector<Value*> floattensor_inputs =
          fmap(it->inputs(), [&](Value* input) {
            // 插入节点：将输入转换为张量
            auto* longtensor =
                subgraph->insertNode(subgraph->createNumToTensor(input))
                    ->output();
            // 复制输入节点的元数据到转换后的节点
            longtensor->node()->copyMetadata(input->node());
            // 插入常量节点，值为0
            auto* nonblocking = subgraph->insertConstant(0);
            // 创建转换节点，将长整型张量转换为浮点型张量
            auto* cast =
                subgraph->create(aten::_cast_Float, {longtensor, nonblocking});
            // 复制当前节点的元数据到转换节点
            cast->copyMetadata(*it);
            // 插入转换节点并返回其输出
            return subgraph->insertNode(cast)->output();
          });

      // 替换当前节点的输入为转换后的浮点数输入
      it->replaceInput(0, floattensor_inputs[0]);
      it->replaceInput(1, floattensor_inputs[1]);
      // 设置当前节点的输出类型为浮点数张量类型
      it->output()->setType(TensorType::fromNumberType(*FloatType::get()));
    }
  }
}

// 函数：为ONNX准备除法操作，处理整个计算图
void PrepareDivisionForONNX(const std::shared_ptr<Graph>& graph) {
  // 调用PrepareDivisionForONNXOnBlock函数，处理整个计算图的根块
  PrepareDivisionForONNXOnBlock(graph->block());
  // 打印处理后的计算图，用于调试和分析
  GRAPH_DUMP("After PrepareDivisionForONNX: ", graph);
}

} // namespace jit
} // namespace torch
```