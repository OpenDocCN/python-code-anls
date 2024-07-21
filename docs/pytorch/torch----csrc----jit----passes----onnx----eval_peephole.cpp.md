# `.\pytorch\torch\csrc\jit\passes\onnx\eval_peephole.cpp`

```
// 包含 Torch 库中的头文件，用于 JIT 日志记录
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch 库中的头文件，用于 ONNX 相关的优化传递
#include <torch/csrc/jit/passes/onnx/eval_peephole.h>
// 包含 Torch 库中的头文件，提供 ONNX 相关的辅助函数
#include <torch/csrc/jit/passes/onnx/helper.h>
// 包含 Torch 主库的头文件
#include <torch/torch.h>

// 包含 C10 库中的 Optional 类型头文件
#include <c10/util/Optional.h>
// 包含 C10 库中的 irange 头文件，提供迭代范围
#include <c10/util/irange.h>
// 包含 C++ 标准库的算法头文件
#include <algorithm>

// Torch 库的 JIT 命名空间
namespace torch {
namespace jit {

// Torch JIT 中的 ONNX 命名空间，使用 C10 和 ONNX 的命名空间
namespace onnx {
using namespace ::c10::onnx;
}

// 获取节点中输入值的张量表示
std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  // 获取节点输入的数量
  size_t numInputs = node->inputs().size();
  // 存储输入值张量的向量
  std::vector<at::Tensor> inputTensorValues;
  // 预留空间以避免重新分配
  inputTensorValues.reserve(numInputs);
  // 遍历节点的每个输入值
  for (auto val : node->inputs()) {
    // 如果输入值关联的节点类型为 prim::Param
    if (val->node()->kind() == prim::Param) {
      // 查找值到参数对的映射
      auto itr = valsToParamsMap.find(val);
      // 如果映射中找不到对应项，则跳过
      if (itr == valsToParamsMap.end()) {
        continue;
      }
      // 将参数对应的张量值添加到输入张量的向量中
      inputTensorValues.push_back(itr->second.second.toTensor());
    }
    // 如果输入值关联的节点类型为 onnx::Constant
    else if (val->node()->kind() == onnx::Constant) {
      // 将常量节点的张量值添加到输入张量的向量中
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      // 其他情况下跳过当前输入值
      continue;
    }
  }
  // 返回收集到的输入张量向量
  return inputTensorValues;
}

// 将 Conv 和 BatchNorm 融合为 Conv 节点的优化传递
static void fuseConvBatchNorm(Block* b, ValueToParamPairMap& valsToParamsMap) {
  // 遍历块中的每个节点
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    // 对于每个节点，如果它包含子块，则递归应用融合操作
    for (auto* child_block : it->blocks()) {
      fuseConvBatchNorm(child_block, valsToParamsMap);
    }
    // 处理当前节点本身
    // 这里可以假定在这个函数中会有融合 Conv 和 BatchNorm 的逻辑处理
  }
}

// 在 ONNX 图上执行评估的轻量级优化传递
void EvalPeepholeONNX(Block* b, ParamMap& paramsDict) {
  // 构建值到参数对的映射
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  // 应用 Conv 和 BatchNorm 融合优化
  fuseConvBatchNorm(b, valsToParamsMap);
  // 根据值到参数映射构建参数映射
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
}

// 在 ONNX 图上执行评估的轻量级优化传递
void EvalPeepholeONNX(std::shared_ptr<Graph>& g, ParamMap& paramsDict) {
  // 对图的主块应用评估的轻量级优化传递
  EvalPeepholeONNX(g->block(), paramsDict);
  // 输出优化传递后的图的描述
  GRAPH_DUMP("After EvalPeepholeONNX:", g);
}

} // namespace jit
} // namespace torch
```