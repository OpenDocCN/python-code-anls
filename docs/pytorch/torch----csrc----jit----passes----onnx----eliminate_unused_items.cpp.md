# `.\pytorch\torch\csrc\jit\passes\onnx\eliminate_unused_items.cpp`

```
#include <torch/csrc/jit/passes/onnx/eliminate_unused_items.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/Optional.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

// 定义函数 EliminateUnusedItemsONNX，用于消除 ONNX 模型中未使用的项
void EliminateUnusedItemsONNX(Block* b, ParamMap& paramsDict) {
  // 构建值到参数映射，传入块和参数字典，返回映射关系
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  // 从映射中删除未使用的值
  eraseUnusedValuesFromMap(valsToParamsMap);
  // 从块的输入中删除未使用的项
  eraseUnusedBlockInputs(b);
  // 从值到参数映射中构建参数映射
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  // 函数结束
  return;
}

} // namespace jit
} // namespace torch
```