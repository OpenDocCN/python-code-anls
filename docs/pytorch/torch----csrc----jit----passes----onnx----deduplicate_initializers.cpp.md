# `.\pytorch\torch\csrc\jit\passes\onnx\deduplicate_initializers.cpp`

```
// 引入 Torch JIT 相关的头文件
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/deduplicate_initializers.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

// 引入 C10 库的实用工具
#include <c10/util/irange.h>

// 定义命名空间 torch::jit
namespace torch {
namespace jit {

// 定义命名空间 onnx，使用 c10::onnx 命名空间
namespace onnx {
using namespace ::c10::onnx;
}

// 函数 DeduplicateInitializers，用于去重初始化器
void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,                   // 输入参数，表示图的共享指针
    ValueToParamPairMap& valsToParamsMap,        // 输入参数，值到参数对的映射
    bool (*comp)(at::Tensor&, at::Tensor&)) {    // 输入参数，比较函数指针
  // lambda 函数 is_same_tensor_as，用于比较两个值是否指向相同的张量
  auto is_same_tensor_as = [&valsToParamsMap, comp](Value* v1) {
    return [&valsToParamsMap, v1, comp](Value* v2) {
      // 检查 v1 和 v2 是否在 valsToParamsMap 中存在
      if ((valsToParamsMap.find(v1) == valsToParamsMap.end()) ||
          (valsToParamsMap.find(v2) == valsToParamsMap.end())) {
        return false;
      }
      auto iv1 = valsToParamsMap.find(v1)->second.second;  // 获取 v1 的值
      auto iv2 = valsToParamsMap.find(v2)->second.second;  // 获取 v2 的值
      if (!iv1.isTensor() || !iv2.isTensor()) {             // 确保 iv1 和 iv2 是张量
        return false;
      }
      auto t1 = iv1.toTensor();   // 转换为张量
      auto t2 = iv2.toTensor();   // 转换为张量
      return comp(t1, t2);        // 使用 comp 函数指针比较张量 t1 和 t2
    };
  };

  std::vector<Value*> uniqueVals;           // 存储唯一值的向量
  std::vector<size_t> inputsIndicesToRemove;  // 存储要移除的输入索引

  auto b = g->block();  // 获取图的基本块

  // 遍历图的输入
  for (auto i : c10::irange(b->inputs().size())) {
    auto v = g->inputs().at(i);  // 获取输入值
    if (valsToParamsMap.find(v) == valsToParamsMap.end()) {
      // 如果值不在 valsToParamsMap 中，则跳过（跳过模型输入）
      continue;
    }
    // 查找是否已经存在相同的张量值
    auto it = std::find_if(
        uniqueVals.begin(), uniqueVals.end(), is_same_tensor_as(v));
    if (it == uniqueVals.end()) {
      uniqueVals.emplace_back(v);  // 将 v 加入到唯一值向量中
    } else {
      inputsIndicesToRemove.emplace_back(i);  // 将要移除的索引加入到向量中

      // 创建一个 Identity 节点
      auto id_node = g->create(onnx::Identity);
      id_node->insertAfter(g->block()->param_node());  // 在参数节点后插入
      id_node->addInput(*it);                          // 添加输入
      id_node->output()->copyMetadata(v);              // 复制元数据
      id_node->copyMetadata(g->block()->param_node()); // 复制块的元数据
      v->replaceAllUsesWith(id_node->output());        // 替换所有使用该值的地方为 Identity 节点的输出
    }
  }

  // 倒序移除要移除的输入
  for (auto it = inputsIndicesToRemove.rbegin();
       it != inputsIndicesToRemove.rend();
       ++it) {
    valsToParamsMap.erase(g->inputs().at(*it));  // 从 valsToParamsMap 中移除对应值
    g->eraseInput(*it);                          // 从图中移除输入
  }
}

// 按照数据指针去重初始化器的比较函数
bool DeduplicateInitializersByDataPtr(at::Tensor& t1, at::Tensor& t2) {
  return t1.sizes().equals(t2.sizes()) && t1.strides().equals(t2.strides()) &&
      (t1.has_storage() && t2.has_storage() && t1.data_ptr() == t2.data_ptr());
}

// 按照值去重初始化器的比较函数
bool DeduplicateInitializersByValue(at::Tensor& t1, at::Tensor& t2) {
  // 检查张量的类型、大小和步长是否相同
  if (t1.dtype() != t2.dtype() || !t1.sizes().equals(t2.sizes()) ||
      !t1.strides().equals(t2.strides())) {
    return false;
  }

  // 检查张量的设备是否相同
  if (t1.device() != t2.device()) {
    return t1.to("cpu").equal(t2.to("cpu"));  // 转换到 CPU 后比较
  }

  return t1.equal(t2);  // 直接比较张量值
}

// 重载 DeduplicateInitializers 函数，使用参数字典而不是参数映射
void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,        // 输入参数，表示图的共享指针
    std::map<std::string, IValue>& paramsDict,  // 输入参数，参数字典
    bool is_train) {                            // 输入参数，是否为训练模式
  auto valsToParamsMap = buildValueToParamsMap(g->block(), paramsDict);  // 构建值到参数映射
  // ONNX 规范不支持具有共享内存的参数。
  // 此步骤用于去重这些参数。训练不受影响。
  DeduplicateInitializers(g, valsToParamsMap, DeduplicateInitializersByDataPtr);
  if (!is_train) {
    // 更加激进的参数去重，基于张量数值。
    // 生成更紧凑的推理模型。
    // 对于训练，此步骤被禁用，
    // 因为参数可能会以不同的方式更新。
    DeduplicateInitializers(g, valsToParamsMap, DeduplicateInitializersByValue);
  }
  // 从值到参数的映射构建参数映射
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
}

} // namespace jit
} // namespace torch
```