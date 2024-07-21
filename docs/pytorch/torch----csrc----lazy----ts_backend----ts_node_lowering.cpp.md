# `.\pytorch\torch\csrc\lazy\ts_backend\ts_node_lowering.cpp`

```
// 引入需要的头文件来支持 Lazy Tensor 引擎节点的降低操作
#include <torch/csrc/lazy/ts_backend/ts_node_lowering.h>

// 引入其他必要的头文件
#include <ATen/Functions.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/ts_backend/ir_builder.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

// 命名空间声明开始
namespace torch {
namespace lazy {

// LowerBuiltin 函数，用于降低内置操作节点
static TSOpVector LowerBuiltin(
    const torch::lazy::Node* node,
    std::shared_ptr<torch::jit::GraphFunction> function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  // 调用 LowerTSBuiltin 函数，将内置操作节点降低为 Lazy Tensor 操作向量
  return LowerTSBuiltin(function, node->op().op, arguments, kwarguments);
}

// 另一个版本的 LowerBuiltin 函数，用于降低内置操作节点
static TSOpVector LowerBuiltin(
    c10::Symbol sym,
    std::shared_ptr<torch::jit::GraphFunction> function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  // 调用 LowerTSBuiltin 函数，将内置操作节点降低为 Lazy Tensor 操作向量
  return LowerTSBuiltin(function, sym, arguments, kwarguments);
}

// LowerTSBuiltin 函数，实现 Lazy Tensor 操作节点的降低
TSOpVector LowerTSBuiltin(
    std::shared_ptr<torch::jit::GraphFunction> function,
    c10::Symbol sym,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments) {
  // 创建一个内置函数对象
  auto builtin =
      std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
  // 创建一个魔术方法对象
  auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
  // 调用魔术方法，执行函数调用并获取返回值
  auto ret = magic_method->call({}, *function, arguments, kwarguments, 0);
  // 将返回值转换为 SimpleValue 类型
  auto& sv = dynamic_cast<torch::jit::SimpleValue&>(*ret);
  // 检查返回的值是否为元组类型
  if (sv.getValue()->type()->kind() == c10::TypeKind::TupleType) {
    // 如果返回值是元组类型，将其转换为 Lazy Tensor 操作向量
    const auto tuple_call_result = sv.asTuple({}, *function);
    TSOpVector tuple_result;
    for (const auto& tuple_component : tuple_call_result) {
      auto tuple_component_sv =
          dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
      tuple_result.push_back(tuple_component_sv->getValue());
    }
    return tuple_result;
  }
  // 如果返回值不是元组类型，直接返回包含单个操作的 Lazy Tensor 操作向量
  return {sv.getValue()};
}

// GenerateClone 函数，生成给定值的克隆
static torch::jit::Value* GenerateClone(
    torch::jit::Value* val,
    std::shared_ptr<torch::jit::GraphFunction> function) {
  // 创建克隆操作的参数列表
  std::vector<torch::jit::NamedValue> clone_arguments;
  clone_arguments.emplace_back(val);
  // 调用 LowerBuiltin 函数，生成克隆操作并获取结果
  TSOpVector cloned = LowerBuiltin(at::aten::clone, function, clone_arguments);
  // 检查克隆结果是否只包含一个操作
  TORCH_CHECK_EQ(cloned.size(), 1);
  // 返回克隆操作的结果值
  return cloned.front();
}

// 节点降低操作的默认实现
// TSNode 类的 Lower 方法
TSOpVector TsNode::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  // 创建参数列表
  std::vector<torch::jit::NamedValue> arguments;
  // 遍历操作数列表
  for (const torch::lazy::Output& output : operands()) {
    // 将 loctx->GetOutputOp(output) 返回的值添加到 arguments 向量的末尾
    arguments.emplace_back(loctx->GetOutputOp(output));
  }
  // 调用 LowerBuiltin 函数，传入当前对象指针 this，函数 function 和参数 arguments，并返回其结果
  return LowerBuiltin(this, function, arguments);
// 关闭 lazy 命名空间
} // namespace lazy

// 关闭 torch 命名空间
} // namespace torch

// Non-native ops

// Cast::Lower 方法的实现，将一个操作转换为指定数据类型的操作
torch::lazy::TSOpVector Cast::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  // 创建参数列表并添加操作数的输出操作
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(dtype);
  // 调用 LowerBuiltin 函数以降低内置操作 at::aten::to
  return LowerBuiltin(at::aten::to, function, arguments);
}

// DeviceData::Lower 方法的实现，用于处理设备数据节点
torch::lazy::TSOpVector DeviceData::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  // 获取设备数据信息指针
  auto infoptr = data_->info();
  auto deviceDataInfoPtr =
      (torch::lazy::LazyGraphExecutor::DeviceDataInfo*)infoptr;
  // 如果启用了图形转储，则记录设备数据节点的信息
  if (GRAPH_DUMP_ENABLED) {
    LOG(ERROR) << "Lowering device data node, tensor id "
               << deviceDataInfoPtr->tensor_id << std::endl;
  }
  // 返回数据节点的参数操作
  return {loctx->GetParameter(data_)};
}

// Expand::Lower 方法的实现，用于处理扩展操作
torch::lazy::TSOpVector Expand::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  // 创建参数列表并添加操作数的输出操作和扩展大小
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(size);
  // 调用 LowerBuiltin 函数以降低内置操作，并获取扩展后的输出
  auto expand_out = LowerBuiltin(this, function, arguments);
  // 如果是标量扩展操作，则进行额外的处理
  if (is_scalar_expand) {
    // 对于 aten::expand 操作，当原张量的秩为 0 时，所有步长都设置为 0。
    // 这导致在检查内部内存重叠时出现误报，因为 at::has_internal_overlap 返回 MemOverlap::YES。
    TORCH_CHECK_EQ(expand_out.size(), 1);
    // 生成扩展操作输出的克隆副本并返回
    return {GenerateClone(expand_out.front(), function)};
  }
  // 返回扩展操作的输出
  return expand_out;
}

// Scalar::Lower 方法的实现，用于处理标量操作
torch::lazy::TSOpVector Scalar::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  // 创建标量的 TensorOptions，并插入常量到图中
  auto options =
      at::TensorOptions()
          .device(torch::lazy::getBackend()->EagerFallbackDeviceType())
          .dtype(shape().scalar_type());
  return {loctx->graph()->insertConstant(at::scalar_tensor(value, options))};
}
```