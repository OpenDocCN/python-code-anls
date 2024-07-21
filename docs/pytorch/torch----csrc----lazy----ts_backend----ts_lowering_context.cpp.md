# `.\pytorch\torch\csrc\lazy\ts_backend\ts_lowering_context.cpp`

```
namespace torch {
namespace lazy {

// TSLoweringContext 类的构造函数，初始化一个 LoweringContext 实例，
// 创建一个新的 torch::jit::Graph，并用它初始化 function_
TSLoweringContext::TSLoweringContext(
    const std::string& name,
    BackendDevice device)
    : torch::lazy::LoweringContext(name, device),
      graph_(std::make_shared<torch::jit::Graph>()),
      function_(
          std::make_shared<torch::jit::GraphFunction>(name, graph_, nullptr)) {}

// TSLoweringContext 类的构造函数，初始化一个 LoweringContext 实例，
// 创建一个新的 torch::jit::Graph，并用它初始化 function_
// 这个构造函数还接受额外的参数 post_order 和 emit_status，对 post_order 中的每个节点进行 Lower 操作
TSLoweringContext::TSLoweringContext(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<const Node*> post_order,
    Util::EmissionMap emit_status)
    : torch::lazy::LoweringContext(name, device, post_order, emit_status),
      graph_(std::make_shared<torch::jit::Graph>()),
      function_(
          std::make_shared<torch::jit::GraphFunction>(name, graph_, nullptr)) {
  for (auto node : post_order) {
    Lower(node); // 对 post_order 中的每个节点进行 Lower 操作
  }
}

// 对给定的节点进行下降（Lower）操作，将其转换为 Torch 的运算图表示
void TSLoweringContext::Lower(const Node* node) {
  if (auto* tsnode = dynamic_cast<const torch::lazy::TsNode*>(node)) {
    // 如果节点是 TsNode 类型，则调用其 Lower 方法生成对应的 Torch 运算操作符
    TSOpVector ops = tsnode->Lower(function_, this);
    TORCH_CHECK(!ops.empty(), "Failed to lower: ", *node);
    TORCH_CHECK_EQ(node->num_outputs(), ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      AssignOutputOp(torch::lazy::Output(node, i), ops[i]); // 将运算结果分配给输出
    }
  } else {
    throw std::runtime_error(
        "Expected torch::lazy::TsNode but could not dynamic cast");
  }
}

// 将输出操作符 op 分配给指定的输出 output，并记录 Python 的堆栈跟踪信息
void TSLoweringContext::AssignOutputOp(
    const Output& output,
    torch::jit::Value* op) {
  const TsNode* ts_node = static_cast<const TsNode*>(output.node);
  std::string stack_trace = ts_node->getPythonStacktrace(); // 获取 Python 堆栈跟踪信息
  if (!stack_trace.empty()) {
    op->node()->s_(c10::Symbol::attr("source"), stack_trace); // 将堆栈跟踪信息作为 source 属性添加到操作符上
  }
  emitted_outputs_[output] = op; // 记录输出操作符
}

// 根据 BackendDataPtr 获取对应的参数，如果参数尚不存在则创建一个新的输入参数
torch::jit::Value* TSLoweringContext::GetParameter(BackendDataPtr data) {
  const auto ts_data = std::static_pointer_cast<TSData>(data);
  BackendData::Handle handle = ts_data->GetHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    torch::jit::Value* param =
        graph_->addInput(c10::str("p", parameters_.size())); // 创建新的输入参数
    if (ts_data->scalar.has_value()) {
      auto scalarType = ts_data->scalar.value().type();
      if (isFloatingType(scalarType)) {
        param->setType(c10::FloatType::get()); // 设置参数类型为浮点类型
      } else if (isIntegralType(scalarType, /*includeBool=*/true)) {
        param->setType(c10::IntType::get()); // 设置参数类型为整数类型
      } else {
        TORCH_CHECK(
            false, "Unhandled scalar type: ", c10::toString(scalarType)); // 处理未知的标量类型错误
      }
    }
    // 将新创建的参数及其索引添加到 parameters_map_ 和 parameters_ 中
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(ts_data);
  }
  parameter_sequence_.push_back(it->second.index); // 记录参数的索引顺序
  return it->second.param; // 返回参数对应的 Torch Value
}

} // namespace lazy
} // namespace torch
} // 结束命名空间 torch
```