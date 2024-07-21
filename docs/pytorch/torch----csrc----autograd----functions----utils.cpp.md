# `.\pytorch\torch\csrc\autograd\functions\utils.cpp`

```
// 定义了一个命名空间 torch::autograd，用于包含自动求导相关的函数和变量
namespace torch {
namespace autograd {

// wrap_outputs 函数用于封装输出变量列表，并根据情况添加梯度信息
variable_list wrap_outputs(
    const variable_list& inputs,      // 输入变量列表
    tensor_list&& outputs,            // 输出张量列表（移动语义）
    const function_constructor& ctr) { // 函数构造器
  variable_list result;               // 结果变量列表
  result.reserve(outputs.size());     // 预先分配结果变量列表的空间
  // 如果所有输入变量都不需要梯度
  if (!any_variable_requires_grad(inputs)) {
    // 遍历输出张量列表
    for (auto& output : outputs) {
      // 如果输出张量已定义
      if (output.defined()) {
        // 将输出张量包装成不需要梯度的变量，并加入结果列表
        result.push_back(make_variable(output, /*requires_grad=*/false));
      } else {
        // 否则，添加一个未定义的变量到结果列表
        result.emplace_back();
      }
    }
  } else {
    // 否则，至少有一个输入变量需要梯度
    // 根据当前梯度模式是否启用，收集下一步边缘信息
    auto grad_fn = ctr(GradMode::is_enabled() ? collect_next_edges(inputs) : edge_list());
    // 遍历输出张量列表
    for (auto& output : outputs) {
      // 如果输出张量已定义
      if (output.defined()) {
        // 创建一个不需要梯度的变量，并与梯度函数关联
        auto variable = autograd::make_variable(output, /*requires_grad=*/false);
        autograd::create_gradient_edge(variable, grad_fn);
        // 将变量移动到结果列表
        result.push_back(std::move(variable));
      } else {
        // 否则，添加一个未定义输入元数据到梯度函数，并添加未定义变量到结果列表
        grad_fn->add_input_metadata(Node::undefined_input());
        result.emplace_back();
      }
    }
  }
  // 返回封装后的结果变量列表
  return result;
}

// check_input_variables 函数用于检查输入变量的合法性
void check_input_variables(
    const char* name,                 // 函数名
    const variable_list& inputs,      // 输入变量列表
    int args,                         // 预期的参数个数
    int required_args,                // 实际需要的参数个数
    bool allow_undefined) {           // 是否允许输入变量未定义
  // 如果实际需要的参数个数为 -1，则设置为预期参数个数
  if (required_args == -1) {
    required_args = args;
  }
  // 如果输入变量个数与预期参数个数不符
  if (inputs.size() != (size_t)args) {
    // 抛出运行时错误，指示参数个数不匹配
    std::stringstream ss;
    ss << name << ": expected " << args << " arguments (got " << inputs.size();
    ss << ")";
    throw std::runtime_error(ss.str());
  }
  // 遍历每一个预期的参数
  for (const auto i : c10::irange(required_args)) {
    // 如果输入变量未定义，并且不允许输入变量未定义
    if (!inputs[i].defined() && !allow_undefined) {
      // 抛出运行时错误，指示期望的 Tensor 参数未提供
      std::stringstream ss;
      ss << name << ": expected Tensor at argument " << i << " (got None)";
      throw std::runtime_error(ss.str());
    }
  }
}

} // namespace autograd
} // namespace torch
```