# `.\pytorch\torch\csrc\autograd\cpp_hook.cpp`

```py
// 包含头文件：包括使用到的 C++ 库的头文件
#include <c10/util/irange.h>
#include <torch/csrc/autograd/cpp_hook.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/variable.h>

// 匿名命名空间：定义了一个局部作用域，内部的符号对外部不可见
namespace {

// 使用声明：引入 torch::autograd::Variable 作为变量类型
using torch::autograd::Variable;

// 检查单一结果函数定义：检查梯度值是否为空，若为空则抛出异常
void check_single_result(
    const at::TensorBase& value,
    const at::TensorBase& result,
    const std::string& hook_name) {
  if (!value.defined()) {
    throw std::runtime_error(
        "can't replace a empty gradient with a non-empty value");
  }
  torch::autograd::check_variable_result(value, result, hook_name);
}

} // namespace

// torch::autograd 命名空间：定义了一些与自动求导相关的函数和类
namespace torch {
namespace autograd {

// CppFunctionTensorPreHook 类定义：构造函数初始化钩子和索引
CppFunctionTensorPreHook::CppFunctionTensorPreHook(
    std::shared_ptr<hooks_list> hooks,
    size_t value_idx)
    : hooks_(std::move(hooks)), value_idx_(value_idx) {}

// 操作符重载：处理张量前钩子的操作符重载，修改输入张量并返回结果列表
variable_list CppFunctionTensorPreHook::operator()(
    const variable_list& values) {
  auto value = values[value_idx_];  // 获取指定索引的输入张量
  for (const auto i : c10::irange(hooks_->size())) {
    auto& hook = (*hooks_)[i];  // 获取当前钩子
    if (!hook) {
      // 钩子已移除
      continue;
    }
    auto res = hook(value);  // 应用钩子函数
    if (!res.defined()) {
      // 不改变梯度
      continue;
    }
    check_single_result(value, res, std::to_string(i));  // 检查单一结果的有效性
    value = std::move(res);  // 移动结果到输入张量
  }
  variable_list results(values);  // 创建结果列表并复制输入张量列表
  results[value_idx_] = value;  // 更新结果列表中的指定索引位置
  return results;  // 返回更新后的结果列表
}

// CppFunctionSingleTensorPreHook 类定义：构造函数初始化钩子和索引
CppFunctionSingleTensorPreHook::CppFunctionSingleTensorPreHook(
    std::function<at::TensorBase(const at::TensorBase&)> hook,
    size_t value_idx)
    : hook_(std::move(hook)), value_idx_(value_idx) {}

// 操作符重载：处理单个张量前钩子的操作符重载，不支持返回结果的钩子
variable_list CppFunctionSingleTensorPreHook::operator()(
    const variable_list& values) {
  const auto& value = values[value_idx_];  // 获取指定索引的输入张量
  auto res = hook_(value);  // 应用钩子函数
  TORCH_INTERNAL_ASSERT(
      !res.defined(),
      "CppFunctionSingleTensorPreHook currently only supports hooks that don't return");
  variable_list results(values);  // 创建结果列表并复制输入张量列表
  return results;  // 返回结果列表
}

} // namespace autograd
} // namespace torch
```