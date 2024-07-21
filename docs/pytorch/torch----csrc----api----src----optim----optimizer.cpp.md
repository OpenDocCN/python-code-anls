# `.\pytorch\torch\csrc\api\src\optim\optimizer.cpp`

```
#include <torch/optim/optimizer.h>  // 包含了 torch 库中优化器的头文件

#include <torch/csrc/autograd/generated/variable_factories.h>  // 包含了 torch 自动求导相关的变量工厂的头文件
#include <torch/types.h>  // 包含了 torch 库中的数据类型定义的头文件

#include <string>  // 包含了处理字符串的标准库
#include <utility>  // 包含了实用程序组件的标准库
#include <vector>  // 包含了处理向量（数组）的标准库

namespace torch {
namespace optim {

bool OptimizerParamGroup::has_options() const {
  return options_ != nullptr;  // 返回 options_ 指针是否为空的布尔值
}

OptimizerOptions& OptimizerParamGroup::options() {
  TORCH_CHECK(has_options());  // 断言 options_ 不为空，否则抛出异常
  return *options_.get();  // 返回 options_ 所指向的 OptimizerOptions 对象的引用
}

const OptimizerOptions& OptimizerParamGroup::options() const {
  TORCH_CHECK(has_options());  // 断言 options_ 不为空，否则抛出异常
  return *options_.get();  // 返回 options_ 所指向的 OptimizerOptions 对象的常量引用
}

void OptimizerParamGroup::set_options(
    std::unique_ptr<OptimizerOptions> options) {
  options_ = std::move(options);  // 移动语义赋值给 options_
}

std::vector<Tensor>& OptimizerParamGroup::params() {
  return params_;  // 返回 params_ 向量的引用
}

const std::vector<Tensor>& OptimizerParamGroup::params() const {
  return params_;  // 返回 params_ 向量的常量引用
}

std::unique_ptr<OptimizerParamState> OptimizerParamState::clone() const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for torch::optim::OptimizerParamState. ",
      "Subclass torch::optim::OptimizerCloneableParamState<YourOptimizerParamState> ",
      "instead of torch::optim::OptimizerParamState to inherit the ability to clone.");
  // 断言抛出异常，表明 clone() 方法尚未在 OptimizerParamState 的子类中实现，提供建议信息
}

void OptimizerParamState::serialize(torch::serialize::InputArchive& archive) {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
  // 断言抛出异常，表明 serialize() 方法尚未在 OptimizerParamState 的子类中实现，提供建议信息
}

void OptimizerParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
  // 断言抛出异常，表明 serialize() 方法尚未在 OptimizerParamState 的子类中实现，提供建议信息
}

double OptimizerOptions::get_lr() const {
  TORCH_CHECK(
      false,
      "double get_lr() has not been overridden and implemented in subclass of torch::optim::OptimizerOptions, you must override it in your subclass.");
  // 断言抛出异常，表明 get_lr() 方法尚未在 OptimizerOptions 的子类中实现，提供建议信息
}

void OptimizerOptions::set_lr(const double lr) {
  TORCH_CHECK(
      false,
      "double set_lr() has not been overridden and implemented in subclass of torch::optim::OptimizerOptions, you must override it in your subclass.");
  // 断言抛出异常，表明 set_lr() 方法尚未在 OptimizerOptions 的子类中实现，提供建议信息
}

std::unique_ptr<OptimizerOptions> OptimizerOptions::clone() const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for torch::optim::OptimizerOptions. ",
      "Subclass torch::optim::OptimizerCloneableOptions<YourOptimizerOptions> ",
      "instead of torch::optim::OptimizerOptions to inherit the ability to clone.");
  // 断言抛出异常，表明 clone() 方法尚未在 OptimizerOptions 中实现，提供建议信息
}

} // namespace optim
} // namespace torch
// 实现了 OptimizerOptions 类的序列化方法，用于输入归档
void OptimizerOptions::serialize(torch::serialize::InputArchive& archive) {
  // 使用 TORCH_CHECK 断言，如果执行到这里，输出错误信息
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

// 实现了 OptimizerOptions 类的序列化方法，用于输出归档
void OptimizerOptions::serialize(
    torch::serialize::OutputArchive& archive) const {
  // 使用 TORCH_CHECK 断言，如果执行到这里，输出错误信息
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

// 将参数组添加到优化器中
void Optimizer::add_param_group(const OptimizerParamGroup& param_group) {
  // 遍历参数组中的每个参数
  for (const auto& param : param_group.params()) {
    // 使用 TORCH_CHECK 断言，检查参数是否为叶子节点张量
    TORCH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
  }
  // 内部断言，确保 defaults_ 不为空指针
  TORCH_INTERNAL_ASSERT(defaults_ != nullptr);
  // 创建参数组的副本 param_group_
  OptimizerParamGroup param_group_(param_group.params());
  // 如果 param_group 没有指定选项，则设置为 defaults_ 的克隆
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    // 否则，设置为 param_group 的选项的克隆
    param_group_.set_options(param_group.options().clone());
  }
  // 检查参数组中的每个参数，确保没有重复出现的参数
  for (const auto& p : param_group_.params()) {
    TORCH_CHECK(
        state_.count(p.unsafeGetTensorImpl()) == 0,
        "some parameters appear in more than one parameter group");
  }
  // 将 param_group_ 移动到 param_groups_ 的末尾
  param_groups_.emplace_back(std::move(param_group_));
}

// 向优化器中添加参数
void Optimizer::add_parameters(const std::vector<Tensor>& parameters) {
  // 发出警告，提示该函数将在 PyTorch 1.6 版本中移除
  TORCH_WARN("Optimizer::add_parameters() will be removed in PyTorch 1.6");
  // 获取第一个参数组的参数引用
  auto& parameters_ = param_groups_[0].params();
  // 将新参数插入到 parameters_ 的末尾
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

// 将所有参数的梯度置零
void Optimizer::zero_grad(bool set_to_none) {
  // 遍历所有参数组
  for (auto& group : param_groups_) {
    // 遍历每个参数组中的参数
    for (auto& p : group.params()) {
      // 如果参数的梯度已定义
      if (p.mutable_grad().defined()) {
        // 分离参数的梯度
        p.mutable_grad().detach_();
        // 根据 set_to_none 的值，重置或清空参数的梯度
        if (set_to_none)
          p.mutable_grad().reset();
        else
          p.mutable_grad().zero_();
      }
    }
  }
}

// 返回第一个参数组的参数的常量引用
const std::vector<Tensor>& Optimizer::parameters() const noexcept {
  // 发出警告，提示该函数将在 PyTorch 1.6 版本中移除
  TORCH_WARN("Optimizer::parameters() will be removed in PyTorch 1.6");
  // 返回第一个参数组的参数的常量引用
  return param_groups_.at(0).params();
}

// 返回第一个参数组的参数的可修改引用
std::vector<Tensor>& Optimizer::parameters() noexcept {
  // 发出警告，提示该函数将在 PyTorch 1.6 版本中移除
  TORCH_WARN("Optimizer::parameters() will be removed in PyTorch 1.6");
  // 返回第一个参数组的参数的可修改引用
  return param_groups_.at(0).params();
}

// 返回所有参数组中参数的总数
size_t Optimizer::size() const noexcept {
  // 发出警告，提示该函数将在 PyTorch 1.6 版本中移除
  TORCH_WARN("Optimizer::size() will be removed in PyTorch 1.6");
  // 计算所有参数组中参数的总数
  size_t count = 0;
  for (const auto& group : param_groups_) {
    count += group.params().size();
  }
  return count;
}

// 返回优化器的默认选项的引用
OptimizerOptions& Optimizer::defaults() noexcept {
  return *defaults_.get();
}

// 返回优化器的默认选项的常量引用
const OptimizerOptions& Optimizer::defaults() const noexcept {
  return *defaults_.get();
}

// 返回参数组的引用
std::vector<OptimizerParamGroup>& Optimizer::param_groups() noexcept {
  return param_groups_;
}

// 返回参数组的常量引用
const std::vector<OptimizerParamGroup>& Optimizer::param_groups()
    const noexcept {
  return param_groups_;
}
// 返回 Optimizer 对象的状态映射，包含指向 OptimizerParamState 对象的唯一指针
ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& Optimizer::state() noexcept {
  return state_;
}

// 返回 Optimizer 对象的状态映射，包含指向 OptimizerParamState 对象的唯一指针，且为常量成员函数
const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& Optimizer::state() const noexcept {
  return state_;
}

// 保存 Optimizer 对象的状态到序列化输出存档中
void Optimizer::save(serialize::OutputArchive& archive) const {}

// 从序列化输入存档中加载 Optimizer 对象的状态
void Optimizer::load(serialize::InputArchive& archive) {}

/// 将 Optimizer 对象序列化到 OutputArchive 中
serialize::OutputArchive& operator<<(serialize::OutputArchive& archive, const Optimizer& optimizer) {
  optimizer.save(archive);
  return archive;
}

/// 从 InputArchive 中反序列化一个 Tensor 对象
serialize::InputArchive& operator>>(serialize::InputArchive& archive, Optimizer& optimizer) {
  optimizer.load(archive);
  return archive;
}

} // namespace optim
} // namespace torch
```