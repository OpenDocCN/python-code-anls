# `.\pytorch\torch\csrc\api\src\optim\schedulers\lr_scheduler.cpp`

```
// 包含头文件 <c10/util/irange.h>，提供用于范围迭代的工具
// 包含头文件 <torch/optim/schedulers/lr_scheduler.h>，提供学习率调度器的定义

namespace torch {
namespace optim {

// 定义 LRScheduler 类，接受一个 Optimizer 对象作为参数
LRScheduler::LRScheduler(torch::optim::Optimizer& optimizer)
    : optimizer_(optimizer) {}

// LRScheduler 类的方法，用于执行一次学习率调度步骤
void LRScheduler::step() {
  // 获取当前学习率列表
  std::vector<double> learning_rates = get_lrs();
  // 设置优化器中的学习率
  set_optimizer_lrs(learning_rates);
  // 增加步数计数器
  step_count_++;
}

// 设置优化器中的学习率
void LRScheduler::set_optimizer_lrs(const std::vector<double>& learning_rates) {
  // 检查学习率的数量是否与优化器中参数组的数量相等
  TORCH_CHECK(
      learning_rates.size() == optimizer_.param_groups().size(),
      "Number of learning rates not equal to the number of param groups\n",
      "Number of learning rates given: ",
      learning_rates.size(),
      "\nNumber of param groups: ",
      optimizer_.param_groups().size());

  // 遍历每个参数组，设置对应的学习率
  for (const auto i : c10::irange(optimizer_.param_groups().size())) {
    optimizer_.param_groups()[i].options().set_lr(learning_rates[i]);
  }
}

// 获取当前优化器中各参数组的学习率列表
std::vector<double> LRScheduler::get_current_lrs() const {
  // 创建一个与参数组数量相同的学习率列表
  std::vector<double> learnings_rates(optimizer_.param_groups().size());
  // 如果学习率列表非空，则为每个参数组获取当前学习率
  if (!learnings_rates.empty()) {
    for (const auto i : c10::irange(optimizer_.param_groups().size())) {
      learnings_rates[i] = optimizer_.param_groups()[i].options().get_lr();
    }
  }
  // 返回获取到的学习率列表
  return learnings_rates;
}

} // namespace optim
} // namespace torch
```