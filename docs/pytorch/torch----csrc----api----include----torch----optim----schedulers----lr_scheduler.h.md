# `.\pytorch\torch\csrc\api\include\torch\optim\schedulers\lr_scheduler.h`

```
#pragma once

#include <torch/optim/optimizer.h>

#include <torch/csrc/Export.h>

namespace torch {
namespace optim {

class TORCH_API LRScheduler {
 public:
  // This class needs to take a reference of an optimizer from outside such that
  // it can modify its learning rates; due to this the lifetime of said
  // optimizer must be maintained
  // 构造函数，接受一个外部优化器的引用，以便于修改其学习率；因此，外部优化器的生命周期必须得到维护
  LRScheduler(torch::optim::Optimizer& optimizer);

  virtual ~LRScheduler() = default;

  // 执行一步学习率调度的操作
  void step();

 protected:
  // 从特定子类计算并返回一组学习率的向量。向量的每个元素代表一个参数组的学习率，
  // 尽管正常使用情况下这些元素可能是相同的。
  virtual std::vector<double> get_lrs() = 0;

  // 从优化器中获取当前的学习率
  std::vector<double> get_current_lrs() const;

  unsigned step_count_{};

 private:
  // 设置优化器的学习率
  void set_optimizer_lrs(const std::vector<double>& learning_rates);

  // 引用外部优化器对象
  torch::optim::Optimizer& optimizer_;
};
} // namespace optim
} // namespace torch
```