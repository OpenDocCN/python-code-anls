# `.\pytorch\torch\csrc\api\include\torch\optim\schedulers\step_lr.h`

```py
#pragma once

# 预处理指令：指示编译器只包含此头文件一次，避免多重包含。


#include <torch/optim/schedulers/lr_scheduler.h>

# 包含 Torch 库中的学习率调度器头文件。


namespace torch {
namespace optim {

# 进入命名空间 torch::optim，用于组织和限定类、函数、变量等的作用域。


class TORCH_API StepLR : public LRScheduler {
 public:
  StepLR(
      torch::optim::Optimizer& optimizer,
      const unsigned step_size,
      const double gamma = 0.1);

# 定义 StepLR 类，继承自 LRScheduler 类。构造函数 StepLR 接受优化器 optimizer、步长 step_size 和可选的衰减因子 gamma。


 private:
  std::vector<double> get_lrs() override;

# 声明私有成员函数 get_lrs()，用于计算并返回学习率的向量。


  const unsigned step_size_;
  const double gamma_;
};

# 声明 StepLR 类的私有成员变量：step_size_（步长）和 gamma_（衰减因子）。


} // namespace optim
} // namespace torch

# 结束命名空间 torch::optim 的定义。
```