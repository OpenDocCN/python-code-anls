# `.\pytorch\torch\csrc\api\src\optim\schedulers\step_lr.cpp`

```py
#include <torch/optim/schedulers/step_lr.h>  // 包含 Torch 库中的 StepLR 调度器头文件

namespace torch {
namespace optim {

StepLR::StepLR(
    torch::optim::Optimizer& optimizer,    // StepLR 类的构造函数，接受一个优化器对象和两个参数
    const unsigned step_size,              // 步长大小，用于调整学习率的频率
    const double gamma)                    // 学习率缩放因子
    : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {}  // 调用基类 LRScheduler 的构造函数，并初始化成员变量

std::vector<double> StepLR::get_lrs() {    // 获取当前学习率的函数定义
  if (step_count_ == 0 || step_count_ % step_size_ != 0)  // 如果步数为0或者不是步长的倍数，则返回当前学习率
    return get_current_lrs();
  else {                                    // 否则，计算新的学习率列表
    std::vector<double> lrs = get_current_lrs();  // 获取当前学习率列表
    std::transform(                         // 使用 lambda 表达式对学习率列表进行缩放
        lrs.begin(), lrs.end(), lrs.begin(), [this](const double& v) {
          return this->gamma_ * v;          // 根据 gamma 缩放每个学习率
        });
    return lrs;                             // 返回缩放后的学习率列表
  }
}

} // namespace optim
} // namespace torch
```