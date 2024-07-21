# `.\pytorch\torch\csrc\api\include\torch\optim\schedulers\reduce_on_plateau_scheduler.h`

```py
#pragma once

#include <torch/optim/optimizer.h>  // 包含了优化器基类的头文件
#include <torch/optim/schedulers/lr_scheduler.h>  // 包含了学习率调度器的头文件

#include <torch/csrc/Export.h>  // Torch导出相关的头文件

#include <string>  // C++标准字符串库

#include <cmath>  // 数学函数库

#include <iostream>  // 输入输出流库

namespace torch {
namespace optim {

class TORCH_API ReduceLROnPlateauScheduler {  // 定义了一个名为ReduceLROnPlateauScheduler的类
 public:
  enum SchedulerMode { min, max };  // 定义了枚举类型SchedulerMode，表示调度模式为最小值或最大值
  enum ThresholdMode { rel, abs };  // 定义了枚举类型ThresholdMode，表示阈值模式为相对或绝对

  // 构造函数，初始化ReduceLROnPlateauScheduler对象
  ReduceLROnPlateauScheduler(
      Optimizer& optimizer,  // 引用类型参数，指定优化器对象
      SchedulerMode mode = min,  // 调度模式，默认为最小值模式
      float factor = 0.1,  // 学习率缩减系数，默认为0.1
      int patience = 10,  // 忍耐期，默认为10个epoch
      double threshold = 1e-4,  // 阈值，默认为0.0001
      ThresholdMode threshold_mode = rel,  // 阈值模式，默认为相对模式
      int cooldown = 0,  // 冷却期，默认为0个epoch
      const std::vector<float>& min_lr = std::vector<float>(),  // 最小学习率，默认为空向量
      double eps = 1e-8,  // 很小的常数eps，默认为0.00000001
      bool verbose = false);  // 是否输出详细信息，默认为false

  virtual ~ReduceLROnPlateauScheduler() = default;  // 虚析构函数，使用默认实现

  void step(float metric);  // 执行调度器的一步操作，根据指标更新学习率

 private:
  void reset();  // 重置调度器状态的私有方法
  void reduce_lr(int epoch);  // 缩减学习率的私有方法，根据epoch数进行判断
  bool in_cooldown();  // 判断当前是否处于冷却期的私有方法
  bool is_better(float a);  // 比较指标a是否更优的私有方法
  void init_is_better(
      SchedulerMode mode,  // 初始化比较方法的私有方法，根据mode和threshold_mode选择初始化方式
      double threshold,
      ThresholdMode threshold_mode);

  Optimizer& optimizer;  // 引用类型的优化器对象
  SchedulerMode mode;  // 调度模式（最小值或最大值）
  float mode_worse;  // 与调度模式相关的参数
  float factor;  // 学习率缩减系数
  int patience;  // 忍耐期
  double threshold;  // 阈值
  ThresholdMode threshold_mode;  // 阈值模式
  int cooldown;  // 冷却期
  int cooldown_counter;  // 冷却计数器
  std::vector<float> min_lrs;  // 最小学习率的向量
  double eps;  // 非常小的常数eps
  float best;  // 最佳指标值
  bool verbose;  // 是否输出详细信息
  int last_epoch;  // 上一个epoch数
  int num_bad_epochs;  // 不良epoch数
};
} // namespace optim
} // namespace torch
```